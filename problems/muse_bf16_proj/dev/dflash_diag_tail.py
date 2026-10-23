# Diagnostic-only prepare extension. No diagnostic state reaches kernel_fn.
_df_scored_prepare = prepare


def _df_diag_equal(a, b):
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_df_diag_equal(x,y) for x,y in zip(a,b))
    return torch.equal(a,b)


def _df_diag_pair(stock, fast, mutate, flush, deadline):
    import statistics,time
    with _df_proj._Graph(stock) as a, _df_proj._Graph(fast) as b:
        old,new=[],[]
        for rep in range(20):
            if time.monotonic() >= deadline:
                raise TimeoutError('20 complete pairs would exceed diagnostic budget')
            mutate()
            for index in ((0,1) if rep%2==0 else (1,0)):
                flush.add_(1)
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();(a if index==0 else b).graph.replay();end.record();end.synchronize()
                (old if index==0 else new).append(begin.elapsed_time(end)*1000)
            if not _df_diag_equal(a.out,b.out):
                raise ValueError('fresh-input result differs from deterministic stock; rejected')
        return statistics.median(old),statistics.median(new)


def _df_run_diagnostics(deadline, lines):
    import time
    device=torch.device('cuda')
    flush=torch.zeros(128<<20,device=device,dtype=torch.uint8)
    # Local TP8 shapes. Context-KV is already a stacked NN GEMM in stock.
    specs=(('conv_coeff',2048,8192,8,10,False),('gate_up',7168,8192,8,5,False),
           ('down',8192,3584,8,5,False),('context_fc',1024,40960,8,1,False),
           ('qkv',1280,8192,8,5,False),('o',8192,1024,8,5,False),
           ('context_kv',1280,8192,8,1,True),
           ('draft_head',25256,8192,7,1,False))
    for name,n,k,mult,count,nn in specs:
        if time.monotonic()+10>=deadline:
            lines.append(f'DFB_SKIP {name} budget');continue
        try:
            canonical=torch.randn((n,k),device=device,dtype=torch.bfloat16)/k**.5
            w=canonical.T.contiguous() if nn else canonical
            ms=tuple(c*mult for c in (1,4,16,32))
            op=DFlashLinear(w,tokens=ms,budget_s=min(8,max(0,deadline-time.monotonic()-10)),stock_nn=nn)
            for c,m in zip((1,4,16,32),ms):
                if time.monotonic()+2>=deadline:
                    lines.append(f'DFB_SKIP {name} c={c} budget');continue
                try:
                    x=torch.randn((m,k),device=device,dtype=torch.bfloat16)
                    old,new=_df_diag_pair(lambda:op._df_stock(x),lambda:op(x),
                                          lambda:x.copy_(torch.randn_like(x)),flush,deadline)
                    chosen=op.plans.get(m,('stock',))
                    credit=0.0 if chosen==('stock',) else count*(old-new)/1000
                    rejects=sum(r[0]==m for r in op.rejections)
                    lines.append(f'DFB linear {name} c={c} m={m} n={n} k={k} old_us={old:.3f} new_us={new:.3f} pairs=20 choice={chosen} rejects={rejects} save_ms={credit:.6f}')
                except Exception as exc:
                    lines.append(f'DFB_REJECT {name} c={c}: {str(exc).splitlines()[0][:100]}')
            del op,w,canonical
        except Exception as exc:
            lines.append(f'DFB_REJECT {name}: {str(exc).splitlines()[0][:100]}')
    for c in (1,4,16,32):
        for label in ('unary','selector_hidden','conv','selector'):
            if time.monotonic()+10>=deadline:
                lines.append(f'DFB_SKIP {label} c={c} budget');continue
            try:
                if label=='unary':
                    # TP8 candidate_topk gathers FP32 values: float() is a view,
                    # not a fifth cast kernel. Keep independent mutable inputs.
                    x=torch.randn((c*7,16),device=device,dtype=torch.float32)
                    fast_x=x.clone()
                    stock=lambda:x.float().mul_(.1767766952966369).div_(20).tanh_().mul_(20)
                    fast=lambda:dflash_unary(fast_x)
                    def mutate():
                        fresh=torch.randn_like(x)*20;x.copy_(fresh);fast_x.copy_(fresh)
                    count=1
                elif label=='selector_hidden':
                    x=torch.randn((c,8,8192),device=device,dtype=torch.bfloat16)
                    view=x[:,1:]
                    w=torch.randn((256,8192),device=device,dtype=torch.bfloat16)/8192**.5
                    # Stock is a strided bmm at C>1. Keep its arithmetic; a
                    # flattened mm is not an exact replacement.
                    stock=lambda:F.linear(view,w)
                    fast=stock
                    mutate=lambda:x.copy_(torch.randn_like(x))
                    count=0
                elif label=='conv':
                    x=torch.randn((c*8,8192),device=device,dtype=torch.bfloat16)
                    full=torch.randn((c*8,2,2,512),device=device,dtype=torch.bfloat16)
                    delta=full[:,1];w=torch.randn((2,8192),device=device,dtype=torch.bfloat16)
                    # The actual model marks these dimensions static before tracing.
                    torch._dynamo.mark_static(x,1);torch._dynamo.mark_static(delta,1);torch._dynamo.mark_static(delta,2)
                    stock=lambda:_df_stock_grouped_conv(x,delta,w,8,512,16,2)
                    fast=lambda:dflash_conv2(x,delta,w)
                    mutate=lambda:(x.copy_(torch.randn_like(x)),full.copy_(torch.randn_like(full)))
                    count=20
                else:
                    v=202048
                    a=torch.randn((v,256),device=device,dtype=torch.bfloat16)
                    b=torch.randn_like(a);ids=torch.randint(v,(c,7,16),device=device)
                    h=torch.randn((c,7,256),device=device,dtype=torch.bfloat16)
                    unary=torch.randn((c,7,16),device=device);anchor=torch.randint(v,(c,),device=device)
                    for tensor in (ids,unary,h):
                        torch._dynamo.mark_static(tensor,1);torch._dynamo.mark_static(tensor,2)
                    stock=lambda:_df_stock_score_edges(predecessor_table=a,successor_table=b,candidate_ids=ids,
                                                       unary_logits=unary,hidden=h,anchor_token_ids=anchor,top_k=16)
                    fast=lambda:dflash_selector_scores(a,b,ids,unary,h,anchor)
                    mutate=lambda:(ids.random_(0,v),h.copy_(torch.randn_like(h)),unary.normal_())
                    count=1
                old,new=_df_diag_pair(stock,fast,mutate,flush,deadline)
                lines.append(f'DFB glue {label} c={c} old_us={old:.3f} new_us={new:.3f} pairs=20 count={count} save_ms={count*(old-new)/1000:.6f}')
            except Exception as exc:
                lines.append(f'DFB_REJECT {label} c={c}: {str(exc).splitlines()[0][:100]}')


def prepare(weights):
    import time
    started=time.monotonic()
    state=_df_scored_prepare(weights)
    lines=[]
    try:
        _df_run_diagnostics(started+225,lines)
    except Exception as exc:
        lines.append(f'DFB_REJECT diagnostic: {str(exc).splitlines()[0][:120]}')
    # Keep every compact result in the final 16K receipt tail, after compilation,
    # trial rejections and the scored dispatch's verbose prepare output.
    print('DFB_BEGIN raw graph CUDA-event medians; cold128MiB; 20 order-swapped pairs; exact-stock guard',flush=True)
    for line in lines:print(line,flush=True)
    print(f'DFB_END prepare_s={time.monotonic()-started:.3f} base=f3786341611f4f1f module_sha={_DF_MODULE_SHA}',flush=True)
    return state
