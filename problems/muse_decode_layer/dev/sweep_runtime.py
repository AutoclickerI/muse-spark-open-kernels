# Diagnostic only: synthetic weight-shaped work in prepare, as in cand_sweep2.
# No workload input, harness state, or computed output is retained by this sweep.
_prepare_before_sweep = prepare


def _sweep_graph(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out = fn()
    torch.cuda.synchronize()
    return graph, out


class _StageTimer:
    def __init__(self, device):
        import statistics
        self.flush = torch.empty(1 << 28, dtype=torch.uint8, device=device)
        self.a = torch.cuda.Event(enable_timing=True)
        self.b = torch.cuda.Event(enable_timing=True)
        x = torch.zeros(1, device=device)
        g, _ = _sweep_graph(lambda: x.add_(1))
        self.fixed = statistics.median(self.raw(g) for _ in range(10))

    def raw(self, graph):
        self.flush.sum()
        self.a.record(); graph.replay(); self.b.record(); self.b.synchronize()
        return self.a.elapsed_time(self.b) * 1000

    def run(self, fn):
        import statistics
        graph, output = _sweep_graph(fn)
        value = statistics.median(self.raw(graph) for _ in range(8)) - self.fixed
        return max(value, 0.001)


def _sweep_data(batch, q_len, window, device, ctx=80000):
    pages = ctx + q_len
    table = torch.randperm(batch * pages, device=device, dtype=torch.int32).reshape(batch,pages)
    positions = (ctx + torch.arange(q_len,device=device)).repeat(batch).long()
    locations = table[:,ctx:ctx+q_len].reshape(-1).long()
    # Synthetic FP8 KV, retained only for this prepare diagnostic.
    cache_shape = (batch*pages,1,2,64)
    k = torch.randn(cache_shape,device=device,dtype=torch.bfloat16).to(FP8)
    v = torch.randn(cache_shape,device=device,dtype=torch.bfloat16).to(FP8)
    return {'config':{'batch':batch,'q_len':q_len,'tokens':batch*q_len,'page_size':1,
      'window':window,'rope_theta':500000. if window else 0.,'sm_scale':.125,
      'qk_scale_factor':1.,'kv_scale':1.}, 'positions':positions,'cache_loc':locations,
      'page_table':table,'seq_lens':torch.full((batch,),ctx,device=device,dtype=torch.int32),
      'cache_seqlens':torch.full((batch,),ctx+q_len,device=device,dtype=torch.int32),
      'cu_seqlens_q':torch.arange(0,batch*q_len+1,q_len,device=device,dtype=torch.int32),
      'scheduler_metadata':None,'k_cache':k,'v_cache':v,
      'k_scale':torch.ones((),device=device),'v_scale':torch.ones((),device=device),
      'local_start':torch.zeros(1,device=device,dtype=torch.int32),
      'stream':torch.randn(batch*q_len,8192,device=device)}


def _stage_sweep(state):
    global PDL, PERSIST_PREFETCH
    dev = state['o'].device
    timer = _StageTimer(dev)
    print(f'LAYER_SWEEP device={torch.cuda.get_device_name(dev)} fixed_us={timer.fixed:.3f} synthetic_only=1',flush=True)
    original_pdl,original_prefetch=PDL,PERSIST_PREFETCH
    cfg0=GEMV_CFG.copy()
    bank4={**state['moe_bank'],'mod':diag4__module(),'cnt':torch.zeros(64,device=dev,dtype=torch.int32)}
    try:
        for t in (1,8,16,32,128,256):
            data={'local_start':torch.zeros(1,device=dev,dtype=torch.int32)}
            stream=torch.randn(t,8192,device=dev)
            attn=torch.randn(t,1024,device=dev,dtype=torch.bfloat16)
            for weight,k in (('qkv_gate',8192),('o',1024),('pool_router',8192),('unpooler_slice' if t<=8 else 'unpooler',4096)):
                x=torch.randn(t,k,device=dev,dtype=torch.bfloat16)
                w=state[weight];n=w.shape[0]
                opts={}
                for variant in ('default','narrow','cublas'):
                    GEMV_CFG.clear();GEMV_CFG.update(cfg0)
                    if variant=='narrow' and t<=GEMV_MAX_T:
                        bn,ks,bk=GEMV_CFG[(n,k)];GEMV_CFG[(n,k)]=(32,ks,bk)
                    fn=(lambda:torch.mm(x,w.t(),out_dtype=torch.float32)) if variant=='cublas' else (lambda:gemv(x,w))
                    opts[variant]=round(timer.run(fn),3)
                print(f'LAYER_SWEEP projection T={t} stage={weight} us={opts}',flush=True)
            GEMV_CFG.clear();GEMV_CFG.update(cfg0)
            if t<=32:
                opts={}
                for pdl,pref in ((False,False),(True,False),(True,True)):
                    PDL,PERSIST_PREFETCH=pdl,pref
                    def prefix():
                        counts=torch.zeros(2+5*t,device=dev,dtype=torch.int32)
                        return _island(attn,stream,state,data,counts,True)
                    opts[f'pdl{int(pdl)}_pref{int(pref)}']=round(timer.run(prefix),3)
                def prefix_control(epilogue):
                    counts=torch.zeros(15*t,device=dev,dtype=torch.int32)
                    si=counts[:3*t].view(3,t)
                    sf=counts[3*t:].view(torch.float32).view(3,t,4)
                    if epilogue:
                        pr,s1=o_and_pool(attn,stream,state,si[0])
                    else:
                        a=gemv(attn,state['o'])
                        h,s1=boundary(a,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],(sf[0],si[0]))
                        pr=gemv(h,state['pool_router'],out_dtype=torch.float32)
                    x,ids,tw,lids=route_and_norm(pr,state,data,(sf[1],si[1]))
                    return x,ids,tw,lids,s1
                opts['unfused']=round(timer.run(lambda:prefix_control(False)),3)
                opts['epilogue']=round(timer.run(lambda:prefix_control(True)),3)
                print(f'LAYER_SWEEP prefix T={t} us={opts}',flush=True)
                x,ids,tw,lids,s1=prefix()
            else:
                x=torch.randn(t,4096,device=dev,dtype=torch.bfloat16)
                # A fixed synthetic route pattern, used only to probe expert scheduling.
                lids=(torch.arange(t*8,device=dev,dtype=torch.int32)%64-16).reshape(t,8)
                tw=torch.full((t,8),.125,device=dev)
            config={'local_count':32,'d_hidden':4096,'d_expert':4096,'local_start':0}
            md={'hidden_states':x,'topk_ids':lids,'topk_weights':tw,'config':config}
            opts={}
            for variant,fn in (('v4',lambda:diag4_kernel_fn(md,bank4)),
                               ('v6',lambda:mv4_kernel_fn(md,state['moe_bank']))):
                opts[variant]=round(timer.run(fn),3)
            print(f'LAYER_SWEEP experts T={t} us={opts}',flush=True)
            if t<=32:
                moe=mv4_kernel_fn(md,state['moe_bank'])
                opts={}
                for pdl,pref in ((False,False),(True,False),(True,True)):
                    PDL,PERSIST_PREFETCH=pdl,pref
                    def suffix():
                        counts=torch.zeros(2+5*t,device=dev,dtype=torch.int32)
                        return _island(moe,stream,state,data,counts,False)
                    opts[f'pdl{int(pdl)}_pref{int(pref)}']=round(timer.run(suffix),3)
                def suffix_control():
                    counts=torch.zeros(5*t,device=dev,dtype=torch.int32)
                    sums=counts[t:].view(torch.float32).view(t,4)
                    b=gemv(moe,state['unpooler_slice' if t<=8 else 'unpooler'])
                    return boundary(b,stream,state['alpha2'],state['beta2'],state['next_pre_attn_w'],
                                    (sums,counts[:t]),postw=state['post_ffn_w'])
                opts['unfused']=round(timer.run(suffix_control),3)
                print(f'LAYER_SWEEP suffix T={t} us={opts}',flush=True)
        PDL=original_pdl
        for batch,ql in ((1,1),(1,8),(16,1),(32,1)):
            for window in (2048,0):
                data=_sweep_data(batch,ql,window,dev)
                t=batch*ql
                qkv=(torch.randn(1,t,QKV_SIZE,device=dev,dtype=torch.bfloat16),1)
                opts={}
                for pref in (False,True):
                    PERSIST_PREFETCH=pref
                    def pair():
                        counts=torch.zeros(2+5*t,device=dev,dtype=torch.int32)
                        a=stage_attention(qkv,data,state)
                        return _island(a,data['stream'],state,data,counts,True)
                    opts[f'pref{int(pref)}']=round(timer.run(pair),3)
                print(f'LAYER_SWEEP attention_prefix B={batch} L={ql} window={window} us={opts}',flush=True)
    finally:
        PDL,PERSIST_PREFETCH=original_pdl,original_prefetch
        GEMV_CFG.clear();GEMV_CFG.update(cfg0)
    print('LAYER_SWEEP complete; timed kernel uses fixed chain6 dispatch',flush=True)


def prepare(weights):
    state=_prepare_before_sweep(weights)
    _stage_sweep(state)
    return state
