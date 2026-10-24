# NT2 is the native V9 body exercised by H200 lab 3f2b8da0f1d244d7.
# NT1 remains the exact CUDA source from best 812f5f36.
_BEST5_NT2_MOD = None
_BEST5_DEADLINE = float('inf')


def _best5_left(reserve=3.):
    import time
    return time.monotonic()+reserve < _BEST5_DEADLINE


def _best5_nt2_module():
    global _BEST5_NT2_MOD
    if _BEST5_NT2_MOD is None:
        import hashlib
        tag=hashlib.sha256(V9_NT2_CUDA_SRC.encode()).hexdigest()[:12]
        _BEST5_NT2_MOD=load_inline(name='muse_best5_v9_nt2_'+tag,cpp_sources=V9_NT2_CPP_SRC,
            cuda_sources=V9_NT2_CUDA_SRC,functions=['max_smem_attr','smem_needed','resident_blocks','run'],
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',
                *(['-gencode=arch=compute_90a,code=sm_90a'] if IS_HOPPER else [])])
        _BEST5_NT2_MOD.smem_optin=_BEST5_NT2_MOD.max_smem_attr()
    return _BEST5_NT2_MOD


def _best5_rms(x):
    x=x.float()
    return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)


def _best5_fixture(weights,m,seed):
    # Public stream distribution and draft sharing; the router itself is loaded
    # anew by each run. No submission inputs or generated weights are inspected.
    import math
    dev=weights['router'].device
    cpu=torch.Generator().manual_seed(5354)
    mean=torch.randn(8192,generator=cpu).to(dev)
    basis=torch.randn((4,8192),generator=cpu).to(dev)
    gen=torch.Generator(device=dev).manual_seed(seed)
    coef=torch.randn((m,4),generator=gen,device=dev)
    length=1 if m==32 else 8
    share=torch.ones((m//length,length,1),device=dev)
    if length>1:share[:,1:]=.6
    share=share.reshape(m,1)
    stream=share*(.7*mean+math.sqrt(.9/4)*(coef@basis))+(1.-.9*share.square()).sqrt()*torch.randn((m,8192),device=dev,generator=gen)
    h=_best5_rms(stream)*(1.+weights['pre_ffn_norm'].float())
    logits=h@weights['router'].float().T*.1767766952966369
    scores=logits.sigmoid();ids=(scores+weights['router_bias'].float()).topk(8,dim=-1).indices
    picked=scores.gather(1,ids);tw=picked/picked.sum(-1,keepdim=True).clamp_min(1e-15)
    local_start=32*int(torch.bincount((ids//32).flatten(),minlength=8).argmax())
    local=(ids-local_start)%256;local=torch.where(local<32,local,torch.full_like(local,-1)).to(torch.int32)
    x=(_best5_rms(h@weights['pooler'].float().T)*(1.+weights['pre_expert_norm'].float())).bfloat16()
    counts=torch.bincount(local[local>=0],minlength=32)
    route=(int(counts.sum()),int((counts>0).sum()),int(counts.max()))
    return dict(hidden_states=x,topk_ids=local,topk_weights=tw.float(),
                config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0)),route


def _best5_fp32(data,weights):
    # All dequantized expert matrices are temporary oracle scratch. Returned
    # candidate state retains only the original packed NVFP4 and E4M3 bank.
    x=data['hidden_states'];ids=data['topk_ids'];tw=data['topk_weights']
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=x.device)
    result=torch.zeros_like(x,dtype=torch.float32)
    for e in range(32):
        if not _best5_left():return None
        rows,slots=torch.where(ids==e)
        if rows.numel()==0:continue
        def decode(name):
            packed=weights[name][e]
            code=torch.stack((packed&15,packed>>4),-1).flatten(-2).long()
            return lut[code&7]*torch.where((code&8)!=0,-1.,1.)*weights[name+'_sf'][e].float().repeat_interleave(16,-1)*weights[name+'_gs'][e]
        first=x[rows].float()@decode('w13').T
        inter=(torch.nn.functional.silu(first[:,:4096])*first[:,4096:]).bfloat16()
        second=inter.float()@decode('w2').T
        second*=1.+weights['post_expert_norm'].float()
        second*=torch.rsqrt(second.square().mean(-1,keepdim=True)+1e-8)
        result.index_add_(0,rows,second*tw[rows,slots,None])
    return result


def _best5_error(got,want):
    actual=got.float()
    if not bool(torch.isfinite(actual).all()):return float('inf'),float('inf')
    delta=(actual-want).norm(dim=-1);norm=want.norm(dim=-1)
    return float(delta.norm()/norm.norm().clamp_min(1e-12)),float((delta/norm.clamp_min(.1*norm.mean()).clamp_min(1e-12)).max())


def _best5_graph(fn,data):
    static={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
    for _ in range(2):fn(static)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn(static)
    return static,graph,out


def _best5_refresh(graph,data):
    for key in ('hidden_states','topk_ids','topk_weights'):graph[0][key].copy_(data[key])


def _best5_tune_moe(state,weights,flush,device):
    import statistics
    state['moe_nt_plan']={m:1 for m in (32,128,256)}
    state['moe_nt_rows']={m:dict(nt1_us=None,nt2_us=None,selected=1,reason='budget') for m in (32,128,256)}
    # The FP32 oracle explicitly disables TF32 for the whole tuning block.
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        for m in (32,128,256):
            if not _best5_left(8):break
            row=state['moe_nt_rows'][m]
            fixtures=[];routes=[];wants=[]
            for j in range(3):
                if not _best5_left(4):break
                data,route=_best5_fixture(weights,m,9000+m+7919*j)
                want=_best5_fp32(data,weights)
                if want is None:break
                fixtures.append(data);routes.append(route);wants.append(want)
            row['routes']=routes
            if len(fixtures)!=3:continue
            bank1=state['moe_bank'];bank2=dict(bank1,mod=state['moe_nt2_mod'],nt=2)
            graphs=[_best5_graph(lambda d,bank=bank:v9_kernel_fn(d,bank),fixtures[0]) for bank in (bank1,bank2)]
            worst=[0.,0.];worst_row=[0.,0.]
            for i,g in enumerate(graphs):
                for data,want in zip(fixtures,wants):
                    if not _best5_left():break
                    _best5_refresh(g,data);g[1].replay()
                    rel,wr=_best5_error(g[2],want);worst[i]=max(worst[i],rel);worst_row[i]=max(worst_row[i],wr)
                if worst[i]>.01 or worst_row[i]>.05:
                    raise RuntimeError(f'V9 NT{i+1} FP32 gate M={m}: {worst[i]:.6f}/{worst_row[i]:.6f}')
            if not _best5_left(3):continue
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            samples=[[],[]]
            for repeat in range(12):
                if not _best5_left(2):break
                data=fixtures[repeat%3]
                for g in graphs:_best5_refresh(g,data)
                for side in ((0,1) if repeat%2==0 else (1,0)):
                    flush.sum();a.record();graphs[side][1].replay();b.record();b.synchronize()
                    samples[side].append(a.elapsed_time(b)*1000.)
            if len(samples[0])!=12:continue
            ratios=[(samples[1][i]/samples[0][i]*samples[1][i+1]/samples[0][i+1])**.5 for i in range(0,12,2)]
            # Agreement must also hold after route changes through graph replay.
            if not _best5_left(3):continue
            for i,g in enumerate(graphs):
                for data,want in zip(fixtures,wants):
                    _best5_refresh(g,data);g[1].replay()
                    rel,wr=_best5_error(g[2],want);worst[i]=max(worst[i],rel);worst_row[i]=max(worst_row[i],wr)
            if max(worst)>.01 or max(worst_row)>.05:raise RuntimeError(f'V9 replay FP32 gate M={m}: {worst}/{worst_row}')
            ratio=statistics.median(ratios)
            state['moe_nt_plan'][m]=2 if ratio<1. else 1
            row.update(nt1_us=round(statistics.median(samples[0]),3),nt2_us=round(statistics.median(samples[1]),3),
                       paired=round(ratio,6),max_pair=round(max(ratios),6),selected=state['moe_nt_plan'][m],
                       fp32=[round(x,6) for x in worst],worst_row=[round(x,6) for x in worst_row],reason='measured')
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous
