def route_call(pr,rs):
    p,s=pr;m=p.shape[1];dev=p.device
    x=torch.empty((m,4096),device=dev,dtype=torch.bfloat16)
    ids=torch.empty((m,8),device=dev,dtype=torch.int32);lids=torch.empty_like(ids);tw=torch.empty((m,8),device=dev)
    warps=4 if m==8 else 8
    timeline_route[(m,1)](p,s,p.stride(0),p.stride(1),rs['bias'],rs['pre'],rs['start'],x,ids,tw,lids,
        REPLICAS=1,USE_PDL=True,launch_pdl=True,num_warps=warps)
    return x,ids,tw,lids


def routed_experts(pr,rs,state,kind='INC'):
    x,ids,tw,lids=route_call(pr,rs)
    d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
    return nt1_call(d,state,kind),x,ids,tw,lids


def routed_fixture(w,m,seed):
    device=w['router'].device;gen=torch.Generator(device=device).manual_seed(seed)
    mean,shared=fx__basis(device);coef=torch.randn((m,fx_STREAM_RANK),generator=gen,device=device)
    q=8 if m==8 else 1;scale=torch.ones((m//q,q,1),device=device)
    if q>1:scale[:,1:]=fx_DRAFT_SHARE
    scale=scale.reshape(m,1)
    stream=scale*(fx_STREAM_MEAN*mean+(fx_STREAM_SHARED/fx_STREAM_RANK)**.5*(coef@shared))
    stream+=(1-fx_STREAM_SHARED*scale.square()).sqrt()*torch.randn((m,fx_HIDDEN),generator=gen,device=device)
    h=fx__rms(stream)*(1+w['pre_ffn_norm'].float())
    logits=h@w['router'].float().T;pooled=h@w['pooler'].float().T
    p=torch.cat((pooled,logits,torch.zeros_like(logits)),-1)[None].contiguous()
    biased=torch.sigmoid(logits*.1767766952966369)+w['router_bias'].float()
    ids=biased.topk(8,-1).indices;start=int(torch.bincount((ids//32).flatten(),minlength=8).argmax())*32
    rs=dict(bias=w['router_bias'].float().contiguous(),pre=(1+w['pre_expert_norm'].float()).contiguous(),
        start=torch.tensor([start],device=device,dtype=torch.int32))
    return (p,1),rs


def route_reference(pr,rs):
    p,s=pr;value=p[:s].double().sum(0)
    logits=(value[:,4096:4352]+value[:,4352:4608])*.1767766952966369
    scores=logits.sigmoid();biased=scores+rs['bias'].double()
    x=(value[:,:4096]*torch.rsqrt(value[:,:4096].square().mean(-1,keepdim=True)+1e-5)*rs['pre'].double()).float().to(torch.bfloat16)
    return x,scores,biased


def route_gate(ids,biased,start):
    # TASK.md/ref check_routing: allow legitimate near ties and gate local load.
    if bool((ids<0).any()) or bool((ids>=256).any()):raise ValueError('router expert out of range')
    ranked=ids.long().sort(-1).values
    if bool((ranked[:,1:]==ranked[:,:-1]).any()):raise ValueError('duplicate routed expert')
    best=biased.topk(8,-1);selected=biased.gather(1,ids.long())
    if float((best.values[:,-1:]-selected).max())>.05:raise ValueError('router score cutoff exceeds0.05')
    ninth=biased.topk(9,-1).values[:,-1:];kept=(best.indices[:,:,None]==ids.long()[:,None,:]).any(-1)
    local=(best.indices-start)%256<32
    if bool((local&~kept&(best.values-ninth>.03)).any()):raise ValueError('router dropped mandatory local expert')
    ref_local=best.indices[local];got_local=ids[((ids-start)%256)<32]
    if got_local.numel()<ref_local.numel()*.92-4:raise ValueError('router local route count')
    if got_local.unique().numel()<ref_local.unique().numel()*.85-2:raise ValueError('router distinct local experts')
    return int((ids.long().sort(-1).values!=best.indices.sort(-1).values).any(-1).sum())
