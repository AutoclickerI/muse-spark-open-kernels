def dc_route_gate(out,x,router,bias,mult,eps,pool_reference):
    T=torch;ids=out[1].long();ranked=ids.sort(-1).values
    assert int(ids.min())>=0 and int(ids.max())<256 and not bool((ranked[:,1:]==ranked[:,:-1]).any()),'unique expert ids'
    scores=T.sigmoid((x.double()@router.double().T).float()*mult);biased=scores+bias;best=biased.topk(9,-1)
    gap=float((best.values[:,7:8]-biased.gather(1,ids)).max());assert gap<1e-5,('route score gap',gap)
    reference=best.indices[:,:8];local=reference<32;kept=(reference[:,:,None]==ids[:,None,:]).any(-1)
    assert not bool((local&~kept&(best.values[:,:8]-best.values[:,8:9]>1e-5)).any()),'missed stable local expert'
    chosen_local=ids<32;routes=int(chosen_local.sum());distinct=int(T.unique(ids[chosen_local]).numel())
    assert routes>=int(local.sum())*.92-4 and distinct>=int(T.unique(reference[local]).numel())*.85-2,'local work bound'
    assert T.equal(out[3],T.where(chosen_local,ids,-1).int()),'local expert map'
    picked=scores.gather(1,ids);expected=picked/(picked.sum(-1,keepdim=True)+1e-15)
    assert float((out[2]-expected).norm()/expected.norm())<1e-4,'routing weight precision'
    p=pool_reference.float();norm=(p*T.rsqrt(p.square().mean(-1,keepdim=True)+eps)).bfloat16()
    error=float((out[0].float()-norm.float()).norm()/norm.float().norm());assert error<.004,('pooled norm',error)
    return max(gap,error)
