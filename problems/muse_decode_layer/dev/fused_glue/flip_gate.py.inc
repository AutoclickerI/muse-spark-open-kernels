def fg_flip_report(m,data,state,plan,actual,want,product=None):
    """Record score-space evidence. Every flipped cell is disabled by the caller."""
    mask=(actual!=want).any(-1);n=int(mask.sum())
    result=dict(flips=n,tokens=actual.shape[0],near_tie=False)
    if not n:return result
    qkv=m._best7_project(data['attn_input'],'qkv_gate',state,plan)
    attn=m._best7_attention(qkv,data,state,plan)
    branch=m._best7_project(attn,'o',state,plan)
    sync=None
    if plan['glue']=='best5':
        t=data['config']['tokens'];raw=torch.zeros(15*t,device=branch[0].device,dtype=torch.int32)
        sync=(raw[3*t:].view(torch.float32).view(3,t,4)[0],raw[:3*t].view(3,t)[0])
    h,_=m._best7_boundary(branch,data['stream'],state,plan,'b1',sync)
    pr,s=m._best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)
    ref=pr[:s].sum(0)
    score=((ref[:,4096:4352]+ref[:,4352:4608])*.1767766952966369).sigmoid()+state['router_bias']
    top=score.topk(9,-1).values;gap=top[:,7]-top[:,8]
    shifted=score
    if product is not None:
        pp=product[:s].sum(0) if product.ndim==3 else product
        shifted=((pp[:,4096:4352]+pp[:,4352:4608])*.1767766952966369).sigmoid()+state['router_bias']
        result['product_l2']=dp_error(pp,ref)[0]
    delta=(shifted-score).abs().amax(-1)
    selected=score.gather(1,actual.long()).amin(-1)
    # A small threshold alone is insufficient evidence: also print the actual
    # output perturbation and the selected score's distance from the cutoff.
    near=(gap<=2*delta+8*torch.finfo(torch.float32).eps)&(selected>=top[:,7]-2*delta-8*torch.finfo(torch.float32).eps)
    if product is not None:near&=result['product_l2']<.005
    rows=mask.nonzero().flatten()
    result.update(near_tie=bool(near[mask].all()),gap8_9=gap[mask].tolist(),
        max_score_delta=delta[mask].tolist(),cutoff_loss=(top[:,7]-selected)[mask].tolist(),
        token_index=rows.tolist(),actual=actual[mask].tolist(),control=want[mask].tolist())
    return result
