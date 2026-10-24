def _best7_run(data,state,plan,prefetch=None):
    t=data['config']['tokens']
    sync=[None,None,None]
    if plan['glue']=='best5':
        raw=(torch.empty if _best7_old_chunks(t)==1 else torch.zeros)((15*t,),device=data['stream'].device,dtype=torch.int32)
        si=raw[:3*t].view(3,t);sf=raw[3*t:].view(torch.float32).view(3,t,4)
        sync=[(sf[i],si[i]) for i in range(3)]
    qkv=_best7_project(data['attn_input'],'qkv_gate',state,plan)
    if prefetch is not None:prefetch.begin('front')
    attn=_best7_attention(qkv,data,state,plan)
    if prefetch is not None:
        prefetch.join('front');prefetch.begin('pool')
    if plan.get('dense_chain', False):
        if prefetch is not None:prefetch.join('pool')
        h,stream1,pr_out=_dense_chain_call(attn,data['stream'],state,data)
        pr=(pr_out.unsqueeze(0),1)
    else:
        a=_best7_project(attn,'o',state,plan)
        h,stream1=_best7_boundary(a,data['stream'],state,plan,'b1',sync[0])
        if prefetch is not None:prefetch.join('pool')
        pr=_best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)
    x,ids,wts,lids=_best7_route(pr,state,data,plan,sync[1])
    if prefetch is not None:prefetch.begin('tail')
    moe=_best7_expert(pr,x,wts,lids,state,plan)
    if prefetch is not None:prefetch.join('tail')
    name='unpooler_slice' if t<=SLICED_MAX_TOKENS else 'unpooler'
    f=_best7_project(moe,name,state,plan)
    hidden,stream2=_best7_boundary(f,stream1,state,plan,'b2',sync[2])
    return hidden,stream2,ids


def kernel_fn(data,state):
    cfg=data['config'];key=(cfg['batch'],cfg['q_len'],cfg['window'])
    plan=state['best7_plans'].get(key,state['best7_plans']['other'])
    return _best7_run(data,state,plan,state['best7_prefetch_ops'].get(key))
