"""Prepare-time FP32 agreement and cold-graph selection for the decode expert producer."""


def mw_install(state,weights,incumbent):
    dev=weights['w13'].device
    bank=mw_add_planes(mw_prepare(weights))
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    sms=torch.cuda.get_device_properties(dev).multi_processor_count
    choices=((64,16,128,2,sms*2,True,True),(64,32,128,2,sms*2,True,True),
             (128,32,128,2,sms*2,True,True))
    selected={};records={}
    for m in (128,256):
        ds=[fx_make_fixture(weights,m,9000+m+7919*j) for j in range(3)]
        for d in ds:
            d['config']=dict(d['config'],bs=m,top_k=8,n_experts=256,sharding='ep8')
        wants=[reuse_fp32(d,{**weights,'post_norm_w':weights['post_expert_norm']}) for d in ds]
        old=MWGraph(lambda d:incumbent(d['hidden_states'],d['topk_weights'],d['topk_ids'],state),ds[0])
        err,row=mw_gate(old,ds,wants)
        if err>.01 or row>.05:raise ValueError(f'mixed WGMMA incumbent gate M={m} rel={err} row={row}')
        scores=[]
        for choice in choices:
            new=MWGraph(lambda d,c=choice:mw_call(d,bank,c),ds[0])
            err,row=mw_gate(new,ds,wants)
            base,us,ratio=mw_pair(old,new,ds,flush)
            valid=err<=.01 and row<=.05
            scores.append(dict(choice=choice,base_us=base,us=us,ratio=ratio,fp32=err,row=row,valid=valid))
            del new
        admitted=[s for s in scores if s['valid'] and s['ratio']<.98]
        if admitted:selected[m]=min(admitted,key=lambda s:s['ratio'])['choice']
        records[m]=scores
        print(f'MW_GATE M={m} selected={selected.get(m)} trials={scores}',flush=True)
        del old,wants,ds
    state['mw_choices']=selected
    state['mw_gate']=records
    if selected:state['mw_bank']=bank
    return state


_mw_incumbent_experts=experts_v9
_mw_incumbent_prepare=prepare


def experts_v9(x,wts,lids,state):
    choice=state['mw_choices'].get(x.shape[0])
    if choice is None:return _mw_incumbent_experts(x,wts,lids,state)
    data={'hidden_states':x,'topk_ids':lids,'topk_weights':wts,
          'config':{'local_start':0,'local_count':32,'d_hidden':4096,'d_expert':4096}}
    return mw_call(data,state['mw_bank'],choice)


def prepare(weights):
    return mw_install(_mw_incumbent_prepare(weights),weights,_mw_incumbent_experts)
