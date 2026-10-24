
def fixed_prepare(weights,kind='ABC',stages=2,grid=264):
    if kind not in NT1_SOURCES:raise ValueError(kind)
    nt1_module(kind);device=weights['w13'].device
    rows=torch.arange(512,device=device)[:,None]*8+torch.arange(8,device=device)
    maps={'w13':torch.cat((rows,rows+4096),1),'w2':torch.arange(4096,device=device).view(256,16)}
    bank={}
    for name,rowmap in maps.items():
        value=torch.stack([_repack(weights[name][e],weights[name+'_sf'][e],rowmap) for e in range(32)])
        tiles=rowmap.shape[0]
        if kind=='K2':
            w=value[...,:32768].reshape(32,tiles,2,16384).permute(0,2,1,3)
            sf=value[...,32768:].reshape(32,tiles,2,2048).permute(0,2,1,3)
            value=torch.cat((w,sf),-1).contiguous().reshape(32,tiles*2,18432)
        elif kind in ('D','ABCD','D128','ABCD128'):
            w=value[...,:32768].reshape(32,tiles,8,8,32,16).permute(0,1,3,2,4,5).reshape(32,tiles,8,4096)
            sf=value[...,32768:].reshape(32,tiles,8,32,8,2).permute(0,1,4,2,3,5).reshape(32,tiles,8,512)
            value=torch.cat((w,sf),-1).contiguous()
        bank[name+'p']=value
    return dict(kind=kind,stages=stages,grid=grid,banks={kind:bank},
        counters={kind:torch.zeros(128,device=device,dtype=torch.int32)},
        tickets=torch.zeros(256*768,device=device,dtype=torch.int32) if kind=='K2' else None,
        gs13=weights['w13_gs'].float().contiguous(),gs2=weights['w2_gs'].float().contiguous(),
        pnw=weights.get('post_norm_w',weights.get('post_expert_norm')).contiguous(),
        sms=torch.cuda.get_device_properties(device).multi_processor_count)


def fixed_experts(x,wts,lids,state):
    return nt1_call(dict(hidden_states=x,topk_ids=lids,topk_weights=wts,config={'local_start':0}),
        state,state['kind'],state['stages'],state['grid'])
