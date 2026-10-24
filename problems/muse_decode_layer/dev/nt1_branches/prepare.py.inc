
def prepare_kind(weights,kind='lop26'):
    if kind not in NT1_SOURCES:raise ValueError(kind)
    nt1_module(kind);device=weights['w13'].device
    rows=torch.arange(512,device=device)[:,None]*8+torch.arange(8,device=device)
    maps={'w13':torch.cat((rows,rows+4096),1),'w2':torch.arange(4096,device=device).view(256,16)}
    bank={}
    for name,rowmap in maps.items():
        repack=_raw_repack if kind=='raw' else _repack
        value=torch.stack([repack(weights[name][e],weights[name+'_sf'][e],rowmap) for e in range(32)])
        if kind=='split2':
            tiles=rowmap.shape[0]
            w=value[...,:32768].reshape(32,tiles,2,16384).permute(0,2,1,3)
            sf=value[...,32768:].reshape(32,tiles,2,2048).permute(0,2,1,3)
            value=torch.cat((w,sf),-1).contiguous().reshape(32,tiles*2,18432)
        bank[name+'p']=value
    return dict(banks={kind:bank},counters={kind:torch.zeros(64,device=device,dtype=torch.int32)},
        tickets=torch.zeros(256*768,device=device,dtype=torch.int32) if kind=='split2' else None,
        gs13=weights['w13_gs'].float().contiguous(),gs2=weights['w2_gs'].float().contiguous(),
        pnw=weights.get('post_norm_w',weights.get('post_expert_norm')).contiguous(),
        sms=torch.cuda.get_device_properties(device).multi_processor_count)
