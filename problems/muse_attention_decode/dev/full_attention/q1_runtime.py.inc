
def q1_module(clocks=False,gather=False):
    key=(clocks,gather)
    if key not in _EXT:
        cap=torch.cuda.get_device_capability();arch='90a' if cap[0]==9 else f'{cap[0]}{cap[1]}'
        src=Q1_CLOCK if clocks or gather else Q1_BASE
        tag=hashlib.sha256(src.encode()).hexdigest()[:12]
        _EXT[key]=load_inline(name=f'fullq1_{tag}_{arch}_{int(clocks)}_{int(gather)}',cpp_sources=CPP,cuda_sources=src,
            functions=['attn_forward','max_cluster','occupancy'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',
            f'-gencode=arch=compute_{arch},code=sm_{arch}',f'-DFULL_CLOCK={int(clocks)}',f'-DGATHER_ONLY={int(gather)}','-lineinfo'],verbose=False)
    return _EXT[key]

def q1_attention(data,clocks=None,gather=False,plan=None):
    cfg=data['config'];B=cfg['batch'];L=cfg['q_len'];HK=cfg['num_kv_heads'];HQ=cfg['num_q_heads'];raw=data['qkv'];dev=raw.device
    assert L==1 and cfg['window']==0 and HQ==8*HK
    ext=q1_module(clocks is not None,gather)
    if plan is None:
        base=q1_module();sms=torch.cuda.get_device_properties(dev).multi_processor_count
        occ=(max(1,base.occupancy(1)),{cs:base.max_cluster(1,cs) for cs in range(1,17)})
        plan=_attn_main_plan(B,HK//2,data['page_table'].shape[1]*cfg['page_size'],1,0,sms,occ)
    ns,cs=plan;rows=B*(HK//2)*(ns//cs)*16
    po=torch.empty(rows*64,device=dev,dtype=torch.float32)
    nclk=B*(HK//2)*ns*4*16*2 if clocks is not None else 0
    ml=torch.zeros(rows*2+nclk,device=dev,dtype=torch.float32)
    key=(str(dev),B,HK,cs)
    if key not in _COUNTERS:_COUNTERS[key]=torch.zeros(B*(HK//2)*cs,device=dev,dtype=torch.int32)
    counters=_COUNTERS[key];out=torch.empty((B,L,HQ*64),device=dev,dtype=torch.bfloat16)
    ext.attn_forward(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),
        data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),
        0,po.data_ptr(),ml.data_ptr(),counters.data_ptr(),po.data_ptr(),1,out.data_ptr(),B,L,raw.shape[-1],HQ,HK,
        data['page_table'].shape[1],cfg['page_size'],0,ns,cs,cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,
        torch.cuda.current_stream(dev).cuda_stream)
    if clocks is not None:clocks.append(ml[rows*2:].view(torch.int64).view(-1,16))
    return out,plan
