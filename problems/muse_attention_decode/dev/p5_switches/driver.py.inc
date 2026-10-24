_GROUP_EXT = {}


def _group_ext():
    key=Q8_DEFERRED_DENOM
    if key not in _GROUP_EXT:
        major,minor=torch.cuda.get_device_capability()
        arch='90a' if major==9 else f'{major}{minor}'
        _GROUP_EXT[key]=load_inline(name=f'muse_p5_group_d{int(key)}_sm{arch}',
            cpp_sources=GROUP_CPP,cuda_sources=GROUP_CUDA,functions=['xforward'],
            extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',
                f'-gencode=arch=compute_{arch},code=sm_{arch}',f'-DDEFER_DENOM={int(key)}','-lineinfo'],verbose=False)
    return _GROUP_EXT[key]


def xqa_group_attention(data,setting=None):
    cfg=data['config'];raw=data['qkv'];B=cfg['batch'];HQ=cfg['num_q_heads'];HK=cfg['num_kv_heads']
    assert cfg['q_len']==8
    maxkeys=data['page_table'].shape[1]*cfg['page_size']
    if cfg['window']:maxkeys=min(maxkeys,cfg['window'])
    if setting is None:
        sms=torch.cuda.get_device_properties(raw.device).multi_processor_count
        ns=min(64,max(1,(sms+B*HK-1)//(B*HK)),max(1,(maxkeys+63)//64))
        setting=(ns,5)
    ns,stages=setting
    assert 1<=ns<=64 and stages in (5,7)
    partial=torch.empty((B*HK*ns*64*68,),device=raw.device,dtype=torch.float32)
    out=torch.empty((B,8,HQ*64),device=raw.device,dtype=torch.bfloat16)
    freq=_inv_freq(cfg['rope_theta'],raw.device) if cfg['rope_theta'] else None
    _group_ext().xforward(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),
        data['page_table'].data_ptr(),data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),
        data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),freq.data_ptr() if freq is not None else 0,
        partial.data_ptr(),_counters(B*HK,raw.device).data_ptr(),out.data_ptr(),B,HQ,HK,raw.shape[-1],
        data['page_table'].shape[1],cfg['page_size'],cfg['window'],ns,
        cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,stages,0,
        torch.cuda.current_stream(raw.device).cuda_stream)
    return out


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg=data['config']
    if cfg['q_len']==1:
        return q1_attention(data)
    if not cfg['window']:
        return verify8_attention(data['qkv'],data)
    if cfg['batch']==1:
        return sliding_attention(data['qkv'],data)
    return xqa_group_attention(data)
