_VERIFY8_EXT = None


def _verify8_ext():
    global _VERIFY8_EXT
    if _VERIFY8_EXT is None:
        major,minor=torch.cuda.get_device_capability()
        arch=f'{major}{minor}'
        _VERIFY8_EXT=load_inline(name=f'muse_flashdecode_q8_sm{arch}',cpp_sources=VERIFY8_CPP,
            cuda_sources=VERIFY8_CUDA,functions=['verify8_forward'],
            extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'],verbose=False)
    return _VERIFY8_EXT


def verify8_attention(raw,data,setting=None,projection_splits=0):
    cfg=data['config'];B=cfg['batch'];HQ=cfg['num_q_heads'];HK=cfg['num_kv_heads']
    assert cfg['q_len']==8 and not cfg['window'] and cfg['head_dim']==64 and HQ==8*HK
    if setting is None:
        # Several waves keep independent page gathers in flight. The only
        # inputs to this launch policy are architecture and tensor shapes.
        sms=torch.cuda.get_device_properties(raw.device).multi_processor_count
        maxkeys=data['page_table'].shape[1]*cfg['page_size']
        waves=8 if B>=32 else 4
        ns=min(128,max(1,(waves*sms+B*HK-1)//(B*HK)),max(1,(maxkeys+127)//128))
        setting=(ns,3,64 if B==1 else 128)
    ns,stages,kt=setting
    assert ns>0 and stages in (3,4) and kt in (64,128)
    parts=torch.empty((B*HK*ns*64*68,),device=raw.device,dtype=torch.float32)
    out=torch.empty((B,8,HQ*64),device=raw.device,dtype=torch.bfloat16)
    _verify8_ext().verify8_forward(raw.data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),
        data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),
        data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),parts.data_ptr(),out.data_ptr(),
        B,HQ,HK,raw.shape[-1],data['page_table'].shape[1],cfg['page_size'],ns,projection_splits,
        cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,stages,kt,
        torch.cuda.current_stream(raw.device).cuda_stream)
    return out


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg=data['config']
    if cfg['q_len']==8 and not cfg['window']:
        return verify8_attention(data['qkv'],data)
    if cfg['q_len']==8 and cfg['batch']>1 and torch.cuda.get_device_capability()[0]==9:
        return _fa3_verify(data)
    return _incumbent_attn_main(data)
