
_NATIVE_EXT={}

def native_ext(force_hopper=False):
    # force_hopper exercises the Hopper Python dispatcher locally; the CUDA
    # shim preserves its stage/barrier/layout work on the sm120 screen GPU.
    major,minor=torch.cuda.get_device_capability()
    arch='90a' if major==9 else f'{major}{minor}'
    key=(arch,bool(force_hopper))
    if key not in _NATIVE_EXT:
        _NATIVE_EXT[key]=load_inline(name=f'verify8_native_{hashlib.sha256(NATIVE_CUDA.encode()).hexdigest()[:12]}_{arch}',
            cpp_sources=NATIVE_CPP,cuda_sources=NATIVE_CUDA,functions=['native_forward','native_resources'],
            extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'],verbose=False)
    return _NATIVE_EXT[key]


def native_attention(qkv,data,setting=(17,4),clocks=False,force_hopper=False):
    cfg=data['config'];B=cfg['batch'];HQ=cfg['num_q_heads'];HK=cfg['num_kv_heads']
    assert cfg['q_len']==8 and not cfg['window'] and HQ==8*HK and cfg['head_dim']==64
    count=0
    if isinstance(qkv,tuple):
        raw,count=qkv
        if raw.dtype==torch.bfloat16:
            assert count==1
            count=0
        else:assert raw.dtype==torch.float32
    else:raw=qkv
    assert raw.is_contiguous()
    ns,st=setting;assert 1<=ns<=128 and st in (4,8)
    parts=torch.empty((B*HK*ns*64*68,),device=raw.device,dtype=torch.float32)
    out=torch.empty((B,8,HQ*64),device=raw.device,dtype=torch.bfloat16)
    trace=torch.empty((B*HK*ns,8,10) if clocks else (1,),device=raw.device,dtype=torch.uint64)
    native_ext(force_hopper).native_forward(raw.data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),
        data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),
        data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),parts.data_ptr(),out.data_ptr(),
        B,HQ,HK,raw.shape[-1],data['page_table'].shape[1],cfg['page_size'],ns,count,
        cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,st,int(clocks),trace.data_ptr(),
        torch.cuda.current_stream(raw.device).cuda_stream)
    return out,trace


def attention_verify8_native(qkv,data,setting=(17,4)):
    cfg=dict(data['config'],num_q_heads=16,num_kv_heads=2,head_dim=64)
    return native_attention(qkv,dict(data,config=cfg),setting,force_hopper=True)[0].view(cfg['batch']*8,1024)
