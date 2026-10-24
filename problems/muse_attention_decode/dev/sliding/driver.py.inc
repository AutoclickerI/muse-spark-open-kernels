_WINDOW_EXT = None


def _window_ext():
    global _WINDOW_EXT
    if _WINDOW_EXT is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f'{major}{minor}'
        _WINDOW_EXT = load_inline(
            name=f'muse_window_mma_sm{arch}', cpp_sources=WINDOW_CPP, cuda_sources=WINDOW_CUDA,
            functions=['window_forward'],
            extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'],
            verbose=False)
    return _WINDOW_EXT


def sliding_attention(raw, data, splits=0):
    """Layer adapter: BF16 projected QKV plus the attention task's cache/config fields."""
    cfg = data['config']
    B, L, HQ, HK = cfg['batch'], cfg['q_len'], cfg['num_q_heads'], cfg['num_kv_heads']
    assert cfg['window'] and cfg['window'] <= 2048
    out = torch.empty((B, L, HQ*64), dtype=torch.bfloat16, device=raw.device)
    freq = _inv_freq(cfg['rope_theta'], raw.device) if cfg['rope_theta'] else None
    ns = 8 if B == 1 else (min(16, max(1, 132 // (B*HK))) if L == 1 else 1)
    parts = torch.empty((B*HQ*ns*L*68 if ns>1 else 1,),dtype=torch.float32,device=raw.device)
    counters = _counters(B*HQ, raw.device)
    _window_ext().window_forward(
        raw.data_ptr(), data['positions'].data_ptr(), data['cache_loc'].data_ptr(),
        data['page_table'].data_ptr(), data['seq_lens'].data_ptr(),
        data['k_cache'].data_ptr(), data['v_cache'].data_ptr(),
        data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        freq.data_ptr() if freq is not None else 0, out.data_ptr(), parts.data_ptr(), counters.data_ptr(),
        B, L, HQ, HK, raw.shape[-1], data['page_table'].shape[1], cfg['page_size'],
        cfg['window'], ns, splits, cfg['qk_scale_factor'], cfg['sm_scale'],
        torch.cuda.current_stream(raw.device).cuda_stream)
    return out


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data['config']
    if cfg['window']:
        return sliding_attention(data['qkv'], data)
    if cfg['q_len'] == 8 and torch.cuda.get_device_capability()[0] == 9:
        return _fa3_verify(data)
    return _full_attn_main(data)
