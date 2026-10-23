# Timed, lossless FP8 cache gather enables FA3's contiguous-cache load path.
@triton.jit
def _gather_cache_kernel(K, V, PT, Seq, KD, VD, Lens, SeqD,
                         L: tl.constexpr, PAGE: tl.constexpr, NP: tl.constexpr,
                         S: tl.constexpr, WINDOW: tl.constexpr, BR: tl.constexpr):
    b = tl.program_id(1)
    seq = tl.load(Seq + b)
    first = tl.maximum(0, seq - WINDOW + 1) if WINDOW else 0
    length = seq + L - first
    if tl.program_id(0) == 0:
        tl.store(Lens + b, length)
        tl.store(SeqD + b, seq - first)
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    pos = r + first
    valid = (r < length) & (r < S)
    pg = tl.load(PT + b * NP + pos // PAGE, valid, 0).to(tl.int64)
    src = pg * PAGE + pos % PAGE
    c = tl.arange(0, 128)
    k = tl.load(K + src[:, None] * 128 + c[None, :], valid[:, None], 0)
    v = tl.load(V + src[:, None] * 128 + c[None, :], valid[:, None], 0)
    dst = (b.to(tl.int64) * S + r[:, None]) * 128 + c[None, :]
    tl.store(KD + dst, k, (r < S)[:, None])
    tl.store(VD + dst, v, (r < S)[:, None])


def _attention_contiguous(data, q8, st):
    cfg = data['config']
    B, L, W = cfg['batch'], cfg['q_len'], cfg['window']
    capacity = data['page_table'].shape[1] * cfg['page_size']
    if W:
        capacity = min(capacity, W + L - 1)
    S = triton.cdiv(capacity, 128) * 128
    kc = torch.empty((B, S, HKV, HEAD_DIM), dtype=data['k_cache'].dtype, device=q8.device)
    vc = torch.empty_like(kc)
    lens = torch.empty_like(data['cache_seqlens'])
    seq = torch.empty_like(data['seq_lens'])
    _gather_cache_kernel[(triton.cdiv(S, 32), B)](
        data['k_cache'].view(torch.uint8), data['v_cache'].view(torch.uint8), data['page_table'], data['seq_lens'], kc.view(torch.uint8), vc.view(torch.uint8), lens, seq,
        L=L, PAGE=cfg['page_size'], NP=data['page_table'].shape[1], S=S, WINDOW=W, BR=32, num_warps=4)
    if IS_HOPPER:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        descale = {}
        if cfg['kv_scale'] != 1.0:
            descale = dict(k_descale=data['k_scale'].reshape(1,1).expand(B,HKV),
                           v_descale=data['v_scale'].reshape(1,1).expand(B,HKV))
        out = flash_attn_with_kvcache(q=q8.view(B*L,HQ,HEAD_DIM), k_cache=kc, v_cache=vc,
                cache_seqlens=lens, cu_seqlens_q=data['cu_seqlens_q'], max_seqlen_q=L,
                softmax_scale=cfg['sm_scale']*cfg['qk_scale_factor'], causal=True,
                window_size=(W-1,0) if W else (-1,-1), **descale)
        if isinstance(out,tuple): out=out[0]
        return out.reshape(B*L,Q_SIZE)
    # The local fallback reads exactly the gathered buffers, with one page per request.
    packed = dict(data, k_cache=kc, v_cache=vc, cache_seqlens=lens, seq_lens=seq,
                  page_table=torch.arange(B,device=q8.device,dtype=torch.int32).view(B,1),
                  config=dict(cfg,page_size=S))
    return _attention(packed, q8, st)
