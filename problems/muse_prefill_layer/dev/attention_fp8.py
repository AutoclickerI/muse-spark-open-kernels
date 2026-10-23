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



"""Paged FP8 prefill attention; copied verbatim into the frozen candidate by its builder."""

@triton.jit
def _attn_fp8_kernel(q_ptr, kc_ptr, vc_ptr, pt_ptr, seq_ptr, o_ptr, ks_ptr, vs_ptr,
                     scale, L, max_pages, PAGE: tl.constexpr, WINDOW: tl.constexpr,
                     BQ: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr,
                     PSCALE: tl.constexpr, CONTIG: tl.constexpr):
    # Each CTA reuses one KV head across GROUP query heads and BQ query tokens.
    # Both tensor-core products consume E4M3 operands; all softmax state stays FP32.
    qb = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // (16 // GROUP)
    hg = bh % (16 // GROUP)
    h = hg // (8 // GROUP)
    M: tl.constexpr = BQ * GROUP
    rows = tl.arange(0, M)
    qi = qb * BQ + rows // GROUP
    qh = hg * GROUP + rows % GROUP
    qmask = qi < L
    seq = tl.load(seq_ptr + b)
    qpos = seq + qi
    d = tl.arange(0, 64)
    tok = (b * L + qi).to(tl.int64)
    q = tl.load(q_ptr + tok[:, None] * 1024 + qh[:, None] * 64 + d[None, :],
                mask=qmask[:, None], other=0.0)
    ks = tl.load(ks_ptr)
    vs = tl.load(vs_ptr)
    sc = scale * ks * 1.4426950408889634
    m_i = tl.full([M], -float('inf'), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, 64], tl.float32)
    q_lo = seq + qb * BQ
    q_hi = tl.minimum(seq + qb * BQ + BQ - 1, seq + L - 1)
    if WINDOW > 0:
        start = tl.maximum(q_lo - WINDOW + 1, 0)
    else:
        start = 0
    start = (start // BN) * BN
    for j0 in range(start, q_hi + 1, BN):
        j = j0 + tl.arange(0, BN)
        jm = j <= q_hi
        if CONTIG:
            r = b.to(tl.int64) * max_pages + j
        else:
            pg = tl.load(pt_ptr + b * max_pages + j // PAGE, mask=jm, other=0)
            r = pg.to(tl.int64) * PAGE + j % PAGE
        k = tl.load(kc_ptr + r[:, None] * 128 + h * 64 + d[None, :],
                    mask=jm[:, None], other=0.0)
        v = tl.load(vc_ptr + r[:, None] * 128 + h * 64 + d[None, :],
                    mask=jm[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * sc
        visible = (j[None, :] <= qpos[:, None]) & jm[None, :]
        if WINDOW > 0:
            visible = visible & (j[None, :] > qpos[:, None] - WINDOW)
        scores = tl.where(visible, scores, -float('inf'))
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        m_use = tl.where(m_new == -float('inf'), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(scores - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        # Scaling P uses the E4M3 range instead of wasting its exponent range on [0,1].
        # Keep the denominator in FP32; only the tensor-core P operand is rounded.
        p8 = (p * PSCALE).to(tl.float8e4nv)
        acc = acc * alpha[:, None] + tl.dot(p8, v)
        m_i = m_new
    out = acc / (l_i[:, None] * PSCALE) * vs
    tl.store(o_ptr + tok[:, None] * 1024 + qh[:, None] * 64 + d[None, :],
             out.to(tl.bfloat16), mask=qmask[:, None])


def _attention_fp8(data, q8, st, *, bq=16, bn=128, group=8, warps=8, stages=2, contig=False):
    cfg = data['config']
    tokens, batch, length = cfg['tokens'], cfg['batch'], cfg['q_len']
    if contig:
        capacity = data['page_table'].shape[1] * cfg['page_size']
        if cfg['window']:
            capacity = min(capacity, cfg['window'] + length - 1)
        S = triton.cdiv(capacity, 128) * 128
        kc = torch.empty((batch, S, HKV, HEAD_DIM), dtype=data['k_cache'].dtype, device=q8.device)
        vc = torch.empty_like(kc)
        lens = torch.empty_like(data['seq_lens'])
        seq = torch.empty_like(data['seq_lens'])
        _gather_cache_kernel[(triton.cdiv(S, 32), batch)](
            data['k_cache'].view(torch.uint8), data['v_cache'].view(torch.uint8),
            data['page_table'], data['seq_lens'], kc.view(torch.uint8), vc.view(torch.uint8), lens, seq,
            L=length, PAGE=cfg['page_size'], NP=data['page_table'].shape[1], S=S, WINDOW=cfg['window'], BR=32, num_warps=4)
        data = dict(data, k_cache=kc, v_cache=vc, seq_lens=seq)
        max_pages = S
    else:
        max_pages = data['page_table'].shape[1]
    out = torch.empty((tokens, Q_SIZE), dtype=torch.bfloat16, device=q8.device)
    _attn_fp8_kernel[(triton.cdiv(length, bq), batch * (HQ // group))](
        q8, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'], out,
        data['k_scale'], data['v_scale'], cfg['sm_scale'] * cfg['qk_scale_factor'],
        length, max_pages, PAGE=cfg['page_size'], WINDOW=cfg['window'],
        BQ=bq, BN=bn, GROUP=group, PSCALE=448.0, CONTIG=contig, num_warps=warps, num_stages=stages)
    return out


ATTENTION_FP8_MIN_L = 2048


def _attention(data, q8, st):
    cfg = data['config']
    if cfg['window'] == 0 and cfg['q_len'] >= ATTENTION_FP8_MIN_L:
        return _attention_fp8(data, q8, st)
    return _attention_fa3(data, q8, st)
