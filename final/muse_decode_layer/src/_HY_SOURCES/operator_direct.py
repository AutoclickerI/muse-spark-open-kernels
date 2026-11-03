# MIT6d9b7765, Copyright2026bigboateng; full notice in shipped source.
# Changes: direct FP32 prologue; selected sliding8 family only.
import torch,triton
import triton.language as tl
RMS_EPS=1e-5
MERGE_ELEMS=tl.constexpr(16384)
_ROPE={};_COUNTERS={};ROPE_POSITIONS=1<<21
_DIRECT_PS=0;_DIRECT_SS=0;_DIRECT_WIDTH=2304
FUSED_CFG={("sliding",8):dict(block_n=128, warps=4, stages=3, max_split=8, per_sm=4, row_blocks=1, small=dict(block_n=128, stages=2, max_split=16, row_blocks=4, qk8=False), big=dict(max_split=4), big_min_batch=32)}
@triton.jit
def _combine_ml(part_o, part_ml, first, count, r, dd, ROWS: tl.constexpr, D: tl.constexpr, NP2: tl.constexpr):
    """Exact log-sum-exp merge of `count` partials starting at index `first`: (unnormalized o, l, m)."""
    SB1: tl.constexpr = MERGE_ELEMS // (ROWS * D) if MERGE_ELEMS // (ROWS * D) > 0 else 1
    if SB1 >= NP2:
        # Every partial fits in one step: issue the m, l and o loads together (one L2 round trip).
        sa = tl.arange(0, NP2)
        sam = sa < count
        mptr1 = part_ml + ((first + sa[:, None]) * ROWS + r[None, :]) * 2
        ma = tl.load(mptr1, mask=sam[:, None], other=-float("inf"), cache_modifier=".cg")
        la = tl.load(mptr1 + 1, mask=sam[:, None], other=0.0, cache_modifier=".cg")
        oa = tl.load(part_o + ((first + sa[:, None, None]) * ROWS + r[None, :, None]) * D + dd[None, None, :],
                     mask=sam[:, None, None], other=0.0, cache_modifier=".cg")
        mx1 = tl.max(ma, 0)
        mx1 = tl.where(mx1 == -float("inf"), 0.0, mx1)
        wa = tl.exp2(ma - mx1[None, :])
        return tl.sum(wa[:, :, None] * oa, 0), tl.sum(wa * la, 0), mx1
    si = tl.arange(0, NP2)
    smask = si[:, None] < count
    mptr = part_ml + ((first + si[:, None]) * ROWS + r[None, :]) * 2
    ms = tl.load(mptr, mask=smask, other=-float("inf"), cache_modifier=".cg")
    ls = tl.load(mptr + 1, mask=smask, other=0.0, cache_modifier=".cg")
    mx = tl.max(ms, 0)
    mx = tl.where(mx == -float("inf"), 0.0, mx)
    lsum = tl.sum(tl.exp2(ms - mx[None, :]) * ls, 0)
    o = tl.zeros([ROWS, D], tl.float32)
    SB: tl.constexpr = MERGE_ELEMS // (ROWS * D) if MERGE_ELEMS // (ROWS * D) > 0 else 1
    sb = tl.arange(0, SB)
    for k0 in range(0, count, SB):
        sj = k0 + sb
        sjm = sj < count
        mj = tl.load(part_ml + ((first + sj[:, None]) * ROWS + r[None, :]) * 2,
                     mask=sjm[:, None], other=-float("inf"), cache_modifier=".cg")
        oj = tl.load(part_o + ((first + sj[:, None, None]) * ROWS + r[None, :, None]) * D + dd[None, None, :],
                     mask=sjm[:, None, None], other=0.0, cache_modifier=".cg")
        o += tl.sum(tl.exp2(mj - mx[None, :])[:, :, None] * oj, 0)
    return o, lsum, mx


@triton.jit
def _combine(part_o, part_ml, first, count, r, dd, ROWS: tl.constexpr, D: tl.constexpr, NP2: tl.constexpr):
    o, lsum, mx = _combine_ml(part_o, part_ml, first, count, r, dd, ROWS, D, NP2)
    return o, lsum


@triton.jit
def _emit_pre(o, lsum, v_scale, eps, gate, out_ptr, tok, qh, rvalid, dd, HQ: tl.constexpr, D: tl.constexpr):
    """_emit with the gate already in registers (loaded at kernel start, off the merge's critical path)."""
    o = (o / lsum[:, None] * v_scale).to(tl.bfloat16).to(tl.float32)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    o = o * tl.sigmoid(gate.to(tl.float32))
    tl.store(out_ptr + (tok.to(tl.int64)[:, None] * HQ + qh[:, None]) * D + dd[None, :], o.to(tl.bfloat16),
             mask=rvalid[:, None])


@triton.jit
def _emit(o, lsum, v_scale, eps, qkv_ptr, out_ptr, tok, qh, rvalid, dd, qkv_stride,
          HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr):
    """Normalize the merged attention, round to BF16 as FA3's output is, then RMSNorm x sigmoid(gate)."""
    o = (o / lsum[:, None] * v_scale).to(tl.bfloat16).to(tl.float32)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    gptr = qkv_ptr + tok.to(tl.int64)[:, None] * qkv_stride + (HQ + 2 * HKV) * D + qh[:, None] * D + dd[None, :]
    gate = tl.load(gptr, mask=rvalid[:, None], other=0.0).to(tl.float32)
    o = o * tl.sigmoid(gate)
    tl.store(out_ptr + (tok.to(tl.int64)[:, None] * HQ + qh[:, None]) * D + dd[None, :], o.to(tl.bfloat16),
             mask=rvalid[:, None])


@triton.jit
def _partial_load(ptr,mask,S:tl.constexpr,SS:tl.constexpr):
 v=tl.load(ptr,mask,other=0).to(tl.float32)
 if S>0:
  for i in tl.static_range(1,S):v+=tl.load(ptr+i*SS,mask,other=0).to(tl.float32)
 return v.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _fused_attn(qkv_ptr, loc_ptr, cos_sin_ptr, ks_ptr, vs_ptr, k_cache, v_cache, pt_ptr, seq_ptr,
                part_o, part_ml, part2_o, part2_ml, cnt_ptr, out_ptr,
                qkv_stride, pt_stride, chunk, sm_scale, eps,
                HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, L: tl.constexpr, HPB: tl.constexpr,
                ROWS: tl.constexpr, LP: tl.constexpr, PAGE: tl.constexpr, BLOCK_N: tl.constexpr,
                NSPLIT: tl.constexpr, NSP2: tl.constexpr, SB: tl.constexpr, WINDOW: tl.constexpr,
                ROPE: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr, NG2: tl.constexpr,
                PREFETCH: tl.constexpr, NRB: tl.constexpr, QK8: tl.constexpr, GPRE: tl.constexpr, PS:tl.constexpr,PSS:tl.constexpr,PWIDTH:tl.constexpr):
    b = tl.program_id(0)
    hblk = tl.program_id(1) // NRB             # block of HPB KV heads
    rb = tl.program_id(1) % NRB                # block of ROWS query rows within it
    s = tl.program_id(2)                       # split of the cached keys
    G: tl.constexpr = HQ // HKV
    NHB: tl.constexpr = HKV // HPB
    DB: tl.constexpr = HPB * D                 # columns of one block: HPB heads side by side
    HALF: tl.constexpr = D // 2
    LOG2E: tl.constexpr = 1.4426950408889634
    seq = tl.load(seq_ptr + b)                 # cached tokens; the block's tokens sit at seq .. seq+L-1
    k_scale = tl.load(ks_ptr)
    v_scale = tl.load(vs_ptr)
    scale = sm_scale * k_scale * LOG2E

    # Query rows r = (head in block, token, group); block-diagonal columns (zero off the row's head).
    r = tl.arange(0, ROWS)
    rg = rb * ROWS + r                         # row index within the head block
    hb = rg // (L * G)
    j = (rg // G) % L
    g = rg % G
    rvalid = rg < HPB * L * G
    qh = (hblk * HPB + hb) * G + g
    tok = b * L + j
    c2 = tl.arange(0, DB // 2)                 # RoPE pair index across the block
    chb = c2 // HALF
    ci = c2 % HALF
    qmask = rvalid[:, None] & (chb[None, :] == hb[:, None])
    qptr = qkv_ptr + tok.to(tl.int64)[:, None] * qkv_stride + qh[:, None] * D + 2 * ci[None, :]
    qe = _partial_load(qptr,qmask,PS,PSS).to(tl.float32)
    qo = _partial_load(qptr+1,qmask,PS,PSS).to(tl.float32)
    inv = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / D + eps)[:, None]
    qe = (qe * inv).to(tl.bfloat16).to(tl.float32)
    qo = (qo * inv).to(tl.bfloat16).to(tl.float32)
    qpos = seq + j
    if ROPE:
        cs = cos_sin_ptr + qpos.to(tl.int64)[:, None] * D + ci[None, :]
        cq = tl.load(cs, mask=qmask, other=0.0)
        sq = tl.load(cs + HALF, mask=qmask, other=0.0)
        qe, qo = (qe * cq - qo * sq).to(tl.bfloat16).to(tl.float32), (qe * sq + qo * cq).to(tl.bfloat16).to(tl.float32)
    # Queries rounded to E4M3, as FA3 does.
    q8 = tl.reshape(tl.join(qe, qo), [ROWS, DB]).to(tl.float8e4nv)
    q = q8.to(tl.float16)
    # The gate is needed only by the merging program, but loading it here takes its DRAM round trip off the
    # merge's critical path (every program loads it; 2 KB).
    if GPRE:
        dgt = tl.arange(0, D)
        if PWIDTH == (HQ+2*HKV)*D:
            gate_pre=tl.full((ROWS,D),float('inf'),tl.float32)
        else:
            gate_pre=_partial_load(qkv_ptr+tok.to(tl.int64)[:,None]*qkv_stride+(HQ+2*HKV)*D+qh[:,None]*D+dgt[None,:],rvalid[:,None],PS,PSS)


    dcol = tl.arange(0, DB)
    m_i = tl.full([ROWS], -float("inf"), tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, DB], tl.float32)

    # Cached keys of this split: [lo, seq) on full layers, the window's cached part on sliding ones.
    if WINDOW > 0:
        lo = tl.maximum(seq - WINDOW + 1, 0)
    else:
        lo = 0
    start = lo + s * chunk
    end = tl.minimum(start + chunk, seq)
    pt_row = pt_ptr + b.to(tl.int64) * pt_stride
    n = start + tl.arange(0, BLOCK_N)
    if PREFETCH:
        page = tl.load(pt_row + n // PAGE, mask=n < end, other=0)
    else:
        page = n
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < end
        if PREFETCH:
            # The next tile's page indices load while this tile's rows are fetched and used.
            nn = n + BLOCK_N
            page_next = tl.load(pt_row + nn // PAGE, mask=nn < end, other=0)
            row = page.to(tl.int64) * PAGE + n % PAGE
            page = page_next
        else:
            row = tl.load(pt_row + n // PAGE, mask=nm, other=0).to(tl.int64) * PAGE + n % PAGE
        off = row[:, None] * (HKV * D) + hblk * DB + dcol[None, :]
        if QK8:
            k = tl.load(k_cache + off, mask=nm[:, None], other=0.0)
        else:
            k = tl.load(k_cache + off, mask=nm[:, None], other=0.0).to(tl.float16)
        v = tl.load(v_cache + off, mask=nm[:, None], other=0.0).to(tl.float16)
        if QK8:
            # E4M3 x E4M3 products are exact in the fp32 accumulator; q is already E4M3-rounded, as FA3 does.
            sc = tl.dot(q8, tl.trans(k), out_dtype=tl.float32) * scale
        else:
            sc = tl.dot(q, tl.trans(k)) * scale
        ok = nm[None, :]
        if WINDOW > 0:
            ok = ok & (n[None, :] > qpos[:, None] - WINDOW)
        sc = tl.where(ok, sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_use = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp2(sc - m_use[:, None])
        alpha = tl.exp2(m_i - m_use)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    if s == 0:
        # The block's new keys and values: normalized (+ RoPE), stored to the cache as E4M3, and attended
        # causally from registers (other programs read only rows below seq).
        jn = tl.arange(0, LP)
        jm = jn < L
        ntok = b * L + jn
        kptr = qkv_ptr + ntok.to(tl.int64)[:, None] * qkv_stride + (HQ + hblk * HPB) * D + 2 * c2[None, :]
        ke = _partial_load(kptr,jm[:,None],PS,PSS).to(tl.float32)
        ko = _partial_load(kptr+1,jm[:,None],PS,PSS).to(tl.float32)
        ke3 = tl.reshape(ke, [LP, HPB, HALF])
        ko3 = tl.reshape(ko, [LP, HPB, HALF])
        kinv = tl.rsqrt((tl.sum(ke3 * ke3, 2) + tl.sum(ko3 * ko3, 2)) / D + eps)[:, :, None]
        ke = tl.reshape((ke3 * kinv).to(tl.bfloat16).to(tl.float32), [LP, DB // 2])
        ko = tl.reshape((ko3 * kinv).to(tl.bfloat16).to(tl.float32), [LP, DB // 2])
        if ROPE:
            npos = seq + jn
            csk = cos_sin_ptr + npos.to(tl.int64)[:, None] * D + ci[None, :]
            ck = tl.load(csk, mask=jm[:, None], other=0.0)
            sk = tl.load(csk + HALF, mask=jm[:, None], other=0.0)
            ke, ko = (ke * ck - ko * sk).to(tl.bfloat16).to(tl.float32), (ke * sk + ko * ck).to(tl.bfloat16).to(tl.float32)
        k8 = (tl.reshape(tl.join(ke, ko), [LP, DB]) / k_scale).to(tl.float8e4nv)
        vraw = _partial_load(qkv_ptr + ntok.to(tl.int64)[:, None] * qkv_stride + (HQ + HKV + hblk * HPB) * D + dcol[None, :],jm[:,None],PS,PSS).to(tl.float32)
        v8 = (vraw / v_scale).to(tl.float8e4nv)
        loc = tl.load(loc_ptr + ntok, mask=jm, other=0)
        coff = loc[:, None] * (HKV * D) + hblk * DB + dcol[None, :]
        tl.store(k_cache + coff, k8, mask=jm[:, None] & (rb == 0))
        tl.store(v_cache + coff, v8, mask=jm[:, None] & (rb == 0))
        sc = tl.dot(q, tl.trans(k8.to(tl.float16))) * scale
        sc = tl.where(jm[None, :] & (jn[None, :] <= j[:, None]), sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        p = tl.exp2(sc - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v8.to(tl.float16))
        m_i = m_new

    # This split's partial: each row keeps only its own head's columns.
    dd = tl.arange(0, D)
    if HPB == 1:
        o_part = acc
    else:
        acc3 = tl.reshape(acc, [ROWS, HPB, D])
        sel = tl.arange(0, HPB)[None, :, None] == hb[:, None, None]
        o_part = tl.sum(tl.where(sel, acc3, 0.0), 1)
    unit = ((b * NHB + hblk) * NRB + rb).to(tl.int64)
    pbase = (unit * NSPLIT + s) * ROWS
    tl.store(part_o + (pbase + r[:, None]) * D + dd[None, :], o_part)
    tl.store(part_ml + (pbase + r) * 2, m_i)
    tl.store(part_ml + (pbase + r) * 2 + 1, l_i)

    tl.debug_barrier()
    if NG == 1:
        done = tl.atomic_add(cnt_ptr + unit, 1)
        if done == NSPLIT - 1:
            # Last program of this unit: exact merge of every split, then RMSNorm x sigmoid(gate).
            o, lsum = _combine(part_o, part_ml, unit * NSPLIT, NSPLIT, r, dd, ROWS, D, NSP2)
            if GPRE:
                _emit_pre(o, lsum, v_scale, eps, gate_pre, out_ptr, tok, qh, rvalid, dd, HQ, D)
            else:
                _emit(o, lsum, v_scale, eps, qkv_ptr, out_ptr, tok, qh, rvalid, dd, qkv_stride, HQ, HKV, D)
            tl.atomic_xchg(cnt_ptr + unit, 0)
    else:
        # Two levels: the last program of each group of GS splits merges the group; the last group
        # merger merges the groups and writes the output.
        grp = s // GS
        gsize = tl.minimum(GS, NSPLIT - grp * GS)
        gcnt = cnt_ptr + unit * (NG + 1) + grp
        done = tl.atomic_add(gcnt, 1)
        if done == gsize - 1:
            o, lsum, mx = _combine_ml(part_o, part_ml, unit * NSPLIT + grp * GS, gsize, r, dd, ROWS, D, GS)
            gbase = (unit * NG + grp) * ROWS
            tl.store(part2_o + (gbase + r[:, None]) * D + dd[None, :], o)
            tl.store(part2_ml + (gbase + r) * 2, mx)
            tl.store(part2_ml + (gbase + r) * 2 + 1, lsum)
            tl.atomic_xchg(gcnt, 0)
            tl.debug_barrier()
            tcnt = cnt_ptr + unit * (NG + 1) + NG
            done2 = tl.atomic_add(tcnt, 1)
            if done2 == NG - 1:
                o, lsum = _combine(part2_o, part2_ml, unit * NG, NG, r, dd, ROWS, D, NG2)
                if GPRE:
                    _emit_pre(o, lsum, v_scale, eps, gate_pre, out_ptr, tok, qh, rvalid, dd, HQ, D)
                else:
                    _emit(o, lsum, v_scale, eps, qkv_ptr, out_ptr, tok, qh, rvalid, dd, qkv_stride, HQ, HKV, D)
                tl.atomic_xchg(tcnt, 0)


def _counter(n, device):
    """Per-unit arrival counters; each merge resets its own to zero (shape-sized, not data-derived)."""
    key = (device, n)
    if key not in _COUNTERS:
        _COUNTERS[key] = torch.zeros(n, dtype=torch.int32, device=device)
    return _COUNTERS[key]


def _fused(data, cfg, qkv, tokens, cos_sin):
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    window = cfg["window"]
    kind = "sliding" if window else "full"
    fc = FUSED_CFG[(kind, 1 if length == 1 else 8)]
    if batch <= 4 and "small" in fc:   # shape-only choice: small batches favour more, smaller programs
        fc = {**fc, **fc["small"]}
    if batch >= fc.get("big_min_batch", 1 << 30) and "big" in fc:
        fc = {**fc, **fc["big"]}
    group = hq // hkv
    hpb=fc.get('hpb',2) if length==1 and hkv%2==0 else 1
    rows=max(16,triton.next_power_of_2(hpb*length*group))
    nrb=max(1,min(fc.get('row_blocks',1),rows//16));rows//=nrb
    page_table = data["page_table"]
    span = (window - 1) if window else page_table.shape[1] * page   # cached keys a request can attend
    units = batch * (hkv // hpb) * nrb
    target = fc["per_sm"] * torch.cuda.get_device_properties(qkv.device).multi_processor_count
    nsplit = max(1, min(fc["max_split"], triton.cdiv(span, max(fc["block_n"], fc.get("min_chunk", 1))),
                        triton.cdiv(target, units)))
    chunk = triton.cdiv(triton.cdiv(span, nsplit), fc["block_n"]) * fc["block_n"]
    nsplit = triton.cdiv(span, chunk)
    part_o = torch.empty((units, nsplit, rows, d), dtype=torch.float32, device=qkv.device)
    part_ml = torch.empty((units, nsplit, rows, 2), dtype=torch.float32, device=qkv.device)
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=qkv.device)
    sb = max(1, min(triton.next_power_of_2(nsplit), 4096 // (rows * d)))
    gs = 16
    ng = triton.cdiv(nsplit, gs) if nsplit > gs else 1
    part2_o = torch.empty((units, ng, rows, d), dtype=torch.float32, device=qkv.device) if ng > 1 else part_o
    part2_ml = torch.empty((units, ng, rows, 2), dtype=torch.float32, device=qkv.device) if ng > 1 else part_ml
    _fused_attn[(batch, hkv // hpb * nrb, nsplit)](
        qkv, data["cache_loc"], cos_sin, data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"],
        page_table, data["seq_lens"], part_o, part_ml, part2_o, part2_ml, _counter(units * (ng + 1), qkv.device), out,
        qkv.stride(0), page_table.stride(0), chunk, cfg["sm_scale"] * cfg["qk_scale_factor"], RMS_EPS,
        HQ=hq, HKV=hkv, D=d, L=length, HPB=hpb, ROWS=rows, LP=max(16, triton.next_power_of_2(length)), PAGE=page,
        BLOCK_N=fc["block_n"], NSPLIT=nsplit, NSP2=triton.next_power_of_2(nsplit), SB=sb, WINDOW=window,
        ROPE=bool(cfg["rope_theta"]), GS=gs, NG=ng, NG2=triton.next_power_of_2(ng), PREFETCH=fc.get("prefetch", False), NRB=nrb, QK8=fc.get("qk8", True), GPRE=True,PS=_DIRECT_PS,PSS=_DIRECT_SS,PWIDTH=_DIRECT_WIDTH,
        num_warps=fc["warps"], num_stages=fc["stages"])
    return out.view(batch, length, hq * d)


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding builds it, cached per theta."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def setup(device):_cos_sin(500000.,64,device)


def attention_pair(pair,data):
 global _DIRECT_PS,_DIRECT_SS,_DIRECT_WIDTH
 p,s=pair;_DIRECT_PS=s if p.dtype==torch.float32 else 0;_DIRECT_SS=p.stride(0);_DIRECT_WIDTH=p.shape[-1]
 c=data['config'];assert c['window'] and c['num_q_heads']==16 and c['page_size']==1
 cs=_cos_sin(c['rope_theta'],64,p.device)
 return _fused(data,c,p[0],c['batch']*c['q_len'],cs)

