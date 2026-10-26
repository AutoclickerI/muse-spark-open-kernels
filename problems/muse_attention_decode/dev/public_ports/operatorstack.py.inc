# Derived from operatorstack/open-kernel-challenge-solutions6d9b77656b6c295e421be9e1424b0f8f99299b0a.
# MIT, Copyright(c)2026 bigboateng. Full notice is shipped in _P5B_PUBLIC_NOTICES.
# Modifications: retain fused paths only; owned S19 ABI, explicit setup; local hook.
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as gcp,mma_v2,mbarrier as gmbar
from triton.experimental.gluon.language.nvidia.hopper import warpgroup_mma,fence_async_shared
RMS_EPS=1e-5
MERGE_ELEMS=tl.constexpr(16384)
_ROPE={};_COUNTERS={};ROPE_POSITIONS=1<<21
_HOPPER_FORCE=False
def _hopper():return _HOPPER_FORCE or torch.cuda.get_device_capability()[0]==9
FUSED_CFG = {
    ("sliding", 1): dict(block_n=128, warps=4, stages=2, max_split=16, per_sm=4, qk8=False,
                         small=dict(warps=8, max_split=32, per_sm=64),
                         big=dict(max_split=8, stages=3), big_min_batch=32),
    ("sliding", 8): dict(block_n=128, warps=4, stages=3, max_split=8, per_sm=4, row_blocks=1,
                         small=dict(block_n=128, stages=2, max_split=16, row_blocks=4, qk8=False), big=dict(max_split=4), big_min_batch=32),
    ("full", 1): dict(block_n=128, warps=4, stages=5, max_split=128, per_sm=2, prefetch=False, min_chunk=256,
                      small=dict(hpb=1, per_sm=4, stages=3, prefetch=True, gluon=False, gate_pre=False), big=dict(per_sm=4),
                      big_min_batch=32, gluon=True, g_block_n=128, g_stages=3, g_warps=4, g_per_sm=4),
    # Full verify (fused: 64 query rows per KV head, wgmma-sized, FP8 QK, 128-key tiles; faster than
    # the FA3 path on the H200 by 6-7%): batch > 4 uses the Gluon verify kernel; batch <= 4 uses the small config.
    ("full", 8): dict(block_n=128, warps=4, stages=3, max_split=8, per_sm=4, prefetch=True, min_chunk=256,
                      row_blocks=1, qk8=True, big=dict(per_sm=8), big_min_batch=32,
                      small=dict(block_n=64, max_split=128, per_sm=8, gluon_v=False),
                      gluon_v=True, gv_block_n=128, gv_stages=4, gv_split=8, gv_pr=24, gv_maxnreg=128),
}

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
def _fused_attn(qkv_ptr, loc_ptr, cos_sin_ptr, ks_ptr, vs_ptr, k_cache, v_cache, pt_ptr, seq_ptr,
                part_o, part_ml, part2_o, part2_ml, cnt_ptr, out_ptr,
                qkv_stride, pt_stride, chunk, sm_scale, eps,
                HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, L: tl.constexpr, HPB: tl.constexpr,
                ROWS: tl.constexpr, LP: tl.constexpr, PAGE: tl.constexpr, BLOCK_N: tl.constexpr,
                NSPLIT: tl.constexpr, NSP2: tl.constexpr, SB: tl.constexpr, WINDOW: tl.constexpr,
                ROPE: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr, NG2: tl.constexpr,
                PREFETCH: tl.constexpr, NRB: tl.constexpr, QK8: tl.constexpr, GPRE: tl.constexpr):
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
    qe = tl.load(qptr, mask=qmask, other=0.0).to(tl.float32)
    qo = tl.load(qptr + 1, mask=qmask, other=0.0).to(tl.float32)
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
        gate_pre = tl.load(qkv_ptr + tok.to(tl.int64)[:, None] * qkv_stride + (HQ + 2 * HKV) * D + qh[:, None] * D
                           + dgt[None, :], mask=rvalid[:, None], other=0.0)

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
        ke = tl.load(kptr, mask=jm[:, None], other=0.0).to(tl.float32)
        ko = tl.load(kptr + 1, mask=jm[:, None], other=0.0).to(tl.float32)
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
        vraw = tl.load(qkv_ptr + ntok.to(tl.int64)[:, None] * qkv_stride + (HQ + HKV + hblk * HPB) * D + dcol[None, :],
                       mask=jm[:, None], other=0.0).to(tl.float32)
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


@gluon.jit
def _g_issue(k_cache, v_cache, k_smem, v_smem, rows, nmask, slot, ROW: gl.constexpr, ld3: gl.constexpr):
    col = gl.arange(0, ROW, layout=gl.SliceLayout(0, gl.SliceLayout(1, ld3)))
    off = rows.to(gl.int64)[:, :, None] * ROW + col[None, None, :]
    m3 = nmask[:, :, None] & (col[None, None, :] < ROW)
    gcp.async_copy_global_to_shared(k_smem.index(slot), k_cache + off, mask=m3)
    gcp.async_copy_global_to_shared(v_smem.index(slot), v_cache + off, mask=m3)
    gcp.commit_group()


@gluon.jit
def _g_head_q(qkv_ptr, qrow, h, eps, G: gl.constexpr, D: gl.constexpr, W: gl.constexpr, bl3: gl.constexpr,
              b8_l: gl.constexpr):
    """Q_h^T [W, D, G] (replicated per warp): RMSNorm, BF16, E4M3 (as _fused_attn, no RoPE on full layers)."""
    qw = gl.arange(0, W, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl3)))
    qd = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl3)))
    qg = gl.arange(0, G, layout=gl.SliceLayout(0, gl.SliceLayout(1, bl3)))
    x = gl.load(qrow + qw[:, None, None] * 0 + (h * G + qg[None, None, :]) * D + qd[None, :, None]).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, axis=1) / D + eps)
    x = (x * inv[:, None, :]).to(gl.bfloat16).to(gl.float32)
    return gl.convert_layout(x.to(gl.float8e4nv), b8_l)


@gluon.jit
def _g_head_loop(k_smem, v_smem, slot, q8, ok, scale, m, l, o, h: gl.constexpr, D: gl.constexpr,
                 W: gl.constexpr, KW: gl.constexpr, G: gl.constexpr, mma3: gl.constexpr, a8_l: gl.constexpr,
                 a16_l: gl.constexpr, b16_l: gl.constexpr):
    kt = k_smem.index(slot)
    vt = v_smem.index(slot)
    sc = mma_v2(kt.slice(h * D, D, dim=2).load(a8_l), q8, gl.zeros([W, KW, G], gl.float32, layout=mma3)) * scale
    sc = gl.where(ok[:, :, None], sc, -float("inf"))
    mn = gl.maximum(m, gl.max(sc, axis=1))
    mu = gl.where(mn == -float("inf"), 0.0, mn)
    p = gl.exp2(sc - mu[:, None, :])
    a = gl.exp2(m - mu)
    l = l * a + gl.sum(p, axis=1)
    v = vt.slice(h * D, D, dim=2).permute([0, 2, 1]).load(a16_l).to(gl.float16)
    o = mma_v2(v, gl.convert_layout(p.to(gl.float16), b16_l), o * a[:, None, :])
    return mn, l, o


@gluon.jit
def _g_head_finish(qkv_ptr, qrow, loc, k_cache, v_cache, k_scale, v_scale, scale, eps, m, l, o, part_o, part_ml,
                   pbase, s, h: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr, D: gl.constexpr,
                   G: gl.constexpr, mma3: gl.constexpr):
    """Merge the warps' states, add the block's new key on split 0 (stored to the cache there), store the
    split's partial for rows h*G .. h*G+G-1."""
    o2_l: gl.constexpr = gl.SliceLayout(0, mma3)
    sg_l: gl.constexpr = gl.SliceLayout(0, o2_l)
    sd_l: gl.constexpr = gl.SliceLayout(1, o2_l)
    mt = gl.max(m, axis=0)
    mz = gl.where(mt == -float("inf"), 0.0, mt)
    w = gl.exp2(m - mz[None, :])
    lt = gl.convert_layout(gl.sum(l * w, axis=0), sg_l)
    ot = gl.sum(o * w[:, None, :], axis=0)
    mt = gl.convert_layout(mt, sg_l)
    dd = gl.arange(0, D, layout=sd_l)
    gg = gl.arange(0, G, layout=sg_l)
    if s == 0:
        kx = gl.load(qrow + (HQ + h) * D + dd).to(gl.float32)
        kinv = gl.rsqrt(gl.sum(kx * kx, axis=0) / D + eps)
        k8 = ((kx * kinv).to(gl.bfloat16).to(gl.float32) / k_scale).to(gl.float8e4nv)
        vx = gl.load(qrow + (HQ + HKV + h) * D + dd).to(gl.float32)
        v8 = (vx / v_scale).to(gl.float8e4nv)
        coff = loc * (HKV * D) + h * D + dd
        gl.store(k_cache + coff, k8)
        gl.store(v_cache + coff, v8)
        qx = gl.load(qrow + (h * G + gg[None, :]) * D + dd[:, None]).to(gl.float32)
        qinv = gl.rsqrt(gl.sum(qx * qx, axis=0) / D + eps)
        qf = (qx * qinv[None, :]).to(gl.bfloat16).to(gl.float32).to(gl.float8e4nv).to(gl.float32)
        sn = gl.sum(qf * k8.to(gl.float32)[:, None], axis=0) * scale
        mn = gl.maximum(mt, sn)
        an = gl.exp2(mt - mn)
        pn = gl.exp2(sn - mn)
        lt = lt * an + pn
        ot = ot * an[None, :] + pn[None, :] * v8.to(gl.float32)[:, None]
        mt = mn
    r = h * G + gg
    gl.store(part_o + (pbase + r[None, :]) * D + dd[:, None], ot)
    gl.store(part_ml + (pbase + r) * 2, mt)
    gl.store(part_ml + (pbase + r) * 2 + 1, lt)


@gluon.jit
def _g_full_decode(qkv_ptr, loc_ptr, ks_ptr, vs_ptr, k_cache, v_cache, pt_ptr, seq_ptr, part_o, part_ml, cnt_ptr,
                   out_ptr, qkv_stride, pt_stride, chunk, sm_scale, eps,
                   HQ: gl.constexpr, HKV: gl.constexpr, D: gl.constexpr, NSPLIT: gl.constexpr,
                   NSP2: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr, W: gl.constexpr):
    G: gl.constexpr = HQ // HKV
    ROW: gl.constexpr = HKV * D
    ROWS: gl.constexpr = HKV * G
    KW: gl.constexpr = BN // W
    LOG2E: gl.constexpr = 1.4426950408889634
    b = gl.program_id(0)
    s = gl.program_id(1)
    seq = gl.load(seq_ptr + b)
    k_scale = gl.load(ks_ptr)
    v_scale = gl.load(vs_ptr)
    scale = sm_scale * k_scale * LOG2E
    start = s * chunk
    end = gl.minimum(start + chunk, seq)
    ntiles = gl.maximum(gl.cdiv(end - start, BN), 0)

    ld3: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 4, 8], [W, 1, 1], [2, 1, 0])
    sm3: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [2, 1, 0])
    mma3: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[W, 1, 1], instr_shape=[1, 16, 8])
    a8_l: gl.constexpr = gl.DotOperandLayout(0, mma3, 4)
    b8_l: gl.constexpr = gl.DotOperandLayout(1, mma3, 4)
    a16_l: gl.constexpr = gl.DotOperandLayout(0, mma3, 2)
    b16_l: gl.constexpr = gl.DotOperandLayout(1, mma3, 2)
    wg_l: gl.constexpr = gl.SliceLayout(1, mma3)
    wk_l: gl.constexpr = gl.SliceLayout(2, mma3)
    bl3: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 4, 8], [W, 1, 1], [2, 1, 0])

    k_smem = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, W, KW, ROW], sm3)
    v_smem = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, W, KW, ROW], sm3)
    qrow = qkv_ptr + b.to(gl.int64) * qkv_stride
    q0 = _g_head_q(qkv_ptr, qrow, 0, eps, G, D, W, bl3, b8_l)
    q1 = _g_head_q(qkv_ptr, qrow, 1, eps, G, D, W, bl3, b8_l)

    pt_row = pt_ptr + b.to(gl.int64) * pt_stride
    lw = gl.arange(0, W, layout=gl.SliceLayout(1, gl.SliceLayout(2, ld3)))
    lk = gl.arange(0, KW, layout=gl.SliceLayout(0, gl.SliceLayout(2, ld3)))
    kidx = lw[:, None] * KW + lk[None, :]
    for i in gl.static_range(STAGES - 1):
        n = start + i * BN + kidx
        nm = n < end
        rows = gl.load(pt_row + n, mask=nm, other=0)
        _g_issue(k_cache, v_cache, k_smem, v_smem, rows, nm, i, ROW, ld3)
    n_nx = start + (STAGES - 1) * BN + kidx
    rows_nx = gl.load(pt_row + n_nx, mask=n_nx < end, other=0)

    sw = gl.arange(0, W, layout=gl.SliceLayout(1, wk_l))
    sk = gl.arange(0, KW, layout=gl.SliceLayout(0, wk_l))
    sidx = sw[:, None] * KW + sk[None, :]
    m0 = gl.full([W, G], -float("inf"), gl.float32, layout=wg_l)
    m1 = gl.full([W, G], -float("inf"), gl.float32, layout=wg_l)
    l0 = gl.zeros([W, G], gl.float32, layout=wg_l)
    l1 = gl.zeros([W, G], gl.float32, layout=wg_l)
    o0 = gl.zeros([W, D, G], gl.float32, layout=mma3)
    o1 = gl.zeros([W, D, G], gl.float32, layout=mma3)
    for t in range(0, ntiles):
        slot = t % STAGES
        gcp.wait_group(STAGES - 2)
        gl.barrier()
        ok = (start + t * BN + sidx) < end
        m0, l0, o0 = _g_head_loop(k_smem, v_smem, slot, q0, ok, scale, m0, l0, o0, 0, D, W, KW, G, mma3, a8_l,
                                  a16_l, b16_l)
        m1, l1, o1 = _g_head_loop(k_smem, v_smem, slot, q1, ok, scale, m1, l1, o1, 1, D, W, KW, G, mma3, a8_l,
                                  a16_l, b16_l)
        tn = t + STAGES - 1
        n = start + tn * BN + kidx
        nm = n < end
        rows = rows_nx
        n2 = n + BN
        rows_nx = gl.load(pt_row + n2, mask=n2 < end, other=0)
        _g_issue(k_cache, v_cache, k_smem, v_smem, rows, nm, tn % STAGES, ROW, ld3)
    gcp.wait_group(0)

    loc = gl.load(loc_ptr + b).to(gl.int64)
    pbase = (b * NSPLIT + s) * ROWS
    _g_head_finish(qkv_ptr, qrow, loc, k_cache, v_cache, k_scale, v_scale, scale, eps, m0, l0, o0, part_o, part_ml,
                   pbase, s, 0, HQ, HKV, D, G, mma3)
    _g_head_finish(qkv_ptr, qrow, loc, k_cache, v_cache, k_scale, v_scale, scale, eps, m1, l1, o1, part_o, part_ml,
                   pbase, s, 1, HQ, HKV, D, G, mma3)

    gl.barrier()
    done = gl.atomic_add(cnt_ptr + b, 1)
    if done == NSPLIT - 1:
        mL: gl.constexpr = gl.BlockedLayout([1, 1], [2, 16], [W, 1], [1, 0])
        oL: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [W, 1], [1, 0])
        si = gl.arange(0, NSP2, layout=gl.SliceLayout(1, mL))
        rr = gl.arange(0, ROWS, layout=gl.SliceLayout(0, mL))
        smask = (si[:, None] < NSPLIT) & (rr[None, :] < ROWS)
        mptr = part_ml + ((b * NSPLIT + si[:, None]) * ROWS + rr[None, :]) * 2
        ms = gl.load(mptr, mask=smask, other=-float("inf"), cache_modifier=".cg")
        ls = gl.load(mptr + 1, mask=smask, other=0.0, cache_modifier=".cg")
        mx = gl.max(ms, axis=0)
        mx = gl.where(mx == -float("inf"), 0.0, mx)
        lsum = gl.sum(gl.exp2(ms - mx[None, :]) * ls, axis=0)
        ro = gl.arange(0, ROWS, layout=gl.SliceLayout(1, oL))
        do = gl.arange(0, D, layout=gl.SliceLayout(0, oL))
        mxo = gl.convert_layout(mx, gl.SliceLayout(1, oL))
        o = gl.zeros([ROWS, D], gl.float32, layout=oL)
        for k in range(0, NSPLIT):
            mj = gl.load(part_ml + ((b * NSPLIT + k) * ROWS + ro) * 2, cache_modifier=".cg")
            oj = gl.load(part_o + ((b * NSPLIT + k) * ROWS + ro[:, None]) * D + do[None, :], cache_modifier=".cg")
            o += gl.exp2(mj - mxo)[:, None] * oj
        lso = gl.convert_layout(lsum, gl.SliceLayout(1, oL))
        o = (o / lso[:, None] * v_scale).to(gl.bfloat16).to(gl.float32)
        o = o * gl.rsqrt(gl.sum(o * o, axis=1) / D + eps)[:, None]
        gate = gl.load(qrow + (HQ + 2 * HKV) * D + ro[:, None] * D + do[None, :]).to(gl.float32)
        o = o * (1.0 / (1.0 + gl.exp(-gate)))
        gl.store(out_ptr + (b.to(gl.int64) * HQ + ro[:, None]) * D + do[None, :], o.to(gl.bfloat16))
        gl.atomic_xchg(cnt_ptr + b, 0)


@gluon.jit
def _gv_softmax_pv(q_s, k_t, v16_s, ok, scale, m, l, o, mma_s: gl.constexpr, mma_o: gl.constexpr,
                   R: gl.constexpr, BN: gl.constexpr):
    sc = warpgroup_mma(q_s, k_t.permute([1, 0]), gl.zeros([R, BN], gl.float32, layout=mma_s)) * scale
    sc = gl.where(ok, sc, -float("inf"))
    mn = gl.maximum(m, gl.max(sc, axis=1))
    mu = gl.where(mn == -float("inf"), 0.0, mn)
    p = gl.exp2(sc - mu[:, None])
    a = gl.exp2(m - mu)
    l = l * a + gl.sum(p, axis=1)
    ao = gl.convert_layout(a, gl.SliceLayout(1, mma_o))
    pa = gl.convert_layout(p.to(gl.float16), gl.DotOperandLayout(0, mma_o, 2))
    o = warpgroup_mma(pa, v16_s, o * ao[:, None])
    return mn, l, o


@gluon.jit
def _gv_producer(pt_row, start, end, ntiles, k_cache, v_cache, k_ring, v_ring, full, empty, h,
                 BN: gl.constexpr, STAGES: gl.constexpr, D: gl.constexpr, ROW: gl.constexpr):
    ld: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    kn = gl.arange(0, BN, layout=gl.SliceLayout(1, ld))
    dq = gl.arange(0, D, layout=gl.SliceLayout(0, ld))
    for t in range(0, ntiles):
        slot = t % STAGES
        ph = (t // STAGES) & 1
        gmbar.wait(empty.index(slot), ph ^ 1)
        n = start + t * BN + kn
        nm = n < end
        rows = gl.load(pt_row + n, mask=nm, other=0).to(gl.int64)
        off = rows[:, None] * ROW + h * D + dq[None, :]
        m2 = nm[:, None] & (dq[None, :] < D)
        gcp.async_copy_global_to_shared(k_ring.index(slot), k_cache + off, mask=m2)
        gcp.async_copy_global_to_shared(v_ring.index(slot), v_cache + off, mask=m2)
        gcp.mbarrier_arrive(full.index(slot), increment_count=False)


@gluon.jit
def _gv_consumer(q_s, k_ring, v_ring, v16_s, full, empty, start, end, ntiles, scale,
                 BN: gl.constexpr, STAGES: gl.constexpr, R: gl.constexpr, D: gl.constexpr):
    blk: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    mma_s: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    mma_o: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, D, 16])
    sn = gl.arange(0, BN, layout=gl.SliceLayout(0, mma_s))
    m = gl.full([R], -float("inf"), gl.float32, layout=gl.SliceLayout(1, mma_s))
    l = gl.zeros([R], gl.float32, layout=gl.SliceLayout(1, mma_s))
    o = gl.zeros([R, D], gl.float32, layout=mma_o)
    for t in range(0, ntiles):
        slot = t % STAGES
        ph = (t // STAGES) & 1
        gmbar.wait(full.index(slot), ph)
        vt = v_ring.index(slot).load(blk)
        v16_s.store(vt.to(gl.float16))
        fence_async_shared()
        gl.barrier()
        ok = ((start + t * BN + sn) < end)[None, :]
        m, l, o = _gv_softmax_pv(q_s, k_ring.index(slot), v16_s, ok, scale, m, l, o, mma_s, mma_o, R, BN)
        gl.barrier()
        gmbar.arrive(empty.index(slot))
    return m, l, o


@gluon.jit
def _g_full_verify(qkv_ptr, loc_ptr, ks_ptr, vs_ptr, k_cache, v_cache, pt_ptr, seq_ptr, part_o, part_ml, cnt_ptr,
                   out_ptr, qkv_stride, pt_stride, chunk, sm_scale, eps,
                   HQ: gl.constexpr, HKV: gl.constexpr, D: gl.constexpr, L: gl.constexpr, NSPLIT: gl.constexpr,
                   NSP2: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr, PR: gl.constexpr):
    G: gl.constexpr = HQ // HKV
    R: gl.constexpr = L * G
    ROW: gl.constexpr = HKV * D
    NP: gl.constexpr = 16                      # the block's new keys, padded to a wgmma K step
    LOG2E: gl.constexpr = 1.4426950408889634
    b = gl.program_id(0)
    s = gl.program_id(1)
    h = gl.program_id(2)
    seq = gl.load(seq_ptr + b)
    k_scale = gl.load(ks_ptr)
    v_scale = gl.load(vs_ptr)
    scale = sm_scale * k_scale * LOG2E
    start = s * chunk
    end = gl.minimum(start + chunk, seq)
    ntiles = gl.maximum(gl.cdiv(end - start, BN), 0)

    blk: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    k_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)
    v_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    mma_s: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    mma_o: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, D, 16])
    mma_n: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, NP, 32])

    # Queries of this KV head: row r = token r // G, query head h * G + r % G; RMSNorm, BF16, E4M3 (as FA3).
    rq = gl.arange(0, R, layout=gl.SliceLayout(1, blk))
    dq = gl.arange(0, D, layout=gl.SliceLayout(0, blk))
    tok = b * L + rq // G
    qh = h * G + rq % G
    qrow = qkv_ptr + tok.to(gl.int64)[:, None] * qkv_stride
    qx = gl.load(qrow + qh[:, None] * D + dq[None, :]).to(gl.float32)
    qinv = gl.rsqrt(gl.sum(qx * qx, axis=1) / D + eps)
    q8 = (qx * qinv[:, None]).to(gl.bfloat16).to(gl.float32).to(gl.float8e4nv)
    q_s = gl.allocate_shared_memory(gl.float8e4nv, [R, D], k_sl, q8)
    k_ring = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, BN, D], k_sl)
    v_ring = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, BN, D], k_sl)
    v16_s = gl.allocate_shared_memory(gl.float16, [BN, D], v_sl)
    full = gmbar.allocate_mbarrier(batch=STAGES)
    empty = gmbar.allocate_mbarrier(batch=STAGES)
    for i in gl.static_range(STAGES):
        gmbar.init(full.index(i), count=128)
        gmbar.init(empty.index(i), count=1)
    fence_async_shared()
    pt_row = pt_ptr + b.to(gl.int64) * pt_stride
    m, l, o = gl.warp_specialize(
        [(_gv_consumer, (q_s, k_ring, v_ring, v16_s, full, empty, start, end, ntiles, scale, BN, STAGES, R, D)),
         (_gv_producer, (pt_row, start, end, ntiles, k_cache, v_cache, k_ring, v_ring, full, empty, h, BN, STAGES,
                         D, ROW))],
        [4], [PR])

    if s == 0:
        # The block's new keys and values: RMSNorm (no RoPE on full layers), stored to the cache as E4M3, and
        # attended causally (other programs read only rows below seq).
        jn = gl.arange(0, NP, layout=gl.SliceLayout(1, blk))
        jm = jn < L
        ntok = b * L + jn
        nrow = qkv_ptr + ntok.to(gl.int64)[:, None] * qkv_stride
        kx = gl.load(nrow + (HQ + h) * D + dq[None, :], mask=jm[:, None], other=0.0).to(gl.float32)
        kinv = gl.rsqrt(gl.sum(kx * kx, axis=1) / D + eps)
        k8 = ((kx * kinv[:, None]).to(gl.bfloat16).to(gl.float32) / k_scale).to(gl.float8e4nv)
        vx = gl.load(nrow + (HQ + HKV + h) * D + dq[None, :], mask=jm[:, None], other=0.0).to(gl.float32)
        v8 = (vx / v_scale).to(gl.float8e4nv)
        loc = gl.load(loc_ptr + ntok, mask=jm, other=0).to(gl.int64)
        coff = loc[:, None] * ROW + h * D + dq[None, :]
        gl.store(k_cache + coff, k8, mask=jm[:, None] & (dq[None, :] < D))
        gl.store(v_cache + coff, v8, mask=jm[:, None] & (dq[None, :] < D))
        kn_s = gl.allocate_shared_memory(gl.float8e4nv, [NP, D], k_sl, k8)
        vn_s = gl.allocate_shared_memory(gl.float16, [NP, D], v_sl, v8.to(gl.float16))
        fence_async_shared()
        rr = gl.arange(0, R, layout=gl.SliceLayout(1, mma_n))
        cc = gl.arange(0, NP, layout=gl.SliceLayout(0, mma_n))
        okn = (cc[None, :] < L) & (cc[None, :] <= (rr // G)[:, None])
        m2 = gl.convert_layout(m, gl.SliceLayout(1, mma_n))
        l2 = gl.convert_layout(l, gl.SliceLayout(1, mma_n))
        mn, ln, o = _gv_softmax_pv(q_s, kn_s, vn_s, okn, scale, m2, l2, o, mma_n, mma_o, R, NP)
        m = gl.convert_layout(mn, gl.SliceLayout(1, mma_s))
        l = gl.convert_layout(ln, gl.SliceLayout(1, mma_s))

    # This split's partial.
    unit = b * HKV + h
    ro = gl.arange(0, R, layout=gl.SliceLayout(1, mma_o))
    do = gl.arange(0, D, layout=gl.SliceLayout(0, mma_o))
    pbase = (unit * NSPLIT + s) * R
    gl.store(part_o + (pbase + ro[:, None]) * D + do[None, :], o)
    r2 = gl.arange(0, R, layout=gl.SliceLayout(1, mma_s))
    gl.store(part_ml + (pbase + r2) * 2, m)
    gl.store(part_ml + (pbase + r2) * 2 + 1, l)

    gl.barrier()
    done = gl.atomic_add(cnt_ptr + unit, 1)
    if done == NSPLIT - 1:
        mL: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
        oL: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
        si = gl.arange(0, NSP2, layout=gl.SliceLayout(1, mL))
        rm = gl.arange(0, R, layout=gl.SliceLayout(0, mL))
        smask = (si[:, None] < NSPLIT) & (rm[None, :] < R)
        mptr = part_ml + ((unit * NSPLIT + si[:, None]) * R + rm[None, :]) * 2
        ms = gl.load(mptr, mask=smask, other=-float("inf"), cache_modifier=".cg")
        ls = gl.load(mptr + 1, mask=smask, other=0.0, cache_modifier=".cg")
        mx = gl.max(ms, axis=0)
        mx = gl.where(mx == -float("inf"), 0.0, mx)
        lsum = gl.sum(gl.exp2(ms - mx[None, :]) * ls, axis=0)
        r3 = gl.arange(0, R, layout=gl.SliceLayout(1, oL))
        d3 = gl.arange(0, D, layout=gl.SliceLayout(0, oL))
        mxo = gl.convert_layout(mx, gl.SliceLayout(1, oL))
        acc = gl.zeros([R, D], gl.float32, layout=oL)
        for k in range(0, NSPLIT):
            mj = gl.load(part_ml + ((unit * NSPLIT + k) * R + r3) * 2, cache_modifier=".cg")
            oj = gl.load(part_o + ((unit * NSPLIT + k) * R + r3[:, None]) * D + d3[None, :], cache_modifier=".cg")
            acc += gl.exp2(mj - mxo)[:, None] * oj
        lso = gl.convert_layout(lsum, gl.SliceLayout(1, oL))
        acc = (acc / lso[:, None] * v_scale).to(gl.bfloat16).to(gl.float32)
        acc = acc * gl.rsqrt(gl.sum(acc * acc, axis=1) / D + eps)[:, None]
        tok3 = b * L + r3 // G
        qh3 = h * G + r3 % G
        grow = qkv_ptr + tok3.to(gl.int64)[:, None] * qkv_stride
        gate = gl.load(grow + (HQ + 2 * HKV) * D + qh3[:, None] * D + d3[None, :]).to(gl.float32)
        acc = acc * (1.0 / (1.0 + gl.exp(-gate)))
        gl.store(out_ptr + (tok3.to(gl.int64)[:, None] * HQ + qh3[:, None]) * D + d3[None, :], acc.to(gl.bfloat16))
        gl.atomic_xchg(cnt_ptr + unit, 0)


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
    if (fc.get("gluon_v") and length == 8 and not window and page == 1 and hq == 16 and hkv == 2 and d == 64
            and _hopper()):   # shape and device-capability choice
        return _gluon_full_verify(data, cfg, qkv, tokens, fc)
    if (fc.get("gluon") and length == 1 and not window and page == 1 and hq == 16 and hkv == 2 and d == 64
            and _hopper()):   # shape and device-capability choice
        return _gluon_full_decode(data, cfg, qkv, tokens, fc)
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
        ROPE=bool(cfg["rope_theta"]), GS=gs, NG=ng, NG2=triton.next_power_of_2(ng), PREFETCH=fc.get("prefetch", False), NRB=nrb, QK8=fc.get("qk8", True), GPRE=fc.get("gate_pre", True),
        num_warps=fc["warps"], num_stages=fc["stages"])
    return out.view(batch, length, hq * d)


def _gluon_full_decode(data, cfg, qkv, tokens, fc):
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page_table = data["page_table"]
    span = page_table.shape[1]
    bn = fc["g_block_n"]
    target = fc["g_per_sm"] * torch.cuda.get_device_properties(qkv.device).multi_processor_count
    nsplit = max(1, min(16, triton.cdiv(span, bn), triton.cdiv(target, batch)))
    chunk = triton.cdiv(triton.cdiv(span, nsplit), bn) * bn
    nsplit = triton.cdiv(span, chunk)
    rows = hq
    part_o = torch.empty((batch, nsplit, rows, d), dtype=torch.float32, device=qkv.device)
    part_ml = torch.empty((batch, nsplit, rows, 2), dtype=torch.float32, device=qkv.device)
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=qkv.device)
    _g_full_decode[(batch, nsplit)](
        qkv, data["cache_loc"], data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"], page_table,
        data["seq_lens"], part_o, part_ml, _counter(batch, qkv.device), out, qkv.stride(0), page_table.stride(0),
        chunk, cfg["sm_scale"] * cfg["qk_scale_factor"], RMS_EPS,
        HQ=hq, HKV=hkv, D=d, NSPLIT=nsplit, NSP2=triton.next_power_of_2(nsplit), BN=bn, STAGES=fc["g_stages"],
        W=fc["g_warps"], num_warps=fc["g_warps"])
    return out.view(batch, 1, hq * d)


def _gluon_full_verify(data, cfg, qkv, tokens, fc):
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page_table = data["page_table"]
    span = page_table.shape[1]
    bn = fc["gv_block_n"]
    nsplit = max(1, min(fc["gv_split"], triton.cdiv(span, bn)))
    chunk = triton.cdiv(triton.cdiv(span, nsplit), bn) * bn
    nsplit = triton.cdiv(span, chunk)
    rows = length * (hq // hkv)
    part_o = torch.empty((batch * hkv, nsplit, rows, d), dtype=torch.float32, device=qkv.device)
    part_ml = torch.empty((batch * hkv, nsplit, rows, 2), dtype=torch.float32, device=qkv.device)
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=qkv.device)
    _g_full_verify[(batch, nsplit, hkv)](
        qkv, data["cache_loc"], data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"], page_table,
        data["seq_lens"], part_o, part_ml, _counter(batch * hkv, qkv.device), out, qkv.stride(0), page_table.stride(0),
        chunk, cfg["sm_scale"] * cfg["qk_scale_factor"], RMS_EPS,
        HQ=hq, HKV=hkv, D=d, L=length, NSPLIT=nsplit, NSP2=triton.next_power_of_2(nsplit), BN=bn,
        STAGES=fc["gv_stages"], PR=fc["gv_pr"], num_warps=4, maxnreg=fc["gv_maxnreg"])
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
def attention(qkv,data):
 c=data['config'];b,l=c['batch'],c['q_len'];qkv=qkv.reshape(b*l,-1)
 cs=_cos_sin(c['rope_theta'],c['head_dim'],qkv.device) if c['rope_theta'] else qkv
 return _fused(data,c,qkv,b*l,cs)
