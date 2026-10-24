import math, time
import torch
from torch.utils.cpp_extension import load_inline
ROW_FACTOR=5.0
ROW_NORM_FLOOR=.1
def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """|a - b| / |b|."""
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def query_precision_bound(e4m3_err: float, factor: float, floor: float, cap: float, slack: float = 0.0) -> float:
    """Attention's bound for one case: `factor` times the reference's own error with queries
    rounded to FP8 E4M3 (`e4m3_err`), plus `slack`, within [floor, cap]."""
    return min(cap, max(floor, factor * e4m3_err + slack))


def compare(output, ref: torch.Tensor, tol: float) -> tuple[bool, str]:
    if not isinstance(output, torch.Tensor):
        return False, f"expected a tensor, got {type(output).__name__}"
    if output.shape != ref.shape or output.dtype != ref.dtype:
        return False, f"expected {tuple(ref.shape)} {ref.dtype}, got {tuple(output.shape)} {output.dtype}"
    out = output.float()
    if not torch.isfinite(out).all():
        return False, "output contains NaN or Inf"
    r = ref.float()
    diff = out - r
    err = float(diff.norm() / r.norm().clamp(min=1e-12))
    if err > tol:
        return False, f"relative L2 error {err:.3e} > {tol:g}"

    rows_diff = diff.reshape(-1, r.shape[-1]).norm(dim=-1)
    rows_ref = r.reshape(-1, r.shape[-1]).norm(dim=-1)
    floor = (ROW_NORM_FLOOR * rows_ref.mean()).clamp(min=1e-12)
    row_err = rows_diff / torch.maximum(rows_ref, floor)
    worst = int(row_err.argmax())
    worst_err = float(row_err[worst])
    if worst_err > ROW_FACTOR * tol:
        index = tuple(int(i) for i in torch.unravel_index(torch.tensor(worst), r.shape[:-1])) if r.dim() > 1 else ()
        return False, (f"relative L2 error {err:.3e}, but row {index} is off by {worst_err:.3e}"
                       f" > {ROW_FACTOR:g} x {tol:g}")
    return True, f"relative L2 error {err:.3e} (worst row {worst_err:.3e})"

HEAD_DIM = 64

HEADS = {"tp8": (16, 2), "dp": (128, 16)}  # (query heads, key/value heads) on one GPU

SLIDING_WINDOW = 2048

ROPE_THETA = 500_000.0

QK_SCALE_FACTOR = 5.4730064863875

SM_SCALE = HEAD_DIM ** -0.5

RMS_EPS = 1e-5

PAGE = 64

FP8 = torch.float8_e4m3fn

K_SCALE = V_SCALE = 1.0

REL_L2_TOL = 0.15

FP8_QUERY_FACTOR = 2.5

FP8_QUERY_FLOOR = 0.03

FP8_QUERY_SLACK = 0.015

QUERY_DTYPE = torch.bfloat16

CACHE_TOL = 8e-2

SOFT_C2 = {"sliding": (0.3, 0.75), "full": (0.5, 0.62)}

COHERENT_HEAD = 2048

COHERENT_ALPHA = 1.0

def _coherent_tail(v_cache, k_cache, q, seq_lens, page_table, layer, seed, page, scale):
    """Add the shared direction to the values of each request's low-attention keys."""
    hkv, hq = v_cache.shape[-2], q.shape[-2]
    group = hq // hkv
    tail_gen = torch.Generator(device=v_cache.device).manual_seed(seed + 0xC0DE)
    u = torch.randn((hkv, HEAD_DIM), generator=tail_gen, device=v_cache.device)
    u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
    qn = _rms(q) * QK_SCALE_FACTOR  # [B, L, Hq, 64], as the reference attends (RoPE on sliding)
    if layer == "sliding":
        pos = seq_lens.long()[:, None] + torch.arange(q.shape[1], device=q.device)
        qn = _rope(qn, pos, ROPE_THETA)
    k_flat = k_cache.view(-1, hkv, HEAD_DIM)
    v_flat = v_cache.view(-1, hkv, HEAD_DIM)
    for b in range(len(seq_lens)):
        seq = int(seq_lens[b])
        first = max(0, seq - SLIDING_WINDOW + 1) if layer == "sliding" else 0
        past = torch.arange(first, seq, device=q.device)
        rows = page_table[b].long()[past // page] * page + past % page
        keys = _dequant(k_flat[rows], scale)  # [S, Hkv, 64]
        # Each key's best logit over its KV head's query heads and the block's tokens.
        logits = torch.einsum("lkgd,skd->kgls", qn[b].view(-1, hkv, group, HEAD_DIM), keys)
        best = logits.flatten(1, 2).amax(1)  # [Hkv, S]
        if best.shape[1] <= COHERENT_HEAD:
            continue
        tail = best < best.topk(COHERENT_HEAD, -1).values[:, -1:]  # [Hkv, S]
        vals = _dequant(v_flat[rows], scale) + COHERENT_ALPHA * tail.T[..., None] * u
        v_flat[rows] = _quant(vals, scale)

AGENTIC_SPREAD = 0.075

def _longest(ctx: int, spread: str) -> int:
    """The longest length _lengths can draw."""
    if spread == "fixed":
        return ctx
    if spread == "agentic":
        return math.ceil(ctx * (1 + AGENTIC_SPREAD))
    return 4 * ctx

def _lengths(gen, batch: int, ctx: int, spread: str, device) -> torch.Tensor:
    if spread == "fixed":
        return torch.full((batch,), ctx, dtype=torch.int64, device=device)
    if spread == "agentic":
        u = torch.rand((batch,), generator=gen, device=device)
        return (ctx * (1 + AGENTIC_SPREAD * (2 * u - 1))).long().clamp(1, _longest(ctx, spread))
    # Log-normal around ctx: many short requests, a long tail.
    z = torch.randn((batch,), generator=gen, device=device)
    return (ctx * torch.exp(z - 0.5)).clamp(16, 4 * ctx).long()

def _shortest_share(spread: str) -> float:
    """A lower bound on mean length / ctx, for the speed-of-light bound."""
    return {"fixed": 1.0, "agentic": 1 - AGENTIC_SPREAD}.get(spread, 0.5)

def _quant(x, scale):
    return (x.float() / scale).to(FP8)

def _dequant(x, scale):
    return x.float() * scale

CACHE_CHUNK = 1 << 20  # cache rows generated at a time

def _fill_cache(gen, shape, device, scale=K_SCALE, seed=0, dist="normal", layer="full"):
    """FP8 key and value caches of `shape`: N(0, 1) entries, soft keys for dist="soft" and
    "coherent" (see SOFT_C2)."""
    hkv = shape[-2]
    k_cache = torch.empty(shape, dtype=FP8, device=device)
    v_cache = torch.empty(shape, dtype=FP8, device=device)
    mix = None
    if dist in ("soft", "coherent"):
        # Its own generator, so the other inputs match the same seed's normal case.
        soft = torch.Generator(device=device).manual_seed(seed + 0x50F7)
        u = torch.randn((hkv, HEAD_DIM), generator=soft, device=device)
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
        lo, hi = SOFT_C2[layer]
        c2 = lo + (hi - lo) * torch.rand((hkv, 1), generator=soft, device=device)
        mix = (c2.sqrt() * u, (1 - c2).sqrt())
    for dst, keys in ((k_cache, True), (v_cache, False)):
        flat = dst.view(-1, hkv, HEAD_DIM)
        for at in range(0, flat.shape[0], CACHE_CHUNK):
            n = min(CACHE_CHUNK, flat.shape[0] - at)
            x = torch.randn((n, hkv, HEAD_DIM), generator=gen, device=device)
            if keys and mix is not None:
                x = mix[0] + mix[1] * x
            flat[at:at + n] = _quant(x, scale)
    return k_cache, v_cache

def generate_input(batch: int, ctx: int, heads: str, layer: str, seed: int, q_len: int = 1,
                   spread: str = "fixed", dist: str = "normal", page: int = PAGE,
                   kv_scale: float = K_SCALE, device: str = "cuda", **_):
    """layer: "sliding" (2,048-token window, RoPE) or "full" (every cached token, no RoPE).
    q_len: 1 for a decode step, 8 for a speculative-decoding verify block (causal inside it).
    dist: "normal", "soft" (see SOFT_C2), "coherent" (see COHERENT_HEAD), or "edges": keys
    aligned with each KV head's queries at the window edges, relative to the block's first
    token (sliding: the oldest position inside the window and the newest just outside it;
    full: position 0)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    hq, hkv = HEADS[heads]
    seq_lens = _lengths(gen, batch, ctx, spread, device)
    pages_per = (seq_lens + q_len + page - 1) // page  # room for the new tokens
    # The pool and page table are sized for the longest possible length, so every
    # repetition's inputs have the same shapes (one CUDA graph is replayed on fresh inputs).
    longest = _longest(ctx, spread)
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    # Physical pages are scattered.
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]

    # Cached keys are post-norm (unit RMS per head); values are projections, scale ~1.
    shape = (total, page, hkv, HEAD_DIM)
    k_cache, v_cache = _fill_cache(gen, shape, device, kv_scale, seed, dist, layer)
    q = (2 * torch.randn((batch, q_len, hq, HEAD_DIM), generator=gen, device=device)).to(torch.bfloat16)
    if dist == "coherent":
        _coherent_tail(v_cache, k_cache, q, seq_lens, page_table, layer, seed, page, kv_scale)
    if dist == "edges":
        # Each KV head's key points along its group's first query head of the block's first
        # token, as the kernel will see it (normalized, then RoPE on sliding layers).
        qn = _rms(q[:, 0])
        if layer == "sliding":
            qn = _rope(qn, seq_lens, ROPE_THETA)
        planted = _quant(qn[:, :: hq // hkv], kv_scale)  # [B, Hkv, 64]
        k_flat = k_cache.view(-1, hkv, HEAD_DIM)
        for b in range(batch):
            p = int(seq_lens[b])
            spots = [p - SLIDING_WINDOW + 1, p - SLIDING_WINDOW] if layer == "sliding" else [0]
            for t in spots:
                if 0 <= t < p:
                    k_flat[int(page_table[b, t // page]) * page + t % page] = planted[b]
    k_new = (2 * torch.randn((batch, q_len, hkv, HEAD_DIM), generator=gen, device=device)).to(torch.bfloat16)
    v_new = torch.randn((batch, q_len, hkv, HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)
    gate = torch.randn((batch, q_len, hq * HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)
    # Fused projection output, per token: q | k | v | gate.
    qkv = torch.cat([q.flatten(2), k_new.flatten(2), v_new.flatten(2), gate], -1)
    # Step metadata: each new token's position and flat cache row, per-request cache lengths
    # (including the new tokens) and query offsets, and on full-attention decode steps FA3's
    # tile schedule.
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=device)
    return {
        "qkv": qkv,
        "positions": positions,
        "cache_loc": cache_loc,
        "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q,
        "scheduler_metadata": _fa3_scheduler_metadata(cache_seqlens, cu_seqlens_q, hq, hkv, page, q_len, layer),
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "page_table": page_table,
        "seq_lens": seq_lens.to(torch.int32),
        "config": {
            "batch": batch,
            "q_len": q_len,
            "num_q_heads": hq,
            "num_kv_heads": hkv,
            "head_dim": HEAD_DIM,
            "page_size": page,
            "window": SLIDING_WINDOW if layer == "sliding" else 0,
            "rope_theta": ROPE_THETA if layer == "sliding" else 0.0,
            "qk_scale_factor": QK_SCALE_FACTOR,
            "sm_scale": SM_SCALE,
            "kv_scale": kv_scale,
        },
    }

def split_qkv(data):
    """q [B, L, Hq, 64], k and v [B, L, Hkv, 64], gate [B, L, Hq*64] from the fused input."""
    cfg = data["config"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    batch, length = cfg["batch"], cfg["q_len"]
    q, k, v, gate = data["qkv"].split([hq * d, hkv * d, hkv * d, hq * d], -1)
    return (q.reshape(batch, length, hq, d), k.reshape(batch, length, hkv, d), v.reshape(batch, length, hkv, d),
            gate)

def _rms(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def _rope(x, pos, theta):
    """Interleaved (GPT-J) RoPE: pairs (2i, 2i+1) rotate by pos * theta^(-2i/64).
    x [..., heads, 64] with pos broadcast over everything but the last two dims."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=x.device).float() / HEAD_DIM))
    ang = pos.float()[..., None] * inv_freq
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)

def _rows(data, positions):
    """Flat cache rows of [B, L] positions."""
    size = data["config"]["page_size"]
    page = torch.gather(data["page_table"].long(), 1, positions // size)
    return page * size + positions % size

def new_slots(data) -> torch.Tensor:
    """[B, L] flat cache rows where the block's new tokens go."""
    pos = data["seq_lens"].long()[:, None] + torch.arange(data["config"]["q_len"], device=data["seq_lens"].device)
    return _rows(data, pos)

def ref_kernel(data, query_dtype=None, new_kv=None):
    """Returns (output [B, L, Hq*64], new keys and values [B, L, Hkv, 64] as stored: FP8).
    query_dtype: the precision queries are rounded to (default QUERY_DTYPE). With new_kv
    (FP8 keys and values [B, L, Hkv, 64]), the block attends those as its own entries."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    q_in, k_in, v_in, gate = split_qkv(data)
    pos = data["seq_lens"].long()[:, None] + torch.arange(length, device=q_in.device)
    q = _rms(q_in) * cfg["qk_scale_factor"]
    k = _rms(k_in)
    if cfg["rope_theta"]:
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    # FP8 rounds the normalized query, before qk_scale_factor.
    query_dtype = query_dtype or QUERY_DTYPE
    if query_dtype == torch.bfloat16:
        q = q.to(torch.bfloat16).float()
    else:
        q = (q / cfg["qk_scale_factor"]).to(query_dtype).float() * cfg["qk_scale_factor"]
    k_new, v_new = _quant(k, data["k_scale"]), _quant(v_in, data["v_scale"])
    if new_kv is not None:
        k_new, v_new = new_kv
    k_flat = data["k_cache"].view(-1, hkv, HEAD_DIM)
    v_flat = data["v_cache"].view(-1, hkv, HEAD_DIM)
    group = hq // hkv

    out = torch.empty((batch, length, hq, HEAD_DIM), device=q.device)
    for b in range(batch):
        seq = int(data["seq_lens"][b])
        past = torch.arange(seq, device=q.device)
        size = cfg["page_size"]
        past_rows = data["page_table"][b].long()[past // size] * size + past % size
        keys_past = _dequant(k_flat[past_rows], data["k_scale"])
        vals_past = _dequant(v_flat[past_rows], data["v_scale"])
        keys = torch.cat([keys_past, _dequant(k_new[b], data["k_scale"])])  # positions 0 .. seq+L-1
        vals = torch.cat([vals_past, _dequant(v_new[b], data["v_scale"])])
        for j in range(length):
            p = seq + j
            start = max(0, p - cfg["window"] + 1) if cfg["window"] else 0
            kb, vb = keys[start:p + 1], vals[start:p + 1]
            att = torch.einsum("kgd,tkd->kgt", q[b, j].view(hkv, group, HEAD_DIM), kb) * cfg["sm_scale"]
            out[b, j] = torch.einsum("kgt,tkd->kgd", att.softmax(-1), vb).reshape(hq, HEAD_DIM)
    out = _rms(out) * torch.sigmoid(gate.float()).view(batch, length, hq, HEAD_DIM)
    return out.reshape(batch, length, hq * HEAD_DIM), k_new, v_new

def collect(data, output) -> torch.Tensor:
    """Runs in the submission's process after a step: the output followed by the new cache
    entries, as raw bytes (see unpack)."""
    cfg = data["config"]
    shape = (cfg["batch"], cfg["q_len"], cfg["num_q_heads"] * HEAD_DIM)
    if type(output) is not torch.Tensor or output.dtype != torch.bfloat16 or tuple(output.shape) != shape:
        raise ValueError(f"expected a bfloat16 tensor of shape {shape}")
    slots = new_slots(data)
    hkv = cfg["num_kv_heads"]
    k = data["k_cache"].view(-1, hkv, HEAD_DIM)[slots]
    v = data["v_cache"].view(-1, hkv, HEAD_DIM)[slots]
    return torch.cat([p.contiguous().view(torch.uint8).reshape(-1) for p in (output, k, v)])

def unpack(raw: torch.Tensor, cfg: dict):
    b, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    shapes = [((b, length, hq * HEAD_DIM), torch.bfloat16), ((b, length, hkv, HEAD_DIM), FP8),
              ((b, length, hkv, HEAD_DIM), FP8)]
    sizes = [math.prod(shape) * dtype.itemsize for shape, dtype in shapes]
    if raw.dtype != torch.uint8 or raw.numel() != sum(sizes):
        return None
    out, at = [], 0
    for (shape, dtype), n in zip(shapes, sizes):
        out.append(raw[at:at + n].view(dtype).view(shape))
        at += n
    return out

def check_implementation(data, raw):
    parts = unpack(raw, data["config"])
    if parts is None:
        return False, "malformed output"
    out, k, v = parts
    _, ref_k, ref_v = ref_kernel(data)
    for name, got, want, scale in (("key", k, ref_k, data["k_scale"]), ("value", v, ref_v, data["v_scale"])):
        ok, kv_msg = compare(_dequant(got, scale), _dequant(want, scale), CACHE_TOL)
        if not ok:
            return False, f"new {name} cache entries: {kv_msg}"
    # The output is checked against the reference attending the submission's own new cache
    # entries (checked above), since FP8 rounding of those entries varies by implementation.
    ref = ref_kernel(data, new_kv=(k, v))[0].to(torch.bfloat16)
    e4m3 = relative_l2(ref_kernel(data, torch.float8_e4m3fn, new_kv=(k, v))[0].to(torch.bfloat16), ref)
    tol = query_precision_bound(e4m3, FP8_QUERY_FACTOR, FP8_QUERY_FLOOR, REL_L2_TOL, FP8_QUERY_SLACK)
    ok, msg = compare(out, ref, tol)
    msg = (f"{msg} [bound {tol:.3f}: {FP8_QUERY_FACTOR:g} x the reference's {e4m3:.3f} with FP8 E4M3 queries"
           f" + {FP8_QUERY_SLACK:g}]")
    return ok, msg


def _fa3_scheduler_metadata(*args,**kwargs): return None

# The gate above is frozen from the task; this caches its three FP32 passes
# per input only inside the untimed lab. Timed kernels never read these values.
def gate_prepare(data):
    c=data['config'];slots=new_slots(data);hk=c['num_kv_heads']
    actual=tuple(data[name].view(-1,hk,64)[slots].clone() for name in ('k_cache','v_cache'))
    _,want_k,want_v=ref_kernel(data)
    want=ref_kernel(data,new_kv=actual)[0].to(torch.bfloat16)
    fp8=ref_kernel(data,torch.float8_e4m3fn,new_kv=actual)[0].to(torch.bfloat16)
    tol=query_precision_bound(relative_l2(fp8,want),FP8_QUERY_FACTOR,FP8_QUERY_FLOOR,REL_L2_TOL,FP8_QUERY_SLACK)
    return dict(out=want,tol=tol,kv=(want_k,want_v),slots=slots)


def gate_cached(data,out,expected):
    c=data['config'];hk=c['num_kv_heads']
    for i,name in enumerate(('k_cache','v_cache')):
        actual=data[name].view(-1,hk,64)[expected['slots']]
        scale=data['k_scale' if i==0 else 'v_scale']
        ok,msg=compare(_dequant(actual,scale),_dequant(expected['kv'][i],scale),CACHE_TOL)
        if not ok:return False,'cache '+msg,float('inf')
    out=out.view_as(expected['out'])
    ok,msg=compare(out,expected['out'],expected['tol'])
    return ok,msg,relative_l2(out,expected['out'])


def timed(fn,data,flush,rounds=5):
    fn(data);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out=fn(data)
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    values=[]
    for _ in range(rounds):
        flush.zero_();start.record();graph.replay();end.record();end.synchronize()
        values.append(start.elapsed_time(end)*1000.)
    return sorted(values)[len(values)//2],out,graph

import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
HQ,HKV,HD=16,2,64
Q_SIZE=1024
PDL=True
@triton.jit
def _pdl_start(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_wait(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_wait()


def _launch(kernel, grid, *args, use_pdl=None, **kw):
    enabled = PDL if use_pdl is None else bool(use_pdl)
    if enabled:
        kw["launch_pdl"] = True
    return kernel[grid](*args, USE_PDL=enabled, **kw)


@triton.jit
def _qk_prep_kernel(P, S: tl.constexpr, ps_stride, POS, LOC, KC, VC, KSCALE, VSCALE, INV_FREQ,
                    QOUT, GATE, q_mult, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                    ROPE: tl.constexpr, Q_FP8: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    _pdl_start(USE_PDL)
    cache_row=tl.load(LOC+t)
    kscale=tl.load(KSCALE);vscale=tl.load(VSCALE)
    if ROPE:
        half=tl.arange(0,D//2)
        angle=tl.load(POS+t).to(tl.float32)*tl.load(INV_FREQ+half)
        rope_cos=tl.cos(angle)[None,:];rope_sin=tl.sin(angle)[None,:]
    _pdl_wait(USE_PDL)
    W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    base = P + t * W
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    d = tl.arange(0, D)
    qo = hq[:, None] * D + d[None, :]
    ko = HQ * D + hk[:, None] * D + d[None, :]
    vo = (HQ + HKV) * D + hk[:, None] * D + d[None, :]
    go = (HQ + 2 * HKV) * D + qo
    q = tl.zeros((HQ, D), tl.float32)
    k = tl.zeros((HKV, D), tl.float32)
    v = tl.zeros((HKV, D), tl.float32)
    g = tl.zeros((HQ, D), tl.float32)
    for s in tl.static_range(S):
        q += tl.load(base + s * ps_stride + qo).to(tl.float32)
        k += tl.load(base + s * ps_stride + ko).to(tl.float32)
        v += tl.load(base + s * ps_stride + vo).to(tl.float32)
        g += tl.load(base + s * ps_stride + go).to(tl.float32)
    tl.store(GATE + t * (HQ * D) + qo, g)
    q = q * tl.rsqrt(tl.sum(q * q, 1) / D + eps)[:, None]
    k = k * tl.rsqrt(tl.sum(k * k, 1) / D + eps)[:, None]
    if ROPE:
        c=rope_cos;sn=rope_sin
        qe, qd = tl.split(tl.reshape(q, (HQ, D // 2, 2)))
        q = tl.reshape(tl.join(qe * c - qd * sn, qe * sn + qd * c), (HQ, D))
        ke, kd = tl.split(tl.reshape(k, (HKV, D // 2, 2)))
        k = tl.reshape(tl.join(ke * c - kd * sn, ke * sn + kd * c), (HKV, D))
    row = cache_row
    kv_off = row * (HKV * D) + hk[:, None] * D + d[None, :]
    tl.store(KC + kv_off, (k / kscale).to(tl.float8e4nv))
    tl.store(VC + kv_off, (v / vscale).to(tl.float8e4nv))
    if Q_FP8:
        tl.store(QOUT + t * (HQ * D) + qo, q.to(tl.float8e4nv))
    else:
        tl.store(QOUT + t * (HQ * D) + qo, (q * q_mult).to(tl.bfloat16))


def qk_prep(qkv, data, state, q_fp8: bool):
    P, S = qkv
    cfg = data["config"]
    tokens = cfg["tokens"]
    q = torch.empty((tokens, HQ, HD), device=P.device, dtype=FP8 if q_fp8 else torch.bfloat16)
    gate = torch.empty((tokens, Q_SIZE), device=P.device, dtype=torch.float32)
    _launch(_qk_prep_kernel, (tokens,),
            P, S, P.stride(0), data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"],
            data["k_scale"], data["v_scale"], state["inv_freq"], q, gate, cfg["qk_scale_factor"], RMS_EPS,
            HQ=HQ, HKV=HKV, D=HD, ROPE=bool(cfg["rope_theta"]), Q_FP8=q_fp8, num_warps=4)
    return q, gate


@triton.jit
def _norm_gate_kernel(X, GATE, OUT, eps, HQ: tl.constexpr, D: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    _pdl_start(USE_PDL)
    h = tl.arange(0, HQ)
    d = tl.arange(0, D)
    off = h[:, None] * D + d[None, :]
    _pdl_wait(USE_PDL)
    o = tl.load(X + t * (HQ * D) + off).to(tl.float32)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    g = tl.load(GATE + t * (HQ * D) + off)
    tl.store(OUT + t * (HQ * D) + off, (o * tl.sigmoid(g)).to(tl.bfloat16))


def _attention_fa3_public(q, gate, data, state):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    cfg = data['config']
    batch, L, tokens = (cfg['batch'], cfg['q_len'], cfg['tokens'])
    window = cfg['window']
    descale = {}
    if cfg['kv_scale'] != 1.0:
        descale = dict(k_descale=data['k_scale'].reshape(1, 1).expand(batch, HKV), v_descale=data['v_scale'].reshape(1, 1).expand(batch, HKV))
    out = flash_attn_with_kvcache(q=q, k_cache=data['k_cache'], v_cache=data['v_cache'], page_table=data['page_table'], cache_seqlens=data['cache_seqlens'], cu_seqlens_q=data['cu_seqlens_q'], max_seqlen_q=L, softmax_scale=cfg['sm_scale'] * cfg['qk_scale_factor'], causal=True, window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data['scheduler_metadata'], **descale)
    if isinstance(out, tuple):
        out = out[0]
    o = torch.empty((tokens, Q_SIZE), device=q.device, dtype=torch.bfloat16)
    _launch(_norm_gate_kernel, (tokens,), out, gate, o, RMS_EPS, HQ=HQ, D=HD, num_warps=4)
    return o


@triton.jit
def _best7_old_qk_prep_kernel(P, S: tl.constexpr, ps_stride, POS, LOC, KC, VC, KSCALE, VSCALE, INV_FREQ, QOUT, GATE, q_mult, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr, Q_FP8: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    _pdl_start(USE_PDL)
    _pdl_wait(USE_PDL)
    W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    base = P + t * W
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    d = tl.arange(0, D)
    qo = hq[:, None] * D + d[None, :]
    ko = HQ * D + hk[:, None] * D + d[None, :]
    vo = (HQ + HKV) * D + hk[:, None] * D + d[None, :]
    go = (HQ + 2 * HKV) * D + qo
    q = tl.zeros((HQ, D), tl.float32)
    k = tl.zeros((HKV, D), tl.float32)
    v = tl.zeros((HKV, D), tl.float32)
    g = tl.zeros((HQ, D), tl.float32)
    for s in tl.static_range(S):
        q += tl.load(base + s * ps_stride + qo).to(tl.float32)
        k += tl.load(base + s * ps_stride + ko).to(tl.float32)
        v += tl.load(base + s * ps_stride + vo).to(tl.float32)
        g += tl.load(base + s * ps_stride + go).to(tl.float32)
    tl.store(GATE + t * (HQ * D) + qo, g)
    q = q * tl.rsqrt(tl.sum(q * q, 1) / D + eps)[:, None]
    k = k * tl.rsqrt(tl.sum(k * k, 1) / D + eps)[:, None]
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        half = tl.arange(0, D // 2)
        ang = pos * tl.load(INV_FREQ + half)
        c = tl.cos(ang)[None, :]
        sn = tl.sin(ang)[None, :]
        qe, qd = tl.split(tl.reshape(q, (HQ, D // 2, 2)))
        q = tl.reshape(tl.join(qe * c - qd * sn, qe * sn + qd * c), (HQ, D))
        ke, kd = tl.split(tl.reshape(k, (HKV, D // 2, 2)))
        k = tl.reshape(tl.join(ke * c - kd * sn, ke * sn + kd * c), (HKV, D))
    row = tl.load(LOC + t)
    kv_off = row * (HKV * D) + hk[:, None] * D + d[None, :]
    tl.store(KC + kv_off, (k / tl.load(KSCALE)).to(tl.float8e4nv))
    tl.store(VC + kv_off, (v / tl.load(VSCALE)).to(tl.float8e4nv))
    if Q_FP8:
        tl.store(QOUT + t * (HQ * D) + qo, q.to(tl.float8e4nv))
    else:
        tl.store(QOUT + t * (HQ * D) + qo, (q * q_mult).to(tl.bfloat16))


def _best7_old_qk_prep(qkv, data, state, q_fp8: bool):
    P, S = qkv
    cfg = data['config']
    tokens = cfg['tokens']
    q = torch.empty((tokens, HQ, HD), device=P.device, dtype=FP8 if q_fp8 else torch.bfloat16)
    gate = torch.empty((tokens, Q_SIZE), device=P.device, dtype=torch.float32)
    _launch(_best7_old_qk_prep_kernel, (tokens,), P, S, P.stride(0), data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], state['inv_freq'], q, gate, cfg['qk_scale_factor'], RMS_EPS, HQ=HQ, HKV=HKV, D=HD, ROPE=bool(cfg['rope_theta']), Q_FP8=q_fp8, num_warps=4)
    return (q, gate)


PHASE_CUDA='#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <cstdint>\n#include <stdexcept>\n#include <vector>\n#define DI __device__ __forceinline__\nnamespace {\nconstexpr int D=64, ROWS=64, PR=68, THREADS=128;\nstruct VP {\n const __nv_bfloat16* qkv; const int64_t* loc; const int* pt; const int* seq;\n uint8_t* kc; uint8_t* vc; const float* ks; const float* vs;\n float* partial; __nv_bfloat16* out;\n int B,HQ,HK,stride,pages,ps,lg,ns,proj_splits; float scale;\n unsigned long long* clocks;\n};\nDI uint32_t su(const void* x){return (uint32_t)__cvta_generic_to_shared(x);}\nDI void cp(void* dst,const void* src,bool valid){\n asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(su(dst)),"l"(src),"r"(valid?16:0):"memory");\n}\nDI void commit(){asm volatile("cp.async.commit_group;":::"memory");}\ntemplate<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}\nDI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}\nDI float2 exp2pair(float x,float y){\n __half2 h=__floats2half2_rn(x,y);uint32_t bits=*(uint32_t*)&h,out;\n asm("ex2.approx.f16x2 %0,%1;":"=r"(out):"r"(bits));\n return __half22float2(*(__half2*)&out);\n}\nDI float wsum(float x){\n#pragma unroll\n for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);\n return x;\n}\nDI uint16_t fp8pair(float x,float y){return __nv_cvt_float2_to_fp8x2(make_float2(x,y),__NV_SATFINITE,__NV_E4M3);}\nDI int off(int r,int c){return r*64+(((c>>4)^(r&3))*16)+(c&15);}\nDI uint32_t word(const uint8_t* p,int r,int c){return *(const uint32_t*)(p+off(r,c));}\nDI float2 input_pair(const VP& p,int token,int col){\n if(!p.proj_splits)return __bfloat1622float2(*(const __nv_bfloat162*)(p.qkv+(int64_t)token*p.stride+col));\n const float* raw=(const float*)p.qkv;float2 sum=make_float2(0,0);\n for(int s=0;s<p.proj_splits;s++){\n  const float2 x=*(const float2*)(raw+((int64_t)s*p.B*8+token)*p.stride+col);\n  sum.x+=x.x;sum.y+=x.y;\n }\n return __bfloat1622float2(__floats2bfloat162_rn(sum.x,sum.y));\n}\nDI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){\n asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "\n "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n}\nstruct State {\n uint32_t q[2][4];\n float o[8][4];\n float m[2],l[2];\n};\n// Convert an 8-key QK accumulator panel into two rows of FP8 pairs. No shared store.\nDI uint32_t probpack(float (&s)[4]){\n return (uint32_t)fp8pair(s[0]*256.f,s[1]*256.f)|((uint32_t)fp8pair(s[2]*256.f,s[3]*256.f)<<16);\n}\n// One CTA transposes V once per stage, amortized over the four query warps.\n// Four lanes transpose each 4x4 byte panel using two PRMT steps.\ntemplate<int N>\nDI void transpose_v(const uint8_t* V,uint8_t* VT){\n const int tid=threadIdx.x,t=tid&3;\n#pragma unroll\n for(int i=0;i<N/8;i++){\n  const int group=tid/4+i*32;\n  const int key=(group/16)*4,dim=(group%16)*4;\n  const uint32_t x=word(V,key+t,dim);\n  const uint32_t a=__shfl_sync(0xffffffff,x,(threadIdx.x&28));\n  const uint32_t b=__shfl_sync(0xffffffff,x,(threadIdx.x&28)+1);\n  const uint32_t c=__shfl_sync(0xffffffff,x,(threadIdx.x&28)+2);\n  const uint32_t d=__shfl_sync(0xffffffff,x,(threadIdx.x&28)+3);\n  const uint32_t ab=__byte_perm(a,b,t*0x11+0x40);\n  const uint32_t cd=__byte_perm(c,d,t*0x11+0x40);\n  const uint32_t out=__byte_perm(ab,cd,0x5410);\n  *(uint32_t*)(VT+(dim+t)*N+((key^((((dim+t)&7)*16)&(N-1)))))=out;\n }\n}\nstruct Trace { unsigned long long c[7]; };\nDI unsigned long long tick(){unsigned long long x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}\ntemplate<int N,bool MASK,int VN,bool CLOCK>\nDI void tile(State& st,const uint8_t* K,const uint8_t* V,int kp,int hi,int seq,float scale,int vkey,Trace& trace){\n unsigned long long ts=0;if constexpr(CLOCK)ts=tick();\n constexpr int NT=N/8;\n const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,warp=threadIdx.x>>5;\n float s[NT][4];\n#pragma unroll\n for(int n=0;n<NT;n++){\n#pragma unroll\n  for(int i=0;i<4;i++)s[n][i]=0;\n#pragma unroll\n  for(int k=0;k<2;k++){\n   const uint32_t b0=word(K,n*8+g,k*32+t*4);\n   const uint32_t b1=word(K,n*8+g,k*32+16+t*4);\n   mma8(s[n],st.q[k],b0,b1);\n  }\n }\n if constexpr(CLOCK){const auto now=tick();trace.c[3]+=now-ts;ts=now;}\n // Four lanes jointly own a query row. Both online reductions and P stay in registers.\n#pragma unroll\n for(int rh=0;rh<2;rh++){\n  float mx=-INFINITY;\n#pragma unroll\n  for(int n=0;n<NT;n++){\n#pragma unroll\n   for(int e=0;e<2;e++){\n    float x=s[n][2*rh+e]*scale;\n    if constexpr(MASK){\n     const int key=kp+8*n+2*t+e;\n     if(key>=hi || key>seq+2*warp+rh)x=-INFINITY;\n    }\n    s[n][2*rh+e]=x;mx=fmaxf(mx,x);\n   }\n  }\n  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));\n  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));\n  const float nm=fmaxf(st.m[rh],mx),use=nm==-INFINITY?0.f:nm;\n  const float alpha=st.m[rh]==-INFINITY?0.f:ex(st.m[rh]-use);\n  float sum=0;\n#pragma unroll\n  for(int n=0;n<NT;n++){\n   const float2 prob=exp2pair(s[n][2*rh]-use,s[n][2*rh+1]-use);\n   const float a=prob.x,b=prob.y;\n   s[n][2*rh]=a;s[n][2*rh+1]=b;sum+=a+b;\n  }\n  sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);\n  st.l[rh]=st.l[rh]*alpha+sum;st.m[rh]=nm;\n#pragma unroll\n  for(int d=0;d<8;d++){st.o[d][2*rh]*=alpha;st.o[d][2*rh+1]*=alpha;}\n }\n if constexpr(CLOCK){const auto now=tick();trace.c[4]+=now-ts;ts=now;}\n uint32_t packed[NT];\n#pragma unroll\n for(int n=0;n<NT;n++)packed[n]=probpack(s[n]);\n if constexpr(CLOCK){const auto now=tick();trace.c[5]+=now-ts;ts=now;}\n#pragma unroll\n for(int k=0;k<N/32;k++){\n  uint32_t pa[4];\n#pragma unroll\n  for(int half=0;half<2;half++){\n   const int src=g*4+2*(t&1);\n   const uint32_t a0=__shfl_sync(0xffffffff,packed[4*k+2*half],src);\n   const uint32_t a1=__shfl_sync(0xffffffff,packed[4*k+2*half],src+1);\n   const uint32_t b0=__shfl_sync(0xffffffff,packed[4*k+2*half+1],src);\n   const uint32_t b1=__shfl_sync(0xffffffff,packed[4*k+2*half+1],src+1);\n   const uint32_t x=t<2?a0:b0,y=t<2?a1:b1;\n   pa[2*half]=(x&0xffffu)|(y<<16);\n   pa[2*half+1]=(x>>16)|(y&0xffff0000u);\n  }\n#pragma unroll\n  for(int d=0;d<4;d++){\n   const int col=16*d+g,kk=32*k+4*t+vkey;\n   const int perm=((col&7)*16)&(VN-1);\n   const uint32_t b00=*(const uint32_t*)(V+col*VN+(kk^perm));\n   const uint32_t b01=*(const uint32_t*)(V+col*VN+((kk+16)^perm));\n   const uint32_t b10=*(const uint32_t*)(V+(col+8)*VN+(kk^perm));\n   const uint32_t b11=*(const uint32_t*)(V+(col+8)*VN+((kk+16)^perm));\n   mma8(st.o[2*d],pa,b00,b01);\n   mma8(st.o[2*d+1],pa,b10,b11);\n  }\n }\n if constexpr(CLOCK)trace.c[6]+=tick()-ts;\n}\ntemplate<int ST,int KT,bool CLOCK>\n__global__ __launch_bounds__(128,2) void verify8_main(VP p){\n Trace trace{};unsigned long long start=0,ts=0,init=0;if constexpr(CLOCK)start=tick();\n constexpr int KB=KT*64;\n extern __shared__ __align__(128) uint8_t sm[];\n uint8_t* K=sm;uint8_t* V=K+ST*KB;uint8_t* Q=V+ST*KB;\n uint8_t* NK=Q+4096;uint8_t* NV=NK+2048;uint8_t* VT=NV+2048;\n const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;\n const int h=blockIdx.x,split=blockIdx.y,b=blockIdx.z,seq=p.seq[b];\n const bool last=split==p.ns-1;\n const int chunk=((seq+p.ns-1)/p.ns+KT-1)&~(KT-1);\n const int begin=min(seq,split*chunk),end=min(seq,begin+chunk),tiles=(end-begin+KT-1)/KT;\n const int* pt=p.pt+(int64_t)b*p.pages;\n constexpr int NR=KT/32;\n int rows[NR];\n auto rowids=[&](int it){\n#pragma unroll\n  for(int rr=0;rr<NR;rr++){\n   const int key=begin+it*KT+tid/4+32*rr;\n   rows[rr]=(it<tiles && key<end)?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):-1;\n  }\n };\n auto issue=[&](int slot){\n#pragma unroll\n  for(int rr=0;rr<NR;rr++){\n   const int row=tid/4+32*rr,col=16*(tid&3),r=rows[rr];\n   const int64_t src=(int64_t)max(r,0)*p.HK*64+h*64+col;\n   cp(K+slot*KB+off(row,col),p.kc+src,r>=0);\n   cp(V+slot*KB+off(row,col),p.vc+src,r>=0);\n  }\n };\n#pragma unroll\n for(int s=0;s<ST-1;s++){rowids(s);issue(s);commit();}\n rowids(ST-1);\n#pragma unroll\n for(int i=0;i<16;i++){\n  const int row=warp+4*i,j=row/8,head=h*8+row%8;\n  const float2 q=input_pair(p,b*8+j,head*64+2*lane);\n  const float r=rsqrtf(wsum(q.x*q.x+q.y*q.y)/64.f+1.e-5f);\n  *(uint16_t*)(Q+row*64+2*lane)=fp8pair(q.x*r,q.y*r);\n }\n if(last){\n  for(int i=tid;i<4096/16;i+=128)*(uint4*)(NK+i*16)=make_uint4(0,0,0,0);\n  __syncthreads();\n#pragma unroll\n  for(int i=0;i<4;i++){\n   const int rr=warp+4*i,j=rr/2,isv=rr&1;\n   float2 x=input_pair(p,b*8+j,(p.HQ+isv*p.HK+h)*64+2*lane);\n   if(!isv){const float r=rsqrtf(wsum(x.x*x.x+x.y*x.y)/64.f+1.e-5f);x.x*=r;x.y*=r;}\n   const float sc=*(isv?p.vs:p.ks);\n   const uint16_t f=fp8pair(x.x/sc,x.y/sc);\n   *(uint16_t*)((isv?p.vc:p.kc)+p.loc[b*8+j]*p.HK*64+h*64+2*lane)=f;\n   *(uint16_t*)((isv?NV:NK)+off(j,2*lane))=f;\n  }\n }\n __syncthreads();State st;\n#pragma unroll\n for(int k=0;k<2;k++){\n  const int qr=warp*16+g;\n  st.q[k][0]=*(const uint32_t*)(Q+qr*64+32*k+4*t);\n  st.q[k][1]=*(const uint32_t*)(Q+(qr+8)*64+32*k+4*t);\n  st.q[k][2]=*(const uint32_t*)(Q+qr*64+32*k+16+4*t);\n  st.q[k][3]=*(const uint32_t*)(Q+(qr+8)*64+32*k+16+4*t);\n }\n st.m[0]=st.m[1]=-INFINITY;st.l[0]=st.l[1]=0;\n#pragma unroll\n for(int d=0;d<8;d++)for(int e=0;e<4;e++)st.o[d][e]=0;\n const float scale=p.scale*(*p.ks);\n if constexpr(CLOCK)init=tick()-start;\n for(int it=0;it<tiles;it++){\n  if constexpr(CLOCK)ts=tick();\n  wait<ST-2>();__syncthreads();\n  if constexpr(CLOCK){const auto now=tick();trace.c[1]+=now-ts;ts=now;}\n  issue((it+ST-1)%ST);commit();rowids(it+ST);\n  if constexpr(CLOCK){const auto now=tick();trace.c[0]+=now-ts;ts=now;}\n  transpose_v<KT>(V+(it%ST)*KB,VT);__syncthreads();\n  if constexpr(CLOCK)trace.c[2]+=tick()-ts;\n  // Register footprint stays at 64 scores while a handoff stages 128 keys.\n  // Subtiles are independent warp work; no barrier or KV reload between them.\n#pragma unroll 1\n  for(int sub=0;sub<KT/64;sub++){\n   const int key=begin+it*KT+sub*64;\n   const uint8_t* kp=K+(it%ST)*KB+sub*4096;\n   const uint8_t* vp=VT;\n   if(key+64<=end)tile<64,false,KT,CLOCK>(st,kp,vp,key,end,seq,scale,sub*64,trace);\n   else tile<64,true,KT,CLOCK>(st,kp,vp,key,end,seq,scale,sub*64,trace);\n  }\n }\n wait<0>();__syncthreads();\n if(last){transpose_v<32>(NV,VT);__syncthreads();tile<32,true,32,CLOCK>(st,NK,VT,seq,seq+8,seq,scale,0,trace);}\n float* out=p.partial+(((int64_t)b*p.HK+h)*p.ns+split)*ROWS*PR;\n#pragma unroll\n for(int rh=0;rh<2;rh++){\n  const int row=warp*16+g+8*rh;\n#pragma unroll\n  for(int d=0;d<8;d++)*(float2*)(out+row*PR+8*d+2*t)=make_float2(st.o[d][2*rh],st.o[d][2*rh+1]);\n  if(t==0)*(float2*)(out+row*PR+64)=make_float2(st.m[rh],st.l[rh]);\n }\n if constexpr(CLOCK){\n  const unsigned long long elapsed=tick()-start;\n  if(lane==0){\n   auto* dst=p.clocks+((((int64_t)b*p.HK+h)*p.ns+split)*4+warp)*10;\n#pragma unroll\n   for(int k=0;k<7;k++)dst[k]=trace.c[k];\n   dst[7]=init;dst[8]=elapsed;dst[9]=tiles;\n  }\n }\n\n}\n// Each warp owns one query/head row. Grid size is B*HK*16 independent CTAs.\n__global__ __launch_bounds__(128,4) void verify8_combine(VP p){\n const int warp=threadIdx.x>>5,lane=threadIdx.x&31;\n const int row=blockIdx.x*4+warp,h=blockIdx.y,b=blockIdx.z;\n const float* base=p.partial+((int64_t)b*p.HK+h)*p.ns*ROWS*PR+row*PR;\n float mx=-INFINITY;\n for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,base[(int64_t)s*ROWS*PR+64]);\n#pragma unroll\n for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));\n float total=0;\n for(int s=lane;s<p.ns;s+=32){const float2 ml=*(const float2*)(base+(int64_t)s*ROWS*PR+64);total+=ml.y*ex(ml.x-mx);}\n total=wsum(total);\n float x=0,y=0;\n for(int first=0;first<p.ns;first+=8){\n  float2 ov[8],ml[8];\n#pragma unroll\n  for(int k=0;k<8;k++)if(first+k<p.ns){\n   const float* pp=base+(int64_t)(first+k)*ROWS*PR;\n   ov[k]=*(const float2*)(pp+2*lane);ml[k]=*(const float2*)(pp+64);\n  }\n#pragma unroll\n  for(int k=0;k<8;k++)if(first+k<p.ns){const float a=ex(ml[k].x-mx);x=fmaf(a,ov[k].x,x);y=fmaf(a,ov[k].y,y);}\n }\n const float sc=(*p.vs)/(total*256.f);x*=sc;y*=sc;\n const float norm=rsqrtf(wsum(x*x+y*y)/64.f+1.e-5f);\n const int j=row/8,head=h*8+row%8;\n const float2 gate=input_pair(p,b*8+j,(p.HQ+2*p.HK+head)*64+2*lane);\n *(__nv_bfloat162*)(p.out+((int64_t)(b*8+j)*p.HQ+head)*64+2*lane)=\n __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate.x)),y*norm/(1.f+__expf(-gate.y)));\n}\ntemplate<int ST,int KT,bool CLOCK>void launch(VP p,cudaStream_t stream){\n constexpr int smem=ST*KT*128+8192+KT*64;\n cudaFuncSetAttribute(verify8_main<ST,KT,CLOCK>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\n verify8_main<ST,KT,CLOCK><<<dim3(p.HK,p.ns,p.B),128,smem,stream>>>(p);\n verify8_combine<<<dim3(16,p.HK,p.B),128,0,stream>>>(p);\n}\n}\nvoid verify8_forward(int64_t qkv,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,\n int64_t ks,int64_t vs,int64_t parts,int64_t out,int64_t B,int64_t HQ,int64_t HK,int64_t stride,\n int64_t pages,int64_t ps,int64_t ns,int64_t proj_splits,double scale,int64_t stages,int64_t kt,int64_t profile,int64_t clocks,int64_t stream){\n VP p{(const __nv_bfloat16*)qkv,(const int64_t*)loc,(const int*)pt,(const int*)seq,(uint8_t*)kc,(uint8_t*)vc,\n (const float*)ks,(const float*)vs,(float*)parts,(__nv_bfloat16*)out,(int)B,(int)HQ,(int)HK,(int)stride,(int)pages,\n (int)ps,0,(int)ns,(int)proj_splits,(float)scale,(unsigned long long*)clocks};\n while((1<<p.lg)<ps)p.lg++;\n const auto s=(cudaStream_t)stream;\n if(profile){ if(kt==64){if(stages==3)launch<3,64,true>(p,s);else launch<4,64,true>(p,s);}\n else {if(stages==3)launch<3,128,true>(p,s);else launch<4,128,true>(p,s);}}else{ if(kt==64){if(stages==3)launch<3,64,false>(p,s);else launch<4,64,false>(p,s);}\n else {if(stages==3)launch<3,128,false>(p,s);else launch<4,128,false>(p,s);}}\n const auto err=cudaGetLastError();if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));\n}\n\nnamespace {\ntemplate<int ST,int KT,bool C>std::vector<int64_t> resources(){\n cudaFuncAttributes a;cudaFuncGetAttributes(&a,verify8_main<ST,KT,C>);\n constexpr int smem=ST*KT*128+8192+KT*64;\n cudaFuncSetAttribute(verify8_main<ST,KT,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\n int active=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,verify8_main<ST,KT,C>,128,smem);\n return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,active};\n}\n}\nstd::vector<int64_t> verify8_resources(int64_t stages,int64_t kt,int64_t profile){\n#define RES(S,K) if(stages==S && kt==K){return profile?resources<S,K,true>():resources<S,K,false>();}\n RES(3,64) RES(4,64) RES(3,128) RES(4,128)\n#undef RES\n throw std::runtime_error("bad resource setting");\n}\n'
PHASE_CPP='#include <cstdint>\n#include <vector>\nvoid verify8_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t,int64_t,int64_t,int64_t);\nstd::vector<int64_t> verify8_resources(int64_t,int64_t,int64_t);\n'
WIDE_CUDA='#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <cstdint>\n#include <stdexcept>\n#include <vector>\n#define DI __device__ __forceinline__\nnamespace {\nconstexpr int D=64, ROWS=64, PR=68, THREADS=128;\nstruct VP {\n const __nv_bfloat16* qkv; const int64_t* loc; const int* pt; const int* seq;\n uint8_t* kc; uint8_t* vc; const float* ks; const float* vs;\n float* partial; __nv_bfloat16* out;\n int B,HQ,HK,stride,pages,ps,lg,ns,proj_splits; float scale;\n unsigned long long* clocks;\n};\nDI uint32_t su(const void* x){return (uint32_t)__cvta_generic_to_shared(x);}\nDI void cp(void* dst,const void* src,bool valid){\n asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(su(dst)),"l"(src),"r"(valid?16:0):"memory");\n}\nDI void commit(){asm volatile("cp.async.commit_group;":::"memory");}\ntemplate<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}\nDI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}\nDI float2 exp2pair(float x,float y){\n __half2 h=__floats2half2_rn(x,y);uint32_t bits=*(uint32_t*)&h,out;\n asm("ex2.approx.f16x2 %0,%1;":"=r"(out):"r"(bits));\n return __half22float2(*(__half2*)&out);\n}\nDI float wsum(float x){\n#pragma unroll\n for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);\n return x;\n}\nDI uint16_t fp8pair(float x,float y){return __nv_cvt_float2_to_fp8x2(make_float2(x,y),__NV_SATFINITE,__NV_E4M3);}\nDI int off(int r,int c){return r*64+(((c>>4)^(r&3))*16)+(c&15);}\nDI uint32_t word(const uint8_t* p,int r,int c){return *(const uint32_t*)(p+off(r,c));}\nDI float2 input_pair(const VP& p,int token,int col){\n if(!p.proj_splits)return __bfloat1622float2(*(const __nv_bfloat162*)(p.qkv+(int64_t)token*p.stride+col));\n const float* raw=(const float*)p.qkv;float2 sum=make_float2(0,0);\n for(int s=0;s<p.proj_splits;s++){\n  const float2 x=*(const float2*)(raw+((int64_t)s*p.B*8+token)*p.stride+col);\n  sum.x+=x.x;sum.y+=x.y;\n }\n return __bfloat1622float2(__floats2bfloat162_rn(sum.x,sum.y));\n}\nDI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){\n asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "\n "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n}\nstruct State {\n uint32_t q[4][4];\n float o[8][4];\n float m[2],l[2];\n};\n// Convert an 8-key QK accumulator panel into two rows of FP8 pairs. No shared store.\nDI uint32_t probpack(float (&s)[4]){\n return (uint32_t)fp8pair(s[0]*256.f,s[1]*256.f)|((uint32_t)fp8pair(s[2]*256.f,s[3]*256.f)<<16);\n}\nstruct Trace { unsigned long long c[7]; };\nDI unsigned long long tick(){unsigned long long x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}\nDI uint32_t widen8(uint16_t x){uint32_t y;asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(y):"h"(x));return y;}\nDI int off16(int r,int c){return r*128+(((c>>3)^(r&7))*16)+((c&7)*2);}\nDI uint32_t word16(const uint8_t* p,int r,int c){return *(const uint32_t*)(p+off16(r,c));}\ntemplate<int N>DI void widen_stage(const uint8_t* K,const uint8_t* V,uint8_t* KH,uint8_t* VH){\n const int tid=threadIdx.x;\n for(int i=tid;i<N*32;i+=128){\n  const int r=i/32,c=(i%32)*2;\n  const uint32_t k=widen8(*(const uint16_t*)(K+off(r,c)));\n  const uint32_t v=widen8(*(const uint16_t*)(V+off(r,c)));\n  *(uint32_t*)(KH+off16(r,c))=k;\n  const int perm0=((c&7)*8)&(N-1),perm1=(((c+1)&7)*8)&(N-1);\n  *(uint16_t*)(VH+2*(c*N+(r^perm0)))=(uint16_t)v;\n  *(uint16_t*)(VH+2*((c+1)*N+(r^perm1)))=(uint16_t)(v>>16);\n }\n}\nDI void mma16(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){\n asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "\n "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n}\nDI uint32_t pack16(float x,float y){__half2 h=__floats2half2_rn(x,y);return *(uint32_t*)&h;}\ntemplate<int N,bool MASK,int VN,bool CLOCK>\nDI void tile(State& st,const uint8_t* K,const uint8_t* V,int kp,int hi,int seq,float scale,int vkey,Trace& trace){\n unsigned long long ts=0;if constexpr(CLOCK)ts=tick();\n constexpr int NT=N/8;\n const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,warp=threadIdx.x>>5;\n float s[NT][4];\n#pragma unroll\n for(int n=0;n<NT;n++){\n#pragma unroll\n  for(int i=0;i<4;i++)s[n][i]=0;\n#pragma unroll\n  for(int k=0;k<4;k++){\n   const uint32_t b0=word16(K,n*8+g,k*16+t*2);\n   const uint32_t b1=word16(K,n*8+g,k*16+8+t*2);\n   mma16(s[n],st.q[k],b0,b1);\n  }\n }\n if constexpr(CLOCK){const auto now=tick();trace.c[3]+=now-ts;ts=now;}\n // Four lanes jointly own a query row. Both online reductions and P stay in registers.\n#pragma unroll\n for(int rh=0;rh<2;rh++){\n  float mx=-INFINITY;\n#pragma unroll\n  for(int n=0;n<NT;n++){\n#pragma unroll\n   for(int e=0;e<2;e++){\n    float x=s[n][2*rh+e]*scale;\n    if constexpr(MASK){\n     const int key=kp+8*n+2*t+e;\n     if(key>=hi || key>seq+2*warp+rh)x=-INFINITY;\n    }\n    s[n][2*rh+e]=x;mx=fmaxf(mx,x);\n   }\n  }\n  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));\n  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));\n  const float nm=fmaxf(st.m[rh],mx),use=nm==-INFINITY?0.f:nm;\n  const float alpha=st.m[rh]==-INFINITY?0.f:ex(st.m[rh]-use);\n  float sum=0;\n#pragma unroll\n  for(int n=0;n<NT;n++){\n   const float2 prob=exp2pair(s[n][2*rh]-use,s[n][2*rh+1]-use);\n   const float a=prob.x,b=prob.y;\n   s[n][2*rh]=a;s[n][2*rh+1]=b;sum+=a+b;\n  }\n  sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);\n  st.l[rh]=st.l[rh]*alpha+sum;st.m[rh]=nm;\n#pragma unroll\n  for(int d=0;d<8;d++){st.o[d][2*rh]*=alpha;st.o[d][2*rh+1]*=alpha;}\n }\n if constexpr(CLOCK){const auto now=tick();trace.c[4]+=now-ts;ts=now;}\n uint32_t packed[NT][2];\n#pragma unroll\n for(int n=0;n<NT;n++){\n  packed[n][0]=pack16(s[n][0],s[n][1]);packed[n][1]=pack16(s[n][2],s[n][3]);\n }\n if constexpr(CLOCK){const auto now=tick();trace.c[5]+=now-ts;ts=now;}\n#pragma unroll\n for(int k=0;k<N/16;k++){\n  const uint32_t pa[4]={packed[2*k][0],packed[2*k][1],packed[2*k+1][0],packed[2*k+1][1]};\n#pragma unroll\n  for(int d=0;d<8;d++){\n   const int col=8*d+g,kk=16*k+2*t+vkey,perm=((col&7)*8)&(VN-1);\n   const uint32_t b0=*(const uint32_t*)(V+2*(col*VN+(kk^perm)));\n   const uint32_t b1=*(const uint32_t*)(V+2*(col*VN+((kk+8)^perm)));\n   mma16(st.o[d],pa,b0,b1);\n  }\n }\n if constexpr(CLOCK)trace.c[6]+=tick()-ts;\n}\ntemplate<int ST,int KT,bool CLOCK>\n__global__ __launch_bounds__(128,2) void verify8_main(VP p){\n Trace trace{};unsigned long long start=0,ts=0,init=0;if constexpr(CLOCK)start=tick();\n constexpr int KB=KT*64;\n extern __shared__ __align__(128) uint8_t sm[];\n uint8_t* K=sm;uint8_t* V=K+ST*KB;uint8_t* Q=V+ST*KB;\n constexpr int QRES=KT*128>8192?KT*128:8192;uint8_t* NK=Q+QRES;uint8_t* NV=NK+2048;uint8_t* KH=Q;uint8_t* VT=NV+2048;\n const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;\n const int h=blockIdx.x,split=blockIdx.y,b=blockIdx.z,seq=p.seq[b];\n const bool last=split==p.ns-1;\n const int chunk=((seq+p.ns-1)/p.ns+KT-1)&~(KT-1);\n const int begin=min(seq,split*chunk),end=min(seq,begin+chunk),tiles=(end-begin+KT-1)/KT;\n const int* pt=p.pt+(int64_t)b*p.pages;\n constexpr int NR=KT/32;\n int rows[NR];\n auto rowids=[&](int it){\n#pragma unroll\n  for(int rr=0;rr<NR;rr++){\n   const int key=begin+it*KT+tid/4+32*rr;\n   rows[rr]=(it<tiles && key<end)?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):-1;\n  }\n };\n auto issue=[&](int slot){\n#pragma unroll\n  for(int rr=0;rr<NR;rr++){\n   const int row=tid/4+32*rr,col=16*(tid&3),r=rows[rr];\n   const int64_t src=(int64_t)max(r,0)*p.HK*64+h*64+col;\n   cp(K+slot*KB+off(row,col),p.kc+src,r>=0);\n   cp(V+slot*KB+off(row,col),p.vc+src,r>=0);\n  }\n };\n#pragma unroll\n for(int s=0;s<ST-1;s++){rowids(s);issue(s);commit();}\n rowids(ST-1);\n#pragma unroll\n for(int i=0;i<16;i++){\n  const int row=warp+4*i,j=row/8,head=h*8+row%8;\n  const float2 q=input_pair(p,b*8+j,head*64+2*lane);\n  const float r=rsqrtf(wsum(q.x*q.x+q.y*q.y)/64.f+1.e-5f);\n  *(uint32_t*)(Q+row*128+4*lane)=widen8(fp8pair(q.x*r,q.y*r));\n }\n if(last){\n  for(int i=tid;i<4096/16;i+=128)*(uint4*)(NK+i*16)=make_uint4(0,0,0,0);\n  __syncthreads();\n#pragma unroll\n  for(int i=0;i<4;i++){\n   const int rr=warp+4*i,j=rr/2,isv=rr&1;\n   float2 x=input_pair(p,b*8+j,(p.HQ+isv*p.HK+h)*64+2*lane);\n   if(!isv){const float r=rsqrtf(wsum(x.x*x.x+x.y*x.y)/64.f+1.e-5f);x.x*=r;x.y*=r;}\n   const float sc=*(isv?p.vs:p.ks);\n   const uint16_t f=fp8pair(x.x/sc,x.y/sc);\n   *(uint16_t*)((isv?p.vc:p.kc)+p.loc[b*8+j]*p.HK*64+h*64+2*lane)=f;\n   *(uint16_t*)((isv?NV:NK)+off(j,2*lane))=f;\n  }\n }\n __syncthreads();State st;\n#pragma unroll\n for(int k=0;k<4;k++){\n  const int qr=warp*16+g;\n  st.q[k][0]=*(const uint32_t*)(Q+qr*128+2*(16*k+2*t));\n  st.q[k][1]=*(const uint32_t*)(Q+(qr+8)*128+2*(16*k+2*t));\n  st.q[k][2]=*(const uint32_t*)(Q+qr*128+2*(16*k+8+2*t));\n  st.q[k][3]=*(const uint32_t*)(Q+(qr+8)*128+2*(16*k+8+2*t));\n }\n st.m[0]=st.m[1]=-INFINITY;st.l[0]=st.l[1]=0;\n#pragma unroll\n for(int d=0;d<8;d++)for(int e=0;e<4;e++)st.o[d][e]=0;\n const float scale=p.scale*(*p.ks);\n if constexpr(CLOCK)init=tick()-start;\n for(int it=0;it<tiles;it++){\n  if constexpr(CLOCK)ts=tick();\n  wait<ST-2>();__syncthreads();\n  if constexpr(CLOCK){const auto now=tick();trace.c[1]+=now-ts;ts=now;}\n  issue((it+ST-1)%ST);commit();rowids(it+ST);\n  if constexpr(CLOCK){const auto now=tick();trace.c[0]+=now-ts;ts=now;}\n  widen_stage<KT>(K+(it%ST)*KB,V+(it%ST)*KB,KH,VT);__syncthreads();\n  if constexpr(CLOCK)trace.c[2]+=tick()-ts;\n  // Register footprint stays at 64 scores while a handoff stages 128 keys.\n  // Subtiles are independent warp work; no barrier or KV reload between them.\n#pragma unroll 1\n  for(int sub=0;sub<KT/64;sub++){\n   const int key=begin+it*KT+sub*64;\n   const uint8_t* kp=KH+sub*8192;\n   const uint8_t* vp=VT;\n   if(key+64<=end)tile<64,false,KT,CLOCK>(st,kp,vp,key,end,seq,scale,sub*64,trace);\n   else tile<64,true,KT,CLOCK>(st,kp,vp,key,end,seq,scale,sub*64,trace);\n  }\n }\n wait<0>();__syncthreads();\n if(last){widen_stage<32>(NK,NV,KH,VT);__syncthreads();tile<32,true,32,CLOCK>(st,KH,VT,seq,seq+8,seq,scale,0,trace);}\n float* out=p.partial+(((int64_t)b*p.HK+h)*p.ns+split)*ROWS*PR;\n#pragma unroll\n for(int rh=0;rh<2;rh++){\n  const int row=warp*16+g+8*rh;\n#pragma unroll\n  for(int d=0;d<8;d++)*(float2*)(out+row*PR+8*d+2*t)=make_float2(st.o[d][2*rh],st.o[d][2*rh+1]);\n  if(t==0)*(float2*)(out+row*PR+64)=make_float2(st.m[rh],st.l[rh]);\n }\n if constexpr(CLOCK){\n  const unsigned long long elapsed=tick()-start;\n  if(lane==0){\n   auto* dst=p.clocks+((((int64_t)b*p.HK+h)*p.ns+split)*4+warp)*10;\n#pragma unroll\n   for(int k=0;k<7;k++)dst[k]=trace.c[k];\n   dst[7]=init;dst[8]=elapsed;dst[9]=tiles;\n  }\n }\n\n}\n// Each warp owns one query/head row. Grid size is B*HK*16 independent CTAs.\n__global__ __launch_bounds__(128,4) void verify8_combine(VP p){\n const int warp=threadIdx.x>>5,lane=threadIdx.x&31;\n const int row=blockIdx.x*4+warp,h=blockIdx.y,b=blockIdx.z;\n const float* base=p.partial+((int64_t)b*p.HK+h)*p.ns*ROWS*PR+row*PR;\n float mx=-INFINITY;\n for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,base[(int64_t)s*ROWS*PR+64]);\n#pragma unroll\n for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));\n float total=0;\n for(int s=lane;s<p.ns;s+=32){const float2 ml=*(const float2*)(base+(int64_t)s*ROWS*PR+64);total+=ml.y*ex(ml.x-mx);}\n total=wsum(total);\n float x=0,y=0;\n for(int first=0;first<p.ns;first+=8){\n  float2 ov[8],ml[8];\n#pragma unroll\n  for(int k=0;k<8;k++)if(first+k<p.ns){\n   const float* pp=base+(int64_t)(first+k)*ROWS*PR;\n   ov[k]=*(const float2*)(pp+2*lane);ml[k]=*(const float2*)(pp+64);\n  }\n#pragma unroll\n  for(int k=0;k<8;k++)if(first+k<p.ns){const float a=ex(ml[k].x-mx);x=fmaf(a,ov[k].x,x);y=fmaf(a,ov[k].y,y);}\n }\n const float sc=(*p.vs)/total;x*=sc;y*=sc;\n const float norm=rsqrtf(wsum(x*x+y*y)/64.f+1.e-5f);\n const int j=row/8,head=h*8+row%8;\n const float2 gate=input_pair(p,b*8+j,(p.HQ+2*p.HK+head)*64+2*lane);\n *(__nv_bfloat162*)(p.out+((int64_t)(b*8+j)*p.HQ+head)*64+2*lane)=\n __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate.x)),y*norm/(1.f+__expf(-gate.y)));\n}\ntemplate<int ST,int KT,bool CLOCK>void launch(VP p,cudaStream_t stream){\n constexpr int smem=ST*KT*128+(KT*128>8192?KT*128:8192)+4096+KT*128;\n cudaFuncSetAttribute(verify8_main<ST,KT,CLOCK>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\n verify8_main<ST,KT,CLOCK><<<dim3(p.HK,p.ns,p.B),128,smem,stream>>>(p);\n verify8_combine<<<dim3(16,p.HK,p.B),128,0,stream>>>(p);\n}\n}\nvoid verify8_forward(int64_t qkv,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,\n int64_t ks,int64_t vs,int64_t parts,int64_t out,int64_t B,int64_t HQ,int64_t HK,int64_t stride,\n int64_t pages,int64_t ps,int64_t ns,int64_t proj_splits,double scale,int64_t stages,int64_t kt,int64_t profile,int64_t clocks,int64_t stream){\n VP p{(const __nv_bfloat16*)qkv,(const int64_t*)loc,(const int*)pt,(const int*)seq,(uint8_t*)kc,(uint8_t*)vc,\n (const float*)ks,(const float*)vs,(float*)parts,(__nv_bfloat16*)out,(int)B,(int)HQ,(int)HK,(int)stride,(int)pages,\n (int)ps,0,(int)ns,(int)proj_splits,(float)scale,(unsigned long long*)clocks};\n while((1<<p.lg)<ps)p.lg++;\n const auto s=(cudaStream_t)stream;\n if(profile){ if(kt==64){if(stages==3)launch<3,64,true>(p,s);else launch<4,64,true>(p,s);}\n else {if(stages==3)launch<3,128,true>(p,s);else launch<4,128,true>(p,s);}}else{ if(kt==64){if(stages==3)launch<3,64,false>(p,s);else launch<4,64,false>(p,s);}\n else {if(stages==3)launch<3,128,false>(p,s);else launch<4,128,false>(p,s);}}\n const auto err=cudaGetLastError();if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));\n}\n\nnamespace {\ntemplate<int ST,int KT,bool C>std::vector<int64_t> resources(){\n cudaFuncAttributes a;cudaFuncGetAttributes(&a,verify8_main<ST,KT,C>);\n constexpr int smem=ST*KT*128+(KT*128>8192?KT*128:8192)+4096+KT*128;\n cudaFuncSetAttribute(verify8_main<ST,KT,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\n int active=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,verify8_main<ST,KT,C>,128,smem);\n return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,active};\n}\n}\nstd::vector<int64_t> verify8_resources(int64_t stages,int64_t kt,int64_t profile){\n#define RES(S,K) if(stages==S && kt==K){return profile?resources<S,K,true>():resources<S,K,false>();}\n RES(3,64) RES(4,64) RES(3,128) RES(4,128)\n#undef RES\n throw std::runtime_error("bad resource setting");\n}\n'

_EXT_PHASE={}
PHASE_NAMES=('issue','wait','transpose','qk','softmax','p_convert','pv_layout')

def phase_ext(wide=False):
    global _EXT_PHASE
    if wide not in _EXT_PHASE:
        import hashlib
        major,minor=torch.cuda.get_device_capability();arch=f'{major}{minor}'
        source=WIDE_CUDA if wide else PHASE_CUDA
        _EXT_PHASE[wide]=load_inline(name=f'verify8_phase_{hashlib.sha256(source.encode()).hexdigest()[:12]}_{arch}',
            cpp_sources=PHASE_CPP,cuda_sources=source,functions=['verify8_forward','verify8_resources'],
            extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'],verbose=False)
    return _EXT_PHASE[wide]


def phase_run(data,setting,profile,wide=False):
    cfg=data['config'];B=cfg['batch'];HQ=cfg['num_q_heads'];HK=cfg['num_kv_heads'];ns,st,kt=setting
    parts=torch.empty((B*HK*ns*64*68,),device='cuda',dtype=torch.float32)
    out=torch.empty((B,8,1024),device='cuda',dtype=torch.bfloat16)
    clocks=torch.empty((B*HK*ns,4,10),device='cuda',dtype=torch.uint64) if profile else torch.empty(1,device='cuda',dtype=torch.uint64)
    phase_ext(wide).verify8_forward(data['qkv'].data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),
        data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),
        data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),parts.data_ptr(),out.data_ptr(),
        B,HQ,HK,data['qkv'].shape[-1],data['page_table'].shape[1],1,ns,0,
        cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,st,kt,int(profile),clocks.data_ptr(),
        torch.cuda.current_stream().cuda_stream)
    return out,clocks



def phase_fa3(data):
    raw=data['qkv'].view(1,-1,2304)
    q,g=qk_prep((raw,1),data,{'inv_freq':None},q_fp8=True)
    return _attention_fa3_public(q,g,data,{})


def lab(deadline,weights):
    import json
    lines=['META measured be19da94 producer; synthetic task inputs; clock cycles per stage; PV includes P layout shuffles']
    if deadline-time.monotonic()<25:return lines+['SKIP budget<25s before compile']
    ext=phase_ext();phase_ext(True);sms=torch.cuda.get_device_properties('cuda').multi_processor_count
    lines.append(f'META sms={sms} clocks instrument every warp; compare plain/profile overhead and resource limits')
    flush=torch.empty(128<<20,dtype=torch.uint8,device='cuda')
    for B in (16,32):
        if deadline-time.monotonic()<12:lines.append(f'SKIP b{B} budget<12s');break
        data=generate_input(B,80000,'tp8','full',741+B,q_len=8,page=1,spread='agentic',dist='coherent')
        data['config']['tokens']=B*8
        control_us,control,control_graph=timed(phase_fa3,data,flush)
        control_expected=gate_prepare(data);cok,cmsg,cerr=gate_cached(data,control,control_expected)
        lines.append(f'FA3 B={B} us={control_us:.2f} gate={"P" if cok else "F"} err={cerr:.6f}')
        if not cok:lines.append('GATE_ERROR '+cmsg)
        base,_=phase_run(data,(17,3,128),False);torch.cuda.synchronize()
        expected=gate_prepare(data);logical=float(data['seq_lens'].sum())*2*2*64
        for wide,setting in ((False,(17,3,128)),(True,(17,3,128)),(True,(34,3,64)),(False,(17,4,128))):
            if deadline-time.monotonic()<4:lines.append(f'SKIP b{B} remaining configs budget<4s');break
            ns,st,kt=setting
            def plain(d):return phase_run(d,setting,False,wide)[0]
            us,out,g=timed(plain,data,flush)
            ok,msg,err=gate_cached(data,out,expected)
            traced,clocks=phase_run(data,setting,True,wide);torch.cuda.synchronize()
            equal=bool(torch.equal(out,traced));tok,tmsg,terr=gate_cached(data,traced,expected)
            def traced_fn(d):return phase_run(d,setting,True,wide)[0]
            instrument_us,_,tg=timed(traced_fn,data,flush,rounds=3)
            values=clocks.cpu().to(torch.float64).reshape(-1,10);counts=values[:,9].clamp_min(1)
            samples=values[:,:7]/counts[:,None]
            med=samples.median(0).values.tolist();p90=torch.quantile(samples,.9,dim=0).tolist()
            resources=[phase_ext(wide).verify8_resources(st,kt,i) for i in (0,1)]
            longest=int(data['seq_lens'].max());chunk=((longest+ns-1)//ns+kt-1)//kt*kt
            row=dict(B=B,wide=wide,setting=setting,us=round(us,2),tb_s=round(logical/us/1.e6,3),gate=bool(ok and tok and equal),
                error=round(err,6),trace_equal=equal,trace_us=round(instrument_us,2),grid=B*2*ns,
                cta_kv_max_bytes=chunk*128,resources=resources,cta_cycles=[round(values[:,7].median().item()),round(torch.quantile(values[:,7],.9).item()),round(values[:,8].median().item()),round(torch.quantile(values[:,8],.9).item())],
                cycles={k:[round(a),round(b)] for k,a,b in zip(PHASE_NAMES,med,p90)})
            lines.append('ROW '+json.dumps(row,separators=(',',':')))
            if not row['gate']:lines.append('GATE_ERROR '+msg+'; '+tmsg)
        # Graphs and closures leave scope on the next iteration; no captured name is deleted.
    return lines
