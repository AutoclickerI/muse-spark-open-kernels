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
"""Verify-1 sliding option: H200 attention kernel win; whole-layer comparison required."""
import torch
from torch.utils.cpp_extension import load_inline

_INV_FREQ = {}

def _inv_freq(theta, device):
    key = (float(theta), str(device))
    if key not in _INV_FREQ:
        _INV_FREQ[key] = (1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()
    return _INV_FREQ[key]


WINDOW_CUDA = '\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <cstdint>\n#include <stdexcept>\n#define DEVI __device__ __forceinline__\nnamespace {\nconstexpr int ROWB=64;\nDEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }\n\nDEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");\n}\nDEVI void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::: "memory"); }\ntemplate <int N>\nDEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N) : "memory"); }\n\nDEVI uint32_t cvt_e4m3x2(uint32_t v) {\n  uint32_t r;\n  uint16_t h = static_cast<uint16_t>(v & 0xffffu);\n  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\\n" : "=r"(r) : "h"(h));\n  return r;\n}\n\nDEVI uint32_t pack_h2(float lo, float hi) {\n  __half2 h = __floats2half2_rn(lo, hi);\n  return *reinterpret_cast<uint32_t*>(&h);\n}\n\nDEVI float ex2(float x) {\n  float y;\n  asm("ex2.approx.ftz.f32 %0, %1;\\n" : "=f"(y) : "f"(x));\n  return y;\n}\n\nDEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "\n      "{%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\nDEVI uint32_t cluster_addr(const void* p, int rank) {\n  uint32_t r;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\\n" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));\n  return r;\n}\nDEVI float ld_cluster_f32(uint32_t a) {\n  float v;\n  asm volatile("ld.shared::cluster.f32 %0, [%1];\\n" : "=f"(v) : "r"(a) : "memory");\n  return v;\n}\nDEVI float2 ld_cluster_f32x2(uint32_t a) {\n  float2 v;\n  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];\\n" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");\n  return v;\n}\nDEVI void cluster_sync_all() {\n  asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;\\n" ::: "memory");\n}\n\n// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.\nDEVI int swz(int t) { return t & 3; }\n\nDEVI float warp_sum(float v) {\n#pragma unroll\n  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n  return v;\n}\n\ntemplate <int L>\nstruct WarpState {\n  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp\n  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims\n  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims\n  float m[MTW][2];                               // running max (log2 domain, scaled)\n  float l[MTW][2];                               // running sum (thread-partial)\n};\n\n// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.\n// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);\n//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).\ntemplate <int L, int NTC>\nDEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,\n                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2, int nq, bool same_query) {\n  constexpr int MTW = WarpState<L>::MTW;\n  constexpr int KK = NTC / 2;\n  const int lane = threadIdx.x & 31;\n  const int g = lane >> 2, q4 = lane & 3;\n\n  float s[MTW][NTC][4];\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt)\n#pragma unroll\n    for (int n = 0; n < NTC; ++n)\n#pragma unroll\n      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;\n\n  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,\n  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q\'s fragments).\n#pragma unroll\n  for (int n = 0; n < NTC; ++n) {\n    const int t = tb + 8 * n + g;\n    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));\n    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};\n#pragma unroll\n    for (int ks = 0; ks < 4; ++ks) {\n      const uint32_t b0 = cvt_e4m3x2(w[ks]);\n      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);\n#pragma unroll\n      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);\n    }\n  }\n\n  if (need_mask) {\n#pragma unroll\n    for (int mt = 0; mt < MTW; ++mt) {\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        const int j = same_query ? 0 : g;\n        const int qp = seq + j;\n        const int lo = window ? (qp - window + 1) : -2147483647;\n        const int hi = min(k_hi, qp + 1);\n#pragma unroll\n        for (int n = 0; n < NTC; ++n) {\n          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);\n          if (kp < lo || kp >= hi || j >= nq) s[mt][n][e] = -INFINITY;\n        }\n      }\n    }\n  }\n\n  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).\n  uint32_t pa[MTW][KK][4];\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt) {\n#pragma unroll\n    for (int rh = 0; rh < 2; ++rh) {\n      if (L == 1 && rh == 1) continue;\n      float mx = -INFINITY;\n#pragma unroll\n      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));\n      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);\n      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;\n      const float corr = ex2(st.m[mt][rh] - m_use);\n      st.m[mt][rh] = m_new;\n      st.l[mt][rh] *= corr;\n#pragma unroll\n      for (int jn = 0; jn < 8; ++jn) {\n        st.o[mt][jn][2 * rh] *= corr;\n        st.o[mt][jn][2 * rh + 1] *= corr;\n      }\n      float sum = 0.f;\n#pragma unroll\n      for (int n = 0; n < NTC; ++n) {\n        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));\n        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));\n        s[mt][n][2 * rh] = p0;\n        s[mt][n][2 * rh + 1] = p1;\n        sum += p0 + p1;\n      }\n      st.l[mt][rh] += sum;\n    }\n#pragma unroll\n    for (int kk = 0; kk < KK; ++kk) {\n      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);\n      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);\n      if (L == 1) {\n        pa[mt][kk][1] = 0u;\n        pa[mt][kk][3] = 0u;\n      } else {\n        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);\n        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);\n      }\n    }\n  }\n\n  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a\n  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.\n#pragma unroll\n  for (int kk = 0; kk < KK; ++kk) {\n    const int t0 = tb + 16 * kk + 2 * q4;\n    const int coff = ((4 * h + (g >> 1))) ;\n    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));\n    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));\n    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));\n    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));\n#pragma unroll\n    for (int jj = 0; jj < 2; ++jj) {\n      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;\n      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;\n      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);\n      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);\n      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};\n      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};\n#pragma unroll\n      for (int x = 0; x < 4; ++x)\n#pragma unroll\n        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);\n    }\n  }\n}\n\nstruct WP {\n const __nv_bfloat16* qkv; const int64_t *pos,*loc; const int *pt,*seq;\n uint8_t *kc,*vc; const float *ks,*vs,*freq; __nv_bfloat16* out;\n float *parts; int* counters;\n int L,HQ,HK,stride,pages,ps,window,ns,proj_splits,tokens; float qscale,smscale;\n};\n// Projection partials are reduced and rounded to BF16 at the same boundary as stock.\nDEVI float2 qkv_pair(const WP& p, int token, int column) {\n if(!p.proj_splits) return __bfloat1622float2(*(const __nv_bfloat162*)(p.qkv+(int64_t)token*p.stride+column));\n const float* raw=(const float*)p.qkv;\n float2 sum=make_float2(0,0);\n for(int s=0;s<p.proj_splits;s++) {\n   const float2 x=*(const float2*)(raw+((int64_t)s*p.tokens+token)*p.stride+column);\n   sum.x+=x.x;sum.y+=x.y;\n }\n return __bfloat1622float2(__floats2bfloat162_rn(sum.x,sum.y));\n}\ntemplate<int L, bool SHARE=false>\n__global__ __launch_bounds__(128,2) void window_mma(WP p) {\n constexpr int SZ=256*64, STAGES=2;\n extern __shared__ __align__(128) uint8_t shared[];\n uint8_t* K=shared; uint8_t* V=K+STAGES*SZ;\n uint8_t* NK=V+STAGES*SZ; uint8_t* NV=NK+16*64;\n __half* Q=(__half*)(NV+16*64);\n const int tid=threadIdx.x, warp=tid/32, lane=tid%32, g=lane/4, q4=lane%4;\n const int split=blockIdx.x,h=blockIdx.y,b=blockIdx.z,hk=SHARE?h:h/8;\n constexpr int R=SHARE?8:L;\n const int groups=SHARE?p.HK:p.HQ;\n const int seq=p.seq[b], lo=max(0,seq-p.window+1), count=seq-lo;\n const int chunk=((count+p.ns-1)/p.ns+63)&~63;\n const int begin=min(seq,lo+split*chunk), end=min(seq,begin+chunk);\n const int tiles=(end-begin+255)/256;\n auto issue=[&](int tile,int slot) {\n#pragma unroll\n   for(int i=0;i<8;i++) {\n     const int ix=tid+i*128, t=ix/4, col=ix%4, key=begin+tile*256+t;\n     const bool valid=tile<tiles && key<end;\n     const int row=valid ? p.pt[(int64_t)b*p.pages+key/p.ps]*p.ps+key%p.ps : 0;\n     const int off=slot*SZ+t*64+((col^swz(t))*16);\n     const int64_t src=(int64_t)row*p.HK*64+hk*64+col*16;\n     cp_async16(smem_u32(K+off),p.kc+src,valid?16:0);\n     cp_async16(smem_u32(V+off),p.vc+src,valid?16:0);\n   }\n   cp_commit();\n };\n issue(0,0);\n const float ks=*p.ks, vs=*p.vs;\n#pragma unroll\n for(int i=0;i<2;i++) {\n   const int t=warp+4*i;\n   float qx=0,qy=0,kx=0,ky=0,vx=0,vy=0;\n   float cs=1,sn=0;\n   if(t<R) {\n     const int qt=SHARE?0:t, qh=SHARE?h*8+t:h;\n     const auto qr=qkv_pair(p,b*L+qt,qh*64+2*lane);\n     const float qn=rsqrtf(warp_sum(qr.x*qr.x+qr.y*qr.y)/64+1.e-5f)*p.qscale;\n     if(p.freq) sincosf((float)p.pos[b*L+qt]*p.freq[lane],&sn,&cs);\n     qx=__bfloat162float(__float2bfloat16_rn((qr.x*cs-qr.y*sn)*qn));\n     qy=__bfloat162float(__float2bfloat16_rn((qr.x*sn+qr.y*cs)*qn));\n   }\n   if(t<L) {\n     const auto kr=qkv_pair(p,b*L+t,(p.HQ+hk)*64+2*lane);\n     const auto vr=qkv_pair(p,b*L+t,(p.HQ+p.HK+hk)*64+2*lane);\n     const float kn=rsqrtf(warp_sum(kr.x*kr.x+kr.y*kr.y)/64+1.e-5f);\n     kx=(kr.x*cs-kr.y*sn)*kn/ks; ky=(kr.x*sn+kr.y*cs)*kn/ks;\n     vx=vr.x/vs; vy=vr.y/vs;\n   }\n   *(__half2*)(Q+t*64+2*lane)=__floats2half2_rn(qx,qy);\n   const auto k8=__nv_cvt_float2_to_fp8x2(make_float2(kx,ky),__NV_SATFINITE,__NV_E4M3);\n   const auto v8=__nv_cvt_float2_to_fp8x2(make_float2(vx,vy),__NV_SATFINITE,__NV_E4M3);\n   const int off=t*64+(((lane/8)^swz(t))*16)+(lane%8)*2;\n   *(unsigned short*)(NK+off)=k8; *(unsigned short*)(NV+off)=v8;\n   // Pad up to 16 rows for the short new-token MMA.\n   const int off2=(t+8)*64+(((lane/8)^swz(t+8))*16)+(lane%8)*2;\n   *(unsigned short*)(NK+off2)=0; *(unsigned short*)(NV+off2)=0;\n   if(t<L && split==0 && (SHARE || h%8==0)) {\n     const int64_t dst=p.loc[b*L+t]*p.HK*64+hk*64+2*lane;\n     *(unsigned short*)(p.kc+dst)=k8; *(unsigned short*)(p.vc+dst)=v8;\n   }\n }\n __syncthreads();\n WarpState<1> st;\n#pragma unroll\n for(int k=0;k<4;k++) {\n   const uint2 x=*(const uint2*)(Q+g*64+16*q4+4*k);\n   st.qa[0][k][0]=x.x; st.qa[0][k][2]=x.y;\n   st.qa[0][k][1]=st.qa[0][k][3]=0;\n }\n st.m[0][0]=st.m[0][1]=-INFINITY; st.l[0][0]=st.l[0][1]=0;\n#pragma unroll\n for(int n=0;n<8;n++)\n#pragma unroll\n for(int e=0;e<4;e++) st.o[0][n][e]=0;\n for(int tile=0;tile<tiles;tile++) {\n   cp_wait<0>(); __syncthreads();\n   if(tile+1<tiles) issue(tile+1,(tile+1)%2);\n   const int kp=begin+tile*256;\n   compute_tile<1,8>(st,K+(tile%2)*SZ,V+(tile%2)*SZ,64*warp,0,0,kp,end,seq,p.window,\n                     (kp<seq+L-p.window || kp+256>end),p.smscale*ks*1.4426950408889634f,R,SHARE);\n }\n cp_wait<0>(); __syncthreads();\n if(split==p.ns-1 && warp==0)\n   compute_tile<1,2>(st,NK,NV,0,0,0,seq,seq+L,seq,p.window,true,p.smscale*ks*1.4426950408889634f,R,SHARE);\n float den=st.l[0][0]; den+=__shfl_xor_sync(0xffffffff,den,1); den+=__shfl_xor_sync(0xffffffff,den,2);\n // The pipeline is drained; reuse it for the four warp partials.\n __syncthreads();\n float* sp=(float*)shared;\n float* dst=sp+(warp*8+g)*68;\n float4* d4=(float4*)(dst+16*q4);\n d4[0]=make_float4(st.o[0][0][0],st.o[0][1][0],st.o[0][2][0],st.o[0][3][0]);\n d4[1]=make_float4(st.o[0][4][0],st.o[0][5][0],st.o[0][6][0],st.o[0][7][0]);\n d4[2]=make_float4(st.o[0][0][1],st.o[0][1][1],st.o[0][2][1],st.o[0][3][1]);\n d4[3]=make_float4(st.o[0][4][1],st.o[0][5][1],st.o[0][6][1],st.o[0][7][1]);\n if(q4==0) {dst[64]=st.m[0][0];dst[65]=den;}\n __syncthreads();\n float results[2][2];\n#pragma unroll\n for(int i=0;i<2;i++) {\n   const int j=warp+4*i;\n   if(j<R) {\n     float mx=-INFINITY;\n#pragma unroll\n     for(int w=0;w<4;w++) mx=fmaxf(mx,sp[(w*8+j)*68+64]);\n     float total=0,x=0,y=0;\n#pragma unroll\n     for(int w=0;w<4;w++) {\n       const float* pp=sp+(w*8+j)*68;\n       const float ww=ex2(pp[64]-(mx == -INFINITY ? 0.f : mx));\n       const float2 oo=*(const float2*)(pp+2*lane);\n       x=fmaf(ww,oo.x,x);y=fmaf(ww,oo.y,y);total=fmaf(ww,pp[65],total);\n     }\n     if(p.ns==1) {results[i][0]=x*vs/total;results[i][1]=y*vs/total;}\n     else {\n       float* pp=p.parts+(((int64_t)b*groups+h)*p.ns+split)*R*68+j*68;\n       *(float2*)(pp+2*lane)=make_float2(x,y);\n       if(lane==0) *(float2*)(pp+64)=make_float2(mx,total);\n     }\n   }\n }\n if(p.ns>1) {\n   __threadfence(); __syncthreads();\n   __shared__ int last;\n   if(tid==0) last=(atomicAdd(p.counters+b*groups+h,1)==p.ns-1);\n   __syncthreads();\n   if(!last) return;\n#pragma unroll\n   for(int i=0;i<2;i++) {\n     const int j=warp+4*i;\n     if(j<R) {\n       const float* base=p.parts+((int64_t)b*groups+h)*p.ns*R*68+j*68;\n       float2 ml[16], ov[16];\n#pragma unroll\n       for(int s=0;s<16;s++) if(s<p.ns) {\n         const float* pp=base+s*R*68;\n         ml[s]=__ldcg((const float2*)(pp+64));\n         ov[s]=__ldcg((const float2*)(pp+2*lane));\n       }\n       float mx=-INFINITY;\n#pragma unroll\n       for(int s=0;s<16;s++) if(s<p.ns) mx=fmaxf(mx,ml[s].x);\n       float total=0,x=0,y=0;\n#pragma unroll\n       for(int s=0;s<16;s++) if(s<p.ns) {\n         const float ww=ex2(ml[s].x-mx);\n         x=fmaf(ww,ov[s].x,x);y=fmaf(ww,ov[s].y,y);total=fmaf(ww,ml[s].y,total);\n       }\n       results[i][0]=x*vs/total;results[i][1]=y*vs/total;\n     }\n   }\n }\n#pragma unroll\n for(int i=0;i<2;i++) {\n   const int j=warp+4*i;\n   if(j<R) {\n     const float x=results[i][0],y=results[i][1];\n     const float norm=rsqrtf(warp_sum(x*x+y*y)/64+1.e-5f);\n     const int qt=SHARE?0:j,qh=SHARE?h*8+j:h;\n     const auto gate=qkv_pair(p,b*L+qt,(p.HQ+2*p.HK+qh)*64+2*lane);\n     *(__nv_bfloat162*)(p.out+(int64_t)(b*L+qt)*p.HQ*64+qh*64+2*lane)=\n        __floats2bfloat162_rn(x*norm/(1+__expf(-gate.x)),y*norm/(1+__expf(-gate.y)));\n   }\n }\n __syncthreads(); if(tid==0 && p.ns>1) atomicExch(p.counters+b*groups+h,0);\n}\n}\nvoid window_forward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,\n int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t out,int64_t parts,int64_t counters,\n int64_t B,int64_t L,int64_t HQ,int64_t HK,int64_t stride,int64_t pages,int64_t ps,int64_t window,\n int64_t ns,int64_t proj_splits,double qscale,double smscale,int64_t stream) {\n WP p{(const __nv_bfloat16*)qkv,(const int64_t*)pos,(const int64_t*)loc,(const int*)pt,(const int*)seq,\n (uint8_t*)kc,(uint8_t*)vc,(const float*)ks,(const float*)vs,(const float*)freq,(__nv_bfloat16*)out,\n (float*)parts,(int*)counters,(int)L,(int)HQ,(int)HK,(int)stride,(int)pages,(int)ps,(int)window,(int)ns,(int)proj_splits,(int)(B*L),(float)qscale,(float)smscale};\n constexpr int smem=2*2*256*64+2*16*64+8*64*2;\n cudaError_t err;\n if(L==1 && B>1) {cudaFuncSetAttribute(window_mma<1,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\n   window_mma<1,true><<<dim3(ns,HK,B),128,smem,(cudaStream_t)stream>>>(p);}\n else if(L==1) {cudaFuncSetAttribute(window_mma<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\n   window_mma<1><<<dim3(ns,HQ,B),128,smem,(cudaStream_t)stream>>>(p);}\n else {cudaFuncSetAttribute(window_mma<8>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\n   window_mma<8><<<dim3(ns,HQ,B),128,smem,(cudaStream_t)stream>>>(p);}\n\n err=cudaGetLastError();if(err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));\n}\n'
WINDOW_CPP = '\n#include <cstdint>\nvoid window_forward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,\n int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t out,int64_t parts,int64_t counters,\n int64_t B,int64_t L,int64_t HQ,int64_t HK,int64_t stride,int64_t pages,int64_t ps,int64_t window,\n int64_t ns,int64_t proj_splits,double qscale,double smscale,int64_t stream);\n'
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
    counters = _window_counters(B*HQ, raw.device)
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



_WINDOW_COUNTERS = {}
def _window_counters(n, device):
    key = str(device)
    if key not in _WINDOW_COUNTERS:
        _WINDOW_COUNTERS[key] = torch.zeros(1 << 16, dtype=torch.int32, device=device)
    assert n <= (1 << 16)
    return _WINDOW_COUNTERS[key]


def attention_sliding(raw, data):
    splits = 0
    if isinstance(raw, tuple):
        partial, count = raw
        assert partial.is_contiguous()
        if partial.dtype == torch.float32:
            splits = count
        else:
            assert partial.dtype == torch.bfloat16 and count == 1
        raw = partial
    cfg = dict(data['config'], num_q_heads=16, num_kv_heads=2)
    metadata = dict(data, config=cfg)
    return sliding_attention(raw, metadata, splits=splits).view(cfg['batch']*cfg['q_len'], 1024)


if torch.cuda.is_available():
    _window_counters(1 << 16, torch.device('cuda', torch.cuda.current_device()))


def attention_verify1_sliding(qkv, data):
    cfg = data['config']
    if (cfg['batch'], cfg['q_len'], cfg['window']) != (1, 8, 2048):
        raise ValueError('verify1 sliding pick accepts only batch=1, q_len=8, window=2048')
    return attention_sliding(qkv, data)

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



def layer_fa3(qkv,data,state,glue='current'):
    # Both current and best5 glue converge on the exact public FA3 call.
    q,gate=(qk_prep if glue=='current' else _best7_old_qk_prep)(qkv,data,state,q_fp8=True)
    return _attention_fa3_public(q,gate,data,state)


def lab(deadline,weights):
    lines=['META verify1 sliding: exact guarded adapter vs decode public FA3 incl qk_prep and norm_gate; no projection GEMV',
           'META q_len=8 B=1 ctx=80000 window2048 page1; PDL=True control; BF16 and FP32 partial ABIs']
    if deadline-time.monotonic()<25:return lines+['SKIP budget<25s before compile']
    _window_ext()
    data=generate_input(1,80000,'tp8','sliding',931,q_len=8,page=1,dist='soft')
    data['config']['tokens']=8
    theta=data['config']['rope_theta']
    state={'inv_freq':1./(theta**(torch.arange(0,64,2,device='cuda').float()/64.))}
    flush=torch.empty(128<<20,dtype=torch.uint8,device='cuda')
    for count in (0,1,4):
        if deadline-time.monotonic()<8:lines.append(f'SKIP partials{count} budget<8s');break
        raw=data['qkv'].view(8,2304)
        if count:
            raw=torch.randn((count,8,2304),device='cuda',dtype=torch.float32)
            data['qkv']=raw.sum(0).to(torch.bfloat16).view(1,8,2304)
            qkv=(raw,count)
        else:qkv=(raw.unsqueeze(0),1)
        first=attention_verify1_sliding(qkv,data);torch.cuda.synchronize()
        expected=gate_prepare(data)
        # Fresh values arrive through the same captured projection-buffer ABI.
        for name in ('single','fa3_current','fa3_best5'):
            if deadline-time.monotonic()<3:lines.append('SKIP paired tail budget<3s');break
            def call(d):
                if name=='single':return attention_verify1_sliding(qkv,d)
                return layer_fa3(qkv,d,state,glue='current' if name=='fa3_current' else 'best5')
            us,out,g=timed(call,data,flush,rounds=7)
            expected=gate_prepare(data)
            ok,msg,err=gate_cached(data,out,expected)
            # Modify values, then verify the captured graph used them.
            saved=raw.clone()
            raw.add_(.015625)
            data['qkv']=(raw.sum(0) if count else raw).to(torch.bfloat16).view(1,8,2304)
            g.replay();torch.cuda.synchronize();fresh=gate_prepare(data)
            fok,fmsg,ferr=gate_cached(data,out,fresh)
            lines.append(f'ROW partials={count} kernel={name} us={us:.2f} gate={"P" if ok and fok else "F"} err={err:.6f} fresh_err={ferr:.6f} graph=P')
            if not(ok and fok):lines.append('GATE_ERROR '+msg+'; '+fmsg)
            raw.copy_(saved)
            data['qkv']=(raw.sum(0) if count else raw).to(torch.bfloat16).view(1,8,2304)
    return lines
