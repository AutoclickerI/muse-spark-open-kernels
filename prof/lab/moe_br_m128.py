import torch
import torch as rv_torch
import torch
import torch as rv_torch
import torch as hv_torch
import torch as rr_torch
import math as fx_math
import statistics as fx_statistics
import time as fx_time
from typing import Callable as fx_Callable
import torch as fx_torch
fx_HIDDEN = 8192
fx_D_EXPERT = 4096
fx_N_EXPERTS = 256
fx_LOCAL_EXPERTS = 32
fx_TOP_K = 8
fx_ROUTER_MULTIPLIER = 0.1767766952966369
fx_RMS_EPS = 1e-05
fx_STREAM_SEED = 5354
fx_STREAM_MEAN = 0.7
fx_STREAM_SHARED = 0.9
fx_STREAM_RANK = 4
fx_DRAFT_SHARE = 0.6

def fx__rms(x: fx_torch.Tensor) -> fx_torch.Tensor:
    xf = x.float()
    return xf * fx_torch.rsqrt(xf.square().mean(-1, keepdim=True) + fx_RMS_EPS)

def fx__basis(device: fx_torch.device) -> tuple[fx_torch.Tensor, fx_torch.Tensor]:
    gen = fx_torch.Generator().manual_seed(fx_STREAM_SEED)
    mean = fx_torch.randn((fx_HIDDEN,), generator=gen).to(device)
    shared = fx_torch.randn((fx_STREAM_RANK, fx_HIDDEN), generator=gen).to(device)
    return (mean, shared)

def fx_make_fixture(weights: dict[str, fx_torch.Tensor], tokens: int, seed: int, *, mode: str='real', q_len: int | None=None) -> dict:
    """Return an expert-call fixture with real router or controlled route IDs.

    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.
    Pass it explicitly for other loads. Draft sharing applies within every verify block.
    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps
    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every
    slot to a uniformly chosen local expert, matching older synthetic tuners.
    """
    device = weights['router'].device
    gen = fx_torch.Generator(device=device).manual_seed(seed)
    mean, shared = fx__basis(device)
    coef = fx_torch.randn((tokens, fx_STREAM_RANK), generator=gen, device=device)
    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)
    if q_len < 1 or tokens % q_len:
        raise ValueError(f'tokens={tokens} must be divisible by q_len={q_len}')
    scale = fx_torch.ones((tokens // q_len, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = fx_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = scale * (fx_STREAM_MEAN * mean + fx_math.sqrt(fx_STREAM_SHARED / fx_STREAM_RANK) * (coef @ shared)) + (1 - fx_STREAM_SHARED * scale.square()).sqrt() * fx_torch.randn((tokens, fx_HIDDEN), generator=gen, device=device)
    h = fx__rms(stream) * (1.0 + weights['pre_ffn_norm'].float())
    router = weights['router'].float()
    biased = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER) + weights['router_bias'].float()
    scores = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER)
    global_ids = biased.topk(fx_TOP_K, dim=-1).indices.to(fx_torch.int32)
    picked = scores.gather(1, global_ids.long())
    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)
    ranges = fx_torch.bincount((global_ids // fx_LOCAL_EXPERTS).flatten(), minlength=fx_N_EXPERTS // fx_LOCAL_EXPERTS)
    local_start = fx_LOCAL_EXPERTS * int(ranges.argmax())
    local = ((global_ids - local_start) % fx_N_EXPERTS).to(fx_torch.int32)
    local_mask = local < fx_LOCAL_EXPERTS
    if mode == 'uniform':
        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)
    elif mode == 'matched_uniform':
        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)
        lids = fx_torch.where(local_mask, lids, fx_torch.full_like(lids, -1))
    elif mode == 'real':
        lids = fx_torch.where(local_mask, local, fx_torch.full_like(local, -1))
    else:
        raise ValueError(mode)
    x = h @ weights['pooler'].float().t()
    x = (fx__rms(x) * (1.0 + weights['pre_expert_norm'].float())).to(fx_torch.bfloat16)
    tw = topk_weights if mode != 'uniform' else fx_torch.rand((tokens, fx_TOP_K), generator=gen, device=device)
    if mode == 'uniform':
        tw = tw / tw.sum(-1, keepdim=True)
    cfg = {'local_count': 32, 'd_hidden': fx_D_EXPERT, 'd_expert': fx_D_EXPERT, 'local_start': 0}
    return {'hidden_states': x, 'topk_ids': lids, 'topk_weights': tw.float(), 'config': cfg, 'local_start': local_start, 'global_ids': global_ids, 'local_mask': local_mask}
import math as rg_math
import functools as rg_functools
import torch as rg_torch
rg_HIDDEN = 8192
rg_Q_SIZE = 1024
rg_QKV_SIZE = 2304
rg_D_EXPERT = 4096
rg_LOCAL_EXPERTS = 32
rg_N_EXPERTS = 256
rg_STREAM_SEED = 5354
rg_STREAM_RANK = 4
rg_ROUTER_STD = 0.045
rg_ROUTER_ALIGN = 0.07
rg_ROUTER_BIAS_STD = 0.055
rg_BLOCK = 16
rg_E4M3_MAX = 448.0
rg_SCALE_LOG2_MEAN = 7.0
rg_SCALE_LOG2_STD = 0.4

def rg_random_weights(rows: int, k: int, *, generator: rg_torch.Generator, device: str | rg_torch.device, leading: tuple[int, ...]=(), wide_scales: bool=False):
    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.

    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)
    clamped to [8, 448]; the global scale
    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448
    instead.
    """
    shape = (*leading, rows, k // 2)
    packed = rg_torch.randint(0, 256, shape, dtype=rg_torch.uint8, generator=generator, device=device)
    scale_shape = (*leading, rows, k // rg_BLOCK)
    if wide_scales:
        u = rg_torch.rand(scale_shape, generator=generator, device=device)
        sf = rg_torch.exp2(-6 + u * (rg_math.log2(rg_E4M3_MAX) + 6)).to(rg_torch.float8_e4m3fn)
        mean_sq = float(sf.float().pow(2).mean())
    else:
        z = rg_torch.randn(scale_shape, generator=generator, device=device)
        sf = rg_torch.exp2(rg_SCALE_LOG2_MEAN + rg_SCALE_LOG2_STD * z).clamp(8.0, rg_E4M3_MAX).to(rg_torch.float8_e4m3fn)
        mean_sq = 4.0 ** rg_SCALE_LOG2_MEAN * rg_math.exp(2 * (rg_math.log(2) * rg_SCALE_LOG2_STD) ** 2)
    gs = rg_torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=rg_torch.float32, device=device)
    return (packed, sf, gs)

def rg_generate_weights(seed: int, device: str='cuda', router_seed: int | None=None):
    """The layer's weights from `seed`. With router_seed, the router and its bias come from that
    seed instead."""
    gen = rg_torch.Generator(device=device).manual_seed(seed)

    def dense(n, k, dtype=rg_torch.bfloat16):
        return (rg_torch.randn((n, k), generator=gen, device=device) / rg_math.sqrt(k)).to(dtype)

    def vec(n, std):
        return (std * rg_torch.randn((n,), generator=gen, device=device)).to(rg_torch.bfloat16)
    w13, w13_sf, gs13 = rg_random_weights(2 * rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))
    w2, w2_sf, gs2 = rg_random_weights(rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))
    gs13 = gs13 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))
    gs2 = gs2 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))
    o, attn_gate, pre_ffn = (dense(rg_HIDDEN, rg_Q_SIZE), vec(rg_HIDDEN, 0.3), vec(rg_HIDDEN, 0.1))
    weights = {'qkv_gate': dense(rg_QKV_SIZE, rg_HIDDEN), 'o': o, 'attn_residual_gate': attn_gate, 'pre_ffn_norm': pre_ffn, **rg__router(gen, device, pre_ffn), 'pooler': dense(rg_D_EXPERT, rg_HIDDEN), 'pre_expert_norm': vec(rg_D_EXPERT, 0.1), 'w13': w13, 'w13_sf': w13_sf, 'w13_gs': gs13, 'w2': w2, 'w2_sf': w2_sf, 'w2_gs': gs2, 'post_expert_norm': vec(rg_D_EXPERT, 0.1), 'unpooler': dense(rg_HIDDEN, rg_D_EXPERT), 'post_ffn_norm': vec(rg_HIDDEN, 0.1), 'ffn_residual_gate': vec(rg_HIDDEN, 0.3), 'next_pre_attn_norm': vec(rg_HIDDEN, 0.1)}
    if router_seed is not None:
        weights.update(rg__router(rg_torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))
    return weights

@rg_functools.cache
def rg__stream_basis(device: str):
    """The residual stream's mean direction [8192] and shared directions [STREAM_RANK, 8192]."""
    gen = rg_torch.Generator().manual_seed(rg_STREAM_SEED)
    mean = rg_torch.randn((rg_HIDDEN,), generator=gen)
    shared = rg_torch.randn((rg_STREAM_RANK, rg_HIDDEN), generator=gen)
    return (mean.to(device), shared.to(device))

def rg__router(gen, device, pre_ffn):
    """FP32 router rows, each with a component along the stream's mean direction, and the router
    bias. pre_ffn is unused."""
    mean, _ = rg__stream_basis(str(device))
    w = rg_ROUTER_STD * rg_torch.randn((rg_N_EXPERTS, rg_HIDDEN), generator=gen, device=device)
    align = rg_ROUTER_ALIGN * rg_torch.randn((rg_N_EXPERTS, 1), generator=gen, device=device)
    rows = w + align * mean[None, :] / rg_math.sqrt(rg_HIDDEN)
    bias = rg_ROUTER_BIAS_STD * rg_torch.randn((rg_N_EXPERTS,), generator=gen, device=device)
    return {'router': rows.contiguous(), 'router_bias': bias.contiguous()}
CAPTURED_R0 = 'eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK'
import base64
import statistics
import time
import zlib
import torch
import triton
import triton.language as tl
import ctypes as pf_ct
import hashlib as pf_hashlib
import logging as pf_logging
import math as pf_math
import os as pf_os
import statistics as pf_statistics
import time as pf_time
import torch as pf_torch
import torch.nn.functional as pf_F
import triton as pf_triton
import triton.language as pf_tl
import json as pf_json
import os as pf_os
from typing import TypedDict as pf_TypedDict
import torch as pf_torch
pf__saved_flag = pf_os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')
pf_os.environ['HUMMING_DISABLE_PARALLEL_BUILD'] = '1'
try:
    from humming import ops as pf_humming_ops
    from humming.config import GemmType as pf_GemmType, WeightScale2Type as pf_WeightScale2Type
    from humming.kernel.humming import HummingKernel as pf_HummingKernel
    from humming.layer import HummingMethod as pf_HummingMethod
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema as pf_ModeloptNvfp4WeightSchema
    from humming.tune import get_heuristics_config as pf_get_heuristics_config
finally:
    if pf__saved_flag is None:
        pf_os.environ.pop('HUMMING_DISABLE_PARALLEL_BUILD', None)
    else:
        pf_os.environ['HUMMING_DISABLE_PARALLEL_BUILD'] = pf__saved_flag
from sglang.kernels.ops.activation.activation import silu_and_mul as pf_silu_and_mul
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as pf_fused_post_expert_norm_weighted_sum
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as pf_moe_align_block_size
import ctypes as pf_ct
import hashlib as pf_hashlib
import logging as pf_logging
import os as pf_os
import statistics as pf_statistics
import time as pf_time
import torch as pf_torch
import torch.nn.functional as pf_F
import triton as pf_triton
import triton.language as pf_tl

class MWGraph:

    def __init__(self, fn, data):
        self.data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}
        fn(self.data)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn(self.data)

    def refresh(self, data):
        for k in ('hidden_states', 'topk_ids', 'topk_weights'):
            self.data[k].copy_(data[k])

    def replay(self):
        self.graph.replay()
        return self.out
import base64
import hashlib
import statistics
import time
import zlib

def dc_weights(seed, device):
    w = rg_generate_weights(seed, device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 1000003)
    w['w13_gs_gate'] = w['w13_gs'].clone()
    w['w13_gs_up'] = w['w13_gs'] * torch.exp(0.55 * torch.randn(32, generator=gen, device=device))
    w['post_norm_w'] = w['post_expert_norm']
    assert bool((w['w13_gs_gate'] != w['w13_gs_up']).all())
    return w

def dc_oracles(ds, w):
    """Independent FP32 scratch oracle; no expanded bank survives this function."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        wants = [torch.zeros_like(d['hidden_states'], dtype=torch.float32) for d in ds]
        lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=w['w13'].device)
        for e in range(32):
            active = [torch.where(d['topk_ids'] == e) for d in ds]
            if not any((t.numel() for t, _ in active)):
                continue

            def dq(key):
                q = w[key][e]
                c = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
                return lut[c & 7] * torch.where(c & 8 != 0, -1.0, 1.0) * w[key + '_sf'][e].float().repeat_interleave(16, -1)
            w13 = dq('w13')
            w2 = dq('w2') * w['w2_gs'][e]
            for d, out, (t, j) in zip(ds, wants, active):
                if t.numel() == 0:
                    continue
                h = d['hidden_states'][t].float() @ w13.t()
                g = h[:, :4096] * w['w13_gs_gate'][e]
                u = h[:, 4096:] * w['w13_gs_up'][e]
                a = (torch.nn.functional.silu(g) * u).bfloat16()
                y = a.float() @ w2.t()
                y *= 1 + w['post_norm_w'].float()
                y *= torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-08)
                out.index_add_(0, t, y * d['topk_weights'][t, j, None])
        return [out.bfloat16() for out in wants]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old

def dc_fixtures(w, m, r0=False):
    ds = [fx_make_fixture(w, m, 9000 + m + 7919 * j) for j in range(3)]
    if r0:
        raw = zlib.decompress(base64.b64decode(CAPTURED_R0))
        ms = (8, 16, 32, 128, 256)
        offset = 8 * sum(ms[:ms.index(m)])
        ids = torch.tensor(list(raw[offset:offset + m * 8]), device=w['w13'].device, dtype=torch.int32).view(m, 8)
        ids = torch.where(ids < 32, ids, -1)
        ds = [dict(d, topk_ids=ids.clone()) for d in ds]
    return ds
import functools
import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl
import functools
import hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline
import statistics
import time
import functools
import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl
import functools
import hashlib
from pathlib import Path
import torch
import triton
from torch.utils.cpp_extension import load_inline
import functools
import hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline
import functools
import hashlib
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import statistics
import time
import functools
import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl
import functools
import hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline
import functools
import hashlib
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import time
import statistics
import functools
import hashlib
import statistics
import time
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline
"""H_H M128 (H200 770809cf recipe): NT1<=16, WGMMA32>16 in both phases; packed uint8 scale codes (S7 kernel)."""

def m128_hh_mw_repack_planes(q):
    """A permutation of all four bits of every nibble, matching RS register ownership.

    Two words encode the 16 codes each lane needs for M16/K32. The extraction
    table is the incumbent V9 bit-plane bijection, used here with WGMMA layout.
    """
    e, n, k2 = q.shape
    k = k2 * 2
    dst = torch.empty((e, n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int32)
    slots = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
    mask = 1613783088
    for expert in range(e):
        codes = torch.stack((q[expert] & 15, q[expert] >> 4), -1).reshape(n // 16, 2, 8, k // 32, 2, 2, 4, 2)
        codes = codes.permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 32, 32, 8, 2).to(torch.int64)
        words = torch.zeros((n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int64)
        for slot, (src, rot) in enumerate(slots):
            for half in (0, 1):
                for bit, dbit in enumerate((6, 7, 8, 15)):
                    p = (dbit + 16 * half + rot) % 32
                    word = (0 if mask >> p & 1 else 1) if src == 'W' else 0 if src == 'A' else 1
                    words[..., word] |= (codes[..., slot, half] >> bit & 1) << p
        dst[expert] = words.to(torch.int32)
    return dst

def m128_hh_issue_prepare(w):
    bank = {}
    for key in ('w13', 'w2'):
        q, sf = (w[key], w[key + '_sf'])
        e, n, k16 = sf.shape
        planes = []
        scales = []
        for expert in range(e):
            code, neg, zero = m128_hh_encode_scales(sf[expert])
            folded = q[expert].reshape(n, k16, 8) ^ neg.to(torch.uint8)[..., None] * 136
            folded = torch.where(zero[..., None], torch.zeros_like(folded), folded)
            planes.append(m128_hh_mw_repack_planes(folded.reshape(1, n, -1)))
            scales.append(code)
        bank[key] = torch.cat(planes)
        sfbytes = torch.stack(scales)
        bank[key + '_sf'] = sfbytes.reshape(e, n // 16, 2, 8, k16 // 2, 2).permute(0, 1, 4, 3, 5, 2).contiguous()
    bank['w13_gs'] = torch.cat((w.get('w13_gs_gate', w['w13_gs']), w.get('w13_gs_up', w['w13_gs']))).float().contiguous()
    bank['w2_gs'] = w['w2_gs'].float().contiguous()
    bank['pnw'] = w['post_norm_w'].contiguous()
    ng = bank['w13'].shape[1]
    rr = torch.arange(ng // 2, device=w['w13'].device).reshape(-1, 2)
    order = torch.cat((rr, rr + ng // 2), 1).flatten()
    bank['w13'] = bank['w13'].index_select(1, order).contiguous()
    bank['w13_sf'] = bank['w13_sf'].index_select(1, order).contiguous()
    return bank
M128_HH_H = 4096
M128_HH_NLOC = 32
M128_HH_TILE_BYTES = 36864
M128_HH_NT1, M128_HH_NT2 = (512, 256)
M128_HH_NCNT = 64
M128_HH_DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
M128_HH_LA_MASK = 1613783088

def m128_hh_bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if M128_HH_LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(M128_HH_DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
M128_HH_TABLE = m128_hh_bit_table()

def m128_hh_encode_scales(sf: torch.Tensor):
    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),
    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for
    every nonzero finite E4M3 value (subnormals included)."""
    s = sf.float()
    if not torch.isfinite(s).all():
        raise ValueError('non-finite NVFP4 block scale')
    a = s.abs()
    nz = a > 0
    m, e = torch.frexp(torch.where(nz, a, torch.ones_like(a)))
    mant = ((m * 2 - 1) * 8).round().to(torch.int32)
    e5 = (e - 1 + 21).to(torch.int32)
    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and (int(mant.max()) <= 7)
    code = torch.where(nz, e5 << 3 | mant, torch.zeros_like(e5))
    back = torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)
    assert torch.equal(torch.where(nz, back, torch.zeros_like(back)), a)
    return (code.to(torch.uint8), s < 0, ~nz)

def m128_hh_repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = m128_hh_encode_scales(sf)
    codes = torch.where(neg[..., None], codes ^ 8, codes)
    codes = torch.where(zero[..., None], torch.zeros_like(codes), codes)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)
    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)
    for k, h, b, w, p in M128_HH_TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return torch.cat((wbytes, sbytes), 1).contiguous()

def m128_hh_issue_nt_prepare(weights):
    dev = weights['w13'].device
    t = torch.arange(M128_HH_NT1, device=dev)
    i = torch.arange(8, device=dev)
    rowmap1 = torch.cat((8 * t[:, None] + i, M128_HH_H + 8 * t[:, None] + i), 1)
    rowmap2 = torch.arange(M128_HH_H, device=dev).view(M128_HH_NT2, 16)
    w13p = torch.empty((M128_HH_NLOC, M128_HH_NT1, M128_HH_TILE_BYTES), dtype=torch.uint8, device=dev)
    w2p = torch.empty((M128_HH_NLOC, M128_HH_NT2, M128_HH_TILE_BYTES), dtype=torch.uint8, device=dev)
    for e in range(M128_HH_NLOC):
        w13p[e] = m128_hh_repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = m128_hh_repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = m128_hh_pm_module()
    props = torch.cuda.get_device_properties(dev)
    torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:M128_HH_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:M128_HH_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': torch.zeros((M128_HH_NCNT,), dtype=torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}
M128_HH_PM_CUDA = '#include <cuda_runtime.h>\n#include <vector>\n#include <stdexcept>\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\n// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\n\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace idist {\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\n}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out){\n unsigned w,mask=0x81c081c0u;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));\n // uint8 code c -> BF16 bits (c<<4)+0x7000 in both halves, formed in registers.\n unsigned s0=__byte_perm(raw,0u,0x4040u)*16u+0x70007000u;\n unsigned s1=__byte_perm(raw,0u,0x4141u)*16u+0x70007000u;\n unsigned s2=__byte_perm(raw,0u,0x4242u)*16u+0x70007000u;\n unsigned s3=__byte_perm(raw,0u,0x4343u)*16u+0x70007000u;\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==32){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\n#pragma unroll\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n}\ntemplate<int NT,int KT>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n int t=threadIdx.x,w=t>>5,l=t&31;\n #pragma unroll\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n  if(l<16)async_copy(q+(g*128+w*32)*8+l*16,Q+z*64+l*4,16,1);\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if constexpr(S>3){\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n  else if constexpr(S>4){\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\n   else if constexpr(S>5){\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\n    else if constexpr(S>6){\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\n     else if constexpr(S>7){\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\n__device__ __forceinline__ int publish(int*p){\n int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;\n}\n__global__ void route_init(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,\n int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){\n int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;\n for(int i=all;i<clear_count;i+=step)state[i]=0;\n for(int i=all;i<M*4096/8;i+=step)reinterpret_cast<uint4*>(out)[i]=make_uint4(0,0,0,0);\n if(blockIdx.x)return;\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\n for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;\n  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}\n }\n for(int token=t;token<M;token+=256){int n=0;\n  #pragma unroll\n  for(int j=0;j<8;++j){int e=ids[token*8+j]-start;n+=(e>=0&&e<32);}\n  state[clear_count+token]=n;\n }\n __syncthreads();\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\n  int scan=pad;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\n  int key=n*32+31-t;\n  #pragma unroll\n  for(int k=2;k<=32;k*=2){\n   #pragma unroll\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\n   }\n  }\n  // Descending keys after reversing the ascending bitonic result.\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\n }\n}\ntemplate<int NT,int KT,int S,int SPLIT,bool FIRST,int MODE>\n__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,\n float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start,bool profiling){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(128) unsigned char rawsm[];\n unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);\n __shared__ int last,flags[NT];\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;\n int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;\n \n int total=TASK[32]*MT*SPLIT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  uint64_t tick=profiling?clock64():0;uint64_t times[6]={};bool islast=false;\n  int task=unit/SPLIT,split=unit%SPLIT,group=task/MT,mt=task%MT,ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\n  int src[NT/16];\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  #pragma unroll\n  for(int i=0;i<min(S-1,NS);++i)\n   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\n  // K64 has one MMA group per stage. Alternate register-A banks across\n  // stages so wait1 retires a bank before its next conversion overwrites it.\n  // K128/K256 already alternate within each stage; their schedule is unchanged.\n  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");\n  #pragma unroll 1\n  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){\n   #pragma unroll\n   for(int ksub=0;ksub<(KT==64?2:1);++ksub){\n   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll 1\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\n    #pragma unroll\n    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;\n     #pragma unroll\n     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n      unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[reg]+g*8);}\n     keep(af[reg]);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n     mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);keep(af[reg^1]);\n     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;\n      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);}\n    }\n   }\n   }\n  }\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  if(profiling){uint64_t now=clock64();times[0]=now-tick;tick=now;}\n  if constexpr(SPLIT>1){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];\n   }\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();\n   if(!last)continue;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;\n    if(r<count){\n     #pragma unroll\n     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];\n    }\n    acc[j]=sum;\n   }\n  }\n  if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n   }\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n   if constexpr(MODE!=1){\n   if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();\n   if(profiling){uint64_t now=clock64();times[2]=now-tick;tick=now;}\n   islast=last;\n   if(last){\n    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];\n     #pragma unroll\n     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);\n    }\n    __threadfence();__syncthreads();\n    if(profiling){uint64_t now=clock64();times[3]=now-tick;tick=now;}\n    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;\n      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}\n    __syncthreads();\n    if(profiling){uint64_t now=clock64();times[4]=now-tick;tick=now;}\n    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;\n     if constexpr(MODE==2){\n     int base[8];float alpha[8];\n     #pragma unroll\n     for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;base[j]=-1;alpha[j]=0;\n      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}\n     for(int c=l*4;c<4096;c+=128){float4 sum=make_float4(0,0,0,0);\n      #pragma unroll\n      for(int j=0;j<8;++j)if(base[j]>=0){float4 z=*(const float4*)(Y+(size_t)base[j]*4096+c);float a=alpha[j];sum.x+=z.x*a;sum.y+=z.y*a;sum.z+=z.z*a;sum.w+=z.w*a;}\n      __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n      *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n     }\n     }else{\n     for(int c=l;c<4096;c+=32){float sum=0;\n      #pragma unroll\n      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;\n       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}\n      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);\n     }\n     }\n    }\n   }\n   }else if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n  }\n  __syncthreads();\n  if(profiling&&t==0){uint64_t*p=(uint64_t*)PART+(size_t)unit*8;\n   for(int i=0;i<5;++i)p[i]=times[i];p[5]=clock64()-tick;p[6]=islast;p[7]=clock64();}\n }\n}\n#define FX_CASES(F) \\\n F(0,16,128,4,1) F(1,32,128,4,1)\n}\nint64_t idist_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n #define INIT(ID,NT,KT,S,SK) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto set=[&]<bool F,int MODE>(){WX_CUDA_CHECK(cudaFuncSetAttribute(idist::fused<NT,KT,S,SK,F,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<true,1>();set.template operator()<false,1>();if constexpr(ID==0)set.template operator()<false,0>();}\n FX_CASES(INIT)\n #undef INIT\n return opt;\n}\nstd::vector<int64_t> idist_info(int64_t which,int64_t first,int64_t mode){\n #define INFO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F,int MODE>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,idist::fused<NT,KT,S,SK,F,MODE>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,idist::fused<NT,KT,S,SK,F,MODE>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(first)return info.template operator()<true,1>();if(mode==1)return info.template operator()<false,1>();if constexpr(ID==0){WX_CHECK(mode==0);return info.template operator()<false,0>();}else WX_CHECK(false,"legacy control exists only forNT16");}\n switch(which){FX_CASES(INFO) default:WX_CHECK(false,"unknown fixed configuration");}\n #undef INFO\n}\nvoid idist_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear_count,int64_t stream){\n idist::route_init<<<128,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear_count);WX_CUDA_CHECK(cudaGetLastError());\n}\nvoid idist_run(std::vector<int64_t> v,int64_t first,int64_t which,int64_t grid,int64_t start,int64_t mode,int64_t profiling,int64_t M,int64_t cap,int64_t stream_value){\n WX_CHECK(v.size()==20);auto stream=(cudaStream_t)stream_value;\n #define GO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2; \\\n auto launch=[&]<bool FIRST,int MODE>(){idist::fused<NT,KT,S,SK,FIRST,MODE><<<grid,128,bytes,stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(const __nv_bfloat16*)(void*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(const __nv_bfloat16*)(void*)v[14],(float*)v[10],(__nv_bfloat16*)(void*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)(void*)v[19],M,cap,start,profiling);};if(first)launch.template operator()<true,1>();else if(mode==1)launch.template operator()<false,1>();else{WX_CHECK(mode==0&&which==0);if constexpr(ID==0)launch.template operator()<false,0>();}break;}\n switch(which){FX_CASES(GO) default:WX_CHECK(false,"unknown fixed configuration");}\n #undef GO\n WX_CUDA_CHECK(cudaGetLastError());\n}\nnamespace idist {\n__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\n int tok=blockIdx.x,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\n __syncthreads();int pp[8];float aa[8];\n #pragma unroll\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\n for(int c=t*4;c<4096;c+=1024){float4 sum=make_float4(0,0,0,0);\n  #pragma unroll\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n }\n}\n}\nvoid idist_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){idist::finish<<<M,256,0,(cudaStream_t)stream>>>((int*)v[5],(int*)v[11],(float*)v[12],(int*)v[13],(float*)v[15],(float*)v[16],(__nv_bfloat16*)(void*)v[19],M,start);WX_CUDA_CHECK(cudaGetLastError());}\n\n#include <cuda_runtime.h>\n#include <vector>\n#include <stdexcept>\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\n\n\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace pmnt {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  uint32_t W;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(W):"r"(A),"r"(B));\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate<int NT,bool FIRST,bool SMALL>\n__global__ void __launch_bounds__(NTHR,2) phase(\n const __nv_bfloat16*x,const __nv_bfloat16*pnw,const uint8_t*w13p,const uint8_t*w2p,\n const float*gs13,const float*gs2,const int*C,const int*OFF,const int*SLOTS,\n __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){\n constexpr int NB=8*NT,RED=NT*4;\n extern __shared__ __align__(128) uint8_t smem[];\n uint8_t*stages=smem;float*red=(float*)(smem+nstage*TILE_BYTES);\n uint64_t*fullb=(uint64_t*)(red+2*NCW*32*RED),*emptyb=fullb+nstage;\n Route*rt=(Route*)(emptyb+nstage);\n const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;\n if(tid==0){for(int z=0;z<nstage;++z){mbar_init(fullb+z,1);mbar_init(emptyb+z,NCW);}\n  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\n if(warp==0){int n=C[lane];int nb=(n+NB-1)/NB;if constexpr(SMALL)if(n>16)nb=0;\n  rt->n[lane]=n;rt->off[lane]=OFF[lane];rt->nblk[lane]=nb;\n  int u1=nb*NT1,u2=nb*NT2;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,u1,d),b=__shfl_up_sync(~0u,u2,d);if(lane>=d){u1+=a;u2+=b;}}\n  rt->us1[lane+1]=u1;rt->us2[lane+1]=u2;\n  if(lane==0){rt->us1[0]=rt->us2[0]=0;}\n  int U1=__shfl_sync(~0u,u1,31),U2=__shfl_sync(~0u,u2,31);\n  if(lane==0){rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));}\n }\n __syncthreads();\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=FIRST?b1-a1:0,total_count=FIRST?first_count:b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = FIRST?0:1; pass < (FIRST?1:2); ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        cur_pass=pass;cur_e=e;cur_nb=nb;seg_tiles=0;\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(SLOTS[e*M+idx] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= gs13[e+32]*256.f;\n            G1=__bfloat162float(__float2bfloat16_rn(G1));\n            U=__bfloat162float(__float2bfloat16_rn(U));\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v*fw[0];\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + (size_t)(off+idx)*64+(m>>2), t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\n\n\nint64_t pm_nt_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n auto set=[&]<bool F,bool S>(){cudaFuncAttributes a{};WX_CUDA_CHECK(cudaFuncGetAttributes(&a,pmnt::phase<1,F,S>));WX_CUDA_CHECK(cudaFuncSetAttribute(pmnt::phase<1,F,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));};\n set.template operator()<true,false>();set.template operator()<false,false>();set.template operator()<true,true>();set.template operator()<false,true>();return opt;}\nint64_t pm_nt_smem(int64_t stages,int64_t M){return pmnt::smem_bytes(1,stages,M);}\nint64_t pm_nt_resident(int64_t stages,int64_t M,int64_t first,int64_t small){int n;\n auto get=[&]<bool F,bool S>(){WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmnt::phase<1,F,S>,pmnt::NTHR,pmnt::smem_bytes(1,stages,M)));};\n if(first){if(small)get.template operator()<true,true>();else get.template operator()<true,false>();}else{if(small)get.template operator()<false,true>();else get.template operator()<false,false>();}return n;}\nvoid pm_nt_run(std::vector<int64_t>v,int64_t first,int64_t small,int64_t M,int64_t stages,int64_t grid,int64_t stream){\n auto launch=[&]<bool F,bool S>(){pmnt::phase<1,F,S><<<grid,pmnt::NTHR,pmnt::smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\n (__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(uint8_t*)v[2],(uint8_t*)v[3],(float*)v[4],(float*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11],M,stages);};\n if(first){if(small)launch.template operator()<true,true>();else launch.template operator()<true,false>();}else{if(small)launch.template operator()<false,true>();else launch.template operator()<false,false>();}WX_CUDA_CHECK(cudaGetLastError());}\n\n\n#include <cuda_runtime.h>\n#include <vector>\n__global__ void pm_meta_kernel(const int*C,const int*ORDER,int*T32,int*T128,int*T16,float*SS,int cap){\n int t=threadIdx.x,idx=blockIdx.x*blockDim.x+t;\n for(int i=idx;i<cap*64;i+=gridDim.x*blockDim.x)SS[i]=0.f;\n if(blockIdx.x==0&&t<32){int n=C[ORDER[t]],v32=n>16?(n+31)/32:0,v128=(n+127)/128,v16=(n+15)/16;\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v32,d),b=__shfl_up_sync(~0u,v128,d),c=__shfl_up_sync(~0u,v16,d);if(t>=d){v32+=a;v128+=b;v16+=c;}}\n  T32[t+1]=v32;T128[t+1]=v128;T16[t+1]=v16;if(t==0){T32[0]=0;T128[0]=0;T16[0]=0;}\n }\n}\nvoid pm_meta(std::vector<int64_t>v,int64_t cap,int64_t stream){pm_meta_kernel<<<128,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(float*)v[5],cap);}\n'
M128_HH_PM_CPP = '#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\nint64_t idist_init();std::vector<int64_t> idist_info(int64_t,int64_t,int64_t);void idist_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void idist_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void idist_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);\nint64_t pm_nt_init();int64_t pm_nt_smem(int64_t,int64_t);int64_t pm_nt_resident(int64_t,int64_t,int64_t,int64_t);void pm_nt_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void pm_meta(std::vector<int64_t>,int64_t,int64_t);\n'
M128_HH_PM_FUNCTIONS = ['idist_init', 'idist_info', 'idist_route', 'idist_run', 'idist_finish', 'pm_nt_init', 'pm_nt_smem', 'pm_nt_resident', 'pm_nt_run', 'pm_meta']
M128_HH_ISSUE_OPTS = {}

@functools.lru_cache(None)
def m128_hh_pm_module():
    arch = '90a' if torch.cuda.get_device_capability() == (9, 0) else '120'
    mod = load_inline(name='m128_phase_mix_' + hashlib.sha256(M128_HH_PM_CUDA.encode()).hexdigest()[:12], cpp_sources=M128_HH_PM_CPP, cuda_sources=M128_HH_PM_CUDA, functions=M128_HH_PM_FUNCTIONS, no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_' + arch + ',code=sm_' + arch], verbose=False)
    for p in ('idist', 'pm_nt'):
        M128_HH_ISSUE_OPTS[p] = getattr(mod, p + '_init')()
    return mod

def m128_hh_pm_plan(data, base, native):
    x = data['hidden_states']
    m = x.shape[0]
    dev = x.device
    nt = 32
    cap = m * 8 + 32 * 128
    c = torch.empty(32, device=dev, dtype=torch.int32)
    off = torch.empty(33, device=dev, dtype=torch.int32)
    order = torch.empty_like(c)
    tasks = [torch.empty_like(off) for _ in range(4)]
    slots = torch.empty((32, m), device=dev, dtype=torch.int32)
    ids = data['topk_ids'].int().contiguous()
    tw = data['topk_weights'].float().contiguous()
    pos = torch.empty_like(ids)
    clear = cap // nt * 193 + m
    state = torch.empty(clear + m, device=dev, dtype=torch.int32)
    act = torch.empty((cap, 4096), device=dev, dtype=torch.bfloat16)
    part = torch.empty(1, device=dev)
    y = torch.empty((cap, 4096), device=dev)
    ss = torch.empty((cap, 64), device=dev)
    inv = torch.empty(cap, device=dev)
    out = torch.empty_like(x)
    common = [c, off, order, tasks[0], slots, act]
    tail = [ids, tw, pos, base['pnw'], y, ss, inv, state, out]

    def vector(bank, phase, task):
        v = [bank['w13' if phase else 'w2'], bank['w13_sf' if phase else 'w2_sf'], bank['w13_gs' if phase else 'w2_gs'], x if phase else act] + common + [part] + tail
        v[7] = task
        return v
    word13 = vector(base, 1, tasks[0])
    word2 = vector(base, 0, tasks[0])
    big13 = vector(base, 1, tasks[1])
    big2 = vector(base, 0, tasks[1])
    nv = [x, base['pnw'], native['w13p'], native['w2p'], base['w13_gs'], base['w2_gs'], c, off, slots, act, y, ss]
    vectors = [word13, word2, big13, big2, nv]
    ptrs = [[v.data_ptr() for v in row] for row in vectors]
    rp = [v.data_ptr() for v in (ids, c, off, order, tasks[0], slots, pos, state, out)]
    mp = [v.data_ptr() for v in (c, order, tasks[1], tasks[2], tasks[3], ss)]
    mod = m128_hh_pm_module()
    start = data['config'].get('local_start', 0)
    owners = (vectors, ids, tw, tasks)
    stages = 2
    while stages > 1 and mod.pm_nt_smem(stages, m) > M128_HH_ISSUE_OPTS['pm_nt']:
        stages -= 1

    def stream():
        return torch.cuda.current_stream().cuda_stream

    def route():
        _ = owners
        mod.idist_route(rp, m, nt, start, clear, stream())
        mod.pm_meta(mp, cap, stream())

    def phase(which):
        _ = owners
        small = 1
        grid = native['grid'] * mod.pm_nt_resident(stages, m, which, small)
        mod.pm_nt_run(ptrs[4], which, small, m, stages, grid, stream())
        mod.idist_run(ptrs[2 if which else 3], which, 1, 1056, start, 1, 0, m, cap, stream())

    def w13():
        phase(1)

    def w2():
        phase(0)

    def finish():
        _ = owners
        mod.idist_finish(ptrs[1], m, start, stream())
        return out
    return dict(steps=(route, w13, w2, finish), out=out, owners=owners, v13=word13, v2=word2, tasks=tasks)

def m128_hh_pm_call(data, base, native):
    for fn in m128_hh_pm_plan(data, base, native)['steps']:
        out = fn()
    return out

def hh_prepare(weights):
    w = dict(weights, post_norm_w=weights.get('post_expert_norm', weights.get('post_norm_w')))
    base = m128_hh_issue_prepare(w)
    return dict(base=base, native=m128_hh_issue_nt_prepare(w))

def hh_experts(x, scores, local_ids, state, local_start=0):
    assert x.shape[0] == 128, 'H_H cell is selected only for M128'
    data = dict(hidden_states=x, topk_ids=local_ids, topk_weights=scores, config=dict(local_start=local_start))
    return m128_hh_pm_call(data, state['base'], state['native'])
BR_HH_CUDA = '#define BR_KINDS(F) F(0,32,true,1,0,0,0) F(1,32,false,1,0,0,0) F(4,32,true,1,2,16,32) F(5,16,true,1,2,0,16) F(8,32,false,1,2,16,32) F(9,16,false,1,2,0,16) F(12,16,false,1,1,8,16) F(20,32,true,2,0,0,0) F(21,32,false,2,0,0,0) F(26,32,true,3,0,0,0) F(28,32,false,4,0,0,0) F(29,32,true,5,0,0,0) F(30,32,true,7,0,0,0) F(31,32,false,6,0,0,0) F(33,32,false,10,0,0,0) F(44,32,true,67,0,0,0) F(45,32,false,66,0,0,0) F(46,32,true,71,0,0,0) F(47,32,false,70,0,0,0) F(48,32,false,130,0,0,0) F(50,64,true,3,0,0,0) F(51,64,false,2,0,0,0) F(52,32,true,259,0,0,0) F(53,32,false,258,0,0,0) F(54,32,true,1283,0,0,0) F(55,32,false,1282,0,0,0) F(56,32,true,1287,0,0,0) F(57,32,false,1286,0,0,0) F(40,32,true,19,0,0,0) F(41,32,false,18,0,0,0) F(42,32,true,35,0,0,0) F(43,32,false,34,0,0,0)\n// Branch/remainder lab kernels. Banks stay packed E2M1 plus uint8 exponent/mantissa scale codes\n// (sign/zero folded into the codes); scales convert to BF16 inside the timed kernel.\n// One task = one 32-row group of one expert x one 64-channel M tile (route NT=32 numbering).\n// A kernel instance owns the groups whose expert count (FM=1) or rows-in-group (FM=2) lies in (LO,HI].\n#include <cuda.h>\n#include <cudaTypedefs.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <vector>\n#include <stdexcept>\n#include <type_traits>\n#define BR_CUDA_CHECK(expr) do{cudaError_t br_s=(expr);if(br_s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(br_s));}while(0)\n#define BR_REQUIRE(c) do{if(!(c))throw std::runtime_error("branch lab argument check");}while(0)\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define BR_HOPPER 1\n#else\n#define BR_HOPPER 0\n#endif\n\nnamespace mwbr {\nconstexpr int BK=64,KT=128,S=4,BIG=1<<30;\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\ntemplate<bool PERM=false>__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out,unsigned scale_base=0x70007000u){\n unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;\n unsigned s0,s1,s2,s3;\n if constexpr(PERM){\n  unsigned even,odd;unsigned rot=rr(raw,8);\n  asm("lop3.b32 %0,%1,0x0ff00ff0,%2,0xea;":"=r"(even):"r"(raw),"r"(scale_base));\n  asm("lop3.b32 %0,%1,0x0ff00ff0,%2,0xea;":"=r"(odd):"r"(rot),"r"(scale_base));\n  s0=__byte_perm(even,even,0x1010);s2=__byte_perm(even,even,0x3232);\n  s1=__byte_perm(odd,odd,0x1010);s3=__byte_perm(odd,odd,0x3232);\n }else{\n  s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u;s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;\n  s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u;s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;\n }\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),\n  "r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT,bool DESC=false>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b,uint64_t desc=0){\n#if BR_HOPPER\n uint64_t d=DESC?desc:(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==8){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3},{%4,%5,%6,%7},%8,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n static_assert(NT==32,"tile width");\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n // Portable stand-in: identical shared layout and fragment ownership, immediate mma.sync.\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT,bool DESC=false,bool ADAPT=false>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs,uint64_t desc=0,int live=NT){\n if constexpr(ADAPT){\n  #pragma unroll\n  for(int g=0;g<NT/32;++g)if(live>g*32){\n   #pragma unroll\n   for(int b=0;b<4;++b)mma<32,DESC>(c+g*16,a+b*4,xs+g*32*64,b,desc+g*256);\n  }\n }else{\n#pragma unroll\n for(int b=0;b<4;++b){\n  if constexpr(NT==64){mma<32,DESC>(c,a+b*4,xs,b,desc);mma<32,DESC>(c+16,a+b*4,xs+32*64,b,desc+256);}\n  else mma<NT,DESC>(c,a+b*4,xs,b,desc);\n }\n }\n#if BR_HOPPER\n asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\n#endif\n}\ntemplate<int NT,int VEC>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*SC,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n constexpr int NJ=NT>=16?NT/16:1;\n int t=threadIdx.x,w=t>>5,l=t&31;\n if constexpr(VEC){\n  if(t<64){unsigned group=t>>4,lane=t&15;\n   unsigned base=((unsigned)e*(N/16)+mt*4+group)*128*64+tile*(KT/32)*64+lane*4;\n   #pragma unroll\n   for(int g=0;g<KT/32;++g){unsigned dst=smaddr(q+g*1024+t*16);const void*sp=Q+base+g*64;\n    asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");}\n  }\n  if(t>=64&&t<64+KT/4){unsigned i=t-64,g=i/8,group=(i%8)/2,half=i%2;\n   unsigned srcidx=(((unsigned)e*(N/16)+mt*4+group)*128+tile*(KT/32)+g)*32+half*16;\n   unsigned dst=smaddr(s+g*128+group*32+half*16);const void*sp=SC+srcidx;\n   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");\n  }\n }else{\n  #pragma unroll\n  for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n   async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\n   if(l<8)async_copy(s+g*128+w*32+l*4,SC+z*32+l*4,4,1);\n  }\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;\n   if(NT>=16||r<NT)async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\ntemplate<int NT>constexpr int smem_bytes(){return 1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;}\n__device__ __forceinline__ void mb_wait(unsigned mb,unsigned parity){\n asm volatile("{ .reg .pred p; MB_WAIT%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra MB_WAIT%=; }"::"r"(mb),"r"(parity):"memory");\n}\n__device__ __forceinline__ void tma2d(unsigned dst,const CUtensorMap*map,int x,int y,unsigned mb){\n asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"\n  ::"r"(dst),"l"(map),"r"(x),"r"(y),"r"(mb):"memory");\n}\n\ntemplate<int NT,bool FIRST,int VEC,int FM,int LO,int HI>\n__global__ void __launch_bounds__(128,2) gemm(const int*Q,const unsigned char*SC,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,unsigned scale_base,const __grid_constant__ CUtensorMap tmx){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT,N=FIRST?8192:4096,MT=N/64,NJ=NT>=16?NT/16:1;\n // Lean-loop flags (VEC-2): 1 = W13 gate/up interleave (SwiGLU in registers), 2 = no per-stage proxy fence (timing\n // probe), 4 = wide banks (nf_wide: 3 LDS.128 per stage), 8 = X tile by TMA (tmx, mbarrier per stage; no proxy fence),\n // 16 / 32 = timing probes that skip the X / code+scale copies (stale smem: wrong output).\n constexpr int FL=VEC>=2?VEC-2:0;constexpr bool ILV=FL&1,NOFENCE=FL&2,WIDE=FL&4,TMA=FL&8,NOX=FL&16,NOW=FL&32,TYPED=FL&64,WPART=FL&128,DESC=FL&256,ADAPT=FL&512,PERM=FL&1024;\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(1024) unsigned char typed_sm[];\n struct __align__(1024) Storage { __nv_bfloat16 x[S*NT*KT]; };\n __shared__ std::conditional_t<TYPED,Storage,unsigned char> static_sm;\n unsigned char*qs=typed_sm,*ss=qs+S*QB;\n auto*bs=TYPED?((Storage*)&static_sm)->x:(__nv_bfloat16*)(qs+BS);\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31;\n __shared__ __align__(8) unsigned long long fullb[S];\n if constexpr(TMA){\n  if(t==0){\n   #pragma unroll\n   for(int s=0;s<S;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(smaddr(&fullb[s])):"memory");\n   asm volatile("fence.mbarrier_init.release.cluster;":::"memory");\n  }\n  __syncthreads();\n }\n int total=TASK[32]*MT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  int group=unit/MT,mt=unit%MT,ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*(NT>=64?NT:32),count=C[e],offset=OFF[e];\n  if constexpr(FM==1){if(count<=LO||count>HI)continue;}\n  if constexpr(FM==2){int rem=min(count-row,32);if(rem<=LO||rem>HI)continue;}\n  int src[NJ];\n  #pragma unroll\n  for(int j=0;j<NJ;++j){int rl=t/8+j*16,r=row+rl;src[j]=(rl<NT&&r<count)?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  if constexpr(VEC>=2){\n   // Lean K loop (same smem layout, copies and math as VEC==1): per-unit global pointers advanced by a fixed\n   // stride, one commit group per stage (empty past the end) so the wait count is a constant, and the stage\n   // ring unrolled so every shared offset and WGMMA descriptor is a compile-time constant.\n   // Wide banks: one LDS.128 brings a thread\'s codes for two K32 blocks and one its four scale words per stage\n   // (smem codes [K64 pair][thread][2][8B], scales [warp][row8][4][4B]).\n   static_assert(NS%S==0,"stage ring");\n   const int*qg=Q;const unsigned char*sg=SC;const __nv_bfloat16*xg[NJ];bool xv[NJ];\n   const bool wq=t<64,ws=t>=64&&t<64+KT/4;\n   if(wq){unsigned gp=t>>4,ln=t&15;qg=Q+((unsigned)e*(N/16)+mt*4+gp)*128*64+ln*4;}\n   unsigned qd=smaddr(qs+(WIDE?(t>>4)*512+(t&15)*16:t*16)),sd=0;\n   if(ws){unsigned i=t-64;\n    if constexpr(WIDE){sg=SC+(((unsigned)e*(N/16)+mt*4+i/8)*128*32+(i%8)*16);sd=smaddr(ss+i*16);}\n    else{unsigned g=i/8,gp=(i%8)/2,hf=i%2;sg=SC+((((unsigned)e*(N/16)+mt*4+gp)*128+g)*32+hf*16);sd=smaddr(ss+g*128+gp*32+hf*16);}}\n   unsigned bd[NJ];\n   #pragma unroll\n   for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;xv[j]=src[j]>=0;xg[j]=X+(size_t)max(src[j],0)*4096+k;\n    bd[j]=smaddr(bs+r*64+(k^((r&7)*8)));}\n   // TMA (W2 only: its rows are the expert\'s contiguous ACT rows): lane 0 of warp 0 arms the stage barrier and loads\n   // one 64x32 box per K half; rows past count are padding the epilogue never stores.\n   static_assert(!(TMA&&FIRST),"W13 rows are gathered: one TMA per row serialises in warp 0");\n   int kx=0;const int xrow=offset+row;\n   uint64_t bd0=(uint64_t)((smaddr(bs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n   auto issue=[&](int slot){\n    if constexpr(TMA){\n     if(t==0){unsigned mb=smaddr(&fullb[slot]),dst=smaddr(bs+slot*NT*KT);\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mb),"r"(NT*KT*2):"memory");\n      #pragma unroll\n      for(int c=0;c<KT/64;++c)tma2d(dst+c*NT*64*2,&tmx,kx+c*64,xrow,mb);\n     }\n     kx+=KT;\n    }\n    if(wq&&!NOW){\n     #pragma unroll\n     for(int g=0;g<KT/32;++g)asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(qd+slot*QB+(WIDE?(g>>1)*2048+(g&1)*256:g*1024)),"l"(qg+g*64):"memory");\n     qg+=(KT/32)*64;}\n    if(ws&&!NOW){asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+slot*SB),"l"(sg):"memory");sg+=(KT/32)*32;}\n    if constexpr(!TMA&&!NOX){\n    #pragma unroll\n    for(int c=0;c<KT/64;++c){\n     #pragma unroll\n     for(int j=0;j<NJ;++j)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(bd[j]+(slot*NT*KT+c*NT*64)*2),"l"(xg[j]+c*64),"r"(xv[j]?16:0):"memory");\n    }\n    #pragma unroll\n    for(int j=0;j<NJ;++j)xg[j]+=KT;\n    }\n   };\n   #pragma unroll\n   for(int i=0;i<S-1;++i){issue(i);asm volatile("cp.async.commit_group;":::"memory");}\n   #pragma unroll 1\n   for(int k0=0;k0<NS;k0+=S){\n    #pragma unroll\n    for(int st=0;st<S;++st){\n     asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n     if constexpr(TMA)mb_wait(smaddr(&fullb[st]),(k0/S)&1);\n     __syncthreads();\n#if BR_HOPPER\n     if constexpr(!NOFENCE&&!TMA)asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n     uint4 sv;\n     if constexpr(WIDE)sv=*(const uint4*)(ss+st*SB+(w*8+(l>>2))*16);\n     #pragma unroll\n     for(int sub=0;sub<2;++sub){\n      if constexpr(WIDE){uint4 q=*(const uint4*)(qs+st*QB+sub*2048+t*16);\n       dq<PERM>(q.x,q.y,sub?sv.z:sv.x,af[sub],scale_base);dq<PERM>(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,scale_base);}\n      else{\n      #pragma unroll\n      for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n       unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq<PERM>(q.x,q.y,sf,af[sub]+g*8,scale_base);}}\n      keep(af[sub]);\n#if BR_HOPPER\n      asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n      mma_group<NT,DESC,ADAPT>(acc,af[sub],bs+st*NT*KT+sub*NT*64,bd0+(st*NT*KT+sub*NT*64)/8,count-row);keep(af[sub^1]);\n      if(sub==0){if(k0+st+S-1<NS)issue((st+S-1)%S);asm volatile("cp.async.commit_group;":::"memory");}\n     }\n    }\n   }\n  }else{\n  #pragma unroll\n  for(int i=0;i<S-1;++i)load_stage<NT,VEC>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SC,X,src,e,mt,N,i);\n  #pragma unroll 1\n  for(int k=0;k<NS;++k){\n   int st=k%S;cp_wait<NS>(k);__syncthreads();\n#if BR_HOPPER\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll\n   for(int sub=0;sub<2;++sub){\n    #pragma unroll\n    for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n     unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8);}\n    keep(af[sub]);\n#if BR_HOPPER\n    asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n    mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64);keep(af[sub^1]);\n    if(sub==0&&k+S-1<NS){int dst=(k+S-1)%S;\n     load_stage<NT,VEC>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SC,X,src,e,mt,N,k+S-1);}\n   }\n  }\n  }\n#if BR_HOPPER\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  if constexpr(FIRST&&ILV){\n   // Gate/up rows interleaved by 8 in prepare (per 64-row tile: G0-7,U0-7,G8-15,U8-15,...): accumulator j holds gate\n   // and j+2 the up row of the same channel, so SwiGLU finishes in registers (same BF16 roundings as below).\n   float gg=GS[e]*256.f,gu=GS[e+32]*256.f;int ch=mt*32+w*8+(l>>2);\n   #pragma unroll\n   for(int j=0;j<NT/2;++j)if(((j>>1)&1)==0){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    float g=__bfloat162float(__float2bfloat16_rn(acc[j]*gg)),u=__bfloat162float(__float2bfloat16_rn(acc[j+2]*gu));\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+ch]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){\n     if constexpr(WPART){\n      int r0=row+i*8+2*l;\n      if(r0<count)SS[(size_t)(offset+r0)*256+mt*4+w]=a;\n      if(r0+1<count)SS[(size_t)(offset+r0+1)*256+mt*4+w]=b;\n     }else{rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n    }\n   }\n   if constexpr(!WPART){\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n   }\n  }\n  __syncthreads();\n }\n}\n\n// Routing only: no output fill, no readiness state. One CTA owns every pair (same as the\n// original CTA 0); with META it also writes the heavy-only (count>16) 32-row task prefix T32, the\n// one-group-per-expert prefix T8 of experts with 8<count<=16.\ntemplate<bool META>__global__ void __launch_bounds__(256) route_live(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,int*T32,\n int*T8,int M,int NT,int start){\n int t=threadIdx.x;\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\n // All eight cold id loads per thread are issued before the first shared atomic.\n for(int base=0;base<M*8;base+=256*8){int v[8];\n  #pragma unroll\n  for(int k=0;k<8;++k){int i=base+k*256+t;v[k]=i<M*8?ids[i]:start-1;}\n  #pragma unroll\n  for(int k=0;k<8;++k){int i=base+k*256+t,e=v[k]-start,rank=-1;\n   if(e>=0&&e<32){rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;}\n   if(i<M*8)POS[i]=rank;}\n }\n __syncthreads();\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\n  int scan=pad;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\n  int key=n*32+31-t;\n  #pragma unroll\n  for(int k=2;k<=32;k*=2){\n   #pragma unroll\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\n   }\n  }\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\n  if constexpr(META){int ns=key/32,v=ns>16?(ns+31)/32:0,u=ns>8&&ns<=16;\n   #pragma unroll\n   for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),b=__shfl_up_sync(~0u,u,d);if(t>=d){v+=a;u+=b;}}\n   T32[t+1]=v;T8[t+1]=u;if(t==0){T32[0]=0;T8[0]=0;}}\n }\n}\n\n// SPLIT blocks per token, each owning 4096/SPLIT channels; per-element math and order are the same for every SPLIT.\ntemplate<int SPLIT,bool WPART=false>__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\n constexpr int W=4096/SPLIT;int tok=blockIdx.x/SPLIT,c0=blockIdx.x%SPLIT*W,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];\n  if constexpr(WPART){\n   for(int k=0;k<2;++k){float v=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)v+=SS[(size_t)p*256+(l+k*32)*4+ww];\n    sum+=v;\n   }\n  }else sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];\n }\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\n __syncthreads();int pp[8];float aa[8];\n #pragma unroll\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\n for(int c=c0+t*4;c<c0+W;c+=1024){float4 sum=make_float4(0,0,0,0);\n  #pragma unroll\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n }\n}\n}  // namespace mwbr\n\n// Pointer ABI for gemm: Q,SC,GS,X,C,OFF,ORDER,TASK,SLOTS,PNW,ACT,Y,SS.\n#define BR_ARGS(v) (const int*)v[0],(const unsigned char*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(const __nv_bfloat16*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12]\n#define BR_DYN(NT,V) ((((V)>=2)&&(((V)-2)&64))?1023+((mwbr::S*(64*mwbr::KT/2+64*mwbr::KT/16)+1023)&~1023):mwbr::smem_bytes<NT>())\n#define BR_INIT(ID,NT,F,V,FM,LO,HI) BR_CUDA_CHECK(cudaFuncSetAttribute(mwbr::gemm<NT,F,V,FM,LO,HI>,cudaFuncAttributeMaxDynamicSharedMemorySize,BR_DYN(NT,V)));\n// X tensor map for TMA kinds (W2): the ACT buffer [M*8+32*128,4096], 64x32 boxes.\nstatic CUtensorMap br_tmap(void*ptr,uint64_t rows){\n static PFN_cuTensorMapEncodeTiled_v12000 enc=nullptr;\n if(!enc){cudaDriverEntryPointQueryResult q;BR_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",(void**)&enc,cudaEnableDefault,&q));BR_REQUIRE(q==cudaDriverEntryPointSuccess&&enc);}\n CUtensorMap m;cuuint64_t dims[2]={4096,rows},strides[1]={8192};cuuint32_t box[2]={64,32},es[2]={1,1};\n BR_REQUIRE(enc(&m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,\n  CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS);\n return m;\n}\n#define BR_GO(ID,NT,F,V,FM,LO,HI) case ID:{CUtensorMap m{};if((V)>=2&&(((V)-2)&8))m=br_tmap((void*)v[3],(uint64_t)(M*8+32*128));\\\n mwbr::gemm<NT,F,V,FM,LO,HI><<<grid,128,BR_DYN(NT,V),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M,0x70007000u,m);}break;\n#define BR_INFO(ID,NT,F,V,FM,LO,HI) case ID:{cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncGetAttributes(&a,mwbr::gemm<NT,F,V,FM,LO,HI>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwbr::gemm<NT,F,V,FM,LO,HI>,128,BR_DYN(NT,V)));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)BR_DYN(NT,V),n};}\nint64_t br_init(){BR_KINDS(BR_INIT) return 0;}\nstd::vector<int64_t> br_info(int64_t kind){switch(kind){BR_KINDS(BR_INFO) default:BR_REQUIRE(false);}return {};}\nvoid br_run(std::vector<int64_t>v,int64_t kind,int64_t grid,int64_t M,int64_t stream){\n BR_REQUIRE(v.size()==13);\n switch(kind){BR_KINDS(BR_GO) default:BR_REQUIRE(false);}\n BR_CUDA_CHECK(cudaGetLastError());\n}\n// v: ids,C,OFF,ORDER,TASK,SLOTS,POS,T32,T8 (T32/T8 written only with meta)\nvoid br_route(std::vector<int64_t>v,int64_t M,int64_t start,int64_t meta,int64_t stream){\n BR_REQUIRE(v.size()==9);\n if(meta)mwbr::route_live<true><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\n else mwbr::route_live<false><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\n BR_CUDA_CHECK(cudaGetLastError());\n}\n// v: OFF,IDS,TW,POS,Y,SS,OUT\nvoid br_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t split,int64_t stream){\n BR_REQUIRE(v.size()==7);\n#define BR_FIN(S) case S:mwbr::finish<S><<<M*S,256,0,(cudaStream_t)stream>>>((const int*)v[0],(const int*)v[1],(const float*)v[2],(const int*)v[3],(const float*)v[4],(const float*)v[5],(__nv_bfloat16*)v[6],(int)M,(int)start);break;\n switch(split){BR_FIN(1) BR_FIN(2) BR_FIN(4) BR_FIN(8) BR_FIN(16)\n case 104:mwbr::finish<4,true><<<M*4,256,0,(cudaStream_t)stream>>>((const int*)v[0],(const int*)v[1],(const float*)v[2],(const int*)v[3],(const float*)v[4],(const float*)v[5],(__nv_bfloat16*)v[6],(int)M,(int)start);break;\n default:BR_REQUIRE(false);}\n#undef BR_FIN\n BR_CUDA_CHECK(cudaGetLastError());\n}\n\nnamespace pmz {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  uint32_t W;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(W):"r"(A),"r"(B));\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>\n__global__ void __launch_bounds__(NTHR,2) phase(\n const __nv_bfloat16*x,const __nv_bfloat16*pnw,const uint8_t*w13p,const uint8_t*w2p,\n const float*gs13,const float*gs2,const int*C,const int*OFF,const int*SLOTS,\n __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){\n constexpr int NB=8*NT,RED=NT*4;\n extern __shared__ __align__(128) uint8_t smem[];\n uint8_t*stages=smem;float*red=(float*)(smem+nstage*TILE_BYTES);\n uint64_t*fullb=(uint64_t*)(red+2*NCW*32*RED),*emptyb=fullb+nstage;\n Route*rt=(Route*)(emptyb+nstage);\n const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;\n if(tid==0){for(int z=0;z<nstage;++z){mbar_init(fullb+z,1);mbar_init(emptyb+z,NCW);}\n  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\n if(warp==0){int n=C[lane];int nb=(n+NB-1)/NB;if constexpr(SMALL)if(n>THR)nb=0;\n  rt->n[lane]=n;rt->off[lane]=OFF[lane];rt->nblk[lane]=nb;\n  int u1=nb*NT1,u2=nb*NT2;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,u1,d),b=__shfl_up_sync(~0u,u2,d);if(lane>=d){u1+=a;u2+=b;}}\n  rt->us1[lane+1]=u1;rt->us2[lane+1]=u2;\n  if(lane==0){rt->us1[0]=rt->us2[0]=0;}\n  int U1=__shfl_sync(~0u,u1,31),U2=__shfl_sync(~0u,u2,31);\n  if(lane==0){rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));}\n }\n __syncthreads();\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=FIRST?b1-a1:0,total_count=FIRST?first_count:b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = FIRST?0:1; pass < (FIRST?1:2); ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        cur_pass=pass;cur_e=e;cur_nb=nb;seg_tiles=0;\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(SLOTS[e*M+idx] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= gs13[e+32]*256.f;\n            G1=__bfloat162float(__float2bfloat16_rn(G1));\n            U=__bfloat162float(__float2bfloat16_rn(U));\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n        if constexpr (ZSS) {\n          if (m == 0) {\n            for (int z = tid; z < NB * 64; z += NTHR) {\n              const int zi = nb * NB + (z >> 6);\n              if (zi < ne) ss[static_cast<size_t>(off + zi) * 64 + (z & 63)] = 0.f;\n            }\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v*fw[0];\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + (size_t)(off+idx)*64+(m>>2), t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t pmz_init(){int dev,opt;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n auto set=[&]<bool F,bool Z,int T>(){cudaFuncAttributes a{};BR_CUDA_CHECK(cudaFuncGetAttributes(&a,pmz::phase<1,F,true,Z,T>));BR_CUDA_CHECK(cudaFuncSetAttribute(pmz::phase<1,F,true,Z,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));};\n set.template operator()<true,true,16>();set.template operator()<false,false,16>();set.template operator()<false,false,8>();return opt;}\nint64_t pmz_smem(int64_t stages,int64_t M){return pmz::smem_bytes(1,stages,M);}\nint64_t pmz_resident(int64_t stages,int64_t M,int64_t first){int n;\n if(first)BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,true,true,true,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));\n else BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,false,true,false,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));return n;}\n// v: x,pnw,w13p,w2p,gs13,gs2,C,OFF,SLOTS,act,y,ss. Experts with count<=thr only; W13 (thr 16) zeroes their SS rows.\nvoid pmz_run(std::vector<int64_t>v,int64_t first,int64_t thr,int64_t M,int64_t stages,int64_t grid,int64_t stream){\n BR_REQUIRE(v.size()==12);BR_REQUIRE(first?thr==16:(thr==16||thr==8));\n auto launch=[&]<bool F,bool Z,int T>(){pmz::phase<1,F,true,Z,T><<<grid,pmz::NTHR,pmz::smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\n (__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(uint8_t*)v[2],(uint8_t*)v[3],(float*)v[4],(float*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11],M,stages);};\n if(first)launch.template operator()<true,true,16>();else if(thr==16)launch.template operator()<false,false,16>();else launch.template operator()<false,false,8>();\n BR_CUDA_CHECK(cudaGetLastError());}\n'
BR_HH_CPP = '#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\nint64_t br_init();std::vector<int64_t> br_info(int64_t);void br_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);void br_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);void br_finish(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);int64_t pmz_init();int64_t pmz_smem(int64_t,int64_t);int64_t pmz_resident(int64_t,int64_t,int64_t);void pmz_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'
BR_S6_CUDA = '#define BR_KINDS(F) F(4,32,true,1,2,16,32) F(5,16,true,1,2,0,16) F(8,32,false,0,2,16,32) F(9,16,false,0,2,0,16)\n// Branch/remainder lab kernels. Banks stay packed E2M1 plus uint8 exponent/mantissa scale codes\n// (sign/zero folded into the codes); scales convert to BF16 inside the timed kernel.\n// One task = one 32-row group of one expert x one 64-channel M tile (route NT=32 numbering).\n// A kernel instance owns the groups whose expert count (FM=1) or rows-in-group (FM=2) lies in (LO,HI].\n#include <cuda.h>\n#include <cudaTypedefs.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <vector>\n#include <stdexcept>\n#include <type_traits>\n#define BR_CUDA_CHECK(expr) do{cudaError_t br_s=(expr);if(br_s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(br_s));}while(0)\n#define BR_REQUIRE(c) do{if(!(c))throw std::runtime_error("branch lab argument check");}while(0)\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define BR_HOPPER 1\n#else\n#define BR_HOPPER 0\n#endif\n\nnamespace mwbr {\nconstexpr int BK=64,KT=128,S=4,BIG=1<<30;\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\ntemplate<bool PERM=false>__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out,unsigned scale_base=0x70007000u){\n unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;\n unsigned s0,s1,s2,s3;\n if constexpr(PERM){\n  unsigned even,odd;unsigned rot=rr(raw,8);\n  asm("lop3.b32 %0,%1,0x0ff00ff0,%2,0xea;":"=r"(even):"r"(raw),"r"(scale_base));\n  asm("lop3.b32 %0,%1,0x0ff00ff0,%2,0xea;":"=r"(odd):"r"(rot),"r"(scale_base));\n  s0=__byte_perm(even,even,0x1010);s2=__byte_perm(even,even,0x3232);\n  s1=__byte_perm(odd,odd,0x1010);s3=__byte_perm(odd,odd,0x3232);\n }else{\n  s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u;s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;\n  s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u;s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;\n }\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),\n  "r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT,bool DESC=false>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b,uint64_t desc=0){\n#if BR_HOPPER\n uint64_t d=DESC?desc:(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==8){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3},{%4,%5,%6,%7},%8,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n static_assert(NT==32,"tile width");\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n // Portable stand-in: identical shared layout and fragment ownership, immediate mma.sync.\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT,bool DESC=false,bool ADAPT=false>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs,uint64_t desc=0,int live=NT){\n if constexpr(ADAPT){\n  #pragma unroll\n  for(int g=0;g<NT/32;++g)if(live>g*32){\n   #pragma unroll\n   for(int b=0;b<4;++b)mma<32,DESC>(c+g*16,a+b*4,xs+g*32*64,b,desc+g*256);\n  }\n }else{\n#pragma unroll\n for(int b=0;b<4;++b){\n  if constexpr(NT==64){mma<32,DESC>(c,a+b*4,xs,b,desc);mma<32,DESC>(c+16,a+b*4,xs+32*64,b,desc+256);}\n  else mma<NT,DESC>(c,a+b*4,xs,b,desc);\n }\n }\n#if BR_HOPPER\n asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\n#endif\n}\ntemplate<int NT,int VEC>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*SC,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n constexpr int NJ=NT>=16?NT/16:1;\n int t=threadIdx.x,w=t>>5,l=t&31;\n if constexpr(VEC){\n  if(t<64){unsigned group=t>>4,lane=t&15;\n   unsigned base=((unsigned)e*(N/16)+mt*4+group)*128*64+tile*(KT/32)*64+lane*4;\n   #pragma unroll\n   for(int g=0;g<KT/32;++g){unsigned dst=smaddr(q+g*1024+t*16);const void*sp=Q+base+g*64;\n    asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");}\n  }\n  if(t>=64&&t<64+KT/4){unsigned i=t-64,g=i/8,group=(i%8)/2,half=i%2;\n   unsigned srcidx=(((unsigned)e*(N/16)+mt*4+group)*128+tile*(KT/32)+g)*32+half*16;\n   unsigned dst=smaddr(s+g*128+group*32+half*16);const void*sp=SC+srcidx;\n   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");\n  }\n }else{\n  #pragma unroll\n  for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n   async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\n   if(l<8)async_copy(s+g*128+w*32+l*4,SC+z*32+l*4,4,1);\n  }\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;\n   if(NT>=16||r<NT)async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\ntemplate<int NT>constexpr int smem_bytes(){return 1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;}\n__device__ __forceinline__ void mb_wait(unsigned mb,unsigned parity){\n asm volatile("{ .reg .pred p; MB_WAIT%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra MB_WAIT%=; }"::"r"(mb),"r"(parity):"memory");\n}\n__device__ __forceinline__ void tma2d(unsigned dst,const CUtensorMap*map,int x,int y,unsigned mb){\n asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"\n  ::"r"(dst),"l"(map),"r"(x),"r"(y),"r"(mb):"memory");\n}\n\ntemplate<int NT,bool FIRST,int VEC,int FM,int LO,int HI>\n__global__ void __launch_bounds__(128,2) gemm(const int*Q,const unsigned char*SC,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,unsigned scale_base,const __grid_constant__ CUtensorMap tmx){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT,N=FIRST?8192:4096,MT=N/64,NJ=NT>=16?NT/16:1;\n // Lean-loop flags (VEC-2): 1 = W13 gate/up interleave (SwiGLU in registers), 2 = no per-stage proxy fence (timing\n // probe), 4 = wide banks (nf_wide: 3 LDS.128 per stage), 8 = X tile by TMA (tmx, mbarrier per stage; no proxy fence),\n // 16 / 32 = timing probes that skip the X / code+scale copies (stale smem: wrong output).\n constexpr int FL=VEC>=2?VEC-2:0;constexpr bool ILV=FL&1,NOFENCE=FL&2,WIDE=FL&4,TMA=FL&8,NOX=FL&16,NOW=FL&32,TYPED=FL&64,WPART=FL&128,DESC=FL&256,ADAPT=FL&512,PERM=FL&1024;\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(1024) unsigned char typed_sm[];\n struct __align__(1024) Storage { __nv_bfloat16 x[S*NT*KT]; };\n __shared__ std::conditional_t<TYPED,Storage,unsigned char> static_sm;\n unsigned char*qs=typed_sm,*ss=qs+S*QB;\n auto*bs=TYPED?((Storage*)&static_sm)->x:(__nv_bfloat16*)(qs+BS);\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31;\n __shared__ __align__(8) unsigned long long fullb[S];\n if constexpr(TMA){\n  if(t==0){\n   #pragma unroll\n   for(int s=0;s<S;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(smaddr(&fullb[s])):"memory");\n   asm volatile("fence.mbarrier_init.release.cluster;":::"memory");\n  }\n  __syncthreads();\n }\n int total=TASK[32]*MT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  int group=unit/MT,mt=unit%MT,ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*(NT>=64?NT:32),count=C[e],offset=OFF[e];\n  if constexpr(FM==1){if(count<=LO||count>HI)continue;}\n  if constexpr(FM==2){int rem=min(count-row,32);if(rem<=LO||rem>HI)continue;}\n  int src[NJ];\n  #pragma unroll\n  for(int j=0;j<NJ;++j){int rl=t/8+j*16,r=row+rl;src[j]=(rl<NT&&r<count)?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  if constexpr(VEC>=2){\n   // Lean K loop (same smem layout, copies and math as VEC==1): per-unit global pointers advanced by a fixed\n   // stride, one commit group per stage (empty past the end) so the wait count is a constant, and the stage\n   // ring unrolled so every shared offset and WGMMA descriptor is a compile-time constant.\n   // Wide banks: one LDS.128 brings a thread\'s codes for two K32 blocks and one its four scale words per stage\n   // (smem codes [K64 pair][thread][2][8B], scales [warp][row8][4][4B]).\n   static_assert(NS%S==0,"stage ring");\n   const int*qg=Q;const unsigned char*sg=SC;const __nv_bfloat16*xg[NJ];bool xv[NJ];\n   const bool wq=t<64,ws=t>=64&&t<64+KT/4;\n   if(wq){unsigned gp=t>>4,ln=t&15;qg=Q+((unsigned)e*(N/16)+mt*4+gp)*128*64+ln*4;}\n   unsigned qd=smaddr(qs+(WIDE?(t>>4)*512+(t&15)*16:t*16)),sd=0;\n   if(ws){unsigned i=t-64;\n    if constexpr(WIDE){sg=SC+(((unsigned)e*(N/16)+mt*4+i/8)*128*32+(i%8)*16);sd=smaddr(ss+i*16);}\n    else{unsigned g=i/8,gp=(i%8)/2,hf=i%2;sg=SC+((((unsigned)e*(N/16)+mt*4+gp)*128+g)*32+hf*16);sd=smaddr(ss+g*128+gp*32+hf*16);}}\n   unsigned bd[NJ];\n   #pragma unroll\n   for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;xv[j]=src[j]>=0;xg[j]=X+(size_t)max(src[j],0)*4096+k;\n    bd[j]=smaddr(bs+r*64+(k^((r&7)*8)));}\n   // TMA (W2 only: its rows are the expert\'s contiguous ACT rows): lane 0 of warp 0 arms the stage barrier and loads\n   // one 64x32 box per K half; rows past count are padding the epilogue never stores.\n   static_assert(!(TMA&&FIRST),"W13 rows are gathered: one TMA per row serialises in warp 0");\n   int kx=0;const int xrow=offset+row;\n   uint64_t bd0=(uint64_t)((smaddr(bs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n   auto issue=[&](int slot){\n    if constexpr(TMA){\n     if(t==0){unsigned mb=smaddr(&fullb[slot]),dst=smaddr(bs+slot*NT*KT);\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(mb),"r"(NT*KT*2):"memory");\n      #pragma unroll\n      for(int c=0;c<KT/64;++c)tma2d(dst+c*NT*64*2,&tmx,kx+c*64,xrow,mb);\n     }\n     kx+=KT;\n    }\n    if(wq&&!NOW){\n     #pragma unroll\n     for(int g=0;g<KT/32;++g)asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(qd+slot*QB+(WIDE?(g>>1)*2048+(g&1)*256:g*1024)),"l"(qg+g*64):"memory");\n     qg+=(KT/32)*64;}\n    if(ws&&!NOW){asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+slot*SB),"l"(sg):"memory");sg+=(KT/32)*32;}\n    if constexpr(!TMA&&!NOX){\n    #pragma unroll\n    for(int c=0;c<KT/64;++c){\n     #pragma unroll\n     for(int j=0;j<NJ;++j)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(bd[j]+(slot*NT*KT+c*NT*64)*2),"l"(xg[j]+c*64),"r"(xv[j]?16:0):"memory");\n    }\n    #pragma unroll\n    for(int j=0;j<NJ;++j)xg[j]+=KT;\n    }\n   };\n   #pragma unroll\n   for(int i=0;i<S-1;++i){issue(i);asm volatile("cp.async.commit_group;":::"memory");}\n   #pragma unroll 1\n   for(int k0=0;k0<NS;k0+=S){\n    #pragma unroll\n    for(int st=0;st<S;++st){\n     asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n     if constexpr(TMA)mb_wait(smaddr(&fullb[st]),(k0/S)&1);\n     __syncthreads();\n#if BR_HOPPER\n     if constexpr(!NOFENCE&&!TMA)asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n     uint4 sv;\n     if constexpr(WIDE)sv=*(const uint4*)(ss+st*SB+(w*8+(l>>2))*16);\n     #pragma unroll\n     for(int sub=0;sub<2;++sub){\n      if constexpr(WIDE){uint4 q=*(const uint4*)(qs+st*QB+sub*2048+t*16);\n       dq<PERM>(q.x,q.y,sub?sv.z:sv.x,af[sub],scale_base);dq<PERM>(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,scale_base);}\n      else{\n      #pragma unroll\n      for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n       unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq<PERM>(q.x,q.y,sf,af[sub]+g*8,scale_base);}}\n      keep(af[sub]);\n#if BR_HOPPER\n      asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n      mma_group<NT,DESC,ADAPT>(acc,af[sub],bs+st*NT*KT+sub*NT*64,bd0+(st*NT*KT+sub*NT*64)/8,count-row);keep(af[sub^1]);\n      if(sub==0){if(k0+st+S-1<NS)issue((st+S-1)%S);asm volatile("cp.async.commit_group;":::"memory");}\n     }\n    }\n   }\n  }else{\n  #pragma unroll\n  for(int i=0;i<S-1;++i)load_stage<NT,VEC>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SC,X,src,e,mt,N,i);\n  #pragma unroll 1\n  for(int k=0;k<NS;++k){\n   int st=k%S;cp_wait<NS>(k);__syncthreads();\n#if BR_HOPPER\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll\n   for(int sub=0;sub<2;++sub){\n    #pragma unroll\n    for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n     unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8);}\n    keep(af[sub]);\n#if BR_HOPPER\n    asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n    mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64);keep(af[sub^1]);\n    if(sub==0&&k+S-1<NS){int dst=(k+S-1)%S;\n     load_stage<NT,VEC>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SC,X,src,e,mt,N,k+S-1);}\n   }\n  }\n  }\n#if BR_HOPPER\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  if constexpr(FIRST&&ILV){\n   // Gate/up rows interleaved by 8 in prepare (per 64-row tile: G0-7,U0-7,G8-15,U8-15,...): accumulator j holds gate\n   // and j+2 the up row of the same channel, so SwiGLU finishes in registers (same BF16 roundings as below).\n   float gg=GS[e]*256.f,gu=GS[e+32]*256.f;int ch=mt*32+w*8+(l>>2);\n   #pragma unroll\n   for(int j=0;j<NT/2;++j)if(((j>>1)&1)==0){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    float g=__bfloat162float(__float2bfloat16_rn(acc[j]*gg)),u=__bfloat162float(__float2bfloat16_rn(acc[j+2]*gu));\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+ch]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){\n     if constexpr(WPART){\n      int r0=row+i*8+2*l;\n      if(r0<count)SS[(size_t)(offset+r0)*256+mt*4+w]=a;\n      if(r0+1<count)SS[(size_t)(offset+r0+1)*256+mt*4+w]=b;\n     }else{rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n    }\n   }\n   if constexpr(!WPART){\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n   }\n  }\n  __syncthreads();\n }\n}\n\n// Routing only: no output fill, no readiness state. One CTA owns every pair (same as the\n// original CTA 0); with META it also writes the heavy-only (count>16) 32-row task prefix T32, the\n// one-group-per-expert prefix T8 of experts with 8<count<=16.\ntemplate<bool META>__global__ void __launch_bounds__(256) route_live(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,int*T32,\n int*T8,int M,int NT,int start){\n int t=threadIdx.x;\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\n // All eight cold id loads per thread are issued before the first shared atomic.\n for(int base=0;base<M*8;base+=256*8){int v[8];\n  #pragma unroll\n  for(int k=0;k<8;++k){int i=base+k*256+t;v[k]=i<M*8?ids[i]:start-1;}\n  #pragma unroll\n  for(int k=0;k<8;++k){int i=base+k*256+t,e=v[k]-start,rank=-1;\n   if(e>=0&&e<32){rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;}\n   if(i<M*8)POS[i]=rank;}\n }\n __syncthreads();\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\n  int scan=pad;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\n  int key=n*32+31-t;\n  #pragma unroll\n  for(int k=2;k<=32;k*=2){\n   #pragma unroll\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\n   }\n  }\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\n  if constexpr(META){int ns=key/32,v=ns>16?(ns+31)/32:0,u=ns>8&&ns<=16;\n   #pragma unroll\n   for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),b=__shfl_up_sync(~0u,u,d);if(t>=d){v+=a;u+=b;}}\n   T32[t+1]=v;T8[t+1]=u;if(t==0){T32[0]=0;T8[0]=0;}}\n }\n}\n\n// SPLIT blocks per token, each owning 4096/SPLIT channels; per-element math and order are the same for every SPLIT.\ntemplate<int SPLIT,bool WPART=false>__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\n constexpr int W=4096/SPLIT;int tok=blockIdx.x/SPLIT,c0=blockIdx.x%SPLIT*W,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];\n  if constexpr(WPART){\n   for(int k=0;k<2;++k){float v=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)v+=SS[(size_t)p*256+(l+k*32)*4+ww];\n    sum+=v;\n   }\n  }else sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];\n }\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\n __syncthreads();int pp[8];float aa[8];\n #pragma unroll\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\n for(int c=c0+t*4;c<c0+W;c+=1024){float4 sum=make_float4(0,0,0,0);\n  #pragma unroll\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n }\n}\n}  // namespace mwbr\n\n// Pointer ABI for gemm: Q,SC,GS,X,C,OFF,ORDER,TASK,SLOTS,PNW,ACT,Y,SS.\n#define BR_ARGS(v) (const int*)v[0],(const unsigned char*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(const __nv_bfloat16*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12]\n#define BR_DYN(NT,V) ((((V)>=2)&&(((V)-2)&64))?1023+((mwbr::S*(64*mwbr::KT/2+64*mwbr::KT/16)+1023)&~1023):mwbr::smem_bytes<NT>())\n#define BR_INIT(ID,NT,F,V,FM,LO,HI) BR_CUDA_CHECK(cudaFuncSetAttribute(mwbr::gemm<NT,F,V,FM,LO,HI>,cudaFuncAttributeMaxDynamicSharedMemorySize,BR_DYN(NT,V)));\n// X tensor map for TMA kinds (W2): the ACT buffer [M*8+32*128,4096], 64x32 boxes.\nstatic CUtensorMap br_tmap(void*ptr,uint64_t rows){\n static PFN_cuTensorMapEncodeTiled_v12000 enc=nullptr;\n if(!enc){cudaDriverEntryPointQueryResult q;BR_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",(void**)&enc,cudaEnableDefault,&q));BR_REQUIRE(q==cudaDriverEntryPointSuccess&&enc);}\n CUtensorMap m;cuuint64_t dims[2]={4096,rows},strides[1]={8192};cuuint32_t box[2]={64,32},es[2]={1,1};\n BR_REQUIRE(enc(&m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,\n  CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS);\n return m;\n}\n#define BR_GO(ID,NT,F,V,FM,LO,HI) case ID:{CUtensorMap m{};if((V)>=2&&(((V)-2)&8))m=br_tmap((void*)v[3],(uint64_t)(M*8+32*128));\\\n mwbr::gemm<NT,F,V,FM,LO,HI><<<grid,128,BR_DYN(NT,V),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M,0x70007000u,m);}break;\n#define BR_INFO(ID,NT,F,V,FM,LO,HI) case ID:{cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncGetAttributes(&a,mwbr::gemm<NT,F,V,FM,LO,HI>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwbr::gemm<NT,F,V,FM,LO,HI>,128,BR_DYN(NT,V)));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)BR_DYN(NT,V),n};}\nint64_t br_init(){BR_KINDS(BR_INIT) return 0;}\nstd::vector<int64_t> br_info(int64_t kind){switch(kind){BR_KINDS(BR_INFO) default:BR_REQUIRE(false);}return {};}\nvoid br_run(std::vector<int64_t>v,int64_t kind,int64_t grid,int64_t M,int64_t stream){\n BR_REQUIRE(v.size()==13);\n switch(kind){BR_KINDS(BR_GO) default:BR_REQUIRE(false);}\n BR_CUDA_CHECK(cudaGetLastError());\n}\n// v: ids,C,OFF,ORDER,TASK,SLOTS,POS,T32,T8 (T32/T8 written only with meta)\nvoid br_route(std::vector<int64_t>v,int64_t M,int64_t start,int64_t meta,int64_t stream){\n BR_REQUIRE(v.size()==9);\n if(meta)mwbr::route_live<true><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\n else mwbr::route_live<false><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\n BR_CUDA_CHECK(cudaGetLastError());\n}\n// v: OFF,IDS,TW,POS,Y,SS,OUT\nvoid br_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t split,int64_t stream){\n BR_REQUIRE(v.size()==7);\n#define BR_FIN(S) case S:mwbr::finish<S><<<M*S,256,0,(cudaStream_t)stream>>>((const int*)v[0],(const int*)v[1],(const float*)v[2],(const int*)v[3],(const float*)v[4],(const float*)v[5],(__nv_bfloat16*)v[6],(int)M,(int)start);break;\n switch(split){BR_FIN(1) BR_FIN(2) BR_FIN(4) BR_FIN(8) BR_FIN(16)\n case 104:mwbr::finish<4,true><<<M*4,256,0,(cudaStream_t)stream>>>((const int*)v[0],(const int*)v[1],(const float*)v[2],(const int*)v[3],(const float*)v[4],(const float*)v[5],(__nv_bfloat16*)v[6],(int)M,(int)start);break;\n default:BR_REQUIRE(false);}\n#undef BR_FIN\n BR_CUDA_CHECK(cudaGetLastError());\n}\n'
BR_S6_CPP = '#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\nint64_t br_init();std::vector<int64_t> br_info(int64_t);void br_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);void br_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);void br_finish(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);'
BR_U16_CUDA = '#include <cuda_runtime.h>\n#include <vector>\n#include <stdexcept>\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\n// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\n\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace idist {\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\n}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned long long raw,unsigned*out){\n unsigned w,mask=0x81c081c0u;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));\n unsigned s0=__byte_perm((unsigned)raw,0u,0x1010u);\n unsigned s1=__byte_perm((unsigned)raw,0u,0x3232u);\n unsigned s2=__byte_perm((unsigned)(raw>>32),0u,0x1010u);\n unsigned s3=__byte_perm((unsigned)(raw>>32),0u,0x3232u);\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==32){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\n#pragma unroll\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n}\ntemplate<int NT,int KT>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n int t=threadIdx.x,w=t>>5,l=t&31;\n #pragma unroll\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n  if(l<16)async_copy(q+(g*128+w*32)*8+l*16,Q+z*64+l*4,16,1);\n  if(l<8)async_copy(s+g*256+w*64+l*8,S+z*64+l*8,8,1);\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if constexpr(S>3){\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n  else if constexpr(S>4){\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\n   else if constexpr(S>5){\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\n    else if constexpr(S>6){\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\n     else if constexpr(S>7){\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\n__device__ __forceinline__ int publish(int*p){\n int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;\n}\n__global__ void route_init(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,\n int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){\n int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;\n for(int i=all;i<clear_count;i+=step)state[i]=0;\n for(int i=all;i<M*4096/8;i+=step)reinterpret_cast<uint4*>(out)[i]=make_uint4(0,0,0,0);\n if(blockIdx.x)return;\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\n for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;\n  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}\n }\n for(int token=t;token<M;token+=256){int n=0;\n  #pragma unroll\n  for(int j=0;j<8;++j){int e=ids[token*8+j]-start;n+=(e>=0&&e<32);}\n  state[clear_count+token]=n;\n }\n __syncthreads();\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\n  int scan=pad;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\n  int key=n*32+31-t;\n  #pragma unroll\n  for(int k=2;k<=32;k*=2){\n   #pragma unroll\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\n   }\n  }\n  // Descending keys after reversing the ascending bitonic result.\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\n }\n}\ntemplate<int NT,int KT,int S,int SPLIT,bool FIRST,int MODE>\n__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,\n float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start,bool profiling){\n constexpr int QB=64*KT/2,SB=64*KT/8,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(128) unsigned char rawsm[];\n unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);\n __shared__ int last,flags[NT];\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;\n int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;\n \n int total=TASK[32]*MT*SPLIT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  uint64_t tick=profiling?clock64():0;uint64_t times[6]={};bool islast=false;\n  int task=unit/SPLIT,split=unit%SPLIT,group=task/MT,mt=task%MT,ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\n  int src[NT/16];\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  #pragma unroll\n  for(int i=0;i<min(S-1,NS);++i)\n   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\n  // K64 has one MMA group per stage. Alternate register-A banks across\n  // stages so wait1 retires a bank before its next conversion overwrites it.\n  // K128/K256 already alternate within each stage; their schedule is unchanged.\n  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");\n  #pragma unroll 1\n  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){\n   #pragma unroll\n   for(int ksub=0;ksub<(KT==64?2:1);++ksub){\n   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll 1\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\n    #pragma unroll\n    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;\n     #pragma unroll\n     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n      unsigned long long sf=*(const unsigned long long*)(ss+st*SB+z*256+w*64+(l>>2)*8);dq(q.x,q.y,sf,af[reg]+g*8);}\n     keep(af[reg]);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n     mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);keep(af[reg^1]);\n     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;\n      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);}\n    }\n   }\n   }\n  }\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  if(profiling){uint64_t now=clock64();times[0]=now-tick;tick=now;}\n  if constexpr(SPLIT>1){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];\n   }\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();\n   if(!last)continue;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;\n    if(r<count){\n     #pragma unroll\n     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];\n    }\n    acc[j]=sum;\n   }\n  }\n  if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n   }\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n   if constexpr(MODE!=1){\n   if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();\n   if(profiling){uint64_t now=clock64();times[2]=now-tick;tick=now;}\n   islast=last;\n   if(last){\n    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];\n     #pragma unroll\n     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);\n    }\n    __threadfence();__syncthreads();\n    if(profiling){uint64_t now=clock64();times[3]=now-tick;tick=now;}\n    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;\n      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}\n    __syncthreads();\n    if(profiling){uint64_t now=clock64();times[4]=now-tick;tick=now;}\n    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;\n     if constexpr(MODE==2){\n     int base[8];float alpha[8];\n     #pragma unroll\n     for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;base[j]=-1;alpha[j]=0;\n      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}\n     for(int c=l*4;c<4096;c+=128){float4 sum=make_float4(0,0,0,0);\n      #pragma unroll\n      for(int j=0;j<8;++j)if(base[j]>=0){float4 z=*(const float4*)(Y+(size_t)base[j]*4096+c);float a=alpha[j];sum.x+=z.x*a;sum.y+=z.y*a;sum.z+=z.z*a;sum.w+=z.w*a;}\n      __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n      *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n     }\n     }else{\n     for(int c=l;c<4096;c+=32){float sum=0;\n      #pragma unroll\n      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;\n       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}\n      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);\n     }\n     }\n    }\n   }\n   }else if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n  }\n  __syncthreads();\n  if(profiling&&t==0){uint64_t*p=(uint64_t*)PART+(size_t)unit*8;\n   for(int i=0;i<5;++i)p[i]=times[i];p[5]=clock64()-tick;p[6]=islast;p[7]=clock64();}\n }\n}\n#define FX_CASES(F) \\\n F(0,16,128,4,1) F(1,32,128,4,1)\n}\nint64_t idist_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n #define INIT(ID,NT,KT,S,SK) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/8)+1023)&~1023)+S*NT*KT*2;auto set=[&]<bool F,int MODE>(){WX_CUDA_CHECK(cudaFuncSetAttribute(idist::fused<NT,KT,S,SK,F,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<true,1>();set.template operator()<false,1>();if constexpr(ID==0)set.template operator()<false,0>();}\n FX_CASES(INIT)\n #undef INIT\n return opt;\n}\nstd::vector<int64_t> idist_info(int64_t which,int64_t first,int64_t mode){\n #define INFO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/8)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F,int MODE>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,idist::fused<NT,KT,S,SK,F,MODE>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,idist::fused<NT,KT,S,SK,F,MODE>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(first)return info.template operator()<true,1>();if(mode==1)return info.template operator()<false,1>();if constexpr(ID==0){WX_CHECK(mode==0);return info.template operator()<false,0>();}else WX_CHECK(false,"legacy control exists only forNT16");}\n switch(which){FX_CASES(INFO) default:WX_CHECK(false,"unknown fixed configuration");}\n #undef INFO\n}\nvoid idist_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear_count,int64_t stream){\n idist::route_init<<<128,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear_count);WX_CUDA_CHECK(cudaGetLastError());\n}\nvoid idist_run(std::vector<int64_t> v,int64_t first,int64_t which,int64_t grid,int64_t start,int64_t mode,int64_t profiling,int64_t M,int64_t cap,int64_t stream_value){\n WX_CHECK(v.size()==20);auto stream=(cudaStream_t)stream_value;\n #define GO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/8)+1023)&~1023)+S*NT*KT*2; \\\n auto launch=[&]<bool FIRST,int MODE>(){idist::fused<NT,KT,S,SK,FIRST,MODE><<<grid,128,bytes,stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(const __nv_bfloat16*)(void*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(const __nv_bfloat16*)(void*)v[14],(float*)v[10],(__nv_bfloat16*)(void*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)(void*)v[19],M,cap,start,profiling);};if(first)launch.template operator()<true,1>();else if(mode==1)launch.template operator()<false,1>();else{WX_CHECK(mode==0&&which==0);if constexpr(ID==0)launch.template operator()<false,0>();}break;}\n switch(which){FX_CASES(GO) default:WX_CHECK(false,"unknown fixed configuration");}\n #undef GO\n WX_CUDA_CHECK(cudaGetLastError());\n}\nnamespace idist {\n__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\n int tok=blockIdx.x,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\n __syncthreads();int pp[8];float aa[8];\n #pragma unroll\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\n for(int c=t*4;c<4096;c+=1024){float4 sum=make_float4(0,0,0,0);\n  #pragma unroll\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n }\n}\n}\nvoid idist_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){idist::finish<<<M,256,0,(cudaStream_t)stream>>>((int*)v[5],(int*)v[11],(float*)v[12],(int*)v[13],(float*)v[15],(float*)v[16],(__nv_bfloat16*)(void*)v[19],M,start);WX_CUDA_CHECK(cudaGetLastError());}\n\n#include <cuda_runtime.h>\n#include <vector>\n#include <stdexcept>\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\n\n\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace pmnt {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  uint32_t W;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(W):"r"(A),"r"(B));\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate<int NT,bool FIRST,bool SMALL>\n__global__ void __launch_bounds__(NTHR,2) phase(\n const __nv_bfloat16*x,const __nv_bfloat16*pnw,const uint8_t*w13p,const uint8_t*w2p,\n const float*gs13,const float*gs2,const int*C,const int*OFF,const int*SLOTS,\n __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){\n constexpr int NB=8*NT,RED=NT*4;\n extern __shared__ __align__(128) uint8_t smem[];\n uint8_t*stages=smem;float*red=(float*)(smem+nstage*TILE_BYTES);\n uint64_t*fullb=(uint64_t*)(red+2*NCW*32*RED),*emptyb=fullb+nstage;\n Route*rt=(Route*)(emptyb+nstage);\n const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;\n if(tid==0){for(int z=0;z<nstage;++z){mbar_init(fullb+z,1);mbar_init(emptyb+z,NCW);}\n  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\n if(warp==0){int n=C[lane];int nb=(n+NB-1)/NB;if constexpr(SMALL)if(n>16)nb=0;\n  rt->n[lane]=n;rt->off[lane]=OFF[lane];rt->nblk[lane]=nb;\n  int u1=nb*NT1,u2=nb*NT2;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,u1,d),b=__shfl_up_sync(~0u,u2,d);if(lane>=d){u1+=a;u2+=b;}}\n  rt->us1[lane+1]=u1;rt->us2[lane+1]=u2;\n  if(lane==0){rt->us1[0]=rt->us2[0]=0;}\n  int U1=__shfl_sync(~0u,u1,31),U2=__shfl_sync(~0u,u2,31);\n  if(lane==0){rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));}\n }\n __syncthreads();\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=FIRST?b1-a1:0,total_count=FIRST?first_count:b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = FIRST?0:1; pass < (FIRST?1:2); ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        cur_pass=pass;cur_e=e;cur_nb=nb;seg_tiles=0;\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(SLOTS[e*M+idx] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= gs13[e+32]*256.f;\n            G1=__bfloat162float(__float2bfloat16_rn(G1));\n            U=__bfloat162float(__float2bfloat16_rn(U));\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v*fw[0];\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + (size_t)(off+idx)*64+(m>>2), t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\n\n\nint64_t pm_nt_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n auto set=[&]<bool F,bool S>(){cudaFuncAttributes a{};WX_CUDA_CHECK(cudaFuncGetAttributes(&a,pmnt::phase<1,F,S>));WX_CUDA_CHECK(cudaFuncSetAttribute(pmnt::phase<1,F,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));};\n set.template operator()<true,false>();set.template operator()<false,false>();set.template operator()<true,true>();set.template operator()<false,true>();return opt;}\nint64_t pm_nt_smem(int64_t stages,int64_t M){return pmnt::smem_bytes(1,stages,M);}\nint64_t pm_nt_resident(int64_t stages,int64_t M,int64_t first,int64_t small){int n;\n auto get=[&]<bool F,bool S>(){WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmnt::phase<1,F,S>,pmnt::NTHR,pmnt::smem_bytes(1,stages,M)));};\n if(first){if(small)get.template operator()<true,true>();else get.template operator()<true,false>();}else{if(small)get.template operator()<false,true>();else get.template operator()<false,false>();}return n;}\nvoid pm_nt_run(std::vector<int64_t>v,int64_t first,int64_t small,int64_t M,int64_t stages,int64_t grid,int64_t stream){\n auto launch=[&]<bool F,bool S>(){pmnt::phase<1,F,S><<<grid,pmnt::NTHR,pmnt::smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\n (__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(uint8_t*)v[2],(uint8_t*)v[3],(float*)v[4],(float*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11],M,stages);};\n if(first){if(small)launch.template operator()<true,true>();else launch.template operator()<true,false>();}else{if(small)launch.template operator()<false,true>();else launch.template operator()<false,false>();}WX_CUDA_CHECK(cudaGetLastError());}\n\n\n#include <cuda_runtime.h>\n#include <vector>\n__global__ void pm_meta_kernel(const int*C,const int*ORDER,int*T32,int*T128,int*T16,float*SS,int cap){\n int t=threadIdx.x,idx=blockIdx.x*blockDim.x+t;\n for(int i=idx;i<cap*64;i+=gridDim.x*blockDim.x)SS[i]=0.f;\n if(blockIdx.x==0&&t<32){int n=C[ORDER[t]],v32=n>16?(n+31)/32:0,v128=(n+127)/128,v16=(n+15)/16;\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v32,d),b=__shfl_up_sync(~0u,v128,d),c=__shfl_up_sync(~0u,v16,d);if(t>=d){v32+=a;v128+=b;v16+=c;}}\n  T32[t+1]=v32;T128[t+1]=v128;T16[t+1]=v16;if(t==0){T32[0]=0;T128[0]=0;T16[0]=0;}\n }\n}\nvoid pm_meta(std::vector<int64_t>v,int64_t cap,int64_t stream){pm_meta_kernel<<<128,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(float*)v[5],cap);}\n'
BR_M = 128
BR_SET = 'branch'
import functools
import hashlib
import statistics
import time
import torch
from torch.utils.cpp_extension import load_inline

BR_DEADLINE = float('inf')
BR_OPTS = {}


class BRBudget(RuntimeError):
    pass


def br_need(seconds, where):
    if BR_DEADLINE - time.monotonic() < seconds:
        raise BRBudget(where)


def br_compile(name, cpp, cuda, functions):
    br_need(30., name + ' compile')
    arch = '90a' if torch.cuda.get_device_capability() == (9, 0) else '120'
    return load_inline(name=name + '_' + hashlib.sha256(cuda.encode()).hexdigest()[:12], cpp_sources=cpp, cuda_sources=cuda,
                       functions=functions, no_implicit_headers=True, with_pytorch_error_handling=False,
                       extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', f'-gencode=arch=compute_{arch},code=sm_{arch}'],
                       verbose=False)


BR_FUNCS = ['br_init', 'br_info', 'br_run', 'br_route', 'br_finish']


@functools.lru_cache(None)
def br_hh_module():
    mod = br_compile('brhh', BR_HH_CPP, BR_HH_CUDA, BR_FUNCS + ['pmz_init', 'pmz_smem', 'pmz_resident', 'pmz_run'])
    mod.br_init()
    BR_OPTS['pmz'] = mod.pmz_init()
    return mod


@functools.lru_cache(None)
def br_wp_module():
    # NT1's 64 norm bins become the first of four per tile; the other three stay zero.
    # No wider weights: only the FP32 activation-side norm scratch grows.
    cu = BR_HH_CUDA
    cu = cu.replace('z < NB * 64;', 'z < NB * 256;').replace('(z >> 6)', '(z >> 8)')
    cu = cu.replace('ss[static_cast<size_t>(off + zi) * 64 + (z & 63)]',
                    'ss[static_cast<size_t>(off + zi) * 256 + (z & 255)]')
    cu = cu.replace('ss + (size_t)(off+idx)*64+(m>>2)', 'ss + (size_t)(off+idx)*256+(m>>2)*4')
    mod = br_compile('brwp', BR_HH_CPP, cu, BR_FUNCS + ['pmz_init', 'pmz_smem', 'pmz_resident', 'pmz_run'])
    mod.br_init();mod.pmz_init()
    return mod


@functools.lru_cache(None)
def br_pair_module(stages):
    cu = BR_HH_CUDA.replace('KT=128,S=4,', f'KT=128,S={stages},')
    cu = cu.replace('ns>16?(ns+31)/32:0', 'ns>16?(ns+63)/64:0')
    mod = br_compile('brpair'+str(stages), BR_HH_CPP, cu, BR_FUNCS + ['pmz_init', 'pmz_smem', 'pmz_resident', 'pmz_run'])
    mod.br_init();mod.pmz_init()
    return mod


@functools.lru_cache(None)
def br_adapt_module(nt):
    macro = '#define BR_KINDS(F) ' + ' '.join(
        f'F({i},{nt},{str(bool(first)).lower()},{514+first+flags},0,0,0)'
        for i,first,flags in ((60,1,0),(61,0,0),(62,1,256),(63,0,256),(64,1,260),(65,0,260),(66,1,1280),(67,0,1280),(68,1,1284),(69,0,1284))) + '\n'
    cu = macro+BR_HH_CUDA[BR_HH_CUDA.index('\n')+1:]
    cu = cu.replace('KT=128,S=4,','KT=128,S=2,')
    cu = cu.replace('ns>16?(ns+31)/32:0',f'ns>16?(ns+{nt-1})/{nt}:0')
    mod=br_compile('bradapt'+str(nt),BR_HH_CPP,cu,BR_FUNCS+['pmz_init','pmz_smem','pmz_resident','pmz_run'])
    mod.br_init();mod.pmz_init()
    return mod


@functools.lru_cache(None)
def br_s6_module():
    mod = br_compile('brs6', BR_S6_CPP, BR_S6_CUDA, BR_FUNCS)
    mod.br_init()
    return mod


@functools.lru_cache(None)
def br_u16_module():
    # S6's H_H CUDA (the heavy WGMMA reads prepare-widened u16 scale words): measurement-only reference
    mod = br_compile('bru16', M128_HH_PM_CPP, BR_U16_CUDA, M128_HH_PM_FUNCTIONS)
    mod.idist_init()
    return mod


def br_modules():
    m128_hh_pm_module()
    br_hh_module()
    if BR_SET == 'adaptive':
        br_adapt_module(64);br_adapt_module(128)
        return
    if BR_SET == 'tune':
        br_wp_module()
        br_pair_module(2)
        br_pair_module(4)
        return
    if BR_M == 128:
        br_u16_module()
    else:
        br_s6_module()


def br_ilv_rows(dev):
    # Input row order such that H_H's issue_prepare (which tiles row groups [g(2i), g(2i+1), g(256+2i), g(256+2i+1)])
    # leaves each 64-row W13 tile as G0-7,U0-7,G8-15,U8-15,G16-23,U16-23,G24-31,U24-31 of its 32 channels.
    perm = torch.empty(8192, dtype=torch.long)
    for i in range(128):
        for q, pos in enumerate((32 * i, 32 * i + 16, 4096 + 32 * i, 4096 + 32 * i + 16)):
            perm[pos:pos + 16] = torch.tensor([32 * i + 8 * q + k for k in range(8)] + [4096 + 32 * i + 8 * q + k for k in range(8)])
    return perm.to(dev)


def br_ilv_bank(weights):
    w = dict(weights, post_norm_w=weights.get('post_expert_norm', weights.get('post_norm_w')))
    perm = br_ilv_rows(w['w13'].device)
    w['w13'] = w['w13'].index_select(1, perm)
    w['w13_sf'] = w['w13_sf'].index_select(1, perm)
    bank = m128_hh_issue_prepare(w)
    return {k: bank[k] for k in ('w13', 'w13_sf')}


def br_wide(bank):
    # Wide-bank layout (kinds with VEC flag 4): per 16-row group, codes [K32 block][thread][8B] -> [K64 pair][thread][2][8B]
    # and scales [K32 block][row8][4B] -> [K128 stage][row8][4 blocks][4B]; a permutation of the same u8 bytes.
    out = {}
    for k in ('w13', 'w2'):
        q, sf = bank[k], bank[k + '_sf']
        e, ng = sf.shape[0], sf.shape[1]
        out[k] = q.view(torch.uint8).reshape(e, ng, 64, 2, 32, 8).transpose(3, 4).contiguous().view(q.dtype).reshape(q.shape)
        out[k + '_sf'] = sf.reshape(e, ng, 32, 4, 8, 4).transpose(3, 4).contiguous().reshape(sf.shape)
    return out


def br_prepare(weights):
    hh = hh_prepare(weights)
    dev = hh['base']['w13'].device
    hh.update(side=torch.cuda.Stream(device=dev), side_hi=torch.cuda.Stream(device=dev, priority=-1), fork_ev=torch.cuda.Event(), join_ev=torch.cuda.Event(), mod=br_hh_module(),
              sms=torch.cuda.get_device_properties(dev).multi_processor_count)
    if BR_SET in ('tune','adaptive'):
        hh['base_ilv'] = br_ilv_bank(weights)
        wide = br_wide(dict(hh['base'], w13=hh['base_ilv']['w13'], w13_sf=hh['base_ilv']['w13_sf']))
        hh['base_wide'] = dict(hh['base'], **wide)
        perm = dict(hh['base'], **hh['base_ilv'])
        for k in ('w13_sf','w2_sf'):
            sf=perm[k];v=sf.contiguous().reshape(-1).view(torch.int32).to(torch.int64)&0xffffffff
            perm[k]=(((v<<4)|(v>>28))&0xffffffff).to(torch.int32).view(torch.uint8).reshape(sf.shape)
        hh['base_perm']=perm
        hh['base_pw']=dict(perm,**br_wide(perm))
    elif BR_M == 128:
        hh['u16_sf'] = {k: ((hh['base'][k].to(torch.int32) << 4) + 0x7000).to(torch.int16) for k in ('w13_sf', 'w2_sf')}
    else:
        hh['s6'] = wi_prepare(weights)
    return hh


# Heavy kernels on H_H's buffers (gemm.cu.inc kinds; u8 scale codes): 0/1 W13/W2 N32 on the heavy-only
# (count>16) T32 list; 4/8 W13/W2 N32 groups with >16 rows, 5/9 N16 groups with <=16 rows (heavy remainders);
# 12 W2 N16 for experts with 8<count<=16 (T8 list). 'n' = NT1 for count<=16, 'n8' = W2 NT1 for count<=8.
# Spec: (W13 side, W13 main, W2 side, W2 main, mode); seq = one stream per phase, phase = fork/join per
# phase, full = fork once (side W13 -> side W2 || main W13 -> main W2), join before the finalizer.
BR_ARMS = {
    'HH': None,        # H_H on u8 codes: M128 = exact S7; M256 = the same policy (ec44b322 H16g1056)
    'S6': 's6',        # M256: exact S6 mixed_parallel
    'U16': 'u16',      # M128: exact u16-word H_H (770809cf / S6): the cost of S7's in-register word
    'R_s6': 's6R',     # M256: 110599ba winner R_full on S6's buffers
    'HH_u8': 'u8',     # H_H with its heavy WGMMA replaced by this lab's typed-smem kernel
    'N_phase': (('n',), (0,), ('n',), (1,), 'phase'),
    'N_full': (('n',), (0,), ('n',), (1,), 'full'),
    'N_swap': ((0,), ('n',), (1,), ('n',), 'full'),
    'L_swap': ((20,), ('n',), (21,), ('n',), 'full'),
    'LI_swap': ((26,), ('n',), (21,), ('n',), 'full'),
    'NR_full': (('n', 5), (4,), ('n', 9), (8,), 'full'),
    'NRm_full': (('n',), (4, 5), ('n',), (8, 9), 'full'),
    'B8_full': (('n',), (0,), ('n8', 12), (1,), 'full'),
    'B8R_full': (('n', 5), (4,), ('n8', 12, 9), (8,), 'full'),
}
BR_GRID = 1056 if BR_M == 128 else 2112
# Heavy-kernel lab vs LI4 = LI_swap (701d2867: gate/up rows interleaved by 8 in prepare, SwiGLU in registers) with the
# finalizer split over 4 blocks per token (5e197b5e: 0.9850 / 0.9885). 5e197b5e also showed heavy alone 99.8 / 177.7 us and
# NT1 alone 127.6 / 133.7 us against 195.0 / 275.0 together: the branches barely overlap (heavy launched first takes every
# SM's shared memory, 3 W13 CTAs per SM, so NT1's CTAs wait). G<g>: heavy as a persistent grid of g CTAs per SM so NT1
# runs beside it; p5: NT1 grid halved (one CTA per SM). LIW: wide banks (3 LDS.128 per stage instead of 8). Probes
# (timing only): PH heavy alone (default grid, 1 or 2 CTAs per SM), PH_nx / PH_nw heavy alone without the X / code+scale
# copies (what the L2 X re-reads and the HBM weight stream cost), PN_p5 NT1 alone at half grid.
LI = ((26,), ('n',), (21,), ('n',), 'full')
HV = ((26,), (), (21,), (), 'full')
BR_TUNE = {
    'LI4': LI + (dict(fin=4),),
    'LIW': ((30,), ('n',), (31,), ('n',), 'full', dict(fin=4)),
    'TYP': ((44,), ('n',), (45,), ('n',), 'full', dict(fin=4)),
    'TYPW': ((46,), ('n',), (47,), ('n',), 'full', dict(fin=4)),
    'WP': ((26,), ('n',), (48,), ('n',), 'full', dict(fin=104, wp=True)),
    'PAIR2': ((50,), ('n',), (51,), ('n',), 'full', dict(fin=4, pair=2)),
    'PAIR4': ((50,), ('n',), (51,), ('n',), 'full', dict(fin=4, pair=4)),
    'DESC': ((52,), ('n',), (53,), ('n',), 'full', dict(fin=4)),
}
BR_PROBES = ()
BR_ARMS.update(BR_TUNE)
BR_ADAPT = {'LI4': BR_TUNE['LI4'], 'LIW': BR_TUNE['LIW']}
BR_ADAPT['PERM']=((54,),('n',),(55,),('n',),'full',dict(fin=4))
BR_ADAPT['PERMW']=((56,),('n',),(57,),('n',),'full',dict(fin=4))
for nt in (64,128):
    for suffix,kind in (('D',62),('P',66),('PW',68)):
        BR_ADAPT[f'AD{nt}{suffix}']=((kind,),('n',),(kind+1,),('n',),'full',dict(fin=4,adapt=nt))
if BR_SET=='adaptive':
    BR_ARMS.update(BR_ADAPT)
BR_ONLY = {'S6': 256, 'R_s6': 256, 'U16': 128}
if BR_SET in ('tune','adaptive'):
    BR_CONTROL = 'LI4'
    BR_NAMES = list(BR_TUNE if BR_SET=='tune' else BR_ADAPT)
else:
    BR_CONTROL = 'HH' if BR_M == 128 else 'S6'
    BR_NAMES = [BR_CONTROL] + [a for a in BR_ARMS if a not in (BR_CONTROL, 'N_swap', 'L_swap', 'LI_swap') and a not in BR_TUNE and BR_ONLY.get(a, BR_M) == BR_M]


def br_cur():
    return torch.cuda.current_stream().cuda_stream


def br_fork(hh, side, main, key='side'):
    parent = torch.cuda.current_stream()
    hh['fork_ev'].record(parent)
    hh[key].wait_event(hh['fork_ev'])
    with torch.cuda.stream(hh[key]):
        side(hh[key].cuda_stream)
        hh['join_ev'].record()
    main(parent.cuda_stream)
    parent.wait_event(hh['join_ev'])


def br_s6_plan(data, hh, spec):
    bank = hh['s6']
    plan = wi_plan(data, bank, 6, 2112)
    v13, v2 = plan['v13'], plan['v2']
    plan = dict(plan, out=v2[19])
    if spec == 's6':
        return plan
    mod = br_s6_module()
    m, start = plan['M'], plan['start']
    q13 = [bank['w13_rec'], bank['w13_rec_sf']] + [v13[i] for i in (2, 3, 4, 5, 6, 7, 8, 14, 9, 15, 16)]
    q2 = [bank['w2_rec'], bank['w2_rec_sf']] + [v2[i] for i in (2, 3, 4, 5, 6, 7, 8, 14, 9, 15, 16)]
    p13, p2 = [t.data_ptr() for t in q13], [t.data_ptr() for t in q2]
    rp = [v2[i].data_ptr() for i in (11, 4, 5, 6, 7, 8, 13, 7, 7)]
    fp = [v2[i].data_ptr() for i in (5, 11, 12, 13, 15, 16, 19)]
    owners = (q13, q2, v13, v2, plan)

    def route():
        _ = owners
        mod.br_route(rp, m, start, 0, br_cur())

    def body():
        _ = owners
        br_fork(hh, lambda s: (mod.br_run(p13, 5, 2112, m, s), mod.br_run(p2, 9, 2112, m, s)),
                lambda s: (mod.br_run(p13, 4, 2112, m, s), mod.br_run(p2, 8, 2112, m, s)))

    def finish():
        _ = owners
        mod.br_finish(fp, m, start, 1, br_cur())
        return v2[19]
    return dict(plan, steps=(route, body, finish))


def br_plan(data, hh, arm):
    spec = BR_ARMS[arm]
    if spec in ('s6', 's6R'):
        return br_s6_plan(data, hh, spec)
    base, native = hh['base'], hh['native']
    plan = m128_hh_pm_plan(data, base, native)
    if spec is None:
        return plan
    opts = spec[5] if isinstance(spec, tuple) and len(spec)>5 else {}
    mod = br_wp_module() if opts.get('wp') else br_pair_module(opts['pair']) if opts.get('pair') else br_adapt_module(opts['adapt']) if opts.get('adapt') else hh['mod']
    hmod = m128_hh_pm_module()
    v13, tasks = plan['v13'], plan['tasks']
    if opts.get('wp'):
        scratch = torch.empty((v13[16].shape[0],256),device=v13[16].device)
        v13[16]=scratch;plan['v2'][16]=scratch
    t8 = torch.empty_like(tasks[0])
    x, c, off, order, slots, act = v13[3], v13[4], v13[5], v13[6], v13[8], v13[9]
    ids, tw, pos, y, ss, out = v13[11], v13[12], v13[13], v13[15], v13[16], v13[19]
    m = x.shape[0]
    start = data['config'].get('local_start', 0)
    h13 = [base['w13'], base['w13_sf'], base['w13_gs'], x, c, off, order, tasks[1], slots, base['pnw'], act, y, ss]
    h2 = [base['w2'], base['w2_sf'], base['w2_gs'], act, c, off, order, tasks[1], slots, base['pnw'], act, y, ss]
    h28 = h2[:7] + [t8] + h2[8:]
    h13i = [hh['base_ilv']['w13'], hh['base_ilv']['w13_sf']] + h13[2:] if 'base_ilv' in hh else h13
    wb = hh.get('base_wide', base)
    h13w, h2w = [wb['w13'], wb['w13_sf']] + h13[2:], [wb['w2'], wb['w2_sf']] + h2[2:]
    pb=hh.get('base_perm',base)
    h13p,h2p=[pb['w13'],pb['w13_sf']]+h13[2:],[pb['w2'],pb['w2_sf']]+h2[2:]
    pwb=hh.get('base_pw',base)
    h13pw,h2pw=[pwb['w13'],pwb['w13_sf']]+h13[2:],[pwb['w2'],pwb['w2_sf']]+h2[2:]
    nv = [x, base['pnw'], native['w13p'], native['w2p'], base['w13_gs'], base['w2_gs'], c, off, slots, act, y, ss]
    p13, p2, p28, p13i, p13w, p2w, p13p, p2p, p13pw, p2pw, pn = ([t.data_ptr() for t in v] for v in (h13, h2, h28, h13i, h13w, h2w, h13p, h2p, h13pw, h2pw, nv))
    rp = [t.data_ptr() for t in (ids, c, off, order, tasks[0], slots, pos, tasks[1], t8)]
    fp = [t.data_ptr() for t in (off, ids, tw, pos, y, ss, out)]
    owners = (plan, h13, h2, h28, h13i, h13w, h2w, h13p, h2p, h13pw, h2pw, nv, t8)
    plan = dict(plan, tasks=list(tasks) + [t8])
    ostages = 2
    while ostages > 1 and hmod.pm_nt_smem(ostages, m) > M128_HH_ISSUE_OPTS['pm_nt']:
        ostages -= 1

    def hh_small(first):
        hmod.pm_nt_run(pn, first, 1, m, ostages, native['grid'] * hmod.pm_nt_resident(ostages, m, first, 1), br_cur())

    if spec in ('u16', 'u8'):
        if spec == 'u16':
            umod = br_u16_module()
            sf = hh['u16_sf']
            vu = [[base['w13'], sf['w13_sf']] + list(plan['v13'][2:]), [base['w2'], sf['w2_sf']] + list(plan['v2'][2:])]
            for v in vu:
                v[7] = tasks[1]
            qu = [[t.data_ptr() for t in v] for v in vu]
            cap = v13[9].shape[0]
            owners = owners + (vu,)

        def phase(first):
            def go():
                _ = owners
                hh_small(first)
                if spec == 'u16':
                    umod.idist_run(qu[0] if first else qu[1], first, 1, 1056, start, 1, 0, m, cap, br_cur())
                else:
                    mod.br_run(p13 if first else p2, 0 if first else 1, BR_GRID, m, br_cur())
            return go
        s = plan['steps']
        return dict(plan, steps=(s[0], phase(1), phase(0), s[3]))

    stages = 2
    while stages > 1 and mod.pmz_smem(stages, m) > BR_OPTS['pmz']:
        stages -= 1
    s13, k13, s2, k2, mode = spec[:5]
    opts = spec[5] if len(spec) > 5 else {}
    grids = {f: max(1, int(hh['sms'] * mod.pmz_resident(stages, m, f) * opts.get('pg', 1.))) for f in (0, 1)}
    hgrid = opts['gs'] * hh['sms'] if 'gs' in opts else opts.get('grid', BR_GRID)
    skey = 'side_hi' if opts.get('hi') else 'side'

    def run(first, kinds, stream):
        for k in kinds:
            if k in ('n', 'n8'):
                mod.pmz_run(pn, first, 16 if k == 'n' else 8, m, stages, grids[first], stream)
            else:
                ptr = p28 if k == 12 else p13pw if k in (56,68) else p2pw if k in (57,69) else p13p if k in (54,66) else p2p if k in (55,67) else p13w if k in (30,46,64) else p2w if k in (31,47,65) else p13i if k in (26,29,40,42,44,50,52,60,62) else p13 if first else p2
                mod.br_run(ptr, k, hgrid, m, stream)

    def route():
        _ = owners
        mod.br_route(rp, m, start, 1, br_cur())

    def finish():
        _ = owners
        if not opts.get('nofin'):
            mod.br_finish(fp, m, start, opts.get('fin', 1), br_cur())
        return out

    if mode == 'full':
        def body():
            _ = owners
            br_fork(hh, lambda st: (run(1, s13, st), run(0, s2, st)), lambda st: (run(1, k13, st), run(0, k2, st)), skey)
        steps = (route, body, finish)
    else:
        def phase(first, side, main):
            def go():
                _ = owners
                if mode == 'seq':
                    run(first, side + main, br_cur())
                else:
                    br_fork(hh, lambda st: run(first, side, st), lambda st: run(first, main, st))
            return go
        steps = (route, phase(1, s13, k13), phase(0, s2, k2), finish)
    return dict(plan, steps=steps)


def br_call(data, hh, arm):
    for fn in br_plan(data, hh, arm)['steps']:
        out = fn()
    return out


def br_error(out, want):
    if not bool(torch.isfinite(out).all()):
        return float('inf'), float('inf')
    diff = out.float() - want.float()
    den = want.float().norm(dim=-1)
    return float(diff.norm() / want.float().norm().clamp_min(1e-12)), float((diff.norm(dim=-1) / den.clamp_min(den.mean() * .1 + 1e-12)).max())


def br_scratch(plan):
    unique = {}
    for v in (plan['v13'], plan['v2']):
        for i in (4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19):
            unique[v[i].data_ptr()] = v[i]
    for t in plan.get('tasks', []):
        unique[t.data_ptr()] = t
    return list(unique.values())

def br_timed(graph, ev0, ev1):
    lab_cold_l2()
    ev0.record()
    graph.replay()
    ev1.record()
    ev1.synchronize()
    return ev0.elapsed_time(ev1) * 1000.


def br_pair(a, b, ev0, ev1, repeats=6):
    aa, bb = [], []
    for r in range(repeats):
        br_need(.3, 'cold pair')
        for g, s in (((a, aa), (b, bb)) if r % 2 == 0 else ((b, bb), (a, aa))):
            s.append(br_timed(g, ev0, ev1))
    return statistics.median(aa), statistics.median(bb)


def br_phases(data, bank, arm, ev0, ev1):
    steps = br_plan(data, bank, arm)['steps']
    for f in steps:
        f()
    vals = []
    for k, fn in enumerate(steps):
        br_need(1., 'phase graph')
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        ss = []
        for _ in range(4):
            for pre in steps[:k]:
                pre()
            ss.append(br_timed(g, ev0, ev1))
        vals.append(statistics.median(ss))
    return vals


BR_META_TUNE = ('META heavy-kernel lab at M%d vs LI4 (LI_swap 701d2867: N_full, heavy N32 lean K loop, W13 gate/up rows '
                'interleaved by 8 with SwiGLU in registers, grid %d, launched first on the side stream || NT1<=16; live route; one '
                'join; finalizer split over 4 blocks per token, 5e197b5e). LIW: wide packed banks; TYP/TYPW: static typed '
                'activation smem (base/wide); WP: four W2 per-warp norm partials summed by finalizer; PAIR2/4: one decode '
                'feeds two separate N32 WGMMA accumulators, 2/4 pipeline stages (no N64 WGMMA). '
                'Adaptive lab: AD64/128D/P/PW uses only live independent N32 groups, capacity 64/128, S2; '
                'D=hoisted descriptor, P=rotated packed scale words, PW=P+wide. PERM/PERMW applies P/PW on N32 S4. '
                '12 seeds x3 fresh routes, lab_cold_l2, AB/BA x6, A:A.')
BR_META = {
    128: 'META M128 vs HH = exact S7 H_H (u8 codes; route128 zero-fill + pm_meta SS clear; NT1<=16 then WGMMA32>16 per phase; finish). '
         'U16 = exact u16-word H_H (S6). ',
    256: 'META M256 vs S6 = exact S6 mixed_parallel. HH = H_H policy on u8 (NT1<=16, WGMMA32>16). R_s6 = 110599ba R_full (S6 buffers). ',
}
BR_META_TAIL = ('HH_u8: H_H with this lab\'s typed-smem WGMMA. Other arms: live route 1CTA (T32/T8 lists), small W13 zeroes its SS rows, '
                'n=NT1<=16; N: heavy N32; NR: heavy remainder rows<=16 on N16 beside NT1; NRm: remainder on the heavy stream; '
                'B8: W2 NT1<=8 + N16 for 8<count<=16 beside it. phase=fork/join per phase, full=fork once, join at finish. '
                '12 seeds x3 fresh routes, judge-style read flush (lab_cold_l2), AB/BA x6 vs control, A:A.')


def lab(deadline, weights, state=None):
    global BR_DEADLINE
    BR_DEADLINE = deadline
    start = time.monotonic()
    names = BR_NAMES
    ctl = BR_CONTROL
    lines = [BR_META_TUNE % (BR_M, BR_GRID) if BR_SET in ('tune','adaptive') else BR_META[BR_M] + BR_META_TAIL,
             f'GATE TASK FP32 L2<=.015,row<=.075,finite before+after timing (failures recorded, still timed); bit=equal to {ctl} output (diagnostic). Phase means are diagnostics.']
    rows, phases, aa = [], [], []
    try:
        br_need(25., 'before compile')
        br_modules()
        lines.append(f'BUILD s={time.monotonic() - start:.1f} left={deadline - time.monotonic():.1f}')
        dev = next(t.device for t in weights.values() if torch.is_tensor(t))
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        peak = 0.
        for seed in range(12):
            br_need(max(8., peak * 1.3 + 1.), f'before seed{seed}')
            tick = time.monotonic()
            w = dc_weights(5353 + 7919 * seed, dev)
            bank = br_prepare(w)
            ds = dc_fixtures(w, BR_M)
            wants = dc_oracles(ds, w)
            graphs = {}
            for arm in names:
                br_need(1., 'graph')
                graphs[arm] = MWGraph(lambda d, a=arm: br_call(d, bank, a), ds[0])
            per = []
            for fi, (data, want) in enumerate(zip(ds, wants)):
                for g in graphs.values():
                    g.refresh(data)
                base = graphs[ctl].replay().clone()
                torch.cuda.synchronize()
                err = {a: br_error(graphs[a].replay().clone(), want) for a in names}
                bit = {a: bool(torch.equal(base, graphs[a].replay())) for a in names}
                a0, a1 = br_pair(graphs[ctl], graphs[ctl], ev0, ev1)
                aa.append(a1 - a0)
                us = {ctl: a0}
                ratio = {ctl: 1.}
                for a in names[1:]:
                    c, x = br_pair(graphs[ctl], graphs[a], ev0, ev1)
                    us[a] = x
                    ratio[a] = x / c
                for a in names:
                    e2 = br_error(graphs[a].replay().clone(), want)
                    err[a] = (max(err[a][0], e2[0]), max(err[a][1], e2[1]))
                rows.append(dict(seed=seed, fixture=fi, us=us, ratio=ratio, err=err, bit=bit))
                per.append(ratio)
            br_need(4., 'phases')
            phases.append({a: br_phases(ds[0], bank, a, ev0, ev1) for a in names})
            lines.append(f'S{seed} ' + ' '.join(f'{statistics.mean(p[a] for p in per):.3f}' for a in names[1:]))
            peak = max(peak, time.monotonic() - tick)
            del graphs, bank
    except BRBudget as e:
        lines.append('BUDGET ' + str(e))
    if rows:
        lines.append('ORDER ' + ' '.join(names[1:]))
        for a in names:
            e = max(r['err'][a][0] for r in rows)
            rw = max(r['err'][a][1] for r in rows)
            rr = [r['ratio'][a] for r in rows]
            lines.append(f'MEAN {a} n={len(rows)} us={statistics.mean(r["us"][a] for r in rows):.2f} med={statistics.median(r["us"][a] for r in rows):.2f} '
                         f'pair={statistics.mean(rr):.4f} lo/hi={min(rr):.3f}/{max(rr):.3f} L2/row={e:.5f}/{rw:.5f} TASK={"PASS" if e <= .015 and rw <= .075 else "FAIL"} bit={sum(r["bit"][a] for r in rows)}')
        if phases:
            for a in names:
                k = len(phases[0][a])
                lines.append(f'PH {a} ' + '/'.join(f'{statistics.mean(p[a][i] for p in phases):.1f}' for i in range(k)))
        lines.append(f'AA signed/abs/max={statistics.mean(aa):.3f}/{statistics.mean(abs(x) for x in aa):.3f}/{max(abs(x) for x in aa):.3f}')
    lines.append(f'END M={BR_M} seeds={len(rows) // 3}/12 used={time.monotonic() - start:.1f} left={deadline - time.monotonic():.1f}')
    return lines
