# Self-contained grouped-masked Humming M128 lab
import base64
import json as pf_json
import math
import functools as rg_functools
import os as pf_os
import statistics
import time
import time as pf_time
import zlib
import torch
import triton
import triton.language as tl
import torch as pf_torch
import triton as pf_triton
import triton.language as pf_tl
from humming import ops as pf_humming_ops
from humming.config import GemmType as pf_GemmType, WeightScale2Type as pf_WeightScale2Type
from humming.kernel.humming import HummingKernel as pf_HummingKernel
from humming.layer import HummingMethod as pf_HummingMethod
from humming.schema.modelopt import ModeloptNvfp4WeightSchema as pf_ModeloptNvfp4WeightSchema
from humming.tune import get_heuristics_config as pf_get_heuristics_config

import torch

import torch as rv_torch

import torch

import torch as rv_torch

import torch as hv_torch

import torch as rr_torch

import math as fx_math

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

import zlib

import torch

import triton

import triton.language as tl

@triton.jit
def mw_route(IDS, COUNTS, SLOTS, POS, M: tl.constexpr, START: tl.constexpr, B: tl.constexpr):
    e = tl.program_id(0)
    t = tl.arange(0, B)
    hit = tl.full((B,), False, tl.int1)
    slot = tl.full((B,), 0, tl.int32)
    for j in tl.static_range(8):
        yes = tl.load(IDS + t * 8 + j, t < M, -999) == e + START
        hit = hit | yes
        slot = tl.where(yes, j, slot)
    rank = tl.cumsum(hit.to(tl.int32)) - 1
    count = tl.sum(hit.to(tl.int32))
    tl.store(COUNTS + e, count)
    tl.store(SLOTS + e * M + rank, t * 8 + slot, hit)
    tl.store(POS + t * 8 + slot, rank, hit)

import math as pf_math

import os as pf_os

import torch as pf_torch

import torch.nn.functional as pf_F

import triton as pf_triton

import triton.language as pf_tl

pf_N_EXPERTS = 256

pf_TOP_K = 8

@pf_triton.jit
def pf__offsets_kernel(counts_ptr, starts_ptr, psum_ptr, ALIGN: pf_tl.constexpr):
    e = pf_tl.arange(0, 32)
    c = pf_tl.load(counts_ptr + e)
    padded = (c + ALIGN - 1) // ALIGN * ALIGN
    incl = pf_tl.cumsum(padded, 0)
    st = incl - padded
    pf_tl.store(starts_ptr + e, st)
    pf_tl.store(psum_ptr + e, st + c)

@pf_triton.jit
def pf__gather_norm_kernel(p_ptr, w_ptr, slot_e_ptr, slot_pos_ptr, starts_ptr, a_ptr, row_of_ptr, D: pf_tl.constexpr):
    """Gather already-normalized input into local-route rows."""
    t = pf_tl.program_id(0).to(pf_tl.int64)
    c = pf_tl.arange(0, D)
    x = pf_tl.load(p_ptr + t * D + c).to(pf_tl.float32)
    x = x.to(pf_tl.bfloat16)
    for k in pf_tl.static_range(8):
        e = pf_tl.load(slot_e_ptr + t * 8 + k)
        if e >= 0:
            r = pf_tl.load(starts_ptr + e) + pf_tl.load(slot_pos_ptr + t * 8 + k)
            pf_tl.store(a_ptr + r.to(pf_tl.int64) * D + c, x)
            pf_tl.store(row_of_ptr + t * 8 + k, r)
        else:
            pf_tl.store(row_of_ptr + t * 8 + k, -1)

@pf_triton.jit
def pf__combine_kernel(y_ptr, row_of_ptr, slot_e_ptr, w_ptr, gs_ptr, pw_ptr, out_ptr, D: pf_tl.constexpr):
    """out[t] = sum_k w[t,k] * rms(gs_e * y[row] * (1 + w_post), 1e-8) over the token's local routes."""
    t = pf_tl.program_id(0).to(pf_tl.int64)
    c = pf_tl.arange(0, D)
    pw = pf_tl.load(pw_ptr + c)
    acc = pf_tl.zeros([D], pf_tl.float32)
    for k in pf_tl.static_range(8):
        r = pf_tl.load(row_of_ptr + t * 8 + k)
        if r >= 0:
            e = pf_tl.load(slot_e_ptr + t * 8 + k)
            y = (pf_tl.load(y_ptr + r.to(pf_tl.int64) * D + c).to(pf_tl.float32) * pf_tl.load(gs_ptr + e)).to(pf_tl.bfloat16).to(pf_tl.float32) * pw
            y = y * pf_tl.rsqrt(pf_tl.sum(y * y, 0) / D + 1e-08)
            acc += pf_tl.load(w_ptr + t * 8 + k) * y
    pf_tl.store(out_ptr + t * D + c, acc.to(pf_tl.bfloat16))

pf_BN = 128

pf_D = 4096

import json as pf_json

import os as pf_os

import torch as pf_torch

class pf__Experts:
    """One GPU's experts [start, start + count) in Humming's layout, and the kernels for each bs."""

    def __init__(self, weights, start: int, count: int, tma=False):
        self.tma = tma
        self.gemm_type = pf_GemmType.INDEXED
        device = weights['w13'].device
        self.count = count
        self.d_expert = weights['w13'].shape[1] // 2
        self.d_hidden = weights['w2'].shape[1]
        self.layer = pf_torch.nn.Module()
        sl = slice(start, start + count)
        for name, n, k, n_stacks in (('w13', 2 * self.d_expert, self.d_hidden, [self.d_expert] * 2), ('w2', self.d_hidden, self.d_expert, [self.d_hidden])):
            checkpoint = {'weight': weights[name][sl].contiguous(), 'weight_scale': weights[f'{name}_sf'][sl].contiguous(), 'weight_scale_2': weights[f'{name}_gs'][sl].contiguous()}
            schema, tensors = pf_ModeloptNvfp4WeightSchema().convert_humming(tensors=checkpoint, shape_n_stacks=n_stacks, shape_k_stacks=[k], param_dtype=pf_torch.bfloat16, num_experts=count)
            assert schema.weight_scale_2_type == pf_WeightScale2Type.TENSOR, schema
            for key, tensor in tensors.items():
                setattr(self.layer, f'{name}_{key}', pf_torch.nn.Parameter(tensor, requires_grad=False))
            pf_HummingMethod.prepare_layer_meta(layer=self.layer, shape_n=n, shape_k=k, weight_schema=schema, num_experts=count, pad_n_to_multiple=256, pad_k_to_multiple=128, torch_dtype=pf_torch.bfloat16, sublayer_name=name)
            pf_HummingMethod.transform_humming_layer(self.layer, sublayer_name=name)
        self.locks = pf_torch.zeros(1024, dtype=pf_torch.int32, device=device)
        self.expert_map = pf_torch.full((pf_N_EXPERTS,), -1, dtype=pf_torch.int32, device=device)
        self.expert_map[sl] = pf_torch.arange(count, dtype=pf_torch.int32, device=device)
        self.kernels = {'w13': {}, 'w2': {}}
        self.plans = {}

    def weight(self, name: str) -> dict:
        return {'weight': getattr(self.layer, f'{name}_weight'), 'weight_scale': getattr(self.layer, f'{name}_weight_scale'), 'weight_scale_2': getattr(self.layer, f'{name}_weight_scale_2')}

    def routes(self, bs: int) -> int:
        """The GEMM's M for Humming's heuristics: the routes expected on this GPU."""
        return max(1, bs * pf_TOP_K * self.count // pf_N_EXPERTS)

    def tuning(self, name: str, routes: int) -> str:
        config = dict(pf_get_heuristics_config(self.layer.humming_metas[name], shape_m=routes, gemm_type=self.gemm_type))
        if self.tma and pf_torch.cuda.get_device_capability()[0] == 9:
            config.update(use_tma=True, use_tma_a=False, use_tma_as=False, use_tma_c=False, use_mbarrier=True, use_warp_spec=True)
        return pf_json.dumps(config)

    def compile(self, batch_sizes) -> None:
        """Compile, in parallel, every kernel these batch sizes need that isn't compiled yet."""
        wanted = {bs: {name: self.tuning(name, self.routes(bs)) for name in self.kernels} for bs in batch_sizes}
        for name, kernels in self.kernels.items():
            new = sorted({t[name] for t in wanted.values()} - kernels.keys())
            if not new:
                continue
            launch = pf_HummingKernel.prepare_kernels(self.layer.humming_metas[name].to_str(), pf_json.dumps({'use_f16_accum': False, 'gemm_type': self.gemm_type.value}), [[i, i + 1, pf_json.loads(t)] for i, t in enumerate(new)])
            for t, (_, _, kernel_id, num_sms) in zip(new, launch.view(-1, 4).tolist()):
                kernels[t] = pf_torch.tensor([0, 1 << 30, kernel_id, num_sms], dtype=pf_torch.int64)
        for bs, tunings in wanted.items():
            self.plans[bs] = (self.routes(bs), {name: (pf_json.loads(t)['block_shape'][0], self.kernels[name][t]) for name, t in tunings.items()})

    def plan(self, bs: int):
        bucket = 1 << (max(bs, 1) - 1).bit_length()
        if bucket not in self.plans:
            self.compile([bucket])
        return (self.routes(bs), self.plans[bucket][1])

import os as pf_os

import torch as pf_torch

import torch.nn.functional as pf_F

import triton as pf_triton

import triton.language as pf_tl

import base64

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

@pf_triton.jit
def m128_swiglu(H, OUT, COUNTS, GATE_GS, UP_GS, STARTS, BR: pf_tl.constexpr, D: pf_tl.constexpr):
    e = pf_tl.program_id(0)
    n = pf_tl.load(COUNTS + e)
    r0 = pf_tl.program_id(1) * BR
    if r0 >= n:
        return
    r = r0 + pf_tl.arange(0, BR)
    ok = r < n
    row = (pf_tl.load(STARTS + e) + r).to(pf_tl.int64)
    c = pf_tl.arange(0, D)
    g = pf_tl.load(H + row[:, None] * (2 * D) + c[None, :], ok[:, None], 0).to(pf_tl.float32) * pf_tl.load(GATE_GS + e)
    u = pf_tl.load(H + row[:, None] * (2 * D) + D + c[None, :], ok[:, None], 0).to(pf_tl.float32) * pf_tl.load(UP_GS + e)
    g = g.to(pf_tl.bfloat16).to(pf_tl.float32)
    u = u.to(pf_tl.bfloat16).to(pf_tl.float32)
    pf_tl.store(OUT + row[:, None] * D + c[None, :], (g * pf_tl.sigmoid(g) * u).to(pf_tl.bfloat16), ok[:, None])

def m128_bank(w):
    hw = dict(w, w13_gs=torch.ones_like(w['w13_gs']))
    b = pf__Experts(hw, 0, 32)
    b.gemm_type = pf_GemmType.GROUPED_MASKED
    return b

def m128_cfg(bank, name, routes, *, block=None, stages=None, stream=None, sms=None, tma=False):
    out = dict(pf_get_heuristics_config(bank.layer.humming_metas[name], shape_m=routes, gemm_type=pf_GemmType.GROUPED_MASKED))
    if block is not None:
        bm = int(block)
        out['block_shape'] = (bm, out['block_shape'][1], out['block_shape'][2])
        out['warp_shape'] = (bm, out['warp_shape'][1], out['warp_shape'][2])
    if stages is not None:
        out['num_stages'] = int(stages)
    if stream is not None:
        out['use_stream_k'] = bool(stream)
    if sms is not None:
        out['num_sms'] = int(sms)
    if tma:
        out.update(use_tma=True, use_tma_a=False, use_tma_as=False, use_tma_c=False, use_mbarrier=True, use_warp_spec=True)
    return out

def m128_launch(bank, name, cfg):
    meta = bank.layer.humming_metas[name].to_str()
    comp = pf_json.dumps({'use_f16_accum': False, 'gemm_type': pf_GemmType.GROUPED_MASKED.value})
    x = pf_HummingKernel.prepare_kernels(meta, comp, [[0, 1, cfg]])
    vals = x.view(-1, 4)[0].tolist()
    _, _, kid, sms = vals
    return pf_torch.tensor([0, 1 << 30, kid, sms], dtype=pf_torch.int64)

def m128_route(d):
    x = d['hidden_states']; t = x.shape[0]; dev = x.device
    counts = pf_torch.empty(32, device=dev, dtype=pf_torch.int32)
    slots = pf_torch.empty((32, t * 8), device=dev, dtype=pf_torch.int32)
    pos = pf_torch.empty((t, 8), device=dev, dtype=pf_torch.int32)
    mw_route[(32,)](d['topk_ids'], counts, slots, pos, t, 0, pf_triton.next_power_of_2(t), num_warps=4)
    local = pf_torch.where((d['topk_ids'] >= 0) & (d['topk_ids'] < 32), d['topk_ids'], -1).int()
    stride = pf_triton.cdiv(t, 128) * 128
    starts = pf_torch.arange(32, device=dev, dtype=pf_torch.int32) * stride
    return x, t, counts, pos, local, stride, starts

def m128_call(d, bank, pair):
    x, t, counts, pos, local, stride, starts = m128_route(d)
    a = pf_torch.empty((32 * stride, 4096), device=x.device, dtype=pf_torch.bfloat16)
    row_of = pf_torch.empty((t, 8), device=x.device, dtype=pf_torch.int32)
    pf__gather_norm_kernel[t,](x, pf_torch.ones(4096, device=x.device), local, pos, starts, a, row_of, D=4096, num_warps=4)
    h = pf_torch.empty((32 * stride, 8192), device=x.device, dtype=pf_torch.bfloat16)
    inter = pf_torch.empty((32 * stride, 4096), device=x.device, dtype=pf_torch.bfloat16)
    y = pf_torch.empty_like(inter)
    routes = max(1, t)
    pf_humming_ops.launch_kernel(configs=pair['w13'], inputs=a, outputs=h, **bank.weight('w13'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)
    m128_swiglu[32, pf_triton.cdiv(t, 4)](h, inter, counts, d['_gate_gs'], d['_up_gs'], starts, BR=4, D=4096, num_warps=8)
    pf_humming_ops.launch_kernel(configs=pair['w2'], inputs=inter, outputs=y, **bank.weight('w2'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)
    out = pf_torch.empty_like(x)
    pf__combine_kernel[t,](y, row_of, local, d['topk_weights'], d['_ones'], (1 + d['post_norm_w'].float()).bfloat16(), out, D=4096, num_warps=4)
    return out

def m128_prepare_data(d, w):
    return dict(d, _gate_gs=w['w13_gs_gate'], _up_gs=w['w13_gs_up'], _w2_gs=w['w2_gs'], _ones=torch.ones(32, device=w['w2_gs'].device), post_norm_w=w['post_norm_w'])

def m128_graph(fn, d):
    st = {k: (v.clone() if torch.is_tensor(v) else v) for k,v in d.items()}
    fn(st); pf_torch.cuda.synchronize()
    g = pf_torch.cuda.CUDAGraph()
    with pf_torch.cuda.graph(g): out = fn(st)
    return g, out, st

def m128_gate(g, d, want):
    for k in ('hidden_states','topk_ids','topk_weights'):
        g[2][k].copy_(d[k])
    pf_torch.cuda.synchronize(); g[0].replay(); pf_torch.cuda.synchronize(); got=g[1]
    diff=(got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12)
    return float(diff)

def m128_time(g, d, flush):
    for k in ('hidden_states','topk_ids','topk_weights'):
        g[2][k].copy_(d[k])
    vals=[]
    a=pf_torch.cuda.Event(enable_timing=True); b=pf_torch.cuda.Event(enable_timing=True)
    for j in range(5):
        flush.zero_(); a.record(); g[0].replay(); b.record(); b.synchronize(); vals.append(a.elapsed_time(b)*1000.0)
    return statistics.median(vals)

def lab(deadline, weights, state=None):
    lines=['META M128 grouped-masked Humming config sweep; exact R0 + 12 seeds; FP32 gate']
    dev='cuda'; flush=pf_torch.empty(1<<27,dtype=pf_torch.uint8,device=dev)
    specs=[('default',{}),('bm16',{'block':16}),('bm32',{'block':32}),('st2',{'stages':2}),('nostream',{'stream':False}),('sms66',{'sms':66}),('sms88',{'sms':88}),('sms132',{'sms':132}),('tma',{'tma':True})]
    w0=dc_weights(5353,dev); b0=m128_bank(w0); routes=128
    pairs={}
    for label,kw in specs:
        try:
            pairs[label]={n:m128_launch(b0,n,m128_cfg(b0,n,routes,**kw)) for n in ('w13','w2')}
        except Exception as e:
            lines.append(f'CONFIG {label} ERROR={type(e).__name__}:{str(e).splitlines()[0][:100]}')
    d0=m128_prepare_data(dc_fixtures(w0,128,r0=True)[0],w0); want0=dc_oracles([d0],w0)[0]
    scores=[]
    for label,pair in pairs.items():
        try:
            g=m128_graph(lambda z,p=pair:m128_call(z,b0,p),d0)
            err=m128_gate(g,d0,want0)
            if err>0.015:
                lines.append(f'R0 M=128 {label} gate={err:.5f} REJECT'); continue
            us=m128_time(g,d0,flush); scores.append((us,label)); lines.append(f'R0 M=128 {label} us={us:.1f} gate={err:.5f} PASS')
        except Exception as e: lines.append(f'R0 M=128 {label} ERROR={type(e).__name__}:{str(e).splitlines()[0][:100]}')
    scores.sort(); finalists=[x[1] for x in scores[:3]]
    lines.append('FINALISTS '+','.join(finalists))
    for i,seed in enumerate((5353+7919*j for j in range(12))):
        if pf_time.monotonic()>deadline-20: lines.append(f'INCOMPLETE seed={seed}'); break
        w=dc_weights(seed,dev); bank=m128_bank(w); ds=[m128_prepare_data(x,w) for x in dc_fixtures(w,128,r0=i==0)]; want=dc_oracles(ds,w)
        for label in finalists:
            try:
                pair={n:m128_launch(bank,n,m128_cfg(bank,n,128,**dict(next(kw for ll,kw in specs if ll==label)))) for n in ('w13','w2')}
                g=m128_graph(lambda z,p=pair:m128_call(z,bank,p),ds[0]); err=max(m128_gate(g,d,want[j]) for j,d in enumerate(ds))
                us=m128_time(g,ds[0],flush)
                lines.append(f'SEED seed={seed} {label} us={us:.1f} gate={err:.5f} '+('PASS' if err<=0.015 else 'REJECT'))
            except Exception as e: lines.append(f'SEED seed={seed} {label} ERROR={type(e).__name__}:{str(e).splitlines()[0][:100]}')
        pf_torch.cuda.empty_cache()
    lines.append(f'END remaining_s={deadline-pf_time.monotonic():.1f}')
    return lines
