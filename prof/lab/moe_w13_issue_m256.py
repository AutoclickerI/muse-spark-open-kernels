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
pf_N_EXPERTS = 256
pf_TOP_K = 8

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

@pf_triton.jit
def pf__fixed_offsets_kernel(Starts, T: pf_tl.constexpr):
    e = pf_tl.arange(0, 32)
    pf_tl.store(Starts + e, e * T)

def pf__experts_humming_masked(pooled, topk_w, slot_e, slot_pos, counts, st):
    T = pooled.shape[0]
    stride = pf_triton.cdiv(T, 128) * 128
    starts = pf_torch.empty(32, device=pooled.device, dtype=pf_torch.int32)
    pf__fixed_offsets_kernel[1,](starts, T=stride)
    a = pf_torch.empty((32 * stride, 4096), device=pooled.device, dtype=pf_torch.bfloat16)
    rows = pf_torch.empty((T, 8), device=pooled.device, dtype=pf_torch.int32)
    pf__gather_norm_kernel[T,](pooled, st['pre_expert'], slot_e, slot_pos, starts, a, rows, D=4096, num_warps=4)
    h = pf_torch.empty((32 * stride, 8192), device=pooled.device, dtype=pf_torch.bfloat16)
    inter = pf_torch.empty_like(a)
    y = pf_torch.empty_like(a)
    bank = st['humming']
    routes, gemms = bank.plan(T)
    pf_humming_ops.launch_kernel(configs=gemms['w13'][1], inputs=a, outputs=h, **bank.weight('w13'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)
    pf__portfolio_swiglu[32, pf_triton.cdiv(T, 4)](h, inter, starts, counts, st['gs13_gate'], st['gs13_up'], BR=4, D=4096, num_warps=8)
    pf_humming_ops.launch_kernel(configs=gemms['w2'][1], inputs=inter, outputs=y, **bank.weight('w2'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)
    out = pf_torch.empty_like(pooled)
    pf__combine_kernel[T,](y, rows, slot_e, topk_w, st['ones'], st['post_expert'], out, D=4096, num_warps=4)
    return out
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

@pf_triton.jit
def pf__portfolio_swiglu(H, A, STARTS, COUNTS, GATE_GS, UP_GS, D: pf_tl.constexpr, BR: pf_tl.constexpr):
    e = pf_tl.program_id(0)
    n = pf_tl.load(COUNTS + e)
    r = pf_tl.program_id(1) * BR + pf_tl.arange(0, BR)
    if pf_tl.program_id(1) * BR >= n:
        return
    row = pf_tl.load(STARTS + e) + r
    c = pf_tl.arange(0, D)
    g = pf_tl.load(H + row[:, None].to(pf_tl.int64) * (2 * D) + c[None, :], r[:, None] < n, 0).to(pf_tl.float32) * pf_tl.load(GATE_GS + e)
    u = pf_tl.load(H + row[:, None].to(pf_tl.int64) * (2 * D) + D + c[None, :], r[:, None] < n, 0).to(pf_tl.float32) * pf_tl.load(UP_GS + e)
    g = g.to(pf_tl.bfloat16).to(pf_tl.float32)
    u = u.to(pf_tl.bfloat16).to(pf_tl.float32)
    pf_tl.store(A + row[:, None].to(pf_tl.int64) * D + c[None, :], (g * pf_tl.sigmoid(g) * u).to(pf_tl.bfloat16), r[:, None] < n)

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

def pp_reference(data, st, name):
    x = data['hidden_states']
    m = x.shape[0]
    dev = x.device
    c = torch.empty(32, device=dev, dtype=torch.int32)
    slots = torch.empty((32, m), device=dev, dtype=torch.int32)
    pos = torch.empty((m, 8), device=dev, dtype=torch.int32)
    mw_route[32,](data['topk_ids'], c, slots, pos, m, 0, triton.next_power_of_2(m), num_warps=4)
    ids = torch.where((data['topk_ids'] >= 0) & (data['topk_ids'] < 32), data['topk_ids'], -1).int()
    args = (x, data['topk_weights'], ids, pos, c)
    return pf__experts_humming_masked(*args, st)
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

def wx_control_prepare(w):
    wx_need(6.0, 'humming compile')
    dev = w['w13'].device
    hum = pf__Experts(dict(w, w13_gs=torch.ones(32, device=dev)), 0, 32)
    hum.gemm_type = pf_GemmType.GROUPED_MASKED
    wx_need(6.0, 'humming compile256')
    hum.compile([256])
    for key in ('w13', 'w2'):
        assert hum.weight(key)['weight_scale'].dtype == torch.float8_e4m3fn
    return dict(pre_expert=torch.ones(4096, device=dev), ones=torch.ones(32, device=dev), post_expert=(1 + w['post_norm_w'].float()).bfloat16(), humming=hum, gs13_gate=w['w13_gs_gate'], gs13_up=w['w13_gs_up'], gs2=w['w2_gs'])
import functools
import tilelang
import tilelang.language as T
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
WX_DEADLINE = float('inf')

class WxBudget(Exception):
    pass

def wx_need(seconds, where):
    left = WX_DEADLINE - time.monotonic()
    if left < seconds:
        raise WxBudget(where + ' left=' + format(left, '.1f'))
import time
import statistics
import functools
import tilelang
import tilelang.language as T
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

def mw_repack_planes(q):
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
import functools, hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

def rc_masks():
    signs = [[0, 0] for _ in range(4)]
    zeros = [[0, 0] for _ in range(4)]
    used = [0, 0]
    slots = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
    for slot, (src, rot) in enumerate(slots):
        sf = (0, 1, 0, 1, 2, 3, 2, 3)[slot]
        for h in (0, 1):
            for bit, dbit in enumerate((6, 7, 8, 15)):
                p = (dbit + 16 * h + rot) % 32
                word = (0 if 1613783088 >> p & 1 else 1) if src == 'W' else 0 if src == 'A' else 1
                assert not used[word] & 1 << p
                used[word] |= 1 << p
                zeros[sf][word] |= 1 << p
                if bit == 3:
                    signs[sf][word] |= 1 << p
    assert used == [4294967295, 4294967295]
    return (tuple((tuple(x) for x in signs)), tuple((tuple(x) for x in zeros)))
RC_SIGNS, RC_ZEROS = rc_masks()

@triton.jit
def rc_repack(Q, S, QR, SR, N: tl.constexpr, SIGNS: tl.constexpr, ZEROS: tl.constexpr, B: tl.constexpr=256):
    i = tl.program_id(0) * B + tl.arange(0, B)
    ok = i < N
    l = i % 32
    sb = i // 32 * 32 + l // 4 * 4
    a = tl.load(Q + 2 * i, ok, 0).to(tl.uint32)
    b = tl.load(Q + 2 * i + 1, ok, 0).to(tl.uint32)
    for g in tl.static_range(4):
        sf = tl.load(S + sb + g, ok, 0).to(tl.uint32)
        mag = sf & 127
        neg = sf >= 128
        zero = mag == 0
        a = (a ^ tl.where(neg, SIGNS[g][0], 0).to(tl.uint32)) & tl.where(zero, 4294967295 ^ ZEROS[g][0], 4294967295).to(tl.uint32)
        b = (b ^ tl.where(neg, SIGNS[g][1], 0).to(tl.uint32)) & tl.where(zero, 4294967295 ^ ZEROS[g][1], 4294967295).to(tl.uint32)
        u = tl.where(mag >= 8, mag + 112, tl.where(mag == 1, 96, tl.where(mag == 2, 104, tl.where(mag == 3, 108, tl.where(mag == 4, 112, tl.where(mag == 5, 114, tl.where(mag == 6, 116, tl.where(mag == 7, 118, 0))))))))
        tl.store(SR + sb + g, u.to(tl.uint8), ok & (l % 4 == 0))
    tl.store(QR + 2 * i, a.to(tl.int32), ok)
    tl.store(QR + 2 * i + 1, b.to(tl.int32), ok)
import triton
'Packed WGMMA pipeline with a distributed vector W2 finalizer.'
import functools
import hashlib
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
FX_CUDA = '#include <cuda_runtime.h>\n#include <vector>\n#include <stdexcept>\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\n// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\n\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace mwfx {\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned sf_bits(unsigned raw){\n unsigned s=raw&127,bit=0;\n if(s>=8)bit=(238u<<7)+(s<<4);\n else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}\n if(s==127)bit=0x7fff;\n bit|=(raw&128)<<8;return bit|(bit<<16);\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\n}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){\n unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;\n unsigned s0=lut[raw&255],s1=lut[(raw>>8)&255],s2=lut[(raw>>16)&255],s3=lut[raw>>24];\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==32){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\n#pragma unroll\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n}\ntemplate<int NT,int KT>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n int t=threadIdx.x,w=t>>5,l=t&31;\n #pragma unroll\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n  async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if constexpr(S>3){\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n  else if constexpr(S>4){\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\n   else if constexpr(S>5){\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\n    else if constexpr(S>6){\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\n     else if constexpr(S>7){\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\n__device__ __forceinline__ int publish(int*p){\n int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;\n}\n__global__ void route_init(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,\n int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){\n int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;\n for(int i=all;i<clear_count;i+=step)state[i]=0;\n for(int i=all;i<M*4096/8;i+=step)reinterpret_cast<uint4*>(out)[i]=make_uint4(0,0,0,0);\n if(blockIdx.x)return;\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\n for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;\n  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}\n }\n for(int token=t;token<M;token+=256){int n=0;\n  #pragma unroll\n  for(int j=0;j<8;++j){int e=ids[token*8+j]-start;n+=(e>=0&&e<32);}\n  state[clear_count+token]=n;\n }\n __syncthreads();\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\n  int scan=pad;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\n  int key=n*32+31-t;\n  #pragma unroll\n  for(int k=2;k<=32;k*=2){\n   #pragma unroll\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\n   }\n  }\n  // Descending keys after reversing the ascending bitonic result.\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\n }\n}\ntemplate<int NT,int KT,int S,int SPLIT,bool FIRST,int MODE>\n__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,\n float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start,bool profiling){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(128) unsigned char rawsm[];\n unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);\n __shared__ unsigned lut[256];__shared__ int last,flags[NT];\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;\n int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;\n lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();\n int total=TASK[32]*MT*SPLIT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  uint64_t tick=profiling?clock64():0;uint64_t times[6]={};bool islast=false;\n  int task=unit/SPLIT,split=unit%SPLIT,group=task/MT,mt=task%MT,ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\n  int src[NT/16];\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  #pragma unroll\n  for(int i=0;i<min(S-1,NS);++i)\n   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\n  // K64 has one MMA group per stage. Alternate register-A banks across\n  // stages so wait1 retires a bank before its next conversion overwrites it.\n  // K128/K256 already alternate within each stage; their schedule is unchanged.\n  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");\n  #pragma unroll 1\n  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){\n   #pragma unroll\n   for(int ksub=0;ksub<(KT==64?2:1);++ksub){\n   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll 1\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\n    #pragma unroll\n    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;\n     #pragma unroll\n     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n      unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,lut,af[reg]+g*8);}\n     keep(af[reg]);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n     mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);keep(af[reg^1]);\n     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;\n      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);}\n    }\n   }\n   }\n  }\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  if(profiling){uint64_t now=clock64();times[0]=now-tick;tick=now;}\n  if constexpr(SPLIT>1){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];\n   }\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();\n   if(!last)continue;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;\n    if(r<count){\n     #pragma unroll\n     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];\n    }\n    acc[j]=sum;\n   }\n  }\n  if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n   }\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n   if constexpr(MODE!=1){\n   if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();\n   if(profiling){uint64_t now=clock64();times[2]=now-tick;tick=now;}\n   islast=last;\n   if(last){\n    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];\n     #pragma unroll\n     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);\n    }\n    __threadfence();__syncthreads();\n    if(profiling){uint64_t now=clock64();times[3]=now-tick;tick=now;}\n    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;\n      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}\n    __syncthreads();\n    if(profiling){uint64_t now=clock64();times[4]=now-tick;tick=now;}\n    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;\n     if constexpr(MODE==2){\n     int base[8];float alpha[8];\n     #pragma unroll\n     for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;base[j]=-1;alpha[j]=0;\n      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}\n     for(int c=l*4;c<4096;c+=128){float4 sum=make_float4(0,0,0,0);\n      #pragma unroll\n      for(int j=0;j<8;++j)if(base[j]>=0){float4 z=*(const float4*)(Y+(size_t)base[j]*4096+c);float a=alpha[j];sum.x+=z.x*a;sum.y+=z.y*a;sum.z+=z.z*a;sum.w+=z.w*a;}\n      __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n      *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n     }\n     }else{\n     for(int c=l;c<4096;c+=32){float sum=0;\n      #pragma unroll\n      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;\n       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}\n      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);\n     }\n     }\n    }\n   }\n   }else if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n  }\n  __syncthreads();\n  if(profiling&&t==0){uint64_t*p=(uint64_t*)PART+(size_t)unit*8;\n   for(int i=0;i<5;++i)p[i]=times[i];p[5]=clock64()-tick;p[6]=islast;p[7]=clock64();}\n }\n}\n#define FX_CASES(F) \\\n F(0,16,128,4,1) F(1,32,128,4,1)\n}\nint64_t fx_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n #define INIT(ID,NT,KT,S,SK) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto set=[&]<bool F,int MODE>(){WX_CUDA_CHECK(cudaFuncSetAttribute(mwfx::fused<NT,KT,S,SK,F,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<true,1>();set.template operator()<false,1>();if constexpr(ID==0)set.template operator()<false,0>();}\n FX_CASES(INIT)\n #undef INIT\n return opt;\n}\nstd::vector<int64_t> fx_info(int64_t which,int64_t first,int64_t mode){\n #define INFO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F,int MODE>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwfx::fused<NT,KT,S,SK,F,MODE>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwfx::fused<NT,KT,S,SK,F,MODE>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(first)return info.template operator()<true,1>();if(mode==1)return info.template operator()<false,1>();if constexpr(ID==0){WX_CHECK(mode==0);return info.template operator()<false,0>();}else WX_CHECK(false,"legacy control exists only forNT16");}\n switch(which){FX_CASES(INFO) default:WX_CHECK(false,"unknown fixed configuration");}\n #undef INFO\n}\nvoid fx_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear_count,int64_t stream){\n mwfx::route_init<<<128,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear_count);WX_CUDA_CHECK(cudaGetLastError());\n}\nvoid fx_run(std::vector<int64_t> v,int64_t first,int64_t which,int64_t grid,int64_t start,int64_t mode,int64_t profiling,int64_t M,int64_t cap,int64_t stream_value){\n WX_CHECK(v.size()==20);auto stream=(cudaStream_t)stream_value;\n #define GO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2; \\\n auto launch=[&]<bool FIRST,int MODE>(){mwfx::fused<NT,KT,S,SK,FIRST,MODE><<<grid,128,bytes,stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(const __nv_bfloat16*)(void*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(const __nv_bfloat16*)(void*)v[14],(float*)v[10],(__nv_bfloat16*)(void*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)(void*)v[19],M,cap,start,profiling);};if(first)launch.template operator()<true,1>();else if(mode==1)launch.template operator()<false,1>();else{WX_CHECK(mode==0&&which==0);if constexpr(ID==0)launch.template operator()<false,0>();}break;}\n switch(which){FX_CASES(GO) default:WX_CHECK(false,"unknown fixed configuration");}\n #undef GO\n WX_CUDA_CHECK(cudaGetLastError());\n}\nnamespace mwfx {\n__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\n int tok=blockIdx.x,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\n __syncthreads();int pp[8];float aa[8];\n #pragma unroll\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\n for(int c=t*4;c<4096;c+=1024){float4 sum=make_float4(0,0,0,0);\n  #pragma unroll\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n }\n}\n}\nvoid fx_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){mwfx::finish<<<M,256,0,(cudaStream_t)stream>>>((int*)v[5],(int*)v[11],(float*)v[12],(int*)v[13],(float*)v[15],(float*)v[16],(__nv_bfloat16*)(void*)v[19],M,start);WX_CUDA_CHECK(cudaGetLastError());}\n#include <cuda_runtime.h>\n#include <vector>\n#include <stdexcept>\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\n// Two lossless banks: raw E4M3 and exponent/mantissa bytes with sign/zero folded into FP4.\n// Both convert to BF16 only inside the timed kernel.\n\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace mwwi {\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned sf_bits(unsigned raw){\n unsigned s=raw&127,bit=0;\n if(s>=8)bit=(238u<<7)+(s<<4);\n else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}\n if(s==127)bit=0x7fff;\n bit|=(raw&128)<<8;return bit|(bit<<16);\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\n}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq_scaled(unsigned a,unsigned b,unsigned s0,unsigned s1,unsigned s2,unsigned s3,unsigned*out){\n unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\ntemplate<bool RECODED>__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){\n unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;\n unsigned s0,s1,s2,s3;\n if constexpr(RECODED){s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u;s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u;s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;}\n else{s0=lut[raw&255];s1=lut[(raw>>8)&255];s2=lut[(raw>>16)&255];s3=lut[raw>>24];}\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==32){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\n#pragma unroll\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n}\ntemplate<int NT,int KT,bool VEC>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n int t=threadIdx.x,w=t>>5,l=t&31;\n if constexpr(VEC){\n  // Q: two warps issue16B, replacing four warps\'8B copies. Keep .ca policy.\n  if(t<64){unsigned group=t>>4,lane=t&15;\n   unsigned base=((unsigned)e*(N/16)+mt*4+group)*128*64+tile*(KT/32)*64+lane*4;\n   #pragma unroll\n   for(int g=0;g<KT/32;++g){unsigned dst=smaddr(q+g*1024+t*16);const void*src=Q+base+g*64;\n    asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(src):"memory");}\n  }\n  // SF: one warp issues all512B once, replacing16 quarter-warp scalar copies.\n  if(t>=64&&t<64+KT/4){unsigned i=t-64,g=i/8,group=(i%8)/2,half=i%2;\n   unsigned srcidx=(((unsigned)e*(N/16)+mt*4+group)*128+tile*(KT/32)+g)*32+half*16;\n   unsigned dst=smaddr(s+g*128+group*32+half*16);const void*src=S+srcidx;\n   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(src):"memory");\n  }\n }else{\n #pragma unroll\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n  async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);\n }\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if constexpr(S>3){\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n  else if constexpr(S>4){\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\n   else if constexpr(S>5){\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\n    else if constexpr(S>6){\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\n     else if constexpr(S>7){\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\n__device__ __forceinline__ int publish(int*p){\n int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;\n}\ntemplate<int NT,int KT,int S,int SPLIT,bool FIRST,int MODE,bool RECODED,bool VEC,int FILTER=0,bool QUAD=false>\n__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,\n float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start,bool profiling){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(1024) unsigned char typed_sm[];\n unsigned char*sm=typed_sm;\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);\n __shared__ unsigned lut[RECODED?1:256];__shared__ int last,flags[NT];\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;\n int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;\n if constexpr(!RECODED){lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();}\n int total=TASK[32]*MT*SPLIT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  uint64_t tick=profiling?clock64():0;uint64_t times[6]={};bool islast=false;\n  int task=unit/SPLIT,split=unit%SPLIT,group=task/MT,mt=task%MT,ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\n  if constexpr(FILTER==1){if(count>16)continue;}\n  if constexpr(FILTER==2){if(count<=16)continue;}\n  int src[NT/16];\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  #pragma unroll\n  for(int i=0;i<min(S-1,NS);++i)\n   load_stage<NT,KT,VEC>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\n  // K64 has one MMA group per stage. Alternate register-A banks across\n  // stages so wait1 retires a bank before its next conversion overwrites it.\n  // K128/K256 already alternate within each stage; their schedule is unchanged.\n  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");\n  #pragma unroll 1\n  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){\n   #pragma unroll\n   for(int ksub=0;ksub<(KT==64?2:1);++ksub){\n   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll 1\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\n    #pragma unroll\n    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;\n     #pragma unroll\n     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n      if constexpr(QUAD){unsigned sc=*(const unsigned char*)(ss+st*SB+z*128+t);unsigned sbits;\n       if constexpr(RECODED)sbits=sc*0x00100010u+0x70007000u;else sbits=lut[sc];\n       dq_scaled(q.x,q.y,__shfl_sync(~0u,sbits,0,4),__shfl_sync(~0u,sbits,1,4),__shfl_sync(~0u,sbits,2,4),__shfl_sync(~0u,sbits,3,4),af[reg]+g*8);\n      }else{unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq<RECODED>(q.x,q.y,sf,lut,af[reg]+g*8);}}\n     keep(af[reg]);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n     mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);keep(af[reg^1]);\n     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;\n      load_stage<NT,KT,VEC>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);}\n    }\n   }\n   }\n  }\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  if(profiling){uint64_t now=clock64();times[0]=now-tick;tick=now;}\n  if constexpr(SPLIT>1){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];\n   }\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();\n   if(!last)continue;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;\n    if(r<count){\n     #pragma unroll\n     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];\n    }\n    acc[j]=sum;\n   }\n  }\n  if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n   }\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n   if constexpr(MODE!=1){\n   if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();\n   if(profiling){uint64_t now=clock64();times[2]=now-tick;tick=now;}\n   islast=last;\n   if(last){\n    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];\n     #pragma unroll\n     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);\n    }\n    __threadfence();__syncthreads();\n    if(profiling){uint64_t now=clock64();times[3]=now-tick;tick=now;}\n    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;\n      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}\n    __syncthreads();\n    if(profiling){uint64_t now=clock64();times[4]=now-tick;tick=now;}\n    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;\n     if constexpr(MODE==2){\n     int base[8];float alpha[8];\n     #pragma unroll\n     for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;base[j]=-1;alpha[j]=0;\n      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}\n     for(int c=l*4;c<4096;c+=128){float4 sum=make_float4(0,0,0,0);\n      #pragma unroll\n      for(int j=0;j<8;++j)if(base[j]>=0){float4 z=*(const float4*)(Y+(size_t)base[j]*4096+c);float a=alpha[j];sum.x+=z.x*a;sum.y+=z.y*a;sum.z+=z.z*a;sum.w+=z.w*a;}\n      __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n      *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n     }\n     }else{\n     for(int c=l;c<4096;c+=32){float sum=0;\n      #pragma unroll\n      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;\n       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}\n      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);\n     }\n     }\n    }\n   }\n   }else if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n  }\n  __syncthreads();\n  if(profiling&&t==0){uint64_t*p=(uint64_t*)PART+(size_t)unit*8;\n   for(int i=0;i<5;++i)p[i]=times[i];p[5]=clock64()-tick;p[6]=islast;p[7]=clock64();}\n }\n}\n\n}\n\nint64_t wi_init(){\n constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*32*128*2;\n auto set=[&]<bool R,bool V>(){WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,R,V>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<false,false>();set.template operator()<false,true>();set.template operator()<true,true>();\n WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,false,true,0,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,true,true,0,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,true,true,2,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));\n constexpr int cold_bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*16*128*2;WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<16,128,4,1,true,1,true,true,1>,cudaFuncAttributeMaxDynamicSharedMemorySize,cold_bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<16,128,4,1,true,1,true,true,1,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,cold_bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,true,true,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));return bytes;\n}\nstd::vector<int64_t> wi_info(int64_t policy){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*32*128*2;\n auto info=[&]<bool R,bool V>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwwi::fused<32,128,4,1,true,1,R,V>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwwi::fused<32,128,4,1,true,1,R,V>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(policy==0)return info.template operator()<false,false>();if(policy==1)return info.template operator()<false,true>();WX_CHECK(policy==2);return info.template operator()<true,true>();}\nvoid wi_run(std::vector<int64_t>v,int64_t policy,int64_t grid,int64_t M,int64_t cap,int64_t start,int64_t stream){\n constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*32*128*2;\n auto launch=[&]<bool R,bool V>(){mwwi::fused<32,128,4,1,true,1,R,V><<<grid,128,bytes,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(float*)v[10],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)v[19],M,cap,start,false);};if(policy==0)launch.template operator()<false,false>();else if(policy==1)launch.template operator()<false,true>();else{WX_CHECK(policy==2);launch.template operator()<true,true>();}WX_CUDA_CHECK(cudaGetLastError());}\n\nvoid wi_mixed(std::vector<int64_t>v,int64_t hot,int64_t grid,int64_t M,int64_t cap,int64_t start,int64_t stream){\n auto launch=[&]<int NT,int FILTER>(){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*NT*128*2;mwwi::fused<NT,128,4,1,true,1,true,true,FILTER><<<grid,128,bytes,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(float*)v[10],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)v[19],M,cap,start,false);};if(hot)launch.template operator()<32,2>();else launch.template operator()<16,1>();WX_CUDA_CHECK(cudaGetLastError());}\n\nstd::vector<int64_t> wi_mixed_info(int64_t hot){auto info=[&]<int NT,int FILTER>(){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*NT*128*2;cudaFuncAttributes a{};int n;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwwi::fused<NT,128,4,1,true,1,true,true,FILTER>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwwi::fused<NT,128,4,1,true,1,true,true,FILTER>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,n};};return hot?info.template operator()<32,2>():info.template operator()<16,1>();}\n\nvoid wi_quad(std::vector<int64_t>v,int64_t kind,int64_t grid,int64_t M,int64_t cap,int64_t start,int64_t stream){\n auto launch=[&]<int NT,bool R,int F>(){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*NT*128*2;mwwi::fused<NT,128,4,1,true,1,R,true,F,true><<<grid,128,bytes,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(float*)v[10],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)v[19],M,cap,start,false);};if(kind==0)launch.template operator()<32,false,0>();else if(kind==1)launch.template operator()<32,true,0>();else if(kind==2)launch.template operator()<16,true,1>();else{WX_CHECK(kind==3);launch.template operator()<32,true,2>();}WX_CUDA_CHECK(cudaGetLastError());}\n#include <cuda_runtime.h>\n#include <vector>\n#include <stdexcept>\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\n// Two lossless banks: raw E4M3 and exponent/mantissa bytes with sign/zero folded into FP4.\n// Both convert to BF16 only inside the timed kernel.\n\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace mwrc {\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned sf_bits(unsigned raw){\n unsigned s=raw&127,bit=0;\n if(s>=8)bit=(238u<<7)+(s<<4);\n else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}\n if(s==127)bit=0x7fff;\n bit|=(raw&128)<<8;return bit|(bit<<16);\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\n}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\ntemplate<bool RECODED>__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){\n unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;\n unsigned s0,s1,s2,s3;\n if constexpr(RECODED){s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u;s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u;s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;}\n else{s0=lut[raw&255];s1=lut[(raw>>8)&255];s2=lut[(raw>>16)&255];s3=lut[raw>>24];}\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==32){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\n#pragma unroll\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n}\ntemplate<int NT,int KT>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n int t=threadIdx.x,w=t>>5,l=t&31;\n #pragma unroll\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n  async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if constexpr(S>3){\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n  else if constexpr(S>4){\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\n   else if constexpr(S>5){\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\n    else if constexpr(S>6){\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\n     else if constexpr(S>7){\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\n__device__ __forceinline__ int publish(int*p){\n int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;\n}\n__global__ void route_init(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,\n int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){\n int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;\n for(int i=all;i<clear_count;i+=step)state[i]=0;\n for(int i=all;i<M*4096/8;i+=step)reinterpret_cast<uint4*>(out)[i]=make_uint4(0,0,0,0);\n if(blockIdx.x)return;\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\n for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;\n  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}\n }\n for(int token=t;token<M;token+=256){int n=0;\n  #pragma unroll\n  for(int j=0;j<8;++j){int e=ids[token*8+j]-start;n+=(e>=0&&e<32);}\n  state[clear_count+token]=n;\n }\n __syncthreads();\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\n  int scan=pad;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\n  int key=n*32+31-t;\n  #pragma unroll\n  for(int k=2;k<=32;k*=2){\n   #pragma unroll\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\n   }\n  }\n  // Descending keys after reversing the ascending bitonic result.\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\n }\n}\ntemplate<int NT,int KT,int S,int SPLIT,bool FIRST,int MODE,bool RECODED>\n__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,\n float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start,bool profiling){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(128) unsigned char rawsm[];\n unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);\n __shared__ unsigned lut[RECODED?1:256];__shared__ int last,flags[NT];\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;\n int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;\n if constexpr(!RECODED){lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();}\n int total=TASK[32]*MT*SPLIT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  uint64_t tick=profiling?clock64():0;uint64_t times[6]={};bool islast=false;\n  int task=unit/SPLIT,split=unit%SPLIT,group=task/MT,mt=task%MT,ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\n  int src[NT/16];\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  #pragma unroll\n  for(int i=0;i<min(S-1,NS);++i)\n   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\n  // K64 has one MMA group per stage. Alternate register-A banks across\n  // stages so wait1 retires a bank before its next conversion overwrites it.\n  // K128/K256 already alternate within each stage; their schedule is unchanged.\n  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");\n  #pragma unroll 1\n  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){\n   #pragma unroll\n   for(int ksub=0;ksub<(KT==64?2:1);++ksub){\n   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll 1\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\n    #pragma unroll\n    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;\n     #pragma unroll\n     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n      unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq<RECODED>(q.x,q.y,sf,lut,af[reg]+g*8);}\n     keep(af[reg]);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n     mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);keep(af[reg^1]);\n     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;\n      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);}\n    }\n   }\n   }\n  }\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  if(profiling){uint64_t now=clock64();times[0]=now-tick;tick=now;}\n  if constexpr(SPLIT>1){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];\n   }\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();\n   if(!last)continue;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;\n    if(r<count){\n     #pragma unroll\n     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];\n    }\n    acc[j]=sum;\n   }\n  }\n  if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n   }\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n   if constexpr(MODE!=1){\n   if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();\n   if(profiling){uint64_t now=clock64();times[2]=now-tick;tick=now;}\n   islast=last;\n   if(last){\n    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];\n     #pragma unroll\n     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);\n    }\n    __threadfence();__syncthreads();\n    if(profiling){uint64_t now=clock64();times[3]=now-tick;tick=now;}\n    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;\n      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}\n    __syncthreads();\n    if(profiling){uint64_t now=clock64();times[4]=now-tick;tick=now;}\n    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;\n     if constexpr(MODE==2){\n     int base[8];float alpha[8];\n     #pragma unroll\n     for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;base[j]=-1;alpha[j]=0;\n      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}\n     for(int c=l*4;c<4096;c+=128){float4 sum=make_float4(0,0,0,0);\n      #pragma unroll\n      for(int j=0;j<8;++j)if(base[j]>=0){float4 z=*(const float4*)(Y+(size_t)base[j]*4096+c);float a=alpha[j];sum.x+=z.x*a;sum.y+=z.y*a;sum.z+=z.z*a;sum.w+=z.w*a;}\n      __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n      *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n     }\n     }else{\n     for(int c=l;c<4096;c+=32){float sum=0;\n      #pragma unroll\n      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;\n       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}\n      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);\n     }\n     }\n    }\n   }\n   }else if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n  }\n  __syncthreads();\n  if(profiling&&t==0){uint64_t*p=(uint64_t*)PART+(size_t)unit*8;\n   for(int i=0;i<5;++i)p[i]=times[i];p[5]=clock64()-tick;p[6]=islast;p[7]=clock64();}\n }\n}\n#define RC_CASES(F) \\\n F(0,16,128,4,1) F(1,32,128,4,1)\n}\nint64_t rc_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n #define INIT(ID,NT,KT,S,SK) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto set=[&]<bool F,bool R>(){WX_CUDA_CHECK(cudaFuncSetAttribute(mwrc::fused<NT,KT,S,SK,F,1,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<true,false>();set.template operator()<false,false>();set.template operator()<true,true>();set.template operator()<false,true>();}\n RC_CASES(INIT)\n #undef INIT\n return opt;\n}\nstd::vector<int64_t> rc_info(int64_t which,int64_t first,int64_t encoded){\n #define INFO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F,bool R>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwrc::fused<NT,KT,S,SK,F,1,R>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwrc::fused<NT,KT,S,SK,F,1,R>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(first)return encoded?info.template operator()<true,true>():info.template operator()<true,false>();return encoded?info.template operator()<false,true>():info.template operator()<false,false>();}\n switch(which){RC_CASES(INFO) default:WX_CHECK(false,"unknown recoded configuration");}\n #undef INFO\n}\nvoid rc_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear_count,int64_t stream){\n mwrc::route_init<<<128,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear_count);WX_CUDA_CHECK(cudaGetLastError());\n}\nvoid rc_run(std::vector<int64_t> v,int64_t first,int64_t which,int64_t grid,int64_t start,int64_t encoded,int64_t profiling,int64_t M,int64_t cap,int64_t stream_value){\n WX_CHECK(v.size()==20);auto stream=(cudaStream_t)stream_value;\n #define GO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2; \\\n auto launch=[&]<bool FIRST,bool RECODED>(){mwrc::fused<NT,KT,S,SK,FIRST,1,RECODED><<<grid,128,bytes,stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(const __nv_bfloat16*)(void*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(const __nv_bfloat16*)(void*)v[14],(float*)v[10],(__nv_bfloat16*)(void*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)(void*)v[19],M,cap,start,profiling);};if(first){if(encoded)launch.template operator()<true,true>();else launch.template operator()<true,false>();}else{if(encoded)launch.template operator()<false,true>();else launch.template operator()<false,false>();}break;}\n switch(which){RC_CASES(GO) default:WX_CHECK(false,"unknown fixed configuration");}\n #undef GO\n WX_CUDA_CHECK(cudaGetLastError());\n}\nnamespace mwrc {\n__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\n int tok=blockIdx.x,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\n __syncthreads();int pp[8];float aa[8];\n #pragma unroll\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\n for(int c=t*4;c<4096;c+=1024){float4 sum=make_float4(0,0,0,0);\n  #pragma unroll\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n }\n}\n}\nvoid rc_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){mwrc::finish<<<M,256,0,(cudaStream_t)stream>>>((int*)v[5],(int*)v[11],(float*)v[12],(int*)v[13],(float*)v[15],(float*)v[16],(__nv_bfloat16*)(void*)v[19],M,start);WX_CUDA_CHECK(cudaGetLastError());}\n'
FX_CPP = '#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\nint64_t fx_init();std::vector<int64_t> fx_info(int64_t,int64_t,int64_t);void fx_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void fx_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void fx_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);'
FX_CONFIGS = ((16, 128, 4, 1), (32, 128, 4, 1))

@functools.lru_cache(None)
def fx_module():
    wx_need(40.0, 'native compile')
    target = '-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability() == (9, 0) else '-gencode=arch=compute_120,code=sm_120'
    m = load_inline(name='w13_issue_' + hashlib.sha256(FX_CUDA.encode()).hexdigest()[:12], cpp_sources=FX_CPP + WI_CPP, cuda_sources=FX_CUDA, functions=['fx_init', 'fx_info', 'fx_route', 'fx_run', 'fx_finish', 'wi_init', 'wi_info', 'wi_run', 'rc_init', 'rc_info', 'rc_run', 'wi_mixed', 'wi_mixed_info', 'wi_quad'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', target], verbose=False)
    m.fx_init()
    m.wi_init()
    m.rc_init()
    return m

def fx_prepare(w):
    bank = {}
    for key in ('w13', 'w2'):
        sf = w[key + '_sf'].view(torch.uint8)
        e, n, k16 = sf.shape
        bank[key] = mw_repack_planes(w[key])
        bank[key + '_sf'] = sf.reshape(e, n // 16, 2, 8, k16 // 2, 2).permute(0, 1, 4, 3, 5, 2).contiguous()
    bank['w13_gs'] = torch.cat((w.get('w13_gs_gate', w['w13_gs']), w.get('w13_gs_up', w['w13_gs']))).float().contiguous()
    bank['w2_gs'] = w['w2_gs'].float().contiguous()
    bank['pnw'] = w.get('post_norm_w', w.get('post_expert_norm')).contiguous()
    return fx_from_packed(bank)

def fx_from_packed(bank):
    ng = bank['w13'].shape[1]
    rr = torch.arange(ng // 2, device=bank['w13'].device).reshape(-1, 2)
    order = torch.cat((rr, rr + ng // 2), 1).flatten()
    return dict(bank, w13=bank['w13'].index_select(1, order).contiguous(), w13_sf=bank['w13_sf'].index_select(1, order).contiguous(), mod=fx_module())

def fx_plan(data, bank, which=0, grid=528, mode=1):
    nt, kt, st, sk = FX_CONFIGS[which]
    x = data['hidden_states']
    m = x.shape[0]
    dev = x.device
    cap = m * 8 + 32 * nt
    gc = cap // nt
    c = torch.empty(32, device=dev, dtype=torch.int32)
    off = torch.empty(33, device=dev, dtype=torch.int32)
    order = torch.empty_like(c)
    task = torch.empty_like(off)
    slots = torch.empty((32, m), device=dev, dtype=torch.int32)
    ids = data['topk_ids'].to(torch.int32).contiguous()
    tw = data['topk_weights'].float().contiguous()
    pos = torch.empty_like(ids)
    clear = gc * 193 + m
    state = torch.empty(clear + m, device=dev, dtype=torch.int32)
    act = torch.empty((cap, 4096), device=dev, dtype=torch.bfloat16)
    p13 = torch.empty((sk, cap, 8192) if sk > 1 else (1,), device=dev, dtype=torch.float32)
    p2 = torch.empty((cap // nt * 64 * 8 * 2,), device=dev, dtype=torch.float32)
    y = torch.empty((cap, 4096), device=dev, dtype=torch.float32)
    ss = torch.empty((cap, 64), device=dev, dtype=torch.float32)
    inv = torch.empty(cap, device=dev, dtype=torch.float32)
    out = torch.empty_like(x)
    common = [c, off, order, task, slots, act]
    tail = [ids, tw, pos, bank['pnw'], y, ss, inv, state, out]
    v13 = [bank['w13'], bank['w13_sf'], bank['w13_gs'], x] + common + [p13] + tail
    v2 = [bank['w2'], bank['w2_sf'], bank['w2_gs'], act] + common + [p2] + tail
    start = data['config'].get('local_start', 0)
    vp13 = [v.data_ptr() for v in v13]
    vp2 = [v.data_ptr() for v in v2]
    rp = [v.data_ptr() for v in (ids, c, off, order, task, slots, pos, state, out)]
    owners = (v13, v2, ids, tw)

    def stream():
        return torch.cuda.current_stream().cuda_stream

    def route():
        _ = owners
        bank['mod'].fx_route(rp, m, nt, start, clear, stream())

    def first():
        _ = owners
        bank['mod'].fx_run(vp13, 1, which, grid, start, 1, 0, m, cap, stream())

    def second():
        _ = owners
        bank['mod'].fx_run(vp2, 0, which, grid, start, mode, 0, m, cap, stream())
        return out

    def finish():
        _ = owners
        bank['mod'].fx_finish(vp2, m, start, stream())
        return out
    steps = (route, first, second, finish) if mode == 1 else (route, first, second)
    return dict(steps=steps, v2=v2, mode=mode, which=which, grid=grid, start=start, profile=p2, pointers=vp2, M=m, cap=cap, v13=v13)

def fx_work(data, bank, which=0, grid=528, mode=1):
    return fx_plan(data, bank, which, grid, mode)['steps']

def fx_call(data, bank, which=0, grid=528, mode=1):
    for f in fx_work(data, bank, which, grid, mode):
        out = f()
    return out
WI_CPP = 'int64_t wi_init();std::vector<int64_t> wi_info(int64_t);void wi_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);int64_t rc_init();std::vector<int64_t> rc_info(int64_t,int64_t,int64_t);void rc_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void wi_mixed(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> wi_mixed_info(int64_t);void wi_quad(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'

def wi_prepare(weights):
    bank = fx_prepare(weights)
    for name in ('w13', 'w2'):
        q = bank[name]
        sf = bank[name + '_sf']
        qr = torch.empty_like(q)
        sr = torch.empty_like(sf)
        rc_repack[triton.cdiv(q.numel() // 2, 256),](q, sf, qr, sr, q.numel() // 2, RC_SIGNS, RC_ZEROS)
        bank[name + '_rec'] = qr
        bank[name + '_rec_sf'] = sr
    bank['mix_stream'] = torch.cuda.Stream(device=bank['w13'].device)
    bank['mix_start'] = torch.cuda.Event()
    bank['mix_done'] = torch.cuda.Event()
    return bank

def wi_plan(data, bank, policy=-1, grid=2112):
    plan = fx_plan(data, bank, 1, grid, 1)
    if policy == -1:
        return plan
    v = plan['v13'].copy()
    v2 = plan['v2'].copy()
    if policy in (2, 3, 4, 5, 6, 8, 9):
        v[:2] = [bank['w13_rec'], bank['w13_rec_sf']]
    if policy in (3, 4, 5, 6, 8, 9):
        v2[:2] = [bank['w2_rec'], bank['w2_rec_sf']]
    ptrs = [x.data_ptr() for x in v]
    p2 = [x.data_ptr() for x in v2]
    m = plan['M']
    cap = plan['cap']
    start = plan['start']

    def first():
        _ = v
        if policy in (7, 8):
            bank['mod'].wi_quad(ptrs, 0 if policy == 7 else 1, grid, m, cap, start, torch.cuda.current_stream().cuda_stream)
        elif policy in (5, 6, 9):
            parent = torch.cuda.current_stream()
            if policy == 5:
                bank['mod'].wi_mixed(ptrs, 0, grid, m, cap, start, parent.cuda_stream)
                bank['mod'].wi_mixed(ptrs, 1, grid, m, cap, start, parent.cuda_stream)
            else:
                bank['mix_start'].record(parent)
                bank['mix_stream'].wait_event(bank['mix_start'])
                with torch.cuda.stream(bank['mix_stream']):
                    if policy == 9:
                        bank['mod'].wi_quad(ptrs, 2, grid, m, cap, start, bank['mix_stream'].cuda_stream)
                    else:
                        bank['mod'].wi_mixed(ptrs, 0, grid, m, cap, start, bank['mix_stream'].cuda_stream)
                    bank['mix_done'].record()
                if policy == 9:
                    bank['mod'].wi_quad(ptrs, 3, grid, m, cap, start, parent.cuda_stream)
                else:
                    bank['mod'].wi_mixed(ptrs, 1, grid, m, cap, start, parent.cuda_stream)
                parent.wait_event(bank['mix_done'])
        elif policy == 3:
            bank['mod'].rc_run(ptrs, 1, 1, grid, start, 1, 0, m, cap, torch.cuda.current_stream().cuda_stream)
        else:
            bank['mod'].wi_run(ptrs, 2 if policy == 4 else policy, grid, m, cap, start, torch.cuda.current_stream().cuda_stream)

    def second():
        _ = v2
        bank['mod'].rc_run(p2, 0, 1, grid, start, 1, 0, m, cap, torch.cuda.current_stream().cuda_stream)
    steps = plan['steps']
    plan['steps'] = (steps[0], first, second if policy in (3, 4, 5, 6, 8, 9) else steps[2], steps[3])
    return plan

def wi_call(data, bank, policy=-1, grid=2112):
    for f in wi_plan(data, bank, policy, grid)['steps']:
        out = f()
    return out
import time
import statistics

WI_NAMES=('CTA1g2112','masked','typed_LOP','vector_LOP','vector_LOP_recW13','C1g2112','vector_LOP_fullrec','mixed_seq','mixed_parallel','quad_raw','quad_fullrec','mixed_parallel_quad')
WI_POLICIES=(-1,0,1,2,3,4,5,6,7,8,9)


def issue_measure(w,bank,ctrl,ds,flush):
    funcs=[lambda d:wi_call(d,bank,-1),lambda d:pp_reference(d,ctrl,'masked')]
    funcs.extend(lambda d,p=p:wi_call(d,bank,p) for p in (0,1,2,3,4,5,6,7,8,9));wx_need(4.,'FP32 oracle');wants=dc_oracles(ds,w);graphs=[];errors=[]
    for f in funcs:
        wx_need(3.,'graph/JIT');g=MWGraph(f,ds[0]);e=0.;r=0.
        for d,want in zip(ds,wants):
            wx_need(.5,'fresh gate');g.refresh(d);got=g.replay();torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()):e=r=float('inf');continue
            diff=got.float()-want.float();den=want.float().norm(dim=-1);e=max(e,float(diff.norm()/want.float().norm().clamp_min(1e-12)));r=max(r,float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max()))
        graphs.append(g);errors.append((e,r))
    if any(e>.01 or r>.05 for e,r in errors[:2]):raise RuntimeError('W13 control gate '+str(errors[:2]))
    ts=[[] for g in graphs];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for rep in range(8):
        wx_need(.5,'timing repeat')
        for g in graphs:g.refresh(ds[rep%3])
        order=list(range(len(graphs)));order=order[rep%len(order):]+order[:rep%len(order)]
        if rep%2:order.reverse()
        for j in order:
            if errors[j][0]>.01 or errors[j][1]>.05:continue
            wx_need(.2,'graph replay');flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();ts[j].append(a.elapsed_time(b)*1000.)
    return [(statistics.median(t) if t else float('inf'),e,r) for t,(e,r) in zip(ts,errors)]


def issue_phases(w,bank,flush):
    wx_need(4.,'R0 oracle');d=dc_fixtures(w,256,True)[0];want=dc_oracles([d],w)[0];ids=d['topk_ids'];cnt=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32);groups=int(((cnt+31)//32).sum());unique=int((cnt>0).sum());lines=[f'R0 P={int(cnt.sum())} E={unique} hot={int(cnt.max())} groups32={groups} fill={int(cnt.sum())/(groups*32):.3f} qsf_reqMB={groups*128*147456/1e6:.1f} uniqueMB={unique*8192*4096*9/16/1e6:.1f}']
    for policy in (-1,1,3,4,6,7,8,9):
        wx_need(3.,'phase capture');plan=wi_plan(d,bank,policy);steps=plan['steps']
        for f in steps:got=f()
        torch.cuda.synchronize();diff=got.float()-want.float();e=float(diff.norm()/want.float().norm());den=want.float().norm(dim=-1);row=float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())
        if not bool(torch.isfinite(got).all()) or e>.01 or row>.05:raise RuntimeError(f'R0 policy={policy} gate {e} {row}')
        vals=[]
        for phase,f in enumerate(steps):
            wx_need(1.,'phase graph');g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):f()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);ts=[]
            for rep in range(4):
                for pre in steps[:phase]:pre()
                wx_need(.2,'phase repeat');flush.zero_();a.record();g.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000.)
            vals.append(statistics.median(ts))
        lines.append(f'PH p={policy} route,w13,w2,finish='+','.join(f'{v:.1f}' for v in vals)+f' W13reqTBs={groups*128*147456/vals[1]/1e6:.2f} uniqueTBs={unique*8192*4096*9/16/vals[1]/1e6:.2f} e={e:.5f}')
    return lines


def lab(deadline,weights,state=None):
    global WX_DEADLINE
    WX_DEADLINE=deadline;start=time.monotonic();records=[];peak=0.
    lines=['META M256 W13-only edits; p3-6/8/9 use same recoded W2,others raw W2. K128/S4/grid2112 fixed; p5/6 split <=16-row experts N16,others N32; seq/parallel. Typed shared align1024; Q16B/SF16B vector copies; immediate LOP3; p7+ quad scale broadcasts; recW13 uses exact scale recode. 12same public weight/router seeds x3fresh inputs x8rotated cold128MiB graphs. Gates finite/L2<=.01,row<=.05; partial n not a win.']
    lines.append('ORDER '+','.join(WI_NAMES))
    try:
        wx_need(40.,'before native compile');t=time.monotonic();mod=fx_module();lines.append(f'BUILD s={time.monotonic()-t:.1f} left={deadline-time.monotonic():.1f}')
        lines.append('RECBASE W13/W2='+str(mod.rc_info(1,1,1))+str(mod.rc_info(1,0,1)))
        lines.append('BASE W13regs,local,static,dynamic,res='+str(mod.fx_info(1,1,1)))
        lines.append('MIXRES cold/hot='+str(mod.wi_mixed_info(0))+str(mod.wi_mixed_info(1)))
        for p in (0,1,2):lines.append(f'RES p={p} regs,local,static,dynamic,res='+str(mod.wi_info(p)))
        dev=next(v.device for v in weights.values() if torch.is_tensor(v));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
        for i in range(12):
            wx_need(max(8. if i==0 else 4.,peak*1.5+1.),'before seed'+str(i));t=time.monotonic();w=dc_weights(5353+7919*i,dev);ctrl=wx_control_prepare(w);wx_need(4.,'bank repack/JIT');bank=wi_prepare(w);ds=dc_fixtures(w,256);vals=issue_measure(w,bank,ctrl,ds,flush);records.append(vals)
            ids=ds[0]['topk_ids'];cnt=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32);lines.append(f'S{i} P={int(cnt.sum())} E={int((cnt>0).sum())} H={int(cnt.max())} us='+','.join(f'{v[0]:.0f}' for v in vals));peak=max(peak,time.monotonic()-t)
        wx_need(6.,'R0 diagnostics');w=dc_weights(5353,dev);wx_need(4.,'R0 repack/JIT');bank=wi_prepare(w);lines.extend(issue_phases(w,bank,flush))
    except WxBudget as e:lines.append('BUDGET '+str(e))
    for j,name in enumerate(WI_NAMES):
        vals=[r[j] for r in records]
        if vals:
            ts=[v[0] for v in vals];good=all(e<=.01 and r<=.05 for _,e,r in vals)
            lines.append(f'AGG {name} n={len(vals)} mean={statistics.mean(ts):.1f} med={statistics.median(ts):.1f} /base={statistics.mean(r[j][0]/r[0][0] for r in records):.3f} l2={max(v[1] for v in vals):.5f} row={max(v[2] for v in vals):.4f} '+('PASS' if good else 'REJECT'))
    lines.append(f'COMPLETE seeds={len(records)}/12 used={time.monotonic()-start:.1f} left={deadline-time.monotonic():.1f}');return lines
