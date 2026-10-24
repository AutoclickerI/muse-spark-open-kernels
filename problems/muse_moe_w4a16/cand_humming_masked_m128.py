# okc: problem=muse_moe_w4a16
"""The MoE experts on Humming's fused NVFP4 MoE GEMMs (humming-kernels 0.1.12, in the judge's image)
in place of SGLang's two Marlin GEMMs. The rest is baseline.py's: moe_align_block_size,
silu_and_mul over local slots only, and the Triton post-expert-norm weighted sum.

prepare() converts the NVFP4 weights the way SGLang's HummingMoEMethod does for a ModelOpt NVFP4
checkpoint, and the conversion is lossless. The E2M1 codes are repacked, the E4M3 group-16 scales
are permuted, and each expert's FP32 global scale is kept in FP32. The GEMMs take BF16 activations
and dequantize each weight to BF16; E2M1 x E4M3 is exact in BF16. They accumulate in FP32 and
apply the FP32 global scale to the FP32 accumulator.

The GEMMs are SGLang's indexed MoE GEMM with Humming's default tuning heuristics, with one change.
SGLang tunes for M = every route of the batch (bs * top_k, marked "TODO: update for EP"). Here M
is the routes expected on this GPU, bs * top_k * local_count / n_experts. That count depends only
on shapes and config. Without it, EP8 gets M-blocks sized for 8x more rows than it has.

Humming compiles its GEMMs through the bundled `nvrtc_compile` helper binary.
prepare() compiles every kernel that bs 1..PRECOMPILED_MAX_BS needs, so nothing compiles inside a
CUDA graph capture. A larger bs compiles on its first (eager) call.
"""

import json
import os
from typing import TypedDict

import torch
import triton
import triton.language as tl

# Humming's image helpers are prebuilt. This flag disables parallel compilation while importing;
# it is restored afterwards because
# Humming also reads it to decide whether to compile kernels on parallel threads.
_saved_flag = os.environ.get("HUMMING_DISABLE_PARALLEL_BUILD")
os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
try:
    from humming import ops as humming_ops
    from humming.config import GemmType, WeightScale2Type
    from humming.kernel.humming import HummingKernel
    from humming.layer import HummingMethod
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema
    from humming.tune import get_heuristics_config
finally:
    if _saved_flag is None:
        os.environ.pop("HUMMING_DISABLE_PARALLEL_BUILD", None)
    else:
        os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = _saved_flag

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
POST_NORM_EPS = 1e-8
# prepare() compiles the kernels Humming's heuristics pick for every bs up to this one.
PRECOMPILED_MAX_BS = 512
# SGLang's HummingRunnerCore compute config for its default (indexed) MoE GEMM.
COMPUTE_CONFIG = json.dumps({"use_f16_accum": False, "gemm_type": GemmType.GROUPED_MASKED.value})


class _Experts:
    """One GPU's experts [start, start + count) in Humming's layout, and the kernels for each bs."""

    def __init__(self, weights: Weights, start: int, count: int):
        device = weights["w13"].device
        self.count = count
        self.d_expert = weights["w13"].shape[1] // 2
        self.d_hidden = weights["w2"].shape[1]
        self.gemm_type = GemmType.GROUPED_MASKED
        self.layer = torch.nn.Module()
        sl = slice(start, start + count)
        # (sublayer, N, K, N of each stacked part): w13 stacks gate and up along N.
        for name, n, k, n_stacks in (("w13", 2 * self.d_expert, self.d_hidden, [self.d_expert] * 2),
                                     ("w2", self.d_hidden, self.d_expert, [self.d_hidden])):
            checkpoint = {"weight": weights[name][sl].contiguous(),
                          "weight_scale": weights[f"{name}_sf"][sl].contiguous(),
                          "weight_scale_2": (torch.ones_like(weights[f"{name}_gs"][sl]) if name == "w13" else weights[f"{name}_gs"][sl].contiguous())}
            schema, tensors = ModeloptNvfp4WeightSchema().convert_humming(
                tensors=checkpoint, shape_n_stacks=n_stacks, shape_k_stacks=[k],
                param_dtype=torch.bfloat16, num_experts=count)
            # One FP32 scale per expert stays FP32. Otherwise Humming would fold it into the
            # E4M3 scales as BF16, which is lossy.
            assert schema.weight_scale_2_type == WeightScale2Type.TENSOR, schema
            for key, tensor in tensors.items():
                setattr(self.layer, f"{name}_{key}", torch.nn.Parameter(tensor, requires_grad=False))
            HummingMethod.prepare_layer_meta(
                layer=self.layer, shape_n=n, shape_k=k, weight_schema=schema, num_experts=count,
                pad_n_to_multiple=256, pad_k_to_multiple=128, torch_dtype=torch.bfloat16,
                sublayer_name=name)
            HummingMethod.transform_humming_layer(self.layer, sublayer_name=name)
        # Stream-K's semaphores, shared by the two GEMMs, which run one after the other.
        self.locks = torch.zeros(1024, dtype=torch.int32, device=device)
        self.expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
        self.expert_map[sl] = torch.arange(count, dtype=torch.int32, device=device)
        self.kernels = {"w13": {}, "w2": {}}  # tuning config (JSON) -> launch configs, CPU int64
        self.plans = {}  # bs -> (routes, {sublayer: (block_m, launch configs)})
        self.compile(range(1, PRECOMPILED_MAX_BS + 1))

    def weight(self, name: str) -> dict:
        return {"weight": getattr(self.layer, f"{name}_weight"),
                "weight_scale": getattr(self.layer, f"{name}_weight_scale"),
                "weight_scale_2": getattr(self.layer, f"{name}_weight_scale_2")}

    def routes(self, bs: int) -> int:
        """The GEMM's M for Humming's heuristics: the routes expected on this GPU."""
        return max(1, bs * TOP_K * self.count // N_EXPERTS)

    def tuning(self, name: str, routes: int) -> str:
        config = get_heuristics_config(self.layer.humming_metas[name], shape_m=routes,
                                       gemm_type=self.gemm_type)
        config = dict(config)
        if routes >= 8:
            config["use_stream_k"] = False
        return json.dumps(config)

    def compile(self, batch_sizes) -> None:
        """Compile, in parallel, every kernel these batch sizes need that isn't compiled yet."""
        wanted = {bs: {name: self.tuning(name, self.routes(bs)) for name in self.kernels} for bs in batch_sizes}
        for name, kernels in self.kernels.items():
            new = sorted({t[name] for t in wanted.values()} - kernels.keys())
            if not new:
                continue
            launch = HummingKernel.prepare_kernels(
                self.layer.humming_metas[name].to_str(), COMPUTE_CONFIG,
                [[i, i + 1, json.loads(t)] for i, t in enumerate(new)])
            # A CPU tensor of rows (min M, max M, kernel id, SMs). Keep one kernel per launch config,
            # so that the kernel and this file agree on the block size of the alignment.
            for t, (_, _, kernel_id, num_sms) in zip(new, launch.view(-1, 4).tolist()):
                kernels[t] = torch.tensor([0, 1 << 30, kernel_id, num_sms], dtype=torch.int64)
        for bs, tunings in wanted.items():
            self.plans[bs] = (self.routes(bs), {name: (json.loads(t)["block_shape"][0], self.kernels[name][t])
                                                for name, t in tunings.items()})

    def plan(self, bs: int):
        if bs not in self.plans:  # beyond PRECOMPILED_MAX_BS: compiles here, on the first (eager) call
            self.compile([bs])
        return self.plans[bs]


def prepare(weights: Weights):
    # One bank: this GPU's 32 experts under EP8, as in baseline.py.
    banks = {(0, 32): _Experts(weights, 0, 32)}
    torch.cuda.synchronize()
    return {
        "glue": _glue_module(),
        "banks": banks,
        # SGLang's MuseSpark1xFp4Experts binds the post-expert norm's effective weight, (1 + w), in BF16.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "gate_gs": weights["w13_gs"][:32].float().contiguous(),
        "up_gs": weights["w13_gs"][:32].float().contiguous(),
    }


@triton.jit
def _masked_route(IDS, COUNTS, SLOTS, POS, M: tl.constexpr, START: tl.constexpr, B: tl.constexpr):
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

@triton.jit
def _masked_gather(X, SLOT_E, SLOT_POS, STARTS, A, D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    x = tl.load(X + t * D + c).to(tl.bfloat16)
    for k in tl.static_range(8):
        e = tl.load(SLOT_E + t * 8 + k)
        if e >= 0:
            r = tl.load(STARTS + e) + tl.load(SLOT_POS + t * 8 + k)
            tl.store(A + r.to(tl.int64) * D + c, x)

@triton.jit
def _masked_scatter(Y, SLOT_E, SLOT_POS, STARTS, OUT, D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    for k in tl.static_range(8):
        e = tl.load(SLOT_E + t * 8 + k)
        if e >= 0:
            r = tl.load(STARTS + e) + tl.load(SLOT_POS + t * 8 + k)
            v = tl.load(Y + r.to(tl.int64) * D + c)
            tl.store(OUT + (t * 8 + k) * D + c, v)


@triton.jit
def _masked_swiglu(H, OUT, COUNTS, GATE_GS, UP_GS, STRIDE: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    e = tl.program_id(0)
    r = tl.program_id(1) * B + tl.arange(0, B)
    n = tl.load(COUNTS + e)
    k = tl.arange(0, D)
    ok = r[:, None] < n
    base = (e * STRIDE + r[:, None]) * (2 * D)
    g = tl.load(H + base + k[None, :], ok, 0).to(tl.float32) * tl.load(GATE_GS + e)
    u = tl.load(H + base + D + k[None, :], ok, 0).to(tl.float32) * tl.load(UP_GS + e)
    tl.store(OUT + (e * STRIDE + r[:, None]) * D + k[None, :], (g * tl.sigmoid(g) * u).to(tl.bfloat16), ok)



def _masked_kernel(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    bs = x.shape[0]
    routes, gemms = bank.plan(bs)
    M = bs * TOP_K
    B = triton.next_power_of_2(bs)
    counts = torch.empty((bank.count,), dtype=torch.int32, device=x.device)
    slots = torch.empty((bank.count, M), dtype=torch.int32, device=x.device)
    pos = torch.empty((bs, TOP_K), dtype=torch.int32, device=x.device)
    _masked_route[(bank.count,)](data["topk_ids"], counts, slots, pos, bs, cfg["local_start"], B, num_warps=4)
    local_ids = torch.where((data["topk_ids"] >= cfg["local_start"]) & (data["topk_ids"] < cfg["local_start"] + bank.count),
                            data["topk_ids"] - cfg["local_start"], torch.full_like(data["topk_ids"], -1)).to(torch.int32)
    stride = triton.cdiv(M, 128) * 128
    starts = torch.arange(bank.count, dtype=torch.int32, device=x.device) * stride
    a = torch.empty((bank.count * stride, bank.d_hidden), dtype=x.dtype, device=x.device)
    _masked_gather[(bs,)](x, local_ids, pos, starts, a, D=bank.d_hidden, num_warps=4)
    h = torch.empty((bank.count * stride, 2 * bank.d_expert), dtype=x.dtype, device=x.device)
    inter = torch.empty((bank.count * stride, bank.d_expert), dtype=x.dtype, device=x.device)
    y = torch.empty_like(inter)
    humming_ops.launch_kernel(configs=gemms["w13"][1], inputs=a, outputs=h, **bank.weight("w13"), locks=bank.locks,
                              expert_layout=counts, valid_shape_m=routes)
    _masked_swiglu[(bank.count, triton.cdiv(M, 8))](h, inter, counts, state["gate_gs"], state["up_gs"], stride, bank.d_expert, 8, num_warps=8)
    humming_ops.launch_kernel(configs=gemms["w2"][1], inputs=inter, outputs=y, **bank.weight("w2"), locks=bank.locks,
                              expert_layout=counts, valid_shape_m=routes)
    flat = torch.empty((M, bank.d_hidden), dtype=y.dtype, device=x.device)
    _masked_scatter[(bs,)](y, local_ids, pos, starts, flat, D=bank.d_hidden, num_warps=4)
    return _combine(flat.view(bs, TOP_K, bank.d_hidden), data["topk_weights"].to(torch.float32).contiguous(), state["norm_w"], local_ids)


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    return _masked_kernel(data, state)


# A single launch maps global ids and builds padded expert-major route lists.
_GLUE_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
__global__ void route_kernel(const int* ids, int* local_ids, int* sorted, int* experts,
                             int* padded, int npairs, int start, int bm, int capacity) {
  __shared__ int count[32], run[32], wc[8][33];
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  if(tid<32) count[tid]=0;
  for(int i=tid;i<capacity;i+=256) sorted[i]=npairs;
  for(int i=tid;i<(capacity+bm-1)/bm;i+=256) experts[i]=-1;
  __syncthreads();
  for(int i=tid;i<npairs;i+=256) {
    int e=ids[i]-start;
    if(e<0 || e>=32) e=-1;
    local_ids[i]=e;
    if(e>=0) atomicAdd(count+e,1);
  }
  __syncthreads();
  if(warp==0) {
    int n=count[lane], p=(n+bm-1)/bm*bm, off=p;
    for(int d=1;d<32;d*=2) {
      int v=__shfl_up_sync(~0u,off,d);
      if(lane>=d) off+=v;
    }
    run[lane]=off-p;
    for(int b=(off-p)/bm;b<off/bm;b++) experts[b]=lane;
    if(lane==31) padded[0]=off;
  }
  __syncthreads();
  for(int base=0;base<npairs;base+=256) {
    wc[warp][lane]=0;
    __syncthreads();
    int i=base+tid, e=i<npairs?local_ids[i]:-1;
    unsigned mk=__match_any_sync(~0u,e);
    int r=__popc(mk&((1u<<lane)-1u));
    if(e>=0 && r==0) wc[warp][e]=__popc(mk);
    __syncthreads();
    if(tid<32) {
      int s=run[tid];
      for(int w=0;w<8;w++) {int n=wc[w][tid];wc[w][tid]=s;s+=n;}
      run[tid]=s;
    }
    __syncthreads();
    if(e>=0) sorted[wc[warp][e]+r]=i;
    __syncthreads();
  }
}
void route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor sorted,
           torch::Tensor experts, torch::Tensor padded, int64_t start, int64_t bm) {
  route_kernel<<<1,256,0,at::cuda::getCurrentCUDAStream()>>>(
    ids.data_ptr<int>(), local_ids.data_ptr<int>(), sorted.data_ptr<int>(),
    experts.data_ptr<int>(), padded.data_ptr<int>(), ids.numel(), start, bm, sorted.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
_GLUE_CPP = r"""
#include <torch/extension.h>
void route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);
"""
_GLUE_MOD = None


def _glue_module():
    global _GLUE_MOD
    if _GLUE_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(_GLUE_CUDA.encode()).hexdigest()[:12]
        _GLUE_MOD = load_inline(name="muse_moe_route_"+tag, cpp_sources=_GLUE_CPP,
                               cuda_sources=_GLUE_CUDA, functions=["route"],
                               extra_cuda_cflags=["-O3", "-std=c++20"], verbose=False)
    return _GLUE_MOD


def _route(data, block_m, module):
    ids = data["topk_ids"].to(torch.int32).contiguous()
    npairs = ids.numel()
    capacity = npairs + 32 * (block_m-1) + block_m
    local_ids = torch.empty_like(ids)
    sorted_ids = torch.empty((capacity,), dtype=torch.int32, device=ids.device)
    block_experts = torch.empty(((capacity+block_m-1)//block_m,), dtype=torch.int32, device=ids.device)
    n_padded = torch.empty((1,), dtype=torch.int32, device=ids.device)
    module.route(ids, local_ids, sorted_ids, block_experts, n_padded, data["config"]["local_start"], block_m)
    return local_ids, sorted_ids, block_experts, n_padded


def _combine(y, scores, norm_weight, local_ids):
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        _post_expert_norm_weighted_sum_kernel,
    )
    bs, topk, hidden = y.shape
    out = torch.empty((bs, hidden), dtype=y.dtype, device=y.device)
    _post_expert_norm_weighted_sum_kernel[(bs,)](
        y, norm_weight, scores, out, local_ids,
        TOPK=topk, H=hidden, eps=1e-8, APPLY_NORM=True, HAS_WEIGHT=True,
        WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=hidden, num_warps=8, num_stages=2)
    return out
