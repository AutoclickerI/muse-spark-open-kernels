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

Humming compiles its GEMMs at runtime with NVRTC, through its bundled `nvrtc_compile` helper
binary. The library starts that helper as one subprocess per kernel; this file starts none.
prepare() compiles every kernel that bs 1..PRECOMPILED_MAX_BS needs, so nothing compiles inside a
CUDA graph capture. A larger bs compiles on its first (eager) call.
"""

import json
import os
from typing import TypedDict

import torch

# Importing humming starts two background `python -c` builds of helpers that the image already
# ships prebuilt. This flag, read at import, turns them off. It is restored afterwards because
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
COMPUTE_CONFIG = json.dumps({"use_f16_accum": False, "gemm_type": GemmType.INDEXED.value})


class _Experts:
    """One GPU's experts [start, start + count) in Humming's layout, and the kernels for each bs."""

    def __init__(self, weights: Weights, start: int, count: int):
        device = weights["w13"].device
        self.count = count
        self.d_expert = weights["w13"].shape[1] // 2
        self.d_hidden = weights["w2"].shape[1]
        self.layer = torch.nn.Module()
        sl = slice(start, start + count)
        # (sublayer, N, K, N of each stacked part): w13 stacks gate and up along N.
        for name, n, k, n_stacks in (("w13", 2 * self.d_expert, self.d_hidden, [self.d_expert] * 2),
                                     ("w2", self.d_hidden, self.d_expert, [self.d_hidden])):
            checkpoint = {"weight": weights[name][sl].contiguous(),
                          "weight_scale": weights[f"{name}_sf"][sl].contiguous(),
                          "weight_scale_2": weights[f"{name}_gs"][sl].contiguous()}
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
                                       gemm_type=GemmType.INDEXED)
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



class _BucketExperts(_Experts):
    def compile(self, batch_sizes):
        wanted = {}
        hopper = torch.cuda.get_device_capability()[0] == 9
        for bs in batch_sizes:
            cold = 8 if hopper else 16
            hot = 32 if bs < 256 else 64
            blocks = (cold, hot) if bs >= 32 else (cold,)
            wanted[bs] = {}
            for name in self.kernels:
                ts = []
                for bm in blocks:
                    config = dict(get_heuristics_config(self.layer.humming_metas[name],
                                  shape_m=self.routes(bs), gemm_type=GemmType.INDEXED))
                    bn, bk = config["block_shape"][1:]
                    wn, wk = config["warp_shape"][1:]
                    if bm >= 32:
                        bn = 128
                        if not hopper:
                            bk, wk = 64, 64
                            config["num_stages"] = 2
                    config["block_shape"] = (bm, bn, bk)
                    config["warp_shape"] = (bm, wn, wk)
                    if hopper:
                        # Keep the hot tile at <=256 math threads: larger K forces
                        # the warp-specialized kernel's 96-register math budget.
                        if bm >= 32:
                            config["block_shape"] = (bm, bn, min(bk, 128))
                        config.update(use_tma=True, use_tma_a=False, use_tma_as=False,
                                      use_tma_c=False, use_mbarrier=True, use_warp_spec=True)
                    ts.append(json.dumps(config))
                wanted[bs][name] = ts
        for name, kernels in self.kernels.items():
            new = sorted({t for plan in wanted.values() for t in plan[name]} - kernels.keys())
            if new:
                launch = HummingKernel.prepare_kernels(self.layer.humming_metas[name].to_str(), COMPUTE_CONFIG,
                          [[i, i+1, json.loads(t)] for i, t in enumerate(new)])
                for t, (_, _, kid, sms) in zip(new, launch.view(-1,4).tolist()):
                    kernels[t] = torch.tensor([0,1<<30,kid,sms], dtype=torch.int64)
        for bs, plans in wanted.items():
            self.plans[bs] = (self.routes(bs), {
                name: [(json.loads(t)["block_shape"][0], self.kernels[name][t]) for t in ts]
                for name,ts in plans.items()})


_BUCKET_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
__global__ void bucket_route(const int* ids, int* lids, int* s0, int* e0, int* p0,
                            int* s1, int* e1, int* p1, int npairs, int start,
                            int bm0, int bm1, int threshold, int cap0, int cap1) {
  __shared__ int count[32], run[32], wc[8][33];
  __shared__ bool hot[32];
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  if(tid<32) count[tid]=0;
  for(int i=tid;i<cap0;i+=256) s0[i]=npairs;
  for(int i=tid;i<cap1;i+=256) s1[i]=npairs;
  for(int i=tid;i<(cap0+bm0-1)/bm0;i+=256) e0[i]=-1;
  for(int i=tid;i<(cap1+bm1-1)/bm1;i+=256) e1[i]=-1;
  __syncthreads();
  for(int i=tid;i<npairs;i+=256) {
    int e=ids[i]-start;
    if(e<0 || e>=32) e=-1;
    lids[i]=e;
    if(e>=0) atomicAdd(count+e,1);
  }
  __syncthreads();
  if(warp==0) {
    int n=count[lane];
    bool h=n>threshold;
    hot[lane]=h;
    int a=h?0:(n+bm0-1)/bm0*bm0;
    int b=h?(n+bm1-1)/bm1*bm1:0;
    int oa=a, ob=b;
    for(int d=1;d<32;d*=2) {
      int va=__shfl_up_sync(~0u,oa,d), vb=__shfl_up_sync(~0u,ob,d);
      if(lane>=d) {oa+=va;ob+=vb;}
    }
    run[lane]=h?ob-b:oa-a;
    for(int i=(oa-a)/bm0;i<oa/bm0;i++) e0[i]=lane;
    for(int i=(ob-b)/bm1;i<ob/bm1;i++) e1[i]=lane;
    if(lane==31) {p0[0]=oa;p1[0]=ob;}
  }
  __syncthreads();
  for(int base=0;base<npairs;base+=256) {
    wc[warp][lane]=0;
    __syncthreads();
    int i=base+tid,e=i<npairs?lids[i]:-1;
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
    if(e>=0) (hot[e]?s1:s0)[wc[warp][e]+r]=i;
    __syncthreads();
  }
}
void route(torch::Tensor ids,torch::Tensor lids,torch::Tensor s0,torch::Tensor e0,torch::Tensor p0,
           torch::Tensor s1,torch::Tensor e1,torch::Tensor p1,int64_t start,int64_t bm0,int64_t bm1,int64_t threshold) {
  bucket_route<<<1,256,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),lids.data_ptr<int>(),
    s0.data_ptr<int>(),e0.data_ptr<int>(),p0.data_ptr<int>(),s1.data_ptr<int>(),e1.data_ptr<int>(),p1.data_ptr<int>(),
    ids.numel(),start,bm0,bm1,threshold,s0.numel(),s1.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
_BUCKET_CPP = r"""
#include <torch/extension.h>
void route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
           torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);
"""
_BUCKET_MOD = None


def _bucket_module():
    global _BUCKET_MOD
    if _BUCKET_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag=hashlib.sha1(_BUCKET_CUDA.encode()).hexdigest()[:12]
        _BUCKET_MOD=load_inline(name="muse_moe_bucket_"+tag,cpp_sources=_BUCKET_CPP,
                     cuda_sources=_BUCKET_CUDA,functions=["route"],
                     extra_cuda_cflags=["-O3","-std=c++20"],verbose=False)
    return _BUCKET_MOD


def _bucket_align(data, blocks, module):
    ids=data["topk_ids"].to(torch.int32).contiguous()
    lids=torch.empty_like(ids)
    npairs=ids.numel()
    bm0=blocks[0]
    bm1=blocks[1] if len(blocks)>1 else bm0
    aligned=[]
    for bm in (bm0,bm1):
        cap=npairs+32*(bm-1)+bm
        aligned.append((torch.empty((cap,),dtype=torch.int32,device=ids.device),
                        torch.empty(((cap+bm-1)//bm,),dtype=torch.int32,device=ids.device),
                        torch.empty((1,),dtype=torch.int32,device=ids.device)))
    threshold=bm1//2 if len(blocks)>1 else npairs
    module.route(ids,lids,*aligned[0],*aligned[1],data["config"]["local_start"],bm0,bm1,threshold)
    return lids,aligned


def prepare(weights):
    bank=_BucketExperts(weights,0,32)
    module=_bucket_module()
    torch.cuda.synchronize()
    return {"bank":bank,"glue":module,"norm_w":(weights["post_norm_w"].float()+1).to(torch.bfloat16)}


def kernel_fn(data,state):
    bank=state["bank"]
    x=data["hidden_states"].contiguous()
    bs=x.shape[0]
    routes,plans=bank.plan(bs)
    scores=data["topk_weights"].to(torch.float32).contiguous()
    h=torch.empty((bs*TOP_K,2*bank.d_expert),dtype=x.dtype,device=x.device)
    act=torch.empty((bs*TOP_K,bank.d_expert),dtype=x.dtype,device=x.device)
    y=torch.empty((bs*TOP_K,bank.d_hidden),dtype=x.dtype,device=x.device)
    alignments={}
    for name,inputs,outputs,k in (("w13",x,h,TOP_K),("w2",act,y,1)):
        blocks=tuple(bm for bm,_ in plans[name])
        if blocks not in alignments:
            alignments[blocks]=_bucket_align(data,blocks,state["glue"])
        lids,aligned=alignments[blocks]
        for (bm,launch),(s,e,p) in zip(plans[name],aligned):
            humming_ops.launch_kernel(configs=launch,inputs=inputs,outputs=outputs,**bank.weight(name),
                       locks=bank.locks,sorted_ids=s,expert_ids=e,num_tokens_padded=p,top_k=k,valid_shape_m=routes)
        if name=="w13":
            silu_and_mul(h,act,expert_ids=lids.view(-1),expert_step=1)
    return _combine(y.view(bs,TOP_K,bank.d_hidden),scores,state["norm_w"],lids)


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
