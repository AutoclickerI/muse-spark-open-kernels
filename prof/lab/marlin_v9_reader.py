"""Direct V9 reader over stock Marlin NVFP4; no second serving bank."""

# Source problems/muse_moe_w4a16/baseline.py sha256=4fc48bd9cdfd749863c93a2dbf4e556e0c38fb04acc4fa48a53ef34cbf6c92a1
"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""
from typing import TypedDict as ma_TypedDict
import types as ma_types
import torch as ma_torch
from sgl_kernel.scalar_type import scalar_types as ma_scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul as ma_silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm as ma_moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as ma_fused_post_expert_norm_weighted_sum
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as ma_moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace as ma_marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin as ma_prepare_moe_nvfp4_layer_for_marlin

class ma_Weights(ma_TypedDict):
    w13: ma_torch.Tensor
    w13_sf: ma_torch.Tensor
    w13_gs: ma_torch.Tensor
    w2: ma_torch.Tensor
    w2_sf: ma_torch.Tensor
    w2_gs: ma_torch.Tensor
    post_norm_w: ma_torch.Tensor

class ma_Config(ma_TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str

class ma_Inputs(ma_TypedDict):
    hidden_states: ma_torch.Tensor
    topk_ids: ma_torch.Tensor
    topk_weights: ma_torch.Tensor
    config: ma_Config
ma_N_EXPERTS = 256
ma_TOP_K = 8
ma_BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ma_POST_NORM_EPS = 1e-08

def ma__marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = ma_types.SimpleNamespace(quant_config=ma_types.SimpleNamespace(group_size=16), moe_runner_config=ma_types.SimpleNamespace(is_gated=True), intermediate_size_per_partition=weights['w13'].shape[1] // 2, params_dtype=ma_torch.bfloat16, w13_weight=weights['w13'][sl].contiguous(), w2_weight=weights['w2'][sl].contiguous(), w13_weight_scale=weights['w13_sf'][sl].contiguous(), w2_weight_scale=weights['w2_sf'][sl].contiguous(), w13_weight_scale_2=weights['w13_gs'][sl].contiguous(), w2_weight_scale_2=weights['w2_gs'][sl].contiguous())
    ma_prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer

def ma_prepare(weights):
    device = weights['w13'].device
    banks = {}
    for start, count in ((0, 32),):
        layer = ma__marlin_layer(weights, start, count)
        expert_map = None
        if count != ma_N_EXPERTS:
            expert_map = ma_torch.full((ma_N_EXPERTS,), -1, dtype=ma_torch.int32, device=device)
            expert_map[start:start + count] = ma_torch.arange(count, dtype=ma_torch.int32, device=device)
        banks[start, count] = {'w13': layer.w13_weight.data, 'w13_s': layer.w13_weight_scale.data, 'w13_gs': layer.w13_weight_scale_2.data, 'w2': layer.w2_weight.data, 'w2_s': layer.w2_weight_scale.data, 'w2_gs': layer.w2_weight_scale_2.data, 'expert_map': expert_map, 'count': count, 'size_n': layer.intermediate_size_per_partition}
    ma_torch.cuda.synchronize()
    return {'banks': banks, 'workspace': ma_marlin_make_workspace(device, max_blocks_per_sm=4), 'norm_w': (weights['post_norm_w'].float() + 1.0).to(ma_torch.bfloat16)}

def ma_kernel_fn(data, state):
    cfg = data['config']
    bank = state['banks'][cfg['local_start'], cfg['local_count']]
    x = data['hidden_states'].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank['size_n']
    gemm1_n = 2 * size_n
    expert_map = bank['expert_map']
    local_ids = data['topk_ids'].to(ma_torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data['topk_weights'].to(ma_torch.float32).contiguous()
    for block_size_m in ma_BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * ma_TOP_K / bank['count'] / block_size_m < 0.9:
            break
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token
        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = ma_moe_align_block_size(local_ids, block_size_m, ma_N_EXPERTS if expert_map is not None else bank['count'])
    cache2 = ma_torch.empty((num_tokens * ma_TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = ma_torch.empty((num_tokens * ma_TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[:num_tokens * ma_TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[:num_tokens * ma_TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None, b_q_type=ma_scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    ws = state['workspace']
    cache1 = ma_moe_wna16_marlin_gemm(x, cache1, bank['w13'], None, bank['w13_s'], bank['w13_gs'], None, None, None, ws, sorted_ids, block_experts, n_padded, topk_weights, top_k=ma_TOP_K, size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    ma_silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = ma_moe_wna16_marlin_gemm(cache2, cache3, bank['w2'], None, bank['w2_s'], bank['w2_gs'], None, None, None, ws, sorted_ids, block_experts, n_padded, topk_weights, top_k=1, size_m=num_tokens * ma_TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, ma_TOP_K, size_k)
    return y

# Source serving/sglang/python/sglang/srt/layers/muse_spark_moe.py sha256=5177060dc5daa558a2580c499169957eed5fb6df9f8f1500d532aba517f945d6
"""Serving kernel for Muse Spark routed NVFP4 experts (V9 native + canonical grouped wide path).

This file is a drop-in source boundary: a newer kernel can replace this module

prepare_layer() repacks each local expert losslessly into 36 KB "m16 tiles" (16 weight rows x 4096):
E2M1 nibbles placed so a rotate + AND gives BF16 bits (value * 2^-126, exact).
E4M3 block-scale bytes are only rearranged; CUDA converts them to BF16 (* 2^118).
No prepared BF16/FP8 weight copy is retained. W13 tiles pair 8 gate rows with 8 up rows.

The kernel (grid = #SMs, one CTA per SM, 8 consumer warps + 1 producer warp):
 1. Routing in shared memory: per-expert counts (the producer starts streaming weights as soon
    as these are known), then a deterministic token list per expert.
 2. Phase 1: units = (expert, 16-token block, m16 tile of W13); each CTA takes an even
    contiguous share. The producer streams whole tiles with cp.async.bulk into an smem ring;
    consumers split K across warps (512 each) with the token activations in registers,
    dequantize to BF16, mma.sync m16n8k16 (weights on M, tokens on N), reduce across warps,
    apply the FP32 global scale and SwiGLU, and write the BF16 intermediate.
 3. Phase 2: the same over W2 tiles once the expert's phase-1 tiles are all published
    (per-expert counters); writes FP32 y and accumulates each pair's sum of squares of
    y * (1 + w) for the post-expert RMSNorm.
 4. Grid barrier, then the combine: out[t] = sum over local pairs of g * rmsnorm(y) in BF16.
Units are ordered (expert, m-chunk, token block, m) with the chunk = one CTA's share, so the
token blocks of a hot expert run on neighbouring CTAs at the same time and share its weights
through L2. All counters return to zero by the end of each call.
"""
import torch as v9_torch
v9_H = 4096
v9_NLOC = 32
v9_TOPK = 8
v9_TILE_BYTES = 36864
v9_NT1, v9_NT2 = (512, 256)
v9_NCNT = 64
v9_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13_gate, const float* __restrict__ gs13_up, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ pos_map, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int dense_mode, int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n        pos_map[i] = pos;\n      } else {\n        pos_of[i] = -1;\n        pos_map[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc_gate = gs13_gate[e] * 256.f;\n        const float sc_up = gs13_up[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc_gate;\n            U *= sc_up;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  if (!dense_mode) for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13_gate, torch::Tensor gs13_up, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor pos_map, torch::Tensor cnt, torch::Tensor out, int64_t dense_mode, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* idp = ids.data_ptr<int>();\n  auto* twp = tw.data_ptr<float>();\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* w13pp = w13p.data_ptr<uint8_t>();\n  auto* w2pp = w2p.data_ptr<uint8_t>();\n  auto* ggp = gs13_gate.data_ptr<float>();\n  auto* gup = gs13_up.data_ptr<float>();\n  auto* g2p = gs2.data_ptr<float>();\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* yp = y.data_ptr<float>();\n  auto* ssp = ss.data_ptr<float>();\n  auto* posp = pos_map.data_ptr<int>();\n  auto* cntp = cnt.data_ptr<int>();\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV) do {                                                                                       \\\n  int dm = static_cast<int>(dense_mode), m = M, ls = static_cast<int>(local_start), ns = static_cast<int>(nstage); \\\n  void* args[] = {&xp, &idp, &twp, &pp, &w13pp, &w2pp, &ggp, &gup, &g2p, &ip, &yp, &ssp, &posp, &cntp, &op, &dm, &m, &ls, &ns}; \\\n  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(moe_fused<NTV>), dim3(static_cast<unsigned>(grid)), dim3(NTHR), args, smem, stream)); \\\n} while (0)\n  LAUNCH(1);\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
v9_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13_gate, torch::Tensor gs13_up, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor pos_map, torch::Tensor cnt, torch::Tensor out, int64_t dense_mode, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid);\n'
v9__MOD = None

def v9__module():
    global v9__MOD
    if v9__MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(v9_CUDA_SRC.encode()).hexdigest()[:12]
        v9__MOD = load_inline(name=f'muse_moe_w4a16_v9_{tag}', cpp_sources=v9_CPP_SRC, cuda_sources=v9_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
        v9__MOD.smem_optin = v9__MOD.max_smem_attr()
    return v9__MOD
v9__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
v9__LA_MASK = 1613783088

def v9__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if v9__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(v9__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
v9__TABLE = v9__bit_table()

def v9__repack(wq: v9_torch.Tensor, sf: v9_torch.Tensor, rowmap: v9_torch.Tensor) -> v9_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = v9_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.view(v9_torch.uint8)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(v9_torch.int64)
    words = v9_torch.zeros((2, T, 8, 8, 32, 2), dtype=v9_torch.int64, device=wq.device)
    for k, h, b, w, p in v9__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(v9_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return v9_torch.cat((wbytes, sbytes), 1).contiguous()
v9_GROUPED_CUDA_SRC = '\n// Canonical V9 36 KiB expert tiles -> exact BF16 register-A WGMMA.\n// No persistent converted or alternative-layout weight bank.\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <stdio.h>\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define USE_WGMMA 1\n#else\n#define USE_WGMMA 0\n#endif\nnamespace canonical {\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\nconstexpr int H=4096, TB=36864, WB=32768;\nconstexpr uint32_t DQ_MASK=0x81C081C0u, LA_MASK=0x60306030u;\n__device__ __forceinline__ uint32_t rotr(uint32_t x,int s){return __funnelshift_r(x,x,s);}\n__device__ __forceinline__ uint32_t hmul2(uint32_t x,uint32_t y){uint32_t z;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(z):"r"(x),"r"(y));return z;}\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n__device__ __forceinline__ uint32_t sptr(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void wfence(){\n#if USE_WGMMA\n asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n}\n__device__ __forceinline__ void wcommit(){\n#if USE_WGMMA\n asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n#endif\n}\ntemplate<int G> __device__ __forceinline__ void wwait(){\n#if USE_WGMMA\n asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(G):"memory");\n#endif\n}\ntemplate<int N> __device__ __forceinline__ void protect(float (&a)[N/2]){\n#pragma unroll\n for(int j=0;j<N/2;++j)asm volatile("":"+f"(a[j])::"memory");\n}\n__device__ __forceinline__ void keep(uint32_t (&a)[4]){asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]));}\ntemplate<int N> __device__ __forceinline__ void atom(float (&d)[N/2],uint32_t (&a)[4],const uint8_t* xs,int s){\n#if USE_WGMMA\n const uint64_t desc=((uint64_t)(sptr(xs)>>4))|((uint64_t)64<<32)|((uint64_t)1<<62);\n const uint64_t base=desc+(s*2);\n { const uint64_t desc=base;\n   if constexpr (N == 8) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 16) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),\n        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 32) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),\n        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),\n        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 64) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "\n      " {%32, %33, %34, %35}, %36, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),\n        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),\n        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),\n        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),\n        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n  else if constexpr (N == 128) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\n      " {%64, %65, %66, %67}, %68, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),\n        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),\n        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),\n        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),\n        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),\n        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),\n        "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),\n        "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),\n        "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n  }\n }\n#else\n const int lane=threadIdx.x&31,c=lane&3,g=lane>>2;\n#pragma unroll\n for(int i=0;i<N/8;++i){\n  const int n=8*i+g,k=s*32+c*4;\n  const uint8_t* r=xs+n*128;\n  uint32_t b0=*(const uint32_t*)(r+(((k>>4)^(n&7))<<4)+(k&15));\n  uint32_t b1=*(const uint32_t*)(r+((((k+16)>>4)^(n&7))<<4)+((k+16)&15));\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n   :"+f"(d[i*4]),"+f"(d[i*4+1]),"+f"(d[i*4+2]),"+f"(d[i*4+3])\n   :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\n__global__ void histogram(const int* ids,int* counts,int P,int start){\n __shared__ int local[32];int t=threadIdx.x;\n if(t<32)local[t]=0;__syncthreads();\n int j=blockIdx.x*blockDim.x+t;\n if(j<P){int e=ids[j]-start;if((unsigned)e<32)atomicAdd(local+e,1);}\n __syncthreads();if(t<32)atomicAdd(counts+t,local[t]);\n}\n__global__ void plan(const int* counts,int* offsets,int* cursors,int* tasks,int fixed_n){\n int e=threadIdx.x,c=counts[e],sum=c;\n#pragma unroll\n for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,sum,d);if(e>=d)sum+=a;}\n offsets[e]=sum-c;if(e==31)offsets[32]=sum;cursors[e]=0;\n int chosen=fixed_n?fixed_n:(c<=8?8:c<=16?16:c<=32?32:c<=64?64:128);\n#pragma unroll\n for(int i=0;i<5;++i){int n=8<<i,v=chosen==n?(c+n-1)/n:0,z=v;\n#pragma unroll\n  for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,z,d);if(e>=d)z+=a;}\n  tasks[i*33+e]=z-v;if(e==31)tasks[i*33+32]=z;\n }\n}\n__global__ void scatter(const int* ids,const int* offsets,int* cursors,int* pairs,int* positions,int P,int start){\n int j=blockIdx.x*blockDim.x+threadIdx.x;\n if(j<P){int e=ids[j]-start,p=-1;if((unsigned)e<32){p=offsets[e]+atomicAdd(cursors+e,1);pairs[p]=j;}positions[j]=p;}\n}\ntemplate<int N,int MODE> __device__ __forceinline__ void load_x(uint8_t* dst,const __nv_bfloat16* x,const int* pairs,int off,int ne,int nb,int kg){\n for(int j=threadIdx.x;j<N*32;j+=128){\n  int row=j/32,l=j%32,idx=nb*N+row;\n  // Native A\'s K permutation: logical pair 8*s+4*j+q -> physical pair 8*q+2*s+j.\n  int physical=8*(l&3)+2*(l>>3)+((l>>2)&1);\n  int pos=off+idx, xr=idx<ne?(MODE?pairs[pos]>>3:pos):0;\n  const void* src=x+(size_t)xr*H+kg*64+physical*2;\n  void* out=dst+row*128+(((l>>2)^(row&7))*4+(l&3))*4;\n  int bytes=idx<ne?4:0;\n  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"::"r"(sptr(out)),"l"(src),"r"(bytes):"memory");\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int N,int MODE,int STAGES>\n__global__ void __launch_bounds__(128,2) grouped(const uint8_t* weights,const float* gs_gate,const float* gs_up,const __nv_bfloat16* x,__nv_bfloat16* out,\n const int* pairs,const int* offsets,const int* tasks){\n extern __shared__ __align__(1024) uint8_t shared[];\n constexpr int XBYTES=N*128, MTILES=MODE?512:256, NFB=MTILES/4;\n const int t=threadIdx.x,warp=t>>5,lane=t&31;\n const int total=tasks[32]*NFB;\n for(int item=blockIdx.x;item<total;item+=gridDim.x){\n  int ti=item/NFB,fb=item%NFB,e=0;\n#pragma unroll\n  for(int d=16;d;d>>=1)if(tasks[e+d]<=ti)e+=d;\n  int nb=ti-tasks[e],off=offsets[e],ne=offsets[e+1]-off;\n  const uint8_t* wt=weights+((size_t)e*MTILES+fb*4+warp)*TB;\n  float acc[N/2];\n#pragma unroll\n  for(int j=0;j<N/2;++j)acc[j]=0.f;\n#pragma unroll\n  for(int k=0;k<STAGES;++k)load_x<N,MODE>(shared+k*XBYTES,x,pairs,off,ne,nb,k);\n  uint32_t A[2][4] = {};\n  for(int kg=0;kg<64;++kg){\n   int st=kg%STAGES;\n   // At the tail, fewer than STAGES groups remain: wait for every outstanding copy.\n   if(kg+STAGES<=64)asm volatile("cp.async.wait_group %0;"::"n"(STAGES-1):"memory");\n   else asm volatile("cp.async.wait_group 0;":::"memory");\n   __syncthreads();\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n   uint4 w=*(const uint4*)(wt+(kg*32+lane)*16);\n   uint16_t sw=*(const uint16_t*)(wt+WB+(kg/8)*512+lane*16+(kg%8)*2);\n   uint32_t sr=sf_bits(sw&255u);\n   uint32_t su=sf_bits((sw>>8)&255u);\n   uint32_t o[16];dq8(w.x,w.y,o);dq8(w.z,w.w,o+8);\n#pragma unroll\n   for(int s=0;s<4;++s){\n    const uint32_t* os=o+(s>>1)*8+(s&1)*4;\n    uint32_t* a=A[s&1];\n    a[0]=hmul2(os[0],sr);a[1]=hmul2(os[1],su);a[2]=hmul2(os[2],sr);a[3]=hmul2(os[3],su);\n    protect<N>(acc);wfence();atom<N>(acc,A[s&1],shared+st*XBYTES,s);wcommit();protect<N>(acc);\n    wwait<1>();keep(A[(s&1)^1]);\n   }\n   wwait<0>();protect<N>(acc);keep(A[1]);\n   __syncthreads();\n   if(kg+STAGES<64)load_x<N,MODE>(shared+st*XBYTES,x,pairs,off,ne,nb,kg+STAGES);\n  }\n  const float sc_gate=gs_gate[e]*256.f, sc_up=gs_up[e]*256.f;\n#pragma unroll\n  for(int j=0;j<N/8;++j){\n#pragma unroll\n   for(int q=0;q<2;++q){int col=8*j+2*(lane&3)+q,idx=nb*N+col;\n    if(idx<ne){int pos=off+idx;\n     if constexpr(MODE){\n      float g=__bfloat162float(__float2bfloat16_rn(acc[4*j+q]*sc_gate));\n      float u=__bfloat162float(__float2bfloat16_rn(acc[4*j+2+q]*sc_up));\n      out[(size_t)pos*H+fb*32+warp*8+(lane>>2)]=__float2bfloat16_rn((g/(1.f+__expf(-g)))*u);\n     }else{\n#pragma unroll\n      for(int h=0;h<2;++h)out[(size_t)pos*H+fb*64+warp*16+(lane>>2)+8*h]=__float2bfloat16_rn(acc[4*j+2*h+q]*sc_gate);\n     }\n    }\n   }\n  }\n  __syncthreads();\n }\n}\n} // namespace canonical\nvoid route(torch::Tensor ids,torch::Tensor counts,torch::Tensor offsets,torch::Tensor cursors,torch::Tensor pairs,torch::Tensor positions,torch::Tensor tasks,int64_t local_start,int64_t fixed_n){\n auto s=at::cuda::getCurrentCUDAStream();int P=ids.numel();cudaMemsetAsync(counts.data_ptr(),0,128,s);\n canonical::histogram<<<(P+255)/256,256,0,s>>>(ids.data_ptr<int>(),counts.data_ptr<int>(),P,local_start);\n canonical::plan<<<1,32,0,s>>>(counts.data_ptr<int>(),offsets.data_ptr<int>(),cursors.data_ptr<int>(),tasks.data_ptr<int>(),fixed_n);\n canonical::scatter<<<(P+255)/256,256,0,s>>>(ids.data_ptr<int>(),offsets.data_ptr<int>(),cursors.data_ptr<int>(),pairs.data_ptr<int>(),positions.data_ptr<int>(),P,local_start);\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\nvoid gemm(torch::Tensor w,torch::Tensor gs_gate,torch::Tensor gs_up,torch::Tensor x,torch::Tensor out,torch::Tensor pairs,torch::Tensor offsets,torch::Tensor tasks,int64_t mode,int64_t fixed_n,int64_t grid,int64_t stages){\n auto stream=at::cuda::getCurrentCUDAStream();\n#define RUN(N,M,S) canonical::grouped<N,M,S><<<grid,128,N*128*S,stream>>>(w.data_ptr<uint8_t>(),gs_gate.data_ptr<float>(),gs_up.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)out.data_ptr(),pairs.data_ptr<int>(),offsets.data_ptr<int>(),tasks.data_ptr<int>()+(__builtin_ctz(N)-3)*33)\n#define WIDTH(N,M,S) if(fixed_n==0||fixed_n==N){RUN(N,M,S);}\n#define SET(M,S) WIDTH(8,M,S);WIDTH(16,M,S);WIDTH(32,M,S);WIDTH(64,M,S);WIDTH(128,M,S)\n if(mode){if(stages==2){SET(1,2);}else{SET(1,3);}}\n else{if(stages==2){SET(0,2);}else{SET(0,3);}}\n#undef SET\n#undef WIDTH\n#undef RUN\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
v9_GROUPED_CPP_SRC = '\n#include <torch/extension.h>\nvoid route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\nvoid gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);\n'
v9__GROUPED_MOD = None

def v9__grouped_module():
    global v9__GROUPED_MOD
    if v9__GROUPED_MOD is None:
        import hashlib
        import os
        from torch.utils.cpp_extension import load_inline
        arch = v9_torch.cuda.get_device_capability()
        previous = os.environ.get('TORCH_CUDA_ARCH_LIST')
        os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' if arch == (9, 0) else f'{arch[0]}.{arch[1]}'
        try:
            tag = hashlib.sha1(v9_GROUPED_CUDA_SRC.encode()).hexdigest()[:12]
            v9__GROUPED_MOD = load_inline(name=f'muse_canonical_serving_{tag}', cpp_sources=v9_GROUPED_CPP_SRC, cuda_sources=v9_GROUPED_CUDA_SRC, functions=['route', 'gemm'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
        finally:
            if previous is None:
                os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else:
                os.environ['TORCH_CUDA_ARCH_LIST'] = previous
    return v9__GROUPED_MOD

def v9__grouped_call(data, state):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(v9_torch.int32).contiguous()
    cfg = data['config']
    assert cfg['local_count'] == v9_NLOC and cfg['d_hidden'] == v9_H and (cfg['d_expert'] == v9_H)
    m = x.shape[0]
    count = m * v9_TOPK
    dev = x.device
    counts = v9_torch.empty((v9_NLOC,), dtype=v9_torch.int32, device=dev)
    offsets = v9_torch.empty((v9_NLOC + 1,), dtype=v9_torch.int32, device=dev)
    cursors = v9_torch.empty((v9_NLOC,), dtype=v9_torch.int32, device=dev)
    pairs = v9_torch.empty((count,), dtype=v9_torch.int32, device=dev)
    positions = v9_torch.empty((count,), dtype=v9_torch.int32, device=dev)
    tasks = v9_torch.empty((5 * (v9_NLOC + 1),), dtype=v9_torch.int32, device=dev)
    inter = v9_torch.empty((count, v9_H), dtype=v9_torch.bfloat16, device=dev)
    y = v9_torch.empty_like(inter)
    mod = state['grouped_mod']
    mod.route(ids, counts, offsets, cursors, pairs, positions, tasks, cfg['local_start'], 0)
    grid = state['grid'] * 4
    mod.gemm(state['w13p'], state['gs13_gate'], state['gs13_up'], x, inter, pairs, offsets, tasks, 1, 0, grid, 2)
    mod.gemm(state['w2p'], state['gs2'], state['gs2'], inter, y, pairs, offsets, tasks, 0, 0, grid, 2)
    safe = positions.clamp_min(0)
    dense = y.index_select(0, safe).view(m, v9_TOPK, v9_H)
    flat = v9_torch.arange(count, device=dev, dtype=v9_torch.int32).view(m, v9_TOPK)
    valid = (ids >= cfg['local_start']) & (ids < cfg['local_start'] + v9_NLOC)
    return (dense, v9_torch.where(valid, flat, v9_torch.full_like(flat, -1)))

def v9__v9_call(data, state, *, return_dense=False):
    cfg = data['config']
    assert cfg['local_count'] == v9_NLOC and cfg['d_hidden'] == v9_H and (cfg['d_expert'] == v9_H)
    mod = state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(v9_torch.int32).contiguous()
    tw = data['topk_weights'].to(v9_torch.float32).contiguous()
    m = x.shape[0]
    npairs = m * v9_TOPK
    nt = 1
    nstage = 2
    while nstage > 1 and mod.smem_needed(nt, nstage, m) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, m) > mod.smem_optin:
        raise ValueError(f'batch {m} too large for fused V9 shared-memory routing')
    dev = x.device
    inter = v9_torch.empty((npairs, v9_H), dtype=v9_torch.bfloat16, device=dev)
    y = v9_torch.zeros((npairs, v9_H), dtype=v9_torch.float32, device=dev)
    ss = v9_torch.empty((npairs,), dtype=v9_torch.float32, device=dev)
    pos_map = v9_torch.empty((npairs,), dtype=v9_torch.int32, device=dev)
    out = v9_torch.empty((m, v9_H), dtype=v9_torch.bfloat16, device=dev)
    cnt = v9_torch.zeros((v9_NCNT,), dtype=v9_torch.int32, device=dev)
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13_gate'], state['gs13_up'], state['gs2'], inter, y, ss, pos_map, cnt, out, 1, cfg['local_start'], nt, nstage, state['grid'] * mod.resident_blocks(nt, nstage, m))
    safe_map = pos_map.clamp_min(0)
    dense = y.index_select(0, safe_map).to(v9_torch.bfloat16).view(m, v9_TOPK, v9_H)
    flat = v9_torch.arange(npairs, device=dev, dtype=v9_torch.int32).view(m, v9_TOPK)
    valid = (ids >= state['local_start']) & (ids < state['local_start'] + v9_NLOC)
    return (dense, v9_torch.where(valid, flat, v9_torch.full_like(flat, -1)))

def v9_prepare(weights, *, local_start=0):
    dev = weights['w13'].device
    t = v9_torch.arange(v9_NT1, device=dev)
    i = v9_torch.arange(8, device=dev)
    rowmap1 = v9_torch.cat((8 * t[:, None] + i, v9_H + 8 * t[:, None] + i), 1)
    rowmap2 = v9_torch.arange(v9_H, device=dev).view(v9_NT2, 16)
    w13p = v9_torch.empty((v9_NLOC, v9_NT1, v9_TILE_BYTES), dtype=v9_torch.uint8, device=dev)
    w2p = v9_torch.empty((v9_NLOC, v9_NT2, v9_TILE_BYTES), dtype=v9_torch.uint8, device=dev)
    for e in range(v9_NLOC):
        w13p[e] = v9__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = v9__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = v9__module()
    grouped_mod = v9__grouped_module()
    props = v9_torch.cuda.get_device_properties(dev)
    v9_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13_gate': weights['w13_gs_gate'][:v9_NLOC].float().contiguous(), 'gs13_up': weights['w13_gs_up'][:v9_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:v9_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'grid': props.multi_processor_count, 'mod': mod, 'grouped_mod': grouped_mod, 'local_start': int(local_start)}

def v9_kernel_fn(data, state, *, return_dense=False):
    m = data['hidden_states'].shape[0]
    dense, local_map = v9__grouped_call(data, state) if m >= 256 else v9__v9_call(data, state, return_dense=return_dense)
    if return_dense:
        return (dense, local_map)
    valid = local_map >= 0
    norm = (1.0 + state['pnw']).to(v9_torch.bfloat16)
    scaled = dense * norm
    inv = v9_torch.rsqrt(scaled.float().square().mean(dim=-1, keepdim=True) + 1e-08)
    weights = data['topk_weights'].to(v9_torch.bfloat16) * valid.to(v9_torch.bfloat16)
    return (scaled * inv.to(v9_torch.bfloat16) * weights.unsqueeze(-1)).sum(dim=1)

def v9_prepare_layer(layer):
    """Pack raw ModelOpt tensors into the single canonical V9 bank."""
    post_norm = getattr(layer, 'post_expert_norm', None)
    if post_norm is None or getattr(post_norm, 'weight', None) is None:
        raise RuntimeError('Muse V9 needs the loaded post-expert norm weight')
    weights = {'w13': layer.gate_up_proj.detach(), 'w13_sf': layer.gate_up_proj_weight_scale.detach(), 'w13_gs_gate': layer.gate_up_proj_weight_scale_2[:, 0].detach(), 'w13_gs_up': layer.gate_up_proj_weight_scale_2[:, 1].detach() if layer.gate_up_proj_weight_scale_2.shape[1] >= 2 else layer.gate_up_proj_weight_scale_2.reshape(-1).detach(), 'w2': layer.down_proj.detach(), 'w2_sf': layer.down_proj_weight_scale.detach(), 'w2_gs': layer.down_proj_weight_scale_2.detach(), 'post_norm_w': post_norm.weight.detach()}
    if weights['w13'].shape[0] != v9_NLOC:
        raise ValueError(f"Muse V9 expects 32 local experts, got {weights['w13'].shape[0]}")
    return v9_prepare(weights, local_start=layer.local_expert_offset)

def v9_run_layer(state, hidden_states, topk_scores, topk_ids, *, return_dense=False):
    """Graph-capturable V9/grouped invocation used by MuseSpark1xFp4Experts."""
    cfg = {'bs': hidden_states.shape[0], 'd_hidden': hidden_states.shape[1], 'd_expert': v9_H, 'local_start': state['local_start'], 'local_count': v9_NLOC}
    return v9_kernel_fn({'hidden_states': hidden_states, 'topk_weights': topk_scores, 'topk_ids': topk_ids, 'config': cfg}, state, return_dense=return_dense)

# Source serving/sglang/python/sglang/kernels/ops/moe/muse_marlin_v9_reader.py sha256=72470ae68c401962c9302066295f244af1719eb8083887971635e95db81995cb
"""V9 expert arithmetic reading stock Marlin's NVFP4 bank directly."""
import hashlib as dr_hashlib
from torch.utils.cpp_extension import load_inline as dr_load_inline
dr__HELPERS = "\n// Marlin K16/N64 words: each word holds two rows and four K positions.\n__device__ __forceinline__ uint32_t marlin_pair(\n    const uint32_t* weights, int e, int row, int k, int n) {\n  size_t at = static_cast<size_t>(e) * n * 512;\n  at += ((k >> 4) * (n >> 6) + (row >> 6)) * 128;\n  at += ((row & 7) * 4 + ((k & 7) >> 1)) * 4 + ((row & 63) >> 4);\n  uint32_t word = __ldg(weights + at);\n  int v = ((row & 15) >> 3) * 4 + ((k & 15) >> 3) * 2;\n  int shift = (v >> 1) * 4;\n  uint32_t lo = (word >> shift) & 15, hi = (word >> (shift + 16)) & 15;\n  return ((lo & 7) << 6) | ((lo & 8) << 12) |\n         ((hi & 7) << 22) | ((hi & 8) << 28);\n}\n\n__device__ __forceinline__ uint32_t marlin_sf(\n    const uint8_t* bytes, int e, int row, int k, int n) {\n  int canonical = (k >> 4) * n + row;\n  int perm = (canonical & ~63) + (canonical & 7) * 8 + ((canonical & 63) >> 3);\n  int j = (perm & ~3) + ((perm & 3) >> 1) + (perm & 1) * 2;\n  size_t codes = static_cast<size_t>(32) * n * 2048;\n  uint32_t value = __ldg(bytes + codes + static_cast<size_t>(e) * n * 256 + j);\n  // Invert stock's FP16 exponent-bias byte encoding inside the kernel.\n  uint32_t raw = value >= 128 ? value - 120 :\n    value == 104 ? 1 : value == 112 ? 2 : value == 116 ? 3 :\n    value == 120 ? 4 : value == 122 ? 5 : value == 124 ? 6 : value == 126 ? 7 : 0;\n  return sf_bits(raw);\n}\n"
dr__READ = '\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      const uint8_t* bank = pass ? w2p : w13p;\n      const uint32_t* weights = reinterpret_cast<const uint32_t*>(bank);\n      const int width = pass ? 4096 : 8192;\n      const int r0 = (pass ? m * 16 : m * 8) + (lane >> 2);\n      const int r1 = r0 + (pass ? 8 : 4096);\n#pragma unroll\n      for (int g = 0; g < 8; ++g) {\n        const int k = warp * 512 + g * 64 + q * 16;\n        const uint32_t s0 = marlin_sf(bank, e, r0, k, width);\n        const uint32_t s1 = marlin_sf(bank, e, r1, k, width);\n#pragma unroll\n        for (int s = 0; s < 4; ++s) {\n          const uint32_t a0 = hmul2(marlin_pair(weights,e,r0,k+s*4,width),s0);\n          const uint32_t a1 = hmul2(marlin_pair(weights,e,r1,k+s*4,width),s1);\n          const uint32_t a2 = hmul2(marlin_pair(weights,e,r0,k+s*4+2,width),s0);\n          const uint32_t a3 = hmul2(marlin_pair(weights,e,r1,k+s*4+2,width),s1);\n#pragma unroll\n          for (int j = 0; j < NT; ++j)\n            mma16816(acc[j],a0,a1,a2,a3,Bf[j][g][2*s],Bf[j][g][2*s+1]);\n        }\n      }\n'

def dr_reader_source(source):
    """Derive only the weight reader; routing, reductions and outputs stay V9."""
    start = source.index('__device__ __forceinline__ void prefetch_one(')
    body = source.index('{', start)
    end = source.index('\ntemplate <int NT>', body)
    source = source[:body] + '{ }\n' + source[end:]
    point = source.index('struct PrefetchState {')
    source = source[:point] + dr__HELPERS + source[point:]
    start = source.index('      mbar_wait(&fullb[st], ph);')
    end = source.index('      __syncwarp();', start)
    source = source[:start] + dr__READ + source[end:]
    line = '      if (lane == 0) mbar_arrive(&emptyb[st]);'
    assert source.count(line) == 1
    return source.replace(line, '')

class dr_MarlinV9Module:
    """Keep V9's Python launch ABI; the direct reader uses no staged bank."""

    def __init__(self, module):
        self.module = module
        self.smem_optin = module.max_smem_attr()

    def smem_needed(self, nt, stages, tokens):
        return self.module.smem_needed(nt, 0, tokens)

    def resident_blocks(self, nt, stages, tokens):
        return self.module.resident_blocks(nt, 0, tokens)

    def run(self, *args):
        args = list(args)
        args[-2] = 0
        return self.module.run(*args)

def dr_compile_reader(cuda_source, cpp_source):
    source = dr_reader_source(cuda_source)
    tag = dr_hashlib.sha256(source.encode()).hexdigest()[:12]
    module = dr_load_inline(name=f'muse_marlin_v9_reader_{tag}', cpp_sources=cpp_source, cuda_sources=source, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
    return dr_MarlinV9Module(module)

# Source problems/muse_decode_layer/moe_tune.py sha256=450f13822b82d959714849c88ff87d944e2eaa1b24af99d88b194fdbc2384668
"""Prepare-time MoE tuner for decode-layer routing.

It reconstructs the layer's router distribution from the loaded router, bias, pre-FFN norm and
pooler weights. It never reads a submission input: callers provide only their loaded weights and
shape list, so the resulting shape dispatch remains CUDA-graph safe.
"""
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

def fx_route_summary(fixture: dict) -> dict[str, float]:
    ids = fixture['topk_ids']
    local = ids[ids >= 0]
    counts = fx_torch.bincount(local, minlength=fx_LOCAL_EXPERTS)
    return {'pairs': float(local.numel()), 'distinct': float((counts > 0).sum()), 'max_expert': float(counts.max()), 'mean_nonzero': float(counts[counts > 0].float().mean())}

# Public generator sha256 23e177b87aee70c0d4990515989b95bff2972270cdac8ab967445b21fa2f5163
import math as rg_math
import functools as rg_functools
import torch as rg_torch
rg_HIDDEN = 8192
rg_HEAD_DIM = 64
rg_HQ = 16
rg_HKV = 2
rg_Q_SIZE = 1024
rg_KV_SIZE = 128
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


import torch


import statistics as bridge_statistics
import time as bridge_time


def bridge_graph(fn, data):
    static={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
    fn(static)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out=fn(static)
    return graph,out,static


def bridge_timer(graph, flush, repeats=20):
    samples=[]
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        flush.sum()
        a.record();graph.replay();b.record();b.synchronize()
        samples.append(a.elapsed_time(b)*1000)
    return bridge_statistics.median(samples)



def lab(deadline, weights, host_state=None):
    lines=[]
    try:
        start=bridge_time.monotonic();device=weights['w13'].device
        stock=ma_prepare(weights);bank=stock['banks'][(0,32)]
        joined=[];shared_bank=dict(bank)
        for name in ('w13','w2'):
            q,s=bank[name],bank[name+'_s']
            data=torch.cat((q.view(torch.uint8).flatten(),s.view(torch.uint8).flatten()))
            count=q.nbytes
            shared_bank[name]=data[:count].view(torch.int32).view_as(q)
            shared_bank[name+'_s']=data[count:].view(torch.float8_e4m3fn).view_as(s)
            assert torch.equal(q,shared_bank[name])
            assert torch.equal(s.view(torch.uint8),shared_bank[name+'_s'].view(torch.uint8))
            joined.append(data)
        shared=dict(stock,banks={(0,32):shared_bank})
        gate=bank['w13_gs'].float()*(2.0**-119);down=bank['w2_gs'].float()*(2.0**-119)
        state=dict(w13p=joined[0],w2p=joined[1],mod=dr_compile_reader(v9_CUDA_SRC,v9_CPP_SRC),
                   gs13_gate=gate,gs13_up=gate,gs2=down,pnw=weights['post_norm_w'],
                   local_start=0,grid=torch.cuda.get_device_properties(device).multi_processor_count)
        # Native V9 is retained only as this lab's same-fixture control.
        t,j=torch.arange(512,device=device),torch.arange(8,device=device)
        rm1=torch.cat((t[:,None]*8+j,4096+t[:,None]*8+j),1)
        rm2=torch.arange(4096,device=device).view(256,16)
        native=dict(state,mod=v9__module())
        for name,rm in [('w13',rm1),('w2',rm2)]:
            out=torch.empty((32,rm.shape[0],36864),device=device,dtype=torch.uint8)
            for e in range(32):out[e]=v9__repack(weights[name][e],weights[name+'_sf'][e],rm)
            native[name+'p']=out
        routing=rg_generate_weights(1234,device=str(device),router_seed=5353)
        flush=torch.empty(128<<20,device=device,dtype=torch.uint8)
        sizes=(8,16,24,32,40,48,56,64,80,96,112,128,144,160,176,192,208,224,240,256)
        for m in sizes+(512,753,16384):
            if bridge_time.monotonic()>deadline-20:
                return lines+[f'REJECT incomplete_before_M={m} budget_exhausted']
            ds=[]
            for seed in (9001,16920,24839):
                d=fx_make_fixture(routing,m,seed,mode='real',q_len=8 if m<=256 else 1)
                d['topk_ids']=torch.where(d['topk_ids']>=0,d['topk_ids'],255)
                ds.append({k:d[k] for k in ('hidden_states','topk_ids','topk_weights','config')})
            def direct(d):return v9__v9_call(d,state,return_dense=True)[0]
            def old(d):return v9__v9_call(d,native,return_dense=True)[0]
            funcs=[lambda d:ma_kernel_fn(d,stock)]
            funcs += [old,direct] if m<=256 else [lambda d:ma_kernel_fn(d,shared)]
            graphs=[bridge_graph(fn,ds[0]) for fn in funcs]
            maxdiff=rel=0.0
            for d in ds:
                outputs=[]
                valid=(d['topk_ids']<32).unsqueeze(-1)
                for graph,out,static in graphs:
                    for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(d[key])
                    graph.replay()
                    # Agreement uses stock's norm/combine, outside all timers.
                    y=torch.where(valid,out,0)
                    outputs.append(ma_fused_post_expert_norm_weighted_sum(
                        y,d['topk_weights'],norm_weight=stock['norm_w'],eps=1e-8,weight_first=True))
                torch.cuda.synchronize()
                if m<=256:
                    assert torch.equal(outputs[1],outputs[2]),(m,'native/direct V9 bit equality')
                for output in outputs[1:]:
                    diff=(output.float()-outputs[0].float()).abs()
                    maxdiff=max(maxdiff,float(diff.max()))
                    rel=max(rel,float(diff.norm()/outputs[0].float().norm().clamp_min(1e-20)))
                    assert rel<=.015,(m,rel,maxdiff)
            times=[[] for _ in graphs]
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for repeat in range(20):
                order=list(range(len(graphs)))
                order=order[repeat%len(order):]+order[:repeat%len(order)]
                for i in order:
                    for key in ('hidden_states','topk_ids','topk_weights'):
                        graphs[i][2][key].copy_(ds[repeat%len(ds)][key])
                    flush.sum();a.record();graphs[i][0].replay();b.record();b.synchronize()
                    times[i].append(a.elapsed_time(b)*1000)
            med=[bridge_statistics.median(x) for x in times]
            line=f'M={m} stock_us={med[0]:.3g} '
            if m<=256:line+=f'V9_native_us={med[1]:.3g} direct_us={med[2]:.3g} speedup={med[0]/med[2]:.3g} '
            else:line+=f'shared_Marlin_us={med[1]:.3g} speedup={med[0]/med[1]:.3g} '
            lines.append(line+f'rel={rel:.3g} maxdiff={maxdiff:.3g}')
            for graph,out,static in graphs:graph.reset()
        lines.append(f'DIRECT_READER_SUMMARY PASS shapes=23 fresh=3 cold_rotated20=1 extra_resident_bytes=0 prepare_s={bridge_time.monotonic()-start:.3g}')
    except Exception as exc:
        lines.append('REJECT '+type(exc).__name__+': '+str(exc)[:500])
    return lines
