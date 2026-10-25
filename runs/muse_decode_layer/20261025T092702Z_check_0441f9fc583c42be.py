# okc: problem=muse_decode_layer
"""Best7 (S11 = S10 + patch_gemv_bal + patch_gate_split + patch_glue_late + nf_patch_e200b0ee + patch_vq8_rows + patch_pin + S10 = S8 + patch_front_bf16 + small-M NT1 flags {1: 95, 8: 69, 16: 69, 32: 69} (H200 849e4017, cf1de160) + M128/M256 N_full (H200 c01b1916, 14f942d5, swap 880c3881) + LT/layer-ext compiles in the prepare pool; S8 = S7R + P7: attn_main early release (H200 4d995f13), verify-dense split-K qkv cell at M=128 (b15a3bde), cluster glue t<=32 (1365a290)): receipt-controlled dispatch for the decode-layer stage portfolio.

Muse Spark Open decoder layer (decode / verify), one GPU of TP8/EP8.

Stages (each a swappable function):
  qkv_gate GEMV -> qk_prep (QK-norm, RoPE, KV-cache store, query cast) -> attention (+ norm-gate)
  -> o GEMV -> boundary 1 (post-attn norm, residual gate, pre-FFN norm)
  -> pooler + router GEMV (router as BF16 hi+lo rows, FP32 out)
  -> routing + pre-expert norm -> experts (V9 NVFP4) -> unpooler GEMV
  -> boundary 2 (post-FFN norm, residual gate, next layer's pre-attn norm).

v1: split-K GEMVs (Triton, tensor cores, swap-AB) leave FP32 partial sums that the next kernel
reduces as it loads them, so the split costs no extra kernel. Every Triton kernel launches with
PDL: it lets its successor launch at once, prefetches its own weights into L2, then waits for its
predecessor (all writes come after the wait)."""

from typing import TypedDict


import torch
from torch.utils.cpp_extension import load_inline
import torch.nn.functional as F
import triton
import triton.language as tl



try:
    from triton.tools.tensor_descriptor import TensorDescriptor as _VDSK_TD
except ImportError:
    _VDSK_TD = None
VDSK_SHORTLIST = {(128, 'qkv_gate'): [((128, 64, 64, 8, 4, 1), 3)]}


@triton.jit
def _vdsk_gemm(X, W, P, x_desc, w_desc, M, N, K, S,
               BT: tl.constexpr, BW: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, TMA: tl.constexpr):
    pid = tl.program_id(0)
    ps = tl.program_id(1)
    nw = N // BW
    pw = pid % nw
    pt = pid // nw
    ksteps = K // BK
    k_lo = ps * ksteps // S
    k_hi = (ps + 1) * ksteps // S
    rt = pt * BT + tl.arange(0, BT)
    rw = pw * BW + tl.arange(0, BW)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BT, BW), dtype=tl.float32)
    for kk in tl.range(k_lo, k_hi, num_stages=NS):
        if TMA:
            x = x_desc.load([pt * BT, kk * BK])
            w = w_desc.load([pw * BW, kk * BK])
        else:
            x = tl.load(X + rt[:, None] * K + (kk * BK + rk)[None, :], mask=rt[:, None] < M, other=0.0)
            w = tl.load(W + rw[:, None] * K + (kk * BK + rk)[None, :])
        acc = tl.dot(x, w.T, acc)
    tl.store(P + ps * M * N + rt[:, None] * N + rw[None, :], acc, mask=rt[:, None] < M)


def _vdsk_project(x, w, variant):
    _, cfg, S = variant
    BT, BW, BK, NS, NWARPS, TMA = cfg
    M, K = x.shape
    N = w.shape[0]
    if TMA and _VDSK_TD is None:
        raise RuntimeError('vdsk TMA variant without TensorDescriptor')
    P = torch.empty((S, M, N), device=x.device, dtype=torch.float32)
    xd = _VDSK_TD.from_tensor(x, [BT, BK]) if TMA else None
    wd = _VDSK_TD.from_tensor(w, [BW, BK]) if TMA else None
    _vdsk_gemm[(-(-M // BT) * (N // BW), S)](x, w, P, xd, wd, M, N, K, S, BT=BT, BW=BW, BK=BK, NS=NS, TMA=TMA,
                                             num_warps=NWARPS, num_stages=NS)
    return P, S


def _vdsk_warm(state):
    """Compile every shortlisted config once (background thread at prepare start)."""
    for (t, name), entries in VDSK_SHORTLIST.items():
        w = state[name]
        x = torch.zeros((t, w.shape[1]), device=w.device, dtype=torch.bfloat16)
        for cfg, S in entries:
            try:
                _vdsk_project(x, w, ('vdsk', cfg, S))
            except Exception as exc:
                print(f'VDSK_WARM_REJECT t={t} name={name} cfg={cfg} S={S} {type(exc).__name__}: {str(exc)[:80]}', flush=True)
    torch.cuda.synchronize()


















BEST7_SETTINGS = {'other': ('best6', 'auto', 'auto', 'auto'), (1, 1, 2048): (('best6', (16, 16), (8, 1), (8, 16)), ('attn_main', None), 'nt1_lop3', {}), (1, 1, 0): (('best6', (16, 16), (8, 1), (8, 16)), ('attn_main', None), 'nt1_lop3', {}), (16, 1, 2048): (('best6', (16, 1), (8, 1), (16, 1)), ('attn_main', None), 'nt1_lop3', 'auto'), (16, 1, 0): (('best6', (16, 1), (8, 1), (16, 1)), ('attn_main', None), 'nt1_lop3', 'auto'), (32, 1, 2048): ('best5', ('attn_main', None), 'nt1_lop3', 'auto'), (32, 1, 0): (('best6', (16, 1), (8, 1), (16, 1)), ('attn_main', None), 'nt1_lop3', 'auto'), (1, 8, 2048): (('best6', (16, 4), (4, 1), (8, 16)), ('layer', None), 'nt1_lop3', 'auto'), (1, 8, 0): (('best6', (16, 4), (4, 1), (8, 16)), ('layer', None), 'nt1_lop3', 'auto'), (16, 8, 2048): (('best6', (16, 1), (4, 1), (8, 1)), ('layer', None), 'hh_m128', 'auto'), (16, 8, 0): (('best6', (16, 1), (4, 1), (8, 1)), ('layer', None), 'hh_m128', 'auto'), (32, 8, 2048): (('best6', (16, 1), (4, 1), (16, 1)), ('layer', None), 'mixed_parallel_m256', 'auto'), (32, 8, 0): (('best6', (16, 1), (4, 1), (16, 1)), ('layer', None), 'mixed_parallel_m256', 'auto')}




BEST7_DENSE_CHAIN = {
    'other': False,
    (1, 1, 2048): False, (1, 1, 0): False,
    (16, 1, 2048): False, (16, 1, 0): False,
    (32, 1, 2048): False, (32, 1, 0): False,
    (1, 8, 2048): False, (1, 8, 0): False,
    (16, 8, 2048): False, (16, 8, 0): False,
    (32, 8, 2048): False, (32, 8, 0): False,
}



GLUE_CL_ROWS = {(1, 1, 2048): ('best6', ('cl', 8, 256, 1), ('cl', 8, 256, 0), ('cl', 8, 256, 1)), (1, 1, 0): ('best6', ('cl', 8, 256, 1), ('cl', 8, 256, 0), ('cl', 8, 256, 1)), (1, 8, 2048): ('best6', ('cl', 8, 256, 1), ('cl', 8, 256, 0), ('cl', 8, 256, 1)), (1, 8, 0): ('best6', ('cl', 8, 256, 1), ('cl', 8, 256, 0), ('cl', 8, 256, 1)), (16, 1, 2048): ('best6', ('cl', 8, 256, 1), ('cl', 8, 256, 0), ('cl', 8, 256, 1)), (16, 1, 0): ('best6', ('cl', 8, 256, 1), ('cl', 8, 256, 0), ('cl', 8, 256, 1)), (32, 1, 2048): ('best6', ('cl', 8, 256, 1), ('cl', 8, 256, 0), ('cl', 8, 256, 1)), (32, 1, 0): ('best6', ('cl', 8, 256, 1), ('cl', 8, 256, 0), ('cl', 8, 256, 1))}
for _key, _glue in GLUE_CL_ROWS.items():
    BEST7_SETTINGS[_key] = (_glue,) + tuple(BEST7_SETTINGS[_key][1:])
GLUE_CL_CUDA = '// Cluster glue for the decode layer (P7): one thread-block cluster of C CTAs per token. Each CTA reads\n// only its 1/C slice of the row (S6\'s Triton glue has every CTA read the whole row, so at S split\n// partials one SM\'s L2 bandwidth bounds it), and the row\'s norm partials meet through distributed\n// shared memory: each CTA pushes its partial sums into every CTA of the cluster, one cluster barrier,\n// then every CTA finishes its slice from the same C partials in the same order.\n//\n// boundary_cl: stream\' = alpha * stream + beta * rmsnorm(branch [* postw]); out = rmsnorm(stream\') * prew.\n//   ROUNDS 2: exact two-pass form (|stream\'|^2 summed after the branch norm is known).\n//   ROUNDS 1: one barrier; |stream\'|^2 = sum (a s)^2 + 2 r sum (a s)(b br) + r^2 sum (b br)^2, r = branch inv.\n// route_cl: x = rmsnorm(pooler) * prew (BF16); logits = sum of router hi + lo rows; sigmoid(logit/sqrt(32)),\n//   top-8 of sigmoid + bias (ties to the lower expert), weights = picked sigmoids / their sum, local ids.\n// Both release dependents at entry, load their weight vectors, then griddepcontrol.wait before any\n// read of the producer\'s output; launched with programmatic stream serialization.\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cooperative_groups.h>\n#include <stdint.h>\n#include <stdexcept>\n\nnamespace cg = cooperative_groups;\n\nnamespace glue_cl {\n\n__device__ __forceinline__ void pdl_release() {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void pdl_wait() {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n#endif\n}\n\n__device__ __forceinline__ float4 ld4(const void* base, int f32, long long off) {\n  if (f32) return *reinterpret_cast<const float4*>(static_cast<const float*>(base) + off);\n  const uint2 v = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(base) + off);\n  return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),\n                     __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));\n}\n\n// Sum S split partials at one float4 position, all loads in flight (groups of 8).\n__device__ __forceinline__ float4 sum_splits(const void* P, int f32, int S, long long ps, long long off) {\n  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);\n  for (int s0 = 0; s0 < S; s0 += 8) {\n    float4 z[8];\n#pragma unroll\n    for (int u = 0; u < 8; ++u) z[u] = (s0 + u < S) ? ld4(P, f32, (s0 + u) * ps + off) : make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int u = 0; u < 8; ++u) {\n      a.x += z[u].x; a.y += z[u].y; a.z += z[u].z; a.w += z[u].w;\n    }\n  }\n  return a;\n}\n\ntemplate <int K, int NT>\n__device__ __forceinline__ void block_sum(float (&v)[K], float* red) {\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n#pragma unroll\n  for (int k = 0; k < K; ++k)\n#pragma unroll\n    for (int d = 16; d; d >>= 1) v[k] += __shfl_xor_sync(0xffffffffu, v[k], d);\n  if (lane == 0)\n#pragma unroll\n    for (int k = 0; k < K; ++k) red[warp * K + k] = v[k];\n  __syncthreads();\n#pragma unroll\n  for (int k = 0; k < K; ++k) {\n    float s = 0.f;\n#pragma unroll\n    for (int w = 0; w < NT / 32; ++w) s += red[w * K + k];\n    v[k] = s;\n  }\n}\n\n// Push this CTA\'s K partials into slot [rank] of every CTA\'s `slots` [C][K], then one cluster barrier;\n// returns the C partials summed in rank order (identical in every CTA).\ntemplate <int K, int C>\n__device__ __forceinline__ void cluster_sum(cg::cluster_group& cl, float (&v)[K], float* slots) {\n  const int rank = (int)cl.block_rank();\n  if (threadIdx.x < C) {\n    float* dst = cl.map_shared_rank(slots, (int)threadIdx.x);\n#pragma unroll\n    for (int k = 0; k < K; ++k) dst[rank * K + k] = v[k];\n  }\n  cl.sync();\n#pragma unroll\n  for (int k = 0; k < K; ++k) {\n    float s = 0.f;\n#pragma unroll\n    for (int r = 0; r < C; ++r) s += slots[r * K + k];\n    v[k] = s;\n  }\n}\n\nconstexpr int HID = 8192;\n\n// grid (C, T), cluster (C, 1, 1), NT threads; each thread owns HID / (C * NT) consecutive float4 groups.\ntemplate <int C, int NT, int ROUNDS>\n__global__ void __launch_bounds__(NT) boundary_cl(const void* __restrict__ P, int pf32, int S, long long ps, long long rs,\n                                                  int NB, const float* __restrict__ stream, float* __restrict__ sout,\n                                                  __nv_bfloat16* __restrict__ out, const float* __restrict__ postw,\n                                                  const float* __restrict__ alpha, const float* __restrict__ beta,\n                                                  const float* __restrict__ prew) {\n  constexpr int G = HID / (4 * C * NT);  // float4 groups per thread\n  static_assert(G >= 1 && HID % (4 * C * NT) == 0, "slice");\n  __shared__ float red[(NT / 32) * 4];\n  __shared__ float slots[2][C * 4];\n  pdl_release();\n  cg::cluster_group cl = cg::this_cluster();\n  const int rank = (int)cl.block_rank(), t = blockIdx.y, tid = threadIdx.x;\n  float4 a[G], b[G], w[G], q[G];\n#pragma unroll\n  for (int g = 0; g < G; ++g) {\n    const int j = (rank * NT * G + g * NT + tid) * 4;\n    a[g] = *reinterpret_cast<const float4*>(alpha + j);\n    b[g] = *reinterpret_cast<const float4*>(beta + j);\n    w[g] = *reinterpret_cast<const float4*>(prew + j);\n    q[g] = postw ? *reinterpret_cast<const float4*>(postw + j) : make_float4(1.f, 1.f, 1.f, 1.f);\n  }\n  pdl_wait();\n  float4 s[G], br[G];\n#pragma unroll\n  for (int g = 0; g < G; ++g) {\n    const int j = (rank * NT * G + g * NT + tid) * 4;\n    s[g] = *reinterpret_cast<const float4*>(stream + (long long)t * HID + j);\n    br[g] = j < NB ? sum_splits(P, pf32, S, ps, (long long)t * rs + j) : make_float4(0.f, 0.f, 0.f, 0.f);\n    br[g].x *= q[g].x; br[g].y *= q[g].y; br[g].z *= q[g].z; br[g].w *= q[g].w;\n  }\n  float inv_br, inv;\n  if (ROUNDS == 1) {\n    float v[4] = {0.f, 0.f, 0.f, 0.f};  // sum br^2, (a s)^2, (a s)(b br), (b br)^2\n#pragma unroll\n    for (int g = 0; g < G; ++g) {\n      const float bb[4] = {br[g].x, br[g].y, br[g].z, br[g].w}, ss[4] = {s[g].x, s[g].y, s[g].z, s[g].w};\n      const float aa[4] = {a[g].x, a[g].y, a[g].z, a[g].w}, be[4] = {b[g].x, b[g].y, b[g].z, b[g].w};\n#pragma unroll\n      for (int k = 0; k < 4; ++k) {\n        const float x = aa[k] * ss[k], y = be[k] * bb[k];\n        v[0] += bb[k] * bb[k]; v[1] += x * x; v[2] += x * y; v[3] += y * y;\n      }\n    }\n    block_sum<4, NT>(v, red);\n    cluster_sum<4, C>(cl, v, slots[0]);\n    inv_br = rsqrtf(v[0] / HID + 1e-8f);\n    inv = rsqrtf((v[1] + 2.f * inv_br * v[2] + inv_br * inv_br * v[3]) / HID + 1e-5f);\n#pragma unroll\n    for (int g = 0; g < G; ++g) {\n      s[g].x = a[g].x * s[g].x + b[g].x * (br[g].x * inv_br); s[g].y = a[g].y * s[g].y + b[g].y * (br[g].y * inv_br);\n      s[g].z = a[g].z * s[g].z + b[g].z * (br[g].z * inv_br); s[g].w = a[g].w * s[g].w + b[g].w * (br[g].w * inv_br);\n    }\n  } else {\n    float v[1] = {0.f};\n#pragma unroll\n    for (int g = 0; g < G; ++g) v[0] += br[g].x * br[g].x + br[g].y * br[g].y + br[g].z * br[g].z + br[g].w * br[g].w;\n    block_sum<1, NT>(v, red);\n    cluster_sum<1, C>(cl, v, slots[0]);\n    inv_br = rsqrtf(v[0] / HID + 1e-8f);\n    float u[1] = {0.f};\n#pragma unroll\n    for (int g = 0; g < G; ++g) {\n      s[g].x = a[g].x * s[g].x + b[g].x * (br[g].x * inv_br); s[g].y = a[g].y * s[g].y + b[g].y * (br[g].y * inv_br);\n      s[g].z = a[g].z * s[g].z + b[g].z * (br[g].z * inv_br); s[g].w = a[g].w * s[g].w + b[g].w * (br[g].w * inv_br);\n      u[0] += s[g].x * s[g].x + s[g].y * s[g].y + s[g].z * s[g].z + s[g].w * s[g].w;\n    }\n    __syncthreads();  // red reuse\n    block_sum<1, NT>(u, red);\n    cluster_sum<1, C>(cl, u, slots[1]);\n    inv = rsqrtf(u[0] / HID + 1e-5f);\n  }\n#pragma unroll\n  for (int g = 0; g < G; ++g) {\n    const int j = (rank * NT * G + g * NT + tid) * 4;\n    *reinterpret_cast<float4*>(sout + (long long)t * HID + j) = s[g];\n    __nv_bfloat162 o0 = __floats2bfloat162_rn(s[g].x * inv * w[g].x, s[g].y * inv * w[g].y);\n    __nv_bfloat162 o1 = __floats2bfloat162_rn(s[g].z * inv * w[g].z, s[g].w * inv * w[g].w);\n    uint2 ov = make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));\n    *reinterpret_cast<uint2*>(out + (long long)t * HID + j) = ov;\n  }\n}\n\n// MoERouteMeta v1 (problems/muse_decode_layer/dev/nt1_branches/route_meta.h): per-call routing description\n// for the small-M MoE (tokens <= 32), written by route_cl\'s last-arriving cluster.\nstruct alignas(128) MoERouteMeta {\n  int32_t version, tokens, local_start, pairs;\n  int32_t count[32];\n  int32_t offset[33];\n  int32_t blocks8[32];\n  int32_t units13[33], units2[33];\n  int32_t token_local[32];\n  int32_t pair_of[256];  // sorted local position -> token*8+slot, -1 unused\n  int32_t pos_of[256];   // token*8+slot -> sorted local position, -1 remote\n};\nstatic_assert(sizeof(MoERouteMeta) == 2944, "MoERouteMeta v1");\n\n// nt1_branches/route_meta.cu.inc\'s make_route_meta body for one 256-thread CTA, reading local ids (-1 remote).\n__device__ __forceinline__ void write_route_meta(const int* __restrict__ lids, int M, int start, MoERouteMeta* dst) {\n  __shared__ int n[32], wc[8][32], off[32];\n  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;\n  if (t < 32) n[t] = 0;\n  dst->pair_of[t] = -1;\n  dst->pos_of[t] = -1;\n  if (t < 32) dst->token_local[t] = 0;\n  __syncthreads();\n  const int e = t < M * 8 ? __ldcg(lids + t) : -1;\n  if (e >= 0) atomicAdd(n + e, 1);\n  __syncthreads();\n  if (warp == 0) {\n    const int count = n[lane], nb = (count + 7) / 8;\n    int p = count, u1 = nb * 512, u2 = nb * 256;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, p, d), b = __shfl_up_sync(~0u, u1, d), c = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { p += a; u1 += b; u2 += c; }\n    }\n    off[lane] = p - count;\n    dst->count[lane] = count; dst->offset[lane] = p - count; dst->blocks8[lane] = nb;\n    dst->units13[lane + 1] = u1; dst->units2[lane + 1] = u2;\n    if (lane == 31) { dst->pairs = p; dst->offset[32] = p; }\n    if (lane == 0) { dst->version = 1; dst->tokens = M; dst->local_start = start; dst->units13[0] = dst->units2[0] = 0; }\n  }\n  wc[warp][lane] = 0;\n  __syncthreads();\n  const unsigned mk = __match_any_sync(~0u, e);\n  const int rnk = __popc(mk & ((1u << lane) - 1));\n  if (e >= 0 && rnk == 0) wc[warp][e] = __popc(mk);\n  __syncthreads();\n  if (t < 32) {\n    int sum = off[t];\n    for (int w = 0; w < 8; ++w) { const int v = wc[w][t]; wc[w][t] = sum; sum += v; }\n  }\n  __syncthreads();\n  if (e >= 0) { const int pos = wc[warp][e] + rnk; dst->pair_of[pos] = t; dst->pos_of[t] = pos; }\n  if (t < M) {\n    int cnt = 0;\n#pragma unroll\n    for (int q = 0; q < 8; ++q) cnt += __ldcg(lids + t * 8 + q) >= 0;\n    dst->token_local[t] = cnt;\n  }\n}\n\nconstexpr int DP = 4096;   // pooler width\nconstexpr int NE = 256;    // experts\n\n// Order-preserving integer image of a float (S6\'s key: larger float -> larger key; never 0 for a non-NaN).\n__device__ __forceinline__ uint32_t route_ord(float biased) {\n  const uint32_t bits = __float_as_uint(biased);\n  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);\n}\n\n// grid (C, T), cluster (C, 1, 1), NT threads. Pooler: HID-style float4 slices of DP / C columns.\n// Logits: CTA rank owns experts [rank * NE / C, ...); its sigmoids and keys go to rank 0\'s smem.\ntemplate <int C, int NT>\n__global__ void __launch_bounds__(NT) route_cl(const void* __restrict__ P, int pf32, int S, long long ps, long long rs,\n                                               const float* __restrict__ bias, const float* __restrict__ prew,\n                                               const int* __restrict__ lstart, __nv_bfloat16* __restrict__ X,\n                                               int* __restrict__ ids, float* __restrict__ wts, int* __restrict__ lids,\n                                               MoERouteMeta* __restrict__ meta, unsigned* __restrict__ arrive) {\n  constexpr int GP = DP / (4 * C * NT);   // pooler float4 groups per thread (0: some threads idle)\n  constexpr int EP = NE / C;              // experts per CTA\n  static_assert(DP % (4 * C) == 0 && NE % C == 0 && EP <= NT, "slice");\n  constexpr int QT = DP / (4 * C);        // pooler float4 groups per CTA\n  __shared__ float red[(NT / 32)];\n  __shared__ float slots[C];\n  __shared__ float score_all[NE];\n  __shared__ uint32_t ord_all[NE];\n  pdl_release();\n  cg::cluster_group cl = cg::this_cluster();\n  const int rank = (int)cl.block_rank(), t = blockIdx.y, tid = threadIdx.x;\n  constexpr int GQ = (QT + NT - 1) / NT;\n  float4 w[GQ];\n#pragma unroll\n  for (int g = 0; g < GQ; ++g) {\n    const int q = g * NT + tid;\n    w[g] = q < QT ? *reinterpret_cast<const float4*>(prew + (rank * QT + q) * 4) : make_float4(0.f, 0.f, 0.f, 0.f);\n  }\n  const int e = rank * EP + tid;\n  const float bs = tid < EP ? bias[e] : 0.f;\n  const int start = *lstart;\n  pdl_wait();\n  float4 p[GQ];\n  float v[1] = {0.f};\n#pragma unroll\n  for (int g = 0; g < GQ; ++g) {\n    const int q = g * NT + tid;\n    p[g] = q < QT ? sum_splits(P, pf32, S, ps, (long long)t * rs + (rank * QT + q) * 4) : make_float4(0.f, 0.f, 0.f, 0.f);\n    v[0] += p[g].x * p[g].x + p[g].y * p[g].y + p[g].z * p[g].z + p[g].w * p[g].w;\n  }\n  if (tid < EP) {  // logit: per split hi + lo, accumulated over splits (S6 order)\n    const long long base = (long long)t * rs + DP + e;\n    float logit = 0.f;\n    for (int s0 = 0; s0 < S; s0 += 8) {\n      float hi[8], lo[8];\n#pragma unroll\n      for (int u = 0; u < 8; ++u) {\n        hi[u] = lo[u] = 0.f;\n        if (s0 + u < S) {\n          const long long o = (s0 + u) * ps + base;\n          if (pf32) { hi[u] = static_cast<const float*>(P)[o]; lo[u] = static_cast<const float*>(P)[o + NE]; }\n          else { hi[u] = __bfloat162float(static_cast<const __nv_bfloat16*>(P)[o]); lo[u] = __bfloat162float(static_cast<const __nv_bfloat16*>(P)[o + NE]); }\n        }\n      }\n#pragma unroll\n      for (int u = 0; u < 8; ++u) logit += hi[u] + lo[u];\n    }\n    const float sc = 1.f / (1.f + __expf(-logit * 0.1767766952966369f));\n    float* rs_ = cl.map_shared_rank(score_all, 0);\n    uint32_t* ro = cl.map_shared_rank(ord_all, 0);\n    rs_[e] = sc;\n    ro[e] = route_ord(sc + bs);\n  }\n  block_sum<1, NT>(v, red);\n  cluster_sum<1, C>(cl, v, slots);   // its barrier also publishes the scores and keys to rank 0\n  const float inv = rsqrtf(v[0] / DP + 1e-5f);\n#pragma unroll\n  for (int g = 0; g < GQ; ++g) {\n    const int q = g * NT + tid;\n    if (q < QT) {\n      __nv_bfloat162 o0 = __floats2bfloat162_rn(p[g].x * inv * w[g].x, p[g].y * inv * w[g].y);\n      __nv_bfloat162 o1 = __floats2bfloat162_rn(p[g].z * inv * w[g].z, p[g].w * inv * w[g].w);\n      *reinterpret_cast<uint2*>(X + (long long)t * DP + (rank * QT + q) * 4) =\n          make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));\n    }\n  }\n  if (rank == 0 && tid < 32) {  // top-8 over 256: 8 per lane; per round redux max of the key, ties to min id\n    const int lane = tid;\n    uint32_t k[8];\n#pragma unroll\n    for (int i = 0; i < 8; ++i) k[i] = ord_all[lane * 8 + i];\n    int pick[8];\n    float sc[8], tot = 0.f;\n#pragma unroll\n    for (int r = 0; r < 8; ++r) {\n      uint32_t bo = k[0];\n      int bi = 0;\n#pragma unroll\n      for (int i = 1; i < 8; ++i)\n        if (k[i] > bo) { bo = k[i]; bi = i; }\n      const uint32_t top = __reduce_max_sync(0xffffffffu, bo);\n      const int ex = (int)__reduce_min_sync(0xffffffffu, bo == top ? (uint32_t)(lane * 8 + bi) : 0xffffffffu);\n#pragma unroll\n      for (int i = 0; i < 8; ++i)\n        if (lane * 8 + i == ex) k[i] = 0u;\n      pick[r] = ex;\n      sc[r] = score_all[ex];\n    }\n#pragma unroll\n    for (int r = 0; r < 8; ++r) tot += sc[r];\n    if (lane < 8) {\n      int ex = pick[0];\n      float s = sc[0];\n#pragma unroll\n      for (int r = 1; r < 8; ++r) if (lane == r) { ex = pick[r]; s = sc[r]; }\n      ids[t * 8 + lane] = ex;\n      wts[t * 8 + lane] = s / (tot + 1e-15f);\n      const int local = (ex - start + 256) % 256;\n      lids[t * 8 + lane] = local < 32 ? local : -1;\n    }\n  }\n  if (meta != nullptr && rank == 0) {  // last-arriving token\'s rank-0 CTA writes MoERouteMeta (no CTA waits)\n    __shared__ int last;\n    __syncthreads();\n    if (tid == 0) {\n      __threadfence();\n      last = atomicAdd(arrive, 1u) == (unsigned)(gridDim.y - 1);\n      if (last) __threadfence();\n    }\n    __syncthreads();\n    if (last) {\n      write_route_meta(lids, (int)gridDim.y, start, meta);\n      if (tid == 0) *arrive = 0u;\n    }\n  }\n}\n\ntemplate <typename K>\nstatic void set_attrs(K kernel) {\n  static bool done = false;\n  if (!done) {\n    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n    done = true;\n  }\n}\n\ntemplate <typename K, typename... A>\nstatic cudaError_t launch(K kernel, int C, int NT, int T, cudaStream_t s, A... args) {\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(C, T, 1);\n  cfg.blockDim = dim3(NT, 1, 1);\n  cfg.stream = s;\n  cudaLaunchAttribute at[2];\n  at[0].id = cudaLaunchAttributeClusterDimension;\n  at[0].val.clusterDim.x = C;\n  at[0].val.clusterDim.y = 1;\n  at[0].val.clusterDim.z = 1;\n  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n  at[1].val.programmaticStreamSerializationAllowed = 1;\n  cfg.attrs = at;\n  cfg.numAttrs = 2;\n  return cudaLaunchKernelEx(&cfg, kernel, args...);\n}\n\n}  // namespace glue_cl\n\n#define GLUE_B(C_, NT_, R_)                                                                                   \\\n  if (C == C_ && NT == NT_ && rounds == R_) {                                                                \\\n    auto k = glue_cl::boundary_cl<C_, NT_, R_>;                                                              \\\n    glue_cl::set_attrs(k);                                                                                   \\\n    e = glue_cl::launch(k, C, NT, (int)T, s, (const void*)P, (int)pf32, (int)S, (long long)ps, (long long)rs, \\\n                        (int)NB, (const float*)stream, (float*)sout, (__nv_bfloat16*)out, (const float*)postw,   \\\n                        (const float*)alpha, (const float*)beta, (const float*)prew);                          \\\n  }\n\nvoid boundary_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t NB, int64_t stream, int64_t sout,\n                 int64_t out, int64_t postw, int64_t alpha, int64_t beta, int64_t prew, int64_t T, int64_t C, int64_t NT,\n                 int64_t rounds, int64_t stream_handle) {\n  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_handle);\n  cudaError_t e = cudaErrorInvalidValue;\n  GLUE_B(16, 128, 1) GLUE_B(16, 128, 2) GLUE_B(8, 256, 1) GLUE_B(8, 256, 2) GLUE_B(8, 128, 1) GLUE_B(8, 128, 2)\n  GLUE_B(4, 256, 1) GLUE_B(4, 256, 2) GLUE_B(4, 512, 1) GLUE_B(4, 512, 2) GLUE_B(2, 512, 1) GLUE_B(2, 512, 2)\n  GLUE_B(2, 256, 1) GLUE_B(2, 256, 2)\n  if (e != cudaSuccess) {\n    cudaGetLastError();  // a rejected launch (e.g. cluster size) must not poison later launches\n    throw std::runtime_error(cudaGetErrorString(e));\n  }\n}\n\n#define GLUE_R(C_, NT_)                                                                                        \\\n  if (C == C_ && NT == NT_) {                                                                                 \\\n    auto k = glue_cl::route_cl<C_, NT_>;                                                                      \\\n    glue_cl::set_attrs(k);                                                                                    \\\n    e = glue_cl::launch(k, C, NT, (int)T, s, (const void*)P, (int)pf32, (int)S, (long long)ps, (long long)rs, \\\n                        (const float*)bias, (const float*)prew, (const int*)lstart, (__nv_bfloat16*)X, (int*)ids, \\\n                        (float*)wts, (int*)lids, (glue_cl::MoERouteMeta*)meta, (unsigned*)arrive);             \\\n  }\n\nvoid route_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t bias, int64_t prew, int64_t lstart,\n              int64_t X, int64_t ids, int64_t wts, int64_t lids, int64_t meta, int64_t arrive, int64_t T, int64_t C,\n              int64_t NT, int64_t stream_handle) {\n  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_handle);\n  cudaError_t e = cudaErrorInvalidValue;\n  if (meta && (NT != 256 || T > 32)) throw std::runtime_error("route_cl metadata: NT 256 and T <= 32");\n  GLUE_R(16, 128) GLUE_R(8, 128) GLUE_R(8, 256) GLUE_R(4, 256) GLUE_R(2, 256)\n  if (e != cudaSuccess) {\n    cudaGetLastError();  // a rejected launch (e.g. cluster size) must not poison later launches\n    throw std::runtime_error(cudaGetErrorString(e));\n  }\n}\n'
GLUE_CL_CPP = '\n#include <cstdint>\nvoid boundary_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t NB, int64_t stream, int64_t sout,\n                 int64_t out, int64_t postw, int64_t alpha, int64_t beta, int64_t prew, int64_t T, int64_t C, int64_t NT,\n                 int64_t rounds, int64_t stream_handle);\nvoid route_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t bias, int64_t prew, int64_t lstart,\n              int64_t X, int64_t ids, int64_t wts, int64_t lids, int64_t meta, int64_t arrive, int64_t T, int64_t C,\n              int64_t NT, int64_t stream_handle);\n'
_GLUE_CL_EXT = None


def _glue_cl_ext():
    global _GLUE_CL_EXT
    if _GLUE_CL_EXT is None:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        _GLUE_CL_EXT = load_inline(
            name=f'muse_decode_glue_cl_{arch}', cpp_sources=GLUE_CL_CPP, cuda_sources=GLUE_CL_CUDA,
            functions=['boundary_cl', 'route_cl'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'], verbose=False)
    return _GLUE_CL_EXT


def _glue_cl_boundary(branch, stream, alpha, beta, prew, postw, geo):
    p, s = branch
    t = stream.shape[0]
    _, c, nt, rounds = geo
    out = torch.empty((t, 8192), device=stream.device, dtype=torch.bfloat16)
    sout = torch.empty_like(stream)
    _glue_cl_ext().boundary_cl(p.data_ptr(), int(p.dtype == torch.float32), s, p.stride(0), p.stride(1), p.shape[2],
                               stream.data_ptr(), sout.data_ptr(), out.data_ptr(),
                               postw.data_ptr() if postw is not None else 0, alpha.data_ptr(), beta.data_ptr(),
                               prew.data_ptr(), t, c, nt, rounds, torch.cuda.current_stream(stream.device).cuda_stream)
    return out, sout


def _glue_cl_route(pr, state, data, geo):
    p, s = pr
    t = p.shape[1]
    dev = p.device
    _, c, nt, _ = geo
    x = torch.empty((t, 4096), device=dev, dtype=torch.bfloat16)
    ids = torch.empty((t, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((t, 8), device=dev, dtype=torch.float32)
    lids = torch.empty_like(ids)
    _glue_cl_ext().route_cl(p.data_ptr(), int(p.dtype == torch.float32), s, p.stride(0), p.stride(1),
                            state['router_bias'].data_ptr(), state['pre_expert_w'].data_ptr(),
                            data['local_start'].data_ptr(), x.data_ptr(), ids.data_ptr(), wts.data_ptr(),
                            lids.data_ptr(), 0, 0, t, c, nt, torch.cuda.current_stream(dev).cuda_stream)
    return x, ids, wts, lids


class Weights(TypedDict):
    qkv_gate: torch.Tensor            
    o: torch.Tensor                   
    attn_residual_gate: torch.Tensor  
    pre_ffn_norm: torch.Tensor        
    router: torch.Tensor              
    router_bias: torch.Tensor         
    pooler: torch.Tensor              
    pre_expert_norm: torch.Tensor     
    w13: torch.Tensor                 
    w13_sf: torch.Tensor              
    w13_gs: torch.Tensor              
    w2: torch.Tensor                  
    w2_sf: torch.Tensor               
    w2_gs: torch.Tensor               
    post_expert_norm: torch.Tensor    
    unpooler: torch.Tensor            
    post_ffn_norm: torch.Tensor       
    ffn_residual_gate: torch.Tensor   
    next_pre_attn_norm: torch.Tensor  


HIDDEN = 8192
D_EXPERT = 4096
HQ, HKV, HD = 16, 2, 64
Q_SIZE, KV_SIZE = HQ * HD, HKV * HD
QKV3_SIZE = 1280  
QKV_SIZE = 2 * Q_SIZE + 2 * KV_SIZE
GATE_OFF = Q_SIZE + 2 * KV_SIZE
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
UNPOOL_SLICE = HIDDEN // 8
SLICED_MAX_TOKENS = 8
RMS_EPS = 1e-5
POST_EPS = 1e-8
ROUTER_MULT = 0.1767766952966369
ROPE_THETA = 500_000.0
FP8 = torch.float8_e4m3fn
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)

IS_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)

ATTN_BACKEND = "fa3" if IS_HOPPER else "triton"
GEMV_MAX_T = 64      
PDL = True           
PREFETCH = True      


@triton.jit
def _l2_prefetch(ptrs):
    return tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [ptrs],
                                     dtype=tl.int32, is_pure=False, pack=1)


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
def _gemv_kernel(X, W, P, T, x_stride,
                 N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr, TB: tl.constexpr, PREFETCH: tl.constexpr, USE_PDL: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    _pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    k0 = ps * KS
    if PREFETCH:
        lines = tl.arange(0, KS // 64)
        _l2_prefetch(W + n[:, None] * K + k0 + lines[None, :] * 64)
    _pdl_wait(USE_PDL)
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for k in range(k0, k0 + KS, BK):
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)


@triton.jit
def _attn_sumcast_kernel(P, Y, PS, RS, N: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr, USE_PDL: tl.constexpr):
    """BF16 qkv [1, T, N]: the S FP32 split partials summed in split order (torch's order), one rounding."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    _pdl_start(USE_PDL)
    _pdl_wait(USE_PDL)
    c = b * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for s in tl.static_range(S):
        acc += tl.load(P + s * PS + t * RS + c, mask=c < N, other=0.0)
    tl.store(Y + t * N + c, acc.to(tl.bfloat16), mask=c < N)


def _attn_sumcast(partial, splits):
    """(BF16 [1, T, N], 1) from FP32 split partials [S, T, N] (unit stride along N)."""
    _, T, N = partial.shape
    y = torch.empty((1, T, N), device=partial.device, dtype=torch.bfloat16)
    _launch(_attn_sumcast_kernel, (T, triton.cdiv(N, 512)), partial, y, partial.stride(0), partial.stride(1), N=N,
            S=splits, BLOCK=512, num_warps=4, use_pdl=True)
    return y, 1


@triton.jit
def _gb_kernel(X, W, P, T, x_stride, N, K, S, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr,
               NS: tl.constexpr, USE_PDL: tl.constexpr):
    """_gemv_kernel with runtime k-block ranges (balanced split-K; P [S, T, N] FP32)."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    _pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    kb_total = K // BK
    kb0 = ps * kb_total // S
    kb1 = (ps + 1) * kb_total // S
    _pdl_wait(USE_PDL)
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for kb in tl.range(kb0, kb1, num_stages=NS):
        k = kb * BK
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)


def _gb_gemv(x, w, choice):
    if x.shape[0] in (8,16,32) and tuple(w.shape)==(4608,8192):
        return gi_run(x,w,6,(128,2,0,0)),6
    _, BN, S, NS = choice
    T, K = x.shape
    N = w.shape[0]
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    _launch(_gb_kernel, (N // BN, S), x, w, P, T, x.stride(0), N, K, S, BN=BN, BK=128,
            TB=max(16, triton.next_power_of_2(T)), NS=NS, num_warps=4)
    return P, S



GB_ROWS = {(1, 'qkv_gate'): ('bal', 64, 7, 4), (8, 'qkv_gate'): ('bal', 64, 7, 4), (16, 'qkv_gate'): ('bal', 64, 7, 4), (32, 'qkv_gate'): ('bal', 64, 7, 4), (1, 'pool_router'): ('bal', 64, 6, 4), (8, 'pool_router'): ('bal', 64, 6, 4), (16, 'pool_router'): ('bal', 64, 6, 4), (32, 'pool_router'): ('bal', 64, 6, 4), (16, 'unpooler'): ('bal', 64, 2, 4), (32, 'unpooler'): ('bal', 64, 2, 4)}


@triton.jit
def _gate_split_kernel(X, W, P, CNT, Y, G, T, x_stride, K, S, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       TB: tl.constexpr, NS: tl.constexpr):
    """Gate rows of qkv_gate, split-K, all before griddepcontrol.wait (reads only the layer input and W); the
    last split CTA of a tile waits (attention complete), sums the S partials in split order and writes
    attn_out * sigmoid(gate) as BF16 [T, N]."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    n = pn * BN + tl.arange(0, BN)
    kb_total = K // BK
    kb0 = ps * kb_total // S
    kb1 = (ps + 1) * kb_total // S
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for kb in tl.range(kb0, kb1, num_stages=NS):
        k = kb * BK
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    mask = t[None, :] < T
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=mask)
    tl.debug_barrier()
    arrived = tl.atomic_add(CNT + pn, 1, sem='acq_rel', scope='gpu')
    if arrived == S - 1:
        tl.extra.cuda.gdc_wait()
        tot = tl.zeros((BN, TB), tl.float32)
        for s in range(S):
            tot += tl.load(P + s * (T * N) + t[None, :] * N + n[:, None], mask=mask, other=0.0, cache_modifier='.cg')
        y = tl.load(Y + t[None, :] * N + n[:, None], mask=mask, other=0.0).to(tl.float32)
        tl.store(G + t[None, :] * N + n[:, None], (y * tl.sigmoid(tot)).to(tl.bfloat16), mask=mask)
        tl.atomic_xchg(CNT + pn, 0, sem='relaxed', scope='gpu')


GATE_SPLIT_CFG = (64, 4, 4)  


GATE_SPLIT_ROWS = {(1, True), (1, False), (16, True), (32, False)}
GATE_SPLIT_QKV3 = ('bal', 64, 10, 4)  
_GATE_SPLIT_BUF = {}


def _gate_split_buffers(device, tokens):
    """Prepare-time (or first eager call) partials [S, T, 1024] and per-tile counters: never per-call memory."""
    buf = _GATE_SPLIT_BUF.get(tokens)
    if buf is None:
        BN, S, _ = GATE_SPLIT_CFG
        buf = _GATE_SPLIT_BUF[tokens] = (torch.empty((S, tokens, Q_SIZE), device=device, dtype=torch.float32),
                                         torch.zeros(Q_SIZE // BN, device=device, dtype=torch.int32))
    return buf


def _gate_split_prepare(state, device):
    state['qkv3'] = state['qkv_gate'][:QKV3_SIZE]
    state['gate_rows'] = state['qkv_gate'][QKV3_SIZE:]
    for t in (1, 8, 16, 32):
        _GEMV_CHOICES[(t, QKV3_SIZE, HIDDEN)] = GATE_SPLIT_QKV3
        _gate_split_buffers(device, t)


def _gate_split_apply(x, y, state):
    """attn_out [T, 1024] (ungated) -> attn_out * sigmoid(x @ gate_rows.T), one PDL kernel."""
    T, K = x.shape
    BN, S, NS = GATE_SPLIT_CFG
    P, cnt = _gate_split_buffers(x.device, T)
    g = torch.empty((T, Q_SIZE), device=x.device, dtype=torch.bfloat16)
    w = state['gate_rows']
    _gate_split_kernel[(Q_SIZE // BN, S)](x, w, P, cnt, y, g, T, x.stride(0), K, S, N=Q_SIZE, BN=BN, BK=128,
                                          TB=max(16, triton.next_power_of_2(T)), NS=NS, num_warps=4, launch_pdl=True)
    return g


def _gate_split_on(data, state, plan):
    cfg = data['config']
    if cfg['q_len'] != 1 or plan.get('dense_chain', False) or (cfg['batch'], bool(cfg['window'])) not in GATE_SPLIT_ROWS:
        return False
    choice = plan['attention']
    if choice == 'auto':
        choice = state['attention_plan'].get((cfg['batch'], cfg['q_len'], bool(cfg['window'])), ('layer', None))
    return choice[0] in ('attn_main', 'layer')



GEMV_CFG = {
    (QKV_SIZE, HIDDEN): (64, 1024, 128),
    (QKV3_SIZE, HIDDEN): (64, 1024, 128),
    (HIDDEN, Q_SIZE): (64, 512, 128),
    (D_EXPERT + 2 * N_EXPERTS, HIDDEN): (64, 2048, 128),
    (UNPOOL_SLICE, D_EXPERT): (64, 512, 128),
    (HIDDEN, D_EXPERT): (64, 2048, 128),
}


def gemv(x, w, out_dtype=torch.bfloat16):
    """x [T, K] bf16, w [N, K] bf16 -> (P [S, T, N], S): FP32 split-K partials, or for T above
    GEMV_MAX_T one cuBLAS product (S = 1, out_dtype)."""
    T, K = x.shape
    N = w.shape[0]
    if T > GEMV_MAX_T:
        if out_dtype == torch.float32:
            out = torch.mm(x, w.t(), out_dtype=torch.float32)
        else:
            out = F.linear(x, w)
        return out.unsqueeze(0), 1
    choice = _GEMV_CHOICES.get((T, N, K))
    if isinstance(choice, tuple) and choice[0] == 'bal':
        return _gb_gemv(x, w, choice)
    if choice == "cublas":
        y = torch.mm(x, w.t(), out_dtype=torch.float32) if out_dtype == torch.float32 else F.linear(x, w)
        return y.unsqueeze(0), 1
    BN, KS, BK, pf = choice if choice is not None else (*GEMV_CFG[(N, K)], PREFETCH)
    S = K // KS
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    TB = max(16, triton.next_power_of_2(T))
    _launch(_gemv_kernel, (N // BN, S), x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK, TB=TB,
            PREFETCH=pf, num_warps=4, num_stages=4)
    return P, S






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
def _attn_split_kernel(Q, KC, VC, PT, SEQ, KSCALE, OP, MP, LP, pt_stride, sm_scale, n_splits,
                       L: tl.constexpr, LB: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
                       PAGE: tl.constexpr, WINDOW: tl.constexpr, SPLIT: tl.constexpr,
                       BLOCK_N: tl.constexpr, USE_PDL: tl.constexpr):
    s = tl.program_id(0)
    b = tl.program_id(1)
    jb = tl.program_id(2)
    _pdl_start(USE_PDL)
    M: tl.constexpr = LB * HQ
    W2: tl.constexpr = 2 * D
    GROUP: tl.constexpr = HQ // 2
    seq = tl.load(SEQ + b)
    if WINDOW > 0:
        klo = tl.maximum(seq + jb * LB - WINDOW + 1, 0)
    else:
        klo = seq * 0
    khi = seq + jb * LB + LB
    start = klo + s * SPLIT
    end = tl.minimum(start + SPLIT, khi)
    r = tl.arange(0, M)
    j = jb * LB + r // HQ
    head = r % HQ
    grp = head // GROUP
    c = tl.arange(0, W2)
    qpos = seq + j
    n = tl.arange(0, BLOCK_N)
    _pdl_wait(USE_PDL)
    qv = tl.load(Q + ((b * L + j) * HQ + head)[:, None] * D + (c[None, :] % D),
                 mask=(c[None, :] // D) == grp[:, None], other=0.0).to(tl.bfloat16)
    m_i = tl.full((M,), -float("inf"), tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, W2), tl.float32)
    qk_scale = sm_scale * tl.load(KSCALE) * 1.4426950408889634
    for kb in range(start, end, BLOCK_N):
        kpos = kb + n
        valid = kpos < end
        if PAGE == 1:
            prow = tl.load(PT + b * pt_stride + kpos, mask=valid, other=0)
        else:
            pg = tl.load(PT + b * pt_stride + kpos // PAGE, mask=valid, other=0)
            prow = pg * PAGE + kpos % PAGE
        prow = prow.to(tl.int64)
        kt = tl.load(KC + prow[:, None] * W2 + c[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
        vt = tl.load(VC + prow[:, None] * W2 + c[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
        sc = tl.dot(qv, tl.trans(kt)) * qk_scale
        vis = valid[None, :] & (kpos[None, :] <= qpos[:, None])
        if WINDOW > 0:
            vis = vis & (kpos[None, :] > qpos[:, None] - WINDOW)
        sc = tl.where(vis, sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt)
        m_i = m_new
    idx = (b * (L // LB) + jb) * n_splits + s
    tl.store(MP + idx * M + r, m_i)
    tl.store(LP + idx * M + r, l_i)
    tl.store(OP + (idx * M + r)[:, None] * W2 + c[None, :], acc)


@triton.jit
def _attn_combine_kernel(OP, MP, LP, GATE, VSCALE, OUT, n_splits, eps,
                         L: tl.constexpr, LB: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
                         SPLITS_P2: tl.constexpr, USE_PDL: tl.constexpr):
    """One token: merge the splits for its 16 heads, then RMS-norm each head and gate."""
    t = tl.program_id(0)
    _pdl_start(USE_PDL)
    b = t // L
    jt = t % L
    jb = jt // LB
    M: tl.constexpr = LB * HQ
    W2: tl.constexpr = 2 * D
    GROUP: tl.constexpr = HQ // 2
    h = tl.arange(0, HQ)
    sp = tl.arange(0, SPLITS_P2)
    d = tl.arange(0, D)
    rr = (jt % LB) * HQ + h
    base = (b * (L // LB) + jb) * n_splits
    smask = sp < n_splits
    _pdl_wait(USE_PDL)
    m = tl.load(MP + (base + sp)[None, :] * M + rr[:, None], mask=smask[None, :], other=-float("inf"))
    l = tl.load(LP + (base + sp)[None, :] * M + rr[:, None], mask=smask[None, :], other=0.0)
    mmax = tl.max(m, 1)
    w = tl.where(m == -float("inf"), 0.0, tl.exp2(m - mmax[:, None]))
    denom = tl.sum(w * l, 1)
    grp = h // GROUP
    acc = tl.zeros((HQ, D), tl.float32)
    for si in range(0, n_splits):
        ws = tl.sum(tl.where(sp[None, :] == si, w, 0.0), 1)
        o = tl.load(OP + ((base + si) * M + rr)[:, None] * W2 + grp[:, None] * D + d[None, :])
        acc += ws[:, None] * o
    o = acc / denom[:, None] * tl.load(VSCALE)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    g = tl.load(GATE + t * (HQ * D) + h[:, None] * D + d[None, :])
    tl.store(OUT + t * (HQ * D) + h[:, None] * D + d[None, :], (o * tl.sigmoid(g)).to(tl.bfloat16))


def attention_triton(q, gate, data, state):
    """q: bf16 [T, 16, 64] normed, scaled by qk_scale_factor. Returns gated, normed [T, 1024]."""
    cfg = data["config"]
    batch, L, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    page, window = cfg["page_size"], cfg["window"]
    pt = data["page_table"]
    LB = min(L, 4)
    span = window + LB - 1 if window else pt.shape[1] * page
    split = 256 if window else max(512, min(4096, triton.next_power_of_2(span // 64)))
    n_splits = triton.cdiv(span, split)
    M = LB * HQ
    dev = q.device
    rows = batch * (L // LB) * n_splits * M
    op = torch.empty((rows, 2 * HD), device=dev, dtype=torch.float32)
    mp = torch.empty((rows,), device=dev, dtype=torch.float32)
    lp = torch.empty_like(mp)
    _launch(_attn_split_kernel, (n_splits, batch, L // LB),
            q, data["k_cache"], data["v_cache"], pt, data["seq_lens"], data["k_scale"], op, mp, lp,
            pt.stride(0), cfg["sm_scale"], n_splits,
            L=L, LB=LB, HQ=HQ, D=HD, PAGE=page, WINDOW=window, SPLIT=split, BLOCK_N=64,
            num_warps=4 if M <= 32 else 8, num_stages=2)
    out = torch.empty((tokens, Q_SIZE), device=dev, dtype=torch.bfloat16)
    _launch(_attn_combine_kernel, (tokens,),
            op, mp, lp, gate, data["v_scale"], out, n_splits, RMS_EPS,
            L=L, LB=LB, HQ=HQ, D=HD, SPLITS_P2=triton.next_power_of_2(n_splits), num_warps=4)
    return out


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


def _attention_base(qkv, data, state):
    if ATTN_BACKEND == "fa3":
        q, gate = qk_prep(qkv, data, state, q_fp8=True)
        return attention_fa3(q, gate, data, state)
    q, gate = qk_prep(qkv, data, state, q_fp8=False)
    return attention_triton(q, gate, data, state)






@triton.jit
def _boundary_fast(BR, S:tl.constexpr, BS:tl.constexpr, RS:tl.constexpr,
                   STREAM, SOUT, OUT, POSTW, ALPHA, BETA, PREW,
                   NB:tl.constexpr, HAS_POSTW:tl.constexpr, REPLICAS:tl.constexpr,
                   USE_PDL:tl.constexpr):
    t=tl.program_id(0);part=tl.program_id(1)
    _pdl_start(USE_PDL)
    j=tl.arange(0,8192)
    alpha=tl.load(ALPHA+j);beta=tl.load(BETA+j);prew=tl.load(PREW+j)
    if HAS_POSTW:postw=tl.load(POSTW+j)
    _pdl_wait(USE_PDL)
    st=tl.load(STREAM+t*8192+j)
    br=tl.full((8192,),0.,tl.float32)
    for s in tl.static_range(S):
        br+=tl.load(BR+s*BS+t*RS+j,j<NB,0.).to(tl.float32)
    if HAS_POSTW:br*=postw
    br*=tl.rsqrt(tl.sum(br*br,0)/8192+1.e-8)
    st=alpha*st+beta*br
    inv=tl.rsqrt(tl.sum(st*st,0)/8192+1.e-5)
    mask=(j//(8192//REPLICAS))==part
    tl.store(SOUT+t*8192+j,st,mask)
    tl.store(OUT+t*8192+j,(st*inv*prew).to(tl.bfloat16),mask)


@triton.jit
def _boundary_late(BR, S: tl.constexpr, BS: tl.constexpr, RS: tl.constexpr, STREAM, SOUT, OUT, POSTW, ALPHA, BETA, PREW,
                   NB: tl.constexpr, HAS_POSTW: tl.constexpr, USE_PDL: tl.constexpr):
    """_boundary_fast (REPLICAS 1) with the weight vectors loaded after the wait, where they are used."""
    t = tl.program_id(0)
    _pdl_start(USE_PDL)
    _pdl_wait(USE_PDL)
    j = tl.arange(0, 8192)
    st = tl.load(STREAM + t * 8192 + j)
    br = tl.full((8192,), 0., tl.float32)
    for s in tl.static_range(S):
        br += tl.load(BR + s * BS + t * RS + j, j < NB, 0.).to(tl.float32)
    if HAS_POSTW:
        br *= tl.load(POSTW + j)
    br *= tl.rsqrt(tl.sum(br * br, 0) / 8192 + 1.e-8)
    st = tl.load(ALPHA + j) * st + tl.load(BETA + j) * br
    inv = tl.rsqrt(tl.sum(st * st, 0) / 8192 + 1.e-5)
    tl.store(SOUT + t * 8192 + j, st)
    tl.store(OUT + t * 8192 + j, (st * inv * tl.load(PREW + j)).to(tl.bfloat16))


@triton.jit
def _route_fast(PR,S:tl.constexpr,PS:tl.constexpr,RS:tl.constexpr,BIAS,PREW,LSTART,
                X,IDS,WTS,LIDS,REPLICAS:tl.constexpr,USE_PDL:tl.constexpr):
    t=tl.program_id(0);part=tl.program_id(1)
    _pdl_start(USE_PDL)
    d=tl.arange(0,4096);e=tl.arange(0,256)
    prew=tl.load(PREW+d);bias=tl.load(BIAS+e);start=tl.load(LSTART)
    _pdl_wait(USE_PDL)
    p=tl.full((4096,),0.,tl.float32)
    logit=tl.full((256,),0.,tl.float32)
    for s in tl.static_range(S):
        row=PR+s*PS+t*RS
        p+=tl.load(row+d).to(tl.float32)
        logit+=tl.load(row+4096+e).to(tl.float32)+tl.load(row+4352+e).to(tl.float32)
    norm=p*tl.rsqrt(tl.sum(p*p,0)/4096+1.e-5)*prew
    tl.store(X+t*4096+d,norm.to(tl.bfloat16),d//(4096//REPLICAS)==part)
    if part==0:
        scores=tl.sigmoid(logit*0.1767766952966369)
        biased=scores+bias
        bits=biased.to(tl.int32,bitcast=True)
        ordered=tl.where(bits<0,~bits,bits^(-2147483648)).to(tl.uint32)
        keys=(ordered.to(tl.uint64)<<32)|(255-e).to(tl.uint64)
        winners=tl.topk(keys,8)
        ids=255-(winners&0xffffffff).to(tl.int32)
        scores=tl.gather(scores,ids,0)
        k=tl.arange(0,8)
        tl.store(IDS+t*8+k,ids)
        tl.store(WTS+t*8+k,scores/(tl.sum(scores,0)+1.e-15))
        local=(ids-start+256)%256
        tl.store(LIDS+t*8+k,tl.where(local<32,local,-1))


def boundary(branch,stream,alpha,beta,prew,sync=None,postw=None,use_pdl=None):
    p,s=branch;t=stream.shape[0]
    out=torch.empty((t,8192),device=stream.device,dtype=torch.bfloat16)
    sout=torch.empty_like(stream)
    warps,replicas=_GLUE_BOUNDARY.get((t,postw is not None),(8,1))
    _launch(_boundary_fast,(t,replicas),p,s,p.stride(0),p.stride(1),stream,sout,out,
            postw if postw is not None else alpha,alpha,beta,prew,NB=p.shape[2],
            HAS_POSTW=postw is not None,REPLICAS=replicas,num_warps=warps,use_pdl=True)
    return out,sout


def route_and_norm(pr,state,data,sync=None):
    p,s=pr;t=p.shape[1];dev=p.device
    x=torch.empty((t,4096),device=dev,dtype=torch.bfloat16)
    ids=torch.empty((t,8),device=dev,dtype=torch.int32)
    wts=torch.empty((t,8),device=dev,dtype=torch.float32);lids=torch.empty_like(ids)
    warps,replicas=_GLUE_ROUTE.get(t,(4,1))
    _launch(_route_fast,(t,replicas),p,s,p.stride(0),p.stride(1),state['router_bias'],
            state['pre_expert_w'],data['local_start'],x,ids,wts,lids,REPLICAS=replicas,
            num_warps=warps,use_pdl=True)
    return x,ids,wts,lids


_GLUE_BOUNDARY={}
_GLUE_ROUTE={}










def _prepare_weights(weights):
    dev = weights["qkv_gate"].device
    f32 = lambda w: w.float().contiguous()  

    def coeffs(gate):
        w = gate.float() / 0.3
        beta = torch.sigmoid(w)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
        return alpha.contiguous(), beta.contiguous()

    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    a1, b1 = coeffs(weights["attn_residual_gate"])
    a2, b2 = coeffs(weights["ffn_residual_gate"])
    state = {
        "qkv_gate": weights["qkv_gate"].contiguous(),
        "o": weights["o"].contiguous(),
        
        "pool_router": torch.cat([weights["pooler"], hi, lo], 0).contiguous(),
        "unpooler": weights["unpooler"].contiguous(),
        "unpooler_slice": weights["unpooler"][:UNPOOL_SLICE].contiguous(),
        "router_bias": f32(weights["router_bias"]),
        "alpha1": a1, "beta1": b1, "alpha2": a2, "beta2": b2,
        "pre_ffn_w": 1.0 + f32(weights["pre_ffn_norm"]),
        "pre_expert_w": 1.0 + f32(weights["pre_expert_norm"]),
        "post_expert_w": (1.0 + weights["post_expert_norm"].float()).to(torch.bfloat16),
        "post_ffn_w": 1.0 + f32(weights["post_ffn_norm"]),
        "next_pre_attn_w": 1.0 + f32(weights["next_pre_attn_norm"]),
        "inv_freq": (1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2, device=dev).float() / HD))).contiguous(),
        "moe_bank": v9_prepare({**weights, "post_norm_w": weights["post_expert_norm"]}),
    }
    torch.cuda.synchronize()
    return state



H = 4096          
NLOC = 32         
TOPK = 8
TILE_BYTES = 36864
NT1, NT2 = 512, 256   
NCNT = 64             

V9_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int HD = 4096;
constexpr int NLOC = 32;
constexpr int TILE_BYTES = 36864;
constexpr int TILE_W_BYTES = 32768;
constexpr int NT1 = 512;
constexpr int NT2 = 256;
constexpr int NCW = 8;                 // consumer warps
constexpr int NCT = NCW * 32;          // consumer threads
constexpr int NTHR = NCT;             // thread 0 also prefetches
constexpr uint32_t DQ_MASK = 0x81C081C0u;
constexpr uint32_t LA_MASK = 0x60306030u;
// persistent counters
enum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));
}
__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  while (!mbar_try(b, parity)) {}
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");
}
__device__ __forceinline__ void bar_consumers() {
  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// E4M3 -> two equal BF16 lanes scaled by 2^118. dq8 supplies 2^-126;
// the existing GEMM epilogue supplies 2^8. Preserve zeros, signs and subnormals.
__device__ __forceinline__ uint32_t scale_e4m3(uint32_t raw) {
  const uint32_t a = raw & 127u;
  uint32_t b = a >= 8 ? 0x7700u + (a << 4) :
               a >= 4 ? 0x7680u + (a << 5) :
               a >= 2 ? 0x7600u + (a << 6) : a ? 0x7600u : 0u;
  b |= (raw & 128u) << 8;
  return b | (b << 16);
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).
__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {
  o[0] = A & DQ_MASK;
  o[1] = rotr(A, 4) & DQ_MASK;
  o[2] = rotr(A, 10) & DQ_MASK;
  o[3] = rotr(B, 2) & DQ_MASK;
  o[4] = rotr(B, 6) & DQ_MASK;
  o[5] = rotr(B, 12) & DQ_MASK;
  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);
  o[6] = rotr(W, 24) & DQ_MASK;
  o[7] = rotr(W, 30) & DQ_MASK;
}

struct PrefetchState {
  volatile int index, stage, phase, pass;
  volatile int e, nb, m, mb, me;
  volatile uint64_t pol_first, pol_normal;
};

struct Route {        // shared-memory routing summary
  int n[NLOC];        // pairs per local expert
  int off[NLOC];      // first sorted position
  int nblk[NLOC];     // token blocks
  int us1[NLOC + 1];  // phase-1 unit prefix
  int us2[NLOC + 1];  // phase-2 unit prefix
  int c1, c2;         // m-chunk sizes
  int run[NLOC];
  int wc[NCW][NLOC];
  PrefetchState pre;
};

// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)
__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,
                                            int& e, int& nb, int& m) {
  e = 0;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    if (us[e + d] <= u) e += d;
  }
  const int rel = u - us[e];
  const int nbk = nblk[e];
  const int nfull = ntile / cc;
  const int span = cc * nbk;
  if (rel < nfull * span) {
    const int mc = rel / span;
    const int r2 = rel - mc * span;
    nb = r2 / cc;
    m = mc * cc + (r2 - nb * cc);
  } else {
    const int L = ntile - nfull * cc;
    const int r3 = rel - nfull * span;
    nb = r3 / L;
    m = nfull * cc + (r3 - nb * L);
  }
}

__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,
                                               const int* nblk, int cc, int ntile) {
  if (++m == me) {
    if (++nb == nblk[e]) {
      nb = 0;
      mb += cc;
      if (mb >= ntile) {
        ++e;
        while (e < NLOC && nblk[e] == 0) ++e;
        mb = 0;
      }
      me = min(ntile, mb + cc);
    }
    m = mb;
  }
}


__device__ __forceinline__ void prefetch_one(
    Route* rt, int total, int first, int a2, int nstage,
    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,
    const uint8_t* w13p, const uint8_t* w2p) {
  auto& p = rt->pre;
  int idx = p.index;
  if (idx >= total) return;
  int st=p.stage, ph=p.phase, pass=p.pass;
  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;
  const int ntile=pass?NT2:NT1;
  const int cc=pass?rt->c2:rt->c1;
  const uint8_t* base=pass?w2p:w13p;
  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;
  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;
  mbar_wait(&emptyb[st], ph^1u);
  mbar_expect_tx(&fullb[st], TILE_BYTES);
  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);
  ++idx;
  if (idx < total) {
    if (idx == first) {
      pass=1;
      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);
      mb=(m/rt->c2)*rt->c2;
      me=min(NT2,mb+rt->c2);
    } else {
      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);
    }
  }
  if (++st == nstage) { st=0; ph^=1; }
  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;
  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;
}

template <int NT>
__global__ void __launch_bounds__(NTHR, 2) moe_fused(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,
    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,
    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,
    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,
    int M, int local_start, int nstage) {
  constexpr int NB = 8 * NT;
  constexpr int RED = NT * 4;
  const int npairs = M * 8;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* stages = smem;
  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]
  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);
  uint64_t* emptyb = fullb + nstage;
  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);
  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1
  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]
  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, c = blockIdx.x;
  int bar_gen0 = 0;

  // ---------------- routing: counts ----------------
  if (tid < NLOC) rt->n[tid] = 0;
  if (tid == 0) {
    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    bar_gen0 = ld_acquire(cnt + C_GEN);
  }
  __syncthreads();
  for (int i = tid; i < npairs; i += NTHR) {
    int e = __ldg(ids + i) - local_start;
    if (e < 0 || e >= NLOC) e = -1;
    eflat[i] = static_cast<int8_t>(e);
    if (e >= 0) atomicAdd(&rt->n[e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    const int n = rt->n[lane];
    const int nb = (n + NB - 1) / NB;
    int off = n, u1 = nb * NT1, u2 = nb * NT2;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);
      if (lane >= d) { off += a; u1 += b; u2 += e2; }
    }
    rt->off[lane] = off - n;
    rt->run[lane] = off - n;
    rt->nblk[lane] = nb;
    rt->us1[lane + 1] = u1;
    rt->us2[lane + 1] = u2;
    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }
    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);
    if (lane == 0) {
      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));
      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));
    }
  }
  __syncthreads();

  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];
  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);
  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);

  const int first_count=b1-a1, total_count=first_count+b2-a2;
  if (tid == 0) {
    uint64_t pf,pn;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));
    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;
    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;
    if (total_count) {
      int e,nb,m;
      const int pass=first_count?0:1;
      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;
      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);
      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;
      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);
    }
    for (int i=0;i<nstage;++i)
      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);
  }
  __syncthreads();

  // ---------------- consumers: deterministic token lists ----------------
  for (int base = 0; base < npairs; base += NCT) {
    rt->wc[warp][lane] = 0;
    bar_consumers();
    const int i = base + tid;
    const int e = i < npairs ? eflat[i] : -1;
    const unsigned mk = __match_any_sync(~0u, e);
    const int rank = __popc(mk & ((1u << lane) - 1u));
    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);
    bar_consumers();
    if (tid < NLOC) {
      int s = rt->run[tid];
      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }
      rt->run[tid] = s;
    }
    bar_consumers();
    if (i < npairs) {
      if (e >= 0) {
        const int pos = rt->wc[warp][e] + rank;
        pos_of[i] = static_cast<int16_t>(pos);
        pair_of[pos] = static_cast<int16_t>(i);
      } else {
        pos_of[i] = -1;
      }
    }
  }
  // zero this CTA's slice of the sum-of-squares accumulators, then publish
  {
    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;
    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);
    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);
  }

  const int q = lane & 3;
  uint32_t Bf[NT][8][8];
  int st = 0;
  uint32_t ph = 0;
  int buf = 0;
  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;
  float fw[2];   // (1 + post_norm_w) for this thread's phase-2 epilogue row, refreshed per tile

  for (int pass = 0; pass < 2; ++pass) {
    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;
    const int* us = pass ? rt->us2 : rt->us1;
    const int cc = pass ? rt->c2 : rt->c1;
    const int ntile = pass ? NT2 : NT1;
    int e=0, nb=0, m=0, mb=0, me=0;
    if (ua < ub) {
      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);
      mb = (m / cc) * cc;
      me = min(ntile, mb + cc);
    }
    for (int u = ua; u < ub; ++u) {
      const int ne = rt->n[e], off = rt->off[e];
      if (pass != cur_pass || e != cur_e || nb != cur_nb) {
        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment
          __threadfence();
          bar_consumers();
          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
        }
        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;
        if (pass == 1) {
          const int target = NT1 * rt->nblk[e];
          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);
          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int idx = nb * NB + j * 8 + (lane >> 2);
          const __nv_bfloat16* row = nullptr;
          if (idx < ne) {
            const int pos = off + idx;
            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;
          }
#pragma unroll
          for (int g = 0; g < 8; ++g) {
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
            if (row) {
              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);
              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }
              else { v0 = __ldg(p); v1 = __ldg(p + 1); }
            }
            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;
            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;
          }
        }
      }
      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);

      mbar_wait(&fullb[st], ph);
      const uint8_t* tile = stages + st * TILE_BYTES;
      float acc[NT][4];
#pragma unroll
      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      {
        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;
        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);
        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          const uint4 wv = wq[g * 32];
          const uint32_t sw = scw[g >> 1];
          const uint32_t Sr = scale_e4m3((sw >> ((g & 1) * 16)) & 255u);
          const uint32_t Sr8 = scale_e4m3((sw >> (8 + (g & 1) * 16)) & 255u);
          uint32_t o[16];
          dq8(wv.x, wv.y, o);
          dq8(wv.z, wv.w, o + 8);
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;
            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);
            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);
#pragma unroll
            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&emptyb[st]);
      if (++st == nstage) { st = 0; ph ^= 1u; }

      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;
#pragma unroll
      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
      bar_consumers();
      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);
      const float* rb = red + buf * NCW * 32 * RED;
      if (pass == 0) {
        const float sc = gs13[e] * 256.f;
        if (tid < 8 * NB) {
          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;
          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);
          float G1 = 0.f, U = 0.f;
#pragma unroll
          for (int w = 0; w < NCW; ++w) {
            G1 += rb[(w * 32 + ls) * RED + rg];
            U += rb[(w * 32 + ls) * RED + rg + 2];
          }
          const int idx = nb * NB + n;
          if (idx < ne) {
            G1 *= sc;
            U *= sc;
            const float h = G1 / (1.f + __expf(-G1)) * U;
            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);
          }
        }
      } else {
        const float sc = gs2[e] * 256.f;
        if (tid < 16 * NB) {
          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;
          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);
          float v = 0.f;
#pragma unroll
          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];
          v *= sc;
          const int idx = nb * NB + n;
          const bool ok = idx < ne;
          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;
          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;
#pragma unroll
          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token
          if (ok && i == 0) atomicAdd(ss + off + idx, t2);
        }
      }
      buf ^= 1;
      ++seg_tiles;
      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);
    }
  }
  if (cur_pass == 0 && seg_tiles > 0) {
    __threadfence();
    bar_consumers();
    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
  }

  // ---------------- grid barrier ----------------
  __threadfence();
  bar_consumers();
  if (tid == 0) {
    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {
      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits
      cnt[C_ZERO] = 0;
      cnt[C_BAR] = 0;
      __threadfence();
      atomicAdd(cnt + C_GEN, 1);
    } else {
      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);
    }
  }
  bar_consumers();

  // ---------------- combine: item = (token, half of the columns) ----------------
  for (int it = c; it < 2 * M; it += G) {
    const int t = it >> 1;
    const int c0 = (it & 1) * 2048 + tid * 8;
    float f[8];
    {
      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);
      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);
    }
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int pos = pos_of[t * 8 + j];
      if (pos >= 0) {
        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);
        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);
        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);
        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];
        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];
      }
    }
    uint4 ov;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;
  }
}

size_t smem_bytes(int nt, int nstage, int M) {
  const int npairs = M * 8;
  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +
         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;
}

}  // namespace

int64_t max_smem_attr() {
  int dev, optin;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
  return optin;
}

int64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }
int64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {
  int n;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));
  return n;
}

void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,
         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,
         int64_t grid) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int M = static_cast<int>(x.size(0));
  const size_t smem = smem_bytes(nt, nstage, M);
  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());
  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define LAUNCH(NTV)                                                                                          \
  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \
      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \
      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \
      static_cast<int>(nstage))
  LAUNCH(1);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

V9_CPP_SRC = r"""
#include <torch/extension.h>
int64_t max_smem_attr();
int64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);
int64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,
         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,
         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,
         int64_t grid);
"""

_V9_MOD = None





_V9_RECODED_SCALE_START = V9_CUDA_SRC.index(
    "__device__ __forceinline__ uint32_t scale_e4m3")
_V9_RECODED_SCALE_END = V9_CUDA_SRC.index(
    "\n__device__ __forceinline__ void mma16816", _V9_RECODED_SCALE_START)
_V9_RECODED_CUDA_SRC = V9_CUDA_SRC[:_V9_RECODED_SCALE_START] + r'''
__device__ __forceinline__ uint32_t scale_e4m3(uint32_t code) {
  // code = exponent<<3 | mantissa for |E4M3|.  The bank has already moved
  // sign into the FP4 codes, so this is a positive BF16 scale in both lanes.
  const uint32_t b = 0x7000u + (code << 4);
  return b | (b << 16);
}''' + V9_CUDA_SRC[_V9_RECODED_SCALE_END:]
assert _V9_RECODED_CUDA_SRC != V9_CUDA_SRC
_V9_RECODED_MOD = None


def _v9_module():
    global _V9_MOD
    if _V9_MOD is None:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        tag = hashlib.sha1(V9_CUDA_SRC.encode()).hexdigest()[:12]
        _V9_MOD = load_inline(
            name=f"muse_moe_w4a16_v9_{tag}",
            cpp_sources=V9_CPP_SRC,
            cuda_sources=V9_CUDA_SRC,
            functions=["max_smem_attr", "smem_needed", "resident_blocks", "run"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo",
                *( ["-gencode=arch=compute_90a,code=sm_90a"] if IS_HOPPER else [] )],
            verbose=False,
        )
        _V9_MOD.smem_optin = _V9_MOD.max_smem_attr()
    return _V9_MOD


def _v9_recoded_module():
    global _V9_RECODED_MOD
    if _V9_RECODED_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(_V9_RECODED_CUDA_SRC.encode()).hexdigest()[:12]
        _V9_RECODED_MOD = load_inline(
            name=f"muse_moe_w4a16_v9_recoded_{tag}",
            cpp_sources=V9_CPP_SRC,
            cuda_sources=_V9_RECODED_CUDA_SRC,
            functions=["max_smem_attr", "smem_needed", "resident_blocks", "run"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo",
                *( ["-gencode=arch=compute_90a,code=sm_90a"] if IS_HOPPER else [] )],
            verbose=False,
        )
        _V9_RECODED_MOD.smem_optin = _V9_RECODED_MOD.max_smem_attr()
    return _V9_RECODED_MOD





_DQ_SLOTS = (("A", 0), ("A", 4), ("A", 10), ("B", 2), ("B", 6), ("B", 12), ("W", 24), ("W", 30))
_LA_MASK = 0x60306030


def _bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if (_LA_MASK >> p) & 1}
    table, used = [], {"A": set(), "B": set()}
    for k, (src, rot) in enumerate(_DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != "W" else ("A" if p in la else "B")
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used["A"]) == 32 and len(used["B"]) == 32
    return table


_TABLE = _bit_table()


def _repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = wq.shape[0], wq.shape[1] * 2
    assert K == 4096
    T = rowmap.shape[0]
    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.contiguous().view(torch.uint8)  
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                           
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)
    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)
    for k, h, b, w, p in _TABLE:
        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()                    
    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)                       
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)                  
    return torch.cat((wbytes, sbytes), 1).contiguous()


def _encode_scales(sf: torch.Tensor):
    """Losslessly encode E4M3 magnitudes for the board's recoded NT1 reader."""
    s = sf.float()
    if not torch.isfinite(s).all():
        raise ValueError("non-finite NVFP4 block scale")
    a = s.abs()
    nz = a > 0
    m, e = torch.frexp(torch.where(nz, a, torch.ones_like(a)))
    mant = ((m * 2 - 1) * 8).round().to(torch.int32)
    e5 = (e - 1 + 21).to(torch.int32)
    if bool(nz.any()):
        assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30
    assert int(mant.max()) <= 7
    code = torch.where(nz, (e5 << 3) | mant, torch.zeros_like(e5))
    back = torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)
    assert torch.equal(torch.where(nz, back, torch.zeros_like(back)), a)
    return code.to(torch.uint8), s < 0, ~nz


def _repack_recoded(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:
    """The same lossless tile layout, with sign/zero moved into FP4 codes."""
    R, K = wq.shape[0], wq.shape[1] * 2
    assert K == 4096
    T = rowmap.shape[0]
    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = _encode_scales(sf)
    codes = torch.where(neg[..., None], codes ^ 8, codes)
    codes = torch.where(zero[..., None], torch.zeros_like(codes), codes)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)
    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)
    for k, h, b, w, p in _TABLE:
        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return torch.cat((wbytes, sbytes), 1).contiguous()


def _prepare_recoded_bank(weights, raw_bank, recoded_mod):
    """Build only packed NVFP4/E4M3 storage; no dequantized weight copy."""
    dev = weights["w13"].device
    t = torch.arange(NT1, device=dev)
    i = torch.arange(8, device=dev)
    rowmap1 = torch.cat((8 * t[:, None] + i, H + 8 * t[:, None] + i), 1)
    rowmap2 = torch.arange(H, device=dev).view(NT2, 16)
    w13p = torch.empty((NLOC, NT1, TILE_BYTES), dtype=torch.uint8, device=dev)
    w2p = torch.empty((NLOC, NT2, TILE_BYTES), dtype=torch.uint8, device=dev)
    for e in range(NLOC):
        w13p[e] = _repack_recoded(weights["w13"][e], weights["w13_sf"][e], rowmap1)
        w2p[e] = _repack_recoded(weights["w2"][e], weights["w2_sf"][e], rowmap2)
    bank = dict(raw_bank)
    bank.update(w13p=w13p, w2p=w2p, mod=recoded_mod,
                cnt=torch.zeros_like(raw_bank["cnt"]))
    return bank


def v9_prepare(weights):
    dev = weights["w13"].device
    t = torch.arange(NT1, device=dev)
    i = torch.arange(8, device=dev)
    rowmap1 = torch.cat((8 * t[:, None] + i, H + 8 * t[:, None] + i), 1)    
    rowmap2 = torch.arange(H, device=dev).view(NT2, 16)
    w13p = torch.empty((NLOC, NT1, TILE_BYTES), dtype=torch.uint8, device=dev)
    w2p = torch.empty((NLOC, NT2, TILE_BYTES), dtype=torch.uint8, device=dev)
    for e in range(NLOC):
        w13p[e] = _repack(weights["w13"][e], weights["w13_sf"][e], rowmap1)
        w2p[e] = _repack(weights["w2"][e], weights["w2_sf"][e], rowmap2)
    mod = _v9_module()
    props = torch.cuda.get_device_properties(dev)
    torch.cuda.synchronize()
    return {
        "w13p": w13p, "w2p": w2p,
        "gs13": weights["w13_gs"][:NLOC].float().contiguous(),
        "gs2": weights["w2_gs"][:NLOC].float().contiguous(),
        "pnw": weights["post_norm_w"].contiguous(),
        
        "cnt": torch.zeros((NCNT,), dtype=torch.int32, device=dev),
        "grid": props.multi_processor_count,
        "mod": mod,
    }


def v9_kernel_fn(data, state):
    cfg = data["config"]
    assert cfg["local_count"] == NLOC and cfg["d_hidden"] == H and cfg["d_expert"] == H
    mod = state["mod"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * TOPK
    nt = state.get("nt",1)
    nstage = 2
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = torch.empty((npairs, H), dtype=torch.bfloat16, device=dev)
    y = torch.empty((npairs, H), dtype=torch.float32, device=dev)
    ss = torch.empty((npairs,), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state["pnw"], state["w13p"], state["w2p"], state["gs13"], state["gs2"], inter, y, ss,
            state["cnt"], out, cfg["local_start"], nt, nstage, state["grid"] * mod.resident_blocks(nt, nstage, M))
    return out



def experts_v9(x, wts, lids, state):
    nt = state.get("moe_nt_plan", {}).get(x.shape[0], 1)
    base = state.get("moe_bank", {}) if nt == 1 else state.get("moe_bank_raw", state["moe_bank"])
    return v9_kernel_fn({"hidden_states": x, "topk_ids": lids,
                         "topk_weights": wts,
                         "config": {"local_count": 32, "d_hidden": 4096,
                                    "d_expert": 4096, "local_start": 0}},
                        dict(base, mod=state["moe_nt2_mod"], nt=2) if nt == 2 else base)


CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace {

constexpr int D = 64;      // head dim
constexpr int G = 8;       // query heads per KV head
constexpr int T = 64;      // tokens per pipeline tile
constexpr int NTHR = 128;  // 4 warps
constexpr int ROWB = 128;  // bytes of one token's K (or V) for a pair of KV heads
constexpr int TILE_BYTES = T * ROWB;

struct Params {
  const void* qkv;
  int qkv_splits, qkv_ps, qkv_float;
  const int64_t* positions;
  const int64_t* cache_loc;
  const int* page_table;
  const int* seq_lens;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;   // nullptr: no RoPE
  float* part_o;           // [B, KVG, NS, L, 16, 64]
  float* part_ml;          // [B, KVG, NS, L, 16, 2]
  __nv_bfloat16* out;      // [B, L, HQ*64]
  int batch;
  int qkv_stride;
  int hq, hkv;
  int max_pages;
  int ps, ps_log2;
  int window;
  int ns;
  long long row_bytes;     // hkv * 64
  float scale_log2;        // sm_scale * qk_scale_factor * log2(e)
};

DEVI __nv_bfloat162 qkv_pair(const Params& p, long long off) {
  if (!p.qkv_float) return *reinterpret_cast<const __nv_bfloat162*>(static_cast<const __nv_bfloat16*>(p.qkv) + off);
  float a = 0.f, b = 0.f;
  for (int s = 0; s < p.qkv_splits; ++s) {
    float2 z = *reinterpret_cast<const float2*>(static_cast<const float*>(p.qkv) + (long long)s * p.qkv_ps + off);
    a += z.x; b += z.y;
  }
  return __floats2bfloat162_rn(a, b);
}

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

DEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

DEVI uint32_t cvt_e4m3x2(uint32_t v) {
  uint32_t r;
  uint16_t h = static_cast<uint16_t>(v & 0xffffu);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(h));
  return r;
}

DEVI uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

DEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.
DEVI int swz(int t) { return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2); }

DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int L>
struct WarpState {
  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp
  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims
  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims
  float m[MTW][2];                               // running max (log2 domain, scaled)
  float l[MTW][2];                               // running sum (thread-partial)
};

// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.
// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);
//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).
template <int L, int NTC>
DEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,
                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2) {
  constexpr int MTW = WarpState<L>::MTW;
  constexpr int KK = NTC / 2;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, q4 = lane & 3;

  float s[MTW][NTC][4];
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
    for (int n = 0; n < NTC; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;

  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,
  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q's fragments).
#pragma unroll
  for (int n = 0; n < NTC; ++n) {
    const int t = tb + 8 * n + g;
    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));
    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint32_t b0 = cvt_e4m3x2(w[ks]);
      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);
#pragma unroll
      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);
    }
  }

  if (need_mask) {
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));
        const int qp = seq + j;
        const int lo = window ? (qp - window + 1) : -2147483647;
        const int hi = min(k_hi, qp + 1);
#pragma unroll
        for (int n = 0; n < NTC; ++n) {
          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);
          if (kp < lo || kp >= hi) s[mt][n][e] = -INFINITY;
        }
      }
    }
  }

  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).
  uint32_t pa[MTW][KK][4];
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      if (L == 1 && rh == 1) continue;
      float mx = -INFINITY;
#pragma unroll
      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);
      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
      const float corr = ex2(st.m[mt][rh] - m_use);
      st.m[mt][rh] = m_new;
      st.l[mt][rh] *= corr;
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) {
        st.o[mt][jn][2 * rh] *= corr;
        st.o[mt][jn][2 * rh + 1] *= corr;
      }
      float sum = 0.f;
#pragma unroll
      for (int n = 0; n < NTC; ++n) {
        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));
        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));
        s[mt][n][2 * rh] = p0;
        s[mt][n][2 * rh + 1] = p1;
        sum += p0 + p1;
      }
      st.l[mt][rh] += sum;
    }
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);
      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);
      if (L == 1) {
        pa[mt][kk][1] = 0u;
        pa[mt][kk][3] = 0u;
      } else {
        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);
        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);
      }
    }
  }

  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a
  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
    const int t0 = tb + 16 * kk + 2 * q4;
    const int coff = ((4 * h + (g >> 1))) ;
    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));
    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));
    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));
    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;
      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;
      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);
      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);
      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};
      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};
#pragma unroll
      for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);
    }
  }
}

template <int L, int STAGES>
__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // attn_combine launches early and waits
#endif
  constexpr int MTW = WarpState<L>::MTW;
  constexpr int NT = (L == 1) ? 4 : 8;  // n-tiles per warp per tile
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sK = smem;
  uint8_t* sV = smem + STAGES * TILE_BYTES;
  uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;  // new tokens: 16 rows of K, 16 of V
  uint8_t* sNV = sNK + 16 * ROWB;
  __half* sQ = reinterpret_cast<__half*>(sNV + 16 * ROWB);  // [2][L*8][64]
  float2* sCS = reinterpret_cast<float2*>(sQ + 2 * L * G * D);  // [L][32]

  const int split = blockIdx.x, kvg = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int h = warp & 1, wh = warp >> 1;
  const bool last = split == p.ns - 1;

  const int seq = p.seq_lens[b];
  const int lo = p.window ? max(0, seq - p.window + 1) : 0;
  const int n_cached = seq - lo;
  int chunk = (n_cached + p.ns - 1) / p.ns;
  chunk = (chunk + 15) & ~15;
  const int c_lo = min(lo + split * chunk, seq);
  const int c_hi = min(c_lo + chunk, seq);
  const int ntiles = (c_hi - c_lo + T - 1) / T;

  const int* pt = p.page_table + (long long)b * p.max_pages;
  const int lc = tid & 7;      // 16-byte chunk of the row this thread copies
  const int lt = tid >> 3;     // first token (of 4: lt + 16k)
  const uint8_t* kbase = p.k_cache + kvg * ROWB + lc * 16;
  const uint8_t* vbase = p.v_cache + kvg * ROWB + lc * 16;

  auto load_rows = [&](int tile, int (&rows)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = c_lo + tile * T + lt + 16 * k;
      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;
    }
  };
  auto issue = [&](int slot, const int (&rows)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = lt + 16 * k;
      const int off = slot * TILE_BYTES + t * ROWB + ((lc ^ swz(t)) << 4);
      const long long r = rows[k] < 0 ? 0 : rows[k];
      cp_async16(smem_u32(sK + off), kbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);
      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);
    }
  };

  // Every independent global load of the prologue goes out first (page-table rows, queries,
  // new keys/values, positions, cache rows), so their latencies overlap.
  int rows[STAGES][4];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) load_rows(s, rows[s]);
  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)
  __nv_bfloat162 qv[QR];
#pragma unroll
  for (int i = 0; i < QR; ++i) {
    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;
    qv[i] = qkv_pair(p, (long long)(b * L + j) * p.qkv_stride +
                                                      (kvg * 16 + hh) * D + 2 * lane);
  }
  __nv_bfloat162 nv[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)
  long long nloc[L];
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;
      nv[i] = qkv_pair(p, (long long)(b * L + j) * p.qkv_stride +
                                                        (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);
      nloc[i] = p.cache_loc[b * L + j];
    }
  }
  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;
  float ang[CSN];
#pragma unroll
  for (int c = 0; c < CSN; ++c) {
    const int i = tid + c * NTHR;
    ang[c] = 0.f;
    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];
  }
  const float ksc = *p.k_scale, vsc = *p.v_scale;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ntiles) issue(s, rows[s]);
    cp_commit();
  }
  int rows_next[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) rows_next[k] = rows[STAGES - 1][k];

  // RoPE angles for the block's tokens.
  if (p.inv_freq) {
#pragma unroll
    for (int c = 0; c < CSN; ++c) {
      const int i = tid + c * NTHR;
      if (i < L * 32) {
        float sn, cs;
        sincosf(ang[c], &sn, &cs);
        sCS[i] = make_float2(cs, sn);
      }
    }
  }
  if (last) {  // zero the new-token rows past L
    for (int i = tid; i < 2 * 16 * ROWB / 16; i += NTHR) {
      if ((i % (16 * ROWB / 16)) >= L * ROWB / 16) reinterpret_cast<uint4*>(sNK)[i] = make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();

  const float eps = 1e-5f;
  // Queries: normalized and roped, FP16 into smem as [kv head][token j * 8 + head][64].
#pragma unroll
  for (int i = 0; i < QR; ++i) {
    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;
    float x0 = __bfloat162float(qv[i].x), x1 = __bfloat162float(qv[i].y);
    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);
    x0 *= inv;
    x1 *= inv;
    if (p.inv_freq) {
      const float2 cs = sCS[j * 32 + lane];
      const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;
      x0 = y0;
      x1 = y1;
    }
    *reinterpret_cast<__half2*>(sQ + ((hh >> 3) * (L * G) + j * G + (hh & 7)) * D + 2 * lane) = __floats2half2_rn(x0, x1);
  }
  // New keys and values (last split): normalize/rope keys, quantize, store to the cache and smem.
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;
      const int head = kvg * 2 + hk;
      float x0 = __bfloat162float(nv[i].x), x1 = __bfloat162float(nv[i].y);
      if (!isv) {
        const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);
        x0 *= inv;
        x1 *= inv;
        if (p.inv_freq) {
          const float2 cs = sCS[j * 32 + lane];
          const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;
          x0 = y0;
          x1 = y1;
        }
      }
      const float sc = isv ? vsc : ksc;
      const __nv_fp8x2_storage_t f8 = __nv_cvt_float2_to_fp8x2(make_float2(x0 / sc, x1 / sc), __NV_SATFINITE, __NV_E4M3);
      uint8_t* cache = isv ? p.v_cache : p.k_cache;
      *reinterpret_cast<__nv_fp8x2_storage_t*>(cache + nloc[i] * p.row_bytes + head * D + 2 * lane) = f8;
      const int byte = hk * D + 2 * lane;
      uint8_t* dst = (isv ? sNV : sNK) + j * ROWB + ((((byte >> 4) ^ swz(j))) << 4) + (byte & 15);
      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;
    }
  }
  __syncthreads();

  WarpState<L> st;
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
    const int mta = (L == 1) ? 0 : (2 * wh + mt);
    const int r0 = 16 * mta + g, r1 = r0 + 8;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint2 x0 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r0) * D + 16 * q4 + 4 * ks);
      st.qa[mt][ks][0] = x0.x;
      st.qa[mt][ks][2] = x0.y;
      if (L == 1) {
        st.qa[mt][ks][1] = 0u;
        st.qa[mt][ks][3] = 0u;
      } else {
        const uint2 x1 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r1) * D + 16 * q4 + 4 * ks);
        st.qa[mt][ks][1] = x1.x;
        st.qa[mt][ks][3] = x1.y;
      }
    }
    st.m[mt][0] = st.m[mt][1] = -INFINITY;
    st.l[mt][0] = st.l[mt][1] = 0.f;
#pragma unroll
    for (int jn = 0; jn < 8; ++jn)
#pragma unroll
      for (int e = 0; e < 4; ++e) st.o[mt][jn][e] = 0.f;
  }

  const float scale_log2 = p.scale_log2 * (*p.k_scale);
  const int tb = (L == 1) ? 32 * wh : 0;
  const int win_edge = p.window ? (seq + L - p.window) : -2147483647;  // keys below need per-row masks

  for (int i = 0; i < ntiles; ++i) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int nxt = i + STAGES - 1;
    if (nxt < ntiles) issue(nxt % STAGES, rows_next);
    cp_commit();
    load_rows(nxt + 1, rows_next);
    const int slot = i % STAGES;
    const int kp0 = c_lo + i * T;
    const bool need_mask = (kp0 + T > c_hi) || (L > 1 && kp0 < win_edge);
    compute_tile<L, NT>(st, sK + slot * TILE_BYTES, sV + slot * TILE_BYTES, tb, h, wh, kp0, c_hi, seq, p.window,
                        need_mask, scale_log2);
  }
  cp_wait<0>();
  if (last && (L > 1 || wh == 0)) {
    compute_tile<L, 2>(st, sNK, sNV, 0, h, wh, seq, seq + L, seq, p.window, true, scale_log2);
  }

  // Thread-partial sums -> row sums.
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float v = st.l[mt][rh];
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      st.l[mt][rh] = v;
    }

  if (L == 1) {
    // Merge the two token halves of each KV head through shared memory.
    __syncthreads();
    float* mrg = reinterpret_cast<float*>(smem);  // [2 heads][32 lanes][18]
    if (wh == 1) {
      float* dst = mrg + (h * 32 + lane) * 18;
      dst[0] = st.m[0][0];
      dst[1] = st.l[0][0];
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) {
        dst[2 + jn] = st.o[0][jn][0];
        dst[10 + jn] = st.o[0][jn][1];
      }
    }
    __syncthreads();
    if (wh == 1) return;
    const float* src = mrg + (h * 32 + lane) * 18;
    const float m1 = src[0], l1 = src[1];
    const float m0 = st.m[0][0];
    const float M = fmaxf(m0, m1);
    const float Mu = (M == -INFINITY) ? 0.f : M;
    const float w0 = ex2(m0 - Mu), w1 = ex2(m1 - Mu);
    st.m[0][0] = M;
    st.l[0][0] = st.l[0][0] * w0 + l1 * w1;
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
      st.o[0][jn][0] = st.o[0][jn][0] * w0 + src[2 + jn] * w1;
      st.o[0][jn][1] = st.o[0][jn][1] * w0 + src[10 + jn] * w1;
    }
  }

  // Write the partial: rows (token j, query head hh), dims 16*q4 .. 16*q4+15 per thread.
  const int KVG = p.hkv / 2;
  const long long pbase = ((long long)(b * KVG + kvg) * p.ns + split) * L;
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      if (L == 1 && rh == 1) continue;
      const int mta = (L == 1) ? 0 : (2 * wh + mt);
      const int j = (L == 1) ? 0 : (2 * mta + rh);
      const int hh = h * 8 + g;
      const long long row = (pbase + j) * 16 + hh;
      float4* dst = reinterpret_cast<float4*>(p.part_o + row * D + 16 * q4);
      dst[0] = make_float4(st.o[mt][0][2 * rh], st.o[mt][1][2 * rh], st.o[mt][2][2 * rh], st.o[mt][3][2 * rh]);
      dst[1] = make_float4(st.o[mt][4][2 * rh], st.o[mt][5][2 * rh], st.o[mt][6][2 * rh], st.o[mt][7][2 * rh]);
      dst[2] = make_float4(st.o[mt][0][2 * rh + 1], st.o[mt][1][2 * rh + 1], st.o[mt][2][2 * rh + 1], st.o[mt][3][2 * rh + 1]);
      dst[3] = make_float4(st.o[mt][4][2 * rh + 1], st.o[mt][5][2 * rh + 1], st.o[mt][6][2 * rh + 1], st.o[mt][7][2 * rh + 1]);
      if (q4 == 0) reinterpret_cast<float2*>(p.part_ml)[row] = make_float2(st.m[mt][rh], st.l[mt][rh]);
    }
  }
}

// One warp per (request, token, query head): merge splits, RMSNorm, x sigmoid(gate).
template <int L>
__global__ void __launch_bounds__(128) attn_combine(const Params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // dependents (the gate kernel, O) wait
  asm volatile("griddepcontrol.wait;" ::: "memory");               // attn_main's partials
#endif
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wid = blockIdx.x * 4 + warp;
  if (wid >= p.batch * L * p.hq) return;
  const int b = wid / (L * p.hq);
  const int rem = wid - b * (L * p.hq);
  const int j = rem / p.hq, hg = rem - j * p.hq;
  const int kvg = hg >> 4, hh = hg & 15;
  const int KVG = p.hkv / 2;
  const long long base = (((long long)(b * KVG + kvg) * p.ns) * L + j) * 16 + hh;  // row of split 0
  const long long sstride = (long long)L * 16;
  const float2* ml = reinterpret_cast<const float2*>(p.part_ml);

  float M = -INFINITY;
  for (int s = lane; s < p.ns; s += 32) M = fmaxf(M, ml[base + s * sstride].x);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  const float Mu = (M == -INFINITY) ? 0.f : M;
  float a0 = 0.f, a1 = 0.f, lsum = 0.f;
#pragma unroll 4
  for (int s = 0; s < p.ns; ++s) {
    const long long row = base + s * sstride;
    const float2 mlv = ml[row];
    const float w = ex2(mlv.x - Mu);
    const float2 ov = reinterpret_cast<const float2*>(p.part_o + row * D)[lane];
    a0 = fmaf(w, ov.x, a0);
    a1 = fmaf(w, ov.y, a1);
    lsum = fmaf(w, mlv.y, lsum);
  }
  const float vs = *p.v_scale;
  const float inv_l = vs / lsum;
  a0 *= inv_l;
  a1 *= inv_l;
  const float r = rsqrtf(warp_sum(a0 * a0 + a1 * a1) * (1.f / D) + 1e-5f);
  const long long tok = (long long)b * L + j;
  float g0 = 1.f, g1 = 1.f;
  if (p.qkv_stride > (p.hq + 2 * p.hkv) * D) {  // gate columns present (else P7 gate split: ungated output)
    const __nv_bfloat162 gt = qkv_pair(p, tok * p.qkv_stride + (p.hq + 2 * p.hkv + hg) * D + 2 * lane);
    g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));
    g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));
  }
  *reinterpret_cast<__nv_bfloat162*>(p.out + tok * (p.hq * D) + hg * D + 2 * lane) =
      __floats2bfloat162_rn(a0 * r * g0, a1 * r * g1);
}

template <int L, int STAGES>
void launch_t(const Params& p, cudaStream_t stream) {
  const int smem = 2 * STAGES * TILE_BYTES + 2 * 16 * ROWB + 2 * L * G * D * 2 + L * 32 * 8;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  dim3 grid(p.ns, p.hkv / 2, p.batch);
  attn_main<L, STAGES><<<grid, NTHR, smem, stream>>>(p);
  const int warps = p.batch * L * p.hq;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((warps + 3) / 4);
  cfg.blockDim = dim3(128);
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, attn_combine<L>, p);
}

}  // namespace

void attn_forward(int64_t qkv, int64_t qkv_splits, int64_t qkv_ps, int64_t qkv_float, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,
                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,
                  int64_t part_o, int64_t part_ml, int64_t out, int64_t batch, int64_t q_len, int64_t qkv_stride,
                  int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window, int64_t ns,
                  double scale_log2, int64_t stream) {
  Params p;
  p.qkv = reinterpret_cast<const void*>(qkv);
  p.qkv_splits = qkv_splits; p.qkv_ps = qkv_ps; p.qkv_float = qkv_float;
  p.positions = reinterpret_cast<const int64_t*>(positions);
  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);
  p.page_table = reinterpret_cast<const int*>(page_table);
  p.seq_lens = reinterpret_cast<const int*>(seq_lens);
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);
  p.k_scale = reinterpret_cast<const float*>(k_scale);
  p.v_scale = reinterpret_cast<const float*>(v_scale);
  p.inv_freq = reinterpret_cast<const float*>(inv_freq);
  p.part_o = reinterpret_cast<float*>(part_o);
  p.part_ml = reinterpret_cast<float*>(part_ml);
  p.out = reinterpret_cast<__nv_bfloat16*>(out);
  p.batch = (int)batch;
  p.qkv_stride = (int)qkv_stride;
  p.hq = (int)hq;
  p.hkv = (int)hkv;
  p.max_pages = (int)max_pages;
  p.ps = (int)ps;
  int lg = 0;
  while ((1 << lg) < ps) ++lg;
  p.ps_log2 = lg;
  p.window = (int)window;
  p.ns = (int)ns;
  p.row_bytes = hkv * 64;
  p.scale_log2 = (float)scale_log2;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  if (q_len == 1) launch_t<1, 4>(p, s);
  else launch_t<8, 3>(p, s);
}
"""

CPP_SRC = r"""
#include <cstdint>
void attn_forward(int64_t qkv, int64_t qkv_splits, int64_t qkv_ps, int64_t qkv_float, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,
                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,
                  int64_t part_o, int64_t part_ml, int64_t out, int64_t batch, int64_t q_len, int64_t qkv_stride,
                  int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window, int64_t ns,
                  double scale_log2, int64_t stream);
"""

_LAYER_ATTN_EXT = None


def _layer_attn_ext():
    global _LAYER_ATTN_EXT
    if _LAYER_ATTN_EXT is None:
        major, minor = torch.cuda.get_device_capability()
        arch = "90a" if major == 9 else f"{major}{minor}"
        _LAYER_ATTN_EXT = load_inline(
            name=f"muse_decode_attn_split_{arch}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["attn_forward"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-std=c++20",
                               "--expt-relaxed-constexpr", "-lineinfo"],
            verbose=False,
        )
    return _LAYER_ATTN_EXT


_LAYER_INV_FREQ = {}


def _inv_freq(theta, device):
    key = (float(theta), str(device))
    if key not in _LAYER_INV_FREQ:
        _LAYER_INV_FREQ[key] = (1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()
    return _LAYER_INV_FREQ[key]


_LAYER_SMS = {}


def _num_sms(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _LAYER_SMS:
        _LAYER_SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _LAYER_SMS[idx]


def _splits(batch, kvg, max_keys, q_len, sms):
    """Splits of each request's cached keys: enough CTAs to fill the GPU about twice over,
    at least one 64-token tile each, at most 128 (the combine reads every split)."""
    target = 2 * sms
    ns = -(-target // (batch * kvg))
    ns = min(ns, max(1, -(-max_keys // 64)), 128)
    return max(1, ns)


def _layer_attention_custom(qkv, data):
    qkv, splits = qkv
    if qkv.dtype == torch.float32 and splits > 1 and qkv.stride(-1) == 1:
        qkv, splits = _attn_sumcast(qkv, splits)
    cfg = data["config"]
    batch, q_len = cfg["batch"], cfg["q_len"]
    hq, hkv, d, ps = 16, 2, 64, cfg["page_size"]
    max_pages = data["page_table"].shape[1]
    window = cfg["window"]
    max_keys = max_pages * ps
    if window:
        max_keys = min(max_keys, window)
    ns = _splits(batch, hkv // 2, max_keys, q_len, _num_sms(qkv.device))
    part_o = torch.empty((batch * (hkv // 2) * ns * q_len * 16 * 64,), dtype=torch.float32, device=qkv.device)
    part_ml = torch.empty((batch * (hkv // 2) * ns * q_len * 16 * 2,), dtype=torch.float32, device=qkv.device)
    out = torch.empty((batch, q_len, hq * d), dtype=torch.bfloat16, device=qkv.device)
    inv_freq = _inv_freq(cfg["rope_theta"], qkv.device) if cfg["rope_theta"] else None
    scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634
    _layer_attn_ext().attn_forward(
        qkv.data_ptr(), splits, qkv.stride(0), int(qkv.dtype == torch.float32), data["positions"].data_ptr(), data["cache_loc"].data_ptr(), data["page_table"].data_ptr(),
        data["seq_lens"].data_ptr(), data["k_cache"].data_ptr(), data["v_cache"].data_ptr(),
        data["k_scale"].data_ptr(), data["v_scale"].data_ptr(), inv_freq.data_ptr() if inv_freq is not None else 0,
        part_o.data_ptr(), part_ml.data_ptr(), out.data_ptr(), batch, q_len, qkv.shape[-1], hq, hkv, max_pages,
        ps, window, ns, scale_log2, torch.cuda.current_stream(qkv.device).cuda_stream)
    return out.view(-1, hq * d)


def _layer_stage_attention(qkv, data, state):
    
    cfg = data["config"]
    if cfg["q_len"] == 1:
        return _layer_attention_custom(qkv, data)
    return _attention_base(qkv, data, state)

_GEMV_CHOICES = {}



import torch.nn.functional as F

CD_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdexcept>

namespace cdg {

__device__ __forceinline__ void pdl_release() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ uint4 ld_w(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// Predicated in the asm: the outputs count as written either way, so the compiler keeps no copy of the old value.
__device__ __forceinline__ uint4 ld_w_if(const uint4* p, int pred) {
  uint4 v;
  asm volatile("{\n.reg .pred q;\nsetp.ne.b32 q, %5, 0;\n"
               "@q ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];\n}"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "r"(pred));
  return v;
}

__device__ __forceinline__ void l2_prefetch(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

__device__ __forceinline__ void cp_async16(void* s, const void* g) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(g));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// One k16 step: low-byte words lw0 (A-fragment elements a0-a3), lw1 (a4-a7) and the code word nw -> 4 A regs.
// Code word byte k: bit 7 = sign of a_k, bits 0-2 = d>>1 of a_k; bit 3 = sign of a_{4+k};
// bits 4-6 = d>>1 of a_{4+((k+1)&3)} (a 4-bit rotate brings them under the same 0x87 mask).
__device__ __forceinline__ void decode_step(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t (&a)[4]) {
  const uint32_t ha = (nw & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (__funnelshift_l(nw, nw, 4) & 0x87878787u) | 0x38383838u;
  a[0] = prmt(lw0, ha, 0x5140);
  a[1] = prmt(lw0, ha, 0x7362);
  a[2] = prmt(lw1, hb, 0x5140);
  a[3] = prmt(lw1, hb, 0x7362);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ float ld_f32(const float* p) {
  float v;
  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ int ld_i32(const int* p) {
  int v;
  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ int2 ld_i64(const int2* p) {
  int2 v;
  asm volatile("ld.global.nc.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait0(uint32_t a) {  // phase 0 (each barrier is used once per launch)
  uint32_t done;
  do {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\nselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done) : "r"(a) : "memory");
  } while (!done);
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}

// Activation chunk of the V3 staging: CB k-blocks (CB * 64 columns) per mbarrier.
constexpr int CB = 4;

// V3+ shared memory: xs [M + 1][nb * 64 + 8] bf16 (row M = zeros), nch mbarriers, then (V4) the partial buffer
// (placed after the largest split's region, so mapa finds it at the same offset in every CTA of the cluster).
__host__ __device__ inline int red_off(int M, int nb, int nch) {
  return ((((M + 1) * (nb * 64 + 8) * 2 + 7) & ~7) + 8 * nch + 15) & ~15;
}

// grid (R / (NW * RT), S), NW warps; warp w of CTA x owns row tiles (x * NW + w) * RT + [0, RT) over the
// split's k-blocks [s * KB / S, (s + 1) * KB / S). NT = n8 token tiles (M <= 8 NT). The CTA stages
// X[0:M, split] in shared memory (row stride L + 8 bf16: conflict-free 16 B fragment loads).
// V (lab steps, paired on the H200):
//   1  scales / escape offsets / escapes loaded after the MMA loop; activations in two cp.async groups with CTA
//      barriers; token tiles past M guarded per fragment load
//   2  1 + scales and escape offsets loaded at entry, the first 32 escapes of each row tile prefetched (one per
//      lane) after the first block, applied from registers by warp shuffles
//   3  2 + activations staged by bulk copies (one per row and CB-block chunk) completing on per-chunk mbarriers
//      (each warp waits only for the chunk it is about to use; no CTA barrier in the loop), and token rows past M
//      read a zero row (no per-fragment branch)
//   4  3 + the C CTAs of a cluster (consecutive splits) sum their partials through distributed shared memory:
//      P gets S / C partials (C = S: one) instead of S, so the consumer reads C times fewer bytes
template <int NT, int RT, int PIPE, int V>
__global__ void __launch_bounds__(256) coded_gemv(const uint4* __restrict__ Wc, const float* __restrict__ scales,
                                                  const int* __restrict__ eoff, const int2* __restrict__ esc,
                                                  const __nv_bfloat16* __restrict__ X, int x_stride, int M,
                                                  float* __restrict__ P, int N, int KB, int S, int pf, int prew,
                                                  int C) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  pdl_release();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int nw = blockDim.x >> 5, s = blockIdx.y;
  const int kb0 = (int)((long long)s * KB / S), kb1 = (int)((long long)(s + 1) * KB / S), nb = kb1 - kb0;
  const int r0 = (blockIdx.x * nw + warp) * RT;
  const int L = nb * 64, RS = L + 8, k0 = kb0 * 64;
  const int nbm = (KB + S - 1) / S;  // the largest split: V4's partial buffer sits at the same offset in every CTA
  float sc[RT];
  int ea[RT], eb[RT];
  int2 epre[RT];
  if constexpr (V >= 2) {  // weight-side constants: issued before griddepcontrol.wait, consumed after the loop
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
      sc[rt] = ld_f32(scales + r0 + rt);
      ea[rt] = ld_i32(eoff + (size_t)(r0 + rt) * KB + kb0);
      eb[rt] = ld_i32(eoff + (size_t)(r0 + rt) * KB + kb1);
    }
  }
  const uint4* wp[RT];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) wp[rt] = Wc + ((size_t)(r0 + rt) * KB + kb0) * 96 + lane;
  uint4 buf[PIPE][RT][3];
  if (!prew) pdl_wait();  // prew: the first PIPE blocks load before griddepcontrol.wait (overlap the producer)
#pragma unroll
  for (int p = 0; p < PIPE; ++p)
    if (p < nb)
#pragma unroll
      for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w(wp[rt] + p * 96 + q * 32);
  if (pf) {  // L2 prefetch of the rest of this warp's range (128 B lines, lane-strided)
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
      const char* base = reinterpret_cast<const char*>(Wc + ((size_t)(r0 + rt) * KB + kb0) * 96);
      for (int off = (PIPE * 1536) + lane * 128; off < nb * 1536; off += 32 * 128) l2_prefetch(base + off);
    }
  }
  const int c0 = nb < PIPE ? nb : PIPE;
  const int nch = (nb + CB - 1) / CB;
  const uint32_t mbar0 = smem_u32(smem_raw + (((size_t)(M + 1) * RS * 2 + 7) & ~(size_t)7));
  if constexpr (V <= 2) {
    if (prew) pdl_wait();
    // Activations X[0:M, split] -> xs in two cp.async groups: the first PIPE blocks' columns, then the rest, so
    // the MMA loop starts once the first group lands.
    const int cpr0 = c0 * 8, cpr1 = (nb - c0) * 8;
    for (int i = threadIdx.x; i < M * cpr0; i += blockDim.x) {
      const int row = i / cpr0, c = i - row * cpr0;
      cp_async16(xs + row * RS + c * 8, X + (size_t)row * x_stride + k0 + c * 8);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    for (int i = threadIdx.x; i < M * cpr1; i += blockDim.x) {
      const int row = i / cpr1, c = cpr0 + (i - row * cpr1);
      cp_async16(xs + row * RS + c * 8, X + (size_t)row * x_stride + k0 + c * 8);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;" ::: "memory");
    __syncthreads();
  } else {
    // Row M of xs is the zero row (token rows past M read it); chunk c's mbarrier completes when its M bulk copies land.
    if (threadIdx.x == 0) {
      for (int c = 0; c < nch; ++c) mbar_init(mbar0 + 8 * c, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    for (int i = threadIdx.x; i < L / 8; i += blockDim.x)
      reinterpret_cast<uint4*>(xs + (size_t)M * RS)[i] = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();
    if (warp == nw - 1) {
      if (prew) pdl_wait();
      if (lane == 0)
        for (int c = 0; c < nch; ++c) mbar_expect(mbar0 + 8 * c, (uint32_t)(M * (min(CB, nb - c * CB) * 128)));
      __syncwarp();
      for (int i = lane; i < M * nch; i += 32) {  // chunk-major: the first chunk's rows go first
        const int c = i / M, row = i - c * M;
        bulk_g2s(smem_u32(xs + (size_t)row * RS + c * CB * 64), X + (size_t)row * x_stride + k0 + c * CB * 64,
                 (uint32_t)(min(CB, nb - c * CB) * 128), mbar0 + 8 * c);
      }
    }
  }
  const __nv_bfloat16* xrow[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int tok = g + 8 * nt;
    xrow[nt] = xs + (size_t)(tok < M ? tok : M) * RS + 16 * t;
  }

  float acc[RT][NT][4];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[rt][nt][i] = 0.f;

  for (int b = 0; b < nb; b += PIPE) {
    if constexpr (V >= 3)
      if (b % CB == 0) mbar_wait0(mbar0 + 8 * (b / CB));
#pragma unroll
    for (int p = 0; p < PIPE; ++p) {
      const int bb = b + p;
      if constexpr (V <= 2)
        if (bb == c0) {  // the second activation group (uniform across the CTA)
          asm volatile("cp.async.wait_group 0;" ::: "memory");
          __syncthreads();
        }
      if (bb < nb) {
        uint4 cur[RT][3];
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
          for (int q = 0; q < 3; ++q) cur[rt][q] = buf[p][rt][q];
        if constexpr (V == 1)
          if (bb + PIPE < nb)
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
              for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w(wp[rt] + (bb + PIPE) * 96 + q * 32);
        uint4 xv[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          if (V >= 3 || g + 8 * nt < M) {
            const uint4* src = reinterpret_cast<const uint4*>(xrow[nt] + bb * 64);
            xv[nt][0] = src[0];
            xv[nt][1] = src[1];
          } else {
            xv[nt][0] = make_uint4(0u, 0u, 0u, 0u);
            xv[nt][1] = xv[nt][0];
          }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int rt = 0; rt < RT; ++rt) {
            uint32_t a[4];
            decode_step(word(cur[rt][j >> 1], (j & 1) * 2), word(cur[rt][j >> 1], (j & 1) * 2 + 1),
                        word(cur[rt][2], j), a);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
              mma16816(acc[rt][nt], a, word(xv[nt][j >> 1], (j & 1) * 2), word(xv[nt][j >> 1], (j & 1) * 2 + 1));
          }
        }
        if constexpr (V >= 2)  // refill this slot once its words are decoded (same registers: no copies)
#pragma unroll
          for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w_if(wp[rt] + (bb + PIPE) * 96 + q * 32, bb + PIPE < nb);
      }
    }
    if constexpr (V >= 2)
      if (b == 0)  // eoff has landed with the first blocks: prefetch up to 32 escapes per row tile
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
          epre[rt] = ea[rt] + lane < eb[rt] ? ld_i64(esc + ea[rt] + lane) : make_int2(0, 0);
  }

  if constexpr (V <= 2)
    if (nb <= c0) asm volatile("cp.async.wait_group 0;" ::: "memory");  // nothing in flight past this point
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    const int r = r0 + rt;
    if constexpr (V == 1) {
      sc[rt] = scales[r];
      ea[rt] = eoff[(size_t)r * KB + kb0];
      eb[rt] = eoff[(size_t)r * KB + kb1];
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[rt][nt][i] *= sc[rt];
    const float wdec = sc[rt] * 3.0517578125e-05f;  // an escape's code decodes to 2^-15 * scale
    auto fix = [&](int2 en) {
      const int rl = en.x >> 16, k = en.x & 0xffff;
      if ((rl & 7) != g) return;
      const float w = __uint_as_float(((uint32_t)en.y) << 16);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int tok = 8 * nt + 2 * t + q;
          if (tok < M) {
            const float xv = __bfloat162float(xs[tok * RS + (k - k0)]);
            if (rl & 8) acc[rt][nt][2 + q] = fmaf(-wdec, xv, fmaf(w, xv, acc[rt][nt][2 + q]));
            else acc[rt][nt][q] = fmaf(-wdec, xv, fmaf(w, xv, acc[rt][nt][q]));
          }
        }
    };
    int e = ea[rt];
    if constexpr (V >= 2) {
      const int n = min(eb[rt] - e, 32);
      for (int i = 0; i < n; ++i)
        fix(make_int2(__shfl_sync(0xffffffffu, epre[rt].x, i), __shfl_sync(0xffffffffu, epre[rt].y, i)));
      e += n;
    }
    for (; e < eb[rt]; ++e) fix(esc[e]);
    if constexpr (V <= 3) {
      float* out = P + (size_t)s * M * N;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int tok = 8 * nt + 2 * t + q;
          if (tok < M) {
            out[(size_t)tok * N + r * 16 + g] = acc[rt][nt][q];
            out[(size_t)tok * N + r * 16 + g + 8] = acc[rt][nt][2 + q];
          }
        }
    } else {  // this CTA's partial -> red [tok][ROWS + 4] (its own region: no wait on the xs readers)
      float* red = reinterpret_cast<float*>(smem_raw + red_off(M, nbm, (nbm + CB - 1) / CB));
      const int RSR = nw * RT * 16 + 4, lr = (warp * RT + rt) * 16 + g;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int tok = 8 * nt + 2 * t + q;
          if (tok < M) {
            red[tok * RSR + lr] = acc[rt][nt][q];
            red[tok * RSR + lr + 8] = acc[rt][nt][2 + q];
          }
        }
    }
  }
  if constexpr (V == 4) {
    // Cluster rank q sums elements [q E / C, (q + 1) E / C) of the C partials (rank order: deterministic) and
    // stores them as partial s / C; the second cluster barrier keeps every CTA's buffer alive until all have read.
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    const int ROWS = nw * RT * 16, RSR = ROWS + 4, E = M * ROWS, q = s % C;
    const uint32_t red = smem_u32(smem_raw + red_off(M, nbm, (nbm + CB - 1) / CB));
    float* out = P + (size_t)(s / C) * M * N + (size_t)blockIdx.x * ROWS;
    for (int e = (int)((long long)q * E / C) + threadIdx.x; e < (int)((long long)(q + 1) * E / C); e += blockDim.x) {
      const int tok = e / ROWS, lr = e - tok * ROWS;
      const uint32_t a = red + 4u * (uint32_t)(tok * RSR + lr);
      float sum = 0.f;
      for (int rr = 0; rr < C; ++rr) {
        uint32_t ra;
        float v;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(rr));
        asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
        sum += v;
      }
      out[(size_t)tok * N + lr] = sum;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  }
}

// Bit-exact check of decode_step: one warp per block writes x0 * scale as bf16 into out [N, K] (escapes are
// patched afterwards from the list). Same fragment mapping as the GEMV.
__global__ void coded_decode(const uint4* __restrict__ Wc, const float* __restrict__ scales, int R, int KB, int K,
                             uint16_t* __restrict__ out) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  if (gw >= R * KB) return;
  const int r = gw / KB, c = gw - r * KB;
  const uint4* blk = Wc + (size_t)gw * 96 + lane;
  const uint4 v0 = blk[0], v1 = blk[32], v2 = blk[64];
  const uint4 cur[3] = {v0, v1, v2};
  const float sc = scales[r];
  for (int j = 0; j < 4; ++j) {
    uint32_t a[4];
    decode_step(word(cur[j >> 1], (j & 1) * 2), word(cur[j >> 1], (j & 1) * 2 + 1), word(cur[2], j), a);
    for (int ri = 0; ri < 4; ++ri)
      for (int q = 0; q < 2; ++q) {
        const uint32_t bits = (a[ri] >> (16 * q)) & 0xffffu;
        const float w = __uint_as_float(bits << 16) * sc;
        const uint32_t wb = __float_as_uint(w) >> 16;  // exact: w is a normal bf16
        const int row = r * 16 + g + 8 * (ri & 1), col = c * 64 + 16 * t + 4 * j + 2 * (ri >> 1) + q;
        out[(size_t)row * K + col] = (uint16_t)wb;
      }
  }
}

// Dynamic shared memory: the staged activations (V3: plus the zero row and one mbarrier per CB-block chunk).
__host__ __device__ inline int smem_bytes(int M, int KB, int S, int V, int rows) {
  const int nbm = (KB + S - 1) / S, RS = nbm * 64 + 8, nch = (nbm + CB - 1) / CB;
  return V <= 2 ? M * RS * 2 : V == 3 ? (((M + 1) * RS * 2 + 7) & ~7) + 8 * nch
                                       : red_off(M, nbm, nch) + M * (rows + 4) * 4;
}

template <int NT, int RT, int PIPE, int V>
static cudaError_t launch(int NW, int pdl, cudaStream_t st, const uint4* Wc, const float* sc, const int* eoff,
                          const int2* esc, const __nv_bfloat16* X, int xs, int M, float* P, int N, int KB, int S,
                          int pf, int prew, int C) {
  const int R = N / 16, smem = smem_bytes(M, KB, S, V, NW * RT * 16);
  auto k = coded_gemv<NT, RT, PIPE, V>;
  static int smem_set = 0;  // one per instantiation
  if (smem > smem_set) {
    const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
    smem_set = smem;
  }
  if (V == 4 && C > 8) {
    static bool np = false;
    if (!np) {
      const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      if (e != cudaSuccess) return e;
      np = true;
    }
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(R / (NW * RT), S, 1);
  cfg.blockDim = dim3(32 * NW, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na++].val.programmaticStreamSerializationAllowed = 1;
  }
  if (V == 4) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = 1;
    at[na].val.clusterDim.y = C;
    at[na++].val.clusterDim.z = 1;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  return cudaLaunchKernelEx(&cfg, k, Wc, sc, eoff, esc, X, xs, M, P, N, KB, S, pf, prew, C);
}

}  // namespace cdg

#define CDG_CASE(NT_, RT_, PIPE_, V_)                                                                         \
  if (NT == NT_ && RT == RT_ && PIPE == PIPE_ && V == V_)                                                    \
    e = cdg::launch<NT_, RT_, PIPE_, V_>((int)NW, (int)pdl, st, (const uint4*)Wc, (const float*)sc, (const int*)eoff, \
                                     (const int2*)esc, (const __nv_bfloat16*)X, (int)xs, (int)M, (float*)P,     \
                                     (int)N, (int)KB, (int)S, (int)pf, (int)prew, (int)C);

void cd_gemv(int64_t Wc, int64_t sc, int64_t eoff, int64_t esc, int64_t X, int64_t xs, int64_t M, int64_t P,
             int64_t N, int64_t KB, int64_t S, int64_t NW, int64_t RT, int64_t PIPE, int64_t pf, int64_t prew,
             int64_t pdl, int64_t V, int64_t C, int64_t stream) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  const int NT = (int)((M + 7) / 8) == 3 ? 4 : (int)((M + 7) / 8);
  if (NT > 4 || (N / 16) % (NW * RT) || S > KB || (V == 4 && (C < 1 || C > 16 || S % C)))
    throw std::runtime_error("cd_gemv shape");
  cudaError_t e = cudaErrorInvalidValue;
  CDG_INSTANCES
  if (e != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(cudaGetErrorString(e));
  }
}

void cd_decode(int64_t Wc, int64_t sc, int64_t N, int64_t K, int64_t out, int64_t stream) {
  const int R = (int)(N / 16), KB = (int)(K / 64);
  const int warps = R * KB;
  cdg::coded_decode<<<(warps + 7) / 8, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const uint4*)Wc, (const float*)sc, R, KB, (int)K, (uint16_t*)out);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
'''

CD_CPP = r'''
#include <cstdint>
void cd_gemv(int64_t Wc, int64_t sc, int64_t eoff, int64_t esc, int64_t X, int64_t xs, int64_t M, int64_t P,
             int64_t N, int64_t KB, int64_t S, int64_t NW, int64_t RT, int64_t PIPE, int64_t pf, int64_t prew,
             int64_t pdl, int64_t V, int64_t C, int64_t stream);
void cd_decode(int64_t Wc, int64_t sc, int64_t N, int64_t K, int64_t out, int64_t stream);
'''


CD_INSTANCES = ((1, 1, 2, 3),)
_CD_MOD = []


def cd_module(instances=CD_INSTANCES):
    if not _CD_MOD:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        cases = ' '.join(f'CDG_CASE({nt}, {rt}, {p}, {v})' for nt, rt, p, v in instances)
        src = CD_CUDA.replace('CDG_INSTANCES', cases)
        tag = hashlib.sha256(src.encode()).hexdigest()[:12]
        _CD_MOD.append(load_inline(name=f'muse_coded_dense_{arch}_{tag}', cpp_sources=CD_CPP, cuda_sources=src,
                                   functions=['cd_gemv', 'cd_decode'], verbose=False,
                                   extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo',
                                                      f'-gencode=arch=compute_{arch},code=sm_{arch}']))
    return _CD_MOD[0]


_CD_BITPOS = {}


def _cd_bitpos(device):
    """Code-word bit positions of each element's sign and d>>1, indexed [half, grp, q] (element k = q + 2 half)."""
    if device not in _CD_BITPOS:
        ps = torch.empty((2, 2, 2), dtype=torch.int64)
        pd = torch.empty((2, 2, 2), dtype=torch.int64)
        for half in range(2):
            for grp in range(2):
                for q in range(2):
                    k = q + 2 * half
                    ps[half, grp, q] = 8 * k + 7 if grp == 0 else 8 * k + 3
                    pd[half, grp, q] = 8 * k if grp == 0 else 8 * ((k + 3) & 3) + 4
        _CD_BITPOS[device] = (ps.to(device), pd.to(device))
    return _CD_BITPOS[device]


def cd_encode(w):
    """bf16 [N, K] -> dict(coded uint8 [R*KB*1536], scales f32 [R], eoff i32 [R*KB+1], esc i32 [E', 2], base i32 [R],
    N, K, nesc). Lossless for every bit pattern; runs on w's device."""
    assert w.dtype == torch.bfloat16 and w.dim() == 2
    N, K = w.shape
    assert N % 16 == 0 and K % 64 == 0 and K <= 65535, (N, K)
    R, KB = N // 16, K // 64
    dev = w.device
    b = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (b >> 7) & 0xFF
    hist = torch.zeros((R, 256), dtype=torch.int32, device=dev)
    hist.scatter_add_(1, e.view(R, 16 * K).long(), torch.ones((R, 16 * K), dtype=torch.int32, device=dev))
    hist[:, 0] = 0
    hist[:, 255] = 0
    cs = F.pad(hist.cumsum(1), (1, 0))                  
    win = (cs[:, 16:] - cs[:, :-16])[:, 1:240]          
    base = win.argmax(1).to(torch.int32) + 1
    d = e - base.repeat_interleave(16).view(N, 1)
    esc = (d < 0) | (d > 15)
    s = torch.where(esc, 0, (b >> 15) & 1)
    m = torch.where(esc, 0, b & 0x7F)
    d = torch.where(esc, 0, d)
    low = (((d & 1) << 7) | m).to(torch.uint8)
    
    lowp = low.view(R, 2, 8, KB, 4, 2, 2, 2, 2).permute(0, 3, 5, 2, 4, 6, 7, 1, 8).reshape(R, KB, 1024)
    ps, pd = _cd_bitpos(dev)
    sv = s.view(R, 2, 8, KB, 4, 4, 2, 2).to(torch.int64)
    dv = ((d >> 1) & 7).view(R, 2, 8, KB, 4, 4, 2, 2).to(torch.int64)
    ps = ps.view(1, 2, 1, 1, 1, 1, 2, 2)
    pd = pd.view(1, 2, 1, 1, 1, 1, 2, 2)
    words = ((sv << ps) | (dv << pd)).sum(dim=(1, 6, 7))          
    del sv, dv
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    nib = words.permute(0, 2, 1, 3, 4).contiguous().view(torch.uint8).view(R, KB, 512)
    coded = torch.cat([lowp, nib], 2).contiguous().view(-1)
    idx = esc.nonzero()
    rows, cols = idx[:, 0], idx[:, 1]
    key = (rows // 16) * KB + cols // 64
    order = key.argsort()
    rows, cols, key = rows[order], cols[order], key[order]
    meta = ((rows % 16) << 16) | cols
    entries = torch.stack([meta, b[rows, cols].long()], 1).to(torch.int32)
    nesc = entries.shape[0]
    if nesc == 0:
        entries = torch.zeros((1, 2), dtype=torch.int32, device=dev)
    eoff = F.pad(torch.bincount(key, minlength=R * KB).cumsum(0), (1, 0)).to(torch.int32)
    scales = torch.pow(2.0, (base - 112).double()).float()
    return dict(coded=coded, scales=scales.contiguous(), eoff=eoff.contiguous(), esc=entries.contiguous(),
                base=base, N=N, K=K, nesc=nesc)


def cd_decode(c):
    """The kernel's decode of a coded weight, escapes patched from the list -> bf16 [N, K]."""
    N, K = c['N'], c['K']
    out = torch.empty((N, K), dtype=torch.int16, device=c['coded'].device)
    cd_module().cd_decode(c['coded'].data_ptr(), c['scales'].data_ptr(), N, K, out.data_ptr(),
                          torch.cuda.current_stream().cuda_stream)
    if c['nesc']:
        en = c['esc'].long()
        rows = (en[:, 0] >> 16) + 16 * (torch.repeat_interleave(
            torch.arange(c['eoff'].numel() - 1, device=out.device), (c['eoff'][1:] - c['eoff'][:-1]).long()) // (K // 64))
        cols = en[:, 0] & 0xFFFF
        vals = en[:, 1]
        out[rows, cols] = torch.where(vals >= 2 ** 15, vals - 2 ** 16, vals).to(torch.int16)
    return out.view(torch.bfloat16)


CD_CB = 4  


def cd_smem(M, K, S, V=3, NW=8, RT=1):
    """Dynamic shared memory of a cd_gemv launch (the kernel's smem_bytes)."""
    nbm = -(-(K // 64) // S)
    RS, nch = nbm * 64 + 8, -(-nbm // CD_CB)
    if V <= 2:
        return M * RS * 2
    v3 = (((M + 1) * RS * 2 + 7) & ~7) + 8 * nch
    return v3 if V == 3 else ((v3 + 15) & ~15) + M * (NW * RT * 16 + 4) * 4


def cd_gemv(x, c, S, NW=8, RT=1, PIPE=2, pf=0, prew=1, pdl=True, V=3, C=1, out=None):
    """x [T, K] bf16 (16 B aligned rows) -> (P [S', T, N] FP32 partials, S'): S' = S, or S / C for V = 4 (the C
    splits of a cluster summed in the kernel)."""
    T, K = x.shape
    N = c['N']
    assert K == c['K'] and x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.stride(0) % 8 == 0
    if V != 4:
        C = 1
    S2 = S // C
    P = torch.empty((S2, T, N), device=x.device, dtype=torch.float32) if out is None else out
    cd_module().cd_gemv(c['coded'].data_ptr(), c['scales'].data_ptr(), c['eoff'].data_ptr(), c['esc'].data_ptr(),
                        x.data_ptr(), x.stride(0), T, P.data_ptr(), N, K // 64, S, NW, RT, PIPE, pf, prew, int(pdl), V, C,
                        torch.cuda.current_stream(x.device).cuda_stream)
    return P, S2



CD_ROWS = {(1, 'qkv_gate'): (8, 1, 2, 22, 1, 3, 1), (1, 'pool_router'): (8, 1, 2, 11, 1, 3, 1)}


def _cd_prepare(state):
    """Store lossless codes in state, so judge clones carry the matching encoded weights."""
    state['coded_dense'] = {}
    for name in sorted({name for _, name in CD_ROWS}):
        w = state[name]
        c = cd_encode(w)
        if not torch.equal(cd_decode(c).view(torch.int16), w.view(torch.int16)):
            raise RuntimeError(f"coded {name}: decode differs from the weight")
        state['coded_dense'][name] = c
        print(f"CODED {name} {tuple(w.shape)} escapes={c['nesc']} bytes={c['coded'].numel()}", flush=True)


def _cd_force(state):
    for (t, name) in CD_ROWS:
        state['proj_plan'][(t, name)] = 'custom'

"""Lossless 12-bit coding of the dense BF16 weights, decoded inside a tensor-core split-K GEMV (P10).

Format, per 16-row tile r (base exponent b_r chosen in prepare to cover the most weights):
  a weight with bf16 bits (s, e, m) and e in [b_r, b_r + 15] is stored as d = e - b_r (4 bits), s and m:
    low byte  = (d & 1) << 7 | m          (the bf16's own low byte with e's LSB replaced by d's)
    high code = s, d >> 1                 (4 bits, packed 8 per 32-bit word)
  every other weight (zeros, denormals, inf/nan, exponents outside the window) is an escape: its code is 0 and
  (row, col, exact bf16 bits) goes into a sparse list sorted by (row tile, 64-column block).
Decode (no table; SASS: SHF + 4 LOP3 + 4 PRMT per 8 weights): high byte = code | 0x38 gives the bf16 x0 = w * 2^(112 - b_r)
with exponent field 112 + d, so the MMA runs on x0 and the FP32 accumulator is scaled by 2^(b_r - 112) (exact).
An escape decodes to 2^(b_r - 127); its correction w*x - 2^(b_r-127)*x is added after the MMA loop.

Layout: R = N/16 x KB = K/64 blocks of 1536 B, row-major [R][KB] or k-major [KB][R]. In a block, lane l = 4g + t owns 16 B at 16l of each of three
512 B planes: low bytes of k16 steps 0-1, low bytes of steps 2-3, the four code words of steps 0-3. Step j's
mma.m16n8k16 A fragment of lane (g, t) holds rows g, g+8 at columns 64c + 16t + 4j + {0,1} (k-slots 2t, 2t+1)
and + {2,3} (k-slots 2t+8, 2t+9): the K permutation makes each lane's activations one contiguous 32 B run.

GEMV ABI = S8's gemv(): FP32 split partials P [S, T, N] (S balanced k-block ranges; V4 sums each cluster's C
splits in the kernel: S / C partials), PDL release at entry, weight registers preloaded before griddepcontrol.wait.
"""
import torch
import torch.nn.functional as F
DCP_CUDA = '\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <stdexcept>\n\nnamespace cdg {\n\n__device__ __forceinline__ void pdl_release() {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void pdl_wait() {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n#endif\n}\n\n// Predicated in the asm: the outputs count as written either way, so the compiler keeps no copy of the old value.\n__device__ __forceinline__ uint4 ld_w_if(const uint4* p, int pred) {\n  uint4 v;\n  asm volatile("{\\n.reg .pred q;\\nsetp.ne.b32 q, %5, 0;\\n"\n               "@q ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];\\n}"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "r"(pred));\n  return v;\n}\n\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t d;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));\n  return d;\n}\n\n// One k16 step: low-byte words lw0 (A-fragment elements a0-a3), lw1 (a4-a7) and the code word nw -> 4 A regs.\n// Code word byte k: bit 7 = sign of a_k, bits 0-2 = d>>1 of a_k; bit 3 = sign of a_{4+k};\n// bits 4-6 = d>>1 of a_{4+((k+1)&3)} (a 4-bit rotate brings them under the same 0x87 mask).\n__device__ __forceinline__ void decode_step(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t (&a)[4], uint32_t origin) {\n  const uint32_t ha = (nw & 0x87878787u) + origin;\n  const uint32_t hb = (__funnelshift_l(nw, nw, 4) & 0x87878787u) + origin;\n  a[0] = prmt(lw0, ha, 0x5140);\n  a[1] = prmt(lw0, ha, 0x7362);\n  a[2] = prmt(lw1, hb, 0x5140);\n  a[3] = prmt(lw1, hb, 0x7362);\n}\n\n__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ uint32_t word(const uint4& v, int i) {\n  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;\n}\n\n__device__ __forceinline__ float ld_f32(const float* p) {\n  float v;\n  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));\n  return v;\n}\n__device__ __forceinline__ int ld_i32(const int* p) {\n  int v;\n  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p));\n  return v;\n}\n__device__ __forceinline__ int2 ld_i64(const int2* p) {\n  int2 v;\n  asm volatile("ld.global.nc.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));\n  return v;\n}\n__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t n) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n) : "memory");\n}\n__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_wait0(uint32_t a) {  // phase 0 (each barrier is used once per launch)\n  uint32_t done;\n  do {\n    asm volatile("{\\n.reg .pred p;\\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\\nselp.u32 %0, 1, 0, p;\\n}"\n                 : "=r"(done) : "r"(a) : "memory");\n  } while (!done);\n}\n__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n               :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");\n}\n\n// Activations are staged in chunks of CB k-blocks (CB * 64 columns), one mbarrier each.\nconstexpr int CB = 4;\n\n// Shared memory: xs [M + 1][nb * 64 + 8] bf16 (row M = zeros), nch mbarriers, then (V4) the partial buffer, placed\n// after the largest split\'s region so mapa finds it at the same offset in every CTA of the cluster.\n__host__ __device__ inline int red_off(int M, int nb, int nch) {\n  return ((((M + 1) * (nb * 64 + 8) * 2 + 7) & ~7) + 8 * nch + 15) & ~15;\n}\n\n// grid (R / (NW * RT), S), NW warps; warp w of CTA x owns row tiles (x * NW + w) * RT + [0, RT) over the split\'s\n// k-blocks [s * KB / S, (s + 1) * KB / S). Block (r, kb) sits at Wc + (r * rs + kb * ks) * 1536 B (row-major\n// rs = KB, ks = 1; k-major rs = 1, ks = R). NT = n8 token tiles (M <= 8 NT).\n//   - scales and escape offsets load at entry (weight-side: before griddepcontrol.wait); the first 32 escapes of each\n//     row tile\'s range are prefetched one per lane after the first blocks and applied by warp shuffles\n//   - the last warp stages X[0:M, split] by bulk copies (one per row and chunk) completing on per-chunk mbarriers;\n//     each warp waits only for the chunk it is about to use (no CTA barrier in the loop); token rows past M read\n//     the zero row (row stride L + 8 bf16: conflict-free 16 B fragment loads)\n//   - each weight slot is refilled once its words are decoded (asm-predicated loads: no register copies)\n// V4: the C CTAs of a cluster (consecutive splits) sum their partials through distributed shared memory, so P gets\n// S / C partials instead of S (C = S: one).\ntemplate <int NT, int RT, int PIPE, int V>\n__global__ void __launch_bounds__(256) coded_gemv(const uint4* __restrict__ Wc, const uint32_t* __restrict__ scales,\n                                                  const int* __restrict__ eoff, const int2* __restrict__ esc,\n                                                  const __nv_bfloat16* __restrict__ X, int x_stride, int M,\n                                                  float* __restrict__ P, int N, int KB, int S, int rs, int ks,\n                                                  int prew, int C) {\n  extern __shared__ __align__(128) unsigned char smem_raw[];\n  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);\n  pdl_release();\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;\n  const int nw = blockDim.x >> 5, s = blockIdx.y;\n  const int kb0 = (int)((long long)s * KB / S), kb1 = (int)((long long)(s + 1) * KB / S), nb = kb1 - kb0;\n  const int r0 = (blockIdx.x * nw + warp) * RT;\n  const int L = nb * 64, RS = L + 8, k0 = kb0 * 64;\n  const int nbm = (KB + S - 1) / S;\n  uint32_t sc[RT];\n  int ea[RT], eb[RT];\n  int2 epre[RT];\n#pragma unroll\n  for (int rt = 0; rt < RT; ++rt) {\n    sc[rt] = ld_i32((const int*)(scales + r0 + rt));\n    ea[rt] = ld_i32(eoff + (size_t)(r0 + rt) * KB + kb0);\n    eb[rt] = ld_i32(eoff + (size_t)(r0 + rt) * KB + kb1);\n  }\n  const size_t bstep = (size_t)ks * 96;  // uint4s from one of a row tile\'s blocks to its next\n  const uint4* wp[RT];\n#pragma unroll\n  for (int rt = 0; rt < RT; ++rt) wp[rt] = Wc + ((size_t)(r0 + rt) * rs + (size_t)kb0 * ks) * 96 + lane;\n  uint4 buf[PIPE][RT][3];\n  if (!prew) pdl_wait();  // prew: the first PIPE blocks load before griddepcontrol.wait (overlap the producer)\n#pragma unroll\n  for (int p = 0; p < PIPE; ++p)\n#pragma unroll\n    for (int rt = 0; rt < RT; ++rt)\n#pragma unroll\n      for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w_if(wp[rt] + p * bstep + q * 32, p < nb);\n  const int nch = (nb + CB - 1) / CB;\n  const uint32_t mbar0 = smem_u32(smem_raw + (((size_t)(M + 1) * RS * 2 + 7) & ~(size_t)7));\n\n  if (prew) pdl_wait();\n  // All CTA threads stage the entire split while the weight preloads are in flight.\n  // One wait and CTA barrier, no per-chunk mbarrier initialization or producer warp.\n  for (int i=threadIdx.x; i<M*(L/8); i+=blockDim.x) {\n    const int row=i/(L/8), col=(i-row*(L/8))*8;\n    const uint32_t sa=smem_u32(xs+(size_t)row*RS+col);\n    const void* ga=X+(size_t)row*x_stride+k0+col;\n    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(ga));\n  }\n  asm volatile("cp.async.commit_group;" ::: "memory");\n  for (int i=threadIdx.x; i<L/8; i+=blockDim.x)\n    reinterpret_cast<uint4*>(xs+(size_t)M*RS)[i]=make_uint4(0u,0u,0u,0u);\n  asm volatile("cp.async.wait_group 0;" ::: "memory");\n  __syncthreads();\n  const __nv_bfloat16* xrow[NT];\n#pragma unroll\n  for (int nt = 0; nt < NT; ++nt) {\n    const int tok = g + 8 * nt;\n    xrow[nt] = xs + (size_t)(tok < M ? tok : M) * RS + 16 * t;\n  }\n\n  float acc[RT][NT][4];\n#pragma unroll\n  for (int rt = 0; rt < RT; ++rt)\n#pragma unroll\n    for (int nt = 0; nt < NT; ++nt)\n#pragma unroll\n      for (int i = 0; i < 4; ++i) acc[rt][nt][i] = 0.f;\n\n  for (int b = 0; b < nb; b += PIPE) {\n#pragma unroll\n    for (int p = 0; p < PIPE; ++p) {\n      const int bb = b + p;\n      if (bb < nb) {\n        uint4 xv[NT][2];\n#pragma unroll\n        for (int nt = 0; nt < NT; ++nt) {\n          const uint4* src = reinterpret_cast<const uint4*>(xrow[nt] + bb * 64);\n          xv[nt][0] = src[0];\n          xv[nt][1] = src[1];\n        }\n#pragma unroll\n        for (int j = 0; j < 4; ++j) {\n#pragma unroll\n          for (int rt = 0; rt < RT; ++rt) {\n            uint32_t a[4];\n            decode_step(word(buf[p][rt][j >> 1], (j & 1) * 2), word(buf[p][rt][j >> 1], (j & 1) * 2 + 1),\n                        word(buf[p][rt][2], j), a, sc[rt]);\n#pragma unroll\n            for (int nt = 0; nt < NT; ++nt)\n              mma16816(acc[rt][nt], a, word(xv[nt][j >> 1], (j & 1) * 2), word(xv[nt][j >> 1], (j & 1) * 2 + 1));\n          }\n        }\n#pragma unroll  // refill this slot once its words are decoded (same registers: no copies)\n        for (int rt = 0; rt < RT; ++rt)\n#pragma unroll\n          for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w_if(wp[rt] + (bb + PIPE) * bstep + q * 32, bb + PIPE < nb);\n      }\n    }\n    if (b == 0)  // eoff has landed with the first blocks: prefetch up to 32 escapes per row tile\n#pragma unroll\n      for (int rt = 0; rt < RT; ++rt)\n        epre[rt] = ea[rt] + lane < eb[rt] ? ld_i64(esc + ea[rt] + lane) : make_int2(0, 0);\n  }\n\n#pragma unroll\n  for (int rt = 0; rt < RT; ++rt) {\n    const int r = r0 + rt;\n\n    const float wdec = __uint_as_float(sc[rt] << 24);  // an escape\'s code decodes to 2^-15 * scale\n    auto fix = [&](int2 en) {\n      const int rl = en.x >> 16, k = en.x & 0xffff;\n      if ((rl & 7) != g) return;\n      const float w = __uint_as_float(((uint32_t)en.y) << 16);\n#pragma unroll\n      for (int nt = 0; nt < NT; ++nt)\n#pragma unroll\n        for (int q = 0; q < 2; ++q) {\n          const int tok = 8 * nt + 2 * t + q;\n          if (tok < M) {\n            const float xv = __bfloat162float(xs[tok * RS + (k - k0)]);\n            if (rl & 8) acc[rt][nt][2 + q] = fmaf(-wdec, xv, fmaf(w, xv, acc[rt][nt][2 + q]));\n            else acc[rt][nt][q] = fmaf(-wdec, xv, fmaf(w, xv, acc[rt][nt][q]));\n          }\n        }\n    };\n    int e = ea[rt];\n    const int n = min(eb[rt] - e, 32);\n    for (int i = 0; i < n; ++i)\n      fix(make_int2(__shfl_sync(0xffffffffu, epre[rt].x, i), __shfl_sync(0xffffffffu, epre[rt].y, i)));\n    for (e += n; e < eb[rt]; ++e) fix(esc[e]);\n    if constexpr (V == 3) {\n      float* out = P + (size_t)s * M * N;\n#pragma unroll\n      for (int nt = 0; nt < NT; ++nt)\n#pragma unroll\n        for (int q = 0; q < 2; ++q) {\n          const int tok = 8 * nt + 2 * t + q;\n          if (tok < M) {\n            out[(size_t)tok * N + r * 16 + g] = acc[rt][nt][q];\n            out[(size_t)tok * N + r * 16 + g + 8] = acc[rt][nt][2 + q];\n          }\n        }\n    } else {  // this CTA\'s partial -> red [tok][ROWS + 4] (its own region: no wait on the xs readers)\n      float* red = reinterpret_cast<float*>(smem_raw + red_off(M, nbm, (nbm + CB - 1) / CB));\n      const int RSR = nw * RT * 16 + 4, lr = (warp * RT + rt) * 16 + g;\n#pragma unroll\n      for (int nt = 0; nt < NT; ++nt)\n#pragma unroll\n        for (int q = 0; q < 2; ++q) {\n          const int tok = 8 * nt + 2 * t + q;\n          if (tok < M) {\n            red[tok * RSR + lr] = acc[rt][nt][q];\n            red[tok * RSR + lr + 8] = acc[rt][nt][2 + q];\n          }\n        }\n    }\n  }\n  if constexpr (V == 4) {\n    // Cluster rank q sums elements [q E / C, (q + 1) E / C) of the C partials (rank order: deterministic) and\n    // stores them as partial s / C; the second cluster barrier keeps every CTA\'s buffer alive until all have read.\n    asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");\n    const int ROWS = nw * RT * 16, RSR = ROWS + 4, E = M * ROWS, q = s % C;\n    const uint32_t red = smem_u32(smem_raw + red_off(M, nbm, (nbm + CB - 1) / CB));\n    float* out = P + (size_t)(s / C) * M * N + (size_t)blockIdx.x * ROWS;\n    for (int e = (int)((long long)q * E / C) + threadIdx.x; e < (int)((long long)(q + 1) * E / C); e += blockDim.x) {\n      const int tok = e / ROWS, lr = e - tok * ROWS;\n      const uint32_t a = red + 4u * (uint32_t)(tok * RSR + lr);\n      float sum = 0.f;\n      for (int rr = 0; rr < C; ++rr) {\n        uint32_t ra;\n        float v;\n        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(rr));\n        asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");\n        sum += v;\n      }\n      out[(size_t)tok * N + lr] = sum;\n    }\n    asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");\n  }\n}\n\n// Bit-exact check of decode_step: one warp per block writes x0 * scale as bf16 into out [N, K] (escapes are\n// patched afterwards from the list). Same fragment mapping as the GEMV.\n__global__ void coded_decode(const uint4* __restrict__ Wc, const uint32_t* __restrict__ scales, int R, int KB, int K,\n                             int rs, int ks, uint16_t* __restrict__ out) {\n  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;\n  if (gw >= R * KB) return;\n  const int r = gw / KB, c = gw - r * KB;\n  const uint4* blk = Wc + ((size_t)r * rs + (size_t)c * ks) * 96 + lane;\n  const uint4 v0 = blk[0], v1 = blk[32], v2 = blk[64];\n  const uint4 cur[3] = {v0, v1, v2};\n  const uint32_t sc = scales[r];\n  for (int j = 0; j < 4; ++j) {\n    uint32_t a[4];\n    decode_step(word(cur[j >> 1], (j & 1) * 2), word(cur[j >> 1], (j & 1) * 2 + 1), word(cur[2], j), a, sc);\n    for (int ri = 0; ri < 4; ++ri)\n      for (int q = 0; q < 2; ++q) {\n        const uint32_t bits = (a[ri] >> (16 * q)) & 0xffffu;\n        const uint32_t wb = bits;\n        const int row = r * 16 + g + 8 * (ri & 1), col = c * 64 + 16 * t + 4 * j + 2 * (ri >> 1) + q;\n        out[(size_t)row * K + col] = (uint16_t)wb;\n      }\n  }\n}\n\n// Dynamic shared memory: staged activations + zero row + one mbarrier per chunk (V4: + the partial buffer).\n__host__ __device__ inline int smem_bytes(int M, int KB, int S, int V, int rows) {\n  const int nbm = (KB + S - 1) / S, RS = nbm * 64 + 8, nch = (nbm + CB - 1) / CB;\n  return V == 3 ? (((M + 1) * RS * 2 + 7) & ~7) + 8 * nch : red_off(M, nbm, nch) + M * (rows + 4) * 4;\n}\n\ntemplate <int NT, int RT, int PIPE, int V>\nstatic cudaError_t launch(int NW, int pdl, cudaStream_t st, const uint4* Wc, const uint32_t* sc, const int* eoff,\n                          const int2* esc, const __nv_bfloat16* X, int xs, int M, float* P, int N, int KB, int S,\n                          int rs, int ks, int prew, int C) {\n  const int R = N / 16, smem = smem_bytes(M, KB, S, V, NW * RT * 16);\n  auto k = coded_gemv<NT, RT, PIPE, V>;\n  static int smem_set = 0;  // one per instantiation\n  if (smem > smem_set) {\n    const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    if (e != cudaSuccess) return e;\n    smem_set = smem;\n  }\n  if (V == 4 && C > 8) {\n    static bool np = false;\n    if (!np) {\n      const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n      if (e != cudaSuccess) return e;\n      np = true;\n    }\n  }\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(R / (NW * RT), S, 1);\n  cfg.blockDim = dim3(32 * NW, 1, 1);\n  cfg.dynamicSmemBytes = smem;\n  cfg.stream = st;\n  cudaLaunchAttribute at[2];\n  int na = 0;\n  if (pdl) {\n    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n    at[na++].val.programmaticStreamSerializationAllowed = 1;\n  }\n  if (V == 4) {\n    at[na].id = cudaLaunchAttributeClusterDimension;\n    at[na].val.clusterDim.x = 1;\n    at[na].val.clusterDim.y = C;\n    at[na++].val.clusterDim.z = 1;\n  }\n  cfg.attrs = at;\n  cfg.numAttrs = na;\n  return cudaLaunchKernelEx(&cfg, k, Wc, sc, eoff, esc, X, xs, M, P, N, KB, S, rs, ks, prew, C);\n}\n\n}  // namespace cdg\n\n#define CDG_CASE(NT_, RT_, PIPE_, V_)                                                                         \\\n  if (NT == NT_ && RT == RT_ && PIPE == PIPE_ && V == V_)                                                    \\\n    e = cdg::launch<NT_, RT_, PIPE_, V_>((int)NW, (int)pdl, st, (const uint4*)Wc, (const uint32_t*)sc, (const int*)eoff, \\\n                                     (const int2*)esc, (const __nv_bfloat16*)X, (int)xs, (int)M, (float*)P,     \\\n                                     (int)N, (int)KB, (int)S, (int)rs, (int)ks, (int)prew, (int)C);\n\nvoid cd_gemv(int64_t Wc, int64_t sc, int64_t eoff, int64_t esc, int64_t X, int64_t xs, int64_t M, int64_t P,\n             int64_t N, int64_t KB, int64_t S, int64_t NW, int64_t RT, int64_t PIPE, int64_t rs, int64_t ks,\n             int64_t prew, int64_t pdl, int64_t V, int64_t C, int64_t stream) {\n  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);\n  const int NT = (int)((M + 7) / 8) == 3 ? 4 : (int)((M + 7) / 8);\n  if (NT > 4 || (N / 16) % (NW * RT) || S > KB || (V == 4 && (C < 1 || C > 16 || S % C)))\n    throw std::runtime_error("cd_gemv shape");\n  cudaError_t e = cudaErrorInvalidValue;\n  CDG_INSTANCES\n  if (e != cudaSuccess) {\n    cudaGetLastError();\n    throw std::runtime_error(cudaGetErrorString(e));\n  }\n}\n\nvoid cd_decode(int64_t Wc, int64_t sc, int64_t N, int64_t K, int64_t rs, int64_t ks, int64_t out, int64_t stream) {\n  const int R = (int)(N / 16), KB = (int)(K / 64);\n  const int warps = R * KB;\n  cdg::coded_decode<<<(warps + 7) / 8, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(\n      (const uint4*)Wc, (const uint32_t*)sc, R, KB, (int)K, (int)rs, (int)ks, (uint16_t*)out);\n  cudaError_t e = cudaGetLastError();\n  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));\n}\n'
DCP_CPP = '\n#include <cstdint>\nvoid cd_gemv(int64_t Wc, int64_t sc, int64_t eoff, int64_t esc, int64_t X, int64_t xs, int64_t M, int64_t P,\n             int64_t N, int64_t KB, int64_t S, int64_t NW, int64_t RT, int64_t PIPE, int64_t rs, int64_t ks,\n             int64_t prew, int64_t pdl, int64_t V, int64_t C, int64_t stream);\nvoid cd_decode(int64_t Wc, int64_t sc, int64_t N, int64_t K, int64_t rs, int64_t ks, int64_t out, int64_t stream);\n'
DCP_INSTANCES = ((1, 1, 2, 3),)
_DCP_MOD = []

def dcp_module(instances=DCP_INSTANCES):
    if not _DCP_MOD:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        cases = ' '.join((f'CDG_CASE({nt}, {rt}, {p}, {v})' for nt, rt, p, v in instances))
        src = DCP_CUDA.replace('CDG_INSTANCES', cases)
        tag = hashlib.sha256(src.encode()).hexdigest()[:12]
        _DCP_MOD.append(load_inline(name=f'muse_coded_dense_{arch}_{tag}', cpp_sources=DCP_CPP, cuda_sources=src, functions=['cd_gemv', 'cd_decode'], verbose=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', f'-gencode=arch=compute_{arch},code=sm_{arch}']))
    return _DCP_MOD[0]
_DCP_BITPOS = {}

def _dcp_bitpos(device):
    """Code-word bit positions of each element's sign and d>>1, indexed [half, grp, q] (element k = q + 2 half)."""
    if device not in _DCP_BITPOS:
        ps = torch.empty((2, 2, 2), dtype=torch.int64)
        pd = torch.empty((2, 2, 2), dtype=torch.int64)
        for half in range(2):
            for grp in range(2):
                for q in range(2):
                    k = q + 2 * half
                    ps[half, grp, q] = 8 * k + 7 if grp == 0 else 8 * k + 3
                    pd[half, grp, q] = 8 * k if grp == 0 else 8 * (k + 3 & 3) + 4
        _DCP_BITPOS[device] = (ps.to(device), pd.to(device))
    return _DCP_BITPOS[device]

def dcp_encode(w, kmajor=False):
    """bf16 [N, K] -> dict(coded uint8 [R*KB*1536], scales f32 [R], eoff i32 [R*KB+1], esc i32 [E', 2], base i32 [R],
    N, K, nesc, rs, ks). Lossless for every bit pattern; runs on w's device. Blocks are row-major [R][KB] (block
    strides rs = KB, ks = 1) or, with kmajor, [KB][R] (rs = 1, ks = R: the row tiles a split's warps read at one
    step sit side by side)."""
    assert w.dtype == torch.bfloat16 and w.dim() == 2
    N, K = w.shape
    assert N % 16 == 0 and K % 64 == 0 and (K <= 65535), (N, K)
    R, KB = (N // 16, K // 64)
    dev = w.device
    b = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    e = b >> 7 & 255
    hist = torch.zeros((R, 256), dtype=torch.int32, device=dev)
    hist.scatter_add_(1, e.view(R, 16 * K).long(), torch.ones((R, 16 * K), dtype=torch.int32, device=dev))
    hist[:, 0] = 0
    hist[:, 255] = 0
    cs = F.pad(hist.cumsum(1), (1, 0))
    win = (cs[:, 16:] - cs[:, :-16])[:, 1:240]
    base = win[:, 1::2].argmax(1).to(torch.int32) * 2 + 2
    d = e - base.repeat_interleave(16).view(N, 1)
    esc = (d < 0) | (d > 15)
    s = torch.where(esc, 0, b >> 15 & 1)
    m = torch.where(esc, 0, b & 127)
    d = torch.where(esc, 0, d)
    low = ((d & 1) << 7 | m).to(torch.uint8)
    lowp = low.view(R, 2, 8, KB, 4, 2, 2, 2, 2).permute(0, 3, 5, 2, 4, 6, 7, 1, 8).reshape(R, KB, 1024)
    ps, pd = _dcp_bitpos(dev)
    sv = s.view(R, 2, 8, KB, 4, 4, 2, 2).to(torch.int64)
    dv = (d >> 1 & 7).view(R, 2, 8, KB, 4, 4, 2, 2).to(torch.int64)
    ps = ps.view(1, 2, 1, 1, 1, 1, 2, 2)
    pd = pd.view(1, 2, 1, 1, 1, 1, 2, 2)
    words = (sv << ps | dv << pd).sum(dim=(1, 6, 7))
    del sv, dv
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    nib = words.permute(0, 2, 1, 3, 4).contiguous().view(torch.uint8).view(R, KB, 512)
    coded = torch.cat([lowp, nib], 2)
    coded = (coded.transpose(0, 1) if kmajor else coded).contiguous().view(-1)
    idx = esc.nonzero()
    rows, cols = (idx[:, 0], idx[:, 1])
    key = rows // 16 * KB + cols // 64
    order = key.argsort()
    rows, cols, key = (rows[order], cols[order], key[order])
    meta = rows % 16 << 16 | cols
    entries = torch.stack([meta, b[rows, cols].long()], 1).to(torch.int32)
    nesc = entries.shape[0]
    if nesc == 0:
        entries = torch.zeros((1, 2), dtype=torch.int32, device=dev)
    eoff = F.pad(torch.bincount(key, minlength=R * KB).cumsum(0), (1, 0)).to(torch.int32)
    origin = ((base >> 1) * 16843009).to(torch.int32)
    return dict(coded=coded, origin=origin.contiguous(), eoff=eoff.contiguous(), esc=entries.contiguous(), rs=1 if kmajor else KB, ks=R if kmajor else 1, base=base, N=N, K=K, nesc=nesc)

def dcp_decode(c):
    """The kernel's decode of a coded weight, escapes patched from the list -> bf16 [N, K]."""
    N, K = (c['N'], c['K'])
    out = torch.empty((N, K), dtype=torch.int16, device=c['coded'].device)
    dcp_module().cd_decode(c['coded'].data_ptr(), c['origin'].data_ptr(), N, K, c['rs'], c['ks'], out.data_ptr(), torch.cuda.current_stream().cuda_stream)
    if c['nesc']:
        en = c['esc'].long()
        rows = (en[:, 0] >> 16) + 16 * (torch.repeat_interleave(torch.arange(c['eoff'].numel() - 1, device=out.device), (c['eoff'][1:] - c['eoff'][:-1]).long()) // (K // 64))
        cols = en[:, 0] & 65535
        vals = en[:, 1]
        out[rows, cols] = torch.where(vals >= 2 ** 15, vals - 2 ** 16, vals).to(torch.int16)
    return out.view(torch.bfloat16)
DCP_CB = 4

def dcp_smem(M, K, S, V=3, NW=8, RT=1):
    """Dynamic shared memory of a cd_gemv launch (the kernel's smem_bytes)."""
    nbm = -(-(K // 64) // S)
    RS, nch = (nbm * 64 + 8, -(-nbm // DCP_CB))
    v3 = ((M + 1) * RS * 2 + 7 & ~7) + 8 * nch
    return v3 if V == 3 else (v3 + 15 & ~15) + M * (NW * RT * 16 + 4) * 4

def dcp_gemv(x, c, S, NW=8, RT=1, PIPE=2, prew=1, pdl=True, V=3, C=1, out=None):
    """x [T, K] bf16 (16 B aligned rows) -> (P [S', T, N] FP32 partials, S'): S' = S, or S / C for V = 4 (the C
    splits of a cluster summed in the kernel)."""
    T, K = x.shape
    N = c['N']
    assert K == c['K'] and x.dtype == torch.bfloat16 and (x.stride(1) == 1) and (x.stride(0) % 8 == 0)
    if V != 4:
        C = 1
    S2 = S // C
    P = torch.empty((S2, T, N), device=x.device, dtype=torch.float32) if out is None else out
    dcp_module().cd_gemv(c['coded'].data_ptr(), c['origin'].data_ptr(), c['eoff'].data_ptr(), c['esc'].data_ptr(), x.data_ptr(), x.stride(0), T, P.data_ptr(), N, K // 64, S, NW, RT, PIPE, c['rs'], c['ks'], prew, int(pdl), V, C, torch.cuda.current_stream(x.device).cuda_stream)
    return (P, S2)


_DCP_ROWS = {(1, 'qkv3'): (4, 32), (1, 'o'): (8, 8)}

def _dcp_prepare(state):
    state['direct_coded']={}
    for t,name in _DCP_ROWS:
        if name=='qkv3':state[name]=state['qkv_gate'][:QKV3_SIZE]
        w=state[name]
        c=dcp_encode(w,kmajor=True)
        if not torch.equal(dcp_decode(c).view(torch.int16),w.view(torch.int16)):
            raise RuntimeError(f'direct {name}: bit-exact decode failed')
        state['direct_coded'][name]=c

def _dcp_force(state):
    for t,name in _DCP_ROWS:state['proj_plan'][(t,name)]='custom'

def _dcp_project(x,name,state):
    NW,S=_DCP_ROWS[(x.shape[0],name)]
    return dcp_gemv(x,state['direct_coded'][name],S,NW=NW,RT=1,PIPE=2,prew=int(name!='o'),pdl=PDL,V=3)



LT_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cstring>
namespace decode_lt {
static cublasLtHandle_t handle() {
  static cublasLtHandle_t h=[] { cublasLtHandle_t p; TORCH_CHECK(cublasLtCreate(&p)==CUBLAS_STATUS_SUCCESS,"cublasLtCreate"); return p; }();
  return h;
}
struct Layout {
  cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a,b,c;
  Layout(const torch::Tensor& w,const torch::Tensor& x,const torch::Tensor& y) {
    int64_t n=y.size(1),m=x.size(0),k=x.size(1);
    TORCH_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"Lt desc");
    cublasOperation_t ta=CUBLAS_OP_T;
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))==CUBLAS_STATUS_SUCCESS,"Lt transpose");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,w.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt A layout");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,x.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt B layout");
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n)==CUBLAS_STATUS_SUCCESS,"Lt C layout");
  }
  ~Layout(){cublasLtMatmulDescDestroy(op);cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(c);}
};
constexpr int CAP=32; constexpr size_t WORK_BYTES=32*1024*1024;
}
torch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out) {
  using namespace decode_lt; Layout layout(W,X,Out); cublasLtMatmulPreference_t pref;
  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS,"Lt preference");
  size_t workspace=WORK_BYTES;
  TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace))==CUBLAS_STATUS_SUCCESS,"Lt workspace");
  cublasLtMatmulHeuristicResult_t results[CAP]; int count=0;
  auto status=cublasLtMatmulAlgoGetHeuristic(handle(),layout.op,layout.a,layout.b,layout.c,layout.c,pref,CAP,results,&count);
  cublasLtMatmulPreferenceDestroy(pref); TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt heuristics");
  auto codes=torch::empty({count,(int64_t)sizeof(cublasLtMatmulAlgo_t)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU));
  for(int i=0;i<count;++i){TORCH_CHECK(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=WORK_BYTES,"Lt invalid heuristic");std::memcpy(codes.data_ptr<uint8_t>()+i*sizeof(cublasLtMatmulAlgo_t),&results[i].algo,sizeof(cublasLtMatmulAlgo_t));}
  return codes;
}
void lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index) {
  using namespace decode_lt; TORCH_CHECK(index>=0 && index<Algos.size(0) && Work.nbytes()>=WORK_BYTES,"Lt invalid algorithm");
  Layout layout(W,X,Out); cublasLtMatmulAlgo_t algo; std::memcpy(&algo,Algos.data_ptr<uint8_t>()+index*sizeof(algo),sizeof(algo));
  const float alpha=1.f,beta=0.f;
  auto status=cublasLtMatmul(handle(),layout.op,&alpha,W.data_ptr(),layout.a,X.data_ptr(),layout.b,&beta,Out.data_ptr(),layout.c,Out.data_ptr(),layout.c,&algo,Work.data_ptr(),WORK_BYTES,at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt matmul");
}
"""
LT_CPP_SRC = r"""
torch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out);
void lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index);
"""
_LT_MOD = None

def _lt_module():
    global _LT_MOD
    if _LT_MOD is None:
        from torch.utils.cpp_extension import load_inline
        import hashlib
        tag = hashlib.sha1(LT_CUDA_SRC.encode()).hexdigest()[:12]
        _LT_MOD = load_inline(name=f"muse_decode_lt_{tag}", cpp_sources=LT_CPP_SRC, cuda_sources=LT_CUDA_SRC,
                              functions=["lt_options", "lt_run"], extra_cuda_cflags=["-O3", "-std=c++20"],
                              extra_ldflags=["-lcublasLt"], verbose=False)
    return _LT_MOD


def _dense_project(x, w, out_dtype):
    if out_dtype == torch.float32:
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    return F.linear(x, w)


def _project_variant(x, w, name, state, variant, out_dtype=torch.bfloat16):
    if (x.shape[0],name) in _DCP_ROWS:
        return _dcp_project(x,name,state)
    cfg = CD_ROWS.get((x.shape[0], name))
    if cfg is not None and 'coded_dense' in state:
        NW, RT, PIPE, S, PREW, V, C = cfg
        return cd_gemv(x, state['coded_dense'][name], S, NW=NW, RT=RT, PIPE=PIPE, prew=PREW,
                       pdl=PDL, V=V, C=C)
    if isinstance(variant,tuple) and variant[0]=='vdsk':return _vdsk_project(x,w,variant)
    if isinstance(variant,tuple) and variant[0]=='short_lt':
        _,nn,ordinal=variant
        bank=state['measured_lt'];weight,codes=bank['layouts'][(x.shape[0],name,nn)]
        out=torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=out_dtype)
        bank['mod'].run(x,weight,out,bank['workspace'],codes,ordinal,nn,out_dtype==torch.float32)
        return out.unsqueeze(0),1
    if variant == "custom":
        return gemv(x, w, out_dtype=out_dtype)
    if variant == "cublas":
        return _dense_project(x, w, out_dtype).unsqueeze(0), 1
    mod = state["lt_mod"]
    out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
    mod.lt_run(w, x, out, state["lt_workspace"], state["lt_algos"][(name, x.shape[0])], int(variant[1]))
    return out.unsqueeze(0), 1


def project(x, w, name, state, out_dtype=torch.bfloat16):
    variant = state.get("proj_plan", {}).get((x.shape[0], name), "custom")
    return _project_variant(x, w, name, state, variant, out_dtype=out_dtype)


def _materialize(value):
    p, s = value
    return p.sum(0) if s > 1 else p[0]


def _measure_graph(fn, flush, repeats=3):
    for _ in range(2): fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    vals=[]
    for _ in range(repeats):
        flush.sum(); a.record(); graph.replay(); b.record(); b.synchronize(); vals.append(a.elapsed_time(b)*1000.0)
    del graph, out
    return float(__import__('statistics').median(vals))

_TUNE_TOKENS = (1, 8, 16, 32, 128, 256)
_FA3_SPLITS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
_FA3_PACK = (None, True, False)


def _fa3_on_device(device):
    return torch.cuda.get_device_capability(device) == (9, 0)


def _fa3_call(q, gate, data, setting):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
    cfg = data["config"]; batch, L = cfg["batch"], cfg["q_len"]
    splits, pack = setting
    metadata = get_scheduler_metadata(
        batch_size=batch, max_seqlen_q=L, max_seqlen_k=data["page_table"].shape[1] * cfg["page_size"],
        num_heads=HQ, num_heads_k=HKV, headdim=HD, cache_seqlens=data["cache_seqlens"], qkv_dtype=q.dtype,
        cu_seqlens_q=data["cu_seqlens_q"], page_size=cfg["page_size"], causal=True,
        window_size=(cfg["window"] - 1, 0) if cfg["window"] else (-1, -1), num_splits=splits, pack_gqa=pack)
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1,1).expand(batch, HKV),
                       v_descale=data["v_scale"].reshape(1,1).expand(batch, HKV))
    out = flash_attn_with_kvcache(
        q=q, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=L,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(cfg["window"] - 1, 0) if cfg["window"] else (-1, -1), scheduler_metadata=metadata,
        num_splits=splits, pack_gqa=pack, **descale)
    if isinstance(out, tuple): out = out[0]
    o = torch.empty((batch * L, Q_SIZE), device=q.device, dtype=torch.bfloat16)
    _launch(_norm_gate_kernel, (batch * L,), out.reshape(batch * L, HQ, HD), gate, o, RMS_EPS,
            HQ=HQ, D=HD, num_warps=4)
    return o


def attention_fa3(q, gate, data, state):
    setting = state.get("fa3_plan", {}).get((data["config"]["batch"], data["config"]["window"]), (0, None))
    return _fa3_call(q, gate, data, setting)



PROJ_PINS = {(1, 'qkv_gate'): 'custom', (1, 'pool_router'): 'custom', (8, 'qkv_gate'): 'custom', (8, 'pool_router'): 'custom', (16, 'qkv_gate'): 'custom', (16, 'pool_router'): 'custom', (32, 'qkv_gate'): 'custom', (32, 'pool_router'): 'custom', (1, 'unpooler_slice'): 'custom', (8, 'unpooler_slice'): 'custom', (16, 'unpooler'): 'custom', (32, 'unpooler'): 'custom', (8, 'o'): 'custom', (16, 'o'): 'custom', (32, 'o'): 'custom', (128, 'qkv_gate'): ('vdsk', (128, 64, 64, 8, 4, 1), 3), (128, 'o'): ('lt', 0), (128, 'pool_router'): 'cublas', (128, 'unpooler'): ('short_lt', 1, 0), (256, 'qkv_gate'): ('lt', 0), (256, 'o'): 'custom', (256, 'pool_router'): 'custom', (256, 'unpooler'): ('short_lt', 0, 0)}


def _tune_projections(state, flush, device):
    plan={}
    if _fa3_on_device(device) and _best5_left(10):
        try:
            state["lt_mod"] = _lt_module(); state["lt_workspace"] = torch.empty((32<<20,), dtype=torch.uint8, device=device)
        except Exception as exc:
            print(f"PROJ_LT_REJECT module reason={str(exc).splitlines()[0][:100]}", flush=True)
            state["lt_mod"] = None
    names=("qkv_gate", "o", "pool_router", "unpooler_slice", "unpooler")
    for t in _TUNE_TOKENS:
        for name in names:
            if name == "unpooler_slice" and t > 8 or name == "unpooler" and t <= 8: continue
            if name == "pool_router" and t <= 32 and PROJ_PINS.get((t, name), "custom") == "custom":
                plan[(t, name)] = "custom"
                print(f"PROJ_TUNE t={t} name={name} best=custom receipt=V9_union", flush=True)
                continue
            if not _best5_left():
                plan[(t,name)]="custom";continue
            w=state[name]; out_dtype=torch.float32 if name == "pool_router" else torch.bfloat16
            x=torch.randn((t,w.shape[1]), device=device, dtype=torch.bfloat16)
            state['_proj_stream']=torch.randn((t,HIDDEN),device=device,dtype=torch.float32)
            state['_proj_data']={
                'positions':torch.arange(t,device=device,dtype=torch.int64),
                'cache_loc':torch.arange(t,device=device,dtype=torch.int64),
                'k_cache':torch.empty((t,1,HKV,HD),device=device,dtype=FP8),
                'v_cache':torch.empty((t,1,HKV,HD),device=device,dtype=FP8),
                'k_scale':torch.ones((),device=device),'v_scale':torch.ones((),device=device),
                'local_start':torch.zeros((),device=device,dtype=torch.int32),
                'config':dict(tokens=t,rope_theta=ROPE_THETA,qk_scale_factor=5.4730064863875)}
            variants=["custom", "cublas"]
            variants += [('vdsk',cfg,S) for cfg,S in VDSK_SHORTLIST.get((t,name),())]
            if _fa3_on_device(device) and state.get("lt_mod") is not None and name != "pool_router":
                try:
                    out=torch.empty((t,w.shape[0]), device=device, dtype=torch.bfloat16)
                    state.setdefault("lt_algos", {})[(name,t)] = state["lt_mod"].lt_options(w,x,out)
                    variants += [("lt",i) for i in range(int(state["lt_algos"][(name,t)].shape[0]))]
                except Exception as exc:
                    print(f"PROJ_LT_REJECT name={name} t={t} reason={str(exc).splitlines()[0][:100]}", flush=True)
            if (t, name) in PROJ_PINS:
                pin = PROJ_PINS[(t, name)]
                variants = ['custom' if isinstance(pin, tuple) and pin[0] == 'short_lt' else pin]
            scores=[]
            want=None
            for variant in variants:
                if not _best5_left():break
                fn=lambda variant=variant: _materialize(_project_variant(x,w,name,state,variant,out_dtype))
                try:
                    y=fn(); torch.cuda.synchronize();
                    want = y.float() if want is None else want  
                    rel=((y.float()-want).norm()/want.norm().clamp_min(1e-12)).item()
                    if not torch.isfinite(y).all().item() or rel > 0.01:
                        print(f"PROJ_REJECT name={name} t={t} variant={variant} rel={rel:.4g}", flush=True); continue
                    scores.append((variant,_measure_graph(
                        lambda variant=variant: _projection_consumer(x,w,name,state,variant,out_dtype),
                        flush,repeats=3)))
                except Exception as exc:
                    print(f"PROJ_REJECT name={name} t={t} variant={variant} reason={str(exc).splitlines()[0][:100]}", flush=True)
            if not scores:
                print(f"PROJ_REJECT name={name} t={t} reason=no_valid_variant fallback=custom", flush=True)
                plan[(t,name)] = "custom"
                continue
            best=min(scores,key=lambda z:z[1]); plan[(t,name)]=best[0]
            print(f"PROJ_TUNE t={t} name={name} best={best[0]} us={best[1]:.2f} all={scores}",flush=True)
    return plan


ATTN_PORT_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <cooperative_groups.h>
#include <stdexcept>

namespace cg = cooperative_groups;

#define DEVI __device__ __forceinline__

namespace {

constexpr int D = 64;      // head dim
constexpr int G = 8;       // query heads per KV head
constexpr int T = 64;      // tokens per pipeline tile
constexpr int NTHR = 128;  // 4 warps
constexpr int ROWB = 128;  // bytes of one token's K (or V) for a pair of KV heads
constexpr int TILE_BYTES = T * ROWB;

struct Params {
  const __nv_bfloat16* qkv;
  const int64_t* positions;
  const int64_t* cache_loc;
  const int* page_table;
  const int* seq_lens;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;   // nullptr: no RoPE
  float* part_o;           // [B, KVG, NC, L*16, 64]   cluster partials (NC > 1 only)
  float* part_ml;          // [B, KVG, NC, L*16, 2]
  int* counters;           // [B, KVG, CS] arrivals per cluster rank, left at zero
  float* part1;            // [B, KVG, NS, NSL, L*16, 68] split partials when dsmem == 0
  int dsmem;               // 1: split partials merge through distributed shared memory
  __nv_bfloat16* out;      // [B, L, HQ*64]
  int batch;
  int qkv_stride;
  int hq, hkv;
  int max_pages;
  int ps, ps_log2;
  int window;
  int ns;                  // splits per request = NC * CS
  int cs;                  // cluster size
  long long row_bytes;     // hkv * 64
  float scale_log2;        // sm_scale * qk_scale_factor * log2(e)
};

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

DEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

DEVI uint32_t cvt_e4m3x2(uint32_t v) {
  uint32_t r;
  uint16_t h = static_cast<uint16_t>(v & 0xffffu);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(h));
  return r;
}

DEVI uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

DEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

DEVI uint32_t cluster_addr(const void* p, int rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));
  return r;
}
DEVI float ld_cluster_f32(uint32_t a) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(a) : "memory");
  return v;
}
DEVI float2 ld_cluster_f32x2(uint32_t a) {
  float2 v;
  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");
  return v;
}
DEVI void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.
DEVI int swz(int t) { return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2); }

DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int L>
struct WarpState {
  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp
  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims
  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims
  float m[MTW][2];                               // running max (log2 domain, scaled)
  float l[MTW][2];                               // running sum (thread-partial)
};

// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.
// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);
//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).
template <int L, int NTC>
DEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,
                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2) {
  constexpr int MTW = WarpState<L>::MTW;
  constexpr int KK = NTC / 2;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, q4 = lane & 3;

  float s[MTW][NTC][4];
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
    for (int n = 0; n < NTC; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;

  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,
  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q's fragments).
#pragma unroll
  for (int n = 0; n < NTC; ++n) {
    const int t = tb + 8 * n + g;
    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));
    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint32_t b0 = cvt_e4m3x2(w[ks]);
      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);
#pragma unroll
      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);
    }
  }

  if (need_mask) {
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));
        const int qp = seq + j;
        const int lo = window ? (qp - window + 1) : -2147483647;
        const int hi = min(k_hi, qp + 1);
#pragma unroll
        for (int n = 0; n < NTC; ++n) {
          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);
          if (kp < lo || kp >= hi) s[mt][n][e] = -INFINITY;
        }
      }
    }
  }

  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).
  uint32_t pa[MTW][KK][4];
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      if (L == 1 && rh == 1) continue;
      float mx = -INFINITY;
#pragma unroll
      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);
      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
      const float corr = ex2(st.m[mt][rh] - m_use);
      st.m[mt][rh] = m_new;
      st.l[mt][rh] *= corr;
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) {
        st.o[mt][jn][2 * rh] *= corr;
        st.o[mt][jn][2 * rh + 1] *= corr;
      }
      float sum = 0.f;
#pragma unroll
      for (int n = 0; n < NTC; ++n) {
        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));
        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));
        s[mt][n][2 * rh] = p0;
        s[mt][n][2 * rh + 1] = p1;
        sum += p0 + p1;
      }
      st.l[mt][rh] += sum;
    }
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);
      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);
      if (L == 1) {
        pa[mt][kk][1] = 0u;
        pa[mt][kk][3] = 0u;
      } else {
        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);
        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);
      }
    }
  }

  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a
  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
    const int t0 = tb + 16 * kk + 2 * q4;
    const int coff = ((4 * h + (g >> 1))) ;
    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));
    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));
    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));
    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;
      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;
      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);
      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);
      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};
      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};
#pragma unroll
      for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);
    }
  }
}

template <int L, int STAGES>
__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  constexpr int MTW = WarpState<L>::MTW;
  constexpr int NT = (L == 1) ? 4 : 8;  // n-tiles per warp per tile
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sK = smem;
  uint8_t* sV = smem + STAGES * TILE_BYTES;
  uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;  // new tokens: 16 rows of K, 16 of V
  uint8_t* sNV = sNK + 16 * ROWB;
  __half* sQ = reinterpret_cast<__half*>(sNV + 16 * ROWB);  // [2][L*8][64]
  float2* sCS = reinterpret_cast<float2*>(sQ + 2 * L * G * D);  // [L][32]
  __nv_bfloat16* sG = reinterpret_cast<__nv_bfloat16*>(sCS + L * 32);  // gates of this CTA's rows [L*16/CS][64]

  const int split = blockIdx.x, kvg = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int h = warp & 1, wh = warp >> 1;
  const bool last = split == p.ns - 1;

  const int seq = p.seq_lens[b];
  const int lo = p.window ? max(0, seq - p.window + 1) : 0;
  const int n_cached = seq - lo;
  int chunk = (n_cached + p.ns - 1) / p.ns;
  chunk = (chunk + 15) & ~15;
  const int c_lo = min(lo + split * chunk, seq);
  const int c_hi = min(c_lo + chunk, seq);
  const int ntiles = (c_hi - c_lo + T - 1) / T;

  const int* pt = p.page_table + (long long)b * p.max_pages;
  const int lc = tid & 7;      // 16-byte chunk of the row this thread copies
  const int lt = tid >> 3;     // first token (of 4: lt + 16k)
  const uint8_t* kbase = p.k_cache + kvg * ROWB + lc * 16;
  const uint8_t* vbase = p.v_cache + kvg * ROWB + lc * 16;

  auto load_rows = [&](int tile, int (&rows)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = c_lo + tile * T + lt + 16 * k;
      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;
    }
  };
  auto issue = [&](int slot, const int (&rows)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = lt + 16 * k;
      const int off = slot * TILE_BYTES + t * ROWB + ((lc ^ swz(t)) << 4);
      const long long r = rows[k] < 0 ? 0 : rows[k];
      cp_async16(smem_u32(sK + off), kbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);
      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);
    }
  };

  // Every independent global load of the prologue goes out first (page-table rows, queries,
  // new keys/values, positions, cache rows), so their latencies overlap.
  int rows[STAGES][4];
#pragma unroll
  for (int s = 0; s < STAGES; ++s) load_rows(s, rows[s]);
  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)
  __nv_bfloat162 qv[QR];
#pragma unroll
  for (int i = 0; i < QR; ++i) {
    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;
    qv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +
                                                      (kvg * 16 + hh) * D + 2 * lane);
  }
  __nv_bfloat162 nv[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)
  long long nloc[L];
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;
      nv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +
                                                        (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);
      nloc[i] = p.cache_loc[b * L + j];
    }
  }
  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;
  float ang[CSN];
#pragma unroll
  for (int c = 0; c < CSN; ++c) {
    const int i = tid + c * NTHR;
    ang[c] = 0.f;
    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];
  }
  const float ksc = *p.k_scale, vsc = *p.v_scale;

  // Gate rows this CTA finalizes (rows rank, rank + CS, ...): 8 x 16 bytes each, with tile 0.
  const bool gated = p.qkv_stride > (p.hq + 2 * p.hkv) * D;  // else ungated output (P7 gate split)
  if (gated) {
    const int rank = split % p.cs;
    const int nrows = (L * 16 - rank + p.cs - 1) / p.cs;
    for (int i = tid; i < nrows * 8; i += NTHR) {
      const int k = i >> 3, c = i & 7, rr = rank + k * p.cs, j = rr >> 4, hh = rr & 15;
      cp_async16(smem_u32(sG + k * D + c * 8),
                 p.qkv + (long long)(b * L + j) * p.qkv_stride + (p.hq + 2 * p.hkv + kvg * 16 + hh) * D + c * 8, 16);
    }
  }
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ntiles) issue(s, rows[s]);
    cp_commit();
  }
  int rows_next[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) rows_next[k] = rows[STAGES - 1][k];

  // RoPE angles for the block's tokens.
  if (p.inv_freq) {
#pragma unroll
    for (int c = 0; c < CSN; ++c) {
      const int i = tid + c * NTHR;
      if (i < L * 32) {
        float sn, cs;
        sincosf(ang[c], &sn, &cs);
        sCS[i] = make_float2(cs, sn);
      }
    }
  }
  if (last) {  // zero the new-token rows past L
    for (int i = tid; i < 2 * 16 * ROWB / 16; i += NTHR) {
      if ((i % (16 * ROWB / 16)) >= L * ROWB / 16) reinterpret_cast<uint4*>(sNK)[i] = make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();

  const float eps = 1e-5f;
  // Queries: normalized and roped, FP16 into smem as [kv head][token j * 8 + head][64].
#pragma unroll
  for (int i = 0; i < QR; ++i) {
    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;
    float x0 = __bfloat162float(qv[i].x), x1 = __bfloat162float(qv[i].y);
    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);
    x0 *= inv;
    x1 *= inv;
    if (p.inv_freq) {
      const float2 cs = sCS[j * 32 + lane];
      const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;
      x0 = y0;
      x1 = y1;
    }
    *reinterpret_cast<__half2*>(sQ + ((hh >> 3) * (L * G) + j * G + (hh & 7)) * D + 2 * lane) = __floats2half2_rn(x0, x1);
  }
  // New keys and values (last split): normalize/rope keys, quantize, store to the cache and smem.
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;
      const int head = kvg * 2 + hk;
      float x0 = __bfloat162float(nv[i].x), x1 = __bfloat162float(nv[i].y);
      if (!isv) {
        const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);
        x0 *= inv;
        x1 *= inv;
        if (p.inv_freq) {
          const float2 cs = sCS[j * 32 + lane];
          const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;
          x0 = y0;
          x1 = y1;
        }
      }
      const float sc = isv ? vsc : ksc;
      const __nv_fp8x2_storage_t f8 = __nv_cvt_float2_to_fp8x2(make_float2(x0 / sc, x1 / sc), __NV_SATFINITE, __NV_E4M3);
      uint8_t* cache = isv ? p.v_cache : p.k_cache;
      *reinterpret_cast<__nv_fp8x2_storage_t*>(cache + nloc[i] * p.row_bytes + head * D + 2 * lane) = f8;
      const int byte = hk * D + 2 * lane;
      uint8_t* dst = (isv ? sNV : sNK) + j * ROWB + ((((byte >> 4) ^ swz(j))) << 4) + (byte & 15);
      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;
    }
  }
  __syncthreads();

  WarpState<L> st;
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
    const int mta = (L == 1) ? 0 : (2 * wh + mt);
    const int r0 = 16 * mta + g, r1 = r0 + 8;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint2 x0 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r0) * D + 16 * q4 + 4 * ks);
      st.qa[mt][ks][0] = x0.x;
      st.qa[mt][ks][2] = x0.y;
      if (L == 1) {
        st.qa[mt][ks][1] = 0u;
        st.qa[mt][ks][3] = 0u;
      } else {
        const uint2 x1 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r1) * D + 16 * q4 + 4 * ks);
        st.qa[mt][ks][1] = x1.x;
        st.qa[mt][ks][3] = x1.y;
      }
    }
    st.m[mt][0] = st.m[mt][1] = -INFINITY;
    st.l[mt][0] = st.l[mt][1] = 0.f;
#pragma unroll
    for (int jn = 0; jn < 8; ++jn)
#pragma unroll
      for (int e = 0; e < 4; ++e) st.o[mt][jn][e] = 0.f;
  }

  const float scale_log2 = p.scale_log2 * (*p.k_scale);
  const int tb = (L == 1) ? 32 * wh : 0;
  const int win_edge = p.window ? (seq + L - p.window) : -2147483647;  // keys below need per-row masks

  for (int i = 0; i < ntiles; ++i) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int nxt = i + STAGES - 1;
    if (nxt < ntiles) issue(nxt % STAGES, rows_next);
    cp_commit();
    load_rows(nxt + 1, rows_next);
    const int slot = i % STAGES;
    const int kp0 = c_lo + i * T;
    const bool need_mask = (kp0 + T > c_hi) || (L > 1 && kp0 < win_edge);
    compute_tile<L, NT>(st, sK + slot * TILE_BYTES, sV + slot * TILE_BYTES, tb, h, wh, kp0, c_hi, seq, p.window,
                        need_mask, scale_log2);
  }
  cp_wait<0>();
  if (last && (L > 1 || wh == 0)) {
    compute_tile<L, 2>(st, sNK, sNV, 0, h, wh, seq, seq + L, seq, p.window, true, scale_log2);
  }

  // Thread-partial sums -> row sums.
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float v = st.l[mt][rh];
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      st.l[mt][rh] = v;
    }

  // Partials into shared memory: [slot][row][PR], row = token j * 16 + head hh of the KV pair,
  // slot = token half (L == 1) or 0. PR floats: 64 dims, max, sum.
  constexpr int ROWS = L * 16;
  constexpr int NSL = (L == 1) ? 2 : 1;
  constexpr int PR = 68;
  __syncthreads();  // every warp is done with the stage buffers
  const int KVG = p.hkv / 2;
  float* sP = p.dsmem ? reinterpret_cast<float*>(smem)
                      : p.part1 + ((long long)(b * KVG + kvg) * p.ns + split) * (NSL * ROWS * PR);
#pragma unroll
  for (int mt = 0; mt < MTW; ++mt) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      if (L == 1 && rh == 1) continue;
      const int mta = (L == 1) ? 0 : (2 * wh + mt);
      const int j = (L == 1) ? 0 : (2 * mta + rh);
      const int row = j * 16 + h * 8 + g;
      const int slot = (L == 1) ? wh : 0;
      float* dst = sP + (slot * ROWS + row) * PR;
      float4* d4 = reinterpret_cast<float4*>(dst + 16 * q4);
      d4[0] = make_float4(st.o[mt][0][2 * rh], st.o[mt][1][2 * rh], st.o[mt][2][2 * rh], st.o[mt][3][2 * rh]);
      d4[1] = make_float4(st.o[mt][4][2 * rh], st.o[mt][5][2 * rh], st.o[mt][6][2 * rh], st.o[mt][7][2 * rh]);
      d4[2] = make_float4(st.o[mt][0][2 * rh + 1], st.o[mt][1][2 * rh + 1], st.o[mt][2][2 * rh + 1], st.o[mt][3][2 * rh + 1]);
      d4[3] = make_float4(st.o[mt][4][2 * rh + 1], st.o[mt][5][2 * rh + 1], st.o[mt][6][2 * rh + 1], st.o[mt][7][2 * rh + 1]);
      if (q4 == 0) {
        dst[64] = st.m[mt][rh];
        dst[65] = st.l[mt][rh];
      }
    }
  }
  if (!p.dsmem) __threadfence();
  cluster_sync_all();

  const int cs = p.cs, rank = split % cs, cidx = split / cs, nc = p.ns / cs;
  // Partial (rank c of this cluster, slot sl, row rr), at float offset off within the row.
  const float* gP0 = p.part1 + ((long long)(b * KVG + kvg) * p.ns + cidx * cs) * (NSL * ROWS * PR);
  auto part2 = [&](int c, int sl, int rr, int off) -> float2 {
    if (p.dsmem) return ld_cluster_f32x2(cluster_addr(sP + (sl * ROWS + rr) * PR + off, c));
    return __ldcg(reinterpret_cast<const float2*>(gP0 + (long long)c * (NSL * ROWS * PR) + (sl * ROWS + rr) * PR + off));
  };
  const float vs = *p.v_scale;
  const float eps2 = 1e-5f;
  auto finalize = [&](int rr, int k, float o0, float o1) {
    const int j = rr >> 4, hh = rr & 15;
    const float r = rsqrtf(warp_sum(o0 * o0 + o1 * o1) * (1.f / D) + eps2);
    float g0 = 1.f, g1 = 1.f;
    if (gated) {
      const __nv_bfloat162 gt = reinterpret_cast<const __nv_bfloat162*>(sG + k * D)[lane];
      g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));
      g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));
    }
    *reinterpret_cast<__nv_bfloat162*>(p.out + (long long)(b * L + j) * (p.hq * D) + (kvg * 16 + hh) * D + 2 * lane) =
        __floats2bfloat162_rn(o0 * r * g0, o1 * r * g1);
  };

  // Level 1: rank r merges rows r, r + CS, ... over the cluster's partials (distributed smem).
  const int np = cs * NSL;
  for (int k = warp; rank + k * cs < ROWS; k += 4) {
    const int rr = rank + k * cs;
    float mi = -INFINITY, li = 0.f;
    if (lane < np) {
      const float2 v = part2(lane / NSL, lane % NSL, rr, 64);
      mi = v.x;
      li = v.y;
    }
    float M = mi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    const float Mu = (M == -INFINITY) ? 0.f : M;
    const float wi = (lane < np) ? ex2(mi - Mu) : 0.f;
    const float lsum = warp_sum(wi * li);
    float o0 = 0.f, o1 = 0.f;
    for (int i0 = 0; i0 < np; i0 += 8) {  // eight remote loads in flight at a time
      float2 v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u)
        if (i0 + u < np)
          v[u] = part2((i0 + u) / NSL, (i0 + u) % NSL, rr, 2 * lane);
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);
        if (i0 + u < np) {
          o0 = fmaf(w, v[u].x, o0);
          o1 = fmaf(w, v[u].y, o1);
        }
      }
    }
    if (nc == 1) {
      const float sc = vs / lsum;
      finalize(rr, k, o0 * sc, o1 * sc);
    } else {
      const long long prow = ((long long)(b * KVG + kvg) * nc + cidx) * ROWS + rr;
      reinterpret_cast<float2*>(p.part_o + prow * D)[lane] = make_float2(o0, o1);
      if (lane == 0) reinterpret_cast<float2*>(p.part_ml)[prow] = make_float2(M, lsum);
    }
  }
  if (p.dsmem) cluster_sync_all();  // no CTA leaves while its shared memory may still be read
  if (nc == 1) return;

  // Level 2: the last cluster to arrive at rank r merges rows r, r + CS, ... over the clusters.
  __shared__ int s_last;
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int prev = atomicAdd(p.counters + (b * KVG + kvg) * cs + rank, 1);
    s_last = (prev == nc - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int k = warp; rank + k * cs < ROWS; k += 4) {
    const int rr = rank + k * cs;
    const long long prow0 = ((long long)(b * KVG + kvg) * nc) * ROWS + rr;
    float mi = -INFINITY, li = 0.f;
    if (lane < nc) {
      const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.part_ml) + prow0 + (long long)lane * ROWS);
      mi = ml.x;
      li = ml.y;
    }
    float M = mi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    const float Mu = (M == -INFINITY) ? 0.f : M;
    const float wi = (lane < nc) ? ex2(mi - Mu) : 0.f;
    const float lsum = warp_sum(wi * li);
    float o0 = 0.f, o1 = 0.f;
    for (int i0 = 0; i0 < nc; i0 += 8) {
      float2 v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u)
        if (i0 + u < nc) v[u] = __ldcg(reinterpret_cast<const float2*>(p.part_o + (prow0 + (long long)(i0 + u) * ROWS) * D) + lane);
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);
        if (i0 + u < nc) {
          o0 = fmaf(w, v[u].x, o0);
          o1 = fmaf(w, v[u].y, o1);
        }
      }
    }
    const float sc = vs / lsum;
    finalize(rr, k, o0 * sc, o1 * sc);
  }
  if (tid == 0) p.counters[(b * KVG + kvg) * cs + rank] = 0;
}

template <int L, int STAGES>
constexpr int smem_bytes() {
  return 2 * STAGES * TILE_BYTES + 2 * 16 * ROWB + 2 * L * G * D * 2 + L * 32 * 8 + L * 16 * D * 2;
}

template <int L, int STAGES>
cudaError_t launch_t(const Params& p, cudaStream_t stream) {
  constexpr int smem = smem_bytes<L, STAGES>();
  static_assert(L * 16 * (L == 1 ? 2 : 1) * 68 * 4 <= 2 * STAGES * TILE_BYTES, "partials fit in the stage buffers");
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(p.ns, p.hkv / 2, p.batch);
  cfg.blockDim = dim3(NTHR, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = p.cs;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, attn_main<L, STAGES>, p);
}

template <int L, int STAGES>
int max_cluster_t(int cs) {
  constexpr int smem = smem_bytes<L, STAGES>();
  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cs, 1, 1);
  cfg.blockDim = dim3(NTHR, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = cs;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  int n = 0;
  if (cudaOccupancyMaxActiveClusters(&n, attn_main<L, STAGES>, &cfg) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return n;
}

}  // namespace

void attn_forward(int64_t qkv, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,
                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,
                  int64_t part_o, int64_t part_ml, int64_t counters, int64_t part1, int64_t dsmem, int64_t out, int64_t batch, int64_t q_len,
                  int64_t qkv_stride, int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window,
                  int64_t ns, int64_t cs, double scale_log2, int64_t stream) {
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);
  p.positions = reinterpret_cast<const int64_t*>(positions);
  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);
  p.page_table = reinterpret_cast<const int*>(page_table);
  p.seq_lens = reinterpret_cast<const int*>(seq_lens);
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);
  p.k_scale = reinterpret_cast<const float*>(k_scale);
  p.v_scale = reinterpret_cast<const float*>(v_scale);
  p.inv_freq = reinterpret_cast<const float*>(inv_freq);
  p.part_o = reinterpret_cast<float*>(part_o);
  p.part_ml = reinterpret_cast<float*>(part_ml);
  p.counters = reinterpret_cast<int*>(counters);
  p.part1 = reinterpret_cast<float*>(part1);
  p.dsmem = (int)dsmem;
  p.out = reinterpret_cast<__nv_bfloat16*>(out);
  p.batch = (int)batch;
  p.qkv_stride = (int)qkv_stride;
  p.hq = (int)hq;
  p.hkv = (int)hkv;
  p.max_pages = (int)max_pages;
  p.ps = (int)ps;
  int lg = 0;
  while ((1 << lg) < ps) ++lg;
  p.ps_log2 = lg;
  p.window = (int)window;
  p.ns = (int)ns;
  p.cs = (int)cs;
  p.row_bytes = hkv * 64;
  p.scale_log2 = (float)scale_log2;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t e = (q_len == 1) ? launch_t<1, 4>(p, s) : launch_t<8, 3>(p, s);
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

template <int L, int STAGES>
int occupancy_t() {
  constexpr int smem = smem_bytes<L, STAGES>();
  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int n = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, attn_main<L, STAGES>, NTHR, smem) != cudaSuccess) {
    cudaGetLastError();
    return 1;
  }
  return n;
}

int64_t occupancy(int64_t q_len) { return (q_len == 1) ? occupancy_t<1, 4>() : occupancy_t<8, 3>(); }

int64_t max_cluster(int64_t q_len, int64_t cs) {
  return (q_len == 1) ? max_cluster_t<1, 4>((int)cs) : max_cluster_t<8, 3>((int)cs);
}
"""


ATTN_PORT_CPP_SRC = r"""
#include <cstdint>
void attn_forward(int64_t qkv, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,
                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,
                  int64_t part_o, int64_t part_ml, int64_t counters, int64_t part1, int64_t dsmem, int64_t out, int64_t batch, int64_t q_len,
                  int64_t qkv_stride, int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window,
                  int64_t ns, int64_t cs, double scale_log2, int64_t stream);
int64_t max_cluster(int64_t q_len, int64_t cs);
int64_t occupancy(int64_t q_len);
"""










ATTN_X_CUDA = r"""


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdexcept>
#define DI __device__ __forceinline__
namespace {
#ifndef XKEYS
#define XKEYS 128
#endif
constexpr int KT=XKEYS,KB=KT*64;
struct XP {
 const __nv_bfloat16* qkv; const int64_t* pos; const int64_t* loc;
 const int* pt; const int* seq; uint8_t* kc; uint8_t* vc;
 const float* ks; const float* vs; const float* freq;
 float* partial; int* count; __nv_bfloat16* out;
 int batch,hq,hkv,stride,maxpages,ps,lg,window,ns; float scale;
};
DI uint32_t su(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
DI void cp(void* dst,const void* src,int valid){asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(su(dst)),"l"(src),"r"(valid?16:0):"memory");}
DI void commit(){asm volatile("cp.async.commit_group;":::"memory");}
template<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}

DI void consumer_bar(){asm volatile("bar.sync 1, 256;":::"memory");}
DI void mb_init(uint64_t* b,int count){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(su(b)),"r"(count):"memory");}
DI void mb_arrive(uint64_t* b){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(su(b)):"memory");}
DI void cp_arrive(uint64_t* b){asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"::"r"(su(b)):"memory");}
DI void mb_wait(uint64_t* b,int phase){uint32_t done;do{asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2;selp.b32 %0,1,0,p;}":"=r"(done):"r"(su(b)),"r"(phase):"memory");}while(!done);}

DI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
DI float wsum(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
DI uint16_t fp8pair(float a,float b){return __nv_cvt_float2_to_fp8x2(make_float2(a,b),__NV_SATFINITE,__NV_E4M3);}
DI uint32_t halfpair(uint16_t v){uint32_t h;asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(h):"h"(v));return h;}
DI uint32_t transpose(uint32_t v){uint32_t r;asm("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;":"=r"(r):"r"(v));return r;}
DI uint16_t h2fp8(uint32_t v){uint16_t r;asm("cvt.rn.satfinite.e4m3x2.f16x2 %0,%1;":"=h"(r):"r"(v));return r;}
DI uint32_t hpack(float a,float b){__half2 v=__floats2half2_rn(a,b);return *reinterpret_cast<uint32_t*>(&v);}
DI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
DI void mma16(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
// A 64-byte row is XOR-swizzled by 16-byte chunks. All four lanes of a row
// issue adjacent global transactions; the shared permutation removes bank overlap.
DI int off(int row,int col){return row*64+((col>>4)^((row&3)^((row>>2)&1)))*16+(col&15);}
DI int voff(int r,int c){return r*128+((c>>3)^(r&7))*16+(c&7)*2;}
template<int TK> DI void widen(const uint8_t* v,uint8_t* vh){
 for(int i=threadIdx.x;i<TK*32;i+=256){int row=i/32,col=(i%32)*2;
  uint16_t f=*reinterpret_cast<const uint16_t*>(v+off(row,col));
  *reinterpret_cast<uint32_t*>(vh+voff(row,col))=halfpair(f);
 }
}
DI void loadv(uint32_t (&a)[4],const uint8_t* v,int kt,int dt){
 int lane=threadIdx.x&31,mi=lane>>3,r=lane&7;
 uint32_t ptr=su(v+voff(kt*16+8*(mi>>1)+r,dt*16+8*(mi&1)));
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(ptr));
}
struct XS {uint32_t qb[1][2][2]; float o[4][1][4];float m[1][2],l[1][2];};
template<int TK,bool P8>
DI void tile(XS& st,const uint8_t* K,const uint8_t* V,int kp,int hi,int seq,int window,float scale,uint8_t* vh){
 if constexpr(!P8){widen<TK>(V,vh);consumer_bar();}
 int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
 float s[TK/16][1][4];
 #pragma unroll
 for(int kt=0;kt<TK/16;kt++){
  #pragma unroll
  for(int nt=0;nt<1;nt++)for(int e=0;e<4;e++)s[kt][nt][e]=0;
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   uint32_t a[4];
   uint32_t ka=su(K+off(kt*16+(lane&7)+8*((lane>>3)&1),kk*32+16*(lane>>4)));
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(ka));
   #pragma unroll
   for(int nt=0;nt<1;nt++)mma8(s[kt][nt],a,st.qb[nt][kk][0],st.qb[nt][kk][1]);
  }
 }
 #pragma unroll
 for(int nt=0;nt<1;nt++){
  #pragma unroll
  for(int e=0;e<2;e++){
   float mx=-INFINITY;
   int query=w*8+nt*8+2*t+e,j=query/8;
   // The interior of a full/sliding split needs no per-score mask.
   if(kp+TK<=hi && kp+TK<=seq && (!window || kp>=seq+8-window)){
   #pragma unroll
   for(int kt=0;kt<TK/16;kt++){
    #pragma unroll
    for(int rh=0;rh<2;rh++){
     float z=s[kt][nt][rh*2+e]*scale;
     s[kt][nt][rh*2+e]=z; mx=fmaxf(mx,z);
    }
   }
   }else{
   #pragma unroll
   for(int kt=0;kt<TK/16;kt++){
    #pragma unroll
    for(int rh=0;rh<2;rh++){
     int key=kp+16*kt+g+8*rh;
     float z=s[kt][nt][rh*2+e]*scale;
     if(key>=hi || key>seq+j || (window && key<seq+j-window+1))z=-INFINITY;
     s[kt][nt][rh*2+e]=z; mx=fmaxf(mx,z);
    }
   }
   }
   mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,4));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,8));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,16));
   float nm=fmaxf(st.m[nt][e],mx);
   // Empty partitions retain zero mass, including before their first valid key.
   float alpha=st.m[nt][e]==-INFINITY?0.f:ex(st.m[nt][e]-nm);
   float add=0;
   #pragma unroll
   for(int kt=0;kt<TK/16;kt++)for(int rh=0;rh<2;rh++){
    float z=s[kt][nt][rh*2+e]; float p=z==-INFINITY?0.f:ex(z-nm);
    s[kt][nt][rh*2+e]=P8?p*256.f:p;add+=p;
   }
   add+=__shfl_xor_sync(0xffffffff,add,4);add+=__shfl_xor_sync(0xffffffff,add,8);add+=__shfl_xor_sync(0xffffffff,add,16);
   st.l[nt][e]=st.l[nt][e]*alpha+add;st.m[nt][e]=nm;
   #pragma unroll
   for(int dt=0;dt<4;dt++){st.o[dt][nt][e]*=alpha;st.o[dt][nt][e+2]*=alpha;}
  }
 }
 if constexpr(P8){
  #pragma unroll
  for(int kt=0;kt<TK/32;kt++){
   uint32_t pb[1][2];
   #pragma unroll
   for(int nt=0;nt<1;nt++){
    #pragma unroll
    for(int half=0;half<2;half++){
     // Transpose each 8x8 probability panel in registers; join adjacent
     // key pairs within the four-lane group into a K32 FP8 B word.
     uint32_t lo=transpose(hpack(s[2*kt+half][nt][0],s[2*kt+half][nt][1]));
     uint32_t hi=transpose(hpack(s[2*kt+half][nt][2],s[2*kt+half][nt][3]));
     uint32_t pairs=(uint32_t)h2fp8(lo)|((uint32_t)h2fp8(hi)<<16);
     uint32_t v0=__shfl_sync(0xffffffff,pairs,g*4+((2*t)&3));
     uint32_t v1=__shfl_sync(0xffffffff,pairs,g*4+((2*t+1)&3));
     pb[nt][half]=__byte_perm(v0,v1,t>=2?0x7632:0x5410);
    }
   }
   #pragma unroll
   for(int dt=0;dt<4;dt++){
    uint32_t r[4],a[4];
    uint32_t va=su(V+off(kt*32+(lane&7)+8*(lane>>3),dt*16));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(va));
    #pragma unroll
    for(int rh=0;rh<2;rh++){
     int src=((g+8*rh)>>1)*4+((2*t)&3);
     uint32_t b[4];
     #pragma unroll
     for(int i=0;i<4;i++){
      uint32_t x=__shfl_sync(0xffffffff,r[i],src);
      uint32_t y=__shfl_sync(0xffffffff,r[i],src+1);
      b[i]=__byte_perm(x,y,(g&1)?0x7531:0x6420);
     }
     a[rh]=t<2?b[0]:b[1];a[rh+2]=t<2?b[2]:b[3];
    }
    #pragma unroll
    for(int nt=0;nt<1;nt++)mma8(st.o[dt][nt],a,pb[nt][0],pb[nt][1]);
   }
  }
 }else{
  #pragma unroll
  for(int kt=0;kt<TK/16;kt++){
   uint32_t pb[1][2];
   #pragma unroll
   for(int nt=0;nt<1;nt++){
    #pragma unroll
    for(int half=0;half<2;half++){
     pb[nt][half]=transpose(hpack(s[kt][nt][half*2],s[kt][nt][half*2+1]));
    }
   }
   #pragma unroll
   for(int dt=0;dt<4;dt++){
    uint32_t a[4];loadv(a,vh,kt,dt);
    #pragma unroll
    for(int nt=0;nt<1;nt++)mma16(st.o[dt][nt],a,pb[nt][0],pb[nt][1]);
   }
  }
 }
}
template<int ST,bool P8,bool ROPE>
__global__ __launch_bounds__(384,1) void xqa(XP p){
 extern __shared__ __align__(128) uint8_t sm[];
 uint8_t* sk=sm; uint8_t* sv=sk+ST*KB; uint8_t* nq=sv+ST*KB;
 uint8_t* nk=nq+4096;uint8_t* nv=nk+2048;uint8_t* vh=nv+2048;
 int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];bool last=sp==p.ns-1;
 int lo=p.window?max(0,seq-p.window+1):0;
 int chunk=((seq-lo+p.ns-1)/p.ns+KT-1)&~(KT-1);
 int begin=min(seq,lo+sp*chunk),end=min(seq,begin+chunk),ntiles=(end-begin+KT-1)/KT;
 const int* pt=p.pt+(long long)b*p.maxpages; long long rowbytes=p.hkv*64;

 __shared__ __align__(16) uint64_t full[ST],empty[ST];
 if(tid<ST){mb_init(&full[tid],128);mb_init(&empty[tid],256);}
 __syncthreads();
 if(tid>=256){
  int ptid=tid-256;
  for(int i=0;i<ntiles;i++){
   int slot=i%ST;
   if(i>=ST)mb_wait(&empty[slot],((i/ST)-1)&1);
   #pragma unroll
   for(int rr=0;rr<KT/32;rr++){
    int row=ptid/4+rr*32,col=(ptid&3)*16,key=begin+i*KT+row;
    bool valid=key<end;
    int physical=valid?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):0;
    const uint8_t* kp=p.kc+(long long)physical*rowbytes+h*64+col;
    const uint8_t* vp=p.vc+(long long)physical*rowbytes+h*64+col;
    cp(sk+slot*KB+off(row,col),kp,valid);
    cp(sv+slot*KB+off(row,col),vp,valid);
   }
   commit();cp_arrive(&full[slot]);
   wait<ST-1>();
  }
  wait<0>();
  return;
 }
 // Every split prepares its own Q; only the final split writes new KV, with
 // no race against the historical row gathers (their upper bound is seq).
 #pragma unroll
 for(int r=0;r<8;r++){
  int query=w+8*r,j=query/8,head=h*8+query%8;
  const __nv_bfloat16* ptr=p.qkv+(long long)(b*8+j)*p.stride+head*64+2*lane;
  __nv_bfloat162 q=*reinterpret_cast<const __nv_bfloat162*>(ptr);
  float x=__bfloat162float(q.x),y=__bfloat162float(q.y),z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
  if constexpr(ROPE){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float a=x*cs-y*sn;y=y*cs+x*sn;x=a;}
  *reinterpret_cast<uint16_t*>(nq+query*64+2*lane)=fp8pair(x,y);
 }
 if(last){
  for(int i=tid;i<4096/16;i+=256)*reinterpret_cast<uint4*>(nk+i*16)=make_uint4(0,0,0,0);
  consumer_bar();
  #pragma unroll
  for(int r=0;r<2;r++){
   int row=w+8*r,j=row/2,isv=row&1;
   auto q=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+isv*p.hkv+h)*64+2*lane);
   float x=__bfloat162float(q.x),y=__bfloat162float(q.y);
   if(!isv){float z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
    if constexpr(ROPE){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float a=x*cs-y*sn;y=y*cs+x*sn;x=a;}}
   float sc=isv?*p.vs:*p.ks;uint16_t f=fp8pair(x/sc,y/sc);
   *reinterpret_cast<uint16_t*>((isv?p.vc:p.kc)+p.loc[b*8+j]*rowbytes+h*64+2*lane)=f;
   *reinterpret_cast<uint16_t*>((isv?nv:nk)+off(j,2*lane))=f;
  }
 }
 consumer_bar();XS st;
 #pragma unroll
 for(int nt=0;nt<1;nt++){
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   st.qb[nt][kk][0]=*reinterpret_cast<uint32_t*>(nq+(w*8+nt*8+g)*64+kk*32+4*t);
   st.qb[nt][kk][1]=*reinterpret_cast<uint32_t*>(nq+(w*8+nt*8+g)*64+kk*32+16+4*t);
  }
  for(int e=0;e<2;e++){st.m[nt][e]=-INFINITY;st.l[nt][e]=0;}
  for(int dt=0;dt<4;dt++)for(int e=0;e<4;e++)st.o[dt][nt][e]=0;
 }
 float scale=p.scale*(*p.ks);
 for(int i=0;i<ntiles;i++){
  mb_wait(&full[i%ST],(i/ST)&1);
  consumer_bar();
  tile<KT,P8>(st,sk+(i%ST)*KB,sv+(i%ST)*KB,begin+i*KT,end,seq,p.window,scale,vh);
  consumer_bar();
  mb_arrive(&empty[i%ST]);
 }
 consumer_bar();
 if(last)tile<32,P8>(st,nk,nv,seq,seq+8,seq,p.window,scale,vh);
 float* po=p.partial+((long long)(b*p.hkv+h)*p.ns+sp)*64*68;
 #pragma unroll
 for(int nt=0;nt<1;nt++)for(int e=0;e<2;e++){
  int query=w*8+nt*8+2*t+e;
  for(int dt=0;dt<4;dt++){
   po[query*68+dt*16+g]=st.o[dt][nt][e];po[query*68+dt*16+g+8]=st.o[dt][nt][e+2];
  }
  if(g==0){po[query*68+64]=st.m[nt][e];po[query*68+65]=st.l[nt][e];}
 }
 __threadfence();consumer_bar();__shared__ int elected;
 if(tid==0)elected=atomicAdd(p.count+b*p.hkv+h,1)==p.ns-1;
 consumer_bar();if(!elected)return;
 // Last-CTA reduction in FP32. Every producer publishes O,m,l before its arrival.
 const volatile float* all=p.partial+(long long)(b*p.hkv+h)*p.ns*64*68;
 for(int query=w;query<64;query+=8){
  float mx=-INFINITY;
  for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,all[(long long)s*64*68+query*68+64]);
  #pragma unroll
  for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
  float lsum=0,x=0,y=0;
  float* sa=reinterpret_cast<float*>(sm)+w*64;
  for(int s=lane;s<p.ns;s+=32){
   long long ix=(long long)s*64*68+query*68;
   float m=all[ix+64],a=m==-INFINITY?0.f:ex(m-mx);sa[s]=a;
   lsum+=all[ix+65]*a;
  }
  lsum=wsum(lsum);__syncwarp();
  for(int s=0;s<p.ns;s+=4){
   float xx[4],yy[4],a[4];
   #pragma unroll
   for(int i=0;i<4;i++){
    int sp=s+i;xx[i]=yy[i]=a[i]=0;
    if(sp<p.ns){long long ix=(long long)sp*64*68+query*68+2*lane;
     const float* ptr=p.partial+(long long)(b*p.hkv+h)*p.ns*64*68+ix;
     asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(xx[i]),"=f"(yy[i]):"l"(ptr):"memory");a[i]=sa[sp];}
   }
   #pragma unroll
   for(int i=0;i<4;i++){x+=xx[i]*a[i];y+=yy[i]*a[i];}
  }
  __syncwarp();
  float sc=(*p.vs)/(lsum*(P8?256.f:1.f));x*=sc;y*=sc;
  float inv=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);
  int j=query/8,head=h*8+query%8;
  auto gate=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+2*p.hkv+head)*64+2*lane);
  x=x*inv/(1.f+expf(-__bfloat162float(gate.x)));y=y*inv/(1.f+expf(-__bfloat162float(gate.y)));
  *reinterpret_cast<__nv_bfloat162*>(p.out+((long long)(b*8+j)*p.hq+head)*64+2*lane)=__floats2bfloat162_rn(x,y);
 }
 consumer_bar();if(tid==0)atomicExch(p.count+b*p.hkv+h,0);
}
template<int ST,bool P8,bool ROPE>void go(XP p,cudaStream_t stream){
 constexpr int smem=ST*2*KB+8192+(P8?0:2*KB);
 cudaFuncSetAttribute(xqa<ST,P8,ROPE>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 xqa<ST,P8,ROPE><<<dim3(p.ns,p.hkv,p.batch),384,smem,stream>>>(p);
}
}
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream){
 XP p{};p.qkv=(const __nv_bfloat16*)qkv;p.pos=(const int64_t*)pos;p.loc=(const int64_t*)loc;p.pt=(const int*)pt;p.seq=(const int*)seq;p.kc=(uint8_t*)kc;p.vc=(uint8_t*)vc;p.ks=(const float*)ks;p.vs=(const float*)vs;p.freq=(const float*)freq;p.partial=(float*)part;p.count=(int*)count;p.out=(__nv_bfloat16*)out;
 p.batch=batch;p.hq=hq;p.hkv=hkv;p.stride=stride;p.maxpages=maxpages;p.ps=ps;while((1<<p.lg)<ps)p.lg++;p.window=window;p.ns=ns;p.scale=scale;
 auto s=(cudaStream_t)stream;
 if(stages==3){
  if(p.window)go<3,false,true>(p,s);else go<3,false,false>(p,s);
 }
 else{
  if(p.window)go<4,false,true>(p,s);else go<4,false,false>(p,s);
 }
 auto err=cudaGetLastError();if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));
}


"""
ATTN_X_CPP = r"""

#include <cstdint>
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream);

"""
ATTN_G_CUDA = r"""

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdexcept>
#define DI __device__ __forceinline__
#ifndef USE_WGMMA
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
#define USE_WGMMA 1
#else
#define USE_WGMMA 0
#endif
#endif
namespace {
struct XP {
 const __nv_bfloat16* qkv; const int64_t* pos; const int64_t* loc;
 const int* pt; const int* seq; uint8_t* kc; uint8_t* vc;
 const float* ks; const float* vs; const float* freq;
 float* partial; int* count; __nv_bfloat16* out;
 int batch,hq,hkv,stride,maxpages,ps,lg,window,ns; float scale;
};
DI uint32_t su(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
DI void cp(void* dst,const void* src,bool v){asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(su(dst)),"l"(src),"r"(v?16:0):"memory");}
DI void commit(){asm volatile("cp.async.commit_group;":::"memory");}
template<int N>DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}
DI void bar(){asm volatile("bar.sync 1,128;":::"memory");}
DI void mb_init(uint64_t* p,int n){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(su(p)),"r"(n):"memory");}
DI void mb_arrive(uint64_t* p){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(su(p)):"memory");}
DI void cp_arrive(uint64_t* p){asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"::"r"(su(p)):"memory");}
DI void mb_wait(uint64_t* p,int parity){uint32_t ok;do{asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2;selp.b32 %0,1,0,p;}":"=r"(ok):"r"(su(p)),"r"(parity):"memory");}while(!ok);}
DI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
DI float wsum(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
DI uint16_t fp8pair(float a,float b){return __nv_cvt_float2_to_fp8x2(make_float2(a,b),__NV_SATFINITE,__NV_E4M3);}
DI uint32_t halfpair(uint16_t x){uint32_t y;asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(y):"h"(x));return y;}
DI uint32_t hpack(float a,float b){__half2 h=__floats2half2_rn(a,b);return *reinterpret_cast<uint32_t*>(&h);}
DI uint32_t trans(uint32_t x){uint32_t y;asm("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;":"=r"(y):"r"(x));return y;}
// K-major 64-byte FP8 rows: PTX 64B swizzle, SBO=512 bytes.
DI int off(int r,int c){return r*64+((c>>4)^((r>>1)&3))*16+(c&15);}
// FP16 rows: 128-byte swizzle; also describes MN-major A for V^T.
DI int hoff(int r,int c){return r*128+((c>>3)^(r&7))*16+(c&7)*2;}
DI uint32_t word(const uint8_t* p,int r,int c){return *reinterpret_cast<const uint32_t*>(p+off(r,c));}
DI uint64_t d64(const uint8_t* p){return (uint64_t)(su(p)>>4)|(1ull<<16)|(32ull<<32)|(2ull<<62);}
DI uint64_t d128(const uint8_t* p){return (uint64_t)(su(p)>>4)|(1ull<<16)|(64ull<<32)|(1ull<<62);}
DI uint64_t d128mn(const uint8_t* p){return (uint64_t)(su(p)>>4)|(64ull<<32)|(1ull<<62);}
DI void fence(){
#if USE_WGMMA
 asm volatile("fence.proxy.async.shared::cta;":::"memory");
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
}
DI void done(){
#if USE_WGMMA
 asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
}
DI void protect(float (&a)[8][4]){
 #pragma unroll
 for(int n=0;n<8;n++)for(int i=0;i<4;i++)asm volatile("":"+f"(a[n][i])::"memory");
}
// WGMMA wrapper is generated below for all 32 FP32 accumulators.
DI void wg8(float (&d)[8][4],uint64_t a,uint64_t b){
#if USE_WGMMA
 asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1;}":"+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),"+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),"+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),"+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),"+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),"+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),"+f"(d[6][0]),"+f"(d[6][1]),"+f"(d[6][2]),"+f"(d[6][3]),"+f"(d[7][0]),"+f"(d[7][1]),"+f"(d[7][2]),"+f"(d[7][3]):"l"(a),"l"(b));
#endif
}
DI void wg16(float (&d)[8][4],uint64_t a,uint64_t b){
#if USE_WGMMA
 asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,1,0;}":"+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),"+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),"+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),"+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),"+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),"+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),"+f"(d[6][0]),"+f"(d[6][1]),"+f"(d[6][2]),"+f"(d[6][3]),"+f"(d[7][0]),"+f"(d[7][1]),"+f"(d[7][2]),"+f"(d[7][3]):"l"(a),"l"(b));
#endif
}
DI void qk(float (&s)[8][4],const uint8_t* k,const uint8_t* q){
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<4;e++)s[n][e]=0;
 fence();
 #pragma unroll
 for(int kk=0;kk<2;kk++){
#if USE_WGMMA
  wg8(s,d64(k+kk*32),d64(q+kk*32));
#else
  int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
  uint32_t a[4]={word(k,w*16+g,kk*32+4*t),word(k,w*16+g+8,kk*32+4*t),word(k,w*16+g,kk*32+16+4*t),word(k,w*16+g+8,kk*32+16+4*t)};
  #pragma unroll
  for(int n=0;n<8;n++){
   uint32_t b0=word(q,n*8+g,kk*32+4*t),b1=word(q,n*8+g,kk*32+16+4*t);
   asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(s[n][0]),"+f"(s[n][1]),"+f"(s[n][2]),"+f"(s[n][3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }
#endif
 }
 done();protect(s);
}
DI void pv(float (&o)[8][4],const uint8_t* vh,const uint8_t* ph){
 fence();
 #pragma unroll
 for(int kk=0;kk<4;kk++){
#if USE_WGMMA
  wg16(o,d128mn(vh+kk*16*128),d128(ph+kk*32));
#else
  int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5,mi=lane>>3;
  uint32_t a[4],ptr=su(vh+hoff(kk*16+8*(mi>>1)+(lane&7),w*16+8*(mi&1)));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(ptr));
  #pragma unroll
  for(int n=0;n<8;n++){
   uint32_t b0=*reinterpret_cast<const uint32_t*>(ph+hoff(n*8+g,kk*16+2*t));
   uint32_t b1=*reinterpret_cast<const uint32_t*>(ph+hoff(n*8+g,kk*16+2*t+8));
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(o[n][0]),"+f"(o[n][1]),"+f"(o[n][2]),"+f"(o[n][3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }
#endif
 }
 done();protect(o);
}
struct State{float o[8][4],m[8][2],l[8][2];};
template<bool ROPE>
DI void tile(State& st,const uint8_t* k,const uint8_t* v,const uint8_t* q,uint8_t* vh,uint8_t* ph,float* red,int kp,int hi,int seq,int window,float scale){
 int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 for(int i=tid;i<2048;i+=128){int row=i/32,col=(i%32)*2;uint16_t f=*reinterpret_cast<const uint16_t*>(v+off(row,col));*reinterpret_cast<uint32_t*>(vh+hoff(row,col))=halfpair(f);}
 bar();float s[8][4];qk(s,k,q);
 // The 64-key reduction spans four warps. Keep a single shared 4x64
 // scratch, then reuse it for sums after every warp has read its maximum.
 float local[8][2];
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++){
  int query=n*8+2*t+e,j=query/8;
  float a=s[n][e]*scale,b=s[n][e+2]*scale;
  int ka=kp+w*16+g,kb=ka+8;
  if(ka>=hi||ka>seq+j||(ROPE&&ka<seq+j-window+1))a=-INFINITY;
  if(kb>=hi||kb>seq+j||(ROPE&&kb<seq+j-window+1))b=-INFINITY;
  s[n][e]=a;s[n][e+2]=b;
  float mx=fmaxf(a,b);mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,4));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,8));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,16));
  if(g==0)red[w*64+query]=mx;
 }
 bar();
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++){
  int query=n*8+2*t+e;
  float mx=fmaxf(fmaxf(red[query],red[64+query]),fmaxf(red[128+query],red[192+query]));
  float nm=fmaxf(st.m[n][e],mx),alpha=st.m[n][e]==-INFINITY?0.f:ex(st.m[n][e]-nm);
  st.m[n][e]=nm;st.l[n][e]*=alpha;st.o[n][e]*=alpha;st.o[n][e+2]*=alpha;
  float a=s[n][e]==-INFINITY?0.f:ex(s[n][e]-nm),b=s[n][e+2]==-INFINITY?0.f:ex(s[n][e+2]-nm);
  s[n][e]=a;s[n][e+2]=b;
  float add=a+b;add+=__shfl_xor_sync(0xffffffff,add,4);add+=__shfl_xor_sync(0xffffffff,add,8);add+=__shfl_xor_sync(0xffffffff,add,16);local[n][e]=add;
 }
 bar();
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++)if(g==0)red[w*64+n*8+2*t+e]=local[n][e];
 bar();
 #pragma unroll
 for(int n=0;n<8;n++){
  #pragma unroll
  for(int e=0;e<2;e++){int query=n*8+2*t+e;st.l[n][e]+=red[query]+red[64+query]+red[128+query]+red[192+query];}
  uint32_t p0=trans(hpack(s[n][0],s[n][1])),p1=trans(hpack(s[n][2],s[n][3]));
  *reinterpret_cast<uint32_t*>(ph+hoff(n*8+g,w*16+2*t))=p0;
  *reinterpret_cast<uint32_t*>(ph+hoff(n*8+g,w*16+2*t+8))=p1;
 }
 bar();pv(st.o,vh,ph);bar();
}
template<int ST,bool ROPE>
__global__ __launch_bounds__(256,1)void xqa(XP p){
 extern __shared__ __align__(1024)uint8_t sm[];
 uint8_t* sk=sm;uint8_t* sv=sk+ST*4096;uint8_t* nq=sv+ST*4096;
 uint8_t* nk=nq+4096;uint8_t* nv=nk+4096;uint8_t* vh=nv+4096;uint8_t* ph=vh+8192;
 float* red=reinterpret_cast<float*>(ph+8192);
 int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];bool last=sp==p.ns-1;
 int lo=ROPE?max(0,seq-p.window+1):0,chunk=((seq-lo+p.ns-1)/p.ns+63)&~63;
 int begin=min(seq,lo+sp*chunk),end=min(seq,begin+chunk),ntiles=(end-begin+63)/64;
 const int* pt=p.pt+(long long)b*p.maxpages;long long rowbytes=p.hkv*64;
 __shared__ __align__(16)uint64_t full[ST],empty[ST];
 if(tid<ST){mb_init(&full[tid],128);mb_init(&empty[tid],128);}__syncthreads();
 if(tid>=128){
  int ptid=tid-128;
  for(int i=0;i<ntiles;i++){
   int slot=i%ST;if(i>=ST)mb_wait(&empty[slot],((i/ST)-1)&1);
   #pragma unroll
   for(int rr=0;rr<2;rr++){
    int row=ptid/4+rr*32,col=(ptid&3)*16,key=begin+i*64+row;bool valid=key<end;
    int physical=valid?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):0;
    cp(sk+slot*4096+off(row,col),p.kc+(long long)physical*rowbytes+h*64+col,valid);
    cp(sv+slot*4096+off(row,col),p.vc+(long long)physical*rowbytes+h*64+col,valid);
   }
   commit();cp_arrive(&full[slot]);wait<ST-1>();
  }
  wait<0>();return;
 }
 #pragma unroll
 for(int r=0;r<16;r++){
  int query=w+4*r,j=query/8,head=h*8+query%8;
  auto a=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+head*64+2*lane);
  float x=__bfloat162float(a.x),y=__bfloat162float(a.y),z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
  if constexpr(ROPE){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float v=x*cs-y*sn;y=y*cs+x*sn;x=v;}
  *reinterpret_cast<uint16_t*>(nq+off(query,2*lane))=fp8pair(x,y);
 }
 if(last){
  for(int i=tid;i<8192/16;i+=128)*reinterpret_cast<uint4*>(nk+i*16)=make_uint4(0,0,0,0);bar();
  #pragma unroll
  for(int r=0;r<4;r++){
   int row=w+4*r,j=row/2,isv=row&1;
   auto a=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+isv*p.hkv+h)*64+2*lane);
   float x=__bfloat162float(a.x),y=__bfloat162float(a.y);
   if(!isv){float z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;if constexpr(ROPE){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float v=x*cs-y*sn;y=y*cs+x*sn;x=v;}}
   float sc=isv?*p.vs:*p.ks;uint16_t v=fp8pair(x/sc,y/sc);
   *reinterpret_cast<uint16_t*>((isv?p.vc:p.kc)+p.loc[b*8+j]*rowbytes+h*64+2*lane)=v;
   *reinterpret_cast<uint16_t*>((isv?nv:nk)+off(j,2*lane))=v;
  }
 }
 bar();State st;
 #pragma unroll
 for(int n=0;n<8;n++){for(int e=0;e<2;e++){st.m[n][e]=-INFINITY;st.l[n][e]=0;}for(int e=0;e<4;e++)st.o[n][e]=0;}
 float scale=p.scale* *p.ks;
 for(int i=0;i<ntiles;i++){
  mb_wait(&full[i%ST],(i/ST)&1);bar();
  tile<ROPE>(st,sk+(i%ST)*4096,sv+(i%ST)*4096,nq,vh,ph,red,begin+i*64,end,seq,p.window,scale);
  bar();mb_arrive(&empty[i%ST]);
 }
 if(last)tile<ROPE>(st,nk,nv,nq,vh,ph,red,seq,seq+8,seq,p.window,scale);
 float* po=p.partial+((long long)(b*p.hkv+h)*p.ns+sp)*64*68;
 #pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<2;e++){
  int query=n*8+2*t+e;
  po[query*68+w*16+g]=st.o[n][e];po[query*68+w*16+g+8]=st.o[n][e+2];
  if(w==0&&g==0){po[query*68+64]=st.m[n][e];po[query*68+65]=st.l[n][e];}
 }
 __threadfence();bar();__shared__ int elected;
 if(tid==0)elected=atomicAdd(p.count+b*p.hkv+h,1)==p.ns-1;bar();if(!elected)return;
 const volatile float* all=p.partial+(long long)(b*p.hkv+h)*p.ns*64*68;
 for(int query=w;query<64;query+=4){
  float mx=-INFINITY;for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,all[(long long)s*64*68+query*68+64]);
  #pragma unroll
  for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
  float ls=0,x=0,y=0;float* a=reinterpret_cast<float*>(sm)+w*64;
  for(int s=lane;s<p.ns;s+=32){long long ix=(long long)s*64*68+query*68;float m=all[ix+64],v=m==-INFINITY?0.f:ex(m-mx);a[s]=v;ls+=all[ix+65]*v;}
  ls=wsum(ls);__syncwarp();
  for(int s=0;s<p.ns;s++){long long ix=(long long)s*64*68+query*68+2*lane;x+=all[ix]*a[s];y+=all[ix+1]*a[s];}
  __syncwarp();float sc= *p.vs/ls;x*=sc;y*=sc;
  float inv=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);int j=query/8,head=h*8+query%8;
  auto gate=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+2*p.hkv+head)*64+2*lane);
  x=x*inv/(1.f+expf(-__bfloat162float(gate.x)));y=y*inv/(1.f+expf(-__bfloat162float(gate.y)));
  *reinterpret_cast<__nv_bfloat162*>(p.out+((long long)(b*8+j)*p.hq+head)*64+2*lane)=__floats2bfloat162_rn(x,y);
 }
 bar();if(tid==0)atomicExch(p.count+b*p.hkv+h,0);
}
template<int ST,bool ROPE>void go(XP p,cudaStream_t s){constexpr int bytes=ST*8192+12288+16384+1024;cudaFuncSetAttribute(xqa<ST,ROPE>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);xqa<ST,ROPE><<<dim3(p.ns,p.hkv,p.batch),256,bytes,s>>>(p);}
}
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream){
 XP p{};p.qkv=(const __nv_bfloat16*)qkv;p.pos=(const int64_t*)pos;p.loc=(const int64_t*)loc;p.pt=(const int*)pt;p.seq=(const int*)seq;p.kc=(uint8_t*)kc;p.vc=(uint8_t*)vc;p.ks=(const float*)ks;p.vs=(const float*)vs;p.freq=(const float*)freq;p.partial=(float*)part;p.count=(int*)count;p.out=(__nv_bfloat16*)out;p.batch=batch;p.hq=hq;p.hkv=hkv;p.stride=stride;p.maxpages=maxpages;p.ps=ps;while((1<<p.lg)<ps)p.lg++;p.window=window;p.ns=ns;p.scale=scale;
 auto s=(cudaStream_t)stream;
 if(stages==7){if(p.window)go<7,true>(p,s);else go<7,false>(p,s);}else{if(p.window)go<5,true>(p,s);else go<5,false>(p,s);}
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}


"""
ATTN_P8_CUDA = r"""


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdexcept>
#define DI __device__ __forceinline__
namespace {
struct XP {
 const __nv_bfloat16* qkv; const int64_t* pos; const int64_t* loc;
 const int* pt; const int* seq; uint8_t* kc; uint8_t* vc;
 const float* ks; const float* vs; const float* freq;
 float* partial; int* count; __nv_bfloat16* out;
 int batch,hq,hkv,stride,maxpages,ps,lg,window,ns; float scale;
};
DI uint32_t su(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
DI void cp(void* dst,const void* src,int valid){asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(su(dst)),"l"(src),"r"(valid?16:0):"memory");}
DI void commit(){asm volatile("cp.async.commit_group;":::"memory");}
template<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}
DI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
DI float wsum(float x){for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
DI uint16_t fp8pair(float a,float b){return __nv_cvt_float2_to_fp8x2(make_float2(a,b),__NV_SATFINITE,__NV_E4M3);}
DI uint32_t halfpair(uint16_t v){uint32_t h;asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(h):"h"(v));return h;}
DI uint32_t transpose(uint32_t v){uint32_t r;asm("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;":"=r"(r):"r"(v));return r;}
DI uint32_t hpack(float a,float b){__half2 v=__floats2half2_rn(a,b);return *reinterpret_cast<uint32_t*>(&v);}
DI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
DI void mma16(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
// A 64-byte row is XOR-swizzled by 16-byte chunks. All four lanes of a row
// issue adjacent global transactions; the shared permutation removes bank overlap.
DI int off(int row,int col){return row*64+((col>>4)^(row&3))*16+(col&15);}
DI uint32_t word(const uint8_t* v,int r,int c){return *reinterpret_cast<const uint32_t*>(v+off(r,c));}
DI uint8_t byte(const uint8_t* v,int r,int c){return v[off(r,c)];}
struct XS {uint32_t qb[2][2][2]; float o[4][2][4];float m[2][2],l[2][2];};
template<bool P8>
DI void tile(XS& st,const uint8_t* K,const uint8_t* V,int kp,int hi,int seq,int window,float scale){
 int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
 float s[4][2][4];
 #pragma unroll
 for(int kt=0;kt<4;kt++){
  #pragma unroll
  for(int nt=0;nt<2;nt++)for(int e=0;e<4;e++)s[kt][nt][e]=0;
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   uint32_t a[4]={word(K,kt*16+g,kk*32+4*t),word(K,kt*16+g+8,kk*32+4*t),word(K,kt*16+g,kk*32+16+4*t),word(K,kt*16+g+8,kk*32+16+4*t)};
   #pragma unroll
   for(int nt=0;nt<2;nt++)mma8(s[kt][nt],a,st.qb[nt][kk][0],st.qb[nt][kk][1]);
  }
 }
 #pragma unroll
 for(int nt=0;nt<2;nt++){
  #pragma unroll
  for(int e=0;e<2;e++){
   float mx=-INFINITY;
   int query=w*16+nt*8+2*t+e,j=query/8;
   #pragma unroll
   for(int kt=0;kt<4;kt++){
    #pragma unroll
    for(int rh=0;rh<2;rh++){
     int key=kp+16*kt+g+8*rh;
     float z=s[kt][nt][rh*2+e]*scale;
     if(key>=hi || key>seq+j || (window && key<seq+j-window+1))z=-INFINITY;
     s[kt][nt][rh*2+e]=z; mx=fmaxf(mx,z);
    }
   }
   mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,4));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,8));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,16));
   float nm=fmaxf(st.m[nt][e],mx);
   // Empty partitions retain zero mass, including before their first valid key.
   float alpha=st.m[nt][e]==-INFINITY?0.f:ex(st.m[nt][e]-nm);
   float add=0;
   #pragma unroll
   for(int kt=0;kt<4;kt++)for(int rh=0;rh<2;rh++){
    float z=s[kt][nt][rh*2+e]; float p=z==-INFINITY?0.f:ex(z-nm);
    s[kt][nt][rh*2+e]=P8?p*256.f:p;add+=p;
   }
   add+=__shfl_xor_sync(0xffffffff,add,4);add+=__shfl_xor_sync(0xffffffff,add,8);add+=__shfl_xor_sync(0xffffffff,add,16);
   st.l[nt][e]=st.l[nt][e]*alpha+add;st.m[nt][e]=nm;
   #pragma unroll
   for(int dt=0;dt<4;dt++){st.o[dt][nt][e]*=alpha;st.o[dt][nt][e+2]*=alpha;}
  }
 }
 if constexpr(P8){
  #pragma unroll
  for(int kt=0;kt<2;kt++){
   uint32_t pb[2][2];
   #pragma unroll
   for(int nt=0;nt<2;nt++){
    #pragma unroll
    for(int half=0;half<2;half++){
     float p[4];
     #pragma unroll
     for(int i=0;i<4;i++){
      int src=((4*t+i)&7)*4+g/2;
      float z0=__shfl_sync(0xffffffff,s[2*kt+half][nt][0],src);
      float z1=__shfl_sync(0xffffffff,s[2*kt+half][nt][1],src);
      float z2=__shfl_sync(0xffffffff,s[2*kt+half][nt][2],src);
      float z3=__shfl_sync(0xffffffff,s[2*kt+half][nt][3],src);
      p[i]=t>=2?((g&1)?z3:z2):((g&1)?z1:z0);
     }
     pb[nt][half]=(uint32_t)fp8pair(p[0],p[1])|((uint32_t)fp8pair(p[2],p[3])<<16);
    }
   }
   #pragma unroll
   for(int dt=0;dt<4;dt++){
    uint32_t a[4]={0,0,0,0};
    #pragma unroll
    for(int i=0;i<4;i++){
     a[0]|=(uint32_t)byte(V,kt*32+4*t+i,dt*16+g)<<(8*i);
     a[1]|=(uint32_t)byte(V,kt*32+4*t+i,dt*16+g+8)<<(8*i);
     a[2]|=(uint32_t)byte(V,kt*32+16+4*t+i,dt*16+g)<<(8*i);
     a[3]|=(uint32_t)byte(V,kt*32+16+4*t+i,dt*16+g+8)<<(8*i);
    }
    #pragma unroll
    for(int nt=0;nt<2;nt++)mma8(st.o[dt][nt],a,pb[nt][0],pb[nt][1]);
   }
  }
 }else{
  #pragma unroll
  for(int kt=0;kt<4;kt++){
   uint32_t pb[2][2];
   #pragma unroll
   for(int nt=0;nt<2;nt++){
    #pragma unroll
    for(int half=0;half<2;half++){
     pb[nt][half]=transpose(hpack(s[kt][nt][half*2],s[kt][nt][half*2+1]));
    }
   }
   #pragma unroll
   for(int dt=0;dt<4;dt++){
    uint32_t a[4];
    #pragma unroll
    for(int rh=0;rh<2;rh++)for(int kh=0;kh<2;kh++){
     uint16_t p=(uint16_t)byte(V,kt*16+2*t+kh*8,dt*16+g+rh*8)|(uint16_t(byte(V,kt*16+2*t+1+kh*8,dt*16+g+rh*8))<<8);
     a[kh*2+rh]=halfpair(p);
    }
    #pragma unroll
    for(int nt=0;nt<2;nt++)mma16(st.o[dt][nt],a,pb[nt][0],pb[nt][1]);
   }
  }
 }
}
template<int ST,bool P8>
__global__ __launch_bounds__(128,2) void xqa(XP p){
 extern __shared__ __align__(128) uint8_t sm[];
 uint8_t* sk=sm; uint8_t* sv=sk+ST*4096; uint8_t* nq=sv+ST*4096;
 uint8_t* nk=nq+4096;uint8_t* nv=nk+4096;
 int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
 int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];bool last=sp==p.ns-1;
 int lo=p.window?max(0,seq-p.window+1):0;
 int chunk=((seq-lo+p.ns-1)/p.ns+63)&~63;
 int begin=min(seq,lo+sp*chunk),end=min(seq,begin+chunk),ntiles=(end-begin+63)/64;
 const int* pt=p.pt+(long long)b*p.maxpages; long long rowbytes=p.hkv*64;
 int rows[ST][2];
 auto load_rows=[&](int tileid,int slot){
  #pragma unroll
  for(int rr=0;rr<2;rr++){
   int key=begin+tileid*64+tid/4+rr*32;
   rows[slot][rr]=(tileid<ntiles&&key<end)?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):-1;
  }
 };
 #pragma unroll
 for(int st=0;st<ST;st++)load_rows(st,st);
 auto issue=[&](int tileid,int slot){
  #pragma unroll
  for(int rr=0;rr<2;rr++){
   int row=tid/4+rr*32,col=(tid&3)*16,key=begin+tileid*64+row;
   int physical=rows[slot][rr];bool valid=physical>=0;physical=valid?physical:0;
   cp(sk+slot*4096+off(row,col),p.kc+physical*rowbytes+h*64+col,valid);
   cp(sv+slot*4096+off(row,col),p.vc+physical*rowbytes+h*64+col,valid);
  }
 };
 #pragma unroll
 for(int stage=0;stage<ST-1;stage++){issue(stage,stage);commit();}
 // Every split prepares its own Q; only the final split writes new KV, with
 // no race against the historical row gathers (their upper bound is seq).
 #pragma unroll
 for(int r=0;r<16;r++){
  int query=w+4*r,j=query/8,head=h*8+query%8;
  const __nv_bfloat16* ptr=p.qkv+(long long)(b*8+j)*p.stride+head*64+2*lane;
  __nv_bfloat162 q=*reinterpret_cast<const __nv_bfloat162*>(ptr);
  float x=__bfloat162float(q.x),y=__bfloat162float(q.y),z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
  if(p.freq){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float a=x*cs-y*sn;y=y*cs+x*sn;x=a;}
  *reinterpret_cast<uint16_t*>(nq+query*64+2*lane)=fp8pair(x,y);
 }
 if(last){
  for(int i=tid;i<8192/16;i+=128)*reinterpret_cast<uint4*>(nk+i*16)=make_uint4(0,0,0,0);
  __syncthreads();
  #pragma unroll
  for(int r=0;r<4;r++){
   int row=w+4*r,j=row/2,isv=row&1;
   auto q=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+isv*p.hkv+h)*64+2*lane);
   float x=__bfloat162float(q.x),y=__bfloat162float(q.y);
   if(!isv){float z=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);x*=z;y*=z;
    if(p.freq){float sn,cs;sincosf((float)p.pos[b*8+j]*p.freq[lane],&sn,&cs);float a=x*cs-y*sn;y=y*cs+x*sn;x=a;}}
   float sc=isv?*p.vs:*p.ks;uint16_t f=fp8pair(x/sc,y/sc);
   *reinterpret_cast<uint16_t*>((isv?p.vc:p.kc)+p.loc[b*8+j]*rowbytes+h*64+2*lane)=f;
   *reinterpret_cast<uint16_t*>((isv?nv:nk)+off(j,2*lane))=f;
  }
 }
 __syncthreads();XS st;
 #pragma unroll
 for(int nt=0;nt<2;nt++){
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   st.qb[nt][kk][0]=*reinterpret_cast<uint32_t*>(nq+(w*16+nt*8+g)*64+kk*32+4*t);
   st.qb[nt][kk][1]=*reinterpret_cast<uint32_t*>(nq+(w*16+nt*8+g)*64+kk*32+16+4*t);
  }
  for(int e=0;e<2;e++){st.m[nt][e]=-INFINITY;st.l[nt][e]=0;}
  for(int dt=0;dt<4;dt++)for(int e=0;e<4;e++)st.o[dt][nt][e]=0;
 }
 float scale=p.scale*(*p.ks);
 for(int i=0;i<ntiles;i++){
  wait<ST-2>();__syncthreads();
  issue(i+ST-1,(i+ST-1)%ST);commit();
  load_rows(i+ST,(i+ST)%ST);
  tile<P8>(st,sk+(i%ST)*4096,sv+(i%ST)*4096,begin+i*64,end,seq,p.window,scale);
  __syncthreads();
 }
 wait<0>();__syncthreads();
 if(last)tile<P8>(st,nk,nv,seq,seq+8,seq,p.window,scale);
 float* po=p.partial+((long long)(b*p.hkv+h)*p.ns+sp)*64*68;
 #pragma unroll
 for(int nt=0;nt<2;nt++)for(int e=0;e<2;e++){
  int query=w*16+nt*8+2*t+e;
  for(int dt=0;dt<4;dt++){
   po[query*68+dt*16+g]=st.o[dt][nt][e];po[query*68+dt*16+g+8]=st.o[dt][nt][e+2];
  }
  if(g==0){po[query*68+64]=st.m[nt][e];po[query*68+65]=st.l[nt][e];}
 }
 __threadfence();__syncthreads();__shared__ int elected;
 if(tid==0)elected=atomicAdd(p.count+b*p.hkv+h,1)==p.ns-1;
 __syncthreads();if(!elected)return;
 // Last-CTA reduction in FP32. Every producer publishes O,m,l before its arrival.
 const volatile float* all=p.partial+(long long)(b*p.hkv+h)*p.ns*64*68;
 for(int query=w;query<64;query+=4){
  float mx=-INFINITY;
  for(int s=0;s<p.ns;s++)mx=fmaxf(mx,all[(long long)s*64*68+query*68+64]);
  float lsum=0,x=0,y=0;
  for(int s=0;s<p.ns;s++){
   long long ix=(long long)s*64*68+query*68;
   float m=all[ix+64],a=m==-INFINITY?0.f:ex(m-mx);
   lsum+=all[ix+65]*a;x+=all[ix+2*lane]*a;y+=all[ix+2*lane+1]*a;
  }
  float sc=(*p.vs)/(lsum*(P8?256.f:1.f));x*=sc;y*=sc;
  float inv=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);
  int j=query/8,head=h*8+query%8;
  auto gate=*reinterpret_cast<const __nv_bfloat162*>(p.qkv+(long long)(b*8+j)*p.stride+(p.hq+2*p.hkv+head)*64+2*lane);
  x=x*inv/(1.f+expf(-__bfloat162float(gate.x)));y=y*inv/(1.f+expf(-__bfloat162float(gate.y)));
  *reinterpret_cast<__nv_bfloat162*>(p.out+((long long)(b*8+j)*p.hq+head)*64+2*lane)=__floats2bfloat162_rn(x,y);
 }
 __syncthreads();if(tid==0)atomicExch(p.count+b*p.hkv+h,0);
}
template<int ST,bool P8>void go(XP p,cudaStream_t stream){
 constexpr int smem=ST*8192+12288;
 cudaFuncSetAttribute(xqa<ST,P8>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 xqa<ST,P8><<<dim3(p.ns,p.hkv,p.batch),128,smem,stream>>>(p);
}
}
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream){
 XP p{};p.qkv=(const __nv_bfloat16*)qkv;p.pos=(const int64_t*)pos;p.loc=(const int64_t*)loc;p.pt=(const int*)pt;p.seq=(const int*)seq;p.kc=(uint8_t*)kc;p.vc=(uint8_t*)vc;p.ks=(const float*)ks;p.vs=(const float*)vs;p.freq=(const float*)freq;p.partial=(float*)part;p.count=(int*)count;p.out=(__nv_bfloat16*)out;
 p.batch=batch;p.hq=hq;p.hkv=hkv;p.stride=stride;p.maxpages=maxpages;p.ps=ps;while((1<<p.lg)<ps)p.lg++;p.window=window;p.ns=ns;p.scale=scale;
 auto s=(cudaStream_t)stream;
 if(stages==5){if(p8)go<5,true>(p,s);else go<5,false>(p,s);}else{if(p8)go<3,true>(p,s);else go<3,false>(p,s);}
 auto err=cudaGetLastError();if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));
}


"""

_ATTN_PORT_EXT = None
_ATTN_PORT_X_EXT = {}
_ATTN_PORT_P8_EXT = {}
_ATTN_PORT_COUNTERS = {}


def _attn_port_hopper(device):
    return torch.cuda.get_device_capability(device)[0] == 9


def _attn_port_ext():
    global _ATTN_PORT_EXT
    if _ATTN_PORT_EXT is None:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        _ATTN_PORT_EXT = load_inline(
            name=f'muse_decode_attn_port_{arch}', cpp_sources=ATTN_PORT_CPP_SRC,
            cuda_sources=ATTN_PORT_CUDA_SRC,
            functions=['attn_forward', 'max_cluster', 'occupancy'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'],
            verbose=False)
    return _ATTN_PORT_EXT


def _attn_port_xext(family):
    if family not in _ATTN_PORT_X_EXT:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        source = ATTN_G_CUDA if family == 'group' else ATTN_X_CUDA
        _ATTN_PORT_X_EXT[family] = load_inline(
            name=f'muse_decode_attn_xqa_{family}_{arch}', cpp_sources=ATTN_X_CPP,
            cuda_sources=source, functions=['xforward'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'],
            verbose=False)
    return _ATTN_PORT_X_EXT[family]


def _attn_port_p8ext():
    major, minor = torch.cuda.get_device_capability()
    arch = '90a' if major == 9 else f'{major}{minor}'
    if arch not in _ATTN_PORT_P8_EXT:
        _ATTN_PORT_P8_EXT[arch] = load_inline(
            name=f'muse_decode_attn_p8_{arch}', cpp_sources=ATTN_X_CPP,
            cuda_sources=ATTN_P8_CUDA, functions=['xforward'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'],
            verbose=False)
    return _ATTN_PORT_P8_EXT[arch]


def _attn_port_counters(n, device):
    key = str(device)
    buf = _ATTN_PORT_COUNTERS.get(key)
    if buf is None or buf.numel() < n:
        buf = torch.zeros(max(n, 1 << 16), dtype=torch.int32, device=device)
        _ATTN_PORT_COUNTERS[key] = buf
    return buf


def _attn_port_raw(qkv, data):
    """Convert the layer GEMV partials to the attention lane's raw qkv layout."""
    partial, splits = qkv
    if partial.dtype == torch.float32 and partial.stride(-1) == 1:
        partial, splits = _attn_sumcast(partial, splits)
    raw = partial.sum(0) if splits > 1 else partial[0]
    cfg = data['config']
    return raw.to(torch.bfloat16).view(cfg['batch'], cfg['q_len'], raw.shape[-1]).contiguous()


def _attn_port_fa3(raw, data, setting):
    
    partial = raw.reshape(-1, QKV_SIZE).unsqueeze(0)
    state = {"inv_freq": _inv_freq(data["config"]["rope_theta"] or ROPE_THETA, raw.device)}
    q, gate = qk_prep((partial, 1), data, state, q_fp8=True)
    return _fa3_call(q, gate, data, setting)


def _attn_port_forward(raw, data):
    """Run the standalone four-stage q1 gather or its L=8 sibling."""
    cfg = data['config']; batch, length = cfg['batch'], cfg['q_len']; device = raw.device
    max_pages = data['page_table'].shape[1]; ps = cfg['page_size']; window = cfg['window']
    max_keys = min(max_pages * ps, window) if window else max_pages * ps
    
    
    ext = _attn_port_ext()
    occ = (max(1, ext.occupancy(length)),
           {cs: ext.max_cluster(length, cs) for cs in range(1, 17)})
    ns, cs = _attn_main_plan(batch, HKV // 2, max_keys, length, window, _num_sms(device), occ)
    rows = batch * (HKV // 2) * max(1, ns // cs) * length * 16
    part_o = torch.empty((max(rows, 1) * 64,), dtype=torch.float32, device=device)
    part_ml = torch.empty((max(rows, 1) * 2,), dtype=torch.float32, device=device)
    counters = _attn_port_counters(batch * (HKV // 2) * cs, device)
    part1 = part_o
    out = torch.empty((batch, length, Q_SIZE), dtype=torch.bfloat16, device=device)
    inv_freq = _inv_freq(cfg['rope_theta'], device) if cfg['rope_theta'] else None
    scale_log2 = cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634
    ext = _attn_port_ext()
    ext.attn_forward(raw.data_ptr(), data['positions'].data_ptr(), data['cache_loc'].data_ptr(),
           data['page_table'].data_ptr(), data['seq_lens'].data_ptr(), data['k_cache'].data_ptr(),
           data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
           inv_freq.data_ptr() if inv_freq is not None else 0, part_o.data_ptr(), part_ml.data_ptr(),
           counters.data_ptr(), part1.data_ptr(), 1, out.data_ptr(), batch, length,
           raw.shape[-1], HQ, HKV, max_pages, ps, window, ns, cs, scale_log2,
           torch.cuda.current_stream(device).cuda_stream)
    return out.view(batch * length, Q_SIZE)


def _attn_port_xqa(raw, data, setting):
    family, stages, waves = setting
    cfg = data['config']; batch, hq, hkv, ps = cfg['batch'], HQ, HKV, cfg['page_size']
    window = cfg['window']; maxpages = data['page_table'].shape[1]
    maxkeys = min(maxpages * ps, window) if window else maxpages * ps
    tile = 64 if family == 'group' else 128
    ns = max(1, min(64, (maxkeys + tile - 1) // tile,
                       (_num_sms(raw.device) * waves + batch * hkv - 1) // (batch * hkv)))
    out = torch.empty((batch, 8, Q_SIZE), dtype=torch.bfloat16, device=raw.device)
    part = torch.empty((batch * hkv * ns * 64 * 68,), dtype=torch.float32, device=raw.device)
    counts = _attn_port_counters(batch * hkv, raw.device)
    freq = _inv_freq(cfg['rope_theta'], raw.device) if cfg['rope_theta'] else None
    _attn_port_xext(family).xforward(
        raw.data_ptr(), data['positions'].data_ptr(), data['cache_loc'].data_ptr(),
        data['page_table'].data_ptr(), data['seq_lens'].data_ptr(), data['k_cache'].data_ptr(),
        data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        freq.data_ptr() if freq is not None else 0, part.data_ptr(), counts.data_ptr(), out.data_ptr(),
        batch, hq, hkv, raw.shape[-1], maxpages, ps, window, ns,
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634,
        stages, 0, torch.cuda.current_stream(raw.device).cuda_stream)
    return out.view(batch * 8, Q_SIZE)


def _attn_port_p8(raw, data, setting):
    stages, waves = setting
    cfg = data['config']; batch, hq, hkv, ps = cfg['batch'], HQ, HKV, cfg['page_size']
    window = cfg['window']; maxpages = data['page_table'].shape[1]
    maxkeys = min(maxpages * ps, window) if window else maxpages * ps
    ns = max(1, min(64, (maxkeys + 63) // 64,
                       (_num_sms(raw.device) * waves + batch * hkv - 1) // (batch * hkv)))
    out = torch.empty((batch, 8, Q_SIZE), dtype=torch.bfloat16, device=raw.device)
    part = torch.empty((batch * hkv * ns * 64 * 68,), dtype=torch.float32, device=raw.device)
    counts = _attn_port_counters(batch * hkv, raw.device)
    freq = _inv_freq(cfg['rope_theta'], raw.device) if cfg['rope_theta'] else None
    _attn_port_p8ext().xforward(
        raw.data_ptr(), data['positions'].data_ptr(), data['cache_loc'].data_ptr(),
        data['page_table'].data_ptr(), data['seq_lens'].data_ptr(), data['k_cache'].data_ptr(),
        data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        freq.data_ptr() if freq is not None else 0, part.data_ptr(), counts.data_ptr(), out.data_ptr(),
        batch, hq, hkv, raw.shape[-1], maxpages, ps, window, ns,
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634,
        stages, 1, torch.cuda.current_stream(raw.device).cuda_stream)
    return out.view(batch * 8, Q_SIZE)


def _attn_port_synthetic(batch, q_len, window, device, ctx=80000):
    gen = torch.Generator(device=device).manual_seed(0xD3C0 + batch * 17 + q_len + window)
    maxpages = ctx + 8 if batch == 1 else (ctx * 1075 + 999) // 1000 + 8
    seq = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    if batch > 1:
        seq = (seq.float() * (1. + .075 * (2 * torch.rand(batch, device=device, generator=gen) - 1))).to(torch.int32)
    pages = torch.randperm(batch * maxpages, generator=gen, device=device).to(torch.int32).view(batch, maxpages)
    positions = (seq[:, None].long() + torch.arange(q_len, device=device)).reshape(-1)
    loc = torch.gather(pages.long(), 1, positions.view(batch, q_len)).reshape(-1)
    shape = (batch * maxpages, 1, HKV, HD)
    k = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(FP8)
    v = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(FP8)
    partial = torch.randn((1, batch * q_len, QKV_SIZE), generator=gen, device=device, dtype=torch.float32)
    data = dict(positions=positions, cache_loc=loc, cache_seqlens=seq + q_len,
                cu_seqlens_q=torch.arange(0, batch * q_len + 1, q_len, device=device, dtype=torch.int32),
                k_cache=k, v_cache=v, k_scale=torch.ones((), device=device), v_scale=torch.ones((), device=device),
                page_table=pages, seq_lens=seq, config=dict(batch=batch, q_len=q_len, tokens=batch * q_len,
                page_size=1, window=window, rope_theta=ROPE_THETA if window else None,
                qk_scale_factor=5.473, sm_scale=.125, kv_scale=1.0),
                stream=torch.cuda.current_stream(device),scheduler_metadata=None)
    return (partial, 1), data


def _attn_port_candidates(q_len):
    out = [('layer', None)] + [('fa3', (sp, pack)) for sp, pack in
        ((sp, pack) for sp in _FA3_SPLITS for pack in _FA3_PACK)]
    if q_len == 1:
        return out + [('attn_main', None)]
    return out + [('port_verify', None)] + \
        [('xqa_group', ('group', st, w)) for st in (5, 7) for w in (1, 2, 4, 8)] + \
        [('xqa_warp', ('warp', st, w)) for st in (3, 4) for w in (1, 2, 4, 8)] + \
        [('xqa_fp8_pv', (st, w)) for st in (3, 5) for w in (1, 2, 4, 8)]


def _attn_port_variant(qkv, data, state, choice):
    label, setting = choice
    if label == 'layer':
        if data['config']['q_len']==8 and _attn_port_hopper(qkv[0].device):
            q,gate=qk_prep(qkv,data,state,q_fp8=True)
            return _attention_fa3_public(q,gate,data,state)
        return _layer_stage_attention(qkv, data, state)
    raw = _attn_port_raw(qkv, data)
    if label == 'fa3':
        return _attn_port_fa3(raw, data, setting)
    if label in ('attn_main', 'port_verify'):
        return _attn_port_forward(raw, data)
    if label.startswith('xqa_group') or label.startswith('xqa_warp'):
        return _attn_port_xqa(raw, data, setting)
    return _attn_port_p8(raw, data, setting)
def _attn_main_plan(batch, kvg, max_keys, q_len, window, sms, occ):
    """(splits per request, cluster size). One wave: every CTA resident at once, since each
    carries ~microseconds of fixed latency (page-table and query loads, the merge). Each split
    gets at least one 64-token tile. A request's splits form clusters (size <= 16) that merge
    through the cluster; more than one cluster per request merge through global memory (at
    most 32 clusters). Sliding layers (<= 2,048 keys) use one cluster per request."""
    per_sm, clusters = occ
    groups = batch * kvg
    fit = max(1, sms * per_sm // groups)
    fit = min(fit, -(-max_keys // 64), 16 if window else 512)
    sizes = [cs for cs in range(1, 17) if clusters.get(cs, 0) >= 1] or [1]
    best = None
    for cs in sizes:
        if cs > fit:
            continue
        nc = min(fit // cs, 32)
        while nc > 0 and clusters[cs] < groups * nc:
            nc -= 1
        if nc == 0:
            continue
        key = (nc * cs, -nc, cs)  
        if best is None or key > best[0]:
            best = (key, nc * cs, cs)
    if best is None:  
        return 1, 1
    return best[1], best[2]





def _projection_consumer(x, w, name, state, variant, out_dtype):
    """Time the projection with its real reduction/norm consumer."""
    p = _project_variant(x, w, name, state, variant, out_dtype)
    t = x.shape[0]
    if name == 'qkv_gate':
        return qk_prep(p, state['_proj_data'], state, q_fp8=True)
    sync = None
    if name == 'pool_router':
        return route_and_norm(p, state, state['_proj_data'], sync)
    if name == 'o':
        return boundary(p, state['_proj_stream'], state['alpha1'], state['beta1'], state['pre_ffn_w'], sync)
    return boundary(p, state['_proj_stream'], state['alpha2'], state['beta2'], state['next_pre_attn_w'],
                    sync, postw=state['post_ffn_w'])


def _seed_v9_gemv_choices(state):
    
    _GEMV_CHOICES.clear()
    for t in (1, 8, 16, 32):
        for name in ('qkv_gate', 'o', 'pool_router', 'unpooler_slice', 'unpooler'):
            if name == 'unpooler_slice' and t > 8 or name == 'unpooler' and t <= 8:
                continue
            n, k = state[name].shape
            if name == 'qkv_gate':
                choice = (32, 1024, 128, False) if t <= 16 else (64, 512, 128, False)
            elif name == 'pool_router':
                choice = (32, 1024, 128, False) if t <= 16 else (64, 512, 128, False)
            elif name == 'unpooler':
                choice = 'cublas'
            else:
                choice = (64, 512, 128, False)
            _GEMV_CHOICES[(t, n, k)] = choice
    for (t, name), choice in GB_ROWS.items():
        n, k = state[name].shape
        _GEMV_CHOICES[(t, n, k)] = choice


def _attention_reference(qkv, data, fp8_query=False):
    """Independent FP32 attention after cache store, used only on synthetic prepare inputs."""
    raw = _attn_port_raw(qkv, data).float()
    cfg = data['config']; batch, length = cfg['batch'], cfg['q_len']
    q = raw[..., :Q_SIZE].reshape(batch, length, HQ, HD)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + RMS_EPS)
    if cfg['rope_theta']:
        angle = data['positions'].float().reshape(batch, length, 1, 1) * _inv_freq(cfg['rope_theta'], raw.device)
        even, odd = q[..., ::2], q[..., 1::2]
        q = torch.stack((even * angle.cos() - odd * angle.sin(),
                         even * angle.sin() + odd * angle.cos()), -1).flatten(-2)
    if fp8_query:
        q = q.to(FP8).float() * cfg['qk_scale_factor']
    else:
        q = (q * cfg['qk_scale_factor']).to(torch.bfloat16).float()
    gate = raw[..., GATE_OFF:].reshape(batch, length, HQ, HD).sigmoid()
    out = torch.empty_like(q)
    kflat = data['k_cache'].view(-1, HKV, HD)
    vflat = data['v_cache'].view(-1, HKV, HD)
    
    for b in range(batch):
        seq = int(data['seq_lens'][b].item())
        for j in range(length):
            end = seq + j + 1
            begin = max(0, end - cfg['window']) if cfg['window'] else 0
            at = torch.arange(begin, end, device=raw.device)
            rows = data['page_table'][b, at // cfg['page_size']].long() * cfg['page_size'] + at % cfg['page_size']
            k = kflat[rows].float() * data['k_scale']
            v = vflat[rows].float() * data['v_scale']
            logits = torch.einsum('hgd,shd->hgs', q[b,j].view(HKV, HQ // HKV, HD), k) * cfg['sm_scale']
            y = torch.einsum('hgs,shd->hgd', logits.softmax(-1), v).reshape(HQ, HD)
            out[b,j] = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + RMS_EPS) * gate[b,j]
    return out.reshape(batch * length, Q_SIZE).to(torch.bfloat16)


def _attention_agrees(actual, ref, rounded):
    ref = ref.float(); got = actual.float()
    norm = ref.norm().clamp_min(1e-12)
    round_error = (rounded.float() - ref).norm() / norm
    
    bound = .8 * max(.03, min(.15, 2.5 * float(round_error)))
    err = float((got - ref).norm() / norm)
    row_norm = ref.norm(dim=-1).clamp_min(.1 * float(ref.norm(dim=-1).mean()))
    worst = float(((got - ref).norm(dim=-1) / row_norm.clamp_min(1e-12)).max())
    return bool(torch.isfinite(got).all()) and err <= bound and worst <= 5 * bound, err, bound


def _attention_store_agrees(qkv, data):
    raw = _attn_port_raw(qkv,data).float()
    cfg = data['config']
    k = raw[...,Q_SIZE:Q_SIZE+KV_SIZE].reshape(-1,HKV,HD)
    k = k * torch.rsqrt(k.square().mean(-1,keepdim=True) + RMS_EPS)
    if cfg['rope_theta']:
        angle = data['positions'].float()[:,None,None] * _inv_freq(cfg['rope_theta'],raw.device)
        even,odd = k[...,::2],k[...,1::2]
        k = torch.stack((even*angle.cos()-odd*angle.sin(),even*angle.sin()+odd*angle.cos()),-1).flatten(-2)
    v = raw[...,Q_SIZE+KV_SIZE:GATE_OFF].reshape(-1,HKV,HD)
    for name,value,scale in (('k_cache',k,data['k_scale']),('v_cache',v,data['v_scale'])):
        expected=(value/scale).to(FP8).float()
        actual=data[name].view(-1,HKV,HD)[data['cache_loc']].float()
        err=float((actual-expected).norm()/expected.norm().clamp_min(1e-12))
        if not torch.isfinite(actual).all() or err>.08:
            raise ValueError(f'{name} store error {err:.5f} exceeds 8%')


_ATTENTION_PUBLIC_SEEDS=(172903,180822,188741)


def _attention_public_draw(batch,length,window,device,seed,state):
    
    spread='fixed' if batch==1 else 'agentic'
    data=_ap_generate_input(batch=batch,ctx=80000,layer='sliding' if window else 'full',
        seed=seed,q_len=length,spread=spread,dist='soft',page=1,device=device)
    qkv=project(data['attn_input'],state['qkv_gate'],'qkv_gate',state)
    return qkv,data


def _tune_attention_portfolio(state,flush,device):
    plan={(b,l,w):('layer',None) for b in (1,16,32) for l in (1,8) for w in (True,False)}
    state['attention_public_rows']=[]
    if not _attn_port_hopper(device) or not _best5_left(10):
        print('ATTN_PORT portable control; public draws exercised by forced screen',flush=True)
        return plan
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        compiles=(pool.submit(_attn_port_ext),pool.submit(_attn_port_xext,'warp'),
                  pool.submit(_attn_port_xext,'group'),pool.submit(_attn_port_p8ext))
        for result in compiles:result.result()
    
    
    for length in (8,1):
        admitted=[]
        for choice in _attn_port_candidates(length):
            if not _best5_left():break
            try:
                errors=[]
                for window,ctx in ((0,257),(2048,2305)):
                    qkv,data=_attn_port_synthetic(1,length,window,device,ctx)
                    if 'scheduler_metadata' not in data:data['scheduler_metadata']=None
                    got=_attn_port_variant(qkv,data,state,choice)
                    _attention_store_agrees(qkv,data)
                    ref=_attention_reference(qkv,data);rounded=_attention_reference(qkv,data,True)
                    ok,err,bound=_attention_agrees(got,ref,rounded)
                    if not ok:raise ValueError(f'FP32 err={err:.5f} bound={bound:.5f} window={window}')
                    errors.append(round(err,5))
                admitted.append(choice)
                print(f'ATTN_PORT_ADMIT q={length} choice={choice} error={errors}',flush=True)
            except Exception as exc:
                print(f'ATTN_PORT_REJECT q={length} choice={choice} reason={str(exc).splitlines()[0][:150]}',flush=True)
        if not admitted and not _best5_left():continue
        if ('layer',None) not in admitted:raise RuntimeError(f'V9 attention control failed independent FP32 gate at q={length}')
        for batch in (32,16,1):
            for window in (0,2048):
                if not _best5_left(4):continue
                fixtures=[];lengths=[]
                for seed in _ATTENTION_PUBLIC_SEEDS:
                    if not _best5_left(2):break
                    qkv,data=_attention_public_draw(batch,length,window,device,seed,state)
                    fixtures.append((qkv,data));lengths.append(data['seq_lens'].cpu().tolist())
                if len(fixtures)!=len(_ATTENTION_PUBLIC_SEEDS):continue
                scores=[];rejected=[]
                for choice in admitted:
                    if not _best5_left(2):break
                    try:
                        per_seed=[]
                        for qkv,data in fixtures:
                            if not _best5_left():break
                            fn=lambda qkv=qkv,data=data,choice=choice:_attn_port_variant(qkv,data,state,choice)
                            per_seed.append(_measure_graph(fn,flush,repeats=3))
                        if len(per_seed)!=len(fixtures):continue
                        scores.append((sum(per_seed)/len(per_seed),choice,per_seed))
                    except Exception as exc:
                        if choice==('layer',None):raise
                        rejected.append((choice,type(exc).__name__+': '+(str(exc).splitlines() or [''])[0][:120]))
                
                
                control=next((r for r in scores if r[1]==('layer',None)),None)
                if control is None:continue
                us,choice,per_seed=min(scores,key=lambda r:r[0])
                plan[batch,length,bool(window)]=choice
                row=dict(batch=batch,q_len=length,window=window,seeds=list(_ATTENTION_PUBLIC_SEEDS),
                    spread='fixed' if batch==1 else 'agentic',seq_lens=lengths,selected=choice,
                    mean_us=round(us,3),control_mean_us=round(control[0],3),
                    trials=[dict(choice=c,mean_us=round(u,3),seed_us=[round(v,3) for v in values]) for u,c,values in scores],
                    rejected=rejected)
                state['attention_public_rows'].append(row)
                print(f'ATTN_PORT_PLAN b={batch} q={length} window={window} best={choice} '
                      f'mean_us={us:.3f} control_us={control[0]:.3f} draws={len(fixtures)} spread={row["spread"]}',flush=True)
    return plan


def _attention_public_summary(state):
    
    for row in state.get('attention_public_rows',[]):
        print('ATTN_PUBLIC',json.dumps({k:v for k,v in row.items() if k not in ('trials','rejected')},separators=(',',':')),flush=True)


def stage_attention(qkv,data,state):
    cfg=data['config'];key=(cfg['batch'],cfg['q_len'],bool(cfg['window']))
    return _attn_port_variant(qkv,data,state,state['attention_plan'].get(key,('layer',None)))


def _best7_run(data,state,plan,prefetch=None):
    t=data['config']['tokens']
    sync=[None,None,None]
    if plan['glue']=='best5':
        raw=(torch.empty if _best7_old_chunks(t)==1 else torch.zeros)((15*t,),device=data['stream'].device,dtype=torch.int32)
        si=raw[:3*t].view(3,t);sf=raw[3*t:].view(torch.float32).view(3,t,4)
        sync=[(sf[i],si[i]) for i in range(3)]
    split=_gate_split_on(data,state,plan)
    qkv=_best7_project(data['attn_input'],'qkv3' if split else 'qkv_gate',state,plan)
    if prefetch is not None:prefetch.begin('front')
    attn=_best7_attention(qkv,data,state,plan)
    if split:attn=_gate_split_apply(data['attn_input'],attn,state)
    if prefetch is not None:
        prefetch.join('front');prefetch.begin('pool')
    if plan.get('dense_chain', False):
        if prefetch is not None:prefetch.join('pool')
        h,stream1,pr_out=_dense_chain_call(attn,data['stream'],state,data)
        pr=(pr_out.unsqueeze(0),1)
    else:
        a=_best7_project(attn,'o',state,plan)
        h,stream1=_best7_boundary(a,data['stream'],state,plan,'b1',sync[0])
        if prefetch is not None:prefetch.join('pool')
        pr=_best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)
    x,ids,wts,lids=_best7_route(pr,state,data,plan,sync[1])
    if prefetch is not None:prefetch.begin('tail')
    moe=_best7_expert(pr,x,wts,lids,state,plan)
    if prefetch is not None:prefetch.join('tail')
    name='unpooler_slice' if t<=SLICED_MAX_TOKENS else 'unpooler'
    f=_best7_project(moe,name,state,plan)
    hidden,stream2=_best7_boundary(f,stream1,state,plan,'b2',sync[2])
    return hidden,stream2,ids


def kernel_fn(data,state):
    cfg=data['config'];key=(cfg['batch'],cfg['q_len'],cfg['window'])
    plan=state['best7_plans'].get(key,state['best7_plans']['other'])
    return _best7_run(data,state,plan,state['best7_prefetch_ops'].get(key))


def prepare(weights):
    import time
    global _BEST5_DEADLINE,_BEST6_GROUPED_ERROR
    _BEST6_GROUPED_ERROR=None
    started=time.monotonic();_BEST5_DEADLINE=started+235.
    from concurrent.futures import ThreadPoolExecutor
    device=weights['qkv_gate'].device
    with ThreadPoolExecutor(max_workers=16) as native_pool:
        dcp_future=native_pool.submit(dcp_module)
        cd_future=native_pool.submit(cd_module)
        grouped_future=native_pool.submit(_pf_grouped_module)
        native_pool.submit(_layer_attn_ext)  
        native_pool.submit(_vq8_module)  
        native_pool.submit(partial_port_ext,_PP_FAST)
        if _fa3_on_device(device):native_pool.submit(_lt_module)  
        nt2_future=native_pool.submit(_best5_nt2_module)
        recoded_future=native_pool.submit(_v9_recoded_module)
        nf_future=native_pool.submit(nf_module)
        s4_small_future=native_pool.submit(smallm_module)
        glue_cl_future=native_pool.submit(_glue_cl_ext)
        attention_futures=[]
        if _attn_port_hopper(device):
            attention_futures=[native_pool.submit(_attn_port_ext),
                native_pool.submit(_attn_port_xext,'group'),native_pool.submit(_attn_port_xext,'warp'),
                native_pool.submit(_attn_port_p8ext)]
        state=_prepare_weights(weights)
        dcp_future.result();_dcp_prepare(state)
        cd_future.result();_cd_prepare(state)
        vdsk_future=native_pool.submit(_vdsk_warm,state)
        try:grouped_future.result()
        except Exception as exc:_BEST6_GROUPED_ERROR=_best6_reason(exc)
        state['moe_nt2_mod']=nt2_future.result()
        vdsk_future.result()
        state['moe_recoded_mod']=recoded_future.result()
        nf_future.result()
        s4_small_future.result()
        glue_cl_future.result()
        try:state['moe6_adapters']=_best6_build_adapters(weights)
        except Exception as exc:state['moe6_adapters']={'available':[],'rejected':{name:_best6_reason(exc) for name in ('raw17','humming_masked','dg_w8')}}
        for future in attention_futures:
            try:future.result()
            except Exception as exc:print('BEST6_OPTIONAL_ATTN_COMPILE',_best6_reason(exc),flush=True)
    phases={'weights':time.monotonic()-started}
    device=weights['qkv_gate'].device
    _seed_v9_gemv_choices(state)
    flush=torch.empty(256<<20,device=device,dtype=torch.uint8)
    _best5_tune_moe(state,weights,flush,device)
    _best6_tune(state,weights,flush,device)
    phases['moe_nt']=time.monotonic()-started-sum(phases.values())
    state['proj_plan']=_tune_projections(state,flush,device)
    _tune_measured_lt(state,flush,device)
    _cd_force(state)
    phases['projections']=time.monotonic()-started-sum(phases.values())
    state.pop('_proj_data',None);state.pop('_proj_stream',None)
    state['attention_plan']=_tune_attention_portfolio(state,flush,device)
    state['vq8_rows']=_vq8_prepare(device)
    state['partial_port_rows']=_partial_port_prepare(device)
    phases['attention']=time.monotonic()-started-sum(phases.values())
    state['pdl_plan']={(t,p):True for t in _TUNE_TOKENS for p in ('b1','b2')}
    state['prefix_plan']={t:('regular',True) for t in _TUNE_TOKENS}
    _tune_glue(state,flush,device)
    scale_started=time.monotonic()
    state['moe_bank_raw']=state['moe_bank']
    state['moe_bank']=_prepare_recoded_bank(weights,state['moe_bank_raw'],state['moe_recoded_mod'])
    phases['scale_recode']=time.monotonic()-scale_started
    state['nf_streams']=(torch.cuda.Stream(device=device),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(device=device),torch.cuda.Event())
    s4_started=time.monotonic()
    _s4_prepare_banks(state,weights)
    torch.cuda.synchronize(device)
    phases['s5_banks']=time.monotonic()-s4_started
    _gate_split_prepare(state,device)
    _dcp_force(state)
    _best7_prepare_settings(state,weights,device)
    phases['pdl_prefix']=time.monotonic()-started-sum(phases.values())
    torch.cuda.synchronize(device)
    elapsed=time.monotonic()-started
    print(f'BEST_PREPARE elapsed={elapsed:.3f}s hard_limit=300s',flush=True)
    print(f'BEST_PHASES seconds={phases}',flush=True)
    for batch in (1,16,32):
        for length in (1,8):
            t=batch*length
            proj={name:state['proj_plan'].get((t,name),'custom') for name in
                  ('qkv_gate','o','pool_router','unpooler_slice' if t<=8 else 'unpooler')}
            for window in (2048,0):
                key=(batch,length,bool(window))
                print(f'BEST6_BASE_SHAPE b={batch} q={length} window={window} attn={state["attention_plan"].get(key,("layer",None))} '
                      f'proj={proj} prefix={state["prefix_plan"][t]} '
                      f'pdl={tuple(state["pdl_plan"][(t,p)] for p in ("b1","b2"))} moe=v9nt{state["moe_nt_plan"].get(t,1)}',flush=True)
    for m,row in state['moe_nt_rows'].items():print(f'BEST5_NT M={m} {row}',flush=True)
    _best6_summary(state)
    _attention_public_summary(state)
    _best7_summary(state)
    return state



import ctypes as ct
class _PgNative:
    def __init__(self, path):
        self.lib = ct.CDLL(path)
        P, I, S = ct.c_void_p, ct.c_int, ct.c_size_t
        self.lib.lean_error_string.restype = ct.c_char_p
        self.lib.pg_algo_bytes.restype = I
        self.algo_bytes = self.lib.pg_algo_bytes()
        self.lib.all_lt_options.argtypes = [I] * 6 + [S, P, I, ct.POINTER(I), ct.POINTER(I)]
        self.lib.all_lt_run.argtypes = [P] * 5 + [I] * 8 + [S, P]

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.lean_error_string().decode())

    def options(self, m, n, k, nn, fp32, workspace):
        codes = torch.empty((65536, self.algo_bytes), dtype=torch.uint8)
        count, heuristics = ct.c_int(), ct.c_int()
        self.check(self.lib.all_lt_options(n, m, k, n if nn else k, nn, int(fp32),
                    workspace, codes.data_ptr(), len(codes), ct.byref(count), ct.byref(heuristics)))
        return codes[:count.value].clone(), heuristics.value

    def run(self, x, w, out, workspace, codes, index, nn, fp32):
        if not 0 <= index < len(codes):
            raise ValueError('unprepared Lt ordinal')
        self.check(self.lib.all_lt_run(w.data_ptr(), x.data_ptr(), out.data_ptr(),
                    workspace.data_ptr(), codes.data_ptr(), out.shape[1], x.shape[0], x.shape[1],
                    w.stride(0), nn, int(fp32), index, 0, workspace.numel(),
                    torch.cuda.current_stream(x.device).cuda_stream))


PG_RAW = r'''

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstring>
#include <string>
#include <initializer_list>
static thread_local std::string pg_error;
extern "C" const char* lean_error_string(){return pg_error.c_str();}
static int lean_fail(const char* msg,int code){pg_error=std::string(msg)+": "+std::to_string(code);return code ? code : -1;}
#define LEAN_LT(expr) do {auto s=(expr);if(s!=CUBLAS_STATUS_SUCCESS)return lean_fail(#expr,(int)s);} while(0)
static thread_local cublasLtHandle_t lean_lt_handle=nullptr;
static int lean_lt_init(){if(!lean_lt_handle){LEAN_LT(cublasLtCreate(&lean_lt_handle));}return 0;}
extern "C" int pg_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}
// All returned cuBLASLt heuristics, plus checked explicit split-K configurations.
#include <cstdlib>
struct AllLtStorage {void* rows=nullptr;void* algorithms=nullptr;~AllLtStorage(){std::free(rows);std::free(algorithms);}};
struct AllLtLayout {
  cublasLtMatmulDesc_t op=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  int init(int n,int m,int k,int stride,int nn,int kind) {
    cudaDataType_t in=kind==2 ? CUDA_R_32F : CUDA_R_16BF;
    cudaDataType_t out=kind ? CUDA_R_32F : CUDA_R_16BF;
    LEAN_LT(cublasLtMatmulDescCreate(&op,kind==2 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;
    LEAN_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    LEAN_LT(cublasLtMatrixLayoutCreate(&a,in,nn ? n : k,nn ? k : n,stride));
    LEAN_LT(cublasLtMatrixLayoutCreate(&b,in,k,m,k));
    LEAN_LT(cublasLtMatrixLayoutCreate(&c,out,n,m,n));
    return 0;
  }
  ~AllLtLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
};
extern "C" int all_lt_options(int n,int m,int k,int stride,int nn,int kind,size_t workspace,
                              void* codes,int capacity,int* count,int* heuristic_count) {
  int err=lean_lt_init();if(err)return err;
  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;
  cublasLtMatmulPreference_t pref;LEAN_LT(cublasLtMatmulPreferenceCreate(&pref));
  auto s=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));
  if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt preference",(int)s);}
  int requested=64,got=0;
  AllLtStorage storage;
  storage.rows=std::calloc(capacity,sizeof(cublasLtMatmulHeuristicResult_t));
  storage.algorithms=std::calloc(capacity,sizeof(cublasLtMatmulAlgo_t));
  if(!storage.rows || !storage.algorithms){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt host storage",-1);}
  auto rows=(cublasLtMatmulHeuristicResult_t*)storage.rows;
  auto valid=(cublasLtMatmulAlgo_t*)storage.algorithms;
  int valid_count=0;bool overflow=false;
  do {

    s=cublasLtMatmulAlgoGetHeuristic(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,pref,requested,rows,&got);
    if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristics",(int)s);}
    if(got<requested)break;
    requested*=2;
    if(requested>capacity){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristic capacity",-1);}
  }while(true);
  cublasLtMatmulPreferenceDestroy(pref);

  auto add=[&](const cublasLtMatmulAlgo_t& algo) {
    for(int i=0;i<valid_count;++i)if(std::memcmp(valid+i,&algo,sizeof(algo))==0)return;
    cublasLtMatmulHeuristicResult_t checked={};
    auto status=cublasLtMatmulAlgoCheck(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,&algo,&checked);
    if(status==CUBLAS_STATUS_SUCCESS && checked.state==CUBLAS_STATUS_SUCCESS && checked.workspaceSize<=workspace)
      {if(valid_count==capacity)overflow=true;else valid[valid_count++]=algo;}
  };
  for(int i=0;i<got;++i)if(rows[i].state==CUBLAS_STATUS_SUCCESS && rows[i].workspaceSize<=workspace)add(rows[i].algo);
  *heuristic_count=valid_count;
  // Preserve every returned heuristic; add supported swizzles and FP32 split-K reductions.
  const int base_count=valid_count;
  for(int bi=0;bi<base_count;++bi) {
    const auto base=valid[bi];
    int32_t split_support=0;uint32_t mask=0,swizzle=0;size_t written=0;
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,&split_support,sizeof(split_support),&written));
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,&mask,sizeof(mask),&written));
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,&swizzle,sizeof(swizzle),&written));
    for(uint32_t sw=0;sw<=swizzle;++sw) {
      auto plain=base;
      if(cublasLtMatmulAlgoConfigSetAttribute(&plain,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,&sw,sizeof(sw))!=CUBLAS_STATUS_SUCCESS)continue;
      add(plain);
      if(split_support && (mask&CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {
        for(uint32_t split:{2u,4u,8u,16u,32u}) {
          auto algo=plain;uint32_t scheme=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS)continue;
          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&scheme,sizeof(scheme))!=CUBLAS_STATUS_SUCCESS)continue;
          add(algo);
        }
      }
    }
  }
  if(overflow)return lean_fail("all Lt option capacity",-1);
  for(int i=0;i<valid_count;++i)std::memcpy((char*)codes+i*sizeof(cublasLtMatmulAlgo_t),&valid[i],sizeof(cublasLtMatmulAlgo_t));
  *count=valid_count;return 0;
}
extern "C" int all_lt_run(const void* w,const void* x,void* out,void* workspace,const void* codes,
                          int n,int m,int k,int stride,int nn,int kind,int index,int accumulate,
                          size_t workspace_bytes,void* stream) {
  int err=lean_lt_init();if(err)return err;
  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;
  cublasLtMatmulAlgo_t algo;std::memcpy(&algo,(const char*)codes+index*sizeof(algo),sizeof(algo));
  float alpha=1.f,beta=accumulate ? 1.f : 0.f;
  auto status=cublasLtMatmul(lean_lt_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&algo,
                            workspace,workspace_bytes,(cudaStream_t)stream);
  return status==CUBLAS_STATUS_SUCCESS ? 0 : lean_fail("all Lt run",(int)status);
}

'''


_MEASURED_LT = {(256,'pool_router'):(0,0),(256,'qkv_gate'):(0,0),(256,'unpooler'):(0,0),
                (128,'pool_router'):(0,0),(128,'qkv_gate'):(1,23),(128,'unpooler'):(1,0),(128,'o'):(1,0)}


def _tune_measured_lt(state,flush,device):
    if not _fa3_on_device(device) or not _best5_left(10):
        return
    import hashlib
    import statistics
    major,minor=torch.cuda.get_device_capability(device)
    arch='90a' if major==9 else f'{major}{minor}'
    tag=hashlib.sha256(PG_RAW.encode()).hexdigest()[:12]
    path=load_inline(name=f'muse_decode_measured_lt_{tag}_{arch}',cpp_sources='',cuda_sources=PG_RAW,
                     functions=None,is_python_module=False,no_implicit_headers=True,
                     extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'],
                     extra_ldflags=['-lcublasLt'],verbose=False)
    bank={'mod':_PgNative(path),'layouts':{},'workspace':torch.empty(4<<30,device=device,dtype=torch.uint8)}
    state['measured_lt']=bank
    for (t,name),(nn,ordinal) in _MEASURED_LT.items():
        if not _best5_left():break
        w=state[name];n,k=w.shape;dtype=torch.float32 if name=='pool_router' else torch.bfloat16
        x=torch.randn((t,k),device=device,dtype=torch.bfloat16)
        codes,_=bank['mod'].options(t,n,k,nn,dtype==torch.float32,bank['workspace'].numel())
        if ordinal>=len(codes):
            print(f'MEASURED_LT_REJECT t={t} name={name} nn={nn} ordinal={ordinal} unavailable',flush=True)
            continue
        key=(t,name,nn);bank['layouts'][key]=(w.T.contiguous() if nn else w,codes)
        choice=('short_lt',nn,ordinal);control=state['proj_plan'][(t,name)]
        
        try:
            for _ in range(3):
                x.copy_(torch.randn_like(x))
                actual=_materialize(_project_variant(x,w,name,state,choice,dtype)).float()
                want=torch.mm(x.float(),w.float().T)
                if not torch.isfinite(actual).all() or float((actual-want).norm()/want.norm().clamp_min(1e-12))>.01:
                    raise ValueError('measured Lt recipe exceeds 1% FP32 guard')
            state['_proj_stream']=torch.randn((t,HIDDEN),device=device,dtype=torch.float32)
            state['_proj_data']={
                'positions':torch.arange(t,device=device,dtype=torch.int64),'cache_loc':torch.arange(t,device=device,dtype=torch.int64),
                'k_cache':torch.empty((t,1,HKV,HD),device=device,dtype=FP8),
                'v_cache':torch.empty((t,1,HKV,HD),device=device,dtype=FP8),
                'k_scale':torch.ones((),device=device),'v_scale':torch.ones((),device=device),
                'local_start':torch.zeros((),device=device,dtype=torch.int32),
                'config':dict(tokens=t,rope_theta=ROPE_THETA,qk_scale_factor=5.4730064863875)}
            old=lambda: _projection_consumer(x,w,name,state,control,dtype)
            new=lambda: _projection_consumer(x,w,name,state,choice,dtype)
            ratios=[]
            for repeat in range(4):
                if not _best5_left():break
                x.copy_(torch.randn_like(x))
                if repeat%2:
                    after=_measure_graph(new,flush);before=_measure_graph(old,flush)
                else:
                    before=_measure_graph(old,flush);after=_measure_graph(new,flush)
                ratios.append(after/before)
            if len(ratios)!=4:continue
            ratio=statistics.median(ratios)
            if (ratio<1.0 if (t,name) not in PROJ_PINS else PROJ_PINS[(t,name)]==choice):state['proj_plan'][(t,name)]=choice
            print(f'MEASURED_LT t={t} name={name} recipe={choice} paired={ratio:.5f} '
                  f'best={state["proj_plan"][(t,name)]}',flush=True)
        except Exception as exc:
            print(f'MEASURED_LT_REJECT t={t} name={name} reason={str(exc).splitlines()[0][:120]}',flush=True)
    keep={(t,name,choice[1]) for (t,name),choice in state['proj_plan'].items()
          if isinstance(choice,tuple) and choice[0]=='short_lt'}
    bank['layouts']={key:value for key,value in bank['layouts'].items() if key in keep}
    if not keep:
        state.pop('measured_lt')


V9_NT2_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, NT == 1 ? 2 : 1) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        for (int z=tid;z<8*NB;z+=NTHR) {\n          const int i = z & 7, n = z >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        for (int z=tid;z<16*NB;z+=NTHR) {\n          const int i = z & 15, n = z >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          const float fw0=1.f+__bfloat162float(pnw[m*16+i]); float t2 = ok ? (v*fw0)*(v*fw0) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  TORCH_CHECK(nt==2,"NT2 extension requires NT2");\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<2>, NTHR, smem_bytes(nt,nstage,M)));\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\n      static_cast<int>(nstage))\n  TORCH_CHECK(nt==2,"NT2 extension requires NT2");\n  LAUNCH(2);\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
V9_NT2_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid);\n'



_BEST5_NT2_MOD = None
_BEST5_DEADLINE = float('inf')


def _best5_left(reserve=3.):
    import time
    return time.monotonic()+reserve < _BEST5_DEADLINE


def _best5_nt2_module():
    global _BEST5_NT2_MOD
    if _BEST5_NT2_MOD is None:
        import hashlib
        tag=hashlib.sha256(V9_NT2_CUDA_SRC.encode()).hexdigest()[:12]
        _BEST5_NT2_MOD=load_inline(name='muse_best5_v9_nt2_'+tag,cpp_sources=V9_NT2_CPP_SRC,
            cuda_sources=V9_NT2_CUDA_SRC,functions=['max_smem_attr','smem_needed','resident_blocks','run'],
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',
                *(['-gencode=arch=compute_90a,code=sm_90a'] if IS_HOPPER else [])])
        _BEST5_NT2_MOD.smem_optin=_BEST5_NT2_MOD.max_smem_attr()
    return _BEST5_NT2_MOD


def _best5_rms(x):
    x=x.float()
    return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)


def _best5_fixture(weights,m,seed):
    
    
    import math
    dev=weights['router'].device
    cpu=torch.Generator().manual_seed(5354)
    mean=torch.randn(8192,generator=cpu).to(dev)
    basis=torch.randn((4,8192),generator=cpu).to(dev)
    gen=torch.Generator(device=dev).manual_seed(seed)
    coef=torch.randn((m,4),generator=gen,device=dev)
    length=1 if m==32 else 8
    share=torch.ones((m//length,length,1),device=dev)
    if length>1:share[:,1:]=.6
    share=share.reshape(m,1)
    stream=share*(.7*mean+math.sqrt(.9/4)*(coef@basis))+(1.-.9*share.square()).sqrt()*torch.randn((m,8192),device=dev,generator=gen)
    h=_best5_rms(stream)*(1.+weights['pre_ffn_norm'].float())
    logits=h@weights['router'].float().T*.1767766952966369
    scores=logits.sigmoid();ids=(scores+weights['router_bias'].float()).topk(8,dim=-1).indices
    picked=scores.gather(1,ids);tw=picked/picked.sum(-1,keepdim=True).clamp_min(1e-15)
    local_start=32*int(torch.bincount((ids//32).flatten(),minlength=8).argmax())
    local=(ids-local_start)%256;local=torch.where(local<32,local,torch.full_like(local,-1)).to(torch.int32)
    x=(_best5_rms(h@weights['pooler'].float().T)*(1.+weights['pre_expert_norm'].float())).bfloat16()
    counts=torch.bincount(local[local>=0],minlength=32)
    route=(int(counts.sum()),int((counts>0).sum()),int(counts.max()))
    return dict(hidden_states=x,topk_ids=local,topk_weights=tw.float(),
                config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0)),route


def _best5_fp32(data,weights):
    
    
    x=data['hidden_states'];ids=data['topk_ids'];tw=data['topk_weights']
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=x.device)
    result=torch.zeros_like(x,dtype=torch.float32)
    for e in range(32):
        if not _best5_left():return None
        rows,slots=torch.where(ids==e)
        if rows.numel()==0:continue
        def decode(name):
            packed=weights[name][e]
            code=torch.stack((packed&15,packed>>4),-1).flatten(-2).long()
            return lut[code&7]*torch.where((code&8)!=0,-1.,1.)*weights[name+'_sf'][e].float().repeat_interleave(16,-1)*weights[name+'_gs'][e]
        first=x[rows].float()@decode('w13').T
        inter=(torch.nn.functional.silu(first[:,:4096])*first[:,4096:]).bfloat16()
        second=inter.float()@decode('w2').T
        second*=1.+weights['post_expert_norm'].float()
        second*=torch.rsqrt(second.square().mean(-1,keepdim=True)+1e-8)
        result.index_add_(0,rows,second*tw[rows,slots,None])
    return result


def _best5_error(got,want):
    actual=got.float()
    if not bool(torch.isfinite(actual).all()):return float('inf'),float('inf')
    delta=(actual-want).norm(dim=-1);norm=want.norm(dim=-1)
    return float(delta.norm()/norm.norm().clamp_min(1e-12)),float((delta/norm.clamp_min(.1*norm.mean()).clamp_min(1e-12)).max())


def _best5_graph(fn,data):
    static={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
    for _ in range(2):fn(static)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn(static)
    return static,graph,out


def _best5_refresh(graph,data):
    for key in ('hidden_states','topk_ids','topk_weights'):graph[0][key].copy_(data[key])


def _best5_tune_moe(state,weights,flush,device):
    import statistics
    state['moe_nt_plan']={m:1 for m in (32,128,256)}
    state['moe_nt_rows']={m:dict(nt1_us=None,nt2_us=None,selected=1,reason='budget') for m in (32,128,256)}
    
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        for m in (32,128,256):
            if not _best5_left(8):break
            row=state['moe_nt_rows'][m]
            fixtures=[];routes=[];wants=[]
            for j in range(3):
                if not _best5_left(4):break
                data,route=_best5_fixture(weights,m,9000+m+7919*j)
                want=_best5_fp32(data,weights)
                if want is None:break
                fixtures.append(data);routes.append(route);wants.append(want)
            row['routes']=routes
            if len(fixtures)!=3:continue
            bank1=state['moe_bank'];bank2=dict(bank1,mod=state['moe_nt2_mod'],nt=2)
            graphs=[_best5_graph(lambda d,bank=bank:v9_kernel_fn(d,bank),fixtures[0]) for bank in (bank1,bank2)]
            worst=[0.,0.];worst_row=[0.,0.]
            for i,g in enumerate(graphs):
                for data,want in zip(fixtures,wants):
                    if not _best5_left():break
                    _best5_refresh(g,data);g[1].replay()
                    rel,wr=_best5_error(g[2],want);worst[i]=max(worst[i],rel);worst_row[i]=max(worst_row[i],wr)
                if worst[i]>.01 or worst_row[i]>.05:
                    raise RuntimeError(f'V9 NT{i+1} FP32 gate M={m}: {worst[i]:.6f}/{worst_row[i]:.6f}')
            if not _best5_left(3):continue
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            samples=[[],[]]
            for repeat in range(12):
                if not _best5_left(2):break
                data=fixtures[repeat%3]
                for g in graphs:_best5_refresh(g,data)
                for side in ((0,1) if repeat%2==0 else (1,0)):
                    flush.sum();a.record();graphs[side][1].replay();b.record();b.synchronize()
                    samples[side].append(a.elapsed_time(b)*1000.)
            if len(samples[0])!=12:continue
            ratios=[(samples[1][i]/samples[0][i]*samples[1][i+1]/samples[0][i+1])**.5 for i in range(0,12,2)]
            
            if not _best5_left(3):continue
            for i,g in enumerate(graphs):
                for data,want in zip(fixtures,wants):
                    _best5_refresh(g,data);g[1].replay()
                    rel,wr=_best5_error(g[2],want);worst[i]=max(worst[i],rel);worst_row[i]=max(worst_row[i],wr)
            if max(worst)>.01 or max(worst_row)>.05:raise RuntimeError(f'V9 replay FP32 gate M={m}: {worst}/{worst_row}')
            ratio=statistics.median(ratios)
            state['moe_nt_plan'][m]=2 if ratio<1. else 1
            row.update(nt1_us=round(statistics.median(samples[0]),3),nt2_us=round(statistics.median(samples[1]),3),
                       paired=round(ratio,6),max_pair=round(max(ratios),6),selected=state['moe_nt_plan'][m],
                       fp32=[round(x,6) for x in worst],worst_row=[round(x,6) for x in worst_row],reason='measured')
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous


def _glue_oracle(value,stream,alpha,beta,prew,postw=None,bias=None):
    p,s=value
    if bias is not None:
        pool=p[:,:,:4096].float().sum(0)
        logits=(p[:,:,4096:4352].float()+p[:,:,4352:].float()).sum(0)
        scores=(logits*ROUTER_MULT).sigmoid();ids=(scores+bias).topk(8,-1).indices
        picked=scores.gather(1,ids);tw=picked/(picked.sum(-1,keepdim=True)+1.e-15)
        local=(ids-243+256)%256;local=torch.where(local<32,local,-1).int()
        norm=(pool*torch.rsqrt(pool.square().mean(-1,keepdim=True)+RMS_EPS)*prew).bfloat16()
        return norm,ids.int(),tw,local
    branch=p.float().sum(0)
    if branch.shape[1]<8192:branch=F.pad(branch,(0,8192-branch.shape[1]))
    if postw is not None:branch=branch*postw
    st=alpha*stream+beta*branch*torch.rsqrt(branch.square().mean(-1,keepdim=True)+POST_EPS)
    norm=(st*torch.rsqrt(st.square().mean(-1,keepdim=True)+RMS_EPS)*prew).bfloat16()
    return norm,st


def _tune_glue(state,flush,device):
    """Choose geometry by mean cold producer+glue time across three fresh seeds."""
    rows=[]
    for t in _TUNE_TOKENS:
        if not _best5_left(12):break
        for phase in ('b1','b2','route'):
            if not _best5_left(12):break
            route=phase=='route';post=phase=='b2'
            name='pool_router' if route else ('o' if not post else ('unpooler_slice' if t<=8 else 'unpooler'))
            k=8192 if route else (4096 if post else 1024)
            dtype=torch.float32 if route else torch.bfloat16
            alpha=state['alpha2' if post else 'alpha1'];beta=state['beta2' if post else 'beta1']
            prew=state['pre_expert_w' if route else ('next_pre_attn_w' if post else 'pre_ffn_w')]
            postw=state['post_ffn_w'] if post else None
            bias=state['router_bias'] if route else None
            data={'local_start':torch.tensor(243,device=device,dtype=torch.int32)} if route else None
            configs=[(4,1),(8,1)] if route else [(nw,r) for nw in (8,16) for r in ((1,4,16) if t<=8 else (1,))]
            def invoke(x,stream):
                value=project(x,state[name],name,state,out_dtype=dtype)
                return route_and_norm(value,state,data) if route else boundary(value,stream,alpha,beta,prew,postw=postw)
            fixtures=[]
            for seed in (1973,9919,17041):
                gen=torch.Generator(device=device).manual_seed(seed+t+(100 if route else 200 if post else 0))
                x=torch.randn((t,k),device=device,dtype=torch.bfloat16,generator=gen)
                stream=torch.randn((t,8192),device=device,generator=gen)
                want=_glue_oracle(project(x,state[name],name,state,out_dtype=dtype),stream,alpha,beta,prew,postw,bias)
                fixtures.append((x,stream,want))
            trials=[]
            for nw,r in configs:
                if not _best5_left(8):break
                if route:_GLUE_ROUTE[t]=(nw,r)
                else:_GLUE_BOUNDARY[t,post]=(nw,r)
                seed_us=[]
                for x,stream,expected in fixtures:
                    got=invoke(x,stream)
                    for actual,want in zip(got,expected):
                        if actual.dtype==torch.int32:
                            if not torch.equal(actual,want):raise ValueError('glue router disagrees with FP32 oracle')
                        else:
                            torch.testing.assert_close(actual.float(),want.float(),atol=.004 if actual.dtype==torch.bfloat16 else .0003,
                                                       rtol=.008 if actual.dtype==torch.bfloat16 else .0003)
                    seed_us.append(_measure_graph(lambda:invoke(x,stream),flush,repeats=5))
                trials.append((sum(seed_us)/len(seed_us),nw,r,seed_us))
            if trials:
                us,nw,r,seed_us=min(trials)
                if route:_GLUE_ROUTE[t]=(nw,r)
                else:_GLUE_BOUNDARY[t,post]=(nw,r)
                rows.append(dict(t=t,phase=phase,selected=(nw,r),producer_glue_mean_us=round(us,3),
                                 trials=[(round(u,3),w,r,[round(v,3) for v in seeds]) for u,w,r,seeds in trials]))
    state['glue_geometry']=rows
    for row in rows:print('GLUE_GEOMETRY',row,flush=True)



import json,os
_pf_N_EXPERTS = 256
_pf_TOP_K = 8

@triton.jit
def _pf_offsets_kernel(counts_ptr, starts_ptr, psum_ptr, ALIGN: tl.constexpr):
    e = tl.arange(0, 32)
    c = tl.load(counts_ptr + e)
    padded = (c + ALIGN - 1) // ALIGN * ALIGN
    incl = tl.cumsum(padded, 0)
    st = incl - padded
    tl.store(starts_ptr + e, st)
    tl.store(psum_ptr + e, st + c)

@triton.jit
def _pf_gather_norm_kernel(p_ptr, w_ptr, slot_e_ptr, slot_pos_ptr, starts_ptr, a_ptr, row_of_ptr, D: tl.constexpr):
    """xn = rms(pooled[t]) * (1 + w); copy it to each of the token's local-route rows."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    x = tl.load(p_ptr + t * D + c).to(tl.float32)
    x = (x * tl.rsqrt(tl.sum(x * x, 0) / D + 1e-05) * tl.load(w_ptr + c)).to(tl.bfloat16)
    for k in tl.static_range(8):
        e = tl.load(slot_e_ptr + t * 8 + k)
        if e >= 0:
            r = tl.load(starts_ptr + e) + tl.load(slot_pos_ptr + t * 8 + k)
            tl.store(a_ptr + r.to(tl.int64) * D + c, x)
            tl.store(row_of_ptr + t * 8 + k, r)
        else:
            tl.store(row_of_ptr + t * 8 + k, -1)

@triton.jit
def _pf_swiglu_kernel(h_ptr, out_ptr, starts_ptr, counts_ptr, gs_ptr, BR: tl.constexpr, D: tl.constexpr):
    """inter = silu(gs * gate) * (gs * up) for expert e's rows (h = [gate | up], unscaled)."""
    e = tl.program_id(0)
    n = tl.load(counts_ptr + e)
    r0 = tl.program_id(1) * BR
    if r0 >= n:
        return
    st = tl.load(starts_ptr + e)
    r = r0 + tl.arange(0, BR)
    rm = r < n
    rows = (st + r).to(tl.int64)
    gs = tl.load(gs_ptr + e)
    c = tl.arange(0, D)
    g = tl.load(h_ptr + rows[:, None] * (2 * D) + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32) * gs
    u = tl.load(h_ptr + rows[:, None] * (2 * D) + D + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32) * gs
    g = g.to(tl.bfloat16).to(tl.float32)
    u = u.to(tl.bfloat16).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    tl.store(out_ptr + rows[:, None] * D + c[None, :], y.to(tl.bfloat16), mask=rm[:, None])

@triton.jit
def _pf_combine_kernel(y_ptr, row_of_ptr, slot_e_ptr, w_ptr, gs_ptr, pw_ptr, out_ptr, D: tl.constexpr):
    """out[t] = sum_k w[t,k] * rms(gs_e * y[row] * (1 + w_post), 1e-8) over the token's local routes."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    pw = tl.load(pw_ptr + c)
    acc = tl.zeros([D], tl.float32)
    for k in tl.static_range(8):
        r = tl.load(row_of_ptr + t * 8 + k)
        if r >= 0:
            e = tl.load(slot_e_ptr + t * 8 + k)
            y = (tl.load(y_ptr + r.to(tl.int64) * D + c).to(tl.float32) * tl.load(gs_ptr + e)).to(tl.bfloat16).to(tl.float32) * pw
            y = y * tl.rsqrt(tl.sum(y * y, 0) / D + 1e-08)
            acc += tl.load(w_ptr + t * 8 + k) * y
    tl.store(out_ptr + t * D + c, acc.to(tl.bfloat16))

@triton.jit
def _pf_pre_expert_norm_kernel(p_ptr, w_ptr, out_ptr, D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    x = tl.load(p_ptr + t * D + c).to(tl.float32)
    tl.store(out_ptr + t * D + c, (x * tl.rsqrt(tl.sum(x * x, 0) / D + 1e-05) * tl.load(w_ptr + c)).to(tl.bfloat16))
_pf_GROUPED_SRC = '// Canonical V6 36 KiB expert tiles -> exact BF16 register-A WGMMA.\n// No persistent converted or alternative-layout weight bank.\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <stdio.h>\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define USE_WGMMA 1\n#else\n#define USE_WGMMA 0\n#endif\n\n// Stored E4M3 -> BF16(s * 2^118), only inside the timed device path.\n__device__ __forceinline__ uint32_t muse_raw_scale118(uint32_t raw) {\n  const uint32_t a=raw&127u;\n  uint32_t b=a>=8 ? 0x7700u+(a<<4) : a>=4 ? 0x7680u+(a<<5) :\n             a>=2 ? 0x7600u+(a<<6) : a ? 0x7600u : 0u;\n  b|=(raw&128u)<<8;\n  return b|(b<<16);\n}\n\nnamespace canonical {\nconstexpr int H=4096, TB=36864, WB=32768;\nconstexpr uint32_t DQ_MASK=0x81C081C0u, LA_MASK=0x60306030u;\n__device__ __forceinline__ uint32_t rotr(uint32_t x,int s){return __funnelshift_r(x,x,s);}\n__device__ __forceinline__ uint32_t hmul2(uint32_t x,uint32_t y){uint32_t z;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(z):"r"(x),"r"(y));return z;}\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n__device__ __forceinline__ uint32_t sptr(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void wfence(){\n#if USE_WGMMA\n asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n}\n__device__ __forceinline__ void wcommit(){\n#if USE_WGMMA\n asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n#endif\n}\ntemplate<int G> __device__ __forceinline__ void wwait(){\n#if USE_WGMMA\n asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(G):"memory");\n#endif\n}\ntemplate<int N> __device__ __forceinline__ void protect(float (&a)[N/2]){\n#pragma unroll\n for(int j=0;j<N/2;++j)asm volatile("":"+f"(a[j])::"memory");\n}\n__device__ __forceinline__ void keep(uint32_t (&a)[4]){asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]));}\ntemplate<int N> __device__ __forceinline__ void atom(float (&d)[N/2],uint32_t (&a)[4],const uint8_t* xs,int s){\n#if USE_WGMMA\n const uint64_t desc=((uint64_t)(sptr(xs)>>4))|((uint64_t)64<<32)|((uint64_t)1<<62);\n const uint64_t base=desc+(s*2);\n { const uint64_t desc=base;\n  if constexpr (N == 8) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));\n  }\n  else if constexpr (N == 16) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])\n      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));\n  }\n  else if constexpr (N == 32) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])\n      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));\n  }\n  else if constexpr (N == 64) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));\n  }\n  else if constexpr (N == 128) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));\n  }\n  else if constexpr (N == 256) {\n    asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, 1, 1, 0; }\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])\n      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));\n  }\n }\n#else\n const int lane=threadIdx.x&31,c=lane&3,g=lane>>2;\n#pragma unroll\n for(int i=0;i<N/8;++i){\n  const int n=8*i+g,k=s*32+c*4;\n  const uint8_t* r=xs+n*128;\n  uint32_t b0=*(const uint32_t*)(r+(((k>>4)^(n&7))<<4)+(k&15));\n  uint32_t b1=*(const uint32_t*)(r+((((k+16)>>4)^(n&7))<<4)+((k+16)&15));\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n   :"+f"(d[i*4]),"+f"(d[i*4+1]),"+f"(d[i*4+2]),"+f"(d[i*4+3])\n   :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int N,int MODE> __device__ __forceinline__ void load_x(uint8_t* dst,const __nv_bfloat16* x,const int* pairs,int off,int ne,int nb,int kg){\n for(int j=threadIdx.x;j<N*32;j+=128){\n  int row=j/32,l=j%32,idx=nb*N+row;\n  // Native A\'s K permutation: logical pair 8*s+4*j+q -> physical pair 8*q+2*s+j.\n  int physical=8*(l&3)+2*(l>>3)+((l>>2)&1);\n  int pos=off+idx, xr=idx<ne?(MODE?pairs[pos]>>3:pos):0;\n  const void* src=x+(size_t)xr*H+kg*64+physical*2;\n  void* out=dst+row*128+(((l>>2)^(row&7))*4+(l&3))*4;\n  int bytes=idx<ne?4:0;\n  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"::"r"(sptr(out)),"l"(src),"r"(bytes):"memory");\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int N,int MODE,int STAGES,int RB>\n__global__ void __launch_bounds__(128,2) grouped(const uint8_t* weights,const float* gs,const float* gs_up,const __nv_bfloat16* x,__nv_bfloat16* out,\n const int* pairs,const int* offsets,const int* tasks){\n extern __shared__ __align__(1024) uint8_t shared[];\n constexpr int XBYTES=N*128, MTILES=MODE?512:256, NFB=MTILES/(4*RB);\n const int t=threadIdx.x,warp=t>>5,lane=t&31;\n const int total=tasks[32]*NFB;\n for(int item=blockIdx.x;item<total;item+=gridDim.x){\n  int ti=item/NFB,fb=item%NFB,e=0;\n#pragma unroll\n  for(int d=16;d;d>>=1)if(tasks[e+d]<=ti)e+=d;\n  int nb=ti-tasks[e],off=offsets[e],ne=offsets[e+1]-off;\n  const uint8_t* wt=weights+((size_t)e*MTILES+fb*(4*RB)+warp)*TB;\n  float acc[RB][N/2];\n#pragma unroll\n  for(int rb=0;rb<RB;++rb)\n#pragma unroll\n  for(int j=0;j<N/2;++j)acc[rb][j]=0.f;\n#pragma unroll\n  for(int k=0;k<STAGES;++k)load_x<N,MODE>(shared+k*XBYTES,x,pairs,off,ne,nb,k);\n  uint32_t A[2][RB][4] = {};\n  for(int kg=0;kg<64;++kg){\n   int st=kg%STAGES;\n   // At the tail, fewer than STAGES groups remain: wait for every outstanding copy.\n   if(kg+STAGES<=64)asm volatile("cp.async.wait_group %0;"::"n"(STAGES-1):"memory");\n   else asm volatile("cp.async.wait_group 0;":::"memory");\n   __syncthreads();\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n   uint32_t o[RB][16],sr[RB],su[RB];\n#pragma unroll\n   for(int rb=0;rb<RB;++rb){\n    const uint8_t* wp=wt+rb*4*TB;\n    uint4 w=*(const uint4*)(wp+(kg*32+lane)*16);\n    uint16_t sw=*(const uint16_t*)(wp+WB+(kg/8)*512+lane*16+(kg%8)*2);\n    sr[rb]=muse_raw_scale118(sw&255u);\n    su[rb]=muse_raw_scale118(sw>>8);\n    dq8(w.x,w.y,o[rb]);dq8(w.z,w.w,o[rb]+8);\n   }\n#pragma unroll\n   for(int k=0;k<4;++k){\n#pragma unroll\n    for(int rb=0;rb<RB;++rb){\n     const uint32_t* os=o[rb]+(k>>1)*8+(k&1)*4;\n     uint32_t* a=A[k&1][rb];\n     a[0]=hmul2(os[0],sr[rb]);a[1]=hmul2(os[1],su[rb]);a[2]=hmul2(os[2],sr[rb]);a[3]=hmul2(os[3],su[rb]);\n     protect<N>(acc[rb]);\n    }\n    wfence();\n#pragma unroll\n    for(int rb=0;rb<RB;++rb)atom<N>(acc[rb],A[k&1][rb],shared+st*XBYTES,k);\n    wcommit();\n#pragma unroll\n    for(int rb=0;rb<RB;++rb)protect<N>(acc[rb]);\n    wwait<1>();\n#pragma unroll\n    for(int rb=0;rb<RB;++rb)keep(A[(k&1)^1][rb]);\n   }\n   wwait<0>();\n#pragma unroll\n   for(int rb=0;rb<RB;++rb){protect<N>(acc[rb]);keep(A[1][rb]);}\n   __syncthreads();\n   if(kg+STAGES<64)load_x<N,MODE>(shared+st*XBYTES,x,pairs,off,ne,nb,kg+STAGES);\n  }\n  const float sc=gs[e]*256.f,scu=gs_up[e]*256.f;\n#pragma unroll\n  for(int rb=0;rb<RB;++rb)\n#pragma unroll\n  for(int j=0;j<N/8;++j){\n#pragma unroll\n   for(int q=0;q<2;++q){int col=8*j+2*(lane&3)+q,idx=nb*N+col;\n    if(idx<ne){int pos=off+idx;\n     if constexpr(MODE){\n      float g=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+q]*sc));\n      float u=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+2+q]*scu));\n      out[(size_t)pos*H+(fb*RB+rb)*32+warp*8+(lane>>2)]=__float2bfloat16_rn((g/(1.f+__expf(-g)))*u);\n     }else{\n#pragma unroll\n      for(int h=0;h<2;++h)out[(size_t)pos*H+(fb*RB+rb)*64+warp*16+(lane>>2)+8*h]=__float2bfloat16_rn(acc[rb][4*j+2*h+q]*sc);\n     }\n    }\n   }\n  }\n  __syncthreads();\n }\n}\n} // namespace canonical\nvoid gemm(torch::Tensor w,torch::Tensor gs,torch::Tensor gs_up,torch::Tensor x,torch::Tensor out,\n torch::Tensor pairs,torch::Tensor offsets,torch::Tensor tasks,int64_t mode,int64_t width,int64_t grid,int64_t stages,int64_t rb){\n auto stream=at::cuda::getCurrentCUDAStream();\n#define RUN(N,M,S,R) do { \\\n cudaFuncSetAttribute(canonical::grouped<N,M,S,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,N*128*S); \\\n canonical::grouped<N,M,S,R><<<grid,128,N*128*S,stream>>>(w.data_ptr<uint8_t>(),gs.data_ptr<float>(),gs_up.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)out.data_ptr(),pairs.data_ptr<int>(),offsets.data_ptr<int>(),tasks.data_ptr<int>()+(__builtin_ctz(N)-3)*33); } while(0)\n#define PICK(M) if(rb==1){if(width==64){RUN(64,M,2,1);}else if(width==128){RUN(128,M,2,1);}else if(stages==2){RUN(256,M,2,1);}else{RUN(256,M,3,1);}} \\\n else if(width==0){RUN(8,M,2,2);RUN(16,M,2,2);RUN(32,M,2,2);RUN(64,M,2,2);RUN(128,M,2,2);} \\\n else if(width==64){RUN(64,M,2,2);}else if(stages==2){RUN(128,M,2,2);}else{RUN(128,M,3,2);}\n if(mode){PICK(1);}else{PICK(0);}\n#undef PICK\n#undef RUN\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n\n// Direct canonical V6 reader; bank contract docs/reference/expert_layout.md.\n// Token-tiled, warp-specialized WGMMA reader of canonical V6 m16 tiles.\n// Each V6 tile is 32768 packed-code bytes followed by 4096 encoded-scale bytes.\n// W13 rows are eight gate + eight up; W2 rows are sixteen natural output rows.\n// Activations and outputs use the inverse V6 K permutation inside 128B-swizzled\n// [token_tile, k64, token128, k64] buffers. Weights load directly into registers;\n// only activation tiles are staged by the producer. Separate gate/up FP32 scales.\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n#include <stdint.h>\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n\n#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define MOE_WGMMA 1\n#else\n#define MOE_WGMMA 0\n#endif\n\nnamespace moe {\n\nconstexpr int BN = 128;              // tokens per tile\nconstexpr int X_BYTES = BN * 128;    // 16 KB\nconstexpr int STAGE_BYTES = X_BYTES;\n// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]\ntemplate <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }\ntemplate <int MODE, int STAGES> constexpr int smem_bytes() {\n  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;\n}\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "WAIT_%=:\\n"\n      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n"\n      "@!p bra WAIT_%=;\\n"\n      "}\\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"\n               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\n}\n__device__ __forceinline__ void bulk_wait_read() {\n  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");\n}\n__device__ __forceinline__ void fence_async_smem() {\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n}\n__device__ __forceinline__ void named_bar(int id, int n) {\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\n}\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t r;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));\n  return r;\n}\n\n// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.\nconstexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;\nconstexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;\n\n// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).\n__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {\n  uint32_t xm = x & 0x77777777u;\n  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);\n  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);\n  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);\n  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);\n  uint32_t se = x << 4;\n  uint32_t s03 = prmt(se, x, 0x5140u);\n  uint32_t s47 = prmt(se, x, 0x7362u);\n  h03 |= s03 & 0x80808080u;\n  h47 |= s47 & 0x80808080u;\n  r[0] = prmt(l03, h03, 0x5140u);\n  r[1] = prmt(l03, h03, 0x7362u);\n  r[2] = prmt(l47, h47, 0x5140u);\n  r[3] = prmt(l47, h47, 0x7362u);\n}\n\n// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).\n// Stored signed E4M3 bytes remain unchanged in the weight bank.\n// Reconstruct |sf| * 2^118 exactly in BF16. Zero has its own code.\n__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {\n  uint32_t lo = two_bytes & 255, hi = (two_bytes >> 8) & 255;\n  uint32_t bits = (muse_raw_scale118(lo)&65535u) | (muse_raw_scale118(hi)<<16);\n  return *reinterpret_cast<__nv_bfloat162*>(&bits);\n}\n\n__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {\n  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);\n  v = __hmul2(v, s);\n  return *reinterpret_cast<uint32_t*>(&v);\n}\n\n// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.\n// Sixteen re-encoded E2M1 codes in A,B -> eight BF16x2 words.\n// The bit planes represent FP4 * 2^-126; multiplication by the encoded scale\n// yields the exact NVFP4 weight / 256. The epilogue restores the power of two.\n__device__ __forceinline__ void make_frag(uint32_t A, uint32_t B, int half,\n      __nv_bfloat162 sf, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {\n  constexpr uint32_t MASK = 0x81C081C0u, LA = 0x60306030u;\n  uint32_t r[4];\n  if (half == 0) {\n    r[0] = A & MASK;\n    r[1] = __funnelshift_r(A,A,4) & MASK;\n    r[2] = __funnelshift_r(A,A,10) & MASK;\n    r[3] = __funnelshift_r(B,B,2) & MASK;\n  } else {\n    r[0] = __funnelshift_r(B,B,6) & MASK;\n    r[1] = __funnelshift_r(B,B,12) & MASK;\n    uint32_t W = (A & LA) | (B & ~LA);\n    r[2] = __funnelshift_r(W,W,24) & MASK;\n    r[3] = __funnelshift_r(W,W,30) & MASK;\n  }\n  __nv_bfloat162 slo = __low2bfloat162(sf), shi = __high2bfloat162(sf);\n  a[0] = bmul(r[0], slo); a[1] = bmul(r[1], shi);\n  a[2] = bmul(r[2], slo); a[3] = bmul(r[3], shi);\n}\n\n__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {\n  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).\n  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\n}\n\n#if MOE_WGMMA\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\ntemplate <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }\n#else\n__device__ __forceinline__ void wg_fence() {}\n__device__ __forceinline__ void wg_commit() {}\ntemplate <int N> __device__ __forceinline__ void wg_wait() {}\n#endif\n\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\n  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));\n}\n__device__ __forceinline__ void fence_acc(float (&acc)[64]) {\n#pragma unroll\n  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");\n}\n\n// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)\n__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {\n#if MOE_WGMMA\n  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %69, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "\n      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "\n      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\n      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\n#else\n  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.\n  const int lane = threadIdx.x & 31;\n  const int g = lane >> 2, c = lane & 3;\n  if (!scale_d) {\n#pragma unroll\n    for (int i = 0; i < 64; ++i) d[i] = 0.f;\n  }\n#pragma unroll\n  for (int i = 0; i < BN / 8; ++i) {\n    const int n = 8 * i + g;\n    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row\n    const uint8_t* row = xs + n * 128;\n    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));\n    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));\n    asm volatile(\n        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "\n        "{%0, %1, %2, %3};"\n        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])\n        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n  }\n#endif\n}\n\nstruct Params {\n  const uint8_t* wblob;\n  const float* gs_up;\n  const float* gs;\n  const uint8_t* x;\n  uint8_t* y;\n  const int* tile_start;\n  int nfb, kt, kt_out;\n};\n\ntemplate <int MODE, int STAGES>\n__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {\n  constexpr int EPI_BYTES = epi_bytes<MODE>();\n  extern __shared__ uint8_t smem_raw[];\n  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));\n  uint8_t* epi = smem + STAGES * STAGE_BYTES;\n  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);\n  uint64_t* empty = full + STAGES;\n  int* tstart = reinterpret_cast<int*>(empty + STAGES);\n\n  const int tid = threadIdx.x;\n  const int wg = tid / 128;\n  if (tid < 33) tstart[tid] = p.tile_start[tid];\n  if (tid == 0) {\n    for (int s = 0; s < STAGES; ++s) {\n      mbar_init(&full[s], 1);\n      mbar_init(&empty[s], 8);\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  __syncthreads();\n  const int num_tiles = tstart[32];\n  const int total = num_tiles * p.nfb;\n  const int KT = p.kt;\n\n  if (wg == 0) {\n#if MOE_WGMMA\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\n#endif\n    if (tid != 0) return;\n    int s = 0;\n    uint32_t phase = 0;\n    for (int item = blockIdx.x; item < total; item += gridDim.x) {\n      const int tt = item / p.nfb, fb = item % p.nfb;\n      int e = 0;\n      while (e < 31 && tstart[e + 1] <= tt) ++e;\n      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;\n      for (int k = 0; k < KT; ++k) {\n        mbar_wait(&empty[s], phase ^ 1);\n        uint8_t* st = smem + s * STAGE_BYTES;\n        mbar_expect_tx(&full[s], STAGE_BYTES);\n        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n      }\n    }\n    return;\n  }\n\n#if MOE_WGMMA\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");\n#endif\n  const int cw = wg - 1;                 // consumer warpgroup 0/1\n  const int t = tid - 128 * wg;          // thread within the warpgroup\n  const int warp = t >> 5, lane = t & 31;\n  const int slot = warp * 8 + (lane >> 2);\n  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;\n  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);\n  int s = 0;\n  uint32_t phase = 0;\n  float acc[2][64];\n\n  for (int item = blockIdx.x; item < total; item += gridDim.x) {\n    const int tt = item / p.nfb, fb = item % p.nfb;\n    int e = 0;\n    while (e < 31 && tstart[e + 1] <= tt) ++e;\n\n    uint32_t A0[2][4], A1[2][4];\n    uint4 raw[2];\n    uint2 sfw[2];\n    // V6 m16 tiles hold paired gate/up rows (or 16 down rows). No converted bank.\n    auto load_words = [&](int k) {\n#pragma unroll\n      for (int j = 0; j < 2; ++j) {\n        constexpr int MTILES = MODE ? 512 : 256;\n        const uint8_t* wt = p.wblob + ((size_t)e * MTILES + fb*16 + cw*8 + j*4 + warp)*36864;\n        raw[j] = *reinterpret_cast<const uint4*>(wt + (k*32 + lane)*16);\n        uint16_t sc = *reinterpret_cast<const uint16_t*>(wt + 32768 + (k/8)*512 + lane*16 + (k%8)*2);\n        // Every k16 instruction uses the same two row scales after V6\'s K permutation.\n        sfw[j].x = uint32_t(sc) | (uint32_t(sc)<<16);\n        sfw[j].y = sfw[j].x;\n      }\n    };\n    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n      wg_fence();\n      mma_block(acc[0], F[0], xs, b, scale_d);\n      mma_block(acc[1], F[1], xs, b, scale_d);\n      wg_commit();\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n    };\n    // prologue: first k-tile\'s words, block 0\'s fragments\n    mbar_wait(&full[s], phase);\n    load_words(0);\n    int cur_s = s;\n#pragma unroll\n    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n\n    for (int k = 0; k < KT; ++k) {\n      const uint8_t* xs = smem + cur_s * STAGE_BYTES;\n      // block 0 (A0)\n      issue(A0, xs, 0, k > 0);\n      if (k > 0 && MOE_WGMMA) {\n        wg_wait<1>();\n        keep(A1[0]); keep(A1[1]);\n        __syncwarp();\n        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done\n      }\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 1, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);\n      // block 1 (A1)\n      issue(A1, xs, 1, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 0, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);\n      // block 2 (A0)\n      issue(A0, xs, 2, 1);\n      wg_wait<1>();\n      keep(A1[0]); keep(A1[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 1, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);\n      // block 3 (A1)\n      issue(A1, xs, 3, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#if !MOE_WGMMA\n      // mma.sync has consumed this tile completely. Release it before waiting\n      // for the next tile, including the one-stage down-projection fallback.\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&empty[cur_s]);\n#endif\n      if (k + 1 < KT) {\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n        mbar_wait(&full[s], phase);\n        load_words(k + 1);\n        cur_s = s;\n#pragma unroll\n        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n      }\n    }\n    wg_wait<0>();\n    fence_acc(acc[0]);\n    fence_acc(acc[1]);\n    keep(A1[0]); keep(A1[1]);\n    __syncwarp();\n    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);\n    if (++s == STAGES) { s = 0; phase ^= 1; }\n\n    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----\n    const float gsc = p.gs[e] * 256.f;\n    const float usc = p.gs_up[e] * 256.f;\n    if (t == 0) bulk_wait_read();       // previous item\'s store has finished reading my_epi\n    named_bar(1 + cw, 128);\n    const int c2 = 2 * (lane & 3);\n    constexpr int NT = MODE == 1 ? 1 : 2;\n#pragma unroll\n    for (int j = 0; j < 2; ++j) {\n      uint8_t* tile = my_epi + (MODE ? 0 : j * X_BYTES);\n#pragma unroll\n      for (int i = 0; i < BN / 8; ++i) {\n#pragma unroll\n        for (int h = 0; h < (MODE ? 1 : 2); ++h) {\n          const int rho = MODE ? j*32 + warp*8 + (lane>>2) : warp*16 + (lane>>2) + h*8;\n          // Natural feature -> V6 activation-K pair order, inverse of load_x\'s permutation.\n          const int pair = rho >> 1;\n          const int rp = 2*((pair>>3) + ((pair&1)<<2) + ((pair&6)<<2)) + (rho&1);\n#pragma unroll\n          for (int q = 0; q < 2; ++q) {\n            const int n = 8 * i + c2 + q;\n            float v;\n            if constexpr (MODE == 1) {\n              float gg = __bfloat162float(__float2bfloat16_rn(acc[j][4*i+q]*gsc));\n              float uu = __bfloat162float(__float2bfloat16_rn(acc[j][4*i+2+q]*usc));\n              v = gg / (1.f + __expf(-gg)) * uu;\n            } else {\n              v = acc[j][4*i+2*h+q]*gsc;\n            }\n            *reinterpret_cast<__nv_bfloat16*>(tile+n*128+(((rp>>3)^(n&7))<<4)+(rp&7)*2) = __float2bfloat16_rn(v);\n          }\n        }\n      }\n    }\n    fence_async_smem();\n    named_bar(1 + cw, 128);\n    if (t == 0) {\n#pragma unroll\n      for (int j = 0; j < NT; ++j) {\n        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;\n        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);\n      }\n    }\n  }\n  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");\n}\n\n}  // namespace moe\n\nvoid moe_gemm(torch::Tensor wblob, torch::Tensor gs, torch::Tensor gs_up, torch::Tensor x, torch::Tensor y,\n              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {\n  moe::Params p;\n  p.wblob = wblob.data_ptr<uint8_t>();\n  p.gs_up = gs_up.data_ptr<float>();\n  p.gs = gs.data_ptr<float>();\n  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());\n  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());\n  p.tile_start = tile_start.data_ptr<int>();\n  p.nfb = (int)nfb;\n  p.kt = (int)kt;\n  p.kt_out = (int)kt_out;\n  auto stream = at::cuda::getCurrentCUDAStream();\n  int dev = 0, major = 0;\n  cudaGetDevice(&dev);\n  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\n#define MOE_LAUNCH(M, S)                                                                                  \\\n  do {                                                                                                     \\\n    cudaFuncSetAttribute(moe::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \\\n                         moe::smem_bytes<M, S>());                                                         \\\n    moe::moe_gemm_kernel<M, S><<<(int)grid, 384, moe::smem_bytes<M, S>(), stream>>>(p);                   \\\n  } while (0)\n  if (major == 9) {\n    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);\n  } else {\n    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);\n  }\n#undef MOE_LAUNCH\n}\n\n// Grouped NVFP4 W4A16 expert GEMM for Hopper (wgmma RS), with an mma.sync emulation of the\n// same data path for other archs (local testing).\n//\n// out^T[feat, tok] = W_e[feat, :] . X[tok, :] for each work item (token tile tt of expert e,\n// weight block fb of 256 rows). Weights are the A operand (decoded from NVFP4 to BF16 in registers,\n// exact: e2m1 x e4m3 has <= 6 significant bits), tokens the B operand (bf16, K-major, 128B swizzle,\n// one 16 KB bulk copy per k-tile from a pre-swizzled token-tile buffer).\n//\n// Layouts (all built by the host / gather kernels):\n//   wblob  [E][NFB][KT][4][128 thr][16 B]   thread t\'s 16 bytes per m64 block per k-tile, in WGMMA\n//                                           A-fragment order (quack\'s nvfp4 repack, per m64 block)\n//   sfblob [E][NFB][KT][4][32 slot][4 blk][2] e4m3 scale of rows (r, r+8) per k16 block\n//   x      [tiles][KT][128 tok][128 B]      bf16 rows, 16-byte chunk c of row r stored at c ^ (r & 7)\n//   y      [tiles][KT_out][128 tok][128 B]  same layout (feeds the next GEMM / the combine)\n//   tile_start [33]                         token tiles of expert e are [tile_start[e], tile_start[e+1])\n//\n// MODE 1 (gate/up, SwiGLU): CTA rows = [gate f0..f0+63, up f0..f0+63, gate f0+64.., up f0+64..];\n//   consumer WG w owns m64 blocks (2w, 2w+1) = (gate, up) of features f0 + 64w ..; out = silu(gs g) * (gs u).\n// MODE 0 (down): WG w owns m64 blocks 2w, 2w+1; out = gs * acc.\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n#include <stdint.h>\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n\n#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define MOE_WGMMA 1\n#else\n#define MOE_WGMMA 0\n#endif\n\nnamespace raw11 {\n\nconstexpr int BN = 128;              // tokens per tile\nconstexpr int X_BYTES = BN * 128;    // 16 KB\nconstexpr int W_BYTES = 4 * 2048;    // 8 KB\nconstexpr int SF_BYTES = 4 * 256;    // 1 KB\nconstexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)\n// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]\ntemplate <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }\ntemplate <int MODE, int STAGES> constexpr int smem_bytes() {\n  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;\n}\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "WAIT_%=:\\n"\n      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n"\n      "@!p bra WAIT_%=;\\n"\n      "}\\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"\n               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\n}\n__device__ __forceinline__ void bulk_wait_read() {\n  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");\n}\n__device__ __forceinline__ void fence_async_smem() {\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n}\n__device__ __forceinline__ void named_bar(int id, int n) {\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\n}\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t r;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));\n  return r;\n}\n\n// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.\nconstexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;\nconstexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;\n\n// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).\n__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {\n  uint32_t xm = x & 0x77777777u;\n  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);\n  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);\n  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);\n  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);\n  uint32_t se = x << 4;\n  uint32_t s03 = prmt(se, x, 0x5140u);\n  uint32_t s47 = prmt(se, x, 0x7362u);\n  h03 |= s03 & 0x80808080u;\n  h47 |= s47 & 0x80808080u;\n  r[0] = prmt(l03, h03, 0x5140u);\n  r[1] = prmt(l03, h03, 0x7362u);\n  r[2] = prmt(l47, h47, 0x5140u);\n  r[3] = prmt(l47, h47, 0x7362u);\n}\n\n// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).\n__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {\n  uint32_t h2;\n  asm("{\\n.reg .b16 t;\\nmov.b16 t, %1;\\ncvt.rn.f16x2.e4m3x2 %0, t;\\n}" : "=r"(h2) : "h"((uint16_t)two_bytes));\n  __half2 h = *reinterpret_cast<__half2*>(&h2);\n  float2 f = __half22float2(h);\n  return __floats2bfloat162_rn(f.x, f.y);\n}\n\n__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {\n  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);\n  v = __hmul2(v, s);\n  return *reinterpret_cast<uint32_t*>(&v);\n}\n\n// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.\n__device__ __forceinline__ void make_frag(uint32_t x, __nv_bfloat162 s, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {\n  uint32_t r[4];\n  decode8(x, lo_a, hi_a, r);\n  __nv_bfloat162 slo = __low2bfloat162(s), shi = __high2bfloat162(s);\n  a[0] = bmul(r[0], slo);\n  a[1] = bmul(r[1], shi);\n  a[2] = bmul(r[2], slo);\n  a[3] = bmul(r[3], shi);\n}\n\n__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {\n  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).\n  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\n}\n\n#if MOE_WGMMA\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\ntemplate <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }\n#else\n__device__ __forceinline__ void wg_fence() {}\n__device__ __forceinline__ void wg_commit() {}\ntemplate <int N> __device__ __forceinline__ void wg_wait() {}\n#endif\n\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\n  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));\n}\n__device__ __forceinline__ void fence_acc(float (&acc)[64]) {\n#pragma unroll\n  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");\n}\n\n// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)\n__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {\n#if MOE_WGMMA\n  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %69, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "\n      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "\n      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\n      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\n#else\n  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.\n  const int lane = threadIdx.x & 31;\n  const int g = lane >> 2, c = lane & 3;\n  if (!scale_d) {\n#pragma unroll\n    for (int i = 0; i < 64; ++i) d[i] = 0.f;\n  }\n#pragma unroll\n  for (int i = 0; i < BN / 8; ++i) {\n    const int n = 8 * i + g;\n    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row\n    const uint8_t* row = xs + n * 128;\n    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));\n    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));\n    asm volatile(\n        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "\n        "{%0, %1, %2, %3};"\n        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])\n        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n  }\n#endif\n}\n\nstruct Params {\n  const uint8_t* wblob;\n  const uint8_t* sfblob;\n  const float* gs;\n  const uint8_t* x;\n  uint8_t* y;\n  const int* tile_start;\n  int nfb, kt, kt_out;\n};\n\ntemplate <int MODE, int STAGES>\n__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {\n  constexpr int EPI_BYTES = epi_bytes<MODE>();\n  extern __shared__ uint8_t smem_raw[];\n  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));\n  uint8_t* epi = smem + STAGES * STAGE_BYTES;\n  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);\n  uint64_t* empty = full + STAGES;\n  int* tstart = reinterpret_cast<int*>(empty + STAGES);\n\n  const int tid = threadIdx.x;\n  const int wg = tid / 128;\n  if (tid < 33) tstart[tid] = p.tile_start[tid];\n  if (tid == 0) {\n    for (int s = 0; s < STAGES; ++s) {\n      mbar_init(&full[s], 1);\n      mbar_init(&empty[s], 8);\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  __syncthreads();\n  const int num_tiles = tstart[32];\n  const int total = num_tiles * p.nfb;\n  const int KT = p.kt;\n\n  if (wg == 0) {\n#if MOE_WGMMA\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\n#endif\n    if (tid != 0) return;\n    int s = 0;\n    uint32_t phase = 0;\n    for (int item = blockIdx.x; item < total; item += gridDim.x) {\n      const int tt = item / p.nfb, fb = item % p.nfb;\n      int e = 0;\n      while (e < 31 && tstart[e + 1] <= tt) ++e;\n      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;\n      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;\n      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;\n      for (int k = 0; k < KT; ++k) {\n        mbar_wait(&empty[s], phase ^ 1);\n        uint8_t* st = smem + s * STAGE_BYTES;\n        mbar_expect_tx(&full[s], STAGE_BYTES);\n        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n      }\n    }\n    return;\n  }\n\n#if MOE_WGMMA\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");\n#endif\n  const int cw = wg - 1;                 // consumer warpgroup 0/1\n  const int t = tid - 128 * wg;          // thread within the warpgroup\n  const int warp = t >> 5, lane = t & 31;\n  const int slot = warp * 8 + (lane >> 2);\n  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;\n  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);\n  int s = 0;\n  uint32_t phase = 0;\n  float acc[2][64];\n\n  for (int item = blockIdx.x; item < total; item += gridDim.x) {\n    const int tt = item / p.nfb, fb = item % p.nfb;\n    int e = 0;\n    while (e < 31 && tstart[e + 1] <= tt) ++e;\n\n    uint32_t A0[2][4], A1[2][4];\n    uint4 raw[2];\n    uint2 sfw[2];\n    auto load_words = [&](int st_idx) {\n      const uint8_t* st = smem + st_idx * STAGE_BYTES;\n#pragma unroll\n      for (int j = 0; j < 2; ++j) {\n        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);\n        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);\n      }\n    };\n    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n      wg_fence();\n      mma_block(acc[0], F[0], xs, b, scale_d);\n      mma_block(acc[1], F[1], xs, b, scale_d);\n      wg_commit();\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n    };\n    // prologue: first k-tile\'s words, block 0\'s fragments\n    mbar_wait(&full[s], phase);\n    load_words(s);\n    int cur_s = s;\n#pragma unroll\n    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n\n    for (int k = 0; k < KT; ++k) {\n      const uint8_t* xs = smem + cur_s * STAGE_BYTES;\n      // block 0 (A0)\n      issue(A0, xs, 0, k > 0);\n      if (k > 0 && MOE_WGMMA) {\n        wg_wait<1>();\n        keep(A1[0]); keep(A1[1]);\n        __syncwarp();\n        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done\n      }\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].y, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);\n      // block 1 (A1)\n      issue(A1, xs, 1, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);\n      // block 2 (A0)\n      issue(A0, xs, 2, 1);\n      wg_wait<1>();\n      keep(A1[0]); keep(A1[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].w, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);\n      // block 3 (A1)\n      issue(A1, xs, 3, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#if !MOE_WGMMA\n      // mma.sync has consumed this tile completely. Release it before waiting\n      // for the next tile, including the one-stage down-projection fallback.\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&empty[cur_s]);\n#endif\n      if (k + 1 < KT) {\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n        mbar_wait(&full[s], phase);\n        load_words(s);\n        cur_s = s;\n#pragma unroll\n        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n      }\n    }\n    wg_wait<0>();\n    fence_acc(acc[0]);\n    fence_acc(acc[1]);\n    keep(A1[0]); keep(A1[1]);\n    __syncwarp();\n    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);\n    if (++s == STAGES) { s = 0; phase ^= 1; }\n\n    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----\n    const float gsc = p.gs[e];\n    if (t == 0) bulk_wait_read();       // previous item\'s store has finished reading my_epi\n    named_bar(1 + cw, 128);\n    const int r0 = warp * 16 + (lane >> 2);\n    const int c2 = 2 * (lane & 3);\n    constexpr int NT = MODE == 1 ? 1 : 2;\n#pragma unroll\n    for (int j = 0; j < NT; ++j) {\n      uint8_t* tile = my_epi + j * X_BYTES;\n#pragma unroll\n      for (int i = 0; i < BN / 8; ++i) {\n#pragma unroll\n        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)\n          const int rho = r0 + 8 * h;\n#pragma unroll\n          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q\n            const int n = 8 * i + c2 + q;\n            float v;\n            if (MODE == 1) {\n              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));\n              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * gsc));\n              v = gg / (1.f + __expf(-gg)) * uu;\n            } else {\n              v = acc[j][4 * i + 2 * h + q] * gsc;\n            }\n            __nv_bfloat16 bv = __float2bfloat16_rn(v);\n            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;\n          }\n        }\n      }\n    }\n    fence_async_smem();\n    named_bar(1 + cw, 128);\n    if (t == 0) {\n#pragma unroll\n      for (int j = 0; j < NT; ++j) {\n        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;\n        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);\n      }\n    }\n  }\n  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");\n}\n\n}  // namespace raw11\n\nvoid raw11_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,\n              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {\n  raw11::Params p;\n  p.wblob = wblob.data_ptr<uint8_t>();\n  p.sfblob = sfblob.data_ptr<uint8_t>();\n  p.gs = gs.data_ptr<float>();\n  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());\n  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());\n  p.tile_start = tile_start.data_ptr<int>();\n  p.nfb = (int)nfb;\n  p.kt = (int)kt;\n  p.kt_out = (int)kt_out;\n  auto stream = at::cuda::getCurrentCUDAStream();\n  int dev = 0, major = 0;\n  cudaGetDevice(&dev);\n  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\n#define MOE_LAUNCH(M, S)                                                                                  \\\n  do {                                                                                                     \\\n    cudaFuncSetAttribute(raw11::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \\\n                         raw11::smem_bytes<M, S>());                                                         \\\n    raw11::moe_gemm_kernel<M, S><<<(int)grid, 384, raw11::smem_bytes<M, S>(), stream>>>(p);                   \\\n  } while (0)\n  if (major == 9) {\n    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);\n  } else {\n    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);\n  }\n#undef MOE_LAUNCH\n}\n\n// Grouped NVFP4 W4A16 expert GEMM for Hopper (wgmma RS), with an mma.sync emulation of the\n// same data path for other archs (local testing).\n//\n// out^T[feat, tok] = W_e[feat, :] . X[tok, :] for each work item (token tile tt of expert e,\n// weight block fb of 256 rows). Weights are the A operand (decoded from NVFP4 to BF16 in registers,\n// exact: e2m1 x e4m3 has <= 6 significant bits), tokens the B operand (bf16, K-major, 128B swizzle,\n// one 16 KB bulk copy per k-tile from a pre-swizzled token-tile buffer).\n//\n// Layouts (all built by the host / gather kernels):\n//   wblob  [E][NFB][KT][4][128 thr][16 B]   thread t\'s 16 bytes per m64 block per k-tile, in WGMMA\n//                                           A-fragment order (quack\'s nvfp4 repack, per m64 block)\n//   sfblob [E][NFB][KT][4][32 slot][4 blk][2] e4m3 scale of rows (r, r+8) per k16 block\n//   x      [tiles][KT][128 tok][128 B]      bf16 rows, 16-byte chunk c of row r stored at c ^ (r & 7)\n//   y      [tiles][KT_out][128 tok][128 B]  same layout (feeds the next GEMM / the combine)\n//   tile_start [33]                         token tiles of expert e are [tile_start[e], tile_start[e+1])\n//\n// MODE 1 (gate/up, SwiGLU): CTA rows = [gate f0..f0+63, up f0..f0+63, gate f0+64.., up f0+64..];\n//   consumer WG w owns m64 blocks (2w, 2w+1) = (gate, up) of features f0 + 64w ..; out = silu(gs g) * (gs u).\n// MODE 0 (down): WG w owns m64 blocks 2w, 2w+1; out = gs * acc.\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n#include <stdint.h>\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n\n#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define MOE_WGMMA 1\n#else\n#define MOE_WGMMA 0\n#endif\n\nnamespace raw15 {\n\nconstexpr int BN = 128;              // tokens per tile\nconstexpr int X_BYTES = BN * 128;    // 16 KB\nconstexpr int W_BYTES = 4 * 2048;    // 8 KB\nconstexpr int SF_BYTES = 4 * 256;    // 1 KB\nconstexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)\n// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]\ntemplate <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }\ntemplate <int MODE, int STAGES> constexpr int smem_bytes() {\n  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;\n}\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "WAIT_%=:\\n"\n      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n"\n      "@!p bra WAIT_%=;\\n"\n      "}\\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"\n               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\n}\n__device__ __forceinline__ void bulk_wait_read() {\n  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");\n}\n__device__ __forceinline__ void fence_async_smem() {\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n}\n__device__ __forceinline__ void named_bar(int id, int n) {\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\n}\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t r;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));\n  return r;\n}\n\n// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.\nconstexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;\nconstexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;\n\n// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).\n__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {\n  uint32_t xm = x & 0x77777777u;\n  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);\n  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);\n  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);\n  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);\n  uint32_t se = x << 4;\n  uint32_t s03 = prmt(se, x, 0x5140u);\n  uint32_t s47 = prmt(se, x, 0x7362u);\n  h03 |= s03 & 0x80808080u;\n  h47 |= s47 & 0x80808080u;\n  r[0] = prmt(l03, h03, 0x5140u);\n  r[1] = prmt(l03, h03, 0x7362u);\n  r[2] = prmt(l47, h47, 0x5140u);\n  r[3] = prmt(l47, h47, 0x7362u);\n}\n\n// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).\n__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {\n  uint32_t h2;\n  asm("{\\n.reg .b16 t;\\nmov.b16 t, %1;\\ncvt.rn.f16x2.e4m3x2 %0, t;\\n}" : "=r"(h2) : "h"((uint16_t)two_bytes));\n  __half2 h = *reinterpret_cast<__half2*>(&h2);\n  float2 f = __half22float2(h);\n  return __floats2bfloat162_rn(f.x, f.y);\n}\n\n__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {\n  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);\n  v = __hmul2(v, s);\n  return *reinterpret_cast<uint32_t*>(&v);\n}\n\n// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.\n__device__ __forceinline__ void make_frag(uint32_t x, __nv_bfloat162 s, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {\n  uint32_t r[4];\n  decode8(x, lo_a, hi_a, r);\n  __nv_bfloat162 slo = __low2bfloat162(s), shi = __high2bfloat162(s);\n  a[0] = bmul(r[0], slo);\n  a[1] = bmul(r[1], shi);\n  a[2] = bmul(r[2], slo);\n  a[3] = bmul(r[3], shi);\n}\n\n__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {\n  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).\n  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\n}\n\n#if MOE_WGMMA\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\ntemplate <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }\n#else\n__device__ __forceinline__ void wg_fence() {}\n__device__ __forceinline__ void wg_commit() {}\ntemplate <int N> __device__ __forceinline__ void wg_wait() {}\n#endif\n\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\n  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));\n}\n__device__ __forceinline__ void fence_acc(float (&acc)[64]) {\n#pragma unroll\n  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");\n}\n\n// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)\n__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {\n#if MOE_WGMMA\n  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %69, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "\n      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "\n      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\n      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\n#else\n  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.\n  const int lane = threadIdx.x & 31;\n  const int g = lane >> 2, c = lane & 3;\n  if (!scale_d) {\n#pragma unroll\n    for (int i = 0; i < 64; ++i) d[i] = 0.f;\n  }\n#pragma unroll\n  for (int i = 0; i < BN / 8; ++i) {\n    const int n = 8 * i + g;\n    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row\n    const uint8_t* row = xs + n * 128;\n    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));\n    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));\n    asm volatile(\n        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "\n        "{%0, %1, %2, %3};"\n        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])\n        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n  }\n#endif\n}\n\n// A and B are both row/K-major 128B-swizzled shared matrices.\n__device__ __forceinline__ void mma_ss(float (&d)[64], const uint8_t* as, const uint8_t* xs, int b, int scale_d) {\n#if MOE_WGMMA\n  uint64_t da = make_desc_b(smem_u32(as)) + (uint64_t)((b * 32) >> 4);\n  uint64_t db = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %66, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "\n      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "\n      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\n      "%64, %65, p, 1, 1, 0, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "l"(da), "l"(db), "r"(scale_d));\n#else\n  const int t = threadIdx.x % 128;\n  const int r = (t >> 5) * 16 + ((t & 31) >> 2);\n  const int kb = b * 32 + 4 * (t & 3);\n  uint32_t a[4];\n#pragma unroll\n  for (int j = 0; j < 4; ++j) {\n    int rr = r + 8 * (j & 1), kk = kb + 16 * (j >> 1);\n    a[j] = *reinterpret_cast<const uint32_t*>(as + rr * 128 + (((kk >> 4) ^ (rr & 7)) << 4) + (kk & 15));\n  }\n  mma_block(d, a, xs, b, scale_d);\n#endif\n}\n\nstruct Params {\n  const uint8_t* wblob;\n  const uint8_t* sfblob;\n  const float* gs;\n  const uint8_t* x;\n  uint8_t* y;\n  const int* tile_start;\n  int nfb, kt, kt_out;\n};\n\ntemplate <int MODE, int STAGES>\n__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {\n  constexpr int EPI_BYTES = epi_bytes<MODE>();\n  extern __shared__ uint8_t smem_raw[];\n  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));\n  constexpr int DEQ_BYTES = MOE_WGMMA ? 65536 : (MODE == 1 ? 32768 : 0);\n  uint8_t* deq = smem + STAGES * STAGE_BYTES;\n  uint8_t* epi = deq + DEQ_BYTES;\n  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);\n  uint64_t* empty = full + STAGES;\n  int* tstart = reinterpret_cast<int*>(empty + STAGES);\n\n  const int tid = threadIdx.x;\n  const int wg = tid / 128;\n  if (tid < 33) tstart[tid] = p.tile_start[tid];\n  if (tid == 0) {\n    for (int s = 0; s < STAGES; ++s) {\n      mbar_init(&full[s], 1);\n      mbar_init(&empty[s], 8);\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  __syncthreads();\n  const int num_tiles = tstart[32];\n  const int total = num_tiles * p.nfb;\n  const int KT = p.kt;\n\n  if (wg == 0) {\n#if MOE_WGMMA\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\n#endif\n    if (tid != 0) return;\n    int s = 0;\n    uint32_t phase = 0;\n    for (int item = blockIdx.x; item < total; item += gridDim.x) {\n      const int tt = item / p.nfb, fb = item % p.nfb;\n      int e = 0;\n      while (e < 31 && tstart[e + 1] <= tt) ++e;\n      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;\n      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;\n      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;\n      for (int k = 0; k < KT; ++k) {\n        mbar_wait(&empty[s], phase ^ 1);\n        uint8_t* st = smem + s * STAGE_BYTES;\n        mbar_expect_tx(&full[s], STAGE_BYTES);\n        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n      }\n    }\n    return;\n  }\n\n#if MOE_WGMMA\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");\n#endif\n  const int cw = wg - 1;                 // consumer warpgroup 0/1\n  const int t = tid - 128 * wg;          // thread within the warpgroup\n  const int warp = t >> 5, lane = t & 31;\n  const int slot = warp * 8 + (lane >> 2);\n  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;\n  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);\n  int s = 0;\n  uint32_t phase = 0;\n  float acc[2][64];\n\n  for (int item = blockIdx.x; item < total; item += gridDim.x) {\n    const int tt = item / p.nfb, fb = item % p.nfb;\n    int e = 0;\n    while (e < 31 && tstart[e + 1] <= tt) ++e;\n\n    uint32_t A0[2][4], A1[2][4];\n    uint4 raw[2];\n    uint2 sfw[2];\n    auto load_words = [&](int st_idx) {\n      const uint8_t* st = smem + st_idx * STAGE_BYTES;\n#pragma unroll\n      for (int j = 0; j < 2; ++j) {\n        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);\n        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);\n      }\n    };\n    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n      wg_fence();\n      mma_block(acc[0], F[0], xs, b, scale_d);\n      mma_block(acc[1], F[1], xs, b, scale_d);\n      wg_commit();\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n    };\n    if constexpr (MOE_WGMMA || MODE == 1) {\n      // Shared A fragments stay alive until the committed group has completed.\n      // Hopper has two A slots, so dequantization of the next tile overlaps MMA.\n      for (int k = 0; k < KT; ++k) {\n        mbar_wait(&full[s], phase);\n        load_words(s);\n        uint8_t* abase = deq + (MOE_WGMMA ? (k & 1) * 32768 : 0) + cw * 16384;\n        const int r = warp * 16 + (lane >> 2);\n#pragma unroll\n        for (int b = 0; b < 4; ++b) {\n#pragma unroll\n          for (int j = 0; j < 2; ++j) {\n            uint32_t words[4] = {raw[j].x, raw[j].y, raw[j].z, raw[j].w};\n            uint32_t scales[2] = {sfw[j].x, sfw[j].y};\n            uint32_t a[4];\n            make_frag(words[b], e4m3x2_to_bf16x2(scales[b >> 1] >> (16 * (b & 1))), lo_a, hi_a, a);\n#pragma unroll\n            for (int q = 0; q < 4; ++q) {\n              const int rr = r + 8 * (q & 1), kk = b * 32 + 4 * (lane & 3) + 16 * (q >> 1);\n              *reinterpret_cast<uint32_t*>(abase + j * 8192 + rr * 128 + (((kk >> 4) ^ (rr & 7)) << 4) + (kk & 15)) = a[q];\n            }\n          }\n        }\n        fence_async_smem();\n        named_bar(1 + cw, 128);\n        fence_acc(acc[0]); fence_acc(acc[1]); wg_fence();\n#pragma unroll\n        for (int b = 0; b < 4; ++b) {\n          mma_ss(acc[0], abase, smem + s * STAGE_BYTES, b, k > 0 || b > 0);\n          mma_ss(acc[1], abase + 8192, smem + s * STAGE_BYTES, b, k > 0 || b > 0);\n        }\n        wg_commit();\n        wg_wait<1>();\n        if (MOE_WGMMA && k > 0) {\n          __syncwarp();\n          if (lane == 0) mbar_arrive(&empty[(s + STAGES - 1) % STAGES]);\n        }\n        if (!MOE_WGMMA) {\n          named_bar(1 + cw, 128);\n          if (lane == 0) mbar_arrive(&empty[s]);\n        }\n        if (k + 1 < KT) {\n          if (++s == STAGES) { s = 0; phase ^= 1; }\n        }\n      }\n      wg_wait<0>();\n      fence_acc(acc[0]); fence_acc(acc[1]);\n      __syncwarp();\n      if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[s]);\n      if (++s == STAGES) { s = 0; phase ^= 1; }\n    } else {\n    // prologue: first k-tile\'s words, block 0\'s fragments\n    mbar_wait(&full[s], phase);\n    load_words(s);\n    int cur_s = s;\n#pragma unroll\n    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n\n    for (int k = 0; k < KT; ++k) {\n      const uint8_t* xs = smem + cur_s * STAGE_BYTES;\n      // block 0 (A0)\n      issue(A0, xs, 0, k > 0);\n      if (k > 0 && MOE_WGMMA) {\n        wg_wait<1>();\n        keep(A1[0]); keep(A1[1]);\n        __syncwarp();\n        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done\n      }\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].y, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);\n      // block 1 (A1)\n      issue(A1, xs, 1, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);\n      // block 2 (A0)\n      issue(A0, xs, 2, 1);\n      wg_wait<1>();\n      keep(A1[0]); keep(A1[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].w, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);\n      // block 3 (A1)\n      issue(A1, xs, 3, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#if !MOE_WGMMA\n      // mma.sync has consumed this tile completely. Release it before waiting\n      // for the next tile, including the one-stage down-projection fallback.\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&empty[cur_s]);\n#endif\n      if (k + 1 < KT) {\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n        mbar_wait(&full[s], phase);\n        load_words(s);\n        cur_s = s;\n#pragma unroll\n        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n      }\n    }\n    wg_wait<0>();\n    fence_acc(acc[0]);\n    fence_acc(acc[1]);\n    keep(A1[0]); keep(A1[1]);\n    __syncwarp();\n    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);\n    if (++s == STAGES) { s = 0; phase ^= 1; }\n\n    }\n\n    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----\n    const float gsc = p.gs[e];\n    if (t == 0) bulk_wait_read();       // previous item\'s store has finished reading my_epi\n    named_bar(1 + cw, 128);\n    const int r0 = warp * 16 + (lane >> 2);\n    const int c2 = 2 * (lane & 3);\n    constexpr int NT = MODE == 1 ? 1 : 2;\n#pragma unroll\n    for (int j = 0; j < NT; ++j) {\n      uint8_t* tile = my_epi + j * X_BYTES;\n#pragma unroll\n      for (int i = 0; i < BN / 8; ++i) {\n#pragma unroll\n        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)\n          const int rho = r0 + 8 * h;\n#pragma unroll\n          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q\n            const int n = 8 * i + c2 + q;\n            float v;\n            if (MODE == 1) {\n              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));\n              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * gsc));\n              v = gg / (1.f + __expf(-gg)) * uu;\n            } else {\n              v = acc[j][4 * i + 2 * h + q] * gsc;\n            }\n            __nv_bfloat16 bv = __float2bfloat16_rn(v);\n            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;\n          }\n        }\n      }\n    }\n    fence_async_smem();\n    named_bar(1 + cw, 128);\n    if (t == 0) {\n#pragma unroll\n      for (int j = 0; j < NT; ++j) {\n        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;\n        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);\n      }\n    }\n  }\n  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");\n}\n\n}  // namespace raw15\n\nvoid raw15_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,\n              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {\n  raw15::Params p;\n  p.wblob = wblob.data_ptr<uint8_t>();\n  p.sfblob = sfblob.data_ptr<uint8_t>();\n  p.gs = gs.data_ptr<float>();\n  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());\n  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());\n  p.tile_start = tile_start.data_ptr<int>();\n  p.nfb = (int)nfb;\n  p.kt = (int)kt;\n  p.kt_out = (int)kt_out;\n  auto stream = at::cuda::getCurrentCUDAStream();\n  int dev = 0, major = 0;\n  cudaGetDevice(&dev);\n  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\n#define MOE_LAUNCH(M, S)                                                                                  \\\n  do {                                                                                                     \\\n    cudaFuncSetAttribute(raw15::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \\\n                         (raw15::smem_bytes<M, S>() + (major == 9 ? 65536 : (M == 1 ? 32768 : 0))));                                                         \\\n    raw15::moe_gemm_kernel<M, S><<<(int)grid, 384, (raw15::smem_bytes<M, S>() + (major == 9 ? 65536 : (M == 1 ? 32768 : 0))), stream>>>(p);                   \\\n  } while (0)\n  if (major == 9) {\n    if (mode == 1) MOE_LAUNCH(1, 4); else MOE_LAUNCH(0, 3);\n  } else {\n    if (mode == 1) MOE_LAUNCH(1, 1); else MOE_LAUNCH(0, 1);\n  }\n#undef MOE_LAUNCH\n}\n\n// Grouped NVFP4 W4A16 expert GEMM for Hopper (wgmma RS), with an mma.sync emulation of the\n// same data path for other archs (local testing).\n//\n// out^T[feat, tok] = W_e[feat, :] . X[tok, :] for each work item (token tile tt of expert e,\n// weight block fb of 256 rows). Weights are the A operand (decoded from NVFP4 to BF16 in registers,\n// exact: e2m1 x e4m3 has <= 6 significant bits), tokens the B operand (bf16, K-major, 128B swizzle,\n// one 16 KB bulk copy per k-tile from a pre-swizzled token-tile buffer).\n//\n// Layouts (all built by the host / gather kernels):\n//   wblob  [E][NFB][KT][4][128 thr][16 B]   thread t\'s 16 bytes per m64 block per k-tile, in WGMMA\n//                                           A-fragment order (quack\'s nvfp4 repack, per m64 block)\n//   sfblob [E][NFB][KT][4][32 slot][4 blk][2] e4m3 scale of rows (r, r+8) per k16 block\n//   x      [tiles][KT][128 tok][128 B]      bf16 rows, 16-byte chunk c of row r stored at c ^ (r & 7)\n//   y      [tiles][KT_out][128 tok][128 B]  same layout (feeds the next GEMM / the combine)\n//   tile_start [33]                         token tiles of expert e are [tile_start[e], tile_start[e+1])\n//\n// MODE 1 (gate/up, SwiGLU): CTA rows = [gate f0..f0+63, up f0..f0+63, gate f0+64.., up f0+64..];\n//   consumer WG w owns m64 blocks (2w, 2w+1) = (gate, up) of features f0 + 64w ..; out = silu(gs g) * (gs u).\n// MODE 0 (down): WG w owns m64 blocks 2w, 2w+1; out = gs * acc.\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n#include <stdint.h>\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n\n#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define MOE_WGMMA 1\n#else\n#define MOE_WGMMA 0\n#endif\n\nnamespace raw17 {\n\nconstexpr int BN = 128;              // tokens per tile\nconstexpr int X_BYTES = BN * 128;    // 16 KB\nconstexpr int W_BYTES = 4 * 2048;    // 8 KB\nconstexpr int SF_BYTES = 4 * 256;    // 1 KB\nconstexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)\n// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]\ntemplate <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }\ntemplate <int MODE, int STAGES> constexpr int smem_bytes() {\n  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;\n}\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "WAIT_%=:\\n"\n      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n"\n      "@!p bra WAIT_%=;\\n"\n      "}\\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"\n               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\n}\n__device__ __forceinline__ void bulk_wait_read() {\n  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");\n}\n__device__ __forceinline__ void fence_async_smem() {\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n}\n__device__ __forceinline__ void named_bar(int id, int n) {\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\n}\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t r;\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));\n  return r;\n}\n\n// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.\nconstexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;\nconstexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;\n\n// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).\n__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {\n  uint32_t xm = x & 0x77777777u;\n  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);\n  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);\n  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);\n  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);\n  uint32_t se = x << 4;\n  uint32_t s03 = prmt(se, x, 0x5140u);\n  uint32_t s47 = prmt(se, x, 0x7362u);\n  h03 |= s03 & 0x80808080u;\n  h47 |= s47 & 0x80808080u;\n  r[0] = prmt(l03, h03, 0x5140u);\n  r[1] = prmt(l03, h03, 0x7362u);\n  r[2] = prmt(l47, h47, 0x5140u);\n  r[3] = prmt(l47, h47, 0x7362u);\n}\n\n// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).\n// Stored signed E4M3 bytes remain unchanged in the weight bank.\n// Reconstruct |sf| * 2^118 exactly in BF16. Zero has its own code.\n__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {\n  uint32_t lo = two_bytes & 255, hi = (two_bytes >> 8) & 255;\n  uint32_t bits = (muse_raw_scale118(lo)&65535u) | (muse_raw_scale118(hi)<<16);\n  return *reinterpret_cast<__nv_bfloat162*>(&bits);\n}\n\n__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {\n  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);\n  v = __hmul2(v, s);\n  return *reinterpret_cast<uint32_t*>(&v);\n}\n\n// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.\n// Sixteen re-encoded E2M1 codes in A,B -> eight BF16x2 words.\n// The bit planes represent FP4 * 2^-126; multiplication by the encoded scale\n// yields the exact NVFP4 weight / 256. The epilogue restores the power of two.\n__device__ __forceinline__ void make_frag(uint32_t A, uint32_t B, int half,\n      __nv_bfloat162 sf, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {\n  constexpr uint32_t MASK = 0x81C081C0u, LA = 0x60306030u;\n  uint32_t r[4];\n  if (half == 0) {\n    r[0] = A & MASK;\n    r[1] = __funnelshift_r(A,A,4) & MASK;\n    r[2] = __funnelshift_r(A,A,10) & MASK;\n    r[3] = __funnelshift_r(B,B,2) & MASK;\n  } else {\n    r[0] = __funnelshift_r(B,B,6) & MASK;\n    r[1] = __funnelshift_r(B,B,12) & MASK;\n    uint32_t W = (A & LA) | (B & ~LA);\n    r[2] = __funnelshift_r(W,W,24) & MASK;\n    r[3] = __funnelshift_r(W,W,30) & MASK;\n  }\n  __nv_bfloat162 slo = __low2bfloat162(sf), shi = __high2bfloat162(sf);\n  a[0] = bmul(r[0], slo); a[1] = bmul(r[1], shi);\n  a[2] = bmul(r[2], slo); a[3] = bmul(r[3], shi);\n}\n\n__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {\n  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).\n  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\n}\n\n#if MOE_WGMMA\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\ntemplate <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }\n#else\n__device__ __forceinline__ void wg_fence() {}\n__device__ __forceinline__ void wg_commit() {}\ntemplate <int N> __device__ __forceinline__ void wg_wait() {}\n#endif\n\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\n  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));\n}\n__device__ __forceinline__ void fence_acc(float (&acc)[64]) {\n#pragma unroll\n  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");\n}\n\n// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)\n__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {\n#if MOE_WGMMA\n  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\n  asm volatile(\n      "{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %69, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "\n      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "\n      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\n      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\n#else\n  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.\n  const int lane = threadIdx.x & 31;\n  const int g = lane >> 2, c = lane & 3;\n  if (!scale_d) {\n#pragma unroll\n    for (int i = 0; i < 64; ++i) d[i] = 0.f;\n  }\n#pragma unroll\n  for (int i = 0; i < BN / 8; ++i) {\n    const int n = 8 * i + g;\n    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row\n    const uint8_t* row = xs + n * 128;\n    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));\n    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));\n    asm volatile(\n        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "\n        "{%0, %1, %2, %3};"\n        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])\n        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n  }\n#endif\n}\n\nstruct Params {\n  const uint8_t* wblob;\n  const uint8_t* sfblob;\n  const float* gs;\n  const uint8_t* x;\n  uint8_t* y;\n  const int* tile_start;\n  int nfb, kt, kt_out;\n};\n\ntemplate <int MODE, int STAGES>\n__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {\n  constexpr int EPI_BYTES = epi_bytes<MODE>();\n  extern __shared__ uint8_t smem_raw[];\n  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));\n  uint8_t* epi = smem + STAGES * STAGE_BYTES;\n  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);\n  uint64_t* empty = full + STAGES;\n  int* tstart = reinterpret_cast<int*>(empty + STAGES);\n\n  const int tid = threadIdx.x;\n  const int wg = tid / 128;\n  if (tid < 33) tstart[tid] = p.tile_start[tid];\n  if (tid == 0) {\n    for (int s = 0; s < STAGES; ++s) {\n      mbar_init(&full[s], 1);\n      mbar_init(&empty[s], 8);\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n  }\n  __syncthreads();\n  const int num_tiles = tstart[32];\n  const int total = num_tiles * p.nfb;\n  const int KT = p.kt;\n\n  if (wg == 0) {\n#if MOE_WGMMA\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\n#endif\n    if (tid != 0) return;\n    int s = 0;\n    uint32_t phase = 0;\n    for (int item = blockIdx.x; item < total; item += gridDim.x) {\n      const int tt = item / p.nfb, fb = item % p.nfb;\n      int e = 0;\n      while (e < 31 && tstart[e + 1] <= tt) ++e;\n      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;\n      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;\n      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;\n      for (int k = 0; k < KT; ++k) {\n        mbar_wait(&empty[s], phase ^ 1);\n        uint8_t* st = smem + s * STAGE_BYTES;\n        mbar_expect_tx(&full[s], STAGE_BYTES);\n        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n      }\n    }\n    return;\n  }\n\n#if MOE_WGMMA\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");\n#endif\n  const int cw = wg - 1;                 // consumer warpgroup 0/1\n  const int t = tid - 128 * wg;          // thread within the warpgroup\n  const int warp = t >> 5, lane = t & 31;\n  const int slot = warp * 8 + (lane >> 2);\n  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;\n  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);\n  int s = 0;\n  uint32_t phase = 0;\n  float acc[2][64];\n\n  for (int item = blockIdx.x; item < total; item += gridDim.x) {\n    const int tt = item / p.nfb, fb = item % p.nfb;\n    int e = 0;\n    while (e < 31 && tstart[e + 1] <= tt) ++e;\n\n    uint32_t A0[2][4], A1[2][4];\n    uint4 raw[2];\n    uint2 sfw[2];\n    auto load_words = [&](int st_idx) {\n      const uint8_t* st = smem + st_idx * STAGE_BYTES;\n#pragma unroll\n      for (int j = 0; j < 2; ++j) {\n        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);\n        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);\n      }\n    };\n    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n      wg_fence();\n      mma_block(acc[0], F[0], xs, b, scale_d);\n      mma_block(acc[1], F[1], xs, b, scale_d);\n      wg_commit();\n      fence_acc(acc[0]);\n      fence_acc(acc[1]);\n    };\n    // prologue: first k-tile\'s words, block 0\'s fragments\n    mbar_wait(&full[s], phase);\n    load_words(s);\n    int cur_s = s;\n#pragma unroll\n    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n\n    for (int k = 0; k < KT; ++k) {\n      const uint8_t* xs = smem + cur_s * STAGE_BYTES;\n      // block 0 (A0)\n      issue(A0, xs, 0, k > 0);\n      if (k > 0 && MOE_WGMMA) {\n        wg_wait<1>();\n        keep(A1[0]); keep(A1[1]);\n        __syncwarp();\n        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done\n      }\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 1, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);\n      // block 1 (A1)\n      issue(A1, xs, 1, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 0, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);\n      // block 2 (A0)\n      issue(A0, xs, 2, 1);\n      wg_wait<1>();\n      keep(A1[0]); keep(A1[1]);\n#pragma unroll\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 1, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);\n      // block 3 (A1)\n      issue(A1, xs, 3, 1);\n      wg_wait<1>();\n      keep(A0[0]); keep(A0[1]);\n#if !MOE_WGMMA\n      // mma.sync has consumed this tile completely. Release it before waiting\n      // for the next tile, including the one-stage down-projection fallback.\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&empty[cur_s]);\n#endif\n      if (k + 1 < KT) {\n        if (++s == STAGES) { s = 0; phase ^= 1; }\n        mbar_wait(&full[s], phase);\n        load_words(s);\n        cur_s = s;\n#pragma unroll\n        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\n      }\n    }\n    wg_wait<0>();\n    fence_acc(acc[0]);\n    fence_acc(acc[1]);\n    keep(A1[0]); keep(A1[1]);\n    __syncwarp();\n    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);\n    if (++s == STAGES) { s = 0; phase ^= 1; }\n\n    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----\n    const float gsc = p.gs[e] * 256.f;\n    if (t == 0) bulk_wait_read();       // previous item\'s store has finished reading my_epi\n    named_bar(1 + cw, 128);\n    const int r0 = warp * 16 + (lane >> 2);\n    const int c2 = 2 * (lane & 3);\n    constexpr int NT = MODE == 1 ? 1 : 2;\n#pragma unroll\n    for (int j = 0; j < NT; ++j) {\n      uint8_t* tile = my_epi + j * X_BYTES;\n#pragma unroll\n      for (int i = 0; i < BN / 8; ++i) {\n#pragma unroll\n        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)\n          const int rho = r0 + 8 * h;\n#pragma unroll\n          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q\n            const int n = 8 * i + c2 + q;\n            float v;\n            if (MODE == 1) {\n              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));\n              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * gsc));\n              v = gg / (1.f + __expf(-gg)) * uu;\n            } else {\n              v = acc[j][4 * i + 2 * h + q] * gsc;\n            }\n            __nv_bfloat16 bv = __float2bfloat16_rn(v);\n            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;\n          }\n        }\n      }\n    }\n    fence_async_smem();\n    named_bar(1 + cw, 128);\n    if (t == 0) {\n#pragma unroll\n      for (int j = 0; j < NT; ++j) {\n        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;\n        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);\n      }\n    }\n  }\n  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");\n}\n\n}  // namespace raw17\n\nvoid raw17_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,\n              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {\n  raw17::Params p;\n  p.wblob = wblob.data_ptr<uint8_t>();\n  p.sfblob = sfblob.data_ptr<uint8_t>();\n  p.gs = gs.data_ptr<float>();\n  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());\n  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());\n  p.tile_start = tile_start.data_ptr<int>();\n  p.nfb = (int)nfb;\n  p.kt = (int)kt;\n  p.kt_out = (int)kt_out;\n  auto stream = at::cuda::getCurrentCUDAStream();\n  int dev = 0, major = 0;\n  cudaGetDevice(&dev);\n  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\n#define MOE_LAUNCH(M, S)                                                                                  \\\n  do {                                                                                                     \\\n    cudaFuncSetAttribute(raw17::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \\\n                         raw17::smem_bytes<M, S>());                                                         \\\n    raw17::moe_gemm_kernel<M, S><<<(int)grid, 384, raw17::smem_bytes<M, S>(), stream>>>(p);                   \\\n  } while (0)\n  if (major == 9) {\n    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);\n  } else {\n    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);\n  }\n#undef MOE_LAUNCH\n}\n\n\n#include <cublasLt.h>\n#include <cstring>\nnamespace muse_lt {\nstatic cublasLtHandle_t handle() {\n  static cublasLtHandle_t h=[] { cublasLtHandle_t p; TORCH_CHECK(cublasLtCreate(&p)==CUBLAS_STATUS_SUCCESS,"cublasLtCreate"); return p; }();\n  return h;\n}\nstruct Layout {\n  cublasLtMatmulDesc_t op;\n  cublasLtMatrixLayout_t a,b,c;\n  Layout(const torch::Tensor& w,const torch::Tensor& x,const torch::Tensor& y) {\n    int64_t n=y.size(1),m=x.size(0),k=x.size(1);\n    TORCH_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"Lt desc");\n    cublasOperation_t ta=CUBLAS_OP_T;\n    TORCH_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))==CUBLAS_STATUS_SUCCESS,"Lt transpose");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,w.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt A layout");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,x.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt B layout");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&c,(y.scalar_type()==at::kFloat ? CUDA_R_32F : CUDA_R_16BF),n,m,n)==CUBLAS_STATUS_SUCCESS,"Lt C layout");\n  }\n  ~Layout() { cublasLtMatmulDescDestroy(op); cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c); }\n};\nconstexpr int CAP=32;\nconstexpr size_t WORK_BYTES=256*1024*1024;\n}\ntorch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out) {\n  using namespace muse_lt;\n  Layout layout(W,X,Out);\n  cublasLtMatmulPreference_t pref;\n  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS,"Lt preference");\n  size_t workspace=WORK_BYTES;\n  TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace))==CUBLAS_STATUS_SUCCESS,"Lt workspace preference");\n  cublasLtMatmulHeuristicResult_t results[CAP]; int count=0;\n  auto status=cublasLtMatmulAlgoGetHeuristic(handle(),layout.op,layout.a,layout.b,layout.c,layout.c,pref,CAP,results,&count);\n  cublasLtMatmulPreferenceDestroy(pref);\n  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt heuristics: ",int(status));\n  auto codes=torch::empty({count,(int64_t)sizeof(cublasLtMatmulAlgo_t)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU));\n  for(int i=0;i<count;++i) {\n    TORCH_CHECK(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=WORK_BYTES,"Lt invalid heuristic");\n    std::memcpy(codes.data_ptr<uint8_t>()+i*sizeof(cublasLtMatmulAlgo_t),&results[i].algo,sizeof(cublasLtMatmulAlgo_t));\n  }\n  return codes;\n}\nvoid lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index) {\n  using namespace muse_lt;\n  TORCH_CHECK(index>=0 && index<Algos.size(0) && Work.nbytes()>=WORK_BYTES,"Lt invalid algorithm/workspace");\n  Layout layout(W,X,Out);\n  cublasLtMatmulAlgo_t algo;\n  std::memcpy(&algo,Algos.data_ptr<uint8_t>()+index*sizeof(algo),sizeof(algo));\n  const float alpha=1.f,beta=0.f;\n  auto status=cublasLtMatmul(handle(),layout.op,&alpha,W.data_ptr(),layout.a,X.data_ptr(),layout.b,&beta,\n                            Out.data_ptr(),layout.c,Out.data_ptr(),layout.c,&algo,Work.data_ptr(),WORK_BYTES,\n                            at::cuda::getCurrentCUDAStream());\n  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt matmul: ",int(status));\n}\n'
_pf_DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
_pf_LA_MASK = 1613783088

def _pf_bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if _pf_LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(_pf_DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
_pf_TABLE = _pf_bit_table()


def _pf_repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.contiguous().view(torch.uint8)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)
    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)
    for k, h, b, w, p in _pf_TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return torch.cat((wbytes, sbytes), 1).contiguous()

def _pf_repack_v6(wq, sf, gated):
    E, R, _ = wq.shape
    dev = wq.device
    if gated:
        t = torch.arange(R // 16, device=dev)[:, None]
        i = torch.arange(8, device=dev)[None, :]
        rows = torch.cat((t * 8 + i, R // 2 + t * 8 + i), 1)
    else:
        rows = torch.arange(R, device=dev).view(-1, 16)
    return torch.stack([_pf_repack(wq[e], sf[e], rows) for e in range(E)])

@triton.jit
def _pf_bf16x2_mul(a, b):
    return tl.inline_asm_elementwise('mul.rn.bf16x2 $0, $1, $2;', '=r,r,r', [a, b], dtype=tl.uint32, is_pure=True, pack=1)

@triton.jit
def _pf_canonical_expand_coalesced(WB, WW, OUT, COUNTS, ROWS: tl.constexpr, GATED: tl.constexpr, KB: tl.constexpr):
    e = tl.program_id(0)
    if tl.load(COUNTS + e) == 0:
        return
    tile = tl.program_id(1)
    i = tl.arange(0, 16 * (KB // 2))
    row16 = i // (KB // 2)
    kp = tl.program_id(2) * (KB // 2) + i % (KB // 2)
    ri = row16 // 8
    rr = row16 % 8
    kg = kp // 32
    q = kp // 8 % 4
    p = kp // 4 % 2
    lane = rr * 4 + q
    base = (e.to(tl.int64) * (ROWS // 16) + tile) * 36864
    srcword = base // 4 + kg * 128 + lane * 4 + p * 2
    A = tl.load(WW + srcword).to(tl.uint32)
    B = tl.load(WW + srcword + 1).to(tl.uint32)
    W = A & 1613783088 | B & 2681184207
    slot = kp % 4 * 2 + ri
    source = tl.where(slot < 3, A, tl.where(slot < 6, B, W))
    shifts = tl.where(ri == 0, 792896, 995396).to(tl.uint32)
    rot = shifts >> kp % 4 * 5 & 31
    code = tl.inline_asm_elementwise('shf.r.wrap.b32 $0, $1, $1, $2;', '=r,r,r', [source, rot], dtype=tl.uint32, is_pure=True, pack=1)
    sf = tl.load(WB + base + 32768 + kg // 8 * 512 + lane * 16 + kg % 8 * 2 + ri).to(tl.uint32)
    a = sf & 127
    b = tl.where(a >= 8, 0x7700 + (a << 4), tl.where(a >= 4, 0x7680 + (a << 5),
        tl.where(a >= 2, 0x7600 + (a << 6), tl.where(a != 0, 0x7600, 0))))
    b = b | ((sf & 128) << 8)
    scale = b | (b << 16)
    weight = _pf_bf16x2_mul(_pf_bf16x2_mul(code & 2176876992, scale), 1132479360)
    row = tile * 8 + rr + ri * (ROWS // 2) if GATED else tile * 16 + row16
    tl.store(OUT + (e.to(tl.int64) * ROWS + row) * 2048 + kp, weight)
_pf_BN = 128
_pf_D = 4096
_pf_mod = None

def _pf_grouped_module():
    global _pf_mod
    if _pf_mod is None:
        import os, hashlib
        from torch.utils.cpp_extension import load_inline
        cap = torch.cuda.get_device_capability()
        previous = os.environ.get('TORCH_CUDA_ARCH_LIST')
        os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' if cap == (9, 0) else f'{cap[0]}.{cap[1]}'
        declaration = 'void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);\n'
        declaration += 'void moe_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);\n'
        for v in (11, 15, 17):
            declaration += f'void raw{v}_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);\n'
        declaration += 'torch::Tensor lt_options(torch::Tensor,torch::Tensor,torch::Tensor);\nvoid lt_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);'
        try:
            _pf_mod = load_inline(name='prefill_score_' + hashlib.sha256(_pf_GROUPED_SRC.encode()).hexdigest()[:12], cpp_sources=declaration, cuda_sources=_pf_GROUPED_SRC, functions=['gemm', 'moe_gemm', 'raw11_gemm', 'raw15_gemm', 'raw17_gemm', 'lt_options', 'lt_run'], extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '-lineinfo', '--ptxas-options=-v'], extra_ldflags=['-lcublasLt'], verbose=False)
        finally:
            if previous is None:
                os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else:
                os.environ['TORCH_CUDA_ARCH_LIST'] = previous
    return _pf_mod

@triton.jit
def _pf_portfolio_swiglu(H, A, STARTS, COUNTS, GATE_GS, UP_GS, D: tl.constexpr, BR: tl.constexpr):
    e = tl.program_id(0)
    n = tl.load(COUNTS + e)
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    if tl.program_id(1) * BR >= n:
        return
    row = tl.load(STARTS + e) + r
    c = tl.arange(0, D)
    g = tl.load(H + row[:, None].to(tl.int64) * (2 * D) + c[None, :], r[:, None] < n, 0).to(tl.float32) * tl.load(GATE_GS + e)
    u = tl.load(H + row[:, None].to(tl.int64) * (2 * D) + D + c[None, :], r[:, None] < n, 0).to(tl.float32) * tl.load(UP_GS + e)
    g = g.to(tl.bfloat16).to(tl.float32)
    u = u.to(tl.bfloat16).to(tl.float32)
    tl.store(A + row[:, None].to(tl.int64) * D + c[None, :], (g * tl.sigmoid(g) * u).to(tl.bfloat16), r[:, None] < n)

def _pf_run_canonical_dg(pooled, probs, slot_e, slot_pos, counts, st, name):
    import deep_gemm
    T, D = pooled.shape
    dev = pooled.device
    cap = T * 8 + 32 * 128
    starts = torch.empty(32, dtype=torch.int32, device=dev)
    psum = torch.empty_like(starts)
    _pf_offsets_kernel[1,](counts, starts, psum, ALIGN=128)
    a = torch.empty((cap, D), dtype=torch.bfloat16, device=dev)
    row_of = torch.empty((T, 8), dtype=torch.int32, device=dev)
    _pf_gather_norm_kernel[T,](pooled, st['pre_expert'], slot_e, slot_pos, starts, a, row_of, D=D, num_warps=8)
    weight_scratch = torch.empty((32, 8192, 4096), dtype=torch.bfloat16, device=dev)
    w2 = weight_scratch.view(-1)[:32 * 4096 * 4096].view(32, 4096, 4096)
    h = torch.empty((cap, 2 * D), dtype=torch.bfloat16, device=dev)
    y = h.view(-1)[:cap * D].view(cap, D)
    kb = 2048 if 'k2048' in name else 1024
    _pf_canonical_expand_coalesced[32, 512, 4096 // kb](st['w13p'], st['w13p'].view(torch.int32), weight_scratch.view(torch.int32), counts, ROWS=8192, GATED=True, KB=kb, num_warps=8)
    hint = {} if not name.startswith('dg_hint') else {'expected_m_for_psum_layout': max(1, triton.cdiv(2 * T, 32))}
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, weight_scratch, h, psum, use_psum_layout=True, **hint)
    _pf_portfolio_swiglu[32, triton.cdiv(T, 4)](h, a, starts, counts, st['gs13_gate'], st['gs13_up'], D=D, BR=4, num_warps=8 if '_w8' in name else 4)
    _pf_canonical_expand_coalesced[32, 256, 4096 // kb](st['w2p'], st['w2p'].view(torch.int32), w2.view(torch.int32), counts, ROWS=4096, GATED=False, KB=kb, num_warps=8)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, w2, y, psum, use_psum_layout=True, **hint)
    out = torch.empty_like(pooled)
    _pf_combine_kernel[T,](y, row_of, slot_e, probs, st['gs2'], st['post_expert'], out, D=D, num_warps=8)
    return out

def _pf_raw_blob_index(dev):
    """quack's per-m64 _raw_repack: thread t's 16 bytes per (m64, k64) tile -> (row in m64, byte in k-tile)."""
    t = torch.arange(128, device=dev)
    j = torch.arange(16, device=dev)
    b, p = (j >> 2, j & 3)
    r16 = 16 * (t >> 5) + (t & 31) // 4
    c = t & 3
    row = r16[:, None] + 8 * (p & 1)[None, :]
    col = 8 * b[None, :] + c[:, None] + 4 * (p >> 1)[None, :]
    s = torch.arange(32, device=dev)
    srow = 16 * (s >> 3) + (s & 7)
    return (row, col, srow)

def _pf_raw_cta_row_order(m, gated, dev):
    """Weight row permutation into 256-row CTA blocks. Gated (w13: gate rows then up rows):
    block fb = [gate fb*128 + 0..63, up .., gate fb*128 + 64..127, up ..]."""
    if not gated:
        return torch.arange(m, device=dev)
    half = m // 2
    fb = torch.arange(m // 256, device=dev)[:, None, None]
    q = torch.arange(4, device=dev)[None, :, None]
    r = torch.arange(64, device=dev)[None, None, :]
    feat = fb * 128 + (q >> 1) * 64 + r
    return (feat + (q & 1) * half).reshape(-1)

def _pf_raw_repack(w, sf, gated):
    """w [E, M, K/2] uint8 packed e2m1, sf [E, M, K/16] e4m3 -> (blob [E, NFB, KT, 4, 128, 16] uint8,
    sfblob [E, NFB, KT, 4, 32, 4, 2] uint8). A lossless permutation of the same bytes."""
    E, M, KB = w.shape
    dev = w.device
    sf = sf.view(torch.uint8)
    order = _pf_raw_cta_row_order(M, gated, dev)
    row, col, srow = _pf_raw_blob_index(dev)
    nfb, kt = (M // 256, KB // 32)
    blobs, sfbs = ([], [])
    for e in range(E):
        we = w[e][order]
        se = sf[e][order]
        g = torch.arange(M // 64, device=dev)
        k = torch.arange(kt, device=dev)
        rows = 64 * g[:, None, None, None] + row[None, None]
        cols = 32 * k[None, :, None, None] + col[None, None]
        bl = we[rows, cols]
        bl = bl.view(nfb, 4, kt, 128, 16).permute(0, 2, 1, 3, 4).contiguous()
        bidx = torch.arange(4, device=dev)
        i = torch.arange(2, device=dev)
        srows = 64 * g[:, None, None, None, None] + srow[None, None, :, None, None] + 8 * i[None, None, None, None, :]
        scols = 4 * k[None, :, None, None, None] + bidx[None, None, None, :, None]
        sb = se[srows, scols]
        sb = sb.view(nfb, 4, kt, 32, 4, 2).permute(0, 2, 1, 3, 4, 5).contiguous()
        blobs.append(bl)
        sfbs.append(sb)
    return (torch.stack(blobs), torch.stack(sfbs))

@triton.jit
def _pf_raw_tile_offsets_kernel(counts_ptr, tstart_ptr, BNC: tl.constexpr):
    e = tl.arange(0, 32)
    c = tl.load(counts_ptr + e)
    nt = (c + BNC - 1) // BNC
    incl = tl.cumsum(nt, 0)
    tl.store(tstart_ptr + e, incl - nt)
    tl.store(tstart_ptr + 1 + e, incl, mask=e == 31)

@triton.jit
def _pf_raw_gather_tiled_kernel(x_ptr, slot_e_ptr, slot_pos_ptr, tstart_ptr, out_ptr, row_ptr, D: tl.constexpr, BNC: tl.constexpr):
    """Each local route's token row x[t] -> its expert's token tile in the swizzled tiled layout;
    row_ptr[t, k] = global sorted row (tile * BN + r) or -1."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    x = tl.load(x_ptr + t * D + c)
    kt = c // 64
    cc = c % 64
    for k in tl.static_range(8):
        e = tl.load(slot_e_ptr + t * 8 + k)
        if e >= 0:
            pos = tl.load(slot_pos_ptr + t * 8 + k)
            tile = tl.load(tstart_ptr + e) + pos // BNC
            r = pos % BNC
            off = ((tile.to(tl.int64) * (D // 64) + kt) * BNC + r) * 64 + (cc // 8 ^ r % 8) * 8 + cc % 8
            tl.store(out_ptr + off, x)
            tl.store(row_ptr + t * 8 + k, tile * BNC + r)
        else:
            tl.store(row_ptr + t * 8 + k, -1)

@triton.jit
def _pf_raw_combine_tiled_kernel(y_ptr, rows_ptr, topk_ptr, pw_ptr, out_ptr, D: tl.constexpr, BNC: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    pw = tl.load(pw_ptr + c).to(tl.float32)
    acc = tl.zeros([D], tl.float32)
    for k in tl.static_range(8):
        g = tl.load(rows_ptr + t * 8 + k).to(tl.int64)
        if g >= 0:
            tile, r = (g // BNC, g % BNC)
            cc = c % 64
            off = ((tile * (D // 64) + c // 64) * BNC + r) * 64 + (cc // 8 ^ r % 8) * 8 + cc % 8
            y = tl.load(y_ptr + off).to(tl.float32) * pw
            y *= tl.rsqrt(tl.sum(y * y, 0) / D + 1e-08)
            acc += tl.load(topk_ptr + t * 8 + k) * y
    tl.store(out_ptr + t * D + c, acc.to(tl.bfloat16))

def _pf_raw_experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st):
    T = pooled.shape[0]
    dev = pooled.device
    max_tiles = triton.cdiv(T * 8 + 32 * (_pf_BN - 1), _pf_BN)
    tstart = torch.empty(33, dtype=torch.int32, device=dev)
    _pf_raw_tile_offsets_kernel[1,](counts, tstart, BNC=_pf_BN)
    xn = torch.empty_like(pooled)
    _pf_pre_expert_norm_kernel[T,](pooled, st['pre_expert'], xn, D=_pf_D, num_warps=4)
    xt = torch.empty((max_tiles * _pf_BN * _pf_D,), dtype=torch.bfloat16, device=dev)
    rows = torch.empty((T, 8), dtype=torch.int32, device=dev)
    _pf_raw_gather_tiled_kernel[T,](xn, slot_e, slot_pos, tstart, xt, rows, D=_pf_D, BNC=_pf_BN, num_warps=4)
    inter = torch.empty_like(xt)
    y = torch.empty_like(xt)
    st['tiled_mod'].moe_gemm(st['blob13'], st['sfb13'], st['w13_gs'], xt, inter, tstart, 32, 64, 64, 1, st['sms'])
    st['tiled_mod'].moe_gemm(st['blob2'], st['sfb2'], st['w2_gs'], inter, y, tstart, 16, 64, 64, 0, st['sms'])
    out = torch.empty_like(pooled)
    _pf_raw_combine_tiled_kernel[T,](y, rows, topk_w, st['post_expert_bf16'], out, D=_pf_D, BNC=_pf_BN, num_warps=4)
    return out

@triton.jit
def _pf_encode_blob_fast(B, S, O):
    pid = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, 128)
    slot = t // 32 * 8 + t % 32 // 4
    for pair in tl.static_range(2):
        x = tl.load(B + pid * 512 + t * 4 + pair * 2).to(tl.uint32)
        y = tl.load(B + pid * 512 + t * 4 + pair * 2 + 1).to(tl.uint32)
        c0 = x >> 0 & 15
        c1 = x >> 4 & 15
        c2 = x >> 8 & 15
        c3 = x >> 12 & 15
        c4 = x >> 16 & 15
        c5 = x >> 20 & 15
        c6 = x >> 24 & 15
        c7 = x >> 28 & 15
        c8 = y >> 0 & 15
        c9 = y >> 4 & 15
        c10 = y >> 8 & 15
        c11 = y >> 12 & 15
        c12 = y >> 16 & 15
        c13 = y >> 20 & 15
        c14 = y >> 24 & 15
        c15 = y >> 28 & 15
        oA = (c0 >> 3 & 1) << 15 | (c0 >> 2 & 1) << 8 | (c0 >> 1 & 1) << 7 | (c0 >> 0 & 1) << 6 | (c1 >> 3 & 1) << 31 | (c1 >> 2 & 1) << 24 | (c1 >> 1 & 1) << 23 | (c1 >> 0 & 1) << 22 | (c2 >> 3 & 1) << 19 | (c2 >> 2 & 1) << 12 | (c2 >> 1 & 1) << 11 | (c2 >> 0 & 1) << 10 | (c3 >> 3 & 1) << 3 | (c3 >> 2 & 1) << 28 | (c3 >> 1 & 1) << 27 | (c3 >> 0 & 1) << 26 | (c4 >> 3 & 1) << 25 | (c4 >> 2 & 1) << 18 | (c4 >> 1 & 1) << 17 | (c4 >> 0 & 1) << 16 | (c5 >> 3 & 1) << 9 | (c5 >> 2 & 1) << 2 | (c5 >> 1 & 1) << 1 | (c5 >> 0 & 1) << 0 | (c12 >> 0 & 1) << 30 | (c13 >> 0 & 1) << 14 | (c14 >> 3 & 1) << 13 | (c14 >> 1 & 1) << 5 | (c14 >> 0 & 1) << 4 | (c15 >> 3 & 1) << 29 | (c15 >> 1 & 1) << 21 | (c15 >> 0 & 1) << 20
        oB = (c6 >> 3 & 1) << 17 | (c6 >> 2 & 1) << 10 | (c6 >> 1 & 1) << 9 | (c6 >> 0 & 1) << 8 | (c7 >> 3 & 1) << 1 | (c7 >> 2 & 1) << 26 | (c7 >> 1 & 1) << 25 | (c7 >> 0 & 1) << 24 | (c8 >> 3 & 1) << 21 | (c8 >> 2 & 1) << 14 | (c8 >> 1 & 1) << 13 | (c8 >> 0 & 1) << 12 | (c9 >> 3 & 1) << 5 | (c9 >> 2 & 1) << 30 | (c9 >> 1 & 1) << 29 | (c9 >> 0 & 1) << 28 | (c10 >> 3 & 1) << 27 | (c10 >> 2 & 1) << 20 | (c10 >> 1 & 1) << 19 | (c10 >> 0 & 1) << 18 | (c11 >> 3 & 1) << 11 | (c11 >> 2 & 1) << 4 | (c11 >> 1 & 1) << 3 | (c11 >> 0 & 1) << 2 | (c12 >> 3 & 1) << 7 | (c12 >> 2 & 1) << 0 | (c12 >> 1 & 1) << 31 | (c13 >> 3 & 1) << 23 | (c13 >> 2 & 1) << 16 | (c13 >> 1 & 1) << 15 | (c14 >> 2 & 1) << 6 | (c15 >> 2 & 1) << 22
        tl.store(O + pid * 512 + t * 4 + pair * 2, oA)
        tl.store(O + pid * 512 + t * 4 + pair * 2 + 1, oB)


class _pf_Experts:
    """One GPU's experts [start, start + count) in Humming's layout, and the kernels for each bs."""

    def __init__(self, weights, start: int, count: int, tma=False):
        self.tma = tma
        self.gemm_type = _pf_GemmType.INDEXED
        device = weights['w13'].device
        self.count = count
        self.d_expert = weights['w13'].shape[1] // 2
        self.d_hidden = weights['w2'].shape[1]
        self.layer = torch.nn.Module()
        sl = slice(start, start + count)
        for name, n, k, n_stacks in (('w13', 2 * self.d_expert, self.d_hidden, [self.d_expert] * 2), ('w2', self.d_hidden, self.d_expert, [self.d_hidden])):
            checkpoint = {'weight': weights[name][sl].contiguous(), 'weight_scale': weights[f'{name}_sf'][sl].contiguous(), 'weight_scale_2': weights[f'{name}_gs'][sl].contiguous()}
            schema, tensors = _pf_ModeloptNvfp4WeightSchema().convert_humming(tensors=checkpoint, shape_n_stacks=n_stacks, shape_k_stacks=[k], param_dtype=torch.bfloat16, num_experts=count)
            assert schema.weight_scale_2_type == _pf_WeightScale2Type.TENSOR, schema
            for key, tensor in tensors.items():
                setattr(self.layer, f'{name}_{key}', torch.nn.Parameter(tensor, requires_grad=False))
            _pf_HummingMethod.prepare_layer_meta(layer=self.layer, shape_n=n, shape_k=k, weight_schema=schema, num_experts=count, pad_n_to_multiple=256, pad_k_to_multiple=128, torch_dtype=torch.bfloat16, sublayer_name=name)
            _pf_HummingMethod.transform_humming_layer(self.layer, sublayer_name=name)
        self.locks = torch.zeros(1024, dtype=torch.int32, device=device)
        self.expert_map = torch.full((_pf_N_EXPERTS,), -1, dtype=torch.int32, device=device)
        self.expert_map[sl] = torch.arange(count, dtype=torch.int32, device=device)
        self.kernels = {'w13': {}, 'w2': {}}
        self.plans = {}
        self.compile([1 << i for i in range(16)])

    def weight(self, name: str) -> dict:
        return {'weight': getattr(self.layer, f'{name}_weight'), 'weight_scale': getattr(self.layer, f'{name}_weight_scale'), 'weight_scale_2': getattr(self.layer, f'{name}_weight_scale_2')}

    def routes(self, bs: int) -> int:
        """The GEMM's M for Humming's heuristics: the routes expected on this GPU."""
        return max(1, bs * _pf_TOP_K * self.count // _pf_N_EXPERTS)

    def tuning(self, name: str, routes: int) -> str:
        config = dict(_pf_get_heuristics_config(self.layer.humming_metas[name], shape_m=routes, gemm_type=self.gemm_type))
        if self.tma and torch.cuda.get_device_capability()[0] == 9:
            config.update(use_tma=True, use_tma_a=False, use_tma_as=False, use_tma_c=False, use_mbarrier=True, use_warp_spec=True)
        return json.dumps(config)

    def compile(self, batch_sizes) -> None:
        """Compile, in parallel, every kernel these batch sizes need that isn't compiled yet."""
        wanted = {bs: {name: self.tuning(name, self.routes(bs)) for name in self.kernels} for bs in batch_sizes}
        for name, kernels in self.kernels.items():
            new = sorted({t[name] for t in wanted.values()} - kernels.keys())
            if not new:
                continue
            launch = _pf_HummingKernel.prepare_kernels(self.layer.humming_metas[name].to_str(), json.dumps({'use_f16_accum': False, 'gemm_type': self.gemm_type.value}), [[i, i + 1, json.loads(t)] for i, t in enumerate(new)])
            for t, (_, _, kernel_id, num_sms) in zip(new, launch.view(-1, 4).tolist()):
                kernels[t] = torch.tensor([0, 1 << 30, kernel_id, num_sms], dtype=torch.int64)
        for bs, tunings in wanted.items():
            self.plans[bs] = (self.routes(bs), {name: (json.loads(t)['block_shape'][0], self.kernels[name][t]) for name, t in tunings.items()})

    def plan(self, bs: int):
        bucket = 1 << (max(bs, 1) - 1).bit_length()
        if bucket not in self.plans:
            self.compile([bucket])
        return (self.routes(bs), self.plans[bucket][1])

@triton.jit
def _pf_fixed_offsets_kernel(Starts, T: tl.constexpr):
    e = tl.arange(0, 32)
    tl.store(Starts + e, e * T)

def _pf_experts_humming_masked(pooled, topk_w, slot_e, slot_pos, counts, st):
    T = pooled.shape[0]
    stride = triton.cdiv(T, 128) * 128
    starts = torch.empty(32, device=pooled.device, dtype=torch.int32)
    _pf_fixed_offsets_kernel[1,](starts, T=stride)
    a = torch.empty((32 * stride, 4096), device=pooled.device, dtype=torch.bfloat16)
    rows = torch.empty((T, 8), device=pooled.device, dtype=torch.int32)
    _pf_gather_norm_kernel[T,](pooled, st['pre_expert'], slot_e, slot_pos, starts, a, rows, D=4096, num_warps=4)
    h = torch.empty((32 * stride, 8192), device=pooled.device, dtype=torch.bfloat16)
    inter = torch.empty_like(a)
    y = torch.empty_like(a)
    bank = st['humming']
    routes, gemms = bank.plan(T)
    _pf_humming_ops.launch_kernel(configs=gemms['w13'][1], inputs=a, outputs=h, **bank.weight('w13'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)
    _pf_swiglu_kernel[32, triton.cdiv(T, 4)](h, inter, starts, counts, st['ones'], BR=4, D=4096, num_warps=8)
    _pf_humming_ops.launch_kernel(configs=gemms['w2'][1], inputs=inter, outputs=y, **bank.weight('w2'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)
    out = torch.empty_like(pooled)
    _pf_combine_kernel[T,](y, rows, slot_e, topk_w, st['ones'], st['post_expert'], out, D=4096, num_warps=4)
    return out

class _pf_RawModule:

    def __init__(self, mod, version):
        self.moe_gemm = getattr(mod, version + '_gemm')


_BEST6_GROUPED_ERROR=None
_BEST6_CHOICES=('v9nt1','v9nt2','raw17','humming_masked','dg_w8')


def _best6_hopper(device):
    return torch.cuda.get_device_capability(device)==(9,0)


def _best6_import_humming():
    global _pf_humming_ops,_pf_GemmType,_pf_WeightScale2Type,_pf_HummingKernel
    global _pf_HummingMethod,_pf_ModeloptNvfp4WeightSchema,_pf_get_heuristics_config
    previous=os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')
    os.environ['HUMMING_DISABLE_PARALLEL_BUILD']='1'
    try:
        from humming import ops as _pf_humming_ops
        from humming.config import GemmType as _pf_GemmType,WeightScale2Type as _pf_WeightScale2Type
        from humming.kernel.humming import HummingKernel as _pf_HummingKernel
        from humming.layer import HummingMethod as _pf_HummingMethod
        from humming.schema.modelopt import ModeloptNvfp4WeightSchema as _pf_ModeloptNvfp4WeightSchema
        from humming.tune import get_heuristics_config as _pf_get_heuristics_config
    finally:
        if previous is None:os.environ.pop('HUMMING_DISABLE_PARALLEL_BUILD',None)
        else:os.environ['HUMMING_DISABLE_PARALLEL_BUILD']=previous


def _best6_humming(weights):
    """Exact prefill weight transformation and compile methods, limited to this task's M."""
    _best6_import_humming()
    bank=_pf_Experts.__new__(_pf_Experts)
    bank.tma=False;bank.gemm_type=_pf_GemmType.GROUPED_MASKED
    bank.count=32;bank.d_expert=4096;bank.d_hidden=4096;bank.layer=torch.nn.Module()
    device=weights['w13'].device
    for name,n,k,stacks in (('w13',8192,4096,[4096,4096]),('w2',4096,4096,[4096])):
        checkpoint={'weight':weights[name].contiguous(),'weight_scale':weights[name+'_sf'].contiguous(),
                    'weight_scale_2':weights[name+'_gs'].contiguous()}
        schema,tensors=_pf_ModeloptNvfp4WeightSchema().convert_humming(tensors=checkpoint,
            shape_n_stacks=stacks,shape_k_stacks=[k],param_dtype=torch.bfloat16,num_experts=32)
        assert schema.weight_scale_2_type==_pf_WeightScale2Type.TENSOR
        for key,value in tensors.items():setattr(bank.layer,name+'_'+key,torch.nn.Parameter(value,requires_grad=False))
        _pf_HummingMethod.prepare_layer_meta(layer=bank.layer,shape_n=n,shape_k=k,weight_schema=schema,num_experts=32,
            pad_n_to_multiple=256,pad_k_to_multiple=128,torch_dtype=torch.bfloat16,sublayer_name=name)
        _pf_HummingMethod.transform_humming_layer(bank.layer,sublayer_name=name)
    bank.locks=torch.zeros(1024,device=device,dtype=torch.int32)
    bank.expert_map=torch.full((256,),-1,device=device,dtype=torch.int32)
    bank.expert_map[:32]=torch.arange(32,device=device,dtype=torch.int32)
    bank.kernels={'w13':{},'w2':{}};bank.plans={}
    bank.compile([128,256])
    return bank


def _best6_build_adapters(weights):
    device=weights['w13'].device
    st=dict(pre_expert=(1.+weights['pre_expert_norm'].float()).contiguous(),
            post_expert=(1.+weights['post_expert_norm'].float()).bfloat16().contiguous(),
            post_expert_bf16=(1.+weights['post_expert_norm'].float()).bfloat16().contiguous(),
            gs13_gate=weights['w13_gs'].float().contiguous(),gs13_up=weights['w13_gs'].float().contiguous(),
            gs2=weights['w2_gs'].float().contiguous(),w13_gs=weights['w13_gs'],w2_gs=weights['w2_gs'],
            ones=torch.ones(32,device=device),available=[],rejected={},sms=torch.cuda.get_device_properties(device).multi_processor_count)
    try:
        if _BEST6_GROUPED_ERROR is not None:raise RuntimeError(_BEST6_GROUPED_ERROR)
        mod=_pf_grouped_module()
        raw=dict(st)
        raw['blob13'],raw['sfb13']=_pf_raw_repack(weights['w13'],weights['w13_sf'],True)
        raw['blob2'],raw['sfb2']=_pf_raw_repack(weights['w2'],weights['w2_sf'],False)
        for key,sfkey in (('blob13','sfb13'),('blob2','sfb2')):
            b=raw[key];sf=raw[sfkey];out=torch.empty_like(b)
            _pf_encode_blob_fast[(b.numel()//2048,)](b.view(torch.int32),sf,out.view(torch.int32),num_warps=4)
            raw[key]=out  
        raw['tiled_mod']=_pf_RawModule(mod,'raw17')
        st['raw17']=raw;st['available'].append('raw17')
    except Exception as exc:st['rejected']['raw17']=_best6_reason(exc)
    if _best6_hopper(device):
        try:
            st['w13p']=_pf_repack_v6(weights['w13'],weights['w13_sf'],True)
            st['w2p']=_pf_repack_v6(weights['w2'],weights['w2_sf'],False)
            import deep_gemm
            deep_gemm.use_deterministic_algorithms(True)
            st['available'].append('dg_w8')
        except Exception as exc:st['rejected']['dg_w8']=_best6_reason(exc)
        try:
            st['humming']=_best6_humming(weights);st['available'].append('humming_masked')
        except Exception as exc:st['rejected']['humming_masked']=_best6_reason(exc)
    else:
        st['rejected'].update({name:'Hopper-only library; forced screen required' for name in ('dg_w8','humming_masked')})
    torch.cuda.synchronize(device)
    return st


def _best6_reason(exc):
    return type(exc).__name__+': '+(str(exc).splitlines() or [''])[0][:180]


@triton.jit
def _best6_pool(PR,POOLED,S:tl.constexpr,PS:tl.constexpr,RS:tl.constexpr):
    t=tl.program_id(0);d=tl.arange(0,4096);p=tl.full((4096,),0.,tl.float32)
    for s in tl.static_range(S):p+=tl.load(PR+s*PS+t*RS+d).to(tl.float32)
    tl.store(POOLED+t*4096+d,p.to(tl.bfloat16))


@triton.jit
def _best6_positions(IDS,POS,COUNTS,SIZE:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0);j=tl.arange(0,B)
    ids=tl.load(IDS+j,j<SIZE,-1);hit=(ids==e)&(j<SIZE)
    pos=tl.cumsum(hit.to(tl.int32),0)-1
    tl.store(POS+j,pos,hit)
    if e==0:tl.store(POS+j,0,(j<SIZE)&(ids<0))
    tl.store(COUNTS+e,tl.sum(hit.to(tl.int32),0))


def _best6_optional(pr,wts,lids,state,name):
    p,s=pr;m=lids.shape[0];device=p.device
    pooled=torch.empty((m,4096),device=device,dtype=torch.bfloat16)
    _best6_pool[(m,)](p,pooled,S=s,PS=p.stride(0),RS=p.stride(1),num_warps=4)
    pos=torch.empty_like(lids);counts=torch.empty(32,device=device,dtype=torch.int32)
    _best6_positions[(32,)](lids,pos,counts,SIZE=m*8,B=triton.next_power_of_2(m*8),num_warps=4)
    st=state['moe6_adapters']
    if name=='raw17':return _pf_raw_experts_tiled(pooled,wts,lids,pos,counts,st['raw17'])
    if name=='humming_masked':return _pf_experts_humming_masked(pooled,wts,lids,pos,counts,st)
    if name=='dg_w8':return _pf_run_canonical_dg(pooled,wts,lids,pos,counts,st,name)
    raise ValueError('unknown best6 optional expert')


def _best6_call(data,state,name):
    if name=='nt1_lop3':return smallm_experts(data,state['s4_nt1_bank'],SMALLM_FLAGS[data['hidden_states'].shape[0]])
    if name=='hh_m128':return nf_call(data,state)
    if name=='mixed_parallel_m256':return nf_call(data,state)
    if name.startswith('v9nt'):
        nt=int(name[-1])
        base=state['moe_bank'] if nt==1 else state.get('moe_bank_raw',state['moe_bank'])
        bank=base if nt==1 else dict(base,nt=2,mod=state['moe_nt2_mod'])
        return v9_kernel_fn(data,bank)
    return _best6_optional((data['pool_partials'],data['pool_splits']),data['topk_weights'],data['topk_ids'],state,name)


def _best6_experts(pr,x,wts,lids,state):
    m=x.shape[0]
    choice=state.get('moe6_plan',{}).get(m)
    if choice is None:return experts_v9(x,wts,lids,state)
    data=dict(hidden_states=x,topk_weights=wts,topk_ids=lids,pool_partials=pr[0],pool_splits=pr[1],
              config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0))
    return _best6_call(data,state,choice)


def _best6_fixture(weights,m,seed):
    
    import math
    dev=weights['router'].device;cpu=torch.Generator().manual_seed(5354)
    mean=torch.randn(8192,generator=cpu).to(dev);basis=torch.randn((4,8192),generator=cpu).to(dev)
    gen=torch.Generator(device=dev).manual_seed(seed)
    coef=torch.randn((m,4),device=dev,generator=gen)
    share=torch.ones((m//8,8,1),device=dev);share[:,1:]=.6;share=share.reshape(m,1)
    stream=share*(.7*mean+math.sqrt(.9/4)*(coef@basis))+(1.-.9*share.square()).sqrt()*torch.randn((m,8192),device=dev,generator=gen)
    h=_best5_rms(stream)*(1.+weights['pre_ffn_norm'].float())
    logits=h@weights['router'].float().T*ROUTER_MULT;scores=logits.sigmoid()
    ids=(scores+weights['router_bias'].float()).topk(8,-1).indices
    picked=scores.gather(1,ids);tw=picked/picked.sum(-1,keepdim=True).clamp_min(1.e-15)
    local_start=32*int(torch.bincount((ids//32).flatten(),minlength=8).argmax())
    local=(ids-local_start)%256;local=torch.where(local<32,local,-1).int()
    pooled=(h@weights['pooler'].float().T).bfloat16()
    x=(_best5_rms(pooled)*(1.+weights['pre_expert_norm'].float())).bfloat16()
    p=torch.cat((pooled.float(),torch.zeros((m,512),device=dev)),1).unsqueeze(0)
    counts=torch.bincount(local[local>=0],minlength=32)
    route=(int(counts.sum()),int((counts>0).sum()),int(counts.max()))
    return dict(hidden_states=x,topk_ids=local,topk_weights=tw.float(),pool_partials=p,pool_splits=1,
                config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0)),route


def _best6_refresh(graph,data):
    for name in ('hidden_states','topk_ids','topk_weights','pool_partials'):graph[0][name].copy_(data[name])


def _best6_tune(state,weights,flush,device):
    import statistics,time
    
    state['moe6_plan']={m:'v9nt'+str(state.get('moe_nt_plan',{}).get(m,1)) for m in (128,256)};state['moe6_rows']={}
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        for m in (128,256):
            row=dict(selected=state['moe6_plan'][m],reason='budget',rejected={},choices={});state['moe6_rows'][m]=row
            if not _best5_left(6):continue
            fixtures=[];wants=[];routes=[]
            for seed in (9000+m,9000+m+7919,9000+m+15838):
                data,route=_best6_fixture(weights,m,seed);want=_best5_fp32(data,weights)
                if want is None:break
                fixtures.append(data);wants.append(want);routes.append(route)
            row['routes']=routes
            if len(fixtures)!=3:continue
            graphs={}
            for name in ('v9nt1','v9nt2',*state['moe6_adapters']['available']):
                if not _best5_left(4):break
                try:
                    g=_best5_graph(lambda d,name=name:_best6_call(d,state,name),fixtures[0])
                    worst=0.;wr=0.
                    for data,want in zip(fixtures,wants):
                        _best6_refresh(g,data);g[1].replay()
                        err,rerr=_best5_error(g[2],want);worst=max(worst,err);wr=max(wr,rerr)
                    if worst>.01 or wr>.05:raise RuntimeError(f'FP32 gate {worst:.6f}/{wr:.6f}')
                    graphs[name]=g;row['choices'][name]=dict(fp32=round(worst,6),worst_row=round(wr,6))
                except Exception as exc:
                    if name.startswith('v9nt'):raise
                    row['rejected'][name]=_best6_reason(exc)
            if 'v9nt1' not in graphs:continue
            samples={name:[[],[],[]] for name in graphs}
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            for repeat in range(12):
                if not _best5_left(2):break
                index=repeat%3;data=fixtures[index]
                order=list(graphs)
                if repeat%2:order.reverse()
                for name in order:
                    if name not in graphs:continue
                    try:
                        g=graphs[name];_best6_refresh(g,data)
                        flush.sum();a.record();g[1].replay();b.record();b.synchronize()
                        samples[name][index].append(a.elapsed_time(b)*1000.)
                    except Exception as exc:
                        if name.startswith('v9nt'):raise
                        row['rejected'][name]=_best6_reason(exc);graphs.pop(name)
            ranked=[]
            for name in graphs:
                if any(len(x)!=4 for x in samples[name]):continue
                try:
                    g=graphs[name]
                    for data,want in zip(fixtures,wants):
                        _best6_refresh(g,data);g[1].replay()
                        err,wr=_best5_error(g[2],want)
                        if err>.01 or wr>.05:raise RuntimeError(f'FP32 replay gate {err:.6f}/{wr:.6f}')
                    means=[sum(x)/len(x) for x in samples[name]];mean=sum(means)/len(means)
                    row['choices'][name].update(seed_mean_us=[round(x,3) for x in means],mean_us=round(mean,3))
                    ranked.append((mean,name))
                except Exception as exc:
                    if name.startswith('v9nt'):raise
                    row['rejected'][name]=_best6_reason(exc)
            if ranked:
                mean,name=min(ranked);state['moe6_plan'][m]=name;row.update(selected=name,reason='measured')
    finally:torch.backends.cuda.matmul.allow_tf32=previous


def _best6_summary(state):
    print('BEST6_BUILD',json.dumps(dict(available=state['moe6_adapters']['available'],rejected=state['moe6_adapters']['rejected']),separators=(',',':')),flush=True)
    for m,row in state['moe6_rows'].items():print('BEST6_MOE',m,json.dumps(row,separators=(',',':')),flush=True)



import math,functools
_ap_HIDDEN = 8192
_ap_HEAD_DIM = 64
_ap_HQ, _ap_HKV = (16, 2)
_ap_SLIDING_WINDOW = 2048
_ap_ROPE_THETA = 500000.0
_ap_QK_SCALE_FACTOR = 5.4730064863875
_ap_SM_SCALE = _ap_HEAD_DIM ** (-0.5)
_ap_RMS_EPS = 1e-05
_ap_PAGE = 64
_ap_FP8 = torch.float8_e4m3fn
_ap_K_SCALE = _ap_V_SCALE = 1.0
_ap_N_EXPERTS = 256
_ap_TOP_K = 8
_ap_LOCAL_EXPERTS = 32
_ap_STREAM_SEED = 5354
_ap_STREAM_MEAN = 0.7
_ap_STREAM_SHARED, _ap_STREAM_RANK = (0.9, 4)
_ap_DRAFT_SHARE = 0.6

@functools.cache
def _ap_stream_basis(device: str):
    """The residual stream's mean direction [8192] and shared directions [STREAM_RANK, 8192]."""
    gen = torch.Generator().manual_seed(_ap_STREAM_SEED)
    mean = torch.randn((_ap_HIDDEN,), generator=gen)
    shared = torch.randn((_ap_STREAM_RANK, _ap_HIDDEN), generator=gen)
    return (mean.to(device), shared.to(device))
_ap_AGENTIC_SPREAD = 0.075

def _ap_longest(ctx: int, spread: str) -> int:
    """The longest length _lengths can draw."""
    if spread == 'fixed':
        return ctx
    if spread == 'agentic':
        return math.ceil(ctx * (1 + _ap_AGENTIC_SPREAD))
    return 4 * ctx

def _ap_lengths(gen, batch: int, ctx: int, spread: str, device) -> torch.Tensor:
    if spread == 'fixed':
        return torch.full((batch,), ctx, dtype=torch.int64, device=device)
    if spread == 'agentic':
        u = torch.rand((batch,), generator=gen, device=device)
        return (ctx * (1 + _ap_AGENTIC_SPREAD * (2 * u - 1))).long().clamp(1, _ap_longest(ctx, spread))
    z = torch.randn((batch,), generator=gen, device=device)
    return (ctx * torch.exp(z - 0.5)).clamp(16, 4 * ctx).long()

def _ap_quant(x, scale):
    return (x.float() / scale).to(_ap_FP8)

def _ap_rms(x, eps=_ap_RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
_ap_CACHE_CHUNK = 1 << 20
_ap_SOFT_C2 = {'sliding': (0.3, 0.75), 'full': (0.5, 0.62)}

def _ap_fill_cache(gen, shape, device, scale=_ap_K_SCALE, seed=0, dist='normal', layer='full'):
    """FP8 key and value caches of `shape`: N(0, 1) entries, or soft keys for dist="soft"."""
    hkv = shape[-2]
    k_cache = torch.empty(shape, dtype=_ap_FP8, device=device)
    v_cache = torch.empty(shape, dtype=_ap_FP8, device=device)
    mix = None
    if dist == 'soft':
        soft = torch.Generator(device=device).manual_seed(seed + 20727)
        u = torch.randn((hkv, _ap_HEAD_DIM), generator=soft, device=device)
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
        lo, hi = _ap_SOFT_C2[layer]
        c2 = lo + (hi - lo) * torch.rand((hkv, 1), generator=soft, device=device)
        mix = (c2.sqrt() * u, (1 - c2).sqrt())
    for dst, keys in ((k_cache, True), (v_cache, False)):
        flat = dst.view(-1, hkv, _ap_HEAD_DIM)
        for at in range(0, flat.shape[0], _ap_CACHE_CHUNK):
            n = min(_ap_CACHE_CHUNK, flat.shape[0] - at)
            x = torch.randn((n, hkv, _ap_HEAD_DIM), generator=gen, device=device)
            if keys and mix is not None:
                x = mix[0] + mix[1] * x
            flat[at:at + n] = _ap_quant(x, scale)
    return (k_cache, v_cache)

def _ap_fa3_scheduler_metadata(cache_seqlens, cu_seqlens_q, hq, hkv, page, q_len, layer):
    """FA3's tile schedule for a decode step on a full-attention layer
    (sgl_kernel.flash_attn.get_scheduler_metadata), which SGLang builds once per step from the
    batch's lengths and shares across those layers. None on sliding-window layers and verify
    blocks, where SGLang leaves it to each call."""
    if layer != 'full' or q_len != 1:
        return None
    try:
        from sgl_kernel.flash_attn import get_scheduler_metadata
    except ImportError:
        return None
    return get_scheduler_metadata(batch_size=cache_seqlens.shape[0], max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()), num_heads=hq, num_heads_k=hkv, headdim=_ap_HEAD_DIM, cache_seqlens=cache_seqlens, qkv_dtype=_ap_FP8, cu_seqlens_q=cu_seqlens_q, page_size=page, causal=True, num_splits=0)

def _ap_generate_input(batch: int, ctx: int, layer: str, seed: int, q_len: int=1, spread: str='fixed', dist: str='normal', page: int=_ap_PAGE, kv_scale: float=_ap_K_SCALE, local_start: int=0, device: str='cuda', **_):
    """layer: "sliding" (2,048-token window, RoPE) or "full" (every cached token, no RoPE).
    q_len: 1 for a decode step, 8 for a speculative-decoding verify block, 200-16384 for a
    prefill chunk; causal inside the block.
    dist="outliers": 1% of residual-stream channels are 10x larger.
    dist="soft": soft cached keys (see SOFT_C2)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = batch * q_len
    seq_lens = _ap_lengths(gen, batch, ctx, spread, device)
    pages_per = (seq_lens + q_len + page - 1) // page
    longest = _ap_longest(ctx, spread)
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = (int(pages_per[b]), int(offsets[b]))
        page_table[b, :n] = perm[o:o + n]
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    shape = (total, page, _ap_HKV, _ap_HEAD_DIM)
    k_cache, v_cache = _ap_fill_cache(gen, shape, device, kv_scale, seed, dist, layer)
    mean, shared = _ap_stream_basis(str(device))
    coef = torch.randn((tokens, _ap_STREAM_RANK), generator=gen, device=device)
    scale = torch.ones((batch, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = _ap_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = scale * (_ap_STREAM_MEAN * mean + math.sqrt(_ap_STREAM_SHARED / _ap_STREAM_RANK) * coef @ shared) + (1 - _ap_STREAM_SHARED * scale * scale).sqrt() * torch.randn((tokens, _ap_HIDDEN), generator=gen, device=device)
    if dist == 'outliers':
        idx = torch.randperm(_ap_HIDDEN, generator=gen, device=device)[:_ap_HIDDEN // 100]
        stream[:, idx] *= 10
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=device)
    return {'stream': stream, 'attn_input': _ap_rms(stream).to(torch.bfloat16), 'positions': positions, 'cache_loc': cache_loc, 'cache_seqlens': cache_seqlens, 'cu_seqlens_q': cu_seqlens_q, 'scheduler_metadata': _ap_fa3_scheduler_metadata(cache_seqlens, cu_seqlens_q, _ap_HQ, _ap_HKV, page, q_len, layer), 'k_cache': k_cache, 'v_cache': v_cache, 'k_scale': torch.tensor(kv_scale, dtype=torch.float32, device=device), 'v_scale': torch.tensor(kv_scale, dtype=torch.float32, device=device), 'page_table': page_table, 'seq_lens': seq_lens.to(torch.int32), 'local_start': torch.tensor([local_start], dtype=torch.int32, device=device), 'config': {'batch': batch, 'q_len': q_len, 'tokens': tokens, 'hidden': _ap_HIDDEN, 'num_q_heads': _ap_HQ, 'num_kv_heads': _ap_HKV, 'head_dim': _ap_HEAD_DIM, 'page_size': page, 'window': _ap_SLIDING_WINDOW if layer == 'sliding' else 0, 'rope_theta': _ap_ROPE_THETA if layer == 'sliding' else 0.0, 'qk_scale_factor': _ap_QK_SCALE_FACTOR, 'sm_scale': _ap_SM_SCALE, 'kv_scale': kv_scale, 'n_experts': _ap_N_EXPERTS, 'top_k': _ap_TOP_K, 'local_experts': _ap_LOCAL_EXPERTS}}



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

@triton.jit
def _best7_old_group_barrier(CNT, n):
    """Spin until n programs have arrived at counter CNT (zeroed per call; the n programs must be
    co-resident, which holds for the small grids that use it)."""
    tl.atomic_add(CNT, 1, sem='release')
    while tl.atomic_add(CNT, 0, sem='acquire') < n:
        pass

@triton.jit
def _best7_old_boundary_kernel(BR, S: tl.constexpr, bs_stride, br_stride, STREAM, STREAM_OUT, OUT, POSTW, ALPHA, BETA, PREW, SUMS, CNT, post_eps, pre_eps, NB: tl.constexpr, H: tl.constexpr, C: tl.constexpr, HAS_POSTW: tl.constexpr, USE_PDL: tl.constexpr):
    """Token t, chunk c of C. With C > 1 the chunks of a token meet at a barrier to share the four
    sums the two norms need: sum b^2, then sum (a s + b r_b)^2 = A + 2 r_b B + r_b^2 Cc."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    _pdl_start(USE_PDL)
    CH: tl.constexpr = H // C
    offs = c * CH + tl.arange(0, CH)
    alpha = tl.load(ALPHA + offs)
    beta = tl.load(BETA + offs)
    prew = tl.load(PREW + offs)
    if HAS_POSTW:
        postw = tl.load(POSTW + offs)
    _pdl_wait(USE_PDL)
    st = tl.load(STREAM + t * H + offs)
    br = tl.zeros((CH,), tl.float32)
    for s in tl.static_range(S):
        br += tl.load(BR + s * bs_stride + t * br_stride + offs, mask=offs < NB, other=0.0).to(tl.float32)
    if HAS_POSTW:
        br = br * postw
    if C == 1:
        br = br * tl.rsqrt(tl.sum(br * br, 0) / H + post_eps)
        st = alpha * st + beta * br
        rs = tl.rsqrt(tl.sum(st * st, 0) / H + pre_eps)
    else:
        a_s = alpha * st
        b_b = beta * br
        tl.atomic_add(SUMS + t * 4 + 0, tl.sum(br * br, 0))
        tl.atomic_add(SUMS + t * 4 + 1, tl.sum(a_s * a_s, 0))
        tl.atomic_add(SUMS + t * 4 + 2, tl.sum(a_s * b_b, 0))
        tl.atomic_add(SUMS + t * 4 + 3, tl.sum(b_b * b_b, 0))
        _best7_old_group_barrier(CNT + t, C)
        sbb = tl.atomic_add(SUMS + t * 4 + 0, 0.0, sem='acquire')
        sss = tl.atomic_add(SUMS + t * 4 + 1, 0.0, sem='acquire')
        ssb = tl.atomic_add(SUMS + t * 4 + 2, 0.0, sem='acquire')
        sbb2 = tl.atomic_add(SUMS + t * 4 + 3, 0.0, sem='acquire')
        rb = tl.rsqrt(sbb / H + post_eps)
        st = a_s + b_b * rb
        rs = tl.rsqrt((sss + 2.0 * rb * ssb + rb * rb * sbb2) / H + pre_eps)
    tl.store(STREAM_OUT + t * H + offs, st)
    tl.store(OUT + t * H + offs, (st * rs * prew).to(tl.bfloat16))

def _best7_old_chunks(tokens):
    return 16 if tokens <= 8 else 1

def _best7_old_boundary(branch, stream, alpha, beta, prew, sync, postw=None, use_pdl=None):
    P, S = branch
    tokens = stream.shape[0]
    C = _best7_old_chunks(tokens)
    s_out = torch.empty_like(stream)
    out = torch.empty((tokens, HIDDEN), device=stream.device, dtype=torch.bfloat16)
    sums, cnt = sync
    _launch(_best7_old_boundary_kernel, (tokens, C), P, S, P.stride(0), P.stride(1), stream, s_out, out, postw if postw is not None else alpha, alpha, beta, prew, sums, cnt, POST_EPS, RMS_EPS, NB=P.shape[2], H=HIDDEN, C=C, HAS_POSTW=postw is not None, num_warps=8 if C == 1 else 4, use_pdl=use_pdl)
    return (out, s_out)

@triton.jit
def _best7_old_route_norm_kernel(PR, S: tl.constexpr, ps_stride, pr_stride, BIAS, PREW, LSTART, X, IDS, WTS, LIDS, SUMS, CNT, mult, eps, D: tl.constexpr, NE: tl.constexpr, K: tl.constexpr, NLOCAL: tl.constexpr, C: tl.constexpr, USE_PDL: tl.constexpr):
    """Token t. Program 0: sigmoid scores from the hi+lo logits, biased top-8, renormalized
    weights, local ids (-1 elsewhere). Programs 1..C: rmsnorm(p) * (1 + w) over D / C each."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    _pdl_start(USE_PDL)
    if c == 0:
        e = tl.arange(0, NE)
        bias = tl.load(BIAS + e)
        lstart = tl.load(LSTART)
        _pdl_wait(USE_PDL)
        logit = tl.zeros((NE,), tl.float32)
        for s in tl.static_range(S):
            row = PR + s * ps_stride + t * pr_stride
            logit += tl.load(row + D + e).to(tl.float32) + tl.load(row + D + NE + e).to(tl.float32)
        score = tl.sigmoid(logit * mult)
        biased = score + bias
        ids = tl.zeros((K,), tl.int32)
        sk = tl.zeros((K,), tl.float32)
        kr = tl.arange(0, K)
        for k in tl.static_range(K):
            idx = tl.argmax(biased, 0)
            sel = e == idx
            ids = tl.where(kr == k, idx, ids)
            sk = tl.where(kr == k, tl.sum(tl.where(sel, score, 0.0), 0), sk)
            biased = tl.where(sel, -float('inf'), biased)
        tl.store(IDS + t * K + kr, ids)
        tl.store(WTS + t * K + kr, sk / (tl.sum(sk, 0) + 1e-15))
        lid = (ids - lstart + NE) % NE
        tl.store(LIDS + t * K + kr, tl.where(lid < NLOCAL, lid, -1))
    else:
        CH: tl.constexpr = D // C
        offs = (c - 1) * CH + tl.arange(0, CH)
        prew = tl.load(PREW + offs)
        _pdl_wait(USE_PDL)
        p = tl.zeros((CH,), tl.float32)
        for s in tl.static_range(S):
            p += tl.load(PR + s * ps_stride + t * pr_stride + offs).to(tl.float32)
        if C == 1:
            ss = tl.sum(p * p, 0)
        else:
            tl.atomic_add(SUMS + t, tl.sum(p * p, 0))
            _best7_old_group_barrier(CNT + t, C)
            ss = tl.atomic_add(SUMS + t, 0.0, sem='acquire')
        tl.store(X + t * D + offs, (p * tl.rsqrt(ss / D + eps) * prew).to(tl.bfloat16))

def _best7_old_route_and_norm(pr, state, data, sync):
    P, S = pr
    tokens = P.shape[1]
    C = _best7_old_chunks(tokens)
    dev = P.device
    x = torch.empty((tokens, D_EXPERT), device=dev, dtype=torch.bfloat16)
    ids = torch.empty((tokens, TOP_K), device=dev, dtype=torch.int32)
    wts = torch.empty((tokens, TOP_K), device=dev, dtype=torch.float32)
    lids = torch.empty((tokens, TOP_K), device=dev, dtype=torch.int32)
    sums, cnt = sync
    _launch(_best7_old_route_norm_kernel, (tokens, 1 + C), P, S, P.stride(0), P.stride(1), state['router_bias'], state['pre_expert_w'], data['local_start'], x, ids, wts, lids, sums, cnt, ROUTER_MULT, RMS_EPS, D=D_EXPERT, NE=N_EXPERTS, K=TOP_K, NLOCAL=LOCAL_EXPERTS, C=C, num_warps=4 if C > 1 else 8)
    return (x, ids, wts, lids)



SP_CUDA = '#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\n// Standalone ABI: no PyTorch headers, no architecture-specific host branch.\n// The exact cp.async + mma.sync path also executes on the local sm_120.\nnamespace stream_proj {\nusing B = __nv_bfloat16;\n__device__ __forceinline__ uint4 load16(const B* p) {\n  uint4 v;\n  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"\n      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n__device__ __forceinline__ void cp16(B* dst, const B* src, bool valid) {\n  unsigned a = (unsigned)__cvta_generic_to_shared(dst);\n  int bytes = valid ? 16 : 0;\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"\n      :: "r"(a), "l"(src), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mma(float* d, unsigned a0, unsigned a1,\n                                   unsigned a2, unsigned a3, unsigned b0, unsigned b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\n      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ void output(void* y, size_t i, float v, bool fp32) {\n  if (fp32) ((float*)y)[i] = v;\n  else ((B*)y)[i] = __float2bfloat16_rn(v);\n}\n\n// No spinning, no extra launch: every split publishes its own disjoint tile;\n// the last CTA reduces in FP32. The private counter returns to zero before exit.\n// scratch/counters belong to this call site and cannot be shared concurrently.\ntemplate<int R, int T, int THREADS>\n__device__ __forceinline__ void finish(float* partial, unsigned* counters, void* y,\n    int M, int N, int splits, int nt, int mt, bool fp32) {\n  __shared__ int last;\n  const int tile = blockIdx.z * nt + blockIdx.x;\n  __threadfence();\n  __syncthreads();\n  if (threadIdx.x == 0) last = atomicAdd(counters + tile, 1u) == (unsigned)(splits - 1);\n  __syncthreads();\n  if (last) {\n    for (int e = threadIdx.x; e < R * T; e += THREADS) {\n      const int r = e % R, t = e / R;\n      if (blockIdx.x * R + r < N && blockIdx.z * T + t < M) {\n        float sum = 0.f;\n        for (int s = 0; s < splits; ++s)\n          sum += ((volatile float*)partial)[((size_t)tile * splits + s) * R * T + e];\n        output(y, (size_t)(blockIdx.z * T + t) * N + blockIdx.x * R + r, sum, fp32);\n      }\n    }\n    __syncthreads();\n    if (threadIdx.x == 0) atomicExch(counters + tile, 0u);\n  }\n}\n\n// A single lane-preserving shuffle creates a true dependency on every\n// outstanding weight vector. The returned FP32 value is exactly +0, but\n// ptxas cannot sink the vector loads to the first FMA that uses each one.\n// Unlike a bare barrier, this survives instruction scheduling in SASS.\ntemplate<int U, int ROWS>\n__device__ __forceinline__ float ring_dependency(uint4 (&w)[ROWS][U]) {\n  unsigned hash = 0;\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r)\n#pragma unroll\n    for (int u = 0; u < U; ++u) hash ^= w[r][u].x;\n  unsigned same;\n  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 31, -1;"\n      : "=r"(same) : "r"(hash), "r"(threadIdx.x & 31));\n  return __uint_as_float(same ^ hash);\n}\n\ntemplate<int U, int WARPS, int ROWS>\n__global__ __launch_bounds__(WARPS * 32)\nvoid scalar(const B* __restrict__ w, const B* __restrict__ x, void* y,\n            float* partial, unsigned* counters, int M, int N, int K, int splits, bool fp32) {\n  constexpr int R = WARPS * ROWS;\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int row = blockIdx.x * R + warp * ROWS;\n  const int ks = K / splits, begin = blockIdx.y * ks, end = begin + ks;\n  const int tok = blockIdx.z;\n  float a[ROWS][8] = {};\n  for (int base = begin; base < end; base += U * 256) {\n    // Issue the entire register ring before consuming it. U=8 is 4 KiB of\n    // outstanding weight requests per warp, before adding another resident CTA.\n    uint4 xv[U], wv[ROWS][U];\n#pragma unroll\n    for (int u = 0; u < U; ++u) {\n      const int k = base + u * 256 + lane * 8;\n      xv[u] = k < end ? *reinterpret_cast<const uint4*>(x + (size_t)tok * K + k) : uint4{};\n#pragma unroll\n      for (int r = 0; r < ROWS; ++r)\n        wv[r][u] = k < end && row + r < N ? load16(w + (size_t)(row + r) * K + k) : uint4{};\n    }\n    const float dependency = ring_dependency(wv);\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r)\n#pragma unroll\n      for (int j = 0; j < 8; ++j) a[r][j] += dependency;\n#pragma unroll\n    for (int u = 0; u < U; ++u) {\n      const B* xx = reinterpret_cast<const B*>(&xv[u]);\n#pragma unroll\n      for (int r = 0; r < ROWS; ++r) {\n        const B* ww = reinterpret_cast<const B*>(&wv[r][u]);\n#pragma unroll\n        for (int j = 0; j < 8; ++j)\n          a[r][j] = fmaf(__bfloat162float(ww[j]), __bfloat162float(xx[j]), a[r][j]);\n      }\n    }\n  }\n  const int nt = (N + R - 1) / R, tile = tok * nt + blockIdx.x;\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r) {\n    float v = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) v += a[r][j];\n#pragma unroll\n    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);\n    if (lane == 0 && row + r < N) {\n      if (splits == 1) output(y, (size_t)tok * N + row + r, v, fp32);\n      else partial[((size_t)tile * splits + blockIdx.y) * R + warp * ROWS + r] = v;\n    }\n  }\n  if (splits > 1) finish<R, 1, WARPS * 32>(partial, counters, y, M, N, splits, nt, M, fp32);\n}\n\ntemplate<int R, int T, int BK, int STAGES, int WARPS>\n__global__ __launch_bounds__(WARPS * 32)\nvoid tensor(const B* __restrict__ w, const B* __restrict__ x, void* y,\n            float* partial, unsigned* counters, int M, int N, int K, int splits, bool fp32) {\n  constexpr int STRIDE = BK + 8, STAGE = (R + T) * STRIDE;\n  constexpr int CELLS = (R / 16) * (T / 8) / WARPS;\n  extern __shared__ __align__(16) B smem[];\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, l = lane & 3;\n  const int row0 = blockIdx.x * R, tok0 = blockIdx.z * T;\n  const int ks = K / splits, begin = blockIdx.y * ks, end = begin + ks;\n  const int tiles = (ks + BK - 1) / BK;\n  auto load = [&](int tile) {\n    B* s = smem + (tile % STAGES) * STAGE;\n    const int kb = begin + tile * BK;\n    // 16-byte cp.async; padded shared rows avoid bank conflicts for the MMA\n    // fragment\'s eight independent rows (four lanes per row).\n    for (int v = tid; v < (R + T) * BK / 8; v += WARPS * 32) {\n      const int r = v / (BK / 8), c = (v % (BK / 8)) * 8;\n      const bool is_w = r < R;\n      const int global_r = is_w ? row0 + r : tok0 + r - R;\n      const bool valid = kb + c < end && global_r < (is_w ? N : M);\n      const B* src = valid ? (is_w ? w : x) + (size_t)global_r * K + kb + c : x;\n      cp16(s + r * STRIDE + c, src, valid);\n    }\n    asm volatile("cp.async.commit_group;" ::: "memory");\n  };\n  float a[CELLS][4] = {};\n#pragma unroll\n  for (int st = 0; st < STAGES - 1; ++st) load(st);\n  for (int tile = 0; tile < tiles; ++tile) {\n    load(tile + STAGES - 1);\n    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 1) : "memory");\n    __syncthreads();\n    B* s = smem + (tile % STAGES) * STAGE;\n#pragma unroll\n    for (int k = 0; k < BK; k += 16) {\n#pragma unroll\n      for (int c = 0; c < CELLS; ++c) {\n        const int cell = warp + c * WARPS;\n        const int rn = (cell / (T / 8)) * 16, tm = (cell % (T / 8)) * 8;\n        const B* wa = s + (rn + g) * STRIDE + k + l * 2;\n        const B* xb = s + (R + tm + g) * STRIDE + k + l * 2;\n        unsigned a0 = *reinterpret_cast<const unsigned*>(wa);\n        unsigned a1 = *reinterpret_cast<const unsigned*>(wa + 8 * STRIDE);\n        unsigned a2 = *reinterpret_cast<const unsigned*>(wa + 8);\n        unsigned a3 = *reinterpret_cast<const unsigned*>(wa + 8 * STRIDE + 8);\n        unsigned b0 = *reinterpret_cast<const unsigned*>(xb);\n        unsigned b1 = *reinterpret_cast<const unsigned*>(xb + 8);\n        mma(a[c], a0, a1, a2, a3, b0, b1);\n      }\n    }\n    __syncthreads();\n  }\n  asm volatile("cp.async.wait_group 0;" ::: "memory");\n  const int nt = (N + R - 1) / R, mt = (M + T - 1) / T;\n  const int tile_id = blockIdx.z * nt + blockIdx.x;\n#pragma unroll\n  for (int c = 0; c < CELLS; ++c) {\n    const int cell = warp + c * WARPS;\n    const int rn = (cell / (T / 8)) * 16 + g, tm = (cell % (T / 8)) * 8 + l * 2;\n#pragma unroll\n    for (int i = 0; i < 4; ++i) {\n      const int r = rn + (i / 2) * 8, t = tm + (i & 1);\n      if (row0 + r < N && tok0 + t < M) {\n        if (splits == 1) output(y, (size_t)(tok0 + t) * N + row0 + r, a[c][i], fp32);\n        else partial[((size_t)tile_id * splits + blockIdx.y) * R * T + t * R + r] = a[c][i];\n      }\n    }\n  }\n  if (splits > 1) finish<R, T, WARPS * 32>(partial, counters, y, M, N, splits, nt, mt, fp32);\n}\n\ntemplate<int U, int W, int R>\nint launch_scalar(const B* w, const B* x, void* y, float* p, unsigned* c,\n                  int M, int N, int K, int splits, bool fp32, cudaStream_t stream) {\n  scalar<U,W,R><<<dim3((N + W*R - 1)/(W*R), splits, M), W*32, 0, stream>>>\n      (w,x,y,p,c,M,N,K,splits,fp32);\n  return (int)cudaGetLastError();\n}\ntemplate<int R, int T, int BK, int S, int W>\nint launch_tensor(const B* w, const B* x, void* y, float* p, unsigned* c,\n                  int M, int N, int K, int splits, bool fp32, cudaStream_t stream) {\n#ifdef SP_LOCAL_SHARED_CAP\n  // Local Blackwell has 99 KiB/CTA; H200 has 227 KiB. Execute the identical\n  // fragments/reducer/glue with fewer stages locally instead of skipping it.\n  constexpr int fit = SP_LOCAL_SHARED_CAP / ((R + T) * (BK + 8) * sizeof(B));\n  constexpr int stages = S < fit ? S : fit;\n#else\n  constexpr int stages = S;\n#endif\n  static_assert(stages > 0);\n  constexpr int bytes = stages * (R + T) * (BK + 8) * sizeof(B);\n  auto kernel = tensor<R,T,BK,stages,W>;\n  auto e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);\n  if (e != cudaSuccess) return (int)e;\n  kernel<<<dim3((N + R - 1)/R, splits, (M + T - 1)/T), W*32, bytes, stream>>>\n      (w,x,y,p,c,M,N,K,splits,fp32);\n  return (int)cudaGetLastError();\n}\n} // namespace stream_proj\n\n// Same-source read control: isolate request throughput from arithmetic and\n// output/reduction costs, with the identical cold-L2 graph/event protocol.\n__global__ void sp_read_kernel(const uint4* w, unsigned* sink, size_t vectors) {\n  const size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n  const size_t step = (size_t)gridDim.x * blockDim.x;\n  unsigned sum = 0;\n  for (size_t b = first; b < vectors; b += step * 8) {\n    uint4 v[8];\n#pragma unroll\n    for (int u = 0; u < 8; ++u) {\n      const size_t i = b + u * step;\n      v[u] = i < vectors ? stream_proj::load16((const stream_proj::B*)(w + i)) : uint4{};\n    }\n#pragma unroll\n    for (int u = 0; u < 8; ++u) sum ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;\n  }\n  // Every vector contributes; each thread has a private sink element.\n  sink[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = sum;\n}\nextern "C" int sp_read(const void* w, void* sink, size_t bytes, int sms, void* stream) {\n  sp_read_kernel<<<sms, 512, 0, (cudaStream_t)stream>>>((const uint4*)w, (unsigned*)sink, bytes/16);\n  return (int)cudaGetLastError();\n}\n\nextern "C" int sp_launch(const void* w, const void* x, void* y, void* p, void* c,\n                         int M, int N, int K, int config, int splits, int fp32, void* stream) {\n  using namespace stream_proj;\n  if (K % (splits * 8)) return (int)cudaErrorInvalidValue;\n#define ARG (const B*)w,(const B*)x,y,(float*)p,(unsigned*)c,M,N,K,splits,(bool)fp32,(cudaStream_t)stream\n  switch(config) {\n    case 0: return launch_scalar<4,4,1>(ARG);\n    case 1: return launch_scalar<8,4,1>(ARG);\n    case 2: return launch_scalar<16,4,1>(ARG);\n    case 3: return launch_scalar<8,8,1>(ARG);\n    case 4: return launch_scalar<4,4,2>(ARG);\n    case 5: return launch_scalar<8,4,2>(ARG);\n    case 10: return launch_tensor<32,16,128,3,4>(ARG);\n    case 11: return launch_tensor<32,16,256,3,4>(ARG);\n    case 12: return launch_tensor<64,8,256,3,4>(ARG);\n    case 13: return launch_tensor<64,16,256,3,8>(ARG);\n    case 14: return launch_tensor<32,32,256,3,8>(ARG);\n    case 15: return launch_tensor<32,16,256,4,4>(ARG);\n    case 16: return launch_tensor<64,16,128,4,8>(ARG);\n    case 20: return launch_tensor<32,64,128,3,8>(ARG);\n    case 21: return launch_tensor<64,64,128,2,8>(ARG);\n    case 22: return launch_tensor<32,128,128,3,8>(ARG);\n    case 23: return launch_tensor<64,128,128,2,8>(ARG);\n    case 24: return launch_tensor<32,64,256,3,8>(ARG);\n    case 25: return launch_tensor<32,128,256,2,8>(ARG);\n    default: return (int)cudaErrorInvalidValue;\n  }\n#undef ARG\n}\n'
import ctypes as _sp_ct
import hashlib as _sp_hash
import os as _sp_os

import torch
from torch.utils.cpp_extension import load_inline as _sp_load_inline



SP_CONFIGS = {
    0: (4, 1, 0, 0, 4, 4), 1: (4, 1, 0, 0, 4, 8),
    2: (4, 1, 0, 0, 4, 16), 3: (8, 1, 0, 0, 8, 8),
    4: (8, 1, 0, 0, 4, 4), 5: (8, 1, 0, 0, 4, 8),
    10: (32, 16, 128, 3, 4, 0), 11: (32, 16, 256, 3, 4, 0),
    12: (64, 8, 256, 3, 4, 0), 13: (64, 16, 256, 3, 8, 0),
    14: (32, 32, 256, 3, 8, 0), 15: (32, 16, 256, 4, 4, 0),
    16: (64, 16, 128, 4, 8, 0),
    20: (32, 64, 128, 3, 8, 0), 21: (64, 64, 128, 2, 8, 0),
    22: (32, 128, 128, 3, 8, 0), 23: (64, 128, 128, 2, 8, 0),
    24: (32, 64, 256, 3, 8, 0), 25: (32, 128, 256, 2, 8, 0),
}
_SP_LIB = None


def sp_library():
    global _SP_LIB
    if _SP_LIB is None:
        arch = "9.0a" if torch.cuda.get_device_capability()[0] == 9 else "12.0"
        old = _sp_os.environ.get("TORCH_CUDA_ARCH_LIST")
        _sp_os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        try:
            path = _sp_load_inline(
                name="muse_stream_proj_" + _sp_hash.sha256(SP_CUDA.encode()).hexdigest()[:16],
                cpp_sources="", cuda_sources=SP_CUDA, functions=None,
                is_python_module=False, no_implicit_headers=True,
                extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--ptxas-options=-v"]
                    + (["-DSP_LOCAL_SHARED_CAP=98304"] if arch == "12.0" else []),
                verbose=False,
            )
        finally:
            if old is None:
                _sp_os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                _sp_os.environ["TORCH_CUDA_ARCH_LIST"] = old
        _SP_LIB = _sp_ct.CDLL(path)
        _SP_LIB.sp_launch.argtypes = [_sp_ct.c_void_p] * 5 + [_sp_ct.c_int] * 6 + [_sp_ct.c_void_p]
        _SP_LIB.sp_launch.restype = _sp_ct.c_int
        _SP_LIB.sp_read.argtypes = [_sp_ct.c_void_p] * 2 + [_sp_ct.c_size_t, _sp_ct.c_int, _sp_ct.c_void_p]
        _SP_LIB.sp_read.restype = _sp_ct.c_int
    return _SP_LIB


class StreamingProjection:
    """One-launch projection with private, replay-safe last-CTA scratch.

    Construct outside capture (prepare). Instances must not run concurrently;
    callers on separate streams use separate instances. Weights are BF16 views
    of the original bank. FP32 pool/router keeps all hi and lo rows separately,
    matching the decode lane's boundary consumer ABI.
    """

    def __init__(self, w, m, config, splits=1, fp32=False):
        if w.dtype != torch.bfloat16 or not w.is_contiguous():
            raise ValueError("expected contiguous original BF16 weights")
        if w.shape[1] % (8 * splits):
            raise ValueError("split must preserve an eight-BF16 vector")
        self.lib, self.w, self.m = sp_library(), w, m
        self.config, self.splits, self.fp32 = config, splits, fp32
        r, t, *_ = SP_CONFIGS[config]
        n, k = w.shape
        self.tiles = ((n + r - 1) // r) * ((m + t - 1) // t)
        self.ctas = self.tiles * splits
        self.partial = torch.empty((self.tiles, splits, r * t), dtype=torch.float32, device=w.device)
        self.counters = torch.zeros(self.tiles, dtype=torch.int32, device=w.device)
        self.out = torch.empty((m, n), dtype=torch.float32 if fp32 else torch.bfloat16, device=w.device)

    def __call__(self, x):
        if x.shape != (self.m, self.w.shape[1]) or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("input shape, precision or stride changed")
        e = self.lib.sp_launch(
            self.w.data_ptr(), x.data_ptr(), self.out.data_ptr(),
            self.partial.data_ptr(), self.counters.data_ptr(), self.m,
            self.w.shape[0], self.w.shape[1], self.config, self.splits, int(self.fp32),
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if e:
            raise RuntimeError(f"streaming projection launch failed CUDA={e} config={self.config}")
        return self.out


def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    choice=plan['projection'].get(name,'auto')
    if choice=='auto':return project(x,state[name],name,state,out_dtype)
    if isinstance(choice,tuple) and choice[0]=='stream':
        op=state['best7_stream_ops'][(x.shape[0],name,choice[1],choice[2])]
        return op(x).unsqueeze(0),1
    return _project_variant(x,state[name],name,state,choice,out_dtype)


def _best7_attention(qkv,data,state,plan):
    cfg=data['config'];choice=plan['attention']
    if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in state['partial_port_rows']:
        return _layer_partial_port(qkv,data)
    vq8=state.get('vq8_rows',{}).get((cfg['batch'],cfg['q_len'],bool(cfg['window'])))
    if vq8 is not None:return _vq8_attention(qkv,data,vq8)
    if choice=='auto':choice=state['attention_plan'].get((cfg['batch'],cfg['q_len'],bool(cfg['window'])),('layer',None))
    
    
    if choice[0]!='layer' and not _attn_port_hopper(qkv[0].device):return _layer_stage_attention(qkv,data,state)
    if plan['glue']=='best5' and (choice[0]=='fa3' or (choice[0]=='layer' and cfg['q_len']==8 and _attn_port_hopper(qkv[0].device))):
        value=qkv
        if choice[0]=='fa3':value=(_attn_port_raw(qkv,data).reshape(-1,QKV_SIZE).unsqueeze(0),1)
        q,gate=_best7_old_qk_prep(value,data,state,q_fp8=True)
        return _attention_fa3_public(q,gate,data,state) if choice[0]=='layer' else _fa3_call(q,gate,data,choice[1])
    return _attn_port_variant(qkv,data,state,choice)


def _best7_boundary(branch,stream,state,plan,phase,sync=None):
    post=phase=='b2';alpha=state['alpha2' if post else 'alpha1'];beta=state['beta2' if post else 'beta1']
    prew=state['next_pre_attn_w' if post else 'pre_ffn_w'];postw=state['post_ffn_w'] if post else None
    if plan['glue']=='best5':return _best7_old_boundary(branch,stream,alpha,beta,prew,sync,postw,use_pdl=True)
    if phase not in plan['geometry']:return boundary(branch,stream,alpha,beta,prew,postw=postw)
    if plan['geometry'][phase][0]=='cl':return _glue_cl_boundary(branch,stream,alpha,beta,prew,postw,plan['geometry'][phase])
    p,s=branch;t=stream.shape[0];warps,replicas=plan['geometry'][phase]
    out=torch.empty((t,8192),device=stream.device,dtype=torch.bfloat16);sout=torch.empty_like(stream)
    if t>128 and replicas==1:
        _launch(_boundary_late,(t,),p,s,p.stride(0),p.stride(1),stream,sout,out,
            postw if post else alpha,alpha,beta,prew,NB=p.shape[2],HAS_POSTW=post,num_warps=warps,use_pdl=True)
    else:
        _launch(_boundary_fast,(t,replicas),p,s,p.stride(0),p.stride(1),stream,sout,out,
            postw if post else alpha,alpha,beta,prew,NB=p.shape[2],HAS_POSTW=post,REPLICAS=replicas,num_warps=warps,use_pdl=True)
    return out,sout


def _best7_route(pr,state,data,plan,sync=None):
    if plan['glue']=='best5':return _best7_old_route_and_norm(pr,state,data,sync)
    if 'route' not in plan['geometry']:return route_and_norm(pr,state,data)
    if plan['geometry']['route'][0]=='cl':return _glue_cl_route(pr,state,data,plan['geometry']['route'])
    p,s=pr;t=p.shape[1];dev=p.device;warps,replicas=plan['geometry']['route']
    x=torch.empty((t,4096),device=dev,dtype=torch.bfloat16)
    ids=torch.empty((t,8),device=dev,dtype=torch.int32);wts=torch.empty((t,8),device=dev,dtype=torch.float32);lids=torch.empty_like(ids)
    _launch(_route_fast,(t,replicas),p,s,p.stride(0),p.stride(1),state['router_bias'],state['pre_expert_w'],
        data['local_start'],x,ids,wts,lids,REPLICAS=replicas,num_warps=warps,use_pdl=True)
    return x,ids,wts,lids


def _best7_expert(pr,x,wts,lids,state,plan):
    name=plan['moe']
    if name=='auto':return _best6_experts(pr,x,wts,lids,state)
    data=dict(hidden_states=x,topk_ids=lids,topk_weights=wts,pool_partials=pr[0],pool_splits=pr[1],
        config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0))
    return _best6_call(data,state,name)


def _best7_moe_fixture(weights,tokens,length,seed):
    dev=weights['router'].device
    
    data=_ap_generate_input(batch=tokens//length,ctx=257,layer='sliding',seed=seed,q_len=length,
        spread='agentic' if tokens>length else 'fixed',dist='soft',page=1,device=dev)
    h=_best5_rms(data['stream'])*(1.+weights['pre_ffn_norm'].float())
    scores=(h@weights['router'].float().T*ROUTER_MULT).sigmoid()
    ids=(scores+weights['router_bias'].float()).topk(8,-1).indices
    picked=scores.gather(1,ids);wts=picked/picked.sum(-1,keepdim=True).clamp_min(1e-15)
    start=32*int(torch.bincount((ids//32).flatten(),minlength=8).argmax())
    lids=(ids-start)%256;lids=torch.where(lids<32,lids,-1).int()
    pooled=(h@weights['pooler'].float().T).bfloat16()
    x=(_best5_rms(pooled)*(1.+weights['pre_expert_norm'].float())).bfloat16()
    partial=torch.cat((pooled.float(),torch.zeros((tokens,512),device=dev)),1).unsqueeze(0)
    return dict(hidden_states=x,topk_ids=lids,topk_weights=wts.float(),pool_partials=partial,pool_splits=1,
        config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0))


def _best7_gate_moe(name,tokens,length,state,weights):
    if name in ('nt1_lop3','hh_m128'):return _s4_gate_moe(name,tokens,length,state,weights)
    if name=='mixed_parallel_m256':return _s6_gate_moe(tokens,length,state,weights)
    
    row=state.get('moe6_rows',{}).get(tokens,{})
    if name not in row.get('rejected',{}) and 'mean_us' in row.get('choices',{}).get(name,{}):
        return dict(source='best6 three-seed final gate',**row['choices'][name])
    if not name.startswith('v9nt') and name not in state['moe6_adapters']['available']:
        raise RuntimeError(state['moe6_adapters']['rejected'].get(name,'backend unavailable'))
    if not name.startswith('v9nt'):raise RuntimeError('optional expert lacks a completed best6 final gate')
    worst=0.;worst_row=0.;fixtures=[];wants=[]
    for seed in _ATTENTION_PUBLIC_SEEDS:
        fixture=_best7_moe_fixture(weights,tokens,length,seed)
        want=_best5_fp32(fixture,weights)
        if want is None:raise RuntimeError('prepare budget before fixed expert oracle')
        fixtures.append(fixture);wants.append(want)
    graph=_best5_graph(lambda data:_best6_call(data,state,name),fixtures[0])
    for _ in range(2):
        for fixture,want in zip(fixtures,wants):
            _best6_refresh(graph,fixture);graph[1].replay()
            error,row_error=_best5_error(graph[2],want);worst=max(worst,error);worst_row=max(worst_row,row_error)
    if worst>.01 or worst_row>.05:raise RuntimeError(f'fixed expert FP32 {worst:.5f}/{worst_row:.5f}')
    return dict(source='three public streams, fresh graph replays',fp32=worst,worst_row=worst_row)


def _best7_gate_attention(choice,length,window,state,plan,device):
    errors=[]
    for ctx in (257,2305):
        qkv,data=_attn_port_synthetic(1,length,window,device,ctx)
        got=_best7_attention(qkv,data,state,plan);_attention_store_agrees(qkv,data)
        reference=_attention_reference(qkv,data);rounded=_attention_reference(qkv,data,True)
        ok,error,bound=_attention_agrees(got,reference,rounded)
        if not ok:raise RuntimeError(f'fixed attention FP32 {choice}: {error:.5f}/{bound:.5f}')
        errors.append(error)
    return errors


def _best7_stream_op(tokens,name,choice,state,device):
    if (tokens,name,choice)==(1,'o',('stream',6,1)):return _s6_o_projection_op(state,device)
    _,config,splits=choice
    op=StreamingProjection(state[name],tokens,config,splits,fp32=name=='pool_router')
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        gen=torch.Generator(device=device).manual_seed(62171+tokens+state[name].shape[0])
        x=torch.randn((tokens,state[name].shape[1]),device=device,dtype=torch.bfloat16,generator=gen)
        graph=torch.cuda.CUDAGraph()
        for _ in range(2):op(x)
        with torch.cuda.graph(graph):out=op(x)
        worst=0.;row=0.
        for repeat in range(3):
            x.normal_(generator=gen);graph.replay()
            expected=x.float()@state[name].float().T
            error,erow=_best5_error(out,expected);worst=max(worst,error);row=max(row,erow)
        
        limit=.0001 if name=='pool_router' else .005
        if worst>limit or row>limit*5:raise RuntimeError(f'stream projection gate {name}/{tokens}: {worst:.6f}/{row:.6f}')
        if not bool((op.counters==0).all()):raise RuntimeError('stream projection counter did not reset')
        return op,dict(fp32=worst,worst_row=row,config=config,splits=splits)
    finally:torch.backends.cuda.matmul.allow_tf32=previous


def _best7_gate_prefetch(op,key,state,plan,device):
    batch,length,window=key
    fixtures=[_ap_generate_input(batch=batch,ctx=257,layer='sliding' if window else 'full',q_len=length,
        spread='agentic' if batch>1 else 'fixed',dist='soft',page=1,seed=seed,device=device)
        for seed in _ATTENTION_PUBLIC_SEEDS]
    data=fixtures[0]
    for _ in range(2):_best7_run(data,state,plan,op)
    graphs=[];outs=[]
    for selected in (None,op):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=_best7_run(data,state,plan,selected)
        graphs.append(graph);outs.append(out)
    worst=0.
    for repeat in range(2):
        for fixture in fixtures:
            for name in ('attn_input','stream','local_start'):data[name].copy_(fixture[name])
            for graph in (graphs if repeat==0 else graphs[::-1]):graph.replay()
            for got,want in zip(outs[1],outs[0]):
                if got.dtype==torch.int32:
                    if not torch.equal(got,want):raise RuntimeError('L2 prefetch changed routing IDs')
                else:
                    error,row=_best5_error(got,want);worst=max(worst,error)
                    if error>.001 or row>.005:raise RuntimeError(f'L2 prefetch full-layer gate {error}/{row}')
    return dict(recipe=op.recipe,graph_draws=6,relative_l2=worst,route_ids='exact',transport='real native prefetch')


def _best7_prepare_settings(state,weights,device):
    plans={};state['best7_plans']=plans;state['best7_stream_ops']={};state['best7_prefetch_ops']={};gates={};state['best7_gates']=gates
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        for key,values in BEST7_SETTINGS.items():
            if len(values)!=4:raise ValueError(f'best7 row needs four columns: {key}')
            glue,attention,moe,projection=values
            geometry={}
            if isinstance(glue,tuple):
                glue,b1,route,b2=glue;geometry=dict(b1=b1,route=route,b2=b2)
            if glue not in ('best5','best6'):raise ValueError('unknown best7 glue')
            if attention!='auto' and (not isinstance(attention,tuple) or len(attention)!=2):raise ValueError('invalid attention choice')
            if moe not in ('auto',*_BEST6_CHOICES,'mixed_parallel_m256','nt1_lop3','hh_m128'):raise ValueError('unknown best7 expert')
            if projection=='auto':projection={}
            if not isinstance(projection,dict):raise ValueError('projection column needs auto or role choices')
            projection=dict(projection);prefetch=projection.pop('l2_prefetch',0)
            if prefetch and prefetch not in P7_PREFETCH_RECIPES:raise ValueError('unknown P7 L2 recipe')
            if key=='other' and (projection or attention!='auto' or moe!='auto'):raise ValueError('other shapes retain shape-native choices')
            if key=='other' and prefetch:raise ValueError('L2 prefetch requires an explicit decode-load row')
            dense_chain=BEST7_DENSE_CHAIN.get(key, False)
            if not isinstance(dense_chain, bool):raise ValueError('dense chain switch must be bool')
            plan=dict(glue=glue,geometry=geometry,attention=attention,moe=moe,
                      projection=dict(projection),prefetch=prefetch,dense_chain=dense_chain)
            plans[key]=plan
            if key=='other':continue
            batch,length,window=key;tokens=batch*length
            if moe!='auto':
                gate_key=('moe',tokens,moe)
                try:
                    if gate_key not in gates:gates[gate_key]=_best7_gate_moe(moe,tokens,length,state,weights)
                except Exception as exc:
                    if moe.startswith('v9nt') or moe in ('mixed_parallel_m256','nt1_lop3','hh_m128'):raise
                    plan['moe']='auto'
                    print(f'BEST7_REJECT key={key} expert={moe} reason={_best6_reason(exc)}',flush=True)
            if attention!='auto':
                gate_key=('attention',length,window,str(attention),glue)
                if gate_key not in gates:gates[gate_key]=_best7_gate_attention(attention,length,window,state,plan,device)
            roles=('qkv_gate','o','pool_router','unpooler_slice' if tokens<=8 else 'unpooler')
            for name,choice in projection.items():
                if name not in roles:raise ValueError(f'projection role {name} does not exist at tokens={tokens}')
                if choice in ('auto','custom','cublas'):continue
                if not isinstance(choice,tuple) or len(choice)!=3 or choice[0]!='stream':raise ValueError('expected stream/config/splits projection')
                op_key=(tokens,name,choice[1],choice[2])
                if op_key not in state['best7_stream_ops']:
                    op,evidence=_best7_stream_op(tokens,name,choice,state,device)
                    state['best7_stream_ops'][op_key]=op;gates[('projection',*op_key)]=evidence
            if prefetch:
                if length!=1 or tokens>32:raise ValueError('P7 L2 prefetch is admitted for decode M1-32 only')
                op=P7L2Prefetch(state,tokens,prefetch)
                gates['l2_prefetch',key]=_best7_gate_prefetch(op,key,state,plan,device)
                state['best7_prefetch_ops'][key]=op
        if any(p.get('dense_chain', False) for p in plans.values()):
            _dense_chain_prepare(state, weights, device)
    finally:torch.backends.cuda.matmul.allow_tf32=previous


def _best7_summary(state):
    for key,plan in state['best7_plans'].items():
        if key=='other':continue
        batch,length,window=key;tokens=batch*length
        selected=plan['moe']
        if selected=='auto':selected=state.get('moe6_plan',{}).get(tokens,'v9nt'+str(state.get('moe_nt_plan',{}).get(tokens,1)))
        attention=plan['attention']
        if attention=='auto':attention=state['attention_plan'].get((batch,length,bool(window)),('layer',None))
        roles=('qkv_gate','o','pool_router','unpooler_slice' if tokens<=8 else 'unpooler')
        projection={name:plan['projection'].get(name,'auto') for name in roles}
        projection={name:state['proj_plan'].get((tokens,name),'custom') if choice=='auto' else choice for name,choice in projection.items()}
        print('BEST7_PLAN',json.dumps(dict(key=key,glue=plan['glue'],geometry=plan['geometry'],attention=attention,
            moe=selected,projection=projection,l2_prefetch=plan['prefetch'],dense_chain=plan['dense_chain']),separators=(',',':')),flush=True)



"""Graph-captured, within-call P7 prefetch transport."""
P7_PREFETCH_CUDA = '#include <cuda_runtime.h>\n#include <stdint.h>\n\n// Pure byte-address ABI. All pointer arithmetic happens on a CUDA allocation,\n// explicitly converted to .global before either PTX instruction consumes it.\n// Normal eviction only: no cache set-aside or state surviving a call is used.\nnamespace p7_l2 {\n__device__ __forceinline__ uint4 read16(const char* ptr) {\n  uint4 v;\n  unsigned long long p=__cvta_generic_to_global(ptr);\n  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"\n      :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");\n  return v;\n}\ntemplate<int MODE>\n__global__ void prefetch(const char* a,size_t na,const char* b,size_t nb,unsigned* sink) {\n  const size_t tid=(size_t)blockIdx.x*blockDim.x+threadIdx.x;\n  const size_t step=(size_t)gridDim.x*blockDim.x;\n  unsigned sum=0;\n  if constexpr(MODE==0) {\n    // A bounded number of CTAs runs full 16-byte loads. The checksum makes\n    // every byte observable, and each CTA writes a private sink location.\n    for(size_t i=tid*16;i<na+nb;i+=step*16*8) {\n      uint4 v[8];\n#pragma unroll\n      for(int u=0;u<8;++u) {\n        size_t j=i+u*step*16;\n        v[u]=j<na?read16(a+j):j<na+nb?read16(b+j-na):uint4{};\n      }\n#pragma unroll\n      for(int u=0;u<8;++u) sum^=v[u].x^v[u].y^v[u].z^v[u].w;\n    }\n    sink[tid]=sum;\n  } else {\n    // Each lane issues an independent 64 KiB bulk prefetch. This is a hint;\n    // correctness never depends on the request completing before the join.\n    constexpr size_t chunk=65536;\n    for(size_t i=tid*chunk;i<na+nb;i+=step*chunk) {\n      const char* ptr=i<na?a+i:b+i-na;\n      size_t remain=i<na?na-i:na+nb-i;\n      unsigned count=(unsigned)(remain<chunk?remain:chunk);\n      unsigned long long address=__cvta_generic_to_global(ptr);\n      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"::"l"(address),"r"(count):"memory");\n    }\n  }\n}\n}\n\nextern "C" int p7_prefetch(const void* a,size_t na,const void* b,size_t nb,\n                            void* sink,int ctas,int mode,void* stream) {\n  if(((uintptr_t)a%16)||(na%16)||((uintptr_t)b%16)||(nb%16)||ctas<1||mode<0||mode>1)\n    return (int)cudaErrorInvalidValue;\n  // Separate regions must end on a chunk boundary for the bulk grid mapping.\n  if(mode && ((na%65536)||(nb%65536))) return (int)cudaErrorInvalidValue;\n  if(mode) p7_l2::prefetch<1><<<ctas,128,0,(cudaStream_t)stream>>>\n      ((const char*)a,na,(const char*)b,nb,(unsigned*)sink);\n  else p7_l2::prefetch<0><<<ctas,128,0,(cudaStream_t)stream>>>\n      ((const char*)a,na,(const char*)b,nb,(unsigned*)sink);\n  return (int)cudaGetLastError();\n}\n'

import ctypes as _p7ct
import hashlib as _p7hash
import os as _p7os
import torch
from torch.utils.cpp_extension import load_inline as _p7load




P7_PREFETCH_RECIPES={
    1:(0,8,16,0,0,0), 2:(1,4,16,0,0,0),
    3:(0,8,16,16,0,0),4:(1,4,16,24,0,0),
    5:(0,8,16,0,24,0),6:(1,4,16,0,24,0),
    7:(0,4,0,0,0,8),8:(0,8,0,0,0,8),9:(1,4,0,0,0,8),
    10:(0,8,16,0,24,8),11:(1,4,16,0,24,8),
    12:(0,16,16,0,24,24),13:(0,8,16,24,0,8),
}
_P7_L2_LIB=None


def p7_l2_library():
    global _P7_L2_LIB
    if _P7_L2_LIB is None:
        arch='9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        old=_p7os.environ.get('TORCH_CUDA_ARCH_LIST');_p7os.environ['TORCH_CUDA_ARCH_LIST']=arch
        try:
            path=_p7load(name='muse_p7_l2_'+_p7hash.sha256(P7_PREFETCH_CUDA.encode()).hexdigest()[:16],
                cpp_sources='',cuda_sources=P7_PREFETCH_CUDA,functions=None,is_python_module=False,
                no_implicit_headers=True,extra_cuda_cflags=['-O3','-std=c++17','-lineinfo'],verbose=False)
        finally:
            if old is None:_p7os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_p7os.environ['TORCH_CUDA_ARCH_LIST']=old
        _P7_L2_LIB=_p7ct.CDLL(path)
        _P7_L2_LIB.p7_prefetch.argtypes=[_p7ct.c_void_p,_p7ct.c_size_t,_p7ct.c_void_p,_p7ct.c_size_t,
                                      _p7ct.c_void_p,_p7ct.c_int,_p7ct.c_int,_p7ct.c_void_p]
        _P7_L2_LIB.p7_prefetch.restype=_p7ct.c_int
    return _P7_L2_LIB


class P7L2Prefetch:
    """Private side stream and event pairs, forked/joined inside each timed call.

    Construct in prepare. Each instance belongs to an ordered root stream;
    concurrent roots use separate instances. Weights and sink remain resident.
    Capture/replay includes every prefetch and dependency, with a fresh L2 flush
    before the root graph. No kernel consumes or changes weights' values.
    """
    def __init__(self,state,tokens,recipe):
        self.recipe=recipe;self.spec=P7_PREFETCH_RECIPES[recipe];self.lib=p7_l2_library()
        mode,ctas,ob,pb,late,tail=self.spec
        self.device=state['o'].device
        names=('o','pool_router','unpooler_slice' if tokens<=8 else 'unpooler')
        self.banks={name:state[name] for name in names}
        for w in self.banks.values():
            if w.device!=self.device or w.dtype!=torch.bfloat16 or not w.is_contiguous():
                raise ValueError('prefetch requires original contiguous CUDA BF16 weights')
        unpool=self.banks[names[2]]
        self.regions={
            'front':(state['o'],min(ob<<20,state['o'].numel()*2),state['pool_router'],min(pb<<20,state['pool_router'].numel()*2)),
            'pool':(state['pool_router'],min(late<<20,state['pool_router'].numel()*2),None,0),
            'tail':(unpool,min(tail<<20,unpool.numel()*2),None,0)}
        self.stream=torch.cuda.Stream(device=self.device,priority=0)
        self.events={phase:(torch.cuda.Event(),torch.cuda.Event()) for phase in self.regions}
        self.sink=torch.empty(ctas*128,device=self.device,dtype=torch.int32)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))

    def begin(self,phase):
        a,na,b,nb=self.regions[phase]
        if not na+nb:return
        root=torch.cuda.current_stream(self.device);ready,done=self.events[phase]
        ready.record(root);self.stream.wait_event(ready)
        with torch.cuda.stream(self.stream):
            e=self.lib.p7_prefetch(a.data_ptr(),na,b.data_ptr() if b is not None else None,nb,
                self.sink.data_ptr(),self.spec[1],self.spec[0],self.stream.cuda_stream)
            if e:raise RuntimeError(f'P7 prefetch launch CUDA={e} phase={phase}')
            done.record(self.stream)

    def join(self,phase):
        _,na,_,nb=self.regions[phase]
        if na+nb:torch.cuda.current_stream(self.device).wait_event(self.events[phase][1])



DENSE_CHAIN_CUDA='\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\nnamespace persistent {\n// Weight stream: read once, keep out of L1. LD 0: 256B L2 prefetch, 1: none, 2: 128B.\ntemplate <int LD>\n__device__ __forceinline__ uint4 ld_stream(const void* p) {\n  uint4 r;\n  if (LD == 0)\n    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"\n                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  else if (LD == 1)\n    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"\n                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  else\n    asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"\n                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  return r;\n}\n\n// Activations: shared by the CTA\'s warps through L1.\n__device__ __forceinline__ uint4 ld_act(const void* p) {\n  uint4 r;\n  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  return r;\n}\n\n__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,\n                                         uint32_t a3, uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "\n      "{%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n\n// One CTA: BR = WN*TR*16 weight rows x BT = 8*NT tokens x KB of K (its cluster rank\'s slice).\n// Warp (wn, wk): TR row tiles of 16 x the wk-th 1/WK of the CTA\'s K slice, in steps of 32*U.\n// Lane (g = lane/4, t = lane%4) loads 16 bytes at k = k0 + 8t of rows g and g+8 (and of\n// token g for activations); the first mma takes k-slots {8t..8t+3}, the second {8t+4..8t+7}.\n// NA = 2: a second weight matrix (the router\'s lo half) at the same offsets, same accumulators.\n// XP: activations prefetched one batch ahead with the weights (else loaded at compute time).\ntemplate <int NT, int TR, int U, int NA, int XP, int LD>\n__device__ __forceinline__ void dense_tile(\n    const __nv_bfloat16* __restrict__ W0, const __nv_bfloat16* __restrict__ W1,\n    const __nv_bfloat16* __restrict__ X, void* __restrict__ Out, int M, int N, int K, int KB,\n    int WN, int WK, int pn, int ps, int pt, float* sred) {\n  constexpr bool OUTF32 = true;\n  constexpr int XU = XP ? U : 1;\n  constexpr int XN = XP ? NT : 1;\n  constexpr int BT = 8 * NT;\n  const int BR = WN * TR * 16;\n  const int S = BR + 4;  // padded token stride of the reduction tile (bank spread)\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int g = lane >> 2, t = lane & 3;\n  const int wn = warp % WN, wk = warp / WN;\n  const int row_cta = pn * BR;\n  const int rowbase = row_cta + wn * TR * 16;\n  const int tok0 = pt * BT;\n  const int KW = KB / WK;\n  const int kbeg = ps * KB + wk * KW;\n  const int nsteps = KW / (32 * U);\n\n  float acc[TR][NT][4];\n#pragma unroll\n  for (int r = 0; r < TR; ++r)\n#pragma unroll\n    for (int j = 0; j < NT; ++j)\n#pragma unroll\n      for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.f;\n\n  const size_t off = (size_t)(rowbase + g) * K + kbeg + 8 * t;\n  const __nv_bfloat16* wp0 = W0 + off;\n  const __nv_bfloat16* wp1 = W1 + off;\n  const size_t rs8 = (size_t)8 * K;\n  const size_t rs16 = (size_t)16 * K;\n  const __nv_bfloat16* xp[NT];\n#pragma unroll\n  for (int j = 0; j < NT; ++j) {\n    int tok = tok0 + j * 8 + g;\n    tok = tok < M ? tok : M - 1;  // padded tokens read a real row; their outputs are dropped\n    xp[j] = X + (size_t)tok * K + kbeg + 8 * t;\n  }\n\n  uint4 a0[U][NA][TR][2], a1[U][NA][TR][2];\n  uint4 b0[XU][XN], b1[XU][XN];\n\n  auto load = [&](uint4 (&a)[U][NA][TR][2], uint4 (&b)[XU][XN], int s) {\n    const int koff = s * 32 * U;\n#pragma unroll\n    for (int u = 0; u < U; ++u)\n#pragma unroll\n      for (int na = 0; na < NA; ++na)\n#pragma unroll\n        for (int r = 0; r < TR; ++r) {\n          const __nv_bfloat16* p = (na == 0 ? wp0 : wp1) + r * rs16 + koff + u * 32;\n          a[u][na][r][0] = ld_stream<LD>(p);\n          a[u][na][r][1] = ld_stream<LD>(p + rs8);\n        }\n    if (XP) {\n#pragma unroll\n      for (int u = 0; u < XU; ++u)\n#pragma unroll\n        for (int j = 0; j < XN; ++j) b[u][j] = ld_act(xp[j] + koff + u * 32);\n    }\n  };\n  auto compute = [&](uint4 (&a)[U][NA][TR][2], uint4 (&bp)[XU][XN], int s) {\n    const int koff = s * 32 * U;\n    uint4 bl[U][NT];\n#pragma unroll\n    for (int u = 0; u < U; ++u)\n#pragma unroll\n      for (int j = 0; j < NT; ++j) {\n        if (XP)\n          bl[u][j] = bp[u < XU ? u : 0][j < XN ? j : 0];\n        else\n          bl[u][j] = ld_act(xp[j] + koff + u * 32);\n      }\n#pragma unroll\n    for (int u = 0; u < U; ++u)\n#pragma unroll\n      for (int na = 0; na < NA; ++na)\n#pragma unroll\n        for (int r = 0; r < TR; ++r)\n#pragma unroll\n          for (int j = 0; j < NT; ++j) {\n            const uint4& lo = a[u][na][r][0];\n            const uint4& hi = a[u][na][r][1];\n            mma16816(acc[r][j], lo.x, hi.x, lo.y, hi.y, bl[u][j].x, bl[u][j].y);\n            mma16816(acc[r][j], lo.z, hi.z, lo.w, hi.w, bl[u][j].z, bl[u][j].w);\n          }\n  };\n\n  if (nsteps > 0) load(a0, b0, 0);\n  for (int s = 0; s < nsteps; s += 2) {\n    if (s + 1 < nsteps) load(a1, b1, s + 1);\n    compute(a0, b0, s);\n    if (s + 1 < nsteps) {\n      if (s + 2 < nsteps) load(a0, b0, s + 2);\n      compute(a1, b1, s + 1);\n    }\n  }\n\n  // Warp partials -> shared tile [WK][BT][S] (token-major so the output stores coalesce).\n#pragma unroll\n  for (int r = 0; r < TR; ++r)\n#pragma unroll\n    for (int j = 0; j < NT; ++j) {\n      const int row = wn * TR * 16 + r * 16 + g;\n      const int tok = j * 8 + 2 * t;\n      float* p = sred + (size_t)(wk * BT + tok) * S + row;\n      p[0] = acc[r][j][0];\n      p[S] = acc[r][j][1];\n      p[8] = acc[r][j][2];\n      p[S + 8] = acc[r][j][3];\n    }\n  __syncthreads();\n  const int E = BT * BR;\n  if (WK > 1) {\n    for (int e = threadIdx.x; e < E; e += blockDim.x) {\n      const int tok = e / BR, row = e - tok * BR;\n      float v = 0.f;\n      for (int w = 0; w < WK; ++w) v += sred[(w * BT + tok) * S + row];\n      sred[tok * S + row] = v;\n    }\n    __syncthreads();\n  }\n\n  auto store = [&](int tok, int row, float v) {\n    const size_t idx = ((size_t)ps * M + tok0 + tok) * N + row_cta + row;\n    if (OUTF32)\n      reinterpret_cast<float*>(Out)[idx] = v;\n    else\n      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);\n  };\n\n  for (int e = threadIdx.x; e < E; e += blockDim.x) {\n    const int tok = e / BR, row = e - tok * BR;\n    if (tok0 + tok < M) store(tok, row, sred[tok * S + row]);\n  }\n  __syncthreads();\n}\n\n\n__device__ __forceinline__ void release_dependents(bool pdl) {\n#if __CUDA_ARCH__ >= 900\n  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void wait_predecessor(bool pdl) {\n#if __CUDA_ARCH__ >= 900\n  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");\n#endif\n}\n__device__ __forceinline__ int acquire(const int* p) {\n  int x;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(x) : "l"(p) : "memory");\n  return x;\n}\n// Grid is bounded by measured occupancy. Every thread publishes its global stores before\n// its CTA arrives. Counter pairs are private to this invocation and start at zero.\n__device__ __forceinline__ void grid_barrier(int* cnt, int phase) {\n  __threadfence();\n  __syncthreads();\n  if (threadIdx.x == 0) {\n    const int old = phase;\n    if (atomicAdd(cnt, 1) == gridDim.x - 1) {\n      cnt[0] = 0;\n      __threadfence();\n      atomicExch(cnt + 1, old + 1);\n    } else {\n      while (acquire(cnt + 1) == old) {}\n    }\n  }\n  __syncthreads();\n}\n__device__ __forceinline__ void prefetch_weights(const __nv_bfloat16* w, int bytes) {\n#if __CUDA_ARCH__ == 900\n  if (threadIdx.x == 0) {\n    for (int off = blockIdx.x * 4096; off < bytes; off += gridDim.x * 4096) {\n      const int size = min(4096, bytes - off);\n      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"((const char*)w + off), "r"(size) : "memory");\n    }\n  }\n#else\n  for (int off = (blockIdx.x * blockDim.x + threadIdx.x) * 128; off < bytes;\n       off += gridDim.x * blockDim.x * 128)\n    asm volatile("prefetch.global.L2 [%0];" :: "l"((const char*)w + off) : "memory");\n#endif\n}\n__device__ __forceinline__ float sum_block(float v, float* tmp) {\n  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(~0u, v, d);\n  if ((threadIdx.x & 31) == 0) tmp[threadIdx.x >> 5] = v;\n  __syncthreads();\n  v = threadIdx.x < 8 ? tmp[threadIdx.x] : 0.f;\n  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(~0u, v, d);\n  if (threadIdx.x == 0) tmp[8] = v;\n  __syncthreads();\n  return tmp[8];\n}\n\n// Small batches distribute one residual row across several resident CTAs. Four\n// sufficient statistics give both norms with one token-local synchronization.\ntemplate<int C>\n__device__ __forceinline__ void norm_chunks(const float* p,int splits,int m,int n,\n    const float* stream,float* stream_out,__nv_bfloat16* hidden,const float* alpha,\n    const float* beta,const float* prew,const float* postw,int tok,int chunk,\n    float* tmp,int* counts,float* sums) {\n  constexpr int J=32/C;\n  float b[J],as[J],bb[J],v[4]={0.f,0.f,0.f,0.f};\n#pragma unroll\n  for(int j=0;j<J;++j) {\n    const int d=chunk*(8192/C)+threadIdx.x+j*256;\n    float z=0.f;\n    if(d<n) for(int k=0;k<splits;++k) z+=p[((size_t)k*m+tok)*n+d];\n    z=__bfloat162float(__float2bfloat16_rn(z));\n    if(postw)z*=postw[d];\n    b[j]=z;as[j]=alpha[d]*stream[(size_t)tok*8192+d];bb[j]=beta[d]*z;\n    v[0]+=z*z;v[1]+=as[j]*as[j];v[2]+=as[j]*bb[j];v[3]+=bb[j]*bb[j];\n  }\n#pragma unroll\n  for(int k=0;k<4;++k) {\n    for(int d=16;d;d>>=1)v[k]+=__shfl_xor_sync(~0u,v[k],d);\n    if((threadIdx.x&31)==0)tmp[k*8+(threadIdx.x>>5)]=v[k];\n  }\n  __syncthreads();\n  if(threadIdx.x<32) {\n#pragma unroll\n    for(int k=0;k<4;++k) {\n      float z=threadIdx.x<8?tmp[k*8+threadIdx.x]:0.f;\n      for(int d=16;d;d>>=1)z+=__shfl_xor_sync(~0u,z,d);\n      if(threadIdx.x==0)atomicAdd(sums+tok*4+k,z);\n    }\n  }\n  __threadfence();__syncthreads();\n  if(threadIdx.x==0) {\n    atomicAdd(counts+tok,1);\n    while(acquire(counts+tok)<C){}\n  }\n  __syncthreads();\n  const volatile float* f=sums+tok*4;\n  const float rb=rsqrtf(f[0]*(1.f/8192)+1e-8f);\n  const float rh=rsqrtf((f[1]+2.f*rb*f[2]+rb*rb*f[3])*(1.f/8192)+1e-5f);\n#pragma unroll\n  for(int j=0;j<J;++j) {\n    const int d=chunk*(8192/C)+threadIdx.x+j*256;\n    const float st=as[j]+bb[j]*rb;\n    stream_out[(size_t)tok*8192+d]=st;\n    hidden[(size_t)tok*8192+d]=__float2bfloat16_rn(st*rh*prew[d]);\n  }\n  __syncthreads();\n}\n\n// Pooler norm and router are independent consumers of the same concatenated dense product.\n__device__ __forceinline__ void route_norm(const float* pr, int m, int tok, const float* bias,\n    const float* prew, const int* local_start, __nv_bfloat16* x, int* ids, float* wts,\n    int* lids, float* tmp) {\n  const int tid = threadIdx.x;\n  const float* row = pr + (size_t)tok * 4608;\n  float v[16], ss = 0.f;\n#pragma unroll\n  for (int j = 0; j < 16; ++j) {\n    const int d = tid + j * 256;\n    v[j] = __bfloat162float(__float2bfloat16_rn(row[d]));\n    ss += v[j] * v[j];\n  }\n  const float r = rsqrtf(sum_block(ss, tmp) * (1.f / 4096) + 1e-5f);\n#pragma unroll\n  for (int j = 0; j < 16; ++j) {\n    const int d = tid + j * 256;\n    x[(size_t)tok * 4096 + d] = __float2bfloat16_rn(v[j] * r * prew[d]);\n  }\n  if (tid < 32) {\n    float scores[8], biased[8], selected[8];\n#pragma unroll\n    for (int j=0;j<8;++j) {\n      const int e=tid+j*32;\n      scores[j]=1.f/(1.f+__expf(-(row[4096+e]+row[4352+e])*0.1767766952966369f));\n      biased[j]=scores[j]+bias[e];\n    }\n    float total=0.f;\n#pragma unroll\n    for (int slot=0;slot<8;++slot) {\n      float best=-INFINITY; int idx=256;\n#pragma unroll\n      for(int j=0;j<8;++j) {\n        const int e=tid+j*32;\n        if(biased[j]>best || (biased[j]==best && e<idx)) { best=biased[j];idx=e; }\n      }\n#pragma unroll\n      for(int d=16;d;d>>=1) {\n        const float v=__shfl_xor_sync(~0u,best,d);\n        const int i=__shfl_xor_sync(~0u,idx,d);\n        if(v>best || (v==best && i<idx)) {best=v;idx=i;}\n      }\n      float chosen=0.f;\n#pragma unroll\n      for(int j=0;j<8;++j) if(tid+j*32==idx) { chosen=scores[j];biased[j]=-INFINITY; }\n      chosen=__shfl_sync(~0u,chosen,idx&31);\n      selected[slot]=chosen;total+=chosen;\n      if(tid==0) {\n        ids[tok*8+slot]=idx;\n        const int local=(idx-*local_start+256)%256;\n        lids[tok*8+slot]=local<32?local:-1;\n      }\n    }\n    if(tid==0) for(int j=0;j<8;++j) wts[tok*8+j]=selected[j]/(total+1e-15f);\n  }\n  __syncthreads();\n}\n\nstruct Params {\n  const __nv_bfloat16 *input, *o, *pr, *unpool;\n  const float *stream, *alpha, *beta, *prew, *postw, *bias, *expertw;\n  const int* local_start;\n  float *partial, *pr_out, *stream_out;\n  __nv_bfloat16 *h, *x;\n  int *ids, *lids, *cnt;\n  float *wts;\n  int m, n, splits, prefetch;\n  bool pdl;\n  bool route;\n  uint64_t* stamps;\n};\n\n__device__ __forceinline__ void mark(const Params& p, int phase) {\n  if (p.stamps && blockIdx.x==0 && threadIdx.x==0) {\n    uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n    p.stamps[phase]=t;\n  }\n}\n\ntemplate <int NT, bool PREFIX>\n__global__ void __launch_bounds__(256, 1) dense_island(Params p) {\n  extern __shared__ float sh[];\n  const int blocks = gridDim.x, cta = blockIdx.x;\n  constexpr int BT = 8 * NT;\n  release_dependents(p.pdl);\n  if (p.prefetch) prefetch_weights(PREFIX ? p.o : p.unpool, p.prefetch);\n  wait_predecessor(p.pdl);\n  mark(p,0);\n  const int n = PREFIX ? 8192 : p.n, k = PREFIX ? 1024 : 4096;\n  const int splits = PREFIX ? 1 : p.splits;\n  const int rows = n / 16;\n  const int groups = (p.m + BT - 1) / BT;\n  for (int unit = cta; unit < rows * splits * groups; unit += blocks) {\n    const int pn = unit % rows, ps = (unit / rows) % splits, pt = unit / (rows * splits);\n    dense_tile<NT,1,4,1,1,0>(PREFIX ? p.o : p.unpool, PREFIX ? p.o : p.unpool,\n        p.input, p.partial, p.m, n, k, k/splits, 1, 8, pn, ps, pt, sh);\n  }\n  // Stream the dependent pool/router rows while the last O tiles drain.\n  if (PREFIX && p.prefetch) prefetch_weights(p.pr, p.prefetch);\n  grid_barrier(p.cnt, 0);\n  mark(p,1);\n  constexpr int C=16/NT;\n  for(int unit=cta;unit<p.m*C;unit+=blocks)\n    norm_chunks<C>(p.partial,splits,p.m,n,p.stream,p.stream_out,p.h,p.alpha,p.beta,p.prew,\n      PREFIX?nullptr:p.postw,unit/C,unit%C,sh,p.cnt+2,reinterpret_cast<float*>(p.cnt+2+p.m));\n  if (PREFIX) {\n    grid_barrier(p.cnt, 1);\n    mark(p,2);\n    for (int unit = cta; unit < 288 * groups; unit += blocks)\n      dense_tile<NT,1,4,1,1,0>(p.pr, p.pr, p.h, p.pr_out, p.m, 4608, 8192, 8192,\n                              1, 8, unit % 288, 0, unit / 288, sh);\n    grid_barrier(p.cnt, 2);\n    mark(p,3);\n    if (p.route) {\n      for (int t = cta; t < p.m; t += blocks)\n        route_norm(p.pr_out, p.m, t, p.bias, p.expertw, p.local_start, p.x, p.ids, p.wts, p.lids, sh);\n    }\n  }\n  mark(p,4);\n}\n\ntemplate <int NT, bool PREFIX>\nvoid launch(Params p, int grid) {\n  constexpr size_t smem = 8 * (8 * NT) * 20 * sizeof(float);\n  auto kernel = dense_island<NT, PREFIX>;\n  int device, active, sms;\n  C10_CUDA_CHECK(cudaGetDevice(&device));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 256, smem));\n  TORCH_CHECK(active > 0, "persistent dense island cannot reside");\n  grid = std::min(grid, sms * active);\n\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = smem;\n  cfg.stream = at::cuda::getCurrentCUDAStream();\n  cudaLaunchAttribute attr{};\n  if (p.pdl) {\n    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;\n    attr.val.programmaticStreamSerializationAllowed = 1;\n    cfg.attrs = &attr; cfg.numAttrs = 1;\n  }\n  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p));\n}\n} // namespace persistent\n\nvoid island(std::vector<torch::Tensor> a, int64_t m, int64_t n, int64_t splits, int64_t nt,\n            int64_t grid, int64_t prefetch, bool prefix, bool pdl, bool route) {\n  using namespace persistent;\n  Params p{};\n#define BF(I) reinterpret_cast<const __nv_bfloat16*>(a[I].data_ptr())\n#define FP(I) a[I].data_ptr<float>()\n#define IP(I) a[I].data_ptr<int>()\n  p.input=BF(0); p.o=BF(1); p.pr=BF(2); p.unpool=BF(3);\n  p.stream=FP(4); p.alpha=FP(5); p.beta=FP(6); p.prew=FP(7); p.postw=FP(8);\n  p.bias=FP(9); p.expertw=FP(10); p.local_start=a[11].data_ptr<int>();\n  p.partial=FP(12); p.pr_out=FP(13); p.stream_out=FP(14);\n  p.h=reinterpret_cast<__nv_bfloat16*>(a[15].data_ptr());\n  p.x=reinterpret_cast<__nv_bfloat16*>(a[16].data_ptr());\n  p.ids=IP(17); p.lids=IP(18); p.cnt=IP(19); p.wts=FP(20);\n  p.stamps=a.size()>21 ? reinterpret_cast<uint64_t*>(a[21].data_ptr()) : nullptr;\n  p.m=m; p.n=n; p.splits=splits; p.prefetch=prefetch; p.pdl=pdl; p.route=route;\n#define CASE(NT, PRE) if (nt==NT && prefix==PRE) { launch<NT,PRE>(p, grid); return; }\n  CASE(1,true); CASE(2,true); CASE(4,true); CASE(1,false); CASE(2,false); CASE(4,false);\n  TORCH_CHECK(false, "unsupported persistent tile");\n}\n'
DENSE_CHAIN_CPP = r'''
#include <torch/extension.h>
void island(std::vector<torch::Tensor> a, int64_t m, int64_t n, int64_t splits, int64_t nt,
            int64_t grid, int64_t prefetch, bool prefix, bool pdl, bool route);
'''
_DENSE_CHAIN_MOD = None


def _dense_chain_module():
    global _DENSE_CHAIN_MOD
    if _DENSE_CHAIN_MOD is None:
        import hashlib
        tag = hashlib.sha256(DENSE_CHAIN_CUDA.encode()).hexdigest()[:12]
        _DENSE_CHAIN_MOD = load_inline(name='muse_dense_chain_' + tag,
            cpp_sources=DENSE_CHAIN_CPP, cuda_sources=DENSE_CHAIN_CUDA,
            functions=['island'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
    return _DENSE_CHAIN_MOD


def _dense_chain_prepare(state, weights, device):
    state['dense_chain_mod'] = _dense_chain_module()
    state['dense_chain_grid'] = torch.cuda.get_device_properties(device).multi_processor_count


def _dense_chain_call(attn, stream, state, data):
    t = attn.shape[0]; dev = attn.device
    partial = torch.empty((1, t, 8192), device=dev, dtype=torch.float32)
    pr_out = torch.empty((t, 4608), device=dev, dtype=torch.float32)
    stream_out = torch.empty_like(stream)
    hidden = torch.empty((t, 8192), device=dev, dtype=torch.bfloat16)
    zbf = torch.empty((t, 4096), device=dev, dtype=torch.bfloat16)
    zi = torch.empty((t, 8), device=dev, dtype=torch.int32)
    zf = torch.empty((t, 8), device=dev, dtype=torch.float32)
    cnt = torch.empty((2 + 5 * t,), device=dev, dtype=torch.int32)
    local = data['local_start']
    args = [attn, state['o'], state['pool_router'], state['pool_router'], stream,
            state['alpha1'], state['beta1'], state['pre_ffn_w'], state['post_ffn_w'],
            state['router_bias'], state['pre_expert_w'], local, partial, pr_out,
            stream_out, hidden, zbf, zi, zi, cnt, zf]
    cnt.zero_()
    prefetch = min(16 << 20, state['pool_router'].numel() * 2)
    state['dense_chain_mod'].island(args, t, 4608, 1, 4, state['dense_chain_grid'],
                                    prefetch, True, False, False)
    return hidden, stream_out, pr_out



import functools


import hashlib


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



SMALLM_FLAGS = {1: 95, 8: 69, 16: 69, 32: 69}
SMALLM_VAR_CUDA = '\n#define C10_CUDA_CHECK(expr) do{cudaError_t e=(expr);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)\n#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())\n\n#include <cuda_runtime.h>\n#include <stdexcept>\n#include <vector>\n#include <algorithm>\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  uint32_t W; asm("lop3.b32 %0,%1,%2,%3,0xe4;":"=r"(W):"r"(A),"r"(B),"n"(LA_MASK));\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n  int erank[8];  // F_SLOT: local expert of each local rank\n  int P;         // F_SLOT: local pairs\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\n// ---- flag-templated variants of the admitted one-LOP3 NT1 kernel (F=0 is the incumbent\'s math) ----\nenum {\n  F_Z = 1,      // zero each pair\'s norm slot in its W13 tile 0 (before expert release); no C_ZERO wait\n  F_SLOT = 2,   // M=1: dispatch straight from the 8 top-k slots (rank = popcount), no counts/scan/lists/search\n  F_ZOUT = 4,   // W2 epilogue stores z = v(1+w); the finalizer drops its post-norm weight loads\n  F_RSQ = 8,    // finalizer: lanes 0-7 compute each pair\'s scale once and shuffle it\n  F_NREG = 16,  // W2 norm partials stay in registers across a CTA\'s run of tiles for one (expert, block)\n  F_PDL = 64,   // launched as a programmatic dependent: setup overlaps the router, griddepcontrol.wait first\n  F_LAST = 128, // the last CTA to finish W2 runs the whole finalizer; no grid-wide release/spin\n  F_TRACE = 256, // globaltimer stamps per CTA (layout of the earlier TRACE build), written to v[13]\n  F_NT2 = 2048  // 16-token blocks: each weight tile is decoded once for up to 16 of an expert\'s tokens; 1 CTA/SM\n};\ntemplate <int Fv> constexpr int nt_of = (Fv & F_NT2) ? 2 : 1;\nconstexpr int NT2_PROBE = F_NT2 | 69;   // instantiated NT2 word that resident_blocks(2, ...) sizes the grid with\n__device__ __forceinline__ uint64_t tg() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory"); return t; }\n__device__ __forceinline__ uint64_t tclk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)::"memory"); return t; }\n\n__device__ __forceinline__ void prefetch_slot(Route* rt, int total, int first, int a1, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb, const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  const int idx = p.index;\n  if (idx >= total) return;\n  const int st = p.stage, ph = p.phase;\n  const bool pass = idx >= first;\n  const int u = pass ? a2 + idx - first : a1 + idx;\n  const int r = pass ? (u >> 8) : (u >> 9), m = pass ? (u & (NT2 - 1)) : (u & (NT1 - 1));\n  const int e = rt->erank[r];\n  const uint8_t* src = (pass ? w2p : w13p) + (static_cast<size_t>(e) * (pass ? NT2 : NT1) + m) * TILE_BYTES;\n  mbar_wait(&emptyb[st], ph ^ 1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages + st * TILE_BYTES, src, TILE_BYTES, &fullb[st], p.pol_first);\n  p.index = idx + 1;\n  if (st + 1 == nstage) { p.stage = 0; p.phase = ph ^ 1; } else p.stage = st + 1;\n}\n\ntemplate <int NT, int F>\n__global__ void __launch_bounds__(NTHR, NT == 1 ? 2 : 1) moe_var(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage, uint64_t* __restrict__ trace) {\n  constexpr bool Z = F & F_Z, SLOT = F & F_SLOT, ZOUT = F & F_ZOUT, RSQ = F & F_RSQ, NREG = F & F_NREG;\n  constexpr bool PDL = F & F_PDL, LAST = F & F_LAST, TRACE = F & F_TRACE;\n  static_assert(((F & F_NT2) != 0) == (NT == 2) && !(NT == 2 && SLOT), "F_NT2 selects NT=2 (M>1 dispatch)");\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n  __shared__ int last_cta;\n  // F_TRACE layout: [0] entry [1] routed [2] main loop [3] W13 end [4] first W2 ready [5] W2 end\n  // [6] barrier [7] combine [8] end [9] exit [10] first copy issue [11,12] first W13 copy wait\n  // [13] first W13 tile done [14,15] first W2 copy wait [16] sum of ready waits\n  // [20,21] clock64 entry/exit [23] finalizer [24] smid [25] grid [26,27] W13/W2 units [32+w] warp entry [40+w] warp exit\n  __shared__ uint64_t probe[TRACE ? 64 : 1];\n  if constexpr (TRACE) {\n    if ((tid & 31) == 0) probe[32 + warp] = tg();\n    if (tid == 0) { probe[0] = probe[32]; probe[20] = tclk(); }\n    if (tid < 64 && tid != 0 && tid != 20 && !(tid >= 32 && tid < 40)) probe[tid] = 0;\n  }\n  auto dump = [&](bool finalizer) {\n    if constexpr (TRACE) {\n      if (tid == 0) { probe[8] = tg(); probe[23] = finalizer; if (!probe[4]) probe[4] = probe[3]; }\n      bar_consumers();\n      if (tid < 64) trace[(size_t)c * 64 + tid] = probe[tid];\n      bar_consumers();\n      if (lane == 0) trace[(size_t)c * 64 + 40 + warp] = tg();\n      if (tid == 0) { trace[(size_t)c * 64 + 21] = tclk(); trace[(size_t)c * 64 + 9] = tg(); }\n    }\n  };\n\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    uint64_t pf, pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first = pf; rt->pre.pol_normal = pn;\n    rt->pre.index = 0; rt->pre.stage = 0; rt->pre.phase = 0;\n  }\n  if constexpr (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");\n\n  int U1, U2, a1, b1, a2, b2, first_count, total_count;\n  if constexpr (SLOT) {\n    // M == 1: a local expert holds at most one pair. Local slots in slot order: rank = popcount.\n    if (warp == 0) {\n      int e = -1;\n      if (lane < 8) {\n        e = __ldg(ids + lane) - local_start;\n        if (e < 0 || e >= NLOC) e = -1;\n      }\n      if (!LAST && lane == 0) bar_gen0 = ld_acquire(cnt + C_GEN);\n      const unsigned mk = __ballot_sync(~0u, e >= 0);\n      const int P = __popc(mk), r = __popc(mk & ((1u << lane) - 1u));\n      if (lane < 8) pos_of[lane] = static_cast<int16_t>(e >= 0 ? r : -1);\n      if (e >= 0) { pair_of[r] = static_cast<int16_t>(lane); rt->erank[r] = e; }\n      __syncwarp();   // lane 0\'s first copies read erank\n      U1 = P * NT1; U2 = P * NT2;\n      a1 = static_cast<int>((long long)U1 * c / G); b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n      a2 = static_cast<int>((long long)U2 * c / G); b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n      first_count = b1 - a1; total_count = first_count + b2 - a2;\n      if (lane == 0) {\n        rt->P = P;\n        if constexpr (TRACE) probe[10] = tg();\n        for (int i = 0; i < nstage; ++i)\n          prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);\n      }\n    }\n    __syncthreads();\n    if (warp != 0) {\n      const int P = rt->P;\n      U1 = P * NT1; U2 = P * NT2;\n      a1 = static_cast<int>((long long)U1 * c / G); b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n      a2 = static_cast<int>((long long)U2 * c / G); b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n      first_count = b1 - a1; total_count = first_count + b2 - a2;\n    }\n  } else {\n    if (tid < NLOC) rt->n[tid] = 0;\n    if (!LAST && tid == 0) bar_gen0 = ld_acquire(cnt + C_GEN);\n    __syncthreads();\n    for (int i = tid; i < npairs; i += NTHR) {\n      int e = __ldg(ids + i) - local_start;\n      if (e < 0 || e >= NLOC) e = -1;\n      eflat[i] = static_cast<int8_t>(e);\n      if (e >= 0) atomicAdd(&rt->n[e], 1);\n    }\n    __syncthreads();\n    if (warp == 0) {\n      const int n = rt->n[lane];\n      const int nb = (n + NB - 1) / NB;\n      int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n      for (int d = 1; d < 32; d <<= 1) {\n        const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n        if (lane >= d) { off += a; u1 += b; u2 += e2; }\n      }\n      rt->off[lane] = off - n;\n      rt->run[lane] = off - n;\n      rt->nblk[lane] = nb;\n      rt->us1[lane + 1] = u1;\n      rt->us2[lane + 1] = u2;\n      if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n      const int T1 = __shfl_sync(~0u, u1, 31), T2 = __shfl_sync(~0u, u2, 31);\n      if (lane == 0) {\n        rt->c1 = min(NT1, max(1, (T1 + G / 2) / G));\n        rt->c2 = min(NT2, max(1, (T2 + G / 2) / G));\n      }\n    }\n    __syncthreads();\n    U1 = rt->us1[NLOC]; U2 = rt->us2[NLOC];\n    a1 = static_cast<int>((long long)U1 * c / G); b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n    a2 = static_cast<int>((long long)U2 * c / G); b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n    first_count = b1 - a1; total_count = first_count + b2 - a2;\n    if (tid == 0) {\n      if (total_count) {\n        int e, nb, m;\n        const int pass = first_count ? 0 : 1;\n        const int cc = pass ? rt->c2 : rt->c1, ntile = pass ? NT2 : NT1;\n        decode_unit(pass ? a2 : a1, pass ? rt->us2 : rt->us1, rt->nblk, cc, ntile, e, nb, m);\n        rt->pre.pass = pass; rt->pre.e = e; rt->pre.nb = nb; rt->pre.m = m;\n        rt->pre.mb = (m / cc) * cc; rt->pre.me = min(ntile, (m / cc) * cc + cc);\n      }\n      if constexpr (TRACE) probe[10] = tg();\n      for (int i = 0; i < nstage; ++i)\n        prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);\n    }\n    __syncthreads();\n    // deterministic token lists (sorted by local expert, then flat slot)\n    for (int base = 0; base < npairs; base += NCT) {\n      rt->wc[warp][lane] = 0;\n      bar_consumers();\n      const int i = base + tid;\n      const int e = i < npairs ? eflat[i] : -1;\n      const unsigned mk = __match_any_sync(~0u, e);\n      const int rank = __popc(mk & ((1u << lane) - 1u));\n      if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n      bar_consumers();\n      if (tid < NLOC) {\n        int s = rt->run[tid];\n        for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n        rt->run[tid] = s;\n      }\n      bar_consumers();\n      if (i < npairs) {\n        if (e >= 0) {\n          const int pos = rt->wc[warp][e] + rank;\n          pos_of[i] = static_cast<int16_t>(pos);\n          pair_of[pos] = static_cast<int16_t>(i);\n        } else {\n          pos_of[i] = -1;\n        }\n      }\n    }\n  }\n  if constexpr (TRACE) {\n    if (tid == 0) {\n      probe[1] = tg(); uint32_t sm; asm("mov.u32 %0, %%smid;" : "=r"(sm)); probe[24] = sm; probe[25] = G;\n      probe[26] = b1 - a1; probe[27] = b2 - a2;\n    }\n  }\n  if constexpr (!Z) {\n    // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n    const int P = SLOT ? rt->P : (U1 > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0);\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  if constexpr (TRACE) { if (tid == 0) probe[2] = tg(); }\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n  float nacc = 0.f;  // F_NREG: this thread\'s squared sum over the current (expert, block) run\n  int npos = -1;\n  auto flush_norm = [&]() {\n    if (tid < 16 * NB) {\n      float t2 = nacc;\n#pragma unroll\n      for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);\n      if (npos >= 0 && (tid & 15) == 0) atomicAdd(ss + npos, t2);\n    }\n    nacc = 0.f; npos = -1;\n  };\n\n  auto unit_body = [&](const int pass, const bool firstp, const int e, const int nb, const int m, const int ne,\n                       const int off) {\n    if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n      if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n        __threadfence();\n        bar_consumers();\n        if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n      }\n      if constexpr (NREG) { if (cur_pass == 1) flush_norm(); }\n      cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n      if (pass == 1) {\n        uint64_t wait0 = 0;\n        if constexpr (TRACE) { if (tid == 0) wait0 = tg(); }\n        const int target = NT1 * (SLOT ? 1 : rt->nblk[e]);\n        while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n        if constexpr (!Z) { while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64); }\n        if constexpr (TRACE) { if (tid == 0) probe[16] += tg() - wait0; }\n      }\n#pragma unroll\n      for (int j = 0; j < NT; ++j) {\n        const int idx = nb * NB + j * 8 + (lane >> 2);\n        const __nv_bfloat16* row = nullptr;\n        if (idx < ne) {\n          const int pos = off + idx;\n          row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n        }\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n          if (row) {\n            const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n            if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n            else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n          }\n          Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n          Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n        }\n      }\n    }\n    if constexpr (TRACE) { if (tid == 0 && pass == 1 && firstp) probe[4] = tg(); }\n    if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n    if constexpr (TRACE) { if (tid == 0 && firstp) probe[pass ? 14 : 11] = tg(); }\n    mbar_wait(&fullb[st], ph);\n    if constexpr (TRACE) { if (tid == 0 && firstp) probe[pass ? 15 : 12] = tg(); }\n    const uint8_t* tile = stages + st * TILE_BYTES;\n    float acc[NT][4];\n#pragma unroll\n    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n    {\n      const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n      const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n      const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n      for (int g = 0; g < 8; ++g) {\n        const uint4 wv = wq[g * 32];\n        const uint32_t sw = scw[g >> 1];\n        const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n        const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n        uint32_t o[16];\n        dq8(wv.x, wv.y, o);\n        dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n        for (int s = 0; s < 4; ++s) {\n          const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n          const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n          const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n          for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n        }\n      }\n    }\n    __syncwarp();\n    if (lane == 0) mbar_arrive(&emptyb[st]);\n    if (++st == nstage) { st = 0; ph ^= 1u; }\n\n    float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n    for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n    bar_consumers();\n    if (tid == 0) {\n      if constexpr (SLOT) prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);\n      else prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);\n    }\n    const float* rb = red + buf * NCW * 32 * RED;\n    if (pass == 0) {\n      const float sc = gs13[e] * 256.f;\n      if (tid < 8 * NB) {\n        const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n        const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n        float G1 = 0.f, U = 0.f;\n#pragma unroll\n        for (int w = 0; w < NCW; ++w) {\n          G1 += rb[(w * 32 + ls) * RED + rg];\n          U += rb[(w * 32 + ls) * RED + rg + 2];\n        }\n        const int idx = nb * NB + n;\n        if (idx < ne) {\n          if constexpr (Z) { if (m == 0 && i == 0) ss[off + idx] = 0.f; }\n          G1 *= sc;\n          U *= sc;\n          const float h = G1 / (1.f + __expf(-G1)) * U;\n          inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n        }\n      }\n    } else {\n      const float sc = gs2[e] * 256.f;\n      if (tid < 16 * NB) {\n        const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n        const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n        float v = 0.f;\n#pragma unroll\n        for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n        v *= sc;\n        const int idx = nb * NB + n;\n        const bool ok = idx < ne;\n        const float z = v * fw[0];\n        if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = ZOUT ? z : v;\n        if constexpr (NREG) {\n          if (ok) { nacc += z * z; npos = off + idx; }\n        } else {\n          float t2 = ok ? z * z : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n    }\n    if constexpr (TRACE) {\n      if (tid == 0 && pass == 0 && firstp) probe[13] = tg();\n    }\n    buf ^= 1;\n    ++seg_tiles;\n  };\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e = 0, nb = 0, m = 0, mb = 0, me = 0;\n    if constexpr (!SLOT) {\n      if (ua < ub) {\n        decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n        mb = (m / cc) * cc;\n        me = min(ntile, mb + cc);\n      }\n    }\n    for (int u0 = ua; u0 < ub; ++u0) {\n      const int u = u0;\n      int ne, off;\n      if constexpr (SLOT) {\n        const int r = pass ? (u >> 8) : (u >> 9);\n        m = u & (ntile - 1); e = rt->erank[r]; nb = 0; ne = 1; off = r;\n      } else {\n        ne = rt->n[e]; off = rt->off[e];\n      }\n      unit_body(pass, u0 == ua, e, nb, m, ne, off);\n      if constexpr (!SLOT) advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n    if constexpr (TRACE) { if (tid == 0) probe[pass ? 5 : 3] = tg(); }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n  if constexpr (NREG) { if (cur_pass == 1) flush_norm(); }\n\n  // ---------------- grid barrier (or last-arriver election) ----------------\n  __threadfence();\n  bar_consumers();\n  if constexpr (TRACE) { if (tid == 0) probe[6] = tg(); }\n  if constexpr (LAST) {\n    if (tid == 0) {\n      const bool last = atomicAdd(cnt + C_BAR, 1) == G - 1;\n      if (last) {\n        __threadfence();\n        for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n        cnt[C_ZERO] = 0;\n        cnt[C_BAR] = 0;\n      }\n      last_cta = last;\n    }\n    bar_consumers();\n    if (!last_cta) { dump(false); return; }\n  } else {\n    if (tid == 0) {\n      if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n        for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n        cnt[C_ZERO] = 0;\n        cnt[C_BAR] = 0;\n        __threadfence();\n        atomicAdd(cnt + C_GEN, 1);\n      } else {\n        while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n      }\n    }\n    bar_consumers();\n  }\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  if constexpr (TRACE) { if (tid == 0) probe[7] = tg(); }\n  for (int it = LAST ? 0 : c; it < 2 * M; it += LAST ? 1 : G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    if constexpr (!ZOUT) {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float scl = 0.f;\n    int posl = -1;\n    if constexpr (RSQ) {\n      if (lane < 8) {\n        posl = pos_of[t * 8 + lane];\n        if (posl >= 0) scl = tw[t * 8 + lane] * rsqrtf(__ldcg(ss + posl) * (1.f / HD) + 1e-8f);\n      }\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      int pos;\n      float sc;\n      if constexpr (RSQ) {\n        pos = __shfl_sync(~0u, posl, j);\n        sc = __shfl_sync(~0u, scl, j);\n      } else {\n        pos = pos_of[t * 8 + j];\n        sc = pos >= 0 ? tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f) : 0.f;\n      }\n      if (pos >= 0) {\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        if constexpr (ZOUT) {\n          acc[0] += sc * u0.x; acc[1] += sc * u0.y; acc[2] += sc * u0.z; acc[3] += sc * u0.w;\n          acc[4] += sc * u1.x; acc[5] += sc * u1.y; acc[6] += sc * u1.z; acc[7] += sc * u1.w;\n        } else {\n          acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n          acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n        }\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n  dump(LAST || c < 2 * M);\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n#define VAR_FLAGS(X) X(69) X(95)\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  int dyn = optin;\n#define SET_ATTR(Fv) { cudaFuncAttributes a = {}; C10_CUDA_CHECK(cudaFuncGetAttributes(&a, moe_var<nt_of<Fv>, Fv>)); \\\n    C10_CUDA_CHECK(cudaFuncSetAttribute(moe_var<nt_of<Fv>, Fv>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)a.sharedSizeBytes)); \\\n    dyn = std::min(dyn, optin - (int)a.sharedSizeBytes); }\n  VAR_FLAGS(SET_ATTR)\n  SET_ATTR(0)  // resident_blocks() sizes NT=1 grids with moe_var<1, 0> (same smem/threads as every NT=1 word)\n#undef SET_ATTR\n  return dyn;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  if (nt == 2) C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_var<2, NT2_PROBE>, NTHR, smem_bytes(2, nstage, M)));\n  else C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_var<1, 0>, NTHR, smem_bytes(1, nstage, M)));\n  return n;\n}\n\ntemplate <int Fv>\nvoid launch_var(std::vector<int64_t>& v, int M, int start, int stages, int grid, cudaStream_t stream) {\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(NTHR); cfg.dynamicSmemBytes = smem_bytes(nt_of<Fv>, stages, M);\n  cfg.stream = stream;\n  cudaLaunchAttribute attr{};\n  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;\n  attr.val.programmaticStreamSerializationAllowed = 1;\n  if (Fv & F_PDL) { cfg.attrs = &attr; cfg.numAttrs = 1; }\n  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_var<nt_of<Fv>, Fv>,\n    (const __nv_bfloat16*)v[0], (const int*)v[1], (const float*)v[2], (const __nv_bfloat16*)v[3],\n    (const uint8_t*)v[4], (const uint8_t*)v[5], (const float*)v[6], (const float*)v[7],\n    (__nv_bfloat16*)v[8], (float*)v[9], (float*)v[10], (int*)v[11], (__nv_bfloat16*)v[12], M, start, stages,\n    (uint64_t*)(v.size() > 13 ? v[13] : 0)));\n}\n\nvoid run(std::vector<int64_t> v, int64_t M, int64_t start, int64_t stages, int64_t grid, int64_t stream, int64_t flags) {\n  if ((flags & F_SLOT) && M != 1) throw std::runtime_error("F_SLOT requires M == 1");\n  if ((flags & F_TRACE) && (v.size() < 14 || !v[13])) throw std::runtime_error("F_TRACE requires a stamp buffer");\n  switch (flags) {\n#define CASE(Fv) case Fv: launch_var<Fv>(v, M, start, stages, grid, (cudaStream_t)stream); return;\n    VAR_FLAGS(CASE)\n#undef CASE\n  }\n  throw std::runtime_error("flag combination not instantiated");\n}\n'
SMALLM_VAR_CPP = """#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <vector>
int64_t max_smem_attr();int64_t smem_needed(int64_t,int64_t,int64_t);int64_t resident_blocks(int64_t,int64_t,int64_t);
void run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
"""
SMALLM_FLAGS_BUILT = [69, 95]
_SMALLM_MODULE = []


def smallm_module():
    if not _SMALLM_MODULE:
        tag = hashlib.sha256(SMALLM_VAR_CUDA.encode()).hexdigest()[:12]
        target = '-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability() == (9, 0) else '-gencode=arch=compute_120,code=sm_120'
        mod = load_inline(name='muse_smallm_var_' + tag, cpp_sources=SMALLM_VAR_CPP, cuda_sources=SMALLM_VAR_CUDA,
            functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], no_implicit_headers=True,
            with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', target], verbose=False)
        mod.smem_optin = mod.max_smem_attr()
        _SMALLM_MODULE.append(mod)
    return _SMALLM_MODULE[0]


def smallm_bind(state, flags_by_m):
    """Prepare-time: compile and give every flags word its own zeroed counter block (64 int32)."""
    mod = smallm_module(); dev = state['gs13'].device
    for m, flags in flags_by_m.items():
        if flags not in SMALLM_FLAGS_BUILT: raise ValueError(f'flags {flags} not instantiated')
        if flags & 2 and m != 1: raise ValueError('SLOT is the M=1 dispatch')
        state.setdefault('smallm_counters', {}).setdefault(flags, torch.zeros(64, device=dev, dtype=torch.int32))
    return state


def smallm_experts(data, state, flags, stages=2, grid=264):
    x = data['hidden_states'].contiguous(); m = x.shape[0]; p = m * 8; dev = x.device
    ids = data['topk_ids'].int().contiguous(); tw = data['topk_weights'].float().contiguous()
    mod = smallm_module()
    if mod.smem_needed(1, stages, m) > mod.smem_optin: raise ValueError('shared memory exceeds device capacity')
    grid = min(grid, state['sms'] * mod.resident_blocks(1, stages, m))
    bank = state['banks']['lop3']
    inter = torch.empty((p, 4096), device=dev, dtype=torch.bfloat16)
    y = torch.empty((p, 4096), device=dev); ss = torch.empty((p,), device=dev); out = torch.empty_like(x)
    tensors = [x, ids, tw, state['pnw'], bank['w13p'], bank['w2p'], state['gs13'], state['gs2'], inter, y, ss,
               state['smallm_counters'][flags], out]
    ptrs = [v.data_ptr() for v in tensors]
    mod.run(ptrs, m, data['config']['local_start'], stages, grid, torch.cuda.current_stream(dev).cuda_stream, flags)
    return out


def _s4_prepare_banks(state,weights):
    bank=state['moe_bank']
    dev=weights['w13'].device
    state['s4_nt1_bank']=dict(gs13=bank['gs13'],gs2=bank['gs2'],pnw=bank['pnw'],sms=bank['grid'],
        banks={'lop3':dict(w13p=bank['w13p'],w2p=bank['w2p'])})
    smallm_bind(state['s4_nt1_bank'],SMALLM_FLAGS)
    state['s5_hh_base']=nf_perm(nf_wide(issue_prepare(nf_ilv(dict(weights,post_norm_w=weights['post_expert_norm'])))))






NF_CUDA='#define BR_KINDS(F) F(26,32,true,7,0,0,0) F(21,32,false,6,0,0,0) F(70,16,true,7,1,8,16) F(71,16,false,6,1,8,16)\n// Branch/remainder lab kernels. Banks stay packed E2M1 plus uint8 exponent/mantissa scale codes\n// (sign/zero folded into the codes); scales convert to BF16 inside the timed kernel.\n// One task = one 32-row group of one expert x one 64-channel M tile (route NT=32 numbering).\n// A kernel instance owns the groups whose expert count (FM=1) or rows-in-group (FM=2) lies in (LO,HI].\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <vector>\n#include <stdexcept>\n#define BR_CUDA_CHECK(expr) do{cudaError_t br_s=(expr);if(br_s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(br_s));}while(0)\n#define BR_REQUIRE(c) do{if(!(c))throw std::runtime_error("branch lab argument check");}while(0)\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define BR_HOPPER 1\n#else\n#define BR_HOPPER 0\n#endif\n\nnamespace mwbr {\nconstexpr int BK=64,KT=128,S=4,BIG=1<<30;\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out,unsigned scale_base){\n unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;\n unsigned even,odd,rot=rr(raw,8);\n asm("lop3.b32 %0,%1,0x0ff00ff0,%2,0xea;":"=r"(even):"r"(raw),"r"(scale_base));\n asm("lop3.b32 %0,%1,0x0ff00ff0,%2,0xea;":"=r"(odd):"r"(rot),"r"(scale_base));\n unsigned s0=__byte_perm(even,even,0x1010),s2=__byte_perm(even,even,0x3232);\n unsigned s1=__byte_perm(odd,odd,0x1010),s3=__byte_perm(odd,odd,0x3232);\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),\n  "r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b,uint64_t desc){\n#if BR_HOPPER\n uint64_t d=desc;\n d+=b*2;\n if constexpr(NT==8){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3},{%4,%5,%6,%7},%8,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n static_assert(NT==32,"tile width");\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n // Portable stand-in: identical shared layout and fragment ownership, immediate mma.sync.\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs,uint64_t desc){\n#pragma unroll\n for(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b,desc);\n#if BR_HOPPER\n asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\n#endif\n}\ntemplate<int NT,int VEC>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*SC,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n constexpr int NJ=NT>=16?NT/16:1;\n int t=threadIdx.x,w=t>>5,l=t&31;\n if constexpr(VEC){\n  if(t<64){unsigned group=t>>4,lane=t&15;\n   unsigned base=((unsigned)e*(N/16)+mt*4+group)*128*64+tile*(KT/32)*64+lane*4;\n   #pragma unroll\n   for(int g=0;g<KT/32;++g){unsigned dst=smaddr(q+g*1024+t*16);const void*sp=Q+base+g*64;\n    asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");}\n  }\n  if(t>=64&&t<64+KT/4){unsigned i=t-64,g=i/8,group=(i%8)/2,half=i%2;\n   unsigned srcidx=(((unsigned)e*(N/16)+mt*4+group)*128+tile*(KT/32)+g)*32+half*16;\n   unsigned dst=smaddr(s+g*128+group*32+half*16);const void*sp=SC+srcidx;\n   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");\n  }\n }else{\n  #pragma unroll\n  for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n   async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\n   if(l<8)async_copy(s+g*128+w*32+l*4,SC+z*32+l*4,4,1);\n  }\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;\n   if(NT>=16||r<NT)async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\ntemplate<int NT>constexpr int smem_bytes(){return 1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;}\n\ntemplate<int NT,bool FIRST,int VEC,int FM,int LO,int HI>\n__global__ void __launch_bounds__(128,2) gemm(const int*Q,const unsigned char*SC,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,unsigned scale_base){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT,N=FIRST?8192:4096,MT=N/64,NJ=NT>=16?NT/16:1;\n constexpr bool WIDE=VEC>=6;constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(1024) unsigned char typed_sm[];\n unsigned char*qs=typed_sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(typed_sm+BS);\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31;\n uint64_t desc0=(uint64_t)((smaddr(bs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n int total=TASK[32]*MT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  int group=unit/MT,mt=unit%MT,ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*32,count=C[e],offset=OFF[e];\n  if constexpr(FM==1){if(count<=LO||count>HI)continue;}\n  if constexpr(FM==2){int rem=min(count-row,32);if(rem<=LO||rem>HI)continue;}\n  int src[NJ];\n  #pragma unroll\n  for(int j=0;j<NJ;++j){int rl=t/8+j*16,r=row+rl;src[j]=(rl<NT&&r<count)?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  if constexpr(VEC>=2){\n   // Lean K loop (same smem layout, copies and math as VEC==1): per-unit global pointers advanced by a fixed\n   // stride, one commit group per stage (empty past the end) so the wait count is a constant, and the stage\n   // ring unrolled so every shared offset and WGMMA descriptor is a compile-time constant.\n   static_assert(NS%S==0,"stage ring");\n   const int*qg=Q;const unsigned char*sg=SC;const __nv_bfloat16*xg[NJ];bool xv[NJ];\n   const bool wq=t<64,ws=t>=64&&t<64+KT/4;\n   if(wq){unsigned gp=t>>4,ln=t&15;qg=Q+((unsigned)e*(N/16)+mt*4+gp)*128*64+ln*4;}\n   if(ws){unsigned i=t-64;if constexpr(WIDE)sg=SC+(((unsigned)e*(N/16)+mt*4+i/8)*128*32+(i%8)*16);else{unsigned g=i/8,gp=(i%8)/2,hf=i%2;sg=SC+((((unsigned)e*(N/16)+mt*4+gp)*128+g)*32+hf*16);}}\n   unsigned qd=smaddr(qs+(WIDE?(t>>4)*512+(t&15)*16:t*16)),sd=0;\n   if(ws){unsigned i=t-64;if constexpr(WIDE)sd=smaddr(ss+i*16);else{unsigned g=i/8,gp=(i%8)/2,hf=i%2;sd=smaddr(ss+g*128+gp*32+hf*16);}}\n   unsigned bd[NJ];\n   #pragma unroll\n   for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;xv[j]=src[j]>=0;xg[j]=X+(size_t)max(src[j],0)*4096+k;\n    bd[j]=smaddr(bs+r*64+(k^((r&7)*8)));}\n   auto issue=[&](int slot){\n    if(wq){\n     #pragma unroll\n     for(int g=0;g<KT/32;++g)asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(qd+slot*QB+(WIDE?(g>>1)*2048+(g&1)*256:g*1024)),"l"(qg+g*64):"memory");\n     qg+=(KT/32)*64;}\n    if(ws){asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+slot*SB),"l"(sg):"memory");sg+=(KT/32)*32;}\n    #pragma unroll\n    for(int c=0;c<KT/64;++c){\n     #pragma unroll\n     for(int j=0;j<NJ;++j)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(bd[j]+(slot*NT*KT+c*NT*64)*2),"l"(xg[j]+c*64),"r"(xv[j]?16:0):"memory");\n    }\n    #pragma unroll\n    for(int j=0;j<NJ;++j)xg[j]+=KT;\n   };\n   #pragma unroll\n   for(int i=0;i<S-1;++i){issue(i);asm volatile("cp.async.commit_group;":::"memory");}\n   #pragma unroll 1\n   for(int k0=0;k0<NS;k0+=S){\n    #pragma unroll\n    for(int st=0;st<S;++st){\n     asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");__syncthreads();\n#if BR_HOPPER\n     if constexpr(VEC<4||WIDE)asm volatile("fence.proxy.async.shared::cta;":::"memory");  // VEC 4/5: timing probe without it\n#endif\n     uint4 sv;if constexpr(WIDE)sv=*(const uint4*)(ss+st*SB+(w*8+(l>>2))*16);\n     #pragma unroll\n     for(int sub=0;sub<2;++sub){\n      if constexpr(WIDE){uint4 q=*(const uint4*)(qs+st*QB+sub*2048+t*16);\n       dq(q.x,q.y,sub?sv.z:sv.x,af[sub],scale_base);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,scale_base);\n      }else{\n      #pragma unroll\n      for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n       unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8,scale_base);}}\n      keep(af[sub]);\n#if BR_HOPPER\n      asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n      mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64,desc0+(st*NT*KT+sub*NT*64)/8);keep(af[sub^1]);\n      if(sub==0){if(k0+st+S-1<NS)issue((st+S-1)%S);asm volatile("cp.async.commit_group;":::"memory");}\n     }\n    }\n   }\n  }else{\n  #pragma unroll\n  for(int i=0;i<S-1;++i)load_stage<NT,VEC>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SC,X,src,e,mt,N,i);\n  #pragma unroll 1\n  for(int k=0;k<NS;++k){\n   int st=k%S;cp_wait<NS>(k);__syncthreads();\n#if BR_HOPPER\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll\n   for(int sub=0;sub<2;++sub){\n    #pragma unroll\n    for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n     unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8,scale_base);}\n    keep(af[sub]);\n#if BR_HOPPER\n    asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n    mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64,desc0+(st*NT*KT+sub*NT*64)/8);keep(af[sub^1]);\n    if(sub==0&&k+S-1<NS){int dst=(k+S-1)%S;\n     load_stage<NT,VEC>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SC,X,src,e,mt,N,k+S-1);}\n   }\n  }\n  }\n#if BR_HOPPER\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  if constexpr(FIRST&&(VEC==3||VEC==5||VEC==7)){\n   // Gate/up rows interleaved by 8 in prepare (per 64-row tile: G0-7,U0-7,G8-15,U8-15,...): accumulator j holds gate\n   // and j+2 the up row of the same channel, so SwiGLU finishes in registers (same BF16 roundings as below).\n   float gg=GS[e]*256.f,gu=GS[e+32]*256.f;int ch=mt*32+w*8+(l>>2);\n   #pragma unroll\n   for(int j=0;j<NT/2;++j)if(((j>>1)&1)==0){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    float g=__bfloat162float(__float2bfloat16_rn(acc[j]*gg)),u=__bfloat162float(__float2bfloat16_rn(acc[j+2]*gu));\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+ch]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n   }\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n  }\n  __syncthreads();\n }\n}\n\n// Routing only: no output fill, no readiness state. One CTA owns every pair (same as the\n// original CTA 0); with META it also writes the heavy-only (count>16) 32-row task prefix T32, the\n// one-group-per-expert prefix T8 of experts with 8<count<=16.\ntemplate<bool META>__global__ void __launch_bounds__(256) route_live(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,int*T32,\n int*T8,int M,int NT,int start){\n int t=threadIdx.x;\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\n // All eight cold id loads per thread are issued before the first shared atomic.\n for(int base=0;base<M*8;base+=256*8){int v[8];\n  #pragma unroll\n  for(int k=0;k<8;++k){int i=base+k*256+t;v[k]=i<M*8?ids[i]:start-1;}\n  #pragma unroll\n  for(int k=0;k<8;++k){int i=base+k*256+t,e=v[k]-start,rank=-1;\n   if(e>=0&&e<32){rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;}\n   if(i<M*8)POS[i]=rank;}\n }\n __syncthreads();\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\n  int scan=pad;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\n  int key=n*32+31-t;\n  #pragma unroll\n  for(int k=2;k<=32;k*=2){\n   #pragma unroll\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\n   }\n  }\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\n  if constexpr(META){int ns=key/32,v=ns>16?(ns+31)/32:0,u=ns>8&&ns<=16;\n   #pragma unroll\n   for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),b=__shfl_up_sync(~0u,u,d);if(t>=d){v+=a;u+=b;}}\n   T32[t+1]=v;T8[t+1]=u;if(t==0){T32[0]=0;T8[0]=0;}}\n }\n}\n\n// SPLIT blocks per token, each owning 4096/SPLIT channels; per-element math and order are the same for every SPLIT.\ntemplate<int SPLIT>__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\n constexpr int W=4096/SPLIT;int tok=blockIdx.x/SPLIT,c0=blockIdx.x%SPLIT*W,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\n __syncthreads();int pp[8];float aa[8];\n #pragma unroll\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\n for(int c=c0+t*4;c<c0+W;c+=1024){float4 sum=make_float4(0,0,0,0);\n  #pragma unroll\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n }\n}\n}  // namespace mwbr\n\n// Pointer ABI for gemm: Q,SC,GS,X,C,OFF,ORDER,TASK,SLOTS,PNW,ACT,Y,SS.\n#define BR_ARGS(v) (const int*)v[0],(const unsigned char*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(const __nv_bfloat16*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12]\n#define BR_INIT(ID,NT,F,V,FM,LO,HI) BR_CUDA_CHECK(cudaFuncSetAttribute(mwbr::gemm<NT,F,V,FM,LO,HI>,cudaFuncAttributeMaxDynamicSharedMemorySize,mwbr::smem_bytes<NT>()));\n#define BR_GO(ID,NT,F,V,FM,LO,HI) case ID:mwbr::gemm<NT,F,V,FM,LO,HI><<<grid,128,mwbr::smem_bytes<NT>(),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M,0x70007000u);break;\n#define BR_INFO(ID,NT,F,V,FM,LO,HI) case ID:{cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncGetAttributes(&a,mwbr::gemm<NT,F,V,FM,LO,HI>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwbr::gemm<NT,F,V,FM,LO,HI>,128,mwbr::smem_bytes<NT>()));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)mwbr::smem_bytes<NT>(),n};}\nint64_t br_init(){BR_KINDS(BR_INIT) return 0;}\nstd::vector<int64_t> br_info(int64_t kind){switch(kind){BR_KINDS(BR_INFO) default:BR_REQUIRE(false);}return {};}\nvoid br_run(std::vector<int64_t>v,int64_t kind,int64_t grid,int64_t M,int64_t stream){\n BR_REQUIRE(v.size()==13);\n switch(kind){BR_KINDS(BR_GO) default:BR_REQUIRE(false);}\n BR_CUDA_CHECK(cudaGetLastError());\n}\n// v: ids,C,OFF,ORDER,TASK,SLOTS,POS,T32,T8 (T32/T8 written only with meta)\nvoid br_route(std::vector<int64_t>v,int64_t M,int64_t start,int64_t meta,int64_t stream){\n BR_REQUIRE(v.size()==9);\n if(meta)mwbr::route_live<true><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\n else mwbr::route_live<false><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\n BR_CUDA_CHECK(cudaGetLastError());\n}\n// v: OFF,IDS,TW,POS,Y,SS,OUT\nvoid br_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t split,int64_t stream){\n BR_REQUIRE(v.size()==7);\n#define BR_FIN(S) case S:mwbr::finish<S><<<M*S,256,0,(cudaStream_t)stream>>>((const int*)v[0],(const int*)v[1],(const float*)v[2],(const int*)v[3],(const float*)v[4],(const float*)v[5],(__nv_bfloat16*)v[6],(int)M,(int)start);break;\n switch(split){BR_FIN(1) BR_FIN(2) BR_FIN(4) BR_FIN(8) BR_FIN(16) default:BR_REQUIRE(false);}\n#undef BR_FIN\n BR_CUDA_CHECK(cudaGetLastError());\n}\n\nnamespace pmz {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  uint32_t W;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(W):"r"(A),"r"(B));\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>\n__global__ void __launch_bounds__(NTHR,2) phase(\n const __nv_bfloat16*x,const __nv_bfloat16*pnw,const uint8_t*w13p,const uint8_t*w2p,\n const float*gs13,const float*gs2,const int*C,const int*OFF,const int*SLOTS,\n __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){\n constexpr int NB=8*NT,RED=NT*4;\n extern __shared__ __align__(128) uint8_t smem[];\n uint8_t*stages=smem;float*red=(float*)(smem+nstage*TILE_BYTES);\n uint64_t*fullb=(uint64_t*)(red+2*NCW*32*RED),*emptyb=fullb+nstage;\n Route*rt=(Route*)(emptyb+nstage);\n const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;\n if(tid==0){for(int z=0;z<nstage;++z){mbar_init(fullb+z,1);mbar_init(emptyb+z,NCW);}\n  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\n if(warp==0){int n=C[lane];int nb=(n+NB-1)/NB;if constexpr(SMALL)if(n>THR)nb=0;\n  rt->n[lane]=n;rt->off[lane]=OFF[lane];rt->nblk[lane]=nb;\n  int u1=nb*NT1,u2=nb*NT2;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,u1,d),b=__shfl_up_sync(~0u,u2,d);if(lane>=d){u1+=a;u2+=b;}}\n  rt->us1[lane+1]=u1;rt->us2[lane+1]=u2;\n  if(lane==0){rt->us1[0]=rt->us2[0]=0;}\n  int U1=__shfl_sync(~0u,u1,31),U2=__shfl_sync(~0u,u2,31);\n  if(lane==0){rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));}\n }\n __syncthreads();\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=FIRST?b1-a1:0,total_count=FIRST?first_count:b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = FIRST?0:1; pass < (FIRST?1:2); ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        cur_pass=pass;cur_e=e;cur_nb=nb;seg_tiles=0;\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(SLOTS[e*M+idx] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= gs13[e+32]*256.f;\n            G1=__bfloat162float(__float2bfloat16_rn(G1));\n            U=__bfloat162float(__float2bfloat16_rn(U));\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n        if constexpr (ZSS) {\n          if (m == 0) {\n            for (int z = tid; z < NB * 64; z += NTHR) {\n              const int zi = nb * NB + (z >> 6);\n              if (zi < ne) ss[static_cast<size_t>(off + zi) * 64 + (z & 63)] = 0.f;\n            }\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v*fw[0];\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + (size_t)(off+idx)*64+(m>>2), t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t pmz_init(){int dev,opt;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n auto set=[&]<bool F,bool Z,int T>(){cudaFuncAttributes a{};BR_CUDA_CHECK(cudaFuncGetAttributes(&a,pmz::phase<1,F,true,Z,T>));BR_CUDA_CHECK(cudaFuncSetAttribute(pmz::phase<1,F,true,Z,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));};\n set.template operator()<true,true,16>();set.template operator()<true,true,8>();set.template operator()<false,false,16>();set.template operator()<false,false,8>();return opt;}\nint64_t pmz_smem(int64_t stages,int64_t M){return pmz::smem_bytes(1,stages,M);}\nint64_t pmz_resident(int64_t stages,int64_t M,int64_t first){int n;\n if(first)BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,true,true,true,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));\n else BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,false,true,false,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));return n;}\n// v: x,pnw,w13p,w2p,gs13,gs2,C,OFF,SLOTS,act,y,ss. Experts with count<=thr only; W13 (thr 16) zeroes their SS rows.\nvoid pmz_run(std::vector<int64_t>v,int64_t first,int64_t thr,int64_t M,int64_t stages,int64_t grid,int64_t stream){\n BR_REQUIRE(v.size()==12);BR_REQUIRE(first?(thr==16||thr==8):(thr==16||thr==8));\n auto launch=[&]<bool F,bool Z,int T>(){pmz::phase<1,F,true,Z,T><<<grid,pmz::NTHR,pmz::smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\n (__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(uint8_t*)v[2],(uint8_t*)v[3],(float*)v[4],(float*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11],M,stages);};\n if(first&&thr==8)launch.template operator()<true,true,8>();else if(first)launch.template operator()<true,true,16>();else if(thr==16)launch.template operator()<false,false,16>();else launch.template operator()<false,false,8>();\n BR_CUDA_CHECK(cudaGetLastError());}\n'
NF_CPP='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\nint64_t br_init();std::vector<int64_t> br_info(int64_t);void br_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);void br_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);void br_finish(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);int64_t pmz_init();int64_t pmz_smem(int64_t,int64_t);int64_t pmz_resident(int64_t,int64_t,int64_t);void pmz_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'
NF_OPTS={}
@functools.lru_cache(None)
def nf_module():
    target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
    mod=load_inline(name='nf_m256_'+hashlib.sha256(NF_CUDA.encode()).hexdigest()[:12],cpp_sources=NF_CPP,cuda_sources=NF_CUDA,
        functions=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run'],
        no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
    mod.br_init();NF_OPTS['pmz']=mod.pmz_init()
    return mod

def nf_ilv(w):
    
    perm=torch.empty(8192,dtype=torch.long)
    for i in range(128):
        for q,pos in enumerate((32*i,32*i+16,4096+32*i,4096+32*i+16)):
            perm[pos:pos+16]=torch.tensor([32*i+8*q+k for k in range(8)]+[4096+32*i+8*q+k for k in range(8)])
    perm=perm.to(w['w13'].device)
    return dict(w,w13=w['w13'].index_select(1,perm),w13_sf=w['w13_sf'].index_select(1,perm))

def nf_wide(bank):
    for k in ('w13','w2'):
        q,sf=bank[k],bank[k+'_sf'];e,ng=sf.shape[:2]
        bank[k]=q.view(torch.uint8).reshape(e,ng,64,2,32,8).transpose(3,4).contiguous().view(q.dtype).reshape(q.shape)
        bank[k+'_sf']=sf.reshape(e,ng,32,4,8,4).transpose(3,4).contiguous().reshape(sf.shape)
    return bank

def nf_perm(bank):
    
    for k in ('w13_sf','w2_sf'):
        sf=bank[k];v=sf.contiguous().reshape(-1).view(torch.int32).to(torch.int64)&0xffffffff
        bank[k]=(((v<<4)|(v>>28))&0xffffffff).to(torch.int32).view(torch.uint8).reshape(sf.shape)
    return bank

def nf_plan(data,base,native,streams):
    light_grid=4
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*128;mod=nf_module()
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(c)
    tasks=[torch.empty_like(off) for _ in range(3)];slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();pos=torch.empty_like(ids)
    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((cap,4096),device=dev);ss=torch.empty((cap,64),device=dev)
    out=torch.empty_like(x);start=data['config'].get('local_start',0)
    h13=[base['w13'],base['w13_sf'],base['w13_gs'],x,c,off,order,tasks[1],slots,base['pnw'],act,y,ss]
    h2=[base['w2'],base['w2_sf'],base['w2_gs'],act,c,off,order,tasks[1],slots,base['pnw'],act,y,ss]
    nv=[x,base['pnw'],native['w13p'],native['w2p'],base['w13_gs'],base['w2_gs'],c,off,slots,act,y,ss]
    p13,p2,pn=([t.data_ptr() for t in v] for v in (h13,h2,nv))
    l13=list(h13);l2=list(h2);l13[7]=tasks[2];l2[7]=tasks[2]
    pl13=[t.data_ptr() for t in l13];pl2=[t.data_ptr() for t in l2]
    rp=[t.data_ptr() for t in (ids,c,off,order,tasks[0],slots,pos,tasks[1],tasks[2])]
    fp=[t.data_ptr() for t in (off,ids,tw,pos,y,ss,out)]
    owners=(h13,h2,nv,ids,tw,pos,tasks,l13,l2)
    stages=2
    while stages>1 and mod.pmz_smem(stages,m)>NF_OPTS['pmz']:stages-=1
    sms=torch.cuda.get_device_properties(x.device).multi_processor_count
    grids=[sms*mod.pmz_resident(stages,m,f) for f in (0,1)]
    side,fork,join,light,ljoin=streams;grid=1056 if m<=128 else 2112  
    def route():
        _=owners;mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)
    def body():
        _=owners;parent=torch.cuda.current_stream()
        fork.record(parent);side.wait_event(fork);light.wait_event(fork)
        with torch.cuda.stream(side):  
            mod.br_run(p13,26,grid,m,side.cuda_stream);mod.br_run(p2,21,grid,m,side.cuda_stream);join.record()
        with torch.cuda.stream(light):
            mod.br_run(pl13,70,sms*light_grid,m,light.cuda_stream);mod.br_run(pl2,71,sms*light_grid,m,light.cuda_stream);ljoin.record()
        mod.pmz_run(pn,1,8,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)
        parent.wait_event(join);parent.wait_event(ljoin)
    def finish():
        _=owners;mod.br_finish(fp,m,start,4,torch.cuda.current_stream().cuda_stream);return out
    return dict(steps=(route,body,finish),out=out,owners=owners)

def nf_call(data,state):
    for f in nf_plan(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'])['steps']:out=f()
    return out

def _s4_gate_moe(name,tokens,length,state,weights):
    if name=='nt1_lop3' and tokens not in (1,8,16,32):raise ValueError('S4 small-M geometry')
    if name=='hh_m128' and tokens!=128:raise ValueError('S5 H_H geometry')
    fixtures=[];wants=[]
    for seed in _ATTENTION_PUBLIC_SEEDS:
        fixture=_best7_moe_fixture(weights,tokens,length,seed)
        want=_best5_fp32(fixture,weights)
        if want is None:raise RuntimeError('prepare budget before S4 expert oracle')
        fixtures.append(fixture);wants.append(want)
    graph=_best5_graph(lambda data:_best6_call(data,state,name),fixtures[0])
    worst=0.;worst_row=0.
    for order in (range(3),range(2,-1,-1)):
        for index in order:
            _best6_refresh(graph,fixtures[index]);graph[1].replay()
            error,row=_best5_error(graph[2],wants[index])
            worst=max(worst,error);worst_row=max(worst_row,row)
    if worst>.01 or worst_row>.05:raise RuntimeError(f'S4 {name} FP32 gate {worst:.6f}/{worst_row:.6f}')
    return dict(source='three public streams, forward/reverse fresh route graph replays',
                fp32=worst,worst_row=worst_row,variant=name)


def _s4_encode_scales(sf: torch.Tensor):
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

def issue_prepare(w):
    bank = {}
    for key in ('w13', 'w2'):
        q, sf = (w[key], w[key + '_sf'])
        e, n, k16 = sf.shape
        planes = []
        scales = []
        for expert in range(e):
            code, neg, zero = _s4_encode_scales(sf[expert])
            folded = q[expert].reshape(n, k16, 8) ^ neg.to(torch.uint8)[..., None] * 136
            folded = torch.where(zero[..., None], torch.zeros_like(folded), folded)
            planes.append(mw_repack_planes(folded.reshape(1, n, -1)))
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


def _s6_gate_moe(tokens,length,state,weights):
    if tokens!=256:raise ValueError('mixed_parallel is admitted only at M256')
    fixtures=[];wants=[]
    for seed in _ATTENTION_PUBLIC_SEEDS:
        fixture=_best7_moe_fixture(weights,tokens,length,seed)
        want=_best5_fp32(fixture,weights)
        if want is None:raise RuntimeError('prepare budget before mixed_parallel oracle')
        fixtures.append(fixture);wants.append(want)
    graph=_best5_graph(lambda data:nf_call(data,state),fixtures[0])
    worst=0.;worst_row=0.
    for order in (range(3),range(2,-1,-1)):
        for index in order:
            _best6_refresh(graph,fixtures[index]);graph[1].replay()
            error,row=_best5_error(graph[2],wants[index])
            worst=max(worst,error);worst_row=max(worst_row,row)
    if worst>.01 or worst_row>.05:raise RuntimeError(f'mixed_parallel FP32 gate {worst:.6f}/{worst_row:.6f}')
    return dict(source='three public streams, forward/reverse fresh route graph replays',
                fp32=worst,worst_row=worst_row,policy=6,grid=2112,
                bank='H_H u8 base + recoded NT1 bank',schedule='N_full: NT1<=16 W13->W2 beside heavy N32 W13->W2, forked once, joined before the finalizer')


def _s6_o_projection_op(state,device):
    op=Best7ProjectionCell(state['o'],1,6,fp32=False)
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        gen=torch.Generator(device=device).manual_seed(62172+state['o'].shape[0])
        x=torch.randn((1,state['o'].shape[1]),device=device,dtype=torch.bfloat16,generator=gen)
        for _ in range(2):op(x)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=op(x)
        worst=0.;row=0.
        for _ in range(3):
            x.normal_(generator=gen);graph.replay()
            error,erow=_best5_error(out,x.float()@state['o'].float().T)
            worst=max(worst,error);row=max(row,erow)
        if worst>.005 or row>.025:raise RuntimeError(f'O/config6 FP32 gate {worst:.6f}/{row:.6f}')
        return op,dict(fp32=worst,worst_row=row,config=6,splits=1,source='exact95ab383d M1 O unsplit projection')
    finally:torch.backends.cuda.matmul.allow_tf32=previous



"""Unsplit register-MMA dense projection cell."""
GF_CUDA='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace gf {\nusing B=__nv_bfloat16;\n__device__ __forceinline__ uint64_t gt(){uint64_t x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}\n__device__ __forceinline__ uint64_t ck(){uint64_t x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}\n__device__ __forceinline__ uint4 ld(const B* p){\n  uint4 v;asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"\n    :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");return v;\n}\n__device__ __forceinline__ void mma(float* d,uint4 a,uint4 b,uint4 x){\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\n    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])\n    :"r"(a.x),"r"(b.x),"r"(a.y),"r"(b.y),"r"(x.x),"r"(x.y));\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\n    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])\n    :"r"(a.z),"r"(b.z),"r"(a.w),"r"(b.w),"r"(x.z),"r"(x.w));\n}\n\n// All K partitions stay inside one CTA. No global partials, counters, or\n// reduction launch. The only barrier is the final intra-CTA reduction.\n// Initial immutable weight vectors issue before waiting for activations.\ntemplate<int R,int T,int KW,int DEPTH,bool TRACE>\n__global__ __launch_bounds__(((R+15)/16)*(T/8)*KW*32)\nvoid direct(const B* __restrict__ w,const B* __restrict__ x,void* __restrict__ y,\n            uint64_t* stamps,int M,int N,int K,bool fp32,bool pdl){\n  constexpr int RG=(R+15)/16,TM=T/8,WARPS=RG*TM*KW;\n  __shared__ float sums[WARPS*128];\n  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;\n  const int kw=warp%KW,cell=warp/KW,rg=cell/TM,tm=cell%TM;\n  const int row0=blockIdx.x*R+rg*16,r=row0+lane/4;\n  const int tok0=blockIdx.y*T+tm*8,tok=tok0+lane/4;\n  const int ks=K/KW,begin=kw*ks,end=begin+ks;\n  uint64_t clocks[7],times[7];\n#define MARK(I) if constexpr(TRACE){if(tid==0){clocks[I]=ck();times[I]=gt();}}\n  MARK(0);\n  if(pdl)asm volatile("griddepcontrol.launch_dependents;":::"memory");\n  uint4 wa[DEPTH],wb[DEPTH];\n  MARK(1);\n#pragma unroll\n  for(int u=0;u<DEPTH;++u){\n    int k=begin+u*32+(lane%4)*8;\n    wa[u]=(r<blockIdx.x*R+R)?ld(w+(size_t)r*K+k):uint4{};\n    wb[u]=(r+8<blockIdx.x*R+R)?ld(w+(size_t)(r+8)*K+k):uint4{};\n  }\n  MARK(2);\n  if(pdl)asm volatile("griddepcontrol.wait;":::"memory");\n  MARK(3);\n  float acc[4]={};\n  for(int base=begin;base<end;base+=DEPTH*32){\n    uint4 na[DEPTH],nb[DEPTH];\n#pragma unroll\n    for(int u=0;u<DEPTH;++u){\n      int k=base+DEPTH*32+u*32+(lane%4)*8;\n      na[u]=(k<end && r<blockIdx.x*R+R)?ld(w+(size_t)r*K+k):uint4{};\n      nb[u]=(k<end && r+8<blockIdx.x*R+R)?ld(w+(size_t)(r+8)*K+k):uint4{};\n    }\n#pragma unroll\n    for(int u=0;u<DEPTH;++u){\n      int k=base+u*32+(lane%4)*8;\n      uint4 xv=tok<M?ld(x+(size_t)tok*K+k):uint4{};\n      mma(acc,wa[u],wb[u],xv);\n      wa[u]=na[u];wb[u]=nb[u];\n    }\n  }\n  MARK(4);\n  const int rr=lane/4,tt=(lane%4)*2;\n  if constexpr(KW>1){\n#pragma unroll\n  for(int i=0;i<4;++i)\n    sums[warp*128+(tt+(i&1))*16+rr+(i/2)*8]=acc[i];\n  __syncthreads();\n  if(kw==0){\n#pragma unroll\n    for(int i=0;i<4;++i){\n      float v=0;\n#pragma unroll\n      for(int s=0;s<KW;++s)v+=sums[(cell*KW+s)*128+(tt+(i&1))*16+rr+(i/2)*8];\n      acc[i]=v;\n    }\n  }\n  }\n  MARK(5);\n  if(kw==0){\n#pragma unroll\n    for(int i=0;i<4;++i){\n      int row=row0+rr+(i/2)*8,t=tok0+tt+(i&1);\n      if(row<blockIdx.x*R+R && t<M){\n        size_t idx=(size_t)t*N+row;\n        if(fp32)((float*)y)[idx]=acc[i];else ((B*)y)[idx]=__float2bfloat16_rn(acc[i]);\n      }\n    }\n  }\n  if constexpr(TRACE)__syncthreads();\n  MARK(6);\n  if constexpr(TRACE){if(tid==0){\n    uint64_t* dst=stamps+((size_t)blockIdx.y*gridDim.x+blockIdx.x)*16;\n#pragma unroll\n    for(int i=0;i<7;++i){dst[i*2]=clocks[i];dst[i*2+1]=times[i];}\n    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));dst[14]=sm;dst[15]=WARPS;\n  }}\n#undef MARK\n}\n__global__ void anchor(uint64_t* p){if(threadIdx.x==0){p[0]=ck();p[1]=gt();}}\ntemplate<int R,int T,int KW,int D>\ncudaError_t launch(const B* w,const B* x,void* y,uint64_t* st,int M,int N,int K,\n                   bool fp32,bool pdl,bool trace,cudaStream_t stream){\n  constexpr int threads=((R+15)/16)*(T/8)*KW*32;\n  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;\n  attr.val.programmaticStreamSerializationAllowed=pdl;\n  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(N/R,(M+T-1)/T);cfg.blockDim=threads;\n  cfg.stream=stream;cfg.attrs=&attr;cfg.numAttrs=1;\n  if(trace)return cudaLaunchKernelEx(&cfg,direct<R,T,KW,D,true>,w,x,y,st,M,N,K,fp32,pdl);\n  return cudaLaunchKernelEx(&cfg,direct<R,T,KW,D,false>,w,x,y,st,M,N,K,fp32,pdl);\n}\n}\nextern "C" int gf_run(const void* w,const void* x,void* y,void* st,int M,int N,int K,\n                       int config,int fp32,int pdl,int trace,void* stream){\n  using namespace gf;cudaError_t e=cudaErrorInvalidValue;\n#define GO(R,T,KW,D) launch<R,T,KW,D>((const B*)w,(const B*)x,y,(uint64_t*)st,M,N,K,fp32,pdl,trace,(cudaStream_t)stream)\n  switch(config){\n    case 0:e=GO(8,8,4,4);break;case 1:e=GO(16,8,4,4);break;\n    case 2:e=GO(16,8,8,4);break;case 3:e=GO(32,8,4,4);break;\n    case 4:e=GO(16,16,4,4);break;case 5:e=GO(16,32,2,4);break;\n    case 6:e=GO(16,8,4,8);break;case 7:e=GO(16,16,4,8);break;\n    case 8:e=GO(8,8,8,4);break;\n    case 9:e=GO(32,8,2,4);break;case 10:e=GO(64,8,2,4);break;\n    case 11:e=GO(32,16,2,4);break;case 12:e=GO(32,32,1,4);break;\n    case 13:e=GO(16,8,1,8);break;\n  }\n#undef GO\n  return (int)(e==cudaSuccess?cudaGetLastError():e);\n}\nextern "C" int gf_anchor(void* p,void* stream){\n  gf::anchor<<<1,32,0,(cudaStream_t)stream>>>((uint64_t*)p);return (int)cudaGetLastError();\n}\n'
import ctypes as _gfct
import hashlib as _gfhash
import os as _gfos
import torch
from torch.utils.cpp_extension import load_inline as _gfload

GF_CONFIGS={0:(8,8,4,4),1:(16,8,4,4),2:(16,8,8,4),3:(32,8,4,4),
            4:(16,16,4,4),5:(16,32,2,4),6:(16,8,4,8),7:(16,16,4,8),8:(8,8,8,4),
            9:(32,8,2,4),10:(64,8,2,4),11:(32,16,2,4),12:(32,32,1,4),13:(16,8,1,8)}
_GF_LIB=None


def gf_library():
    global _GF_LIB
    if _GF_LIB is None:
        old=_gfos.environ.get('TORCH_CUDA_ARCH_LIST')
        _gfos.environ['TORCH_CUDA_ARCH_LIST']='9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        try:
            path=_gfload(name='muse_gemv_fixed_'+_gfhash.sha256(GF_CUDA.encode()).hexdigest()[:16],
                cpp_sources='',cuda_sources=GF_CUDA,functions=None,is_python_module=False,
                no_implicit_headers=True,extra_cuda_cflags=['-O3','-std=c++17','-lineinfo','--ptxas-options=-v'],verbose=False)
        finally:
            if old is None:_gfos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_gfos.environ['TORCH_CUDA_ARCH_LIST']=old
        _GF_LIB=_gfct.CDLL(path)
        _GF_LIB.gf_run.argtypes=[_gfct.c_void_p]*4+[_gfct.c_int]*7+[_gfct.c_void_p]
        _GF_LIB.gf_run.restype=_gfct.c_int
        _GF_LIB.gf_anchor.argtypes=[_gfct.c_void_p]*2;_GF_LIB.gf_anchor.restype=_gfct.c_int
    return _GF_LIB


class FixedProjection:
    """Unsplit projection cell. Construct per call site outside graph capture."""
    def __init__(self,w,tokens,config,fp32=False,pdl=True,trace=False):
        r,t,kw,d=GF_CONFIGS[config]
        if w.dtype!=torch.bfloat16 or not w.is_contiguous() or w.shape[0]%r or w.shape[1]%(kw*d*32):
            raise ValueError('invalid original BF16 bank')
        self.lib=gf_library();self.w=w;self.tokens=tokens;self.config=config
        self.fp32=bool(fp32);self.pdl=bool(pdl);self.trace=bool(trace)
        self.ctas=(w.shape[0]//r)*((tokens+t-1)//t)
        self.stamps=torch.empty((self.ctas,16),device=w.device,dtype=torch.int64) if trace else None
        self.out=torch.empty((1,tokens,w.shape[0]),device=w.device,dtype=torch.float32 if fp32 else torch.bfloat16)

    def __call__(self,x):
        if x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.shape!=(self.tokens,self.w.shape[1]):
            raise ValueError('activation ABI mismatch')
        error=self.lib.gf_run(self.w.data_ptr(),x.data_ptr(),self.out.data_ptr(),
            self.stamps.data_ptr() if self.stamps is not None else 0,self.tokens,*self.w.shape,
            self.config,int(self.fp32),int(self.pdl),int(self.trace),torch.cuda.current_stream(x.device).cuda_stream)
        if error:raise RuntimeError(f'gf_run CUDA={error}')
        return self.out,1


def gf_anchor(buffer):
    error=gf_library().gf_anchor(buffer.data_ptr(),torch.cuda.current_stream(buffer.device).cuda_stream)
    if error:raise RuntimeError(f'gf_anchor CUDA={error}')


class Best7ProjectionCell(FixedProjection):
    """Tensor-returning ABI for best7's existing ('stream', config, 1) cell."""
    def __call__(self,x):
        partial,_=super().__call__(x)
        return partial[0]



VQ8_CUDA = '// vq8: verify-block (q_len 8) attention over a paged FP8 cache, both KV heads of a pair per CTA.\n//\n// Layout: one persistent CTA per SM (stream-K over the (request, head pair, 64-key tile) space).\n//   warps 0-3  consumer warpgroup for head 2*hp,   64 query rows (8 tokens x 8 GQA heads)\n//   warps 4-7  consumer warpgroup for head 2*hp+1\n//   warps 8-11 producer: page-table lookups (4 tiles ahead), cp.async gather of 128-byte K and V\n//              rows (both heads), then the V transpose into the K-major PV operand.\n// Consumers: QK = native FP8 WGMMA (Q, K from shared memory), online softmax in registers,\n// P packed to E4M3 in registers and used directly as WGMMA\'s A operand (keys permuted inside each\n// 16-key group to match the accumulator layout; the producer\'s transpose applies the same\n// permutation to V). QK of tile j+1 is issued before PV of tile j and the softmax of j+1 overlaps\n// the PV. Each CTA writes one unnormalized partial (O, max, sum) per segment it touches; the\n// combine kernel (PDL) merges them and applies RMSNorm x sigmoid(gate).\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n#include <cuda_fp8.h>\n#include <cstdint>\n#include <climits>\n#include <stdexcept>\n#include <vector>\n#define DI __device__ __forceinline__\n// VQ8_CLK=1 (lab builds only): per-CTA phase clocks after the partials (row map in vq8_main and\n// vq8_combine); the shared memory map moves up by CLKPAD bytes, which hold the per-warp wait counters.\n#ifndef VQ8_CLK\n#define VQ8_CLK 0\n#endif\n#ifndef USE_WGMMA\n#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)\n#define USE_WGMMA 1\n#else\n#define USE_WGMMA 0\n#endif\n#endif\n\nnamespace {\nconstexpr int NSMAX = 128;  // segments = B * HK/2\nconstexpr int PR = 68;      // floats per partial row: 64 O + max + sum + pad\nconstexpr int SCHED = 256;  // floats after the partials: the partial-slot prefix over segments (main -> combine)\n\nstruct VP {\n  const void* qkv; const int64_t* loc; const int* pt; const int* seq;\n  uint8_t* kc; uint8_t* vc; const float* ks; const float* vs; const float* inv_freq;\n  float* part; __nv_bfloat16* out;\n  int B, L, HQ, HK, HP, NS, stride, pages, ps, lg, window, proj_splits, min_tiles, grid;\n  int wpr;  // combine: warps per output row (1, 2, 4 or 8), each merging a slice of the row\'s partials\n  int gate;  // combine applies sigmoid(gate) (0: RMSNorm only, the gate is applied by its own kernel)\n  int64_t split_stride;  // elements between projection partials (proj_splits > 0)\n  float sc;  // sm_scale * qk_scale_factor * log2(e); k_scale is applied on device\n};\n\nstruct Sched {\n  int segP[NSMAX + 1];   // tile prefix over segments (each segment: cached tiles + 1 new-token tile)\n  int segSeq[NSMAX];\n  int partB[NSMAX + 1];  // partial-slot prefix over segments\n  int T, G, npo, nreal;\n  int po_s[NSMAX], po_ka[NSMAX], po_kb[NSMAX];\n};\n\nDI uint32_t su(const void* x) { return (uint32_t)__cvta_generic_to_shared(x); }\nDI void cp16(void* dst, const void* src, bool valid) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(su(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");\n}\nDI void cp4(void* dst, const void* src) {\n  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(su(dst)), "l"(src) : "memory");\n}\nDI void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }\nconstexpr int CLKPAD = VQ8_CLK ? 1024 : 0;\n#if VQ8_CLK\nDI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }\n// [w] cycles warp w spent in mbarrier waits, [12+w] globaltimer at warp w\'s first completed wait,\n// [24+w] cycles in cp.async.wait_group.\nDI unsigned long long* clk_area() { extern __shared__ __align__(1024) uint8_t sm[]; return (unsigned long long*)sm; }\n// Phase cycles of one thread (thread 0: consumer WG0; thread 256: producer warp 8), summed over tiles:\n// [48] QK wait, [49] softmax, [50] PV wait, [51] step issue (fences + 4 wgmma), [52] producer publish\n// (transpose + proxy fence + arrive), [53] producer issue (walk, page ring, 8 cp.async, commit).\n#define CLK_MARK(v) const long long v = clock64()\n#define CLK_ADD(tid, slot, v) if (threadIdx.x == (tid)) clk_area()[slot] += clock64() - (v)\n#else\n#define CLK_MARK(v)\n#define CLK_ADD(tid, slot, v)\n#endif\ntemplate <int N> DI void cp_wait() {\n#if VQ8_CLK\n  const long long c0 = clock64();\n#endif\n  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");\n#if VQ8_CLK\n  if ((threadIdx.x & 31) == 0) clk_area()[24 + (threadIdx.x >> 5)] += clock64() - c0;\n#endif\n}\nDI float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(y) : "f"(x)); return y; }\nDI int wsum_i(int x) {\n#pragma unroll\n  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);\n  return x;\n}\nDI float wsum(float x) {\n#pragma unroll\n  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);\n  return x;\n}\nDI uint16_t fp8pair(float x, float y) {\n  return __nv_cvt_float2_to_fp8x2(make_float2(x, y), __NV_SATFINITE, __NV_E4M3);\n}\nDI uint32_t pack4(float a, float b, float c, float d) {\n  return (uint32_t)fp8pair(a, b) | ((uint32_t)fp8pair(c, d) << 16);\n}\n// 64-byte rows, 16-byte chunks XOR-swizzled by (row>>1)&3: CUTLASS Layout_K_SW64_Atom<uint8_t>.\nDI int off(int r, int c) { return r * 64 + ((((c >> 4) ^ ((r >> 1) & 3))) << 4) + (c & 15); }\nDI uint32_t word(const uint8_t* p, int r, int c) { return *(const uint32_t*)(p + off(r, c)); }\nDI uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }\nDI void ldsm4t(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t addr) {\n  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"\n               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr) : "memory");\n}\nDI void mb_init(uint64_t* p, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" ::"r"(su(p)), "r"(n) : "memory"); }\nDI void mb_arrive(uint64_t* p) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" ::"r"(su(p)) : "memory"); }\nDI void mb_wait(uint64_t* p, int phase) {\n#if VQ8_CLK\n  const long long c0 = clock64();\n#endif\n  uint32_t ok;\n  do {\n    asm volatile("{.reg .pred q;mbarrier.try_wait.parity.shared::cta.b64 q,[%1],%2;selp.b32 %0,1,0,q;}"\n                 : "=r"(ok) : "r"(su(p)), "r"(phase) : "memory");\n  } while (!ok);\n#if VQ8_CLK\n  if ((threadIdx.x & 31) == 0) {\n    unsigned long long* a = clk_area();\n    const int w = threadIdx.x >> 5;\n    a[w] += clock64() - c0;\n    if (!a[12 + w]) a[12 + w] = gtime();\n  }\n#endif\n}\nDI void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }\nDI void wg_bar(int id) { asm volatile("bar.sync %0,128;" ::"r"(id) : "memory"); }\nDI uint64_t desc(uint32_t a) { return (uint64_t)(a >> 4) | (1ull << 16) | (32ull << 32) | (2ull << 62); }\nDI void wg_fence() {\n#if USE_WGMMA\n  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");\n#endif\n}\nDI void wg_commit() {\n#if USE_WGMMA\n  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");\n#endif\n}\ntemplate <int N> DI void wg_wait() {\n#if USE_WGMMA\n  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");\n#endif\n}\nDI void protect(float (&a)[8][4]) {\n#pragma unroll\n  for (int n = 0; n < 8; n++)\n#pragma unroll\n    for (int e = 0; e < 4; e++) asm volatile("" : "+f"(a[n][e])::"memory");\n}\nDI void protect(uint32_t (&a)[8]) {\n#pragma unroll\n  for (int n = 0; n < 8; n++) asm volatile("" : "+r"(a[n])::"memory");\n}\nDI void mma8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n#if USE_WGMMA\n#define WG_D "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"\n#define WG_O(d) "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), \\\n  "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]),     \\\n  "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]),     \\\n  "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]),     \\\n  "+f"(d[7][3])\nDI void wg_ss(float (&d)[8][4], uint64_t a, uint64_t b, int acc) {\n  asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D\n               ",%32,%33,p,1,1;}" : WG_O(d) : "l"(a), "l"(b), "r"(acc));\n}\nDI void wg_rs(float (&d)[8][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {\n  asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D\n               ",{%32,%33,%34,%35},%36,p,1,1;}" : WG_O(d) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b));\n}\n#endif\n\nDI void pin(uint64_t& d) { asm volatile("" : "+l"(d)::"memory"); }\n// S[64 rows x 64 keys] = Q[64 x 64] K^T. Both operands K-major SW64 in shared memory.\nDI void qk(float (&s)[8][4], const uint8_t* Q, const uint8_t* K) {\n#if USE_WGMMA\n  uint64_t qa = desc(su(Q)), ka = desc(su(K)), qb = qa + 2, kb = ka + 2;\n  pin(qa); pin(ka); pin(qb); pin(kb);\n  wg_ss(s, qa, ka, 0);\n  wg_ss(s, qb, kb, 1);\n#else\n  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, w = (threadIdx.x >> 5) & 3;\n#pragma unroll\n  for (int n = 0; n < 8; n++)\n#pragma unroll\n    for (int e = 0; e < 4; e++) s[n][e] = 0.f;\n#pragma unroll\n  for (int kk = 0; kk < 2; kk++) {\n    const uint32_t a0 = word(Q, w * 16 + g, 32 * kk + 4 * t), a1 = word(Q, w * 16 + g + 8, 32 * kk + 4 * t);\n    const uint32_t a2 = word(Q, w * 16 + g, 32 * kk + 16 + 4 * t), a3 = word(Q, w * 16 + g + 8, 32 * kk + 16 + 4 * t);\n#pragma unroll\n    for (int n = 0; n < 8; n++)\n      mma8(s[n], a0, a1, a2, a3, word(K, n * 8 + g, 32 * kk + 4 * t), word(K, n * 8 + g, 32 * kk + 16 + 4 * t));\n  }\n#endif\n}\n// O[64 x 64 dims] += P[64 x 64 permuted keys] VT^T. P in registers (A fragment), VT K-major SW64.\nDI void pv(float (&o)[8][4], const uint32_t (&a)[8], const uint8_t* VT) {\n#if USE_WGMMA\n  uint64_t va = desc(su(VT)), vb = va + 2;\n  pin(va); pin(vb);\n  wg_rs(o, a[0], a[1], a[2], a[3], va);\n  wg_rs(o, a[4], a[5], a[6], a[7], vb);\n#else\n  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;\n#pragma unroll\n  for (int kk = 0; kk < 2; kk++)\n#pragma unroll\n    for (int n = 0; n < 8; n++)\n      mma8(o[n], a[4 * kk], a[4 * kk + 1], a[4 * kk + 2], a[4 * kk + 3], word(VT, n * 8 + g, 32 * kk + 4 * t),\n           word(VT, n * 8 + g, 32 * kk + 16 + 4 * t));\n#endif\n}\n\n// N (token, col) pairs of the block\'s input with every load in flight at once (a per-pair branch\n// between the BF16 and split paths serialized them: one memory latency per pair). Every token must be\n// a valid row; pairs with valid[i] false return 0.\ntemplate <int N>\nDI void input_pairs(const VP& p, const int (&token)[N], const int (&col)[N], const bool (&valid)[N], float2 (&x)[N]) {\n  if (!p.proj_splits) {\n    uint32_t raw[N];\n#pragma unroll\n    for (int i = 0; i < N; i++)\n      raw[i] = *(const uint32_t*)((const __nv_bfloat16*)p.qkv + (int64_t)token[i] * p.stride + col[i]);\n#pragma unroll\n    for (int i = 0; i < N; i++) {\n      __nv_bfloat162 h;\n      memcpy(&h, &raw[i], 4);\n      x[i] = valid[i] ? __bfloat1622float2(h) : make_float2(0.f, 0.f);\n    }\n    return;\n  }\n  const float* base = (const float*)p.qkv;\n#pragma unroll\n  for (int i = 0; i < N; i++) x[i] = make_float2(0.f, 0.f);\n#pragma unroll 2\n  for (int s = 0; s < p.proj_splits; s++) {\n    float2 v[N];\n#pragma unroll\n    for (int i = 0; i < N; i++)\n      v[i] = *(const float2*)(base + (int64_t)s * p.split_stride + (int64_t)token[i] * p.stride + col[i]);\n#pragma unroll\n    for (int i = 0; i < N; i++) { x[i].x += v[i].x; x[i].y += v[i].y; }\n  }\n#pragma unroll\n  for (int i = 0; i < N; i++)\n    x[i] = valid[i] ? __bfloat1622float2(__floats2bfloat162_rn(x[i].x, x[i].y)) : make_float2(0.f, 0.f);\n}\n// (cos, sin) of a RoPE angle, out of line: one copy of sincosf\'s large-argument path instead of one per\n// row keeps the prologue\'s executed code contiguous (cold instruction fetch dominates short runs).\n__device__ __noinline__ float2 rope_cs(float a) {\n  float sn, cs;\n  sincosf(a, &sn, &cs);\n  return make_float2(cs, sn);\n}\nDI float2 rot(float2 x, float2 cs) { return make_float2(x.x * cs.x - x.y * cs.y, x.x * cs.y + x.y * cs.x); }\n// part: (grid + NS) * 128 partial rows of PR floats, then SCHED floats of partial-slot prefix, then (VQ8_CLK)\n// the clock rows.\nDI int* sched_area(const VP& p) { return (int*)(p.part + (int64_t)(p.grid + p.NS) * 128 * PR); }\nDI unsigned long long* clock_rows(const VP& p) {\n  return (unsigned long long*)(p.part + (int64_t)(p.grid + p.NS) * 128 * PR + SCHED);\n}\nDI int cta_of(int tau, int G, int T) { return (int)(((long long)(tau + 1) * G - 1) / T); }\n\n// One full warp: tile and partial-slot prefixes over the segments, the effective CTA count.\nDI void schedule_warp(const VP& p, Sched& S) {\n  const int lane = threadIdx.x & 31;\n  int carry = 0;\n  for (int base = 0; base < p.NS; base += 32) {\n    const int s = base + lane;\n    int ts = 0;\n    if (s < p.NS) {\n      const int seq = p.seq[s / p.HP];\n      const int lo = p.window ? max(0, seq - p.window + 1) : 0;\n      S.segSeq[s] = seq;\n      ts = ((seq - lo + 63) >> 6) + 1;\n    }\n    int x = ts;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, d); if (lane >= d) x += y; }\n    if (s < p.NS) S.segP[s + 1] = carry + x;\n    carry += __shfl_sync(0xffffffffu, x, 31);\n  }\n  const int T = carry;\n  const int G = max(1, min(p.grid, T / max(1, p.min_tiles)));\n  __syncwarp();\n  int pc = 0;\n  for (int base = 0; base < p.NS; base += 32) {\n    const int s = base + lane;\n    int np = 0;\n    if (s < p.NS) {\n      const int a = s ? S.segP[s] : 0, e = S.segP[s + 1];\n      np = cta_of(e - 1, G, T) - cta_of(a, G, T) + 1;\n    }\n    int x = np;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, d); if (lane >= d) x += y; }\n    if (s < p.NS) S.partB[s + 1] = pc + x;\n    pc += __shfl_sync(0xffffffffu, x, 31);\n  }\n  if (lane == 0) { S.segP[0] = 0; S.partB[0] = 0; S.T = T; S.G = G; }\n  __syncwarp();\n}\n\n// The producer warp pw transposes key group pw (16 keys) of both heads: V[key][dim] -> VT[dim][perm key].\n// Within each 16-key group, operand column 4t+j holds key (2t, 2t+1, 8+2t, 9+2t)[j]: the order in which\n// a thread\'s softmax accumulator already holds P, so P needs no shuffles.\nDI void transpose_group(const uint8_t* VRs, uint8_t* VTs, int pw, int lane) {\n  const int mi = lane >> 3, rr = lane & 7, t = lane & 3, cc = lane >> 2;\n#pragma unroll\n  for (int h = 0; h < 2; h++) {\n#pragma unroll\n    for (int dp = 0; dp < 2; dp++) {\n      const int key = 16 * pw + ((mi & 1) << 3) + rr, dg = 2 * dp + (mi >> 1);\n      uint32_t r0, r1, r2, r3;\n      ldsm4t(r0, r1, r2, r3, su(VRs + h * 4096 + off(key, 16 * dg)));\n      uint8_t* base = VTs + h * 4096;\n      const int d0 = 32 * dp + 2 * cc, d1 = d0 + 16, col = 16 * pw + 4 * t;\n      *(uint32_t*)(base + off(d0, col)) = prmt(r0, r1, 0x6420);\n      *(uint32_t*)(base + off(d0 + 1, col)) = prmt(r0, r1, 0x7531);\n      *(uint32_t*)(base + off(d1, col)) = prmt(r2, r3, 0x6420);\n      *(uint32_t*)(base + off(d1 + 1, col)) = prmt(r2, r3, 0x7531);\n    }\n  }\n}\n\nstruct Walk { int po, k, kend, b, hp, seq, lo; };\nDI bool walk_next(const VP& p, const Sched& S, Walk& w, int& b, int& hp, int& seq, int& key0) {\n  while (w.k >= w.kend) {\n    if (++w.po >= S.npo) return false;\n    const int s = S.po_s[w.po], sq = S.segSeq[s];\n    const int lo = p.window ? max(0, sq - p.window + 1) : 0, nc = (sq - lo + 63) >> 6;\n    w.k = S.po_ka[w.po]; w.kend = min(S.po_kb[w.po], nc);\n    w.b = s / p.HP; w.hp = s % p.HP; w.seq = sq; w.lo = lo;\n  }\n  b = w.b; hp = w.hp; seq = w.seq; key0 = w.lo + 64 * w.k; w.k++;\n  return true;\n}\n// Producer timing: LAG tiles between issuing a tile\'s cp.async and publishing it, and PD tiles of page-table\n// lookahead. A consumer step waits for tile rr+1 while it still holds rr (PV of rr overlaps QK of rr+1), so the\n// publish of rr+1 must not wait on the release of rr-1, or the two sides serialize (period = consumer +\n// producer work: 2,700-2,900 cycles per tile in H200 runs c9c4ef99 / d5bcbc7d, with LAG = ST-2 in the old\n// wait-issue-publish order and ST-1 in this publish-first one). Here publish(rr+1) follows the empty wait for\n// tile rr+LAG, i.e. the release of rr+LAG-ST: LAG <= ST-2 keeps one consumer step of slack (LAG = ST-1\n// serializes, LAG = ST deadlocks; scratchpad pipe_time.py). LAG = ST-2 measured ~1,800 cycles (44aa67ba).\ntemplate <int ST> __host__ __device__ constexpr int lag_of() { return ST > 2 ? ST - 2 : 1; }\ntemplate <int ST> __host__ __device__ constexpr int pd_of() { return lag_of<ST>() + 2 > 4 ? lag_of<ST>() + 2 : 4; }\n\n// Page-table entries of the ahead walk\'s next tile into ring slot `slot` (64 ints: key k of the tile at\n// [k]); warp pw fetches its own 16 keys with 4-byte cp.async (keys past the end read the last valid\n// entry). They ride the current commit group: the publish\'s cp.async wait (LAG <= PD) covers them before\n// the tile is issued, PD tiles later. (A register ring, unrolled PD times, cost registers and code: PD 6-8\n// at ST 8 was ~30 us slower at B32 full, d5bcbc7d; with PD = 4 < LAG the page loads, not the KV, bounded\n// the bytes in flight.)\nDI void page_fetch(const VP& p, const Sched& S, Walk& w, int* ring, int slot, int pw, int lane) {\n  int b = 0, hp = 0, seq = 1, key0 = 0;\n  walk_next(p, S, w, b, hp, seq, key0);\n  const int key = min(key0 + 16 * pw + (lane & 15), max(seq - 1, 0));\n  if (lane < 16) cp4(ring + slot * 64 + 16 * pw + lane, p.pt + (int64_t)b * p.pages + (key >> p.lg));\n}\n\ntemplate <int ST>\nDI void producer(const VP& p, const Sched& S, uint8_t* Ks, uint8_t* VR, uint8_t* VT, uint64_t* full, uint64_t* empty,\n                 int* ring) {\n  const int pw = __shfl_sync(0xffffffffu, (threadIdx.x >> 5) - 8, 0), lane = threadIdx.x & 31, q = lane >> 3, c = lane & 7;\n  const int head = c >> 2, cb = (c & 3) * 16;\n  const int64_t rowbytes = (int64_t)p.HK * 64;\n  Walk wm{-1, 0, 0, 0, 0, 0, 0}, wa = wm;\n  // Iteration it publishes tile it-LAG (once its data landed: cp.async groups complete in order), then\n  // waits for tile it\'s slot and issues it (see lag_of).\n  constexpr int LAG = lag_of<ST>(), PD = pd_of<ST>();\n  for (int u = 0; u < PD; u++) page_fetch(p, S, wa, ring, u, pw, lane);\n  commit();\n  cp_wait<0>();\n  __syncwarp();\n  const int nreal = __shfl_sync(0xffffffffu, S.nreal, 0);\n#pragma unroll 1\n  for (int it = 0; it < nreal + LAG; it++) {\n    const int tr = it - LAG;\n    if (tr >= 0 && tr < nreal) {\n      cp_wait<LAG - 1>();  // tile tr\'s K/V and (PD >= LAG) the pages of every tile up to it\n      __syncwarp();\n      CLK_MARK(t0);\n      const int slot = tr % ST;\n      transpose_group(VR + slot * 8192, VT + slot * 8192, pw, lane);\n      proxy_fence();\n      __syncwarp();\n      if (lane == 0) mb_arrive(&full[slot]);\n      CLK_ADD(256, 52, t0);\n    }\n    if (it < nreal) {\n      int b, hp, seq, key0;\n      walk_next(p, S, wm, b, hp, seq, key0);\n      const int slot = it % ST, ps = it % PD;\n      if (it >= ST) mb_wait(&empty[slot], ((it / ST) - 1) & 1);\n      CLK_MARK(t1);\n      uint8_t* kd = Ks + slot * 8192 + head * 4096;\n      uint8_t* vd = VR + slot * 8192 + head * 4096;\n      const int64_t hoff = (int64_t)hp * 128 + c * 16;\n#pragma unroll\n      for (int i = 0; i < 4; i++) {\n        const int kk = 16 * pw + 4 * i + q;\n        const int page = ring[ps * 64 + kk], key = key0 + kk;\n        const bool valid = key < seq;\n        const int64_t src = ((int64_t)page * p.ps + (key & (p.ps - 1))) * rowbytes + hoff;\n        cp16(kd + off(kk, cb), p.kc + src, valid);\n        cp16(vd + off(kk, cb), p.vc + src, valid);\n      }\n      __syncwarp();  // every lane has read slot ps before it is refilled\n      page_fetch(p, S, wa, ring, ps, pw, lane);\n      CLK_ADD(256, 53, t1);\n    }\n    commit();\n  }\n}\n\n// Online softmax of one 64-key tile; P x256 packed in place into the PV A fragment.\n// (ex2.approx.f16x2 lowers to two MUFU.EX2.F16 on sm_90, so FP32 exponentials cost the same MUFU work.)\ntemplate <bool MASK>\nDI void softmax(float (&s)[8][4], float (&m)[2], float (&l)[2], float (&alpha)[2], uint32_t (&pk)[8], float sc,\n                int kbase, const int (&lo)[2], const int (&hi)[2], bool live) {\n  const int t = threadIdx.x & 3;\n  if (!live) {  // q_len 1: this warp\'s 16 query rows are padding\n    alpha[0] = alpha[1] = 1.f;\n#pragma unroll\n    for (int i = 0; i < 8; i++) pk[i] = 0u;\n    return;\n  }\n  if constexpr (MASK) {\n#pragma unroll\n    for (int rh = 0; rh < 2; rh++) {\n      const int a = lo[rh] - kbase - 2 * t, z = hi[rh] - kbase - 2 * t;  // valid: a <= 8n+e < z\n#pragma unroll\n      for (int n = 0; n < 8; n++)\n#pragma unroll\n        for (int e = 0; e < 2; e++)\n          if (8 * n + e < a || 8 * n + e >= z) s[n][2 * rh + e] = -INFINITY;\n    }\n  }\n  float mx[2];\n#pragma unroll\n  for (int rh = 0; rh < 2; rh++) {\n    float v[8];\n#pragma unroll\n    for (int n = 0; n < 8; n++) v[n] = fmaxf(s[n][2 * rh], s[n][2 * rh + 1]);\n#pragma unroll\n    for (int n = 0; n < 4; n++) v[n] = fmaxf(v[n], v[n + 4]);\n    mx[rh] = fmaxf(fmaxf(v[0], v[2]), fmaxf(v[1], v[3]));\n  }\n#pragma unroll\n  for (int rh = 0; rh < 2; rh++) mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 1));\n#pragma unroll\n  for (int rh = 0; rh < 2; rh++) mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 2));\n#pragma unroll\n  for (int rh = 0; rh < 2; rh++) {\n    const float mn = fmaxf(m[rh], mx[rh] * sc);\n    const float safe = mn == -INFINITY ? 0.f : mn;\n    alpha[rh] = ex2(m[rh] - safe);\n    m[rh] = mn;\n    const float bias = 8.f - safe;  // P x 256 uses E4M3\'s range below 1\n    float sum0 = 0.f, sum1 = 0.f;\n#pragma unroll\n    for (int n = 0; n < 8; n++) {\n      const float p0 = ex2(fmaf(s[n][2 * rh], sc, bias)), p1 = ex2(fmaf(s[n][2 * rh + 1], sc, bias));\n      s[n][2 * rh] = p0;\n      s[n][2 * rh + 1] = p1;\n      sum0 += p0;\n      sum1 += p1;\n    }\n    l[rh] = l[rh] * alpha[rh] + (sum0 + sum1);\n  }\n  // A fragment for PV chunk kk: reg 4kk+2hf (row g) / 4kk+2hf+1 (row g+8) hold n-blocks 4kk+2hf, 4kk+2hf+1.\n#pragma unroll\n  for (int kk = 0; kk < 2; kk++)\n#pragma unroll\n    for (int hf = 0; hf < 2; hf++) {\n      const int n0 = 4 * kk + 2 * hf;\n      pk[4 * kk + 2 * hf] = pack4(s[n0][0], s[n0][1], s[n0 + 1][0], s[n0 + 1][1]);\n      pk[4 * kk + 2 * hf + 1] = pack4(s[n0][2], s[n0][3], s[n0 + 1][2], s[n0 + 1][3]);\n    }\n}\nDI void rescale(float (&O)[8][4], const float (&alpha)[2]) {\n  if (__any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {\n#pragma unroll\n    for (int n = 0; n < 8; n++) {\n      O[n][0] *= alpha[0]; O[n][1] *= alpha[0]; O[n][2] *= alpha[1]; O[n][3] *= alpha[1];\n    }\n  }\n}\n\n// A portion\'s prologue: the warp\'s 16 query rows (RMSNorm, RoPE, E4M3 into Q) and, when the portion\n// ends with the segment\'s virtual tile, its 2 of the block\'s L new keys (RMSNorm, RoPE, E4M3 into the\n// cache and into KN / VN, already zeroed). All global loads go out together.\nDI void prologue(const VP& p, uint8_t* Q, uint8_t* KN, uint8_t* VN, int b, int hp, int hh, int seq, int w, int lane,\n                 bool virt, float ksc, float vsc) {\n  constexpr int N = 20;\n  int tk[N], cl[N];\n  bool vd[N];\n  const int kvh = 2 * hp + hh;\n#pragma unroll\n  for (int i = 0; i < 16; i++) {\n    const int row = 16 * w + i, tok = row >> 3;\n    vd[i] = tok < p.L;\n    tk[i] = b * p.L + (vd[i] ? tok : 0);\n    cl[i] = ((2 * hp + hh) * 8 + (row & 7)) * 64 + 2 * lane;\n  }\n#pragma unroll\n  for (int tt = 0; tt < 2; tt++) {\n    const int tok = 2 * w + tt;\n    const bool ok = virt && tok < p.L;\n    vd[16 + tt] = vd[18 + tt] = ok;\n    tk[16 + tt] = tk[18 + tt] = b * p.L + (ok ? tok : 0);\n    cl[16 + tt] = (p.HQ + kvh) * 64 + 2 * lane;\n    cl[18 + tt] = (p.HQ + p.HK + kvh) * 64 + 2 * lane;\n  }\n  const int64_t row0 = p.loc[tk[16]], row1 = p.loc[tk[17]];\n  const float invf = p.window ? p.inv_freq[lane] : 0.f;\n  float2 x[N];\n  input_pairs<N>(p, tk, cl, vd, x);\n#if VQ8_CLK\n  if (threadIdx.x == 0 && !clk_area()[44]) clk_area()[44] = gtime() | (unsigned long long)(x[0].x != 12345.f) << 63;\n#endif\n  // The warp\'s rows hold tokens 2w (rows 0-7) and 2w+1 (rows 8-15); full layers have no RoPE (identity).\n  float2 cs[2];\n#pragma unroll\n  for (int tt = 0; tt < 2; tt++) cs[tt] = p.window ? rope_cs((float)(seq + 2 * w + tt) * invf) : make_float2(1.f, 0.f);\n#pragma unroll\n  for (int i = 0; i < 16; i++) {\n    const int row = 16 * w + i;\n    const float n = rsqrtf(wsum(x[i].x * x[i].x + x[i].y * x[i].y) * (1.f / 64.f) + 1e-5f);  // 0 rows stay 0\n    const float2 y = rot(make_float2(x[i].x * n, x[i].y * n), cs[i >> 3]);\n    *(uint16_t*)(Q + off(row, 2 * lane)) = fp8pair(y.x, y.y);\n  }\n#pragma unroll\n  for (int tt = 0; tt < 2; tt++) {\n    const int tok = 2 * w + tt;\n    if (!vd[16 + tt]) break;\n    float2 k = x[16 + tt];\n    const float2 v = x[18 + tt];\n    const float n = rsqrtf(wsum(k.x * k.x + k.y * k.y) * (1.f / 64.f) + 1e-5f);\n    k = rot(make_float2(k.x * n, k.y * n), cs[tt]);\n    const uint16_t kq = fp8pair(k.x / ksc, k.y / ksc), vq = fp8pair(v.x / vsc, v.y / vsc);\n    const int64_t row = tt ? row1 : row0;\n    *(uint16_t*)(p.kc + (row * p.HK + kvh) * 64 + 2 * lane) = kq;\n    *(uint16_t*)(p.vc + (row * p.HK + kvh) * 64 + 2 * lane) = vq;\n    *(uint16_t*)(KN + off(tok, 2 * lane)) = kq;\n    const int pos = 4 * (tok >> 1) + (tok & 1);\n    VN[off(2 * lane, pos)] = (uint8_t)(vq & 0xff);\n    VN[off(2 * lane + 1, pos)] = (uint8_t)(vq >> 8);\n  }\n}\n\ntemplate <int ST, bool MASK>\nDI void step(const VP& p, uint8_t* Q, uint8_t* Ks, uint8_t* VT, uint64_t* full, uint64_t* empty, int hh, int rr,\n             float (&O)[8][4], float (&sx)[8][4], uint32_t (&pc)[8], uint32_t (&pn)[8], float (&m)[2], float (&l)[2],\n             float sc, int kbase, const int (&lo)[2], const int (&hi)[2]) {\n  // PV of ring tile rr (P in pc) overlaps QK and softmax of rr+1 (P into pn). No register copies of P:\n  // a copy into a live wgmma operand makes ptxas serialize every wgmma.\n  const int lane = threadIdx.x & 31;\n  float alpha[2];\n  mb_wait(&full[(rr + 1) % ST], ((rr + 1) / ST) & 1);\n  CLK_MARK(t0);\n  protect(O);\n  protect(pc);\n  protect(sx);\n  wg_fence();\n  qk(sx, Q, Ks + ((rr + 1) % ST) * 8192 + hh * 4096);\n  wg_commit();\n  wg_fence();\n  pv(O, pc, VT + (rr % ST) * 8192 + hh * 4096);\n  wg_commit();\n  CLK_ADD(0, 51, t0);\n  CLK_MARK(t1);\n  wg_wait<1>();\n  protect(sx);\n  CLK_ADD(0, 48, t1);\n  CLK_MARK(t2);\n  softmax<MASK>(sx, m, l, alpha, pn, sc, kbase, lo, hi, 2 * ((threadIdx.x >> 5) & 3) < p.L);\n  CLK_ADD(0, 49, t2);\n  CLK_MARK(t3);\n  wg_wait<0>();\n  protect(O);\n  CLK_ADD(0, 50, t3);\n  protect(pc);\n  __syncwarp();\n  if (lane == 0) mb_arrive(&empty[rr % ST]);\n  rescale(O, alpha);\n}\n\n// The portion\'s last steps: an optional masked step (segment\'s last cached tile), then the final PV.\ntemplate <int ST>\nDI void finish(const VP& p, uint8_t* Q, uint8_t* Ks, uint8_t* VT, uint64_t* full, uint64_t* empty, int hh, int rlast,\n               bool lastmask, float (&O)[8][4], float (&sx)[8][4], uint32_t (&pc)[8], uint32_t (&pn)[8], float (&m)[2],\n               float (&l)[2], float sc, int kbase, const int (&lo)[2], const int (&hi)[2]) {\n  const int lane = threadIdx.x & 31;\n  if (lastmask) {\n    step<ST, true>(p, Q, Ks, VT, full, empty, hh, rlast - 1, O, sx, pc, pn, m, l, sc, kbase, lo, hi);\n    protect(O);\n    protect(pn);\n    wg_fence();\n    pv(O, pn, VT + (rlast % ST) * 8192 + hh * 4096);\n    wg_commit();\n    wg_wait<0>();\n    protect(O);\n    protect(pn);\n  } else {\n    protect(O);\n    protect(pc);\n    wg_fence();\n    pv(O, pc, VT + (rlast % ST) * 8192 + hh * 4096);\n    wg_commit();\n    wg_wait<0>();\n    protect(O);\n    protect(pc);\n  }\n  __syncwarp();\n  if (lane == 0) mb_arrive(&empty[rlast % ST]);\n}\n\ntemplate <int ST>\nDI void consumer(const VP& p, const Sched& S, uint8_t* Qs, uint8_t* Kn, uint8_t* VTn, uint8_t* Ks, uint8_t* VT,\n                 uint64_t* full, uint64_t* empty) {\n  const int hh = __shfl_sync(0xffffffffu, threadIdx.x >> 7, 0);\n  const int w = __shfl_sync(0xffffffffu, (threadIdx.x >> 5) & 3, 0);\n  const int wtid = threadIdx.x & 127, lane = threadIdx.x & 31;\n  const int g = lane >> 2, t = lane & 3, bar = 1 + hh;\n  uint8_t* Q = Qs + hh * 4096;\n  uint8_t* KN = Kn + hh * 4096;\n  uint8_t* VN = VTn + hh * 4096;\n  const float ksc = *p.ks, vsc = *p.vs, sc = p.sc * ksc;\n  const int G = S.G, T = S.T, cta = blockIdx.x;\n  const int npo = __shfl_sync(0xffffffffu, S.npo, 0);\n  const bool live = 2 * w < p.L;\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n#if VQ8_CLK\n  if (threadIdx.x == 0) clk_area()[40] = gtime();\n#endif\n  int r = 0;\n  for (int po = 0; po < npo; po++) {\n    const int s = __shfl_sync(0xffffffffu, S.po_s[po], 0);\n    const int ka = __shfl_sync(0xffffffffu, S.po_ka[po], 0), kb = __shfl_sync(0xffffffffu, S.po_kb[po], 0);\n    const int seq = __shfl_sync(0xffffffffu, S.segSeq[s], 0);\n    const int b = s / p.HP, hp = s % p.HP;\n    const int lo = p.window ? max(0, seq - p.window + 1) : 0, nc = (seq - lo + 63) >> 6;\n    const int nr = max(0, min(kb, nc) - ka);\n    const bool virt = kb > nc;\n    int lo_r[2], hi_r[2];\n#pragma unroll\n    for (int rh = 0; rh < 2; rh++) { lo_r[rh] = p.window ? seq + 2 * w + rh - p.window + 1 : INT_MIN / 2; hi_r[rh] = seq; }\n    wg_bar(bar);  // the previous portion is done with Q, KN and VN\n#if VQ8_CLK\n    if (threadIdx.x == 0 && !clk_area()[42]) clk_area()[42] = gtime();\n#endif\n    if (virt)\n      for (int i = wtid; i < 256; i += 128) {\n        *(uint4*)(KN + 16 * i) = make_uint4(0, 0, 0, 0);\n        *(uint4*)(VN + 16 * i) = make_uint4(0, 0, 0, 0);\n      }\n    wg_bar(bar);\n#if VQ8_CLK\n    if (threadIdx.x == 0 && !clk_area()[43]) clk_area()[43] = gtime();\n#endif\n    prologue(p, Q, KN, VN, b, hp, hh, seq, w, lane, virt, ksc, vsc);\n    proxy_fence();\n    wg_bar(bar);\n#if VQ8_CLK\n    if (threadIdx.x == 0 && !clk_area()[41]) clk_area()[41] = gtime();\n#endif\n    float O[8][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, alpha[2];\n#pragma unroll\n    for (int n = 0; n < 8; n++)\n#pragma unroll\n      for (int e = 0; e < 4; e++) O[n][e] = 0.f;\n    if (nr > 0) {\n      uint32_t pa[8], pb[8];\n      float sx[8][4];\n      mb_wait(&full[r % ST], (r / ST) & 1);\n      protect(sx);\n      wg_fence();\n      qk(sx, Q, Ks + (r % ST) * 8192 + hh * 4096);\n      wg_commit();\n      wg_wait<0>();\n      protect(sx);\n      if ((ka == 0 && p.window) || ka == nc - 1) softmax<true>(sx, m, l, alpha, pa, sc, lo + 64 * ka, lo_r, hi_r, live);\n      else softmax<false>(sx, m, l, alpha, pa, sc, lo + 64 * ka, lo_r, hi_r, live);\n      // Steps j: PV(j) with QK/softmax(j+1). Only the segment\'s last cached tile needs the key mask.\n      const bool lastmask = nr >= 2 && ka + nr == nc;\n      const int nplain = nr - 1 - (lastmask ? 1 : 0);\n      int j = 0;\n      for (; j + 2 <= nplain; j += 2) {\n        step<ST, false>(p, Q, Ks, VT, full, empty, hh, r + j, O, sx, pa, pb, m, l, sc, lo + 64 * (ka + j + 1), lo_r, hi_r);\n        step<ST, false>(p, Q, Ks, VT, full, empty, hh, r + j + 1, O, sx, pb, pa, m, l, sc, lo + 64 * (ka + j + 2), lo_r, hi_r);\n      }\n      const int kb2 = lo + 64 * (nc - 1);\n      if (nplain & 1) {\n        step<ST, false>(p, Q, Ks, VT, full, empty, hh, r + j, O, sx, pa, pb, m, l, sc, lo + 64 * (ka + j + 1), lo_r, hi_r);\n        finish<ST>(p, Q, Ks, VT, full, empty, hh, r + nr - 1, lastmask, O, sx, pb, pa, m, l, sc, kb2, lo_r, hi_r);\n      } else {\n        finish<ST>(p, Q, Ks, VT, full, empty, hh, r + nr - 1, lastmask, O, sx, pa, pb, m, l, sc, kb2, lo_r, hi_r);\n      }\n    }\n    r += nr;\n    if (virt) {\n      // The block\'s own L keys (prepared in the prologue; this CTA alone writes them to the cache).\n      float sx[8][4];\n      uint32_t pk[8];\n      protect(sx);\n      wg_fence();\n      qk(sx, Q, KN);\n      wg_commit();\n      wg_wait<0>();\n      protect(sx);\n      const int hi2[2] = {seq + 2 * w + 1, seq + 2 * w + 2};\n      softmax<true>(sx, m, l, alpha, pk, sc, seq, lo_r, hi2, live);\n      rescale(O, alpha);\n      protect(O);\n      protect(pk);\n      wg_fence();\n      pv(O, pk, VN);\n      wg_commit();\n      wg_wait<0>();\n      protect(O);\n    }\n    const int slot = S.partB[s] + (cta - cta_of(S.segP[s], G, T));\n#pragma unroll\n    for (int rh = 0; rh < 2; rh++) {\n      float lt = l[rh];\n      lt += __shfl_xor_sync(0xffffffffu, lt, 1);\n      lt += __shfl_xor_sync(0xffffffffu, lt, 2);\n      float* dst = p.part + ((int64_t)slot * 128 + hh * 64 + w * 16 + g + 8 * rh) * PR;\n#pragma unroll\n      for (int n = 0; n < 8; n++) *(float2*)(dst + 8 * n + 2 * t) = make_float2(O[n][2 * rh], O[n][2 * rh + 1]);\n      if (t == 0) *(float2*)(dst + 64) = make_float2(m[rh], lt);\n    }\n  }\n}\n\ntemplate <int ST>\n__global__ void __launch_bounds__(384, 1) vq8_main(VP p) {\n  extern __shared__ __align__(1024) uint8_t sm[];\n  uint8_t* Qs = sm + CLKPAD;\n  uint8_t* Kn = Qs + 8192;\n  uint8_t* VTn = Qs + 16384;\n  uint8_t* Ks = Qs + 24576;\n  uint8_t* VR = Ks + ST * 8192;\n  uint8_t* VT = VR + ST * 8192;\n  __shared__ __align__(16) uint64_t full[ST], empty[ST];\n  __shared__ __align__(16) int pages[pd_of<ST>() * 64];\n  __shared__ Sched S;\n  // Dependents (the combine) may launch now; their griddepcontrol.wait still waits for this grid to finish.\n  asm volatile("griddepcontrol.launch_dependents;");\n  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0);\n#if VQ8_CLK\n  // Row per CTA: 0 entry, 1 scheduled, 2 WG0 first tile, 3 WG0 end, 4 WG0 wait cyc, 5 WG0 cyc, 6 producer\n  // end, 7 producer empty-wait cyc, 8 producer cp-wait cyc, 9 producer cyc, 10 tiles, 11 WG1 wait cyc,\n  // 12 portions, 13 WG1 end, 14-18 WG0 prologue stamps, 19-22 WG0 QK wait / softmax / PV wait / issue cyc,\n  // 23-24 producer publish / issue cyc (globaltimer ns; cyc = SM clocks).\n  unsigned long long* row = clock_rows(p) + blockIdx.x * 32;\n  if (tid == 0) row[0] = gtime();\n  if (tid < 64) clk_area()[tid] = 0;\n#endif\n  if (warp == 0) {\n    schedule_warp(p, S);\n    // This CTA\'s portions (segments overlapping its tile range), compacted by ballot.\n    const int lane = tid & 31, G = S.G, T = S.T, i = blockIdx.x;\n    const int t0 = (int)((long long)i * T / G), t1 = (int)((long long)(i + 1) * T / G);\n    int npo = 0, nreal = 0;\n    for (int base = 0; base < p.NS; base += 32) {\n      const int s = base + lane;\n      const int a = s < p.NS ? S.segP[s] : INT_MAX, e = s < p.NS ? S.segP[s + 1] : INT_MAX;\n      const bool hit = i < G && e > t0 && a < t1;\n      const unsigned m = __ballot_sync(0xffffffffu, hit);\n      if (hit) {\n        const int ka = max(t0, a) - a, kb = min(t1, e) - a, nc = e - a - 1, at = npo + __popc(m & ((1u << lane) - 1));\n        S.po_s[at] = s; S.po_ka[at] = ka; S.po_kb[at] = kb;\n        nreal += max(0, min(kb, nc) - ka);\n      }\n      npo += __popc(m);\n    }\n    nreal = wsum_i(nreal);\n    if (lane == 0) { S.npo = npo; S.nreal = nreal; }\n    if (i == 0)  // the combine reads the partial-slot prefix after its griddepcontrol.wait (this grid done)\n      for (int s = lane; s <= p.NS; s += 32) sched_area(p)[s] = S.partB[s];\n  }\n  if (tid < ST) { mb_init(&full[tid], 4); mb_init(&empty[tid], 8); }\n  __syncthreads();\n  if ((int)blockIdx.x >= S.G) {\n#if VQ8_CLK\n    if (tid == 0) row[10] = ~0ull;\n#endif\n    return;\n  }\n#if VQ8_CLK\n  if (tid == 0) { row[1] = gtime(); row[10] = S.nreal; row[12] = S.npo; }\n  const long long c0 = clock64();\n#endif\n  if (warp >= 8) {\n    producer<ST>(p, S, Ks, VR, VT, full, empty, pages);\n#if VQ8_CLK\n    if (tid == 256) {\n      const unsigned long long* a = clk_area();\n      row[6] = gtime(); row[7] = a[8]; row[8] = a[32]; row[9] = clock64() - c0; row[23] = a[52]; row[24] = a[53];\n    }\n#endif\n    return;\n  }\n  consumer<ST>(p, S, Qs, Kn, VTn, Ks, VT, full, empty);\n#if VQ8_CLK\n  if (tid == 0) {\n    const unsigned long long* a = clk_area();\n    row[2] = a[12]; row[3] = gtime(); row[4] = a[0]; row[5] = clock64() - c0; row[14] = a[40]; row[15] = a[41];\n    row[16] = a[42]; row[17] = a[43]; row[18] = a[44] & ~(1ull << 63);\n    row[19] = a[48]; row[20] = a[49]; row[21] = a[50]; row[22] = a[51];\n  }\n  if (tid == 128) { row[11] = clk_area()[4]; row[13] = gtime(); }\n#endif\n}\n\n// wpr warps per output row (8 / wpr rows per CTA): merge the row\'s partials, RMSNorm, sigmoid gate.\n// 4 CTAs per SM (<= 64 registers): one wave for every scored shape (B 32, q_len 8: 512 CTAs).\n__global__ void __launch_bounds__(256, 4) vq8_combine(VP p) {\n  asm volatile("griddepcontrol.launch_dependents;");\n#if VQ8_CLK\n  // Combine rows follow the main grid\'s: 0 entry, 1 after griddepcontrol.wait, 2 end (thread 0).\n  unsigned long long* crow = clock_rows(p) + (p.grid + blockIdx.y * gridDim.x + blockIdx.x) * 32;\n  if (threadIdx.x == 0) crow[0] = gtime();\n#endif\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n#if VQ8_CLK\n  if (threadIdx.x == 0) crow[1] = gtime();\n#endif\n  const int s = blockIdx.y, lane = threadIdx.x & 31;\n  const int gw = blockIdx.x * 8 + (threadIdx.x >> 5), rr = gw / p.wpr, sub = gw % p.wpr;  // row\'s warps: one block\n  const int r = (rr >> 3) / p.L * 64 + (rr >> 3) % p.L * 8 + (rr & 7);  // real rows only\n  const int pb = sched_area(p)[s], np = sched_area(p)[s + 1] - pb;\n  const int k_lo = sub * np / p.wpr, k_hi = (sub + 1) * np / p.wpr;\n  const float* base = p.part + ((int64_t)pb * 128 + r) * PR;\n  const int b = s / p.HP, hp = s % p.HP, hh = r >> 6, row = r & 63, tok = row >> 3;\n  const int head = (2 * hp + hh) * 8 + (row & 7);\n  const float vsc = *p.vs;\n  float2 gate = make_float2(0.f, 0.f);  // read only with p.gate\n  if (p.gate) {\n    const int tk[1] = {b * p.L + tok}, cl[1] = {(p.HQ + 2 * p.HK + head) * 64 + 2 * lane};\n    const bool vd[1] = {true};\n    float2 g1[1];\n    input_pairs<1>(p, tk, cl, vd, g1);\n    gate = g1[0];\n  }\n  // This warp\'s slice of the row\'s partials, CH at a time with all their loads in flight (online max).\n  constexpr int CH = 8;\n  float mx = -INFINITY, total = 0.f, x = 0.f, y = 0.f;\n  for (int k0 = k_lo; k0 < k_hi; k0 += CH) {\n    float2 ov[CH], ml[CH];\n#pragma unroll\n    for (int k = 0; k < CH; k++) {\n      const float* pp = base + (int64_t)min(k0 + k, np - 1) * 128 * PR;\n      ov[k] = *(const float2*)(pp + 2 * lane);\n      ml[k] = *(const float2*)(pp + 64);\n    }\n    float cm = mx;\n#pragma unroll\n    for (int k = 0; k < CH; k++)\n      if (k0 + k < k_hi) cm = fmaxf(cm, ml[k].x);\n    const float sa = mx == -INFINITY ? 0.f : ex2(mx - cm);\n    total *= sa; x *= sa; y *= sa;\n#pragma unroll\n    for (int k = 0; k < CH; k++) {\n      const float wgt = (k0 + k < k_hi && ml[k].x != -INFINITY) ? ex2(ml[k].x - cm) : 0.f;\n      total = fmaf(wgt, ml[k].y, total);\n      x = fmaf(wgt, ov[k].x, x);\n      y = fmaf(wgt, ov[k].y, y);\n    }\n    mx = cm;\n  }\n  if (p.wpr > 1) {  // merge the row\'s warps through shared memory; its first warp finishes the row\n    __shared__ float4 red[8][32];\n    red[threadIdx.x >> 5][lane] = make_float4(mx, total, x, y);\n    __syncthreads();\n    if (sub) return;\n    const int w0 = threadIdx.x >> 5;\n    float cm = mx;\n    for (int j = 1; j < p.wpr; j++) cm = fmaxf(cm, red[w0 + j][lane].x);\n    const float sa = mx == -INFINITY ? 0.f : ex2(mx - cm);\n    total *= sa; x *= sa; y *= sa;\n    for (int j = 1; j < p.wpr; j++) {\n      const float4 v = red[w0 + j][lane];\n      const float wgt = v.x == -INFINITY ? 0.f : ex2(v.x - cm);\n      total = fmaf(wgt, v.y, total);\n      x = fmaf(wgt, v.z, x);\n      y = fmaf(wgt, v.w, y);\n    }\n  }\n  const float scl = vsc / total;  // P and its row sum both carry the x256\n  x *= scl; y *= scl;\n  const float norm = rsqrtf(wsum(x * x + y * y) * (1.f / 64.f) + 1e-5f);\n  *(__nv_bfloat162*)(p.out + ((int64_t)(b * p.L + tok) * p.HQ + head) * 64 + 2 * lane) =\n      p.gate ? __floats2bfloat162_rn(x * norm / (1.f + __expf(-gate.x)), y * norm / (1.f + __expf(-gate.y)))\n             : __floats2bfloat162_rn(x * norm, y * norm);\n#if VQ8_CLK\n  if (threadIdx.x == 0) crow[2] = gtime();\n#endif\n}\n\ntemplate <int ST> constexpr int smem_bytes() { return CLKPAD + 24576 + ST * 24576; }\n\ntemplate <int ST>\nvoid launch(const VP& p, cudaStream_t stream, bool pdl) {\n  constexpr int smem = smem_bytes<ST>();\n  cudaFuncSetAttribute(vq8_main<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n  cudaLaunchAttribute attr[1];\n  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n  attr[0].val.programmaticStreamSerializationAllowed = 1;\n  {\n    // With pdl, the producer starts gathering the cache while the preceding kernel (the QKV\n    // projection) finishes; consumers read qkv only after griddepcontrol.wait.\n    cudaLaunchConfig_t cfg = {};\n    cfg.gridDim = dim3(p.grid);\n    cfg.blockDim = dim3(384);\n    cfg.dynamicSmemBytes = smem;\n    cfg.stream = stream;\n    cfg.attrs = attr;\n    cfg.numAttrs = pdl ? 1 : 0;\n    cudaLaunchKernelEx(&cfg, vq8_main<ST>, p);\n  }\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(2 * p.L * p.wpr, p.NS);\n  cfg.blockDim = dim3(256);\n  cfg.dynamicSmemBytes = 0;\n  cfg.stream = stream;\n  cfg.attrs = attr;\n  cfg.numAttrs = 1;\n  cudaLaunchKernelEx(&cfg, vq8_combine, p);\n}\n\ntemplate <int ST>\nstd::vector<int64_t> resources() {\n  cudaFuncAttributes a;\n  cudaFuncGetAttributes(&a, vq8_main<ST>);\n  constexpr int smem = smem_bytes<ST>();\n  cudaFuncSetAttribute(vq8_main<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n  int active = 0;\n  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, vq8_main<ST>, 384, smem);\n  cudaGetLastError();  // a stage count over this device\'s shared-memory limit reports 0 CTAs, not a sticky error\n  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, smem, active};\n}\n}  // namespace\n\nvoid vq8_forward(int64_t qkv, int64_t loc, int64_t pt, int64_t seq, int64_t kc, int64_t vc, int64_t ks, int64_t vs,\n                 int64_t inv_freq, int64_t part, int64_t out, int64_t B, int64_t L, int64_t HQ, int64_t HK, int64_t stride,\n                 int64_t pages, int64_t ps, int64_t window, int64_t proj_splits, int64_t min_tiles, int64_t grid,\n                 int64_t stages, double sc, int64_t split_stride, int64_t flags, int64_t stream) {\n  // flags: bit 0 PDL launch (after the QKV projection), bit 1 no gate (RMSNorm only).\n  const bool pdl = flags & 1;\n  VP p{(const void*)qkv, (const int64_t*)loc, (const int*)pt, (const int*)seq, (uint8_t*)kc, (uint8_t*)vc,\n       (const float*)ks, (const float*)vs, (const float*)inv_freq, (float*)part, (__nv_bfloat16*)out,\n       (int)B, (int)L, (int)HQ, (int)HK, (int)(HK / 2), (int)(B * (HK / 2)), (int)stride, (int)pages, (int)ps, 0,\n       (int)window, (int)proj_splits, (int)min_tiles, (int)grid, 1, !(flags & 2), split_stride, (float)sc};\n  // A segment has at most grid/NS + 2 partials; give each combine warp about 16 of them.\n  const int np_max = (int)(grid / p.NS) + 2;\n  while (p.wpr < 8 && p.wpr * 16 < np_max) p.wpr *= 2;\n  if (HK % 2 || HQ != 8 * HK || p.NS > NSMAX || (L != 1 && L != 8)) throw std::runtime_error("vq8: unsupported shape");\n  while ((1 << p.lg) < ps) p.lg++;\n  auto st = (cudaStream_t)stream;\n  switch (stages) {\n    case 2: launch<2>(p, st, pdl); break;\n    case 4: launch<4>(p, st, pdl); break;\n    case 6: launch<6>(p, st, pdl); break;\n    case 8: launch<8>(p, st, pdl); break;\n    default: throw std::runtime_error("vq8: stages must be 2, 4, 6 or 8");\n  }\n  auto err = cudaGetLastError();\n  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));\n}\n\nstd::vector<int64_t> vq8_resources(int64_t stages) {\n  if (stages == 2) return resources<2>();\n  if (stages == 4) return resources<4>();\n  if (stages == 6) return resources<6>();\n  return resources<8>();\n}\n'
VQ8_CPP = r"""
#include <cstdint>
#include <vector>
void vq8_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
 int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t,int64_t);
std::vector<int64_t> vq8_resources(int64_t);
"""
_VQ8_MOD = None
_VQ8_ROWS = {(16, 8, False): (6, 8), (32, 8, False): (6, 8)}


def _vq8_module():
    global _VQ8_MOD
    if _VQ8_MOD is None:
        major, minor = torch.cuda.get_device_capability()
        arch = "90a" if major == 9 else f"{major}{minor}"
        _VQ8_MOD = load_inline(name=f"muse_vq8_c262e4c5f9c3_{arch}", cpp_sources=VQ8_CPP, cuda_sources=VQ8_CUDA,
                               functions=["vq8_forward", "vq8_resources"],
                               extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"], verbose=False)
    return _VQ8_MOD


def _vq8_attention(qkv, data, setting):
    """vq8 on the layer ABI (see dev/vq8/patch_vq8_verify.py): BF16 [T, Q_SIZE], new K/V written to the cache."""
    partial, splits = qkv
    cfg = data['config']
    batch, length, window = cfg['batch'], cfg['q_len'], cfg['window']
    device = partial.device
    stages, min_tiles = setting if torch.cuda.get_device_capability(device)[0] == 9 else (2, setting[1])
    if partial.dtype == torch.bfloat16:
        assert splits == 1
        raw, nsplit, sstride = (partial[0] if partial.dim() == 3 else partial), 0, 0
    else:
        assert partial.dtype == torch.float32 and partial.stride(-1) == 1
        raw, nsplit, sstride = partial, splits, partial.stride(0)
    grid = _num_sms(device)
    segments = batch * (HKV // 2)
    part = torch.empty(((grid + segments) * 128 * 68 + 256,), device=device, dtype=torch.float32)
    out = torch.empty((batch * length, Q_SIZE), device=device, dtype=torch.bfloat16)
    invf = _inv_freq(cfg['rope_theta'], device) if window else None
    _vq8_module().vq8_forward(
        raw.data_ptr(), data['cache_loc'].data_ptr(), data['page_table'].data_ptr(), data['seq_lens'].data_ptr(),
        data['k_cache'].data_ptr(), data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        invf.data_ptr() if invf is not None else 0, part.data_ptr(), out.data_ptr(), batch, length, HQ, HKV,
        raw.stride(-2), data['page_table'].shape[1], cfg['page_size'], window, nsplit, min_tiles, grid, stages,
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634, sstride, 1,
        torch.cuda.current_stream(device).cuda_stream)
    return out


def _vq8_prepare(device):
    """Rows routed to vq8 once it passes the layer's own attention gate (new K/V and output) on them."""
    errors = []
    try:
        for (batch, length, sliding), setting in _VQ8_ROWS.items():
            for ctx in (257, 2305):
                qkv, data = _attn_port_synthetic(batch, length, 2048 if sliding else 0, device, ctx)
                got = _vq8_attention(qkv, data, setting)
                _attention_store_agrees(qkv, data)
                ok, error, bound = _attention_agrees(got, _attention_reference(qkv, data),
                                                     _attention_reference(qkv, data, True))
                if not ok:
                    raise RuntimeError(f'vq8 {(batch, length, sliding)} ctx {ctx}: {error:.5f} > {bound:.5f}')
                errors.append(round(error, 5))
    except Exception as exc:
        print('VQ8_REJECT', str(exc).splitlines()[0][:200], flush=True)
        return {}
    print('VQ8_ROWS', _VQ8_ROWS, 'errors', errors, flush=True)
    return _VQ8_ROWS


PARTIAL_CUDA = '\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <stdint.h>\n#include <cooperative_groups.h>\n#include <stdexcept>\n\nnamespace cg = cooperative_groups;\n\n#define DEVI __device__ __forceinline__\n\nnamespace {\n\nconstexpr int D = 64;      // head dim\nconstexpr int G = 8;       // query heads per KV head\nconstexpr int T = 64;      // tokens per pipeline tile\nconstexpr int NTHR = 128;  // 4 warps\nconstexpr int ROWB = 128;  // bytes of one token\'s K (or V) for a pair of KV heads\nconstexpr int TILE_BYTES = T * ROWB;\n\nstruct Params {\n  const __nv_bfloat16* qkv;\n  const int64_t* positions;\n  const int64_t* cache_loc;\n  const int* page_table;\n  const int* seq_lens;\n  uint8_t* k_cache;\n  uint8_t* v_cache;\n  const float* k_scale;\n  const float* v_scale;\n  const float* inv_freq;   // nullptr: no RoPE\n  float* part_o;           // [B, KVG, NC, L*16, 64]   cluster partials (NC > 1 only)\n  float* part_ml;          // [B, KVG, NC, L*16, 2]\n  int* counters;           // [B, KVG, CS] arrivals per cluster rank, left at zero\n  float* part1;            // [B, KVG, NS, NSL, L*16, 68] split partials when dsmem == 0\n  int dsmem;               // 1: split partials merge through distributed shared memory\n  __nv_bfloat16* out;      // [B, L, HQ*64]\n  int batch;\n  int qkv_stride;\n  int proj_splits;\n  int gated;\n  int64_t split_stride;\n  int hq, hkv;\n  int max_pages;\n  int ps, ps_log2;\n  int window;\n  int ns;                  // splits per request = NC * CS\n  int cs;                  // cluster size\n  long long row_bytes;     // hkv * 64\n  float scale_log2;        // sm_scale * qk_scale_factor * log2(e)\n};\n\nDEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }\n\nDEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");\n}\nDEVI void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::: "memory"); }\ntemplate <int N>\nDEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N) : "memory"); }\n\nDEVI uint32_t cvt_e4m3x2(uint32_t v) {\n  uint32_t r;\n  uint16_t h = static_cast<uint16_t>(v & 0xffffu);\n  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\\n" : "=r"(r) : "h"(h));\n  return r;\n}\n\nDEVI uint32_t pack_h2(float lo, float hi) {\n  __half2 h = __floats2half2_rn(lo, hi);\n  return *reinterpret_cast<uint32_t*>(&h);\n}\n\nDEVI float ex2(float x) {\n  float y;\n  asm("ex2.approx.ftz.f32 %0, %1;\\n" : "=f"(y) : "f"(x));\n  return y;\n}\n\nDEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "\n      "{%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n}\n\nDEVI uint32_t cluster_addr(const void* p, int rank) {\n  uint32_t r;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\\n" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));\n  return r;\n}\nDEVI float ld_cluster_f32(uint32_t a) {\n  float v;\n  asm volatile("ld.shared::cluster.f32 %0, [%1];\\n" : "=f"(v) : "r"(a) : "memory");\n  return v;\n}\nDEVI float2 ld_cluster_f32x2(uint32_t a) {\n  float2 v;\n  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];\\n" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");\n  return v;\n}\nDEVI void cluster_sync_all() {\n  asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;\\n" ::: "memory");\n}\n\n// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.\nDEVI int swz(int t) { return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2); }\n\nDEVI float warp_sum(float v) {\n#pragma unroll\n  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n  return v;\n}\n\n\n// Load all projection partials together, then round at the original BF16 boundary.\ntemplate <int N>\nDEVI void proj_pairs(const Params& p, const int (&tk)[N], const int (&col)[N], __nv_bfloat162 (&out)[N]) {\n  if (!p.proj_splits) {\n    uint32_t raw[N];\n#pragma unroll\n    for (int i=0; i<N; i++) raw[i] = *(const uint32_t*)(p.qkv+(int64_t)tk[i]*p.qkv_stride+col[i]);\n#pragma unroll\n    for (int i=0; i<N; i++) memcpy(&out[i], &raw[i], 4);\n  } else {\n    float2 acc[N];\n#pragma unroll\n    for (int i=0; i<N; i++) acc[i]=make_float2(0.f,0.f);\n    for (int s=0; s<p.proj_splits; s++) {\n      float2 v[N];\n#pragma unroll\n      for (int i=0; i<N; i++) v[i]=*(const float2*)((const float*)p.qkv+(int64_t)s*p.split_stride+(int64_t)tk[i]*p.qkv_stride+col[i]);\n#pragma unroll\n      for (int i=0; i<N; i++) {acc[i].x+=v[i].x;acc[i].y+=v[i].y;}\n    }\n#pragma unroll\n    for (int i=0; i<N; i++) out[i]=__floats2bfloat162_rn(acc[i].x,acc[i].y);\n  }\n}\n\ntemplate <int L>\nstruct WarpState {\n  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp\n  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims\n  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims\n  float m[MTW][2];                               // running max (log2 domain, scaled)\n  float l[MTW][2];                               // running sum (thread-partial)\n};\n\n// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.\n// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);\n//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).\ntemplate <int L, int NTC>\nDEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,\n                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2) {\n  constexpr int MTW = WarpState<L>::MTW;\n  constexpr int KK = NTC / 2;\n  const int lane = threadIdx.x & 31;\n  const int g = lane >> 2, q4 = lane & 3;\n\n  float s[MTW][NTC][4];\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt)\n#pragma unroll\n    for (int n = 0; n < NTC; ++n)\n#pragma unroll\n      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;\n\n  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,\n  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q\'s fragments).\n#pragma unroll\n  for (int n = 0; n < NTC; ++n) {\n    const int t = tb + 8 * n + g;\n    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));\n    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};\n#pragma unroll\n    for (int ks = 0; ks < 4; ++ks) {\n      const uint32_t b0 = cvt_e4m3x2(w[ks]);\n      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);\n#pragma unroll\n      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);\n    }\n  }\n\n  if (need_mask) {\n#pragma unroll\n    for (int mt = 0; mt < MTW; ++mt) {\n#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n        const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));\n        const int qp = seq + j;\n        const int lo = window ? (qp - window + 1) : -2147483647;\n        const int hi = min(k_hi, qp + 1);\n#pragma unroll\n        for (int n = 0; n < NTC; ++n) {\n          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);\n          if (kp < lo || kp >= hi) s[mt][n][e] = -INFINITY;\n        }\n      }\n    }\n  }\n\n  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).\n  uint32_t pa[MTW][KK][4];\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt) {\n#pragma unroll\n    for (int rh = 0; rh < 2; ++rh) {\n      if (L == 1 && rh == 1) continue;\n      float mx = -INFINITY;\n#pragma unroll\n      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));\n      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);\n      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;\n      const float corr = ex2(st.m[mt][rh] - m_use);\n      st.m[mt][rh] = m_new;\n      st.l[mt][rh] *= corr;\n#pragma unroll\n      for (int jn = 0; jn < 8; ++jn) {\n        st.o[mt][jn][2 * rh] *= corr;\n        st.o[mt][jn][2 * rh + 1] *= corr;\n      }\n      float sum = 0.f;\n#pragma unroll\n      for (int n = 0; n < NTC; ++n) {\n        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));\n        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));\n        s[mt][n][2 * rh] = p0;\n        s[mt][n][2 * rh + 1] = p1;\n        sum += p0 + p1;\n      }\n      st.l[mt][rh] += sum;\n    }\n#pragma unroll\n    for (int kk = 0; kk < KK; ++kk) {\n      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);\n      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);\n      if (L == 1) {\n        pa[mt][kk][1] = 0u;\n        pa[mt][kk][3] = 0u;\n      } else {\n        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);\n        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);\n      }\n    }\n  }\n\n  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a\n  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.\n#pragma unroll\n  for (int kk = 0; kk < KK; ++kk) {\n    const int t0 = tb + 16 * kk + 2 * q4;\n    const int coff = ((4 * h + (g >> 1))) ;\n    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));\n    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));\n    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));\n    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));\n#pragma unroll\n    for (int jj = 0; jj < 2; ++jj) {\n      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;\n      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;\n      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);\n      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);\n      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};\n      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};\n#pragma unroll\n      for (int x = 0; x < 4; ++x)\n#pragma unroll\n        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);\n    }\n  }\n}\n\ntemplate <int L, int STAGES>\n__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n#endif\n  constexpr int MTW = WarpState<L>::MTW;\n  constexpr int NT = (L == 1) ? 4 : 8;  // n-tiles per warp per tile\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* sK = smem;\n  uint8_t* sV = smem + STAGES * TILE_BYTES;\n  uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;  // new tokens: 16 rows of K, 16 of V\n  uint8_t* sNV = sNK + 16 * ROWB;\n  __half* sQ = reinterpret_cast<__half*>(sNV + 16 * ROWB);  // [2][L*8][64]\n  float2* sCS = reinterpret_cast<float2*>(sQ + 2 * L * G * D);  // [L][32]\n  __nv_bfloat16* sG = reinterpret_cast<__nv_bfloat16*>(sCS + L * 32);  // gates of this CTA\'s rows [L*16/CS][64]\n\n  const int split = blockIdx.x, kvg = blockIdx.y, b = blockIdx.z;\n  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n  const int g = lane >> 2, q4 = lane & 3;\n  const int h = warp & 1, wh = warp >> 1;\n  const bool last = split == p.ns - 1;\n\n  const int seq = p.seq_lens[b];\n  const int lo = p.window ? max(0, seq - p.window + 1) : 0;\n  const int n_cached = seq - lo;\n  int chunk = (n_cached + p.ns - 1) / p.ns;\n  chunk = (chunk + 15) & ~15;\n  const int c_lo = min(lo + split * chunk, seq);\n  const int c_hi = min(c_lo + chunk, seq);\n  const int ntiles = (c_hi - c_lo + T - 1) / T;\n\n  const int* pt = p.page_table + (long long)b * p.max_pages;\n  const int lc = tid & 7;      // 16-byte chunk of the row this thread copies\n  const int lt = tid >> 3;     // first token (of 4: lt + 16k)\n  const uint8_t* kbase = p.k_cache + kvg * ROWB + lc * 16;\n  const uint8_t* vbase = p.v_cache + kvg * ROWB + lc * 16;\n\n  auto load_rows = [&](int tile, int (&rows)[4]) {\n#pragma unroll\n    for (int k = 0; k < 4; ++k) {\n      const int t = c_lo + tile * T + lt + 16 * k;\n      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;\n    }\n  };\n  auto issue = [&](int slot, const int (&rows)[4]) {\n#pragma unroll\n    for (int k = 0; k < 4; ++k) {\n      const int t = lt + 16 * k;\n      const int off = slot * TILE_BYTES + t * ROWB + ((lc ^ swz(t)) << 4);\n      const long long r = rows[k] < 0 ? 0 : rows[k];\n      cp_async16(smem_u32(sK + off), kbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\n      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, rows[k] < 0 ? 0 : 16);\n    }\n  };\n\n  // Every independent global load of the prologue goes out first (page-table rows, queries,\n  // new keys/values, positions, cache rows), so their latencies overlap.\n  int rows[STAGES][4];\n#pragma unroll\n  for (int s = 0; s < STAGES; ++s) load_rows(s, rows[s]);\n#pragma unroll\n  for (int s = 0; s < STAGES - 1; ++s) {\n    if (s < ntiles) issue(s, rows[s]);\n    cp_commit();\n  }\n  int rows_next[4];\n#pragma unroll\n  for (int k = 0; k < 4; ++k) rows_next[k] = rows[STAGES - 1][k];\n\n  constexpr int QR = 4 * L;\n  int toks[QR+L], cols[QR+L];\n#pragma unroll\n  for (int i=0; i<QR; i++) {\n    const int r=warp+4*i, j=r>>4, hh=r&15;\n    toks[i]=b*L+j; cols[i]=(kvg*16+hh)*D+2*lane;\n  }\n  long long nloc[L];\n#pragma unroll\n  for (int i=0; i<L; i++) {\n    const int r=warp+4*i, j=r>>2, hk=(r>>1)&1, isv=r&1;\n    toks[QR+i]=b*L+j; cols[QR+i]=(p.hq+(isv?p.hkv:0)+kvg*2+hk)*D+2*lane;\n    nloc[i]=p.cache_loc[b*L+j];\n  }\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n  __nv_bfloat162 qn[QR+L], qv[QR], nv[L];\n  proj_pairs<QR+L>(p,toks,cols,qn);\n#pragma unroll\n  for (int i=0; i<QR; i++) qv[i]=qn[i];\n#pragma unroll\n  for (int i=0; i<L; i++) nv[i]=qn[QR+i];\n  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;\n  float ang[CSN];\n#pragma unroll\n  for (int c = 0; c < CSN; ++c) {\n    const int i = tid + c * NTHR;\n    ang[c] = 0.f;\n    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];\n  }\n  const float ksc = *p.k_scale, vsc = *p.v_scale;\n\n  // Gate rows this CTA finalizes (rows rank, rank + CS, ...): 8 x 16 bytes each, with tile 0.\n  const bool gated = p.gated;  // else ungated output (P7 gate split)\n  if (gated) {\n    const int rank = split % p.cs;\n    const int nrows = (L * 16 - rank + p.cs - 1) / p.cs;\n    for (int i = tid; i < nrows * 8; i += NTHR) {\n      const int k = i >> 3, c = i & 7, rr = rank + k * p.cs, j = rr >> 4, hh = rr & 15;\n\n      int gt[4], gc[4]; __nv_bfloat162 gv[4];\n#pragma unroll\n      for (int z=0; z<4; z++) {\n        gt[z]=b*L+j;\n        gc[z]=(p.hq+2*p.hkv+kvg*16+hh)*D+c*8+2*z;\n      }\n      proj_pairs<4>(p,gt,gc,gv);\n#pragma unroll\n      for (int z=0; z<4; z++) *(__nv_bfloat162*)(sG+k*D+c*8+2*z)=gv[z];\n    }\n  }\n\n  // RoPE angles for the block\'s tokens.\n  if (p.inv_freq) {\n#pragma unroll\n    for (int c = 0; c < CSN; ++c) {\n      const int i = tid + c * NTHR;\n      if (i < L * 32) {\n        float sn, cs;\n        sincosf(ang[c], &sn, &cs);\n        sCS[i] = make_float2(cs, sn);\n      }\n    }\n  }\n  if (last) {  // zero the new-token rows past L\n    for (int i = tid; i < 2 * 16 * ROWB / 16; i += NTHR) {\n      if ((i % (16 * ROWB / 16)) >= L * ROWB / 16) reinterpret_cast<uint4*>(sNK)[i] = make_uint4(0, 0, 0, 0);\n    }\n  }\n  __syncthreads();\n\n  const float eps = 1e-5f;\n  // Queries: normalized and roped, FP16 into smem as [kv head][token j * 8 + head][64].\n#pragma unroll\n  for (int i = 0; i < QR; ++i) {\n    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;\n    float x0 = __bfloat162float(qv[i].x), x1 = __bfloat162float(qv[i].y);\n    const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\n    x0 *= inv;\n    x1 *= inv;\n    if (p.inv_freq) {\n      const float2 cs = sCS[j * 32 + lane];\n      const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\n      x0 = y0;\n      x1 = y1;\n    }\n    *reinterpret_cast<__half2*>(sQ + ((hh >> 3) * (L * G) + j * G + (hh & 7)) * D + 2 * lane) = __floats2half2_rn(x0, x1);\n  }\n  // New keys and values (last split): normalize/rope keys, quantize, store to the cache and smem.\n  if (last) {\n#pragma unroll\n    for (int i = 0; i < L; ++i) {\n      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;\n      const int head = kvg * 2 + hk;\n      float x0 = __bfloat162float(nv[i].x), x1 = __bfloat162float(nv[i].y);\n      if (!isv) {\n        const float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / D) + eps);\n        x0 *= inv;\n        x1 *= inv;\n        if (p.inv_freq) {\n          const float2 cs = sCS[j * 32 + lane];\n          const float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;\n          x0 = y0;\n          x1 = y1;\n        }\n      }\n      const float sc = isv ? vsc : ksc;\n      const __nv_fp8x2_storage_t f8 = __nv_cvt_float2_to_fp8x2(make_float2(x0 / sc, x1 / sc), __NV_SATFINITE, __NV_E4M3);\n      uint8_t* cache = isv ? p.v_cache : p.k_cache;\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(cache + nloc[i] * p.row_bytes + head * D + 2 * lane) = f8;\n      const int byte = hk * D + 2 * lane;\n      uint8_t* dst = (isv ? sNV : sNK) + j * ROWB + ((((byte >> 4) ^ swz(j))) << 4) + (byte & 15);\n      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;\n    }\n  }\n  __syncthreads();\n\n  WarpState<L> st;\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt) {\n    const int mta = (L == 1) ? 0 : (2 * wh + mt);\n    const int r0 = 16 * mta + g, r1 = r0 + 8;\n#pragma unroll\n    for (int ks = 0; ks < 4; ++ks) {\n      const uint2 x0 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r0) * D + 16 * q4 + 4 * ks);\n      st.qa[mt][ks][0] = x0.x;\n      st.qa[mt][ks][2] = x0.y;\n      if (L == 1) {\n        st.qa[mt][ks][1] = 0u;\n        st.qa[mt][ks][3] = 0u;\n      } else {\n        const uint2 x1 = *reinterpret_cast<const uint2*>(sQ + (h * L * G + r1) * D + 16 * q4 + 4 * ks);\n        st.qa[mt][ks][1] = x1.x;\n        st.qa[mt][ks][3] = x1.y;\n      }\n    }\n    st.m[mt][0] = st.m[mt][1] = -INFINITY;\n    st.l[mt][0] = st.l[mt][1] = 0.f;\n#pragma unroll\n    for (int jn = 0; jn < 8; ++jn)\n#pragma unroll\n      for (int e = 0; e < 4; ++e) st.o[mt][jn][e] = 0.f;\n  }\n\n  const float scale_log2 = p.scale_log2 * (*p.k_scale);\n  const int tb = (L == 1) ? 32 * wh : 0;\n  const int win_edge = p.window ? (seq + L - p.window) : -2147483647;  // keys below need per-row masks\n\n  for (int i = 0; i < ntiles; ++i) {\n    cp_wait<STAGES - 2>();\n    __syncthreads();\n    const int nxt = i + STAGES - 1;\n    if (nxt < ntiles) issue(nxt % STAGES, rows_next);\n    cp_commit();\n    load_rows(nxt + 1, rows_next);\n    const int slot = i % STAGES;\n    const int kp0 = c_lo + i * T;\n    const bool need_mask = (kp0 + T > c_hi) || (L > 1 && kp0 < win_edge);\n    compute_tile<L, NT>(st, sK + slot * TILE_BYTES, sV + slot * TILE_BYTES, tb, h, wh, kp0, c_hi, seq, p.window,\n                        need_mask, scale_log2);\n  }\n  cp_wait<0>();\n  if (last && (L > 1 || wh == 0)) {\n    compute_tile<L, 2>(st, sNK, sNV, 0, h, wh, seq, seq + L, seq, p.window, true, scale_log2);\n  }\n\n  // Thread-partial sums -> row sums.\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt)\n#pragma unroll\n    for (int rh = 0; rh < 2; ++rh) {\n      float v = st.l[mt][rh];\n      v += __shfl_xor_sync(0xffffffffu, v, 1);\n      v += __shfl_xor_sync(0xffffffffu, v, 2);\n      st.l[mt][rh] = v;\n    }\n\n  // Partials into shared memory: [slot][row][PR], row = token j * 16 + head hh of the KV pair,\n  // slot = token half (L == 1) or 0. PR floats: 64 dims, max, sum.\n  constexpr int ROWS = L * 16;\n  constexpr int NSL = (L == 1) ? 2 : 1;\n  constexpr int PR = 68;\n  __syncthreads();  // every warp is done with the stage buffers\n  const int KVG = p.hkv / 2;\n  float* sP = p.dsmem ? reinterpret_cast<float*>(smem)\n                      : p.part1 + ((long long)(b * KVG + kvg) * p.ns + split) * (NSL * ROWS * PR);\n#pragma unroll\n  for (int mt = 0; mt < MTW; ++mt) {\n#pragma unroll\n    for (int rh = 0; rh < 2; ++rh) {\n      if (L == 1 && rh == 1) continue;\n      const int mta = (L == 1) ? 0 : (2 * wh + mt);\n      const int j = (L == 1) ? 0 : (2 * mta + rh);\n      const int row = j * 16 + h * 8 + g;\n      const int slot = (L == 1) ? wh : 0;\n      float* dst = sP + (slot * ROWS + row) * PR;\n      float4* d4 = reinterpret_cast<float4*>(dst + 16 * q4);\n      d4[0] = make_float4(st.o[mt][0][2 * rh], st.o[mt][1][2 * rh], st.o[mt][2][2 * rh], st.o[mt][3][2 * rh]);\n      d4[1] = make_float4(st.o[mt][4][2 * rh], st.o[mt][5][2 * rh], st.o[mt][6][2 * rh], st.o[mt][7][2 * rh]);\n      d4[2] = make_float4(st.o[mt][0][2 * rh + 1], st.o[mt][1][2 * rh + 1], st.o[mt][2][2 * rh + 1], st.o[mt][3][2 * rh + 1]);\n      d4[3] = make_float4(st.o[mt][4][2 * rh + 1], st.o[mt][5][2 * rh + 1], st.o[mt][6][2 * rh + 1], st.o[mt][7][2 * rh + 1]);\n      if (q4 == 0) {\n        dst[64] = st.m[mt][rh];\n        dst[65] = st.l[mt][rh];\n      }\n    }\n  }\n  if (!p.dsmem) __threadfence();\n  cluster_sync_all();\n\n  const int cs = p.cs, rank = split % cs, cidx = split / cs, nc = p.ns / cs;\n  // Partial (rank c of this cluster, slot sl, row rr), at float offset off within the row.\n  const float* gP0 = p.part1 + ((long long)(b * KVG + kvg) * p.ns + cidx * cs) * (NSL * ROWS * PR);\n  auto part2 = [&](int c, int sl, int rr, int off) -> float2 {\n    if (p.dsmem) return ld_cluster_f32x2(cluster_addr(sP + (sl * ROWS + rr) * PR + off, c));\n    return __ldcg(reinterpret_cast<const float2*>(gP0 + (long long)c * (NSL * ROWS * PR) + (sl * ROWS + rr) * PR + off));\n  };\n  const float vs = *p.v_scale;\n  const float eps2 = 1e-5f;\n  auto finalize = [&](int rr, int k, float o0, float o1) {\n    const int j = rr >> 4, hh = rr & 15;\n    const float r = rsqrtf(warp_sum(o0 * o0 + o1 * o1) * (1.f / D) + eps2);\n    float g0 = 1.f, g1 = 1.f;\n    if (gated) {\n      const __nv_bfloat162 gt = reinterpret_cast<const __nv_bfloat162*>(sG + k * D)[lane];\n      g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));\n      g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));\n    }\n    *reinterpret_cast<__nv_bfloat162*>(p.out + (long long)(b * L + j) * (p.hq * D) + (kvg * 16 + hh) * D + 2 * lane) =\n        __floats2bfloat162_rn(o0 * r * g0, o1 * r * g1);\n  };\n\n  // Level 1: rank r merges rows r, r + CS, ... over the cluster\'s partials (distributed smem).\n  const int np = cs * NSL;\n  for (int k = warp; rank + k * cs < ROWS; k += 4) {\n    const int rr = rank + k * cs;\n    float mi = -INFINITY, li = 0.f;\n    if (lane < np) {\n      const float2 v = part2(lane / NSL, lane % NSL, rr, 64);\n      mi = v.x;\n      li = v.y;\n    }\n    float M = mi;\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));\n    const float Mu = (M == -INFINITY) ? 0.f : M;\n    const float wi = (lane < np) ? ex2(mi - Mu) : 0.f;\n    const float lsum = warp_sum(wi * li);\n    float o0 = 0.f, o1 = 0.f;\n    for (int i0 = 0; i0 < np; i0 += 8) {  // eight remote loads in flight at a time\n      float2 v[8];\n#pragma unroll\n      for (int u = 0; u < 8; ++u)\n        if (i0 + u < np)\n          v[u] = part2((i0 + u) / NSL, (i0 + u) % NSL, rr, 2 * lane);\n#pragma unroll\n      for (int u = 0; u < 8; ++u) {\n        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);\n        if (i0 + u < np) {\n          o0 = fmaf(w, v[u].x, o0);\n          o1 = fmaf(w, v[u].y, o1);\n        }\n      }\n    }\n    if (nc == 1) {\n      const float sc = vs / lsum;\n      finalize(rr, k, o0 * sc, o1 * sc);\n    } else {\n      const long long prow = ((long long)(b * KVG + kvg) * nc + cidx) * ROWS + rr;\n      reinterpret_cast<float2*>(p.part_o + prow * D)[lane] = make_float2(o0, o1);\n      if (lane == 0) reinterpret_cast<float2*>(p.part_ml)[prow] = make_float2(M, lsum);\n    }\n  }\n  if (p.dsmem) cluster_sync_all();  // no CTA leaves while its shared memory may still be read\n  if (nc == 1) return;\n\n  // Level 2: the last cluster to arrive at rank r merges rows r, r + CS, ... over the clusters.\n  __shared__ int s_last;\n  __threadfence();\n  __syncthreads();\n  if (tid == 0) {\n    const int prev = atomicAdd(p.counters + (b * KVG + kvg) * cs + rank, 1);\n    s_last = (prev == nc - 1);\n  }\n  __syncthreads();\n  if (!s_last) return;\n  __threadfence();\n  for (int k = warp; rank + k * cs < ROWS; k += 4) {\n    const int rr = rank + k * cs;\n    const long long prow0 = ((long long)(b * KVG + kvg) * nc) * ROWS + rr;\n    float mi = -INFINITY, li = 0.f;\n    if (lane < nc) {\n      const float2 ml = __ldcg(reinterpret_cast<const float2*>(p.part_ml) + prow0 + (long long)lane * ROWS);\n      mi = ml.x;\n      li = ml.y;\n    }\n    float M = mi;\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));\n    const float Mu = (M == -INFINITY) ? 0.f : M;\n    const float wi = (lane < nc) ? ex2(mi - Mu) : 0.f;\n    const float lsum = warp_sum(wi * li);\n    float o0 = 0.f, o1 = 0.f;\n    for (int i0 = 0; i0 < nc; i0 += 8) {\n      float2 v[8];\n#pragma unroll\n      for (int u = 0; u < 8; ++u)\n        if (i0 + u < nc) v[u] = __ldcg(reinterpret_cast<const float2*>(p.part_o + (prow0 + (long long)(i0 + u) * ROWS) * D) + lane);\n#pragma unroll\n      for (int u = 0; u < 8; ++u) {\n        const float w = __shfl_sync(0xffffffffu, wi, (i0 + u) & 31);\n        if (i0 + u < nc) {\n          o0 = fmaf(w, v[u].x, o0);\n          o1 = fmaf(w, v[u].y, o1);\n        }\n      }\n    }\n    const float sc = vs / lsum;\n    finalize(rr, k, o0 * sc, o1 * sc);\n  }\n  if (tid == 0) p.counters[(b * KVG + kvg) * cs + rank] = 0;\n}\n\ntemplate <int L, int STAGES>\nconstexpr int smem_bytes() {\n  return 2 * STAGES * TILE_BYTES + 2 * 16 * ROWB + 2 * L * G * D * 2 + L * 32 * 8 + L * 16 * D * 2;\n}\n\ntemplate <int L, int STAGES>\ncudaError_t launch_t(const Params& p, cudaStream_t stream) {\n  constexpr int smem = smem_bytes<L, STAGES>();\n  static_assert(L * 16 * (L == 1 ? 2 : 1) * 68 * 4 <= 2 * STAGES * TILE_BYTES, "partials fit in the stage buffers");\n  static bool attr = false;\n  if (!attr) {\n    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n    attr = true;\n  }\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(p.ns, p.hkv / 2, p.batch);\n  cfg.blockDim = dim3(NTHR, 1, 1);\n  cfg.dynamicSmemBytes = smem;\n  cfg.stream = stream;\n  cudaLaunchAttribute at[2];\n  at[0].id = cudaLaunchAttributeClusterDimension;\n  at[0].val.clusterDim.x = p.cs;\n  at[0].val.clusterDim.y = 1;\n  at[0].val.clusterDim.z = 1;\n  cfg.attrs = at;\n  at[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;\n  at[1].val.programmaticStreamSerializationAllowed=1;\n  cfg.numAttrs = 2;\n  return cudaLaunchKernelEx(&cfg, attn_main<L, STAGES>, p);\n}\n\ntemplate <int L, int STAGES>\nint max_cluster_t(int cs) {\n  constexpr int smem = smem_bytes<L, STAGES>();\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(cs, 1, 1);\n  cfg.blockDim = dim3(NTHR, 1, 1);\n  cfg.dynamicSmemBytes = smem;\n  cudaLaunchAttribute at[1];\n  at[0].id = cudaLaunchAttributeClusterDimension;\n  at[0].val.clusterDim.x = cs;\n  at[0].val.clusterDim.y = 1;\n  at[0].val.clusterDim.z = 1;\n  cfg.attrs = at;\n  cfg.numAttrs = 1;\n  int n = 0;\n  if (cudaOccupancyMaxActiveClusters(&n, attn_main<L, STAGES>, &cfg) != cudaSuccess) {\n    cudaGetLastError();\n    return 0;\n  }\n  return n;\n}\n\n}  // namespace\n\nvoid attn_forward(int64_t qkv, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,\n                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,\n                  int64_t part_o, int64_t part_ml, int64_t counters, int64_t part1, int64_t dsmem, int64_t out, int64_t batch, int64_t q_len,\n                  int64_t qkv_stride, int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window,\n                  int64_t ns, int64_t cs, double scale_log2, int64_t proj_splits, int64_t split_stride, int64_t qkv_width, int64_t stream) {\n  Params p;\n  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);\n  p.positions = reinterpret_cast<const int64_t*>(positions);\n  p.cache_loc = reinterpret_cast<const int64_t*>(cache_loc);\n  p.page_table = reinterpret_cast<const int*>(page_table);\n  p.seq_lens = reinterpret_cast<const int*>(seq_lens);\n  p.k_cache = reinterpret_cast<uint8_t*>(k_cache);\n  p.v_cache = reinterpret_cast<uint8_t*>(v_cache);\n  p.k_scale = reinterpret_cast<const float*>(k_scale);\n  p.v_scale = reinterpret_cast<const float*>(v_scale);\n  p.inv_freq = reinterpret_cast<const float*>(inv_freq);\n  p.part_o = reinterpret_cast<float*>(part_o);\n  p.part_ml = reinterpret_cast<float*>(part_ml);\n  p.counters = reinterpret_cast<int*>(counters);\n  p.part1 = reinterpret_cast<float*>(part1);\n  p.dsmem = (int)dsmem;\n  p.gated = qkv_width > (hq + 2*hkv)*64;\n  p.out = reinterpret_cast<__nv_bfloat16*>(out);\n  p.batch = (int)batch;\n  p.qkv_stride = (int)qkv_stride;\n  p.proj_splits = (int)proj_splits;\n  p.split_stride = split_stride;\n  p.hq = (int)hq;\n  p.hkv = (int)hkv;\n  p.max_pages = (int)max_pages;\n  p.ps = (int)ps;\n  int lg = 0;\n  while ((1 << lg) < ps) ++lg;\n  p.ps_log2 = lg;\n  p.window = (int)window;\n  p.ns = (int)ns;\n  p.cs = (int)cs;\n  p.row_bytes = hkv * 64;\n  p.scale_log2 = (float)scale_log2;\n  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);\n  cudaError_t e = (q_len == 1) ? launch_t<1, 4>(p, s) : launch_t<8, 3>(p, s);\n  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));\n}\n\ntemplate <int L, int STAGES>\nint occupancy_t() {\n  constexpr int smem = smem_bytes<L, STAGES>();\n  cudaFuncSetAttribute(attn_main<L, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n  int n = 0;\n  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, attn_main<L, STAGES>, NTHR, smem) != cudaSuccess) {\n    cudaGetLastError();\n    return 1;\n  }\n  return n;\n}\n\nint64_t occupancy(int64_t q_len) { return (q_len == 1) ? occupancy_t<1, 4>() : occupancy_t<8, 3>(); }\n\nint64_t max_cluster(int64_t q_len, int64_t cs) {\n  return (q_len == 1) ? max_cluster_t<1, 4>((int)cs) : max_cluster_t<8, 3>((int)cs);\n}\n'
PARTIAL_FAST_CUDA = PARTIAL_CUDA
PARTIAL_CPP = '\n#include <cstdint>\nvoid attn_forward(int64_t qkv, int64_t positions, int64_t cache_loc, int64_t page_table, int64_t seq_lens,\n                  int64_t k_cache, int64_t v_cache, int64_t k_scale, int64_t v_scale, int64_t inv_freq,\n                  int64_t part_o, int64_t part_ml, int64_t counters, int64_t part1, int64_t dsmem, int64_t out, int64_t batch, int64_t q_len,\n                  int64_t qkv_stride, int64_t hq, int64_t hkv, int64_t max_pages, int64_t ps, int64_t window,\n                  int64_t ns, int64_t cs, double scale_log2, int64_t proj_splits, int64_t split_stride, int64_t qkv_width, int64_t stream);\nint64_t max_cluster(int64_t q_len, int64_t cs);\nint64_t occupancy(int64_t q_len);\n'

_PP_MOD = {}

def partial_port_ext(fast=False):
    if fast not in _PP_MOD:
        major, minor = torch.cuda.get_device_capability()
        arch='90a' if major==9 else f'{major}{minor}'
        cu=PARTIAL_FAST_CUDA if fast else PARTIAL_CUDA
        tag=hashlib.sha256(cu.encode()).hexdigest()[:12]
        _PP_MOD[fast]=load_inline(name=f'attn_partial_{tag}_{arch}', cpp_sources=PARTIAL_CPP,
            cuda_sources=cu,functions=['attn_forward','max_cluster','occupancy'],
            extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return _PP_MOD[fast]


def attention_partial_port(qkv, data, fast=False):
    P, S=qkv
    cfg=data['config']; B,L=cfg['batch'],cfg['q_len']; HQ_,HK_=cfg['num_q_heads'],cfg['num_kv_heads']
    dev=P.device; window=cfg['window']; ps=cfg['page_size']; pages=data['page_table'].shape[1]
    ext=partial_port_ext(fast)
    occ=(max(1,ext.occupancy(L)),{c:ext.max_cluster(L,c) for c in range(1,17)})
    ns,cs=_attn_main_plan(B,HK_//2,min(pages*ps,window) if window else pages*ps,L,window,_num_sms(dev),occ)
    rows=B*(HK_//2)*max(1,ns//cs)*L*16
    po=torch.empty((max(rows,1)*64,),device=dev,dtype=torch.float32)
    ml=torch.empty((max(rows,1)*2,),device=dev,dtype=torch.float32)
    counters=_attn_port_counters(B*(HK_//2)*cs,dev)
    out=torch.empty((B*L,HQ_*64),device=dev,dtype=torch.bfloat16)
    raw=P[0] if P.dtype==torch.bfloat16 and P.dim()==3 else P
    nsplit=S if P.dtype==torch.float32 else 0
    inv=_inv_freq(cfg['rope_theta'],dev) if window else None
    ext.attn_forward(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),
        data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),data['k_scale'].data_ptr(),
        data['v_scale'].data_ptr(),inv.data_ptr() if inv is not None else 0,po.data_ptr(),ml.data_ptr(),counters.data_ptr(),
        po.data_ptr(),1,out.data_ptr(),B,L,raw.stride(-2),HQ_,HK_,pages,ps,window,ns,cs,
        cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,nsplit,P.stride(0) if nsplit else 0,P.shape[-1],
        torch.cuda.current_stream(dev).cuda_stream)
    return out

_PP_ROWS = ((1, 1, False), (1, 1, True))
_PP_FAST = False


def _layer_partial_port(qkv,data):
    cfg=dict(data['config'],num_q_heads=HQ,num_kv_heads=HKV)
    return attention_partial_port(qkv,dict(data,config=cfg),_PP_FAST)


def _partial_port_prepare(device):
    for batch,length,sliding in _PP_ROWS:
        for ctx in (257,2305):
            qkv,data=_attn_port_synthetic(batch,length,2048 if sliding else 0,device,ctx)
            got=_layer_partial_port(qkv,data)
            _attention_store_agrees(qkv,data)
            ok,error,bound=_attention_agrees(got,_attention_reference(qkv,data),_attention_reference(qkv,data,True))
            if not ok:raise RuntimeError(f'partial_port {(batch,length,sliding)} ctx {ctx}: {error:.5f}>{bound:.5f}')
    print('PARTIAL_PORT_ROWS',_PP_ROWS,flush=True)
    return _PP_ROWS




@triton.jit
def _gi_kernel(X,W,P,T,x_stride,N,K,S,BN:tl.constexpr,BK:tl.constexpr,TB:tl.constexpr,
               NS:tl.constexpr,TRANS:tl.constexpr,CG:tl.constexpr):
    pn=tl.program_id(0);ps=tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    n=pn*BN+tl.arange(0,BN)
    kb0=ps*(K//BK)//S;kb1=(ps+1)*(K//BK)//S
    tl.extra.cuda.gdc_wait()
    t=tl.arange(0,TB);kk=tl.arange(0,BK)
    if TRANS:
        acc=tl.zeros((TB,BN),tl.float32)
    else:
        acc=tl.zeros((BN,TB),tl.float32)
    for kb in tl.range(kb0,kb1,num_stages=NS):
        w=tl.load(W+n[:,None]*K+(kb*BK+kk)[None,:],cache_modifier='.cg' if CG else '')
        x=tl.load(X+t[:,None]*x_stride+(kb*BK+kk)[None,:],t[:,None]<T,0.)
        if TRANS:
            acc=tl.dot(x,w.T,acc)
        else:
            acc=tl.dot(w,x.T,acc)
    if TRANS:
        tl.store(P+ps*T*N+t[:,None]*N+n[None,:],acc,t[:,None]<T)
    else:
        tl.store(P+ps*T*N+t[None,:]*N+n[:,None],acc,t[None,:]<T)

def gi_run(x,w,s,cfg):
    bk,ns,trans,cg=cfg
    t,k=x.shape;n=w.shape[0]
    p=torch.empty((s,t,n),device=x.device,dtype=torch.float32)
    _gi_kernel[(n//64,s)](x,w,p,t,x.stride(0),n,k,s,BN=64,BK=bk,TB=max(16,triton.next_power_of_2(t)),
                         NS=ns,TRANS=bool(trans),CG=bool(cg),num_warps=4,launch_pdl=True)
    return p


# ---- h200_lab wrapper: payloads s14ab_payload d0773fbadd336203 on base S14 72bb46f6 ----
_LAB_PAYLOADS = [('s14ab_payload', 'd0773fbadd336203', 'import base64, lzma, hashlib, linecache\n_s14_body = lzma.decompress(base64.b85decode(\'{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;Wq>C{#^hOXc5f@^K)lHO<*fof4pu)l+@0?H16~xz2zWQERsfZ<fy2LtAQbcBL?s6oo!lM(iMCJW7=Sg+{9CgJrD@~vz@ApcNE?{z1hQ5hf(pjQdBs{0thh{t1|ZS8wP*Nhf>g$CGGbW`G=0M$UVkRS#78y{7Bp4)082L(mtw0b(fQ&`fyuL=Zuack$V^k001HIo()@$=taD;yN{debycx%zrmhxO4jSXDU!gQC_QV_AgT0uzkYP2qE^C6Cba`mJBqwd()q9o3*;E&G%4+}#IFo~@|C&N`9q3uC^LK`k3Ol|s!>i3T&GW4Yx5>FY9D<jdB~K|+yBE~KP`zwZ|~u0cUpUdrmv#{zI>-63g!oG7EO#vd)R%{oPe7)z8!w=(Hc+r^_2$%xTo(lk33RZ^EsuK@Q3I-b9jm+;22CYY*qbO*yZ*@Wtoa)^b3vtaL9Ol#+~bAu85Gz?mW8Wrxuwj{ynP_pr<F4K9gG1%}Zw*SbncHfO?!1#6RB8)Eyu=C!<>4o^j&wE9cnuj$3!@)a%t_Lhwc2EY;rMca5mGr995591EZ{l(6|Lfa5N;r`$7o*}+-#E}*>br$o&&AG0dLPkKG=V;CksTrk(cwccg~1wrN5c}$f_9$#>1oX~IPR9IONGStW7^tx^qyyHi`s^McQ?BIG0w9OazDK)x(3y;k`R&krBlFn`f&x05<yYL@LR{+ap_nZ%0;#0r2aEkq+2yI-dx8H9@bq&3-`BXH2b?;rrPBGlp3V)e(I?KwgX0>?YIzO{lS=z{|nYJxEuA7yLN0}73rC@To*58%BTsgrOLw?v5;$=XE$3Ae}Hdf@gpUt+AKy316mIP>jD*nbyryNe?!0ow7;^Jhc2m_gnASYK@iUjXUj*^;*``-}`5{9yfM&f-?5)GH|xS41~xs$d!Jyws<u5D|`otHdxV>GC*QmzHYaZ9l#Pao#yOX+~L>o8A__S4lx+KaW(^%Ri)w%HbE8wOQZ?s*K*SrVVuw>yal?CJ;}#W6nwAUS7A0EXI+B+4!iqh@fD3JbPh_wR&Ro=TGYD=Q{$ZkaV7L8QRwY5KEq-Nc0St>S=Le>#-NjRn6v&}3H0M1s1jC5flbXMtS$F)yuV7=E{1wDhKWqk+UZ{YMVGl{AD#cF~n>e#up&SC6OU2}Ng-SiOS;ryzAPXGhF5utNGp+qj!g@161f4L-r``%@$gylxawIvf(?2j%pi{2esOy0Y+gc~i;(okw(1ix%iZ*L_sE3uMqRs~wHBr~@Yel+nXh9|DNK&7|XuRhIrG1m%?wQbxERsZP#>Fan4KAR~?OxqF-n2C2lGTY+_K$}MsaA7FBOZzd~}WRw1%&{_q9ts+*lL$SqW$(f=mo+Rm7k}MSah1dfIQYQ0wMZ|MNkSuuomK{_M5E@(ww0!3vW=wdthhfAzJ8n|Df)Ux9s#_utLBk_ypl~o#U-)U8I7K4h#}zsOOOQXXZHOW2o^^`_2*UorcDRiaK&w{F8Wdk#;dD=CgHvEW+ld@+;Y$B%A{C#+`8(mUm!==2<u$3KyoRkQWOKM_dAdOafGO$W3x4e0k8O>hffJ;bUWK;MSKRIK1Hx)=3cDII-pI^|!fubZLuY>U>{tj}#8gWEcAV+!!Fav+-D?h7Ht?^X6js^P+Cw!DgBJeZXAc3pFW;K9bP^iV-cH&R`ZxOhR0gMWp%?A%sbE1Myvn-57x{S>-xk081+J#y;#2mKI~uSgZ32MfD~8UgM-UXCKz`ZP;X`Jt5VagS-nplL2rxy^3*B+m04}yvI8our!MlIL+`JUx>A2X1<J-%W_ZroO)8_H|M#jZjy$2PrjR%gx|EcuFs(FL?BOE9@#-E3~Ell!0WDwIpTe!Rq5j&El|3zx_{-iuLrATQ)8c=oB1;Nyc{T`qzkjJ=Ux5rOVH(`eV)ffr~(iuL+)}8R0tOViI3~=vtJhTZ=+x_qAOl`&M7-VmBZ6|d8aUSia=&`4dEA&o<y$GC}NLTB3GD{lN;}$UYz#7I0C8oiLDJ+iCeS;)n&{nVlvdZh<AAF#F&6whUOzlY4ZJcsBL3gt<%C>biEVQtw{fcyv#_1FRrAnHTeU^&gmB_z%gGB1|`EG~nEK6*D;+2J+g{^kEK?2R%<FcBhZrmCGKxNBl)c4)z7R5W8-8!N<7|mG@5m8FP>DWphK%sSklwk&#L8Y7gp(+h5f)oq<v<T2@_nc`}fZT!lidzVTrAH>HSX3h(YjcBsk(h7Lf%+ORg#do0&Zwn%vjK4}Hfm4xOe98Iq>d|0LkczrHU|jSwsB#nTT#1SJBRn$pK<DPQL~qw_bCOu-KWgknj7B}J^3$E)H3j4jmUWlyWvTEZUAvs8;b%hX&a@Gi>vQ$Fd|>kP2t>58jGCHcJTaXx-YMS>1KQ-qdRq*<PV78<S#S@2Mwi^#85I(8JF+$MQ+QNiMF3TYrqOdOM?l70=~ed0hZ`!qTpQ$`vBS(Ifwc#;fYuR?K^n(P}a_nH;I|MBrq&>9;Er5sSMKZfI2SRPiaL7O@ElaW&aLoeJ$ay6A{AE`xYGWNCa65rH}yKJAj3F@$R02IUKJKOP^i8>LAAY=mpKjX1(N0MaP*mteB=-h)-=OFH<h`hdW@tp-decpWok9$weL-S{t!`qOw-m0a5U#NS*V?jH8-Qr5?pLncC)imxzmeE$nl~e`fboWmu-EFNSSuZRoWhN6uV!J)7;`JJsO7Db$UNrsOE*$M8hPO_DKmw7@^;j@uPSlURK=rE09L;)9dx*eM=|kd1C(ujAam9k5VQ9sve>_%?x2{7j;-Xr<2pr88fgqtB>oc~W$hsgQmXVG^Jf_<JdqYD##a0B_EWB<Gmb_yz_7pL|8&yLTZ<y}yi>-uQx7qqUejy(P)Vy{aUOAqqv6^c6ira9h^m0+3D(#RmO2xe_ee+9zRb%|!A1`s$b7vIS>$43<Q(yP;y;{8>R)9{u#Dumh~9?ibAKSl9~JR7^@qRkGhGP&d3N@23<<eGLw9j8Bxb%dOl&Q4m#aY<n2`_hd&wnEPuCppJ&I>mv0UsADy)0-LZzM0^-bMmDuD*nu9Rfqr9|cFwo#8VT!HaSP90xRl9j+c)wa*O}fsQ#xHhLggK8^eG$7$YUCi@sM1QN|ZIVen$GZ1gM7U0kbhCio3h$g?>#-=wbRB-95SW9lZqg3qHNP$hf8*flRNwN=-$L>)Xad!=1|H`h2rg^DbO*m>qFP73TxzFL)i`#X7@2@zN!A)$Ulku`ChR?YDXbDgie;0cDPVwac@zkz?cB>?t+#=hBfN1RYXpr#$kaD1m<^kq*nXu@?<%#j==FH~#BO?BUaLL)rtI#%e;=T=`$qZpX0MCUj??3AqwSNCE>W>K#USOimPt#ze$9osa=7y~%HTkI`P6GGY}}HNy_R9a4==a1&#`L}^ELKwaua(#m3|P}Qf0!|U>4cL+6t7Ig}w5s`r0xw!o=X~0TtzI*Mu7xn3J-&f;w_=`@Fmb+P~8$!&zBR;bR(39z@7+eX2-ThWjNa@X+D~<1^8y29H6L^N94onW(9ltbQg(W6Q{lzizWz;fx|1bE0D_8=-LLdyNQJ6-f^gyv?K?vH7w=!kawWG*jg6R8h`FD-<CX;MB5L$IMj0A)B^vnN1oBG?$rAQHIt))0c%^a3zT^mYsl8;_E9uQd|h{B(b9T4lGnWSwP8o@=S_hgKgMFJU=LI`}htOIDZIDI{tGA0WJkBV#YI(IDa9z2=?E!w*$TH1s>X%gBeyXFQ?@TS$YTLQ;DKz+K4ZTUGo?!^`z4;Cs@Mdg?B1E6h3jk{=@GX{D@eKw?rT;0H`L~3}9g1Ut)L|l7<y5@<6Q1Z%{lvV;<y{LM&b!!`hV^i+27NiIMFS>^Kq`u}{@I5>zqR@wF-M2ceLc4q&=dQ^`Xp=LFbUSQRQO^AM`Wf2JH$cxJm6ZlMPN6s|!7v(Tgf@!9xrcrhmA={-eVW3UuxAQGP~1DP5y<yB6JBn~s)roXbLep=StpG_NJSEM+r8D7(s2746A}tIR@Z&V%F*L=7B0{Eok>!*=S5T*k=nqON8uIMB9H7yuezwUQTQWN>rEP$j91N=cdzy9lx-(1t9_M)lzqAM-084nGDNi3w?ji~>-<AtM2tEKly$T5Ecx5pbMd8tt{&`@!0-nl^#7{1)RJo~)O5eKb4lyv;&+~-j?x`O(I&H^6OEh~hYn295Mzz$Nba`x&PQ4uDcP(^vTmwTiuFIv{JW0ahZjoi*(?agS>9_13SXRP=7KYdq?EmwcIZaT?S((??T0%|iva@G24Ks~zYKy|FeTk@+xNuX<>OBsPU?wha6=LR8i4s)qleK+;`V<5|GveZwS{EHn4mj!WZ(b2rtNQ|Uqpm95NNCBylm6iGq^ONec9tCwQHehQR$=#@+N0kPUb#8#p>5pC;aB*-STm-tk-1bLQ$llGc!9sg=wO&-8H1nb?{=~#*Xa`U)5dh`^jKkEEe-W#OAjN2DLHm_4L0QGe^hVEgx^Y`3rg#zlJu+?#I*g<8*eb!fsDx8wWDh&J2}=$G@s69#Hb!HwMD@K5Uq5E&e4Tpv=HvW58&Ze0jv11R%0@rZ&bovOpZ!n21=ToVNh$b$>?=zYYrdsjFC3Gi;x_mW?I0?n`vWpqk9kTcn?yJMGdx8PO2YJ(IU$Z_O;9;6yEL1Q6G%T%QN<Dq+ql_Xn*=6q)63+lYy?Pn0d+gV4l-!ZKa7w-X;_t*1sB8~C2e!S&&)74do)F>sd8QmteZDJ-BH-6#d*{&rHw=>1#q_5Xb4>}&bq4Vs++7Uj|^@=bbWSJ5|sSCg*kK-oolzICk~iWGEZW<z<YThfk#3m*l(70mY1mA`r?Xnt6&<PT;`DC_k*Qn9F`q^lwOY|lx4kO>m0CFM<&P1nn=St~nRo)GJ~05qb!C8@MHbiIc~L~0_Iflx$W4$0-r=(Y>Eb#ZhWkqP)xx-8JaVWfU+5ayc44iQMM9TQdk_P;laEQ~0sEe{7>oT3LpkR%Hocm>F|+ls;lQD&8Gk(;DQ{ZJQFL=#0G?j`yeWFXZ*^r=#}HKjFDI-s#)jt(|!%a={}VURdlMFd?S<mDjMpO9_eJ!#6k<%?(NG>~4(QUDV}40?1xYE4{!V<!5UH|`fs-P}_=qOp9FD8*o@A^gXcy|W|gX}(}yUl85tQT+elEh%FSCxZ^Z&LOLcI(;0$N}Ku%z-Juja!LJT5TjVntDiViWz5*Ro=4zdGWC6T{g&EiUkt_1O6~JGEtZi+><+{&#jxz2E?DBL3wI~fbKc?gBz*dC<TZEYe_%xgVo@;uv=&~+48&4W@vDr#M=n7)`WV?~Y&g8l`R8i^Q&=%T-w*-Pmi;)4A85VIm-j6eB@j_46M1|10$?wXl|3j!hP)jqjS~?ouThb42o0t>-sd3UOn=VAGO)X2$rUD|)IaKi7tUIbyw^q-6*!mrz67@J?p>+D`U_ht%*)Fapn0G2Z5RY}Gm>mAuGG}i)(fKd9XxmeQ)H&lM5<S&CTrf;$^iof+_7<k3~E@0g&W`NWIe}Q7`4eau>-uggLYo2fBU!#v^N70)?&Gr8&IL$$Fcw(5Bb@&1GPr2iPpG659oe)A2aVbxu1m_(|P&At>G5vYs+S-mOe`f<vZXq4}=a7y9dh`u^TzwsUc9a!QP=9yH5XUH^MV|Wx?+n!(*aEC4muB-iaNJJ))|b1Ybk10M)r({hbmk2VO&FWlt^5ek|fp+5M@_@ic(q4J8a(1|5BLx>n47$LvSjkkrG>7d?UBD<-a4pI@=}z1$HBF@l3aR#r35t}@?6gA&q2IBZ0Zlp7n<X+w7&K%s5`+_t%MHl?<rBIGQIFAV3D3qZR6hc=tE#9<O#BJvc7BomMHWLPw7>zG&s&s#LkSoH6u9}qMRAiIz?6oGe#(K@5p_;G;h-HxGot+!RE4YeZ*sPq*8FI-FMrRbD(c9Bd|9p1jpclU06_W89x-Iyk{^18+}YWr1i|4xS49&M|4FO{6M%ahVEb6k|?uzp^>7oVPwFwwx3o=hsu@3K=AQ}SxwLBCU1u)2?1q*g&(W#t0SXgm>3N%_ndDkq|yBM*hrPMhqpv%q4e(~L9bVijSq`bWneejtRL-oe(D54p^_75$SxF;xUCeXa{5jtee9gpF(}h5N`3CV4XTZ8;<}JHxqNx>km4YkpGJdj{%xYg1qcpA7|HTEelNvi3o=G&7o`;>19oM${}l0$!G^_*4DTv}-#(%DcM&jT9tH5YuDeK<KP4jYEQBM`4hRiWXZ7B^6yQF(xDg#rl*I1PGqx5Murf>4Et%@BpDLuL|xHct!`OXUX;g_AyA&l~HPR3KNIaM&|ijysy>+)6sr%tj4K=Ag}R_8bjS@%(*R;!ADnm352f;3pofog?F@a<%-q3&sDAff+zvdYef`<S9q63$F_|ZPg#5`dD=#U+E=W3nB)?uK5s`kOt07BNTbv=yZTtoDn<HvczPLxR!vvX5RC=On83b~Ju0=jY88}nE@q~Oc2?3f5konGtcb=p1&ZcCJk-yt<Zy}V`Yu8?5Z67v?yX>qY^P^AViGC;bVaX0dPk2bZ=1XiuJ{I=jA9|x@k2L*k8+S%d~ia!n!%U_Mk5&NuSjT-!Qs)!XGiR_ie$jz`seFa3eoxQx87WH>%SMLT?`DSIceEBoJvPK&^l&XvZ5PM3#0cOMIojSxB!)&MQbe~Jc5uyYLk+mUfiPdeAFtU>Y>QwBs4TG{xL>)i5W_${@1uK0g>U!!5|0?wGIS%hJE-Ljg9y)a>%IfEhw->>@=GB=1a$29NG*s);OiINNAGFH-d5ZOY#{$jr_8p^!eVe-7eiGt56Y-SikGZ$(Iys>w@Z0*CO)>U^Kvm07l=p%ihYY!hMd*VdeC^)uE^gZC}G{`C}Ghds3EB6N^|7g-GJ|Z%B%z<ncEH`~s+8UJ|O7fIugzb+rf(S33sjTfWy=ibIm0Mo4f3+iG!SblCjv34`~6f`gJ;knVpn0QVGJ?_dFnuY<YtrkbPQ#FksyW(!uvGNqDbN`o!cj8QgP|6FPu|3>HEI;A)c`%8ldx(h3K60eO$Kj;z%MlN5JRVHoj`(*|L7i0;*PV*#mv<J?IrLJcwef=bdA7#pbWSErhCQVxjg+vNNEH|RzqUV+`bPsxGUY%4*3D*Y2Br&J#fO@7sceC(CELBV^NV2UjtPbY-w=MW$L*5%C<s*6(HhB)rHSeE3OXC$$M>Vz?8#?|;m6+>vPpDDNLVL~>Vx9t2$fb_(^JdOGt3$%rm)t+!L;5E|Vl$Txc&>ph*tQ&qXp6qc>5Ex*UOQlCxn07sQ$sPQ9KVY;8%yM>0pTB2K)`hU&ydC2$@*sw)`;SN-g=k%<aac#WUjm0E!#aOh_}pu9+1dY?QQ|)9-8pSP{(G|e3s^zl4hsMxA~9gTAe9z3Kd4#ffDYYZs8@MXsX%a_y=lAhsVZ#oKE?rKgA<NrdC%(-@nfemxEfXBTOpcsk>%zdbmZ&{5&_Tl-Me5l6HY65%zJ!z(ob!31xwM3{3}QrI9yq?u;bl3#7>gwj{isi*A@Gg^ffXJWO5lf-$dh062ExjSRpX7;j+Ru)?UB`{BvGmvCFktyR|R)WB!-0PsRw9Hl+JSi|p<76P$p{C>f9jlW@-?MeX5%(mWCUV4nK3S0An^387PP5~DV#O`(5_*uluX(O+$c;|NH{A)2Zy%!l?{p%oN{F71G%vgP@oG$|1e2)T2*V&>`#+{)I+A&s}p%E0RqRDtUKApD?E87z<PE9J>Iz}l`pVhIWY~zhvPfmNqm(#?+2lGL^+wLou82LfJYiDZ>ytu@gLmMQB%u|%XGvc_JL9_urg3~J+06asC9P}C~NsVW`kBPd&ZLW)@!<H&N127BeSMhHErSKyjp@2ANc~BA`C@fL3ilAz8Wsvsxk)CxI+{W3t)SnR1G%w#ezk;|~DC7C3>7t+{FCm+7w^;4(79fTa14T!*j_ek%Fp_~nw2kfNs}l5mYFp8%G!#uwH7<#QHa$_6NYkxIwaY)ozyMM(jLXo*5(V+VQ;^B4y`bxDHh^bt3R1uu*QVrXdIWh=ap0aDU~AETsbkw$#;tXuD+)}B^T3*<XqbQ}U6Bx9#2;Og$9m)d(TO}1c28y3*S8s^rXIF_9uH<KGm<vaQKpxb5UF_wNBwj4>;GuPS{af!bf8cNkP#;JG>3>GH}dC;-VR9DF`RWgiIAE<PX_g+K`I1EL>H)5cwfKR{=3Fv=ewgGyjZA6jdeH#?~sLYQ^`)Z&;j6c*OJn0Cvb@zlj+}=lJ&8e-OzG`UMjUjp}G)EL(JXfSz%+PbVSq7+DTl~E4JKhu$3=+ZN1EjO1Kq(rL4#MH}c;p$~GrJ`)Yzsk_vKm9^B3RWK;pbN?L_Fby6V=W#xfjYc+wDLB+V?yDaMM22bqj;4sT=^%eQGEjI|ybr*lspUbLI4=)9kfFPTk&4n9=Gf!3P%~*cb8x|n+z%#%e=w_Q;CBm@O-=LgsqYuTV#eij}+QoKj#JeUl=_yyQSxv?Qp3zLj_^A)!*iHx#>t?#<=F8gG`lRuw_e-frw}rnSk^=x|k|pD#r}s%r4jM8MSRcT^Frby0?ir$@Qpqe(ahK6ae|dIBK2aKyVgQx=;v&ad+<A^@>nCkkoqY0)8sp^z>MFkq5o~^_D9l|tn+?wz`n@-FG{;a_jUUR^#d+^N;-_i2%3gBQlB@10E4NG`^tfK4AbP-7?2d(pGv{?E)lV^S@Fzm~Pm!l@k|yu5m^mZj1c;-+H;JaOQ(D&#0*hhOUiiX8t}M5WAMzYU9mS5jP2v4%F6v~_qQ@j+FxoF|inW`KeTiZa_>>Kq04qtRrXP-D<l`S#B9*qSkSA7g?`ntp1r=DN4LAqy73I8;6&hXk<0_zPlK0|;AYX-ILK5(faqR(FDB%5y*C@4(PD;;Cv?3X%onKZ}hMEroxrtEqJwP?QwurD@MBnQ-$0$l<UJInm_s*anviz{BhMEO4sma13gXgrFt}>xDc;Okq)FimC-FcgQ?gqb!JPX~b-_|gQXsb$~p>sa2K(*rC>NAMu7V<EPi(y`jp$|uIMl1C5&L-^PD27BB9;ptb4+^Adl^S^bWgE`9;<Fi1f!(T~OewAd)Q`>1{Rk3^gAEh1XN-5<iwswK!RP^2TOO=nL=7wmch&c<T^zh<C;IX>{NM?Otub6YeVPse*M$`9$*83UjDsx(p*SeTh+Ot4G~3jP=$%a(^~YWq+$ingn&jj!CVmFsi<E%P@U=~^oM^1=&q$F%S1#jSf$Qsi5bXaek^S#s4gf9Vced(sH;2ePkg@;pkbe-0`7S$Oe-W%+2}~plFM+1yJml#p<y6T&mEuFw^|V6P*QjtBS&T^u)E`efU>`JD3~FZKZ!Rdnl(k~QnEsH1O}0I=p->{X{8VEVBIR6PPH;$AL@IE3?<utdyyL@rs<a-MI>Sdct$K5Iq%oSopmtoV;)zZ})QCr<1{4z0LY}^kRq4KRvdawu91$v*7Y2KidBL)Ma(a!)D(k4~PW+U#&q?Qn0NG*ksk#CsE$XkX3@gwq>zj@m7ZSQHgql*$)}?p6B_|&>lQ_vBNO7R`$N&3!>?^P+%c`RN>V*kULRzy%(wFs*S))KWI*E%(p^^13SbDI#9WT6QqOMnnJFdmhxAeRgk_5YpB2~(+_-3}NC9V_h$6>zVY1Up&&m*^u6kVv>=&ksR`oRjUHEp?s(vOt*i#MH-&S>xtlu@?0dznML4*g?LlOgf(hwo_fZjse=!Oay*-1ki3yOzo6lEHP!J|^TX!lS5<C#$~%=M3Ds^)y*2+?%iQP@Fu+-@KGkXeg&6bv)o64rc>jHj7hvgBN~Z?FtQ$u4+CE#5&B_y^$jW&L}cf{CCe<<K02Nuvs;#++5N_4UB?VCmO8cz%JAG!~;7#HITe<M|^460^a9aEyM%qzk+P4{g}}-T$~%U!gC%7zUv5fw_>%A&e>`IE}_@2)Ql-pp7()mThh4V<5w>6Z*Zxutcd0$<@YS*dxoLV;*UeWXA9lG_|tvBO5?+OQdQmIt9K1Ni-eWHO^ezyp}JZto42DEVO{dC7prUH`g6z6`2PkfJti6=K4T-{!AFaNbyRz2l7QL$&cnnD!?YVifn<s%t-tRG`C(KPFZYvoT#kMp^@Gq0cZ2;v*m4=dxJrdXZ?hFGP_T!&W`=`lwr(_u()Qa(N_OWamFu9t(DLeRY>S1wh)>6b*lnlC$^7S^FafM;N~FCKZ%0PlO<Vg|L4{8||I0Vtl|9xiqS#z1>+|4R#Uq+nR*N5$@s=~N3X*5vFqp>6X!P!5<QF^&RYSK?)FftJXf~1b#*Ddk&k189#6S|jjnE!2X=5D$Vi_ZBzA9bv5&kI!NJ`+2dgYIsD|0prov^uK0OWxq8WGw%YeDWipfy?jE>Ulpi7UUDNk_<#yak?NJ_!^zg0YB`Pq4O4xA{pP=^M?tode=EO87e}0eoInhRNvKuDk~Uj!_P^LGOF`aFu)?#%>UOl6c?Sx@>T#Ydd1c8LTfX?s4LD_ZvNd@&63Dhshuk@xL1wTHx~r#V5(7SX}r<d^nBVUaL2tzHVVTy2jr?!4w$7K@X!{oS(%bj6eqd8^NYE`(lnf1|>I5EjoF1swC61C08_xslnakO?i5kIR%(Z?u(pY>8o?;n!8782e8J304eZyY4;#3nF33&!e$6?OZ-#Kgjh`*;R?8_y+oykcHJo~7diZPb|2{U`vmn4S!Orr-GGXU)gv*P>SRUHf?;-w;REYAM!hk`XHZIZJ%g4=r!O<Hk}B?%%iN{8(FyHd;C*EDQ0TOEoa7*_W5$iYbRv|WBR@v>^*;tAZsPP`qCR<z8($?D$q1Ay$2Q&^JReRoMxC;9xc-3c39}>n>##)fqAzmnY_}2l*(;PBc3IlK0b>Zg`0XTJL*ulYB*=je7qf6Cce=QZwJ$#SAs5u;dUXWiHI3MSH{m+L3gsc8B()vup-*N>9iwTYwA=>Ct3`sN*tCEnm6^<v1#QHSnfX!D(%K&HPHlcR62U`}!r7gvWgc$N;>kfBHx_1eZ5-6AcC8zw$J=X2E{BWN5OPReuW))+kq2sk5Z|mCJru#dWZgB+0f&i*!5>7y5kKtzCGk+ln*Rr@CY|pDCe3}zpdvlXc!MC14f|*ZLcYznbV=fg+}?lwJsxZNpcOV58VuY$qE|&3!z_~G%-5w8;PKY3QgC8juw7J>cZLB&yepBds7}5-nZ{9Fb&$1Tn*O-<eR(dajp}kSWSV@{DKKCy+gC8gYi&R+`~;ETn96FNP*xTVRQj+C@c_K#EfebpF$Te&&GfPJ1vrafCjitUv6M9w@LB@NuC80q`)unQeDLTex1`MyaZ1=-6dy9l$206}_rD!k)z>$i{6OZhd&1P&7?UH51JW*WX8t}ZR{ZpwL6;O0oiScWBNay%u@p&iDF&M09msn4?{$Hxm)ew_qQQ&}$t1g~BZ2<T19bc!*mCbxCMpUsV&%lNNwP*>Z+MC*RsujA9(chbO`82hc)wg7*QOb7@3@`FlFGIOj_675aSB4JXoqhKr~WUmbm}x<@%WE)AJ)@~fQ7S4vhY#<oD^Fz<h_|DlMIb-hT3>*J4Zt~SE(FTzF%7+OM6dKyOk-c3cXDgOd(FO)xR;n9q;!DgB+86DIHps!iOu#XDHOQ<-@9yYEsuV`<Ki2cnirW1`<2U*ccyY2J^f~fB;c4s(y9p-XMqK6Pv$6)hM0>XOA0n;MvIE9uan>`h1^wc}e3>*NE$x^Z`wcb`Sq<Y2TI82Xgv&SnqCp3AM+(b6XV~?`^V=z9R*se!xGL&3X*WqCvdl<O~k{;xq0l(m&$zwDQbps9p!U*ZuDWajR09N>2`xj7I0x!`=0HgJw#S34Den@W$=ZbkNA2?->`~?EXTr$2D+4I3h{R<aqO=LlnpMxZLXXHpn^Bw_o4HP!H!3%X4??^7tY-<GM^Wa_^_>zx(tG4|JNvR|Dlt*&5sHLt{QoKcgFnJh|}%|AZK=u{l%~OQ~diC!W*lk%+*Fl;KwxHUKItAt0au_nBb^Zv=m|3Lu}^DwxRc<IL^t&|rgp3B&&T`leP1EL#cp3vM<55s>VJ3GPDX_mJiH6`|?)2qAr81$Q8Ae<8z0%B9BMWo&|&hh||BAurV=u0Z;J+=nnUo4+n*wM&T*$+7sOVWUY=OF?j-%laZ11;UCjgD+@snLU|{S^Psf!^JrjY4Tv}5A`J+MMb=VYt4RLc-ZsJ6%Rk)YqY)_|3`le7oH&_C$;Rt=4iNh1zTHx;IJJR`w?L6u!7fi)gG0IZ8^%`NjoiHa=0Tb2lH<|zS!T1<H}G<N`hqyeB(MKoy5{UeS!~_7HuOwdY&oR$L2Y$G7MA)-Vmh?iT&(VHkMA@if}C_4uVAaa>#{5!BJW-_tbCpVg^kS|3HtKf@!gnDa1wUIr1HLGN{}U%=}e+f45y#PuFXee&;^p+PlpeEw076s(5VBSjArJJ0y@z3?kp~&Oe53+}4y6bT$NJU7=Wnqn+<I!nt;IsI+8=r;fg;IsoZ&1Exc!2|GRlz?E*~L|0!HQ<*E2iqj<_)Fh!B`LO{ChYGsTo-b7Rm7Oh1<YqIfek+!9N_mY3kc(tWpJf=D9G<<X#j0zPvs*S|aEuDF6p$rn;SOL+NA>?PMh*_Jy!wJ55n?#zc4tk)(H2)9_bFss8(qPwO#LxZa*0F9(`$dHD(Ms3KD~412d!3-Kp*i*<X6|<5>N}|DUzBzQEiYJXtrH#G4c&ND<Msp{UeL1p+faoAN<U(G$B8QhF%u?WnYKf+lwYwnIdd<G2Sjqtq+wGLr{ZUQyR28-|{W~+3I4$NXOo?t9zM}lXql{(ys%m?8aT|_3i?YsEOYWR@)~hT`iV{XSh6qBASErQ-upAf-u4x{W_`X-%(}gjsleJS^xCQ-+c=%kzn1-2cyU{@~2Qv71hfxq7cj1|JbK#x>QmCy-&o0VP=7j4{c=SWpN@oT<55AV7Db>0KWzIY?&kmgG;Ma+>LOOQd041_hhxFnzrTib0*KJ)~%V3QOEa56NLunPOF*?P7hzYtZGVF6n947(`)I%MHJlf^T-mTpY6%umpa`%`Gg!ycE#GnxdbrH+C<+{g{`EmaH0KnFXCE`VMH<xeQE*YxHny-pvZ_5bI8Bn4DPE=@w(u&Q?6fp?*Lr)PGi;td?NU&9yO>^*KdZz0jKw2M;Y;nXF=*UEpRz@y5k9f*L5_BZpY^qY5pbJtOwO8%lPu`+yh17hbL|UFqIX>j8dyltu{!zw3WSYe@saRXv)omv;_DYv_<0d`3%x<>=^>$^knA4a*`KrFcP1VGA^e|RXE3F``b?i38k7XJPE~?k8xeED6wy|3GH)bU1%$tHT->3>7iKhG6siTtc}Lsb!j0EO7MyD%R&2_yoR!Bqg@r|g~@T&*q>}45GWv{QqF^HUSkh3-C!s1JjVpp^xHPqGuORq#s9e<&>9be5o1h-CfsCuiuM`uCY6Ut`iJ`d&b)n7Fil%J`B2RjE0rRiq!CU|<{Wr4sRzKoPsc=tx>v_zL9}an17FDtEJ+XyupxH!h}0Ezz;hZ$D31`*d!%no&!c7rG#}oXnBo$w<5xfSo`@4rX6mJE1O5CVYQOAjG=V~zaMS;l$xsn_tTAyIeYcKL2u%Vjc3IE{MbA#PE)DXB6YG(VFY5W4g~awg%?b?s0c;iuBne(LsTcpi#Y^T=qQ>j^bqzjd19x<(fRdDh@@FtkaZ4~&O%3Uxjx7k`!Uq;DYSJNpQXQtI??%@IvzB_`#UP~@F%8G`IBFNq=bUCun3l#`85jVrr^w$`C^v#OFx)Y0zr+a>G(MZfh2S3bJhOos?xw5=iN0D|pAXj*5`7^+vc4OXidB&u;MiYsU(+!;X`iQNmm~5wz;r@A2}JAohFx5OuHb#Crb|!E93&eq%@FEZ5|zmYN{U}o<hK2QV)8D+2teolN_WrGa8L=w8{)IhPBOfc4GzXKzS7Q+`J~RsEvV3!H(V|T07OD!;xx2lQ52^gL!ed<pzkPbhyT<^wtVsdKeEP&xnW7Gp3XKR-V67$hl4hon{%(qMnp~EDf_qmSh@w;0ucYk!LjlYif>o0MkcjIp2v0v-34xX{;`A(XiZ5)?QRWEY!>4y6<3f=zQl<ccw&}&(7o%^zfBr+cBfh0()?3UFYE_k2GjEM9Lrksh&g_;Bc0G(BHKnRhqRW{Uwpv3jAt~*G_BA^kptoNCi{wVH87ZSpu+Sp=*eYxW-fjupKv?zg=hG6piPt-$N!bQR)O4z+87#RY^u><`U7>&w4M#?Us&#7VanO%fz(}|HG@90ucCPtp(%ONavp{{p@C&lHolUOr{Jrfg8lI}m$Ww?`F?6{_wvDJo@c;LIA;P=pC$|qri0MezsXyakm@44R?Xu!4@Bbf(c|2`=E^YzvT6<KtVzSz$4L(g?Y*Je9IC${*ZiZeNYhwVb-s}%Sn`B>g)%UcUT*&VJ~*n>TMBD=gcyc4f1u<&XSmB+i1opwA~dK<U4b_+Q_kE}oI9e^d{XI!!+Y`@=C586RxV#HD@`CkeDW#0ozrsD4oM>ej29REuELrehSe9<<=-7rNo)E3#iXU1un(%5t^98=n1K35e{I_dRPm;3f1BrV!p=^xEEt0uTf;Rz7WYSFGLy~2?B_G2LLl_heO53WR99VL`C|0ZS`QT%O@YAsU8X-wH5C0*%eSO<$3nm><y^T4|NXQjs{7*ps)G+q{=z)CeV3EZe`92BOg_@Y`_c7xV99ZDwT9%N^0rCFo6)r|68!Ra`3Jmg-hd|o!BVnUG30Y^z{3PWia)RhQF%xO(CM2XR}^)6w*VI3+S3IQJu2Hswi1JWXXRwOLm2*Q9B-2PCr!Gg`c79wXAEyb-|v_;j_Bvn^U4%gYoC!*(%WYI!g-QoGRfLL-{D!rQBU$3#L&*5xad!o0mh^i;PnsF(e#{wD7L>>7ywW6))~{&Fs$BdV5o|8iLqoCOpD{iX^qr+N4P|cWarf5_zNXmmIE~Vhx2^>1PRh*=;wj?PHuDJtQtdCl8@;a05`D=ap~@da_v1``sGKMh@NMl46d7nO<RhPAX=;RZC^f)=Tet*eZ1t)nu#bI`l`j<31Gi%OXhwgY6OK-9zZq^8d`?(EBV?6R5?e0{?hP>21kWOF!q>%f~p@hz3`c0n~F&gDy5Xfb*O4#e-C49$l)N2;hpqZ;$e>^lHchG68!0R&wX65&<maA-i<&f)<$1m%BoHkEK`j2ullNxdg?QaM}5CI&<1YtJz4eFUTg1JZ#$Y#E$HqbS+x<k3skH#x)b^Ki+aPX8!JqZk@H~#W4m}7raUeq7*u}jclS_6gOi2wL};!#5y|&ysLoD!KsGm{elA`b;DRIFy_U%KGtwxZ#ZGV<1)EcX_aF;%E>v`XRBk-W?>l<QvIatH2|9iap4io|7B{#oU275oiH1lWr+z*;38Ty%0^}eNBOUpnAJeLqd+V^keNr0oBmGPFJE3X9k<0-`P8;e%k~3yyDId89ZlPCfc3&@5N?w^l4645{#c1sq4W)Ze(3nHPr0B}S&^oaTk)l$lgK^A~HVO)5-z59DRJp9Z$GbIE+(oy=DJ|lBzS%8ws$DJYRR3*K(5RV+e7=)mr>)>MGnLe*!r^P?=4zDvG@-@dw7E%3R<(4*Hm*V~P_oSJh*915P%7|~xZNfet4p@j%VeJ>-<BJKmvfPlgMvtIOHC_q{#klr&hfOZy6!>*@W8Gsh5T@_XF`^N#)>taX9I@tevH@0H<Attq}uvQ7EK9I;CsUZUlJ=k5u!Iy2F`bKvC3g~xavF!XQq|hBr)8Q4mcZxCMCj940Plz+Fr<p@nb}vJ{tNK>E`rU2I+sx9=?9TPNv1iVL5d!lzqffA#~t$F|Qu7u>e9teSpl#metKkUWm4ooL#Vvt$>0a*wWXuH*_)xP{+Qdk7#(yD5lJ`GrPEm&L+m+P<!bRv&62Rk89EH3scl`aCxF+hP(o!&y3_ldthQ8@4SadTsG2({GEe};j~Uxt806*Wq13R!gFWV4S71TSVVJ*!tQj7<kZR7yo&%s{t)&uv!|(I8BZSny7&}Cq)4Iee$IE3ZSJgow%b-RJ|RSz=hd!rE2|UA>GS(^ygb}OX2%m#eyEv++h0o!hT-0`&mU6-9Px_0M|3j(nPhZ0rjj*Ggv6&%-;Jx7$#yv(o>p&YWo^e1`_DiC+xfh2aw4=Wo;rqI7GHS(obIxtTaq^2PsmVM!>$dJKe-SdT5x<&I2eV-6!X>q*iDzcTmU9VrB#O@zbz|HmDDGz_x~L)VwqWC2uWN9NR=@U9}rdOrfwT-b*uk!+$_P@Q9N4(TsB)}*m6*~%=^g@d8CQh(VhCb`GOLa-{Fy=pIuEU*VqTe)~+9f`YAvD5iol9$b`~G!E5n!34nJg0kg~hQsdIS>S|o>MGBz=+nwz&Gmk{Ad(KPqMu_Q5?5q3T%#|<v&l5S~u%#j;1FAz+mfCkUiX|QgN+2zHXPi*{M}bUSEe^vJHKP8bM;oWlYCPA=f<wp*P<2*Qm5^KyNUK5CM+tLpk`l&h3_jiQMD)7<ZG64LHS`-sqR+*Z%1NQ9V59nG6^Xx0#F9PJPMixqs9#lna%?%Ovr7=QzBK1bRy0QkCOQ8wf%a@o?8d1d_pimUC)-#u-Cu9Hx@WD_;RNn_j;7f$-KKou$E?REqgdMyT(D8H=*#!}f&1L?3{g|^LKa3Js+Q>@RF^$gjJb)2Bdc_N*<v5-;%R4<;nj+RVlpTaTyEm0HMXcXs*_oEjs_b%Y994xEKSWu^@6@%-B5~^qEGxmul5+1Lsj?iezZe40{G2r`)4W?reQJxn=01V?9y`7x7jKYW3Es^4)-0-9>x?zNscwpl#4ykaaefVCXXgzYaAI}JYP;l9IgeDOTdYOD`~MnHi-{Tl|jX42RaFKwNu8(41|hkll6Z60OwzudPq*y5??cuLq*SqS+(#fNq)&@M~(mf(`anWYZ%Kzbn4ylFK7(#kg<r6krKbLkJ4VnU9UOGQiE_hRh63RQZ}{x1p+OqK?qrxbYm3eP4Z&*=zNMGdA%#~n*=NsxpBIYSM#rZMJzG>4vzo1=>XP&W=Zb9fj|@F0KQZws7B!$yG=%#;|bmGHIDjx&n_PoS2&)EC#|Q~0}V|d*2{h4uBXCXI>2(9JKH|LbfFBhVq15uFKcT5FKrt_t6@vuuE5A=qG1#yK+2xqE#O9>3CNFXV{Yt6%<VL!0lD;R1dvC3YH<><KmwTTW=5bX6S~4Cy^5j%)LG2Y&5{RaGHgwuo)l4^OL>Yj4=q1*3x(<X@2w~iaOwcQ_M5_6$15-`JZL7RV}lvUmV2c(P|8A0VMIlFzZC`;hx`OpgXKGv;$o8}pW9SWw%f?Y>uD}xlI-tCa3SZa?B0h_DWE;Oq_mbB1Hl71CA-JR>M6f<g}liXqu2Hl#qJ7F<OT1aqpi9R6yM0~Hd81uX?R+}FY3*d6Mi#Xc>)|Ed<{#>f&m?`utta?kOgB@y!#*u-zlq3hQDp-h6bb5NRdj~3a*%OB8OlW?e{;$5^)6WdQhX2&C;a%{vF9tP=3u!!{)_wTH^@{YpiaFxYEJDNBX79!?OGbe>Xlj5%6zqp-`&_+?zx1eo@<^b3r~v_DbQNVksH+EgE+wv1{W$3fL`lWr4bl50pEp0t8^nYOFw&Mn2M$F7jJ0;$moUp_hCNU0iM@Eb|guSJ={xS*~*Tv-3rj400AJ!>B&amj1I*<UO8Y8r@q!Dm<cCY9XVktx}-lq(kGaL@~HyYz4SGCIPRODmp@{l`?WMnLkFg{tMOH0@pRT9mBh)pG2_M#AP}D1a`o@(e?xdvg%@C_=}?w%bT9*6lWbNbd@3iP`z4I;gU*0s$O^ngVa2r-jZA*_)o;4P4pPBUhP{Zwl!LEAqChYceBz*SPN*qvR&c0sCjIp_0{{%842d)59@sEUbxOWA-FVdp4>kVOwJDE9`TVAXGKhb%Jlk$JWyUuv4tJ|8A;FeocH<?Z#oGPJxD}*L*<KPf6>Q4?Ia9GWNA}pC^kd<b$T|Cz9Y95*3y!dg!PAz_fd%~@*q8Wy`MUlc}i@BzWd&W9~@o#=$2D}Ul7hjp$U25(LPQH=LTzvTz!*w>3aED6S^7AHc`sxGX6#cc*9Rg8VFoYPax%jPF8;R;7^2sqag4YF+HWy_gn890JU(NEHi5h4k-<KMJIWq;I}icPO5s7bWRT(u8s5hdDispfap!cN6LEojraE<7&9ltghHUyRuFV86DK7_M^cr|2P^(*)oF`TI|u!93f0#%gdD84z)A;6G3gmNiO)WC<E@H;t+NEMopWyn7cx9q@QA?W(q^GQp7Sh02Y(c%P7;4<5LT5=f_vk|AYn3?haCNs|D)-kK80q@1tCQ?mkQOqF^E@WPOH-`R`%cPEpZ{5B#s7;n1AKV&VH`s9ZHHfBIhAGHz?gO!)6!b;;;r8d@M9!16xQy_{yhO4Oc^p^cS>}`QVxzcYsp_e&nCcQooqGighK4pL^}9w^y*RB;G@@N{Rs%qd|!-wRma{ZUbHrC>?9CPVy-oJNijQwlr8?*X0=En|9)(!DB&x+7}<0)V14qru&IVjex6&v8)u<R)XL1?+9<EBINCSeGU@J7>@-U;kM{tXAsAiJyck1neXKNTbtIRs8f1N)zH?U!nb#lwzg*I14^VI{(CI__jF<ro|~j|rFBej+&s+&jI%dQX$@U6OnRA;vU}F^5YSvmqaZkCGZZ(zyO+-8BSaQUW(i&+OC_9(K)8r_OkW(`^tnD&Qml4ASRCFLs>6rqS}_t6Uldu%=>Y-upWGyd>gBh|19VmC5~E7HE|{ORL&MVA)C?V35~3+H>=!{H7Sa%P!C35nhYD9s!mQfg#rp4WQtdZDGy_XWEuixoBoB}Mod2oD?r)(X!9~)h43b%fmy4wB`!Xm6f4~NIlN6XJiT_O7?3%YJcA`truAew_S!(4j1YpD-1Q}U*pG10%Yq!28cknXdD(yMfPC{O}R*1uLqKJOaVRj4vMhP%fzQ215w(j0ql}_F%mMC?25O8iJNh<qhl;j$cQA!q<{&`8^^c)(BhwmG2zOhzF<u=L!VL5jEgY}H@;I4SvwCY&*Ou9@6FDJXo;MMy8N!+wVGDyX{^_Vp8$sQ%hZJ7Bk++N>&KS<~d#zP@{IWEQdHaev4F|-vZHKpS>1G{(|Ov5JTqkbjr-#e*2EGmtAM&2J}f0?6NMy~*o-I!IgOLBDrEC`JEr+Ybk%!77O0jY-!8dqBmQ{#>Uy4KiLGp~Kmgql0!K6{33zg=}t$Ds+R&;i29&IYe|(?}aB&&a5UPYE>~38ikr6oH;jzlv;RWN~#pmyu-uN1bO+G;%c)Lu*Brmi#OLW;}YcQ9Clg&CJYzGZM_P^v1^<OSOi7g{@fmgI2>;b;>NQr)H8D;7xz3rP~;9f&}$~{IZIk-+%(anB#)E1XGPd3&(xmB6F|8GN7liE)XRyBh0mTcB8I38C52if`k?YR2(t|9#_RZ8*rpQr$_AqL+7ST+Ta1}1mYPYMJmgDs?V!%HLJ5M-#r)8{tLi@$*Z9!V0Trtwst<?v&*c=MZjP$7F&il40VnVe~DTvkP!-Trsa{9iK!!%+`Vwi=j5k@>E3Qi2%tMGB$a%H0E%;}C7C$(b|VVB6_^mx3&5Zu)B)Wr%+sTDBWSSP&&;BRI`i0FZBm^Urt80p?=A{4V|`tjx17mS3Uz>$<VaS-ISj-)F=7~ApQbQ#A5yM)^ym`x(>r|i*%=TJSiZ)%NZ-%e(4rL5oegV)DWo)B>bqjK93Teml3ew_1n)Ng8pUjnmCdBV7lkx2g%TNHOhoQRttd=}fKFCKAQogW!%yJa+W83O)&^k5W%M|uKWsz_`&JB&LI<iLze!0KY+}V>NwP)H3kRiL&j6kw&Mgj3E`S6ePuxVTbVP;6k?S(+0r#br9e1(9Rc!<rFH9syd46^3+T^9<pN-v^SZN46a)sD^Xr|If$+}bQ$8BD8#2E4)DMs(pVGKoNxx*XBNwYw&LxmUy5#x`Gxt<AD{qANo41!TcF)am9*^L4ESwWx_aU9}Mk=asBb(r#ctf%rVeY}WpSUBl>EKSx6I?u^s>>?RE7n00K6lo6MFTvP}cRUl9-$#?p1)*3ze9C)516uUfqVtE*EA&py(0A~a@`OqrU1n|liEy6~#07iPE1GLCFl>_wll&`7K4ek*h(^OKe0O=ec3+-03BVbnn<24}7)RBr*|(g<f_}P(MKkLR<8_=}$2d;78eLSu6veBkug{a=<|0r*_aq*Rx%j!DJ>3ZI$t$r#Aih`Q)O8ttIEhY0zHaq-N;0<JoWNO=l+m#rNSJK9!T+l0`f=F&aOtXLcng}pKEfq5nY=d;^v+k1)e}V4W7qw&bzH*Tmv0=VaQ^O863$gFUFSPuyQnxC_w0N#(qYFHqEsk2g-H@!xFIt$4s5~?(L5xM5f-Fh#iRL1NYN2>I&%Y~w&dXEMKi%m%~P_>()L6Xf!p*Oh!TZ$2{71wj@9PpAYmAoaO^S`RFg6cja70+{_*jj&P{5m>561y__j^KvhR?{Z1L~4>y8hP2{sFT9a6W$@RF{mvkQkK@&XmT)@UcUj@VowV+f(SndK^cAUAx6@zl+-)Ewqd3k1>LR@!O0G}wxpQPkct#9pLFs*^TDu!!IGCjk|vIsUkW<CMQa$U<Digo#wDw4>?#!Z&RNomOwG-2taIH<mFiNIAZhWn0mjXiv;6{IIctltKymQNPQvtu!Wb>p}EBCjyy(^6*8<T<9)YF;d}|#MB`qZd1z=-Dbm8d}K@y$cznFB4Bjt&>xgR%NNS%-mj7IN&1Z&yR{Li^Sh@wna|H@H0vQe>$1K1|NGVSbO}RMV#D6fBT{Eap9kj-LhH(omter$tzJBVN9X3GEcsK>3j?J(yg*3BCei~M(Yz36y7?}2)xvZ#$zZTyt#*5tt#nJ6>z6wd)oPFqz$6r<^Big9C)6(J$`BI2K77aw3j7HS_I`wV();+ipl~??>lx4Q!;McG0U$Y32luwH%CeY9MP&4*_N+rFE)?{d2XQM?35yxh#w{IX_m;BwvQnd)2<WHn4F`gEOF#<m(e=*fA@<fy-V*&xO4Kch%$NSc+6arbl?{um=+7!f+e+r@#&h3j2p)~ir-UKK8f#a3yj1Z9w(>`=rfH&Ge7-%)#I_tid7^HX;~F3{)zSRTvQtPk?PgUx?;TNW5GcmVKt1n_->LOlj_K*JLSp&_W*;*+y2UZLLuFrEI%m|nd1h`-Hs0WUo3`@!cI4J!2_DeIhw?@}`&?uUUN;51SQqz@w3$w*oLdJrYqw`oC7JC9*7v4B>u`g=n|(}NvU7=Z*?2AG1o44sWX!w57|Mlh?Ik!U2xO>h@q@2QvA$V<o=nE=?VqUd&MWr09U+|?ZkAB`MGTaf9{!63!R*k!yQ26=H$0umKw1jDGGt5<$|(l_GNf6F(E9Wosu1aoyW1JrT*EIan#Xt*sKr$!MJ9y&9H@dV=ov?a#?A$lPg?&(Ft4k}3lKPc0#nrd8G^<J6Txe-f|wy9JTT=`n7SP|qMMJa0S`q~?d%JrgQouGu!DZ@l7d4{Zuq}HC^bprhMl!dTepBx`H{qA06>WAlfQ%g3=YU`c%zrJIoTTAsEPoq&{~6#1}M8B26szAs{a?qEj2?bND9jwF*VGVY=;y25J2A2jsbBjnJ^e>6!eBAvOc&tukyT#?Afg$XkTadCAkJbC!`p^aQ03W3lH+S)_Y-OKyCA$U%NXBs6<zPiyU1W0+alrv`p=zePFrYoNP04XcR~YhUccNh82-*F%oxmc-@8tK{^~LOxOwq6uueh2yR4b>H!sT_P_d%5uo`XZB5g%{H2DcI!eIE{wszJ0$vpP{k4-f4|4RY#4ktlF*)VAUZBmX-Jx;^`Qd>NBf+-9nQHV^f8|rDN3Et|<)Cb@5#DB8;sKwgfy!kBR~#{L{N23YHtF%DJ*R4vZD!wJR5NNa;sB5QHe)6Xxf1MVxRoQ&`dQ#9=V+j%i069gYGX2|Lqul#0iz6Z`u0XdWTD2`1%Gnku>J5R8XQ%yawbIl+C5&LS%Znl-qE>NFb&vkj$%E-e@6AQ_y6xxtAmBoOnY`u2fvQ~5JC5j9X&~>mOn%q?&}cmiZknK>i%^vy&Z5)oiW!CjP89b%!e>BJ)!%p2l>VU6L)4Dfy-7U`d@>-l2wVf0>%{K7XO=|mcs(mX;QbC4<B)DzqN2LX(S&J>sj+-SUi&$AgGscE>~w1Fv=mP&lG2^^CvkRO2rfyV-H)S?0yf2!;!W6Ry<YgCf8vdFwGz<uO{+wMkK_DxSppKl|cFAgTH0~&>yqy3f{mS@mLsO0h5+;A2z(Ye6KITL4##D!kwLCrZGK?ysoKNODNL)r)K6g3sj9@g(b2eFy14IkNU>5Jy{86cl+iB)37oW(a*PCeNJVVZ^aN*jx$rSh$wH-H}2_LWK!C9n<DEI7lg-DPH6q<pi7_6VHDm@^>DT(e?x@Uj`yVDk!pc&FX^14e}I#BQusG1v2X0z;4Fqef;3vN|CSQSec`K?whshGs4g84d*Y5nTbgkDIW<&LDLMq|^R-SkpS>8#K9z*FgzYyl7_NjzMYX}QtNnt$aVU_t0LDKpn@+bPG8y$-_>!R4F!H*3!Iz9!9MnP^_4aI%umBSS5Z?b@Oj4ORn>D%KsaD}8s?|o=2ikZt$_?$f;AgG-?k7i5^f-!hw+<ar_er{(+l)@KUjXQ$N@T#)FFPjZhT6FE<&<K3<f0n(B(a<c-1okHBZk>+m^2J$;H3l~_g+FFrC4|tEI!s-5tSC(bJ7V8hVJc2uitdqtS67waMK8~b8SrO5g~T@dpD09*WJ(tI!lx{LfZPWzJcfA@8*-+394Ugw3@peMr&{KDi-c`1AVqpW@FtlK^IYBGfmFbPY<yJAr;atvg&BQr%v516ZDSy7#L2~1>G0V&)qFHe@HHr&?2KP+Ef5px^1=X!*+jmpUVV#{!I_J>7z^_7F7o)W4K>H1!Pz|@^06}ePRmHlt;vDAhmZ~XFm6jeYuQ=+PM`3n2}~P((No0l!a1ihRuf6QGvsV2+ZJqsbeT>$ZK8q;|{S3%n2tWe=rZH)YvDWQQ#==;q<yW)UD{vnk9B05fvBhuLREDp~8ihvkx~LrHE^I%QRNkNM26GYW|`8;6Xd`IIP=`xu~JdTsw|JUj5ZC1o%Vafq)(AFpS_O$x&@PN#b63Ud|skmGiS)&{G)E*dcTis^0meF384PSEUY8?-FsrobsP#kQAt2C_*7N3q2X^6Tt>eWnxRNSw5>yW#6*WKA5yspI|2FS>>oqa(V$2Yw{gBKzzaR1Ta>)BrGn#BE`qxw(NAoYG+|KL~68(M3iu4e2lR!7aujdtxgId^3&G%1*mONKO)Np6?e8)(eX_(=1QQYh<qp;Dd93D=b?8!W<G$&nX=-&v!Fu}W%wrG_%aLrb_F2PRwsc{VwPt$OJk(avPm{KW<9hQMi}gZNqXXlrZa!i0_761n0(-Xjg6kF)gk20ZaKdtwe@#XC?=W9j2M*q>H*~;hoAb=%5m?gi8Ptt`o_Q=4w%bX3Wb!<ncC*ygMp%$?wjm6Rk4$bxUs$(-!VNlS4a#Gz3U*AgB|=s430rwofH4Ke7{sgqTbY^obZGO-O$_RU!69vC^x;Twwt{<h$XiS$bsNraM>NrFpUW}xQ$|!UUU++`%ScTU3ccF2)Dbg=Nrv+avgW!PTYc|hy`9;P}nUXKq0wl4#7t5vKQ+&T!R~%+No1YI7P1YB-FP0jed+CK_=yIS}%RwpE=?4z=TRpbi#}(hc9wIAheYZCF#UvhXIV$>fXCEKxHLi<QH{bMsV#r>GthbuT=(6AP{PW-i(dfS1Rg!FU5owTcJ}0^>QZ&sA{G;4}#SaK;yC7_4)LH3SF>jj+Yky1Kr?9nnJ~}n*n<4su!ZnkR6yzUP4Nzd}nC=UK#<0J~v+d*sBmg1wa<00(11uhwE1dFfW)LkPuFI(x)SJ`Yq{78}P8}jiY}t=BZy{MMP0P3cp%i|3cgnmC|CEFW(v_`@j$M=T_GG`)(0GsDK_a*=i}<9<s#mQ4oZ?N`W6~XIl3z;BtfA>KES))Q&wIN!ZB%X+TP$tO*KyqvvmqRJdpG2T$jeAS4vA5J~HM`Nj*T(X#pxd1MqfFd)tiO&zoe*S&eL`+e4opC4nF`iCS$TFYU%vY&AG2&f}LFdM%9=o$Vq&*)mtfm&iRHA&B8MaC`z*n3HR{j#$T#5!SDJ{pzv9($7yEtKq;g2g5%Z=_247P+wdm^gf%6fraR1))JZ%B{*Z<OzcsP)-K6=1!M&l+J#?$}(MiMG2O^@qP`cNhKS!$f}VU_1JQ*zY|^C9X17WKVj7{MQFPc90M|O{!NaFaj&kO=@tV)8X*xD+fm>m#pW1DRo}6y_mNamU*wiWhWc6EG*@B$>MG)o5^jmk)RUv+S~Uj4<QB1NC|by0;evkC_$Yan0?Gt3rsOR21+ip?#NQ=HF+8mXjU+SqB*<7Z3~yM=`5l*%5mXy%L(ssT_aBiOW{(i`08frITt0Sq9JSAMtx*?ds*qMLkSf1Sm|YE7nC|sccFkh{jk+$<AoovcDrqS0Fta|dQu1xKX-32oxsY7VraRQ%VS7iOrW9rtew*@nMJ-5rq9%ESIt{9rD8xa3T^!cdfb4Ym1vcs2(cHNc4Nw^UC`OHmx4rlwobLVwTs4NRE~h;!8~@Wre4FS(VCMLzSu5WsDZcVQc<H#{ox*mXDO!c85*jw(cOB+Y)J*t)+VC0LV+dN(%a^>k@T6MIbdPp22(KesF{Hi0thXjJMmF2{6B8q{z|xsgEBZmqaIFKIE1l9*GE!vv{k0SvG&32ligG{j7qmnkm~j<OoVM5>R|k^yBoEE2LWa`59DT?yrnkj+o{_r~ZdlpTV&B&my=F=wxYTUpF0i_Qcw4nEsS&WMtPIoTXD(wx9G-Ja3sFhucB_YzAO05H%Ai-ZzYt*>;zbRhaYt0;#Y;kg<+k-c4|GZ*u>uiC*pMFT2-6YE%|7<-ryAQ~tf9K{Rn7!Z+$x3wMHuzCN|sETB<g>EGJoV}zF8|@m2wk{G>8mjYq(C`?Q`+Yq1uqT{HTbsn5Fb~X+Et^%`Mt&S2ShuUm&$4GuJ+8Zcpv}{N*SUBn-k@aGk~o%g^D1%;Oqa*U$CGUElfl@N;{M{kAzaw38{&gxZ*_i^YVa;mbQ5HB)goIvoa=tGdX6XC!WtBzjN4H8F99hMwg>6P_(#n#+nLdsH~|AV_8r#CSWTf-!(V#lwD`>GzHcX6KP?T162_*R%GHPgdGq7xL%$*V--`OqeIo!)G-x?=Zf#p}?KS^(ytE@cYGIBx8FLvI5ta_xJHKIp(5=$*1dYARX)HM5+M&^8c`*CwfYhUMN;hk^_U@fl*V8NKdTwPfQxLV#CCRp9jOeQtu7QHXvYOV6bEYJv)k=5t7LLoX@c9@$~>z+4_vwPtX)-i_p+6jZ`08Wq%?2bunt*EVYh!7W@s4!2~%J{7^^QX$r_{gyO-ReX>s#0P2#vNfZu)^m~bQ9G|vb66k@XD}m?rZvFqhGI;!jjDU6&Q8B3dOwmBiD~+#1TQJ3R6VJl_)%2Tta!mF^rmE(A^(Ey4rKTE(e_tB3x&TJmj?UQj`4KTkKeGY+xV3mhaqMhy@+r8|VuKl)+o3Sj0tAbskJ>yRf`0Ges@S^b2|DbOJ6y%l1YWCPcs+qjK;h2+=i3a$@z&`;${sjjZdXsG-&>a1b1-1R3^Zn62i4W^n;B4#;>4Y5tLI`9?;Iw?2V3(}Kf9nHriCE1V+~bwDlHU2P|>TxnDt*E+|C_%Q4F;Ual8aDjTb7kwiS;Mio5fB)KIly_hm>KbmM*R%+<uua^UCJ{fCyr?8wr1(+Jrt;U`=wmdKnh+&_Fiatqi))r9NrQE>tE9TzTnj-HrlWJ5>}RPM#n4kLwqR*ZsODiCHKjA1&uqrwB?n)!KJHci0wEZTtHF52=Jl1E0w6O>&`M~aGXdG+G60PL7J!JEsCZMTFV|Ml0KNCM+_MSkc~UmtpCXDw_Og|sFTa{q)jCyC-cXr{KU9Mj^Q2<||gL+j(ttv(i{b~*iX09pA@b!@X8?1ZlzJ%&Kn>LI+(ZQ}#;ZR?(f=&l*~#nrxN7MVIIM%bNt@0qDxWEhI>+SgI+e<1zXOSMQ?@@*v9_?QYn>=s|KZKjEFcXhewk@Qs5<JQ~TBHMi}>(f))y?8%^%PzAxOSZye`4f4lu<BDxeMA6RW0LRCE&9PD%nFtH)0Iv5vlUD0iK-<q+iV<sh37U&N8bmEHV$-?a5(U>IR(P(x(IW`f48`-@<XPclH?r!0<22N7ij=Y;tKalfFn2h28U?-6xU!hwEMr;tx{uXj7ft4xxhgt;&p3LkYTR8p`t||^8`(u6FSIymokr6H7q$9tTOA~{V^wzzf0fPkhe~&(N*@RUiGbZria~B?3f3vk5sj&MGw{VGspZmP(s8>d`v((Elg!ziCdKMtUOQhyIIp6!C?YX5{k=Hb(+XzH|4VU%@;3tF+BN?S%8k4h~_y@)F6$LLIE*v$cb=#!?oz+l^D5a&!Jl6)g&K;ETQ`L8@08sz}>XhpEp$iqw<Z~{|q%(v;Ok%OwIDJG=Y&|0F2)(H?~Cryy$c#uF)&4uT1`G$bP9P#<3{;h*CDJ19#B4_uVnlWP2jH1Eb^k4U)Q>(J$-WAJ=;KY@602`B=V@KKaYOmHmL^N$<Tw8*}QU%VnIgWcd<cMS!YiPwVxtxgW!LwJ()T;T+#3v@NP7gVP+>#$G{Cfm${g?j*AhT>0*0Bg8l3MeE7~1uR^8jfh2nS;Q@<9bE41wzSqVw$^i<F-`qjOv1tqW*^hnz=io3Qn6x)4C%?3O9FO|ZC9Ge1?zZNNjJBQLU@Yg0&>EFcV@}e7sWo!N-%B=j>3IErKr!23%6UB5k7g(B6?M`D6UI@_dYl(ei3Ms_a+%|@|Vo1E-$v_z=k=Bb2A5l9kBK^qQ0Hf07}6}r|<d>HhIiO?82lSUPvd}+2BjFQhuf>dNIb4kZQM_0)<AoD1E;_vdy+sd9E}`;qK-ZbQq9eRF^dvrAgYn?GD)Xh=QIWz|f8Z{(T7@C*kZmUzmFa-BTJPG2j%5dgCm}&BRP6_tKMYt5iw5%X&|yzjA5cmNzS?u_C4o$Jfk3V*#CQc?zmA1tgrwAr_ZRUhu9KDL<+7Q`#vsLdSgZh5u?f=6H)a@_3D;o|h+tE1#*QrcunalIzVP?c~>UM`+-9Q6iwZLp|9gCtrxpV6s0w;HWr*gKX`PT=AUU2b2_dS2I#FI1U6G;u3<-79_`|qEPT(G`Zm9XaVc+697B|LYuM!d52>mzTWRs3rH9c8%O3oVAaGKG7NPH)ur)QK+CusHmDb<sNAp_j~01%nOH9_q6>pT`blr`q6@(;?vhdSWJ5Qf>8rl&vB(r!DdBqbLbF(j)Dhjy*IUAe)Z4~zZD0e4WPv%$*eI#_$}cR9z(ID+-m~*yS`fZN^61X=Os&1;L(p_f-H5(mOF^qaL}iv2_d}#eSUS^s4W%$ewMjELj9KIdg(9Qd^Zp-qvW0oj%s^bGi*{Bf8a*ge&dw2AvP>aXN@dSC1wL5|@eFwkG6sp1n(Q6pIFcxD(r8rwVE<x#N|}t#{_%@=2B3{BXL!8%`0@T)mVZMLcFY+vU}mCNP0UbX@#l61XhIJ)w)t6ML_Cy=@d{7iAZMvkNdgl(bX<yUI4{Mn$rG^?kMD*hFsi=XnrGFPd_)N$v?btj_3}Z~<$cl!sD0$3hffQezaN?d($G!IIq3!|*uq<b5Oio8c;_fmpW5nPI+=X(*a@EK$k6%E`)(l=!QS=&%g#zVLvlJ1Vs(-erblYef*}l`Z%stD?I<_04BZVnMo9cZJ!R?O-eo<elf1`A`xq6V1QBdUrHD=N^h(x0$SeXoq(MTVac_v8=sV;K6imjZoRR(p?TTXoXe1)<WM=4{Ekj6{q9`B4>bAfs$jg>A-&iQlD}fjK2KF~nZ;xbj)Y2VfU@okhU>HR$wdssDLpqyCVH*zOffor8mwogxmXv@mMCHufBl{?xS2!FZq;DP5;>&H8^51qHyc$@XlL&vC1%co)D+eS54;aW4Sf&#qcl>P}p^(F6I!G0-yB&KqC~K_9f{1tpfPr?PbI79%^E0GmN~EVvbgO1KdjecdhG#B!G{<gHkQR3%=2ICXZ8k}Kp&L*Nv%!sWDtfsOi&V*o#jeWM9WE<%W~^4FP=;8b&@Z9IUWZWo@oyTB6W*YF{n|okZ#*#yj8L-leo)r9`L}~m==Ow~!|%uhKv?aMPWERgG{Q#Sp-AKvB#@0j92g@7{8v_cW6!pA4FYfgQcI~KkQ%koVX7Xs5AvV9Be%9(bCP@^wkP5HWO}6)&KeR&id^@<{HP;9II|DSzZsS_bal_>Uuh!fcO!yvs#@UDHTW1{)=+BBK8%ioo;)213v#1fz>e-%8qWBxnaxf3#%BP6W*|fImhxXNOV?ymwl}MD$BX!&ic;hnXdc`{BU?QZ7Y$@Dew*jHd=(^+H^zv^YT7+{WKwU+E`nBMGy;q<rST22#PPbTR;IhJS%&~YrUJ{nv*aoOj5^295FkcLy28^L7%%P`p+MnLq`DCSnceJjp=T$Vs+S6qN}=?_+*Z=c5>HePg&HJ4``%xd*&&FmfNp;T^YcRP`Zf0cL45I-xHdyQdq@Z5<RrCryMoD(VplWG;{eGlaa6uD*<W`7Hz7}WSw_8AX9V%9`^sFz8NX^{w^d&Kqjyg_qCxOjrI!a;O2rh+PS?Y#*j`|x>w-UXs#fI3Z+;?+z!NInoH)#x7}eWO;TMaO)u7fi)$2C^UnL+$P}E`H0#U_CKHL(*2&48ow}J|#_AHBw{Et@^iu=KTg5J0>VDk<vAoqX+&}?7ALovgc!&`4|V}R>kzNF1%&VgEqA<iT53pFQ~)Giq4d>&dATcW8q&eLDJ?@!!4oH#2n)k+GMpaJ5n6?BM6jObax$S-k3R-J}|d6H-UA0R6V!2fhc6uZ!W<W+>K)+zB<;XmR89o&j%d$!la0{0)m$+yyqQ%np=Ikze#d?fun5PMF4<qFRM1F?<?bUnzUwEQMkVJL||ErIQF5~NH%GzjOv+ewiq$R5SBLa|r0;Ja38-c!2Qbs)eS#}x8j9Y&LXysKg}@&oz@8RDkFGVXd>iiy;Aw+~%IGi>ji`n*fDFSVOlbN#AI9owcl@irA<RGFEL$~@3KU?4<)%(3HSwiX)66?TU-OH>IqjZO|OxlKMpC9_i-h@JH;`H(jm!p$hqg59WAxd;+gP?iWJKST=@F9!T}V6X>RS7uRU;`l~B<??<+Rx#Fo+0F?}=wl?B7bo{r*Qx;pOf}e5x~?m^(W0rloA<*`ktbgL!wYDeG&M4=qBnDwp4sgzS?docpeIEiNS(de^n5PpkC3vq*tapCnJE3&KsE5qtzfIxLNjv@Nj@G(=dlT1)OA|s<pq5iqbytCrg^_Iqpc#?w$w<wlfi@|Ss3dXmb}T(@&$FUYrxVvu79f%_CEsleyk->F8|wDflqNWE3J00-nrNrzdlLIii*Hk6ivOM9Ya8hS!ndaLXNK)P;UJQfNcjl{xi*0i{Bc<W1!JAUm})_OK77gu#Z&{ciZb(`QhnRWOM;#)V>^!k9)C^2mB%KHlo?*|7d8AwwtJ8_pQ-ETM&aUPFxbm^=g*$+-L`d^xZMbMT{D77;ff@M*&(HWgw>C(Rg?kh)>P!E%4QtHjl+AdmFE$)S^o|`J22wR-UR9xd>HOo0p?7&{)&aS#s-1gf5LN4s*N9KkE!O*F%7gp!GiU=_ZnHP2;h`AY$5e*dO1*qrdH<eUC7=J)>hC9RJshR!J+U?EI+peBmsPkR1Aq%S6L$viG-^%xP&S&*<d5C)L>XmG&35Q<O%Dh?~IF7ht^$$YBJds)R+h<c@UpB_i4>i3N}9tNe;>smC=?0N|e9lBkB5a0mhpYX;UIDT;I!-i^&!;O%LF6w>v8e3u>%8+^?wQ^FHBlx=5CXTu}8WLO8wg@oAR3d4nF+&K0!cY!D(8f)rVp7Q^n&X6R&Ko%F!D^Ii`!AzCMp}`(icr&h5NFAUJsVykVa(}bbZn#8X3EiO$9Q!Ma1;w<V2XlP({dnN~otXTi+RGOG_(bi2AXAX`qASf^h$S7+Hq>^i8wwa}&(wZ}xWp;Ig!ZATYK+}$VQ6>l{A<Re@B95WQNR&>_C(fHZ?v$6TK=q<YOXsXGl7h5V-G3^jX*V&9_D8d4aJ!-!Df=%pCexNBN^GVY7+C_%i`U!FwJeG+3l3g&a^qQxqvJpd`>bK*-2?zom2BsioJx~Ai2wDb*A8H`yU|aEonDfd8o*f3xNW6Kf$wni6QlZ3$XgJo0t@i+)U%`dznUmUMTaH^F}v_`I_fL-sucg5FM@p&!7D<=~XFjuKB)}?{mPHjD`FyszZ!$z22<2*XGMcsr7WB<YxP<h1vmy@sdHF0}I)a`IT3zuM6;P)v#XznFJO0n5z9CHbdd!?@zHU*oaGL^^7PRBgO<}EfbA$rVA=@{M?h_K~osiRiM|szOClaiP5TxR*1t}N|cveIL^|MeoD-X+TDWhunEz?!@^f#M5eGItCO(*i*HmYA@J|uyaw7FAD57!fizZh#?yV{i2cvqlI<v%<lkRRJ7Izng$k%42@V-aU>*hs+6M}Z_oo>rA59ErZ|S;wSBY*C0UbFj<uZ{Kj;`|>P3obNK`iR3q+`-|7rKy(|F@a&<#>RxJwal0CA85`S}{h4q(AvmFSEtlmt5J=RpK6Vm|5a6CqJfC-3sM**Z&3L=z0Fo%LJru$CnXw-h=t_8z#X?W5z5!GWSo9Do>?#ncA?PSt4{Cu`0@Ub@$KU90@G4RYW9>;?8lsIF1$VB<*dOi+1|R)34tg*^~INjhdLN6*j@{+BFAnmtY%;0^(t$nQ>cFr30Ks$goP_X=0FyPkY)q0haF!<m`Qo{gLRqKx=Rx8EiKb4kiQxkrXzR{h&%S?-ZSEn4nZ7omAoPPld%?MY&}cieWs$d%4xiz@hEH7}Di?s_0r%hLPBbz4{XpPB1e2cLkBE(MeE!?&_YvqDH(R1yke6djaT3`6HBmEK0A?hSGwxWQOvJo~+QQExmAc=>@BcW{ZTuZNu|1maNHZc+%=m*4R_&yart&84)4Q$He)g%kKz><Bz9QA4h*G-}($Fa?W3TJx6Dr=X84RsVFtK$#Sp*wheU|Q;+<nZYC19O}dF?Pl+YS)pj=lZYJq-94p?_4-L6crv_s`cFHSf4uBx|9hsh*#XpfwyzjvBL$1+ZF<>_-Fg(Wd#fIV{PvF3G7o^O<^r7S8^n!P_qn{N8Eo#~QMJKanaO@yTvR>5V%Pu5b?i59bM^8RfLT*iSTNwUmG^V_D5lG+-oT>%+f;h(l`vi^PonR-oYZ8XsA(!T$_6dZp|30kXmMK*3r~25>%H2i-5x6sbI@Biaow7C1E=OicbgZ<#5R%PKO5$=Mn_yy5FU!ZLsE31M_Twi&$dKOrD;oG<<;5K@IrWO7oroOsq@l-WijY&RJpk1wRk8&Db*SxxK^8)uA|UZ#n2RRn=Q>9dJFVS1GGzTpvbA;T+1`OpppTk0*cjMhPQ9i?DD1;bkM_t-fNRq*25||s9Z>XhVqk`l6X0cN*6w=7xW%Sl)9RG4c&t?izZ@t7U~VYiY_CS-Tr@N_8FtG&m51LwG8Sh?=$Pf|uxK9PhId0X)Z8S#6E1FUI68XR^NpSZWI#t2Gt<KE8nytB#&PxV3-z8{f0ssrIHz-Zg>d075H1a7MsdqVgj)K_81i>Q0;xW2G=dImz%q464dO<&DZs(qO430#j&i!u4js^usmxtYjGS!}&{pt=D;x%ch3(Oq;=G6SIZnt&Ec;FzRCzuI*lJZI629=9J0a;vY$E#HJUIUFtICI6l`p>Yd2TuZLz)>>ozdWb|L#mPnudz7#UknrH18$Fo@$Z#wDuI8W+GTa)NSuzx07XHo9j7L&J+77q=paBqsKEo_(8Kny7>$mn>L2~q?J<Dkfn;3sM5{7Fr*pcp0v{Z`{vV-Q0F%VNvb4bSBnIiP(~CcDpw)=ymE|8AouwV@*|;#mRuFL{MjvxtRuzn<zYZ4E%&hwPfJYZ9FzKP>Yd^Hg<G1w>wsaVixhFc(%9*SQ@CFQ@DC@GJHIJ$6c=!(_FUs7^S}Lm27cRf^Z(@m-=G!A!1sp|bcNz{vPLV02j;)R7oZ-IWzZAb;D*Ev9g*xbD*1CygItCETpm4HY>EkyyuUC0L!m9vSCsI+)HPGg4@kTKW8A*daCWjLxIVUXO!>Z2yE^_k-5o1uzyC3G+6Bs1oTLO}N*wo}=|WOY3KDaMW*9dx+*YR6?F>xT$?2d<83G45?UdYKilykecbdG;7?;RO+#+DBQA75<Xh#$20I$fffs}2wESlO%(<Qd;SKH5kkd}<W=9U92kbCGQ{yX;X4tPo5*7^I5k%+&f2n(<T;d8O5X(ipR;R@zYzsy5133&W8vpPHM{S_LyG)iYn;3<#@n90vkG$c(zwFstk6hOS<>lzuqU)TN7s}^(tjs^&GjDOf1Ga+_UuQs4>i*k*n5l=O4sRw1Z80gV!<EL-tq5PW&0xv3CF-OMb>Czwe;&ly>F9sqG9mDrF5ZvU+O@ts6jd?9^wMb8l(&eAKET78EuL|i!F(E?!JzohWQ$`1Wjl*E-R^Ch1Af2<nK=MV@)vXd!A*2vuy5J>|K)}TEJEldhBB{@=L4_#_6<hcUpS8wo2wcAb1u0$S@aj0!P*ubf@qlDbLbxY{RpA|zMNUlCMCy9aTUk&K=$x>#%Sh1aJ1dDTxmtOcmOtf#RO*MT3XM2*@$Y?mQB#mNc88#A*QC1f9`DHTNx*bwfb9cjoKpvzJ=WoBybGYZpAGKIcj4CQgSZ~m?6+=Rwg5P$S95B&AQS`}>3GR@y&gFx9PT`$aHvc1P6CV~M8e8MC6>jLo+Ki2TP-dn53??=S~1G9H{hvDuxW+lpH#dG+}&N|C`$YIac0TT)G(DIqJ@ljO%e{8R>WGe_NdE*q#9P0-E+i}db%O)+Y!NAR64G27eYX0UC68cC6;qN5=qSBXmQZQwhyZ<S=CG^-`mIk_Fm$Dw{g6F`hBTEgcgcr(mclt@l?nEYHasGQ_|Qp&x`V_X@)%8s=qgG0=DGxv%A<Tg@_5^LMMs{kX^sskl;C^ZuPY#HKMXfDw?tr!MHSLu2>-A)wwiOv)HKM+?74ql2E$>bL$CE`7r&TP+{fxb#AF}alG??oW7+Sg@b&ui901He)a0%-tk>{JzyXpa5RzZSS+~uT&>G08mV~8oHv!%;B~ZkF)D;sU%Vi?b!D}~>g6n_XP-dkfQ^Q_@%fV2sVh-i^$u7}eMfzJV6*rLuWZn~BQAX<nSbti4oDRm9BB9${AbZSQE$M+MV1P8%#5pD{Fe+tVgO;Ls8an~0Mk}lpzr!j0WKzKp~nNwS!jiU4Knz)-s83flYT}?eXims%CinHUF<1HW=5K@80&h0h@68tlOvcO5vx#~+FzD_!G=&a<XVExEMOM>!+|E|)&7t1YN{lk!r4-`wd$Fh8a655Mt5*b{!@QL&#Q~+M_cCc<Mb2)P8Cz3E>WugC7gwr-04sG`I`Zno=P$-xpW`c!FM?{)y0g=1#rt$)}@BX22u7j0OKMZcWixgj@GRrTM4AXJH-(-2-fceJLZg>4wEhsOMR{}S|K+;4x*=UfqOK<C}JpIWnHdPW7aWMAqsYtTs~S1<2d<KLtA@lKZSO<CT!&X9EiObHIYk(fa_QwlmG07k+zC89>nNr<_FB}xX3z4=2Rrax+VfjB2q)m+YB~>eCt_5_frVwT8|YqX`+qd|H(e94#BGYD|iV)KSqo`*qfgZQvahlF9-WfbSpoHc4bbtma-d&N){u5d{kzR^i)#k-Ct1P<e7$Q;d2{p=Wo6l`ob-1nrkqzs6Zph@(MW)o!(3&(=lvLSu-}NF}9G}K&WLzt7zjyw7@Ekks`*lC?~^iY&i@v!G2ljN%lx0lmP1oON)koY+P*AP0|{{=caBpHDKxt4tCDB4526Hf1SV#9=#cx`t%H!eRsroV5XYdH4mCHS7zxU;=kbP11yv6VGSX28TC<;`}E0)D!cHQ(T1lO!)Z`{&;djm=s)69gt_&og$o}$N%)&VnEigmnWg6dj(9KZE4i)TcrGL%wVpd}2Ye01=$hK`P}~SclGHDHTnfkN*kIS|D$b8n{>aJPR?CNr*2gcCs*%uoQI})OAB|u{$v(=J7vDMn*^Y1loWk6jg{(d(LqiG57|+sU>l?JYRO6F}jyBz&VrEZz0R-<p!Z-mP9b-Z9Fi!b}Cqy}5iSvNj@NDS<mTFp)MH30dvBCkVqFPL|UV+W$KtOb>&@g3dAv*^qj393GrpMKqCVVgDkTF)D!UPa#5w-*=W+24IKx5w2bv_ug@xUDD^*)=yx0oR3xDG#IXOYfGtK@U2NxrT4U&jqs&;|}@F}&1C&Q`!|y|Qt}%e9hrHudFO5AQ^0aU$MKsKeP=nL8vV-EK6ancpDQ3gc1>1IT4KHXtbdlJX-A{ye>ET@-J1LQbp?{T1a!b0|6?bvi8~0yuEn!Mj>6iTME4A_H+GHE%#<Os()r26m1>x{vTcm;~V80?atTKl6#lDz?zN?86G&nTI*Ax&Z>>=GW;cmT8f-0o}de&Gn8#+t#^x4j6=$D2oYmYJ-;uxTSW@p%mexQ6WSw@W~Y<vhPM1lrV17AX8f&g~~4w=gAGd!A7b>yT7Pa0fqvJ4TvRceWm5zS|~<k0VCx|f!UaK?xGM{GrwSN>k*0~x%WaOXG+j=WOKzx8DmPQ-AG?5Qs&g5;}!8WT3etAi@5r2syU$?F*_qm=sAKg@BNyfWgU%;447fSp!S4E5rX=*m)5nZWH_-rNb{O(zm+g^ld}ZviPK|Y?A5ar?TXeN!%7OlT_5f8OFu$1`U0ZC2L}?wv+TTKgzGmx5IR^gJ6WukG`zrGFTSXJcjo7GC~+f9Orj60&yT|KyEt2<$hi5scps}V`sRHlU|`R|Kk#daSj-{mPT(#~AU4PMcC?0QIJnw{YLW+lrrwVB@eNU??yK|4r0YkQfnpS(#?gWfpd)1(g>x|AboRNpp$61DLfU(MMEeJY#=1`AB}%8ep9U~yL+?3jv%tN4jk4{D#xS6R%jau=TZzmLcD|}w-2oa{MG+S@KzRgMQ0(?aO?W8mmw%&mjrjj2FiKonP+BPvko48_Z$J&C_Q75oidq2*q6jhBIt@2K3MW~wsoY~(!3rsH6)tg>0b$h?Ovt?efRXcE-cDUr$BP#w*HxLXsyZzO#K8?1k7cndyDVjOCrx$^fJLg)>xPJf>yOH7GKrSA#b6@m_Pg`aV=@eYcT7PSs`a0qgdw=kQiM*Zr;m5Dnzo#d8sX)5c;6m1@0=9;O0DN?GY4GRP@YGDi&@!)4Pr=D_=XpDJ5-jd*=k93v={9zYeD#?5onN<&~T7?>#nzP?XPl|rh97G3vVO06cGst3ZQ$C6F1<W6=@_o#YWswpbEiS1SI8=Z~ev8+j&C3^LN8#dvvzlh*u``KrM<xuGngKygB+k3v=t3by7Ed<mtvm&vQ<E0oo8@cmND9`p-U)5_#WkI@#}S)7X(w_}sML@pZY@1d}DF%mqkNcRkbdu$<hBsJ5b!MPE~)Zm%H&aFH4RdsdP6j0gI%0>p~fg&eJ2#c=t+8XQ#$iDD^;MQQK=p7&1d7GVD2&TnVlI0s$My(0*N51u(0C*uo^LFxE+{^Yx&paNl2QXZ=4KS9lCwm_9oGG*<L_WK)Sr*}wL4<|eCG|bw9yl$LuAf|l>h6v7pNjd|MIhv^2M@GN*Q1P-bUaV$hi0+i6zfaTIa2Ez&A`rHx#*0U9dmMtNPq>QJDtQr4O==hyVgkNCUB;Q2-``-!hCRBRl0#HKD;FRa?%nY7zbEyyvTy_VoupVF0IiQ-@KV+Hpb8=!**@fcH-XXh4HfT3s;SdZr{Pgwvet&tC!?GX8EE*^UvJ&@p!PcoOx0>o<wB9J4<bC;{Pa|cU%AjgA10eZ(7%c&II;W-Cy~d^&Y*Nu-Fa5sf4grTJ%jjy=utS^>5LYL^vcvaWZ^OLW|b}xEbqRTcRqHtj8#-uL|`sIlDf=7RQX%P3P47`ru4OQ_-pJj<M2w`y@p1$+IXbw=Aw)<?hp`}7E5j5`Q;|McR!6LVJU-I>x3YAvhuCO3W3#`-UCuNbWpwJ_QW(yHtfbh3Y?ROo^*=$$uSjuf^}KN!9%TaZdh7Da}JrI@a5ZragRyO!#Li^mnQ9i86~zcr262a0uDoIzlvn#7=6#540o?Rq2BWBrR2=&ZU(>3LQ1z~(S84wEfBapJa1#Av&ewAuAaS*37E4`^;JefZVYq$!tTo@aXiiyJn72Cq53YU2CqXarstpM&PQF{iHB?whgLmZBEEJ?gm5u<oN06lgP9KiGbr}+X!_1O0ur1Q5^#q2sx_|ipAVDx_Yki9>_Oj~H$2FrusZZeHxlg~3`aq@8nke<OX#{#fEp99(4@D@s4ra379fUUA2+iykFC4>STdb-TZIY|p)y`LOYvnQ>xSL=OWaoo0k3N^Xj`G&&qKoijN-O~|Ns1%5fzO<ZGAZQ4I9%LGnVaNw0Yzk^czeGI$PfllCXQlPblo&kgdZ(Q$#0(+U|9C{C%sBoTEJ}*Cq=Wr3~XE;808pn;}N%II%9NtdwFkCQIJ)d&?Y*@K5AQ8Dg1>^jo`%l#v069-N>~JJ$QRF(i$)xIVK8{_U20=9iidv3V_ZDSE8Y!;oD>%*le=s=`FuXj36LM=WYUMZz)q4S+kvDZ*e^oa}naW=4{)z)tNj2Qx(AoTB~L&?On05I*Vvtj=BMofi*7d<bJnh4#<tvyp`G-gXWR#*f_%w#r!<fT{@VtjlRNN5~eq@8ubt2r+q0>}GLm+$mp}cI8jKu1N1pXlKC~^ZVur&__z4Py5wxm?ANbdo)84YUlC~5AgGgP{^MeowU?_hw1K>;&cx9$>UvaB0u=Ffwci}ZOdMo39oe$s9qVb{jmJxNgutbBDTYSs_ofqz670l8!+)t`yb+VdkNML{3HUI_A%Qx=E6VRw6XAc#0&d<1?TW8#z!^YXJ#PJcrsRCFyJQ$#Yh_Ee&XzDKG@QvPm_WOBaxaV3fbJhE2#bzeGn*-(GkwD5-a6i<WpH1j4zc9u`rLd#^PRhzX?fqi~Tvpp`m$YOAdY}VZWaNo#ou<-OX@-iHG(=z^KMt>PO~XQ8-{WQI?kes@)9VS#^T7L``xR9UrUO4QYCx)5x+da1L#_HQhzo`-9pXytsQ)7Q$(rG?FAU%Gh{t4W5%*odyzg*>6hykHEs8dP>^xF6;X68XOKb-Yp>m(C;=D$-fgDCM%9{J>wWn@H(~cA&a?IO6>Bw{9%D#%2$S|-35&zKWkNG6LK$jFh;(havXARsoB~)EziGa7g~yKH3~0Twr*JeG!w(R9CXb#Y^VnVrE`80Wds%-8$g&P!AGV*6J{N4AFheXNF6>bU=N>Bawn39BdE!2ZI1Jr03Mx2zS;laVNKE;HClt3Fu+XdCWFM@XOfB^(^ck|yMRIk79y|l2xDpcBaX)?W!SppnF!0NB1$_oGa&w@qC=`8-#ocB{-Nqd6aLfiD-9;BrUB;FT&(wWm3EfabbWHk>jtan>gfM_AfgA1LI%xRgvcdp3j9!H9-G69Zf}wMR&!!R5?#RTpC#o-Bs4Rj&DjDY*(zFE2PyYfF%*QHCx;HJE{!swem4;+g@AWoTeY(T#J|NlS^u0oKeoh@d|;TyON_F!knLN5XFZ$&&_#}`FJ<V#6G~PeF7MFl;$*7Q*N};HzA2I^5#_<O%vRD~Y(d)LEI#3Jg^fseMjJxMc6nd)ZI6GGJKCEcX#!mxvY<gHbD~G^M|DB~L4($e+~^s!rn{ZkP$dV=WLxOSs4?`CSiS8LwH2phcx(yuhffUs%jA7gpsyR^%b4^>XiusJwRtVSq);rUZ#?>N0JKoFOSa`=5Mh)SKqC0hcURU!nuLF=I0h@6O~mAmC^kB)Yn93y9z^2^F^W9fm8I5EncenlxDScPF}f4`Q2=r+UY#=R&hKI%Cq?`wQ3yAk$(}Kx`48T3Lj}!Q8If_$`0p5~h}=5N>n}2##m67B{8_46?BXlm5n2P;C@0bYOVVv)*yvi{2E+%BsLm{QHK6IHx|6K+#zzmAHwXz#2*w1>k~ny4(C^YlA#LiCVc)4ZyV|m0$5hypa!Y9Y^94B&{+sWGs^ePyD;2am-W8S2#2R$VQNB455r9`3=y_SiLzSh0=p<#NV0Ru9#gGcHer5Bia^_XJgP2th&4B9EOv=`_$R4JN7yV9(><moZfH-wfe9Dd8en2=6B)X781ln$D0iNHXM9z!_vCUji>?N(_T{WqM3!MJ_|7PsM-7)3(dk;v5YjPcoUj5>S&<23KKx(l+iT~Lms-Deob53O<n~?kpEP~Mx6^|^ZhwyK8z{mUIJYZb%*2S3|1db$`{5^ICVBxc6cyBi}TJa*7ux??tY8^by52+C>l1eGp%UoL%ECB2(t5OY9i+qJ7XPvceM4-Qnmi7Jm^s^I*vFG=ZgAWwDE6jOm1i~YmLC=RW7jDplj6{)gUr4Uou2YPOKzKIG(Cg)pqnsg8TVPt+!**AUnOlZ!{#vnk*S=Zeuyd)gDoYNcp+E*B)g$;VF3y@5dX|gP!dxVWp0P5_IM|BoMYuG<9gW~|cLlp=uyX@y;?dJdP99&G+)6dGDAM2W;#6Y#-2u+S@gc1CI2SCfvfQV*{+|Z)zhu0cfb<~BG30C|-voe8nif@_PeSc0_Hq48gI;}-vU93u74v_f9!nZ-GjjH1cdo;1h%WOZsQ+^SGwM5{UZGF6Q6gTMXR9;e^vS~=htictoJ2gi6WZ}!eXh7Zbyy-fd?j(uYM6=|ij^W>JVZkxH_NmhR<eWF!e+c*HZzs-i{Wu{$q(TOR*By`kJn5yBEML4-nX-qwhR-x17g{|EGv!fXn#MO|BC<NW3}E=fK9e)KVhz5(9oRd7*=Uu(!`QizH&=mr0LizQaLkPpC7vTX|@+ZLS0BWI8ZX5r}G(6IUs29D#_QcNhTp{*Zm$sbKn9t;0g8eNO~(bC-mVD#K(WQAFTo{$&G^Qo^BOB6T-nlk{GLm-AJCwS8@H~c+I1OY!B8l6ZY*OoFwc?=ghK+JHeI*!=-jr7MU7r_tQVImRGKyw_<k&kAV-I;qrd)ttc!@sf1Vi7sK7~2UFDvkD-#9PSM2jD7atPrR(pEi&OAIHL~k**>u4VHnU;EC8NR0c)M^|#5*9FkS&9t1YziFfU%ReUC!AHZpoTgdr~Vep`2%c>Ch2n`hl#{nOXt#5UuwGzk6uucw4X&yU3-AjL=E3fHjrva^nSDx>27oB@zawd#eh~59^*d4$X(4exLa0O{5ecPSb%%{TdAo<Gvs(b<kOdbfMZPyMPKK%xG1#6Ed1h5$b3MMhu|(RfNpg`3ls+Z2J$2@G&_%_(a~<ZBMHOV*bou@2uH!EG+(Fxzso_dqy>hxZZw0Pmu@pbFyVXmN2%COdqB9&1DkWMEAr?_mw@z2!}gBOIEE?<$QdW>>&hw6QnZeDh$y=Slm(bm{qaM`X<wZ1Ic%14W7Gz6iE>0w*nv&cyPwlvK5+i>+F0F-p-vzcMSO1tyo8MnS)-$dVZH<{kxusc{JruQp~ol0zEoS5dYyMb}PTJ6Hq&W$%YW}3x@D4Dz3Ih&*Yy{1DzgGtf_e$ooaplN~EPx<>Q5)tH7c)UJ@+{QD#(qQwsJTtQWA6ZyjZ6lndcdBWH4id);Qz&ivTdgi{`zX(H{Ynrj|-Wj8<h-G=hxjWb}4Zoy9`Dj)?@=X15ZD~3(6eF>cv1%DIx1M#}iLPl&z-2J`|?&a3}j8qKotoHf)t2u6&^7X|o1X%=?y4$eA2iZzH!8`H|AM@hKjabV@B{PNri+eqoCbTY_IF{0T5_d|;@kA$5eI7{q)}Ea&TwD+I0eLCel9=6lgDlcl$~ba4s_u1oWJKs#oYd6+F8=)_{zeC3LDzB@?osfDe_YV{0_>&S3SpZtzkK4z9Y0pJ>B!ilu-<b;XB?}&tb`7`{0Q031YbvH;?A=5xH7b^@7XW=u31dDlLgI7#%Ba3wPsORy+S>}KZ7Foh=a{n7DN(m1VOvl9CF+I8nL@jzSR7rF8)<HVw+DpvxSh-?q-!tI?L~##V}7xH|d$iBN&`w?-b<a+_g=87-1*%nxp!0hhZb+8v#uEe9<lnP|EvRyH_EBe!q)<>rttwq6wFXv61t9n%~NWF%~alPZO^EOg~VKaJJ<6gh;TtPd%1jH5Fa_IbQkWt@~CUGRjxG7w%nj%S|s7(BOErdsd3_P|Q5<*YX(8b#BKHo8tstW8;~F!90U2B|H8Q;c4E(?4Y};lrPD`vbG8){jr>eHg)YAg8%PF1Rv6reo8B$U*ODm)&sR&jPs#&POs9|WSD56$>ev(ADyMgOT(<S(E^U}JoqYNw3_oKpNQfINTKKw=#?=MPvjths(j-FaG_>eQgCPTQgX>ckSQ{Xj5-Jr>+$(<a#k6}2sda2n^Sk9(Q{_y_jJF|SQ>1tSR8uxt&}6xW{|x{h1t#LR0i#{Xbw)tHf(pn51{(=EH{YT@-UOA0-tT+Ry_J;>qefh|4&JKR7ySRu6roi`%jLYVaY+6vJK^n21iN=?&vPYs=Ovl$SwoEc0Zcg8-h~?_XEYtu35Gcxitv{)&@Mt1!03NqYR@w7U<JMCPu;3<?kCvJ^rul9Um_wvzyT4A{~{ux5uouO}(&Y@bS{aZftmx#YFBIHv-Iv2{aN|7;?VUK5-y9JgOTi82@|4{9J_bggo**hYZ2dgAfHXd(Zk_=-f8Hoa@W#P8621(?+-S@#%`n3AICTw}N=UERZnXj$(TOOJOgVtK51B@1NScR7)l4;n0#bjH|k0tX`45E0Pk5IoS`B4Rru#+n@DmIgv60(R#DM^8Fl@2}_s*!Z!p8ImCf3$1InfaYlT}Z6IwCF`hS#^^GAstY}haXpx%vI=FIx{-{X~d8vyGGk#YcfVSW8zF=_vPQObCp!^#}ST2>h1verJ@Q8smwb}!mgw$zyxm{=(`gNYAgWo@<`}8XY-1I=zIY$59i`<da3@9>O-MZuyL+VIw=I_r~C-c#!&%d<kAOkc228k|YdB%b+bHwtF?SWcJOl%Lp6r*5Y>ktb%_x($=e*J-+nFRT!nT7Xzfg32hQE8yZi1cDLzJ%<z?1_mzD>d^R3{&4JXWb(ZXh3Ro9GC{^nFk?D#g8nMC-Qxb%6krc3YAm?Dd7x`UpI944$b8;&WKx`C`GlZ@hFfn#UGaAj2k8ZjKld{>O0Zs%<H}fwR#rKOfNSCqN-#SAxP+wWumBL>+Hx<`KDfbF0RtZ*qfd5h#21~8t7Hp>tIVU@_2Sfz4@%nS3X9^=K-NLYSbr57S8rFZKa~Zy;f`s?O<o6?>jXQrt&22nvhKbTN;sOk56cQN1i;(zEyO}=O0WVHA$jB70qx_Gx-?-ZRT#zlDg#W&PchunkZjnG&m}p0mdV*(sVqnBquK-jmAQP78`%B^=KaBshFQ-7(L8MxjI1$wO<@56P3)1*R@QC$_A9u9T<j<$u@uDV!}ziukf1ZFaivtDN_E*l@m2O9_WF~4EU8*pV^MHp}ymuX!GA_*ID5B%k_vlN1KPY;+mLmR;=f>i=80t>97k{LL;b21!y68VVruI$=@!+)>qUrC{9p5l_jDDAbQ%{4p^4x7q3gKm9b)KW=S`epZ<NSH}uL^_&s&XBy~Q(1jWR~llr-6f{2We^cJ02xp=t8jiwaH&@}?7d_cS8k}&T>I~jZq@=85n&GAHoHPRB{Y>D#{DEzBK7B<%gk_PQSxHeS+u3L&fywqr`K_IH7(FB{0z5R2e@IZfE6god~4L2;f&8AaH-}QQ8;7T-3nAe)H3f5zhf3URF48fdmL78>Ju&FRF8M5lKX*glBMnTDP{Bx$z-|+vQ`u}!<qNO%GB#3q7S77BNZ1pkgXexK24lnUw$=Th^WfIz0cG>@8y~i+w4-j;$sd)wXmlTGMkJ3xawZf%0y@syf$HPtR!bG=L*DOGjYX~lW%`N;PmoDyWHESm8a`4vyD6D5c^i^IJh4FSr_r&M+F;A|V4f~BYoWO}W_6`=HHu90<qf)W^d{1ETwmY7Oe^9vQIHasPHO1+XIVIxxGg);_l+UXQjrh=oZ7Z_eb(OFT|5ICIl%GY5Bv$uMT(zI`ppj64u?hUPW3`-l=*T&y8?jk-fK9sE8%vko>zNDNWs7yNqu|K6<99DG9x+Y9y7H}CxX%xWhM%3zqTS75{Q#|jb6k?PY7;W&VZ~;j<oKY}!nJTYP|0ix6_b_vPCWwr^mLAVa9+#5+c!8bgtQyk=p|DQJ3Em5pu#|g4|gd~CTN2r)MvQ}x&CKU306ia%ee%YK=dbix2S{<{jJ{Qu!kV5`^nahdh%$Su{Pt$#ef=FY*E+=$~a=^7~>FurAbf~^)6DM3e9K^IDOTnHDmqt+SHF}u8qsntC9Y_D9ngDm_o_|5(8OVExDM8#j`Dnoh%(J8-d{Ew-19j*52b^cPfLrOxfUnriIoJ?5|tCi{viKmk?0KP2_P;VV&kd47e1js{dFpDKKDYFF<R;(M26C+wu*l#p$wv36nV!#1@=O)Z;%cJld_R<)$)t`og;aE_sD5U&T0gO^rF=^8e2u#3Z^K4n3i=AvG<h4DT_<*66()t$^UPYxI90B3&;|&;O6{zyI|BUk^Gg5jia7$~htBemB@&00|1M=mA((kx(s}@Nzw(=$1|{AQ=NPTj7>jLHCsO9T%Q9`tqCNVCNe#O9f;_WC8DlQaj)$><T>89BpY3^{w78x&13ETv_*FS{z>Mg;s-DaK6@-PYI1b)?NFstC4D*&)ly+nl}N4WU#XP(;6t}PHoI)>@p!ze!2_U<uEzwiX8(4ubve$W|_>=I-0db)Df6ycY{&1Hga2`X&&^l*Y*hW8o1q$0U-|BecZxIPXEPC40+aaUirY@;?{YWYQtjY;r%<8E_cnQ=LDodOPs~OM>Pqk%ORCo`yNconM`NS#=liB*IMrmKzni_BM1E&o8oMna*0{HSi~Z-8AucX1%>^HYdWfu(I}KzXawKO(8rZS@KRbZ#rAr<u+-8Frt-m^>8Oje$Vw@#7`ZEwFp$|o*qgffGjwfsEXCj&kkw=yv<Wifxi<g-?~2}NrJ6%A*T@n_3{?aG9Ey6!n8`hN=<9S2^f}+04~>e&cn)#Ph{rfWZxiWU?^WARI%&vH{y#KCs~q`&(j6A1(&cK$+AC~BnXju1u|y637Bfgz(!l)XV`XJoGRm)fpP6Ff&4Sknh(FFU%y0Y(A_Z<hy9V!SX3JhE35VjWu6NZrEfsnc8uS4r9Nr81I3CQ2ziVHHPz$*7AQHW!Dx-&=^-EHm+8wktyF6rEbZ|RG1VWFqt9gJWEN5?=J3(l^=7>jy9^8YmYM*0Fd}J>L)M1!4?;|PyG$RyP38eOF*Je5%%hOCyg@_+*M>_0~WhQIS8XN=iC(eSbmz+4^0gI~?=aQ>hSTlIjUlKs98ZOyJeg}Kf2Eqtm3g&px(ND~{6@E9^v|_5QYH_J^o=$Aidtcnzsi6usOV{-DssBU>hLDo5=YR8cz;d#K&vK?Qq!o({4p%D`9t%gWJUaTg*`~B|)v|g8x(kTMYNtsC3O{}w;O7)(0@DZXFOMo8mCz${+~r_l)I2QC2;h(TT}M|xwzvtDwcDIc_r_d||GCML{6AB9#&r^6x@|+wX~1}p^sU)A;rh<JQ6E?*+d`UK>oj6IsFT1@0NX*R69x%qO}>f&#D|0nSNd|cqOsoXM1K<tiD$i-{eW2Q&y|XkS?Z;*qM?P$wY#AO@L%F`Yry+g1KGkCosR0?d%eSuvvv+YrKi1Wjq=$%tLXjc@6C%6v*9;MoM!OD#5y-sN>O7Xr9cfhr;<|O`(A;Kn0+f!ISIZe*Xx(vk5xNK9v+@n31ylncBiMWcxVT8)^muA;uIC)*eY@N+1z^Kcp&r~L8~g8NkWup0%SWYw(05G1L}AC$XQGzM7nB36wnK5>BBypA{lT?Q?j#dbz`GlOsk7b8ga0m`Z~~gIUBf&cLTnM)Tev|6sR`8!&)tEss6TZBoM2g@;})xrW+bh;(M8gHWi!%f}aIqvS~umrb7<ENEWJp8|9bpLv<3He(2AZP~<<H-!||qGDkWlelW<pP(y{SDHKjZGCW0}as~$DYKdjhwLn1}PnUW?xm*i{_Idl!=scOf%+An+31?>%>8ufWOQ8JC77z;o=oipd0Oz@=BCfP9X}sf1V|cplOxescBcAWjasufUA_p+&4U8-OuXzd$FmnLs6j?E1k`C#82P0S*!>+HxlaWOO{d}pw^y%p5)rWrfLeGmfyGGyJe}KUm3AwEy3sa1jagO3Bi_FxvnH=(hoHauon0c8pg4d$~R&Oz|Mh7V9oLhI;v!GR6sMCN*lvZeGUo>dlL!9!>dJ}kH(ryNi@cPG<tg$ty8vTbXLvo*T4KhU5Dn_g?WdICsDnTizsjcICBu=cvq_})rimt$=wFFL;by%d?W+lA25hokGm2B{a>1mY*{m^(sDVbe0ecEV0gI!p1j|(Jyd9+i!9w;1-yaj?Hm}IR%^{J?Zy*KFB?O1JuxzlN_HSn5xD;lkhh7b$+1*SOdV&B^szQnA-mu=d54h&Onkng&LRr!Cr>x<{+$ueTc)_BG4Q-sx!HgyRg=70c(tiE<q=yZYlOR}{HkPYe$zJvTnq?2gl!E0cCJMCwsRJ5x#ET_wn7;pNJYs}O#0GE-%Kz}tNVD;bb*Ait}1s*>yN7ltb4-5vfNvI}NifI`q?{i6W;+!*gV@@6hcbpph&$++8LP@G?zb=hljo|J1R5y4<W{(N)unFn3ek$RpRvvCf#>A5h17epJx3b(tE+~M+q<-1gcl_Aj!Va4Xafb_Pb0Mn+1Dq|z1Y5u`?-E8C0>E}j81NiI7!hFVsg_`7uj8GTD>r^EY-9$rEH{vF`t5Trzj>~?kb#HVbf7qK-7ifV{Gu8d9=$07-<v;4O5%7~hT_rXfGw8Mbk76l$`O<YWMrS0>q=JY!9XlX#hE5Nn6kU-_>mX~uA1R>G8ai(GS`e~4MGf(8cdTaMEG%w3Gd<v6o=^>$rR0|GQ;*$AAr`%8XR%?9zZRd|5Lee7jxQKzf|SkBTt6NKy4jFLGwp)&M-e-v{F;?5mhL$QgR_6gLF+vytc8MI#y&v&7C~?!~}{huWQm1zq&z2(wFRJuc-lNWKZHu?SMU0{?daC*DrNGWr>=R@x~A*=i;va3P94d(t^%l1>t}oBVaL$ldZ2u+4x=oEWX|uRy&Fzey>mquk3@IfHnsl7O(KIUiQUayR@IaavY<&%z?0#1d$1jFQd+gsMYoc8&4MP;A=!d(6?H>k6gTjSq!k>J?^W{i^tODl4!w`KllQ|=AD<xpP6wHo^(B5Boo1}C-{jX@y@BCRU`-bTvO3#?kToT5YD`_3D0ovaH|L}%$5ZTzPU~Z!;Ue?TnHCuN~1QCoEeeRZyf~|Lzh-loI^YnLUciMWb_;T&mbQ6KoRK7uHXJ6+5UoHMqzzc^95P6`a1;4{NFZe%2w!Sc1(5Xi~|ml(9G~FDV;wA<<e2&0;bMdDq5~Vv3^L(Q+Ol&7rUQadoU%I_Ow%c$J<g0PV>frT>rX%ZI~&p-Sa3Z?HQHwLSNuJUIS~ibSGQ$(4wwx>UUo2P!|p`aukHx<t?2|!wktCo9h(}@;NhtQHs1d5^vUAnlSf1-%DzKE(6TD1k>ll>i2$t9-W;3Ev(>6a-vYIQCb*~2=*kY_pH})=c~ih%J8@aM{(#wqldDu--UJJ*_~P@aR<Uz3hmA3i7mW;3K_?ltCqA4X$b-y(F_grVIvprQAXs^FvP?(r=;ECqrk<MaZ0qo{5C#ARf!G(jlKh0NJL(_9Nwx^V4CKxfutRASb1O0>P<EKGnx$k_Nay<9m>+`zBF8S=9IDXr{Q}`6R!J1!amQ}DXG6T-Juy~;3224hEE+!^1TijYEa{H1yyFg4826}B%-X}3#T;>Mwhn^(xV!%ON#T%Z*viyE+!Nj(O5(G!F!e&5_Fd}{6EnX*K3<~u5!EYknK=`m2MebnsFp+1Ttk*(Kp9lT*ATTI(;=Ng-Ti*2u-o8wwiI^iH2H6%lC)}NlnYkgBQK)&IG<I1zh*u{cG3f|ETIQLTKh?4WR)i8ROi0MW27>-MCpagaB5E5gI+FlCD}q-de^CiV7ckB^zkBdN`d}Gfb^4a_yFbUr^+drJ8J$Mv7CFfpK#hLBI!Jg=MiTsN-K*RGr77E^qXqg??e&oe5ml{A$hqJ7ls|Tg`V+Q&B(tl=Iq|IKaeEJ=s{DkIMrZRA6OH`H`liYSz6qBb{&B_kre511#RWa{!_NVG;V@%S8NN^VsFwh;r#^HcUNo60f-mNNjT)rz4;rAs(sghYtl6YJGz;dXn7*SMpd<$<0TBzyz`nb8z?vXSIc6Wfd>8-gdUGOj+%Llx%~A33L}@h#C#pwo5HyhxNz9Ld2{;gQ(&AV}BQRWv>JoIS}0`{F8-|3m0m4@FOBlo^It3?h(^bZ0>)K8Qx9yxx*;!S#D|8BmHPSp042jllD`@s-{$_(fW%Tj@N0IIZ&w}p9A&bOwLW#pd3lmNC2e6`9J1D@w6oHx>fGX62g}<h<zxbMDwUMU6Z;Yp?rG{DP6?Su&qvw#hjH@U}I#~aF;ruIJLEa8)8&Pbis}~(>1hF&W11dp#)5#qT1dJcyh@4(rxYOrGQ+nc52a!_gkjj+2ENl*AyQV(LhTL2$&`Rl!M#oilscqP`@UMxYR)1WShhb_J+_xnv)<;?m8}Y$U$<-3ZL6#l_khdhY6wyy;D{19=|KfkvhjpFFP#o$o}WMOqt=~Vq7@@<7z==H=?b8pBA2n+J-Oz$<|rC^BC$^DuwRjo8Emkg-p9S*?O(oaE!+f@C)HnnQ-;E%elMg*~ZeB7G4oIcx#F)N_F#CUlgZUv7Ih#iogNJxks=pYYgM(c1%}ID!}kYn^$BRcx^3A@A=Hbz<iqyQ@Jy}>+2*|BIZ}HK+6t6>HSUESzz~HS=cwi;W0#kUF);%6c7vpmv9j|duMzfL~XjYDBWB;FN}>3cxAdTcA$ZTqob-oXT$p~+`D(~;9UWSiV0K{;{iy4e&!ci^ck#I30CE_EF7`dHmUf^9L|n5=39ap6K^C5Or8?1HI!49U8DW#jW?VK3_VFv0ido5klKh$CU-_&Su=8+zdmKt(VjPqc$Tz0c$wVd+Lc<-hOh~N%Id16&X6u@-5C|U4@^XqJVm6Zu%!D&iixc`k#5XeU-f5KcL54V$gcxe)&B8By1$;`s6-sA^x3vVsQ$MyZwM=0NKk56)$D;K<cY9Fo;plZ+A&Y;flrruD*Y6^-of#i<|qV5#>V!Z?9A>(-03dCx!YG~znnVVXq<taqHVF*x~NtLM!8668HdU>+-w3JRz056F>)*l0D0?n#kVZ91=fgwr2wbEC}8pc5du6ta&aPMbS-T;5>X4HN!RHZ=ZGlic2q%10k8p7UtjaVsh2{DUGmm_`Q&csNms1whDB*co1_if<Rc$ua!w2Zy!WYv^p*>njntbXnxe6sqzQoST{qd@A(kZH^?6^J6v-+(I=AsbQ^jD?WwlL7Hpz@@51A;*0TD=Wh+)X$*~I_J%1{~Q|EwEhp|<aliHaPV%D%wpb~+k;H%<CL@f4QpIH>sXtqvW5f(Q*@qpB)s-^Sxeg}OVjhKTj<gFW3a+$+em(BKL=S8-f|Cj=qDe`DLxrgttGc^)Y)2n^GQDr1R$b-X^{Wgm*W_*!Q2->#)fGp^S#Q8#?(%aqu=AtC;DkVC@^5+T)P<CBhEZavbwIk5}$E%!pnSTD4JjSXkWp9O?nQt)NOk+z@N*;JSDm4Wog4?KHPcNw-jn*u}48GUDBm8l;N_9`hEsTJ6=nkQ_u1JZi+>B{_;Jz<(#f|cK_>!BcXr%1*iF`ME^EHdU7>w1#&e!{%$e}{+8Jdxu+@v42A!gt*-evmA7*2=YQ?w#f{OsRgF*V~#CqRfcFNalFyAw6DtZ<mS3Zajz0vFiR!o4I`C2mp>|xPOs&+_0EQ!DyGr9N%#Aos9<W*d&yo$xV$vwwc_FkNZoopqr}Hs5x~o9?H4ZI0Jp&#$@XO!fnXEQ2n?^2-(;uj|a797?uGzP10axQ|$ys+CG{&Yj7sz6o4<sTEcN4hSpWvcS`T=b#}Tda6SbzLaCWcJTH>@r;;xJFACXi<pz;Mf{T}{%nK|Q!BF`+3&`d{3Ar)&vWKHY@o-N>_)?#58@KmGKvq%x;a1CUju7y?W!kC@(rE@BP>yHAw8YO?eWEwz(B_omx(_4r@29rY{Paal3g%FwQHjNk92^CQ)uHEvOu`wwC?t3KbUl=86-}+KFCUuiu2OfUKp<qKd;VOHube}=O)Fw4-5ymiIDjBrpFNU&OD-KLRRaWEELxW(VQ;$eRB3!lvN>`b7R<_!{A!rQ%TeT<YLB|-PtBmw`HGMWs31(yvV9b6d^lQa`8E*&5OzX#%cZ)=-kTil_&T9i*RvR6q%VA4>}U-P<k5zt7kC%D7jUPRCV7MAmv>BcAz)By;LUxesB%<?$^Oz!fnwkd3kIJzk?P+tzfrI*Axme%DEIsI!5a->u~N|oBRbw>NpZY8|HIXLIB&0@BtEv0W0B_XS&Pl|2cpWncx8Z2w%8N2n>DovG$9WY@$0ytvsGn=n9WdqVX{@>G9x`V-}39D6AwApu|H7?9q7YPtT~4+27_!e(x*D7f=Q=-7NPcI;S7t*arLNS%aci*22SCP8){md+drX46mCMF!W*>?>S?Iwt<NnS{Fc9tPz{H#Iqs!Pne`O$+?>U6tlXJQ)c1+Sa5=mjz%Y;g6?anqvR9HDJ(pT^PO*lS^@PGf|H%gto~E;(`vuhRtX!GK(8HfthG4Q#dOuUo1GiiUixw(X>0jBj%CL?P&>ZvBZG#XJB%jKdR86dCrHbs=4ifZm->0!zrA=^Nws+LFe04%qCPcRpmTbKLy8}`N4XHf~ElO=q_S9OMe$Yq|=uN4}e@0Gh6+=@*ryw*<U4a|de^C{^f6%|u|5B|c&Y;=X&l>i`)&Oj;=@}OQ!4>DVrWV{xVNpTY3&Ml{oChc<$<_p^dnF{F>Va4HUu`E8#~h8Y3Eh5m#f){t^k%yc+X$Q8*Q&FfSu+5N>$Q^-i(X@TC9d%e*2nLCr8!*$hDZ9-S+<erM1F&(C5<2QCCOB0d^j``zphJ~`$(UX39;Yq2FqN5fBA!OHQE`b)GAqLBq3eWeM!Q;E{V614$mEjze7{jh1g;tX7LaDgnRjHm}c-EY%JI^(+>ZZvZv=-b%GSu$cS|Ft(AaKk+z}haiwgxC;D|E6uEeJ266DgdyqKSJA1K<M$a*GWrQDFSB_;|War3o1|1TTBvCf939CAkc8d1qQ9PDS=%0e`qnm@QDF4c}A7w_^tQR14^`GvGX#vLx$lOO3uPvj94(9nm3JlU1%I(LF7H;agi_nbXHo5ipglsvEenX~y8;7xKFy6t?FvHn{`_iT3dN<V7S$E8pWqxJV$X^uzd&%#DNU7?efFGZnr}f|rUcjMfOyjuuwfaY4{1-+8Q`R$)z*!5)IUu2oH2!)W7g0i^ZF`XJ179!{Lv^373&A6M3xj(tCGT9QuhAwa-oFBI)(|W3O(LXI<XwNNlSQK2VaIAi+(kP-JpiO1v|(<2;QM?M(I{luATx}Ly8@;4%hvjmA=oqvGe*);!p~Ir6vXwn*+Uqm1EtGAPTX%?XMdq(h!g`{ajG3hUR@XMd{SnI=KXi_1M<zB5pL0hucVyIx?GzriLO)DM?}_Op}49bFpgI}sU@ttSI=z~tiV2oDJxasX@$aCG*R=esLTiYOT+pd%GZ<GC{MflD0&b<toVfD=kw@=3Uc$Vw=3{sV@kX6ND5^HV6$frI;=giYG0L7ZzEc0+?4yx{+pQ4;^H<zz5o*drr_cgl`~e`NLY$A&#uW;32aITC8Z5O7ksb<wvz};jvEQsQh~apXltSC1NeYKFiNJzC`3BO2TtWJM1wbMB5nsA!O#XGz1sc*sFCOt49i?H<@u~yU=s#%jSA6ff*ajceapA$Or%22F%TwvGz`=x|HUsA^7rGcp6;ZS9BfQgh)`K0CRs6TP0|La-|HJ2xhPDJCqNZm{Lub3q6|>v9m4obrhcJ2=Mf>z7tYkpe-HUUhggEMh0BdP?B|Wm(aL2iAaYU0Z_g6DY3!|99OU2P?qx9rm}flTK;?rS&DYE3NyPhXmx#wOxTlmZq!p#%wca{V3xrTqIj}s2$IUbjv1)kiPC5o^rF7Hx@cj$HwNw=H>+nvLK)&p8Z*QT$@Gh)Zsp}%(KMdf!*=8mBZ2GX|PQcRfZ-DQDjYoBn{pk$*GL`|pmx7H0sQ;mSm0~>G%zgHMy%1ExuJJ1Sj=Oo!?^qqw$P{Inwk;RJqesot?kvQZw*&xXrV|w`7I^43&Gb?|9<0?>+mbsG0z3JoW7nV3@;U9a-89Z<ab^cW$2w_2Uu!5ev<brtgG_M02+)-;`;97`3*y6~AkoRk<R=r#w47Jtmaj<2jn~GW&vzN-?MX@Wk|5tT{U*+*NjAXqQ1>E#;VOpSFrpqY0D3pIG`4=w`X9iI3*(`3NR%ygzsMSVqgsykwnMR}58O9jfJ@`E_nQA$0cxlv(HPI_z_N~;JYax4g;g*so|O+opM2h81Ov+znUKsGv{!=a5;@3V@e#Fo2nRZ(wcik^xMP>E!P{JGIpybTa)x6>KBaD#B$7rvjj>Bu{RfGlgU;q5!XV`08QmFQrN>*%XEnASf=Tz=c}9BypQ4=71%E)xcClO4m&3wsOW1~;J6mZm5RCJCNQ+3-dS3w4tSTz(4fz*WUZaG^UYZPyk>#ErdaGlbVynfS{Mgx?`uIch#Q!YhIVg$+HDTP~1D|#Q)L7%;W7QCFyXz%Hw|qaml4$FeN<&5~CFseH0`mwFsXJs^=9F5;I2IPz#=$3^PzpEpY|6|(2y<*DFhU$FIZc?<0o}_Ei)G+PD%w_31$m5^O<p4+-@Wl4wFaudnyEPMH&xuiGOa5T<lScXJF4v=K*gkd1oUt5O_Ix}VGzACKQGUq)VRSW`!8*%7bhW++=o473Dm-4$Bk-H(`!|-4Iz|pe94`ZP+>A)G!ib9&A#eu)zo2HS;+uDMsg(lWlOE77kin0Oc#Rn3gkF_aG+0*H0fo=gO#RfW;8~mRajyIxDOgN?3N@>cLiUmq!ZrrYby^|F4g0k`;dTgKqaKh+`k4u8mZ>blD}2M7Qk>_CjDU%-@BLor?3#NRvB=-tn=qnjWj}hDBj&>+})>NBpg;Az5$}PHR@m0`fNH9m4qIO1oX?zJ6~z}Z=HUylN8lS^qFe8IpqZwqm!$hOi9DlaypZ`Aii3LF|}g~o<y~y+{E4st@K|I<LXS)nTN#hx%NxgcCm!!1TP*?Qvl9dK#k(vpoH_j4?xIl5h*``Ppl-XBxk;bqx@WFRaG;2rXO1T^*}vrDhZ^+Nb-KSMT`%F-aojUOOCshH%1P<w47@_D0rbSlE@QDOu`0U?cg=%jzZVfP;Hknwau79xmN54>_e2}l~eJI-A)q}975ZtjZl_9jST&)umtFc5{>!?nj-e$Z^LEvH2-ZJr5NE`d5DmFZ=@amYYt<oHakG+#4@<62NZRRYma%at6!>XqyoN|@_T##>X*x!5H`Cbzbdhe6sWm#ltalT@z2%)QnD3Td|j-fiWS`}W)Es)6F1MBlvrJ?-&*xSDTp4_b7STf{ijH%8yC=5V5F6Ql^8LpHF5A8&`%wpw=faT+srO4ddOqUjG#{!*Dq67YAW%RYBtIED%M6|$F5Qg48QcQ(1b>@Wr7`v66t_eJ#SLT1BfW&Z#k$9UKDWenk@i^JHseYCEa$LV=|madVZ8<sQq2DcO&5I-GDVKfwKYF!->zBU+K^SFzqH373`?I;gEy~yrZ=vb#%t2H?F3A|4HO8_32Bnk7&IRfFw1|ET>!9LH99=02LVmbx=Ux_I3)sg7=qKWHRUSh`u`9Bj#Px*JQ?WEu8*dI@SI`fk<@wqVYK@3V8nMI?ggId0*J91`bER>O^Wnn!mCQsAuZC`k@c!FivY=s*4MHTDx@6i8@=&=dkCe{5kE=Cb(h(PB(6tVqp=G_1Nd!0!gBdXM{`!!~W4($hd~D{Vu89T(PcNjEAJ7ae6my=e?d&hLk{dnPs0E_olVn+|Aem(d@2%$HKx>8q>|6svV^lxbyA$ti9Jaj-C)wB$XO24}U$xp*rh!%YAm;Zq-w#DC~W`E-(dH@li3dSW?9op_k%1Ywfn(MMP}xG5a&mK7x6yG(>qnNJNo;mBoSsr7spzZt2i<IO;2GU78PHc{o8@FAr04uaf4?;8$kPd4{denr&zqv@Vu!B%i5<8XeQElTA0auu5Wsp8nJ~S&1S((BuAPFPs_89*X>b7Nx_k(Rqd8G*tkENKSnRW?}<o=kC7oS+_rizo}{!C+}by(e>_p@rwyhkBC$^Pvo$8e+y0Ax_~?n;!Cy;gvKsQcYS5|L2?7yDc`A#XT)eH$&F8?6RPEGPK``o+VP%S>6;B64~3g;%PeymAchUj|10^uijIWIF4XJ*_8`bTsiR>dNBMxvKvhtx{4fKNx12@Mt5+{RjeRs{ok@V3iOfj{oM|04*7TO#?**nn&vd9}QGUUV3ScgV&HRG4BODv)MHe<}Tv4i9BZk*@Q_}eraUCt;<+L=OmK>-w9_J}qF#g35lVQhYv_Y$_FDfa8yx?CY4@PN@+IG+lmz=En8w%#xgMxLVY=Wn!Z$|V%3y(V68o0usbr2R<@kqx&>R}H5|Je&eG_s0S#cCWB)Le*p(OGYMn!&{K#X<?IS(dO4j?82JCu}d1Lt*YY&{JhJoBn*nTKCY}(4V+ZnYJXV-E@O|y78}kZ``;a#g!9Ng80{m1)bUU4!(<qK+8rKzVP=5X*TCaoMS0x>J#>Hw~^#M0($15Qcnu`vNnR5fLUkB?pz%Jv{h3QKxKow!?90(%Q3a^I;W+!GrTzA9Xw*_!?63{FH<I&VLsEu^YNz1%KmQb{(fbe@$q_FJoHa(t@Y?QNxVfFEs+u+1FnozL>Ro1C>q7d^QpDkz1ZL|Fu_Y;H)HyrH`Fa~SW^k1+(2VZGFute5B>BkF(#E@m%+%18d~QDo@P1ZL5ES(kC}H|vHgp<YDdYAt<uw({93^j$xFSfe`+EoqXMk9VR)q#B_`if2VshXs%q=fOeLNOc-+B@rchw9#NokA5x;S!!VAsZM?$T}W%GYmvh?6L7F`x`Afhd{|J1f)N@Nncfgkd(z((^<l9kjZpFiYM<=f10nA)9&&sy?NY2K}~oz@ZRycvgdl0xNxOTM1c-G%H+EcO3(opVn+3g?~q)H{L0Jg6$mf67JL21|_PGYyl}-C2Et9!XKR*w0H3tOup^3X5I*gV!=C)o%gZZc|m8`z2_zE$kF|i+F_B#t5`X*VHso8o(C#R@$QD2m@nym@6fW=u{pim}3h>!Hj3I$DQSk*BAovqPE<=id6~GL&2#Uj85}x+$hakLTSHt?4|_NiBvZH<aO?l1BMYooB&<1czCch&)wJ26EXHUt0C<Uv#ct&@`^$5*!hIDr%T)|qB+c)<qc>)!E3&bo8pV{@|Q>9X9<U=^b#q~GHR9(Rou|I=ELDrfnjNbMK;C2YO}ApHUK!6^gw6hc>S7FuAlf^I|?#TS+eseG3rL3KJn6!75;5Q3T72fz{%Uw>GM(fuq@>E?Bmz7ed_|rh5cj?+67Ktp3I$zYoc}@BcL)io~LL!n%4doW?7OTh<zjvUTK}9RbY3JnhCD?T2V;T4`~AwR;OJwg;5gSp-dV~;<e}-HV;y8;nz${dWpAngm!wwb5)Bfg>hthp6*yMmtlvfEJ!LHx?Hdy&>}YuX&WKa9ws;w_J&`VT1D}ePyj#EMXn1I$=`gnZE?)H?%d1^=UBkRJ`m>tT{Na=6eHiM0)t`LV<yc^EEDanbKGEsg&)}HoY1|NsX`*<E^4)Ow?hfb3wH~jhsMxINO8Vj1TCnfzu{*^1C7x}y$=JMfqELrdcoI62>62`_kH01ucIm4%~_M+O^%0u_*P9H2?1N?TnbRqu4XsrF<TZm^}F^|Yt8EY*Lu+vf!$COi!qoB8T*(;&<ad&5r_fn|I#p}y!lBa@kC7PKI-yF++%+<MBtP&lOopv<kQl%zs!4E2m^fNMGqn!L@}a4Z%hQN1idY#*ec7bS1{?^!5#4Q=%L#Bq6a<7wuVq5lWjZ74O{5rOuq@$B*j#-mR=Tlu~0`nz=6HJ(LkyBCW^i^Q~)ACH=|#HGj~FfKT<TxNrI{+w}RZr_)@n$jpKS0#2whZO?{@Q*1HcJtW}hT)i+$_qFC>-tRqUE**DKXf9K)7$&^sKkvo^y?z&MaYNDqs^Pmk<R(_b9fez+}-~ljXqAKGX%Zd4x;6+RZU#@io|G98K!lLSq4V(4EVDOOb7!Q!5afoldEkucTYC*g3EzTwa6uRY;t<0qB;1iy7k0ye}H2Qt-ycmk8`LpjIR?B~h1Om}v+xiVmtIpywh$sXLfsiHxj)d!#(SysL4Pvm*2|aP3DX`h^u%W1Z_>%S3e&xDzS-Un>xFd*BGbSw>VOkbXSmlp_7!x&^1BU6B`R8WHDk5TY<XUz&w-w(H;H1ef!C*fMJ`~q$He(Cn$jsgxLzjc&xmcc_;qWTDAFlhg8=yw^R|SVAz0G587<{OF$a4mt#CD||ujy2$crN#VM}pT;)lCoW{mBtXaWFDk<G}mCqmd}*c2{y+AHl{=j~{3i&~#4|Lm$zYZ1AW2GzJc6o(|fWO^#ov*1nT?J<O`B(-Bl`y_NI{kF)TSsT=%%HQbg;xq!kygb*pFSn|r3!Cgj={%PM<7oEknmw-&x{I^{<i$;yt!k;=}t9maTof;H%3__8yl$_{?bnegkU?nOEz`ATM^3rIPs8Dk(SKBuYj0JOqEy3*tFfFL7PWiQNu3W&geI#U1H+XLaryZ-Cu_Imy33VDS6o^o(31rJy@s1};4l`b?Ox02>?Yg{?TQQfkSB?l5#X&voxCaB^aexo|9jDhKqJqQ}6aidCLYGaK&=r)c#UYMJjB}L`34i8&7T=;aZ|+z2-3%s$d{e}^d~=&@iGY`cMS|a1xDHmSy@J@)QQ&80V}40)A6Y)J(vl<u?^+u4Stht%jVu{hdj+5<9nnrNdQj9X3sG+|9f44I4CeghrmFsSeof4!Bw<aSC1O;L7j@vQ)u|qDD0=u$s5uY4SSBEhLL;8iFmHmpYW&h3dP2(4Y$zc@JJ}?Vsqri;NHjNZyAiTwMtqa&%!@=rL;#KHCb9~_q$anR^1jF=P9e7JGzZ9GZZvvM$8Tzv1mzx;Qt|)0dTSO$jnej{VF|d0DrU|eLw2#CW~k2gt-CG>Kr!#QsECHXYu<iqBw;w1IP%<&wW6TQ4D=d_W7hf>Iwcy$?7T<z6J(aTH1`)51^#G^ohX(qhtRkNh+K_tw=5!O*-LD3=_Tp+grpuZR?8l)T+R`13^4dMyLnpXhB9W_XT<#iuw*jJ8A>|RG+erSLQO0R!8tpdHrl);-LcV28@FlvTA8n1m?ZVjlG*gc!VRXx(HT~f#J%!hsOOQ~Y)!1N?9O=ou`7z1fL$)$j_6mps~fJUmwZBA22U)Tr!+FQr+Wb&RkrQIXAwSrNDnovX*}ErGl7sm0}P<sycO|pX96eueas}I1dV4lo8fYz<m`R7^2KIWyURwG?&ZUc9SOnITTi%GKzBcVN2;t$@{YJTaEfN|4b6BRh^jO0RXG}Z=__)(I^9EEe(K#9x}yE8RH!`}cUSMP{aQ<XxUQyb_TT%+&iRF9)2D+1#g;InI74Dh-(yq-z)*LcoMbj64gN1<p7_Kv86(4tt7(e;m%9W2O5}<{26NgjOMn7i!eTFyFtXz<4JoSV{0>919>+LzlC@|ahrPDu?A%Ol`mN?=6YREJT6`T7RUv$I+sO%K;m%pe^AM)hU(0Yg#lcbpv>J~)FgR4>O_iT%5YOjb&g22L2#{BJ@GgY;f9F4n_zYGXBa&PMjDl?22w_;yC{wvL$PpA%qCp@O3Qvz@d9J>u*-tdXP&)<9j}#Yz#xz+N40AqMoLi5Z*Q=B=VM=Bb!O2!|b=d&XPY*Qzha_?7!cn!RlQ9e)v<nFUZe_11=cCxvUGIyU^|^Y%anI)Z3}%~hm~zL#GB>WbSAc%*`~W-ei@#3Si9*GPF&YOt@4aB`=@yo{;dh^e^eyPdQ?Hl>Hj?ngh}B~OmEK0VwZNDt*u)pH0~m78%43*UhO_Ulj)ui9Qvo~OAKLxlMvIotvc`><!47)8e*ggoF)mjQ8x1%L?Eh(b!~lgG4e2GfhkpayL#ftv<pBfws>G%2btt(7$%=gA5^UWHKE*1;hShSzX|J1#vV2-?Mr_w!6gHpmr11ZAcor`x`;8r+FE+J^hjhn$Q9p{f;3nGkxwL9)omRLR<6D*Th96}ro;xBXk`WOSjYH47gQQk4C~HbOK`dY+PiLI{%Fy+*U<_Vh;KlU-+fr5P$1rn5Nj-;JklRgG_8RINt7xkCUB~-)6i~(6rlKsvXLolVffabG7lrt3ddwHsq+DT>faI|B@RB%ZIi1hdP4<?|?+t}qIAbkZ=N&YQz>O|ydJRwnk%J+D8S|Jl4}jOMvO^?Dlqpn83HQ4u0ri&Wz_d4^-A~i26>kmM{aTZ5eVt3PgBtr6o!g(49L~g_X+Q-W6mD2EKS_feKr8xJ#|J#ytl2x*JgUm!tYyzoK_-9h@J{1bXVPSkHo1a)w0SCBaqYjjuUB5sR;72uarI$n!US@Eqf2zlHj?N?&$4@^PfB1FrJa;Ev~o9#k>i{*W&7FTV{tX8ePV*&kQnEkC!;lRW)nf@X?<fcN5@3Yb@-l(kfsWC_sJePJcNAw11B^gsU#(vz0}f=j$qPCXnXn>M+QW-%t=C}?@7*Ky%OVuxOVrl2b+$qScDj#ZOM!@Ur6~&X*1Zqi{BQ^Op(Jub553OV=51`w?Bnp#~(ZyPyC$0bgG0GPj=zY%8T2+k^MrJ&LNCpdj8Cn8LsD;yC7|_I0pFwBG5k#<VQnn)1o*}Fxw6G6prGH?N60ND`7;maG#a;3vsQzKKxI+&Tkh00y=IFZu#{zjMa(qFj^JfZWgOiSQ6T70oSAX82*AN^f%JNdAeJlQ@_24e~cW;>o_)d_C3gMWh&>3aVW6=z**T-g-HcMN===PkI4kV0uT^msX;rV7fZpxE5A91MoY;?SyvS<t|ljfS+>$pK6G6@0IL8_hm%lD5~M^z<0BW92KufPDV7WJf>D3-&iF(w5Gx~cr6;pe`sylgm0??{qC*8f=y{d+G-V(jh}Tag0W1y>to@P)gIkJ<&#-pfmbM6(%4>~-l&seuin;&bG&iTE>@Ppp=doH752*E!0V>8T<@;Qy7RywGar9?b?5F^BXyI&V-uUjY1|id`ey>;m`d4t3ov#%7=Q<JQ;Z8|>tXvi~n*CI;_9+1+fK%E^_1>vx*2@r=V4U#Puf);opO{dC=g}aFcp2zpSt3t9Dt^Z`L_9-9x|baR8=4JP$l7^XjSt&{{*vYS0YP3gSR&6Qh{9r?6N`<{i{V=yR<7w)PCdbfCbw8UVRl5}KrLdr6ie<ee{ROmQKKSSvb9`@hZ$^W$sS)T)Q)uT0Wcam{wCA5_%~i+sWZ`y56b-OQ}#c=s7sz<nMs-#3;*YNpsm7R&@PPx(OaZs*GZo&>v@b|QtmT1Eo4<sy0rtf2uQtb!}ImD7}fT0;PLm<p(tj8dlpVE<&tz79+g&UU3HPQotfbrQ+)H->^zLgA4XVKXhtnq48x39>Tx)VrST0QYAmx?GufrbtnUq=(-))vHyY(})Md~)7oVINq3tG|R2@R=gj_NQ!7<nR+2Ifp`!z=>+eqGR=++`(F5e!I#EiLVv!l@ljMMFB`|hyDOm{U<?gc|DYpxQ$#x7rH^eNAV6XONfnZMK*44^zAUG#g9GHVaGsWv}oU2e^v=_YnUhL<m>wPTlA{&M*R_4JMub3AH@i$Xhmt4*(f%!j-x+;xT?V9MH5sBCF+u({154%6teI^!OzOB50zc$DUlzQd%|8py8s2y3&Cm)_NODrV(!)K}P!{aD;O*u}ssoJ$K9Z^7|1A}NwCq8`oEWTi4*Lo$Xc4j8L2jML`PJgE^#7Rd*Dj+pnud!_a{mQ0M^D<=Y{6DpShcD*TgL<6}-(Fr=#R4b4aG@6%gldPJu-GoVlNoLT-N!XH5=88w(?r>2(sycv)-C&R?GnEobNlQorM*?4I0k*~^7Ex7&vi_~21?w+n*b&!Mv;IvPgtejGYZwN{YD5I}hYqP<P0I_w(4(X5A8oyx<ATa`OGDz3+qIFr2EG`?UWR9_07XT0v<aFV?riRYXl`qGaW^}jREz+Cwv2|+GXAEk36%{R88uTJHj*6jn8KJHLs;a6yHJk&jiA%wP8U?VBZlUpnT<h_c!;?ab|@yVJ<S&PUwuo+emYM}xAQcuQK<Mfe*+!IFYTD88NevmVa(WOdt4Vw!d+CXb)0R?<<`fgQ3VsxHorHa+awMp+p~wcUO8|`9g?z#+g7NWN7+zR{vM;*M=m`Q#7~#4l{{NOkSeKko_=9Q9H`9=l|w^pjQz-RB2nTZ>~;hi0ZPTbLR6fje=K#v^uChTmK%|McD1^VRT!pkCir&UKQFq9C<-q&!2m#>tU;&JCkir%zcW$C#B76t>k{D;D(M5=<Cs(2%W7WorP`cpvGTh30sIUd#WyV2A?$JETeW5u4%XajIDJKW?q8*h0|+vGjltehAi^rGI*p8Ai?=HC`kS%dZ*6)Pz;Eq>erhGoDW%H`Ldmq(KPR#UNq1)<X#8?cZ1n3-@|~}gU0kx5@o1!@`qr;2A%`Wb)4*J;N%eLXID`6&A~s-e3D2GDMSC@GTb@Yvs-GgHQyg5Y=}>Fn6)q|^&4|@?msh;2w|y6BY9+&?vwoWH9^|C0PgQtU1%uOAO?A6z+AUOwa08NdqBWBN3H4iwDBd-3?tAjI7DG16pd3O|JSHMRHVRxWLnBCO`MN<jU*Ch&q=0qh?Z11BTKs!F)mt1V2BK(4WZ3I;DX>z}Ar{$5=KEBX;4SC*@2F7FfqrXg2#w{VU<Ax`41{*I)>pQn5zR*iaL%K#397#y;CjM6UA8s~=DMA7!R`tELh*4}QE49dsp9{vij0P!0rMx~iQiv5hFnzv+tR=s;gnqm0Y7g&{2ujSo(_yM3*0Ohh@5&9*NvNIoz>Ju4VM#iFwk4#em7kPU6dVj=B2GK(7J5hg0KrLUbCgxRcRx?z#b2(M>b?03Khriwuxn}D5P9j_T@9J4gK&{pjr|!BFxT$yvHBC_uzA3gPG?u$m(3np*_XMFs;*-!hDYS;#uPR$!p+;$uSAR`J39;ezPR(ykh@uvZd1OE$nc(=ufpxzAt<nu<r3I-(*OPffZ=E1Nx$k<X_~C4Rg!Sa5*j$wW%zPogaIyN*t6-+5t+}4dcBw>{(~}lvwpk$-8Z9YYxAm2&@?zGFYT?vs8K=w<9)7w6zun2_65KFJlU&|5i15PgFsD8aEM(xt%7Z?HOukCk72rV4dk;P*s*54mnE2Ez6K%@n3PvmAkxwJ!*iRUn&hKdq?-59e8o$Zmq~ta28<*4BzFH&_NfGYj)vlrkQxuPeEn5hz}}hP{@-HTG1+{UUb+6aIlHzSILU<*qH8eWuimg$;s!uG}`S}Xmv%<^^r=3#LXD40C#!ULGKZxB~Rc2RZbO+@ehj{n{%Nz)eMLGpn>kXlG(OUautAYM<KrsWBewG=<`N!LF(gZJ*=ytVkyuPd-3Dh2&_En<5Of3U77O}#t?PqyXPAhM|L>#d*7V(G0^NqSY@lK`UpvXzy`dgoc_Ku%<=?Tp|`3p?vbbPsRR@DAGyJ=EV?AXx*B>*a!a{%XTLUw?nE(K7dfOA0H07k4--AW%aJ+v#m%48>!Wr<7|F2~-kn{f1?6RAUzp+6r@jP?uYZz($%cIf=fQj8&&(QSi%YD1QzPJUuZCUVU(fZfv#<Zvy0olJJJg|6S$bPJW57lIZIM6yvkiO}rm(Qyr$$fktg0I2VX$|iXEw(#+moV6p5q-UPlS)<CD%1(JRN$zoRxgItq#+Pc7%25hob_w8~vm3vT*b;Uplcc5{@uv@Wymt9Dnl;+H^$Dym-j;*(B-oB-=n(Y<yV0@=+u{dn_Z##Mzf7xhJ~=edk(+!4t&Gy&%wr-{9w$mjeRi#A2<?K*loqbE#V)5DEmEJo+V-Ga|24v9|W|tI0DAFWx;dXVO88hV@*r2l8Jn4l5Pfi)}>$JiRCzO{D>qF%f*7@Ik@%sGTB<#%5>fc|&lyp!YeTn#p-5YuYR-OS_VLF$GKsU`8_=E;CAhV12fD?J7Xu0-05Ko(OPdxV?^`oZb|82uGpWPmP5Ao6f6g?RYmuixztnt*W9{RyG3}W9gCV1hH1*RE3Inz2D1uGxpL(8`B7Pt#3t3%qC*jt9>Vc+9@#=D7D??v+CuajdCE@Cm!B)T%(+Zqk9Udj661D^QU)Yi75A!e}jFwL9Up>FJE4uarvVrc??Y|VLp$8`Q-&0(@P(Ijb~4hw&B3|<^fzWms@yb4|rehjzHx~_r%rZ?_5=XH(pLx87ZQBvUQH<){44@SBYa;Ix2eZV<u&vYn*wuDsM%;O`EfZW_CjD!*KaDuNz-&2b;dIc|f;j0A_kWyYI^)?UkZ~3oCzKj0w3k?Eka}9;A2zJ?=T7T-e|Jj8xLx{DS0uBj1)mhU#1VKlklB06SA%^%3}u&@^EFl22Q90cZ$6ch&cIFdR9!QyrMvqv)>m-WNN8!(BhZpjf+w`$$m-QT+?t0{(I5&tQIyh~N+nx~<WgGm$Aa4K0k@5YUWRYB_?p<Zrnk!Q;Y=OJ{4|)|AX?p>=N4Gp;Cb*C!iswmCq?sJ71&<7d~~>;ZiFwBd)FAl|heO&*Tu3K!{ag%PC2g;>a)zhaF8Zk%gmB_bzO|67H4{YoK6vV4{&<&R{Dmx%yHCIBmhN$hf*)Y<<e`DDEdj5%cW2+~Yl9<f8jAo;j;+P;JNrBO|``628E`&ci<qJp8Vr|JxsWLSXa2(yob-qRoK?P`5+&NUP9g7ndC@!XHgON{-F5M*f1-$H!EmbS-(Z)(vRv9lR_T$S1oT1^yo+R)V|^11OFlq&6TD$ut$gGpPh*H!G_F{p2&##&SrBF^4XXzwj<cHE&{**jZ_esNy@vi5$?`M+U0<Ir}ZsxVnB|B*1QDf@He^ua`a9x?qa<cWk!4D9tuvCOj=9D@u=sVXk+Lo<2|c8UDka2%f7>GBd?XB7&)2BcK$c7JPA!#Uzx-^Vw!EmLJM#ptA3Je|HN+GQS+`3o^C8T|FK!r?wju&WRkJeE?>K|V|z&H6fxP<~XTmus6PXiD7dW(m%on<~A@edOohJ|XbmZnl+`#%P~Cv_SkhjhsN2yK&(L0RNI0)Ii~nIIBrIM9vqDlGK(_gZ3$Uwq{zqMagH_OSE!}87NRY7qo+QBx@OBa*OMR_WP1+<_KHHNFdze%Vdroi$KW$U|_e6_d;EI6vmR`H?mFLlj#_h9@;fTah4{fgq5+^5l%mN&x@A58OVGg*{0rNvLI4o^U;T!3SvnWpb0yxC$%K}CPf*<S&Uv~ybXT<k&a^8Z;$2QI3iwZYtx6xe{{L4NK4WnXKq>^5SVw>_37>_OY_i8n`;*%HPm}<Vl?hQ1u#AWq!Kt^DHf^*u<!s>&p8>&Ou9`1dIx95`EfFRT^*RM?(;PLTx2#w42cr1C!EY$O035+p1fJfY56dPzxBQ$AZapLUuqCTL=bg3478;3*p^r5EiYjNEG*UYJ|+}r_3G?%B=B1hmHQ-9XpfL^X}0SiHgCac#vI<B9lETS-w}4_6;7Vj@AhUXqbr~Stu7cmIRl0heAKrWM&Ejb_?&v&Q=#8ltf&bQ0GD<Jv`Naj5Uo={BF_<$T;aonObApuLtCw>{hBSiv?8iUjtQWVxR7!;wM*uvXnfh)z&0N8L(XPB2yoM$dqudYF02v}F(CAbNdNxt{4ngM)vC>f8(!VPOYxfmGbo=O)x$kf5DY`3aZoaq@j%Ecuv}A<Pc1*^#rLp_R2J~W>@+h9%u!Nus8RlMcAF((N(agdz<zIydx&z18)+x`{nV&g`KGEuo;-$buL)nic1*S1>kUn0^Inc4sFb7OE*K#+D`PN+EGZS;<)ICF@jZzPx6-vl5T;%Gfcc}9%x@Fc1t(nap=JD%MzV^lKeSwewN<AXJZAS$HMW<fQx7>L8Po-g$q~{%Lm%uYR&Ix<G(fd|y)}1F5koZI0So#^PhdDZDrU=8#$*D|!st{sn5R|u01-<!IrZ^fq}TV%vlC>_vg;WL=1a%r%tdk4sOHBQ@Mef0d6l`Ae#{>ACZgZL^ZJ(pIEL@OC^Zu_(-?ELzkiPJi61CmBD+7-r1~I;biF$H3WGT5cOu~dNLnwZrJe$|*95J1=}l>{+9|!dv=lJ&#UJl&Ch%<Nwhp!F2jv5N)Bs!;Md`B)!P!qq)q_Svx={mYD_H@C1jE%5yKewNtDTe{388a@M+}(5q32O+CKvbL_+2f+!Uf*<&K}gk5$G!Y>KJC9K|pWhT*;2$Q+0=$Rc2P_bc#y2y_ZeqoD|arz0sDFk!S`Tsp%y9@beTVQ|HM1BDCkERp80<veIFw8wLbb%eHgU%DDg9@PpjL2M$nTBhZBsyXJ8>Z<ZebEhgqV7-5F|YKShGcEP7EzmG-+&wE$!sgX)x>Y%bwNMe>;V4u_?7X@DM*<`Kl9Azndgli%uaL*AMFM^P?h>>$h9-P&4IksJS6h&k$KxOhQX)h(7`n~$ke>y6$&>mh5N43eDS;?RZ;LBPk3TCYxdk_N!`bd__{vQY3Sj)Ghp+*3T_~B7?3a_}t>^d=k$w3&y39d?v<bK?J!I7k{8dM%v11C$p*>HWD&XI<(i7RWsjx&IFAW;mN6n~qtW0Nvr1|lrpNi_Fvz8#?_fLS|}K`kfEUKW!TSOoAPul`Da-jXztc&tT+c`7?NW#`l0DU2HKa;wW@m7g+=4*9O4Z&E5u(T||*&YV-ngEO>=E;WU1WZu=b6!+fcBJ%g!06E>v2(h_h1gIqUVQHsQY;8)wwbfi6KiFQ9Z0tq2g!~TsNPrxfBmhwgo#1%;#d8#-L$Gr_u~!`@EHL?9DN!<=39IkBb>sVT?j#Aw*Xv_8k61f-*)^wS=h+@YxYt&e{N>fq;%Ryv<6SF!^yVy-$7P3P3>h?n#$*0c-HfGMd)oC89*sc~=&LL5KNS0N@m|uu`y4?;n&d}JnGTdQi<&XrIwp{lvx;H<1FIBug|%uYZt>inGD_8NWKETJ-;02mq{b=HJ*`X=n#>t8F-aEU^CSbZt7rI<vv<#rE(PiS_h`#U|J!9Moy+je>K1#dcxrEc7x`|h<;JH=l2K3$^+fNl-$z#%jL+$j44IIcLfixY+Qo}$`?+Ga_z?AuFZhPZJ+?nyF(8ggez8UzASlQ9ELlmrdLh{tSo5=QI7n>q3$V+u_4=#u+RgH$o{aW`$IM?mngQR|T&`Mvcz2e2>}FtdT<&!sDY6?q+Ami^CKR6V<O}^m+)%HU2uW`XoB3_$%N(fMB@I^}cj?n&?3iRizdbL-SPZUJ<EaJ#ZAymJ5p}R>sS=GW@%Ll+KfjsHXIl8QZZPM%-+5%savL8vxv6dj8o^7{WsAAvt7H?!WF@DQ$CQ_*nQbTl+&<UnUKVI`MBKd>t)skQlJ^)C$U+GA=p>p<PjnA+*Uyc5gzW1@xgV1r+h}kx%Qg@04=5=XQ`X0b!KT^o5*C+(7;_od4-X2gD8oMAprjIY^dBS<u#f4az|&IYr+LMR&<d+nlZ%O{ZPmec!_g|JF2v+pm8jGGK*HmgLp~}KgX|IWs|Mhlx&b^Mqi^f}bzqRfl6gks-Zf>kvs+>zU7_Eu(Q@TJHqdi-9u@(}?3GA4V_I#W#N<Km4o7X|G@xQhV)qyiqBoC2?CmR!fz{AA8V6Aip$k~HNRwtn@<*bmBv~#LNDl~uXG2_Tq-6B5yRfJCRxks_j=hBBY%sv+xr|GOZ>n5nrpYj9C=Y4dQQOy>s?S}(r)Hfa4z!y9T#L>*G3aM+emn1V_+Q@H&XN7Uz~W3230IeIqQK}$5(`$!f)p%sKjGG^kt%%*BvWy&5>BmimFgvJOQzssd;6iCOjO9cM1cX&*7}syd$=a&F$;?5VJ8Ss4`%e<RucAC;F*@lx~wD3E?kAiZb@#Ofin9SL{$Y9%|7k#{=>hUAr{p_fVVHf{%HrTV{CPsewrF;Sg%fKM(If&14M2AqPD9P6#;CfBN&{hU9EAP^^bOvn~y<esL3SfF;ol`AGeRwrM=ZC&`W(r;Y)P5(E(u#;2VFl9Y-Ex;6`nxyyI3b|7f%CbGnm&v4#>c1wSa%q)$<U6y0Wg|Nk@Vvt8%<Ubf2!fHWzm*T?^dn3{bG#A8k;w3Q2&(3}O?+_{vIj|tZ1{^aw*Oz~kpW_Kmj0GYQZ(`3J|0Edke_d?WE>zb@8H1`K4GSDfCdR>Y=*(OvzUwWtXz+u73T4*RKZ!@ChG4Xj<D<LkM>(P_;BXc;nhM54^&p7T<Gb9Gqm?D_+a0SuAv$V%+1cvUff^~kAlGbPd#Uui@{}T_{+GqZ!ypQE0h7P$jJgR`z5LB<f15~d-<0vL5fQEh3Ew8?RIvUW(PLs!gae^i=T@|5kVQWL%tfUrQ!}$6|EM1yJoQdssKKJ=*0F-MKjx;pc-oNXh>AiuR`Ip^P<)2BnBlU#PS-Zd}>0V4uTjGZ}kIxqgX2`5wbF2Ty-86tTOskhzb0w?5fEDD{uWvq|G<M5OoM>}ptPe7|yKdUtOoX(kOm17S)tThVuna%cM;Hf=%ov>maXwx`MnPQene0d#nrK9RW~^nb7?_9<Iw?GUbWiJ~0?9p6Pi|SD(^U&mBUWiU<fk?}z6K1FyeJ}FFZC-a-J!6?v3dp95k)YDL#hOpUw*^tnx?Qg=6Z1ax_zs964b|?iG7|)Pc(^fV7s>%B?-F7(64q|M~+V@pT6P)rYq@nZDbL7ds-2nF&$r(bwy@bkTZjN2hw92xWU*l&Y|a+ozcB+YJ{9l{*{j3;VfC2Bxa1Uge}z}=@{OQO`PGQwFlq6C%6)l0IU5J2T_aD_Z!Aa5E2yUQm7FaAjbxGli?>Q+!O<J>2Ywx&X*mSWl3`&r6OH{FiDjS6M?~N6h%o%)s-pO!AI_oUA}jAjnyfSZbx3zbfujFpB33ht-P~GF~RT5!I}r;y=Ov-Y!$3v`4%bpRuf~<*I1w_RPi$oB`d;5BCY2!KrbOMrWH<xbmZ)v1HqXqbX+g6!rI1o);dIqbN6Jfn$pFOO<^L)-WY>y|2?6TBPd!>G*_U)>0vdvUm(-X*1l{ocAgmml&-0OPs`6%MV7xE%8yJa7kthGrw#d-8=hcNKh$Y#S!z-8Ria`bsSZ?Y23&weQM=M<btT8$BVm>a-|415hFZLjwh(%iSy0=DLcy(j+r^=oj=3-P#2)xHLsE$G(%ggLDKA*Sd2nUyFy1Sbga=V1Tdm5bDejJ%eH>mA4uD=QDYBoy0ubFME;JH_CNX*~1iIs8Et_k@9cSRW6iT31W`Rn|XlAWNnn&Pk?s1vf0*W`^G*oJA&NYZA{JbQl@gu)cXla~gH5tnD8+;7N6Y~&d;*jx5T!u&~fbV-e5ze?|BwMdB;QlJJIjhJ$EMJ^P?@9)GrbnXmK^2HTs??x{eOBoR`I~UaLZ2l|^SOj|$C+BpWllHSM}}4Jv<C38B(-|oT&>wDntt?|_}e3YMJjIGUM;V>OL1z@x)$>0QNUVML>Z-ueElq79=oe9u#3K#o}NTm(dRB$4^IAyU)?d4fZWW%H=w#|CY#P`oigjI*Iy4Pb-65t`kL=A{%=D+P0SNR<%w_QYLFXMl7uFvEjQ@wcPm~xSr<o&3)cbWB|Dlx;%7>~Pm1d!*S<C{?;9G<YZG_lV~CT=^2pbtM$MT@`)az)+(R~sX&nEIX3_##?EUl5qzoLzKd$i{n`fcHNiq7eAvM8e1v7L>&*ljxqgC!q8@rS78pSv;ltN*q55s;n(VZ`zhC%deRs`mv6!{znD+WOWw2x7U7L(h6AIM9@(H(Y_i!r7c_PIkgW)v{1rn)unh%k#ihxA_u#>+9>w0rzIFCT+NFVL)E88C;k!=h>|>&@Qy6sO7<qE_c^Ew#p5@);uHnr*>ZFrH?y%*uD)umS=|)%c1XBT1J8Fb0h|^w%+pTo7x^pUhSl3_ye*=YK9_Q)WD&e>y9Qcv^7^0{AHT<^GB)4|tNE|M9GmDW}={5_w%oZ*v%b*{lv48b1bc^=hcIuJ0}gHPj(POZ`EK%iz7*L$*v;Z;NV~ia*O^i{|F6?Oz>2OBS7M*!E)W38k?qdP92JDaQsGliMeda62Dak^X3zS?xV&!|7jmtv}I~b-{i5`<76&zjgL$me7hTDF>gp51UTiD1s~*64@6WlE~xWTtUA|#B76t{uA7iYm19XosNKY9|;#>eH67`WebUfzy2Xtc=(KrC^-Q0l#6*ul~iJe1~LHb_1g~zYT(1Q=5`KuW=|Q^w6>kXEd`p8ma+R^vtLVW-Oxv2Jn?OL#=6o9+@{`=Iv|7qVK@C$wO+qk6Vn^hD#S1TG<~0Srm}kV&`VA{KegE4N<;K|pS&N>;(C=WS+Xm8UvguA>gqPCId6G4#)_;S*`sv+>w?@QKg(BLqyCOZ)&m`L5v^XeGVz0>dcC%-wdw*Cec>l|EShd&<;K#94Q~?7<MFVV0|G-$U<E@rXU}{XVUJ{~S2P(<<f*{$oHUaIes*5|akn-rj2j=9#~h&+ZeLS_i##|qdC>!Q`#ewe_NAF^93<(pJeE|~(v*SDTP?Ffm$Yc87^tv|*~|J6)g$fO!<$&v`1Az6T5xWiOf)6*$b8*JSgm78^}&}Vf(BhpQc8Z5y>ARE(zkxFD1+ys1M6Xf$<6rORhK427vl{^HAy{kLM^2Wa*Va)#D-^~z-;!cekCFM13)&cA>arH=#4=)Dol#`T@m)+--ibE9CzD&*)6w+?W7%dfch5+_om;*F~p1G&=*=U2`o}#gxpy|5#o`eI%7H#FKMzDJrsFoA^OOC@^OE67xrtN7qRy;+<3R>ZG>#{{WNAb?J^ekO{*tLQ_{v=cOk@d)!CP$QL%aiie+z&XXU%REK$v%v9Ha7e4Mo?Kp&mSa!}#n&njprVy={MvJdk9;8^l0?SUd5&ZbQ!vZ7L{EUA<?l<fp4Tt3jqE4rlIC&^^^F}2`=7WX}f`r1HEOh4-VVAwY!v>qcULM}Y1cK!q{REAn$uib{Ghv68YQQZnNh)zYyfK<g3B#X@)VAif9!U(B_CN<Rx?R{GAPkqf5F!~z*FpStI-!dmMeAAQxTT2t8S*?qo9?=cVjAzr7A+I@H=1yXBGM!}YnS=9qObq3{!>W+9BHD1g!oqkHI*(N-hcO%PNzP`f8$47$ZaNE|VW9T3wpip<dOoG>KSz%TICYb%G(ZUe;S)^Wk2SMvI-bB(6E*rK)*8&uY&^R)DQ9B`bs0D$0!0f&K$SOW0!<lww=#Qw5QT;fXW_tcaqcmcVKtv~V4CXky!BWxrwsVh5Dlis)SqGIx;eY##8E5_oM<C{Wf?v=LEN2d$Tw~5QJy;#VREkG1#9C8w_GxaXj{G$V>$i;HXgCE=d(oC587f|-k`;_oaD)zn?rC-h#`8q3b|(`H7hW}7Nu&xUu1U92}p;Ym+mXVI<_lNNY#LIJQIE}?*9Rc?Ej9rGvMwxhfX$#mgy+0>2>TNSw>gd+*HLhT?~`1*Hl(}FVsFnh8x#+g?{7Pne9e_<X4uj^*7|-Mdk^Tr`xeJ%(0p4*(Q(sZh630w6qWNVtOx2A}&Y$>NlVUS5S~DgVF@35B~n?Soz)WY;@A9tL{*B?@(v&{Ef+3Uz^-4%g|Ib@>O#;u0$K960VExzo-Kbv)<qf7{X1Zy}$|Ep(o40I;)k~(>~4Yb#!+Y=IyZTSTl4%o=axB35Yf_u(}xzi{96ZCm|3cydNHa^5OCfhmhIq{ZpBnSLucmr@pI@#IPredn(F#Tg1Q=d6qS((>tUOjO4`|XUg<^iD?3xPco&#u`#qhbLe?oF3zzQ?2yO&B1af)kNT^rAV=UHDPdoOAlIAGG%Uu^y|}?zqc;hK2i|%m4{M?%BU@w8J(lvcHs+39jS|_%vEQhhTzPMQH=IEgfe@B2kx2}{$hF9GH3qpG%kb0ZJB|1vNg*#;U6*1umR6r6d?BTG?=UxpQ;;beJS*f-vf!O5%l-)<kTO#h13FNJ5JI5J-%dY6uek(6xJ(wblf0<fM7v!NCoNcE=C4UDvyV_V6>7$qOYAfO&(q;!e#y^{)!iw;&!ePs|J$M8+B{KOP}~st=Akksq{Vu5{&LzkaNv^-KS8D?*LK#F=5jobUDh!4ut?{b0t^ev)?C6Cz!8P>ce*EDdlak18-u@cgy-v11mlX%CZyD1RZd#pfYP`PpV6-muW`$_@e#s3L%+6W=8($A$hH)V;_{5?&uYVxr{@EDZ5`za#t0Q&SV<5dd~D6N1RHd!cF82TmspCzX4%;Lm(TQISn70jJDF)tuO<a!xb^6#Hx9>1GteCm(lqjRDxadVHVtG8_DnM~uhmnB?zj>Xpj<EPZ~m|4A^&hWzo^%@Hyw#yy=+Mq1iXbxj#jr#-td7C@gVmj$IJ7STIn=)Xn!6MlIT?j5AYs>RGe~<Ff-6A-)#<ElgDFCx^;OI{`3r7mde8&-Do%0X-?Ad$lCV+fqF%mo%R_(CQ9JnRMZ#Ir2$}?XywT2r^X4cyp~D_G5Ql(?zl{Eq(F(Y8wT!%BAin=#^>TN=E1C-DMmOsej=)gmM70~JpSq}fYyjAyhUOVZ#^Kit0C3I+^cR*QgV{BQvZUwfT#j$4X`?ITJxkWhhQXrFsZ)!BAe_ksD0*Rc~qIzd^U%j!da42-XDxrE9x}uUE8b>R{Z(DLaRIMeJC5$+m;?XcOo`qDCZ@8LBvmShzPM$7qP!mwbd!#uxFWe<l(zbi;0;ufnVYt%xb$4{;L3WYyBU+M{yZPJ21`JF#_RMd*4Ja&RK^<o)FslhMX<nC4Ja!#>rGdFon4^u+{QuXW=q)9)-7|sKRa3iYAcE#Fj>3P0Bl>=NTvg3zx$Zn7$3~6lnk(G6~jftpCMWi&{&B0L4jolzX~77aGFGUYAxeZJvkQcZ{UO+-op}%V1jM<lF`OnjetJ@>BhYv6_zR^>cQsweLjj%J_yjiL%Lp;Gw+&dyq23E#R%R#m()4uB6$Wfni<d09)DBKuydCpi(Qv#JZ|D@$*+&-1gRU|3gZ#SpTXMN*vTXZM^sqShw0h2^^eB6*Q<y3arAIMn%wf&dtdlFlWJhZ6*Yazg={M3tk#Hxe85l-BDibf7&#wZrx;SBDQ$3+@K2SxOqW|$~OZC;m4B?dWUQ7b%?DLi!)$D2+S)5v6?)wXGYpwjz_{gs}a|kkIm3D>Hb0Dz&Dck40C4Y07lnzCp^5zn-glv&sazG(($&5|7QUlzPr&R1{)qEwyTbg_%hCkQ@{fi63Y?t@NY6^y}R=iB_*h%L3H|@=Jfo+EJTz&!+VIh04i^=EbVdSmAQJk&dU%Mhnhr>F4G}Ttw=`raIR@YmPJt(@i_<G$i*@3e`d_mipF@8myh2p7tH!>_vw=ssvM7e)G%(Iimg3{tZ`S9gtXAA{&rzZ7WffQp&e)3=67q(#d)`058?56SpIhP7*OLqEL_~QgLwJ1$Wz>axXy%|B!PU7EGX9P({#KG`twu+ny<XDd<BXX!sEuG=*4c9VFLShB`hI3!MxtVwg2`0@8re~x95{SLrmd?`B$91Jo6r^!k<6QN^XyMs}(^blyal;L)52DiGmE+@p(<n$8A`!;q{jCt4@*8>&RLA%bIklX0ASiv{g~&*cV!`zfWtVh!|RDKUv{x>d`i0w_PB!>n9C0s3U7`g{$=v=Z9h^HWCKz;c#XiT3h_N&*`3Y_7{LTQb@pCn(!MH_}+YTWQqRF;y1WuAMbPu(j8F1jO}Bp-6T_cZXZ1Yp;*WXvtt_%ti%S<A;+gV1;4Dr3*34fxd;(m;KcQ_WyyEz;uL^}3`RDMRz8>WD+e!KbZhz!CiJ~089hYkNv97HT;kp6)mwtMSp&S!y450_Ky&8cK~JRjyN$)a8+|hRAXt&Pk`+~XH`iCf=kM#@JR4JLB97Y9TkI3nPtsXZAd$a_Tirp}i9=>ILKpZZ#G2o?p-viLeImvBv#{P9X*d%Y_=peH0h9LWG|G?89|vhpLJWJN;9XEkVNzhV!Xm`}_|ZB0A?s~kM$_cg^8hBtW0L}ThIdA<?aLwhZs`FZ9KWUZBl$u5WIOAnflfu_=?1JI7>6P;zRPAbQEo|4dPY?Y8SrSWF5MVx5ncjB1q=E_0C<Rf1>nvrUO6(_Ronfz7rH1!!I~aVr8-E&Xg^?8Pad(~L?VD20wkfzLwOPJtJW;?$Tgzdj?Qg$dd3wme)1#3{dQQI1=#stN3G!6<b?K~aC<1)QC-#O$~5k+b7tK1@*}Nzd!8vaRyj#8wwI_D0SGMk^Q*|7Gp=eIot!>{D2*k<_V}{ya2w4U=|+D5u&Ce4xJoG`82f48J&uq9|9Yf?i8C!j!wT)`u5+`at#oU>P=;I>ra2s<Ve3#-waSK6rp|;KyZ?J7&49HbyEoBC8`bxv0%H&~1eCeVyu1ntj@fsb-KofE$rFyyt?cdWzIC0Ryg6%+NsFZ?5%}x4_nnJddAmW=nr(Bw+g#>WO>PP~9d|Dju6O2ksJ{RsPs7<#q=%QqyeUat+8EyOgZ}U}13=a`b>INBVngpklYB4sTPUoLm^8Cr#E_1B6(2!C6l@JX_8Mp&-8ewXzaZxP_#Dzi`GONkOxG;H>*1UxLK@q52gw<oGtOFaN#({%>7;g)!G)Zz-|LKKh0aOZZkqY}1s8Q*X+o&G;%Bo-L!r7D&SzLCEeW3yrUX9qwK+Gk-CxY4dZN)qgs+Z#Q2^NcSUb6i5%Y;oxRiGW>^?}+;zINRFSSW#%nm7{t=7u@T^%<Hm~gN~ao1-B5XdJ~7l#FXBIynBr99e1+;p>7lOlT0hqwMK+=!JCG3HK?Dm$5ok8QF7wtTdQ;Ge}lE#xtEJ|TaV(b7I3(1w#tL$8~8{LS_PaqUBcB@s`SQ=X~>XY!f|Eoe`AGgOAh&^du}&swgEZF8JYR1WR@hDW2H66C@eL>)c;5Y;5`CWXD-Ioo|=EK1-z<hq@g&kgbox>Hmx_+b_8P4K*4J5G8AAu8I_TNwQrmV2hVMvVi43V&u=AYBc)qp?LI={i&!E;w2y=G&o*w)Rv{@iZ-823)UD#tw3rZ5XcRMLG@CvIr+7wb8RPskq3LyYSYZXzTG_qH|x-9}Ka>5ZbI{rU01TWVd}&qsDP&iSweaG3p+aPq-TqyP;#<5ys9^6HF|8$4A~yE2prnx|uHj{o^+q#5z0Sn%Y?jMAk~PPSs*-_Q4X*v6Pid3_#jUauT)EFBIs|WRgT}(V`Lp_CvS1xC@_ZEZzHd&5Pg@#I8^hiB&Edw;GeW<QMZOTaJMwSgcuntObERYE%HPm&OP(9q&yqd@AD6sQ#Fe<g7q9eiKlIr*gbHufEq8^hEUUrjdw~ZZoODMW9p?<5+uHixJLA?t}{1Yl^ZOsf4%Yyu*NX7yYJW4U!<YlVls()F{3!Qk*-J>{%j+E-j#eiXAl=qN9t5T;=+y%wEDuZatJ=y}zu~-+$Ji88T)cWNH<2AgEiDz(w=&DXU2Z(t9u|mu#P!G%L%Yx*1+ZoQDc+kH0fM0dO>J1|YD{;!OQdg}tH;RuZI)GV|Wgg&uw_w+K@24sMZk22N9j5)6J>fu^hlKp@BtIG{=I$LoFlgH7b^w#q7EY-{(k=Oc27Hm5v%6&9;wto#Rz*9$3W^EN&=Vnfr)qwt8C5cob&bH{S_WH=}Kfq$G-eJKz4kS*A3>3)|+Mu{j*SYC}3LViYEqE{`-j%c*_jO+ZxQ^@SB1+g1t4vsYm1nAD;J;~~9rv0ClkB~c1hJ}b`OSt=#!h&n*A0jSzG~MeeYfvhDAsW@K1>kYqu}-$C`#!zuTmiBnZ{1Pt+flbLNwxo>bbV~E@^)NinUDAVVpBrXNbWU$VpUukMc4NDJ&32Y?JOQy))Aj^qw<4<z(#0Q@Br#NFRt=aEeuu5#8FJ@A&zkH$cIkR#V<>zD5nznd2%>5s+e2>q7<sd0_B`gh}5IWF;qyxe*zV+3G&z4NKzl)5ezFMj+dizBItgE61ET6zk8U+nw)L;5=#bG$q>qAg?QjkmK+P(hWv^7PA=Gt>{YNt3zGAmQJtYPguB@Ny|vt^clV17%pR9)p77_&+>MNl4j606nYwvvJ(({ugdj~Sg*w3rgQY+wfR4g^C9uTGD8(ijVO~c*bXE1{wQ*END@G^G=QzgxcupJh(8#6@6#t?&f(2-hr_<T3omjh~Zz^lB;+7_();@wa8@D4=>z86aeo;EFJsqsLFY0iTQi~^{2|tUruqQN7BXrhV6H%vBlJbrc5B-7N@>ohBs>}IqQm_;!rTREOjRc?F@ShE+t(nC8U(98<pbHiKqPR~YC<+?F*G|}5GoVYFWsekRWp|wV$OwZz6h#r8nv|i_8MGrGui{dojP=i)iVKdZHFpY{lH5!~o~UVY7*M;uh=mpwRr6UIM+zsI1)eEH5szkd(&SU&u~7H?$0W)dzp~EmY<W+@=w;1^2J?R%sf<Ij2Jd(anbOJh9HO(t+^4q@gY8*Lq;7#U+~VO8wAW4!QI;(8uB3P1U!=$9zu6e)dY55@iW3&jnI=++CVDeYOYO6#PZ1y&98TTVKf=K|;Bar^!|YoaNV#~>>%69ZoZ4lblN4m_n>j%GZZZp0wsOZy9f~eg6d&gZvTU`uD*^ydLefEv3;=w+G@@P3Iyscnc0WZ5)Cr021Ol<$9H*My84&hgBp3A=Zfh|vW|Eb*r6OjR)5d`Ty@cxU^(qW2pV-*M=kc%|ShSSbc9(brFkctJ^KRx+q_Zd4E=sqUZFC{Kr34O{<!gq1_S{!q_#F}|g^fhNOh*+6SS3Mthb;PPqp1h@#bA`2P(R?+$@^<>dZQneVe<KG6cCEEc-TJUz`sTzVI6{ZxOde+=q|nOJqcw~`3i)ABVmYX9ao);g%*Dy(mf48oj`wi^GNbgEDkRmQ4|VKaYDWynvn~5uQXeh5K%;~*3Vd1j?`WyKW?0rF8N?4b+}RatWr0V9WyJ3^taj<jUNCR0yY7@jepUGR!0EUw4b1E?7niOF`}x)lxosWPjUoFePpJzdKuOpHS*IfCuJoa?UhO_OlO6!F$X3?`_}K28P(ukr_K7dnKWdD%KW$ON5_y<#yD_yjw5d5{gdE^dSS6Obby!lS#c%(wI1Iaj=y!ms)+Okdsp=&v2U4eiWhOu;mgs5E4A6?U*gO;I-##QXY!bWU$AW1C~@Ti^@^H;p*?(rWJv|w0E?$~tN6vI1t8f{&|^;XoVEd`Axy8)!7du82yU~HJsfcRkE60>(biXUl675fx0YkEyn@vvi#XowuPs;h{KEjuro9diiH~4w5t`6)<HC<N@kkFBG?nvbav1uDXMCQM6D!N(W@cgF%C<BekcL2i{{A!?y2i+o9qWic0eOI#0DlBwTX~A~Onh|l1p9-no=qydrxF^Tt*A~sVBs+@QW6gy9QM7r8>4!)+6JWM*DP^*#>%kNp4~;`9@R_vkrhbryo6%RV0(*)e1^kp7ye(TE()U|rs`RJdJE-e!I@zU*Fm(o1#r+uXSbcOEhP(LX=_?Ho0N@(iJB>iK>$L>#g#7J^*jiYb)xf$rxQovLK0z2jT*c-IS-$$!2>rCN(~{e?iC~4*Rl%^&){ejZD>N=AoffycCF4_|2Sb~`sBRk?2aVSAOY;5Okiz^nq1PM^>Japa?^V}Qt7}3{mnniG3mc>$M$!Ey+}KE_AHqfFU`RwXDSGct$c-pkR;mI>R``p?5*Y`Y8tVNB}zXcbq@531;{<he$n<>-U5#EC!1DLQwBMZukejkgy}3L>s4AP($}$`JA}$J)!sWKLGJP|t5*!OH^Iva)+~q)BhJx0^vXJsAugFIU_PoCgMEyNU$ats*GzbO4#GWcgcVHzpo<B=9Xb~L;$oOAqfVdH6~oDs8<=6huy|Tl*~|Upd&J@+w(CEjp%WY3vi2m5dR(~eh6Uy&1m2o%O@f0fAEsw=Lm7KFz^=U*g~K;j4}8TO_FQ1$S$MM-2&N*`Xtpi1>0*`=iEbi!$y?Ops@>uLV##d`l+vTpZofB!w&Id@1=y7(J@lY|gMAi`CZ#me@k6^r-jTxv;U_aw^FpKXl3Q`>p%q7T3YtE~%UGl)BE4th_;9~{%S2DFt+!g7Rf|EN(A<SHj44#{+^YM?dhK9D7%`^l7!5V1MWPsD_8W_eCHR}&k9-6l^fFmpMoG-PHOp4^qBDN!_AXxCdmWKZ;=XFO9~ra*@q8c|J+hLt9lk&)g2(DwTN!f`gP9)^<J&do+oU+zGcU@4(_UOM6*nrA*_X!d?bKfh&WRV`19)>qE>(=OtpO{M<vbNyMUU%M{ht*7w~bBjOZ>!YtyFSQoK!*dk<l0CY{6Yd5OoF;jmjj%s3Z9-NL~Ly?PlfpggBL1S#LYu*aw9(Z+|61gKJO_17OVxX!6F59MKbEZN(uDfr3&KSvXbruIW7y{NSBak`rlIk&%^{n-&2o9H8W$)YFh?yx1H7C^FJvn)-4H%3{wW-P>zj+`m8nt~=oc*CQn=->_o+GpLxV>D)h2Mc4!Iaugt0L>6kpskpe(;P8ML=UINxzMUM_LhFuuCrh!Gxllpt2>kX8Ia`<h0Ff8=hYchx3>|a@=Cm99xp!d4JG9DRVfO&A?sC0Ma5_IL3d#d2*6}d<Q})~ow!dEo1F3+O^^j!)V5ZIW1hV>~T^}Z}+p9q0AxPb;#0B5#A{Nqp24c(Y4j5G~$E!bB*KrjO@qnmbXl;Gba!F7Z08{*T%t1&)&J5@9IO`_bz__VBK%bwD264`-6;ENu6j7E4XCKBU{)OQ&ndlkOG&&=8MA6Un1f@`$vWvBMA%JASDL0Jl@HhMox{7lAvtJ&lH}e*%qT3FeCr@i@3zUP+ee$?<ved3p`<`clv^j%en&>6uy{khME6tkk)RE6_v}!8SvK*<r3w`kO29@+G+|`E*$?V<;*4f|4Vk;F5>1Qzl(QQ+6mA7!`cc{uDYKTSFiQZo^nOoBGctg$ifQiKD#m^NwQX?$yNEDj+!3on#MspR(zU?aP4R?4QtL(I?$v6joK%-?m5lBU@Zad<rK;wf7u!<zo(`p>l_HU#ephdG3BqbFtl9^MQo&I*lA2U#za4XwKgi&*2*s{LGEj4mo2p-d{QL&0vX*h<0P-h$Jyymd0=y!4YngLb}0>FS-T>Iee6SAKaZj_2V?)`ajmUGkxXoU#~)wl-DF>lCE_l3VQ5p9xwt3XQRfV!u{N!~cs5>7f=>I)v>{IM_D1ELyl2BPwebV#}$DC+<M`;$cMT#U_aZlnucHg2CSe=q2?qKl_;7h`Q5KZi`7D7j^u%u*AE8}y6-&Hco+3C&$Oa%EQHx3vME4PiTSfhulE&RukCA?}v&iMVhA`42P;wGK`rRqZD!4VM1LIPR4p1bE93`i=ABi5H=_(lW_*tx_Tn>O1`lBuwUFL((HY)=2B)ljTVo<y*xRpQ?`6RoMZaFh(#nouiNa;5-<jeJl>?dYJX(6U(%RU+*o4*&!L>o>CFIx4wNRfUBwNP}M!I{|nYNY+;J(L}0`QpbFYZ-=p&Kisw$TlRBLx;)(U{?H3B0%sv=H)sQt()WA|Au_bBHWj*i;0+nc}`I?w2fw<Pi&Cn@deIJEJLhh4Ww%yc1<nkgM1a17Evnyj|Oq@k(Ub{ggNnD9Lf}=ibF^BJ_6q;$&5nq)A!wa<_Ku8@)1D*Qn(J+1QU$q1OX8M-WGeLl);?35=^5BMbHCd*`TXp|dx=bJCl^d#!=3(+med-jw*b;#xMBc1cQEEk5L==fa!K<eDWe>!K@d&}8y)zjAoh;wf^XT{Y5}laq;I2kW#QM>nXcRHYVB*1>Zn-)h>m8~g@5rf-%yKJIlUOYH2RS;YdDJoee%zto=Q1LqfrG0!#&Y2@wm&hXSnF*l9V+_3vXhH~vE;CXS&|3}lNJ&W5$0BOmaHu09+dLl&DhI2t{?m5Ak8yX1h{DR)}9a${EKCkCCX<#ql*!~)O&uA@`vgc&`!4g$Xb3BELs`w-?+V^zuQ6DgCQ}YF3bs=9$bwo`H0+*XwJ<*^fLX6J1CZ$C+sFKAd3_^VHq^+7uxaLaD6^+CdNsR3;e?h!HOl5y6nwl^RxIpv|oy|{GpC2J!#yo<juiZ!JIzE0=b4Q(dc5{tOf<HIBCdT0C!KB+zVe5>!`eWbW62CO7{iLXMKr&LU`#DiF!UxN86i6jY-jyf}bp}gLnuk7>vnhzzQHuGmQu8OVz>E;LA&{^aIr!S}Us4-f7`tH!}eo&{Pz0x-KFEXLeCN+Rs(_wijF+k-EF-F8U-R?-oFt8oXgRPZd?EDXltb$J^5pzU5u8z!Pm$`Wr|RF)<LY_pPp{qr%VaZ2BM6s|36t03|=WY_Nw6jza_Y=B{eD4naDkd;Z+m#R#F2-Vx`M4sNpuU#6+yT~!SJFIy0os8Jsq?3n4rZB0dVJj*EBHe>Sc>TI4^&IjmtLreotAa6vQN46DdX_eKF+$4|e`GxLIz=cX38LX{{R9aV<Va=Q0u5jTjsqq`piZJ0FZ?1Mc5_(vBiUq8+$*eFISx2f`)N>ZwN#Lv|1lCGDyv#Bu1;Pe#M<VA2TdUBlZz$jzl|*u=9a{R5#Ur=dq5b2la?7AHml49cK8+eHrD6Q<ZsY}7wl#Tx10lIKZ+*9wFYSybs@Y9>=-F~X3)|*Tl7aWb+YX#@SJlQ+oxWNpQNKpo1?1dgGg`}smF2s9?|am6h5_=u;2%!GM8UN#?!?@~a(61KbMmNK(x99sqp-Is8TdKbU3;ur_z^yb@-L71{2Z)3(m`O0{WCiW2b=uZ;@1HrqIg=Ez<PcfYt~UL{;)CiBT%`)`ailT!=I!Uf#_;?VzARJ(0Dnak;vpS27Xu?bm<yoeNCf8)TzV7V}ro{4H{?4)|BtFM9@#+8||}HH<I)rx9ZrWg!=sAi8GA-QP96J*yQa$$M07kCoCoEe$MK8^=w+g|6h#LyNrLgR85X|C+jWJP<D7-$D9z-gQck}AbZkiVYeElIXatav!a~Rjr_xOS3yB`6o+lIve@7#gX(0o*Ja9N5@$_`J${S$DLLbMi-0}S;v>ilq9RP6$BOgH6LBqZMm{xV2u}qv8V6<MXiIph`Hi>NdP<mAzZLz(o`6BXo1Wr!6u)Crzzape%G=-M@mr249$1gmrnWh6=vt~*VeBpXWd}NT*lwWcjGsKg72V4~=uxXt^9qOPJe20j0GaCy@H!81JEhqJo>7%mfE;F#WW=0}mdeJe7f13;5Ve(w)V4s?Alre3M?J5z&4ju>{a34mR7qR)kP&b}lU4q<(Qx`~{|AFaWKt0Q`GYBBGcg}}@W9n;>c590n2d&Z`E9NU^R5Mg89BOY82+VnJ9O}70=JC$3;#}LGD&I1F(`&?4eYUjs3@W!kOlAqmLQe2FLF76ys(+)=vxQu7Ht#*o(UJ}<CB~7uMN#mq=?8*`eN56>^GDr!s6{n44!D5x4GZ0QCck2L*ow>Wzf0Jk%HBM>DGmeI4fl=wa*H%fs$Q)(X`MprR~O!7DapNUeV1|lj&&8qgEe|{a5H-h*J;$7A%MxG3E&7+3!=+f{E5!Nlvts!1*utcfS_VfXU4&Wt#BhbqnU@3Jv?LBB5mlu<zK4MYRU{;Kk8`P9gx~aP|u8ivp@~pvM{1a9w@d-dxs1QS|3m=6He9`Eq6G2r$3<Z`}Ey`sDC04}dDR;Cf-R^Cko-JnIg$5G;vxA=@eEkSqWYfnzvlMupD37|h@)<$^C2qflCfao((FC#CZ8^&@d;u&VNlAT9wK76gRB#6AU8GAl+KUaKP|rT-(h-FD>X<x{MDky5Bo=Qkw$JHvUeUvv2_dWwPDyHT*Sufx_K>cJDuzAb2@Q1(UWgD~uPLUIFjUL1-vk_C`F9Rwlv6*EZBy0QwH=25ECWGB5pO;DW}W#C0tQ{pf$WHN(P`$rN_1;>Fd)|VM7uBvs`fEi5Kq8g)!*{9hA1!|AyIDsj`TOgwM_TFsE{KA4s@@<i@L0~t}1kMZg22rWdsDb=2ih(7g@BRRGe-X*1gejqL0$h20B^}j8L{?f&8Jem<hoN=p8J$X3)qGGi^W}3BU0@;!`@POmBC7ioX~j=@fOFFcC?p@0jR;jLr`nI%dGFSCJdQ_uHlRsEl=@!PFI>ed9;9Bl!eJ$!WzA+UA%jmL>DUC3@*-B*zA9iubOS3sJ1&U$<|i5=Tv>G}fg}ie&_qxD&V=roOj9jiVQ!`nsl6O*8m0+OJMy{~;tFofa(`z*SaNi5tM56N1XPRh^;9?Q*T|aAJ4UC0^BIVXBKnQe75GvNNik#$7q|1XA$|v~I>+l8TY<$Q140Q>NVPUhwF-L&zBrZT6KyNO3ht$OlO3P?yX9rJA+ZD0ozn#mrfL>52c4#|9J!=|(^J4%RKZH?KPgQH>!+!Jg!MS2QG!#CC}stwYsC*STsrU9wbGW~Oqrr^f*H)=$0&~!B#c}hdfS(>tl*_n;VI06NtAsX>M8{SgUxrh&xPN~A1Oor^wfg+rP|^GE|c8iORHy{qo4(dJ%FoRU-7<_0-4!uxbE`j!{Sa++S3>uoAHwe(0uMn*RcfZ;H{8&X+D0Ft;rZ0NL!+P8TLxf${T(5n|hJ_%ZHAr?~4hN38Nv<Ia@mX%^c`Tklv@KttR~y1oL|TW2i3-EpkSE<SDTIT$t)M4$k-G>dw2^9EBA-uSL<fXy^1dn3$ZoS}P!wn=`P<!nCY*VDV~)3kx0ZLkQis>O6y`WNnYCzJ9+3XI4bn+V2V8JnQ^Qy_)BQf!0hU^c*EF7WIsUZQ{H8KiMfB)`KG8)P<YlK8xZrn|k~^e=Xu~dpJ%AkLz0yvGbwAe$!AgcHFVSj|z0lC?R~X|LZm|A9_P{qUT(AuQ_iCH?QN!g)@bVA!|Eej9bj70mTQe+MgCDNW1$Qaj1^}2AFH^bZ)NC{34w?ouOf+>~&r)MnMiXTS&(q3e;v?`5PcfxU)fWR*YfH>)gJ;!9eZfn`D?jgdl4xLPMgHdYNE{>H7MccAIY=VK~(>ZAUF9n$lYUXs=60bY#Y7Lh-{3{MEpicw=*g$n>mR+tn=zcz1#wn9{dns{@XK1p2CHLFDIKbMbSdQi%lTw`%G`{G>w9^sWP*Jh7m_Lb%fwM*tZZM|s~<n`FD2YNJnTjOUFZ#cJq)`_;rJV5ti@9dB;GNH^_EO3aO69<)Jabc)9E+>v$&@03Q<5I=_Sy!SiTg-=N3p|aA<v7@=eydw5BY8*WdRn?Z&!vf%m@;yYfy+0v05>wfe9fmxQn1D(vrs15<{uxtsRy=ROCJ|~Jco2k}dYx$D`5T8;FG`^%b|7tOR_%4WD11qL=5SV5Wv{i_l9{8^x_vXOWM4--=pteWUj^{Kbf(a0P6HVkhjdI;9Z4p;703TE#t!e~_2VIFL-XWh$B7V98TZ0mfIv^bzPyQEg4aWAf3>CM*!W|j)JD=q{fJ(|^I!@Owfv+^pBbat7%U`B8}r?<Lgq{*iwmx&NJXqKl4sFzXNi$S+|0QBntPR+;G|Lrs)r$*74K~2A~hFezg{gpMyvmeBUk{rSu%*92Nftaw508**|qo`xCQmbQHQ7RCY8OLz=vK_3u3yPjq>h~j}@G~#nA}kE~tl`kR*t05_3l@6rn=LctOLHGec;IkhEVQGt|8wUX6M*>Sqgj-yapmsz?|BEq3b!`Eb%uz3jP3u0ZP8j;O6s5WITIPAE0oU+d|{foHx3tO}~+1HvQCwWO?epPrZY>SsZAcK13U)E_2zUyP!OjP+l)L?EWCwDox}e}I%}c<zM{4fk+)W<I0~uB<PzYL-i1W#SCCiDMzkqt1)8x;<qZKGPsZ$hf$xlyHd!Ow6~9jpSG&1Ph@8#V(cf7bUkfC1F~U3mLCF4T=hKVgQz9jLHE!PY+TMg;WZU>7(G@JpVWd!jbe&vD2y<oNI&ySW5qqXw2Y?sNGu@@0|te*scpZio_{G)PH|}rA#fqPtI$QX@g@vnPk$ASC<w61s*1Q)f+_Bn9TpGK2EE?_)3%De)S@C;aQ`wQ15S0<pvSL5fmH<&MN}rSy#9PFmnWxB<qKQ7jUH%=!Yce5G`FEZ5sjfq_{|f@$TAf;Xb!-Lv<v%3G-K0H0LqxnqQEqYaIklY$F@U0^-V8MUkxiR!RhIaf{_=L7&uCeFr)lGwtw}Y9taru4-d<r+9T;PIs13mrwU%sSi?LgkwT*4K*tUsi?w20;=drHzNB)saqt`3)^0rR3HohZ^V~UmCe!w&LPun8XTBn!I3XPA1em^g>^DK8+~9ec@P9%`T*F4(%-`JPiqLb;G2piO#8g&aq0Hp+X?v!=Mc~J*6<MzMYyWI;-^xk$==wOYN_$cjV(yyjS`Nn7v>Z4s)Sgm8+<sb1#}#{!nVrfU+yO#+o<+eU7qk){F;r3L@Z#_U=zqoHewTShltheTA$3$<8_M@nSR`jqSX`i-X3qdfQbW3SATV`+#Sggv%5w*m#PC9Az_Q+7=>5C!^eWoHUDCV%HElHa!`uP*ZcIbO>zg=jIQO01vq`jS<um^u!9d(6Mg1d>Q|5)?q`e)>EHpuPmrmxt=2ZEgy9e4Tg*;Jw0sRe3&l*wO7qWkj-o8B(NKb?EL;<a35+#AicCx)r#)@)DJ#8y__MA^+WEqUFm1@uNARe&Cde?o<l$g*cv@*%LkKfAM&&>a#o)qJf_>44{^1yQOvKC?50?J&HUp<NiQYv1L3UR3aj)(6@--X%oKc2Vn&s3i$0Fr}|No*u_A|?bE$VnED<7o-0uCt)%>Bh#Ue`E(e)S)||F|1FZLj1t9XS4=y19^q8RDzIaXnY*pD4B5)TDu|r|D%}+I1GpL0?hqrV*{%iZ?e`qh%ALWH2|w?Y|<lY+HyJ>Iy-b>?>-iwjr!3iVnHsaqf|s7BdqSQ~gN9qraHs{qUsghLD}jGiFT4nnb$_3jaeG7VZck*1S^tDE4l~gL*JP8&7AH!Qq3lonYG*`eb6xW!KOFr&~;KVNKz0fm?<qL8J%=rbvu#$62Gre8?C(+5a%@1U(XWZ(&N4h$xs~#|(k91F`~gr&Iq}6GJUk(VwwYGIPwdP?dZ3G$Wl<PJ&!Y3@wq<K5h3{rYOnDB>6ozx<eREsXU6^g~yPi=`eajY(#U&HoDbqpbwMUIRz#=EzTsoD`JjJ670`8es;bGxlZL4)K%7b#NG<LL)!Qli1zXNhox&+DtxY5T5?*Nmxv-5D*;@H2PkeOftR{GhX~Lz!YV^?o=K~<(^C5Rc81&2>-C{$_|U=>S+$~Lkm9lBMm^HSk#$1<AeZMtG1jj~_23H?ia{CSh_z<IjZ<kXMClz$lX5ZNyx0dGD=xI08gx=j$4N7_bCxfMN9c*}?t50-m<|!A*c_0nR)u)<ck)fSlvV9*BG<_DRjj4zx`Oad78Ss%FUNEPj-W&Po>;)n^`<Zhz1591bKcXo$Z+>)A)whWUc@$_)0zt!abI`>k&OAm9%8N)4SE_M4L?@@`NYc%hm?U{dpxiC&o)al;LhuuFFxkcM8Bb%iK{_QBmLXy;^@l_OV8{}VMbPYU~d7pp&6oOG~G2UXTKNgj_PW;4zWjaN)u+E%T=UWEYXvDzQ?R!t}r&|$IbU&0#-={|0jN@QV{wBAOT<i;>xbO6G7Z0cQ57Dh1=#%Es>uvii6DUZuQeq)cjjJoUTUu;(0mZp}}53+XNQPd4bQGPD`OiXIllL>kL!MJYyX6v!C)N(xTI1-MA7L5og~h+1NXnz#Vv)jJ5p39d4c()-AsJklOhzmYdB$OXyOadzp9~s<?Y&!XXrigv#`J-M2HUiZkdW4x7$ZT-%Qgkm;pGFl=;K$k>g6Kp6q+w~mwZXq=wmbOm>OSJgvO7&gkTF!6y3KTCZN*GnWhY_?Rxy2j0+#1h6ZJewUM9DbL!AP>Ks<|gevZbj}Yj$@c&N$gvgZX3VDfX5!tcJ)NOALDu&LQpqpmV1*DSu{5C%p30H*>vSf4lq2^rx%J$l((>nc=Y|!S{>mBj=U9|!=j|>r3pA(7x~()#C&ztd`w--IogOBO&!+k9nj{6!&ankHc*9u@s)Q4GGh+iB3`w%faRi1q94bk5273xqWRW}N=3#|<rbo+s{mmFS0hU+2DA+7KVyjK669b(0_T6Yr}7-Utc(bDL@v*F|3(~VKMoTTtW=wnK_px%0i^P#u*8B;ia$02KUR|~DLQ5ZQOS>(ih?mlzOd1Me>!YIvhfxnr6>LSHTj2Q!_GEkL=FunmO3AagKa;ACMJMCH=_dl6})7!l|$eOYnPFMZ<FksQ3SWsn6i))ecWZ?wc740Vz1H@PQpMNxTyuT;7#_@Ct>>G4_j7frFD7LX7xM~ziXZs75T8G@>fIZ?3r+|%&iA(hH)nuF&t@rnmSTV)+(sa7*x&e)H7#S=Np?EW7SS|y1{W&r?N&2W6n;0HlNqOkRWr@zCI2@XED31R58iJAR8t&qpq6aJuGL*;rL3@`Sw^xLrP=B>e+w~fxnV>=t5@s4S7=5Bv4ktL^U)b@9fjT7DjPnV*OjnVMHw(75=K?#V8cE<usda0GchyYT!OfR>`ipjvEZ`#f@<0^xRYth?%CETbT#N+Y9cun{-uv3dG=tk}?jHW)ax9>@r6KSmw1%7QJp$`xL}w1273<sgWI(vq{@bc@yai^=aCi;ZG=3Ng%)_D!I}bM`fqyskF`s*!%2-Z5t{eZGd{~&2BVjv5U@JQc6gO(#{Gb@(UsLpZapo(T$z)jC7r1D7bsiq*KRpF;4{YRz=7Abe1=H0*w%oVKcvul#2($&s`Q9=Ho`Me~lf_u$H!~j49nV*D`M1U2yD^cA<d7=hm^Z|F{gUVfYSf`pTYFj{tLrz<!cNfENNLi-(l2oT0*rwzKm!RNK-{Yes-dEBy+N{;D|4)g(pUQg-<uBte`-A1X;VM}^ijlgYX)W5cVpBOFu|=Xg20>hQf{|M&P~V;6qv$i)j#MOie1xh~Ut@va8)F8w|Xpgku4I!|Z}NmQYi>vtj}g~;^A!o<TmDonHgLT3<0njZ3s9D{aBw5ND-RbaboB{gb^dcPH(pOafB8NASZ)^}h9`@HxXlbYCaUPkZ;@O{;?-HIAN@n0QT*cY;mpF$xu6*7yuTr9Hddx&bEd~(hW{y<s&q_~2E4vyK^a#rVnfzI9Sm<;@!OJkt@WWbdoZeeNxZ8beCy^d^!x)84Gv@$wOMjb+2MW~fx!qJ=;zN)0<hK|ZsNz`59vmG6e#@$pPidXtu3NCkdosX?M<4<dEvWw9L=p#{Y-z1b;{oBQ_YiT%)*avMLD+c8CKU>I~SB52Gq10DqTr8_a0geR+VmgU$h%7W;=i4YeI!n4Q<g#ox0tOd-wfPIZA*m?9U1JTsDmeFZJ!m)Tyf&la#VyFiaU9QA;=YwV{tmB*Pmk4^&K{@mI&U&T_`1RKvI>;5oPuE$88G*+sCAZH<>t+6!vu|BdIQIbsAAyYSy7S9sa7vV6(C_uV_(yQruXKr9aW!L0N~nq4G$9agEmdvT$<HoiWhv?7S04$KE`A;(G);i?!~wmyR)uk1};cKs&iN65ylw4ySo5>!Y1=Hi52V`L}Wsr6nPF4*XO7MJ%=ZI9S0I#-qjLoVpZQShA#l4$d`QUz8RMquE1a=hWs2-w(Llp7;03?Y9n1rNNHagf6{`=A0Bz)nPRm|+yYEm$tjgH&|K538U-PO7ylt{IS2g87xhmWbI+8kks*qmSsA;q@-lt8WIBlmg-2q<2MJY?2mtX5*9xs`0J^1~3*PSdKs3T&ekg4@9GUuy8RlOmS2>+Hx1R+LuS#WLcyuN+ztqyugvW);3!X=oZ@#L6^L8g_I|bnniX)9#AN^>>o~zI7@^9#lW3VICKS8HOMR9;e@icVM-b5mIdba(rltznUC+562xxNoT7JfY<AUF91$G2=C+nc+?&?*R}nqsUSv+CL-aAiJw4!>U?K;OVg>lugzBm#nXfh>Gbr{hOa{<qQUk-(QpmLPA_H{0^wrxh@cNW3czjxKw+mV{k<k4HtA1xZAoDwEbrseQx<e|XHJ7woC5*TbQ>iCjv@{O#?nF9P!rTbD8fcWTEFrMvRLam5e3EtXNiEeYTkjJv?=$iraUTMvA}cQagyiQ*A^=qbIzR4efz&h~zKaV<;`=WId)3tn$yGtb(kJ|l~sxl)_*PcLCY$JE)w`KOXvYK@YUk}AQ@SAOC{)Xf!O_n-Vd8%Vbo@pO_;e1;x?cYy)sOhAXMA&WzHROb`~;yQ?_YumPBsgeDG7-f8<i6;iws7{g{77GO;Sy5;6uJtaFGA!-K{knsqahL)dbQqF^t*h3Oq7e%fFQ$rV;6A}a*z72JtVXZqR8vB-6Jm9t;l*bt=MCmj({b&Y^N|8A^JV@Z)Tz)MS>c~^6(KyPG4W`-`y8R)G}LySccn!<reM2GtFGzp+aHcTq4ewXh;2b>H2h|jt|_T8vCF}tcfeda;f<<=XhT>Ip9iR+{gUuyhI}4IA;A*x-L61<YBMFbZpUcnL!oP0pcq?Sb9fWrcB~slLX$oP#!b^<9xH3mgFW+%S6L|QTtvOJyPikT;w}ZeaMq=*^n1(PdrIh)4qk+-AML=>j;?Ka5`!|b=9p`Ssp2|BfEEv>iYRMJJ@dXS)*H;C4gf8YoBpNCR|gK<7z4>*k(K>4_rICPy>Ge;1Dc0~Y`1AvLS)z@67We&FY;1y)0=_*w1u;FMLBEe5v*RQULwc%@R7a0ep(MIkshD7OyN4YOG9}lE@(@rBp88uo_P9QDw^@5kvhTz*?Sxn;Z{g@_OvMG^xg~&SC7>%{J!z(<%Zq@@wmM<LEUBBE#K+V>WuS*m<0`BAe#ftpxapXO$S~<oA(XX`m3I9tfVnOGpmLw`*xH?e=|b0Gv&pl-w2n3<(d<#dVb#dTc+C#HetR~ahm|WxFyFF!go#;<g7dNI}PH%f`v@#KJ)ffSAb_n@4hw?ne!dws0PwIexvcHt6g8c|6iLD)U3@Og>>Ww$I~aaTHhEJY^y4(ZW3~5el7J6Te3ys!9%n!B2vHiN=fT(2k~Ckjhc95Ey*TMJFbL6weZ+t&Lscu^d}Xw9JKSXL16+8uSNGW$raTbpH5)5u;`GT>wvwF7{t1W^2pB2evI6F^yoJ%*14~&P-k;WzU3=yjb<y$6^=-GImF=erf2@CU9h9XIdRL%`iVuYeo}rp8n&c0N>wE!D*d3@W)yYOD>!LBkY61;+yxA$%VmnZGtD@tUqz$L1vSz9pgpt#@ED<-E2&*dJrg@rDA+t2)?$l6ivtbvErnn>dQ{5h;jqcu1y+4E?pk<1`S#-WZ_&a}qg)J4lYxB23a#;)oSB!)n;y|T#fpfYQjS0;bjB)~y^OoDdK!K<24AY;F~rrX8ihK76+Z$n>Zr1J{<kF^N#kyt*lnmwhzW}C^2<2V1K!=fb-X$xE<@5X{DzuZ?W)Omaxn`2gTW(xnD#5yvV;(v<BGn_mjmEagpVid5e<1Sv*{`<kuRP{!P4s~dO!b_P%AFEUzgs5;v~*Nc$^go5GomgSeMxr+)P7MX%RP{{ZF!X-zv$aQR-ETV4qxv$Bg`Ct?FI2kt}w?(s(rVPHWJZeze2{)yCdw;tY!pTl5id?6wk&PtjIH2zjaO_rXg!kZw(Tdjf3zTShRjy=1mw^ro|`-GVBYqt>qR>*TKXPRvJ#)b@sKzL*zZ09;b1{m(<b1s7&O{%Wfsr&^gzPzfhntkV@y%u&;p{hwPqn$HccRd$U?5REE|Mn=>$E9ygnuF%caC?TM2KEVAP5VJdD_pC}f7kRbES~Hf4F#m^0HtI%7B9!;*C=1py*;whatwcCVW1fu$Z~|HkRsbV*L6MDuKxv4DYT_*&)c&;;t`+AC-3IdFE|bS%a1{1g^f?HrXZsS64f;podoX*MSjKY=*a%h8^MWTe@uQyQXlef+5xPJ4ho;p<U)>eE7>i4-LxHqh_)xgy(QwoR8b3DB&_owQ$T%fc4f}YOv4YZq<MCG=*rmIYSpwkc{=6Lv0+)9oX~q+0Ovu5`-iv`vZ*XJBM>6n>cj#4Eb}Qhxfzmh>H1m<HE?1Rr;eC{^=2_$IH`4mZ?vUPJ-%S{iv<s!JJY3ZLZu*cKdA9%ws0G(6y}3RJk9}XM8vI#cvmxctZQ;}55JYj75ESHEO`&61)=LK0`<>z>`e8EZwqBoshj4%s)0m6aK@`el=-EV7a!-^@oH~g_bvM^>Qs_U0`=83a3?frEgb*H5!p6$)MJR64YP~}If`igKq;Lp~N_JBWGRdT7b+3LR$ft6(61qdIsz0g}ss}0S#nzaF<s|Itvig1MEEf#kKhyQKr=rh9d6qNFO-&LQC{{d`H$LMyQIx5=GR%sK{U<|PB60XeIX+xi3TEA3?#8C;f*5Xm{`cB|G)1MW>;mhk4{qyEFKIq2;0pzQE5p%Ug}2XO)8yhQuS=3oUJcjTKv6n7mevn=cTJ#<;<WHDJA$roDhxDXuM^=kV~iXBM*?PvB`U(3AJdCerUfz{Fp{<bVq>A6TU9xl&`J2&IwW#lH8eJFgt-5RHuH+~pJUg?Nd|Ib$(L3}8n$mKvL^e>N`ruJx;3EwH%IN6x=1c%(*YJ1nq%N?*9-#X07oASDAdi$6U`atBNLjE%*7zEP$8q1F~{)>{{J2PA2!PPZxCgQrllcPOxZfTpZLF`S`Gy?YWj?ToGI}-!N|Myyb1te-j!^8XD|p<Nbf93zmk;!sDGj67lqxBGuSr;Qesn5PeVaqSnKB*y`IGkmKP<h&3=_Ki4LYtONOeNm9@n(Dm<J%j;Aj5avf2nbXA}fg6c=Bg%dV;><?-?-;P1{fu~frPLdw06FVk~5=y<)3|AXGq0cEFE)}4JIw9uKx(5?H^U(+6EUWXvAJCA{smdEG$PW)Hjsk1q%i4!rDXZ9-D-y722$H_|(Q(n*F!FMlZ){mtq|U-@6C^)gJ0I~%-oqIsRS7%p&<X|uP5moF&Frn$_kht=$RPd~*bOtGmpPXFh{@4`Y6`=TFs~FN+O8nxmV{|99@1g1;$?v_EEW;(#uQ|^TwDLKXKNeJpHTqMrVcS2TH9E(2z?W6t0EIv>uWA5xd+W_{Fg4~_nk4Qdc%M^c9oZK2waD5nqmRKV(&xHA5&~tnMs1E+fQ3}!B-O@LgK>CwzyUN&cSkt5L(t!G%SX)@bI!TfjuY~-6sJw>R#%m;e?IAlLA>(bEp*=X6kra^PM{170iQWjnz(@sg1os#hnxepSmXpQBK8~o=?;J`>Xz@UPArux&9^}l&cJHvObN=8|Sg^QkjbP;S_^3(VmpWZd`cs_cvo6742tHDvsQko8G0$G*Jx}qK$sO4n4-@v^zdCVu>@^bc+*LgOiWIf0a2QP2GmD!JLCH2vE`5=qytQF+G9zZ_li<iO*2=@G9PFm9-%=hUYnDFw#Dwg7}QyB%I_rF8KxC(0DNj%5veSEiu=saTlB(U00BmpqvJDf3~7I<n~NuzHTT-ydu=e6ZDKuq>n_f+c#;oZi`&C3_`|`o=zGyBfr2$dw(u$o!(p%xie%m%`qyhY@~~r6**b?QII9Bj~f&*6E?K&#qO_@JaB)f6_m|z`1ceZmH`VZQOC7dES&%gv8-k)U!OTZuuCfiemok6UOPW04+s8F+p8sqc2rQPc73YrO`35Mv5og)l7lv+UV`L+@$@`<y@P>FbyBt+ZzX4mUSK(@yPu59xpx=30wkB=uDX{yCnB(A#IVI4PzJ?{M40^jXq*!|;zE)`MdMwPCtWzf`&-mbp>p5R8>>@+#B#F}Zq3MMHCjD?usQJX5S*aLct|$A!`UNh#0o4~D-`IpJ_=)bJNl2Pm`Gp|5sCNv`e3E=0}RW&;GVOK*!2$CoB{6$In;VOGMKZ?hSFTH@qr<OQY2D~zlJ!&&rR)BJ{AI$Njb9{@0$_v-90bFjx=C~X5N;`0KAU4MjLO*1Rz6;^A=LhqvVi7foteJV`5RkqeHEGcyfpjab~KpXwW!luEyN#J<H!_xpTDE7k#chogbFIv-_oz-w(51i5|4|j&fM_z(S`X&Exh{KSak&nt_0iEvUwBU~FT*4*(m#{5!_py%2WP$c2HdMMtH5`q(oRmYy>n$3Yr}gU4cfS57%K-d{BdgsX}`|JWn7SLD0?w#J;Pt5)~i@gtD=Vu&G0jq7+cq5P!oHF67L@b*_B6%!e~s$+?l7(xIga#I@o*(N(L*#gKw6<Nw7^CtU?X1qQ^-DX6AuHQ#O(r*ahbXv2kxbS|BO}T$8fpu-1*M=mjr6&e@WB8#87;8M{<SC}vHDTt6vxu;I10t4_)%9?np_v*{@`*8g0zqgjoshad4FJ@F6cG?Z@G!a`Qk%gc#(vh5%5S8X4J;bPDi=IP`sPyXR7!>YW9@kDp%KM!U=>B#u*c|Nvld$(!xbl(7(y)Zdsp*NnDKCO1OvCXgP~NFqS2sTIw|fh{&mbo71F<^ve%5=2Irh(Z$wY;b~G7hnE@1p#+A^xq@fHjMV=Y!6OBqvqRUr8tX1E%AX=TG3uOQCAG=`RRBRw9LQQ?hqy{%u&DtIUQ1AjS$!vP>g18f;H;%kVK$X_9@YJtwBF<>VM;8;}?JbZ3zFmc&(XfD_d>IMCmDheW9(iwFe81S>K3&?4EDfuAx!ccWdB4<1(@Bl*ah;j>$crOS;0J-?eI0t>7|C{1WWdNyKls>ReA~NoQTBWsNMhZFB@=D@DJH|~OSOxR4>KOnP%LS<p`4u`_*(5F0{m$3Y_%$(U6`lg;HRFh8N)M5yedq18l#*s){QDDg5k9Na_K18sak=|n0>H>QBM<FAUu8HH-A`Pn*%K?DnpN=zG!zD&RNppC;4Uf1d+JuaW9yt8Up`+Z68Nryf^}b)^lq0N(E&fZ#r*>0O9U_=^lR$X4SA5A>yt9!ceJvsv&Xilbc5N4Y*S&D-9B(IZF96oi1?i2jEzN9h-&*kmveu0w=bPSlmWPAjmF(Ph_3TLm}Szrv9!o`K2GGzbd08G|OuRMCcE{c7;TVfgXzScd4$1O8=<Z*D+iX(-L10r0SxF7!14q&)IamH#1P;VhAqxFu&`HFr*j|R7W*rXMU>a<jatbeANHkhW+AA4W}^C9E!xR{HPsdL?jg*dG)5q92i9bnt=rIxv@Vf9OkHylA6mHh8D)`W6a$f70co5jI=11Oc#upUv)alCv*JMbr%HK&6zCT3^*ocb$td@d(-28MZBpKfDJq%@?K_7_O%|d^7h7RO<EOvpY6MsQau?uh+2~V=xPSLTI)PYYKDEWajqgV&`d63$7GN!?oD1G4oBL%SQb^ZkYCx(p)5BX9kJ(~bNRe1C~5krMM3fCbUJ=h^MrBXL>=-x;_&c`9=z-8te&j%Us%Oi%*mm_9^}tfb4=T7emgXSW^YaxeD5+WuuPmoIdi`+R}!Eb&LsPL=|uw5pBUN0Xa@{qf!Fy*_$=6G$gdi0RTi9t+bgDA`F#QUhzYY<)g2F0jNds73TTVs^ZeDcp+a2&bR|s*Q?@i*SStaaJx!eY%&SFjx(Xi0!C3j~Lo0M_TIW+XgxlJED+4~krLfepfh<N?ihD8>gR#3{Bq_<aT{z)q85-c2Ge6R+DwqWdx7@q{V0}Njh0q9m;Dg6IuEVHv9F4dbrfS=1rJIk_R$-!bkL7<%UC04?i~DisyhD)GQ0e5L2fKioZD3)VVZO3o%AsFRb^}ce-Y!>5X&_V7^;mxzL(J3pOwRrNVu{}Ff{m+-U_#;jF4gOLCi<x3l5*LHarYnJO8SRUQ$avD>CPRttZYa7q)Z3kY)&bIY%*HPf6QS&L8iukKgD%_*RoZQ;H+*|iS(LS#7)SU7o%6$u;CQBmr>FSuk*l~vg)3vDOM8VuzZ~w1ze`th;l5Gh{i(N8Yi!cpIy{uL^Hl7Pge>$4a4~BJE#x{iiI0hCeaKo#;ojuLsI)I^mYGn0;07fx8^hDLZJk*g#PZ&iv3x#idCN90pqZoa~rt!53P79&4IOPJ!E_d9Rg0pmrr@YRbTaFBl%`@4_h%4W&!C&09^(gv~l7Kn%ZSbDaO&Hfdc3<fJ(wBc!&oJNBYrZ{wYbO=nqHErDJ}3%u0@Cpg=kWV*8mHTv*bS@#;dAe%~ZPfS=29k0uog?yc8xZ@yt{epiwGGU@6%B!lOIQ?)!GuCV02f-41#p)9}nF0}t-zfL4i)}FIqJ=|=gPA3ZA7z@-*w&0$P+-y>bM7^`f$A0E!gFuk^UZ9m$X3D%#!pLyAKJJz$*YPr=dk_|L>r)%ibEkR!;*9HZ_0B`9g$9QZ66iC|koL)ta@~px03K?ChmgAh2#%eO&S1xXW#z(gs2LeXVd=11Ou?7pI-etA<%Y9|V0||-1E3T992C@WPPdOjR+9smyr<tml^{s<I`w&b8F1>dep@d69EttM^z8#lTC#GMbLzyW2wk<ji>1A!;<x`{t8@qr=KEmoM}w|^*+Ln?Q{c@Q#di^&gU0CQS!>xdY|>vSE6Npi!1Mj_E5CTxlC^u;^!f$IN+D$GE-uiHb$r?&(HGI7A$!-o5#_gF*N;4;rFJyl;jWYjs1Ce6jXq&y*`f7r_;2I>$l=ZAniifV=EKUL?_CCjWUh9M^IT#I+|g>vo>Xh1PxV4BQ2j8v!JB{F`mg{h(pvUenGm6cMk9=!SQC#65^m8^Bkh5;#xrqiWYw>%M_*V9kBF<^0U;$hGMp*St7F-L@Mf(#L;)M8$^o5w?edooqM{GRRPou*y;eVQG=L1_?|v=I)|*DfsQo>tOrxWVI(u&k9{5$tX?4HY{hYYd2H!M|J8&O{GK*A?-TjL6E`5Rf-G{Ql&n7`Np(ZvK3Hqf^Rwr}MNn81}9WJ}$Dl7C*6+_LlS&zWc-w(*8O3b9<1)BgQ3wLuD=KbubATn89^tbr}q_-71Zyi5dY{GQk@14&~x||gmT0&Av)lU%l-7*b|dZjbuu$A_S+FBuHzsmObVX2FOkHE5Q+<cXs4#387*biHh)uW9O5#9dDN2&yk<>C500_sFjl4TzO%JnqMix)q~%gwQ9g=W_G2k}QX`a)08AlmB}f7#uhcouoR>NVXyNpL$72vXYBGu2*Jr*Cy<mG9Mn-SIPnvaWQxHiZfGy0g-mu*RpzXFt|hP9KIyLyB}FHRr0gY+CoIH_~3bfOLOOj%Dc6u^Q`P0xR`h{PBDH4B4w(dYvb&U`O1XTJiB-Xbw<AXG`~*Iu&cFTws~)BuQ!lJ=#o-Tm;yhVO@L|$|CI)r+vmLaG@j*o0ZtFbsKf(Pvd->x2aZR3l=40%as(r-6qJ;O`Gxux|#CKPpwU@Yi9AW*aCDMZ-n7$=2GO$q9e?XV>wmE>TKTI+?8VusSs5BhN{eTV|8^^v&&ma_Eal3d6_!2s(bqr%I3#EkA=8CP7uGD;{r2n=is;J3$M$L0IHiFPadpI1FWW)0)l$a#`3DFDHI9m1j)~JcqT0%X^;b?X(A<gl1)<PTTEolz6o&LCQ|)^nC1<}MY~cdnS_~*e+x;v`EpeUv-|6`S{QbA3F;Q9nyaIG=Hkidb~1Dr6wJ)z$RM~x?*1wc>Os7r2p2CWJh{d&LKWuQNJ*7i#!hw?Ihs%MrAr6sFkepayIid%DmxBgGdqhVmOgnpZwj~y6WD*^ygmzwm*T}{RQu3K5>|ksHVd>(o+KK3%)!$_bSss@E=}zX!AKpNvEzG7q5cFxDRk((N?Tr))La8C;0e9Rs}3%2?v`XEy)GHGp<KK$sOH~QvS@JzKIJ3lhWJ4$9pA_^pmt=RH2O<1tdCYT`Cxu>mfX=jtma4Yx>MvXU0faf@MsKfQ85#_YsPBhZwotvBhs;DgT{-T#c}J*+`QMTb^Ew(a^NFzcEXfN-+`H{rBV2@X(9ND4z_Yca2Kx&ZVmZm04Wb|hVRD^2|M?*8JaEk#KWz3{H$TC_@zDxZM{Z>f}?^YyFVS%=~7>FDh`;%cS7<I#yk=E#$w2^sdpd{#_oSw$^0yv1^^{2d@-O~i*Z1MT4*_PmQEG@$h-(>aVUb>ezwGhzyCW;GBz(0gSrzTkKl|QgW}cle>*5nc!Pj~3uvR3dR>vVI2I>i!v?1>a2XKwobf48fJAYMJW9)~>u)#$gj^c^@dHx?Xfup9(-~L-=gSF$=HQ@=ftRv^!iH5=`c(q0-h;D1sGOtR9jcO9t3+sXA(p>DxLqHA#}E$-i!0Fwn7fMcuKZ)YNzd3l4Io5Q6Yehv=&WP>=q&-?lzPj1SVvc7^}vdt4_+~*y2}Qiu4qnJLfZ{6?MrlWoHs)D6N~VV)CFYRz=f=*Wk~BIB*JJ>mU;*%RVb3y2!Z%4h(i(x5Mf<fe<3}A)jeUoQfVno<ESyF$4SCcteiU=z&oY-MsdR$iL208<8O%RjS`YUD|ME`g5GNK%><}*5&Me<$|C!eCMs@A<)DW#kHXet)ttm$7-)Cf(J`32norH^`|g+pMuCBXU!*&pzUuypI{U;){iWI|Ln-%P|H}f3&XkCP^d^^|#{vQ%d-5B#hdD9B8egg%G-~`K#6&2CG*U+l40mfe82cBgV-j&me{U~D^p4P5#AhA?r6T6ZI#QM-{~5?wxR8JWuEt(xCe@=UDT<WK!ExU#@Yi}g>m4~rLw3E@F0)y|H)wXjaFo~dj`Ll-_>X|cTt}_2HlBTd2gN>}(-P92!p6%*2gJXuqNpd~jQ6z&Y05>~V86|MY5a9}GQJI<K-z?ro1{+e8v^<SolEI<-EnLTQ9~j9b+NaFv0|q7(zcP1a^Gvx-b#0TI*l~_?gDTUfOzr>V8<I|VAR={g09Qs8TB?Ffr`1u%;p%?VK`;x$XP<46n6;*djLEvfv%U5@z0N!-QwYEyFFFr>3StWA1`R0#E*3CmBN41(9eCFQv_n2QZKrI$FJo!Y*VfDy--BX5CGQ)T+glcH?z*K&;?icZ`${gYVY2p^80KiSJZtQ%sUb*peQt3in7&nB(ro%cs@HiU+A|pIB2yD&I9S8m7TGc&1&n=7P648onsv1{xj~&)@Xd^M<SJwfcD0`59E?QiK`{tyyYhyU0o4B5OL=Gl(%v1Z@@;wFc-RL%}D4?=+*e$504$o^=u~cd&puX`|9%PVL+LWR8gh5#p|ibuFl6kRT<|Gk_+r{2axh${h&1c(+REsPp<3N*W}{-r@NEJT$6ra|0m~58mi;_;x<dSE^z(Hg~4Wi5^Yz}USS2o78&HWG`2>^(1uPSI?e(*P)jof(H9V0F>F%T%YRIQZq%Z4Xk?n=fk~>lACGRY#&ic<nL)?@UPj3*J~z`i@dz}+*lsyUNP_A`ye_uQ0v%#Wr_xhge^}r8b^zBiMv1j7sTS9N{9}9(;;YL5ZLGh;LVXji0DjX%ajAfkVhe{`O?g!T*jK#BGqp~_d`Tz~ZEFkq>2-mLvSM_)UqrSRN^lMdtjvBBDhFl7(U%5cf=gp+^;*cWQ*)O*;`8R3;+P#G4|>i56LLi&%GmM)kv|RBx$ev|KY|%fi=lW>3k%$b|MP<ciz6_UJtmEkPWo--y?$qx6>845MclaQt=gopNo)P0<nDgkJLn<>wXKF_iz{NrZ`qy({y{BO+rOlrnX1dA@o;l)NFf&>60kgNd}@!Uo!pLMtH&d4Y$_dh^ZpFhcV&i$FeJcN#J|?xrx7Y0gvoYOCLKWoUE-THqx+r%ld0RYpCs@k-{?zH{hQvb8jm`K=KAqKK4*ZkNhMb|M@x$Os`waG=KJ>j^8DyPjB=EDwC&QtiJD?!^21%KqsP{M-`hNBFIlbTppy#IJTj=(yIroRThW79&#oqf&~&!?iezA1vcraGs#+Fu=ne{_H+d0jjMloqAw`-*4U`Dsmb+u%dBp)CRg~y*336<yQ2*luELae6<ry38g0A}(BO0lV+}OW_+j(@>^T8XhnipR@23S-Xl^5LSd>CG1)(dbf{dZ7==;mz;q|U2z7p=Yqz3*fNP9{&q9;dyz_F?lx@SIbvw^0)T=DLL%_#Xs$nl4Kk<6bp$Tg_z)v5#2VpyKLWSuKA~J_aoip}97losA6M!?BFQK;N=e>+QJPV8%VZmBn!TNK5)={%4qW$FgX1S&~U5N}~BH&p{5P4`M6}dkUw906Hx%>2YC`Fnv2kXwbcWMahDZ=A0&wSUa$yMl_yLSDQ^^P(jSLYK!N>kFI^#=#U?K-DrRWSK(V&=~}tPL!+~;r8X#gWU0?D#`cgrroX{ypWm?jb$4e&Ru>cha`Jm+%{2nDejB_j=q%BU?zXqD6(7MR@Uh_cv##b~<2Th`jn9gENmW-i8b?SR2}T6m$b`Etg^ErGI1PJM9u=$J&++lHYp&aLNnSmXR!9ZC&xZ_)fA3gv296sZE2+{(LXfW`ockT*MHhz-Bg!<Lna`GvbcNA7d$ZEMhK;5s`k%0W!3lViRk@jd)+do`U^YaR22w~rJveSWOLQDUEelKSt4rO17uq{{&+xuiiY*zIAzpqAyAt=EYpT7hlAa~{9^(LQL_A=}!pvlQ!=a7=^~8-=63M4Zw#xb7q|o{q?)|w0op3;M^_Rk*VLzb2LIs_`;mBFRdjs|3%?_&2yYYyPiC$B<=Z4LvpCV&xG<;NT=I<Fuo(?(mczdv=Qn=0u?Ek7vq;q=aSF;Su+6iZUZJ`bN1&c|ld<toB`F%L}eU|91ug_MldjJbJYOOP^oZe<x(qR*CF_NCnVz~Zy+8T*}t&FO0rSoOaa6OFCH!%i2!G1w|V`v)0H)O5xVc(qp%;)u@l3y=%7h7$D`s-HKR$eUt=PBU4vHR<%msNJKTxlN_YC)eG9;E|874bLs<lk?p@yf!5#TUwYH}@JYtHi$bY4#(MK_i;mYwQU>iGaFgatx~5T}fa3A5WsqPrtj_!UwnD2$+7i2&D{vYk(7t?kS@aW>6<|e%Cl1C?U*~apH$j9Z2BmL}?2^c68q}NIG3|Vy*k+Y{_xj=<#-=Z}SjJ0R619CKL41gCS_+49WS^96uSlqxPtDsH=JXbDdS$MG4q-bkPX~I{R|+K?e>Bt6;#Ry|+B){r)D#DWzSLSCbpKKo#!C>QXTPNj`E$YZ@qNE_R^X%5Ir_CbeGTNAnlle`*r7-5ywE9e&{B8Y=Az1Mzjg7XIqir-*DW%K&!5aO}zU(YZB4!*DL}vphGGVQ~W^)9I9Lw!Qv?_P#Sml9ct{A)Yu^au$X4hc(4E`dV?<xlYTu%L+B>N1l5)`bP?0Q7uL&v4ynZDdUxzVGCAJTYPzazqy0OUXu=;JPEs$oEfUnZok9QN7$(O)!K~GAq?3^d_=&jw<hd)y?@sy{!sP0X?farkaQJwHhDt+Lo_d%H?VKr_7Ct_REO0}^qszY>Z2Z^AW#)^m8K4uDw8!DH?m6}p9}FD!kDpbKi7vBL)Ctzl|aXE2IsM5@|HM{qd`+2m{x2owXp$!jT^FYr=JGdLXMjh+gXnR{zUFYzVQ|2S?b`fS$t!!uoE{O@D6}0Fv+XEYHbsJc4qSbZ1$k{F&YEP8?{PMkw+V6$1fs6df?-0WyJm-SNN7E+h7#^E|Pqzz?Hdqy`b*t4|JhZ1_c~^tuO{Gfo6vREd(Oi6_^8xjs{O0XyB=Nn%FGFIC1TN>K14LQ^1vYly5D9LlFtlS#~m7mnv_*YUc&rjuc$BSSyfY6<SYfes89!xFynrY~-?z-8K!((I2q_s+B8F^TPvmFBXWtS_PVDh|VBOz8*izh?gBbtC^MUM+wiNRmnixQxwY_QrZrgogh=_G(0_lO$r%F1HVX2z`jDT4vSc^r8&XMQ62U$QbehUq5=#NE}o3aUKT9Qyhf>Q^zEOr13~={Pq>_D6pNPe`Q8<Fp#3QM4J8hw95a}3`H6sw5NosOkR&L9LE$hFIn}%@rooicLeK`#sn58)Ot_^Xdw^_?0VPO&jgF}iw)q%*TglR_$-{iG-?;{fQJw|59I)d|?qw-)sP19OBX0EuzV02MqXNFe_W@d&j_xDUjX|%;O&n3Xv$(6Tcxm1H8+nD$5){o{>*zwA+edlaUG|plLuGeKpa<YuyCj%#he%9YLX;#1S<eIxkM$?gZad*p5$+)Oe=`WE_qk^Gry6nODJ-RsT;b9iC-SXA`E%lU+D(zEG`IIGMZQ5cDHCHLA}nI|kt(>3CR9jC9pPMvS-cx<Y1HV+1@S@mo34|--YwSZ%okbi;nlyMXi|HrUd<tY>KI{fi~mlf_Xy?3PLHI5)VwCnfWLVz0Ktpda8fZ?WOiqr+SvLB7hOm+IjScNr2y*M+8%utEMc!i0SgsNsZ5I8G5cy(5_3UHbz%4O*p$rx?r7HTS}>$<ak)IYGUc1%#4dO7j(ckL4_a!2Ye54Ha1wWy9%>a_j~J!{udF2M3No^xIL*b3uSH~QLe;q7J4&#mm!%x?y%q3r;iZf##2zG)uhm7%2~80-DcOF=1!TWo%dNEjEpQ%;IQ?JiEZTywIil8keuFDfHKm5KUoZ1}tV1EZ{B~2INN+KklT9WC`#p4ei1A7h7V*uIuRq=iI~hq-h}~H**2me4z$Gr3Mz(w&;&`jlXGCbo^RPr~Yj71Q_1n>@wMu$d_^AWf-0N@r{R2bNc(0Bt^t^%~M1k2{*gdh51X4yiUH!|&hKfqiP^^9b)?7uhHJs3r&J{mps+sfP)_=?yD^h34OWVL|{yYLzj#A`qyNGpi)x4AcXRm?M5JijdJe}ZsfL=wzhqav}GFKUeONI2bWeCU<LuNo431WfN0;JjD--9^^HW`x~h3q}#VuEuWc&48`NJxj+AhZ@~*)ZbDa5?H6N#Oe%T>biwWtOBvK2DtX7NB8pogc0@mP*$S8tsJhyxA(^4Yx&nUa}YUhmxLqDEJ$JI}3?d8P;wD3}EAIn+*WVU9dUwmHDTu4pCXa+CX<;dvFl2`H+uYX*Z{+Z@r5E_PE1`@DSk6B;E0Y9?AVZTuzoNRg_0jCo@<Ku1anPz#<dcGnAJJ^;G=(xk=)rnuTFNmF<Y=TlsS74Os3b;J8~;!;XCJq}1l#{fEJgT>Q84iOrH2%ZN={@KQ@vdPy!_14Dx0FXD)C;+xqCEx^_O1i_;RQH4viG$KhEXoo%~UGTBIv+TW+9h_P@@>QP{h?K6;8K(TO$u-0+`~eG^H5~;F)6#UqCTe>iUEf!f<M(N}C@>*pmV7dIhm>KXB@%di6dyo-l(J@>M~cZxwhS@~WvlAccgAud0zloKs%H%`l^{>4dku+<_i-}=nF|_tCXtodnEk@OM2Nr?_fKNxV@c;Yf&sn3X7rX4XSchb^0|M%rI;NRnu-)b1U&7_Bs0CR>Z?Hl;SaR*W<aPIAE^=X1Tt1R;fA{Ll2NuS^#;@Iy~Bk*gmLt-SXDOeYwFiHn$h^|Eg>jZC()qTi58A!7>2<iFp3gRU|yuoZH%T2ib2t?)?;w&+DKSTP-uF(=R4+9NkZ+<IY%=mY<y~<^KaW;9|oR@zX0Gh{Hmdx)^^H3L;%}6LVbO$fef2m_K#6|*g~^FnDy_i1>c&tJeq`^`o4a3unDsUS~IG067d*M282pn><iL!#WcL4mHa+XkZqD<KzosCq4w?EgmN>xJ8{)j)quitb&0_rb!v}@cdG2k1VTm5&;h~Os76g@%7Z<13vFexM)@NPl^**Pc;uN}+w|_cV3U{hGMV>YlM51A?4LV=i&?d#Il>-*tjsS2UqlE|;cQz_v?kUUSKR4)$&g^zfP(LI^i%$pp!5i*V_s$)8@?Hpo5@8U`Rg!=iW1X%m9#vNOuLkt7*GC$p`e=rivv2DJ-k|Oc2ojL+ce){i?U?C*N0#&4*qY!$u)3HScLb=X<%TABPdgu?qp9c-(x;7kBmg@$ipO#rR`sqR{0ZsD(s^kIGwF>T@N%rHCPCO$04DWD*KgXSHKw_=4VY)Lg8z%X?l?0SiQm|UN*Bae>=bU2&ZOE;fSHa`?8y0G_-o+*NXuuYOCM(<NxoP;Hxp7uI>FSk#;45j_|>TVg&ExUi_^>KJ~a;`EbVK4I97;#8Cl}p>VYd>9H&g9_xH2GJ`dVjEBPCE$xhY=9rmQ$7A0C#zK472twuq9pYFfYcM_kEo6v(b{XzmIREu+I2CQzOnHIh#Qi|T9gvaeM;L3dgrf7c*t5W-QR0yj9|%lYbw)g%U5C*cGJ|<M(EP7{0SyJnICF(umBgfhsVYGh;%qF$TBu29sv;zdxL7gf8bGn~57W-$Ki%>_87bRmj-wN{skb0~Zmt$T1F;c)TB;hx<^)?UGTblUuJQ!g3F$yqx#E%>wu?jo?)_4S3a?17zZeth^@lgb)ix_nHrPj{O8~0%n5v}YaKg2Ob@HrKIe7^U6si!Y`7wgX7t6%@#)(?P#$qJZq}(uIow90#hg~wAb6b9(6|yi9s*9FXPfQi3FGinb<sF`{Qkk(HQ`ul{ztPRkdS7`2PNub+JTn4txp{Gb8Q$-K%{T>He(19e8S7)T8n*f?*Gz;@7|NsoSEiRVWi2X6<k9@<_mP9ph5FY6=H#<=+0Rq17o#wC*gA{h(~IMyJrObO!RHNBL5uFe<&LUkc@(H?7{OgOi)SbkysRG2C3U?k9m%He;+{-6Z47^TVnQUFmj&KHD+W~kL%1FNcj_HW{~GZjrd}#oCp2v&5^UFG1KW@ta^jD8Hj<>}3<u{fioF9XPAj~PPw4oHZqiwN@3R06eR?s~rfw<LiRht+;m}$grh2V@#06vtRUvi@5%>il<4O_QNd1@nJLTi?4wrT@aC&-_fKqz)0X|b5R%Id6zFD!5(x|cYzZrSuqMsjuyN4q$Z~Ag_3oHgolw$zimbN=ZO#HJUTj|T69wKxQ1=no}n%oEq3@?!`Jv+U*6kr@jU~UQ{P|PO6oj`*An)D?ub?&4gHV8aBR$o>+24upmYmB@NMuiq|`+#nE4TL4N!2ELgXI}N4wnUF?A79M{?hvX?HCE4%VEv>jVK^(!^9i=X^&J}B`t`3fEUCr4%Rg8rLdz?S)(yUt^|m5r(C+5?iyh&ul||vH6L+^-grs#Q9<9rmUv)fm^yNewq1mKmRIYDRh9XV2g*md~te=PMQwS}HUFkP?LqA5w(f}4oGmKyT(Kk>RWJ)^o9NdR^^&C`KENX?dBhFSmdEar(lW=dQtn-`8D91YVHa<pm_0GS%AeG8;#zmU6zteV*#KicyyQPHOnbngJW&;;iLr#1UIN}d^q(1CTLuDjYJIN19kJh}J*4-{c656zvGA3qE*&ei=!eRDFbv#3z&CDJK)xd8JGG&#{?V16gjbypTqNz7D+mazDl>ujDjD5ZH^}RK`4`SC2!IbgcDHt*+0}qj!n4NEa<j1|VcA4GiS%BON%7~8^&`^rdf!<W1p~G@`%0_P#hX<N`xXB#XtL4+S8JS)7NWD|tO67sj8HNepWvi{#6pYO^s*@x39{j;k*{cqSgIuH4=Quh|c04}%(kKZV^*-~AMAudaI<%sE{f`uBa;E1|A;8!C$14d2Bd*~u!EtGx^+OF7jXHm6y3yb-^t~1>jeZTv%}zPjd)`jLz^SWAJcEUjbk_irlu5)z=GnDP5><Ku{xUK>c^C+l`!lX1Tk6*qQD3RrRvBj_BQb!E9c`^OCY-6tTtvsL9|fH~63=YM1C$d`$ql$S(&cfZ27}))P-8(-5v$oG?hCyuTnR&_Nl=hcpmPZzrO#Sw1gn2yqcOkjlBAmDurqluIySNQImvm6hLlkEB00+o5!1c5jeYb1hR!YK4VM(q!+^6fvjmd`524|&usQSS+Mb6TL-!y~n{5*9udk`P7@A2dg5qXYAJl5n82Pz<uJqnSw&ueeaB!k=s$vr}JU9*`=_3+mxbKnZ5`izKt+wU89x0ELej<xL2ufV?VkmRGP!g$i(>*kt0A6JCuj-M%7lRkruV969Ky(hiN=_yUrAEp6;M)w|If_piSr#<#ixEf#RbL@l4Lb`8?W1;qBn$yJq*qU7wo=K$^1^l`$!C)?1W!8XG!lfTX4E1nyGDLa-TV8;3sI9B#7B?;*y)sIwW)!(H<o!~D3LcW9B>}gDc)UQh@%MYpXGArKsom5Tg;^+LYaM+au=Ao8oorvaM-d+H+%9cWBV;R)QxygV7WB&Vh~N0nR9jwudHsA_oHpNt@1>vMPg^kIEb*?y*|hMkc?1a6iF-!Ce1ie;8kdrp8F=|mmItzGPQ|B1<yoM*ifAxV!d5#$Z)q&r6B!L$vdNgz%aNs1TR+$yA~*o?ngItE5-d(l8Q<HlS~m>vvRbTDi`v4l4}u=3D~yUe)Mk%LRNgmd%|5(<w-=kKqi-JVjf)k>iXPrUPW0O;9N#)x$c)Mk-`(I@*G}1VBE2O#uQ2#uZTk31@w?@yZ@bPbKHxWNM3)VvFx%7#WMWJQDlwVm(YqXAHHE?OG>jYjlo=Tj7NP-YZx<+)LMLvI&BDg%#2fBs7^Lo8|p%z&FxQ8a;W>#pbm@i@q+j!ai<3V^!7kAq;7F5wlat*XQdURj0$-1pV<RItOqT{mX}E1;Zd7$(eD9Mv*=j<UryMT+3ZRnlfbwJpeXWST5@MfHPQTG1~!yJLbF|*go!J6h5&nMZJ5mJ5@MYtvlTR<#HBlCSS%VF^mc7U17!|Z>1E&y#MK9(6G(><o;yiUiy8{nmPhTH0Z`B1+3+1yU}?5AD-C*TlUyefmrqkzG*a73{WI^62lxcHGEe4lI9u1!lzB*0Dzu;Y*;n*#Vd)HQtluBtFj@BYEpsf6(GqK{!h4jrwl-ao`lO37m#!^M#oQnV3l$}2YmKKJC91_zN;AI9v#O2#MX>x|FD?6rC+8owUEb8*h)!<iK16x?&*xK@IsD3pG3Owjb%u7o8&Khb*Mo0ex4zO_<DB3#rom8c@FVgH_Tuwo^zqFnt{c71pAxLHuxYfrB}`hQ;Yp7j?t|J>kKq`dlKPjt5QoekaE}HDyaB48jcSzV9&S{Ic^%o~#FBJF1@lkjbtQkh`on&Rk#Uu2YbWziqnXJxiv*3*m)DT%Sq=o;gqE-2@aIvHfqP=iMqdNOQ4iH7+MTPx;q-RkSoA!czcOqT6;8%}NGT>COnO(G8j_3V$%z7zX#mPH={vY+YF^?X>n;d^&0>Ef5dt@&8?peMd3R1Cj-_9<6~@pqk>$x)Wy1-2lz}l8@V5^OaMYqkFL;$0{m|BqGag5Qa+wZe)qupZ4Stl_y|yEL(1i=}pyq!1GJtlG7lMn>TXqaaY&1}n2T=GJGGE6*AGK72(;WK|K`EfdAflWC@qRvU6#$+Z?x3g4RxZszeirvuRCA^igh_`)eiU)+gs{UCZm1M53EqHP9WgYM5d=FIDdF+q|Hi<d?aSzaf9`pKRQoIql$`guC{Rwg-Bt9yj{O9Rt=wd-dSndEd9WIDm@EvFQ|M85ucwOApD<ovJ6wbx(xIgZ=(cY-7=k@EiwUR&Lkq4uBxII9@&Ut+q!Wk&eCaA<D+C9t-~@7xHj=(_pX?)`hydW-FanZoR+NoU8-$I8s-N)zj=lT^IU=Ey5g!+wYSS~xZGtz)fYKXndKH+&;?=l|D{6^?8J3$KDnt7MO~hBd&d%uuw322u9UAu(k54-yYaks+Fhlz}xV-g36%N6&afc;q<h4+dm-O&Dn61+V;D1L8pN*V)qK=2TXR~>4>ceW*j2s8YPR4}eEWkFFjQ<7jy$~R_JRz9UE-C$bS0^r|s!G?%ZJaASB>^nxj1+MIn;^lc8JJV3Bowm0ot%$a+)z_Ov3S5C+r~uO(KSQHY3Qgx8VN@C&h<0_G)!A29*b6(%|ZakEzl_KRrJd5&CKabXAKV~Nxc#)rUV-q_{k{ZoGasy)}BZ~cpAEpV-uAEk}yjmtA*2*@9*J)8hh_}w&Da6P!)XK{t=#?d@LBLi1F*3z6t6^!OWU&8|zYe%1Vs?YFzpbE+>*MF<D%?yev7mmJ_VGbQjZtNxgTO(Scz2kKpan+;OJ8Z)aBPgm2#G!ia!?5E9QCCyT+Aks5nM9!OWJ7e_MA3w~#)h!R*I^f$G#td4M>E8R$qb4a?edIal{gfMV$YGcTZ#T>|<2&q}NJRv8g9?5H?wPh2+XCu2edVk4rZ?m3=By5QUYPe888d=wL!6zuZvPmq>Yt_yYgpeL?Wzd&p1+w#<&!aA-U^n({CU*@Z4U{aX<(mRtNqh`1kj}~?ln6OEb!VJsU;T{5uBu{}j&{c3Wk%KL06M<zI9n(gKQa1$tMJNf8KT>qvJv<VDq;<dCuO5dYao7Cpp&tfEb2kpNG~L>WkUqk?uo*|&mTtL$oSl~dx5K6Uvn+j!$su-3S0XA7VKXKAP2(I3Ewsu@Q)0^omYx??2JrLzHY0XLU+zggfAo_310~>*yuP=;O#(2qb<f1rA6LaGl(^a{fs+Q28pjQcZdV2cq&8+AreA<WHd{$Tk;ewPou(-DOc)JQ99T>Ld6&_4n<KloYCE_B~wtSyz(Yz!$bp<O?nqnSE1$f2aqGP%|hcc{z895XSCDv3;SbzY+**D4+IA7`I?jw8f@O7Lye8dE@>9Y_IX&ZjT)$%Fl^ii;I=oD>(_v|Rk#Xv(mN9!T+=Ga`qKaniQ0q^pEEg?@AbX#bm+I`7a%7xKnE4wu>L3T`T$$e-Ync!;gqz_j|%ZbQrMR0=i#9p7JB{NX%^@&pNg>SPCAnOiWf)Ze7#}x|4PURkc*+rki27y18eZu!C@_hj3-2y|7r?JGf?a&&6=c6CIlA2+=K+*u0+57S_;>Wd<fh$)xY9G_>XN@?d@C`feuv0)jfZCnVR7b95}@P1meQ`5(|M~BgYusPQ38}qBw_Bm_+l&Gno!@acfLT>hFXJjKDOuDyXyLDj7Dv_eS4&g}&i#NxC3XWFVw{8k~SeFuPogP{=lZT^2KhjnY{&+_G@LT2n(Vn|{wB`d5om64gTpWF{h=t}%o$e@Wp<D&lx|#52kBS2u}2+U^Wt;pv9I^U{v?Fev=vnB?x-ZE)iF^+b7)t~_*u|Iv8TnmILG2oKX)FAK=<Ol#<4o^b32{`Zpeox}^T4e{K^-0s=wbe6>@M7Emub@zR9Z!0fzi*Ze2BmWUYN3eW+JCx3cpgJA+Vz4r<Gi({Y)fJemcA;azRvPr*<Qv02H`|WQ(t@L;WC~sgdVA>tR?^@c^t{?bWe5$Lv@zEh)x_YUwqK3(T$3!GoV_ZaR5n>V(<dj(;G<hMeN48Yy;&xeF2btU{0A>JLAYc1_~%+G<Yu;49?h)j6|E9!U6<V>Mi3WFaA7Mow?h&4jzDj$_g>v^r5Q4BsAM$0DSjEgHg2>z#9>=U44uU>eCRx27zjEIN1&2r-jU(Rf(KS2rJQ67EoVoXBKuSN)JPIO8EZoP1L#rr>!?e#aAej7P>-C0PK(lfmvPN&A2pTwHHt><J`LpLn-di5258)RGIy`yBDQ&^@jWuMV-VN<A-2O`UW#9B#4X&1`F#f*V58L0|Hn7ePRQ}ruYZFt6X2cxZI&0NO{r5GH@MV?@u9%u#I5bbM=2wfv5ACXiR_fewtH5iPb-Pu0Qg1Bsr}%}yaWY$ItfBgYQ*1prI%eSPZ|bK=?MpMsVH#Gt+crD6`+xLi}~<1bW#r1m7s@+?(6(l>fOKu-w%t|^mvot$&G|`$^*qdqOi@<7O2@$xij2b{|6yP=t<@~t+%x~(Sd9M2dnsNZdaj_=SI8d$|Iwcf4!?~Y!N7B9r|c3S0%`w4=*WvGU-H`S~qKCK+aG=4)fKIM5y+fXj&{ApTd?AUV;K2Df@@U%Sg1zD4wC)2>n;%lm*1Mi5Y9%!+GU*G&!VmV(UE^UNk_e#J1yYQW<V{#$Tqs-l{2>%I&pA<$rCouYEGqTWmOu*g95gP(>-cmE`>#?YG)PP65if=e67xpsKBiC-GS*m$mVrSw8?8f!5W81FXz(6E$bbRjnh`wcn5jh*vituX@*(a2kga&;|czLSUOkz=VWVY)M?UXHFdD>?V7h*fZU$g%gs?c<$mFQHxO@A9cP{SY&{328oqr3_NE6!TeJz;G`ki-ZETRrzFTB!Sx{txhBStT#u;6%TIGy(HB<uH6Ba|Wd5tDm@>W<+W1$rR5&EW&uCvWwh!pMNmbrLXsDR00eQkFg%=V`e{Nk7>j?Rq&A<FkGsL`a2aC)4g2{cM=6RH+qie5!PcLT~prqC}H?P&cnG2_;EVw_X$RCgYj=Q<I)9`2Q8TP%vnxDEW^v||TX;z7LJ`kbzNzE{bdyL+WG%oIr1I+Jhi4qosqPw_aF~0GJYfo)IaUm7r?haSvvF*1RTxA@Cm{b^dA}6=>q6qxVIiEDwk*ffm@9hBJT})JU(F_nL??)1oXVoc|(r%f%Mq)vu4NxkhR<k~2NAUKzW+&A8J@#a5Jta9E`|M3^?O2;m%z#$bVoo_XE98p@gv9#|K>2m(7|D%kk82@+US0hCrRB}|rxfqvyApQ^uC%d@vSgi9QyI|iLv>99dy8iM5{t~oC=}R>qBD~T6ii7F$PE+^#KFKcZScJ=XTQHHgyeiPtr-fyzkPbE6a2+Nd7&f1gUcoHs>&23u=z-2uiWI7MY26zCk_2H(|2XL<!Zh7cU*@Xf1lp`fmJrIh_nk&#ePpv?Pzb9@}wNNd!YlBMIg@o+!j@cNVzKo$0o;$=vNS_O=SaOL}mD-;^zo9RWevbg;4DCmaPZeN1;qio~1wIbA?xYILHp;SQnM9O?FB0)B@i)-E;s2nUqvIJdA%#Ro=h`2LUKDulg-UF=xHu!7H|6RP^e!Q;JAOVCaeoNGW~w!-V<c7~=VOm*ui*CxYsVngSKGp96YXE*@r!{t7qc^46gGQk|q``ta0{zCz>MCFm+IkP-e4{F2X~%#47v+W#%J!~7EZjvlt`Xs%ErZd;X*k!}q{l~eLbP9da`RPR(<?t^u%5aCc7u;yxsV&Mz1Q->q>ztA_G6J-=il<?dhc;?yOJ$wi*5A>eVO?=|mcw$0v4Bpq5Nh+a@gfKL1;spI*LD*=ktRPtJly)(}S)k+-(u1a?i^jyK5$GZv{N<SU<?mT~+GLW>rldY1Q|}HMS%%niuCMK|w(x8`ijkcx?O-~YJ*n1FaEH#wKQ9y>1ybHuSq*CRoKc>3{0!7~6qXlhJzRJ4kbtTJuVpa?!~vemXrUyD+Lc{Ordf*o0Km6g?Wgy!yQq6d`o+nEn^K0Lzo0|=Kf+llgXkaw+0*nwZSOtUYKJTuo#m9HgiX+q74V=`9$O=)#WyWMsKr9EU_fL_R)=5tXukGPLjP*q*oH_?oz^TvJHhkz>c_rUj!Lp;d4zChXZZ+2Uzx7f^SgZ}R(pby=|EE+wEszx;}9@tM8wIsU5!#AI~r4QU7eeTA7V)HxWc=DbA_GflM3HUoL>PR{**}eS^gW+OgHFee91CEMKrhz?`(8KZW+BYMjQ}9hLf~5yb;F#CWH*2odY?X!n{BGV-LFwOmbFsa1Rq^A2$=~cIHsI*HYziBX1Is18-R=5z&Lkb8tx+t!IKYq@scz1&VB2(jt?8`ddAbc7Rz_8%K(<$)U7@=4%Ul6tG_uFC0ZhGAR**Q<A%9)h}uT4pFcFkW+?RFH&=9B0C(6kPRxK8m64gUWZ{JEpjWpexWTs?6#aS-Nx%h;Bf<zhI8a~qJF7XdYmx~RGY}G_7%P<RCS<EO&QOx4VT2Wv`ll5-{aQ>GJyy+*(4VL4KPDcbuDv>^oC06N+*Qza0ETbaMBP8_yLq;&+nVSPlTun)1k^?I}n6Asc+)!TRm~LUhZ506o+XI#Y`4BMXw2pNDop$<e9R0ewI@ZJr!0~_r3(vhKceLZuxWl$&q+^D+7?Ke}=ont|LG4NbIJC9+9r~^e3cfiB{Gxoa9GaA~rx}uLHwwE-@D^x<6zNTc2dHCf8k|s7Y856%5fLw<5R5K)U+WBN=5=A>qf`s^_?rkOG&l4Lrv407*m2?4)$<Ub5gUFeUKR$>-toY-(}OKb2uNgFP!eaJf0niW<eaKW+9|gf`7W!?o4nhcXVIIho$Rys5?&&4>9QSCHTPM<>A0)q~~k3J%-38idmrvT;wjsQ7kys3(#)A4cut)fNcib)j7M$*P2};vF8iIcGi@TH_B$_If<hrC10bZTqBGj955)fTscNoi%Hxy+~B>6OxwZ>DMBN=go6YJrE!54<2BS=u1kXWE508U@IkLkJ(TdQw03_xq%}vyh7gF7wWd19yCWLosOcaL#2^Ex+Hbhm5;6&bB4Aq>HDp}|B#`{%u}9wg1S-=COhz*z%6MVJ`@|TL0KSPvnwZQH}kWa|6eZ8NGmF|nu)-^O`HVR{d&aB(c16(`Rlv%gQ?e>T(?rVZCyeciA(IOlKwS)P%Lrqf}{cg(E$+fM)SI6h*GVArV<nN^YoGv4X_D~#-){iMn-9dG*(qNn@zQGu35V=^o0a@p-FW4P1|=H51}S3&Qz<&G*GeC3i;ZlV0wN?;_{803E#~~kebNm68g8iEEi1nsz$m-a>awEHq`f9d+xUr;0UE#ksOO8(;XtX!gky&)@em?!sXZ@>->ZkBtEzLG5TE5;P*;^2K4sD)84~KY;V$!FBN<oo}p5=z7Pv^pSovD{LR4ASlNBe1_w0(zzDm<6~|XitK)Ta+b_l7;lO6X0=kK0T7=XNR+-&K9BYKxL;`oShU9UHY@rAIL@83PbIbLfU2R|)UBrs{S{TVJT{w^-Ny2A}#um={hw3OroMji`1%eKRq43+^n7Xz09R2bgdYe9J8byVLWVV1oN|icy9Xd7=pNqT!umZs+KZUHVL!=V^fLx#RtjtV{3zkzCjWi`*zE>?<|2cl{NZ4P2i009Z@fm$={v9T1SSKWxG)qC1ASr<Drocw;bg%^(%vXdFxBFmG_*HISriv3ThfFbQ9!TqMAW68mhi%!oo=uSKIs-lV%I61P%YYl-3Q`hW6sg}NVdM*xNzsvJf5^D6wB^n7OBb@k_J<U1ecStta8`8EU(&OaH1ts>y?+8j-2yW0Px+p2k+?X=27b;(B5`F_@lb_j-;C$*R|2WVy?>N>nYkJNgR)Fh>`9K{{E$V%f9LICQ-El>wzMTFJm(2}93l8-uVNQQ5Lt^6!skx}q0bQ?>0<Ao-9fed@V3?rpQEU9D;#&=2{~m4=b71Bs|xF_tWs`unG(zkj&1w9pb>56LKkW8J*FBle%af_E+*O6g>w|KTc~56IWNi45|J^(CL*TO{;;cR+mpX&dHtPvbgY4OiEvU>J_j$+kbf{OladE+dYG@+z7z;{^_F~<mwvM}sQZlb-;(1jya4X&1A;%Wd^0WyAo&@B1lGf18y*qT%T1+1fx2i*O^*;HK6&6-4;z_h$^XVz@z<#GDnrTR05M?o`wyqS_IS3Vnc8*Qv?kUxa<j8?R7#*8#*U95-+qFsbtL;X4oMONQ>KK_y?5V`)yqx#t;$Y8?i-I<j&9w-0$%W5lLZ<i(4<OB*ttZ17Eqp5FQL*er@Bs9?%IFFWBoc*zB0oicm)p1FGI^DaEVWyC9#>!&ih%X)#ixdod(ncH`io{FQw}gg%5=StJ6V!0!yF~<AR{bTkg8;UP})?Hqiieq^)fiUsn9K!^6H>0#K5gyS0H!#ttKdB@w6vUBC~`Us6NjpeQ4X#>FttISiyWVsX%Ll_;#eX$(HslN%!Qqyk)jBA68TPz|;56IQ7ok=O}%Nva>DKE^lu4SGx+>>&t@u;l}4(Lz~LjSP}=Wz=s^)&@oX_^M;#;a06a*X;#z)+OXr2NH{M5qcL3+SmNk1{G!{c8MNLT@47D*UMqbt10pYZfKn$MRWl7IfRO<jcGLT<?l8l)r_gnBC(w@BH3oFlCqNg=4fC7VTk1)97~Ua=I;8%u-|J82Zq6HES>=HtviK!3OX(ANWiPZn4>3?*PH!`j&rv+F|1`PTp%>b06sjaz3FK~lQ!O5YeZxOc;u9gKZ~#mZe3%1OOc%eBQ%V8IDn(LNyY}4pR#<k!lZavVW^#uleoKw2-5D*e%&LQ;tHJ4Cnx-$hEBKud!?K)mX(^2aidflhv+6Hv86OKbm#=IDmQhMiV4iU`zKpQ+pEn$MnkcUk(9}6Vs=<5sBAChd7ojWv-T7;8}_`wzvVLG;Sq|9I3B>Oo-zyV?$>gS8w`~si-oG4-0_W^Dq^|r$3f}rq6tTyf15?Gv`%`e{D3Sy&v6-JGvY>(K0zm6CRTxB?X6k5PGty#eByvsxSRh89WdH7a#NQ*OSyJ$LcO~9_$RM<LjEs>!_lbGbfjoIE&ou&(`)qg(f!qBFX~x9N<-D~!aRgm*I|LsI62>s*MR2AP<D~PihzBt7L8FW7PD@n6_J^QaeC!oOVre<Ie~T9uH642CtoP(KBhO8Bf_~z-Wi_HUW&+;ZU}k0=6cr4X~vrgofQBf%mh)Tw(IN8*<g<|DJgT$9eTMU16E1QePnZWu?mwaCVnh$uBnO6iCMg4%T$r*noelzEaqmtta>J5FodspM6T!Vm}iRg=E`RSO0t*AxXPAClVD@uH}dUrCc1@tvX@C8AU@}RYbDwZ;XF;$B~CGc6{FYD)h`?#O5+TMbLoqlrWoW*ROvJOeV~y3uFiH1Vc$$&FiW)*A45Sk;VJSEgN7jV%7xm_XMzF(ux1#`@IY5!-BvnZl52Udc`5jQ05+RaVc&{%7Eyg<ya!xH9epD8$Ge5=2s_7OUB#o)4`iMVSJ)8i7856mBu>sP6sycAK2@x|;6;;(Xo$sao_qXbu_7gfd?J5}R^vyE$4@kJUrXL6=tWo8f^#%ks{hV%PI(xMehY8}x0vdrRnm3F{sGOTec1UmKw<2&6gr|h116^-Qm{yI?O(brt!ea2siBg6Ck1_oDOA=zFu1S1E7Pgg0Z;#iCl)84`yjQ1UsYCYKPZ{wc~KG>(ko=?^KCwT9X<G8ccoU}9qJ?G9K=7JJKow_P${0>j57vdU<8ly8ow3@|5mE2nd3}g;3zF5e;XW<Woxu7T}M^8m!=VBW6d7X3hLJx{Ujoj=S?lF3G;-~A_^L@v2MQmGiL(N^p!ifJKFj|GWHpd8TZR2pY63Q!e~ND8Fn#kd8LC1qhMf_UhR~fZqYH2_dt%^tcn#s7o8ZL5V*$yayHr2R(e5o2k3ElhZDwZ3MPMd=VkVpsVsKKo`~-I5y2f+go)?V&Q+gVip&ce+AUYgi;=@F{*VLfRqcoCZPs_)?4GD=1E*p2=nSeZ9N4<)^YdG^ZAqm_3fONWuf_$r9**9?AbdH_{YI1izDM*B+Nj(n{&&dgW=TUP9Lyavxm7O2-JiD1Bc*+JDZ)Zyx;Oh$tIGxUw}!x6W{y6%U{b%n49W^nCCJusA{!%%C!CTvkpxA;+_WO};_;pQk8CT@vIr5VuM~sP$WM!0z<kZO1f(NR9opuNwaO0?oU{Fdu_V|B8V%M)D;8U|VgI_LtTo!JrqmZi7fe&Uelms86&$fPV$KWp1t18;EGEoiL3nfIlCxZ!7!JOp_oZ{dtHVMXr0nR~TLX>UQX)s2yVvqBEllyUpFGZ4Woq5`%JKxO(I_?`S&yBNfG*MT16BA$hP2N(Vo|vk9x*#e`AXS~Vv)yv|Bq+k)H3&iwYjrDO}9arvOP+C@Qos^>Ya_NtbCMku+yB6DI35_I1hwyKr-q(Q6I%zE&1vCarl1hdS!AnwQ1UP-GoP7kOHv$Q@vkqcM);IkBn7SFeyBLDa=dce~&0CVk5KiSBR?aMsaJOnN&<4zC)N)_3Cmcc#Y6e9C)O=k4_d3d;kEO3Pmo`7|5J`Mp^w|3_#0WEE|EXppIbwWTpM7pw(&4#hM6@vlwqr2|ET;Q3$8xC*@eVNqOZkPQx>`f{BI$5;!aVl8~&~ShFeL911&SDtN51;$t-|u{DsCl<nq`SkIXC--#|C@^7K_XP;4RWc!1y{mEDV+TEJV@EeZa_Qcy?qm8YN6Tx616zVX)B*FSMs~_Itn~Oz(`YcD$l{Ao_Fvl8!_I^MEu(`$29@o`cToN<x%<Gc!<I(oTlRK|Jn7sXX@?73;G5wwPiO@8CTL1ZBRz!Y!v$Sj*kOAa+`{V@9AoNcT@bG}JZhlWdkl$>CcAn?swOd-j4qYMR3Jcd5wCqo}BPCytQkbv-5N@cS=k)!jb|i0l1^oh7<I*1q_2TjX1emJgp{tTPI=P_thavpcd7Z<(qNvp-ZmmAfIM&Kzf<`@ci*r)hQcBuhWM_h<_By-$1!%@xGPXJ2jI(=-bl4JD8i(b5&j1D%i|xsmx+#I_!<vAGo7s+O5F-y-ywuRTknd|c(a^zFXd~kO>P45uVam~u`$+741Rn1IY&2bMy1x~rVr#$1CPKme_x6mmTYIx4A<-P!U|aT*r4X-7BB-Q4Q`W~es!CO$q&7Bl6=)178{JCwr`+~i?#-o8V6Poww-yi;Br_0%%5lIU=TrfKbYdJ2?~}bQd_?&iDHrMVI~^XsY!-`lB2e(@-yJ!*7zPJrwP1aNcw;dvIyA0VO?5=L6C4u3XlN@Q!h4?5KCQXMY@?CxBx+^D6v~fWsB@fCE;kdF2g%^+MLV?P&mR{$FR!3j^@u@ee$;~_Mb^HhK6EX8>3m*Q@hcSe<16C5=NyN~8f$Y718Zz*<48gxn6WSXfOsfFmmJ5%kzp1+4YDrTNwDJh#DClLP_H^`4Y6RxE$eVmLNjLURZd5mSz51sB;@)u3rBrD0t=*-t!+PtLI<d+pi{OlqRVHr2Y@1h<boT}UkXrsHjEhX*Nr`3bIfS?l<jr?S`~=o3uVKXeq<l2^g2ZQXOK~dJU1w4d#h1XxUs?3Y@*1|_yQgK4(fYMsTp>2X<69p(Oz#r&a1>4aqPB3c457VLWcs+21nL9*x`~#pauouT_EYoK`6Y{5>hHm0i@mb@04_Kx6iqbY7KtjN0{jI(Pe_q0bGsSWf~k{4cS&z-PjUW2c{uORq6|(i6dw0OItL7+Uve<fx(c)pWFff167wg|JVG=Z~3jv5|)wBGIC#(G;20ViKX*qCu;bEo7Om%N~<F;%6=U92`B9?9ZT*sjTrI_Lp?gH8?l8IRbiW#$O5M*bkmQY=@R(13q^(j4$a%pgyS0Rw02<aq-~UB?UBB<!=M9vo}S|VT5WBnoJ!4HHBP(;Xsr`GrIv@V!~tMhpc`QkreJ~u)GB-qB=xu~;@baDcV%l9<R=xc5GAyl#kr)rp+Y6*?sowi7<yHQ&K33*lXqSkoA3$^RSZ&>vgGs8G1_CKMDHk`)(5nOg(aXheOcMURx_;by9X&$<83NjxJwFlAHX6!Nzk_NIp!-{y^=xU*`>t2c{u-^*G325Dt-9qRAI^3ti&MKv}b4~4DXQJmODj2@J8hG!uaCH!E|yI*e<BWrD3LBz`pWmDO7^ZIxvp#@plC@*4$Kj)uUkqo6lM2;XgpP7kR-2dTo*lHh1_*lv;pqF~=_|X#USb_MVph41h?Bu4y)PZH0rtk9Bn6hQ!dO19yo-Ur9KNVb}x0SHI73LS2!qe9g}AKBF%IW)kLLHCQ5@svgV~$nuL;cb?=Q$*c=5ojMef>uI(<T)^KVOLdU*ZH!pSuuA&(mE&I3_m?M2X+`ZYbBB|AI0*Jvp9hqI4v+0rI*Dst%?ahJN!gnf>=ZA}bxFzkl(E@Nhqt6@Vmk@RHHL{8C+gy7`ijlU2_b+c$&L*}_cU+G1eo{wV28mBk2<_|Z06ep(iEvRA}-8+PH=gOO_y=t_91&`?j75yRwKb?O-L|EwGnbIYTSlU3pSGq^h#}WuF^t{4^d4@s3+G0-C?@vpZ+7&>f4BG(aOwlgCH)eHgotxB}P?D(~6bpQ5$RWQxby#*WC^<u#899YbawEsxdZV`tqTc(fwU8%mF+!*Po=8CVFqp{U8`YFS_LBG`@kbfWPg0eymmZZ%OFkgkJ;s?{c9W@=_Ba*qi1jnldzOuThO{XXojN=`Za&2<w__BS9l1Od^+G#@gWp``43`ZpXW5yZSD(j1MgG;I0Hg1_lA>X{D|V@xE58v#Oj<P>Awo03DJYJ{eXE&qtjUk1CDv{GeBdFf>T14<**XL^0Tgy~pt{<QLPpJHJ@NkB>7AiJOS-kfER`LItY~JFhlDFLW@XVie}*e^)e&4T;QY-4KxfnPmwSpEkAu3Hp&RHT6$ZD-7<Dy57+2nQSerC!$hn&cu+Dh&DGJ8MnBzj`H{J>grKx1^hJYYaZuK@@aK9%Cvf&?A|10_!q9`uprqeCUeA*!WPF<`L9CSiJMe@=wK0&%xV|PR3qbn2QR)y+kPg&>pVvhqZ64WvNIS2NX_CkcJ{tw{RC=nb_$le)IMf9wt<b<hubfpErh|bV#5Ko->N=HT&!sk_Id!jt2S#cQ?N`?yj4?EP@`Tc{)vg+5OlSc8o=I$us)C~Ac*FKsY|?*Yeh{l<eGgSZ^6eLX+VW_772?=wvBb3Iz-D)(y@E(0__e4ySr6Z%)AA#KpE?q+bMJ+y0k4|YET9XCbtRBDXTDm<~ANJS3xYNwfb>;XF}nShjydrhzSjkC*8FUuo^a6lc(Qhc7i~)LYdt}U;91iS3+kOmvlU5uu-N^v6_?Pm*SR9F&Z8YVjK}C-m-)&M2Y07rVmhaHdh3u=Vj-!Lc~}vlr+5MnUKif?9&dF;{soqB)deN#;12?819s8YLgXmVS%<(akkzT#9H18THDV-=usO^So2qp&Mqt*#{?|a6_0$lS!>{=S4|BF6rG@1I!uGk1?jw!!#4vOrMz_w8N8t{BxBR)bd>$v2KDJCdU-JN0{T8JY3;@cA@e(594#<RUwdI*V}YNfOTappWArz=BnEck2Sq&$0-mqpVn$!EW$wTHp;f$3bO*x5KC)NjV<-`JBIs@Q(C+flLW;oXLLZrc2toF_EA07wm-H#)fsNtPEN04!rhB9<ZLu?ive#6&4Jo*;Wg*W8GI|zWw{jGzq%8xpnZz)@zSUuSIZPkM51;r}reNA%IxaYtrU8X;9`Ph!6lNShBaEZ77TOKFE}8EjbA8et@W05<)y3lu_;K}YC`4h-<r#_-o3Hy%Wz|NdSUdD1-gQ-}+fFV=Q6K-}CUEe71%<=i@OJBGCxA_Pi28Cb8-w&F*!VD>#(H9hU*X~rrw_M9Y|Rh=BVKk@FKF75%#bs1H`n{A`Uz6)a?EO~j->*?4ign$1a8&l|A~VrxYR-vVfJi>hS+e~Ou@T&pkf)Y)9`6jWl3v!5{4hw8TP)5;9@omM%WbxqW_KEERfwM6>P}DS;u7M3|A2CrinSsD7z}v*qF1n2Y;*O7cXvl9O9I&f5|<Ee&aGbz0$^p*8CyxB%=Y#4xD#I4x)c!LOJoQqq@|yLrg{K)z&8JM{&LkR)IsgyU2f*JWk&F7vKh<&jN`Hj>4;3>D@Q6ginj*<(*{9stxq}5Zkvyau=oZZLI2yfxdXhoCXI#An(8Z7egxAGd^_Ph7KnZ5L}xXs8hvnU|^Fwa;7ZvkG~6l4i72&^<bojGM}u-8z0yJwP5wbCHK7Ojxm=y>Ot`W6=vzy<}Fu1P%7OomoeL4!a1&~Aep=YX8A2{hl851H9DP~9f$aqckG#PQF^>)D3sk$SrGjP^9nFC{Gk-Qi*rd6VP$z48cx@%7Z2<`3CPi|c86p^<#=qJvC66E5^fqhJ1t<w(j4O$8f)mE3e76Nem0HTZ#ue9_*T&oR{$;<q}C7+O9UT?3D&ZlqK9RD+AO=rskN^FSy8OjmS93%?ukbYDsaM80Iq{fdVNcFE>8oQ1>-uicqzhTNxb@d`MNeeY<5vqZulabkYr3{k0JsRnL9$QF!EV3>6GA$&g)^f<F!6u>_aD-D@@WI1M)UI5Y<!IGo*xkRtJyiS_t9Adm#1(Ytbzk5xsce{b#f_>HJ_}z=u$+u>~a~O|#bhC{DP$m7d~<^5}r2(bxkY1<S5d0V<S>%~&PU-5^}nU{f~5h5izY9P=Wh+4m7TaUsSavoS1cWYihP&aH?z_zEel@1xSy#GPouRamMeQmYb+>BtjtMu#JKP{{LJ_V?6gJmZqA1(IU`8O`a+c5r}t7tJkFXBwzZUvQJBUk@rXOvT1d1<MH7JL!{CCIcBPp^u61hV<h0hIj5}yn>9`e>5N*&Th;I`P2M7!;*1eRD(ZgU?LJ6no9(&N{ufLE%k?3bT2JDll*HxiD5a>Q{&H=^Upsi6p856J<B;P3w-=_>psmL9g-Cd@8s9-!4|P0$VFP;9tli>1nSemqaH**11QANT)st0aJ?7`r92w~2kRhZ`(FQFb9YDP<uvH@D<!#FeB|#aN4CQ3Q5Lh%WpT7*wD;|-B)^p!`9C8y<f)m`Xu=bQu>2pD%7xc+e<3RoTY^xY?PpG)rouJwI`Hlmol>xp=RbU;X9{=4|BUwi2gj&Q$`c@yqgZNK3H$D)GRQ(ySj~4qECdPuk;{DHrhl+dut69Tv(oBiV2eaHb?jYBi<jrlwxC{!q^IQ=;4>3-mO5rx_8M-B^y@Q!+D!}wz2g>zg*Ut|C;Gg5VK{sM`wa$#26MWE!=L47?<+OLIDNj8%J*+q`DeUit{%7fG?e{EZcJpz$K<YOZcn-#@|-5P=3h~D813byew=(P&sYAbHdkRB#@07nr3#w2jyRT@t^Qs@za*?&#V<%iJ~OtsG_Re2fcq~hqDTrLJOI1)D>j1u&hv|$n(}>}MC$TG^Nufze;l)dE*X}sz(UMWZ24Er252}zx(}aJ_u2Z*S_3L-^Fk_T9x4w7uYyM@^GZN$`E~RR&{8sUwssD(1sU;!WhtUhc^v$oAK`OAt{wJX*K0(Vqn*`}WCH@{7wHzJ-N~mdrFZ}X6@+(cX}+l=RtN{L4#d4-(5Sa1vJc9|sM1FD#>|+Ay7E@S!FT(HOQGBe+=@$FLacCPJIi%vk>|8t8eby6$k3*w%etl)H+E{_hQQPhipp5kBX0dDon6$y^)e=3T#m{Y2L||)rh&iBg|<hOdK<D@*(XK0ni0PrzFiC_xJT1v!k5BLspNZzX?J3s1lB1{raj#9>BOVzmjNt$(2&wod#v`lT0=^L)+fESZJW6((l7EuUl+L_71?0?&Zz4eZoSH<dpL7uhT6#2$MIjjiV26X^pd+*%hLS)0JvXR_C2~08bkObu-#5pe7;J-xQ}S$x=s!CQN>6nCzgA5-&g0xZBCn`=p9ku752JqSKt4W{0#+MioVKa$Q6U96-}b|;y)+}T}*Wb^QW$2?vS)KaJm9_4;D3xn(=|Z|015xtp62%q%wZ$1*X(F%xmgYDqL$yFeRJ|(7n_TmBAt$aJ!{Ay#EZ5w_e!EflHvY@qKFlyP`YzZ*OU#dfh^2{I%#Cg@U>WRPdWN;?_FbQ$OVx;mica-IRej>^7j(Z<06CVnku&D+xqo;KSV{D@rL;%t!XybY!_@P6-L#;wxz#TvKjD5I0h%29=sym0i7+soj4p<q%$2$nz!@8ZgH&!8PX?=GQ&wXia{MQAU?29p_j}WO_ZR=9DVMhMl}EEf|a9W!OII=W?&^x-?vW{k}oi_u*Jir$pp@Zp_7A0ca>7FPb!7Xhj)0sd{kc&gy>ln1fFMdv8EeMG!?5`^JL2jg^2*QvX0GR}|}9lmC~rGmE0N*l1BRjzb6JK-Idfg9pe5T!+qKpY7z)u>&OkkjN=2bly6wBN=~eoaRjVH`y}PSQr;{(hh9X$(^XeN6?t=*bC)8J(miJNRf}Y^`<o=Rast6Mvc625O(G^@IxCQTMnhqa+}*So8o%nuSIoSP#`i9TfO;CL|H%kHE`Jt4RLtPInpnJt2N-dOkwxsF*~BCKROkCX?ff^=0;{HwXDC)0RQ-x%(e{4#9$T0bT_u6%RT@k@F%~`u$6E>bT7O~GQqk7Z_+rOG7wCE4`I2am|Vi)Vw1<_lFN3K407x2?x_Sfp`X+(NvfZS4P6uvIk+JY=1P)^WG<|mzhCIQp->ht=8`0sl9V#sLni!!>oM0mdCNl>X5?(xWKo>{xxrYCmG9{}h{{cKRp6PAaqDnv`?3i0-gd&j4%9C3MiHz$u(XTcTkc)YUtjw8V9$L{RdlzAK`wpyO+Q-d=#V^8?Z7Cq_mE#}yxhCnrPZx_`96`_pJ8tWNqYU*$4Sad$<Zy?gEM!MMFx|R7$MOqG`qeJQS9H>Z@08>mZwQZ#^QhMciRJ*%@Qy4VO9a}StAihv@+)kfqpaNF19&`IM3ZwHIFbJn=3L;{94B#STXm5wxa*cy?NuOUX9Al8i>)ncmNT-#)*i7?QtfjCz~OWKv*roSWUc*pjm~Xbi5sG#Fi2fyl10jP=+ZsB7Itn=!f6?`Vn|5i}+-aOOUZzwH8BMN9y+>X35xSM(D>37sySfpmxEK_@wf{A~fVj6gmakckhhM4KiWCK8s_`IZmER$^;fDB&}Q~&tP82(CAumwLF8;m-DEWz71#QTYk;y;<4<7(M6p~@&`VZ3%H~S-tUi7v@Ov@k@bAI7KJ6tp^P^okspn1qOVhQ^52GA^RkxC?BKS|M^+2q6(kU+@Hf0s#7}sJ`7EP{c4&j-LypPxeW>1MqLC=C^VnpV6JF~*6ppne7FvZ*tYiE^O<4%GMej+}M`yR!?G)?RhyU%M$IHZg35EI4`*rh{(_l=aKr5SfFnk4q+07y#`^r<_D1zX9(~Pn5L?nO=Dpd0;RR!Tr&3K2eg;XPKRxz9lsS6FU_YXouhFtr+f4{RXwV{|WVQ_SwUjwUUAe~`|tH+`wlPJ300~iNu1&Tsr-|sU7@{bR?nSuBwOE9g3K!m<mC)6mfHl?Ge1H#^a-ST%!2Li-J!K1?$(9+UdnV)K-Yv~sQE~W>lfx(B<uk^rMzt!PJ5+cqYiH=!T=|Srg=sZG%6E(J&I|cdy|EgXmhw2o@iMFKf^2Vz-qf|bH{y}MBNRj@}uIBzS+EN9t>}Y^vP_*Zt?|$55Zk^WcaCZUx6PX!=5<YFF#HlylTA5GHhIS=1dFNR5W9|1kp?{;sZ*=@<<Yj~H3~fLBK9UbilAy3o_Qj0RW0Me#*<0E0M-?F6mMb9(tPu1Q2s$1D0b+U`e^L2O8D%Td%uViKVCG>LHE}Eqy1Dl^Ecg9Ib9YD_w9q2&+V<8;1G-XW&UZx%9Hif0Bb=CU-K!@7FX#MVdqScJ4+{n%Y>aaXy{Xx<Ttt}8bokh{`bruxU*IOe6M?!;eMq<f4Z74B)5!Y>EPk}f?$|?p1|V%6s)FTPutV<G(wNpYH;@TCnym;_l~J@W)tOig8#8B3rWC5TxcAhuMWodbLvSgxpDV1xh%Cu!`OtA7AmC4fc-Tgze($2MtV#3$w)oi`67Bl!C;F=d10e<{BlPGU(sOF4VrmLg{5!n<y0TB_96q`heiqNVPjYKO0q`v0Ci=}^tEGb*OmRxFbxqXc;N8C$?g9lXr$4(a5k7&`iZVm8PF3jhW22$V4fUbyyupK5P2o2vTyu_4uCf|+AFRH^^*jQ8-|Al&{oTL>{mI-|I_`#|qiT>m230n_SfSXHAUITQQGF=5c|$fZ88+;?Nw1rFmIS&{x5pR85h5Rkh)pA&ph~A1NTeIK8n@Cw?W@en-j_Lr<!JYvNK7c(4L4N<X%1Nz^Xx>u(gh<+^3es5i`yc4;>p!$@55Gn-Bj+dVB=rD(pP7kgze>O8|3JC(?2W$%W3h4J+7q2Q>k%cjz9iSukBEb?JS8*x|8cqP(fL4Jy8})YXW?U4O2--Md){sdrNDWj-^URf6KTsFZ#D7Ib$|bVo_rWUd%>8z7+DAU=m#48iyQI5Ipum*nHA9Z|pv(8Z&xbqNf^-R%3Ndn{I+WG^w{hTfwXlH3#Q)x;pz+R2$P1bF{j0UAx$e3hdvJAAR})g8~^=OefKqLLKbDXX)h!bmu#6kd>d}$PG{8rCeWbAaak3B+g{YMD|yv^>74L3H0YswR8fXvL^Q4MBd*sr!Pr*4#mh+#k2Zph}W9TNl12_h3Z*d783P$Fnn!{eH~|+6jBUje@Y$yZ{{*^4<Kz`DVK)mnRI0&c7VO2UWn*-mrDI#LyI@&n#)U}Ac#J<-Y2sN(O!lL#+s>>=zqCIAHe56vA4FsyxKOL5~%BBg*fFmW*I0iKeC6N^L`E_>c3`~>mfAJvWF_TO*!cAE<;-(8SGhst|(b;D6P;08JR8AMqAmwrTFnCEs5auPw4x800W^l5295*3x!cnx&v~XldYpVH_d$Gij^ZWUI@c8srhlp<GrM?ccCA|y*n@~J3}y-iN(&vN&=^+d(Os5zw*r8PjN4u6pP2a?-o23>M`}%+Ms_j*HDlh4tpJ{-TW3~I2V3LF`0y%w64(NxElLSpk5!qAkYRxDINF8`GjsFu37=`7ZBGn(5%%1%$y?Vd`y_Tmr2SD_K?ahkP$8p-Uue`+3lt>lD*`3zPSs%t9w*@bL5dh<E{{nI|rc4YP-6}Wb0Z@7HFF21OXIQX4d1;fPL_j2tn*N<aho0DHw7Ntd7T<<I_pB^H&MY{KfkM*2?4Pdf=zj57U9;49w7}Z?(_>aF&q_{#X`5ESB8Jw-Q9@o~)5d%ON5|ikoN9N!y1GH7G#pBud1>n?x3=aE3-&XnNv{Xi4#yttUxC<FNyluKz}|WL7sI7{QWW@uQZ6N+Is!B^D9@|E+UO)Zi07aY}c3OG38g+45&8G3%KnL1iV~+1h1!d(?>03&vm4I<3oL+6B~d_=XVVToHCbxzx_(te9ypoZfkpZ<7d1-u*2)H(id0Q)qv_CooiH<Cb3NWMdv>u(c5K&f;tCmek02A~O5W5j;6^M*F$Z99R@D{ok?RfS1V}5fVes?o-0{7vOEZ{V-<rfeR!q`LYTP@y*zxf#_w#89oQCS*xOcBoalOV~14oru+)LdRe5>5L*Ucnc^Ls@XCO`<dK+hutfT(>Nq!3YP(5fP&gCZF&7|DSok-j?Hn_54=wB?JQ7BxS;w(tfM|YpMzg#7FStA1Umw1)`-h3g8HbTN*0x_-+m$(pJdPIR*?44l9y$Ej@^~_wXUMa!Mc<4G>i?XhgMleEb;)V?<j*Vcx9+bz42&m1fs07>kvW{@vk%~5La?edHoC!J`-G5q=x2!QPHl6q12>}`ye<8EDCXnBfbe5=z@AsH^CG!fi}LHa$JDDH?->0+v;gpz%Gi%vf4nOlJfxx^FBgS;E-8)0SCk8I&_+7OD|oVT_vOGSp&%~*Sv3LY%9FkG;-2EpsOcnRv5MHD(__Z2a*TS-;3j$7n*I#dxz3$UU;E;=su!<jJ2wb;;#y-CTb+P>P<~GU_t{;2306v(VIX41No6m6XZ^_j1Cih9Lo+%MZp5n6(&1@3LD)h34ig+5#z)Znlw@+edzACkn3yHTFp7Nw)$)2|X5B3K@LW~BXQ1*j(GO=Xa%x!7dN+vp7c1!#Y}ap!pD;BKaJj&mUxmfitXh@6c-KhHKBE*|i!5D;8MoE;3VxBvzBcg;Mkh|c9-2p~DHVImQC^L6tUyKe+Ro{MR5;s)5Ik)6#Y`f=pgApQC>>31g`O(~dcZ4Y24g4sBm?TMfK_n15Zl_$t$*AfjkV9X=mMrL!FQ@qeh}Zy-Rbi@#k2}yDoxf1!+VJ`wk0Y6T@bgWrNVPbqY+!4FJwSu6gwkHMNew{o3e$R-j_FY7rAtQ>bmT&wBjiW4%v;0n?;Sk+E+Ni+z-jy5i4eaew)Xb5a!8<I~UXI{c$b2-@|DYQH3r+?wxB)<cjY~>;5z$!+83BW<Y9UndtO4Q`Cq@2&q2-Syw6-U10<<oMqsRe;<^X#Bwd4<GsH!{c12H@k!U0zQM-F#5_{AsFJcGIsG$(_145IV)rE8mG^8h;pljLv{kEZN^*qxdk{~d>(J9DFKYN|Z+<Z)<U9(uo`ed24f?MNOO3^Ic#HlS{4A^8!dv!d9kh&4kVR3GjH+NX%vgTP5v3}W;_w-1Z}%)@@7uZaU4Y51?)gObd8Vz-DM9AGnhq$4A)Y4vbK1UEiI2qX-^lUd_fx#)*%v&SFE#UsP#O{}`S0B<8UFyQZ2&^J{M06;otnPtQ_|}((~}72ni{zDPex>}od;88)LKR$Q~c@SqCW(lFJWg_AVN_u+}B=)ggyC*95uN&%U-v@9b%M6KBAk?z}xhiSeRaF-({E=9}<ttW@>)8I5rA>!$iI&+hviM@THF6|Hl?7zox~Yral(osAa0_=S7);%3&u@f|k_q+Y(GwN$TR%DxX+Cz9vE$B7CntS6`wlnqpLN7}Jv#+$7O+KcDsSpsUIZlWG=7GzQ*vp+<BVShCn|OAum1WjdnNyS1`Oug0Xh;KZ5-4=i&HLzR>^3B(~|e<M#%8p}>Bg}s2YaVub^@3~Oe&qny67Tc*y%E!Ho5HlhDXEG5dbtee>1AT9V5osbV&^zysIU$OEHLU32n5f3zfQfSAk554xnx^^4@Jiz|))GK~c~wCaXkPq7k7_})`_`CXuPG0YYC!evRq~)uD5~a-jg(jMr3pg&X<_Hw0y4=LQiGB)*k>Ub7v3dFx)=p7;L%$pS}yx}U;t+LB;NEo4FG$)Fr>5BB|=DAnEiZCHDRoyK{j~m*m4(Azw4;VdSwT8lCx`}-w<3HIh(R??;PD7VG#btDhNqYfv`*@<1-7hI?jXzz_VOY`+CNTnwyfev1Swk<zmt|NBl{_P;3lgYzTajagddG`Op>rj%e_@@1@=Y`St34$C$5VtS407G1XZ=WXOG^j$=4I!W!Rf>(PQ9Mvon%gx1Cm2u6cvh|2Z$r&^%<-IXR<2kRgivKoI4`nOoFDkqQCCEN5i;>PCFs6$ICS?2NB$xULrg2l?<o)TvH0;ziK>(WxJOl?y_<B>6@wsp(m<i(Gu{2Zvdq+zexfa9a7dN))OI7dy_;)||DcH_Z5S_R_r8QYTB{m+#f%*y}^sWc*R=X+OnL>O1x|4zYTC0q_-ga3r?%-wCs9~c9`iJc}H#YSV=<gUq}HMvk`3jvY8=ws}tB^L?9=!u|mOV<ki%@04A%p)d*NsjKe2Azm390l`Ym}e!$#(76b=H<`vad8W_D8!}}gUdZ4OVw2w=lrdT)U=>0z!Qw@hQ>)B5NYo&!SRzBk)y!Q!&1^1-W^>Zbml5NhJ_P(pba2Kr|tR(MD8Lb4sokb-B}WWE+eBY=cGKu&<6wb4DuvTyv49K(bQdp!i_X30}W%-1v67{I`*>iNZ3T<O$1|cp^9=~K+i=cE1RTE7hy{0DFKdu$;`!)o5ag!Rtx<Ws07E*?$<@);`hb%!X@`}H;Ye#j2+Hr+{hBaOxlG1zz|5B3pTY4lcj$H$8-}0{DO&4-A531l}-(6f;lNf%5F`(oX<EJzmj<G9L60H#|vb3Zx&cQ_=uX}7d#lE?*sS_+uh^OD1sSgac;_sDcKde*FV_d@5O~~MDEB8b~V4KYzt1=&^-Yv8Fp<Q6X{uBPEa#jVzLT9ct(YKS!mG~6xjzD?r2)i)^MICW^t-(kR}IuZwT$u?h}Y7F#j{st|QqFeM<>MOx4>&DrZkf_73^I<5{Wix{rr^Is@6lFg?Qu0Q=v;@IpL3Xi?I?JycXG#OMFeVku_7zm*j<uV#%4Lg=0fjr}BRQBXl2-l!7vs?wg3iA^Z7&7ehH8)YE#%WguYxu66&)r_GrwQr~S6x{bU4i0EL4M14U4`}CSuG_h8NlU(q+cXnc-`}Y1pY|s*T4S^<KF@|#<Z{;b4~6^hHAuV8=8s&VVnb<+(BCb(zOUiz9z)hK7Gzx<^-XN&o+=&53pdpUW~5@lv`gz6-0Dsmc8h-9n|eJWfGhE(-V+PT7UqCTPVWwJ_QTc8(hBNOwy3-ljS&16aE$#I=AT@2{V-4?&AlNntXZ#O%wJ8C*VSbp^)pFJjf$h)KX}D2>fVEt#Xe&cw0{Rm_uG3CA4iWZzwDakQIMm`^Vs<$_le|144bM<QwEtytqM^rUPmBsyZ0vr2p@@>QF+ZDT<A!8&MU4HvMA9h6;;W)GZ;`}adBQ*joDv0UVj*!Z@I21=_bFbYnqA_ofIUGA!y~jr4l9jf9k^FVaL8tIRQBMiL!FuiPoH$(PId%Z^pH(-bsD2xQIq0h82uk$kHY3q00X#%5m+Z`b5e|1HaEnQFMhl70qmHG(C}_%Q+XZ_IO5_ImoLwOvxHaN)XKn^^>!)an=EKw5=ynEG397i5MATJVr`kgkoSVfLWab5<XZ(QMtbA_<6|TYo2rt>aFwZLhuh5%`1oULypr`8}nA9+<acUIR!5M0(ng>CHX>yA>?k$QInE~mb09A4l8DcVBxp0uFU2~Q=h^@;lvy)AV3V6o~{O|Q=3ng^_n?-vB)(uFtXmK9#mEr!WAs?xl&pG^v8=14cbE_<K_9^Nw_%~zwg^0&W+dNYXF(%BSK9m5qL<V12HDZX5!L|&-uYLB2<!op0#^`5j#8Plb~?@9PTNB)0rBLIBpv+=p(mG>DaGPbI(H+O}%9D^hMxuToZx}9>2q8*u-`}W13c%NR(SnvIAu51i0XO*~}K4wsl>xZMeaR`8>ad6n9`X*xWI33SKMYHx#}J!v^saMfBwBJT?~_tPUc4s<OYZMvx+5mmjJ@6FGN7SW=s2?6(X=K~{GSz(}jlIIhamXV;bL=*~K(mtC*wS_ORB;ZSV4%*id+>*o+Py@yr#>I-DS00$v$<aDg6x4V`=V5NhiGaC~X$G8sc&PAJPMaY*WV7K$d{#Dc5(PYv)C}o%>4``SG>5jHkiL%HJr~@O12lpqLuYgcbkF-8KKBlfPU1Ye+aWKIDj47UC_<-f4j<bA$<3tKtG#N7>@kL66kwBM1p0<QUYHQ}t%&CdZ@o)zfsGNYCE(N@Zy#DSsqO&}+U#p)zHK?2|7WUdTch`_R9N5oU6gI&e#8laAKRZ8_i-VyHXpeQ#=X0D2h<?{93+Ywm^)8T<%Ti!<Ok6Jjn!Ux?)s?b*(GV^1Ee#ww&V+%%N<$sBJF#7<luWNDRH3~=(=D#<ooJ&o9aSnwHVuCk)*rVsmT7j2tZvo8>@)Th9WW|^sAP=h1%hBhe7lMVWHa>Lf{H21n#<CNmko6g$}>TVWF{IxqhG1vghN+JF8!C}jj6H6Tqoq>)QeNv1Wvnj-@xdlgQv$m4vo!g=6a%NGK2(gTq~qo>B+Z>uu|;WxrxX8UOQLwUUG_9MAa{fbWHt^e6dl8gI7QW#rOI&qCE#D=ev!CPiY}gAvaU+ktT;l6_(I30ER~E?Y6<Bz0c~v+$beu^)qoP+8Z&<iJU~N0q?_6h1!z>mz?}v_H7124npN-TpFRpZ0GkA4Rcz|X3Nq_2Awjx>`iOHxts_tiyFfwU2W#>jLv<l&`=_PX^tDQK;au4jPTd&D94?Ub<2VwS#R`2Yi&7FfPH_*E9x|e&FO++baxC145b}6d|jHdV}ER6s)c2X={v3I;iTcSNh0UVzLS=MmJmuWodJfF=j%Sq{;9X8zUt2O;(QCcja$PTeW}iu3EJ&&I00!IUOspCG$8-4{RE$gj&7`)<kKX>MTQW9krScQ=CSH0;Qr=Rh5=)0wDL=M*?E&KUCz7KQ;NxQf6s)<mggD2aw+Rg#86Alv_NIZ&jB~xj-fF>7WBc8|5NJ&*S2D-(!IC0Ub-<!=GIgbphKxWy}JB%{_VoJ2Pw`jENPnRZLn#U+f+Ufq%C*>2)YQ%j0Wu`5~B-&zY1`@Q|WNWUfhTm_hk$oY~69<Ze-IEmf%Z-^816<tmw5H>aAI=eSTYn({%9lE6S@$;22LbQt4<Y4sBD{`>tW)bZ@E?P*F_#+DyP?KGmi8iR9=;>0$Xk4peW&21sq*GRKeoeh>$&YaVh8h_xE&T&2LQJxJqZYvueK^d&kH8@L|ku1K}DNyP+UjcBH-{g3e>DPhAvqAjhL!wy@hm(fl5)w#tK^qRlj-Ckt@Zp?$-E6b=V`nw@X%=&`+cS_(~hAh2&6N5C#T0XX5lX2ptd@y8)kF^L}%n#btnKuhfvD)ipK1Vx2=Q%oXO`m!kj!b-B-8TT@7;ih4e49=WF9i<V;%wMl<H$Vyy_cK{IUd8B_@HO4L>B*ok~jFVhoJ82!aKg39tQ~!*l8$2Pz~9R2BNZ`-H_du?vbiQVZ@LquvpSG2HD$l%=zm&!TD(PTYP8M7QYA;J>$paj(<p>Wkvu5`~8+Rt3MG6H@v7~7@9Cx3yn_87x)Gg_7LS!u>SR^$jb)p^Is=@bH3V1L&9dO9fo(A#L_jQg)HCr(evOFa3J{#=rQr!2(r8wE*2I8vRS*e8h0PbqpDe18F_Cf?e;!9;cU;tTZvFNE?Zee?`@PA_d6p#I4H0$p%h_i*f{^xq?_BjZYE1VN}nt%{H#~lp;k~s+n1A4oK=UH$(Q`QT*7tWwZ>au)!_gqnY@qU$aX!b0;G;3KBc%ISUI$QewcFm@YpYZw(iQV%8HOam~s_`05WnNdVu@x4g7i+j>reqn;55H8)u7k9JWhZCjhp!>53KfKff$~!yv0;on0NWIq>TW)nh0QA54~=S{rUS?y7H8z&Q6+s%~;BX<XV-b;QF9UQyR_F^gGd7G+BksTEZgEd0vBs&}CrC~ianL@ku!bwYz|B<rs1I_K>CG3|Fm;TU>Bht<u_%Gnxc2ZKC8kx@4V3>#gsWn$uE&Xk3Kh)=7yqb$s0mh^TmESKTpaC@yZEN6Cx^Q%Ackp|}c^-s#)v-!SN3sj+#P%PCp`XUl!QTO|T9`W@>cs>JC)|<p8Wgwf{NEVlWISB<UCCZ-re(x>^TfRu?idnveUwtsb2)|_^F!$zK%AeD;N>N90_>L8;nrb;)Xgzn)F7pKv2+f8Oz#c`$s`o^o<!t_~LGMGCjO3`R3V#!SRtAk(lREm9^!9VIx?Cl2-mnY)Hvyq))QY=~g#dyZV+-tz?;vOQ5MJO`5`0^3F`5}tNb1Hajs@8_ai0_sv%GB&404%BTlkz#9I@I>3Yg&h{(nR7cNXj*nbQrpE7GkUJ%Zl&-&(*SsQ{Bk5ZZiVK0)12ECzDQD)+zm^iG(ruO=XOGktFXEb<qe7!2|sjZ%HZjU7wWew1`plUGi@Fy&q9pOqnP(HH#%e|Ld?aOM3H@NjX6m?njy+9o{k3#Tv722b+83#}t%yhYP8&5oPJY`m7h2JG08EFI(Jv&nv9D0#dK7<H^UXpZD%&D&HH9Y88J0q)KX+`LhwF9M%+#~(!9e6SwjbJVDyon5K~#O;`gHvH~S(c2)iWcAOZ6+-m4t#=63LMdR?-PbFmxQX9>QS0rcSpVsT;%wm2bJ5Z&;-j$@@<Zs!noeG*E${oNJnqWv?U{s$z*Ec^8sGBTJO+6myRPvh1sBjv<PSDaeGir*jOOyLm6o<K)vu0{)Gq0EkpKHpKdKzSB+0Z>`-@4|SgL|r&V`0gAGo34em$sGEZU#aLNIFXt*Vg*>P-47vk3kMvq4-8xG3>9YL33!B34AWVA?CaL!fq*QY;7#h`A%dEcZX;ArA$pjn&0qj~DUOex5RS-~%#myw9vQQ;&IIn)kjcY*>3FgL^H$$K!r%dY!sQVDzwK?;*Wkh62%d5(P`=`Sf)%tgkXnx1vIc?M><VfS|{?rcU|Bw$n&v$8eNJxYxPLfPGLmM6qvAN2LGhE`rSN*RC#5RPApOE{n^&6+_-D&Y(qToxt&}LhR+`qvaS-1Sex9pZ|fF9`k{NO{wt-szIb34hnmmjrd(gnKeRtj@zD+gtFiN*SOz}Gp7^=FClRJamm~cSVds~#kd8*sfF$255|a!j)2zwd_@utr{t&1Rs;K)h@th|7Y|F=_AVn_uAG8;I7<Na0#hl1lNI=CDqPfVIR(1AQBvvE{8u@F&~&8caEgQXt_Y|^-FC~Tl^Fa)aCcU3VJVPm2d5dN`;v2ZBb6C@+HYT#iA_?SEt{8_5&1!j3z2hNCeugCregt&CV@4=3c8{q+uqY%88|7Pk-W~_7H4huSBgO1fT(@kQs<X9Xm9`}emOl<L-xWjFkz{0XDbmprjXhweGUBfLlGTYJOm}7IcmjXCHq;!+gOf&{tCwTEVt#48V!~Dg&6k`anXbzBWODggnYRuZ7jn<8p6OU{1)W0C;<Wf-YnFBhLxNK@sO8&?lR1Fcp~QNraMowzh~bz7O3v<5zqreJGJ_ZW`%`G)IZEQdY}dxqO~pG=&G{gfR?DTU;Z?Ccym9MvpyE(2zkp|cg3GlF_;}4G+WZ)b>FnX;LVdhLMd1)Ua#6^(TxcyoQ<mj+)oAH^jx*s(i*Xd@!_O2?$qv@P^g87!@;Q}*%s3!R7VO~y_*;v+nT4Lf-cp6u9XAO=zFsRxic_M8q>6250cMY{saMUW<#&*J>mq8q%xYvFK<h?U^T$ZB4iESTeqhOo;MTUx^7w@<o?d42&Nn8|9Lx27$tB=eU$$AocTAxb`d%s78NB%ogQ8J2M*)5e1Yr(i5OWu4=GKF!BQLESues=hl(FEVpOI9ZCdkSe77rY*>MK3SjukAPV@5QfyBL>BSp!)(FR%MJ^qhN^`S4Wxos=%^s-*2FeAq&$rGlK!vVBXBUjgyPy+ac*SPOtvtD=Bt(MrO??Z@XlW(}n*XhBEauT__75&&?`Gzc)3ni^JKiNX@T4?T7VGj&#nd&3H+N}J}IXw@c>y6{1LnIAZBxnj@E$*%MkkT{yd}GZT6DJAT0@<f;uoY_xBxTU3r^b4*0>TKsuq4x-_`^CYsJxc0ChJql=1vJzcU{kIpacP4)0;Xiu>sm@2X(EC^7+U3*SDh=<y>mW)#|tL_7wfjoyKXn?jCtH{_)0nj_k*?il%OHYzC+&{3{}h6%Y`%WtOZipBPPfZoI!bOpv5d-wA+5w<$4}hSsRdoWG^gArV|_+|?LnXj?7mN#qk>TAWl&;7NN!aIuVX`^IA^>(Eh_Osdy34O*9V&HxJE6C*T_^^F8kMYP^{DiigW19xt+i`c}c#IK;C#3g}yt!TgZzmY2DDh2yK3mR!MvXX67tvFFwcDH(PK2QhnT4RY#vETo^FA!(W7FGh@cb@BY##c@DWY7Zix`0hvJ%Y(6u36p0Kv)X&ELPv(Mzg3tgH)~JQ|`UbWz9UFV@uhHai%tCpw!%_?0?1dR(8<CkA>EEU9_F)H3Yjpm7)wU$gqQbb96XHa1d7@%zU0MlN=qSC$;>2yWw4(1Y>3&49b9vo5&b#U41YziDTTAnb#$^Iw>>ks%Q8&_)}n_GU@%9FhO67UDANFnO>z3apWbW?;Iwm_?f|!^#EG@y?M?4Lfj-nUt}uJW5xVIPt~lmwj)%2D06*d>uL70qK6pMr1DlcK)^g@TrqS(ty`>9z>e4-F|add<E`o~w>@fmR<*L~4U17%KYDHJft;FXH=F0p?T_AC0PN@w!e9gx1iPM&V9mOku^}LJEEO$5MtA!!KuSM3?Tg6749H*Ce~Xie9DkdaX0W<#;fI%FdeCMKG?xWpb&Jk<))#_#Gq|gxE=Va5jYV-cBy&{0)Fc!gVsw%q1c=^KCl*XAPct80aA?8|7Bhi8WM{M`fb6@f{%va`grt+Yxuasx#HaXl*Z!JAO=Mbaq2bNn;y?CX+1wW*p5YqRw>g};rmUmSHb;}faKr}2>Vp1QJbs3+KOmvBe|m@iVeKsv*Ct4HkJ71Nx;g7*tlkH(6#6*IBr+u2+ROgwsR+#p%#tO}Z1L(tLLb!aNF#h22L)4hEM+P)0UHjd4oI-|%H0x9QF=w@CRduuOH^cOHhChZayZNvsIpyFSCTpmpw6wDOkmHuUVdQfo!;8K<RZK0d^69Ut#hbkk<^7;_-_f-5nBZmL7**g_JRA3iZY+<CF3ha+yrKlsX1^Je@sb%z_lq1N0Qad8>Q(@d=}GcNdQ_?AbMKWIZlReM+OsLi7{#aS>Zk)n82mUX;dG`Iz*^Keqkjqj&n*um&CS2tTbY2e05Fj{=rg6S;V-&nb2NN3bq__!F;}*#tf&iQmN8Yw9_#UjTZq={W5BMyh`J_V|1ni5rm0kLAceVwU)4a9JPypQ1*SP2c5L3nM=mtF#86hP8hLrNtT*mt(<?S5MLiU7g5VIai|!WCmVB=!UF;q3aLsPD+Jc2=ol?9EKUHM>_We``8dqI<jN?qawDy@ZX$o0SpEs)a~_Ze5`18D9R_`JxL|C%fv4!}YH+8j81mv#e|Lf}j5aR?jkt3%dr~a*t^f28qBzK!7diOkQRh9?4(&7_FLyVxx6S_;TcM)ueQu-vVKb_VJ5$w8-37HO75rLeA2KS1&**XX42S{6S|mFWAhOk>)ilEdlotQfkX;bNc|TAX&9vi?%xEGkB#3?SE5~+>m&Y_SI1{o=?Xa4KH`%u*`;OzW_ktFdPVZFvbJ<;W-FY79mrZWN352iQ0@6LDrsEO64NhWzROXJ8J7jx|!UjV(?l0E51i0qQ^KW<t)#tiitd(UH)ItG=5b4XB0{gQQ9=Qg8e)qJzB4}9t_rs1-aAY%YHq-V{D)+X8Qn1B=Or;KtH&bYpLGz~fR3}QtUHKG0CD|q5D6i)$I3?e`sakvX-IF1q(8St2j?(Ljw=QPa-fX1^G1oU=9&FdJxGo+#5)qfX^Qd+9y6%B;%>Ot!(eAj&L6M5NGCK$9*&UHZqrC9zfHPW$c-vNQJv_1%{$^`m%8A)Sa>HbRQ+!8dA|KUPnWMC)L-F>#mR7cu?<L-dxhTh$_Z!Jrom5AiSn_-P%zrgTYG7jry%-fH%rnEV4a|7B(oJ{mNbZ-&bzFI05IQqiRKHX)th%(bC=8fviq!@jhVR`>9oehUAgQi%1?*2Pz=38|uwHoVXP``@u881+pDiP<4kF8s9iWK?E$ZFsCTPFyc$a|&wjNoO)i&8HVVh9>Bw)BTC~02#5v`9WzZ!}Q^QhHzC3mjTRBbWOoc!-yLvr5zoK(>2b%CyiGTL|kY(ba#r?#6aEE!3z&<H$md5p;hpOD(@!$wRwE{<izH5xiHPDQ}-!+!U+aKsA}&&{tsp42DsT4hgw%e9}z4bu)*4|+7R;5>JoL6rm-dNMD%c{e}Fbu1qIiXpx?%?-6p8mNDvflKn`)7R~6>pf1g=Rv<Q#Bi5}g7r@vesA*cr}S(5)~!G%I~TZ2Fh31ZrN<Y>0f2r?*=E14ohF8o(RGO{`wO{_$!T>OXON@NGtPG0FWU5mgev^x==K4?z&j}1NcFn)!*^h(d}oqlNqO4y*6F>uS}C_vfAz3=o2@S{4nNxVeudGgnYM;)4229ns9k5!2f>6+D++)?yKdD5l^b`=<p5D)UvX{x<XNZsM@1bZ*v=@2__VCK#TtNnnhh8AMl{Te>ChJj8m~l=N$r<Q*Q>_k?O)H=!%6N_<(s<|bldzyK{Fq8!(-a9^a2e3a4*<peC84i(CUORP@A!lA5M+Fe<!2xy4p2riZW|wH*z7wBg=-uN5vjm%em=yEtUxfVPp9peAl~mgnMfh#6Ej;X>|i*&icBG)7qx0)%{oCN0ea&R?<{E1a8Nxu_;{aBaB<ATemfSaBn}HDK&XKlByj^DpA++Be(AAz#JDgA!O}Bt~;A{<9laAmjS6E4nNhkKM*&l18O+3lIN8Wx4IKEph@6B7+(_+YnC6;n~v}Q1FQox4a|edb8cH-FqLh-*h1(~ZZ<L11P9(Lt$z-z-0OC0f}Ij%pc1*U7p%LyoIpCLHk&Fk|DR7gx$Yaq`pea9`dNo!=Zrz2j2Dvg-6I8?7HnFud?~SuJ-+_T?z&K<ZjD>LJ^P%j)QQY(pOzZIS0EmQi$h6Sa31v6|9aVmNC~=5JQr#>fU_YeBE^(GKO#<h?g{eiy>wisjajhHB1r)w!?Hcft!8c~THYdbWLC!r+e=pWy``FfNQdmqbV~I`&Yig546B@WvOe_46U<$48GZfz%hUD*^o?x3q)}a%F8o!n@}cG|Ff)La^U>M4W_CCD=oFk@6-7;$&;_vcoDtO24k<Cs+FetU`^DQ9Ks5~|1+z@XMdW@mcQ8DKi=HT70DQ1se#cVUFsFE7mX@)whP{F@>O7;JP^yhZLX?yKi6<d~tk&hh$;A5qnm=xwc_ywW1c|auPrcM1Y#BZ5yl;6FPfm4TcC=OBgXnw85HZvoh%~wj$uLF2+*ye(syu;Sz!z4I2=-8>5n3jCSr!ExwfSR3$o=g-o%|muYXkboi(y&?<bodh;qsU<?@1LPx6ZO7Cb2JQgeuz!QFw~sw7c?^J89eqxd7YYm$7b}<%Az7i>KKc5d*?5%tN$Fra}r={9!#jNh|lL(A<NK+&6Fk`XxhZrGKBv%rKFAm9`XYyVFkTJzl_|^BFCAoHAhoH%!wOZsUZ5`P9*?iX}L|{lQP6mN=M2;M=?xiDrE9py-mHoCuzH>hz5Hn$1rt7npQ$K#^O_i`<oGk^C0=vI>E4#y)owF|OYvnki*FIkO+N;lAIH7HCg_QwEx3kahc+6M?j<e;Yq<el_x<MJtGo+H;|ZIGCbjY;6&I){tr>VFL2OFnl_#8i{46f+Ii(gbLIAo3k7;13!L^;{;F2a&qJ!!YLSOi}v$2Sw@_*pfDV7^aGy^;h)2bd>Uj@59;OWX8S&*9=SX2b*hmX<yUHXoRKF6BsD>_vvzdrrIYhKl))X9U3SEjmP=n`^(he6aB5c(1)^#lZ`X3R-06bP@Md>F4-lcez}!?QW(pc#Or@)Aw0ypsd8H;|pv_t<fU@fX>^6is&&yM5j7>Q-exyU^6dWxNw0{@h@n*fv(}A1mhX*yr`@gPk%cR`&uTy1Pho`<xdh-u?sM`1~0ipdLpHwYd>(nhd(j{kTjp7Z!{v69ye39!hwAmnyPr=ZAYDqk;H`tE$6WW-D|Nc3qD~M1sj90t0A1;u7H!y!tp9iT`enl(MiFR?S*HjPRJPoN@CV~5|VL6vQIj687qZC72>ZbqeWXuCaU_NYel{-kK4KOC_<Cown;LibAM1lf8DSZyDV|r(Y6@aRUjZ#@NddnkP?`EgT|Fk2sjoPG!28TcXiSp6RvL&XAOv$(7&1NkDMTNHRn6W9YrUBGf?0;MHw;pNe=>cxm<b=s#I`*52CP08j=BmEwC}7Nxlho+`_kh<qD<-z+P_p|#gU+Z&jR}z1;m6;+IYv5|!UfV){SA_i8tINxhVxbp0ORJ@*zaL_fNJ>B(9<p6^T&0AxDhm+4Z|@@-}?oluu^O7g$Citi&m@urF>eORT(cVBxgV=ht89oW4waBu+=?`75d&mYQ`}%XrFCai74dfFMW1rS@drB);s3hrzkRxiu=~&yBrc^>V+cp!&3_Y*%RFFJt=2FM@WFn-(45j3uvP?(zcC$+7bQdp0gnCqGD<MeVlK9So`Dfg`hyIGRmM|ibn3XYgo&VP6UWr!(pNd!wI)toX$8I-6GpMIZxmxrF`?}?`&B!kXaW!^&?eijFw|1WieY2dr%152nLCqJXs&)&;Fp2#?IHx`#ogd4HUEno!muA0wE7~7mA4X0lEpqeyvr@y3VWCAJ(>rk$%v6K}_1_Gw(__(r-y>j4mmH-s;uQkZb?6+|W5~&la|SsVQ{b!>)<FUSl4nFCg<GA{Xu<dfBlZ9yIT;nEX3?QBZDQ5YmA3<n8}rX)BbBwcs+cE2^nw)70mA{(3*be8eZ@1?Ki0s8LDuSNEQQN~ugzF{>7EoO4+z1Bo}3MXy)s6U(sYUv*~l7!)*N3X?a<cPC0N{lR1rBAo;xU(5I^obhN%V%1_w%U;#PKevgqwc&71>h2fB-gjxYUqT$nb<is*YMsXQ>p79$$((74p;#2(<hH)~?{taZQ1PtIz5QyrOdv;;0|SG?#X8HS+9NAPm_$r`D#6UvL}oE{mqMeD1N;>kp(<RP8?L&93;AA<k^xG;4<*^gB3!{LfjH;ctzhE;Vl>5$H=RmrTE!<{02>+ZNoB|@tdOqNfvzTFI0LW0v5@xC(}MA#cIbR6XNfVgy_Dj7?Kv}ok|3)!D~b$&H%6j{J(MpH2hYsBu9}9@t%xVfe`ZX(9+I)t9J~zj;w1Z&oK4gd_>R0AmTx{8jc*F@9QW-2TMEo2mDl<#{Co0CzRv=fNKi->K+t73^Y4SoKODqHn@i_o6au-an0ofJK65%Jj{3FeF5oAJicDG!@m8&b7#~`>Qar+w!^58m<jvDwC`y`r?IddTg%RLN$o3#9xo{%ufJaX7f3?MVPVeVpSK>R4G1yz3wrfQLa_sH-K$PmT+RWD$77@1h0qrW_g=h1#o&BP^Y1I|g{J1N5A?)HsKvOCC%$BGDuU1jLMm6QRTSe92yIM=3E8of4;5RaUO<eqc0J!eJr^*SMsCPUmg4fNs_p|WQDuUZBsyy7%;qL0$|Bl-S_Bs~G9v#&paqROqG;U;9uL;RS0XHyD0D8vGuIN@oiq68LdKw!N2q-vPgwJXrnLkF)Qg6}$j3O7_QNXXLxB1F*XVbC9Wn>WrMg-)}$d4xi&iKh;>`%!`v{6S8P!VIX!mLAUQYjPu6nS~6Yk7H09^af&sGos<EGHQ6ZsX!<)>F>nN&Ugio10$A{3~^HSP2{9WoJvz#aa~<Y7IVl=;C`YR)g~jhqu+M6W?AQMCl_<Tu(gy2;b)EN9P(qvfVJe`J1y+Z~c8o(cE{O{P8opaNWK_S(|?Ww^0LA-R7y00ohsMP`c8cIc6^+Op5??5Z?{Fn#JqJPYfEn5sY!lxKlHq-!FgOnm(!=mrz@5dcoyMjFA775QlB;ahXu~lejyF#xe1&pzC?4P-<53n!@0p{3J=lO2?;5dR7i-2tvcu{+H_`wp^;l^{%CUhI_%T_#mA0OEhFbWQL&*cmfN?+KAdXqj9NdbCz6aYfIRi>Ul3Hy|u_KHEK7sy^J<&&nAG69uZ<S7ZmrL)-i5;>@i(I8YiF}T#BClH5$=ZenmS^)KB!?j@UpU;OdeghMh*y2((;pW%E@yz_qR*<cadxUs6Ofi}I&8Ox?7+zp4uEba?th^Hnj>HE+bl7M>c(Z_kDiBvKg7s}z4|XoBxE_L6oH3u5%)5Et9Qk91%z=au6c$uI{6*l5Aag^cJk55kQ$gD+?WM$>tP%wHbg`0~L>;?O+CbebxKeu%8Yh<u^aUO&k0AlGQq?u`v)Y9SD+%^7sdhop1v;=C&TEKz9ZO}a`$ekm)6%lE=oNX+1%47o!so%%CBm7g&~44G#2GwU6#d(J`)Mzb(xZvep+u0r$!s#QXZ#2D5W#t3Z?F=d#R)V$zXAjvPXim28RK+#@Tz_fG;M709@By&NxB{aQ-JK}XI)~~3{Ga(pA`(9JAo9Ht*Gepy01tsywmRi=XL@i1RQrQh69gkoe1nvLs`>r1em)>>w1>14OeQL%R6tzwUj>aN|H9!^J?ZXp6$JMxMjoq}fn-X)>6g4UC+v|lZh*lhv4*!ol9a;x<l~(mXkWnbEy)F?nMQPJQp5m8}0>jxHnJt9E;GFTg+5P&YJGg;dY70HPeJ_{Cl74#t<9)VgN9Z}tJi~#8Wep`xnZQn(%^Wb$i&n1Oq%7O=qI)pr{aP<H5>`<~BW1z&<>bBT%^&b3u7&%f9X`8({2NK{(i24?*lh){veSySVDGYt>hP{lA~;+*K552TX#bY-P1zw~3ad7msdU{ScR5QY=K?z_`^iPjE9+)9@<<_mr!i`Fu%%^|Duo*~a}kX4+B}z^T~H~z3lXg1C+>Y^%Y!Bs+$z@hq*nIShFZuPgxtdY7+dnqhPKUC9{`INLZx(btyqks(&!7RZ`~=45RA+7pHXbOkXKmZo$P^~>qs;MJ)W&)3wZ{=-%g~j?(q!%3#An+vZgOtZb;MUmDEAInN`R&#U1Ag_SJQgA=bFnfV@*P-?WLIA~8hPqs=t0><{>vU+|Zx-K9EOlv0c_yR=Tc)514X1rpNI9B{$A+Oe^aVG$-&k2aL$y6b#~q4yf$54X>+go%IEHw1QX=6OcrX39E@I$bp=NM+vg*5rYxQjF>0)*WnMRBeTn4xBNj;BwvToewq=3Yg6$x73)a#_Q8z=JsY+aB2T1q_YMfqm=9-h6U&S_dzQ+c2j<&=L(drP<z?IEV@ys;2Cj|N8c^9;_teuXUh%rrW<OWUIj0dz^h0B^R$D<54s5x=}Qrg<K4GzzG2B~J4OBty8h8QNYTT=jATF;58Wi1<DZs6!!i65YxZT&GqkqOcL}0tz~T585^!W^U!%-U(j%DNO>KRU4<hFr{xXe%=miQQi0tgiZ9=E3lJ``-OHCxT5aTO_t(qG+u9XNkL7kq?1o!$gwN4#nU7IL`kVo;D$Ojy|j}AwQm1<U11cI~xkXFfl7>x~4PKRHAM!-HrYBUfI7Og-ThM`?{c^5)OSONTl*`TX8F^8UiK}o+0qfV(GjfopOYh0p7!@Jz=V!>?50FEI9rQPT%alYURNJl3uXshF@DLp)e_EI0;AH5pK{(RaUQwDh#zPgB(ZUK5i<*?Y7j^6-fFZlr%-(v*kYsyiBZ@g%3Xv>FqMSfPoy&S3e+1$ozOrt99Me2E1{bRim|MBba_1=Yb*LsX=m{bO$VTw+=B1c3L(_`jSd%o=$!7Dc#&)mj_E`p=e)B*`4*}c_|O0y{0`=mR_T*}{kKw)4(oeOen1JA7AO?iyks-A7R;ZGp3U_7n4ff~|a&35G@I&EM|A*Ylg*(kGejzw>#JIVyM*c#fTaZe!2;q`k+e0FLHun*LmitYfoT^xAfo@M|oZJJ^y6$*uVB}?2wGEe+Y_I3u9SHTe=U02MyjF7%s5V+#WKW~RgRpamM+eT!pY1Y->*phcjs3BMjoax!U2fR5)5&SnW$|L^aaYV@L*~&VLia~DpoW4mzbgVg)Kw4^jN*8Efl^_0Vj?R|0g2D`>g+r(V%#xd$qT?h{Iv{NnyhbF78T*UqY^VkZc9zmT?5IMAR45!CnR3L64`J)uEVb2LAZ$N3@N`5dQj+e5L!?uRViV2tKJmw#sXg=$PoFkiX4092_7c??f;ZU&H@Ag{$2L*&!VHQz`0%srtSA*x$J#~_@W$~{3B}X|g=oNI%C^KOig8IwcfMD2n1}@5oGZ78f+V5y@5cxq&s?Z98*Kn0&b#!!+}-@?Q92a1Q#{i($m9M>Q(%8L9AD5T`5k<X=+~}-Fo@W?7&Z(fw#oOlyh8jX<1^PCWB{>o{jh$@HLFk(di@+B2A1Y(deI{pRwGNeCEUDM?Y@wr3`-)a-Z_xen&bt}5?kpdW0|ugPv9F4z1LCHWl|0zl<eBxiFo7D?tpz*(QfQ`T4c=OtZY2~QN*|RYN?RZV{uj&Q6kRcFQ9rWh4GaOA;)p<1ZScGdo+kshK54Rkvp+qGqk2SO@^9EcZhvl^|O?=>r{R;SA=Q0uL8Qua-a7b)&0HNMXX%wM3j-G0kYIamlVZT3w-2ez~3Ul@Yny4N?E*Z|1dPoycZQnWLa}O1;x!ef`D0rxA>;;M9UzfzqbEqRQf9NOEZdd%s~hscoVxQ9*+X*Z6bs!?rrAV=2Ce$D`#F?zrbYbpd%mvsmob>?iWEO&{;@iEX07Ja_y}lg5id3yaI45gOfL@Xt#h9vjOQnhEuA^=c*;Mjs*Y`oxh5K2%X*u;814X9M2vq{mFYeb5ISdN>m$&cQR$r*!oTlO$8O6Q;Ce-*x|u3vvqV(M)j1~Ym`}HY>wcD42Z(Tf&EwG`ekjww}&d0w>spu<CFnb-Ke8&qVwcd38+_;<r<`vEy{}=xOdJ*+~kNzASond<yJ%@=p-w|I1($IMaMGOU7sqF>@~Ro=}R#6tf@b^`hwUjtrX=J*fMv8pWJ0T4td1pQx10=tr^FmBC!_o&E~A9Bf($U^y9;w*GQrod3kZu`DIme8wQCV1CsPUl#2;(un`!QMY|0qD^-n8L~~{ZCqzOrSVTP81qAx}QFeHB-8qDa^Wne}v+v7DuDpE@nalMYh{tjkfy)s)L$##TDGNa5Y8SDfqB2#N;pZhe4tG2}XQ-+zHGaSgO$ckPJ+X&P;ckyL%8plpNXeX)vTiVzd*+3P>jTrip@Ao>o>(zKxgOcXEngXU#?KJ;S;lu3vSlCnOd<&pAnDu@(Beh>V<+R!vr!wbdryM?ci9$|UDM^yo6{OQ*w9(M#D>^RU{N7QdxiKFgq#V7;4YDWy(lXn7qCQI4m}+@*n>gtNck(5M!Rl){zyirXj+)n)(J4G5xEr4no;W&1n{)QfEu@1kEE;1u*umef|h%WRLty;VM{UL1)}GBh#{~y{!ovk_zU-6P1>cjU_H7B=aWb7!&VrVlo1~%<f263oS<=?^{R?sIJP-=qh^=hq)~T+l`iBR2TjSo_TLg;R6rsBrn{N~bQKY8;cVlF-P1lmk(9%Hc>VU~+RTVlD(77@(~y)I^jjm+QuEXOgBiZ#Za4{uW1;xPQ;kPsPk&EykW1!;r;NRFy~*%HY{3A?EPX&z{+<;Dwv0@qw}-zUaZKUOo5|~+si!qZ0S*B2`i7ZOjs)eAqRJfUdN>W<!m9k-{w~D9KmRS@O^>Y%lju{z#*QEC^MA>rj8!z)YQP#Z3OZFL$a1$k->+(HI^ZnVYfP)&_@pIgz9111p53@0HGX4l-(2XGqZVf|TPaA_XhNqtN<Nh&%p?%ZEB3Q@jAGXjALSB>;FxP3+vTCj3K%nxr&bly?@A~E3R5@p+cerD_}GXVYTG8pd1k&%D6iExF%?i7?vbY3hCj&i<uU|hPHQkv_j#xJ^p-+`z(8!2^F8H$m48{>W3j=<B-+n<@xJlpV6No%@QGctH_X6jmXC1jvy5TQI$-8;=hes*in*nG0SLGJ$g&`iwC~L_IzJRZ-QY21zjOSzL#n81=9MQgsG-;?UC|z%HD)i@7a$08Kj`tb5YFQ0#em`xqrQplQ%sJ;=)@1Z^m8hMxh_LRh~1~K$_!SB32=~*jLro)y4-c3BkwIH7J5J^MY~P5h3D542H!OUNCtH@)L;wl*8xaEFjv$&>;GHa8d03+%vo#N$vaYdP4}-SXoAefSk~6pXzsex(v)n(<~+LI%-?C5yMoxH&v@S<n4P(<&T$n4=Zu~1%CI`+hKWfdjmYq2_;Ie}y!N>ecuvVo+CIDH!c|?)7}ODBP0)f-09T&q0Bx?h=p~>cUv}7Ku;87*c7z!pgf_YH&mTW9aa0C<%V+)l$c$q7fFn)8j7Hwo(L3TOGCT85&KJX>Ig426?z-kLYrgkQoqAJQQM>)87yHoxOe!(gzIW_rqivMi$66?rb>eu}8Q{3x7S4Q5vanfT)_P(tje3Mc@^?;M))A-}l{E?=aCO6Skz7^;xKO0spz>Xm06mU>T#1jlh4D-Zs(X?6vtDJ%x0N`6Xp{=+c{u1b`D}kzx;CoaydP-g)=J#qkKplc(+1gkGZy1|9XhFSbBJ+jv7P-BaaVDW#-f{007G}jNg@0cNKj^5Zn(sv52RJ9O=lI=h1Qm8Q}q?mfP~P`TAg%z4j^U3qTp`s`!>)D3?=bvj6wOu{?7zji;n{yaWw+nbLX)+X>(+IcWZneq!hKrs56qf!OPd=>b;*28c!M7+!93euyB!vEE6-LH^j#amnSYsEHo3_bELU}Ys|b+fEE#%JNV0;hMFOifNlBm{}(mxTmO&MC+;Cc1sp5k%@Fl^|GR9FoZ*<A`@5Dl6x~vgPG=1??>CyHV>ysP+aVDp(+1Z5{48qT|5|Y(^eMCF!mj&m7?kM^a$3Rxf-w39%u=h!H={SR0UufQTsj0wzg9?UT;?-$2D$mui+<kg3WeOva5=LWy)(LalMB{*`eZU>i6YX52XgW&+Wh>_a;U&U{2Wj}{vPSnc-9wsRovp8I+4E<XQt2P4dOKF74{C5BqGp=q#P7vZ~c0+q}3Fg@2J|vhwEn)6-!vg;g_oH+eWjj`>ZjXf`L3EeK*KhyZG}NR!2~O62?{sY%H4<LkKlOkS(TQvOm;n?s^cvFWTf<9EAAB^yq&`FD}Z&F1P5&n}5AFiLcTwmw+IaaDquGIFRLs#gcxcy1U=QOW8AUo(`|X{X!^_88%TH`M65<K`9!gzZXxeymOlk%Nx-0&Wb8-Y)uxJN5P#sJKg<Xp5q7X`$g2hc`+eL8hHWtskpN!bWXf`=*9eBtY=3u1HV_-EuCteke2b24N2xl3wQmIh%n?(e9jJL=ef2K!KItViERWv#TWBL8>YCuF-D^8RV>Vq5CUNac9gVg^6pT=58O6ivr_!ZrXU1h|Fu;!cHZwt;_j1%F}QQXfh7Q%@?Xk}SI)&nw;NEW^_&J7Q^>Vm>I7+Z#(6}X`QR{O90DN|%FZpG&;x&Tz>U<%kG5uv^`=jTH4AOS`1<peFo3dtG~2&|fW}`~(TcuJ27oNZS>i$^_+DY}(w}>zQJ%7*<XNzagmW-n9Kl&3JprA(Tj0NLsNw%`)xS0dubZCKMlO5ArO{C$K~T)pWCW&N0ay5<)GBbK63#^p){?C>=Sb{SmjbCT-~!_h`=dFBd=N5laCZZPNW<{lQz$yPAT%G#&M?V!PdR)WK*8#QHi>11sq$}cSxb6@up9!?9sTivmr>x>&jZv%?rlLV&&ZVP|E|dXS)kEi4xE}x$7W|<aXX~3SX~OSfLjRjk`Qy7ITA=6jY%0HV>Qa26I5Q1f+2IEl34=#Q|DBSb2w`_tbdiKhne?O%YcZkGkT)$1Rodvr#br`eETLRlRo8k!_{&M4^9Rt?(ziRWrZegdJGs(S9EM6&UY%f@-xH2&}2^9a#_PKlkpw6M~OI~6;<1<7Zc8YVlC%xhG*hbo;$nuhzy_8$*nwvT|KsK*08L5rLeRWH|~<n<jQLJ@gLb|g>wVak`Ju~wEt@gv5E5O9>IG!GA6N1;DW*byY9vqvE%(*&QJZ63CCB4O^rlMj>6JeeikJJq+DVJ0<dAybh{A<kdt}>F}H7ntOc=EjrP8WN(#n#QIXN|uDrAEL__c>D`iw0ma-VR^GnV487|0TxqNm~3UT~)4(8Fs$yBh2%_`))M2W!%x#>@}@qVS`_{HT!in1Jj(4HoE&GhpTQ11)P+>i92`?$JU88uaLj`n8K9b3I{8iPOzga5RsYN99J^9*$UHs7SdGRW3WR(2;^QrAV^9V{FJ)NArbOeIhIRi3%a6u;2qs66Yjq<0x=I!Zr-mf|Q_^+-SI`oB2m8a{{PWC0}VajbM0Nn&Ui14{Q_+b*_}1MG|LYb8u^Sy$y^aU!xp<E%eB<a}@8S<g}vdo$D95oq+!)G1ysWWHCE)5cK7D&uu#KeO((fX6e74(C({QBaUZG%)bBOSIfep^3fVX4?L)OUyi{%46!W{^ZosN5OggKXVgIj1<y6>Gxj<&yX5O?`I23Vy~jV5FKb=Z5<s-hCG0ugb1h=ODb0<C1esgxb{3(#POacS4XHp`V`+Al#TIgt`9D<xYUJv&S=KJkaGzcNWw_vq<0u-WTzgbirAVXwlw=sjbq}OB(Tuqk93EfiLzJrB~NP*rmDYcSe?>%`n&RN^j9Y&(-=HcVXHmwVcyq1N<}~)_sndlc<>&7CUP7VF%gVKhBm}R?#_CAhctR(t#&40a&n{5!<Gkl;)@?2E<-K;R=m)w|Mw4Kxv+M(y}=4u8~v#Z?LAjmQh^R$<FA8n6{1io7`>rFA+Y5HEE_Vv>(7#hdXHvGI2jengwj5eN}1Qtd@*7sm5E8Jtaa6<L4x5ak0+&kDA?I?)-)VvGKlWD$0(F^z-3(#40>o<=O>+=G(CdZY2c#5phUk3ud{c16K-yvTF0FL&1a1)B~a3$IK*Ng-n)<3!0ocgJ64amXdua`nE4%Roq{qCA?$c&&%ALZ7@D>Wun>{vDs2{^hwQ$~qX2{KlD&c&Tb3DwE2Xk^iE#dzAx1Vjmaf93U6!kqzbHrBLQXtBpuVJN`yw^x-H&vG(}nD~3jDoLRrVV`U0$`G!|4DJFc}{+i@Ac3#@P%qO_`x=zzV*4`*VVM|AzDe5JRYld75A77N1Y2KxQoR6cgF3*kQO6Mjdq5$*0*aVL47jCRVmbJs4V0@DA%=_8<J-ZWFdSOJM1Y0G8le54W~aD6{~ii+^64WMnSW3|=G>Me9wBPiGt&ckL0}$N{ABL)`u|J-99@O)&Q$`voA&v$SmR9~f#*COU5!GAEe-+ZuK#x#SvXI$pHf*T#2UKMc%KWs*QE2g}SdiHr!VxV(40mEbvw8)frnVju!Qf<$bl>u!_u9DO$7vh<Co_L*cN2pOtjrE^A(@!kooUF-i8qlM-R?s!VmpzGD#%ea=XTsK@PxvsCU7r;&?@G;T1-EXXfrdmD1m`wb2LC+12&5t7NWxQEmm;`;qKC9W_tv~+x5XM_W&3m|_{LegAE(>v#t4CIXl07uRU%M5C2VkrzrL2>yb6J1M4V7(2D6c5NR#5qG(Di3X>1P1^T?TyQkXg0<6av5OQ^@iB6?mKOs%l&)ZXT^bmU{q^3)FbG8IzI;hh`q}dTlwOBUz)><ec{{@nAov28fTD*D9^k-OxIA7J$QJr#stIPM_b1D41k!7fEXMzBBfqT*!w2D<%Y?(z*6;gA!D#q@0+uwT<ay>E=tfd5G6eNcs*sbYeGrkKc|>^4Ss!ffOgj&{c|cH#UyEL=acc8hPKCrueNVop4eMTh6*Hxulhqx73S6w0v>zF`iXjAmI-KSuGs__zk6{2|LYlRp$lUln(kN2iWHcmv#>Emi^0<)D2P<JQ)fzq*j=L^|1pbiClAGUTH%FO4_Bla*fs(moL7ztdvj{HELy}bbazpWqIKC743Lp@k|$oN!T8)u`LjG7h3UXh4G?GMY<@sWDeU3R_rSVxH5G6_9pN>2)}(j#-a+Kc5AUNGjv-8?15tWy+L(<PAdJkDIWZ9;D73k@@;Sqdw!xghRSMzHp%|L;K@tGs3}t?T>_San)>n6U~m5L(aDt%HWfa#%vY&{qBkxqfV6?jcM%F8JQsVL`rLwTIy{(lX2BKrVbc^)FWRH_)O##{yI9<%vqjQcOt7-cFK0V(TD50obzJ31AxMbVawM>A)KdN>ri`sWQJS$Rd49>Q1@fW;SK#|?q_@#2K<213=Uso*d$lUvY?n9gY{scya8v2WdtSPbBhU^xzF4MJ^*g}5)|JE~z8;iz)&aI@kNvUEqAI%kBaOs(;|;-tH#5^&av9m`;T%}ZEFJ-hOxxJyD|<)dZCU)vkEGfE%&5EUOz^$+{^op@3dBEDhpH=95c=1($G?R*>&0PH0p@i^=k^yjs=_Z0A+thbR@88lcAf6Ka?L_c?<8y&l39E$UmKSt(NVF{2faFpv4<ZCRx4IenYfA}z+iZ?@z8EjBH$WDO+Wu~<yN^DYRh0WYOlDj6PGUbg%D17@aISIlFi1)-ydAWHi=-`f+KsD%KUkYrfU8s3;kypwAd95X<V2Zmw+%jT9m6o1fn}iXos9W3=YiZs54EEze;@A&trLxJI4!#F8NsygisK+amP$x%C5hU15fP^x_#d?6y+0nqhD()-kUDMpRPpV3RpVbb|d28DP$~a<EylEUM}I=tCJW`6g+4Oa38={W_iJ;3?yOBW*m+vzwCQg3$oHW+KKyh$QQgf7OZfXt?nE8vx`3?*AO;Rj619(Zx2`D9p2%rZ^en0c1<&*mfEw3{#I8iULo*stne^B=KCt4OMu#CMBE**mXxx$dEgC9{xPg}sxVpY^OqeJ<k+`13ZXHzWNHA;F^&cxd?GKZ;j5a<A$UbUo0Wq!i{sQJja*f_c75?I@O?(oDLa48)U?a~t#OMC<>N#HK%1)$VyP4$aKBlU&7OuE1IZNI-_&x*TfI)1Y09hAG5y&oB<Dl(7u)AhK!`6vQ6*nwC%v7I*ROGn=)Y7VL}*~M5jQ}MM$<H!kg%0GI-mt7g-o3Uc{7^P<2x~571s7^LX3$Tv<q)J>}()~eke^(9Oe#@(OaV6ApHd-mTDaatQWb~?#>GhEzA48=lZ0Tr%r0jX<t1)cqXG$qp}Sa)lKO>fD-obq1)dAAt^fN6r&-yquDtL0Xt@+*tMsunKYsm2&XkObFdIK+rhW(Iiuy2>WF@mHT4;sVOlw2M4hoKB*V>0EK0-`Md6+dYWQ#c1;g0V|9s^S+6ckI1IGGbr{d?2fPN7>uIS@+Zd@Em+j`mW>i^mXUP&`5rCxu)X`Ed2n$7||WpUcCBe*t5)Rq~}JSJ<HrH0ub9wue}$UU9rj@rP3`?Eq}sw{VPin)s*c~**54!dm9rYmn_DJ4nRR-{0t{LRLZYQRwKJ6W%BlTz?r4(ANsXM1NyXAa?J`o-(~EAkyjl^!fcnq6)Wod=7aSJKa{z9eCvzK!ieUtRh^UN(gdFPt01sBmH{m7uv&6H@U|?_QtfH`!yKGW3X2nZ_3mZJ-3iCt0<Qo3|!?M3LBAk08Vlj@z1ryO)g5zC#2)i3_9K@5&Ds;3s}j(eyYEf<Rqs22j$SBl&F9vdp7ub{Kx%BPdi+sNJruAk21e!mu8v#cR2}V!l6>xm^VUoa>tf!@!~fSi#7hcIZtJ4F_DD?8>UxA_MhKdF{xXRLVcAYdI+e7&BErkxzRd<1%-LY&1NYu@>hhQVKj7vDfs{vR>vk%Qof4wYeVw*M@kJBG(ROXzz8aI>|G;YMoRf;LPg?z3R1OoUP7*Ma&3j5r1pzT;AmHi`01)FOaVbUhc^TxEf#{oT3yRGHmF$Y*QIPIDoyRM(4Ypsey*$Os14qG60-vwZ=PCw{><m1;GJj+Zgi~4~ruJwdu0|!y4O;=$@7V_CiD9CUik%aN8Z=Ra{=&sok*jkeMFt4Vrjr@vW>%Aa2Tg8#e`Fbud^tVFpEFL_&?3DjsxADnFP@jw0Nc$s0(#DnD1y4{t^UNOn_IlH5~mujNO1Nmb{CquUz;F3XYlO3JNB2NAhNYFt0NK2$bhMa>hr-lE6^!fWg2y4S|nwwIolU2}Toz61KB_OW0mbAC8&WSDI@(g!`<w~o2gPW5jrJqYF)os_LMc13ZpNu$T=9e92=+wu~%vbLR*pEGrHsDRpQoKtB2ATJUf@gHI%&BB(V*6ixpDK&8qycfKQj=<=qqIy(?tU*_);FpbRcht_F^lH)28bnk<5^zg;e{WiTWGziBD;>C$FkYAwEuLD&+KT6grZo7^gPK2}GDKxBBl+JWr>j&o;*e6yeMMFy3h6(K9lcaAW)oXlYpx+%6fI5J$NT(M_#>VRhBl66zUrVIJ=isB6XayOu!E}eDEKO26?^j1R9}nc^{fl>rxQVLgVrUTiAh`RC?bu2X<FYe7wv@-GxuczaZH4GCSWm|W0H$)FzHMRO43J<x=a0yb~|wSvXQGh`0nMI=zM({;!v*oxqwYjO2{e$jUBUN=-e!@bs-=PO6D^0`0W|fp-&TPYU1I~Z_`<d5&w1xR|5i5_y>S}w|z55g=s^ZxG`J2vapfxMzh_At`pU2Opj@I=DtQI&I1)lr1MlKynr=dAW&)6ro$LVsA1q-&QP*$Cq*!5X_W<gbLa<2Dxxp9x!C#nOk*C8EVB%&><R6+++#pc&gMz8fb5<>fRg|I3Zn^X@Fdo;ze%jfV!&@9384BN=S?%{#>M#4aNggQx(?y9O)&id+JoWGSe!I=PP<nH@r<f>FNam)?B7a%``&hBC=X*LqH&B$Yb?At-hAU4#eS0)3yH=c7jF7{+ntX7AM!Inp5r<gVk`U@yL;f$TT&zqYPSZy{cnCk>1-+$jkPY=lQGvGu!sCl!;`Limss!AC4eIMVr}3rDy&Jwai2|jv=fZFh2-Qm3~}J57m~#L?+VC{X0_DNBJfR$DOo`+)1j^3$|xDlv!2!}7Z({ePQEE9ngM^%hrhFvT9tUxD8y{uL<^V8FFVU?LmeT<V{^@hFFB?l3g~6xoZ7&cLa=s7VNxauYHBgQglvLQY^#2K=$s%x6=wYYklZ;25$w@cdHN|Tk|_<toz?^$V(_gHZhCTQ?b+w%*m(K))7(Oy!nRr<sF!sm!pJoJ{zDcUsf~AzOM8FFaHJN?&3Y_^^QpWFi=7Nu;kI1n_U2eB=np0M5o~P!(}GXW8oQ8R2zpN8sL+}if~cEbEW5MwguT6LR$?Yr;b)u?uY2mWSZun=wEEy|rkQWZSM5E)9ZqCnW%hvRbIfJK)MN6Jhyq;h59%C;o0zvf+D7*pTlO6NI-@vA{f0QUm2153G+{hR)wE}tFe<d6bwv8JChpVRp;6-}h(<LRxV9y}-uhtsK2?_d0u9M-QkM^&@w#wv3O?f}$<o${+lQyhLuTPalTn8kDas(L6?q3eCA_&Q`6vhwbQ$?HKuS`yz4J@_;j*b^*QZmAcvv*Ozb)o-Dl_Jseaj7lloE7NHs|<4UeQ-I0Qyv3z)Bx4|94YAe}VB<e#`eo9Ro3-Vmo>jbgd=4FXQp97-b^|Z&BXE&Lz+jl39N;=<pW>H8cITtn5}ixZ<(vP^bpxxVl)mRjyU$EkqEIdOB65s*Y%Hx<BS|21Ty*%p>+c;S2plR|<+5HmyooE)#qYdwQCiJD1_~IN^us(<ENg=hSwTcn6Ma_vd`Y?}g6PK3W|2IjWL?#<Jv-`CL!rbdU!RENVz&u?Iou9BUE*hd~oXKnu`=k}M(u)pZ!YJ|Dr|hTVH--)q1fWb}+P#_uc^90&PolxJxI>~Ko%FkbgtmbUIKzX!vorNK`JI95pxj?sExLpi5<-bUT#C}M3DuChufY-{f%kEm@&@{1&?*ILJ^F_6(s0=Rlz&Q@`zcBfk1u=`@V{dwG9uWn4Bog0Z2I(91^;U1u6wv8gt{J8M7EuPejeOrA)g|nkM1XD3!3K>|ZSOKBy(C>$?dm%|*bY1|9wNy?P@bVxEvXh_{(=q6h+_FE6FVkQ_1K7ok5W{Np`c8-B5hR8GvdzD3TkmXmyElJaVoz{jGqy?W!0H>xBYkm72yEKMmP{pid{sT~V36F*Q8Rk&Xd+;7S(x#jW(0KkC5bklk@rbP=N_n=A~iM7F=3~=KYcS|26^|Qk^l^kkz|f;`tvKH0>MTXtztBYw=tD0YWOqd@b3gr`ja_mE<4SoO5iB=7@zFxcEmGQ3PXeOLYEXV7JjM$xBI2d=w3R{0q&`Hs*1|pfZJet@4rXx0AYNv^2J#g`UYUGX_@+#s#hK^Z47!QQ5luCwD|9}=J;U05mtZsYBtgZo}b5T;jrGVTO?i_^Cpo0%KX`aQ{QbuCKgzoWg${Seq=ioXJ1yV7DLnhnN*cs3gkbA`3&fz=;lzJ%D%+FTL;l%-(hqrzbZRiSlXkx)4C`7$#?_FgX#_PR--R2`ZcxaM95!4qdk9@qaz~q#@E>ro-;2fLO&+GsPOx@!6R<^z}#IsemX(&CLY2`a8f;4`srV{BJySBR%fMo&kdfqYnV^;tbo65p6>j%<TzY<VQ`<sY?|L0OZ)CpljLFMl+c`mx6!*hZcKu(bxj%vc@CQ_g->yDvP^GoEw#s79jH#>&>cjqo?<uw8sH@R4!Ty)VJo_E<Pk6qhJagPoj=I#6PMdBVG1eQi7+eNtwJM1xgLw2tZ0^*J+CQ3F=pfEa9$?c4kn207&#2}mR9d_P*`Q(*Eki-MTID;AedVMuS*eTGe^U>Kn6nfUy$zPyQ3B@n{rVEXpA<W+Zqp!@+gv7JRePqZbL{6W|mlOruyC>?>wb9AT~_hW*?v?D@fE~H`<k*dps9!=eyB>>U^U$YVl!rSHgv5f)sm|li}o+VkQ0xm<RlG^QO~IyD|8D1~i6>utr<)0dSrGGPih3xX-kkZjXPOJOxx#(Cc1u{UVNCu^ky}BSO}W0z)3`g*`ai?Or15?~%j=UJZP=lrlFXxRElG&=2&$zws(V(17I7J>hmp>|0-kKf?LCBc!|e!?XhNoU}<YJ<|9mwm|*^Y_JzeZj!`e(o94j2F+9c>HgsiMM>(Z;Rly>_^aOf45d0-m6GVM^l52?4EEbVKhinjsxMZYsx!}D$eQLHPc|N|;Kr?R2OZTFB(cpwKjs;Xx~3FAWex`z_{NmD&?(>@_3Xd?Y+X^|rB#k}FB!s_ydkX+4F`Zu|J1A_3j3R$65$A6WTtim0IA^aenqy+PEnLJF$uN&uBc_G&=Ktm7}x`n|FB-M``_F1eF`y`_QJWKdVol6Muh{GgkvA@%J_4qlx?eLd>a6LXy<zz0cI)udV>owh`$K`i&sKRi&<qQ6aC*gREQ}GB#1!=MRqM$$Dq|UwO+O~NRIx<J70KXhrDWNn#)QOwTW*`aQ?tIRKcm^iJ%m*WAhAPo%GY=W-{#@YJ3$+ZD3xCNaK@q7V#(3=*3~oX7hOk8VVyu0PO_kWB=uBND$PsFXe|O)Rj#bJI9eqN6bQc_>(3{mWPd*HDGeNI4Iuf9sZyQFGNOtkmN-QJ86>2{bis+LnK*c7m|#^Itzw(#YzKwV;xha=8-LLY@pJa4!zyu8OGVy<`5ci{~a_Wp9~tm^8=Ky`|JTFj`U7}2DFb)So%}3u@Dcj%foT}K8}@l8C_@6R~+nmsK0Y>4GHZ^-RN$$%j9bNF8TJ{vI?IY8U*cEho^QJKjvW#L(Neg{ZaFCGLmmu`E5n{eUgc@tI4PhMA7b3*Bv1!&)CofN7PTcDq)}Iv^qZf&c<wLh`jk1E$4bjO-USo5nVz>`CrdOMMaV{n4QX$tU-D`7qY>OE8#7ZA;O3}$!f)=aX*!aPDe&xeSRdbLq(>$2vBVp6moy--kkSr&j9oC31S6_g?dZo>oQh+YQeG>BlD)VbSojN6?3KU?+sOsNwSbF-8g+Mg30AED#Q}}c2T3eaLyCJXqzrggLlYdShi{J@Pn|L_*HJeQ!{2QO#uNwZ^j_9D_o9c&zkI;UK<0e1R@&@n5Ct-p=Z|_Ko@hR=?5n4WQgjAbz!dZfJrN(xd^)MX2!J$Zgh)R=kT-<9nR20o+n^|Ii!;^9d}oadikGWQ#8~o%3igqk>5=)16PAE_$u@ZLbYpiVtSRSxcAL)OXirYe8FO}7D1Sz0u%BB$0y#8PS}A97CHPk4hR=OnCN+NOBBz0bwEsF*k7D0X5@+@^+II>QIqe7efBq=p{`c&w)1X%AZQgIU0=F>=rwx!eKunZf!Vx$t5>jMyLdJ(vg5aG4I!pt&SWIn;~E~(sU^-?0VW2_IX$D;eD>m<Vh~cp>Ug{XIo5fr_^NT(G=SMXqVMG#b#V-Q1m(l-8#<9&VO4SRkHND53<4+<9Kz&O4Le%u2O2`&O5L(3_BSPM=Qciwe)-FT{Hv1u6G(~<!Kx^p69bIO#`PzUJWak^WNEVSp&S!;&Dpzm)b_f4$a3lpkx~R9-2;ymGnc*E-Ua=^2-zj@9!qrx+YRC*{j6Kuoy9PsKxp?VkKP`P&xAz_WQJs3aT(k`nq|dFUc2EOI9ntz_WLQjG^;{uUUg9`A&>R+Psr5gwaiHJllX#6=)#@{1*&awvFV7f*r!2kyve-koSMI|!{~4*k^?%<0av!O^sqX<?>9RBNba0poynt*q4Pj?TfH5{!hyx%(3i8$3W|AY2Z+=KvvjWguRB|&-cWx(XR`5|60y~S7Dfq85*EBwx(`W))G|Vc^lT1@(J>_z=JOm#H&b5KRLDIZ$C$`wap`CzbRfL4b(mE`vLi7Pia0`AjZvS4ST8*f$<IX<Q-a3p$U72f>HlfYNcK7L%Bw*W%$7l{x%8jAB^KoaH^Z^3+xVa|GU#em&Fe5SJ}wq|UoUKaB8@gb6Kshm4}C%_jfM7cf??{#iB51Znm5?jEnu(bU7&di&PCdanPYPypg|NIr7DwZJ%*WCU8sz-?o2+5RyEV*r`hXr1sp<6R+K^Qe_H#Fxk38(+)t!b;J$f}xHU)=j$TFkwKWT74}0nyX-SW4y?^HIFYj3(b$Ir?bd4<^^Fk>mmk+k)iK;23VRwXCr5a#{g8o(Wh&1v(GM>f$%LL=JL*GXn4ZE8?S3HVv{cXs_mw-ffG+<R>qc@ddQOHTWUR!hgU)Q+@BIk4z8CQ}!sdLi*z5nLrow}=M6je$<UD|V{&m}W#WUP9-oyeI$4y5nuhyrYa77yON2$RO(=faHao6ccg466I^d$SY&J5HH!^!d?tvp-1a3oP@wBLSLUnKFRY`|}O+@|e)}LgFahedF?_Ek~Gg1g@4HKNOdy6IekEm1{5RVsMdQjmJRN-&#S66N-9AAu3~`q{+wyK8oExe^W`s`lbGHH7H?PBE(`y*GE6l#~U;+HZDxmz^9*sPy;&HLxaTbk8+bCB3Q3PiaI31duZAO$Q%5`*C&2UG-6~J=77k(LSSj)8Tggj@eVfle)1a%{vJ(rJHv3%P`8nTDwJEz26@4~2xT5hWadr2;!XX)1h-A>;Ze;II+xtetiufySw1hT>h1BAVW+e#+}Pn~p&^9s+-yuQV(lH3rXU*aTD7bu2*Gn4b2ma094BxV!_LZkCr!3Tm7;H}CfPZE;^$ZceW_7fv#^3!4BtGw>R(#dm-@;C)W(RuAQKzaYP-ED@*o*tD|U(Wy_79G?8L1dsX)-`IJ%x%riuhk`aHs@Vb4Q^=DBNe5_8n?T_mEiexCFK`fmJvM1KAV45pb`7ju?)z{UE%?-cN6<PrYqQ=~o{n!cC$P+L?Dp@x3b;C~Toqhx@l%X-ioG)@ft2|Yp~^`O&fLVC{(NJ#ub4U6psoe@ttPjh&0SU8vP)~SK+yHF-k_wF92**nid;`&{|x?BrnSun?g$Ocb^>@jahJ{V*D0s_#m>y%RQCBM<yC<8=$WRYx~NRR)N%T`x`T~>9Gwv95a7h{hlGLQaC;)2B?&%<8WFq7z?Yi6O9l<GbX7)8KhldybF0_)kp{ML&72wd+|W{9FtVKIK2a5xE|Ksoa;UG57l<$t}=F~}nobZ(y0f=7~@kcDkLilD_M({S=Jgpq(CrE7?i^I*sT>a+co0=6!ci#0^N&b1?&ToPIde{B8{Q33IhR-5LE|5#Cm4fBT4Z-NsVYZkYNBYJ4pq|Is_=w|blh=_6KI+F&B1}*7?F5*`M;U0Mb)|Ur1C=<mFuBg^edC>BXtWuph;T!=-h{s?OVQ*97f-C|sp5jK|h>)o)MD6ZXb(=+V-`bnpBzzC)6pIFC5nCXM1uH2Zauv;2$MX$Vy2H`~9AkpH0<WfycsIm~Tn1!Qlm%UM+bcKcF69ChCdJ-+hpApy?l^U+^Lbd6x{b$*!lb~G8<y<IoU)&3vR@LK>4s{cZ0R|Fv?8>@XW2BbXZD+7ZyXybRYKqyN!2;SIqTol`m)#7m^JKZ@Y{+4`56T~WPIS&SM{GNivy1KXtC>a)J0`%d^vsikF<^hJ7k1+A8MVW=MVoulV|AU*6@>`v`E-Ppy6|Dg!d3^O$|qyC;pJ^8IbWz2;+u`k;8>dtbUw?K{jQ#z6wj)H1z`#UcWrPwSz&uF8(%D;8O1FiNcbse%be@QvCgKh)GhJ5Gp&mUEUXs<wS^?Q_rATLDPFj=h!Zms(wA@LXePGmw_K{eehV3Cbg<MZY*P;jgyAi{VM;=wIEDx5A1K%osfaTO{HeyzYd0QJEa>hUXMHUYa#O^ovpLaWznpmYZVWU;ws2_sAq*oZREZ)jIJLg|9Dq9Z38V86NXIJt>V9{p>1+H@iBnQo3Mv1u6Pl4b|3Ty8rqJ5*_z%}3bWl6IhhA=3=>?{G+$0H@Fau~sc-jQF}q?LoqNgzSNpo$H7})>tSOnKMF&(XOVW6_yZjNB0?!?Q!&*_cMG917V{nVtI?Au)Aq(F#h=;P)UJ&0FYIXAsNn7GL=02LVMo?HBmSxbh+4NcOWps|~7_@QJ+)yU~X59tw%Mph6K&Y-9%=>;*LHyU;Wu)}V4)t+C!GFO58{9@#+Lt08QMp7BQr1!v@l|o|RzB!DtI_m#EO>3_`hLSw9mt&BUGa51JZ==xkJz1Nc$25m7D*wFJ!w@DP+~5W>kr&eSUVEVcM!NhJ_SK{`;7oxZmWp^zVH8HNIg{@yQ4RRib#6;nrMyo6A!%<?v^3dyjjB#u0s^^PsdjruQhSKvdU~g7|G*8Vs6O4PZ=<9oTT~k8HXSSlzx8o1)BWL4%Qb4!-){!H;}a_eQik-R5iSv;13{NDa8}|hAz|aUp(oHw^p-ujO%ca64MzP<NYB+^?!{qsmt8}^c{ULak2>|MHjzkLg!1onxAQ?_$(gxF;|pZ5YIX8y0D}VmEzsN4&k)t*t7b(4F?al#(F+Ep`dE18|CQM7C+lWM%sm%<z-m!dS<eS<}BJRVKKpsIk8~@v)PAFFP8b(ITe`jH2h$9wX6CI3B+$bpBPp@0wWu}&LoWp6+^Q+(<?ll>>N+!^(&tcE#$l`(H}xlXbFv={QzT7JIV35WDB+GR<{V;eDXnkQG*A0u7PFuaKL)(f~77Y;n;5Vokv@g%%L%<DyHe9vGQ7l7Zzex<zatAx~oO%SO?o<duQtiml_U)i8=#6#ZR5M+*V1d=4g1&(>}MRFK_cQ@I~f5)nB&jgIa(yFqm2zH3Wct2DtJOuvv%W@+S=83s(bWcnY`89ZB&q4_tRX*b}Du5`MjHf-Q)Ay|U@7u%AL86npSx-KY8D><M*Sg&1#`TOE?sb0J=a`)zDku=uo*(O*CimmA#R3A9JprSLTjM-1|^lA;q_yiTbyV5}?fnFj$@pW~{i9R8SZs*816ssIvFgIgkLXQAwPbT&!*3dJn_a?49+GAl<3ofQbMKqb&qCO6o<`aXW+_p9azNEK&DmpI+uO&=x&xU??N-9kJ=qK?1xu(f-O!G3AQCWXX;csNSoMgKBzJU!18yctEiTpkmHe_L1m)^R%yWgMFp+TGQ-Avj0ysSc3fAYI8}K@dLGM<-g*0A@(Qd+keS7qLG2`{v+xMILBkw@1AW+l@)fb`gD?*QnF?qd#>x_v3%L5&Kc|M`z)E-bQP^R1zqPq~$5$G50zMSf)B=UAMYb6Lb~&frF$mz*@J7{G8k3eB$zA6}fYupy!(P0xBwo@fM&>x4>Qpc4;#6X9FQ^F+0UWS`SU){J=UeYc}(PIPoc9Df<CzLZ>kVvJ|3-Ec7bTlD7FgZe;0Eu{gJ;8_m6Uehyan`ggSsGiMgK`~fXp7o#P-?ju^=OsQQ0;HbPk7}W@VEUKlP@S!LP&0-A7ShS5cH{)pZ<du>pgBm2$=Y#Jc`eusnkMeuh9TgLHK;Qn9BV7sh6d9Q}nRjk-5a2Sw%S+IovMba{H*dp_9UTPXv%;#u><yIa^%JIxX^s+Z7QZ56-a3==NjcksHyR&<MA+gLO44wgA6@qjwe!#aW_*;hi&tb+*XB%WpWS)1PK}isA!lXj6JXZ(CeJCXgk{PDJwtXgqWqfXz(GEB!1V2g-PS+Hn%08#4U8L|4q!?Efm4T7C5m+JcY0fs%M}WgmyHE+;_Kvx-u28->a$aTd${~K_uk%O7j5B&pFfe#yZ!Gtp-ul>$kPHw1+x{Bgy=BUS|72=g$m1Q;W1I#a`h7hEL;$Bc^3vOs_|)Sr;{UgO@~E^+74rRymgGQ2&o@1eq~))jeD)i5q1x|cz_<DRZZggnK~9}EW0R^ROS_+5n$liDGUo+W^x-U(a4i*OEDBwp=9I$zjAsU%8AhMvrmKa9<R=IN9P9+&Pz|GK!N(9i;F)Ep}8MDN&XBF*A%EsCzo}S-`A3bj6$VlTtL@NGuB&!$QlC&ZTkMW>=ligHqRn)5v_88MW3AlEg~ZQzzy2Nm;Wr%6(vU|VC8)Uz9O@O&5r0l;35NFSDnE{{{AmWKI88H*rCuFNKY@5&a<8dE72JkPBt(ljn!nV?S5bBkR?YFvfTuMqOj6?(0%Ipc-_=IARKc)uyRN^tH0S{tF$f2z9w_xFchP#JBAxf00{#fBhID}{PYn1%-YLZkzMO{VE*oQ04Jt0ld7i)nUxuFf4D^x+?Nsd^sb?3kYnIm`tD^hNy9<<HrO<QusOJIKBCCy;$kQ9I>&Iw$wtse0`+=Q*6oEl@7|RPzX~4s^K)OZ`ynuPxfKCLsF96o1_18aGC#y}<thM7iY!?nT}`V4ftaVn10z=jU2dxQiiYOTQ}-8a2gDx--?UrB|Kw0YO9G99GG)bQ@%3}sld4S;%0#vyo$IX*ZFpNs(-NcsM;yMTUfYixNwTqZWMDg>W)`kFh3$V(Xs-p19X>W=GJj(7j!i88dvaSNJzpsq`>==cM!@n`9dV8njif(&mNXVk8;f*~%RDVDm;TG6o4$1*JLJTbNYa`>YeW>LSpIJn2q%PQ_LnWcm0d=H+w+#53x{}{Rn0t1qlDd87TU;5LARrd9C|`ZgWw07g&RQ6K9hKevoDji!aKxy&^kuDZZd7BlX`&l8u|y{LDrGmqNB52Q^P&A&z_YYa6I$|jI?Y^ar%4tPTr6$N!Pl9{nd_{!O-`M5$W(RQy+Or1|^>q6(lM%wspXUlFtoheF49OzezgiFx&Atk6YH!{LQ%%ytC3ri<!r_KyP1P!hl;XMB&(@hE)FH{UYF%;f8y*lsGF3tsLV`q0_p)EM^zi#r(LNa4|uI!hT%uBO?Vcg$Ha#pSRVjh@;wDN9{@lODO9<lvBxNm@ji5T}mxP43^;`G#(&QgZ{(q2-YfYC~<N=>G3IAAq^M_y=JApyG=?cixyIWdf=sr?s_Nlcv(*aZ@#20>*7mTS|JAuU^D{p&Gy0864MvVn=kzHOO$+mh!Su}oiWxWJtzAY;-d#zSkXHF7$;I+j>$<{HrGb?nq(lgSLD9WM&S8eKBE-z{$esYT1o$$bcwK!v4fURAn(@Aq_P{~vEv%#Pkx9}jcX)+9WV9%$QyuDlT`P+g>=1$AhgV|+9sQ|_UEkFHa^HAtDxQ9JONs%sq$`okAY8)@PGzGbW-MTuhwQJx3peX8yB~3n`kw^L6-I+K+nhFE%P)xxEZZ0pMW8oPLgtB`b%Y+s;u@}$#^MaxGn%OP1_;3G?#dcZk6kK!1pOb;B`b_)FX5q21~>Fah4Kg9jGkUY#yB`d;B<Pp|Z2hI!HtScqStyvr8$b=fusJh6TH+Ha&Yntoux27j3#%bxg{|#|4r>@I_?yr|tle?wkSj5Xj4)(`v-X$r*(tVzy<{JZ+@~TI?%qzHJSGkq(2&(AcrMYL&(aS?`tD3jX8be}*Zb$yjY6>w+_hpemoFUPQ2ryZ6NK6;!qKFDUB7SBJw4KFfZs6j6mi@C{fz<@uGE{R|R|#^UK-BKje$lR*~cAB;w0zcKOF5U`Afwn0;G2&X7{<{_|BU<a_>(pIV5Ze<`Cjt<T|iZn5|C4!1bGui0+h90JuK8N{hJ=QI`o7Jy)7Tu6MVp(@)f+?JLp)-U+#g(LbYkFM#p-bu=Z0IwxHcN~Oze8d+(f{_SUePBXuygh&K`YD81X8iD<S!aLh&{>2ke$W_;r>FYQ-OZ<e^p68#fWerVm8r_;Kk$>Vlo1UB=<ul0}4Xo6K;5~7ZoCd%=@HAntGD?JNGU>2o`40;vpGz9O2DOo_Fy+QJq8@6cQnTj!0zk`9&CNIlYfcNtvUWaCN#TOxNhV0|R2_81YQBwKBlCeWJ8e7HP~iF8u@%@IV~Pjc!<CM>_$Fx%^yVCXJ<}*F8W_@=jdgZxQB~MdH(CdE_g#Cq9|-CvAs?ALZJHUja+vY~OhehL!=qS9$D9;yT1nFdsi1X%^12ocsisJp|+~Gx}6jzL>cO><zFQFyRlGOb@2_kfo~Sny-NZ7KF-yL?&wh8@`caX1qMAr_Kr_!artf$Lr-uT7_94TVXKl!4W^|q);XenT1YXBhp#qb}P}nWJ9}y8k4@3)7O-@o4la>TjM3NFRWc|zuPTQv(H36^Bm<?4~LY*_El~pi>W%PLH7X&$`TPQxCm5FiCgm_+r%)2fmO$v<)O9DprZk+!&3U->R3v0oY={!nr6=#&&e&uf~o``EE>{FwcT7kiC1MPB@Cy<(xdUYC6@>t?m9&mDC^}Ro=fe-!g&#+x+}&7KX6r$Tw;Y_gYo-PK)q!Q(Gr|b@8Tbcvo}A3#4{`xdbpQ5Hg5-^LFlhC??Vp3eu-yH;@PEd)wBVjDQw<RZn^FcCF#UR$n;m4C?cZ=55hH|_QT3-Gc+EmT}a?@P1KY+I!m_GzpuGTURqFmbB-tsLiLGmD9ELZ`T~A?71ZR^3gs?pt_48qgh=>jJ=FAj6+9)G(}c=)YZLJV=Gy)?2hrLuRQcsfX+0EJm*89-yS$2Hq^u<D`X3tixKCXR_|QrMH$^Qkx(qS4!tYn?e}FP@Y4=+OifU05a-Uub!R7=rh{Py2gY@z#&NRxC*Ll&TSCopALb|Ji@0dV9#_~>hLWS?f)grGlK(YG|dUb}KMQ`5wBXnuT{JoP>VqhjOLdr1dK;)qHq-SdE8uUu@-@ecLVm4JFs(ESHO9%`(Hahryg#?kDhCJGZDw2~^xO4gi!)bU~_(qU=>E!C}<vhcXFs;hJnDd7?IJFli4}FciaSXv%^K)pEO}MCl)20OiTRLGkR>27Tgr;3Q*owsi#6K?zmOmEF&@R@tYQ1+FFUrO&YcX$ol9oahx6U~aFya|Jj9e7Gnf<l$j<ex(i5M&A>$cdYHLA}C9?f0hij_^2zH&K_C$Z=U92|F!yPM25=5=1ggnX;4Uv^Ie2<0pQv{(2bRkB&yP-0ez`bj9153x$8<+v@h`9gkDco9p#yb|k`;(TzinkN(V^&0qTN6yKUQ_E}k{zvKt1%~!ah{GwHUO4+yD2`lBwf{zRSMA-ej1~z<g>@TsvJi@e>E4esbmUN1F)=zLwrcPZqn^&&t5K*%e3))}D-evqS)7nE?N3ZxbC``G`ap+cjd&r$_2+&{bqSOJuwES(4sz@*RQfPCC(O?s&f+wB=%SAB;Pb(ZHB`CoVocTg#MR<)9fqr2cf@MmIO0uLWEp@!<tiz?OPD!4hgDZl`enk2kTGxhhC!#B3~Z0?g-Ht7Z<r5hoaa^~nf|tl<Wx6_KU$VVf{ex3%&ejA6J*FAjP5@&zVfku(kmM(s?X`i;Ryl@$YaLMX<;Q4rrcU@qfERDGS*ws@0wWfQLn3lJ8KGD{^q!mlin!}xxqrQ&Pm`_>1dTHfV+bxEdJSefR302FJN1k?FNyM+D0O8Udr`mfYpsFonE(bz$9<UJo{<yBlGb9E=0nQFlzTVOz)HXm9dWI3fQLat;E%geQQK@cOGW-ZQYC!lqdfPCEZc&z%X?kl3oC_xtuF_4}m+lHJXD)0tbS>oi@C9kaXc7>DL^oLZU~`lHQG3m9q4Svuu!iSQ|O=#j&c||7@a&H2-F6kn)Q#0lb6TD^J6%wZ;kbg$pn<y%E7m$*LALkM=j~`*`nJ`ep0uymygA1|w@An;l)~Nt12uztT~2_PA5@nrLXv>bKZb=Xtn!1ouM|%zxC0SvQG&h71WVQ(TWw&n3vT_o)=S+8iV=tRMFdaZc7@UDJthl&Q8@p|~s1EN%Qf=ox!FL%UUnyIg3S$})$yxA~NG7Yg;$CuclASz;5y{?wfVdJ!#LAmX5_NM8k}%gN4WSDzidzUR4+gijP7)z2fb0NAJ0g^lu8xI;UfL#Jko3*a`9N_|1u$sM`~g{fb3yvVdwLI+D>V>UoVhb{j|EuKo|qCA}kJbpOIcD0Lzp6NsJ{_|N{O11as&*{foGhx9Ng0tz%;~ba_reti{yWg(W-VshWA!wIzHqFNvwJYp+CAi17^zjA$BJ{gG>Ed4s@vr2X>SW>f54rea1Fx&*VCyky*`6gSFzrmQrC`mr$frg37DY?i;R~nB%eBAE8ZgXhAm3PMbZV*XQ>tF=_F|rM6EM}6aEu~g1eTB+5bx+Ot{iiZV}=4auqID-^Wt{$<aeIWggcl^t)3$=b3?7p1Euv9b5)jk%!>Z|eCtKWvLGT`9xfHA=Ggzcve1HIzKw6>A;#qLvs?UR?T;o8Lz>H#C@^OMT(M|mTnqz=HNx4xNU<?uXUragW^t_4K*bUkI<URvxY=&EjIwgPodWdtbhxkgjxTBU@lYs8Z@;IbRd)I2mLCi|IRl~+4D1Q5s_^gvTFd4=Hn(k8sRmvpIc~1ek+{_KF!#U1QyK-rHH;Lo1cBbhw_S0_{CNrgQXaq;io;Mm+!8W=iy#mL2SRYxp@uwt^z|-SF{{vo1~`Kg0MMrvuA6?hMa+gQJp=k4gVKnK#c&dxQKL!Y82908rLs!caTd5$%^!xA49I)Pg;$Axp#i8VKNxy#=}i=m-&=SAZM8nlROI}Mdj?*TsCgxU?)1q~x+;;LuNUM}*XKu3a1)zX<(%geMpsn&=AKFAn+=$7hMBvR{~#9iYX9khmPSJZ_IGhAW&K`_5-)K8R(ng4wZ7dAMNMvCvwF&^w?yZIm@+Equ=*dVPMEzAw?r;UP9uiPjUM?P997Kv9V@Yun}kxIIDz=5={zxF&2tsz(P!bpUh~JS-2jofDvRt`2bLjP=_AcXNg*S>T8v+QBpF&UK~!UnYw8A6X7S8Qwds#zD-EBu`!#0Tju_8tJ-!<o6JtetvlBXXss>4NEM3kf|2q5bv>e-NCcv}%u>db=E^#|6DLjLUw)iVc0m*mWL2$GK|09r|#xtX-WfO>{mX;%?;St}9GDSIq%T&Oz9mYzTVr_X9NF_;<Z1@AsIvLN3y1WV?B}pJN9ReduY#IZ+=w40!6)grdDx$?O{|qEmzL}!VL+iq|EgGy2gGk-&ObIphnNrqP)Xd9|IixrpQ(~sA3o~b!+I4({`qf)2rpBTlr>jY9!2KXhRP+eD@&1t;QIH!mJHcKuJ5+(}&1%`$l*<HX6?}l%$H(ZXB9qFJC&=b1Li<)t7ti!5-LaD^X5y@m_F#WOx7=x)Z!Lo@`<o2!qSA$KA@1Q-vZoGCDq+Ngii(JWar3!!+~#m}yTfk`&ito=F<wZ(hz%^0wV39?QlXEWhhjATn9ycldm1D^@s0TJ)~kzHaUYLwPJLOCxN_!}-VJrLTV@7|0haz~C#7x;F{U**t5|i<rGJ!lwsjiZlACnisFQB;68SVocz}bVSC{~$X=lpg@V!eRov#E4$s50vGj~YP%Ti%f$mJ!pXYv~BN^p(>vF<@g!_6$)S!Zv)Q+{atfVHF<5VpBiS<{ygDXKg@i(S%&MI$4RZZ*xobAtNPZM*nq4j&E^^BLAP)BcRdddKOzcCl?kt8jfMJ_DHk$=2S~#LiNZaMnkZ6nIFfK>2RU2q?}2#UyT{e$8>2v61b(xEg(>mR`t~@v;9>K%cleZ^TXaek#QC3Fn{gPc~FDx-HB|j5n{+C1<k?h`Q^((w@jfgCEHpCzUl7lSc;&=c!yt;hKetfZC8@pFJ;jONT3opo2@3qLpn06me%_s-io^fjKJYfsgyz;B&2kg#MvpnRgAPaY>Ruq#S^@IzjD7S&sUXp~hA%h#tt%Bhbo_2>1piFKsF1O87oOJN**K2E!IPvUPDMCX7o<K-aMT7z*wgl8ls__05;PSt~N?K9KP;By6ChU*LvO*NmS^O>?P4e^I-`t$L2!9UT%kS|{bFcy>R_7#yR#@;>?6O31URznv1fl1YeF-5NGe1ubSx{K^AI1sQB)QO*5$&4rb{y!Oy`l?G4+vXolho?oBP78pK+g<bifn>Sz2BVJY7(9ow>-|dz2@LrT7C?oiedkdHV^?8m5Mp1h+@Tj?bP}jE0x*nZSM6#3X0uypiHgAwpK7W1%=qFDc(5wpGxd?A-!-`cnC^4S(r2VHOSlvCW@YNGg?i{S>$~y5_ksZZ6hogQe>y!TVu&Ccz=T6wq+=jwmn;XEu0~u-oV|euWu=BtCy3)BqPH@NAPc1%H3$vo?shmRBUfK^hUVYIf`X%YmXU9mh;U>39!`TnAaqO1LrQJrmganhC7cbH8NAGF9X$U!~-n-IGvxx!mI8O%cyA=&C#K76aWajA!F<9q3@}7qJf!7yV$=_X^0(J(Kq`Zb%^nSK~+t7R7Bf02V;cFR<_x~3AP(zeGiGskA&aK4g1v{p~u$bHq>7fiUy+kqV^4^hkRCh6Ts`UG;Br#s=30!!*EfYg0+oWd`8~nROZ2~=Sg8A>h;h-l)m~?~0o|(?@$n6mrK8sVLrR?9y>Z=WHFNL>?TAF2Lh<4nuB-H(eHQ3jjn>!r1`{n~0kG}ViTGw@TOkbyW|CPB5ADO?^?kIj<?!Y#Kn;yKp;2y`B*7I}Q7}e~VM;MK1bOQ*=Oj{~_cZvg6JGk@AkZh`301L!t_Ca<$@iO6}_Y<JyEBlsJYwnY7i1Ma}&O@!iqF<fW@h|>^P?eq_6Wa28kb@te+0_NXHV_{dI_T_hChbR7+}Kqu%vri{F+Q)1TNb)%*cQU-^sYi7hD}TQlR`7v8Y|7U>dI8Amxqp7I558|>rCYO=_+l8utVEWmK>Y(nuQZ7z?b5dLrwmQ;Of)W@Ljr%jCKVfrONa&u(aG2Z%ELVqa}Y+rpBRDUMV*d3U4NEm$C$1CdbLp*E0EUoQ}grC+=8q=5i7oetiKDU_?AYPO*X!If`7$ez<+jY!g98nR}IfAm^jme6$}mltju;s5s3`EBfS-f>6N+G6fL*!qBH&euRFEx<l_l>b<WSIXw`4V&z_KL0mNUF6<}V9Qg4}#c0^XO1ER3Q-VUcgOeY03{LWpt6|tC*Y}=1bjz$9!54K=fwFUm9;9gQ6{K-(U2gepK`@aCg4n{Ou)pc$81&YuMXfI3(mBa?ysiRQ=<L_`FOcU1=JhabiV1n+8fF;2fA876vvVD*rL&BaH`7X*w!Iw~)ci2M3JujNf7&A!8vEsyeq*_Ck>PXCDK;6mJjwteZ+wGlm~L3&**k)5TK@`W3f!KTj1#qGGoy3>2gd-kD^y6GloqEgpwf+G$E&?o(?gaym*-9kIT`GL{S(g!DTwcr5`W*TUHjWkaL?4@gf}L+dRPpagKEgWDy&8-U(3HEG~{FU%Ir%c>l#4Q6}3n*!!C!hhnNstr1KXJa<YnzkzrrrE~Z-^efLQ!+laEG-)gs2LcPMu7S`<PZnsl!%9SRBFnn^Vl75RisY0DfI$gyUtP+zPxi~O`XQCP53K>8&&k0%EIgajIEvb~-EkHhFNW%LFx@~W<p!V5DBa0oX;d6vnR`cx>hFSZpHY<TEyht$JVaRPCRtOOlV!+w#y?)JLpe9u)^K)x2&c0{nL;X538jIKuBcNI$(qkS;7Lr6mDymV4E`cit*JIyAQzymXomx&BO6{$aEY>P(5#(qzM$eTZKf$I%sw^kF$x9b*^MmnVM3+0=d{Fs7BjKcpthiJ%V_B0`L19Yv4>O5t=K39O2+g5~|7c*^NS9R+jm6>LXO{f_uJbv*63l|k5&%4z=$kWGL5ZI?f`0xC7XGJ;OP`@<BZO*1zcxKP1&_p`rV&rsxHcpml4^QFUMccG`-VuSb)<1okWon25W?~s+5ZS&=a4Z)uq7URmV6c)i%BZO@0=)k`r4Yf!v#;ne+1<W2VY>`Zacuq4;MmQD!+ZwW$Ec9v*5bA!lIy0fxt2Hw0&Lwp)M%{)_E>10J*&i&tL-`tPC`6Y!~!l@Rb8pal`Ha36A1la9-M)k*c0GPmhfd<q~4&sIG*^oJA3qyJ=MsOk6osl@?#K=h>immuQQ`<Z$E_L<H?$KXMeK<LPXIwQ+%=0F3@%Vt4VoFR2Em!5WtS>ln)?^w}nQpfDASQ8k*PXxM8w_#IW6e=%Ss_nIz4e{BZBz#${$*l&w5ljs?U{i^jECtf_1PK}nyXh6D?b4`DwSU7S@H46CFM7fywHbO^8jBDD6i<1w6Xd$Qj{As#-rO^pVb!YP8Z1Gj#{Y~@bY2mPY*REGE_Q*<JR#kgKNn#BUi?mNXsliemSJ_2(*CCaZR9#We)9f3V5;!zjVIss7$(Cn%5entysuyc)52^c^M4=#^y659KEE@j=;+`QEq4|)$8i1{{kEFRN*hLIkx0`tv1^I_lUWci_Due9s1~Bsh5I<48Q%WBH`}KAKF1a687M(!iwqpidR_`qhP<z_*?dXR&Y@`0x6M{&3TmOjx>@Nl!k8m>xn`@9Kp}$>Hr_NcWJQAXX2ds>JA@y!jG6-UP6?zD_Gi6^_gR-1;rS#^m%M@DD@6CN6mpNA4nuFI!wYEBV(xw;1V}3~%it-6ga}t@&fW``O$p!5nnwpoteC_G|Kb)HxqSP_w$#(F>=Oc(5FXk*XkL<cvB?|MxM@vsg^WSlsHW(yNYo#c|y*!lAIvA>&dB&C*UeKvig+8C%C@Vdr(WF1S8wLfMC;1mT-z+{gM2Wq<@Hd!MOku@5%QR=@-$Tp)v>?w_uGvx~)=bIL#2O`@PtUH{ZY_ZW&+Y*Ft{TB$)6K&G*ax`#2u~t3=Jz12TUr3(@+G{P*rgoF-N)?~(25u4P7jn61^RLKv|1LIBviM}mhix*70{|2-s#XuG1NHZ$8oB(KC~|+Iid<f^Rc=kL9ii*d}^6wl_^v-t{oy7ep$PfSjiuER0KBP_+U=Cx%z>Dw&pUs`WV7))Tpmj(@8AcG}46B{sS~Ga$Gn3y98A(RgmP$OO>Z%wcG*Gt!+1MVq>t<Vg?}qpCV5X-H#BT5f#{A#U?>&H1%-vP#`SidZ`M=#8#P44kLB$ALpaE+dn*um)wpbr$S}A*U~eFSQIaxfFf5J3PpLIHle8k@;$@>;E_-Wqo4J{@TmQO{NKLtdR!`#BX3H<S??^}=W0ipNo)vx(65xQ3BbZ})`%)eSN3-`J@Z_S`KlH<9K<|9Gys0PupF?j;sKNApYIJa8&UAnY<XE@j_tDH-{o*0#BAR8K3IW&#LdV%#y-5xC~6t5c)Vnp-q6_3Vo5tBEAwt0fmAZnpW?3_NUe!XqL=n9_FzW-8OVXUxvM{jE6zL+5`=Svlm8PMC<Dj$nqipoDlkN74L>Ub1)tpzx5k1#20)wXU(nlJ3!7OB-e5!{S^XKTv#C(<)&THK@ZFabRdm?y6lyu5%f-4VTzs0_KJ!t4-J~hNc%ak)0rf6JaAZVhEzha}QS)rDIK~dNzByjN-(pEGUb)kSl{p7;gcJoyf{cuSb?~z!b>D9(J5+S&lof#1JB~4H+ONEuayim4F>ahoe==ja4|f4Nl*^^{e~Drwc<gH<iD=Ve9TR4Bl-W3~QNZUHgod10N@kx27^k9k-&4w6@DP1ovEQy<f!jb)JjS6_H<@ol@Qo=e=BdDJTd-o;477Z59vP^@(TT>_jl{rw*?e{q%UW9Vx3Q2~Lh@<8=6&sCI;oZYwo_7a<P{o4Dzk;aM;B|3#LQoY56<z=m!|y|@l_9O@dY1bpCT$lO9M%rd?YLA(#g4Q#ERARqb7U46)v1uW@ROWW~87^t_E(_zPnokv$E36+=LS$lFK}5pvJo^_b@*<>*Iv06e7>`&ER%c{Cvv%#9e5L8rp@M7MMD%sc@4Rknxj+9Qnc1=K8D)vmQ5zX%KNu(S5{%{k7A!dF$coJJ|oxFB!BUQkh~?<$#zMagj*MrdgZZv8z<%2Yc5owjYwAg68aa=F+)r+H0nuLjaj9Wfo4h?BUi8PW;`uk_|$8{MLq2(QjOLfuwz47ufuJ=Xj5{T{gj=Dyab-!gsmIOJ(l8jG7$9(o)g?Zti%INV{qVMDfPgUJfn`^Ylq2U=N(nb+$vZ@sD)~H-rBsefAOY&*_r32*&jI8@T)ARGrrLzk<MUgK-|)5M+MJy`mqAcT%7hG&mw!l{65J4q_3V>hg1LQlTIsOrdrj9&|(*NMZ9HatB{#!GESKDUD_(FQ4?vdd8ut<|~I8-m{A1=a?D%OXGMEqHnf5nwmwmbHVC+ZEYAVJ6oMtPn8<C__{K3R_Kqm5o-)tDmXJH;F!U%Mu7=MLQ>2^A;s^C%%@XnC(gV*6ymbgP4&ABsvDgrosLGMj|<n9KuG1HAp7rV>iL29c30A#0Aau#t!hn^tyW~m3JZUp26^!%G;bw|q6_0nz59g2Bo>UIrMkNb%qx)q8{+=B+D_#&)~1FqrYJMAVYDbX7UX1ze@qDt&4`C-O=3bFxa?l^RehaWmK4Ds32x_e)9O>_Ns*94ncb~!jxKp&Yqw!l4P{mZO`zbr`s|?A^E^6Wov|A>Z#mwxFp@F_PPvyP(<U6fr4p2bFH*`6H_Lsa2jRsCmIoq=ehhXjYEw4+u1cIbQ1a}B9N-Pze$hWNA}lXeW%%{l<HL2y%T|ryp*qaivr$OLFDjyfP<!BJ=mEC`+vV_c=7rz(a+Tp=P;1Ck;gm<pt+n$qIeJry2)lY3Sc5JIfmiSfhVv>@Lj=6rO#UFOZfG`=Q9mbxNlL_=I-n5|+{9j7yb-5QsGsuqzKDtWXbI~OI^))El1*QMJ6SZOdTS>ggeoHShE@N)sTRy8^`Ke=%z3=P`X_yJGZ6N0zM)%J#39hX@(=V#>(05Arp0m-hgFQBG+FcTugJJR-U>#%*)?h6(X`%bLebcT)W#YQ>yfKUV+iJRxI5v1Y_C82@6TWwkg%fY=qg8a!_50`Vge81Qw`B~d373)06scTaTT@`+(#l01PHRdS<V{-)^=RW%EHyDus<ySK#VGim;1^n(eI8a>9l`DW$ttP2omIs4g%l5j*6sob-d#q5@g2Fn=T)u!H<njs+iPQ)MJ$wF*{lRi9GtgGXqk97pX3RCXaMfIE|np;xJ{6qe*OWls|0FL5RkrR2jTY2^9bEtR2pU4;whcRgOlMl~ip$CW3#T^?cym4`6w=(#kzr4L>q7n*7((JJgvSYzSA^Y(w45-3l;xrk@@&H3QRNDSIHEZHm6%vvhn>W_0L5)G6EM<`DrT*Bb=Yr1`>FX?QRUEdg}Ul5lqP=QA=+y03iAR4pWb?JFIadsbSJ+a?X4ct%XVx~psJ_oip+V$xfr$ptW&pt=ALXt~kzsp@n<CnA7>1cn9{rG}DIvBqOeg^kSgTJ6pjVppIoyJS=|<HZUDTu_-s2|K>&UYE@H>@s*56JCa4x&;T95+@6_?<uX9l6O1U>$w<(<e7~t98W<U!Ej^lS+~2(A(y=J9#_Rw=y#=B`uI4jtcX)HtX(%vj|oXYeLNJxXzmh{EI7l{8Whx@2!WKs3P~B;@9;1Gp`is6ft`&55ZWX0P&2r^Wik*aroh@xnq(AWU$RPU^4~`R+@a`clf>OCDkzUA5p3gu`N~JtJM0f;uJ`ItSu!Wf*oHqx#yE2}3Us?~jTX_M0B7ZhU&yG@V*iVche9`mb4J0{4KTm5D{+gi`SeAbl}_s_BqUW(pE-kt+}|}1g!&*`2a>g-y@J%}$s|FAZ?i&`!@Q%_{+^UTabf>ppL_ujysil_O=5pm(cYg?kr>|yizQwnPS8{Ck9f470YCebEa3*0nxVTh9<dWNqkf<D0YU$3<Pjy#+-VFWZQSQSujIfATn+4Mu0QNhktyw?mojcjaZ>!KCjM7-2D<ws96Uby{t4DR+miqoyXO+ynby!Ek6XpY_%J){Vd)TnhaCHa4i$o!9ZwYHTQ&$GivOc}Dz^dff;JK@BxHXs&v*08Yq^=aZo9~0y)<#x_Vq!yNSc=a(3LG|h?v*N@`jzJ@f4Xy!b%8t``%M&$vc~VAIfaZg=5<GS_Sk)jws|zv5=wm)id6aaC=z}0}7>41%RQ}9c1zaC9!MHkxY-lR9E(8r&H_mWl_h{aTwMolybiMQ}o?`$j`R~4*l~25a-Ge;D|6}B4{pY`u>?`Uc)<G14zlW1)GbCc-;c7gXMim0mL4+&p-OyOyiY9OIW<8e*G7$P&ss+D<!r)U;uuUq(DO9`j9d_H&)MN?F3DMuKUt5dENtm^X$e|Y;cYF@=-kFKNo$aN49oZXxUoiczbE~0%XNteT-^#BZ3Dx+`H_#-}u9|?Vt`_=hX**5e7yHs3YZKif$~-f($4*R8sZ$AK2eBWBXyXqWzFfdcL>|-w)=bWTp;?`g@Ihlo^3WLi8cmEw><VZ3eLZ<iaFOs_saRMLZjQ+R>uNx4ubp*<#Fa+cwbD5|iqhQx}p#xn1n9v!Vl2!IC_IECzDGZ@X(-c$2A>^OUp}s6zBjtQ|?`>aoEXPT`U|vUa}l<vCJhoIe4^7n`E`^3yg@Y5MN#TThUo1|DnBZ&o0DmV!(2B+wMNtr=O`zy#WJI*Kze1h0MDItJm~)0D-~Y)0!@kRh*da5N4EbfExVKz3nPMlvS0G(9wVMc)l;0g#^L9a%Ggho_R6!Z?cO)k`^-4R0X1hkrPd7N8et>MG_B9u%aFXos}Ddz(%@auD!}^!rFB6x(|ZJ-Wx<PaMwqqW6u7a;Tg#DHgRsy4bjW(ZSSq7mK_63lHJlP~|U%jt^er*y&q~(-v?+3q=uu4!+dPgm>^iMpkO+uNe%D;Z(~xBYFsQcgH9i+bM78c+6AO7(rg!8gVMC5|`*W(a<=}uA<|xaW#V_Ipd)u!&qHUKAmQ&@~-1OXW0mU$Z!&@9C<;744&}GpG9{^`IsuW&y|N;6m}*Q6e)&@)ap9_rH;I)Jravz3`}mB%qzceJ7$46U%MZ)<Ye32vpNn54@aFo+7>);ghknr>5}J69d10^%MfgmZaUmy4$GgoaBTzOT)LY3kAV1D@ee$=V5QU2;N+}}D)_`n6@2&;GOxmC8?$Ltjwi)s8Eu;DUSf^q1T=`4d?$7m2a6`tN;8-l%xsCPYcF7FO?y(kx&+f+G_;!*DYnC>!<&?wOD+hh{@(HOUN<%o$*{Wkvq1?GbTBoaf;jCo=&%xfe$S9jnOOHB3#PveU6AVZU)pm=DC=)U^TdXFNH=zwtw%Q54hVh3rq2yyk<QqK{yu%au6r2Qxs%!INIr4THTSK!3U{A*`A9iZWUHZhD+Os=h7Tmo+nPcr7@|h^pFvCG4oXFU=r!fDTs3T5?|gjrh(&0%*5iQ>Q=V4wJ`h+eCz-@~d}^0$)uE|`fE--~Y2dDKWm)!S73Sb76b#<v#|@+o9gt;TmL{u|VT}U=?y>Epx>FlPAj*QAOI{YjHg)pQ;=c{Q<EDN*>$N&P5BK2n$Vzy}+H_^V|96lD-EYw8t^DUJ4Y>CujVRH1+q*m>=m@Z8hi;K>B5>>%!Wd?%iV)aEjL~RcQmqr{8ld~0H{AR@kxcgIsuCZ682}e^@b{CRQKJs)-bW@pN)h9zyqHMswkQ)QjqGY}v}-JAJg2*g)!UBf*{bQ2p=}R_4TPP6i2WR@iwyuOPfPS|<LLl)%#%&l36;?_YL8)n$+qV*`Niqhy$O8RBPc^G4B=8Z<Y8kmawYSncwMC7%d*GYY(#LSEI-*xWC&Pa6fiC$AbR>tEB5f_V-v=Sp;t@KmoBA*4|z5K$kk^0gtL~(5%VSAFH9dgHCru~>=An((3lOl#S$1iaW!NNg-np2BWy^E_54Kwy;{9339We(r!T2=IC3{(=LxfVkN4rGoX4<u2b7vS0$CEX*pk4vX%p-*+NA~tacz&&UP(p%{MSu6JfsgIQ(eaU&*fN9K#c$3(>DSq<YO96iEbE>U4vem@D6hJ0F=cyG*Sm@oW=o#oGggMnwhYv_&gc5+{STDEvLa6^%s+OC*^mc8Y{xPDdede`QNZ&!2n=VTmaZ*#vUL@QB5=@-wnN37P;xzukij$)WlA3_rc@301N9WzNoJ#8vsN-S#C{Y7Qh91z;16>_^HDRL_*0WjgKnf7Kcv_wA==C(MN@BK5Ps|?yyg&j>3MpR0VGZIhK>fV1q*Wcf5;XG`?Gg`t!c>8h4HIeH28M8o&6RZXk?7V++!B20tth^DJCh0~O+FP9A*N-Tck65K1Oq|CBVX74M?g)YO7~yi$k?QfLkq9iE18*C{fvml0I&q7RCwWk+MTWJbZryNQ%)n$jy7U?_2SbbuzWFraPa?hgh}GHZ~JU7J)~86zSB-VVV(r)^1^SwxM&_eqQ=Mk9(t;6ismdZSbS38hESr@4;$q14SbT5i`X^(o+MHvGlv)d&6F!x|xuMaEJvlF^ic+Wi+q#AY{7v${Ck_%A<NvFUzE5|qH0+jyJ{J+#WJVv@ry#;##q{2KzK;j+MlcYBMF((L-(ptQ^sw$KC$1jOgSwklyYp`@`c46La`kARAHAsoKAVv0KZ*ug={&uY^FvL3!M(SGO&Ap!F-vzej+{GVyREec~a>i?VGY~!dEzV5dRdp!A=Fv#EFQaHsCRN>%v^x>D>LVc<#M(qr4;4*4%j#4X63Gq+fv~LDs%E`zDE|`+qa)e!d=(to%_0ZQq6p!7eJC}Qtfh-`9Vx!+S+DU6ZgzMEH@C`G<5wQ~KX+6VZD17%CF~$$gTYs8Uu>rqVqW5k6JH<v&-z%h%sqa6NAFICw+t2XkxQlK*nYY%ohFJetpDV{~Lx4NTZf}@Jp-R6k^!}}VYLxP;(Nh**ZeM7@&zp|&*6!+SkhdSUBtlZ?=E#Vaslk)baVXl{=mZ|kPh5T}{EjuWw*pQ$q{arqJw<4BI{1_RyoAIqxdd~+s1Kl3rcAJ@vd<*=SYVqo2o7E@7obm<&4MlEv<K$jVebpvZ*Xqu$z0y>YNUc#JvmeSOAyAif}52hkwB~86<|-&#%%1N`x_?D1CNazi`Ue-f`Z8T6OaHie=$oB(;xdP{iq%(rEZ+gHlb}}hm3=6RHys!UpOr>aj~qyiI|Ha)9OB#Sd~|eSJ?uQy;{Vgs~pbIoj5O$n8HYqO}-eC#gQ#3QU_E+XO9_4+uwLBcKzf!n<Hu>_*B%^EPJhj#Aw5_tkaM1v4YzTQewXW>Z9p^(QX-M5wl`2lq9?u$F;oL5W1jB?pP8Y@0>=sbc=ltgGR?CbGQd<HrB$atNET2KEI^9kPt+*GLfceqR>-Fp&WHoB&#aGh*Q=~4a>FI#{XEs{|3S{QF)pGE~ZqabZ#pE16jMe{bW`Z%Jzv-puhi`^p146M}KS@-yKqaP*Y>Q1q~_%S{enz0<3?C6vAlR`Kx+zQsk{0yoO&!Nr)GDtr&#yI*ye_2ZPFe&sF8+>+Y5kHRWs)cwR&13t0rXR-2%N8@KKUBUP)Vr*HE64)p;<HOZq^zwQ7plN4PcyacFd<pSXkfTm|atip^S2O)~He?XFS^8CW%#zX?p6xT^k7O0#yw$=sDb2_UxdbJ5Y2nJGsC}9SbdFlH}G<@h^z%<xWW0OCmi`X1|@!gPd`Rt=eo3Tc|Ae}ow;yFdPooZ}VibKTU&~KL`*lf(+TfVx5$S+R)VLu5oD1V%`G<J34%Y^)4Lp)n;mH^O?st=4p_S3h5p-_T9b@q@U9G<0NIxD$cGc~rAOl*Xrtm=_U1j;fGH5e>)@evh5JIBig9(N=grLJL@rcS_BUdaX8#zXFEA+AX+>YeGZlS&K3eXAYBVO?z~q-*jR*O2|CAn8?I+P0{JRKqluwJ!kv_~D1}!X>*!vgjx;uA>%w7wiOdHGiD5@)ctP`8I`>fbdj?*A1CB$MlBzvCs!#uIp`D?tDc%WG#3piJ_bk1sk<esC1EqMzM+!Xo2uv;ZL6^Un(Mh8uA=SMK$Zz93_(DwVbTTL+MxuRClr5f3+y@_FHe5Aw+x=w;`hCo9Kyh?bW9&M|N#v!;e!I|6ngBvB$6~bgICCE+P%tu>HZUJT3aW)wEa}1Cqy@YY7HM^3&=U>OVcr|2HSzYZ5rhGVwt#mmBJ<m~+_Ts-`Cr=oL<(-KSNmh|>1<%^*!e|Iz<$T=Mo>$s2V|4nhgv?yMdpMEy0iafCLRzS&PZTVMF$`13zPOg6uCYKDAdMi^s3)IFu?D!aNP;3TC|Ji997L-OrisKkm8a#0<^AI?9e`8=0#dD+qR^Uv0F0OhoLs0?_L&V_XMfWE^{E2Nc6xbUgAC%J5DjoEVYT_I0E+p{p%G()F4)yDuQ5t6NCk@itJv$KTNVE$-d5m#zlKL}u)SO`wjN_PT?=W@9cO4via7Q6IR>%bZlvUUKaYYrN$_1JOQy-zdRKny~J!3*C7@~`BCg~VrC*(<NnL>I}fu_+Q<)`3nWAxCb%Ngb-Jo`8%ld5~E-5^Baj2phy`rI}^w+$?p_b~t<@N^;kWB@kJs{ZqsllHT?Z3K>st8D&OtR{coWl)^(^GJF1#I52$Ny`@Wp8GskOXo<}GqHc$QRa%LuqWk_u{u@1u(p_B{vN?5XJRatsFZxLDTwvmG%|=0XS@_TDxTb&K_<cNWi!!LpcgIW{xX@ShyLF`^C{wY(f9GLMYccDb<dPDB4uQq3Qb>Tg@sw@Oi~ABLan~bL&);y(K6rhp*p_7t)~z*u55!%lBcFaO(fcJ9A3Qb96~L5)b%pa`ez^nPe!^eE2Xoy$(TwAAvSjFHyxef#Vs|cO&@~(EWu55^fw#GEH+Un@br(OSQ(~3`m%t$8_&r_MLwB~;6fbUO{IBiHzi4!s#{E(ToRLl|*Nf<xU{qW;s)3^p=!9j78OMff_g#|Ukf5A#)Ll7V=q3VY{cCdI!=En}5b3rf;M|{eLrNl87{DudS}!3FhS~#4TDV1pB;GKT6%eBdrJ(fLrV3#ss5NwZo>2FO?>)1*ZJehP1{MbIy(zI~+J^4>Bbzxcr|&Sqo_^=>44Gkb&LU@}qwGieee2(K?jGiAQE7dCcx9uq(#In}$kLA{ltY^)dWR&~V^jQ3&7fj?Xh(KOjOMhly3eV@Lv)K(4$uaU#LKJtYzTkDy9eG}0?z}uLQ;f~q3hO>^M6~m-*S9?7GGe#{&62u!t+fL(kq0f(TCXuwiwO}#G>adG~BbfrA98y>r8=1B3ys?$o}Sh^VWhKrk88hi$7!>Bv<q(x5<fZ4&yn^7S^$+{7SE=blf`sq|c5n%1?-}Q_wom-Qfhp=k;&H7aRTIWv*9WWaHff3OTTl>hZbo9d2XMz%D9ordT3uhQ0yr4?gHdz-WOTB8*{WEc~g>Wf=<cZ+N6|9gX}Na>e2_b;GSgaXQ#jAV-dR-w+(frTdJQViQ{dvHmedzuj&}z|Z0rp@;i)QJEH5Bw@@mUvR?K0T6rOF_N=jrXSyZ5@tMCG=A24XMr0RiWe$<8;IrBO=MkU;&)lpEtZ#088xXpPAx%|E|z=*Ou#@dQV3z?2E3V1dyw$imgb~C73LGdlo%HdvCw`>_DSlTG^E<dBxT#*A`2$XUvMbKdHXTT>UWqJM=pPZF82DkwvzerhIAonq<;ckydtuN9HT=2UotNAg~~+9J(VBaPCC&s3s{)7-Ro7FZi+b`{K*z?BCD6?e8qE`a3^dPv{=?rII!x09V3(&?X?v%KUxLDjfjBSqp|!#WqiqHO^G;$*_(l*#NGcH8%y8Kk)UU<bVB(2r<P-S!EP=D(GWV=sSbmQz<E?S9WWTN$naTcD54<+WSv^~l%_&VUyV436M;x!^zI(&$c^<gummfx9`I0O`lj2<!l^W)^`6hOPDH*D{oXC<^+rK=b{C5C#Qa#=wH;>o5gO|{KR-Z-A`&0tTwLwjAF|>EA>pDzig+z^I>IP0xnD4j!#aEmxS$mme-8fHr{13X@o(|`34PnAypn5DBRhCb1^+uX$6TT^(LSeL>om~>X4VJre)|V@Cl;yI-VUa~5F)KcOt_k8Ytj*R@??;MXA&ziV!c?z=1jote>c`rI!_BRF3_HO%dtz!@X#5zR~xM*+q=e}U%A6p11)<b!INVgj^PC0?Nhx|jXE@{Hku89OnB;dZ_7W>qjd4klgfw#D9WyVoJe-U5g9_;e)TBTxXLG>J8-4DydOM@30RFE@IQ#(G@F|v5<74mVWklD&#}d8Q}Ah>iQk)kNR8vF+Yf6Ennci*+JTJ1+!zDW$6NU)P5H}87{M}M3h5+@#u%j5_>8(BZT!F^nz$>K@HfOS%$q{bC#gEe0!g%iWrlw(X{u;N*zL)`ojv--fe^n-wNfXiw@^?t)cSvKpA%to?mS6+`_+L2%v?_7P7+&WBy1ge!w-FL3<W1s38HQ2*FtO&gcYg#-)vZy1)^X=oe-=Uu2pL78BZ7{i$5X2+1rNb-!}z15~j~=YI{{d{>S%#Lg83JHjY%p^&YhmMQt3BN)ia!v=k%AxywNlyl%}uPDO_HWD;uca#_;Rt`=Za+WAM8(RInnq_2-MCdRH+ZkE9W9%i&+Two5S7x%E*L;B<&EUGj;|F6n0Yj@c2BJ2Gjyv#AG3zjtz%QZT{Xy5!mfs%o0=52M?>9*UcNG;eWOWjcLG79jo{dv^P9^z9Mc$yAkqeGPA1m7v}m)pO5fy?x*;Cy-L)NOFnLKK+r7G2-(UM45F(k0z0&h^^;=~0&4>~-0O>PAN9NhvPHH}w;HY!3H+OOod_tcgoDbz4qR91}{2{))<Ixj6-Szg2og+Moixrn1re_AU8Q5>W?GEFI3H0%XzlHs&V|SbPJsQt;I41ChHA*Pn<2hz9YHbDYHytZsDsf8G#|kR`?#Y9D!*Klv<tO0O>;;*|_e)^9c8{^A!TtwuJ-g_93AWPfG&{ly&fb$i;KSbvVY+1U$Y$upEO|2kI|>$ouWo07bU3oCK~$`3S_HRu`QN>)~$sM0*+eil|Xc5yRLUvu;u5YfytPO6{F$tg8p2!15UWHi03Tc16zD1Rxn)u({vZwkFL0x{MnFj{BQX1VnBBWGj#kapzQ9rja?7HUb4BDZ+!+^Mx#hwi{2GeE05oR$-$+(H77TO=bV5dWhfiwZmUwdiP>+}M10(B%Pq1IJr|`S&B<85Nk`v*)%^oPHeX+C$y|u^>v&iSHfT=(rHC3#2Q>=ChFSjmk=ls|fqFSPl^NAe_?IODV6B_-~YGK&yVTEH{kUfgs+0G?$PM`{;X4`+AM^$7&fK59Y=@2g6mhPDZFg48uLr4Aey%G{1#?c7+*}Ijs#A0O;*I-@U?pMvkG-&bw>Z({9QaC(me0-w<nu69VNF{+{K&^TX==XF=5c6q%qZs8AHr6YLF(5R<BjY+Mks1I4YA{4J$tjN)T$oCoWE9TQgQ%nJWpl@VNuwD<0>V+Dfc48fF72ukttngD}(wFiszCbFxIdIoeLIrLxkQ88jwqc*|TYW~Lgdm;DI^PvMSoPa;MzG=pZDP5SEpwm~3y(!RQQBbawxg=+YP<yXNkO)^8VR}*z9msB(P?|if(i&4%YXU&jaT^&k2r*qcg*=^|qIOBkekLqnPkW@Ad}R0}Rwh{_DP!G^Abkh52PbLpt?+O5f;AwBX*5)3;nLaSgD#LJxERc-f|Ng{2+flr#}yg1@6cs#c;miHA}^L|rjRYy=UGc`8C@%EIHosw7*sBH?mlaBpM7l0b%JMYVP1x0ZKasZ(SCRu`3fEg*m@ZdbaA9yPyc?YiRt;1a`c117>$BmSiisb{f@5i7rO0xol=bjDQ}=@6;OPyWUtqv6|4MtTYV8%a~|B2^u(YqI@T&i0n4V|)WIaRVPZmJzJ`UW15z=1YxXC-l90Jd9b{{f*5_xpi4uI&PauRf^C9)~(^_Se$oivqUQ>+C4Lp{*>y>y6YWRV8x+3j0X-viklHgGqi};j_+Zg_XSL;Op-EGC5yv`is?59w!HMN_685X=j-Bj$d)0otBvv#g2Crgj3aE_Oi<N8QY-JD1eGt?(7)6V{}aFLU}K{ali6a?&Be;r*wZ_$8bRk{L7a%kicFg{Zn6k8&+mSk)U_8U&RdQg{W;9udyizqyb96l~aR5Z&-ydH@YGPn{FjW+f}YjHh`27|pCCvS%@*714Y4G)g-O&d=NAeYM+m4dMV6kW=GL&bl0GsC@4_bQ<Tz6IITQ9zXHS55%_fqC0Q{AIB>W+V@;mlYHSz15PHo~ymA*KM$s6vWAgrc6}WtsgruEm@QjMmokphe*|?H-rk2fz{wj&&xuhcUzOBU3$~a+W}!a+A~IFk*&tK_4un1grOtcD$v>>rU&JS@jW`(44Mr=KH^NqqK4j#)n|)KQ$5my4zfdbvv@K$G~7tHZ<uNUmcK&A17ni=f&sIcTokmi{>hTuVx>PA8HiHtDZyRQfG)YXp`qF(caa1T6aDd3rwOH*?c?7@;XL%GOFgiNA5b~i64P%JpJ;oSoG)5V`P)e`$&A$wXfz?DFvQrzEV?3b7Q9~oA3}ZY`g_K^jY$+(@S#AUPir=s&pkJ3^lCZ+f-~`5<KcfRfz9}orV$8JVX0Ae%Mn|+EdNkmgy~eOqZ`E&g@qRW5d!@<yQX?3BQv^BEz@F*pFs!dCnfk374y6qXwf-o=YLjjMl~aybf*alOuW&=T_zaoKPJEXj5Au$>LCnCC^qL{>o~%sra==%-sRvNm!9?le9-<Zoifz@t<-{Pc`!BD%}QJIymSTzmDHNSuQQU2E*SKgrrHeLI<fMH4age>Lu)bzY%O@rGS1mHdo$Y0sdJ5un6PwHt^DHnJWqs4_$Q~3=jfdJEbL+~hI-uKl9A7g%4((iCp3lJ^*xJPDJTVpK-|@ff>~6YPg2pcJ;kOl4g`&?L4)Jp0Te<5cW{FTLMs@?W~8-!{Cl0q?=7yu7Ea58g^GUJaE+_;d4iLaj0I3cYwkCY?)7CbRfDaq6hVcIV5a~r!&)tnv+bhRD*rT268?3RF{gT|g4aOdk{J{*l7yMypKL&@TzO$ybQnakb`>+jQ5-4`eA_wmlX`T{+N!9W4K?B!B=K4`48S=`YR{eQBFsx%5GmUP-&d;^NQZ~a`tZd3Y4948Sc5ablZ5(rkLDyVjtYO-Wr}EZHc&mXF?ygK>OTcTf;#K`6G%uRVbh4DylGK6#Ip<s;uOb+>BS-YeXC_7bBW-Vk71!ax{2$xNQ*0y`~c@rFWwp_ri*)2#J1HMiTJyqO~@(Td<Cki0ctkU31S-&Jx(v~GISm`**Pc}T*g~5GWVf$pe2-J2L)hxxN|#$WI)>~k*|5(SxE5;5Xny!VGOZiMq0I>jcvCYC%s#{!J@iP$VQ7w^}KLg`$}l4^e_J}WiTIc9lF`odKmq>Iqefp7j7V1n0h2@I}Es>jwqoZA*ZxO!vKqnLd^JvQrY|mUq@5~P}e(7-T#IoY?{;;l5v?jJv;H>1@frH-(3;ial_k>bq=+{*re44yfeA&VCCihf1Jr;!doGq(xH`$*oK!=x|AOWy8}`_rH0h{BT>9k4jBT~c2|9AS;^pnAa-)V8g#lsWQFUW7o<dJZ{5Gl%xvR?5HnyG)c|A{+g`WwdCiPT{Ya`%*^(47WDmjEJN_C@&9r%|a3sWUobad=mH-7e`>w-X;BXyd4AUH`j!@C<s>}q4C`d-mQZtQtRr<@^Fz}6F(^dYe@hfUhpAwHBUsYM%<nnfZHDba*&{;=3+$eneNWrdwjw=D9%i;fqj$J-;0W(9h5rHTa1t|!P0(7Vyk-3?wZq;-tE@B#TkMdr6v1OmY-ISRuMM-N$9eT4t@Qxt)$U62_9Z2BLONiv&4>$Z1kHunsa)l(`%whG?;=A%p2B~a(Kw7{_^v06j_#J6=5EQP-2EYJ3m&Ehg8hV+9ZyP@5)Xs|PGQLOhN&-o(B)8|<yX5i6k~JoOEzut{KLDiR5*INZO#S2yjwngwG^aiJSlAR6T22Cr*-7lhZ&G`pUWAO9KEGhGzp7|DseH6y3;XzNJgMCqIW7&-+21hkU`qH9F`ab=zN*gIo59DDag_!g0WBA;jByBY-~{m-XAKSB*ugkOVe3($BCBiBiaX6d3_0c}(AcQf&46vJS}LKMC*6-(D)O&#_&w~{Oeahn7X6re6XuxIbJ%9V{PlF$5h~Cj8bT)o_F-kJW{fn8Oqttu0IZWyd}FHxbJSu1%sdE+AhxqjC;<%hp(N{rJ*XM=ky|vi;ZU@(tO%^}0jYm-NpdPf%mj%hXUSVy(?E|4q#ByH>~Fn&m|&}tLKeuz0E;Uhe+?Htv^{i6&aZ3&)c#zJ(k|{45bemG$O%lEFYE%Vel^?w6?uDgS{ONpC*<09!mw_8*HkL1`{GWXy(Tifug0*<Mzc4Q6gYw!9a`OvmGbQbln7Da(#(FIB2cAtzaBeht*U&Kw6~SZ^ia6_YmerGif8z{Q!%$hXcZU!4P!fmT5KYYRJj3zI)OLuk9i7FqCRuNS!K*nIhPPJ8)2ihXPG6uuLaV;p2^-PFvgNkW?0_<S{E-a8f$sn?@rt2?g<%`7$<q3yD(31VbAlj2$uUt9=S8{BN>uCUH*(q$L8d__zYYv8G1hUg%LkQHT9aY#7OXXHI@a*DgZ+kf}?39d2J8xodD(smL_apE?Tpr2SnhLdcm#2OdI6Yfl*9Fi`Bc+%zA<s#H1g3$a}MavJ`n?+PPk}D7iKspDpdavG9HzI@pTCuD?iXWE!Le?aw2PAmd4xr>zE>m?y7k1_>0yI)^4N^nncS(lz$$OD*GZQ8zYHoG%BHbdX|I=9y*V<qqZg>!)lKwyT}d1pz@-h(w{4Gcda_7D9mZI;w;Kb+fXxz4mZ`J-K&DqY?SmD#e7gTxWBqPTI;YEZIt&yS#Wuo0~$^+J&7UJ<FB?TXcv_7A*6;b#3<?*~TDYzvAKVS_Yl=@3d}<?2tts>f-!R?N^4OhGp0v=-s7-j!hJR{D0vRO6t%s61P>dPvYh<oOMVeK4VReH%;B*@T+T$BO}uOQQO}z&lOz7X`8Vb$3HzrXwlO)3PNfNjV|uinnI1@N^rF5@U=LvxyiA$4;6t%tn6EygA5apWMC680Gt)m`YGaFM|Y={ifXaYz)r7txCP_Y|4ud%=TxBBr9N0V&b4z%Z>bp0hqx@9^rLA1DD^~?+r1dtfD*{y1$!zov$pbUmdC52|AvhalJ=xq%>v7BGShaRC1=?aioDsMshzXy7d!nHt9sUPKMunetG>F6{w)_6PID4A9Kyz&RjEggl=Ved<pZ2L`_P2K$zI+5;4cnY2Lnla^<83yb1Z_nk!F`b<aJX}4fUL5nd~MJ0F8b4_PwNGtNY3HFfc(6?N;926$i@lZFolVn)<N3yMs*eY<ZS7G&wB(E49P=VrelVcb#RXGNC|CuZoqcD39mUtE}?Z=JNA!hPtA3)&ffu$VFNx3tx?;qcNY_ZY5Czm>?&mJNTW<zPeM@EtY!^ahS+v_G%V;-qgWN6mXl1FZ+J{$UIBPP%MsS06XXJ<EVMPg|4_yQ$_m}*_i4EZ;-rUNqXK=LbS0Z2Yr_}$!n!9ng6I*{QsWkKG=eeM~#v#F_kRoBtD{C%af>ep6^=`Km%8>2%W731!66TepAJ!vHAWptq6?E;y^5U!nZ3v`y5d0sfIF~(woSwPElNZUZsOIJ0u|)jV2cihZeO!iUo`THv~MC)>mE=Rdo9kx0rvqt_Z}-?G*l1@B){zD-x&qFTC@Q$`J5LC+#pvL%oo3ehWt7<md#99|9B|=xq4Z=CWB~?b$l~-enp@i;rvTbhxr)G*^`Fw+$R&PZxZuj6hKyL|Z>HV~VXJV2W+1Ir$ySA)bd86{GI(<nFjL8|uZ*<k-?9q2!~ALx<M_LSkv}rt3so&p-}YSm)++_eGd!yAzS^3H~oBAeGyv=h!dX;3qmK;S#ht8Q~ukNwNZH^7m>hCpV1iOFGIvb8sd9-4bn+*7RUMkfdZ4Qd33zo4<3=iyCTvpZ#-59_Z2?);ks>3iD3nRY4bjXd0A1PSP8n=y>KFjSE_?`t!^p`+MR9SmwZ`Th+775=G>x6fQB4O)V8#nMR`32CbP#ve<$Q?<}DYTYI+-1E~0}B&4cyV3&3B&QQdkLJUpCEC@l*u1+vA*Hg)R=j3=f9#kK(d)hZ$v3D;YHIK;_fJPj}od_Oxv&LPGDcpi<@d@c5DHGCED3OUiThj@k?egF7>S=w`jJ{ZST`<1^<d<iidNkRD;x)QD_&rhJddULVu1R~BK46Ij$zS0N0#3NyZ~xWDt?RF)ROwqhEFZPk)U&T!tbij(9{Q2iqa><#s-{c2Rm_6kRB8myD?vo{G&f3*zJc+U(`r9orGzfZ=l_3=Kz_H3(?%o4NmI(wT>d0fx}SxBl=UPkJaM+=qZm*lZ5@=wimV(1N7PG9fbM>Z>4HMkJMip_?%|s00ov}Yg;2|u&EaO2`RM%4@WL903ilf$CN)k5*(G2NSVB2#9MAsM{PZqo_Lh^Ka@90tW>V!Df4;bcFwEdh0X3Wru;7U6>SQZ27J;dN7&=hkfbVJp5^8-LLq1DCjTxqUL);cdt|Te@hw3F{owRCL3%BMSstV>GM>8*Ko!i{KM-iA_P80;o>e?ogXGnG-c0o$KLV9HyCfi6#wOrc<1>#z=Xmg#kqbHRxxw*An?!9SOP$?D;Bdnct^NK0scX|$e(bK3+=-p*A(MXUlE2)m`F})+Xhl&tXh~to|8cezs@ZUDb&o6LuRvND|ht??HuRi(+q8N5~J0&-#S<c(<oh&xB9Z&AcqbgW}yt$kWsOXd%R%%pRx~kb<VzFp2IX&oK^Aw9LQvXesflou6Abh*rrW{My0yczEh_txZNR-|!&XN2Jk=!jH?YE6P*J@cJYFZk_6x6{Vw3Sh?L7llsroNe>o>A>|s!L{D+AaGT<MTeMx_(&xU93t3;J)n8JdEChpEV0iCHnDf7{%k2sYI*%3kS>t@4zm+`&q8Q8F*qChq3k{d&IOxdS>)c?gnI;1B_y6>?PWKsPARAI)NgLt9EfH+;V>RdcX+q2`BNx5FDSqT6<+qz-?N$)p?)tJI!;>!c*Oj+CFA?H}3<T;&cIZ9ikQ0ZJ2UfV-#WxN_{7lPWA_&X^Po*F{caZm=&H$3;BL^x5B}f;LflS67v%o89(chTo>s;?;*U>nqhb?K}F0^!&$}-j8va}iEH@P<v<hKNr4b`-l$^|f}M||7lm9<T;?oz=~d&!JEfc&<!U&8_jlqV@J<XbOnJWTi4p<~Gl=^xEpq$Qp|pd=X=jA~8iyOb&Z-2`@1TpZE({0B`=l0H<(b&Dr7NJwy*c!3#_aCk*w>mAtS2gEh723lu>HKZmqf_E+(+mV$kcO`?XVROrBuC(Z?r&__ncB5U+`u;|DWxxPsrHwc+Y+oN-n$bg^-Mcdq^pa3q+T)&Cs&pj4KbP_Q&{C(CTU^!J5ZJYw~qi6K2|VqTG?2Egf@@C#x=e+gSA!ih_5=#9x_J_|cK9Rxdhd)*Z3U|Nf^nf2jG#X`DV<AeyAw6zcN2XHHWOxPmX1$UO(S4X{e=>IKWDj&v&@>qSms6HUn=ceWIeDvc|ZsEqH(2ihnx@gLCbD-4nJoH2lz@dfb7g#h{`EeERXpZtC1#N3e}T3dHjyJ9UBLc&;+aDa7shHM6dbgF&Xyw%IaC37}LxqI)W*#D7w9yf1~_++z4Z1290N*v<F{DGus^wY(a;1L3>r)oeufs3wofP%<2iyPU?yl}%cO9rObj7KuzsJ8L3zlDcS*Mz(^+;rI&r)NE3Wz3f2so6Dq`J1ys3<8h;=mZ)4*V@0uke<mjWbfUWQJyHLltst@J`ZF9`A&fYl>~tEU7MJoFRVd1%k8~?p{xbx!7b|ul7q7E_YL(gEtc(eWh(-q63Q^ae)sG9*=5c^+-w#SZN^7)r&p-j-9KyK(gS8-+}&8YYyf6~nJCDdl#vC6E-^EIdlX4S=~pWbQq~G2B*$Zk>D4L~z;S?pde5a!B_J91!z^Aro46^k&WkqJTP6F#cMNsMD9S5vk0rEs^M@0AUrXK4sNQ15W9{NrW`7o_?b+nwl3-p06QzU5)M_W~V<b+w_;Ph?!V<pg?Asxkuq861Ta3zSL<Smv4Qyy3Z~^9qyYz}&bK(;@NN^wg%hg;`=;=8n?!|fa264|Iaf^u%($2H1$yvnG|8UfHxhxtDhqG_qS$k+4+=1z&*Au$0sl?d$>DdkKx7&w?*hy(f<o^A^lZ$r;2*(pHg9CC+ySeHLx*eb!lGryI`MAPqwYoNl@4#3cgC`lH%QS>yx8Z^M*Mjm3qwQk=m;pNPVz!xRwo3hljHq!hWjodrP~6q~oxm<+ha5z8k#EZJ@4XUX>hvw8p?$4a9tT<#c{F&*d+Kc`z<4R`&7NJ?rDkAecKRwROe~YJ-MRbpNQ^4G6<dam3}xsz1i&(P07Q$<mDNp+Lc7MhWj*u7BV?EwgcOu-q?_chY2}c^iVw~R&sVCweIEx2+E2F7AiN%F{+q<?<MZcZBEKj-1#P93o8XB(Qo8-KZ86FRz<LTGJT*AeqIfX`?MV5IS^W}AlY^=?mOOR-8vz<|F3@VGG-D`G%-xg11sZz}-l9#4fMQI+&Vi=xe4zTgzEh|JQcKTL1vAR@nr`?|0t9WWt4;b@a@(YBJakaDc(U!sh!eQD?R8sB>l6m}mK)H!wkc5p*C1>wA^L&>Zuq~}(=Vy75Vn0nr0{WP7K{)gBmWci=d4pcyBw!=lL01WbM8bjrJuE7?pq1zg{!%X7atv!(N~0Op>dPh<J*VJpl`yo2RFPtZNlcmB67!>ccfPtbYF8EXH<yiZ(zXZR+ws-lgW#>FdFQPfm1uuo+E;cH;yp6yG1VSp%_EUq^V3}rwfIK?QOBfRh>^Wy}O}3%TI~-RaZSHT(|5D_%wT<2rP~7t=OB$!4^8kLjtG+!gmT#L8f`P_pa_wB}}7#fRb?{5nsoMqP=wS`WjZ{g#-mldHiJ($!4qlOrfA2Kh0gH{vOvz9nczeowF$C=}@|Y)E!87QbtH#!W-lqS(K#Ni*B;Sl8cfD$Hjv6-vljG4myEUCt>k+w1YLFnQ^c5&b#Zz2$BYO^zEQA4fUHrxbdKeyp?H%<{_fN5;#Xr)&Q4zXJWjn(FCVqROShG*qsN*kCGUQgZ3t|uS#QHuk~}NpSV0*<X(bG_OTqtEDBu*qMwGU{g-+Z-xl*{O!#x94;As|LVT6Em@ZetGWWDV2{i&ah9g;&L`lO4%54iX@|)l-VG@Hc)^5oRofHqtmnj1;?13lSZjlGcMqx28;9YtYpC-DnGW-lSHWkX{0Z9ns5sok@uagKW2F9%vD-cj><urj_)MYXM360<s?g4yZV#k0QDUoQ>2|?5PXs?U<GxN!DM`JyM9CdWRLR1hXp==1+nks=hQ!p$-mARDq7x?90UaT8T{zf%Vd{7Oan#X6fSD?k0F~a2%UJJVT*yN(Q|7S(lS1m$jLq?9Phjqz1B;-fAULS(;=K=K0*yW5n2%qgc)G1H57RPEJec5iD-W!fA`X$ve9w5cr+!}*-(VgY4hHxUv8bttZJ(opsj#MIPSv<8#$frnJ_eftQ`d2@@uH%rOH4O@>W4pS1#S35hOoSA>riVU%H5-iikwE|BluY;i8c5I%p`sD{t8W#b=uGy_&q6706MGtoFUtVV43rXlU)bk(;~F{p4T9BFp6eTGi(Jr<{+@#&@^7p1?L+j9JrR3Bx6NKT51a`8K$tW}rvNa!AmVc7<Zj=6VdIitOO3&%sYSm2`ZHstCAc}fLS?r|Ql+AK;aSWw^T(swEXkLt$^qWsi~+rP>C>?oS3g<y$Jnbxp6FLW0LQs28ka*G{<B-d=0r+`j?#|O%Bu2lGUQSi2GZa`zxOBEaG+#kafI#AxutgzT?r9CbxI3@>j4Ua-dFBq!s^Jh{#6B3B{ei}<Xr{J8T9QOY&@uxS~|$5ql4GLN@B6bLM3XJHXfPsXesMaxc@s%RS(1@L5Z<@=Yk3Tz+Q)8#hGUY<b-m2PozHhfQtkSwg4VunG?iCi-nwu0+}Ef)kvEV^{;zS_wgVhVO|BQUl5{HpKi9;!ICS>J}G+sj6)6nkrO9P;Iz1o{;((d15~Xy!qC}0rfmb7!70r!5jyyk*SJq#O|EB>&hKLmZZagXWO+=iY`!enB@T*(Q)_j#S?gIv*NF*PydaA=+&=qle3{To(2AsShgfsHoJLX%TF+<(PFHIri~s9Q3vr5sUX2rS4As?~$<_#=j<n4^p#p8E?TC4D4tMiX*X`!)9Eze@q+xpwbP>@{M;-J|D1Yu>gM{2Y<l-bt3e`!bO2T%2q~LWMx#x+s>L`D@{!iVL1h_HY@9S>lJvIF@jcx9fm??1$VIyx4<sU2%Ip%1pY{7dGMvKn;Di^C{NbilA!euBuxknM8QdJo}UIVbnANXYg+Db|2C3$OL=s32LJDD(k(CcCt?DJ~j975TrSGo7przar8Nd;9As2O1+^#A-1@4uBPRIugc_Ip}Ke~O|@EDNPaAPs<Ve?Ud3d|4gL?oUS9>ff_y8)#&uJq$I1b-H;-O$*x4Os8KC)03tAn33`;)W{=e7LD8NdAG{=uMVV?Gzg(dh~q2zSLrhbFKHIip1TkYAe_+qKKWLf96o)LB#V<c;n@}%2Gb?~Vz2)}faa)1%S#YHTXZvMv>C{X3_Qtp6UetkqF0bLs~LiBHH~yW-W1i^Zx0{JqTfU>oWNnIjb<1_DNwSaK1B)$$HYkF0EY*9in59L`t|W&y}3ik2Mx8*8Y1h()t?}(j)*&QmUzHjtXOxv1+F~h{&?MIC?P(8&IMk?uq0pcVMl<M!!f-o?h`C{*du{~nGN=T#pYm}Y+OB;YmHKUs++No0$|yU%8}L84y2$K?q%qUd~(hZQVw4bJH^A#|4$nsWPe^Khs1`J{!~LhW^({z%qnNHII#Kr05?}=6mtVzk_FU&%@P_xd^Do<=l$xH3mZ8cX{w8SjzmD=(g#L6;MI4htHfLsXAwp_8%3K%O1i|xJi$amjvKIA(q8leYO?{iKlQxhT&W5o4*fEb>A<stu?y_ry}C?lYyXCj2nO6apyCtCRMehZ!M6iL89BNo=x0iBFY%E+s|lb)8zNi5Q6sFIevv<&=FBFLm=OF>7tPILGoH}86*(JIEU&c}3)pvIceHk2HBlKGn_(0dZLoM^?dg0rG-*Q>huel-dge^n@L}BedI$ih<Q4h<+i?8GOFSR)gmC6EWZ434QE=L+>m8U5xA7kUuG_U0iI$gAIHLCUj1*5+dhaEOy?A|e#@K%Df^5O}q#MtQ^6l-D2)ZyknUvh9_Jyg+$R2I3u;yiw0o{zv`<!qGb7r2pR4qsfmU@vXrAqS>f@&fl=^XbY@v#YWh`#KUi@9Q}CXQN53*<_l?#n3pmMwLQiFY?=+!^-!)JE!tnI-gU<jb^tuVK(_;P+J>bHDXOq5O~xE4Al~d+U%UmfRBDETF%3SwKOH#p|Za%qHz<JGHsOeliJT;A?Ah=N!;=ltk=Yq3l=G5VVzfN+}8p8y0qGQU6cni8&Pt>oXgj1T14?4cvG8j3}6c7K_G4gjeW5qIN{do!!uB2#pJbs!c!A)NpuB6Y)ynZU$gSGI(1wCw5hdCYMeT@{nlqFMTSN*vSqXI^&NJIq><_ngx(5^3baeEnQqM`f&J9eDM&iYmu5$xbs~+lG38{Yp6<Nb=8?kgR~z*r}ZYOpv-!KIyJJuT*0Uz*!aQZY%{-+nr=ESv`9?c)L`GZ4Mz-3e8%}TaY8E?(z(jOIru_ZvhzpWob|BsiCgG;Uw@mdM;~XZzmTn^;cmFjG&m+UwPdne6A)6X$28PZ!>R~Zd$##14E~P-wkygW!_J}g0;D5atRJgE^7OZDOU5|i36nRgG>DvY-;Ol(Ktg$3vV+5s<9|7`Ygqp*aAURlAN0nfX~fLFw(y4swyO?J1D1t4x*{7vm|SBb9T>k3-1O}^JUte+5lx@x>~Qs_hVx_dLse@#?+e@VaKW&JpmidWblf(`$L%rFmpH$@Xy$P;0@iR(dCFsfeuo06z;0)M!!yssON&A+G^N02`;qF-Iv=8e?}?TV;?>1>u7L?xbYp{twk7iQg9z76;;m4lr05DX4_wjuX)OcJ-8Pz<)(<R`N#x2ivfBjEA*v3<EbGQGv?D%V&_(kR3PyX&m?D1SMMnPS%&)lE0>-DF4U3O#Q=3PpJqT~0Bg5ex4T?McPVL=pyPGYpRIA!Gr$z637JUZ>qTZC1!Q(QM0}NDjXsi`TLvQSb8iQg?$MD02;+6-Tb{ucf4D#npY^jcLLxl)VCoI-6@~3^Z9YFFhqgU4TJ1{Q{e6VkSF7it=dvHVt3bWsuep9#O8u^Sjod8z*CG~W-9;9BTX79ib035_%26^@nlLXn7sCaNe-Z%QCeANN1u~ZnW;sgl8?6*A;DLIUof*y~jTE{Wf!+yBQO;IKlQ*TVgX5e>MS<AH<n8~{<D=XX**|s7zqA|b$a{qL7!-@*ZvO(X3_fl)@Dvn4MaXw;Zm}+tKb|WY(dY!SVVdev}a6=Z0vR0lm>(AvsF_qqftwg&t*-Lz9pY{K7e6IDQ;GvkKLWP&@QCK9eg#BP*5e{IxLw1yhY9^sH@~f4J*+B9N0|7kW2T@=rDsP%d_SeT;oQ#V)JrS_i`?b5LT-X5^{1|E-(W7~u!vB?^*#-&*$zhafsYXPIwEoSNdH=m<v^0Cn?9ol<ghZHzQBFgCX1B!0ZFQYWi4SMYTw*H2|MlSM>wq}+t6IK%_%3(b!@$Rxs>0wey?xWfH|Iqcy3KXbG_Cf7mo!ByB-6>0k3B}MD>xNHiDc9>Hkqu=qbZ;J)O({Z>u*$$XvHx4)DxzivMjJD?wA?poe~>KLcCJ>qPo{MJaD8TQGR9RW)B@NuoL%sv^S7%Cm3+ie9`|X&?$USJzL%6JeVur;B!zPY=fyV`#qdJ8b9ZiS!i*J+s&tmK~XUD0J2eb^s9$xm!6xVAvNsrArn*zgjf1-*q_zPU^xG}2(Pz&;<WTEh`Q8Si>b-q_rKDsWwE0<Eh!rAYhM@$Z3a>jst}4R!4r^26*T`U_qyv<G6q81qPT5!1Xn-MyIppf6pI_3-267E^YKAr4;nT(t>Drd#JinZm+hA!8u3T@v*`T6g{jUKv<5j*#aHnj+`4UdO?t)>9|UeB*#@rs#v>$&$9GwbEt1jVE^lSDHQfixQ7=V2ducVy0GuN`XraM|MpJpb6F(?tl5>8TzB*IH!#5HO(C!*BC<fVhA4dmMg@(tkNjIu!y_eU%xEWy!0ah=s)9u!y{c~dkvI$0M&Qdqh2TL`Ri|S@+a8b_9f7s!&woK$860Y8ug+$RFugoq61P0JQ56ZIf<aSTW(q6}IUE@$qbDtS?R*C}vcD!56g3Bj7`dwsIYo<<Je^D$I-O2+dfk40X*U{6Bx{e2FwHZ^x*1VhxXq?mXif~mrJ#9Ya)N4MQX;4$xjIIMFK%0YN8B_bNU~mBR9ek5^oR`ELaKXh)lwW=+UM3C>_ooMko19iCoy90S<mlgFVXfIKnRu+tC#q5xWLLD$&tPBPJmuMah&;>G#Qo*W>l23S2hp6boZOD&=uNuBv2PTMHOPgR{R1&iWl&ms?3gs*F2H_(&FSf1m&f=hmeS{qWGh;}$A4QU$-zx-L^Retz58=w%*^~|xT4t;DB*X$1jYy_u{V&9F?IR!>?j{GGvuBnaBwOPu0J(xl@q2*x~4i>t0Lqh?^lF%t3g|@Y|$XuFdH4U8pCno3ZI>oSZ#5YA=TgWY%BNC1!RXg7<4l+?&|z3A8A?Z*nhuTh8LFdO?FPu)8AEZ@_jo-X@1ySG2j+~TZns8s92Ic-a&%f+phgHoKDFOdORL+?=1RRHAR#EP=@dE<kK<_uFrV{cXe8Q7ozYcTgS9GG6v4SC~yva|3=DU7G<q9Lmx&l4}wz?Ux|ZF^TnCCpAe!jb4ja?7rDzYe9G$`O6c7a^CVOqZ7w9LoMRcUA3mcO+!{@8{TAKso=7kWP_gVCU`@lp_mF9*)_jqre}RJ;jB^W=>UHA!7FZuP6d%pMC+s9VPW%um&nNl*P5nwMO%zhY`}=ZqqQ=C6c0iDHSQnr8Xf53Ko>ZQNGw<u(D?;;S?>`|i6hZlXkE{2r!z)9EK;RsWa(~|jRwc;HYhqi;XP>rLRuIQ#r7+Y2O|xJfBdd?{l|i#M-X8QG(9PbMh4Eqn9zZPv0GggJF?s8@CW3?7>(F9y>$!&$6yz#;F+}~_NWSk-sNpW4UNbgp;up;Q_21+#HMu8$zHKtHvRzl8gbdPbOHx;-+(T8@JZ(@J6GQoqEP>p_2+OCPW@X?+rAB5rgHkbFaGm>R3M_>c<|pub`|Z#RZ>XQl<hbou5WZ`3+;LL9rKF0vIx;6waA%0kWudd{m)}F)jx6Leg&b;~tkSUMZi;=-PbBH<(IO)oUjmSFcN~#tS<!Sj<-4V#AaKPC077DetYm6^W|TU#GLL#^K0KGmB%PQhBu{lZ0HYJ>bAh6&`KHRl_ee>Q!6>!G=)ApG_th_G5Ch#pgmUx9M-6*G<gN!{tb6pRZP&;~oS|QDSUOvaYy6|kwlzcbyEz($`&nOE62?c8gQcvsHLcSopqW+UfK_R&142nAtsy9bnHWf^wHbyMcC{4)g^@{TpDlBKbZNhxy&Ua!kxxmHaT)_(5kPp(&F;r8S&d2g3m_v1o6R!seyD{9?@`4}A2$lj)aEX0{VL^5uLFs0{<X8aa8UT|X!;)S{2R0IXH~7I8t$K75?PI0O<*vi!pyq}OE(f|UWd=~eZr`K4*=s5lq(?_=5h~oRbItfdUHn;&6)C+Z~;S&Z)No#Keuc2n>XHr$T;DZT$MpTpadej{Rt-fC#qR(myrjj<g=vOcE;l5ADbylEj*hKf-*Xd>ovzy6Kf_0i@i}p46FacJyrKthUHCLiH&!=eV(w!mDBw+-sG=DhUHDV8}9O+OA=XEEL0$J6WrZJ34z5`8Ns30!`I;{K8{Ny-Fo}Tsih!N<{iN+Yt?|HU^kqPk6u45q1Tn>Z$jO-JvQy(AqC>Ti=j`Zk7%MhwqaV)d~ki)j@F~^`%Ul1Ilbdx&JE1J@t(%DJ^Sk^KFb7i@2mM&vYEdu3yUsgYFJa)>A;2IPgz4}@#h4cys&$=DU?`TpU(QK>15pF+T9TKGO2ALQ)sld^GWd;**8+*zs`^C2s%-?!<$DCRAwU-X1iW@$S?6ff2QArsW#OA$ctNuM=l)Y+Vv}g=&d|5a(~rBlGiyPh>8%#XGpvT$!(Bjm0+eh`<B`q(*K=G#>qSfZ(i^V5&d{K9gH%3WV*unYFv#-3I;fDSEj>Kdo&(%d&Z;Woq%^XfVt#;x#H1<Qo~0;`RvNY$}^AGD`aSyDy1*JD3pb_Q-~nwQ{6Yjw@iW$xb8<5E*kB3`XIF!>ff$#GSlr;#$$i5We`>$>*>@Uy|MG(LmRWJD{62#R(v}mhRR*1v?i#6Ks=9OPgXD}L_>-2`w>CFH?+>-D8_+EAKZM6rCK&=P3tfKo0;Saa~$R7eHulGJ=qb=`}9J21-+|QGw&@&JjRxh3vmf_4dR@)GB_g4b`=OM_R9F%+-EmV1@UPY#pOCr`bx?64-Lp%?Im*4vIU}7;yr^sC<9u6vJvqW&1qu|GO5$?5zRW!XkGpE>1F_8qLTnWtmLss8eUs>!8+z&JGVl6l-R!c3?m);@C#0~aNyWOT_T&<>|V5!cMB7q^>hUe6kByg+!`kcb}R`5=i@$s&6kqXb+kbDH}ySZ2A)C6Dyz}ZqebaCAgXNxwm!m{p9&2@h)h5;0|L}(uC|t@wQp8mSDc1Z2RS;j4T^AaXdNfVX%K_HVvMFyK1~B(IKwrifAVR~wG})P=tUq)sE<aO8=sepGNuv*10;aUh-kEcLBtnWLLfEThUJ`>m5OD?MG1m(!Y1LTXj{&;yiJOfn+@$W?dG6NS|-6*=r~dm_c_M9trk5tDS;p|DLU$S8cOGZ{sbifDSJdtoPJ4(+S_wu@foQB{fDt5Q)Y6$&<6Wiym*Lbat9spF_`sKk8owT+od+!A#A538!K==7I1Ly%1&}PJ#e$eAfH7B1~W_)PPX9l_b{>ur@ib{Ts4(sYw3rGRXafQQ9zUyqkZqiYaoMEHqe`zDT5$%J!mgR<qJE!^x`ZWyJkR9%8g1s`GH)q`OJ4$fJRvg6xLJwub9eK2tqTJ-v0S|iW>*ac=EEr+)IqyVnz19Ho_npg-W7as1MY5$iK8Q%0=W#Pb^6QeYt-i7keUq00F%32+P0=RS3TTvBYQl0ssI200dcD\')).decode()\nassert hashlib.sha256(_s14_body.encode()).hexdigest() == \'2c800267df03a76e3452c1de2084d2ac1fe2e272d630519a512bd358aeb7283b\'\n_s14_file = "<s14ab-body-2c800267df03>"\nlinecache.cache[_s14_file] = (len(_s14_body), None, _s14_body.splitlines(True), _s14_file)\nexec(compile(_s14_body, _s14_file, "exec"), globals())\n_s14_lab = lab\ndef lab(deadline, weights, state):\n    return _s14_lab(deadline, weights, state)\n')]
_lab_base_prepare = prepare
_lab_ran = False
_lab_flush_buf = None


def lab_cold_l2():
    # The judge's cold-L2 step (TASK.md): read a buffer larger than L2 before each timed replay. Never flush with
    # zero_()/fill_(): that leaves ~50 MB of dirty lines for the timed call to write back (+5-16 us at M=1-32).
    global _lab_flush_buf
    if _lab_flush_buf is None:
        _lab_flush_buf = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    _lab_flush_buf.view(torch.int64).sum()  # int64 view: the uint8 sum upcasts and runs ~4x slower


def _lab_run_one(name, src, deadline, weights, state):
    import inspect as _inspect, linecache as _linecache
    fname = f"<h200_lab {name}>"  # in linecache, so inspect.getsource (triton.jit) finds the source
    _linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns = dict(globals())
    exec(compile(src, fname, "exec"), ns)
    fn = ns["lab"]
    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)


def prepare(weights):
    global _lab_ran
    import time as _t
    _start = _t.monotonic()
    state = _lab_base_prepare(weights)
    if not _lab_ran:
        _lab_ran = True
        import traceback as _tb
        _end = _start + 285
        for _i, (_name, _sha, _src) in enumerate(_LAB_PAYLOADS):
            _t0 = _t.monotonic()
            _deadline = _t0 + max(0.0, _end - _t0) / (len(_LAB_PAYLOADS) - _i)
            _lines = []
            try:
                _lines = [str(x) for x in (_lab_run_one(_name, _src, _deadline, weights, state) or [])]
                torch.cuda.synchronize()  # inside the try: a sticky CUDA error still prints this payload's lines
                torch.cuda.empty_cache()
            except BaseException as _e:  # the error is the result; a sticky CUDA error still fails the run after
                _lines += [f"LAB_ERROR {type(_e).__name__}: {_e}"] + _tb.format_exc().splitlines()[-8:]
            # Over the cap, keep the first two lines (META) and the last lines (summaries, errors) and drop the middle.
            _cap = 9000 // len(_LAB_PAYLOADS)
            _head, _tail, _n = _lines[:2], [], sum(len(_l) + 1 for _l in _lines[:2]) + 40
            if sum(len(_l) + 1 for _l in _lines) > _cap:
                for _l in reversed(_lines[2:]):
                    if _n + len(_l) + 1 > _cap:
                        break
                    _tail.insert(0, _l)
                    _n += len(_l) + 1
                _out = _head + [f"LAB_TRUNCATED {len(_lines) - 2 - len(_tail)} middle lines"] + _tail
            else:
                _out = _lines
            print(f"LAB_BEGIN {_name} {_sha}", flush=True)
            for _l in _out:
                print(_l, flush=True)
            print(f"LAB_END lab_s={_t.monotonic() - _t0:.1f} prepare_s={_t.monotonic() - _start:.1f}", flush=True)
    return state
