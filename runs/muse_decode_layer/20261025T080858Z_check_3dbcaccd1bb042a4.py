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


















BEST7_SETTINGS = {'other': ('best6', 'auto', 'auto', 'auto'), (1, 1, 2048): (('best6', (16, 16), (8, 1), (8, 16)), ('attn_main', None), 'nt1_lop3', {'o': ('stream', 6, 1)}), (1, 1, 0): (('best6', (16, 16), (8, 1), (8, 16)), ('attn_main', None), 'nt1_lop3', {'o': ('stream', 6, 1)}), (16, 1, 2048): (('best6', (16, 1), (8, 1), (16, 1)), ('attn_main', None), 'nt1_lop3', 'auto'), (16, 1, 0): (('best6', (16, 1), (8, 1), (16, 1)), ('attn_main', None), 'nt1_lop3', 'auto'), (32, 1, 2048): ('best5', ('attn_main', None), 'nt1_lop3', 'auto'), (32, 1, 0): (('best6', (16, 1), (8, 1), (16, 1)), ('attn_main', None), 'nt1_lop3', 'auto'), (1, 8, 2048): (('best6', (16, 4), (4, 1), (8, 16)), ('layer', None), 'nt1_lop3', 'auto'), (1, 8, 0): (('best6', (16, 4), (4, 1), (8, 16)), ('layer', None), 'nt1_lop3', 'auto'), (16, 8, 2048): (('best6', (16, 1), (4, 1), (8, 1)), ('layer', None), 'hh_m128', 'auto'), (16, 8, 0): (('best6', (16, 1), (4, 1), (8, 1)), ('layer', None), 'hh_m128', 'auto'), (32, 8, 2048): (('best6', (16, 1), (4, 1), (16, 1)), ('layer', None), 'mixed_parallel_m256', 'auto'), (32, 8, 0): (('best6', (16, 1), (4, 1), (16, 1)), ('layer', None), 'mixed_parallel_m256', 'auto')}




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

// Predicated in the asm: the outputs count as written either way, so the compiler keeps no copy of the old value.
__device__ __forceinline__ uint4 ld_w_if(const uint4* p, int pred) {
  uint4 v;
  asm volatile("{\n.reg .pred q;\nsetp.ne.b32 q, %5, 0;\n"
               "@q ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];\n}"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "r"(pred));
  return v;
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

// Activations are staged in chunks of CB k-blocks (CB * 64 columns), one mbarrier each.
constexpr int CB = 4;

// Shared memory: xs [M + 1][nb * 64 + 8] bf16 (row M = zeros), nch mbarriers, then (V4) the partial buffer, placed
// after the largest split's region so mapa finds it at the same offset in every CTA of the cluster.
__host__ __device__ inline int red_off(int M, int nb, int nch) {
  return ((((M + 1) * (nb * 64 + 8) * 2 + 7) & ~7) + 8 * nch + 15) & ~15;
}

// grid (R / (NW * RT), S), NW warps; warp w of CTA x owns row tiles (x * NW + w) * RT + [0, RT) over the split's
// k-blocks [s * KB / S, (s + 1) * KB / S). Block (r, kb) sits at Wc + (r * rs + kb * ks) * 1536 B (row-major
// rs = KB, ks = 1; k-major rs = 1, ks = R). NT = n8 token tiles (M <= 8 NT).
//   - scales and escape offsets load at entry (weight-side: before griddepcontrol.wait); the first 32 escapes of each
//     row tile's range are prefetched one per lane after the first blocks and applied by warp shuffles
//   - the last warp stages X[0:M, split] by bulk copies (one per row and chunk) completing on per-chunk mbarriers;
//     each warp waits only for the chunk it is about to use (no CTA barrier in the loop); token rows past M read
//     the zero row (row stride L + 8 bf16: conflict-free 16 B fragment loads)
//   - each weight slot is refilled once its words are decoded (asm-predicated loads: no register copies)
// V4: the C CTAs of a cluster (consecutive splits) sum their partials through distributed shared memory, so P gets
// S / C partials instead of S (C = S: one).
template <int NT, int RT, int PIPE, int V>
__global__ void __launch_bounds__(256) coded_gemv(const uint4* __restrict__ Wc, const float* __restrict__ scales,
                                                  const int* __restrict__ eoff, const int2* __restrict__ esc,
                                                  const __nv_bfloat16* __restrict__ X, int x_stride, int M,
                                                  float* __restrict__ P, int N, int KB, int S, int rs, int ks,
                                                  int prew, int C) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  pdl_release();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int nw = blockDim.x >> 5, s = blockIdx.y;
  const int kb0 = (int)((long long)s * KB / S), kb1 = (int)((long long)(s + 1) * KB / S), nb = kb1 - kb0;
  const int r0 = (blockIdx.x * nw + warp) * RT;
  const int L = nb * 64, RS = L + 8, k0 = kb0 * 64;
  const int nbm = (KB + S - 1) / S;
  float sc[RT];
  int ea[RT], eb[RT];
  int2 epre[RT];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    sc[rt] = ld_f32(scales + r0 + rt);
    ea[rt] = ld_i32(eoff + (size_t)(r0 + rt) * KB + kb0);
    eb[rt] = ld_i32(eoff + (size_t)(r0 + rt) * KB + kb1);
  }
  const size_t bstep = (size_t)ks * 96;  // uint4s from one of a row tile's blocks to its next
  const uint4* wp[RT];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) wp[rt] = Wc + ((size_t)(r0 + rt) * rs + (size_t)kb0 * ks) * 96 + lane;
  uint4 buf[PIPE][RT][3];
  if (!prew) pdl_wait();  // prew: the first PIPE blocks load before griddepcontrol.wait (overlap the producer)
#pragma unroll
  for (int p = 0; p < PIPE; ++p)
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
      for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w_if(wp[rt] + p * bstep + q * 32, p < nb);
  const int nch = (nb + CB - 1) / CB;
  const uint32_t mbar0 = smem_u32(smem_raw + (((size_t)(M + 1) * RS * 2 + 7) & ~(size_t)7));

  if (prew) pdl_wait();
  // All CTA threads stage the entire split while the weight preloads are in flight.
  // One wait and CTA barrier, no per-chunk mbarrier initialization or producer warp.
  for (int i=threadIdx.x; i<M*(L/8); i+=blockDim.x) {
    const int row=i/(L/8), col=(i-row*(L/8))*8;
    const uint32_t sa=smem_u32(xs+(size_t)row*RS+col);
    const void* ga=X+(size_t)row*x_stride+k0+col;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(ga));
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  for (int i=threadIdx.x; i<L/8; i+=blockDim.x)
    reinterpret_cast<uint4*>(xs+(size_t)M*RS)[i]=make_uint4(0u,0u,0u,0u);
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
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
#pragma unroll
    for (int p = 0; p < PIPE; ++p) {
      const int bb = b + p;
      if (bb < nb) {
        uint4 xv[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const uint4* src = reinterpret_cast<const uint4*>(xrow[nt] + bb * 64);
          xv[nt][0] = src[0];
          xv[nt][1] = src[1];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int rt = 0; rt < RT; ++rt) {
            uint32_t a[4];
            decode_step(word(buf[p][rt][j >> 1], (j & 1) * 2), word(buf[p][rt][j >> 1], (j & 1) * 2 + 1),
                        word(buf[p][rt][2], j), a);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
              mma16816(acc[rt][nt], a, word(xv[nt][j >> 1], (j & 1) * 2), word(xv[nt][j >> 1], (j & 1) * 2 + 1));
          }
        }
#pragma unroll  // refill this slot once its words are decoded (same registers: no copies)
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
          for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w_if(wp[rt] + (bb + PIPE) * bstep + q * 32, bb + PIPE < nb);
      }
    }
    if (b == 0)  // eoff has landed with the first blocks: prefetch up to 32 escapes per row tile
#pragma unroll
      for (int rt = 0; rt < RT; ++rt)
        epre[rt] = ea[rt] + lane < eb[rt] ? ld_i64(esc + ea[rt] + lane) : make_int2(0, 0);
  }

#pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    const int r = r0 + rt;
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
    const int n = min(eb[rt] - e, 32);
    for (int i = 0; i < n; ++i)
      fix(make_int2(__shfl_sync(0xffffffffu, epre[rt].x, i), __shfl_sync(0xffffffffu, epre[rt].y, i)));
    for (e += n; e < eb[rt]; ++e) fix(esc[e]);
    if constexpr (V == 3) {
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
                             int rs, int ks, uint16_t* __restrict__ out) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  if (gw >= R * KB) return;
  const int r = gw / KB, c = gw - r * KB;
  const uint4* blk = Wc + ((size_t)r * rs + (size_t)c * ks) * 96 + lane;
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

// Dynamic shared memory: staged activations + zero row + one mbarrier per chunk (V4: + the partial buffer).
__host__ __device__ inline int smem_bytes(int M, int KB, int S, int V, int rows) {
  const int nbm = (KB + S - 1) / S, RS = nbm * 64 + 8, nch = (nbm + CB - 1) / CB;
  return V == 3 ? (((M + 1) * RS * 2 + 7) & ~7) + 8 * nch : red_off(M, nbm, nch) + M * (rows + 4) * 4;
}

template <int NT, int RT, int PIPE, int V>
static cudaError_t launch(int NW, int pdl, cudaStream_t st, const uint4* Wc, const float* sc, const int* eoff,
                          const int2* esc, const __nv_bfloat16* X, int xs, int M, float* P, int N, int KB, int S,
                          int rs, int ks, int prew, int C) {
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
  return cudaLaunchKernelEx(&cfg, k, Wc, sc, eoff, esc, X, xs, M, P, N, KB, S, rs, ks, prew, C);
}

}  // namespace cdg

#define CDG_CASE(NT_, RT_, PIPE_, V_)                                                                         \
  if (NT == NT_ && RT == RT_ && PIPE == PIPE_ && V == V_)                                                    \
    e = cdg::launch<NT_, RT_, PIPE_, V_>((int)NW, (int)pdl, st, (const uint4*)Wc, (const float*)sc, (const int*)eoff, \
                                     (const int2*)esc, (const __nv_bfloat16*)X, (int)xs, (int)M, (float*)P,     \
                                     (int)N, (int)KB, (int)S, (int)rs, (int)ks, (int)prew, (int)C);

void cd_gemv(int64_t Wc, int64_t sc, int64_t eoff, int64_t esc, int64_t X, int64_t xs, int64_t M, int64_t P,
             int64_t N, int64_t KB, int64_t S, int64_t NW, int64_t RT, int64_t PIPE, int64_t rs, int64_t ks,
             int64_t prew, int64_t pdl, int64_t V, int64_t C, int64_t stream) {
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

void cd_decode(int64_t Wc, int64_t sc, int64_t N, int64_t K, int64_t rs, int64_t ks, int64_t out, int64_t stream) {
  const int R = (int)(N / 16), KB = (int)(K / 64);
  const int warps = R * KB;
  cdg::coded_decode<<<(warps + 7) / 8, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const uint4*)Wc, (const float*)sc, R, KB, (int)K, (int)rs, (int)ks, (uint16_t*)out);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
'''

CD_CPP = r'''
#include <cstdint>
void cd_gemv(int64_t Wc, int64_t sc, int64_t eoff, int64_t esc, int64_t X, int64_t xs, int64_t M, int64_t P,
             int64_t N, int64_t KB, int64_t S, int64_t NW, int64_t RT, int64_t PIPE, int64_t rs, int64_t ks,
             int64_t prew, int64_t pdl, int64_t V, int64_t C, int64_t stream);
void cd_decode(int64_t Wc, int64_t sc, int64_t N, int64_t K, int64_t rs, int64_t ks, int64_t out, int64_t stream);
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


def cd_encode(w, kmajor=False):
    """bf16 [N, K] -> dict(coded uint8 [R*KB*1536], scales f32 [R], eoff i32 [R*KB+1], esc i32 [E', 2], base i32 [R],
    N, K, nesc, rs, ks). Lossless for every bit pattern; runs on w's device. Blocks are row-major [R][KB] (block
    strides rs = KB, ks = 1) or, with kmajor, [KB][R] (rs = 1, ks = R: the row tiles a split's warps read at one
    step sit side by side)."""
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
    coded = torch.cat([lowp, nib], 2)
    coded = (coded.transpose(0, 1) if kmajor else coded).contiguous().view(-1)
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
                rs=1 if kmajor else KB, ks=R if kmajor else 1,
                base=base, N=N, K=K, nesc=nesc)


def cd_decode(c):
    """The kernel's decode of a coded weight, escapes patched from the list -> bf16 [N, K]."""
    N, K = c['N'], c['K']
    out = torch.empty((N, K), dtype=torch.int16, device=c['coded'].device)
    cd_module().cd_decode(c['coded'].data_ptr(), c['scales'].data_ptr(), N, K, c['rs'], c['ks'], out.data_ptr(),
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
    v3 = (((M + 1) * RS * 2 + 7) & ~7) + 8 * nch
    return v3 if V == 3 else ((v3 + 15) & ~15) + M * (NW * RT * 16 + 4) * 4


def cd_gemv(x, c, S, NW=8, RT=1, PIPE=2, prew=1, pdl=True, V=3, C=1, out=None):
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
                        x.data_ptr(), x.stride(0), T, P.data_ptr(), N, K // 64, S, NW, RT, PIPE, c['rs'], c['ks'], prew, int(pdl), V, C,
                        torch.cuda.current_stream(x.device).cuda_stream)
    return P, S2



CD_ROWS = {(1, 'qkv_gate'): (8, 1, 2, 22, 1, 3, 1), (1, 'pool_router'): (8, 1, 2, 11, 1, 3, 1)}


def _cd_prepare(state):
    """Store lossless codes in state, so judge clones carry the matching encoded weights."""
    state['coded_dense'] = {}
    for name in sorted({name for _, name in CD_ROWS}):
        w = state[name]
        c = cd_encode(w, kmajor=True)
        if not torch.equal(cd_decode(c).view(torch.int16), w.view(torch.int16)):
            raise RuntimeError(f"coded {name}: decode differs from the weight")
        state['coded_dense'][name] = c
        print(f"CODED {name} {tuple(w.shape)} escapes={c['nesc']} bytes={c['coded'].numel()}", flush=True)


def _cd_force(state):
    for (t, name) in CD_ROWS:
        state['proj_plan'][(t, name)] = 'custom'



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
    with ThreadPoolExecutor(max_workers=15) as native_pool:
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
    state['nf_streams']=(torch.cuda.Stream(device=device),torch.cuda.Event(),torch.cuda.Event())
    s4_started=time.monotonic()
    _s4_prepare_banks(state,weights)
    torch.cuda.synchronize(device)
    phases['s5_banks']=time.monotonic()-s4_started
    _gate_split_prepare(state,device)
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






NF_CUDA='#define BR_KINDS(F) F(26,32,true,7,0,0,0) F(21,32,false,6,0,0,0)\n// Branch/remainder lab kernels. Banks stay packed E2M1 plus uint8 exponent/mantissa scale codes\n// (sign/zero folded into the codes); scales convert to BF16 inside the timed kernel.\n// One task = one 32-row group of one expert x one 64-channel M tile (route NT=32 numbering).\n// A kernel instance owns the groups whose expert count (FM=1) or rows-in-group (FM=2) lies in (LO,HI].\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <vector>\n#include <stdexcept>\n#define BR_CUDA_CHECK(expr) do{cudaError_t br_s=(expr);if(br_s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(br_s));}while(0)\n#define BR_REQUIRE(c) do{if(!(c))throw std::runtime_error("branch lab argument check");}while(0)\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define BR_HOPPER 1\n#else\n#define BR_HOPPER 0\n#endif\n\nnamespace mwbr {\nconstexpr int BK=64,KT=128,S=4,BIG=1<<30;\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out,unsigned scale_base){\n unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;\n unsigned even,odd,rot=rr(raw,8);\n asm("lop3.b32 %0,%1,0x0ff00ff0,%2,0xea;":"=r"(even):"r"(raw),"r"(scale_base));\n asm("lop3.b32 %0,%1,0x0ff00ff0,%2,0xea;":"=r"(odd):"r"(rot),"r"(scale_base));\n unsigned s0=__byte_perm(even,even,0x1010),s2=__byte_perm(even,even,0x3232);\n unsigned s1=__byte_perm(odd,odd,0x1010),s3=__byte_perm(odd,odd,0x3232);\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),\n  "r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b,uint64_t desc){\n#if BR_HOPPER\n uint64_t d=desc;\n d+=b*2;\n if constexpr(NT==8){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3},{%4,%5,%6,%7},%8,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n static_assert(NT==32,"tile width");\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n // Portable stand-in: identical shared layout and fragment ownership, immediate mma.sync.\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs,uint64_t desc){\n#pragma unroll\n for(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b,desc);\n#if BR_HOPPER\n asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\n#endif\n}\ntemplate<int NT,int VEC>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*SC,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n constexpr int NJ=NT>=16?NT/16:1;\n int t=threadIdx.x,w=t>>5,l=t&31;\n if constexpr(VEC){\n  if(t<64){unsigned group=t>>4,lane=t&15;\n   unsigned base=((unsigned)e*(N/16)+mt*4+group)*128*64+tile*(KT/32)*64+lane*4;\n   #pragma unroll\n   for(int g=0;g<KT/32;++g){unsigned dst=smaddr(q+g*1024+t*16);const void*sp=Q+base+g*64;\n    asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");}\n  }\n  if(t>=64&&t<64+KT/4){unsigned i=t-64,g=i/8,group=(i%8)/2,half=i%2;\n   unsigned srcidx=(((unsigned)e*(N/16)+mt*4+group)*128+tile*(KT/32)+g)*32+half*16;\n   unsigned dst=smaddr(s+g*128+group*32+half*16);const void*sp=SC+srcidx;\n   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");\n  }\n }else{\n  #pragma unroll\n  for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n   async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\n   if(l<8)async_copy(s+g*128+w*32+l*4,SC+z*32+l*4,4,1);\n  }\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;\n   if(NT>=16||r<NT)async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\ntemplate<int NT>constexpr int smem_bytes(){return 1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;}\n\ntemplate<int NT,bool FIRST,int VEC,int FM,int LO,int HI>\n__global__ void __launch_bounds__(128,2) gemm(const int*Q,const unsigned char*SC,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,unsigned scale_base){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT,N=FIRST?8192:4096,MT=N/64,NJ=NT>=16?NT/16:1;\n constexpr bool WIDE=VEC>=6;constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(1024) unsigned char typed_sm[];\n unsigned char*qs=typed_sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(typed_sm+BS);\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31;\n uint64_t desc0=(uint64_t)((smaddr(bs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n int total=TASK[32]*MT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  int group=unit/MT,mt=unit%MT,ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*32,count=C[e],offset=OFF[e];\n  if constexpr(FM==1){if(count<=LO||count>HI)continue;}\n  if constexpr(FM==2){int rem=min(count-row,32);if(rem<=LO||rem>HI)continue;}\n  int src[NJ];\n  #pragma unroll\n  for(int j=0;j<NJ;++j){int rl=t/8+j*16,r=row+rl;src[j]=(rl<NT&&r<count)?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  if constexpr(VEC>=2){\n   // Lean K loop (same smem layout, copies and math as VEC==1): per-unit global pointers advanced by a fixed\n   // stride, one commit group per stage (empty past the end) so the wait count is a constant, and the stage\n   // ring unrolled so every shared offset and WGMMA descriptor is a compile-time constant.\n   static_assert(NS%S==0,"stage ring");\n   const int*qg=Q;const unsigned char*sg=SC;const __nv_bfloat16*xg[NJ];bool xv[NJ];\n   const bool wq=t<64,ws=t>=64&&t<64+KT/4;\n   if(wq){unsigned gp=t>>4,ln=t&15;qg=Q+((unsigned)e*(N/16)+mt*4+gp)*128*64+ln*4;}\n   if(ws){unsigned i=t-64;if constexpr(WIDE)sg=SC+(((unsigned)e*(N/16)+mt*4+i/8)*128*32+(i%8)*16);else{unsigned g=i/8,gp=(i%8)/2,hf=i%2;sg=SC+((((unsigned)e*(N/16)+mt*4+gp)*128+g)*32+hf*16);}}\n   unsigned qd=smaddr(qs+(WIDE?(t>>4)*512+(t&15)*16:t*16)),sd=0;\n   if(ws){unsigned i=t-64;if constexpr(WIDE)sd=smaddr(ss+i*16);else{unsigned g=i/8,gp=(i%8)/2,hf=i%2;sd=smaddr(ss+g*128+gp*32+hf*16);}}\n   unsigned bd[NJ];\n   #pragma unroll\n   for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;xv[j]=src[j]>=0;xg[j]=X+(size_t)max(src[j],0)*4096+k;\n    bd[j]=smaddr(bs+r*64+(k^((r&7)*8)));}\n   auto issue=[&](int slot){\n    if(wq){\n     #pragma unroll\n     for(int g=0;g<KT/32;++g)asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(qd+slot*QB+(WIDE?(g>>1)*2048+(g&1)*256:g*1024)),"l"(qg+g*64):"memory");\n     qg+=(KT/32)*64;}\n    if(ws){asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+slot*SB),"l"(sg):"memory");sg+=(KT/32)*32;}\n    #pragma unroll\n    for(int c=0;c<KT/64;++c){\n     #pragma unroll\n     for(int j=0;j<NJ;++j)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(bd[j]+(slot*NT*KT+c*NT*64)*2),"l"(xg[j]+c*64),"r"(xv[j]?16:0):"memory");\n    }\n    #pragma unroll\n    for(int j=0;j<NJ;++j)xg[j]+=KT;\n   };\n   #pragma unroll\n   for(int i=0;i<S-1;++i){issue(i);asm volatile("cp.async.commit_group;":::"memory");}\n   #pragma unroll 1\n   for(int k0=0;k0<NS;k0+=S){\n    #pragma unroll\n    for(int st=0;st<S;++st){\n     asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");__syncthreads();\n#if BR_HOPPER\n     if constexpr(VEC<4||WIDE)asm volatile("fence.proxy.async.shared::cta;":::"memory");  // VEC 4/5: timing probe without it\n#endif\n     uint4 sv;if constexpr(WIDE)sv=*(const uint4*)(ss+st*SB+(w*8+(l>>2))*16);\n     #pragma unroll\n     for(int sub=0;sub<2;++sub){\n      if constexpr(WIDE){uint4 q=*(const uint4*)(qs+st*QB+sub*2048+t*16);\n       dq(q.x,q.y,sub?sv.z:sv.x,af[sub],scale_base);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,scale_base);\n      }else{\n      #pragma unroll\n      for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n       unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8,scale_base);}}\n      keep(af[sub]);\n#if BR_HOPPER\n      asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n      mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64,desc0+(st*NT*KT+sub*NT*64)/8);keep(af[sub^1]);\n      if(sub==0){if(k0+st+S-1<NS)issue((st+S-1)%S);asm volatile("cp.async.commit_group;":::"memory");}\n     }\n    }\n   }\n  }else{\n  #pragma unroll\n  for(int i=0;i<S-1;++i)load_stage<NT,VEC>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SC,X,src,e,mt,N,i);\n  #pragma unroll 1\n  for(int k=0;k<NS;++k){\n   int st=k%S;cp_wait<NS>(k);__syncthreads();\n#if BR_HOPPER\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll\n   for(int sub=0;sub<2;++sub){\n    #pragma unroll\n    for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n     unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8,scale_base);}\n    keep(af[sub]);\n#if BR_HOPPER\n    asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n    mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64,desc0+(st*NT*KT+sub*NT*64)/8);keep(af[sub^1]);\n    if(sub==0&&k+S-1<NS){int dst=(k+S-1)%S;\n     load_stage<NT,VEC>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SC,X,src,e,mt,N,k+S-1);}\n   }\n  }\n  }\n#if BR_HOPPER\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  if constexpr(FIRST&&(VEC==3||VEC==5||VEC==7)){\n   // Gate/up rows interleaved by 8 in prepare (per 64-row tile: G0-7,U0-7,G8-15,U8-15,...): accumulator j holds gate\n   // and j+2 the up row of the same channel, so SwiGLU finishes in registers (same BF16 roundings as below).\n   float gg=GS[e]*256.f,gu=GS[e+32]*256.f;int ch=mt*32+w*8+(l>>2);\n   #pragma unroll\n   for(int j=0;j<NT/2;++j)if(((j>>1)&1)==0){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    float g=__bfloat162float(__float2bfloat16_rn(acc[j]*gg)),u=__bfloat162float(__float2bfloat16_rn(acc[j+2]*gu));\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+ch]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n   }\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n  }\n  __syncthreads();\n }\n}\n\n// Routing only: no output fill, no readiness state. One CTA owns every pair (same as the\n// original CTA 0); with META it also writes the heavy-only (count>16) 32-row task prefix T32, the\n// one-group-per-expert prefix T8 of experts with 8<count<=16.\ntemplate<bool META>__global__ void __launch_bounds__(256) route_live(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,int*T32,\n int*T8,int M,int NT,int start){\n int t=threadIdx.x;\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\n // All eight cold id loads per thread are issued before the first shared atomic.\n for(int base=0;base<M*8;base+=256*8){int v[8];\n  #pragma unroll\n  for(int k=0;k<8;++k){int i=base+k*256+t;v[k]=i<M*8?ids[i]:start-1;}\n  #pragma unroll\n  for(int k=0;k<8;++k){int i=base+k*256+t,e=v[k]-start,rank=-1;\n   if(e>=0&&e<32){rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;}\n   if(i<M*8)POS[i]=rank;}\n }\n __syncthreads();\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\n  int scan=pad;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\n  int key=n*32+31-t;\n  #pragma unroll\n  for(int k=2;k<=32;k*=2){\n   #pragma unroll\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\n   }\n  }\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\n  if constexpr(META){int ns=key/32,v=ns>16?(ns+31)/32:0,u=ns>8&&ns<=16;\n   #pragma unroll\n   for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),b=__shfl_up_sync(~0u,u,d);if(t>=d){v+=a;u+=b;}}\n   T32[t+1]=v;T8[t+1]=u;if(t==0){T32[0]=0;T8[0]=0;}}\n }\n}\n\n// SPLIT blocks per token, each owning 4096/SPLIT channels; per-element math and order are the same for every SPLIT.\ntemplate<int SPLIT>__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\n constexpr int W=4096/SPLIT;int tok=blockIdx.x/SPLIT,c0=blockIdx.x%SPLIT*W,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\n __syncthreads();int pp[8];float aa[8];\n #pragma unroll\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\n for(int c=c0+t*4;c<c0+W;c+=1024){float4 sum=make_float4(0,0,0,0);\n  #pragma unroll\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n }\n}\n}  // namespace mwbr\n\n// Pointer ABI for gemm: Q,SC,GS,X,C,OFF,ORDER,TASK,SLOTS,PNW,ACT,Y,SS.\n#define BR_ARGS(v) (const int*)v[0],(const unsigned char*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(const __nv_bfloat16*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12]\n#define BR_INIT(ID,NT,F,V,FM,LO,HI) BR_CUDA_CHECK(cudaFuncSetAttribute(mwbr::gemm<NT,F,V,FM,LO,HI>,cudaFuncAttributeMaxDynamicSharedMemorySize,mwbr::smem_bytes<NT>()));\n#define BR_GO(ID,NT,F,V,FM,LO,HI) case ID:mwbr::gemm<NT,F,V,FM,LO,HI><<<grid,128,mwbr::smem_bytes<NT>(),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M,0x70007000u);break;\n#define BR_INFO(ID,NT,F,V,FM,LO,HI) case ID:{cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncGetAttributes(&a,mwbr::gemm<NT,F,V,FM,LO,HI>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwbr::gemm<NT,F,V,FM,LO,HI>,128,mwbr::smem_bytes<NT>()));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)mwbr::smem_bytes<NT>(),n};}\nint64_t br_init(){BR_KINDS(BR_INIT) return 0;}\nstd::vector<int64_t> br_info(int64_t kind){switch(kind){BR_KINDS(BR_INFO) default:BR_REQUIRE(false);}return {};}\nvoid br_run(std::vector<int64_t>v,int64_t kind,int64_t grid,int64_t M,int64_t stream){\n BR_REQUIRE(v.size()==13);\n switch(kind){BR_KINDS(BR_GO) default:BR_REQUIRE(false);}\n BR_CUDA_CHECK(cudaGetLastError());\n}\n// v: ids,C,OFF,ORDER,TASK,SLOTS,POS,T32,T8 (T32/T8 written only with meta)\nvoid br_route(std::vector<int64_t>v,int64_t M,int64_t start,int64_t meta,int64_t stream){\n BR_REQUIRE(v.size()==9);\n if(meta)mwbr::route_live<true><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\n else mwbr::route_live<false><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\n BR_CUDA_CHECK(cudaGetLastError());\n}\n// v: OFF,IDS,TW,POS,Y,SS,OUT\nvoid br_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t split,int64_t stream){\n BR_REQUIRE(v.size()==7);\n#define BR_FIN(S) case S:mwbr::finish<S><<<M*S,256,0,(cudaStream_t)stream>>>((const int*)v[0],(const int*)v[1],(const float*)v[2],(const int*)v[3],(const float*)v[4],(const float*)v[5],(__nv_bfloat16*)v[6],(int)M,(int)start);break;\n switch(split){BR_FIN(1) BR_FIN(2) BR_FIN(4) BR_FIN(8) BR_FIN(16) default:BR_REQUIRE(false);}\n#undef BR_FIN\n BR_CUDA_CHECK(cudaGetLastError());\n}\n\nnamespace pmz {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  uint32_t W;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(W):"r"(A),"r"(B));\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>\n__global__ void __launch_bounds__(NTHR,2) phase(\n const __nv_bfloat16*x,const __nv_bfloat16*pnw,const uint8_t*w13p,const uint8_t*w2p,\n const float*gs13,const float*gs2,const int*C,const int*OFF,const int*SLOTS,\n __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){\n constexpr int NB=8*NT,RED=NT*4;\n extern __shared__ __align__(128) uint8_t smem[];\n uint8_t*stages=smem;float*red=(float*)(smem+nstage*TILE_BYTES);\n uint64_t*fullb=(uint64_t*)(red+2*NCW*32*RED),*emptyb=fullb+nstage;\n Route*rt=(Route*)(emptyb+nstage);\n const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;\n if(tid==0){for(int z=0;z<nstage;++z){mbar_init(fullb+z,1);mbar_init(emptyb+z,NCW);}\n  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\n if(warp==0){int n=C[lane];int nb=(n+NB-1)/NB;if constexpr(SMALL)if(n>THR)nb=0;\n  rt->n[lane]=n;rt->off[lane]=OFF[lane];rt->nblk[lane]=nb;\n  int u1=nb*NT1,u2=nb*NT2;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,u1,d),b=__shfl_up_sync(~0u,u2,d);if(lane>=d){u1+=a;u2+=b;}}\n  rt->us1[lane+1]=u1;rt->us2[lane+1]=u2;\n  if(lane==0){rt->us1[0]=rt->us2[0]=0;}\n  int U1=__shfl_sync(~0u,u1,31),U2=__shfl_sync(~0u,u2,31);\n  if(lane==0){rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));}\n }\n __syncthreads();\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=FIRST?b1-a1:0,total_count=FIRST?first_count:b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = FIRST?0:1; pass < (FIRST?1:2); ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        cur_pass=pass;cur_e=e;cur_nb=nb;seg_tiles=0;\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(SLOTS[e*M+idx] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= gs13[e+32]*256.f;\n            G1=__bfloat162float(__float2bfloat16_rn(G1));\n            U=__bfloat162float(__float2bfloat16_rn(U));\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n        if constexpr (ZSS) {\n          if (m == 0) {\n            for (int z = tid; z < NB * 64; z += NTHR) {\n              const int zi = nb * NB + (z >> 6);\n              if (zi < ne) ss[static_cast<size_t>(off + zi) * 64 + (z & 63)] = 0.f;\n            }\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v*fw[0];\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + (size_t)(off+idx)*64+(m>>2), t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t pmz_init(){int dev,opt;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n auto set=[&]<bool F,bool Z,int T>(){cudaFuncAttributes a{};BR_CUDA_CHECK(cudaFuncGetAttributes(&a,pmz::phase<1,F,true,Z,T>));BR_CUDA_CHECK(cudaFuncSetAttribute(pmz::phase<1,F,true,Z,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));};\n set.template operator()<true,true,16>();set.template operator()<false,false,16>();set.template operator()<false,false,8>();return opt;}\nint64_t pmz_smem(int64_t stages,int64_t M){return pmz::smem_bytes(1,stages,M);}\nint64_t pmz_resident(int64_t stages,int64_t M,int64_t first){int n;\n if(first)BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,true,true,true,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));\n else BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,false,true,false,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));return n;}\n// v: x,pnw,w13p,w2p,gs13,gs2,C,OFF,SLOTS,act,y,ss. Experts with count<=thr only; W13 (thr 16) zeroes their SS rows.\nvoid pmz_run(std::vector<int64_t>v,int64_t first,int64_t thr,int64_t M,int64_t stages,int64_t grid,int64_t stream){\n BR_REQUIRE(v.size()==12);BR_REQUIRE(first?thr==16:(thr==16||thr==8));\n auto launch=[&]<bool F,bool Z,int T>(){pmz::phase<1,F,true,Z,T><<<grid,pmz::NTHR,pmz::smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\n (__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(uint8_t*)v[2],(uint8_t*)v[3],(float*)v[4],(float*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11],M,stages);};\n if(first)launch.template operator()<true,true,16>();else if(thr==16)launch.template operator()<false,false,16>();else launch.template operator()<false,false,8>();\n BR_CUDA_CHECK(cudaGetLastError());}\n'
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
    rp=[t.data_ptr() for t in (ids,c,off,order,tasks[0],slots,pos,tasks[1],tasks[2])]
    fp=[t.data_ptr() for t in (off,ids,tw,pos,y,ss,out)]
    owners=(h13,h2,nv,ids,tw,pos,tasks)
    stages=2
    while stages>1 and mod.pmz_smem(stages,m)>NF_OPTS['pmz']:stages-=1
    sms=torch.cuda.get_device_properties(x.device).multi_processor_count
    grids=[sms*mod.pmz_resident(stages,m,f) for f in (0,1)]
    side,fork,join=streams;grid=1056 if m<=128 else 2112  
    def route():
        _=owners;mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)
    def body():
        _=owners;parent=torch.cuda.current_stream()
        fork.record(parent);side.wait_event(fork)
        with torch.cuda.stream(side):  
            mod.br_run(p13,26,grid,m,side.cuda_stream);mod.br_run(p2,21,grid,m,side.cuda_stream);join.record()
        mod.pmz_run(pn,1,16,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,16,m,stages,grids[0],parent.cuda_stream)
        parent.wait_event(join)
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


# ---- h200_lab wrapper: payloads s13ab_payload 26f061f9ac04992b on base S13 62eca97e ----
_LAB_PAYLOADS = [('s13ab_payload', '26f061f9ac04992b', 'import base64, zlib, hashlib, linecache\n_s13_body = zlib.decompress(base64.b85decode(\'c-qXmXP2T%*FXGzK80R}1|9kUMn*A<5p$k(M53Z3Ns8&S{|oJ&nLbxo_lxJn(`(i=RPEZeYscR%`d|P0*L<fdBBHRKXa)>cHbo&&{;LwuwSc`713TTa->d*M#?Utl4j86rus(@pv$ps>APb^q%RH+D42M%>J|Js`V_V+?dq>!ch-G_<7}%Puu?v8eU=!>8+qW5O%B*Gst|H6t0y22gPg<Zlt(8A$bt!NVO-)o7N&A$<VpZ|E?XO*O0nrsrFCaM@Z_9uTEc=u>pz4AH_BKUMcQoOftl4^i4e+|*`DFf{TSezs#R{Bcd-q(Y18ycbkrm)XML&HHSo&{%V1=lFg;qfGcMb46R@;gK{Nr07P!0rs6SCGXE#<%z4tBLfUe|;G&l<L4ivIrP095`_4rsb;i7?<m0S=bRDoTJ3Kz1Nc1`;^@BQVYa-Rqi&bo+xupf|w+kgAJ}<XEDR#|qyAnvF3)`wH~_{yp%f=tko0djJRj(n%QpqB#@DAOOR0tabn!`aiN}*&-`^s!dYZEn<r*Fp~`kc@anh5vc%c2WS%e9!L|f$LwEN`Y#|J;L-zJDioFu91m_lGIh-c%%mRyhP7>tu^g3Webre{{x4s#y{BHE1S%_Q0aHI&0YDmv2cFa~bLb}U3>GNOU-@!Q@oawxZUU)aJgGqf4_JmG`#uLW5BQC<zCM9uoDLX4PZd5@i_kAdzTiXtuc7kSkP8HWEoI3IU_e=l_v6XbKa64Od-3H}5|9Px1NaZ`ShlVP8g+UW_zS#My5(^fSX|la0UV4_?~1I6tO-mA*nD_Fz;-mz{E*y``2*kuy(zYDDEK8XkkE9r=W*X&d%Qg-2oQ4LXwi2y*;0Y?YBo@U<(uKdM*`57i12NozHBav!rLpZfE<#12HrHH10+zlV~ar7v|fJ&+xLJ!?7y-KaJBE>hUGpO_=ghWzQug|J)isU-|Bwo`n&vi1=Z7o<nz<PpTYK&;!okef24qqsQrj^jTfH>-vXxtK^9=QtnZh8guFy;)hu9ombbq-hHnHs67Y45*s|}kvhm~#Tvm3)Ko|Gbf<<y5=p<GK*4V!M7`PzVTsMJZ1NXCKFl5Ur2;c9%oLZn<VMhIFuQvAeqX~egpPIhC1opZ7o6NvIEFks|2&Q#Y03P~uU4`WsURMN0!67&RVS$ERd3r1i(hBe;@yYr_7w_M?z+<c*fo>pg?N~!B1oG}HJr)8cnO8;jnbr)^M0{mGr4E>2Q{4)<R-jw}HX!qsEd)UB(#=9Z_*mpv0YOx3wh$<j`7(Gd1T;VhcvyvRKYeZgoeOmP<ys)G=&bOMeUs^crO1MRw0Ho-x8wkgokV%Nv%d)Ok|7tU{uckkOACSb_xIel!S4xhUaj!{?O*@;*Ed=9Z4ub=pHC`l?>?WOP6KSjTc0ETe!reQL;OAAr-^Sbr}rP<9bj7Poi_{yh#+c~@2(%T{wWz*^CR@zH(+m&O_^uOAp>u3zXk?_U+?339ryzY;@40_2+QCEngH6!5Ev1N1!O-jfdBmr=2rp;OT+ND-+n{}vi@7OJg)mM5Ck$P_=ls(EO5GyH}vB%`(+~gIg$RlB>j6$0{=4g@d~Cs^33b!xg`6QL-r32*{>X)Q_nN_e&I`feaZ<^$D8mh9|7#`Q;<Ig({G<L{4tuq-#(7x4}PZE&-Gsh!TPUqKc)GD0`R(jV_r4_V*UL4+qZ9mC<Q>Mfp8NQ=pcH}N({38_rURc-~_0^*tb`;07*#$IadH#RSFyfBuEhfQL#k8{|yEvwQLjO{0Hc5{G>U3OjB?0@16Ww`$84a$6kO^)9t{AU_XAH;lty0thG~Q?$@x*Zh=@IqwgRr@KhT5!xP^{&DR_R!|yxsEXZ5YvLX2A4-EhN4<6=Lw0$2%wl8v$1&&Yhx(afv$Pg&|{W)r>{`2!Ruh8Hffqe&S89{VqUIehgU^$Q?Z4aa=pc07i&Hr)m|I8f2(O=BL^V=_}fLee^e?%Ch0n9<ZsQ{=v^`^+GY%@=M0@Mr?@jc)Nsiixhwp!%x=mX<G&VQQ#+WG1tUw=Lkphf<CQ`s|PsiNv*5EHoD+w(;Oi4Rmn@M?<I6NxaKqVosqIpk<>e*-Y3Gr)2H#R5@xAYqH9MNUjd{9il!DW5UCr;3?p3T8mH7W~RXQUF~7j0`f7$uiGl@jk}#g+%&3P<sOaSs?j%9O%-2^ED@Hk`7*8b_PO$2;g*(<DbmnKQ;dTJeY2P0tA%r4}3W~^qcs@LEl-!5H$fnNCEycGCrn%qJM<d6JkOFYqNg6{Qd-omjB2CuOvZuZUIszeVF-jdyv(iIOZ21{aBiOn%@VV|AtS$z*C<h06ljJ$XgARNdbQMONGz;49@5C`}ww$1z<bo*?m}Vf4}?1M&3HQV|^S~2Eg!>*w&2$0|o#cKqL_Sg*q3oPhTlt<;|y-fG7Wx{>fbhE6(rFf(Epe3mCrQK{3eYK(Te7ldtId_B$&6kvTpS$1elEJkw8<KrbpBDEz~=KR|r}T|!{SDvnt9Jq&s(^1~JY{_dU~8CakLh$5bOzs3h%a`-(53;fx8du{An&6io$AYcODo<Bi+fHQ+b_}l&?IDSd>TKzN<`CQ~bf@MEnHhtwMpkx4o$d7yj8VQhEAme)=i4tj0V_p)G;1_uI%Q*QW<eusV<@3qJ`@b*%*0ZdCC;TT3z>i;vKpr}X(D}6!ui4MTyj!9Tyq<Lw8+!9A3OEr!aE^Z(KN6qIFQTbikgu};b_{TO&=kE``_Mx-eF=Y_{U#Yij(vJ*+t5yauJQHv^X-Hsj6dG+=ZjCZwXnqRXWV~$YWI7<Z=V1=ApCd#e1Cs>+B-;M&`<Z>cWB1&!~Y3TTpyr(hy$3Y-#<JaxV9v3Uj_v6>PM7m``dhk_V+)l2Je`r?-Za;imVFx%6)&O>F2<gWCCRRqhx~6wtxeQ(68S5cfP1UAvHjSK*Ik+3m^FJE&K<F|I)+%N*>!M6!^GIcxRZ8H4O8d^@CQBc^lfkPT}heq$<f*+xA@s?XdM(?LPunw*3Bt1w8;HeSg8xcj`m@33!8_?Tqy2PJa8b{qd!W?0m=kx3!SJ%&)Kv12X^jCl_Cb{;U7|;ilH-JFn;#Xt)>#{>mIo9ZtWA!CulohW?G-R|3PlI|lHNmmIH=ubkbVhrfLc^U~cX^>5?;=?U&{Kk#h^34R^1O#U}``a_k^$(PGbp5FRZ>4G1oe%cKLNuS5-<;N>CjzFk_4&jv>rMJ&6^hJMt!S`Ql;JeS_WAqnW`2P0$p#G>YG2^d-bFbt4H<^7c{UPhqe|+ZwlRy4AIndkl@Zk4rBT@Me%pgU6>cJNd{s&i3O*x<OMK=CW<>wau>>pV5*{DL2_QAtn&-`vxaQKI$eUzd<!rbySerM{MeEF<XuYQ;OlgNi+jVJdP$?xl$4}|%gd;SmqcqQOpdp$?rU!5<g!2fgW4=4J#qWF?kKGGQk2LGckf$@L((g(MnzcS{3Sn&_#e=f0~+S7gQ`N#2J>}U7?ham9Zf6Dx4d*V0R&{hY0PoTH{o&4UO$lQ-#G55mUEO1!AYXBM5(zW+M@AH!5bq(^o>Igu2(!T)lv!s3iyTbuRWL@#&{tlGUfREKdBk!~Ll>tAuf1#02slhNPn12jBC4Bh{*vt@pq|nnf@!NlL!GE6VL#gchvnKjf&>wa2(;ndE7i9eWd0qg5{)6?=haZgK=@Vjo4<H!)4y3ao|H1V!kmUz3_lubO-^B1ct9kzWmpy&N>|;WI?{~{lA^7XO-`V*{j6M4h=(jw9zy1oO&|l6Wy@A?()$V^?f0NwbzHs`j%I7%&<IfRL6nyQztp7Q9dy>5(3)sXr8T$VJy#b!cR&&HZSN<WF-_<^4^Gn$urw47~2i5N}pM=wO1^S%v^$p_ZDJyWepO23TALsmj`6{#KsGy5~_Mzy%3&(;&%Sk@7PiPL_KOg^O<jeh(<>RoHBx|6CL#zbc05m1P!}?A00gylW+gzUt_pSbX{{D(72&@kvKmzz3G$z1T0)K&*kLPz@VO4|i5j7?v7(jqu@|X11#mVcF+mH98bbnnhJU1!?-qFu{h_^qo8z^P3{OR9pK5rv`rwtzr1dzh>KJYmgaDIITdA)ypW_evQ{_p!rfFgB8@GnjOxS0HvHJ;eb@9IHT{E!EvZ22{K1#QXCzn?y-`K+JB_g{W${;oA&PT>E%u1d+*lAmTE0#cavye|M$e+S|p6F<JreDN253jguR==uBSj~@v3n@Du6*FDjn74#|Tllp4{@^5>;7C8V-L2?i^mQzIEAbvOR-vq$l|4&q3xPQ41&ROsNhZs`;k1w76)cyuh^t&jEzX`x0K3jH3`~DR3k2Li^tLLvHzn1w22i$%<%?Eb;WNW>P3TuE)|K)W6yT$?-ilXmdeE7NQw}XOosj{XA0Uu=i`((d;u|)E%{Px>VKj(m#Tzv!n2z*>h-~aaA_hSrt4n8lEft=#^7t!<D8jyYul%M}=@coxlz5vDbx2HXA`nelW<d<Bi`T4{b?%z|&t2y&^0fO23=@Y%&+4_^=iwPL8{>}-%S0MkV6~Fef#Al!Rd8_?yfp*h{Ov}$B{>AsZbp$2L_jND^`y4)=L5jdW%Y&bX-ZWPG1E&II_|lLcAAw%XeHHp^Ukd%_gz%jB$KZcG450m#>c6&ludhpx`)Vw|e|{MDFZ(~-!T*v(g3jTM;Y7>MFdtu)-u{MOt?2Vp;PVra-+#UV@+WF~`^xLd$|wE8>(@sqzzr<?PeQyF{uc>+hG0+qKWzqJC-Y_5vke8F@2~spqhB5Ht5yHC9&qHax3A66f7l#s@MeH&#XJ$7`SuSUV37Iqg5l`{{&&}3x(D?K9N94cyYA263;-qgAK#SM)!1)Ap1T4u{PiCzUu=G>;NOH7&%ihAk45J{IR8U`zeGH;`v2d3<9}G_zq!Y6?+35U^TPWedLKj|NAssX;$Hmu_Q4s9`L||(`0t$k`bObnsMz!2qkfsX@~pAYpT~X5@DFc!b2NXz4^~Cy#n)ZG{{b}nfhs=dKU~p2m!Cb&A5Y<azWx5b`}Vj0X#fI7ePbP4e}n%h<Gwiee=_(umOt`5)dvFzU?vl!jnB{6&uqr~Y52wWB^dwW4)Dh_+W;P)H!b%+P5Bb?4>S7A{(Ug`mHCHczbL+l_tN+G4|RWkfkpoX{~Pu{gS|cfLICL2N53Cg@70z8AjGPizy|mqzSv*dN8Y!*`T6bSuebNT4&b258Xz(Ir~H~eA?a-pxPJL`bq}bzNM4_3`xkip*WdT263FXw^QRvIqAh4>KB@d?pX_V55f8itKDL+Mt^gG!|Gs}R;bDM2|C{1r->E;oTlv+%zv<kACh~jW;(x+o##n->OnbF%9caRzc=x~dSdc&a?tk%9K_Rd#=(YBp%cfqR0iib@G39&WbTNVUCphEe@%$#<53%YV@v8jB5oB%Z6r+oH>87MkUFtNCt!r(gN^6x)EL(05&H`G(4l&Esig0LuUPKNVDT%36l9IxTcB%Gj<Kw(m493E-!62J8hDfr00LDI*+)Av1$j4b#80n-$8|SM!&u@`<Arz&0bWDX0Ei<&NZ{&-~_vz5dgSRz3c$TR}X@)nrN23j~J1KZ9BVEc#Yd42JQd~T_i8i~7eyTV03MCG{^}IB?%`H-!WIr~iHpA35+d)z!l`q@Z{3K5|`^{FnP~jmiiozjUqDW7}uqAEu+?YE^Dc6qZ-L*|R*yN$qCcX94di#2s7OCsLd=DZgBVy!YN=;&8+iGnW<Ljy7LSh&rxGNlq>4}e3u}hwX$BAnC(FsM;>L@kYuYy!l-v>*{R%^1N+6E&mX54h0TrsP~)hL!u^+=qH)$UV+a_ZY0OGzq41z}>{x0;Pmi|d#~a**xh%#K$mh96FOjBC-YSc^g{tDR7UTrq_S&`K$n)=|$ndO@X#Am->cPMnJM3ad}65+1pSwH!aESM`E*5f@Dzm(hp~4TSiy-J%q4rc;ApabiWddBl$3a68|kkN3u9v*_}JUQ?mb?RHU?puxDsL$zv`L@wbLg`c`ukZOg|8fQ;C7Tz3UO}rH|0sKyFb@Xmtvoxn&FRgqYIWIFu5s5+hNdygh`{EoVPGs18%$90QFYcV4mUN49Cfe*5));a<rPl1Sx7fjoz;-%tvv6q_B2-GQ7bHXqb<I{Q&MtDpReM^z<uffbIkAF7JtNLm5<y<deNr>Db-aocVXc|VK!-(Wb(WA$j9t=`M00*iujwK}*!|pmWoKBd*jHP#Vm5A!#{;N`#@U8}^WHWaI%CH}zX_$w=f(`%82psXZgz6-)ZUd<ZDh`JN>mngdfeS@jW&lFwph73qdYJ5kK$=HcLygeyHQGu`)0geOoo*yLQcbRvFtsnRsM44tZvVg_Ks6tR_JE#WS@?qL!;3h*T`Z#>{U>su$t5}Q?oUYc6g=`PF?VfTYqFM5*bT7wX!HOiPHWsgW>Fui7nTt*uGpV*|iXA#%4J(pIjZC-LZxi(Ni+i3s2x`X?&1NA$luCp`y|sIy6x{Bs%NhhU77HRm;t`DPm5^EHJaI(NXS(?;zE?T8jvu9feNKQYoUQG)!Ngq*`CcIVhX1(hJx_t7bgtBtza<#U|puO;yjE12-R`7EVXYS9zQ&-=$f$b7G_Y1be@tDBthav03ZZjvCGxWy{f0CDlBS`tee!6Ee3g{k)J%jUW>{OF~k;6BNDZfS|nID)}OGO6-v2mcSd^!<orL=~ZyQF2uGsh9cU9@iuA=Gx0`U9<4GRl0lAWd8Qv-T?i(vN3Xf87S0{JPn|mzCWcGY^xjXPsZC<mTldY>04i#%2yZ8+p<5lm`~^FqNl(2FF-&LK6to>?q;h!K74{E0v#`!96x(I)wea-b!KQL{G^6`TYRfatkx*dHSzW7?bVSg-zMHJs`&zEE!LMC<d^$2%v>YlUloZkO&21IZ#U2`4rt%FeMBQu2rD;uN>|W1TO5Mojerq>_(yBdg*F(kaz!q&JjS{hRU5}|w_lBp>OSyE4XPe`K-r6zzXm^a{bR4c$kt9Bby%m3>4v$SnI`+jxZ#pAN+<oPwASKbSiEiFn3~a6#GJ*&dj^0KoVUXIV^73F=+@Ye~*wy+Mn+HiPTIEJ&qk@m{LN_^Lwf?R(F}o{>@6M&l`M8L2%S4b`msw;D-$zVbp}dJ}wbo}Wa=El~G1027dp0?7v$>@@o~^Iv!E{|*)X*5iXHO579KnUuF{H&IC`Gk117j80AJUYgP~%!7B60CuWY(}p*I=SP--NJUcN$k34qBhP`jSdaw&nATh33<nSaDm~&b*HqEivh09+HbUWHv;Ykh6wHscSbk#ds{JJOsU2;*x_*CECOhr_*g<mP?XDgyy+Pec0Zuxb?QWDp;p$(VWH)OBA8WZsYdo&a`XfAYe^QiO``mSz!u`I-F?+o8_n)S|lo$&3d>>)>_MS=&{wOJH3M4D@YYczf>G(Qfy@4Md4KDJL6sM!5^p!w41L4YlU{R*-}HoX7$D;+$pRh^)>wHS^}1E^u=Rpj1MSDpv-b6v;9MPp0j5)>ljfh9xy1|4AYBfag1TB$GP6K7VW%rNT1iYYuY5UF_yiEG?K?rQ_Y!!kr5{5{oJHHo-(DQ5^3p!gLP#kcMLCu?EX<to`=WkhBx5M;9klLi(2#^tq2`4cWZ5<lvv*0a=m)cJKciTf|?%EAt+K|KqX(62dA;UDbGtHu`Q5osW2G3WWN_T<!n$`JfMBN89&<)Le-t>>P#-%P9uXN*fkf6;&E<XE<Fa7W}7;mm<V?3dK`+)X8Uk*O~sQPUdwIo=M<EP+!EuA$V(0^Vdq7Kp!bZh%Q6?F=*@UMS;5Kbeh`ZEnbt9toe*sly-h-Q^RRTN>6|!6v1qShpU+-(khS5em#eb(`zn;kHnL{x;#C>toWY5%X$5DK_`Hs)P$k$+S@~t`kP2>QP9j4t7JRm`I-0p@kvMAO`*g_{%BgX8({t18>0(2ZcG@%I^Je!xvKsrv1%Bj@<%PQ+N)LsP4v^^$4K`}tKA#*OD`Gk^K!Hxznyc=6)NT4eA#y$9R$VWGyDZ-_?>a@^y6dpg6;eH>w6(+Zb%IaNq2{o=$?ZZ`Y*DJnZCUkD5o_+I)3|x(`KuO;kw*bf$5vXaU2UJXSHs|u8c$8aRhu^o36~OD=@F?`r|f_^A#o>>gVV*B)}0zAUC&$4L${J^>C~-fb6qT|mCN~xIg^HK$W9%D-7UgMbt05A+Zl3GEJou7sYVaT7Uu7}aCBMPlotg$dl?2h35=kL>o6FJ=c2)4^E?>$9VikOcqpsEDz&Tv;I~*{#bPI#;wX691eUF}$Eo-=Wa6#DAf#`Lm!Ki1V6(Z@YB}XZ9s76G-V|oLW&yjbQ$n*hG?sk0am02FYK|%jHYy(?!B}JQXbjWLaX)uQm)w$_C&q2;QBsMSqobKcb_^vB;jla~%IjvkHYjNeO3n0;<%z2(wWK6EliX_1QEvD&wL1(n<b2tU#%m7<Dnt7^;<DRwB_0jMI?GI}BR+0dMCZm>(21?`Xd{w2uQ~u}H2e5G+z1y9aZG^Z9=oi@=UW|$#2Zj(9qgBIC$+ykWPK8eEtjQy##s|EKUy_vN@%>(tB8fR5B0>}9?WrOUR_w(@S--`Hj3?~brz7?Xxutuh^X-WgSt$Ncn32k=BYF|&Nvf4Gr`s_oT5@4b2Sp1o;$j^d}ed+u3N!c#DOzv6`1g9^4LW%PpNj{0Nv`Lb@g=U420?aQpd&d9m)kgyE)Vn4A$>9?#mic56&n%SHbgVMT2`SxMP?ry*P_Z;l{cS%6`F<)CDpsi(Yk&98;ws+AwO>!|}`%u>NrYZ%f3~N!*6_XfCd-%2R65WENUH*uO|B-oz95zR~PQMJn?Mquy*vSMj7W-ki{BQ*jO>8e6BA`C(GLGeZH>R)$ofnMsPUH_UW!uSZxp!uINA!Oa!JOj68_$B_g(9+!!zn<DDvJsGbnJA06gjM3&XUEZE4q0CYP_qa<H8+sCF6&&vfpjy!SGR*CIiLP0*$yE-WN_%L%%j;4>IT;d3^?TXrVi}^4=%&JVqs|nyTk9EC&ge$mxNY&x)PS5-$4doMVSQRJLcLVJ(nDfVII;pr;V~uA(g?Xb%TlY|AqF|UV{W%?@7{EZ7u1+T8DzSthNgVDKpboKt&(ep>%qAd+aGp{T*Yx_T&Bcn3T}2?xZH7zbtg{QJh5!f%MMAFwc2dT9F_<wTzV0h+qJxb-s`x%;+WY%wP3KB?b_{5UVjKNv9FQw4U||+3S8#Fb9J<>wvK0FGpWZ1Xt%2uCWY0a!ybjg+I|c?Ej-9&>E;$CAm<ue%rE4H7+6*!!L)Y`(er9>Dv?rFO4Md>@9v0+*r*q2M~GmuAbCtE)6o`>aOaT8#F0*jLK!Mk>AQCF6o=|_<vd+WgVow)(Uw&PJxjC^ePj}eexAhJXNim)cY8=?&pKy~(FBNnLTRB}tC>#K7xguppR<GBb=f<VCUNwb33@GS&^}a<gBYgaZRWajZ_ZU6K1#uT9zPsIq0^+!m-buza2ocCZpr5AGV7Mh+f^){UOtGPLM~&bAuOo5s04HEqvVCllYD;=D>iPk3zu?FiH4nz7IY*t4VGj!+)wH_uM_3uBUv#gl}C-k=d;1#j3TZ4vh-kXp<s8}ovB$i6z)%A-I-^>^F?19X^C_8?A?@%q#JHEsScA}e2Fq*VtVV2lHs;?Eyk>x%?*%)F7Lf-3lk@o<gy`0b-NHfKgMgEcGJbY$(tIq4`ciN*j((~T5q+lO|*o34MylqJR`>)qQBXg=}Ooe!ZCN<x3@Q!idO2Wa%zLe?#X^WbBm@@DOE9Ojzi0%{v;1qj{SO1O^k#MLnWG%#v)3rZjmu`z2yr<hah{~(YoSwDs|xzW;#g*XVWcU7Ei3!CuDf;#p{B!lq-hV&R;{YZfco$p}=fyLafVIWh|xei#cDVW3iciorl^j=A1)2^<HZpWEWvqum~XnQ~WSm4rj+Vvp8oWF%UnGLRC(t_wK|+kLnHHiEX@AsFpJn(OT!2@ZO6y3zNi63YDxNVBMvZ(Bgc!g_-IRnx@)_VxEpyhVMV9npW;Fy=Ar{a;CH4c1MTIb*~pQWL<IdLCf3f<2vOcRWY$1aJJHQ&~zz56I_>L%-L2S_fO^PMiDb^aId&nbKF=BqIBPJtS%ytqM>W}P;abL*Kn$w?B}xETT_foN|DlH!88$6G$L1W(oByQsZ+mgN|ThU?d5fyw@<qioT~TX!j8nYCFL^d?Gcopa(WORN$u=44w1;bt_4XimT7W}t9a>P)nU0$3!`CY(Su8ht7o^z<uX<cRuKZm1WRRwthYMj^`zRsa;dYMfa`Qy^OA?6>UNv!6u!Nu=FqBYlV!JcOQ4M*v5`D=F{3v7U}ANgJer)HZo2s+qHvk64)YOd(%KP}V*v~%uD8izcx14hc~=E}eR?!0S~0wH(1t~#p5_NyCdACj-r*z?P8_yy{F0jC$w{no826H5Jvc~B*WD$hL;9sd_m$psM98B}WV4JjbUB+cqDCW@n8YvZ!F?01jFx;S7%J}6&11g|rn`5ZN~m#_I;CdqVV^r@iS?CRs;rlaH-LOJTy)oMe%Q?^RBbWr>z2--b}(+(2h6+c_k`RyLD$2PS8YO{2Yc&=l4^6urZQ%@QvP0=UWVdMONOGjRNiM$v9l<#(^_;;EJNw~{187|=`^yLaIl^)C){+f9)_l)a<X`yafX16LczzX2-@`yB{y=>Q&8K142;$n(qlF$!TIhZG7HsOc_UZE=@WnOY%+PC@1T0Oj0fu#jmk7uo1*0!>oYwnSj|b6pD>q6SL<k*^PVMd;jE*KgZIYcLJwoiFeIC7#e5`VrFL78VvYV@cNsC6nBksS3b77ySYNv>y)!CQ=YtNGYY<ph>Q8cmwOzi36$Y86;~pB}2gKEg9rSyo(JVGr*X_CI%uh1mv{`m>f`!vUcWNw@%3B*P(itKrt}r`3MnvBoV~hTT4J(%zz8sBurIx@FWhH&gb2Jx2*TQiooZ9Bmn>(jswQPp-!P=x3L9Wha-}JJVNZf5)bJCp62D7-ia42Y>DNyD0V?P;XqT5Si;1;+s{-E~}k-RlM>4af72sIu@OWtnS3)BrxMs$We?V6o}(cG>NYkJ&`r`oGsF+7a%VX>xX%juZt=wydHWfzZ3B-qG<ra9cG&yAziPuKG2W^G#Q3?V7J(UuuFN`{!th~2cv>I$od9)*y8FoTmruv6VkaD}+A(WH~>s@{a_u<0(r?0co^NuN12nbL0g*%Tg#qkKIjBznON=E1eF+wSr-RDDF;YnOAE#dY{3b)8sJNL$&Q+74%@W?VW|T0(*?_ZCrZm20zgUiY*WKL!QUWW8wnAteWr6`MiGD4b5gS!`NQx7G4`m5-eWsO>rFvl)rsr}<*r8fAA~Y(4Dh=S%B)YZ3TrKGd*dpX}vES-XNmws}C0c;3h_;z`gE<0U7LGLa3uDGAA0;glzcdUI;e+)k*39P{FAIYPCVoej^A(cV_N$Ma~IiF10SihJ4NBfGxDSZG}16Gk*N!ihe%cnI~LaNDu{Ww?E&h0d-#O(*9X4lSv5HK<Qbe4(DE6)9zxj%6*w^wGRDPh<HpAKTombK>f@3Zoc~XQ6Gf2^vCozzt%H*tntf9VHl@^ydXK3I&6e6EYL!A@xW`oPwF=gIVa5NjqE+CJ~Rr@=?g;jr9E}_76}IyV||{mXqYI9&Zedx)&Xf7Alt9vufyyQCBoxJL^bqdu>*@WA`!A%2^zM*@$uo9)ozVbC^C3mnKHWZ!NFhABr9JEQ;DC2GNk+LX=WBK`)2$Z5XOSlgyq39c%Yoxjj@@uSS=rjKO3@zR*5nyoS(>xs}mPTG%VqTOx71#|FFTv2*e^t#VANN;q&wl-R<pid?WfGBzewi87WS^;b)4g)9;|GE<3F7~I@urkFg2r_^zc>PBmjNyfG8C4t5Xo@vh&-Nu=<WhW+-I*L&2jGhw%n$FbDtysDn9L}eyNU(Owp&6cEm1Bz;g9-(_S$$OT>qKRj%wU&jqS@OZtktr+5;^UsPi)sv5YFq$p@Xq#rki(W7H>u2<BCn~=Df*k$y(jX+tQrSHo4)Nsixg|L1<^P`&=%RfOp%=slP1ukX~EBu7q4TDLSiZu{+N!vd6Xz&5G)=Hz3W$EY3?SFPPWJoOz@7NVT&q5R!9iEeg;?(Kd5DU20^CR8Ynyv!DzxP-ZmSq^eMn-#n5OVIXtXDhhDFKpw+$c5xYquzHRll_Ke0ASISChnqonG!`IrEC{$a%%Gi9Z`>%_%Bj8|kz%d~5L#Y|*<Q(B%K7W*;5rTyk{dET;>V?ppjqX!m8h1xN@>P2ExzGO%DI2w^Ro`G1{JiAT(g~IdDLEO&W+d8s#QGlbMp?fyPZ_tW=HZOf?-@v9(Hu{)-_i>i-|$Ue6M_p4LPyml`i*tXs9!4dc97UPCazNj$pVd7bE02oEfIYNFjg8M6ctOhd~!RWj2K&eKCpd#H&+=p$*UL313X$^MlzHLUrNV23}lsbbjobL98-$nHb#)^P`$Y%(nC1As6G56aECRNlX{}-E5aT4@VEq4mOB1-{*5kIUd|c7Tm4^5k^7dh8|fxrtZnHeQ;OyB{n3BF4bw%^pvVXXguLw=G2H$4~f-HiSF|YryuW!LJOiM7)LQ(80B=ft9KjC8oubyXZEsjZHCldzE?M_xoh*sbu6Y3b-B0L7as(3-InRqn&kEzt|WpUxipwu<)EQ+!5wA>cP=DC5x6Vy6zq&I+qr<c>0V8bcnNb=NhGSRKD(526_cneu~SZ-^;;QEfRCAKlqlg-Zc;0S!&7T|x2)?ZM&FWCWz!!v8j9tWafPgCecX1F>v=uVXhxx#A<vq<je5J@lgxO?v-IxL?4MUu@LHcVhSy$=4i(0w(^^lRmy$Iv6-Q8~QJGgAwbGK-#7rLd)ao2|jrcmdjsp<6!Q;DY*}lwHq8^_N$M<IJdS<*{V>+-~F(LxdOo)wgGf_CUEIvQxPRGfJvf%9^yf3#i)_Rt#hFTOnWFD4WLEZQ+T9@U=z&_hn^V04JIbB~DO{QyV8a|J<)Y-a?Z<*?JAIG)TNr$nG(azXI5ee%}fZ?KKujCHMhtvo%le8tu!I`Gfs*aD3YP^^V<pv6Uj`Ps9>eND$BU@bK55eu5!7FK7;c8sA?{%SpEM}v8`m`>rE?uKNf@~o;81DtD&kiEyxsDf;9#bZ-W_voy7OxU!U+>WvwQ}1Dt#jQRLbnH-PnK|K)2*(mqmes2<`+c$VueaYbVs9VE)&tT9_*O&EsaKxO0cD0qF4k{Bm2W_QQJV;E)`5-;R3>oYU!}dnJtNK-xoT$D`yn?ae1_~23JB?sS{niHXsA*Z{|AEZ}h9<)<VkQ1s&yT!+lUTxmhKB;=O3Jca5FIyLW3<CLv5Z+!owzB&vC=zs{}@orzJ$;w6W&t}<wcK{>4G&8;$SOD9%JFU9Dd;_L!Dnd{L)JvndjTs)Q-dC|<aHEl%jv6RoAh`6jK9~mP!4k||?SkH!g$hE1&>FyC`>|>d_mW5U&KhP5yl-*9S<I+ZhX_={v<&f3f^L(@+1Z(vD)QzpHt^vD)emZ{LwwO{(z82eS1xl%6FFDhM%mOrx^>&8MHOtF!)Y~Mn#wNq;*A-;x=Jt=y`cQ<@$MSeNw$)&0Tq)Ocr%doJbw+Uho`}<V`@AVtxaHUyV|x5tArI#!W}4@@f;Mb?I~W>QbKR>RHrvw-YgCIRP-Zs$M|zbH=3RMXh0D{$LUm80YjW%zoxDjCq<lBqB@rGVbg*X)EEieq!#tXvZ`|CpnBf=uYmlMUcGbc<@To%`6V#mTUCMem+^dyG(uUYZNjRFV6hvmEv|QX}b0U`5SvjUx7*N|r$E>d}_`tb5B6Xo-rHx5Ythh28y7Raaooz%KhC@m^l5@0Cq4JOp&HZB9Yn3#6);(UjY?|edqj1)cW2J|h6*G@v*6v|-m`FrVPNPr9bCFB7vDZm8MZ}rh6hzaIT6fO0((d;c1*$*Frmc80cqSB1vcedbT~co0uptgS9VXgUWU8CVes^Bb1c3U|l$`YXVK-I4;j3AZ+oN12DdM{=Xiq2ms}c@x^NmeqBk1EupS?o|p~UA>O^(a5%e%cR_u^Z7X|@?fuF2DA<Wipr^Z>rP+Ynb+V+p0tHCCPS4n0^+Q5#=RiTL`uxQEiGQNG=dkyU(FYFC9$Q842XKHbZcbeyNNu5+AgYrNR+nAAYp<2b%P%JX@u4l(LrO(^m;o>&|8sl^5xlwHKo$%=kVls%(6BF5CiSjRT@H(VZyZju_T;Yl=yZPi@1RB*j{o3HdRXBAtL)T)yZ6)FM|cx@RyQnmJS80s`bHKIBo!=<woDNfL1e;i$f8uODl!^W%)P!HTkhjzD=8<3~^{B|J2`&JK+m3MT!7rpJ0WMtAvFhz%%$8wcsMXBeP#jL#CE{swc@r0~!ys?dNJZ&}?NTX*TvW=Dtu-SIHV346=F2alSAU7IFmAzQZoSI|Tqk|`-KkAlRD>m)GM^LOpIm~lLAzVf;wM%C_*;gA6I@3?)jK%a|npqJSwQwV!*u~_Uc`V3ma4^`0K!?yOXtch45Rq*6p0x;O7-T1t%AscLxsA(2Y{lJXx#7CNwVF*_8l<P`=sp@W-1dpUC9{KL*KM{oNq1@4C8V%>4QBA|`IZ)=7@E38_;Gi%3N7f_qB>$?t7W2FxiSr5P21TMUFXPQk8yhGeu!+PtKojhyJC~_<WkRP56X@iggp0QR;r~%g*}F=gJTs9MxgPQ#01r--$U^kqmfrs&xM0igwG~W^fm;DbRU9ZSF&3=Sj&?OT?(P}Wxb3<2I?+P6Nzrj3dL#Tw4f$OjhIA>>=o1C#3?*Ud2TE;np{PDAL}MB=?>24&P^y%%N)XNwFPa~m-U$M!0>uLT&6ZFWkR!fWmFr9l6V}?nCneCE~t9PE3{BPoSP(wfs%0wv9UI%%nmlFwhyt!DY`-Q*06CsbyW>D_|zSOR^hnF6SZ}<w7fX?JljSW#G;$qCi<v#aLI#6c9SZ>^UdX>N2RL4W_Zp>k-lxNfx<@WX!OW&sl;p;(!4a9h@UV)laE6_c-ScUkUR|UnkkFk8zr=z%j6cBS~^~#jN?4(c-eY8-kR6YQ3#ag-Z@AW*M+`)*{kg{xmv4S4|i(kptNJdGIAyok;#5^C^=N8QYag(Wp$ABjPXWFi<5znh=tc}VsRMn=XW=}B$lv@q~%<saZSb7mrbft6<GFBjC9w#Io9#^d=PC2<w-)QM`k9Sfy@FEX;h)!rYj=Na3vZoln+!9t;TQxy;Krx-431SlR}EP?MktFyL`Xcjuz%J)$m0=Cu)oNKC@2INwdBTo+ISVEkd($VV#qr!}u|8JBv)7a|^YYnVyEE{=$>0(`u)ISK!|KFdgPK3f)TagU;@6JxaTTUD`@7s-QQU;rc)g)^FhsotbRXw^J^&5Z7)g9mGc_w`|m=3CfMdM`fGe?y%b`+nQ)G(o$kY)<~@49h~UI+4C9KljFo8%BWq+ZmH_x6pc)T-2Iwxr-ijxmTSFBu_kgT91#oXY8_?kLXjV9bo9_HawRW+(XyOcAPW*cYYi)V`GG`F`A|2t7$%}dH3L25*t{^?BJr_qdiQ$AI48=}QTSMK<IRjQ+%!fBN;%~0Q@F;QH&c~F@)=RpdnJKWaIdf#q^)2t0xQWfmwKG$=xUhU6icV=s&gA~54Bz4#|$5x3tA_cZ08pqpO;(nId0brO(qhW8a6!5?>Y$0=UNvtKFc=Ly;&6Mafs@!x^Ps=1ee$DNsLE%WoDa1uquSsJ@b@_Tyk)JJ(SGdqTXIkGV&o@#A>)zMh>%)#(DBCTWNQlqW6%xUAu5;ohIu{s+(N3BE?#Gy}0TgfrxNeSzY+`xZal%r;CSVX46p{YNt9qXVu%u%?DZwW-z&G4zxpO;MIv(evoJ;jH8_E4EEccExS>7HlCc$*F$nPi8tt0G<Vye&3tZT4KuA<SWfduV|^HeyS1FFZu@dx%%YPPlK@zvbgsz9^o^r0=V_8$EblE_2_q9lQ5fFHP9+iTwh^zKJ4)4fhger@BX}EMt8RZ_swCEu=aZVm<@$Fv)wIG;gDxz%mB6R>nX+OxYem*^EP|VyqtqhVh$&8y(_zQhY^oPp9-^g7nb(?(ns4@MhlT@3)k~BZOgCyD&+klZ7p~9a4ugh@$E0Jsef6@W0l>{k#hFu@av+b*db>!_x>JSSu$Y|hty*<lb>l^5#v4VGZ6Y(Q1sZ~SE}kzTJU5*&m6(Pt^EGCnZcDpFZH2ThyYY%CEv9x`N27A~%<(un4L1i#b5EYeExuJM5^%V&Q_}E&&l8c&JWM~r2Z%sY%hSG5anFhHG+cL#;;ANk?Lx8%-$CPo(r52t9+Hzp$ifk`b3|9$Jk`(Moc5~NnhX?gm(R7j$*q^*&MB{V9G_S+r07T`biXyTd9^Yg=uNLI_34xc)m!9!GqO%58|hU$%X~901W%P!E!-hf=_IL{$%GePQr8|SU+AtkC}b|w1(mW@8t-;GeELcy>UwmgZIEiemSivIPIEXvE?ddN{wfh<avQ3^b>tim)rOZ8+Z(1sxAGZLXUejd&qUQU>P-`Cw!}gv6f@!eIvAREQ(Wg>UR40#JK6(2p-8DpG=*KvcH~C%=vn>}y{_$H96nMi(t`J+R-rnKHwJhyh{mi8RXh(lY*VUg(})+RDfwm&xx8t>!$G>TK4&gU&@GdV?5RNU;!r7>;we_Y9}`?_r4{Kz_d3fHoqn^D$rL)oIFZ>&iOVE%lHrjkHZ*~^R{Lou%{um^gNBm1awj!DCv|Eq-REuut_=H!<zm|2Q@KMk1|36AEZ!VJnkOzA^W)ts(xv`N6HoO=SBx|jc^@CD23CYXt2Y^)+nlo<Cy=XgPIZTqy+B-uWoOK<TBynBRTj6bAzELQMu(`VXZelMI}fk*@qIeHaK+nbw+&&Y%1zRlrj?3W@!9x1?*%>Y;)VB_ELJG?Z{0CE33uYTO?B0-sj>L!K0lUgRFAxNyXDiWI<6;$@rc<+P4c)X47ifqy)`{-c#*@>0g27F@$fNOtgVjw9h)~1Z+O%urS6ig-f62N4Hi}jr_Ffl5ZUiHiDGo^%p-+tq;$J@XXAo0#&H+sJL^SH?=78q8m1bBzOWisCJWG!^zbRt#OSV(;k<6I7o|=iOjcVnoZlx)db#hF<N7XY$89mUYY#F?XQ?S%t9IF^lcRLqPC0kET^)&ACtIjRKyf%G^^sL-FXo7)kB#wKI4A9Ry|CS6>kPjOxA-NRZSIB#1-fA5Xo?Sy@BvRbszw!}05T+E*frgrw+1-EZE7iujtD`u44?NnPnbDK#7WQ<vuU^5TLht$T}s@Ba-8qhj98dl_8O^5FLvva^9kIx5-SYq7*b|JFyrYewJ^@&Xd33GB`^1YdZM@ESmSKdJlny~kZa+zlYokc<r*FMf6%fX!nP>W1vpZ<K9I`_DWdabv{2=R8}CdMH?bIOqcs&fRpyw=b<-if(9MX2Tqk)zF3CmlO6J(vOo=(6PGVP@m%_7Nw3un{V+VCQGRIh{lvQ)J?0nj5HxqNG4|V#ooJx1Gb$d8imI^JQSeg;(+g3YO0l-}iCMRrIIu`fDnr1ITE3Cw=W?d@NMx>h823EAOoN5u8suj;pu0Nt%lvRVwt2<`!XbUOnO@+{Td7SOt;^g$UUyg@X3)hx?Wf_Zd&CzD8R_($jIfM2+G|R2Bd~ckAcQLl@B#*-?N%!FKb&#be0^Xpc;JP!#256UA=8|S|%_WA}P(0*>_;$I;Em98RFwIg|fp|0|6=H3t*rL6A!Q~6RM0o5d`6|7u<(l)1-s#Zo^oCctLah6ccKggN4hr06Ho60-i8pQPk>+=VhP9*ogzc1WO9tw-u_WWJL&I)3I^HGGNT${`@&IM^=ka)E9`v+D>0U50#-Lj*;Vpi8Jx11xIjFb{Yss<rf`Gi-HeQ1DW3j@g>iVFpThrP^jFu4|LinMOOmPH5ZNo_|$a8$~uC8u%B77;5J-j6Dx2a3gp1{^%tX=bo&a8Y6##`7il(eq7E)(+lceg(st;0xGdW1G~<7C%kB13C<yuYe6qk^@$xtG#9l(<ESvOvv()NXW4Ig4Vcr}Qr5nLanl<#5u_J8CmCmXZ*T&h1HrNx~X&A6N-YnbVsazfG{gLJwY^=yq#*Ty=zDdgx7<*shi*&TLvNw)HL1OT=WRZH$;`tk4t^T{)S{mv@gssd5^Wk;xz_?qo$$HTK?09Z<cg%k;`?JEangoleo{$c)I*n^-)}M`Ddy=lboIS-Ec#g#+VVr&yFHtW-*k>QoRtYzIQO-f=guH8iJoG2dYKgQcYmyha;4L?$wsIBrE|X<cGxr;^@u%9X?{4lQcqIL}cYMmIKfeVp1Q5$W3Es-=`2tw|&SGgSIPEzggOmuOq#ir#*x;mkI4ZMoZHb5zCqk^PO^OH<6sL~}(nQ?6nKHfe`kwmGt0AfaamcfH{(GfmuG>3Sfp%6V4q<Ot{r&FLlHh!ze+r8<df#}yrZI7oynFhf$?2$|UYap_F1?xu^hX)lyIy7RqMn?Qx*8amGC7OlFC#UnN_d)hJqVq$U|7V9RTUY#fBhh&=C<T5UXr`_Q6RxQ@dQ7Ch#8-wP(Q#8lTNV127Mjbd_x(e3|0NAH6;sukm+<?m26mR7F%WgAwNb_W{ab+Hmn9J4d$e6h!^8wX$5Az#wzDyajlPw63-V_l?ezuNK*ZzGSe2fPwy5n|MX*BHh=X~m7Z$<1p2ofd#4@Mi@XjOb<$-E#Pn>N0gdKs=HP1RX!foPy2<PGbZ@3op0w6w&t0$bTqYOZ4*m+MqpjnM>ajf99>UrH79Uf?GqW8YRo2O}km)8bfUYEY3NnS61wl2?b8m)nrdBX#YdB338Wf{YsUG))~ZgSr<i<O{blP4B}4p1K^UOb5MGtB5|9<3!WWT<RO-G)zG&_^_6P;x#YW+DucrhorYvg*~Ot7Mnwv-oU9<h`3?sW0;(>bOPP#HTi&;^0jn;`eLVu!Hdip*}Ka_=AJDi1-mzy7Pk4>ZWANQ`_Wbo&JOE%9oO13!!5$)Zg!$q=G`@`wVBg&!1kBBWap6-g2{*{G!oqIK*ei~3p=rxL$g)H$89|#wpz8yIp1NDIfd-(r(<p093uGHz4S%{w(%H@Do~ESh0b1{zS4w_Oz)yMB(CDVzmle@Vl670^M|7a^FwKgE4fBFaajzm0JY2KjXrKH^-yS=M=_i&%$cbg!!at0bi#2gk+c_dDj!Df3wF3(rJ{{G)bfajc%!7V!B(iPZMcae9ZKbzCCYA=cgQ|AxNoV%C8>_Iqe)WXeg*S#7wRxc#^F9Dr3fM$T|i9e9wO7QkXaacOUfG5<n%zT@L=f?IDKN7!!kIj%bJeW6b3`CH@R1grOp>Tl`%U-s#NFh6=_CLt<xpF?OA9rURzd3d$0~7{qf$@(8-k;n8{8zl2*=3blVb8#7V_3g>`!PND~mWn62Dy^K>EZqMhkguh4Ycr7>;lD0((|O^45|aH)E(?AMdUF|q5hb*ideI?;2ebv{qvN@J_mrSY^O)w{I@N6nQYuI0|rL4O>HR1IZ7V#xS$TwIVo#jVoK^Q167$K2(PS`ploDq#F!D)o=?QA9d##VvZ-ZQC9D>dIC0<esV%k1yQ&rBneZV!WiAl}J5$rl(UeRWGr*W~jM{qDBRe%-ijisBVV`dDT7F(=M!pT36`iOeE7)wdA;oMgg=umuA#sb~>hZ*t|#Ki{<+Nad56p%W+{C{w;<YY0M}@(>~fLl91|yL}gP*D4XiHzxVTgZKi3Kb*=L}jv=;N+jn;v2d8>p7K9Kywb{zZ1mp1(I@9;flFzFy&#?~{t4IyxzKPi{^&@n+Q|YD1_BP<kabs{H*-N#_!&IwzM<~C&E>B7IEIHHG^R&iV{J-ao@<VYQK-W>ZIT1wRESw7($Z=ljDue7fI0DPjx>X@Hwrz$7xvw9`im!n**BVcMt4=2f+w~7+@QXQ0Hu;K!hBaHVs|Tp5zuu31n{=;EnSd^<3D&oY6YsKWW4t|M4wU{q44-d%Y)5ZR1*Sm6;azXxG&rCeOKnB+UMsI{+5KhNEv3o;c(~z2w0P7Sg`9|LL0Q|iJ*CsfJZHVYE5If%{h27~$!>C<DgjBQROf=#{XbG~JcqUSn%thB{S2ASk51n_g)QW-Bxpsk`DrISVKSz1s^UXdmt^9jb*WVPMuByP+GK7czuQp!I?c&w>YyEa*En}RAGLOcW&^_*K3#0_QMm(*ToQ!25f^lzRm4lmhJmDk0Vq(*6zp^ys2nivHy%vnt2w{gv|inf*Usv%BN~75KnV@*h4x5?)9c*NUg$E~)&FGbX<^tPHWy(x`9_mfh|QlrO`hqMYrC}~C^&dn<!z8w+^%60oz8}etNpCGD2WUGA*G?TREz>*C)7Kbg>$MRzYh22)`<s2_X~IK<Iv~2(Kfenz1$u1i+#|}x3%f7nYGuGn4+)6k=58v2gFh6Wz?x3F+FE!VBYsrDH&~CJujXer1?4b&y(gxSV{=BT%oi6o+Y~rfDm(izg=#|BB6NT?6i0NDpe;Rv&z0o2NEq`ox4|{@Nx26ulo>tpWMem+h{+~AAi;NsyEnf7YTn;R_US=oB-Fby4JP5GXc%)@67YPdRT4FFfU$`2eNO_j?$ZhH;2?u;{=>dm0?7(y@+^DUdqxTD-e-<+^wfzl`aiK%}BZlGjvbx<7M+0nQ@SN@>c1MHd|_4Pv7;yaO9}x(IpCew9?tvj_K>vwP<JFzAs;OoP)+n*TI(c67#|Rr~|2gF2;bpTt1B4Z1QJa;uOOA2n}47Iq{M1t(By&>hslre<kWnh!@VIH<khy9k}CtQVj)n?7>ScoZx6ON)aI8G4|)HRCe7oRQjEO=tky<ZgW2vU)V)&nzF=QI92WQmy;ytIDY+bn(dWH(zZplJYBkJn221X(XlV5PP2Xq5~WqN&2hJw>^R=Lp<{(hUQ}fT-l1}4U$@k_=S1d&t)1-a?h3xSLAMdKCSAw!^6FtR@mJW(k6SJYWA|!V{kjAKCDQKXxj*uA&#yDE)FXpAew1g`;#TEJ83-ouWm=V14)YtK`t~Q4^S$3{`**w1_`~8MD?qd5-*gxCgSxV+MH$!Kp+>A{_GR0vPLJQ)LADoMMuo~*#NR9EeHq@qf^qE5-KuQo_3O&(2boQov(BhAZ#^A2BHhANPF*T`%|NA1kTwo+fn(>vtWj<!e;HMKFPx4hRLkBeRrgmNHJSp;@4|{x?MF2nXiw4aex-s`Rk_Eu0n)XaQ$BCh=R+OC-H|Z&Li+kvuVn?Qy-K74*I_1~6o+OD$wxn3th9Om&)=0N!?&BzllJ<L^fGD8!(#WD-W+8w{#73DbV_xYc76dgy%v${G-ibK(DT}|CfiJ4Zc$<0l`ly$A9t?(?CGX-3qG35h7S!7knvV=cs9r@qcqAp=SGwOhgPj>w9o4gfqBObTOUzte19V)<TM)+^=%I_kDHlU8DM)=@FxC=dv;DlKkz$NLc_<|h|=Eoq4%JY0sbleEO$7b1HwXv2gz4*T>N|>@9BAOhV9Gi$7Nr&G6x`R`5KFq1T+Nt%M5m(-3z~r{AK429b(JizU3Y6mMq=s@uUYFsp1|)+hRl<(S*#h&WE-Qfvdbg=!i3;O?FU_xG06q8$U&mA&_^N5gUcv4YiHc07W{Cuou$brRso`N+y#5AfSc$D+d0mB%Z=|aXxClY+}`{`Ne*a*0#)Jo3Co~xy3MNedxY-7(MV-yBT;GssL$l0YlVwwLP>VXWgVpqxX8c&x9^_)7p8{RVtyLPFu6SA<s$6ztFb1-Rp2RTsrG_!sB`yaX{QiBo&!*&jT!{?=HOj0JBcrd8{G!b9kN|NMD@pSZV$|>^#5Rw|H*;#TcUSaWSnf9WzDfct(!cV`UnS_T|}`$I5<V{^`2x<4vd9X)Jh%!RY&~Rn)k)tk~>nC)I_M*x(ZXMQom%iv8MEZk6Zy(3+sDGH)Qp;wY@v5j35@M##LjexUWcRJYq#y~lLG0o(swrHe^?Onep3W`o^gmd}CfZ1wuGB3>Cm5?#}mN^5B6&0tv@t!Gn8ct+*Mo85_$zp{ZxBCPm!4!N*>l!z~_s|MzimJ=)GhWo<?YjPS4lA~x3)W(ZhlB<IS!MA2xsf|eMo!!(>{R3|gZ=<u=B?6UgmQlI7g^|{N6+$EW-R;^B3*@A}%jm_4i!+heEpJMnmtioqdNkrfO*Q%olL{Bg6Jf)T9MD6Z29vZgq#KL<o!4WRpTV193TbnlUwgcJRM=<j_1G;ZhWl7tcaveqc#g{3a<(1qCx500BSutIr#ku<EVzLbk8D0n_BNz(%ueQ-ltu;vO}n=*mx+@?OL0H#(92l;$-~<|T=L@-Yppj6v!Dkw-1@p_b4G{ybg^nn$3Ka7o|&fZ)?0l|tR^zBcFl)EyqF%Iwkg=FUC_qdTB8oCLph87(IToy#;NPALIc079+h7`8LH|pOG@T4W!6?Tki^+3Jy-Erw~gfUVp+phTKwnq_WRI2GffZtoLH4BsqjxDH~q@s?QT23030b>=NQb&6}pT_aVsr8PwNT`qisEml^4J0OzmYZ?Dkj_j9ug=-Zzzn)W*8pYF@5>u`+p_zTY`2*adRT-D$vqzGOo^JnM?O-`MYVV(WtFpN=2ix~uf^lV-+HKGROo_SB>4kGa_cf6%wu;<}M~fHe>sc~5M#i^Z<{9;L=ENd7=J)7tKqqEKcF<`tRqa!S6dm3%*88TOcJQ{y(LHT=i07{L$S^878LzfmVAxbw?_`xkv@$FU~PzTbWv*62@bjInHypNt07(GhUBamQ79N(eU5pZAT_`;$g+hP}idR>+T?@FA1a{#wUgf!$Ii>b{HF+JfRwh|v7jJAOw#hqGg2M~C05^4XB^n_7VJJCPXn?t%OM0+#Y&H^n!H6Vyo;&P3UdwWh&wa8yxh;yU~E@^R(7mwaH1fnN?(mw*5P_0hay7+?0iHPH>ZT~Qw6LI?7+LW<qNOW(#M?<($EvMEUN>fsqSVN~evAG_Ztt?bIaADXX#`umB_7{C8DK-7JjwO#K@H=<HDn&mz$UA2*z7NE?OPSoR3V&8j$X;Yq#*r`D^hKoBv&o@fW?xorQYs`^r|JBR~)UJUW>AhdpwU*V&(FiRSvUZLRcdFNzoM#=!vyQcjE?;2U92GoZ5wMYcvA=ZfRK~ZJ6gEm&HYep{Gnq|Ce~VS4hITZq)BM0R<57QKfQOa){VUwkmf8bUQAf?yttVlf!D+fu(PevH#Whv-I^CK1U{7>i(`!I(soCO=wOhaa%f&01I*MP1u`?UzT0F!CtcX#s-F1=cU9#0w=+>rK^zl<i*h})V{s2<Cv#*~gWE@VPGtTIe@6oOP7=*27wc6M%nw^k%gf;-LE1M_~-XC>)WhGWIzT5uuaRQI`2l(f=?@4^Q(RjOTXhTWAC-y!LhEF<KO*LWM$Fq2;*D;&@!Y^!WdiQCy(tc=td)`si#~zWddXb?f+&TEER$#?tIQh~wvLK{QC(<SeUfs6SRVAgon_U}C$-m?CXWQCYlaK@I1&2;=g26Q}=^cdkV|j(6*#>={?t&q7x-Hcf^A6f#t1aSqnNSCLBfa)^)i^g=o#&ufTDEvf@6Y}FSS3!?$#t#a-0-oBXZ?n#*B96D5KbTgfX`TSCzW>8lox;ArKCAGxm=;swn~HY&)J~Wbx{PvN*huxXr%QJN!a?4RXuV8##?Qj>5I~mG##xg$_pwMxW7x5bY25q!L+#@1$CDOZ;zbn3GaDx{JfBSWvfr?kO6Uo`Fs7Q6kd5)KXMbV-4DYYydpYB)xrA}Ir-3-xE1p=I*5=7a`k!l1oW3pqur<WhWKGyqZ-L($OQ$n;VjGVhkf-m*+<oDJ50{|3JP2o%|}5g=gp-DGrhI;3;>?e)UyIT*A)u}<Dy>7&{gV;>XBT>?i9MBj3+_>eZ`p`G7MVRD9R`K+RYr4<KDQW{^rg%S4-$oo1<3m2E8$#(7LR4oF62FmuEG9G0M^t6D*zO9I&8C`V-f=&qT?~TEo>QOb?Aa?jMkQyW%?9t<qo$E_#TUeN{olSP1(&e#{%3nzfvEi#yvbJ%yn8ZX3LebQkY_;wY(BaIrC{6dw?3z16`jFW}%<y-mWOb!-Cp9#NORF{oHKz0)r=6HKR*Xauyr8{jf-QP<tJMr_*Q+U`EUX7N)iZO(IpuGgO1qV@1p5EHuA$x*dp=4Mx~#@23EF`9pvfn=8SWB90@F{)L)SO{2ov(g;FkiBZ3v%dPrie=>^Ot-yBH0g8Q?()5_&xuhp?Zxn~hW@+%i+5Xf2_%A=uXgFN3!xQXti4*lHB6B6o7u+iZF%%Su|ED$=gn~PqjTn5S(aPxR~Nn?9hA_Oa5v>%Kn13&8{QwfoZ?iqy-u&$69kXGT>rHDWuB^>At9;QGq&+h*(?T{w77aQ4#i`mKaBz6iO#2weZ005=o~6}RkQPtHWPmU7w<c6mhaD-`WcbN?8B^w(`{|^!M;}m+sD0gUDqG&s%FCEm*iykko=Bp#sd23>Cxw*jpvUoN8$K8YHYA*KI0RFW0*ehMf9Um99PxbbO;55P^o$SiJzVOZy=sV3N1CI;=bEoJ5R(ycCZ*MyXChbGbH!gK89atIeRbV^2uxxm>X{K?xYF0eXpUU7&(`K=hW-mJ{z-jl}0tKgJ!9Sj23`AmtRzIi*B~6IfE~p2`dy>kjCXqPx$dMVpYqKMGd7p{f#nlonyDwC`s9T4r=*s$ZZCAp^}y52~N7A9Q1_N<vM##Z@uS7s47L3v-ytCiY32udbSfLyAhjVcackT_A$7g*TY)nMcTR2*dL&)_L$r84XPBr-VuHedSC561<8ooHgi5j3u5CJNTM~oUecPJ;>cC(%rgEuk5?_Y@}BP(xABolF1-FgDk3&Mw54+MN0#fGsLsHXzkPq${a-0N)_ai+xS8qT4ZYWxtt}b?8@Jwz8lNFP{d7Z*fu&;cJ5{<>5}o<B){St9{kyKyqzA@I*3=n)z`DT}BPJhyMS|DWr}r|+(F-=gGEpLz9=5?r_qLt9-o(ra@BLM?scoWd4h1K(2ow9BV1MZ{*+9;1b|;SwZFV)YQRrOjHpN)vp}$Jji*GP@X=!rWL_9aIY5v@;+%Dl(eg9T1Xu4V#%(OVfqxvXM6Fn1q<4K1WuS~5+&>}>ugSqlyRi=H)mFmL>FPrCDo}&xlt7FjY-@RaQJM%m?83U2payv!w)S&gKb!lbr7dXyUsy48r$tJ_;)HOHsfbVaI-uzF6C9+eJRy2pQW`UhUzSDEzsrz_YUu?+rJ}68>8*E{1`%)whd-(+)6kswEcXwXEcdNQ}rb60tJ~bkY+g|NM_~3TE=sQral}AE69(&i09j<>S%iZDcaFM=Y9H@u8%M=FJZa?a5WbJW}9up9@V~jz3yEk}~({s5P1Jstx`?JRu8l`?4pv1+sT5(+RQC)e6r|#(>{@ayBo*dyWP6`inv<G;9N9mj1y?<XDVQY3zhT3L1VAsNNNioEw3JBNB&+qOE?1pM<!u6*7af};eYtxwrg0#(DGkZ46uTotenX@SH$>Yw*Rgnf=VJ6#>rEK<6^<;IQI2)CJ!#n^E<JPSs9c|1|8imk|-H?7<(s6mE$m3LI4@Qgx><#UQ?!pCM^Xe#jHM(PUdAhk!SWbV~DO5EvdkFPXT^ybM!f&@QCB~_3u(@WVSEpv~qnU2oRK2?PBsyh`%I$|gu=;=yXRpPJ>C}$)gnDGyQ+DQ`?%8+EYwgBto`9j*QJ;U1;8E9^-8LuY{+2pqobSu4Sqs~C&g(U-A~d@~viH|gsrKWhUbOSEzp3AQJ!dRkY7j!WZ?TxKr){!|r*b4O_-rITqPlj|nX&tt2u_`V$MIMNdzUNxX-fhu2}mdKM`V%6h6M5qT45FDcYvYO$2*0){>?jg!F*#+rxb~7F{ks}u4;?)pxf?oh>lLT(J@9~3;lMCB5Y<~Cv6W1<&)P_2JbC{Ud9GOTWjr%d1vW9d?wg*kxv0V`@*jpw^tmWn5Sg4#j1V*0EGx(a3Wt`XjL7YEblQ{@g|mUu|EjXms*Nw4Y!?o%-rhDg8sf04V|ON)e>^wTujQE1AdT2!6x0mv)D&#Kt);m7R^b4+IwUVE2B-H=Iqh9aRcw6FlaubH5#6l^0BW*im+TJS*CiJ4+5=Y)uN}gX))9F?oBTI2IifqbkA`0b|(7o@S;K511*R}W6PSVaQKODyxu`hLhJVl`-^EhZ0G!XzeDgBI^L<vqFiEjHobh8tSp}E!)M9J*2||9u@+`LtWpOXcy4nfH_&Om?0wHo{&aAwG8Ft8;+GuKLPw+t=Vkh%*S@-{4%A1)!*2WhJkmD;y7(ObK?FowkGPD8%*X<78A}+)(TqX|%jacNIo5tVc;ERo?6Ax{d54@lcQ6l3AMoqbmF}U_4yfPw9!fSEOwUx5P3K790qu^eJmy2%-u@&rr(UbJ?b1Ci(<{YycdPM&+qM4Y)Otuh4jLQ-<^6D#v|KGdu2rJ8zsOTo9gW-;qRmA~Nfz~(X_Je#dL)sr(N}`xs%`pzehV2psje>mxLZiZb_q;P{0#?4@Y}oF1KgL2;rYU38<M=x{<m%3@vpnT(6<JzZgvc>uxFme^lx+AfV1GUDmaiSHkSb$u(fyN=X>UR4?*26NLg1l`DSMD7%XW!-a~r(?7VT>tv9O{+=0qNxD*G?LnG^{k@VOicdv%F8TDSsN&VHKN=v9N76YY_%OULHY<}T3LlVd|D&<=M#C9I7>^0i|9P&}T)2LCsgRQljW*2KYRsss04?uDV5CD<G;@>ysyjpEcn|k~>^(TkPXUU$THW{j3li^On<u4=d{O7c#zse7oiR5ryk41=CbOpewJ)tIjF^5Vh_#vs%M#H%_-3P_giG0VQr*;j4{B=B?CDbNC?*Vouing=}5PKE#VDeGm)xsZGm-gvTV%3QO{sH)3N`Bp7P~V(CRR1tk^EXe|-@j|IQ4SMsH@QL86fw#}=Tsj^+(oYV6ZLyjbmmcMw3~CeS4r~r2EKlm!`*azUbGkzY8<-jcv;M;2R-1}Lz8WvY~k?wPPsoK*--H~Kz3&o8@0H}`~j{l`8gWbgs1_6CAzk0b~~R{T1;9pyBa85ycm1P{el{7w78gchqGsGJ{jRY9Gc*_eogg1E;$`F<`e#UcaYSAKVZ;edh<E>O)(~Y-A1E*fo-qw<>ji~q`r$+9l_e|xu6NL4o4hdN|7%O7rDrB>y-Led48{X4|@?qTVF3N`P$`?Ki%M5>PzBGwnt$5m2{w<26LYWC|6_9J;TFs1XqXZtRYY5+X#|vv!y4(efm<%>~-h!<Cz(AqgAIic{LvQkEvj*o4P~qJG38r95>u^HLr>oS%qyvi!H^D5UcUdUAVmPX)7{@(rx~m8hp(Rs?1Xh8+3Eh!QVMf9Jls1ox9W7Ae=*^YvcYWc^#uA-0hPFgBePw=p#Jgl~2{o5gnYS+I(GX#r5B+q2&G~x*z0zG>iEm0H!MZ#mMMyjGy%_^1Kzd`SBzf<?_REHz7=Du(7gM&Z9~Hh58`{AG=o5mMEtA@2-D0ol=8?(m~#|y2Ba?vb-04rxx9?#y$;b*#ragwL+d@tvlo=ANpNC7;3;l?dg<(4EJhv*V&_K#x}Cvcu{r6#4m<AT;Bs_(dfZtJwondO$J(Whr(JxDWA7i^T#!ke6jxHUS{!lt}?Ikt2W9k>ATGgBxdalnWpya{vhTvu$o!h^-@>$y1w&8v&>|g4WiHe(5kd!^TfvR)XMWlD{R&q#E}t+E?LtX(!sZ~P`Y=`YrB1^%@LHb&9~kQ&DR%LZHQ)bQJj>I*Ey_tDOp0!0J8r=I5<re4G9)^)kYH0z~AEf3+sICDbuvT1R&7evG_7^v}104m*H?;7vsDfvrTT<CC5$c@OQ*4i04=gXmx5R5Bfw8S4ThY;GVJ529nPCdvwF6Q)2Gsk!oP@lZ3D69PUk9pXKKIRx+i*pQf!^-0pT|quFNxT6sm0jXY(uy@lQJj+w3cLhsH?DI?u_m04TIfd^6jY<+uDa}2%#<7fw&S7)$vZX}=m_F?-7AaeKl^Sg8}Lm}9C!m@^wZcV$lp0(&aD_V7@F<V{^_j%(zKQubaNoQ@)PZR85#C!53^yIROX77aBxynU4WXG*adaK{Nx5-<dYSTYK+lTf`8`X(>bZQB6`_*Akol>g|p4nj^rqIzueEY)ovhA!FSJr=RuiMsZjoQHaB(wTn90|iRY4=a;`0AuL@uZop<EH;PyrEJSyIbHPTL*tbd<utiN*4%;zGT+{WB0z=XaO+>arCKA6B{ZF=U2Fg0kDEp#J&_-3^*$y**oEN*&2`iKh8UfWUYhj*q9$y<`uVf+ciGJ=bzVUbI|`iG22^kQcnxPaj1t=IKqcucD~;`ASHE}UhhwpYd>~vED=N7xS_G*>d$Vs9L!ht^ERD4rtdbpf)CO)F0ui>-RbL6s=o(2;N1(|YXNHe^io^yxZYlQ+vAv2?Kr*Z&KSPKpDy;ASA4T_0rA$gh^LlG<NMjQv#4JB<vQN?xp3vU4Iu!S(Zye0KHub8+ni};*X1R5;!f_hS~46ZFIniyzXxc!w+|0;fP}1fEpu)s=pJpEt4h69-aG~Dp6@yQ0e%ssPGmtwVnXf9m+kBiG`BX`1~6XR|BfTcaz%at@bY={ou89y{XA~o&O6i}wd8^ef-QP^tHv_w`c_U>*dKA3ti3rsPZJ-k&sfx^u1t9tuGEq{+*E6nZZ?8=Oz<GKUYoCvCasxWJ7{S*SOeDN)Cis%32myq=4V6A`7g%9-tH9NDsL7bCeIPt<7dxEa8!DHG`<X-^z+L5Stq0@slKy5Jl$o$zOqIikHw4Et=0k3$3(*(kJ12(!h;DnH&pUy$@H%S%5P(4H=D)Jv31ybIOrwNPZM}dX3DGTZB-9%J?g+%iRXwpau+c*u4C^ea)Wbh3+xpwQg6W7(oSLIcRaYCk5J)tstsz;V>&saCNyZfQ)8iV^*!5A7S+)rzO}}6V)~a>(^vPL?3*N8E^CE!d0(C<;(V?M(l7;8ZA5=<!qL%N`uvVEuNy$U;P4E<KY||(q>XmZ<?`7N`0Kt~9-3&orcf`UJC-zemk(BM6~4__vQb04jH_2qW3LB6XB#C4XFWI`UGkh(qvN+x5Lgh1N6AH4dfH^#Jpnv+KFo2b9+JmZx}KI=EpaUOpzR;;)J}I_VBaFYgJQaR9?6dreb&*P9Ov%oZM$NJZ=7PXIt;SYrfNkA`DGH-1#-hrHH%x*!=*}|x+7e{$01z%%>)$W7L8z8HL0vC<_Q1lLk{k}#pGa*P2PtuSg_b9#I>+rOW!v~%bpJ&R3mIu+R+0i>ZcpGxJtvmUj;5pb$1=GKeZ7=`f}WIS>$Y(@JFrZ=yVeBHFw7LANT%{{$tK~bf46r4DIbZ6Xqr1CHV~=%uQamL!pH5Gq&@%gEP|?*kW|itOk)l=Yh?~s)oC>pmB+W`Mlwb?pJa}w)4=)na`}hZJM6!O^AF`$v<Z4ec_RbZoDi#0v5-sHF$>S>*a}czTGWhRDr0K(qW~%kM_yM%hvnb=-F)o-!mp%hOYB_f5z3Od;D;{GYnSb#RE9bwZt;)!D{^1vFQ6NvYRkFUtOq&KKBiyLu;shoom0!@K4Y$hXZjY574X{?&fIzvC?GRJfe%&J*y1iAD=abW?dL|D&a2Jo~Jaj@V8yMtXzuU{V&ikuegYRDLMyM5!Y1<PK@QL->4{>KPN}#3F$KUt*hECuqh=VWG!cvt|2AbnoPG|SYlm8x_3;aJ=x8_dP^Ej4Qo!a*obxW*L8PyLJ!#lQGWLGyA+`mt+a0Cj_c$1@O@q$?PEKY8gr>v$wyQQoP=D`edM<`pG0;e3)^mQ2tmCG{8H=7J!sv!0m*B$JS_+Gbg`_b!ByhVXm3F5K%HoHgl=#>bW{4XN9qNtFG>JG0PRicDa%d**16r|By-_jL7MeyZ+D$^gm2{&H;xSxJ4}@oxy{b_zO1g2>hbVd%Y0X#>?b!1myL{xm(nU{_uU+9)a<;7!S7;yHNoNjW*@-Y+uG_{qtV{!uE9M|qU$z|QzLdp56&gun@NE+VtjRZR_(c)U4n3XCr%*yZk-jN$T<0Ctdqnf9$r{u2JCUs6Z7<Lk|rhe{pLniWE`E2ZabaewaO+KV#1&^$LR%rq*hOb8eZ~S??_Y|U00n^M_5=!*z0}2n~-&Eft`D@>U8-=^PbNpV#V21CeN()*k&G6&?6`P);smK*s%=(9$|g!c4(oe!c;6$@*t~UbflG%cCO{Ucdh-6&d~}FS@ijhJ}|`wi}dZ|HRoYu^8G3G9{oy8q?JvUi2!mHTRX)s)*;>~Y506hK7_JARnF*V;&yO(1W?h=dC6sXm^I}eA@$tK*>x=94v)JV9yZ*S{(ZEE-ZcxSbJO1S)*Fs>1msXN-`9g<kNmH?LY?ZeK0u#|zFyy!+jhGS4s#^CE6wZ000u{Dvu<$x_TUmY@nDxVGs0;W@9G#BO}9i7M%-wOcYM<DfDfw*V#VTw+A$WCRZ}iCLh!V%rZ&Et%eCQ=#lnZb+v(xE?0pUyc`XlFRh#>hwSD~?-=W~5VPp6@ecN^Fu{2vV!7SgG^AFiRVh5dOGre}Qk}KU|8&Bpml6r$@3wci8yBkbIDQ45R9KSs>$>4Vv{XAKQkN{(k+#sKVlOi09MCJm$kzLj3-z;n6HnFBt{my~W5p)iZWuD24-hSO38QV|goQQ-_F11dE4p;s3JbeBMTe>LfS0gIh#QA6(oNb=hR#mIc!1>Z@o)e-?*R=?oREtQnah7utwH9O~^ZuilZ+Q!E_v%S!GZ2u=S>i(UaR&{e)X6V{mMa$EfcoOXjS!K-YGRPvLaMhZqKA9?CH#I$AcbY`dGGLFf|RooXbx0=lYQoT$?|)8l^)V`t47RPaUC@}#Xc?3S<%5(6S0L=@J*S@srPI2_hQBWN-^Pm)}#AD@AaPDY1!z$+2k>`A=>Kc>Z#G^kQs;fEC8!k{l;44XIR{x3meU;ndfcgDgzjX4_$L!rHyk?Mcdl@Ld~NuQ$trdsSSV@A!FcmHvo=7wvXvlXx)T&smyv-_5qF~M}M{smpH1-t!g-7t^{1@G?>6VCq2cPQiy9W?B19DtVNKUxtAh~3V~w9_iuBFI2NmKz1^7p9!HuX>K{*&-7!|4ha;YpOt)`#cb{ujG5K_V@d;J6SSGZAmP^Tq^3A>_I_`H{EAkbvR5USZ{&o>>TwOK0=fiofJL1F2S`+=|HH*dY6dcvlx&I;e)c3mKchPF`>V!R5Z+j{E7q~n|KT__(Fh;p+?QgpAhmOA-zq_`Eq1l&Mfc<sKsH?Oqw=qMz9It6@fwJ4l?%hxOhSGl&Lk{6P_0$W<St67#M)w&%k!aFK#-A>mMs)W7yST9C$$Txm)o$6iW>fXD)%pw`v3f;PDy6%zAaZ}KqtSj!n#&xv*II8U!y5s5?7qaGdYGhAJGI`e^`nLAo_+KS|1T2UP5wRf;J*8MQ{cyDD$|X#y^UxEyW+Qt;0`Y{uQ?3_<M0{q#w`M09btazX16VJHI(hy(I=;JlMPPi>lE5@iz#)W&uj{<tp_v)<ldq-!t@Q$oRcVP0OQBfyTjGlbyoZITjcz=#+CM772fVTTrb;<^=+5~9zE#$XMK*@q?FJSq#lOw_=&<Q6qBuHE8gp0P^dPQ*;8SyPT}Tz{xaZ;)nn@P@!GS}V~YFKlZN$&c(~N|cbp>#1zqn0br9=d&Ew1Jwjrw>mr=($c6p38=X34aKR#~oUAr>h)kSaJU=`MN9p3+(2vnTUQu4#?Iyeox^@RxE+WmzuX@m1^&!>74@|~;yVi9EqGKk~&JosBT<s&Iyiy7ps8s}m?ICHa<)i1ASeD{pX@F-KhRQt?V{&vNwJF{VoM1$*0)0tPB@S5L)YaB#frbdwH!L(f8h0WxeNh>IY=IW&VY5Fy1HI>kwJ4Lb*m#q8c6SfOkW1)tt5Oe9fmB}L6z*OcN2VsGR_B7{bQI+zJ=&iL}Q=^jnP>V*<&v)lerLyV~1?Q<pr;mzjgd4?5%$cmEVbhu+O&TNASuoT=JWs_3d_x2b-j9>}sCGe@v+Ka*;w(H}<OX;yno}cLqwhei?=5YQ)30m^OpM959lHVE(g-==TFkK>W2keNdkogiTih&MZwHGva_x_>Z3CXUN4*)G4=Xp>3U|LbpNH?Oq9(19u2OAkGW+NXc5A9qnSIt8o{(`RkB8|j?UveJN5c0(uLy6oQH19kK;<63S#0u4H~!wJjTTLJBXQ`B?TM@%gqzBkqF9nae|04Z9BC$Cu-%OC(GxYNFICIE^>4T+<ay$_uYASVpM$(jJ0QFlGiUnj!EPzhIy&l~cJg)vjC`Z%eR-N6TNA5wpXbcpjW3s9ph`;Hg7a_Kw8k<xRD-WI+m`}AFyL{VL$IJzCl~70R)6ldtM1%7p19lN)A~wqoxr;ITNO-yx_^=D>%C0Z6Q5-6!3{2Mn-dG<zO9LcdO?TF!XLC)>@q{QcT1hy%OeFNiZj6Ww8!3dl~IGe2BdxdW`<G!!9XowXss7CqX}Y(7PzjzcjNP1V_^RyBfax>pTABHYXKT?;x9&Zw>Igod6Q1bPKw?ibC<yw<qhCxa<&BMvmV@>>q=1SegD?IuO;zuUYV^P<NhU!{xKf<_gV`ll~KGhv(4Y?$<YSrJ)*%j_zL5s-%JrR$N1IPP58FA!uGEM=fQsOE3@mbV26H_vF3+K{qJ%|^}mw%N;SgD&ztn!ZZ8;Yps%v+ZDnIP{+tPqMtE+l<3$$>`jdX+KrY5ZpJmodJef*P_4Ds}bG?6TVz0BW2x;>N@b9`Ws7uAMrUSY?aN2RrjUnLMzc&MsSv)W4tgP-L7BuEGKO)N&na4%@sF}h22%Yl7nP)!>!er^2{@w)W#2V;Cc+Cda3}=Dx>NLoq>uX!FO{~@QxRI9&03TpsxY?$%b{B3PO_GgsrVQ5O-SC%QDuERjfXcs}-sl@I7vkS$X@<6Gb*`~l+q0?rVw3PWg<jVYol%pb6+gfVYGJm%vsN`Ekf+W7;zDv*vuZDM>vi2TwY^1+Wb%?c1R%NXsvjL1z}DS{-Q#9@appQn5urJHE-WZZnf@Dvw3FEGBxAH$Y3UohJNUF16n%D=dk_ji8P*tkXe7vzmG*f3>M+B2GO4ufbOlr5AYI&XZ>Z&KTV2Guf4+O$<;J;cL>=hVw?zT;&QW)_b=>Qep{?DPb&<D1DDiDOAw;T{F+qJb{N#serNq84r!8qNN1Hd<r^xuyhFW)x$*Kf)#yhBf>-B0%n-`5ogxvkyC1%t*f=+1rY}F@`T3@`vUTg?pestb0d`nKzazfXzE?8&teJw1j-R*bF;?H&si^u%?GVkVq)4=ovplz!07H{=GNun6ZQs_Qf+4OR2DdMZ2nxTf$?f^mA%sw`_y)l@r9$6ZP_R-896-Sd+dM{gAWAUei;;91RY;i-VcMT~!Pjsd9S@R1+=m>j`(>t!txQh5=ZHINs4Y*Vm^}n9$`75mQd>r~x<#hJ$^4kzz`D6o}ss2WU=reWr5fw2Ng<W~jXif6{xm1~Yz<u?8JY1_?^|wk-x`@y^u3*j0VZFuScdkGD@f<rhjf6o~bB3LcTq=w8nj3)d;~l)!y1ZU@`v^#Did3KR+M>V`oZyV&N{R8AH<Ih%q~=!rSgvvS8utP%0HS327Xb~}{$m2<tZW|_yKBtURfDZ?kBa}iH{KA+wO`sEz76a{^Vf!<#b32!w=I4DDmQeXy?_*0cbMCx7&DY_ZTaoyggRu7<?ivXR)^R-MS)DWVP$g36-i^povYUl>BDsu><tJ5@zZJye%1XZ2--zFw})#p0EbmqIi>LJVf0oj&S~EHpIjf<gb46OohUnRobc8rUp;*Upqvj^@D%JGDnnPHb8kdhqvrE<O|PZk-k-L6I%?$wFbnt+QqT+bY<PMM-xl#%4YhdRsn;xbSJ!%z+n+Ce>clAC;m7AN04_oGC5S4c8XmH$l})A`?%O>s<7STAWehCtOkP%Hyxp2^$>SXY68QcsZLK!mYMx$|cdi7^Nxl`$B|{JPbJD(lZ%`CG^_C#(S8M5pMZN?<r}@b5*&i|Jm+KZH2i?hj>9oM=dPEK$VwGCO4+U3kyHIp_h&7Bp6iVlLVGZ*vq`H^rJHe>mA)8awOzLfq+OfR5hnH>iQ7av-=p&NEL>tmad3a-o6%Xp3007}rk{!b#{w2DDPUEs9MqylkzMF!3<#hq=!@njH3tZ^Jzh|nGl`YbqANNP1IqjRDIis{{)y$ih6CA4Rk0>nX7f9;gvMaa@2I};59#&cuMcg~~c*#Hxeo=|pq5Jv0d&oW*@6G9Te!B(7rgbgk+0c}(TJeG_jd;6kPM11-ZF>)^Czz*%VzmAMlFn+-55N!{cP{K3eNAW_d0B@+j%!-aTP@`}FWty0={xl^C*61RCEp7#b-A>x;thW0?-^|QX`{DBKg)FI>&&Y?{!|*Wa5VQn)?dr)rM!gV?^7$=y~DM-^G1YoA<v~5^W2k^18qN$!8B;REm3}_PPaUdFJc5^&<}DO!$-70ZrkU6F-5FfOSOq^&X)2mwe}33!Cfz<ucK$j4$gjPYGb@|Ax01$+m!LF!t=k+qVmo`yWF2`p3NASN5Y^x!vg}7+um^x56KH_?8ocF<~cy(WVa{ub-T|?HR*6~b9`=Yj~90z0-`vS<AJlbV^p;e-x*8Z>$$)FVjZ{BY*LIwqR^wMQG=#do*epZ&DBu+{aP|Bt8~<A=OT1p5BWVFl^^-hHJ;q^uI(dJnT{XyWK&k<d{HomYZJnz<3fE6M7UeI7tVI2CAa|80EC*l+`3U`%s)AbX)i~N49?`*z59F`@M*-Ss+u$NOc~Wcj7(lGu@`#l!}O7z;uc1&W<&9;V_Z9L${)`58uB{yWYze-D@xtEEsVh-#o4i2dm{s;-`;@P3xeu{BMQCet?W7Zgb)Ph(9=FHLs6@(>4gWm6ZAMtrCBQMnM}wq#DDW59V%uG$*wEV4i`oB@io&`hdZ?9cZ+fEl_aX8$?@gU)!Whwo7c1z1O96`h0nZwRUf0zm(u3$LKGKF&_;@CSD)a=${3lvutHdvk)+Vfz&m)%sHaD8=0q@bOoYu~a{I{4X)zG+R)?&=dlhcAC%o}}j%%`7d2CTU0z8opw&4%Hwvzdx`?NJ&&o}bCy>F~Kw`Z3g{5ngU-!2k;dFauAF;yGk{QWN7^bbt#A$mP>iQU`Z1@R}}%%bDvOYIR~%;@YU2EK);mA(mAi_MVZ7c!tPqMxQbTC?5h@^WKKKQGkg74Yiv@8Pw#r@_B}N{;5Utp6li7TVvs#|JNl%q7A1Z)){nqm=}>Z%jQQr*WCJ6H2X@w*yl()xWdfzjVvq<UBhaxXa?9{1x`w#pGcKX0SulS>#D@Aq(J2p>2A_wow1@)LM@h)ogJ#)hJs)<>MRudyW2q?pKZOL8($C32=Ye#=m{X9hq<gZHfFrtUG7DKhDB%W+;q7bWd+0s>oRtJE|8yQQsG->pwbD0a8zDwtWw4NoM}Q$&AmGqcyj%=;e<WB8ZK+^KM?PSkl$(9X|@z>y5M_2p^8O{Z%^xJ3(x*C%Dy_<acyBTr@SH@p&&qdRyi7n5W(2-ECI(0VSNvm+Qgnss2Y|@(WLK+5q$5V4li#`90{j-Izi+SbbElBmVcEl&nEF<p`u|^ePpd9}bgmXu(E{&xdevcCi`zZr|JUcw~({`LrZpK-!s=zbjH^Z(`onr@Q6!&$PGYS(?FtP+}b|DgJcPUSD~uv*9rPfD{CRcvUwF?H{trg6)fpJKc}Z0KswF4Y1dIo2tEcvmF%g{9SunudP2t&gaz__s@jeB5~vKdz_SkV|LtUv}nD*GW6i^ZxROTx}uY~SZucEYF|+%iy=|Ho+mHDKO~<q&eXa)<Ba{l#G?T?TFhMZ#gpT~p*E>hN0n9Ji-O^K2NQZHY&)rbe41TIJ|Se}#kxh?Y2~x>>||fIja7eO$v<haRU2>mD*-bN=#Tb5aO$J8SPj70yDiJhNn^L5X6~u>gnun?EmaFpD*K&27OzzHO23Td=P~qT+j17WGdHDMM}S0@G#Oa!{ZvKfud^|-o*V?d4@)*uJbt!oHOG}lZ=n75_2JPr>mAx+=JMDRA>-W{j<c*4s^zztM_CdoI{D2)<s$y-J%jFGtTStA?F$A9h{|Gool%>w4aI+Nt@l!!_Q!xrE0&Eli<gz36u{BzrD{#zkxrRgW(SXF`(xBx^w-e=OPff;Yt*LPViUGIwZ7QPiTfB_d$U<acbxdj$tfV-cPE7vh^V}>zs9XGs0<=P=JMrrnC7ErmfXL?-ZU>=fHY>k{EKB}5jLKp9f{KEBF&ns5yH|xZbZMh$>9jGy)XR*ndG>t02Q+2cePb@rx>E^AMW+$m}tCb$fBu~g+0c%Vdbz{ni_}p?1|Vhe9aufBGzWdy+z{wy5m0&`aAa>>?;pU@+1<c3cEf}9@5!G&53d6>h}@8)YLmrLf?L~$AE(E1O)wM4lf_?Q*PX4;J+Q~ul*1N@9o3;+Qed6YWV7HvfhnG^BiG+>(d32F$3Wzp-H4m-AL#&ar(Jq;8n^EpauFh3feV**!gbA9B7$K_*BR+RdkdSS3jLBje)9{-i<&Qza7l4d}2loKHMK_Hl@twnsx0zk}-4dI0+9<;*WPuhtD8;)o}ILn4LTqm{w(RtrV~eMR-hzrO<IBq{9O7WSp!<@RV9AN~c1=-O+VeE5Ykwcf`jMeVcHXvb(J1X;`UE9=C!wcgQ+2`6qbvq2{dELH@903<E853Nh~QY3~-*#`^|%lJQwR@jCa7ild8WVTO<22GD0M`I;M_aSQtWN{Vlf<+pO_kI%i~>Q@Ifw8r0$V1IXi9aX{01}DECOy3);st2`vQZ5>t7VNcKp7Ug{jk)%XGdEICJIyWvukJ>t`Zk=mrfINQfj8v2Z6!%t?zjQ37~G@+3w7Ok{R%q1t$xn8c4=*D?ZW#a!{P1GZra^8n)~155U)3N7q$0GXwCH^=0A$gwP`sh3d6s}C{v7?NC#8WNu&@Rln)V!aw_Ez)o*{F_q)l=-fOM<x@NNX24-I4yAId)0!-`Q%0L|f75D#5NN*P?j`N^}<r-LnYaz3u%2%{Y$G2>%G-C&U><&xeyk3mmq&2?$3=Aw9fb)oQrgl%k!}w@*xM)Fo6>-3#)#{b+{&p~TXW#TEE@~?@8;%yxrt>bi<C|*Kw2q1wP+6j5(6|~Rq6PFpU%!bdo$E*M(+*bp>s&z2UzX7PDP~^rUmGtR^1_U(uK{&myQm1$ZEoB%o`3#o=lN9~5uN)tpo`Iq^8>og8qoUBs~Q)hgzv3xv~4zyp`<RxlLZVy=UZ62!_|57;Wf#9^69kbFRnLc=X<}(T^M_-YkIGG0+ziArCK*VS=?_yZ;7I{D)p&+aBCsrh=}U0P>jtpam3n_MhkUzsJ#SN)8TgX0*iRU%2N9!RNpAQ;_<ysjvrtIct)+C)I*{?x6%H6oRS^#QYB9R|B}Q*CDkUnb4_dVwJmjBrp><BtXDi2_Poe&=BOa@GyLs~QTM3V2hF}4fmoe7kK+BIRu-!x9bU{1ABI}*R^{0Q5TNb1`_4vBlCznkoWdVg|I=Ha;~=EGPZq0kORI=+U-Y$4B%b@z<vftr-q$z{x6@Pot5i3C6`{q*37d{VVG+fLaOxaBJ;spAHwU!WN-j;q45)&lFRS}?T1-#Jq_yq*ZF9YGu8Mtc_^N)Hd*b~iu>2pPKjf&-bT^0fc7jr~2HR!^GiC6wXJ+G`<AO3a<>$q`Od8Fi!GOi?{~3I1EZ$1D(e?k4pVywB3H_;G@%<qs5Rf=sbz*4$UA^&nB3^3oy<IV;Pc&ZE{0-a6H<PZ8HxBW$xghTJ$4<<#dg@J(%Y!?A#%BA@EUV%Z6Dk00zk%%}G&)Jplg6iJ0Z*`*n7!a*=}Q`EvW+X1x$*kknU}lTtsA^pW}yTad=MFiKYSK`W>zh|*f(6QR!5-++^Cgu;CM*$qib^q5)W}dZTUrukB7#FZ~&_IUHIBhV=wD)wz>M-2-{%N*eLID*5`-?$h0n3;=U`Cbx*O>)sQsCD+XyM5ZI{rRqYW?-R`l{Je(T4)9Z9~#_vDtk1Hh)_ek*wJ)VtxSnoo^0NzQIIdWP6m%lMDS8v04g;U-m`l`?7$GA~(v1J9!R~Z;mxmY(h^8<QY&wk=ld&Ks8M+Cuds`qR9y7DTipITdGuUxIyu!l+3=IHlMys*6vCd#1$oCKP_q{B9wBEubk?a6J;qlUKqD+K$9a(lu1;bkZ76er-Do5z9F-OlP>4mH;Z{j2Is>k)15gJU_F$Ccjw(SFg_Z(?tf9>g~m0!biV=hIkyP@Jdb_ZQmp)m5wQ4b*IGZfvVU3cGfz_AAtb1GSw^jtsk-L_cOP-Z#S1aM7>ZKV0EjE2-Cl`tOw2>RhsX4;p{LODE&OL&03p*!pX2VJg_#g1Pg}H@@`xOcNR;(wJx8eKJ4Pgg0{0zXn&ooQ2o@HQ7}5i_rd?{WS@%uV3}o;kI3b8(VJ?JQFbW)nwA8Zj;-a<}>^WK;rwc`Ln5RvK|&nzq#2N&quvsssbx4Wp(;NX?f2(UT@iWaZ+;$XusX{uTUG`qC3eNB*j*e+_xL^`SP&6&4<-i60HkENMVk~cQs&pq~2+H*q7r3un5=j#va%`Irp(LZDzv0t<rbnDz5%|zj8egw2B6414YbCsa$3rc=8}(K1k4}c_}J*_V#v<ecZE)$#diF6!%4Sr%L@YJ3SX6vQ?4!Ol(v70Ei}r$rZ6`&!dKf%x*`z_+ttF9WJ9G=-!CacHNt7!S5`>YglL5O}iK5TQec7GZBZ~t~q)Aeq{f*qPt&Y$=ojCq$Q8lZUOhlxEzoiQPY}tes(+6)U&O1Uus8uU6y5|4=#3fIye<M@N3xcc-<P*qe<;JPA*jddCn%}G*;K_;hHW+>3D_SRN&_X)U6NH=KIyF{x;W}X7f4Z5&hb4j4FAo@OjmEG>qM`UfwEy;R@qX?=pX^?`=T*^+Iny!EIFUE}1$9cCS6Q;hw+8<7YY};t0IP2^P5r@ZoH|in^i>&awb#P^jz<ivvw$Mg*N%)Fs9naT?S%s;SU%f!G#k)nHy85GDGFdzQT&d_p@I?dVkw&@blNv(+5_J@!QN1n|V_^Qc_)O>_1jpEvlnhbBjRu@qV|-hJ!&F_X|MGaZZ*5xGqV`>`ffd&9OE`aE_Wc0knbxy<Hrm#}>%mfLGnZoF5`cGYdPNY_lQ2ll(N5|3rpg17<Z^o^%naH`*L%(I9Rt=|7D6z1y6K1lz)G7!En-fT4T_wfQBK3X5yV(GrvaJzCkECv-!;iyF{>ZB?jjms0!oZye%7Hc|(yFKj|GDM+{Bw0uk>+4Zl%8Lz!yBp{W3Zb;n=W<xH=0~*XO)qb=*Wi+HUppNSkdW1=Fx|3`?#T<I4i7_sKVIJZRN3DWHnRwUBh7ex6KYjC-nrZwoC#Z5ny)#WLnr7(pn-x3y?M6%KC5hFdavib!eJ&8^c$R|-Qreyhv{Q{oYO-aJY(6SV^$v#$TqFkSr4rS+Q5H-c<$uNJHH&SkH_2nf0G_uVc+$VT+Pi1Q!fVm*p`!~fUrakWzI2t#WXck_8dq_)1b+8BBE(Bz_fcFyN#{>E#;}z87H8D?2;=+Y4r`yo~%)L>w~TPwwc^U*^-9v^1#%k?PGuLW08>(&b5Xql7l9i&4rzM&#;0??SW%`Y%;LVhtc*n3Xwh6+%jFv`w?c*Db6ue!2BHuwe1Sj)6-&!Pp8{nU*G-d+20oJChlnGSo8F{z~$tznzcMNGb;V5pbqu9x=Y4GD)QXZDWfM1vLC#LofVM}{uFENcdlS(s+#NjCz#p0vuR(+vU;&QkW{sfE<#pAa5R$YD`OTU&~BDlMJ|vW+#1k_lQMi(vYtD-?~`REqs;cBFZCB)?tSJf(6n)<Rmdw(&6?|o@$V`SBfdE3vV9LKMcv<4)!rlu2)V`Y@5QM#ogMU&u10ZcjY_9uowo0zgfMhl<kr6pvLrU)FKt+8_ymjkWT4+;2Qbo_PA#Yo{yc8<mbCI(^Y=RTp=Z)*Op=SNDAezQnVuHKf2?OygTL`zTp(jGCVYvwZZN-B0gt@pDFnc;r}R_Wv<5f-P?r(0DZsPy`jz??vZ?mm8<@FM&wX>(>pD?^Rs9-8BNFu{ja5Y(L5s_cnKdgxY<SGW9YV0ax^(tHWbD*u$9q|KqD_&rug+w{NzKPhygAj^ZhY<E{5s!nt-qg!mym4!VS&ES=ktS*1&{Gc9G<5_3lFpA_B&R(&o5ZK=JT(W5Sa~$SL0O*-e@M$0exJx!zrrsYOmrctufK*U-)&vsNWocj_)=vrv<+7pMKKPC=>OdeY@hf#b(F2Z=?609zO?#FEnS!1mP9Bx9K|k4S)t+F>f7N>MmAgf4+ynXDfCE5kH>Mr>QIE<hYySL-Geao(E_OiQ#m-h(E7)C0gmtY@OLV7RG1Or+-o~ZW$(74c$bQmc^%9ueH1%1E8MO9Pl!=|EjUp$E;hc-%lRYgGqCK@u#I*28+j4Pwp!-PwCttoLaPnL^?i1e|>c%ha_6z>1$P(A%ab3(560Hj|SEVnmc#&xgXC3R-HzC^>~raqcIZtGw*CMKOY#W+`9_7;EU>sLlD0jy}jBdF3lu6x4PWs*iy;lBy+buvD!-1aXS3VN|Ct`&)Km-+JhN1`D`Jd10a~`R0?kr2czp3|GGCy8NBcbwZtmzV&fa#*RC=rAC1>ALyz*`k1bnI@+#WtTOOQu@Ul{Un%gjH9Y+Wjjf_zrZI}IeB{*-z{c<?huZ;F;B!~OrS@g@gyFOVmJKqqyTUP-N6iC0rT&^Z<?6Oc~0lv}imGk%m1?aMNhx!XiRyhQIZWbqhxyN>CeZ_t|jd6F9d&|Sq>Oz-4Blq%dO359KV0!76yN`zabeCN;Z83tojio+UYQ*rU4$ohtT^%6Y6M4V-yr0WH_enQy0h20hgNha+eMuOj<3%qWOiZz(wsI%g^JOkd$?<n~>5@IR8ov?(E+=h#HA<uve7fA$F!oM<*koG5ISC6c@25rh)*EbV`bd@Aj<j;4f*FOHrQmYAJze_oLx2{9CMbVCT3ria)o@q&^j(b%XxDkS>pzAZrG2hH-4mR?>#19Dh$N-o>b_eVN~K?<|Grz2Vs3D|ZS3TY{N@`#H#7BveBNx-jIbLB#-Wwk-QHj%OKvyRc91?!=g;gs_o@TlpsPH!m6n^_N4rxkA<Yf!SI)iOrA^bQB?%|-vhH3Xk%>u1YH#>sXCFInbe2}${u2O~Re*Eei3{Xy8W`59=X!ffpKxHc(aGc)4f(~H;eqF*oPzbHj@;kZ6?EO;-`RZ26JtR(KX7r>rU}OAkA`C|fc}t1z{DQIB?Q*z-;};q%JqA6(P1HY_iK(mdZDAs3)Dw-<GQZEE>T(Zj@RS$kOY*bR8F#46_Cr;N|=};ZK*SI0M+Dy9*xj-*r=I*-5E-~=jGHL*Q;E=*_2s(xpGI7oOxfpBz>aHUq4?7*_UI0!0=R3DvvF;825I|!G6b6y#1U?PC>#aBHE~ZlPS#4li88@hcTJ@q)zNNZ)DZ#wmmOP?fj45PRj+^vC^EQZ~c*l&n5GG&Mj8fABc9?s#~OC&T6cShRM8hDeLkK?W4OXbB3q18uUI>l-{hGEvW29pNr^#x7_c2T3!3_eX;WD#zofk@TgSg)6u0G_3q={BPN9j;fU<WtemGuOdL4S47P6o&VkOLazH`OJ)TA<o_A+!W?t?W{v{cjybjeBHtu>a@*Y}mlOuvR^f&myF`^D`yu=A=sq|^uG&cdgA0vBtbS<v^{`Kgih4=1v?90F{7DLuud&SN%n6SsKL}^JkeX4dCmWVqGMY-zmh$<7WCTsC<Vv!o>dM1C(7ByJ<wswPhetVCUMqcBW>!|?P!)7o2-Fxf`sqRk_I#k*E|0}-q8PdDNgpUf1C?1Tq%XgYAquV{{)WjQ7N~_5bsF4$%#c#p&4_THJIZpSTy5O+AXMa9=F(2;7x6mqxP7Y&>k#!@*_DLfSu6Wu$&pymHsUur>658kP&)mBzv1e#pznhut5PyQH$rtVv@jru3SG8qYM0yR{UhU=u9lToa!|7lem^+Igo3%@2yXa+;0TXQsJYVdiUwpty<h<$`&G&hCpCy0jHnHaYyL20#Wbe2?G=^wL`84L2UB5ybI2;_uwH0n>cVcN#z`Qn73t%dZ+oJeR%;HK?S8&n8N{IKRGE)S?Qmcm{NqYWLHvg2P{`E<2R>e0;+-tTv-+URKO$A76)vX`r?FqkozgT<Rq((<~3KJE+XS8MDW0cE-eppY6cuFj;(^nml&Gwq8g#WLL*gqkPbgU&lZY|nlE-1Je!Pe!XhGR;pFBNxqGH+tx-00rU+Ka5Yt2*4z{C?Z^vhf*DpXD0rSu&CCpygjOcm}xTtm$ptmC!XvEc<(jA6FvH2*UO>_8Q9gbZt-ZA2psBxaGo{aQNI^*(J7G6zpb?9$P*io5%fiKCb8`fj>Sgr8>mHJvc6E2JQR9+2sBi%s$o4OdUqXV79NEXm%O}3Kv*4e{l?-zdRo{SFoXn{-v(}pvFPj1gm3oBOZ%RbDtA&Yen8U3S@;&x0ZryON@{)N&6b~1sCYPKJ-F_u66&2>od@2m!WmduRDMtBy59xFz?%BRsqp^*^fHJh#qHt78%-m(x5bSTJu@AHHz`vzFvOlB4wob0bWdb{<>FC-98^#VG#a$=?T7|YJfJ<m-CIo0vFZ5<yNxWHk!LZcKWQe83Hxvcd(3p;%+d@2=2NMeJ_Q8)2B_86YQ+juh;8g@#x84xaW4lxPwv2V1HeXP!KP^;Jr`$^d_`0Zq>;oe9EXaX4||na}B2j5mzK#d}kfn=GLe}9?6$cF05I2RiN72YaKdf(wZCWkfN);H*Mgj@2*gy?qe19pGhMbkS(L^fH^v?T-vRC6CB>HUUng)cBI<c8TzGQV&02+(3RUTu`5)PcpRA333ZKnRo`3CEX%+au9&v4&sp8Uj9~L~CfEDkh=<NO@3+MOX>>z!e^`f2<czT7<Rie{`Dnb!_-;OVXi9=}gX6a`f@)o|Dy*p;;hF4Wpf8Z~)879?V+pqhdgLlWcMYGPUw3`X5p>8=+$R~=8l-7Zr{gYi?k)&gs$)MMsxfzNt%a8BS5IL;^XdAX)xq9^2H0bjSUyWYa}3^mnS9Kt3tyvn?4msC=>N}Xsi^6%M4BrJa{#`yS!6I4;<jc(cHWyS_?8fEFrhf7g>*JP!pE*Hr_7&*_OgSsbzBp^K=B05yPa}8`Fxj4A2M#?;<dc}%HN+pLOEAS@N-wi2Ws^XlDh1x&CgaG=wHz9#2e<L((mr)0t58*z1$kN({_Irhu)07C3V1e)?xh~$a}KZd<7JK0!#FK-o=7i>A6Zj9;V1av}C?En?kSd&bJu*Q|i6xN6=+)^xIxmOF>uz<foBRzJ*r@-L}i1xJy~J`$!X{C|OM6F7W38gWP_zI*X*xqdwt~&!gJ!UeE8-DcGF;#^xY*wyToej#Y<z94GmTIAb#(45CHRJx%mgh24scM2aFBXjVnGGMR?{e3c9?R$uLo#%7~-!0*_2L0TPxSl6V<YXU#U6161Vg_34$R?&xH{~qD><LA*?Cm%=72ZA)~uN=PKxHjjRMm>9OLr^A!KR`Zk{)PjZR{|tG!xZ$+nskz)gzfVcKXau;V1yA%^pwVD*sTFOE0|T@@4&7ktzD*&B<5mbbL297WA)vfVf3O(QVmxw_&ToD-E8QI1pkf1KWc0bGia4$d#jQy@yS>^?{zYgRuOdYf>vDrEmX|-(v3wgn!WNM+tcz)f#-sw?5{sy$n59scS?Gvytd!<m?nAFAHY+Gt!(t{-Prm=515-w0GJCtrrlOL%WKE(##E0H8o51w=1v6ELYm1@8YSCzx7b|k%SPp%D(+1G%`52J3ZHrYXs<?UwDh`q?QfXj^1L}udeN3jeoRlq?!yti^#^8<dVqUk{KHT{n><ILg5G!6=odF=r8aOyy@G2m4IgOG<@2KbX4{`Rr_q=ig_Giiqbps(co17O%tab2{pCF>qaXTeVd05c-N_&BPg~<8{k(<VUqa5NLZ#@gdUqF)DBzjihi0^&7DmL3(X)-Qqty;)mdJ$`bDM^nCQAb3d7D+2z$a-77}!5-SUC`)tw*`fp7HFC6n&VyofV#T>cwK}#Dj~AeL1bJE!p^TOJ?W|!)6)Eul|6$+dJ63Y66C*>0{kAh*?>^S)WgJ_czWvbVWF=VTD@oKRUXM8h5Ba$$2r9-szPq8(9~qw}kUwb)JjApo@0Y=8zjpe-Yy6?FD#1=hw;>yH5Rj#0R~oW3U#&!JWZPWzGlZ7#+ekz;KW&n2kxwUQU#dRllTgObl^f?-z9Ah$IYjoal8fy2p6jaVi+7&iPv=BIRdvuD%5HpxYAOTK2C${;uSpAdUBvq;RdFO2`O_{bKU#9(?0FC*Mw%W#>|&?53qMPJ^vnHjOSVf3Cw*yt!QMXW-33#=4Pzs2|R&Z{<693Tu6d<LAN}-8TzuH>#i{=SKEEs`Nss1$V-pw|xd%h2Ni=BS|~B<zv$By;=?1I4Sq}?BUvE5K-^okhWK>1Ty7bvQQ*t=9OA!I=9PJw%Dn}Xpa#Kv+Dei9y8~5<53~Gvpw}AK<$@wvoq9EqI_+hqu}6v2z;^-z5Z=6YqQO3a0kU^XIkw|I+N7`40OJjHyYSMc-rZA3*8R;yqU-&{#q*ZIRrn;csyxjQn=zzlUDys!QJ)hcnoTr;skqO9rV~0*dGbAX{GmYrUQC0Dce=6`RgI3gpxJ%dnNtjA+y&hN{$U5t)FLCzjyl>7hjol=AWA$v>uI|y%m%dF7^^PF<<jKdq9Ak8&cx<sb1rt+23&6OE(-J##;;TO}{Gv6Ag0q0n`HQ!f6j}H#>vrw*s#2NTnYb?y&*~5mp_z1plV>mDDrO4EPRvQJMJ3iHWoN1Y>U<>c~ZA<HQnkJlW}BD{oEBS_Q~|^^xXfLe?rOWojHLeC%euD#^;!laRlt32d%?*-m%k>1uv&cT#`4yqavJ`p4-w?>+U=))so*<Nxzq?Wp|Iikh=zrDKDO{Vv$CRrcPC`}f%h-^59Y6jtoi$E&E#UTeG7p1+~+>uVn{9&KMVaveDK>01SqlPg|^3f!pmTSuynZR^&Uytc*GJwGivY-Ky0C6D{eoPBsmNYzM~RP$-K3+j+l_sm1mt&lFw3vbjB<9Kn0z{j8kNU)&L*TZ%KZH?Kobcub#f|~=)c_l1yJ$x5b9<X`y{l~q%Zp?#AqqN%2T|hQqhs8s2sL5ot`FNM{tYfWcXuz)@+Q$f5QhU5?1IKOcdlXk#&t~H<`P^o=T+-4(J{*|?sA!$<V55P)<^E}RKsDZW{k0wXpOb2K*gy59aEPpcCn4oHWS?#y=M~hny~LT6zd-IKC@;w9%gEPTX@()&mPs+9mQ)Az=N>carD23@e~2KnkXivk(r%53k1?1&)A)N!7R|Tj7~7shm$vdePltrn5a>^I`o0JA<<bosGMv=r;6+~4AJO1k)n`VMJYD#tjZMEZ>IKu~fHhXU*+f7mgucg)vnST;$qQKhk;#(Y9hK3U!Ms6%7brYy5yw0m-P=2NMFB!qk=vuK!bu1ADF2Dj9{|3x!?ET}YaC7(OPU_=j@_@%)IQfwPrbqJy8BSsvVG71!AF<;fu<IZ4%n_1suM2HiGO<1(Q#UVk3-_609$)<va8H((0p78FIeu{d0UR5wj4@7dx6#+fUajPOMk{#eVDzR7iN}6YD#~(0H_3%XxDzhPfxX)Y&}$FLt5w`PSm<;PUqOKTD9sOlNMem*FCq&b042Rpqu{Gy158!9!H$Hnv~5|MCc$hue-{Civ}FLqQ=p@g`#04>}p;)L)ROKmJ0YduQ`Zq5N|K9{i*;+hHg~NzE;}4mXGN88D6>Jlpbdyt36q5_#N@{_GVN2iAyqyf;b2d9ScJmq(LqvcU85o@((Uv2)?05-mvD0CCK1#qX(-IF3Gkhmx*%66~{W%E5_RKlb+)e8VgK6)2<eywUg9E6XXU}liidf%T*Zcy}!kK-=7Z6Z?hB|?Y8gv2R#zx2O^~4nP2K>E#>_o#c)g*jJnz{|6Bj(Ov8rit3b7JquoADQ=CIoOzK3qe8!UMidhZJce?^u@+q1Bm@GsV2}JQmvT>gS06o-thhbHXFz#Ks50bD>{N>H}qtymFxy$O`)o)ALgUNTDWcYPqUEY$X!ksog6(+NdR(Qf?dZV2_B9G*x@;K(NP2y@86tLi$+aB08=FV3E4$|;*qxU!kpsj;M2*hFu%=l5(YMy({6`76)5ZQmOy8FpW8JN&mt8Mnl9XP%{J^V_n{9cDLZhtHL?YPIR6-10`LVP*R{Icwfmps@LoABsn%&XzvE;yIzLd$|I^3^mb`|f-4%-g?(lCh(g!0*R<0IW4vbF$MvdoOlV8*#dKXrAsf@`)xq_Ka=<Szpf;jlC__#XGlU=~->C<!M)e*jQWHsV;9Gw+%hi0@nl)?R;V=VrM$o^l@Lyl^xV|1h9J8c$eXP)lAfEBN}vTIY}`35(TF|@M|=m&UHQeTsT~CAKhw%Y+_-Y8z|d4_p~ZSOG935ttl`xzSM;UlnUOFYT+{Yyqd&)(oEgV)CO?<)`8ACihxs}!`d^49ZRi>n0Uz_xZ&Ftq9MyMnt#a~)kZYVuH#Yql?YcH{6-Efw~m+@s#|#dT~{cUWLvHWTAh8{nBfq+{|e*vBgxdP67h6auaAQBRf@XdeZGF{x6d<jPzK;KlT`PY%G9@3SNr(2rGSvJo4Xn2fKr;&OZfG#GGBe7^XjYM?(6vF`qlD6o?A8c-J9ac-=nEnKTqs{+qbFqaP`9`Zf`=b9<S!|f|t%0w*VhqX!>~cd}g<Ho@VrIIY~#)G*6{&x%42oYV}CpphnEQ_Ix|c>ArmAKF9t5eBE^lP{66oHzb||60I3}qItuL=r=sAe0`JHY&aVUF%^yfqD-r!EiNn_>-sBAV(2=8XWkDHIBJ4h^U3@^mz9-0t0!~-%(5+MRr`~hXDoOQUMlH+fYuJ*7g|JKAVIIuvcWlPS07e5qnxSTt4ws)`};6AlD}{}_6P4VAtdCYlJV=M&htQ?g=?F664mYDe75#&zQS(?L&D+9sCWCO*Evvnffdf>diNs?eeHy|`aLq1J9%+I?N>5s<9q)zDe}epfw=Dnx+DsEJV6*o+~KkM;_#ri5g^z1iS2l9p-j+3xY_>?FwOIQKQ$+dJ$?M!qP4tjh#&se+8<YkN)>(k5!A)H243-zdUL!-8XN&sSzzO{6SBKkRRofOHB5&1v3oD;;HI-*9^29i1)on?3$KA>O)13(%*??<|Nh-wYH91Qp(>Ff(Mc^v(tFvXG+3WeWY)pQq<qMQ;W#O^E*K~iXCx*48?^-A!fHrQ2lIV{ROqcPjjVGtxZJNrqg|`Hb~sI&tD#DtocvB5j(V{>4J2IsDN+sk@|dUOEc2F9xu)k2mqSf_LzQ3kLA-B4q<8uV^<Mp&n-%DkTQ<p+kQ1));25#v`!DFNV7;=zX5;+J1lSw=T}-zexy0@0B#*bDVQ3huyTu|5*gKZq%KSPwNvtk1Dk+U?Z0~jG>U~O{+HNp_N?X9YkKe1z<#yh$90(%2j+X{^s6{R5aAK@&11q&fzdt`+b^v*DTl2=GzBNAZ@lqHx8F~Gnx|qndh0oEg4^0=iSDhfgMQbzvV+hKw_Ml+*DemSOJ6V1Nh*c&HF=k=ICv<;2R_Evat&~7ogqc5br0*x|5LJ|0(&8em-TuDzQQO2h;c`=AkC-1)urG7+%=TWRLOamszGf}YFiBq2wZEP?{wZnF&78Jf^0|K+R>fTHCFd$<NZ6<=!e@F0d5>YZQcnZ5Hn8+7{@{VNche4&D-hiOw8_2iL+Vu-1grCMuZ;vOf4YzLwFi-)l?8m$Udh8vWn8_w4Q)$Q-p3Sv3!oHEI-P<&y(=*Qsp}f~m$Y@WQu5k;Y6I!ewQB6*0oWD7LT0@fm9&Rp@)-HZp|bf$>*5QzCY!nVm;UpN!ulvJtS8#X{>kqbwYkXk_GqraMCM_;{<6AXYfwkNU6yp0e*YpF-oAoF?fjDEC_rbi<YQy8k<%#-+{#z$Jpr8YLj(5tlbQw{7M!oQ+gr<&n%DhOn#hB+I-ZK3vObqQGal%-T;5=h`|pM}<X<rFEN7Y24tr~LD!zj)c3s8#yIp$~zzA!<+I07&Uq(Z7nV~G%-gUonrDN&p3pY1xQG85l+G>i>tJ=4;GhFYxeW+qalV3J`zs5mJjwY?)CQx^`Vmhy{327Ra10Sozo2t5U3~N~cuU-bY`<>D~?fGfl-^c-~yxV&3E%aOx0mljCpi^wK+H?MKkX^z=C?@KA8<xJ?=Yd<e9uz=2-QPIi*^NiH$z;&O;{ktgU-#Y;Zcp%5gEHIkTwbD{#tzX^y(!f0X)zF6Tf(!_O3?Dz!OwMc(!lO~a*d(U2Tx~2|6uw`h#%H(RxEDoJj$vYYqzXf&x&ROd$jN4^To+TH+zO}Xm=HanZvRw@|Mzo)(?mdjMozg@8pxe!OmfzCd;mzKby%Ef{}f3i>$xORgsv1C*?aI%$;X|)^qDVk%VoHyl=Qzj||XH5@IW~e`a8fa6Ts}nL0!W<4?(oN28<Bv&IROf%%s@-nGW=^iW+|jdHbVp0}558mtE|5(+ROv-}Eg9$)oQdkMLS#$14XtIfD4;cJ*|y#A<-t9(D#-*vF4LWW1E8?AW<n%eEQo!!&>d?Hh+JpFX-;}mGuW;B`YhVP*7y*5^psC@dmao%FXFncz`78E~D%ahgth@%m)78ftNW_e{s(vdmB(DnVu_O+OU8#JOh<yh)=vu?-?w&T}m0Q=`xFQNtpxBn`|_i0ZgOP_VOiRD){Jy#DzcCl7cbiZJ-f7I50^eH*gZ}J2f<=iNM{Ux0j9CBJbePLYxX4}<fhVAe^loYQ>wOq|!O*auExXoK<9EJ17&O3wP?SK@>8%gFDHJ%&y`D+0yr&r0P1wAUKb{03mMVs25o$>Mtn~6@K(}bHjtC7@ekLZ=K{89n#A3eYsaWB6$Y&nBI{C2_Waoh&*#I_o5q8=Bf)575wj{HXR%Hs4`2FDF@f{B#NJ5Tt=9S_}r`ir4nEot9~bNjxz*=$Q+wYi~c&)?^G>%6A4U3lbj%I%S}-PTsM>)n_VKp}=2;Ga+5<97;t+ICIiD~DOVPjnAj{_r~7D!H9TOw!s-(*r4o%TfTIK-WkDCoL_pf~@gJnEo~+8Pn^mzLk*nwBHk%?!Z#?x`*(>khO>H+Ab_YwbPoucLxD6PFh;-cE{P-u)sQCg{#%%*e|}5f`Jq~Uo2}6h9LKRqNzrJtY1n5F6)s4TJ_h5)Aahc!tYns#@vqX^pw<`N;zPYgnl3EBQuiv*H;^U0OOv9SreYoeHrw}v%~gz>rnNwQS+ikVDjPhF@4I$j1V5n-X)*$6U9hscT6<A$L1T@zj~}Xk-F;Ebc4#OEwqXx`!6^my+*^T#jOIi)awXqRutZB@kGahmklJ{?nI!Li89j$Cy%NX3_4|feP_Ne!*sx<b5>Sc={p&8cCW*^6y9ihl!O&ePD(5sUaVHkKhdkzgk#pNRP4p`vXtS+N8I0|TJKmpt`EWRaa*?chmL-OT1L`Ws9g40wK<!OHXhv(wsdb!QOUklaF2E*tfO@Qk`R4Wn+=pLeQGvpwGAiUD;g=H2)1kgiu>j4THA>dFCRtMr&?pDhYeUe(c5cTQMjP)Mh#`g-|C9Ie4fWNatpWndKkO3cfItI&+mXdn<rplTP1>IcRYlZG7)x+N%mP=q!%seu3`&xX#0XT`=Vnj%8uHeT>0%62-K`ptikFEF*0x=Y_*|d!mSe!SPLCy9KH7po89h(a@qX-1v}=ML+*`^d>nJLs*~xf-gtMT>HG*N=-L^~JA%hs=-^n8*pwUos=8X(=L2?;`U*b2ZF5$XdZYVtDbhAIU3UPNxZoC6-0C|>+<q51JuKyFS2f2{k-K~c?DQ43w`2%Mlvx+u(NeWX%z+IEo$0pT?6nsx1RcQ`Kd(_<PY=+po{k&i{zLWBf-u~{vmrgQ{wOh#UB9JNz;h$d?(k>{=d-DH`+r9Dt$l#u-3NRV;?(P7=0<1NYj@P_K_F+<6$+U&bVfB7)u2fo^2MP(^Y6vtUhiw?&T!TyZsYzIX9c#g>8N7dC7JPVeTv6;hR1KajAD8qErj%>zBfg=skVPXL$V)FVsLq-!f$@|&biaFa<MaaZ+3hZ=f-|oCco?G#dNKEE8o7v)<$7Trs;RS{j(m|J0nL6#@$K8d8LR4vYbnS_g?Kn#1VU4M&O<B$Gum;-!hn1(tD1$a<)H;Kjn7U=5=O*Ml!8oB{Z@Utkd4n94H}!G3kkpcEN&JNwAGFJD$#(pe!4~WzoGNuhF!Sf9}0y?I0p*_eFu~#@pz~=i?W;(<Fj|Z;A$e8s@`|P^_Uxn6BuDtB(MmyN3fOC%c(gDhIFsx3UPy!)WfYbFR+MZK)Nz%x6Ax1~nFHTe5mh>4#Lt>b~X*@e^FW0&hLct)(Eo)JvJb?ieUe&tbU(gUP`?kvPV`VenEc6cglY>z(&j(?Q0AhsB&VlzQ)$ASZ{EPGOwR2eS{i58T@8aw~??dk_OG`qY7ebWt$Yrmlyfs7FD216P*4+V{)U<XJZ-e&1}D$rV5hx`*^pXiw~=drj?pXgkjWTMWfq)&0NiJ5K7S`a5Uhu<=@bhU?qQsEsXbK5B4rk=E-yt9K0foA#9L>!gC$sK4gCE?!(2r5gXelVKBD>2F@w3=T;)hf^_3D>ue_OI5a9Md+u4&s-vTMB;jHk(>7AC4rTW>U)*>YcvN7&BYH1jCq7;?jb2^o;ttg%iq#VyL#-vSEG8_4r4X6XJ&r$Y`!9L`nL0K9YDMQsvD$ox3kT0RUVXXUWm8xXCE__R@<j*n^zr&UGw9>;sUQL<9dxhJS{l9n|^}>sMCY-m{Eu-kJ+C<Z?PjEH^|!RF0#s?F6rY0e>Jg0A7sj5vOqd=KlzQ0zWe;8oi_FHS^>2;>1wY*ImNuo-Gh1-^sx4(+NrD8Eizc`_v>$Uq`Zk6rPh&mZM#0`a%;c{^X3%)8`*FaeRkQ{oV5<LTgi<S=o8%S&9XJoLow%)46XjN*grL`uKLw@Yy8dgyN-CU1f8=OJNe_s2xPxq34?;y*qXE}ezsmfBjZnfyL;xNq+S;K`SGH+%wuxOrV_`vOL$LOb&7nf2PbB5<7m=UUKQk09J-+5sDV`^NWkxV-`#pmsF->_n<mpeNr|K!_yN3Ah1Oh|pU~_&hqL|{8<4TxIgv|oR-fz818(z$$*N0VNl)eS#@`o9U41&svC)eY$#*~Gj5yg($8~D2>wK`J(BW=5+|`~F&%2+-m|)@+a2@z~?sgtV{S?wW4ba57w@eJ^FJv-uP|-fM{HoZ{$#@0h{#*YWNROx9ZM9ADBjRLw)a1ck*#sd)z+wQ^O#3aC!8*JaxAtHD4B7U%y5YS@ogo1}B#t$2qW8H>>yN*0FlwTBM;^(`ZFCc6yt~Na0U0*#+GPrb`_etZOgsjq_-gmN{%RGd88)SZNB*iE>D_*g^WE^+2&L_rznx=8k8Z`NV_UmbcYK#gb|$9zrPI>@<-Q#ouM$3W;oJtN>e67y+Y3&r0XLkyYD?g&)sF{v|L!^vOeeeR?Jv)O4zXL$%BVHYTh8X!4BkU04%I>7?RV47)xJ^-M;koj?|?OG^GlK<0JgM^d|RP@HR7)xnr(uAwm5oBW9Y{gjox8@ziiIxo^l^(t2)^G$Wxyv%t@5jbGK6j{2}yLjl9`$;d{(Ee{~a0PdelJ-?QsZyiJS`OwFtsj(GOVBmS{Fw4${tDFc_6>7=I1z2lHLZ7O({93Q>Ks%!dsWp<-`JVmEFs8!F8(>i6twH963cF#ooIX+S1x!KsgiwIT&@G%HKPL00=OE1)`jk<N8wVkUzYn$<(6rM#<xRuLF=1n$|+)zDV(s+5lyJTl{vPiwnbPaC6*7~TB=%IBhT-=P(n|Jy=3B-<a%X`i3&DvSMCZFv(x_NXu8nnR^*DKoO^$777ZE-||b4i<8If{a18KPi>aO~5qZtp!(D2{$BoBsuJt@a6OFo|WK>fw2jsrpMqMmKXcf3xZ6)U%L%JzX?bP?G>Q6=tsX8_VW$*$?FPqvP>MiX_`*G)cALfid0LMzVmj(4FXF`HFwl%JYVS@&Gp-%=&}Nwmq|i^ffxsc|Ox*JFvm`pCzEiu|iV6rVz~T@n+VQBW8s5jYZ+0-UkCue<%X6RVnQF3-~(}{Lq*6_{JRB*J#(dw0lldn^@ZuqdCWvT%FFHvlS+H`n?j8<?M4VZ;-NluSa#RQ(WPo%buF3POMGhvwN%Lz;~kLXZ=5fY<hyryv1Qqqqc&rgZJJ{>&mj9mwnkfti{cu`i}JR$Fhr+hOtDmq#7;YR;B2-nLa~#b*OLc$W=WF+g+d;{<_3xZ2R8I9$CCE<3?A$V9kSUi$3o+O(8`vxP=XW3&e%X^<Ein@0%ZjB~r3evKEqqnK!7O!QE%x=KUu)R?m`K460wTF7%G`!z%kcvUZTS-R+_8O<v@DVIS8v`V<UuJzQzsb6H5#W|L!+zaGK5QuA?Vk7ccnCEc3ZZJveS+O9TU{Q}SD&)t!`f0KjJW#l!{jd=uVEG=z3?xgPq6<Zjr%EPt1dd#ZF&7N?Neq0%>)(-|0OR(0S)7N$U!;#AtiCdHAx1k)=&ZpV;DXAE+dIW7MmA!cxoi%`cy(WX_ZuCNMNwM|rF5l|i`GD<O@3;@T2+VgP>EVB?^DC;Rj}O}M8&lbXl`89+wRejjt%Bq4+R?MLK5nNMgg1u!Ql!my^;;JKrZYOsYDU_RS_fOGgY^Z3Mt6FPsI>;o42_P7#|^Qo>nb)lR;17KNpb#1<4xN&Gwac8Tgfazt73HG8JpOiV6=t&lzknJ2K`2|sa$`{P2SC!up(c8HNLNG_gA>1?ni78z29r9+%zij{;7!T(cz7L1>({6XJoQpCt-P!Bbd9?jXiDcfkG{K<~PgT!`zfjzq2XN24(y#Mc7Ps=gDlidI{m2m~|H7Cq>`X{=>AK2Zn*)?7Hh%hC~m+%(X@><vbne%R@Cx$6$59PVKHJJ+OXqSghRbHL8+C2J9P_9@$M1LCFIHi(_|K?Iz2<z*lSMq;;0Go7cNs6-AnmvmUz)`}<yX@Ez}SL_gT~S5&eQM8|!+=w0T;s4di+7Lw_EwmsUEl^L=e>)d;IGMG8}irOCiwlQ6#FJdVleDm7CwTfdnctBp1(t2qVWZ)lx!_wT2M`s*ryzZNnn6u+IvLE)hMU>AMTIB5)C=D#6mori*onOzQwf9lAE|RB{UolIEKZ!Lui+D1!SX|{1$Tq2UEv(Mat8X3Wb`dt)upXQ*p1RpLQ_6Cn6%{n^#I?5->;BUY9Y9gzpTYC(h&shRu08GD_BLOx?|gk87lcxriSeuG+zqLf?BF%5){wr$W5e`C#?#_R3|;#6!KP~pyOUe7#TdweoY5>CaA}&(buj^9ZN)L`R2bh(`h1#}XEb}fdsCwdErfWkFq^hKG{M^K?fk!rcI7=jxGRsc&j$`WVK^Yfi+>RL%HbkxM>lM%KlEl}96!zj!=XyGx+N&?@#QPJxzDWc&V6KGUK*b-wz~VUyYBWmcwCkJmwgVTXBu3r_;bR~zwgGjI(N0S@a*#(Z33_h4gDZj>>*uTt{}Y<zQYmUI0lt&8XkYMPGfk}7jwbhs)5+ludQ7R4-Ca@_UB!-8qbgFXRR)42KOgP_s6>8(Nm_)6xWkSq>r_5c}EN8-X2G+ep?U+mfLx5CjYI%u5h=;jrOFz;oZ*X*KO8O6K2#sz%G{16+ypn&qVx=cJ{W2ho^acbi;>RAa0+&GY{4ewf;rV#+%IY1z=*T>M$aQ#>}I4RSkA7=xQ4{Bg&)OE16I4$m`X;Mek%11iQa<O0}g*MDTn#X{OzLD``NWH;2KcMVa#}v+SRfd*!~^t2G=pvAdt^qtM&V?^5HGVK;8=uF38@^r9Rpa9rtB$vZ$yub2!}4=Er=Ak`)Ur@bLqWFB2;U))#!8XCB72J)m|o%OzKA?wu-c|&&Du$M4&=!o#)swEb2di#rbTfDuzK4gEvQo)7IYtYkRzx7QlVbYdVHB*X(O8s>&+HBHZzFy04fgGQYUUw=}yFjFpOGrWKrrA^l66m!G@WNgpR9SZueLat}89Gf){}0){j4Qh>xvk=lXxGTP*&omduL_D!Rp#@Kp~iw*!e}{=*VQgQZe2d5W_E9*nc_VBDs@O*4cq?X>D->za~e4Qw-|?s#!M=wDHNfTN=S7Pk)i`Blum#9^ZEWRp6hh6Yu|^L+1YIfX%>k4)h(ftV{mxS$~wHmDSE%DB9BnHY6W~hUV+xFY*9Jk@bHOz?C-hF{|)^yArG7P>yy-QV)_|A^sD&}4|hdubo|3noZgc`w*pHq=>y!7G}X|T!f6Xy%+t<UMmONO@}$P4Gud?a_$ML$`tN!-&h1_^!=~3_VeaAGh7O!);M_(_8D4~f3Yj0{)GB;yCiRa$@H+2egE-(PeY*$rk^v26W_v?EPSs{a)drtObt0pSXp6Kloy9Z2jDc@+X^w<?vnk1o#_3n;GQg(Z*{e)i>Fsa>D}a?9lEp)wTz3^d7?K9=OlKxcPOSEw+ICds*myrK)6IsUb^$9*suH=zlBn^m00I6cPvqzmU)0Z7oL{Uy>kAZ_FqeYEk43Usj@kWG&B!i$f2}sx_A8iNEK`4S{Yu4bD)DoakHfvW*?nH)edA&J0P(4h_0#<h1$G-Rt65$wz`l`979K<PO+7|{3*5a`eZ6YF#@iT2oO43qc^!Dp#?W%JUtn__&6jBaPP9X*|KQf8d;SdIx9QIg3>i&bMFu45saL7?$o*4Kz1qFi*YGrfNf)T7I8qO_s3HA9n#*sS*AgC(z|Uw#6ZhMgi0D?`6}_c3lp5p9XVP<OgRpbSt!!s-WIhg>JXys@!Rc{ov8f2jcF`l2Qss9Z%x}_3+coLum}|r5h6DFcU{s}NbxDCslsN5q`&*XR&*y5Tgm~zu1U&=m|84X^);P1Zlu?)PC&?c+O`qI8sqHT(wJ4toxml!m+9b|CevTlp@?nz?pj1{O)}0>V({3=nJy@yq*2$50y5r^ka2uSQU)b9h`8%9W(a64R$^^f4*7A8Lx4n+N3YepR8Z&{hc4rA)9~fd;zG_2TYSo8os)!W@btz&YIhKE0SoBHLCxbR1wAxrlsJf_o`c;2clinIKaKVXGL>M3y+1Aeg{ednNPVgP0)RLqj&J(wb)-0rKKZ+4dC3UlbuA}oVawEkrtwXzmFI1J7dbrctfFQYQ^y~9|D#T{$N$o9o;eYLl<MR@{^Gf~6A8X}v@0fCo=4K-)?8*D#S~CJ^3EVBacUxT+pL*u%UcUwx)63D>_%gfi;_9L;BfX38uCxzJsQKx4gX%@*tL)Sm!mHYn)@<f!yBc3_hS23)o{DdVg8FjXTaZsJXgOlOyQ3p{hJK#0IJ}~l5i<M3J42wvOMcMj%Bgsz-kc*UlO-?Dz7?*Rci~xU>~wx9)x04n?`^(Il?4Q;KN^TkRqUf(o>XUFE5!y<aysAc?kU=Q|9wn2SM6q}&Jj$y81~u~_m_cbnq3Sc)V{YQr}neo$*B|YByCN?yc+<wy@GSrep$OfDEsM-KF8#F9~b)i29D{np~dDUy!^#WMW=UM)!QW86sX`kkGV_y3|Q7LLaqIpg7=G1f&YjFO?>c_P=fa@9_Q2HJ=UCFV@#a<`tsFZ)J%Tw_-w5`=CS7Y*->jzTkK4Mc1A+eOwn^_198SXzh#y|wR&87$PbCe7;SsS*ndi3^V0E!KdVjU<V-xToEXvl@>%R^3>VgAyV{Y2%1#)c)WEN*<S5DyX6HRtMmp9hQHE@^E9=#(SB;wQ(d%@B(u;9_HBPFcwLXoh6WfWhzHf#i08S>)3d{Bv;PB`<L``zrm1xrEPee_gJ~+SFV1PMsZ-I7uLFr940HMC|^CL~MLs}n-#$8^R5T-iHMU=<YkA;^5@tDQeb|U=J3KjA#ZAh3EpCOq<Mk~O*L(T`NTC>)=I3je4y{)|^{Pvm?&n8~CrMxXc<q5my=7j#+nz~4?z<hTeAgB2!l^3!9UM1LdnAv2$A<^q_fE!pyNh(*K<D~OmvW{&&go>CI$86W^&fd}LqEmm91`ghAp*4Ium7m#uIxLyaZ=Q0S$3qY`|El)Imz+}GQkL*1#2`Q*l4#R_6fcxR6|KHIm_tRk_i$l-ZFq%)52MvF#8rPT6HWTE+SJf4hMRMVtmIgi$Er%-{j~#aRfYx6?1k1fdNrM!Ii$tS8(wlHVt-%gC{vb6{xv!UO$O-$6ZX`6A2u*56ITn!*48e_PdHZ3S87O}YLEBtmG+sfVlA~02w$DYDLh@y-l^5dWb_Udy7K9{N>3jgeFA<AALgs)&>3s|dK^Ro@STP)$7$odc=Uebuvqhg!w%43vf>cXZX1h*vT-kAH%kbHxGWQLy+Kn<ea#JgT!tUviJ!YPtQys6dlh#leod{1Levn?;uR6ay0dz^#Dpdrfw3Zb(u1x#G+km5^%x7mSh9+?(Qa)Np*cv12?1=>)R@Zta&hcaKa(;t`k|UYc3$xWXzYhWQ9)w=IHIPS-M9m*zhAyP5zX%&G3x%TorB_-WDLiBt7nJ?*m9qMqn!WxgZ%jnV;x<<D43HQ!0)na;Ml*_G0VUgS%@A_hx&=^o_Aq0{k;cIgCwstX5(5Fn&-fTET!X?AP-Azr4{edWUP2f^3K;6OXj8)tiTU8#1iH#*=zA@lHz<t-6}7w`!%0s)XCl#0FUc_Aq+bA*dn{wWhogW&-z}!1miAHy4_eg@CdVRTWW3H-`Aq;aaD74Z8DR4hz_IbwyO%0r(9_sPWSoz*uY}5i`Sk0$Gr4L>u}vw=k-&%nLLT2F^eC?ArIAWl^({(?rr$Np!FNu1y+8@buzNsz;>zaNR)~|mnaMy8IQ9_soz%95X1fu{=wuX%vBA8OFp+wWwts6LQAO*#cfp?lf7!$QQuj)l;~9iIM215R^Lu`jZ9ppFFy7(Yu0-y3+;;!)%rtMw7vc`x4WltDU3#O2%)ZB&CcTq_?XMkb<;XePYOaT4qoznN~`AJapu|6<Jp2WfD+Zl<oQd(`ptyCtuKprM&)77wENn=(rRTT2hDw>J-XWM*SpI_R4+EK-K2krKtOwbSQ4-IC8l}qIyEQXRdqCLbX&V4L7vv>oq9dSbPd&zdIO@YfKN^pQcN;+fs?hiJ-tf59pCxxiZ^_2PAAFj<*1BFK)<)r-!UDnKU>&D9gx`0wqDy_Zu)!Y_(zcGLhrMZ>cB)*|5{VG>x>C(qp%b8S83>cqlLWjYg>L?G_Rl0CU$8dgJ4g--_w!zr>?i%XI=Jk=>bbz==sh`?hFTCb8H)jwLL-X!F&B7r5B*RPfY1)(lAN(D&a>R&;vhP<O-MouC2+UHUgwwmxzZ;b<3J?Oni~-RHdu74`yV7;t$L#lI?Y%pBD($k>>XWRBJrQwzwb=sm)z3KEpYc&Xb-KRHq7l*7$yy97J1hWdiZq9T>PhRk=vQGdi^vcIP|ZSQxGZP`h_Said*J+KDbF`5)14R!hSn1YVkY@W$$DmsQP~Xm?Ow&^U%2U~OLBh(70O#dV4;Q6&(niwt7s&YzTq+pC<9D)7SkG0%uk!5cEt>1E`l%;|KABB{B0u1?%k#;KBf*3*042hD)s$f*2LTlnTw?oA|9#`}2^4%RoO6Jd^7+o(U(D!Zle>vyE5Fh7<%ss$&n(fltXj8z|;LoC7Hf7A1vU)n=OW|pjBL+`M6$wl%Kombz{!ocCUvZovUKNhPGOtY^x-k#>Qq9aX<WNWwpO!&}SwI5|8$EEw80N}@6o{LM&k47Q>=r`hRuQCzY;mU4a{@kev*GjwsM@v-P(jgMeW9T0Dt5Ws)P|B7nw;iDIF`q8^ZK4+mI*{_)ABW*^OhqoYvr4%g;*OC7qZim6Ujf=ns%ekJkB1Wd^HUtz4GXphSh^R-M}uD}eTfy3YDrQ&RSI4Z4RNV<B_CTIdYg*6s%-qzUk>S}`ZFsm25UG(_R6H&hdvGN)gSRc(BY!9d+i)WfZwcKtVwZKnOtU{;E)YFRlXS_1^E}dGnX#b#7S*XTWlaS8xgvopP)L3elwVJB>vda_nz3s#ZmsEd2ha_mHC2$E{kGh0p$ZZ7h>rY$UUnr7ung4(TXO#kNE>$yu`~T^FACO;#W5Dh8?K3t;ObEMIQ0-&?wJV_z;*52cl6&Y0;Oe-V3^(9Mnq`fr`N*`&RJ{75~;T5dWC&)@kK=d=fh5S2%>dB9x?j`L>+F&wYM+rL6#`$^NgSeGgCY7{(K8l~;qms;})ljzn<g1tI-e5gNzKwAid`lUmEEwfwo5cI(#pyt@z=W?;RDM#$OwysZq!@gy0$$F6CV*05i;v@_p&3oB|@6I__8!>>L0N>XW7nL4N2@iXRnjoE!Y-%8?-{nKJ#{AIzU!I2p@=v-SP-2O1L!QcTn?Ae8~Q$D(|)c)B*Rb{B3tM6_81y>j?#+9euFSc#%u61s2Ve5`%NpHVPLU!Ap@Jx;cEm^T1-B1mi^Ip>s!LP|`w>52+fAxEPaZRvjZnATO_MN)B*ep-&dPwNJ>-E;U5$ql?jR?*zqr1VJLu!>`eAQ<Eb|DQ1lm455uNUmJUYmb7oTz$i&mOy<@1Fjw{W-4SAt@s)cV>*Ktq&B8?_4g?;LSwcxdZcP#Z={+*sfpM6r41Qb!N#ZKePKee(u>{urf?rXy8t~TkKGdy|AN_=hzuMJuBTZY>ziS)JQoFt7>+?0WWgTdDn`nmh7^h>AZK?YExef@?H7E=mYP5>&p#zJeZVoA=Yfs5c<-vGKqf^KCpc?WIN*1yhtqwUu~6n`Enc24W@NBL-l;wi95ute5%1Da07oSDL~T1E^m!L$S|<s&}vcpc_+0^l8_S?k6HW+61dq~MzL_>XD=4(?lkwD%=`0jvu8jjyCnGMuZQ)gYF0h6+U#&0PAP20RMNvH*Q<ECL4zAo06Qs}i*ve>Nnh=>8y@cWS(KvssOn$15q|aB`0TrJ$DJGZSLDn&7COUV+LQLp8b=k@uTD+k1WOR@)DUiQ)CHzI@wq6VERch^P)#KQU-YJRbYmu1g;hP>4_kjrfU@J+y#Ad;0}zTS(tx=g!Io7?@KbR1o^{LGsa3r9`0NpGh{}56>|fs;QGY<Cy|1w#ist8MAgJ21L#_T`x%uo%pdQvy@cX_d5rAR#dHg&r``11YTGelM<7PFnGxPR{Zb8jr<v1U%R-d<XI>%q}O+J^z;|#7u(GqTErI<oF21w2P!(@ZD_ZDQ`W4!n_6=3p^>a9pwA=4c}2ct;d)_Q%#a^Iff`LrHKZAw4K+$OHqn*wmLedo<~MVYU|JN{ue`+!8C3rwSa7Xhd|Mqx1;_Nw(7w<!;9(&$kzUO8LOYHKnFUbFkaq}LnoJ34y{3%j)oI>wJu*k3O0$L=sOQLdm5!7SV}>|bV{C9sib)mVwo%O$VO{5dqKBX-8?@6SnB{FTS6olS^P5M1V&aK|sdRu9OZD|%E_e`(4gp*q>p$LLA=X#iNS539Y-3BEnM0BS;~7t6<6J{tC~f8;0s#>%PhsqeKYhXZ8So~+03Jr>W-mF-0QR9gOE`IqV*<5E|%>SbG7&__s6)Mh=xzPBG~=|Qhiar6~x^+G;)Yk^yB^mLo`{^qdre6PRz?Qz|Eb^h>PTMn_UaYsq8?Hk9b=q=g<M%`=trPI--h2^i_-E;k2YaLEQN|FVBVloH8xQ3>8%CDObDNia{{UhEd-3H~>Y*pLZ%RUsfeypG%FBUP_*zK&w<xJP-*J48-nFnCJkljZGTu@jjD}zDdR`vc7v%%$1bjx8II+3QXgZBEM80b0hs@~3ctC#9$qoi0t(Rtb#tkJ2gE0e?gzLaF1n0M`CQ=9MhGvMZh-G?`#n*gA-vIZ(YKc9Hheur%}zg!V;rtXQj#(j4^{@7veHuy#k3#C6(`vz&phr@G^*s{_-jt0MYSxg4?ytxur?`O1|u7C4Cyai%*iSLo`VU6&*I@9l}5zc1KVyN@V9KFpOkJp_x_o3EHcKUAx^y$`k`%vC%C#x9KJ6~!!bgV@{)ZF@$t<Y{|8zf&@7KY5V0D&Bn1~=WB_-6sAbRHwqwB*H4SyWg1jo4jw*8H<yTiuYx4m}i$OtmMUXg@n#8v9Tw?`<w1&Mt4X5!5`lpW_`LemlL~8I7PNbblUNXi9fPz7T_ca^@Fw3J?{?{3Rz$y|teCS>Ls74+eV>!xl90E|L?wkQzeSJ6%BPFena{o+Fiz#|vA~d(Sq+`W=kj4f0(<)C7VKjas)e>RyJMWu&#bnpBKC>Bd>#dW=0E%*!c1(TJ<3&oe)q@6SJFLflhko&+6a2@Be_vmgl7%holdxu(13Yc~O#=JaC+Z8wgeGj*#*KFD_~i-?)8h81CZGdm))%mwE@(@uUJB8U!l;?&v+lN&W2?@f{6PF!!l$?>mwQ+;Vvu`i>;8G+f?YVm8G)`$4$F149Nx(TRaEv+p$Hqg&oTxA1Cy?SR~M4$6zxm&$Z0%L#6Lmcj~n&shcZ5&=_^tFBEU@>p<nmkLjS<O23p4EH-)byPVt?Jrw;nfC3e@3V_zCRtE@t^LkW(rlAYwh*kw2V<|PB!{#Ld-f<Y%^PRVL0nmJx{*lTLeSCmj%)yyHj|3r`pZYT5Ww6%aac+TK%zSD(mvK+o7S=;otAoAmY&BZQtm?L{(+uAdR`_{o_hr^xgYjcQ04D`xDrg)@6e&{~hJxA8Z@%%FK6(rnO0B=o*j1WzF{~d3L#B#Wq_>(-qZ221HkA4HS4iY^L<kfG(A~#Wwb<n@wJ~16>NAnLJd~AnjpYJ~@KNN|Z+1H+8+Vcl*a4xDMK<7CyM5v~`|LZ=PGJ^?KSnYTgUfc${iSpL;|dVi{l%^k?GUg_;BwsAnHS{3giuepa6{A`aX(|5XN3t#ScBh+KJm!o%h?a7eiSId)#dp|qRiqdPzBNvKw|qTRlBb)aL})-%hf#k|@;5^7JSS~A{a%>GI{idj#0xe&mET|<@rAfssb^_!&(d!0t99OW?_PDev-gh5-Vk6W!_tskuGf@>f5grh!}Z%^89`%E10FdScojePVc$YVH$l^O%jC+ns29*&3j6%mqymE_;5oP`r<6hf`o`4zHkh@Y9>k$bt#z#ym{9_C?4%bK3NesB9c?>~>`2GhA;@Dkh5p~@ELdT_HcZGpZzVXZb@K5rbJHMVa4@XibAft9+yDIcH5qN8SIeFSfe?n7;|eX`CDF?GAR+w;Q#iK~{;UQVs>@&>u}keV!q_#oSZm5#_YJ{^MA`8i+dtkdA*pG@4_#0+aL*YwKpRszpCpOAYQhbK>47Izh6bfz>W2KsE?&qIk>9kl_9(fzzPKIBuM)3J2={<;-yqSH(|o^>PVsO-j9d)s_Uw?ow+_LT}zMLW~c%0MOtVfAjuMj)KJQ}*kni9C=_IyoFq#D=pw48_?j8J;8t;veRUVkV?20?bR@pIQ%tZw(`%S2Un?=TR(`Dj@uXWjt>881hwlvy~h<+x6<Ox<1SGdc|MYuBjh9M3SYSdZlz%D`^=`ccwoX4lDd;p-0=kyU@>;TJA1vFT(4Yzs0VQ488l(<c+ST+X14d;vb%L3<F4h$P-{pn3I2t?G0L$l64OmnVuZsN~HqFqaC`T)9tvZelE3R)P$6NMg@PtHK6=Bj$cEf+OqTP&HR>OjlZ92aeKC&?%uCRhIl12d2eyp^(|vGi@%kD7Alh<j3P5!9q+WIO?#Z@>^>~oIV%0)_~}oFu*$Z?gsN+exeib$bKBNjDW6K^+P+W7SM~7(6iXZU7etNQOTO3EC-VA!cG}k~DXg7Za6dxwv6-e&{V_gPfpAtnC1F4k{gG4n#&F#DcD@7|M?wgY`L-EeyFB6~4(w;^rqF!e8sEQ8A*gr#7jOk)zAJMI&l@Z3knI=4b0>9g@%<ugFuYD2G)fCMyu==ymC1tTkhF2nl*rMKb!UheXY6qszeWG7INx3U{8#fYucmsj8`Htp<t38OuDF^DU7oPpNuwH#X3ax`N2HT5dpEc00_}{*+Q4-e&0AZ@oML+(eNH0PZ{7!MOV&CE_T$$dZ}L3b&2}F=YKI$nC=`_ij*nhz`OL|SF=kNxT?^I|JV7Af)qtZ$?>T6#HIlAK;33|S>fm#jCo57MvtA2PAC(Gqy~B^z@g{>Udu2qAO6;TKVcGM+TH!7$&7sq*!J)I4m5wwIizQTFioo*uqxQ{!XJg`et{v%yl6(yFB>0=@)q3vsp#j}>PCKcXvn4za-mI=WDz@-v`{10iCCJpUA6`ny?QT8ZAv`Q0&?JZ3!S=ll<-Ey)?&G@gUeRW%HZ^L#{DRHeq4uVBALRw+@G0d?W0CkP3VRSU=}as(jm8Q1St;P_=cccYbZE|4^}~5DF}_Ay4@D{+yV2Ly!4zGUTjP0tg!xjZ3!thM*Sh4z`IA`pt6zRw?pGHXpp_0cNCutluo><SS)D-C%PQ+Q^;aOZ(AFVXt<aIMVGlc&bT|YERiA?tsEhD4c@ZmUcu{12^Z6isOzY3hBg$Uw?z|>7^>jLPbzF9&Z<D|{;TQ!uIJmZFy$!#OX5uuC?$qbxACl;W0@1LC8^0+h_K&rO;ahU0g116^+{}QH7Ac3j#(vSQ8UC${y0jv!#kl`&2~`@edp03q^J7f6>%_l94PMQvGLg=&Rb|hbv#PYrN8?4NZE8c8G7jRX!Ckyin#F)psUf}TPF=o}YG6Qc6RLGLfnXmRcaJ?hX6j>3niY<OdifYm&2cfuepNtcfdv{asPLWwe>PN|Uz*h#tvgZho=#A*>%{}&dKU|!aiEXyGZi;(?eP)DRRf5p9V%!|2Qag}Sp)I7c$lZ`4HWc3s>8|aXvhPOR^C_i!a>XFMV*`bXzD!=ebCe7F-m1m#~gXavUE4h`rEq}8n>@^ln)Y9M=&y|H%6|h0j$0Jm3s}JpV{MRzP^WEhgo#ahb2@=Q>*va04MNE&Ner+QdNq(2IMq2Nm$}_z^2*!!aT|vbecIx$^hSDGg2RXZPf3Nck&ovmiv`20!?ny3Jd({R4-XEyUgQZ+@nV~<}y)$+R=uulPU;=xIU{-VFW+7KJa@~l{Ck%H<65~4cXSn1quYjFaEDM8rMsdkA^aBZ~7h4vetlLe%p7+Dcr$%#Ju}J870@>sp91+WOb2vFaw!&9a>#X;BmO)oV2XI*}qsD_EGb4f=910_mYy@<4(@dU#p6DjHA8_QPf@Yqo00$j8zVm?<SIs($++L$K|IIl!!Y92s4TH{p;>vA?~CZ0?z|g4*DSMbKs{4P7Q1%v|p%qDtcxA>^#$g|IYOD^YF_fZ*46%gUSGk3+4wutB1El7`a{csl<cvCFQf_@8%p`qwQ%7<o39ii6boodaFJ%2wKdUrAzrE|JHT3=s<5T2Z`Bx0%A>N@&)(j<(Sg6{@^h4AUidi$-MbG_w?5G0>t_|nL5?}AL!uj9nfWznYtorH(%J-y<&QxPbx~*vF*9u_s7qAe6Ic0?|0+<F4w(5Vu(upI+`a(#o2uy_l?P~y=MPAS_ck7Fx*6L;_vcDj<!2@n!U+t#cv(1jo~>p^e+boH))%HuB%2Mcf;Xi2z{D7ELTWE!Q3laLHEbbckuu|?^&7Ot5c)_roU&S18p+&aO;;NE#q1x{v@{x)ZSsu>JRuvfgHnD?3zmXdg62MW~nN~Y{cpsOsxKfCwavbdM!Z44)A#S-dB0I#!o6Eovt<?E;;~?Vy#|(?q;<Y(?9!WCd{Eby#G;;&LObu>Dz<<?A(p7!?oQ90l~@Gl)MTey#?>@YP%#p?}2*79gE(LXC&1X>TmT~Ie(0om36=Vz-ISjrpZ9%CzMw~o~DlVhBQVx0Ka-68$7uL23*ZY9v`+>;p-3Kdu@U~*kK>>!P8MD?SO)Ys^K)MzK2z~Qb#vF=BznxKV!S-w0?`(%e(W!ZmwL&-1^ea8$d3A`-5DTnBZ+DIQv=((K{uxpKg|rYNA`c)~Vso^F|q+%Vb1q5;u5FudE@Qc>rTH|2;mYx%U2zhkeWD^ihtJec~2Z)YrNHc<b{12p`uo3jC|-qm`KB7|vAjb$!~HsL|xwy90tZ%|{WRj(a1|uO6QAMc7@NE`z>m*Gc6tL!rY*#4Bo14$iMxQIoruM3U#XAU0@(9C|<C_;}tHm0MzGy8-d&(V@K>BtzpV=16aewuCrd9rTsCIe(K@fkwLaKt=tw|L!aMRSgcdmcXwnl1+BkUS|bh??~iI(#~C@=?TD-i_}<aF?LOFU$%?aY=7tKb9=0&Q&$+=D{j*`Tw&T$5SALdGdEcKD^Z*+uIEjsVBbu^&sekP;O!&(!i*u(k+>_<V^u*)l&zPo0@mqHJN=PIp8B7MD|l5XVd4rlY$O5hBo>KOh_4a;c(xc-YuJa3v{LPNjkEG0S*4j2b<2m^8WsiQEo5h-=ITZ~v&LdpKCoXqQI!h5Z`9T?N<a>9U0?^bg*82AF6bX*-~J6Owe98sHQs&lJ23=+yB!Ue{J_^2ytj$0UP3opiM8HouIw+B@83i+T?fzI`z*;C@u<?oa1090+kM2AJ2NBLK#`JQnUdxiTpuTnS`ns#{O?HnA9+1&HiL2fyV1Lh%6Wt}z#efTzoYQ7PD&^}^@f9d^<Et}^;SIIpFDRzRuWUIMDD9j;KMNS$g#O>z`7vxf0fb7*>`s`cxs6=m1$+>+pkVqdN<oYRTnSx*WPC|@6;}2kR^g{OkO!-(0h$P>cloGGe;cP9<2ed%;EH;nKi9lf$}l>@c($Fs!5nTCq~F#1R6e0>b(u(Nv9NhW=&+KUiM6eu3Nah&|GgXLb^RxPUBjAlMMJ?t5V%so^}Wx0lHf?*6&ukNAqRA<bDILy1r^O^7wlXFGd5BS7U@?U}N(stJH&Hg-r#)IeOiTr!qg+<wxRqTwTlC+}~WtGKCM8uF@11n17GDcIfI9=o+$dG?89C`?_^&ajC4PzL-7|UO%t53!Jb22FU^1*7w)D_42!;=WTjjkI(IxomQyoFk-g*ig@cCW;xB=D0jX&|0-0E1IE_<q55ERa<aE=K<`}KF_z(Mc3CWzN|_g2Qv1%3`+MK4y{*QdiMilk#YQjX`<*r#({azsmiz%{j!Up=Sp(pPu>qAe9<Hy>Z(g%AqfV@`EQnX@g`*XpWoF%C;je*gK847I9r^>M*HAogHejZ`W;Zxx)}7Oc`~!1EsI5L*Dwah~-F8Rk7TW~RM`5(%f;&~13y<42d}ups`Cb}{jfTGBx-*)qkNT6*>(q6Q=(2bW{Y;N)%lqfm2EfNoI!jkO8i=?03^e}wn8SZgk$O#94?jN#e|A>;wHwb9*E!ez%o$%VB7a%`!$qGkUQBzbYzFJltv3pdI>}5~wc*WN;X>9cUF<lhiD+;8z^MHp5HC>D^Un1p(5U9)<;J<cI=DVu_l1Ao`{Cc`&%$JVQ*-NLy``npa?p&<%FUH@5B*g*z=(~+Sg^HgBi$WcC>hcmQ+<nHDYqE0{=hlWT-uo<esm%ASSZ+nC#g2Mz|Sbp`~&yl=E8eE4orDB=#aTWq4Pc<T%L(4k~F_xEolIh%Q|^`1lg7LciA3Q*5R^;7N`3za^WfiTEh-dO*^k3=10<aRPfMk1z6RTH@iD&17K(bR+4n*NbNb#D%tHa$D-r0+P<~Xqao~F`tXZ5qsL<u{23v*4!^(GqXu_}Bsus)ZC;aC$ZIg<r{g<FR5uesK|_N8k;(XRZ}x}om64Aoeg5tfBIU!U<FjtfK%WoIFw|F*wx620&*1=K>KjB!S)ICmvD>RgHh$C5zCD67q_weqkTO@cZ{H8q`bDJkoyo3K=xLn+EMYO%<JHD`Wg;K+a{BS>Wn4Q6t!diB+uJ{=hsYYbP(hmQYUViK8-vcL(aF}u?ki(ENUr3sr+v?pp7?%0ijlgQPJ{FkHq70^$Tp{t>{#2wFs{}WVk=0kU1yi{U~=6JpPPBS<*nY8cy<;E5WyDSZOMf&eHTvkz5Zx1#dV$`m@e9lK;PBsiS5t1tF_^#_rn}&Fq5Y-=8D;1x;Ky1ZF|7{aizA@4n(iFqz^@RibYVb?RkLg9V%n#x|w?S>G;udPUPLj6wkfiq~)sL?-lCSuFZa^)OIf)k~D2;TGShi*|f+;xT)AadxliHOV;Wo`5ayJyf(`^(0xKC2ds83j{55c-6AGV5UjGo;|;^iJ4ORAoaOQGZv346n-zD8)$sf%C@V{MLw{6$)dbVi7!gGsgsju@y7-<g^%cKI?YnNb^=>lb`<u(63m(7rey3!M7t#}w9duSplwSIJ(N4Yiq&x+ITpnAh`$%qnlMwp_d%sQ1{bY4}_V;_d`E6*^x?*~q&~!S%ejl|@&b<-uE2|WSnVZ!-otcdAx$mV?<%|83z5x}jL4wR^?uV_SJOBJ<Fj(E@0X~pd_2TK+k@MU2C+sizHc!W8fK6<r@<zq#NZ^NQ!8k=7zE1c2iiEE-X8TxtY3uKewT|b;FB^giDw@G6TEu&{goM$;p~&l>@vR<)-28qf-5)pQcXul;_4E`R=}TTE?Do>rzMs>&pHuW!%VyPcTl3PtvK1*}w&RU;2R)V`^lUARzxsBJYx@el6U6*`jy6SSQ4OE`;=)Mh9kp8RSAY=gtNwJ1&ZcmMiZ5Ad+jY5_jfCY4t*EEm+hhHihs#$_x`oRi)2?-*$!tz7rt#&OP@)>2(}6^8j(&L*i~Pvp6#Wi-*V=Cm$@B|W$OBvT?O|hkn7;B+T2FfKL|HY|6w^2sY2jIMKK3|kI5Rk~#=>H@j4tz91MYp$ru*RXn;-T#(u~xk@saH_vX#ga*zQy=O**<6;pmTF0s$U3mSb>lY(Gb2LZM*&X`|JZwP{CUZqE9U2W|N4>U3AH3iVQ-#OgXxXYFoupzb7$kZ-uXmN!Ue)2>dTPIT8jaoy-h_2=lspmmdKKM8OmvW_=d+_bDg^jCwk-QJu#YR`GRBBnLc-9Kzi!+U#aGh$Gv)__b0^0>cSs}reBU%XX5vd}RyZ9D7!-|!*g!*_$p{&l>pX4Up$@yRjyq2fVSTGek*A9ZPF-M|Y9jl*^HJ3$DAJQ`x~xgif79G*Eu-oKwGYBvDHdr7{lv^5IO<#IrZ_Z7zHRc$-p6=1me5kbI>CmQ_#Z#@32ye!V9`fDOEyorC5+ZNOm<6**i!}WDA0$YM5qVzErzNhtdD9)!u*{tkdZZNKM@BS*Kd}Fg}5J3G?r!Sl4Qxha1J&tQ|3ZL`m7FwdT*kq~h6j5gIX4bfzsDs`%ude~XT=|+s#7p3rL~poXq4WJ|zT1DydlM?h*ZA|%P|P0?(-2jJ$)rA-;I(BSc7)gVb-U{<FYP|kh*c)-&e*e`KDD_Tm=kCMqAJhUj*#BlsokshjXN1^Mk9J23b=M$%Nl@ANZ%7@Q%{7e$n}ODDu9;tk!YWyqusUonGD<1?GNqy^2$h@&~`Zo`+>x&%84vcQjE7IjHa{L1}8Z%^FO+LU$ZP;nU8i5nCgCC?}HtrHKjfd)E4UfL4D9JmPB^&M~#f?B<ReQXG|KQG34pvo}|Fq=Gog`wlemL2gd}}|DG`Sv!)U>f0VVVqxM=NhqYmsxNdf!13pfShojJ-%UUHqdm$u~wD2Q1zQTy;_H@26qlWRf(O`nlU*@8NxzlS8Z(CbgW<5)#o5AZa>x9go%CcrcUCY{BkvC(7bJpqciSGtCuo(2vF-NT;zHp|a^iIX5^8)<189lhksDIX7B;JXGH_(qi!xm^i;AoTq#cL-|$KShG&rz{HzB@Ji{#UE@Vs8@DG=O5mVpZa)th&FYD*zQvV7-(aOp&a57~M25ov<lr8(rc_=5*NV+L|^=CMjRNy*cZEq^~SaR`9TQo9>jb-<(7Sy%)FDODV1uR*{bn(Iz$nx+iGD(km^7D0d&N=&Ye3#;T>#;<e4?Wnn+)U{kZ6f53|+$2tJHmjtag#SSDaMR(!qe2nvP@m2R-o2;Lu5*DELW`}OW1mOJ>=;mjv-||Ca=gf%9cx+@0#pP#m{>15K5pY;{-~F}SO;1JjKwLJC<FS3{PtZ>`-S58KqYCwXxEh=v@p0vX3^{kd_VTq@v+&_HeJ$oas?vFbE_jBw_hf#qAkL?6tR<MWUacpv?wlI09Q?eR;^}#6j?k|$+2alT{(QXfydzaB>$0ssqv<a{iXfm~)BY0ZU0~*mD>QE{sp7RbwC2t<`R)8aM;q`K3csH7Pwiy}_r_{W=YNc=R{YI@SkmN>#2>$6?43s)xQ|#-d1KkvhFHUIZ&3^pyIvbvI(71<R(w6z^Z-jy`8vHFC*Y#{Agsn^==0NMN=FJJ%KSlsA~DQIyqj5Y(gda=GEVT4c}wrRF7vebr!x8Xq~FlM=^Y0O59Wi)Aif&+HhyQ|+V=L)`PCV2-Q;7rw6(=pzCf+JBSKtL;Ya%$JDYewm)e!Ezh2|*73DnNH^+0+USz_|X;F@K*YBueJl&nX^0JzT=dcZpkKvT-YwpRY_N_(bHDr3V17J=6nzi$qZo5T9QO#Ix(M1v7-7*Hypy4hh<ZoS{(tNm7rLKa}p>057jyK>5+n+h#Lmj5u2%c2MHGCFs2Bja%=Q;v=FZYyVT692lSplr5)AEEUX>aL^vkB!=#cR#6&pYekL32`3>G10dxC8HFu5RGn{Dsg%n>eTT{#~fu9T$no?5EzCO74|+;wB1UTxPd{2_cin8O8qLF6YaCN9u5oj-sn>``5Z<++%J{PnG>kO(VBw!XhbbS-Q`*PWx=#Js{QIURrqL>P8#=k$|a#wCyXK$I`4xmk0|l7L&@h;iYxQSj*>JTtnHbyWA|!O<M$q+j=w^x<NW5J{)pcmf!ku41$*;e(Zs+FnHzG3$R8LENU3IM0Hf}K(__o8ASYO!WiAd+d&Surbsa9O(`$OLG_wlC-k@6Ht{_$#z#{>$X&y*>qIYS@Wbi#b0)J}wf?bhF~`Q=6XN1ns6I!FI;Byt#$|H%@kg7SN39#5cYI>A={uBbh5fzNipj(HT&kCO8*Dwl<4ql(WBL&Qhl@OO4-j7@_if*}0j{~7xt;Vo*k66&z8$ambqKSUc<5vV6UFutg7`CTK{0a2p?6swQ-?5%_qzs@Bkh9&NB!QzdBPk(e*$OU(!FglX;xniyCnv|&<o?w;0j9ufB05oVBi$%MR2>`B3;?+(tC3IoW6!z6Nwo2(&|H%Rclx;=0o=c^!PFO+Aaubn^sA4FmDC8hJ>0?<|hOpR)5+(&l%v^p5NYnrPUNl@>P(d#uQ>w^AzNh$2*iRZaGuq2C5z#7`Q$}0&Q%;)ud&prTaP&@Wd`lZ}-%@=|VoQ{d-b$G7x%2GZE>FcjOvtTcnHBxGv{(V7Xl73Of%69)3SF+ao`9T_`1`f(TB3h4CU!KTZd=74Eh9|1B8T|8Bv!9pil7MHz6STKoHyt$A_x-wk_CTf9P?V5IH&z#YI{U7dX86^(WH)`L~*h=8XX{l04ZhHw*L>f4a+a9STYNmBYN<cJaXX2m@2xr-uLlkR2q0Mu#GssF`GvyItm=e*ngb>kSd&3b%r7Omh#&^(2t<S_pGixUccE-c4?dDU&1I0i3eb2E7bMG=#Q5k`KcuEW>b0(qk&Cb@YhEgyXJ8)im{l{>72FUGy_<Mx5=?xOTio7sK`DFS)k^c}Z93;rv9di^SA+`5zb9Uje(+xcy{Onkz~Sgtjg@21|uilgM0(i+mR>rVn28D=h`x7`?(G6;kCb`z+4dw&yzeRz3HcMZLl*+O%wQx(3sdiwLpU1&Tf<;`pKw%(OPK;}<Z6Z&<gtA%gPVW>-sjn9lVNQR88*Xu++7L(9lro?W~ABu2>Md{|TJp?ptjP7LzUmk;c9d3<}zkUTBMcfDIVwZ@14eQ7XR)@(v((B~1H@?UZ{!0vbsM%t?0{Yn5KL35!Qh%hYR*Hnj?y(T*<N24=#gb^-%mOkZvekTt?cEa&xy9~Y6l>|74CgKtq)JIc^=jFTH|v>N?R`uvqH?JDn*P2n6)CSK^chdWs96qnCq(nd<=NSKcu<#<smk(vrWq$I_a~k9^$)|rfZkqCI{@5T&QWjkv#2I`G&Jm-XC@fz+=%w>=2Bd!!BuF^YiGZojjl~#(9$YO*{f4&+&oVZK{|1u#J+1}<Sl*`O!i;+zn+#LG3f@@av@fS%E9kmsmPr6leVnt4QUQmffLSaiaM&$do-oTs~xk87Pz5hi+X0I-{shBM4mMB<%rR-OQ9bWd(v2~Wjy+YPePw<F8G($HS7ENSSai4vRwQnHgg_T7^TY6^mIMPuxz=h)`d&uEWppi337B|ODex>7BBYDt@Q$K;REd*bm`zbs><En^OG;uY%jYlddlu#IVEnawC~JkdsDq{NxZm@{He5qr6x0)9T4gM=>x-qFf#k)l0uCYx?uD|0?J9{vp}|7zo}rU)x9Ie4wq)wfbFeRXVsijHSY4D@L$+#UDUif*LszEMPE<490BZ=BR9I1*o-~t18DZSAIg`3nW(f><6C)+w$SU(CHC(u(+;)AQ0kYB!*Yh<Nbohv`|L50!I9L=DAFE@Ay6UsYm1s+PxWumjT?jO*IS+maQT?;pSLR69G?m0phE=F?yNM4^rqAEHJ)^qh=+d`?IxUd-e=Od@6<`6Em^fSH8fDT!|zKQCA<^$J}t5Nfl>5TvZqn$Z&cGc!xo~*&lc*hPi(}g$!bWKnZq@<edbu5;*4|SNuSue*=u3Ka(FC(RbD`PQ2XNwGi(TC!_Qm&asG$A^-42{WVd$b7aO{cC;k1NUo)|#4VR@EttX2vIGV3ol3d@OBaUF=jeW))PB4i-0`<bNQ5n^@-^*~7g>8Qkv+Dr!0k^BT*!{BN8)yow6-cvYqil5P)(&dbY|k*m?eu;LZYjwMPXeLgvE*K;tfsi|O#ZlF;2pi^7!tqD#<8kg+`pOG0)-shg)-^jh$#ZfEwOkw-yTOV^ZTM$;S6(hr-nQA9`C2%bkIL^vKr^bD}RYiDo=EF1cAX{t*BDag&LnYKHJLO$@JCVV}qykTC+g152X|OBb>;(^|TePTjiXXH5rn-io?kdcBQ|?!?q9L?A-IEKYo7w^4X$6Z;u{LWx<%cEr{n+Fz~s`Fu-lugF37H6>;}6UQ5n&qu|zP+97Fd1^M9oWn}cs4?0_Wa+RLu&pDWn&X;I>S>G`=o2#fM_t?n;-D$|`DG!d@%VhWGE7WN;I5IJ?YwR}d=zAjrpgx2$ZQBSp&-mPIyOLa7@7Gqdg<0mQzWiYwi!UnX>yr0jw=|k6VLSjP-$dkVxF;VQ((6L#<{9iJGOQ__ehjNzU0%FyeG<#}^|0014Q6I7ZH2=9%TTLje|@}0J*Mc7JKnk~f4bg)RwB^_PDP3iatxg0dZ6_Aivho%kR7Tuwbofig9i>q{TM44`{<N6OCtu-+{PeTE{LLLPDR*kwv8T^L05xa@1f6SH9^*pP-3?`JDA(>!5>)T(G#RP9A`6gPNdac?bMT!x3^hB`W$xSDC!vS;(hIGC#ylQH-hB&byma4wQC)|=;{wTgi3hc)^GU#7(2J-bG<f<zm|qfG|fmz6YZpg?AoXf5@};oq(s7N|9^kaD|pW2q}E#Zx~}UpP3uO6cT;-JPbVHm%KQB|;-GcWb@k@9cW$nMJa)t~JLzx$h|A5m`<s2xIr0$&e7kMBX=^^HDds$H1t(vUYAbf51xFxr8oS=Yc8*QGU|v?x&E2%Z^x#EMug!bC-jhdfX3g7J)H!B#bN*S@Ym6=Z)#vdHdo}TtlTA`<_qy=OtIahTY^)~?zcrQLUnqRRSPNS9y8YCx?#TYS>vp=pOWa~Bf1Aq9-&AIwO<`DfnVmk>ah&WQYIKt9u+~u67BQ#QZZxWN0%-@k%y@Q}&vmGqB-hHbUmt4R_36Zp1{cp?yN<VBEg%N-v6GJ827ta2Asf0x<)9EmO*r>c(Hx<ouiHCt3y}Ig<z)*69PXuJ+qcLf4?&wlzask`h6cKY^YS@X&OCLd0Hj-G`iata80bPiUi4^y+6Na+wew`@QLs?)Yd#mrk79F_X2@-K<=%_o;Pql}?-j^b^da27_FM-AW+|#;uPPO0QcFK>W|c(`+|2LlDXvWB<a$o<+i0d8?wHHaFLYi%CfnmGz6Z@i5Og)<LCkj$PW+nPXL6_;9K7?+`0{#Ey6f?Z27Drx^_3Fwby>w;SYr-Q$byF4&3M&;{l?+=yr6>ty<2Y3UHiOI24*zMAG@?aD2@0Lmks78zGmlTcblF@M>z6N7d!Wumirn;&rqT}qoYZQ*6YC3LTk3*nel4Ag5?UZ{f@vvs7$+cbw2*CK;rR|llI{D80G)!TtFTWwlC+pk|S+KEDHWuy*=L!?!M*q?^iheAS2BQI)v3KudmQwn8(nEPL2WQW7euJ>$YCX)Yh;XD%^__!KKO+3Vqai0EybJ9l$+tyjX^$uAh_TXSLEG#(3QAHbV|AwgA#Wvm%JG_&|OmF}+|ocajR62)B%!8aIrz6;0Sd)a7TcTLt?^qeQ#%nB>~cpK3gG7>@ID`M+)+Pj+>>J^7{hQW|0*u7-R1q1#f?Z3VAry^{+9uNg7(2^;c9K6iS;Us=~{s#GWT+xSw}KgXdhRbuH6L#QW6U|E2{PoE4th^g#Hk2Jhy&*8GqF0XZ_Ax&GkETvog5_$AW73ngQIu)GD{WTR!M`!9M(EcMP4auJOKTn5j&Q<V?<Q4O^>W~*?>B?;quXOdYm+xDf`Tm@JV5P13pCD&7efq@r-P53{FL2v3Y~eT%5A0(kO=iI*=`hI<zG6xC-skUO-AwtGa_-H-@n=OLfzk}xpWv%h%P02e%V&fA4WE6yvgun+g{#%aWuQv%{Jw5Ce)aBPm?_@rnIC4Sj{PkW=K$Ud{#kdc5~gqS?HtalJ)HmZLgD+I2OoP3qVi$KM8R%xbB18dgOl_8_{VBZ--&X{e(x7)ao%qsFF8G4n!0=r7PB>1gmsatT9vvQbmj5R$B{tpJQmAWFK^wJABPmsWq0&irzx)OaLIy#d3LCf+L#i(&Yszl8~gDd^ahJ-9`FXWIJf(7FapTm+|C!phy*9ZYN`>zyM64<`_A7}PVxi)<J-~EmH$fRHumJj`}A;&dKE?+*zZB?TXp*ais%+W^xD-}xD9IkX$5nfYX*M!yy|064ln&+xIay~EUVTAGIk5cFOQ7Cv8{KR$I1bE(bHwT;7O>TR<lyq7Zro=Z}k4>>)$?i%0)WBNM@HKs>B(rfBzK7v3)(#QKwOw8;cJ;w#Uu7TgFm+F<E+oFSqPbavkF9%w-2-7l$|CM4`AFk96Oi$#a28&;i*1jc6?80L!Q69`RMM(`?!8UM&0S^6ghVEkS);;?N;wdCFf&8&g2XFtZ;DMECx9zi^(4_h<uV>Jz`JG4=(oh}~s{_V*%Mr=V8Tes>1<CgcF8o@6aY=Sj0yf}IoXw>Y(*e~g(z$ntpD8+YfOu!2s0tK@_~XE$S<NxwXfG2ab>ovO_~O>2WYhkf`+Zr@?s)BB`B=Od&>c0>V20i)l@{$!FdZrM}?xk0@K;z%ypv~vg9*=6AkFX!$}j6s+f{X0%GJAL`<kJ-1E)wpDi>iv10dzT+e#50?pP$a&DMJ&>4^o=d2eZDpcDmMMRztTITAiF{xF|SSg^;)wIM%K($szdP;K&!LVkeh>``FPH%H@-o3MZ?GX25{|i@O|U&dPGnqPRnupUMQ7IUuh5%Z{DA*#?^h)AJvd)S9g)4AbD53m5~D}Ymcq&sj9qa<jQ?!+J#w(TOe3;*i>j3$4!OHR~}bZF#c9+u0!Y>d$Tk<Efw46-rm3W-pFGSmD04k`fYbLb^RHnCWPN6@17igqI)hrRN%X#mG%PovZolb#=sLr$_k_?1k+xI9HQ(lZm-%fV4ymhhqv{GzPE&e7+cSDBEk25{fFph^4yYFPpOzZpJgMf`%5NoirM4J4OJO5@bHqIM(E{sM48Uly)~(3PdF__Ht%w%H&!!-^6uxxVTz6W>ax-7K&#l>OQXlkMZM848RPY~@?d6Xlr(!9-KU~$cy9p-ABOy~Y+py409X0K-~gyftd}RU{fe*x8Bn2$+#7|t|68oHbUHRxVr9H%p1#|;?Q)*G=_U^}-##~OQ=R_C?H%67Zy(yvAD%N8!Y(S6<EWra?y~J=3B^NehubT`d=-axM#`3VBFKQZG8V2LU(Bb#uOl4J(UI2BxH95Yd=0L}9QTO8>kl_Hm&;!@14ptseyvV(rGdjbQU$)h?dbeA(mi!`ZM7iE+5_a9r97Iw$D2>9bDtkq)dfTD-eKa)G_;<(I(?pwakLV5r^yUkKYp;t`JBL6W3C?DZnG}7ucw3h>x!&9;!yc1Q3szQ?z0)_<%UWc%Zo0AU;Zsdt!)KKZZAaF?8E7L8|Y)Yt*t-#O}dMZ*O{yC=l$LVIoU?iRDp5^?S7=;oI=w~maPhkjH<o6oX(SQ<5uy<;7=F!lPt1yv)?&hFT(|2rUAF64k~n}Ax{01rs0_m@0Y2aggQo7nd0$C<cTob(%3Ey0rS?e7v*P&_p?(CoN(y6R4#dC5QPsKX`IZ#poR|)V@xQF<4MxG;EFCts>{UDN8RSl*T+vJ+kzLf0$ielqVCf;jUs%6>y)!<EDL=!7jA~p`xB4@cj%0P=t`<GHQv0<wY6uy;IgsopHS#8tZu7)G&}Y0PW(e@SWz0+gGV`6lCQpltL~!JZ88X)%key`AD>@n2O{-XyqL_@7%h*hQNML(wv9;>W1};dv};%4|LLTnh~LjvHEXCQjfnk-q6$Av{N<IreNpsrWn|7#3wGDVrE!BJqmY!^)ZsUqCqGu?d)8ppXx5GcrC7~kuY;qLL-3?zz}np}S1r!q`?UV*m6E>$(PrJ)9g5Atm%`>g8<BI2LjEq&B&@5=;xfE3R5{1$g9Gg@=PJH?8s6oy;x8Ct4@U$8%+$K#<wDj7+fuLDtTg6>0Bc_nq%u~|Qw-=5H*iLbKG<uTE*$6pb}XhHud1JIE#8u!jTkL%T>zEa;=6$&Roj$Dku^B5S=2qsJRpFpjA_EZ?&atof{g=20hgIywq+g74D+qs`QC_{eR!D}a9AFM_tW1c*m7g+z~j#1hk-a)8`pU30%smN3C-2F5u{hG;`*OS@08EcU0h4jPigl_+Zvjv{0@AR5}rqF5vP4$IdD5DYledt&4X=hD<|exs)d2lSxP&HrNOjdm^>N5l?kG**5O}-^h^~Nzg!-q_ow{PRr&|i_wPf8@|(P_T>Bc2oCfcr>THDO{ZG%|m$7F&`?TZ=u^nDJ(spEgIH5X(amVYV;BNiWuQa<plH*#<#A_FAKk2&p9qm%tj9R9N!8y%QaBrZB)IZ(4UV6vq*>tj-jAnl7%G-NMm>t`4OluYIuZg`Or1PWIegLF<yTSnSxNHxo9qDBVi*CI9Wrn={*nKY>IRd#17n$^&b^&&CPY0r&9jsG0kA_#MomCeyFk`sM*RbxPECt<oCSINku9MEG>V5ix5oD~?F;!_d<ZI_@*V~6D!LWc)!egaxv>VPTsg7AtqU!;C>oen?bTy~I-+q3FR(~8o%!Y3Zaaj)^-ddAc;xq#h<)wUw9@leL&0Tvxf(L`Le$yAl&Ev}?pijRWo6NbfO$q7=?sb2YZ$@3KRY6vav;iaAtn0A9l&TJ2&CUq2#ig6ij(@wD?zSr)*kIYyIAMi{EqUMz1!7BWl@O%V2c$yXI{4}S?QAZ{(E*3>-X6k4ra&*@bu8b>a;}LN@<vlLe$V&K)32<Jf&TuyomvqwS7s4#81EMQW4Jopz1h-8XESr^H<vA>Ifc%(Yu6~d<TN&#<T?DR#@Zq;Gg+QuCW-#8GDGUgV+Q2}f5hUzga%6*mOoN<xNVJ1Kv}E4wcPS~t7*oAaRJKCmOg<aqv<z5y^q!&;ttw9JGIVPNM+obHji9(fL8Y!khd6s!+NoK?TO8+J$pCjm1Es`9MB%85eA^c6||EF>M&#X4BQi0^H!&^rpc@we&I9<r9vU``m7H8M4%>i3u|Kdf;mKxnU?ETl~|ss##BftMd;PJ9$32ctv>3!N&(fngr`x1yeu8~y;~N)ZFgs4&Y14L`Ab_{rp;jMvlFYei0cBqtb2$KJ59e|fsX690Ft*KK3C4$9*x;HpXvSU70PA+qo&I9GjYd;;nhgvynIda)?lmA#YkOh8{6<($?)~j3+3L&>*#y>>KVZ;TzJhRwGSBj{_K0xse2Nw-vaxB;HWM)?zN=WcI5MEr4&M}2v$_UrOSE4$%vY8Yxyq0&O18FuZA`yYmKn8V^CFeM&2uvy2U)HjXv9!H8aHc_T5fHFuK+!gPs5G#2<or+|K(^t@n!T7r7aJ#AegZwm<dgo$9sO^zm9s>&4*6bYE`N)?@ixV}7Jq?CPywTcNM1Q`t?Qa=ZboFc74U!7Qr~F`MXr^X6N<kdp0B-(5oTYa56amg`qC&+ti@>amxWs*?DeJsV*=2v?<-u>O4A-9;#{n(IC?Gx}c+ob{IzAlqIRS}Qyp5HvV!c_a1p0@!a9p|H!re7w=upwT2g_hjwX8^+9JJHgMJo_AOXabvLCy3Ma|{w9Vbh_sqjS_RvZy&OT1cAy`uQ@gaTlr7=TVxyK(@yv)v1|#+1FA<E@YqPs$ea`4$z6_VuZ)^#J)gXoU>sRW0NoA1go3ndz4yzr5MO1w_x<MN_bAjt__~0)eGFz&Z+S6$Q&ojk@pZR=aUHfRo{)8O<T2y;0e(2WWPn)9eDSk|v$XncP@*%3MQcR1CEB|X<?A=tJK&$p)?OZp<qH~N8-4O{^9|!{J2yOIohbQv*9GNrcW=P<gnh@gGIDp`h4v6wvFOgmyHIZa;kZh3Y3=h3vHCg%IVxk~IRn4n4e67z<k9lvUMapVFNm2hg9m87sOnu<8)7ZxFJQ?Ur1t+Q)we6sHg|DVOMCuWI>vGhHFv|1Tu1vl0`?Q9zq|y&B@kxXucTaM?z1k&)DpvtD=+F$lW#y|11o8o6HIYb-p1ITJsNWB!Uzol<M(i>8b9`fjJQB^-lgqt=&Bh~O77ID8o|dgsC&Bb$RP(xCN2}}WQ#!a3ukm4=){f4@uh?yZHdYieY&C&swu;8MQFPE@LAdE)S&4F_&ZU&8(AYqI?T3zP!^2C}T+_oS=y8UBN!3I9=sb<2^m_`#o`L)G(IwYq2Ql925-0$od+@kUzTviRs#QG|N6u|}-riKs*VS{ia4|Gt1h)oN?7+iHFlslNcclsk`$Lrg-w%G1;aN!_{C+OaF=B=|U61U-lTo?e#PxSpXTCWuOZgp-<@FqT_lLiI6pP_;+KFy|<@mD3G@WiA^6|S|R({1EZcLN+UCQY3w6*+xo{P;MX+3*PXKw^GYQcZ4kv<i)<|deuDWfsLy6UeZH*Q1BdRFDAn%Hk+UU&+JOiw`9bNL82D$y*?iv*J=e~8aAt5X~1c5(6UvD&u(Aj|6-pHE5!lG+73k<ufQ+4{l*5Bt%&^K0yNg58KB<zH5mLtANHaYy5NP)^R+&TJdo>1=Y!1lGm|o_gr_a@yYaxAS<}Zh!V{f5aKj&30t>4K^cL+h!NWs9x2ohgGSz*9&_kv1QbGb81klK-M;V&dzowxo%$7O!!{&aYSkRyF6)ioexMO=@^ByKMML&&;i<KzaL+EWrm6UO!#+Z9m8msz(FY|<nY_K=lAqa6|<}zCqJwtihhF~uvNHxt|#se_xY2z#qn$De$l2<es^VfKaPIX3uC=3ecXDotKC({aeL%t{@c9VMk2Ikb0GaG4`Mwyvd1Mhk8bD->#gjtFTN}Epig|!(Y6-i<{L+?ek;|Q+wVi~Aq?UON{nB&lX?R=mr-kvRmqFk0amYSWANy3rM^+N)!il&R7}~jef+UFRVi%0P_li|wlSEP^N&Wg?|8Ju4RXz6T|3rpuo(ZKSoS6Db?GBwKQ7Hti}AdbJ%o_{cvPSATN}%VyMANZxxsc&axfNF+aqT-ZdG)TD3{81Hjp>=0Np-M7jVkH|FCRbNh}eq1=U>#G`<UIbf{N)+)rG|1IMh>8Z?`_byUC|ZH*E1^;25Q$U=1-XmtWMxg+US(W?7Hp{XT4QfdUzj@X<4|J>rPRVlsQOk!MKKvs2Ns2NmQ!gG~d%pPp-m@{Lc_`<R0=VJkJrzPz0clj7$=`9&oR)3N1%RhGQmQ)8yTWmud#fq;<Z+I`4-&p8jV9g-lHNt8}3oK`UG<nwtqpkeDwhCRQ5-uGOP1X<wU9IvgJ{Hm*Oms}i+q^%h>n)~N!rm}60nw{l!&6kwiW+<`^_hCSF+eI=O)M{8<rC9S_(FMp(eqsW!qzavPlVNPw#1<wotg!2w)T@Jxn2!sc$0tizXmqqMtlnW{iC-);5CO|@!n~>2_odAZZt--RVwNz&>Ic#UIqBw#}Iy-o-rS1KCIk<nAyGCj37&YEUa)RkG0b@sq-(?3Cr@d5UD0K9|seXI8VM#T$RDj*ev>&_|Iv=bASEu`*@eqhz9-v&#jp@ATfpR&vSSz!eh$_)=OCJj<aOyLgc+NR<zn_@t6JVlPagbZ?qzc+GgCsH>1V-yrjNAy5>zLhxK973jeYh3=6wnFAO$UdZ=D;+^5&q?It<a{Pje)^L=*N_E-Mh%-B~IxyH&{iAFni72Nb|<v7+O1sEA)Qg!F+XPQRIGU7ok7jj|Mn4d#_ut3YFfMC8oS2~T?5&}+t1E;1Z-g!<ENIUV^&Q7eP4V`=nyh4RMlVr&1eaL2=1-VYJ9QdC?l2X<rjIIJbSIx_rGI_QDzwGqL2?5v7{%p9!tL+o??^Ma<u{#KkY0iHZNz>wi=l*~z`$=s{<QClXLzo~T#WME??9hVFN_+ObIJy65ui~9Kc($V<0mR*eTD`aE0)2>=3sud(GRSC4@zOpbOKxusIiQ3lBNNhVH>geR*E@4^FgxS(l1Kg?W9A=WwnkbPhw+xxJ*A0ph4LHYDmt2t&sENyZ>LYYXGDiNxRQI%$=P~ej^KN>xD_XOt2qUmry{z`JZnUlb(B}U`Dn8RZ>A>0+b%G2Zs!Ig?NVHR`TC}MGh0X3iz-_t9R<PjFZdSDMCw#pYu`a5GrA03h?;H;Y6y~_`?5zl;`Vg6is~uPxliV);N;DEeobjn{Qa$b5`1%?odi!aP7-|y9=OvvnlK*45H#Hmlh`lle@$bcasS~MK&(3P{M_;s>vI>@LE!dy*S_hypcxk2G~Ye1ueZ6Ber)w<p7-kJRLAE)l|8XNP_7?M<TFr&{Xml@4fZs&8--&J93iklsJ#6VEsd}Dt!MttYJVP_qw7tPt+Z42D0}7EK@(|7Al|;7dXwdO8r;icbFJ$b%FCawkPW8;)Tc`I5Y+X@=E7fT^~ZH*Yp1RukiD%JT>R|WMQv4e!asjLlBy;D*!3lC44!loGUy>tE5Wl?mF)Z$U||tvAF+ncmLg1!7drTu`{S#*sWRjGQkslTo-H$?!cTR787|kBwv`TF5-XZ_JTxTxEMG3n-C&>XyVn|g>jJ0MDYCQaR~Y$TznB`XQpxY?LD|9_T0L0T@3%+!a#|~|X2@b~gY)D0JBg&4&AVnl-}>!lm=z80c{$(pG@G>-YbG;W>uVX+yA@P}r~LE8oL~GSUxdqJXZ1zCcDMax#;ZETF^BbVA3+LHfcf8#8wa*J(VuDmjbZ%408G_{AC9iW?&Y>t-UEC;*SgDnwYF^8+-8#Mi}yBwKcmYUvyUxYqwvYB)8AZwe)TdnP<%gW@0zRJ%Z|&5r|>HjoeVG#9Z|=yM)&#cPu3qjamT>gI*w@q+Witg8xNKT2bivUL%4q$u3)IjwuP2-5Ys<C?*Z$khSsoA>y*u6yxP&L5DIAa+0ARVO&DMG_I}c@jk!>EUC7FAe!91b__60#SD>|@K8adi(bxBTiG0VbVplD}OGD+d8e(16`r9JaQ82Ggzps@)hqs1IHJ5RH(7w|(_cY6mfayMldKMg#-|YmF*zrgJhxS|Hdo|JMCe6{N`8SWFY5vFP5Obuuiv_b~CX1QVl)e&IHmC5Yn47%Hi&NUu1&j+%V6CgwlRx-oKu`@_kvH<fM5$w~2^(Q=)hah@q>jy@q8ZOx&R;@fmOAR<^||wUbngM&>5zTAsWcsX-6O2;a_<F?DB;Moz6l1AKj%2|-jT(7EpC16T051lTrl3^mqP{O1g=n9k#Hadn>sF*W})YY0kJx;ht;<>w1=a`A-MNJMOu`DhyQ+eWN8s4WLiq)0VUtJB$(~<CYb&W7BB}o`fZZEBX5_IXPHZXw$T2>%3ZdWq`deNY-syq(gO|N$=W~cMmV6W{=f@lc?<ThAcLh!!u@Fdx!oXZCRl(q`MbXzbpTwhlF@ZAS}fj3(%6(QFdvxRA$h(x>J_B?RN_D)VNYzwPOhOIw7ySLkO;bW+~qs7unPYMhh;1S-Ou@yz}?MXjy}nc@&L7cL7*Gc=HO<^hv#Vsv_I_P9C(&8x;me~3f<#rV;(={+zCVZ!+Wj8r_&|s^;Oe9^q#0(-<`YF8p2`&|8M64*WtW+)k9^xd4{)r)Dk2Z+~CK)<Yfswmiu9{JJ_7dQq{!VwI8;d&Jdl|?P1Ok<`k+m!u#qfhIJ$~JN1iI+?&9*Vc3lgV3+3|m5!dzkl56>)1cG2uY}G@MagLQ5YIkA5e!dQvdE&JACbo`?j!6EGqSSK0jWciA9eodG~vf&3dfTxlJ5HWcAfSMA5FeC<!xxhIkt{^Vo|T#?&;m>CzbpAYP9%@i-GFa&a?i0!ByR_@ih;8H+%v~)JB~FMI=;Jb8{M9B7JxOCwZU|Sy$(s&%JeG|LheA=ePI!s!-jA);p6@twn{GPr6#!{blPyeJNjg@6dzSOW?&{w0ixSLXA;#hRm*_rL1I$RHa0U3HdB8CK9=}+i)FiF6jox2M6LQ_m03tzb%)sXl2A@G-;sZSD8NO-DhxolW}%^HP4lkh`i6G6P-K88~JvP@RAejd_D}yo6HG*!@?V6<ceL$Q>JO#6v^EdqZ2Y(T`;nex#)IA{~k-pkPkB{{e^4?Thy*<z3GTZb#Il3i)PLHLSkv5);1!^019okgi{e$rItL^p2CY9{i0;FN#w?AB2+$i@M$i!+l1EQ$F4<u374{PdpG%Vxaq}P;OR2+nZbMqcthQ+PG~scly9|MJZifJzlMrlo$R+h+F-UF<`fNJ*CO#N{#Kb>03L9NTD&B}9Wr;GCBazYb$i$>?h8z9{EA!ej?=<1ZWxm2W7cW;>l{j0t~@`zZ_ts^`;~0LSv7KN>(2BY-(hZ2@jpY5B_N`92P13t8d=Q=HJeUzmjRGiepZ&h>E{RI-O85JTibckD~!O0$xgh$&dKvtFQ=~=vSQ{mb~|-!bMhNLu<v$ae<q~tHWZH_2Lf8`M~^p*wkIt^V1`BIjSgn5!Fe^W9<o)XfxreBD2oWv-<8`jMf;<VS>Fw0(m_6MXJ@#xDc{q7M9-tCsMpGg&6cmK#VwBET~c=P?ixr6S+G>S(eyzB2xFZ%!C|YuS!ll0^0j!R6DlF(*w>STZqyx9^B?SwIyRSEx^bS2=HAnMzF#%2j2G4S8x&xSCkr7**-jl65hol@_yRyW_kDbN;0JoRLz;}lKnAg$w-doC3VErXim6ceFI`9H^r)(jGsnCg$7;IT(BHa&tOJqhQ*0=WMg88FN^>xUJE&X96clM~DowaKT;DI*B}Up!zUeLrjXnbWY~&HtT~T)Wm^frZuBJwP6l^8_j;xOzW!}Y1Y<Ar3S2>qgsiUgfo6Bla7w6E<X#hVWGBkHkwYwPmRqm6#*>l4}Y2mF+H>TC|S*?AlEU|T>pS|0Fm6QILkoWg|i0yp$H|$5R;U23A&bi9E>z-`X^fuwtqXzi^@XAzCgRIzobc#`SW+UT|XGrkGj2f5M{jSf;t8nlhh7}g>#xpJe&gt&CwwD)v?KiKvK>3*336Lqc^1o*-NbB@#S()B?aZN_Xw+AxsKPfq4Z=WACsYvJ6lG(Z4ZPCAXKDfI*zfuFFcVYC$Twqoaq$>5msD3NJq}Cgn)e+PP&xgK0b$o60Wa<!zp*I!{`5t^Mu|@I=&#O(iEBN*tF&>+(U{@Qx74;mhT2zWDUvuc`wdr4UE&#*vJhraI5wkn;;PU?1pazu5Pe5IbASqZ~U%ksy6RHpibRWPDxwdJc`r6srIx_d?;Y?m^u-Z0LxMI%~YPDAVt}=<p+fGLri^F>Iz7_5Hg83z#0lQ48y4}?<uw1_Fa?~j383R5%af1{r{+U!!KU`PNK?y8NKjJuv*#FpU7?=Fv(A}n=nzyLfsg?ASF_ZSC)r)aCbIgjNym=T-|3@EXHb@rvWT9D34VI)KK)-idK+UglC3{XL-bAZ|qo1`=QCt#9>K4p(F729w-^@LUX|-nE39DiH{v>r@>!N$|#O5A9tzU=5D}R>$ysMMym=xQdRd-+8ynD0F$7gPgho=(-zg9ArYq#5<6`cPZ<%IW%!Tvlcc8}(E`uJoWFNBZjcSU^XUKU=R`}iHbf0J=Dwx-oxqwk0u7<2|_XLY%k%Abh-$Td2tmkABk8)^(V>R}@uC-jP%w&_Mh{7DV{$4A%6vtxgVYMtAb;VX@=yYy{U#KA!~l~vc-s@fa}tFln81Mbxag>!eQi1D2O`&Es3uY-2=ld&5?)jHdgEmZN<b$NEq!X`L7jk(X$i#vtBSnqB0ivm|oq#QtgJ0qdltiIswXp&vh>Xb1989J2fO~abMr>g05gU{!*+^okil$PXJyPF)^T6AYvjmi1|$hK2}P5V1WQk!}Spi0oB*DBHKAnPTmV9AwO?;a7~2mo4qlae%ZS?%t>ICYA8X^R`)uWL^`TjS!l`QFc8JkVwcoE<x=TuX@Mv=K4{Dou;0h4(U+9ntHyFjm$GH<QDc=Jf^IIt^M^^<Q7E?T!gMMOO`RF>Ei;QOB#S<<sb&U!)tMN@MDcG|%hapX5|y^tc$|lrMy*^4Ykt*7jJWg+jC?9J|2t#3R9^RW>r4=<l+I%0sv1`g2ep`8>V<Ug_{Ny<(Dc4zzF4*f%#pFmv1e3)|F?BbhFz0=aF8eX?ELKaqI44o{#+@Q>lEYu?|H(A_{$OXSn<huj_AtY)zg!(L+M&*jkRCUHN<7b%R5*FkqFbm;XdBl2czs|Um`@TbXp8feP+q^0h!N@&cA4I2FcS~fU#b3;3@Qv}z-&uX{c8gPD?&WDe|eMu1eCf<e3T24jzXSClf3<0cQN$q9A4p*!%kV<39b5^{jzr<;;i}nS)yn4et){`mTx_9r-Z2|&M@y9h61+=M7AN_Y)+0L|+dMsPZ${#V~Kn&=g9hN+%xtIyiSTgl@U;WcK&FjF3r*UMU9i3e8+}#D!tCAd!UY2T0VqY<L^%gVP-EKs)5<7nLK<x7Udq})g7u=5eCVSAhlTSRH8Zq#n3&b268gLB?p?<q&FW?H-j*V%%G1se~4h~mFfX=jVE+@~zeOH22pclOuko2ROvsNanrnU^+MLE=&b`mxY(ZIP@>aAAe_8aim(*^!DkKb_0V1)TwdVdK<MzTY(d`uPx^84AY&5nB@(c7bw5Di$UT^q-o7Wcko{k;1L2e4*<UN<viw;L^1>S;Q9kcMas4^{ah)~3>Z|GO?=WL#WKHJW;jDDN4K(*{!-PH(l{A9a7+$X;fAKE1TfYg%7Vz0F{G>yB?u2H+qCYb~;b;8pL&{cD0-TVLL^5HV&BYP|O8lou8doZ>ei*vl0cAlAR!kWPK~ar#5DE$G60RFgGia8jqEY@pQs-~tWG8EFD%Qq^t|15jRhJCd*>_AUcrJLap4R^N?A6~55#wahh_isjDFp<;RabM>VOn96^KY~~*CEt{_-=(u?NiW5@&`>3#fs;Ux##AO?2*>CsUaz}KN7ePps+Et6jTK%kjYyRs1vRsfi=kcKQ?c-^(S%29N4iX;5<Pumc4+=Zq%sYztIYFI-bbC+Vi?)|}W_$eBD@T!-J+|+->e9VbRKH;V-D{=hXl*@_x5`Z1H0x7Q)x2Mt)AaJpngJ3Q`$k$B(8v_%c8rY@E!)TMW16YiHrdr(8}6cmyvL{Kpqd74raW^nkG-!QrsXF7fhz3jXg_&21ou|?%z8=QeLl=&g04@Sq&cW};7NCY82bb97BSU(7=w8`yr`vrfVeNwxDG=E@;0bwcos8rO6whNwtvfIwSqsDeD9t+@R-+Vw8XE}LG0vX;+xL#4lu$iJvFkwcuD_J>E?~iovK=|Ryr|QdnfzbkPBwK$Y?A7p0HcI&gPHBp{cUVc+TF4Dt}rx<hJ2bvc04vT=tEe2)nBj#|@+jI;EJ}MBevUk#$SHzkXyXg|PisTvhh2>Y5#g$p&;~w)>-+Li4dEN}XjLe7BA+{jEwZNBXRTu3eQY&=H@T$~tr!-RQ*PuX<Ix9I6lSh`u(qUe=rA`^R5-z@8b7;m0M71;3cTZwsO#ZTImKS&h)=gB(nj{7&Aiv)R_&(zZV>s!vVh-n+QGt{8;CiNQu?!Bl6*FE70%-BM85SX*C4qO;tPyROZt`>uFM{^|z@CWrRv%tj-!DX}>*-kw|Sa=eME_08xVY^$fvVNxXMEbJ2QBuI|9|HSM!XGMPjw;zdzBi3jvK5t4{aA-gOkk{ToYg~vGT$@id0p(!qyOMWH79U%ECa==_Kl}sETh}oR?wk|kk*UN6<_5<2PFKWwlBW2g38}b7KPFIvb01Rn11}#eoL)Csc&OEW=9F7JPF8Wbb$NC5g`|!((uU7#b9vm|8_&ndvyQdUxH%_%qRdylXRdAMO!#(OU`UxhZB=%zt7x)#p)JF_c@@@+kGy2yAY@GM)n=>Fj|%({&w=Kog(S$J@*}dHdcE|;_s_cBc{JZFRkU+Fip9N#kbXL=ZxQV`E+_c?IX}L99n}Qbbh|z206yJjp|9r-_+*j}b$l*K8If<_G02r4zQA8U=jkAJX#T+n4C0?C!-uI=e;u3mIn&{Kt1d8bevMh_74P;a=Ek#cQ@Rnu&pv0Z+A&riGkf<Q6)e%RKPOSiXn41nz{!64C95yfJ*seH$8Zm+M<ZUM7+!SHy*QyZpZjI4VT0)5;(9ui)>9M3w1GV<y1OgcZ3Xs@IN#0}JP!A)*OR6%d)9tAx~3%x(s(Tz53M-Ntj8FU+dau?ykZ}leOnxbH>mx>i1Xp(Q4?Qr9f;+)v)lKk%~w{5%h7uqH9ifdrK8)i{B0{~K$X=;+;{8wN*fu!-6_9l8}JKpNIA0_$SG>3di3Y+S}-HV)YiD(&lmK!AH1RbI9x8k3AUa+AL}eQ?91=`Jiy!iiT#$WI?@SkVl|z3y;`$|c2Kj`T$Llscb0R4|D&JXuqak|LsO+_T)!@+4k2u?W@A197Vu#+s?h*9Gn43vG^-X1-sE{@KRrFuPjk??T*_{Z@l{V<*!XckF0y9pg%ue7xGX&W+Fl(7&wW(2vKQT_PTCoqmtyzF*GT_Ng5u6<PZW1Xne?ei%>k00CNE_QP(8kZPsx+mc7*vBuI*UZkWNS$9mme-HvUzbc=B+-(Ob%F`mv1q653`202^4NN2zz3lnAW;(nGsa?PL=L?bYvPQc<wwre0hqP3d=!Z-ZFLLk{j(%N5)fC2)58)HCV2?)}M4PsJ}99VoAB=SJ<spoc3x5j|YdjT=@Fk(0c_>{KgXzB<3jyu~`{!xjhNper2VK{ylS?qs%=zR53^e%<r6Lay8nH@HgG0%51>a6T>99abUU$vsiCE3mzVU6yusVwDkTh&*&&EmJl6&wOBMw@dZv9QVcnNl;H*{R+x3C7Ru{-Nf9;wap@oy?fX{Hn7(PdZYBn`+^%p^=o<doU?ANKXaDnBSk-^ll`35U&KAKC)LYo-ldEQ**ehi4ZUle(PGEW=Z{oHdH_wwnH`)qb)KK<;59V2Q31YtkJW?f#O3yw@A~oi!uT%QKJH~|+Z7f-835EVo>vU4;h&}?`3?E=EgUzd)fdJ2W%p6tsR#-@wFMkpdYjGMId?U}%6Er}s%~qbbnIi%=rOWW2vwX+zdVH6Zc&NSS%n!ekOdKV`dnXNsP^J-^`GoI0$|J4tACgbXlRduv|$+dtt6Cgt9tESbi1<g<<6`Jo!`4d;~bw&q&;Y*IXmZUKA$ajU&o~PXG~MZ?wrI(I6a(P1e&_l3FMRy?(6a#UAKj6dF(V6OSodYt6KN=X#31-^`yu4>{Gc+cv(z)#Dgg5BQ$!CZtJ+4oLXH7+6ZMa1<iNTIgP58dk#k1!y*QFQ&}K?Q_A$ANc?l**)pnt=^pI*P7@Js{X&l!c-(6}ex1AAN5;VnS3Ac|?c+~XCwohk7ZtF@L}jw=dP?TXAAcs(pKFe+ZiAlrQtMLlh*$5-FH0~^E8%jBx2n`?@;D>&l`m|1!mTyQFvA2g`*MF=Pv+n9r|EVCo{ZK_VWf*74Vg|IT>WJ8CJXEJ3PxMr1&!tFFB&rS{J|_5<D*)n_xmj_O{uZqw8iAI_VRM*QRXHc*B<KY?=o+DgLC67NF&Tj%gwkbHRh%-v&plI&3utL;B5PWf*>Q!Hl?+X>Qgxl;JUYQ{&gR1mauVlB4|<|T(#-f)eQ&U_x8=cLtYP(hCI4nk=7;^`1&Q^1=LPw-+{I6YFcl7dDP>@E&m9$?PFUREzi@i)W5e;dF`L9&vA9MI^$04xkusg;tw=z7gLW4FWO@3DK>p){=D-IR$7NG6WytfUfE!jQ~}n9F?pK}mlN+ZhpWIQ9#5cd&;=hH|Gw&nLMSbg1V*E+xVTxa00GMt9r8lkd-b~SJN@2(pT`NcPU82<Tqb!}jg({dyHJ=g>ty8N+571OFmaP|s;emmtj}xSc%{v~aS3{Y@Y9yd+2P5ZR&wu^IPXMH-alQvqWpd_2?R#TUsxb5SzZ5_KklVxgBPkkd+3)CorBCA?$3*TC}>u_tTXFx6`HlzFb8G@a#OfNk!_1n6g8+aKYtF@Cv(NC_|n*z9b)jR0%lr5_I3kT<Twej^wWMoH#Z^QJoq#&YHg^5)dOxS+Zmc|La;MYMT~C}7mqa9;a|PJ<*OUN@f)&$`#!$Xc16hMA+>qajmHT;9ea)49DTky3FGBP?a`vOvvE01mmy!d!zrKinnmU7JlsulAM0r67fJD;g2tHm`e>U{wWtpTT|-K<p0d|_H~7`N6v@Vt-UIw2?(*E^{HL6&+6}(U_3<l`-JK7qv$jsdw++$xwYd<na_xd=E>-M@?)ZGS;7?g0x@&AMF)UwMj|M(xC?8%s(m(G<CznQ>AmJl!=<7CmNk+2$Dx-{;so`Yggl3uPCwMm%`p-7jZ9&l+J3SwzJ3{wsfm)oz5C3T*!KBiw_eT}DXp|kvW=`J^H=SmzDrK~0S2%|_z_{9z<tiRfjo*CDALbLS1|u~#c;BMUP!(LTS9FEtewaUoWHH(pp3=!}MX%Tm{2uFbu%6UkhfI@wO0BEUQM&TaLx%gb-qqPFJo3?EO`4Km#txmpU$uh|PQ+m9!vom?*33DF!+ZM+E?|yXP!1X==?!7l#6yEZ0NREp`$YTZV2*WfcWYno@7m+Dd+_TQZ#!;$tDo13TWzmJLz)wlKRT;oYfY7JOWqtU94>!qCpYBEOUt^+X}!@SJ4*n0T)QRwT!<rP-$%@24RFRuZy58KtxN^>o%%nLJ2qG6vqx0+MPBlkX->9lopUiw5bXsJ<xY*(yUfS?`4k$mU*L40q4Ck-Qs>Y6vHh(D6e{UWq%U^yVL+cWVsfyj){7~+X*JJWoS)a|j}X1ie4xw^g?@{d1qpzk%Oj}3*|)mC#y!>A_~AkQ_>~ARfs*L-!gK=y=ZPE*h>Az>=IPq&S-*W8IoiNa2d8xXVOsQP;Xxm36E1B?HRktySLt%Y{<V>FL@Ex?2|~ReZ*JhoDc_GOa89(y9ZjdeyK=|2KT@o><3@+Ku0KpaI$$t+>8qd4ZRi^_RL$C5zHfJL^f6|#zAk<nO{Fl$_^kB+TIQs{N6K$Kz&;(%TYBgZ=}lnRJv*PbPEa&(nH}`z<%%tVmK#w6XnXt3$G+dBxMnm4@>8$rn!{t3pIV;~p>A|k*;A*sEXI02k~Y6^QNu+j<OkPksj6Civ~PE(vi7*2S94?j(p~<gxc*=|!KyaEzzf^b`Acd0jRM=e9=hi<GouuI2PvoYP-@hCzHtB(_M_R;D&g5XGO;@~jD1`JX&@d#(&(EBT4!Fc{fnyG_<-wm`v!cUm1`9_;gyfbR^B(Qg3hlMeTo@EubH*ROjP|7OK+?WXWr=Wba&t6@|%QP;=8<uFX}TihVr*(lm6?Kc-^h~(r!QSGj+vQ#pqi%vQ?rw6>&fFb?H@=X^@`oUQ)sPdwid--so2KiphD>uut$MD3##+2{9U8*I4Lp<<FmXS1WA3>uJ}`c6TbKohxf+2HEzf;5&@-BOJoyjcc%==10f3GU|K%i@Vn$?f8vr=AM_=kIA8*TJ-DKmP}Ayx#mX=>gt?p6;GjM>eFRlZb3m##tUG&#9x!+RjAa*?{lNK>jaO{6g-Db{%Yp`j?Y;vx<>T+SZn$+?3iiZYF(%45=rK2jojOp+TqwHVW<uF%{3a&y+7$43MtVUqCZ8FXo#ureO9ecUohQiFHXBkdS#lkLwSw|T;(%EZusIgZQXIQSGf^p-b(E8N!YniH(D*S>uqqHl`pO?e|^9wqWXRxrIthA^!m%9&a-s6YsB_46g1U_f_-<x-_1fGn;{aa^@YMu8{InP&D}tK&nNzyIQ2rvD0~GNY$0dV*68nQ@E5N)uR4P)*IR*+s4GWU&s9`iO%nm#iFs2%-(yP$<t;f5btkS}dD4oK9d-m^Lw<Dhl&iDpcfP*1M|Tek3bq?s*YSWm!Bf3+F-a{uLK(l!y_XX=XT;vJAP?jFe};JOiPjM1)Mldyg2OdDeHxdz&9%*?f8UJ+y+QH+adfWD=Q42|el13sV$7%{O%Y1zAf!YON+LQaQgVLn_wRo{*3h=QpYL^zcDH((&9DA7nXq+L=vVF0W@hCZssD)JO;Ja~A@i2b*zR46Dkl7G^q(GEb*3F4dq)$Bq;8Y+*~1$M`6_1PtdN-DN<t~;uVH#+wGT!V{PTjoRwcpSe$C)gLx&INR`EX9dt~XWw(;$@J{zLj+63;PF-;9V{nS1Hk^j!`_jQz@yuNMH_)s`UYoV#kH)h9PFYRuej#pNH(b$+$p{x;Uw^pA4tya95guY3*&th8r++wij(L!<~>2ry*)dz=H2ldgt9{jlJCXN%_i=Z_aya0ORWZ4mC5A6$`t6etFIA*p2CsK^`MpNLaa?0WGy43L9sXY#7O4H1GkB7U<!K}dhDMhc&y%`+p{Q&8%q~bGwypFG!>!X)<qX6N7uHM!|4DVRK&=B^a#QfgwFFuvyYxA6_ZF-~32(}Y)SnZV_Wpt+)sJjX-8~~X8UsX8WFnCGyGGo6LhJ(Se8_yEDu}vW4+m{YLaed+ArMB?~gqv3m=iq?8<Gp*`&eHX46G{Mo7;dsLob%7!XOw4Ixif;@`J~bMo=W%DI+A>OF}n|@d>>#BLt^OuW~=o!_5aH9x{D(*w;>F(BXF<d_Q2lFeATd$@{v(o$ERQGadTg{qz+Zxj<iGL)jce&gj-vqbw8{Kx68_?v-RqXd57pr^32{Sw{DMTb$c_uOa}DsSQW+f8Cc*TX<LLT&josTbz$lDNIuXTAxQVY2`H1#`*|yyw0PznaMg+Q?ADOQ+pzO^0%ADWyE4C2BR2s>KDd-rXOn01x5zjo3sa?L98w7>bajVQZkS|m=@4NiycIiwwo%8XW_7t!<uNH=%Ze%$Dsxx5Gv$R1-gSZh@a*TB27oJEJ?<;-R&WJ3sZ3iVYp+$E&c%XtOSC)x<68np;QYC-dyw{6Gz-0slSUjUR&RGb2k$%mT0FZ=*X(8e)A{*~ZU?)%Adp(SAZA(>e=->*G<PEu(5)j>cL8~H`t7@weNo)Ax(>)}<@hXuNN8UDM$xYAn#njnDd)XEwg!@X_ngY7j%tj^JZ+WI5Y_;zpJr>r^!r0Co~ozAJl@rB*8)CYD5LU6Xu(iaX9VfDI*ntPZ_Aim1kD@TIqlAKw7!D8$<nAh(t?+**CzHoN$y{yRR@E!scd{m<F~UeIl+0t+T0&GOc~z-cF%#cEf_AfL8}5kb{7w&%=`IK!HYxkT2CrMMGKGTo<HP60GH$8pk3*S<4c${8%r*oqq8?L8Gb~3mw?)1-p1?R$)IL)WD_OB{cKChH%)r(T8G}3CP0_Ffun3|zw3LhpgG}L@2pw=Tmp<OoYt$!2@)@J86L{lTOZRI)LXHpsdU{+#|XgRQX@Se3O7*;Py|wxcnkgsB7*gtVEgSq8a?@s&l<s~Lg8m2C|#~iAsr9U1_P_`XSldX{(57;V5XmDBQM(@0_xw68kKbdRDtX@*ju1n-B?{S^`)Is7It`$E1in<ss)$H0jL~#>R3M5N!rin^=);vqiZ9VPVW5%<Ug?=9JMBs#(_Wf=1ZHiSxIx0C*N+=vD{kcRl!G(TWgPf8Bu_&CXw|ea>#qurMtaXhS`$%HW^zUJh@K=8Qch5=p=CGyki5x?-eBTRL<UYUz^U;V|y&LXF*9O1Ei2Sb>~e__V+T_zoC|ljj262FRP9#ykE9UoC3$|J<>Vm1=}~XXn^!$JdG_!yy_a)Pa!qhfrk{a$N3;~U#r9MaNh{Z@a2Tk;M$nvyO({`@@rL>hP1~w`yY+l?7UU`OleDe7f?e8@0AGfs3|MzQb$RVJv98j`BD1Baapr0HF!N-@Q4ncNwa+~2N~jT85AE(2Z-EwJ;tZsU@*A~nloicf+U+8X;G*9tuC_A?3}6O6R4u?s<o_HuE(p(XXF;WBQ$-jo6?b-sK3jj`gCCVlRj6HEmXa%u8<2zcw-jT?*5@rUe}AwQ<n1MfNIWn*uk*EX>Sv@FH89nJ$+Vg5a<}oQ&pp-=37A*=_2oGPy4E-s(3)~zZoVTrC~kQ8)KtOpl`huEf8k)M(Ia6YitM^-nXfRt<5oh)=`wNR7t06aQ>_mQkoIjNk;0P^F}{MdF9HJ6qCA!R$5r&dLNdnjASSL6*rS*eAJ3z=1pC@2lPGEd9#(tX8N|~9h;L{n3r{GJ~%-a=N=LR3!P{>?s)_)-ZVm?Tr()=@*m%3*zW+K72p+1BoU682@SQW?UHE5XQfg<8EQ+C=shxxH<gZ41Uc2dPRG!u)3M@qFccQ&?jIlsc#S`e0Y(OI^?r>AhBsmL{w&zVg}m5wz;z;Y#5z?d`8}avxa>W)LB#P7%<KN%bu_oaU?=-8&yo*kk^)u})gLua=%!^S&zt1+yv@$VISm@@ozt$8t>*wV(bB>AD?Gw&D-j+|&bGSS$F4G)QfL`Gj?ghHQrvb^x4C={Zb&VJ$J^cg`T!%L;h#2)5~ojWblC=Miyzlt7W`+Aih;DBhfXUB$)R@k$BzYm0tT0QM+{fd;A<_(&UR2B@mrHu3sD1^8NQu9-u>&V4+qD`ILdr<4wSxQy7jthv7J)F%ztNOv=D#sNq2Pq6RIwxT8;74+{}L0p2x1i?#my|J~oA?EtsvcG4J-U^_(h8E8dYN<%{LxfkS@Ke6B^I>xI@`+CaC7K<m<Zkq_tXQ{&bp&Z$c5RI`)CPSdK0`Qs6oC%tfnnOxThS_xA?-yq^vU7zD@H>2&(pgzXSl={>gr?@YSFYLc9RJw8U(yme1wA|vrZ#UgumxR~*{R>)5&id#@w1jRO{66h>r2Vw)eiFJzr-qJBs}}*(vo3sqYZ&bmH6IHc|F_WQAOAodEz0Hyo;$tP`Y0-9I)LPAM|E<vd+d8s5uSA88_m|liTNe(){ZTPGoje+<1N#nnR_-!1=uKRjues8AJjP1jPj!{O&O9In8XOube-$<IEIqPRs0MBWogyQ{(%@*9(DwDFD28XQn0@SR}_gf*H^?}%{((ow(houI)8J&UpfAE^yu~B>}x{2rmz9Ao<c6i=iX;`yvHs~jFGGaY!vnfArj44WLzzbZFTy{G;dnDH1T%SIWzlOI>_^C$zfaBub|W@k-n?{7o+8f0^;#x2w1=6zm?G*ayfRyxwlD!7Q{_)_vUP_LzAP728X=&9$MD-B~0jHZSt{S(fjE(PyGw%O1{ylTY_87gNq+e&)aJ6h4bCJKxk}eNoCu*hAlce?SF%#)ZDkl{DNLakjV7`$$U+F+V`;S&q<pr4mFt=T807SeQR97EIq$gG=U1Ns5(aFv(=bnd<34@1@~FFE&7(9M>cVsD*Xa!-c%JE9P9gYyXfxm-3Bn+&vJM*Ff^_E5rx*M8li~QSKQ4Cp?}4hCjNJm99W$W!t#XNGNsfIY+{|zL>GU1GBoSc#|#mi_be4rcRKKIxB2I`_%fEd#+LRk!9fMVnNDwIe!wKo>k)Fqo7>nD{l3Ou>qM*Imvj2h4i+o8eSkt>C8mdTKW}xa*V4Ri+W@I9y}A8ysaWEcrW!W-P>ZhM8vO9J!*)9GewkLe@{7@H(aVpBQv1`9bTbi3a)gz<BpdXz46T(v`MywC-dC$~jT=ply<q}fK3lKCgPm@wb|Dx-2G=UY!+~||0}X9f64U<piR`DB7gI8U?IgY8TUnZ~dXlZxQY3kd8#I&te0S({RP2dPXU53V9?lU!RQ?wC+?D%XN&QV_%}o$AZD>g6`fb!VRH^C9HH7ddw*p98RE*6_*Q?*+4(d`uus35UF)}F8!ufJj(F5z1qmsie4{7&Ge3IPt7gb_2T3Pf247`!n1Y++)ym|4$FP@vJK>!9kn{swb_^jUGsB&8BxY5%<ui6^n&dqS*S0t_(njnYq!fW3SN0r#;een8{!O*LPc8I(q=Z3dI<Fb<fu4!Wz=0YG2IAHq*GNS^nh8BN9`ZqYOZDMGRX7WqI(>Qp1Dxb0OG~;f+=xy{3pKp@;u+rD<J5iXk`r3VNTj9NbH_c{kek>&=hlJPj*7EKzKwrbZ(Rkg|jA>(6atyK4H^WV3a-)FPd-Tp)fmIQj6WOn)@yeg~HPSiEu!`3NzVcR_m?M90OV3Y#fmYDu&16NnTA6=X;<VS06NkJE#4nIu5otJArZXyWqoLC^Y_}JR%4Rf?i?NV6ZRUt7!|G;PE~4IFF14g^@*7yOrQg8{HTz-P+!5EeAkXXJ3cd{y;#Y-}b*JAQ?7xyd3D47kF+N-)Ls3<47$M?uJ(y;-r&kau%l&26PLkK=vm7r@3+OwV2gP;Yc}z}?pv^bNlZrwAD9VfI#_KlQd)SiPO3pN_EHb6?DWtO_5zEnYLr(L?qW<j5UWa_h(YE(wsp1bM>MJI5gJE&~j@5CSE8~EFeiSS0igSXan14rZKj0`>O!NKdj^3NNT4`X9^0d?2x3asR%wV<P6!%y$_pdbFMPaI@m2BRwz5g7Byc^474c-COb(tuK;T~nqGpZ-;)w-%H)^oJjMAxY^^j_QZ<~FLjmCC94GM<heJXpD})(;tRaT&yU_rO0V$s?E(`9kzZ{`zV;cJ~%^wWOUBNhb9X@yGLfe%)xUT4!S2bQ`#mKHT<bhA$e#!!*~(W&0>YI?rZVt8Wo=OfuK?GtjDT!A2QwY)@>+5AU-l2Y;zuzjx3r(awg+^mBaTPDm7eEH&m!iaIZc+l#g@%-wQHOpgofQ(MpH2Y02rmkHISwIL3<R-c$JMd9>D+v69VbPoGACR@4wEo<BrMC=?Xmd&n~Y8%|=upj*xK5WARE;+QGxRQ4gxPH{siLO09*Bbd+z{<0OG!mg+SDJ1DSUZG8pZpcRDZR60iir(ume5SNOeX7<C<A+9I{l4?KdxCzSngH^em#Ify7d12;gmakEjdb^a#8&Yyj#hnym%wbdF|>{v4pblX#4@hkD`(jq>QL8`W7+obK{AmQJWJ$CSUmtRG_W*nMb6c+q0TZi^WI{(SyB{cJTJ#1bdQw>xq8n-(KnEx98=5d6p`##`4(kO%oU$@Q2qLR$Btinf&#mUJe=Brrlt>k4+nAQ2P2J%hG>F6S`oOFPPudcE^e4zh_mf(R`V=y-Y@z=q3S*ZI7z1Mis%vF8AFuSoc{UG-h`f>!|YU(z;(X?EI9;@%@o?(^vwRm*TSg<ah+o_ustWF=KrFYy7I&n%dYYRO7$7b`ZZu!-Nzz?;N`pjK4AccJI;E_3ctAUhLI7iR~9&Iq%;eA=bK@`4S~C(mJj1_)UYg#RL5K!E}`NvoI*)F!HJe2yfjK8zJ<(IS*i0UYWP1VPmpE79c;{tStx>&ud@*J>e<{+$<%??h3tBJ@;;|ThfdUh?8!~%@0vWHc!E1DnZ?qC}L1o`-9yrThv}Z_4Mb{{l<DVBLiUY6coqDw@M2J?xno-w8c#O%IuQl)PuLJgpyaX;umRLexKvs*-@NiQxlQDuEe-3M)F;6dkeeyHP!`v=Tmi<_2zw$zPO9smAtKt=c94cgLMCVZ+cj%3`VzGWs@7dNi2z*E1txEX>z7i(uMV4)!OtIgfS-N$t?aWG7DXNBxFOpE%6V|`!it`#QWX7u;!#x_m}&a>y>*orBc(gu0fk@DENFd9ctPD@5b$SZ+VoLQJdXlW_J-Z>RF3__L%|c-K)r#=7_((I4y4NF@PA7+FhF%E`YrjU!8zA-5;#07iTO~<=Ga4_$T9x7f?$8R>yx){S1bidUeM-%j#OeJM}qAv|*h~q~PyT6g-1hI{qH9j^5cClR^-VtNnGvIcMzvwzg!uoUZ8R`f?Cy<K~-3TQPm>{lo^3dh;q!QzYcyC(;DWDi8;MYCJ0G4%fe;hpXIwo2yHwn3D;raiH{~*m-|=x@RmA;h%kCxQDO;_4>Du?CM8Vl<h%PSU~v7fVPdJ92M;T@WkuA!G?eR=%+3sdA-S9!w-ZMou7~;e&bJ=jo|?orWJ}=t=!pl)1_YQhK{l3zAQ7K8FkZYJqUx#UkbjD{rA8>hX)-y+T`J0Quk374Udb<TS;%YYjWWHdOPwaaavNZ$^h!7t50~#Vo#oNk9*TW;^Dfn1H}G1(C;s52}&Drj2u2)_h_c&18W*n0Xk1X$2Z;yaa@#|J=6hu1ly<D@3|2ya>encqRy@yKU~50g<czkSW)bd?4qdN?AWf!4q(?eU63(_v{OWY4Ai~rk3(kN$*FzVG*+>hoT<Q$51u|Uy7c))Liib4_5UbhwxVjY(#SS#Q6QLO>ou)Sn9f@aTyu`n_}+XrstC<Dz}X6;p(LqSnEsAZw;Qah!@*xY5RI$_HP7vAyg5IL6)e9BF+Wt=I6Zxm?gcHrCl`_Ko1w`}F6Lj<2BJ@9*~z-t`g#3w*a)2-{x~5ezSN0K?S6p2NE3X{K25XA&v|1xB=j{gMQ59dH2yH{ehi{0ew#X8eLM)g^VL@LR?)K==yb0}#rqgs8hb61eQ^~J!#_vUCuJ_G?6TWKR)fp=4D>g*=5rFe#I;>MXS?4Mh>_i%QP&AH;8>$0k#(s1>(61DP{~;TwSQkOu!H(mtbLdT=ytv)z1sf8bLw|1!pTu|chUhMyiSMdbruXf1R8v+h}5q}m|(nxsj#V04fxK5>)w!GHao&9Wr82(hn(2C18_L;YFa$WnzyPMN9i|DzFPH$L&!meoiG~K&K~7e)T=$*17t>lebo=`4Ij;7Y6=HDlNbixw6EYIfkzFnQ+rv$5E%{lI?KiBSWsY&bPtzeufA7@1JSNu8{}m1saRn+?>iiOLtfj;f##<q;tZkkX<oZ@_Q7j3yofyZLZ8_BtPv2;%8zNB%<Xv;ciabHJr>8}t0S)iGYE2}K07~?!DkJ623Ov#D|`YLT}B(Ph2LhZZ4yY<KRTnGJ~u9>a)5)#btiuI4tHg$bYcCNOWXN|fnDR_L+v9LgUhoNBQ2U<aV?=Wvlltnyfjs}DwAv1E78+KdO*ltb|&8w#S?rUGUb`9G4rhK*7@%+TcJ`?JAC(@>jk^6N693zWz@FYDGEE*5g%5Dl{(ICrRu{MsRfpuARUBulVCr*My~pozLqggih>-rxYm75c8qGh-#ImIt1*gi?k|1ysz?#2m!J2tJFRVIFa5YBiX$4We|9OhlAameMF5kD#3mXbL!aZuBN!*u`9{BVHQUMEO@5(g*4%By9X({*(j`5j+>vfJE>)HpbZvh9@)LV<v76K1Y*vrjkKaRd3?h?%;gtc{6U@-lLppAet(u_i*NwmosL61@*6#4bZrlS)69b%i2=5rAt16a`qqiz|Q1a0dXNUt~y(3u5Hc301@=5fK-BtB4+O=l5moYVFVwBEX+et$6v+v+#cyH3iA^L0kwCzS~eaM11xF~zVx#`XhefU8TZ*Qh6i`kVq{R#bIGyO~Jfb3uBeSUzOd-a&tWH}Ix+Y*u22E=uTNI&-K%jdW3f-fm!UU6PDglfxccjzuR@0e&~zv{%BD%E2%&Be;@^aDh!qhBn<*W(+eE*2G`&%r@l4jtY&4rC#@Km1Cvx?VfUrNLsztrGWd#Go_kcA#;aCyW|N=(oD5@0Fxc6qWJ)>`fnyWZn{#(Y-ES&8p`BUk>4<iCMq=hDkIZIHOi2GPTA28~PiOSk-fxvR6KuPKwziX*98R*4-etud&d(l7pU^U8`_xv(~;#GO*YpVjn-PHz_haOt#fVoEH3IYMFcIVHM}<Epa>5XDtY3-W9I)^L6)d=w52e`Km^aS^jaqeIASm|COw%%eJfFahAm(58ff_-H^lG`7)8W+gX0uS6D;kwjE;fm<?BFsq=-SZ)v$75EHG~7u!~o37&Z6xc!q!eov*;w*^0ET4X{cS?<DQsjQ*ly*ci|X#G7lj3v={6&G(fp1fB27;CxJhIi8CxZC=jaq7G=n_S0+MS>i4dY>bEpNdH>7Vv_zun!CVL>P{Il!EVywuW&>_0_5mFv|~o?KhsQX>afW)$!S-ta4-rLv`2*5FuKE*@nFVbMb@m#u<y6+n^tR{(h9|Sbw*$yAW#ah|OLna!Y6Y{r3abR(}9)ucK-2wB6z6Rc<BkORqFg=@i9_82M;D)rt1*V5~U;yi~pw&)QYp(dEOO3d(F%pAo+|lAgMaGH$|Ds*Egl?Yg6TJ5ZUpca1HN&7+Aww<^1(=A#|j^Uh5E8HLKkIPvu~d?)YzPzPsB@9^`jd&4lh*>OsZ2HYe#L`K_!6UGP^z5b+tbmsbss9G;Nb_O7`%-aK#G1N)x6|~H}V{19k*R|-@oA2H&l1<33seaf7;X6M48&k)Tq89FeXy3@|kU9>0db57~=w|u++;=F0flIhHc$AhoVM(sot<CY^xr8ir-%JV-M*j?Bg~Sn9oZ5xDzlH8Y#UH#phNqnZi3vn5;Qki{nq-U23-@&)aQHGZ4`ZsONq%_Bs^YFQympA9OHgMInevOn`?lKij`R0SFt{8@pCc~}Es3eX6idJ48(UtaZ`@!$$|0Fb%)<?VcJMcL0L&qd=uUFJtC+-FY250MnVi1t>wpWYr|atbc$u^=-#Gu+=Wct&Zz7#NmumQ?c9`Cs&MnV2!;-*k>}H?v=COI0hiIS5SCO@s9;WyzO<_K7)D3h2fyFs9RgJjHm1Cx%iz`Il{ZC3osl3K-U?EeE%nMSq9WR#Lr~@q7OZW1sJVtd$&?#&fRPh6@*$q9%&*5>4Hq%D@JxRUQE?XMjr|DPrYjnHdpTlwMu0ehaez#4gbEqLo*?hiVlvm(4C8VaOWu%O+HQe<F>{?z0+k(TtlNEh$@ja{4RM?e(T+bIT0Lk52|BHtO3FZZ=H4?}<`>`@E{fqLfB#4o&D%7TylR<NSDXu$tr`6XLK^zZkpSchF*L(wRaq#!EY1$Yfw>3d-A?Oi}t7NU=<(99D*5pE6-voWv6(wvmuLo1A1j3d$PSzoDC~^{>-ge#Mb{k71#vP8o2GX9KC1<f#y&W0dw<`rf*^5<q(&-Zr-DVqM=+$~`-Nfir(bM8)3R*?3o2&Bce+tPQcdAfubfEWeX-}4Ags0x!obws)4-x9Z#r@p8tpeo~1+$TaK<mnuv+ZNN18@6WSQG&L`PCBL_+TRm7^^^UIsLPk-z~mU01G=H`iS4JxM0P<p*MQ|lodw2LBUW*H}tSjT&=zSBVqrv-^QK41}a<po<f#l`}N=Lmu2XjIE@;Qzotz}re1{ysEW11%kjK=RY%FCU6M6L^2dV-V4m6}k9cOa;TV_YRG>GDST;nxSN$T2(0>i4x#aHgSJ=rvmmR8|jv-S~X6-hk;}3RFA0>Q5?61T%eLmq4jI_E3^MgCj-pcLHnu&WXrx0OfhyG?%)|b5k84E%HL>DmBXNfOYw_~{d>nie*M8N);8jxK-sa!?jqEJqVXk}1>`2O}J9g42gBJN=I%<hPz*)ASFh2|RWeUH<14KqhL>|x#3X!E$NXoUnK0vMHDoXQrDCc3%CqwRTf95&jmH#Qgf)^c_YSZ{foszlRju;;f}@ebK_*6jS%8O3z#xIGw?$kl?VN&P)b-%`ALP{oLH1MII5pP4T9a8CQk+NlP%MrRjtJ?%{avOfv|+dXmL=^O>Yh}q7W#wZ3WmCoMal0*<9;3~hgjoNwZ9f&>fF@V~T^#Is=ep<`U<<NS2K*H#4yT=AJ9V|Q}h93wT)E4+?F28Pz${jw0&wF!?N$=HG9JO9^d|2)=>&li1AAUMllx1;sm+k1$kh*<Nm4nHRyD#}(eLYw1ruRXV3XH?M$IMr;F_+(be5xk&t)el~&&|5aranm^?#Q3%$;UgMfb(qs`81ARFsC|oZ190^@O5<V5#ltcaTnjgBQ<cZB4l707O2}<lf$D%_Qyp4b7CAh7=No*ZS(ZHqjNg(7Md`6z{osIR4uT{8MiWLW&}WD#55=tAwS{Z?Jv=MQh)>Bfz7!c)>kX}4y*?w<mivcvDPZ42G*PF)u**zDZj_d({pqzj{1|96kqGr0tV;t=#&#n$4D>O?zE(Dt%MHJ{PCkUFWMWdrhmb@sk(#tYm8)cY7B~8i-~1VvYBJkOPBq*u!^R`^}xM!)!!qiQNl8stRm%p)MgC3tPDQR=_OY;%DFkzNTsePq_zkVQ3`h0N=!j+MjERnz17#YKU=lGSkw)-$)2WR!>(|p`M`-;*t5>kh)w<R<wkQx`%eTOc(<Kv-hJO}oQd_jMQ#1qq8PwhREUl{yw2r!7fc*237-miLfX*yaP4H<EwvZ1jlH=eGDy(O;yYLm2}azfpD?d$Lw?(6EULE?Bvjsqb@fO2Fs#4_)(qE;k4MeSMtPFQi&G=7-Tnwg;@M-;G>m}6#QGn!oDOua4U}$e_EN7;v}I=lb0?2(8(?tQK0y3#DDqlj4Q9G0jVWQg^yYU~9%?myb1~ugjIP16a3_$-+*$s@v!4xOjrE#G?%jQxv-J3FR57(&j?pdBTeMe`=Bvt?ANU}DQKbi-F*ZhZ|7I*%$m{``*8%w{9W}W&OM<GN2qSb_aI+>ZKfj~zaMeyX;$^LEMI>t;POb$?>(lO{jQ-^9_#5?!alja0_}q|iHt^JRy1$mESHGM_(!jZ=#WZaT+iY^dd-F-{oI111nn)161#bx!efayNkMI`f6-w!2A0M$kSAG18FX)ndIdgq(SokAmf-5EF)Sls4J4*F-H}m<Ao89A=e_5!)YV8-^tGTZD>1Iqn+i|13_54RsITI1}x?i4L9Gl|vsQ7b#6gJ(*rC&Mm--^1#SFf%O6ZC71ev8qA7-0h(Der$!V2RsF7xRv{suo!)OgGyTyV>tZJ8wpa&0S}*vll{ZF<TLvKk#II<F8nG@quaWdMNpPsyjCOQCXk;`7NugMn>oW(lstr&usoY?K$SeaDI{Qt)F0t8~61c&NxQ%kqy^O<6mW&a!+t^uvvDvG~W8MV959}Ka-rk(nz?bo#C1FQ!H{G5=}cL&Qy5Vux{z1{6}k5lwN>_VXJ@_P|Hhqu8TKXe>hh=-IMI+Z971eZZ|Oodc1Ca`UJ+Rfj9rW!i~&B?=7N+%KG6xo_FZTXJCcV+zq_jf+*KH?Z~~Ixd?FPUW-|$EL;u%BCHUP{h?hs{B>H?Jp9>Z*s!}uG(JL~Wpy`uU=<0K{at6j78Y&v+X#YN$rm1fLR!nmrkIcHPfAa(^yH#IQ5<~KCh}+<C)Fl6o&)r6a@!0_rN0u|kEi1WnZ4{TlLxVC^cvMW*-!+w3|l7x|BFCZv;pBRSPQF<l#6;T#L_~HA&+J^&Z36Bgvkfs3uheeE89_{HpA7<Vq$%<9|Oi=XHMop{Jud8<QOm(*1qs~mk0J?2S;V7KmR$N^{$`5*&fv^OJjDSC#wFwIm$RthHA6e-EjZ%ZuPz&`}u9J(^=b}m9RM(1GKm89lP<NAst$QOD1sC?A}?cy{!PWQjXoqy8Z2Pc&&@)xT@jpsw>u7l!mVQS>12!`o`#PWVfL*{_cmIz@DDx7L`QBFFwrh@$mI;eNz$*yOcS}N+6gozrN=ebOm`0uOG4sTn}4qWm)Td;TzCRnyU249)x$-_vLcfY%!6t+uWJEhek_4hpm2Tj|aL>*qZcP)04-2SWWZJlKT1J2Cx-ib{e$ew|kxxtw|Q1l9YvaTF5!@FIG?x$RE8ZS%3B3*~7y}s+avs+kzyLXkF#)&|>?y+nF!s3@8k|dH$e#{Z4KCutwv}F)jMN5hZi8M4#?5V2T|i2hNsH+s;>E{fkb_o#&@5amCGVF3So<b|spP3^SBwYySiGnG==M$r74WbUxuB7=EdNU9UA?ma{RIwcB@p+^?odt0p!M`w~5j8r8+9xp}pzZ4s8tH%1Mw#Ka@bq6|4Xo}>|45xa|MhL5s5yqu=P8)r}8aR04^=Jh=b6A6$IlgE1P=k#;$iCY$KNY-bJ9G?I}xX)P>zb!GUt=*TL+*=i~_lxUZF_8bH{x;VrwExH2w^{kVJv{Z^<S+O6SuEuftL@WC<baYZ^z7C-MF-<<o%(CvTg1{Y-8qx_l+WEW-hSlaXd0T>c)WewDteojYvP`$#_glok)LqlUrR&!4tI*4_1=}v?vg{LJ_U?9zOVhZYQ1KS6Ejn99(|~#gj{QGJOje*;TpU9bMKPX-SH>9Trz<;sr!WQQtq(di#y!kC?SJwXn{Iyf99>VP{M%%&B;Y;jn#r`a<X1Fl>5U>I)sr|jeFs2G+Pb1!Qw`hryk1|$$mXoiB>h5nafr*Lsp~1vfXlq(yrRAJsYDAuTEus+Xe21FnEX1j9NjH4+K^5RIDE<7nLJa#{Ym^|BwQc-cM6i+S2)a5|2h#kK^Zf-ae#U`Zz3tXw~>8Tj=hAI%*t{5^KA9_i46AHh|CUpYsvr@rkakF|_l}m;CIZSBB#*sL+pFL1huR^#@?n5Z4gwp{?>cy;MPP%m%!O(*wtCHd@cmz!%?-KmB&#GddY-Z^~QAgE?aEM2(MnOtrsn^+|0GVZPJn)XOAfSMyrr;+=pPuJMW7D^{I|p&;6v^gN`-5KE}_c>jDIyY#?YMbR2>@7S!-(RC}&@5qvu9|PX>H>XwSppV0Wnm2wpsoNr~$Ia>q`*YwvWRMfJVZo$7-U>C&i_G8U)tU9B>FPe(F|G7GoV}WCUaxcBd*c%2H+-M{I=Ot~hpx*mZX~(eUfb7;XU_ww7WUt-cfQMNCDY-ma6Dr`<xTVBbs=x66=J}zzln5z-yiSo?+NwUGUG?P%!S@RL6Rq@??mQ<NwcH6RHmAJ=i$whqo&sn=^N0kbT3ZphZ{_`CY9Cw37!cT`PFBw*vra|My;qA7r2A%zgw;R$;}`07mcJHDLh$@GQ({5adqbj7g@`~zmiA3L2ib-&Ptv~t*`H%LnzdZaW2Z;VNLo1OLLP1fBK5qhsI9cfS1!If?IQ`SqKM5AlSj&OYDnIUB0X5!t|++3&C9%$F^y40l!1xoadE})BhdY(RO-S<&))YakK&(zbdCp&G)APX8QCxZInZ}n!Uku@nF*V^`3as;#RqMJGE}9=i})0!87uq4;eEuzt=kplfVS2s1f2|PrP=(cOT?_7+s`QZ;w>;X6;pzTTda~sIvz-9Jp=24#!TQ{OWv2WzET1mNS1hJYmm5tX0=f$^zXVK{sDQ`^Us4%dz4q+3SR}-8|U|vq$ifO20$Qdddg?M$;4N!^o`v0Pi!=JRp-TneI@(Ht_$wsH8QYU8WnK;4<4S>6Z)n$#(E3PautN_on(9{pbl92#C4mW}R1$GQ7{W=<Ff@Hgt4n*Od*_ZC#JnegLuQB5`ccG5&N*9bEg*qz^*U<zj3vWuMO)$__A!8Ft?V-wXEcw#x29DjOvfJ@U^Sin&El`*z)!jj_@6B7CI5pPc+4fDwru<|;4I%p+{j@Agw@v|XZhZ8NaZzJP9R;H+kUm~N&!F<I2{sSFU);q&NrtuHq<(E53$K+~=<7`=#!u8?jpZ{CJOAnym6601r&opw)lCCBk^7g$;i(wE)aFfd^mW9;WTs>BWb0g;=ar-s%ar>vusxx$tCDI8a0Q{aKF&bWQ=(kS2c(1I>VI>ds8^D^mXdbQeAdwE^V8__mZe@`20z@0`uWKZv;4i9!)whj^S8JcfK=Z~`+A&xuvZ~-;(D0cGqStrluguQ%*<+k58!TGS+SU=GP))^YDz53+aGmswfv`e)4d-*0lXYgiyja{3)fomWhboASw$Mls33I@LPg^R__YmFYeQ0`#rfvd`Q{eo5h?5CO@7X8Q6u0ThD)f5?L-=xQRxt*I=bi2zebb|LvFY>z!VznDw+0m~u@aY`lx1DR%8ohSu$ucX03P?!PJ{ZQLeazZ2wyG`8?Mj(YC!HYLg9#G8M<asPlJh(tIYdPX#j|@ZhW+VCDS2LRIjnB7Y{)F^#q+|8%f8O+4>$J03fbcR8a-)4m+Yqj=%8c}#RIPs>S=nIP*IaUbh-K2HfrPL^%3rx!Q+DU?CaPV_R4(*<kV>LnY-^xD-WFpc9!3#R;H|U*1wk}A9Ha!OgfYoH005qAF$K?C$uJq@3^YMi|?JyxIFr_YB#xEpA*v!`uwoLVbk+_(I=0I#`Sg332DTyS>9EFw0Eoli|4>@ed;~@K@7*dQ&ro`@Aa6Ko#g830%Uy@zpM4Mp}SbaJ1)_{IMTlcRh4(GgH`*>UNVBD>d#2EG}JMz7uD@sQTtx><hQ6ECP(R+czUFvUvSlo#4aQkws6%jj_+=W_4$&Lr+4NWjzJuD(xP)_qg}Kwl5@aIL~n#X9=4d?Bji{e!+Y}`PdOeQCxbj6Bh4aQBsZ)sJdgweXBI3+22Kk%MZRCqLgCt^yjZQ~?P=XjI*Fk8u1dd#$d-Dxi2h|mb3b~8fO#k0(9F@YsMo3S=J~~8o^of_e%79Ba)-}XV%d(=7A!w&uj%!tAzyy?d-wd4U=z)BP_9NFXGwN>O<XdXa^u6f>qgesI;a8m8TkisI(*E&ZQ|_(L4|9Q@R1j;8*b&G%kKCawzAz-SlU3{{0T4fI#s7iA&y8oC+FQ!8<56SzlxsZL~Z;E2GFOhHEl7{f$UF$%1%sonb@-~(;btE?DCH8d85Aigt~+H?(2pP$KovQpS97atA6v^aGwbKB{_}NW|PHhbDbiR&-GEOg7XnyKgfAcmIvc_)@?@9U5Xv9sy~%aZ3h@`@asSk#}DS*_qjQ4jQ+ie6MPu&To{KvxZxUC|ARGm4|4pVYr8*Wm&w|m%duuR4PGzZNPh##cwnJ-l;o723^}7Su04BgsRaMLYIk02TN;M-fWBcLwzT}r$Jg>h`t-#x2g!2Q^O&QsfzAq*=Bkc-yb948tNlO~r(G$pHCSgWj@z}dzb-GmR<-6qlDp(4dweu|lI9W;9(QLqPuU$2vyDc*8ZHJKOTWPpEeO?_JY^r%$-P$Yw^CJqsZWH?T$Btp$kCu*JmGIfex^?MZ%1DcM-bKE@poDW2q$XH8+Yb?uIx&DwystvNwD{WaXvf+4g4M|O<ho*$8mSI#W|(s4$B4309^nPsMTWb3GRy^K(PMIH~frP{`Spfnm5SNJt^9|S#;RtQWH58gQ0uwWeB93&EEDQ7OeTHCJRdPr4pitt;=KE)IstDI+VJ~yNFWS$BD{=_eS&cO_c7^V^e0Z?6oT724~|)6F=i#<$(@&F<PM%@<Fg^E9g(}$%oZ7XWP54QTK55g=V?RwtLK<T`#X&%OSL)hCQIQ_1>~4;L=^>c9S@^`s?+vlptcI-H=e2Prz2`8UcMGQzq24d}uHDZnZZ2y6fHTxqgKUBxfb=eDTbQxY6&eTlf5Ez^;KgRZ;D2H*L1xX#+!PWJU?&A1;-X%P-DGyJw1seh4SvV5x>HAzBw~|2F{h3F1U61(w#^!FF(Pw6o4wA2ZMIRPC$`T8nXaqv<_<_f1jn&_*bp#SNAA?#jYCLBHdn=Q;PPGu}lfHC;Z2gUwG?2t$4=0X*$_I&(jU`tj0wj*}^27my4=3}g=M;Ttkt>%#vo%|0zN<a#1!r3Do5_P&K@k8JeCH`TW?dg+}$pm%Gjbp=m{1A{(y_i1}jxxJv4G<JJwn?Sxvd+u!4SiHL^8S!58MyL8D_i_!sl0Y@pxfE(Vifq!mRq^Yowr{FJ%_-R<zNgm3=hvmPzD%djn%RNqM+@w?$;qXPA$zzSW32`?BQWPJOe5;}TEi<Uf#}I3Ur%pCb*XPtZ<_cjy4-XoA0f7Vqr$~x<2hICF*c?d&@c+9vCEfBllN0|3thI4%Wc|esJ4FSeD8h4d=oeM#6NQ7u6%ZXPxIx>?zwyWH2~;ittQ-O{md`8Yj~6JeV0LNt^&WHZzExd7MOe4E*pQ-qgPfO8^K~)2|-SO&F+|678w<;B~3AVt~N-VojzV%Y6pmnYxAUQ<V|*cP(@Nq$EWaZH~8uLvKnpE`eew1U467N%-z*g*!*f7nCYw2oN6-;HaWDS=W{y++`ljPtKx%buYj)+|2tQ)Gg?!On<2Gv7jK{Zn3;q6dVk0lNZDvzhPA5PMenuZZeLIsc1@L&1{-0tTy>Vy8gwS7elUfs13sCQvyaS;T-aWh<L=^ojGxNj^;f6#O^in78UW>F);-HQxPRouyv9;8wk*K?2-~GR)#xmmlX#iTRCoWZ`hQcnfienh9h3fN6Nwg5mw+pt3<gqG9sItH`Hj?q&U>XKz5ZJ0EJfQP_kBFEPFMHG#?a<6h1VwArj08@p1e+D6fRdokKm5$K^O7j)WW8vQjUM89AiU%M?iVV^*X&_on_wpX{%j@qG-_;T-jWZ0hUizjeV5CeY~>mfM&TH((i`ajs?MZ1RkZ1wdGJ5z(Z{zLh-WQk2{s5wFSKM`m$ALj(<y5t;W-z<HtcoL2;f(EV)v*p6}x+hGMYxN5xU!7B$eFcIe}NCj7<m)IN&BxYY1xw;CRm5p~|KD`V@tJM{F&yt%iL4ohCW1OBVxEidn$zA~Id(oq_z&wsDZ#;3>YBoR>Eaqoj`n4juwa|rAglZkZ<(}jG3Ou+h<1x4)b>-09zU;ceIIcS|NOdId*U$9N^L96z@GCcsT8pAs)5YtM3WfNoa@?eS&Qhf=uJA6JSjkA0dKX+KEJ1xUPp+Ha}zSuq#6MrqoUH<e%qO3Na?G_5rw}toi-Zr|st}>^9d%em{AH|(W%Tmog^k3EYvuWgVd#du06O>q}Yp6q{4tIYF@y~o;UqNqk924D5FrwT83<yebY7p^!YR}XYd8~8A;Wm{y4y--2IR`7kh25!b=IFBJraohwI*TyKbQ<>@(CIr82$c*H%xy+|T@1Cuh!Xw+fA{pY^D0^oUB;W<R@j{^xj<R2yc;{BJ{-2LTAS1KkJE_JfC`WLsV*7L6>aAFVTtTsEJ}*Z1y9azv@*#Jt1q-s)}fyHW5D3iRMK1_RqMEP9kkwi>}xLd-e4w8-^oH0?Sm7X_Jn4ThXJYB-c8g75?=j{s*JvSz07Jmk@t^!YR#P1b!>8jI!QeSFwYVLZ9KYFyLQw6E(dFOdF(I+>Tda-7Fd*TwuuJkFJ!z=Rm43TE~R`?t!blx9WEHOV?jfCAx5=yHlQ!a&pX#zt6`q3v1awrJl?OE#Z36&MzW^tcV9Fz(O6n+2qb>$oVMHB@q2_~{_MF*`~{P1fYI)8%CM`Zy2QfkRT7JM>V;Obp4L#wUiDA3tfheyevTg!m3uIt4+b|m*w|e64`!|r5wEmH;+r2s6436mS8qCC#}&~cTJxtuesh^Bbq|cQlkC@e^J03p`zW)|RO9?v-0@ovS)41CYh@(Nx4$u9bv}qN2VPyf^TGD5WwfOm%^CJSbZ^#harVc6dEjg0XY_P|LupKs{)N5Vx>2TidhRtr;j<K+>SM8&vV%lN{Lw$+_dXZAbJ(M#`}<jBL-YtQS2><B@N{SsGwuz%#Cp~f|KsXRc9sPaDEcf0C4wOVWzc|%12TwADj*AS0#r~G1q6p@@9*zbrxO>H8%$B$drpW&mUi2n_n-$BgUP(q=L>Z($+~vi-F{>iIFj{mzuo~<s|6rt=6k^Q!ZFg%hI!hkb?T~ev3HYT)Sc&KO<Ew_l0NSZ3w<sqJjA}W;-0PA9?g0;R}d3-y~sCSo!Rq0jpnl6VZ^*PIk`X8M5oK5gfY7fSkHbb+v&J_4$9>M;M`Yj65HY(y4n$7Hkza$M``9b;h2`Q#scJD&B?}ivRZ!(Q31bIZYS&NQoelg^i_?ytK^gsXS7yQ^Zq=VAEdpv!5C=J2-v+Znr1N|^M?N5pvn7}AA?|t<QMnTdy^Njb4e4AL$$i|ij<mbsdv>>tCP`<fBEcv(jL{)qX70_QGTs44_>7ElP<!r{M{djZSV@qqY=`3iV$qkE4<1q!~OpG=xqQ#j~etI0GM-q`Mu;lMG$-smz|Km->S)|vAagh1|3;W)DA1EU18oxjhj}y<FT~y%Z~-=_D_;BeK;!>VSUS5Er8(mTMQ{H=<vQKj#*VO_^j3kHF~FtLt;!xgHEuJZdjL7EaI2!q5usIXNvVIpAaJ#D{t{0Z9fG-QtNko*t<h7ApdPtQ3nyb752jItGbsSqb>pG$({}wNu*<u1g7yT+Hi#y>H*sPf?C`{*6rm1N-=(BRyfnYJK*0EzE^Y(pqOL4{;Ab^^7A(<5nflXx8@sAnA6pZhaYZQHdGM$6K?S4@Rg?dXQIe+1^A}ho5Fmn7SWgZgX%5zXbrO3``suPe@7JZHw~3;yh-*wn#ud|j`^l4@CD;~_C=K+nEJ15wEA)HnUX~XqXIKD&QBS4{wZ@JotZvb8;<kBt6JZLp$4Op4%@B9W4i*=<y$7^X*6WXy?*zCPV>El7Cr4Z{S#6%4kF<sB8B`pvEgyZI6z+SYgZ;y>ppbzMuiG{8XX<a;d8p~yXU>1@Wh!6a&q3bOYYq)psbcU);8(f(xQLZ9NTRK`wRV+{iN}i*B$!Cr(d6}kVCP;o|S_<9L76aJ&~j1AgrINPxGD;z@F=>3zfFyQviRoPTBx!u}Of}q&72Fq1L@7%C*I@S)J;}y&<ZbGjBYn$w!s2wB^3D$vP`OT+fJ}9TnMSnt$#@QXd(sgLFHqMz0?qLGYOFKMTu^5Qz`kv;qqk{nI46XO^dfybMF}Sw#6ohi1k(oV#XrS>2A};78qFZsRYW+U`itepmIB;M)g01?#MTC<UtC$Igwu!p(0y|B>7aYud=8y}1uFh`=s~j>Im*+kQTTJ0uhnv;7Q|c5}(KWTW~te!lxW9o(6=xGqilw8bgSD5i+6EKv5oIxSzH!S5cFor`#0)K+F>0A7DHkJ0N_+1w??RED9jp<(W*+lMzw$H}aBBb25+Y2#7IGtKur1>_|e|4B!;HqDt>L<yXEKlLV>*@KEy_G#&uQ!82>y2`K!KB9%PF#}@xcIV65{3<5QW-Ia6w%3STu|c*xvT>Y1hww7MIq2B`OpxJ9nExu)?{>1`pj+H^I!(77Z1CLQ#*6P`)d`;QD3OTb43nM51?ZCnI5_=4-=1D_6)^1&+-k<g3_n@$!P{;Sfv;wd*x`7<YfsCL`&EkeLEqi_Zcoys%6;06v3#7QIQgik%vkJWFCaZex2^TYh<(p2`cQYr?wluiWjhzdAdimO*aW_B({0nd;Xzn)$4%b&zU+)ca_>=#s;#g|cl2t#?!@elc_xMR-pCBx)V8hDlIhw<c;~Nt$$BEZ_l?KbQS|d((stN>t*_1N<Y{uve%nD?9Bz$V?999()hc`88LPtPO~5`xmr%UTm^vq$=`1lVeIPWm`gCj>tvY8$v7rOodlqzhaEXBh)&l4Dqr40c$j%xa&;glhF4W*M=X}uC+Vp#DNbqRYhx_d{N0b>TLegq`=t0Uu32x$RY_)da$#m5?XSV1$nnBwyw^{TC?e!31iEUHxTY^nX*xLU%$I7NQ5AiPX*<3LM^A7zK-<|M%qbiMMv<v9Oe(tEsU__9|^zKHT;})K+R}*Y&yUqx7j!Aqzz8Gg3kPLX~FB^ctPr}dC^=MSD%vh5-A9R_1-#VXKYjJS+-+%_M`&IWmN2Jg6`?)^z^`z6-9_pESlz__9O4{<${#=UY2vo%p?bq&9joXlyW?rMpw}3xRxKr5<v@4u8Uid(&Mz1Ac+~~?>(C%exdU$2b*Q?XLa^~SQ_WAZx8pXi;cR1v)j<ay%>e-aj76dEE?JkF__9$@Xl-O5+i$GW~h(0BIKypq0k6&(D_wuL9er<CZGbS`wk0hICu;1V0vB-VoWHPyLz9XR{7sZB&`YVh({Au24*60+D90ikDO=MhqyGPa<fZ@~9f9T%!w};9s0P(D)UBrqpjRrbQcPI43ci}3#TAZ)$){kIwltAn6m;!^=AY9V<1D!pZcSpA-0&4|1%I6^781K&A=JNET#W&wio@m7jef#cT`-7XkL)JidLDu2nRFuiA5=Yj1DtCLV(m$Lw`?k#y?TN8GuOF<_7i-#n<25R;tT;81^CNJ8`Gvo{m@wD&L5yDx`|n?PW}|o+PDj?Tz1FemVACj;u<kbpV9NGo@3f2j#-EyedEiZZy-4T<%w#mDhOyMD1Jt8B#L8Br=&bY>KJ7J7g9k-FVHFd2mC^nADn_%9Z^snW1IZGE;5N(tFoDr8Xd%2!NUu{;ZeY<Ea_$eaukIjy8$-*7F_|g&;Q2KWnBLA?>{wvo?5)4&f?(JB#+=Mbogt60^V7r!UO_K!5!Qx?C21SfWvEq(8m2-de=OJ=CbPH9PNe<nc+6j21ZCGSVz*7Vd%9KM6C2uH(J8SzRL`qTf*;;ulFQg)F<k`T&*Vi&A7=UnT`R%Hv05GUBhJ34IrtZJby6Ug-L8xcnx&cbg+fokYa7`x&`^3Xw&5SM*>ZpKl;(_ReA`*1FP+yK*t3+=!J%r_my){FqQe?-Uo7vBO-zRNjbN!yQuEe^!qSam`_Ze5`3~Ddrh1G-g=;JLw<CX5E4KddvkED>xr!vQ)4g|Y(_0;?ZCmxYa(_Sc{<*a&j`7efbmwG4<7+RG$xgoTXLt5)NFwhu{opbNBT($S$H|r7qhDeja7*^y7SflbTo<*neszgI7I}l(>3ZHS_fUbtX8~|ZI$-1J55o-KKNwt@59P#;Wy;zrh3pWOfvmojIh~wL<Xk83VY`#|)aT~#P5U}-j9TO)2>6EbM|qQ`=QZ93ws$Y_k9(+}UeH3EsZMZPb$1>+-xfK4s?ZaswH_C%L$WgB_?(m~M$U)kNxu*4D?wM|9(QhZDsSbSl2r*;i`@#Thm$ofDwlN>pUw|^WzIo?IqHWwFn;x6b>h%Zck)*H+Yjfx66|b(?zsm1P2h~3&pYgLc!QQZFWBaKLiO^;?`1!?nraHncQQ5B8sWeC{^7FvZje)@p~*9RymW|D$B5Ks&x>_pS!wZc({|hwT_J-k%d7b6h(PV{NlJuy_x7w}L=c@ZKzfgJN?tBL<~R*6Bs;ju`<K+C#T(<)!R<@&E3<$90ckXXa19s$D~#gCbSwbN3iqf9N9R#t5~9jXK6-H#oA-y{)fe`3*FcW!b+&;{Dem;Y4?aoWP?_f^=+a`w>jhW)xSj5TYhzpb)2L2&(@L}=rkUTMuZYoI{JeP`qO{usSY$f)Y*$AP9z-14w-KZL=@IJd`;B>_e>?>vsHZzMHNA@<X?phxd8{@-)%WQ0m`4Tp-D<Y1K)1h#N-oasceZ;9`>Q{M-FNKI9Y<9)X0iOc3Gm~~+e74SFfI>O+7YmIZIz9>3BvZHGuTOjEgFWZ@SW8us)k$O1I(NAJ%mk*0jQ3e&2e)O3({=tEH`*9jK5v<d((`OFvfy#WHtw@4&0oyH{1|a?HwmVEs>Zo$QO67cXloNhboc0KCe1B=N0+{99@0}R>owus(z+@{o~k&aef`nx7S}Ds+F!1VK*B!mG<C-T+i#aP|srU!R&ufbXwowQ){H>?c;kbwvNr7a(ONz`cF@oQ9wpgc5$~{3|VXv*M-vr{`Vw21iLExVGXN>2bU_B-S|Zv9Tk%dj-s+Cf8+`W98*b}dj!Eg7W<gEY3=H+i!|Gw9Cv!yB!U6dG+f2T9pc>iIK;|K<Yl{=p0g=<J>g-tIKM)}pxdO;nOtySyxH~{EoIr(jmxkj4g`lUnmxV7Fz9J^e;(xe+o>9_#9WRg@-Cl_PHv)W)K&VMwUj+qzg}`qf!WbmTvmr;0@lNWzK<HLK)AfMeI4F*j~PB{d;7Z{h}~VIcJDNm){<E(+@~T*sM;$BJ1UY?3F{v7I<S^`^HUbH<r@Xsc|a=*-RPF9y3u$T_1PwZws*)iWc0$fk<TrSh12s;S1Jfadfo4byllCmUtK&8dzqA17P9%&^{(?U%1`)I&!Y%fuX16<wpZ1<g*=Y_r<^t;10QT=P^h>8G&s)UUA%wWXN|u~QVTwQSJ}zOb>R~fzuRzpl0fUp&edqL-C=ZPE{+%JxKlvL_X-m?;I}dFT%R!sxovsK=CvFyXO;1+5;!Wj-Hcuv0BSz5bn5<@o!{8tz12AwSC$2R`*|lU!<}4LYO|Ubnjvv)55Efn0WOXGQb<-$t`#HH@;o%HDpJ4d3jleqKlmBzft@^>q_162Q|=~Ifa<Nc{Eja|B1lTx;!L}D`y7XB)>ur?tvj_cFYEUj1LB@@w0A0Ls4^UDPGfZwL}N8ypZ#^Jj-iFINm>_4o1gd5^;pe2e&z2v!rjfn+F6~^M?SCTwE~*&Pd4gq6<m2)rJ>Ekqo11l>fYZRk?pLLe#Q=X5U-4lJMJz>xIB**M4WKf)@egm-ATX>`zYcT9lAFewf$^4_K-=@sWq5Bj8{X50U91k?15A78hDG&%lJG>FQwY6=B^>_w||UB73zT!M9C>QxQTY}>vdNj)Nif6JqmSpybE+`I#3rbSFgwV>RV*9VRiKe2F9Rr+UyuqS$~f#W0oI92#Wc-D@J`1bB??5U{jZuXuS*Em<dwZX046hrPJx4yE)R^EHg(FYW8WGmzRG5kMB7?=^vX#2}#0e-#)^{qPjlJ<rAqjrQ?{Y+$hVV!)IeHLWgYiml_wrh*u*BKKG!vic0Dw9{f%ZdTQaD8klLz{s1A}WLtXk`+Ozksq#3Ij=#vTWcAx1R$!il=I)nU-`1@W-^rgDfpa8({E3+`bsS)HJ%)?WpNOXdXo_{REqn%sbCpNc=PcJ}`l!@62hoJ?zkQpB6u@+M9d49un$vlFy*qD+;=tyG1(r4Qbf%pjwtpdCt#+d1wmt;yN=04NhxO*OqL1datA%ccZ%<DzjFnfsU*oTPJzTQDypx~0bE*YWk3cwFKN-V7I*EM?<o!;Uc(7{Oe)Z9jdL;72(vjEqXUgVau4>h|?f|Gi{}XbF{i5Z3cnDjH0X?E!Y_ejh8%YTFDH<efT%V<fePgQ;W%c^FFcAWD=F5H_4_Dtd@P%sdL!_Nzc}41tmZ!eclo+K2w<?6XHVoO5q>O)eaPO%-?#QZDVe;%(ChgpFLq@L)hwlpI=6S+Tx-|cq^l4n6M>8K8^t+Q^lK1FToakO!*H+M<_#D&9?UWtQ>(4r?nL$H%9m+W!c&5Wv<X?BQe%aH?BoJ{L>P91`2cF4Gvg0c2hnW$ySqwF|W$u*6kj76`eg^EL+*n4B%7`3Jgk^rLT%3N%{?)+xG}#Wm-7PrU&~|kCZuVcaxm+y8zh0T=xL1kB<t;}Sc+iyb`kgG<c=kK^he`WBxdiMzN^n)}WX^)S+wRp-->to#Y~0r{I7$sv><QQ9y$1_I^SEw3FOOR>y{tzg5OYT6;8t|W4;A4Y35)aXwkGPq@x4sd^I^=-@$)okBALB^O}2Qx=$q<L`O5108|7OxKQ}0do!<L7(K%j3j73JB95m2HQOtwdvL(^mZ5<U~*><uwRRLNwBXrw9o`5xFna%~&aE9MLhc5J`=tbf0q8_zY$0y@;GAS62)omb?+-2zn0%^=@U@Ms&t5e(MI*`^EUqKOT>)M{Hb>rQn9t&OZeu~?TA#62Q&4G9Narl|7z*=$m_Vz<4-&4_McUe~uaC>+(kG;Ni7Oo3RiAr=Kaf7ri@;ru;D=>|$<w=->&q4T!k=EbhiDYLwiKAxQs7$k0EArFwj9kvWN?5j2Kum*QVD8>`?ephsE)LAw1cr^dJy)*oe@=AV#2HJ<w_P=OU1$}OhWh&9LL^~4CqKe(?%u376&e<U9;+56{3@{<bq575q~Cnl^0U8)-&{%}yxp8JEpvy6CC<aFxynq~;5Mgi0k6?{HS~JON+n;{+Wq)22K<KZAm8DHJ5;f@c0W&cjC&mG1PU1R&b_P;jYt0)Zn)pR<F(54Vhe@FkA6QnpQ&Sja(yoqH#a%AuC}&qjkoPF-eflP%z#(T4s_nkODS?qry4^aUy@OtNIg!NRz4%QO1(!-ur-F3!b?P&*Dkl3v`>rcs@n3-J@Q6Tfn=6`D{p)&X{ybmbhNdhIoogplV<Zr9Y_jn>;Q=<oZ&y`Dr<+|0xY6IXN*}9a)m4Hq7_(5vC`hwZ-4r;@Qo_^2f>D2%S_L;GlStxI7Z8Dk1{y$yF;t3r`GY7T5bDb_!Hi-v<Dh&*gbULIs)F(SJXD@0oaGuU*_*!J#L(DCwnia;jXt}Fms*a^l>_09ryFxRLtIxDQkcASM@F?|HxWR9?sz|RVl6keT&!o&8%q-m@-@8PjjrB*PciaV5vCIw<*jj-+ggmv|xMmZbl)mHmB=?%G-v9HmZMQu2%<yYki&j9VhVBcIxOfPFJO#4(i~++;=_WOb)Ypk(KOyG`Ty$r+mGhtpfRR?9>kzzz+Te)86QCd*6U19TE=MyKZYcW-A<!9*Bl-mN2z`c<)E+e)V48Ox!p5xG#KBi`heOvi$Z-oE(9E?LNl%`T}<OdZEPuN4zuQbk0vvoav!%&&)kjz3;$4J2KI?V%~3+PPdNt$Bz>==%(x1Lyr{)qE}wlV;+l!#ohW;-D=*eeJM~OeiwSxp>uSLFzHSmVAYmtGSq{Q3&jZs4N)C0AN>2gE#tTz8_&o_wzmL5h+tz-j)xV`Zg!LEHHf>_6^z`;m-F`UHI3e%^Br3V>l|CI{j3z6(=jb_&=Hn9bKI4x{&6VlU+yD6@Uid!<PR}gADZfn8zCK?vZ_OE)n(@q-Kz~=s7y-IYN>Ow+0yId_hd^}zs>C?Mo_2JUb8;MRj;!jf^kagN%c0$jzCn)zH&}mZV|V)Kc$}s=bXM1hsdU#+^u3)ZJ2E+74AV|x4rkoE~)p)RiGS5B2g<`T{=Be8|8;Bwm)CoZ1saovEL8cJ*I~1JX>u>NGhX8<I-U_GH<chXe%gn7&>+Ot1WB`SlOnG%)-$1F9O<=X8E?~T>+f-C8yo!8dLvzmc|INH^s2q_;N$6**<9logg1I&y@sc{jvB{6c&W_u>33gX}tRnk1QV@z8;IGEnJ~0p1PI-qC;D)YvqrvbMp0oM!!?&7t0XT<TAjg;u_k;m&C=sC!P=ETXrNjEJWY@Ng0lDED}1}Y4qa$K3#)O2**39+fEoq=+j_FjSY6oS-Nr<@YhNwO7`5*o7LeO3iaVKVK+>RUnue<ch)JiIDj*Q{SHIp*|V>1M_7cyqurzNa78e}*B8DWQy+@9MIcziw!HqQfx$1Yw=#b9{*B<9`fQ{GqNu+2^BM}Cqj&wIW<aRY5<#4V+8o$J%`qs$ZS}OG#>wP^<K?ioB8CILj^P)K!y|EX#z<n8Hrk=cq@UJn(rFS?EO?+@xXniA{c(M43iZB{Hr{`5c9AjdNjB~e`8DD6ERxpJQw^5#BOhU?PR6X3^w(=PmO^T6+IOa*Y{!KXzQ{XwxZJl#OdiylnY)G_&5!Zm*vtAh{d^fU=;F8sR1Jf7WaZDhR3I%$hD<v9>fkcQH@dfl*aXawevRX~;VhphPgalC&c_YUblx0KUf=#JLC?}peZF$tU=zbD?{%a)j7bjf;q`9aZ*7<GTh>qew5*Jp7u19OLj*ws(T0<E_9nmvH*ks)3sCjgTGVf!ZO@-A8_kBJWCyFJB{ig`<ctdKfr?>Q8U5z(n&r!AV4c_Usoj1q6uNq=@$4pdF>1~{uH|1%45~}Qq=sen+{)OIgiMt=;p%eI80PkDwmQSFzRW>9qrN>~2K|eH`V`QR>a+K+ZrAhm^Q{$Oh=kjr+R-^tybhLN!CvNrU-ZPf(!+AHkC#P<vA<AZZ9K2v;=F$qvflU?{Eq6H>s0sgxy<LY=X}(Z%=Qy-ssnhwQ@pM%S2oE6B(T|~3L850X80HHs-<J+6}>Z6<JVtLq$`PbSrBa8l})V~t@>htAZGeqlYCEV_O+W*{uR8I-o{xekKwt5H9>fPd${$Rf0en5P<)Q;NjJ8ym>>X9d1ryJBWC*FaM>Dubwct2v=*q%C0HZ1Xnye~MU;GrFlF<xvC~@M&!}HRw}}Sk@FQK8<+o~ySzI4*bDK}$`MOQ3;Y>xjp|VJ_dI>;$4@j^MGnf7D8{ucQNB#D99bN0t|BQ)GQYB}M4V?NJnJBb828UPDyAEYbo%PQ(dNFU##tj$#75GZOY>+|V6+;CB%D?s1q*?$)<SQ3Kw9l^`jlT86fqC-vcXa|f^cx@Pd#vtFs$o|N=cFwu0^%p`*rDVH!=p6S)5al|IM`~JGagm<uDkVksVf%;{XOwnkbuDKtg{EWgDekqdaCT^W^2zB)ri|ZJ7=)oDBFH(KTRQYp+9q0nM_C65r!f<y%xnAdS<<c7-9bAtb5fS`uiPOL0j$IN=Yo2{`Hg?R4-em#HVYhi<>JEm2@;aoKIGL)IE==ycOLf2L-$RHLcG+R6JV3l{fn5RDr*nkKq$H3_L)%t2ZuacWVm5eV;`kv@}nCxJyE4O~7h(w>#C-0n8q#5rYp4*6pvadxN3cYcs4ost0y(T!`2&IwalMYu2E8!#{@X*dO9?!>ze3Z5vB<!qNpTH%_S_9-Ec4_Ir`r{M&D`QvAS%miOvxy~iGACymAM?w1t;ikH@=6A%a}NJ1^HCOoa3w)NNBIe+-dlwtn88`oWrUX@ayby<e&`tvHBHnza!sJOS<^WBu0#xh_&nsVdGF45@M8C18?K`D%}AS%iBLk89Rs?V>j%h66QEZse?`tkjJJrr9jVm4L*(a-l{_amPgHLG{z1EkxRPyMiOx85(%P78^s?wNb}*5<FhMQfNW-`!vXfUuMWwn+?Se7Vxb`h?v3bWyHvHqj!DyErrt>Z|)Y^L_Y!3t;~HtKR+YkaZ?ex<13AH>3`gAp>f>W}dI$t1^Fi7wP3(1l2j=gTbI{K%n9|le6Yh*By0k5)-2N=J)NsLzf?MA!Sz-`XUqBz%KSoEsN7)V|4&MbsDwB>UHq_AV_^oSGBttd82X73K~#`tMkOr)bu^89UevhK3eRXRaYKQ+Mb4OsX;Hs?(!3(sfoGE?gOuK-brNqoS+*ZTWrSX2O~c#*EYtT)bKsTwcf-UYv^X1R~Sd__m{K7rXI>aLrK1B8uU2q;7nTCU+C1uaoE<HEj?H%bzN+#jj0|M7Hlrhc-J}zc%>Qy$5XAX1DsJ-E;mVJt^NAC@EKBj)s|sdn^Bhmy{v%9xZZ%qm33Y@mmPAJeIY#q7R?j-#}%jjR>w|(vWcpY0qY%92!Xyl>qMnK388q^XSbI=NzkH?^U7#9>yWLa11W7`VRLr7xs>d5#V?iV;>{5H<K9>{$Mk`2uQW*f`4_MINt(4LC9nJT$+}hBn{YwDlV_U&e%-+k$gbPzY49333@FSri5Tqn@#HEf1Uu;{b{FM9|1%y9EOn6w&HgO0#uek{!D9(9+@KbHFIMq<bTjexRxZDdPMFZAPI(umm>GeZSDa@fsuTmI)8f8^PP>1$n<PA45v200AV+!YPY-R<Jn{9XX^bM|`<trGIG-ihaM8k-@#8(HL+J=ue_*=ugFCfDRL<1?@Zy^a2hZ{<x^ND3bH5Rbq?(fan*3-?>P=Xc3LJ^JsM4}d@nfhMy9=Mr{xoB8g==qN8|}J;8E^dhU2STEDESQbx);$V5xU`Lctf^W<1LbnXS`f*&f(qK4WR3;_T>`c%MMoa79Mr`{`WYs;B8~qygovS+AV$y+PI~i04b1hSWVa{A1N9rH2!{gotd~lZrgA>j(klhthS}Ivb!C3&i?X11FMa{O@~&j$0m387Ph7NkSRAFL>RZFeajw;)^#v9x)aJh?@IlBO7`j;^!)|NjK#BZ;l-5yBrG^Ei`pS21|a`vHd~anp4RHHps$A23w04MU3?U*Tf>v`xu9-w9V^>m3jLWqFd3KSnYwVVo#RI>s<Nn<o5D&y@J{!Yx_h3v<sRNMdy*%5S0$ARbPW+ne*mzPFl<9blpkgmUBqViPQ#{mxup*@WWd2?^*fp!@@DrZy`|noYO7-_tVE*+?wUho&UUszcP^1rX_#(bks?FdGTAEN*=1=we#>fi!x0?x4)8_Zy58-12A6`lD!>lPqo3+<xgCy6Y+4tolQ(Hj<}V&-O|EgrK7Z>=vH9yxXJn1q-=B;5EDfkH3=NLOA3OF8w|%g2_GW!%^~qV-P7hsc65YJH%RgspzOvo_{G;o=w<R__H$=)q7H^amS1lZPNEGfEL1Svu(|D(B6G_O7>VQTP<EgeX`gB$lUV4cXhT7ND+)K|gssRt6GIrl*>C(Hax|Va=5cyS2k-r8N0(h45=hoVoGCJOqWQ@O`BTh!j2(A@VJl~EEYvZ=7++J%0q4rO<VjL!uPE?yUKNE<0w*%h}i9TfPsjQ7*ua0C4reNi~UoTuVqn0<{D|=5(P0sOT^60DrfgdD8ry1tx1-y3_ciKFCzs;~jTQ}w@c%&#y2~rE8y^$lxUq7$#`#G`fWg0<Qa|377KjVMHe%m(m*5fgI&I(w6H{#i@(ox4QbgX&&6K=ze&&mA_Wns76@AfsO?Mv2V@SVKQ?c0hrU%iMMzIuEsx?gXTXm=N8`}9&-ZTL*Scc(85*DB`ChIz#QrMxi{2F&d|tg4bI+cW2gWU_d^5ccPVAl_g7Ra+&0+&%W=+u6UwyV5th-p3OnVE#B=qsNL`)j4%4BZrE!To~?-5r<o-O@hJ-vGW1isk7ofr@Kuwmey~@3aJj^-?N82d^*SaIeKz@mkg)XUkEODQw^(HTYqmrX8Ab(wRYx%`2Mc~{dyU{4t$Vm{X4cF!=GcQv-{4xl^(}uy4cb8*wF|3XvL!qYIyvJx@?SY-mBCG9K2s_2FZxp+O0j0EIa(?-+t_n11l-kEbc682@lk|B%YMN!u<{bA!c1}h^4+9p1SvON+4tFd^L9ISt}l_eK$}As+K6|pITQ%;IF6HZp|-pT)zn8ws{5q%GRpW=wt`_SBp&a0kVkfEA#FC8eR@{gYV9*slT$TrLX2tJsB&Z`m*WKl!6*1cPiJ{k2B5Y<FWFReyBA*-Z#%OS#+(7l;fWCKDt*q*=QHKM;i{H@ly7=Z{l#$%0YM;EHUpadunaBRx+K}$;mNxJcwOnOVMgK!t}JJou@j?8*`}+T=t#M`Z?!&MpZja#-ZSjh2+ZRp*?V7xHEW7KD%c<xplYA9%Y)g_Ii&qkDBI<4w-_mAQ#K#WNO=u9%><%a<#3Hjm`9-uB)4M^`O9wtO@{Sb3gx8t8H-CSTydR3UkqWOvH1Oo(`?8G_XSC{<}K8?e4+yT`=u^H;FgDuYxao8*i8fw09;Y**YhkK^pVW&NRZEJ?Uz+^tvC4ZmT=D+x1x`_CmyE#o4}1iGZLAC$8V<T`OEzD6j2@Q9E0C7C(q_S|N0v#@0tR0UnEb$!_UtyFZ-FlAUt$3`<uwW_mX=#r8D23F*(Zi|=}ep5Ns<>*1r(QIeOncWB>iT`E)VH0(5J9mN;)?li{f`S!aQtwux<{SJ@)4NQZgfvBZ5T;0k8*g@>H3Edd}BF@s%2i6ZZp1zo-b0MzFqAOAWJv&>gTD32|ul}j%yV!?C00%v|{8t#~9f|*K<q`5nVV`P;F2jp?8%T>uwW$0Wgq>yad5Ie<3xCQH@yXerCauf$0AtARm1>ov?Zl)gp?%|lz9(0X%&V=JGNHgR=g{n$drK%j>xdOeOEDqscNOz7BLop$hhIk_bA7}FkCpa#+`7AF>{PAdn!#FGg&U&$X|hCBy|S-HW&JchU`i`Qhuk=V;`eG${pNl}7#dDO<AqD#tAep;x1uuqcDNo5H!oLxiQYgY)#DlY&Oc&sS+?B%4!P!El717x0c`E64A?Si0kbEreN6~m8X1E%?A{&EZE!jq7~oUf*QU1-ha&A+rtR`%<lzqahEt?SscS3x(>GZ0*C*!EiLM@cxTp6E8n&?;o}V_2RxM;1v<I*Y3p-aCQi2D~Y@UU$`G{;H^lBl6{1i$yo!fmoSZK}0g2bpru<u^+E^jFV;l7ZP%&is67GOz4j~y1EQ%z?Sgs@SMcq#{N|B+jsP{qmC?HuTx7u{N#ANHnR0gW6%ag}!OVK{QC&X67aB&K$?>iEJb7Vcwp0p7D+{{C$npaD0*ylmjYkGAxpGoIwMv|IXg^2hxxcClvA`2ad!+bC8r9ENzETD$(N^=fNauT-F@MikWd&IT<`bV(a@2wyvomSH=3YPs}X9agS#zsg&F+IHA(XJ*pz>6i2@NO-y>T)~Km)mP#QJ#>2NODg*51bPiO#A-Svt`;?1XHc-HQdVth8qBz#s4mvLJ-6_6VAayXlsXF6&~Gy0A0L1Cq3Gq}z;7PQT018mOW8G#>#DuIFRV2te-U;5V(gAIpLb3(9JvpLFmAs1X}TCW%Z_^=e^gCI-rVFCd}6KnGDn-&0?F5?OspV-jp~Ez2Y*kAQx<yUSvo;jYOYpXh^qsBR@~mh)luu$yZ+t5Iza6L?%lan+R|&^GqD!3sI~48c`1K#TSlh6<_V-mt8Wyq7cYh!Wf(*@SOJ=hrm*|+H&fIYHmwiF#{74avjvWmOI+7Zzi;%EqE-E<uaI{%MG(3Byv-K%kfqo4LGd1QgGY6AZ^5q%KE5uL{$kM|L;Ha>qv}vztEw8vf11MTDeC-}zSy0TxT8c%4sO*DT%*T6os3&`pV<|YV9a0McyIpPG}q^~)?F+|zrKS;QBn=lOBu%%>?A*sUzzb@vrj*{Z*gd6o&0{TH^tN4&+f4|{Zi{(gebdF$wyt4EG~`s(YjpZk8v*=M(R60d4$wD;<c$@Zg9V5vo}HBCd9m-9!|d&Fq0|3N_N@$asy0-i#lQ<8%C)#b`=-g&<9EQbe|uj<kxwF1usL<)0d~Mz1q!3{shL<403*z{@zp`2G)Gu)pWHDxq0~=mf~g`f4AUjSEYNwN4_|9Y)&B5SxfTkZc?k+)i4%tvOgpr&U0AqDE}vX8?##9O!_n(bq3qNB;1|h9V!<7!)%}5@T<Ms0lkO0a=hR4TG`8T$tN*?J-`$zy>1xNFCAdm-x9nAeGVM@e5b9py)Kh|)`;xZ$Xc@34m?_uzw9m~6TTR2+MVN*kJSMKxMd99zuWX^A+Q&a?dtVdGv52hD+<RN(pD(Qd@JN;<}O9_#sbmi$SG;FGjkB?kjFz~(9@3g_gtN!>*_WQyN6YK!f!;tD6;{~PrKH(XwSZwXI5{Hka~$Z9V6!ScVWN&VcE8+>TA}v&V9PJ=fe%K3e?u@WqkJyEmf{amoTXH5yh5+aMWzD{-JrRRa8M$ZHoxldR<@tInXX4FXlUX(>J#tEyoouZ#5t!?fM8??+Vq`y6KZ(?KB!DW|jJZQ>*u|r|b6bOJLomwZolOWr5H3-{nE6oI^*Z?eP9R6;Zb-p&yrh?!t<yjOx9cFi7q>dmfpK0S#f_WY>NI+_pA)_+P>GYI>iFk!X0AUfy}JYdsrg$*!SCIB`TX)Bke)@`qYwzxP?8t(F5Um|%75Ozn9EI_+*dd^d2Y!VI6he}evaA-)Yq{HcE{J<97l%VGAOLdrrNMps3<SK{HLDlc}m%CT<^!8`4}YLoh`+lfWQ18FV7_E`pb7o;y<mqD*9CT+IyB8jcGQTr2AXsGwJ3lC3XG{0$9$#-voo;dc?D`Z`n0SOB=A^b-5L32yhe35~mPD63SxoSN_0g2c1e{Zl1v=FZ;lzcE#EIL1kcnY*A5@zSkp(SL=CH}^8D0ia?cvk(*e&v~nnW)SA0MQWTGH<t=(xP%d+(UAI`P|C4Yb?GRI=CDf^-lM|9|y;QKLgu+V09n_NyY6A`$(_a!9E*Vz!}ZRqo2j9+|d12|If6yK9u@@r)`;egPh)(2DR{{Mle@)!e*#TIyMv*WjozpnlRpfab>?iq5;xVlipbPeE&U=Tk11v{Bs%ss5OiR|MmE{vhH~<vpL8M3~!M6CdjAfc3T@<kUMz!x6|`Zg(DEQ)$SDEgm;PG1r)QY7PsznZtk<{03ipT3SkeG!{Ct_QdY3jv_%`%ZoNA}gaMuIHRVckU};yiFy}q!5HG{OJ}NGP<3zZorXu%vH?~YPX$s3DFh@TeU}ah@X^4g{Yhvfs%U*Kwn&eNum<E(fYKz`p|8mKDqO2ITs}2{mRKp;UHabR?2d1-!MZF01NlHm=%_LnYq#*hHQteSgO?1_MXN8iXt;U^=FZpm(p1-ZZTZ_|oK32{y>McI9taPl^q&7biag<-6$}hgvJT$Y8qpbhBTr37gYU2m2L}DYoZ}nSu^f<2R%G$)RT1S+V%R^zf0l!R(smMQH^BMmPH&TPUOSAE#S?+&~@EZlQ_IKy!c7G9l+c24oYDGu;vOpres7d7|)YTfON$riHE#>zaMdThrYU1n-z7KQ);kk{6ECf4^$TT&^k+;1Z#Y6HQKji(ekrI%-M;U<dUd)M#tbMN0ebr)&YelTD-Q%B9HZ*o7f&>3LtZvCX&9vDM%ce=q*-ittcm7LD><z9Wu|k>Pxc=f`(SNvi`msG#&UW$Vuxy&l6f_sI&Gml#cuMqp4ctT{=r*2Zw`TWhsW%R@H~widx^P4N4qeI0yz0-9>;4(dCntCOM|UOs7Z<mNK4oOcO<sZD%GpyD9oB$tYl*HZ?E0@JDkU;>F=ZXH=5c4z^a0DEpY3a#<&y?{^NpX*z89dLJ87K}_Dmy2F<Im-aiE)UbA#sd`5F@3+F)}4DPYU3MZ)Ab)D;Mc!rNK@HjrOUvE_I0`Rwtj<zs{wU`QA(bf#|hvz`Uq=EmT=!H&ZgVaP4uGt?p_%SLe>&Q9_*5&E@hSo-4vyr;F}>oiAW5K+e8Qj$hnj%)#J{tIpLb`7`X`0+3k-{a?R?Kqs3*TU*|!?U_wicP3~Fnd)B4`0`Wwkvp}IGv09{P>K<@+|$08MJUov1Lq;36wDLa7DPN6Fghp(Qr&#3Q_x-s@rH_j04Q6VY@$@&|U)@QAhDF|Hio0fe%lwE}hixu!{{}H?g%@3KSdG8jj%(NHrhv=Qrh^5pl+6lMeNE<)?LBK<{+9*)!*-%+kP<xF!8&nBZZ4=%+nQctZN&l9vF_MmV^nr_cN=<%4xTruKxO-GRqf=uOwx)b@D)_w3$<(|EBF$f>u}r9edgns1DE1bRWNYe{Y2??J;?KG?s~P%k)leYsB8=->@)U_L@iX-GEPbJRO3N++7!xL`ih9tT}L)4hpa!FAm4b5*0wK<saEt%CfG;*>}F$FEQu6~3tAZw5`}a1AY{Mrdhe*>s8$F%KNMj;1?XmoO0*lyJ!^Qt&*J?N88E#?;k_V_1dS*NQB4j?8H^ex$AX<KDa5dT-M#&_5?k4$Xw0!<J1_6qG0pXTId`Q}#0gV0b9qJrxJiq^S$NM@JD(P%jThM!FbKEgEAfz@MYX?&NJ8IrCgqh1PBjM1uVemS&6z7^s)JQ~uu6TpASS1Mo)P>S&%nRyx^(n=_?VnRYv0TFkKGAYatjkBzolZAf2UE1AZAFkV{ILl<EA-+Wl{9@P5bHnW7f!X6s#rF1nqbnM-goX~OhM4J;1@UnhYvZgy}kl1Ckz(!EFNTsQ&jKz4IJTsYUzsJ(^VZBGOQ_E;yKF$0t%#;eeo>%ZmLm*4?Csoc}vOO7*+y1T_mQyK@??KqYetCb#Kce}m(qlUS0-0NnZENN8eKVFfT-J1~BQ9qtj7gQbp7bPR8dW}nx&ic7h8{1P^)^As5N@jppuN_=kXU3mq<=i(r)jtt*suBN&3&hQGVGntKLVbApgN_sAJ!kVFFfGc%$_o#wT!=;*Zq{3?1Rpxodn1PkyNUrMmtf4`w#xV-{JXAsxkdxb-aTOZxOa8_Ix0AFZBy;7ux8oljdb^tg~@~?OCW1wcwWD9s}q3V5WeK%<M1#Z(FmVe;uaJ7!rG*UW<GN^@iiBPSV1KPb)Bg>DtJMvaY6|wdWY(iyf_kQR&S_LY)~`l{|gSVO@Rp(&>A;4d}{kv7S3x|G?wb-{Uy#oDZv;u}wRVr`f`k+xu!<j6=jBP0XXO|2-0c12TI+8?`=L?33E@wBp0n<X07zFJN_@7EtTpiOq%DIq%NnM?GTU#}jCA$J@-bjK@S^+f_+LD=m$zUd1-!GUbA$uI(yhkf3Yjz+?VAl>oZW>GzhInaxz8w|4*V67WNpM`KBNo)-=KU@m?_^5}ds3i^pnt$nFV?!V#X>3r}jUhc3DL~C*V80?-Op#Xltc{S?Img)u+NpyNcDb-cSbKf?f^=UsI4qi8*GeZ461^Kc<?vk(YL4PMcZ7TI@wRItl6z_{%0j6zy_}Z-Y(x7(MbsAGJcPi4!LY9c4SFzLHzBa7i!p-+(lDBymwKkBO1rJ1+hsF*hSq)B7?NRJHS?6pE<>L{m7KRRN=YNKSH`maBSM}Yww-%?J|K2DDv46d<$H%<)H<QxbxaO(V))Mtk0N>-mgRGSUIa|-&9X!iU|2MREEZ#m@jh1AsxC%7=?jD!n=8o~%cN~ckdID$I@A^G4!P}c&F1=>wFOI|g*iq(&$_NMlP@<Cy6CJKp%s32(0GEt#%Waak2Zm`pq~3bq?Qr)MF-CaKh#(WiMg8<T7l#krFFN%Vjvu=NKa<L<6SZwW-s_;Y-=W>}!#-=w`y>XlFlb8dHW-Zg6bbE48#Ihf{|&b;>qWk0)?t;ZWb2~0q>k>>jp9!^5nf~c{a3t8q_2}!xIw%*;0+JouP%S_-89-Yq!fS|C714ar4XylOi=3Kc;VP5RX>g5XrDRWONNYhQGLA(lht?Bfp23OJno(S;q)~u7KdKdNQmavmU&?!PY)75+$_XF6@)pAg83~?P0v{<-rNn@K)laB*z_ad1JvR5Xd$=-WRy0qF;s2#f`f=o7(UXEgGQxJq0H%0x%)?J7`+#byqnZcnX{o=_7w=rtg?r-i9Ek8YL#h0p$@roUoQ{6ul~c=xiy^&_HFpHG^MeaW>S(yT8S146%`wWLb6Il2%r7A|9kKIY&@H0sPVh5>pYLcn5m;Nhr+wo<<}_=J4rJfId)5$0ZXk_37{_nW((B=ov*4lkO6Sbh%DB}#CmTw*H<UMeHaX1%6tBHbC4Y~UY^2{>dc4b;e8he+r(ZxTHFF`!OUM+uP^XRIE%`dC!yNDbyvpVk%c9e*)BiR?GL#RrXhatra^`LO}*xtJ~rV`#o%Ap>SFai2}(UJSQxF{f!f0vgKQQ~?3{J(F+M>~ao3~&C=M>MOF_f(6!qULM}HUvN|^mZqYYBqh%@wS=O5`g@dvY8-|4rlwEaATjsB-w7V3ArUiltW5<=PW@iTiH?nmfT#cu6IHRpCjGVBcP(`$ynwI84i;@LwkM>;f~)|S0^Y5scKC!tB^KzoiypLiS@_Wb(1x^cfeJjW*pZ1QzOMW*||j^&$K*YqK9@827N<M>vwLeqEJ9Z*%uobnw}68`4<L3ZIy7L_1Lg&n#~mTmcp`r}_K>MQs08YJKPfJYO{IWx=s<;AU&L_7%u>*R6txib`yx)*qKTooHGFsv)q^4o3kTS`TC#{x1mht=>!l_mz3JSQUMaTUeS3u`)Ndu^=o{>f^udhs;qD8Hker2fGu)_?M|@%ODij3TXTwKfz_p9>G~slfmD@4DUb@?jGTsm9Y<f7&e2yHtjO*IR|%HA{zEtxpf`K)M(3r{uLO-Qct?*w$`oR%;sBOKqz9B8;D^4PTv$3foup3e6SvyU~)FS-hT;#I~wQ^+}rJmfRd}+DfhXj3|40xxCE3w(cFo-q*@z_Ik07jB~RC9KM@hP;4qtZ&_N1^G&iV>8wP5_NztXvKPEwyX9)W7@`=h{<VgRWxXDoZ8{4WwHb@S`&+*4&g0#((+SJ9@uU_|72#@g^x?OUxFUbEFa0ekqFHvU3uu>YX=n3vrhhu?ig7C*o}o?qIn%HmApEKnlEd^Esi(7u)<6ALvzo8;&TaUs0GHS`28ORISHB*d=yXl5YU%sQo!JffeSTDRf>?X9WspLpaC7v{Qg#t(<UBfX;MMQD<V?Xd1)-FNzPWOHotyBhJ6!FkvE6BY-=Dg_Xf=c01TDRJ0Z9i|^U<{yQw`2P{V5enpK<e)B*lEm95lRmKl2Q<>J5ZT4stIFOiqnPR*Sb3OFs9$D~s<fM`1q?`HdV-YYt~8$FsHR0c~`UN}pfCR=3A<_D)SB9JZnwro>ZpS|tv%Z8KWi)^zgP9(%#C!WKfUWjzCxSv=q(OMOGHKfxy&F?VLD=U5iQ3Hp0Ps<?rM*-@G$c&B(}tTG}|q`w#f=n)ChiN2e!M*WJh{!OW7HWBL6UY*zv%%B@LyZ3RAMCQC_uuy_MacNNvYQAc8wct`44bIBDxL}zRmksUFJtv310vOMQZ`CdUlr)##==r0sEw#RqfECkS$qZj_(@^(4nokURfGX?zj+LrM+bX48I#8qW9hCOd$#=^+*35{AbGbYMl`@P6)bm>p9@t#G``)sZDQ|HPpKR}s6#7#kX1^0*^2T|iO!a9G@h7(?>h|V!G|4%PAGqyDfZB7AScjpzvxeRwX>Ysb*ugnj#npavLdoqaO<|I6c;j?Gzljw3P8Ih73&*NCl#pXUb}+G4g|?zLfSdAJn-#)2CNtHH+JhI+axWi5x|ENK_ZZ;ih<mka>1sbn!Q1GwdeIdR*?*+b#~+^4EU7-yO0B+_BZ~!pyDMP#7+S+`vkOm0%-jvXw>k!qwQa3Bi<B<^bdOIl1qUP8?C8p839<6+N@nr2Q==i!4@sx39)RldcU`=+j#1u)!>O^HoKAU<Rozdt;OgJvMG39343@Q-()r5;{L-j(Jn*;B@NI3D-HE7dHCE-<9G;^QfkZECQpPQTt&Wkua<zs}KgGAvgP*4QVojOM?6fBlaCHs4n<KV)Xvv#dyEh<|ZO`mGC$4J>_eI5&XY;Lo@0sdHv*f3hV)GF&{;P$_c+%Wy&_g9(inhpCMnB@z&Sku3A6CP9=%$mA^kA4>9lF0Y;bVR75Uz`@8*1}~g{N<~yG!$VrN%C^kmUt>?XOzpYVydIEI5DOuBXSFe`)|dZiKOLg;?)Ojbg-_48W{?W+OZ})74J$JYnFzdgkXeFw^$4oBEMGChKnjcGAc;RlI)x{Bn=~@?%k-_kf^3IH!&ajFdMf9=TxfgQ3PMVD;IM8t8pSI38(#2-1W3t;tb^882?uuC^zrc-O9P4s*E+W-;XuA6jx>Vh@(~=*laC{<TO9v$AR9j5|@7!L6Wq@KiNi&3Y|&2jSWEt7#?aeXi!i9yJATblRT9N;f;7!R7uPk%rkCqT5h=Zd`97bM>IMpv)&&`_qH{=6Q22m0P@fAX&-pP7=*`^G!0Gc)bP|<Fh5)Aj5iUz%?x_zsaH9ev-SBYba-U1RnWf4-1v%-9L*@M~LRz1gyw)p0udnx!kZp-ESu-?gyTHDs{1ka??^HgabF2o7z=n&pp!0@a^S}E5}yKGJ^aQh$b%s&-wm2Z{D%(P2ZO-tZ|^7-+rFoSP**3%Bb3DkJ>e)<Zja!*de>hMWo2E$6Pc1edV;~FZ?|!Xe@YZty#OjH}`R^c%Ir6_Vd)kyi1h&<wiPcpurzPdY5OQw#iu^neDFs+S1}K--5NP)9Q)-3{@!v1^Bx=tSBW-o$kC`=^*Z>eQWSITVc}rn*N#z&q|K`T_k7-oNnp`;j{0-Y~O+NcB4Ab{95vCeal;MnfdjC!kp>0oOuaBCe1U`4&}zeh$Lbi5#*ul1C4o2{Jqs_`&o!PLKrPOSi94mg=?0gc<soY!*;1CyzCelYTv!;pVxwO8?3F+N4n1n7p`ry?+;bd)3|BG_3z?CxjSH~PiiN6Hx9a@aoeJ#+<g{gtyig&$0EDR0)wqex1C$v?rPh5whyFr8T%iAH7PfAGq~O+Vh~|087%7WqTnRbRbCGO$9j<(n1(#X`oN#1f2!dl3L*4V55~kTs4S_jxRl+`j!P0~acvL2N&9+WN1sYC!2tigAdGgn1h7hiN>_0WB5kMGFP{<b9$<)nM;#*Mur~~CpuJ!luKNd%*gj6UEU$KXyVrS%Ubp#ZoP9jC>CZ}#=xe|)jr@4#KWcXx{%KufK2v#UD~yuS`fs&%L!eqPz>pcQC|NNeVYS}_;c&VLrmES$nyUfPau2P9q#hp^SmS^Y+5v;(huZf&dMm~Ikgd#onTSm5sE+f;t#vV^p0Pte$e>A}5N1&2_kkQ_C&PniG-#YE_$b$j)6{s-g)T33v#YLIZ#{*Q5oROWYl<^AN=DxE&0W>Tik^S=Xg=Poj~BravU`|@mh)D+?6>w21?kh0;%%$RzUjBUhm|;m0Pb`fQ^~4$k;5Dy$MG@HK*xj`d<Vln1K}Ty{j<ikt69I!J=FCb{RPfp4IkN2YpVC*o~O<BrCiUGA&YM4*0?@Roc3h7ZyuC<<As~abOONqI9@||*6~dXafxQkX@?Kp-Q3o8v^~1?bPV@Sop-i|l|r4w$+`3057VAXgTP|nC+u~8k=@tK*a75#ECks6qlHa%m+#_aC1g||=fPkTdkno!1GIjx-@c2*rv1fyHiC9#(irfOF~v@MbgLdg@lep=n{Bc>xE;wAuy=^I7m3`sZc@L8fC)sOlwzpGTBoaL0Lr-Ym7HK<@mY@Yvm~(f`i~-N>usxUFb(Vg)Ah>I_HdgDp5kHR>&H3(^4QI*=j>;H+7HfyEq(8Eo#{g@mjkCaBm3LVg8;r|P@8?r=~f0mw^x4oaW{}FKIk;5uaAgT7+?7$P$O@2iN0kWoL&?`m;>uqN9nUm1ivI|A8O&2C@k}MxMUAA&s=J1ct!gwRGn?tnh+$P`iR)hE72Y6`kh%D_FcG+*0<HGHZl&*`+-ya!3N__{$JBl4=%x_&02n;8D!s{zup6}E^Zqi7ij;m*RaE&34(+mkGh2jy`VGRzguV%gb0H;J}zxytk{2Rzvl7J%{9j_1fqeK7(My)D4#OaqQ$I$I<3dj%C}vqfi{rWq4go2&OLlq+toFtPxwx9%=v}<o=p(XYQ0o)WSF}w9*)S*@SYSo+W?M~U@hg&YBT~-l{rPPesV2(6E^6(h;&05y)ZB_ZZZGaT?tyP=^U^NdQBXE<na9C5h(8M;>8rSYZq>w39IGY>m5k~(fhS*K+?4<&~L3Ph@B!DxU9GI&p<!1v-8nsuI2^<GHOM=q}3AZ%X3BWQQLpYjEEyMNEvupQL`ib)^ZNA#I3XN+VuSHpM1_I{c;7^je+XCs>AsE<qy&y<^|U}Q9U;O^cYO8O{{XBK{8L3>rdt8Y!~Gm>abIpvBH!(8;hX3^*zFTn9<!CaFs71y1ygVRnLd6`fj`y*0YinKpMmV_P)+`a2Maq(8+<q;p!j$qq_%+Bc3^~GMAAQw%)*C#$?7+IIp3!(F6bX<K(qBPUG$KI~_O&4Omrd`QTspDSYo)HE!~TwEV^H1h11{YObX&A1JE`t$+wC3?V)YsJFI;YF(FJBsf`>?a-lHbmx3MLZd@wtleQ-#-Y{c)Vp{VHl6{Hho818nDl@tEc(_DZ_1aUblpwsi-^DxF{I{8ywi>K#pkw^zGgBXGk@7{Y`Puj><!J<Uxz8n5z7faSwB2E?t4%vYt~t>u#)wg!gg*^j<;WfX+QC^%xXX8(Mz=lKd^iZ!@i_ePnE{w(HX_6nqkRCfg94ZeplJYUTw!~-A@-UaUH<~9A2)wHLzO8?+L7+w}x*K+s<-s?Hi2eYp?wQOAbU|3HR>6?EU`ceJxz_y!(7+Cq<o`mFD~lWJfe@(YKKkjRn5_s}=6eus7=Rg}!IM8F@Uc#?@8phQvGKFVVo~4|7NFl#MBC@vMXk7ltNXBX`B`^2tN@)z5f%`+nD;-ysMdCnNryUykNN`P)B@`@I{{Q4+U?grVF96^=N!|5(@Ucw^?RVehErJ*D8W0b`jTD!&$2kuGP9<Fc5Ejl&|{X82=y<>l_~o%XBp<+EM0*Hvw|ECcqC948{m<YRXrY0gZNyZ|r{C#TL_zVvIncMruUE=56EbJrwIUY+_^A@YaaG!{nV=q#!~`3?E?UhT_WvCpC9SHpt`kbU#WT<LSss86wpmIn}nWZ!mL^LLFUH^N~Pm?ys{c15A7cEuRB-P-4{?5&C`U+sL9(Zs4;qnSN{m)Ig*V~xoZ_2RLS8N!5Ksam{k>uYd7snof+J#OBYs;~-lgP+~L1AmrcvUaR#cE*$L<~v*Yh1AhFgN%5N5|(nWbaJ^ANUCu%p&d}xWqpD>^=q*esqEv(E*IR;ri*}I0zT%-9$D=Y4!Ij<1MjrW>{W*ob0e{*)2$WFP9LleUJsY@o_0MTsFc6=<-<?c@Pa{rn!2mg(>H$unI&^P^-q5<_0``?_Soe3o_6<%;ul|w%VpN?X*XPL{>)JsR#&}AKhG`E>&ro8lGEnI3(f;T!?-$Hg1VrT$jQ2J97cGJ#Zc=VVOIj{cV-Oxs~42^^OZ2De}*YBTpVutFD=nVHt?pK!}~%*_a{<1i(SmTt(4%D@*Od(V(@2&4)%*eYh>+3fIim#VD$wGZ6^=c^9emqE84v9FkY&`p?sW<f!6epCGS!+J$3ASANj7{_wcY0eSbhg=WxZSxb|njiro=^r1Q&8&#k-duT0^PH%KMja{R9g>&|ER8h3yulrN<1zJD@@wZ#?Fp5yFyX2FJH_hh8Fi_dE_xP#4>Q3~b2*t~i5;*0#>x?tX7*F2_d*Hp|9AD?@YI;K-ex!E%uLx&>uT({KU>d2ca0pkh9e@4m_EE--e9i6T`{mdVriwR5}oE5LfqMvuQ4QB+^;1acl3Hiw{C!c8oE1zN1X08c`N`Xb!2Env<p*z5}zH0yYws0yN_2edPgqq|3Ui<l4{IoH7ILcWz@<i}QiS6eFEj}Ly@eavUYc*T+cwPSGpOeqeWIK7r3(VgOfjzB(>&0zRb1F4Osz(~UOy^B%CxVsUa^J1mqdO(fMS=3V2M?0A5U>Fo3s&@p5xG|9?pDlpe*fAE;ZzBTa<EW@nWvAZtkf;Nhs96N^RyXlYKQDO>VfFP`q*k#t6AQ>Ev?QYf~t&Z0UnCa$&%PCFKWz;M%H0G;|IEZmxxgs8j@L|X}ie*&5Lug-IVqFz|NeQ?7kf<)yR?ms)(_;4~vM`vU*%i$lLGu*Gn&cuuW02x~dJ&v_CjKwp?SbpS!AYTe>q)&MxC#c~}Dpa`vwLt@}u!c89p@oO3!;Tm@^!<L?A*+~NlBHB`HD>c*ejAaB+K6bzD?-+0f-P<+jsh~RX!$iW-G`EB|EPo3-2>XG?<b`urh4W112`)wEp(c4}R<zGhky-iDn0q{N<?=RsgDB+5^Vd|UVpmOd{?%LJZ=*y<VJM^cbkAz%(5WiUL;t}QyiR%Pz4d`d2OvYn<vI54{P8nVvJp?7`)yk&99nl7IT<mr&gdn;1X^Pvk1cCbb0L3R8+?S`pZ&OUzLAKOj6(VWpM000Gn69XL_eL{$Wi;Nj5Uu{Lq>7T%-zcdrbAQ;*oWz8hr<}=6;42Pt)P5(9<AgZfdNh&MDcEqGtWkG!Q*pC9f)G+?Al%2+O`13Arx9dX{yak)XOHo9cTS?x5#O}as6UoaCUzUlr8A_*S~KV@lTR2QL%~?qt?3jwHRZ*7cekWp06*ao0fDdInrI#F-B@*#<L(|~2p^}S1X&OaDK+Q~-)>T96VRo3-HQp?t<(qbgGe_m_E?*N%(Mp&{5|+5EI+2Gbi<Qh8$P;!X-imR5`nQR%wTxQFb%+&y>?x8j#Aandkn}MqMlwVCBJRN^CA3-KeV6sI#nHdq}Sp42~1x0oJN0Cb;DWHM!k4eF0FWI@p}k7{Be}hr@($)?v3Z=(%m`V<FH7Vo2ExkhNJZX;r@tYgMB_k@NNFFdv({;94UvE3J3v}M?8rJ$RbqrhbdB{7rt#~R0E%K-?2INaMn4e>&HhRfG2R6)Pnf3YCXc%<U7Ak5&qE2T@TU*5p*ay5?=rC>1fx=v0kt6f=^C9m2HQH7pH1+nH1r|CbmJu_8I$rtZJvGI|OFghpYr=qE-x%{>>s+%X&$yz1s-^lfvOuV|g}`;~<M^x89w<IQ0p>GV!TX%fCLnz(#$9o&u#cLf;(IJpo@(`ML-R=7CyRugZ5en6xiQ;|LhHXN9@r#1^<wyXxCIyVp!SMETCUU;3Z+2%H-4`u?-|+<PaA+JAeke)uDgZMY*GZWpr++IL}l5<0cIkW3;nj-*2eDx-mYoZNUkTSm9Su28)y0cf*>S<#(6DDT#~UA}Mi?PRm_-rZJD92~pFF&%|p)e>8qc{3%t(sZ|(pI>p&FU2{#t%!=ftu1Z&uBg#;-kLQt=CFS`LpgdaSN%NIEAl|fc8e1bozhAz+c?H*#k99)WZM3#+nAm{4EE_-Q?IWLuy1tcGM<h&@pkQL$1_iRaU8R;(8P}Np(pk^c2eyKWJ4TfWIO^9mma<!r~t-I_1W7jrq?<8+mI;^KkU{0F(WtmBk|_b<^A2O>m%)b^O(=|K4`;xpVc(5vSgN-?<BijJT~2R`U@9L7*<~OQG3zj>Y=`e=;!o4E4ggvu0HtpT9P@lF+bpWFMhkX0j1ZW-(cH-)7x(@nPE5}PTh8%b}>l9YaO;Qx_2X0u!9K0;oliLjaB2aT}p(h=`G%0g5K3BpYL#UT#Y&{Mf5H(!BgQgbPtDnHNPx&y@D_$2RpIUEcr8}?#C?BQ|}zB`{ZgA=f1*);%yW>ry(|XR9W=a$1r(*P~KLn#|HwHAG1_3V>THvy;lJ$zuUm8RTeDS^LjVDQZokq*7!k-=uuLb>SRQT``4`igR#ua+Eawn<8U52#_XAR%&o=xtZacO-{4OV$nfTWWZ9K#Mxv|zHfOiD<0EWWIh*j^{e8KaB$w|@v@he}ni?Ad=iJvE29nqKvV((dQ(x9j`(y*WkcQrUeUlq{mL~`E+7gfnwm*YcLksTahw6k=Uk{m%vg0QcJ>4O9dlM|}GBS6@BAPgBjCHULM7-t|?tA^(%r`BNB*kW>T?YDM!iD`|Zr<$gnyu2qyuPW@_dc`Nq7otF)x2SCjZ^!SC8US!!U~7JcPBD$w4S31F}YY*OQMc+!5u$**4bT7B-h7~vh!A>b}5eGRfwb0ihTmZmEX4>!?T3TvGIV+8-+2QQ+V%iW6ynm*!sKrhPes952)yutT(lH#Dfp~J#Ddz;<Q23>fqQOP}*%e8V&t+fGt{^&3-X2aPls5Yu?g;F2F2U^w@b5HH%La$=tSxlg>`>ZJ-+LTCI&~Ef&P9I&D`rAH5Hci?qBTUoy_uyiOCYbl4`g`9Uz97jFs-G?XPeZWl}Tg;s<iWL3BKVg!WSqhn`>;AZ-(C|<nl0tIhO7{3OIpuer?wk^^e?;I_>rhnY4LjEQn6spS5z<%1@O7#n(0!thi^(XE+m2r@P-{&ZZ^sK&F)myCa9!<Uww+f#Lq^yhe)X~(><*qB+UZ1MA>gREeDzDxOUMV}9y}ChL4NTk<YQ6on-gS4ML9||>*ZOZimMmUfV>JM@6X=1%%}3S>Q=&meU!61it1HE~8DT<H-UDNYg?y-WXCd_Z0XM*Nn)5sn9j22D#8lpeyV=}_7oG)urY53PVn#-R)tTE;ZSF9f{w<dT%H^Y3o%O8|_Liz|znqIcqh?GCD_wfbYc@+&53d(`NW7yLZR@J4%H1@-yq1O}Y%aLmZZflPG()PeGS|LteQP}R>5sV&@OMMK&Y5i;Yd)c9L{7)hsej$Vd1db(4Z3>s*Vx|~+605ckV(hXJOk9>4!a)LDVe=I$Qz;;Z8m&q2?N)Ul8!<R{QVev-?Dm9KlI^w6gD*hl)0l=NI4x;xowkb)rc*0!iK5~w0;P7|0Abo^yxKj{RgnlZ%W2F`vz&Or>vk281Hs#^o&3&+(B)11ObLhhT{6te@@QRqX-6b>DZ4Stt**?HJ{)iQ)E^wdC`ApRHZi`hpC;tWO(25)tz@a=y>OXSBFnb(itpd=B*EsaW|uObd=+`T72}y_ClqFyGtHJ=MA4WjRxtU*mHJ6<HLJina+B))=aVQ_D#Fs;NvzvQCnWgY5U(3ZhjA>Xq6v8K6Yk*8;qm1mS18+1Dzjk<9Eib_4Y%4PE^$4-h92rSXs}*#lyNv$e(4{d)$k&Gps;mFR_mjXcO%pSy}5$5FWS0WvY_TTCW7g%W%(Cwye^o6h@S9Gt+GrFa2`^`e;@2fTVgH_pE$=;i2M4VIJ~MPa5Z#-ljI<E26PVjX{t|MyFQv(}!46pt!3lQQv2furJNmR)6hpnW$%i<IW%0_n+5@*tYu*l7{eY!$<#!w=$N#Gfj9oyL&5S9b}EIOKeyBTsM7FfuY+2y^c{9_U8AtU3u3(FLf*w6CAFuTABjUa`u=+s0WW-kblp2bJzF6oxLGty*uXF6mcW+-KnT?%C!Iv7;n;rd#%FaWjj`C@dSpc^6uA)AmTrR!6gk<xzb19J+!jh)#H!2$w~cvJU)`bx)JfmZ+6XZ^MQ++RSEM*RN@YE)W_+h5JzSR?wSEv$$UL^J1S`PRvCGEua5F=Q)!Y_{_upYVDqT<D$K6y_zU#?J*Bte2202BQoq@U#b4cSEOscIfm?X2{mmvj>qWD<OGl3voXh=r+<ggt3PQ__d7bl#FRqQzecM)91DQ^=cYn1Iit+9|Qm-sKu9(GNdxS4@zm&%ayfdi#`;b>#8;TXDZ=5#=D-3|Z<E{}`>RG3*RPH}+PXx=y&D}$FOrbtYCVUW$ORau=a}il|2L4HcySTxA$6vAc@PxzP>@Go2kyBN5U+sxa?RPI*d-&(e4CPb))D+Hp5JB7-uQKwW=-u6Eekpma4`>=ji{q9(%nmp~s16Z;WK{=v%x|nNQ5OO3*y^%-Ktvg`FJ1*3x&OSSf0PmZevI^d+x+Ugn~lP*y*;bmGvhAFR$0;gqVt|_u)iHHcG|lM;{tDcPsFweg)M?v9$w3~y1c6F)mLQt9ucQKVKt#WVRPc4Q{Oyc=IN#M*H<Mp8pKi`k`upq$cIuG^-20s(@wJ|-ev~_oT!Ij8|(W`?mh7DSa<TA<78}**WuQA$HGm1oe&?n1W0~4jU?yk2etNM>3kO34uny8KwKG&@{tmv&+<aqw2WqXOM9$tZ`ZR?;JS*Yu;ZlC>-<5U{^RolumbZP3qR)j&34p}D~SkH<D51w^3H!$sifBqHf{Pny6;|tZTBRtYCw-gcjw<5yGF74Yc#7bHX5BV*{&EZ^H^V1XbI_YkJ$Rnek&i&(L#i#%5!~HeI1Y+&YYIz#*js`L9jW<OEf(vyScv}G4U~e(f9K2lVW|q&CH(A@VR1X9V$AcGx6p5Vl%6+0g!X}3yy1iJP$ykCjH{geL^^$<lOUr<@j`0>Yw1nS%oU90adc~4lt_jibH_y6m_j;(3Sn8!0!xxs3e}a{B3brR7Ss<H|rkG=x17M0Cj1DckAYnd>w9WzNXftq!#Uhv?3Ra!K{^P^MLOj2C%X4_xeOGG)p&Pb5uJj!nP2@H2V(kH_=><s_82ps7q=20iJB7GXmr5kS*=@<Gtxkr_m_iDp{c->oSMOZ)Ej>`&ne1+-|hQ&b@m1dWHvWb*y7!utx*EtLQ&B?{WTj98kEu?6-<6o*rj+fBu-)Z5Yl^S*=~S*i3P&u)FrgYjesTi(aL!v`5ZM*2mRqZD|tC@#Qz(o;%E4qWVWAr~%?2$?wlY#4ss3oEm9RVOo{VaM2(n-?-!xsx>WMx1(Y}ov>p;Z$CX9%FNcBUGc4#4H^WSBg1H1E(3jSmgYfmM=n9Xcv1Ezf0wEa{@0uLP?NIu+!LPcu&s|g2SecTNLR(%kFc0Q>zlFPXniDhJ+nnz8~JOY)+E~bH|vdpE&1U^@cI5i>n(fE*3JKS%lqSltM1X@+`r72HXPKMx|-~7@nt?%K#kmr%P)WL&NK$Q<eUA(_>lh|TMGXP&~1U;da<(2r-|f=^y#X6$Ue7T@KDBgitW~}fXN4Z=Hr9W(P80pc##vK({6m`2dh3DOyk<-U8RdZTsZX~wd?QAHb#qa>DoZLYOQFbo>G3hs5=MI%O{)mRdj>tBzQevmHv^Mvc2Y7$nMq3VsSh2)s`gem+o^So5$GWrdSW@<_6@z%r#FcxZaWc+W1klV~*WD_b<bO*x@82X(mNN2sG=zIvw_At?TC&uUhZXvAaG*Zo+M9F!mQ8tUV{ge7;DtCu{yLyYf%JlYmUMD=|<WqFMDcNu+RpJ?A^exEybK=IjaZs{NE05g{h*rA-X^niDawP3WM}KqPC$1H=s6nFsd04?h`Wo2c7APd+blb~^+G0yIX~_6vWpTB9$c_!emz?a%(K_Mzc?14}3ioIgkC%<a3HG*%gZR8IdCmdxJ%JhPPB-^`o#RxXb=G$AndeS*$Y(lkeG?f7#{aaLa6Vp^mseE)46VXqk5{DAYFUQbLni^*@e=|}IEU4QM^X*yjYe|5S?I|n)UaF^!&zSe%`(sCu&sLhCh*if0W@4GW0eoyim(Yc<lKQpNxIjh4Sp`mk|PuT^^#hafvYYzmmSg>Qr!xpzGG+OTaFM|_~i<*B6i^=A<-ovm4Z?Wcf{E%zo6fb1;R=>tn`gg*beet*a(^^3@1?X?+8(m+_{%kS__8y{CrP=Tah$vT_kq1fTt5~s%JzJAxO5fy%2@#6qR;M3rYF3!)qq}QYmoQ&G`t@O>2<~*z$M!EmERF4Z@|zzJ4Y><&S%=b*G737#Gy~iBh8zq{=veLosN_G*-ygx!So#>Kr^#yWp5v8PQkBknOtH81dDy7ECqjRU{c#^JY4^X3tWp_J)Oh|@hHaD!#;SE)@vH4s!@^nhV@~>(zrF2rUtLJS_deO<gJbs?FKOZ=Nn%ZJpsgQt^_QOW9(;7*m^{cmX+URu`r@Lwc{OitWf@#A$I&K;l;8b34QQCKf)()ToDm<0D=$bQkymw)nOBm{1BsmPw{-Ln?aB}shE(ur;X4XwNiSi&O<hQ~3xSKi+rNwUUX8d-h%t2^?$hX;Wd7fM-(Bv|j_)LW)d<y`e?~3((x;Ezlk=q8NKc^K4JH>S#*krg`|{>`{n-vrNev>E-C`3PyC;P7B(uJ5d8jk_-soL#J(*cZZ}$!QUdk^#wdY)@>sN2qf+6!Re-ll*eHykX%$3>iIqII;Z*VkxtsUBH9f#U*xUD+omz$^3DvbbkzaDw=U6<v(9DjeCw>`sW3~X7^yiU8wALmfF-YyB{`|a7SK<x!O-QIq$#>Hh}J?R|Sqgacri3D3MO>A4AW!>MG3c3?>6v?)2Tkq)a8JNI~)NOBS(=D?dp;tZKPdOW1kAD?)U@OF}Y=3jA)j>B&$zaG6y8@4Cy*6*$YP*L9oXyd)9SUb~!ZlPwF+J2=j9OPV!T+WnA@_3Tee(c$KJmUucv^J9M~lzUyjIAZ<V$b1V+@g+4tpfRow}`;I}|#oX%Htk_*!$bRTbCtN-2NBV7uEkPgJuabb{WT`Q*TwDu7goq)#p4;8s?FyU-LHwI`445WkMM`*gP=tM^fF8==>uRJX4Z^hEm~I}dJ6TGg^L?rn4lepZ0~0@pWPUzEzt$cwxs!6M)3-M{LsTM^@TX;8nRzJCmJ>IH%JCtMsFjc|2h5_P)aI?d0g3!Ih~AOl=w=GbD0Qh_IXGG5}Hmy+J-?yANs%=`C8kQ`!E79{I6I>TK=<Q&|YE@>XNvb^_9*dZpz>#jP~DvcLV(&(Ff^poZc-OpRg4*l*%*5SF;y?4_i&0o_KXf(<xWNOpD?6M;2t-b->>i5yUrh{GfyXhyc+Gjgl8i3yKORY+u?-3;x48>jR_JVoMXc?DrEgYN=?+y0f)cm+-*hd_054&~-eU`t+Li;~Bozp@KXlY|n+`&K-y1oxp<vu95x@A0IURTf1rMg_5%iol_uCCJPlBT~or`~Gqqv%$5!ud}WoT;$k=H7Njto!FxCpk^LlTVb;roQ9l)4HtP4!7I_vC*OwN0n6QZ$xlb3q#*Nz85KA^seQ1x3`0{HWVaoFaCw_XxC~!utAQUs$RtOJ^jb}A~!AEde9iRYKE&<dqZYvQAs>nnw`sQdKy0`##o1dVfQAC23fzm=}@Q96>X&GAs^iJ?ok>en&HUmn!b?F2URPEr@0n9ipCc6n^Na@)Vvtass4-8IcRyu=OHix(nV-fr^ycgTu+L*yzTbYB0Rvw^A+=tBpCJ?R~^II7lfRte~NrRn3{AW=3yONg{%9GU}mRdeb^-FRIp?7Ivn5!b;^6IR)B05)w1ko-!2FCD*m+yXW`)TWctak_En@{*M$qe&s~qMc;<HbCr_0mT0)j~f4IXOD_H1Td#?90<>d>u?25OI7XF2r-KtNh7g|(WG|A!VFnb^0Ngo)MbFV)`%JXsTNcuQ4PmKxcEGvsQ+LzL84cW_&vf6Hj>gfJKPr80PFNSJ=P<^J0ZsJeSv3qm_TqwW&N{^(r7X*i2nO=_DQL_*m8@b#lz@AefY#nk#a9H{HNprjDRBnCNB&UZ<l%9d-dV8~<TKBXE->-p;zW2KW#;Bifc=}miCZ%@Ki=FvTc22eAgkS!6$sRSuk&U&YmbO&)@H;FxWQeefTE+kV45q@Kd$bP&)%F_gVwkwM&$ne~ay$eGH4e`0Wm#4Q+}#OvP0uV17nJG5-yX3C)!oViis>Au%y;PG_AmN5a!zh(lfTwX`bJ!4L0z`+4hd}RKh7i1vA39CU7%IhyzTyFB9*Y8H#Nu>N8>JN>!(6s=SyxSB{MIZq}BC&8OuOYziM(Z^|R0(t@O)k*(%Tnheu514Jd=CR`#kLL{PQ*w0%3WiIpTHIXxb#H03_xVE0+=?f$^mp0y28`(@<DpAtu*`q$9n_}V=j=iN;OaNQz+PGF$iTOZP{jjZK|BI3*F^inz35UklCzKl_av7}}tBm)h*_>U+HlY8#2<UCn)`<ZRIR5?gpaelfW#6_x4e8Eh1k$|*=Syvs8F87`ny5&RlHn>7OVEO&KeL4#Y`%9MzpY<)qs?9*OAl}yBx467k3=lRbM%wh|GwO~PeXHWy4~bu^kGb<FO`Mj{HSm=fhvu*nPUF9z&YXS`oSQSpn9Od4JwPrQIzYOKHtU^HUcINwaT;nQJ`h?J2dm{%gA!Vi4rQSE&1Uz%7F8MX0S_cIX%yqD3D=UY53Ca)=qRjle0tCUw!VPG&#6OlLPmCUlCyb+gnfM*J=KV2sUzY&>S*B7FV^eLQ6#O(zIpI~#X|`WCCJx1;h#d(_v$n_n0tADxdzIM81T``9$uXxQHhLdh;?iJkE4e!kg*?UCD?~S(r}0C?yfu155$QUZ>hJ(A;5e8sbY{VjH*7-=pH)F{_17gUkI?Zr~p0+3RCI2SL*tEZ}Gos6bFBbfULqf+2AIfe#2SZo)1BHe2ZE)ysNEEMDEzGHFU-e88b=VUXU0jF?JL@jN0PbAvikVD*GPm&WYxsMW?&PP3usBimnHxJ)GQvBhi~AHL@<gs~-xc)w!9UmjOgxpL#sF&x|#Edi})SuJQ8+6wQSm!}V;Kd5zsewrFo$X1tUp=Edu)jU>$X+z#J$C(IV(B$nC<#iwPr8elEjVs2+r`L4XQN2zhcg$|ExuzlfNXch3Z>Dcl`eX)zfx+J0jmj2FJoqvrx@gXo<lU6jZam7lyoaftgej8$YfbFbIbpLKXPhJxvUqPu)+)??k*X&>gw{Cav`>I3MwcVH2Ogw7>F4`Ofw8tAxqHEbs8Bkz+e887cYXU5tvTsgnHN+B{<zZ7)Oj5b|+^m?jD|wpN<n&@GU-GGU7Ozvs*jSzA`hGq=dZ!$l#6#3vy?4=V-&_o4i_S$#6=@eq>%6i3%!w?g=Z1Zq-(T?`>`kUY)98&3&gyY|#on)}Y48<ox<dG<w!>CuPw4Bk_LW=KZioHF<!1PrBA*_5>AhYvgd3{P&Kqv&r<K*>wzL>pKD7E6qxEWe0)^j}K?YUbVu+`u<?zl!Sf<;pay4)cKE3UvCT1PrySKLO<F&*W3ErMRg*|FKYF`GbToY+p?S;;gG7`TQ)L9~gl#w>y%>Au~Z$U*`TkF=u#J9HSEw>Bk1zjJb&7sx&-oC%;rK-&451khjWj8x8hb=SjG^*<?j?TeL-X1shPO=5q*^y6S{*}FnZnbx*#iM>LJg6Yohdm>U{mKfuZMxef1o*#d?R~cW0XAYP@X}c?skL+F!ZOsa2Y#+r`Bl@e)s=&-33r9ot>x<Q4jm_C4?|Y0W9;uyY0v3i4NF%0Y!g#Az3Sg?NUayL^>~Q~;;@J(^hpI7HE^o*bg~e@X1$o?wHM%F+w&!lk7~GY0?4n6Nez04U%5qi>q3z*JAlYv^kI(lFz3pgT3Kf0{k-b-TdMY!*R;LCN2|lBLt<~}t|$8*;&-VS!b5GHtoVrqsk*oSHQ;p9FB<jLXL2wU>2>~xore>|oz<y!-rk1&y@K}ZkOKXI?(z<W+3VgwLS!;AkEXMEz3eOihUp!5k}4+!<DcZdpGwfJq_oyL5K+FKR3wWXod5mkQDj>OiGR%Y)?eo^KKglw8mAgF60FGcek4&~FuRiRqTe)+AhciZNyX82eSzK*pk>cC`Eowq94g9#dp#GOZI(V`s`Rh-x1&$~J(t~uFTW~|5<&g2q+OSxhBhwe2eLX0+xsYOWRBMBRd<EbT9A4LwQ2tOlB|B5_T*L-YwoB`x~aR=wYlNkp1P5Vmp*)rg?nWojHVGHqQY(>jnKHLZ{=RrY=ezIlw7bLKj=d4dr3&~fR-kM<TO-dh<H`zmTT1nJx@cg%<u;jh6218u+y=B?Bcp}8-VDgT0MSI^*f3T{M7nVM@+sxI)62F394QHrSj|Ku<i<n`xrm3l9c6_4%u57SdSN?-m!%~Nlnhk$FZIxR<}sO;*>AlB~M5T2cHwWeoh=N74T@hhzC%)2tjb#_uKUSovkRKw+Ln-T4d_^m=EYun`Ls3;|Gp5G!2z&OvYc`|3s4b?8xSXxCU!7WX7M&tv?@LRoG_)0bq7)Uy2uKro$~O&H5i|g*m}ff9QApmbT??2eU^FuWea)YhPD$^V(^eam`T&+)L;tpH&MH(~kS3o~P5MMa#kKo~IryU84PJPtPXQaSu(C_AGbiP0JAP1b>4qa<F&o#cTBhx?yiVM5f3<#UyyXgR_03KKp?9V?5l=p?aUb^d#$b=3T;0IAJ!)t{al{UK#f*cz3rY(uz~t>Rr96?G21(!l2pTzbXBamu|1k!O#Bf=eKs`uQeiwPkPaerKCw8&?6V$^*Nhf2Tz(o?cE@<<x2gqb$FukTz~~>)?(qdBdV|4bhyB8G!aZcairjW4LGwW<{hlxQ&A=HKD+W!=yh)!>;8}VYmLQO{mdGR;Athh)rtgOizi<Zl*OLGNgvm(ohw8XQ$_<K?%QNjz|aNyhWxYU=tU?K<=1p!DsymY?5wF@pQ*YvLCyOVrEKIuWy3BPKnKU^x+^nJ_Z*Dp*;8Jw_WriBCA%k7($(bMnGMdd$l&uB;QCH=p@c7=km;_mtaInf<egy8^;bVms8%*=no1CC9K`4`f9gBd^h=cqtff<vt>uV#2`7JtCZaEB0R4MkJXVd%;R&XZyW%X3!HnOY7?w9k0&_3P!}h4-H4z`ap3M*{+6TJbt6Zmx)@tVU6!_|mXnZ;d3>GunZP4qB^+Y0IkO(CB-UfvRv>aR-Eq!?U+sg}Ke(fg1+3cQ&wlU|!tA&hmr5xPPH>+BY-mTTJS9VGpdKo*a$F6#R0v6fTDt7*>o5ouE9eZz8xwIU~R<!rqXxhKotV&Ag??8UX9XAtR4wi|=>W`Ot^aN`*{iJXM6a}ewu1A$Xd{U-%2$6Spb8e?<NADGv<#_4+hO6Op+aa$&ajj~fo{{E9JUe%7&Z#5%OD0qBz0Uz%4uE~D9d8cJeDsV@QX@;$_A|8UZl`)dlZa_($_g*B86X%o{)&d1F4Ebk8_FcBOZ!Kelj?WIJM+rol;5gZEzzz=Gk9qye~$>00J8dvsNQ@8?Uy6leB1x+mtN!d!NbmYg_Qob|8b=&p5dCi8IPvW0Cx1RS&4do#@DKc#dZ0qHLvFR)=v4K!u}nWYqNf>952s@7NXIu`RIi8>SajDOH5kq!iBSK`}PSjE>O98>B{^a+n5J|dj&gQw{QCoobYZL4x6(idm6(Ual|NZ*a^pvCvP2^U;Zy)xyEXAV#1_KLF-!=t!v>rm{1)PVEMtK9_wdx*&Vcwzq|v(uSL2qGuj6y-4QRxT;wwLO|5*$iYc9!=?Z`Lr}p~ZN>*bB3jEHn?d>ZaXX9a&PW{{WkkJv_k;Tb{Tf}4cjfhUs!%j(&Tq-EJ-MDSk%TG_Rn|Hfh_1zL?jSDrOT-xe1R|a~o0*LC4PjKl!9Gz+NdTboUUrQr~8Z$+<skEttv>??36(Wg3*;-!v{rlfftEq80-*Y~PYZ8si>#{n!weKlJ<+pS;>F(-YG6x8gfRkk(pYnu6uWWX=)2@@G&t4y&fj2DI>n*mav=3=KsJp+;dbm!Z#8#EYewHHX`H|@{^L_S=gF6m)6rgo3e{Pcpe3JQwh}CX`PxED*w-Ua+KPOl7!`|=udG^O?OKhsp6@$gc>xMPR869LQu0Awxu~ZGbb-3Ka%@}Jta@Q}5BiMXEufDn8$Mx>|QCPqIvR1XHN9j=$@ArN@ry0KQBI^uWs|>`?mb;<wl7HFZ<aJ2neA3gk*GqeuwSVd2AqFaVB4o|se)NQ{3YT?!;3n>M!6vbwE1Y}PVY=MC>)7Z7c|$~)YrCWu)m?qPaeEl^V(VDw6P_6eCyE2BMzmPI%sc$tq`jF}eXV(BhEJ4!=v4X(9--}S|L&sw>NC8-HZw0{r_I4FTkXpS?fweM@Ogah`2K3Xn5f&7>WqD0bq`yI;6P#aVtVY#3<v;KNGwt>h^=00nUVEcK%J(NBkx<)Eljk@^IJbHn#YN(c=qT!+_7>~h9(+47}h5%o_uWKk<_k1&9U(|e>dk)4)zpVY#EF?ub0K7?|wh|5P7e2Hvf)$<Nh!kaF3xdJGJ^*7Bt(u3{Qr(w;S0@A;kTKZM>!|c4e(FDoLH*#xih#<wi~`ppJh`>OG{1?PmUnCiiMxo7pJHDBErK;+z7aVL0+mb|WLGUh!JX^x6pMQHMm@U5N2Td`PZ94w64`eU^zhG-U7_<et14(dFbIc;VDxjiXHJN%f$DmX#>3Uu~|8`^aH2{aT$tBY!yVX5euS*iCQud}%X2>B#O+pm+hIB|8b2#qDBk$J1$BMXM5b+5shsI~;Uw#nuj(FUBrT?{CC4I8L<5yp_v{ECQ*nhKs>QDm~vB$!@_&#G}9vI9)*U-fH-r#;}2kqqc3^HL#ejlM~liUU}xZIgOzE>U|x`TerGP$Ky|o5A~~Q{n5SPtNYHo7GhJs?btx|o`1U@VNovqUSKcz%znUm5HGFiDrgeiOY88+1z9(iy~ZLLNY>D<c-E<k9vS-C{`ei~Sh&iTNooEu`-S}KayekNud!p=?sCOeiYXf#uMRuRFD1I@fHi>u@nN{l#$r+*{?<2c2KSP=xSqInx9F&UQ<-(b-h=^<<+}!#=nZ@p*Y}|rcBNA@dY${g$p017zcImGlzErLWoqE>WAJ%S_LtmC)(v<Ptn>Wr7x3>kzHXAmMo0QAW$u6F+|cOFbKi+BZF4}(gRmrVM=!#$UZ~UdNc2B_c^7qJC37BLvJ#A0PrHr8O&sU+x=q*0toavSz4Ti43xg7xz4j0!TDOr;!|`6n!OrSAS5AnzLVo+^hPzaf$>wal9&4R3f35WU+O6QGSMKo&aJ1xE|DqcS4}`}1b!wkz4mfP$GJ)PZ=(Xxw^`)NPAIUKs5Qp2eUC9eczOwc~Q)PnxISg1c;O(TpX?vCJ5Gp&rhrhl!HzU}lzvK#(^Wz>ozx(gaYd#;8_t!4(GyBbuFtMFAX^;Nxju|&ybgFf##_SfndUKN$dan){n4j~jVlLgct8E#aYgL_3JBU`h4X<EmNqr;L+HbGVsAh9ywa)qcF-xX^97xZ;(4pW_N4ZJp(cYBiTw5xGW*O)UWgic7eZ-$vb+!T)hx3aygm@!3(#d`6MHlE7v-CV1u^D3)nP%L6`cVK0x3h*076CGK$pX$aCS_j*^QfTJ9=&eZ_Ksg&R}QlV!kE*zHo|BE5q4$eZ?q1#<}`v9mkqFw&aJ(KjOiD)qu<->P!KO}ziqzI#kYEbxSi9-TE(Ce{m!ix6<vas^Nh@NSMwKx8Gft<K<#zcdVDKk6&}L(+0~51f=4H<hIHrh=1JN91!`bl@L{rR2Y2c~&l5C{@9!ym=SF)RG2ef6t)l#J7qK|+ar}C#hTGEX5_w^_Nwx5Ov0Qw|zv)cyeoJ;d!gj9fa%7$@hJPw}`{8&@UB&}3ypQ)JlQu5)ynvF|WyT=q1nuzoRJ~lwygm{D@7gxTxZy9udN@CIIR`o7%o)Ox4s?DV;mo0Gk(GeJon}<glPkr|Z%tc3(4!6-iQRfLutZCxL@ge_=gj%26f#8y3w+8_m|={gf&P`8{`hdT(UcW9K7i1J6|uEb$GNU+d-rsl9%<+#$i!y!A+%fy89GZ+`N9e^ei*Ow=k&?z()VZAdD&FgKXx7Re}o~s&n*npsrDx)@M;#|5y*FL^*i~!P35N|siSi&%qw?Ns#WX?vKa#Wc2-{5;5Bge>kAuRx6&{!<9xTjYJ2!l&@E&i)T{M_1Y7mE<bv{--93Z3U5C}nM`&hBb<!Lv-*aB=&xFG}yuVM>Wsg$3q;Dw|eK9FZVBq~ov_jv?f&oU&e#2{MOMQG!4@Q3>Ie@<?8}D2g4CvACc`sg-%U{F)sb-3{h>arEUG9mR&>5*$Z~C5H|8D9tev{+mq85+Hm9AcIInVB!)Y;ayyXF3&Ak|u<S|2^>NCgKp{Box=YxQg<$p;<0`oG(rnfG?1%c~kPOL8|UQv~_BkLak<L!dq`LI<L2k6>B-@HgPd-PzO1T#RtrWT5Ser~uQa+Wjy)bBcv8kB-@|FdWsbQ$NC|ZM@t2ZFh%pmCbXkJv^c7T?fA~-nP>%@npUc9+%g%(8We8wmbCh{kdtE3w2xeL}sJ)j85{X%nbE?2U%fikA(PJ&atYKKbA;{A^Uop^2X;*uzn9Mcc#I|F0=w0Amc!1?zJGH)jHTfG|?sP(F4D|5_qPBHTB4e`i!=FhZDT1O<JV99F6GNgsjbapPAl{I^)F>p@#O}YOljq)DP{Y`bosf2OV|Q`I@xerCZ><c#0Y$3hJI7;VUWh9jf#e8||F2z%EE!=f}1NLvz;Z0>C-z61C|Fe;%@4aSadc(ZM<(n`e2lHsr@B8Z~_g>n(nrv6e|!>^-+rgv*g;EgU>1<8QWWp0E+(wVV3l+9JO~jsERkGj1@I^VMt+eCX3)0Xoxccm7^Sy`YCJoNcgQ!<z+4-c+=*Q5m(ru!|QSw1<C&Xq){v6%mGTOlIwdGTP3=sZ~3F*!iycB$wv#h)xgd-(b|fo+`;MCrG?$=I+Dzxz`u?20qgZQ=c9A4YuA1zW7qqYDwHcATjX|gNIf&t*zJf;d`uhD%snhCv9DyZ+|oD2+jCovQo7xg)PiPYm0kg@EvM5MOcaT(d+xIl=SnD)$P*)Z&hl4nESi;Z!;v85T1hAvg12AHT9=8j4RVTxzslY;ScE=qfcI01`4}CC<3fy)t4oIA|xL2g!SpoF5P_$mLBSNO+M^$xA~?uXkVfNt3|)*fc~W0u=T_Ti=lsork5(d%AqyiKNfEm1rDp#L2K`L#n3u#FRxE`wDKTFqm*LvY9=y|czW;Ji!r(1-!~dWIMrA66D;L2{9Pu7#oIgOch>H3N2MgzIf;Y(a|fiUyM%CEet3R$dFiv=OX@1{TYg79$_{Z6h57E=9R5ml&2L^Nx`p0xLjDdc*J|vF?iu8BO5Ab}{~{8sK5NtI?uBVXh4aGSG@OqT*xf&CX7xk2QDst&PbA(#*+GjNVah7NHI2GG)#&^0;^rU}EJ5qPR7my=Z`|58mzw$Jy6|g1%{4yP*rhWw4z#qugG*=-2mS@rVgOEm?Zt~PDsDrXY=Nr!f`BUin+C^iQAcF%+f7Qkb)rxK?4cS!7iVj7)n=&09nEXO_pC*p-THM1km{Qx&eTzs3_H}zc?8XPKn~3D_IecL`CfZmyUUAi(&A-R$XoKNF=6LKTy%e9J!v{Fx1YCuV|b?Zpil-Og2fk;_VXE=q|bS|Qv**~{)u20bj3(GP*+Y&-{+=pUTky23o{61LEm~9+0RIu?Zu{^WrM+N?&s^B)wuL~+Ws<#yWta$C3N|QDGG!z^|<e77G@piAb%%O6!^kds}XmsV6=h`ZnFlYb_GXtp&IMmq;0V^W;%+w7H>^seSN5z?21!wBevV~hu~Y2mO<8&z*ifEyECxu<+X-eIZg9j+RWB*bcDrXWPLPjIGdx?XISgA+_C~JyXSW(O#S{w3i>8^>uM$X>7A7gr?12|j*n)Y^Cw;tWBthdROytM<Jj<xw7r5=zKVZYwF+?U$+ny|@)|_0j{fG>NAJARD(?0y^%N9nw2S<TjYjebZ_SVhUYYuD6#r#dwEdL7uRK8q2z#nXbxeA<5BIXA!gjNUAGh9PpBJ^}sv@`Mz;)K09br>1jdXaPw;#U&;~t^@t$=KniSTCd*bN+hI||P$J9Xy!TgRA)K)v`0m~vM8l|!dBw0FP0#a1?ZXE7pu`j+gl>&aO2$_wGXYYaCxkfH1oUPg@a*V<}((FIuV$*%6%e%jYKnAX)swp_ht52)yySx*!&yvM5gqznP9BE`4I<}#Jnc+FR4K!gwj^dPCynbaQ1xQE@IVb=1s)pjyj%<U_q=EwF&EnS2_B~)fcbPbHFt7Mc=V@U=Zmo1g-KEQm~U2SG-RqFQ^!#-G8y$ZM7ROIJfxDNl=hdiD6S3;>*=+hhhzQC;n-z%Y&Zh8|t9sIjR=gm)Q-K?k&=P%XLKxg2}n)e%bOTGnKJ^J?Yr>P7MT$QUmkQ$+ZpBepK@(r5RNMbphXwF*d`H%wvlbCq3mEEyY*#|*CLnZGvRSqq5bCHSpVkqodM`_w?9B;fy!kwaO*Zd(mfV1)Sl*LW2fAn|sK+`&#-^y+;MH_iXsNWv2KMU3*MCE;BjOHdgkH{+w_lfQDsBPMf(dLXrL}=vNQjN(=^i<Z3&7==CIsKVdHY+O|jp-}TitpooMBnotyut6iuDss*);Jt0=Y90r=k5Jx@p~4Gy=(Q5pA}pYtOix<Q@u+6MLdI|{lqC|4R|*Tsi1Snn*SJ6ROT;uK0|Gve=W-NtpXa+fp**~3P|v`Ode8KE$@u}Q!A}7zwTGIT<nM!5rkrv{#t)NQF(<=MnVj7gXgRI0mbUGJC9$CW!)OQy=odNqgHbX#okGtKh7<w^6OOE#uEka80j9Cy9y%X0+#wwvwm7htLm}tHowge){`L9Z7gZ1MfY)pslN!zZCF&D?X>e-fBb}-iMM73lXLi;b@%f((;2sq9JToO(Canu>xDBiH>F!eSHSLzx2NHZkdGN;u7&S=+YI^?M86`Ix{WYzOxd4^3dX(I^;W-*>4A;*wf^R~dfADvks!&nolng~QJk`?;?4g;N*Yc2^E`cU-x6E?o*RFq!f)(r23!@_Do_O>xqg*>aI+EFIfhDZH~if&DcU2!z9n0(&BWt0K7!-@0{HeZfbph_!N#p+pXY0CI--8qsnoL_DRbm~NGHB==}>Eb<iV+R*iS)~vTko=+^+}=%(|t}Ntp>257d9QJl1Z{&(*2y_bUCl?2K<_;8G_TpCy`qXh6)s!!~>-Ui6keoy~O72AHl?tvqk{O%;1o&y}^aVTQ#}YcSnkgS=cm=48*lmQCbpzapz0EkWJv_8v~LxSokgA2lHHewa9mBRgNzo$S^wocNpyy)_|9^{y;o(P+GR7-PgLE9_e$SMhUmn9KK|Udazl=xW`G)ia2H82>%gpvl)jYDcP{jW$E<J`WFZ)mZGC^JjzXf09m=jCSmEjKclZpXj!V)xT&A17DW6I+HmYf~zd=GkX;|dwEaZG5mK`I9J-u@8`08%ftxkf?-#8eES^dHD!0_Z^^OuhT=WUxBBnP<#urzblk3P{~V^>jf!w^@J}zqIY0UOzL!nkJW8iJ1QM1Iu9n)1lc$eEFx<@BNv1EbKc-;p23|_@-eoQw<<bMfu0jHh`e@NK#=%3}-2QUg7G<{6`wq;lrLOT5?k-*pY>ANTqPx2ccB0gItT<WNoY{x)tqlZw8b!wb1C<-duUM0J%;G_}E67Z3+v9thwPd?d{)R5Y&heFc=HU;e#S_!3M8Yb*Ac6g+#VOHmmMO2M+g+Zll`F5H>9AsLeHwh8!<%tL4b)U-&k9r@OYXg6j$7JHJ-LjO_F7{)YgN*#j{hX9ILvGJy4%M(a|tQk!&UXXTD-M;t~j01qvK#KrVbL?6|tBS2cNC~&h<?g%2O^FKWY<M#GB!|wSR|(i8#(-u@>#4c4u!qE0HZAu~iiRJM<`;&2X|{Ru-K-hd6@WXAwV~+#>OW3@72Rvk#N0KCB6@nC4xQn_bI6xvFvTfHM<4+Uyq`bd{9u{3>+wNv)CI%8awRWLFy)(Rn`$XR++PIsRVqzkREmOf2JkiIRon->ZhU%bZ%bD&NY1b5xC>Of)?X+{ARJVQapp+X>@a5kUTk(*EU)eqXDVXY2k5KHzJ_+#jXNyq(^*=9%aeNA)lp^(P5T0Cfz2!Sk6~K3BRX*2XP&`wMX@L|5mwYU|$vwUB1%kmYxUbGiyzGi6|CVb<;`B(5Os#&JT()$^#r(u*e_$~B+=3ma2j!fCZf#t&k2uB^Vm>R9WSPj1nipZlG72X+_TbyJ)GOv6bMa?&LBPvWho$Inzum&<}Yw_nXe1YK>>`yRiv33B{vdvVu@4&21S`*jFAy&>4Ku2pVc0eg?<;ngF^=g#O}kJYX<xO~sJ49<klVf>ggA9~#QwZ~_jCN#DTnjIT(@<8bGrU2c@UTg{5MPIr5SC1Idi$%B0PvG!~ZoILiJ?n0KB$O>uHq}WA-%ss!*bNm)escG))@@~RtyMB>Dw17cxA2T=+RmQ(UolwEf(uwduIRCK*Ra(^HOJF&ja+xTVph%N?awWL)msgO7tcwPqF2umy1xci^*&mzkb3`CX@C0r*|8sdm4)yA>UYS;6147gUef8~U5JIHSJvXxX=XdF+ORj(X%|wPjRPn-9oX+sw+k-~$In!GCDE6tkYe#|benQYg#s<)2z_$VZ}+YAyXM%KO72NG+@9Hsrncg<M>wNi*sNn8&TK8wq#AyX^Zhj0oGgn}e3hGScjLIIz>_K2X;irJ*EVVme0_)K^yy%gZD4X?@<IK+IxwXVQ<Yn3ZH~UAyssdKhK4e1$rBa!F23>Jk+0vb2p8I$7$x1TIJa&+x&7eHJFN8Y)vO}+B|`;k&Bk?C0Xbs5nd`~ut(P;+?o)n9U1lux*X>~$Ky&BJ@hxiO+_ulstbBiNMik#`tPq!)Ccuwe08GGI&mz^}14V5)g!fI>te(EFVlt~H<#&)iC)ad3zg%W(e<{wu<vF|Sm;%m!8)q`GEbUq~Q^T|U?VC<$tI?bsx>ahGKnt#jc$03^!|J0eSw{c&auJ}JGs}amo%_b)+;6+T)<R;2FKw&dBW76dEUu5Cb)9Q-+0#E4s5_{Nge)PB<Y$A9Hp|&8PB**n^H{AUh%zu=_2R{i2AlEvbK+Lxre9nhs6yA{qTcNmJqnB88}HLQtYViSKR(Rd>$?PHv0iRAW-h2no#k`LOUSi5m|U>lcVnM9v7#59T3;OQhqv>q8v)rXyVeE1d#gU{sn?u|c9kpD49sskeE(~WzyXZpn*2FCa9Omo8`<ys+T)Nnlyi4<e3aAW{lfsgS2uOZ^5`bo;iUP!KOQ$-G&a+}U#VZaH}UX^FW5;*!2a{N*c$yY3m%1VDlX#&L3Pqj<M$*tiOvDd?h2C?`<{I3y+ifo&#?f1tf>*_T&mY4vtBtUe~K7+cdDa{UY8aBoKiu6Fa5l{TGoW2U$)e>Usqt@Xsv_CrFvr~KG@3veI~u2#}{Zd0im%qAMmbUB@*@Dl`kOCU$BwJv8O>5jk-3ggAf|{kP)1X%6WBOytO{IpEzi1HyB&=*tuV$$yh?~n0`d{i`V?a8}BDk(hwFah&XV2A$+QI$47_IG4n$}v=nta4VM^GcL<#!c7FT9=u1jpjkS75l)Cg9d^#3G2ILr}Zx+cnZ|;lB2FmMEbY!K7++J2enhd|dvIyvCUj(MzboL2uQ70J(<*fax_W72kF|FpGzTipc&6OQ!-kaZE)J94UAOO0g0DrPJ2=x{a;@J+Jywf~Byf}JbQ`2cZy?=L44J?(%@#UVtG@ORXptArNUN6SZ)Q2Y6>pnL&6ZG<vY8p2BpmwTY$MnFq(kWQteW|<Bmm8|uFVFS8Gh<9V1&QfPxTh;}<a|GsF&Z>(@8?k84Hh&_z9ty*UN?V6%{YC(^Vz7pS==I6H4Ea54374C7dJBF@VQXK!ncflwA}s?DBcj@)IGq>$CK3sAi9E*Sv575=GzOA`>HuPT&vV_yk20qwS^ZRnbS3|alEW0+sfyDapU-6HlHfgS{mr<u-RN(r0tig0QUlL{n_EY$*^08Mhm7$dL-hlJGbP}oIk|p3)tY~A_T!9kb+|yg5D78UZ4QmU=1bOUeTP=x^u9;rn)|G$(M%{Le{(drt9D|X|`%j$G&d#^`Lj^9;MEwzQdQRXA?X-%R|4_EPjJ_Vq|Lsbq19Iks!l$bhuhGav98O9rxt-29HjHn(M`)`JHFfL6$9uxQ1X?e<q*5{7Oqn*zgS>OqNvI7K`WOb<D*SKQZdDQ%3eG&nh1Qm^QWJg1{<qKDwi1&!$6ilJ%SW!yR}}xmZRLyKx^J>dC>deljwi&Am%+#J|SZ(Xt2E2KoMcyR*$(=Qnq%QLu&?Flg@D9zp~MQgj@E8rz|pq24@HS_h)Ip?_bzoCv$k8i?CHy>dby9l<cyhk%7`_M80Py<c5t#%i}D#a6I@60mE)Jn0Xv@u2uyFeo)q^-`s)LlSfxN1p5FGN>cxiuA=t%6d4%_Fons(AM=j{1k&|N6XY+00?WblZv+*LscHt`n6)k88z+7Hn$tl`$b*u3fORU-xU(|#PRLojPQQfZpCW~#zH|FN!zYE@(#Ht2%E=c8t^|~fu>GxFsQXUZzQK$;$;^@#Gc5pb&s@~okFP<;N7Y9&co#}0P7HuRlw<UWJAnA)OO=#!%B+k<6r}`YtNW&zloRh+}C-*trnE7AUL;O(Q5Z>EmJJc_ZBRjxi^L)XMj5}WBho<EWA%j(*Am9TFmucsW&u-agOA95WxNeS-VB2B_`t;&y@9K6Yk70xJhqhNh#ptKD2*CZtb4CCxUxUr(R!Cf0}~N`#pEP+~VaMy$J~-)koq(bNM^xJbHh+!O}g19;*B`_DUX;T0^3&@9H4-SS%&c_`U+tSAvvw^2bUF0=DsX>B}g-`mT}rlk_ohzXIeb?S$8rUA)K^D&Lwa{NXd$N0vV9t=M5W)H9K`^)vq3Z9EwW>Z%&Hy@~=YpKGDbB{A7D&tx|`_2)J|KRn-vf^1s7>?s}EatEDWMcf~Wi5}18&t=4+JJ}uX75DmBZ2Ye)J@)-<B7P8`E-AQ?6;rt%RW4%3(kkQ0>M+VDanWnPFoJ}gs84XU+T6ni5FaP&<E3U&cd=TRzxd=kI~Be4AbYR^Qad{@Zig*S)_gzy<-6WpE9BHYOWGSV7E>8v=rsW)O3SF<JajTv*ooe1CD}EaDx})DO=Yb@Yy=0_&YdkJ3+CxDs`({w^30EU^~o+mvwb#5D`Kl(I!kX$8(y9KO56Kv@8J3=-lly2McTlF1AqQudVIyU<@c3d;H6t*8vg`>NUU>QIZ9CrIK|Q(dNl?=QfP*syhLlkI^>=O<>OuI;BG!4Gb2Ah1~Km^5B~O<4;tR7sptNUd~TU=G0!&8U6E&-sIfp^^q3^qrx`dyzWsMdq`DM7M2zw-<hl*1op-2u*W&O;*`vvzC#<iY&@@uxqc6J?udGmH(LG?KP1=8u&=3dlYo}f2s^9+H^U<M$(+)0o#ZjFc+FvcMp<SQ7xLbRrI8kt2O?mDUwdTaTEI(R!itNg0UMD2v%yjwc%o3z0--s4)t!f@d^3}idk2rk|s<Q_ZgzDoN?<c2fbQ;QimLMyQp1qId{S9ZDBerU8zw?24y-kk-LNMC~OP%xiu`k=po7k><tjnZxbp36Sv+cV&Hte-_>ME~oQkkI&vG6EZl}wXBo&7lJ3U(8ncNVs|ew#Q@Z!T(eXUhAn(m1ah<7KSO$R2f^?uSjR^lDy6z9H&UuVK%<IfOQy4n<U_?9;5fRqi<X{Kj@X$fURW^gsig=pD+hcc6P^vG!KQAVan*`B1<?-C#q`=;-bHUz069mCxHGRv}Jbki&LE$PV{3h8E>-Tx14!==*A<=<@~yh}Xuws5zv&xZIE1$8p%~pmo*Kzi(Z6EPn}`qV};G9GjC2@}9T|)FkaI>nU;GBBz02`nyc4V#_VLQ74<m3~gnhIJm^Y87&3&eCa0pT`||2S7UpFkwe8)-v@XI550%)i1|rToc8HcLtC?z5_FY7CA+Iil!KGw4Lz*(OQ^a@p?AIN*;<#%>$A^9T^xz3@z5AehY)o9{-UorEjcOkw76ulUrN|xppP^-9KNE8#BZMb5JukG#;^SWLN!0l)!kjj%Uf$&R49;Wfn1m<E8Ptxuu=zoV3^dn7dPSj?L>+?Ut|Oj*TkSt@h@_?b9X#Hy0EF8&gShawcnNOoK-yi52@cw)hou;RPR**rE!t!6LWkgj6wSy)z4kv@cuom)!q2Bkx@nMdjfvKM(e5ycD&`}lq%Oo;`A5Lslnci2%zW&HERt3kEf%#fwEds+;TJ<i!pgcA?7%r`fzgC{rDQpY00et_Q?B+e>U-f>G3Uk4)8mMS=?ptfuo0HmEe_!ESA^#WVn*%eJ8X_L=V4!DxCA<oBC_kD)TuXckyl5elHIx9aqn<nzjST#=bhPo-fLBXDNf_WAi?;jZ1D-P9+~%%@C<f<Oh^8AFwkU9G-uj)v-Qe<<&3tjY2a9Z)Am)8soZjg@6o_qTcoC-Fm;&HH$E5$<~CwECAo=dEDrf`tZ5oW|1Hv)AyBAb!Ku{b@HoVR^{RO&$pRlqOc=_UJ!bzZ-2w}pUBjG`DDEFGbqJ*IltUl0NT98ifL@l(CaTl{Ua;S-;*t~!PGO;aaO;E$1v%=wi~9@+UnkaMoIt0p|!}u6EeS{ugD8Av;(Gx=4%{N`zLZ0<I;tV{^=gn;l-{JDKjnXLV9Swq1cVC0{o0_RU_z}g5MWsI)8Y)+cU+JVlUF4*>;HOIa2n5x*f##A#^%6e;yxyym)p>-~C4~Ed4WAW{#JCdSYz}>YA6z`w(n&YuC=OA+o8s^qx~F3w@))K_2yP9yha-j{&%BH+#@)zc_t#vYWN?5Umh~=^`~j-LEx~HxX-5e#_D5)2%~wGDqo8$4-(3TOAtiiN5P}e9pzKPx<e4+lhxO!>GCAMJ-`OB)&lGg&f=)xEWoK(i8lNpTXd5HmCdg{Gn{r`|~?L5fCFlKc$ANh@KvV4(l^J+1V}ND_+0Ks`;?Z+gWU?FR3+6T7F7ZkjC^C{2sSdHfk)wGBda~=T!Bu04bLE<<mD=<=cEKKoIy=<!c=U4UlUley1MLxe9HhF!vWmZZ~bPA?aJC8bJi7(v))}qlT)3&k=Q<6M%Ml8ku`si5+#|o8?O00{-B6kmLSGki%Bt#>Bl(Z<nJ-@7-2gkf5y{sn)s{I697mE<=aGa<bMQ`MwS8qvrveOJWMPhkJ4c$vPkCRnt5p09j<j?61Hkt9Znk_wUF!J8f!+(7l)TmuQVEv;Tfm+AD6;c^K^Jg=A}B0g?-9kVWG4BLa(Y;hu)oeCoFAYR1zzP7#`p;j$CririT_$8s^cW`8&W>n$}>edJ}<oE@<mwnx~Ssc`3Mtv<`No~Pb`fw_O5+VQ)7LnoOgBf~FvwGWfG)_fY>waaLYQWs+wfcv0W?H)Y2P_yxQ<OT<{jjZ=N&F$u6FgOpGQ~0ZPS>Qsk+sOu<Mb0@bs~V>8UuUyfy*C{9_-p}HjPIm6XkcFh`^osXOLDKpLAY)aZe2V`8H7h&U|v=;v~xLXMv#l|=8116`-hjbPWAefKkxKcIHY1TJG=Vd3GB(U<;DbF;@T?>&Z5sHG^xC&=jD2v`*=GJTvV(a;rwo92#4)&>WMOfBxv;g`mI-^yGol4s(B!Lqe>Cfzay+Q&k*;T?>(35$KUI=yxkf=Ye#N|-Pslx?{4^Q`gzK09v)CIY1U?@J2@bVJCK4<L>h0l69^}qc}psmlcEW@4<>@S#)8)&d^T`ZaL$?1t{+EMtADCb6vYQ~SI!sbf!h$oHnPddb2=)bPNj$CLxZ?0J&WHfgFJjU;e+Vr%KJHcnmq}m=S{PB30SK@ehA8waILpKZZN9*R!?>gq}%tg6+jWG!A%oeq0MC_g5X>4Gc4U+qF&Q-#41KChx`7TN*NS*Nj2yvHftsa!>i$C0~ZR<_Lq&?QPa73HhI}dVC;<uKJ@O;;?USDAu=78@-s;~8>8l3pbI;u!^3r~OBN7x+IL_zu}tr$zU>&~cZ#!il9jj(1H|{VDeLFN-YV_njjN(XrhwU|LwpQta}Tkbh2;916_Zb^CVKEeVF#h<>UQ2k^~PIq=R##Sk))8RtHU)h+|tTEdd!ZM8)+z(mvCM(D0_|cfl5~-eqB?Ns!dX=dE4KE2yaSf-7|-mPBGX5ZoWcSC%E-IIo*5LmW;dkr;UJ4I@TPX2P^M&yni!q%<pO!b(2p>Lxi^s1y>XD)-=mtJKjzZ4}`t5!2a>de&RwZ->l*nMxlQls&^aHd@6HQ?eYZl^=)W^WdAo`dQ#dV7Ek()%!XoVd&^|tiUEIrY~+|&okrf2<a>RfY=XmOx9oi4uo#k%gZW<ob7zxUdO?2bHH1h42~EG+M2gP%)aZ;K?5xS&gxpTJO;z4jbv~}EVR!T!i{t^pyrK7WZ0gka7qt8cQ)OCy<$A7V_;^zx^uKd(tgD|{2gH(wRspk-D7euGtNxbuOxGO9L2$lTtLIRgI*qTxBu(1?Zxk;sQQT>O08(>7d33R;)6ieE7o!Jr5SYxpW|R`XI@@{6#**$A&2i`8_6BQwX6+U_KP|_=+plMy4j#I*&TGXr#g50%{)X$uR(k<NY~`k*oOIy%l7fkqeQwRH1%}6m{B3Xzf&d89AEiS}lWY6&r}tex%mr52+v-3=)iS#12%M)Gadj&cJDV%T(!|fJ{WsUsRj1Le;;Cg|Y&WhVtaO<u2GW5IJ}$(~7OlmtKi`=@o4;)FaDbP_t+ifbo=UAUC5}pSG^=M)b3L8*T-WJfA^ct?G2-gA1>;<n(gPedG}<X~^tO+RB3k^NdHL237x6J9Wisg;RWzA_qNqa(H_k9&D}=*b%wG*%>3)+|yyt!$IT$ThGlg1(yLhi$z}@eFjMLd^eLfIS8~%CG9RMz(v)qXHNw3oZ>Lw)0lO1>|3GRkP_z>CQ<r`n3yS?IYN(0QV5j0f?yfR(2^_{C=1OR1PAJBUfcJ4JsPuBJ)4ddVs+U`r67N$cZ>j~g{@u0bhfh2+yU{M(Od;>kf7aRL8L+w3Vw#<OFOR&HGqRs#`0M?;?8Q&<+Pv<~_#ww&uW)e<17$QXD-|Two>BD6aw!0}RMDzEm;nVD2&av`fJH`rP+@Q%L*KJR9_5C~yzs3lw8+tHm*Vo9c0JQGmiI^gE7k@->@p&V^rM|7!_D%cVyImk=R)cMX`@LS(%Goz3v-49X;`hg<pNSsX4|5fXwe-P7x(&}S^h`6&9Dp@#oL+mw3QeA7wPR;VKq(Tz!F;y)HctY6<&RD#ja#C>X=mZS-tb8_$w$MdS0VTKV^IZePoGtH7Xkm%<NAQJy@dnx?sam0LPw(VtOK|Y1K6Tz-C8v#2~nBsuIGE&bsIo_*&qX;mF>TOmD7^pX6FR&B3f(J{Jb`WV`x#Xs>ATAV26`{O`;&C)9;t<Bz5WfeByWP++9A1W~MO%YjY{Jav?f&^%X0vI(eJ0-g$m*&SbGOZMA2~D74R5oTn&q%c}fIuwxcxy*~$xvF;f)cyJ<tT%Quw-{8U=8=O`xxTLr4@7wTDn$&0<^he5k=?|CLd9tFc=4*9$(yg8{WVwD9xRb*5M3CjZC5YSK`t>1%?aK*I(6SY6h1z#j-{7rj)q&Aj<&hVUX^bZ=@4jqJ)R;Idwhi*Dy~}N1p53pD>BPNp@OEG>PvLNqfKfz{-CwgeRfl&t=if1T(w35cTlCrXfL!A)>d4rry#j`RZ!zNIlQSp+o5&!c#>CRWuGZ+6!s0xdRbo6lK8o+;DQm6Xb&L1<;2N%HJ@D<oBU*sYyxzR69=Be#aCLE$9vX|5yQ}X8r(Egzi^ZM`Ytz(PPtW}_-IQ(a4Jgfr-J&it3$omxJF_~657p|J>FLmf{c@jaQR_XLW47Kd4Hq&Sj`rF8>hj13-D?W*_JH9-GC{J>dpgHEV|Gz`&onPM3ml#&^`-u-X}5c~-K$enqc}eLp4%flKEe2XE@g?aRVJWWG{+Cp{L>72?#Od|+%mR8=Tu{H>C~|=+rbz+W9H<QblM1ASBL|#s8$q}VmPmVwlhiad&zZtJ8Q0Z?W;_kC{C^2++A-k@62hOt=qI@>+w=J3mN7H?}#|P4GrG4y4~&{rm$7!ck8cC?Zar9w~rsf!I}_K6AKl8j)Ux;*L$r_lpfb1{<qA*Vh$8~VJz=>5818UUY${liz%KzO-|O+ir)qg-l~0m5Fv>n&qw&IUHXa?VG(yT3}KJm-?&3zJTQ^gU{(AM2=lT2N>hcRV7<miyZW5#i`qG@{E1O9j8AiPAO-;%K-ngmVEH23oLzhOP6gg{UbRKuOzF{uasPtu=r;13&z|=CHA-GE!#<8J(|qq^j40%Lu=Dl+#P^}ewKW-JD{jYE@kT;HLi2U}a1ooFy|w=Kj&)bmKii+NyGPwwKZ%I!1JQ<~rUPIyt=I#uPe}1}{@UCo;Y>|h*56Um`wQ3tsDFR!v1=EU#w>O6vb|iPg<Oa1;oW<*QDXYZ>625?+eh?;@A~N}byx3wyZS){3Rp}VB{8$2<_qYwW&0_(e)6ixbW0Wmkel=LBuTq)Kb?<<5dyJ(JH69ZQsKIos)obC_|eK2(WYx$flv<o@^QSx-;}pb%XPEXTON=ls$n(kInG)36)Z$b9kgs`lv)q&^y;|PW)x(T!`_w<6dhKxDy%Ol5fn#bg`EbiYoi^srKR6FR~~gWzo}B4-OaG^=e?8fKFO7*n))k#mD7Ai-N$>hJDA9WkA2weFj%{Gq#xi(<VS$j&pm72HzU>QGy14Kol!+|>xmFBkW~0{F+gftNy)EppMZZ&`*#(g_M)>~4%HoK*)3UsH79IfRKMSw3KO|?xg6KfuAJFo+w#xaj@jm#Q#mbsew6{csx=GymJ{tzOGM`BrN8RLlc~7t_pb{xZWNE&x+nzbZ{1Xdl2&VT*mjF^?R;(SWo_c<8ECMBPmAf97eTH`wRu2XY74XSVCht~v0k%sDo@SzaU5}O{Vq^He4+cV-(^Zk{=7+4GQGpJ^yQZ%x7c*w#$7X?)T{wbj&~U(H`(RA*It-I-hY76vu$PAMi(;aJ#4fyt7Mfy@XaKxsP-E>@XGxWQ#pQEqtVda2se<!a8?NN^>?H=0nwb#ry)B&>(B3Vm^=%keMWzm#^3x7Hi7+ft6Maq8}8JwnTe5sD}f!|&WiL*2Ikc3*oV*GLJW)Mj|^ug*qgSHnf$Q&f-tpo26`RDTmSFnvl-N`2kEt2Jm(VUU$aM`0}dmH-urtXZZ9vR@3^(2nq41}G#K5qyn|}nDB%!OcF>|Z=TcGL$+4_5tUkw8RjxhvlRV#wm%%7M3F>){o-fJGcqIpJm5&cd3b@eC<GrgSIlk0g@;L0w%XbWQY%;uz-)-eoi<=W)HOHS+>V+)=T#VE2uO}VBC#qbRsb>rrx|e?)2nJ8hTC)NKl^TlsMA5Z_`95bBb?`cUh;UKsB8Qnk2Q2G~wXBCOk)2jc+yQ0pYP(g6=mamkAZ{Wx$Ig9sCYN`cIadmK7Ig1&hxDV!o^F;RZi$U}rzpe>0mco2w@fkU$gxGYR%aDi4-=c-mf_%)5x4<F#;V?e>dxVS#%)Gi!D+5fcV{IzE9T0$9joJ|^QcbT1V@{$!#t)M_w}XCl=R~O{5k;!l=b)NVa9j(W)Y*-8Ls$T*Ayt)8eDovXCr4YHzR}bvIUR63w~@>29z%rktA?Y(|UWpo+ZnDI1BdYVV9WV#ib-yy;&dQCgQ&>5HgQn1ano6NpV~}8}DoLU5~5~NSz-=RxeNFzI&{v-Jo|Vr9#Exb3^n{{;#)hy-ferifVngg`Rg2YkNy`{c;FXyY0Tqdy9F+nG>91bH59oDL(#I3^}j!4XKwcQG~){{EiO%iVF*5L}tWX4P$$UP=|GP9K5s}s%7X)HoAFQT}eV0yz^-~!ax4Qo{9RHJ`gG+M!#zJ^dSaiWAY?lXA!kmE7)tH^W*{L_1D`A4^8{h5pddWgYKpEf}GWSSUDZt?{KywZC<A)LO?b>{x9oR#noz{1;g&#7|D&#7+zG~%rXq)Tjz!!Ai8lpa6luss;v#S#YXxI=ss@3PKwtiiFJP?9y#zm4C`0?sJ)AG!C&uNleWst8G->e&``-)vp*MJygI%ghfBhLZ<fb4Veb`i^PJOkV|pFn*7Gz5h#_J29Okp|GVuxF-D}3G?s(TGB8};JT&guDv#3P|7jMRa`l`GX_!WaU&BsBjtXcl<knO~-{SqkSzzG0epp-cU1F<L6mTD(7)HYi^=BnI1cf!2OSp{O>rK6=Zl!{$>wsB~Yo!q^Bt3D7|&Y5cTEbR;1KATFdnkx%KfqAr5IcCdi!F{@N<U5znQqNA>f~(Y8_vGUL@fH~L^|PE4(f6Jev+GCU`{n6E=R@4@VH7m&4CmUhdAd$OFg^CPtP`js1zb(twvzU{PYN`uZRc3oQJ1$^&0k;E;D@*2L=^g>G)Vf3XtH#SFY1H$^HsG!Kq8yXWhXVaYd?#rJX(GIOLU5(;iCDMb6kvcwnWD6mI*t_5|LA3OW${aovA~)sv(vZdpI?Kadiazh=X${71x9v@!#{vpY#w>hvU=6D9t&?_)+C)8FY{tpo!kQ$5VLqZp19em-3V;)9rO}FT~%Q+(GRoQf);uc-n6siY0ZW2F3(P8dAR;yU)0PfXSW%MA{d<uLo8}0l2X2X;Md6*Ngrn6z{bg3Ce;v7bbUdm0)47b>-fQMA@iGg<Dbkn_w+@z*hsCUWzaB&E8*7Zp=#IeV*xc^9wKL$g@Xs6<bl%I7`r(+s|hCFVHYhX>y0&OdE(r80g;-Yl%u&v3ac?RovqE8KX@BkCHKE@gyb}lW{KJ)TVsM+be%BnqhYay*g{R3A*3sgB*1S0rBFU<Gj;eVcVKFe4VK!hIL@2cacY6`sSi`vUU{tyb4Em18?m5+bG~gMd0qzoL>a<OU~0ZJ?DYm!srGreQyg#zSdQ`gU!PEy}@z(y0=@R7tf+xr5A+wSY4yp13@w4TluoUAsWBSa!x!^`&>P})IzBD7AIlRsb<f$jP0?%Pi_9A=v<qY1EVPXTZ}ZRX{Hlp5FL?FO5{UCDmmrQ0o8ARpZ8-k&BVR;UTa-5&Ayj;)D*tOX)vkUxr-~^!=pPn^&_o6s)SJ+{Cj~*58(&S@QL%qcQuLMHd8}ebJM~IzF)qoyF+db%AS`LClPX-dM7=?#%00IyoLdsF7J6?WUl96aWTJIAd<_wCzwHRG&5ES4sLZ08gybjhwWiYnp!ME&UVw`hbBW<wzqzMERaaQVh^JEtJb@5gRn@vv2{MlQ^l_v{4Zo`a<a=LXj$ubG+>bKYIkf?Ys@L*nHKH$Gu&xVS9q9d89t({RqRo^GY0(``QX9xV+1$x+zN+X#94Nf`26bjotxFj?hY$`{<wT*-4>siq*XwHY_hFp@g}BWBg;-t=ULC?kNIhgbH|Q)AlXb3^VPmd*G?;mm{pxp8&2HjVcUoy=ISlzcX|Ub#lVLjZi{htZL#G8%<lWfl_TB|Za^=I5~uLHzs;xBQ`<1;nO3W3O>aQ<j|qR#u)$QH4L@R-K}=j&=aFv+tr>6dm1z}esqjf_K=be4=Y{aTFf67&=L=@EJ-0okdL?9~3UQ8<lQZh()_Ul?@L6l+*S7Tn9@Th$n6MV*AlFN97MN8rQYOFmlWY;OD^ttqYopNK=S5!cA<=L8pX?hgB)%PVhx{lndd=sWYjz*6gVPYT#icJC+RF14h2t~Rtd@Uvs5jOp`1!krSp@tA*{MC=vX{Fm4|{sW>yFgJo|gH<U8uds_b5hHTpKD)Y=5Kar=K=9O=j9JrLwItp5)DN_X<EF(7Wb8wS(0B%5gTz&CwZ?c@B{Cz#i}+H&+Y|na;zoapTcPn?aB$<4v*cpTKDTegQ&f;ypK9%naS|p%rfiFKTrq48fgeQc{(k{{@_joN7cD;Xq)9Pr8^Zj!zT0Q9fUgqZ{jK6+HqZ|6L#F#VcFqCJ0VU=aTUsa(>Zdpn2U`7wFDrCzd$mm74eCKXzkxo!wOLJ;G=#ioaS%2leF(0L_Lj>Ab%;1f~MXplmO!Kiy)owb;a(BMo^{SH7S1_8_3cVwkuJO*hjK>yM1F-F&0ocMK?}B@LWKZ>djeKi6wkq^1L9?7MUPlREaWS9E9-Q1uGmxQ*I9D5M{k_ojJudg@aY(|vvz8n?&hT<%8hdy2DjR3|W%WgZpxb^9fqJO|fvKwW1YuLnH&HFn>TT(ViYp!^Ylmo=eeZ4MaOVA?Bd{}*U&*+PAvmf*hUf26g&9i_g8OJBNvo@lj4ZB^-S+g%cAcB^s1wyyk2Ikdg=$%jMuwAg!q=A^f;p!Y-dc^F24v>6bTvU6*Ty)ZzAR3)o;+q5}w5-#cCBrjDgVz!@f&;?zySH793unFj00XKr4no-jD9JXJ`JDw4gT6_2Qe(sXegH!g^*?JF;#n-A&_KD{8b2YE`I$!)nAL!~fy8L>fPAqaz8Px8=x(!W}4gYMa74))l2|G@|d$wPj4xcmksJGaK)6YhbE$n}|eZ0KSo{~0NR8%}zmH~o5o`NDnlmILMxTeU1Y@4-Pi}L-PtqJY^v~^|=cKc{GvFKkD)bvWrH~z+N5_Xdlh@B@*y6z*F##Id_clGCz$3g7K^C~?&Dd%<jQk7DFGSltJD#?wwpP`x6y!hrKvg9I|F*hx7w%vS!J}QkjXtL&hjI^uOsOfZnK>U<IfcUIS{{g-_*fUeKWU<%9MW!HsOxY;U$0t4S^~c6?LQ?RY^v91;4nl7NjfZ_F<9fXuHCjm8Jgssy?({I*1@l_JKbsmHRB+2tbZsT$P3XN-b5+zgQw!sk@?mTYb~f6h>9MvMK4ZEwFYSAr^1a?WA8o_q`Ns`VRU86npW4pD=!NwH<~Z?9(8PtLDwyr=S@VP>zJj7@jjOjz2_M%s(sb}#j-PFagyE8?mV8vSkBuMWsC2LIe%VEbYSz0*mM*CcugMosO2iHSsN$9oO@l%m@7|MYF<5TR!(VQtAzeG`T)S0nh_VQ}zr{LN?u)lBY>kFX>kzXFl>KF$j;@t1X}oAKexL4-&J(`??O(p`r$IITek=Dycv?}=Xx}b&PlZ|1HyKSDEbsTmxD4g6s-_G5sG8X^n;JiogTyXvK3g~A&IU7`uSGv{6O5tc(0Z%fy>hHu#0uFjxD6N_$gbu(C;YdCVPGrXhC_9RKc&j%9Zs%cv`KrX1Am@=kD2GK&0`AV(bYr3XuvIMBu*R9@>IExPRAj$xLU^I3m>>}_NF-0X3H|{TW1n5AkwpWKD~Rb8GL<pLhfN%`2*M;xO3GGc6$Dx(<xmg=Z~oQwo<x!HlNtOjT`!)yvo$=Gty~}(KWEm<9d&^&vA1)Za9-kZ|agX-wM)4<s7O%HU$t$VypPx){2r}o+G<+2&4B1XYTLt;t(Y>p2VNxMUl$E){f81Ay;jEA-W<Aquk%Y)i@rX+j^+O<H2useYa)n{+?X8AAWza_(iB*=Sc9YS2(*cNvV<5n)ji2n_a<WS@X6C^3h(O!-z$d(;>W&%2#~|89RG_Uh6zEs6$}{xjrl=MWX{}%*Zu+32~@o1lVmZoThne{$TF;WVv&_d(d>BNzc!*_3fQ{a|`MWD>BD5*~8T6(YHi#XX$zPggPuXI(6ti5m&-R*c=!{-m^-QEGk9K+h3o@No#4#tx?7nD-R~rmNy>tU^%R(m3wWNqK`?hKkE!vP{QN!*`r%afZk!3t`>6;T0%y@`5dF(!CVP(g>o)Kgr2;DR`Jo;RGj5P@M$%<<GbMxJBCvb81}17=HuGgWFF7XU!zTt{JnXQ)%oBz2V?~;=<9S`Ih;AP%6I*N?)H@<JY{f+V;bcW(#&35yeoA29q0UL%U0Cl>vn13$yzOLZs=1&e_qYna6V<15c!xZ!_B<fK0+eWkD<Som1;YMs2LYf<+Eg2uW3PTDN5!$f6Kdtzz6JoI#}(7g_^|Cls#7{!6(-jSd*i%O(H|4BcFnsOgE(&oWI;9yH;NZQx1;(9(Gd1eL0UhpG_rJlwf=(s<oYKAbaujC)nG^47&78OYf25iN8+Ug}r%J78-mm3xnGn%WGSvOy#N;y;6C*)tm4D90Oe~t~0J`N_<XM`|2WkU|hvd7sLE8##V4fI!2N1_Ni7|JZBIdbvCICHlaI8Ju8#T(B?k=|C<+2BM;>)QzN*&XZ$jPsss6xoX_$JkiDcg*m%>aHms<Dgw)$#XR+V4AO|i3Sp6;x&x7HO`R(j&hh}-}*f;`Qg);XMem;tyDhTh7Jr4Mebq8I301S=tJ*Hd3I>PGJq^8|pkYrk!Eok6j#69!-KpA4U=|dt8WYY7H`+IYX=JWFloONQ=m<Oxhf>Jk=hi2@%PMA4(nOFE2u|8?gnPsMZav}$Ax9Wk5N-!2VC;d2_`fF6f7yJFNyQ<VWwEI3}=<GOOEi1?Qtvc_7wVf*99k1%Y;|&s2p~PoK%rbcks~R!Y@8*iC1f$7cSQ@xOw+zp_{o@Ix`n$dhR_h5bKgL6M^kOk$sSRHZmv`6a{>p$@jov*lFnEvIR;dGbq4n|}d|iN1Qk8tOy=j@r2!M%JRo<A#<1eD{2~3<e%N+Mk2<xHxj>6e1!VN|{Pcy2{M0DmeY4MSOBvgav5$z(9rQF_NcnF|fn-L;3J0VRY=i@bJ?-p!Pt4VREnAEP_<E@KX+T<O9r#Bb{yGPf22&ODO*;J_CU`46JKCj{khonJG8rMbxr=QH48ukQ+oFq4w7jhlMi>a_Q!}GSB4E<h`RTFd1^tL6ZJq}gru-M>byq|1aFMW8&jZ0sf7D%5P&VT(^m{z0dB9GiRc-a10N6wxre`v6J6M0fA)GGYv35EpRZZN!6fL@C{eVX_Fte1Z0Y23b_ATiGUv;d#7aX0`2Q&)BxbtKt5oMERbXxD?;XHgKB;~(q>Vk`<4fsqqtEL8qjp4x=F?;_mui?J^-ECA=}N^nrR8kR#r2REBFu-9&N2x7h=K3)Fy7`f4w4^ZN!%0kV@W$oBo{Ym$&vhL8^%X2q5=E{nI&mpX=yvKFf#MPg3PP-@HjYX35UWD+T=Tul<+itTR@3izhXsI8R?tMGMzq8YB>)Pjts_@x;-1{q_**tH$a>)HQ)OMpxdi-{NGTpSKO>|?u&lTwZzrD24y)o}k_D&-W0q*G=`-4`Qd>2`I0~_g}8NB^tbEsIOmGMmKE}0)=p??J~f}>&@u<$sL#G39STmt(;z7RfVkz1|ngR6SJRYRmx2OE)GTW%F*AIDUAeDr$WXZJ@{A-cGcY4K17{dDmR?or$FX@>*MUGW7!#dp}cj~kVM@R$Cy0(Zpt$(BDSIDLusD-z?(m~t>e2|-?DW4gUM%eb_hW{JDL9X?BWy|^7X-3Pn5_aezPHOIWyr$w-!C)++T|GX{w3G%p_we{WcM}7p6Xogu*J-u4wnd^d`ugtDo5*9lAqWjpGt9P<@hW%R;L{9DZq{*I^3jkTPdfaV5F+*Rd?oDQJL4vFt=;mKA@g)euLaoM9tBhGSw!X1lHsf^VpPpSYFy6Dcpwig9EZ6?|TwWH<VjL^Eva|NBCkNo<Rpk#=$XWY)w5;Iojnm#|{D~@50ac#6l2jTyX~2BwPf&b64nbL;DG6$K%pMsT)Vz~+JbvYxG1hpi;LS7TAA+8e?y}EkNQb^1Al;8U8|_~8`nh6y#Pa=Zj*l0jJ_>O(_#Hpr@nt0S;lDJDW8&^pnHY`7;hJjjH;_gNp;=FLP`>GI8(qNXv;M^@?Vfbk{3cexkb=~wY$dZ-FAS-UnQU&QPO{O+`}BUCB}4quK<qnzpZ06RiNAz4M+(1ADTMxx0(0@oDhHSYYUjB+?A2EI96ZX&x>uZcql85t_;tG;y@J`!s5KR;TpOqHie6;(2}B5_4qdWmZQs}~*AoCgkQcp^@4-eD>yDC1bGn3AErR?w`O;Dcl^QnX0IaJtPOx$vyH8;HJ9ppvd3Srg9<ZyX^g{)&XUza`z?<Zy9pp_Y9V@)Qq1OQpF!+TpDo3d!f0LyQ1d-CXHi_$J9x=TWO5q-LzfQeiUSTJq7OeZ-L6M^3U6<%_6@4A)(`S9yJ>O_|bGuQMC38plhIBqM1AM`6ALiNV-&fVKHQ0(DJ0b@2YHfD8DAm6q2x$O5tJ7Zkr-#|>I2<1sVADyq)%UBWyeMx09?iRE?wH+qayQ$ZD*^=|<tyhN%XjqALaG9+7@ek6r%-3=x#U`8E4v~6%JrO18mGw!k7qDA%1NOyH!l;)RjI*a{H+IeZN#Sfh&`6hroQf<c7x&(GE${Ftr6#ZdKt~1;CaIUNB>m$bkuy%yBEh4-)K1Irpq<I*<3rPi*TscjcsiOTyLOMYXl9M!{MOeLM6R&|7h(<Jr49Z&$bnZSm6f`zcul}Gg-cZ@`|%Nw-~O{bL;q<9&)4+4RutGix>(T<N56@p8fV~FYZ#;yz}p+I=d_NRdwxA`n3|SrL8;ei{+=@vFks1JbV<VUT|Ey6>pJ1#RS_-e90mXyD_LqAD**->S47(5ojVY{D(mNtutn#Np&8pTyKZ$@oNJ4&Ckd;Ix%KnD541<W#9w*Qnsv*kOY5Kqebu`oP;8!jQGRNoZRxPYnk$n258!rQ2@-CG+Wd6RasS~z95|)us3-M@btUH)FXlS<?2=rmNVU=!&@!7+Jn~ff`?U$e$%C4Ub~>nN=D+mf0IdubuLM<ZGEPzpcB8e^8RYjy$<x3^G1#}oWp3jb#>fyux)mEk4O7M7BTHd7!W^@qJBJ72bmiqLI<T8+;kp{Zc`c|tE-D{@WT~QqTAyFDEh?)iCF@}b@tVr4SzRJ;S_C=ZngBm;|8nW?AB5jwLdfY8)_FJl!fRJGx2tPt?7K^#P)ZGTpvG3pPY8+qd&ZTNReQk<!OeD4}cgub$=bzhs<p`IxuG3udlb52K`Pc%3`P(sSx&6#>1tSF0~oG!X07fwWIX)GMz6)j1NnXn<$^HJ?<OZ$@5^t?6I$N>QH0~w(^%M?dEOnv2DUtYj<0TrB=5Tu3Y`^G?umBV-~b=j2q$55=rakP6*M$6<vJUG#9W;qNgfL-Hx9Aln?&N_Ihsl8?aTTrjpi5dvW7v@Z)}VIrDX!Iv#j_u#9<ZHFp1a;R|~a1u6&21(U3`iB3S#L$#TZ_ol{<P*#h61+F;?I7`O0IdJ4o;Z#<BC5GN*TkW=xLwz~cS&NfCOK$;u>V0SJ9jSAi0Ti_2$UXEf2U?@jD9+FPx6AIs!2_uh?s-@KDkfkYu6N8>C8sGd+<FrQpFa8{&A(>tjgFGuwc8ZKT*5~^%WaFxhri{XbPST*@ESe~f~9JY+i-gpUnGw}s<*AQjU#vNbUT$@-;qAuun1qyASV`_ei_MIWliYRYnY!|upRkEY);1G=K|jXO}+;+&-ZmbRkGEfY=8d1b`Lyi#+Y18ku(<5MOgfZaoKcNK4yT$nVMW)o31%hl+zR_I*;U=To>E<``0d)-^-@!<Agi!?Yuiq9ATk)9BsOtB`jUg*;;DUvAUKt`p~^;?Z^N`qHs_i*J|iVj#xSFMvWOut{i=byYJg-?zAgVR!dI-Vj16P(_h)J;d^(YJN&IH-uz?hmh@eF`+D0SNC!so)x!*(^dB*s2$jOBn<>LX>yVgK54-yxtIwW3>rVCYayvQi?hyaUl9=7RZm?GI_;PfE;<Vz2_wV)u4N!}&jx6V-nhMdkXzTjsBycvhBZI4r)a&Zf>C;c~KvXvHAR0y&w=xZEGT2|KWygN^B0jy<K2W}R8dKJjFLQR%gzP)bju&%M%Zl5CYC!cDkzR@q50_QOfn6L9@9OVaREg1$)9ER<4#E_aV|8-ad`>|n^7lq_xxCl)&j0J~%n;3wI^CMTSN_qN{<W(~y?!!Rg%s)+Ot-1klfZ*UFNl%TKW;ndvs^PXHxk9I)4Fn}oiN$8zWi?Z93|9i-df8ye2_~k=!JOMqVK)-tIpVNBA5<!--lL@aKQ|#Ne`i#SDdj8CEdW?8+Hre5xJ#RQ|s+4yTNpJdPe;u2NF!HIP8KqrdZ?NL_3z!fpCAP<F#kaVkcB|TbLERff%7p-ZtHEHxJwXA1;@E-9b`xzMa}{XrCZ=r}|2)+p>GEU43x?TU|?L29z-TYxQ`l=<~g05%URXftz)X4tC$x<Q4w<@C(J$w|PCt?P>L>d{f!4&JI5P+$3q~T|q8~Of0{DpC|l8ph5lD+dRr@G=ZO;cYX91cAD8*QWU2dTGbxWz%wlB?nJ3Qo(GKs2G{26nmV0!k_vv(9X06QCz;lvBAZ7){W&8{Bh&yJQ>?SJ5gLQU-!m7(NA0TB2Ph>4dr)e=>c7V7q|GXm9%0we2Ba?TxgL&YjPo-~Kopk?$?9E3)kb-JZIWX%e~^mY4q$*Vg{=t}?)OQiLt0nG7t}1wex$3R#uDoV#q|7;SL>d!n9*P&$&RP(9iz9-cLCUb4yH)WzpWZw@gpp^V!bI&vhMKNAOsh--`!($Z<K^vpSH(85hy3*{ex`U?BJLztY&0v@i#+GNQkVV4iHFH*tt!P0(-9&y^6UhA38ZzTm6q!#?nF}6Ynnww)^aG{o(oWS0t2zUEr|{-3RZ%iC7XdYH5n@X~$!lOEJ2&6sN=sEzT-R>@WdnnkX$k=LO7K%6i<owb5m5KN%X4a$er0>kJ}kQ1db9ytB#2G0~z;z55Z^UD@3V57J%kB1yT(;<ew>A@3ZGXIm!t>B=Voz}%I@`3mfc&k55x>vET77KR|J$5S|eT&jQmIx@J+BpNlp*#!L4{*3+s0XB}#ytMS{fC@Sz@OJ7B=V^W~nOiBq5-jmK+(KT;QM+8>ql%K+Upps&QXBN)@6vmIrh%Fuht?qP*2l~REBwU0w3=e}#d7&CYJ*u&xxE5lm*jJN+S%-OV{f(>B7W+E>*2>Pzk*j&e0jo_Z;14}3!+=wNXIUC88HX&0}DWfJg+BWvfCY3K5z=hFVCIT8}4?vJzR@gW^31Oi2TN%%YdHm2kK&MuAxKYC|9y{A8&5aptM5kdHYM)-^t>)ZMP6TdXg>S^t=J&C3*Q%qtZI}nxQL|^2IdnzA*oFdf0<sejiMDKa^2_$X6lb#;y?)?O?S6taFspHNCO*xox&gJ}UqXT3LhJq?I21mg;vwJ7JrH>vH(~&i<yS)NPGcbST-}&PuPb$V+j2NvaC6yZL+44#*b5<bhUMt^9#d)$7HpiVd<__B7c<r5Xia_^Wkw|2G_+Y{WR)ay^nxKjyQuN_3asnMr@Y>-Sop1=0$<Zb$o}vmg2ts~H8r%jf`F=1<9f(W7<I%lKY#X8Y4&t8VJcZVY*|3WoOJ4>B1}znI{hN7fJ6`t6Q8+p7h%hnA0$l_#^?$SnArW<d2y4qkd!SUoJMyfTl;NvyV`z`Jo*V~p+bE!H%l!`7<U(Nzno2SNUoryF)zA@DGs%qyMx1~d`MeHTTqHr;Z>Z9nyiemxg%(}TAMe`PV^FB1qDI8ve4+TDHkeN>{$@t4KZE@|`+TrnT(r}kg&*{r>9gRUGHfvXf)hYFOMBYeVN5BIi2NR1gR7ybRAWsg?9IWT>XWHVAbQWCYzxv9sG$ItqL4pFA}OFR>YFd=<|K3(pqb;Ere)>H!#h4#WPM_#{C*HWCb8uE93b`XZa_NsoQM0&L}b4jUIt7oe1c-`)WpZfqCvTyM+#?*bYX?;@0Ita&h#Q_qSyB*8(T_a~jH`-Q^tI=){$@HZ6F2YOv-SNf^IvA%}RM#)mL&~{pXyp0v^{&6e`O#Wii1lLEr&V>i$XZMoeXD*iAh$JVVk7uTX_xB_w_SV6V6;|co~{eC*+ki-#=Dq-oxLHrk&sp!f;5%-A=L}8V7(rzO68L*b*lfEfCN=fE>r2<u76pe{x05lERY}4{alLlhU~Ysd2RdJb8vWFU(P+FKzpmddN-MB?Y94<pmQqI#|9dY-<`U;JUe3mSq}!*snJ?~ephl)FMHj>E&<}}O%4&hvL2eyAb01LmJ;emzf7@@qsZ}QTDQrGSShO=^WgWhFK;Yr1E#!lwuw<-*uj0EPmzD!H+dAT-K*UgZMH!4eEr(3-7K`NiE2I_cQd5pq1k};>12IQHv+IZMFk;?lQnn2qWZQYcPF=F2o*&u>6)b5k5lmRe7t9Dh_$u0z}5HY=0^<)?X>r9+u6AlOAWDA6_f0G1(_ov{<4CMifvgPrut6JI#TV}F*d?=JgyN*nf4a#oZKFh9IXtr9NqQu%KqN%CeZt-POh_e4u{syy0V_|n-I7S%~_6L7);=*)9<H|x!=?~)BbZm6K%mC8QmTH=?pNWwL`D@vogA4^Dg?duo=kP%cr#uRgVT58vK{|YkQ}kWe>>RUPxpM&J_~27xb<2*#mYKmD~w|8d!TzT;t$;ZpT7~CMd|gh37|rm=BB5;A7soLSV_v7)9uQ%Ri92?0UEU9UGqrBN`IFa<JQouZ^Cc?q3s(Ocn3{`AGI?uaSMbcFG$YVNmwiV2a-m<BSwt`dNCdI&tZZ;*!H_P2+6^F?<i}#}}(qHQGS^#Ytr14s5}3x>#f{_IOwhb6STs@p1KNTt&ZKglB#+9^6;?wh0X9b41pXGH_<SQd|SI+2Q~0yP7hjeY5>Ba?X%O<M`2yvFiI*xvzg=Jb4hn`RPfj^H;k+Atr-1<(T;asjlYwqGLv)Od?(Ey>W@Y9R}E-|MqBdHD5b`tT{L5)sAm3K~iIRtwhRHU*`d`C=hNWjz|62b38w$K-m@V-agN?o$=asfc4iZ5(9{a{KSY3Ni*85(i61l+rxfSsYX`y-K;pq3Pt&4hL10sy^8IB?2~waC9gU}bW9k%>;W_-e@1wvb3L6-3-{nO-D@C3n{ohLG_GFODphZ2L0|k4m1tlL_{J`Sec*GPAJ`*MX=T)<JHeGOUiIjgeEM)NKkgMeF&ys$;ZT(%IGL`n4`mDvZ6{a|!_v0vpC$<z=!#<L6W1dazbR$4-$PJZ{E!>UnVw|U6y}_}Y=&IDTi5vRd*JB-CrfJ|EHzxo0ejt;jiH-Xz|B>a{oKEI9vVw1ylreLYbGz{E<;C0>?D3lEZq)L4SSoo=&q{s+J>oA24Q7-d>%w#1oTI;J0G&BeKmr~&Uy&(@bE(JE^oNaTW@K$r>FNU>I|z&tJP5n(-X?2)4(gz97ab$V@0+9P~gS&cXV1hr@tPC7LyKOslD=BYt*(mT1WN^Y_<8z)W<mKs~vg$2QB*@b3Rm5eu|7<OR1SKAW=Pki_Hyb!7K4)O-(n@t0X3fYq`H3+=i>`dU?L7?-E;rb!c%Blk<~m&rRCj?S9ki&iTMGI@f3T=I&LUPA@lp$~{yxZdPup$nMZ#8BISJ?G+oZHj_avtbk9N=$}nwI=kN)wfq`^3B)2ukd$%`7&A|+-A?w(uivkN9ai+_D<jLiK7CZ3euFcpp$8LtX}tKJc9V6zf%K+}$;8)gRZd63Av%`_gcFH&n{m5Rv4W+s6kOx&TsqXvnB~W9#badDm=u-csQv3RhU9P?i5T1#uMH+3M?kiXczLx}*4ci2KJyvNs(+GR&8jEL)5gi!%sZMn2m`njZmTAZ51;8OTe062UH8rdzuATdCx@J8!iCuY+B{DrfqIBa2|q!y-am4~?pnA<zh@*fe?@yR`>p^*l$s73Pl9SU2FGjl{>~C>H>yDb3{Qv9Hl?u+{f*l0iiz;}X7Xfqm@YtjJRVOvr@tw@Hz%zQ<JYbsb>~6PKvpeO3=O@=7hG#>Tbu6r6^?mE`$(E|&~w2EH&Skupqc)nd21}S+6-z!^3O9!<dPQ)@4<6pa!u4e#Y_&@zfBg|P8yEJm(>TY{zh!*oi?1(o^#6~N7ngM3h&$Rd*sZ1N)0)s<>d3E=gtA#4hOrNr*si6{*y%UCQFABxjgA(UuBQ%zUG>Jh!YCFRv)S&QBIB?@Up663yqNLT#8+r7`K|LOoTlhYg>DFlIX*GG(+Son)zY+?qwH$;*QZlN)LCvdE6WvTSP}$F=j8ArN9Sx1DS!q0?10;`v{$*OOJ?r!9P@-ttem3WBbzTu_JpXF|$f}DA;g+JB%%=3r{Wq()gCvtGXr4hi1O?WXErs!+A7?ALk*HTMtHxlHY8DHgaNPKsVjCr~MONa|Ya#1OHN69~;W``CbtVlR5Wea__?{Q0iJy*fMX)l7W%)p)riQ<4di#w&Ygh2)3xUobMm%v_mYNk2iA9^%4kf5aQLl^&Q#g`wj3<?Pc6~c4lW+fRN6xraIBH^IOR+qQ#7g(GH{%{qPq$t4bBReBuSzwb^N%ck`7w!$$5zr|x>KV$5HwU`ZlSL?d6$$>M?8O5eb<T1yARUTbk4H_jV{`ePIpId3rYAZ41hXzv?YX*XVvP~nztuO#dCrA+|0gF*CZp6K!`-+2EF#NkfjBjRA%2JzTi&i4Bm`-FV&ei*_XI(<gF-8?jv+0Y*==LxYKjf%<DY#hoe)@S`H))0L=zL>1u>F?pDOAL~pMAFh|QeQ`AMUl{~Tl3?_KB|K_19#RG&duvE4J`qb?~bCRJqpnArBbY<`?^sB*I`MN7IAXdSAQG;jqtoc>o2vVRtTt}5+qQmH0QPu1lI9h`|He)t#h{IUTv~~`@}}a^D94H>GUXzA@ouWo)4n(`8jC!6xQ#nM`eqd8zNp#kH8fBwPXeb_zyG~93ruy4UhMF>wdrQYf4o8wl|d0oE^a0+kh@2vri)EN72Dd-J79&s7ymcl{bizz~8A-xy6H9<#L6%<1ki@pm%&z%NKUKyZx$Mlz7@lQ(yM(>2!Sb3h^nDm9O0Mv(w6L_M8fvt>Zu6X!`o>?f`8yGSUWM@T0T6n)aJJ@C(WHx5w?<`)-{GH4|22Dm7-?!+UW?dSvT-14?{fttJ~N^U&Pq^7;^HO^T$C0YD@=U8eu~o8N5M93_qKQ5d+@Va8LKxWaHIMyA`HcsKnEFsd^LbD6+p2=kD$hU@Rfe$;4@FV^49giH4iuv(38Z?q5Bm;0vA-P)4<+RMv7uMO9?r?(;Wuc6oculhB>)Kc3ze89B&Bx&?-GGlMuK2h1U(-&3Q?eFK$px4`Gne#)zQO{Kn=~BHlx5@JQe=p}6d_ZIG{R>cROOroF-QjcagS6^egW8S3SVOu<17x#^|Hg24IEJkg#vD4x{AyNJ2bSj7H*uY9+uo0VGClc$o81AjJ@&|x*1t4~qkP9N3aD?(4smVY+QY^-ChCIYq&OUmUD-qI-QjWtn&e_Wy7L0Iy$tL6_?De5Uf>!(sNaF9djHiO^tr?93M=BhI1EoRCBISELT|xD`Iy6Ad8`bO&7bQTfc%X)Zq;4)ceeXjL&T8f=B0&mZvZTYHI@&M6@9~8YNv|SEt6`b*PjV~Y(rDB-M8txw{xbeacaZWiFC!!xF|0-Tp^%KI(p0XrO@rO%3)i@^aYJJach81#^Tr5_IUaNUDDw|_+l+jv&cmf4m39V0Q_UbEb==`tFRg@m>CZEQ<;%Wb~eVqOEt@0_;)qw-ZqZcSnm&LD}UxrKkZDbQU8t#k7v3luP%(gRpF^GLHQ}@L!YwQu{@G;e_Y^VS1CR()+H+OiOlWZlWWGc)RsG#)YgS}Cahn#t7x@$t^IgTgQ7^gSob2ny0^3s{1ITAbaAQSW7ILAFtyNUS@b7kaEJ1i*X#LoX}*T^{KM{D$Gjen-r-$NtbxT6p}vs9rj*deC}uS2Be1NOf`ev0>TlK^_$z?54CtK_;=FnX-MQlpS9*LHr+*U9U$f~8B<x$WSvOzfl<}SkdKJ#c{>zmHoP0nZt}U%c8>48?N6oVLwPJfUJT>{RE_0)fQ*!gwF4G9W3Pu*JO*@n42Ds^)af|u+-&f<^4>vWt7alI((FoB&Z!hu-__4AEtRBv=a^6XuInm9h`F^myf9B2-Xw+-fW0Wo3a@A>8?E7zkp<UH&V8?86aUwg(Pi8qQ@w#$H9HUvkHxjg2yE^fO6MM7hvAIus?A{g^T|<}e?=bsLy>kBAg{a&x^>~US(G!u}<n~$4I~HW@Ecdx>xE-`LbmjP~Tm&~BEyk57B2IaMMbcC_aFCCL87)nHH7AEvF1eI>AcRUY$!ZXXya~)5O|<zIuS}npW^<Wpc1GM!bpvTI^<&ql)YC!5maX(X=y@wd9T<{jU$|qNJu3YfwX2NTnmg3058Sg_K7CQw0*xw{35uGd6^%?=jSBhM*Xo1MnB5MI5VwH#dIeUd>YQ4n>G1aChj4GnvA(vKl-Y4lEKI_yjxG!}l;+Vb1}^&nIdr6hyaWd?`qn?x%U5e`yp8V(0~aTEI2&Fy3>gxC9eeKfSNb8yy7ZlY3j^{MB2R(hyV`yh4(n9qIDaD(NEzLcN#D6TGJmEBnmcFVBX7`ih+2%klV{ZeB8e8=&bPf}_ovr9ytk%%(sWAYG56lSs@34%Mz$QVV858cEH=O0P3OMAO$v4$f?k+E-1&2$%VO4w<9E~I9y>@$=B=|+GPKGKL$U3H`O!62A7NLw8<S1e2kV6_uIDb(#*{o0xp7n63g9nvS#l`PmMF7X?&`;}yFTBSt=s#7&pVBz+lolUhI^~*RS&iL_UP(nUS&qJJxlvDQJW#258e!>E2Bq{lFze2@DBY?ALY)=I7mRF**BZsyD<SiPT}A2$0@XtXC;Xl#SpL&nGIg8=iyY5q<hAEh!zHXg1{TvCQF|EN^{>oGS+p^*vqx@7DvQj5n$s6qDs8W6r--08M0rW&gB<B^u*TrdFX$3+8f@yzO8QfxGMG#<Lp-*PG0hCuU@z^<-X0;W+e;(wnD0~{P>(VaIL52@2-Qq!Y9Kpxt>biLMC)<dZ;&lT+^eXIoQN@tyZv^$uFP%`>LNepx$MB8G~nId4)fJjnb=C>wLoRzPn)m%X+t0qsau*ZPDW~qUdEGdg`q2^6z5tmqjCaZ{1V;L0=E7<(!%t1A%0h@LC$TQ3zX?>fdKwYemfiy4n3bSgWiE;|`QArT1?y=B*oML2kQ&Khp0bAI)D!&O$?4ASeE9{yC_d!k#VUmGOte&I@IxD)#_yjnI;wqYCN1#zYZR*I`oAtJ}eNEzURnTJ3pKnwLYfD0`Q4oq?p*XOVjRo3v5=i-6}RA-AoFDeKU0eISmDCHX!8@J*{6)x+vi(4q$>T1GMJ;{clVcMTMAJH7Rh0UH7ZLAHEkqg(q@D;Qdd8-`w<US966&~?&3#oSypR+TpEEYraXnkP}bJ#Y6zF+P0T=hY<ja$slus;O3US|u<4{p^lgk$)NCRe|%I#FFQJu1eiwXO^z{Op_erHqeFrqPPJpFFh;sNUY({iC9Hx`ng@a{5{P}G%F9u*+{P3TJ*Yj3*py)Ndg9dj)f!FscBfI#lL&^T9A`4_r<kA>tM}Bi7msblWn^d0qLLs8m70~`v_jK;O<74d?{wZZBOj;c87u5cd*2Fk8NK5+2n#OTjS+az{D5XgXq(&Kdz_!{T5tjUwhB0!A+C`drUo)V=~OT_6I-_WrK~8WOsdS*t_H3w#=;GrVzD3z0M8E+0(%!>+N0^pe%95Ha2k@cfj&>Q5V2u9Ya1fvjrYQB>>^i+^OXC9pIvEeLkslw&y@GtDVH@<|J2u-$pCXOZ-Y{2r*p{KVv#AB<tS04%SUyE?$kQSaYUHV;qzXG`$Xwlb-SIbK%Xynrgm4as76?!hQwfMO?pkUTRk@I^Q#JeRi0dbo8Y0V13rq|2y#=Z`KtXyWK#1t;vk9p|2Z4p`A(EptsTF>dF4gJJM=@NZO}r54h7T*abgr3|$Hf(ep;XL<(|fu)7?Vk8{_efbJhCS^M>nNJrbdLjO>QlE<$5ApVLUZg>HY!Rj=-ULTL!he)M)_s{*f3x_PIF-!Glj!>=Gl=y5MuI*2!qg)0J<@V#)b|)7DCKsxksu{sn&g_YJq7lw}ay&Pp^mjjRPJo#mlQ`|VO%_hF%TQU^PuOqhbCOkdPSu(xn(-6b!(`?lytAv`>Gw;KQO<YXTsCM8j%R@e+)lF-J{!*#fAx^2(nvud;ZM_}+sn4w@~eC?3!Eo)&0Teayc}l5DIK)f)S55cYTJ#8+vpLypoAPeu>lJ1io#%Bys8X-44UsL_-q`Fmup|mN%*0*rc^SmanN0xI0_u2JMQ`2>id(muggbY!4T?<)aZn2TuxJ^pI;j7uwuiVL7#DpoB`6u?HU7wd0<=3z+$o6J!0YKz-E~HuKqsto5LWEeuSw^j3+~F|KMFPenHibb)_-m&y8d`9nCoB@RkLuiPZGZJb1;|&O_@Bs_MQJ-&eHUjryL9_oHlb>e7L%AXc&^&ZyQJk0)@yOO!2GG_LeFUs2g(_{zhlqefZZZaQ`6e154UQSNdUK!xTl(ye6`Sj^3+wNFW`x44rJX4q6|@{NMK0)4!Q;_Ww9{8nA<F*#c69C|J0{)(0^;?>t`Ry0_n&FNYSz6`)kA-?`Pe24SCu8vseJ=~XAdmF<~*}hFjHr`sFV=KpAc%!qp^?uGP9p7o~fZthUdw^@feryO6#5i~A=p@k|o&8>;FGre1pBsF&X6ubu(IMP=k|X7UhveRVfz;uSQbW{xbyxVaPDBoEkFM2|U`}tX5h2!kA7_F0DdBPZv58dUkL_RBJ#mYj(tI4kqQ%#2l4<tv!A?81N`O8{%jXu4LnCGI#|7;3YI?Ao9f9JM3?v4n(Rm9W%hQxSefp5<mhHozWzTQ<tZS9t$K_zj9G`vdo2R7KLguxiGGyE2H}GCHwiqaxGy+$ku@^HLeuQqP1+JcxsWw;6-_D?T@3ZkqO%|${yt|QBcIUej(hI#F!3S}0AK3)j)C}0`;SC!d-1*@L91ha{^U9vf#q>-+DOrQ#vbE#_BKB5_)M*0O&vY^TV0h0{r5YTn!t2?`!pq=s8VKlNXg&60)+Ay|`Qor2Z<zD=ak!I*()~DprnK$l;f3g-;@4FZ!hSvlt267>FFPv|8tsI7*i62ibvbOOKi+>jD=_V09rAR$E*}mS^SF*cZQy=LoR+cU)gch`X#4S18`O8Vx4gZA`^s^3y1qK!E6`{Zbos0E--Y-E`WYN-XH$MF=;KRnYO|0b>dJYj@!Z2|oXifo8zj|sQK<~fb!|`HmIfC`*R1$m%#JV%>?z!WKc)xbgp+<Y7Sry*#m>a${g^~yTy-}y<J^s+m&l(smfYaV$B&ecufw8VDX`|eI_co|-}Ga=UVb^a^nVP~+xI8c`enXVevJz_j{Yde6F(udX)AwYR|&5SyLDy{ey^9&>32!rcAhc2?L6^M$Ihw>LV!Ge-hiL7)qDFM28bD}mwN5jgtPmP&YJJ^HpSP<x4r9>QU#rDui6BiTsHT_dFU=SIe})_-%yzG8F>TO4OVe=6^#2&wV(T4t4*zw{o{RZ`#<=}Z8U7uqNCO1T7XFT!&5xlvPQ7LhUqb`Gi`5Z^c`_wq}cGRZ>mZab_OGI0yOQ}ZGf0(Q@K5Eq+f12A5?0(s>%Z6v8N(CdM#5U#G9s_x&#~zb?Pn0XSJS>f7`KvAZ&r`|Hj-S>$5O9WFEJ9ZR!xu8jax2iG%#a+TRy_{eGj`&%o-;dxhS`=C2i`ODZ8)ezHEE^-GU#)IMMCY~;1NU#D?JpSG%+KoC7=;Qvs?{B^8Obbq<gt5Mmx4^yul_~SD0@H5qUG!UpLmsyjx<_IZvYpp$c&F2EI`z`J!1Iv|$DhRnO?KXevmTT{LMg})raM-fT(3k0k(;VQ8<NH@<cQO``u*^O0Hrc$&{xGKo?Q_+(E<V4iY7om6PjnAVi!!O$lib*AFS_3R_{L~At~FRGZ#L%hZY35TB312?*h5Y`CB_yeD6w^QHjgrRo0%|fx|^k*(1+$4kCoun+<41p&xzIMtXtU%A&S8B`62Q|3+$dVtBj7}6Fjb)JF%^Ox+&};5eambuHt+rW)-jLrnK&73NAlPC(4tR>e<L`Rn{N#?n+E**Kd6JdIzl`1Ym2zb(8zB4FJGqXamjGnt%1VM-xETjx!H)$Cs(;p`E0ivD-;b*!JSK;z8QHe|wOuUf2!aT8NJYAsUbMYnt8?@wPvx>k5Q>GP%fjwrE$1jr8~M?(8rSTq;^u!Evj%yypCV+gTBZd!xVZ{@Kk8Hj2;eqWSp7J08|2#cR4&Gg*+q(E%pJ1yjrK#AQ#TClTWP|Epw^I#YY(m5Po3amB(G^ETBbqCs4M3w>~1Xzv!q8j<8Y9){6;j_ylh=WORaVT+8osru<w7oB@OBsTK@>+4~-ufvW~`|Z=PvIHDvWw#G3bc!<(yY$Le*m;NBV#Eo80F54ha@T!`s|IE^e{k<{i*N6@7VGf?+Nnv;#`Ja7I3Qp6$e&m8cI|z-4|1LoHyX~&>tJWQw*_$;bO29*v{@epC~-7VYtttX;m#Q6qOVW;J5&!bcaIyB`p<Ye6fj9wH=y;8qcd+lmJ7h}XKB!=rWv$gin6qjEn3tck&qTdixfWl`+MFmkm)S<eO)s;M;?IkQQn1zcRw3~PaG5c4T#r=>y@weUsI5v!RK>-$GKC7zkU27n!1!lv&8IN^uxA61-wv~uiGx&r*tFkZ0_r)D|N>pPUh#9u5F8Z_|@ZjMvp6es;*M}SP4(1LcXfjMiVpRAx`hER~JLiJzUTo;NM!6KZrS2aWB31Khq9(GM+&C>g|G81!)4x6yi>MHLGfDqG;Ie!Q<`=zTV29`rOzrKChQ5qAtS5qg5w~z158y!^g1I7{8IL@hulhA)NkN?&Gc1GW<*LZaWEdbx>91uDqWrtFE>BwKj`Yv2(cod0-yx;m-f|2JaVe$`k%4tY(ytHZg7J64Twd9i(cflo#7RGw^xiKK1#_3f)dEeiUsskd<q<+}*%MvXDB|sGcoms|}UV8H4dYEZp~e8WgYVIB%w#%PSp@yR06o{Pr_OXK5<(?~c?Q30-%;^_@9RKAW4Z<BC_NZ$M#r`iciG>6<HJG@8_=`}NNJ{jn{SP0e3#y4%4i{6IIg=eJSoO)x*d$;In@fbohmru^%GqBw6)?gW#F$q{VC;*BzD9YQw;3H;-yHfGkhk6#}ec&F|0=~;HuOUf0xWB+D;u-nw@l~J|njy~_geW}RsETXTuL2Kpc2QYMfrcKem+Q`v>miwQ(YTmCcdF?y<d%iXQ9zMg4qTj1}yNXGV2aD6#Tr<jYvSCw-9Q$FxIiuo}wAU151`St~I=zK@nV9+vFGL6bfG;(~<Hnf{j&)ROD3z$r>-negQ+#E+!S3I9dO5s5gxu4_#{`*o8?|^&DHiCI>!VJ3gQD`6zA`KU`P%p4YcD{49kjZnY7=6#>T=tvF=(=9!3hkKB@X2|ZAy`IF;}GB4EdngeCO(PkBp*iM-QL9(+ZmYh}wDlXUy9a(mE>%2J0*1)D7Xjv6?Y6z3JZ;mS<sNE2}lF(z<0dbX^ody;62j)!Dw$sqFlO^>42$U#8n`MIu$v$NtkffjC$SdwQL1<jfX;J`4+|Ui;aiDEvCzY`pmTK)1vYtUi<*$Mk+*J0KCs>5m%fYsV8p|JW-(2QHssdAZ)M=j{TR8aXv+NYfCJ&QiMPiEi84YrjlO<I>m5(YwEqov%I9Fk^WMAl>L^iBRq2MvCx*MNq%ikJp^<dEuZ-KS0^`F8rZMnHzc1oFX_let~Nbnfdj_CN?&k9WfE*=AoxpH?{&CI;!q0$9Xgtw7os7mMoAhomY|M<u3Q%yRca!Hy4rm9Z%)c3@<CkeoV#guUAu}91;6ye-~G4(B804JpvQ5sW&d>e?c$bzr23`ZJHkKY_Z!5+;gqv)(NN}qcZ+G{l4-}_wqb@;_dR*<y-S^L4PHw((NCH*MOL<NOE`D_zUOFsm~pn-@)=cE?p$D%yAC@)tB?Ce$dR%mXe&F(bW#1N9^{wV7&JdXN=+c>~E4(g1^|Ux%GfzM(_-=o(iZidg^;+Mf8#iW3(l&fcDpib2H43S);N4@9n-b!XW-FCev2a8CNI7CCz~e2*3LS7&)$1@0kU02)MnFSQLx#*bb&ZTqU4Joj!G{cH!;<(RjV96B*jqTbB;B<tKXdVI?6S#G<E+>9l!=lcaUCxaWKDi6ToUGx4UqFOg@pcOu86NRmE&EvT#Z`<?pZdf##znrafyO#Jc1J~bCgUk91iK%Sc3g!%?Uq)wi-ZixWgT-`KU!`><>!Fbi@_c(w4^TF?-<3h=^s@%r=F<cML^=@UXID?wc!U(HQymHJB8H8X6{c9hVRuRvUb2Njs&*%6;4FHQ8-*2_k=`-2Pj#-^32m5W(9gw0Ma>8@4y2W=IrzH0;4QFjfvY_2sD>`>VnyA;N0=|yYW>U#+f<i}iIIGsOG~z2Li+t6l6Mk1=ePWF`aos&mfcK(&kA#W|ARob@x@WpWR^Gp1K<gx|u}pp8RG!|IpJBUfe|k!phSILEAKxaR8O+&#N88$5cX~kDw~9lg7c*Kb&E9Md%+!`LijJ3zAFeRBRgX!JgSPcp!myoQSK14syCC{G$R0wW%G4m?+%3#l@@S>uHQ#st#?twdi-a~i7Zqw}0&RmO^`=Dl$9DG(IrF$0-1%-6=G9qG29ZV7UYYWB#%&O9S>8(1QlG*hTAQ1$u<CKz>HDO|)c$wfOV!8aX;!b6T>jeAyVGH!@l*0(6^w<xu-oMvw3!sSZ9VSr>cxfEDl+Gn+Cl$)@uP6Mt>&K-xHu@C6f?iSz4D&b0uSE4t=y<KI@f6mn!G+OF4r^gh56R6`DP#^uJ1Q5Q<?v5`fO0&;>04t1`?gkmTD9R)BUa9e(u368G23)yV}x`1}==(22HE*u2-5m6wv7s6O*N4l0|;{oT?!`IzLzlTjM>=oZL<q?;=0t&4plE-mxSeo&{CC6R00A6=lqwyW2c`zX|TFmpx>iy~gNpyw#yju?x{SlLr<#3nv1qnnrUOw{-T$uLq2U=_^tL{kKlKw{T8+#v;c#F*|kld-wjuzjbT5PrLpFu{K+~T5zbju78}5Y*?f5bO8io5P+L`qT;9dRjC5ye!NWn<SA(`=(O9*&T%5lkQ>H~8!%*E7S`q&jI!zDS$i61-qG|#5Jc);znZt{mR7)B#qp4ZziCCyKy36mw?Fvnv_^TF(YHW8cRh5_S#;Zeyaw8Vm7v$-wZt#d67)AbX2*=9Jws*{)WDP9@Xp89sCj3H?ddHU>Ce(`wMWNl`;j`6*}Pe)w+VWQRb+XRPNLzbaaj*EeZUmq<Fo9xF9dzO^vJcU&A@~%ru3PJdV{~rdZx8H)!xr4aqE2V%(+X{8ZPh-Ue~B3wNJ+H6;10^`!4zmc~|*0E2CEz>K@`x?=Y*oPK5kyy*J7WK!}zK*pIt^>kJ4yl?<ODn8XS+AdnBed{mCrkaW@uDvAvVc9uSjRb_4Z5;PL5?8{SSUOyeq3@R-{4_jPu4!d-QUB$bqQ-Jzu?`O~}4^Iw@WBhOv_;>)-_+8u3Q9WPvgPBHao@cafIr8Stb%E?I6aO$VPm<LKpVc|&*S?)EL+*Fb`s(9@lT7sOK})OtTF8vom-W~-t@iir1`hX#*N*fgA1g;!6@l47z^b|_XZ`Gcfqw*GDeGu<%~<sU=-Y<PcF;3<q=o`()jsC(&<=;`5+PHReAnsTsh^CG(~~aI4Dn#+KXyoU3l%`8KU8=+akh|1Q5^cnaJchma@%yPpHDVo%TDFl1sVe#{$!K-Q{zyz+{J7hsDHb*`J1Bgep5LN&3ZIgrZazo!xwV6bO+LlNTT^ch~C?$RO^!sfs*}&qI%hw+kLC^{CFF*r>45OCi7<+gnX74`FY+;?Ai2g`Fx-WB({DrZ>Hq8+aa)T-?sM(u@f761<d>T@OlY`Mo?RA1vRBOjZhEnN)sC=7#ecXRBjYh@f%i_`|tU><!O5o9%J{uQn@yqg6l1}W>w#rwuY~@c86}v2BL@X<(1cjK9~VJc(lcHtKZxFvIT|uNWzCcarju>%i?4r#Es5U=c6fqfstn_iL%OMG~k<7|KM3W3*Elp*x!0@Z0z6d92qTp+!kmRh<4kXU2^GFtm2*!?dl%29OW_WV-#1;;?5X6lg(67zB+)zryupCvx*Z^gZ*Rn%{c5&eE^E~mI=72=fqdQ<;O>o_=&=dh%LQags+yx!|JN7RE0m{*0dpb#54JVG~Rt1UOd}eMLXYe$h^7zhx!ShgM)fXitcm2E(wMigv4|lyju&{HR}b}Sqs7eNnC1k%(|**2{%HjA_6Ven3J<KU}~6ITD%<O5MO`w;1<4|ZMXVf*hQ5dTP-PT8cyYFb#Qq0CTE#Yk?)kMil;6H`0C{OK8A02ccfoMURs;;TB#(4aD48fyMx(j8}?XCLylqN36lig-_<79FWZ$}KpcU^D`bYUScI+X@HpBziYd2@GrJIyU*}@j@(q67?wi@0_R;Hjgl)vT2$C|sLgE(I8y3cM3pB4?wWbn_$wLuF+kMm2&c`;`3A{;UFBg%pGd2q?QoP>RV71pGti=R2nb!sO#WP6Q<?ZJmnqL>K;i2uo^V!)7`u)Zqwq;KD%l0n?^0`pybbfSD3Ek6^k^Pp$M)QOUd$>BtfP!jA8!|&cdULY|_Vh+0&<G00oV+;{1}Pv&D~?X5!R?!?Q%A{n=JqT}WV;*f&#C+ByPcH@bgD9-2M<iwq_|&~p0QoO9U<5vu<Pe{vD0vI8%v{5dZ`-R&2$@cA0w`^#=~v8(z2yCPXnQSYfkBo%Jl}UM!jb97Wl)orb%*hnk<LM3DmpXZ%Thbw0&%{sV{Tp?bq^k*VD7dbvUp(`5hXKBY)3iPdQm#>V{gRZkMcfcfrQg?rDX_Nz=SlA*m%nAx6#PvDyXYRQU{r+u<ThpW;lGNL>~}t-fFNSU&pB@p6yx+hYhCKmdV<B-P4p03Z7EEk?jR11d<<cQkV+O~(!)w=2E%apYzs>I=@Uh9XICgRU<tu7ZSqT4=$k>EQE?t5zPfal%z3ENZ$<C|O=#u9Y7xyKVm_>HhhfIqtT<-c2@$6?O`LT5r=^v^D>7WAFrRV>RU>M!Q}TGnIl}F{nEJnp!S_oq0N4hVkPuAa|gZxa5Z4fV1^|LS0+yGP`;wefr}EebJzx7u-14(HfYSH?1gk_<lPtTLo7;93ga=_4X5i`aGTpvN_6lZ-hf}yF!|Hd0c)dU~8o@=&_&~CQ04GcPcu!zVN@VORJl{sDSI0Oy94TE=|jVuWDLrt!^t4j{}%-`JFbSdsN&j&4RK1LXZ%+8iV~x64g4!bIE0=@k=dx%4ZE`_zQKe1-1bK;P-CzL^SwOnI+3gxIzwR0EdL7+XpI^p{{M}7bx3di^}E}L|=Re>TNXT#K8CI=ZB@la(#9-Q0(1Wt||oF4bAANVdt3E3O3VPE;MI7cT<s?tJ}MQ(pEZu;qVT`b0IPH<I-qHFC#4N({i)~w$;(RT5s%Opb`R5+dMttdp<l7+((!zSf;6?-S<ZyN0NOV8Vv?E-luG(9X!;boJ;tnp^08er|{y7p0`ab9s_3vUcWc&HI0b7uAeEMnxjK_NnI*e=ve=L{Y%EC;J{iJzft-c$=YS$#myrD)cIFm-&i#PIPCY)^N<fKv*)TiWc=<{ggDMN;rhziY~aQ8v8-+f{6VDNqpN@X(3Q&dO?3`^*PV~>IJ-VhG%3~&0OE-3akC<-0L;9Ra?$@z*>)?C^>pbBsGwh-Uy`Rk1W;c##KX9%x2S~^%^vFkOYur=_*+eW6^FDQh0}dqazbVcud1SKZqdsLZ4TzA9BuADE%#F+`QJCrD0ZWBjiHXEaC`1S%4$0K3)oQk8Z_U<k=shq7p*Vqw2tW)Di|95CBUTNh4b~#SpLcKIr8=WaiHYTtXquhwS7Z9FWF&SyzQK#e{*+p8q3@53S$qzdL7UP8s6SGxTWqxpB_{f7wz!g>c7dw-Qt3DVRs}SzP%SbbL&^vqZ-3i)c}ce^*cfyVPikjXIKsOWcU#8>}nKUkn85O5|$5cB{evhq86pUTm@}9UG+Xz!;{Jaa7Oc4&~mN}c^Ub%72K`JhgS)RwFY~ly>pIgA)H-w)4dZNi<ds2JZT24eRCZP3?kHx>gG`qD>IgKzZ?7ubb{h~zI0crls_tWL6(5=X}<}k5b!e@djlm7ArzMZSAV=H=eRpt_@$C%&$-Fgdbt;%B6plYJr5PTyLu!p*()Bjr#7t+<H!y8*HrSA$?wZ2qr>|$ugTWIZf1?n&Vl2PndApd6*IAq>}X_LzBg$G<_7Xy%XehBXvOLvonWhNqIcI9U^xs@W~o8ex-$AMXWg2<V1~tKbGtTg(Cv0fBUNNZ#}hHN5eFn$HGB=`gGFjIUz!ilFrKJl7hK@MBeU;06e3imjf};a_~Px!t#On`R8#J0YU{nYI}dXOeB)xHj&kLSZRl;SX?W8W^f1UCD)ww@aeo)^3PpG0XWN(4*9q;Zl|v^x{B=Ed&d%hJ9Yh%8bHg|@ZQ_avwdbi4J>7bxi=bbycW*Sh?TNILTSt)|(5)}#iR-HVEX4fdJ$_`6v9$}ae!;sCy6zddbvo3}oR?$xdGwtqcd9!AUybe_KMvl@<@k3xx6rxH0y}5+@cg!X&fFc{(M9$0nXxOyZB28FaFc9sri2PDIVu?3ex3KV1R3{XF9zgkc4>e2m|K$;_JjC?_wRdZ!s6ju`*$_%5bXJloqJUq=R1N<1{|s6<L+hpb!G+G+HKR>B$szBz85RQPDPOb;1MI2Bnf*lsd9=I6A#Y0wFc$7TL&Qf7<5mtaO(SsJ#W@x-cUsyGR8s)V-IUQN4Wy|chdg$jTa$w9Bmej`*oNlB)$g*_u2qo^5O_b|1_{UGg@x$5BH{UJ@7=ujSoK5=rwzvo@VoOJB5FHkk0&XJUN~VJh<+=ht-4Kl?dCH&1*b8;5c;4*Sm0QHy0xNiVw5#cpDA6%XVjSEE-D@5}1#e1h?BD>;%C+VnjIDuY+A;Z*B<97_*)~5cBwn9@bg(Y3(1`fx)*N@TqppKQ2Wj{oUF8;V@j?rz88y%2?cc*SY&eTehbUZ+4nt{V$n*2GwNJ<f*!Vtn)65lZWCu7=9Pf8<^^CiG0K8bUF-qAQbbAs63SC{f)cwXk7kT+SKjuD#LfO8z+L)EIvEITawyZTpSf5_;ZcXr^_Ga2>WQBLz8*k&#z~i+s(+xSzD}*pM^`GyXLkH${$j=pk-fmxxw^wUbSrtEYat@V0%3dQ_D41yg|HN6!l}f)^D#ItT^=-x)I;T%`-76^7p7WMf2tIc#SE|8Ikg!GJ|T=_OP-IW?Et9S+d!9_R@`z)qd4Ht0E0kgbHu-MeB+*y?diz-^>ea>(^&(ep$D9!9ZTXzgy+<d^LF6YAr$iF7T;sEmSJ|wXghg<qOz3zQ#ydkI#d{?RwQG&)&Mw(kn!G8h1L&!X3HN12{(8;IPTXfXocm`|iAdsMR~+eNm3ZG(u^NeLXJvybzczV6P-w!q|@4T^EX6Y3!O?#5t9nfH#X>d2BN<h11dfAk^CTk&{?-LuTKdm-dU70`=TA>>CsOh96(~8*2gTv?*-!Ic++FXKjB#dB8uPWKc(r*#Vn0OA5zz)f4Fj`^TZjq{{iR#0G=J?Z<E4eGZ+&q{@F2NBFCI*~1&s=}&p?BbB!KsP=j8G%PI(Pba$5?nG=-OqAC^ImA)#mcD7~s$!>c)MN9tvd`nug=nRLhf>?K{GgH_Vwx3mx1326ZQ+m+eiAc|M^9tmgKxF2H^86(KWd-*A!G-G+H+65iAh%9hg|>558nL)@FLD83askWDSv*s`EL@J-zZ@+b&=c^suBkhlpVQM6-eG;Q+V;_nrp0Y2ljYDE-wR<cr29Byxy;V!3MVR=;^A^qZx`aF26gdL}xV$Uxpu8_#jTukHT2#bnh%Ax#@D(utfEGqt7Y^gzVd*U432cVrBQY-kPZPSvmMI<iS|P*P({c?C){RA19FDb48WT<Z_4Y|J~G@W8uo1$fk4YH%_m~_+=(dAyXt?o-V;A{)oVCwwu`neX0eW{=8Oo8122Nr7P=3s15#+o;G}CX0~az{YCho!<GICAR&y2h@(>+C_13GSQk~u$yEd&a1BkfZM#Y@zWrnO)WxY0WJ&cm)ohy>8d7KTk6X}Yy(?2%W6t;FJR#xqdN8h`hA#(%eQ*7GX8G7W<}J3V<ij(%GipU;71{<?n~XoA8l{}nJ9l%gm%qn^ylf6QeNROB=(*;8d59c7;k~;2jYf9IMbT~R7R0YUm^@#>jwQDm&@fAdygzc=-?7-kRe=e6p5RJKpP;TK_ha(AY)_tbMMjp6w0Wn;^CkKn*L9(M2@e5v$JH*W!reoQ!ja|Kah`{+r<|zKP+R?Vo^Yi3{^X29<8N^mQ_>)7%UNzT>e`_D(0iMok(wc~u1&PF@Tw^DJv(}a|Bea0PSkM9e}lR|w2b&z8P%-k8e4}9)OObHuMBEPU5eHP^Afj<Q#Snejoe8*PB8X1W3Min-rfKm?>3nnc^2=CF?`gVZ?BuD;~d1!!AHvd8jPbG^K*bt0!^+R+U4th>(sl<Dv`=xA;FM7w|T2ZqPOg@Kk)0OOKiq#rgqcsKWp~2nZwpA0Cdo4$oq4h`6&HgXSQs=YW+3%^(qDINXl!>G#)?-MDCJ*1KuBJbH(QOA?6k4H)<Fcrr39q>lg>S(&+@vzopwo+8y?4r<OnXR`=m^`P~E6d{digW8BuK%b6}#cJL=#B_HGqJF$|C`X|LcA=-kOjd@Hnd=0s}=F8a)<p(6!Ys4mLLXE>(TxqMZ?tMsm&~b0-K+4f)C#;cIkO&l}OgdvxtS=NdH<;%Pa}V2O)AO;R=*m9*8rGqe{SNe;ZWlH``8Jvo(uZuoNH5}MbfmKSt6M)<Q{+yG<kQN>yt>?tL$shy0vBANC9s+tu^xJRENcsEy%~&Up;az0za7<|cF8=qo!)0Ko1LaEGA&D|ZQ`Fj@wX<B?{{|2A06K4mGiugpFv{qfEa1=+tzEAf!@}XGLFxaS1WDv0dQa)7G=l`$rYL1*N>;_kfW|}IUfRW^Bdg1AuttPjU4uj@D_zKgU6%!au;CLaSN?ijzc#=dli2@xb^en<Z&}XXG1ent@xJW!Eou-XZCGIVHa$!{ejGyUT@(?*rHx^kwse&{BHAD!8h_Dy^_}`3|^0BtF<F0ID7Hv2`0->qL)IkgBHuNQK^b~m#5U0tUn%qj<Z`ihNCtUJY4;jw@r;C`|hu<>*3rf;;hFgTcU4nhjityl|Z&8#@fp|AeG9x;Am;bYe|#MhwL@6FW-2Yr2yIrrVA3t4lQ<X&<DD$*4;?3L>h;ROY;$*Ig(WNQb8+AJ_fGws97Tr8FP~lG!$z?3)^7Zv2tNo1x{R6Gw&USV-_4c9e(dTu8rJ)4)Wb8d&g^qiq`TMD4$tNG*fl)fSbPio_V^ro&U&?3eGlqdkWRpSDE%Z+x9B{oXBDQbQl&AA&^F;X^R_3X_h+|xB&=_pO{-M{dq)z_rtzIPU};>ael<TavD8F+sh8Y&27m8+U<-Iw7Mi7h_U95e&apnDu?TMSC!82J*y|Rvb%k&`ckbtjfKBm(b+iide=&M_GhodSHBdEu-=WFwAa1Q5ScF_LRp+brqhhYRRgVU)m~?fdEmpA$L&wK8*cUi>Jz{9;P|y~uNBaaj9Vi}y}zxo!MHaB<F7~dT>b{lVPmU<c$@!V6$V<YmK1n@PrT)=^TVd==7nvndQiZ_3dY=FyD~a9@>NHXY6P%gJ&e-R5r0j7v^YxTRH-m<^>{2C6{CT4;Pw>w^P09o{R~xI*+_EU%tpA-^3s)NH5~)vz0Or5Q@FIEU2lRstG6KmtySyU*&q|AZJn${%kBF-9c<RO(>BF^266N*BZsOvl{+C->FXVf>5+g`>9lJBoOa-)i{_WV!Y!${J+*$+{57o&$GEp(e_mC&S33_57*yiDbC~tk`z@~dY-hQE)+V!fzuiF%?OREtCK3=Q-Q%ZOX}*o9JL*x9^@xxODAc0yVZHPkjVA}ZjN*XVPK{^CXRA&eRV%58O*(G1M&pN>OtxG0vD7=z3-9}(aGwucNB7zrW^7{IlHHO*FZ}I;);eZ)o%AvK$VBfo|BFLMOiXK=JayN~Cfj#wgd?O^y}VHb);7$iAN0A%8UPI&ZrFF`#dI3EsA6GsE4K>Z3wZB*DV0B#9kDDj?9Jxhf|!BvS3)uQXC#;6_q<F_CVV(JH>>0=8q--}rjlG{HFmxIls2@uL^cZ;ihGEv{|#)rC13eQIK2SG&i6+1fwGp$+q}K>=@zUk5>Q_byW5|_@BIO{8}ejbb(-GsN~ynFv3jhov?H`?z}fZe5AJqJVtLX}(RAx}nuohPTT}q`PqH?m-_f>xrjDwFJq+!mi45WBQRS%3k3jZ0{BjvmicjlrIVmTAw`TbS728ULIQKCVrZ;;j=zZ;2rt8KSn*ZXVc33Ztr|u2=Xx+%8M*EYZg{w{5ge%T9IS)?xYeqD~wh^^)_YwHb0C`CQ(Pq&l@Q#5_Z{XTHMIV<*@7;Ymvxw)p-ZT@FJFGh`yKHt2)4EPN!Ku3Mq367kj?B%51{;rHu6C;?yClwH*SVVu_0_p>+*(E5#bnz&sS*Kv#`lv8vih;Jj%T~1p2fyBMo&r4$-33rXdLRoq=|KQmOB7!>@f(P7VD~w<?F8Q$LIXA;>cSe_rg^jyLL<|z7SaUY)vdJsEs7uvEL{@`y5(Ux#O9BbAPI80C0*+4Q_GXr16Inb9eYX0UWvU8VUDc%)L!2Vx9X?tG1WE>erqSkomLKD=+<UHz$CLa-Ph7<6dK6ybV-(7OzfqZokIOfG<N|Tfc^n+j0;xlx_?7R60MQ&eiLgw@w{MW=;7?Av?JP|4#h$QrWR#&v`fIkK<fnx_74nU2*%@g2fC~q<Fg*Bf9$=_6Bc!TX{a#I`f5t^nICj&!g2awkgGIGyVGMulRd(dRWH{*hiQb5qhw_FE+W9d!k{koPGkwA~MObs$UUqp8tHdn@BaxvDbKTt4z40mw>GE8XmFsw_=!lcX(MY5u1-!Qcw``b9_swgDw3@-lU@8^5`;AlB!ZyDg{ZOiQi+aGEBi<mTieZ@dFKvs?XzRA?VDUrl%rcO7H81A{KwI-wm}Fb+y$7V93d;JIl|0^^Jru83(}clAM)Jx|=smgvEr9Hu!<xYu0hdE?BG^OqFMhy&zV(yTL)&UPx1in{JXdRvBo_m~%SH(8K062Ibfq;}>nSRn?L?wcA!V)*mhpDY8NAMPd_59jqK9o<Quk<t?@^{L;LH%LP5+jL#E^)S37??#(v=llSb>2-nIiei5_J=vcP>09s%7O}K&ec9=CGR@J*abx!tDKL0gPVq2OI4L*O!K5k`~YLf;*T7I_BO7Mx1o+ziT)d<v44rsNGp|ys12QBfY@$oygf!XtoDU@gR26p&M6k)V{>z?c&vD=P%<%$L0i(uG$GuA7s2KRSY_f4QbTL9A!6;Cc5zrL*Y;}HLf>t@4F=<B8%2`sGx^?f;6FM{}9HYBR--c~QP<#Z3~exY!KWv`iYA9!A@bZ(YR-Gi-(w>h!uO#5y1j9A_|e7LqYYQ6aTYuUi?&-qLdr_)#$m+DP{6SNNnQ-HCAji2*l3usn^)KQ4Z`FLwD<VAm{@xT>7x%ZpbP9m~UFAsvf&r9Tr@=h<S{)U~r%8ftNwI5+S06vGq<F0F$OO#(uacF4C=bbyl8`o%Pr&q!z1e`mq8fR#C66zFQk)L;m*%!as&`GvCPWwWQ<xW4&t470omd<0<;>3`|Jg5@W>)NRLg{LhDluJGM`duz{fJt6cmn!%<j?_;M0+R*p6sx66MX-(|x$qz#=gsUaX|>|Hb+e@DtVxI3i6;nEX=c`$RkP-!@@QP!rZrT***H}NV8@J<d!Ko`x!fj$<q1rwx8HG9n{MJcdN~7?T1UE_$D8F%Y*(zuPG*lMXILMqYeMNAjHNscdus(V!iq^q<mtsaPxY`K)<iLv&d~}0!XCIYBuIOPNhGz-3#;*!*8~eyp^C740-$W9{B0%;q9mJy{g6X;_`>s&i&LTfuVd@IIzV4Kzd!Rb>rH+7UeUp0NK7yLc5^=GflqhNtR86nr-X`gUx314B+qI+SExLd&V--@TkI{ngKE)|eMgcr>MNx6<1<zE0tQ1l8#ItO+xp+*Njlfma&O+I$e38r>c^J1@z*{Lg6iFWSUsYKCBV?nm5w^%b0}JhAh4e${?c7Dqf7hy!rQR@I{2OJ*#1q-R=hZ%gJf?nIRp7wZzma0mCN(}$0KZOL2Q;6x*c!WodGWXk`RD%dZ2@|`n-E)!o{xK#^XlNszprxt(bClSNKx{=}%hI-MP!63BN}qtvz-L{GNne9?A{{7T~E~EY1ZLou?r$bf(i*Rb01UmsvyOYt|G`Bs#3t`W2`~+L<)`ZWz?&z<>Yls5(;$eBRfO7WEil#m8=ZGIruz{ImA>)IPt)7nry$DrED3xkQK&RmO?EXsgR7Jw`2X4yyNQKzmmx8X+u4uu8gts^P6xbl;EXM;g(kRLxp#%s|KT^w3L>%P)GkDfQS7f3a1qAW~oM)#Bj;;)V8lvpz!?7=)ZAqbuXqSUc3brv{&E=!_q0^|>@rs8461@9XTd*P;@xwi!%jtGL#DTg2HsJ6x^SRc}DG)of=byi`<tFXK&NX+ov?Tc3GljLbcG)N4+P^o{mc;SsEiMdelZw+sE&MFgQwfz4aH#n|@hh_@PDHFVhVoz3%B%IvY{C?2Gz8)^u7Pe7CNhY+j>PkR04WoqxCND={1<*ceQcXo5<s<B#zU>JdFv29NqfOYjgTs#vL0$W*MA8;$@jM$8zo|E%8SspdVD{G|W-+>|a58Z3pYaYbZdR?f5CU9Bp)*obUrVV{s=I$+>lc!<b<pU!-8^;x-_Ykn=1m^7myyh_9q1pCAT@g7i;`JM2IqDn@7R`#MMe%A33)8#8{(g4kJ1gCiT*L`0V8@dxoewD`sQI`0J(l-d2ItO|VqRzu@Yie4zT&0Im?^ew!@U(b=;?i9s9Lu)Si9;E_nIjs-DgU24AW|}I02)uQC*FXc_n^aw!a6xenYY_{xt{K^LCUEVFQt>@o8er9>UPCR{j3S9(>9^#+QOUx#K>}tSn!I`mM$U*q+qQRHo!F?oin8gWd5}U?QW#HX2uJy{5vuJ;dQ%9C+Kt{E5?z<!3Suj%rlz*4@QhIFoIUsrof}->9`}>$Kq<(;gk3l5FsFs5Ku-)^$rP->2?cmiVV9$?ip)&o&x}RATKW@U1~39$hQzcZ&on@=Y9NjiWe>4&+6lPFm;lh@~UCMb>F`pU~E&aaA9v->=mJ3A!SuRsg#d^(=R4ZA<M<r{4o%)CI<$zp7)jL$#;u-S$9AC06$_KYn*i_VG5XL}s&{L(!e^tMJ~Sd3<cS2&>Hto2hJvM+X1B$3fdwe7L*U1(V<d-{|RA{tVPcL4A3irQl&S*;K9c-9P{NVRSz)WUGdCYbHE_Hbjba`B?7?cdIuQ!j*<;LPx1SV|AD{2_c>q`~JA!;=X11j6{z)ZyL3{*el7W7AY)yo*K22<Bj!H@A(I<KQW^GxvnwT?p<MHmub2cv`jkC-A#g%FJ17mv|QENI}Q3boHkD7-nj0CPrtH0H=jH0GrhZGOM$O038hwj83HSqIFeP??!SdgzcZh9HZwA1^S2U*XYB~>Zds+ys(Ml*N%ivNS_~9!eUIM1M_u^0>x?yN-cZNnLmQsYDO>T*QLS&}7CdTgPSUac_*1z$xP++U(Bs64lsYG+yPGt%sqH$C@#5gVlVvLBhgEyfCq1?<eHTJMzxO`xV<Fj<262(wETMo$v9OQpBV7L@GqX7N#m<`1kM6(Y4tQkzehfza)8eX>wQ{>UqP@Cv=xuGg`Z@Qj<f)unkK*J+%b@=nU6w)TzN@^E9A$PSj@RGFOtzPKcuaL-yG3d!UOgMv2BCuW`D=f^bb39br8L4i66dw!%Z30W>$%naAN1K=<+0rw+_O)<2DM*R?#xI$>6Q5)dIcI=bFI#N^lVQyie*DB<yEwaV$rK^-gT6+INy&9Lix!z?Apdwt+sH`ZZTMYZ^kWa900_n(cF&I^rql(8rVH%yP)X@QyFj;^a8U1FW&5du#92T4v2;gCPw`@eyZ(YK-D{~&;E~sYkZ;(;v$FMEq31^mNp{+AYh5L{PKyaG#A}>jat34x86*%`$B4+y14WP&!c?viq?{eilTthMP+7JC|CdH{CTyu`kx^WdRz+`=^om_h*$xQSCvPx?c#ys;lrxDFrSC21e&_@NU27Sr%pw(d#9(n=>%OYgs(OU^kE{Fi<PNoRTv2;l67i@+O9Ky`dtqj{)+O}8}|$j$cba)0P;Nli0{dVZm2dh1h#-!`_3c0Mb)&!^`XAAC0#X4dUj0*0$8_?R&;%Oa--2)MOT05?4fnPidNAeQ7emDdvvR`cj9I#-H2#8A30!9^IFvAapqn}v7F-NtUVT4vthj5dw8A!t10v=YB=$RFDv9vVo$?))mjSSvFx&^i+|(624rRrxeGg1^W8uV@g@tT^PX3C0BLDh&Iis*?a1t!7E?KsZhq^*4)OQx>>KgkoXgi~dncIFM~@x`Q)k5AFQ@(^k;a;wi?i|9G`p!fgln}-+9p)qKi(GVhU2(jx2m@fvs5+a%(~x)IE1`jtD|~V_>2~>*YSs!)rLw2`~yQ-!7qC{b$mu`ceHFh!e?Oi)`AqI{Af8mz~NdXvHGSZsruzc`xWfZ9)60qp9w<kjm+PR%CjlP@0RKH*e&_Ge&V8>-_YtIMnQNXb>?$=o75L35lJPyd?)HYG5p#aDxHQtyY~*h@-Bw0Oy{KyrMttYH=1YfahtV#c?LYnjO2I5^&Fp$bMZ%bHaHEywY@~zKE5l`)%a?m+SC=^6-NQ^_1nJ8>?KqEpxw#xxJ(yQn*t=s-*Sh?R#yj5lYh^nRp)y=PN_GM;1Zhsl`o-DtGNA+2c|@R9qM?Moci1p(`eV3O$X^MwZSSc-}S~6AD!ACa{+(5AmW1Ln`nxq#ErF$Mo~ZJP;g|WnI&uczV0;YdgRYd=fUxl6`ei^<I-FMv+LiISGp_t-$fO(fck}@X^{z^1eNGld#p!CxlA0tw;W1!J^T*fVW++r1M*LL;5sE*q}-b9sM`0oAEqN4>-|w?^lpWt)Sh(Gc~%G~bAKvE7WqQ(?&<TAq{hSRZeOiSRd+tk%N-cu9N>h>?ERUdf5`dW+T9+Sy#9W<JKIjyf5CmYv*Y2K6|l4mwfq%18@c-D@E5V4Ya&l7$56PS?LD;%p|?L9;t6fzx7Bfl4{Gg0u<B2C!?@mLka{xR=lA~Bn>Sq0Y_0a-r3r(ZS66fx>D=rf22Q?4Yi3KlZts_|P5x+Vw~thdnismitzVHdjNEj}tf@2k&<X$MqMUh?e%dy&cT?PwoN)t%wYzZ$*V_rhZSx33_$0Wa3)gftE$;0E&%6k<E*SW=*4p_CU9NMUbUtmg{Q3?0+hvAoe}(tV54}bI<r=GDduP<@Adv^BUEO(&z|HTuY=y!4?nKOJh>T(P(}PZC+Y=#;R1f-T#Vq=JYj<0c;Ix=uUB&I^k~tbi*WxmHCSExCNh^acc6w^V&>_UP@5aK1ujvUsp)EE6U<O6y&)~FenK0?i@}OF|)hA+|8lu^U>(8K)W$w$3Kkn7o1H4cM;{j6$=&}rucMF0Aderb8tcnhTH_I9}=sx$gq(i?h?2}idzgn4ND-fqnygHpKUX7ZyA*MgBTzu~>tf0?$7&l(=YMXbVM8h(Qytw6JvgthG1pb5@5iH@V?V!PqmpCm7I5eogw(QmX2Hm;!S%F`F@jl5m`@O?Y4Rkg5^tkksX}Ma0*Qk$|5RKi~D*0O$TJP`ux`k069l8rgy~fY5-$Q@Or*&s8Sz3|MRJBjKhOi_1jl9`aFz?W&ajHQ*S~<YI)5e|Xr2><@;3k(EJvJ6_R%_vIN4Rn8eiCpASO%BmTBm+r;W%2Ja0p??)N9Zr(C}u+MsGYY`_WYTA=jq=o|%n~t1I<zpWF&5O~bzt1K`d>HYPAh-y)cbYU2h6+u4BOMCKQ1PygVigR*D@=sjKkRzr4M<DH;c>7%pBSr4Aj`p~f&<|C@$-A*UsFUfi|4Md_*?Yu_)VcGnJb-&WQ9E!1aiDucXXtN5sbK6Y;84$C}{I_qfOs_#b4OQ50upM?Ra4-{}bL=tSEV{G#;?i!4GxHucxj(JHCN-@Ywwj%grj3b;%x*rrz8_!e1es<Qa;=EXp^j{a_=UNqQNHe$LqlJlPV#p#tx4T+7)@Pt@Ml1Vb!))%Y>TY==Du;Bor!l5pvQh20u~GQvctjH6>!&G@cMU>!Ap|)q{tY7*0*1uiMAziJ_d}DC9GEZZJ;uO)8#^`1_{!Z=_SL-+iD7%AN<}a7^AzX2ani){muX6K%bpI{YHm##f8Kn;#oPe@TX3A-I^#1Zc>OH__f@7?LqLM+FNXL7<K0k*?-g@*z1N*y|Z!nT+Dq)ocn3HJ6C)TD^2P{v$yG|BGUk4E?a@eKxfl}?s;7vo8U{+9p_qN=?I2?hxGX(eS&-N+9gwzm-pPiHK)fn&STv<-FQlgyUPn(h(B`NUPJ^UL~gS`CJsZltaA27s{1PdI7Vmo{jK3+0f3w*NXw@=tO|>nJJg56B>522XT;5Zv(Zgz<+IJHi~HU6SLb)u<yj}(Hc40Ue!0i?zkPa`7d|^G-7!iAO>n|kUkt`o#=EW`_v4ed_v$3CPV_EcX#-8~byLQ|ZHDjnyIX~LW-kmP>FezN8n&NGrU(segOZDuS|QRGm6tPvHN-5F0ll`NI?5+=_B8xJ8q-;(S)6@&u2voTn;jiv2uUe8_XpXQ6PE2hM{n7`A^-a18sa{P(xqwJHbRbD_NTSGex3<6<Tb`tZHu54wmu(_QMK3bkR&U_S-xh(Z!;U&>9Uw*135b_sM*Zq;7&IpO=+a8VWy%U-h4Cb^G&of^LDeYx)^y*w`AS$Iut&aAUlk`>~A@lS!y;g?|xxFBMh#&p#tY`@<Q$t=(`he;j(EJIE*UnR2~hgT%s472bcdE2z<XT=6d?cUYquJ!~xXcPwq{qwr-5aaC;pk_`Im3R2Ssw6Tj>{T8EYkN3WU{|2niz`pi$f&+Cm!QRiuAm6h3bH7e8!{RgXVxTxNg(N-VmVqD$fmG6+PmXk@8G-lU^5-&>v`l0weLo>TRBj(eUxUxc%GcdW{g5p|d`kRXHo(DI~_11Db($%aw>UPhn+`v8aRO8&P(|!lkwB-`pRlLdWM&I`87FvnacOxB_p|<QuU@s0OwQYh3-u~Qo*Js#viW>~gPYv<_{ACL84r@;57Xdv>7|GSstvthSZzYArpo2%HJxhmo@MZJ<C_yN}Zg0iuJRG&ug_!)@N7Fl`!TEOFjy`Rx1;VoM>j|R;FD1TX%%E`wJPYRZw)&^<VT-|=H9<L^F&_*&cLAa5qo%(&zgISJ71F9dcF+aHtPkZ-XcRrlp3_EIzc#;O`5GRb_sGif<1372`Jn-Xl!`RmsCye{XHZ8@uUfoWjN7<(K0+E=X=VKb`nR9VMFC5pR6RY;ujcz%)m|Jv-E^~?m(TPOiRYLE-y8V~EZ<32&$Gsypq@mP)az{HaThQYW~Ih6&2OH68{x+AVou(wA&*+04tau%02c>xeD1KeZYjb&ZqAvnB!20;ErJJqoJi|J!XkWddw-l2^|}r!-QMUueS`i?%RBR9KJ1QDPG-dy6)ryCh90$M8rQdj+H}i0$ZS6wc>UmTG3OAkUP`@L1;zikI<K8&!EFm)3qg$~CMuv9X@Uq=6e+@i2vP(Kpi(W@zP>-}VD0P+fDo8-jPV>WUg=P|&!*WVV;|cIMW}AnN>}cnT)}U9yAy`AyP@?6Xddp9b-rEe)V_p&S<$oQZbpdvdKatY+lg%XJ+WT@jP8fSklc5ea=|^N#qgDO!jCAqwNGflX^~_7Uj2t!_@pEFG40$JeQ$5dZCiRTNBAhA54#CYnWO$YGLo=3FB72H#A#aj+nS&?D`4K$1bsONLKKkr&rD)B&3W9htt{d|@A?gZp8ROdqs30BEKgO>iqxYzu^2miQ-bKPd$bcsoz_eNn}5;G+f=ad*ZwkVmaBL{-?(bgfx!4WN@?X<BK+vB^BWwm#Qb?E7Y(=-r;4m&8sq_ZyUL&Ds7ST!`oW^CC$_99#KxJU>S)1&Du$khGBQEaiHDwCeNpSb{sat^-=~xaR|^r(TGPiwbf(DhrOU3T%^Yjs#@`A`|K>V=<9D^_wbKadAtlA!`mE+JxBu=Ut>U{$pZDM|<B2!)r0-5qcJlOl%pR<ua<nN5Y1zp{1J?nc-6%lWa~-zCs-Z|r>jp9(#`3sJVOvkOmg@SU2_awRpZ>uBN2H46jq^Bqw=n^2^l<D}x{@~Kb}Tn=xy>iJ6$G^VOR5{KvmS16@BP*wv|Tb}6EgR^&F}2uB^K5y3Pb&3HWJ{kYOXrZx5#YNLqlYia1{Dw3tmp_P*B^S3c1qJ)A2AKv){1&JuLHDku9IMY#BGwTeQOee2qQ{7I?)K^H7OE!38Y4@@Qql#j0a|h&DK2MIXhNgH<Dn+1p?rw=2?cF(*oY-z%y`lK77isr9PxUj}F1dbbkF_oCVMLtJ)2{}Mcs>{sJkKpOmIJvX<;HdEG>FCOaHODZsMSg5NP{{+OeY^~Xy95NW#sO{WZpB88ZcF5Ick8kR{`<bhMNP{UIW`o%E??T$g>|)J@BrXAq-LE}ZM*6y{`D5y}0QR>J{FRW`!{|TruEN#3-jkTx66i`Npd6JSQ5;2pu5M)Q-JthQr-oL(P3%3xg1bwO^Y0q%jGmkfa;!T$U(Yy24+mD^NrBk0+98h3$~ytwH%rmPXDgmIBfg59_J&7NJ8f@t>sQwhR9zhRXWLQ9%$G&_^+)5X6Mc}de0}Ckv32<2zaO(s(g0#wpIVw^b}*rnO5P!d9}M{m4UgbL*6w%3gy-qGlHHz_t2LNjp<^KHhxq4zWqPKR4NWkomJDH4m_0Cje>^&Ah4WGiFLD<%p@H<*o%X+jYUd(*$m4fq-JKq83S*gnD9T%_zHiYw7>*(vYSq65Vtlp$Bi{9W-CsT7{$lvq{S@lE&F=YiRR!(Fhl{=qr2TvrrK^Ws;$i<NK%yeRJ-0g5>qvjQpV@q0pYq$uI2e-wlmX8HcvbsCwAt;5GWSn6L>Gof^<uZr53^A@BXM7`;{{urGrYThD24=T*tl3?19DA2w#V`@*X0&!k57fh-kaU|s4eri{mn}oyUj8WjzrOW98Ub(6D;OL-J^PY#_!$mWi)XzT_1aPBgLTVT0^ncxc`JB&&_Y_BCU^Gp&KGle$!+32HexV0G@@nx`b-?+&4{mH)s!b)H(c)-Dy1i3i4%q!ayim8TV4_4eE%rU&;!~p~>!V5uWL#(P>WZSL!=@&-_tks$Iav-@aULg)`tKcXd~V_P_(o{vj@gX=uvp#_erFy&`Y!(O>QUYD|~wqWnd0OjNWHZFYecv)sQMjX%+aaA)sW_N_B+5XtYl-P_i^f2hU7{+mkAo2~oA-H$^HwaMJ*4R(7iowVc1nq?_=t5ypO_16vWgy+TVD9XLX=~T17#(U@m%>gxdQ5)KrZV>Y~@639~BmAfx=yo_(YSN*$Ra5t2!Qsi=H@Zws?Q?T&<EoFPjps}<ZrqPFroj*TJx#LSN9hvN2f}SJJkpT(4`JP{*9$Pd`F;?GvKoQI=aqK7M#ZJs-T4@GT!~)1X@SSP7>mI(Vuu2IC~I1~yaOEN_KsH<%{DU8n0ul0*n4BvFqwrt6lVVKz3+IV4QgcpzitkCdvoN94v(e;&x40B4quN)Xe6~IWL3ji<O%q@ysFYSM*GH*_ucNa?ZE3d-_;q)<mZQfKacCcI|xtPraW>IqyM-MpxGw}i*1ULg#m`ezGpDC-3A|er`{>HmV~}pQne}K5dhLuz{oI@)@$;vi+|$;ajLV-xuH|dZ$hgpR@Ym7w)m3T^x6K_$FVxF=2=VQ*AS8Ri?^Vd_@-y=l=TCr7BrPkCfrVm=;rk=)V|-{?Z|&8?aOSnm@%~_kW9=z0JD4m2hsDI=FPG3o%eL-a(M|0t)1MHRyOq-_d}RYS|~-j&sD_`QCV&-c4OWS2P(0At!5&2|8d}?19gB~x3;Q3;p?Kwm|Bo9Ir@5w!-P}xXaT$4Z5U+ekdg6X`J8>~krO_ON+xen>oJmQT7Ta7k$vA^mONkZ^tlhWuB+Gm(ioc-^#b>wkDGWT?+hyH1?y6?!JwD)?)~)MLDBtF=@l2LAJoRyNJpwk-BYPBAFp9-2v6a=q}Ws5IN>|`BluiH2l(Jd8QwH2;YlX<lGaR@US5BO6WDXXOUtPJ>2I6MDtXw1jKwq~^$~P*G~es;<kJV9vt2@(=NZne5&P{b+_w7a;;q_w0B;-i%4Na{h5qW?=|?tA@16zwCHY?M7ZvjPwGaADterMfWb`r@AOOFvondJAfz~MD+KMgxUdD0Gs7{ffRPg-RXuRpka5-I>(In{ooOqZ0bPQZI-7T0FbUsIK3QWCh9CdArzlIGZsy;=^NL^#tD6^f^YA_Ef3U5PVP@S&6LdX$>PKj8L8*vSTY0O?!85!4Qr9TIvuXh{_-!FYk(edv%jo=X+?1Sv*u!|}QlJsz08@6)u+8)1W7sS2v1tO_`v`7o?S<V258nTk{#-ABID_=x^kt?fG9du6=i|y{)61%sI%T#{=@1c+2x39@z&<uhXLg=ZljW@0-S!B^Nve@QzYCNm2A85XubRJ)6vK+QI*AbH~{l4Y({_uv$q<bK?Rs$UGL5Bz6?6^Udjt0lbf7~p^N<O!?dw>s9p=Fk3Uel2})&dCu;*v$ow8P~6daKWya=woldwG1($)9iCI4E}p_~?=E%7NMJ@k{SJ>NIK;1((kV_Bu~(|2p#dK5Ej(wIvVFzc3xFIWAS|%Pa|JF+26oGIcF)x%=p?L1D%3@-E{s&{%nrtzHfDIkIa=oHE`l!%$?`O94_UEq|PZ8#g0@$~?f{4rbcpCyfctK8EF?2_C0z?ZO}N^XaDc%l&goSx}9@;Yu;A93Ac~sir(-2d`hXVM2fE`8ssYunt;7((R_`dJ6u%Dh@nUextHQPj0KfIPR;9beu$w<BhI06d`F0?4`je?P|i3?j~z$*D%Yhu2K<kd!1*TJ}RhHOE-qSPtnS@V`UaU+B)7k92Qh)pxxrxHqX<Yjy#KeKzsmo>>eQ*e&42|ing0#_}!bYJ0{mEVc_i2{L|j78gMW)r5y5e_{@%)!LJfm0k0x+Zt0yz=hk-7RlDow?owq|?ahygp|THUFx$RT+lb@l7?L!Dc_~C?6UEZV_MLHIZS5{-wZZD%g1GabuU9;y;KSZ50A`gLGHmz9i_7N3t*K*HXM3aljZu|F)k`18sOG%svN1{zci{L|zxgceCZ_0WTVs}k`eA=;Wci=cuHmMZ$kY5AR^&=RiH&)74Qek~*rK1-bY-Y$HJmOEQ!Rp9R-35E%&dif?{dY?S}(OWeA%`(*&fzwfN?B-$=7{g+sK`^%v|B~X#6G{OwEgz(zJ~vR(=M8!|bxm9IS3Tb@verA#LYW0E&rDFulgKgsD!*x(>d7>HNSa)ZSqlS`%$ntH)?*Uz4t{0=^mUcS|##q0pw*9M6j8EIfpCl|=!cFP+TtOzU1wrlq9&{G3J|Mpfta?UNduYR;&Vy^h_>dSj~%sgCOkdZ`h_3PLN{d^qV(Kx`KA>v#5nUTv+vl~Hf@y!jZB6Qdcg2G1QuH|LA=0z@}pAW27Z^t%i!Pa(C&y?&hr)7*H6)9;lU*<(I8Pug|x23wOQdr$IC{6QcbaG1w3k``LTU4vk~e4ZcBt2z>Srn@54>g%|^w9T8)QA>_rBHG*m+S1;RMvHCr_(SWB0lYp%muGx8xc!TZdbHF7Nh6qfEFF;evsK3@2(|5`g1O_TGw%3lewar!@)`@bLPLu;sfY)=wb>Be(@_<b_WJTZ(duYG<QE@|$HN}%v->wj3jsbfdyG6;4INGZuEG4s#KbqZ;aa@^&UefC{I%${TU84q^nO1-7c&VpW8YdjN{nrn&*vdaV>Mb4tJ9^WKJJe220E*@Vbyo{wf-bILJ&%WIoIKM@<6NPM4DfugL!vRrLS@Mc5^0A_*9*~1*aSD!mG!%Pj~O}QTVm~rmo%ixTHTaEiKO!9#5!s4O_U04C3Wj3x-#N_bS2gQuG%Kt4k|k@N6|6ZTMc<U&o!iDA~z-2%$H)t5*}|WIP>P-M7bTx~?TsMC52M_L2+pq>MdaGUZuz`_^sfJ!YFtxdJli+g`vu5yrOqzfd?x18`p-KVrbN4vw8+&@uzp+vDOB<xoRFf4S}$W4tO!6QV+>b$o9q&$sUl?%7BK{Ca!UR-_BH&%o|=e~kuubBCt@Ef;2fN2ANaG4(~rJd3vI4j-cjh_FwRAMe=dEZyNwuT4C5N4mA;_1)zqGY)yjS$Ybnz$-!g-^LhohBRrmVt?T3cBiYV_l}6c`Ev}F<JNM?R(sOJajZ^yHPuo;t2wu4MB0#TOYHN`exuMqayCVDxLu&BH_$FD(S<AXr*?R#*Sxzs<h1)Be`Pjo;Po04wqAdz&<RXY>hB_NXA;}ZcF4Sn>tYwY9CMS`qg`v%eq&y+9%sax=OODAx@xV9^Kz0ki@z_0Ji8qpfJjgP&ZO0$y7SGMc*gVr=nBM5)WVm;P;MfE+Ba+Xr=r#yQx^`8w(n+<<UPECqb+@hJX7WH#$I*tSbX+N#<iw3)UvLr@-bHD0e+<S0bO5A7<YR$2eo=YYU(@OaTA^*wLSVAFM9)Ta|)lfvt!}={$&{uHJr}t_AI@hdV=%4b#$az`VCM1lTyQj31qncma8dl-#>rL!~Vt6@bm%lZhp0-ZHVEP?-id<>h73~>`gn%4W&kR>1OQr$|1SQ9$#}G0`gvsR{O)Vo&ODKz#Qa*gRho}G4RpHt-&CT2>$f$jj%rJuK3=ey&s9~Rm*a3o%_fcH}xKjiyX?PVpm5hI6f?qNwwREx9)o}h&B+VukZNDZi`K9B8|-2oIrFxTMn`5=FK6*{qtrQT3vf6oOu#*_tHH>VuBvu{BnM7{Dl=;+t1Q+f!zYkz@vuW;okfEpmT__=kY_0-avbHRhZ%O{{9;>zHBpBuPb9Cw#+}wHmfutoSuK```-N5(N<_By$i25iq<HSh{mUwM+<Y8UND(JFSz)K#uJn3f$sub@2=fH3@uPUK<j(J5@NHazWU07?Cv!}QL%D%OYZj((=gMVy5u_i`L@<d_eR2=J<Z1QCG!t~<OU_f>yJKD+s9m`xpW3omNjFiH>^C5c!hEXXKdv)ZMUDpbFBy61UPi&%tW2+Iy<1t;%JmdcFi(hhqwOr=Lz|K@R`w|V4CQS13u+?vv2{BJ)CL18rNlC|M_t<gcbxka%3GR#5#3ypN``iHKTO}>My$1=Y`q6`SikF09l)6dnK$xruwVS@>9RP$rGY|>y?Aj69;SLWuw<8I9Y2|j{$Z^X!f8Ry9a1B9xzTkwEFIgrk4A;13a%xvgV_&$^qTU^G*I+IKWFcys~)=k9=5Kk-)>HGufnT2;}0BPE&*!spL8sIgk20UNBU8z#9ksp07M{ShEpaG9_y+)P9YOa;-~Ry8(p7e*}rK@WT^4RI$FXzgELV6|cQ!#i~c=J4&zG=O<lOMg^_&%|W|`-()m#I>T~Ok!(TskHl(DTG7Zfy=cTPkM;47gjcS-&wTh<Ob8KTHfv{I9i7`#1smeiQ7a;~4zreSmF`E~$fl1_XBVm|m9mG*ROP`EaQyD|VdTOmm~wI&bI40gUTkG-r+KDp5AO<-XJA>|AmZh-nj`j2>mF1<8hryikFf5N3pd3vhALQDoLyWQ=K#zk5DWGMZ*V<-Pg0x(#Er})EFLZNnA=c8d-~{)!M~+g8LNzl^M-h<3xoJa(tRsc8d7_7xo^XwPK7*XPddoXR^pvH!WL;`(cUD(*ML*1>vR26)pt|G=m3*4zHr+=OTIddVY^t`aAQzknIggvrx&~><}BReUgB)=hbPpdJl8wBAzFPtceuKExm3Esh02rQj74EMvltzI<YiLJQQZVEr@f`sPde*P>rV*iXM5V_<^ZuZ%`Uo9?qvt~siU8?h_SweL-+jE_3jrh(x(^%+G+dm+t`7E!M`{UkI0#jOV{F)i8mRvV5};ZiHM&B-o)esB|P`j?$5W+4|T)*O4<{GUjwt6P`rK5HdS(AAht7Uyq3Vg)>hFwy^7Z{_3O_fu8%ZeG?@BsC5V7cA1NSo@!dv?a#w?cM_NE;uZsR?MV8+!rB$eCV?!o~;f99Y{&hCM4Q2X6T0FDt>M~`vZM`0*8;m_3zUzyw%|Ep1c2P=#7o$%1J>Xz6wiIx!cel_$iuNncpWL~w$oq$wMR=_kWLHsE54UGdwfT7U{tIdPRR6_163~#_VdBO90tL(++wGS?>&69hEi}5y<GYW%`Qa))j3qcr1V?RMtn{uK+4ZQ`deTv2N5A1q**UA<Q+cxID-nC(xn^zsmYg{CP_9<F^MXaK?UsX(?d+)s#!8efh(I=u-*7d1ZDbMe9@8vqw-1Zfy0KlZH&{5f73naQJ5xJkG|!)(xqW_G)SNu(Veig6(gfpWjob08&HX}p)L0yr_N3`#S%&HLcvibO(6(KhAhvM&c#cA}KFrMg7PYmVPTu2wNrA?`Ho2tdKm>qnqnsV@gK>GF&thF|K=HgwdBXHefi11OubPzu6U`nSR2m49fB&>jm#W)7d20i?EjIOjivKvi6hQFWTu734T#n}b1{Vy;aB#Bw<Q~~W9^30ys$8}5dN9Y)EZjx&rs}X_!wY89Ljty|Ey6bD-0D+GdrN&E8I>AYx>}$2y;t?uX@$!MV$CD$B6sy}ZTvoQ<_K|te+@0wVe?vakXt+;dCkBAeg$0e*VZWf*ChMSkzU#%mhj)x?;o}CZ~VLkfxFB{6Z!&IEBZaXON$7G+(uJbR{6=Hqi2X!hLd1ku&NqvW9Wvvt4sBENZEIM8cu8b`nNXeL7U>o;TtK?$mHsGDQ@Rd_JjX@)*A~Kr2g5|e)GSfOE6)(=?~XT9B|Nh#8uxep%mGT787K9er3HPmx<Y#Z`1Q~n6<D6JqXIH-?WZWwQ;}diU7pw?x~+raO`(lw1ju(I~gXbEmgZXOK^2spl5d=T`wCyYd7B2C;2mSizVyxUaeM=*Sy*vy`!r9;>G%}_S4kCzKKcsjq+#pgV^M0S=elxX~)psyAIeIT)_%9*(~Iynl&E3hWv0Z<3@_`*Fz{x%PPJ*R=j^FFu&@VgviVI8-v246yC;6`^)$$_td&QV07^3{PUA`F0z|tbX&k05H}vK5h0ESBRoRqMt$xJys$X>4e0f>nmhX(*9BjHKD;@~4aS$@M!yvL->xRQcU^B696|hHc=|mJu5$N>zIfl%Gx!$1Y6?*O&86(-qcyzR!;Y(gt5z@v*U9TXa?@GpYv|fz!*Jt3MH_|D)AFhZ=v6=1$zW=_bakH(q_WD41?RQe?^?u~9dye_-S1zAHxK^Cn%IkW!n6IAn)!A(fC|Kg?cSGhs-1ohngiCeIX@ZtZ2~0(ZGicu0~}i<R2$c>L3#WV5n;?a@Sypg*pc(-w_c&@waV}^wwYud=BHV0Sr@VizL;(r6)7?w^g|_P{NGjU^p>^F{I}oqYV?*HpvSCcu;h*_@*m)+a<4mn2W6PlZjZM$kjI-$goQ_{ZC8(3QuP<XCQ#oasDGNnXVhdnLA}@iumJb8asz@p!a}@YtL^V=e%iDObsy)UZFHsTZadAY=A-d*_ATb-YU*0-_t4uivWKe#L##<yoptLAq1Ss}H@5__$FIYj$y?J(4^h6?=X*5%EbGSLgZ7bI_@0fnHS2u&Qq7Iqw6)8`jJB|-uM!_@?YH8xqHG3WNyhKd^wRRcN9)$q<#VwJN{*!C^|wZNWmHuT6431*sUZ`AfVPtI`qL_Jg?HMHPVtG`O%GBOH;&g&tj^w5e~lckclVnvOkFjyfN%Wh{_susar!<62sh~KE=#hV4P<1WI(J}Bs~kYi>Fs>hJG0q-cueczR(h4|ck#M(>FI+b-%7k%<|}bLL#R4EZkVm6(vXSRRuNyT;>9tvu5&Hh#`>yh-`Oh7Pu}Tl8M|aejI2XiG3Ud0Jb9=iQbjB067JE7A_06jFpc&}C@W+q9@Z2}qvMqj$*bg4J-Ix{t+PT2YMahX)wq9^qJ2E!^t+VubGvEHge~%G6t<cI!J$_C(d~XZ)vCg1MQ(25D5?xf2wr2^8}o;s@H!v93hvP8-e~Gi$_=H0+s-l??l<n};pHIzd4z2qa?Ir2{_<y@gon!LHc_tT{rnQZ>Z2}T;70({ABU~y>nC&{!s&}U>r4htKjhRQPRm~TS;cL1RtF|sq^Jd}nN5$yA9(%nn&xCR|1h~#ugfOtr~R~3BQ0BjGg5uBZ~fr2EqCJF+JE^MU*56fvc@*A^D258_4PHo>os@C7iW)ZntS-{wiCZ&RjwxkxEww3X=1f^y7$dz@o>E#cKf|00AqRM746%ojA*q@PPjD64s@c1RT{*}7N)c6_<Wmozd7y!YA#h)^!L=Lk3N2r=$^yeE;6pp%1GBx!O5sE|MLCfNd5$ZIDJabnNh?UcvX>AeK4=i3bBBegxrM2{PQ~Ry%2noz)D%}!5Uc~Ce4jbk`k{<%E=T*{qA+fUBUMwgqe*4P-2si+xy+IT@}T3>p_pNw~`>rY1h#xZ+EM#QyV!hvsg~!w=RTY$P02+7x_DT=+Nm~_J0_EnykWDCcgFk>eV#mpVc=-xVhYl&E&F3l27f)&FaJla<8B|klELb7^V^`L>gbf)4s?k*XB$2qSx_|R60UdHi8iF2Ez5Y&iB{VOCMIY4M{$*Zk1y_N?g2v<`4IOY1n4;#QsG6OhBa<Lb)@B)=5s!po<yHaSLnq*O=cN=#Auxb5FoNCuB2jJy!0-hK61lr^4$15r-2vxzdc^n#P0g1HRnrov8o5RIu;}1?R%sOpSMXWMHR*!PU4TM+M7o4A6&P8n6`RwZ$=4M=3}y%=A}y`NE*C#+}-}0kCl&_jK*nM|{8Wy3{;oJ{e)5EN;^uxN^q0xknt*lH*$+<%+eXQQWXVCgN2_SdGCB5wup`(AjSAw^7EnMfaNkKq_pvi`H(phA2Or!^gJ2bF6cOf`_5_7R8PCQNhJP67^$sAnEBPFfLcDA#T>|hsG^pvvOLVS7r4G+V{Jx1IqpJCNMEuSnDub0@iEjum2o;QbUpznr?ghA=K+1ru(8_S6oApemJ#bgb%uZe0>U40Nlfa%Xwt1Jcv}l`D^{iX6HFrYf|ZLGJt%TzuDS3>K;06P{p?A*&L=qc5dyuGHgq>`**vI3Z9?CTC-?(YfCU*I7oe8b*S-UF6_Nl{dyD0_VTPH)kzX;5k#Ff#t=~4E1cN{(}}K+EL_s$Fc4=${bF?yvVJQ7O^Lsc28hPnGvy1Qhos(%oeqv#6+A1#_i}j2@}n^@lvQ<Pdy?j(2NUEZv}`BC7--RcYk_f!Q_P0b!@QQvzt`XbTA$ZJx}dUsulBAREU<#}sYM!5${=B?POQxWey+bP9U_mt)IGnV$duM)OA%4%2vV(;4~T6ROd8$y?9!HdP*z1o$Z+59*--!U9xhs~+BrW1Xsmr7x_1x*Sg9_r`wEt{td7na&f{dRr`zXL>Sle=vmyCW6*@yjY+a5`@Xp*0i30D$S$kn0+8brHp75b&ER3L`1?0iQoA+{e>vIhgT~I4ByeMF+RUUWf9n$M_m;_UKHz+CF<~r?u)vuyif5r@rKd;~>%IBC25sAbW9~91;c4g^?Oya$AB7ZN^tr#v84&3x0E!!9TQitZ_huK2VjCyV74mGG;)8%u}$G)UImlnc$L{6vkivk_ot$NvYVB;@huZ^QC{3Rz$kPn^Ida`oQBH_C&ztNl;nDs~5MLN-=E*fWj(d+&pAX9x0E_y3A&2(h%X89A?#0UHrRrjg+Nj@InX$|IrQc8iUw~bx)MuVvXn7dRUrKr9<!Y|77<EpfqB&>p`{=WN#TUYS<=tYOaZRZ(BZk(nqumG;cwN!CrsNOYab{;=l_pI%F!EpSMMaa3}({b!xAiUTDr+oBOKdw+NswF)e6Q|*PBPlk*zU!N%JG*j|*&`~HZ%yOW+W?^>n_aRP7{XrIf)jI5W9NN)^D>6_NOr5+cQoqz&skRBVI!4bSw7&ptEl>dNon@`6dQ=DM?-z`rm(0u^$#t~TP-(_{<b`7$OWlHj}!5=LunKp8x8pM+`k%!-*$y$+alcL$8#e;PPL-=${pNMPt~2tozNpUm|VWyHP;%n|42A^&mt>+H#xampL8Id8rDOTsygqA4tUU(6A{&)9O=WxGd@{aj}!EQ@^gI13uY}Hvg(8K%Kc`&%K2FWuU^(^ud*$;-oi&evHu=ky7}Uok<ww&(P!f>Iv8F;^KdzRF0Iv`3)>+&<XPRO%|W=|9Dv{af~PP-#g^r@8q$LhZurMykWBmbZM1h1e#n|?&g-R^M_`YvfxFxqPWDuV2tZ=3#wVeB;}@yet?Al(|5Tsz35TtI`9t@&l;72E@NBwD^<(8g<MTUpIFE%ZIT4lJ;7iOiJdKnm3BE(oGj@eQun#QCv`&V0Y)QajR2jd|DydCF-xy_+?;dRIR(8|>_IiBF3*=su?_}l0Eg)yQjo_65AT#S&yOPgeg}=Kvy(rc*16lSSw6W>7rz>g3-u9IZ*#Glsch&I!3@x*bOlx5zD~6S6Zxm%|_C#CJX0w#j^F0KM;a#;i-(9r6<$J|uBd9KJli;E*br+&cdvRI<zH#-+TNysEhfYrGhaESto{7Qc?eW-pY`aolK07;ev-y6BTROlEui<$~`~^vdI<}COusuJq;m2;kB}IBor63P>pepf>ov^K{c_sG@?Y7-5)_;vm<}^$e(i|p0?I9?u-OD05cjp&pAPHNqgeFJ6KI&%V4&-Tw;U=7B+*?>Nh}0!R(840HXtG%B!Bs6cav)pH%2NY2ex-Uc^;Ygx)f4{UJZkZ)-{9tg{~;EkMU}vXD|m`qK43f1H(!O155Yokuo>59Q*lJ{)Q9J>``{HpqrF%ec(u$UZ^b!z0JCMs1Sf_@kzn&BJv(MPS(WOivLx-#*qnO3>)WeO&ed)w-7Vs5L(UGq>@*=Oj+zU1wjOqlwO4yS)vZ+qQR5CDr4VC4hN}O1qjsSVX|*=(rT1G73hnXJp^QItHEKB_58hX3A9O=BtU=vU^>ez?kD2~(sFnUioOOHi#{1CSf6r}@z{*SW?i?+3P>eRtgP)XtT0f_&K{{&=<UFpR8<d=73j`u$B1<a5aw*lX3+4-XmnHfzKsWFcpSj<?y*`nxZ@)i%A1YA$22`|qB9n^HyNt|Rs7M9U-m&@19KRuKX+zq1`%K|l<*h)W3*N38WU5B*`3uA{wt6q#d%JWvp?}@i+j?MZkp0oLvI=&KAM!4@6Y(HzhwAZ@9lxXF`MkdPU`Y0PF&qB+15ExY+Z!`qbcQ_RUA%VjEyGFc`=^B8&-_w}huK{edfwHTh^La-EAIML43H;{5m(l=2bVFEz-#Yv3^B-DiUKo!FqPr?a~oSY-k&zF4+i(f46q;6rXIcjdew0oOSI0>evHwb!}>|gt)~zl)KG-G^x5^;-pZD<G3xS3O{TTrVMc?C-IDN8rL)AmH#NlLJ(^_E8Q1n1IBRVU`8CCM$K`-$`ri5(&m4BJzb?xGSk$$Oi=5xj0KG-TEbS)uWA>G|A&B~3?fPd^`o^d2AzleHH>$n+21=i*rLvAQXD{kp+Cnx`c<lO(>V4kdteW-y6n#$~wPbrzy`d;7PL;3@^FQw@olYY57Tk^LA5FJ!a9rjdh(9`iZ&=?{#z8+DAJ*cLVpD!Oc6);}TkExiB)?JXN+;}n3zywdWraaMD*!ZUNlR;5)LUJ4fppdF76p_qInpQW;@UfUv-mOo?&kZi(1e3Gr%+718jO%T#c<>9+qAlYRuO;|2UN4`7s***o<`eW<9R&qRkc}94&H4stOhDooO(F;aQfD?Ld=Tq(cL%uMb;dE`N0}WcUmW;8u)nwKdHN!4`6D$zJ4d0<50gJaL27K$Fsf0-jMqn(J^;By}KjX*CXb6|2<Rr`Yu`<5-jQIG*h;xE-_hlo`=Wi&#+W=I_&J6C}G9+X`d-4K8ic=xf<)qbG7*n>oSz#-NABo4;lWMPo}T>;-fv}R}MTGOC67!t5&6iJ1HZETBazY{FgmX?k8~0H~lgh?9lCsl=~~;K>H0&#A=u5aWEIBHdCGS&-CMerHo6<ny+r^kdPg{`I<Y`qV{W|@%?hP3`QozCH;Bfh^kSKE$&I?g7j`#il;&rt41a=>>n%Z<}<&|q#ii$kO<Zr0}(h<QiIziZCYP)x#mUX@9RMI7dHB`pk^3hs4?4FGg@2nH6aZco+;LBa67Z?b@nW}M}G_Xb0)fvs^cbmxtV=u>k&y)vZIQ~AS%l8i+CH{z=N@9SBH%Nte;7@43a-$qJN{od_yHy$rn((RU@Qi`T1O5fg)g)bs$gpUf7D-wVYB->QJl>s{<3zHhQS7xNCiSu(5}2VTaH#RJyk(_U~EuP)_|V`@*3<i97E^2DveJC-&#Xal?06StmM2HLzuf%Pj6ps{8wF^+^%oYo!aalK$>BrkRcNUR`P*{l}FM*K(_}J3QJPjG{i9{X%oNua1>DZgj=vf@!Y|=J^}7x6R5KLFb2~0+6>Jf5d2u^%ssB<8&Q2{wq9ZdADKVLySNZiAX1(GOOvwS8`k>4D_!GBiqbcgeY{fn6&npo^(%7#@=@;7LXgZjE#4d9%7REsFEFTH!`vn>_)b>Hsgo&*I3&~zqGO+W?eIz1xxc`m8W%hTOPozSF8cs7!JXj%^La4-zX0zj;ZYbH9fwc%-)Q)T2Jq<Al%Pycvb|<;P`sKb_)po5s!X+yX6b$&|Kv=|KQHYyW#4<k%+0^l81t#Rg?kyBPG21)S3g{e07TSp?)xveS52Ig=EgX8<Q1$3<ux@D8>djtr)>kxW}0hCJxzHq>WtiTjSFii6=NOBvnYczp**+LKrscUk*+D^^6$?#DBpKP`VXWwC>m#b?Lj2-0ja*iGRPMKJ`W_{ToA<=JF11yOf67!^w9mRXyfp3<`iF`6B{1?AwW-Ag8^xJayE0KiY^sO4qTLlNMgAG<RRGl5CCFq&gW~Pj|#_=H)(k2$JNmQYTD))$V<ZZ8{UZ-?mNfw7>ln)^0TRKFvEm2L=3RskgLfarbHK_Ej)&{yVXY`JbpkRHb(VCV-NYx8@K#KFu@q8tpE%-4#2h$FW|Uqq6r};7V!|;(|j+Z_@f3=WYKSA_&-wH^Drsufho0-J4Hc9L3>nE)yM?nNUu&_WzxW-K@j)>K2||E9T3_%VDdk=fmICL6`vRh~<*%1-#x_O`cHgDO9*QVtfV@A3ojM!zMT9ua8bu+MaC+K;_qDGg+ffuGpRN%`Sbx$DBoT7DBu~iB*;@kk~P&vhPTf1G1VSYOxZ2S}TQrZY4U`nQLhx&(Y+z0)X^I0ZiHf!fWS6I+0*dZMO?cWFWjRCOZd+aQCh%$Z$)<Yr~vcXg=P#tUEwEnjw9Kimg|d;A$<h<3_t)Irj7OWTUKabzAi!w_2~)PxD8w_!-CZnrv9`%r^HIdq(@-b3yy@w4-?Jr--(TSDmHqn1zH6nsv4lY!cZxIgz$-y_6)bB2xk)sRoC4R`tcH%c={~*?#ybMPG$XtI099WVWZ425fCP<q#~}<_Rq(Y<u)6-RLE+x#cio#D5KrPs^WErzOAL*4wvyh80!|r3b&mMTu9Ir*<cEJEm68di~{jC^Lfz;ML)XKGj(_Jgqd+pxYnV+))^FR2TU;SPnv;$LvP-4nM!wN_~>?BPLvy+F-bS_X99@*lKTa(CEGcz`(8b<wI1NxjhY}5bn76i^N<P-zfjL!Lg-&|1~&n|JUHy8QA6y-bF4sT1SzkT^}ge$Hz$PxS79)#W|cT3`kt6=HH<FXy!ugrWdnFKuFUYv*ZP6$qHp|QliT}DWTxYY--(<Z5C@QR|fYD0^U29bgRv)!*C!?d+Wv-1R{;PTbl0iwGq)C&iW{ckSJh>W`_uLhui((!XKU${yyJhMz69)?&7@Fb4f!!NSq~ew{5pk)}N@22u8DY2%caK{qU{kq9!ucPH~{Ohw-*ntmWDAyb{En({%2SaJ0`x{2ScWM>u>=DFkT8&24m)Ue?IA>Bkc$3BVq;0mb_=Q_yK;hj8EwO|-Urg`txRE=-ut*ZC~q_=pGSDaYzgk>JS(zDEmI)^?|3Ug`Eppwm8-b#RoSUlBrfG*$HL=cs>u;y)$0GVhKCkF#!+jV8Npb2f*X*WonYi-kdgVzck_8X+(jl)k@+4Lfu$^@Mc&a^*3dU&G#3+1o4QGj|8O>*%^^uvEQ+LfO|S3Zfi+8$GM*3_IJkqP*2inj2w6*Y7sNRfe}jJPyf~S6+<kvuJ3&&g^~y?=nEHO0o(fhI`%08LY>o3pRS`{9c#6+~5NH(EdpizgbJ$Bl2}6>!ZEz$O~X5EB0q_#Zj$F1|*l{$+J=@>}-OzXM;ySnJ>_#EE}GF7@&RR)Ozi)tUBAw*YK3>UuCD#uaj7QVn^#)FeLOT^f2A$ZnL`_9SDsra~YkLX5ECN`)<MN=jNGW+y`Xwp$$xM^|*eg+3VW0VOw*lVn9?Xfzz*yy7n&W^|kxF6Bd6KBJOHV4OueZrlR$nt$~EsN^Ta#UTA*01$GRa^<x?6*Ubs^|F%85nxU{#Gi&3Q`>7A|r&Uw0v`_7iwyeDqvM}QbF{<*G3rUsM;8pX<DzUQRU;)%`S&xX8Kch~B;<I=f()nC2mmm&~Cl{>-)FJa-8%HZ-X3{^K#q?J(Qjcw~8a>$L`{=zsle3%@-}9yQIKHqyc3D;HjmD)G+Ko4wYqvLDq0z=PX<O3BBiixNMgGk7d@+#o%03K~&oUbTU3{+`v`Kr#K&Ui3VL&CKl6q$tT&+`QEmZH4l<M)GvY&xexmg-H@96<U5E@q>xBpw*HH04<r0%?hI)y}k@0QD5+F8$B`f0-U&ggCySfhpX_8)oP<u|RI<?fR`TPYA~8H*2;Fo`9q-)i`6pj~KSF73)%&B@b+^h~?IA*>D0KJ;<lM-jR@=j+#8t%NQfb%ZW)=5A=D+nePw9FMQLs8w@7`x0_>8Q0$GgfF-9&&rA3EaOYP`y987j>b=G74J`SZhYnB`#59@OgYJf?0v_BX?-9!+Ay{uyhnvijB`^i7Gauyx=j1@wdOmrVNewMVx&p9v}AgE%iP+>)#|hEQLCP<>sy62uWd6;<vEk(8<}kBr_ua_pR^XSV$!J9M|>lGLc)8k2bCIwS5d6YzhO4fO#D2iKSsB@I_0x`%GRO58SFHCF^){Gatl!v_lwkrA06^htTFfCOP{n`qhITLeuS7)Z&$kX^rz1Fq&Aq`?&vy6r*wgB3HM~0uYD%H1a<%&Z#3Jlwj2592y%UAiC*UJ^_PJDZE4wp`oz8J1Q{aT#lzicdT>%tqwZ2|F}Ej#$sYX?u)VxBu6zL(e+rMdV?07D1hyDzL(QU|MUy(_Z3QT_cK?a6gWyC>ZQ&)1y-T$1#=F@Q{?PoetfRvPRuP?^-*;B(2>dl9uFp;S9z-nvGT_gLck0U;G99@Bbo%@6%{($i)!L=rMjTfmn(S!(!Nge{1|Rhp9Vfe1`7En{@W|(F^Lkzp+voAPO@5Q+M#ZIg?J)ZU#$szMv(!eh&+58Eu~-Uknv?Dr7%l65?d+<{V0?V^+N;B7$|?Is=cCSc!F%0kUPAFeMo4|P&c;KO6!Gy`CHyC%X;~BTPMIsBL_i#2;vlA3ySn=YVpwk+s#2wiENrT8ReZ)Z>dp9jIW`{G?$$o7Z3r|*r`~%F5Fm0iu}Zg$KArxNhk65gJD}Ein=i3Ca;}DcJ7p6)t?-`!G2XxS?s^PHuzVj_2JH(`>(9?zWBWp?dhZ#0NABZMZGDacHmPTg9JoD{l^0!SlgDJ<&2viY$y?O7$~W6o#D`w6H!(T3+S`T}jqaBczO>>Dd?HUng-C0uRItkC=mS*(A8aFRYSvB`J8(8k=i#v%i$I`fZa5EcYA=b?2K-k{57x!26tZeVU|?~x+Es2jUYze!kKZhMhg1`IqyyYZdVQ$o8%HDJCUc#fx8M4DPPUF=NJXo*Pt(#aKK$+6&N{_=eWx<m?%f8JPUB0No?O{oUQMRF^lQsm>)E({Drv&(>y1ZocOjx)!FDw0AaQ5T<5|Py=S;v#*GLISFhm66vu<^|ou2b~R`M^d822Mq?~WO@`CH*-=WA<r2}~DUZ=6C|eNeRMf+dcUfb{fwy`R;gH_7y8V|)@-8ujyOj+Ub(wzTG0+C1NnJGSS$T#vKejUeCY%5mT{#@;7I+s}?zbe0g)CcV#5+05w0=kwc{=hvXx#1=?cHI7{?{jrw@@$F;Vm0mnR5b^6|@MAmu*PM{~bC-Rts#N9u>)`|PEq|fohur1k;bah2C*t%|M=4_c;pXj?xXsu$*hE<N@dFcC+(=um`K6Lhr)6F={$0zun`!qft%4yi--J<bvQy@r18{$LZ+`@QrcI6I=9Wy20kaR$$t+L85EK4D^GQ>ye&j;GHmY3fn{xMb5>-$R<H_i_`>Zy>H?B0A)de?qOK)U%2Gzy)Om(a>D}STxv__Z)qb?G9HC*_a=jZ17$mfG5#7{bNWAK22d=gW;;e4!F%^zT_ys)q0wez+S*FX~2N7f=lN8cs=s|aChGHS2xH)GJZH%PlJJ@Yo@(iE_-c0e)iU<rMr4~^9r?Q|ZHh9LVd55mj-ws(|+=2ji?#h;k9PqSqHT7KdUzX2!IvYFC6O!dxfHUy^1ODuNeOvrBZ_Ia17Py0JX1_S3%l!pm5tsb6jb_X1U8vZyfT3*{-Ci3ia(FYr+9o=%QrhTMct>$m&O>%nznAZ=AxtH92u2<&MmDv3>DjWZroXY!4VsZKNhvHSXPaVZ{lak)@h2CGA1E0burH`lM>!tV3#?*MpYQw22ENgm)?NY1--54R*(5&Qu-hVhcvt4DCAPPT=5eZ;S96%%rC=Lk9pok(D3@U@5APC}sKKt4Kx#yz0x@xU&k#u?{JWe8n6DK1{pNnh}uGc?jbQs5>Bh>w~-j{^>ec|i9lfovje^5NLiR+X3IP2XLFdNn_w6pcb_|ocMCJF72*NaO$cBMPguce#2R{2zJb!<QPP_0<3I?*8T`=x@9iJVqG)lst6cf)tn1)jSu-&f{zTkNz9gs9aXT=zU7dLB1w|9VxylOM{cf?u=ry59uC!6%X-_j#D0;H(Oy+6Qo@_AX}nV{P1yV|wihJ~Xpld^PU<%9Y@BS}T)8I`1*PzfNuKHq%Ej8ja<HvpGDz+<ik(18meB&!g=E{i{~B50c;8qs2AA6%s#@8`3)LsoWm6z@iIg?{HrMD<s10=VLdtzYFP{SfeUkkS~!J51PHVklC4i=FNCkJ01R}D=^>8<hn}?Wxk2;H|B0UI^;wn&#_d;j#Rb(b4Su)Y<51T9@lO6ujW&}-T3&TcU%wWcq8(Lt9scy-ag)XEK<~A#!;6=51~No0}_ESD;pK1%ATrce`cW!+p@222jiW4ADW4rcS3yiwa%S+gS^Q$p;S6Qx=!Gu4{CzNI`5bJB89fK=9KYegh=M*Wzi@D%Xvw=c@f?7qc=dXqyKoz^NL%4J;vjNaNm$$J_c@h40Z6=x(-FLoFl0IR;=rvjQcPBO8@O|jpF)1-W}>jv%KKfFa6~9xPCmAo3W+H=tz%XVqLWp6!$k3-(O>kvp6hvWK!|#Ur*71-oEa3&AT?Urfxaz3MQab|Jpu6Eq-MJ;PPS2A`-hedA+QTYl3@T{4NS!yjGi!SWX}Kw@e+VaZ@15A6DPs4!*vOM)y!o%yf~i%M#0>_k3RK4t}`W9im{hbNQxpU&HwW8E6aQQhNr&2>f<kq}e4orxhtKUn!--c7F#*Bpu1^ni@jCN-I=BUhsO~9N~S`?Y=Md6vz)+@%<4DBg|nOULfZbHX+kx264#5Hf<x=ZwHE2lX|bF=%oSOKiaBM8H~K%b`x!rfJ3VA4AIjFUh~>eKLls<`h`^``muH4Z;mdj)@o`?pNVZo_1%RMOj#pul}4Hw!&Y`Xy^g;mf!t&=iv0BWQ*Y(_9uyUw8Ox$KfqTq8zkc7}fSqF^P9BB^6eyF$(wYzE=`#t&lRXm<U>dOi)dPsFv&q@TPb<gVdkE>PZbY%OmeAck%3YE3!G*jE7jRp<6>paZ+9UYBj*Qc#f55Y*Y&iJjaG9*@JoxWs8LmxP{SI{gqFn9zhs0GCaR#p?GVFG5Hg}@EnbK5spD3hrwZFQiWmH@6Qn5$-B)JfAX_Sfy+ut<?!|ysNUbWLQf6L87ZSvr|CGhgVxVJ(2r>8$N>6J%yc3?8)j#WPMIH=JPcL{m@!?>6n=8c7o^4*&x3jA7qGRX6<GfHAQv*Tq9<KpJAs6Dac(L_)lPezRQO5eW6DtaALIs~U<>y@-c%~Krn^s7cSElRb>7TkLyIF7QT&YEpvCEB0#E5K~)v}(0K`Hm4vBUV5HPk)bgHdAsE_u3Pz+Pwcv@O*z_je@av2k*|~6>P+2?*j()8}Zj6eGOcP0V(IKnRZr^b`h-O)P{zW=E!H2+aC(P8mVxipZ0mqg;B?lrt3Pm8Aa3qbdE7{G;5pTpDsUJe`p3mtjmu<%KR+v+`O%6D0yp~nB~E;D{exP<SFSlUN?Xnn0&LUb*zwL1qS4I@_O8%N%%P?yUx=wJsx%N%$)?k7@r~i67(;%gSUKC%Gt{m0OY5}7uu!O+8ORLxpYtGm2SZ~@(Z}P(pR|E2hQ5ImQ8SA0i$O9e$&U}`~sZoJ3bHeR=gSk&zG?ID%k9bG~_G|X~>ne-)G^s(naN+PpGLsfa23|_xJ-K&l0Is>#K`DkOPE%IUG*xQLhhI*qNpa>pJU^!yonkJ>kC5E%C6&B*v--*Q%|1^ePA4g4WmyxRbQT8+=t7@vrvYh{QRU0P5|0rXQ^__WE@W&KEh@55FIE#52ki0MA=+e9iQ5WU?&TsVV-m*?j{GviFx-_S^Yx`Cjs?w4s(8`E@Iu4J>T-R7Jx{adn)k!GT*;UOSL{W8^3yH}zfirB`7FNMZ|tQCZE4F@k>@qd3HdC^LCd2jo>YHcl7oMGIOkT!?9L5~>eB$8aROx_WjG^=jIvSE_pZI=xR8kuf;1Hrts#9xgP5KCQ}<bMs*0ZjzvNm>ZY%s@sqHqxEn>EnhnYS$LfM{ZaHfd?(}1%!EhV>YmuJXZhABiu8s+nR2xLNMlQ#FFlv&tnnW#T7<l{hm7P<^rU#rL%bV=CgQVwBWic=WQIk9*{5FLQET)#R<^5Av?mgk47`TSgRf598zKze8w^sr)(?^%ga74Hi2AmpcVj(>JDz~Cu(*xUKX2-I7SwTw+AFyd?6&hn`{-_T|KaV2h-at4Q9=+}Wlq%0>ipmjx|*OI+gn!lRngtk&<v}P*b3Yi*y_|ZMftu5L88n5#8PumysPQ&Vb~Zu+O5-Rj1*29tgFBYlFdPR-YvXzH*-7@LiT+HXMQ~1>U;a&TErU+@aZ)s1%Jy*qn^zc(p$c*hz_M6DS|NVY`g!I6p+1Mi}$w2ysA+=H=Z$cLGA5qw>t?<qI?Zx5U3-Tm{v|eLA1WZNq+W{hBW#P7Ie6~?gFp1k%4>e0AKh_@!|9I2Lan(Hfww67K7ZYZGdiPe61K7(0A31d8h#|yt~SSW<NQuuO7Argz|k1{H&wql=m?Qep)BROJX!f$Gd$zLRyhjF_4pxJoRGrdC(BM`8(S3zXvj!AH<-l!jF5*P7$`gb-uv6dVRNw!qsolB;4UrG1Y2k2=Bb#MM3iNSe$(G%GgUp8((Igc*w!WHgJJFPA@rZS?ZsS<dJzhk2g@|CU)i^T`52|Z*Q^5qLI*1{Hnj;@%9u0PF6=H+!Qw6g)`O@y~p;NCLdL5&V<kBVb({L-@U@Uf%-X>$rf_?t7|@%d3OJ>Z|azlA>0AqgDFzij<Wil$)n$%rehr79zidAat`0SgT(3K3S{sHE#ce0nt8C<wTIAnXzQ`ox%OTb&)5OHn>38ycno$8Pidg<P#d1U6?R~Ifpt)T_gLg=;B?^9&3R9mXv-HisA|GmTHUH|<y`yAz0k1UiL)1ss|#!J8=d!*F-!7+ZFS6ftoT-B6+P$eZ90QyC4UX}%!E=|dr2h%va8ols@E<N2DJTC{=7nnpo$0XIlP}A((1TZ_L;nbc6!SL8r$vdeSVMmKVCn}+wX{(*LdVU(j8wcvtV#lnddQZ`(()08$a7Z2ijMncdbE(q|lijOcj*46xYql{#xN-0t}ZxYZtLKrTig>Te|IZw7yJo{k2EK>%k7%<_L4U%|F;|^HWVl_d-E>exJ;50e3}tw~{b3wRq3D#M|NHNjwfXMK>{sCL(_k2qSGO$<>Zn!Dc&BYaYIK+C5Bg!mW;AldQ+b*6S3Jowq8SG0b@kOs2BG?uBL(!Zm6V1NGh{<p;~Q#!{K(wiFHzfGuYa^>3XHy~79C9Cxz4ek9}%rtg=@-hek11!G%f<tFv?4t&cGtwyU_PLai>zFCG{dADa?i?e3f*D0_LkqJ^frAGBKq=7U2nRU2FJ@nksCnw0uotj?}(i+x6sx?&?NiMgc?kIb38_sg&0cqdwYMJ#N#5PsGLd`b#MVxMRKi`(!-eHIh_vG=_yYG*)fgF-#0-^d%#h<liZ{WUldPsMHMab5AL}Lbl-eEc!UDDota;a301P=f<skXkn5w5A><(!u8s4DU3GUiwMltQhtHoaR<wY}UKuj9yR&)2od*_|?Pt-Aq=B_^)zoSrc7m%a~I;kau4x@YVJ`~dLB<;Ho$5D%R`z_-=om}&jx+nt?iusew6Tkoz?W4z?L(tM~6HQcIy*znb*%X$~)s$e@6c>P7!$}4`|q~rRpz7`<<JYfL4(Uv+be)lKVQ(ChYxB1na57Vt#2*7g1nAww&hIYmTn#Lx3l(R+-^T#0%RQ;v%z{9udG+;*1Jssc7VINywmvx>?p>n>uAm`4Quh<QoxHfMT@{-pMq<&jRa2kbCWqtn}=3pSbgl>w$J2??LJ88Y6tsGyI!|%EM7he7py|+FlBRE^);7BaJcJb~YtgG)%xUGL&r}|WtVK1y)8ryeSg^pcFPCWN<^457e0@C*Iy>Y@F7QdByRjubbIvtOg_i)#z&W!=!#zcyqQ2+V3K@MRmmB08}1=Ve{PyFe?bU&CEjmOKnowSA>&8}}`&*WC-SH2SRDKF>vERsfc^ttA3KMdJv*0sSJF2ke6y<GRcU-uN;<}6SlW>|b^7oQQ9El>9WC0kT_T>nH&R?7Ol{A{||+c(E|aAA~q|F3Wd@#D^A@D{&JAb#ct&pJ`Nbrak^2a>1cL}ld{uxF>>>+$?<DzmmPZrAkV0*%0%VVM06ybgz*;H=3_es<#Ku<P2BWv{&w&QoWhR>StNWGAE#JRCXHZtJsATdNQcWPgn@FdP2VzTMJSZCXxtweEYCrtOS0a*Mz_bRR#xTb0JFQbVVFeT&Wj^Lcrf#-w(8)l55dM0k-Te+~&@^@a+a%oa4OcY;o}`q~T#x(bNj^i_GC#&zrzo?GUjv#JRgM0q3jjEpbsz0HZ6>aD)YOs15mt-WI7gszcq#6}KoOm~!RrgW=_vjsWFI_Zz`>V3)YrsTptSv{2tptox+MMtd*X-q5Gt2!myNx6EK=;|CD#4H-~%6Xc0>HKhY3qt-en<l~$<KfA}!*R6RT4!@G86WA*&^IW?i|b3Z3b(34b2kPZtl>aw6Ykc;UB{}!cC{_Mp*$`<IY9Ckl;^9-e*D|*^+^dVvR@PJy;#ezmmz->8oL$H^nW_(jGl1OyZvqPMGY(5G+L@%Whd!>Le~s>^(6c$7q7dE*=4U{DbIR&!s1=%cLMuzzt@-3+c9d0>7f^mB<r5{=EFTka#%tK{>#X|+uw}ZO(O4e3+{TQVh*9X{=OeNS>{L}!<yFe{pp@E#KJnY;`rF)CR^Xn>-^m(qFh)fjoKM&mA6w(9iAUib=IP6{EGG7y70Q3zM>>xx6{5l)&*%l-N1Ey@e_Y{X_~b0X>@5Wxi7X-#CzM_lRrt-)q&ymSa$(0?IHd6%?0dPKQt4Z=<uka9aF_F`dL$HuikEaQ^nrpgFwb5XMRG!L6|d!v1+)hzDvzmzvK>=hv|*%+;XTTS8}Q~27*U?dgPNEVxEX*$hsAh%g<2yq-<?oi-t_MB;n5rcRgUOV9G1=AKk{LG%pr)a7UZT&3&VnaHPC;qf$)8o7-Cp0{fV9Apu)AS9W;<ou)Lfqvn|a^EJ(V2d{~rEYa$vt!!tCz>4Gj70~-1c02zS)g?niUrYtRVSB!HqTC-f0-_fJOi;RT*!<r1aLa}F4MvwrM60mE7jcN|ZN&*W*;O5315E4MJSM|Sq8YM)RN2{?n9ZzxEH|%+IrZ9P5M6_Gmo1|G$Yw0=qs2Q$UY0L6=&m1&VS_;CqvXsKesfpO$5caXwv#x@6U11rQ~K5<Rz)ir(~njkelp8eLx?Jygxf08SLrY5BNFfaOl`po8cQYUZ_P(^YbDEdg*J1`tq{~K*fnkooF$9UnC|#Wf4AR^NNKa2B$*WD^v+oJ-V>|cn|z*|BiIbRXKcL9{$Dt+>dk#U^J7=KdA0Ml-7RI&TX<U~L{<JPJch2P9ec4c?H2J%4eJ1dmfYyw?28V%t}Thv>U{bz$7ULxC!;HUNH-mf4?qvim*b5pC}<v5BhJ>Q==!4A<6RfK*Qx5^b&<mL6{?lH@xd$C4%N|-U&AX+Q{aC?WpzUa*q>Hr+Xm7)uM58OUAWi)67+pq-zMDCJ)2r}Uq2Q&VI=`&9qY+&!lmQ>b|Iq=PRxWhPol?UJ9&YC^Nn{k-jVPge}?yy9F*w<1$UEndcXCQc)zs-@Nw_=^8V79mUG0VPrWxJw)AynU7Mad@<M}DPdoTwv-f%wqyW{zxM?Xwc|bq2Y;k7wz^^zpGz9C)>Qf7`8ghtDNMazjS~lGL9=JoW1)V7PNuHV}+I$~xLh((Nw$-zwwD;GJ0y&@UyD2wmAj1mP*S(Cq=`$Y5$=+nCH4xunpp9~XXSJS8t}DmtU~}tgu70s4Z_3StF>qdd0=X(r@0%MzQL#hq_a%8;AB!>)K5s!@r-6F;Gu@v*DY%2T9kY^bj*2-etqD;>08i_p+hPS*?vr9=F~z60k6w4!1sd~6)f>VZ${NhvmfTGnR8cYJ8p#A*7!b|$GUp-2Pc`Ut#^R<sdgR)>{hQU|n4d}a-V7(GX|~*lNA25kj=wo3ndso04(>rU6^<uGn_h?ei4#oG={fm)X85iHH1XPMBFi41FGKqAIbzF`@V+!QK~1{qovmE<tRJ|<+o5MSTLVyX&wXOv<ccDg3y50Vzr10Uy4T5pIIBzB3Yprcz=DTlvVq*6F*$dPn^=44_e3KryW8>xog=uiNZW0gIZ`^+i?Z9J@|3R?l~}NSYS>CO7ogbLc(?BEW?IM(D(gQ^BSwv_eSPx2F4-HNKU^A#?aqmv*|+9`l`n*raUAZ|yIsG(+CHmUCvl3dICQjq*p(MFqGtQ`Vh^p-gOwkFUh9&38t)FB-_CJLM1FLrFBVQuilhF52L)`ZZ)x{BJ5kcD=lK(rv`MJVEn7>i4$!!%&f{@AsUh1g>qK~2pQTT7WVrYK$ot4*+G?{j&N0c3*;W(n-q;4)%axqQOFIXd{pGBJJz$8=b{WRh6#4Q*;~Up@IzSz)yjbUYcGB4*TPH?5XUE*vtfiBhlrK};S&KFoKipNnqaq7AW^3tgx;{wv2?1|*=<WHsvC=z!uNQ-*6wUV)^E27+D;B;(?wy(q0tGjmLQ?ZA<Md#2jGU)7?r4X~?IajX(F79l<(32Dk&l=9C(o~~Qy;bN^#iUXXsmBvw^{GAf=hy&J@jMTg{te80PFEMFvsn*Iv@MJd)|5=@=^Q=Ab5wq8M??i@q4(jJ45$^3Jcp%3c+CK_vo`7p%Hb~n+NPs2c6nHAnqD9b6*rB2{$tIukM2To<9?qq_$#=upeoTOGEwx0L*qBsMTX<*q^qt!!8BA$M6Cj(esws#KaPecbD8U?lQ4$lb<7ACgae-gn8GG?CRZzXHy==2D}YtBeoU}ZjjeBRkJTsx<FVpk)4>(5zME`Mf?1aJ6kr_lYFM<5ej>oAGA3HjrO2A7O?}>WZUN5*q9%Qd+7}Go7|4RQSZG`$f({2b;vQN+I3GJ5c67BM&)pie1rYy;QX!yQK>b_@sc(>+xFhyy|2>&+}ZVis966Nj~yL29#yk;heLF8MR;XhVCXt*IunHD9Cc$7op~u!yZ14L?p;c)qcK?hjKo947~|{FmtBCXWq(ljp6!)D0dJFu2Lx?F^X!{lbJH!a5A~6|r9ynR(^M?10`qy%8eqpn%Wu4IF3W~`Xg0YoVg7ruh2<v!rPP}4?xmouqNRAJ)wQq4tJQ!u#~xn(36#qQqvIV*#>)K#lmeo;-9hWTsf@<+RZ%pVjqlWoUT;PbxjCEk#2jD7Rhh&MTU2hvVFC1wTc2NPqC5RiDJX0+eY*gs_n#eBFtRrdr5z^MfBo|n>1SB?gx`4|^8+35knJ?OXW6zagq~W6e>@SoMYybcKkHbDC)s3F$yG8*k;cV%+0Q^E$Df3tUNaLo4IrF-9ZCuIu>n2(l~WN7-VxZSR#%1~DZ#qu{+*{yFAmRWa!YWo`rXu?_hI&6wynPQu8dfrwcf!m?l2l?4|`Mnpv9nb`8LdO&~Cpfft)i{UvkpxjNtCAf>_gfg#<tH;@W(L0r7S^u#*kF6JEv5BDr1l0v@r$#qq8Ti5Wf8J0EFrzdDE+e_l?bt6-Z4;1_q|Hk9X$WCS60$&Zbi4OGF9rJ#)Kuklay;AHPF3qC*N4^PVVc5-fQoy+HN$0u*2>#vv}-Uy%Gprp@u%N!rMVYb`OD3}*zxw~wl?Dnv7y8=5Om~f|i?DpTXsBi@6m5bitHZDTxrtBT<+W;tNf_z<aAjAaLt=W2?)*5}aHFH1JI**}8ZA79MduUk5<T8L|iMfm7HHa_9%D=6^YiH`7$9Huat&d$6Y<7&nD)t3rM0$Du$$lG`eru!N-Hd9s0Eq9*Pgp)p08{;5+eZbP_vJc3?NX=P0zZhRO0g*Xq4?eF*drtExzBx#m9{6K`kNk)tlr{Ur}^oGr`DP<7Q6o3jT=29ogS0qsb-JqU0U}i?)_E$s+EZNxf`;e!SMufC9pcFhr7qCa&PN~>b3%_!5%l0+6UrL+toZhnx1cXztS{%2Df|Av_~-mPIh!%b=k7~;pfq{atb^9?28Ocdj4WiNE_$;L}-?$Fqa;7aiT~2nH4>Pk%29(yR^6rn5$kJT#X6BA-UgTlj^qG87wwNT65pC^HluvsNnGkTSwfPy?V)u&g-AP=RVBakEem<bi>!>=k-+edFrGJG&u&^(zsnN=t#XS#Q`8b|GKHowR|5$hPaKr(GqKpj_`IJ?OBvtu9{)H{`<m}<X-eLZbLV1e)H(*uQ&H0cU|K%aJ*?B$<O8@ZNVol;@j5g0zAp+dXA<RK)ib+zJ6i?KI8X`d90Kpd7m4B)ydUZjpU{(5e3C>6K7NJ%{2Ky_w?&>`#C?tmu&fR_4Lf^#T#8VJQsJaV&dG|M^xL&)N{yMuSz|A4B8KxHy``jM8gi{*D*c;vuiYDDKD61$kcP0yH4UQQ?5$g{tihhpR|llg}~k00jR8;0XA*Bf!Vue{EV?(lPyKPAGPl?xw$ez+?ZDD-4)t65d7hEDE7FT>U_=G=tN=HA|)*<`8xheR3J^-9#nsNHn7rvm5W%P_L<kW@Qc~~hwV}2GUE+p?L?WW%i;@1?uYgRlb}}T_Q3*gaQp-CL#t+2SF2S7QdQGO4%hS8L~#Xgng?<HE6Bz&`n({~e{E&BUhDmOS@t8Ui*?=!(W`cNv}kS7I*E2`Dk<#>WD<B_w+)}FAkEC8MCQ=p4`JBIOawy5<t6=%CVyMqUHgQbeFjgm*DFp1`+UaJ59uXO9icg|)^AoWA!6zchG?Yt@Y|5y-I-<d;M?bHT`EJYvjGN&32fJ%LtgIC{tz+WKdFA0R0{sH;CTz#eY^IaxCVboaCYU|GtllwJPSAPZ&&dbi}W*Q3u%5aTsZ=@?eAC{dw=eXpNr?F=FG=EwhC{oxEy1D32%a^Bm&i?;IZ7yCaJxw3t08SS+CZ0vG2xvmi`16BBEm2NWRe@O^-<(1hwYs32rlSyKf4tx)=O(ph5?`xL4lB@JwTDxUa}BdbZW(<r(C24)J>2;x`yf?*|v{wdGm154TgIUP)|zUfCn1zV`BYU+tf}YQl@PW(8T-pU%%;Hl}xg_>;MJ*XoRp3#*tM|CXV)ypCygk>mJEMF)5xGg*8fFk?Q{cD+&rTEq9Bfn?3H+Zk3y@Jbb1{6S{;msuZ0#Ni3+t5P*+kcEy*WUV1+ZSvwa->Nm?tMkXe#-T&S^6NJDdbavSvQ>h1sa{?$lce6oh;`rN`LL{DPvJ8;v;uz0EB6IKwq?`9-{C%$#u_bOKc;obR`ybtl_ohPmo&9+b&Gkmk=}emXhL03;4M)k=6^X5e}9z_ioHMZz?5OsZyn8*HQZ8WbK>Y4=^`If&}!BSH*;j3Eix~yfmzs?#XwzQ9$O)Og5RDhT9c$IKim7wc)97=>4}rf<NmB%)4^{F{rbzxC%DrDsxOm$^s6q6dbd87v*O$66O%o0zH&tuGt0b*9Z>07oPx@uP_I+CGNTH@1&KkkZ;jKz@EE1+kM*!|#$Dv;@34BUq_^E(TBc|aDISd;#WB1X^Yru?9)s=#O^sd!K-Hn{Wb>YR;kBbjtC=zAIX8P-;4=F)KjwYm5#6S-ZjVBOLo0mzjBsXk<sW&49H5}93OJ!mcfNCIomcL{&EHTodl&cG*lbe}X&ioFXP=&pZMm9W?#te3ENjof@3_&`SP_>naP7Wgm5nRKc2f<#tH6G4#J(YIpX00w!qt*wj=!|ZWrNSaA3M9w5EX8&@y@351+yN#2dF!sJ-puMTMF+_K3#LUIe8#iZ|i=j%X@G+PqN|ij)r+gb%GiiJX&(b{}F2`GF3z)!10g+p-Se;4~>}Ak=10CPPqQ>bPa$6Yn@DJXxHa4F8|4@e?3G?;e&stL!I=asAwEDu5M&la=Fam;t&^KsaqCTK0TKMn!|aASJnKrnMG3^eM};8J8!nRyc<aB(wiN|w`M2TTu0gXX*8&3qu;7K-}%>1n~!eOQUR4w^*(PsM%Bu0UWM0Nq)-5Ex9md&e{1i1@$V1kElsC_<7i==j7}sgTy$frZ2JgiqQ)BLd+@PyP08i!mhFUBOYpPpXDd9w1YcPXzXkM;<G*1mav`}Pbh$Mo{o~MPMZ67w3mroOxW*Q{Zt*tUpEhG}%xlOW?KMQyvPGU7WtEjl@h@9HP7kXkV$W4Ib>hDJJqV{WOlh%)`?ZJZeQtwVs5SVX8k5bt+w>y@8?4p2uVj*PEPL^*^?_g|CcMuTsVe8it-R`o?zL-f@7w3e_4}*8e3m%g{8POjl+qGr$+T9&leiCp&&D&HRPEl4I{6q54LKT=mva%UjoT*5euwWTk@-zNB9rGzbQWiCRo*s!P#l)5$Umpv0K?l)wWj@UJAJxj^Gjf~ls54CJ9G@N4Wb)*iA!`wa1^+;;JO-LC0x<avytT(2wvIeWd*5F_htotq#X}`Q!(~0X56Wlrll`l_w}aC(!C2&({Qz6$)^K^mbQA`JN=P5^SZiUH0HYedt1cY24M70H}OE5|F!YK9&UM1IK-AD(obfRH}d`f(umJQ!1eG19#}|Rofxmu!w)^h6vXs4bNC+bI1a3BNaulzPc-|sXk3^NhU>H``52*GQSsDXR&&SwTMTV4KTOcuADoKG&c-jc=kfRRX{PQj)f>427nDePOI52)HTW_rkyGCaX6hbw=vB}kog&2;&s^u_RT>AY6=HpC7R>|Xkms};pV{pX(`h5-Gv^AZ5zxe5`*scJVMOv<gJUziWJ|=YzO9cWvt))28~C!EwRg^zX@v;vj=1_f>Vy~D0Oo6dhtEQJ*l4n^$b3t|v+Sj6yadDU#9hB;r}-$|^rqx5CKY8-;;DH4ZY6P3$gkGsldky2P~M8{(|!$c>wImB+Is+FTX2qFBnVaH(gf#kr-8BP8LW#_$h9500w(7sv9oik-WpZETG8E`>w`+H#w#YtXsEIuuA%Sc^Yc;cRi~49n<L+m2K|t_pN|G);*e==7@y3X3_sl5mk!kFuPZkA!+vhA18c~f!2}!o1BV<U&9iryXz~mLcn|L*#4Q7g`x>8z&?>MXMU$R~cu={~21O*FQLar;U6-TfHbbg}cEY0iVs<I3%b94s23*pGIA%fvMa&sVZLv`O&qMpHPZy#NiqBIv@`dmAow?xo@;u*Rsty<+{1S1Th%bw5t%v%ixIxs6dUn@cFzJr5N>>^ZswnCzsH-+}yBQn!5!Au$ktfY691pua`3J*+_<Kstg?0O`U3OM^?_a(B>VtNfL94h<-<S{4cQQBb7t$DB#6V1ch|s^*iuY|He2j}K{ZXq2wV+XvYLe{ydo7L&1=-WV?`o3+08MP_ywt=ZV=g~dH}hORP}p42hT0ec{&h#bw!?Kt;*Mf;UbQYe;<jkC7H5uaKjO1Jm3i~{T|}AsrC7fv?`h6tvOIIIUekM7zgZ!cox8lF$jWt<>@W4E)mn0o$Kq$qr$U#h`_u0QTcX7LDczi{pH6EWv^ibF_CBv+WOKYbWMC5SP?noMHDf$*CE+w#)T*>jDxbJEgmpax)Mpiba<APLJzwGbXaa88p#|nB6FfD3VzkT3+6Y)~WW_UwnD*yqp}FwtmG^O7@3lyJDihKNK5e_-ZSX^PpUrv()RzVHCZTue0qbq?D`(@yrnemhoqNqk--pnJ#7z%QLsCF}ZFGtw%%}y>Xx#C>TrL`eqn6U9k8|(X+tZ`Bmdq0;u<BfrV#P`sW;ZkLaf9DX5~ruC{+T}Swtp2leDASV;pb|Zfq{k?A}<UZ{g&K-x@+KhC>WtFPC1+y-Kyh(y*#;+@2L#eq4|bW<F0w^m>A7&t0YuYiAd=B#j{){M|FcR?X$g4*m@dEEsbLqYwhefMA1?<NAhvmG~MMVUt(A*@NTE#+8TMc3(10^1Ci%FvGx{|wL{rfV^VR86&<Aud_JvHFGWfjqcD!le(zl$`r?Cj&TLNIteMnwPWA9U(9f^Okj8rS%NTc3_R(uhuKM?i9=2)x`<&%<p?!2JnWNP%z#HeFp1khQp6~zi@lQQRbGp3u6~_PEP8*qq1Aj1y7FV_fqbKYzJiNa>x=st*?s{7vt;Cb5iCHB~bsDaWv<64B^sedJ7jZv6S*4w%5VgGUOTE4Ko5!VQ;kPjSG3UPYx6e*9&@TuCx5(DLUfFG-FV_7;AvI6q%>{Fxiu;(zQ<X{le!@GN`|R4mLV8?3lOuJm9iG79d*t#vrfd5CA!YOV%XXY&<&T)^bu4Dv;m{(d+Jx_w!h$45tXIgZ_2ti#6zC*vm-d+(uO_&4!pKKLjuehS`F2Z8&3^0oHLJMR4<RlS7O1FX^&5Oje!%chWs}rS1iu3hj$`(C*>;?vD?sUB1T$UNIt#%qvVi+TsvPlbODL~c9(Jf@s*eVVSld=lT`5uP)_SBgp0B>9TpF+;t@!b?q*tz=KRmUd3<5oJzvdr4L!4!qUb-MoNV7fPP`<)5_a$H0%D|Z4_JDJL@~icHf~V$Y5Pptpv-ek|wjD@l5sOW84M*5f<9{hf?)6*Ou;l!lY|h(QWS~vZSvLdgOIfq^qU!~_Q~Nnl256DD+WH>a*RzHvdkaQimjJ7dqqU**${~9Ph9P__?mt9YqTc-NVgmoF(D<8rcdB)2Jg(@gjb{GHlcz@cNg+npVxoTv^H%IP;5<r^`5$p@Z|v|`UKu~$sFM2HVHST4an?Q-t3I|F*ju7e?00W`wA9FO8<>xQkcybMGwG2@AJv1@=%2w$lDt~fc)vG}uiE;^nAipAn!yC(e-<M>A04#hUMMY{HomEMDSI>PXYp#$VVlo&d$k@|3!wT4)S+0+1(h{1v!EE0t<;ZBZq(|+M~#CgGVQyEhp~yTS%B;^cIfRpHy@b1!<3{H-Wej_s{&!Bo1$$oBC1j{rQe-_dsG>wo(;c|F<DJp=Fz*JtYzA;UiP2&nBP8uVI;iSXiQ@mpp{@%iTr%~;4AE3sh$K709JCY-ujhBe&itP1j1geT%khcvqV1gn);c4y_RWO@6+($lE8FwoOZ5^a>WKq&3~hY_v>D_#wGCLE>Kklf|g$R;5vu#-7?<af4hNXMEl8O_bc-Bl&r18EX)>X|Az|I<|f|~)Zq(tuh-R~Z(l1n2w86i7|{V5tyQqvdY04Q|BP!V$f4@Tp3lS{2V!%7(3CxnX(cMv+ZlbO+EfP4ZWo;a;`yj_@1O<xEXa?^8*{WCb59Ka`jDEYhn{XVPN5SdWxrmnRGq~fULEsILi7gk58qEfIvb~Z|21Ake;Fpjma$Lm`*Ee#uBqT$@|xDkkT5q1`OmJPZ(rd)5UYsVxqkPx9nvzI1vcnE!cixw9OrQyz3yt1!Kv}?uii+PlAmv5Ghr)g@$U8xOW<7zc>ecTKh#aYbJfpVLRR_u?xQzRGRT*D+Z$>Wa?qrIp}vb)YRK`svL}m0qjOFgTRM5V$hG>CH^pOKt#gh2EQZ%g`MTGKZdG)m?qc#_Rvkhcp2e9sY6#pPT)n4Xe$Wp@_)-$sX)_&9H+g>TiCgjyG_+>)=TTzD&~5eRCkIMkw(V-fedhHKe#!o9WmSWPju3hx3?x>Rom@R+LU>A;%#<gH!LV1W8$DI95Z<6?yXo$;m8T==w_VG_XxzrHL1%ewQPW0Mxn1r-7Ldgdp@V{{Jee7Bwp;GWt2IqBI{Gtdy~+VmCaB+|XsO#jlj~mE3(2q3<oSU1;?n7+GBlJO4w)}C*DT&^E4UofdS?<=-R!DIwYOddL+`NM<W(7mdoAJ3pjL$-NAaTA+=kO<<gU|Kcesx$<uhov@3p;MKB>OC{GL1EP3kpI)d_w|ayvdNxj4Qbw4M+hRyzwpnJ{S$Wo;0g%6c0bDNy=q#L}N;mW9>jW)iG1!r7T2t0&h5qiIe)5T_u6*sv$gyP#{=L#)yrZcdYVyY)G@XUJVg{F0m>R^?^fUUCBq+LF>PTS_FroG;B`yr3I9sB_Lnz<aAky?Oe~L6eMMzbU6t{gbZue#`Rcz8e$qb1OzZzAYT-5Y8LRxlPK?$%}v-9KJVU3-#@>-RurgV5`}U`=?SIjEMz@OWvIU$Y?dzH{!fFZ48|--rzytld64*mIn4n@SU~44tI^^l}vhqxercUr{d2m$=hqZJ1dZrYtiqJM3T12-|dBfas6$1S*q=m#CSf{()%G?153xBk_^m_)Oo{6-Ihb1sy+ftopjf<Ys-XCJ?|=$V@Vj-C9_+@(=o7&t5n*KBl1Ct>2zy&)f<}0oy+0EY^yu>-Rl&yC>ChC3tOKcAgP|dTFy*=mT*~!VOnrMK2FCjDbYvv`I#7gRgs^ueXq3Mwoz>~W<w`rnwMWv-@YAc!aTjPH?@y6gF(?J;oMK_FcYBVeWT4t?pqmH+wAI$hvUsIZlT@g{64FF6O5-XiLo@B3wVa00>n$LEx+yu4DmjW7UAr;JcPed=d1SZ=w|0HU~cH3BT-^HX#ct2+>{=Z6Qe!G6{xH<^x)TI+u91hca#+_=z>^-7ORIJm0wIy=iVw4N435*ZCl7b8hk$|8TgyTM_Ke_c4lAt0xKl_{(L{5#m&>Xd$wQaX0hPMD|#y+Gr--3^zmo5fg~X1w2=#GZRsbsl;OYA=dh{CZcVWYU_C42O1if&;azX-ABXy7#US@OF7<!G6(qZ-4Y)^Pgr>B2dN{YsN4@=+?wEDV3_cU{`En#q@cwXHPSuJw*9Kcn0BCF3Yc|+r%dw`OIX~9tJhs992%Iu{>)v{MIWW4+0aPkAbm}~D&@Y-LjnwhOq{LP<{_kJxi>PR<EIe1Oht&5;^COL9gJ<HyU~r8({S8Gd)){`1I%YWjvF#I}wj0Z+up+wk-N55>&Zz*}c<!Dt6PDZVSk%RVal0vx6dJ?h!F5UOAEKXX7BqYMOAq#I^+C=*cfP&$ADc{d#+rXdSt{FN_xS+Et!~}}5Y!&3-<1NqKY%uwHNO4&xObG482&U59Z7s<1=Ahf`#974tZt|GJLNaG-g0Vq*U`W_B^N%qpE@wu+j1>?(P=*)<>s=}l+oMSoA;tFmZYQKi9ahHj3<rrtyrm#Zg?`P^dDe><*5&kUB3ADY>_qVGFY_ilyt$o+v?@S5{7l}-IDy{+O5uXgmxWOC$_o$VY+{$A-JnksRWI>mmPayH~G@jjJutlwe69@cYb{V46u7DvDLv3?#Qh<&c*8P8eJ$V3H?^%XkT+P>xUaoMjK@Hb{bSCf3`N`7WU!Ht%$w+H>=9oyyDl;&z}na_g!Fe$hR)XVZHWkQ}JvpKYDjN&F&Ysr&3_vg$}M8ttx3#_d2D?Zn}fD7TZk&k~&?W<&U)IhW0C!9<R&(pXn3LS+zC097e8zhwe>e&}Rne-P0$Q`|NiziuCFHDx1I0!&4fU?W?-GZnF9pyyRFcY)8ohBUsmfLP^M7BwEkCc0n8LpA}To_*f<0Fy`S@ex|C1^8qilr)h{>rxm`R9XkEEhNfGHM!fk}RR(YCXoRj--8W7FsN&rcKAB%vtKn>E4M$Bqb8{K2MM$sX29;(VOS59|^*|4c^k*+GM~4YXX|$00ZR|<9=xNgi4g3b~Tp-*#Kv&bJ6~G^;J6BuJtE1t|T@cH`!`spyj-;QDK`q-ag>-(rI%xi=ZL=-QP$Ai^(uxYe8b2(%*6lrw7%^}TFLv*=qva+!(JyZP39N(V6X|@_LAxYNu8r=_Wg5#?)p|h_>2-ztkJO^8A5`8Co?R7KRTtp(NOe{s@|%KHm5R2{Kc0U~rUTsar=#ZMiu~%N|GI$PS*s~;rp=W#x2=Ijb)}6W>>=8McQJj;XkdO^)h-XBMCk5pxj9;c-ucxfaR$&Y`RQZS6i@udR6KOpCe`)+gg6y(uX%}TzuU97KjZ$&)QSj;u{AxyB@4`xYCt43DM47S1D-4He)0Eks`sUL_lP6o=p-7Wh{3P~m_=(y7+!5JLD)}x`M@tPH9)MBAD3#pk*nNa?&J$F{hGb<96jISV};Q7H{jiYe6U(JN2{cD34ir0U<ZIs?fU0s$y7M~VZIy3IWe-*<wAPGa=6Fpn;7EF*Uz`%1Dsu7ohCprGyj=ALV<9^20P3<?Z%{Pmd_nZXpL{>>?EvbVV)fE@JI=3!RZKh*9TnvjNXB8fo5Y!@4U$6?AcWo%97Q_Pi71$7~teXBPy8bb%xpG&+$f~h7&IMEW2wK3WuEpABD<2p{D;X9bVbd@#x9#Wuy)~__D9JKGNb+mwsCFsnnL;7aDRqc=pu_mshu`nvd_Nd}Gc-SQe9AMim7u?>b06(9z)eqWj`W_)8CN+HRenx28F9^2To=xEt7tC2_vEejeMo_(b*J?9U5~HU<|W4R5|9EzGzinyi*7t#)T;hP_01?{hV)r3t@aWx<Wz*acR{V`XP-;MCjKnziM4w6QKtb+$W6?d>Gl?Uk;uvZ!$Cvdn?|h5lam>usSsqi}!zJ1y_%-7&kNt$cUL<4gdT^gY49>h5dzYrq)^jC)1Kr|Mh~2U+iR=<GD3>q2W=WNQ)voQj?Y1__)s4LzS7HYqVyR1xA4F&P(^L+`NZaTUP&V_0|1O`dhV#}5AhAKqjF8mNz}43DQ%Z$H~GIH0T@yOACJ5?_eJU7YOO3^kHxzPih#&qTXy8A8FdI@RBYl1ofQZqB2WdqKy?d5a+{-jy)+heJ@w;{JUBKdQIiWdmdvu(d`mOC2oHzNO#)0zK%uC#8pWpiV~vl`hyGO~BQr(QCrIPT4DaysOMiY1m>m#c^nw?dgJ7Keb<{eQ^|po^7VR*7~7T4o@P4MZX5b_wTy*JW~;|fH+mCU$FbppLf#%H*t4aK2n)b|2!Gm@_u=UFs+aP9wt_9)#q$BGt@DH-fXCI(O$3Ox`ZTL9M9HR#!o-bzAx+i>b*Jk;5fmk`hFrG{|4|+TblL$b>Dst{rhs^wEE$;{?5I2pZOErC6BF14#6HZT>V9&g)BbsGW@jOvU|pp_~!tVO^)XPoj1!03z5gX(Z}6s6O&(mR6lpjr=^!;0+ziQd-(&c2{)q7DYD}><CVA}6*>JaU=<tQYm59A_3HCu^&~mdM@;jNzq2ZLu@&xgc`|H)=xed-jf6a|=@<fQ=%+Ef)z+V(`s|)co4fa7xBZ!#lGYd%JkZep<LF$U*JHvs{I?h~VVWtam_l;ssB%d1q9chyNTI{u{(OJW+osX3z4vop*G#iJ`<uggWn&40`iUc$=gw^jEQXtE<2mci@yajks`J~~V8##IZXKN~cu2Pc<yk##$%R9{bMig}9ZaGgZB$z3NXEP>;HN`$ha?<VnD3$ABIJ?WGqq*lHus1Rl0=)os#QBb?+#61^RAgcu2AfY04Xag_?}j@1tQqwc?jOz*oaPFL*F@UeehDe>-YG=Ru53ZHg1#?hmnCfExj{5w}zv&CDLcJg;jKVIUJSo4tXxGxAw-l-H+COKc7|rB=47DexcK+s##<zRSD@i(h;MJerdg3#|IF!Y(AxDPpU*ypnivY-}M>Hb4<TV6KA;US~w29=E3vL9q-JvRxhWbIeJ?zH5si1%spdoJ+6R;)|=D#Rxd7hHrc^1A=LZNO>ekwzM$jS{ZU<yhfM3zXrH*HcJjD$(Qa}Jp$9Pbet?ifipkLpslQ(pgWYAb=9Jqy^z~40NLQsO*XCrC^1%WNy~qFq<9BO5l`cs%l*RCzG}^?SKhL(;n?&E|JUDAW>r18E@(YL`I%@S&pDgCC#XrA<5^P_;_i{0D%E7dFYNcAuxj)Cm^UZa-u8c(_BdUqFTsLN)dPg$d&w==~;7HU)+j-(NGb(+Pxz;EVZWNt;R9zpdoe6M0%DXr^D&Al^)B`@<90&Xh5u=N4Cj4Gci51oDl&684^~BY$5B}CNb*L}s)B@$Md=tgVQ}h4gT#V01vnDrcT62`DVg>f4VL)l_F)5g~MuN(4Gddjw2;%S`IQ;6P?%UI>{FLQkrRPF{RYoLD$F<t|L!*P`bEk#shSj^tR1v#ncWvWNKqSjAC9FDsfWDo?G#%daYW&{0`q{iiAjNw^Y>q?v-gR2-q4NUQlDR>O$|=#i_50aG(i4yG6rWoe70|}e#+4}5+p5{-Or@;$byE+iMOI6%ZRKypm|@aDTh1TZqN>Lr`#%NjiM_*lKJ4XZ7yLk~i}O^+l7KL*A7CP~@4xZT{2i{lT>Y}iaJdS%z}l;jm8oF0${XD@%NF|GhYdNEU&W;|S&Neh{ASKq-XTOdh8yjBOZqSj^kpn1Xc|ISvf9Vu6@ArC{srh~vFUUcKGx#nTD^)iSB=<4!d|bTz9fL$JqAG50bOf=@!~Ip$o0-_U(TZ+8BK6;-)m&!dgrg0cHL`td;y8tF#&4Q_&Y1m=W$s(St^2l;rcq7sk?M-yQdKO;|rcVvNsmeiv8^^(h)a*Im1QoDZCCfI)SIZ&Y+#V+u{vK4(9yTOt#dbl8xPG3Z_$>d~DBz0D5Kp=e95UH$QLJ);ry52O?wIk0)oBnW(nx#hnb)w3b6_tY`rpG$<8du?_?9#58OR-{EEb)RAH%1V_{Xxg&tPv)!^r)H1by|6Mh`DF}T~Y>IEO8*MK;rFiu)1-(`|@=U_-JGOnFAZ!?SUSs51Sg14BFG+9S!^N?DZUm=RI(R=AbMadhsQ(;((xq)*NMPfERB-hN{R^1^O8MQF>_>i3Y_XXfpC~ylq#|9k-*0KN4p5>TJ>rXV#nx40wnvuGp74DXeBHamQd^yW9BmsoeE>!QCp}K-VCpW_r!-RQC61q9X$m`E*!%6nAF^6i%gqV6q4Mep4(;TMt@^mqOWU8nS4!V)_`?7hrzuxRtRs&WH3Nvn&OWeXpMtSlL)n+pzcRFLYfRmv&2QbW80#AcNLNbyk=|uLqcd?18SQgX9@dknbvhqWxHpt41brf{uG0&Y5w7X*lN#bvr?On+6~e`H-=K^&X!C2Pzx45V8qDua?Ff3JhbGzcrl+wp4ySu}voZH3d{;UJZ@y=($wbbV=p{pE<to~LhF-6_9{{?ztM0$j)3P=lAlEB*978Wt^>)B@Uj*{E=o$EmWnilIQ}Oz$&iZjY9sdsbxA=iIyWBlO_BgV(JN64ZEVkg9=D~-l{Dh5J%$0|Xv)t}!&mSFE^Jc3F<?WN%xlqB@ue-IhfvnZb0~F$WvLyENr7n~mVZF~HrvLhSo4DJ`ytvh@cZy^vAiGLE+y-ENtStjpNkRXdSvHrnW(vHST~2>JhKRYzcKYHV+8H3&+q*mAsFG8lGA1lhdt}j7we0FX#zT5j-^{w~pcQ$ApfxVt>Qeh8YXuI~mc-GVzZ=K|-R~FR!n5r-#qNuj^x!u%G01`)c8-i-a!t$DdmML>S&wUr+^ZwslWImsFY!jNlFQ6K`+ap*l4%ROQOvuYS1(fk$C_VckgWSexu{i6^4W=YTW871=W$9C9lO(g9X45Ys%G;+L-e*qt+h+-0Hw}E2c}6|B#RW_s`dKboZmWYyLW;oQFYv_oxyF_dt(;pG?{1d@G3b8?*?>w-skBbf(@GHb~6QgL-F?KA8a1*d^8@y#AzU#tCzIXFBSUP`@GeFdCQYihmhaDwAK&yuyW^Wn+tjZ(OUn-RYrWJ-1zJ9=OOgn+ML|AS32Ro^84ZwOnF^DrgP`mZ?Aq*=f{3KCj=xbEfrlH$Cx4*)%7D_!QC;C*oj%IOC^5Qhg7#8-hPb%Rws!i5?m!UaArcrfWPtD)yMURcSsT#V!L6-GNXH{yib%u($u+AI$GzE>s0+Ut(-PvfT+!0!~DVI+2Wg3S_vlHuLp8IE=ZxVHJrZ*S69E@<W`<=qB=JZqt|?&?K8rTR`Ff2e<f#*9lx(Ln(O4UHKnh~dY)T(npLeO;urL3-_}sAO?KXwpxc}^Hiu{(G)Vir;3xjc6CvAN{PkL_c4&r|Q{xU`m}=7F7hP_pIvrA&-Bg=#WoaI*L;4$hA0nM^>=)ZUI3@4zd+YT%=Cr^TGg`g<&YS6dja0>c)HK^)7v8m#H8(pu_f9UHocSpbzRc_dk4>HB2X~sP9sb>a`d5WuovzmXM0Q@|k#W=Cq3nHqucoM~^l*bphn%9j-64(h#pf>@jyim&a!gz~Eh^aKytBIO#5eEgKe_P4Xr(z|YCNG@>-P76i+^AnGT;C81`qqwu0HDSK}&naGs~SjM=D04^+s*%=^OXWQNQEgnc|00tM1-$PU*gt%X_;Jg2AuiHO6yta^f`@PedG6%chBsI*oOf^XD6vYzER0ZkoSnNVTa4R9y<?dDw72Q}8o+c=^@bCu`t)M!w=!-2LwPW&3%N=mlk0+aABLQO{(%;EVrUHiJ{>LzsBn2SI2)Hcl#Tn=ki%3g=gl1N0x1sP0CbT^_63j91T*cM;O)aqqn4Wenk)%ken|Ui{_8%!C~XS5;|BR0jq16iBQ^M>gvp%DTWGcrA`|&^9xbb-8VuVvv>*{Ea*B({`0wtrKF>Slr<kTz&pf>|VWo#_sw~(cG_-sJqel39wi5heOX$*nH-Zmv0sm=Jw#v+q-wJdNH=*%{=i#Cjx7m#lvf{F}Tu;_X)qoy&4OTzv_AvEGI*wl7tJ9UZYM;+Aj}oS}f{+8$wf=2JC2c47lLgt3R#Yt#4(7qMf)a4C{lk+`B}0-QODb>jJ6RGJ5FBls9R$Rt7zXmvhdh#EV(3JIIxdP+&7)ZQ;g5*kPZI!#}Axz}u5gGelGEH1Kks;dT#ee0HsYf9gi3j2%?5a?wyMdxY+2&{lvauTQ_ht?ohl6!w(QyxcXVM%CV3Z$hqcQlgpuavR@;T6!UaSMM%_jo0W4+#vzH=sP*V#38hLjxyrnJ`V06eY%@zbvx6;Hay1%-6#euetbc>E;L`smHBD_Eo(qFznHc5X=lZuc$3SzrF7yR;=NDDN1K-yMtLbBus-1*ZG*P4Ey8GihF+~y84{BJFb_aUS!v4Oe&#jpUdGt$MnCVX&rZBTgAlr>Omv0$S}%@aJ6n3gz2QkSY1T_$!wX%`PiAnSfB4X4FMpq})UPHr{6o-eo<j5Y4PWE(v^GTW_9<`9Xrk`IVWrEL(xd-awy~)nr}Mcp?8H0!J346g<h|%#&HEujK6Ny%;Bh(}kdIyxAK&*wpS#ngq+B}mYsb|ep>S!(6^`8#DZr{{;`G`>wR5dcZ)($XEmq%SD_gJkZE7>M198j*hQ}<H=KlO@_ce~3Nlqyky(z+j)}Iv#KV%L^00jS054k_02#(S8d^lZ^DviUN+k1}Ahp=F}IjDE?()LlwWo}(#QnE$&>;4#TsO5CULP$MDdCsroZud`NiPSp2`+G!t`nT$F=Mo-C>XaJ*&sWx<<RyRjvZ-(@YJe&j*&^YgVA5t5pSoI$S2cMAbMNr+A#RoXI(z)BYvkm+-)h;QJ9X+<MK}xhE*M#JqF`D{AQh{4K8C{X-d=ekb$YZC>%-f0yMwgp$z?!+A?xe1&^yDJRSVh?)aq_utH0{uxE4MKWsWeCjx4gu&<M^Lw8G*GmO4f@tjI?~nY=f{=^pKlK%r^H+;n?a2ztCKj{cuix~CUnR1-bj&aOC;JrIH3Yo@+aMwP*5R(0d@jz3CgTX|*lF*!Z;@A_yqGNxq6Yi7iobEEFjU%2~08WZX&HPh)<ngnx7(N2;w>yk@+J4VC<fjEQHC-ct@Y`7f7RGe$UR(V{JoxaHjGBR23=br&5Ruo8W4gw|UtOudntDGl(?{aQrVQ(4Lq{Skkc;zEnJHg&gZf~y<v~TtG&^Qn{*&>~5<9TYnwcjrtv2@YjEK|k%tVhObwPY?ERd9a?24&#IqWbuD2AZV&#oTde38UpqdM|%+WVLra?hWxOq-WcW3c+=uH=|>WOLq7R0)>}UMZ*EZW4$3hq(QSXyx|pnaK{Q~pevAwPMl!WLIXOi@3s=i>96H;TpYWI)rroR_a}7vy;aX60q#}V6RI5yq4Q37b}#d|@fHPfVT_y|-_88n!~i+I!}+yWhdqehCFAz1ypmk+&nqn4ksERAy{JdEl0L%QZav13alCr*{8;-4fsYQT-yzAh&qXMN<Mlvy<XV1y&Zvr{JUPU*FTN?v9@2Qcv+%{Z|2`jd-p8KD1^k=#t<j}+YI+rWylJVf)iaOBW%?6(hP2CJ_t0D89IYPi8dCh_Q}_k{vhpF#w%2hKT}pD%`BqzgLRX*Ghh`_vy;wZ|q^G3!c&rl+vs&s!-i;!Al3t;iQ3oD@mdhUoiB|F)i*NTw3dcUc)syLfLkE7vy!Vtlpl<2`SE=MXY#@Ep{aK9KDu&^WjPJUG>HJCG{m}4|NqvgZm&+3_5bLjTr=B$26pyv+o~7=Qx)Q3W`zEuQPTrH>NZX=l3k=gnZ#+I&_@~xy11j_WsrJntPToWzlk15q3fH}b2G1Ufl(T${D3rCIyRC7wX(Qh#P3Wmxdt8jOR6}F1-ztkyF!zknF-T@&Z--s=G-IcdPL+V#81hOn$CmE6l32zo53i`YN`Wdj(QCU%ro<7;oXtv<4HjNU+)l?7)L&yeIOzSk*&CYOPO{jpmS{;I;~j^0!foZOdkwGTe&sqIEvp*^xlAY|lhs4PY(LI{V8@FE*o-cj=|7<8xOr`y`@fT9akoh&pYesj34gYnT>HoMOl?Z1yH<4wI_d6)$J;g#8(iby?+rk-+x?kkzM1RBjTbFcuXZ~2Q;oQI)CD-QBjt~;sxSAo2A0vk6nR-zwM4c{S#0f160b2moSlr}90Fd~uv^R|J?eMnQaqG-{&x9DhjQLLm_XKRY`@YF2u$`7ml!_@E5`f8QM1+KR)$YLn*#}t(_8r+9-NAT*cnllU9Y_B7wMFe9e!H_Um;xkb&A^=cunIGy3G_rm$g<xCw-t>a62>pjf!GthPPqkvD`j7HEsLQ8Xx*}ekTzMF%DJk`cbM8+2?jw#qP_WVbhy1#aa(}D82$38(a;JzGm8_oDS4#H$ba}(Ha<yOSSzh8?x}4Ru?T;evTHGlpO%id9In+s2-nE>(`v66iZhihp99r!E!SCP%=TQb!)_acjNcR54~}u!2?Xso==rOZB<5WtoU;*ZeP{;Zs$Z4yc5aFI{FgJ!^jRpkBkMmN9~-t_P&+D72a5z`5F?zW32tS-q=3HeUgXsOnZRwfNZ?DQ$Kxpal1n<D{iws^;?}_Dm0>re@w`$U|2tq%TDYl@p|C;WS2^1dnCdNo?lV8skPe6MH)59`mM@d<Ld26uj6W;3ET}}bANfW{x;C23CH|cCE=bU&QiE9Z(C`V>`tQ5Y3LQ&Qsv(#?aMT8-|v&MV!WrguT$NqH<yja<mo@0%CoFBS9cibB31p5NR2w%ygNq^cJ$+%vBplQ?Y%j)^=hv??Y&lGCG^)l=~<>)xbb$Rv*}OhX^Z^+!C4zGxc0w)yHA#%33`Um`}T9Kiqt(Via}4wX}x7^A(g#+%o7iLBH3;A{tT_m*U`O43ZoMf@z=_XfuDF3M$0)ha==@tyeGJ4N`Q2zu~)pV^9QJ4#Hhak+LWnD&b@tI`oiy1UzPBPN!xwIkVg6UcDnp+eu&>-$LeXY$16FWt*ZCM_aY1Yi0iX*H8Zu<AKe`XOYuFeFgP8lEQ;)(@{7$CQFP&2mECR)G5hi~F5I?1hEz@{9uM@@Zrk?I=x(qZUP6-=^nHTVdD>rG1A&?P)2&Wz7pru&9EO~~yfb6-+gk;PRpdD`t)po<yj6z#(UjXyU)1h*c^b_b;mIvzwhw-l*R$1vTIE}KLyLOtg@k>Njko1?=VN^9>2F8QU)n~A*21@9uP=Tpn7HP_y3soof%%&K8Sn;TqNMxn&bd(-x3eXDnFx_Kp|IAq%RMxRArZ7%Nn~`=Ke-C%O7G?lJBrWl+{K`efh=Qm@C#g^uOBBixdDQCwrb2R<<gJZb{Fb9T!(Bv>lZrNv*ycmfU1VE_!j+M^6KVu2cVq{#%x&Zo%T-P8jlMw7~bbPU+Cw;SwMOVcgOY&uM^;CExpx=Jj9XD;o!c8-TCp6ZlC#CaM+G3?jz2a@)c>IZI8(9Lo|>0zQHvw&l3~^q2p>kI{bbE$*H!)&Zd>QXEqk}a`a~g>$~CSA><Ed{4?d!t~^KIaocFD?wnxA)pKn@m#fi?nERsK<p}+KULel?sb8*zB{>gqTaGahEr#!`R~p%buxzYLUoxl3LiY+-eTQPUjR~4PegZj!Re9{Xaeq}uS~j=+tH*AI|8~=6290%n`5mF&DE`!!x@12;({6x}QtF;M6Z~xv=Rk+Zn)vChQgqazpOyO1dt4(<?&FJIxSsy>w<DjPpxIaQ+6>#JTNkK*cj+IEOohDOjPynuY#{b_I-Tdj?KV;zyv1So$t;Ap-KB<NO&5SAXK}YGicN+9Zd^QGl*iQ3R#3|6FKQn&Pd6DBnv=<AF_nbxq(wQ=Ih_}&MY%|}dgQM3Fi+o&NAw^nEBf@V6<wRlcV$Ks5!HQX9rU1`-!c3Lb?E{HKO!`nR4LyKm*|zIR*`bn4_2WElieKYMgHTv&ysGpLtrZ0I-S>45BVLJT=Sk?yEyR&G1Jq994FHvsX&KxLq;#z?ExF&Rz1<_vNiR{nRCl0qy3*tXbZVh_ca!0RciaOy*IwZt3jx)rmoJ6?N1@DKy-Y@9LbeUWBZwR_wP2R!{ZNcX`fNQ?F+?Hm{|$-*RtN@cl)?(&6V;QGie8XEgl4%xg$Ut4_AnF7KiQ0yhRMR7k)wOE@2S!a&l!`E8ymj?NODiydTyP=jT`J^STdRbZu{3G-xu_Ps{C(uKfKyr(5^%WVDZQ)LQvj%^Dub)m^M_fgcZePQs`K9T5-sc{+f)6*li^7_{Vi4VOU_T)}@ZJNC9zbVz(q6TU>yq70a=4U1LBKStcSkf!p`NYG8RBJk;xaJ`qznoezYiT_$IG~SNH75oQ!EmpR?ogU{ud+j1oRldoFfbWFL6MQ`ab4yF=g&3cc#&ihJ$Dp$m;W)oFf0VW{ec)3Xms>AdCZB9<Jb}kBjOeY`EYD6+q@;LErX}87<hR~x5IPIx!@lSj-A0D_BOj8RBWkpIZI6hpC$g<%Z=%0-NT9RlX3$Pyocpr%9Q>j-@_gyz&X4}{z}x5#R8w9H0)o&^(f6k7*r-=f%8=67;NhH$*Iw#*@NUd<$*|SS-NjZNX03bLxKzRzG}<cxe5-{8jmZpErwYJ4=fQ8YuQw<8p-Ye2o2FI!3dp&VC>U){Bz{Oplwm49_?M15SzdLoL}lNC#_?fsaD=l|c^^V2=@WC-?~#cWVANwX;@D!fASV#@_N_&)6HXA@HO@xlFIm_B((4fIu7V%#gW*>l_Nx6W@H&0?rLNI)X3~&-7}JB3csugi=3R~PhN{1g^TYI=Cx(P?dMS4EgkM(Q4z(v<fLf0UzSq85^_Je_{5?ATy>An)<X@%&?7o)E;9Y{JEp9K%O41Rk^hF;uqD4nJUxj&-CZ1441?y4Mbk^23*2uJ8Sw0`b1E1F=_zmwGJ82v?=EX&H$wl^S9rJxN=nyz~9G$oIV2+&rhG6k6)6Vz%)%fg~=nzhGtp-Y*LtEWmKafnL$a0Zg+Lxg!cLV<+<5aQsCOTSAYV{$+TD0?ftVB3tDOnwPeiO9=sy|5no^`qxAUPT4Y&$vRSmf7%o6?t}la`DIGTqSu(D>Q?+0-|}F2zO7`g(9|IR%6DXSl~oRc5VEDy{wJ+K_>1LHNm%pWqX+_<mj(EZl(3E?4T?gB?@C=4S1HOCL~&$~1jcS6Hb;>{`D1H`qD90i@PApZsG-eM(bh90#xBYcwh$;6%`)f>4mn_U)cu?C$#$n2n&exjQ|3mqEpXRXC%@GH9waLQKts<xQfji>&@>kz8#1^akxx6EUHXx4)T`kEf)z`qloqY9MH54WKu3-K&YH<~5fm-(1ArdI%n2`x$gUma+uW5rn_I^N<Aege5*T<-pO)Y{g7v(X7k3ZZ6^KkLXN%ORtP)fN{T0ox)fCo_Xf_Sf`mwz6NnC?{xif(gr;Iocg;RyZl20=19>BE5iyjFREX*bHUD$u<vcb8wEl8x-(SHqduu0ep~mbX^c-AnmKfa2l?Su_`PuqHdTS3%qKmXC;ZJQAGPH*DaJOb0c`r0UZ;z$SrZpXwLW|0$EcUq$((EK$R#3GG(iPk7K^uEjN!DwHLL`4d=Wp7YLGl^zTeJvu3;89r*DJCxfwEhz7Cz*&1s$O?WU;ov^=21b$c^*WoY`?$m$SKZpn>MPgJcc<HdsZ3b?9`QX>u4Dc%YujN0hzcJ|7I;j>ko`|Di^3C!`5mcWO+{kB12Rl0~7nf~^BW*>{>8TT$LsEyaE`kvl-UDbSb>j(6?@3<aRR++3fzR;{uYtVDIIw2hyq@!Si)P0|4#k-R$323J-nQGlnu<8y7dNU|@#geEU*MXzK<}U?D7<E)f6uM2T`MuE_oeGCoE$icY--8~KaKLVG&>i(WPob2-x+!*Eyz-5g)te6M2XY8Z7O|W}6MEP<@tp%u7n>PrK<{Q#WncT<?z@Kq_OQ6u?pjN!I37Ki2KKgdID{9X&!yO*vKzj^cSn7jo6AQXKBSF-^qq6J(%a6^vl?{XmbU536(-WXZdZ{U@r!~>j;UG!wVk^QXQ0q6z*L8f8goXgD9R6|UaZr1+n^J@)VwXwSS@e7;n_D}_3RMKeJ^edKecs5tmI+GSuv5}HU@JO+a1_<z(@HFQh_U{rTp;RmGd@43CL*s5Dy9ae7rfM@e|gUnvgu#7F@mh))sXCw#VV+s$A7sJQj~w8}@*6Sbd9YZ@51NVC~Tz`mT;LB(zUx{$6RJI}>l7d&Ni_K;s|0!w0>0ooqX=GLbMSyU0Eg(V8tn>UN?p?)#~`nLW0I)c9$DJKfmovr*0`yTC&&T%@Al3LbXJ(W>6-EgUpiW6HW<hCs9LX8hGX!0FZ7D4lI_&RfH@Qq2{=YaY8;-yVM7+GF}j=%p!T@K2bt;_mBF+G+6_+0yOwyEI)30D#=>`kn5ncKdk5THAi-PXEJ5uloLZf6<?TIIS#52ArFqyRO*_*2wRmSY?4vKC4kvTU+C|caJ<O0?^!%pYy!2nc)fj`~yJ$7aZVtUYrh2rk3Y@>Kw$ZGH;D{SRD%n40b$-1XXkXoOKi{fUjq4cyAu|-=#F&W$mF%92*qX`t7kxDI@(&C(TFC(dyEU-=@Ze>PEnA>51&$0oZQwmjW$M+cW7JTW>~S4Bo5N<X(yoMIXyfat)g0)h_;mx_w`vYlSxf9ZH;K3BQ-7U}BdQcjk`X&gj`n(S}kFFCg<(s8h+OuJ)NONBrPB1RAfuY=go(`nVoc2FS8m)s7@sSGYE@D)B<$!phc)2Fv1Q-5d+wI|f+N-mbObw!obHG`IJ|;Y6ax*2im@aecpj6_c9YqJ5~b>RX5Y82B}-t6QCG9!(hmpWI?~`7L)lt2kY4zDMUII96{A>X+fS@g>-P(3oAKnvKgF5#^+ZY8<HTNGhXSA)TA>njTqz?WS$)@GX~@6oA1;_dM_>wTuhzdj@?R4uK%rt?jdYZ3ZBHe8<sKyA-lGk0mBu=fWSj58X{ORqBoQD^4Ek{(OC37GgHt1Z_~iQ>|vI-(v2)50*l*XRs1)Jx;fkDE&0{0g<b6?byqkKq%v?7FIYkZOpN5bs*cjm{`PbMD4H1gA02J!X(3B(f{fT<>Xc0`=SMx-7y1zk6t8NOMZK6^Y0KaFJSQ`qyb&qD;)rM%sHhvc3wC8?X{%E&K_v;WjVT>{jc2?-CyV5eR<b!Jd#CQ8pKm-ldnJ7rV}=~65n^hWO|`yTz_f5=)-_8{r$GL6yT}YwCPsv%oR2gZu7kLJ1aU*$p0T3BlEi~bNY`Fy|l=-wu#`&&A#){HiyDV;|79ivnr0HaIYd`-WBX|5oX-L?pEe3D6b}Pq-ExB0eR=Y+%H3>lbq&%i0%eFR-4`L%yfj5NDxVqCfk*<(Fe`vPsOWrd<M4OdJq`4<yYg?Li==0?yxgVGj80~=INw}?j|i8#{<W?H3RLq2^mTp^cv5;jYsPX+Fg0Q`E~Wn@is5Jausfm#&IH*o6#8X6ynxiGZxI&_Yda<RuAok!Vsf&-uf%+%l=#)et7oH=&=oSj|P}DH}+D}Ok_|RT4&qqQHnL_G6R3W<<rSyG=WE9NEU#uzq*u<n=y_mM&yitt_EPk?e=wRFK&T+4&>8A8;I<)-?fqsezo4(40uMl!RP~*vevHu5E527joeln>8aoPXi&*}{0DM(&TKOjwMk-cNIG6~U&L(eBVc`59Vvm-iRPY9$HON96-o=?)9&0(!qax4NbYHl@MLSf(~vsd-1T1Sn6XHux@9`BJs97Z?2K;6W~`DnQ2=UuZUQ?qtDpV1VA>z78x5PDoZ*X4yI<En4e{d^b!mo=C-ui3st@zX;&#yedIs|!9HF}|M%0iKV*a*asdSw6nyYvFmpXZpzY8mEP!;$g^31@>gPmSu4?=TSuKnUt>nfe^T(Co}d>Ym8Op5zo)U8~1(yvc(Ky{^$rt&hs6W{fYA{U$Hoh9!Sdg=ztX1Mem+Ud$t-WQS;7Jey97{8g#S2pNfN3qt-g{p<U00YL{Rp0%M=Jp6XjVKND)iCC%@8{Tprq)`PUY9<}EWQXm4S^2qST*OqUia+oqb+Slg`$agKY^#4Gg#5J%lq1*vUL63slNdwR8P|<6x2-oE@M9Ww}$9)5XZb)=%|e=qmSC%;IDzd(4`r<n-7`ps5bXk>gO=&Ye=Ljc7J$lI4axquG&}Et4w)N59X#qfVY~b1FNGi_CJ!E#cP0q)6>*k1Ubn3I|R&0CY!32=(ia*(*9l3m41E|8ild`bPV*8ulNYQE3cT(5=X0vz}S*6gO>pHvi;eZH?fNQU}GDeRccn270yGnZ03HeClWJz+!PAAGD>bX3#w`a__ot6m_iHB%`}@2>Dn50hDKJ&*zwJ3PrLRp4UGglLuX<)zwTELTcz>&F|Ht=r!$^2-jPUdFMGc4y$9k)m3?4)t;_RnZPP>`_CY^;J~Vj-07PE<>sSUZNR-S!Fuvd(M7oxwXMM2KlOb_>7~R?#@2so$GhB~WmAL!(OvcaE&g>1k+gE-F7qAEO<5a|7%WOWS8{sU$hR~;mKe3up;98Xwjy2}{6ZW-R?bDpW_>J$@p3`YB1&wr2!C0Wj&o|4>Rd$|RI|a4D#k6b31!v-&qXhiWc2W+H2OfTa%MPDb2Fr#3t-JB-arm5$hv_H>XCu?>6D?lwVe0p%a3g5-0M>D&);=6EX_S+oIR1!aGL%Vm^gVMJoA$q3_Ha?|<jy!<<MvNkPXSrfs1Ne%Ud$Wx`2(6G|8(eH93t{}`;5qSelG|$77glP?kdBrJMu5#@dn&K>`dmdW`(^o0%D-t6Q+yS&A_e)8L~foh1)$(h-%MSY^}^0(ZW~1i6z|06Qa>=Pd$WIbqL;t-X6pyRz6j6tvKqZ_4RiwN12K*F8C1cxc2loIee#FU~p-F937jjb#~m&J(Svx7xUKD+3gy{v3Q+cs~GAIqt6PzwpP*N6Blivr}U2yVINO{-l9VG`5k8Ia-*wrGlujx3N2<4+vIZ>*S!v0Irp^2OY#NpJ^tM&MZ0-M_8{M<(eBSE|KPp&i@_jNo#yxChP>f^<}N-OU@fWPinT^{2U4|SQSysH|8pc8&ttl<58!N8H2xgtwCcl)m7(%|k=XV@>E7y}%e6%ZCD5AxAqLswyt%3T^|1bf>bD$OIp%GD*(Rdbp|hNTh_&SI3UEyWbwtqifR;uZ%p#+GuW}`acyifE$0gr9tAOn_o3+k{7qQ}oFi%nT66H+Z)OIi_5CiPS+wv8egi(mAT@v&^^O_~SpMULkfZF}t$X!T2(P0lUbLY#)3b;3637&no#jn~eHDC^)t`>Ckm9-7N;9vxplTY=7H~#|f)#s*HG2)QV&B&&-)+`g*i&e3;Q5%iFq9%wpn)9t)Q9sJbiW9fE+X1GV$RNDma};u~7DT^Wm=xE9j2riKx3gr|6()YQd;N_(MLfT~sdrAwc@rq{^G{gNw4SySU+a{8bW;%;VqsJKj-#&Im}3`vvB4dxhrH5BkOPtaeP?W#I8poSHSCS|R}opoFrx#v)~nq?y1R3LK~M0StYm-oGuJsd37Ypy#4Nf>*c|U)@3Zx6xJ}s$j?@9ha8$RsY(2PEDD&l>+hY<1gw5x%B(%jxc-J{_KF#31VNmnwbu)6nI1&;@7=u?UTucn%^O^0v=czGxzFX?*wXa6<&i6HbnJ|}Y56fRS_vHT&J(uBmPgPau(nb;Pck?^^6UkI`3{4v7mt3mZ)~q|a9T$7FA!DpkJ<S}bdgbUd#|)3?>(U;VHp86FCdIAr#3wEOQN2v|kH}#LWI22nRlh$zKf<ioz~hbmYF|i5VN0&L_GdCe?>G_o6U_PF`jh^?H{|{}gD<1Ywo`iy$)5PAPL}9cLmRF7vQv{+TKm~ptZIEV+uMC%4<D81>oCxI=Y?xautR<IJa4O@F_1%CwC7<5c~v20&&+)baCk}BVeEO;mfS`qsd1g{sQ>=_#^<72c_E{IU94RzY&GFmCsufYAp>dPqjmINV`=)*llm5EnnhFE#(`UXT`ZxRWqm5NSSgOHCl3{bPR`Dgk6sg+Ch_a;F7?)<^2iFHyV;+osMUOWDLfPPgMbs%*8sqw{4G$jte{_QLy&H_a@+@9#xS1Lc0YKT(hH=|r*VE6ya$<4Z#N%%bc^$1em*tQ+VU4~c}YF_VfNep%;t`|U5n}yzSSEoZQKe=c~sx2hx;Mk4~_kz)g?l8dG%V%V~$hr`a*AL@2~zFfvq)NMTS4qd`>0OHb(5e+f*xGk37_8X`74KVf-5!Iwn^{LaaP`tAyceV>(-;#PhWm_pax5ofTE%R)Jgs&YyuiY8`FZ*XpJ#-_S<6EGWl2smhHAj!GYq-m5+T2)4(!GalFrfL);{Qo=2K862;tpT3yb@$|fw)?uBK>OVIb41&vDfGK`6TFa<F$~<NL_yPX!g1Py`6x1hIsaMSo3bzEVnXT9JBJ|7OQ+uY*4})I#yHh(km(lTlzAjhi+fVJ;VOTtthY?caZ*lP&|CQEoVEv?(z6unNx)NjPc(3puQ;;w0%%F$GpzWTP&&v9;J!+lo8X!%7a2}udyqM(IrrA}m3ERZxtHQZx_qSY;(h)#aL57=AN!HMS$Xc(n23_3sOU*8&0g>D9sLJ>L1a0N1pW#tzSu&a`D@I~?V#3ht%@p?3)9mwhh>m@07nbZFitKmk5UWuaiI2v`!gfczi)v9lt?U$%${$^PCLDya)1;-~gwpkN1HCOP=&>~=7oL=%OWWClT^OY!kF&BjH2edp9hjv!JzRV2qo>2nMb!_@WD+*U`t>>)&cVq4mem^uy~tK2-c&geYpB}Tlmy)v;bek%Bs6zXv~?P(0Tx~lO~#o3*n9OFKG8SU`P;zF6qxMh^ePJPkESoY#*KC<C%igbxRzat)O5ZP-=Bzxn%i-Ufu!`XL6$NjC$_QTcF|)_qi*f~rb1`a((PTiXC?uL4ksI?RC;G8J(ooPQwbJw9jojrbT&ip0yCUcMJr3T@_UPa#0kAyt1ds?ad}T@z|DFdJ;6;sT21ZCUjn_u;ZiQHqBICctLEX(Ni16&YTO?rRq%1M;HCGwUb*!OqP=fEw^SRx1-Dy2hOSd3zaX9TB&@LIhU(DleLT_Hg9^Ju&_+aXjO!iL9*iL@&=l+Gec$B+r>#`Z-uwNRlwvYiFD0k{R}c>~?nFGz`E~q?Me}94zoScQ+QO6CMof|e^X;%%w`{IH$oV|D`QG1q{>of?Cf@^+-b&Y3i>bp4RV9qpA9W|hV^kkTbp2wvcQWe4X$xAFF_%Y?vF5I4X``@0Al)gv0M``N;-+7S+^F$ncevEo7)oHh_^U+^rq(|W`<v$-M_QXlH&D57oj;)wzZQimera4zuKb#CMd)erqGk3-2^7F`^z$wJGYo{+B|Ka}mIGyFU+WaF(YXILx^=;H;(M}nAcz#-Pj)2qARjq;1$zX?8_T;aqy9yv)D!VYmA)T-2O&X<9`eFQnb%{r!y(KX`DD><zeDWbZC|5=s(dY+2_Bo_^9|3h>=fdAeztW{GlD@3S%Hny7jHPr$~c;abNuH%$LHiucBDjp&R=Kdm!UbObOke?`;eqGS4r|^pL!AgIgA_3s#L2}%bwr*<*MiS`#!vVlVSlBM6$Wu*(1ECwbgk?SNRIG0Qz7ikWXjBURLi>S4C^x1{$*F2;5Lay~y|NziyxXfIDmS6jkfU_OjEcb;9*PugZtlvSjC3hzlhycb2E+Xot4pRl(i@@x)rX8%tET*)G0U=b8>*#X@-xr2eos_v6!JxmNvWGyXjXoi=~hdj2$>_47T^YbCAwUrq@7EBWH5fP!D`7Am`c^X5JZukZYvX+2j<IPBvhG^w4F^x+Pwr={Fn32LyMW@B>g_~3mZ?p`m__A$@Do+p!TmkSXZD-pUq?a*Vp%`uf>3^`k%I?|qQ-@1K<^SOlVP$-)YtlbKt%in%(o!(Ms{Fvu!tAMp?y!b*tw(|}19n;Zoi<f=YZMrji&LnZQppo}VFgJt#H62`~^(stJzuh-l{2-w=+@e2yaL=cJwWj;M-ykTVLvDqJqxn_<;rWm}lKEDKM)gBoU5c;oQ9kIBv>=f>U4_Y-+I|`7B>YBoWaB)Rz)sPG)NBJcTg`$AARtkpnc-P0U;4x8v>nb>!DMyOTi!RK&l>(lxv94sFI~1vei?7IXL07>NRm{0PpXs4Aq}RDWg+}NyS@OtAh7jW>^2q=cKhGT%d11L%gpF~C1%vx$Ft@;XV#m(@!_t|+M%B8xw(05(zYKIm02C%!u{u_O?M}KaN5WEWVd=)A3M{so=v9^$IH{<Gf161F;CafICEbEy58FkU&1KIUKLJF6l^_{3~|~luJD-ZXRRZ9Is3{nohbFabu4E~0ToGY70J5qtl>p8{eE-f)gRy6_SG`Yu+t~go-lnxMcRkPAl%cx?^|<eS?5BY)1s?qAiG@hCOL@wwZ@R-SPVf);Ja5{GLL-Om{olK8awMchXzZrEjvxaXRwY|xl6C%X!@Z8R$2G#9O3$;wS2Xm9?n1wYAQx0wrK57J14(`&v(7yLOpIGy<<mQSAJk+PXq-W1}f-JD*oz?3su`hqUpp!($=CX``xmCIG*{UmEQ3DNr2JpXOmLltdA}<-p+fg4MyLV>9NJKyzp&7UZekQV>Jm|y*Hp--<U(LapCf_Tiyy$HD<TPK{!I)*=AHZznsIHLx8A02HU_T7ykOG>8teHyc>2obO(PT@{Q8@=&>JdP%vJ-LV3GMI>`}HRnKS+0d&ycOF7$oczgammoL(v-o4MF_c`3pj1TO4<0<4#mk07O8F|eVb}|q3vZjIEI`=xwIbDNDW`Ox!oS<W;(|4^en=hD<in2+k_4nhB6SHOn4btgL$uh<muLS7vsJeCPc<zMUgL&Q!zxnG$JLyV3_aDl|I6S~>fxuEw2$9x}Oux|b*s>Mh<|)pPSLMt|fv{h35qBtUOLNrgr!xNTa7WUyy7M2m{?cF*AYJ)B;VworlN*NyL!LHcw}<lLJ6tpL(&%Gf*sSGSWFZ7OHIF5u$G4|<$NiJ>e4OI*C6t7>ow+@GXAC~&3J&(!;&Er%-07aDy`|rOr%(z{B%~i3*Kbw360Zd(Y)q&#^1gam0Marx;brIPJ*4&GvUIwgKlvMs%jteo{(%S*>ES+q2wAHJpsx*Y(;n7`ug|O3!G^c%XfccT&i9pH2m2G%Fsi-7cVazR`tD*W8Z|cG8j~)^2WPk)W2aMc7;I@38G7rreD#7|;xV->2~KTyIM39}xOqds>+1+Kklr|rifEeiNN!{u(G{lULbU)etk5-8-E-Sx9XNR9R_7pk9#)Gv)4bt^!bcz2oW7h@cx*Djk;QIyBMz^fX`9*Gtp&b$<{_Z^Q{Po*{qt>hzd*{RYM|G*3X8<I{#iddo9UnVTwbsIL5#|dsie}{`YYJGi1erZj(P9iz3Q_06-yt58V;x(7VfdvD;-F_UfPiUJRj$h>Mz9&Y#o2bUy4fvUikwPZ3x`w{(cRstlf^Z!Q{&L6m#lP$bG%K+<xILUhL6dWpO$7JCBtVb_EDIkhAb`uKsOy8mo#1zNZ!q1X(E8pUtXdA3q-MWy^mnGE|3`(U0bd?f$JN`8~aLqfB3_Yh(dLA*nK3<m|{uO<c7a3d|$6p@~gv@_u~Ai&<mqyqGFD=j0XB+9mDtrBlxup<vhls^9L!&4^fEkCU%@j^djod%X#8qwSk^Ki2DqB;Whh<^2epMxc9c)HWCGD{1I!Sd@cCNXBoX6;uqPhc(*HNRTVd%R7#m)(Y&8w9{ZS(EJWp-p0{TYO_P4GCms0lVkQzzCRvw6!;=m%)NHl=MKTNAUnFOkh`*yO%Nt};`phYJe%EI_n$WMx^3KD?2E#i=H}}L#yflFFkQ$C@3!Fe`NgXz3xM{bkS{SzaZonnsda`b@(3Z;=jLA%hMY{AdCmZye!FR=xF_XRe~lWA{U6FQGj#TuT>GcC!u&_knKds5#$ougm}sPF#*(E$L{Ul6CJrcB+LR)dme2nF-?L6Q@cf?pzOHH>6FeAM8ugtleEM4E>ehUorTdshmQMD!B_{D-{8gJaR@~4_stz&lHAqp*Y8M^#SZXrpj#Ep9XrtNP%PzdmsI47+&g*r5ES76!W&F&dm-Mn4zrHIXMre11kZS0Fe=iz?7z{y=g+1S{ye<dnGlmx=-n(9zX_U<=Ssn7N<6AcdC-?ZFZWl$mzh8~SoPO{A;diz!FWt_`?&l<KcJ0lQSsTnpNanT8;c369!Sanc*|s+-_>t>v;@+l;U*-fsH~C``>%Ihz8&PH@k3;_@@#p)+c-CYIXkx}-0G*v>x?H`YIx(`)4ON*mFAox^u4%UjCKK&?)haRAn0%`p`hx1K{ivVZq~loXPXq+|gP4N!zN^l=O`_+_bc?naetoW7&)TSO)Z;!IuZHz!6isUArcWFe+vq_D?KZ3WU^t`jM(x@JnxA0RJyssfGploEXW^o^m!H0OmiwgF$?#s(!GqG81;8HI|E*N5ypMk~v2*;r!hyUP%UC-D7a<hzs35V{*WcBlRZXbq>f$o_qUr=iZ<QI_@ei_b6Krxn*fwWDsZ~w1+|91z`JZyEetq4cUvM*WFHQdXWs?4^^L94H<@6T-$>D)f!am~tbT)08#e|@{gy#qN+j=&;BtECjTnU3uO=!HpZjc%3@oEe3@*_CB*v_v0#rCe3i}w}CiB50Q`U&L(WnF#t<&rF?tnLQSKkyhfU#8(M$It#{S<ClYt<J5)gDcSYzZhD)Hc=_+8lg)KC-Xk|^M@cqZ8`ti6QQ0^o0)$kVeVUxlh5QWSmA`e{f?Ws^yJV?McJQONl&b*`_*xZ(()Zuioo7r!EjcMIWlZ?)|!1v1FOL{LS7wAc8lc<8ac$IF{zSg1tJkP9c$BUf&shWgx@!v>LL1hQy=!DdMbZu|HkIqfEBVkW%TFlXfW$W!#p0o*9}hZfZKNaNKapY_}uT)*5R+bA6NaZ4&LnG+*?D&PHA452$Q+vjVZPvEfLfB;!4w3^LsR+a5Paz=&BY+KZ{yZ04qCliQ>u9-8@iNYr}Z=h9ROi8V<H%offWw*C7=pq65Q6E7@ARhKIK+hx~<KW8A2_)Z0Ijy@{$gsKnX1?6mhA#~~YHZ|zO?y}bIhiUE8hH=|uWZYP7{e!xpabFKij(h|O)yVKpf*ls{vIFNQ1Rrob9fnBvYjU!lt$-!}7PXVZVK|UDLs)1Ru`7uO$1n{ce+qS6FGFwTzrMd2|dxTVHnp`r~IfCsf{a}C-TV%_+d$+gz9I*y8{k$<#xn484Hynt!=vy82zUAu0{~3y#&b;8AWHeCeLi*mq<)<{&A6b#ObFW^zJVsxz?Ll5RJGLp58hhix!fN_bIX|WgEjjGxmsQ)T&qfIK^m?<(3IJ!D@$>F{kpE{N|E^D6;mQ<8$;}pBK?BH|oEOtuZkDv5w9R1MK|AnT#d~EjJNda%G%n?Xv$Z^iZ&#zqg`7$5M^7%dq8GVe`Z)a?8R6tb9ErL7x`!)Yf1Y;8jBX86p!&v}=Z`3f=8t&7%v30u0;3a~wcd9VP5M|qp`oCo_h$goxHa3x5k0?ec67RRN0A?&GP9b6;Ix`ymlyb|n!9x?=acRScPH1$^kwhyUUnr``KP-QU&`Z%C2cd8W=9Kd{DeK=b+OZG1)<0DP31%{@;f1Vz;jZo9@148$A;+ioYi_M!2QM+a(mtQ>iK;^8)9^bt7F96{>J@veY$8KMsB{Hy&KYTk`H@i6`Sk5(_!d%^@M?U=LvS@D*McOnA}<zi_f{V_yQvaUPcT-y_3S^M{onCvr)ZS9oW7MyL9s&tQ2^R0<TT)cU#7flNZlR{y{fq{$Mu+*)a>+?Fkv~D{vdD6y?DAq3mVYVa)00IjX>sV*WyIwitkK5Jp@<C2#!9M@CP%j||dl<@s&_@BR4sdAtYc)sBe^0R_>njN2TOTb*dge!2X%{>A1ow*u($Hu<lH+OXSR(DLzvv;L{wY<z~yb$W|BGHA=hq$=&ZIx+$~{l?>U9Q|O%20G3yXfIGZVL3Ml6H{m90U>UUnjL*t=55%xxsH*Ph6_!dPn{`RP?Mv?&4h1i!yV1|wcbWOE}Y4x`MtDL|0unlK2nQE<b70*{)#^$AR0fh4VZlLf1j`k{}6-@)?r?29PSZ+`Men~`vY<Ic0P;xjM*vEThd&Ollh+#ZX4EaBJ8(2hp4_3_5V@Mzj?;=XXqrDOHeItFVecqi}Dkg`~ZW-KeJUf=yeM7YTq<{7^k6Ai)z@QGw??XX2Y9|135s!OpU`<jMoOettgK#I25n?YBH*Ji#1E+_%xRI==o`VNU~j3U$em>GC5;+F^!j-c*H#)%ugqN-D}OB>SQ^&D=l-m5!9aBX?JQhRe-jxG`{el`|WNvQL_6ab@e`9Ro61T$a0(LcQ9H{W4|KNFyREnpV~C4&7nx4aZV18;gTIPMa6zAF~x0~C$}fMU26Ne-yRQc)GvPiY&v~6>^l9JY|g_@b>1<@lg8EHg5NrrW%d)9a-h4CkM}=bn^z8J#>&!h=onXrdVKaeMZa6|kMG`V%}EiMNixvq#qzP2`p;svCq1pMH9KQykh|s+_S!}OR{yefNI3kX{2FFwJe-WS@GQK{mg)-NsOS0$s3Ev7rYyfR_Us}6NAGI(XVdKW{ZP9h+W7q~tUjO?v%7#KU^B_T;qkMX&jU@IFx+7L6e>Y&%6A0JG$H3nL7v56WwrRL+)8o`Z`14cLtNeuoBNSBw`=sX>1pA*s6MoI>Z?vOoV}PbKZHPAee<ZR>`5^|@!Ht?qv;K~wjaUbv0Zt2JL24lJhUy$p<XF^s)f3{acdb6=41BZG`fS$+<lWyCd>X}vfAxsL98|*k7<x)g-F<GKxWQW>ZW_-kYDbbmF$EHf}elYnK4PWOY#|cRb#zqcv$X$<+0;DM3;8hi4ZWry*VW>)YDo+RSOl{tffvT@IyV}kDY1b(WRx{4;c99+Xn+BK0(@a4?TWZ*xITMM0cExBfsYnQDVPe5V?DP2A3T}PU|4M%6WoWjZ8eRVV&+46n;quP#?|(t#{pL7&=wmsa%F&$?dpaZv^d>lb0v36!%!EsPq!)oD<vJYW23W8hG#rqwa)0iqqNPVf_gqyNQL&=jHk=`JT>l*n4(k|5h2>{$t((kGbbSTHfu4qhv@M6Zqz&$)&B@CnoP-2x^AlW2Lpztsm7Lpp}vi4Lk(Pn%KzqAN}2NH|$`p1uWxa?$KADzWZBdzQ5>5Jg#fPX49^5W;l_mc|(=iy+sAgcH9kUZJZQ>TfnRn4<)!-Jgbvsbl!jLZ+CeI;J()8>fcwV5+ofE>JPm5`BIG#Zv#&Ynk2(-89&y>tEFY(riqK?|DQY3Vobm8cs=emYUEPZ&IyC0+2bVbV&R%@<)+u4c+cKb7GYygj!{cn!TV>@%7XROrAMElKtqEE523W`^yW8&P4=u^gBqEb1AJHWvS4&7Z1%XEk44L>*}5`5yug#?7sR_>-|$YeK^_@h`ctG+0yB%nlX;lW#jp(xTiDnnSCvoc)yKQ31II(SCASZE<X(TT?kPnspND0Q)M=!%8)8-QGHyV;y<nbI;8F=Sly=Tj6@{wT-vnC!p>Hd}wc?j4khR0^C!pPB*wvS&*PUlG0$QoHjO|fOD4!(CVWA1^A39y>T+uvr(JM1@JB@Ut&GYZAdM$!I0DkTs5ys~SKEG?N`FRVPUi(RiRbl7v!eQLI4AYSeUDUu&5d67bmxp1xy!MLPVusyL%O{D)&DYc*T1sYZB3jJpeQmnRnEi2%9J-tH#q@!-eH?sLdtg_*tmdoK$oF!Ac7A`;VSOmdx9s5-a=Wu}H~B5JF_E4!;ASlH>2@gBWFbdhUA)cCiPx~&d<idZ+?x&0O#<D^w|3dEk(DU)PyLs%T+OR=6Yut?NYh_Q4ueiM$#tGCVnu(q0Zux!KhvCnn|C<P^A8J@K70scGyb+VZjaEguTZfN^w6b=HzI~i_j`+*_2tlstoj#c$|v?BwDk6PibRauuugL(m7XpM0t_kBE#ObN&Yax_a~D6=8shDw?_uM;e&ovq(VQMbA40ja6fU#q<?NE$(uoMn{A0Zn+c^fu4UP`y>gY8s^u`pvu8S5$&wryuG69r^vm;jGAF4Iajo<CXtaO|b%?F{Kw5_|TLc7x8!In%AlcTwK>ozrZ;ng8uDDf6el-2F@Yqu3{TMb0fs#Q!<%*%=BNXuI*OL$PM39Yiy=xi18^Ukq^p8O!qWMrr*kg#<<ytaF(gY;Ek-WBVsI%2#!#0<H9lhyy%Ew-P7A$)LGyt4P?-N*tKFeen6fx&o`GuD_?R-)x86BM*3yITHm9k4bH`te@7r>&^3QwX(OvSi!_Kt>{T2yXSeI@*QiziH*Kn}f|Xrx0NmUT+QQaPBFiz6tWZ{_)Twnr+SA_Uor2>;DoByw@1ft1As)(b|lp!^!QA--R(BC4_q6uZ^R4d(&jGQ{!-!y`txPuVeG{vBBXsH)~7{b$wxAMT|dl#qe-pv*hlK>#}K2@*o+nfv?U*j>>VrCCQ3LKJO#Fc^r-9{%S{59~t+3_c)(E|F&myy#%wp2&njqY7gYuyl8mgeM&qWH}=JPtGyXL_%&G;&+!+t#k+d{9-y@G{xI|FQ&@%WRN$=c1^jEc<!J&p?yr<-DHZHq-@-v7omKWb8+WGf`ffh`3udpqD;|r*;A9TgX{C<f{)hkT*ZDpVE@|h+;WN~p-2Bd@!Pwk!kf*(1p1QyOrT&*W)sYS44Q6nYv4LN4foxVi(9d3%U*p^WbTfKC0I8?De?-5=5?6PgPz;vT5<mWa0-EY}J-p=e1zhwboq9kPcm?MyIDt+Sm=9Vulzt8`5xUODijNy_yLNj)ku;rlg_QXPz=U%7pkDc%+N{@s!gO_NUT>I0y7=53-a!2=De-~rX%_0X|5TKAE?#nH8oQ>BjD7U4M)+$5-{}ChUM8FS=WyUyec*ob*<$;t0VML+&LVqwL|U7<w`JFj9K2tL#I0NIAhjTvIX9?s3@P9?(|4Q3%*$p{l@hhXcyBll^g22aD!xdY(jJ0NIq7WO$8BnJlb?$ehM~>8YCjBUY6BQ+FAb+SNuRW_3k?5Ugjj!+QD;E!^oy{ad<M%CM~?Jx31cVLFvP*U7t+QxRf4wo>_4+h8PI&|yRIoddK&FLtXFTZCkpK*Xn%MaWf4_G@A8487eI##jo(~r<EDyTI6-=uIkiY{O{&_bLn@>mSF6c%p(B)e6lN!#&#ze3D{I^&mfcs_yP{<`sH|@&o!s$VB|6#Q*s-wlTlP@fSq;5MjF3QUDS)&$e^!+aNSRGxxoqrLB%wEM18RM23@~ooM!x(hs|o9QeA>jiH>+P(;5Yjuwj@-&WIW@m$Mmeu_H%w8pFWq-X<P$AH12lUIXWhdaLRF;(``GO;5oVxW`m_gI#X+Swn;eqaSOZkLFx_}{gO>x<k2p1wY-PCpp4eG^U_K8J9&fM@z4R7{VL`8%9r@o-&;2L<}to?{Qc|;kc?YAu5HbzsiKix{>=AY^>@^2_^yMqY-CpQtuKv6;uK;A(;DrZ>i3GQp7z&UGXLm2rtirr8v?sc=DMv@C!I*`4Y29Bz2m%SmlAiP_z`g*NL5h&==C%ICgGUY68hBiNZTqtjI_Foz<vc6lGGiD>Ze+3;7jHlhSm9|-not^iv`<zgdKsnPEIbaJ^0P!=()A0Q^TEntk$5(o3iG!-ov-b<{=b^$>e7HF9P2^^A)LPgOlWa+sI*p%v(WzAHjzA_E9QAcZTxHI(gb`uuGV?q}1(bVcH)ah?E>e6`)d0BGDnR-t2N&f9HP!1Iwq+WYvFaRnQ9GQie}vL%eN*?5>jA(0DikLYe&@rOL~i9zeXCQJuyt|8)AKIo`EekIu<AvgUp{63bp`Wxr(9mXEH81zXjVv4CbRo{!qCeb}O<kolEzawI8+c?Sb<R6F+83qWbznqhQyqw{O~!m{4$#ZSnFUsf-aGCHJFX?_1Q3y9G!;RExEgAW2C!wn<yoR))zC;K_OTtmAcJ>P%Y_#~Ix)2(cz=5}NhQ6s9Xm(hJP9zFl41FXD}&RbY~|DH!gm)~8tgVdw`1+@_)ubjf$K{*3hwT^464D9yjv)7dEtX>ar6jbCay;RFDe?}cT@$+<SeBAEwdvIre?aDA%6r9_`^LH;5?`h<|5B#&Yp=%$?%k~ep-7IJQsSFA{hfk$JT$^2LOZ5a!$%Xx{if!Jp+|IpNj(aa{6;wT?pm%8hE}A0@Y*fv5qqe&4@jj(eeM3Jj7R!F@5k~sE1W%I0?`SgQr<rtdGX<9&`g+`JpM}2C>!;mx{_Dy&<1rXb@Tw-ueX;)<MB*j5h%FHL!g@bCUb8<s8cZF1d9*bvd8<fGI=$<^5S%w$1#83~7@0Omc(5Md!f3e_s3@q_2J5QOO5=bp?XZWLwY9OD^@<k^1ESDA_Ssud;U+buB~LCr2GB|Cax-|vS}bMK<M|#JzcwDHGEAK~Zi^e9Z&JUDf#YUbE6bM~)(-v3qPuPL&xT$$mKW*!P{TrCy2CT83?#4Gx_;priuD`S3DPsL`>OqSb7Hx{f5@^F!z=j?s2bBZrlUlqpNSRC8?$$Hl9OMG;%hu+5$4bAaO0x&L%)UB@SAyho-BuubgmQQ(f&*#dhh+JF_%)YM3Xo&$!F7blYZUVHM+1XwYM3y+^0etdK+^Vc^5bDSxq~y$!7LPB80UW-(T$DPXnIS53bJf7TNW#eR*&+YYP6+YR`IQkeAecijT9(<N`JIDzZ2OCi<8x{c`qIJZz{Z>UA#ZoR$@|{T)MkM?IO~k{T_b-WgrxKLXDi^sN-U_<N;VKhvcBb#oCMtiK86kNkWj2$Pf;H=Tpjc;uvVSeCoByy($fdyg;Y9lmP-jZ16zGdGbih_O9B8mg;zxf#@eUwo?bOS>;Er!Nx+q+6+OqQR}Qe19@ucEr}R;iJ{Y{$sM8aDNf`nYhLHLwCKgM~oPpvDKq-<hvj$0Bvu#M<D&l&o+bWS3K}o@5;l=odTsx1yMx>qERbiLjOL2LpWPMce;Zx?_|Jke52C`>s!hJyKYbCst{n!lgB~m{N+DX>|Vk2*(-_Q2>4g+NEh~Oz)IX>6gF5}CSA!ydXTVH!~S4|dLn~t<A@Dk{9y=nf;7Rm7x}acf(G30JzMa5=GJu>W3L;(I|>KiMggaYww^uttHG0tL7N<Xq37ueN<D7hZ@M=Z-yGk%AceNq)}6h4h3HT1TdeQZl2C8^?plAe6TmTc9!s9(n&Yhd^KR`<4boixV?5(;Rwl_d<z-h5HXehK(uWS`*J*hT&&LxVWOSiDI^OGB?ZLZU?{*!fv*o9IuHkB%EBiaw4%uZ_m8ub~fIkmC{*m{#;htUl@H}$#z4d-@d~?tbFR}Wb#>Bl74D+e6s$KV79Va?>#foSf8v!dNSMK~>q<%dP4q$4x|0>2hUwn_R%3^fxLDC;qu#MHKm}leb*6S?m{k|T++dtyj&)01(I_P0w%rjTov=IJ4Jn+f4HvZdc##xrk^h+eqw>PmS90c|7gE`zvo9%hxCfUGyTu19(%>H`Mz!W`#RTZbqmvO5=uZ4?D56A2;j>Mi(HM*Js+~_;CVyY{1%UYKM5}Eq|NVf`hTD(^C@oM^Hs#mK#cExQu^6q0St(P^W?-lG`5E8kDA&s}{{yBU^cvXSS_@xUe*<SY7`?PcP?`oHZ&iXtXX2DBn1G`Tl9esqdfQ{v|3MBX2wVv+|*MKxKgTBbz9m*p}?NZIj_TVlQnfA=+1c$2LQytjI$*_zWh)ixpy|IGjTh_ZepYntWBXhW{+7ErO+&NLVDL%g<L|h&0yi>Ye>7JQ~%enn~9Ky5fT;OG!*NdQwM&02BY<7TiGkwEYGSqgNwP)6#uI?*qMmbJ<gND48sBSZCj7+of;JEb63hbI$$D6Rl52>QN%ec$>j8&CMFHYg47fcT&Tvug=>Q1WgWz?v)2H_CCMb!)$cP#_wq04w9&fv4J*6Bk&?F_HT>pnk5_h!BY4N)>i!_{@CVcKC~{;?)0c`G(cpZL=o-ps>u)qK3Zy>>QLk!S0b!CZ9m`h7ajPxT*Vfo=C4UMO+Z!^zsayYQfCuYowT&VtsL)}@bH1H4dBv}@g;9R?n7hw^)@H`}W_^l7p6e3@G^ERy)hEycfWziRGNX7nwm(W9WG%5s=!fY>lX7(74~K_Pv~UOEMGvHXo_?B0i_No8z6KQ;kg-*?;m@we%E&KD@O>H=~9YCv{U_ukjM2eqntqo-I!^rP}+w)C^xbK`P$Je0uli8ZXg5;c{n;-!Ljx(=ZRU;UWp8NA_TJ-&UB=~c?7+<D)-?KW$rag}Rm4O$)4cC4K`Xlf>QuV#Dk=G%5LKDaKNt3&Lf%Jdin6@i*seB|_H0hJIyPlMX>vTl#(mtz*%E)!Gt^ItF;gfh{SbQrzTeA}h+j^7)G-aYF*>h6^A_v-HRu7>pMQG2n`FmH4>?05f~w9T^Mj(;qj*0DNS0&eH3i!A~&M)wwRM?gxHYI+M$H@6`)={`qv{^dpfxRPDzvU8((wYI^rN$ImU%ff0({+b>K-UrOA7Uoacp-LWr01xD6lxQAWMvZpESr62GnK=ulyumlCxB|7~@M<x}-dCq`ak{diV$Owp{iyD{6^#hSU=C2}u5)M2j^&A2M%c~?a``KwmcM9+{altdJK<<@y9_P8u8+PqWeYj{qr2bda;th}=9_F?JDv;)W#QTD!M85)U?uCc)aiOn=Tgq9u^IhV+Wo|Cfuqj2`|$^edygfyl^N@uB!cvQP4Ir20mg<^Wl-rIM@rQ{`bp_Gv=a(9y7994r?M$MRd4<hN`iwbZ<EeD(pF~W9uf&hRFcMVj3n*-;Qx15QG;I)2lLnZvgp3@YK<Zw{>$y<t+#Eqgum-_j%=vwZChI)4hYh_OK&wYc`>^If%9B94Z70+AHKZae{G=o44R_!7p^NzJMng>RPIgRn$?d)vsF~E!%y#X<OyC;1}VTS{!Gt0%q@f%{=8NrN%%9x6LQ2_BZYwEV;Ccz*gn+FJh)$Exv{_g0RttWcUGd%UvGyB+w@#>VS%Q~`l7a+3{}B+xjSxLA$BvO1H<=ST5`J^STpL7I-Bjl{A<k@Wb?Gl_z8E_t?HiO>ryhcuI*PkK)?Ye^2xML&sv25TjQw*;p}r=FHiI!Itj4vC&O5~@NBX_tcU6wjO;*6{7r?gZ%c8KK+zOWp(dCt0!-LKDRO9WRcfXQa9b0j-H$zBZ!vWm`vub#_*I=BEF0%uMf^pLSDdW{-fD9n>k&uB?$O@PF4VTSb}+?K4{W6s3Jz-|l-bj%tg=>5?QnsZ82#Npd3tAy+dsXyUkZ5}jQg)l={)ff2{O)a=f&tp!K4};MCCYy4r_G7R&F1q$P4AwUQX-yr{gt%rUOX}(*6!k!nSe(@%nVfgUFNWej82YBgEa9iDu|_zPewWtJ0(Kw@D)8kyDERG+=Vt+G_4<!2y1~`s&)EH)8Kg5pPvtr)My%^u?re%+pz!v8{Z0)wcGHtcG>1PY7VRb=2{is0;%%;+p;<R8M#IqWWKZQJ*+WBklsmiEA_s>|_v81z+Ebdi;nc)V-KbE@5vm%*St8%}{AY%KNDOdo@pS?P0z%EXy$O3@cb$<pK7!Yz@<L{Vk*BZ$yE>*m0XzV{gwDEdR?jqD7Ba4p!eilpCT=$lH9MU}=xd<hVTL47=?vDTln)Y5&-i4f8vofx+O}+X9=t)0lS`hbas7>H&|8XN^=sdAUL!y3X9C^ro(W<(WLs9*3jnTd%|7$?JEpD2NIm{T{s5P7<a|44PMaUOkHY#@cBQy9o@oPUt;7j5au1V%Hj!zklcZ6#GI~5xXeLI0Aw2kM4jpxFwIz5!(SN&VV~^Csg9K(tQZ01J|!MQ8``TkBPvM#}=XYX@E&K;<hNf?y$qIRci{3%i0M!oT2LN9>|SEHM{lOqG}3z^N6em!{2(-pnu&l@9Akn^8cTgxYNf>54*kKi$^E6)K<fVUXWF1E6sQ%@J3BKibyluwPVRSnc?{RS62P7`TShr%d$0UK3dBOp&)xvNjml!FB{Qoq3&bnCumOM8IAV?*FNX-dxD>&`_fz7XbtSY+d&7qgc(x^U2Jn+yn8!3eeBryd9KX2%OuYoW_&9I>4`Nf-HTQ3mxEUOx@**b;&|`yw^zl0iz$Ch>)Hak+#A#5#@uRkK);m8mBKg%g80>6l+l+<>(D#WnZth|aJ(m%(PQMxe|`B_#OSpmN5|3Q&B~$J*v79_;(YK`AgcrV)7{qI7mo0kS^W^s6Y47df^bq>u<d*!tWKAsCf^^H(of;_03cNMgmqk7S;<cC%Hi*)4(5&Jkvj-EwjM~tI<J1GSFMrBf8h5UwJL)cyb+k4$nRhA>DI&6i<S6Z|KpY>f-dOa!py!AJ;3FcTd)IweV@OjN@wsylWQQ5X`LoNQzL0rEy{6hT-VWD9m*ud3zsj&zt(MfR3xYyanu1vcIn|(+=@pPpT%c>+yM@cO8GN?#~$v+Ub`JMCi?CPh275b(&S&ziWxtDU!kwGp&qqVxADXyGU#$yKB?07c4*w+^?JVbEs*P9-PpJ{U5Jy~6F>*%og+eZGBHcOnRTx_`3Nx3XVu?<xLGgVt2wCXHfQ2fnq1xbz1HJ7WCWL(pU|y31p~OVz(Hw_@ynCYmFK|<0)JT7qw&t_vNgkFhx?`tRt(tFF1~J2&<intiHVP+VL-Ju`vdRZ<Mdnx`7dgZ9t*qeZwBqd=&UMjJGsqX*9~xz<o4IteeX-Mc>d-h{9C%AbDMs;6xg^Gh4{xA=jJqg4mUiQfx~8YxSoQS<lfm69Rrg<y8A5v-ITkdUHM5chR>Sv5S(N*-5q@^%lkv@Oje(l)ni<3{{c{Oysk7ipMF@@h>qaao?tc{^BR$5QEPh#Lf~0a&q(0p=0q6YYmMg0uZW8ZEBw-Xt7#8C1l-94@8q<_b_+bIzs>Q>v>CN&N~(RqDBX;@X)wuk&bz;^7ncjgygr&9h>g=svBBKw|C)z!O~>SVlm#KkFb)+B;fL8Z5oqH+B=5OChHupLg&s_Oh0bXK_5CRkgLjwo4#U+1TC6u1;?^4~HC{|}_xbmr99jD#DC{pWXHVq<_njsLC04&zH=1cPe1F2__PElx(O`U*^xze*{PAu+pYFoa@uEw(-8H#l`@%VW(%{?eWqIe)Q|-<ZubqC(;3hH_skM$Q9(=V*ACk2VY-U5Z&h=lnrU<eH<N@_e+gP%Jyl&1b)1Y>%c95u`!}ZJl3yOLazB-S@>1|PZJ=m1VdiudnRpm35ZWT^{j{3^Ysn-KxJD<+YS|X2&K4;DRRkyX}<;>Xf>u$}dP9KA=WQ=QqJaCkO)yJUEkz>Z&yWeSL@2OYv=;&u;ya!Re+k1^11>3GLYcqma`BZGozc}2Y(HcD0y4IpUV{b?Wx<HTW?THNK!IH#rOY7|)(PbS>MeN}+2k<jsl6=z&zn32}_+7!c6I`yN*18up&I&Rx4RECl40Es5z%T5T2Za7gLHX@JNDsSyqGa;xTa0Ynl8~^{=Ba6OdTct8oyz%fxP{v-%2qa2z-$iQ#z?HU9Ro#M@1A;nkgd~QB(m0qe0;5ZBoh1=`CIMHQ9HR2d8HOk9^s5EaTJ?WpDEI0R;MMx$@AJ%9P4Z^P2_z=Kq?or)1#{baa&0*xZ?SPuMm}>zMLU(=??~fX@;`LMU&W$`JuDw5Yw4UJGI-ZkP<G3(`#~R)h2(@O|P{K`M7RZ$m<*9Rqn|HHWn|_b*-F^kj?dmDpHw^wY>>*Ix2Z7eOVlmKLX`%+{K%z)WDYZ_v>Rh1klL271VYdqcCwSKLl1so|~!Attp&)wRGHnQHY35?gQkwT_NGXeBy8W&OYk8p`Z*C{t#)0;nTB8MR@Afcms0LSXQPh-CyX#<7YQs)d~Ey8=Wh+cDS5=yj;>m83P3wdJ^2dH#!Do7oN_(t-U;K{O<S1l-*sd_ny&SpJ^{CtjBxSjT`v45Phb$&E^}0<B|j2-}FJK;D&K>q4d7XEijB4@A<{Lb5yc!a<?%>+y4HXL_K5drdWAoaMqfb_p;re8^vo}{A%9XsDCyH+<I1_*KGJC`?Cpl^{`l)qm^6N0Lkff{>Fi7kgNmEk3;kOS2O7Pt1erYN-P+76Zd!VYd34ImD#c!lILRsF3`uQIs#ZERxgiIGmb*#&m$He?VeKW7m-n)3(736m*!6RX75JytaP5{<<DQ1W43~UMTgzwJwl5w@ppdse{lO8o_(^5?i$%?{Obv?>h*r~CAVCWVo%Jp2R*VTwU9lhZqv0z&EO?}PL2ul2b<M=yXx4Z^9glV%3%y}{qU`_NEg~5Oq6xx+i0k~L7-l|8b3HpUfc&AO=sYK{ne1R()Luc?wsB(x0B~eyn-~fIC2pbNkC)xR;;vD;a+Ds5+2KgGC<Bmc(#Mk-TZBpvslkNVvulTb+`OpjRo|EqQesbg)LX#woXqkzxA?%z@5UwyTR@FPpWKHlMWPI0xHaI7E!nFwbzaQcJKJuv~iKCtV8r3_3l=2pA~*B6;|bfRi9S#i#_R$QajEoJCxp2d+!C_j!cHbQwBY*+=&~%z`Vnv>(5(y?aak`0j2x=ithJZ`u=_NgI}!dAxDK}k6*oS@fnDu=SSvx_ph~Y`x>0=L$6p=Q|^tmW4-6>@3vKp$78=SLHX&pqt>I_toOcR^a8c4bZJcv*U;^r*xz(#H7R}}WP5BYogv!D0&TQD63f^cd@S|y=Cs4jk!Pd)bf|VEVs{y?fsV14uH*zg65M5dxXYobdOE@!Udx}kUGK*~u423y#2vX+!|IN|1mIY9bzL=_D}q(Jm??~<U+e={<gnVGw^Q%v_if^vzt)1O{<d55_#nyPQ>oeeCe|=2c3}L*XKiO6s!-9!T7Ua@xg|mAu5HrUJ$ZL){zRtt=CgN_j9JZ$r1M$E(P-ua)L-<A8(ByD_xG%yr<iJaN|Om(CZy8JTQK>TdiyxuxjM#d1bZR$5v}R0e>0<BYduY&>9t!QE74*@Qs`sw@ZT>MB;RhgdU&>`j6EDrTu&@D5uma&v0iz(x@@X%*Ql}LMR_I8toY6Kz2t7l_o>*=KHq_=5A`uhzfd|@Y=qk(tQS0LAC|;#J6VTXr()Bm`w_2ae?}L7r%j-;#C9}lJ@WYnUp9O5hkgdN&Yv32dhF&_G}e=v38f4HRR;pr5xmA>wb9e&E2yLBSYTSKL#EU>`g03>)Cs3Bl79L*<2qD@dDEOLdO2=RVQ|;GD#c2U<n8l^M*8c?^6j*~4bLoAJJFSDtOq?L>-_m<PppZ|h$iLpx84jT;*L%7=&=^jN`AFq^Ah5w+5)lWeOv3a=h|&Gp6C@D`t6RoJ6g=(o6D0eadM?P=-E1r=TddFbVeS|UIzC=fp@Qga7fJSwzxs-g<_X4p{JU|v-M$bAQi9Q19{wK2*t|}#?)=vy-KGOj?_*&JpK$ko`O}_%o}{QKmKjt^X`cc>n%)pAcQkPpZ0wgb96DXF0!@-Qc>l9gXLIk%<Vh%cq~@%>0MQwQ9@afOjEP`gr}2--Q35>Vz=?y#x46CE_VF@6qjh+L243<q;+n4MvyIwtYnX-k9YBR*`~MO5<#6ud}(0rzDPFg{qda0NK++gJnuKT*5OW6iN>c~^(x1S)p@lOaUb<DJ_fk<Z6(Z5wenyNd*Zkgq2WzCSGK%&BD(PGJZvqe+E^N{o9!8B%@K&_XyYQ*;@h?pu4NNv1ONb>O2&9C^YB&gwN)^l><6Glk0%biaqoY^UvZ>ww;LLc(CX>&i>yWmej#O8)MQbvk2M0YyBgFZ5Sq=NL$=wvKL7?L?uQR`u5VD=i>9sNpR4iM7v0ipr@w;-uyWUT?QhjJX8GeHed~5E^r3rl-+hJsqVC`UGYlU3V4-mQW$NEcU_Im5<3kWv)sfdXBzOsW0pg#&qsO7>GJNRMJm;emX1w13c-rumnR~3C;oW4{2kekHfuU_ocH4LQ+nDl^M0kw1dJC39=uLlpTO5k*<nw;Z(*P7Mf-lgC^wYdC7!M?nYm-1tsOzt64uO}g>(5{}T#Uatt$m|E<NblRO-1gH)lugvr;Dp<?ku}^gotTg#+oG-+k5-!zS{e<e{Cn5Aek=xfo83rb1{*NXYm{A*INx4!8@0Pq>j4CA3#`aq|ZVbJCl^|JPu73(X!Fu=@&1)MSi~gcL1-;kUMPwYU4fkzY*d0Dth-2oyQOyB;)}K7t%W<(<`UtHFXd40M%aP_nxQp>VM04fm&8m3P}BQ2PxO7x*FDoD)iPi;itXHX~srWr?XT4n*9`9QTMfHj*^!v4D~1L5k3^cvW^B#v|A2)xN=GjfnAb)ggjPe{mO3vud0*9XFo7^Z*VpMfpfY8my<bMwpLdhO2E|&F?#IF?klH-$3*#&GzjB`KNjyPwojk-TS!(XD421V-6!C)HOltBAt)blIk<!6qYmyKpPu$kXm;K(s0a0Tp}Fs6pVMCG3iGm;YNwNbUMp>C{zb{M7u>8;w#O%e)ZEjyh=kq{ns&qY$(Y#9=|qi3>K|*opSSOrl--In7P9la%EWo)99`)jdOxwV;p2PaCRdJ*gnLaLIt2%m)hzn;itQ~KO8WLNo1MhnrIQO4+KO)PzvU`9!oVjx9V*3uUNi>u;}9567WYtkCogTX{?fLC4eHtBXfS;??c)V<18A~5?c22<tEb@ZTW9%O0zGUHh>n5L^2(?a3`@nTBLDyXdtUk!9D~$m9=N#E=q|W9s17QF+jB_RyJK}{E@A4=G3d&+#9Pz#(XKuo_2_6SBW}UM=bp39JFq^2x+v0kZwEw}jo%LjE{Tjw0jNmNSo5dXlP9TNv4}q<qw&5`;eXrb>cx=`xjVH5rv(~~72n~GL*qz_XWk?Z?Z}MoNQM-CYfffT%l^C3CM?4AL$$e_UOx2s8a&RPW22x8wpTl!EFgcp;*l2jQG9w7&tdRz9i2PfLb2tqztGof?3_NATDx+(qY*IcOX%G9?I-1K_RG9~yHuyA_MaAgUKMK~LZ5oH?Z1Kcud?p^%zF$TPg=Ba>DTqwK9>^3K!SkIH_kh)Tght;Z(aH>e5<Uq-=4+i?mUiO_Y=o%s?AQV`RFF$D*1q&@`nl+suyC(b$Q$Fk)VH-?kZy(f3KF!E!DuDIY{G?U#E{g%SkxIa;9X}YA#g?!V0a5nvT30>2OS2pw|~K1C_MZm^Y5dFV5zd<R_5#^MTPDxX@`@vDA^z%`4r_RUW|mCSn6_atmuskt|XFimj(`7tEwUps?ZV`zI+J^PMSNtu)E-B2jz~-$3h@e8_ViAPAsrRao(?!m*&pybRojayeQp926&^&$b&jR7UU;tQJOQ?R}U`V3`o`{97*bwTo9DMvt5CXHH{fcVcWbAI1H%_hDvUsO#AI>-)>3?+%Uoh#_X~Z}+PwMdT#&X`e3FX=lLctC;Z0{2TpEHhmH<;|p=0PmTo*jD<do^|d$fI<8Vqi)LiY$`wkPL*;vmZKM4SG0OHIxtznIP>+NA_lK-m+&F!6qFXad#Mji?Wsa7{Mz^9(nr4l)k)icD@;TH;$n=ra(=`^Jj_o^jnf>5y%agTw0e|p!HTXP>%QhOl*Dhvs{HPBvtkH`aj2@YEoC6H<6H>Zb0FM{(sBI*z=OYeVm{lm@L3l=nJ9=-${B>(Kl^o%^2Y(VW+WF_Vz2M9Lj=NKFCSv%>#c3?a*vY8sc+=lrxk75idftBQgvRA2g9l!o@}d0=pR6mp%m7cCXEBCr#X5}#c6H2Nrb;{f6S|Upw>oz7bJ~?HW@K2tcw8jtdT@Y8O0_=4C9_+#x%NeB{|z#fi}3KZm6usQE?;Oo^t4XT8H0_6fK}^ru4vJF53kl*+1mw>&piLNf1TO(Gp-hdmT982Yh)~tMXtmx;q_a5;Tt*P@HW$?I6IMlXs!(^e4FU*knRc(iMETc`kT|e)ehpn=~fpY+Oygbp~H~+8X&7r=`|H^v%fA)`WApzIUls5LzCJiLVaGNctNkUdSE@Ua;LTRTMlr11ps}-H6Uk5W-k?f0Y=Tf!qoIJ9ZSitiG8NZu<m6K;pIasbUrit-Pl^QtHbA#y}e=G2cFtuNL?!01M%yJMt6+*VfEHs_0Ep481)r({zzbO!n6;_DWHcD7G<qf2zjzX#{E7v*3Iq579Ed~Pbwm)bM5$*%=5PsbPn&&5wz5A{ro%%Xq{~%bN!>Qv|BzR{=k8XeMVrpUcK)i_I}Ic2G^{%_RZ60W8w|I4{VjXTb(WMI0INXH*I>8LFF*ap+)_veT3Ntd?_yU3UVUa397#3x+A+txAD?c{c2ww{OrV!hv)N*L_%-_b@dwASh>T-0jn6~kEu7u;|h0)6;ZC?ndQ&xFR7bA!D%|!9eM!GJp8oosBzGdAJ#Q%%~N-^o}A_5+w*a1$X<O9|HQNb_TsS7&08f)Wr}^=`&ry_fk(>;bZ*3=`b=mmz`dS+Ynf_a@<}a_wxiM4pBcCFE=I-{_j}akPg>W7@2onObVk&eb!}5a^U$@M$y8xHWH=^f3(e^~Hh;r;ZX#uEf2j1szy98psV~n3G;90$r%>*veIp(=&G8!<d2U?7%RkCn`_}b^0nf_)-Fsb-@|7Z!o9k6>8*F}o&VTbw0%!2mXH4TZ8#?)W>VBa7HxpQfddO2RA$_r_99pcN?5n_ClP>DH);~mcfqrr)(X}hRam4068kzUGmN`5Bq(&|MLk{6}OB?iOHDdXRhfiaG`Jkc1=BC4@pXZutOy)THZ7&-j4U%hooxjt*#tWWdJ$uiKJ7#ZH_%C%ngRNjvQUzxjukOIt*Ymc6I<An|H%+3YZL#BOF>I)(NS$f2{?jz%v2bqgWj$+sml&a`(>u|ov=O&OpL<pp&`q`vmg&3E>vb$V;nT?TxlwP1F6$>M?SA9I-a6QO!`nEO>+jySi`)0W_j}%xY1VDe^c97lx%E*Fm5bQn)1O?6!ev9@Ov~1OTNCfq#glF6rnGL*+p;}tz1WHwxqse7?Wk6LX$KfMYWf3Sb9P}xW}e@|LIJTp(hk{^=b2wPp6M@g_k+|8S^i$8HSQ9-zbR1Q_Ot5`4(-`+4!l?W#qL{N^`TYVwOe>Bi%&(`fpuVRR*@U>tq-PG<ROFJU{td)y_LdOvT^0r!=$)>*kj1W!^Lv^+6315eKyYv@_}d;T_XJ$xg%^Q<J`qJeKK3LIwz(X-09>mc2@xv6m=s#4&e5%etC==ym9McC)dH7!<%>5MU{=BF(~CBW{TT?BaxAFdU*nHYe?Tp3Sz+YvEEb9e8cvQHzZRkL=6A&b!I!O0>~DA7Q+zKm>@Dn1OyQQ0q5yLL>xd61wj$iXFvb$+tcYA6BERts@D3Jk&?0o!?#JXk3;gAqx;o@8FNrrDM%nH6dtS28IQ-FfMN`bG}PzAtumaP!mH{(jzs3)Z?#P5V51V&;qYwe%pkutNAKkb5XbkK7!h)R9PRHaZq>o%23snedI~6Fgg7(ldMJU#c=|f&EVz}|IN?-Y^zJYHywo`|w}kjHoduiXwC@}8XD5x~NGs561e<RC<yaHaC}yTznh_@LH%sucuL;}ra5`8WMl>Z5Yvhs}$OMr=3uVFIZcEgETZCbde6<++gT%afj#*og0Cg-v;HY}SgnlkZmmN8%_G1-;;i#jzFB+&*#DhOT=&gshuQLfj%i4Ci626J(FS=l|N#O3RzPwnyu4{FmyU{6~bdva%Z2aMU4PHq0NqD#B&+DtYY&Sgy76oIWw2x&=74=J%WKF&4)jzjOk<3nxKFRG7pD3}^(#-xN@7=u8*snSiptT<S-e*Omm~lMZNI$bKdzhQ6{rLPN@2#FQuFw~&ZyR-NgLi6^KMm}3ANV)YsTE_uBOhCc71r(B7#o~(TMC<BE#ChKApB_cwGw}j5zM))1<u;0xt}jkuaE8-<u-7q&Cx*a8HYa4PVP-+N*oBtLyG1KR1$m90pabq_93<=3FOT`eNydlp8g$@iX8g84XfmL1PK1lejQJBey8Fq%?(SA7}5H@RZrW|b*U{jzwtfE4!&1Y6<7d<@EElCRyKZ~`DBXJKEQ}v*Ix4XRs*Wd9(T>V?60}%q14FCrhi`lt5mB55m81_JCqsG%Nl)uaXxJZzpsY@#ek#xNW=K>lN$r<(f4Y(z8`l@W9fqHKv=Y!m)o9Xuhyg2n7gMhiraho5nX{zOL$`Va>-ywaJRO#)j9xI;HrK^o9iCaSr_IJ!%m;H$1O?_YKi8mPqrHKFuPp2-T2jKw$lQSe`E3Z937&dKHXN=@mEd`(rSI%TKPMvj-LkK?7_#w5h8az^$_<$OzYiuZ5McLmHOW6&NqV@(8xMP!}PUN&J|33zCYH+i*dKQW?t<ZhCczPr=!*W<drqzJgKR#8)_lwyDcD}a^4=l11t{mfBPlgr+&EGfrFusv0M)_R^?C0)yaKP-(6hUZEBZ>QGq-4{Kq~hv^(t0=jODrW%PA~>M)^4XRUtL`o9YLE<e)n<oBsSt8VxCw<<E57iiTO;WfRIjVn1}Y^4}!HupgN?SOkYSCHAi*t9qj!t$XzB74rO`}lNFUi+`6<=urgkRfoS|K%6<l<$A58RIh!J1tg9L}f}rMP2Z&jId?tRz9_MFP+}dR<$_`5vTGu1e;=Ni+jUr-ygfqegSlBhA+F_ry(Hr@#oAqcDZUTEHStvfnj6#WLq#QvrA$pVIF9@^O!r!wlRkmXtJ1j680|U!;OAyW?+Ab#5v>9dOGk1lS<fUWaDSU8MOrPq<C>Qf$*76JJYdt?xL-_?ls`4R*iS?<?WLB7`<<KmSU9BhX+&Kq?^i)XBNvXhI6;FV~+dJ!nq$!D1jv<9Q&*HDVXiHvnw>L*EfsK`q3Pp^i8VK=IF2ZxsUG$$^7aKJN2&7OwQnzpp;5BH^K(jJm@!31t)D;17L4<G-ulZcEUzu1F5{@2sfSQ%sgLYqX(Nr&+B4jwq%C<QsiN;;ojRnXa0WE)%(!{J-qWf&YSSbN3&(qi*~&im0Yd+Fz&GT{(g5<^b%@6H_O4U{ua)|!?^PY#v3D?m{04F5?XA2kdy>jV(u?gK^wOd3f>SSjIdkF;Q)MpO>A^OlXr?}ow$Y2{j%OMCx~;7^nh(*ZC#$u^^V`hFCiVcFnxUe%$xwKrxY9*lfJ9Hu7k}a=G9Yn8M)F@tbzA=>3vq%#F>;(Ruy~3SBPpy+j6^2p|Zf{Q2O#V-;(<l-DMSy%zV&z;?M#LgY6mDIO4dyo!Z1yIS>4Kw4sAtqEtqe38?j+-P_V~o@b2^`01s+-7~X6;|*`@<9a{u(wc{k!*kx6{{bW0tzJeP%*e&F!LIKfcvK(Qi#YlgNIqtVmLx*SsdC{C0Icg4xmj1*%A7`Czge{vw1&{01t)T7C+JFymZ|iBs`-3Bef?VW=Ol*L<I&q)PR-6um#~bSjK=<(d*0IHl<MRw#P7T9`@PrSJj`t(^5|XJ7BaQBnLBR+P{thJ=j*Nx$@9*KV4-d&n^e8Fw<y3XJ$BR}fBw<Vc=3!k`1j{^_r1sMiNx2}xhJiKX#ecbie{)yk~;d%5AV~T!r3TcSr?e^DZtHW`GHZn@r~aJ*fgump8^*TwKKv`1K6$iItlKR%$RCCzUr8fA&W`C`p_J=jtr|jNEO|*0NP%!G7J<0VO_Y&@LBt`!Y558)AwsW#{KE<u)Gdu{?HL&tXi&mL{c7v3Iiiz=;tp9Sqb4spS^08$w(am)Zb?@JK#ikd+Lnz#c%sAdw;O%bP>gU=Y6xE-x;alf|w4HDY+!*^6pgweD%FE_X-l2QbZRmdr&I_x2wh5jMWui7_}OD`T4NucSKSB`SD{N%L@csj(KX)9dBF1_Gt1RK}D?->e%28-JFA^L@*(&nvozlG<aG3MW-%A*foE+dEJ>tABwm8%XtHtAJw|q*YM!Hup{%_6AL0ov_XGxU%E%E@AdRg+BZX`A+=DP-!y-X`MiqsG0%(ZOXI~kI^k?a&%jURvy>KI*mO_hucsQbb-Ewq?-5WBrS{8iceiPO8CSp<ns!Iww8TkjVd=fXdQ!)(z6a?F{qjkoT4zZ=h1bV#9mFd;4V2_+*aL<HDNM>-pS6jq5(&o<@PTbpM*@E=8%BNSuT}oRDnvOX42s{smbk=U=3`-h@{f11TJT_Zs7p2YN|zSgWW)!5HL0&w;l{{C`7Nb3;A-#qA+C7InY8k$G;ej*wpLx$myd~&wu$msU2}ofhXT2QHK|U>D=pLYvHqnKXG)r!WkkQuH<xaUrH<*IAs=PwsG6`lvvQTXc>bSW`v;vYQMcNe2I~CJ9PeAne{Xp6WzQ~N>Ie_{ubGZVc-(K3-wMb!&)Zg&d+q{Hh+cbHo2|6y2Esw??Kdn|<@Vyls<yQGV82kpkNkVy&}nHJ!=y=NWwpmWG;P3)lGM~rYJZRNg<u=A+F10B6wNCAl~vT*Q^@E}+EYOq>f}NkzAxPK*k<?jJt-?SC*e{j;nAsma;{}&W4{~O9OzIT5R+DY-hBMBl-RF+&y;?4Gch}&`$Yu-!Am5aS9(v|x68|28Mb?e^%_`xfBs%hCo6gL@G6zwYsV^@>(sTcYR!}Let}c%1D(Lz<?p)gRQk2&u`HR>Tl<EG-!@*f<_{DP>DNZr<C>W~fBgxHA6SudCZF3ZpVPo;n4eTZox<<KjLK%T>OMCe#;i^!(>53P%s#tgXk&M<p-?_gQSI;Df9S9JC&8I7uJ=Lm><U}aEVI*eNx+i>Soe@{zgnIdau8|5!SheM%+~Iqb)FG(4>@6SJCrSL+}K^c?H2J5A<>w@5H$(t;~E7~bUnw<zYyLZ?wf@d#(NE^1o_XF(^`mcdXyHUb+uf*nCqX<J4Sq~M6WIbyk9!~c5ZF{v!#vAz5+xbzH+#N3{Cz$o#_2;^g<OGBcgy4-R|7lHygsn_ZIcx>)RQu^!<H4k;nVGud44Cs1SKx&4XY}pC;CCsEva8V+Rz#deaffzbWdD^)%_iaphPYF!Hu@>Lu&KtGFX%_g0-7ae|MxcDn7I@8%Ba9eSP?Scewd`8W?jg#F}rmrI+gDlY#gVI&#9@$Wc`OER^W>g_~BB9d42-pLdPRIT1=wKuel9)@p4>ei(=TPIhvCvX+3i}<olaJE+-`vZI!p=O(nMX082Pq$VPCqc7P->!P<6Sm7S`7#dfu{YZvL%sVOqZLZt_aEsq<l=Xov>CMI|4Or3_-#<D8_DrprLXq&axUWYx_u<-)SK&_hr=@JS<a6h#qs9Q%iG1a*Xl)*^*ax>elDOuYcLyEKf}ZnJ0s)e4QHm8!P&ZzLuKmZTaAA4m*%`W8?-L=>h;l-&PUt#Uyn<jPPwOn49zEiJsWG)&E}R9)~@t?@}%SSAn6)@{2hG>(@EpUPs8G~dW{z2k#U6JFG{nXh3JdunU1c@t=9+<yza?OV%$)A)P$C*fK!cM`Zn)H%U@K}+vc`bc)!;pF+0QCupAA=hA|n$>+XfJ{PqVh)?qMy<edyg&h;<-N3DMn9So-IGSI!(XZRjVd^#!y*g1M_UjDW*oi9ztyAc8UvxfBrhaqc+99DkvqL!m^x6*fiCMACVfOJ8im37X;z+z84<14-lTc8^h`hwq#PorKuVPI9!=9hCiDnqCpPx#U8$oGf&PT%2VB-9`1b1zL_^!x9AjfQJg2N(PgtDJE({M1T(V>d4r+u82FS^A@Vbn&ithfu<gWARew18mS|=Y-IK{3^&8k5Z2ohs;|&IpO(Gl=*L)xiZ;mY5is6GbcXXH-26f@4qIye-4!s(UZ3)?<z+`bDf2srVaaVXFd`h;K3UF6#SD+?SkrWHmfbx6<X{KG@b-Lx*kvVJZObAh9c8l@%T0yRe<oZ&K@nzEpU8pLPQ6j0KXC*%fDohN7Lu$HNMQ=CB{%#!ml6uQQN7Bb$EeqD0bM<j<Nv8<Qf<X(5J;__ey7@|186YCa6QV+w$y4Z-;!7)^n5MxILhU8q)gxl5&Y=qUagqX3u?Yue+$QvquAno(oEus2*+D_kIiF@DqDOkNj?yPHtVzDZe%P!zSgay+U$s!2pWB(JGT^7j+tGQ$2{`va9dX@(o%W958V2LGM;`yViQG+W8-}yeaw*nLCTqJ(<-Zzxh;lvym^H?`mV4sp1rF&7+qfA9^@2IZGS&vKsJ~f*$*QQ@zGAtVrYM;roQ)i3MLk&*@zLYEo}ro!Q1an;1klcWnd-uzu`eI(N8+JqB`^Dpz%4e93(q!km3Xd)pj_MAouh<$C<AWnNfa6bTpSLY-lE;rSd7S0H+snw@%O=CwbFeS(QbIc$H<`cB8kpntJ%440JR;Y8UR%bRqFNWyUR-7k96Y>Td+p)Q{G-cefj&8ouZ-6QuhTg`0kQqOCwIZ*?SY9)lNA2ncj<n;<OxRKPKLp8zr?a$>({v8lN=c)<E&uFFJAuK1UHDCo=<D+i?A|%6Z{p@n@sdc`L^Yv&41$?!Q9hSB44=M3ni?!ac(H4ZOeGIxE&e+rrSiSAN+>7rA#Ktela{>T&+)*~#dsnP-jwVXZ#45&i|N7Je`#qDv>TOAHcYFJtJ(a67AD3(Co0CIYF4(0EqvW-IdxDUD|Nfy?|9g!97~VHY`AFA>W&uH|m987u<LkCfPzX2IA39&^ps`*ArAmJJG4KLGWt+Oet@am*a?qLXhu8eGl!a8sACNV(oH4nY{S_}cAK%HHNd?G5ew)s$`<EX5dH=pTo~5PpaRB;{wr(pP>&mZ2!ok>?KtksryxH$1wt?p*Jk{hXzBo)gJt;4&&YVIosANz&tvY+TvZZLSd#q6%K(%4597PTJQQ4wRF-G8(UX!!E4F$;Z#4IfqKSNgMB((rnq?`eOY!7jr4ea~Rxc@x?xt>f&S$O19)s_?b9Tf3L?~J~($*n~9gKoIyc6V1wC_EhZzddk$*bH%pC;j#`sIvQw@9#1lEE}hvyUz?M>4l@OIRUHhTEW`H0A1KMO~s|oHdm+LeJSp0Ua)^9!>KRIo8Dz(Tgx=0*OQ&elVh_Xa;DZ*Vqx(39qPlA_3-ZLO+7ZNm;PHxMCe1#FsL{0>GtO7eNYsf79$ByKP0&&PlY8n;SN<VPq$h9*N(CFufe<vqOU?Bt4T};1p{J_gU7$jN$LL<MS<CS*1&FurDcW>w3b;P1Bs}*g1_20Og>E!t4gu~boF?W?FRKrx{WZ^lCWD#z-r=kKrT0h(df<j-Fi`0ZwmrZF8uJA-&@|w8P*Ij=%sWMV(q~q>tXaX7S3=+9d||Z+&H`Y8xLw_b<?VMLpQ8$0&7t$$NM5~lLo7xGTv{nw}%3O_b8!Z-Wxl0YjXDL+xD^%+8`6o)33PY<n9_pWyVg?vcE$oHU8%LGdX$fUYC0@a?U2nXXfJBo03f6Sor5g?yjw1t*yvWZZRi!PM+#Kj*!(taBf2K#qUo2f(7XM!WhbY3=#CK0_5r=U!zH{iz#@QinCCZ#;!A2!QV0P;aGe8;?IlyLfV}*<REm^wWePH7C7e9F8)5lZMxRBR`KBv&-a=Kp9lB_UDJIhk!2y=+0Jr_WqZfz;blFQTy-m5WY}KEztD3_%b1#k`W|nC{-$0(;<@R7DTwye<-4<dHgbQyWFJZT-WmFY(L{Qn@0^04th@%Iy{%5|J`i;B^JHskcsB#c{dCd|?5s$9;Dhbn9?`y0aJqCFIhOUIiqUd5<K{~<+JRXv%6KN)&6W79m+#()*xo{<YyH(Zt(M>zPR7n6NG|o=1%ATG^_aD->`F|9mCC-eSg%SpknEQorjRJc{fp<Kda%>UNE!!MLs;0GJJY;r-+MtLw9F}Y$hX7B<uS<H<L!7IeaL+0o+9eFY7g@JtoaK6eJcn$=S!K%!tKLi-v?ip!}a8Qaa*5x!Ou_DtD#@58HE3^S@Q3yc%>SkQO6-9(L?EIBy2EBBp9N%`y}o7-4UJ@>`aOkB7+V;)GdB<dHhpbMq9Zq?#Fd2)!xf2v*K1)QSUiXgj@9L$tp+pt8+|qjVVZ<DfN3ua&j-PZx@CaeWG3sj`hs$v81Y36*OtwOS@U>Ir@!%lVDpsqY3C_ovA^PxxhHDTB{ab6N3JI`&PUaL>Bpr1~|dH{R14eu78L8#p4aq*?Q1tCPB)2HzF&iHj^w)J7jbx%v$;eL4BHEx+t=kvY6hm4~}=t?z-VeP>6Q9XjNUpS7>z}d@Bbk$9?3fH}q-x{Fd!%<wiJKI?9U0(G1DrY_!UJ-bAVK0~GbGFcz@yDXQ8~r|#8?a@n>hW{eAKuD|}Iz0>9c2{y~|ONl>!n|?k9$z5sZW5xnb6>H^}pO5X%@>_5W&&Bl+?;EYR|Jx7pK!{Jj{gZ(wt~Em^&E~ng6y}{ASaDls)+t`j{V=|xy(SK_?xIQ#lNEYdCc-VA{ktHcPU>0}@6A`!C*Dx$%xTsEcyfde!%{fn*w_@3Exl?;4UK)(W;5~n9VG9Zta_)~HIW;)JAeIoLt(rcjcd<*)~UKXc5oI9axsg({jnes<B;J>;5MAsjO4PI%6d83@%IjA&W6m#{A+dO-~X%&O%t3v`kz&=WG}}$zNl6N9OyVR)onNLo9LJiv7|DXfzZS<_x8#i6tlfwMQ1<jwVxPpiog~jDtow{Hg{#~XwD06iZ%DePO62=Q%`|OJ_NKawr;I7$dlhwBew5et}=h=123Y-wmZ#dS!;4>Zwzn!r10lShy%3Mu9y5IKq7MU<ldFKoV4P}58e^Ss**>{g=xR*M4+;4`IwP!qkSoYNo{y+o>rrI$LE_sc0+F85iXm%E<)EENHN8ihJ4yIF6^9lEM8`yZwf3+>-%V|;X2<v5SyI#>_-Kx5N(OJuF$jm4hmOS_*nOU!}D&a0I3Ub{g1dHwchpKrDJZ23`uEN4nJ6VXlR#D7kz7v8|i7wpbOgX(J|B;_cJ0EhZCVT*w;`OuT0*(ke&-tcgi!Hbxg@P!OVLL^s3Uv?c*c2?rH3I%m8fIi#}`K6VvD|E;`r{qHCp7WA*bQIBc8QO&C$~$?GC&kHm-9nNIEq!{FA$`sBp^tzq-0uh_`66|#@I*Ow|h<89lKC3@S=a#KG_kQ<-&z5C{_tWW1z#)yMdnSbPZ?^eGzHhpKS58u^I%}*v%oEoy{y3}AYGyJyQeyeV*cr~_?==sC@vo6y8iZfIF)Uvn!<$KR+6)S|Kwc#v>;HXK}`wThDZCur{CjiSA3aTMfiJO;zRv4)<4!Dh5_|<yH66dFk3ztmr7hq^kJ^R&03tfOuZUTN6@_aL=I^1bGmTK?!w)I&({T<%>L&5gaxO59`R%Ko_Uinq$yIS*vx=(|<RHc=xE?A-gU#X9q+4{+>cJ!l6yplaS60X(6kCympvu-#XIqhxEYt9?@@p<~!Vbg(enYZEJ?EY9yKKY<!cDD7l)T-1+v)@XzD7?Q7gk`Rr?(^$%MRD_xT3Y~fUok}MldN;}gf+zt8>(7looTYrq_X7#Cc6%%Z)Ik-(J=ixq27f))Al@hjor38xv7HTI)?^dFQ+XPqoR0BfV2mj%!`LQ9nO*Vz&dW-9db52tLOat%pcB*wpK;l?*-(S%GD4>kBmso887-Y>Q5p>p3Jjf@t8pjS2W4^vl>N%<dNH92e<YZ@p>e3lO?gH51w0ry|tK^H{{FeGL~=G=K!Kb-V%t5tE4IgLpHZ(>V5Ei*Qh5T+=~=HH@hoqAIj!~oNPk3f||1_KNt(8-`^ZwTcdGV%mTLvz%7n97RvY`WTWZu^rE$4H=CO1Gn&rDh495vp{%c`>HL^C4bl9h^ZT7z0WbP$ZW`&jCMIBgRe4!_1yH~MaJ;j<%NU~oHD~aphS$%u@#hx#RqQsBEv#HRmi`7}A|I*ylln@F=&Tmo$K-V)z?NJGymC3D8sxmQ4r&zq9r6ZLOYbigHe-Lw-%pWyY!0FdT014$H;kDTrNp&d<aVle87wec6E)4=ZcHzC+S|^gbsQ8Od4npwhLp&oFP9Z2Fnbca<0CQByXE)7zPiMJa1B~$M$JEO4m+cT$butZj_)9IU}z=`Ye7#c+dg)4HUif(Ii=Oc7q{&^L%OE7ReBTn_FmFy_^{r)&6D1>K6-!m6zh>bI7)&wnRDJQFe<Hq+mc5i^?F$IY+)PEbG!NX2tsd5hXyJUeWU`e*_@T37n4rQ>j2B&+-T-Hl()f$#rT&yRr6PE8;{yL_&{f%$2|Ns+Z#UcgpFQhj;1B5oLym4tkN$0hcB@E38Tj}Irm9KuD9J_T=g~xfy@y)J$#qt;Xq3RJ4>C5mG1erjzhJ{ffNmIxcQT3N(2YYRTGmXwtI0M0n=+{S;&J}z4q7Zlv6*%8^oTVec{8h5h^AiGKb1!jTPo{QCZyzU}P8oxL7Z4Rr}@NTdJ}US^33#_332U>QZ*R3(TbGr#@*X1$U1~UVi*ik@4{Dud-Um^lGH`b#e|Ry(Yq0k~0~|qVqgc)21oBK4hRpx5dzF@&rz-Xr2BHWPcbog0kblVO2`R*NQ@Tr`<WRwM&UEQ1IGh5+E?@FK`0t7omDZE>G!oc+2HMGzQ+gsxhm5)&9%;%k`^sRhlxaig5Pcablb!a{6AnklNx#_#N7G&wr_km^=>%0G?mUHVKG^wVzZZc2Vn2wFA>qpUAQsey_usQbyAmd>y4zU)%6V@SN{!*5h5x9CbFY>iK!W?)=7qyi;<seo`b$jg?1ZxCJ@ZSxv4-&eZY7Mi|R4Vjn;fV!mzM{n6!K`y=P;rR}}HUY6C?-lMzQJ;!eSwLfp|;Msw4h9qr2ah1_ztXaKr6nu#@D>p|R3((JxLJxAjUuQnD(3?c>i{_6?C;-ORx7DOQ{xXg)O*BWdPq$ZDZt|1uZl?0K;t}AlORN)-u+ZUQ2)Y5<eN>hfbRo~cKJG;$TH`b>1-998b}SB+s|WTbyXhE7rwIRHZoxy|D*9=_-0JISmc~<iV2t&Uv|w2gUS=AgH@-YydPVoJKisB|q#NaQoC5aim3PW-^8T((^PL_F@ZG>m<1*|3;bm8lh8f~pl<C~}Aai5SM^IigMWE3RzYX@ILw11$!?D}Zf+;<{P8_EV`SpQ)+D8Jshj)_Ry=vtif7Pfaw|^teYg8ZC+4jl3<!hzTm>u+WoubJNuGbs3l0J6@Ja*QG3n-^zh*-PUVgJ5cZWDXc+jaN+MTh+>M?dy=%U;i$*0KrDHt_=1f(<J#lgC8D$KpUcN!VESYvZ0%l}w|SsKT}Hz&zSnBI-Qcosol#j2B2H!W$fPs&jjUNn-Hl6ygIAh{gmzPaqPu1~cZP@?TK!O}k0w*ZN&`fht7WOrBC|v720xcC$Nasq$_;>%^D`0VqT-2$LR0_AQyA^SUL>P~^tN9gL2Uk5>JS{s^uEwko}lRpIG}Dej18lAM#ab>Z?J9beN*&${?3T?NkbnrQY+>19#?HYl!(8qycb1(RT(Qv7XBcji7bJ#2Q~N#k&TdV;5?uYA$6OThc(_`KF6f8nR^PaAt;;|Hb2dY7Tb3)qeh3ne^E>x8~PSg&n168SmT?mr(cS@>)M*|RA8MvGskZ4omJ8v2j^c)yl=E^HJYB^Kl_o;|2f90+}V=}zX-1TneAoUOr~MVnmC#!2^-;RETB6*5%xid!#yG{DB~(G4aSwr1~TLHxeT#P;C*CnMEW8IJ~3)0&?2fgPVQ_tQcRF)8N)4dvou*}j`S-D{2dc0?o_#-_PXg=OhJxI^dRSB5Ngn)qrAn8V8I_>E5eq|$uQM!j!jfu+88i{@DE_q$_pAD-IRxt4B&R5;%U2P*2N*Z#y%YKYxM4<okMRW0RyM^t}uL00F_h@0<A`G?(F+3xV{EtAbR%l82#FthFTDXqfi)a4F&t~wIVF&Ma+Tv3kH<F>t6cgdoo4Y}p$mVAH5=UAI+&bYM^QM#^T?gFu&ltL#@ef*^BwqGq})Bl|#k_P$c$S)PTSuZgqV!qHKdGs(~vLF~pTGomO&0wB0X2AN(m~q2-`09Zsa9vG*PjMovj<L?060xm5_*7bgO>}yl`#Yevl*5j`kTndX&FyOEfH!gWZ`0yZT=X{Xn@;#}<49I5*<_3PYe%)pTef+l>qmV-Hq&Pli2GX($)LBLRmA^n`Z3~9w9)$S@3#P5zr{ynvAa)<%C3Y6&g-<v$KX|wHypEyRh*08fT$%PXuDtQo-J-%JzQ!LPoLwMtS)N5<&9S-y`9{3VCix_-Y)A-0iNeP{Tpb#Wyd@7<a5wDAcN6$0iC_^U!Jzp2rM^Vrx{Kz=RGqf90>cts~#e4e-kjNVB+OFZ*AmV``YO1E{2@D19P+7v!3mm_h%)S?J9-S>k-`tHrO6GZWOa)3X6xBQR2nSqhK-JU#$a8HYOgnD{f17F1K1)ZZxe)+FGjz{ieKIB;Y2q6DEk)v&Y$-M-2`4YMi6kfz~(p@wk8b2tQhqvxoBSS;7T&7Z<YgUgpE<_Y6)|>k$GY(*qi)17ME;fZpF4zo(=XgdGqIuJ4-3LAwZgM%B&nx!|$~HH9lCF}f|=(P1al7qLACAK<*UVz@+1?mwulzv%U;zyw{qO3T|sJ+WGkQC^WV1I0nRt?}$3l!*273TX%KYEA?Rd>oK6K5k4Ub$ZX~Zjd5B8iwe>J+W)wWpH|zi}9f1;THvYT$wMx_c@kZl$$Q&qO6dgnlgEQq!o7x<M<|oRejg1j$wwD=+o<uxts8~XSmeiz7Gy)?@hBK>=+2oUd_BP>#y~Z;sNNOw7-_&UH7*gJl12rx;tUzG_%F_B?!Fme7{!_tyOFUiR}Ta`g%=~hxx2s$?-`?HixI4IcDltx9r>KYljZhwuNDWVa&JptSkKu8<_6PZ~pxtzYT7nXKek2>iuw^<!-`u4dAfpAN2MPcuk-mfNG8IWq%$u%`<sCjU!QD%VD~^He^cYGd^Y}L!_w-TNSl!0-osM8{K$Z`f%=-&e-YCCmUy(yN&g!i7VXrFX3?g5jN%3;d?S4#!{eG>xFg@sBz7hUZx{ZXuJ>cfD(>bq)b-WqB|OhUGDY><5^>Ir@S%yy1E78@Y5|9=I1|CRc98y;OIFnbw9H=Os;IdLB3kz)9CxlEJ5MJ{@!Z6qVL*zt$>Z&yv(6$bA_=^#ME=@c($_j!svXrG9S5VTP_QCf54+Kw9(p7dPX4j=QPe4uZ$*CkFb$G%H;JuC{?6%cnjn3&$aU-6vXS`V0yHVYb5l-=;d-s#MF3H1A|{R0B>*QcREZLydIaGXzzbF15>M!>)ct8M>F|A+6mr~mDgW(@XWCZr;^KZ-POX@af(Ly@V_nexi(j3H;*4Ul`xI(Aej3Hi%EC0U2`?1A4;^@_2<j2IO`v!`6=SxpO&81p2*0SZZt;x^sf|b#80wxd9@ziiTyUH<xSW1*~u+_ylUGSyXkj*G}YdZcyR6h4xqwvmC>^y3X{p!_CFI<q$fTeM@Odpms5T2J_Sccbykzk&E9zTK=dm1IOVOzrdrb#y=g(RK%o$)U99Lq6?7#zvHs+xQ)2Cb`&)7r$@*R>FpBo4o$)$9!rKEn0{swO8t>xO{55t0$Pca4aqc^h*HWKs2H`GlqR{E0)q8QL3tk1cmfws9Yeu+mAldd+7^$dutIhd+3O=tTv)j6)9TU5}Q;e5I->oKd?g-T3;#!?WscE(>zfqH(r-dGPzbSiveZ6J&Dze9<p^a92OQ+kj+4nk4dcD_gRb!+Eu<cfto95ryF2962HI&t2dAIbIjV=LJ7_vi4mhxdMSdAYX7%i9<YkgsP`-Rt(B!I;pm94gKI9c7r`1U#rFf7N~gA+~vjq2@WGF9(9jtbEG+TO~d7AGXBD|bfuF8^QyU;Ox4p^8@PO$A+KB_=De@vU|BIjb}V^Yos8>V&F<_o&ZRCDAPE2(+3mZhxv^Mi+Z|i!Sf7F}xef2}gH{&Ea#lv7pbBm$@Ah4(-+WV))ZHh&5Y&XH0uKlfzpkj^3c(G0nfxR_?aHd(XTb%Y<{JWowe6f|FCNz9fQC(mVFfDv#3d9?xPmpw*`Mf4}&Ye8r!`JC@s32K^G-M!}~#WxYQzdo8?JRnZGMCQ&qo(D{wBEM$oYXmb@yw@OEUl<&p+b!qf!EojwWZI+jdB1RLs%mS@420tn{0}tUyPg=zTS~t@0+daOV*#<?xejpH?qM6^vRrM#k%i)ZW!bTj@qx{iYIuBwXuA9jtKo=Eh@;ZEKXL5Sjwj6uY&5TR&%HoKTgY(gkDttBcX?_{c8^&kw4quPu?aS&n9^U*lE%(}{$T|0{(Fv@Gh~bu<p&%m_k*r#-b?)z9Nfj);^4cW+8mx5)>4>r1Uh%Lp3fu%@AUxJhp&_|kr8Ls?0BLDRWkb}7HqgyKaXD*6-qb}i_VPwlzF3`%_VDyWln(WJeYBt32r_2JzmRMuY>2$e*-}LKJ88Hc|6;vYlE!)vXY&0W2~*optQxFS8m+-HbZk7j#42mU6~bSwR+`a`zL!P161Ss2ySd*y;jSfDhI|oFwn|^!Z9ohtg5rr;O>=R-zM0KP+7D!4PmdOzH0bz;p#feXGQFLfN`>C{@Rmd&#{)jBgYX?}ApIEk&ZAd58ma8&a4i2)6aszjm6Nr5+w|P)If80l@U4%=>H+1g`33(COU&UXZ6IrdTL!3)H&@EuHcmd-fD}LQ(JB-hH|a^?(=%KizC!Ck1fA!w@4R8Ass7!3ic6~smpDVIYjGRRErzV!p0+9e5Xki5W$MI{{{{&LQG5DiK3BR6l$&hRFW;utZQ9X|zSPp#^_kf&Rpev?OZxQuLtLj0wE$~G&TCC4NzT*jdV5p68&o%*!h7&3-b&474`?|28yv4$-I6a<gjzk9q<<Cd+CzHV`FeEjUyR%7t~%KL!p76a^jv>-wDIFerM-aJ&tI)#0KR8J|FP<sDnFef4}iN5xr~0wC9PdIgRiCr_en}6PJ=q&`{9z&jV_u$?@jGc=j7R466ShiH51(;pY*$<Dgq<Kkt29OdacVzcdYrLu1t+}<tUHpa2I@(BXNJU*XO!mJ@erxYE7yzlfANbdhVhr`JykAl>x+6ks`m%uNc>U*m;z|&R;qlh{&a`$DUku&w>tE+2^JCgSxz1O{b{7&<w3fQ6pR4YVR(7-+RTn13Ip1O@<wM)C!bBt#*}bt5VDtbwKP@YOSo%Z_9~&eki2eCGJcq3!4KA1?O=*mEijN_pWhU>(YndCx2@<JN3Fb2R$;L9lzaOz{)Q*$eht=M>Q6vx=_Br{stABW~ECh@80BlYEK>PTcu2C*!kdHt3w<XM9r)PscqiMD43?`N1+bgasG+Z?tOHt)Me&20s$j_81brk3t(IjWuWlLkHpO<_n=(|rT5Y>=rfc$rWkj;*qBuPd{-{~ImWtjvzaX`u^Y)MW%$nVFKOKKRi#6^9f>pRq)>jvmx9(UeJIX+!QABIDLGsTQ^Pzly-L%OxrqA^EOL1>Qwd93O;uQ017O~-zh@L%K(93dJOQrdT`TW!&grc^`_b_&paMR4*y$X$QOZRHAj7`ADn_cOwdm?`dsa`4HnAM^e)nu<g^~>ps(a)5H28~vsLy>c*kNvQ4%50hGM4i36fG%s4XkNYX?G^Bw!SVp>)l_7K*g<i7-+4xgDmO^KWhe7>$#I@7>fcB$~yfWbvObSfqb}U=%6myvTk?&Bq-v<Qy|+?Rlpw9y57%~-BaAWta^YRYS{GdK6CL;@`YWea^Y(YtIDc}mxs<;=?AG@WxJ;hchA}o$*$emd9WtWD|{@5E1Ebu=gQ~4mO5jtn^`UC#OrryIxq6J8tj##@s*f6Vhc=#yhN?fWWFu;4#yt`m)VmDg;RLyHMz?6*jVa+v|zUD4A`$pdIH$1jr)tRas3#oKVEwO60^58=1|a7c-AwzYV8=uq1^b(j&Dt;0%+iR?er7hB)u0cqpUnm^{3pbA_RI;A0vT5B>drySJKSwy}t*gwB?g}uh8xayx)(^U&~>Q$tw!x&$4}IRcfj!y8c(^G>i|;JEYI3ucLg(ePKn*S);Y>x^7RaTsNDMB3iL%UOV(>G>tlqZh@X}_0bUpXJPwOg}N9kXZHeV5=vy&Fz&J5@$leNyDXnp0Y<sJ{v(O!b_L`od8A3!rk<9Z`Yg^TzHUy8`mHx6C=VG5X=Qi$EdMH@gVXwOEvr$(ET?|=T=dWU{TldoA|wxgmBAS}x1VIpkpp6#xA!sNfAii6HNnA`Um0%Y_Y$qfth<TwpUE!qn~<iD5PNTpe6qgGU@)1q?a}r8l!Gj3`o|%Wycq>T;uDzNkIHlC-}mM4HHLEb_4A&c8}`8Jsf|VfYrBsZkS!Bx^!iPHYVfLQU%lJg^xrKW*IN06*nlQgIgmS?$Vam%GT-KYiG#jdWfPRyEFV39wZ`-7Tj}b;^;Mgv0}u#xQMoUOt>}u!bqh?nbPINBUw^nSAp7(g>i$U{Cw6=qw22#){V~^)K2x8Z<%7dW?0%oEdnFT6(Qjl_$&aDM)tatBhpqFzBu_6S?n_Xw%LM&sYOf9}iF6QmIvh;<^AjYi+9mrw?JQ!(C?M_rkkJ10CY^IQk{=VI(;qaptwj2Muaw{qN7>G-6QjC!zMepaJ#)sQ3^4Dj+$0`YUA0=yGw2_B{kg#9faPoyctg(kELta6$2txMM9R3MJv-CR2Z}qbL2X8yj#EdTR&Z6RXfpt<4u5dNFr7{f8z9@UXe*Op0rHnpzJ@x~<GWdyKZxe>ja(SP9M)a}rH$}3s?4M=>Sf9qtvrrwbmbZ7c?9yuzNpA1l-{tfA(mV8)9E*qozw82!+4O8<!bWqto4=bbz(W-@tEyYvMTQfkJ@;gm=#!^UYc8X@!+);<v)Y?7eV=dw&oZU<kX=+UpRv{V0T@p6H-&b)9(m=o7q^-j}Ex;!e&I*1&a&oFAZY*s_f$qpS?CWrkyFZTI==rD-?1p7~XWGYWZ2;1__?GFVrzx;Vm7re%_6_saLf@evtZk0Jc)#N_7ClsitCfJ%?-k3{g4cE7|#pb=%S4-FMiw_oiH)TkMW`b%nm!-|ss*E?!qW+8z(%O-dieZ0FJ``7irz-1`;mGblzKe)@Sa_3-<42}OML?pr6Z)7EG-nCm-BveuI=!z+vHv9MQ-)myP9TNxM6<Br72{EHDW2NnwH{qA0=akrUU>CJ78EL5fAz4~H`N-;t=_T<lh*L^^4j!*QxkPfd|`6r4nf2OBulOZXw8(q2nb+>yo_q}PG+@hZ7ywb09tG2STkN!ykPPe~Yy6*j!RX}Ga^B!}FDaP6~`L`sh%G5${(`?x72YO9>JNV^2frj3r_w1mXXG7h5(~4AGleH@MklK)60j`*F#Qc^R$p!xsg=aguKkMsiLv)q-U_m`Ix^a<l$YwfS3WVL8?KF(0YdFb^mLf7B;6?xEwjSmJSz-hAm?i0H8p^Ym+U^3{vSqc*8~0i)QfP)1>%HhxWjO6#{$37F&p+tski)*az2#L-C)RaQwK%6~d4IXW3k+c-3_6>xzkO%@NdSvV_9c4N!uYF6=9S=NP*A`9&zZ8bln1(_)D)z}C{)Vxro?21Iq$69o!0NC;?(;*o#nt!R^6%)D)XVmciabexLmwJYdD2pYx`H8irT0>J6^G%_0eK|yKMiswT}LE0|g0C@#_=UYWv#+J4anmmy-~(9E_)&2-EI&9@OqLXh2-|EZ=YGpbukR7BuI7Kqw5Vb!(Prk><{*HF4~pu14eN?hA(QrWfiH9rLM^qkAOB_)|r=RlVl&J%1Q>&{_;RoQd62tO|eBwj3QV!8@?<-Y)&heyLGfI(G3rw0mxtc<#M&AZZwNyIFX?cNsU6;<1>M<*oBu41{>K1ntIprgxV1M+&|=iE0v7m)h*6RcKm=b37!}k#I41*0ts0_V!ij-If1fs?Dn&!JMbwchQ=zrkx8bIHR>+Q4dSNd91+mQCzj6>weLXcMa{ncMcW#_~1_zu1dcC{%Wmn)PRN!=i9Qf7u2DXuhrTXja-INdn!c@mh5{s58b1aOjotAAw+F%z3B7llqag#)dF`rtABsb-RgCkr&hN-8^~xr)PY>EIwoGYmKy)BRX06B(8pq}!L5zAO15s@(-rBsuHBpLtKG$-a^$j12_*F}+mDm#*hpsQPlLVV@X&d%f5q51jsLgXhP;u@Q__!I+*P4Q6X<J94t+y$JO~KYRmQ=E#tU$BU1;JGgmM}w3KdS#+IE{A72%Lz+v`Q0$O^+L>0q`{_}TH|5`GgHv2$R!TJLZdn&lv{!`sh5cY6=mO;?>jg+a<0+$NSbnX-C)KHAJZcgWZ8R_nUVSNp`rCiY|~Y`_7cBe2yDPT-1WbiJ3l_)<dsg^syb{d1~OCH9`iguxG@=9LF_w_M%`E*C;w5HwCfYs3Bo$1vDwW0lB@gM3-X#gmz~5=uDv_jEhRRo|LDn#p2#i65<gS2|5#8GCGsfwCq^JfRo6?Fg)B>+eN8v7I9W3Crp2RRdNceYk>g1{@QHNIOVBxZ5-%Z(&1{Kp6LY4xB%0UVdBBxd<O;8qYQM9?dlukKx#ua^p>?W)~_G`!%?~=)!(`#NzMfhpHb=6wyFV4{`XJuDY#{F#?#=5$|U&V0`xGkeKVOIYbOXz6!sCfs<D$6dX#*x2+0FemX$({CDk4xUBdB{hwA1{N<sF5A-VxhQ`w#AI@P^?P;z;2KJi=(_gDMD+kZ@7UbK(^+8M0e!pL%xq`V6boFGndKfp3Pf;HC0~?0_ueI-PQXEO%{(nCO9Hu20dWDAp;e_`f5N2W}WRZ|X1`nV8Edgz3?(TOHZ%1^mC{<-;rFrs6v7zarOQCHwvc6oq3mT(l1|z6UonSRDU1@n+Y|l)L%&ZwZ4Vft=Pm3XX6AN)laz*h%Ihtst7M8(omIqeRi6ECAbwxRCnEvXj(l@k5voNr<aodJ_jFLsq9DW$aBN5gb!J$ofvE5GSigCkh^R9h`aJ_W$@2aWlXuZvOb0F5osuN0QIu4w20>r+=jK}?k8kT!GF(2L_wl9gJH>xY94vEDUM0#?tPRT-5&z3H*&|9@<hs3dlQDm!eo>D>oY&i-m$&E2Y=w!ye2#e#43}n@-re_Xz8Mdzrcqw8A6~Bco610)|waGH=v6!}@g13zMxL#jQL*&vY#8SJQoI{!lTH#NPqz&Qq7E~OX37l4j{-)4~;j#KS=mV5@tL2dOb&*!AMbohK65my@?Rn8iQP(!kZ|jvUJ#aFFT=$(WU7va^?T-0=F0{ta>bY_YaoZ77$|Yk(leiR!w#<?_6r*!f9ZgbgWtU%eHQp%e+?EcS8#Fw}<Dsix?v|9~<<d-tE1`UD$DCK|@o0SJy~_z&T-CgK`pj6mHZo!%d~2Oj`PyP*KygKyaOD|QQKG0X65ovd8FVvEqs`SZA1h-^FFmxcTy}RWc_W8olUPlB6(vOt#A`OobW<oFY+3?$DMT4_G2iBr5Xzqd-NLqU*aQiF$ZL5yrtB6wF0gZ|!R)BP-<Tv(C`7JoelzS8S;E*H)JX`sRuB83EQIX!AzA`t8fwI&NMkWZ$+I@0dFeWeYlmH?mf|k~k_sO-@uroF)WeC1r9`{))_U)~#IyBqI4svQ1;IG1^VWKpGLLyOojry_qSij`;uu++7%h!y)v0<L#df!bbE07>!YUnMffNp#fDqIWMv5ki$XZPVPU*9rNw6!NK5d8zhvxPP0zaMioi>(SU%iNJL~T)?XENn5;a`NN#{zXY=nE(T<uo&Wh$iDULe;jLCc>YDGdzu)<PN%xG^Z`joW#e~qcdyw(2Flk@$=~>(hFINw4*n(0aR>OIEs!I%E`mYAMvGHp(3V-A;Zqpvo~W|VC#o$VRo$+@d^?Rb`QAN5l<#vs1|x98cm<Zy+Ly*q<j^%5yj6#Z#KR*)ggfF{szvhz3JsC(98%_t9i7UFopDHw|-_gk|r~9e&9kgEJKd^S94rgwz~ULX`L#xiY+o75XG{d4sg52J8RgC#e8{ZO=O7M<rNwx7VN@DdhzD{oi*%SmQ(k4)@;^$!u#LfS)(}H>LCp!wE!l?oAq1v79Z7GUa5UJ-MnO{c1w?0B+;}F1#w(k?6=XqAKA>}vOH{tN>tjvfa+vifvd;q4owe_?M7YIuyP(WXSdX@)GM#{Lsslm<3q9;YNB&vC<(}@h9JF@fRrr`=C9|1ST0LzE29vJ@Z5>i)Ui`qQn7q~P2%B5L@G6<@Mv9Kcf~z9(&2eFFRZ(q5jqG|Juen6a9Jt~N!m!9L-Ea;%1|RTsPwY!WwbX?%40e*W{FsKooPZF#;7BufuB9%6Q92@73piTIBXPW#p|VXNKDCq|HkJI#Cqd97x#rod_^Xc{y<KX)ls*nT$bWFpkC6pR!d_(W^yL2jTqz_cGvL2txaHVmxR<>I){UaO;$qTMa#D6Y%HD2hUC(4c{5~b(J9odb+>jlj57mq&E~D#Nq6gg8{R_wNJcb9X|`o%F(h1_9vsm3XG%c=o2C5a+M?TUoC%fL!|15fRU^KIVudUfTP|A5E>X#r5~V1@7FJ8My=nKfP(2&Dm=T8@MuV;WCgX$?1BF&e44*9b%UqZ%BdJ0mMjC0rbob^gOK4+pEFZ`&eXZ;Yjg88T{rU1-T}E1`^kUG6&GCpDV7tfSh!7K8i#ezJu~`nwHI;rntR1!l-e^Hxa~;^{D%RPXqc^*JLt9qKu<Wdq^0|{rt6EqMsBrw$0)zyvw08PU$(=65(ZoiRa;bf?76B(QsWM~;N>CRik=$U7ZMMAh?zqG<z9Y?aVGjCbZz>zW`%&M{-yMcc-Y0PN(kyIqQ?%MpW^}k?>^2p6n@d;v*VcvuohKRYUDq)#Ju$YEL?L!ijx^;D$WA(s4;$I;xxrnABQw8G^|3(^-JVuv&fyw+!e{q->2%%lF0RQP5!w&+_Q)LW$KhU%xP-j+G*9*zUTxO!idYp>8C)$ygGxHP?G211)~Gdy?HlI5iadTS1~3BK90ImFg2L0qy4B8NbkinFq3F7$`Qo`0ywK_F4xZ%l=~2e4l&kg-%d1-@*dCHcWpN4b^S(<*(1iL4k`Jj#s$S~1uLUS+opzODem7;iYCa<cg(M_<yK%i0>&4pTxtUXPY6~eZa@`_Lmpj@X6Nc*86bdxuOCz0r_RJsx#`*cSGcu3eN(fJGEv}0d;|I`8RnCn~Az5#wmhnp->zIKNdA(ShY7#R2{#cipRgK^}8>nRozt$4BBZ%V%u@xy6$Y5=jjaE8&e~t~eikR6-)W!#g)>Af;Y9J{?n}k=*(#(zzlKX8eU<aFI>`*s}$?ED0TE#1mC&c}9b8N*H^;4}}k3<|~KI~^At2vz!^;YrL6IX4Y6W`A-gUxJ~I^_ymbFkWD`*ld`<STihEXI&xAhWHJNeRn0^Lt3s%2Oznm|OtR=31vkqPk&)>TTQ^t0A=rDAqbCw`WDpq%=Rp4vcnE?gv@KAM^4%e=nbF7m}x*l8trGetR@3iuGuQ=LUL?X!ET|OX_43SO3afq2}qtu4~95lssU!=y)5S0oGp}If~}pGAC~#d4$lmASBx}ZEK=uhgJf9p>sK@`lzdCsbVB!Z=&&Ky9oGre`TUW(tE&8flS7zM@kZYh@&*sD9<;NKaPuIpE*cXqYa@ki!Jc@W!j3i$h;BX==_>Um(Rue3?8RC8_U0Bx6pZ4Nb!n;4J)Wo&g82lx^f0|+sYEF06E-c<VI{pHF}ai&sI0JGglIcSuPr*6uBx4=2;1W*n^^@=u)J`!)$1ILp@Ty@R1?gkV~@vMEljieBcjVMJO?!M2Cq{q+B~fpt6^>LyQ#aJzAdCdhBK}lUMRg8pbaPLECImEs15&)a;-j$)Pw_n8Ya2k5tG0mKYB7VKufZwG1?JqL)pha|<c4tRF>>@s6CkO`&kIM#jBJr?$O`ZHV2C3#5~q6*c;J3M{qB7>>z??q9DtO~m${zDjMm-kj*L?2yY4G2I?R2ox@5(r~4{D6fgWEVS9_u((O-u-(rs^Y(69No_;H-K;sHX2Jftv06{^`ILXjoXgE|v?1d>*D7WF)#8EkY5cJgkoL2Hor43x04>vl;JC0EUfEhenpX^}=cGDq6e_9_Wm#r<U$=3ZK=^2dw{MBlICBi0%J{^)T;S15Z$$(n6^sbvtmekN2qm~=9K_os7r^$*kzc;za%Ofc=VG@WxfTt0!E#egCs{6;&}2W>!x?DTI^jJtEM96Nb8)Z>pSmQ5RyKpR6Bj=g_iGDS)9dk+!5jjbuxYbPDdx}YeBKm+_SJTSsI5pUb{<ka|9n&pZRK`mKPwu3__W_A3cW_+z^`VlNJ;Of>C(o}T^1+0T5Rv~O?5ag^qn%?_uFYaJ*JZLE5FaQ_akuvMYq%TIb54kCLmbb)J1aUJ*i$w=59%=mC(idsHp^ZXn5am1gDB{2}#`yyA8m+Qxq(j>~%W)4MG;6i_ODy@VK7%X4&yxj2;;#kFci&;AoNHK#y&eg#_z(3NI#OyJcF~wvTmGD?ohLTPg1tXJ&P(#2~UsQsTw17ou9-1ul%g8yg?ftI~{)=a+d{?%SPZM68zwkWejR#p~73Xm2^k%Kk7z)I)mlJSmg4L@?Sv<~GBoJUvCVV{k=Ca$PWp;ewLV^0K6~bId+aEE+Ke4&EAUUX3HGN$FgLDPpM>=b;K7J>}?e9Dq{|Dc)vtCKL;cjp@ixHsjeqt5$sNBIxUpR_Lr(!u6Un9Ixa~<PgQD^kK9r`8%bE9^I%3C>U+-%HHI<y$_eI*d#$?!zt{;`)NWe?tEE#5e^QcPFNFrS#NK=o>}8v+7^&_$<BslXqX<ajWo2CN8y2UNTj%I@DK?^2kAYnwrsvoI(7HHsodc60oFUqu52k<?v&I1b1o4Ij(Mzm%JRt_UL#}J*bI+q`z}7FAkMl9^J6w2P%~rel9)t-7@r@gMnP<@V!M%AErw?8NIF}&jZlS^L0H81&bCyooTvofx`AvBP?k=sbT=^XZp*n4dfCsmmjI5hn@t$0@S{LtHm#<xGOvyM@dCatc-2A7$#ogJ)30yDY;0RoNA=M@K40f&<j~5=KJqG;bDAIS`g;i@PKujCt<giT>DD3v3+1&hU3DgnN^cfhQ9GHATDK!ef#V@KL{81ZbQO$@azw8+8*TjKb_xjv4$}(f^-UIot(kJBN4guXa*asTIVyOe$xF5mkZs5~2x{Eo33adMk?L|1B>U5jUYiuETRw3hW5sdonC-xw+6jTuD{ddoR~y|z5ZmOPX|{##pp~8Jj|bfoKWvw&X)jnD<IXgmuI~ry-DY!3!kCWbe5O}b0mJkUd;@2TQ1qqfo2X2jjfg3AN%pF<wKEHFc~S{w;?&LbRwpxUG2AputW#}uXIiS+%M3%j5zv`!JvBY}PMNNktIQ>t)x3hfNRQG6UhJQJ!Sz13Ly@{dhxK>__ai&$yc$`M<WM(3#K`EbSW5^UjS%)|H0_sym=BFGtFcob0yZ+MCS$m7XQ5kTRth%@Bytj)$$a+a3zPd)ci+fbK|Gl9GU>5yc5c;vqZ&W;YEiV9)3UmELej@jdKKyoh6OQ5EX(m#D|>P#PNUVKiQ({=)$o|nH1y3Rh%FPdwK@oryIv#@=ef?5s0u-`)L>AfcZ?XR&@^Z|8t31bY%^E&qSGz260(rJXck5?fiw^*#%G6=o`aoo;C#fd)6%t=OAuoE?Drn3d#KcdQ~Jqi_9lTttbw#Q*uLU#!NGGfqe+m?wCFP6um$PtOH>-S=KPW`9mBO+wA?;HXYr8q#{;1wy3ecEaBOK;r>M6vbUNi?CopPk5}_+*&{@si++a1q6X{NIRKM-(sr+h=_@~?c(N_y4(q!A>536%^o&wEK=@^it`q;;Y7@XNvuY0~139@T7yf*M~E0vvQtT<88&$-@c-%eJJsiaU+$1z`j<(&t`)D5O-au7}vSeIbR-qGzMO{|O+Ij8)$>dX!|iDrJY@63s1YXT>bNGXs!9SSzK&cbVo=~a$eWwGZ9(OS#In)PLRlWB;{x^LPpmQzqJybz;ia@o_hT=%>cb)F}Pf&|)5=~lnBVwzE&$g_2S*h}@+$$&p|)gnXy9o2lZ1DA^Sl;OD^8P5vSlRsfZs;5{8o|zdUu1zs(UKo^$Sa@+8BbRpMBzUoX_A;1_{jjk${quGyo0O9H{J3bVSe<XE)fGhTPjOwn789Je*<WceIX34?GaHI^FAPJ>ifY9PPi}Cy6hpQ*vQjq>NO}@q(t%N@*H3Jb@+`XA_HuzVL}L+^$o7MkU^7>{MJttwMILn_Bra=hBv853mP{nmCGC8*1DR^gk3-c!Yf(z~n@!R?>9aB43=h=Vh;ZH*ivqIE8gV_GE&I<-BPkuXh!e?=dUTuFvem0T+{5|Crqe$~>$iD<D>XDZ6!Hez!OUnfbb9f4Zqx3DlbJ#_DEEkUSrYfaBG|0CZ*>O^XC$xtXtNkfBuk?c#vp2V+v9xnL@q-n8#bE5zG$E!u7r&hpUoW=DNgQ++OeS|u~0O8F6oP8bj>G1@kNWwEVe<IO@{F7vCydGmR^(0T-6+1nyH%uTUa;fbpyGMiSYcQ!Kv^hZ|7Lkf~M|^Va{lfE~SyuwAs@AM7*%O_LLml&bQX*5-Dd8D2L4NFNUcd!cnNBp;1HNP$yJp=t$}QVwkn#)MkmMzg&5|7)Dyq!E`<c*qN8a@|gf}>bHBtQKR4N7H0i!QJFOuJ(2V-bK4bjR>(|UJ!GrJjL%+Tixyrx9E8>!Po0ACT_Y9D!xq`od?5nXBK~Zr?ob7Crq}0!fcMXKD-l*|6E3c0J6_6ea+|&yigdMMB@v}MA*DU9_2dbkl*+|qk@0qMe9<%=_J%dj-Mkr##BaBNu_vxlmp?@plWAU)^S4Y8UR3ilM9@#`15zS1z7rKMJh^cS%nUtjjjbh1CeIm2g7x7RdrTcuP*Pu}Muk>F#N)Re=U-mVEP1@4=qjhkqZs3hqy5etTW{LPT%`%z$#<arHqj7?=CW?rs}`FmO+#543le=T9C5nmT<Uh7ICNoS`-e1AOtpN6xvdTsl?|616IP|w4YUfkuuV=Tfx5FQQ81kBGUa}Y4k+DzCc5tIPeL_&=ugIevV>)1!1KC7s8-0!d?rYi3xmc2R@quHties6<t(N7szhKXqAWt|;>}U=1AnSNFqVt;@|Y5$75W-mRr@VARvsV9_jkhZ#a*f3wX|hl!7jy3-C3Sag{#CRIrey+alY&1G-|_)vLgMx>!i_v@ok{K4;$kP=BKb;l=d5uE{0g=p;u;{G+G_v1OupBc91Z9{ZbxhlS6%Y-FLi=PDxes7|rsVx{!-AeW8m_NgmpzCWp9n#AuY0YGQ3NC&m2^u^Zv>!Yf(&2+5<PIh-es7-3=cUORPx=DST6L~0^(Xy(Lt4x<wo8Xqp^r6@nv<y(Y0l?UuK3q{U6<`uTFVs5?jwo;2-;i7A-p~rgoskWb8T1a(L2-wlZZt84%-Wzqq3$~bs0&2EMOa|*{HzkMl{HW{8m8~tdBLd}Nw-eGW>CzF7MrFdTOVg3Kz=*@ZOP3OZKs6Jk5}QS%ol-BInvBSmfG>q2Q~A)<Ok>u>LhRDwB28j|4A6dlVT|LKv7`4G|NanXE<(C~4A^NE>U1K|sdd~d>BTlCrKxl%0QpY&Q8hO>p4C$uB^H(%20ha4*z@JF37(h)H|MnNZ@2r0N~Pt#-Mz%sL(Nfh6sehTd?B71OeZ<>)lW>$dyCj}MFZH899pr;5!uFj2B9Q0tHT$lHh?8FI=oy;xl{#R2=T4JUOMeWyLHGK<PGV|L9QvBdkcz+;Al2OqB&@FER~Ly9XQC+aJiCGGIE@x{X`?bmB%vLG)c#3pvWe~`T~Q@7R&01bgq)_4x?kKE7QYpvBH;+E3^O~tbw_>wGXp=xazA#3t=_7EJ*>?>F13J+tz|uxUs{gyji&J4!Lu)UG#}qyT~wC|B5ay%`{`2uI2P7#}s|*n`2jsU7YL|mgu=!9Lxq*qA?~y)*yw~2Kk&<trap=Lv_0K?5W!J7nTj@G;eK(QR-CM+FWeT*1O25xYsLWZzE(*{T`c(F_>^`&Y<cN&&OuoVT@9bHIANC#7$$9X+woYLaT{`_KA$l^WruFAHqbkK3$%@eD<=chN}1^axlDQ-6$*~K6L`C;Z9GM8kq@|SZw-Ysa*0744-nWQ0O+fnvw-NcbLQ?yF;;6pjw!aEp!uw@yc&lej<@fg)c)bOi=Bn_v(+EgmpG21DOty<FPszor~9)NR|n2-v+H!cQ}f{R7|DPf*o0xoY?p}DkiLbHaJt)yRdq?wW<R&#sw^U=+|z`UieJz91|EQC(=}{vY!+~!wj9p=&Lfz?9U+IV|}N0%jAx^VS`9&BQw}RkGDXgaL6sK8#Eo}<N}kLsLAduO{e2%x_}wWQ@yhG#tHenO><m7SV)EJez0{}1r|<vy6KX^>aG~{m1}Vc_Bmc8or&^wE|4Nao%o@#B_|1=7U>_OksMX=*IK@mY}FaKVe-4sLCp>AO=iE2mz+9D;yFT&;&K+%YsNaE9ARu`#;6oS^o}Kw=}WuXp@;gLjXFa@ql%vHlvZ84OC@KMcrm`o@0K=yMg3Ex+SpgodVpORbqbeHWp8?$^D+8p5D!+pm61)#qlQr*!0Svkyb?Ne;1(Yh+Vr8jwDrL_+A=EH@gNTSd?A0mWyJ>0vGo0QFivI{4=$H;qFnby*7LqrWBc{=H9fW0c!_c5jR-$G&k~(Vizgj<w&Io9u3UDSm842oMct1m<pVv#N_Yaah#JkU#yx1zXl>GxgJgY$&_YdNzETHk%>BdNI#`!;0ol)AD%d2q<^`3n%8Ak*DS_5V6)%3gi&PecQmwPds@+1TiyJ(vRqwBZ3HVCwMvHCIV?SjCy6JvK)0bvW3~f7Ac9(8leW62=?~PaUoKn{WIDF`25F3w0Lt;U!b|UHEsT6RcS8_k-lx7IDYGi8WRxL#od=m^K<T`%ZFvn?7-<CMM+0Gi{Q~U-Y@N9Yu6hSjcKr^8-IF0OVxtT3o2c0O#QRh>lZ1*w<lc=#9zFHuU)osbwg0amsEL>Y~GA(b`q|sM{wwa1amo(PQq;9)tL9J`+Akn^%jX72zlDktLGkQze$T!NW$b^XMRO)P()p8)W^l$lG{l-WNVrozraNM*pp{J=Lyvi{lGsvBiWS|>#xXCIU9HixC)~aIkK5Mnj>6{D3f+0U+$JOER<QFf6xM?@ERU#VEnI<;o%(+}mp2~GEv1a#H7>&d1Y1*IHSSm1_C)53n62mK0JL-herN*$MN~4UkT7-AS?r0J29@Yfjy$b14cPi>xE9)ESgUtdUYJc3BuY9s!s^yB2_QfBbn%-725jeBPqR5;W$Y>_p#Vr$`6;S`l?`T@S2(v4yy)O0AdTfguX+fCQ`7y&TJAz-WKpO(jw8*SKBA|t~cbM6)4#MO{_Ae;r#i8-a?AFF!lb6G>nLj@#RuzZ^1)Mqe6Xj#VmP>`WGpnDgD3of-$&jY!uN~anGehCFqDB)!M7jk$lCLSvVyBp%X*2G)S}p2^-tiv&V`g-jP}8OyfdXQi-EP#(B|>96q;h`~jC82&%J(<HlyjW~9hyfDbPwiWlb@2=g&E;4TR3cO4r-U`UpkaxrH3)dZ?})Mjp04IsZV@Zr(Nn?y54;j%_Ph<**$e|dM;UJJ1{>}A!?dS=g9`+teR}Pq~?)5bVCCELbR20Qlnl`>@`<``8Yf6NlG%<%d94hf3U^%T7YaOcu`M70y)XoCT2(&(7Pa|c5?V`kv;c!5ZWra|4Is0uP?jNG41{z19V+b3@W%{W6495kV_MICE&3=#)oZmS`HM6O0Mf;3OEv~XJex|k&HBj#--=Wgp~ez;#C)kLBm=sHtle84MEXDFT>PV-hu|(?5W%hoDZ|0kKt02f?iKd6Q#I_j9Qc9Dunm;+=?Ghx8Ca+%1n-=C<wlM>arWI@W3wVMG3=p`et_@LR#m*5aQHHJt32Ey)#{qm$aH4PZM3K(SY|It`^j>n8?-wB3GVM)?m+b+pyEKOWi0FQsU;48#lasl7M!GKRpQVLHU;p+S%9+q~gW)#XATBZ5M0SU(Qo!yxqtPRJ~-x6M1;8)Dj_VmnhohH9YAIy_Zq|L$0dW$X4q#8>H8q#gbf+wQw2s?`AhmbW-drfirEU&)NJjxh`GLq!miWXK0=*E#=_?Z3oaXSxyY=L?^Scbo_W?F34biEE0=Gade6{XA^#XoR)W45=L{qe915e>B%X#yfw{8r{0b)7jrQ_aIVVEm@!V<DdHkb$#V<aR$IDuK@8@Wvz1jN5-J`f%v(6@){J>MP}!QWpZAC=8yi&Pt9&YPGg|$gAI{eadwjx+_4a7cY#Gotg2d-b?XoE_M)};M0(CsxhM0}lM~xyPRPC8<S*e?y6qk<On#)tF(UQ}9<2=43uk{uYir3p?8?pT{CfW_Gs=?}Hl0mu?t)HSrD&Cf-alK+O$=pU+TJmx;h^ufhdm<!iylBOJ{7RDA&28~S3bP|1Q`!6kE99X-UTx<YUt*2tt9l*p^`b$0Asw{yxlJ3P3>WOg=xvA>rBQ%x>!(X(lE)MFys^)n8wjuYuKLy);)!aAYi3oglHurmFSu>XgC>?<<h#_0UP!G(GrnmBQ@;K@6VO+E&_ra@k;&e=6^L|k!w7NR8X(7MU^Z9E_Uh2x`5Re#b|MSAKHN{Wl2vakx=v{q2D)w!Y&>~dCVawramJQDex#+>M5g6)vIVccR{g|4E}UU~+A^k1RhzFdn96RqD}N(!?UnWRN~NZh7b^_R5Dtm+UGy9YN~|%cEHHK$Wd^!&U0H>b5FL8CI2M3)IApR@vDK?PDI!&@p~T#n&Wy$wE~Jzlqc1wOOx@=@mr5r^j_)0~NENc1<3Ngr8TTJbVWX#lfw#7!Okq;>Zw7?UR?Q4QJ!qw7dm4;G*Fdg%iOS*REPXnYe%TJJ3X<dX;hoO3?Uf=yLr+!v5>>M1r`9+dmM6@1U<;PI+8rBVUwOyP=Ki*ZWQ<4@bZhc1eMl|M-C$Rpg>QUsS)BE$GiyV~LxdgU3ayx3W*}9(@Opp??WSJqHOSE=zaz#yGL9Ys*P23eS1(biqEViK;`=o|MCbFYZ$$=013t}amrO@ROly^x^+o{hb3CG;4W3~4INs~`mKNTc&)Fz<JMltdvN1zq%l)Uc8?h?@xMq)y&PVyAz#sbJ7OmaRyOS?gB@T;)i5gsk&IVI4y?>4^qYVxbM(g7s<cDudC-DCM7wYV7nMQEZrY}_)H`_a*Z;tCnjGUF!4l_X)=U@f%YveXh-zdUYJjYkUfCZ=+_T$?`wNdn;#f3d5eQ2CrsOu0?I>!9N`}<#(uz*RR3n>H>jiNSAZiADRFW*Jh#ioCqN2wE!HOEB7oM$?_>izvM`c+Big9?0vTP?nv0~oa%k%`!C?ZDH8l<;naHhYdzsxgq`i=(bwrEz|?T&jzeuu;=7ggM{Ve%)TgFC8bl4W7o`EZdhW+2z6UlV?5T&y$GLjF_8-;CP9IuiXoU*aSTo45(U3(hCfGYQl0Xjp*x3bvx*t>bFF&+KnBNV7`q&v&d)(T8*2RuS^Ew#@b9fs<WK>>i+PCAR8TIl*T#)BofWo3k<`5;n6tN9N-fQA56zs&4YR<cc?P$SG%=e{}Oe}{?|W${p(-<0;3*<r+0ir*TAo1zj{oRHr2n&IZq+#QF&UkJnp2uq*<??vZ8n_p64u&$!{I^*^NigO^;#fs%}fV_6kJ)C8>sP+8#yHc$Q{G{%yb(O`hf?ZT&VtTlU+fNL!*JF@N1PC(7eixj(_btG%O5iPmfjsQ%Yq94~k*aMR>1TQ_;g0_+(6`IiU$xI0^5@i!~HT81Lo_nVN+^9HYR7CJE<9=@+~4?_R&Fh4vtf3Q802F|$SO^fI5vVXnsoMc-s|G+=*B7V_7nSa10zuejX+$C7LW3oILdwF@?=#mEACqAUpmc@UZ^VJgRSTY4YR=n~Wt8+XA!>=NL;G{J;5(Ilt)brT*G5O`OiOJiJsXh1lWCyT;wANG}*ykzsX$?VlG!7_rzP=w*1G~8g1P*vLXmG^06|aoW9UsfwA4%{?7I3^xYb+1BtA5<ql<#}N9~0jM+wOaJ@3wO%zdk-%U+4WU_%ZDvx=*V>@w?8y9^<bP=i1r5$vLWVR|EcRiu>eGme7C*%b;D2!PSqu5y4-Zw%<+S-f;bt2lFE!+#@fZu;)G)_PpFbe6{?qpV8Pq?!CgxS5H5u-9J7Z^d9%G_rJgHo4F@|6kdBX^bb!TB;1dupwE$c56B-66(7gc+*sx$fc@0{Y!rpQUHtG+lprbY(5D6bfoCCBR}FCIx#3gb56_Ex^95|6Jp<DRvX=)zz=OnAc-FQ&ZbH(G!h2}j2H^zK+N0<!ttdwi4I)X>BwL~t=>&32qit!&d+rAX@PN^#dGzQYYdns|inO-ox#uRI!t;k`F9OXx&j}uES}p&r&_{qi9#+2x1<=CX%FVx{M*y<!TDVCIPy{GusxOvD-)o+=qBFE&z20|O13SsQsqqRWXs;CYyh%~_yDsQGv&41Wx0LyY6(5iw^T!_^_L(sMCedCcn^!Fe{*pOhA;^MzGC(g>qt+uRtT||5HT(s*p<ky4!^!=U!m3nnP<?^#Xa1z-D~Nu8RL46M+*NQVxN?Ea_5Fjpf$JLAU(os&xcASuwA8HoC^+)t_R}9YK=HdRpRo%J&e{&GxLCg7ANFQNU1E9QFG=I>a0$)~f<PDU3*D;VDhrmX&Ld!+3tpdL`|E{riv_jIQfpq_QP0~i|3Hsq{G7y^^~_*!6o><L!p#N{DZI26-H`d5zTn!lc^A`cN#$Qv9aurvBo=x|<oKO*PtND!UoTsEN3Fs7mw#R#Hy`1`YrIK=;^EdA+Mu4Fc}mg@$NuZ-MsS}MUs;EvU3XBA^%P4RG$VneJwlHW*y9Bmp8erLKH1;*NF0ja0mE{V<r>NI)XQ;?1rmmH6p$z?Z_^xY(;i@OP+x#O1l|M%!HqYdhFf$E16bV_Y1?DD5cQUypSBUC1F$0RVR(_=N#H~xkdeE)Y2$Ef-s4zcWA~$ON^402>U?MbHg3ic<n>+NXT#9UCp<mBxONigIQ71odei^qCvct0aI_}`s%R)A?|<Gg?h5~U`Q5sou+l}hUoXAdr0n^PjpRJh)RS;P;dC~d1;CZ@m^ugsz$3r~Kz_?jPlfYzW8nB8WBDg+08|5@l6&Jb9tiN6HzotW{P<dv9{Yod7V4t^%a3OT|4c_<_eRcBj-lnBfF9w|7z#4aqw#x>A@Lxv!ImljKQ8xyejos7pu7vO0NQL*SFGpJ@2Gx$DFeHGwec0czD2pse@wc6eB60LJNLuGo2G6813akJee(6k^Y@tx!V{#Q#c=m2JZ<y%6)SeXUNnJHb=R{XOX0u1)$A9Rd%NpG(C#N#@G<m#!JC@D-U^^w+TLgI2bMQ%PwoC67j0<E9sBt>()$|W<!m1f9s;-E*SE5F{du-*^F9VDGlKkcefi)v?=|05@w5MaN9nKC`&kvf5X!$k!RVVx|BYjO*jIER>W%fjPVQGvwu3&r0!WX!w$P`Pw@pBlb2Nx0?GrtH2)w)bT@zdUbnEpFi_lwDFF)w3i<3Z4*$u@<iPi4u<U);GKp9$-J&YqMwnqRg{wqefH3vkOTV#MC0nqOyP=p?GUZ4A5z{_rUKUf6)C>(dDXz2i#v^!HUK|(9XC!q#2+*ARObl;6XfXZ{PBzI+A-{aQ(L_Ns~btlMQc<m#CdH%@_zwp@i3D4^CK^_5FhqOnaxbfi%xRc)RT=&67>rd)Y<%LIe)?Odf5pYM!O1x|5s%nAG>w{?AmA}_)^bZI5Bip!7`^fsQV!i;t{~G82R*$a<0Dq=0_cc%c@=3F=fB}5=q^93O|8vUx`h%%l3h-Xtzb^B6CgKM<1gk(o0^;$AEQVKfZ4E-#4JTXoSV>A0flP~k_28l#hi*uHz^|nOh>rB`C;0Jg8%bjo$IZ-pGCabN1r@~xHIZ=yP=>905I=$Y3FVyrAYIR}cQlh%XgABiMxfZcT3WB)j08-apfo#rKl}q|TyuRJ{<7$k$$z=7cr5xruE(7>Aoy+eCvNZw2;5tMz^jJBtM4XN>7z>zJT!2maOWQO6EPyDzJEgvH~7ny*6Sz1W8TFL?ruFrw=BSItOuUFBZ=fvMhQ@$4@{}l-IDIA`ojYRTP~si8s`G!=kU9y*_}GG2C%HllR?3K`9^NP0OF%!tMmavUIBnUIix4zi6h|8cd7{gR^8s*>ej3~^}cshKTTQ!))|Ai01|3$b%NeH#h)IREIcaK`wF1xQ;Nss%-~=z-}d_EypQ~Oga3b*MxfR{)98&ue^~(%#Zm5L*=60Xx_9pGYNPNr|A`&^`sZ_h<J!4gp|CPu!@smJ-)-h8d^T(9t1mt_dZi8b<PhLBfX(yDy@!EwZXQY6Cknv)@N5AH+`G18&V>@j6Y^}|@2fHxdXh@_NIyTADa6n=E22OGYah@I0)BqDp~{=+i={|+tGg%Idw2zS??rI#F|?&X4~}lXB7tSNL4>~0>wB=?*D>Iz(YLkIf#<+tK;Ug4T^DTd!2q+-=)=^g`{&2A??9jZ!21?=a*u+;K)}X$tyaZ<A)j3Q_E77c=s|AdRDSvAjZT1gD#!jXe*Nq7lb8Rc9DGhOx01Pu<#C_h+*{mxNdQ;~*a>CpGOq!&{^JvzM{5gg5RGMicw)&E{QI>3ze)Pvll70&{~Vp~N$lqEmr#8l{l8@P?_vGjv+s!xy`}OG&!5wMZPJDaa(4a30X~TQS3g5LR|I?~;m=+E>}HqZKcv2E0gieffu5PQL+`>+%l_#$0Phb_0K~YVd*A4*8~z=2Uw?tM7tkZSP3hf|zjOF^BY!aY4D!!s^Z~*CMnE1%_<FQ^@A&4-m)dwd7<>NcN>Blxs(1w2jTg_`ws(d9<U#{9wr*0pt?55s?0+A9;N?A+f6J=>jG2i)WYTYu33{F{hq#u`?zYz#ILClglD}_x*U5`|KH<wxAb{14<I{xzeB4Rj{OF!%{li0jSon2ED-Q2Q(fak#KY#1#K0OVwj(}yE;E*RC`d#ibPN{p52Ms^`X_G%j>Ss^_XK|hHat;U_*a9K;^AWhfVbDY1kEgUdCjN^41H+H_d$o8QbPqJBgwLdXNceURw>UnM!M%juG7p$=Jec;Y_}A%g@$lxMvn%;Q$w5(&SnJjOrUZIe5B_9;$H=|YaCdw=`t=Ldev|&yAkS+4_VmuJrk10sw0VTy7Qj!a_(m_@tN+{DxA}iUl6%YM)-e~26*pLD%DF#N0gUJFHA{ENBlOhg?V%)C`hX`G2J!ZGF!(>@sPNMDd<$&P-H=z0iJtFz{CrIaSD*_D0{(H$q363=Kf|vv@Ue`ZUG;vxM!^RSJ>Ru_cK?6;346f4o_e_j_Wp!V*O$*E^MSXIYX6z6eVFMPg>Q5Jb4`7M)_*kE&;P4w9;!d~2MF-3RsiRIA`tHLwvP$yE2}H+BYg<Yeac&2HS8mGe`F7RJm!X6G6v$wZE~LY)*Y0qe_#(lyZa+M=t19NNFoE)JwGAv7YKJP0oXevf3nAeb-DwtB>3lV;7ilL!>+;$AmdH9U36ETBm)XTp9B4$Z(_f>hNj<44JA^5WV-TxxxC8#RwwF?++{z8V9(#4FVlXX1t!6t6b9M#Z0eqHU;zx;K{V}(Qfc?$%+|E;UjTtI*fq`b`h#Z*!1OlsqS4wXFMo@Ya@lisJ#@W)GYhih8vp@;KVMTIH545^gP!VRK$O4$U|sJ6|0ZXI7dQLvCwj~SyF69>9DQ$@;oAG%jtWra>l4%8^1}^=KjaATAIORS8X0$3)c5!B`?B&AMT4S%S0c~fFOwd6{)+$OhAI4Yv#-s!=BTba9TnQ!uG95BSoOSkstGPZ>+&9qu2w(XpaE<+^_{kYCioBH{#PmA0Rv>`TQGc}sH^$k?uK{gy}<w0{w}V6PM<%R|F;0SL$rnd0t$xinQ@2Ga5NQqq{V~H{{4R2O_Pti_oVr^K>eQN|4WYl^P4zLH`V(SM-KgE{!KrO{DXcs@cjfoKrsKl^qYN$<<|I{egB+u51IcttG;H}Z#eN3@%LAxc#i&GCA&xeUAqs1{4a9;sO2Y7-!%Pn2mXQ@&m+7AqWg^Xo$Y?DfG@2+Hs}qzwMp$x&tE*xZ(u$aJoDk3pKAT8#}a9Sx1L)*QXank9gP8Cr40+j!udB=9(nr~@lGYId+fPX<DIF0U-6jt^5sj*yDESEKmIRMs{\')).decode()\nassert hashlib.sha256(_s13_body.encode()).hexdigest() == \'ff557c4c020649d06b42cb4f2d55bac3472f9f6c3b7733268892c4aa9cb70f6c\'\n_s13_file = "<s13ab-body-ff557c4c0206>"\nlinecache.cache[_s13_file] = (len(_s13_body), None, _s13_body.splitlines(True), _s13_file)\nexec(compile(_s13_body, _s13_file, "exec"), globals())\n_s13_lab = lab\ndef lab(deadline, weights, state):\n    return _s13_lab(deadline, weights, state)\n')]
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
