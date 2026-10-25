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
        nf128_future=native_pool.submit(nf128_module)
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
        nf128_future.result();nf_future.result()
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
    state['s5_hh_base']=issue_prepare(nf_ilv(dict(weights,post_norm_w=weights['post_expert_norm'])))
    state['s13_permw_base']=nf_perm(nf_wide(dict(state['s5_hh_base'])))






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
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*128;mod=nf128_module() if m<=128 else nf_module()
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
    for f in nf_plan(data,state['s13_permw_base'] if data['hidden_states'].shape[0]>128 else state['s5_hh_base'],state['moe_bank'],state['nf_streams'])['steps']:out=f()
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

NF128_CUDA = '#define BR_KINDS(F) F(26,32,true,3,0,0,0) F(21,32,false,2,0,0,0)\n// Branch/remainder lab kernels. Banks stay packed E2M1 plus uint8 exponent/mantissa scale codes\n// (sign/zero folded into the codes); scales convert to BF16 inside the timed kernel.\n// One task = one 32-row group of one expert x one 64-channel M tile (route NT=32 numbering).\n// A kernel instance owns the groups whose expert count (FM=1) or rows-in-group (FM=2) lies in (LO,HI].\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <vector>\n#include <stdexcept>\n#define BR_CUDA_CHECK(expr) do{cudaError_t br_s=(expr);if(br_s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(br_s));}while(0)\n#define BR_REQUIRE(c) do{if(!(c))throw std::runtime_error("branch lab argument check");}while(0)\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define BR_HOPPER 1\n#else\n#define BR_HOPPER 0\n#endif\n\nnamespace mwbr {\nconstexpr int BK=64,KT=128,S=4,BIG=1<<30;\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out){\n unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;\n unsigned s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u,s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;\n unsigned s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u,s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),\n  "r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\n#if BR_HOPPER\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==8){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3},{%4,%5,%6,%7},%8,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n static_assert(NT==32,"tile width");\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n // Portable stand-in: identical shared layout and fragment ownership, immediate mma.sync.\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\n#pragma unroll\n for(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\n#if BR_HOPPER\n asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\n#endif\n}\ntemplate<int NT,int VEC>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*SC,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\n constexpr int NJ=NT>=16?NT/16:1;\n int t=threadIdx.x,w=t>>5,l=t&31;\n if constexpr(VEC){\n  if(t<64){unsigned group=t>>4,lane=t&15;\n   unsigned base=((unsigned)e*(N/16)+mt*4+group)*128*64+tile*(KT/32)*64+lane*4;\n   #pragma unroll\n   for(int g=0;g<KT/32;++g){unsigned dst=smaddr(q+g*1024+t*16);const void*sp=Q+base+g*64;\n    asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");}\n  }\n  if(t>=64&&t<64+KT/4){unsigned i=t-64,g=i/8,group=(i%8)/2,half=i%2;\n   unsigned srcidx=(((unsigned)e*(N/16)+mt*4+group)*128+tile*(KT/32)+g)*32+half*16;\n   unsigned dst=smaddr(s+g*128+group*32+half*16);const void*sp=SC+srcidx;\n   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");\n  }\n }else{\n  #pragma unroll\n  for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n   async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\n   if(l<8)async_copy(s+g*128+w*32+l*4,SC+z*32+l*4,4,1);\n  }\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;\n   if(NT>=16||r<NT)async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\ntemplate<int NT>constexpr int smem_bytes(){return 1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;}\n\ntemplate<int NT,bool FIRST,int VEC,int FM,int LO,int HI>\n__global__ void __launch_bounds__(128,2) gemm(const int*Q,const unsigned char*SC,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT,N=FIRST?8192:4096,MT=N/64,NJ=NT>=16?NT/16:1;\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(1024) unsigned char typed_sm[];\n unsigned char*qs=typed_sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(typed_sm+BS);\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31;\n int total=TASK[32]*MT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  int group=unit/MT,mt=unit%MT,ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*32,count=C[e],offset=OFF[e];\n  if constexpr(FM==1){if(count<=LO||count>HI)continue;}\n  if constexpr(FM==2){int rem=min(count-row,32);if(rem<=LO||rem>HI)continue;}\n  int src[NJ];\n  #pragma unroll\n  for(int j=0;j<NJ;++j){int rl=t/8+j*16,r=row+rl;src[j]=(rl<NT&&r<count)?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  if constexpr(VEC>=2){\n   // Lean K loop (same smem layout, copies and math as VEC==1): per-unit global pointers advanced by a fixed\n   // stride, one commit group per stage (empty past the end) so the wait count is a constant, and the stage\n   // ring unrolled so every shared offset and WGMMA descriptor is a compile-time constant.\n   static_assert(NS%S==0,"stage ring");\n   const int*qg=Q;const unsigned char*sg=SC;const __nv_bfloat16*xg[NJ];bool xv[NJ];\n   const bool wq=t<64,ws=t>=64&&t<64+KT/4;\n   if(wq){unsigned gp=t>>4,ln=t&15;qg=Q+((unsigned)e*(N/16)+mt*4+gp)*128*64+ln*4;}\n   if(ws){unsigned i=t-64,g=i/8,gp=(i%8)/2,hf=i%2;sg=SC+((((unsigned)e*(N/16)+mt*4+gp)*128+g)*32+hf*16);}\n   unsigned qd=smaddr(qs+t*16),sd=0;\n   if(ws){unsigned i=t-64,g=i/8,gp=(i%8)/2,hf=i%2;sd=smaddr(ss+g*128+gp*32+hf*16);}\n   unsigned bd[NJ];\n   #pragma unroll\n   for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;xv[j]=src[j]>=0;xg[j]=X+(size_t)max(src[j],0)*4096+k;\n    bd[j]=smaddr(bs+r*64+(k^((r&7)*8)));}\n   auto issue=[&](int slot){\n    if(wq){\n     #pragma unroll\n     for(int g=0;g<KT/32;++g)asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(qd+slot*QB+g*1024),"l"(qg+g*64):"memory");\n     qg+=(KT/32)*64;}\n    if(ws){asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+slot*SB),"l"(sg):"memory");sg+=(KT/32)*32;}\n    #pragma unroll\n    for(int c=0;c<KT/64;++c){\n     #pragma unroll\n     for(int j=0;j<NJ;++j)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(bd[j]+(slot*NT*KT+c*NT*64)*2),"l"(xg[j]+c*64),"r"(xv[j]?16:0):"memory");\n    }\n    #pragma unroll\n    for(int j=0;j<NJ;++j)xg[j]+=KT;\n   };\n   #pragma unroll\n   for(int i=0;i<S-1;++i){issue(i);asm volatile("cp.async.commit_group;":::"memory");}\n   #pragma unroll 1\n   for(int k0=0;k0<NS;k0+=S){\n    #pragma unroll\n    for(int st=0;st<S;++st){\n     asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");__syncthreads();\n#if BR_HOPPER\n     if constexpr(VEC<4)asm volatile("fence.proxy.async.shared::cta;":::"memory");  // VEC 4/5: timing probe without it\n#endif\n     #pragma unroll\n     for(int sub=0;sub<2;++sub){\n      #pragma unroll\n      for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n       unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8);}\n      keep(af[sub]);\n#if BR_HOPPER\n      asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n      mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64);keep(af[sub^1]);\n      if(sub==0){if(k0+st+S-1<NS)issue((st+S-1)%S);asm volatile("cp.async.commit_group;":::"memory");}\n     }\n    }\n   }\n  }else{\n  #pragma unroll\n  for(int i=0;i<S-1;++i)load_stage<NT,VEC>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SC,X,src,e,mt,N,i);\n  #pragma unroll 1\n  for(int k=0;k<NS;++k){\n   int st=k%S;cp_wait<NS>(k);__syncthreads();\n#if BR_HOPPER\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll\n   for(int sub=0;sub<2;++sub){\n    #pragma unroll\n    for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n     unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8);}\n    keep(af[sub]);\n#if BR_HOPPER\n    asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n    mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64);keep(af[sub^1]);\n    if(sub==0&&k+S-1<NS){int dst=(k+S-1)%S;\n     load_stage<NT,VEC>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SC,X,src,e,mt,N,k+S-1);}\n   }\n  }\n  }\n#if BR_HOPPER\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n  if constexpr(FIRST&&(VEC==3||VEC==5)){\n   // Gate/up rows interleaved by 8 in prepare (per 64-row tile: G0-7,U0-7,G8-15,U8-15,...): accumulator j holds gate\n   // and j+2 the up row of the same channel, so SwiGLU finishes in registers (same BF16 roundings as below).\n   float gg=GS[e]*256.f,gu=GS[e+32]*256.f;int ch=mt*32+w*8+(l>>2);\n   #pragma unroll\n   for(int j=0;j<NT/2;++j)if(((j>>1)&1)==0){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    float g=__bfloat162float(__float2bfloat16_rn(acc[j]*gg)),u=__bfloat162float(__float2bfloat16_rn(acc[j+2]*gu));\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+ch]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n   }\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n  }\n  __syncthreads();\n }\n}\n\n// Routing only: no output fill, no readiness state. One CTA owns every pair (same as the\n// original CTA 0); with META it also writes the heavy-only (count>16) 32-row task prefix T32, the\n// one-group-per-expert prefix T8 of experts with 8<count<=16.\ntemplate<bool META>__global__ void __launch_bounds__(256) route_live(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,int*T32,\n int*T8,int M,int NT,int start){\n int t=threadIdx.x;\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\n // All eight cold id loads per thread are issued before the first shared atomic.\n for(int base=0;base<M*8;base+=256*8){int v[8];\n  #pragma unroll\n  for(int k=0;k<8;++k){int i=base+k*256+t;v[k]=i<M*8?ids[i]:start-1;}\n  #pragma unroll\n  for(int k=0;k<8;++k){int i=base+k*256+t,e=v[k]-start,rank=-1;\n   if(e>=0&&e<32){rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;}\n   if(i<M*8)POS[i]=rank;}\n }\n __syncthreads();\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\n  int scan=pad;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\n  int key=n*32+31-t;\n  #pragma unroll\n  for(int k=2;k<=32;k*=2){\n   #pragma unroll\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\n   }\n  }\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\n  if constexpr(META){int ns=key/32,v=ns>16?(ns+31)/32:0,u=ns>8&&ns<=16;\n   #pragma unroll\n   for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),b=__shfl_up_sync(~0u,u,d);if(t>=d){v+=a;u+=b;}}\n   T32[t+1]=v;T8[t+1]=u;if(t==0){T32[0]=0;T8[0]=0;}}\n }\n}\n\n// SPLIT blocks per token, each owning 4096/SPLIT channels; per-element math and order are the same for every SPLIT.\ntemplate<int SPLIT>__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\n constexpr int W=4096/SPLIT;int tok=blockIdx.x/SPLIT,c0=blockIdx.x%SPLIT*W,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\n __syncthreads();int pp[8];float aa[8];\n #pragma unroll\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\n for(int c=c0+t*4;c<c0+W;c+=1024){float4 sum=make_float4(0,0,0,0);\n  #pragma unroll\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\n }\n}\n}  // namespace mwbr\n\n// Pointer ABI for gemm: Q,SC,GS,X,C,OFF,ORDER,TASK,SLOTS,PNW,ACT,Y,SS.\n#define BR_ARGS(v) (const int*)v[0],(const unsigned char*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(const __nv_bfloat16*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12]\n#define BR_INIT(ID,NT,F,V,FM,LO,HI) BR_CUDA_CHECK(cudaFuncSetAttribute(mwbr::gemm<NT,F,V,FM,LO,HI>,cudaFuncAttributeMaxDynamicSharedMemorySize,mwbr::smem_bytes<NT>()));\n#define BR_GO(ID,NT,F,V,FM,LO,HI) case ID:mwbr::gemm<NT,F,V,FM,LO,HI><<<grid,128,mwbr::smem_bytes<NT>(),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M);break;\n#define BR_INFO(ID,NT,F,V,FM,LO,HI) case ID:{cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncGetAttributes(&a,mwbr::gemm<NT,F,V,FM,LO,HI>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwbr::gemm<NT,F,V,FM,LO,HI>,128,mwbr::smem_bytes<NT>()));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)mwbr::smem_bytes<NT>(),n};}\nint64_t br_init(){BR_KINDS(BR_INIT) return 0;}\nstd::vector<int64_t> br_info(int64_t kind){switch(kind){BR_KINDS(BR_INFO) default:BR_REQUIRE(false);}return {};}\nvoid br_run(std::vector<int64_t>v,int64_t kind,int64_t grid,int64_t M,int64_t stream){\n BR_REQUIRE(v.size()==13);\n switch(kind){BR_KINDS(BR_GO) default:BR_REQUIRE(false);}\n BR_CUDA_CHECK(cudaGetLastError());\n}\n// v: ids,C,OFF,ORDER,TASK,SLOTS,POS,T32,T8 (T32/T8 written only with meta)\nvoid br_route(std::vector<int64_t>v,int64_t M,int64_t start,int64_t meta,int64_t stream){\n BR_REQUIRE(v.size()==9);\n if(meta)mwbr::route_live<true><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\n else mwbr::route_live<false><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\n BR_CUDA_CHECK(cudaGetLastError());\n}\n// v: OFF,IDS,TW,POS,Y,SS,OUT\nvoid br_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t split,int64_t stream){\n BR_REQUIRE(v.size()==7);\n#define BR_FIN(S) case S:mwbr::finish<S><<<M*S,256,0,(cudaStream_t)stream>>>((const int*)v[0],(const int*)v[1],(const float*)v[2],(const int*)v[3],(const float*)v[4],(const float*)v[5],(__nv_bfloat16*)v[6],(int)M,(int)start);break;\n switch(split){BR_FIN(1) BR_FIN(2) BR_FIN(4) BR_FIN(8) BR_FIN(16) default:BR_REQUIRE(false);}\n#undef BR_FIN\n BR_CUDA_CHECK(cudaGetLastError());\n}\n\nnamespace pmz {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  uint32_t W;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(W):"r"(A),"r"(B));\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>\n__global__ void __launch_bounds__(NTHR,2) phase(\n const __nv_bfloat16*x,const __nv_bfloat16*pnw,const uint8_t*w13p,const uint8_t*w2p,\n const float*gs13,const float*gs2,const int*C,const int*OFF,const int*SLOTS,\n __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){\n constexpr int NB=8*NT,RED=NT*4;\n extern __shared__ __align__(128) uint8_t smem[];\n uint8_t*stages=smem;float*red=(float*)(smem+nstage*TILE_BYTES);\n uint64_t*fullb=(uint64_t*)(red+2*NCW*32*RED),*emptyb=fullb+nstage;\n Route*rt=(Route*)(emptyb+nstage);\n const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;\n if(tid==0){for(int z=0;z<nstage;++z){mbar_init(fullb+z,1);mbar_init(emptyb+z,NCW);}\n  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\n if(warp==0){int n=C[lane];int nb=(n+NB-1)/NB;if constexpr(SMALL)if(n>THR)nb=0;\n  rt->n[lane]=n;rt->off[lane]=OFF[lane];rt->nblk[lane]=nb;\n  int u1=nb*NT1,u2=nb*NT2;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,u1,d),b=__shfl_up_sync(~0u,u2,d);if(lane>=d){u1+=a;u2+=b;}}\n  rt->us1[lane+1]=u1;rt->us2[lane+1]=u2;\n  if(lane==0){rt->us1[0]=rt->us2[0]=0;}\n  int U1=__shfl_sync(~0u,u1,31),U2=__shfl_sync(~0u,u2,31);\n  if(lane==0){rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));}\n }\n __syncthreads();\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=FIRST?b1-a1:0,total_count=FIRST?first_count:b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = FIRST?0:1; pass < (FIRST?1:2); ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        cur_pass=pass;cur_e=e;cur_nb=nb;seg_tiles=0;\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(SLOTS[e*M+idx] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= gs13[e+32]*256.f;\n            G1=__bfloat162float(__float2bfloat16_rn(G1));\n            U=__bfloat162float(__float2bfloat16_rn(U));\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n        if constexpr (ZSS) {\n          if (m == 0) {\n            for (int z = tid; z < NB * 64; z += NTHR) {\n              const int zi = nb * NB + (z >> 6);\n              if (zi < ne) ss[static_cast<size_t>(off + zi) * 64 + (z & 63)] = 0.f;\n            }\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v*fw[0];\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + (size_t)(off+idx)*64+(m>>2), t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t pmz_init(){int dev,opt;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n auto set=[&]<bool F,bool Z,int T>(){cudaFuncAttributes a{};BR_CUDA_CHECK(cudaFuncGetAttributes(&a,pmz::phase<1,F,true,Z,T>));BR_CUDA_CHECK(cudaFuncSetAttribute(pmz::phase<1,F,true,Z,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));};\n set.template operator()<true,true,16>();set.template operator()<false,false,16>();set.template operator()<false,false,8>();return opt;}\nint64_t pmz_smem(int64_t stages,int64_t M){return pmz::smem_bytes(1,stages,M);}\nint64_t pmz_resident(int64_t stages,int64_t M,int64_t first){int n;\n if(first)BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,true,true,true,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));\n else BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,false,true,false,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));return n;}\n// v: x,pnw,w13p,w2p,gs13,gs2,C,OFF,SLOTS,act,y,ss. Experts with count<=thr only; W13 (thr 16) zeroes their SS rows.\nvoid pmz_run(std::vector<int64_t>v,int64_t first,int64_t thr,int64_t M,int64_t stages,int64_t grid,int64_t stream){\n BR_REQUIRE(v.size()==12);BR_REQUIRE(first?thr==16:(thr==16||thr==8));\n auto launch=[&]<bool F,bool Z,int T>(){pmz::phase<1,F,true,Z,T><<<grid,pmz::NTHR,pmz::smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\n (__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(uint8_t*)v[2],(uint8_t*)v[3],(float*)v[4],(float*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11],M,stages);};\n if(first)launch.template operator()<true,true,16>();else if(thr==16)launch.template operator()<false,false,16>();else launch.template operator()<false,false,8>();\n BR_CUDA_CHECK(cudaGetLastError());}\n'

@functools.lru_cache(None)
def nf128_module():
    target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
    mod=load_inline(name='nf_m256_'+hashlib.sha256(NF128_CUDA.encode()).hexdigest()[:12],cpp_sources=NF_CPP,cuda_sources=NF128_CUDA,
        functions=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run'],
        no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
    mod.br_init();NF_OPTS['pmz']=mod.pmz_init()
    return mod


# ---- h200_lab wrapper: payloads s13fix_payload 6c5582b4e1c3cdd4 on base S13 4bb0fdd5 ----
_LAB_PAYLOADS = [('s13fix_payload', '6c5582b4e1c3cdd4', 'import base64, lzma, hashlib, linecache\n_s13_body = lzma.decompress(base64.b85decode(\'{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;Wqg1{#^hOXc5f@^K)lHO<*fof4pu)l+@0?H16~xz2zWQERsfZ<fy2LtAQbcBL?s6oo!lM(iMCJW7=Sg+{9CgJrD@~vz@ApcNE?{z1hQ5hf(pjQdBs{0thh{t1|ZS8wP*Nhf>g$CGGbW`G=0M$UVkRS#78y{7Bp4)082L(mtw0b(fQ&`fyuL=Zuack$V^k001HIo()@$=taD;yN{debycx%zrmhxO4jSXDU!gQC_QV_AgT0uzkYP2qE^C6Cba`mJBqwd()q9o3*;E&G%4+}#IFo~@|C&N`9q3uC^LK`k3Ol|s!>i3T&GW4Yx5>FY9D<jdB~K|+yBE~KP`zwZ|~u0cUpUdrmv#{zI>-63g!oG7EO#vd)R%{oPe7)z8!w=(Hc+r^_2$%xTo(lk33RZ^EsuK@Q3I-b9jm+;22CYY*qbO*yZ*@Wtoa)^b3vtaL9Ol#+~bAu85Gz?mW8Wrxuwj{ynP_pr<F4K9gG1%}Zw*SbncHfO?!1#6RB8)Eyu=C!<>4o^j&wE9cnuj$3!@)a%t_Lhwc2EY;rMca5mGr995591EZ{l(6|Lfa5N;r`$7o*}+-#E}*>br$o&&AG0dLPkKG=V;CksTrk(cwccg~1wrN5c}$f_9$#>1oX~IPR9IONGStW7^tx^qyyHi`s^McQ?BIG0w9OazDK)x(3y;k`R&krBlFn`f&x05<yYL@LR{+ap_nZ%0;#0r2aEkq+2yI-dx8H9@bq&3-`BXH2b?;rrPBGlp3V)e(I?KwgX0>?YIzO{lS=z{|nYJxEuA7yLN0}73rC@To*58%BTsgrOLw?v5;$=XE$3Ae}Hdf@gpUt+AKy316mIP>jD*nbyryNe?!0ow7;^Jhc2m_gnASYK@iUjXUj*^;*``-}`5{9yfM&f-?5)GH|xS41~xs$d!Jyws<u5D|`otHdxV>GC*QmzHYaZ9l#Pao#yOX+~L>o8A__S4lx+KaW(^%Ri)w%HbE8wOQZ?s*K*SrVVuw>yal?CJ;}#W6nwAUS7A0EXI+B+4!iqh@fD3JbPh_wR&Ro=TGYD=Q{$ZkaV7L8QRwY5KEq-Nc0St>S=Le>#-NjRn6v&}3H0M1s1jC5flbXMtS$F)yuV7=E{1wDhKWqk+UZ{YMVGl{AD#cF~n>e#up&SC6OU2}Ng-SiOS;ryzAPXGhF5utNGp+qj!g@161f4L-r``%@$gylxawIvf(?2j%pi{2esOy0Y+gc~i;(okw(1ix%iZ*L_sE3uMqRs~wHBr~@Yel+nXh9|DNK&7|XuRhIrG1m%?wQbxERsZP#>Fan4KAR~?OxqF-n2C2lGTY+_K$}MsaA7FBOZzd~}WRw1%&{_q9ts+*lL$SqW$(f=mo+Rm7k}MSah1dfIQYQ0wMZ|MNkSuuomK{_M5E@(ww0!3vW=wdthhfAzJ8n|Df)Ux9s#_utLBk_ypl~o#U-)U8I7K4h#}zsOOOQXXZHOW2o^^`_2*UorcDRiaK&w{F8Wdk#;dD=CgHvEW+ld@+;Y$B%A{C#+`8(mUm!==2<u$3KyoRkQWOKM_dAdOafGO$W3x4e0k8O>hffJ;bUWK;MSKRIK1Hx)=3cDII-pI^|!fubZLuY>U>{tj}#8gWEcAV+!!Fav+-D?h7Ht?^X6js^P+Cw!DgBJeZXAc3pFW;K9bP^iV-cH&R`ZxOhR0gMWp%?A%sbE1Myvn-57x{S>-xk081+J#y;#2mKI~uSgZ32MfD~8UgM-UXCKz`ZP;X`Jt5VagS-nplL2rxy^3*B+m04}yvI8our!MlIL+`JUx>A2X1<J-%W_ZroO)8_H|M#jZjy$2PrjR%gx|EcuFs(FL?BOE9@#-E3~Ell!0WDwIpTe!Rq5j&El|3zx_{-iuLrATQ)8c=oB1;Nyc{T`qzkjJ=Ux5rOVH(`eV)ffr~(iuL+)}8R0tOViI3~=vtJhTZ=+x_qAOl`&M7-VmBZ6|d8aUSia=&`4dEA&o<y$GC}NLTB3GD{lN;}$UYz#7I0C8oiLDJ+iCeS;)n&{nVlvdZh<AAF#F&6whUOzlY4ZJcsBL3gt<%C>biEVQtw{fcyv#_1FRrAnHTeU^&gmB_z%gGB1|`EG~nEK6*D;+2J+g{^kEK?2R%<FcBhZrmCGKxNBl)c4)z7R5W8-8!N<7|mG@5m8FP>DWphK%sSklwk&#L8Y7gp(+h5f)oq<v<T2@_nc`}fZT!lidzVTrAH>HSX3h(YjcBsk(h7Lf%+ORg#do0&Zwn%vjK4}Hfm4xOe98Iq>d|0LkczrHU|jSwsB#nTT#1SJBRn$pK<DPQL~qw_bCOu-KWgknj7B}J^3$E)H3j4jmUWlyWvTEZUAvs8;b%hX&a@Gi>vQ$Fd|>kP2t>58jGCHcJTaXx-YMS>1KQ-qdRq*<PV78<S#S@2Mwi^#85I(8JF+$MQ+QNiMF3TYrqOdOM?l70=~ed0hZ`!qTpQ$`vBS(Ifwc#;fYuR?K^n(P}a_nH;I|MBrq&>9;Er5sSMKZfI2SRPiaL7O@ElaW&aLoeJ$ay6A{AE`xYGWNCa65rH}yKJAj3F@$R02IUKJKOP^i8>LAAY=mpKjX1(N0MaP*mteB=-h)-=OFH<h`hdW@tp-decpWok9$weL-S{t!`qOw-m0a5U#NS*V?jH8-Qr5?pLncC)imxzmeE$nl~e`fboWmu-EFNSSuZRoWhN6uV!J)7;`JJsO7Db$UNrsOE*$M8hPO_DKmw7@^;j@uPSlURK=rE09L;)9dx*eM=|kd1C(ujAam9k5VQ9sve>_%?x2{7j;-Xr<2pr88fgqtB>oc~W$hsgQmXVG^Jf_<JdqYD##a0B_EWB<Gmb_yz_7pL|8&yLTZ<y}yi>-uQx7qqUejy(P)Vy{aUOAqqv6^c6ira9h^m0+3D(#RmO2xe_ee+9zRb%|!A1`s$b7vIS>$43<Q(yP;y;{8>R)9{u#Dumh~9?ibAKSl9~JR7^@qRkGhGP&d3N@23<<eGLw9j8Bxb%dOl&Q4m#aY<n2`_hd&wnEPuCppJ&I>mv0UsADy)0-LZzM0^-bMmDuD*nu9Rfqr9|cFwo#8VT!HaSP90xRl9j+c)wa*O}fsQ#xHhLggK8^eG$7$YUCi@sM1QN|ZIVen$GZ1gM7U0kbhCio3h$g?>#-=wbRB-95SW9lZqg3qHNP$hf8*flRNwN=-$L>)Xad!=1|H`h2rg^DbO*m>qFP73TxzFL)i`#X7@2@zN!A)$Ulku`ChR?YDXbDgie;0cDPVwac@zkz?cB>?t+#=hBfN1RYXpr#$kaD1m<^kq*nXu@?<%#j==FH~#BO?BUaLL)rtI#%e;=T=`$qZpX0MCUj??3AqwSNCE>W>K#USOimPt#ze$9osa=7y~%HTkI`P6GGY}}HNy_R9a4==a1&#`L}^ELKwaua(#m3|P}Qf0!|U>4cL+6t7Ig}w5s`r0xw!o=X~0TtzI*Mu7xn3J-&f;w_=`@Fmb+P~8$!&zBR;bR(39z@7+eX2-ThWjNa@X+D~<1^8y29H6L^N94onW(9ltbQg(W6Q{lzizWz;fx|1bE0D_8=-LLdyNQJ6-f^gyv?K?vH7w=!kawWG*jg6R8h`FD-<CX;MB5L$IMj0A)B^vnN1oBG?$rAQHIt))0c%^a3zT^mYsl8;_E9uQd|h{B(b9T4lGnWSwP8o@=S_hgKgMFJU=LI`}htOIDZIDI{tGA0WJkBV#YI(IDa9z2=?E!w*$TH1s>X%gBeyXFQ?@TS$YTLQ;DKz+K4ZTUGo?!^`z4;Cs@Mdg?B1E6h3jk{=@GX{D@eKw?rT;0H`L~3}9g1Ut)L|l7<y5@<6Q1Z%{lvV;<y{LM&b!!`hV^i+27NiIMFS>^Kq`u}{@I5>zqR@wF-M2ceLc4q&=dQ^`Xp=LFbUSQRQO^AM`Wf2JH$cxJm6ZlMPN6s|!7v(Tgf@!9xrcrhmA={-eVW3UuxAQGP~1DP5y<yB6JBn~s)roXbLep=StpG_NJSEM+r8D7(s2746A}tIR@Z&V%F*L=7B0{Eok>!*=S5T*k=nqON8uIMB9H7yuezwUQTQWN>rEP$j91N=cdzy9lx-(1t9_M)lzqAM-084nGDNi3w?ji~>-<AtM2tEKly$T5Ecx5pbMd8tt{&`@!0-nl^#7{1)RJo~)O5eKb4lyv;&+~-j?x`O(I&H^6OEh~hYn295Mzz$Nba`x&PQ4uDcP(^vTmwTiuFIv{JW0ahZjoi*(?agS>9_13SXRP=7KYdq?EmwcIZaT?S((??T0%|iva@G24Ks~zYKy|FeTk@+xNuX<>OBsPU?wha6=LR8i4s)qleK+;`V<5|GveZwS{EHn4mj!WZ(b2rtNQ|Uqpm95NNCBylm6iGq^ONec9tCwQHehQR$=#@+N0kPUb#8#p>5pC;aB*-STm-tk-1bLQ$llGc!9sg=wO&-8H1nb?{=~#*Xa`U)5dh`^jKkEEe-W#OAjN2DLHm_4L0QGe^hVEgx^Y`3rg#zlJu+?#I*g<8*eb!fsDx8wWDh&J2}=$G@s69#Hb!HwMD@K5Uq5E&e4Tpv=HvW58&Ze0jv11R%0@rZ&bovOpZ!n21=ToVNh$b$>?=zYYrdsjFC3Gi;x_mW?I0?n`vWpqk9kTcn?yJMGdx8PO2YJ(IU$Z_O;9;6yEL1Q6G%T%QN<Dq+ql_Xn*=6q)63+lYy?Pn0d+gV4l-!ZKa7w-X;_t*1sB8~C2e!S&&)74do)F>sd8QmteZDJ-BH-6#d*{&rHw=>1#q_5Xb4>}&bq4Vs++7Uj|^@=bbWSJ5|sSCg*kK-oolzICk~iWGEZW<z<YThfk#3m*l(70mY1mA`r?Xnt6&<PT;`DC_k*Qn9F`q^lwOWo0s`Hd8tsuW7m*?=c~Lx5O53rNRf71^L06{6BxykSi(l<i2(kkI2KnHvH2R%n@mW7j)Xp=gUaMa%WspCswt*W!BjGC|$X-dNP1ZU{D-mN?3<+qfA-^CW?&iiHifSB!!&QDrn6iwf&YIOgkfFS2*8q0Ld^|qy$*{=GeP;V(<9c_qP;?+|G2;Arg>oiM<}^B?|(B&c-iG%5q}x=2Khu9)ju<Ca^(oVEFTRw0MV77BMClW51W%zB~D5hbE{pTSX`0<5T%C_Al2J2<Dy=U#X~rl?N}ty)qEf00KKxms-E1h(s1#*(m<Yotg-NDu>(i5x067j}Wh4&x2<L6Q9!;=DcBnXgi$>Z<8Xwaxy{_UFs1WF8hM)KulDun|!3<5G~s}i=6+QlwWsn0ZGQ(#H?gBgUn5FE_hRS5|6q*q@B~2TF+Dfcm2>P**w|DbVgtA@bVd`oSoN!Hr7jKB)3Q1c1O_HCr?)-E7_`)*7xvHgepZovr1q*(;Y+l;Kbg+Skgl?NQ+0x+c~CRc7@{`FOaKoBA%$2ySFu;MLuAi6+!cqm(NhhcjL(oYI`<~rcZR{nlTm0;q%Ug=u8HJk!8Cf@pCZk60K5FLqd&+zsQb(d?eE0A5|onzsHQZGV@9~+6CT=QUrL@RX!!NvUW>?-Ou|hKpQ4n<g8ltFf~KyzsD|3VIQWW%Nx~0B)QQEz{pkIm6D=OlC1sVs9;@EK$AxiWqJIE?+3mC2?_Y*KtW$u;X1?Qor6lie~u7_n86<W<zw<R1fFu5I}gWxGm;NR8afbLW4N#F$jj|?-BC;j2%5HmEFaYw5}Pb4X*(A>m&*{^WNHcIU2eiiNEON1YZ>^n@pm<JW=`TzY&Ye)lJ8JC^*hn;(pjCzN<~bUnZ{+uQWXu|k@RNKO|uid|K4r)H<bM6H+T8KyV37g3tl&QXJA2PxWMN`0DroU?meECaZ=@7=<HBsE?|ZK)E}*Xgp<fMIK;UV4ButncDiR2NNaT#uR8PT^TF8>s`tJO{%=Xgbo}1w8j+k)&C<KtApWHa*L|YstDKZb{wI-qt3V9Vgw=D5O_FEvm+Jvq96MA`)q7eTke|b(3_+?1e&z>Wz5sf!QZwq_pFp-!aCV<%!GR;JdEq-Am1#m6UMZMbtdN+4;oF0B=UG(N6d~bHFt<xkJe3xt%ehz0uHTE7J(Z=pnV=LV&<Xv5n!~e6*=rx4_>4`kGAMhEysD<Mv*(w1yTUP1<l+Xq(ZuD3L0YTPgt!d)LUPS^KPIh9wI)70+>BtE;m=H;_^!<qCVLd+5SIV8WglfB!Fu#KZTyg-!VR`}EM19o>_U7mIPKy8gR)JAx^c0BSZv*G4hXBoI|B&Q5sj|RGMRSg73=aZjY%tpCwJTh44-ZhyKBby(4j$9Vp8>eAs~sJa5_i&+xpLpNxDOL#6aZBU*;Imq9_wQSg{u5aGfL)_Q|xla+DvrzTH!;eUn`R`qiX05t77(xhcG``M4gm)(q$RmQ)-8G1G*jDzBR~RB*PGr}t(pT5wv^SBV&eQ3v6N=m!3<i%CRk==NHFrt3lCj9fF}=f&)aE$&liWe%)-DRdrvwVs{CwWw0(U%LO?8+)TD84`8`H9gn@?q?7Ho3p^0jq{~=BOse|A;?vK`lYefY|=f@|1qt8G`4$KhhB^>G0555^u}`BDcltjYl@cdQ&i^BMvJ0}Z(j~0Lt2S;)Zkg*fY3;_1b*W_SSy@?nTd^F=db_q4kv@b)*WEb0Mgwu=4s6bScEQwXE4<SXHT3wXgPKAE}d!{17KNXmk|^XUvrDsGw-!gPkV9uq`Or#LGu<0e`+T>_+mZCOee1H{-?rPNjNDT-s))qoB_^^fj5Km0#P!c@(tKUfqXb8>gw0M+~jMC5byenHh4t#@x3qzQT?LE^gT{P4BuwW)ewIjvIO_^1M6mFnc!T4OW!cYzy!JDS^u8GCF-HsVGp=aD`Fusk_Sop?#!Q<7EDriZk3I5|NLL1p(}4A6bsl|j4IhgZ^>DP0_Bj6w!S{n!W)lJRyACYg%TQTZ|p>A-XdScsE{a2^-UlxGeIdY`o8J-)3X@6Zue9O6Te5vy2}X;*-D!mve1;N`E|}qAS`xqPEDzQV0@1W7Sh<!mAa5Y7)~lm9tC-qY@T|xgSM84-Rk%UJ=7LOi$!~a98+OqBot?u99)%`GEtbvQy{+q6<z<y<*VgN%rJw*&CJ?*I51bcaGfQ~)h@nqe?FuxVO)Hh<d{HlaAo6YbfY1bQ~$>9QHZ^ma?=<YS^9a{adf^Ox?72nwskrjV4vI@1TuAR8y_;yED2{H5^@+o4m^;Hs%+JQ6aR)4B=&YBlLI4C@obc~n3H^&>myBZ$~naC$1SStjS|#Qv<r}1tf*eC5ldRuaKn!gXhj~^UX2OtIr$FsXcTKgO)K+$4neLdLr^E~wjo}_0+fFwt(9a}F+q>iT^LVJ4~csi)g~`Wvx9ug{<Pz+qZjNr0)n69|M7VQxbUxBnT_P+TPWUN2HC$rH#KI|nPS;%(w$rou}3`u4VOaooK0$p$t>36C^=eVvwv*tSCtKw7rri88^Sz~TpjZ*o+fF~1a}yLYh0yMj;5cihJ_kD&1g;bK6jfTkz_x(w09?D7lvj+^`|onV#T0C1pOeGNE;y$w3SPB)}QzVjG81iBS~Ie9cF4eA>(pG1hxI>n09_g4Lb+}xvD!N4?2%ss{$WOAa#X#qme|BG|y;xn3iEY%|$u{NMpztBB4ZTx1%}=*3NKwVrI=&yebqx-|9?Q|EIxz^~38lJ|h*$yZtWve*Vo9CL|nB*VQ4f%N!HDd|kKn!%5QhYP6oNuYQ0b8WmJ4n~oZwP92E#6<`~4G&+#6ww6mYav^0Ae@~~cE6%jMUSd7ou(4Lvwo)OF^sS^!ZcZ_NjscGRLJEGsVYApi*m_PKGCJr+iF4{=g&Bj^v@NeJku@KEH><7MlR`l5OAVdOB9DtLohF1^d=r<Y9aA>Z>;Fwaej?>J$rh#FhdSJyXHGx;>@uH!b2|yKhP}#{(q<^}DJ9JjGJ0iGqv!;%8+g97EvT|WE#hWo0DG}C3YKIvaq&@axtS92onQ7vc#SM^$<ACm$8$SCq75BQ_NGek27f~0n5FDrEnaY(-O;c_+4h?0<(50GJ*3xHaP2x>Y0k=fpTPgu*$ey-A&SW2i6D@A+!N&nam*A;=cU7+hyA6DFnV_`Oq(i?WQDr>3>Wf*nad}3f^J{sD>C(CsMQ(e#O6(19kaLH(Nm+>nrw}OsLA&UD|uhHj+VF$&fPd$#GfK<#KeiV96lwM-K%UvftaW2xgMJ<Ervx>@h~n-;1oAeXqA8Vb3f(0HTL(221rSl{d|EZ<EJV{mS+A9&L}A+P#UsZuh&@h<1%`|mfd!YES8m0MjQZpkiGl9H;t?p3&QBM&!j+dS}*8l`4L_4AB`&aKKB}ydFQe-B(Q~0Xveg&tC&(iWK*R%9JSKFZvpJLz;+R+L;{4D!iipT_v>jW;~vj_EM&MDxTSv6X*cclF@lr;W-}#g1pC+Eel@-r%z5z_fnBqz-SvKuz`qDU58TMcwpRbCrQR^$@&W&C)7WUAj#G((EGjC?bn`<EK5P(Up}-5`FFXch#7IPih<#GwWbrBAiwElNQ0=ELbMuLx?W4KdhCGa!EQtIe-7n|}v&hyDfVq^>W_XM!1)5Q=1u6-eKL@c6Wuq;gA$jZTtN#Xi$E;<V!*M0t(NIYzcm!wS#+-20c9XiJp-8v=cI@G&vX_`_9nN8-WgY453Ci|#OSIjHoUY>JKtrKFE?FACM4>x(3#zQd)~CSXjp>|b=z(kO$o7B3k$deHC+HNO_f9?q1SHOF=}&15BwUREO|erIv$pC6{T732%NX9_!5m7TMbABD>;0%?UxlU-Ik5&>U|hx?E|&r2TA(FCe*`7=0Hi&$XPAtq?sbVd<XP8<+$?B?Ubh4-8D(isB@L*G5{ua^%Ao~Z0?O08(VX>K2c(&nvr~B*$U`xKMD$C0sz8L>cW|gqQ5HsbG-P-x2i3bR!Lu|o)?17donz;am>i-5qusI*lUcH+2i?|6`{H2yqG|l>nmHD2f%XAs6Msg$8I5oRxs{WLgzcoc-c2Hh!s~ow#SnaMvWtX+U~u2#7dm#LtU`eO9~|DVbZpnuN%1`@p+2Pl@5+ptG{Nhg%->6?*2auw=iM=yo!iTL_8(c(@9%t4Pn)}%d6vk`U>J(E4r2RKzOc2>^EUr%jeu)?H}pDx^$R0|WS?tT0hWP@HYH3;9N>3cKkKmu)Ov@j;?$e<1GA8F_RPlt^WONk)19?X6D<qZV}Jh>dN3g0310QGG5^S=o+F1BZ-n=T5c46jHsmFMfua<DH!9!Ly~7kLiyUWQc{+|pt87==nz$t957y7yYG}IDhvPN^cSWz{j?3An9gj(0a>4wn?U}ZZt88pcI?iyzKnbl9WHiJBtUE}GdN}Qs0rksR4c1VvaOssZ>3(L5GYl|R`=-J<xMo2pO*^H|5ZNZ7Ae`kz8rC9Bhq|>U+t);m7m|=Os_cX*LiQWU**E}E)^;78U@SLBHM+3q<2(tE0Drg)AxPz=kr<fSOW`}(CQOPFvJi^(`6w)c!$cuJ)~`x@$#MYm+nD$mtX*+y?$@NZj_7mFRdlU4)+<RcT%qunP%`Z>$Eobo8ge;)Fz#rcc)Z8$Xw4{Zws^<V!mBRUO#e$cX*x6_3rmPfr*CTAGD?q{<=ofSA;p>xh~E*imPC2oy!c*Ez<O@#`>lyAU!#XY)}POSVSxT%1!8<8_OV}yDpkShvwi}wzC^K-U?SOYC(+0X_^vsiS0#p<W^&mszXzx>EqOniR=FG>(NM%tz1g2EkOCB!Vpcc?2VzNZkqTTM$*#m{RMO!0DR?SO?O_zZp?-*3PANh_tDBB;`ut%Bq&n}Jhl8#E`(2kr)q^)lRsfHgYNp@Hz8<UvGQS?)rvqtOZZy8L=~FB-hrPj$n!Qw9Ra-)#p`dc}AuQt=;+wEKvj2g3DC&W<GrtfY4xmPe(txJ}8S3}I$Ruq++{CME!dlJ7{ijE{i0bl|B%BE?o_yaeHU1n&nK5?3>H$S9*pbzk*OtN1+F&^cMOo#qb98{^G133{wItI+y4iqx4N7$EooL=UJ-(ssW&P^wBkm3F-sW0pfb^-$owo&>9Pvr_AS(6JOUs$}?G}G{HpvCV{?<>_=!D{Yi9Pu|P9V+*qADC(mIGf0Gw9h-$l7=q>@waL$@k>GL1v|f6cd2c`-XQPvXoZ^uR(1cpJKD2Ts~=xzDC3>gpjT5#sh$h1tvMbA7EN`MZypX+={fjNGJc7^~*#B;XwH|py=}$AU8Q#<BVSzGhRzr)K7^=FAWnZ2(`di*z1zbjh;@P3$$U%UL(V$kU64|8la}Qw`o-iER6WQF?qLnINEVlBvQ>A<}8=iRTWlp_m)TObFH~cyUbAc`i4{AzLEyizS%(|47&WbJlyeM#JhoG`Vm}eQC5z<nogI(S}I}3=qI2sg%@p3Db44?=b(ghWj@%XD-_JlqA?@^AWpNz>YshH{+7o6!>ra8*G{LT=`TIao~R`ZJ{!Hjd_i}}xdKDgsJWh{-1IN%>Ve3N1t7&$BNiGwi(s<oBsu|WuCzt?-f<_Wz89VTT#Z(Go>aJ<x4eVCF7JKpoaZp&H%Q!z9=$Uih&)b@AAq3B)WQHms^+Bg-vCx=zKh~az<VY14m&1}I`NIx%AXBfL4ohrNVRPR8$6Z>GERL|B~xB5eN>LA#=r`Gg&4)I?Fde?Axg=`_e<c8n3ORucnSt_-meC5#0-&v2}$JH^sl{HpE|D@z?b7kxQfM>K7UgW<%N0o!~7-VX`?!_{(P)_w+hWcxl)u|2KPNa-;HihvsH0QSeZ`LG8^WWP1lFEALY8epBcOw7OrqRqM7#+DWW=B+|Elkh~hh$SB>pb*C}!?QM}pXWSPDr*nTb&>i>lSynbGa{wZ%NUJv{U>0>Id7=^$D|K(6yp?YoE0#qb>YdF4=dNG=vFxY3guBDieEnl}K4}3h*su<~NB$tp**M)oUL{!aT2;0n(fGI}6%UvIbsN`BXzh>BHMOrzLk5648o^v~fo9m6<GvQLe1~24iKL&UQ;xU23oSZCH!S=g&bbj|Y4mTgkD7U5kIZ`0>nkMUbtyHi@;>=i!tW10j{`s%abBPYM=cV}UKm#~0_X|}!Jy9fO8;~Zz7RH2G8i5iAGTcI4hDY5f#Cw9Ny64Yp{?VFU^o~+L*33PPe>gWWEvJjvVH&x<brxx{27CX6uDCyYF{UAIm_OD7_nL14w81TN>$;Q^uf^>*2L*bQ(7}R+c*_+4#BBf057@TC$o|eGHI;H9uQW_GCSEN2O5|VRsmE)Pm8T=QH~wR}TskLa%3#HV;DNS7XLIK+HW$-(?b0N-No}ufS|heooJR@bn%BKXJlxr#J6R{W!8^~@Hy|yw#<4v9m#?R;ipIQyYp)!ZVZbjMo_S>7ji2T35hgdCn?I=3j?N=+*mcy=C?M%nM3c{Okkk9tef=6{Yp5c{z#uWDS~IX10{*8CN*UX?-YJU|hONVQs1zy;WdHJ4g%}LbgE^xrYV5<q<y5WEk-O|WBZ3cG<!jSv@(q_x=M$YkD0A6NFC?fTA40@dR^<(%0Au8`sZeloY`f7+IUAhE8Zeh0Hlm7|S_8&(yDtNt{BsPdP!gdi9dA~P<D4+4n-Abi1xI8q9e!8z1o3I#HTHSCjbF;TJX{<I%3l{m&k~Aht0BvLc{jn&M3SSAEm5EVC)BozkYuI@wdRh9E*D3WL2>n~$g&Qr8bGum{%Hy;NUOyp4vEO%f>EF&YURl}XlpNE5|pC$IJbT5ZUX36v1_++j?PW%8lI_)f9}1S^lXk%SY)myEj7@MtbB5Yy00e3g0zZo@5OxE#N3|lV?0ji2*cHNLD9mG*FwV@f;(PtTjU3#S}(<=k+uwpDD{4i@WqWEl&$zGH^cWDhHq8Ql15XFK{aOI-Qr0FCV&9wSU@aYv%b}XxZgqg!Bnmkct#@@Bk->WxJ&Rrs%cLd(q-_5s(t{IRqEewC)+r$h(Tn$h6=we@p&yE#24}bgqs2a!N>|ZCXQvgFl}R`d~>d5|Bqz6aa{s>)=5)2ssGP9wzV#bLj(8AZ20{wg;>->M>u#6S!G;H5B2N*C8%I(lO<G6_C3Ol^*ThZPQ@<~R=>_UFa!<J=!#VFI{X2ip8LzBVTF3LIJ%q5c14HDMAGy(mskDC|E-kZ6Vd%<0hBNc0aXf%eaRI3nC5OwD4z0yBj@75`ME}7sfWDDhIfzz_GB&(^*h`aRb>UoTPu44JZ#+W{GuKQwIXAxOCr9qOHJ;Ntzv$~S_9pjBjG#pz$Vx6O``P$)kIZ!qCMx_c?$S~@!lvxe<W2vbeBdS@>iS8_+slsFpri^*g8d%x>9r2$s2Rm|7$F{SiM?#3VK6$Jr%XuNSkM?o(_&}@+!h!&JTWQ$Ip9R@~8xrz9Bar?k|65+b#IzUm@`siKZMm9#b6la}Dj8Z`D47BX%O7RLW=w?U0X&#Ye05-z;^ahtuI&K-?#3SX5+w9!892e(ui|`{d$-Kj}S9X_cdiB$shUsOSD@q^}uviAA=m4TBK4#7=E?{fiu)<iDS4iTK^7qeIK!`vYFBT7w}nj}-Fq%T;SjL^KP3<(*W?AZ*T+2gWn8&SyChqnsP*4}06NM6P$^g+l4h%nw2|cQW<dJ{dM7iZBgyqD|hi-t++!<EDSf{R3p!av-yRNTbPR2*fhu$hD$LZNUo*wewBFfnNjt>8ph<qltJ}q(p_X(%qG!OKdCPm;N}<I4y2!siTPU4>Up4xqKay#AWJGsl^jh4w^rBo07%xB9<`|1&wN10>hBYL+mS#=-YpMtsl2yCxZ1YNK7^ic?2mp(s|+=$gN5ozc_dm_@+;aK_^ypa=9xX(_gBf8J)w$2ycvpFC}3FB*Vl`1|rnQ3(6UNyY=7i?jP(0-&h!xC8>xai|hRc4m~u)H`IdjW`GPt1@HTJp?;q-+6sCH6w8(Qa0~&Ta#>GZ91od@297>Q9W)a~deuYu?jj4fm+2O?yOs{1p)6&RH{>j>+XiATLpwMbV!b&hzM`bpkH~r3nxOeua~zO9Lf#2;!1_m|wOT3o7*9E!#w)>RED930c)$dGcF(WfUP|bOKG#Eha{IIn%NhiclFx=eLMB8*f|SIRkXtv)$<L;OmNKJ^jWfpR+Hy*;D?R`=2|wI(^3gwWs}z(XL<YZCX6LtE9RmqGVMWY_U7}5S87rrW;ch7Qs_eXvLcBdn=Y_Ci;b%Nd&Sc=Fq0AtkPi+y&v!h+{X%q~*pKvci%nGVpc%Cg-CbOQRYYa(71B<tyv_&}8+NGX;5wNFeC6xLh6n3+}f8RigmS}5>b-XprUrr;)Bz_RV9}abSBhbl4UJJI|$*CJ-Aii89mxH1hh*=p?oAhHIw<};il!#hvd_-`I=Ku~8i3WI$3Mdp35i@rrutOmjy~S|bdg}Jz&bNo+2ZGB)MGQJh-cW1KE`9^%ZxP9C8}gEk1{+f;+wQCa`I9#M4O%4xl>RC5fU&S++B!v&ZB9(`6DkR_OvIv`i)4rc1e_c)f@}AYs=;s;fa;#Kuoqz(|M@XWMMcX6P$h%HLnY8&s()UXwGQqrVTUS(D~A!@DDSw0!$UeOL^eF%@Xs<CkMVt?4D7Fl%wIc7ke>VEH>{zDF$~A6gS&W2FU`G}3;5`m+s^2s6)uV<kjPC5q%)E1IAgB$L#4h}5(<w21+rb|CveHrQ^qI#vRlMk8?z&X>C!JB;`FY?Jn7gN7XQ#|eaQNeRMTUMEsPyRD5CGex_)J&rM9}8UHOfccssEsLN)pBu=5&@Y5$5<E}SeQWEyP4#^P!BZT$3-mc;Mk1>K#E*HuK>WG<)E5*+B%^oq3$oRtk=+NgYhP^8;KBLhzNr><t&<k_;GSb66%1_BAsZ-W7Ew{hX#eL1vBlx2DeM6EjGh#H&;sSbxf(cH8<geFk2tO%kDIjl6;@tu%Db({bCST-t$aoqrbtuJBx-v!V1`Q?F+2t)8jwIddiDSPer8VjwCIzqWHS_u9v(?rfD?p$SKtSC5)1!%;&;_o)M%{^_32Buf{_$$yoc?&T@R$go!ZI`ZLmYYKw08A)>1Ih_)(>vVMeA&1wVHRj@f_ky1srq&;tXLfMRse6*`BBniGHy21lM5)qUR}Bk5|TeBps?ceosxlp9QS(CR5OREHTuKy(N{wBusba_F9sqqkMY;dxjgxb<h@{`BQA80S2kBUpGc>Tp{5}p+^Yc6SdXhuQ1iQ7AMv)TlQB0XhPaZdBj9>bk=BD}8l!9Ah$C*w;V3I_d*98GC-dtr>XIY*5<Qk~0I9Ly8vMDB(s#;*)SsJ;kLra`(x#%bRxiVex>!*u@+jD)jtVR-7CBan4X3Znz%gCp0lfZ=%iMNdL@_d#h@P<#3_sD{z)s!WO>wu;x4aV3Pcyk!IXJgI{pB(_I~$ZEMCEvB945>1gKviF>=i}5VSfRzI7f$m2?!NR<?a)Ys3N3(9acCc<tfhJQGdw$x-Ka8G%&6Q{=Db;61MLsYMLYqb7Cj9KqYX1fSj9NCf3dEf1l&bbKgG2G^3Q6tzG_$p(%{`jiUo!iRkgw*0O)SK_>gLf!ACNu+hZH7wVWaQ~d1*{;Rb<mTe#s^en*ge+=uN3^VaF){O?2R~{G`=?aDB(3A~)I(es3(?_z9->)(YJ+o77iH-z?LUy)I{uXYJBQuiufL>cVKx8<^nH^-9u+(bO<<k38U%%iR1Fr!da~uYmhYVvh6__)-6y3B3_Tu<asP@j3ZeBk6`o??eSBagJ>8p#cBWft8Xy6@u;7lTE{s#>rDkV0{G#1z^@V=zqpILsXk>{t_Nz@MGKri^?p=B&d-@h>Ndg++PpBLGLl&^+*XIf)o`A@!mwql<{e;{f=3|ss-i4{vN3<3WPJJUe?5q2!#Nh}V6dn{k7u>T+cqF<;2mJup4pM*l<PnytMZ2H-5A!2cC4Pz@)^2IYz@9PoiT1V2o2;+A`P=?DE930hHr4txR2=my_@dEknH3N$Y_6~{$p_w6IDyZtiA3<eSkOurONVf>=7e{UNTX@I5HXcli^js}}golU|jVM`r8{~Wa5GvJAyFWb8T9yiJIC|B;qHSCc+c^$X{N7WG+WFYjM@EIGSiB$_p+VC<T@z74f%)0u;5<_5rSYq|MUSxCIC8>xF|)`M!;2m#E4r74Z~G2kV#<HrR3EveHZ?hWpQVmZ;E2QX<BPqf5N`NoQj7x)WDiF<0p>9n!{Uw>P5KcdH8I}f(phTS(OIahsVONMrsXU_YqL^e8t&)e*UJb7D)QB9F}UiUESwr5)^#isvoDVrq4FOvwRsVV1g`~Y|ErIZQz0%Z^u@i85OSHb$4JG#!8X>AzfC!c?W78JezvX)0mG%ta}UQJlme%{%JCO>F7U1-_u$KQli`X!xBFLf_l6?C9lwX@4N#;#RA)c~-?#%Bf;sCtX?TQ7WKMzUH_rI=Ibzu}w#4y7*JFofBLo|;a{ayw5UgWBa!0uf%@RM(#RxGj-zkU$;2%D}qdx0ns0;J3Y5C-J)Uy2_MAbp}f#PghJLWw;ji{ApT6%!wO-wyLPEz?sK>m`h&|xU0+sO6H2CzKZxMy+^%#w736DO7Dz7QYlT&jp5x*cCN5#wNm;kVY+y|<X6={=$CBYrU!asMXC_*Tq)A_1ODtNaq!IY!dbw4=V5LaBOk$FI%4X3XA8dEJ*;JhhZ$NcJq(uK9EfSY~=X%i^`559Bs#vDj$SEEVD1^kC9)EFRCZU7et4YIH6x8>0=i@e;l(n9)ld-MU4frrJjw%5knOHCTp2<PwLP@LAhb`A-gOtT!*-IgQ27*!N4QJu@JcTC>d0oOhs!6$$K$kfcYuUD@1t)e7n;$vxl%6)Hl?PyIN#q~U%dNJQSG^?~BZs0_{)fU23$_Y?M?l7TsuiG^s@+OMKc#xhPK#O>_blIW$|pUjr^D7CG~F*cBUlr3tKgcb5$vOuOmV8iTz_G0oL%Q11M&U3*u&^z)J4u16*y4jCJJ2bkyb(*u~@T;{YD7dgYL}Dc)9CVbcqn=uamSA7FtK5~uHbA@lxYZGckUnVh!Bt;%l&VAwl%C!uC0jX<;v@Ho!(O_~W{})9&Bf}h%W4*uD#*icQ%u`Fd_-pAws0SH-H5?bCU;{XdG5`eQm#Tzy!U+tYM!-40WY&7oqv7YY|GMj%)vA#lTyc{A0cjDa4XI5U^aKG^C88jgWrm&>~60E;Sz#iFS__b&nqennQY`)k;!@1zGUZyWR}N<K>}C)oVunaf8=H3`{o|K)x<TVTuJZ0nMGV&e^Z}(-$Ja3_g>)C%iR9fT50K@X-X`kKVFRsuA{akQOV=cNC+VylqWwE)DC~8-eXgbJ3)?mRPb3RPLEHGbUHhwTuARq%%b@nRT&9%4{+XuL@>mX4aQtr<^f`#72+hYtluUYmy#?6mA?5<fDIC(MyYOt$xCj_skkYxESfRzXvN;vz@<3BpiQA&y~?feJ~*-yHd8$DnzM8<88Ra`6zq+{HqQaHH#md^Y_Hjo_(;R;H{P?Ze^5Sp7u{FM^A+8!R+|I=<wBJ12&-79{K=h931~wWMFxy%XA>GU<`1Enz~bfhdE%44?tR{Gp0M?S6pVzBs~9$opnwEzJ_%RWj?!2eNB{HB&kyY=JTK}*9pxj0V>aTYFDWTbgn9=Ha?P<ZTHq&HSpbe$@CvY2mafSKlZY?y&tGfhAQQSZ{>W*<3b}`30=%uQi7iNP`4BE4&sPPIC^(s6ykZ|~vR5!(u|+O68L=qC()<g-?n4iZ#Hr|~K95J%m+Nbkve!Hwpw4Cl_NV4gUPr>XQ1I9cCKv&pl7cNvVesj>`*2PN*F4_1EdiEiq~{f{%<y6kh_>mX`at%CdM@Uwt3^Em*7tDyg#5Zo2f_!cH$uT_Fa<p6R5)($yVvqqQS#QG{_I;5(*#vPDO4D+-hGCxmb)RDT05p$<@V1DTwlG(9j>Lu6J@pI1?etb@8$@@GR7+FqsOPn<%!L%5yO~LA0K9{G`K$5>?h1ax8_iS%WJxdPDde|a0C&}d8hbu729L>U|hIH!SpEv>#z1_0zPtR+;G83#8yX$src|36Np3M0YX^%TK8Y@cd}=mkrdq|9#@#~l(^6w=h*pnTG_=&hV3wV7;lJy##Y9@k`GS;shg6>b_!9Yn%1vU^AUYs2lGNzXePh33j@6q)$ty*6H)${#jcizGG292AZvK`Z6bKkM~IQ4sK1t;js&d9Di=-z5TaLjXhakJCPY0vCKo9v@xdjxD|(#mz^lY}!mA;KfoCGdfido0;a!v^<_m?Ch`DXgLBF~4^3Kz8Gk5MIA=-NruI}C-_mGmX_%p0bSneG)D+=pSB!VjbGgi)$z_2cW4A##7|5M*va7zAi1HpM^oNfHr;s14gEQl8YsZk0!V|CP5Y61A{d=GNPZ*@e)SP|g@rg@R&XC4Cn5vg5q(BQmZVDGge607jc;2Tf+s4W4o#8*d8-RR8g{*c+ov7Xtons{dDEEbtLQzu6)p?9yuL>L-}tk?!%Ph+t4zu0g~@!rJcNwtFfBgFiP;2*<iA=A5Lx8~T3Z%nCeVwY!U^$KIW#5&<8Ea&EOe+JuOCTL0CGf1(vI3}-nbUYUAjk~H}{VMJtZ&P7p>CE6*=<O_xSqkYYI7Wi3q8?*`p+H>_u0P}`nsv!5hBRodjP_IF4!nrtnyip~z9EGMPTkO>m70uBtd~pzTU5^@?@C>}4S`BnQx+ZjcjbRgkQc#fAz6W1kdD(Ho00?uj2Xb|yChGhO16<`c5fLpUS&<7E_KKfKEX;T03#sq$_itXP{cQE?FiSrCN4T;VqJEKI4)CU?>cH{-By8VAh8#5eFBe@v$ge2Qo3^coY+9}T7jmhip%50xGo(mR)W)h2~0o2_D}hee*13Ncgn<4sl!Qag*EmQf^@x{Qms!(&tPEXZ0Xkc%;RObBL_Om?CE|{$}eF=h{}5X^w(LOwN&pctts4j!N11-+&aZ8*}`aHrG6YyGk#pyCqTk#wZh~X^yaB_{UN-bv^8hGE`1*(mI+I*4PJW1BWrxXX^uQxOUM(zW|Iv9Ck;S%Zd>WwjgK}Bnvb9>^(^$Igp6>MV=nfgXf*<W_GgkEux|v2{Wh=%qfojH*4BCP#!3?VS)&I<XTlIAAkz>HI9B%%hD=*Y*-EDdHt4ap$d@dh*!`nlPVzuDHKKlk78PCY=|6AphL<mU<;~L<e7=F$2!uq>VK!ID2tP<tXLH(%b4__81Q9DopQMI|B1-!jz$$|UsKxA<bj%%rhk5RbOJm;$bh&rJ+^F50g}Jf_dJ*-NOu<S69G_f7H;ww6yRkAC$OHl2<+j5_PnWfddSkD2+p-+ACs^_Pa|AQq)CHD%bMbuNXR?=ysNpYKjlf+<uodrX)4<)2si+x*_VTBfLN}<#Po)pt5nuuRVC4h;VC<S>2=A2wKt6TOp$?M>t~(*hN)?sK70++;aa1sG1k>x@dP9n0dh4_H1u>ht6@iMiB8j3aD`<0PHhdJcv+eBWO=pV7a~%nBW^JSLA+2Kc9e4Q7WsX3&%j51XEK_exD>fB&2-yc7s^1Badt>?+<A&4mt1FTTHNOf!E86H)9;^k*D<4G?@Qknw)7j^;qPRl=;61nD3x?@TH=hHb)EtMUHR#WEmScm;z&rJ3iTs7P8dJhFZJ4;gr;IH*0%LRVrckvMk4O^(F{tmKx`nHDSd0KR6aoak94#t);;+ZZ5L`MpO~yRmW+%g+CSaTui_7tm^gJ7deWX#DQ`4{-;TRK=F&&ikL(+4f%mpG-G}l#g!VK^^0s|74!BSY#BoeNerby@3>(p0N?&e{*N*Gaxi)r->RJUHA0QwU7P;}qZLub$PVq#LJO~D5UH0d!S32(0*G3%hfplqf*wjLk%BujC3M~mI{dGdsCZuc+DTHbu#dJso4N{z?N6CSOz+o47*ODV+3Z>0fJulO|7k{j&$Kf|^L5--qu_Via*gP(}u#yE!tuTfp3d{MOI)tXSi>qfsNxN-jz3CD;kj@MMVZ=86g`SBilKhBF|x3`SsEZ*DcSB+Sc%hY`SC;p!3$q=+KDX;POFxueOJGcjCd;ky5J=%6&vUucn0Y4QDk75eQII&)<X(>qk14nXXw2ov6pne4E{}oKB6Xz!I&Tc1Ou-$D<L#aD+QvTZt$BjrdK4pgmf3MJ>!B{)n^Zg$p#>O$eK$fdM0PEhSBdC)QeVVrYm2Z!5$Cu-Mn1Ay<q!4U{vnX0*on@GLDJ+M^Grr@xM+z7uyhr=kl#?ndc)mf}XJnJb@gMP~Nd`$F_BEj7u$H*xnSy_2#N@erRNmfVU>ekqny!lEC5*c?93w0RRM(AP*W-;021n2!Q)z=9U%(;3bRCDe0oMoDj5MGN)1}Cj870oP#ZVB@2HbLxh2`OXN>ba%q8dfHaqLd{Bf1HVY;4}GV)C^0IGVzVodNMjX?}f<dA(P(1UDr|&5FJZ#TXemEMxQWGA$LF$wfOHUbNZ`-}xllKloUYPE_!T!{~0!x0`BcSTH-w)7G??Ta(c2g=IcvG}Y%xblym73{4TAR<H*?JM?%P^2R4BBM<E;*;m}u@8;XzXnU2ZkXQ9_5l2mbm0tT#ZBn%G_mD9X?JL*rG1_+l0GTWlt@4svRl|HVH^F_ZuQGD|jjS|4!lH>qNk@u?$OON*skq_vA)tcfT3!DUXSy=&_T419%Hv;;3YxZwnliR&YEKMvN*+IIpo1RT7}cv?t}^imY{o3eiy9tJP^%WF>4G=P?U3$SH5ur%tc8+n0GAxXIj)KLz7PgHJg2yBpGFvX5W*ri9vw6-8`lNvP5WQbJZMRsIm%`sz2ipJhM!o_JY)nFqD3|*AQ{+lR3x5YtH2a*YDLlaBihG%%kuLzFyz>bM~liW*9^}*|GEXf0;)9zhjSv`#;+<V@$n5qYG%hR|HvKcLn?Yu{!4i=`(R;c2Ewo7heYesje|=Za(TkBcNuo=3MAY@sI_}0l~@yFbX`73-RR;$sQ|zUeFR80l}TD8NC=+%t5e`?C!xG8%8j`idi-_g#i&l1NzA?)9?p7g`j|#XX;m%?pXR_<%s+na^T)+*avBZByt>Gq^oV(<gm}3bzrQl~4P1VN;eZ)3hoTe#Q0Ug2w%8Yse@1FND5L4gHu-QxR^zQH+cJUVVnjH+$xP=j8MbDUFEHv|U^fXd@E}b@?d!JBso~!p#OyyJ%uMUn|DNBe`KYQv>0~cYZk~FPgZeaLklm6H*0w>N?3{`FHY1y1B81pOJl%Di_m{D+E8#sb_rj5V(EZ*-rHH22%{$EXwR9rYaW+)?(NKvQ`+d^{#UMJh<q~iAt^GG7gP-Y8RxCX}keND;c!X^^c0YUlXyKU7^~kiS#l865h|>`^V)79VHq5Bx83dz~9{0`UtZnecC85uglgI4P{Fz9}046HnT%Y$npQI%@Su8dFmYHimhS-+1@rFS&#UkNf&<@n@^KK`2?LquL$gnXK-M0KT4Zx6_fJB~5JWxLrK-qt5i<_|?6n`RG!OHF144x$eug2GlmN!M@fO(K+rUSPpTNf=53QNj=XZ9yBTgbeD<o|0XJ|;_{ca&?P9)L1-w~qoAZ!@-w5W-0{HY|%VlvI4+&6dGSVr7g?VvfL0Mi1CVOCSjBmG(*nxEXhO(2W5OVB@LbcyNVu2kEsv-xxX1X**qk3+CEssJ8OTY~Ty%*dp_pp_4S_zA@`7UJ?O-W(j-}t5)iJe{aOolW_wX)OtIk;bCaDhpMZX|8+5cX}v23?dbif)ldzv#csj~zb{?YpKfKAz-Qf3rCK(1<XNs0siPuap*;d$mWB-SfN0L>JtXgKbhG0w$*JWt8FSvKk)3HabAZuR-{>o+{q&kITVD*s$N~Ldrs-~j*c)<m%79un6c|`XA4c>aS_e4g!4a_DEy>#eXeTyVGT{8%W#f<SdrCT%9tk8DMz1||oOi9~H2tDnTV;%mVSVb`imw@bj<<#BOeK%Q2OzcRSDj%<i(d6>RJp<e@;s%c%R^{p<26A>Z_*uLt?JmXU(_pxm0!@4IYF`c`LZ*=B+PxbNC?zfgCp`i29YjchqvCGSa#aAOO26CS=E)U>`EFYug84MjV|g#YJ@xf`Y%FMMK=L4cv|Lf1_K9^nb3cz(s{|X7<-ZKF*SomTW^8-+ZXzH1rLDWc&abQwUe=tl<3JHEoange4Qdl3%2aLet{m!igJLOnw0cQ^w(q5!wwcNB$iqk?C<UQs)pp<L-s35j&iHnDMVE277Z8KTyXQo5x?hn$9Ef4_S|)rI+`^Kif^ukr`?xSp`;PLut<QOl=V5!iHVw{&;*L=dn|puuG_I3V0Oa8S0n1=g5+j;yvXzn@l0!KR@^z?;~O@~Aw(F#0L{&zfSF|%a%M4?xW*V<!8hJ_%nzbYwLJg)%>jmX;Nb>FixwaGKTfWF6NPsb{zpWBWJw>hDB`i^E6luiQYd>}L>1`kguoST;SHd76%*!U97P617~UHWn0}79ryu3<QDoo;Z^TMOoB6!QI*s8GNOa#Q^e#T=Q(lQ!Po`jCQ4a+G^(#M$`y07d-Lwzq+0u2*!B!|#>)YHv@UZ8l;<Ufh=wW&8w||}bdZIB42ZB(MP-zlC;$&Je!+w9haSeXra%I!dzDLHE$E#$_ov|?TgVhZeDfX*jFWDHsw~9oXzbJMlU7I0N>Rwj5;&xM(q=)#>2Dfa6zX(C3`otf=b&~2_E9p%?rAvHXgZ<*LVCDE<^An>A==K`94=$?Ht#`IR6319mA2;O*;tEw?#L<6EkWj`;7%~b=E>IarG)@jA<4mVTpz@ka*F5y9r&y(HX;r-#_6Lf6G*HZgr$0*`)l!TX9fpuuLd~>=9xsW=v)s~oEnmU^mZA!(&;1H+>t@P;QAA3n?fFANHCyC(078%IvK{@+1e^$U(ooZgfKG0Fv-CCAUyqS$7>;(T+=v7~fT`V9-0X79Lo$RX>EXpde)^8_v&R>mn8Z5W-idUUd0j?;l=l?#iYsgvWjx+}Q)vLADgoqplg8h3;`q~S<g!=t_kmRJfdktbW(R+0P6+~3kkUSCtU5Nm)ewpxOVRAM%^7C(asUpp!xx=SBGIneRMJ}JpyWV;*)*Re8&@%4LZgfu0{u^Ew5+$n3Z~r6Pl&K#G3Dm+XO{m-Hv0>(o)`nP!BV6&nTt3X>TSql^KAiwsk`{qzyoC_kH=2i`tN<9hJd<~zRhUDaoTVmw2{aq(=~`kh4mXl^o#`lYm5^<@T1vb9H<_E5e9h6299ttMK}IPo;_jzf^B7`5ujCp7na&ip+z9G6`t{cPefDEoBK5-c<yh-7AFPvnbmzFxdEH*@LmkFbC4Wr-{W5H0A?0<`4Zd8tnQjl`Au0(%j62rYYIA9F93!ioB_jOU!w$P<jc8Y5;?w=`ylk#5^nxIrwxr_alJcLPHb<mpyUG2;}NDPcVg=$V(bEpdPEklS<c6Py3%KZ93l-lc#&40ZOxSAQMDC`lWI@?hunv?;B#jxGc=~=gJ6$FF4IDkD+{u7cOHff(az9#=WQe%MG%?iGA*4Ibm<&*`l@$SrOi8dcA`GS+>ReZtFZ$^5|JFTF1)xyfn^!f0$E{Ut%GEe4JBZXw83%0^zmb)tmXIC(nse=k?gs{bpKrSi0INzg6gwT$vxrLoTRQ)t3z*NHn=-(txF*QUdMqAzt6g$xU2m87VHzg#qE>SZQhHgsx0IDRXe(2vVAg3q#T^RdiZd%4Krz`WJB4Sha)K9r{3Z}za$bU5a=0kK}0e<_OGyP2OqzABEDnHU)Rc?<3Z7EeID(Gct~?G&mR(Xkayz*Di)6*iw<@CooFHR6=rjX6@u{W(EH;Ikj$_7lSA80AY1hcmEn;ieWPT#X%g1m9l_98RF=kHvgTd`v}4BxnMvJ>4<uv|*gg<m-8pBUvrAuRpPDbqz<+}mLsvjw+r#T>H44WWl^!jmiuY{98HAqKKKUdpF{sJTmYM_Sm~LuMX3eYa+*6K6bo%V4#_4o#sgJ&PRl?b1N3`zs8c#{Cy|_bOI#QJ=>1*BB#QS}$8CU^V`*dPP$6P)--b?^t=MS7rm~+h}^AbD7DDbG5V6=nuE*skz>3P_)O~tySy0$zV&Hf(MfvwRLn!FE304|xW)T@!RIRK&+9jgCCZc>7Sz!kxS$ozkv4Zdd@Dnun1&>oiu>{nrp)L=m!9Cvx0Xgd^Ha~Xibl0)afO3{elO#l7f?EqX_MGHkEq2LvXJNfFhnI2Lu(?~R*h1OXrb<@vv6F%x%^w{#PHVs<*!LY@qQn-F-6c76nw}?v&BA9S52$p*8lmS$2+dct(gKn$PS{pwZx%K`dC#Qjv^*#~;b?6XGZ~SCLlrw|kEnN0I#%$<)Q|KIA-q*7^T-tckg0lTqsO8$TK;+fr<z^t)-qp#&@+&5vEP6;v=Z<4Yyek_NC4dcn3j3aL$*aNsk((nlp>}A<^m)(yoZ==Gykd^oxijHgG+N1=svP%|RC)kPc}WmojdmTQE{Cv(;NJN)sqbV?SBrKtf)zN{Oc*>L0TA1d$+`l%i+QsJCV=BI9emLRFk8}|S?&!qaM+duS}noFtqTAQgTlSO5#2>j)uzw`HaofFJ!llAL#BTbm_mH0y7);WA&fP;TOu#R`CT^sY%9S_vYFssmv|d5_;p6$@n`h60i)?hCZxq@mFB&(!M-cVziPyNJ}NiV;x9sCcXUK3X0lF$E~UxnF`z1<+_|a3R%*G*Jsp^AFU~~<OT@wCam?>FjMW3pUD2R=n1NF<4SjbqnLzRK36tgx{{0X5o_847CIY44eYthVRzrD@$TnzkR#j?Wm3khGTGZXp9S3o29%_pN-3T~%RW!bvpqRJ>G4NX^?S0MKxsCi<gtNK4Eh2rFvGzuz4SBkoZOy_)?mvw2Fv$Al?QI~^Vd%*|ECrU_?kt@ifx(HOOn!iUm-Kfj$CQAbp3kI`={en#xSA(H<s`lh%0AkEg)rln+|DmBWsSK~4HG>Kgy{fu%udNo|3=h7={-Nb)RJ4YhID!x`@HY7zAcS51Oremn3J4}9G-G5H)ThkM%qk-xC-jxqii`u1Y_Y;47X#OOw~0<+Ut}j{(&O2yi8u^{k3YXQptxQywmH}kkGA(IMSkWbJLVb=f8AqIT(m8%8FBBfc`ER;<Aet24Bo_FY4}wC36aw0_|x111CBgs-uR<(D#XWT@bjYkF12~YiuJ65U<&4?z%M&&GEVbbZfcj>%BW93^0x}i2n;R>JlbW!CVE*X1~F}mG4P*_5Hy0u^0hH&uPGfs@ze#SNSBOU9TkmEUP|Y_+HtG0`O#;1M5QMCmg$|ifC44$DC~IDWios!-7dU>OSxOfde4pn{A1sXPNG~gy)J(3Q2Gj-2XgHb7Oqio3?%{-U~Z^IF*f*Y=eC=X(!|PB2q=T)WztZ;p;DZ6W{qUx<*GbriyEmt>oI_{b)J^UlIIO5(jE0cGP5-$)AM+3_ZK^Z*nYZI*1~Y2VwAmuUS#Il~C$6;-J%b3V_F-H`zY$7472i-^QGjU7YbV{usZA_yt5mbE&%h8Y&P#rj1M|S#fj(LfHCSp0~2=+>P+J-P2!M$XjlTNM$O?EhBMw<Z~Xs)%v5}Yzb1wGOYmr@bS&%r|o#w#$a)9#^L(d6)vum@e*L*R~^-rW7Mf`!azeZ4d15<%G}p=invtglnj=MpNcJ|Q9e)2ZE33?dhh$!7k{1h4k9r&tb-I}`%b#NjX6P>gnKYpe`yCtImK+5jBAq3j+oE?<aHhd{U?->t~b1~-;SmX@)bj3F_Jx&mp6M+DE<D43Y`>Aj4RroCv4^Ns5<uP&eyAN$<E_pQ#z1*5q>GabM`6QiF=#W(Mm>6MyiiuIl*N4Q6uJRDQ?r+G%Yr{)>W}&pCYubl^~3kqA!Mu8Z<)<r@~QuidXn+7aTJLv?&&4B7V{;I*@;{G#CeN2(zpBNqa#Jz)uAF9Xs3YXTEFrhyqx4UC@=fChxa-gHZC!b~AT;{EgVk-vHG8`RDjbL$x8rSto}PJ6DCre43w4oBL`_cZymQyqUa3AKQI!#toM5!)*$#zND^QTL*6f9qJ->mY9=M<>%QwXyFaH+f~fLiYo4y)uHEyPO5(=)6ipIhxo1W#^P~8|1|KA2aA@kgTRSgjesA4`I#(rM|qpmp4)wcA*>JxL+|%7g?`vXMh>91m<DRIZN5ZXiubO)sXhR9dZVggoBy^sC8&at7YO|)+@3w;E;1e1g?h=z{>v|LwsBR^#SBx`zb1dxUA|}lr9k>%+{kQ9o|rBBydYtAYLYyYLmTa3Nlibw<3K4f>kL?=O2o4bc*$@Fou{8%$Uo9r1$r<*M#p76J3VL^r<!3~XS3^RUvN}WOT+k-B+xBssFpYS4s`j^`<pXPZQSBVm6sHy7tQp-$0fk}twux=P<`e@%0y4#9Z+e$uU#_I9zRXSo^#&vK<w;lC4!tvFU3)G9XH0=B%juYB-b*41}Q@VgmIhwnMp{p6~~a*JVTkS>;Ka_E9r_(#hbIJYnX<v{9(0}eINk7&s2HVS%{pX-8604SLiz!fV;>dR7p-3y-+;!IrlY+_@Xb|(<<P$w|j<w-})q<sld{<OI*RgJLPbQi|`Qa|L|h);cyU)m8L=~%OHYW&o0)7OUzbgog#{dbOdn6Ah=P%Nl(K%*6EI<nEUr!IujDuD?IPY7qKh_!@ov#c#W<=hcU#x>LZ8}u_K~xE3S2z@H;;jMeE<Zs&7!g{@VBeAO5r|2ErPMKirxH^>Sb&=Z#T<l+f2PdUsx7WTM71z5Va!V`=uQvf4pZVHi4(2Gj?HvZWvgY|92&4vtRwkxR-0q7*)-5k{5pRQg&RKD!C40CSIq>FvK}>e$bf-rAfEQIUQ|ObDP~KJ*I+p#W;&-~3k!kxF>t2P-2KsP0@@=zapogwYp&T>FF)5cu4*YboWhj0Y|8q+;uR?t=HtDQ8^j)eYLOC(JB$&HVy+a!WmS61-X3rc=Zn5Q1dfSd#~%H}VFgVc8~Cf_3ODe~3Qxn47w`H=}(S4b8rwlutI}Y9b@eN*l$ZeH*BJGq!a6hF$lMUs;i>2?+7x`JHy2@_+iCKbt%#aI+_Yo{hol=EtnPCBSzklD4d95&N1SnR9*?hEXLOVxfMB1f)2zNAwz2uO20Q2V4OY?hYCDhuK_qS2uHzV+sALUXXZ9x8*6tf_gxrEP_AVN~JM8wXYOa78RWk>~%(a3d>mT(S7H3EMuDN%^)y-J2-w)*{&R4bo@Z?_KC;*6vonleeiQ54rE3*^P~C-zp2hfDUUwssCQ9wu|tJyMt|b3$D<xYr%Pn@!{}?xGoP)fN$h9>w`s$yNu2#j{Vs9;|1jps7+=|<x-vkHPDHLR1a5Q!n;f4=V=Pl9+6V99Q0Z%<3bx^NUz}j_h-udYG6Ti}u^<;Vk-P|P5G=0Goa-4x%}PFFA6Eot);taR7un$W+3`E<_saDECym{TfE6tJj=}mYkHn|@6W#xN6$|Bs@LT{?dKN!WrK(kjah?)-cX}`{7w;Ca4!*8qdY!>-i+>iGzouk?#|NJyi6^SXJP4&2Jo`VN-SBk9OCBs?9clPO3YRzZhJXFG_;-!@icI1!jDnZ=6{HZyVjO|0uBp#kHjO?pHYTWwHaXuHojSxSIK>MdxUZacJeMCX1AG*I1OxO$9gTuVzE(m3s>6Q0H81p3$HrON>m`)c=i2L_0kRZ>gZjDBCgW39Q5~+5+wGUstA<%Hxo7kn@nI4SI3Oy)o04;ch&Rqi5lINA2$8R64<#}_b0oM=N_8m!whY%cDv~NX6Cd-;pz1jXwhYFM$c-)eT}U92*Iy=Rq|CuXh?T`wZxP^JrxhJEc)7Fg2#y;R-3(a-bc!#Q;Z4(mWFIv2`l&I;G@re?&5&elI<nVyG@5@^;Pc^4DNq9L|0SnlfVH-pngwKCMl4CcAa;_0BV0=v{cJr4_>+xqRimA3d<A07VP88ZS`o*oc;u1@A^9-@TQVXMsjn65Wk7sj`-aX4v-m}M&P!cdO(O^usvK7f-$6HMX!i)MeJ6c{C$ojxpR!aDQBg3|ISIk2Md1<Bqb^meNV_p#DX&&C&DbTDsiZ2m2CsYITG1w`HRc<yZk-x7ujW>_wPB~|gH@*frlRka(9Faaij43y${~D(cyw=Rtp*!kSqNy66+(y|+fA``E}w&ZHiyhre=bUYNY0XR0UPDUQGKB`aR*!=aVC(VKJYM*+tRGSew0$?fG0AuPx|pV%vgzWGJ8q^fm-W*sByg7XhqJwkIw&wJK~xZrH(O5B=pz@;&)bx`A^+4pw{bt$U;i*TyC2JRYq1dlq}Eg?kGhz)mpK>)=<g0fxyvwBw3>PDdAGO7^JHq52!1LVWFbJavQh3WgMn+7YJQNy&wwO4b}oY<VvZ6ow1#QF<s**k{ss&7WM5rjG3Mg$gz}gizxQ-+FK-3A^_Nqmev3#!%s^B^U+dzPFP=Nx4<H9#S}vir}!QLz9s^%1x%%)dENNNJ&Xp0%6G>@FkPpBXde2ex*HFxzCM_`P!+WxDiUD;dLoKl36G<uvQg90@+|sLg#VJ`RW%iAo|lOD1`TCl3=*`E4cMJ0WC?py<Q}*K$Ivt*PQwChK{b<kO|h|p_cIRsFwu3Nk~tc_xkT!$0OzjKD8C9rfii=x!~YH50!neWb;BoM!B`DaL{5sl$BntniK~+4pDyy!^WiH(1MOqB58;rFtnawj<hR?|Mpu;>h;Gzl>bCz9IDH7kqz7_51CzNOx06>XCby+GC;Y(^g6K`!qmE0Fc#A0I2+rEBm!$-DZH>?XgyXKCB{HU#i(+Ra)DUK-h389_ZT;D{8&;x-oSsrp@WhzmT(|)-idyAOMW7Kf0CJh)2J_~JQVu|r-G`_8i{0pL=d$%<i#OytwhHf1ed6~tuo305flWB2v=`jxLL=#WyFkd7D@rhflhe*ot0MeUfE|m=C~oVe`duN0sYh8AlIyap<??{fK6%nvrmLmqHj<nDL&c&pTesdYjc?!OU(xjbYwmtV--u7cLUZ@*GM6i+0Uz<qTnel7^Oh-C&+m*FsSQUK*F+yKmsyd#p28?_`7lhps{?X;DDV@2F#$R!>^w|pQq`ir&a9EjQ9)+D=n?w=@8vU&@CcO+vN@J#q&e;&LJ}x}yY~-CU)$K7mH8wWF|`F94%M-keldB5EG+nBiqjLihPwtXoFWvK4S8Suhz+&|v06tIl>bgySYWssfDt~YOHG69LyZ&lvPx}H`&1IVf)d#c*Ry)XqeDJbp=0rNKv!Ao?#46(&1#J5TH~dGkpT9TTvGkJolcs6M7URp^ZP16Tz)#n!k5pEYDfa?%P#x?<WR|l$}t#?8fN&is#D$T<8&}LX)s71L|pa+6={CBv>2H;ED6`l%D!tEB7HXhiE+;@)`j^5W%$F#xQe=>fbwme??uR?RPBE9g=cFlOC);IEP8qAG<PWmTLrRMY74O`r>fhUH5CMq5|E{ZxRV|Pd6_pNvAlL1Lyxes{jKqQ1DZ(O4=7;8kY(TpI&#P=)87i7O;Qp~=0d4ttm{quCb=qGI}ziWqPjA-4os;Z96ocu@5bg&F~%_oh!tL5?qQ_m6Wu}xcW9rQ&<oWo9@wcUbD9HU5$mPrUwozrR>#DvKKK}0n!`_|Jf$FY&`iZ8Ze}zzeNzDZAuLMJmSn<Zwgz%&t|(B`!1(?{o6>;Pj6LY_xeI~f;)9#iC8APP+4-hsio!Iyn?&-_kWWdB&1Q7eo6acc25{fqez;afW^vl8z%}38gjiO3dCL$LSb|ZGuHxBRs)*7Qdcln<*1=p!$KuOMBv^H3+L1W?ke|lknx}lePuuZkb0>t`&1z$yhDnR9xt**CBZ!|hr`y22)nft$_eCN-%Ej1GJiUV>QFb-~g+|D~BVCnP;TJopTC{mw7E&Xg-6M@<o;90vqec&)dQc(Ea8Nb=azf0Jth>Jr7FDARe@s-{jo|f!sgn-tD$#!TQ__D7HA(|Bq(-el8DpBh+{)vdR&Qq?TsxREpLE``c_sJ3z1<9^Dns$PXn!Ht{hQTI6<)<0_ox9hrurRt!lfMM#uJ?|1<<BbLYmq_{7crT+$wp27Jk56k$#?(6=<(mmyVl&@=o)`lE9nbhPrl>q5D<X_SFGAmOhXgzM(2&jfzuS4+>I1{2?0$%nS0#2A0<J)PUo+KX|8O47%%M1ViiW+G$aK4J<6A{*LJ!fwDqG{5@82G}K%0rCqCSviTRZ9oZ}IoJ=P2!FFiU@EX!zK=yTDe^MjQ<K#Qf18%EodmsqT4J(-_Ghc)1cH2Bjw7gk~6Y(fsP3yz^w2hdVGIosao}Xa+=|Tp-?*%EdIvoq#P4@b&7gzY~Boq1`|IQiUH?OzX2ua$5nxsRpv{5A_I8-*&!jw-2Es7rsZx!mY)826;>LD?VIQSh0?z&(I1q0H+C|H~R0A4#h>B-{G)#4htw}t8Og~pd@jmgn7MX5z)I@o1Uv#^@7F2Bdz+;T6heB977BjohINZIbu<#@&;&Z|IEA_TVIQD&oYB%71&=Z^_w{>Ht76+!-Lu91o^%FYjCGm#W8w%Wul>3Q|G>-RAs5Wt4E)9#;51U$5NOjrjLCqDs%u+Br%;=3BA@blilBUqyOvCPmPBmiipgps&swK6AUucUMPqBO=}VW)XC!%1nsHJ*_(?|lV#L2j<$YIU^mAL0Gevq!KNB*+p02XbND9gM6)mqQgfXz^57#XmZI&^l*f_4Ph*TyeEHd%2tGL{R~wh)|)UtSbRVx?;4yhj9Jkv)Tf$jZ{eB)ExtshF=r(&u-N>s^?(n<c>Gcbh$EPBL~-_DUm_{t$-ybM&T(|0qZj&l^``(I#Nf3`~6}Af$c3UG}d%;)n4E{5;%rJ>;wDldo__B&PbcocA@;qWa#7uA985>y<OC2bXat5wt_aJEU92=8{`wgzgRPaJjfWV=C1bvG4w-(a3*6ZwGO4pf*J=rr|W{MHSsj@IBb+=AmAY&_rP6i!2_QNE%LW8R!ijNC+<>-jnbo5JS`~(I2+vS3BCkU!!YH338GMR#NFv-cMn`?-Vp!nu$!qgd|9phJvZKwJfExcM$$ttxi&HPXwU8@1MPSX-mL84SKZAi?Mq5_r$Gv~;%PxhO$uACQKHd7f@g^GYL?7V=5W54#2flbo6zxL04znT5&hj>69p?HJks_rGH-i>N#g{~Qw|HV&BacoP<^U{7+Hc9?u!g1-J(`OnEd1c8#d;a*R?HBn(o&NTS!0DrwcXk4HmbhGM}h#h5_UKA?Rm=KyemGrT6t*u=Z>bY3Sk#%*z+SmQ23zQ}2C+uIXO(o=y1AAjAmOqcC@6<857`N!lV!<z3&aLCUKh*F><jn|g}pH2QdO#y`6#<ni0e*-jMG#<`ZP#fsP9+!?&oapfY9LIX<HEWP_KR`bHPc&FuOs$WY0gcI&Im0B4NK|a+()Uslf;Y#>qmP|F=GHsxwFv$51Rz`iaJUpbMVlPu+;=oC`_o8h)^lR%wG@RTQlb|HYp9v(X$pE;23hFXxxk?>+R~Zu9UCfO@b4C=|AVQ_MrUq@BVyhPRmv}UnHt-bzV53Y@Dib?uSdoHhxIgd>3;LYM55GgoV&+lQmiN+I+}t#%x9H|&Qy`p;AZ9?SNkXdQcs$4;j>@65piCSi&b>c7?s?kN*y#B|@N&frEe#;O*B*IL1l70vP*k$kEWp1<p<YfOr=-A4EO@H`?v;b??)lSso2HF8j+u)8R_`-9)hsBVDyHzGnFM;q2=!_1GupxRgsvq72z@lygb81oRSV8+oAlR_r4RdPN9zOrSQuF8$8O*Q4BZyOjldfmoSAD;^fm9*&*0wpUfv(6E{l4ay5h)lLuGN8U^^Kp@yX7;9+|c~z-%lmapwY&qHg*;57a;}qw!$%Y}nl$@8&1FMw+0#Mc6%`qOIgte*{cgub-(>_J*EHoR2bfo#hOjU25EUrObHUG+HxINT_L+gr9W9cK2NU<mn|mv`^rSw>@y#AEDzQLB}xpMs5L#MhJ2fARUq+TG2cwwL^k4TPoREd-;<ux*5v%?BTn|MM!A)FYE(#2m6}?!bhozm@jrhXAE4K{kmsArX1>wG?$e+azFjOnG+~kE7*n}Ndt&Nl3(rvb0snjS{ck_uL2x`+;8ilUo$LmN-!qQS~IM+@;m^%qCp-ur!O$8gJ`%o`gYb$u}I6-bU59WN<vh3>+}#gsytP%3;--=FU`=l7`xf5bYcQ^n}+%v`!&be0fT|ikn|E5`{n}6^-2t}EqIrlI2omu9$Pd&J`<UoEG*7-8lSxm8&3B6cu+j;;6}qJUu*rg_f@Y#t>|o$6)fH}R?uK9owP>aED`g;5pHA-HEK+`o?gHV3N`jj_>K8)5!e9Vov9=&{8e6EYx-^-r)IZFxwpl;hQ5J9!Zt7$i6^5x;5I|BO&<<^xYe^xH!dB=Uj2a8K|uu_VRrsfs9%qSA6Ey(YL0~|h@H9j)Z7Yup2X~SZ7BGr$I9X-b_5(q{6V^B6mP|jn#&n2{6c2$;NXJ@yh#C5d1kY}C*z2%^Y|^aL#Y&@jJ^<kogepR+E2{IB8x+qP33UlKc2$~4db+=i3_=o0`~Ey|5R?m;NHA`qRZva$$Ip=kYn2K54SxAy`i|sD9I{$I`@$7iP%k^s^yv5;z|{YJ<r{+`8?~|p!lA=E}lfWa6J=y{t*_pDBF?ITSbf1jicB)($c~w&lDIr_f#1`N5n-e!1q!ac%zU53$k(11_IORGNHcv)CF07Q^`fEjs&ti?-YU&u-uV5kH-Z*KsFyY8Zg?*BK}k4b<knDjO}HT^R?F+F25to8JhwEU(@vS=3UiCVo=2R4nQQPeQ>j%WZn8-KV=Tm>s94*0XbjluNk{k515d?Uh!ntd`+xsA&&u}1agxp*yHvQx^;<HFqNw5RHYtG5`*J3?!nwro<sE?=O}3=1T}|llhcP9MBz|10T5vY-M(i?!AFKUUJ^Nk2?U1vF7%UZA)>7C><g<d#5;0FrVV^pFCho{qdRv)&B&%&xvY7LG92o^(f^KnMLRzy?CNX&@XedMC%clt*+`0JuK`}P-RQ&iDL=eHX_S&;)MqUn^%Ppq9}amz0KAoaC;YMSjqMIph>49K5x9SH6lNG<Cjf#mKG-F=Zd#`51-?A9k>+M}!AVz}54dIS(;c#Uu5yUcN%+~O;n{OyrLnloZRMwB3DR*GQL?AgWVFiSPXQNQ^4MC1vn2pZgI5WJ)X8W+I&Z_DHZTBi(&MKtl{cB=VhtuJEpa)^utfz?7FYnFp_0`mJ5<tSH-LA^CsrX$204#$=TbK#JSN&K-?idBHyJPXu4HC;11%u^*jkhcPCwd;hp4f55fueqUtj%1j?g2hNFkPaJQMEi&lRrRYU7K)#x&7tQ?1j6jHeNlR?SASW+^K7E@7cLeb#=>E+{X%O99BY#^t|YlGJ<vDom44>{RvZ*wm_1Tn5}7(Z>cvF<B%t*BeGYMVt+mc`cOg8>FQ+V^K_OQ+Y`Nfo3%4)kSSQD>RhjhQ5sO%iDiOU+zztreW+O<irqp+KBD`)+bxAPGZ4b9FHjv*r;@BX3TC^HJ`(|B;u}r$30a^u11EHJV-|(U#Q6clJa8J--K%FN{2h;U&NIP-g4XCO?n)@w*MWyjvntg*$Z>Yqyx2S`&;N-E6B|+h#@#;pMw#Er8=iQGfq$Ff$0Vkqw0jFEN!V#hM!zIO)r(e&}knGUj9wQya;XDGMd&2e2@t`=F*xl99Wqd1nxyasoZBIa5rKW|NZ{choOFRFKqv!5Y1|T5Q;R4-0-BAnxt79AAA?+9miZS1&N>k5sH^OQg=L?+8sp8dNR}~LX?An$gJ6b(>>wJCU}$t$+woFUs(%etYb(~Y>|}s9<(d6Rjl8;=teiHO3S(H*9h=k#q6Z(lNWRjTIwf5$w}nQBG}w_kN%|FKX{LywsKv%qFr<DLrX;W_uHACe*B~U9B`v2H2#9WD@+dR&Tdy*hZ+mn*P}9K9d|dEMc=^OapIt2-SXY!wzcP{@r;~#UF^ux0l0_|<I!$oDc7^i9r}$nlHj0Y=Be^XH_JF&{G9^TbXC%JOn@pqN<X!BH(1@?AGvyzn$msAD-%<Jp^_pyFkOyWarmsmfB1)X$7pziQeK&8C-H^`#xXJVdho~T%lnePEv~20%Ma$Ms`j@@ip1M&eud8Df=ODF*npo{A*JY)YUq0eL{Ag3<ZyW`0e~AjGW%(2{5PS7t8pYAjS56qyXkV*X=Jt6#wje?8RSDJ_$tftx^liTXNhOeB=NnFbS}{QjQ#}Kc230;%&}+(#Y{+lBxxY)_GG+0RSgf>_v00Yhvejjmt<t;^y5=1S6^^N47JVC^^_ay4Ju!6qW^hf>F1_v4x^wZLj$NXR`sX9ZQS_aK8wW(ENb)Ukeds0IMeWPOX8%O7y+vw1Zk{@dw5VD(79y1(MYi&v>X|MGp5g&MXty~7!of#JSJ7jXx@YFp4K}cHhBqeO>{TZ>UGzmBmvF~i#BGw0U@f6^Pw>xLVKNN*i+{VI@RW`OS!-}e5wJI^cH46Hff43n6O5zt9mf6fe*M_FH_aV{Qp=;{&ReQf9Q@iN`M=RxG^2z79dQY5r)3oE8uk47EN)@Elc2NVlI^i+Fo%&`I$6OZitt55v?*>S}N);xxXl5iL5AD3O07wrNB946^maY(rQjR78LG~pi$K0l^wV0c0sbls&xQ+H-MWTObj*?6fcM!kJJ1BvDeB^xKBSpHvLEO?3?nUghxQ!2rA8q30aTypeYWD0u}A*wFrzFnl-$esnUwX!HoZ`$+5vutyyYYk#&-XHuAx|#V1sJkN+CVnOG!N>SZo#K_!7epM<O4XTwS9DvR5EZrxmc-%$FTlo6HO($pn@_xlKqKJD^X@jvLo6C%;OmIjs~kpFjB_ns$HlR4z*ZMsgx#DONt)eBUN{wYIeaCyZmv!{0j%(|b>xJoS5lHGdBqQoiErhio`7({+k^+sJ(jhs4#4|(3Xw>H33zXB%M1!<N<P|=<8Q^QTfI5rt3i2ObF(r_KOAH3|$-LQ-X;tBXJ$TG8edQJ*eEbC-8eU1uZPz^a~!>vOUe1zoLdl)26M~R;z@{g8|ca0h$eL0ATpcouz+BVUGrvcvWQ><N59za~Q&44toJM#O27Iq~tB?IonyQ2frEP92N#?)<Lc_ekZxfQejH>(<drYCf7^<C7CwLV>PALSg*ry=R`k>exot+cG076WzKm1Mm+<U`S*f5TKHaWRTHevL+&U|8q91X=f?wqJA%6TuKd75siQHp9$<jw95dH1HNn&{Jf8PU0M)8^s}F6^;Vo<ue2xpM0(Abxt>}LnZPwpWZ2oW7L74@K0Wu>@R`;XOVE@+rpP>b#f>J2Iiuy%>jAJ4ghg6Fg+LHfaBbEk#YI`Y35Z9pjoO>Qy~no>+N%8rxr#ms;&Ggqawle0cx6yV>NblQx%~d{r!O0j5wO!@ha1Om{qkR=vX5Cvt6-r%G@4ndTXTj(3QWAKo$=!t*Sy%KqEHi)GQ91Ovtti<7&%>xL6dq$42;t&-u=6fO|{63;k4V>$qeR2wIy59PP5RJg$!2wl25_T)HK5aM{&DeIa|*4v_FSG<x4-8t5rmafxJ_;;VcqA<l|}q4}swl`)o8<I%@7+4I##s-^Q!b(^-{9q<@~UBOc4QfqYLYU4-9+&W$V<*WrL-!nIhFPY-{*gCEP91T4`z#A$tkl9jM{5Jg#5AQ33@PYZcCVc$YY(#NVM9;)$OY@w*(@B!iq3Upu5h1Jwv(Oh32n`yue+M+Xz>g(d8zvoGpk8I36$+f);#mn$N>6C%G1@RH`Gqhwy{vkeFF?L`@Y8gytnl8U@0exM`-*E>4$l9yxwJ&Z#uhNcVMUx1x{+iqtp#`*@)_m<M>0_Ve<z1*u5S5>J$8Ac$LBjQl9g2$5u%fIsjF&8`2z=3$9NQ|nLPm)Li4)x^J_VHlM{}_Y#4@gaBrIUJwYNhF@MVQrV^}hJHKH$DYfDEI6eDJEBMlO?x&`o>|yfyV1c1+bX2{^d#SVz_Y%FNcz0#bei~@w-&uI6bO=#%LdKVj@?mxGO4&vBu{|1P=r4>~WwSo;GBWcF5t`h)NK#t4Bc1Bb63kj@b?OlCzO%atrKnl;Wtv}AP_>2x>?6V7JUK-{2Xbqjz7wZDT4>^M;DjNxChWnat{}8`@%uV?;k4|>$uuOes*0%~U&k3A^=O;FA12goR8L}P*!J|Wd}^MIDs<6>_-aE^AipH6cGk{1FVFc<8};>P5zL5jg!D^PdISl3TQ#G>XYfN(<}K$!zuON6M6%%VdqC2-8mI&4Pd2uvEHzQeHSUTWHv=i?sM2hZ%fTqMWEOkAWu_@T_k|62F`o3pmYqLgDB4S995%gd`z^5vIxN4rR0aLta)|4u=ciq4u+G{8-4#`f%6z=JWjZZrZWujl7N?|Z^I>MvdgPS@;_)t1X(-dY!d~4cG7ji+ra!KT9|gXEcVEs6inxc%{K4dTMl9OP!%O2lTX=|PvqBa8)_L(6+aqN`ayJ4-F>$zm@d`7w6sPMlt|l2%5@<1zqr6I#U*cBO$lq;qH>X>!+x(G`6IV&~&-H;TN8w05c7~h%*u-%8=~DmW27SjErG|K5<us@!#p`P_dKApuiA4UycEO^PLB$Ke<H`%>!m<k+HQsJ$QBIY_oGum(eFYq6HKg|Akf1Fa3>^mha*r%WeK&g0(tnH{rny&|I?WkQP}B^To3;`cQ+mDTkFU<mGyOLXKz1w!Xs&#Eld{$Jh?#8;CxPwYbQIumNJ2_g)4_Y5*&{cI($^g)D+U*h;N9w!oe)bVj%LqF!mQVoG{xAKLU6z-uc&Z*mco8<`81o2Yop;d;b#%;tGklgQh+Y4c*3+l=wu=#)VhAq(S6M^T0x1KWXN^pA1(!xRE}@6v!sir7S7UcZBN0-bo?m-$?{AcL))I9X1jka6r9}$Wic{P%Aksin@V=j>g<B2RyAQWa~d^54ci;K!5;vKVBov&(-OwiroYQX;(_(SFtj&$lV0)g45%a_^-Fxf#geT(#bn3L7|4U^j3L+0_+YfRn2W#I$L=};Jh@mg(4eJM02a*OPN$~;=k&alCiXwQ+=~ev2xmk|z-D6NnQyid%HVU>^K)^MA*6g?A#Z$&YPiU4&KIBvzG7}qN_jeBK_~C2Y^gyppwy)1s^_hlw0F?<H5fh?%-;j$l~$n7KSx#*77cMIkut{qNt(9>Ay&Jniv}=H4yS(WbrIw0uepI_e$O#~E<!ETJ>E`SdR#`zMj*dzQ{XZ{KPAKlX(aC*euur+mAEnF4qHds7hs~F<3L?^#Lx*L_i2t0#OTioZ=!CSz@{zCt}GWHP}`*kAdgsU&q_&r1Go%=PLH-fYFnIJ%(T5bZ`}fqKZ$J;QXSHWrtqb5sJa9Sqn2;1g%uGYZ#|1f-pR^X4@L<)uSE263p9l*o!kc&5hFAe<ggN<u>%pum~}Wk2q|!TsSxE?4$eS4WLQdo1zIkVC4-?!lWfXuK=jDoXzqfc0q`RFb=wX0nfHHnf>(p1MK@t9%r;PEoY3PE@ck>+2~wg#t}h&w@wdxOcRQCN1Yr>*1LOdpjWBO_s`OvvjZ7aiZH!n>@?fKL*Q3^yX11Fb>DvTM;KMdlLm`W_^bhpbaLMk5Z?3m;)ap!!?h<a3=qFD9a!+;poOYR#XCgVq`JBixeN~$4Xs@&Pw7~BO!>=s-%*Cs29g+VX1}MaK$9ksOx=wYd=>If~y_NQb1?WjnqMjCHUF8@YibZN}-bi6Z7CEBepK0RZ%4C9$#H8F0oR<m_)Eo@-<<9BsWf$0in$YMsqqkf9K$dTGDh&xuc_sSaI(r}~P69@MObbwfM1`GW(*-bcea9&@I@IJ}kxRxJh=8dm$v}9XLBbp`<Gg?Kd257PpAol4X%G;sSUv@jfyheMP42`iUg4l5nX34}fYi?qs4)90WG@jG8tkd7&{HdHXlE%w<0gWX*_=E-8||#7IOQh8od$I(1W;PM@`pYf+8nDpxDA@RY%pkZia>|$HIb;@4~)lDXRdRvwFh{Cx#qywnf`Boyc-(7eou$p9E6VkAb%MNAx9Icf-V<K6y8@@7q5#!v@9p%sP^|QjzQvZv)d-q-Yljdl7n5#3ev9oqX^h8ud8HLV3)#`W6nWIT-DXOXfnELJr^|aeU7)fll$xz40`+^E*&@-U)3{l21Kxp-%5zZg!tnW%mTmv{i?bI>qHK+=lKsS!KlMjP8%U*!kY`b7WlkQl%RmKnDXAVK78)M*LvDSu8+WEAY;OR${XhH`(jOmcfFp6v+KBvAbfa%R@nI|i&p+b&^rNHY^u%kR>BIT_${$&amAgL&^5YV8qPgR@^P&I+|7VRQE!LB$TDi82Kd#u=5SjLo=m&=NBOwVspXskqx#@YB*H$*-)>``y~aw5^K%Sd&afIh?RvORHI{?;GrxkKhBWI4pN*epcS>CX!;PPjfS9ELJ{=TAiB<pl#<&lP9>X9U>aK~@p;#L-daH#;YRiq``&ZtS7RtLleWlgM)yyQ<Sh&IXjlOU!a%eV61Srd0%P(~CHRXR+#<$;%jvt#+KU$nelr7--5jDRU3eFNlO_;o@COyI|);qLhPA|`geqwmL(t`!5P5O8;$p*fdXs#Q3&M<TQ<OOPPFD~vFjtaro&j#4*47c1AEMwu6u#2*|oxYQ$3LC{$p69%jvdi%r(y1KBtf^{|_$XT5d*4AMkQ4#(tnuk+xLS#SSMLe)Bb!}#*7%s{pc1G~OU|mD+AONC!m*SF5nO5)vN#2veGvuJkpi5MbPE_txVw|_W9-R_xe$TH>&p}&bn?IGME#n`mPg;VX8du>3rM_XDJXkRx;U$+<!dZeSTg37SXn0w$Yg&`*AQPX<7kxP{G5D|hK1l<ZX8SSun5KWG`!AT+eD>rWb86XLlL?dj-NXtkq;kLw2Cj=qdF*O3RxMKP?*sx(?{q$Jt~dIgVw#GmFEG*$?4c4!OSb|8b;6LVc!JLf}>yyZiEW7P;_k?UaI%+yqq~cDkTsZec}>|8SQk!g#N0<Ay?ox)Gqvq+GO9&Q$D&d)cpfp{6)b#(n#VXAJMb602-wMtU-=p?_y4nB-(s~`QBY0FlErozlYz1UQtx_x7UOv_BW_^F3yu@9nooYMhCpw!TO;9hW)P?q4qO*O;l~AV$|a339FVLP55g$HMohabvvY+>F-G3Yh*A_O1m*+hm)>PP8o67xkM=W{Qh@YspvIwfl-8`8|5?LoeQE$z(U9ue2F+A0yYy{tBcpj1LI2Ib9rI`S=o#d$N97{vLM!q!CPF>?sNt-SYisD_>=Q0Jpl5}xNd>E0h1>%{W3ttu=Sw6er)CZ0RW&y5NxA!Ze6n8SDPis1tbL|OBZA@@b+W^jdaC>K_4rGF@IgnR;s1L+2U}6V%KzyeU1va33WYJBR;n@ayQ|#hG%tY?zsi0m_LesiwA8tM~WkJ-#(3yUpN_9+myW@en~nA*zI!g3$H;|3r3qS<7}3k<Z$h5m5iLD_MH&iI!&1&*K1@jNO#Q*i%tkj9E*G-M(B62zXLum@S_l)+|JY)2ikWr-R{~LIZxkLAS^2jn#|U)+NqK?kA!b`HeE2NS_V8mNh1l2rH~WlAekx+DaXxKK;sTT+17j#byGR}QxTa$b0HdIy~rI-^&>T@*_CLcSM%gSl$a(T$?PV};1HKdc)0wTN%&)c?tI2dlFtXV@|{z@MmnkWnjMQFTZqVr)O0;KRb+B!X+EkFtePMg6n^B;;%SiyoU1Pirc5dp8;KkN?Tq@i2%n*%5C#S6_e%xd?%+KkH&oaLFD<;6B-({~$mfC+*0f|V<-RLg%4(h^E<+R>8<QUs5D@iXfZ3-Bv-@^rjF1j(n87gJAEBSLljd5!+mc|Fyf&iCCBIs3QwuMl$Y4PqjqmmB)b3<8$8`4C)Z`y}x;l9w#QDA@AOa;p!hs)i<!CxfjBQyFL1qb|W5|1NV$>0|B!`?%bIB7$$YrtLdcOp&Mtwdbr3o>OKzH!>+VnL)KHzWf#IRyeyb;OA&cS4N=Yy7Np?k$n9$;&P$t}keM>6l)BQia<R3-1?X*F4rZ16DsiRQ$K%%(t1OrbVJh2he!cR=c$A_d$PlH>0n;Z9>P=wJPbyl3#H<{%&r?S-7Rt*`rTGIk}qHK1ihNLh43**5C@b<7-KG-mb&ZAy<BdJfkXkk*)~lG<d^5aBhSc5=rGW8IEX$VRT-S*C%2*8vaTNY7Z^G<RQ)Z_#tIX`aNFe>twXb6>0T-J|N^A7$FzDzX$)atqT@!(U;+>lct?^xJa^gH#Bl$%^y?k#m)EXye~eiB!P%k5-)S&NW5O#em;3tkc2T#A}1_^NCKTJxxLfEqFX=fBoQHZi&o0iiGo-LYsc*ScI_q!4LQU!rZn+vI-@JlZ9Y>(!4D3_KC3ljov{)&N|53U>Sa1u+O__<C@p(?w_kt@~70{Y3m7ew_e#PT)xMF+rNjTd|Y{==E14vxf#igzI$~#a_Hf$$Ga-=XB_I*Q?J{NB#<1`TjTGwt)fNog(Xg+61RY=7DQZoo<jQ^=7T%ic@TKff4=Pr1bKn%_Ma>|<ArNvfY(;k6delgHiAFPA_#1V|A>d)cxoAZp&WDOD!*7L47ndEY9jmFwAjAM^gnuM&DR;B{0KiVG6h;jx}k)Y9o01#?5+@sSp0pIn5ey|37bEdA=x~?w;_HHf&{<0L^YazX-g@O?Af>VjR0gC>{tU~v${jndY(J_A#*K2w?kH40tfT=-9PDSE?JDmLGn3?_!`z7P37~@1&_HKN**j5NF3W3Aen!O!2Bdb!%cV^Kko8T|8cAp<C>RN1DEoP<Vu4pW)J5sJbOI17j{ls;rr}t!51Ep25E>aPtnW8gYP_8m1(q@3>*=)XrO!qAN$-_atUe((Fi6{Ch+qjo~NZTzW6TKA|A-+@QymJ9&ZA(X;G1J%5W(~CyAMuqi*pEhg*hfq>-g9sk1xOPgMwMU(ntWv>A<83X9>wi!ulmYDK-@3@UNn+<)y%75`3Sl-5TB?UDdvxb<eCwo_Bb9LFc{<^PgTMJu-Q6kD&lff{QWFJ5~^N_kDvM4VsizzP^ZUr5&E@2%QkGpabzREI*8Hc#s8G{C=%h}m%JfbI%Vbr>ByF>fdaNKmXxtHweeCZ6kEqc{~d<&rpJ112=8k+!B{iSjFK>lq}@H?vdmOa<qW^>vscqi=UM3@EGJ4YobF5X<qa_IWxu>9Yy3$3t4jzvV7(A<mB)1@ijV4KM0Y!<0yQU+bS8W2;@kCKEo-B8MFmw*KVz(JgUPm|=(1cQ)?=P^esS89R3SsPmIsD1`A`$l2Sdab}OHDttp&XGkKxvyfuE+85&&@h}X4$im^?EomQYC6P;c@wZ$m8%$gLLgqjNf_m7QR-sDvbsW^0c*LSel*ZO<pP4q8H7$+R@+31Jrc*Ok0RwKPMjsm;Fk}t^{ED75s58HM1-32Dz0tj^6ihmJQwGXxS#6yro-~h9DADAp;EkS7+Hm<+@iEFDpYz>cm&)ziOH>4PUvH}{3B3-im}3?)Lr`WA_dhGPHM;D06=%DZ8caSRY*e9~Z_4S|H!c|vaja@;pT}S|?pfrXz!WlP>wSzHjLQLU$g42KBy<r!Rct;_pxBEjY01Nw&&K_P&%4ofAHf*@4izNi_J1NSZ#69DpNBr~+l@i3E<xF_3WjNRoQ}&RabzyCh{Ey9ZxAKmf{^T3LG{NtM@_H}FD{6E{)uBvJ7au|ZB3y9W^iMjgM@ovzk}EDvDH5X3QUTE>tyub+c&jir)5Z56c74>w?1tj?sU;T8H)p#;`}dgNn`*|b@rh>l^PbSf7dI{!Rkv90a;c3_a4(|5#I68;PVreSte0oWa}yZ3sr#d_Y5`MM=*ogCdMV8cS(rkohELW<qLwG;FIFaQ3;+#F9v@(<Ppl)12g)2pITzK90?}L^vzhZQui+`US#mk4?LdcVHYM$^wT!2Dcj}2w#Wf<Rp|SZfx3t(PQg-(1m?T?!;$fs*+3m4Y#YU41Zgc67@n0S@)PihHYW&~d2QfbG)3(YE7u6!q2w8}3MPE!?TWDjI}&H+o)->|13YJvWO(y-#$(pel?om=i06Rx%}7>ohV~Hwe_O}$Rf@8?$43tZKJomu*SiS6p__VBiWRM21`LAW$d1|s<Pa?W6L7_$2DFw^-?EJ}iZZ3Ktob}oz-+~~@UD=7yW~p3!8?VqGZHMnhmdr7HjNFu3BV5wfN*c`Q6)sA8HFbOxMj;h+zKf*JwVav^*u~Yf#r7VhWi`A(qxN64`^&#_LrwbVFyAdjb7A6TQr3UPj}6k`z?sGt7Y));@F#9Eivwc`cV_wOuF*6pcRQIPn_$GQ{rH_+~-SRATPgckPt$3=<L$mAJ%-ct*nk1cu_4Slfz)TT<37@g2E(7lYWtK^8fr=I}*%@K2wSudIr&(=%SEOZxA=%S3quJ$?Dm&3o~(VpkC)^zdI|i<ckn0&<8ztjbOQlmjeSsw15js<<gs;3>AJ6=jJ~VHeuCPZAdR7X3Z(CCO&F$4{0V)8XVklR&UaH>w!Z}k4Rn-)Jvvb*QyY|1ag4%FHD~HYW$b2SbZLaDMMg9@V4jqb%3Fz7XM<wOesLQiv}wv$!_8(;>q$^n7<qJ?7j$P7;oiX(=5L@S*e#|W(OE#w{#}=PuDIetVM8F^aw8Rb^AMVgJyJPmx|vVCE6$|QOROm2dREDNrw~^p3G~7gFHQOs!{<8`T}36tplYK#hfr%PGFiED7Hu~qsIM)Yl%Gxk|L|=SV7GXm-xiHo5W^jNQAbI^-Z}zkORF^L1EB}YQBAIVPzlN;2?FmQDEY}%^u^-Efj~d-|5T&=A$uuhZ=r>5uoFbt=^lZ{A_(!6b}5=i|mVuRu^iMFd8Im6Tcpe)s#WQTjnURT5}9O1fWsItB5{|Wc<QB4CG+s92*UD7CXA7YCK#Q>%D~B=hnw$LQlAk^W~#|l-cpLj_BI%iUz$G+R%{;lX<?r-7)=I6MXv@E=!a3;<>;8i&~MHYf20DkH*43#nV1sytDWswxH?T5LKukZ3f0U)Jdo{Q98()M{2(LF67Fv_RctFOM+G_)|xo=EEXove%=><bqGzBd+^wX41gZzp3opK=jcVV$&tG75Aq6=*o6gEY*Q<5Q}%HYXIEjEA@w~tf;(0?^xH4{(@-IWS{Jk8-DfoF#8v5N37YBEQFGY!x)k3}@L<whkdO@E7XqG)>gWpDwz6WGnG?(Y4kRdbdjf2##ymwkq%Sc4yIT~@Yja$`Dcj6=K4Vn6aY+0X0yp+S?RQ*63X)Q&lq?0Tn2^mjpwKgQdHb@9Gr@JjUqGZj*e8#C&M<#&*C}M~iPUhK!_~Op3sTuZ>e)?Zh#IZ^))gPHslCoYXFxBxXix0;Mb*YB{3Z8pnq`SfqAxCIm7{F!dTYphyYJ>-h#3hJzG;-oG<pWo1IV%!&kiG-gEhRIlbN9u{l*OEGZ{1)rSz>@V44rhO{+{5VXkyxPg&$)+?!}#Esh1J<f|fLE$98#@~G3j$VE=>V3^dlse3bKYZ$<gsz0~JdnUjTSM&ID*w3oHLy~X#mM~tymJTZpGo<eFR`UPY3))PNZ6#Pb<dqAHH6y_0?s7|j+nQXU_ol<SK7D!N$KmR9-^wM+g=|pc%5@`P)|~U#mbsgVzRiokCcM}vUa^t%EcX+U8&bBm#s*&9qTfLy5X#AgmV1DxVuW?C&JpM)!^$w%m-(1m0uW2Ghyuz+;+$!YiD1xO*f7FqCV!mU96tJT58M(SYZUOPE0`F5z3c0Gtc1Fikcimwgr~K|<t@IeYAT@S4fb04iE$m_;T3LF*G6Lf_Sk|Xm=x>*YW0F+?^r;XLwC1WPGB6qsE8LD4dBzb(pd$l%RFv5img!oc?~{~!z4^x!!3M?8^#``#<3Weo#Vto3!!+{1)|Sfs4+A)=RrGKx+KB$L*#}hKzX<1s%1SygI$yL&cGS!ijy7QafBZ1ArX+g*HcS;`)SW{_^iW!ekr&6gMG*gts8J>TA7QnUyYKpgquxGyacaDyQ1}H+9`R`VVH=sD7%F@g~>IC>6?HHC+4IWOH-0yF;00m(!d&2prpTpZ4(t!4a$tXPdEOzZ(T>r+dY&ee|Htq+<bHBRSj)5+qsXw{+`Q^bDjZS(>6i%9>^kVhKxy}VV^c!QyUCWv46B-++C`}<7q5%{wE;r%sTP*v60x;Ro)5p%V)bq%ALD<YQ}fWF;K1|)yP<Y6l1-6k!OgJmSYH<t+Q{9;5#SNIhw6dL#lhJGJx#{O}Wy|!HjIuegu6tI&39?+=ZB&nCdJFdMLNciVrb$Y|lD!wB-&l1;JgUYjt}>bU^2_VVZ#3&bLbC9qXg<qjpocD`)PvsYhbh;FTTWp1S<uFUzJM33Eju6}};UJ#Kep?zvB<V-}Nm|3Zf24{HiHURLx3{yol%BP}Be7R957HJS3Ix)!TL{Nb1IJ@&6SN#Vp2A)l#eDVMfXF^UWmj<3M4Hq`z4ibh?!5WP$Om7|^u6;C}6*<K?n2P#6_U?c?+`;SCFerzef__^nBI^#hv$yB2DQLUp+@Q~rKjzNW_S&g%nPsn6Ixx5<?i0(XmsNb>PH<_@ud50%>kx6PXwhN%)0#LtiE6ZHTjB>_td{Ac_1cSQKb-bWZZVlQcffsg!#q<2_ZsT`_P?SvkKvwNp60SJ?;rQa4|AeQU+JY<JZp@EP$Tlj!eVdw}%kK?yt=_FZGAf1&*Xu;*_)K%kgW{?Ck27DV1LSz|{}(1cu-<}*Gi8G_uolJ-e85ab@d|OTXB=VZoDDCu25@hRyUZx71x?9u<IZ~KO97mo@xTA*m8Ra5SA44m(^O!}xsJOB1UA6kvC`&W$Y4Swm-Bn_yRDrnfGLj{k{<lVLAoqUJa!hl+^bg;5E2Q&232MYJcF5{xvC#2987K{<7Rc)e+B^b6Av^IY@Fib4*$q9j_c=FZgpYt*F;I*E7=dU;86n#>>4Ltd^5M{Lj2-5T(`7pU;?v{?u@iIveUrza4BzMtSkabx0$Vkf+!N2=f$oG!xJFD;$Z*f@~(t?5!6hdU7oH^&Y*o+`@DTYxzzYF6)B<gEWm$*EZhfaeAxYoX&s?^cI-RRh#89@p3p*IfV;2i-R+SiG#xliwKRYzV3SP|m*uFmmhf>S5+D7WKTMwYpaj;Cm`S9CES+7V!#Q^)mfb|!3gYCXk}ZprmDE@@qfM4dWK5#Dq{q3+*St&9yL+5#vq`6s2QbdJA#&S6&UtGCFO;)-!(;ybrv9Zs!K7&y@dH>?MSBnx<m@iy$uZhOEEa#zzcR@>@Vx&Z-4OR}NGnWd!1l}QYtGFinn%iB|Hf!lpIoWjZj%DLzywjk)A7SNKRAkF{G+&HyXqu%n4VI&Jl{q_94<=c^?_1_ee>(IxWOB}ffE05Q4+>JnEMBlcHnF73n-yL4f;QT)ZfvB+P^&)aLcD)-r0|&V*Ao!((Xwzz2Lxc)E=t_(qffTIKjX(Ad9am)3lI$C)KAxh)@++-jn*A$@`XnXlW4~(}xr3e|}bUqUN_fP!ETi9fIIYDSs`ov;AvlI`)pDMS^cA|H(ghm^Pn@nc-rX?mdy}!?O0dLGbp}28a+rj4?=sZRb-?0|fF*>K9aE+m;n+5{(q}9M(t_vd?f9rm4<qw0jM|p*`h*%=r*ysV+Nf5>A^Nu8(_fg-CoU*WOLJ7#rn0Gyx0u3gCq3=I&@xY-{^PhD&*=_lmU-Ekq+F7$p$obfh<pku^%{OVjSdz~&2BPAa7-3AvsF4i6bA*3S*RM!Y`l4v^*|BosBFd;+!3`1<QVp{&v;YJ}B|Ojv+NKEK2=x9^hi2g^4lI(rQcG8--euRKdYS)k&oq$m5u|5Ct1VZ*`JV_DYjOI>A2{!sf~q*nJAz|T5OwMSgv_FWck8ky)lr*F7pXtY?T>U*6g*F366OT0R95@o-uU}Z(ln@+R>BME&2lwrfHzP->#^hU-!5(Kco!df<h1zT6rA1zUsrX6VnrJ8xH^NmLY!(pTgKuH8$c5h3SXh-=sJF<smXiewj3B=*|YhOSh7s))(74Scr*1^BUll!J4=q1AK62bx=C4T}wWq(;|I@-sC^it&4L6HZ=?P5})s-0G~zm~j6Nd8vq%sjc@**6>16A_7PzmckB(*YO*E>A|j++Y`Vqr(J;@mBTXxF0L>l#g9fQ`c#(M-g|Nr{Y&0R%46esgJ&=@XZ3eWAgHLbXs6!*XysV0Z$5<{YB-<MOCog4f_7U>U2d{@N_}TP(Whb4L4NVRR8e?k1RJ{74==y!FO?w)l%xCfIz{o)gynyX)}(L<6JbMqmb&8e#T1!Q798P4n|5favPGmmqx4ZYtL9EL!&MZB=et~pa$K+hDAh!$7K5bAWAg!X23}uiUXw+RczbNlmzG6h#aaH0J1|SM6p0;@5W-kv4bRcH)zKL2EMu*#-lwe>S`W%)L5;8C$bwNa307?h6S;_a4nVXX?CHTn#X77n-99(SzAnrY1a!ZZ974t3#`Rv#|WnJK9j)0DafXHt!cfh7W%ATPbqa&u+8SB0ZSliV62LCMH{K$yWBB7=%h!itC!VL^HXUEo+o^+^+i8hhXF?+?yFw+nC1Weg!btSzGjo(vsTE!vQ(<V*A47Qu7dIEf{UXrP!;qsI}Qsqw{K@$dU?%MNY?iIVO=N!xCHTBv-d9U0SuQ^l_xKyi|Ur}-WxOx?&`qL4Dl%R!B0{>5H(+N)Fzncts>N;4%wl0VuX4E1G47j&#8dh-x{BKLv7l`tOvICym~n_4PnRiQ+NwA*-oRJ$;(Rt2dTSD_cU61ZN?o7K5QVEUDGS{d48C}UKA40^vD1*4y}B!JXF$dt4-a`d9YOa|JGQR4(A5UG=q()22cJ&I(z>{;4N1Jm9RB!4er=tu<Fj2Bm|}!;1|^en<PGMM7{U5ER`(O9o(6P2@A5}Icpk8s$%^q9ws}F$c(KlmO|~XVtH)IDM4#O1(ZUtd3_~t_}gNuQw}9<@8%&QpOA)0gAL@WeB7`>$unFIPD4lZ!W4qaaCS0f5AwWJ4Hxl`S;72>lBf*q-x@t3GJQxIJ=GYpqWQth#j(GgbcP%Y3_jKKr$1x!02|+l{&^yp^Xvt=?L-X7-t*C;a@4lft|E_XEwT=d|2E$Q5K&ls$AcL}Ucqp=zJ%l0kmRcUk}SGtJ|;4M?x3A}$LHK|X>cNeCKe<3iKT{%_ymow3v#&)cyEbJNm1fU9sY-Drf(nec4qHD>2lkJ>0XRL`$rH1Z+bkxxvB19NZJmF<m-k)%U!x!K2$fiuh7|bVn|>#6uae%P<jwO4tLY%G^`GY8H&qsrG|HZS6=Ef{UaBxA~pO$G*CLG?HARF7b*ac?goGM<eE<2G2SDlvz2Rqx3zaX95MiIFUuLO4%6#wcXTBuRCu>yF0@ZCiO`HXNrP4H+h|j<1XP(Tm?g_tE*3kkkE1|Esxk$vSV8|6T5bNS@plk>#7<cZqnkh(|L?P6AHM5yWf}zba)e(N6P@44knTG!1dMcFl%hvk!hH6HZM4QQ<w0%K&wK9a^Mnu2$pbfhhr?tpwGyrO#QL~mm30BV^49B~hO{XXK)84U5Y~J9&G)*8=gS7Jk62(}ssFPC|CwSzn*Flh4KYceW2}Tt2<;!kh$lx&9C9}AmO}pdqI{Q++~mPyOfTLEP@g%JA25N#*{aAoDl9%@t};YV17u_Ov<!XH3oEe;O5lh8l-oy%)FS^wti9r)k+D&_15XyPD5m!hOgtTc1UQVFa!qWD884D$);<qhN3qnd8$f}uoYG&hN^de!ASjr{F+P>#@JA7M*VNCWzpt}9R6UO3b)nE6nLrTiC<!1#6db1&jk);p#P#a&IWvc#)R$b<Ft83y9Iq^n6#56etIN5<X0EsI|7Ra)u*TNHWL2!@TFuZfSJEaEjUn7=qfJ^*0-vcma41y|V<;*~bc3^&DGING@ScN6CE7ZRUjMGZBwiau_c}n{oS$tX{k60do_##mGt8wF(zsnG?r`sK&nKMm=S|o#4`%<Ws7Osr^8k+N;b3#Nv1ok6qqAnvhoI^d)VSuU(^1Upx@NYy8)(Da`}hn+Mwsq&oGL{#DW|im>(r@#tvTC;DA#z)fZ!ZiJ%D`X+HW;z)e$?DN0^fdqwE?@P)=mE^ojizSI?DYs+B2AhwSs}drImop8;=Km<{a_mY}HE;8iw=OnkaW;|3fO=%}8~ru$5ir2uA=7%SJf_f+61q^`eWJ-ud(p+T4|5C1zrBF^yHhQj$_opu%my*C$UoJ_;wDp(&P)=NaXd;I@(0jS0u%<z(+WkQ)MceEN6+!K)(xCrYn*BU=KBO}(7kQMAMO0cJ-@1v^Z+F5RQQm4-?{zJ#k{%ILt`%!}=$Q67+HWK;Cz<s^N<m4q#8g~Y;KKD1HGKCzb+uDJZ+Lu&UGRPXk$Z_b5KV`u0CXDa(W5syY-0B;WBJj2?C2!;VP#kJ)Q~rd`(RDl@nN8MVyKE*o6C%$sFDg^Lv>A2W^v|71i2nDqRLSJ<zqLmP*N%ng%r49T1O`{O3z*{jc6PY4h+QhlCm^NLHOKhku_FSr^(|DNQ<^b(*_vjsY=GPLw3K`_$9Kb>1hX7@f+mcU{Ilz1e;^p-1cuHO5M(s-Tyw`y?)Sk!1;Xz1+^S|yO{ZAY*=JfOJ4AE-c(gm}H%hcVH{fErN<*^pcG-fzmS9GUyOt52K(Q~@05d>X4RvbyC&*5Acs3fjh(r3Ml4(+|jlRh~do?dpf!5PlY+FA6D&yH6hRaoQVN&Cod=sapPU13fo4cPl=Yh|4_>Wlnge8!T0NeE(4O{mg)x{>5?!1y@bd0%;4K-?2548k?8b#8z`YhDAk-}ZHt=alE6b(@qm$Na)AV!zkUp-!cVeGJRqpvO(QZJzjpCj=SSrQTf5~%afJ}9#kZ6B+)Qt%rMVE2$@rrwRhNs)!fSxEHm?o^eacms&y)W~?9qq#Axd-RwjD^f+&;GmL(Hgy~T&$3fxQ2d&baOj!G$$WkKPHdwCAf-Q{;HTalx?<59@nJcsx7RC&h+Mw?qhL>mIE_kxN!rZidyua7z|N^&{oq(ps$$ap%@_CuP1`I3t-kcZf%YOFAg0VvgA91_IR~{A@In53!CSc8cBDuFG$xMBT><r7RxX)i>pj2_K^gJIrZBLS+z|vmm!JX=WVUE<dZ`_|{?4|y?DYGpuN}S0hspOhK&I5Q_4JkGIL<tEwExHu_z!<rwdk{0994YNV4p-sh(?9^G+`HQH{e5|^J&Uf%7J4Z{l*Hmh|5gN6cQ)UE1CCyIdf5eVuJJ!Fn@kkb)RqkUW+tEAHZ?e#d|?<B{Ym>QnF7X@$D4Z@M+^T_lbNcE(thEQv#CD8>Vv&{Z4A6%VjN_)rAC%(1XplQg`9X$Ee!O{`HFyb};L4CRWI26%WeXSiBJ1zBec}vstYMQ=9Dqk6NK#qSsbY@D0^55V<UsSt{7g=qsVt>7LKiJ1vvduE0V#!WosywLZ>gb6@Me*G-lcd_#q&ivP;#IDyKy)B&@gF!|*V5)*ao7TJRrs@L1h%Z&kfm{{b%+qJaqvw3!6%o^hCnuMdm&bHSf=qT!Q58%ytF8_6wS|4um5l^8W4kCXBq9cq@L_Y}+pSzN$2=!|ByZL93_4HTMx~Zsy9oEh%=mLO<368b?F}A$A%z?>nt^h5fzA5!8W0cl%IGAd0=H1NcRqy1)BEPJISWoy(|LboC2&I)5gnQrz>TD#N7U^xHMS$ZMM`mm@O8Q|%63AMEqth_G)X6CqKu%@`a#>sQb-<TDK%*6FuA%0Oe52LTX_C;B-oL@~906=dxUf?5&sTC^`9pzW9<2$PH_r}aB1?2#S3F$g3`f2*J47h&;QhGYR^rpw1%hOvB({B0jb1p==F)r%1R5h&;x_$vF5cSb5c6P0lM6N=1#g@Ga=;peI)e)o3d+a+0*xDl1SUmD|3qD?`FvBff_?ZR(J#~`&Ds=TsHBr{-XptwB{jCmM5R?rl>>ayg+hP#TGUJI@H&hWiTe&65R&6|%4nplj%bgsVJFS=Jby&gg4YCS^Dag}T~f3LWcS$!C!Dens?H-(MQ>Auj{y_72$GyXw+2BFSyqcS{CA+EuJC?QQsP$av5Ju_h-$|Wh#<vN1GGkZ=8vgkx%#XjL)G{ps{4F0@v0Xly+xWM-@TC$Sw_!exnGV6zj#Nc?IdrC0r@4r%7nj_Mt40gDo4)rUv;;mQG~zHklAdT|Nk~IzY8Cl2#9_O$-l4Ew;~Fieb?&7ji!-7>US0(HPox5OXi+#5yTDJJcVKiEm12nZa4oehr~p%_2CaoOs26k*A-~b3reeh9$L2>4LwJjtSp?kT*50o*EYbSx7K5Wn|gnCpTrDegt*Fs6;&ezh3}NYmPK~vq-GvO`{EL06r}<UR<H3^Ab)z8E~lc>qJIkh#^Yi$;_0!dptY2oEj9gYuA06>3J#5uiF>^Jml^z!H7(P^@Uo9of?8Qj@k9>8kR!uL$Iw<bN+G;zLpNt24$XKsI0}VX)((iZ`+wQHLX%92DH;n-cohBZ-C|gJIC<TMojby}6J8v>7*Vc1J;3kOm0`FAiXF-hC)rzfq*)+b2mlrOPvy6rvk8=)>n!&Q_|%g-XK`-~x(>ENvb}JqX#inmtE-=wqY=@pi+5jWw-OQH85=_y36tb}J=k&$2*BAy{QidfQZPT#{|E9&n1PO`R%-9ke%(wqw;@%JrPFkPC63_s)C<WGChm1R*rB6e;6*eX*m-hQpj*A6O(%VSXV$4tY9P<oHr~(oWx(hG8sY$r)4wwO3?z4@0OZ<v)4oI9OIZSwPa&>KRO33ts7U2;$u@}*!6S<wPm*X8npW=3h=g=imOjU3Gyat-1>%FXLA@WWiA^V%b^429k8<7yPsE+o2x!yq36YWBC4Dk4-LbvFi^&a1e#rZ)?kB*rno`n%BM)rC(5iGp{2iB-jXgww=j%l8fO2TJzC8+B-@Q>+1(LNwkLO*~8aAnB<gXKSbJvEWqD&cl{J&+##(2*VIm5;>yEApD@KM=aXtd+cXXFu{pPpz~ifY*LF8%!os%*`-M(~8(hn3X6tTj(JHMAyaLP!}gtNR8;0xff$f@N=>m8tJbE(`-Hq5^su3}@6FXhTKZHtr`#NRihVk6r2z6(3xb@R>VJZ$>3T@$>eNH0isA{!ez&(p7SGXh(i2h=jr%P)cRR`L)FC5)>jq*N$79|Mb!x(dUNUVsi3<cgZ>=Ne9U$Nl92;=jJ62U|@k$DSfu+#C{9wsD@Cuu4P!LZFDKxoQ8@dTO`yQ9q~7|z^#{lXTby>wFPzi*}VKY5pRy%x#QTqg|w}Z%jX@CgOU^H-t$DGI14wp6a01frJGd!w!*7xq+xQkOfV}HHvows6P4a{fEe4QMc@@h@J78D9s;r!S7(ddFLkEL)B}+r5CpCu5eXhMjoZw(JcOV<;RFic9dPoC4a{M8(O;b#&gX)@*&ZkYYhAlCbH-mqF=2}^$=E-5mJZ1?h846LWv58Hy4y)rj6UOF@gsY6c2zt$d1cMX59x{aaaD)jD}}v)zSl$Cvf}e+Oqe^~wT*FGLlk2870(0f*^^&!GL`j?59HbtN@!0NpV@<gSt(`)Lm6djHEtC*ZhnbA{;jxBSiTjUgH81BBsym`Z(SU+zP5aOw;_j`?t5{yd*3NIUcK?Fjk+)?N5>7BKD)@1C}t;GCH0;!uIKUJ?R^5)0RuT<woqPG9_=B1A@3&r!DLF~*=UFK(CXg&;opX#^&|k$@ZuZ+d@fsWQ;f4SvGa6MpKQQx*VR;sug5gX8F8Ds{8em>m4I1hcJcHl>yw9E;=U35YAdra>O=A8%dW(1loKKVpz?0cQBqok{gf~6DrR}OwDo`iD*E}6lPWkQZDIiJ@1)+9J)f}LnpVvb70~2F-2L01w4i?IR%p5!26>EsnYWQ~IJ&LzFk+vtO>|b~#q1G8I))Rq0))S(&k6W>3%a96Ia=XrBPfK#e`4DI_G2HD^)5$k;DGY%C2=G?)FBQbr2XaV!pAmknjzsrF6QBl%%-{WwSsb_(T9VK<hM81o?gGFQKFTR+O<ToK@&HNeSiqGX$<~QS))XMDt0d`nW4M>;q5RF+d#Ngbsd&(elEU6aj%WwNBFSsTPT3Vpi@7$a?vm)@#?gdnIv*!E?}c=|G~|We@x5+sYv$vP4nux821YvAO~g<>achy9im{mq+B4#Mqy-<$ieH^N*_H4-|*Gcs2A?ogzwBovExwHgBDV=Q0AbCpjK?Ste(&A<<`CP!QHRKXHb3(8<PfEKL^P27!pNi2kt)@A7qtxa0$=u3`?p?l?#Z*(WXg<RAQrh<(SabSJj0|BxWU2{+8brB}zW$ZGX_eF?tCWw~8Tw!sV91R;tZy`YIZChQaxu#nJs-*={57h^%wr?$dl34U5I>eb19%NpwpA4iB*4IDe3lko!jf-$-yoF&Axj9q)hR<(ibnf1O#%M|?XLhkt>c1#D^%^Oc{rBHa;(yxo5<CBTEdU;GayrJ3cG`3gjkj){|A8Km=HQfc8Qn;n3<-`8G3J;SK;8t);WAK<9H*uFV-Eby7T1#0Dwavo1bLqahPoiWK{zV6gH8n|5JofQZdO6)Tc0%$|HxF1DG1;9}q{tX(^ZQ|p{(+1qi_>#RT9K@SRkY(S0yUq+(+|xC4Nwiw@*<6jf=<)t;0RYD?phsykr+JV;2U}>3naCQ-RV_DaRt)!L;)8%*81B6WWZBm#?U5rJta>Vj9{wqAp3*^$t?!X@L(G6f$AFOQX<zPPZqn$_s^3opcHtR(r=L8^NC3jTVd^r4`%+GmE~sT?{yG)Qy1ob>hNtlZE<%oz;_asH#S*w}jSdq3FXHG=*z!C9{3~1o;zKzY^{PJpoQR<NFuHm*9@z)@$2bMK3Mt7{I?*|;JYD<zhxX9b{Mwn}E$>1&-CU4gI@AvjZK`j>q%#eAEdzJwHs(J|`wk+s9g_#a@WXJe&$go3jIfT{dinC)TzwqFP|y)>#6L+%OAt!uIO4!4ytm-5oL3VJ+LaM{oFPR(w>2;I7HLuy!>(g*JS&jhv%!RrAhM+DrVoIx6`3okS1JQy!wEY<=56Wz*FtKcRFcSDgTOSN9rB$80WvTvgzK_`?%lL8e9eedZaC4Kstnk&m!vUCZ9|WChW1B{sYUq=Pb(UxA4gG^gN@Ywd+Z@&Ut=oKMS*Y0WKVvYX$CVhbo@BOjz?3TtkP1;zS`*Tta1AOm3g02Pp1;w=XN16-15&NB*T}XK|hR3xx|hh_HC|L{3MVPT$8SGqNS2dpLfBMSu@<{?Jau{SnUxML_!AWZyt-T0T2NmIp`8!mt)+)-?;ESc7>K9JXsJa;_q!eW%8Yb_3R|c&Jrq)pi<=u61e|t0YFmcCUMM0Bwy)fe{LyG2FOXw4d5<z(n<|~Q8O3{f7w(o>pS{Us_l+`TPHaDcZ9*Nf*gi~HQpjHH;xTSEw8YQL}?nf*R*KZYPO!9Ed(pA^@`$yFKD(TW7g?|mDQ&u;0OH=3@_=-_;|wY*PBS!)X99S|F`I>M<gog)_P45fmmSg3WDQB-{FQ~RbS~}+C|HtUOnTn^bclf@yaM1=ytION9sOnM32{5|IHPw8Ir*M%2C>ykER>ek4@7`UQkJtEmC^a23Eb}4Mqk=@xk)v57U=!&Zp+Ao$J^mn$&|@SqFzMpH_w-Lug79OR?GnyV0N5g2@|TBM?!!1?du5xnH;zt(b;<L){g{YR7u)gALw}p9&zY0y`)QHfXGfW^mE;nL}BxSH_U2&M3v&c%KS!SQZQ?g|-?@!JQ7aCqhM{`#@RtcSxH#S*x_`!ns9A$HixTSXffTHV-Lr{ZEi(y!&-Goiec`%z#px+R@whFLU2A0*-vA^8j!IXD;pb7F9Qe7jzzMuxf?7pfHGZZ(pv^Jwai4npy3Gon7%autT^k4q<ya-p}$SSNq?%i0T*;_#l7YFlT#Kn!z#r;2h(N!#Bl(g+PlM7itff;5X3QE!sRY=3<iP^WV*wl;Z%F;9+sNeBQj%tjB^kwUzctpXu3^J50g=C6rVGtN};_7~F9yZJ)8i{b-;Yp{{i>$yiB!&6?l}PfI+tp%vP~{O1zUf%*2?kY2<hg|4?#=iC&!qO)xET2TX96su%8`K`@1vJ1StdWAWP$@kNeU=Lvz`(B(y(=Jc_a-?eZHj&_~TYW0I@|B(@m_2kMkcA7F16AmH(twpn%-!=M{3ccHR$P7k4+lU4QTp{D)l3$}mg!CPbiRVTR>As2ZUV{ZA~Ve&uza<1>aD#DbL%NHn-ba^4x)Q&XcLE&OOuR5@!oGwaWbecighDT@y=M~Zl3QxN{e}r7j5t?t3bh5a9LrjLUe<NigHFF$}z(hHNQYk38dz6?+Agv3RbZ*sI{sojbV2tUih@(h7)BSdh)R54bsnaRuj6)CuPEWe`&Qq-oFYX?Z#S<TJ^$3tiJngys6thYLVZ?fr1L@9_}LGZCs2bv$u*hOdC?(Nj5ukFsj;UTC9@{=Xl;O0o;<>2jjnE@Euzz(47Xg&(6ZZFB&<p-#KvvP}-~9Jkjwk9b8Tb251OH;muii@Xit}r^Q9dIcD3rb1;L->jZs?!;n-NaXQrDx3(|%A>@&FimBY3`VF;SYLZUokS$7oO`ALVw}2ZXszmmL9MX_kfK@r+L6+@JC}SuHeuaxr9VU8}>Z8=u2zR(-Y}8Rwfv3G8=A1!$QZ6>MAov|xiQ=1GnV6=hq+}-sgy+lRV7(yE#3HY^ct(75xOFL%@AeNXI=<6v+iBModnvuV7b{orN)R>Mu?c8pG}?t6VfL;<TRlp7uVo0V(a5ILkfC|XsLq~(q=NYux+6m7oh$~^cT>lgD#?Q;G~x*k*5^;~y}*DZkG}_U-Ols|ym(QYi?9<Sxx{Zi&F0M7P847P(d)IxNxVH9Y=ViB<&?Q(pKtL>8KA|?_Ab?SGd~auUHBn2aO1-cs;m0c=Yeb-xCuMDM^^GpuBJy)+x*}B&=&z4M&}4J_jCY=E*}6%W^lyXlPFswWiZOM(qo(Ef#&_pV1EwD&=xH~S=j;$ci!q<Xo%pVFWa0D8O_BJNy{ER-P3uce(za8G0AdOM<RrIFyQzCc`j*v0`N7=y*W&1VhHw$l5~lzqykVMRQ{LO>TwUX8z2R#5Tm}IqtBf!_E}ns0<~_LIv~^!_Hg++AQjLb$3^47wVU}O4Ipf>H#}&r4ATlFYcG<wNIccXX^T41{#ATq)2bd$060h@r70`k!=MqWUUW7a<)>79EGW0l1E_UP&UXLq$4xkhqy~1ZDfh0_6yLgRI)KLiU?MTr_gl3H#rd1ln3C7IC2ZB{Z28?QwnIja3)Y+TbkV2_D3vdU8y&+*EhPzw&*s6eSGSA=mp?Li>0<7et^xEZLtSR@o2|o2<YfYN0a863#<C~7Ic<R%YE|>=EiY?V0!CogIX&yBXFDeWDMxS9=kOCes!Aj1d$;4N!Hk7WAnyV&`TmjYAa`*PAOV^5z)oG1wqEIOin;#X&G9;R$!y1@PlDPlOA$-`b*@J7RgeE<t=$e5V1GN`CTrypQpY2<PkyONwT#X^k@Mh~(7-oE0BKV>VVVRK05}wH)--qmj(-J#m>$c0(F}=<n`-Svn0!uEct$~HwV8U8D=+O@SKY{RVPr@iA6;FMK28_v5lik*A(15R*3AJjpTld41Sk=^1sK(RJ*!M}ygl%G@zGPSXe6cDYRNz)MC1oHIk!B98rinFzwVIYud9&Sumfjio@Qtr!^5Y~U}zg8y}lcZ6dhbod#}C-DzQ{%If0vZhN%yhAcgh8+nR(#gXYhzg66<3sbskS$+4O{$ee0Cr3U$tW@gfYJ-C)a#a0-8%OElj5mK-Kovm*df&J$Wm#i<>)vt6~_DS(!LJeUJPQzn&#`+WF0Hx}0De3!!@L*C`T@+zMvBFvL<2zXy#aLDV{0vwgbqF<gJ`&69mh(fV>H!TqvPBdNi|m;0ifz<pAI`N+xAOy)_le<{<1=-h1glA25b#(Wj$3n{b4T6H;4aqI{_oJ=g&4~}u}9v6_K5`*;dz>n9qDH``G(BlFUa3Mm3~b=2>D!TNr5cnBe$swE(^=JD9FVt*G-&^VV%x(=f~#F7*n!@mih!3NLVXZYQkRUkbC$$*6Q4ni)wOX?b;nm;6+nmKCKE`ruP8X3#DNw=>q%M;MT@QhfC55`k_^MA(ma6r$1X3StnuHbR_zpg=k%*z2V6|cx;A{hI;}{UsGLPG2k+3i0=doyflV5nyRte==(cc(!(1@-(;nx(ak#b?tcCCtGY@oplHT>25;{TVX}}=aZj`7BO8?#wWH~TjP#<!M@E1)6o_~e)P0qG&gUjPIv7*1UZ?NY@VBRJXaDfh*yHHk3NUa^{#kGEcjKO#WZri@qdiLR!FmVDDCTAXLe7tAShos*8n!ig!@^&BT-AKwjQQ0hnZkLx((pxg^f3yEV|Jy9s(7=jNgpjHf@G|;)w(9w{dwLvxX^}!u?VZbo%d9A(izkbHMO(9=-cDhb^IeZalznq=s>w$kBLSNZ$iDKz+taGuuSP<yFL6@wP{aRZblhI{8oO2Vs^k9XQw~B{nmzC+Y1^U{r$iK1J9}uwT1is*^6<v^a!8KnHJ?8SP%5)8ziMFZiJDpdua0LCs=GtUae^cXb+l8vY|a(8)*~{tt^DQqi2hwky`{Xy9;M>$KcwKc9amF$eRKLY~hs9aH2c3BtNcaOdCtNCVB7^hl~>g_5dk=ehxssOYAbeYgyGb>iafFN%kw+#LMo1ZT5B=Jf7x34Wv>cOhy|$D~EJQ!FSBcS`Z}aWp_5)fJ9*Xm?576Zw{BI6wFXXrO~@6dO<$OD!E1$!wJ(}s4m((Pce-bbOGsXw31vC06GxR!)Cn2v~9;YhLYXtW1P1L(06vK8vs0FhAL@Xl4Iuoh3t{rUS8+S3xPq_!w3fjeH`4qRak*|8f0-b)Z5}d*H@H*$A4!u)>nv+U5NOOj#WMCYjZrn-JQ<n&t#)?PqfglB4|*-yjV#;lt&@6JZ-R!n6&omikV&MVE~303t)#7f;MBwQx)1Ka?F=r(3E0Keh3Pcb{dUoaPex6X4cCyUR5-VUAx9xy3Xz^h!iI|NDSdyZw#}%jHlxl6uOZ~_-*W^_He>H)0*}jWUW=SvrG_8mk;)u(2(tIg3g|tYVGddp;5{I_m`Ncwe;7lP*pV7V=FKFm3ek^yBDxyeTI(R>fXp~ojw}XC6h@6_SU)=)DyoyhNiCQ;>Hw)w|I$%Ab~9d8CL~bkbPbp7byt+WT|9=A6Y3C6}Htu=<zl7UByVIvc{kJSP1+|SoHzqBtkoymq1Ms?3J6-RbH+3L337xwAGwCy>xb3Q5!wx0D7aN8ik=>nUDUaGTy9@e!&zV$_zqrUf}oBpH!9*B}804)8@<smKHzWp0mq*1)0?WeoD`5+*m=hztsSRfM_1duWGr>7Z|x!x@Gx$sIjR6d->n8^tE%K6%cE8v>Swx8%W%<ArF^M_kHZ1Mz+*1@<V;fP?iu%<bHft4UT}BD3k-rT3LRS#q)RlDo5xIkAmTd{Hj$JNfsf)TvlFJO?NRugQstI4VHK66Cy?hkwdSaDgsmZtT)r}2hsScjJT=J!rEAk;KOxT6{&#i7vM-UOF$RrUubU(62f5CFS$##f&qN!b6hc~h-lkp&hEIFyrQ-%P~AKEp=053(~2$xqN6l&?zofNV^=LcimdTf%YJfGe+XXZK1zQ1)qPlqe)SjK;fb#+|5T@_UV*vz>_Mj&6ij>wtm{Q&Zjf$ZBv&Y27I47d<Z<ggy)1o!7&$%*V2D13d*!`gyJyQldnp{rMfO~QnI~6251tg4s&k4>zJfEj#%DZM4GMN(&T34>dUP=`dd6WMCLiB<@KNTjXU>or0!cm6pR*R8naD~c>2Sh5I=%p3AY{xJ$0E+Gv6FLN5!fDUC{Z<yaVnm}1H(Iq5M-HBTX|vNzYM5~04px)wq>(g61x#cMoczq6>%SIL8<u&z_?}I?!tD<)fyTFt-W&o^qqT0ivd|*xD}();CyksAkILY%GXNArO}v$o1r2lkz4n>#(Ve#a$}ElDj36BU+dz!#-vii>bV2a3=4^wRqlcmH*N{VHpI7yA#=$02*Fcoj*2|QbRLW5$>ml7jK?wf+`CvAkDRjC|8^uP*?@1J^QNE~X=W_{wW_^|g^tqGRy_x9G<DDoLz*b-D&5&#V4FT=g|dzVE+Jw|1bqJR&BFEHywWvUIEo&K`rny#Dw*!r{TgMemB!J-slQmF%49bjV<)muQg^^@e>)9kE@*Edk)N8fY=;2nLyS&RW|Lxr8@dHM(lmY3A;tMvq~wUb6MjrRt8e4|!0OpxrUp~%#CFxR!GWvy!=~SdUr!4_&p>Zoy#6%QJ~?5*Ek0PLa=pX3EHUb&?VD6t4xazFBQRpE@PtAQXqfJ3J!2b5Tj+=ocxshj_lKwdDRnhTDw)TY&BJ5YF~vhVSTLjnP_d||KIO0UnW1ZdLd=W5<semcT7qUCAHDNkNLXF8uv<+?Cm@Nm0KG`wXv*Qz3A)U;w;vVeW{F~;qJ}IIouk|WfB~uh+h?T({Wh$3?R$_kJq2^-Di3Z};qT)4=3_t>nV})la__bF(legIO0*gLF|%cAJ@m4VfHuT3?SA$KEV%aB@l42mlv~Q~eCHu53!MFJ3YRB!zg~GVRxg?H4~dQ=mB~tEk1z9O4>n1HDtutPHQM2G8g*!X8^ETf!&&E;nX*Xu7LkP-hLMl+KrGV8t<U0R9~2LdN%V%7hjmlma`X#(V{`m@f%!f3faJpxQ4^pT;t<X|K+EVCdE^c_a)&SUPhg6HQui<N*0sssSRWRxT8_xl>bADmO-R8}?rav0c!65^VEjPY|9*EG$G1X>+naCFJGVv=TeN{4fFcrK2iYJAhFKo%Ow}sdE5@9?g|0lGv<;Z$*I#-J>?uuGu>+N>;lO*2z_t}I9N`&ZMJ%5I8(j!<cJLn0G&FRRA9l=d+U49nheo!CfvsrEYx`BtsjO6Ad3gO#6AWZlJ`}ADaqhMSBu}kng>N7~v1uX>ARv4aOfJ?!26pf+89=f9Foty1d5?7Ej@`R%j*b8aXSvAq4Cde%gug|@$UB5kNxRA2ijG&+%sfmN7YEwofh=M777mmfjM}b(JC2{4W5~bm_i3l?<g78MEG;JYdxUL8wecUOfd)PxYDR)#;L)cBB1{Qw?Yww&6+0*4Cg6Be#C$a=vWxSl2pi`(Hkb1-Zc;km9Y<7sHD(fCUMmYn|M<C$ah^KJwK8s?m#H^;E>vn!<^OGUeKJ{~?jVB@f9=;O_8f+#mnC<Q#ER2&R?Zl42J^F$Y%PEl;HxGCKM}<`rQ{%zBKy5<juTC>a@ne8tQKMM;=+y}{Z2phsJcg3=e>}IJv_Z9pKV@r(uhoEJ2|-gP!1HDr}7reeG5dnTIiJaOUx0&M;~>~W&C@+OdRu&Pz(L_O%_YLcW78%^_fcbOrr&sQE8j@x|)vuhfDc_0VwrebDGyyhuK@k4Gr5P8rBQUDb1ChA%(!I`62{HrXDW;^akCyFS1lopBjcEBos$<n4CLaQc=|SyO33_80J9L!1z4dsV9;PIfb^*<Hpw)Dr*JrgJ!0@gzZ*N@nr3>3>tsgt3&DgGyH$=tWEoSh!?My=?Xcrp`BQEbTA!jC;Bu%pi$o4?1>GqjZ7u%rl6i4_9hWq){&dq-padbJjRs%Sb&t?Ri?Vwl?TTZwK2(Jz`2KVL(QBXEPDJsu{1Tu$hyJ@J%G)DqT^pc5fR-%CtZ{IET$wjhstmPWe1W8Ciu_d^)A~A{kTC%dqzVSaUjV<FGk@V$e@{QwNMKZ|B6#6M|skaXdWBOfxnH8-kz<~IshOoX%3N3;4Ngk44fm>C+!~NL@&l58J+^vnGG>aL|O~cH~>g^JXP=q8(k{d(3fqoZUTwl+P5#|HR<-hYkK&AjEA*$5K?MJcntsYj0U}^_!s#^>c16cTiH+)l<pbPq9>hA&D-?I>o9B{a^)kWQ~Q!13pF0BO3h&bT!0I{F=+lAHjKK!Wm9H_sfTupVHaa4%;8h7s74cjid?I_fc9}W>6n)y*)8Y2b&vgs*Ef0V0%6HwC3!Qsj*!85XuZ9i|4p9q=20Ov8Hx=EtNrV_?6^$#pJeje`^P--CA8GOlw;2q0BlFUhcsDst=iv)V~=sDV7sDl<A<B=E;BHX3EM(h*s*u}yOS3{Bp;2`(IyV;NAXLmjS4tWo+qj8XUK!3n4O$p6+8!7o+@ZD?M_8vNFH02<XvFGw&dhJUZrOfH^|}Sme`AWH5(Hfjeuh-oJq_w?5`jre@ITuFGwecg_W-bOE2x!C=u~^ODz{{n~uF_=$F36!W=}JEs6QftB|?M-*0B=4-eU5a@}GCATTZ4C8c&{4dln2PeGc2)Hh&I9u@<(@YS?ix5IMG?Yj05eetRVTqLia3!rI8m?F~3^JJ13^66fxKy~0NN84ai4Wx^sR?k$wy}e)kCmPV=Bc;HX?!g0;o-XD$w{pIBM6=E;jTJrS+OysX3hZwYnK9bF48OUgyH>X9Y-Z*_ln)O$DC_WLTUdya5D{nI!0{2dd@KcvnLYKIX(2nKGP9eZ+Ac$@`mUryI<%W4{^9gw0VSXPi+A-S$`1_U6a!!roT6aEHRxGLpCl7!t(*Ma?BY}koiRw@SMj&9L8m(9XYyDP-rc6bDfrk=z!3}C!CS2fpEM07hJCg->~wN6TD2BmZYr1?!E{Wf#0DxGPuv01xj`eGn9Xn?D(m#mC=k6hjVknSq~jhFJx;XsY)t)L$5@|;ggzUJi(<*ImA#|ixquyMzfx<@Mk4>%JQ+vMw{Y%lUN``QkmdhQFSJSM)=H+oGuWZ-qNlTRW#P+{g==ZD$`9s9;pI_`S&aEmOqCJw!<54HQPKW6m3g*I@9U1*f3S$f?<(A!r|i``I=VjS#*?lD7I*X1f)b&A-mXWyPRVKRWCEEFk$qSQ67NSMOO~jbR*GD62SFv^bitpVM~Bji1ap$LX6@?EL<3^?**EQsW(Z&?P2-t>N+pWAARWt7LfKkU&;d*@vZ*$8ljrn{+igKVw`Lp2|292ewTxz9fCLU&H=EH-%6E!FJJBHhkdQ^;TYgURs~8I^D>1K5*^3~n$OpEfs_eTqUv})Lhn7&87-UtrsiAEp^=OPZrKb&2HWSFbB)zyJ1ngZOuH!$4^}b-FUn2%K)*XBfbv$PxH&7^*Y>|kY+S_tWLMv;nFi?*4`e-@l`DP_VO?*ZP=zH|g3pp7Mto=|H)_^dmXZ5>R{G-UV`N7P*%0V798=plpQ$@ql4yRJ)S!~~v?QtTmOK38=Isejg8i;H=5N-hS#!ZryIXDXYwb1HK(JX9qAf{#Z*cVJsyv>LdN$;MBd}x%kDSYL-{_r=u1n0je%^F?#eFOjLKrB7@ZLQlY>k~%sfEQZiJ?L)7Pqq&9hh4sXc=sb+-R3niMn!o$%<@>ASRAMdrP)q7G5(RBtIEc5`_=1&T&GGspi-}s7ld;da@?P`rXxs^V{T<wD<n_|tX55!+lL6(0`4C@N^F}$Hez)my^{+~6lB|V>^%(ySF;fayA!5+c4haDZ4v+EB-8wsiV?P9Q>*G(YgOuyd~6q(t{%4^X%TUTYlbj;E*V39O~EQaRfXL0)bnW4&ViSQOHrIero-25P-B$&`vr_HO2G96j$C!p2fhUESHvmcz|*Q>PwFhXMp;AqSxE96?7KvyZuOt-hF&Av=@yOV6sWapGDXc_8qW*!IsT`HZ2S`CA(!w2`tpU}CSTS^1d%761!Rx+QyZ$x(AjmA{wfbEub)@Fm2s^~nL^3avYNDM-Avc|;(u#IrtGn*E9hqc-#qy+kY-zj6#fIW&!^}i0b$Q9T8~S+u2VphV4r?^TmsH5>J7A^sKx&f^bz&MB$T)iE{U(}>-`(v_DDgdVQX89fxjuGEJm!mFf;c_i`k8;lD_Y=Vk%AoSaTC+4`P5_g#^bZZ)0!yDDigte}^y`eq?{hHaUlj3||DH17+r~fTu~k(Jq0<UTbOEF;k3DBPfr%FWB6E3Ehg$Dr@IwHDiY8Gm;V%QDmrreAdFd=Z~KWC2V|QcMt02E8H(C(wy{*p~%|?oxqV373LBAG8CtjP0hcG7t>f-BPdJ|I4AsUBB7e~^WS6EomgC*^JIz<{mJ*vim8A9H|V30{EXNlm!`5kp<^3|FtjfIh1tl5&8<+TxV6fw>?S~}ZrZ=Xq)z?W@cqsj=5&;&v57j!m5i>w$a?)@bV*2iD7!rpBfTelh2<1QhcF_?6K;e=re-jE+MUj}Pi5|Lehm&o)GSGA%PM3B9iv3g!FVccBJBL6RQ9-rswE|~ZecDcENpw3NGMPHEb`h9gX>7X*^luR;Bt)%AnhT-o}A@{%@a9xApaEWMD{Ng^`%^xdS9s!_v1gQtIb~)u7sldVOO}EHSsJh1*>CtC$)b(-{Lbvj-;nRE)g0^&u~4?Y+1vCE`iNLcIp%_F4hjoH>`EuE4mx&9AhjHI`)?o1TlZmJ0t`No&91rWTgQ44RaZbt>3-H8!IY>KvFC{Ig$Ui_xBZ?y->KnVaqG$LyuAI_&W5!9q`r=(X?9QNU|l$C+Q|cgc5GM%9oH-d0r$80{sz8H?UjKD05nw{P;U!V`(QEiCh?ZfT=G*W~+o~?1&HoU{G4f)#mKu?qNmYolbuic@*$4ZRX|5vbiy_hucOcJ<Bw8Th;v!6bfEfo!2R#>w6fVV?`B|{ArMpdFPnvkb{b0UnFXo1L7}0C{=M3qcx?mL&X(9$DMoz5vPwUsSk;)y|7VNk|gNBj4K>GGr*mAmso8ymAz>A-FOYZmWN0tBLYG_1sGJkv=8e+^VQ8Hx>mLGP;vUS{>S}os*3pXb?5VNH#lPg0hf~JxC9g=QYL6n0L#;(GMmI;9ri<O=ey~X6piyuqcDW6*a-;^`r{4-<K_44`xM=J#l|JdNH&NV;NVc3nAdn!OoCBPOUmTBedU6i?@#zf`xca>vlm_I)@|T~shMD_o^aIL;X$YsO#o-`waQe&+Qp^Ey}mB-I8U<X$YtD=2cA7}Bm&}x-C4{vHCKz(VRATO+4C$OQs~QbhLq9#I?~Tldk5E$6+deoN2Uw?8SkPf$bNaO&1QD9O0<@n237UKYRp0j7D6zgokstwSORTlSS)rFu>sM6MmxW9u@PzYhL1&Ka?c8O0?Wd_fsLnq5=K1b!KgTn>7E{=QI<9nI#-B^U~v#2rf}m9>|%aY7p%V)!X;=+0Wh)`>`wmH&F^uy0>n{ESOvG7WC?c>v}_Go4O`)QJmWEPw;6+wB7l|=R$*iqWrAXMr*lAc`aX%qZ@?Y8XJo`i{`VV5RZdovgKePkJ$+&mGMxac9meZmY2Ci_C}h1-R?6rcYMffcVa&QoLLsGp2qh;XV*J}d@!IE5M`J*>`KZ(I&Tqgs;HivaX=qfD=cJEj5VzRYI-OnM;h9T-z=nx>XlGwgajStEXR?mdI<rX_f%l^{Tr2NYv;mt4h#1Or3oH>Jct9JoPOraOJZ1Qz12m{48vA{-heizQe&q>lnfg`Oq5Z($vJOZ!e&g|XdKnZ4?n*ub`U-n-jbIjrV_KJWekp1b6=V|qu<bk*U1xim21>uXe)DZ0;Jt9^ZUb075I--KbBDVKFB*gD*6tdtJ-aH%*desMKY@%owS6_?#k&LcYOsX{<&ZnNAueMgV#9s68G;3Lv)md<o6rE}X6rm|6iX5N_gd+cS?+Mq16-uT^!Y7SDY)sOu{%8kU1Q^YTMy3apIyT9%KHax)}RGxDk=5Ce6Sa+yioA^<Fw{)Njhnxm%`J+zcj+aF6~$ybaOyS{0~VK&iUQBNdDS}s8Z&raP=)os@C+^+@J&6Ktp9d95o{C#F==zB&}zh4IPuy*X+K)VSbNEB^6Cj?<9;zaG^go;&iJANr`n{B$?1)(wT-<iSbfnx`KC|wx@%1oC2np&HCVmhjE}cFVt?x>@~OK-5D7z{FEsmXSm6{txyBcGgtf87QGL1`j#A|xr2q-I^*h!lcXu~jT5u^4myz7RVTxKP%U8fKaW`SpH$94skO~z)YoTcBzYLZ2m#?Ly!7ZT1zgo(zYIAAChl-=T70<ku3W5o-2;~ooh-W4;*FTnva7(nJ%V`6WQ)6SF7Y<*B3n019ENnWOu2w$Ol(M=Sn6Hbs<qxhEB79_Y+_8z{qJ<&fOY?C%yOyqGU<cs)ym^zJJt^-DwX3a@&;qsWqh@i$Ex%-H5PfxQ3nHe?p1SpE%{SYMs{8h$iH%*t`RzQ;JKo@XPAEt@2m>Xyu1>v6+zQo$ii*Q@S`he0UG&B!qs(2wqSINEbO((w8crP!Gzb6WeLByEz;u~QAvy;h>!9??=eWCeY1S~$_1H*V0#N^>`@$QV?y+D3EecZt}hhz(IS#u1fIc?f}`6yK)DhvtSc$@@c(|}e&0b=7Fj>_lG~n`=(!f4ybWyq@HeOp8U%Dy9-bIt&NbYhitj+gF9V3v*i>$e=PgK?mM(#zN&!8_F7~5Eq=2DcOB31QSP?{uo}JcNIEB*{Bca}i(aLiB(a4ajPnzKZ(R-!e^6nNlBkZ11dS&aFwunO)H%!z*82(X`Hr792^dzzF(M2zP<tFXVpy3kXU8vN4EYr1Cb8D&*LfvU7RqZ*%NsRfu9le~q7AUZ(h%ONi&jdIKMq?m=^HXw9qP%@uEm5-IJMUb?+>)_)v9<EQC>*=FYOM(d)RtJo<P9&<x0a5lzjsxjKN2t2euX-N##c<?oBj1{?u<GsN&oTBdC)5rC9~XrvtiE%KX!K;nEZnI0=8X_Vd9m-83+~V9z>XW1{;Rd**V8<Hwdxpu_E{r%bs;HiCbh)XcevR)Tg@uUJeB12|Tm>l-y+fqc{$M0gtM?e2f8;(qqhV=-f{UH#SqRUtnM@32L}TP&^n{ec2A%R@$?PsPr?`)#||(HV&AV>7hFe)yyOQwpcPDYaoJYqWG0`tK|9KkAl&ugQ0+D(O3ejHLcdeTiy>QQ|IrGmHXRqLw3nn^pHM;gzKODQ}2Gcj1Y1KS+qHt^hrg|p0#gWL42k*P|~SDii`YC*A!Vq{dIxQUIXSRF)=T0cOG+Ut@53T;0<%dPVxuPh>oRZA#tku>}+=+2$u9jP4OyMPx#MAwDV8k1D6{O4ov;*OVYY^yWzj!n@&s`%3Agwz`HettZEDoGPFx!`foVMP_HLgdt;xh_7WjB+a+X<27Ey}`Vx@WCW3&Z;CqF^=DzW!Tgp|X0go<(wm=m{5aLL!9h(`ojfn+<Aq>_K-jmB~3O<0}3;#5JHg;W$4aFJ^8p3rW*7Z#1g9Dom0|wwN-%vsHhEF9psPY}`%s8Qb#m=Fy9Rai#3%Z9)5Ky|5Ploa9&RQRduIGJCTbV&`vtq;GCpmgwr>?<TyB#N6deVkQ;n#ccXW7+^hgAY4O{U`4zvY%4b_2@6YBy^zQK5_kk7>aQe{ka#UgBKr*W>?s1Ss>qARC_4L!gEA+b566VnG|FbbqzO&QF`rNuRKS=6XO*QvNa)lyC7x<qV}7efVUt67uoHlRJd*8g6WS<35k9yh)Vy+oX%nRrQcdU<)W~MP7oW+x*4f^&xS6oh(%gP3H+DuW)_&Nt1IIa2zOK3%3hhdt*^8Y<xd2SryQ!3~y_s#xo)3Z1|3*=lLzaFlEtAQID(<YA8R9@3ZS|t+=ob&(H2*$&7>7UJ0$>>7lL17{IX>>>-TH**)hoQcS{wgg*|oK+00-<0ln#?}X|R1IWX);{>Y99#cjV_t6r>QsN?pJDhJ;Aj!2m=QO6xsR+z*6<krZlIj_)^wMR;_FBihB8~5pJUQ*k7_bZ#;%=GZJ`V4vpk6oR=C_7}lpvm?s<FKHI8IQ`H_JYzc)y^Tc0ECS@}meYgJ-4N6&?#ATqmlBI_a_EH9C3rhRxZ3%+axGN(rqvJs=Hxnl4~@sEiUv6_E8!dj`?`9AVdaQ%v26+a<_5EYmJ{O}TYK-XOxh%YH+*b{JrO$s~ajqR@|OhnACtAM>W42dh~2d@ucy21BL~?a_4)KK&dW-k~<ysu`q_DH5&U4m5TidC4$_@jzN`KXQgbL=l=+^dKNtze#IRKFL)-maFI>&d+sOq(v0p#e1kdOLDP>Qp0NI;J+HXLaGnj&Ju*9V<eNc+DRBa8>t|S1Yx>`XI)`CYg1TU1-e>nR}%Y@Zvi2t9)^XtaXgdY;L%ogYwWJ{qBDOEx!XC4toqfRV>wx5ejHR?mH`RiTwX$mMk?)o9Mq@cK)ubf+;k~#9}OM)!~BsfllzveifNh-OluxdAq80^A-JZ_rU`kx+9$oTl%@A4Parhx4ysP($awM&>g<?y>B>>!eD@;kCH5hb&z0zqZk637Wg1EON#jTI+;RidNp)7j%`)uwm0C^65+?it^qhdxo*KaK0b)~-({nZ=!-TN8;Y_aALL2Tcf6k5#M+HK9dFN#3l!*Z^CmoYPDZU97TIb2Z-6xixrNQl~%vsP8Vyy8w^}-yf6Xr*y;zhJ%ols;>CS$^9c<%Jut=A>46F>ukRLjcWmOk7U=F>F+F3Biq)0t7WHddV`II^Vi??>KN-plGA%Rx6W70JAW%i(EUvy$TSP>-6uDbGJ@>WtW6>oWJKlCo&=#0U__P|xERK{UWicEZuk3YXgGE#`*AQ#Sh&Qx9lxbL6SrS%%Z$&AN1%+(u~W(v^HXLyc?t6Q|{*EPfKx@;tSkJ0*mLmkNJt&BYh7;`d*l&Uh~!k{cEtmP*E8aM3)b1P}FP(4J@D1=Txn$z-LbTWktl6PP5)ADUU38F6PAn2R8&m=P0Ac7X*hc=zkl^(#m(yx^M3?}y&=ArlHuN~n720`r>(>+;RDN?&x1r-WeJ>|E&u&jM-Y>7G&>4EiJMMU~!`rzV9So#^F?equVc*P(49Yvx0yo|t|Ao{DTESe*^&l~Pv})Wgx_iWR-P#N9<?j9*KGefD0XGj@#rhv=2GM97mh!4(AVuZRT#9H|-1N+Q9u9czyqutTFX@voD1mp=%V!^jtYzFCYf+4wWm9jj*nPg}ZFY|ii*wtC#lGG`F8(HhoqxkVi82WD8Xx;;2UML6?4xuJHmW|&CG4FBFbgZA)8`H!I6Z{ElAtZ4LnqiSeqepB_Gxf!iU2&13N`8v?5bMRwnrPZnHl_MX>nW0`n>mkI~b1cdgJFOG*k+I9ztMa3Q(v`xT4V@+qO^FK3Hg9WpqkrJCkE)W$dxZ}1Wk8*$n*aajjoI-?u^qUO*j11pu(uQpu*sk9P_4)@ck7U#Uvtt)MK$+yC)Qg!l3~Q3cxWs>)bs7cB!--jhhgvOIv)u1_E_KqEi^^?lj{$H>IH)i<J-!wHP}|WP&y|DovujtY;}mMN>UVs@%K?X{D<C}L+L6@Ft;n6Zs7fg)jYGtD96Jf!D;{+jA{DJFa33h!XFS;qmWg}qRSz%=83ehj#7pfOZ|S_wdB{^I31sI5s3Y@D1&w%awg;c-)tDag=4l(zTZNN+qE5kUgyIKb_AKV>)}98MV^&sSXTWk7~!l-M2p5cih5Croay_mBB6IM;HJJQJ{W&19Yhg_)k9}#?cfV0;w4F^HkY?F_W1D^IktT7>YvPutBo(06q?o{%LEqmvQY^4Vr9&ss7-HBQh)YhjcaGXr28rszfsla1nC~cy?yU5-&RJC^;U@!aXMF4Y)FGJw5|lCFu`HGk(832vjO80D}V0hzkj@5PIRv_WTE6x{_Kv^>_}9`nx5_l<Y1`x@2-U;`@9-19QZEvRAVu-c8eSHgEQ;ozy*LwS^G+MyB{OJJmg(Z1q2JN8A@?uwzeS=Yv81;l5OP?now+<&N?$&q#Qo8GTWGPX{KmxVytUxofqZ#ir>sn--2MthxW3fM{a}VT24FjM&Eiho|g#O7k_B8xw9jJOuQ8^N~L-$Kcb+<T>$J6K4`eDnxX-&KjH5AB5dWE&svo0!+Y8(w>J-b*>LaXAsyDvN%ZVEl{_fA<Yn=#2a~g!sBH-Mb2fr;gfVbpeCW*A{{HPHNe!Y8Y*5_I(gbHBrWb#aykuO#iVqw~=qq7KkStuhKsAzm5_E&Q%Zj77`_s)4>Gyyoo(5%Xmz5S5^W|GK7+rfM3tB#yy6H2Om-7z{`TlL;*7t;gy<c0D=)e_b;l`RKL5|$S`tR=mVnihu)N*5Yep4a+b3n3Qa(oZd+HF96@?ZnE+vKx)Q?O=oRBM4Yac{6+VhaPu>zOe<g#QU$`k)jAP4D97ViRI~X}e}i<EM*F=pySHxgTJu^$l(j<kQs#iih?{*!WYTs>5>ly*tXAzn>j87if9nwmJkCZdqJN+hdgOm$Kx}mQh=mM=5~_<`<EKH7F-hd4?*fr%qlN-f#b)a2r5-5Hs3O(+*fJpLv784Y&S3Q^Es;6q3Yk9Zi1KX!HFe!e^5}k#t!trn}Z|b^stFeD@Hd^|O^gfYZL{9xZhe`xg*}haDXh^9ahzj^wol8trRo{2C1I@ZER%N!JE@wcChKXi#e+Q`zX3K%{i@vM<~Ou)pl-B07ED(rGiIEZ2KrkjK14t%&y4lzF&h!=W)^`odcK3=9a3@;qL-LT5$kyP~NgttT*G&lvOP%C5Wd6^$x22o;%KBG9HKf&Q>!_rwvx;zK6C`oso0pwh$+x^Ga_Ecghm-t>R13{ZRPVX&md8vp2d8G?Ork&yjeo|+^J<K4mOwq+z~7|$e4`&F+D1#wFZv4w0vUinBM9a|vm<>E$GyJ)PmlzGNu8MF5z;unQC`U3rD6Qhli<#|H;Y@}M`5!p(1o)su_*iprLX0I_4d@!{{J@87Zj-e&RRT~?NY}?d=k4_3YP)E%{az+wfQS38&QO+bM4BO!tRSXjgeN}#%4v&Q9?wxdx&UH=kVl@L2A^4(ITqY(FBv63L&}H+^-up_tgm5=c3(bK(bmxp0f7nUALdDUrO8!C6{7230_yL)WiQfqtV*p9bJl!n4^OoA%PC6F7E*;wBcdTwGHmCT5n#^6!x!~V;oE7)yeC$0bRrJ)X?>WmLV^icHJVDtx86s6`{PllOg@{Ow02?j?<8Qle-heYI*nP5r=^MC?E6ey$m=(<`E}RpJqti&jr@n_QiFct2Mu$F=#V>>D+n2rSBRFYWsQW%p<$!m3x!gq$Zx~1I&j1s~u$q8Yi7zWaV3r-c-X;7$+yev>LJ+^Pj>|E9hvl<~O46Sr5cIOKrPwwP{nvhhnZK%Kv*A27s1ooI7l_?Z>rQsptm<f_YJ6}k>+Ia>nftEgZA$H*F(sc{o9DUy+kOPij{Z+q3ivMc6jSHc&lG!KINu6jjvUD6)wVDVfse)VM57z${k*B-Bnn?bbXze%ktNIQT_}wR_nb;#ouK7%8L^eo34}7X37yju)S#52_tT^QGBlVuwj#w|ST*j=KaLzmTwF`t_-he`=i2_^&ZAggA1x<uZ4#<uW%Y89APUiX)Zg4{On|2%D<@FSWo$KhvwIzy?Xd`&3!ziGuGRnT3U6CU#w}>A#4T|IMF=}#&0ZALZR`fT$>Fke@PINpC=f1-)@$smAit2Fvl}@2sgn!j2L)1d%Ib5>ip<cdqaztK(C9W3i}uwGa+Pv0qrqG}YNxSeCidTG{Yw-H%xMI_>*)hbtmtEj5*}@4F1KEo5<9?vaf^A)+FNaG25SKoD8r?U)7~4(oSzniSuI^Mn2Ygu;4B-3XzGz(0!j3~Lro)~i_@xN%kOGCagV6M)q^FTT_}?vc5L=%Ldi6(N)sp>s^#9)0O@jbKgl0RXasmTS~kYc$d5t-RXpf!50SR_ltC7h%?-s#aL!R#6w%<BVFWMEcs07bIsr$HqRQ)z37{Nk;M?h;toddWkiOz_`Wb*hY2mC@!&eA3BS^y@y=G~JsW>~b(`Dd3_Gt#~I%IJE14y?R9QF|uXF3QVm2>?AK60gE?TQ;Xi-%d?^cz9{6kfn~pkFoexXB?o+(#c2hA01m7Klw!;|a1SIykn4OG;qeo;-}P-x4{$4a7faaA~K{iFI7LNnUJ+e%gFJk&g4z3~iuzuSD6OIT!2s3UMn*RJ#9DD)0M&TW1*`MeXcERxt?h-1HWm+7@`O7`idXT4agZY#NoeLla#aYOuU`hc?@*Cg`kA%3by?TK)cjjgW?6HDnhb5jVAao|X|P#tii_W5t5SeOVGe2HeaIMJgMUB0qw88_9C}jRIrp>EJdB_B|CfrIz<~^Abfv*=aMDB_WW|ZTJ-)z#?-o4X7P_@8oO4WJ3i6f{4T={l<KDS@z4%8Vl}3h)(3t$gNbCyNK<HD0y)9pT$E^)b!!q;ZC|SR*h7_q+mq;AZv1N^<mgbf7s+rX}n5@$hWi}g4Bey*}KM9UX`;vWnQz$zg}AT;T_|5JA-P|@Yn9K)s2m|*Q+`U(&52V9!!baGVd_*6l|0H_mPi+Fc;bqfU%&pnoYkUGRZg0=zAw%^>-+X+`Ml&Q>;>A@+)C5Ff4^a-f^o2H<`S{MKtP`PAd9d$KVY5hyCw81{-+35qT-+XqyODsXMlBg^2GK0leh~1XW!|6dp$z!|_+7RYJ~M>~AzbMMMvbM^caTX@<=Dq$qRXnMAt~s7p$wZHUL$rL|`U+bYkZA+x@ch(E{;LMR66Gf56TdqoX%<z^41zE}WmZM=StDh);aKLVLJW653{Sh9E(D?4wAb==Zhjrzb5k-wKFJ$t4y8pO2eXhW%#ro<)tAoY!8qhIj4E?QYn<*;MtFvK<OD7r9a+CS_)k8`yYf8^$AFqjGAxtU%*4*!=MB~Lkjj`5qf&OklI%i}v5({Kv%yt%uCL8^K!9b}mdz|)1CnIubQVfUqRU!}>MvsioeMfa(a)TlLwp_39kDOZRGTTjdl(4+mOT2TIJ!<sfsMbR{{Un*adm<6&s73<bdrI%%4*C<|a4NFQrrldvMug3nYG$vnb2|YZSqL8%v2TD$(6DDF#1R1O<{I^jc41RW6euI;!dxJ-phon<zN(Kz1gF#`PsAiBtVMObUhuhBy&-$BFFV?@#*1Bclj0Zx;Lvf+SGztKrTubC#T7!}SJyMUL)a->oZnIpwn!1)6&F#MXBBCkSsiXn#?4ZmRD|)@A$&x^~9()f{@h*V~TJ3QaA741?)_B1hsUDl@mspU}7-9o`L5=InzvxJuC9NC7QeZ{9g8LUck$O3u3UlQ5i*7GIvksrv{ckmNz8d*#aTUmO!F=?x6q{uRt*Ag~*)<U^b<EX>!#Hg>S40#WudIh`5U_)Ap4md-0!#<PP~rzH%{<lrIlXcbm5zT;t1t~&<mAC+cfTp-`m}~ls*GZ<L{S`rLNg4^Iv6137NOoTINEqk-dc^cq*Ow#R%cNAKL^fo^aGF3HdjKtMK-lI0f#}ecU9D}suF6K_t`Lk7Zv*+q&&fbXp_Q@RvNB^Mf`YKi{n&P(HcP{m}NHdwdht6uHJ8<OoHSH#n2ikl#z!Zc6oX;pN(z~^to!YcrY%YQp2E8X(hq7j>tZ$%lery<>Y}KyWE8Cc=Ed$gH%kT^!<Zm+mfG99nro-`^TvwbtA?z<{n;IGE9~6K<zc&oFPAYbLsp{SOd-O-G-STw`cSGhTD&WjG=KFN7JFU;NKxw(n^|;(M03f+S27^n~47f6z+g^-<lMPEpTKJ63Wx}Q@5FK(dyF*3zro`8LKvRH|a*L)Sc0;Gv}>?*fVEhyXCiU9Yp)CUGi?go!|oiko;kPP7yNmZPD_w?Fvb->2>+-`r&dO5QAjUX5c=6q&qry=T4$|d}_0<!N53f1_77Dtop>9QgEvvy0M&rv^44mA0APma8J{qCirRazYJ}A%wNU9iDjLd`IFZawz(*?^lK8uE473ly7;KG!>L5901J9!OnoGWlTj;5Lv6zRZ+S0gY}C{TVIf}V>MqN&@R+9kWb2)WaTY|miC5Gq=1tqrkN|+p-8O5ML4CyQkUp_$F$zAIqAdFztKeT{y>nPI!9*AtsL^AFeQ&ZVrOXy4ft^i1`oy1HHHLI)E9#Q}$irMv97o8!&;<^~Ncv4_Q^S^!y{Q_K?=qif836Z2NO>!fOwri2yl;<rV4%>meq4!4YMitNz)2!mM-#hl#OB7}NIwBVq7lrGx`rj6Tm%D@M)1Uow>13AzG|k=vY&_W6{douhOInhOtZJ8M&mv&@Gu~zlKJOYg&R*-Wuf~-2Jg=<U#^jxm#%Hz-lPdhkSxFUo=-srP-x=TYAAr^jZY44m@G_`_tn?;J+028-T7>pe>a4Be-cL$@jqtkRoxwSGr~jiL0Udi6EpyBoyJ<?-{mN%DXKa_vw+fcWk!@{uDt08pvKv*7&t$rMtF!CMtm^^7;K@Lr#gUiYqm83h1%P}|KHCG)Q^R2k~zeSxcJE(dvXL_Ub?ZJ*f*;~X@M}}%3G>iw|3;C=Wqy~mXZks8OnPpGVVDA?R6PRt8{2#z}5y-s5_*+q0n369!vpUh4t68=-mJa`2gYS?C95dO@HNVb;5V~{IU!Qfv+@lInUYfhtq!d{w8!NU(%Ggu>o0Z8w|q*v<e+W1cQI~%IeQWg0)MX0uUEv6sq>{Nzp!Y7L`IrSN8&DDO9RSygzz-7Q|1l<#%|=FX1vRgQeiIF~^L8zQ9u&tI1Yhn2`8<Cd3Bbq7`AvPRi{wSAx*AL@e#wK13mW(^O2aNQV<|i52en+B!Z!=9sll&ee6fCn(A$u%rb8>!zA~x{!beSU5<h--jYv0O?n`bWJam%x4Vw_Dj}wefr4hIru25<9oXXDk<0S$JJH!_PqC2L%XdEPSSk7ns27eY2+K7g82fM9`?f3NNxi4%D~m#3buJcvzX6dbupjH=5g?Cgb@L85j!!;v?LDRM{;cB6-!S~jB;!5b&{_v{uJpGDbnMK{x>n(FVnorpiOL5eA2}J5s{+{uVCJhvp!j5dyT72UGiMj9#ZMklC}#Es0hvJp__nGT>fcZ3M)G(^02}?@gYZ{Fsru9?$}i3iOi%Df(M!l?rd|wETb!68q<V2wqi@u!y0o`WH`7=Ke%5(bW!h)bi!*f$M$p<0<ZMdxVA`02d<wa4)=LhY&#_lSxEeKlZ=`BZ#Jit{6;OX=B7Z3x?fWGHb7j#(qg%UcP9#~p^e0AP2GWm|FX~V67EW0=ThgxPT_?U^$fWv?BJ8>BH&Z2A1M$iB{=tX0SQ@)N5Vn$t9vUofs!5NCh9CR^~>n>)^F_yg}ZUYwMj~TH6iP74Z%spKHj&b8ynh2mR9x=N$0Cp!B(7!w;wlRu6Y73M^&utmBpM^sMSt!q<4^>Kb4_J#;DuH?p-5luI`NUV1(I(9Eb*V@$zX*skL2`WqKyG+t~&fcAZ-O%Q}#@5KhjgYVyt4Vs0eQ!AP1RD0d#PTC|trz|e=zrXp>gzvifuh(nMbdNO<yK?xI&;|}KZHDeQqM1c84mR=de_As4L5+HYD3dljAV&yAx{reSN(6N`*%?`j9qarp$40ywS$HB`3XWe)M$~gFA>;;+zcR5+aFMaxzM%|NKN`#HDJ?rwMV!-UD?{1RCS5mlSt6Dr3*8KQw@E9PR9DVoakRiM~rq<qT&ia0lDd<-SPHs--NY-?J%X}eT{=}Qg$NW9RdMTnPrRi{C13xqir;>!EYrtwv<nj|jSW;?xl)zI8y^(PpJ(NqooXL=04dAV;%W1y)Ay&Fjk07JNEcONd^Wj{D*@pt|e(&xyxQWEB=5j<~RC<e|;oq;}*p$DZugI%nQCMewtNbOD?6-<u4@V#Pu-3?EN*x5;f4yu0#-PUe*swl40vj!5uwJ0@-j;5=(`u%#Y&vigxUL~}g1bb$<|i)$uE;m7-L;mgsGXR&u~sLt@-;_HH|yZS{1TskGDdRbKQ0-mQz&r`VFq>va$CTItSObj-==wOdAW+iixv0WgvoCEwNvhSE@cSiy14PPD{rp7587Nn8Hj+c^Oe_DXjC>>&cMlvc3go*7nd7f5TK=KV}+9VnB+Aiou_dwFEvDer@Ym%6JTI!4k-f}VfD*zlIH{--9mo&UZ1>r<i=pkV(O(k+Qh+LwZQIgYUK3oDil`Nl3CYZlez3}-?F<LTi1G9bu5AWt@bXepVrQld>QLx8l}*xJzaeDmYK_Dnlw5B^1rshV6l*nkEjZgGB$xskd4^DI+M&xFg+dx*zhmSBFW(m<BhG{CwT8Uc2@DGNCIsPFG?%F4QZ+2tx4@V6d%=_v4VwW5k5oj7kOgm*4JmoM)!80b)kG8?v`P!L5%&yb^{usvU!R(W@X3|x{cd@KC9R+f2A7<&XUK6f#Uv!8uQTsgyx`KyQP~Y=TL|wMe_J|<8uko2*R;kt27+<dr0~ypVm$^2@c9)bUrerIx-BFvD(=Xh*t(@a#tHKpmNeD4W6@YGPbm9cj)Sp7^S{d-2y$F2La+8X=qbl(_F>5LZba{p0G8DJN0{mCyQ@oHW{n_Dh$3*ysUZ`O}$CH{>d8zqSv&Gsn+6%(N|8}x#LP8bWr7nQ<@QE7F1+}8JoD6ywqxTz`agj!C{2{HK4A}xVd-@h)GZI+A%+&*1=aQiJM(h&^mm()reM4RINE$MMqwF9h0(k81<OVk3D@;0&|zDwZ<ugVmTbE-{#R&VagC=-EZL_c<Ag1lL=UDs8KeXo$%9c#tE~%eDfWK4I3{tNoXI0wsfpHN|s3JYE)m3w2+GM?m(H-s0wRgJ06HB4E7MwPtCR_j2lTD7b6?I)Yz31zhpGSOv)5A)dQb&3VYT|yzRIGLFypqzZ+~$oe3skV1eWCm<%=jo>IRwi-?euw=(S__2hnDOQ3i30bN=^@k}mIlPgCo*oh}gjx(*gQJ3~V-{K6u)p0O_T<tOl(>6=QpXGQN_j)G<7vsZfGgSX(RCKf?$9GB)i`jbYbJa7P4bS?2b-X$0EoyFLpF}3b;YoLUJGIELodQW!Ezu&MAbjJ5)STc;IMMAI>K+c~2peWVZKZD&Gpj-Ry?p<G=2<2N-MZ_#+q*fP)GCCUeo(St_O~m6MhwmUXP@6FeZ14ECx@~hZ}SB*_mJ=4UH4OiQDXK;?8}Dp{FDTzFGP>8!HWV8uXWW`a1@V=%8wQ-C6Rl+ln1DV9vYy8h&uiR49gTJlc0hq?+A*PSqJVoo;}1jD4T%(-6Gf7{~?OCp+L=i7kU^?2-mCL$Q33RxDcf`D*}N%B}0{y2!^P0$&52%D!cxP1_Y_NFNCh6cjochK2|~j-GXUS?Dzija#j=ijeS*?qdGw`XAvD$4^}dqTRgr21Tt{!Vu;4NTdw-n#1Iac^;uosF3m+>uuG9yXtvU*^<kST7AQV8*&}NeafQ+AdqQznPkW*84tAYFEMrmX11&aeNdL2MH@E~z2{<tQj?wkAt8BcEa3WD6IoZ7{h<>QcAD45?%uByaGQpHV7*PiN%@qY5)IG4aQxRBzkD>1ne-kjE+medLJntJN4GW7%=9PH65ifzv6@A^M!xX=Ff0>*<lgAtbkSw&-t=sXIY@<To8W2Qf88mu#v>1`wk|}QpyjmKy=jH|WF-h$=MSQ{MrGXNO_uCA%5^vEk$?)*-;U`>kB9I(N^J#C*lqQpoj@t9PPY_BSl`-!q)5M8>+_&;~W0Dp(HOcUyqD)6+C9xiiQDO#!UONtrTGtd+ZU`mnTS&Y*j@MzKmTCnRb&oX-Q)D$aiz{B87Xe*6-*dK<*tR=~G?NCCxEf&iJW8%6HlZOkEw7)ghJ6J&&B_uFb_(NLHt-QsuWVmcDwK|z|4VJ&!yn0HE<a%Sue>P*LnLZzmanr>KOSgavICiWxC>_;K*6GU2M=>=V#lhpxK>@sQ^*BN;s)o{PduXA-Ph4XA;0PiB()mMcZy*1BR5MC;H*_oeJPffX;A06*$>M2mn6jD10e7SOxbD==xYQEx&8_3aMab5o?TgOkomE0K7a^MDpd2#4&<P;tJ%a{4s0)P!gujmKI`|2H;lzeMW}t9aM0<GYRp1;q4UM%J&DKQwz;}LyHSEWYKQTqflcFr9geDxYNB3bwz}QdYqnamV>`rjcG$Y=`u2<C@(&j5cw*h8`S@M{#udb{xP&*!0kBML0L;Z_!0MUPA&*C>@pNhUvK89iwdwSZ9$X&Un%EuJw^MZ0+S5_BK2dJT&H}M`ETe+{{)p&XZbL)<NiFYN*!WVnfIWC|w-8gsssSgDkPL%%bFtpySnopr4&*2Di@^gCCc#WjqmvXv%J2AM^GV!&)g9#!z3>(XlYmzRLnIANehGoj;L-+ng=qYIe$veTRhfv6tJ3bS*M&&fP<TB5#s7cJe1+cVkr;lc6Sr#CHj}4Higu2ySxK}mGjjX%_!vcL3+F4J1E)@OrZK3EAFJWub=kiX$j8{9q}V$;b)PWoO_ncH%IR8ceS$ELjLn>Od0KI)RY{-Mkwv;l*R@M0@+S!wg~wJz3}(=GMX?uPbndK|QkNHY%$pn@N&287e;*^9#$;R-Cp*T<ZNdnS6p##cx5>!?YCm5k(3$}ln}3FZk8gNEUJQ~tZ{Yt)^wOACm32F9bN5nj<><s)y_2%Ft__@?D36||Hi71D<>U$$huH4zjiYKoi(MvU@#A=ti{7&;!wyG6G2(vwTV04;fp~#*q9L}3)Kn!L4>GhP)wGo>V^wI8p)3tE9l@UgupXE2WX<)qNKm&}JRg$){?BADRX{a!7pbTBy4#@A{xTE_c1lJV9;UjFRo3Q+wMb}3w8y0u`UerVfHP5`gBI8X!ZSH(y;=YM6#$jX1b6Cq)i@DM%nm`H$=p91HT5|RIG}xX$YeNGsK2tfmwzFvI`W;|lprKxyau5SBXm&-)f8zzz$-qgiVXi8e6`7q=H{i?dLLs*+Q(abzs??PGR+E+?0cj+AO2ipk*F1S6dR%d*|iYQ!^x`GIRyynQHWj{7HUp*esbny>lFIXG4M4d31Tf#m2(2kr~oGKbk^EYBKL9Pj&<#n4j8x!cA!acQH}-<2L*ltFMU#<;UWqIY^K|r)u_}^a9v3oG!6G`1Y<5A{*SY%AwSq!3*sS~;BK6g`vk9iW#UJnd2=ReReO8PSx)WQ5~5%VFPN#9dI2T}4Fw5H!_Dyt+u3*0NveM$Q=f23Tj*i3XP0UOkFgf%Wf>P}|6vd~jx>kgmHFrj01}XfuI{=~&5czh{ZS*j+Mt+u{DPn)!1Uq~7I2=AJ?g-qz$fy4Gx=N%ajs@pMJUdba6H?fq>ATSjG<tfv6&`Z#;)vd*{cYgG|k5^he&e^9&3S{p8<^K29T1jGVg<*Wkj8h9fZO<%v4b>*67^N*g9byagtIShah+V3Pe^}17mzq59?;|yh95*Y(Kc^ucl{w?tGP!n~S&|n}vhhCN<8_bTbU4^cTYgP3_7$lhhExYv*2Q>fzm%crai$v2YAmYMVQCbb~?rrQYAc&`F9eDYJe-sM=U5u12ul<3`egaP5%_i&JqAXuR#5$^5o`;tsB<O1Z=?9oby#<f<1a>68{)wdUS&H?(zCb6fHcTsJM&el`I*j+Y7?%49qfg^L=A#G)Zxp35q33q>3eECfVSe<?z#HEnTZgbR8FCuuYw%!XtMvJoo2wPxZC{f#I}am|$Thc+#98!w+1ri#VB>}*U4rrndoJP<+rQ;59=O3ftCwiqPi5<0daEJ$Kek%y{O=1ayuuN_qYEDSct#)FcZWkb7xM_x#`ecnk&F6KcFX&|Kd8~Ff4^Pm`_3~i(uY=DBeGGY5Chy2Upi-C=W@3rfKYD#Rw`QO?s!^QAjd<n~osQJJ%*)&vqYyC3w2D3pIZRqJ+<)4>2+8a?Z;5g?HJ!6Vi*clr1YRH=yYRAWTL!G^fD9zaBA%_b@h`<~0y6bzSjXH~YNUivSINZtMtSwn$?{z=loJ|?0D}Gt2x5&y<n%}w$8(iD<Ru_78`^92PXjQwNSvRA!zfYdn0Pl7)SkRni<?iMCRa>HSMefVp$tT7{xr4c7Hdd1kiK>XDTo^3Rb(h{}MH#!`CcOO|gRKnc$-66(z#m`hyKp^g5H-Lkk*m$Em@pysv7bA&LHVRwVVlo6sT3^k@^gD(Kv`{?D?+UvQdPKYZAUDFKBhz8gp=WIg>wits<*M$nnUF>M6^gh@8b^F)5R{vWHm?s8<uO9vmc#14VMdlUv|Mqqc;F^^timBa)G}1^mr=!fTZCZsy$1LUOj-M6>xQcGX@QBpd`})m3>*XT592~>O3;jNd3E`C7myXKP#j^K@<NMB|<?j?-ux#REu0LYPK}Tq$g7IS1jjtb^}36i;rSMXCqWfS9;Z<S=nJs$s(uC+osK5ysc_f>d$ZGZx&-77aMw+Rv8aZ7iXzI#9|>u37QfCJ0pQB4X-<p08uw_NA3_*J;^z^v;#XEI9sPfk?N0G8k62s6)N7usP;09fwog$6JJet0N5KtQ%XxR%+au|nvC;W2(;4SlBl+^N;ru*V8h>|mc4>v;gJejMdu@^IDkWq*YGi_tV-zZR`$Eo*v=u4{;rcuN>F5*`*$vX3SCqP7q_sC%FGj|V2f(Bcl-?kNaV;#6{NGkfK{YWqXs+hib|%5kxVe+6kD8X$9@i<FPuT0fKTS)x|95)4^eeIvJfy)c<ii$mWUIEF||Ie57ATwNMCK#E!t7Kvs>ic-A=;G&0_DH(NR=L0d8b6$Fj`l;;yIJjD3qarmLMNVla?L-v9RDh>RD`+RK!S=Px9Z#{wa=`73)Y=x*EFJ<<g;u4`>M@acWI9|P4xEBmpOdMJ@FH_qxT@p<tSAi1Nw&L`fl8f=<SMu=Ya#Q;5&Qjs02j;DbO!QJIBsYitv=FvK~vS})sz+OEAx%5GZw|%ea-l08xLGl*0*kP^se|PCsBQ2SR6Qy-Bd;;70gCYz9-O-3-PUnXTY;5f0U()pS1_RIT#zBIGx~Ya?lEt4iBsm<j5$FjDrr>6$9#9{p#fbVYUQ%dPdE7o+(ZFgYs!3$~7b3KUg4YP%-CT*|KrlLW5RkYM&g$Q9ri@C`DXib#BfxFxOy_DkcH~d@Gjc4U*t4gCXlF?mQ7Cq0m9_uL;`aEa;?NR55&ALWG$A%5Y(DGW=&2`VVg?HF8^({=Elc!zV=hkj_uY7?jkgmaOy#)AQkGQ3L1%1U4X?N2K*K|P%&=cHwa*_Je0b>=Y4tAt?%|tAq2lTSM7(TF^(=Nb;D=LUI$JhzS`h1xEL#ydoIN`^AX6$M5_H85w;8jUq_K!zfmJI~oCs<I+BQt)LNq{-VhB!(Jivp~UH2uyNsKfc5|pZ^irDDf$eZdF-MqjE(hoQW$t?5w$ZkrQEle9{lJ%cfv{1>P8BZ@T^sE#0ru~<z<BHSc)6*NaTbE@=7j~Q}EFL@<(FnqdhqkB(Flb%$t?<-DpQKqit{dfS2rwMg0Npo+H9->o^HC`B0%cfkzD!f)zrWKnF{wd;DG#U)eb{kEcGGkG>h`pvO6z>KhLxP`AEtR!kL0+d2L-wRy8t(t%qbn?haGu*(>34=)d+X{TdJLqt6C5^N@FF_MkFu%#qM?mL*oL$3w*M*_NeIfwBI|8oXYu46OAAga780icSLEenmJh`%z^}SYCRBvtTsnWm`e-$sY#J@-P4D!ju3@ga2QQ5rYj;M<yDEX?!IWhNlL<aMO#htE$lMDCX;dq)z+&28!7MZV7}c9Se8NI(av>~KgX7e#4nSsDDG%DiVQdX@IP*qf9fwKi(0S#IB4mD>OfC{+8h@z7z#}ni|;JNCV?ff@BRSKw|5N7LM;H;a3lCzZ&mQH1(J4JZ<rg{%O>VIuk>|JO(y>HKNWbGSsR1p1#d1@^Gc)8w7WVju1=(bfY!pSCYi9$mMImjrux^wbiI=!YQNZLSWs<l!2v+=$#v<DKS+r0_L5ORc+jVOZx7c=P0YFDAxZXI+(RxMH5LRvQ*v5{+GQ0vK18KrTY=jsD+9T(loxrWz;hib-@lg{y?>zgm747-i10i?*}Y?5=uol{^4fE6AlzL$p{2MZIf9Cr+rTU*MvgTI6HdAjdH;GqmTZ?rDcANU@L3vEbb>|Y|1D6R55GC}i8QylBNe_8<OkL>B4kg>TBdZ@7`DZ*ZRG!N{KE$46;rI;prUQcO2P(`y6TEjJ-ZEk=RJ~}NDz}&{#gk?w+~DMPornFA2PB=(kxYgtS$CE#l$Y7F6K$zhWMDwN1uv^E2&x3n7~UsAoQu7vYMriEuleX=t%tqLUB4@xN1l)BnnugpE6-+C(FoA80i@)C4AS={LhC*Sb1K*YB#TG_xm~dEW+uJd`0Ad03<he@zvRcyvbJKHw8*Hr_d{(m+~ueILM@DFCT-a!ZFX=KA%Z6=bX^TgXwMwOCGCDQ6nSn_N;V5HcW^1U=Xg`P)wb4RGb~_p&%kh-_jUwwZD*oUWkh6gJ-nmg{l`3GISWv_?c%F+pYe-8cj<MciMTq9un`QR^Ve^W{hT7;~JocS|B*Jlsaz#1xwf&jYDua27SH?bsq!ic8|WP{#Tv7Bt!Qu?hW!(($P4@5(S=}-`U6G=OZu<by`3D57+Ao07C$^m?-+Yml1V@9uiXjJvSMo;}ZWlb*|JBpX&E-S&wK>UHjlpa36T&EXF;k6S*v0K<f_%*aZYw$FCpl<@Zmvy;NgXr}log`}&ds-Z8#cczRd1Eg%Kb9Jr0o9n+!9%l8_mz^NfDeEQmVYF1^p`}L~iqPZrOu$N@Hk{Y?&ciFY{bW#&Sm)~bzHEgw!!bOEzX7jn3>=mSA693a9H0YBe=!<$8-+R0C`j%1%Ea}oH&D|`*-y9A=f2;7$xYi(F`Lh}=zy8u)n=eJRQZjc%jAY>s2pp0NPD-6aO9^G(rO_;8XH+UGS%-UW`3PLjxITF+#1e&c>gTv`{<~nVGwARl+A)a4+e<f;tu-RB;w>&K9R$fq8nq8HCSx!(WnOAdx>aEa)8aIf*LbaNAXjrTS(}MN?SpX_t^_obz)ABHpn73Z>yB*pi30#ZYhB}DGl_#kQ(7bXQVGbSE99p!LkEK2-Lt(UFu!cOD;YK4GA_|hm9Jpbn=@Nasz4upR$P_X)szlQ+w?o_f1n86OU#L2&IsA{$?~}bkT7NH`(--xRgL>*QuJq@(ci@nauzZcZ{r{cq8;Uga3%BL)iZBz7;M)#gaLHt6B!?{QA$Oz6|@*mrGd^}s{6u8db#M=g?S*<n^oVQO-$67X2fFhS4@Z`Oj9tSEiA4CT=>u-TG|A#24L8au@j@bD*KYm0926wu^DkH)=qgTNILu%f9SUp92y316@p=wvLEzP6D;jB9&t(~7v2I5o^G<36(uUC3BaMKJrdZR2iQ*7gihJR@s7m?nlNvTL?KAbsp7nRaQ>JyM?%2-{YZv<fM8+w#K!iS0rm$Gf$Qw2ehU-@+#}i<UxzNey-V30!=<$Xz%vzPWTPAPQOutw`c&eL#Kg^@SjAU7E5fKUEotIIr9ZBvV;7zRSzV^Y11NMT9WSJVfj{G$k_k#28Sb>K44nP5JNHL%H(aly-zj5wMiPP|ilY&pn&P<Ny_2H~pNdf1%)B;|Xm<WesNx2y#`JN)(L179#^?3$QNgzE9~8v^eihUHs`>UdbV})Dvg!{Zse&E@f8c>k5dBghM?sqA*M1b^_ry+GY2L->DyvLn_{VXFG=gE*IF^6};RHvDCj)dT(4f_IB;z{V>6#9Oufc7vX1Qsj8InBZ2od?!90HmTbY{2kaovo*V~J(=1F5C)@KG%Aho&$?RGBIJj|6Jj#_wT~)XhvmO-56Z%>;KL?u+6&P->qagZcvf;Pfm1SFAPxSAPP@XC@A*<sg5)K)wWPho~B0>_SJ*kATKd%kA8S9hg56j&S7ng0-=Zyoy&T&EdExG;IB4Q&umv{1(>mXn{dWIpGoFLnXmHh>CHl9|OQCn3>mZ=uOA~2D@{~joX5&J?9@xIO@Y{GOT12>6@i6u5UVHmqsRUhrv?+o8R#s9Kf@}%+T?UK<BS(u4@kXBh&MNff{kiqOcT1-ZUvmtvrn8;Q=%ncfssAKbR9FABN)Fj^=)53{gNgta9p?8kz&(49T7)azd&!Bu^aedfz<qf=uW^^Lfi}q~r**F@O7zCCKZw9@$KHaWM$+M_8arB4PZn*Xu?$H#g-E=zgnLWzL}z?=|fR{q6Bu8U3d7BS2Dns5||V>~1&lrk=_I%j;wPbKZX`<x_5aG4aj;{__U)lZ)8$6cv%ah6|!V3^?OCDrBLhh7-?(R!PyiZFF8}0dw}&Mp7l-#j;#%ZV1^JomeEm$}(@T5dpH^F2M&;UbZDpu?;nbf%jo{k-ROu8oT~30D|%gDLa=WPEJ~dtb$%x8C38|k2e9n&a+n!dtfxvX<j+0lHu|{?@wlQRWgBDJ68@gxzWz;pMfZnv}OXDq1{fx#fI<Bvx_BT0bl)+d|AAz!p0^BV+vPlUV0%npne5Eq+Gx_R0Ry$-QAJM)c^T7#8}}Ez+yurgT#25geLB&e20P>&|I&~#lc)s>%+3raGPd{ldaHu<~)#)nT8|AzIJ$nU~Ea+fmFMh@(n*Fy8Ku5)7n4_svA+?-OV!v(d{ny-_!$d6bIHN8n#;qLmP^2MD4K7IY{|WD4h#)?k%$k2Xszhbx0+AVO8`1@-QcM<W$_~jl>r|W_vqsNkB+a*G<%Y2L_&`S5a+-^>hSI%$rkXwseCa(q-FA4vcYfiocL4Z1Rh9b-~9CS*ct~^7qJ+0c>E<apm4m!lQ)Q$~=%p!qgIxJhH2U7B<Rgk8hTWi9d2Zdq2twJLT$p{Cc@@OrbCP^?lM5;1m;NOLIVzJAC{XJkMx5S^<jVS(66{qL}Wa1oduel@FS87BArL_0l2By=D#=*3D)!gVDQQEfq!^#sPl!LV~c?x7${iAZ_kXJr2T(EQt)5pY#8GRzdeY-dY#_$(9IH`bPS+uvvscuRgN9I`^Yw_Y}nwv@%f2M){_d?A}Z=;Yw&*PPa@eMQrh$km*=xYh95{%@N~L+2BcmY=5axoMmL?zD1MRh+qRt834|Kvf?<rv<A`Fi!Q6xI)1-Ls=MCljPP-2_+(06!-t2kWd`jjc4Zr8>ggq^Ew}F^Zdq7c|G_bo%l%-=Jp3|5Cs`*PffBJ$@~Hr0&&%`;{2Co9mkIxiz?wHkXYYiQ#sVB^_(A75gc3T4vTEv}%$pMUfRCHNG<*xhJeg!(NgUHCjU^FwFME^!?D<+lvd@HbGBWL->0iee)LcSA$>+we>Xf(_J7>cJO=1MHN?r@)vLqq+Hw&w^^(}|gh-joUP{5>W5+U*EQpq>?Z#OCJ*spW3293_4_mX7`G=1)kZI<Lm1)5s5a;ag5Rw5XbuSKTngaxfJ$U>HPuWH@sOU(Tgq$>qu2nrJL*$z0Lpx|*{Oc>~lS8DYU)7X{0)-O^eTFBd1u}^SRuI%!1#v!iFFt<5~_+uwl(GAEN5v{L{dZH&OO0Q`>vu=(-t#1)>>GY@$BtVNalP;V3W;$`hJke}ucjkO1HBMpJ3(zWOVe3hSC!B6L;Si7v3tyg6HeVBPsVLc+B$QdJ9lQ0Hn#$nzGitLVBRv?2x|hVQ3>X5H)~3WW^QtEV3yPsI8Sknt2jj9m;SM?Dy;N%uAmhgt)qRMswD!rbS|4S%+(JbOJg3y#BbUBG%Ed~))VWBp+z&4Lp;+gdnJU~s1`OZ|pE^DEJWcNMd-3i^z~t-ca!AfNl=$j|POylV9{?mPZ9?>uqGzjJp+N_Cu+qt>YRiZCXo5x6Zs3$wD7arH%uydXROBrzJq53dvRj#oSB~Whh^-n*6xy)`KB@~>(<)&d(H^2u@{vYqLz-RlMq)0ub+;4^o(L9qZ7To$W~-zmQRi%k_3LJ9S=HZgJ%#sBn>9FPT}~#oFf>73NsC(CW5-85hVLsBmJTX2+#^}^yH^c`<o6V<TtlmG50{UzsW(mQd_z3G&aQOsf7lQbf`v|-(*5O)iIh8*x&sh~JHvny-S{##eMlVhFSy4RxpU`{Uk#(Wb%W(GGcLD8u!h&TT72>zBO}?>w@VMD(rVNr3qTfll^^`sY)%tZPQ%&OsmA_t7D^Gd!~`0h=H_e|pFH<HN8o3{Pq~M<SDP#crnwFIywSN<nP4Om1|xS!JHsLB6GaD*XZRh<v}4iFuu8?I)_g+^^27?oYkKb1CzLxbdZUL=9v3$TG@hxp<HzV6)tyDxx<|(ND4&Vzp|psc@I}2zP+qO;jzSCSGpwwN{*u_R*DLmIUd3DSJMJF@C{03@cYKZbPQBYDr}CN{sziWNC4l6Jol=hYNkiW<4{z$Jcpn^j`A{#7#9z*KJRiD^0DDm{CS*r9mxL#lJi-=tO)5vJ_q=Nt7M+UqcTy(my7PcXq;Q0+PV6qFaUJGL3_YYcV{B!%DUfg@EC|8u(u%8`Sdu5Lr_cYS&~9>$yiri27xucwh_}};@*7k5-aU?JV|I1qf@rq*yPw%&OnY~1uScD1j@?SrF$0#$RpGl!V=i52D|y}D>(a<hEYx7sZFN%bv$N&6#gA+fK&E23$SkF7H){8NYOBd7_s9=!zCh2`BAlr3sZOwm%C?j-O}O~zv6X+j;ShZzamXGT-KrJYF{3Yx)!7GcgQ{p;UXAuH?2AJy5Hg%3nR8sp$rd=VCLRRWB=p61w9_+7CIx&+<P5nDjLhnv+~1Yjd)so`+Ah`-WVp~sNz7t)U!!;@wHP1*9=)AWB~C6eiPH`Fb7i4P)2c2Sq~|JqOKRQTsj4qiZ^JQGU;s5^V5dCd*LNz}q^A+0Z0+8TJ(W^1XY+6uh_c%M@+ve5p`PJO^)<qyDUBV4%f#ncwE1|9|2PYx_S4Y${<I)jXcAg%#~af7kfPe&aJ<wI{9vg`8$Fg^8i0bxEeviH?&gSG@<O8wA4lCtfAYNaN9LgQ$8BhF;WkWuVC#|pUlhIztk-K@`<ag7F$LD~AF-oauIZ=Y$0O#3Pph(hf_Q1~i0&pGRJ>Up7Z@g~q@7c{zOWd;5-!2uEg8BIs-3U>>u_4p+m>M8toi0=sj$nA`YSOnIewlTFB_P7tbY4orJ*FI-g=Jy(o_KiJ!P&6b~p`Z59_QwDN~n38oMhNlIB}x>@OAPB&-$ug7;y4t=HonuKV&W#oCN<7cqC#{f0OA6LcKUOeKkq{^y=J)Wb9MlPo;ZHoG{v8b!*B1cM!j5(wsVnENw?_?Y3NA<wKf7Y#y%v+a>k+eUy<Zi`4V?12Juu-WxI)@k{cCp$Mp{uZ>2>d`DH9V!BgyOT**<zO`EF$WkLcl7kLHP~?kPsnpM+|5^?P1iqsfRPLQQ8iG^Oq8H)&V(R|^&0%4AL;$oqhJA7b>=xvjxKga^Hn`Sl#=$ma0@Cr{=;0dz&hLkQ7Qq*qTbvQa>+Ouxx5#fn|!Po>r-hJw$Iv)E4Kry(G+b6US1C&>(h^&ul=sjsXyVpnX6{5Ha0bbGgL)=!xJAD4$xFD1Dz)qme0~;`Y%fD9iHVRmZG`e1BIcSYH5AU4F{I~opRCA!9rS1jCcp=!=>-%N!1?87jRRes?UD)>4h@qQru586K?6wTdanc5{RVe>toD{sZb$g9yFOZ-uB|cidi3mwxQWn@E0Nn3XMdWr!U~c5dvBL!lu02OQ1=WQ->_as>@bPM?5t%zdoWRo)g`tP}2mq3HS&%Y8=#A$`=#R|Mz3_4)0{VawpHaYcE+i4T<TnL3Tgpw%d^l2uj(G{>GFVsfn#%y7CD++PHFA_Qto97M2RvWZx+`p4ryHAlp-2SD5UKjbAimJ(%v^#w-bZxPr>Nbw1f|O2%37GYl3S!kg$N5U{KK!<7o)x>9!CWMnvH+IbZ-5=vHKMY+DP5D7G1{o<7vlqEH`Z($=7+JPp#C-C3ms~<w2^!#Ue=nKyHH|FtwtgpaO)xCO8GP=5wPSV`CV*MBAwl0l%6)?oqOpS^dkSUwHOThTn^}$>+5~J+<i`$!uvPkgpTDbjZILUwyA5lU(Cg+GiO@j2~pD22CjFCtUVZf&qQSXmI87nm}P4||?ubO5j{#HTQDuldcJhvie#NkeCR77I=oa?Fzw_y7^k%dOf)NvVQUlfSLSX8vzRu+ix7#Ml&MjBHe1BWdYr!XQws<Bil{e>IC60lsMqi{_fRnt#<%R6qvlxf-}{(cW!q{;8BE1M|kA_6n!nl=ACM~FGHd)o0zqoznhrjm>0AxjeBWbV-Zo`K%26g0R(kVvD>rZJE+H{Wi?!=;VM26=bo*_l75>S!P7(K+$0su##&zX4(ls@XQ0$p)5?5+eVnXk4rM8NLF1lAYXrscqZyGmI6QIje;ER70ah1b$yKS3W`NUfX}Blbotf^BHFGI$9DzLIlcISuihqCH5KR9$<ZwN_tX%$_D9fG@}ixdBZV`q8zsms;tXdtxE^*sJK(F`i@{CF=~lNxb0vKZX!aE5S>~PBtwQi#ptQfLgT=2Z~h7#xg`Vv(iPYRo?WY`JvJ2M35BW`iU(AEi}ZOuZ0I^F+@~U-J+M@PC@((<Ts}2bG_w@Y(Vn_Bk>`AnN?$jHFvL3{;@4$Rl}BE$%P=UiO|ZOLJ>b})AyZIRZgea1GQx+UGUpg$u$iU7#waiQ;~X`Mgi@vOa%Sl~4Z26S4R{(lvWO+Llx=2%<t@szMA*hR(*i$1$vqKi)Dr<b`F-}AV=T6)!7xxr#Qa|&AA=_$><j=b4kULET(#lfCDt{*{C3QkdM6L6Irx&Y>$3=+pMk1*aeX-01P-g46-4%9zo2lHvQTz{fi4$35+aJGX+h%52xT8VVJbNHgT9xfs`g9|HVBJ0GP=FVSt$>;Q$Te(6AB^`Yofx{Vnnz>hIt<gKGzE;RruVJ_sJtLgb>0&(2D%IQg`^UZ7%u^0*@jc3IFA+cBu^Wiwl8Hl*rB94;y0>7KJLh1EqTuf1ucShZI%s1`KV4x`C4dHTjRxfeK;RdhUBvIC%||W+Su#%B&#H*POoPfy1XQ3^=ha_%d}oW=^d?;nTeo2J8+YNsuC2H=oL-_dp#VPFRNo&+?jG{1ED>q_sv@AhN_#6~oFp(^%vXm-xh9imL)nYdjW+v}u4jltoD<1DAY(9(CtwtasR5@PO(aWgut=!SxUDe*+~`Sw|2zJmjj)_InP6g1ivGPB#A3_v+R~IbxBTbi|h;?#|1&c1tH$5Jq1sLn&LWy3Cm}FE7ETuPJ|qX?55yCnSs*qe8-Ol7HZRrt<S*OTktvr+?I}BsbMZddeQuA<i^~!*k2KLJcgYZ1a_>YY!K}-I9SsO}_5X3naIBx}lQEW&>G84IR}Xt!ET?7F8jL(I@M_{As|NMK2XZ9q;TyZ|?!RM}2q)VHz@tep}ta3Em@9S;GZ#O$X17$abP0!b2i9Re4AKz|<;3$6ueh#u^{SgI20vndP6DEg}OH4I>$u(Y?MVsV&W<LW~uDkVV6_$%DS8y0nN5(=WH>X$Aslf(RZvV3+$E*En8x1*!6E&Cr>62g;bLH(&z_Gik~DF{yEZ{nvq3Hwh$*&B!OkSYZ9R^BuluO6oj_!o?EaId^(RLn3y(0J9c6I3W<SP97eti||1jowd9v-22Tl@(Z&BGqr3AMCNNwp{LXgC`z;0-QXvb<uOr)e3MVFNpyqphyk#lAyWv$3*Jc}ppGaPDT{;(5It)?)S>3q7CoYC6bb?zCG;GfCklf@R|S%CXktA0*yXeJ>0Z{BLF6Gny-siTAd-MCEEe6VfH-RApz>HR&0y(BUVSe^*&ILEW}s+_mGx^FsPDoqbkoC3(ptZ@obUWKMQJhtJQn%rH;|S|s?VmMfL<ZH2Do22K<<H3N}x?-Ep%awmt6tEyH1{c%#hBm*)jgiP4aEAA?ISWm?8o5G+Fzgg<l^PL$0vqk&u;{`QLQ`IU)DU*797+GpP?*6Tv!WI*NC_ZBu=b7O<0RECJCwfjoYsW2n=pd*h8#Z7PHwuP?FEM$kR`_-}`tLC!O2RU=R3#I+#gC6Z$hgkVx9)W5kQ_@wQA(u$0ZLDxC_rcg@i|G0*beJ?LTx$HWG=y%t4GftN)RkB;5??{j@67K_8L^d3JOb34p!)3%7ixxA1*ser@hd(_<FUXaMOtfXU&R8+dx2TIlxgr<|?J~rwbfmLJaZYIRWvZcWvdAWgK&)>QWJ@gkQI>c(mPTMEYWgq4csvL>!``<p2}VN(ZX9`1h!pF3?smD%#UImllkU>z;m*AUMlJ$Zvv}=vG*eUZC7wGHD7hw%!b8^c%tw>=#c;d*13=I54n^lwm@O4Zm#px~9>48b_t+EVp{5r_$y1#a2M06G)>L7nQdN9~rAP^W;@&b4Xs<vI?@H|QW5|S0#fR3?I7r#BCWw{zqua>c@GEYA>wKHg-=?<#|IQqxOaJQV>_^F}&TB{^;HirmmMpb_%S3)Nq3Dm!0baOaUlFKfM7};LuiPkd9&xrSe<y)^iWSZ<vk=l6+gcqX2>wxFn5M!7m*>8z1S=0h2RY;T+ke8o^sSjxtk?>HnLG;BI2)KZ9i`Ni<FA;EqVNYzi?4JcDKoS;?PMQ$bWJiK&mc3La(qBOOZFqsw_ue0nv;0qM;AJz)14zWusBp|aL4E!V^9|zO?_nxnXBE>T?Xc1yr~(#424R}xt}C`wsC%v-N1Mh!JG>l8z-<@1jxkoq+b%NY*40}Zlk`u%$9fkn1Ao6w|=1x_2;~6tz_m*tz7R9Wl|e+l8{CFE|r*ST><!|+muHj>D*KtD4T~rlNJ$_2PYQ;Q|bZ)fnee;Cl-POS<K(f3%z-2Y;g%P8vAMO1pymM$7q1?L9QBDTqC${w1%mAN0}yngC{ns1$_Bka+rPPV=--&3yVBpYaYI97{If<&bSA3?FzS|i0p84^$a9MZ4vi*&3)+aSy-f^m-L0%%p&cH^nuIy>6fVGt>I?Jm(#@?5I<p$ASn$W2#*n_MjJ~?wr37|rztfab?kkgtM-yBn<eB_H%ytLx-wxDWpCyVQDOHdQtVjRiA=tfXZ7!r_ZF;)n(;F3Ll3UMpLRzvogGQ-_0?d4Qo*&z)G8~J$W)EqyNCI{2-PUvrhb$2pfPoCGZw~Dw&c5sX4x>Y<d*yYfuZ4(uV$n5_5YIWPi!f;sxuM#$<-EI&PPf~WB}<&Xe$-FLOP5&2f|B(DSOOVW1%$s>!LWK@<NkiIsTQ?9C@<xBOL*JDeT*l#g>Zf?MR16galiL`|u0o<@2Kt{Igp9-GHSHn!60qToS0-eZH|a;|t;QKP;R)tt+RYUI2(qV0kqO_k{{;@}%QH-Rr6_>Wed`*Ne~>k;Tt+jwTk@sM1DOJ=94m41v^-^_j4hF2k<*#WXlLhRt<QnPjLM_8H+;v$5H!pw3X4$(oX^@3Ml(m_4gigz&$S9tsAenC6VW(tVFlh?$NYrmLgxFJ_hTbU+5)L)_4^-V7gg#V(84KG4CyWS{HU-Ee`FBZ6at{fvRrjjwaS8?Xb4;clk4$hzND$t-<ogTE54^_m%jyHf;n5u1&AE#!1QV!BP9-QA@LayDDt81ZsFK|FjpJF)ApWgEgHOIaZxWiLR=FolYrzgU~(pXp>wMsZhD>>k%>qde;7mGLWrfmys;GbJ|debNn*i>Z4*3T?niHbB8Lf<13$CgT~5eQ{iVTL~0kb@LmUlcK@Mhu!pQQY&r{mpKqZ%2=_Oz}s2VPsHSZ>5e>5iuPsFdY%HbFFRcmmQtycS`|wIOp-NoM=^Cx?Q<^_3m%*@7b|r!<<gd*{+9?wxTHDOrgZcs7i_VGz98L3<#o_c;Qxtl&r!waz+H*taLI>owAbUk-ZSDd;r+OZC)L4=7Ro2=gy|g7$_|chDvl1X`g~2OSHc72<Y*}!N>&(wWPK~%F!k<Qj&s}WNH2H>N|P-E<V@S2E29|U)}TYg0X~@Elw{EwRd&!wnmj!Z0S83|g8xc@HCTRajpzBYTNhvp+om{%qq3#VmN-=P)I^3jKv5qNB}w|_iOI-C<aN9c2X)k8cuYixN?PS^cZ(Oyv;BVm8{)MfrZN$lRi!+^$kfrWN3VL?@W6w+f*5@xm%@?VZ0VV+EiDJCl?>J>!4$mf19Xk?bCk1H95$rIKh1+zp=Obm(#Sdqf+d3529-Lf*w+k@t51c`rlfOVV^$?70j%Pp_)ceY>43k$LUCY#A=_>$dvDZdlSmDAwZ#Z7pZ2FS<y#R}O}YPQ;+63~lU`uoc*AK`=1bmXA8BsxWBZ6;R!+j|<0q9=|Ki*?9Nwnx*7{d7n>PG)E5_#5mCoWXSCF$hlfKEgLu{+x@EUtS4W|(v7fYM1*&W^gfIc=c>GYH*c1zwLDJ!J3upUy(i#ftf`W(R*>5Jy1DOLG5AwN?es&!Wx6_k^K=tMedd{639js-*B5EY+r<Qc+58oqN2o-S2c39s;ZR))SzzFjcRPUb#tkh3;!)O?gVmwAflR}M-eVU$RYFEb`S>uvtpyhHS?Zqp{TFNO@dsn(h%?#|=ayG_|d7nI=l=6#PIo$_Uj$+y^on3l;RJez5m@hRSioC=7+7#0)7aA~z4?Ei;q_n9F4t%`+8ah%`*v8mlSug=jWIIxb95Q$q0@M2WC1biH3%Sv;%4DqyalMjmY#t!x_br{|M#h0UeRxF6`&UQ?nWk@I%l~C&p*i|P^XqwH4EX7H7&^bA$g9-Qod#Fx5J$m=v;*h1%WDVsM4B$0mdY<HXwqXmMF4_Z66?3FXjZIYppAfm>;iDcdyNj=&`LKcc>-pZ*Tn+Q@Nn@!kAtFHdZvR3|AI4HVgphn&TwL$w_(fi$W#SyT5*0)QQ#zmHv)Cy|>@CtlvwDJUuiixf?%DjHHU+)yqO0LDsD2J}9_5PyL6Nc$39+#K)ikIt&fK1<eH<vvZhm`~5-yY1?CP?;0N3;Kp_Ks(vML6|R%Ih-Xp*r98d;YY<<%Y{KK<YQ;QP?-cBy|JQ1o<22hi^M-=!y3p9{`<k5pounq00V;nEe9k3L8)T&EM40gq=vzmq>SxJi}GiQWwZQlj%OF(8`+yE0Tz&)`CI4Fu|ADe~^-;llBJVC&(rN2+1X1rmW9JaRl|wL*y^grNG!rQKGte(HXp3GyHrT(1s$7WDL=p2^y5E>Aj(45FQHWo9YVT29QK)es%rZX0jee5mA(ADj~+67-+kdb)U@njjaYF{0#quL3_)sjL=F-^LiajY`Qjv!jE=IVEJvQSbR<;8;A0XO<i@*68U}PN$q@H7lL!v44=<3_b6xNZ<aXXMR@2CBJc60F;W}+n1lKg;yO|H6$C{LtNx=d7P~wN5JxI>*HeZMWMrv2^3#)+B|DvxX<!h0G9Cnaq1<xnd_T1?v<?FYkVc#E9jT&t)(1LQ&#m>IlaQb2pC}er7J4i33mVSvX0cDTr7h-t}cPcD*;M%^*6Y->M{IIQW`2?t!F&Hd(73a4A&YfSSzRb6z_TG*0EaE45k@y{d$*zwc8fz#&7jb7j&5_M5awJUGEV|S<=QhSt!U{by4s(z{e68>{%n~qwQFrYM7{m?3c0G29#}b<EKp2r1y`@#4?r%!GfO&QRzBiqaJ88Atu=68^5PXY{gu*1DM<d9)`k=3UHm^Mg0soQ~PA{87{p6aE!)TN>M^J+!-L6PmsQOZmu^*$9?HbUS<%(l8)Vj$@TIwtamzo+4LPnRw)sy(IgdIPW!lVjo4V*enW{XZ7+6zi?c?O6>!yy(lW^`k;#>El?kg7B!nqML_&sLCI-KfMjBxE(0f6R9A#ppvtaOhrovVacC5t8a=|Q|;=^n9*9*k$Dgib8`B`HE$_g*1?J=0MDW?WBx?gGAVhIrk*Ug+2>Y$UZ6jMOzOpxkVtulmUdp8sOjdC0Z5zaq=C{k%JvLFljoVw$}(sbf%hh(6w$1+!q?wHsmk~2Q^)O<t9$f!2Gf1q5dNAG`R*Cs~44)h95&WoZA#RJn64UA5&lx)RafC1e(OBzdg-Xy*u0+Wcd{lYCOU#1@aNh^sZQdNCmS@E*W=RN09km4&|W-aAj50Yk-cca6SB3f}(o`8Z>4K#z?R4$9Png3J&P2-flu>p*d=GSdD7p*G;p0&S(L<A3<E$-&|r}e;E>N809KK;^pvGgQmPxr2f0|)`I1yT+ao2+-M0A84Jqt#vU1SUmfp+-LOGy=ojt9~*bua_Og|1V!Fgf|xJQ*{tjK%ffwnt{Mq@aWshKMigt0k=9oV8o>P4dUKTl>>oc;|(bsUwh6*MAspO)D7)Tl%A!NP#O0=dgEbFK`;kEVpui<;Ol)1&Jp13Mdgd5l4ap%nPHbqwP-up3NaE~Xt%?E*z19(SF?YLo%;+a6Rd;d|0`O60GuwStf{AZI`-oM%Sk3lTchzYUnM8&_47G-&S3ug*1Ipyz0V>DZUBW&_>d?XwM$4VSA5Q~x9!c3rrpP-MZU%&fk{8UDz#C>$K|JfuXM_FEX-l8SkY^*GTx_r)_r**w%$w{mE>x;Z$o($elkaf0opnJx4!WLiv_A+EGnp88rv3V8jD?7w&J15m>T;?X(bKBX^8I)o>?6Yr)C^)Zn2?=v@SkxJEx+jg|6^ufEl6MKO$uv3=65|MVuE~6`fjyk^X~DKi10J4p&nRNU4cBz{1O&uZck4H@Z5?pWgqi2-j|Zfd19#GX7qAB32e9A4l>J^xW@bENr;$1tkfI$8MxkP>t5m)IUV9E)moN0!-!&s3V~!>+ZdD2Is{52!P(M&t=-f-QVkWR(Yoz7(Bec^EZ*b%S`W<AyI{ARM@s@{GF9UW$>m6aavrUQx<ZCfGAw`h1gLm+la|DurRfK)mU2=LX;1-#WiLWh-{3+ht%bQQnNqDw2->ksK<<OjKj^VtOYk!odUC1+d9VD5&L<{R=~VL#VAN3gpe$6YRYv{9%%cER^sW|@KB4zHptsRHV?wh<8U}8@mV0JfAH6r+DyPl_`7=827yRZ&XS-?5bX$IQFjZ1TjA{SW}E=Vx(!)cB1OW9*Xg!}x(h-&kKe3ouW@%KJ`?ymKhcb(6^3|kN;*7Oq!}AtoYNw{<$i*ZYg5>uKJJ~2yMY`JYm>LTSSBjhC=81YtPdAdBIi}R?_!v%m*2&9g>0Hys9hqzZqcY6t_oEisHs1EV>IPANBP}ZS&POT#wFc>PLvb$LGq3nK=kCZ##H9gYOpR}0UQ{rBZnB&AYB59t#__!gN&PQgI6sJj|bH>&0D~Iga8@pZAi&T7VcG^l4RnB6gFMs397@(7bVYO768i&1h*5w!~`37d2$a@o3bJ`jS;+K5X-2<e@q{-MnI22?Jo!_&d!3dgSF=8sGF?sG||yY0q+M3rBA8X`|yuuxMJ5*A&7wk`hF@E7|5xt|7Eip+V_X4JDAd~;Q6ulh6pwYEH*M%fE{!xBbmvSx@DglvsF81r04#%`N|N#6bWi)Wkv0g0L5OHozN=&B=rK4)4LMd67-Y@Q*2Yc7xuMI${r4n3=S<5>!!1$uy=(<ls2fWA-<Pv<PF!)ucZB#?;iu9?cgvjFtXb`g~K7p3W%jX;@1^ka*rmlFSr8rY7Wj^{GF)`${jH-p~kgz@<WJa<=%M^ZBd%$_K$-uYanOS`i)rqP^T`5*K-c)($&Bh5)&}e{F1<IH5-Zmb-+|nFpZeN&G*UM1Nwlqyhf<$gAEuu@?D7?!v3|ox*U<*v5bKA)!Iw5ejnzC1PnLg<O4(>(??Fqb}78imm$b86|S0gfmvlqKgv9?p)EwlUQ`#zkxy{!@v7L{Aa<uiq%xfzy`qW5&^N)grX#!up=@#-QM1Y5g&H&yY$eOeHW0W~&)O<S5qUcEv3Hf>&Yi0y{Fu??C|k>)*k*@0yZ1ja2mhl%t*L0~$M#!2Y!gB`97B6uSX{~$WCl9_;XSx?Cb$5jv@DFcMFdl^aBqBh4)R|hLys=>G`Ll!+{tqQ{i&}n+VvEC+WPAhoqa4$ZS=$efFW?t(O9=td);YCHeU0ajgCma?cltrwAo38zx*zLPk70zj^FSI{3M&`r@35#T7xL(>v}&;=U%4D;wH8&Tx1o9|9#g3<l=wi=QpIbjj$F4ECkHj{*lpLofL1SSJA=GT))MP$-(C@s>3{@8#?51uBb6YRjCyQY+I&kAy{Rby#6=T=lx(DHST0F-Xs;Pw}ILQkRyZ7LMu3*WNrv|7nq^=Hf=g-IOTEkp_LIroF(x!qIh>{OOxQr8BavqH<O6Rd!Qd!AjWrUMJh5SbB7Z$<LzAT`Syt3E$EB{$Y)z94<kKKuK)|#3-`#Y-aqmZz55VNN~#m@+|PWDi~A}Y8i{zbV2zi@xsPr{=r$uj0$hS>_iZ28b+hHAo7ydsLcAwFKRfac`^rqoh?uzCe<vohNqIYMpH8})5~pEz09YCpTfxJ|J*wc`E~k1kfP(Hky)c|;Lo$0hQ<`F#LB9$Wu%m_ih$G4$*TmfThiUE2@v1A7KEW-eB+s-Tx3`ig6MY|1KKnNYgR%L83Ksp;KnIHeieUUa`H3J-Yel0`KI2&#T%kdEgExOpm7Sz>dvQEd>?5A`sA%o2*I^Vf=YzQwG;uCEr2vcktyopWoTq-c*;q+djyLQ0i|M;wBT@zBqf|H*>8O)(S<HIw@DT9R(GeY*$CFZPdz*k7cyN{wyhp){1J?bgyo^doS@As=Dg)=dB+XDvz!bj&4ct4`uajB6%pQ6ittsA77tL4NR4lV-II>R>hZ}LXwRm?=ylIH6<IxHX?3H4-Nr?S+`ak3{>naU_Ex9uIc`X&%YDO^i$NKK)LjhXdHAGjOx_Wvp{wY5yH<12dDCw+3Q96jcS#I{qyGvEZwu%P!WJ|8hV8Z3&a}CC|(~5{X^rNsN4G+i4cro3D*czo?i}TXGet^3fus8U<aw6Nz@s$5n9)UqAU$cRXTNgThK?>jG!W34&cxN#jh2+-O3#=P7RG-f$abxh^-kQ^l=OHNgMB29%N+5n=9i#pKmi#7OYI|P=6<^4Jy!}X36>J-THY$Sq?}v{!aFWISqJ2w*g7Z6I@KE!i|AwmaZt}LhGuH%weKux=58B$}{joo>$S^iM<mX09*ooF_om2=CiDR<bO$rC6)bYJr2`PoIoReuX6Rw!=GstHl%lYpIj(({Rasf&2P|89dFur4&IE$#?o2kk6j6pNdAO_c6ITnpzUdk@W*`z<|gFvvAV}8-zJg56d6wcF$(0JyiLH);4I1Hi&Pi&m5%H`734Pr*fkmX-4sXv@w2gH~FXw|NcSNw;Eq*JGS44tmASX3-XSTR4~z6L=LozuOazB~MvrDuMZzZ^|kIbT5t(X@s3ryW95H<brQPDf0A3`)Lk6t6bKe;novUEh)sp8N;o%0DO_?hCF+J7ZJjg3w5$%5P`mbEGa;E>+0NG2K4MD+cZE@2&EyP?VOQX-}knjJs{sl-{k>g2lAwWAsLW?~fb#U}Z~LHC;Hfx&r46t}w^ZF+Th%s*4gWh$4sAdK*p{*BMLVT%JfirQF^{;tD__>1!XDMl^>@-ME+KF1ilU53w;1AJsMZG29y7Q<ez*2wYU7Y_q=oX3;?k=ym~e6cv>QTWYE~OB2UkrH@}gH?cYfJh;*kK*HubZnXN+4pTJx?fu;LsQ|!I=sfLyteizggd7WlRAouQxW}<w{7~vY!SxmA8k#I2KU73Fvd_lv9<ih#UQEy~xwa<tP+hnbZJ}^Ix!yS`LrFGNM-E;((W))t64M<EX5!&4h&!C%74lL`zCV@kT2S}*xO_`!$+NCMbph)^zHWZ#thXA4zo;k7D0Td*o^FJRYICjxN0@U6<Ulv3Q2kU-ByO4rm*4&m{yxs-8w>JGYwiJg@bFR%CXNZ|H?RA<94rv!Q@49tG~hL;Kz|0GemkoFLN}7+)7mAAW~D{A^VYB)JVAV|v;dm~JJ{04{mo;JtoaAn1>Kl(5wy2mPL+&4E9XR7y(=<BWpytYkhEj=6I7ve1WhQ4Ig`y6dwWyuIk?2VQh{ZoeqE<~j*xq^5;5Ej1I+4!Z=4e0{!m(4kBh#q#Kc-bTk);~+r`jRr&gGM^+>RI>f^^#Am7CrfvPAKsHq)~!pSKWj6L^GghHW#YbiH3-fAdzzE|+HgUTzP2F+`sDK`=xG#Xo9ZA!unWrb8%S=nD$csv|zuK+4;j3$VAzS%+W{3>ZgY@{Tx2eg;Es&#lg?@FDe1?`vnkYqXag9O0v`H5o(BW+7(;)&-Reu&vSi{pdTx~WhxkcR|h4N<5i+O|7D$42MHXHURimMFx&j>W$~t`W*Zs<-h}_3u0a_gqoGGBkl3Xg@buTt6s9Ei>+A+#}iYyK0o?i(JOu8#3kv&S78L_#E74bqWh*hJ8<{c`?Unt5=HD9ZHJzPb!whOvdg9#Qy`r4rGQLNy${-z6+czI&nr8|AZlmIZ+F_49fT3)hL(|+_{#}JR@}t$(9HG4S9en29Fo$_N48~=zHP(QnSZS19XG&rC+?p+0b|ZTZ8U7)UOK<LF_&$R9e~%FZv!&FfLyu;LsJ1c80FrhVXbH8DaPUc-BI0rJiD4C>fFfb2PoiBG&<6gRTRx6lHGTaK-o5+<N`61EGZ1x-dN+wX5O7MlRWBDrG;JNpa&|Q><SiT=^ht9qx9((q}*p6@2|_xHXKE(SpDiL6GrH@t%?^bkk@g6WnL|Y9d`jlm!tNJ)3LZHu;m~X5sP&V}n#fn219j9qc_MxG7YxjwNTI*7=dDi3@Z`azC@JayZJ$8{<;a>vM}&R)O{%_zXAvQg5cULQO2Ip+EEXkZjJ2f^0ys=&reqqw5JEYLSn;^h0CM+1vbIYlaGdThwehiC4JwLFXp)@tV#w;B$Iy>ZrVeete4^hMA^EE)s1pP8L@37!0c5Ha8}!->hU45)4_N@~Zd9!kig~sW&JMwA&!<As(>8(0@1e*HBoLSQyx}FN#(TdodfkJ(r*=-aD0$^Z~tK69ghRWIYIsyY2dEhI%FsefnfOx4GsmvVUc$odzR%eyhg9aE-k>in6Rfn#@w;TKt<C&=0DWB;^!m^V|Ng{nCK7RE_}LQ!<Ib267R-Ibh7t{n}opOd}Ga6X9J-$hGDx6IgD6CYHkuQU@KoF}g5B4C-6;qCt$V!I|k2?anNvc*BG)=X;>R#G%YTVdL-AUmB`a{;lV2Jxb?~?@@l(s`xlpQ}=L2)!gpQPRfWmvJJ%1Il!_LQ3fPdiT(pf%GcwL*j7Qv39@>)nkaJT-rjRxmr?xM)|oFKMBl(Z+_{DAIuc4I(cLB`AD*4)SXesEfwiE*z31Ir@b+jdtknl#282D54*wI+6Y?gzn@)U=ix%xi`N%gtGk}X{Q<^ZlPPe(te$$kqus2hz$60P|e&8f&X5gLqf)rE8CT>qOv0u_jNu}8)H0&wTrF-(O?}U&=1Uzu0^|Uk@p3<QyXFG$f0E(0FpgD3;JqML8sY5?=qZnQEnF;3al)G<4llV+yp)-JZ5#D#X758w(EunfrX627&Z60^8EJ(uG=xRM8H(dhk<#gOVo)0xIZtBMG-VHz4JihSO7C5%|RAmN35x`_?@B##3b%LEN!r|}N{tG0wGe_on^^U_pcP4g4j1A+s&@w$&!s%bnEni+|%?*D#s>0&~0*XSA{hbDEMiU>Cn%?ebF)(4^t7PzrCTy{fB5=NY<9PPv-N(fh_>B61+sq<ONAe%-4w$@FP3b%4$E=V{A_T+n-g+EgZFquQl=C5(-i`&V(c!(9u9q*eo(G<Ez5MNRKoVh#CR+2QT$N(Roy%ms2`}b`sm_P<(!u<&88+yBnwbZWiBlar4FY5xeTm`OR_e8?eCqE8h3L?MLkWW1hO7FhCvipmkhz2kUS$=I+Ioy+hKU?t6v7Sm`73ZqSkLel$Ab)*(+V)vrAasPx&PY5F*hsZ3!i7sKQO|?vCz@tJgl_Fj%C-GI%Q?zZ5hIsbd)!Ih!_l{Cu!^0Q92#|7}{N~xkJl0Z?z)^h}S!@s2neqKb=wlvt-i2zrJaX>13ES%biV`(r>x8p~L56L{UA9^~H0!>_<nhie~it+^5)!ft;?(!?YZ(o}hMIu?S$x3GEEFG=%#_$apLMu1oCKR*)(AI}8E21}WrO=9}Sg53iVW^G8pHeC;28Tkx&^h`{+zOg7POKSTRq5pBY{a;=V(e_7D3!!n;_@gR<iF}1l-?A#54b?Ijh*B}29CQhaH{8DWkfpU1O_It{HMf3?)Qa%G&<96G>^gf&Te|n_(mD~=_ImE{FFFVp_R#{q<(ZU8NdY)XXO)9zK5V+WpN9cBHI$F<|Ru5cJsg_x?J+jDNMPWN#o104YfNBz6o=Yk4PsV2uawQV}K@5Jk`A}u70f=y(al0M5$S}>L+|FBQaYUM?EdJ?h>bOHzgCHixo!MalME;+jm*RH{A`xZHG!B+L4jr}sDSD=g0hOfR?2;+re9qz38=DR>TRg0i?!INbCr0=!jj9%}qX~@Qe-ocr?1H(>?#ADEW$GCY|6V3L5uY3xRx1}{#x*7&H(eU0iY*K-7v_zyW4E8S7l7<TIyU#nr{PFs3)QrvFoi-bX^kDbuH|_1K1eg`b^ZX}$}~X@x0@BMsJ)8Uyz7C`j<!HWD9&+M1560B1=MGkd5O;J>b6l_9G0&3g7NhMTV-CF+q0~uX0cI*1*q*UIssKEzID-weyIkq=>siaE}D+ihO=wCjiJY~4+^p=5|k=Q=XRi*T@aJP@h>Ob1B%G!%P3ZayQR}!#`~fDO2mzGBz3TmM{&jn!I1YAB4!wCaUeEz@pr;y2Une|{dC_r<ATN64_7$8_#&S~AE;zuECgi4+q$P;VdOKfX+0cB-m+%f>kkj3Y&96=*b%JsE$zw_4DFX!unvAyhYrR}<YR+}MN|>M@_%#MeE?e*rUGd&`)w5u2o}T>k<gHx09wR;+RiiqoTE;=5yK$(4uesq>Bx}D2^=CZma@7Jy9Pv4v~bpigLGG@)gFm9{ouKa4Gf*?Ge%Gzm(~nPlCiEIsP57M3h$Lz5N!U{0#$hby3=cT5i}*#CuVcfdsNSA^kx}mmDiO|0?K2r!xzI}-oj^XZSS^$b^RYJt^gC27LOAQn_C0(iT!58A}<{I4h}EpRxG97_?a|0L#>jsDLvMqIefaa7jm4PS(G)uZ0)*Cup7xE{+<`aY8Fwk!Dx-MPyXXs-tIKpZeqeO4^?bnv{2GQ=Bht{u|2jFDntk%Qo!4qehjH=j)xDpFL_lua@w7kwi(W#A{W*uiU*7|o$WL@=Ja4X!_C&!HLsHq+7pqe=}JmW2uEq13W3qNz^+cdTg2iWR_ZAo3smv{yHRVnYj^}J4MrUwnoNdHg(@Db`5N%g;y5lajsh8d0j*;n6%;>{4cDa)z#u4We^6)*=STNBcDRLLf@f&~G{j!OS=gYB;MEt*a_5fL`3l?M+g7N0Kt8UasDmfcqHVbGRvcwUo~yLV!MIhfz9v^Xv{?%$<uMBVPiWCmdzntkNGM~L@X!5D+Z^O4?Byvl1KKS~vYrZt*<RP&Kx2dy2cjk~&vKe8-hFD0ShWjGhI2HCsRQ@e*Ew8@WH<stii6oG;%IgZg?f1EktI&9Y>|U*`BIJe2u&A6QgEWX-al2pL+@26hJL$ptRgR2<(PBp>t$1-m6#FSh@9EP=hdv4)$KL5!=r~#+IpNZOx7mZOe$m+;oL~OzcuYa$9f4DuVeJ{oQVG2G{ZX{eJvu{a0hb)OLb9ZyVVx{WHJ)-KAL*86{?*R4^dY+33G0I=reyU!~&@0phr84Q>@3WTCk+X)mEP<do}(Z$%z(|q+5aP^exv3fxW2McV%x*O|Y9CuL*&qJlGmPxUkMw#vb*w0TI|q2?j4-GkL}o8QDfNXcy^FLl}UbEr2|z!cn1gj9MiJGA?mAlz5T_xT>o{itmd<6r|<4ko%l=;cxctBE2{L0G*flOJVjbmrrZ@_zV)P7siUIJIRV@B;YJP>7ifKPgec{!e8OORpJGtudt-ccT$QYm?_c0w1Fc-^YR7t^*Rt0Y8Cc{k@z~@2QGfsAVAobWERp)V@mY?t6No6O#u_Hg!O8DxeI!d4dYdLCOP-G5v(fXELGDYVke94;&uGXkEhdNq&6v#oyLfMF-q|>Jt=U37<>#7@EWgFNUURV?PYitZ6mbdKTgB1b&1g3)ANG&T`iDI8)u)$aBcMqq`)ZeRn$wM&no7)tK8XDg%4dKMslix!Q1(2%X`U`t6Uo#gSZ`faD|<oA47SH>_qlW!K?r>z+GDm@~WTktHlIl)*o$^G`kb)xuu-JGJlh%wBH2LGngb*o`MKK-7zrVM?{Mlsg6y7zLZ^e+N&{NIEpN%c~z9<37_hK1L9%Q{5!pmgoy|WE}^HU$F0N>o&>2eB#**Wd-&O7&tcn!b0XM?F!@@cb2fb4U+OR042RIEW7gZ6OGaA(P=+SwJauD8-u5(>Knh=*-;w=E`JG?_+x^Lxc*0+)>W2{{?g((dT0~F9Pbd>U-k+;Xf1ft&U>R2!(JKm*n8XiR8{E-|GCl-y)&oeo)pe2HbF*aYSXR_?8~x}uD;wJyT4=<w-GlJd8a4>fl&Dvy-6Ht9B8s5e`=BOeFQN=tI_d<i7yms{5$>v)P#4>b*-pjlR>fiaT9FzBhWmusIez;_P)mH}!W~DO;mm7W*<Yy0h1Z|>Xrb-m2yHR-es`SMumC@VmQTlLK=^pJ2tX8rdMaikC7013-y21iTRHA_f22jHH^C(sBt_*s#spMSZAU008+lZNg2*k1q<;{1EBzk>X?L~*YyWoTD9<N@5K-tA>(J~+g=t)*1@wK0UG>`<eurTSt#-$2v}y6{ze}R<$Qd&`6^_pTdTj4`N7&Ofb#(*!uB(Z$8Ozj*>3R_u|E)`SX-76rV~OUKN%qT8FNhoOC;9-xl=G+#^II`r9Ry<&kwU2|8#Nm3pBFXA>hYP^RQC0uHkT%M3wifd9!gl;aqJFLF0!|~ko!#gb%i-%eCBLk$lxSzwGL*qIATPdtJYYoH&cU2Ku#d4mfp5sw<Qpcb}*v=PtmCXKA_lFsIleP8}3IZbaD<)ns7g3n{-t46+Z@0ayQIfSrS@O^D%S7cDphq^LW>^bt@!4C&uhAtR@oFd2bo~oE{KF3ex_(jwTG8p4mp}WMuZ5Y_<&)ltB@|H91Mfm2Qlcv---a;cK3R+TWLE@MG|*V^odGSYF57KAIfWjnx`g+`j7E!2DFk>>Pamg@vUC%!VRHb$LrdRr_?a$d|NUpNPZun1I~JX`-h3ds>2=GaD9IBVu<jbW;K==UzX54k)e`w~=O+7BWTw%^s?n4Wq?)T95v$HY)0f-ls^d1T8;#b_d9=<xtnd0Y0kkZxXN9p}S=wg2;c1rMxIUzGmmMV}jZ1YpX7eiIV+}Z7#1t>qo`M&LJ;sKryLmf%qaytu7Q=4WNk}7hGqbyC9JHL!>w6{I4)lh!7@My->gVbH>r$giRMWdT#6X7FUHo#^+rO^C;akro8Y*7X}CdTOQLTSi(E(E5W8Ynt>phayD3?Py9@FW1o&K-?|<|PA)eRi7|l(Vq5I8UI8V&f!`*Rz}TK8SECF$R!9l=P<{5y9$92fp>mw1Y74>-j#XS(lVbibs))f}d4uhAO4tc!HLaCDq-P}w(JaqKJ&eLlsPSg-@dD{Q;vA^fFh<=;<%Z|q#iFY9c*hyCg4XK+u+*DGKo-rp?2@tQ*nYrE^KfZmA*ut@4Y}%rTr|4r--Rg1(*01eCTHdYMf3pWg6pIeCT7oI@VZrF$D*<p5)5tR;(QOMzttARiJK{k))A7|o66jW$%J+k*9%NC5NJBUW)1VKfYm%hXl+PcF<yf1OlGdMpn7WPX>JJ|@S%))mUED`_}BP_i9q%?-cQ8XM6<nBhQHB<@><%(!U=piJVLjn!39$T!g`<4z)eUxg~Oq}m;ucDg92&J#ij{5%8f1b?@D8kY@AGbCL!6lG|0z)akDZ}HMk^>G4w#Jk0tmr30JN0Pr&K<IaiBLwQ4Lfm|);OF8FY}V5>qNdF4GDkb?Knig3Z<X~b&*C0mb_xEL~U!3ZMgG$=P^_pNaU2L2b`mfx^i<H!71H9Rh=q8m^nWtB9Q_e;CS<@hZ?A)t^Fj2i)h&Uqollz*OO=&ht?iGM1<Apw@-1g-Ywibi3izzz%^HQ;5u<8nJ4aWMY}I<HL;SL2i>$=0(fj3K2zh#Cj}Iu&nR)br?JRyu_tD26<z`z9P4;{nUQ%_2vv!J`9T@|D&qn~tbv?P`G!K(f#-O6!S7ytk_E4Re)$fp8IHCI*%ugN=kinIY!bEpqM`h7VpJqRK$LsD2+YtSwZCTSyI_>GY|<UAZ#Vr^7;m?gEl_bJhPvm9#-PkA<M<yK|_`2tZSfjmdCy5GMOJ!ny7~)iLMMz5UgEfN9h>8}t~(CsL{;OK6T+@6(!EpCpotiW&kFUr7f}r&}7Zci@Ixui#`TF8$O4_WHI{=bQ(>YPU1@=91&tM@rK(n=rG1)=vBS-k(L|pQFYEf7X;GHK2`>Ox0-W8A5yi4s!C1;z-x*(mkOKjW`Qnr*!(2KvIS9He*T=4W=_anB$%1ua+S;pepaGP=7jrqkocv+2~hPQnr{^9HGZZD03#?ue_NInDacNJ=qvxDYX9wzH3tr*QG}euM=m~Z(XspoVDp?`5z&qOJeMHNr@&b{>D~#akoEFurVAo&Lp9Urf#zZ#G;@~K9c^K05kI430Yee4Kce~h;CX-tLmhLj`(VnV=N?WRaAbCv>$?Fp_<}&sW4yK!alOo?dDT2guyHXg4+3<I{~k=2U^4!brf3DNUr9OZ>Q>TlK|Qw4^J62st@$B?%aF?w>CcQvq#t;r&3>8v)WGqm)7FJL0j(hW?N(Y@Zxj7Cs#j|b>B$;Jp*&fj5QV_YKWQyX_vIfDg-!dp#!(Ap;l{4TM>4=cPMOC{gdY;L4ONLouVYNS_Ixi8y`AITfSe`6XpXckRZ}y%W!xP8}V?Iy6q}7lo`m9dxo}O?af#C<EwamlsvqYSNa5yI`5-M*hrv9VfO=slB$ti{a(aODJ`hC+B~LA=2gq)1<t{ZN60SGz+<;klUW&i#*5qg=YyX~e{X@aTs>tnxo!nx+sGHgtJfS(ZkuyB$l^pfb@Z`ozgo|&{2|Zp`?x6*AiZ&dUSPq}?E)x$Zx!g^7;t%dX)tUyidw;(c(TCwM^PMf@Td$R$ri8c7IOA_J+N)(Jf4?{BAmYSY026UkOGt_X=m!b%@sWkNr`o#`z9!X@RS>&w2lRpgLNUgqPYsPDMC+vbw(z%*HVrsb!>8xCtd0w@$xTE$R1R5)DuZ#Hwqu!iL`pZPhRa!6pTt{eN_E@Den7ib%v$BP*wJ0=FJrqXm4rNz$FH_w{Oyb2E=LkF6<JsPny2sVAQt#+PV=Mb2XvXXstXOl~aur2=nZ+46VdCNE6G2$-jK0F->>YUTP0LT(QehlgU6k?{&8nnu9konx+>Px^WgopW?~N(f=^*BNwvsr|Z+)p2Co05yy3X4!olg^8e)`x$?^<%Wg2)ZlimiX#5CRO&#mtH5hWF$&1C9Wb@7uufwu!Or<faHX7UnxToKE_1C&YDl3@^56t^Tnx$sJN~KU!*B<eM*P1o<c)Vq$tWTQ|jupt83O&XKPG4dPb~{aoWj^}yu`?5H|J2-rGcV77pL1l?we$I)i0Uy2Y8onu%H$6IJU^;Qhk<D2U;GnMS&_!_#Le-m#!xV$>Fwx;CrBDA`)C>l78vLs2L)qlF3NjJeSxJ%Yu%xC$|qKwF2z*;Txc-1r#cn6!#bfuU(2f0sy5-vX;>Kz1n}pyfg65q`V|izp{{xJ5)F+fTs$p6IZ}V){(Bs6q_;U|FpLXp;y0mCa+Wpbw1JE%8t6}xyt^Zll|1O0e-u(oEEIqI7=cfbY$3^>RVcPFT-9G3y-rVGCT7qE%e#Ql!J}G#{_<CCe<)!n+tY$9Ct2OK{EAs#vF_cfqJ^f{&Hs~uG#Jsjy_gCl+%%b(XMjcsF=zYeOZvo#lbyME`<Ed+K{!396$QjSrOhz%WEnr;snkLTqBy*Ni5;V&q{oR#I&T1PoL!#fh56lxr*B=AxXbWULijk8%4v>H#Y_aXg_MWz@dFD*1pFj+8=%UVE7`m0tMqf=;S!!(QiF-2=`b?_BepckaB^U%`T1l`wXQ!|1YUbpGxJgu15h3jX?GH<LxIn%+taT3xuu6`vb|J~z8`h)PILZ}>#(R<JjxD*(4LJ;g$JkSH7*h;t(D(gTLLcz`FWqZvz4!Ps|4c|LmB2a^~wV-X(%JP!djtEI8`iTEHyf3cT<S26NtlBunRMI7}Z^$PH!-H{ScSz0@r$bN|73|%*7Xbi_Tk3WpfYLORDxU3zOtiYCc%pUtOvqS-<l9RCuE%{Wt_7lz*RIp|G9qM4d)O%A2H$j(gtwLSO}_roA5l)n>=Xc+&@$#JY;YHL9)!osINI<_+v8OeDT55Sw_cua%v<rTxGL$uye7HSIBKX@;HuoI4aAL+Sy`EGZKF;UUJ)o88MLx%#+vvl_&0yu;N$%VI2Xa0*VRq^V|MpW@MNS<weYUmr(PVjNRI7+vFB>T^m6PB|^jC}(y>9=SKnm8E^ZS_d~;n5?(6Cd@Th2T~R4J34x`+nDfB4}38~|7bz=aX78X>l?23_JcP*=jmw;fFe(t9D25V9HyvNiSwC6TxgP=8uscrLJ0FcFbx?FRGM)}WqbvMlt52DHSnVKmmMJiXh6{w2>nc}mN2?_UARTB=MMpxJGmA$!C2z-TSrHED&rsr(S*HsheSx*7I&T?u_kku_(JBC-8LZUUz{WQx8yb6f3xnZ+V(e{Pf=GI!Y~uo^Ex<F<=V{_Q_<62rTSDmQ13tWkFvz*!;t`TvrzAmx_?o?Cs2G1w<O3mW9I~eiSWacNu!0(AEvzo;vzFgX*LoPE8qE}0f@hjW;?!tCa%3CGvFKUHOTxOvHANWacU<ll`iF=ql{n_?1gHbV*+oFamwPA%gN7s3zoSC+^x^<_1i4-9__&!F*dbKdb~-GB=yAJen~?mclGc=G*9nS0?DwT@Po3g7dlvsqNIPAn(2<?=@NlOgaJ-T*N9H^aZ`g2Xn|8=7-2;PZsZDzmMq!Z@w@7UrCM>idzXgIEi<hvg`(2H-6)Tj!C|Hp2yj>SwDdmB)<a7?T+~a*HDPfyqXP>bk|or<E#z*gJ{Gz>HOpRLGaNy_%aTW%Z4(6H4z!+v?lHc0ko~LzouE15-=Yu@I<Qxe?`I#yusFfhzktxwpdiA-7oUK|o6GTuOHJlFR%E_AL4`l4`J_*B?~HQqsN?A)ZY#{4Fk4Z}S0@FXev4mZ?~mnk0wT-ul7y~u5?=b;;?NLFpfkY(Y#$(O!I>R1WaEE8W)!*tlD3)dN(I~bYLn%`PH+clvp7cVJ$~Yj5Kot(G55txr3pS>949OWb@@ivPDdYqZKT{Z1b?z<wdKciSQiSYl&dbcLzhMd`7ddlmGLndf=W6hb8c!<k{TvipS`gees)+|x@`Gs#SkV)v^DeOWLlR-Cdk0G@>5Q69Y=rWzsrIRgztcysTuaGHdu+ABx^c3s2@G^Wv9?KrV%(Qu4<6@p;1IlC2*RV{xPwR)r-x&Y`APej5lX?`7GHy4YfD)EtB6?OJEK!axmK&bY&NqtT*KRXLa_8zBtc5>4*4`g+tXnV25egd%-nJLV!LF|2vw1=|$vh6O39utSZvdei*YtbN3f8_?KuEV7-ayg9=jro~ds9r-#UZxb$qm&E{H{Mu$c4-9mBqS=<1;J^xi$ZE_&!XVNV$i>^1<O|hDUjJs`7wJJp>xIe0mRF5MCsFceo=L0HLh)`kT_I`{S^xOjC+2HX=y5TUx@1)A@JDBH~bS#36It2%KGKgi)^M<kFf+3qi(%-cGJj1BBujGy9^ql>%70$<~oN4C~EIK5o{;9Ifgh_J0j4N9b#a_*M&~r9*_>wz_oQyAF8bHl-)YhVA6Q@IGY@3B(FBcNL_Fo*MM$NtUvdZw1c?zu2LF+JGIXpo9u*;c`X1o*Ani_Y$Z#u6`QU=eL9|q#g&&SQx6j}M7^BR=S7Y?r-0LL9FOBw2c=6;Fh#upeiyecAWd05ew51Z*0y<OHNBG_F4OrE3e@qJ}<U;`gn1^e=e$N{t0a1`8w15OMF_mph|j^d=tnyEorAy+Rg&yzCf$(;?1$y|9_rZkK`oxtImP$1xWLx3BDm!-zT)?yHbjYFxkHX!C;*-bNQk{Rdv&{b$mFAd;?Q`-XY`0=a!Ql@}lLHx?h<_MeG*(oSxCou+=5nek3$j)O7l@eE}R$c9K=@Om!Uz#vQw*5Kve>c!e$jY7$;lM8}f4hA~9^h^Yx$xe8w2wzS=5~wQODf32pla;u&7vh8xB%=0M=+yr@hDg_049SZ1Toy*8yv67?@O39LnvJMf-;I8-EXI#u>Y;F8RSU0oF6zcqOES@I)dOcaa(&?zTN;S!B<q_HT1<iGj1B(pCGR{z<<BJeHc`x@(XO(jRkcU`zKEck;Um85MR#?Ou1?HI?>%qK2g@oRiitNj-_R{6gx9^Z0YW~SF=mk>^XO6U&5rRyp0_)v3~CUPPv;{{EXkX8~8`7^Zd(fPpjFxlmf7>_e&8{z!B|rF8Lm34U?5W4XP2k!DX=!Wjb>+9y}VmzVB4Q&Lr~!vA+_gk|FHul5$%QwY^cQ>W>&Wv<Lju!<f#OAfFT4Mo~!)VN=`qC4Q5wWpC^nN4lS7geNK0gMAzk8|a|E%4D(2;fyID0XhFPG~9HXh)hhEv;ACVbYa{JSh@LUyopeX!>WavZ7Xe{ARu!)HB?aDZkmtUP)&_M5hWr0eNY(%dRCC;OBTB<2=Ag8L~Y0HWCc?r%z6p^gS=4%_zovMI{8;5uAhG9KOzGRos0~tQ$t3CE2?0yySpnmFC30A3q34#Y4zhr4Ijf!_k3QFalgVQv1twII6&=b$ZSG=c|qQM;l03sw(V9pmBnuxvSL0@M5VNPxX8-D1*)Nf5Za!Qc!Gl?02;m;fdJvL<NEoF!qxL7be5ULLz<zEVu*GdSUBTq{mJ7BjXTRmA*&*FVo1Qx^}_r=blgPv?V<FmWi_J+nvUGOEU$!rPZpOI&MaxT6$8h?>PlX<3KoCKJd;vWrF&89mDEgg>SX(>v8>pf$k=Vf-}ZOy!2DX}HV?)}#hWH&$yX}KC-b#7`pU?JeE^lpUm4mny2I_z@3QXmIj`G2I6m<gpjC%P>RRx0vYo=yuy3}l<}+QRZ|S@>xAs9INVA0<k{nqTH`#8&&5!268}57QxjCqOz0tw1lIsnD#>C5yWfiMyhN0XG!^Y;?U7RZ3BIFvVwQs`X&)Ld!cZSs!t}^(3$gM0fsG2poMgO+<I%NB2>ufabOzzLr)yB)pU>>@$B8>7;s>6Z0sOaQ)^src@W}7q_h}V(tF<8nEB5-)~{}we|Q}K`p;+%_TcsT{ue7LBID=HQ;hceVe#CBgL-=q?s0%poqIeTFbtbm;oE3aXh(}Gen1VMM0Z*I!*{{_kW;pk%q;kO}#5P)kD;vUMS)4rS&rNYm)qT0DD%!;yRZgBn}@Y529Mi~5wiV`HmvXyHsg5Qe6T>GX&?fw~hDjcY?>y@oH`db&1&yC=}Z2XVv0fe{owwC#WJISW3L5bMebYv561@7@^g3(XQE|dK&h(}TDrN#-@mpi!irfBY5^Z75J{{&x`<f*eXk0gEOb`Qb{X@c(nRy?^A>rl4J@8G~EY6Dc8I+``bjNrymR@-U&_sxgm0L%&x#~~gvaY}|Pk+HD$j}K6}TxaL7Nrl+;B5Z!tQbl?-ErgAwSofzy4!?3sM}+`5&!%ue-r?%+fxwgFawuEB739`j|6@ROTHWE@@p(mNZD!THh_@=PHZToaVa`g<+n>h@o}f?`nW47hQF}M_<eVfee-oqM^`H^vK=RgZ(lm;TcEI-j1*qQiYI;4J^exOf%)!l$#z|#A_=qo$4B|E(f&WNvoDp4Bc8`-=UN~9JM?v9TSI*cNsaH{Z$5+MMT2t(N5CxLxSC(!Xtr{t}u&vQM$ag(Zyf8gRU556N4M@_n70Ml+rMbb?5+UWZl;=8IHj<r7Jq**bMF@K}CAzfV+%emh+2)K~Hj!)ex$kL)gEoP5xTdylg^=c2ZUl6L3XK2&@e59V0vw{;DTTeTcQ!+nQPN*p=_LHr01n`D*GmUoVfrPmflb6Eoc|nt54%&u<Ol04Oz)M}%J(P+JGdF)C*$MX;Sz^)GC1I>H(I7?mzL&AwyKHvmgRjw!&Y>7{nwSLc7r<QfD5VyyRztKv=c&QN}xTXbk0S#Ynks!WT>GR8$J+hwQn9N4)=0O`~9A~q(;v~K)&~hfbrdq{zs<-8x>a3u{-b%Rb5nN$w>CRB@Xrmtc$Z5auq4qn}gTsNiYB3NzQ%{bkt8);_!VEkiwMO?1i7mz=uzb=H_}{QRu8vy&#%wQF&K8W5Bh-D|>#BjC^R@HI+voI<`Q?{b=3oWNb06c^V7I>+owbQQ!&c+a0n>w*1K%scYhb`IT&r?-KO+nvG$2DqK=U0dMfe(2uLtqvxqZ*KDB&WWC^}BZLcfub1^Uc(T-(Mz8p<aClk&Oes2GoeV`+!{zFX0?iY%M`6pDwu>PifE~@$_>JD6n!)z5&4o>=EV)aGJe1xKu0`)1OmHbsKn2KDKZOzpOW?M^RL|d|+3;o?cM{%uA&==2{QGJ1oyk5&#EDF)KFh&cEu7b!H}b}Rp9R6nYi0Rnb|I#<K?`HX_fmsG@T0xk0-;mDdhs#?_N&LM>WLhlY>~X2b4T=&1fCO6?#sqBX9~$AXZfwIhSLfRh%qpcSI{}mwky2K37WGajT-kRYwJL4MUfueqUOK7CbXNjVH2ddUb(oA`<r_n;LjKaYrjtJ#qprsb2`ELE<GJEl*i$En4w>o93=XM3ibpCz&W*kq7`{6g6U0HuN(K`A};A)MpxS=tU~^a(~e-H_`ct*G*)I*#Ko`4vyy`BzHH0E^kN93OA{TRy1VQIz1RuDwyRseD>P&D!g4VUAQtNrWXvS?!-CM2?#qP<<r2wel0^|4ftr$WE?#2y_w}xG<EfP|-Ch|u093cdi3D)-In9)T_8q7ekd6)B-;*6Rr@n8}h5d)o0GstDnh_oaLU?=_nX16tu3ad4N&N&o@3I`=V~JD#qjCN56Cw%xA2tG(szlXVc@}TqF-_?K7WGVfoGTzO@O1jh=i0({+YwG51hBB18gwZjWs74!wfI>+n0R-i7r}x)Vp{Vn*i6~U6NlffZq$76G0+hkMFZr0FK2N~Mwm(}!)g6hxW+&4(m;(2B1W7G+<^MciD|*4ms#w+t(xj(NsHjrb^}PbFZ|V-HU8D*h-uP4Queup2sv|s0}ms_Y7h^McsHBkZr1nbQ6lT&iZ+7I9URNhwZvQ3@(OMeAqNq6uYX2X<HH!rA$Kcd6R_gh8>W!@A3NFZd8*AHC&-Cf&x^&aMO-}1eq+-d0zM-gAu17xiCWWN`d^&-qEJ?WS+R~TfbVEG*}l|}_>n5W2C$%9H`27UbsWtn;7fyqSM!`JVpIJVz{w9*7S*w_+449Z9LLV`CBJ-bDjGV!x(V?y2fE?4>j@93ady4VDWh=cJ>9}+yBXRZp`JiyB1G+SFeamCJ`g~OQ4tqFmZ~GBB;<;u%&9#eaGxAVJ+&3UK7qsnRCCQWg7{t~BKDwC@0@lDi8Snw^VQ+rbw4b*n@+d?-^JN6^Nmw==?-8$Y57k&P*{rx{Lm>B4jr<mpv!Xy4h_|2bA1!UgybPkQy4AF-LdM8p;9L5Kp-k|{i9-UH#MoK?2I_Aowz)G+1tXb&<+I>xv(uHY;5#azyAR(#{>UK>$P`)sOG+77vcTwM?{^{l0YlOk8!`6_V{JxXg#lyMs$74tZ%VQO%|Yw$rura;>oi#l|URnPE3%VNMPH@dGIM0ocK5ngSViBzUrj?)aJ35SrK@{F=v*zRwFU%1@Ci5mJ_KZobJhVkUjLLAusERTC}7znw>((e5<+@zZ(JZ?Ae0`peALx+5ha6Z3p$)U8S+8h8ZgJZ6vBgcR-+tLE^l(bQON9y{83wsh5%~8wCqtYyN|Z#V7-@F|E6q?3eH;WuOVOAzqyQr-2kRgd(B@pt-Uj+{_0RmkGwg_r?)b!2PQ0;gzQ0=cy`=DAJLxUN-X_SUocsIF9tk56z!5GP-|*={Tn(cyNV~ut!FMbuUKDcNU*Q+b+iRa-ZbJQ3vAA!H21NJBafbKr!zRi4lEYZ_Q9p1bUvRU8)3|R=PU`{B+||-sXh_Zvy9t_i5)uzl!t{M8tB<YA(E(OB=k@r*iUt1FUv!h-xXdYj3EpeMZ$pO%TEyo#>Q^KE$w2DEKa+g|pO~JpWUBxnB!^%dWQON&LS!nEON-ve8sr*eX0xi6m2wMoeuZwuHG?=k8c;!*$8j@4E6*f@?oTM|0Nf?wOclY>VDgrSEBdhP$y`(bB_jlPiSb7TX(V|GMa1ZIL4KQlF#$h%9M!6V^mIEewL3ue{?6+qQooU-(>Ag2;S+I>PhacjH`payLerEckTTYDxlC=Es-b2hAn16|*BbZ+b@HNf_GVs-$<d)Jp*dB83{qN85v&ArW6pq2^7NYc_uBbeTvLuM-tH=+>63boc6m@}3}17Ey9f2mb%Nd0?qJVza+Yqmml_RydP<T{W>@Q@|&{XV4S2k$r$~2Iv`k3ov{)ffK|n?m)s}Mm#^ED9{3_JT0nr1Pr%fQIxn7On|v|lnUyn=tc1GyL^-L;&&^*Jc7&cQ_Lllh)6-%3o@|3yF_4%I-9(QrN;jUok65pp(DrSoLD%^LXAHCGL(#{&F*@5=Lo{Bd!an-_lAtT6c~oMN3)v;&;3ro*Ei0|C)~hKUrjW&EzI+%jtQzQb;HSwPy=Dp_VrpFLeoG#WD84m9KK<}0`fciWO?05^I)evX-X(nq2_-X-cVcgqNbEjP4=x#iXSuP*Z!LKL(6OtXwYXeen^Wkvk$s6dC+<K82);s-Ql5jB%@p}{=~s&i?25$1oB2`2I|m5B8|(kAN4(YhA>q^H8H5fhIoLNnHnyG>pc?5|I~1QVn_wdv1Bsba;kRFWT!&DG0C+#W&=9Wlb&gyM>!XbF(|@>1g5@R{nL_K<Cr>LmQ6NKU(Qjzco;~W`CNnIuyW7|IPKs;Wc$2;tG6GT;iePT_bpx-TwQR8J|30Mg4Bn}tAe@RSWLq;`G<ZQ2~U&v=Y1JRW6mGwC0DVF<9G*9(=;hq^JrM$3QJE6??)7rjY<aVz!KuEIHZzEABKnwwo26{UF|#~{7_LC!yKTr7>rhYNulKX*ufp_*8|RjE|3rJncuNqxN&mx7o*<A0((2eoGOn~A$(ueC3PPPmJ$dRO#L2c3A*IrkEdEx^PF3EFd~rpT>qu8V&43XQ)4Ha<KcaSX{fX<_C=sG7B<muJPL1NbaXZ)zw=<8s-v<_VuJejvW(_k3#t~$ChF>rRnScWPpx#|=t3J-4zH1&?zKy`+!F)M3_Om4sdzm7-VKJ#|LU#TGe+qb-JIoIgzG|?$iJcA1QYPg*$#zsjgQ#|@?$!kmTT{LAa{>LP^T-(!kj;Kd!7)Qj9Y0w6ieue&lrUxzx_dQif(y<4Rt#{aP;K8#y1CFDJ2qw#v^|2=$_=bFlB?Y32FgdkN^axtItYQ2Mom1*s+N9rd6D$2sPu%$pT*O`1;4bIE2vzLsGl0PrJuw*OQ9y8QhW&3iNIlGBo}x9Ha6CjdO5*l7CGz4bzQS8y1Ey!}6WDITcuH2R)E7Lt<dZzJe<V>i>6v?-Js^>R&Socp&RnqRyivwKM8f^6m9(`fATB-fTu`YP#nz4Cka2vyTEia*}WG_~g60VSQ)z8k}JhjO6Q-RjI9v-tn}E5Y~jxlNdb<&Vqcu>T_uVLwsl<2?r*|?jnrqWFeUXZaxcE5gSy4*dHYzWFhwD$4a9XHa@Ya4FtxJMW~cy_)j<0Sd6i7sejlqyE9-IgpH|S@o1O*ZUI$xOhPOhx}Y0E{fuT|Cqd?7su8+(Tivqv$7gZ>z){6bR?$c$@&%t)R0@d2iOsLrA>#Kv74f;^g`l^tuTb`F^g6SGiPZ5d@e{ye2~Kxb6XJDfXr2hC5_Aa))Rk?w%~Qko*R<p0MzO<u&8CC9-fJ{y`V6RhViK%HMsa@}ZCm(9g8(OBWyJEKTST!*B=o2k9Si<=w!Si`gu_!$FS92_A@}@ZLJw$(>oWlWsf0s%dD9so9X@!GUy)U0qF?$lUU<zJPh5!NzTv0+MbpDiFU-3HBTozF?d#^|8e7N|G?!K}T^EGxp*v(+&cLs`0v`)+aG}}n$G{VF$j7Zn7Cz?@hOHE?nnads5x=E1%%^`c%JnMPe*lVwxq*fgR}NfZPTzwXaK5_Tx$Ianb1iuhz`EE%2x`d&S8=$R7?S<Zv1_h)rbbCL(H$LQotP<Pg6+{LJqc9v?+D@Nv9Z-i@VecXCeBocn`GwCAYVoew)Ss2PHR(yC?vUl=Tp<^Oe_HnZjb+_KxnS1LtLa$X|j6)4mu5fpa1f+=Qv%x7k5$XWb(ZWBELx}cQUiE@-QeXz8ek638rK$Eh7>LGi9b;&Edib+|`yRbql_Es$)dzZYX6%Y_jpCL66eF4g7>p2xGE<&@R}hiP4{+v`j1|k13-89iIeuy|wS#l`Ss7D$ri|(L{><zoWF<VOrZ54oR+Vs3xkgEu}2EqbwS0M*EA|&NKcFa*%Q#jyTmXE#o_i!1pqgV@f1TLN3Dno6}eNI|Cg~YxBNQF~X0IZ&YKh6yP)*B0~dQ!tnbot^Nl8DX`SrrGvgBrD;l5Z}pde%TP-;D-(V!zv7w*xEXt|g14R1KhQa6Oje8&^kJ+O=h8Pli;c}S16+^7mj2mf>Mj>d4Z%0_UIvl~`g~+??w}0i^&|Oz4DkSkg`j<5F{GCsU#i)-Wp<l0JM^zNjPpCLqknTAiH#wcJ}VD~o^s>07p#aviI07|DF_7jsZ&Hlm1n%Q0MaPd;@-yHMJSvi84!*=3#N`-K-HPg-xryuoZ-N+AL&_sRr?~?vX|jU-UU+8z5Oja3y&=PTIWWkEfP{L4M@9zF>=`3`lMqjAhojb=~iE2R`(~N(Bambfn{?@dx;lpqwGt=y=8eTphZpu5{R1^ZHGM5N<wDq;)cv)mp3yLw%ew^wPz525!5WTN)<jLuY?+4Z-XAS@@E=>WTTQ~YmUx{fj1^H{n4_!p`;>4yjB}T0-arPIdq;~%!qsHG?klvm>nNf80o^G5=nK8>^x_-59varPERxMd`7=Iv#6>lLLe)oD77`D)qWQ`81p)4xH`g-N5O0SErUiY@T<B~=;KGoj9`wwRdMWs#JIuFXbF5ben9YOLxHHs=mThLjtOi_q0845jltp3o{rJL)c9y&05QX?EA-9DK_9s=nFZx*+!oc;pAK--3nwGIhetH1p(pH?H9UVce$o;q8Cff^G3wZX#pt4~=L8fT_|~fUdwg_rO-xt5GWDw&TQa(#;hZQO2Kw2fsxt1KGq1>hu70gnt7>bxKSn)}QBFH--7N&s*NgT==pq3AMdXRll&Z}f_R1~nd0bx7(v@Q?YVJ3Pk`|>JDw$-jrpL--$^nasvTRh#a=q8k`?}qJCXq8K!{yV82>lny;SoQX_l-I%T$#l5xF5U!5}1BK-A`v2Vl^e^!w{l=Um0q6f8cfXH&Fw_MMlrwJ+JTsd@`x#l#T-~Lk!=mm5I#EG(@o^jv*GN7c(V~l;oqg3RTEndS|=!G6XgJ9G$oiOR-XtY1fgLa!GCS`lr<Fax{(k-V0JEe*GkeFW^@TuV2bd#43#w_+8CCRMg#@#5cuzmU6V2s-9TxaKiv7>)bO{AgQXghN4uwcS-yg|7#JFmy!m_BYp^XaB;@_t_S{{%q!PEoc<Wy(5*uEC^Qeb2}tf$MtVUva_AayTlmcyL!X1KA=#=L_JF!c928ZGO3(&B5OIh{>GTWEc47xD=>_xiQIuI{CG3dIb#YF;2k6joO@k=@W#2tlj?K$}G0q1$;9SciIMCMhPuU!--v%}V*6IU}E*9fZT|u>IVM#xn68ZkV1l%=c9wLBqod2UGQbjtVs)wagYo$j+lMq^j>9Ou%N;QC?x-v_3DFBv>wrxX7yGyObMETK`c>wfR=EbKo8N5hk*kc-x^dUW>ggzj@J)Z8qWj>$keclINP6bdhp<W%RGU4Gtw7RfCgNDV2Vu(({do|`KbL<X1JK!6N6a4cIKH8$K=IN}hfy~^PgPadQ2`3RcUg(n^jFE{wjQ|@anH$kfLiKYEbE9XQ`M6S1a5n?OB;OQ_033yNfcv932_yflY3~zy&}#*{*3vFd_{QPX83<XMqaGWVDl(-Wl@Hxs<7OJ*AOi7Wt<Wh|&ejbzbv$j)6E@jMA1U8;lrc+?`77Ko?zko$=>X&8huOp2*Vii8U~iM8Q{Gm2Yva-`Xw&p6gG_}`P!??G5X+1;fIMK+pd6I$mOAJB&)_ZlR0-BnZ>4_j(yf8?Knyfi>2{jMIV9L}{`t(WVcbJ#7M1ea_rG}_6-|G)Fz#78*|9uxpf4ZAmvdMt5r3cMY+R?*2z8>cgY*!>*S+zTKP3C%3t#g{b$|PvcLZ7X0{=K&UrH!%Ey~6t!<X1)WH?I4E)Pw^_aL=Ek-$O`LseDAir%M%0Zf0dk{&`5M)jo+B<jAxj@``{V+M@&8;N|4d+xGKb*G$}y+Kul7_(EajZQwKts9C3SIbU7sPRGe$FcwABfJBg!|0hift;Zf<5KhYQVS`Bh0WlsoQ)TTCOs!kHX0BZF0Bw>ARG+E>vwNjw-UB_YugiaMnrhBCxOGvD6NCOG||=5GIW}2A3L~uy+YQhtP%x6B-14OE>9%E{pj8Ock&z^<jsg{(2;>Z&E~hkgW1^NN&}9(&s$2pM_XT&zWr?@*05upFhS1oNM=i^C%ar^=d=7H;l+f@(z<0G)Pk4QDW}9`jRKIVLDR&-TcT#JcYy}{d_^P}lj8anpO<Hb6Xvs7SNR;4Vxa1vcFC5Rdsmhqb?xCIaIVEo#(R;<Z0-N6R1+5|2eeJ$OKFfMj#e^$9fKU5Q7IdzT`lJv`dCl0Hy|`h)w;JRVE*z(sgL|+M2&-_#L@T``&vBfu=>s<3^_TvS&eV5>IL5_7+u(im9<6^N1V3uRI4&G8RFGa0Coy+k#9NX@ktu48m8OZ#laLb#r%<_2x{UaBk(yt7i_1Q@ub_ZDaU~J=elVYP$rImJE;UU;o+A#eN68^#^_G^S6>YTLZcPO9c)^=Vq=<E>>W}FTmE(Du-@?xb0}H2l|eItB(0F0K+ZEPzigXNU~F(sugevu!vd3KawF-O&sJ}&QL&|I(+Gb0WvZ{p&U>aHTYBnt){7*gW`P^=hDGBw#D2@PQ`>I$pNW1vv7-}qhA8%sL^{${FDY{Wc8>&BSvJ;M=WPtVE7%at6mDvY?ZSSD6>x)jV@A>;4cfkiZlLc1wBe~vNQ>p+p(^pZ6a-UTs64DgF=M|Y)q0=BdMA2tZLN7!H=K`^odJ&Er-=c|O^=Lk!NIV$IdRL$4>mA2%iF1__)wc~gmss0&_v#Bjzg{Vr@b}GCKDXxlI%I%$dvw0mZJ*)_6qkRgFkii38V%}Eh?sC3kxUjUw&Yxc<tgcITKM;&$_{hLoLNzk)<X-v~bA>rbqZr3EXBE3D>9qMgyOIqD0*iR0Et$?XuS7$v&muLTP-xMCQ;&I!z+7_)p8-TpyVZR`s_f-uf;@j$Kd^FE3|5&S5s5DR38LsBHvSX+jT~&_>YIu%Pg&cZS*S8ByJrmf@?deH<2HP)=j$exw9}q9|`;7{`Rc#T-whyR*AWb5ZD~j(?kZ4|-{DzK*D}uKG$w-?NZeR(hG<pyTRc1;ZrUmL|uq-_O?K%@EsDz6;1eWh*|-Cx+p5UN&s4U1aBPe!}Jc3rDu~K(MNRzzu=<6!BG^I3tKywiUB>D(lFJ#7>o<N#3lI!1#B8r_p*Fpmx0=8+v1#5*Elq|L7D9ls|P3LZVI_Qf|wP!dzpCL9Eg-@g!*!37(yvz3VN8DcAdF-QVPV&MX>yMfN{8Wzm;>Th@f%9SOpZ5or23!k_TO)GDXlCMTXCdYl*;HtG2Z>0R6%l}#hbt9J6Q(;U<H4UI13xy^=i^tfscovvfR?949*!{a;TY(|RfIZzN`bRpcIEo%zBTY@BoIY1za08aJGj9DGp!he_G>;%dp0<2r!ubvkFMF20W97DFH7=^zNdm0(EylI}`d+sQ0s8@Zrx}blRxfFXag_J^r_-adQoT4_ZV!|3j_iEC4cSi!~;BT?L-VJMzZ)%bX&&H4(3C2G_Oa`IhBttK#rJ@OTUE1@H@m+ZZ%?Fn{?j5OBx=Vf*aI>5e@`3T3CW*&15^y&*XEbr&f<JbY_ma*ts5L<`Ut3MOt_~y3#-%jHdZ_SCaPih_Q|YhwFj)3w;#7qWogrw|HTBcVdWg<Ekl3k=XKyWhCl%RU@t8`=si%?`!iJ)f=e1BcJZkO-)BHawe;aPV?hV;fJuW}qT7hD~v*}<85jaYNGw(A0n%Qm}(D`Ag;r)(us|R}c+Jw@(bZ+q?4&S7_q=FQsGzUJX&nvx@)k{fg+CFiH)1*~DuPVlcxq1JS*`^^Ztlpvs90dK2W>chz1={3LyYYJV?g;6xP)aCjNOK)q)EDWHdJwp?i=ea<LV?v{;{2^8oqFcpKv+ao4b`=m7(Os(`$4qQ#i=bA3a`pE8m0}RjCiID{qN)fV&&KfQQ5^<^^Rs32dAsH{-)?B2SW0E7rU^Y<K;o720P{9UO~<9QLL90W6-~vG><Z5E}J1%&B~HD5B{z*`pG%Q*K+NERtuEi8#GoA58F8dIkS6{eUq+y2{?fAvpY>10r%DvF?_IYxW(HEj-jmO1UPm5mj^&uW5xT4N&D5?Rg?#P&#eJf4-1cOt9G&YWcH;%HO7xRlJPq?=+Vgn^z-l3m#!8OB|J~;f$~seSx(SDuRtcof+p=Q3fQltbCo*q-c0Lv`68dWp9diwY~{6G1~1Dd5^z|)s}p|T=*wBJ6>JO9DS%ro;DopT#$|4lQlAUr!mt<N>^YFA7~%N(r<DfRDPMrW=%%Nh4LAob%@&5azMVLlP>uKly>Q7Cw>#Sk8Lv1R-#W_aY3j|1LR6N@JY3^2FHi{UBb_EDI-uO6h2IC&bwgkbU##`{^rM6!lWcXy`UuCphU2Pxc8B~g!zd7qeic-C-KaGSA|$;BNtq}Bzns7Log(mapfB!$UHt6B8V_OeEhE0kZ^(`hD55xTDL!2jQtb9x(?<zI>7)yR#<?`Dh5!7Ow<(QNt}M-7ICcMuo7k9KQhY@hNrXmp-Kmsg9Tvyv0rED59&y1gGdcF116E8)8ss^x0>Dg9Gssq8@kC*h)Nd?p+Pl+4ql$`&K)|!zBw=Y>PjM~o;`R1u5bA^YHPORGm%2Pbm1BpFzX;bgy&oLj?a20qJQ@6BUvK-%516<a$)lC|E_TsJpxj!NYP^Gd3F}_UzloT%iS674rR`oEDwD>4bTE(AA=MHgJYhASTJPJP3{dw?<s}%xpTdjCAD#iac-c1&maR51!ti96c$@4dk`Rn&9<M)rlWSOt+zy{V0PNEQ_(KRGBE{@yZv;-*@rA|G>`RM@@I(D<sTDH8FuUmB5}$-p0t}kr+)62)=tKIcw$%xQGYv(A)U%7b42n|gx0w<R?K-0(mn9W&hhHQy#|<EH1{UW_2g*@0hm6C}Aqn)xxRj23PH%u?mr=_Ek6MmZQ*OUd2CCfCb3?efi1j%d3wgRk^tP1M6S+YGCTl36&Q0ct$e4>x_{kY#0Ht%Tvvi`8H5Q(ZqYU){7tr`4CMsOExIAVl_2bey4InKZL?7>BH2fW?#&u`j-ClK0s!@>4O#V^)DyTfM)YTKAh%S~2LgKScUY?CM50DPeXGSQ&ogVmUz86BL%VvVfYwUr44S&mzW}1Om^n*sjZLHbAr)tOW+AJLtIx(Uzxt21-{$uCHqkdVhg#_@UG44nzVIVP7$9*$`Nid1eh$!%?;FMA2t=WhO^wLAcip{)UcJC%UvK8BK)t9a+dxtf)<8OqXpzRw)^y?REo|gDFg?#XSKf3KwB8gAwp5Wy6{C4gxW}?^NJ%aOl8hPNCrAqk_DtvoG1*GV(TH6eYV{cz-rO2Y^Mny+o-z2D#+d>!%zTpEP05z85XC`^JGMX^8f!`0u)-j`;AA0>N#b2}djAb4zBk*J(`!Ekm^-&lDg}B7Cs5Z_J_fM6jw^Rais8A7W>3-}_+g-Br1Q@cOnhj7~ZA(jv*f}Ria_b$n#x3aIkDVN&GyyQkFRnD9SmVq3jZ~kaROc!(-5RYKMRIs03|~g~AopJeFCW9lN-%;pTcmdUqEGSj=rTDQzNYugTo|x-*xlD<nk{bSKw!a+zJo_A9ZR#im~HQn8L?0%0oSBg-{xhcHJf>1uLy{WzTwJUH!-L?DfFzE(0&-t7`494xQ}i)N}0c`O-S|wta~93H>_UI-|GwOxEFJwKmnsrzcNR-xD2S<pkr}PsO(|sEnc#gUn1lnv~mebrR9<Rjfa{}qsrPCE{+|Dh{B99FMoeou++5cwW!oIAy{yEN!LWEnR3J)Tobc2rH0g48EJ}HBF#|ZXvFnc2wbuE6Dnx`bo$5OBzo6D`}aX}5d>fzHasU6@E#(sPAG0vL2lPIbv^2Bm`yEIGHZ;-Bt_vp&~W;R%m7&wdp`A5EFh)BB$&*gh4g{<u|u`!WB&}Bv}t`t5=#0t&5)%toTjA|c7FTrm7QlrraIS~v4m=$o~B>pcRJ;^yerwLgKw=U8shX)H4S0Os_wo^g>%PLUm>{$#nEB^UULYeE5tBy655mx7@eP4cvBABqV#nx0<SJvp|k8?G<kd{>(%?&AOE~He{M}CAnYpPf5@DD=%u1_OEL{>LQrL}Yxb-ANGVSy{-i#KFoIAqU{5CKqi?s;HMJhPtkzh=Na!xS!?SO%3hfrH7*|kZ!hpAMRzBmeJT2`=^&lnjUcjsQYM#q(BOa2EevprgjZlGP-lAixO62PB=wu?IB@q!V@N9y^gkeHZorpJ0MSY*LfLp6tQlddA8u?82M*shHR$yNQmXDn7EpyLcv{FGbnR;`b0$m#8qT^j4wDDUw9>G^nI18-)SIrRh*Gxp=JvUUe?34_LZ|Z9oR3n$L=m%n#pb(-3Ha5I9vE6Ar*l!j?umVz_Tn6=qd=|fWuS*4M5#Hi<>3h7Y$O?E%=&$q|Ft}>dy-9r>RRLzN$seV`!0X!&AS=hAdG7fe83VFq#M>QT-f<e>rhpf^q#<HnZ7cgQkwx#G&1r1CN@EHAwDCV)dqY=G)?(G9(#L_ih)?BV)tV1D7C29Qt&Xngumab4N*adwD0tjk{@Bmc0-P;VcfJg6$Tk>x;OOv?fTFQ06$oQCl#{Gcrydp}y_i?bFSl^C9=*p;Q6&8t&eBy9*%ml6y7No?6paz4Z8<juvf)q;+!+qI+u_dn>SZ{c*Jk-3^SIe$T%T3M&+7FO+w#X<AZFkMe|k{9h0wk8d1&}oFl70CU^~xG*2Y5v{aVYu8{gfdwksQ@{)cu@TI}Sx!#%t*>x2R!OiZ{^{-VXB@j0;<Q+#9|?xB6$ghjkqPO%1)>#Q1ka>jZ6R4kiq=%6f8p1*2~d0%2Y*&tB-`*xj8Adv-uDmaJABv#?^&e3rVdEHNNJ{1FrK@nh1y%^8>8k$m%_+%862!9+Vtvw)9vhrNd`L~F`)nNY!SVPob4f$j#z<&G8#<K#}P7MaU%$iF%dPs1K4A;gH_Tv9xFz**}cz;x;E*Lf2-P=`ZG$Npk2UV&FX@4PB#-5F0>3fxl7yBc}ahvy$>y^Ac=-MA>1li)?fxC>xv|oNi@qc>e^Co>lcsOnG$lKsIE`@@-VN0)!y{H_5o$okPoe5XNk7lIQTM*k;mN0Bx4B?9HBdYJT?mWb8wm8()fpyQeqTY{ITL!xr!iWCx!g536JTRhCn-YYF6IlavpBgqJ@ME#xKIu~!!d%W`d?93JNR0-&Q%^bqpe8Rtjx$Pzv;tqn3h3l$z_+nD@YKDP48VkD#LfTL;6avjYXNMf#j_mw%86Gs!diT^%n>JFs7K4d&+7YxeN)f^hWW`Rw|j&lkL1*|hh*A!8%9U}V58<B3jOh7`P9kjR<&~sIq}^-D`FZGo#k1?@G`K!rkKeVfCz23_cx(<yP-Lg+NWzhD#*FYr!@5s$>?VG-;KO=KeG!{8?@kum|p2JoX3Y+V!dL%Yb@X3i>RVxSVAz67XHn&Z=f;^?<ZyyC9C&SiDjU|O?96cgorZAofQue9Mo-cGR?C(0gsMm@Rg+_L$1H+V7@znd$PYkG~<f=K?yIE|6|Wi5iY7=uIkYcwvnE)OcwaT(0hLvj<a-{(e^djC2qPdmkUAKk$5jdo3w#CfLc;w>t{Tw!QA~%{EG>B?CaJ9zfdboHD!VzUI^czgxv!*0U6mI=^*y7bi&IT3@e7k7&ncSyY7H>#7NEdpcVVMx@CtS5wXaoWUjZCW_HhDIkW$*pSQy&CtatHM)P32N5BS5-nx`&dv7L-4}f4xe2sQIfDd6s7+PRV*fWXcw#1p+F6Xj*HPJ)(BwIt!(T{EPPp*$nP^NbdY;-4cJ@DJETzo!Yc*a|2=OYx^h)LnlBJF~XZ=};<79k*I<D`Y_ugLngS}&$Y`EXsVA6VRPuKF(VC6VSV4~4Vn$f%4Gk0Ji=6<?d}!Y4hHYb~NMC`Xog9DHmw(8T^_ChxTfcl|hYNpfbD>4-t#`fE-$3uIJ>#WTsyl|*pqF!jH*mJ6M>HEV;y4h?xwpdnhVpuEgcf!7kl^crfqmZz3MX9YH!4P|Y^*GJ!sB=t}(pjED276_Zv7eAfQ26aV7tQ1MsC5ywu@GV6?_USo<js*8lOiL?@r}W;{E@1$%aBT_R>S!Gn3jZ=(z9$UC@USw3a%yU=KO;~WNS8&s5RPo+T^4g6?MYL^iQ<C3nkYEA-Npg=*qpmngTNST4Xzjq9gIcm5OJpa+!MTp+4vE653L@eO+tRdO}0nHrIjEtat!LR%x#|my>_^#TFSkHibq~R8@OIJU(;QsuomDuij>+;M7f2^U36+{Xp0~m82lWQ?>+J8p6ZVf58X!@q3$}H-K8LIBZW%NaA7KYr6Fp+Q*;+=33SH~HDb#rx_nHotbBaJKqRa2YY`mpZ3C>KgP65$R2(QPlH{a%F!|id`oO?bTOeeIeBV50m><B{Asv2zZ`xA_nyeJ!RWP@vrlSl^fp4Tolsn&c++uuZFOWV%!Oj3t7bII%xqUbX{E`iu7<I}2?Tw^`ygTJ~04y^~!St{VJc`k2orQ2C<Fnz%+k_G%fUgsu-q2)CZs~2gs%50h6HkORaYR}gNuZ6@JLn1xLcsc|3{)Q{qqQ}xP89J~jHLtsvWBKpqX$FY9K;j2tQa)ryLtu(n15SPvWo2j>%*YI0%Lk6IF@L{A^`baaUp$j6J@g-d}5#IZM=Z_3#;6G9?M349RA4VkIPZ`4}7b4I`?F_(3jkptKkO85Z(87pi^whBNpmExtf)P#B(Oh!zuly#g=z>V7v8)R@P%6AO1<%nyVK8OLZjwC>RWyHim6_H&qsJb!*aOgL;q?OJ65~?2tW0l#oo4APY{jj=G0z(P?*y2WP)p5<7ghJ^pU5gxO04s@V{j=dHWAD(w&_W+-glTjp(lA?fArdGqbG>Ho@1^x@Ui=QUguZX5NuR0HcX$k<)m+FXDJ%ps&Byu;WgGfD3CK2_!xd<|J<$Mf3khptLw3s&2M8V)(URiSAJFQl|_5<{|K;x?}2=qXvRZXE5v<-1Y(Q@=C$lg5#+CM73boj}I)6@QqPciWIXj4DV!7!p;klGvp`a@4ACEIC1P{XfvBohO!c4X^v6x_DDcr(iICSW@^J&KUP&_>it`QrLccbW)IgLqx8=beZZg@1)PyA~{*rt+&tUGyVZg>C4MX_(Q~AfpiI4^qlw2YuE6nDa5(g`{O>X#+qX~R*Yh#&)N^ymLp7}spV;dEp%U%_+L!py(ZCjVBPKmY@+D+>@Vc33HoerNgJYFeCM|?%wI7D^2}MX=AKah0&Ayi$Lp%=SL<Wb!sUi45Hu-BK6KnQOCb#+sBw;s?Y}mu5G$`K8R8hS1cM_yc`9##+-;*Cqmvl{;F<l;Hb;Y}40!o^CsMDt?QH04s!ZtIpi)s{j?=D4a2_Tep7x*)+MBDDAiO_S**e|VF*8HeYad=G`<))k7N(P#1P3~W<y8P^s8@T78oWuCpS1@N(~si_g7yxwFbaUv$_Bl1FSfIR@bB$JA(xzKxeKQeXjv-wsuvsFGe0iU5qYFgRpxVWaaR28)g2TL9n%uf$?q!25~d48<CW4~K&%7j1bmpa^qy5oH^^j{c=7`$MT8z$yh&ocK%Z6O{eu->M4O6wU#aJ8*m6)D*RXW>)Iv~EDuj4piZu_%^Bs1|+fYys1bSEne!J)n4@onW#0<<-Z%(WXll5`Zm+KAZp4*cGgdUN3JDu@NfvkGc>^dwFX`${lH2PEa!~u=FE9I`VJ}!Pjx_u+%TJWd1zyJ|wl5kHeQ;tAz08?SWe$KnlT;$-HtX#`>wiFg(Aw2WHyBRiu<bs7t1k^?^5wcpf_Lrr7#Io^z^UEU3III~qTDds&_2_l%l!Kf?q7(?0LVz?M@Xl(dn@nUh?HxQ;C^xZ1tQc-+;^ochv#}#0tZUm~1@5YHM;A5ZY)Xmx$qgrK8Jo{=)s<u=PZGQEBQjv>q@;1*PnNz2T=Dyt?Sv!Cc#f5@n>b;sY5*MhRw-I50H>p2LhX_;L2(K^rrZ*133v&DK?xMNY7svDb8#`K`(E^EI&H~o2$Ln;=DECeYccy?F{*AJn2L;EZb@+Z?P(<QRYJ3@!JNGx3lWW=UZYq#Q0yq7bSp07(wE2#DAB(6ZU(JP*ro7MhIO0dpiu-^TPUePd@db%?`aQ)?+8krHdvSq$neZ9Ifm157-H0d->eL~lVpDOeQ;96L1_hVW#_e+->U)>rB;`WS4-McNVxPJ*f9YwtWjcx#*AQGB_h0uqIs}ZAX#4O9~f@L_}rUE8G-MGg8w16C=O%biwrAg<A%LvzlEbv?|$)crvFh*`;`jzTT>B&lbjMc^V$2+xAp+6jpo&RVJW++OWLCy<Bi`c+@*UpL(#>+hskjLetk~hi-=aOAa7;=aRNLNM-zk3(j+3_l;UUCwOt4?Ie{_fwUqFbMn)rNuwE9EiArK+dKAp7*=j#|XFxRR*!TW#ovGQi5S82PfjMt+W)q!`!MDK|h*-{HS_mb#jmBFdIC)k&r<7lO^y9Bs@0a{#Ea2m@WEcuM8gcuA9Ap#&E>$V>w^wz#iMp@f`bH$Vt9u4d*RyCF)-l0dx8Q{RX|m>zSJio#XOia(hn|_d^RThJ4)7#_Pw?plSj^#pjZlPYcFB)WT~BNE?)c<Af@O>Oh-aw-3prztk%3oS5Tgy<PDNDU!nhHgJnL}mPfKIyXw$>nwYWw6bN`vtiXO_#OJG*yc;VR&PxNmsx)iLWRRwypvT5tGKrz6}nv4jUnX_?S?7XqcswM>xg~?@RHXrO{EW$!M&-KB^Qqp+mrz8{eQXp_9HCKr<`8gUkTAx^8kNA%A@wcpJPk~$NIfkseq0ikBIZAb=QCEmro*g;A3Z}qpHM_8K4OiL-34tEmuO^vN7VCg(A%TMxe?6DVh0>_2=^V(0wWoSWsuS}co9ByJd7dgd;i4x>2_>-rkXC$!mL#LL*9u4F{8zwv&={uD*Q)kz2n3xLqfexit2UmeDBKEpgw0C$zr2E1uQ_itWa7Lr1A~`l`-!bitkkgCQ&*UBFvyE~CrPCM9eFq7jOzi6y!MWk7<YA68eE26@qvs%`DHt-XlSJ_b%yRowEu3smY6E2!U0I6@vIuC5^r(S*Nruk%;j8=AvUH~JE*~uJwh2rZbEoo>~=SL!M~NYnlI7H?R7m16ZXbL>vhx2z0QcM=x3x?C6@U%UO7qA!hFN49N|lT#fD-=Ms9U+oDD&ct*)=I7E9QBIXkF^h7!k=7jlOjr40=xBBh4|AWHi<4NsbK?^iwY5r>jS%M)p@H7Y1YM2FE;Vs5|WX3~bxOahw?ojZ2wZ4M9zAM9LFiuk%3IM*lG48s2XDzBTEO(1<ze7rW8RSLP-P`8_7SGHV(D&8!(8J=CUmDWE?{;K%GIiaEZQR7#dH&F-=Nh6m*3Yx^k&uoS4hi_P?Bj8VFvm?s3g3e?bb;<2PAE1fUmF1As0J?1ZmyIY$Wr(vIrAlU96P}8~adh!3R+iC~hGK<<A(;hEA&$dxF&g!Kc>Zn@H`(&Mxt3p8uJW7TfIZpRY*>U*#dbOKefdXwGFo!4)t@_yNJ)_CE}M_$XUT^l%NE(xDLFR?G&CQbjwTsCeTH3HX0lk?e&5IM19LvYwA1WCdz0!j_*ziHg;1A?NTi>EEf0s-s`h*iIUpC)yr#4n1lI}47kZiuNr2PSG_Jb|E?IV{L6%+2!2F}6TVP=_C2iG98N^T0ke!lY?1Ig=S{a@pU`?mfA{5EYbgskN?xU*+uECT`D~k`nr;(-vE21=0z*?bYF$8khjri^TsH8132hC&;zSpQhV;{{On%dD*VG3eUvtX1M0#b+KpdIh(tIPJt+7VY&5g)Q}L1wGi*3nw!uS9le_aH8<h!m&PZ1img-7R_#Rxb#c&yH{lGGqd_KzJ*1IdyW-y{}F7j<ue_x({1I%;d4AlzXX-bTjPIH+7N`b$;jtBtRO(+CmSQ{K_c!Ph+c+`wMS>U8QNPTXC%SP^Nn>vPv!m<<*(;WEAOiE7jpN$L_0tR74&H$6>IhN|#vjrIe-z*;LaygY(lr310vS`_RKXJ2<sF;L90#jSmGhW%@rdM*iN%^GnMPZ&Jc%wovk!v13oTV#N8GZC{xZ6KGyLXzAX4_>S)K89eG1i;C8YwP-Otj#P!ZYu)}-!?wzGaD6eT^Nb8+p!r#}H1Ip?2+dMD@pSb3K>=C_+i(C4`_&t_Ynn(ZHHAe+iI)*N#_g5f%^KWV!E+CG0Jpef|GxKx;J!F%3{`qF1nD6&sA2$!<0ry?2{~j^_a;*D-8l&uz>LJsZMsp0Y|zbiu}i}>Lwodi%A9Mg^|zF$Y~l?%;c*)7am8H-fRWo)?W76v<d$DyO{fk31~h?f3$>_Yd*((IQ2|+Tli^SxO$(F;`HC<tJdE=xDon*(s8wc({Qy=7B#7U`?}t<-R4?RY2o-2yT4BKW_`-}%rA5|`8;<>juzrB8hmd}7u$2(D4fQAo^3PD)3-Y}s+YjTf^#U;k>;EAu6&8V)vmqcV4#Qr7pcNEMF37(pbbDcfP6q)mO_4`4dPO<1+0af+S-0YSn@PWq@W{H%Jvpqx&ui!i#otGz$XGUOOL5Z?q^OU5<H?OLp1+jFlVPc~2#TU8ZqMq^M1@BgIL9Xe=h4vFt?~tH&Sb|Bu&Vh&p5YkdIQc7Rso^dUT7-X=#&Yh~d-gbIdI8u#(ZvUuBI+L_ulDdh5+X*F<Fff!a_m@)onbAX?)R#xlOnl&a2g4_^~`Liklc9?D-|?}BC5Jisi3QZ1!{z*8b;2AH@NPxrOEVGJ{q!G0_Z!L<Z~i1+7|3<`P@@Cz1l>cB3u+=a#_o;@6wwGBW`NIKg_Q&7&wfBa0F0EWvaHE_fl1gp}XIHo4F#k<ak^-S0sDGY~kar-MLo$>qi(f6ukn3-Si<@Qf1BEV&4DO|DoCT<ev6{O*uehUO%)AiAftz(?<_j%@bU5_NY@OQoRul%6ZZzAGuK&f>XbprFp_b$R=W0OMBB-F4@0M!TfcwviRxG%y|;i%B}>Z4A2~VLG1eM-ztz&RLt_LC3<Mv!KjMg%695|$>CcVk3CsrZC*#Ip_|%Hurqj9jJGxRThAXoMQuZ>rXeRVs9<S<XJ}|hss|N^q96Qe;2z=!r+N&Y;wNjA3%GJ6B`1X?$~7`<gVCv4J%^T}iVTHLj#+F;?ghg`1-}0&_>gy+F<g37IyC!aJO0^Ei$L}*DLh!cf^}~`_R;K)su0ACU3J2{rF8o~2M8+vgk7o6ufHQIU08Q9#N>ld+vP~(iV|sxE7ypTUa}g!O~R#gdY-BP&IMx8+P!kz^P{_BTb}PA;5BFI@+z{g?_-YU>T=?uwDDm_uX*P<b)Y=!i}m_I1l<LqQ_kmRWhoAe3P_`FII(`ZlACGn6UnVb{9?*14~f+$i(O_-ruX)5XX=(;0tn$?-<V7xKTc&40{S@<POD{%D{`)z_d8e?D?Ha-JZ$$cc#L<k-uc%#%c)E>yKm0v98+hi{V4Q@dvNvo^(#ZN(;Sp1x>F}Ix}vT1*fzY*NuRd~4SX`g;<QeIS;Rw+AJ3Ad2}jN(-Z|h5{11x1sxE%34$Bn{jUzs6tDcfiUGto5IAeaM@!KglGk^auCPg$%Y8Q|Loj@sLbM=nZV9iW3#j1}y`RYWWyGc^prWzLx4$o6g*wGks(Nu2|#^{&$7y08<9JKv>0(K7ej3}zV+;O=_aMRzM6yf*Pw#*il;<|IkJofOehy7-6q;qJ6FuRI*X$EttqPx;;96ic{fgzn9MYH2jNq_^q*h_vJw`9^HU(bYtbr5E&(Q4c1%Z!sEINF{4vF=mKYg?RP?J|bZ3LnN0Tq~cdd$ip}Dd3gf$G&?^X(~gykr510`IqmpT>N+4y{4hoH=65BFxDT0{L1TDiLdOMiNbK<jqMi}S${wlTMje<im(`cg22*LYCpu^#XBahGDiDzeI5%)bH41rp4xI~x_SU;Zx6zAj9pdM%`w)@|GC5FimO@s*5d07*r|VOQ;Q&HV`65;KzqgG%Ie~7tj2dE3R{Gp++?HtL&&oQQzXib<T+JeMQk>bo=xKR2Z0x52wmVJ+m*=<A{Zj4UrF)4KA|hzp7n1Is&KH;!hOjSrXAxtQr4_XsM1G)Zu!njPw`;wAiB)1#1F^4VqTftlt7|TmSS%}<U8-2mDO`T!2rzUAe_)xFT9=*>?Q_k>b2-CyF#BTq6F&`h|eosT-64hoq*wuZ9?+(b62=*3oKt!m94TkYQbl-utzCT0VaK*ag~mD9ydM|={2cu+E%tD896(~i`9P6D5+Jzg3Up5kvSKlhvm9fV*28mGc}M|c4|IPq?d(T-SiNjHF9tZP=Zkk{-S8Q8R$S^6I|kd@LrfJJxdydt>;5e-@pRMXwCsD-;g!NTAHe7=`gtalFn<DBJ#A|pUwp#f4c#7`!7xv05V%`T;Pu{^zMd7gGaKQZRD`{U@~^n-Jy(r%CFHJI0{9{3}h_aIGO2O4}h}hi(bf>*;o*&c2^nx1~o3rLB;|tc#btNpW9>ni|FPddD||;cMxI6c22Q>fM%L6M9#@MIyW$n!#R__<@&$%eB?<4gaZ&%G-l!=w|BxeQ$1Lm>a~#4*%_3)IuyVQX*$L0xuiCqC3TGFFXxO!rYtt$)tPgqiCo7SPoED_<w6CRWr*7C3aEi4p_}TC6Bz{c+TdBxDgKqpe_0bNhMiRlQAWS%pu{yBOa%;mapf|}agCeCnK2}YYkWKP%+Cjd7<^M`+)jJ1Zy3)dN>EQf#BmR*`5&CKkQ|&;%4x{(YvUT?Q0iNU(k?W;bi1Nr$w(_Orp@iGn4@`7S|J}?)9S2nzp!>*MHjseXVuaXNEhuL#Y%4j^)X<PczcpX*<q?QMXFXAF&{m3iTep9EYeeUCH50xFMSaw5^6vtYu55V4dnbvd4R&k&+(X8w;6`zBR+rnkP}3CWAuHgm$Il4LC~02{q!r4dRcb@;pYS7nORI9Sj^|E9(I{$3RRvRd5!0C-SrGx43V8HbRBMtxNAc1NHJz=WEMe6ZNN38YmY_`4GntGV*qWg^5YojpglJSs4$O=KdYd2=>8rSDF%1e^wFx;sWhReA`j{zn~ViDJLRuC&Ja?qLWWFs5jJO{thI#`Ng$7srsmi5b?ljyhK~D6F||vERkxEeVB&@T?S7j1Y(R2vQZQCLG4oG>UA`S55BnSX{{xRvm&XQn;S}$~+8hO|1roM<r;j%bB`A=K1j<f=vMEYVIcJ3lSE!7Cq!HpsSEl>_HEh`$H1|`)>w_23nU`l)>Wy)fEhv@zO~Qn1XDAs|{G&J<ip<t8prUDP10ZX5<p_}fC$B4a=+3L}zUu*Nlj5IbKmSYLl-P>$tG7b3zNbro!*8f?<k0p2kWxQfgjUAm^XK(42Uo9FBA%Vi;92F<0~P-tNTYzRY7Kmcj+-ZU|NC*6i1}&p$ca<PVD4Tp+;o_8kv~x?PSgQl-6iK?W@^lCn`Mb1ErbYqH9J?J>ek0cgTCUjfa5?S`2(+KM$QVaVt3kgu2~=+taU~-M_mKH26nc<V{sb!?s+_e*7Lr4l5k1=sH#^}UB|iUb@B^F`yguSH*8!C)hTgY9xr#j<IE7h_T(#y`sG2l)K{Y1Fps;UQz2F5*|m^fjQbF_x)SL0V)~HD^LBx9&d%tsmclUq6=R~aTF_N|MSyWAnu*BTrKAKWVippS8xz`CDh}nI4xlHUHgK6c{~KW%LnD#eW;s0>tyQ(wke)`#KgB`!K~l!&jt-XZ&SO{hib;3$M<m0p&T*5{gH@gWHqE3(EwY3kHJV88FCBmp3LCBEG<J5p?tzT7<R-RKg4^Ml8;g{9Gq!>{)3)Z3I_-=<+T0ae<kS<7*`Dea>1gbu&Z>npugNTg*&*Xqw+h=Sst$jvLyC^2mU#R#k6J@X$Hn24_3@fUc9H1Go=WDu`0;Ik^1wGRtE;&V+jz%`(xG;CqL~C)@w<>uI(fS+Hko)I@~#!)gbVT6uSLR{$515)tmqs4)T%!?VQ((gV_5yH_7i5g2-)$l=Wxkz3G_eOD2rUddYu*`#o*G|1LgG0kraXz+uN|F!}$66hxSqHaiKclkNB+a{c;u;w`5i2aJmh-E@0-7`|k|{@sI{FXN!D%t~uIg1up~wEpB`PVRZvzq%5j{<x~&*=I3|3GQ@bIg>C;2%u_|{G7E_8tbrzK>ZR16q%gAGJLWf?=Icryva|tqy6V_f;j99f8B7L~@MSW9jvz+R0^#g!h`LaC1|UKvS<v=f^D!r<`38Zxd$Q%3X*LiL5Wpjlxxar3esNBE;f#Kf(@~k8eWsPxrXcfQ`IX&yc)qLbv^z+4EC*YYT>{a@he=c-m^?hyP;+hR|MG!rFmu!+rMY#3n%z7xMx7>@?vZW<1rw_>#>jmbli)xXjwgWtQAJk;a_uSRF)={aXE+EL9$(kip65fB68Ts<?45Ui_Np2^3?XpL8kqeXG|k3!Tlcy`+&nrbz|yjKLw>42J(Qxhxt?;v<7QR=qo6@!vyf(`BVGjfnt?tqWs7=LWizj$l)VB3#U3#Y%@V@tP(KJs4+f?<k1fY2yyWB|rZO*hV;@zt8qd=VLU+wQI&I4@kxxu`-;O6{j`xa~^7>)|&k5Badu>LZ;tTFzE|GdpWP7E=UWZnagGHJN5%>l^o`^O4#^ItolN}UPgYNyb1k?LVa*GD8`bdh?Py%7IJ?r8jEV51=@7uJu2$B)Bq&wj-Y}TQd*|p!^wTnm<#;>?~nkpP33I}a09!0G7FfacLYe~$&C1a>=R9v~Cz!Yvqx#=>z8f=iCt|DPDR?l@3-z=h+R`b%C6}9zE-^Z`r#AowBkCp3mXlncqc>p>*P1Rk2d9AQ4E_g4PEoZpr@OIl-ARKX&40D%l4D=vCR#Y;Yp-8km&fEUd^>NI;PyC;eh$V-~H~_JScEk)ItTM;vEI%9(DUCJMG-@tUFF)#JzczkTVZetN!oFRu8hVIXT=+yBJR}F>8|&@OW{K;liOOH9t*~)e?QeI-+LZWYldZ}IwegG7wA|(7eiXP`)%hM^E18l43~d5zD(xktm*H-K2@_VD!+T-`hRd(;i^PlJ$Uy^F?mi`I<W!-~>a9uR9E=DN=^pUT{6IQ^v)oc!3Entvb3kCLD5AQ_wI_^*Gt508HC`K5*HS$LuxO$b9E@Egs|)IS!ir}c{A~mit1WyDqtF6FR0qP50iu?w>yuAI%JCZi)%1E0^6Ecd(Xtk`fxuHf32aX@1_Vjq&98W$TM|}mfU9+;NrfW%BLpe}XiI84sSst!-eBp+Wjhg+K;XPJE1CzmMVc}~C1+O@v_FRDjC!QhV!2d}C*-6Xv>!-0%uH{q4K06WF^tWj4{vhwE(f>c<q52~;+H*Ms=N!+lhOj}l65AO5TE*iq?Gj&;P}wLm-FFZYS@`$eiX?0Tz{Inr(!yE<r{`mSPFdJJmGDj{;nJ^Q{7;Vtz8_!+OU}$<5Z}XXv6xh<}%6F86(wStdALCGt`BOubmYa1#1|Unv{PddIyT&J6&O*w)HPmsxBUElPpNnzAa)27M2PDEdmSxPl!|Ld$3ZDU@R>RlvJ1vrsFdi9dvUJ_;jts2fCkmUE)c9y@(UZpkbbFnP+fo3>E+KT9E($g=%1n@kh~JXvLa!TRxJDUF3*Bm8xD(AQNmk5tpy{rv!8X;SKAX$f+`zNAl6DW#Imf*wdAL1({!P<E?5=Vg3jQaw;{HyI2FPq8G!u$nFI88QW6fm?HD#-@;_pnMNh3vb+XJCnO;k`T}{zxD4rMn{>to9-EKwL4Ei(S5EBUP}JO72RTaYke0wCgeiAT0$`z*>-2)V+F^4>o^<;V4mL3Jj+%g?sjSrSc0DQ}@jiOHl0))42l2qj7YDkJZ)rlwM&VcSSeYism67R3*QBgxv7SX)ASfTGx)i9hrwwU1?WP~-fw+?5Jn>cpyr4&a8wPyBD__6tmNf_d)_#sy_e2oWlnNhpcW@~H5cP=1_)MiK$1jkjJNprPUU=B;KDWq_DMrzSc%D+{W>ZCCHIR)>WpH{dKlsw257f^F)S`ljvO+RejF3;MsysvhXKltg+6Hx=`XE%(CtlOG-(r~WpgiRFOoG48O8&F-|Lmknk~rCoH#K?Jxr~4fp;8!IxG|?^SzD&2_acr}pk~JOkYK5kZBvp;j3Tbj;=jq5V6H*djHo2{pUCNMi#a{m0hVCnwNKg>CTsWrag~zWKCJC~H_F|ToouZICPme?wUnD`4?RwlV*KP<x|X>LC)lU?$Qv%5fAn0yl17V6O8}}U2=IX!;A=h_cOW+x$A3QNTlYRGnT_@X1!Hrd*>LLqJqYJIXUz3w6iIoS1)*tghJ<H-L3}D{x-~=*(*V^u{%hn=rf$u=Nsi|}0AP4RTv@<X-@A7)Wmx#rzy7L=$cBT|z&e`v7qR5rIe5o1KD9~$0yVljT-&@Ea#>p0ZDiGVO<xw<l}WON4JiVYewcBj!di~kdoFwdxfDs)1wRz-m84+<NJmOo6ay(6*Kz;J%{wziTwZ7<_^!4tBCp)ZM!GftrUZR0NN%nPS5LLBC#8B^Xf3lJ9igpsFPIgWT<d?_aSi(lm{J1nVYANhTars7YXJ6&I}oGB$yfjo)DSz#btrvp9JHTOx~*@7UVMf<0YE?gl=~<{T$4t9fFmT6cB(@68D>p08sbI#$L-bE^&AG@ISf$N*`_YyuyDYiEBGX6xAv|xr(%cb60~)Gb{7#R6=8@`hob?P2ptaA@Q;Q&;_F|4vVqdFsoE1>gDvZ4At!<%vW%t=10fD-!EYlZcS)u)*E2l@3Z4LRR02Ledd@iV#uurj9+XF-CD}xnu_GJ(5fK_AwrCuWQ@pHaDW17>>Ug7ejO~#93LeTh?(E}>{X@nyI%fr8jnv6#>pPVqHmK+#xqx=u3$MtwLB&?4KoE`Pyx%nyBn}i4NUWRfFGfe$_Q#^j?OiKBrJM60Z2vm}wH<?9qF2u{Iqk3|i>#ei2=Hv)1MsY41&KH<?H3NVzyG%vQdOc{yIdRlVG4Dn;g?>hL-n2#80pf}xP)tN9<MfrN0&;vrtHYRh>M<Q!h$6>9%&<B*vlsU^1TG0I)Ff#dYg0ytxrCM#u}GOlL8QHJQDVkH%A8Dv@r+%b4Ni9kEclsEX7yOv43XypDy2RHILqX1pa*wMV&PFVbY1OWG*(J^0rn$kr#cxSl3azT$#PCkKN6QB?s&RihPLEa*<0Y5tN0-o9U;cp3WWec6sVp(aH*KLSCZ~q^%;*JrRwc@-Fg52i`>;=~&aG^bdid)rhz1y?p{S2(gbr$W>87VTM%b_1~oty3au}uIT%;^Ts`xKTlOU>c-4gNrO+zO9LJ*fK@e|kS(bh5XF)=${9WN%arN-P6(K^fhD{%D>`6evVD;-1Gc2<`{NAU0%Z}V&V#&2iB;Q}uHCW)LyCc3(d2l#$rRy3>NSaFLgFly`{JEc*_U&aUOQxzO|L+|rQVd;$je8$Rov3a{1{08v`3hc8WQ6Yp;|6n0WNoCuHSygus?HdJ2g|$kz`Tn9D&I@T$5<4>L!%;?5n9Iqw|R$OyHUo2rONC>E#1Sn6j$q7q40WEa=br+CZzTA6@7<039yV>G=9akMj)<6ZT2eFc2{ARA6cd@!y+CxB#n)iXVI|cqB}cVi4P3^ZhW?ANTtBkxwq)3-y8|mFK%}3L&lH*f2x`6U$e!#p-zd!NL_=4ruT>l{Ol5^@urNVpwg;g1n>61Wkr~d&b5Z5a>FnfEONCq%6%z=%(Ijl`$gk#0`&U0B$ISWbcgqb$jS;Hrorz$9MlT#k%A4GaF|NE51DV&$2!YK^!vFe*k=)ehu7aBDY&s&TgFj2b1Gc@4m7$^H2TBgPTNY7z}kP^|)|3{?AlOFh;v?QYy|;>8vIzZbiWvmn4M^G@^GOkvJm!W5mNO#?_NHMB8Y{F{%Ngp?cvp-M#%or}k|cDf$U3;eBPi^z)K}T5k!>;*X5H>WV!B+-!>uO$B1tA$Ri|7D+^JhF4Wqfeq}J_)l4nGtV5ugkKL6@=JaaX8U{En~6{ihAZoxr?mdgqScnVhpzJ>z;`GbzhQ)vg%^o-jV)3TXnw!DE<IVVc7igmIB}XEP|EEv$L#qeIR*|qbKuBEu_xRMC0b4fTmC#(!^)}Xn4Br#&6W(>+nUKm!4%PCkok73vx<E_?ea!Otj-+W<W#l7m08nsK*03B<%lZl#}DI~gK`oRjKOAe!d_?eDI~2j0A;MGTA!z&gT<zk!@ouxEX4}cU(;^KS}m}&nk!+-(;sFLhRGAVl5qdO1mVAsW=-kC4wZAczxn-a@!p)&_`pnGmtw=X&lQeRdnNrU<j6f8L&8Ns4QVqbl2v~i-Qm<mTy9ZvpA7A(=8!x{?4!zr5A2w+ms)<(v3FeKU6NlY@K87ahRs$GCm*7L6HwS}1bd&IpWi%m>H)M=#qB%6@lMUNDuh74Bc-|nM!Ny*$ATTVFRrNyPC|pWQiRno;E*NOJYQSZgc>Ixn(^03BK8w2Kbs|Pw_VL_IH>t+*}N<uC>pm^eOEwJix!zkVUlm##PlYSVf5^d77UzPl{qd;MyejZe^trF!^_}zB~mn({BnptVgm*lB5;!`hFxF{T><Xf+Ym<j&kDs_J^D)v_)>#nJq|~7)w=0wMf+1@Hk0Tlu)t|55#r?%GRGWdQ6=S{8$C5#&xNLd2Z!Z1hY-Ykz|<M=`er5)NVbDytR;6cO<d=dG+_I=hOSfNWZH06sFDu1M)?&j;1dPt4sc473>BK-;1l&fr2sndf?g$y=lmn!#^$yFzAmWda9TudRz4<8k}e`Kq*XQM4%O^c(T8DJwgEbfz39y14X&hPKTI%m=_&p>G`_(DCnBX=XfL`7u&xu$Mw5ohb!O;)v$@@f9jt+}LQ^c}Zf8R}p@)Z1Ivmg5cgPp!$f65{Q+Av0&}!d7HoSmd<?LUYF<mX+yLVgT{x68zWfBhq{NqNr{C{((fl2LO>s@97MQ6(s*H+Rrs7}Q6cO$~xmrTTFikqN>6U?Yt%wEA*7N5Ncv8rP|45(11VCuFLRnW3X1N36`oUttxX9Y)E9}M^U&P`@H%#^pl6xQ1^o6s(kmXM=<BbG=X%itO!j&y{ekd2*%aP>s$pJCV|km)*ri5BpbTYDAE7vI5SwUyBIk$|uVxpm*?hA%>nV_iB)c3+e1D5nGD_uKUWZH{aX_;p1bnz8gf?c^nAjPfCk*O2QNc6<(3|N2krKSmXn9N))FefEKmzKQEwP>65*FZP)7)ia^@NC7NhztI5RjX8vPe&N}&ksHM~`O8$Ka86861di~DUWMhbCyC05cRK{)chLIS`YOA-)GMDM)}LHnI^(V#^86OYS?PClTdogy;WEK(9g3=_FZfvobhd3U5`0d-#_yr5<#BgcB)*Hf*_EK$1r*AF`}iC-YK1cF7oerWMto1wJZju|7l?JLvcYL4?(=H->J6xI_@03`T5JH@#~<kQ$~>_M_mtKKast$vyWnBw(<eghYvB$Amb>IhSoFl_vYHPtq2S=Py%FT%I;mO1mUzy%tnw~lGeZr<rvpyKefX2gq!xL9;8;+^H}l-%#XgFIi{JUJt+(;3ws)|ynN1-RHiGsbcq2G?OR8ahFV3hD$oIh^pHu{L#Xku$;EEqJN{(bF0m2CRen4Bh4M$V4Zwt|8q5@ay?&f>ylPKU5CxMYlH{Z0K0f}(K93KdKW>kq8<7QO^V-6w4J-jE>4^!ylwd_6--Kxa_=0FMitItH=D3(H<d>{L~7qj9)HD*^)S7MhqsJ!l>I`D#3vpobERSV3Ioe=uSRlQ^EdA;`aZQ+)0Jli9e01pPDv9csqa{KOaXMLE?eI5NVZXizb>$bWOdr+K4L66qDUwpiJ1@`P*4Heh)%)BLsxWLeAqsjVSy{y3o)I+ZEG5&uyZC>Q$5wInqvv>>kUBe@l%G`1cmkuNYC!rLQo45xg${p2q!v>?~EsL7w1r6IwTpIc&V<Fk=fjTF=mk>YElzriszOMIm&*(1-3A9{+p0cZzNdF-Ivk}L?w4V(O3<WLIri|4mHkwGevh*Hq=mKO3P6vXJ7p0c&#muViDQ+=6Wy1zA`PX&tiV|XdU8hRIT75xim%x~}Q&eM*<$+5zRzeuxHVZqJ*!4Rc-uVqZTk)S#Rhm8$)i;o>SpPV-Z4d3axb?LWNhO3>BdMJib5|`>l$?H!k!yAz(H%$OR*&wPktSCL@={k)*XWLXt@^L}Vh1U*j!1t<9YC(j93{@QIlOj)%t<uA<;ZoYi@3~W#4bQ)SIcrjiSjR*&)em)A4AreCrryup-o4BF*>Kvzi_HMyf>!@**{NLQ6TWq`t+SR4`!!ALlAW)Mx*>wTO<+v#4Y0dB*5UyaqteG5`|`AEMn`GW|1rKq(G+qVuQ2%<tC<>hInd^*noc$NRs45BG^nX*kD%6nzb|PP7W1Kts@0D!mBny@%0KNz=e!!wA|x^MhTslcvT9WczQukis>+C7KWq*Nnw*N>3HwpzlZj1%R>N11Wfl)Sy8a09|ylx$`QqAwZfE&73pedGks#!hvsLAE@+WEG6a5hSam@RefHJjQ^lU|K?U~hzQ*98w6&|lF1j_iM@p;)Rj|dEsZHA^RFkH)&VFkU4zk?$es}SG=oz4ASPtg<;2UK+s1O?E1KXP;djA}s2aNs^mRZ>1Zeq3?j4}Y+xF1_56CjEU!!8Mv8&}qz>2#rCa%V=R-wCQP@Kmi5wsb&1enQ$If2UAo9|@2f7X$y5F9{y4)%*#@9YP5rqJoJC|3{y!2m}sYUdGB%@IRAi9uW)D%_qnjL*TEa&RX1UQ5u`Y^FJCTLb`etV;dToeBq!n+i@DSL;;BCW!8Sv98NUg5<QgCyur930QbtQ;8cbY^>c+-zHtS*7UZR89Q^<tMX-9`QK-mzJFfGi()A>Gq!2ZV1!o$KDygac5ljSl=<@IB%dmU#b-8jqH#@lD0cF4W%FoS@r<iT2+MmpXy1(m?;TrRUY)Irn7E`YD!Ki$o_`LbYKEp!wSgwKm;)@LeBgA@QO;*nN&AHW<a1L78Fm%zlapIp=CcN<3uA(=UZJ4X{k7J3QBjAA-S^9fUkK-(5t)ekgUM84BG`)G1pVTyBOeVE#!QDK<_#%xEB-FKvf>w{cpep897PD=~pU1yu#x1T7nqqSRzcq$1fWgISY-LZ}Fgxypk9Xh+HN?i*h0Re9)&tb+aK}E&TTOeVWDZnJB=ORJtu@;!gkuIR1N?qIj7)pqgJ*V!x`<d`W>~pyxMG7lOA^Q0vcQ%Lg-Ayo?>}?LGC(>&DqdYzq<Rppymzjba<N$t^F)F7x>HW$RVXG9CcbGu;^zI!qvIx$sqoUf&WoQnch&IF6#P2Zz3RiV@Cu@b9k+?RQQsh?0#8U}P1TB!ZzDVxVttq#q*!0kLV*jR-#rl%{;MtI)Kq&a%Cz|6;JTqS7{q0>0+6BdioVYC5j5X}a40Bu6D1<Qk;!9wNua`{<d-VbK*YzMtTf&<;Lz}#5%^*(CdCg4p8)!6$m`*CKhr_!x(b#vWspkY0;0s(u3_4DCWwUjNy9Wj)<|)(<xDYnuoC^_I!o~|M5uTaO~YZ0BK)W?z)~X%vjARTdAbvpteaUm$Nhz!4+w2?jbcRlD62Uw3mfJ<V8}x#k7-gK$!Py>MH1%>tu*2~*hg=q*cE;g){YPSfAH@b@p5uJr)2DzDuVg|Qr^CHv9JByC1JpIsll<T>fol_%|9echuM8;>}sZ;`ddwA3WvjCS8j2wcqg6y8=vt?lFRUCDMg!Tsg^R-ntLF;T_j#ebUoqvaYuDuO;`5RMaRNT3YO7W_zw8O52gL}q6@`b8Dz7k)P+eU%ZAS#Jqk`}^wOUlV7zCvoR`*Gu_pE!4=mET_`kjpcq2|o0A%i>-$4kG^*A1=90q1kTvlrM`#`fW#}&6|POq$t$(?5g>F<sOgB}|+bc~S{E`<KAXwDKL8+Jc?(Ibv%`Hs<pv~xx03aRnx=>wVoxtqv;MoVNeJ;*?~*gmyS1y9vJZk)p{j!Z9S^Wm$&P7Txs{ZFSsh&?=YQN!Y^IXbG)-Y9(Dq!#H~kY3p87U_{Q7nAXVp;i+&$~wvMPPb~EZOI*Jk3sAWOv=Zh*cQ|uj_45$hA)7C0Vv=Po#-)l-*mQbWKFbUi%ZGEBXtC8>vYL4^Q)2dIwx#nOGM%dtt)>qdyS0z4Pn0z1<>US(VxOcQOmfF;|h4-DH*CGV6?v%34P{#MQ!&b<2Y>2$bvs=T-<!J`8BzHhG)SXC60C-<cd@5eyTTvqUkJcFKj+ic+ZQzh^Oj|3sU{os9!}remu*p-zE!*9NLUrFf{f)!^|{89+j?*1f}iyG6hP*f&!ThC^P2To=!%}zKI3)`?YDefcgp#pHB@l^Xe17D{NYKe;eCI4C0*>k25I3L-XE#jx=;_Awt$nh5P#+kVD$eIA-Sr=?Ry<VWAyU_cs!WG3_^j7XdJ|t5WL4jG-2=JNV196j!^tBctPO2VqCnHQcR`-Nx(tY^6R4wr|*gZV6^K*&3Hhzh-f@wc0#(^mdeDw)AsB<|)>4i7K!83>6!LmmEK0>5F9^m4a@2i8|*{38sN%bP@gImpls?nkp?~YEY;QGV*p@=XA`|{4>9eP>MaK7%9;M{L;SIi6bVa0w;tna9+eRRKw2_l~CO(Jv$LhPL3xPg%@Elh9lq?hak?Zibc}?OnwWtm~^JPgomx1dnQR{m&FyQqmy}ZUJUV!2JZ&O5%Pe}8-GGTN|p)lFQBfmwC6z@ItnZ(OeiEFT{h3<qjZLb=Cooe8q4mkG1ek}zp`pBWwW2_s;$T|F|y~=N~WagC1l<Tc2a?PD`gaz+<Y2_o+(;Z(ulh3wSn$tXMhT26Wb2$hj2V5CWcpdD{ZsD#XI;`sr6XpRuHiKHoYVY!$G`%z+EaTv1;ZsX^G+8zG1|GwEzmcvmwlqw_t64vch>+IH3cCWAMof|HZKa_<jc~@`+r!91APm%eahu`F(;YRC*Jf!cuN69xM!vdP9oM1Cs`VqKuJGO-$YjCnVm?)wAj;w2?{UUS!0zv9;C5NT_~pz}sH01dIX%deHj>b1daigA2%Hq4EUBuRH{eW>#~WClTkzFe3|K#Q7yGp|<W1eJk^ZBA_TpH*ir*0@RA3$7$4pL<HwHBp#DM<HxZK8We&sl`_W)>zi6$_@2=^V^PGpK3ykDT34HNQhoc~jp-&lifb->9X8<h27%fLW4(#Aj?)@`xXvc$N;91AsInX+HO_Q74jXOW_vHF0*ws<HKIc+Fzy3R@IVSAo7CSgw3865K=_e3<7;$Erq+?(ba9d7+RrSIJ-PhNHAfqCQBI7XDH1Cn+leYK_{&pOZ=yl@VlDdf@m4ltkz!1r<pi(P*qx>v}st$=-*%eSr=imQU^Q}p)Tibc-SH?J<Jq{RIXtGS;1)Rir!b=A*hMwh{#hnFKC4m<=4qDURnn1`Ho8XIiX7Z%|W6mem$!Ve9B^B>ILvTm|tT@OCbBmcQLbGP<WR@{07Ll&{h&J|x4yWaWX2sSY7EXAY&UQ43<RK>h2!IMI*=a~x(8tQT=B$+k1KL0YPhmHG>nPa7fzhG}J*b<C;pl(-b7fRdmJB7CTAc`7dv3c<S>bT_+ieVFG6TGlCCcY$iX#FF@VQdc&2yEGBp#ivt=v1k{)kp}#M$LraW^98m62^^-GsJ(9qRCq8f>>z!gmqJKn~+czqQBIl+%I0;pb3KxLNwS55)evG1>IKtB#v}#-Zs$Jew{P13eG^MIP!eInu(mxP764M<|7siZX{4{GQ`vPS!}leWsE*rLIi4eKO`1>$_@y!trqhZ1a1Z=E4-HXjA-^AV0Q31i*ORCCWFZay=HC<9c#cgpVNWwn7yyUHX62oo;k4F~-@X1RQ%P46zQom%`rE&&b6@z138p^=k_1B_X;2sdUc$J~SnrOaKnPMlVkz`~b*0l{CltW*W$tn7Bjo9Y~-kjA1G85`5rv@$t?PAZbWW>b)^DmF=Nz=`?atX#G@mu%uNkmyA85gjNnfgkwz%$ijhib6<^Kxbr+}GlSpriR@m3QJD6d!ubJGE_d97C3}9K<$o8;<lX#X-J5zh$%poJJBdjTv;%Jk>yePicBl|SjVgA5KO8n}7$HZLH9)4fkGYXT?Ic_!1|h7uPDz<<#UB0Th%RG})lgv+x7E&%J9VdJRJy9%_db3#o40$+l(NkK*SebPdCttzK)_(_2U|F<sX(YeU`ng~_4)iEJpc_ji|zWAJ+n+|Wy_!Qsl8%Wfv<oJqj0iGK%k}PYvJ2;Usf2rlSQRO=%x18IsHPXYU;|NiJpUqa4J7Zrd90yd;+J`xV>BY*RYi&M^g+fL=-ZiePA{p{{Zh6%mwv{2<erQJ+D5q*iUjNmbY}f0Vrg0jB7T48rR$BvzDdbIOOM^#blh2eF`6*hv=AEZ6h+2Fs6{%R7O3Q?hFfIQ5JeakQO{D_QKwDw%yNnZUXj7+M8@RcvXV>V~8Ll7}haD$0J}Nh#JBeVlrOoL&6PMiC_XtSMrLYvmxTQ@VyC@wq%5-6)|U<uXG%KWdPY~Uq@%&Fzq2~Jem-U73T@bz2ESsQ1D-?XFtKO>RZPA6ObBE5?xKbs=3Z>gOw{I@AT|EXD&iJa&Ya?L**DSXSJ~jA8jf-8YI|P_E<?(Ou%&#?jPMP2$^0vTX^PyCaPj!ScEORa0xgDJa<SB^j&B;((B8FU31W!H>Y@RKJ-ah)MA<uFn2xBJWGw>Rv;JZLiGg<c5@Al0X8x;yOwhjW>~6!X~JuFV%RV?Hhmju7XLHj-Y5@yVVkcHQ%9?p`>U4+#^tgG?`0PlA?%?{dozocpP?cip}E-!pX_`gl`^nANJGorVa0uxYg2!x*G|}PPZ<$jUF#94Tnvd<WTx&fjrobVrrW>GMwWK?vWLoF<`7SQ20dTZRD`TA8b&S8o|9Ca(ue%_ZTS;fWE0w#Oy|huDpGw6%d_g8;D%1kGIpv4QRBtcM+?EIo#9H6q5Xg;qnEi7C*nJJrTEBB)!%DPb|AC$#eW%&?>hHhC`2o<?T+hBG>h;pGab9}=CPUy`!!A7s|pYnK!k*wLuP6UZLQetyAoxHq@--cHg&p-V?^(h%#2N8{52U#e-G$eX+GKHz=skMV!pYC=AWK-wjSW!=Q=-hsJau`k7fdS2eNqwE`sT^q>NY=l=|iNd3?U%BYhu*`_Tka06hO>?tyRg`gfcG1ekoE!OL(Q`@Y*dDEx6(<WV{aF<|S?y%#KGu8B`rEHy*d^G`@_!Klw|bQwxX7rXbRiGn?9+<LLY&rd<?W#}o&Z7USfvu04Dn?gSIBbOFSazLftsWk%%F;L?{os9~OY*Qlkr<p?8t*lt8h{_}ETa6Ru51WT?qAn%UJ5Q;AfaYf1Jli}MhuIcaM8XT{jQ0QQXHESZSJ-}_H40jn;{}6s{C^UG0e1ggt|V&Z)A~_%Q;)nJ%+Wt>26SO@)65j71&l%69bk@$Z+KlQ>42Y9Krtfg9#(|K5JHT%J5kCH;m#)a^&vV{4Y>Or-|+;1wgg-NKL1FZrxmt7gxwTh0h&pXyg1gzU~4N|2y8P*28YoTDtm;)@@=9CDM||LBuB`6=z$gU7=9eE;7cb)-7hIGUh~!H$w&WPG0v0Ww=b65QIul_iH#wsS?aM?GhIMb9u^FMzxbvw$0gR(qD!7%whxY-C^R<*JHotHO3JUi+G|nM2-|P)5NH*ya-jGZ%7PxBzMwJdj5$!|O*5q`qww*;3}!0g{&s}$;PZQ(T;yo^<EJurt4zdzRke_S)Cur{v8G^P;qft&(s*BH5f%ht0XWbnqXy-Y=;UdDBp`=Fu-oC5$2rqqkCYxe3w-X(QJnvv$t=P&p9(}rF#e5$xG7Ib%o23`3v>SzVtN@s1+9TZG=56;hk_tvNo~7DWIWTw-P>)g_@^EQtrrWn`5Sm%M#@s0J1<zYRu$&WUIGI!cC0WO93#`Jz$!6ienVw0!TVQ#<cizCEX1J<dYo*u{7h;^IdF`DeE2VC@g*TY=BG*kh#BZn=m#Y=iG2tLS~;X5?h<FtS_1VQXftNtz0c=Wo=fAWm7M^wAHJCvlt|IiQRqE0{nV;YY0ESPdgHz@*6JR?ciT_48vAtI*c~lwYcH8GmFi4v3@xd7_}y)G+B88G=!H>!!ku%vw~HbcsrlEG!vd;#w}D1EUHBr@*D^|}j1g_2$AnqmYMz|j6O7*!26JhN-l~fEto&c4HtF5#`Nh8CI_`2ry(^3Qi7~`6F>@5O?b8qaD<Cn0{@l-;<Wfkv!VJacEI7zaa`*3Qxo$Nayzn1G`(B$}_pbfg5e@f95i~vMl$DUwfLP*b2pjOxwn?d%Ggw8WiwvSokwmkPcK5xG<v@%Nxvbv>Bkn~(g<ukUSh9sy^2{@QElyIR=lRgxwY~!b5Qc(6i=LJAfvUAcM*;s(e`g2GTt5I$3DDDypfh7I#G+t<#Mwey%A)pBgj@bb95()@!C7-dvH3JsXS4%(c}#!6S0N#7Ud2SK47b^}X7W6CO^XAtQY9bstQGSc-&jn=C)`T~6wO^&@QT%A3k8dGHp0&Dnmc9(6hBc99w*nw=JjcBnI{xZpX_t;#~q2b{u_UiTJWoLaFB$EJPXJ0G>qgGip;^!3kK%W*ki#3Kd}>Uso_bDk{N1qKt?p|R2k=jW>Hd0Pl61(0VyX`c~NcwdZi4-mek6uU|Rz-C%TvTYbR|2qUOtSQ$#E>g$$xbJppYzi11Ys;kSxfGg$u8m3roOTZ1*R-9(wOL6=w2h}*2nbtcVpXQ2beKk=m;ddxg?4@yEPypUws(?pBS6{NZ`A|yYC&-7o8zrqx1+6=VZH00_TIxdD7$2QrySm#y8(Y0U9gdd)0trgj2OLG%$IWPS*jx(y2(r}O{p;N_Yma%2@U%}8D=_v&Y<DL83kAHlU+7V&r*TY})$dNmle`os(w`CXl5FlCb1*n+;miZmCPa$q1tyUB`BIEGAvZrH+QAbKN@XY9_P%f@|(?t73vfqW_Q6X7VWyb(|XmXcsS`-_NB5_6%6X6P?RTVd)zO46n!_nGu{Jl@_5j<pB;k*m;9r7trfF!pMIVb4v@|*spg}eJu^XiFTBgU~7$BJo1kI;>rDhsmRxY_#^UmzS`&<dNVx{G3W6=|m>+jm5jpXFb={H9i}J24+q!9}cB`>aCEB0~KsXb&%QQ7e;K^XgL?s_a-5>)_n?+3b%H1~-*HJU)=`ew+_9ee|I(xEdp%5;OHLG*;4Q3FipDm%bdE9?f2x{rvN9XX9m_vfQWx)m16-R9Medt^LA#E>TsXWG+TJ_>OeejvaJ#zC<&o4pU^KkfDeP_t3~xky(bZ=hzT5usx-SpriigO9AS|0TT33h>u};#7$nS_S=E>JI_`j4nZ6%g^>*9c}tcNP?W?37pZZ+>4bqnaTj9OjkeZ*M5Baky!$^a*2D)8G(XIgx0^IX=qcd0-}90>rA(JKps(z%6Jll1En&;UZjV2up$wA<tJ(zysTzeF7&x})6^stHb-T=?7FpXE|A)MNPa^*Y=JCe0VR4U-T{uoqJ@*qBQaSAT@!Il3jN7^G1i>Og!y6LN-ToxY<zLd>u_xWVkAHqi$zAxU`V(|*eUZ--=C4}eLY6CGZB`}QgBE=?rtT5g$@nCf|J`s*D(_*gsDo0sHhr<xve4SB2wwgDKB1kjl*SBpeS5KoC0xpEC)3q|&os;e;xu0FIxG4oBf86<U8<DEgN-NaheWr<NuyYSJ}}BtqKo(ZD`g@x<lkQbATOSy2~=4Sh9CM?><9RYcZ_%(U=pW|VwOGv@<<FE6DQEAu@!KGnrsp@CuqT%`a&HC?sz$VD<lfA7(umCZ{}!I>jfRwSn}n2+mFPbuppq4A}KKBhpPNH(XO{kU352RpreQgh)3f7W%6|IuEL^#tl&KXEDpTye($0f*{9=nTZT(~g*5wS8KH*q)496ijCvm1;(|ls66HOopEmeCXY@9an%_y2fJR+|evUMd-UTKc<*t~G6|d@l$j563H`?mYGZzF0RMZ?lq^bEyT}!KX`eYLcaBhj<H3ela*8#PS3r3Dc0*fGKZ1QLPV(AUg<g478cU%>u)@1>U<)nl$4Mm`Uh!^pk0W?}OjGU&p&%P*|Qp7Dt`z7D#Ex%zEjltI&;z%bhGd>$j@u^delu)&uuVMvJ4JX|B0<m-b2@o6v_p4@VpZ?w=U84O^y2p8l8x75R+B)(Z+3P$h+S$(mFiys*5q<CbSDL}9M#qw^k3i`lgbqGv8$9JI<z~%%e~o#}vUPB*B-zK11zAL?TTnm`2(JfeC;Co?F}Ud;J^~(`+Xa;&Hc~%SP2pZU;;BY2=ePc(da{92Kq3AXKHFk@(%l?EB<}NmwKuvg?>ZlI*E1b5TQ?Z8ji-Ucc@F0Eozv~DL0bI?nz}6v5cltMIeZj8^S2=>zqr@`@I;Mzxax{#_fz5cDxfuufCc(rFv6LhMQwo3D`w;&mQ7Q6;Yy4hxKpMiqu9`KOvEEy<AT3*L#@*5?i0ZPT_+UVOZFsdj8{w(CM2`CA=#V<#s{u6Ep%w7uY%4|n=GEeK^LBC7mz5Eb%Mr-)hW}{DWfh;)^_Xg`7T&FkewaaLvV}7>fv4)-?L*gm-@s^-*Zx$RHGZIUS+Y6OMam;R^zs`_v$g#W42;MsPyoP`y4681XMthm}^s8(XiG2O3*t{W*XI}9MR3#kUzUutURdY)KMSv4n&E;YFx)Hun2W?jI&%ZreTtQ$C6x3Ywb_T&%Us`<U%}SpX@qTMx}=R!Mm;Z(QQLY=EY>I<_G%qSfgxU>82^T0$rZkFab+ncy&bEaRVQsctqVFpI8nEmFT~G)IlaqBA=e9)r<4fzIN|kB>MD6W(3Mn)>j3ie09a<s_{Z}{KKQ$*rX*t;%w8W_HS1LCh{TJ^TjD)m@qP5!M@{M9XiI2zm=5pBM&lzWHo&3+a_8%{`c5KE)wR`^^(9=*yo8CH-Njuq?Djg_pd@^D0VNV{x#8@;8~5ATlMh}A`1lkECot8@E}-_=LZWu&P?&Rlw<I-8l!z&f!KwUEiszi2WQ{f;&2^J4>1z<J6a6Jvi{br<ah&#zSSG`fkH14fPX1`!}iDtvwC2MmJ)*I!nXt*ob#7hxHH>=%(gDvM&XLE4IVjnNr7Wy9W5bFICVSIhi><YZdmqTpvmAuOa%-bo;d9+!GZNTGR_Z#z#1@y30UWr@!^xSwWSU}d3a0#i}`VgcKiz(I$%i%_3L^GrOcu<E1i0DS)F6^zqb+?<N2gZ>*X?}`(MPgJIUio!Nrurd4x4$>*jX4)u-`P<GOF4Jaw;TK5LD`db4AhgqD-t-!1$E6EI3*<Te=p*aVfA475lidWLIg!V`^m1){6Tg2fyTO~v}pMCU^*^7m!-(;8VB8nc$&2kC-ut-5vJ*2lYBhIn?J?X5lzN&u9Z26+fW5*AhJ1255VWIp=cmw&&U|JV(Lc%fxdOKeYk0obe>ihB8p1Wnn&Eadh0cO=Nq;~c_^aedf0JK|8KOTOW@VBDKR7S|s>I|CS7K8xmWa!zyB9~<qJK23$hMt?DPWwcSlXADDJ_8jM|XzpczQN&6L6{CNKz;k;ug7#A~BF;=$i{OP5z<d*8rj4JmKwZ9pAf(7`%SiBM^tVuMp>~fg9MetyVMvHpKq!(jtyVNN9eNV;c3DN`C09?=`tBx10FdvB&)O4;(aXiB_g-6~kYxXz41JuOuM{YhI0xUS@m&evh#sdXF*>@n7#@F7=;*SKL7&k9xO?B(mV66^h=VGA$P4i5_?d~<j#5lEGWe-gM_QUjXX$)+F^UdnD%dQh5{xXiI@d+VE90q*!?^)rZ;R6DlhX+WXqtC7M8Jp_T(O4F4x+n9TqWV%pLfG|^Ld$m=cq*WDnB*|6u*)%NG@Y4*+EqpxY+}xLELi~-*pMm`3hJK?R&Af<q=nmwMju`t?Fo26!}2v%LUZ=ZJX0zLT3L_O2=FYRI&?za~n9}@Le1iEO&8G*GMGB{uyHbToFx8qrPw_9JkK>0`{W#c;x(?-1Yk!9aOyWKVsX>8AK%4pf+G;*7#+U&?`cpzb~AH_k$EGYvG+i=e!xx6`om>kk+P}eN!C)Qt3mTq>L9SKavD43wh@8n%l*U?)<|pGdy)2eB`ekIMz92EE>dvQ#6c=-%y;Ts+b8b;)6F7zgl)^Wn=#(<kyvyr8n4Ur~5nV9<4hq|99_H-U{Ou&~Mv8DbDTVk4>Mt>*lvoVVEo=bTr729Ty#M+0Vq~s0+sCpTO!H8yi$>Zs)=8oiYRaxu~3(`g+-Orh(RBp}Gu+Q8b^bBY&0I$M<0@RCe3RCUEY`36&FrYK1>4BJ<m70E6St+pz--EJ6lY-`Bo9+UO?n<O*JZN$~wim!0Y}$4S>6Mboqq>O#KXGQ!YpgF9cUzFO3L)44I7Rz8ZA`s~t<?+i~WN_~>2$9J>gRob3ch)re8rmF_i3*8B0K6lgEO`j9*JQ>6zCB3u&vCGh1o;ticScM3C8icZEx!f3L)_(qp>v%`r)m+5~PR^qVCUc0n>+GXetFi2G$zULh#x0YcnKiA3BI_5nfW$RM2=>Q6fPMx{BwD}M?_PiKguwfjX8a`Bu90r6a4GU9$Cq`@W7+{F(TM^?A628Ab5!kMJoXvc_JM3Oqut)j+|uExfs@DxP74P7hv`qe)nsXswZi=}g$Cn+Fy`0<QXv1=B=iZ>cv=E<vKb2DQpX%#5>oU1l{R`3Rd`-VCG@Jll)&>7fk*MyGTzDdaWJd|z`CJhuggrxmi!HdV?sCt5i&IlDEI(UQXKPY2G@_(U-dZ8t6%Am{Ej*H)64AliAggPNawYqY(W#}fvC46CKV*w6vh8mLrR}WRG$}!)*;wQs?i&WdpI~zdHAvF-D;g%F(<_lV0-KQ-XABGO15W@NddC0l1LN3EC_=)R5Dx%uIyYf%<)vu)8M!DWLviyR|N@0U4QG={E>DsYt*a<^u;8L_Bu&JSW;Fj;%DUM%^11FKOME2j-un0@oAr-RIdRq?73-;o;|7n=G2+qGa5HA7OnLpjY$!YjvQ|9M#osK%#4O@&4!n%ieN4q{DGj#j9q3GTYvOCSeEZr!K`k{!;#Ck(&G5&9&X%wz}^;v6oXWdgA!<prO^}e%y%!<){g+by^Ld*l&HrM)(5e`QPsnM1TW(WV+A~(2IFBh7#t~gPk=i>%t1G;)nfD;2R{oiEVrO3@yS#HZQo(o8k_}ymTJ;~v4N~$q7q_u+Ah=0z8k%T+On^zyfXljgQS;_GoRkol`q|c@|_)735+Nvy==2Q*J-gr5kqhHq*DR$dsEzD{v6Y|N-I0qyB=1-)CgoA{pmv>XF0FwwPyw<&g+=HSxt)sGO5kcTp3N>0M|h~5m6ADW}gv(0u#+V<f_>(tJ;%JR@3Z9Qy}Q12W0VNIp8JIB?^v+A8c0EMO|LS1I5>o!KktuZqLR6U^nLkgg7@1oc}*SI@@T>d={T2I`37%eSXYcWIYz02IJZq^sopN)2`OcGqrIZm2k(s>paKL_80C+ab!!5CZPxoJH~PU!t0NMV|e-lekgq^pILRqGl2UGzLGFid~X{e1Eg%c(DaB1MkZTm+7a<SbfoxE6{6g<9k8t0(w5nYHZm@#gm&d0_?wvGrce3!b{M2wgB%M^43B+S!&-N?-+;_W+9{pyH^Dg+m9K}3y9LQJl7d8b_C#n3KeLm}ehsRBA4}AzY!z1S*1S$lB;BZ(9=85T>a%=G3F`etZ(o!X4$D2T;^5&(3tZ+_>!z!9zxUac_Tfq}dyOhmMA><M`RbV8*!)?Y27x%dRX|b@H^bj$j8zm`z`O~<Qil1&=7dvY90BhUn5f810a*fJk*Ljbl9q)+Gx2{)DH~O=I9qc>o~`Bqj_SB5ajfK~m6R`$9gs&i+cnW=WNBgy&j7$5{7U!I`e!iizqViv{uf>#JDK0$>O{=}E_(x0!GBIowL3Z4E>RHT<?n;PKeBcSa9!~Jb*b@zG+I2wTP^O&m+SW|fL6pAt2iw063Zr-fz`HK)Fh(xyc@JfN~;LROHn{`Gw`NY+8Nf{jC*G*2T#)2r(}9XMAXFm=jEByT5J0acxw+cG4zU%dCIBz2%Z6s9xZWTp0IoPW;Uj@1uHGD)FLC)UTpNo3$1(_G%^okTm&&Ds3h;d{mfVi3d%#h3X=DHg9Iq>W(nxzZc~;av=jHb8vM_Z{ONWR12?1U<tiUVPQ#5$9%snt6Fl+IaMVh^1Uu4t5UFSevC*`>pa|u^=I16t(K2ZI$I8XdHVR^=%bRub;;Ywxo3UWTPLsSCq$wK2c8O`*ZZfc$AvHm^o{Yx>iNsOz#34y@(lI?(MG*Q_nX@DLtDfkTJHC8ffiB(nMN|P=CWa$AV1~2fp-$}8%5<KVIDve)xy4ZmQw%(0k%xEEhnaK;;gmmmVl5|z6|#0p6aw5JHJXX?M=$<62zO*pEFeLXcJ`v;;a@r+qO*wIz<BgbRUF0b=mfQb9zD;cX%yderQzQIg&1Cn8#=~X@>sqd#VezCj;@vroa7sQ>@wJ7iKm}j=F5h(#_v^w*-Hgcpu1c4=-Qzo&{n>q-|&DLrVKnP_dz7;syIkh+V(;9u!Kr8QrU{dR}u3>|5eddf#}E&16<RXWA$8C+XmH5YGmYPW09<&Ms}OvZ<FNP2p&uuENSKm5bNZt)vzOxV{}G0lR}}vieChSEKAODn~F7HY>$vYPxiM_9_LUtG8)r>`hroA)*c#~F_I}-Gw?5eJeQKE#7H4b8>PBwx;}zHyQ9@XW)RjPiDWwKFNgZ@QhoD(m`Je`Mn-8Tltev|<ej3@^ZjwR7Q6X%7ra*j;q+-7cc?Wu`g#mn73KI5%a7tR3yBKYYbp!uJ0CVZpqy;zEM@3{|NgoENC6OnpW+9~wQFaVdRZ_^&?|+Hd=5p^=1=&Y4cyC;8t}d;OCVcG^g@!q^KOA&sYt?KP_F_7&ZC?ip12iQLQv4ynrPHg>luvCK1qDvpKdVOHQ>qb*BXnS@Q^TraiU@6W%<^%(&eQW%ieC!FnDx^uTSo`TCv!N3d5pO>IR`w9;sXec>p9VIvI4d#=ET65kT5J@ztyCh|TFici5cwtmPH9U>+JL<gU0o+k3BI3rKLl3Mv%bAy(?M7t_-=gZP779=WZ+UWL%@$8)mZM%D_n@H^vD;#v|<=U@NyS9gQ_ELYKN&U9UoDkChz#An{_Za?_eJoe9g%Zj#d9RIV40Sm$7N2GEKgLsbGjNea(Yh-Z+p4k=}CYp062RehaD!$1EYvb=hS@7Ag@1nfk26_>oDkEt=5>)y*EwnGcw6(#8%_U}NsESDo2Yk7AZ#=>yIq;{YNxc7d)sV-|hPFh$G&p1GcAN0R&3`ExL>noO<|@dGVJ_XU<P+SBiAg4@mS^2@vI|!2<qU`fluXJ)(|Kmg{zxg5(Sb%sIKL(~Jc&ptsaMVe1Q9137HS!*<|}140=e4KCJ2_7cF^*>G8L6VjuN0c8jC_pwTsP6if2JO1%m(gr_U(e-!r8E4d1^`**b_RXFYN1#psE`YPRd<C!$Zt(Rj!=bRre7yQG}(pR;I1faQOq6-KXwR#Wn87%tR*)I61*Toaqu)~9!r2352eAOc|&l~;*{FP1jo@-qY`vf&$IBwf8`%LbgN#E-fik^N)t*}<n-3T)Y+4Q-N_p+Aij9zH~OOOFtu%E<pb&lp5btO}CIN_Ivp@AT*f`leXTuC$yPxWmkw!d?|EM%xhXyAz`mO(Q=HVG7|<uz|av9Vls1MpDyG;|34gl$UoZ8D4lVO<7Iqo4oV2N|TqAN{0-ze|uSH^25a6)D%%q6K1P~X|<?Hf1#(Q>57)iyPCS~-b84Uk(qhEtXc#@aJ6Vx_&`1mC+wBBWP-w6-B28@0j&b45=6Lw<#kt>+)QKS>C3fsEvK5KwdrlRA#g$#pleQ<eX(#V8}ui&?qg<T?Ea*8dLJ1(N2Cw>x&DPPh2Qt^Fk>$ao4%IxoJE|*iVI@-#6*wQ5N<U9E=n`cCjw9`NZB$C-Ovw71tNdWul{DaShs=BmS7;!_ayDz%C<@{vy2TaBuIpNKpp5yI<ymURG8z*{9M=Lhl7xL6G?_tr*2{>rHKe*e~R2RGp9=RI&q-`h0C(j5Mzr@#)}Vu_@iWt->dZr&e0LiHe^5--C-$3f@JvvF&452!naNIbWjg`O+<_dPlnJNh5lry^zDo<@YJm&6|e}bz~@b|#wcO`nmX^f5@R2ga@$KxegCrQ=`4lmS3K|qsA6df2yY~g$m#1U0l}RJAz`gau6;BxZNJgqoqPm5_2ne_y|0uHXmkx)A|0bV68GV(ub2VhuK&vxisbQueJrLq4~{@(Y?vJX%ay7fDU~}CF)VkN(DKz(>rzls>3h2TTd$I&T(X1^hl6q0y&ng8kCrok{w%r~XzmjJOhu46d_%QSxWy0LGby276A#YmS2g2^<1j5YnMbUz_x^)|<@aE(wfNBUD(>nYHr@v6L|)dkWBSb*Yl1xHV|FshYfza$K{iyOOOL_buM+c9Id@5YnN4erYhndshU=n@+;0y@HZS-1eOEFm2{@tdb!_lr_T^a5-|EWAAQ@ALiC~{JL%nh_?jC%f^}gNJi>}H5xoINf_9I@2CtAI}B=GcC*Isggs1~7z-36R83^>rC>XK1SvGy)b;DfifO=m6hkdsN$A0B6J|CGv(;^I}XZG^!IJhnda26S~sb}apfxWqFD9Dw;dapyM3L`~$3Sq}){DgojuL38f^LS`)+yIQmg628p|I;9@+DT`g?*%-Y2^-b_ow`pSd87xSodylSjXu2HrjNm)XnX%zg-i$|#9EkK|4G9}QuLzAr+FqCehqfa}=+%~(^l%cplUY#&)Ok=tFS!+O5IaKZjQ%MtjON_gpuo!sgFCr4TxwZLLvJHkybUERVQ|T(ZM$ztwg(su##;iL9yg~5HwLr{Qw2+cmj%3p<DWR#b*=2ZEBpZhq}~rN|FY(sqMe^yah;q-Fy;ZP<0H{;Y+{SZQuP@f5n047L$&LYzFXH^YZ9}{G!7>@az=|!L)I>#8$TFTm+n7F6x37lc4JXt%b{7k+5HlSF0p{OJaY+!-!U8P`}&ats_Vp>@Q}E)AG2DI6$-$A%oI^$_;a@K3qX>~g8*0f{>?Iv#DL9|u5spDTs-l6U(OHP^!8nGJBVTlDif(dJDeOsHOK>~Q18qYF&C|kbHF|xF3oa(`R$hS(-gI|IH=JcMdx}4<-S(RGY{M*+)96-h})W!LK{~+H;~!Xat1wQ^BwbP;aMPopT_%2mqx6>YgzoW!QvzNbLo=Tf5nd6RgJ(o1}pIb#XlMf8s&$0Dkd1YqdrHEG{|I!Cxw!sKKFcK5v$_bMV!nzEAe+90lWNC!v{SH1jW&mWzbeZ)=Ch0OY2*Fe|`JtnVzR3?e5FZLNJAZDs}#nrbO&BpSnYw2C*(12u&S%WLm3o`ivO&XOL}_VS=ZZWoWAsQX&(^#KY<ba4`=cHNF27-l@MPA?Hn6Iif1mU*g%I0YC_76^IkBzl$TWOw}B>jQlYQJ^lB3mT%k3C0CB5NZhHQtdSqbudJt7kX6a^eMX)b^J+v80Vs)XWc@t_tX2l)AcV=5=UWtqj5-R7W5fkT9+_#Rtz5TLE|zx4DML0XH;F)N&?AhT+nqmBW(jYl^O&}Dh)c7X?Zst=GC#*BacuC>M;Ue_i<_?yaL^#v{P>Hn7h0x{^t&^1)T1LDk_i$kZV-_K_~3*hxDue%XL6butd7obpgY}yZ~S<BcU3ey{3nh70o3=3ewW<tvxsdV)l=B8$}Ykg0_9upU7+nrrroFUrH<#rxihG93vK)1X^B6_cLL}rS36zd{D)XP5aup)B18B4T4k1DbaLZ}XaWKIX=0)AwXo@tt%zU!P7iqc`gPn!bAeItNdr=^GW;SAKDVgzhW&9`IW9t_5d=Og$e5!nYg#wN{;l;w@LEGuQn9{;ud&RrQVfa6xa|WZ3vW80=cw?qdy4?_sRqD$$Z+h1U2wFL`FMvI`>I`A%dbX2OJ5KcEJu0Q))JRd)eVa~DDI2P{mr4zfcWm9gK2VGeuJi1{NCsYdZo2U+@^VzZ8b*N5b+fv0XGdF@6U$jAoZoI`%L&Uc@A^evGfn`*2b7#RQXq*lKcnC*yN?KAj2$zgOWu6*J-9&7I-OFmrd7HW(O5D)~~hE_;&p*B!f6RtRd6laSeo_nJfDvf<XsrV|EAShYhMAoRN)XF|u8NfjsdT%qGJZoJ>dmNHl5C{7RcL7{z2)1Thh`r2Na(n0zerK)ms@&vUSS0(qB%eo6pC`XZhyVfk6yz>QWbP)H4u1c1n|2&m3W$4@T4S|)seLsYi+L8_uR_Q=f47$`OhlYP^=g66=bkLkS*42CaQ@Id5W`}R5iZyV#MwcdGP#sD+pQtB4mDb8Z6E*yh48Hk@0h*9}~AF#g7J~VfNZor)4zS4b=dU7$}X`}jLULO8pwPI@J&=)J11XKn2*j{8|qizRAbx*A4#&K1}G;5&#<W^O%yQNw8yV5wmU80bNXb1jrdDRaHsZvnGE9%j}dtO^R(k*h9Y;mAUJW24d+v2X*UzOP@7oj|ian8l!Kn77cTyacn2BV$LT-l`=OBvUGG8J+}H|0hOBaW4Gk@W?b;OSx@%7`cZ6;no0_TsZa$o$cCA~b|tb6-8E#<qI*A`||iNC}_tH&iUh0vM<~Z}?JQu$}K*mniy9G<rO}%OyJp03d56sc^S10s5E<&)>VXQ9yEBp$UYY5@O`*-jF3UYASJvjcRRKOH|SIB(d*ClKt~U5sMbbAsRgNAK&O^V(slDJcxL{qWI`sn1BzU&Hz`4CrZeZvr~B_)ufS^z1NI(!TD2PnPD4eg?{SUW%YVK?Zk*aM5kCzJh@l~(yHz2_{I{)nm>hyZbw0rgXkVuXdDHF)l%fM@MT_!I923E9HRnw8H5D47h`;1in)3~q(koi@hT+so#B;ERj`t+{IW^N5DKl%XLJGw)7beXj_0occZ-eLGpKKSf_MOp!hfbH#lZwItz;WLCfD-UFdJA_dk)$c6CNK5fH9#oQEL4^u-L)p7>@AcTpCRARLH|1!TL$mp*x$Q2}tj6gO>0U*#V+r#?-sl+td0Iu&mMl()UD|*}k>Bs**1{F@J~psQ!TZlC*>j9_B550~@Ou;KXL16S-Oz#QfJZoZJf6TUd@I_+SVfQY)1-G*!r(h80%^rG-K3;7FEg5uyH(m6mrA(&k}|0e+68?kDIUW`0%Q8O?lVM<SJ=;wB-bCvuaUNJ~(5HL;b)-_Gd&6JrwyyY{v%oWS%Bg%k#z$Z^j#?n#0?vP;vOC}q=&cchY*#+L|O0hi|sM~yppe{Ii-*24AFZo9B6tlr2s{VB_`yX4CC=532^Wabze@&YMYE^W1y9U;dxKAL9z<ks3PVqrGgY5^1wO}@TT@hXe^^^PsUY+H7&OD+NwhRgA<c)pEb+(h6)R>W*w?9TAKj#i)n#W*T(BpsWD^ZIJ|Rg=cmu!U}w;Ib)hIY(P@0#IU^dD2Ru1Fc9r&L71+dr_W7M`t=PJ;u`aEmGxF>sBup80y{Zza*LLZ;+ZS9`g=hO{)HS+`xhglhWuSBo1J{C{KdSjz&IenX!YUi|KI|SXUw2p=}7yU$bKX-byYeg*Q}OwcZ1))0AQx5#&z5zw#+MdM}>d^WY3JvhhzjWyHmr7R)qnX0Nl@q#H^!&6EykbW7&um+)T&+Tlf+_gHq=ad=dnU`X`xnp4Jg$>lf|D%|dat)Ik2WgN#GA=e~bRsdb*LjlDiGr?BVs&8!Bkg}cb14%ZDY_a#Hl&|T`7x*&jAVpl|Q|86cg$>hjwz4bG`Bh>#Emw&9ZoIPawBaQzPO~K}qYwk8tibDb@l_<w<8N;*=eybJEvYSonUtzghMp%b!XZE%aG)V%mt*UsGiij^yUY_TMgl=M#-RIdk*p_4@I8}d(Q$Hu;sljgdy@i3rgKHuApF@ddlp{6fXk{P%93oo2ZJ7TBC|PkL4vbH?lFVE^v%_eZ5xj>QeKJ~DQZSR5^)e>@;&zUq}!Pb=;BYw%__uKS8&OaE5G2-sU5#BC_e`j%0+5u)b66~qoPYm&UZgwkrF2I3-VI-&_S`l#9WJd+=5nlO7u{+trfLyq{{ya3`<CeLSxZPMTS;K_+|0k{lfxU1fcWV7YM|z6)MHyU^6(zGZGyNCGitteuWVucPFh>{b(_FPU;e9sRzoJ<R~af=2?U4wNd5=R6L56z5-=i0Pi_W5GytrbRQkpuw2|i-WV#wGmb=W!_cn`L;K5U{JgFmlp-sEDs=Flfio0<y;Es-VE^VUs%3icw{#FtScKp%`VjA|F`^MHVjII5TSgm+|1epij=%jYxjjeQ_`Gn+awGCP6rICB7+8=f(IAtQIg|s?Yd56ShPXL4H}3##sWlCzFPuO_U0QD3xVB-&>}&X=6E#rlIba}lkq*cXtpb)(0f<4qy1fVK?3$%ZVRK0Z)UeA$_W+710nc+|rKRW=>?u-{>LHazi&>Xah5;4C@w_SRC%)c>jBPkeG?=>e6AG#5V!h4P0XQdF<1SV>9gy2L3OoPQ?Oh@gRX9EYV)O9R7=!bWQ->5-=KM1b>Fa2NI$YX9#I5&&Gk;=@=Ct%KaOh<)ocNEj{hHy3(S|3FT07*F_uoHhW>a-?KC6pB%l;j@!b-ma<Uugjyu*jD{d`P`D8^7)nFE&sv&5H+bgQa8{;{IPdO1TRB0;W6B@OY)iC01p_&YrlWGWbnVXsL*-Uz6~B4%bVNUqI6V8*a4t7&2dx=w6M8b2+pBA)=1yXhTJljIT&@ZjjEeLGLWv9GPXZBrc%I7Kh&UJO(qW#MxqsA`#XpYkakuD<r|;;fZT_&6|GWbbnr?l{#;GH2rFj({0d@n|V;MA)oz7@Is|?11-6eykQG$%i7Q!jr!5yYfoIc$O`k_98_C@aN+KI>~1`ccG$WjH6Ckwfh7O(31-Rqq;a3fvmFSL0Egpi-dyndT|eE_nGBdIU5)xLoAUqY9f}|5Ia^q%AJ*__V<9@?x`=#-paws4uiVtApDCiPqyG(e*kh+8HO48ue0Q3Bw)a-fmu3x2ShA^!}|VJvX$P8MQidqU?{&#Ecub`4^zkmmNUxGTC)qta6G>!9Dk=ciZ?W^dgXpdf!Mw*12lzDiWjsRsMYRL?*+h87DP3Bu7Wj*CY9=4c}xe-mQ*mpv&jqyIM`>>Ka(65H1WO8`w$+4*=N8+blfx&{!0wEqVowm#j21Dd_ZB51dxhs<y`3n)16MFAk!j)k>{eq@0meFABr2v<~0D3!^|sb`7^A7(-H4)oiN~bxXfq2x19=tml-$JD~D~9)z$W5xhd@3HAojUq^*JbzMDC&ksSrm@Y(_{>TdQXiqRcdM`4op9}DAcdfPPgmkvRd29*wMg(_?s{&lT)oD8o<g?rhFGV|_0U%%(&kw@f>Oz7KX9julO{iypC+s#*O5YLZODan|(@aPQC(R;IUN_~h%S(WbKEI3r091_WIz5R^fCpJ@rDG^aj1WTpXjO<7*hE@gPvq7YQJjE49TZ2H?dsgIF{#u#R5wfL*{zk24*68yfkhtrV+cD0ZQHItqn%7assj&ula9O+b)uOhKFNHjoQt_v^g4jp$d6@e7q#ymNknS!S!@$D9p!y7lUYkHkJzJBJv=UL9Ve++FOu=CLpVo;pCqi>OhOEGV1W}U$e$1e;hsxXBd+joajD{5Zm!j-sr|?r#*I*DSp|9nh$m(jr<ZC4X3defi9+jpOY9L=g-i#`<2OoE>mT@JnkqfS>;FI~NuDAhK6zoaWsx`S~k)`JRR9WfDU`aW!NJ@p`r&n2-ihvh*<iUXyN^kO=g!I9$ZO%l?DbN2p2$7^N78}Sfn{vbsCJr~4xNeKp*g6Xt#mWy7h3Xfu>C=x@WmOn!*17_mVn!r9XEIZY-DNg6YqpKsMJ_^083!S$txEypreRZSp)J9%T66yhpaS}MYCB3-%qJi1Q*Y)3sQHa-tO)<s`Si+{#I^E4w7WjTqIxk<Ha8=6@9tg@)M14+s11wK4_tx*W8rSRdzMO98hU70Px_5FB3*Ft>RDv|@1K+2VZ0+whH+<^*@UmY0Zr#CJ$$sW)MhU$$DqZG*2CUfR-P~;MPq&)1A-($tL0fJn7|^=J9)Z$;sm3*4pp^&z<~_0juTx|@0SM(3yS4N_;;mQ&OCZ#&Mb+7x+i>#4Neo-d$ZMfIhb2&I<|Fp3$`}r6+0Xaq~I11tgmDfC$crM4bG}8E0;y&4Y!*wO1H^XT;8;k(<EBfLi$2_0F#`tEuCtlhwcnyIh@lA<Ky2gyXO5`p~f@@1)S-%R>01e*r<_Eu$)3+y0q1RXZ4xjQOou0WOmWxqdI)e+EL*;G#uY`CE+a_2T>JncUe3ywAX;2;o4XFWS+QZ{V0%o8Ta9Q!!BZg*NkqtH^Uzy@@yZQM@m~A>$AniA$+Hdx37<QOifl2z$wP=qU^l8X@2><1~?ML_-opv38>!d;m^gBA;q9-4Wb8KtB7ikEwBq4<Rjm$YN-_8G}l5}{DFp*KyJ0$fr1ETHM39rl3sjEcLIjqD(?Gjyfnd+`@X%26srnLBujzK*g)BXAM?yk_#N|{6Bo76EBqu!auW=1p%*#hzObmuBCjo!@lmFj%M?f2p)?!>31=s@lenw~+QDMzs5P)RjGV7>V50iWKgs<k`!rcnHan1m7FIYxeqy0up_0_Fuqu}rkOQKE73tXT7ZK8Kgn~!YXwSjX7Wq#rbVoxKykiS3XKtha|0?{2B*q=w?4s&0Fe`15O7V^7gXH0NUsf|ssvn2yn{C_Uboab+b*AUbv%$un1V-oB0KuP30TZ+LDGdtoA1kcIxnw8UErn31e}G7T6fxcn7|P2rIF0fQENxn)WeDZ=F}RrP!NmTGYWo?2gtVje+Ofxsi1mJhQmqURW@WEqnK#h}lb5!DaY%P1st{Eo)=A;%3TEXfqK6<(wlaOys&qx(S@`HNV5M;6YEYJq*bg0qUv>-qclnQd8lek1E#xn&zLc(M;u&6ZK$T7)D%e@jAwLg|7J*}zRpS;oCG&8|Uh)PgVY0I^k!YAHZw`z#SdA4pg|K(zfwl>YV&L}$|9lhL4J<x_d9B29@Mi1V#KkUQSrr*7T}jQ99!B*lNM~+*TNqe9=1npW72X9*EkA`nv#3brPVp)jWFJVo14nB7rh<HIP{`a@{2*@e7vv?i8HJ11cWmAriH6so9rV+ia|BDIcFx?U6vg3x#lGmYzz7CF+{~*48^=(TBfI<=nZiC6b##Q75i<~C(`Rl8vZcp8!^shk&H}9c;I#v*jj#ce$elQlM*{epnSfUA__g1wi_c>bCnw=nOW41Hq|e9+5e0b9m!O|p<q)DV5M3~&sr+CJb!Ek@?9S)gJXSYi+iXnoOD*e_7`cQBY)RS8B;J@+Hidu=Vl0<}J%S>n(W9bX-7e};?a~6m=mOErA~o3E1Qs&zo#IyBSbF7nSo)kznkx^kK4DG+7EKiG|1mqIN>y8p#0fY(GBfU76EHwpPG-xf>v3FcE~RZdY%t_zF#r3CWsh=W@qHXeh^KMGRZ#rDp})a|MMB3Z{z5tcdxd#hL_RWw_o{CwA#SxvTUZ$gNb@F?uzq}PAuV2X*0S@SMNNs8$8CMIrx!$Xp`OFmH@jmMnY+mq1hdK3Co+H1$|MCV5Y{*|wS48nGpaY}S*r=g*Ym>a<uPHdZG$5ULTvk6OA$SY(=?8IpBt&Y(=QZs6LYgMk&%!-S3RpEgkq<&t#3r!SJMrzVSJQttF{nwe!ccRi62P~Ti6~_0^?W!#r0akR10VX@SGxUs<GG%ScD&JM;5Y3&CPj{eCdH`deF5;tX6;Zombgz?{gNjtl1gcWtsd<yI8!w*Pw*ysuM~-_C(eUFYef?*7%ptlTylm827nLS)>~a2i=?;YUPTpn)={w22rPfr3(uek76F-5nJ+OM%kUT9DRZ^r4@?M5fZ4e?4pU(zOt)7m~Zfsii1_vqp6=|FNCH2E;cK6(c$UVFC7Ki>SZ}kuR)fgp34RZ+-_s9c0SL&A=`@3*8!~MiWu*!D2|e2U~UEy)Zr2fF^~`s#%qW?s&VN^r_Ot9CX7>S|J=_0lLH_F-Z-wNmhsq%hT1y4^HkD7O>e0rhfLcw36{S?19ELc0s>A3ncNK$N@>;>Au1O|9i2XLA^wQXYlTyGg6(qbH0?xJxruDlIa{?cq!A<fr`ebFInEBw^9U3z;|RYV(Osz{sTWR8uAb(B1tpxNsn|1f2wIM~J@cfR=_S1T5obVUEHEYs2X?V9YgjlC`P}(?l=X?Z5DMj-(oyG-s`NnJV=jeY_Lp_7+%3V7!?-V8Ns<dwZCqHWjLT44CNc+#1J@eQFZ%fDl4IU0U{2*LqdZ$R!b!`6pDi8H!EEN4vF>M(VKW<LNeP07nt=rW=b0g>T!6{=)^WSyY@b6j0D!Y0rs>f^`@hBfS0P(F?L{;9DQQjyTY-4U^hCF%e6jdVup|3^XW~b)ZAUMpqUNzku{#**JiygR#pcw}X6Cuy|2fPC`eF07#2cMYOUD!}j0Gz{Yi-IU;9Q<ZdHgH3?(*+)q3tLYe@fgq9H60r1@TnH)+jo_R>_*-ckgRVLqqhYm2AswWlD_cbf=6zi152s=)Lss`c$+^xnjdiH&W?%SaOV6V8=H-o8Q$699{Af$aL_?Nr3ZNc$&oU7O@90(H<9%aC^22;F2hWm94nWys(dcnLlV33qo(q0>FLnt1Y|%p{&XN1@qW=^W(Vaq0M1|w=A?fK>&AT#s7$O?t6x4!F+-+*(<U4F&G;<@4IfGG7ieLj_|(iTT--06P>)HS+Gq3t?yMVOG|R;?{i6<bpunJ-Fn5PJErsNr-j@r1ll){8YELJC>{<#Bguwfi;a4Y?nU$4nWKgi5@IRPF&DX@j)8rKs|0!|BvaB$LqPK{nS)8&Lly*1*^Y^OD4$s7#z2oLe?Nt|{{RQnz5zPR%P%tZebJ$=*vFd7IXk<S*N`xiTFG_MH#L#yrS;r|&WDc78$AyeWpe(g1Fvqd*Ce|v?!|O{y~8oeMYDH~t64Ynl1}2*M;x{KxTg2(4?h_OaeV4ojDFna2Rd2skx!vN1(77uvOpX<?{V-!rgLOwzf>Nq71sPF`(PyZegloK*^l)?70pbl`5vORw{WzDmD9|l8+$oZ)znl}vK`78$u3pyBf=Q9OITE>xf?<WZ|*rY4OffC0(VbEz0u@nC>4b6Gwu8z#?z3Ss#%KB+*QwDX$}^$@xnMMmCGOK>|(bJQ#BhM8!iIgyI-!4M2SdKzU8qzuFbkZ6unmZCFA}753o&roh-q*N(%irPiqiYlKWXb>@qh=RK9E@MMkY^snPNw6Hf-q#)MmqKCJ-jTycUG7)+$Daf>(J!*MOt-SVo?r!o7}EKmpbvKV+55Uu{L&O5qs*f95A@~1N?Q+3@#8I79-to>T8EByaJGJx2%S=&3O{&gs>oe7mPh=xzE#Hexj6;8UXS$pTzs9*1Co2l!>G7(sbG%NYdsKOnBZdKM`;A}XF%aTs;Yq7P|cKM9#nIv_UcAK@|UD0-X3*YRHX@Baip+mCgM#f&Sfu-Zgke_dp1?Frq(kPc^a@I=Ig`$_O9A71F^glk-T6J@P53(eTlG#$^l|BK%%4)k;{M~=UKnQ@`33UM8<)isN5rBE)PEY_`niRP2Zv!ok?-;xe$9IFuRTz|NTIpMZr}+!bNf6A&d0)RZZZhYs(|SRCWpA_Tud%$B_*-qEl=u#51>BsurpE1Dc^JP+JB^coRoH8>hXhSB#hZ~MgQ{-ucn_!|mh<=Mm$O}2sJsW`n-|Nr=rX5p7eItn=7m|$_R~*AZU1bvwFI8{+@mOn(61Lw!MKt#z3d{}_9iJ9E9Bv@$Vxwf3QpO(8@oP1k%?Xue5{Icb`G6rr>G?unInSisYJcbe@zStp`iXF+izquI7^2$GbJ`^S0FeX3bY)pA^`=&+X6=04Zv8g`VBcvPE}UJ6oH@(F@k8ka0M&xqiXGVi$UIcwiS92(ncwJ@205%IuYhG)|M3_jT*!e5*O+|m``XyTQX3#-7vh*RMAU<3If#f{9L}JO+L-CrH>_B-Qz`sh~pOTu>KKefk^4@c9EEYV~Qo@1^VI4ZH`ir@NlF0h<=V_oJ>d65$ic<9_Ybv2d{9aNtx<qIiX(WexpiP5GN_yT0MEepK{C8m<Us*WSr_cD@D&H%pz%HcI+5avG!oX40png%;%Sp+Zc`*IVB}HZ!xSVkeI%iIpK1$GmcpTR*;PV0xfI6!JYlvy-p#Mu6oRC3Ou;Qu3|^;1g9}Crh=!EoQ^Dd$mt!d6IEdZd?&#Eu6!Ec1uJme1a4xeN*0gaQRj=pnpwM!k7)`NE`Uj`vA<meX|816G%w2*SmQ_xE_;y*Ovj#T3nDpQUk^H9s+?8;aL?$U<?0$`Ui*&pJ^<4kRa2il=8guoF0($~&85)6nG!)CPW$N|#}vCZ>WQ7ia%0M)bEo54(VMm5y#FG0CgYnSf594%PF^`IB1!)CE;@NA7vp*w1c5y2!K=43aIf=%sp`Qus5dyJIyHJ{s01Um>;%am>9bzzbWye`3c*rrd<_CTDfkIKHObs=#WnpVJ7FrWB*_vivQ@+aEifF$z*e$Sk1=S6h;?1o27wHnE+{GhlfLUciYjw{@*A4%8TYC?Hp)SXti*r7Z+^nBq>BuD^pT|y6}LLtHc)I|iHNcW#>3MgjAIhZGQ2a#xCi!a1li-PBh?bTR^^}|DMzouAOMEcl!>{^nhwT>7aM42D_uUTfs!uuYuKuj%Rx#bR=d&gdD{oK;tPM}X+!7FX!mPnZcJkf@8qDtS|O&t?01GOfvlU%3LnjiQjd}EC2IQ#0b(>wf!rq3uxcna`io--B1cDR$)TJf{$VG<22oPuSHL3!*FU_2I`#=6DkS?4CWl;p8(q*mO5V$lUc#EuxLRT_ftW}!?MbKr49w_k6nv^cw@nhu?03P>D>CowFl)T)yD)3jLZ<6}SsQ4!`p0crKwAENpOkH5RX}_cg#y~Q@(QGio^SvU(=#7%Z1_^tisQzcR9SIo|4`)Hs!{BPvDYY>pxB~l6u*J~X4S-_2uUcX2fcHEG2pQdLeXSLQE|yqqe%S}zaCs^xJn0zNYuZ}^v857q`OOI=MUeyMx~5RteH<1at(rv<2eOFjX<|&ild{FY-^%Cm5>-mF&dH?ydUWlHqm4}8wx?-WfN|EuUSMO=@|)MBUawGfb2laz!s6j2bTescuAr*N5Q2y@axc~v<mWZqgxE0{0Q+4anL%KT%z@{BLQ9Enpv!AgSACr-~*S>9z%-3J!J~y<8E?MPH#`IPl0x>>8K9x8i^N1F+?IPM+8G=mpP-R);Y1gq&njkS*H2VJHFgvA6^z?tunIwLJsTb#aUX<yp^;=qa@Ccl}O!d(bd$DcPJi>wmzQye#C_M0=~-{^wH54CL=rwboIbf>+L-y%P13}a~fD5C_Ry)7%g^VbPEw6l}z2!6)~W;5=(xmJFw6iG?v67tj*1QySNBRe(KuliZ>|fxYZIf$xn}tc8lzOl46B_B#Z&FQk<|*h?&)fa1qtSe~bVmhPNrts<j3Y?h~zALOkXtJ+s^lRe^|kt$`=xo;;rDsyZed*dgM)>dSBrLk9}x@LLod_k{a4JcgH*0smWQ58@GCyGe<xmuJmzLLQYveO8E)UVrkdCKn<PN{Cqk;^Lg%noKv^)9H>rxD(79Cey{P@zABdJ&lJTfUYV5Db#<|Mu>}$6b#&J4DkULvz1!7)&f8r{b_u551k+%UE#qCp{hMz+(H%6y6x&cF`N)tpPUNa7G~RpD6q5)B@#%sm5%vW>FH88&)Xi_d{f(bu~#f<SSu=Eh}lU0$8-(&SVn{Q*9@VDFl7WL8m9AV!CQ+n1#<L+6^5uLs@P8-@-dG&t{C^Z9BXxrTv}mo@3?0O)o~9`mCbfy_ilNdFgCZ2-{MX>gjHoWXNF|Vmcr;7(>0(zuRDpg`is$}*mPd)-i5Nb{E%d&02m+3bIT01f)lCmtd9>7+D)DDYc$8&f)%_~IffH7UC!bX#s=)s*Tr9&9d;+Oav5(irhM#R_lS0Q-<f5XE#koo3)3Pey_b%ppP*wdc-#b2jY;aL$FqYB4P)=x!Z=CHKs8wk8emAK0(t1XGctasn?(}(C~XplCW$DDO&Cqi57UrP!A#W1WF>p)lz%xuZe+2<p4t&<GDQg?c!^)Qhzz>2fUHO4y-7ESxo?l#gWR3UaCt{*yjbe^Y%*6HJvDG`!By1HxuB}DihJ3-1!dcvZFW4&lhoG@!QxOpGyS&)+HK7xHUXcv6Y10BR=Sa$njVLIC=tsC;J#gHIgBnqd)y|ck%9D!fc-4;xqOeczgw0RGpHfh_ZYE@62PSeMXWtV4w&ZQKpbPudD4G4Of|n^I=q(-wMVXFh{#v6AKt|}M?zwc8&EbdKVtYlhcR5sI)tF!oI0l)Ti`|hHWY)P7?)ZoUq$SiKyRaHae1}e$*^Xs*=x=BCtE2v_dHSQAm5$IbHX&W5)l+cCD~*tKfz?y^hV`mYr3q(6x~XDG+|j!C~Y{hPu1x_eGhc5*<hxO6fvMG-*~y@#4)fS#uzb@4iUfBDA<~6I)_iM@^W-yj4GV}40${*5aS+n9T#Bhh{I4q&{_afRkV3kjEi_><EE?Xw*zrWrV6h;>~D~j&52l^TLL_Ib38{Bw(Hl9&eGA(+HUe4!K6~con@KMLQwh{ybE`tU%n)WOn|nc<*S6vTs*-cO#l8m=-@)<)bpis*w!F)wLuR6vH+ruNJ1XQ;}?NFnOcLDa3A{L`)dMS$3O<ZPR|1<LFp^V$+eyKh}@73xTr<s-L-u~1)`Z2w6d#2Kg3%7VQ;U$;z|cIK})RLCB+_5M1r>&;i42>F_Oh@mcxLgX@n{X40Np$fMALmgf1Kx=lp=EVF11703CyoW<a9q4S?ofNmy)qktXG)g;R5%p4w%H2s{2Jzy;%<1;)9`@}DdchieEsaG7Y@5Z(o8#QuGN%&?rhh^-lB=0o*?HX{J}h}D#`eS4ms1+Ba_&7rQfbuQ`t$^ly~Nhr#+fTJ2Rr#@^K!uA9f5(~5o`C=GH2?8pkT(?B$GENN!$)=6-?q+0*cCfac^RG0btU2e}-~9s<Wii;`4+H_zEz;JwwpH72rC5ER?I(CH^ETsYSS7lGN@=rEHQ{ZA8`pKln4^Ti%@Xhz0H*G8iHcx_@L%gX$XHZ=&qX9U-RqO);(&Q1z1nnbLeCjVrI|!ltjDx7BC5;9=aj7vaK1N#BB-t&J9uZP=$*@h>3%-pjLwdN9b+s({-LV+yAbT-cG;^{mdZ>USrLR{82}Ol1aOBPnTn<i4Bwx<@;90EO?P(?)6%=ci>A=IeIhutY1|XVc?;?q)ph9Ia#_#D000H%$a2_GK2EBU&7U0LNWc<2hd=#Sm4K`pZfSIP#^C}DF#kRs|Ag5bjfOQ^{PECuYHK&s*-YLDfNzIlNa2VFq)#FZH3=nc{$@7&YfcjLQ1!Q3f-|1MNJG|t`3XOo$ZwrjbdBRnh->_>^rmbIhAA=AqMlzHEIa8G>D{zLx+fvX%Ru;jod9s8;+Y;=wFQFK%bK;W3ntyHueP4udF~k_K8T(Ib%|@?9kIAkI2e}GuoH;Wb2+o9VPPd>^qfGJ3t`=ykM*bkjNu7>o?tk++`k92WZN8-yd?K4&=LyeLiF&T%|;_*PWNx9x6e{K>s~xAgB^nHk(0cxMM$3rAIp2S?=$QQ>Nb)QyZLY<f!(_9pkbEoQVD!;0!s8vDtvz6yR9u^1+zbG0}QH-@EE$_uU1`K+ABgVfs}0^A9BUq5=?~<%?zI3Emc6HmkGgs$lFP5)JoPWN7|$Rnm{o>%YEH3$bZ1OxXYG>OD=QF&TH}9@g6oLkHuOn&a!QgY--=U_eAGUtpTd)9iwlA=J+jc+R|gN7?BJ>&2n%_)fYv1|Ga|j))~a!B-!FF*MRN{@@neX5RPzSFRvvoz+lw<bLzIpa%&ULQkn%-th2OfWW6&~1llg6pSQ5<zoY`EKR>q0;b%dcpwAkz?rR`#aP=AE@td)nk-W9%^AZ;L-kJf0p0<;Zuz>N8Mb|(4cFp_c>>NE>$#8|O!z_pt3(R$?NDnZp+Vq|Wb(%M83!e@iyI5*LBPF*Kz>s=l)gmPCQ{gxyA!nac>+7$kcsCs;gb7X{zy6#{RzHd(^MTIRe%rp4AgpYJlPP@JO*zGnNM!H~6DrKzAb33MmKn&J()icaN(G2e4~;4RM|bD&rR?5&fEdWXfb3N6*D~4kMHGg@A38(}vXl+|3^Muq7F4Z0Q&acj$BNGNhAP!{0cb!uN6kBxw>?@Z+{H(7?71taseyN#Tz#4F-O2mDH*+jIn10G&t4bktm{--hH=7UXQ9*+^d1#XeurQEiO{611ROaoHkvCgAd$0gyRm-gU6OQ*|4Ge%SSm3dQt)&jlHNxc-^wr-L)GV2;)Ta<-vc|yw;sx0JYY9A%nPB?^_GBj{a6RgGnBw5?(UUbxas=l-DHoDgorFMa)<k?PY&*ggzW3%ub5JeiZEg)VO2DC`x@F5yR6RvGaRwJv6-%|jU!f`*)3~DHoj2Ac4gbl$J_;w}`V%F}8EQf3;#41rx;z(L71jLEP1yqaBsO5=I6wo4pVU6wnXkF`2S+Td=0!I-pO5sqbGc7F$RU*<o7e}-b>`xZvHD~LQs0_2xKOv|dQDMWfBdv@+>UmUMAMp|77OCb5o8s2hsW^Kod)2rWM8N-=nim>wxENq0T$<#tPGm+Yo~sE((eM2trstUIv5Aq3t(Bx&W5HZyG1drj>>W}ge0_;zg5|NaI3ba&1c)fuFHc$Bpd-jJBe=h6f`aL9Oi*^X+Baal59gQXtOD(WP_CYJ~PDszFPLA2>z3HR2eOv^Y<5XeoG*Gm7RUD5!X`RAoe(TVBZ@P=#Hd+biZ15g<ErtXeRU``N{rkO*4Ozd75DUmp~l8k;TW!dpP+M^vgGrDL13!K!|b-)$sbH`T~!x9Q||=BpObXw9k+lbP`*56(Qsf4qNwEmA3jgm`3`&DUv<sDj%?$o};ld+DwtpG$+r~17AaXTaLmXGJK;&vsHoq38WmdR<RF8Rhpd|8m)ITyE8@}p?sm;*S<m5{fo>)(?2H+6S!l?$agXG8~;i_QUDJ<G_v1pjsCyQY4@n_*S)~Nhq2~*O?r0IOt94y_29l@2>G-}$N3&#{DLb2T;j}k@B&Hn-2a1`HWc{Cl>Ic^p3+)F|FGk}n3-(LVZpQmWXhc9468`kA>>p|!P&n##VKvK!)_tulvEviuIN)4uW*5p*AeN)K0yXFqEFH=#T(~uO+fjk4KWHyIj#fy#HBwho2~rw=YPP=GS=5RP|@8VU!$)sWMSxqKOeZU{A&{Pl*L6>Zav~+r&noPw}1t2mtmNqF)E2b(1ghjVksbwS^@I+3@L+(#G`m^$Oii<>6r?xXb85lS!Qa~a0DuzcoNcg*BC(=+7jNhnNR9Mz`29;vFqfopCst>G%E_k`gD~L{65ni6HNbF1|9QRzUmJ`>TJml=`$@X0`*Stq^yQf&*N4v_yTLSo}&p3Jh7p4Le1lrWtXycht*xBa1I;g_13nGe{i`*;Xbl);x`Hi%ThDa>9D?~X}5Rz*k1a6oZYh3%~?V+sK70-+NkI3$3vPAf|A?c>@aNDjUp%JC1&%LIZ4j*+6i}+ZS5GI&F>QySS<$L*&s33EW967cmBag4@lSx_yC3t8I8k61!Tr-Gn|V5@qYXNClPP^Y?=Idg>y%LOy$Ba7?!<rW$;u>0hZ*1`Lcz%+q5;*tH<p&-=%HXS#})ddKbxk3)PXG$w0)1sG9x3Ko0I#5g&hguRq7fb&^w>;O6e5!A2zmCS9ayfd2?_LG1He@ptlVId<YD*EIyH6JcSCh0+vih<|Yk1>c0!{Q#*iymG|-{Hz5`iR4C@JHnDB4Jc(-xl5_3822h8h6)OR&jVdV5VOc9QfNdkU(8Aoz92{jDmBqH_e5h^fJC>&yybn5^HL-*!UkA+tSYlvx_n-c;@3`dpC9rX7bJ_#av5OHR3F17DUyeO4eHuZ)R$M+1@2B6DUSpBt%kr&l&mN54SuOZNVB30ELt0TZS;0`r>-9~NH0ui4PbD^0jpjHQa@yl-P(M}fIg4ib{cX|VP4A`upnOTPCok#rJDjr>U}A^V?Tpp1=HU}0w7t4V%Q4P>Jxw97nuo&Dw`B~U6?q%V#uRUN;7gOsze3#oX!}$wN2k+r<H>TOH8hinQ%2!J%c^43m3V);t#(&f6o@whcXv6NX)V7xl!|kSS*4jWkkLmd}aJ_JgxjIm+a9)M3VDa2M3KG+CdbRj_{MrrldbYG4-H#&@r2^rZlOTLY#5Bz5mCpw9wL(I;+3MJ;fy^6&eb|#<zx5Rrj~QYg}m5+5QzgZp=KvEu9*!*EwED`MI>wsUPw#(I`K~_$24ZQ~WSKj~SUMOY9OODkAz1bIKbiF3VuY0l*9~I_eQS57WBx$5|*(_b?+S_C9aEnrTs~Yt+96QAMEx`fjlrvcmH0tT^I)*G(3Gq?wOb?%0a`68U5amFr=jlyOP7d7#0q<c1$2JpJ(RHamWwP7|GJZM)iaERAh`&z$Ybx5K{H`Ii!v_1FizfAMwetLvpN>xbCzKP{Mk3(~k;$+bIi_&3cJ8jpr>Rs#BTp})>t=v(!zVB8WIQMEFh-Y0sWgu1%YYej7cD3zMd(j|fdf@)=}nL4=K$fN{bObSX55_*nEAp);w_j65wnt&T0dLmP%SEczIZgiJXUO;~2absf8CTFw4JL%;}r@EDJ<?y8g&k)~})4S&#OQ9tns*oSSg(&pj&x{VJ{pCz*_z*tAA9-<r7+zLHkc;(R-JAzf81vDyas$L`SNkV>g{{8YiQ{|offF&Gikv!x=F$ZlN6IVR$k79cIEOCl|EdNj+33Xk2jgwqv#cyz*@Dc%K9rqfz2g>I4icazhFkpg{JbBXP;2dgY4&`d>NPlfevGEM5{ih6+s99VY&6S{C)0E7SJd;CbE6X=5&C<1NqB_SJWMgR^2M0!m;B<KV5&)`1W;Um^}vJ_5E4z=eXW4{n`!IS_K4AS9yc!sbJwQC+Aes{6B~$>tk&a6eT%e0NV2hC#Y!L^C>|_sO$CIAj_a(_OB$6g)Tq-0<E{rASpz0ca+VXC_4s5O${st9Gvj1}vqx9(4g)Jn@BL{l%?g9>oJdsY+plijbf+1~?W%OT^Z~c@@psYWb!|q5`~67e28Vx!KhjBqx99St4>k@nZ5n_|be9T3?NQ}FC8W@n&k|dsf%Tt{9$g+{gbRu0Ebq98r*TzkFzC55e*h>4Bkzx2PU7Bkx#xav^MF^lU!-~UIY0hvY2wB$*1);FxSn8?yQGaJuzPj#EHQfr(U24ds{cU}E*~=s*`1qpXCA$ZWzu&fcATAr>>Uy{^I$zJO|1`}?1QpQS1c2LUtNz#9A9)#QFzKtwvG2Hi4~lj5EA%^V)S<EuDHTX40Iz%(CTAihTW;pH$99<)Nt_YZtI-)%taUHl|uNrIRG%Y*=B)NZi(G1mCOl=nT+{4xKV&6I7PTk(YLq9e6dydOcz-Yi{HVvWz)&qvHa0p{Z%s~9ifLmVV2u9ACy?42$>uD9f|)aO9Xst@`35c{X>W;SBL}yV`;#hd}b?rrv^}mMujefK_wjh6Yh3b1PIlA8!R4;qH(7rXuy8ZJpUl^&;fEgQwUMAJ_9Z4q~Vn53f*?VtCZ~E&i!77)novdDBIpuh!9Mm{K>7vWuR5)!+O&mwr6C16BUMeXTbL7#lc4#WawoB3P8M8DLfR<L0>`SN%X#?OwgVggD60BIe*4bWgYGX?&pL%ydauAx?5A^a#nKm?fF^Mwez}Tp)xHbohr#wIQY&*`*3=vJvGzMZY$|TY?q50WvR>MxI$yDE?Ibux5@6ajyp`V%Kd*u?(^fyDkF{GL6J;@K*QFJ-6{-eNq<zwE<jODS|?maW)o{{<c9VVBlza?Z}R2Ab#;JDBIB)P5|xW5|3I^;<zx_&jzUy(#8WzRo>}1pg!)aL#AEO%bH95e%FKEd!(L{>OoY$VZ)4O-u#-D7%CK}iAJNGL^dE&(4*GgZ9{;7Ijsct)8<TwA?VBh}ZefeA*5c)$G;%<(wr+2%2(h%-6~0-<Fd!(KzR+6X>4=whg@h!N+euD8b1q@!mb4q$tD;7ZG??}hXDJ7-{ANjIBQwm;a3&I3O%*?WcRd%A{mpxjw!b}8<5dUezp#YkHNvz*fhb%}%zMNyXI^Hr_LJb3M+3ul`gHZO0csz3y&QBNUEe5NVp-<VUfRpSP)KWrI;S7V92nQegLt62fIZNaT6PqD(2N;f{bqa^Wdvw!3|QYP341yOJ+udioku5UoN*x?HNVj)r1&ScpGHBSm4+*((Rrc}<&sOqTgwZpFA{tMCgPYA=Fe)mal@0PQj-~6r_`gxoernQA22t8DHEii%7a!CNA-Lj*LBRN*P@O<zx37FKr`7s{NDy!8^KblU`B4^woag+Y<M0=sJp_Rc?4DNg2E_~WH!25QK^grb6LNJ06mDOQqLzIAYTW|aN8Ez&lpdaeXGNy>7}##Yknl_8dJ<wiaB^~VwA64@wzR=dajSp4T2udJ>iyKb)=Mj+bU@ZT7@9<VtK!^Q5bf#2{2wn-rS?)*X_^>&c%idI>**sq3D1jk+~$~siOz>phJFV@iE#}@hnuW!UF#T)n<$tS7<B`NGx95f;H-YU{yCHk5)4$3^FYM>&bCs$wBZ6?)u4Yjd;G2T3!DhdT}MEP=RhIntc)}(@+>%BhWfrH$Qo?d-;i&T)Dt@Xuv|%kEK_B;k%Cxe|z==tVoo$KUPES`{2_<r?~ia!<u3MCGs(FRAnJQCrDqRp8hM^8bs`g=sDqggn>o|6^KE`GN`Pl+bW>MKZIy|7ntWL)UdS&7-s@RnmbQNc`0DpOH<l<R$Uf}<%oZ~Rl1)PCsxinu`jMNelG%6k0baouH=lPIRLUQL>D7$Gc>0x48$F|hp5mDmOh)h8(7ka?JA{x$)4Ft+f(fv)3Q~IW}c+~HC_ZCxFmdH*?MXC94hH|!JeCC9gXHJPl2+qKCDO<UjqV7Z8QLLpc+(afec;JF9kE(3=9@2*T2xKe+6;w&wL-x%Q(*8r&*urEWcuyAUl>hc9#Bkn6oImgr6)ky>wYMoH(`9h32D>R~Nwtf-?f&9JIE`yhY!0WVgA(PZW6XI^zZ=bR?r2TSEwdDfWjJ-UU=TCSgCKaOZ><8h-{iQ`o=s0%WeeZBI%uRo^En$WhIkpVp=8LG@d3r!q@^6RWpU?$O}-7<Skm#w{GcwjF7#o6L9jEXCmT$Jw^08pvvfaF90KWTom?rCnC|Y@P&K2OW*!ITI#{;*+Neqj*~GX@FLh++kNhFB_yP)q2ORUiOdzPvLo3vmIhAF!S?8*${g!>y0{&Pj?BDCx~D?2WlDu4=)GLa4x}BU0{zqmivXr29znhCr6ZST*6&*LLVDZ*iBP%C6a2dAUv%aa?*cc!?zvWEL}tk=9;J@f7S%4wD)k#dqtcmnbK}N!;|+D72D@*5Ej>)1>Rh+)eTo~MZd?ICyizO-C88C*?_nsku$!KR|H)D&aoG-yAcvdyKGF5XWEd%Udz>zjok}MUj<-l1h3ZNA!yG7ltg<9n<|n%Pl*@6+jsSiJA6$I+Hom5Z&5V)d@KP^6Xp@#%wNdAAm+!ZWY-qo?-hWC(8DkJN$w*4dNm7j@>xk9w_ixLwYmLFzljZ6S2}$u-P`%-&{$E%)N0R%zKwh3S|0}Xfmcx*{U-3syA^t^WoptfBKKE;9#$wTeO99<4WzA(Y)5Zkkm}`%SFMO>6Yi^m!ZBJ_(I1A^nxp4fX0^E<D+i34yo=@uu5Ha=q)S_OF0MMJob%CXW`h(El7$U5oU(?B#-j%B>>V2gJmX;*fv3?mgXW`5oPv`M*s}9<>jpd<9jbP~_<-@Do4j~$6esoZY`J||s+bq@I~h+VgRjh3A*Y*^n>ssavk-~o&al(DkLw&Ze1!`xhwwp(&=X4c2grKzkCq5gN_V{))@A0A@^J_parxW{Oy~Fc0-VFtyF}_Vd6LwSnT+Vp=E&jA6BlMH*yTGy$NZ|_Mnsc8i%6FjYS)`V=Ds<1O5MgLpEk+lw^(~hYkn(T2o!3)@?Tp%1&}x$B_^_*SS#;ZdsHNm0wKZ0;+F(H5K;CHb*t96@A~H5(uUn+Lv^t$CvfVh4Jq!+2ieI)-DcbSTz2H{*aWpkX9QKYV#0<-FLTN3`Bpu^GY9qrYLgsWr1#wmp+jU%Xmp3_ADoUrtLM`!NnX#eQR`25;XQZL4De75q>owj%4r3IP~dtVuIlul2O#~hk%!h#R8AG)&^8>rKEcl8lbsJ-<>oG$tio)teM;;L&9=h|d%k8&!NKA|NYX4F<zSjs5>pSE!W<T$Md=D`k8Wp0{zAi+jMeN*(uQhD0qtCUu@~P|8|(q{77(6Imhlk|l#V>BwXL1Nq{OyS?XJ4OY^oW(-kpHP@#x27uAHx-7*rG00zl3`sAYb}_hZ&km#-(Z>U(BlsqrcKt13=w29{=(;CyMzh9sC_2idd3%^o-9Y}h?0xR1sNqqxQte(6O+CD{j{&->)l!B?<5gp4P%3Mgmrm*1y{St~@S6CsBcUte!hK_4z*r`YB$cmU4v@K(BiGi@@=<y^QYWSD;SxlO?n9ya~sc2^who4^i290@h#EwQwwHE0y`AvW+LRU3Vq_xbN8&Rq<RFj^U}SJT$+Ie0pjI_g7qZ1r$%7;Y?yKp%QB57D(v2$lC~z@w1n#a1jM1`QuU5(b-Y6zBBJMG9b`7CGTNr^f$F7z>TJ;{Bs<qlR=s^cTE6Mbi5_!F4NS0pIsT1~YjTPZ(JN1W0kB`q^t0>}{$h&qnC|R=1a!WS|hP#i2!NgJOPLp>#%Qn%Uiu`da)iuN_aMnFC1Z;QnoW#?<E8VmAF<YC_I;GB^mW^Ewtf5lU?f9M-niwThErd9+`Tk+WR*dy9THNE2gpNk%)7X`y{m=T(2Y24R88EF{b=S(tEGu1>1sTqR7TE=uFI*1kb3M-F<Foj)$#stu`O1Bd^q8cvf)e&$iM%8rQpcJE^7hC_4F2)|eS)yEob(y*wzIq{s(QH{^_1zuq}`_zyhKaQmjqz2}nekt1aAnv?Vetp#VNGK2t7u3@V{IVlca~$Vl@EG{nt2&H`@w5=!{Fa|kVd0ZKs&bZfz2nk`?EQh{4i*{hvRs;N?$n!<{?{Qjm`?Oo(c0F^r2E#<&v@#u7zr{*0ZvkuMcJSHTTJaYJ2?J6DMdV@?8F@`cO&RRG2e0ccG1JkzWoRcA$C^0p}is9Z42}J2=7GrHa9cPmkiyAr#TGQq@kk9073-BAP~dPlwTPzdo8<;?u4Fb`7nlD!^vu;Yn34C-Bjtakfo-``Mh`dV+?bx6QbWJ8F*z(m8s=BVy%eT%lC-)tm}$u@Mx+sTbh|%BcKK*7US=SWqEc$ww0D=8L2D7$>XJ%K+^n(CN17@(N7$mFh#}$qo!jELoZMfRFMlJOt-5A-gh<8vp6nG;Ep>>6l&_R`sT;_X6$G5{meH=uWTeJfsUm*0@4|BAlJS^aW-x!hcIo<4zXD_B3v0wFc7q2_dMA%t|kosx<T->*wuO$b@)wA*EFQz#|yZR*nj3oONAkE{uJMp4{#VQHs_gnDiis)za`#HZJkY0_f>{TNp%7<vJySXM{J2ty2hzBC~r09vw<X>ed7VYcF{Mj7<Syr`0bD|8AI&j^&;bYPmC{Lge!b)syxo+w`^Pa)Qs2y@;0UFE0KZqDBg@&=poL*G;u(Qzt!YSzq5xLNe9t6&i=~)k|&pPi?Texjd(C~eH!T^2q5_}QlF=(^ILFxgchR`wK9}REmTbA{Kt)p+n|BL8=!@W@ME(znKCf+0aJ)ji4kKZ0G4hy_DWf#3Puaba(3wFz&27kvE-fU{|rOfom`NF(B-{%ySe=k4*wMHCx@64nT3q(KrCP9QhEzNX_fSv8F+h*B}~T|WA_3dM5T5`+Z$5{aSXiA*&2Qe1Q;HdSM+E=<3*$8i>^E|6dY1sH+{dgh^**jyuq^lj?n;bJNLaF$~-}#(}K~X_}`UT#!6={X4`jqnU#}tyOqYb7m3kjGOrhu4I}I_!nT(A{uK5mW9JC8-y(L5V$BcfHNJMsWcX(}-$g#Lgmu|^b^%xvDzTGNe~U^?^=t1O70s*!Bg`RnR10hl8@#lHib-+AeeVe+l0J5kc>gq2YHUE@1<r?$uf&8hQkJ*VARpAmpcNz7N?su;ZSg_kcpcy~^|vmmmcnZ9(;t%doC|S|`SRCxC8Jtr)Xe6C-QzAM8S`yGv<zixaW-z6X;P@yslT%ZbUeXa$LMF!l#~3O2w|O+fCfl23oZS3Ie01E?r><L2XT)AFc?b1N@QLcpJH)-nn!`E2~z!;X=nzJte;T}fa6bue%y$<ORTHLs}}8>xw0KB9j`Fd*7wGGGYPP$4Fh)2Bnt_wY&95Z*oh$A>dJGRqX`;!VW<owZIR-npsyJ;=VhSgTds1c`{uuulsEwZ@QIEPl4^Qi(Ovej$J4DDj`E)vxjG*^Y7UFQGVq0l_o5+qCM(_q0vZF#ud64Qm=5!60B>VQgY2|^mImd*a&baUa*`O`l5hA+NSi{|jx1%i`^sO)7FwO73UT$CZV_%c79GqBR5wBT$`7hPPuH*R7|B7cTHa1B+jKs<bW6b77H|m{pCV<boBbu2kW^+8W-O=KUB8o=@{CWfN&fO*i(2TCZPb%~Z>f3vHf1}~0RM2Ej0j>yonZn@>X>i}ueQanR9Mm$aYx!AB@mN(gb5Q)A7O|brwK<Nb(aJ7-`N%xa$4ou9iLX04`Pn4&84egmTTsP4~}j6^5+U)M`m(iC0X61^&BrvJSFJ!8T@8c(SLI-dno7^X!G!G3|y$Fe8b!V`0K**=^aF2Q~w<Fx{5p%#K3spv!iruUG2MX+N8p0F)gDZK>Le`PXkeuS5<oVsA>;VcCHdqpo~rmS-wHMgstwc)1+i4%9mMm?vf~0i601!M$g4^LK66Zh`Vb3Uc|-l(sY!EcOadZP`VrI6DVOC_p!%G>0#JmCp3+|_B4HHS$^M!cgaDwq~PVL(}&UF@TF`_8OSzRI84WMdiLIi#^1FiW3vy-4X`#e2f2L<UTV@VWfYpfO5VhEz11IR!^ZiRTH!hbKxhKuwE2VEd)`?vj_LpVT|=T_1aU>K;&E7K3L~q!p#;ip?b2(gcX7tc=3R?>CktCOAwaTEp<mVw6Q4uaA|%lLe}zY#7&;<QGiULgq50o^suX4}7I7c)*IHC}e5&K;GD@&v%|}W@YcmnHX;{xDyR^X(9%V4+%5DJY4f<4<BW~{xpK34Fwhl*DY~<giP{ixRY$8<^`9Sw`@{OL#D7hbOzrlmybY#%PRBq#K+Ru0E9}93xcbCRJk6H;K(|$3mpvLgsGRZmk)~M{ZRn<ylvETP}aK5LUXwPkPR5_Z4IEsShxG%Q9{ol6lxe&1n9<xzVT`Rci01RcaB=@CeMIrOgfWZ1>$Y|x`+sPa<j2Q@&>I`tLCdA|}n(16|Z`+ehLrSOn>RbX8DWeiKGd)jv9DhE}HkXn{I^=FtY952BWOZKtOP&a03-A2DB{EB8u9GP8pjDeIko*C*t#TchGjqhZ2>aOTmp_3-hqoKJ;U!RLeS8<go}93awqivIS-FL+TD#!w41Xvp>Ub#}YyNKFhCKj>r>&y){Eu`a1J&C+9?M<fRh}70LTOP2&1RDMurk(C0%ql`Fw~fRa!7Q>nb57Mvt}BB40&@aP#ihUL3ky>EHR<GtlJj|J3BCruS5_dO~s(!9w7D@l2d_+`HT5!X1k<hV!Ktk-FsS|8ve2=+(`NTpzE`Yt_MBMieT7~*4B2x;nxnp^P*pa!a&lGG5Z-WXXy`~P+;sq0rnRBWE7`R1?VbGn$nN)Nhumt95Yj}(UAsa?=rtXwRd@EJ<l*KiZ+%&g^e?DfM9$3QD~4l6$r)vfsz$Hsh2erVLp_GOliKwE5cUum>_t`%eM?A)ryc@sCrXghzRtQ$kklG?)}HForB}Bc*p^o!TA%!vxr!e-V^4}N7)ZVA3AXl$UEV<+B^eEbE<7)<ALPh@@zPJmV~^ab4!da&CnGdg7k8pLtGC>^B<%6-lyT7CierEC14uk-JiRIA^-UfePgLk`#b>R3;-=Ce}#1i9y@?CE_T*nNhbjvu@x-t(9F9j#dfbz4`BLan&Dnk1vK#tDDlfQfc}^#!*uS1_q66YI7%u`QK{x78slGPoPhaF1-qiUHs(^o^eNVlJ6|45r(8fcw!tw|TlIslsgn{wIrE?`-5A7%CGvD9w;DH>)qrg4PKcZ&DvA8m3g{8BWmBASd;u60-@J#1V?ii%nn6y5B_H$cKLc@zwVx3JEbt?90>*^P5O`;@2u>|WmN}u58xn3)&COKcNz|W3Gc+!A{+d$bt50SCj9LCvIOnYj7UN1&Sh%F#%xcL<U_g5>WV{z@zy3&#ijR-6xvzN&&9{HaTatxR<`6w$#--YxA^lgham(@y4_BkBi~m6wD210u*G6XnAJxa%odgsYg5%Ywuc+ZLc-bEhYCSGTC)8sz1=Q)sQyMdI7p6!xrfczbs54jd`$971X2Wb`XL631CPPGyl9qB3b0`dCf39abbO+kXYqWgh?JBXBS6lhayy$-h$#oD84mwZaq?16(M{%m2yD{((vMtA*I9Kc|Yi#1^9s<`<MMT)tPyj`uVC|+&y#y5zco+F+rnRd9QH(XCwhRQE90B4HRgUadlnv}gv&B;ls@5(Wzyu-2Jq>rJnKPojLCS=gAXZ&i*##$~;DHcnw<hlnEhijo53Kxl_I&s)<=JzG3Zdu&dYAApUvS3%ae&s`e1L?cahh-wz2wLs1U?xgN$M14`3py#dscCK&oxHNMk6%@8CK&25|g}#Log{c)d!XJ_bVdaYt|F+R<^V7Nfs7pBmZ9-u#9@=@PTZ|bsUY_j~Tg1Zl(*=Um|CcN(MFmOu%Qxo{yXy_Q|p;X$)k@qd8Mk%;{oWVqxU|=6u^EXJ&R)#@Td3HQHZygkX#v{Wo6yDg#dBw+O>2U$Sm;JZt-;TN6_Wn{DgWXGqPk9C+6AfXN`JmU`Ob-cH8Wc~P~t_m7rKCZTD?6}~}5;?G7dzp$y)!biH{xr#9fUt{ic#K*~az1Kj7wruX95bjKYF$$4mbjO+Pu0Lq+Z|+{N9*IE3MVo)Tq~3-+mbf6WskZScO9g|QzpDB}H3{@cUw#X7v*;G5!$_``Bo!h#FLOfVdl`gaedk(-6=rW~Za#RBD(ccuN)Lw|63cau{Sks{If>V>Znt>KJ{jsjI?cg7zukUE8vSyKZ-blh>0*|K<Y=XkoGQ5Ngv)Ve@+z8O0};@ses)FVA%w`c92+Vfhu|QV|E`j83LA`5SrSE!i=}ba2Lv@jk5N4>!`0Q+$zPK=?5FeF@c#6<{^WjzwB|%RUX4wiHE<^R=@F_=O<|N7Tk+QD;M}~+1uhW>+^WKn*`nx7dmX_}*1&IvqYiL8y86pFf6?bt<7Aj#du8`&A`BCN3C>Sr6-%bA*P8NuJ~toAM>6Si$CM9Pfx{8X?&dy>!lccZU)4!WRplUO#hghHwYxf3{Z#N+F|@YzIV;Gp{BZ}UM-%z^2J9jU8W7mV0+=HPYj1&>{MpWxuk{H)2&ne=_1yY3NJ~_`C7FH<Zga`7rbJG~3k*Lzq1#uSNCl1p$j0GAo-f)<JTiT3#kaHPwxuHbm>uLYE)J|pTFAVF)kJf!pQe<drs>*r83S?G^MH4URA^#$e6kdFt4^x(4UuL)AqkH&tPK5q*zHjfsMEfm5QAibelEhCa!)SE8jJ(uskH=r7_iaOt=N+1fPoYIXjcr7+-vsmHIO-(AKD#k)NsO?3;?g<7dF=bMLm?={~`f3>Cabyq}j`F90{ud4uA4r&nUp;h@q2((r#BoyNkBrP6!^o%02z~uyiAC)s3?Y8gb^p^*mS9cz>lxD@f|Go3^Vk_smdqzlV*b34R{*`j{#|-W#S6<y72%)mx>uEIRk?o*;ixPHl<@cSFlF*52L{Y6Hq!|Grl&d6Ci6VSqoJxQ`t+!OFz@NBZnWoU_<~%oAzrh3U)v9hAt6)T<xWimDp^P+2?y5vT|Z5aGeMEcLMq{dN6RnT&zJDuaR>;VY0hiRxr6KynEpRZvj`!)<IiPO*Hh(GC#T=BV#*&ks^cm*=*%|6TAS3O(B#_RQD)IlN|3{QiGq2YNRSakz*wEsQ|dUA+fl90wzkJ)dk8ad{gMALu>FpO;%zecC7f71V)(l)Zz}Cc7Jv^5-ii!V~QxJdSf@BNKsK1Ms7~00pA@d_JNklqD64Z9#OcFVip6p9E2Dw0AwcOR3$cTq*!%6}`VoCjjDp*!ME=s_X@~IRvRxT4RObJqp@>fmHZTMKc0QKLlo$_K4vHHlxxfY}Xi_XVRa+gQPajYlWt^K$B=P9A?3d6vM^CR#`5Q5<}6HOS^DIl1%zgD8@F1T6Mu<6qW_J#+p&TV9cYRm2a!rW&|x>c?t%EGmH>DkVDi*SfJx<ofmVX)J_1z-3q@5hNg+z4Xu7F?_pRZ<(bruw+i6gm<0BQh=FAHc*7e-VrDi9d>B5}b1{qR^2<;~ngxL=-PSk~N&i1o8&!LGP}rcTuO)BxwW5-aLmJJ+bnN<H<c6&hij~O{+CrF8$e#UZs6So6BZ=yDI;`A+d74InWA<#heC8ww)|mUX@e|bHaz^sU!@<h+2t!(|cZ(ts!R7I8^RYeMnJ5`oUuLtb<2R}}vIK67eN5J(5Os(7!_B2kK_gz}qjJ-_yjaG)otczL7i%jj7L4HHRf9aDpGmOD2DI{G8NMyHil+72>ooPx{OBDa{I~%82K0jF4R5RG0)XO5ka*~53uZY8XwxNS?#;N&3@G*n=iO;x^%qkLeZl=%0j7m>!7oYGvX<Cjd~wN}ac0|7PKz%;gRg4NK0FDsrpBLh8Xw5W?@$3eliDu?ljXj7b`ndYlzri&aN5DMV1{#!m)Km&%MUd@83XszUa-cU1F+<B+oXrjMxhpR9_&XW(ImTBYHvzfI;aGE_v#~%?5{?P+>Md%=z}se%z@#{Wo0RpZaTo8bex3!_M$GqA1EyTns-rgP3Ug7_QiQ@gUMx~MKF3*E+*t)i)JH_554in&e1RcvD8J=pIaam+0u^R4$m+ziC(bwQU=FcLH1VCcexNDD-ukO`SFy%l#Jode?-=A;>))ep~l%DXA7dAigkj%B;S3~6W@f4PeQ-LQ>=TB+$#v9wb`PMrmt*sgs7*b&WaB0xQd%)gjT2T8ThB};JXO~Scr!!!(L^xatQp(SN?61O*QtkmU<~D9e2j6*O!yOq8x-m(3Mku$1Uz~7r#zs>)zO7y(J9p+>GqZfq-g&nyBcPakF746gnA$mGL}If~wtEah9C8@+Q#TKflnFP@YP*K;BmGpl@Fq+*&VaAY74B_@!;0v3Qoawr5mc_UGNrOColBGxw{J1{wyvHTMdtCMd!$@UKF4nR#j^w4ku!Y*%7UiA>tM$383$-nxbsWg#DBjr0j5^ua~-n8g`~>p^u}btZgWoqzp3%MJ?+98=wM7X{rRKqL4h23!9|e*2tbHADPG0w4%l!Va2rXmXY<_Id*9R0ZzB0ql2Z6RRCM<E%yNx+?o5;32_^n49fg1v8eWT88S5?n*{!Rpm3sCtRe8h^vEaiKHo=XX9o@=mEi>gQN>FFwVObHK9&fB}x8$V>#l=MT~Hdm1DDE;rF8*k8wJjPCdVn@d1>JLot1}64TKGaZkK>c)cH3&9TWzS<%*Q56Vv-u!frSY(U8TY0!otg6VmY4g6UOv}wa#IB_OQo3|;7qI-|norIf-lH{mG&2%LdveAFf{3qv$pvI%5$SMmH%2*BxH__@ENtFCV95yur8^tmNafy-!!SlO&VSup&hvAiDC|s3B4S*UlUl$#wHGqY9`GMs27vc;RMwA4F{JwvU)uWh9Q<0e77#cXpZQ5$`QB1c0x!=#;cjeOp7aO9Z5zI9YIP4RlQ&^hZT~~q%j#@uPNsOMt0n^~bf@gd&EVP5)y)U&IT`X8UbDUlccJz*$uox~l5jZZUib!QVpw+WIZtAFs<LN;IRqf#<r#kzBZrvk?qh$+jQvhXIO<f`t_Y65h4+J)PH$5*7L^-9YW&1sR(m-`$kGHJyIiV`?*kxGsc^4QZn#F-JP0n}yp2p{&o$7LwGh_8rR@4T}Dy4-LreRHz+D0$A>T54;@Sz%zgXC7{(Tr|BZQzK_7)LB%tBt?Bx~c~R&Cc1)4&1CHjJw08QK?R<hl-CLfeeO!lYQ%3iuShiCF?{sglAgUnQY%$XdohWGBMj$o@IOY91T1yP8!Q<siwV1E_ra;71pwM?Hqg5eht>uPvzw{36`oaY-{`AV=-wvI%+uDIdZOw<5RjZeWR#@Clz^NtPs`l1{G_wrj3B&5?^dGM?jffp}a;(-lG`maNjawhA65uL8jSmW(%{a$q#tr<kEAiBJwl1h2k-7Q$t1|!ZzhwQ8RH&PJ=F?3orH+m!h$Ug`|0^dQ6!_A?)7d+GO5}|9uWl-<LEOC!2ne9w?cp3760oo;;9U(g^bHI<V*_#G|b6A_$9Xvirs|N9#jM_1mn0=IUJ<DJFbq-W{FvK#S0ZiGXR2Us4KG7cx~44|9(O#fiep46j{+9LwEL{l<Ky6kf_%ulgPsEO;!JxnDF3uZ*scZR;poL16=5EZC%msma4=Kl;1GuA3(cbW?8V<_@F5e*6w`rFrm*F_|CT;wHg1eXjzk^$~H9CCgvPpn03e>!7;IZ$on9%8Q)#+$Te+YQZ#D$N0YvTMe-v3}2lepGw|~>qnj9P=`qfnFwDs-bI^<a#uIEu0xgPvY+nmO~0Zj079g*TcAsiY&|*gjomEgsvR=sZ(`ySJAEtRL$z)oB(=xSVP!n!r7Ps5Bp|9G>hJK~h#P^H<em~qnz0s-jTRe+c}6N9F&PnHM{C)zSBOb5Z9USW98a;PT&0(PI*SztoSW-J=3f`o937bqUQZcvH?s+XYLzuT<@-J=T%^D&clQuy>i?hkt(Bk)!s!A1(eT|#le8QsH+%Xa+=K^LU1gI6`0*C@CwK2az$cMrQ&fiNKFeW4#9<3^>55~0Vc|UN@^D-t2qkQ_Z#5?_2x(_w*B^-pMH<kt=Bn;ccx_75!Y<n9vWuP&mC@P;tOl<UmP7BjN}8lz09;|qnnoP($=d!r)GdFsG=y)PRJ|?z^^q?%3;(JMv3})6CGJu^KSVi#|28Rrobn&kU%K>e9gx+c=9tz9lNMQg2yQ{uwO({3OI=*?W3Bd{jv09xUNz;pV`z+*iE*`5Y1rqPf45?kno{M|KJgd?ZLedzDWM@EUUCpOAu0}f)X=N}M7L$EGf`Jd8Vx!@M`8A-_5(g-?Z(kX@3)3*#cJhpY}*jLiid&pPq>)W)aiS_WXGOOer}fXvN;a-!1~JIgk5OKyCi~8RjJF}pNzr2K+=i19<Bk$MrvV5Ql|l3pYiKKRHLCLR5cV~RjzrY=YJ7rz2dct<)#L)QP4bw@$@!%%BRAkUV`#nF|K7R5*dxqOCWDRxrUu}opzhD75cRT^QCqBYz)EpylUCDoUkTQPKGYlf&rrPb^2)fEm>HE=gJ>_`-h3``+D=*jgQboywk?K6+gz^O+<L&01adE5%8N)J8{HD-0h)D-8RR?HFdJ}fJ*^<l^G)rYZRVDA|N=}V~w|{8*8vsNhSe_Lsnbv4u1tqj4jc`XsAi}zgi)31>hbihy>;x`D`n?##`EL^5l({2Mh_k{&uZF3z)lP)+&#d5c0@ZyiN1eaYy<#O>S!vHlLQbOMNc!+w6?GSJ$>CY7S*N328x4o3}dTqO5If4x{|a5<5g@w01ySb#6&tS(X>vT*Q{=?$CES`@n)Mlh7jfaXZ6FIhoR5$GwOiD%mLnUFfZajjAv-TrLGOArwN@=3d}+V3lZMaaMCJa|%g}hc20#h3eqgIEZw5V}~l*={eXeSsbbp>QEBQVFga&FG*;o%*itAUGEd8j&ePUpc@Our?yIz3yEdT3{H$@1Q}NThUyo=nn3fxbb^{Qp4d(P?Zx`qG?JlSm6bwk#bMCRqQ-WH$E7U2=<S|8njSVI@---&l@1;*<H&KRUgt*|{A{ynxP`pq3+uAkE7DKe2#7i^u{}@B1!=Ci#hZfM0wObSeJp!DK*;T0&8zWHNlI5V24F7{GN@w)0x86?Av)+b0LI6&m>J+&7=iSj1--SD$W6$C4d!;4w|BnFb_LG#1|U3GX1N3B(C%)4$3VG4(}4+ii4(QZqOG}*@LlO7JmA?PBpoNM*s*x0yNoaf3fb%!MYxV@J90K#6WMBY40c&!Nz;>GK_Q<0%-VGLzm2a|;szXdp!oxyryfh%R5b1a#Enj@AS^YsXPvn$HXmAL_()PNai4@Dh*3K^O3QswOVkE;Jq>8#GqqyNg#xjyX2D&!D-K|k$%fSAGZ$L;)?jfE!JRdOhAE{i*Y?JMUE18+97D%Yx-3bIdMj%ekAicgx!DMV08TYYtHFOsr7izAZ<VwgzStHNsw=zbggg%6E4U9FL`{rgF-0Y8LbI*5cMF^JuJqk2hgS^tX2RMZXojd%Dy(-G)xF86-#&*F`vjP6&S%jU$2TI@hw_QZTf^xh?ETP(b_`BcGoHCI=`q4dYsy)%R({qBfk^LUlCeX2$vYGJroINgq&WQ$r!fC_5>W2C-%p_3v5VqIV33p&8EzAQiiZ+y(kT}Eu((_GXTf%;PU1T$ZNv1I=vqIh<JjDXk=#+0>5{SeVV+xPpSa5#YJj1Bt-(cy2Jzv-gn^+HW3Xw{jZs;Tfww+327MH@GTlLNnPki6kiFl5wu5V-w|5KVcId?Gq(3M$kI{?k(|OcZUoanu&hN)_f*VpX@(%&l^p*_qi;FW>U+LJ=Q5Kkwys0jx+nfQ>*CseX7VV`7et;2aF2V%Z4j<hO&;Sa|FY#}hjmvJ5MT)h(o3`yfY$Mu%z6KugV_-||pR|{4Bh|9KJUPhlp2x*H@H4A{Oi&-jy{%Pv?u_ZO02hO`1r`6{m}SB{8MiPqA6fxqh!HzYQ!rnhW<X3I;JvcDHKo3TZ$pj#|4A<ZuUPryU+#{WOe4kK$*UayU&D(bCOkpmSG3VgSQk3Oa^=}0K@H@F+?6j_85?~J&`zP#;(RX>l_)}h({$t&Rs4v2kTkzrRLB^}v0Nb0;_551hai7c%Ja6s<lv^XV=qp!Na*r4%yYMX>U8QSwKvW3_p~WZF$dyV6X&_vjZNQK<kIzf&OxqFozRn7z0mTnErLfQVeV}OHu0}mDi8^a(p9^UfphaJnp^B-1w8%G%KEw5(k%m5+JlRV1o9Qi%8b^imGt<``BJXCCTFB|TsG7N32wfVf*G^-RfE?9P%^I))d)KIJ2SC13s=bCt|RHpZ%fFqW(00XIAoXgyv6##20DL2Zd2_%Bgr-UPj}8$x^~9W@tPv$G@d7oR=0!bV74Hmuvqh(C0Jb}&v|}wF4Xs%P*Yi1YQkw^=xaY$j2IU5erM5m5+U6Xzkv^#P;cV%QgK1Ac9GyvP-zy2BSx7a+_Lb#Wg>3!^d2AXI${-t&J>eJA}|ux0iC98EbwDIMxo{VaZ4-_P=Nq7@-<ZbumQu$n>t`e$ia7(Y7{zouzF9ZmY>JftvIb&&>$Io9PO(!19^t&Po3P~?1r1tzm<@-2(`vDS!Jp%2b|!=xLS_(j9ojKyZYLjk%=@5R5_WAYr9%EE7=?g%}%TY&$i5XF85&&<g@1jy$(K_t0tgmZxux7*HXsnCk(+I_2OQ-^if0)PDzO@MZuAl9gkj+QR*qz`oM*?8;s7-C}P2Lrg>J#a@lK3N0!fa+&t?j)sU&hX{piA;F}EL{oZH?L8_7FA-l|<oHI$GIU7^}uM4_G8Lv(-D+M@D@v+sf*u?{iT+07q2t$D@E?JU_OM8h15j+X#J|mRe_0!21+Z4Jf?+6`J{E05|Sh6s(6%7XO0T!>K?kPttF4|ggYLhI57}H8w_wC%RByJWLV6lkN@VKkd4|aJ81gMxMXY1>F*>igbcI$t4n4$Y!yRP=*BZCtEgQ!bjY`E(rmM^Rz^^<J<3gQVKwzAsEDd2iRcSVa*MMA@Be#!KNzR9d><G`nNMT)opdY3q)*9V&;Sl=lxTTFnHgXW+Bjs>uB#1WL+4gfyce07}PgEu_a5$2l;C2Sn_cdal=x<!CATtUHh3;~yU**#RM7)$C{V02M!eFz6XpB>bBixTA<e8U~7W<=12&1Vvfu*(-fm<ovy?N(e-`OkpW(k@(shctL$zn{_P-UTIStUJs7(uMd>-lAM|u=HVOb?wO$M}Jrb6y<S=^GQ=_4lU*V!_;l+W%wE>8nqE_)2dnL%3vum8-5gk3f1)6_%UM8RM`z({HPbm%>HapEFmy6qcJVDr$9(H4><bx1dCKcv0cF*ncyxXYaaPRM9@Cy?7s#hF+XiIz%##`tNO|S<=o<)Hs|hh)MtId=Ah+aB5M+0Zy9t6*J4=MJCb3z;C-qU@s$On(zao5J*ZdmWX{odHrLcPTa{>u0sK(B^8T#SdN^HU30C}J-stwF5?W|^(vMfgSM&OmA@qUdpd@S6y>9x|3i{lNAhm!Sk!1Gty4qj4#N3W$WTkscOVF#r!%rrnomza!h+{U>0adfbk`3U7NgFo=CM!uk+K{c=Za5C!X!+Wc^`)6KRmt%(aFaB@>7B(Gk&Zl4@+e+LPSZ$6!!@b+a00n?Q>g@v?@sORuIQnV-OUjUgAhF#s0ccBprMpKU5F2oV`Ai%xLlU-piNTv&i$#-B39?1x<D9>gSLCHGAHTtA1cCIQ!K+i{j!}DF$0bE(UV1@$1fSQQuxF6i3f^s6&59eq<ZL=W!+lCZBG4QawG+m9AB0uW`fY4Dc$$i9x1<=(MW7;?)`tbpQ*Zh!AJ3TV-PRNs!~IOL4!l8{#Z{%D2jRcMLF-tYL`qA9VBBbcFsVzR!_Qsp#4Sb<Qw{w#KvE;u!oI>><60d^A;dHjzO#-r_@@juDKwMk<MMW9&j7Y=awfJf=(rRXEo32tHN%0?~{m5dZVzy%5;z=C62IWb471!ba)I<)i8qqYRk(s_&Iq8P<-!+k;Pg9$FLWt0}mG(bBms}@sPD{A8g2p>wG;x*?m9LlrNc{fFfg!+3Rxez<AgSbB~+HvB^3)oDE+<02|hU`GGT@f-)RipC##8@cgy$6#-U6ES|2|cFLwFOKBicb60g5mgl_QaL-zq#lPs`D(~cbC^P?d&%o-j<c|Jti%O%6PWB^MP4&&4Qen#+nnNC`swf0GvN@Hn)C8FqXbwHTqnTuizN*x!;_g&u@s4?rl;QzVhtk$Kpj9qK8I<TzLExwA<MI|Zl{Gs71H~o^ru!0O5}!?yM$V6zJ4wlhT#YIjF=;G+Z|@EbJiX1Aw~yr*-4V&@Fye#3QuSRsK=F_hM<V%Yw}|?brm~)s92(7uU`2w5Y3*Y|-O--ky4gfSP_pshU<>B{eai)eN5NLcsJ`-Z@pp1+k24$cnb~Jo^f*Gel|JO^2RjkI-H@DCf^Fc?st8cdhk$Et6F+32c97543LCOdIjKEdLSho|K?}l1%uCTdFi3ffejR+|)viRXm|MqtDOEVEwgM{X?bA>Y!p%W`=4`6qDCF_1YK|^yt*U`Eq1Rg}VsJK30{V#3spv`S#ZoPF<|@1YWcpE|AQ*KiTbesM2G%~x&wQRVl-MG5{>9*^iDlt=IBu4~KSzTaFqC;Q%D{fjAbjj4Jrok%mdR+VpsQgG<arJ}f|P5)Dm{mh8Y{2|DYm=W)(p6*S@l9iYEzs4$ZABiU^%5D?QWKd`2|z>)U+~{7UO7?aKVnu5M$baArG7r93f*sDi?Hs<oDE0p-JqyXAC`){QqsJZtNlA*nqvSfm{r@NqsyiV$%z51PajubJ8_`c>R85Of@zD4w7CC`G@jX=c_B>Zn0r5&bd@<et>DPq!LEp9qw~oY=>_hk}mqR&S@F_>KlYXu&}6l?(eQ^nR}`P9Nj-tB7(dtLdG;0<339sL)-T6;#aiHgYzsko$t@}KiJ~(zu6*%oMV#Y0vJ_lQjPMk$paE=uliaj*q-A?kyL~c)Oo&o#|BIZU+Zj8m!8dUV~g7cKIO5`4x36pUemfzoxnDb58JLBKCsUbYrSfjLUwU^s5VM$DFTbv>5g6Wll#e<@dc##_IrzMCv+h{$a5|+s}4UB6vvuY7b57~Cgd<41-86IO9TAORmPV&wEf&ukz1Qna{?RtP4YAQgac`<?u(m|A$*q(<>(Gi%1E-V@X`k&NAFL`bikpqY0AAgyL!B1CP@5=f?gTC-FZ8i)kE4(u=u5NFEzZDopRBix<!m1eU3LtxyuB}@l74$0jj&FhMeTTubG;l=2L1ZHHTeuJS4JQ5MzGrYKsvU=<3tausUMkp*ngqq<f=0+l(%z1o;!<LLfP%@%6G0;k16zThCLwvC+($2-3&Y4%`?DWCh!h3;X^lQK|Q8?%IM)SA^|?=c+f0!>bHIw^11EqP3lj^K7?tQv_$^prDe#C|ktf$RjK`&j@5N6O7K}|Bzb(90UL#q@0d0y}V6Yoz1RKtoOTZPY8$>wq>!&9|6D(#XzRz_Mk2Z=?tr<x9nKp4{Dx$HjnE6gT0R!b!}~fu(#tp6?p~Ep#5QB8ZlymLvDE{wA#a!OXzX*bmLB2TVbu}<+r^>JQ2{(i#{YvjSCNi$(!LX8$Ys=S+?l%)k(2>S~9ngJw1qwzWwP9+(xK->R5=f!|jIL1N)*=rR%VUFV>NFa${LO;<b5`bdAIi1SVPU2LiX_<GkXj{FoLU?zT>%K4qfCytU3Wuhy=yaCg!Q&;MZlJTxGL*_;n>M$ZK{RcB=*FQ}?RJYCE9ysVaY6<7Z7k&_Wg`LJ8XW`qgAnApbLl~V1~&+}~e*>^A>0*7NCPML3BmI!vuWSglrIqTZpZG6~ASI$lvZ!5dP!wE*9>^o)rO?h8lwnEK$3DCpCu+CM8kbMUmU&zE>UOJIAd06hizz_6)tHMmT3)lr<iuGoY!>o0AegX;0GM5$eeipN5PJB&*O&|eix7Am&Omd9U@|S8Ctr+8js#8#bBSWz3Z$@uy9jmGpEz`Vl1M7;y-w0Xp*)H#PSsv7j3C{s+15foLmnp#*iw9&q&?DAYD8F{7cjlDTme}l)G=Dg``E<h=1%7%iv5z+S%$=63X)&4QUGcUG3&V%uTs>r@mZ1at9j3Rt&5_ywWZ8?CqR}F?00H#e2#>7`^m6s0vBYQl0ssI200dcD\')).decode()\nassert hashlib.sha256(_s13_body.encode()).hexdigest() == \'4df70b73b138acdbfa0856fa452ce0ad4858357661199344fb2dfb1b87185185\'\n_s13_file = "<s13ab-body-4df70b73b138>"\nlinecache.cache[_s13_file] = (len(_s13_body), None, _s13_body.splitlines(True), _s13_file)\nexec(compile(_s13_body, _s13_file, "exec"), globals())\n_s13_lab = lab\ndef lab(deadline, weights, state):\n    return _s13_lab(deadline, weights, state)\n')]
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
