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


# ---- h200_lab wrapper: payloads s13finalab_payload 963be4ec7c3c4827 on base S13 958f1258 ----
_LAB_PAYLOADS = [('s13finalab_payload', '963be4ec7c3c4827', 'import base64, lzma, hashlib, linecache\n_s13_body = lzma.decompress(base64.b85decode(\'{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;WqN`{#^hOXc5f@^K)lHO<*fof4pu)l+@0?H16~xz2zWQERsfZ<fy2LtAQbcBL?s6oo!lM(iMCJW7=Sg+{9CgJrD@~vz@ApcNE?{z1hQ5hf(pjQdBs{0thh{t1|ZS8wP*Nhf>g$CGGbW`G=0M$UVkRS#78y{7Bp4)082L(mtw0b(fQ&`fyuL=Zuack$V^k001HIo()@$=taD;yN{debycx%zrmhxO4jSXDU!gQC_QV_AgT0uzkYP2qE^C6Cba`mJBqwd()q9o3*;E&G%4+}#IFo~@|C&N`9q3uC^LK`k3Ol|s!>i3T&GW4Yx5>FY9D<jdB~K|+yBE~KP`zwZ|~u0cUpUdrmv#{zI>-63g!oG7EO#vd)R%{oPe7)z8!w=(Hc+r^_2$%xTo(lk33RZ^EsuK@Q3I-b9jm+;22CYY*qbO*yZ*@Wtoa)^b3vtaL9Ol#+~bAu85Gz?mW8Wrxuwj{ynP_pr<F4K9gG1%}Zw*SbncHfO?!1#6RB8)Eyu=C!<>4o^j&wE9cnuj$3!@)a%t_Lhwc2EY;rMca5mGr995591EZ{l(6|Lfa5N;r`$7o*}+-#E}*>br$o&&AG0dLPkKG=V;CksTrk(cwccg~1wrN5c}$f_9$#>1oX~IPR9IONGStW7^tx^qyyHi`s^McQ?BIG0w9OazDK)x(3y;k`R&krBlFn`f&x05<yYL@LR{+ap_nZ%0;#0r2aEkq+2yI-dx8H9@bq&3-`BXH2b?;rrPBGlp3V)e(I?KwgX0>?YIzO{lS=z{|nYJxEuA7yLN0}73rC@To*58%BTsgrOLw?v5;$=XE$3Ae}Hdf@gpUt+AKy316mIP>jD*nbyryNe?!0ow7;^Jhc2m_gnASYK@iUjXUj*^;*``-}`5{9yfM&f-?5)GH|xS41~xs$d!Jyws<u5D|`otHdxV>GC*QmzHYaZ9l#Pao#yOX+~L>o8A__S4lx+KaW(^%Ri)w%HbE8wOQZ?s*K*SrVVuw>yal?CJ;}#W6nwAUS7A0EXI+B+4!iqh@fD3JbPh_wR&Ro=TGYD=Q{$ZkaV7L8QRwY5KEq-Nc0St>S=Le>#-NjRn6v&}3H0M1s1jC5flbXMtS$F)yuV7=E{1wDhKWqk+UZ{YMVGl{AD#cF~n>e#up&SC6OU2}Ng-SiOS;ryzAPXGhF5utNGp+qj!g@161f4L-r``%@$gylxawIvf(?2j%pi{2esOy0Y+gc~i;(okw(1ix%iZ*L_sE3uMqRs~wHBr~@Yel+nXh9|DNK&7|XuRhIrG1m%?wQbxERsZP#>Fan4KAR~?OxqF-n2C2lGTY+_K$}MsaA7FBOZzd~}WRw1%&{_q9ts+*lL$SqW$(f=mo+Rm7k}MSah1dfIQYQ0wMZ|MNkSuuomK{_M5E@(ww0!3vW=wdthhfAzJ8n|Df)Ux9s#_utLBk_ypl~o#U-)U8I7K4h#}zsOOOQXXZHOW2o^^`_2*UorcDRiaK&w{F8Wdk#;dD=CgHvEW+ld@+;Y$B%A{C#+`8(mUm!==2<u$3KyoRkQWOKM_dAdOafGO$W3x4e0k8O>hffJ;bUWK;MSKRIK1Hx)=3cDII-pI^|!fubZLuY>U>{tj}#8gWEcAV+!!Fav+-D?h7Ht?^X6js^P+Cw!DgBJeZXAc3pFW;K9bP^iV-cH&R`ZxOhR0gMWp%?A%sbE1Myvn-57x{S>-xk081+J#y;#2mKI~uSgZ32MfD~8UgM-UXCKz`ZP;X`Jt5VagS-nplL2rxy^3*B+m04}yvI8our!MlIL+`JUx>A2X1<J-%W_ZroO)8_H|M#jZjy$2PrjR%gx|EcuFs(FL?BOE9@#-E3~Ell!0WDwIpTe!Rq5j&El|3zx_{-iuLrATQ)8c=oB1;Nyc{T`qzkjJ=Ux5rOVH(`eV)ffr~(iuL+)}8R0tOViI3~=vtJhTZ=+x_qAOl`&M7-VmBZ6|d8aUSia=&`4dEA&o<y$GC}NLTB3GD{lN;}$UYz#7I0C8oiLDJ+iCeS;)n&{nVlvdZh<AAF#F&6whUOzlY4ZJcsBL3gt<%C>biEVQtw{fcyv#_1FRrAnHTeU^&gmB_z%gGB1|`EG~nEK6*D;+2J+g{^kEK?2R%<FcBhZrmCGKxNBl)c4)z7R5W8-8!N<7|mG@5m8FP>DWphK%sSklwk&#L8Y7gp(+h5f)oq<v<T2@_nc`}fZT!lidzVTrAH>HSX3h(YjcBsk(h7Lf%+ORg#do0&Zwn%vjK4}Hfm4xOe98Iq>d|0LkczrHU|jSwsB#nTT#1SJBRn$pK<DPQL~qw_bCOu-KWgknj7B}J^3$E)H3j4jmUWlyWvTEZUAvs8;b%hX&a@Gi>vQ$Fd|>kP2t>58jGCHcJTaXx-YMS>1KQ-qdRq*<PV78<S#S@2Mwi^#85I(8JF+$MQ+QNiMF3TYrqOdOM?l70=~ed0hZ`!qTpQ$`vBS(Ifwc#;fYuR?K^n(P}a_nH;I|MBrq&>9;Er5sSMKZfI2SRPiaL7O@ElaW&aLoeJ$ay6A{AE`xYGWNCa65rH}yKJAj3F@$R02IUKJKOP^i8>LAAY=mpKjX1(N0MaP*mteB=-h)-=OFH<h`hdW@tp-decpWok9$weL-S{t!`qOw-m0a5U#NS*V?jH8-Qr5?pLncC)imxzmeE$nl~e`fboWmu-EFNSSuZRoWhN6uV!J)7;`JJsO7Db$UNrsOE*$M8hPO_DKmw7@^;j@uPSlURK=rE09L;)9dx*eM=|kd1C(ujAam9k5VQ9sve>_%?x2{7j;-Xr<2pr88fgqtB>oc~W$hsgQmXVG^Jf_<JdqYD##a0B_EWB<Gmb_yz_7pL|8&yLTZ<y}yi>-uQx7qqUejy(P)Vy{aUOAqqv6^c6ira9h^m0+3D(#RmO2xe_ee+9zRb%|!A1`s$b7vIS>$43<Q(yP;y;{8>R)9{u#Dumh~9?ibAKSl9~JR7^@qRkGhGP&d3N@23<<eGLw9j8Bxb%dOl&Q4m#aY<n2`_hd&wnEPuCppJ&I>mv0UsADy)0-LZzM0^-bMmDuD*nu9Rfqr9|cFwo#8VT!HaSP90xRl9j+c)wa*O}fsQ#xHhLggK8^eG$7$YUCi@sM1QN|ZIVen$GZ1gM7U0kbhCio3h$g?>#-=wbRB-95SW9lZqg3qHNP$hf8*flRNwN=-$L>)Xad!=1|H`h2rg^DbO*m>qFP73TxzFL)i`#X7@2@zN!A)$Ulku`ChR?YDXbDgie;0cDPVwac@zkz?cB>?t+#=hBfN1RYXpr#$kaD1m<^kq*nXu@?<%#j==FH~#BO?BUaLL)rtI#%e;=T=`$qZpX0MCUj??3AqwSNCE>W>K#USOimPt#ze$9osa=7y~%HTkI`P6GGY}}HNy_R9a4==a1&#`L}^ELKwaua(#m3|P}Qf0!|U>4cL+6t7Ig}w5s`r0xw!o=X~0TtzI*Mu7xn3J-&f;w_=`@Fmb+P~8$!&zBR;bR(39z@7+eX2-ThWjNa@X+D~<1^8y29H6L^N94onW(9ltbQg(W6Q{lzizWz;fx|1bE0D_8=-LLdyNQJ6-f^gyv?K?vH7w=!kawWG*jg6R8h`FD-<CX;MB5L$IMj0A)B^vnN1oBG?$rAQHIt))0c%^a3zT^mYsl8;_E9uQd|h{B(b9T4lGnWSwP8o@=S_hgKgMFJU=LI`}htOIDZIDI{tGA0WJkBV#YI(IDa9z2=?E!w*$TH1s>X%gBeyXFQ?@TS$YTLQ;DKz+K4ZTUGo?!^`z4;Cs@Mdg?B1E6h3jk{=@GX{D@eKw?rT;0H`L~3}9g1Ut)L|l7<y5@<6Q1Z%{lvV;<y{LM&b!!`hV^i+27NiIMFS>^Kq`u}{@I5>zqR@wF-M2ceLc4q&=dQ^`Xp=LFbUSQRQO^AM`Wf2JH$cxJm6ZlMPN6s|!7v(Tgf@!9xrcrhmA={-eVW3UuxAQGP~1DP5y<yB6JBn~s)roXbLep=StpG_NJSEM+r8D7(s2746A}tIR@Z&V%F*L=7B0{Eok>!*=S5T*k=nqON8uIMB9H7yuezwUQTQWN>rEP$j91N=cdzy9lx-(1t9_M)lzqAM-084nGDNi3w?ji~>-<AtM2tEKly$T5Ecx5pbMd8tt{&`@!0-nl^#7{1)RJo~)O5eKb4lyv;&+~-j?x`O(I&H^6OEh~hYn295Mzz$Nba`x&PQ4uDcP(^vTmwTiuFIv{JW0ahZjoi*(?agS>9_13SXRP=7KYdq?EmwcIZaT?S((??T0%|iva@G24Ks~zYKy|FeTk@+xNuX<>OBsPU?wha6=LR8i4s)qleK+;`V<5|GveZwS{EHn4mj!WZ(b2rtNQ|Uqpm95NNCBylm6iGq^ONec9tCwQHehQR$=#@+N0kPUb#8#p>5pC;aB*-STm-tk-1bLQ$llGc!9sg=wO&-8H1nb?{=~#*Xa`U)5dh`^jKkEEe-W#OAjN2DLHm_4L0QGe^hVEgx^Y`3rg#zlJu+?#I*g<8*eb!fsDx8wWDh&J2}=$G@s69#Hb!HwMD@K5Uq5E&e4Tpv=HvW58&Ze0jv11R%0@rZ&bovOpZ!n21=ToVNh$b$>?=zYYrdsjFC3Gi;x_mW?I0?n`vWpqk9kTcn?yJMGdx8PO2YJ(IU$Z_O;9;6yEL1Q6G%T%QN<Dq+ql_Xn*=6q)63+lYy?Pn0d+gV4l-!ZKa7w-X;_t*1sB8~C2e!S&&)74do)F>sd8QmteZDJ-BH-6#d*{&rHw=>1#q_5Xb4>}&bq4Vs++7Uj|^@=bbWSJ5|sSCg*kK-oolzICk~iWGEZW<z<YThfk#3m*l(70mY1mA`r?Xnt6&<PT;`DC_k*Qn9F`q^lwOWo0s`Hd8tsuW7m*?=c~Lx5O53rNRf71^L06{6BxykSi(l<i2(kkI2KnHvH2R%n@mW7j)Xp=gUaMa%WspCp7UNY(>KseN!R~v=VEZwQrAqUe%s))4-r4kGoz$PR$Q4f}E-3i7wZm8kg+Y=qr4zRh*;}1|nn+9|bBn3xIr8AU*n|vd$Yj{^%-=D<a}JD5ig=!yx>g#|Z_w{ttJ;RM}O78Ke&>`ewi8{C##+9xs}po0fqfg;@Zge1_J>BFY!GNp39AMNyd_vlSz_1wRShN6mIvmTq^diTc{BK6uHYd3()J7UhR)zOytjEUIbqyT(p`pliL~tP(o1DZPP!sj9HjhIo-HH|wJ1(>L_jzO~p-AcIrgmlfN_NE9puY!Ql3_%}{BXsPGd8s%?f`vSi9pQMI#kqt>3=0dRf$kQADd72afn-3N3d|+!SJp~v)g={m6(VWPsq(6lt^*G<yOwW`P_h<3RePX&IB2prWp--P8pM^=q8Iz^7oOx$>XybcdQVgDGv7%+0t508_iDAVv!xuRtOJ?0K8Xr}W!!zB)n~YM~9;wP$qD)eC1xYf;9$|n7q8fUBSmN`qKpA~o-CMKhCoi*Do5aG3ki+rKdrf4jdxe1wEMd#z&pSWN0Tl@xb<Zo31+v`dyl{i2uV7zN&7n1c#goSML)HmNurG187OX+@?=fXBLOxuu<aezpL*PGY*m^s?vsTc;VT@dA0aT7t>@a|$W?(QZZqbS&)Xmoe92|32y?;<N0sNS_gyA%MfH4Hpp2@17Py<JGU+!rAV&Ek%59AQH0vVMhRd1!C;NKPekp)YM?Ju>uxPdf@tLZGg^H~%}NjQ2WVXe&sr}B#(IO>XSaHGE?C>~krCxUiP-@EJ#W#vu`ha(dg6a0j>?c;T{%}^grr*fge$QkB%=c2fgTV5o(En%bzrN;JtVt+Tm^Z1Eam|)$eliN;eu=NgSn1cmkUd+^PF1(Oq0P>bWNdrLcxWRvpUM|d*V}UZYt+TBltd7!~CG??U4j7?&N@F$ewvaq16DcW0lK%!_imb>54DLb8Ug1S%FXRS!_cP;bz_uS-YM>qlqHge?AMuw}w*Cs5|BJf!#@S=7NN_L^+RW(xM@E<Yzx+@u_m5PpgE-=^P_M+&;avI+E1Tz@4md1+cmYfRw|GK)n=53P894n8*O@}5L>6uc%R@4AO1RSex7xushh~nUU#m!fZh53tt##~BgJLRT9=-y&twp*f>h8pfh)Q+hvR5dRz7Mz)QUuCR+;mlk;{5zga5kKMj%lFKG#)L`u=UrC(J`}L=Rxj2@VdpJ1O*Tn7~xh0Taum;iNjT=Rw{kQcs2tw+8_xkLxQQXDYn*zDhG2y>_w1h#Un-TqzUGj^X;BPvYxRa5LAF=d#99vo77hS(yT25j5{ZT!k)gL_NYGn|H=L>OsH@C>ix{!kLn{;s4Nb^aDfL{UL@m2)Ssh6iX9d;0tcw_UKSIEQ$s9lbrc*YGCnT9R^;Er%qH=R!4@oVXU_c<zgMPsYrTJP9_MpK;K^wE)*$$?U{xcZUuu;y>g?ff_a<L~)&ZLl5(vUF2DDhTMC@Xn>(-AtoAI(A??;r@abN|J*W82aLg1jdhWpyi84$o$6f5d7APwdu6l+b=Tixb_S)uRrhEB!f^z%wLJ?H$0fXRcxMZc_%gp>Rk%2w+85}5CuJW~;JbGaZ;s&gri){jJNxN5;O^-Z#bNF71gW^bi!|IQ#VPk3KT0(Y92>yIJRX$9J|>{W(cI!A&(HA>d>1pZO6isZaYrWHJh4T(KZS<psOIS)WzODKp)!tMUrpxMMGpfkDl1u6%ppD)nOV-7tpXfOp6gyhqCd!oEjI!N+GdB>2WDNymo(TRJnSC(-+Sc%Gyh%BAu+%?AG`<r_cnJ(p;;RBdD$8Af%xM}|PyfDEhTAT4QG#>|%BDC3oDRn}z1T(|xKlPN8ETA@x2%Y#QvZkvj5p>q(HMXls3A>gJ*Ne8p2FKb!2;=~iXLr}!Er1;VsrDGZH@##<*yGnc(WNz_)=D=_o}2vXQ+%78(gTTg?2%%c)Oz6qXUutV%71>Q5S>M-K6xISkxnDjR%g49J6?(5JMFic^vClq)dMb8kuY%GH0jT}1<($US|Pz!(+W9VP4`9i{oOWuGM;yR_@pMG(BrrU`mfu3iMPk1UxX>6tqIYWelHq$Mn5)^BmC0{G<<|ZT<+CTP?)!b6Go(06}!k-HZYmMPBDXk7oBKrXbZNTiAoxtq4t~Q{-=tk9dk|c%XV2x2XitBe+SQ3>JJ&@H9BHtOqQH7mjEgx?;!glL}vhaz9Yx0ln`O=(J&fny4s~DO;_pVfUe1f_M}iLJXeM&C^NJ<)l<zpKoRgYa;};3L164;J<cvU5?N+jg@FZ0cMx`->Y^lGzj{m@=CV@rRooOF)(Da@I^AGOO=NmEy*)qK@NT|e{=u&`Xk<G#boj!V<Z{?%S{s(XT3R0m&vg?XYkxHx0H@lf*GqwuD9sRdn8M#n*;Aiv1UOCbd|Md2U2tR-HjlI*sOVT>3$>%*7n;z_FnLP`QNvoH@~2hN2Aa3~6cBQY&by^AB7B#)oG_)ET1pute@jzbk9(R+A&j22hdh3z61PLx-G1W3_`)I!Z+jto`J_$AQWW8Jqid*F24M2TGhTJBq`ByxE(5-AN0c}{4UX`aN&XkP;5wn;?A+o-{Pr(Z-Gc+g?kTgKl)lJLb2I5|!Ib~zRPtm)RvUR;uYwj6FI@^aNYksIW(~m)iT_?L9AeWw)KGy1&XUI5u8Cd%h{L^1WWK%FNHpI})fOhd8Q`%SFEyx^`=o8Bm!+`=wM4Ogik2!}S~3&V-D${jL@?m$wE__VTd^JT=8}a@cFsEuf1bRIw6BHwf}3eBq;7<;c}Qaf)081+oL|}`=OLx&<4Uuc?tS0VfOk0&N~2SbzD17yYN(gA5e;L|JM~IIcT=r2JvZjiZj0qmFYkWDKs*}Mh~QWK;!8`ZL?=B10SLV_fX?_f=Do1+aSJ&Dv7~;%%<@mkZR+J3y2#kg35wx#jmew&Y#lKoQ#S)s#VwiDr_yq6)42TE?it_-D@JUR2ZJvSfH_wV-Y^1SYAcTY)g1ATa6<`s2k{-euvd^h3d@$cIUwu)eI==QRX)L$%9x#96H7{uYUSBvRPZxU%uq`)$fKVnBB!kHinF*brV&ObMw=bbyvwx$?eSGLUWsw}tG)Vf6>6UPiU=Ldc>rUZ{~q2%?@!7>^`~Lxq7|f_r)`Z7Pak#E@HL^<2}OGSf%S(MrDqF|=!Vu!1eB>rYNJ>I{Nb;Z3e+9%eEEeo?%j!H1#k`3NVj(8N}^@x_o!LD@L&1n0dUC{<qa#L3OM7zd#<bElSXz>er8JHPgI%<%{T?=RB~O;PQb6m^Uo&XV$!pg;*|UHSaFcX5YUFEWrshel{Upr@ZuLImyRbChpFUVx!zKf06}>2X{P;LN~`I^;gIxCs0k$T*If)^RkxLW_?dR7D2~*xULED)ZcX(z2X!ru8BL(*>&?MidAVU_$IWNN>LWJyzcGAM2J3WIzDsR8TY&$^fq0ivTtNGUxIONurA0cN{1QGeVuC1=7CSz!{~y45wkr8ju)<^-bf;=o*2A=*a5{9k#8a6HE%9qQr_kcRlYvR$`Jw-Rv+C4+vz3C@%Xlb1{8hDLKt0YRjT&eSPhhj2BmN@1x<cF&g>wKwSD9PD?W|2NNK_V}x%{x=k_S-ca(W#1%}lHXCl^ifemqhNhMQr-F$I)Z>{Y|3U#jcSSh1<gH>?tRdWkGJD<O(unF7=6UNOdEsjsXd)@3a^D5)IQi%<6?LVwOD1zNlI0Ezt-E<Rp3`_E>2b^j+Njz%yL)N-fvvkDO{S?o0b(^fCOF+P+Cs{4$M0&?{>AenI0Uz)amJ8oHEDIXD?sxrgALPGR_y!Q|0-5ZvW9pSB=ZjkX6=E*Dlb))c=rm%@m)A?U43-)WKpGm)i6&I3ka~#K-B{}%0NMgZ|>$*M|{~h$8M9RO^KxtM}vh{Oy7W8a=MCDa+WT~LuOCLZy6XSc1*^WyN4oo9FK%YptRT8l5u+tH;00w8GMP(@v;9dYj(7DfwH<eKHcohh?3%!g@!i%iY_PnBX`7PqqRe>c$==rP(n@4R+H<ta>kYkMCNR$Hnn9Y1L;F2$mZpg})r3-HTtV+DqC(aL$LJ!}zHrnUIfhr;wTr}n?m{ncWeLTyZeqN>FRm9U$QM`~H2w&I@&v6fsRe8}zuPpdg+J{crNd8e{PH7i(n^%F=kUqsn3sxpu?R$x^yPC>2z#pkXQ#zlK(XI9Qlx2)v(ATV7;BAfhL(V8msIr(!-FxGRhx0rW;`IP^CIM&f>i!vQHu57<)>6GDYn9Q{xu^+eC?a<FSam}f#r!85sR!WT%Ko<@Y*bUiY@=&^GmE2zEzBKl77r=<mtJIA^n<uue-#|sq#x+$2?u96D{O7H;^L(fKk39~H30k)gAqe6$GM<b8M$iQjCS7zh*(I3-hqZ%oadj)Ij~|2op!&u*_{h^QCa&2hr9(a<-bYgy$wtvnWplX&@DmeXs^l<dH?j8aMmCAAURcLn@urtZ%e!$uW-zo&VAJj2+oOtHPl~vm1>Z{w}a)~%;MgEZaFS&UO>P;PfAoAsJd>Je;j(fHSY^0^+$870u9)b_1;d(*sXd5AJ-`alx51(Fsenw78a!DxjAD|FDd1bBi6P7=PWF**#cc@a{<26-AtF`ZICN?gvM`R9|*7HIIF5t9*s=rV&%IS7ccNd!?7sykLcKDxSHDq0@}wLpY6wl9_CZnM9Zzk;IXjn36gC`s-x*gEM<X78R|n;F^YVk#1KF{EL!zM{RTLFneQF8qkN%Hf`O_(`oG=t0c${$TGyLC(?qg}x{bmWx6PBm(!*JrBm+x>R2FZfm^UmQ2tU?P*QRX(iX_l?p6;?QT(REY%tDr^0{w1)`ys>g0$~#)GS%jT?OO#jf5)~R&Y5mWoyJJ<2A%e&w&yuK`5{owWL&2MP$ljl=Y|HFu%q}YZ8P7#uu*T=JcNfv<J9c#Bnep#rp%aWu#!vHT$i{T(MUH!BN@kQaV@fxaeayI5Kx&oe>5xjJx{O#;kPG|>&{s9KIn<2!x#lfI>m4{qAEz0M{btRL?}x!{!6cLPPS3~+IFlfd`PmoX~b5$A1HBeGszt7cKK*f(IMOspi`U)qo~O8Krdgw4V|u-Ls}u{kXR5e-8A2SL%T!iA{B~3U9K$k(;C3!P5Gic;f1IPyFc<**G{ZkZh7VPdjc6sRm8u*a2$kY;GAO90p$G5w$wZmQYqGXpCD|@uQ^vFr9-x@eWui#EBQDbv&B*h&O3X;gJ|QyAkQ%4bR;L2wnub_^O_0&6r6kb3v}<`(}kW0aRZ_pBw;DsYKFu^Ek>7FZ6k503!S8(q~O1p>Ky_PICa1y%s7w`eABqh)edG3w#>H?SE?Hh=Ha~~_Pbzv2}pK>ihfZ>?v(k#4w?!#dRf$&2Hc8S3gI*VyK60fxh9!zK<S13^Vc+o7q!9Mt1?!QsOxvQPDf@=IYYoebpMsW*j?*XPM5u(FS03eybbEDILgJSm->J_s11Pvxnf-D;aKqUmbwFPt@XSq1u<k?LZ?srsxfkDONq~;8I=U1H`HPqs8My6q(&I+mn_R>Jv-R&$&v+M;*F{6$tuO2gL}djd_?gGD<0$dFWj1_fvfRzi=`l}ZD#UexeI;`FA}+<1y#X6-@y>LIMWg~T*RXg_!p~K7v7s}Uy}jGrxs7d-{(@&>6IoDC4Pw*r0BE>YOA4ebJ;82vsdEbMm%uiSUKEiNunRAITn7txlDDynfyG4#!3#A#kJs?k{Ep#D_-|usb_8`|Mz^a>4-*f)?JSZZ3bOVNmOBWKvJqXMhzL<^El6WWg)awkfJC!ty;Qa2?hG~yUSabC?82`);kO7KZlHu2cg4SGxcYSd-DY!<VDEa9L$4a)CdEO#9$kPa@UAhH0Iw6C<lhMz9t-IHb{?$rkV|RCTcW<N4Kr*TSFcVU9BSh1=eEf=nVscWhomgjSw`LQXbR952*X{zt+1?AE5q30^LE$-1S$uO0tw~0OnM57*yl-6qZfI=m=T2mG!YlvLOpWTBz6y^5~_4fF%@+FYx(NwJ;sXBf4Un3Y13*12Tbw(N?n+rB#xw)+;uCX;Q}WjKLqnyztt{AZ?N)dmep|6f>t=J2x2Dd16H1+X|mUw;P<3Xnr{GCl=Y6(sYUb7SFGI2DG7~BHiZd+o_65uC(o;FE7xUh#3oiKSkP<SchJgb{@pyiSd4UF0lKDJMEq`SyN?H*XK2;N-?<^*?~ypi?_A*>eF7_ap#6oGf=%?l?})OD!!Q;<*8GOLkRz4M2G;$%z+iV3~8@E?Kb;f0F>hYD9$I&ie-U&QZH#ON<^6EP2DT`8jZ}t3m5L3zV>*5?v9_}tt|eIGdZ1m+D=q#>P5Lr7iQEdsbpsaU_oT4_nWNiQH0n)UI&~g?#gpae7EJEQc<CXUj!DQJx;2=c=*B|w;|LLx+c%P!-*d{%Jbpl{9?+N@e4Wl2h~}18iRI!BbX4Wdk2*y7DcxrI=#_TA+jGyUVu77%ozxPzLQBGyZ9E2sZH>cUW6labiS6e+}}G9M{ue`Tx(vPcS64aLufO91I1>2!-^RM`@fD#l-%(Ld&QpeJQzB-8bjp|R_#sP*>@&MA*6up(^{dH^k2NU0#5lwgT-vHfh@q*foW`d_YD#3aYhtWnzgqYdAe7AV6fT_L3c!kh=qgO<s8!PzwK?{+m3pZu*h|SUDaqjFNF1)j+?a%@p5>7Sv@wZuz*r>@c|3UEPKKJ;c<oVeglR9*+Cyh(?Igd6PbJlFOmt-xzw%fXsv$o%>`UU36HhyQ(m6~+zSeh@Iv`B#Qa#R^nG@ek?E`Oa@aG*7TZ^Q)lQa1luf@I#PGn)rQ0WmGi+sIP;SK%Tm+I@+SI&tEDb%(XMzWM_@YSLXY3aCFh->WUl4m#si!_i)`ih|o(SEduR9Ap^`HkKPDRi}KweGOvbSN6$mgOOmg~D@Q%8r#V!cMinv`F9rL{X((V$_di>Y6>^cpb1Pq5qDF*E~C44HpG;vxN-+$T+ejE{@C!dXMC6R7E-YI-K^-WAR|ug_ZLGIJp4w1~fq&|D3rin;+ai{kC{GdmX#xk$1@_U1<F(677345{`R3l1j2L0;BL>y=Nm(M72}hZi7nm|P-zlzqqlnE}zhh(SM)66Qj0N-@Ay=suRN6a*7SqjL_geyi{|d9wM89m4_csLXK*yd7vXz2D8asNGE*^+~}eOdXGEcTb3W+~aP5WOC?!<|PP(xy9wGBbuR1Gmm#fvw+;tJtE_iO~Ix9r3J6pel@{rFoOI*wgMCMVH9-c`;2ly-_#?vg5L)6f(Pefc3`r%&^2x^tXP1leVJAXy~V}Id29_(aa=e#Q^o7z@gj{~Bw>wk*At;X#qdp`iv7shp*K<V7NoYSc0r?Smo!hisa?g(HHL)ej0liM`e<^ilwQGHx3`a0uk7s-%kcwNmdxMoiZMI+9ICd7v!apjINCJ({nBNIh~Jm2KFt0`yq+Emuav;vP>X1U5G~Iuos1RwUva~zlZbd-US<@^+a6{&jXSGun;IYeEr>L?=l*2l07(lX#-`D|YRWOgsW4^k+LAjMJiay8XWRnnZ@?Kiv3SerJG_PrsGMY<=GrjuIW^RMjh;FbOIk>OT2slzA)g>QC~!LmLel%-sXVEPCzM^EtJctisMkZWrNv_>_7j<9mz@GldzbH{*9RoEaxWCJR#2+mZ%cXo%;BTF=+7wle~IKgiP%S}XUQ2We!RWzr>A+~wmZhuHD6}VM&Z)G-SNT8BT>Nfop5<jn682q5Pcd`xY-nUF-UX6yF0Q&V4syXnK+`)CI8m_oV7t=M@I6=b$ZSq5=B2F2(@kJDroJIN6mdH7=$XOG&!bNb0)I)Hk<cMXs^A=hd#0fg2auzl<L2<kBH?aRIoq&n4C|$ycGjJ)%8qbC)ePd4GC>#YssBlDnVFvg=LHJZiFqx&MM_63BHAyHx$w;k!tt85L>%??K$8nrk9?qDQuQ@Pl2_9w0CarOPeCh&Kv^uz@{%~JByMz07}A4;?O6@NUNSP5rp+t2p0_Z#Q~282C5CNO$fAWyxm2BjIMOlFBkr+8VpYmlc;@WFR{A^9~?{#r07;GwHs&p#2wLKhqul0#MZ3vZZ_8EVKPCEKa^<EterHaW(Wcj|MtH?R@*{Mw7-+db*?L!&gSB4ON4Mzt(U2~cX<%3i2&z4;JVX;_|~Mpd6Lr_db&J?gy$Nkg0dnAcC37v+%zbZ!D=kHj2XyhC{7$N4{M_-J+3HPL+^bAewYCeBh4=L`9_j*OjC>-*Sff@ldY{cdE%p_Xv=_2)BtxyPbg8EV%`TqV-AsQ6V)NmW)+6tIm}@n?&21v3<E4{8wbqyZ9F(Fx~~PfEw&WDT;oZRxs(!GeL5ER;sWe-8oD6KB0wKSwN)|$JwD0-iK2g`bbV9!m1M{(p{oSP=ka}4OfI?SIGOwRo@u8rmGxuvy2fKCKaEaElTB4J)BC`-&3+mH1?Z)wUq`~FJybY>JkrON$@DVWHPdqA7fJ-ai%F5N82no#$0k`IfXU4%wSBh&d7}KP#+k&YOBubrG3IiR6kXtDy*JU0pAx8QFs!`yVEgi!Or*t1afl3}Sv1g>vF%a|{v@wQP)aUtRM1{l9)lA0hp|ZXt=toWP#r$wR-Yqk)s8tpUjY}}$gWiXQy0e&ZsD$Oob*T6>+`6<c0;LeS+(3QV&aq@@3a3Kwo2_+M;&TQd0!;eR>@KZYb+9IV7Pj#TquTe-YUfs?e==UupK9Yc)^`j=|9HeS6>s+!UQ_JpJLTkHFK0e?iY^AJ6T3Gt(W+sYMDQBEdG9E9-x$FKb8zyTD;cY)|0S6`PD|0X19!$TQJkDL5y-W4?N2}=!X$L?`=E@rQ|I&^f-dMgTk$(WvWz1;pC)y;HVke5cuE@W3Q(m6dC)AE<0OmFp9d(OY@!rGC8iK>v#cbOM-2~kPfGnXmY#?0XI(^EWBY|()^HX-=-GPove7Nt(Dm(%OG}9$I@*K2Qz9tXx=w1_#L`eOz@b!SpmeR*fmKM7GuQPlRt^F+-DtzZII2c8=Wp0kihGbZ~n_wFSo0Ya@tNJc`t`sF_nm7zV9$BjMkiZ{iR;U=(Wz!Bc0J+60)c?@hdwK)W6@}DXGDh@9ee{IctrQ>%1L9`53>`axV4-M&B~1D4(PGnv~gbRKK1hCdr4OzKVvuHb|E2JXp;T$*V1*Q;fP(xDRsvf$)nE<AhH^TJ}wHaNKv_J1d72c`94}dA#)?;o%mMW*uE}&*2LfT3cI;kfz8?4N({#(4F|-a#&G#JoX1;s~jrphS4_rH1+%c!-iA=3qEJffkOK#!C+s`koU3@z*2-OU8bA0%Ri@!8_^ANX%^yV>CK%&3^{by(m{oMkNywnoE>^pEcBusUPpC}_jpueZ05;Fj1D(IUq=pCsZ290yP8<4YpylVKfV@ml_zZ^54We>Q#~h5i54oU<x(>wR|5H3_S|b|(6FQXZq$OHA$2OE&g8JZIO(a43TRC7;@g_m^!VB);o;C@iJk}_pZlO>HwF-7<K*Hlc{Jdjh_A_W9tBBe>4D|_K{3+TMo<I&neaI%PEeEXF1ff!xi~qZmmCFa4*FNdV34&<c?e3PUIJxB=DFr=UEzZn9vD(9O`R45%1p|5gFO%(!ug9)g)yaYV@@py7<utzl)Ogg8(Y9%L!2%_&j}xYORDS#(_EAPXK$U|00F9KzpM`z>rW9tB;eIRt%pYL14CZ%%gk}Mtb1Y$+T|9^*A!v+$Y|Ia*o>kFxowehjWW2Qa0=(rwXPif)E4eiPD1voy=)u$)4Q~l9?5nq`rfjD^4et80|6zDcZrF{fqFX9FFyKu%8iyWn{UwpLG}3`9Z@T8zlqC%DT2CJKqwpu4<)1#+G1~p;fC38u>v{DnnWkLKvVlbaHy;+U40tMWU+U<m&({}V1FBKXE0AToggQIbwUI+z;m|u`&?qnk5i)q8Y=~gB%>V6v6P8xI8OmEtDLg{Qr*iz>f%o9sGc53>x10rJ$t=r*`k}f+tt>^IjkASj!-w;E}nWM3f8&>d1}RNX}*nMzl1yH2x7oj>n3No7Xs_DWw2tm@m57LfEU>O+x`$fuiFxBZxKk2)gTZ%1T)eCe<*TeO=t7oH$nYXu5;2V8{P7kOVTZ%u4yn^Pcz4G)R>ZIP+zfnU=*1g4|uWMi?h~*xKO31g@F5u0P%b66D}81X~UAsd&kK?SVq_`o^@n(NBk1>JjOHa@z2}>$6IqLtbUjQ$~iE)kpWL-YE|oN4x~M9r;HVfm9|P^rQz+ETsPg6P~GH%uU|E(e+j7O#VO9Aa|tvfrCE}~H6C1&B*BOGzX07-;4{r7hGUD)T}c%&+BqFAq0gDrv<8NL4X5qbnkcS`s&%7#VoCUVL6x-i>!-eZpFk_iQCxEyJl4JQv?)PErP?zgGAlsD5JSRjR~v~+7>{hh`JhM_Fx@fmVgLTM@|XCMSnUQgHlZ3KBVRI9L4iH<-zakOKT`{{`MEjC-y>(uS7_D+$Lvgn6O)xP>jaN0Ij4Ba0<B9VC%jN;@g**WZ3g+gRsdKf4)*HDV8vsRq|U9#_3*K-aC{msSaU52frPS&GiYv|uWAPWf5<L&o-Br;BM`DEvCz26Mr=y+YjFcDm<)xu-JJ>L#l9rCE@36H>7Pz5s449xn62yqP0`P;DtFiG9laC^ONUV<ls!^?zxWs~BIJyQ1!UlGTE^Ml*XG>*gLdB-lw6?USLohVu_Lto1_sF;R%lwe@2cV09ef|oepmENd{PG>mZQ?El%9(x?>mZI((q6bp6^~#>TQmJ*mpaYrk8p|HW&ZQ5|F)9;J1I2_5X~^>IhK0M~$mG8)~4T&F-;4dTUT{Z1$*|i^@2Ao)|u%{5uMpQ9sr(0Ozc*j4?B<5sfCDg(sNzL6;(^fsAEXz8rx++wxYILLe9+yOL3Uh-P1oSH}q80*!Hs+7?|4yd@bD4FSC33|Gu316*cf4!Mq&+R|lB>QJ8rwOp4SL&eQEa-W<#oimp0qNr4-&y6^Gc)lgo{k2&FAOG-2Mw2V$ib-`mehe*HU{AthS+grf#$p|--s}Mh+uzPiU<kzO@ASgBIhh`8qj|bkLRfqB-5{evqoL)FZ_+6}GR6g%SanqqQR&Oaew)!8yQoQH`iKicP;+t@drc}`YIudk1}gQZBX>|Fj6U-*O@w<@zdF4Ub1)~ipdnf!pt#&|Mq^)NTdA^X$P8YyFh(06RjLOAin@VM=z_t@8{hxJwj29A5~H<V{uP^ZOc3Q}6op6%Sr)6xhZBMAK>Hp0-(*h7ZT6l#e{XpF6nSwGKt0qw7O9i)7?ESgd`k8sbQbo-#`(lB?Q9f5%%OiTzpuzIRJJiPDpMDZ;E9c@w4#p%KoweG%^U_$VLhu9X|+yPoXe<LK%EU_|8^*y@<^!JMdtrZmJ&(KTB^@fYUi6HL3HTb8!Ij)o2K}kul?Xkpx`2hLNrqjuEIMfr?An4IrfT6v749V8_sZQDtgFDrL-DkHYhg7aW|*iYsPDbKhSU8^T8DdJFR=9GUXAfa)!TBaa1CddqorhZ(WvWb3axkbYm1Y;Id8HJ_>YNelA?L)v=6c7`mx_FAgV{wI`2;q<cUQN`Isb3l$3M2Q%NpL=mH9U`JrHS<u3UwFx?pCU|PNPQ1#BT45U5hqL>!Onlc4o?D(K`I_fv?D9gzzBc$^XS_A;=@<s?>PV0zG6v8HU=0wm3+$ZBB3@$B8$yXrB5Z_P;K*;by}ZTZ=1bf781(L;Uwn)4YE$Mb^=w?f@N4F!P1`C97f6FWq1&nJc#2Qk<B$VXmg&N@moq5|DGPfqT4MJP!CG!2j{2o_^ESjV*;Q!$JifgX1;>j?E6;e)&<x!^GAOx)pIQ%HkYb^1pyWK6RWOmWfIu_vuC%XopS;~Lor6O|Q3|E;Vrd{{7*6mt4ru5lgYv;9taqp-&y>b>8?mA*(Bd5ik9YngJ<W2fA#zG}F&DVe$xJeQ3n?soU2P<x{d@g!riVn2HlC1R_rhl@2K(l0sO$aI)ao_8mm8V@WV*2bvfjRRQdS?WomG9CyK24KXRB0DzgDXXXpuYp>XxUt0F~zFcPmX*FE!BM$8Cs-hOZ2N>VCjr#R`We6AcG!Eppp8nA=a3ar<tk4ujD(CO%X15B0B!q9mbK#uo<ENWY7u=yZGeh7i}}hc3(CsFIrvpxb3*{S_@swB%qEZi8#y-VrC{c`mL(^di`<FZyMqMtWm9x~D0vi@(M_xNF`0>alRs;JyhtPb8o+DTQun?>y-)FR$NG53HPh20l@(d8QvCcW8)a4a~z}Y0@2%tx^g%Vu?I?K&!{>(NqmOj5{4!Dd>H-`AVCbmmgpm#R2M^<O@71*Q{*w@}A*TfQsu4(jK)PSsteZ5Jcbiz*tI*TjS>NBzWS2LA3QrwhR`!4=>CoOoF2}$@^yP)MyFKw9b5;Xd&nzJ#VIIf?twb@XhS1An5&Im+JwGyF)Dr<q(isL9SPebLa!vKl9}!l32tt#a4Qt-)IgVpILBjPf@l3F~DA-rPXvUC@-XWi1*{b#(51^GmopH5nJ*<%;A7$$_B~%SaH%>yxM-yrk@29-2;>@%z)f)F`NEGaa9UMKKqbCx<RgU`bs(&@@5)_ZRo&XDZA*}33mFY>$Z~lL&iW+tX8zyG-Zn}><@^qPL*Ci2U06w<Db7u-tYH#xZc|Hy1nz+58<%ed(F1VVpIaVIC0V+=msHkPd=fKY@4Gbb`+`1#N85!9$6zr5~pgYI(PzoQ3eI;aN<oYBQef29~Yie=a=C_cZK)rB(8Nlhbswhs+F)fAPE8q2~5f6?+mMnZS%LQo{ucNc#A+`QH)B%cRzSoySw)<{PP2IQM7g-?W19P=e6aimIynK$_%49r^^#47DW)a2ih9?Y1&=XMfRwmTXKVpT?HxiT1fLLVd4dmU8{&nVboZH0}-=T(%LIYFjXh#F!sQi#TeNBF*-PxYo~oU5SJy1D1=!`xR^%*5)7+>-^qiv)Y40fv*NYHJv!mvN5)9>Id7hp;V0;tccy&w!uK{$TBT1|2UaqiI;)l0+uaQQ_mW4a6#_jYrF}fxnW322%|DZ8K+7+GewZ|-KADz>1Z$X}35HfM1&2uX*4hgSZI}6cEcHV=nwwY|$blWKByP^=ysTd#JD)8F8vaE;@Yw1dspOso5jLURf~uJbrgs{){>0%Of4w;X%-+&#60XGb3x(+}f%n)urSGthI;I`<7(6nuQIWC1WaffFzT^5c^C(8jR6Q8RFxhA%mgKjHBCaEgZ_=bQV15gq`9?ZYT#mhtRh9AgEKMl8I+EEBz$|^{+fp%Xi=mgb4a>xhs}So4k{~%-zPt+hod*SR6dfQ1m(c>r#TE-p{!AbBbKsii$tf6J4Pxi&7vG4%l!%M-ule7MK{2-6ME^*L!C=p%%s<yLau{n&tl1Yq4rJ6BZgiAcQHLMrttdFuDzT1oaI}DHj@>Lrj9i9P;&;FWpDU>>CAeb#Qxc-}VpJRP)eM#cU1~&EWIo@>t1iM)A?i+>>tMqHtHwx27xr25T)6yGL?{DH!S%{385t1&F}F?YZUUMA7b3Yx4}*-7GE1;u)x{RX`l-P|)=B{!To%W13YW`MU%hZJ746lS$OS?iBoZ7U-qv|mwwAuGOKYg@&*S>hl2Y%Mg741}k!HP1K_N^Pw`c{2oIA?P?C6mX@xyTRa?{eo(kMCr=thO@(V9Yz|NFj@mzgY2Hz{=#t`j0{L3c8b$@v6>TukwwsR}S+Scu2xXT{Z_tZJKLf0_)0;S_lKr69|#I23J~KQ{XacJfFO*Lz(8{=2i~J@mBXyt1+6&!V13ox9hfcTq_MC&AFDGey!M>Ac1T8p0mHW`=T)PrZKj4F|0qO6^dy7&|83)6?gIDcMT9eP4Dcf#M8XW8~;x15q^NkesK>wyeJ(x=+IExn844-bl||_pOEey@4h)%Usc^nh04uX9xmMs;wo~hZjBRUm+Lp$c*4>QeD1741p`Sl<4w4;8U6sdyQS(=w{Z-@V);`oZ`D}44W4l8rs(#5AIGxW_!1)NH7YQf}qo;>vWoJdMM6ElHT=Vxwn?Mes$UcC*mR5XO9V!kfOBh1j_snxIr-3I;5@v4w*LN#ffHR0De|9s?mX^$<52jJF#+hGHT$@<%fZ;cpRS7dYR%`BtrmGBf+$UjExEQ&j*V)X$|au0`W!TqtEs2H@DehTt-QqOjW6J9{^!KU6ei;gc)4U#sC#7DQ5(50ni-Ra6Bl5nzTUpGJ!Wzrq=CL;vMf0p##(mu84|-(^wnj3(rxA>#k;C&~z#6<sL&{{y`X#%e9p}ljIqx%MGSlu<J!$r+xksBGuK0;w!5$%TIUH?C5Kqugx6E_wY|J>m@+Gu!y<t3yR%X_<>vOPr#I%ZpWGGtiruE<*Z=W<b}78wUAdtiuc)YFpvSikCX}UtG%|mSV!ccg%ebiW!6n4E~+y3Cpi_jcP^;-{O>=&6JT#vkq1$2FKmt^;)mVu0x6f01*KTWI1f=9zfUN*=yR7U1oZq2njgb{v;mZiE_O_A0r3?(98ZsQa|*}}F<~zgF*6Af#^_CxaWH}mcyM%O+lebNQM(!`r)Q7UG+w;E?4=l$Cn-lhK*SMn0L{&xQ*(FU8Pb|o010yqJc5KNXsNdFoXwwssIsdH5kEw_#!{jAoZ(NPE-VWDwtvmeSN?LC<0SV)j=EFypbZRAiS@#iuk2POo$f`Y<3%QmEpcp_x#`#}SyCsIS`UX(eW`JTX<QE*vaSrP<HW?Zuw#F|e^9bj%D8uOXw@82Gtb|!5)!(GtXd=iQOkbf0sAbQ70%rJoQ}-mhH6vPtuuhNICx?4$jV#)KJceXwlAh@&rxf(7j#am`%l~v>GV9w00c!&^_>F#AKU1ep04uRyMtXQ=!E=}1GwGVf1X3-uVw!>xP4c8Eu~IB>rXu|x{{RIZi0O%nX&&Z(?K!S$xeUQl#e<tf;js_szPu%c4djbG<N<yg-$#kqHVoE03G8CR&Sj~1y&+I``fY3*PMs2F4SrqK_?4sy-lyprc4OAlpp=CZsY{jM-A#bNGfY+`-?m*S5G80-p^aX_~X9@MoR1h$<mkng!&7sE2ee%2S82*Z1}>j!lgTz-uV|c(#W=wJhL~=gZW0nsuNzO_q2%~0R@hV%c+YYuRfSMXLJ5-9QG37ig!qI5hC8oa{Q4iw%26lFv9v5Q7|yXY~6C~zWNu?m#SdeR&pIhUtl{49iH)qO>ISk#c_R#HTjHNfha!1P<1!eYPTvZfePlg3kI?JkT&o@F^B|QaH`T8f+A*Pe4kj{kTcCB>UgW$sConjxw)UsPUWFtp$aNkH8>M9Xey5%gA8-&naVlRtFHU1M2)cZYatN+qIrnlcksn;JEJso`M3?Hu}T>i<JjDu44ZS5L<AX@Bz8BJw1gCkOw}39e}=b#->nByVawM(UY)+$AFS(8<c`+g3(v+a0Ogby%DVWaT1Fp&P_cX?26QYgZfYQ7|4v4|4S|`MU`?*$=fo*&f*!L{tDP;?C*+k8@mb)1g4HndyM*aO-|Ul=vnRK$Hl%&_8F{}T@xwC_@~m0CrQh_rnutsf`%W0k;aN&nFKiOP>qX`^Vx*hL|9L+HRV{D8EY&BsBPFfVRIF~gT=~MtPLDlSMQr1iT9;V3;uHMh>-^^wsE1{63c&+Sit`@%``jcHU(z}hJ6r&huj!+iqa<E&^i+-FH}B2T_UQ!10PVpHGX*_jF<pKNX^ST>o>jb>Xr~0Z$mxw>W9Bgp`l6o-@7uwaldb#G;4(oaU`A(PosZ4pZ3KC1b4(Rhw`Bjt?ZJa1g#Y9#vx)U}2qwGyFt`24)vz6w-iBsc?X@22>)5v7Wyq<`77EbL^`s>ysHR<8EKT{SphCS;UEwTMB$1qE<EH-KIZ}h03)}qW#<~+$6(HM%$?yv~#K`=MY~NE9Ow5C9soDHzZMWt;sg+B5LYF9Sgi@go5ZJuGGK@-vLOaRp7;2Zz2O{oGo&UF6Sq|S?ow4uif;o0kU~HlQOdTv$1hZ5DH;n7|KTAU+f_{Tyo<i#o^uEU3o6MBnvN~4qLkUQjVq*;EVF?{Sg&xQG)^ym(A+ayIW#!a<QBO_RmdPr=LwX6C3;YDCsb)0ENx?TTDuaao76vx(hflt>(~LunWN5|`)*@_6j$m``1mb`i?xr>MExxb(HY%ooz0Fr)XP_E9Y+3Nk$Ez;)S*p`qJp;DI)_C#Kh6Lc_Z?KLySrJxL+>zM&;zl6^HAvnnYK|hmV`y{_uS6mwgXC46)~9v2$qf|B&;$nzIL7Qk*zf06B9nq(?)`m7$PSj~f>-rS@2_ZtREg`0?o2rM#Jej;fNfHy**eGrzOiVH8d8UVNjn{dw~kdOYxQ)uX`_uqZJETBda?*&N!Bfcb<JjKX<&wRtLOUzPVtsH<@(}Mi0h~fhf}t-%vgF^Cmvmqk2~7OYr=kWzA&1jxCd%f#yor;Kn+K!65xw2PYJL`g~UsKb@Q>sUnG33ZRb*2;G3U>w5diO6mI+iZ`z8A*gc5B9zw@JIDB>7WJ8amdFc&l-)mic3d#HwvKR2^oU7mi59K&HeraE)-b0&avi?3Hjg2|>`c7wTl13ns(nh+9whp1)QgD2k+y6Y%6ab(1<9^DP&x5%OP@3D{3Esz0LfGJ2m;JmGo99WM_On7AEuf`rS_G!bmxx%*aaEQgpLcODAM)EWY-n7j0d=+WomVO(8L}FKR%VU^doPa#Y_5TzPq>qxkstCxd~8;g?J(gBtuLiK^R9IKvSrYX$O1k5Qw!WAdNi%_%YM<Wd0IYk2I&%H=q4C2O~uc8sKcWJ#+vN9(gCWHR3};VVPZMMfpVBG{W~B}9?$F-U)(9WcGJIaS`~`wQBODm1VE8KUOyYKgj^a&op)CfUGt(^B!Gz1^&zXAf!5P<tKkk~d_uadcJqwT2P{CjWD!Ga_ywd)38$;}z>-<T9+?q=Ie7rtHI<@u<AgJ1AQMlfi(v6D1`lFkw1q#>ECFljFK4IUW*S}kI*)Q|sl+jKDsF5ux0(0Kclh=NtqbGJ+Otc$V1q3s3nYsNkTAtV%E7fauzPef%Z0(jeMRy1^x`Q0tVPDGYkwk>8NzIP+&=L!{OzjAk}hyg;#NZqcl|KUXpn2oW0OqYCAuLcS1Nd(UL}W{r3@}Yz{e!7p5vo~)HbS%9rAK)hOouX!As1NCTBx$f(&%b#5g!x)B@Ej|96}`t=})&#^Cl$aLWH@TRRfI81rFpA0c8;$Og5!pjcwh;Qwfx08Y!$a8a8iRve|$^VwFBm9e@u3rX3)(qpLt$4RESwK*)z-)E&>MdE#7zf0p7)${XH4bBv^JC)!T<>|pvJ2>;xy>uJ8_4ZN4ZWTznxzm?4N10Qy_-A@!=2=}u=_VG4c;LuEylWA21Bd8OEVdu59VhU;#G(-!S%N=cmjUxd&WhG`>BO?8%296^?Gjiw6PdG}!n0lSQRRJ9>uvSBIjF)xT%rkq5<*W_Yi%uIPC8T;$}DZ9pLNo|DlmIA7mrxA>S(Vrt3Vmqusm)X+l>`GEYZ_fp}(+w0xH3M$>H=_&r7F5@QK#^F(r$QaeQ=BEZ%Ak5rE=zvai)ORSYvRSZt9!bo|6a#|n?gE(O{2UgKcH*16hms_MgtlMNR!na2R0=R5ihHNKh!2i=^U>;S87`~OAUvxvJg38BbbF<fU)Ypxb!t#dgL9M3XGSs<CS|E&@-RGCQ;k;ONah`zs}%%1MjJ5l8fNynmmzs`*V&KC?kjR9WTN-*wIt}4N)*$4B;fC<`rB(I_BIJUd#*<HE`#Zo6K=6~$ATsoM^*mX*0Z>E@DWxTI>q7c8ZoYHR4k8ADS4_1fk>C_f~g~^wQA@anU-0bnPIsEb`mGN8)vo%)dRpTzux{dK6n?*|??v7V8qKpp$ilAGI5dN!lV0s72)V9}LIheZ7Z$%?lyWeBolZMJsBa?FL^y1I8gR;Ig+pdw{K;W^5i^5E?CFAeijF^#H_ar(S!+A6Z1amS3)xZ7aR7j*hK^Da?L&#6pqDzdteD$t$NP3}7t@z|HU+p-`sxi_2tRlrts&|o?^(S8Ka9Q5f`$5|K=U&+$m<!pj#Yj0SHwfe}ibFL4Bv4#E2H%WY-weN-cOj96v3=0dj}#4G&B(T`VfO<Rg4+v!dza4USC{#>I~o<KHj;-yf;%HAMzd2RC7TimRlP-0V!ml;%!2djC){QbR_KG-$#Y{nxuv|_n!KdM@Qy@qb4P5HqGF>k!EL@KsQo-#keLT%L*1FuFVz$#i^x623;X9nxWs-G)=@YFhfau!y?@W`s-k<;(^>R&v~Gw!5Z!Cx&u-`}5don6Y{Dd*^Y89hJZ2s$kMKm$itJJTwJ;*)n%G1~wuI(d_(p|;N0;Uc3R?(L05+Zi#~Voz5F}8M9ln;sK@~_`KbJxFH7uu;IroJRbiN`k@t=E7B${q!<fzkavnL+d7%2izy&1dGX^vbb1S$(6mJidta!=OZQ6G_P>Tb!k-7}9AqWd!^-_XTNFE;{$YIQP~JQWfTt^(FOuJIA!#LmAzatgT`=`t$tsNGXQ`t$~o^d1lqfhMZ~1eLy;E3^F}_h;uIpzOJP=8}dnY=G$4JQ<$fVG3e5q!+!C@3b4{@3MVzW*TFxEVWUA6PW;dPHv>(SrT30e8ZuL=!>ENg;k5d8tbJ>J^TcJxJL*k`O=GjBBpq$_x>2YAO9A+2e|@4CXwFQ-y$~l!GYA?om1cD#^6TXO;Cbr=Y#u)T|wIP|7W1Hy_4A(d&)%Q_X(n-JKdZa^ZEuwu*ui)*ltbB3>{F$Pbw}mbOqk(u|`kCapq^KxH>mkqszd7bydt&jlR72^W~K7J0!0D0lkn8age12o-8SxkFvPxyI4oX0jy_|rOJU|`%iK|--9{Sx~qnyYOhnIdVH?LlFQ;P#?9}EV4y+O#O{M+(vI{wi}HAq9}ShGIKNs{@(+*&CZcL??}0f-V3l*<It@d+cAf({p}za?V-T&SHVph@Z8Wg<ckBva5oX2pe~)+aLB(0=--Vmf{~a4p`5u`S{B_ujPCI!g{`vr609x5?9*?dL7O;gz&J$xB`6@?p27@1`IciW`U2wK-cYD;kUCDgW)R4t#!g@@|8|h;lh*ggXgAVx{iLV+#;NZ3;H8e;!P36vA2^q4jP&0Nryz=5;rU&UFU~r7v0|jo#MS(yrZps)?r!WSvYFnI2Wd^)-`RU9>*1$BYLmu}0kIvJyt1A?io9|H1XVRgD@PZIFihsZgT~{Bs$B*cwDs~#BfUOb@sQ?vd40pUmx{@m1%}HmT+GV@PmVAp9r1Id;$AUYfv=dlmgM2F(c8NNC=cKj2C!UdGj()y>t6rjfJmF^+W(?Gp^1RPLj3T`oFj<S?YJswL_SpuVeEKGu1F7bKXu(h{ZQt6UH~Zi7NDs+7!r`=HWPcw*u3;r-^M+)&Loo>bj-vEBtv87;Lbu~5Vqm<*9j!#?*Y^-(0m*1$!*(eJ3{kVuiBkK<XL?2ieURf6@RdmWuBy29OqUy5@;evAZ7Hu*V&veXKvnURE*17yun(oaC26rcw5+1ISol@sdBu!|p8U2c|2m|O^InwpT+v&X!4srJ7xQK{(5M`#r6-RRDn0v?&dQ;5Qie*)N>Wi|IoVYwWi#BOw2^3WdV~Ol{9Q(G91PzvPr%*{G1ZU2K!4ix?4);prw|tQbE5oxh88gk46i6+Bm$_XZx3ax5Gix*J^io$rEbNprrK?%rFzfJC=#*fkR&c$*q3L1x@kYPMo<of72(>n(O;6NyBxB_r7hM+a2w02tZXSBY?mCzvkc+<`J#q(FgBtiXy@|BCpY^o3)`=x`5urQh%>~br5k3Bi7+|zpw0xm>cEPgxLDNteH1|i#DnAiOu_E*>m)%~-tMZ&3&vi`E@D)9UboAFS^pk7wN0d#%_h36yAt=?7xP-gRvFuhu!S5zq)cicJc_(&-8)Ux1ag(UGL096rasufHi40;ThacU0!|zBkPUaQ&GS7N-qg0Bu^;gF&GiaNx=^Ht0n;H%oE_UbCo^*UrRpGBr%cBiQB*Mb8q1bm@9yIZ@@reDNdwe?=b~#OD#4_C_>}Q+nJDstokGPH!`y=qkfRZHf_W$sQrzMo`VfiRbHSw7y@v%s0vu!MWuPUo!5jScq4gkqbMD{gde|EkhN<@K&e-DG>q^eJSR>}V^5!V8NW^TWMN>ZC<m}m*kEAqbg(!nhR%-9UxRlx9;;A^cAt#in%7IW`C_UlYw_|#h)MS1Ww+hUK8)oN4{0mo61IaKa_Trk?V@f+`*0c<>1&mkJEWn2RlGqwwV8~Cc_oUJ$Ind)*zIW)-fG3%xbi340S|v1@=|-3NJV_k*tCJ~QEqu8_{-w~U4<xy<3?}TIx=3dyFAdP?psJ6Wc;A*0x3^&=wn)$<3%ES6^0axPw4&$;DTBo6k#4<CY*=sj5dW)l3HQS@TV)n5w%}l!$$Z0CbmdZ)d6aI}G<RQTLIkOs9o|?7ICAC~XtoTFyd#$(h~xdLow4VZx9?fH>-S#wQR7=DKaZkJhg*8!NGA<2z*FTHo=0XmS_c4!q+}tZmLGqo1+v?5sN+pb+B)dRVLn<@>{JecgOVuW0wA{uxg2q<9{C9{Cl&7(ds0ApJxGpZe=Bdh%r}i3{Tl#~t0Z!ZpFSO+d$>N;QzajuwhEDT*$CSJ2{9f4FOuY9o3yxGEZ&<KD8qctuv4~oV|BsVHQW#pN;5<XdC=!(`zx7q`^HZap~7HcD-i<S3r|ZR+5cSB0?|kEscu$51g?U;Ff1V2%udy&_rFs?lpr!VG*2<Z{UAGDO|czZk7v7^KA5g+@ug(r%rC(OTD2jpi@+Hb*&S#4*<E&#p%p~~Y)-A>93LdmYbl%qG21BvT6=UoFT1ajlcU0T#Kj!J-X$GJU}E(P_#1zvbsu1mR|N|HoI2}XvsYFT#8%4cz@J;Ty6%XkEA*~#sY)Zi@<6|Z?pVJ%D`e+p_mQ4>HT?VBr#c%yDNv$+>2u-L^N{!6ew)!P5~G<;5@v6uyEJLzZf$otwD-7~*1AiqTv(XR#>dDS0O1rli#X$S&Z`YkJ8(G_&+%HX9Vou-z2edl<GP!Z!a4Ho?<})v-DT58@f@9S>Q#|C8-o6K7t_es7}ZQ<4dAA^l&V$XY8k~>|1_+*3DxG11Cjv_P{q((JuU}q45#F*sN<P2wDyvN6|9%DqC(S=rMcOh58I?B5i8Z8B*;Q)vYU^<)<~ty5bs~`L>%_Ux3{1KlNGfeG6x7USzRFve>R2to1dL;pTj)xM48^Bvot=_@z8So<sJ}MxH(*``$$Zk+=BgBi}BcojhG%+qkAx0Wu?pte8VxQS-xV-7lZRX@geG;aUQ9a+08xUMDxHEtb<!0?rO}BXfEL#4}8xf2~j8i7!r?#Yb}qF_ivd@<O$h~O67Bjob1=(7yQ~-#m{uVZX$ixKau;MAC_ZzBA+c8We|{QFD@8RF0iwD&76AVQa4z*dD*$SHcjREFawr9QtgnMszo5|20^G)6YgFqe6`4C&UFTxiErS7Pz_4nOo0uh{_|3<VV;>`V@z-y=6VNR%qq#!r?p}p9dHLrV4zI-JDeU&pouFC#v<h)z#oT~j2vWtB}D$x9+gIA;2NQyv3IMMWAdsP%hE2|>~GGDoVmVsOhmX%O!%&7%D7M*sv?NE>V9<M_5u8)4!79ll1LI)qD-%{P6~d$J<+7gBcL_w+V@g?=IijMlE)+GbBl@qrLT^iwoi8Pk9xI*n<e9Y<J0j+wpkiP^r?9C8XRR{E!fw<C4_I@$&A7o8GUn$R6E=gxrIKb@)y&uSsr((YFXO@$)>MGVOZgQJ$=x~4Uw{wL=!gFIMX#C7Dnk7^Zk21^6Egk?LP$A41uP##N;{hl@z0Xw;68g<eoO?!ve;#r{ewVnw1WPaJvrv3{DLEPU>I5PgWzT#|{S7>_1rc{tIf02W$Cw%yjB>6&0>0w{|X%I@4*bMMEcOt3`Ur61B>M!x=nz?_-F622=6N7X~p9cEGjDRq;FGXayICoo~fNl1dBE(q~Ml?;wP<8q5K45k*OO^)U+R=84<EXqT7pLsow)jy?hBePxMG{1^SLlLWat8b_-d^1ud@jn~g$*EiShqgNr%*TuPy_*o0Lt#lPIr7j#T^vJ*s*~UErgemoG0FLE1Tq#ITH3lK&ug9~wU#8+oVbjL6mwI<vh`A~jBDU*eeAbL%LQs^qC4|ups<>XiD0C1CGdE)44dJ*ppr+1}0TvvVBFYw^;BJJ+=(Cd*<bZ6?ytUe&rKzox)Usjfno$w#09vy9)9zUbdT-Mz;_t;=0bm>XGo_6iPZ|9lx{9Njh@~O?u46q=%JT|#UFYQv7t43z*=)IOMETdezle8|UP=EJ!nwRZF`VHkPl7fFm~#Trs4^2r`4y3+HwJLCM6m=9bGP5CBsox?62FS-(HO&e_Hs#f3o$Bai?3OGRL4q<Ms;C>%~hp0Ff|>F3_)my%!@wv%{1(NXW2vq4MS;`xicW-tl&@PL|$?S{+<=7WOd5bR7?6*?y`!)l4OtZxy{gfK(P<=Ln*WS3M+Jr`f(TBke){(AI$KED<x|2qeMWcUbRo$j6ozgTsDShuqs-!L>m_Yi{~DAA1CL1CiW_EJti{c-6QmO$JcR>7G(1^pC{(5H-B5eS+{&7f}}3q(NYrf35}zhN5g0`cc!-Z<oTC5ectobW>9T5jCl+FShre=x)^%Z9*2YiWy&JNV=7>^?w5i^&OFm`&C$l&?<dj=K8Oam5*(0v4?zN3Rniw3>jD7DYHcw9w%M(T9Lf0nEXyFA<B`?H%rAs=8FB+pAV2<vnk1pzl6IAQB${Y-_-AsH)Chg(qbdEZ?TExt2F`CZattr?e*H$@OSzh*RZLcf>Lc{}AIYpC>8XGd7WmX4On+(!d*JaFV4cqGg7`yWzoxI%k@z4w^R3hYLS60JYTBgoD$pBNb*a`sY8(HXh#$9XF(I5ZiQQw%&P#Yu6|Tj+7ubS-o}@hHtU#t3EN+G;NE>@-2N>xK;d-(*@zbo$VKR9b9y{543v->c9-Czq+<S^IP;%U)yYC6|u4V@Q)q^17x#$b1&<5}t=AL{0Kia;AsJ8-F7UHk5K0^Cc&eR`aX)IQ1f%~Rv>U`mMmoUDGO8uVl()Tq@0>X+5$2f^T)jZin){=;An8T07vsO;q@|Wgfj<uO2%VZ<%pIDg4t+;r&?!*jTMxKpxHEx>cN|QU0ekLd*ZOb+eNzz$@>CvxT(KoI;ubqKX5+v|>Jz~U$4;O1xiwxx<-rJt<56i0N{>QxXJgq$M9)lM1xRG&WXucvx=ek#w#HpZoje8oaPRACY-ZxuvwgZz6qF?^z82#@b`h57qKwRJQW=5MLTUsHZp>pTzq)!|b@_lTW==wMJd}N#)q>y{SM)#u)Jpl^)?#)OglkpjwLS75_Te4}ukjT>7@DUi(ky<)$%?<$Xr>47dbj$`O2*zLzdU6&<VH<^ZLtBH3U3B+X*GJBkz{KUqKoY(i+mTFZr$Tj+D(*YIj<Yk$bie$|)c2s3>9<C1HA(J05@<?1p%qJI(v_6Kn-Y@oOiheXfXJz3i)PY4aF-8Yo$43Q3dY4LuEt=CdG3u56N)gW!d=ioUnOC;5R*sVbjQvI0)-{CT=nUn+RahIun7gl>(Toz_BDhDBAL0hkJQd<?$d~kit$%ZwS<LtLsk!q+V5x4b{2cpdC}SX+=52O6xA5Rb2w4PxAe!wYoQ0~#N_nKRwThK)M^@bZ=)Nsmy|<q@NZfr3iD}75wgVx_!xiwf(x|7695jLcaiCu$wzVhd^O!NOplBEkZy}T1pOxC5Yu!wnoR22yyjxdkhzHG$BJ>-RtJ)OFxpZ$0)0~NpLd9_=%m+N18$u9tZQoK{ftv=0G*pvAo*eBt0#O(5iI1f;1jd<Rg3TyHDJH0o?g139f`$_aV%CQUfh?7a;os8h3>1(5e?28=$IFmsUu2+%dqJCuKYWi(2~QhWoDuCV_9_wVT3qM$>iO32*II&l~I~5BVYEq5ZIhzzMR?%O3SNO`K<&lI##o8Zf$if4c;{7vK%qaZ3BsKJLR!t<#ja_&<3+_t|B1|Ks?vt*A1SU!<LoG4B~TSdo`IAV)8b-zzXxRWtT?l?Z3}UmY`tl1hZHF`tb%4$<u?;^x!>MNygy~R5;=P0yaU4EDb~nBFh65(fAc6^&K(5u9$B<$emU7%bg^&dLsa)C!qjirjX*d9G5lynk}T;pojGelxoE9f_C5QRosU@qi&~oQV5ZfAIvDXXV#$AmoBar?J-x#0r(*ClB#f!>~fxFRQ(?A`+$p%gb^X7y#`T~rqJoi%Yd;gHTEvo!9u+w9S{3m^mrtTfuY!c3@ZL}ga_4MX9U0SW<RkvCy-L$>hTQs<5pl7lqum1;6)vMO^OE-es9$00t16gA|M%<DDUQWNbD=lv9M_eCs7Zw;3pLnc*l=r<v(}GOqU4vz8ZM>mNjYw@5@#+ZpAYj^+-8@ls>**6r>v5fD1%Qh+yO%khb|Ph>0)<ZeIG!Ng<;y&L8=4DU{)yLez2v2ObE6?rSG;;jw8-4V<x6;ane=Q}0Lf#fZEU+eF_R1mB&$%gN;Qyxv1_D#D%EW#R$K(=E~N|4oLY`z+;Pi7npgh(T;>e>hR8BFPMsHNoGtDct8sQryB6{#+xv%#m>Ow2!frX_G88PyJ)$ivc~eU9^@e3-z!A4&_91XfC2o{sh3vz(j3uQ>21;hD`PZXR*2{y!%I|PrEJ0DjJe=Vm4NkdOqv}p+5yT+kUHV=Z{<yBx<>4oSw6mzQ6rZFsM9;OQ2t|5!ceaqVHam-<)bf)M-YA0(Znd6uS8s6~qvg6F_vdkz*Qo+g+k6jMlzh6Z9%;P_HCOiBtJwNRNTLjAxe;KBR2QXziV-MnRv>YnFKHSVLH;`o$;znno4!nnA=!c-;`8ol&eLu=0x7z#?t_S_tiBt1$!8&ekg8dvT`GZDvKhytY-wbh*xfY}b;oVRl&-54L<j#Y2j=78yD4U4UWr>HNX6`^mYA`V}~RQ9A>6uRUa%fNo?YZ4BwmdYIuOhIM&U$SwvxZI0V=Io2@-bFfhPy-GqA_Rt4*uY!~*$GB#Hs`s6n9ZmZqb6On9E})gNd&qO$W^ul>=Fu4M(fO326OuY^hf;%_#)=2+xSrs(ysl~Lx=$GmN<qT-fTu(AE!-8pzRRl?1*;OxZv97p4$`o#7hy{|9W=c4olJ)PyR=dpQM+~e`}#W}P}xY`!FBtDl+mjHK3;?!CJln0jKpaihvwlXJYdILaqiKq=9S5bTN-${Up%JZQ0l0ED&)GH7q-n`^<*onEa%m3>@SH!qkk?*$Husj!49wy4n<sGGj~t3qqCSAK+`e~vjb&YOe@aMy!)l2>q%@4f32h3{0bb>utum%h2J=LltqIXad93$-KCzC1*WW^p!*t{$WBTDGogeA_SoMy>G$5SX$6pS%Jhz6(+z0luiJ&cUO^;NLuy!m@IpuVpPmo>2GU51T6Kpgqe&&7!Np)F^4dn4`L;7r*Ic3*SMAfQ<bCP%iQzIXGmbqiv9J(~Rd^MJ!{wO`MS;z-w;-S)hersTPq@Y3PYH~!;Y3d?(Plm)`WZ<1Ul^S=XymvI5dZ%(dXo>#FK4%%Qhf6<k1s9x`eI)(emsg4&(~KLLHD4;9#I+vlCgW|*W)*nZ@n_<o%Uk|R~B9QR~R-V4T7zivhEo6$nfe}4TmEqo$-;js1Rievi4_4XeRlli!;$o*MueP5+7Q>&N8QOqoG3ZlqX7Lh@T1LfH39lF<zf+j<bC@BHK;dT_XybJNGhXSg5?+CJ>fiA)aN*)(XS09f;`=Z%x6(&)i@#3_g|y_#b9zP3<W4zRXZsC7WkgAamVEMa6RG&R<)?avs=B8ADBz65R&c&lc?lga4VRq=qqJkgHgf_1gw-00Kx%G+E#<x9yhPX)4~})=USP0OZ>YWNqo1-~GC3c;=&aQVKwW>r`3*>yDdf;neHLChNF8>?uDXAg{7UhtCBuS4E)mL%EzYOKHm9<Ty$V>8#3bzV#gnyH>p0xE8DwzCmAcCX$RkWt@$fhdx;-)7eWKkV{9wYyfM4Q*j2B4iq0vfXg~NfdEX6?P}}-xnbUl#j_j0A*qRmMKHvAyIzQLU;oK5jC!?xaSrvxn>mO4lP|)wZENs1#RF&wcQYtnm*G#^Xpr-j?0yn=Q)jKrmxFN#1kI9pF06gV)hoBO0+Xp=?87E)9|~!OAJznr`TGzc>|LLq#`{d!CSn=<AmEJEAkWvnXFGDnXV=8e`8N)}Ad_tN$@xqF;`U+A#ySw(b2WPtV<J%s14eM&(?8{+>5a26Lg_sm%C79m?2vz$E|ptRr0R8Cb{y=xsEf$>=Us(!k}v)5T3ft|^|HCf=ZU|0BT9S+ITF!YH=2uPm)x|8=7AKCP$qqa-lMMC%Vi*{;e+>BU4S<-ENzMqY<g5d{V98FRgf&P{d><*aw4cFlu>9FL_CfBx3%slBqYY=u2{~gP%y^qGYZ%6!4&jdcoeM@aKr_faQwgum^vl0gHsOP?n4wB*qSK>)iYwNVVBGx^@WA#5hQ~?B1I-#(hb?g&O=kqwp;r}T_8$o^W6AE;ls+(Erd8j79*gt-(JuRKwIcuyp9}%knlmR1X0J&-=J^UH$D4JWWSEm;~vSMRdJ2{XK%7T_={|2FsdP_%(h8#@V8)%LqB9lNFvpx;@+8hfbcrP=>6eX&Ba<mS++P!;Kv~7aGkb_%-yk|_)*Ii{hqbXk#hnr+ms-J(u<IL&pdx|LTwDP8@-Jw<^s)}XxmqmbA>rp@TS|}V`Bz7EB;Cv@rpIf*0@eEy+qc-i&SQ6CTNUM_yUg0$g-|}{Gr_%B0`g`A+$d@0e4l!;H^DMV@*alOlRf>=U!B`)Cr66699s`&)v(}{L`lKY+4dIgexlgS#6Ts06(C0d-Y;OL2ZU-8PUm7dN*V=AOb(LT;}X%5I6axb58=y{$!LLyvK~TqfLsYO^BUMiRflFa>QiG+N+|_zHnc}n?>#ZE04H@m|<BAgH@mim!SV~HoQtq1gVo0u8;}XY+fIeB#s)zl~-JhWZY5h=-ubE35=sR;a1CB1Y&+Bc*i<n;o$i0jv&r^?ec5uO-Ft!b_Y^ON9Fwu&Y5f-MD5<?ZC!@1GT#a)1d4XTLa7e~E$6PZi*WHzPODa5Gx{awAq@?46LQe_on-<20W43iJO$PznwI914nFEsAvPNwEPE$<LXwm;47+(qxCc470_h`$<BHs2Geh*&43B0id=ie@2DLP=ARlV`mrH0$cP<1!cvnZZ{Y1LJ7phVdjOQ!A`pjm_0_ejgK@ZVqR&fXw!l_npZqqflBAP3fQF?vMVbMggETbxA1wa9h$MB)d$HZo;7ruga(8|dDY75|YM%J>ze~3`zqjJo#9F913b;8OtX-E|PaQ>92ZTip`HE9{8q8adHtdoa)-BQ<bzNe6kVnnXef{{6_20~v{f&Z3lxEMkNs({L#VazqBqnhw}{!1PAFId3(Y1XdBP2eygdfH_~1fW|gq%hib628JSAFyyo9uZGSJT+klFjPWIBZ5uQ5qF}>=sbjjbvKU;3{#GYS>oPWkvSxT)!&owI)_Vr2^x*hdQ6Qw;6WMzK^Q%2z3X*wAjOjyx1HBImxjkM4&!V5A-Ex!je7~!tc=TX6p3^Wi-K4>`Z+Lbd|Gh5AWWotv`fF&N2rUe^&Bagl|z1E!lM7L3DQuEGW3clwPWH#>h0vL6zj}&GP+0H{)uvg*LSfzQv*#JevG9@&WJ%Zt{dAruVTj45t3G+RKNNC=Z|2Dzerpv&RP3pqHTbEuri0z=aPdrElM*!rmG00WCMa^POZoQKN2D_r6@#Yak2`d+z_{*Zc#Iv3|JR(kcVy{Q2Jh&)m~4Qwv#_#nC2BAGF4iYZr8d-cG+Obc7mg7*DVn8V3W26>@sXTAwW)^D~~+SClea41xV^Z%k33@G}&e$2t|pj?Ru9O*;u?ti^JKKE<j9~i9lgh?FNtAclVs!b}0!7i@BUu9e=BgWCKc)K#(a>Zymvadp1^ZZCzHE+lKh)-#Rd6BSMpG+(gVdUnwG<OH(O4SbHw}NZR(5GJw|8gUjaAyIsyn$x7^EkWKW{tCd-5iLjZ8m+-L-Dw{p!_+Suh$B6m3vtB>w*rcN=3TsZ4xs<K7y}6c1KV2iJgdl-dZ`2MS`sV%BfKZ&yzn(w!`YDZ75+S*!Dq8xx7Z;B04O#neyL67HIoy=N4XVrUR-*%^B%=*e$9Bhy+%R>DNyHStb)!ZwBgSk;<lF+2mqh5bYF<;VDAoyl7!;-xak9ITZDP&pW@n}=O@`LauLnWi37$51>8_$4nP}z??#4#aLHnyw4oB=qwOc9TizBHd12b(W!O1bHUEzIE9@~XxO)>J=;U%GUmu4!F6lf-T`x(sM91`{1-nG0Pa~cv5Pbby#3^l#g%5Fg6wFnsV)QMA1;F6&K?4T*0zMlRnP|_3BvXsPay)?JSC{5Z%)EQK2pE<z3yvK$YVjrM7M*(XpJr!qK5K;H513E>2KqqOxJ*+;1q1gqHZAnDD>%RytMq=ptFYvznMqN8;@MhVTb_@$lvEgGi12Y==`SC<u*Ccerk=io{7usYkX+Ubz&)D-g*!cwUgmnMdhArtt2bOz|Pm3jlbHoC8WF=zJri&BGbh@In&M^ss$(#N@TFV5Dj0Fde1ioPzpMZcuu0?q*{}*;>Vo;oNw!}2*>BldlW&#&M+>Z7_0Bg38d_pwYw5KMltB&yD--iPyaz7C%CUQG!e^*|<fq<xKnDf~qeJUs$4urK~s0sePJ}V?Bt?2!{A0jxDtAfMV{T%C3K!f7H%c03QQ<A;xRHD|RvJUoDv1+O%s>!CkCuP6<PvM14(O;tl^UMBmFvTars?*&{&E(1UL%;~w6zDng5RH_U7Izl8gy`ebG^g|CYHP<p%=Y~<Rsk?Hh9&hcT-#?pUi6ZP(=vR6Liwxts!IJ~^L&%jyznNEkw((N<0`D>>C+?;CfVWEp~v+Fx0sI;gA?AAwS=cNVz%yv+R|iTZrAYVjo{`PCPub7e)VrWSCzI_IyEWg7tw{!25G6)o598(i0hY1NpC@W8o@vEM@iwJ1N9V|T}6j&<Dd0o>G&g<jNv+Cx#r?+q`{FIJ-M0;Go!{X*ckW*rDc}^iJt$Mi$}^76xyy_bswd5z~3m9&z3Va7mUx0Dd9Cm12?lL)(AuaO^<$y0#+FCgN=KL$CIAz<#V>bP|j*@vhn+t`?Ck_O4atgx=gYR#|rxOoL_K_Zq{SbnFI8@O(dU3AZom>Pb7-=bR0)|n8U`G^X-M$5-%(ihIG2-d?M)#^CA+0P}(Ol@(dO@3<mR}STnk?coi6`Rp+c^=9R&`g)62!Nf@z%;<M>~EZK3?W$qsW@cZ>6-3#J=2NBwRQ^P>%f9+bnnC}7s?wD+?^7@PkT6j1ZA9@9L4|^7H!}Y)<l-%Qka;CyxiMY7L<ctJW7B<U<lpukQ@6ft3zMg)(*bQ~_&S>(HVv}{nN=flb*b?t{M+!QRu(CzlFb07vl7U5Xy5M)`;LeC&nO9Xws^j9%4SfPM)mWL0T{`h=<DBc5*QY604O1fFfS&)J0YU>}9VV2ThVk*6%CTaL!)J4v<{$>_**1auuR^Np4Mhn(<=N)<=)`x!UBb`ohYyg|^N_Z0UDN)mTYzbbGUpEUR5bgK!z7o5d*GU*p8Vzzvtc*k=d(E@w+YZ}o<C$TWM0_S=Chs=_WNW1QNav?dRppGSlAMR8}g}l#i=~ug!PbUUumLJgBqW_6}wryw7p&IV=&s*yiY(fc@3%gth^ZVx`-Pvi*ZYyH52%L@iS)B(->G){w&WK?(**uDysCkj3|J@(#5IOY!iCL(m*gD?E(be0%zHYU3G>aOe3|_W6g5AY)sX}JG=<r{zZ){_~!6MKjrvb^MqgXO$BK73q|u_&=n-0V#>l&l@(pMKPX3Svt|0lf1Q3?Zpr&Q)44o(-wgP8J+e|}_yh&^4+Nqm_^f5y9)R8Flj-Ig-2qa2rZ$IMnL^Im(@!ONw78<^Kdu)JdD!qy*wETA^`lbE-mlBk43t__F>hW=`p*XiLK+^;PqEunqFhS7_mxixGN(h<&W|((W%0I}s8&uqTXM74GoDT0Pyx+6hfKb%J%$2FwjY(~675#~St9PWk(V&#BjL1_az^U2v*-*TrJV8!fYmK`CLTi6=}GYVuTu96RA`=hpfY3``0vL4o=&Q_(SaY<5LOwy6sbo&nc9Nfz|#5=CI6LnUY~0aGM7%Vvfezl_;e~nv8S?nUus>GkU@caPn03@?Mw<$2C_`CYD05Muw|_8aAr;L2ordl=QG=Wu$e2}qUt#S-c=?Q+Zh*6Pb@On>TX#SHoBcrnH369uObidb|p6=tUb=GcvDme4qbDBTg70-)M4GHS#kf!(MEl<cqtuw)T_-l^uLJ(*iMW0*lHf2g^sFtuI_v1)_>6fAmu-EMQA}<MvwDe?stHMDI9-HB5I;Dv`;vHU?$H%z}>@!!ebApH$<8PSSLQ1Qrao=5?0XwhuXLBCWMEHY9Wm$V7EfTT`PEL4o2;PAXa1gg|W1Gx--%>Jszi=b+(iY#%TYBj^N-aOV5S+t$7;gG%*rOGBlQ?NO~{h8Hc1?K*md3VHldAS{<3(M~k|O_;r#pdfQC79)R@tng6zm9HT>^TYP|Bpd~5w%lzbs!Bc-=6~fjEXlLo;dGn>%ucPC54~8d)$vSe&1#qC9@AP?xd2~SsL#1Ub=Cnrn)VP@JMmZ-*o0<@Sc5zK$`=rZ1Iu4NsP&Bw*9Nc2h11a6Z6R7G1cm99iGERlEK&F11Z~|)P1_jImd%-6%T)&+u<gb)r?W5>Iy{4uhf6zmIOn)CZD|8+TRn2i6CMd?yI-t<w1DZ4s2EcyN$V#>>TP_1N%t)W&0BlJi4nwZVz~IktVzn|e)bF@;y%-mfJ|4V}vVO0&AAO+Z@<?tK3OO-5%q=b0BCb?@1@N;C1vSrqAP*@-<wF@Y!g~lNW$)UOqR(jC6sxN@m+#zcjG*t@C*Q3=Dd^mQ1ss$JBsCJynv^QS9={C;u%9XZX3a_dHUHsP;t7b;6-tY;W-izA@LF4ew54JM@zdlW!x=J!Kf}_!?BPU=R#5xTXVI?vHeW-50E;l709?^;GF||Fr{#_b#)LR~FI_mB@({%z4;w73y4R?O4)z1@o|3)_Wrm|qVEwbI&sW>wTjgH~$q%w)LYgOtC)-91Lk?@_v3q@G&wMnC7Y~t(t;H{H2n<Lbh(leigID`Jdz8>PFlYb^h<qAmczhNKkmo9|PtS;m{*?j%J=pkwj9$LIMs_S35m%Do*eCK$4MB6lb8mBoBWVMAh!n7_Ox?W?$$aSfiA+FxbkQt;2*o0QG)E9iU=vn~TwQ0*aGv)dYYPU7So_%;7fD!75ECGolrwmHlFp&P@mOKco95u&s9xgDz1+nA4j-BW)_+8^&QZ%xV^G-qtAK7}TooZFBbN`**-_c1KpmMDi`iiLnW*Gw>_aaXPE|VV=P1&eI;c?eT5>Mb69m+t#p_&*>=WejEP{?9d>0d9yjl>5h(Hg!Zkx3Nju-aPi%%Sm-JRq@XUpYiMrN^>GpE~(kpajH_6RLPjsO*p<4<3>XtsH`&0!)LGWwql{WPKYR-^k%lG5uH+`NSRKq|a@>hiZoGsxoA7v06P5QSsh$4QI>y6$Efw9;6+l}406mCe=qJqm;xoZ`#ShtR_lm^T%5NI(ub+{%V~cr|Cg6Br=pVj#S`sNxg(Z2lwJ?Ct$SFuxn<6fgWLnBB=wM;S{Ehff}zJu9X(l_h$85}jok>wA6;V$VOyYMI2R$g`&)E8ac8>2$GL)uWHbNc1D}Ln?_E2)V7bEy6w8SmNFgFYct<C8&;Y`Mg0#9|>TY^CC>eC~!i4i3*>kBo9YU?$A$(-A`RE25qCG5^5X+<K5zmzlQ3M+tTXNiAfX@`3d=6igC!N3nx6<GqA5N9YchADymsWc(LP&oJ~2195|eTl2t>&0ru1Dv_u1$tX~@b@p+P8#xeh*gjJl#_IuNyT?&Su81tC^{ZJ9TukrrM?D%GtMm{G~^Ndj40Jb_5*rGiR$6ZPSO`aDaSpYa;;zH2$AAee=4A~DH^s63I+IO+Rt`ih5(1nsg`>BL03{dF>d#-b++9q<5&u8LwS1Y{XK$-s4<27HL-TY`hIZje2-g`jsu47vkgLD3ggN7rRfOOjDYu7<3GOsm9q(zkF>^TKV&=o|P=CIEZ`7r9{Ay%qAMy;3^!n1XnS=q|h3v-@VAO`-1!}K*O#Y)-n{1*p7m<`u*e($~3n8hD%0(^<~jy^16V`4bDN*JSzvsNkiY){*)wVHV)Sg@rK$NkFTLch2X$EYW1rI$fmFsKOadfn#vO5f%Cjd}HVn|NEFNx$AcZ+BBJunbzh8hbB(jg3(J=@O<mF09ghpuQkQqk6?2>2946Hfei<N+M}-Pqz^H*Ro&HXb@FmEo1qFConpg93pBvEEt}xsIZ~hShzIT1yCe5qORq@=tUXmIq%`dAOJ^NNlg|?gD=9@-fClg&3%=L^NeJ|BP)N+1v8$r>eJNGv~T&>?c)2e*~h}cBDgNmyj7E7MWRsK2VgZ;D*F04kaCN^=vYdeqFf})WGd{G-OlSA&%qN#rlPf|(PV!*^*iiE6s_;oTn{TnJs?MsTymy{U2~Me_^+G}$#9e9DPm>(={s6<aJA6{$ip+QauM@QXwqr`drD~@s|fvyV@xiz`Cf2YDmd$y5l9yLwPrVlVNR|g`P5_@6z_FkV>G_DR|MGT6)yIO0jslw%OT`AHAoY`{>BcI=%5B7tu68r8b6GhGn$URk6n|aa}=>v72ETFc7P_Io?-nb>i>~_wET{g)sWAHJAK24MH=#ybKT@kUc7b400Udk<i}@Gu9xV#!t{&y_>!u#ot%-5qq~3cYi%P5x#EGMsw+3=LqLu~f$jr&Bq0vACAn&{Ctag*`+A%*ZFvY<M6VdBJ|I3!^@{uWq%k&Godyl-N-;7)q$XrIyA@g!z*z`z&32O&(BYrp)LNGxB2C%}4VGEPiPW*)Th##!4RIrjE^otyVZh15cUQx<HMjjS4UXk2zJVGY29Al0VNVrp^4rpEtLN^8%vRr;$vi;-tBQLMpnuLcZ<QGU6lRn`y>n{RmLvD1CMC-sRXc%A2I|1tk+=kEt)#HA!&db2f@p9Phu*ThQoQH!5-9sp@R{GCe(l6=TxY12d$lHj^fI+!pTM#jiIH8_YyH%(I$B|eT}1#sRVl_JRaz?B<SHyKKe4Q%>fCyC4Yn!MSrizLslyH;aEHngvPg}N<1tq&+fH2L$Cae$08QhrH}eO<b8^21TWI9y#trD@YA0jKj`D}J?&0RIAY6b7eJ*q|HR$G(B5ll|`n!-ZJ-!vu#5dzdKVq)|n58bLI4YEaxMR{jD^%uOW=Sn##;C7AV%$RqDXa1~xcBtF>Dmx{VCt)<SFyG+E#qpGOhZ~Tc-9K>F)d!#gu_DWV(A~BN3IMU-%w`OxDz0OW3gLD+>??t-8j6=E_4*J6zuG;`D#jUmWW(MXHj1sPjvQc;hPorWW-8Hd1A8XA7^gX{yEm(^jQ79B<)Y7pIGojYAHipN?(^WtQEo1!bUWObD#GD!`TVmo2A9dj5N47pOm(Y<m5Ayx-T`i7O8qKtSstJgZO`?&21M*t={t$0MlLogkeqL_j3m}QE#cMvZ9(z_sF2*yGXK>SBT)cKFDDgO`y^hKmfqH3oYD8PY_#U^`kCAF`$tKuawx8wlF_lassLquhOIw*298AWMlMTs`ibN)D}SOtZ~)y8)T#oHJY}9<xetd{Nzpw-a9LK(iR<E1TdvMjjuZ~4sI%jkB(kG;V0pHS0Am0t|=}}HB>12fJmJ`Vb~yyx17nMrSU0$d!S9udS{N<e~amvvERXsHTU7dpD$q-M+$1fXO!xTQEE4Mo+;6R#xtA6X0@N``x#4<u4b{){_WxoPE-axGp-a$jVqI0$CJw}hzrB{6X&@l**&YyP4QQY0VVq}p#5O)f&e}erpp7tBxRJx`FGZA)C;)dZoblBLXu@n-KE1l#ShbNA-r!Si!tI+3!pwV5Vq4j*k#&jSxZ)GPY8JS=VF5U)@i)fDpUo3USN*4^TUsk#3Q6aaY4guA?QQs_I3s0Td$+Cr2#s4V#L4__)W7Xon!-gi>JhcS_ShBUY2-Wg<2I~0|^}rg>_}bua=30_H4PG5Lzw4MeU+6`V9AqFRxY{{DspiQGIe?rxhRbueFba7-JCt#m-ICw9y(zUqEbaO#ag6iwk?@_-Szjc&0w^x`%l8?82{-v8RcR8go*yU`X74n%sLHO*SV|{IjLGlkaYW$(aqWN=QPQPF92>CTOr%tMt6&uyq`booVYrBa9QkN&`HL@Ra_aQoT970~=71m|?>9#zL$A1JQ0#-8{Qy4lh?yl+gOl*kbT1_O1~Ypp19S7X{EP%?(Zw4ov2$2-X&?lyO`3U1DsJzU3v)gKP<T<+p+2z`(^^ii%J@^9toHQ`JoW7p+@v*Qb13X#^+w;#moCnK~;Pek?{w@f2Tl<%jPDjm*R_|27$48NK{b6TXrK9s7|!75GyL6+@loT<H&y^w(AP))@2<wLbs^ccKZl19hI390JQWb%_BdgX0*@T!Lh38i$q<7+iLaja4<Y+prLDxvWpKD-|a(*tlOXv`_4ySi9%5uBwDH)-WzDbDO=$oaR~;euZ#Lm%ohz7IR9kn28DUMSQeo5|}B8@d_V$d^$SYDFNN4irBKDCCUnNo_$dd?{k`^(nEV|aDhCp5ls3Po0U=_J+4c0WerA(jdLoim55jpiBqYoB6n^obO?*+(Cx8a>mVjs95!T5Kk-hHZiECFSW9VM-%;eXl^Xj~scr*Z;C8hktKd|m(NH9Mtcu;z(H_OZnR7_9+RS1QMUh8CzC({JyPVcGE&{GGQ7M_70QFdb1=B8hfN#StsaYmc?BOZl8_I`-Tf;e~C>Mmi(cTdI2ZejV8o-zw%1rk#Y3{bR-fjD#4@;a!khP6jZQbp4nZ`^LgS)=Z;<I4mnfOD1f}}Ku*Jej@1WB}O3JR944?=uW#*<Gb^&_(FNlSA1fwd=Cq`h;JMGp+I`|}@vl-Hg2KV!shg3v73I`J`JzXm@$(!~?d0bCz(7DG)ZH^QT}g}tYG^}Mn+q?4gA8o1DX|B*K#rj$|*=Bom6OiXDWtf^S&P134$gw7X2^8*=wY@z&kjRcmwqyxH=uz0DW;NJ5Rb}A11o;+=?4}@G>;aQ+ziCFB-%cy_c_S3-=x&OkW8huzvV-N2G0X__A+a|EA0?B*j@H}dBww}nsZ{A6DW4v>6EcuiNO?9UuMy04xO6_AiD#K?umyQ_li`vN*A+m@Yy&b?QHd1TCjnc}uDV0xQ6r))C9V!IYe1H$1ti7@t`#a@+sQMr7y(jydwkWO%=@pn>35DVkvtsiI)An5&L?NA&%$Bfg2&W80Vhu8!dJ}NzoU<n?IGz^l_g<I`fiXL!I2l*-7{i$>mI_5QLm75t<JxKV*pB@|_5_V6RY)6$w_7qhYj`dbMZjr5#rs`<?H6yk!@>0YOvfG`Hu3`4M`&B@kaJy(2B|VUX{dKa9@Qs>f(O=;<DidN9bt9gUY*jhg}tf9ts^(EaxR9+0J&9;pdXoK>#jOyRGPZj(r~*v#$0J5tMq)JZ(-HKgrvMsbjST&Fb(*-mw5R>S5?pMb2i#^V_N`OO+#Ff(h$UzO)NseWg1Z+{r*rA<h5$uzjUubzRt+M;_v9$^q7p*w53qLdpWF+N!u*wN}ogU2>#T1t!Xq;mNNbefK4DsrkHE)WZV8I!x9XE|6;D{Rj;I2%^qkxo(8(|yvQE+UU3w(d~L9~?NyDk347eN^H#+d8%}iLqAR|owp^sD&C{(Fn_2H23rlSRB~IjuHN~It%Q$*_d$mIt<^#W65Y-xX!AwmO5B^k>w3j<FrKEf3JcKWArLt&*{$4Gn{vlauNKR!E)Dp?8sH+vlW)-Il$q@X67DKDhVJ>UXUWEYPr^vCODwGcMb5<SB7JqL;i6E8GrH2@L>lDEUUIGSW9`O;Apt8Rrj^Vf;PWkW#sWO!iM^|eCtY)bE(trmEcd4=(Q0uB)gvJuu6zFN&vWHez^<8+=$}yBs3xJA{=SaIo_x~@&pj2DupUC(C#h)*wz^7K6(n)tO#{drHp+8bo$NR}7i;I<l=`I@q@!SB=0ZLjE&cZZvZmlyLW{2_=<PX@Gr2(3_u$xQB*m%7A7n4)^*kz?uy2p(_ihqf-ul(+jZI7WJcK9eBO$!&Cs#W~Ww)vD~#d`)rYCy)EWvFks3uqjf@lJ&VUpWiGx6(7I{>d}AA7x%pie?D|gJB5W5DvQEUPC_$eqOi|gxFSt*_CKCF&}(;-uY;vHuXnc`ARb;hxHw{tEym+v5Dl2p8xUg9`55}Le4L3sql{C@voORLQW9Hho=fYL(6M@!02DldR08bzZ)XC-9^}2)_5}uh?eGbdL7;VB({GYOLW0ll_IvmpH6R<7Ys*x!|sPy4W>aoC^+_sDSzdbP?cFc9LS$CtYUA|d`tgjP100FQ|A>o8SlhS0$&h$28yFvIp-;iy|U5&3`<D}r2e08ZOGkqni=YPOof7b4rSouL?S%pYQzpLX0~Xf-a;eAU8Mo=0%g0tcE$oZxS(!>i3P0%iMNKMuZUFU(`4z$)#?c{#(8eScD*Wn*9!onE=9aN1UfEp;iHi?z?3JmO_gz)zr%r7$CV#WYFxjy^x&N+A1^3&NhOTSf$-buWk?tMlY??uwG^~f;iXif{i+ULX5szN1+p6+iTV~sdh--ni77(nmohYMiY&2C0?1V1U7u=FX>rMvrvg$dUi0n#qQMsw%!ALYq`k5)lSgGrImT0}BlrjsX?ZvN+f<gCfQ1VUFKry~bnxQ)7vh~;5fDV(bRjMcvB-7)e-(-^4A8CZ%7ch0Un`nJkn<6~xn?(sNabjT(dQjgBPMXeL55r?Pq3cqeP0Y-jDo9vW#O`6^Xi@H`XM+VF!ZUmWX4)Su)>i$+)%cOH_WpK)`<^Yu=+7ptZy~wHnAuUwp`e^GiH~kM12wM*P$?v9yh7s1ZFs-@SlW3{qV|hjj$`dZzxUr;Cg{*G!rJKak^GbM{(<vh-c_h`pJrdngw!Lz(%KXcfGt<?@os*KuQY+mewVzC+$1pBUySZedS|SEA3~^4~M&+S@Tjn4r~Xy+F&BTH=Tpch}04|<-UinqWJyifdh#zCx8V93!SIk!*BpWD-`R%D869`GFnU=Jere1yN^VDxNo|Mw3;8ni_`cz_tX83X}bs9fhB6AXrW2ZVLDA5Z62knK&(lJRM7v^DrgH26NQcYEU+_g$;Ld3d)he&wvl0bybo7c5=OW0(o*w^@E|it9>f9O-pg$HX*H9;R~^l8R(oNT2ag1_tt~2jqs&InQ;U+IiMIyqz1v87)?F2i#X)NlqP&o6pf8g$YMR!$-zt**&jhG5eTiT;bE^*D@!vd~G|ODPFw#}fYLpb$E=<4fQrZf-9sw?mG9n}!<cAj+Yk+k0HjrWx0zD99iJ^;dnpz!UkQSmQYtPd|CpQdJl!!2@l?oxnNtVT)3qI3!>A03~tZ<Tog*<>9E<^A4)@cgbf_Bnw<n{fTzv9#HoXq`Yma+*O)picHV|dY^EQ^~!g^##UR^47+*0v2f>230+ZxdcWhll_h{F5q;ONan4XPig(n$wx1;&{gmll0fN;WFY=ze+s{$mh}xeRS#%>3;K!iZ1~Yw^PhEV29B<tSdcbiy<Wm?yuh{*-RV&Xn*n=BC70dn`|9|6ofh4EsInB7zqHD!2w_rb+bM38T)(!1o94a{JFV=StPwES@)=LW^^AIJvZDu9aI*hVb#X5FiqeiV*hv!gR-NR#EMeREgRN54)s9aDfQIYaGlL1a8j1`(zMn8Dd<N(y_bqgig#9NSLPj%V^2tI^=8p?{1V3qSa46V;uJr)xCGdKh)zErK7+0hglcahD5G2$saiw!COQ94Ts~p2P&()~`SIl*t9!|SA-EA6Ga^3Nm6D~J@AAyC%XF)Ajxv`&6J*GmC`3`UOMpZ)pj2Br;QqE-#|ZHfH_ww?wA>Eh20ahx>meFrgTDn)-@Iip8YSzZx>i^^*2c022X&LnXLynXuRm70^H%0FWv#?914I~M?f}^1>}8}!(}L=!f;?TRwul~2-8ajd=$-+>yG#GE8CihMF+F##9n2}F&4xMeZ$k4*@^liaj^1_@Y_pW7EGUI=o3m9Tfu5-mew5tVoeSv4(=Z6;gd2dZcu0(vzo<^lx|#or#Zz;X3C&9P`<ZWuY8DN4=w-GwFPK)?NPH&1Woyp@_}+q)60E}X4M|Xh7y&OoR($z^GO(pkQ)Pk;*bxxuLzeAQ_JVTY{v=ZEPcOiA6GRe73k9~5Eqzrh2fWgC(5+jy5$*vQD3rzO?pmR^S~1vU0JEg3jSrvQ)=(NQ{V$B8vk0AO!fyWwYQKa+f!E>~${vQzk;%qAP}rd&$;o4eE(5?n!-4HdHroD)STTl?lBx(c5pM)8HtaA?Ce2^l4E!-!7{^jm8gl{Ne4(amFkVmg2H_YQ^`>Y(va|!Sj*VuUH}B28$I2LMe0)RiZqqm#DAnQ>9CnjUdNGnW0>GZp0J9*yApqsT?chntIlJ5))F7*w&FygAB6znPr4KQlz`=<x`RH<U2B_Bh0DerefiWnv%_WTm))8P+Qtn`cf`qZmsv$41Ef_Y^VGwClNYKtJuhfgq`RzIa)K>&2uIl)2x4SXAi{D?a8WYr#a!0)SFy>vM?oenP<7jNnaM`bRWh0S`kJ?-Ut#smx8#>=12_x5l<~4Cs@6alsaty_qeuKFV3wG5iEuums1@qwTVjR>R*>BC(X)isa%;Jl%>29<LrtF96srF%i%?${*<?M%N3nNIG|3|HV%lb{cWT2u-RTcMhgCSD`a4EMXQr^sd-h5GcJURD=3{I)T^)Td+r%%4WJN77Fq(9t`SyzE9!{lmXguST6neKUVqPX0@Y@U<QFGuehTzuu*0;snKZVLBcUI{ioP#YPo&VKR0vU!k?wgG7$L_ogxLvJ#NoqLQZfRhiVCJmw13)r({DjA@)@S7EIec*PkS0?80V4fxSwJKP>q{X*2DkWKf4jJX78J2<U85x~eZM~&0fAhPdD*q(U;F(W{!zZY_L(pDLR0Qn@rRns29c}X2|Kl;#%6ZTwMpJ7_Z%+Im?lmIRR(pFSC46#LF7RzRMI)iIs`cyfTRgYeJvoC?!6~@q$~Hu^Ck+Adk6hQgpP<duG%P)1Gsg9MP_rs{+cASGD4rl?`ZuM`*cvSYmqvpc86u6;7&y|^Bo6zCPj{=*c~He*L9F))A0}qHbh@O5;;cZ?CflHr?{PIkCIW~MwU7TnYL&uKIN0Oxn*XkHfsv_|+N+=J<vz=_z4pvEdTC4}8c>cD(6EnY2TRBlaA|ruT$lV-<S94q5mw2jjJF6uj(xC)zLI!vpVjT#E?3*j3p4a>v^k1mHG%+%*HsYf=G||j`n9WBRYup~j&i_9SJzdJQ!~?Q&jHK^)Z=>;W|4;sHtyzsdz=N7@crrWE!hqe*1#CiJv?9ABb)$RT1Cc4n`Rkw7fg%ZICm?<@^8eDJ74_ro@1_iU(Ht$&T`k67Zm<dJnnTxZe-h*!v~M$ZFEMALs;o~+~=ds$7iATd$MyV`aNNg$fW(_6K)q*Tsku`Zii@aF4B{iT*;GRyw~?sJo?4!{wt;Fo}SI6nV$2dO7Mtt{QScF3l~a2HqwF-Z(E;>na*}omuBTOr>w=Dh%~p17ZN7l#;4Svxe?e#ml;7p8Znj&`G~DybcAAJhU08}5QUbRF!-yEkXkn^Ss+m5Uu*zO>wBIH4$0mWOmxFY9SO~3_{q&i+$bb~v-$O|6~Q>&e&R<rwtfq;^kWF?Bo+hoMW`^>E`FO%SY7LxBl3nYb7%+V&aYePbZedclD(il2!;b@-cSsInp=zv5fZ7miHWiyYGOs>El|))!WPz6Rc7~Q+8Pjiob*(ie0MR{w|DoU)7Q~AkAyfyE$m1%_CTT1f@no;LnE3&{Qf_jR-6iz>pVp8QY&wD;R)vIkC`NQ4nOv8*7I7-2+2R4_*?v%S*sd2N-z4929YTszHi0ZOOp0#K`9-)jBH3Bc!DY%l3h<OP9O;#FjLUgX1vsO<a&C~&8Fa>0Z;%Cx*E)ph9XxRvD$0LXn!zVWC<u1GbhTshk6~5-#m=+RpQC|pI&>YwCyX5PTZ{R_2HvPYAM2eWZ>6A$ZH1|vREiS!Y4ZyQufR#oo$gJ{uYUvfJ*1((JZ&feds7ss-hX`lkX=h-8g1v2MOC^r0vsX2qBjhMLt%2Sr5=STPo!g)rt|rc#ff?$tw<X2c6lb2=$AXH-LdH7YW~dO|U9sC@YEw^{d)=xfV>=IA9ajGbLtgMc9C8uF-1g<{Kn5Tm?aBS}+6nq(~CP3zE`+>uc2d1;O~y!<>MEr-X0`S&sl0Sy2}PRjZq{r?4el7$5cqyH(|i6rZ8yPb7O2T08i12GE2^xtCIW+CC6cAg#|E=bxoLV`}<e5dZ=frhx1;l>KlTqSoc>$~YGkZ!UG}hq++PQu)?q)8}z^Nvgozv{Kv>k88u&sV1)m?v$-pPD-H--JLOkgZ=T!C41~FW7*8%J7Xs<qV9zEU>>)gX5TMAVmyuK<p*9<#3q`qss>I7&6x)gM@zPips8q77%FwoXuFUBSDtMu&SB>J@%DP}S3!58As_?{CF0jXwST)g@jyI33rEAzje9q`J4XU2pa%%JwQVz0%_eX7?pMxh-HxD9(m^ewq0(nI$A}$!+6F)?ChJ+@U=B-TJU-?qp7o2wuju4xK_+ihM~d(+nyA^mp_K}`pR6($wV-`E&#{6d#G_@vPH}?gyev>zsrDx((Tp}qlxAWwJ8EZNCM+j?>&uQkv3};O{$6<ll{yF-I(g$!A+X8R36oj7dS<y-p|@>B7|6DlGLRPwVS?D0L(7(RyfM^m@$eMv{*+Ld{0M(C6(x1Ao2#E?5Z$zEMHnrj_afG@GUBTAt04!mf!Tm>KPRUphbea@Srx_4o75P+Sxlx?UnmY42Dx?XCavEw0VG>oyPxnf<~>GPum+Zo^B&r2-`V4oL3#|01Ds!;*)M{@;lZ?yGb2HLhOtJEqu7VgHkjPrwJM6;CT)FCdGRPH6IP&}{W(jCnaIz+OOv_2ToWbeYyB&D2=Q(yY1+a%mOl9@PStyEau$PuH*|J8g}Mx{0H^kAwU{E>f*9=WkJ)ud<8+@&=f$AAVv*%VQWXV|^{;sIF7BQB4;@+xYDu;&+s4KgXE`51Ffc%rR?phYX$%PtA<02W$mP#f@{^sQ4LlFi`MKggi&E?ZJVmzy9f2P>L-Gs8>0Oy5`*>qrn!6yc`yR#+59!m+#LL;uGLtdAo#}{jr;P`*Ps?>o;YSMdfJl=}mSPCB4@4!u_+5H_orujTAt5_RCIub&d4#h{6(5yAvy^VHyOupOwxM6gXsxmoM=p?AVdUHl(wblYkP~;BiCMpjEI`^0FSyuLhd6#pDa<=Z&R(Xi?o<ep3uVyR-0rNxjE<S3>EUSXxn{8(a7v<T`gW?z%^C{?ClLwOY~CO(J({*d)BvDH?^gBoPHxN|2QmIiIp*4_%0>hb${{I!02vz_IIr(L_?6_!v+qN==l?*tBgSH@vWCngKJQ=c<Jazxl}%LbOl=N}U&M9EKA+aF$~c~8z&S|6k2S0KQ+sYjVRw);gid?E6eztL0T^Wx%laS6Bm*#x;0qm}w0NfNX4Wc<{{@Xa^cHdo_ONl-8K-MYIK_L*p8E_hRiEy;G(FQw{qb>XtCkl3<oJg>Eq9i6&5?v*Kmg^uLM^AtL~i{zwsY<f3HKcN^9!o*4N)OahLzhUCZC3G3|^4ru-#Bn`Scx#qeAegozLkLl;(E78~F;D95g)NqE3m4b!|;o=&@dAf3G8Un3F@<q^M*+sD)-XnI8s51{wgBESk2Rno#A>+fM;3>h!l7l)xxZt@qMohH%U%zk9}q(7U~xmPl>em{6ow%aa5;+R{|X1UUd?rIIzr+)OktIJ+&F|Dq98efBVPgeJtorN|&A1Wt0HQ-Y(sc8*Nqgm0m?znwe0T2t<GsJ1nhd7w#jUspamP<4DTJ-zlyg4dPwm@E6vExw0=t|x_bL2W&&tAg~72D;7XgE~}7Hamgdls)TUp+eCpFw=>vRb;JBNZqdmF)OA>2&|=Q?^9<2NT>^iE6f+1k0gy-#8L$Q$dShAgE)s3x8n3y4H#S->5=sj7~{N3X--z@tU1}|LU#Q_G4Kw`9^Tu4-;7jsF)Y|ZXyU#B@PP3&2LWp$qNCobNH_K}>p(&dc{$z*4vI1v7Rn0Z(APjpp%M~JfQ}jmo(rk`s0I-VoOKTNa{9^I0jC`}BN9~6up!>f`II@PFS5VA-*|(^8V6OyWJqaB`iQ>1a{XKKl~HJrJbNJ&MK8FvhJKY+W6$RP;sY-;Mv-viJG)MxE_oAAen0lkK0)P=WFh4mlOAAsDsKQ${$N)C$B<<({vIPbDLU`rzQnJ|D|D=tHncektjL~I5^?EtG1bx_g3%7S5UN2cZ1M1H{gHW{e;TH>uy@so*q8{_RDj)ZaHCm(mgCh-c3N;@DVMh@A{F-UhI<H(r~ai&_KOHHH(889Av`r>ZN6}6>M>+uCB7jR9d1GzWNnR)xHKDfn^`C%S5>4d)TCH-_4!^OXC0MdMeYy|#2Czc@e~IPzN-V8Ts+<(5F{nUhX}oQt|0#-a2diFyZU$R27qv-NI;gy=4nG%jbuGe__QU6<roN-=;UfT)te3?(s~FIE@F>65$?orm_~f60jgWg@0%P7hF}voYKzJ^nIF{IK&n{qn#xJj^EQPqRPiDPguEh|&?$X*(;l>sMvp&9Fu5>V(a#vz^an>5EB<i-nAwDDOdepSR4RaQM@iDP@JmY|x0#)R4_NR;^^xETIRtM=ptD9g-n{2h$F_#Mb6ZoFDxZNEyOQvcSS}UiL(H5sp{TH!+4tjGowZ4rD-GQe!vdv>qpc+-662G2K;SqjYbdy-ZG0|O2K{CY$Qk+onT_j%35OYz$UzIs9OM%k;6oVGd_)@(8VupueDCd;Tu_w_{~M(kI|Ys_@SRHSv&pn;1N>PtbDG_F^-r4vJWMs^4_Xt`#TzoKl{S}!_W$q%B|ixsv=*>r@h~(GD4Nb>Ck<P?OEn)*yl>lMbZKos*Uib7czgBn&$zKlF=fFo8^MDqs87A1YKw`N8GC{ufPVZ#22{lH&)FbrpH?yt581GSG}+;Wm<n*mAssD^Vm!}rRpwQ{+$7dA;kxJ)S<??KKh4;Z$FC5W`4ENuO+U1gZNel$;SyONVuS~#;;af65SxC5t4a{<5&+A&q_s@k{s4qtmr~&pN%=QuTXNbk=y4TtPipT&p_Ll>Y`@7gVd3Cxw2vtt<G7DF%qw3D1dOdf;=uEWh-7ukF=ws8B%?z*1soW*k^a{XFrprPy8kqHBZzbUIT_U2E@08xT4PFw)PmHOJ8w4b7Bj{MDiBhmmLP2mN?rn+tty}4hOt1({@8pov532;wz4kxRU9iQ@WY$$7ULQ#Uwx;TF&TTFxd|}BMz(WwL)ol1URv6|0-6DB_jX2JKI9ve+tN^>Uy<$oPpQMjigLLYXqS}Y9fWZ>4Q09@`MfiVcn^}r30FV}+P77bF3|rOjr!sIOxP=28amX2#Dc22b-}Ei3nDoyIX;T>+RYlXTj5UGGia0`PtIkr$j8CRdUhKLXA#Y?fbVYjs3QEvO+g+AsnFox#SKmPeTOspX<xB7=NlMWNNc_vqDAaXnloyISHAQ5VF9O3pxN~Vm(w@8GFdPvj4u;&3&Zlhav~Y}4K2rbdn+w>+VLp>a+>Ix!FABlA*1UFAfZ6mTpn@L=W~QdY{bOQw7C<&o9#RyHZVf6*kZc3oz52Z9Kf$O|1jEZ%?1z`tPKn1w+jLy;L%tQaC)8_X;mp2<3NGR{qf(B3h^y;(VX};N<%*Zd({hY%!xaPm6qCUOhJQ+oo20ve+%Ch^~wqB`<=aTnr~KK*!<^dxTD*M+(90O<nQ!tusz^(1o5k{0NoI2*VQO-z`$Aa<Q?c^mtfWaySuu+rfToeG=>>><VB+*x~`Kp05gq*f53afMqW&Y*8f^IG^aA4@)|TTXOxp?&Hj|;uD+K8;$FPElNM^ACR)M9W2s!AVwtX8jdGHhY8O(^sfL0e15|ne-wK66y45ifYYd|G&&jJY{TtEZIM<9hrt+W#82$^cf))qUpBEpf*|+m=fc>(?AB3Cg$!8#9U*uSkg^^g`=(+ArWS$G`deRJO#2EA<K;>9#urw4aQ$(s<<A-rjDx?vk&QJ~GdIa$X%Rv%%w-L1-N#t=@^uVr7g%vlus8x$jJLC+1+9DM)s=hn~iBnrqm!9q}UAh*wjSDw@6D%IAHnd+PKKAGx>1LVOB{=YQu8eLOoNA{S$K)RgIDSR7e6^%8n0D0#G8gW6B+$o)?=U37?lZqZDP#15Q42%tQkwRKOrX0EPxEWfyN0Zt%8a8sR6|wbjg!Abq?6H;`~~oOFT|8`ce63!^O^Zp-Xv>+VK><XfTjLCfQ@we#gMqI4o{aQV?v|UE5H0|q>chtd1D-*kwBMF@ZQ<6)(qI&>err_q0;>n4rMo3yIbKpwpUk9wmLM<b2#Wmdj^O1$BO3JB!XAKL~q%`Z9(nIlVSh+QfX{dfW-+HEJT~y0pI5bpD{@Zk|`hT|IiD|rdEj^H{h|m4*N~L!Qo69$oih9#~d>LeoU%4Ki=9kMlp`Sn$pU~f{TO;#pPf|mevbqRx6lw!HY(A%9dHGr0dBa^63@G(DVFwxAdGB0fw^!#k<HuvY;vQRSD-~DyX?Thl*=2V6c?K(PSie4?;k4lBgcZp~iWjU?~~=HN}VJ(e+WZG{aGy>syq?M)~K|^84sqME8Z54kT%RcCnwsM^fVx{IKQGIah!AaM2Y7eVJ^!^_24&SSMxvdFMJdCSnFNuSuQ`v1k39m2!gsA;L8;m@o+Lp`*3Uz>NFq{K~L#ZJ-t9+p-CDz9G>XG{x8*x>-x`Afg8S^`1eVjLc@-`)bGp;Ut>JO_a@Xp)yY^PC348jX0CtfVmh)$R{^7h&_m7#$w$>Vma=MfN+>U2@~!)#iF=jkas0UQ(<~zbN`mg)99t)vGTqIJKjtNV;1t+>Tiz@+J>DnPJq>hdcn{UBNRqWDFK)i4J&1MDVuA{UIr!Y(aL-`UXoBr<=mXf#3My2ZeLsRbY@_cDE7&ZTcXEgs7KrI4O1ykdsyrBJ~%=hY4%?B??Ak)t6s9vU{?bDji=4gCAX8Gu>Su9(TQt*i7vR?bFW9WI#33pr`XE?O-k)|YiM;-$RM}il@}IQo~5@T{P>B`uqRNK@s6%BQv1?r1)^X9|Ln^h6)`V-2nLMCk6lGK6RQ`vb2h4AP}r+0qitrrwj^~+kz9rj^c(;o3&x}9fck|2AlK>evpzT9^w(CBN#$R@3q>CEcQGHl;6zgUV_Ih?Kpu#zG1zr&zL<At=MQy1w|A*AltBuwz!}-emk|I>d=hasqp*TC?7!5AwlGHli-P$Mw<HT==B+J~->_i3gl22n$8u7`-fBC=n<Uv5b7;6YUW7Y~0$Y{!wT-Qa0C}ho445=+*YiEd{Dmz8pUx9<N%(*bS9zH<w1fkSE1(i^>7_6+@0DDCZ83%euSw!dPTL>u(84ViyI^w=$1icCbDk5l+yhr3!^5dE?{eKVPyH>DFGGt+&n0D~t;WC@YL29X8+JWyl%T6TkG8lbwv#0Ar3y(H*x<E*UonV)Kewd9q-64$l2H$)JoN<eT-QwbM)CP*KDB87ppmGkg~183b{9jWv($jdwZ$y5Qw}$`=`cISV4s)B-z~`lW3-Z-BslXY)14isLxhd`wqWnz0pu&nu@A2|&Nj8D+Km{vL3o<AL`L<R`(MEL`<utjTl+1mDte6YX_5+Lkuws?0>Sw`#c=nA_?%#hy+x-aNY-@Yq7F$IK_Id*HwZ06K&Qe(W}B#&e$1&r{B=Naw3GjbSSNrxO|ffFI4u=ZO;9imeYj_f7+h)ll6gxrSd)|Bq4?22W*T~p(}447$NxbVbw1~b<kqwFUrfUeJJ7#<lp3*p!@~S9`sNa3`4^N%q2yojho<73`jN7qOiF9osR6PxARcdU#nSlm@!7sSk*1!yqKm`lp?{goij{34#pjzBj(NMVI;cC?u20V+qe(c+M=SJAWF=XvRm`p_YpIkS3=`ZF>eM^@pM#*!f~LO-dv%O*9GbQR9?O3kfH$<rZx{aTo~j^~s)0{Z<rCA6tvV1ZD5ad=##M7-#lDf&=Ts7Pu|q+@Utoy?$Za3)3mHTIX<DyaF`B?ZK15?2+#O&fP>B|sXvBesnkgl6O(arxdt?d>5mizv(I&mWMqg~2(b(?acWbZUn@%v!$o1-7H3H^@H7_{T3+dbe>$%N1X}2jU8!f{O*0Rem<21|~EAw)tzj4W@=23ydtFD>W;<Y_t6Mw}6uP7v7cRZ6Tf*v7L%C5zprtn{z53C@IuKPp}Fo98kxDTgrS|d#(S+X5-_`yZe(0FiaDwq<0fNY+k;E_(Yp$+O<1O!rIP)!dfZI@KG+^97}E$6Wr!XGy<A$T`7kj-UgqDGHclvKJ<OEBGY0KV}MD%RNVTw2Glvr$TbIau9dDDNd=Pq71b*2MvEG|k&f7lR$x88k*hpeUq+w}30#MB%}Fe|#pW3>rDDbqHN%4MbP!zNL8ye8VNir3C`_ls|MvgBr}<`Y1cMV!0{b7$h5~J*PDJpT7i*UO^tzbhIamGj?lJ-%e8_Ago)q+s(V5B3lRLwr@*WYO%@JgFUnKzOuK41-YF{^@2tp9SdF|X&od?t>-m7a$oM)IVIG2nOM$zp(cQV1($3S!KMYZor<_FLv3z<GEZb<;2dZ$LQk}xX)6{5`aOVvlxuFxtMHsiEU~0dQHg1oVfvHh+q@>4rW;Cv4vYA61SQ-GmIH?a`M%!)Kj5MJyntMOO7;9*EP*sg$xSM()KA3Q@yMQT90*x0w>O`09$`Ia?A~xx8?ZAkyq2*T)g+JHFm<=^CPW0v(ODKVrz(3DJ7enz5d7fsAJ3!`4n(?`cP%aGRwA419}UzPjc~99jd8&-gh_^1D`#&DH=58ZT_o_d|Ik$#@%*0~3{Pl1=+|wk?KOV%$t!<jKn*G@crP#rz3o8mjpa+lapu%;psu+I_}vL!|IX)TbTC=3BRcYi)djDOAPbXFjHRE$XZ(*ENPH6}DO8Jox)||7ksP-<p|NPm@`Z;?b3}ePY0H|;`=&=`Q_2h|yrYsXQh}4Ehh>LwsO#K~hvR}4t~E|i>_<Vg)Vx}y7n@yo&&U*|Fiv-mw~i{rA5c9@9W_qgxEJR$t&LN$MIg~Fugga%(h!UwESt%JcQP_@9K`Q(AMC=!mHA&d_hX!3rWU@Ag4YJuPTT7B8alk)4JaeGKlj8dEn)2eq}g9-<&n+aNuoQW0Kv;K=}!o0a@9<Mc`%|9S24JLT-E3Zg`3_Tez!+VYx9*kFm4Q%d|l1wJd$WMs&V5Xjs?y6@moxLNOc*&wXnk*ZnC`f=2B2T{#g<#l6em-s7LQ4^`%q?(Fo3eWI(T*TVz{NYTzZM;y6~C2o9k5>TyA{AqsN0x*`gy^3*o8XRRe1ntgc3nt2;G1v6A?SXhvbEwRVEIS~KLnlt5BJuaF;n#q-$Rji(5swS%T_;~e)4ZM4Sva9*W#sVf7Sol%prTP732cge-V?AFb$wgy_J#HBhl0kQ#OuHb*=rJw>VOizupTh6&2+zg*0H6-%w`j}4Ymz7n;Ppy4c#MaCow;w<Qq6x#<*waj;;=jN8HF#|RXNq)H-F5%6|W(KPes~ORA>07l$!e`jy5!pEQZb9=%fGH<+p??>JiW^qouAF8%zo41vBe!=!WB#F`pCh`@#%WSeeoP2LNsK3aBV>s%QBWM6gq$B?2pbsozyRhK32kTMsK3KO1$^z!Q?G$Vm-DK35R1d6JOyPu&Ib<?(+@si2&zMF0l5?3#`~Bw$$@X!b%287INjH_I4v7H$;S5Lx20rj1@d{#2PiY^Xkertpta&L2I|b1%^W!;Kg6&2^zMS$gO75rFU$*w6{TP^!sW*H%*YthbH0(u1#oWT)BbMd(rpf{xj)03ETqX)s%mt<1-X&lKqEXBWEnD1jzmdBlf-kKLUTiN~3bQHBdbT}*a&5|Yv0){T~-!g$$z)yB~+>jkD{Jap;>ZllVrV*B{B2iO>mRC;~)q0C?LhT7rRb>EqnNYKR50w)54*Faj<X+b6{qJQZjGAkZBITO3`gkffVKb!3e;D=Mpk~0E|t24I$nfRzVw)bYf>j<OSVQXXYIkqxcKiwK#p5FO_u)Tc5M<Oa93l!14=H=)PhSd<Hn2D=xJn{j9LBwR8T;V3#AM{FEw!6LDd!TE|;D&OZ5Ps*<&@soIJEPy|pZ`O)r?E_NzY6&mgL8ExV7!hmw45x34?R@2Z9)Zj#sX@`q`K+i=2w9g_Dj6tq5~A|8eClzVrZKn57<CoaZ5X%t5TM{TBS|}s{N;aiGT*673rB)&)H;U<EVs=XSZxt=IO5-eq8gPhTTLo4z%2{04NC$VnQDZZC{m(1_*DK9>brnKbr?*=qIgZ{_5N4QO4B$JXpU1w9|iJI9G9WNZqXRO{TVhewj`rHCW~_>wnR)XB#^bgzfnnsTFXeQf$>O6<E574*;id+WLVZ=SK=ktgkdJ%o#RuMyYh*2B>iGedL%LbOe?5!k>{E@jjK9A+hw?x~DA><Oo8*t#EFQcGsZ?q-WGq!jqa&mJfO9DmTC07DyCw3de?i2F)~~eBZfN*na8da{7G`Cu;>gxYhuVY@7^9LZ;%H34cSa#7W~_%)Prd3r}l?fW1>-XtQbxFPEi{kjgsM{<>Mo|9!n5p!5C)mp#IRk|KC*q(O>LHMm}XfL#+@`}|j9bj>OZ?nJVQkb>Q@Lg3vRA8#Q+$Y6_U<^~HmzX$uqVYRyWhmgBdQEPhPybRT7*%48LZk1&6B}-bi?zt^(24q3M{c>@QCwtXIj+P`iUlvGwFjD#vssT8Gp>o&Ll=s_pq>f^agci~#=y$-s4;N~NCMfoML{<70vJOf%WiVONnd<2MD}K2FK#UXkbRld4lNna^o)njH@)#P#UQ1mD=4V(QG?tZQYdLa+qbhn0Z+*<(uOcVup()|+vZwwOrwSGF;=%X16uzURgYMdxaEUpu$bG%3i&3fciBq+S;I=x~^>{93Bj}D%8-6R9&2JuH(%f&aPbT0CGuo%}U{<d(JNKZt=3r9z9-~20zi0dXMMrhH?zQ1)WQOTJV?PXRfE~|pf5RU_=9nB)S?Gj5lF(JdP{?8)8QZnIfCNQK_>5XG6WXlOjP1|NRXafcV@byzxhhWDZWBafeVX`nkAE)YVha3MJePxeo~v>sF{ug1;+s~0Rs95Zc%s9ToiH{1>c9(yo=z%bz6j7LYkVIM41!119+Sp9w%m4}mhlJaya2i~h@kORzo9_$8MMsb{J%T!HC@d*oo21c_OyM39|9hd*!tbK>qiVcI}t+KZ<BEo9pE8cVUTl@0Tk6V51)d<wPY;BV>Wa=pes^56M8il#(bDxLm0Y(%0pGyjX%hKub~~2D0G16dsXnQc-c<=t4HEKO}>N0RKuXiLHpq0VZ`5>4*0WtHSE^{)v0~dpj>B5c7UixNLcysY)!PGH;Vx;F)}uV4-s3>?-OJ8H}rt!#a9yoXqM9S!R)=+L2Ui6DK$3+!4J#64J?kE8(aopxS$GbhpBtpgCWR+on9>52F=(8SS3Q9X7O~iow@*tkTAO8=+vaq<fKLKnLdwJxdcv^jFtc$yQ(D;3H7#8^PBgP&m9RPh_3EY4)xsci(+C2j8g-D3=pBt6$EHr(SXsBM~*H)q(T6^8xzk=xb<!(>Y$YED7L3z`zegNx@kUGBb`cj+R8NTZcoTH(cWPPcZp?w&RV<ih6CbRVpU$~hu3-;Lu+=M2ira3bR!0cluyF+{aGf=Xzbp4UCyJrjgc1=hYK0GVCh`gByhWx%XoX3oa-+5_iSpZD|%({?2_7=$c(jxB%=Gi52ub9A+`Zk1wS(*MO<?c*kl;<SKo}4A+$OynU3lk_cXu=6oTs$_vm*I4qg@Y*b6;>$<l?7B9TZr42b^?wU0NU=yMsM>*UxbioaC%I~iwdSjTM#`>YdJSv2bbW;~zG?9B$B%}rB1a^#7)bMbsHXEJBC$~uGuFosIjK4a*)OaHaZ{Dn#Ai~PfbliZ1UpfcquMQxkcXWeDhXQOE&lHe;Bm!W^W7>6M4z-D?OAbe{T_RP*Mi&mkYz|**a55D#u$%1--dl*f8Fq>*hssZF$7@W+(>pCMW$GDcyPtUT)-x4t7XBbVoQqjA+QEp;ay-g>*g36DgwoZ-CMgj)}aEBfzcMuA?+NyYREk&JXm8bY*s+jj~=yS{4{?)ON={KK1k0Q@bG@Rt6`PNFKD&3N#7Fbuw1k>_OB-evHTx(^DIVC1INVo=fJ4Dr+-XItdcx_ybV5{5?8=;1w)Uc}gqlGymW@o!0L&Wm73eErFu(aYQ1ahaeqXmg+G9&m$`zAfHzpF)q{uE5X&Mo~;QDFwPU29WnctqRYYHMAyea3gZFMkqPu$xphMbSKH%MzXBpLL}3Kxjp!AuWzGVj$VX_p3;{IU)vJDAx!a#3ad;On{0`C#AzzTiXVCX-79V9kR^IzV(9LUR#wuFKiZ_NU~}&9XOFQvBt@w&B5{KxS}+z>5{PHMaMcbVy`ugVycC8^ud`@J_gI>cEn<mwy!I!JU~dwfC;oC5A%4d!l60ywF(U!s{W^%wqB-D(RD7FIzn46k75C|c08I|BdQ=4cm1?!G*93z$&{&&dhCG9itHK3Wl5`;b%owA(2GFJd0v4gZ)2ZL2np5PLUnh1M2)wK0xBYV_4j2hC;G2#56A?sj$=JipPgxj9PnUZgX`#V*H8@{0$Oany5px&h{6rDU5#>|!hQ+Gjo$wn<!06P<hAc9e9f$y-{<1<2C4cPi!?@6S1|uuNvqCjJOq@LijPuIS023^nU#J#%ml+qA2N1njG@!CO=meve+}EXu)}3L(zb1n0&k%6oMzhAqrhs^<eu04Mm~e5D$B9@CX`^8pliVaK5ZNFmd4WgTgo@A=R<@;H$=NT986?q|K_EZ+=>tDJCBPt+BblK9?{$PhMa%>1wj^~gI)mRx42u;#0^`-hYZk1&cwhY1ee0ActiakTV3Md_wKAS@`?lyYyNT8GW|<mv4S)v0;j&>+nc(O+JppmaIDRTWe&c(wRZ;!Jy_0T0Qhr(3%#hP#N9%p^m00*;!QG>Y^iE}0RqQDUm>(eO;TW40aIaQ+bJe>QnT~3qg-^tcSl{xJEHXTxw&@qhSBPwhv*tK${1NOa93E<chk9G%#9@g)E8Q25KcvEee8OUSWhn9DQ&c$!V?8r7LZcS1KI?eH8=`Knh<^U!*-OTAzfe`(n3i3&>e3QvpJMDgKW=TyU$Uo#PN{k0Xt`8)iEapz!o{+`%3iyR9mDQh^zw=i*Z;#@+;Bdjl@UE;t_df?!BOHk9M`QkMR@z7c22DBnbIWL?#iOjKa>@-hE3S88!FwU^BCm^5=A7Y%K1-+vzM~%`ncl?W|^7H+ItqbR4Z}PG1G)TOqa^S*ouA!HjWqR2c@3D#H3r)!c5Fg;uuaaH%7r$Jj~JCb(~}?lUATl|KSanY(D-byF=ZYsp4!1|=&anjFCG56|O}qp?x>ieGRa`=i7+J-6KDPMLlHnzMgWcYnT*E6!p8V~`pz?!7^&1fQdFfY!iAKMQIto1so6{g?}1j+U<oky~FwEafhTZ`0dDndONV%Z5qxyFt{KvkdbVQuOZJYMr<QWkAi9@yJgN0~p%1p2jAJxt#R_k93wmwhsC*eUlygZZnnP!T2Wk9D8MXBT+Xbq5?N|SS@X&m&WdQ$2w_d7#125?+Lw}6j}utZ4wmd!6;7&HWOrP_@sqrhm>?$d4zgOIwIIJ<vD+MBO*4?hP^*lswW{n9@OoS_Xs#$zpuh#TC5FU{m2>0P;6ODwQyH2H!^yh{dh-oQv-%y3j&&vbl<kB4~R<NknLZ`OL0Sx@`l7FmM~f>IBZLvH!*>-d>9}w{6%o=KKisJFq&+kS`8EKQ*ohKd+=@mwtY^sseQ|Fy6(GBnZF&X3WaUo+n~_5|L}QURVXlkc)<M_B^Vs;>Yq|jUr~ehwUtONC<+;sP)0CEi4j9EI0x?GgChFrfe1s19{>M#mEAk(g3ut9s6GSBnQdaM&O-SG%*d~rEABc~WtR123UrjngMcPG&Eu7TSVVclt|Uuo{616=eK4mK7;s?KMdyCFn|!n2_okjT`?TlGHxlN9LX{d+SF57a(yihVzTyg&HoOFSj!EUNYaMN+y#-v7T-paZqlOY1T&TH3_G=jiL?|EQ*%XLH2{Gk}KUEqkcDCi=E9AaWfi`=B4Ct}7-s$)K`cLl&<iu0i<}AoH&l<JyC$P%g1_QoXx>nL{piH7)Gp6tx*N(pC+jKgAr3@m$W5XduNnSE@CZ^Mc6{c?JV>KrRLE!^H(3P}*LY&IvUwh4q{bu9R?S`swwNpH4lm|apTv#u=zX?EEL_QiHF02Nw#^pF~#k26r4#*OwHDgpY;xRj^7k7n@s_0s?*<^8ylkINRC76KrG{unKcgRukRHwkhO}D{<hXJ-hXVI%%u}LD6yoEv<>yF7cQn<Xr_dW;u#XW#V*Vu<O_+Att)$2?b#}+{XDV@Y2o(4iTJU?k7(xoGgeIO(EDSS9NSI>b@5PEHPEIWZgE+{h&wDV@>ZbqnUzKM1MBYQ##;M_ySlb`wWSH@$iUScUuCx9mznk6u3n|Gzo$N4H*!2;>7P0KfYi+P&ozW&TL*D(n0c<kV_>D;p9iI|!#9`00!Qbo=pbcU1XZbIWY;t%#M_#rYYA#H`ZrqG*laEvH3OiwTO=b{XeW-k+Qbt0h&JV5VlW6Q^&(U~J@_?#Et2o1#7&qu~0OVW;jWhS`2zW<7^AmReQ#+ZR;Y-Pyedg{SGs0)<A(smf_4VpK-ZG=$u)TW<r5Fo*pQ|h*G0esc%^PGdHl`uXTDWukvBWWYDKOOwKp3p@Uf`lM~fdUWAJIYtdf(9qJW#UdO%+QIk_fD}o;Uw(OD@h@vJeO(^`I1fG(qtv=vQ|az#A_r4PHr^90r8D|+*@#n9sPw)?p2s%nMYoEux~(r?ZS0d{)~iQ^+{!BCZLW|oLsUhvXvrCYe}Y{ZsqZTYtPDoxxbtTbTW4PZV!-ZPPgDj==X9`=pXi6Y%)z+4=l@eJ=|e6^)4sti5W?m=hGBykwMPCZI?qdTmKe_$EZ{=()#p_M!eOKm_2DT5uJ#O{W=yUA-LE_VM5C6_(G*t;;F87Ukq%4%FS4%VXwU4cS5FOf=S}qG>E@UpUqzVode?1o;!{;KH5sc^)_^!JH;>8J}ohUIA2-q*8oYyb6EN4^Z*Yt3nsN0{{{QZkn;{vN(i6A%}U0aY?B|OSJ*AvfT(ZIza4fW26H|I<oa30^~zU4X>tNr+yl#6uzI{Pn9DNz2c(2pS$A*Dwr+*7RorL+9S`)l#uKk_yDd1+sb)XXGNqf@A4rHvf)7CFSZ}oc8Z({A@AM1nzsx?KhmgI0`uKt(hi4VZtN85laMrWUSg93F^Cd8$k+knfo7$KCTPoMq)-G72`JRtEExQ&_(wuL_x%LCe_S^L((@s|Ra*N-*@SteW>V(<j>K(4pO56uj8%WWfsXfwry)MHzVe<#s{`Hza>RWZ#0jGqY4g0+_KGzI6pF4mEYHR5w$p@ubiRi1;X((oV<I&z5`LmZdWUXGGZBiN_&f4k**@S+$gBDieqAS0QXB&3Cy>T-uORQS6!w`oE)***Js*q4VR?%d9KD`#TTJ@oT$7R5VBO924gA=~sfFB2@!WnJ6knTJ$xRHD5<ntdko#r(kYoSWZ47p+o`*31zB`)?e!`BoN{1lRjZ?3iy=gqPLd@vDNs%KN0U5a8rQb}=*npPg|;OtJ`tUupg@?+7}1osH90G;)!zf=Rs$%n$SXi4CxNe(YjiY%!ziLlVMS8~C#C-ReYVp3Fq1S>y7cWNyV$!($y(UA^${KK$j1lU(yss%Uy;hG~@>t5p&G|H_^(b{oN;KSk7((%`EckReHpQcU(?b(<=PUh<)^$Qeod=~e8osI0`j}{}xbQQoC<xW57cE?DLwjhVgJkNfcp8WZSVl1aVb?A*G$rQH1P+fnuW4Ez5S9ub>^e?AgDN<-WLmurvZ!|F4yef5Q{C*jvy*~z)&xrxL4_k*<4hZPE5Z>@EcFIjc>LJw#Z_~4`MZm8>itQ5oAVB?s%z9eMz`@i--IJ@&h!2R!pX`L<5Q+VP;zVzE#9f}pIn#7ve({$wHh^v=766@}ZKl1@l=d?U0cKss{(QnNT7@G{N-cbvRyoHB`XeshP_H8XA$DE2FZEmz-SqHhyF_?3=3m%qDe3ljdN>v)F%gZGHf!vpV!*qqK0nF&t^td&Rw2mBB{;IV>aM`VP51ctHu`2BVgiQncMX7*(q0+Ycs-J_!}K&<a67TrhH(W9UIz1wpp9H-3egF*>b8=v_32y@?!c?pxZ5~So+2wR(oXG?0XtFGOYcEIkfkzuIP`MeQoTNHahxb9R}4GDrogcs=hG#em%2>`x}xscnCjOk=?P@qogt`9b~RyVn!#W))Gs*2kb$eH3G_&*BdDS~;Uyj)&R)p)Q+Lu;4_bBw;>&3cI*Ks`Z7SeHRfepmbPjk9@;sShC}*Th&YcQsc&|ygS^z%RJkhu6n>QW7yl>pqW<0SKV@E#|0VP*1X{X?OA_HGxfo6%Whs-f&tx6vE%2;p4$mV1J_KTPF-8#GqY{v|Mg@y9RK>uDN6-B+cnJi1&7Qgvy0732O5mG{f-g~xX#d(Y}FW2B9W<B}x`IEvvQ(7r<7&DJnEOI`u=!(VIz?)R?(MhnFZ_yG28cPveolv=qbH8HV<x;_#^8)B2dH8}*{UIqXHZD=C%5AB%xGkT7Q@4~KpNJq7a_snsP6FNoOWKfir~B309dc7rl+P4N_#{D(N1T`OA#Y?h*QKM*=Ft7l{u-^8qM`eij>xEd0kd1R$G%u`Ue{P?vt3{yF_?zN39Pp}@yBj|V0qeQ-C=JqGz-hWI&y+C6#^zH7}5SRNCKBOttjVQK);>a0T{HEfD*%RZ8mL&u|L?z+afWH6eezWe%kw42^RRoC3Y9s_6h1-rQ9tEJ^z7~suFP=B8~!n-*tA41hbKuFTt;Wo+d|a7i7?Y`co!xX+{EvNQG(8c`%E3B2ljvQ#)YSd!dilwt34i|6K7X&c@*+VIL>Y==*R!MB(AC=VH%@_XL;80t>WZLudGT<WIw>&zqx84&R3hldQ61ZoDd(#f}U(9a}~S_E8%5@(*o>2}e>oJYypv7ufl^NeMd>sfG+*c<SW3Ra{q^WPYya=+#{|EBr}eT+92{t`BDt{hD;B4l-!gpo)nvupu9~yKvbL{2nJvT!U{T2vq4RJqw{&dZE3^4g@ROU<>sRicWjYNM-$UzwKE{JQmB&5@BSu3=|HC0<c#IZUeEM2f_?{QuX9;TNa3S(gwIZf>7vKu?tUR;^uc~7a79DIz%ijxXJcee^xELvi1W<hswCkMamMoc+SlTe7B{@%pL7(NyPH3A!i5`t$5n8NBj;>@CD1r+2-_tpExqYQyq)b2%7QfF|7-V*}sL1x99z-(n7`s{q=oUh;52Cpgpw2l`q;~Y3y)j_M9g^uBzM1fxYDir&=PNs%lWp_guip3b;l<5sZa{qeqHyQ+lA9kRe@zawW#9FrjcxsCc<g3jm&|Ylx^zIM{oMFuWab^TM!5oYZ)ktH78F0I1_E^$%bPIo+x%?7qfAqaP0$7`Cvp%U7e~&US{#BU`j&oGs+ckHYe0In|KaRjg{Ncv2?t`8xd-8^B4Dtj$xXmJ9}{kLBDWg@`GWyKd3r_p*JG!wGO=3*M~-e!PZLN8~$YtGdAer{g$_$Q6(^sqmwbmqW9pSuyNlrR{8qkvtw_d%ke}BH1m3l2AR25)^2h2=UAK*5CZ)ydba=qXmIOIGs#h(R5W#)iy&^<G?YubaImBul<iOdgtD{WghyWGj`+FQKKA+Ww%D?s}EAhB(6611o|Mq4MW0k>OKz|YS|!x#)KH!GxLNtTI1KAC^|%Rezr$5enw>#uTHfPR}$AV)h$wsD5omqtczbXBUb_%{R7C;31e`ZnTknG<-|T(gP*U7FgES=Twa(_k(f}cRy}{Up>N27&E(?g`kV9MiK$jVFq9r3_-wo|sTZdCxqVH5DOg_ewwFm1S0i$3f~$RCvq_(o<EQAq9UtjH4}=XU*}x?KhyZm6b{1!iJ)L!@;sTm&;-FhB`~%*hu@F@!6pAGG$q7*)%Ng;HzeuB+Z!8pO^7b=hU_TgNTWuc34h<i=cHVy-_>Gr7F2YGN2u6CRGLv|`jo&^J=>*}%sgSb1y_mPQveV2~VX)%Q>-N5*cZs$0NK3F$aPqOEZID2FFKu+{<}46k%5R{N!P2CmTBe|=u%r2Bf>`zO)g|`+?GSmHSy;;q>7UMSibVUf`#mhZfvT>T0*JO4?JqmEYD|_Jp7E#~^WY4>&7q#X6Top5H>b3(0Hny#t|=bnsuHp5A?(x^J+Z2QBFI9MIW}l-@}8*AQg}WMr@adaGV8Amj;pF~#-LC0uWasXYdI&X+ns!r<UJ{PG=_ysp4F^3d2Mijo{v^q>=*w7(@#umr+4qJw~IEkw}?x~TiKm_G#NVR-bs7qv9z;#y*N{AIngQrZcd{c!#_<rfCd2soI=U=Fd|s<+^dT4llyO<7d6+=X~SYT3zfeh8A(7W@2<9HX+-&Z5zHhjq2*jfhG&jKO?S3U9u9mAs#S*78<=;Or}R6=QZivpzCZ)cM43dljY~@j4XM+B%Vf<nCYjCN^<B5<X$bQG6(kUfnC@eoJHHv?hsy$>HH5Bp2aEDFq#9@r{%|MYw@?DCW&ggGD2ff}L?x+@_FheO`H2N3^(1GE9Na#iBq``k)fyUKEzEw97W9xOiHmR=61s*>zBf@IQAyo+m-W#+z&71=o&jTzjc48a+NuVtK!;iJv#}37u&)BjaIAL9uRnL2XqXDU(T{TPT6XffS2TjTbebPlxvJ^j2bT@oRx6*v2_zFc|A?}q5`pRfne~}$9d+efvKocR#z!-+G|O9(&XFw?P(yB*NX))M{VpX-J#i09C_o%4H9fkoU+6PldRL_MwqfRKmSLncEG*g);-0TEgby4}G;NGSmVMUHdMwaQe9wUbV2irEZrn~o-@&7o^m4xYAtc!E<3>%RB8Nr^y57V|2+ViVRV?u1KSUJSTyP2aXUlg_C)-Gg0n0yXrYdYgh)^#BgF`sozkwWf@D~XM;Zq<Lh#Iw39EvNFq{mm~E|UH!U@gTvq*n>tz`<Gv?8}o0hgV_n1H@<G1J=aqqvI$Pvx?RMu&wy)Z)9Qy2NTj3F@FioX<!mKcJh~&`G_I%G0V3eD07e@R{}b{Z*}CKnO5P{_59(js&CS82#y~yx2H=9bxu!_k5&IGu@_HJDml`9MZwv*0lpvPwH;${s)h1{=0N(dB9Qr4?%EB?y$)1EU<lPp%ND(>b`xHlTzxgSHI4?b{joD73#>btmuK8_^D!{arv@Cxv)~X891d_5mvg~$q3;QNdbD-NMhF{y&p=KNFCTOPp~fs5>k3F{in<<k)%1LG(O~4i)bppzBvZgDyxM9LNOO&@!t$)T?mNWeB1RHg3TWf5!Myy(;vLT(B@_1W*?^N6oJVajq4u;)CtIAPk^9`AoZQVCHNPl2$M8Lv3Ub&ax_G@>QbLqNb4BCHg#UX{4}iC1latOfVGJuGRs{JQ7p461EoN;H^8c8%qYZ18`UIV!-bC-tAgy^~-Yd9|AZOvq&eTcyZA%A$E_H4gWcNt^0`{hYu)fh;O~OcoqdsTEiYk#y#b|rLa#h25H4H)dkGYeNt%HMy7Op|L?9s35pA^^no*Y1H@XprNzuSYg-?Gn1XcTmJ$!F(y<Vv-2npBNZdB3CpLFZ&md2(FXY6LTEoKVWtX&R;m5LbSvWIP^(Mm@zjizvJ4=sV*$6E>dpDy#5(<;bxhZ1}tG{tGIxFJEzPXZB`;#Nw{7ST^Q#a6QOpe$}&Qz+nCQxf+>gqWcni#s{BP4&R@bcq__CTN0jm3nu|u4?{*(WfmK6X;8|+X`jsWLrtrekhdX`em8U%umA5Zk6iV4D-It-hZ^>#a>kO!-&5m?Hii+>E~r{vd6>B?*AdnCW@w{ts1E~=`%EglA%-2xz!4qoSv*~in3JNf!!V*um$o+EHZX#FXz_&r*VVF}Xbe>eLt-!-WY{MaNJ$4EeE?}x4ALH+aoox)tw6W%!Xe1OC<;jVRv;|FLb%FzKNYCsIDUL#pXD-MLrAg4ZagyBRhWtyK`rda@U-lYPJneE;;cy!JY>eD7fN+)Npi+6w#c9difYd{P1c3x(Eos<m9&<&1&&cBk%PouK``x_lq=)Du-#d3EX3R2k?i>N3&+(->==yGwQPpVR^=X{?=tD?eiS3mAkC-^Kwq81eqAI~$8lHzmQGZc2`e;WRy2lb`%-l;nJ!=|jyKC?M6<p_PPydzrp0q4<r|W1D1sk)N?j6?DLpby5yUOlUoVGajs-*f8|JIY*;&b?o?=K#y0$WIt0|hZBioaJHQ32POlLa`A@S52NHW%fd#6wzyINk#NolnG@`oF)MYSe>fnXj)Nu}MGdfKg6^Nzns`ef?vKBa?psF5hj0ID*vzNtY+>uqS84vUh!|JP{j5o%0=r2@*PHv)gwdv4T9;X`SwvH67gCHp)jMETadRJa|nnrQYL!zc5lC)B6GQi#3^f(<8~jF1cVim3#%5v9_LMN7y6emR$gb4A2));>lUw-gQia8rP-Hq+QB1_q|^D`vhd3rmk-qviD1g^6;q8=xMqPdi(KZ*Qf$h>_!AsPa8~wMrcc()=va$q8L>l>6HMNHj4ms5Wu0nZ%Pp&b)M%hS41zf!0n_S4-(w9I-h6vcJ(P0T(ox2AcSB-ekccADGz;+$>-f<SHtPlEx%DpZQqxm1_NfPCHq(%*foEcVuG0aR{VtO=DOAr0%l_2N0;?)U^o$o=;U)bM|hf_$khhI)bUWli%I!17c|^;i`fz;JR#7elvJKr&l9N9>4r7={&?ve&aAg$B*MP007zJQS@{d0-&@`1KdnE2rJD3i~=mj-N<-)%y1eWdWT9E4}C27F|8>QX=a%v8p`Kfj(QvyhDRdtSny~T7WvtS$fOABZ;wn%OE>5#^&N}-0W@h8a73uNZKVy09u#{#qRJ$3Bj@o_8mDHBN>+mfTtRzP%N!_c&b>P*z=+X|!n71jX>S{X!^;0bal}<LJ`RrFT<f3RTEOUXU~qVyp&r~hkSZmizH}eX=nM)({g=G9Cb!H23j1JxN7ClCkt9Q+Wb9e9BQ5XqWnLXd?Bd`hd}5gV2y4E<SC%VN<NdpMN_*|0$hG$IV*-w0RpFdzKwe@#FFDJFIF^9YE7*v1-Pag)zKiJ81U0nZ(75&u6U8-*a7z~IddRL*DJ09;r{9MO?P|bomDrEaLz`0Y(Xxu|9IwI{vsP;s_TSjv@r>Qq^p~ZS&Akl#TfZbR*PDER-)O%g%*ab`6Bhphd^1-Bp`P5le`un||NI{U;dL?D4timTJFnnJRilsb+j}exllFIaqgG$WHiRNt*<a!vPhA+OnG&t-I$^@3T#pfNlo9CjkRtH9=NyhWf0H^PG>L@rppl0Vm2}ZFXk%ujb9Q0lGdMYTv^(ECC`S9%uuB)>e>LGPkTd_>sx$;6+i!;3O=dqG2mbY^WecgnCG8~?&9E|)$lBr#yU?Dm;<l2k4Oz$M{l}Dt&vcmkf?oADa`F2jN@@wdRt3eocA2mOB^b_%sb7JJ@a^_z#iU&FSQq|=`zP<y%3bQ6_xfk+Ei1B$=|^n3l69>*tjkvjork#UKhi<BP&Ik|#jaQynfZ6XC%aWaO#~=;BtzgZ3z677yxdPLeqGlk7WI@tYUWW7iMpl{XpCe{zt*5^=BeY*y?`79tPJt@vc_POi)OG}8=Ws1L(T-itVJOvwlzv;P1oO#<A0AqV)liOmKtJJ&eIlzSuvlj$7fsHGT5kh0dj?a4)oy9@uK9B>uD^qyzpz$Ez^Sd<L_cdo(HzG0$2|LF8!^o_lTIG$I^l4FVDE5up@>-U#q=CvKg7d<3VO|D6T~RyVqj=a$s?2d5ra30M+c`q=370*nuuvz=2y)z<jRMlQW(r$56>QABIayye%B7wcVsW9O3E|yQ-*@WxIH|xqaX%MUx&|l5S0#1q1gqftBum6@FBthtzerqb=bl;QC3RH|_qe-3p<=;HIhEa=<Tjq{3E623*asa!M&`bjw*JSM-Hj0ukm0<0N=T{oZtHq<*DmtDydI+#IS?ca?#YC4P>HLClM|pDDU;6{Jks3w}TXpToNcmOfY!Hg80!CXv27<uP@NDX+ndj((}qfs3FGC~lhfJU1grG8?Gw6Eq3*1RR!y&txBpu~T+U^O5WE_UaG(iP9&=kpIGauiVr;u*%iW96JV9uC$?(KG$fZ7zW*|b}QT4<N-TVwvll`_h8I+PLOY=7hoDTmuiUNq8@O%KYvf(K4Y&_7M7h6)<Zw=90VZLTUExGY707^N@)h*2;(C#&^;IlijLrheSeT>_pTZm+mK6cMwakO<e0odlf5FPNv9tH&GC`>?d4fVbDfwi)2qAD-1%d4blPo$KsQZnKnN;>Z$!{@$C6&NA5E*IwnNaqVEMkKpcHc7f6;zeE#(Xt|Lj1+rUDZwpM(J=_x#jA6N)>Ja<nmSihgB@qoZF`W4ouKAEa>ZAN<$6mS-**+ksi;O>$Pbp63V%QC@tCGcK+tf*@j89b8doaQZnWx2zwkSBujOsdG8|Zbaoh4}kPn#zfH5*Bs&OE|P?WFd+I^dzgD`UmafA5#wE@oYb}_&s0h{LJ+{v1*NoR|4(oMhKp>?L12G+;Wd#oAF=7}QlGR|37uj-%_lqk`UgETxZmcID(#0{3S?z2g4Pv&?J|mY31}BBf!#Fy04#*H=yFjrUkCDU<<&BJ30Lj>(~aHIa5?2y<hDj>KI1`C(Q^+;dpfY40b4jB^NiIS&ba|yE@Bc$4o(t0YD5BFRHWyz&iWtZo#PRbvg9Ub3p}G;N{E?`wIV{duu-Oi&uD~Xc7>_EHti5cP9t6?<SkNaoyjEA2f@+Zz~P9pwOlj7>(!sb*%Ia2+R(1;iw@aL1=w2Q*yI?G5n~|>AmWK0{7fas&kkx)$3`w`1PdyVnw#OEr(d0{SIS}%{`ncm^STcTJ&J!EFhVJ#CKL^6f(rQSVQs#^=D~cRnS|xjVdLE~R(7>>a&i5V<Mul4X?zbS#;3Xn^0d_!T{YVj*mnIHpn7dtm#I!tDrd(mD_p4otg>C;YDfEChe}hW0^WWm=|-$DpX}%B*w=to^G+6vCG#oJsuerU&@15MWgAd7eYVy0G-9J}0rGU~_RtBCyFG7NnjCFsFyVzvDy#(=dL_+1Kr##BFhh|<y9p@hC4W~a@$j4Xy0Z=@G2A7KJSjxvsSUZ@5v*GQ#55`>=e8%kij}F>d+D;*xKT<2h?NO`v*{}fz*k5nMZ(qd+$xgp5+b|A%n9~PmWIgB)du*o!Msbw{+pq(B_%>2r+cox2n#(1VwUo5)%()NHs|!azxY)Z{Xd1kM!y_=Gu}Ktlwg(i4@fiEV`L5#b~IL~E1Fg@159l_oxz$jawjd>aZW|$x;S<K#Xko|q1WvYXxn9h(Z^;AyYNF`f*HO{4iZacOWnv8klP6SN+9Qw{~_1NzG*iW<d??{#S1%#_)%*|%}G|Dg-7+1tZpV0w}Zl%#;YovFs$0BVT7yUy=il^DrDo)S)}q~q(iYgkExC_`86nZ#ET#-yBMjSE%j3c6lfL&cI+k}KNA?S!`<>TiEkt|#NC2PsChFqxGed)ii!Ibor^(-Lr`=c5Yp{}`e6x+TrkEnL59$c_ye^o2cH<U^aGKdWRf>#tooE=Apc2xnO&B}s;EKiGVCF}{-#+Ug<$%!?7)`G$^8I^wBjnO?BE4!WJ5A~z{MMS#lY<7Q-`NDzf=N0H0_PsO{$#(@~*mUSNE2y+K!fjy2}ield?j0{?`ZX0T{Ul#st?Pd7>4w_=&supW!db&`Gj$k_Y~Pe1FzM`+sY{-X9;Eii`&Zo`+Yl#p9`1rD^|ggwK(vlqvh%O`b<}81BG9djC1we!v|0jnie?=NL`gdWA(C>Y4@E+&~!LA7^W-TaxdsqBw7t`@r#ZL%G2J>H{*4@bS)z5iWnx!;cy@nxNtfp)Vc>^60z+%>kB&`a;QrDz8w#@JkG*PGyQ{^QDCkqa(aPvG`bOjFCr<fhc`$GzkM|hrk{K&un^A<#l_I=BVU!TCnKXturNledn<%b>d@-@ck=OGIj>)9Ctm)t_s;4i{qFLd-%d|jlsG@e?d99OrvdFh+)sR@1J#`G!U@%vETu9%%*Z=KVjBdW&l^g;g$-u-C^g*INHeE7>IvC4NZLiYm5$cDEI*S$o!=!GzYWoZlTqFK5oz@dOyeq!H0MLEuus816l-y=tNxXL)$}s1lw)f*BNKET8^zMboej6VKORrU-&dwW<Y8YGiwK1RZrEbdE+lRp_GzZZPC~9R7UR+XAHFa2Hn<FvOp3Ell01B&NnzOZof;a86Bc+Pd>A8CWLt7j{^bModg~+EDH*U5=V&JmU*T#m(uAW!rPSI3e@rMR+nAf3rwu4F8J%wc{RTWvR;z|QEoP;i|qF0u6enaLZ07JPlGzbEZtpl{Q$%>n;sU<B(0~g2>`dPQUi85;Lr8^_77{&(&YIPANfzG?EpL<_jNt0AWvU;tPb;KN{aGca~qmx{|{xl2DQwIRMJyKi!OzoA)2x*C4V#Jq-fH3?R(<iT<&9#f0E*0>z6vnWr->BmzbFNB-*4RdHNP=o;V^dWpQxAD(i2#|Ee|CI>%*=8=M9`_LtM}<R_lty}!}f1O_)#I4tX>6%QtJQWcE(81*A+F~)r?!Z4n~wcWje>XFSAKi{o9z_O;zrMemEt=Xr8ft2AZP`ho&upy*sZ{#$|l}N9mId0dKogb~s0e4Sf)7KKU&c9xdu>XYT_}q_?dj_T7?gNBbuD^<KedzK5$^<-ts^$imX0Gw+e}#?rlW^pmcJ|v+I4XExz`d!7Ur<?9%a|)WXLJmy1sp@Av>5xrE_A8?^uFIhN{~6_2mJDY*#Zh4c-}qlNJUE=n$nF+@Vr}a-$4|fQkq)-mW9ig$l{-3HnyfxIn?HOM`Z3LR;|h*j5{TMs@#4*zr4YYh^u#y8wg)^r?(#j^axI<{|a7QJ@i_id<HdET1ZL?9A!u8QID4}HUPB(wKXvB#C1cL0f6VwXHyn%6|V>X_E0U>SOJ)(Fd_mr2=#B=jSo@K^AnxlWYcW+a3Dj+{Hm5nV-wy^n+9WIl%h0&lX?dbuCcDLF1##^+pq$yBSa&8lTqaecB&KC@Gyyp5_cft`~K>RHrda*m}nh(A|XQissC|l28$<jEI!2ydSlAR<y@Eq+`x-*B}non#mMLga+$Y*Esox3Z2OW#)IW8`&78b+c`>YbA!T`5UL+r%zNpzi<8j6ea)KF9w;`t+^~!Yx|NmSebl!IIKPD2wY->!Dxl56&Xhk)-4M11vX-{1_h13;?yHzU9ArvKh$H}tQ(F;{BV!uL~PVqzb0`z`TO^@vJmkFw*x=3aQn?5KqTF|a%%IlVaV<T%q7qb=18kG9>oI-{7DKvmy|A~VbfzKNyS0m9TvO3CdsB!gHgx?TxZ51b@AP>DBJ>M&U;7Ty7wJjRX{`-M}7?Ao^hsDFyHX42C8|8s?dBOeK*|1nH+(4fylf+l%lk^lrlgD{zG9)oMUHH7Db{$Kq%5B+BuS2Efl^u5qTFan0JxjvFL4><N_R}kAyq_;YP2ai{=NB37T=Gi9Kq2khalDBPbfyHPRQsi*#Uu_v_T?ybzj9e%un@(WU5i))y=*?HkQ5g7mfVh_FRik~<!z7m+$h7|noRcvQpN7qaYy6JtB-p^awdt25EzxnPA7qeLI)ef2>qeXe>Q^ul9-atjLi9B9q~RF3nma+jBgs}Kc`d*OI`>*&uOY1bRR*Mbo^SZ0wkt*Bg`Pl_9utor7vNUDl(5&aI)LsZMS6%WS=k)b;v3<t0R5+3Njn~#<~t-22qokUKpORoL$IiwRu(v=ob6nxyapKmQziKE&9_DzX>zV+;pauDfn|;?wgw7%|#VBiz5CklcCV6%BN%cOl_}^;PisCV2E1m+yK~#y^6*yarck^V-F~9JFN}d+qJ^y`MhmKe8aJR5*+4BIT#)O$k{R@`e?WPx`$9iwFNC>R^ocfIA#Ut^>5dd(em^;c!Mp%ny<EacU^kHb#a&wvkxgFApwjbTuJ6|n`ybfIo~yxH>s*}6Z;-a49WoTWFUYnAGs|lY91x)xbsipFQI#r^OkSgHl-VlzN<%Ghfa|fBy60J++I0i-r!vrc7*JuZc+ZofQV|0l0{uV0P6R4GF*y3!>}zm64tl_h*l_efI$ZS;AJe$2B|3@k0H+hAUKv~)X@}-;DZUvQcN0sIdBcBvT{<N5(4VXh$-jmo3ysD)mbHV4Bbl_Enh}EZ0G4ysz6o0<Wt>nk2Az78jJNc_0=uw7eYB;6Z;s~cW0Sw(_A8{#8+TqfzMnfbmn)Fc(F-UG_}Welq|zA^QlA!HOU%4NKmk)Y?oC*EWx0ib3C7kkx=B&D$vGqlw*b=h(4x1uCUP^<Z=Bw)PM{6j3i^o3KsK24863!_+f#BL1+H&$k3~L`F0_LAO{{26-Nwd9AB$vTN<JgkFjG>0x<=Ci6~@ADIVzbo-h*=Ytb@X(p{^;Zm%5lF1=I-j}jA>-gW`gGS0i7JS=26fcNL77W+P<V4$6m2XrQDz@1XBYiVb4z#{Mqx|`6zW97c7U4_iM^BpIvm#s!#2?F<tEbq}F2s*l0YG)iO5(d3R@RBwEwAD#J5e&H}>dyg!1%7EWqpT8sGbE2i$7D@~ueCC_Xde@^L^t6hZYeLp=P2)*ZkfNssrS;RqiAGtW{ptD3j()(cKDyLOO6)L=%yKJ#Q&1blVfd5V}DVxX)>G1FFP=6qiBjbK`I?!{&RTMo=!tS@(3RuqZGDkP6`6@;P1*fGe<}nx=+*#GuthdIxOyZ$?1VxXgp?k%eEx_;Zy3&+~1|LEvz497O%-1gpc%5%|7LXVRS<E884LdHs|cGg+7`T9a=Cg{C*NlA5{rpwUijub)$0pFtB}r92gO%kiP7{2bqVyzbfpNfAt9U8c8#W$nRb^<+kO-o0e@UMbGldu$D7r_l8crS5u496J=G}r<RxeL2vsl@W9Ku%9Fk-eGdP8^FHUSwh}8O-(Mt!Y}5@1?y%qV4pid}psLfT9bM1NO{ITCT<-OM79Lqrz54(Vzej?N&rf6rA-Brldv)u@+H*sd96^_2RJM@IR`7CFg(!|eJCHXn9ya=>W+_V$Mnoa|!1&9I71<KFeFa779{>xgM^CDn|4Lhsqio&RlezK~A@kjjKrAG?B=*|pblQrO5FwhSbv);V=e(8y@fOH{&#*};b75wl!Y7YztlQ#acEcMMLJ({TFnLU+GqwxjX%MS71N?0q<QiQxfzg>Fp;F)`10B>Hhv@cSr*~qSc+Qy-!HrG=y2hjXDFD9=Ru5KZxKh5XAL4`LNy9+j*x*|#uveyT3@45R+0aReP9|fS#AcjiJpH)a{}02xA9~&79&c`5^CbXD1-0_0$VuOx)}NJKtob~$P~O5YJ`gh5d<<C?Z62S2PywGNm0G>GOFa6CF<g^ZS?%C~3*pqI{Wl%ith2Ema<0I*5Gr%-1%wr!;QEj8gPBD-0Dq(-jHX7nT^Pd`EYfG+$$4FTxNF!UB;k_cDYHtLeBBy?d?PBgjzO!qC!rgce)q~p^?k1ZuJU;d-nK*VhF#KvwEX)m3=UuKX*xW<D8DakkxX{{8=?u=<H}u96&%Tur*00is=U}t95B*`Ml^RV`F_>Ut1~yWZz=NQ*q<b3B;>|8QtV|YEF3ULsIByJR>P_$(7enRr_?d5dLexebMzXw59)ja`9Ny6<>u2G1VwMB>qN~N++Mot)fQ@CcQUo7>(U#h_h6P!LUJ2kuAc~y3jmnr(hXOsE-~aGf=kH|cg{ZFs-8vp0fBBfzD6e)-FOG9euUifT&8RLP91|avch%!h@|(#f>I*$D~jZ3?3|dgAlYzeE@sbt3lBN|$<_seo631-qL1Ii$Y#cLZ?;ibv180bd#KAyZ3_p}2=8>!J2{x@k7u{NRmiPP%t@#6H8NqIDyyq$|C^e3{2>Y?HBWH4SXgt_c{*m^g1=A50aR9`WL(GxmCT)0sN*8>8#MeP=nut)IIG9wto>Zs26)3#OOBDa%RCewI{Z)GP0?V{V=>TW6?BR}3JgH&;tc$ub9&RONq&YTv?|0fNPt{dgs$Nd9|Zc!T3q=9IeZ5Dr`ASz57k{FF)CHIY|%`JSyl-&#u&Ee&i2Yh7&$+&V?kk1A@NmhH?vk6v1YudRP%g<Gb)x;bxT-4^p)Plh4>ssNtw+-d&bTcdx8#3s8|P^(B5%LJ_>#e`P{*8Jq{v(j87hUjIG7oI7*61!((MR_56)Q7qYUesZy;!r$Bg)nQW?pW5vb@bWWmDL0?h~=_9RS5L7MZBn?3^KiiO#W6d7H7x1h3TF1Gtg9d3$Sr9E|l6b;%%U&muVHdv%o5@JmPN3#7L!k<$lpNFe3u^Qdn=BG{`L>ZbX)XUNyzDKMZ@#s1r?&f@<hu48$L8tzV)xsElEv!eu<aQWnT`GL7M|VUfCALhjLaBBtT+UtlQZYl3T!k5=iJIvpzVyTcS8JJuk$9Kb=8xFL~$(sfgKiB=zFz!-w56PhOJBu(XVC$9lX7y&3@?o(1N!EK`6nNj!f%8<aQS0S2pOW&?PRll$$~K1;e-wzf+haTpveOI@HrRlc&%FQEB36w#0bM4Ah{EH~T_b1m9N^H@kt>f;7y-@xfduWCnT!tB*SkW}|dLO`O4$#e<kgNS$F7ePv0a1+8GdOW+px3PbS4`c?J8Bxs{ARL!VIrJGmWvg4OMloV6_xYp=t0B^%c+67Oi(sM2#_mnRXQmKlKN+t9f7U3f;>-1boAW}OVp;rL&Og&H4{ve5ywZP7Zf|`5$U?sd|pXfq+Wr*Y(;I_b@guc29cdL8lI15oA4$t5$#z~y{X}e^@35_My1xyf%<oHGmr!4xm33T`eKZODY#uI~PEP7!hsI_~I=?T#LMOSQ9?{`P3m3*ykOb=q*EYwuuDG0|`v76Ce^4~%GS?w4=3robf!+W$t^CGe9RD;Bfagv4RLh<Sg(QkrF5?|!dM|Rq1XFwdiZQ8e6_00wDL)`o&tlOj~OnPsmcBNYO(S=x`iiM_n@2}OSumo8a+3SdFoVyK4mZCSk;*NK#3Ok{=Xd=9nO3zo)U_vt%t4;7RWT@Jf@)xzg!#`swMFy}*WSJc}9ny0;AOuy&$HtRr{R(vw!_$cWpsl|%*0Oi2z>2}m+NZ6udQ;R|R0rZ_lj^P!&Aqob0^xe=cX1ECZQbf=z%vTaiReM=%-QvjTyfoW+`itScz(so5&%2f*09)BPZZqqc+>}SoB^UJ%~FmqqkmB=F)?+wZdAEq|MkY=Unh9csu13@ZZ@A4OK7O5u}ZQz*?c{uQvj<(p?aj`Vb3=<xR*6ZkX;;WfH>b|xzBQY0JORWEJO1{;NO94u&8<Ylix^=Ifv&70$_P~M7(3vnl|&Km$5Yc(Z-${c#Gz4&MZ<Gb%xKMG<0j1U=eHw%FNlwhOU1j%d(`IR8b2il76up323gGIt02T6F(olHNm);(qNQjy3JIyAdLShCnTufwSt@XB$W--mOsr@gbryRxl*FfdFBThhNQSnu_K@^ovi=^TBQ)mA1Y{$?<etucxyl%02Ea0pNLmt*;L{fNRo7W<K_%j5kTp;(cLVbN~)c$jemOY8UkAZ2?*7@p=QkRNn`^ppNFLdpM4E&BsECj4!ZlLt5v)acE!&sJbD3R3BRm0DRMH-WCF^RdltIdYA&cxS(M4Pk<CHl?>Vv`>PnRjz02`r8aATZR@ZS_xTsb5yXnu7s%2+HRl9MNE|d6Ruk$EDPnH^NF2*%0Vu3$EhjYC8w%|`07)`w65w1yfRDDxf$LxjaqD7j?#XMk-G-+2X%~3Un?$UeOnqUZNjX3k!)fu9dP{v&$TA=z`KEH~EU=L$2Sh&SX{oHkhZZwi^`_mP}LC|G`2StmWkFo{@$*91R(r7^13>|!Q%lJ9;x=gJOU%SEIfhlb5q&=lPueEqu$GAge;!mpri4nfy+X-NA4#cKlF<Z#mw2j~Qwki4q!C}?I?_jdkFu`4l3}3xe{7;U+icow>U8he;b#=>$_2CK61x*l$Mc&-JYDt;0zvhfk2NUx)jdMXB@B%!`BVXMnhG$$2O|oM@>D6^cnuc@A;*ELyPxr(3Qo|gZPloyyvxAUlaWKqaD2oB{rXi`icj68OTP!^%pqEA3i=xBag|2#ZTRp;;cJM>icM?=r89AT2>?Epu^7Rn#RE1$rfmyDcVJ>T?Cp7Sxdc&7<-z9O81UuAYRt*CP8wcWr30p8VjmGmzH|};;ZJ3=l3_==U>l;xfwb)3q0%Q4vU!8%B3Ou7W&}Xe?7En`r1T*)B_kqQ}P45e0k+WDe*igIac^%Mrv~;)D{ffL-CJ<nmxWcK&;DlgZX6gV8bJ$aunRrw(Eftj)i1y*s6d(v93U(NcRw4zsc=_ipNYBA`1PZ#3Ql6YFafy}13p|QJk4c2E6dHG;Sp3H!`P5q0P>1Wclm|vqiWyYINO=Rhz1#b5iJ<RxNH{{6D(#L~%S2OK=5B5`fcBQ4_)-d~(PF{#SdSUh9B<xBiCsLes)5y9v`Sr@rCl)rF|>>FFXxxhj@gtabq=|}Zr061STzQ7P)F_aX$J?URu2|5a#3}_i{igmFon2UkHtff4PB%O$nsruI}*>IHSMOS-yZ=a@2FIdUd2YpmJciWgJ|ZyX`}zH%LEVgowTH9@2hXJkUeyX*MtdClMpi0vrE|d5Gro#;Kfy&h(yD67)59LBd93G)hsdMv89FUva0a<;9#<$8!AluMMa*4DySWST4X!2%xnr;9LH^~=H=L@PY!`3)~&7qYN3Gt_Slmvg)>dGa18CKdAtODjMD4ujIJi_+s;BPrV!u$mQhVk7Ot{5YHDf~s;pURpk0tIW(#rcRD1<%X?`PhdyH*$4a;A8pSm$b*_GXn^J6<NnWZSD3^FdE?4^lfM|+=v!%<BP;a839ioceyKeLO$Na;y#QZenAVhBohTrD-5*xcn`T1~-uY8hIRF(_6wATQ7?IO%>WWE*b-fjGhzh-s*lse};t79WhB3d=|b^2I^dgp_*8@{&aZK3<bFj)<X+pz+VKm-qGJFEo8DHB}|NOU~ps%VD#1b)a(s0-V~sNwIrW;|(R<o9AX!EnOZS0l@Npr<@geY&Xi~fZS^-I9RZJFzXoj(xemynsu<Gycm9JgB=r9$C;Q@v6M~U`)T3WprZ`lwhs#Op9n;^j<_jHXf2eI(U^Q1qC|zF0%~!9q~Kv&!}l~L8!3HE&vJ=KYEs*g#VdMB!d5(h6t;8Ov)5ucMYse}y0lw*bC?<W(4-82b3WnP5+HRy_$LnN71XiUwuQ-QBNqM?9u2?Yc@@B~d8I;Vv6C9AS?M3s!*$k|bO6gnHk^Kc0VXr&C~Wb(&?mdi?tB*^Tc=u<0mX5g*$wy7It>+iaF!LyN$FaL$*Y=jW$irK-y#3}R{<oPavfTchVxSo6O`Lm6eF(?t3Sc_ylMbaMkQoPM2&w06*V^TTjU#?;{co)#*f0=nK62<%9<lw6Z6Wp<Hkg)V9GQ+5^o^|JST&QIXcsh9&n2o=pG{BYNy<<sb0CEKp{9SLf1=eFh+qCtn2ju&d**&i1(F%=s}dHa)FnNW(K>*GYFLVHt(GCDrg5fgJvPP`t8CD$NnH|{PxWW)X|b&p!X1R%kgnc4335p=o<4DQUU{h{jec=R7DQ6_*5FSQ*J0Y|4gCO7TB#scKGf#s|E|uqH+yih@geK#XU7(D$kVkpC33~3oo$Dy`REXDRt*Nsp3%*n5}%p3R5g}oEM8zLp8PwwXF^HJROCJ<6;BhWX?;XDS`>apzt(IU6^4D>3L<rCkFeol<g<{V@bjy@G1)zLeZ~qiG9UFxJzDl#)KajmGj+JDzFMC;jt+tw6!<Yfo5!G0ZFDc*%d8%?A7VSx6djhP6C^v^V<@gf2cWGd_opyt>3r?4P*^;KHk$kAyQ&c(&T~xRux-_6;{Q9L0_qD|ACHqfhE4QZX|Jgw<Sx-TVKL51LgXo@JP_SQDa^M<OwTJ4@|eRIi~^y0$V-{SWYpx+@Da=s;HEqj`^aBYEpz5{(yHa^qdEdLsfuS#->!J)o2Dm$NWctyvxpqmZ|Tn`ngO?DiHt!>8-qqrCcAUCEUxN+2P-zTS0;6@RkLPe^JHw{2)bWI_8mvtZOcp`f9rv7)N$$@=zLtLP<PZB8$3_1xE@t4>(&o$6R0E#v$(Lme)yotPPcLz5H=dH68jPEFcf$L|G?1?ezLGz&jJk0+i`V1UQ5?)$FJO%scvj;eNxZQ3o(z%ws<dZOT#eX2ssR8-BOHF=bo&Pwg7i!68&YqAyoe94C9`xp-f;ghed+kK+R{WQ<OPvIQIZ3amyXB~qKMcyfU?;*6hEfuD|*9*d?P8nsIPs$B|kZl%dzwzl10%EdbsFVNvS<J9$Z`pdZ~Szk*%N+M&t4HnODU)%x5@3eQk?bN29;!&X|3hfAJuT5-OrTn3vy&;b#X;p{vGye*2cQY4^9FCq{O$f@mPpw-1VAaGur}JR|^Wd4pP0k6pkNKG?&<~P?8#K*f&xyi-`CTANv-zVoZ%+|~D52-0#Q6s~bHMYh`?b)J)E{`>FLvEaIVQS{**e#->f`gFI*TV2y2ueJ@?(CV9VpMGCC{R{+G?s74up{uEwY)Szjv>QRwLC?e`{n#$q{!<R}J8*X9aHgc=El(=?VmIp8P}b-AppskZf->Q)cE(%MCH_1cKUW7yFJ5K$#9Sewax*u$}%%TOr*u{s+o<*BWwNfQjkGDve*bjM8Fnk6lMoXkftW$?g&VYq?w0rgxDd$Br3G{?gf;L>S9qyMp3he$#!ABY(&D<Lo6tuRs?wj0XSg!AJB89Lq*m8$|IUG-KmJTO%ouiB?X3nNn=r`~HS~Bn>4DTzmqh?K4#xlG~|L7BL2aWvf3zwZEH>Ju_s6v;B2<CO}*@(YNCKo)Wo5XC{(nW>3Q8?QU1B%%=`!zB$1HJeTSz0hcCNA*Vk>@j)iJE1g}b;L^Gh{nCh}5jg3u?SKH4hgf4Og^iHb)UA%9Jd#tS(k)z`u$%R7oM9X(fX=qWK&w~GD#xUR_`%^n(h)(w*j0b+*AQTxAB$vI8t8!D^&KEQ9W}w|dQJe1rOtof#m3fBMvcjW1&sKoVTVozrQXA)xdY%H8b|@wXZhP>))r`Z=#{ryf5jzq&)@p@F}vxv;v=AjDCo5h<dZHM-<F9uMdZH$DX>PqcueG}N`(U&-Kz3Vt0L36?$@DT2G9X9Z>2*Ua}w7lhdl<}#x~nu8aj2=Sa=p^ZDfTa*sM38mdqb+ahG5xMtB|}^Q9qpLbXjoYZ8o6jemDG)VL_C0N)~9{tV-}tF(0lP+|H~RX|B_`59s<Ur|CMrxTLrl8MKekPcR>C;k@sx%Ib8C(Nx3t&qr;W*yN{FB?FU%I?9vR`x<)tI^fx|BCY>%H$oYnfy_p{ydk7l`hge<}o%}!}UZtH2-<^H+Oja1T2dAKCa91OXa7WZ+90nAK5!p2+eB|g~e*()Qo@I1;^y)-j`EOn*B1|M4^x?EQpiU&0%FM`dlB%ieG!+KT4tFDHqejnZ>$CT~S67G5ehO`s}hLkMN4?n`F3C_iJm_>S77`aZj{#=&{vGd{*A47o3<f;T&Z!Q41%^I)^l^g&kvF6#f8OSv}r$hb1i%2WDr-pV7*aX)~?C_Dm1)kT9@Dq{UMQ^$f=RQa_>>ojQ<18_e%z^%AH_9Z~2)$sGyq&lDA)f|k2_DIxr(2RprRsu}x2`5DC9B46&kl}#_VzTPUYutG1t`?pcI`%VY#jo!1xD8SKmZ_Xf^`%eewY(~IVLYLDetr~TpcuqT52Tsv`OPSOgF6&n;eL(iyd46TxheXq&n|=;Li(%k$*a1x_cIbi3Stguav+^RLrkPh2>wf)xm?@)R$wL~P@BX533s6GNzj|vI8^odnNPAQ`8>54MIpZdmjFsH^RrF|imt4e8BlqoI!y|@_o>&)@#be(s@HTS~ymeMQN<gCoaszz&DEzvz3LreuQ+?LRK|Q{{BI0NNpxQ|h`N(`UU<J(GZsfY#`by7>Z@xfDBYDvZo)i#sC@}EC`xrZN`umc`Mk|iaBnDf=^D)4;*q5jLro~#}8kgAxtK%rVq9OVTi>7b-C;Jbj@u?Y)XDs&P4*hj>kx&HY4+PzMttwG5@fZwDinq3Y#T7<GJ!i;#yy+n%#7ECg`E<R2K}K-m5P#jfJy_K;yf)!f+UPT=LS}plO^l|vVNSY7xV~pTNR|H@4&}PGgIof?UOjL4(Hey?1zfRzKV$XvT}CZCT1_l#oowu-%cezapO8W?o1SO#mbFI+#U{TD2IgSXYa>?fRs!^8*e5^gj#Z0Um6rw6aqC=v;~yRrUGNMn(a|)Y742yazF5@(U>Cekw}g?8iVh<pt9L45hXoG7#E~r)>fj(?KEeh;`3E*EZOhjCIE!sU=#~Og>zw2+iN!NcVe&EsbaJ4f1_`DF=Z${rxU!qYO|5?9;L#9(&f(Or(L$LM{BROnYX@c=9tHP|$5S||Cd5h&r2?!Il2xDOATc1b-ukGy3b#1YN4^8HlR>jkL;%iyFV7pa^em=HN_v^Je(7n>!H6PF7>t&hF0}yUD|<j2)G};N`)m+O9&KPy3}#Wy96$#0o@RX|CG*T{$RZm_H+v0I?eM2uNKr^JXfJh8b4o)9Ski9y$%+lw<H$!Topv<}V62zOu=*CCSQBUYdOUwI_8k`=zC21O6E=OF>z&H-#ei25@@|=f5+Jx`Ejpa<rMqgo<mR?PfYqR7mo_0F=uT_eWH5PPtQ|;E8~I7BV_*4gre-cp86<(512YS<sFakvVnwWgJSmZy6N^PB-bDE$T-poh*&6IL@?6A-?ntSxwH^Ldy9&Rr2rY6@E7I@27C;#Yd)tPD{B8&Hn9;c9_7`6Lw+rqBxC4AM+?Rk7ySH;@8y4Cs;Tvnmhd9vW_dIu!;!P+^CmZn!LXRynr^fCUB~EG{fYr^9N2IUygMpomgG{37vvoHcYR`b86WJ)&5r1S2{WwPbefI9f#`j%tom`b@DCfWa7P89ii=m2ra#LnTMzvuA0D&d7@BRS3&NWbvKd^889Fs3l^#P~XzylU(69{Lfg6x6{A97{BU^F}CE3i$vU@!M9z-d8HWw!RF7$#6OZLwf=M{WmT>j0Y54~V%nQIqwk(6-16v?|1dx@`Y(wsI{IeL8W>l`1lbKicP+P+=Yx^ow8X-xXWU4b;+(0pvY#EJbgBW=ZFJte{9kHBW?-M*PTKlKnk)doc>&r<SOG<0ECqxi1pd;!OunOJ0jIgVeo-S5<bOZKVD4YMisiZPfA<;)@93g^lsyn-XNkkATMju*rJveqW>GzoQS|jT!d?#unjQosR7>nG+r5>8HFX4+$TdP_VM<_PJR3&)kzwo3%<fgY07codwGo1bupkrIH<=*`nphf;AriU<~PT7XQ|XZ)McsPR~curE;q52%cRA)8lG$Uo#&;RNnY=CHI!ZbgLT^JzbfG<N~EE$yVg*L!E1O{z}kGuk=n5F0b#-!%!&~Y!Nv<>Ug-*?yn#=zR;PBTziB4M6nxn($r3#@RL5MihX)TIkKE#XjBYLCZ!44){*ze#St9NX!zmkh2AP}%hm6o)ogl9G`Z2Rslw2|$A+H|zjCqnA?}>W7rTR&T8YLiaYqD@z&nh!RjO+yG!o14^yhx5)xIDSOjzbOfmwijidn3{3i4RLjuE7P#M}x^?SNKou7d$H*1Jpt%KQwx3FDkBVH=_-OU7s8_Xw6!iZ&7}su)drd(E=41Xlw^R>`YU{xHDq7}*&R9<6E`Ui>6lF7a8L{Xc$QQaId=)p(5d+|kJDAxrGtmW$Q3zm|h&zc(9?p0V;RrwIM)gn%gX*5HaF%@aWz2q4C`ZRNYbXq_FQ%vu=vo?!!N>6SHE5bC~Eq@e!2vQx<Zco_VM7<Ga=`1&&VdTXPA215qHf6+50`E|zq#0tb$jK}*u7Ux*jzIAfu^Gfv;4t3C`@@8C$NV{}sF?`{rGqM>u(1qjJrJks35t4?!{SG_-{DWtOs>G383)7Rv5B5Qif6qgOerNDcYpSp1(C}}Dt)|6hw`|_6{BI<(8wjFfN#Onctiy~hq*!KxM;*^0D3oueK~^btVsi7FyEq|HOG;wm_GV#kYW)B*Jnl$|M)9V=coadt;(-wj;w@$d%UpVflXg)MA;e&?N&5L_Q3=)OQ2+CdVoFvKLvq<~?2r1F-*?{(<14XzKUs^4?r#7IK^2~Ojjt1~`b8|u{Yur6$dzmZD13*o(vU;h#MDX;DTnOaoj0@*$dIZ8@Kdp4Sjs`vk6RN3RV~Iv?4wl4@6sso!M}G3_8jI)0ILWgKau6c#`v&QgIHtv_}G}SmFYkDFbN__(YN<qst4KBh-s1t*qAfOSR^0u1F0%^%54a8(3%E@1?r#pI8^4sn_Q;Td<gaYIcB<Ma7@bm_U`^s)Orap90pYrDGz>!JM#vl&38d2TC)vl>*7})C>QP}6KF(sgPjJ<-vY#gePnI#gY!bTRS%X!mnKCHcV>=QA^R%f8~p}Rg*hM>;B+U1x~%SEAYK$vvrKiDIxeranoEU`Ej#8pIVG(mpyl(j^`|4n5&Iw5(?L1TZ*#q6dfkyb<q{hOn@4rV=$#aX_hop^tl1V8T+9oefV4B)iGJxhSAp5><WP%~pjMT}V8E+(i23!gniQmj&sr7H!&Ow)0K+~CMWCl+^vLL&H%}mJz39UMLH@E!7?_J0p<O$)!xm!sheH_nR)oTmp_pg>PBa$++7t(g2KKR$o_8#jAfJ%bcT(uIppPfqs}`WFn!3ku`}yI%&YhjDVsOZtY!T<j-^<eIWZ(w`{g9MnH!60XjYnCW)RLq*Coks#geJ@B8<HFXoQggwvjrk(HrtvK2(O4udGX_%Q7w00L8bQV?}e+ATeRKbv=3g%+}*0rV=%I93gz_#2GlXTysDA_gd@JlaXkFa*#^yuySmUw60~f^r@wqZ<w0e?QxfPl2|2X9#DH0%)Vl`UI*?2`RKRs)^Wl|Hygi=tqzTYXS*MSFOv|2c4)6?~YI_#@>3xz7$JhYB{6h(Z>P)~}_6_;6q^j#m6O^qU?K?_DH<JoJCaEgpSH$%2*st;Kof;&rLvckAXwrnfw9yWWy9M{{aatZ_toTQNCN3}wrH<VW`f_k0td?^+Vh8#>wYLigt8+;|9L-r-p!u%Tn#dpRXscCAI`H<QK|HYhZ&Ho37Bleet=$I*)}yWk(qF?CY(^;K-X(_s!yThYafloK1>a%=oxnIJy6e&JV|@|f$%QfkE<nR~mO$D?h1L+bs#)*z9Vv^DyT<u27I2Er=7_=X7>Xv<CZzhkXo{2c`O#W{o53Nun&TrFrR5_(Df&K?lr7sz45CNhjf<YoL#40)1Z7#(<jzG%EiiQgled%Ls-ma_bRarJ6x5@9M%t^-+2I>fV9e#_S;rdfcG*H$)uoZ}IqGxw(stR4GyTF~CWkm~;^2?9>0fgnzTdHH=cox*IA3@$$n7KYyBPgSOpA9_e=xlwbOFYszZ=UBpj|7;2yRl8rK=|tZedeO`Whl`0S63<U8(F&HTOG8#uC=l`_%5(s46fHLdMch>E1p4=JWB!g<XdZgO4J97Z3T%)1vjwHj!T^Rk0k^3P#B4DdY}ad-AGkSd;yc$D^A#?E{ZY!aT9GJ%>j{3f%}cSyCxARCgDKu6qs~q(4va&7h5tmLf$&PqIm5kq9pK+Y^>hR-_b>IFH7S4rH8vjMGm2{5N5pK@<({OiYF^uhCC&{(%QTn=16ow41gY;K%#g{YDPM0h!Q+3MY3frRmK}2kklZ--9pA0b(LsgKff88r1uk%&*qI9tA!!RxH5-OSyOL(nF7jcFkp<#hhV2>;&QT8<|l2a~5C4EG!SZKb%>v4-W^5P_}X}9eGbfXzUUf>NBGhbhwP($RDoAmq%w8RR`sen<QluInmNU*$ix0T>7=#A4#|&BJu(8O#H$7)-ED#tA@4lCwwTk+LFFU0hHb}I~U@#Yv+Z47R_hZP$QEm1HdlxI{;j8cJr>!Rb$!t$B13p_=y&IbZPX^>XWUc9r)~!_f#{zb+UxSm%;J+m?(OGhw8mYTDIGb0cdYfA?JHa_t?o|MXP9JQyEO7WswPI-|qtaCLaL{oXGCO;DcI?ht$;gOVhT+vHQH8l}?9910UqH>XdRTK|ODQmU@~lkdZH269*WtSzmRg)U3lW8psv%u-P$G3RNB}uelBmYsRv=D@2dboToCZsl05B55!npv~>jWEEhdJG#;f~pTL~D!gLfhrmdn&Nln`Nu5rN?NN~3F%!qQ4kE?s#>U(u+HEztWI!G(jJvJ$tgmHuRU)H1PYn@W*L%ma|p1k@pPUT<2HlBc=Qf%Ov*h)ven3@m(xWgD3rQsJ@hgbCMFiQwx@9_D}Z6}QJ<j+}tQAOus_9_zfL>dMywsz1L`y375<~0l_fO2m!6iX8`zdfyKJu01p2i<n5O5-BGg><Py;67ZbM5|7t0J(<O%}$B+P|@Wbc5O8MA{E6dW4Q*Mov$Z^H_h3%#~+O*G#HZdRo7W*ujYN9-3~y5h5d3pXxxVoM`Hks3uNXr94ep){02@~zJ?5{j!oFF7lLEKrar>N5MF_)5o3-uHrHJXgh8W}$Xp!F`37~u?*;D22vZ=gv4{>l$=o+*)YgQ+<&mnL+){R-B9@ga=TBWYs6;AykC+yB68>@N?qkIZDJG>~`%<t=f4JR$6|?Y<6zFEGNMwU#s{nbg@P?*?wb!U6&eW8@UeI&^z?abL`$rvxkyKH{Za?0cpdDtf+xCh7vm7641&j-dc|yFO2*x>OiVFAo5A@7t^!dG>WNhHGjdx%4Zh&YSXp>N4wLPOTI_)BfHbN{W72UlT-9Q5;6Ok?T#Ol<T#FHdrRqUYmAVZfO)HwUS1n0Xs@P%gWJz^Ii3kzk*Zzr3HyMccAjESN}(K6ao$Ymu?{1vw?ePZY~k$`{cB^g7=Jgnfu*>yiT09g{^W~LXEK`<lN+32@qkCUIL%vfMImhbcelPKv;Rn+5;&wT1w2yyc9K!M(EipDcD7VKt_cxc+$qAcNgS1dM^1fbD4D3IVM_Ae|-xv1P%&+wUcXGQloO=z?9k4y#MHJ%-WQagIb<C4;@Ix`S9^S1A7BQv#J2d>~Z<>d789a`B^o)6O5xbkZ#DnTLVK*wCg<M+W$p9-U~bjy#RV<nS7FCZ+YjU_cLNNj>a2NrzSKVST(S0<xuJ{x>=La`59gY%E69yH`Qycxn%#GF0elhw+-chGbWQp;iVqQ6d%@$Gm-3DmGzx{Isctqow>WW@pi=G(zBSqLBg)>sXqB`fZ;r7Wvrq5%ddU^&>=mXe?W?DYaLI-O+$84je{cC6)OtI}+S$K#K){l*(<{tW8gQ({_}LrfaI>u4nY*&UlSVSz4c<Yv>yqj3awa7lt02HC2H%#>fxzFBCQ!7HzzTeV<=IKqE&T14@g=aH-w6`{7JY=SWclpW5c1(DYlwZA9c328Z8Ab~yF`<<Wm7Vm#VW4#61Z&}Q-*km3S0f`Ze^G`6pY-s0C2KUqU7Pm*6$zqjI9zzh*>A#ct#Bq?)!s^Hrf1j$z$eo`Y-|j%2Sr4CdVD@s8_!)cqRJ`^kLqfKS{5!tWhD{!qPy5Xj7Tocv;vc_6_RZ(qG4W9^(QSU<wJhEq4q(>{<XzstGJ4iGIwM>5fBIqgSxGvD`IAyDxI*dso|<S!JJa8;<MAske+xUmn>7WFKK099;iRQqC`T3_p3R*v6^79~Wl4AtNWMXlY!$)2Lysk9+z8<KIYAUF)^6#<<YlSI4mw+e(-=U3@{rh!zYU2^8c)k{Y;wo!(X{EcB!{Tv!*;tU7z5bsNt-nBqV72)RX(T@7W!!}8Ou8u*t@;r9N6uz)G!;xHS`Re4+B#Hp1t!(A44EN@jfG2^)HU3QHtHFlp-$<u`Xt)ut;IFp<ncc<XgZRXH^|W7c$(Ev=Wr&8zV&)1C0}(F~VjZtg+$#bVh}XpwEm8Xu%}l4qLXb*xK@-exM+#Nr)NXXW;Tz2*~pVPCvIx6z8w$0XH!%0T&p&n_y8ES2-Y_x~p973jnE{m+l}ZJ8Co1B{uR@Vve0bX(|Oo5pMwxMR;j5&UrL^t*?(=nrWvL!{zYSbuXv8(<Z*HO1VIysxK=v`a$~@uF-0?)nh}uh|whon6Fda_yjyVtF^1-wV&0YbtE;rC!cgZ#hZ12(?8rld7iczXGg~-`Z${Pl$;5ar8HPY)L>&Qp>ya96ynN25AE7&xIM9>7`e&h$*=WmhA0JL=t`U8i@{RQ?oT7P^ZZ&L3O&$5z1(}Ji|;mT%VKWRK&Q6V=;JlxcLe&(@Mq^LR>y^Nt;gkPV&^OpV@wTIgJ2gmBZ=wvkCa81WF(`NFWl%~IsVCkuYQN_U5jlJj72dU#F^A=$G6)BJYVG51iow}`u>WYUh$ouIokF;qP9`HF19NVe?c51)w3N*&6qY3uIl;UayRa8W#$_ViouFd7-;AclK)5a?Au16MknB(zsm;{F5xzmiQKp;p3!a5i~Nm=$KE8FKx9?G*9oq_i08F-6vKmw-@bs!(dCzqi&73vr84o>1doU650EGIy2rcQ4upZ4QdG%4JVX{MSiimC$%Kw9(Ymm;^a^ACB<^Pd2_k#faSu&$p`1ojNzi-H#S3XH7JOf${$Q&?0<9ECN<H^tbl`@?px~#Q<H7Oi2XimEumqYn)*sSX3^1(%z?x|7+s>Hr|A5_-%754Jj>LcS(5$h><x{hS`*S5rxtF@2dzo#pD?@EadOl-m5QK`jm6nif+1?6cuM7&F_WbjcVcsh9Sw>Hj<L2h+=|&X_w*5l$U5IN%<n?3UL2c}lsl@orlpm(FN3$)bw$1)CXkHt(sW6&yiV-?x*{6DMb9&fQvDp>4PMBW5E)&(^w@Q?o)vG;I=w)y+XwvnF<w;*3j$9C!NV5ij@TzT1@mLqj;m#|MOEkI~JFKe3BneA@8qRakqy~KsN@AoeBxu~_W!Dd8v{^#fL?eNl^1%U$cwo!yflDHKhY2<Pz$Fb0mRIjE0z{ynYcY=}wFm2XfQWdIO04kyltT$ea%vftv{5=8oZ_umeRzsf3fCWID@Ah^xKIPfrps^NJlWapLaAS=S^qEJkB=oyoH9e%uU=~2d`To&Tj_F)5T9q-aJ>a=AQ**$EVZI*+9_+_;_}>RTytrYwx;?ynRAHn-xi-sh&w+b%(=*NUN_%aB})t_g>+BQqhI3KoQf`j`u>7++lSIPWZmIP4(>037;lwn!+!iemTY5jUw~~7=4E6rSC`(r^O+hveRg!5k)yLaYlQm*JiINV_&wUyA)(TCz&g=K_WG&iU@(jxG+w+=%ycm2hujPIL^%u*?w{=vPY_cDFzSOs?~)!iTGp_BKKNfn!mI&mCJFDj%`qH$rw%1nh-N_^5xHxdD;FbuX;=GeBv77avu5q!kB=hDil)?%k7APSMiWSs95GQ8$@b#7wf4`2o_?V>m?8-<=Z;@dGF_LV0hV!j4h;ZCkg&fY*?2jHFgXg^xpg})`Gz5Q-c!98xK)HF^?)41?GKaR>a4oqI-UlfkAk9En(QtqzfV}We(wd)=wc~uVYAA;2?~h^j~@mvd3~ux_xuIHX1#A)E4pI!s_O`1NSgnqx3)p@IxXrAyr2uuctfNn^)EH`o)}?rH<A-pu2y|zuaht8&vd^bliw55VJHOiMNX5P9p;Ql*Zqc-G8?Ya6BVBJp93Aa0j@|4LlyQ)42sN`B`+m|bR2xH3&K%UX#9gJ83DDzGHU6_0`UzaDTzb$dBFJ?weAVJs<5yF>lLEtu0en24_v?6#&7BnOv8zn-n_3TP9fDzFS<JjP-G?sXhn6ZvHy&DP0cU1z$E$p_R=$=?_8;KT@G2`gNq6$eQS=>wyLX=!9_`9du?_QQ!BtSyOb(h03mpjARg(eYMw*L@FZ{t8~QQ^X63utd{JYOJPh2wyK)Ty3sybwXhiWE%*Z}3iTGJf?t8q7)ZAo;w=d>|Mgf9TFj`*Z6M5y3pe}%AGxiioDgTgPFpQjiKUDgNu_css2Xj#HvWa{A*1qDY=~vOiII~%`0)|lN`*kKgQ#6ZCxh&=n{4x3px=#1TVkSGC0|r}{Hl0B)k6DeXbNoEJM*9~HWE<HI=5eFw%?;K^aVP|KbfTs5r}U3NwTc5l{}k83Z52pguuu361No0RPJ8_KJgdc+&#sri8mmg&5W1s43Iaxvw9_}Wd#U2G<~3t|B~^p{HY)D~k+0Lbpp6$Egz&+t>;JoOF7>lc%V72|Wxm8ES5Hp!wapD$F0Xvs{hN>>i`WJe(v!G$DyC?}v4iJEb(lUBA%YnR(VHZSPn^Fhq_NE)ENMDOolfk=flb|06iATjc%D_S2?&}}bKWoR(@Y@<;7F@}F&Pa_05LDH0g)qME9`@GIa;Fo@(FJlkJmnCrgYG`CvOJJ&-*(#q+zOy$`p<Bl0e`9VDNDApXV|mA|DRBdR$vX0vbc+V?3)JS$%~F7aw5uN0$AUD(+iii|4^s^?u2d)7ivi-8<g#q4ASsV3c6AtnIN3q*wX1#QIY&{OAyPuQpNd;0NxKARNt)A;~=KG~%W~y{6%%01g?jDTECv+(<aFHO0A=uH9|sX!|YdyS7k+Azbu>)wrYQ;h@$+0XX#l$dcv&p2*=AlsjHCHPr(h@}`2(Ui2N%@$N<uZMzVC)bg&A0Ax(BHw)MnGygi3p}J4-w>7dDtKcO_vyJuXN4kUlA`{xgUz`~E4)>QWH-nOVZ1ak02qL_~{ItF3f+lXzI(ftlp*xkA;a8`Yn0B4^)(p@~0q~*eNZ{6rblYN7oV6Jh>ig*++^3}84|jeEkwBQPBb$U0fE;OLxha|l%!L;vBkh1)SG#9mT#@?{gLIx*Z8$MH*ZB?{_R*xgYV<;=f&aSi`lZ^#AF73XFUhe4bA$thXx4z`3gNeMiirH#?N+af=$uXy$mNsZ<ndRQ%0&Ks4s#$AjzO9TYpPg)r5nKcUWrxbP8I0l{DuSpqnmN-^VG}vLyo(!B{mbzUJp3y^IJ!%b)xQ7cOR?9f3c<QaP{u#gSF7A$UJ&?48MvK66QtZ$GwW^FiXJOz)}K*wek}!0+%(LYN6?itbrdV4CDlgim|fIpACQPugPYO9Prx!$pObXk|Z&)ayA$eQ2O%;|MQllAfQ3y5c)%z#3ruTs0kP0O;gLtWqxziAD$)J2z8>SX&r;qo8OO3bw#=&EwfQDW~z_qZqGqyK^8X<Z5KmW?Rqk3nJC9;f`Qm8H+A#}#L`*d1oR5mOO1Fq_IQIMOZJ2ZYkk+Q$>o+AB6Q4dg-nncvY0sqr-9^;qEwke6M`4+P4}@)rn9w@HgfIwr~?qEtR)T7))e!@D~1?IMVfFBn!>hS`j&$c>x0p@nv}Nu$TxWfbHc$$Xy0+D$Z_)JTW9^;wWQH@o7Bo)vlISJWXN^XP)rdXnD?=I>#|9tAVbpFC7}o{UFi)W0jjR|+3~UXE|O+*zJ<6>MnsUV)6FGj9X1Zr>K>u!&i`_rnq#xG(y)L-%=r!BP+99PPfe_Y4!>4GizCYhp-ZDYb1B5RB5nI(Mq@fBv(@a3?%@NNgXJ_XHnsS6D+_){K5opvtTZF~PADOp9`~!@3lLZ+*$mSUJs$PBQ2=p)<mEhu8A#F~8CG>*&6X7Jvit4q)UV6FGy*U+2*K;?i)xx!wIE+$**hF#nQw2oE9j%flZCOxCIkA1?%qEBo|wy+FOZMP>jh>M&u_;%YHlP^%UA}pF5lMm2l%S$zpo0EsIoi=ZoA|BF8>GwzjMEx!f``**ooK4$<OcK<UHKd-?`hNosQ^2Sk`&8C+=LBI}Y5itejQwNQ^8$dJl|G6sY8;WM>CQF$l2p+qj7d6UCYFA)P%kZu$0NL+Y4I8jP*d+)TC`=`3|ksQDiNo5)IRUPN=Rr6x!S4ID3Cg0PMuqKG<s>+l5116TyIT3C>u$XU16f40+3urk&i3oI~qx`hn6h8GctxU8ri*RS~;rOO8m@IlO-Qv-0X9q-LT*uU6M>4i#%siZ(<v#gn@d3qRu!X7Xcqid$gjVIr-I2hdPvbim3pYK3ep_G>>iZJ^I>JAUM{iC(a^^!hI*||zggMKFW)8mG4X?);`p8<TTA=EYh)sQ0Sx>@F(6QSB1)cbu%{DORy2kN@Xt)!hq7>rrfvo}XfqD4hh-M<Ls()u$WaCQb;?CG!td|^l!E4@qpoqG9ybE}^gpj97~o>hEhi%VlV&jr`Sw|o|Ii#X8lESOAr%Mdr{nzmM9{0dc`cKq73xgTA3J@jrd(2(H;CGr(376O9o1ngUp1<mLZ-UVyBKzCFswG7q9Jw86?_0pZ5CBCQu7z>Ct05~WEEHm*oXMu}K@T+H2Ru_4t3xf~`Zb`65W_PJqLgPuh!fFvSrPF+fft@F#9E6LOC-xJ8LcOmB9UY>KqAy4&zuNKtzzh<vuOh;sTO!~ZWnN<<x$B)YAv@IIIWZ?h$GxN+pl8T)=4`QB=dZw}jTML=(HFY=8wln)UEZv+2B(T1@WMRGp(X7Na(WnY^T?v9AD^>H+i?DAu+fncWV74LHB1%{HCP%~Gv3l!a^*mtuwa{xlMp$t9A<b$kIjtC&{y`btlVs80(3J3*OkN1i)HAW?9bErSO&3WJsW`R2x06;PeO<c*|^Daf%0hS1%pSqje|vpJHhJTA_}Eu*9xRg*>}KEf%_ETNiI-n!8lq;@pGkrof?boYE-R;{jIi`f@%Z9L?F%N!3FwESu0emhM;DQzi{1|^J&$-lzVxJH_4`G&(6>OJ4wgTS&3Fj8Te_$%_Y1CXxl>^aD>+f9F`hS`%S^512_Id$24qg4N%AZo%#Y$=aL+c?keL3#V-t#T3Hafp0OS7^{hK;DCB;EgAz<;62sf)!cY`eWn?S%Q9@c57Q1_Gdy|WITMCkTY0$M|nup5bkgg#M_oaXAoHWAhlt08+nEqg?Ptm{edG>&T=<UE2OxmPRz)aovfp$nY#%rx`9sFvu85`L{rO<m6k@W{DvwghA;yEKTb?Q}#tsJ;G|K%?-kt-XJI4tv_i=j3a;1sxoU81b!+Qk0}2Vdv#@Zo2zp+g57zyr!Guoj7fov?+XU=pmZ2RfwlO&5M|`Z-xyIx#b?*rhegeu;rqA0PWHY@enZ{{qxfVH;}`N`F${pFKnxI#CdNUA`i_LaD{#ASu8(20DaF6+g7w?5TnFDfvDa7ADzUGxTh8^z2Ql(JUSKG1=Ep>};A^%!!Gg-&3|?Mt5k^{%8RscvU?-F<-1xqw8i*50I{D7ckp1C0}*u-r6g-QJrXj7Sl(x>LwsBmi~k-P^^jdT?KhGQW3@(Ta+~<l%Q=z{ApU4i3wLovIFD4-KdJ@nR?ZVB;PLq0jvASN7;dQRemfOaak2q+^<-z1N7b?{9Ur9O4l;&*#|ryY)&F~ZIENuqKJ<N2y>4*xhDFa7shi|ll#tLE1Z+)I#ooWCHU&2eygIe;8NN5)tYa+6Eq);pOrV%AhED@5*nk@#2gBU_7s1UaZ(DR5wvZw&(jp<tvHK$wO+X+9z!Ku#8j%Z1nx_Y$FU*Dc>MBxdLL7btFM?LJ){TDWJ|kboi>0AdfdN9jH*rU#xw0tMDhse8HfwYQ$K8~pEB2Vc1boLN{sz)BVm=~L4_UWW?J5&dE>C%@_u5J)eujolJ2QXvXt|+GC8su9=z3k>(P+rG7=iEw#aCpcAuIK1m~N)Z7@swvCV0g{;vW~X#&q2&r4G-ae+Ou7vF4BOz-UJr1^Nf76x=q_1mNpE+Y$a1>dg}gu2Kk)jjUX4QBa%DztAUMr7vU4!8{=(-(n%D=X0!jWIjo;u`Byud^q#fvY?Gm&EtIPK(~Imo!$thMFedmq6yJ&T{Cp*POJm`M07u-@1;uR#1_DTFK#e%IDG8un1m@)mPe3%6Sq+5+#q^yRu=<a@{zkm}28(e+I^1gRHfZ8niHKS|$UJy4#C%*Zw3x5)DX#9DC&kIKHDnc@)(hGn%`JoIsJpRox58FH$~*cX<j^*Nbr(8+OZCY2|2_M#Q<ztfg9U8~y;kb&@z-vvo<yGrRNZ#j<M<nT(5fEQ|~w&9ZG1cP66tx28+ZC3&q^R0mGcZ=KbCl0T)f17dkiI9c@fXNL_?WXY@&`u`_m%s}US>|6%3zQd^u4uAE_pqe-3>w_q6)<HJ@a)Vi2G)bFRQ12@6lTc+9I)W_=YbYgccx}|#Lyn2WzoU=@x+MBCkqH@DSci?k*5Y-7-3*1ohk1M7?{!$XWAB_zhd?7li<V;s2MTv4RpRcY1)3wcff~F9Mdyz9nbo%r<1WVn{X`M0Wc!ZucnrqyGH`evdDGRP4#^Hd%0wUfApxxD*H}VT)am^Y{Iqt{uXW26!Is&Xh6#@>W_L85o}WmZxJTU^?RFWYlpy4hU*ss9cUp$3gmmb?`p^;u0$wYQ>S?meZV}0J<lNw#0<Hq8jyuVUg)m&w@OlemZ5y(iD|j?E`6U)37Y!tAML^M@pM$Np+3j*GJaGysv!KSbyTQfK&7_r$ZX0JxcYP4d^kk%1p1@G(9*%Z9he(kUR3b{;Za3=YBts50NhLh`+wJqH+JW=nF<W65t88p8LZTA=vhD|nUc>s9y0c%QwiRj><-2DGH%35qlIX0$TB+iS0lm#QU&YV2nNmihG`5IjI8*rM&&XkHw)+cJEpL;j@Q<mPFCLjH^Or*(ob;dGV2mJl^uq<nr8l8%;V(ZLRu=?xY|c*j<A(NEbv;!MqiLs<nItGurKsr#n9csSRNJy#VO<D9vjCC?pb$FfkSmVp5rxc(#_)#SvwhRn8*cLEax9yUt&K1*>G3O0weM|kIfviO%}|HBL~eY2N!KAqx#-k1SfU-Nmi<0S=&8rfc*lI3+u6z|D2+&X#%5HPujn^i{h3<BuNMB!R{x!)CNeW9NkWSR22-kA=pV=kYHdiXxw@ie500XG0iReH8Wt;`bNL*<p*Z_CRh7(_#E{~-gY!flrOJV|1gFEg&x2s0gF5S*v3%oWO?{QiP@gJ;72_Pkk!}`RloiFm5_7vzRT@6_c+ehnpdNVXP_?DGjsf&-gpRfsn~+73IKT_yo4N(^^pW+WfYCmd45?3@(wMfG+Tbc9)!oUwVZ)fdwb-t408AN2=!_0XWmzwLk7^<hz#4N23+<%`Wjbfpt8OzfvNm<=rs$zn&fddVNVZs9UQ^qqfoABqqk^ws|1WH9HDzKx;-<n%oM){NP@Vv#@Nzv!F95MnCJ7b;o6Nox<!;KR(vQ(DBN6gpYebxvd%_8+h&8M1Faor*>A*38-45Q-@tHzBnLTRK@q?AmYeuwUg;SCy;2(_p?3~4#8ta;Gi;X3!V7$GNqP7nf_Zf#-wqVN+V|{xfGoE&PQ0H&`JU|@iNtJG~WxRCS6(T<G?;)#NtDvMQp8hN3_YNJ&pW$vol6vW2AjMy>4UN0mh21R}s5C;!`~Mmmte5bEIuO3z4qz)B0A{1%L1zVfl<1_GqRTenA;!g8TA^6!teTi+a}ctcmljVqYOC(WMamsm=c~-a&y!xxeE#%JEat<8SoYxHAC15xB-zFEW+r-t4b<|3155fCu@L7c9C}p&FRotqN*Q;x5(kcEOf`e4)YY@e@l2&$(;eP_D6l@(&0pQ`n}2b4@iPKC^;RO7_o_9A7e@#otX5_uNvWb<a<7BBfxG6*Ts7Hjj2E<(e!rru9e<7VxCIh9nm|0jERg?V5?J1C6YR7-Dux%q!AkY;)^~Sqqr<bZBm3>)Hlb*c7lG=ZHrtDoe(AhyZ_|{_?W6*SyKyKzbVZK7uTr0xSs7b=5eG8sjx44I@k$OR&7}ygnFI9=g1@-L@zLaZLW2=^lZqD%ePdJEv}uUfv2u>~J(t|_PCg|~r3DSMW+$wVHuYkqs}WLcpSk!b%5oUI0TB8yv=E=$zIIVc83+WY#mS)Y;}9`)u#TkzTz@uTNj4oJ<kOX|C6&fk*a!#X<FEO5?Pu;@Jc0eh7;H3~%b!-B<y^daT~Z^2nNFU8ViT-2Lx;-+wm3tvbJkE@@Mv{&>Gu-E4F!A)H2QVif;0Dn6p+S@Zk{VgNlRYVklM`(1M}7~n1?;An^2q4TLRTFH4VGQR^Ru`Ha%qNBVhYzrpRQkz#TJNoozx`wE2<9xRzn*8-mM!W&O<J9|FGVTO`KIyha{h5hzqmB6@_WmaO0gpHiWYx@o;H^99DI2&%?KE5ifgP8!*IJCJc7#}TOKO>aSYo^BW8B1u|kJ)##RX}gp2eeTzUGn$1sGS^BBU}L__Y_J2_+Q5!NeYZz!=u&jHezyp_vz4$b^jBl-YaHKJ;p<cN=^?AO8&b%vj$fr+(?0gtgIt+#!mRUGm9l(E*TN^>q<%;hh{w2DRN>=p=g`aAe^w8%1mcr(B}6bLAS`NBJCZv(ZyX$L2VN=%PZXimHe`L1iP)wZi3(25In!z`4}>PRYVkUpn{;INNT8QpX@-PTQ8jI!;P85<_nHLtp@4MFHP-mDRCC`|s4LH@ca&o_FX2i{2UX<VLFi3n04ewLgT-|unSJ*iVp#`Ec|W<I!{+O-CR&!emZ<ZJ$Bs*c(n;*#y5giI*-3T~Q2(F-WRw{~2qWoHc}odDo9`9ObI;(e2Q5S(gbaO%KQ4Y~@n`d`rxhUEFB=;OwKu(ox4gcLhF_f)cu<x(@wSN#S&YHE&ZhP=(SCqpBVEG67wfrh6UYqLH^UBhAfywJbK9;4=Hg(3@`gH=hxc>0nsX!(g!+jn;_*0W0>eeh)2MbHD5N9#VTdRq?3XIqAotvHVvnD-HArQ2R?V8B00a}|9FYdr!+a~LYc4wzC<C@NA#sg{0f`fR28DHCW<Qet{n~^?>FSFNH|{NIc04`)O<>7YN96jokLN1I7Y~;bAkvS}+Z&H}%_Cwz0%m7XTo9HG5rxN9^j-QFYGHJ}0gluM?HX0<s49VtEK=}mDXGDR29FqLW=7WE9&1%bl<tuY?5=#!^to)cWxpO_-WJ|&Ba5d}tK7T(c2JbjZ?8cZJ-;BsPkrgcD)DzY=0tc+t~9n;%edhG&NVXvdb&%N43SwoOKX~H5l8)twX7f{(dBxLrlrC&6Q7N6-nEx^Jf=djTdw+)Frp#ag#Wp_+cdCZJY^+JQ={ET33Rt}gatm$0kd|jRAE1iM;N-Hu6r?SXY_P=jV2|k&1stM)KE(Cp!J1|f4!Z^Av~!BiY+bh{3;9>cEl+)M1l{fVje~CbAASdoe!m!!<vmbodl&c6Tg))ed@h5_vq-N_MbLjR``Kns7*{HxB1@<i5P51$v5a&LvbuiB!%yKigvQH-D(Q1IH=`n?`p<7www{C13P`lW&!ReA;kmVO14FDz4avyYCGXySV2|lX<8QNlc={Qj&#z2m#J(BLsfn!F;P;FWG>`OoUePn`T<!<=q{ga79vg_%yuB-Etk|y;JC5It@+yJg$V;cg!~om5CV8k(gD&B!^T2Tz)H{5s4dE+nA~qG2>1%-I<gAiRPk$%)&EFC7@cs2nK>iO6W~Kc2(HB6m}_OIrWvun<oVOdmXU^ZbeRQ{Y5(7zE!W<VDTpX!C^9~d4Zf=Hq(3jYah_NC#^x+B*BM)cDi#^j_w7ehJTlJHxq0*a^?kIJNOS(mXrFgW%ANoyez+4v!j>e^HDvb<?^F!O^ZqNK33eUOn?dS@uq56_KDNr5f3PXh--oh9Pi*rAZM=!@M~OH_w-O|r)&lqO+F5_r8`!6A+7JJM!W3stWCdt_MaMBmSCz2KvGn8vwsn1N04i>9lKfjnM})J_IKZ|?gX=N_+X_4JTpNM|7B0|fMfG3s4|bF2c?J<?%Wm=_C7n)Gds^aNf0Bn^3R4CTSp+dh(qtlzI3@?i=Io5m>;P>Qp#(BR285`7uWa)2n%dPd+1*a~LZS*H#a}EdL5fK*FV<Zhx7ppyotmKP0hQvBV!SF%(z`+1EA<TnnVPd(S80H}h0qPu$SU{s<3Gx%r^8riW7_a-v^GT$x7JD;#GTI@X;T-O9%h+!bPK$0-JHV@^%?7p@Ox=q#wt9K*!MyBOJ8adUb@CL5s1Z<rgHym4H5(`c`pbClJ8)>B-RR;8)Z4Uh?Z@d&=w5)o-iZecIL)!N~TxRQhJ1RX0Dk?H#dPycf^#sJI+TObtMqX?DrVv*WQ#FU>p$dr5p-&nW@o$XeEaOHot`UAIu8~Aos9nH|Yh-x>-SuS_B9Hyp~6*sw7krk)OKKv~rL=P%VhQKdj>88!v;Xl{E$|5<$HzDzz1$ncTHh^a-z37P~XqQ$L>-JI~TE_si<*Tk%0d%b}Z+GX%vttTlz>bhZMI1oik$%m}t64%b|*&Pjx_vR%7b`K8?ptCGl;PHlXLMjIvry$*aGvC&6ex7UCg;{IB(*lP%)ACgO=RwCE=qqOU}5WU-r_kF@AML|iKyGFNLOyNG!i7WfS=*c3%z>_*Kzcn5iv-II?Iv)BzcAZ7gzF=y`y@$Yj<5?ULg9qOObvUi!Fhg<<zIX0DRhA+;hvA`$WU2&Hud5&~K%mcHe}n(X({T#0Oh{#sis-xJbjscBi0nc%z_%T#II_C8?7THZ9bjeaz*2qDIvm@3`9a&>xt#mgGQLi=nzM3$E4Phr%C3X=JpD;aM?5xuNG_4!eW!c#9hzPu`<99&Y1nRGq>W<X`L+r>xROMDzO2DGDWMlCOT4KVky4*(%z2U)`+{38Umby=?5tTKN+m1dj9c+OONL1W!ACX5sx-;_8!D@X2c-^2RfH7BABv&R6CQtlwLWe&eY!xlk$_l1vAZ`0Wh1&q>fY+FvMhsg_^b1oCB|KVSr;QUN+1p;hZCHm-;JCvPcJOrR6rOnMmo~Da5kFtoTkn(w~f7pC^oxVkACH0*)Fd?Fx=X|i_$OICDh16r4@w>^D=)*b_@8eAj6*i6G||0*`6D1(qd>!m$Tv*mi3<@CP)7M-#Kon%X6h_vD&c{4tEiQN3l}A+VUmjz;~yO`+>9(1!P3miB4}BpMy$GQ<hE8`A;?GSr;xmHaf%!f}0tF*3+@?p#`5LubjfF_mqp7lehdQ2E!A6?Bi;P$x{bb&fT~DfSteT_4|nc5~cR@^7c=<G`yUB!;_jE>cpyVXc5EV*54mdDIdW#aH*VIMxdiJ#@a@zsaou1Di~ln1bSC?o<V1^A(H>GvX|osd{Sv2B=4&F;e^5#Ssa5B$N63Gg300<{~iI|YJVFgt!e<zrYT)p#RTG}2RBXtF$w`XEL_?()PfNeLYVZ4dsex`5EZd(xuCT{X1X&QSSF?s;I+An>2VbKUH4J&bCgoaFQvnyi;$Hg+zFqtO*LXLvj3KMW0(CMJmHKT=4D&*8hUnzdNn_N+tuushfKusX)$4N*?v(miWv3}WQNe4D4(;=plMe-__T|&G^^xm)D_LiDK9Wbh9v?f&XtigR~bn&0<U5G1l(LN(2*~lT<Nn~c(m|)Qq<dL_J6bmN49qlGkb4@+yl9W$5txT-Lu}J95vb0E)%5d;5?C0BmU@9Si-DBgpKyDT3?@wouE4w`Uxc7b)9orsfvF<7mDNbI|4fj&44IP^oR=K*dDQcU@7qpDKn<K?dw_hyY~_m(rqAI?JdM?1$mDZIc&%mSHsnF-FlHMf}`y}Js~iKqKXOxm&W_JhB81X#%Hr;#LRYa6#Ykv=lMO<mqon+;Cv_zy-5gV?oPW}k(SAu!uVBcz&qa!DkfW0y(bit>bB&lSOP9^6ekJbDlu21!2T>f<W%qc$np#lBkmzX*j>I4U|2Gs6FJP3$v-H3NX*J?PVP-Fs+e$y^QgNCCQDxLN5rLtAo|p94MSnoQQlP^TLqxK>@2{~vAej0t$q(#As(8(Sqg?aljpuJL(O?s<sc-5c?i*+%Nx7}lT*p6Mc{OtaQK8?17?7j&D{Ea)SjY)kuPQ33Z6hYu&=IBgZYKhCg~}kWYCTq0@50e(54ndv2z0eWI({f0cm_Z>NVj1yosRP-rdKFrYtABeve$N+&&HqqV+xg!BAh5I5Gsbf2gI|724dHq7FY7ANSJ(y+r7kuyFLaWb4a8H(Cz75lxul2G1>iZcs_Ua8*`~oe1}v`4{&?HCLV3_g{a7Gf?ve=w%S!N?Gh;jp=T|)ts{lOpJBQaOyCK!vcZP1R#{~O%n$!un%sqy4Mp_Q0uzl4?Ngx{D2SsdWU?nT+XXLvdu)BBB-OTd6W$y_9`<OOU$WfF3`)Zh(U}`wp$`Puf3<1-%xF{`)F4N;IKw#vz|W?4BtEZVv3o~80E_D1bd+*JSe7IvbGnigCU>+e(Q18WLT+ukydmYe%fFR+Ljb)+NjJ9Hg4fat-5TOgh}v1wSZjC00j8$>2bf}>qo+7tYzE4@=kCGQecvs9@>nI&|IT>=m*RNjO?RWQW=g^bY!Kq>d-~}y|A00-6b4>OVM=kXES<DQC(Ml*nl61K2P}x9#9BjFGaP<m+Qlime-AB013-OdijP&eY8zmex)<{Os2O@mBTDYcs<vBClPI%jJx>SUv~y{lGT%$a;!l$?~8Fz)rdC<(kMJ-Dn{NoIUKGr<u>tO@uKiHu{mZ#pVmTS2Y<B75RP)~<qJeEk?iTJTjBq3f%Wm=BV&#OGuREx`&p+r<RG7$!CPJ!60LzDX@F+9{IMlFu!&%NnRqQTr{eEQbdn!2^zw?K4ly4A!X!|LIh+Ofc?xvkefa5f-i8Z$%%zCLAj$cAZ`z?0<cZJczWcV05z_YKG+hMUpMeF3!>+TH8Ui{eS%k0@u+x!tY|nDwPaz~e&fbE=sn4RbHCx=gVxQ3(A*G<AM;utCdA;rC=hgDAY6yolhgh=`8m%2VSPA@qoWuq;8uf?5P@(1WDkE2#lk265wnf?@tzQ^SR?dRbLzb-R%%#>U@FB>N&PFSuD_PWaRAbB}xGz40a(qp2m3;$(@QuXgDzEof7qAhU@IPw`W3j5LDAV9#gTJLA20GAns&t~Oqk&IAe7$>);+DEMYNqV^BLT@c*ZoN{)cB?G8N&`#n*O7J0x8)a_^P^*6<{ixYS11=eo^vd5NCM2>SSHr6am%cB0xogFT+9Lcott0mP~MZlRsZr#iHa4(=9`w^-ld1KOa>cM^b|2=-ss@mGI?TpoyE77bF~-(*upxH#Ojn>L+OXx>qXoOeE(u2yWR!AU3b@H?ZBB-+ePxM~p?zDOoq>3UoNa`Nn{MeU7KhbLlJ9MO-F4!_-ER2<9QJ@Grg~rAqlnGT%TpvDSqF#M?#C-!(E7so<v{XKzrwc3gGLb|_QNcwg63>#M6JyBScp4}#6jMQq2Qb5wKM(5w(iU!1@B4h#kgrGs}1=tDU7={IAlOf*OcHZ!0WX0E#A>(|fUY-sfi3c$tUT#pAmw1rIAL@;FU--PwLO~j)pZ2g-9$o#*Cr30%LO)o=wWqP>I;B+A6vwYn}ip{Js!|?mArs~!~@v4yMl}0A%q3x4u_5Myk4Jcj?q#iWZ&hjsMxm8FMH(;8RN!*oukWB_Sr`*jFRMe_H6_#{SMs8YT^2xtFnxf<9+HFq53{I(b>D6BVitJPPJq0=qfEzmb`$+_z8&>(IrD}wS(vIy%cXV91=sdVXtuLyU!BgDK7fZZ-2ZkX?#t0>!Hlghv?x9`kld<gYF3BZ6^WZ*q@!a$!FkV*X=KwVW^7hux=M->fk^5PiYI|UuKn>rrImXPF05FXL?ARO_1&Q!%tRcE}b<D>EAo2QPp2Jx!xT+M6(X@Ei5XK2}3Iny>AoN3CbL%?Ti;8aam3OTslK8SmP{?-?ud>OQF{ENO{XBYR<R{`j;9BX#a;_6f0f;>a!3p0cBpgF^1^w;Z<$fv;S_DK~*$}MSZI9qN)&U>!j0Xj)#-ho&JqS6UM_@j`%2%_zSM-+0s^}0MG^z>sdcp)L{ItRt-gI$%fT}EfF^zV}NzQ8<$sG%OH#QN?fV^$Bd%o8en%0iTI!_Y}qLg%tzF$(BxtJ1Rn(wH4&5|F)Y+l0Vt=>V}PzFZdfTW75PBk&FSp{KJfO8pv+0cM~HS~^NvU%_~#N{rMuAXk|evho`H}f`nDy;IsgJ&UP)1<vN3$zbZ9Q4%8r#??z_imF|b7PlRmtLX`zBFBGj5gIuhajN<-YJKS`dr6L1%eo0UYWp}5EdR0Jn9|4y?+Du<fGQIGG-lpB`Gj&e!?+L=AQMIXX=8K!)OgV$O^@`T@6pMnS*~6Ie<Pl(8CHpLF#HWb*xOmzJTusSRaqv<@NtQ8x7wwdL0Z%4Os6L2B~Jb<1X@{6mj=&V9-jZP0dNBEH^RKdx(AKeJVWiJ_99^aOURzAp-0oL@b-A%taqW@kNDm;>Q?H8hwxX#2dsC-2gPEQj}DB4cJ&cK?Rvo9Mqn6r$Emg6=Dsz$fI;J%_qtXbt^>mTpO)f`Y2=t3bI(87KVarrsPNRcBYO-L*LVZI|lXJT<pwpZP5RI9`uLvukj;^14p9T;)IE-(qX&q&J-FlrV?fKI7#!vuPY9kndcVF^<|sVW~3Q@?!kjP|0iqI_b^V#N@iimaLst|7K6G(SWMKCf3rQJ=u^VNlBw#L{SS@P1GSiXHI}@&iXlCDrtEm56^UG%FjO1ng!6(dl3F$rgVoZqHV=HrV^iD)P~yLjX7npfw3O=;I1XrVLi1wY{G%0RR?=az%N9<1A4vYyGuD1*FQV<h5J#%QY&wj*INf<I4=~UQR!*-V&9=on1WY=aOpI8HP7N2B_z^drGT0cLZysF$8;vJ76fOBs(7!TnAwO-)p}M7Y0~HPd%iods*qOYgTRUqPdLvskSbj^Z^5Puc4)uspg~8IX4)i*z^z-+r_b4uag8#cRLd1Yh_`{--oE8?OHBiQhqm=in$Ge*|#c8ua&6~ovfrxSrzHz5Kl_0it1Qi8kwF=|7l92A%)`U-}RE-imSLv-R-O!!2e$B!_bUG9<i#UtK?XPXnkDqZ`fLOhC?YSAC;YGJ4ZVD|zNl`ljWUHCuA&X%vHX4;0^|x_20Oh{cjm5JyzR`{YU{QEPrRx);aLPbB3L_WdsEh*|4_6Ux3X$ST2zNvC+~P5rhE9j`Eqiy~9=9m6ENSf0Ko#mh1tQwmAl2KM!)wD`4IuSa1?h%HvoN0EhPMrGU(Q{4TF|Y0F>9+1#@FZq^ry5KMa8z7nivL{JPHM3>-Bn~!d&K!GeuFg?&?6;KA9*ZLQFQBBY$C{e5xrhW3vq(hb=w>DUWi2`x2dY(#7mN><-oC$pE8@_@7+Tg11m4uw2~G3ts_YZKH1D=9DS``5D2mtX?lBp1MCmKa%Qi%lor<Mmv!L{R0*uzGO1;kG&-%<a?Ve8sp>HeaNTh=j^lCR`qKa;Qbl%j3%5(_j;;MM0K60`@pPVV<JexU3v*#rG+Z_PxOJXAO2vl(_^C%HLs_;yVn!SbB3`YC)0m4{BPvY!$mLSo+1QuCY48Gu`|;kC{$ZTS`w2%X!V6YMtv~7Wy}i%EM>u|omo-Nw2e^P{Jw`?ZB7O35eAN)0CG5crCR)&dwtG)pI0l@j7P=s=CjX^|3TQU+$8&OzN589caUs(4cq59kP&(vt_SxR7~t2GYJZbcTRd>AXZ9G68PZMom6kPy&F5W7h$pOtMVHP_DK;o{NTC}`DWe8_f@%aKfx$_*pW%VjNTz!(!?wr^39`>`9EMxL8N!h$aOH$CWI-H{mo(JG@x7Y??asVx@|N>$2jD<M8%BGR5*%-W(7)Gm7Y|dERM?c2UZ~xZ>u`t>y?^5`mZ`3STBYJtZJ_Z?VgG<+0(Wdl*Sd4{GNWt;FnLT>>(%51Q%>q#otnd*uxIN4n(?bGC*W+24AmD0!b0evDv-7jPg<z;Y0s0W_kVT><_skhj5mIS_99_5ZllOR%{$gMag+AD+K@C!lZr_<jx(FQb~Pdzl>uVK##I0!g<@O#6H5sOW|Z>6k~#C7Z-|=lf0q-O9~f`79SIp9!(Usa!75Fo{A<BF5js*&?Fk0xKTa&<m3Bs7?}jF$+g@Qg0V}0$Ea(PrX=qW<yy(;#(M<1Yjo|#6i)9q++!jj|bM=@)_-$qFI%=v$X7^WFckyO7xQ&Wg`I@%GaTL1y<QAK{12C4W3=0_=1NAf)OADsmQ7}g5{rDr|?4Uq8oE+``u3+lciHoD3SmmJps*}LSbzTc86H+r|?B=KlePrrv%CUEAxg`!b#Jx>gl!?)l+P$I%q{RGWZBTbuKCpc3QoaQyTamB-N<(J&9;mYE2h##0l+z^KiNEf!j5do%4)}bEkWKGzL^LbZG`>fEkOQ<h4wN!Ri4rbN@tk$|59mJs;t4~hRKgrI-nDU%90xb_XWK|brO_f_G3geTC8@A<s*K!l8*3OliJ%>|lpSZLY_bD38U$Y0(A8o%AK)!G{#JTO*&y>jHF|H@r$TxN;gGQA&-pmz#}c&iMi@RXvE21E(mVMu%QtI8|9G?1pe7xD;5*<>4%u6XQ&|EQ5eHj>X8-aRZ!<`SAUcFd)6URfILx9IEEzr(3;QD-Uj^bw%ra%N$Ii1Co7-11<#dRPz8g4ullgU)vi8jmkT4f)PH>3XhsYT*jFR3r6a+uHNsCmtC<xl*s=hElWODgE$sqO8<A~fuV%RgIp==>RCe@xqBNbr+)N_gZyv^VTyqRc1^PYxHA6O`273EwD>bE=ev|}QjH(Lq}thcMF(MUm1frGI4Wop*X782{l2az`%$I-el348P6!8jh__7idyx0Fbm6V?lLAJBXtBUeSwB1)V<#LFfsIFhKu9@h`c4uE2BG)=98P&@S^r{<@%DKXpk8`OJ{1gGN#z}K39+?IRSx#e&iL&(bClWrUslqk}%!#{)HvXPA^?lnjuwiJcv8B<i<x}`SP^8i<vteCb+v;(>e#*x*pk-y*6TI0TrxeHm;6&8<hx&AI(dX6n;L+)+imFreVrwnn3T;T-&x}y<`V20mj(WbH6WMnoP)1OsSYi8z68{tf8#wDddh;vpySF5eZ0FwLii9hK2Kqt*g<Q;wMH>z5jBY5U@OD>T4Q=H-nSMlwBIRwK_M1^T|&CNV@Awb|paksMCuhn1%W0y`%i*F6BcmM7><GrfVkHZRHy;J<tdy#xGA*R<l<Kya2@>xeUKLa-!93%R15i;eMqhPfi=$kpEE78D0s|mVHJ<?$xl~svL*23a4YFa~H1doqE`XaJ=)^QJ3PJ|^N__q6^Q-Z}Bhebt?RGgnB-?*12%`92Qk2Dae>GPjP;tMtCKTjy&{#Wa}b{IP8==zwlVUb0nDc3HC$4)f`F@SZg!{s$>C04*1^R1|H`|~2IdI#_^(ex>ezwq|@*RG;kT-st$u|`soJUFvrrBN5%(Hqmc3VU4O-8zauHZ#wkT%m#8Og27^qv@PDlC|<tzNo2KwErz6Vlbreag>%kr^JHa`a|d#TI<<fG#zCmQZBXk9)juCGKO`tX%87rZ8)^l(*WO1Z;Vyd(7ld-xhZ?d;8Tr!P8$(^tGQ&i{7m{J+6J3Do|^N30CPnL9qF2WxBCj}P{KilDO6WrzdWz!9VK(L&!Cz_<~wFbHB=egg^+M6#EB`tMC~8?v5lyCX9&>M?|=Pax=#f%aD`y^UDS{G4DGgRNgcih&BZF9A5Ui-a|}Ih2<oi|QOrnm)t0EJZC%?-Gg~@+Ey_{yf(}i=Pc1mmoVOWtx*Y3(Lz|R~VDo}+Nn29hGH_oF;UFlDmh+r&{*>GQe)~ZNVrlj=cepFulI*p?tEqVQP_yb`&NvA;xl`c=^6I74#Wu3folad3jnH<tad)%<&p^wql@T0d;Hcl{l@7DiD<rj-bGxOOdivIw=waerZnZ6oJ*z|~C^V1Lv<{5<!p~8?FapP^uLN5Oe1hG6X#iI1HX?z@PN0VU%6bu}q?%RD447IU@`@?*K$TtfF;soTF+Y@s$#64dnS{N5tO_EQt7lm3u@xe5u$>8dCM5Rs?`-b8yyU!Dm4nYnxwYnI8xnw+%$@?Wu3sk6h`t>H(HM}>;aB?_eOKaeF1`Y+oU}8?ho+3(u5gbOQ=Cq7TROMDA8?JNN{5FqNAr12n8>vUcDo?i^&jRkDs&PJD}yJyHK%0l_fC1XVM>Jl{dLVH&}>asu65^RLDdkbE^+O;1LW8(VFO`<4im5w;{o98J!$JdBJ<Z>0U^3bm;abqm}}{QPpFV}LEygVD_!i<Sv#S@0eOTdFC<n?%uINh(`)`H<ddX7i}-EpmgCNR)@U(|d}0(KT9#hZ&re{BJxU-V?&u^y4XBe271i|PF6_E(G654ZmpeKTW2?th+y>sDDajYAI3V2%%|mU;g|)_9>~iV=Zo*4)j-=o4-EK<=e<MMUjRya^`O?fP*3tNGsxPNs9>lY1R18L$P>lSr{zdLqSAE7*(6)gHJO9p`Hwmp~EMMom+nUcXZX!sbjiV(X=@+Uat>0Ek$Jcn&2Xt#C;<ZdP-R4<3w<&KyEg5-jDo<c>AD6j>%b@PmVfBPm{N7vD(w!yo|HH`e<o<rmKHeG~YyE>*X9_S<L+UMXyqV`*qF>(!n(Mh_k+Z@4g4J@)my7NNX(J-r<jnQiCXzl)G6z>2$m+B;FHs;AAr)|N1tTqGkn|T~A+0*@n_EZ5<0LeOzrtdHvrbJ55Xw2luI+8bbnK;+h0?{NPxg<>t{r2*<BPbENrd(O=#<{5A6QH62FgoY_YiP<fv4z&y^51X=L+zy>zn&naj>HhmOF=q0yWi$$yGDyE5t>|>k2ayGt`<b1x|>~W9Y&(`#!9F{ixQ-)Gn2$1B9p=)skmvK{?BQqy8e+vs@=buoNle1ku%niAikMbEq0LrC{yx4C#fw6jHI(fh5bQLsxx^V3tx#?`ucypilKcKSC9gk5_dr-i%Qx`Qw_bAFKgqpeorjPskId3!76;7jL9E>6E8Q1d}V-S1;BXS_j+2L7==Tn**c@6BhTl#eWn)h*+M;6S2&X^=irQ&(4=tJZZ^H$n^QDeNo+9Dk4vP8q6$XojfToFf%)8I}<>ho0kxEbZ?;s__c^7LmALT0TLSQNFrukCaN+iIF{(K>$4XwCrToB(i-g_C6gy9<|v-357o-re<zGUnqZA69`x`uce;6;qr}Y40q;@>!OZs8)M5Mz0C;p7mIO+PVZEGeR_@3({k$8CT=0deA}o#%V3RVJBjt>aRh3puF?%r%(e8u2J5iN_E#IWkP7k!KvWvM0_~TWy!Ps2z+gqjWr=f>)fad2^&X+5%u9+;d?AArx_bpC`*Iou*S3VjJ{Zt59>HK$!xoreCibH2Fz(zX$<)((SC!e;(&ux3d<+bZgHKbMC#z9bUkPy(JfWt00!-s<#6y^vADyHS0G4!0f_y?IoAszu}Gf}j3;I>A%Y7mdh`tb`NbXrirhv$LKyhNww{p*n$NCZmt8scBgv;`1mQ2{tV3>{EWqb~Bp6_mf5h1R)8Tcp&$)il`x0{>`Pqi39$-YFT<rhKIe*XmX4NCnu!ieu>UL3SBKHG|7rhbAC=Y03ChKYme6ftq1teZkL*evK<onuc0TPPb9t=^`8dr#6dk;k)ciFsKDelZf3`wm&$6YUE$dS?ARpz_bpsr=9s&r@jg(YN^-drZBG8C99;e3%7ssV;*uu*v@|rB<#;jN|a(E?40~}5V_x+#jJlZPE<sip_u5^hfH%(AB=^)#eV4N=FxicY*ENOgh2=yoep_@AiS(=fKMEXYQi|mxrRO@XkV!Tl^ea|Ci=wRF`m8k7ASiw{i{GSL+H3#Z<rOZQnq8}E%UjE-y~lHt()|c&7NDM-*aeQB~|=|1E4xZ16U@kBnM`?v3Z-omwPYa7tSu>zDSsWpW8r`rEE9EX#iiFMEYMmo2!N(VH^}HcC-QDp6+(fFbWjPFcSk*=AaKb-oNOTm93m4;@*^pF|2;(0h^Q5u5Yf(4o_O`R!H$-1ZzS-Ze=;+UJ;D;#7d|Go6%d@h-Y9L*9=a9;X=!EcRPGM{xVE@$E2$`PiWc(Iuc2_9oCQ?B2N{SatoYXk5et^9^c*~2X}fHf0H$4Lk_~_f^wDb!A72JdZ@M^V=})2Z_kaWeDHnShX?=lRU=P69_#_=jy;h@sEtc9J>@k!pnEf=TvM1125V=cq)`Glqbe=2(0Qvn84Z4A`YV8Qwqg94Ed6+&x8eOPCZj#dM6yhw^$XEyK}^XD$~_7bBBWf-zjms*p7#~=4Hs=HsA%K8@gGI{R<#*4?OJKb>i$>Jq@@&}4V(q^oI=`u*8iuBTgTV7@${VGIO);O&m=g?iqcVPm*wK*EIOJN?h6!7h&Y8cb31@UW++I+bil~|>du<u=R{S*5Bs$q?nvN%j8kDAYyJs?j@A3B2#h|G_dV6+G|9-k?o*pZh-p>;JS_Z#r3UBcx56wi*k|bobv<3kw4-j&=(*qzO0<{@jMzhO(KFy0w7sTaTcy?KnDa@Y^$Jup;kNryDu4{44;{os55XCjaE<P5Au(mjC@~r_9?jK>Om8fU$@y>j96usB;z$)Gd?Yn3&ty8&JAm7qy-6?E0RS2W%DUNc%tqYy!@?%^3@Q!>Hv>0AmR#`dD!F2IMee90lx1oO?+NUiI;n=TnXaAPevo8TG>gqQkT<I<fau>)&x*=_%8{!Ot2t!<!b^o7HiHhu%BBv{x%-+CVT>+Xju5OeJsJi$aRs^e@SZmR5FJ4YM#8bKv{=7um;}2b!YlOhv3&O{V&rP3Q0hr_Ihr=Oe=Gk6RDqOmrad(g0%`Gb<3~WE0eAL!LV3EuHX^glC6(V8?N$8ZVeJ-RPku=Gi@IzolC0&K36<!iX$>Kte7ASNidCiJB7>}<3~TuPcR^GajLRJCHsoYdK5~&!W73z|^0Uzl{mZO;Oo}g_AD*CYtf78I#rg$6w;OtnG_zDWIgBITyG=F+ihI}FSFuo<FO7xg`mUqu*2Z-4K=%Mdh2yCpLhQKQ_Z_6WRB)@<aIUtGdzM8}4_sP7-$KloVytkCwGntcqBYmNV|gIv<NndJWP>U-C8?bj^|W7VEhT`(?X;G7-MP0Nd`9<nbFBu!P<WhiZk_-QETii?%?VT0c$5*HR_wFg#R9$)V%%c=#l(v%aMud5bS2!21tgsTKn7}-a8-Xigp&_Kr^IU+4oC3B8OEqmbl<=Oi3e<!KI8~c%2=aJMCdFAja#|WSr&lU2HQ!3>IJAd7PTl8$^IFjp+<6>7oqq!Z9q|4Zu~ci%!+okP!3e|VDNrg?)TavHtG(}3r6n?asN3trj(MAd43A{e`+g-@07J$U2&WGdH)ZjVmw9vLkh%iZBL!k!V^%LAhQzK-Ih63ObPDzJ~5Ur_Y?Rj_!QHSJV_d%2_$7ojU&up(q>jfr9)!p5hAWuyw$<svnth3l4i)AF7c7n!?KkGR|~M8QqWZoN<5Du*b?7qyW4TZ5aY}x7$JKXbpW9^#E|U(IxVDz!-VzuuE43GcGM2gw$Y0IfAGzff&ye``UwV<iR4x@^ptmwn@xMc>ADb%(n&Yz_qFgEDEHS%s*a_{-S!`P%v-O&YxplP1j;PJpuEU!PZ?a{8RF~Xbkl~vM8v3A!R@y<Oh{1B)<e^3^JGf>a3aT~D)@$fhCpLrE@|tR9|{?zA}N@ZT~bAjPnD~#A9>zyRW=xDCSp0H5Zo<*pQv@90DcdMP?tsM8Q)n>wr(kM7;-u6yt`%U9L(y1TSC88>72v>!$s`AU=euI7bhyUh^Ld?BF)3<-ee+Y44mxlHaU=4BS*5DYb>=R&HzX=bKexYd>}4`EOxoq9FT^;(Eb)>Sjm9iCdnGQA!DIdx4OLaNP5#26nkKgtA*vNA#EFysH&Mm*hTX|(N^J-pd<~2?}}cUwBgSY=)wFSmngBvFdf2X3A2)E_WfKycLY|k6mHddt@3vjQNWVl8uS_p8l!9J>U%SJx^`d!2GPHDhC4R}TpV1e>63P2F#le6E0@MW<zM9Y8ve|fb5&R>aIyC4?+w6(jr5`st>QFgqelf79AB}@!kwM6P&#<KXN--bM~1Wf%7!%pKJsIA<pV$PYfd}5)b5&tWuJmw<5RP@FkB4vbZneo31lmUX&2zl2#VZ<sLxN`xm@GxizQEmWX}mH>B?Nvd|odQ0JR;+wjddB!QX-MnobTo#D^stI*ZxH>A%PR?HiA0-6D4rh9jqqmKzw2Dig0XN(zRE69{5%CWt^NC@aZ;>je=6-LpY&<l783f9F+aC(;`vGHLvJ+VpV>$B#<U#Gp`bcZ5bbf>q~fE*y+nB&h`l8*dpDyM11Za!tM%*Y{`0RI}H@O4V-OY>iGP?Po4lqK5qdE^xp+i?7o0HE{)AI`iCQn^nQBX;Zi#9QPv7($#%TlDE(dNqZ2?k;G@$czIgJt6-1?6_B$wmu}D1XRL;jC!{^97w|+dX3OThh5IcSbmU2fHme4g=KOv9p6pq8I847FrTRI=ZOW(CSFO2CRu4?Jyqywmz_4<ydwp^I*#&WTgyDpUytR3w8u62`M2xt)VzmrCZ1(v=nE}zV=C-HrV~Of~4`qGPKet9SLfLXa;>ZEJFu~B#h}`dvKUlaSO^GwpP&}3~@{EG^PDNR2NYpWb%K4+$m3N*(&pysm|2_*pa#zZ#0$p;bq0(RcN29JiAJ)jm38t%;jc-<pK41n1dDzJke?$%y)1(MtRQMsn?V`bIdFdS^tXdN|Edc*d@Yidk4-x7GjA=EH!8yc09rDE8hdoK~+3WsApvcvxXNXCcqobpTrj&#X)cRG*EE_LxUNUe^1)zJxKcZf9w0EdSr#O5d<c4HEpDuk_O5saGN&cm%GG}E6b~t7q=OD%d(mYr5KnI!-ZDTQczdP(_{I@xXFep}gc?OpW{Qe<b8<)^6N7m8NJWcu6OgYJQLq+69x>Wm_&<7p3t~yi;s{A=fwKm-N!1IzYQ~B%qE}dnOnFGFLj8~d(Go03y+JY8!n!)1w@n?jXY{{RI45|Gy^N57#)jH)gnDC)%>)bpSr*G;~<F)?=&?$)O1-Qnj<G}3)(D_3BT^*PDp=SL<%sB(CFsE5g0YZgie+w317&y-#4_N-NHUIbV>VnKISm|txpgV)(p06OD;j--UsH<b7FUFaG1b(-uU|Zd!k^sVHWBD^(zC0#F=$TKJl$~{k^y<*e*=qG&jx~@4=LpR+){gUg8)FcO*g%;W+g$^(CN`LlO8H9Bv{dV4sw~Us3x_Hm;#uty^P(WySbp)%-O3*9zvErh_#SZe0l=@F_H*Q2qk*A6(Iz({i?+5fqR&4Y|C9czs2%QhL|h8o!=88|Z=_Od#sp##?#$IM!`JGAyW97GNPM!Mk~vYAVLP~Y6W6qEU6GN?Fb>Q%h0Yv=`u#U&KYcg{$=I}ON(7iTyl5qx&>3f{N5`Q-AA(}1ETYXI{zC<9#-fI;?wh5+8>mwX-CUf-7^OmVXVQ4+NE=r_Ifx+#VM&OZD<Q=3TMu@{-dvP9ZOL9X3*5$ac2?M{V!at`0L2FbEJEQCW@H;c^`rUd8S{w0s(QUh<WNVJg)u8pH@Y(Mk`~au{h;|*mcGdy#pK{@8+U-O(MB}t72)V01@wjl>@EDsH|nwP+>x{Pjt{ogEiUR}p!9jZA*Rx+qHxnSsO4(@gA2ih?~C)+^+_A3n!Lga!A)YL`#pu_lbq($;9sw=Y@S{;diMk{k*PHvc5>U@Fi^RSK|pVq&$E|cDEwF%#loc_TQv^mFXIZIu6|7$U(jj0K_cg_QqK0hM;Bw0#OF=lbtdk>Dk?hM(DhyA300`c#=VpHO7SW35wKTxkcKt6ht^J0=06+rf{zEVr$jD9h`00MjPEs_?N9a*jUWZ`CJt!wjwe2{V&6Pe#_{8u;$>a<!rW*JX2CD5iz2mfL%Reoi_yX<(Y8=+H*azk32E%|rQeRz>a@(8BNxHTEVE;+BYJNR!nQC&Sf$=-g&~Ri@I!OR`4$8C^z2U~K&9pu@o%#MIiq*1c-Jlw4;Whg>MFb)cLx}VaV-~H>9l^CiDJBzOWd$quHOE08ITyj=AW4Dzg-@z7{U1O$Lib~vWH{5y#``kdmZ;k65)|h57gfF$tO;s_1^;QN};dGfSPnxv{Q~ZMkjtA(eT^lI=i)1m2snRC8912a>o2J+K=}pL}YbP9^OuF6o{j*ooM7cL%)v%^@M(n7d^oVt-w+*Mdg`8&=$nv=T4DiGG5u*e1{5^(mG7L;Fe%?@@ukuj1}>{Fcj28jwRr`d>Whn4zXINZ!K>49NO%?5n(Zjb>!CbL{OgWPV4hk9#8baDa0$=wP<s|?63)il~8(a@VS{R$nZ?Bb!XSOHf~)ld1pfpo*HW_6yd53R6Dq&onEh=@xhvGnaB(W01XmxTKpeUCV8=4tQrTM`T=bgSy8E+$Rr|&h=|zYhambqa8lvlKiRI3j0YOV=nCOgPe%wy6v)nF0w_S4#S7V7%Yq}<_bF62a^ROm!vU*m@tNHh!wYahZi@r)Rcec|d$#Aho$=(YS<*>r@N1i+L{?<B|B^~l1?49fWHA4GIZaEwitD%8(1kabZ)m6a4~%=u8(dsnzlDaPOCwLhN>}|=#xq#lg-ev1X<dXM$9zaQLqL)0ETS3;D3Kpy>68z;@;K==syK;zXM8X4>@1F<RPh~0PSlpHwMSmrksEAMh0KMzEulv03%0__LUXNO*X)_$)QqnE@oL=Rx1YSUjBbC_hzu?WUPg-q%dv(h=!<EpCkjb$#iOMGzmIkIzX`i1d6|#ly8AL+m%W09d@At!(5-Tq0w(cd&r?iPIVaK-&aN5r%fyr~R<F%~#{ed2(Lof|mCaTwuo}c-U^QJ&{^wK9i+duB5&8&J;7ZC}dB&vg#T|iFZKKpH7)P^B>0w5B$dQQf+FN8beJYXTgHI3$3cZDFkV)TzJy*?ArTe%rj<mLEt|=d+f9iU+9$t`#V^A|ju)vc$y_qjVLEcT*p;%tIAp#ey`h=%;E$kBMx6#{IDiy5#y?;QD+`(f%JA^`!vdSs(p0Xjv+l)ol$ZinA^-XA$wqLMxpHgf_KC|S#wT9Tx2gVG4o94O}sa<Z?V&>A}yM&!e8e?ao@b3lt4!o*Ay-?&Rk?01Ik>MOf6;-kUy(ZTIj<gog5r$VjEE_b0sC$|OvbQc~m(tqth5H4V`eUJlJdEMomBP9Kq45@p$+mG^)p#rclH47)r=blJ6#DR6z#pVVs;C;UD_;UgB^juDF@W|7)8+WFd?4i96UIwXJdzIH8Z3~4Eb)f}iS6HYGW2AcAbG6CC@|-JebIkx+PdOSPg=o+HjClegq+mjr6Mud*KHUPgMVj`BY`A01GS>xO!84dUsEW><VL;;fLYeU4pC}hs~o86s*2!Ewm5yK27qS2sAbdW5&;$gR(|6#QXu?BRF=~B?`j#XoHu$n8bqox5ln87|Ga(Q46moZ*7TqlWvg%2itq!LZ{r)&@-`5XOF{MT2y<?#udDk%vq}mSoWzXqRwl)i@}gU53~vU=s4tlSqMB^$q>m2-3kR~dygrBZ{AF{;WI4YK{2uMzp=W{SQ8)npY0F7b7PJ@+RTHw4*q3zrk8`Q4>|H>e1J5pAg>I&SqP3D*I1KEifhQ|9g5lazuUK9ZKnMPTo0JzID^0^v7<pX!*#9LVCz?>@?OXN}WX5KDxQ;eNT}XjA3Lg7iJ<ocBqA$l~=y#{nRZ;xuzh9S}PNS>V@1&$OwlL}6z>?UVX)jT@#bh><b_c<Hp^L*d)~GJoKZ+Z!{pXSs`A46=95Rm4<QKgqvUXLXA*D}CrE=iwfDr2}y*z!MK#?2c12_iQsi%16lNRT+T0SEch1g7|6QjS&roM*8v)<c12)(21Ojl}3ec`;9qxExstYLY_Ml_v%=glBK4iy6Z!gTXxUBv6nSVk$Gx1-N8)0MI8%wQCard=yn`aepeCXWdfP`5-XoS6P$Ma{&iWpMaG?jRemSO~%n;APx>#ktuKeY#{KRs3w)>v{T4g%#N$<s<ZH%mC%k6|Qpi1+RF&gSD^ILeLKHTx<ZMdcj~$W6e*=8`$KN$5+5qzHqCBsR;8J+&uOc0u4)hL@%uIpBqO#b<8f%tRa?2qBDGG$fEHffItJu^e3;0+R|B6cpsjwh<6H-!XO|Yne~)?G!r%wldn!13~mg$TXcMnJCIuuF-g8@_Uq=&aukfeGBPR5ynCY>KTVX$z@EIl=vg<xYPA4tN@LXOnqc49Y5s|Rs(uV&KzB9$(}2jvUP{-L&X(}i_)i2rI{~cEDk)N=VD-8lfxzYVTcBDDb)A!uTvs-3m-g^+e0{{~kP0Ue18ZaVwgC#32nPmrO+*Nc9B>ywqvwR2HZ)D;T_zwUK5gSKK&(gYcJMJ?_R#~{@FjP=ynrIwit1VFKpcbGt2f`2V2qm>bv*bJYt6TW&vt0grypuq3#F}-Cf2xuRtZWAqA&R@CQ0%NQ2DZ<GR(8wPUbgZG#;E|mHM%VKGF=xEzv{7x8=`9YJbPQlYyt6J|QNoPvI#%Z;R9YS)t;TDFaiCDhkp>>Pkzw3`PWUDVG;KV-D}$QsZ@4#4-<}{=)<#zITTO;$oo<X1N?C81op~+fi%tvenApJ?7M34K@ud4zE}fW;T(cpbo9h3%{ob{=>h?3!j(v!`Y;`B7&-pGDO|u5gvm5O8g>+I4z$%^UEIL5Nq1@(-MP3tA?@5WN8UUcG8z&V;w@N`itTy$20QrfpZ{D_kN-4oSneBCdaIf3qP<X_?E?^)$RVbMJ~=UFA+ApQ5YZKpJo~@IZ<_QU828SGrTCFKQ#xbezwpd%13j0{CHi#gqIV=jW6lJ+4-e_k5(e5dV<M8mLY^Wu*Z@t;B$u~(A0)UcPcTr_$M=hUuAAO5q4~gB~ui(_647}Jz1eDuLI)pT<wZyl))k`^5EdjG9`K3pHeHdEyG@nDu=5>w}%>*hwUSoP_vYp3rpl{sw>V*QBk$oZE2W6lA~QFm}W05H|$KlX#byfk6yu8)ZS+I5GT8!g}t~@-mm-+m6HfZWd;;jdFmPlznOioZaEKEU-R=dl=)svUv7UrDWKRJtPfm_$)41$TXVM`ng^PfOYw(vc76=Dv{gC3bR_pZ-inmtQ7s^3JBmKL9%|s$L6tSp=<Rd~2AW2XlEr!Me4n!GFl->}6Z%#b38dqk-7CLUlFsTK%~@ab!Q`6ht&%G-AAVYOpTF<Xa-lU^b)|)G=3`5)85VL2wQ%HKz<LYBL-lsRZwmuJ<%B+Pt6@I&Vnn7b_`c34M{w{*&MpQYg9c$Shsa(@Kml`%&j%vz8C%@K_3f@ISM#o%b%wFAwXO|4P}jP?8q~U}W9Dp+BDn>H#&5^A+!P>q{{g}Ze;C{1U|?{<mki1>O(1V1ONj@DE^dyPvbOZJei?RU6KubW!?+T?P(whn=gDMN)R535J05eoQ&d9(_zx}~258M8NeIeKsJ-?KZYH1zt-YCE%cxg&I?#98Sp*4VL|R=aJknN@?D;(oGT?`+N1FnB6@H3QiQK-gS@0-Ov+SF|EE@k@()Cil&lbt(w3K+0yK-`1u~iF;>8&~Jax^YJqVISsz-e~ezE1qbIB}@0+}(}TKG}Jy#*{xs!O`lkR~&{gP9QgvtFD}zrXWiH-^UZF$$iJuYq!NWaZ^9Xyo)X9<|5~=3L6+c;AHW)GvZwFCh!_4N&T|BPc&gnlVBz0Db1pkEP@3K?zQ@d+=Nqtpl(vk!)-|&VcS4AgMA_yMhk0472(M6_k`d#f+KqPgnss`OJ!Rtm*a1G-@Zkcr#N5$(L?8r3go2QcM1g7UpAyQp`WQ{3{{IndGsy7M1H>$Mr6c)bD87+8UG_Sq(r9qOzuUm6}|bCFz3*rF>!PwSlh3$eNBRq^H{hd%p{TUfGh8M1D7^X?=REkfZ`$$7^il!jiMNO*zNb&%f06543NGxu$;f_=~Wy%!DeKOA^AkVGU@a1D!b@NqCIiO)tN`0t}zghuO%-M@l|{<A1@33R3W{iWwKyQVa6z>>U~pR09@q>u&}3_Vuok1Za2`47o5;VaVDG(mxn}pK*`H%6#^H`?S(~z$|{YqDoUB^xYF6wFpmlLG6#p<sgr*PWnI>sBm;PxLovs(1~G@R5|W;@_(?^VfmCdYA$2A@E$Y6<t|U@JVGuR+^u3b>I5(<`?@F%gR|m;^*>oh%yL6`?v{1B@CGItNL7Mlcg2yJ`R6U*Q78HCwn12W7J!RA~L=b(D(q~8QLdw+ZA3>?{kW@3sNI(>Trf{Q!VAiVP$JHBMXy|5}_!t1Vk0l)1#O)AJ0B-=+x*11M(P->0DbcKmLo{fR`WbqbWS7UIIv0$LH*_;FeOHb(5nUueuc0bI<OjeK5f~A*{$1Srae{*H^*YVEcT57T5dxPBeDGU@Fs$Zy@nm(1f%pqr6&Bagj5)%SGxX81e3L(Mos@-_ZF!=!o{PT@CtP1ACMW~bV}~1QeaXFTkfMw==Ljz&<6&d+jB-_LKPsCn?yGc~R(OP)Eudk?mQ63@3CkO46;-no?dKa}?%p3KgY5y;!jtLS=vq0M@A+Px_zFTEshL=K0J@^v8-$P=Ufo`iJp;;&BL>HRuCCIZWn{8n2UsZSn)nwb^1z(hdv=uN!1(MQh6{?qW;#Q44EHaL$_V=5Ex!`zmk%WoTAby=MiWx$*fOmkell_fD%V5mwW?+JNiUQaD2IpBs{mr~D(sM>Y%(Yh@oLD@P0zTIXA*|?(6XF0yo&w&zC;l5A?)PRI+kJy9a$yJ5XjPqLgX(jy597gT%X@7SHb+N>UQub$<Vm!`q_!w^>M*xMTSniTM`E$8Koc302e<hTLR$;mitVQx6KCzSk|;>*_y}<?n3#nQ&V-_Rs-L+_{39@UhNlvtHF88bEq_1J!RjCs0)u?YqEWADa3pc7f)?X)Kspb39in<2Q>&f{7%|e=l3bmA6P#{uso?lUC80si8G^;S|^%8rM=I)Ij?82K3GRsvJ<BuCIX%L{3m5+Vv*-4VOt7402;7_e<3plDJaY$)LJ-k<=_0sHP6+ETjTA%+}u=<VnLE4>Z4PLkWLcB-{}vX=e&BaS3v>caZ_TJ*%18S-P1<C7s@G<7ClKw?}(udtqM5C;Nzzvsi2%cdU6`-Qo%**=tQ9j+<n>Cze2!eYCOjd#(~aLptx#_;g*ej{sSbTrsg6~Tm!x+DmUhET(73pc?XCzw!e|8j672A^n(eQv}gFO1)EqC!j*Z?wIKwMk@01EX`5=98+BpWjRYC04G8AV4%FC%WEac(@><&pN2sBOjp?N@ql-WDDmoqk$FB(F;GlB+C$4w7P*$f5<#7HHi60?q0S=5^Lo(qX3kNbh&v8fP7vfri>P|Pa#)n@0%LONg_P14W2qz2eM?fb-0J<noR9!y6Dw>7;ILN%KxEU4Kz@l`bg8y*6mrtes(_@?w@p-dh0`?5$)@hWeTgCqmx-ayIb2<|)_Wpnm6rgQ9nL)=s!=CzV_$Nx`V_;V28J%1U<0qg5`LEqIlx*zsRMC6@AwU=0LWP13qIzxaaUa0P<aB!i>DR7ilV>%S)sV4rD_6o_4m2!J^7AidlGnlxOntam(;`8>pp^30)I|To6w|C{Oh^a|@vRhWqfGMd)pdV-Z}E=BRs%ev-0^1WJmYAh3YT)9A5#iaAW=IUg30K)+Pq?@2B$|RVtHoBPjF?&YOO<HZWc<8mlF$YZ*8Fq@(GA*&gnS;_=*KI+}PDZQZ((8;t0N3dJ%5*ARvMxrZ0i7pbB;=BUDAYy}-Tpn3ROEPqQI*Li>^NyY9+ltOQb6Qsi8r=`y~TB6WTnBguMZWtB9lprcU7`Nmv};!8PtS3M_*jNpt!7$6wb&Rx#+w0cvZ)V`;ga}W?poNt+B+)eP9%A_<@p24yX%cy~aJrByVvU@i_-&3|^KJe)s=6i`(yl~kElOk0H{1K#hS$pkM@l4H9n{2%*T@6xR|J{1AGmT2U*uD*Jfdn|MJA9R%bIozh7O9^YQVM}VqN9d*Z-*-Ks`1CU%J-045k+gTi9)kg3`N?(QB`+|K0jg}5Eca7wqMMe(`N{3`qUj_Q++;*F{ls}I;_Y1+`fwcnuzGd(?;zJgpkPVx5R1n`ehWmB%!=F^YgSO?;^CT;>fn5g4_!qstzv6S9XV99Tt#At|KaG58iN`!yMrnOdBYZY$FGJjdy^u1w52Kc$w-%hZ}BA(k@3U1eD4&)aZ`PqSrvwzjzV7ngB_nbJFQk6y;Yg`D=hL%Ry_qz;4B5a%TlAXUp$8tbxG5<zL1pJWf<nQeD4O>=g%hQ_hE@>xb;P`aO5nj_<$|rP)~*^7BI|OId)f+oPHn*@SoKMqNn_asECPa7UCuSw@|5E6(iUtj%t(8*N}#vmUII9d~0li65<CA7^AWukDkF<0X@L4-b3=E^G<uFU}aBJ@LdKP}jeG80B1lnSD#6%Fk$m=U%Y-ryKfkf(mfp9}vJ-X$<o{+PDFd8Rt%xtJDTIotBWJKxmbq_#q+P49!Sg0&7Sa{C#u0BZks;2tbhO$td7|2WZ6Bv=VND?7u>JRNeIlZx{X5mk-!6(lO!Ub==R+W3Z(XD4579^eMuOe#lVNdp|#V{4ANO-9s|^fhSqw=ANq$EKP43;N)9(10k$mCZjvC^zR`}OA*IYDzmG0${iy(?^UT2PL@?U(>Yg&rfr*}7l1e%_2Rvn+tHrD(~m?GBl~7C<eZ6y&bdJnE=Uc2+~X)&2ju3U?W=Q<C(p6d$g{<mRwYD|9*t~byRd?bj@-tPD<w}DHs7wd0MhkvwT438?i^fK?2G_W6M<4bbd>-eVov0_Q3_)Joqr%KuCxBNGq~?_vml0R$bpk&7)>1e{=n7hiG5v!Dbb^*h@6OGaNgi7<^t70lTB<1fSM)d-3O3#)Z*#Y@BF_vxoq9)QJW-D+U#MBtm~#3ZFVWV^n#8>*G0f*LL9S+wYFQrj1Oh}T6n--L4ZA2jc$C3CVv%LCmZ%lFlb;d%n3kiWJisbL>&)mgK}Up;@H+K_Y>w3{`E_FbK|aNCQp@v?mha0Esg$M(+y=fnUxpt-^<^(8i>e|7t6FHN+ABm&Ny@Wnn(JKK%gKUo3(pU0<PP>H`a^{!bgw8HIUaY)@MMfSi#fk+AtQX9&eJAVYF}g4o%sVwBvxm=b3+^5!j4pRduUMuQX9kOgkd{3o#gD)tMlAl6Or|ZW%E*Q4SUAM5MQh^hX+nn(`*cnQGsSPX&KTa$ZZe6sYgLAj(ZSZyKJfvJ)B)<cw7#VU{!|n?G8l*M!|2*;4BNVQT<VXQrVx@TSq6V7@-&Ts!I(_#Cadut?(#G@Ir+_3WTXRhVwsqm9ZvJuLvWuiFmj+rAg&tHrjqY9`1Z8M0S-jJ)k=hWk?I&*$XY<=(tL4<bul0WA%C5}dN==avlFYcf`}<g*L2#18U?(_}Vnb~NcWV(?zD=8)+3bwrXFYH$7YOY}O@7N@6|zv0FZ?L>CWwi`3r+;A^5ehbmK`D8sHJhoY7=CK}Xg@`k}_I#p|OV*0y_YGsw5e9PMXAofffPA&UGWufPxd+2fX0}pPR@hRo({xcDNE~7B4&Sn1>CWJB7R%LfV6$~S(}t8KUUNl|_YPl!hA4v$3<$K08FDMwQ*(HIc>dP<2dJQt+N0%JKGG2H5LyB0&~p`Blj1irAE#~of>Csr-_HBrh`xa^^<=Zn#wL&Cj2cfZ+ZN0c_aK(l!blJWQ6J7ReX$&GMx`m3py-*Z;LCGcG9SjLSOwKc)3f|g>5QJ$!o~eMEObIu6}hDm4%HJ-8f_v&5*Oj$&Qn@430dUe%thXeiqx<)_BHFl^<@yAf030{V`ERu8)R}3F<h)L8D70`*`N?w$*hxsIIZ3D1T038zd{IxW3;q$-SRwF>!27R_>_6;|NJ|WWRK2ZQ?oa7B;c0D#Cah-O~J*hc9M7PTBRaK1t-;R08?(qpYH|i1|~-{cOqewe!ph^3N&oPweN8vmTkMVkU<;y1L0y%_v|?`oVQ+^QqE+C^ml0xw8(DK^m%@#erq&7Dn|9@3>E{Q@;Z=%ec6?iaIa4RgSJW)sq=4CH;n4&1IttjXlwod%{5T2xy*@-|F=L1l=EOvqlo08uZfM7{VZy5Kv1A2J?T)ARmg!_GHQh?88qPGp9h!GNtP>7e50we71Gh@5h+o2@>D(w<Eb!}WD6hiKPE{MJP4|w6xiPr>_1%FLv%wJcz3J8I-sCh{q!p#R~8TpqolV9iD+x>5-yUTN0Q(Po}Zhy)?xwIh77rv93)p=9w<~%YM`%mLp2Vr`FqkWTk4A7EK4kg2|pj&5iE%nrIV#jS+i`==Qf2O@Ku?!CQrcT&M>1v-v7wNohRq!QSN5QTj08MG7A`Z0MXl~U+9F~7{~7GkqTMl5MKk~jrBy5nVqUp@Ctf6-k`u47?fXXk!nl-+7!Phd*<a{03p)JVez?G^cUo&@NlC19}G1{$W~Xk%chJmlAO3ZjHRg0=7u&8G0X5pwE;dtHRf~rA>{C}!b6Vu5+2ny0^)5PK{*EE-~g3oNRm$*-Q#$>ek6!3HFsC3$P0<k8)2XHQ2YAchP_AnjJSC(yxu5wR{Zx)n}V+x=2(JxE+otuHkKiOzeR1WZuRiBuOwFFH25#Xa8A^z0tf~jW8g-jHYuhE6wHN3>VkFP?$UgwZ!qZQbQvY2;*H{D1eTVYt|xvD{%463TGh&^+E-I7t!Lg^Kku6x{}LZ&S!hz?)v!@z!0S2<W7hpK;St@+)fwvL(&#R-<`m4)1W(A=+UjH<RsJ@rWT?^kaOyr89w@Y8GL6N5UvQmvuaviLvtVn;!)d$P(C;ImYs_n*on5k1f{zN-{4lct1hVc*>M>)uVfYmE`<F3U96vqWSQ-C4=q)twhhyrNZ-5B>vfGpf#3vj(HQz2%WMJvSVUQg3x|PM-DRDk?hpvI?SH&sj+I7{5WH)N}gZ9owvp)=SNt-m!S5&~UX&_^Xbn2f?HXyd2jzwCv!?2dXln2w-#Vbp*{+@jxG`%R3mJIOvZqr0`c!h<NP{-AgD;$6qqAxL@Bef7ipyzh=v(m^x1cXVDKdqftDV()lYVF^`=swwDb4Oa_?vJPfKC{SAFauA?wyW8Ab}a+~uYhx_0}0`2>!h@PSLz}gQ<$O{x;o>@uS?8J;Cm9@#S?!@YnX24d$VezpbvgJw|aoKOjYT$$R5w=6oDReU-NTTn9gPxG>-tDyO|!w#a(}ES1f@221wvD&@^<Ca9B=RS7L$PkFBuw7H)i$TRp`T0DQ#8mxcYlyui3jOA)}^$YhZE#Y9Mj1fBQOS4bkvSpw<01^L~SGhUpM_9ei#+;6Par?|-|v%_OQ=E0-F4V`cxcn+Cw@3t?n(dP;@30$ZAS=UrM(S!KlHno6F2Dg+y3G|uyI=20sY}r+Ve^}IPqLw71_%d;k-JXhpU(?H_n2AG!`~nbpD=X&6ID54oTj{sMwrLJy6|6?qkdy6VvkZ|14di09i9f8(pM|>W`VPeV#@P~ZpXp4RUN1lf6f&pLqtL~O`&6sS!K+8>wm(oqc;A;C=eJ}+*V9C|?puM=-GQq|HYx8|<Pk9jI<O+&T`qWW`bU|9iQ(`b4(=X%LpY8x2y!j}_X&@60ZOX9)MG2Bp!@PF8x!y#$17scseRA9*Ravv2K5&y9qzNL(Qc=>A%M0}5ZY#N(}$u@;&)1d84*o>p{>Tq10&+nvgwq&%5u~+UVsSb=XMd~Fx<yh#T=!K2r}CfIoB&*y#j&@D}p6pHj^W9cnk#<-9|Rjn>v{A7ghz~@9B#!0ZoqfQ5&0>d|R@8mTJ9!499I_#<4JMW1E8D*pBwXLg3Po#{XCY!^j#GYPlg@ssEunn~Te$0zEV(y_@l_$ko?zlwuPZ3U2`+3@S5quCY-di20&=ue$Xy(XgP16<u1CxLy&7s*FwyrAl7kPrbKaqJO4u2pyfZOd9RExlDq%qg4@OV_iqZFh<gz=<tkyQB?t9Va-KHby(Wxi;lUqq5QXIlv;?>_&T2-%9p-Gf~~C0*pDE%JXqsa4*HmPsTnw0(fgpFgGVwniH_Q`*yAoaG}6BZzr;+*{9W2yAY=w16Q=%Ju>7@~L+kZBUVF(R?M^PSX_w)+1{ZVkrWT<;FtM_=akpGQGQ?jGoX9LhW)B4r&+?7JLF2tO{gmu~&cklwB{1FZ*HT-agJ=P)6ha)hTXa6FMr2fLNFI$$6t7(&moHdIzsQ(=4Dbk}Z&zuQK@1;=R#30+(X7bvfP3eD(O%6u@Oi6kgmZ1sa-fS>!2Z@DJ+-d%y~!sda*asOOzS%Ll;4&<`%Buo0-A?5k?V2#y=!GV#7F+f1QN^_b{z8#7NN^Z^8`_v^9<bm5?W|2O&fty37e*L+c|K+pX#XUzlXjm{osZlsvL@DiT3^d#OwA0bLCdAKr?X~E9zgA#~y<<+bG^i1PwigIN)+c;A0Zd`N&_JbyuaPfO6v-I3FyWEh?I>5c1=6H{-<)%Qkk1^A5)1Zd=gnTY<SzmHJx|yjERbIvCIcB95%)X|eZh4=`t$(?>o)+D*R|`&u6k#UxglqA&$4K?VTeLw<n6Crm;p;!a$pn}ME)YWP|J!;@x)@Lfl4CBE8Po_khtJIc0Z>VVQyE=)PFWcJ#c5i}igZ%u>m;MMS^a1hSvq9~Y}IEL`$@<v)5UG2A4#tVLk7Aj0{v;&4&-{W8NELU{6Lk)ymPRlVQ@fhcsgRK$8vXs5Z0g?b%>2~)dwW>)xNhW7hTN0e_mIYMYUf?Rz4(xB|zqtkiUFUwkn9xWHe*Tfsib=WySaGsXB>*klkS<{x;f?*XMxER!^Nr{mnf}#0khb#TPZ*bdznZgnYXz--wy!;`pFL*PFe9zfQa>_Ca!+EEw0kr;`j9XTgPE=Jpf3UvjE#f}GTYv7%nSt+sOpNgys3plCP2KFlZ4|u8>6XOn%Ns<VyJ9WELHRjiw+v4n}G&oAZw9`&vheYtNK02TO-cMRJQRan{F|ir$0IQf^#bg-&uDW$0Alb2+7C3?yx{c5gYcV{G($8gNAJ`LoYIF{7QhJdsU-MgxS0xRmuH%Z<}}u@2_i7`V{{Gs5sw7MOx~qyjQ*5gJBb;{i1~iF8{ev?~<*!czsaWEHzr|$6wYys51;~H5JPDC)Wc;r=8eXpTYxvGAYK=d5qT_M(e(yadtr>^YUvIG6NY*hSmR%PRP;$jvekLw<~#<mT`*N6Z-UAvXt@O_Lmb=`h!nR%Jk?k5JT6=xSqOV$C0Uk<<>5Q2V*);1q1ogx$(U<4pk4<+CW+lz0EfbkCjAB>Ix>{+v8wTq3n?_p<8t|@h<f`Xv?IYiBt7D4q90QM7bZyG?l@=*7T*$1#x<QpGGe68?sqlz9a|NG{o-|*Vfs_FNU`X$L{0u?tPK51N2VbfPe;LJr*xhRDaq#1dPZTxvOaduSK&x&WgVmC)&fVbWgGIvX7U-6`Sb9iDsXgQq(VTFR>u`3FuCjxs2&!+*J;Xz~_`VXA7MK#1ayo3Rv*HreI3bQnL9cuqy*$+&7!Uuky9%U>~2>t=o+NdS15^VjyXW7>-H)EY+sY*7-Za$6np~cs?0s33Wk^irDY+gg;?S4>60~c3OF`98h>v50zrUU~+8tX_A}(z?_(9XD-zJO$s=lfzJ-=ATnMCsnc)5UuFB0A0*PHj#F%n*4bvbb<4-Ot|WcX#E`vTJcMn~__{Qlb=@!H`xvJe-9UYj4Rh*klv9Tagk7K+A$L?!l8pSUmer3VR42W&6qs#@ndY|mqBFAGjb;&p*0;4!G*RnB13w)|8^NvWiSyU$<y0KvUa4FJx4R6W6E$0X7YI3}8TA1=1^gd%9`&o!mNdLTgsJ-i-&l?r>7}ZmhQkG*#r(-$#;gl%3Ew+KF_}BJQF^8dQX_Kr2v=70SOX;1_p9GM;r2W)l<|BQ21eLac^{X8w0AfxnYORI6fpq$cIuG!drd^dMv#hTdvv#1qjzgQ7Z5J+ijs6v@IRfRh0aI(lGI0te?}Z!M;|k;=~N=FM&(KlO<aD;2FsohuH+MB;=urA-U^sert`eQxjf8RL-^@3Zpyv6B}=U^#Te27S$}zQh&_Nq#2vx1Lfh9LGNQ@6;nrGD0N0!la)qMEtBq)>pw2d9)GoiIbqGz|M)aQy(qL%xN_bR&R#VH1Qv%|$AD<E?Y^QtQSa9hJ%#Uh>Esn1Gslw9s%XZ%+ZBJ-9-pg|TZ;V#GQ-B)4Mq4qC$r|}L$@Tu2zjgW+2841vRh9i=$^v--1rZ8)E|E=#WsQT7_voBPXRB5gC?`^hM854#djv5=2#df_Ui_F=OL)(}vNjM~7WuCLe<(=CVUdsFxv}NLLtRvkX$!pm*I8q)&DKNFcVGL~hiJvNR06G8_+{P?IqRHOmpw2WUP5jBNIx_IMfq^<4r!deCk=z^b^m1D#eUSh35)e)awmD?-xyAe#%txzOq8n9L9q@|u4;To(X45&D?%4olg6NY?1ML{>Ry7%p9Iz?Qx)0G7EONxd&uEsgtIj0l#Z5gEigD&3jhl*P(&y>0d<FAd9D-V!WWqz*5ybx7EX9XE+nQYt-8S}*znS?pb2JIsH);HG^p#WgK7~c1=D;!+=L<0RW&L#J-X84rn`4Ed^l%_sT;6KVCJ6l!PFBWfM{CC1AoMIo#V4XD%+IBL;_Q)G#^0uB9oJz{E9cnSCf&_TRB#}WtV0A`~S+*L^|FpbThjnw1=~6e$e}zoPN^DGs*MNn?Z=QcjAS&8x;!C);4~%rei`^w|yJ3sP*#44!M4K)&!zE?G>~9Y@!X1)N0q($f4P_m}9{0E^npBRIyzXR%se^?V6@4??ZT$2OMxN?SsMXjiHvmC{`VV;X$uZ&1kVAe?fp?TajPyQS!Ox!mS54znzv=G()N}I-fJw{3(Xu{5%`x?OJRPZqGm$gHNSEm-S8~dy|Qk`V@<P7eoQDc!`3(&voyQguOD9h4DfyMyYlI8Ph{)Q&N<|G_4b-zlF2<-*M;~DhRS-2j1n##uAQmNCq_qtss7lVzX22$!OPPH(QAByF$VRE&p1XapnF-Vi+?HGd9KskO}r?jS@PW#A9~6IiNtkz=0^Vu?7R~^#YBEuTUyx>rE8rkzf3K?Qwxm*g;i2=pQEu#~W=8LcC}B?My&KqAy;2Ru64f2jtfwNCQaPhb~H@0GPr9)(CSovnv9%+t*-CpDLlEQl{NK>9z-Zwr+G&^7rTMmnr=XqU}cFI*>f9fx+8bQ$%!w|HjkP1lkjTb>}+aViTB5FsCQToc_j#e?4;7doQ3TFn&9b;G82!l0O1;*s1D<skC?=r3dX^QxS)49X&Lv1__o626w%0Yi<LeOeDUT=&0Dj!LH8WwnkNE!NC^Id@mN5(B*=tj<gHw3jq^4@C4ZsppWX-yZwib(iUt~2QC2PbSFR({Uh>+06~^qwqtcRzT>&K)!QNoXL5NG>zUJs1OxBL4ZORPz-Oa0!94@-GYZ$;M5EcwYvj?R9BUBCSvzqVSMM0oHwcqF8SbBlL-J(fhG>WHhnq*ES+`xMwac8iGKxHZ=KNjxAy87o12ls)>2@-+&vg#JUNL?)RZTvJJkHkzJia7-k93W!*I&wAhbG&_r%aV19_UbglXqO>uL;*maS#w~;idJGIxstavg~{Xj73}8yiEe-%4_0E*RDc9k8ssSXcQs)4Lrbip)B8x=9p}EmQU^vM7BbPoEi>A!)pcv6QgxzNn%ORZ)Omyhub8}{%%Wd5b3cv3o%i-(M@HTw`Szj+r8OpT-zeUlf{m%jnN^YRhm}#@|e)4S1Et9l^5;B?(HlC@rH0h?QrN$b_MzeLxivcKnT(^u3=32Jlwnj>rj8efKP}29o8&75EQogvYEz;W`3j>uEIDM!cn_nkAoJWne|M3AqHVlCoIpqcg~h3rH*^JwmU)e-2)b|;awb71m6Y#y43=qRPGQA9Y+2NC-8Y{D><c5yvK#6F;<Yv;ol%fAg&VWQH+FqyfItE2`&c;j$K&lkQrB_Jkmi`W@F?9=d{9(PDedB14~}nAS2%O1Prv7>&{R8b+FD32J4N!JFe_x|Dq>u@V%?Vg0kk(gk3_=*}X{hbf%*;K5mz>OSaB6umL!*juD(nOX3_ggX%BZEVsbl?DA~n_kM_Kg9nt-_9+}8km|>^@W5WA-~s^;JXt0GLKKN*tbHY4Pcb<idHcq{76>ipUG!4xaMfkn^dnepkZICujTf|%FlHMtR2iB2Mlef@HkC)*yO+hBcfIQ)%0SAS3rex=4Z_twjY133Z<-5uKiV10E&s5(Aio`GRG14+l}6|(_$Q1yD`UD{zMfd98`#mfKlrRM^g0#GBM2WQYuY?Yh>KyF{pk?C+vo*2T=OcIg;xZGD%#dMR_QuD_*TCt8J!#}FzL#+Qc34Q!&LZGVskoB3;B)EGmocHI}e2n?{rSE?wPb8^Y#jEo}}^Ed^f7pH@Qk3k#>lBqit*6?K9cC?!?cgSuoS7QwLeH{tT4*Re4X3lUD769d9F&5rq(2UAb^}nfBw}ekdxqs#NRo?Qvw+9zq%b#W#;700qWSoHgbwe2qb##@1tY_<=VfhN@*9;b)~M*6ugP%68FaLnSQ{1e->!10~=WjnCBF9ZbZHsle#ndmnr~)l=jLxKS;KZ$<)Wn%i*YqHx)bB+KDP)pt{w87O^d2?=E?8;@^VQ=sbLmu)+Lu(k#p)!E{}eACl3+lGAt$T6fkugN)+@j8=jB;eZRFHS=*yhYFxUzV%*5Ez$@+7nHv&vhF>7H)3280MfDA2M>p2g*muuj7+YF>3|0Y`}(#OP3+9@UvM&53sfPn%9^5|A1#?G$y6>(9?R$)YS0kU3FZ`CD^Hc=->_4addaykJekrIQ{7ROZ1~m9El!oyrGEeQaI9_3z$qHox^Ks2^){7)pCX<pnD%?(T_xqh=$%8p9Z2Gybk9M;KkO$Y#_Rp8|>$a;<Xd6l0)wcY7PD7%n|!<&?L#hA+U!8C-O!?UY#g&>@lM(tI>mXR&PRCu6Y82e5tu*fX6`XRmke#AzImlfm#ft|5zn$&7MA&{blP)N!DL75>92+_~H^0tf(f04!!AZ{uldcP;09<Raq4#8aALM3oaLCmcIDa;uO5OG2o20iK&lne~)3%P=ii~9(RBC<?M4n7c85odj+JYkPMn%vZ=CInJx9=+z2?-|DTJU8y-RhKH}O;yvDioBKBsWd4UN9(+cU6!mlH!gfn)(&aHucH#C}nD2jGCJ7ACD5oX2YjSVJ)fmCsex8rJNx+&HdQ`N2l*I_rmiVp%FLTIn3A6Kef`>q^OQb?uYeX{rk&<5q<K!pKOi&ZO#n3c9VJWmGn!*Gk04CsA3EMrjp<>!1T<A2417QX<BVCgc<eNm(|;eW|Hb(juNg?XN(M}w&zCFy)oMVL@)X9_M^2YX9we+sNXzc$a2<{|bgQ@Fj@A+Y`545kp__F4`(UJx>tN219Bm0Fn5-YCDI6k8`$sVIKw^2h(t$3gZ@yalZc(ADH`^X-re&&k_i-#lk(BUJs9y~DDP;K&QQSAkw@R#;!j$R*Eo2c|;2P<gD^-+1+-7vg2H*ZHyr32tm`k&V_(6DspG0agtXY4fqg#QYm{F;uL$FrAZ>^uKiF92z_pu*zA68Qt@GFLPt;Dp5d9)hx7<X)Tonwm&5Z#3v`m$%aKbmI`$B%l#EK=dye<xZ6oD%Cr0Q%$5wK2k}&v`TD<9@3+eri%jjYos7BQ6gNeR4+>~wM{GDY;1Dz?M?4!jVC4%)2&9btc;*t09-hqDcjXrZyM+|5kJEE`@9oJ6)(U-mor+(#=^$yxymB$gM#}83rLE0rYR6r`=w5TW!!_X@{aU>WB*p_U%7Hw*pwHn?m>@?2bTyrUX-3Tum+ITK?WG=oI!Uts@RqP`a7DXrKE3<MJ(@ONt@NnSIouB(Y4nG<mC?<`$(sb1F-8O;83k<364;AC?4ZD>2&DDFA)UDF(2XUh4(~i`-rWUeCRv`HT(tErg?pd(HFN9U3Nsjz8kG5S7EE7Ky@<lQrp8<bqnx}Sg%5lvF3SirIqxfdY^br$wR<Li-A@Q`_m|_MX$3QHMdcRsJhC^@=>Ao_qTPbE=0j$8r3(O62Vyq%!d?CWfuGfsj9Q!_BBp_dZMg{ClYS@F<H8*<wJaM^)}i=IDZ3WqdC;G_;I$*()bvY$O;6SHO~n`Yy1%Z!geBEkuCKcpoufR6)OfEiy(stegRg;;ixzO6CZ3dT^tiRZV52Ffj<+5;L@)3^$C=kMogMD*zn`LE=U8}XwvEE}$9v~F2irwrAe$eJb%1qL^aIY)yEz8nJ-A#WiFl5nO2miw6l8c_nSiwM9X)+QC!D7<0X6iN-9``X1bvlT<o{4B3V|fqkE9U0r+3to{Jf9#ds3Z^=~nH_J6OcIQ);sO_zFPg#1Kg){`h|K9xG-52lf0)o4*T~?B{t-4dWUp*j>hU77Y9ZDv#T~Ujia16snx#AmB|_h)8dme!t0vn@s}Ws^1o^T0?DSlz_|)WVbD%vN0%J0NwYIoaKUx;?42RC9=Vc42{B|-}bI9xU#}`)-)|Hs<e8r%>SRnB(tq6(D3!M6EYEno>?u7*q>Gz3lm_OYwokZMg1MtDXnD{`_>wQkFsU-`0DZN*iJ*rPm>%Fg)6AihFY=gDMK;qK;ZZQo1GT0%eCVFRQup}sp)C%=W*qj1bQ<<jf5J1_-x+(7NRFbqFClxs;|<)F94WhdP2~f3H{A0GO%4p;mFpC|CM`jcn?#)6VFw!iOOB(9f_u+t5hXm`DRHP1Vn<<mB25x=vS!W@)8^k+3aNP3SG>V4U>mgU_370eI}lsS@RVDr5arBi_9qtA!YdR@i$9cpwnETToS@vx>bZ+%@UAhnYuAkc{Y4nemRxIhh)XI<fJsxs+B#M_q?{!)7`?Ey*lIK_k!N^@evFM(YJy(eLsILw_r`%G9n3#ympbJR>i;Ef9G)XLhw47u^yOi<Qv%zQ379g-nUFILYLt>{f2$v9{s}N6Ym^ic>Q$N1a`i9!IP;xezF{xtudsu&{haSjZ~Q+&4})h*zTp#T~=lBO|){)01^8xt>6bq0-$7Q<qMv#3oALhcXMU!jIoW{_vwSdloLCKqtk{iWSa*S8*egW%XX}R3=%vqdVCPPqg}A@q~#R$Z+zc#!NPQ$YIlg!)H|76nLEJ-x{unX6dX5SU<tE`_%BZ-Ozu|V+=;7zUldMsW&he$dNb54Aj!1sUbP-NyAP9@w)6mOdRYduoK+mz!Ir#Ye$5-WD8ZUPiCEs$&cys)3T^LIlbu&5${VAG@YBxYKyKO}uNOMf<4g%Ek?3A69^3=ioRdW8D=unZ({YfZkjIJ_H<x9bdoY!aL=+JJG)YW>`_Dnu6QDutbn*31?BEUdQ<>;#Uq7ty2E++uE&!%oq0uk}S<_+YqxMjKR~<+^U#ELJZdGMr#I4@`E}8h9S^HeTg!Kv9Q{DiCkm-D;rVr0F;jUS_ck9*@MyL$Y%w<7Nvh<!GWXsLQusnCJ?nhA0cH`s@Z%oE4it#nlafv8Oqzf>fj@<tyK2X`o=Pq1@_53+B>j6$LnK@@L1Os;vU+3{KSv>Kr*lpn;o_QmVj=W5p$lU*}ZXu{3(v<`SkhG^%U>A3hYu`%8V30(nFgbsi5uK(k=n*BI;94cVcYdr9n4Ci>r{hhZ#nPB=IeoQnvHABzc2&&6x0C&ABRme3SL4|Ui&84dJ#6$PGa#Il+}X#X&%*~VM&T_wJ==CCBYJ|!L@oG5;OrNc6MJl*6N8e7hcLt*G}H_bhU<+M<St(qmUG3TaLO2e4+NMQSms^8FPTx|0rqe}4_^M^xjC4*Ijir}aY2JD|Ii}ROjtT3yxfcocvUuOPCUr5Z-VvAOYzcem_0s%!3!<h-gFPn<)POakt<nFW6g6^T3VAcQ&2aZw@*-7+ra`R{7bmixSPoEJ{qJ1sR}JrF1+)qCXY^9R;bL+OC4$=`;L77s2iO0ssB2U;$xv8hGrQf{ThIB!@zjx1-vhiqgp3az<j4ETZW!9HwTjugl0p<JLTlj@V+YOH^k`er~Gdp3>2mGBI%>oPZ(kSb=b}xhf$~?qyZOV-7k5Ra6DHMal@p)a{gf5zQ4v87kM7?z$K-^1jen@;zMvlUAL&>PO}oFWQM<6A%u>?=j8RNYB5Ar5yf7wa?&1@F}Z*A3Y5ZBCtW_WAF(qeSXEyOSwlZ`+k1JdPd#(`9}3<q)BsJPNNF!us`~7qwR8{S35$-E8LxgK{P(`NmpoqPtWK|ey3PN=LW9C_=dYB|spyI;w&f3b(yU;VEM+;#1mT*5xOtv{XHuqtJVlmP^TN6jB=XM;(#LdoBn=|1q`ffydAI{YAVcm`9Wp9XHi>LH>O{YvC4e4Ww_FOgfbO`(mYYElzL=QN53FrP6o`dR;FZ@#vqIz4&lNCZ%EUFO-`Z&A?2xotm2)k7MTryB`ufP-%eg&n=#dkfTRPo+*fZ^<=vKr4(tc$$eL$nb-n?k_w0sG7?SLny8~Sy1^!%g4$?zu;o9hc!7ptDqb9pGG0t5r(yT2lza4+z~;)vX}*3-q(7&(k!XaSd?t>`73TaPaZ1(2>nOJ&;>EK93~%h)fWQGe}XB#4;L8x;XXLrT4GOY$}88Kc!rWg?I|kAH?V4yfejt!E`0M~K^xKNRz}{8F%oY{1dcn2tPm)?h><)n8Wl4ZP@B$m+!^Y|yoa=vGONM>?{}n-l>5$)7<woy0D{#vG)FVJ3MXp|_1$4t=?{R};L=EtnA^-W)s?p7g8dRpj5T^3N3MHDu&ZW83}RqcmOfC6{R*;oDdh&^V||+ny=I@G}jD+uoWtE@HR;1#O`}PLFTOo~zA`umQhm0f4A>7z5S@ttglD!Vbv6#XkDNu1mF{LY*^nFq)KZ6qUvW|2-~==Ok9^4iY6EMGS{fE3J7zjpN!t0hznx{pq5YioxY6p#DccRsDzIR#GNtHrn5%G^d=A29zBX$fC*fDa^U6i~6mfAky}e*cMg?N?nQvj!`r;Wni-2z>=*A?fdGX&+{iuK$CP({huB2tTIUgaq(<UVt}$2#KP8BwvYF|wHRz2{etm;E|r8ah<`Fm$|_JuI({`0`SebJPDzQ(`l0!O{f6ISLYx9`lV`-J9gVHZlT@Y@576776sveE59UOZT$^5gzc=13#ojHRdq3b43ZwY;(pf4?F+&b9z&idG%g9Wzbll+#GGkBfT#ftZ`V30+6A`ieo}675Qz1l08wl_UNcxI-lFCr;`f*-tplFmgqw{OWDLsoays+IzHX79I6HG>3`@nt~J>22oJY?l|%><LjD5nk_>90xGR__|GgqT%uNaQ#Hnb9p;)k{C&De#I+rc}Qri@U0zxWN5#DZB!OuixxT91$!W8J|r}nXiBf&`ho0kxC=3Lx9x5*wH$BKrA&D7XFS&hXV_OpCq0zeB!I9Vz|gRIC@nC6f;uG)F6_eqo!?3?P??`S+2dC$6GM{npSkOv?mx@BsKHfJ|AvME=B&^R5Rw6PEn=R(i)BF=xuf{O}$@Psv)}WVlcc!VO~T31bWjq2p5eLOYwCoPdL%nS(lHyMiN}0FOjVMRMaK@gERJge%049SUU}h4-OzRBDWdHuv;HW2PPRond+#nD8~0*i@7Ka3K8|LqXGu4K1E{Y!YNSA%yfxpFB00~ww(49s3uo?JF~I>Jhw3PUeo4thr_ziWv%RhJRffO{U@cT;Yta_tS6-+&67(nQ~cp<gB?mkMHOKU4TpC7=>c87_m+^Y`vc$#4LX3mkA-z`iiNZjXk0C|)5=03TPzl27?mop(@#lPa2^xO$|B+BhRCVImpu0phpBqxzX6Jy4A;W*WL*ww(C4J=&gKx&Ac1M3VK~H&g#kwSmB0I>Kac<(ImHB{#oOm{b<ym1oTd4mbTLS9;IOTpTv`TzPD;fgkVQ~w{heQbgOnNb@Mw3amYNh5(?4f~6@^+vn_%HkNN!`xpCCe!3Vte@?({;A$s`d@Y?R=pieev0T4W@7E6#DbX{d);+$Wnl(zbBI(2h5{Tq64XG@wpx11k*U@o3j)Jn-2s`nkgOIK7t^43r1M1JrPkl%lggPgG)ee|Dg`!u?Ed9I>L9%ge#z<;@w@aG1ETd@y-gL?Okv>uv=)X305q%vR6u!6>ha9F}xC7_*-|xDMpKJY#fYtE=Ntr8>OLbh17Lf29-tWui9}aXw5&{p`6vT_&SNKoDw!N=@?4)gw<NvMy;5)$<zGvQ{Dal~eqS8!9SXbA?($N)9}00N?0SWo{`Ur&tuUi5OVEFM9f3P`K+JVIV`*);M9&H4_}IT%2U$Sv1=RiR3WE%vepZD<F1C30qwYzW>JEOlw{Fx4pz~Af0jFs-Uv~ZZ1e879x{uZEBTRLi}f$r>lu9t^!jG2`Qz<{B%{Qz~~C*{m#h3It)Qiy7*CR8Q!7J>T(rCUe70Zd<>_k1GbbcCHQUJV_B@was1+!wevUZe(h7))#0ARiL=8{U!1bCp@vtTMK#(6lZ2_)uui{<I17Zp6u@;*qHpbIuP8XvxZ)kqaOV{VkMHe>vXe7#Rlp*<S5&hnqAgKVl}pjRHT4Zwo3}HXg4-Z;Qm)BZtk8VBAjl1x6rlqY$bk_0AAcIL1%K+P#74d!^pO$o(22B@9>6_Gpj0ef_5=0bVU)@WIFRTMM{|A#R9wBw4osPT`87aHNNDyM#~fJt>+n$F)FXqJ2O>bD_d6aLo$rnk&Gjb-9K@RxXE0u}IAHy%xLP28dN*3)RpjH?Qk63~ZUMFpbtJ*HXpVebo^EfjF)}9_0oLhUHC#yDDa(tQrAE%Bk(^Uc8~cZmNpt!dLZ#gRsN`3faPG(62B%qz8Z)fBA$0)`ivKbMWzW@VSrE(P#3u>Xi`E>)A?P2%O4>&e7Y`7y^CmXvIKKMVBM^tAS(Vy;!WYPTP<1JPG9+SD_5Tkjxp*zRykFHx0G+hzEt6QR2Qo7?J8V&LPieEHl=-CmY!`=7b+Rb<c~3uzt@@UdHfcFDM~cvW3;22;M!Y+oirJBsZP62d-`FNp_}_8WilQy|#o1GO=Mf7rQ!6kxd`fBcZpzfP>|~4csJR`(OHeg`$1uMZiyB`ePF|@+xUqpvVzkoX?iA`Slq0+_3`XmKN4sb(*cJKdinQ7E=i*b;i20bPOt;@phqC1sQ=)T!^Z}1Y7n_pYSbMG9ck{c;EWx1%-OCTt49F?Nad)s@6pw`<atD9ww*~W?0pZ%dJ*k_nVw`CYGP5>~aucWa1t%{7GVrMMuoL*|o5v4aDT><j*=TqiyWL91NOd;i{uyc@5U%EM94cCu3EBdBPb|~x`VKT5`h3ff4(_0$-w)sCPcV#%KM47rWKc@r<yBOiYndYh@=bX!kwklA)a*Hc_Q1kU>yV_{t$DuQ6thO?3L0dTy#;l3uMt2eGL?nOsGv1UqeU7Np?W^udNH#=SP*VLShaLj>S|lGPNQGB+c}@t)%CSing!{c%1yV=9+veqGjO15$gr>S`sDqp-EPhUg9OdU{73DdFs0yY-EY`FP&xQc39ll{FyEKm=X&2rma8wTM`V`od>V^QKVBQCIJH?D-S|sb5affZml&lMsOmAN{u21JabL>GmuR6iCV~_zB6pSjhCc8%Eq>{!T9qgJ+LMC}?BF@1E%s~!BU86|`Bbj;xd;i|C}-dh4(Q>r0?lEUYdV@Do(Le$iGfd_CCqb_rdO}nLUhjG$oEi;<MJ$5WkKPfr9V^zuqdH#-d=Fg<L3+Q=wBBu>IxHbx<ggks-Wgjs8o#|+$}|mq=8|~BY}wVsEJ?8|85R11AtsKnK!!TsYVv{fWvkl$6PTVm})GiUt?9z^cY=llYY4SZrB{2ofmRvvo|AYF#*9m3QUvK>~CO;`CQn~cw?`rR>+NcRVV#pS9K?<fV56fW1dr@R>wDV(m82)mbbyQRn;19r+ER)((!$0g!Ttgq1qz+{0AXO68UH+O}c#;kwKmV35`B%X0PSdT~|R-a{t?5`DB&irwwjTvtKMZNLo^FsT+KvL^uPxR3S6Z`m*k$;3qHkR3y~R9&L4aoZI@kLB8Ey^$RdVgB4y_?@08nk9TsqbR3!>G-m(S2P#V%G4vTMU+S@ighA-|nmB=vDFgeNPuzpem;~NWjeEFV+G0}kbHeU7yQLz_{I%S2vK}EOFEk!<Ld(GVr>pDzRVgFa{=H{&WKm1$T%|A`PqJlv0nB20S#9-<B&r*sh@V%^vr=Y&@}DxXh*{~Z(>DOSXh*9oLWG+?m^J}~?E8r~zx|CBp)SfsY6%3`!Qqace)4%*r#2jrGS1pPtUzEXD2XH`WhWzpjqOk<sw(w%pFjccTwfKD!Qo7gvMbgnUZjM2df&Mp4eM4JV<<B)|Fan`H>s}N*VE=XLq$&`cemL-qNlh?#2d+DG$(?uK-TrjT_Q5OsQj}!wRzrZ1X!?1K@b)kmwK8%aTK_Hu&^eyRvnY^8YUyD+|HFPSlK`-m;AF4K`X#n;_NnUaIFk0FElH(Y(l<Sy;-m|j7b?M3wHKf^tHu}JDSN~$cJrPO~@ckamF#tWiI;O;Q$^|OUEvBb*B*IV|Fsoje9NtAdquD;siCmwC&VYr>`#_k;Uu_BxR!-4aw(c2vI+Nu?|}6aGf=2JQ{RvXX3hOmB=5vbn6W)G_H@%{m;<cZsq<rizbP5?k`svJVZISU<lZIwkh4UO2oB1rCBNJ0`f)Yr>KW`c`YW|)Ztm$@*DWEAYR6IX>WAK+~$NMoNU{^<S82m!U`&X>050+xjY5)I>^vAN-+_Pb5ujqzuCbiSq3D?7^hWsfxypR&p_~!aNvbi>mV3y2&+IugsVqFXI)&Lp2qb9gqx)Y>lUo&xWZ=~KUndy<jk|mlq@sD7UJ}#QGAHe{fS1yJ{bYDtch!?itJ(@ufg&lY1Q7_<m|lxe55pQV4WrX`LWGBkI~5L@>X5vH~9A>Z%qJ=3rvI|5BOU((*gRs>cOub$#U>7&6V>vR~{-DWE~S%|GB@$UsL3etga*SOpUn;g|eu^=;*~8lr&ymN#m@_ObKnW-OUo5dgHe&7#aDgc3kVxpAYogFsPi>t+H3y#pJGg_t|f1V;0ZEQ18Qb-8Pu|6KRm@N|#-1EOy*s7#!NChtgt|Dyx|;KIWK;uXTPPG#f0od(4a4LsLB8LZ`DTA9Se&N+?~aD;=xdv-ZiS<o<QBYSty&?cd0w+hpf4VYGwaK#wBUukGH@8sXcFdWT@q(aGnv0HN1F>sgd~`<A@5Pi&#N_1lE}4plh!ATN0_QeXs%NWl|CV*u4LS3|#uVl2mgF28`)fG*{uJ-V0u!FPgO_bTMAqHh)ggi#C#G|%+CQ0`!Rxn)<BGylwx>o7c6DPg-F8pNNY?b-O`*oEx$at~Xv6o1?)`>|}OupjLZt(4@j_P}73VZJXxJ9I?e!$UXF4BUi*ER)%++7x07ycjBFVFj+GEFf&gvDqIv-I*(!Op?kL&!A`DjtIofHi5^o97bz2Y7qMz<r)V;J_OW1$>W@tsM_LUf762{A~w_gkN#TVuk?`evd<u}yP(w?;A}fK5Fos{kJ*jpDq8{#<4<z{v@hYa2w-jc`t-XW{i#{d|2t}0Vo<GPM~j&X|2;PbMsh^A&YX;KQ;?BWqTR(z7E9E}0vT+zC|jsjR|HcVXhc(}7_;-|^s)l_DHqnBC<Z0kmkDFOOf;do-$1C|xG4Rw`M857Up7iQ`a}b?+`C0XTFEUmNBhS>AfqW|y4;YzoQySb39sw0(8KIUunCF>ym;7ioy2}G=K9Ge9XI{Q<Tc*!RT`*9-H1%;lu|q>!&fA!lF*^z8OEgm&i&_r)3z}W&=T(c$=6>6{UUBr8Ef%rIuA5@IR3+d-mWr*k;NjlM-9Wrvr@fx13sz^@9UMh9%kf%$%Xd0#U57gUbSqL-V)}9=HQMw>41@;ofFJkB$bvNq)M@}I7l1kgV?+3H+q`wG<qHYU$Y%AY7lXTQOgcLbJTadOKTO#*1L8SZ&Kn5J6sI%aa%0?SCLwHMyZt#X~`C^*b9u{#H=?U3V$x%1#V!5ET)KU3M=%Z@VY;h7wlmwgB}<o9!Y>H$xC`%3p*8?*PvndVb?|0TYG-J%;xF20(iz12JkoF@v+JsoHQiahy`o+b(zynX`-QJ8)XQlwl%tq{~G>o?-5V=gg_FI!^;|S!90<-*4NOP%K7NE8txfoE&3x0Fu$>J<etjU-4yOKZ7*~raMRecy4S#~loE6b#aP2RH~<j?*t!c&`vFL&DJ3rV-hp-J^?J}p2HtWFk97zZuzwh|u1{lRZ$!};gL~;fHf^)2RD#RU{Yvd(I{|;C$+Eyzh#=i<E@pZcEwXUJf@_>2kkqzEBg^=)2-$3e{<3-$w>)(oqp{-LOB~X=B*-%){thwCPwviuc(gG)m#;vLhAesv?E|Fc<GVzb7J_bleOC!=b2$tw#x$_K%kF8M1lQvClwl+<pw`;wdcF`2t|}r)LnIKK<=!&RV1TpksOaK!Su8z2#oe>6$9U)CA+nc(x%lSY{)M`ROjC2&^ZSG@Cv?Xjz=k!g)cJYPR9(4)Io6<w5KXDFRz1dbLfuVQ_HvPYjf1&N-T2Ml{M&xH#`*EXev55|Hwn-3OWYeW-}_3uMxSAG)zuhgA|<a>8|luI>!I6Vb68Mb%_s0;qW&}prw+QT>|nBwvdCnI`WTWunXYKGlM{4fZWA(5tMn3pgIQGBiWq=zep6D`b#i+MSX$9&;q8B@kcS}!%LD>&CoV!eB^Fd`WX?{}%J&iQ_=R%~%wcW%MjF<aZTK$UsZCt1B@+1TL=bfX+w4NKNS;^7;b}^wOzmJIxi0K33eE5xaOJ2c_<X*}M=0mrpNE(WQPZyYr0m}`F`l(YJbTNE;vACSdmqY`>?1||kvAY%8)b>P?Z0DwC_oRSR)u9CtSVjE<WY!|Zb?GSpeg^~b~$i!rEn>PuLEh?S?DKk)j=MKkxHDD&Qry<*jQM=j=dnT#KAqtLoG<sSgDw?{*?A};z2sI42a7Pmg!wKcsLDDE!fGu$1ZVyf)?|nq(@>BW3zR))JL5e<a3Q%N*kUh>F@ZigEvPeg~bi!>v{XTe6$-N0{|Y4Edkp){q;xiDUBl?2Xsj;<5Pk|tQ#wa7X(OrNX0w>)Pf`;1ZouzQL9Cgf)3x-#N16_zqM{pEXs;PM+!|Y4#iEv%!ItjH8uL1pf1~{*g5C!U)CmAF*_zcu^S29OK!dyN<`idI`h1hTk;_0g)GY8F{26cH&r=?RMT|ObQpG+Agm256DMPCHE+!Dw-0&DV)?MI>C$qSYbPfUm$o?S==O0c`lv)T5wLE#`6$LNoKmLM8*x4R&fD1Q^!3ZwPW!d$-C{88EHc*UO~;PE4D<qesa2{;7{+q1REA%e!R19R0~`e~-k9}o!>R!}A1(cr%5aeR`-v6qCyR;IEf^MFk%u}jzTkBSHI$F>hENtS4172h@gaZ_VpF9v^k;dZEtJi0ev!~H%6MU><^D~r_B|TaBu@OOm8!6uZpY9Dnfoz3nf)*RUTZle8uSwM7Yh1(9@m1kQxoMC);tG)7B5W~f0vkJn2#8exQ4qFNdH>Xq4?>o$~g_7sr_)$i+)JE?MPTiAD2--gpGm(>(Bv$F**|s7Qq$GZDU-*_klG2zyuh@ubZz;I08%CJb(rzf~C_1uD=MJGpsFEMY-?KQ5Iw)%Ah_s;4RvJ)>g4-X0*gpC++x!ZMnk#T*6mkBR<py9or)%&&uIi4_XQVqI8C;cJi7Ll>nf6dnQ3#&^jgWQN;BxHi3dMAmni@IZFYI6#c|Ubr@@((`aQ{KzBv8`l*-LG>o)s$XtxM9}LzIEV)tQ@}XDJ%ZS|%4YwO?lEy!(PEG{BQ9x-J%`Dum0wTE&WN0A$duzAr9jy;YdeN`DrOJY!GROS7>3EIsEZxyifqcy)L4NRto3;G!UQjMldIp>o<TLUe8avZtR47G{OD61~Y?!4I{U>F%$VE$RT!1t($6Z*tI@$;uZbI-$O(y_S+Ig2Jd(;38xxKr7cgsYz`sT&@gD?@~GwkVZLN6JTkty+pfU~HE#4w_CZF&4l#m(C%$6NeM$5a(L45mc^vvxWOWwdE)Du<2~xUFDN!?)15U8*LfDhyLsGAPSZfNBO{23an{WyIM4WmvZ02$?)K(JKT?ZnhB|q&Cys{uOc-_jm(d<*ygOoxOyO4pp^^%w00hkDrSlIt)x;tlJah+YuSKaV>ZZpwoiZNY|3`Y@0Z$6mW08G8#5!X0+4od``yEL}Iu6_7%0xpJQ>c$gq3~!%s(T0Z<>$+&ZwC2Fv<q;S?|&Xh`*Gm>g*N%-eT7q$8~wRndf$nsFD8$yQ&n^Kpzw2-_{BDb)$Hg07^0$yRGhA2^K_FJv1Va#hr+5Y@?o*TbC<(rU=>@y3uyRFo+%KWN-aRUQD6ZLd+2bc8#2eA*CplSI1*fj$<L@`vdCZ^r)ie%nXHATy26?+h|CVvs71;6B+$u_2rzlk<2NpIyTCKn+568m4q83uV`o{Qq5BJ$bNRqN8R;Aw^{}=il$w_S4)D#Cr2L|3ZpG{88s%=tZWYp@>6!Jw1@FHL+eJrGwN(0TA^6l!d2te&9bywZggy@?T4duChLHA77U_Cn*P3LU1;CuG;Tf!h=;P*B2<qy0LoiMrHQvjf7-2ZewCV2aYvgVHpvRB!>#NE9OnmjN3R82AawQ!rM2x+mpj8d?&UkJYlu+QK~4SV1p$u74T8%4U6-h+Wz3SfJh+4r<!_^v8V|DX(8c&-g7O!5o)Ct!lzfiBzg&ZP|DgnT7KrsN%R;Z=V{x^+k*Mw3^kfoFi`lN^6*$Gy4E!AROg$;d3A+CiWnND`VJqMNyW<M%}}pkUA_N!0J+=9oA+i+WIP+xGH^Xz*8g(QZ#Ue(WKHgDE~<vJI!MIg`xr_<!t2BNVhm<*9)(M$s-!Y{{~E#lBm76akau0|F-Lol3Jmwde;i2VlC5;UU++mtiVo;A1e&?&ACoVXoLhu(OZOR_UfFspM)Pz&F@dv%nXY{QoJCds5Xym{M|eU9TKu=XSg3lQm;Ak7UdOJLDTR<VHRLgG`6KCdcjvj|kTjQsK;2B)Y5LtIvmGbDBWpGGE6ncMuIieFWIsxqzF7>`M@I1S6dbR(rH&zZfpG*bL>LRzzAg%!mvCG&|LGSP5DD(NX>se6LpfcVq${|4ZJiD734&j{i|LGw>c28pWC{{G>-^fkhm$7uhPbo!Y!4H&V=Us%@~(JP5^N;2wgd?3W=5j~S@;4Lu@zU%RuN_CTeuV4wYMs&L+6Hl7e<&-8KYQY(hi?>%k`G>hB8TXM^{1{4{|b*wYb-$+SSLG-qtQsNm;-QeGWy$;l)b)%_1x38ylfKVx0)aviu3Gv<Dq`HOO3Jpn_XWhO8M%ZYuo4d3IRwMxVd!c(NwDy$$)Rk$hgIfqGIt{b2@Hr|na}Ys1Xlb)91FxG?azn4p$FRY+c?ZGm&x;Ut4S2Mis_AkA#$7m*d|3{%KCo{8Ghvq!{atc@dCC89|BW~S(5Fk;M`B#x_jW6=Kf<Nu%NWNyF%n-d%MKGe{r?u$gx8JnxK9Aa5hD5{YGKjT^t1oKBSlVIBk$@)7DmISY_m{}Ly9i4FNTqy4Qx*c5-O~TA@=Ir4;Ff&j<E7og<AuzhaDp#!z!f6fxDhT8(iGjBiL#K8f`%0DT*@+i26?77*s7?uTIC5Q7vaf{~%awGpgEV|>ZPay&4(@44VX&I#jt4LZKw#Qo3%#iHHq^b}rwD_jLOygi%^1*$=HBnM`$8<txca<bC;=8f4~XM4No1*G&lBFrNEq{aY07|O+CgA=wbwgf5b;k};{Jm14PP##a8v+|%^6@GkdCrLYmY1A=ifVDs;0EaeP+xyKJ~=Si!GZyS8nO>sBo~<Eji;xp2u-}cTZnic=mD;h^-k;*?;#QQ<v%tIkv5J^C&3{+Uu3MiNUT&m2Qm_AqbTCeu;Z}K2*=L0Rn!M%U!QKDv#CLPy#&amQR&HA~XEbDWP*+T!;T%P<erY$<9yANHn8xRElAP1te8wqI$K%&tgRlhS3u!b(k)rU($54z<|o7xLl3{Xtm0w{3L7nafqWL3pkM^O0hcTPjuRM5|tk7ZYrPSgYs7jU3$gZCuC`D0Wsol5ihH;E9^56e|E{gNEk0}*Qo0NjejcouPLHR5=yCAn!Di%N&v7%-xRLioAb#6CkKW2^OcQ+d!yvZyKz8~&fdH2wbAA!J+99NZ#I9XW8^~oszmkT+A|UTyc{WdXoiwb--4wwv`$`!5Vr*Ax5fojR>5WJ_FWmXW>Qadl(juJkW=-gGilk|<cf{=f<5{q=>-DQi>=5DfWtlqXyozF-5iUReip1+2f`c8x^usZGUuWQgoMWo`zEi_T7B{rD+=zI{J=E_ZJjygsQ3o)VaiJS_;-Od@`(IexG4&SHh`~86eUaI;_m&t+h@VDxt(=Ab1OokcRS~jasrokUqIZ=W~!bsA|_mbbczDq-V&DYehbxJxtG8SvPl}Ux%&Uq2gxA|Hg~I>V|Cv6)Fc}MOBGGb>+|Du_**pOSzul>I}tCorbh^eSB(MN*Zonrb*~f<A3DlhGycVr%7jQ3TS>b4U+e*oH((_I8T{CdI$C(n6P83y`yT%CaTEr4z9cS}<$uOO1=eB`2X?~t22Qr%C_7Dfh4pge;$$~$iuneSpg=b%G2V<49SqD~0iPQ=6!!Fa$Wg*#5NApM=*vQL%{4Q(-NcuRwo@E^q92)|v<T&^L*ym3RH&taX7&(RzMflZ=wIe31%F#icSOd+!KWBKXu)ugN|MdqZ^IoI(@R<;Mb>KhecifCY8C&do@SM6q;@jkkSsYz2MHQ+$4sCqv3P{PlLT0pxROevPk%x+_H&viEP?|pQhKS=xY^rU<HRMZAihLFYPC?4-<7LlaYeso3hb%%P99vyJrwV&{aV`eyw+~oE-LM^#&J5+LMT&HwVNlA7!6*{G*lu*a$Y1?^OFn2phAc0U)Hg+?E7$L+h)4*!9T85L%$<NGl{JLr?;Ur&S<O62YtoBaJt~MPe);=97x*0N=d3w=ARF8V<NA4hqy^*tL44C2BgW;*X}nG$h|uu(;os|`4Race?u1hw+kgAGd)MBh;(^p4g=3SlfW4j4et0Dy)+h2Pcs!}y@Xl&MZQ=3%Gii?55IAw)Xh_vqtP#YNR-S|HQ4}(;fm=feo>2!ba?SSSP*T@!uT_np&PsAD&e*h$;P)f{B3}}QOwqH7PkDy=Yz8_79}D^bqRh2Re<z@Mve{CrNn4`z3RwHi+2Hu6mI)$J3mDMG)SwjtXTXNJu}o<H7Zj#sj?0hOoa^QN8_(~-D$KOrB*C!Apwv>DDyoyRL<Vs*G1^(T0KU_HPaJ_b6*6>iX*X4ArbsNCZ_jwuj(&SB;c6n_Gis2))|n|bU@H=wL)jbJQWZB(Xg}_g4~w?Pbfdp7y%D9=K;e@*RN!omfuQ;F+e4+C@=^dp75&?{sj!WRVLTmAAs2BdbbP=cN<e(^qXMKXTw4Mrz&u-)H@CHQ>T>2<M1A>m!5f2rDv>$49PU`ngy=LnEEuI4>Gd))H3>l;IcWy$;?b^-yFPjIa@u{3#SH@>6E5JLDFm;9AUXNq_QlEp^YDcIYm;Y0r>xMi;-yH^PtBJfl0(diPmy=xk@CKi-R{y57II7V9-en|8~vP2am+k3c`Yp=TP_cxSO{REpoH0nPwCof#k!6>X_#XpnGgM(g`mnI!~#buA@h@1b0?LseZ0PXH-qt933&;i77BXYCaaqnK)hhVGJp_`>ztqOdFd2CncMeiOX5}F<ch(@wTsC|0LlF1zA}||7O(EY7A=++N{20Y*?*)JzAbFzTxfwFE9sK(Lq!g8vQBeR;v&OWJ1(SN^?{G>j-`H%C|i4m_IMCmlp^iP_6%TDp5RjwZFfY2*bgrIr&>66`~Z?PNcksPwS|gP#D6a!GuDr(yWQ{bx_v>tps9R8(8u(Rdfuvao~JqfDo1BWQl80P{{j@1X@gGRj@Y8U=Lq2gGt4hR(l?fj`U(#MO(gg9nyUKahWCGnJY@=LW?)L&rUqY7pNYbPI|&I;u8MH3fh-qUC|XLf`oHOHW_*s`tdw0l5x0uUz{)R_+?TI?wxeo7(7bk8w5GTrn*Yb{)F^cK9N8xpnd|(eGV^v(N}Ax#44hhxab#`bzEg)ipf26&U<6FVS+RDfB?A`4ciRV4_GIXF4LxmqrC}wI1O>a!?&tixUS9@JZ)8wy0Z4u2so?1Q=@K=d*9Q*>Ney#8ss=o`;A^;Uc%~v0>X8_ngR#9C3RGBM>4@rFtA2O?!p{DW{Lw$8YHa_!L@}j2Tf-SJ~If!-xhRAg8)yD5^<8cqq4M<*(q5q(~H+6xD4Z9fPCJxtp1~KOG_ovnB(EjNR&VC(BFMceMs>#n|!bTx`h-OW0myOhq^ky6QYQ}^a|-H=L!Q0qNH=wcPdGaXn3jD_yUXj*NP*E#`06aB%jJAJ@gH-{oEfy5SkQ;x$KQop9CFT@*WRz5GWkfC^fuQ2<qB@OS6VxohedQxQ%`(-u19RRmYyEUdHcEVWH^J0c?*w28TO~dnXIipR@RIB>@TL*rdZ!V*QB&WAkzzOan2oWD<T8*ZzcRCEGDh8W$HM-Rf#u?_nu=);kebL@Q(<Go&r2Gjs)X4b@^9eR}gzp+Q9|H1v3R5}kXLLgRxyHRriuE|7Ti<CY~eqmEmRy?%Qt^V|dG)>q&MNz|KA!ZiWyJ772t>Z={+L42Vc@~!r)o^RnC&<|be-3ReU<#oDniGqM`4PwNb7=uE{(o|aAumE2h7x~b^45}R$5`-ub{-(^2M_~)zlup6*%xbBIOe{#1sf0p*6=%dbmi@&~YGuK8HcRfq1K1BaQ8x3=pSTwy27!D}=bT$17aasz|F0bI#y64Z5N6g}>hc3~LHqGd-+<RdiA?~*g`gb%h*ZzOm3J<lwQ0`(k>%Dr4sb6V0J^&_B22zSNtr`#zozyRXlMClrJ4V4ahPt}0}Pcx`Uw1IvP#Dxf4tJvEi{`1_F$B=^#9Ww_dyKjPY^HS01AJZR&8<|6BX`hVup;T3P_E<N_=A7(Sf}uCnE>1M2~4ZQ`ORx4Q60gJt}v0jtm85#18EAGT^$-pc2Lohnd7c1&YR7{rK2ZHw(gc{o}#Z-+5KT$-}zPp<!@4AM&tm=uX1>mhr%&^71XZtMyIm?sWVs$zshBtU@tHRMA2OJ2fBdo<d?VUV}GSizp;y0gRapuzAP$+PRH;Vq39KPqBvCc>Qgb;s#WOOBn!V_r5p4zU+NN3P$4<(`!zw!fkf&wOlX!&7km<X*uoMU=574WF9DQqo}nx*3O^)JJg8GLh{Q?vIyYi9i~>@&c(J*yRhYQVw6IQTROEU6R&LyE=#w21T!zLB&(Z3T}QfPE%eh)m24(oC5-__RYUuK)BXa9jw3(RO;q(j^UEH+8NI>y&kQ2@OW8blL6~U1wr6J@T|f1)+l#{`W&`hCNww%*2M>@AQ$SQC>sKBiSKzWT)%Ta|-z6GrZHp3z0i#XDBFd_dB>5#Fe`Ee=JbtQ&Q9$Xz{w0h{kAxVbQcAUZIr}0}ybSvV(pW^?MzzA0xXD|?HN6ey%dFBWep-lLwrs6Mpm#Yl-m9Qek6W_mUc99Dp**s3ibDYgioc-sFk<j#P(0LkF&1Bp@7gRigl%f22pcTk-67uEgiW`)48ksXZ_jn^mo6+ew<b2p!Hb=l_Vk9q-n%U!2iWVMVUP7UMJ7;YmXj&mL<y}BrkZ`&xg}fg@y+vYls$$Rz%%dq8D=nStPV@jfPHTv{g^Yg{2a<WF+dT$J>DDj%2WbO+D22;n<eeNi0E$^z$F^R(b%d6rl08I_qj)cD4@U-se!#nIsd;cPM-t`uL*5xW3J~!fYefns642xb6DCZG2+>6t*BQilIAR;R<lFgL(T$sU%51=6rjnx$KEflrf`UCkOCa@rq;Iru;yOTam1ah$0~9atp{2S_40ask9t^G&Cg=aBK(a$o+xDzL23wLyLTweJOss(KZHuUuE<o5<hATKH_LY1PJ;S;U<=uBSVj;v2<_?cs>0s3h-L?rG$8cy4MyX5g%b?<6IX|~UfW7|;V5cu$imMoTUY2EF`U$=VkzQggqvMCoOzXx7d!WRQUWy~_7UaV|E;>VP7gsZ`vrmgyYHk(;P?7D!(`I(uh%4NbTRyy-UD6GpVXJwziK$e8HPj__(CF((IwJlu!WV>^A+oHm>H~3sufdp9Dmgr2fFCvc+ZK%<D%nGb1QoXbQy1=rZ;>X<7rfkJd!y|gxJsBFlh}&qN}$l3RP`2i#4fLsO-+oSkF!w+%P;2u}Mm(iSkKd58<*eMJ7nqh+~>y`+6y~I{fd57v>+zE;hWU&3U#2+pu1$Ct#?^0OTBIt#oD?GD^=76~r$2?z1{t2~%gN&6}DJHzqx&kb89F4ANKdg|qx-EdzcxG7QTpuJnJ;#B*xAm(OJ~mv~s&1qu&xh*Ez%7pS>JKG&XcBlN*02D>4NZv_Rm+h*;Hb9}at?0SgiAv8n+>51E61rerOA-)i&>(5eKbeHMjq2wN7DyVS-O=y8a4eHbQmQK;my`m*yf0nrWwcoHCE?(}c{Ok*_!|?CMv`!Tg9zG|i5Yp!@`u%}$#$)twUJ_Bx#6Iaeot3BDI&|$2P3v|4Q@-VVuFhSkIeRhaIA~Y&uIw==897&DG+iBC;t}0*j^sjxmZv%+44V(3@02QWZ&MV>Ae<+$d1T_xXxd<1Vu1jL)>e!ubm17v3iyH1axWD~y?4~lOd?~FBEK1FUSl-7Hb49bQn}A|_V0OG+6BJPMosz;I}ga9_;1f3E6Z>#g3`24Xt#36W4sTmpk!%i41-J*+~4<xPIBewj2w`J+B$L}DeLq<`P9MW!&AB2YHa@8YLwj)7ASre{)|69?eUY_*sh-WhxYDnI$yUaJ{D(lv}geM-lCCr5vpFD9#p0vdsK&P1vptVN7xJMLh1n!wBlD0{;h#)pABiZd#mtp+O#a4Fyv5$;uX>&%exU5$E2y#Y1-G4P_RU-^v(ihL$+GvHHwzZTz8*olYOU%MOCVLY1U#+XqWisasX{--SXp$p00;{&Mw@_wE+9&y2s&w68t7r3-7QW+P`#vIHx>l2Sryz1x3yK&9IH<WsjDNDWVLaW0T@APx`v1zZV;x%bWIx81x4aoJBw87E$r`5t#Nvp>nzywPKt*0fJy5Zpbw%>SH+(eiSDB>)iywBTx6pHfMhmBXwRy{uQkZsb>YHUYcV}AJ-?ab-qCJA;z2#{Q0{31!+Rxu-$F2W4Umi-GvzPQjC}L<Xg{hM&Vqc0|}2a75ABbK9TEa(9+f2xe?20B%~zj^PC5X00b3-v0=v($|f60JUTMW$zN~*fAum@^kQy_A1Rrxl~*3040J(=;vt^ycdisUZlC1BIBO(lG5^Bq%2e$^S_BcTOnn}*Y-D!}YLBr~IN&+rhap)M2?u1kwqzIL^U54cP5it8d?)}m1$U>`kZ^Na%DX^jtPAc^7v#~~E%uQd;;^frO{wNq<&k`FWYyGRiAytPl_Im_cG>Me-y@btP2<c@d>gLnjHe5~+9A}`D;yBI3?1W*eMD`x9m#^kMhZ4f()0sRHW(i~|Idp`lVxi)kk;mpDnZN;!v5Jxk4OIs6cZ_gyBf&~j>h3v?+-+Xm7fzH8f$q#MDmjk(?Oij;k2kTV=agqnsbqwuoK-QZSDNx|8o7G+DsMlB9;Qfe&~RXzgJbd+XSZgGs&3-4}6J>w9f=qlFuDWh~7Ysk6(JZwoSm!$&<37Zf^lttby5R@+!C%Z+|&V*ZjaEXXpM*8I2Ric;bWA?cUa`Mz=h@@f9>3LxUJU6^vO*E~6LesH}|we5_6qUaky*+Tm!R=6=NLN~aR65iA-`qc|O?g(eix?}+~dxqHqRzJTO#q?{b7$St1VZktZB7CNGVp5KPifH8sX%@`d&XUjh9Oj7;vXG?QlV4?E%&j5Mmr6{HL*sw)4X2t0!%B8YExtw8r<I|z$Y{8srOmipQX+sv0UBCY4DR7H4Bo1R!zKpooFFn>Fb)0hHAQZBgWSluxOrLk~KJU01IaL2bSYj59uDINj!#DR*SM>(am1E^##%1BG|KFmuBn@~OjR^rAFkJO}5t7Q0zsx=AhRc2txC;YFitZN3Zh*cDczOdsQ-VJlu=-TSEH*)0t$4wJ;ZxF{hiTvz-DHP(&&1;c?Coqw&h6`^fP_@&{vCV3)Se1_s~!<BlZvVO>oB};ITBMsn`Ost9-n;n>tC2H+WNW)n)mN!>@N2Ea8`D6_fb~BDQknV)*`}g+q65sLg1GkF9iF2k>Tl6Rpn^Fe!E%p*_Joq-E^vKamq~4y)$@rp6;@=(vEaF?rAjU%p@<VNeiQCZmQS#dUzd2>@^936u#hNM`2@>u{bErehD;P3DI}{Axf(Y!b|vh0dN0nA3e;OWAB3cOlT)(Z6iQzWwlXv6;GQm+gq}ZYoyfBFAP|tXq3^Sjkk%M(Nj`KKtCThka)1;ao*E*tfWnWG>0bLl=1_B?_kl3AU_B@Pi$MpXJ4@d#Q^hN%kyY4{wmABT}5}q_v<S|b4hMAx1-Ts%ulM7*J>zk8Ec&V*|gqn_C=UO+4*D~d{UhA+QgBJ0SVNA)~;(&|3KLIlDpj|{#6YgmNcMeIXYG-&Wjy#`GBYsn0ZnfW3zn*0bZx66a!9FP6inAMM+GV;g_quJmoki)nwb}e4$hDU14-V9yuNJff_`{+i|`56EV~fZR%!OtVXMVs~N?%0b5Y-l4Kv?f!Mkjp68D7=9!0>4*saO9C@zd%O?dh?ced3E)HZ3;@i6<Z{-@B{{$DOe`??^HLiX;i;s<Q^JZd#)A9{e*$fo(-t<Rz>kD99f*|;)^H@XWo9P>wI()`~0*kv;;q%<d4pz_u{jvacIQj9G2g7dR1h1i^X+rJ$cIADlbEzmXU@S>xA^2O{=iZf8@hl*Z1ck~u1KPv+9{a<gw`Y~|Lx7&Reh80GH)9nIZX)8enZfdse7-joK^!!?Q>vWvGE=!e9*EoVq!D2(%#V#rzXv``=EC#HU4k9Yb4QCs!I&?4YVBFIWAYpMXw3`W`#EwPMRIR2G^dO?spbLS_O+v;DXi9*TeFXHGj<0BM+3!+!G(@P)c?y^_0b1wROS(><Y^=kLM<s2zuL1R>X&#O<OE6#P{uu6PJwNBO%Yx?%5DQ;(UkP{I|=g*G3=$rU<_PhN$<WeJB>%a`OSpiK!>k0L%)R-_#}Rfb^87;ubcDnubeyuE2#)Ev{h@zrPAlm&vTe14VKMk!}5Y(BFtDJxMC9}ypWkv6U;=%T}563>8sgy%Iw&VAZd`*&g#xdA=8y|kMjm|JpK%()Ppx6rBh(K$lV<S2uqfb&qL;t&&M-Z!yL~`&`?5=@RFYyGyH(FrMn8Zd1~cHhs2q5IkP40gq6$7Zp6G)laVU^ziWTu$(kyvfcBR&=2*2D4a@$8ErAqGD)w$-Hg;Qw)A`O(EE$x5gAhycHc7}VV{{Y5wp^7+WXEUQ8$Tnp0Ssrwam%$#bc9n9+M7jfoF)2(?t%EfQW<i}#3|hN^>|)cqg<XwhC)K;8_S6FYHP{mHn2()&u67>)}lqpEP-7x|MBDO+L^W(K73G6F$1M5DtTD`>hve0&N04X(3Eo#ctgWw!1GYenpA~$2|aOM;w;dQ$d$)m!H9_NAYEH1)M!Ydxv8fPNdrEgInJkjMJYyubZspe;MW3fOcG$L<|Kn5yWrn_G#+YoSPXtF3_G>r>G0Nm#`ma35kmTRdJmUuPpWl|c_gm6mo)%Rb_B2A%5w%t@pfTEo%*)W)MJe9<p?L&sY8zJnzbw6p;b@$Ba&Zal1*}GRgOMkb|BE%EapwW)WmYx>#hbE1kpH!lE>6{sX&Goa--R}qv9irJy>+$7Dfci@RUF#1e_xu(#}^F?&$EldL|b+ot`elw8&%a(fS!OSnS~(??C<%$?JF<FB&{u#cE-N#FS?e>n>#Cs)mLWv|8@{|AnKglme<Q36IdOd;ODjH;HA2dAqLe5;;2|ae!LvY1Bep!ko%c9LuhE3`Ul3hD9}LK<kI>5|z|cDcHAkpJiidXcQ+Ft78Kn-@BCB0BOM>TmnT2rl)M8k$*jCzj4I5H&e;7O4(5O`md#=pAxLhcX_Z_x4wg;2wquyyi6_CPv=)EAdGVFR*np#gnoM`;ljp)$TSu7xbEI?Lb+!i1y#f}t^5hJR4%eLSa+9WEY2Gmqg+JfI!A~i%gI!dyr~CVGD(ji8Obf=c{D0mk1d7E%lI1Rdg=hE&DZ9?=c0WNn`c@Sf(W!3Ed`Yj_)7+t?Q;;2LMu2T^zv0BrDZ*3{y-I!C<|rbJ$@bnsxsr(LimW!@<Xc<_U^;l37)PXjBEc0?Anm7C(g7JK?d~17G}+GB(K81%XAZ=sWNe*aj=L{L}xMHG=jHkki+UsEWy_=RWCrnbEXL6O|RgR+kW3MJQ_<YzdPmTc+<3K1rnv$-d@ART6xNL$j~?U?vN$BcB(Xjiglc$Q{f4QQ%Y!v;`)hM4N)B@lgouUIHnP8&D0kpWSh|)z8$c%Me=9U`NkiTHND_X91nj?3)jmak4TD6v+%As$cfMYHlLm#5)37JoPbxSWj&u3V6VZIzp&_~Z=5~N=^#^w{U=^^lzpX`%ybfW`z)wxh85fYI*7Q*xf@?Rv&xUzJ#A8C!vcNftr@a$O3UA)D#w8?5J^r4-C731Gy5W-Cx2@Q-A%KhcFhL-_6-Qx9l`s6$lo7o*rL72(`4n2SPGKHtDUUMRynqYjVV-ES0_Oi+dGe0vpV`-eSE~@)h-5I%E6F8-5+_B*hd-pCXaLcXfXt_RMwlsK_Ya3ZntAyk}&Ins~*MzYe_LOrTB5VxtkJX*5%$HwTn;J#~vOKrRNN$o+N`k44tYZw@M0Os}@8PV9v+f3uw<|J2DO_rDWM9c+TotI|nlc*<Uuc#@F%oj-u8dP24YI#oX>i&n1%^w+R)K|N7n+909_DPL3&K^`X;PmYLPQ{S$TyXL|!P77n!1^NssDR?Z+Cu|Qh7%%$)9`Y|nN4ZhWeR8R<3Y#z}KMYZ`$v~PJh_DZ+8zP0dVNhU;l2cr(?c`4D)l0+e-jjp|z$*NuZ`=#&{*g1vh6+s5{9dq7<r1)(+D-T<1widQTs&|u31l!vXbj*1lWGk?LXr{&(#JdXm)PQ3VJkl65!ukCL8JRDP(D6YDu=3n-7C;ye&<*1>QN`$!h<kZX&e{eI`k5%&Cco-UxUk&jcMq7}!_LTpF$;M0z@Z1cVL`zetzWOrb@X+}EVl0ssQSKFOa?L>s0o<9a?|2k__fs)L{CkvGID^uSHzoVTbdgWcATGL5(|KlU{WO6>(!X|V~Jb!X4#uR-uhj?ctD724$S;%>{cfkK-3N85y|g?C%mMtQzcA+y#e4#=3IGTHWU0V`$F-*ANG<px?(G|1ei6D^zQlJ1QOk%&NKFVMrDjIY^*Lo6bVP}u8PLjgC@&_^tR9`70cZcWEtUjFdCI%ib;aYACC#h`O`IMw3L4-TY`9|Mpx}c&d04l$dmgMG`<Os;08JE&BIAFsYIsDj5v^4`<nUR2xJ}!EVQ;K>z5M!c%;>}0mic-d&p%lHVhbmT|~N@#2HCm{z~y8RSi6mge;2wji^U9s?JucjdMmcDlnc@-?uvqp}W5S5f!Zx-%Q^q;r)lg{dqCbETG)A&NouKz%)0%Mw6OxbI{T6H)=C<$b5Y{vHb8~h%_UBLx=M9x_&vQs*0%1R$;AIoBw&0twKQalm25{v~@tW>Wslcf=m=5>C)e>Qu_-pS*!ebf9<0%XNA0oVu?B%+>UlflRz>Rm{wj(0=1EM8#z!M&CMh++gPoMqaYY``phe=sIEq9dF+BpirsLoF%vj0fRgxV5grc$B~j~5vhxkfIFsX0uuE?n*cZyiLSjii9bzKfDAK&LNdks?aVv#MgStYs`Qx*c&yJ3Y(S<-)3aF;hz#XXaH8h2c)O4w=`_V+Pt(R_;bvWZX5Yy-n8?b!mNLwT#s4Qzn)Pm3cFCz9Mqq4(T9d}2<8=klA+8x^Fe5`ICd#TS46rTZo<xs+(pm*_cSQwR$A8G!Ee?|V#*OfBIULKe;^JM%43%L1dVo>vTAoQ>O-pdGcy)RyATxILOAz5a>@EZAK_ApQWOA0nTOGbFVGK^214swYah^xf0iJ#5^h3KVl3G&4vZdi<=4@xgkp75olgZKiaD>%Nm%!Ssn>ri0mAb($D<P|K?*M{E|@Zs+>8!ri{BX4n`-1B-9z;GAUQyZy3%y17)`Hdura@s*Qjql=ePM=h%>*|a5cP5LsJgW@=PLt19@m^051l`K3oyDZTo!jhC45;}mWv@5&Ex!G}hOb2xD<sdOLnwSuLj#t$*XU5rTwH15d(4EX>KF8D3X#h}nn{Yxuz}CJvI_apkl-?aX$?}62}+rqx>G756RR%{Njh?XUoe<_BM8H66@pi0nKr^;fDj_lxe$fi?t7SptpsU?&NAV7G+2Pr3t8;8=^F+E8q7S1z9lkmA7Vd@Q+gbC=3ZCghv>ZhC=<Yj;Ho_HiN)srT^e&D8Tz8>V!bISXDhkSgdqvX0SP78P%f-fw$5E0Gyim}(Kj@V%OJj6a39~1zy#^z|L$zLD&^v=C=g)D*f&AC+o^)hllS#f-4x)4izg$!d&MJxOuC{7@zrQ^YD4&<^4e=@qL^zOPc}(MjkzV-4)Xwp{EXD*ui#i)#<0UGw`x8*;dy!2?a|u0+8$zg^1EcZ%t*CqnLcK4c6`rI0Pb-@nW8I~$ZzZBxx5fIV5-O)!;ALw-Z4y(u?sUrqj^H4v+E>Gz^WS<E9iFHx>C^4j0Aq*A8TEmR-($Z@KqHs(JUX(!T3qnZ%%m56+^7+`2O>*cxcxO_co#&I1<BKCLn5$o{;cby;y##&wLUw^UVx_MaN1F6ZdAOw$0Yr&ejsU?$#Sa<*)Mjjw~U0zoX68L?N^#h{V1=a&w)+2TARs-En0y&7Kwua0eHHz8@qYeRHL4T}b>)SEpBkhsv^TzxYnXb5YOU5bq=Rcx<ii#odF@YY5LFHSzyn`t;>VQ<588_T7BiC-krW7MXshwJh-IK>&E0wo3t8*gn=X0YgD(?T=Wp#J+$%;PDGRbwq@ntDp{7by^<#xR|(s$8W-yK(msPdf#F2&cZpUv0Cq#*R0+&Xq*-fty$rZAv+^)&R89+WW-)!vtw;p0<PADTRyb(0&@=ATEeI_em+F-Y$E#?7iGlQElmY>ZEBPJjM1{n?x75<^JKt<03(Zyq{4%XierpNw`)>yR^Hr3QnO!QJk0bQd!>)mY7y{+`mu?ivEHapqBk5tseyL3cal(hh4X-N)#(!!N1a$_-WM=_=;bd#YE(^nQVo8uR=vKrL4DDQDS4?>ReazDA=P1X?aY1yY?XtaEx4g8c+XFVi|`(#`F6L5Qw7&<5p}%dzsGvF-_wiE2I2WXx`i*+!YEVu5J!+xJ}+>xx4F%j`pf5^Yf^d9iaxq#(Q>>k*9M9IPy7dje3tpuLYqx|escc`T*jyI&RWB_qjHu4nU<8_@@b+gmM1e}leLqN4ql8lG2UW#oQ?{mU5Qjk+;5}cZL7Penp@nyrZ79)X^t87Q{+Yszs0}%XErK33`!@AA04*VQ8HN)8;qK&YmS#s`F+!R-X9$)s-tmifNa2lT9m5_5|~MvQTKK$G!-6*MvoK$<txkt5F%(KPcCYKuSA8%XRD~2;@AK8#_I4#JY8;9Iuf!4<bQ8USW-<G5;!9CWp*5nHE^O}z3AY^8;1c}9MQIv{gIsNza`t133~5P28vqP9-N6A-mA2K9Y8|J9ml~*fLJ$jbr%k#7t%@H-(Z2tH+_n*nUtG+nSAnskHG<Q0`bb=Z_II>LlMuwCC@WS$#l?n5iLc;y2Urc`g^2=z26UkwDKM@zbP_E0dMid;;W?8zTxEc&n9m4;vwW3AN`HJa!_*WU5@S89g(;8@dhvD+M|_IgEjDF2bMAWc}?`UUc7aKKb$EoN>iYO=@p26dVv*wQxw?$udV@0bZ5IqeTkamtvP%XKbD9h1G^E~ejF5>S0-{z0!x*N(I451xP$;1RTSi&LSLsBF_74X{{NYOiXcmp;Yz-Su+LOdNyRkOF`4RDd=*MsR`<QnK$Q=}uwS&7v^yaBT>xUo48ct`vL7U=NEH!J#daMD+Ju<b+2FfQO76l-#dqJKF1uu!7SqZcq3sme$Z-o{3x~JGZE#VPQdPMqM-9<f+kUg^#~1!W>Or(;i4Q8@BqEq!`Pq6Ufp^pU1yjLBH<##_P^iWmbz*|nFE8L)Ns74Q$r#cYTqP}74pUn*&Bo%m5l>wZ<Xf=7lNUKitkm%@;8ijkD&SplhPP5Sp3&`3J#=2iaJ6nYPLGmzbqb6kpHED`XEH%kMi)5RpB=B4KqBol!J5b|MKNsqB}!Y4D9>=OGe6^(W&yWOUemnEB5X%~GzDvMYm_lTQ`@Ee<{xQNz0Rj~mrsdQe@jGV*}yAdfHtO(!mghmHxM1?qpjlsUdfccPW>D}4d%cMmVEby*IWo11Ut8qHP9^5p<IJU)!X!ZaSHGeUC?=_(|s3Kt6EdhyQAv;#TgEXfb=BcoB8v#=7=;snX&<#Szi0wK|xr{dYc?weH5KL3d5^L+|n+3HI6IPDKMbU9>IKS*;QHduzjo$cjv9qr*Y>Z^f2KfdFnls{O_ksPNe@s+}j|Qx3mj*1rU%Zwd5$imk|`_Qba6?2m3-?qQnDkQNs2->V)=(>r%)0UC|?S7g^a&Q?#laQhR~CszlT@ICpEq2UxWMvEH(S;MtVt{g?b%h?AxZ&qrnJ9re!QXAY@X*$+Ik*oAanFN?V(QWE<se%oU^Z>#s&&RUjf2aS7e5>FodiiZ;HbX+D&%Mz|dNRpi$t(#)JtO;bG14<B&r(~m+QU!}{irZ<ga+-woPlN_4;Tk+xhu`vA@v|229<ivq!rvHBpv#?LnOy9g$PS{(ntS2~2WO^Pd5awB4m01ky)$n?_W+C*fKoH<O03J?5b?0h!xS(i@r9$67R`~Cb4^SxTT1ij{R4Z2Sr)<GA2qBcRN)r6Rwq`m@e@WRn+%%Qrya{YN=PNyAG9El%CDjXWP%MLBsKiaTN0}*s+3{WW8G8OuuM;H^xRq2E0uNpD!u$bJ58QOjua@0Sq5l!xK|mH#1j|7j0GwHO@`3-n?V~g;qHb#B)&y&pf6LdP1}ySB0Jh}0-rfiSb6idOLN^nr31jjga>Foh?k%($O6t&O@Q!H_|OPdggMjO(W$#ffJEhHlr*689ZWuhPuw>llOj_}1V?S(#%5sj_>f*-*(pH}y+fB)s+75cWJmUnj9T63KLCFS@61nTfRf(Vy=lHXi7awpPwc{2&N{JK?q7s{321_hxh!q?($6V#E-5#DaA|?LuNpGfrn`9mg0J@<KLI2ra^(}UkB3D`XMT77iZn3}cr*w>Yz^hxmy?T7;2NB;FyQuljn;3lR7zG$&qY#Ab}tAWv)xL<cSut>?O7bN1%8^pJuQ2pbF&;#^6KPrv9m8nJ51|a;34|dc3gL$=)ir_bkkq+qBtd4Uitll9upIs-wzD0iFjtuI=^NN`t+#`XfjhEy~Hc?f`LFFMF`ejNSIp?{-$Nse!heX(nPTROT$k?oBL`7K0>#)up8;D<h^z;;}>tQwL$KqFHc2s>dzXM1L*vl(s8#giDyY-gxuL{SI7Un1QP|xR9KpHH`yaw`YBTS-cekyq%MD<DvG+)S~<f+;O_fod4p~*{|5XKc#|$)<@Y!JB#mps`Sl4=8(mU|A~^1#eB+BTc;;A_EScUwg9^Kv0h@CvkG_27eUkLMx7rBu2}}&v&cNY00wfF|Vl6xpL)kkqwZs~&#ToAEcwl-@81OIyQ5%F8yBGb-{u;RhQ%E^ea=(8841U+-!Dp<w^>KFw(}mb;T`v=UM;&I8I=aJOJ-uQn^NKuM{~WYt1HNrYtqgP!Yf&EvubUiZ^-=--G3y-)W{{AAJfYu`tSf6Y>d{F|)KyiWiu=v7f(<Jh#ewh{6$X0_t~4SGW&^zuR4^u`5Jlz`gOQ*aE>&a*Jv&*w6F5WWhMe41{%R<UkSqKxbbTGRg@KCg>O_1tKM_$yF;B}`&;Gc9UNL?)5Jz@eiR)qWKt<h2osx~7(o;0EmlDgXt0<u=b^^2~5s396jP3VXuKs@?Rd-aip7-g@IPXCOv=7ST+55f{H9-WWSl&Q@HUK{!;j6<!4N2?@QUcIatlg;U3^nCDk(oFUhelxrc(I>H#x+_e_Gv&ojxQ&=evZiA_g-})`B^7AH_%aSPT8DkD}=8+>R?^!Oo${;IC#uMmqa)VO4?8U(|h%Ik3V$*iP{^V@~;fdCm#xvQgK&lRFtw(1D5KU%f)$#k6=bE*01(3@TvW9nZ;kD$@JK9zJ|5Kc0n&^1J9h}1w{lbJ|?)vuv)L{9qKU-7gH%cc`qp{3sS@D3P_N|usg8@$+u=3jH`_*|CjPKsF$hrwv1nVU(yiYZ(fP^iBZvx%G39Y>>M`@W>j0=y(nAPp7c4lZpia2S_TV@*a^S^pOJ6Hzh6Zl-Q`)D4E)?63}CG8tov=+gZML40LnQ#C1tU(;|9Hdo?mqX>!m~s#_;bQ(;Y=%^Dea>8JsRxW#U&cz7{6$kP;0H_AFD+m2zkG$sPiXB{yX0lPq~DyYsby&&9&792lp3Q%4GZ;EWBHCClqk@;BvX@3RV=^<6O#531im=iVXpYR$hyN=7Sl>gh2rC|vutx?nYCy9jz@wx3*Hu{y)y$hVHIsatZ3$b%7(>`2l;+ORv>P^p`cr9343nD-vZa+p}V`K`wHMFT@oppaOgPIYi)3svAgmw8DO5R!#Ka#N8xA5&F`qp?aZ*gMmAbieE?7eRR*6~Cs=WtgT|0rqG4{}~>>o&;-73W7!k89G6s?|9~K@E*jA&haihQ%4p|jU_<#$C2?qD>MN0to4WOUTo0zuP)GT;N|A^XOPlnFi+~@yb|Z!Q9{7Fn1sibqr{bqh{v;5vNfjIvM~$|`VhI|<OB2JJ^hT#RoQ13QX1i4u7)LDE(Sz;^9~!#80-06|EgCo@35wNsFSa<1XU*f9cpQ^UKnGPVC3ITBE6Xbr%&cJ+BHZ2E#sw#W(r-qajJ7%DG&Vf54|8bxSM+oR?|o(T{N#946lrq(td+F=7XPdNN#=WI>`@xaOnxaP$;?e{A4g)>du=Bxn&O0WG>>?DS~a8<zX{2_n<fi^|0Cct>v*7(VY(a1vPb5y{2Soq4U@9)$Ovg>bsSN)VWR2WQw_cPcNr7ss@wP+Q{W#HTWvW7w@uKBBfE^TDoW05N!;=b_ajNr73^DrDjh)(~AY@@z8rIcuQstbh{6?vH<aA6|B7hM075qK515Yazwh0KRRv=C5QdT$t9v<CAu!m4z2Agc{p)KvmVTb-K-|4?pkwtbsv${oJw--M%XVa%i#n=DWUagm=3%~KxOSPb*`bb{>EX<D0z7VOTT&D0SXCtc@=;r+YMFF!3h31)!F{Wvd=&MWv3jzm}GuH!PgAhb~+#h1WXF{NF3)Ftx}_V)zhNcP$V!8+aLbY9sOG-8ii<7;D|A!A5XI{>ea~MNxu`^NAtIeggtwW8-#ZY#C$-5d1=K%i|`R4Q~E`+yisXsj4MQ_zAE!e5^Qki_7!I)_7BI)n^geZg^2NPH1n~XfIPOdT7n$_bpoABN=~D6_Nce~ddXQ(MLv?~56h5hoNc!mMKs9K@uerBTJ(}IYW`5kAe8pcI%B|*Lv?g1dd9B(kzUtnubSiN{S_uHB{kWG=rm1-s5%~)uKZxSl~5b=WxWoJB(xVGIawps1as(B$I4W7uyE^|!7g=zV4A5SkQwfVXfA+9vV0HuX2_e*^chH8MLQf|D7nf>Q0py2aoSJ^fz#EyB_Kl`%qw&v04vn_wZV9PExEKky5RlnzWb!~xSqO2!g0Wi)cL~hkfF{}6p?%)g9G#~(k<Uf`c%j0AFL_+Yx*pTp?8sa?85ya;#}!+Jm;sBgWX0{IS-vI`KU;Gl7^jBiM-#1w|uK8mBr0YEC%N<Pj_x1Mu_Eiv;{#Yq=l~vE5D6V&(Cp>lKvzMu~dpm^P{okJWzNHO6sPPgSETa;W?bdE%-D26+FF|=(sKGmt-fp39MEdus=`Og3Nqoc+Pzy-*I%isaVjN{C^tV5gZ`)gZ~)7ep5Y(^)3DbRw};0Ux?3H9O##e8-UFeIoh{3>*nFO#@qBGiGVta7R~Ci?@ycflCQ$d@oVnEpgogO^Sam@_=<8;@iB}^3=pF8CVB|ayX-OLB?1izS$&L$XuU3&g7p|rgz*@R)5qk&-L(skiC=>GB`ErI8-k^KUd;Ss9@_i6=RzB}B(j^s6{gxxZS>-l%*!GQX)G2C%q3&3_Ne{&&K3)^QochH9};8;oZHZIpw_d==8J3Tn#Jtu@bY@OcnbWJ|3$1mD-_X1^TY;_HvTue>N<gxVxIm=72rHNbkY6kpZ`_Tj8#Ic(`Tre=_Yuna`RlPZB55{L-BuONQ2E<Ec7bmuymQg2e2tLSryUi{{+i=V~gmgbBNoSp~T&k=QJ_+@++&|g=e}<g48`{J|~y3$xqt;@3wS~y`bArOb~~i$1F%uIAivLH1I-h@M9NT8imgWN}l2due7?@R(tJ`p&B&t`PpA;gO(SQ*X(z2Pw1A;)Q4z4kq9%MC*blO;QqK_?Qg;4P;hCIlDZ5v)cXKr2=o$YTR7yMr^JMt7E|?iCC!|fYc-~JOJJ<wy?tXwjfE(OT1+9v%PjFa2Kz3ZoI<Z2sF9C5fjCS&uTa;>$Vrzp$by_QOXy!=xy0b9(Q<LjPqfgjyt6}&4bj!NIjGmsDBz*t=N=_BuP2~c;K21lAn3`(N_aT-GdK+y!G{=)d+DRAd?&ptq!XIO%9_=XlV3b82Kb3DWKFMALpO|--CpskYKAan6@GGMKQMLB5HVrjZW(aPbd*_#TiMQL<G^Tmrd0Q{4z!rsrK9g;JJ>z>Hl#hF#}?8q%r8&To{j+RWV8V|5vv-WOg%;l)$C^G^E6W@^vAo_&zmw+rCB<MGPtlSqiK>y(${@f-Lj|v*`Q>oN2w8=f3@-tFVTjpC8iXKYm-}^5UjRlrc)5(@*{Q1L{4lPewolDxng3aoj0|gD%1D2gC0I1Fo*3AqdSNEH$4S7O#<WYkoGWM$&O*Z3_rv68k7bsD7-b&!7o*OA>OZbOe&v(>(a)Azm)`~K`d;)k(CV*f^vCJq6Ilu@S>RnLL>>O!sVy@@(UfQD;#{{!46RB``Lc5Q>#bmWK(=9W<3@?JQdPyjFOh`KNy#%Zd|<#6E`!bKfZqGpSp2QMT=$4zNZ*XrHxOm0S(1%U0^1asaK>=!ct(G{YedN=^HCW7|U0SE3Pv%^^$Zc>|sSzHcflqvyf$G=w=}>iNK3K@~#MY(#CXY0a3;(rxo#bKe<<M3<SdhN1DOms42IyuLbRsC-deRnxZ7(HRgN`>4&Y^NMvNq&iDTcpY~U-eI;giEKv8N*kgT2gSU9Fo$93*BDEV$Bpg6G5uB)SLmlm#Q;8upt)VS+exzgVbGK?NHbBF2I~slHjimC5m9;_iI}xS_4-K8GzHaobMPM2JCKg?EH!eATBQs&yK(5Wh6y96j_M}DXky1r1;+w<I%IbPvw&u!7UVD59>cP6HakcPuJ>`8`_(ChVnJn4Y^^K~3c#A)7#QD>-ot^r&X;yu16>B_tX1cQ@_9XecD=UjBB9Hb3gAJDIqCzUB4?aJ$oy11f13i+gNr6)#Y(qE;!dpxBu~~)?r3yd4a7?GI^Y!@kub2XD)DP-yTovB_%M_3Sa|(ZQ5=GtDnD=|b?>)9%Pg*7Ns4}r7#TN~~<~a9o>Y{0){Ejs-^!yuFyh38=Gl6+Hn%uRE6seOFZz7fo?;5Xk2LLH`_mqV~>f+Kn`5qDk!uDWC+b-FYKtiri_y5NxYS}ykGWCr`SVL4HudB2o(9!RY8BXrLTa_w>g2Jic;$Hkc@Qjt!hy`s0t;-5IDEN{bBBXLIc9r0t)~!FoT(i(=<p11!d?=7TVViPj+XNzb@8;mcakyh@OIA6#3wa&AXK_1|nyQ_{z-Q&^+tOdnf;RKTAEK1bLJ;6F+De;KF}qDFgcXpAtsoS&B0Y<2FLkq<NYBm^+&pbgCm8kp#;b_Hn@cgTKP5#qQ-9rfDD8A&RstXT_FlgAIh-J4npXvH_aQi}Wgb6W%CAvFmkd^TDUwwrQM40&HDKE2VPp)C`JP%D_IwqrfDU@nh(tYBJ7)dQy|Sj{p9Dk-|KV4Fg2w@=fN-4oY}5D(aZp0`U}!hj*`W%zQ8eah>S%;Cww;~bNw4?9gOI_Mt0aab)E6u8Ii{B(Cw<$Fvh~QN9u-AyYCq!hrVpoeeEI`(`8H(k#eL2F5qsTF*-CTx_oeEuoZ%e3&T%QI`CShJ>joG-9fTiD%N+Vuo2e?v!uE&H5ZTWq{Mx?-XzM}W+nyq-=NTJRGQQPB_lbK_ba)3d)=P!oDAk#^j6bAY@W8uWTwsBrVw`hW8=!o=rzN#3qL;9o`_8iBC`zP*n01~O=;e&=YNnTer;6%<hsnz#okjueM#$Epl@(^94}qwb`7N!;49x4g@i0CWT25wkoRW98)y7sdDQe(B&!;PMTwV;YhaDX2>+1V}-z-&jCkJBojSM(R5$m7@W+H?H=e3^FD1uMG&B{T+<~IC)?`o4C`m$D#{_K$BHE^roIDbDm(?HXsWgyXLupKx&py(T?e7}CX?`m1;Cl=gcJv_M%CJk13QFwR>{*ANwjxex`YL$_<T8r%|ol|6IuqzS1tjeV=2|hmXvC34*h&VaHWdpGVxEHS6%K_3VFfKqS&KnL(Wo#rGrQuL8UVb+?rUo;vP5J4$Jc@yosFVF@Wq`L4OW^l`o*4p#-kQu(Qn<)&NBLBxd8;+9!q=LlNdF@b*quQBY*6Uo+V7iAfO?Y&pm&wL#MqXH(zrfIC=C<}<bGq|jaWpOMtUb00P^`!0Uy}|h<Bt47n96^@<I!gbH1YnN~PIVMXSt3OL)nK?-idXZ&S}4Ik+*?X=0XiC+qE}+j*sglf;wBvdD+<so%lR5-}2;wyU89A7c1-_?C&xDf&}-;hXEH!9=dnISc}$q^$EWKX6$@538?pFKsI@)+$JrwD4=)YBPrcGb7aG*Yd3#JP7Dv^yVZ*@-RC$|Msx?*6T{I=1B^G`(8Fdu;;eWb77^}^(7kxr)6b-!k4!|lOP!nmO3>ndk`1xf-zz<_8G6HAA*?gtq5fPWbjVzfHq3D;Zno(E>aXc^jy<wQ*)|h!5wKXvX$GJ=G4*?ZISK*m&-~O31S!>?S!gEeUgps&&zWV;%2cB!&&e3C>ivKOlI@Ya#7{Y8fX;p4WP68Vh)~mcW!_&dG$!bd`o5=w!E4#tKo+3qLZ6&5GwGAD2@*~TM4l8G4f>&P}%D0QMn}W5rl#5;wZq|#dHjiyH9QTTygxIyh}b$5UW+pngk!y>+GBLVUPLA39`Dn2<njiHg&%Yo;zC2FRg})4af#}ZcYe5Il+Rf&DfTzN~C}<&ODOGLq1$b46Q&+FenUeazh`f606%)Y%+1B1#OVrX}`O{zN`k}SJ5^;yJoJdH)awioAAKK5rQOVcII>?&U$wLtO{qUVj#T2ndkYx#iH!Jsf=yM22q{dj_oH;N|JGHcw|U;W`tOWN|-e1PxwuO%J^(69R)gd9pQ3i_n92z&_-l*<gV<ZXZD!vPaUruBf19fxry_DkLFf2K~zw&d1$N!f4!RCP)Klr5K32rvVVd^Z5487NZvqOJB(tq_4*i+ce48Qatg#a;ljS!2!i(iWHtL9=$A;A8ir2@n?ZqX_<mmUik5cfD<sS^AuVV@*DDHSkm0<RKdYNH07JyrNxYnyx@|K@p_DotD-dQYI6ElAiJ;z?IP_-m@!M&%j3)Lo%h!^+3Hyft8Y_*eddr7*Aw!R!!{@js1y$XPb|^nu<wCWtOIZ-ziN0(9>`EuHu;oEZ&o6TkktG@S-SJ@t3>LYm+r%sEp1oto4!iq<vpF$;;%MM*Pd3JvImKJYKJj}jwAtSPP65>@O@kINkH1<z7(*d4o$+^%486MRFL63`J-VLh5Qi~^-vlp~wia}_{__&EYz*lv^D&7?TTHw^6x34J(N2IA<9=i9>Hj?EufoM3Q?Z}NIyE|pg$h+tg3fX$u5tYBrx%D;3tTM3S!wirO_Hoh()68O^ds@j9f|z?nMBEioxd6>=!rH(7PtT+MY(`Kg(8251H^J$8Z4R;kat?LwB?6K8CbB1YZi~mj=*8~Q_^ENi|64K;$xd@gB5;VCtLu;1^r4wG!6yk*I~tJJBN-{=9^BqU(6_ea#0l3J+XhJ7>tC-Ih`p$4B#r@KykjAZz~~fB>=MIg6I1yt9cg#b2kJy_?6iBtMCw-tUj^D!dg0$sn$hD_%iB2?tfKa5fqhwfJlEvO7!roHAJlgsj%K;Y;#)sd*UTbDelD1M}t*C?N(EHwpUz=W=e9TOchrCzt~T>I&pS#lnX+>s5s3sT@kVGi)PthI-pkv<W}zX6ypC~MA4A+#bF=O&1D1P=`2ems7Vu9QPKp89EEDvEr61QOu_z#qk$tz{yKG@L%o6Z)vU8X7`|j&_jqsz@S(U^#9FK+pewNY4sTtu+o<cNa^ohf(o(GH!L(v;RTEdM^mJ{l*u_?!utPLPiuNuQPr^{o{$i$INas|r29vS}K5p&&g|=#Trt%ZJP~#HUXTE-?U!fj_-})?B+vgHv<im5>>0PuOYLu69i+aPXXaw@_tG>@3JGl~GyiGBRD(Vh0nm6wngwL%p&-fP21@4ze7ACI1{q{r%rZr5R87_K;Ug6Pi5$1VX5?xeJ^FKCDVo4j)q@}3*E(qrDWnFqRnYpwH+n}r?ZXrGS^;tbYOe6H=4KAV|4%*09tYjfsO?t(*BncvWH;MA^*gxA@s8o~jsnot$2_B*Ucaxs$rVWWme2}OtSH$kQpVM?XR5lzHid#}}b5z~n9*(8Juh4_%Qzp<lPKhkSH#*9Ek+paw8()5r_7vu99|N)DlFZ>-s(LXes{v+Ahsq}4@@HEHVV9(tSSdgNbBgBI8F7_arD$cn(W(ALlHbVX-(<skK7c;EdRGXmR$E*_cbo|kn?Ar&$4~<5Ecf*<t|memAahIcRf#~|dL3xYUN`$hFm>7KPPtBgFvQt#)jaBB=@<onL=)aG+QL!77x<Jl`l@smbD{Q1)+C&v2bc9Fd6zRuh}&<^+@&(`hw9;_;F$66|2}t8H8?zg5rBvexwX3>EyzFU%KyMg-*XN2HTB(+R`PA2l51h}-QF-NbNd|*>XQk>=pvqZ+J5)qk*@0IALz?hl=Ya)mfs^!1j4(~NK+9eo3`SO(;Qb~9fB92b<+7xZs1QxB5e+IatY-^1mm$Kgb4*it2TXzvX$lNn2?d6mQG@rKJ&nnX5Q6norl*gVPCB5Q?d_QHU-fG0th%IJ@x=IULS>Gr&7QXIA55Y$V7yN*U1me8JI>r40l6HkMeuXvW8i$vV5c#I+%1ip`!+!7`;CaqHyRd-~peY8oKK-ID@dl+Oq1eA;jo}3CN>>YxB85j%PZ>v|&u+2`c^|=}}*aT>74ANC#yKafh~8S%9&1<7;0H<_DwYpvyzG-g-1LjmO#E_>g?gw`tvIO|T;N=PGiuGcMo8$C<{4JiUP&+XH2Inn<VbDnd1jYX=P<<_+eW1DpVr<k<sc#eg*NMtV*@I33DF(UB12Y}G6Gl}sQdf?D?g<L4tH0DyA^NTC29y`xzq#tK)euqVY00XL_*8G%whgN^B0;WE>q;L+ecfJ?;aGfs=n>O8{e=CXVtCTa46zKr@pTy9d=jcqr=dA%gI28^bdrSif<lVlfDcG2MV#OvmuoVgYv{f;KQI<#KRSP7M?%6VJ23&p7b4{nttr``*QAYyM4))iexA7teV*S`@++*4|s!o_t<;T`m>zCS3a^|lKA^|X=}+A0KZ-3JV6KstwawiP*I4j`Vts_>f0yK_J*`0|#**Jp>IhqiM{f4}Odz!jv#E`ylsnC+M1rl}m>?#LAYm!Rd9ZZl!Tx>sx+hO!i0X`aMWQ=$O0Yg7NRGDb-u$r;xQb`KswW!d*YiXiNV)$AEq;Apo`SGharn7V=qB_Smq3}mU3WG%(p>T+RafGLdIUi-E8lSL_TWrMH4jMj@6^3sUb^>DbLnGBh?fU%((MkFSV8`wj(yj*llweNXad6aS~>7_t<L?4S?XvpfQn+R89Xya8LaY@xVV3h0edt$E<S*;P}1r0D35xeDyQ(c|yCukIYr)dC=3#kD&b|%wiR*KM9CxU;b4lnRNRJ#aI_Z{a<&Bzd?v#KSGr3&5>nYH7iBnzMqHa@|(1EzHanJg12ZK}e?Wua^=*a11>odH2|vI6!UkAsR)TOdtgw2NQx3&85`Pz3kbsVJXLLF9y3BOa&4w=nOleFvyefcW5S7X9KLY!6pixy8CRykA@q5dXTvG-x~InlOw=%UOCaC!r_9MoQDr2GyF8DLkw;1Z<9x5?3q^|AuW7V^Mb-`rlgRg}%@m3=V4>`Zokc$r(7>F}r0@nuJ%vYZ!N{%Ui=$>K}hEp}dAs##!)D?2N^1W-t8d3u9C`>M2POr6o&h^ICZo?K)p(OsPhgfY-0QLv>t2|B``O{<g+0eMBU^#$^Fe;+2xgv{CPgvLGt$o(O66%$;KMTz1X*GAJ;Cp`6SVju<2+9pN&-5(A{Sz@7S*_ydlW99|{jN7F{cIb5<46=V-a*SI6w7V8%IW!udkx4Te!W09Xd^?g=o&#R#+X=u3eZ6sJZof+Fwkp2(J-a@x1)sf0Rw@|10v6h60LA`ljx`Ym!iK2fx><fCn8}}!<EEyYUwa5<dO_ZC6DMnDa_2w%1YYuB6#oIEL?M#~R6?hkI1)n1}sh?ro-p$P}pN8ij18_UdUK2QeC|abpC!;l^&H4>5Chph+_n!tu&nY~drbvkp?Yqh-6}Sl(qtVuIg{GTtY?!v(sy}WzLEL5|sYfYZA=1jcdd|R+*cYb}9Q`RBjVEJH!_!Y=HYP+zHr{zq=HrgHJ)wyzM`{U*UTwheHhHy1>f<4~CWFIym7>Z{{!F|I1lLA-O!iK+z}JKeVX;c#c1??vsp~ymvZy|xb5U!5AzLq8VS0Q)MCh|*M`9uQbQZbVzTN&X7{pRHrZ@h1%TMI!yl=#rh;<bmL)^=zxUodlr?%>gyJYWLWqs9rI6$I2_cI_xS&Y~L$mmg20-4Xd@RMwN>-~Lf!ZrWs=Z4D7Xbc5LJ$bE9$cDUtvNsj+agVkTYf-f$zIFfUeW>D;2v8nLGkY-t5I5ntA7;2C?}+u-@oJCXYv<iLhkC$ONG-0N91-qs4x)xkx`Mw#8hUq0HIkBYAeD|6i)sobSM6;La_`!yP6{C3SMVYw=6q*7ym#?Zy&96#FosA~ZfCQW$a0={YISraRW}i*<Z}5SC1d2z65G6I^za1&`I<m=O_l*;T5&YJ*h~D2J9-;w#BQjf6m^!Q49j8Qq(Qx_D~w6lD&h+YMnaz7MyyviV2(1Vmf787Sg3r1cd|QQZL~mPpzC^p)HY*LVWO<7w5|K?D8PNUz~Hw)pDv}5s8s@E)h@Ad1hNGPj}*1DnSCZ=(6Wcr!4HOm@A1iB8pTgq4kxEK_fNQ_^ZO}hSN*468}@lEO>YE<Uml6zh*1O(YQ?Y=C*~!L$0G+Nun)#wY4V(g>QU;DBrAv6<)cL!SF}w^hk_0`4C|ee6z;*l3|OPqjeeW@-rA7NQR`#68Y`uy#e>J%r$TK>2+-^|*$6ubUawT=d5a7)rT`^Bl+1iQn!TEdIx?krGda_>)Evq92L!PA`90yk_Z5dZn^?{9X0RLBTIv+Y)E3EmKHO^we%tY|SaAUf@z;#>BZf^Hcz%7%j4x<(S#9j8WuGfFhQvBK5um?Q%uk4?Q_|&teoA5gAY_7CEx!7F;WH2ZY_)b&K{1enj#Vwh^|O1^D2kpTtt_Y<r6DsVnn%x%Sjm&6_sU@%^?8n}wuoZTe$zP4qkEoak_3<!7f$a__KGU<_<(<d2SJ|o;8`U{VByjZz2R;nQexb=FW=MMf<b*(G`1aa;DtX*<wGY)Yx|Jpg`mBWu;;%)BdK1(oqoP1m%LVi9EJuVpK~~GHXAKW_;-_(KxZ~mf+hHOt0?l>n=^vxy?ib=<_W5<Uf<uBTc8*Af&{}EQgK29F^Dy$w8CeJw{Dd%*t5b8E}eT@14+9edwBL^1(jJb=SC=w_nh|rk>HU-++q#h8HM6bvy3j}il8ZI!B|Qc{+<U@3CZ4(%{E^r_Qih&*=~-M8eyRIgC;rt?*}hasr<td&U%qJ@j5E{7pMKOmv7h9-t{42ht)#zb=EWJB4=^a=78MT`I{yqt<}gzhajUBG%MO`6;-mH6iv*vc@fw-s^l<*2M26WNj;BaZ(1Dt@fN|leOg;qm_<cZp%s?*oSb;<Z&0oMV^c$XAW)!kPDWLG(8-b(&buDyxm6_Z-mW&q9&E{-6{?p{0)XjO@>kPRhRRP3rgg2M`7(nsnaO(0={Z7wky+k^c0RlkxJZpu&vAhgat5VSi&LJ#sI#vz!jU%dHk-n{PVJWTC`Ltm_`EZ!m{~s80O_1JUJ_HZNs>0akqyI0uT*wiI6$I{pbY?5&XI8MQ}n^2Ap@R7c9_CCK$0c9IJ}U4#asqqYy-pKt3C{<k+@UPCDd7owfu)jeBujgcl(??>A6~tnck2OF>5N)_fiTK#`|JzhRxfxj*zp_$mpBDIj|i_0&{Gw^TPv73fyrGCSRH9LZS(z$XF+>gJ?D=x=B6cu&gxYTVmTC-5}~?L8=mimM1S~w-g)y`rXR_E%Bc2qwAOQM9x|Gy$r`Kkj|GalZINXV(!HNv(hGmI>`5bY&FAIwWoM~jKkc^Lg!YyOySJ`1GwbRNYl|J=DT#Mkzsr$3u57%3#|N(5bY!TOx}f-gUaFHLWPQeB!|CrHX0)5G?11!P@)E@@?hHrf^FOIM+tY6L+4>)u5YZ5wComOD;e78I2&V+Svg`^z{&iO_H+@lN~gj1{T@T#!#UVx)$#wo<axsacg+duzK2M`BO%=n`n~GDX2$`UK)57&fXF7wYpyOcUwp2CR86lJ7CO9_tX7suYN$w@o;N-3;XhO0B+<IY80S=dtus%eP(<}ghQ(FH?9Xnil=Ph+v*3$${{?*?IdMAdKEZ`^-9!wmOov13ZgE=A^ou~M>i?YijpI4fLZFPA#>vE9mrsh)5Q<xE53i#DNf>)uJ0x75-T=~Xj>yuBZo?%FgWln(jgX-BL&V_Fb1($D9y^n5m?nj!vaVK~&nx1-b%+okD})_jHitDH7uh;d5WSWrcaVUo8x*Gp5^X*;1ksX!=0g9_UxXwn#{-)c^NsoeiSYL5(6*~!%TmRP*D8;5q6iaZ%3Ihq=xh|0M!<D+muhK%@SC4laueODX$hcET!u?gcx~?5t=q-2A=b`>2N#mX&b}@pFI7IyX+v8#Cx#hd<)lQjwmc<yBQSn!w7m&bbsfo4lMOa;w~XrcFV4HM1hdh86~p7s$XkdQ(ZUiOVu=xG4r%e7?Q<Zp*&}>4_}Ax#r20LzJw^I@9v&2<Q42R`rQ8}h#npPAC~(`ByBy&)mQ^oZT7zfRD7v$!<_pY_Hr5L%#UTG&L6_nkMNe{3GGxHb*ts!OgS?BocHDx8eqWY_zTF|$!igFxQb`%M;qk;7As&B|#Tm6X0aO4Xh(=4dmnF2`Ai9$YrZ-?3>t6ZsmPy$;VWA1N5&WX=P1&rLTa;k>LGl6PArh<D3^>V`^)D_Z*Pv_>b~Re{QWGy;OXpA|Ui5vzQY3d(0xyp2sU_2}{hI0$z8MCr$ejMswkenoK#sMcahu(JNSRo8x>;9Y>nyn8j`QVi4)DEmAy<-T)o*NNE0}kogAQ5t;R<AgOUsD2@t`n_s;~3V#fh=Z_V$Y%X^{H+9}a{BWA_zPFN&C960~ty6yH|-?Dx34@-`btmdnzhHB7p)gRX#)5$hE-4aO!f20vUf^2F(wP2pKJ60S?f@G5#;VpyAk%5XTi#yMA_9(d`tl}U)FP^RTeWk#mbjL{)%v?wFdll>*pvG}cJn)r&&)QGpZax9ldMNRAB0}KP+)c}suscpphmksYB67SGA<%nudamq!3D;Zh{ssBy;=x-`^8cDyZ0d(vp+R`ZHd4Lvc)|;9o-=(Mr05pogts(nsM@d8Ozw`-Fz}(tfN%DqsAYt<S7k?jF1Cv;TbJ~qk!;EqPB(Kc=s%CZ4oMBHZW^?SQ`fEx85K*m$t$%~yRJgh@L0A6ABOC#2Es^4wX6;caP>*l<d#b<23=tE|jclaRxr1?}jzfh;ujJPbk(bhQ7o$DbBK41wmp^4t3oYTvp3UI``HX%!W?895J3?4+{k+=7-62D7U|!9PaZ$IfbXVCvCyBq2th`6#RBZW)(vZfConBUe)C6hj@L~%@jLBTKbp`uzpyfe6BTQNvf~=aPMR5bKK@B`Yu$N27Ch-f4Ld!^qL6!Y0BD+utp(x4BLQ-SRt1Z@?Tcza1su-}3g}f_>ruPd!QeeNyS=0C&o*zrjX{&M`XS5*f>oaZISL+1v5)Uq-DWm1LVi@llc+DV8w=Xs;=Ee!Nv8?L#yn%bN+?+pUu~I<EM`9rLmauc^lKXUphHk0mIjv;71xe{_jr`bL^UX5P8Lr!m^}1-3!Z^6#&jjm5+=DtHzA<d@1+AQ(0|ML9ZLPrIaiz!gze`<^POp5717;n6>%I7a(2|A1EFn&fC*hgpa|-HXnRl5!pLnJu6_pcjZi&1$J7VeIx}tA$!H@q<*?PR=RYQx*rVT-TwDY+JZP%>%Q=kk%<b4uXPFE4Yh(xPClvnbNmHVk!eo0QmO7!cUrCsPTOCB;$o;+W#SNQaj!QAyCK3IE420HYg;-`olP(eGnoG+oxIP~YwOCW9LyBpm`oTNR{Dq4n@(n+7rR7{B^SeSpe6R_Ny-q?#4s(u|gEGB=>P2DCT%;1}i_U)xUa4Z=0B_fm8bh9L%i?tlsN%AjDTUR%muw+f-V!qL$Mvkeo)FJHUOY*)mzUc{Dd`AjWpK#brHXtfsAK_5HZQ+H88wWJ>^%X|E^bl57Qsj6NV>1<+>Rf{yn#e!D{2W?=J<9QfJ<Ry2j`r3^9320Q59R-E|8nJLtDwj)kE*zC@J^0q`L+fLMT?y8EjLirfgDM>+$a1;N_lH8aAHyi!Wr77gyq+iu-Uhxmenns=F6erO-%s_(4}`dY}>J;O?Jwt&@J^+cil6t9Xhza9pwfvA<t<!uZcGq4BV4%zpR6K>d7aAO#OV3P8G3G17(x*oxrdG&aM#xf`nvXcF=atAbD=Cp~`>yxzy7iFu0%tqw~_jdA-veDZB^wh7!rS-u4;JrmV%AII0;C@BCsg5R&TOWwv^`I`tOV4aq$)4?b6H3B~=3Xw^d#iXo65_fh-+v;@P4vz^*{NDl<z=2iHQ`G21{3kBfJ7r(R)SQ8yd#Vq%~S$yj)wUY1u8`=<27~BRgl?f&{N|0KWv^`KSec*=%sHX-Z^X|xGwuP=pJ=$QiJl>o|3T+cnQ|bCrok^n5dV1f1Y1FRfYgD@k2swt$e_~TQ-~#ouuB5F9tEcfW{ze9cBopPh8N3^0JN{IAnBzFkTAcSISh&IPDf%_UBqJjq?$=$BVK8VtXwm8*!0<Loj8`^oP2)h8Y0mnn=tW0mQIWe5AMQRu`9!sSpU1l8b`U06@6pzBK@XfbXR8rM5&RM#=+y-rE`^ou|8-1$6sD^%Vf!%TeQz<IBZ!rITw_D`JTBe8JoETaQ%?bY%cz+E^Wj7`PIRId@>a?oZ|7HrHdlWpo^0&3YGP+390q(#>GVl0w3D^|Iv^3hns3GCSDc3iHvZ&IFM8=<Zh$R#m&KfH6!8*Pl6_^8+h@sDv6bbfd4+bj<&n58mE2QI2y%|F#w;+<r<J4^KSq3~ZZ6&<iP3`{(=9q`53^$e49hj>nQ;FFT)mcXXanXJOvX3hIn9Ch{}=OX#XEIV>X52uKH%g(A9H28;zEKR%76Om+nhbWy7VyopWhV4(NmfW{UT>Fg})4Od`6h#*G*(ekoJdJ-b;I-aElwmtvwx6;cv}p9W^p=yr0o#8q=L$gQFW--ygw*^e3^t*!!&^&UDv4ioZlcskRbsaV@{Q7>8z}l-iev14186lwMRpt0>}4e!mm7Fxus~Q4t%ICgwmpy0Lex%ePCju1I3M<l5Z5yasFN3NEAKCxL7)-}rIelC+2>p%<@w24WsR0Rvm=Qamyk2K6$$dP5e}2<BLz{J~pP9yc<MN<`lJW`cl~?`)=9hwah$#YI2)_{!_2J3+xx*~bFHGGFUNc+@lgQni7L#cvp;4X1HSbEU#5y%*eDKxH654m*U7jNA3S4v}iF8~jnDBP&T776HZKxKd*=;!X&Ayn5LJ1gQ$<?1q-#4jWT+6Q>n2>zZMKj$YAj{>9o2{a(NClx8z;A~6$$+PHAh+p^{EgM~{wv&@~)?<O?c&{+|kI&W@B%g#^lsm(?cXl@<Xox@xste$&UJw;!ho)nX+cssU*W3v?eg#{5;t?xM;E#JCiOH*T48*k&uzU|T=FNT~778JNWCmddK$aju~oi4zBfEee#PlpN$i3(nFlE#UkDx3G7OK=*`Np{ZWq^XMkGsY4J*7dQEzST69T-Am$qKu)jF&t^AB093)8KqQD6@rc=60pg-BA*kd5-2%SCa+vmZc88s-A(?>!8dijkHw$5WA?`;7pojPtr?8;6K1RWiRz6sf?=(OSv<p{b~F4CK~T(xYnm5~JnsTNqXf#8<4|KNFXbbEyJmc@z=}?6^q$q$1_9JMT+QmeLop<19hp3O&zBl(x)-E7I6YyMqFz7UVq&*Y?TxyiU`+9~ntV`3{!`kX*6kV2{YCX53I?t;l!6LUcXo1boMDdGgVf=zI#=V(-L4Kyx5T#_ij#v|`p2*j&xCwco`~be?<Phw7n4g0)X3d<e+2$Le=FbTjn8&qcBt15zP&$q+`jeN%}0=W`XL8p*GRZe6dym)G1wB$QZk7T3@~b<-7f{!w+tIs;ZbnnxO2!aqj-^Q5i!#J_+dA<WriOF%g)J$2=yCud)ONR<^Pe&jK#9AB<i5i#kmbQ7`sF0jQGVyBv13%0D+{av{wlzUMaG*H|Q8`>|z#*s0;p&WvkiVL*|{(Z~$8sx;0~*B&zPp$Q4I?jCOpzG)OP(WrH&v`zpeHlgNXG!lpMoa~=P|&5gwgrhIzNtJ%$;3EXhLOI%l4Z+mE51M>o2`msf6+T_0-1RGTF($d_F+S>VOVZqiY#xt({&c4izW~m8Bw4wJUTmw!3y$#=dubW{XlTvxB1ahpi^+A!OHt0>@PI?A4aeC1D?3l{%7lhh#8PlB2Gc-0WK=zHA8yhPDBZG5_w0TEdB!13IAO=n6tGO6;mD?ZBcrgsnka@m>Fik#bw)YbxwxAIO=OCayQ_U>p5k7122w*Gw!{IpO1;Epoz}%Njzb)_Ajrbm+65AQ)Y5&~)!x;XgRCNlVBysm|h{WyHsg2E0W?3564p$li-9xeDto3YU5OjULYM!9u>qVn&Gw8*MyJI4tYhEcth-WkqFmQ+?EnFPJUF!9B_wt|&oH1*Pmx2dWx2ZpCc79v;J|+d0rrkyIBAH;DK*%ZH2;7)9_+7o)B)&<MWvS>ZUFH+o_lRv<s;aXKaN#@p2AzzeIZ+!@KwECNk8|=dWe}xR@$I<Wso1P+Pi<_cYcPNi{$}och;%U;c+WU%=XEr70eo)WAI3{c{nlq%@n|U=DuB-T&uzx!pmk5^djsW$`*sPjd?PZo{IM+#%fU`Z^r;_<4RS`NgbmTSRUc@{;Wt?(N6A-5or0RLiFNwR5Co;4@%}Qi3+_3r*k#*{sMyK{zdMuaYw!vhJJMRfvPN+41i8tz&<*e>Ca*-jN-IEwB+f?#(NO)r5%yWt%8?`JGT6e4%b!7K!8Wz7atIV6Vzr{WW%cczpqfBqhj0)z!%P7tIY`92ZvE1pV?Rxj+$@hFBqwk(nlv+KM0&h_1w(b6%gH&EbiU8lZ#YO%>$z-C(?XCza_Ob0rAtz^9Y8Ik`m(zJaP@l5j?*LjO?)!t?r>E+@B5qO*oLm63B`;fzUUI^SWE}(6hSPsJpFl&6#`OM@#d%x0qkBjJJ@WHTUWmm>*75TYIcA58#g`k=^m|}L$N}wLTTO~nI~m@Erpv9JrQJxm%RHMaM5(oEuUCK1#zw1kUj-}HPwOXaGjf>9QxGHW_Us;xP7_+z<a83iLCzsdiPgcjry)~FG-zKrM8fD@4KO%uih{wbJ>An27Xa$F<uA_vMY^$pQuebRWf?wpv6mfCRVGHg&O{RHojwQ;EI<Uflp12zj0KH!5nx}0&qz*fDQR&;C%a^gsuMJKM>5z1>WiJkLsa|Dl!P_if5WMM37EdGlw-i6F-uaf|U;nd261k8#N=NUF!#({fNQcArlz;%WTWjF?VvZ7u$9eIo9*pgH*>30#|~JW)h9`pt{P@3y=iwItfSz#05Lacth;t+RDXRsqie@G&?lEYRz;c*I15lxhS|})tX~i=$J0gjHD7nYRmZljo_`Axd#s=Q(K?a#rxDddM&s5c5f+*3uCcE{_cX#PIA|(Q!@1nXtmd_pqU5k0mv5H*#LO9XO+OjTY>H*80Aa_sF3#q4jhJn8VMAqW8*NM)YhuapuaQpfg7XS*l+1*kY`eHC!)#g?@r9Gz3o}`b57okB;gZORq640G~Cj!bP~!iK+~{mX6t<=rpC1kc3d2bjOl>0S#Jfv22yTBw6--O&tk727h4Zt2C6w_@UTG4<SsO!hULKLDd$aEjIJFR#1`2^G793C-Alipi5$SsmkLJoq4TGFrdl9a90T?{wepdA6Z3Yz5V-~^9or+}KI^3!<zJWv)gl9!Eu`Agt#SCia<ucyl5Jz-rzYIWwCMjMbo0wwe~Grv)U8jCx2rC1v=*=}Omc~K9eDOiwPu+<I%@goh+HKsHCyaZVvda*1jX(siH)7|dMOq&Y>TNi?!fKjxBqtS6@o=t7PJ!cq{qfjs7Gp_OU70ph5?1eHd1AIdj-r%nF_wUmM2?CT@l0WI-RnPHadAk(Vnrt$)5E?em+k<<y`;7Egl=A8+myBsqY=q#8#c`fUzYF=|sC~2pM-F;}{u@9pf+&<R02?LwB__)~#nv`ATWun$UD|z-e`)?)7!j@u(~QeBRwga}|K@ty-%Y<XBD0SYO_Jxj&GpZcgU(5v5%LNb8vh02p*f5hbdak^eE82Q&eym{hr=YL-$<n#6m~%6y!RG*0RCDAc~e5Ya~R#51a1>DYJ!CJOeX;UC%cQL+Ve(x}(pJC;c+x`6zvQ*qNt7jSFtz1DTzx~+9V+g?p|MN%q>T8&v$|AISq%l?bXIuxFkElB8eje}Vj?!7Oo8dg6DS&vUoJQe6%eC~EZ9wJyBgLopCWel6qu&MSV;UFscxU_1qgqV@>K>;2e4Uy%J(*IuKWLwN@XW}H3b;gjCp!y$Bo5N(8(vU<)f~ZLdJ3QmtuhRIkR;|S?KNyt0r#!LbL9;9+Au0sIs^z*NcNImXFhC!%G7|wWf7U^C4RsfHpHE#$%?H(6W$5E~3>E|%o(S@sUt8m5O(bLtaVno{%&&-i{v0rO!cK;F;BE~WpstSZ{b^Cd+r$3vBmllo;;PrFO#9+9meooy+~)KpXx@Zx{_K)Z{iFYVSy4dL$*;_4kNd1+8WLhBa8=;2={MzQ$k{EhQlJQuz}AFr;5T!pE4%nI_Ir;5q~E0g09P*;;E#bQZy%23Y=(+>vb>UD*fcDRe1~<nZMKrHM6qzZ=J}x|zzl@jfO`(?=r;Xkd?_SoW3+v3WV!&=ph6RaYhAlg0Vftdi5LU&79<!;ot<5!C~ZvqyVu1V-GctZGYAS8U>CnRLOR{V)YT<{1x%Z?jNdxez7TXKrT<n@&<v+Fbk1>G()rq4U$3fLczs&!q#&ofh5;-hNC9ObA9L*3JZ>DWMZpZt>5mH`_>0><rd+0oTCqbE*0_WXhUPP}7MXWOK^H`^lzjQ>{#FD^KXYyjwV2(mdq4EwForLb-b83^w*}x#L&tj?m?cw#>Jq*cGspc_blM+bka)aQj=?>On%=XqXXBWU@<g~eIdfnQp;_yvb7+k~;{O)*5EkYo;B~=hQI#-6ZH1e3hHJ+O+g%j=++O76AiI1neBsud+iVXA7gkg{P*<tZkUoyUO-e%^Z8+!<iE;En(R&BWm5wWLyln#5_;)Ev3lQCQx_7C=^oX@=`I$nI>hK+F{nmCw4u~@#$zGETJnneC9_PDNy)xDB6eLKq|7!_16-*L!H#Z*5So@TkKDsGi|33FJvrrygmQ+{Od;R(y5S7NYr01ku^W3qTr<PnSlBn11na>jjy<kh9gJ9HSXDW&qv*BJO5Yw-uu!Nybaou6)-C(Xylg8|fvpz@dqLdViSy>Ju*_^m?Tx!*i1kY6wRe?{$;bq-Zi>|stS65p0_WA}!Pf>8s{cuF%d;&Lda*O?<bki2w!Aqdc>~Kc7LE$617nH09Q{@o<w_eaq7X4tV0tHj6<{TrPE*4}NDv*l0WtX=(`&AL7LkSRlnKuKXadZLioput=Z54?@Afe}a*}wMduDI|wFyZ(-6iU}p<5&0eKqAhc+La1XYJjFttO$NggX(pFm+NT^zp3G_Z*4=~-A(f9MMH<rNVBNBoYyrQD&*kkr<XLbIA{i=)5c4_ezMf=Z+!rt>M8erZ;rNG%M^Zd5l9dVHuf8wkQIe_d!9DLNwyp;<Z8vTlnXM_2RA71_PYD=J%S>G07enxa3=t+it_o@jJQWBJ{ZxHx>G|g(N03IgQL;O$q;TzTxQ@+-6JSo67>CeTggGMf;n0N0OG>D<q#l9F^!Gzy}8gJuXjY}FZG|RY==>RZu#x8%HNSmdYl0TxcK~KLEGQCa}mZFeSk=|`2F$-kPZhfc}m{MbD*T|m)Z*rcz@aC-)(_?(Xbv_x{ASC$2dxAal4w|+RnYpzZrW$6nKT#3UuLJK};t5frKZ5{N1A4bhWP1B^*}V_u-Kwy*pgcPW1=x&O`KGqi!EcP60*SsVUVU_x^kfYm!q+QS|#lENB_5kywVh+v0n$kp=UvuNAClnv8@nL#_LCpXy_F-rb#FcOv3#LVF?R$wAe&)P@E2oq6}eq{exP8aLS}nIx8pu%|*s&Pi#KBc2OaP_AO#FHw^ojTobD>bk~Snqz#2`I)KBPE!15sH>E7JE!r`Nq5}nXs0d430dBUvc8eV=5AJY5@rP^vWz*P@V|#0h5|vaeA0~is~yfVDinF6M!<iDT55bu(MlofZ5OvGdo!lK;shP`-?{ck-!J%I+JaHur3v=6f@TZhkWx^VNL=T1(4|#N(`LDqdqeJ1)LU|-cp(0*JOsMWaDFe{?~c9lA<IbOke};FlVV~A&liotFRFA(z$~D_>IaKA91x~z8q2mt9}#ot*!c94W89s@j*s|1^M5E<?tuVZH+aW;zZIFh#>+>4tkgg}ia8Fhbjx<ooFg|QUiZjlx%mb;a`~Zb>NxEwCv{vj!Cxkft3*bh@tIb1`=X&zjN#slA0a&A3SLN6oCdI+>fv`7p^qaKIJ0vM?^Qp~rWgUnGn6=KhdF%2#;BE50<bFO4D7OkMa%Zv`PiB%QYnmjs{Y$ylpjUR-Jp~<UDQZR9vm&??34s%sIlTFXR^PV5iH-S(MQ)OWiNqKQ9n~-k}}vcxmvFKti!Q8q5bte%s_PzU(fJ3YgE_SqY(MS9El~7(%7K?JRjpA-L>3ty}k&CRLDS%=<LjCz3008#h}NMb@L>0?LXd&pFM%O>f}?xt;!_b__dha!9SLIk*k)fRYAAES;YA2T{Tce^u$Ib&J!$4pC<LSJN+r`5#}viIQ@mveI-$~!1}I}|3akdWvOOK0_*@GCP*V$ZR_n}5OCiW>XMqwUgZVmDJ0wZ9<l;kz}(H?en#F`-Y&%dHc#4U`G6K$uOBKWH7YFEJmyJFWXpy4yTB(0<QEErTWeer&Hf{*Qa!KqtI{9goqr*uAy8!aHi`x)e*%38IFRmiM^|=Q*yN2;fi_r_&uBI8-2w6yqL|C6WeBtgoI5x#*WtlYq}sBm$@fC^g#Fw-8gb|XsP*QW>G1ztcfBm+ihoWVE@3*Hu-!$6E0!P;B>NKHCEVuBb#J6m2A=Z@!F?qaCEx3kyZLyBKfPtaw9=&&3t&h$msj+(8-2BGuGh#-??B#XrIXfeo9m$7h{9m8ovC}?OU<Fb(^}z>+Xgnjl|+yysy*JmvP6X}QTkvpA^V-F`mSWL`RjVCvdls`CR?%Cqo>Ugr@TpvOTt$~H~r&gc|4q)zc4M~HMu8zJyh6R{{vFMh*to3qm9W8*7ZuO>54-&!FGtraS#FK_}J|mTYn~?Rb%n)NZFWUlOnQSzYtYtIK|mQwc)lq=XxcOM}?PZQ$g@ge;oogKT6~@XKOuK>C3Q>!7j@mglSSYdzslh=rZ;O82t25dj#8r_vDZF(b^^#q7)lmBy*gB56!V@QwTUVI=i|R@O1-;gTGY)EWd60rD-4V;kIVN7fc$218KI6rdwdfIR>7AsiH{2XYU|2@Jqa*@D`Ttwkb}P2!%t)_bEXampy_Z)E4{IU0w@9HD@Gg<F@A69As_obQ3!zI2bCfI3m(9ugSky3xpLJEDU}ZO-&`ea)<MArc;7=bIrS>h^G`~Cn(Xkny+SgA!)g`xq4034`ll8GVC3f7Kz3(>yG`c76-FCd^?cbSG1qBi^`2FH~uuUz+iSd>4E`v0_@~?cbqy9vQ+em4i~g+oF8Hs!Q)$M)N)L@Y$3mSMw`#>LHTJ%$4MnXB@TOZcxK4kUP6Fkgw&K>mxk)#@p(KROvTq1^6zgDXcaN;Ar<?^4()1LjSf$+njA>rvBS(YDn1OrF$;Duv294t-zSYqI2=tN8dsjaiiet#gR>ioPc#DY-sXqaO|z%%-ejY0=RRu14&p2EVDW|;NZ}gW=K`p+eeyc+OB=6Di}A?AXs983!ahVp(HPQz_bicRBQx1y$bSmEtZ^GmH{N&+{v0+*@ZZONEm~2aO5|U{>%Oq?Kt)@aSYdKCT>jaAK}ul)5ExJh%e!aS*H+|67jFx18Vw<``6yog&&rSFfk{Yt2Z%_k;vpF-H&d2<jbWCFH8Fd+8ge*I8Jp>a^jn2cz2@Nd0>R3K2RJA+^<Un4)r<ijh>6pMg5@K<gd*h@faHdW=LE!Vupt?UQT6hFcIvXY;*CS9SpQnyfJE<@@sAQqqaoLK<bw-3TAJ5Him)gz)-;`apVSl7nuA?nFVqMA?tcFzz3*GjU&Er?v`K*j>VB#P$u?K>9rPk?A=<;%<#rM{tqA9<rg1#0c*<3Q9+f(c4mNJ+5;<2d&LI~3KJS{Y;^Da6blO$(_(@EigLVOlv{yzMrC9H8@ZzDX^90IF<6;DMK<w4&dCP9}X|f~m{v_EH&>v(#F~rhV?xK`3zwFgqt+4uO^VzX2&D2SR?Vc0E3?p*}DM?!D%hGm@m6u1{$Z5>?OM+AzU;eg&lDjL-8s0C5@4OFOYqz<kbu|=Q(+P_RRL)z9Cy-N4A9JIg5L9YJi;^Q{VP^%*i^R5q$%ChJK_YQ>jN#hBxh^utQml@;$kAi4DA$OuW8e6a=r&E8K|S1&kyYSF0$x#}s%;4C#?om>-eAbYidgNv{-0Wd)f^#Fb$Z<^)?Sc;BE%)pS^ad;W7S(Ac8^8*<I*BMTRI&x$~FEJu~BHs=nhN=LP9~;wul&kjDVv^EM^w3_NTuE72}UzQ{Q4IVaJk?Nk|bO58ddnrkfktGLa)0mI9-U1I=?15aP&9Fu8)Upv2eb!Dluh``%fcxf+nQ%OX`48JL*Bc)S*aUiah4P)qXwA7UJPINi@SO4X!CX>;X=9-H{SX>4ORt#j(GZiVR4c80;sB~@P!9lA)s8boF}-6K=aI+tgv@&D9lR3%C3E7Ratdn@7Y3Ip!<h9z9|otRTxCZ{FZK>|S^ZJ$7T(yrpUz==t%%#KzXGde-#QeE^`9EG4pUK29QhsYCzAM<XzjyW3cK?2xUTXSX99e5#Ue;pla>kVA|f3XAPA3KvTV+f0}d?A9HuH8VjFG)@yn`<bWT8~FD`4zrJHb(OpP0F8!s!z!7`F+`^LG&b0R6z4&1tqJ`Ocm{xzY~&|oOv4B5A}<_2bU??DSb0OkqxzKhQyBh=TXigKMQ^^zF(*VzoKwt4;q#lVlVU%q7H%eUAojFGhPrF{yB+__4RV6l>UkyECT5bF}~yQ{s-P|f>$3-n;?LL<&NC(_tX9izYTVyrFOI3!5;N0wI2lw?ayaUFQHd7%u5ptTJWhJPw!*ucN<jwgV0}-V?m6)OwfTXs-0<5y=AkfD6`3J_kVG^%QKjC^+WW@uta*xI`q47zlcfRjElBq+_@Kcm@<4h^If1pId4g{y4tS@9!l5dh^G9ou<_B;@owm;LR6?~Arx{1gA=}ZD0M6Id=zWKEP&8|=xa%$GO)ic2Bk<_o&2!W15wY@>Xt;8YqSM~omBe@ck1RN9H2EaD?8G_4x!rs!yrV{N@(AD@k6Vs(nlq8fr)*QspwM#{B2DE>sD-Itab7vtWC$gW|2lJTy{&eS74Dh7X?!fTx(|{qROj*vUR$%#giY`aWFde;7;s<;{bIroQGIFXS-Yrl_JF4GNota0brLDrOYkcc-&&Qb{qd^oN4rM!3L`Y+p!;KxG_8T+sHVcTJJ$8!@2N#v<DfqQM{efyQK3)D!{;G=fGtTBeivez>E-da^0Jl!8S@&1|rsbyvc7TcCwm!7MC<tqQ+UN^w1C_TtSphi}XTSC$gYhFsqyyMr?|Kv^-&<PbdigLm~)T=m8R8c~loZXg{(|GYC(dg1Edw6RrksAx0)deog*tS^>e=etW)*l~_72Wv$I15hPWa-X#_$uZXJ<_4w@Y{Ydk0HDO9Z<S?S;X(2wqhtyFS2Gg`3m<q-7ZO0yP?AwGm0(uUF?|~h8F46o_c=+EZuHHPRpvV+Fx&z7B%+%acX_DcbqjQ&nDo{rLT`cgdk312w-#hg&=9MJGn4`XI_#n>E1qS;z%N$>5juVpNSlv@JJUo-6ffjA*PfF|SJ`6YwaF7r=gbPKyKBnUE3jJc3=6_tRy~<Ofia$N3u|w<DYt$leV<JhiMWZ$_7~+<Q@Bw>EDG<J%a<x4MF0^6&=lDpC#({!r8lxH<0f-ejME;d9Q-FBp0R7?at6gj}l0+wK7`I;ddiyzifeHEQU9t_~16+j<P+Z&{gbZRwo(wm!4WRv^OmuaLJ~E6XL?V_W>5u9&w^V7u05~_I^sXYZG4jFvjEV7)?>ik+h_j;Ce(Tu?u|<KSx-E9kGY*JQK3`~lT{K6g?h3|)>QI?VoE7?T=)A7YWiz>Ee_40^&MYloh2!CaP&-nPdpXmj$|-BGv(w=cTT2ZA!?=0*3?+`*v~IwXS{CiDsAS@k5KLb$6n$_OPpxq3N`Ak;D;L-}x?ZUFctgj+>^;7gcwQA-&zb5qqR#>Qpp<H*TyzD$)>wKKsnb9%SiG`fCaKQkf#y27(&5i7)klKdZCLxm^gbF~8@sxr2ba;8zVd+o5aU<bjh1U~6aFM$2#Th9-o^v^@vo%k;w#lKIBkKexU(Fn_L}In;|td5Y+^WmcbaTN)#DSB7m6CLLJQeH;#cT|KZD1FtrxF01HgE%LQBa{F3iwnC9cb#n$hm21o}4U%$kT+7oXNTgu5Vn;>=6#)l57OH5GX+c->>_Bzs0VIbnP#-nIPSolkHSY?Pn0YvzbN;P?p(nu%Q56|Y8e4`T4JWemzu+heQE(!+*I=b+7E>0mnpM0AnCF66#9fDZr#+ou4-a(IqdIO}&A+t4GXmk<DQsM#~wGd#I9eRkUW4j||N6ld8^I@U<zX}u_HS6{H>NFV0V1<!g-JqBNtp#Y_-&Tk7)yjXS+*BVr{S%<F%o<N?2xaGiVN|e2>lfV3EjrsiciU3$L3VZBiRzN2~Y%W_tK)*&3Xcrh(O*TV+UK$S#HPPfom-N*qK{pH@C;7rZk_xsnKuYhHz24IpRNqS$DSS2RsLPXG*aEs*JIdoA;QY0d10$0;#2=&WIXYIp>%u&<=W#Fms`&|I{DGqQN%H#nN@LS2l1(Z9t%_rVNH<~~1R<nd8w&Kug)24RC3i5kE(XV|igk7@uY<V=H+T(<yctd5u3@2NF4EAH;I8lLcbSmQYRdRDIR5{^q@V7YD|y{;iJ+}cWG-$12X*|Lts9sM7rj%;5QVt$GH%<8-pm`!Z!`dG8Q6!Xk1t=?<U@l@9Wa!=gc?*|(6`RLT2XRFQA)8PLp3|y;0xzN$m0a$7hGEnGx%`Z-l=nm1jXzT%qV)r9KLPUkH03$j(kPXk*CLk1{E#p{kK(XL-v7dzR%$Mh>17Qu`^PwD4f8pZ{J)eyc8Zzt0s1rOLHAGP^NZw2bG6k8^{KY%R-rIbJG)3&&8x|HRdxlx*)@#8vQimj%VEK$OdqtAFOA}EUG`ROz2rf8NZzG5_Q_(t{55Rsz&j-nC=R;ck#4%bWJ>SG$(FR##g)`SSB`9Y4S5dBbh1=qUM6=1aKz5A~{GVzGn*q@NfN%fG^~*tUsd9rPo56u{`JZbI#!HtGFGP`2gA2MyQl?py~O>B<Ez|CB1xc5T<E~6V`IL9Ti)g9j?T*4qM_bzT>cGyDJ~UC;^`mfGM1g_)>7y8k&QhrngIgcEPwhYRd(e8M6%>{nLsyFX9Z?1>|_gjg1vJN!(G4*A0)P#Z#1?3zWHoLm2f3y&VdePF0hZOygt(F*>o)md16~jkp?c+<%xXC~leYA2)}nu-2W1Mlc5l0x}t~hkAJd6)t^~1&9+CucGy?Mj&)bq!8#Qy1F#bQc&4!{~SpXS|W+nq_2ntosnAMpli{uZzTEcX|LMTlpnH-4@p~#oJmC-yiJ6B|N49*!UmOfD(+{Q8AA_h2Jg6oNEuVQ9agugp55DKr7|~kl41{1!BC7(?)^?f8Ai^gTe)5m*Bjy$Gz{IdnLS|eNkJMDtf}LJ5aBw#VLw^z-=uz0;v<$g|K#*5=AURejoJ7|NMnf~yhEQwudCNhL00_DuB#@86=E0Tx^iUq?%HF`QcNl!))M$}kf?!(D=@;}q(0jEm*zHe0y5)Su)D1Wh3Tiu;O(Xv>H;_CFqaNRLtdg3jHzBHr`#@i8{J;0N4k8@D9|?gE`uHZsqo)#$>Uuc&09x?-zQ7ZIdR{8B}e|skgpAl_vemy3hsK`hkJHJ#R=aeePMiN``0oQ?7-)q5<L1%4FV|iwa}ik$?4f&Lm%qN0|`!{){xL#rYId$E%TjWc>(cSOpO2M2=|NJ{05nER(OBkn-?5vbZ1O16|;L&U@ahsIX$i9aG+WNH3=&A4AHOZKapWlIw*^)^6tFY40;9Hf_$|a{*C~+0YthMK|!qZ(7?eAOJCNs^xL~J=U!i?Qeh+1M(iu<Ma}x9L%%aD7vq_B^SETp#BHrKyQH`8HL?n2*hFM$%<6Knvb1d+qJR+aqnbHkCbbgAhJj@Di|$xZ%)9XY*XOthnJ(bM5rIZS<!TG?JKUFp!e*=>uKTZq&)Tc(X(a&TDOxvd#+|gILjBzx#HkikCuJl07ANCG(39R{@7KD(=3=K^Oq*qT@?e-t)Vbj^DwnVpqoFv2-obYCZ-V$_bvR!&x&Eic#QD{N{Wg#?IP4=t|2jzr|Cdp;+o@Fp8C31F2T^0kpwkch;rQ5C>ipDc1c9L3w*DfY8#+~Zq}6G<ns;d4Kj8-=&fC=fDQ5w><g*iNb}#`7f0L@S?JmH<YT9gO(iMSD&30LKb8G8<s8ze{F8||V`X@b%Ltrs60XV4U4Xas~iTS{4H)eWwc_9Jz4~U7d`<CYAw}=awE>&yeKDDH9%@84lZ~b2PK?GN|Bc^)wj4}VC*zJ9UA}(h}zi=-Tg)BQ=HXGy;lvC-7XZ8;wHVUkv)WYpr@}DAe|7KH@$Mf2vtFELf-momtzc*N{k)kJ0Hn?KP>EdC_oVgT}D8EOuEb-q^J16rO!mE8%bd-eWOt_*igwIhv=V*;I!r%>{2SJ7jb*=g`jfh;+YxRKFi$RjZ7weZS-YhFO7usjGuM&pYP~0W+7ae?hQQqodOn6CM<`fc|@$1oK^|{VGf*Yx)G@nI6d7P*}^t)!j8aT=O?&a!C9Lpdg;CilUJHv**JlA-9G<IJGp50>;o!NqfoqkTrF0?7_Yt8g)I+%86Z5qnw1!FoAg=@~63P3dB$kd}e{a_M)s}xh99R3R4CSt*p+%k2`S%*ktBJP$^nO|j&Fbok1t+;qa?G+c%r=dNpJzhbbPru=I>!u`TerUl?&|A%yqOyZ61>rdv9oGo%Wvzk@5zMlIjmc{$y!O?OS!BS;LTzCD&%7FiucRcX9rb{y)B|JS9U5kkPVZ8bLdVF2uPGfPJ$@4c`!6t(I*-1rPgU&d=c$KuAq?yV8+UFcAzl!eDYUqA)fr`c+jfIsoXK<*i4-4l6(y75<_0O9F3KXcgzqO;slX_f$5W&JI<Zq1;GsFm_wxJhjZ_GR!kz9N>+7^OyV}xe!;6=ysjt`w78c8yH1x3BwTLWEMe9vV!4tK-`$bqA8?>Neo0GjyDl@-0L^FU~c%b`S>gdK0qV#K?6D{Yb9XYUZs8rI80;Tq{i>C=*)*|r5CfYJuaChX&ryQH0MYrDe(;bS|LLkiboN$6zu)(>2LJid)LTp09&8aI3wFTKHNbFTFq6l2*pU2xBf>#Lj)ef{4N%|_%^{9^Em0bV#9<Kc^6on4c2^!+@m|92eK+Qu0MWAcA{5`@(C~~5AeVy@JP;@l9g;jSuH|+d|0(#^0F)D-iHX`wS8<+NJkIcodRrGfPPZ1K-tLMkN9;0Ed^G%-To~M%Yth_L%Y-65BJXb9)W%ph1bnx2y#;&=?gt;9=Gv{ML;TuB4S_XIy*?0sYSzj=!0;|kwF+FfxL2I|E^aivQzq_0Vgwm!p;ik!@ga6L@%PTBcTEeK|b5&=w)EQj7$^4N5`h!stlz><>Q~WOD8+6jddGYguh;?R;Wr-QZ1PP<w`?35Es<P)3#5nVj>Ii7SrgMSn5_9&In<`4Yx}xo<a_>@Q;p+TV78#sxjc)tC*n}H1^<E4#AS(h*!Sa&2jh++|7$d7OePQI+-rzH>Br+yY2bg0-_^17QmeP@Ii``g$)87CtU~9Gth>AO40i$&Tj~&hZhqHSDs9|j`Bpb+{UAQ&!@K@Cb(bya<!t#9Y^Kz-ttVyXrndL~-MSjUT$z{K@?qu5ouNY>~VQEn)v$;G%Y>qV+vO$(s<~7iBZ3hs3?|loS*|mgg@F8$eNKGhNl~_}A<andA!%(t~yo(ZO-!u%t!d&ePIaB+$n#c|z5~q&ykibpsBfqw^Up6(!SwxzVxDM{1Z0ag1E;jJ9$J$UU7p_mGr5;N@-K~x>+V*|sDWw-m-TsbAAa{+mBsR~Nk4PC*Wyg8kuos}QO#k>JO5IleyWu4`^&*K1srQhEWqbOMF!=mH*0}*rchg%jF+{{1#d-Yx@@_|;FImgoZc^>=xM^_CxQqduAwdIi<~N4pCH|LO&)z<wi-P<<Im9$oX2G3)AKtQB54#k@8N&!?pU^)}Cj0A-vOn1DBd3tW1lq;^s>aK=+aBQbp4b%Y7zWN6T!2mwa{^Q?fg~air^}jfR`SI96s!@}jXeAb64%E><8?i$XbDH$-);1uhMe?(gm*y8EaieiMmYQ&k$VF%_aedF50c+%L9c_#=bVgR3-!A{Uu8*{O5TB=3Pi1`uoChD^!n@7U{66^1bg$Jv}m8FF76o{9A!CG%N1QWe45@j|M;mXYnWo1|LavQHVit1UzulwFSsbww>w%YmA8C)hYv_UaDIBL%~qJndHbGi9I!CCxzk|5aX%}P$)q=zw4(S_-6*5noOdn9mK+O2ECmfJneW)xPpxgJRZ1Go`xq^9I%kk4?%T)6ydG&53P2FeQC{*~6zRwWcM-xjk%vX|Y~NzFJUX!u9UvY5ZvXK;6TwB)0X4pFu&?X$A*J941&@Nb&+oK;rgr;3im#X8HH=%Wj(+t3Qd9~(Igxqp1F;RN<_wiB^r$pJquYQ&^)xJP>AgXD2`%)dK};yvhP>OuO;>w9lUQ+XoIy(XiGqtN)pvyNe!lqh@@cQR`vaLE5Z7sBG|gDEvD|zXqM)1hhY-u-!i5*wgg^gL+X&OkS$_%N2k((pjr3Y09yMdfG5I+f&QBY(F5|CHorQh~Au`*1iiODOwakb2-y$;si>X|e9{fptHoumh*-I~<@n@$BDpbwEo2A-NTwGsfIHDq2>Rk*hIM(`!IdVP%KL8QcdeQS;<=4kb`O&CW0eui!i?xy$YmWZ<lVQ-KN54tG+a9pBqcrqc0jk$?gOki|<gmTMHXAMf+K4dOvu<gYD_~D&L&!Og-#bkG40jAdJ+azLWO`FFj6fHdw|bG9CPQsYm8<`uaQ{>ZOy}7!*aaaRK9P;_88jmmOB?u5TV!?`TvS*rfy3m+cjHhOv_Z?Qx0Nxpdw~*6Mp0g?oyRZ+wcn|n;wcoi_YcYzBDi>IM|FHo;70yehHIQ)GS@N0d^X|6wKZ7zYu|STeO{R!;8pczE%7Uz2VN80$_BspRiCRF`r=;|v%+Q8VNyZ)vZVM%JvLs9yNXEW(N5U_khxdE0uCNFqD@hC;=?GTW%Q0sdQi}>B%xk~^flgNiF-Q;I+53zknUH7tZLt!Hx*j5C(mY4^jZkfl^Jw${O?B7gwg`m7gzkSnXS4Zxj2DS76vl8!q!m5Na0}24f_UCVa6fUyI5km8WyeWRCV{#To-$mPT(VJSiUS(;WxgyHaOj`*Y==Zqrc5GC0oTOwr*q)MtZw<PFIKon4a?a&GSQ1@lhjfLNO+Fv=pwlOV0xU=~xP>>FNmN=c@0-4$JuVX(c1+6IPrduK1eVMeKcYBL2*q3U;zpY^<#v<|#~Le?KCTU)$~eCkF$H&5RhT2<aMmcd;E<=9Z`+L;ny^eu@|uqZjodr<X7@x5hIPu(30d-U5G4{YVNCgz^ij*(>xUyVF#jH_xrzfO<#arK4M}IG)5!)@H)Tu~h1gls7)!7=1Y~#i!i08XLSB)?8AQm7QYaWH!|;H7*97r(etQmhoGihu&SKW+t}j|Fl}tQw=7V2!m8@m-<`w#~Zkn#$x!0J7zh$XV=i>t@&F|Oryty+}>dg)D5f^?85@4+IbH%p@(wXsS}y#2`8b06a@C#xh;+wH@fDUTO-2g=wMOlJ$jBl?^7{z{<RAp7dQ+xE?rsfGcO7HE7KIaqY8B%_!q|Io1}EAi~dS+<*<m@&%4q6v`w4A9fn5HWRL6KJTD;!rGD%^T#;d-A>w|$Tv&#hMc9K@v9~si*Tx*lr<BOHyS$aVR+N(2#sw+H632r0_8(wCM(XEXu(4Po%3u@k%g4z4B^!*xq;6)}K9_DiZi9g#As33;b(0w;arY)+^p{i(I*PK<Hzk2qJqzq)qLQ*$tzpHsYXAC5cYQEE{yARvzu@B}`e;9LoacYJjQsIfzX1X#5bY*NagspF)OkjdXk%&taDRAehmDfNMMJk!DKez=r7p)#<!JN3t1ey%k^^uNt~afNu9+wR=eMe}y`vUf{vpp`gJafbQP&p05UO$!OJE88OLlTpq0|i4<I_+^5q)xK++_|<F#Bw0O?823)(-;8iYV2J>7JzG0C9trUUk|kPyZB_?dXly;tU}wjV}8#t2_K+27m4ZY0qMg)n(!yjyc~ag&!;Qn{%BPYJ+EGNhJcjDSS%t%>e53Yz^!h+Gx2qdWPI)w9O2#IUMjDb_U5^_l7tk{NSD^7BWd^RKhnch+-MD_Uvuug%HO+ofsF|@5@!y8T%#5uhXyOHggcqiaE#><ofr&1~aH4p8pR8C<Z!FgM;8~2h#_%lJenT*L&TBZah^Z-8cJ;Q}F&BqHLx+y5FK;L)R!uqMH>w2&27;lE;s>R(f<@?3`LkoHkH~i)upwuk(u1iPj-IrgkEp9N#;Zm1Gd(U~G87$H;sAMMlIo?G&<YitOg#1hGy*frSgG4S|EqVf>J<f{|?Vo-V>Z5P(K>B#_l^s;%Zy@}rh-Ui023S$69b1ZNObvQt`Uw-PfcuU^E+Q2G_5!c1BGCTFH}U$`Ymzs)znfVtsHY42r;DstcF*H?)4;PSJ1ir-seiO2@Dn3)<1&PS&PilpZKs@oHuvh<`RdxF{=c`v(S)Mt1&dpa!PnH0V%_1+gbWWM#2Zr&!0S@f>!WE<}YZQErje?Q8o?z-#iM6f}<ydbf@GHOU3*y+U{jC9Nn6J&?C{Q~zO$T5c&CXgvSU0T5_g1F9CL5JR8?G$ND=g~R75a&0kBdl1PSF%Q9xeJ(-1#d$F9U7g(ihZ8=(tD<)1P2iIyc}LG9j$bL<a1)h!Y8AotT#h_e}eH0#%c{~cFYk77zQZpoR`tGQja4GYBywir)fyeg%d_?_jE?INO~qD0=0>2TW)A$AG1AEU{P7&?dqo0DOl29Y~1Ij$uHhQ5?J&J%H3swQd2~RQz#VSb2MqQ!AY)8GihP*uw0?EJk}p*-iy500VwlNMt?!$lM;1|nqijL@0d5!Z)mZ%^d=LDXYn>dz{VNfrOMYrnqf9;$l2T&4JW5NR#d)0-D*8^ob$nE70@^G6B{HJcuDf^^{ztt+$)#RCK8KX2PoYpCy?4sbgwSUGjrsq%oX+JEz5y@H?dU1D^&$RVMi^iNg9lnw~(hM=Eus<Ofk_Wd~7N8K1^m}N4@@3Z}P6Lrz5I!%XT&b%%9FtLt51g+c^u3Ns&@Mp3h{=wr^*4RqJO(K6aygB4(weRN+9*LW6gy{sC?O0=~glD}lq85*@^beZeJyvLewuojnInTZSg`tWs<r9)sKaIGm}mhb=ar8MnvQ-Ch5{5yDL0YGn@I(;;hIm|C%;*Iqm&veJo9XqpT$DWoWX;wn}MVuzqL5X8=I7KOQT(9|zn1ljf}^$-@F4YJbJFRyiK^vt6do3Q@YJZ2ejBb~2c{bdfZrh2Q}DJCCnow(n3gE9c=Pb}M89D5(>U8)3CB{0IZS|PsfztgXuZ#(M$02Xk2wKhu?z+V|y*pBZUcdF0@t->jgLUt@dca%q8Opg|Z58r9PnBl>rjygOU=AaFR8vx$FSRfyR;PfF9sb`Df4uJoa>Fh7USiJX9z_`|o9>ZDvrBPt3+y-K8nI?+i49tSxs$5xqm<|Q@HG*gb2dN@e@K{F$l=H{oIwJs}@2@OdY@uZ{NIu-y-pLY=7N(k?E_TCSXy5$K^+FH*S=wb`9~r)&*EI$G``QVQ%>7{45a8G;ohsz=kWkI6S~oaq<C^??a0XEfW`oSvU+LJO`V`(Cf<mIx!*c4s@GoD#O<EQ}lk{up9@-!XXxFpHt2HgZBt*qR4g@NX&M*BLb^Fw9=CM(--n|agG&!SP%xg7Zq^#4|i@p-2C6BsUpX~}pJQ^Z-#f}IL7xV<l0*@gD_d3B|b)vGrM^gWyKsAi|-@J&ffsP(gRg1S{%rsax00000sVFCA{iPI(00G6(2(O<D+;DzCvBYQl0ssI200dcD\')).decode()\nassert hashlib.sha256(_s13_body.encode()).hexdigest() == \'96300f2d51b6343e5417517f02f3c98b96240e3f83ee8fcef732a41dac4b7c9a\'\n_s13_file = "<s13ab-body-96300f2d51b6>"\nlinecache.cache[_s13_file] = (len(_s13_body), None, _s13_body.splitlines(True), _s13_file)\nexec(compile(_s13_body, _s13_file, "exec"), globals())\n_s13_lab = lab\ndef lab(deadline, weights, state):\n    return _s13_lab(deadline, weights, state)\n')]
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
