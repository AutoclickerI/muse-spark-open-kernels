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
    if x.shape[0] in (8,32) and tuple(w.shape)==(4608,8192):
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
    state['nf_streams']=(torch.cuda.Stream(device=device),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(device=device),torch.cuda.Event())
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


# ---- h200_lab wrapper: payloads s14stage2_payload b3bb8f7ade8e61b6 on base S14 f1e61081 ----
_LAB_PAYLOADS = [('s14stage2_payload', 'b3bb8f7ade8e61b6', 'import base64, lzma, hashlib, linecache\n_s14_body = lzma.decompress(base64.b85decode(\'{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;Wq>C{#^hOXc5f@^K)lHO<*fof4pu)l+@0?H16~xz2zWQERsfZ<fy2LtAQbcBL?s6oo!lM(iMCJW7=Sg+{9CgJrD@~vz@ApcNE?{z1hQ5hf(pjQdBs{0thh{t1|ZS8wP*Nhf>g$CGGbW`G=0M$UVkRS#78y{7Bp4)082L(mtw0b(fQ&`fyuL=Zuack$V^k001HIo()@$=taD;yN{debycx%zrmhxO4jSXDU!gQC_QV_AgT0uzkYP2qE^C6Cba`mJBqwd()q9o3*;E&G%4+}#IFo~@|C&N`9q3uC^LK`k3Ol|s!>i3T&GW4Yx5>FY9D<jdB~K|+yBE~KP`zwZ|~u0cUpUdrmv#{zI>-63g!oG7EO#vd)R%{oPe7)z8!w=(Hc+r^_2$%xTo(lk33RZ^EsuK@Q3I-b9jm+;22CYY*qbO*yZ*@Wtoa)^b3vtaL9Ol#+~bAu85Gz?mW8Wrxuwj{ynP_pr<F4K9gG1%}Zw*SbncHfO?!1#6RB8)Eyu=C!<>4o^j&wE9cnuj$3!@)a%t_Lhwc2EY;rMca5mGr995591EZ{l(6|Lfa5N;r`$7o*}+-#E}*>br$o&&AG0dLPkKG=V;CksTrk(cwccg~1wrN5c}$f_9$#>1oX~IPR9IONGStW7^tx^qyyHi`s^McQ?BIG0w9OazDK)x(3y;k`R&krBlFn`f&x05<yYL@LR{+ap_nZ%0;#0r2aEkq+2yI-dx8H9@bq&3-`BXH2b?;rrPBGlp3V)e(I?KwgX0>?YIzO{lS=z{|nYJxEuA7yLN0}73rC@To*58%BTsgrOLw?v5;$=XE$3Ae}Hdf@gpUt+AKy316mIP>jD*nbyryNe?!0ow7;^Jhc2m_gnASYK@iUjXUj*^;*``-}`5{9yfM&f-?5)GH|xS41~xs$d!Jyws<u5D|`otHdxV>GC*QmzHYaZ9l#Pao#yOX+~L>o8A__S4lx+KaW(^%Ri)w%HbE8wOQZ?s*K*SrVVuw>yal?CJ;}#W6nwAUS7A0EXI+B+4!iqh@fD3JbPh_wR&Ro=TGYD=Q{$ZkaV7L8QRwY5KEq-Nc0St>S=Le>#-NjRn6v&}3H0M1s1jC5flbXMtS$F)yuV7=E{1wDhKWqk+UZ{YMVGl{AD#cF~n>e#up&SC6OU2}Ng-SiOS;ryzAPXGhF5utNGp+qj!g@161f4L-r``%@$gylxawIvf(?2j%pi{2esOy0Y+gc~i;(okw(1ix%iZ*L_sE3uMqRs~wHBr~@Yel+nXh9|DNK&7|XuRhIrG1m%?wQbxERsZP#>Fan4KAR~?OxqF-n2C2lGTY+_K$}MsaA7FBOZzd~}WRw1%&{_q9ts+*lL$SqW$(f=mo+Rm7k}MSah1dfIQYQ0wMZ|MNkSuuomK{_M5E@(ww0!3vW=wdthhfAzJ8n|Df)Ux9s#_utLBk_ypl~o#U-)U8I7K4h#}zsOOOQXXZHOW2o^^`_2*UorcDRiaK&w{F8Wdk#;dD=CgHvEW+ld@+;Y$B%A{C#+`8(mUm!==2<u$3KyoRkQWOKM_dAdOafGO$W3x4e0k8O>hffJ;bUWK;MSKRIK1Hx)=3cDII-pI^|!fubZLuY>U>{tj}#8gWEcAV+!!Fav+-D?h7Ht?^X6js^P+Cw!DgBJeZXAc3pFW;K9bP^iV-cH&R`ZxOhR0gMWp%?A%sbE1Myvn-57x{S>-xk081+J#y;#2mKI~uSgZ32MfD~8UgM-UXCKz`ZP;X`Jt5VagS-nplL2rxy^3*B+m04}yvI8our!MlIL+`JUx>A2X1<J-%W_ZroO)8_H|M#jZjy$2PrjR%gx|EcuFs(FL?BOE9@#-E3~Ell!0WDwIpTe!Rq5j&El|3zx_{-iuLrATQ)8c=oB1;Nyc{T`qzkjJ=Ux5rOVH(`eV)ffr~(iuL+)}8R0tOViI3~=vtJhTZ=+x_qAOl`&M7-VmBZ6|d8aUSia=&`4dEA&o<y$GC}NLTB3GD{lN;}$UYz#7I0C8oiLDJ+iCeS;)n&{nVlvdZh<AAF#F&6whUOzlY4ZJcsBL3gt<%C>biEVQtw{fcyv#_1FRrAnHTeU^&gmB_z%gGB1|`EG~nEK6*D;+2J+g{^kEK?2R%<FcBhZrmCGKxNBl)c4)z7R5W8-8!N<7|mG@5m8FP>DWphK%sSklwk&#L8Y7gp(+h5f)oq<v<T2@_nc`}fZT!lidzVTrAH>HSX3h(YjcBsk(h7Lf%+ORg#do0&Zwn%vjK4}Hfm4xOe98Iq>d|0LkczrHU|jSwsB#nTT#1SJBRn$pK<DPQL~qw_bCOu-KWgknj7B}J^3$E)H3j4jmUWlyWvTEZUAvs8;b%hX&a@Gi>vQ$Fd|>kP2t>58jGCHcJTaXx-YMS>1KQ-qdRq*<PV78<S#S@2Mwi^#85I(8JF+$MQ+QNiMF3TYrqOdOM?l70=~ed0hZ`!qTpQ$`vBS(Ifwc#;fYuR?K^n(P}a_nH;I|MBrq&>9;Er5sSMKZfI2SRPiaL7O@ElaW&aLoeJ$ay6A{AE`xYGWNCa65rH}yKJAj3F@$R02IUKJKOP^i8>LAAY=mpKjX1(N0MaP*mteB=-h)-=OFH<h`hdW@tp-decpWok9$weL-S{t!`qOw-m0a5U#NS*V?jH8-Qr5?pLncC)imxzmeE$nl~e`fboWmu-EFNSSuZRoWhN6uV!J)7;`JJsO7Db$UNrsOE*$M8hPO_DKmw7@^;j@uPSlURK=rE09L;)9dx*eM=|kd1C(ujAam9k5VQ9sve>_%?x2{7j;-Xr<2pr88fgqtB>oc~W$hsgQmXVG^Jf_<JdqYD##a0B_EWB<Gmb_yz_7pL|8&yLTZ<y}yi>-uQx7qqUejy(P)Vy{aUOAqqv6^c6ira9h^m0+3D(#RmO2xe_ee+9zRb%|!A1`s$b7vIS>$43<Q(yP;y;{8>R)9{u#Dumh~9?ibAKSl9~JR7^@qRkGhGP&d3N@23<<eGLw9j8Bxb%dOl&Q4m#aY<n2`_hd&wnEPuCppJ&I>mv0UsADy)0-LZzM0^-bMmDuD*nu9Rfqr9|cFwo#8VT!HaSP90xRl9j+c)wa*O}fsQ#xHhLggK8^eG$7$YUCi@sM1QN|ZIVen$GZ1gM7U0kbhCio3h$g?>#-=wbRB-95SW9lZqg3qHNP$hf8*flRNwN=-$L>)Xad!=1|H`h2rg^DbO*m>qFP73TxzFL)i`#X7@2@zN!A)$Ulku`ChR?YDXbDgie;0cDPVwac@zkz?cB>?t+#=hBfN1RYXpr#$kaD1m<^kq*nXu@?<%#j==FH~#BO?BUaLL)rtI#%e;=T=`$qZpX0MCUj??3AqwSNCE>W>K#USOimPt#ze$9osa=7y~%HTkI`P6GGY}}HNy_R9a4==a1&#`L}^ELKwaua(#m3|P}Qf0!|U>4cL+6t7Ig}w5s`r0xw!o=X~0TtzI*Mu7xn3J-&f;w_=`@Fmb+P~8$!&zBR;bR(39z@7+eX2-ThWjNa@X+D~<1^8y29H6L^N94onW(9ltbQg(W6Q{lzizWz;fx|1bE0D_8=-LLdyNQJ6-f^gyv?K?vH7w=!kawWG*jg6R8h`FD-<CX;MB5L$IMj0A)B^vnN1oBG?$rAQHIt))0c%^a3zT^mYsl8;_E9uQd|h{B(b9T4lGnWSwP8o@=S_hgKgMFJU=LI`}htOIDZIDI{tGA0WJkBV#YI(IDa9z2=?E!w*$TH1s>X%gBeyXFQ?@TS$YTLQ;DKz+K4ZTUGo?!^`z4;Cs@Mdg?B1E6h3jk{=@GX{D@eKw?rT;0H`L~3}9g1Ut)L|l7<y5@<6Q1Z%{lvV;<y{LM&b!!`hV^i+27NiIMFS>^Kq`u}{@I5>zqR@wF-M2ceLc4q&=dQ^`Xp=LFbUSQRQO^AM`Wf2JH$cxJm6ZlMPN6s|!7v(Tgf@!9xrcrhmA={-eVW3UuxAQGP~1DP5y<yB6JBn~s)roXbLep=StpG_NJSEM+r8D7(s2746A}tIR@Z&V%F*L=7B0{Eok>!*=S5T*k=nqON8uIMB9H7yuezwUQTQWN>rEP$j91N=cdzy9lx-(1t9_M)lzqAM-084nGDNi3w?ji~>-<AtM2tEKly$T5Ecx5pbMd8tt{&`@!0-nl^#7{1)RJo~)O5eKb4lyv;&+~-j?x`O(I&H^6OEh~hYn295Mzz$Nba`x&PQ4uDcP(^vTmwTiuFIv{JW0ahZjoi*(?agS>9_13SXRP=7KYdq?EmwcIZaT?S((??T0%|iva@G24Ks~zYKy|FeTk@+xNuX<>OBsPU?wha6=LR8i4s)qleK+;`V<5|GveZwS{EHn4mj!WZ(b2rtNQ|Uqpm95NNCBylm6iGq^ONec9tCwQHehQR$=#@+N0kPUb#8#p>5pC;aB*-STm-tk-1bLQ$llGc!9sg=wO&-8H1nb?{=~#*Xa`U)5dh`^jKkEEe-W#OAjN2DLHm_4L0QGe^hVEgx^Y`3rg#zlJu+?#I*g<8*eb!fsDx8wWDh&J2}=$G@s69#Hb!HwMD@K5Uq5E&e4Tpv=HvW58&Ze0jv11R%0@rZ&bovOpZ!n21=ToVNh$b$>?=zYYrdsjFC3Gi;x_mW?I0?n`vWpqk9kTcn?yJMGdx8PO2YJ(IU$Z_O;9;6yEL1Q6G%T%QN<Dq+ql_Xn*=6q)63+lYy?Pn0d+gV4l-!ZKa7w-X;_t*1sB8~C2e!S&&)74do)F>sd8QmteZDJ-BH-6#d*{&rHw=>1#q_5Xb4>}&bq4Vs++7Uj|^@=bbWSJ5|sSCg*kK-oolzICk~iWGEZW<z<YThfk#3m*l(70mY1mA`r?Xnt6&<PT;`DC_k*Qn9F`q^lwOY|lx4kO>m0CFM<&P1nn=St~nRo)GJ~05qb!C8@MHbiIc~L~0_Iflx$W4$0-r=(Y>Eb#ZhWkqP)xx-8JaSUZEs@CSD@B{zf?8ldkXC#5XwZ@?tCb{~-zJ##tg&$Bqi5r{BFPPhB!-maxg6OhI#q*CBUGE65STMZ&>LizoU2i<b%t}?+fZ>xWWKOk6e{U)EV(x}lD-q=&Z$f|MK0Y4$Su%{~vRPz?V1Uvvpmyi4c8mODX-0>4tZ<_p;cAxt5{1eM2NNmv-gz+B2{uu82RHyG2`{hpbVTX?o&!aL4m^A>^Yat}hiElWtT2In@=>IFC2%wKJu<@^DP9L(HJsnZ9kudkV+|6bgf<t~)oIqk3EOtq(de)d4ph?WvT(NzS&nIrN0eUNe3x0ED9f(;uq7jtas!~Kg+-KMbtwkG{^>>mIqqa4#k}cRm4rbm$3L$u3j<6h&I3<89G>N{}t5qj2He5E@I7lOz?~MbbBGftsJ%?GR#>jY#AR?bWd?~WOjW~DTzn3hd8xYV8M#kA;Jo0y?%MZa4q)aNc)h;1^fFDG+Lrt#<qu_r2??954LKaZ+hPvjKn638oFszzmf>|NGSWcwN#KUAu@?DG9Ty8Cx5-k_9HVRTMKPH+*if*ENX2K2i@jF(hdEMSWhcT^vz%b$F<`HnaQoyrtBg!EutB0_62<dunVQ%k?rzZSM4t(;+*vVu>fS?n<)+(d*J&r1N-!-}QJquW0HI9H`*gkrE5sv#j+KcVoX;EPpU36An^@w+B!{UW^fwn%2A9LKLb58BzsF0UeifSv!DxCJDy4%ye3WEk|&`V?p`NPAW+4nNkw-Hnwh-9i+y~_`=eZ|43U}u>JZkiB}Rxb*2Zr*QFo87aIEN}E?9TXh8J#@^zy23#@aq(Ebj6-URoq?8IJ`udf7ed;=^9th|<z<gXL!!^+{e{W0Wt|5Rl~t4Cx!;Zp%-NQ&vY!gv7pR7AjQzS+L9uW>*C4vd7}(F%NlApVzh4MZ;0wIKcc?M##3@4Qj0*N&CCKw{6TY-RHA7^(UGmUQ{s}wNCwT-cf+6Ql8~lqbv2vvfX>Rov-v@q7oSw0II|KW+8QJF1%M!H2_{oPb*{crpl23csX>k+$_{H?q_lMXPYr)%$E<zUl`XfP`7;dxGlM*xnp)ma<q(e^bE3Yt=E~@CCf3gojSXX`-3Xp3h5`Ne|5k&HxhHEgO$?5v;dF?`G6MG5>!DxyI7fv9=z7jnCI*U#e3*8$ssPBj`a<Re`1NFXph9=e}Tf1UWHXrZm6Q@8?Yk#=BuHwPDvT-Df#1ItvR*jML-08Mxe)$+fMlYT!q%VHa@0vDs;YUTY2|y*oHY^<#5^lbVzsXMB;Cn>}FQOP(uxw{}43^pBG9M==&ALd(#<Z<8(~kUw{Z_QVGw6hFukHz{M%|{a0d9{O7(31ew&LUhBF;1I+xuZ5Z4tKmot_O66g4&Xlo?DY4%&>N&*>K0fx;y6Y+6sLY9@OB89u*)a<V4{9mm9xk6_~T-|<rQBbxaj<JHNe8~Th=UFhCd-i8~HQa!G%k;s=61SJd-1;s!o3xExWcAY61^GUS3nf7QfEKFOV3yd6E3qGBIGEpaP^pR6c>7G8KDn{@lFD@-!k96r+G{CD8k&KEc>5SR(V8Z<Ux-QgxtfM`pJ7XsAdUFp?(|`!hdGj^Z-18g_M6Xuq9Pa!*pkyM}HV@u>>r`0u-!clu@ISWV{a&<=kQ?eGTD>E4Q!w9J0n{9h9~V3$4o)w-ALX$!6}tgJ<$+i@TVwb2kGLc8dgx*vE^)?HC_D4GK#K?%ii!$?g}!TfNQL!&K)jRD^?1c}cSEg?KFj?xklBZ3hdEOjb8qqg0Y%psl?SpD5j!?*ba0v|0WvDW`YwN%HsY2;g{cK0YUn2q-sJ2UT|iJa+s?H{CWMgCrAd~Q$Ls-yp>Bxn+#J?KHIuSOG--l>NVTC~^@9=?CV2%ahzaDvpg-~3+6@Fb#2m6SAA)Qv4U=XGYRjSg<A2Z#jrCM+^njov$V-)(mOf(A%pddA?Sar~UeV5#0i9lTUk8IpuBh+Sh|yV^rv#Q`MZc)rkl7SlUnv^Fl?EJ1zq}-D^pB~j;hB<n8wyTOjA>FBz}$Rk8@w?1-V333gCv`5iE*!qOn@J~FK1>ns|{0$z^1{3r~scp0etf+KtN20{v~A}6i>dhW#DS~&YeMme^gfx*~<RcV-*|k$&2sai>Z(8K;N|CR6j<s<NK{HG|IXMS9|%{!+sYz{ef-Tm{(tJg+=&@0cu7;%*7Aa98s7mf?V~>)@t6rBJumxGZJ>1bK~Il2BtWX+a6SBvoV(Foy<!AIB~FElu7<{UuN@ARADnrkv<Axj648mEkB%iDEU1f6-wF($@v{)_=kh1*fFN&kYst}x^ot;xdn9N$Sfq)Wh#K#a%bx)m3c@A@_hHVkh#jg@Hvb*Ki&Iln6%IPo5Z!9o>8!`+QQ9!Y!zhQ;heLt23vefS^4j(zAW=wOnHU~4ov<ORh{_@b}p>5;lL?pf$fU2m0AgMMVtW>r=WJ3?um401N`sU;~_%L^NwpshQN{-(0O@TA%aen2_mEGr@ND0`u*}yyUNjgCa_Uy0r^WPj{Z-~-d*oV+kAk+YjH;XkwoTB1f#<gu+~h*{h{cCHQ5))DvR`);fEHxe+AG8&_G?2Wc%)cFL*a(DYNn@0!*ao%}x=);>HSgFMqL{8%CgLhzUO!VzyK8=Pvt8@>BO^ASiMkPt^%2r*W$}{CZ^MdzwM=`6wM@glPt&nv5ivCp;kAMQW~l6u)DfPdu;F`*Et4vlLTx5Q>iU!HXw8moRa8c}5?b;J}zgu)wJVF7>Y0J5NjBrbPA1xS3e+;Bx}-bd!;y!+NZFbt-+2BAs(4hVDoq(ETTd#Jv*alGb4vH}B`>Q6yODC3dB6m-70HyMS>gWS*5oeSc1o{T4yFX%1$?5`RTDHESSmq`!cJF=Oq|s&BB7$VFLTDOP=a5)ddERZlfm2>5^!nHp>zc<q`?yQI&a)^9-lWX>{m1NOe399J$TC=(ve-O$y#Fnf#vZ~VijJy<Urf+Hi{%sWoMzUs8oC7U9phx_KJiPMQ5U~PSrICuXHs=%kl+{89&g<0O94bYE7C0Mms^~+w>sh=5+;J!J1LU?!V!{UfegN|Y`9o9`dkh?lu8Vt6O#Mm?OfZE<MLAFUf&QRL*H&HBcwY_ttES3$fO@$$?pUN79=}5%-CZVJJ{PY&mRy&6QMrNhCw}+!<q*c>#hw!wn!slq_F@k59XwvqvnuCoV{#$v*=;n)ylkWAx;Q@kOm@Ykcs%TFLjsV5jQx<~+t?3Jum8%~|<t-;<u@oz<tufXdjo_hXPrwtbFHD`y+o9H=nKiaH+Uisv%@7Ged**b$?@ZL%<=7-}J@1AP0HzgP$jDQW7gB!^k9<UmBOnvLed!hI(wK72Vk(|C=cY+dM=<+m($g{lE7v$~1|BiY0CAVEH1m=5F7(f8*B2Kd5@O{jL`9REpY*H5=LL%$=;<)26u89vb4h>GL1jyFD0W=Ooiwcw2+f1)N?jX1^R1y(AOY(KKp)H>RSgcJ%+Rr2c%PqGgBj2$#f_<OMA5IqEo8=|L=ke~Msfz#M?rm|ua;_5fiD{z_SvMTZk{3|rcc09nVrUU1h(m>7S;rhIULOX#SvKh<`I`gf2qqq-u$90>9cKvFKgR1|KAI+HsRkHL)i9f#9Wx=?#&(6rpc$n`?!)r4XW{j6!AKU1ni2d8%tu-Y*GvMtO*&jtY~}1-tU%W;ovgpH%Q#COwRpF31>{-EYxZk;!jFU4*pC2v$K&C>d#7E?ttGb-b)<3Y(8APIw7=+WuNv6*>ZH@DX6olnqfz9`p|YHD{0xffTD>5i<-p%W+-#a<%mXUsH{XDP>9dk^GR(g>X8YVT?;gb>HrQ<STk;;0I}Vt4i0vKEf44CkzCG|T0~hCC4Zb6)3?nT6)Rjo5Z!6dV1rjxf!-eWeUzNr-=}y#YuWUxB)gn0q7EDTyYp<x49^M%4a-QvcVmpu2fw1&f-<%S%G?J>SH-W};#pC4oU!S@qbY`whcY2V6SHqS_*e<_)n>=MnrAj%^L!BkpTIM(h<^+CAaV^v_SHWMdWQqys=yX+|B|Pf-3+JgI3%6W@A5$7lRP?f74h_D%!%R_w?}l*O<68a;sHB=k*MrQB%gbzWheHmZ`ZmwCDGf*y9w)Xu>(<oyEIxQ18z7q${Je#*c8M<BUM#5zT!lF<a^1rg|`cujlN?({QQ=<`Y9cI$?xpi6RI<TlX&~LG*U^%mNjFOerm^8n`egL6`9Fh_{pe<tp7UEL~TN8Z5TBBUkx8~jc7AZhf4BB)-v(M;LvP4&%s&o#_oRdB7MUUz0*L5dLf{Y#Y(j1yXmd79VkoykzPbGR5*J7vXOkuy2^0i-SmH9yR<eX@9v|$^i5v5KZ8c@bi$~eUD6-@Yn+sBC?XxO0{_qcA+RDdD+n}NrrvScOg>)QetJRn<AZ*BAeA9{;?K~j?d852Z<&1!hZp9c5mLb^eR8+(M&QVX+M7C)taGjMm=L6`U%HLf7uubr`g29&LSX)B8{68tK|4<nW|@(Mgf~KO;T7`O{%QU%Gic8L0B3RP;2FcG(e<ibjwlZXJV@u?&Pz^_+7>UJ=48jjs00Zt>UHY+-{*CJDNGPK$e3qk(*xm~JZjzPv`T(@6QHnCrC*5D6vumAMEhUf&e~||H>PA}SU5P6mNaLX1zei*5X#XWuk>C3N>g>4i?vrXE<W=WQW&2|!?w{!5MbewWR)ATsGd1P3_{+NCjLl7q+8fzI7@@9_um*NS3{lt%sSOEt_wa33PF_scWYHE<-BHOq@$Lpt3@SBQNQ6^W01IaWqa3M#$-OHqFC__`AO?*on38qX9}{KYq!0|vqfArBHgF+jX?$v8sY?_M5NI$EfNqfNKyh~MS;9z83E>%d$+8H_DOZ0w#hDeiXIF1ba?Wl{m%r^jyeb~{^PC;Vb;(I2XZb{AP@5ocW!?0nf<PF^=TBx0zG>Ksi3b|-lZsarM(E^`Z{Temyah^yN#QkO;GNYk)+)QhK<ZAiI?Zb0;xj`cQyt49^%c|L9h2g5IJ&C4aAd@-%0raWw0X8q)!vHCZR-6|8%1NkNW4uuZ45A^mY<R6KkPs9RIk9nNYv%>K|=EtKB9WMh5e(oTN($NfLzIj{J-DwC%LxjVOTft5J?DcCG?RW-oGn-9Z_7mZ4tUm00$=b&saDi2i(_p>zG*ra%o9%9qa1`Fxt*75>91W$O=s=bCXb7^9`J*c@*uCKAf_FcDZsI0dDhmF;trpP;}_ZYmWGu~>0pVyWyApCHHh2HPhI<8cGn$@1ncH$#QL*86}{S!74+i`>a5mmRtHw23o>5aQ0TGQWS{I=~Apq$DW=r{bmHN7H$Mr~xYS*(9L4zooEo2;gU_j$x*wJ?6La%&x{985xqP9z_-Wj8rA3SUmh9V<{DGAU|zKuV%zT0amc%czAxF3K4R{HF@dFX!m?g3tiTU#3E?Tq+D>g`r>vktJZsInLhV6ovd__PlqW>b<97KUB_;P8!wG8WTY+DBnWVhmSFR;+9ZF=w*5*VMSE5B9bq}1eeGsV_3<ydN>(vDV)PZ@aX0m4KoV54`C=@{hO^rg%szoB4c+8#2z|_)6u241wGBY_h~75!N{c*T$i^2N*S%y=xi9FHqa4Z;#~S%i<rqooWR)0HB7M|8i_PUzdd3$&77OuKu@Do9h{pT#k3&uaJ_a;42pjE!REv37_1z|(DP8evL<Lv6F<2tOKS^|LiI3hr6a1Bd8qeJU>1hO+{2#oZ)HtWxaI@h%mHC{0@angrSOQrqkQk08nQF~s6*jzG0(ZO>mcX5l8M`{S4S?er5impVi%ojgV=X!cFbb3KWJ|!nLzvA~uQgr3n<9jnm7$IeV)Y-u>V655GmzvY^yjrUkvZ2fT?~121HP#Wz<Hy>28zR-o7@}+x@pj;K@djpF=u~twJVu{95AV$4O)W~{ylvOhGlphZp4;eSg5;5fgz_PUE*#Kxgm1dvZaP}w?Fbw^V9O{b&kC3g9JuQT)1f?2gm!ENJJGS?RDmyrbQ7?ZE~h)qSyt%fVD}?DUO<bKGARCSez9Zd1sUlxvf*7Q|C5i*yqAry^hd_>Qb{=LCRfIcy-mS{-+M$pFjyj_^rFlqh9_NL_k&DH}<X7GkRFAxT3~cKnG!CWxUwYRyDNquHKoI$1bjIdD$T#1OO%#`ePjV&~zPhFISh8kM>Y9U!7pP(HcYB&se1;V#?yiI5g>T<eY&8p^SU3ERY|ZD|`>dSmzTw8+fK*5A?~8oohXo^L$s@M2<o8YX6l<2cBQb*Csrp(RQj;c}7|!k(Q^-=D~w|A^qE|ejen3oLO>HJ4T|cfdp)TR<VLB47O@EnNhFjQUe3ipzwBxj<W`fCRY>SHv<H~EpWS-7730~&)M2{as7zmqx<<VV=&H}_Uk%R>ZUJFz0Q1w$62Y+x(!+MSaR?t=Sef*v^*~777~=V6rJ&Y%!f++od55GB7RPwK2#23<AVo}oEe80YPo*`2L*hmV-wtFMSRJZoFjDAiH=>TYCy6w0fftt^E|u6<4X0~B!VNm?q*-V>h0Lp2)e`iQ+H>Hm-W)F!9L3rG>WZ@O(+b70)-`FQ~1rlttZl9h*l$X5S44x`39SUBmopwJm66}L1=U;P$7ZATM~&^P|$pPXqyfE>=^A#5G|VzFw-932RnpLyB_11Qp=zI<|wVu@_Po;aI1WFWTdWYl%w?b067(~#Tn(=Zi115@uD43o;{9U4PjZ%61GGVOz|Gp8!g@cb}JhyG(85<W+|%y36yz81klOhaSl}BOIuwq?azplJ~9iCf$Iav9tdS;g|Q9|ih6?Vn#DmZqCv#aWg9(6n3VTlWlP3R`80LE72u6>O|aY6*IUAV6j*-oBP;R#8g8pLX|1gL@*+XJ=1E~4qW1co<Mbo@HD@QmWfC1V(Q_!wtl-FBQ7P#yAg_)Q-qI}+sjhTN7J^#nMQe-u8StYpN0D^{o+Vry3FAGm8Vx=H?sWreHd2(V{X{I(Vs~X)Ya0F;8A!(B=tMQ_c3W@=AjL1xJ<#DF$wK_z1)j$65eGG-dir+Y=l;Q)3u*iHu@Y4hF@xs{GTejY8MIF&b1x@WucLxCAO*D&C4iF-D1wsQm+jq!=KBu{jV7P&HXM5idTj+c>;og_z?P8O>|xoLh!Z1%MP&(7|0UO1mH(FElflrn;?oAZj|!|Sp$<$~4S&Nntcx`Xg8ZTKH$(ko-B<F$^1gR|1R_`D?{^sIgN#p+r6@<CI((5(H;DRqfD33rBl<8Joz22J=lU0c!^!09!aRVUS~tTiXeqktub2a2UPTP=l`(u@-(#6-|1z6G{A@s_rb*`aA9LDQub{^Q4pu3LweVI)f{;0F6td#=IPaJjDyEh)=Wwn}>Y8=>>93R~e2WT5WG<!=9Z$h-J-#;P4fR7@ogP92=C?8?SAY~xTElqv^OxuP0I>Sp@r*R2-9CqDYg|}t>k<7i-M!N)5lUWOwVHCnDk2C9;T<5D91x(4APCsX`oeA3dmsj)n{RPS@NXrCMPig+S;-m(uTcR)Qed=x8Q{qVK2rMWwAK~|&O>LrbDY<eDS%&g=G^2nS!7)(Y6UBt6|`H(zii!H+C=NZB>WO@03)RC&G&tJQe?Ed#I%`WUMh{}JXz^kzgx<`_BnT0`jYSc%8*M~eEc<6CqVNLQzivouFsb?#Dh%zdBdKKU2Vp9P!r+#%Zhi5&_xY-n5@Ky@8m_)W?RM~qi3!_9k&5-(;&X<Xbm4o76J~#;l4XO(alGiepn86QVf?^N^<T(Yw^F#v24~^aK7`6qF7m#8^oe9Ud&3?Zbru1WjsJ9#4WO6XKN6=FaDSMxaF4Omf|8Y(r`50byNLfkeMT#GCTYe_n3ory-jUrJZPOx&oH^jzj<Cyzkn7MdVI+^vY#XfuAQ;J<I2_mvD@pfPbzKI3-C9}Re3`8-y#4f_8d8e1!9;QKEvOe%6A5d1ryn3=v(es&za&0yjh8})?|1-DT6gNI@H+((MAynGNsPv&`d%4ySalVfQDj%K@Tvs4f=EqeA+4ML!Xd^sk_#at|$4(&6c_p>-^GBXb{jpG8NP&JTM@F2mIXJfZtfX0cCT3Y>wz(W@Tf9hIpQcpM%b#NGiMO*2?D#x%$c89Ve24-oiM99&FFh-DRcBNsYF66m~JqteiLTjXKqQz?F45`**%kqp@721HCD3yWQ8gjM;}a$~~xM6lg*i(>3ed)1vF{o(6+JH&#CDz`T=F_|_ZVdU&@2EMCOv2Y|f#TTh->jm_#iP@v9{{7ccz8TGv*pbf1A3RCq+-QSVwr%$#XF1?+(4nx*-ulAeLI@1bzMh^{18Qn>7Ni7BPi8Rzz0h^Io$t|tuIGTzC`$7s4p$+%akAyYJ?<gm5o~+5jJ%i19>$DF|!AT>CJ0A}x1Xp(@!<m3b^0Y&}r27!Xa-Y)>&EpS8K=d%@xlyGwFk<pUWoDR3{)Pbw^q!z(#bpiGU@HI;+4|bN(JZQF6st5TByafj*>;6H#+BT(mSN`?{&Zpgo7ZpGNV6<{i}4kLk6TdYz=a(YinN;yU<FRH3c&!glDf)`zIm^fq+}>BvCMc)kXNVr2D!dT=htK=KlNKGBJCSUE<Y0n(Qs0`F(&a1r+prHIuKjxOiLcHH4y4X5DtKPZw<@odB5+D(SawmU5J0H&&|IMzT@LifV9(~g_G-zEGoy$@E?;y>SM>uO*hQbzs^GAa9GmkhdnRUywzgq?h}h<J&&fhKr5m1Ppl>ARvQSvB>p{mtOpiL%W}q9ob#-fZkAFqUHO47QY=RTX@8_jzEcLQ*Ftn(+zE}oYr5h;@j;R1Cfc!%C}RqecMq)5a@b|K&AcY1$7=U2iU}go&-Ztt!Lh;%z^q;L*12`!C5-WjiU%ZvNcs25TY`${6=Y->=w!nCl%XygzWyfn#GSlovB~$;J9^BG9qf~1lz5?rLGtgP0BX+}Fb_LnYMzglVU!Mc?Fk}nc2s(!*a)kNHD2+PZM}dcQ+}qXPH_-RblHci6ZNgG3GGCJG-4lOo1#p*+MBfTA-#j$`f*JP!**UMHKOcYxhrcCrZc3Gv19futOX>Vh7BE_l|FZ;q&E9=to6HMjyCG=*=TrjPTrEm>`P45bn~S*>v8=kn|O^wdOvctS6OyKmxYX)Cy?xp)&d!#H3KJ~-S9N|fax=dO3cdj#p@p}GP*Hx(hj~_PA&;5Um4{0BINMN3jM&0PU)4qEMM}{q}Bx5>Y%b<)~t`xaN!b~X;3F_)v5?LitA+Hntwp@70*=18=d53D~wu>BvaKactYTSx`&{ORLN$pwo849lca7~Uc%8A(%BQdau`d8nZtQ`M1%;(Q%IDn@w3-FRNz68a9D5+j~GChyfb4m$s<aO=kLDrVm#^)w)?l_b1Q<G50pXjyXlnPlRoi4o^#;=Y*34WA(K(O(?Mm(W<C{xG~a0>B>?%~j~Hbal3l=G7Zj|+gv!Sa6$L^c=LAt)O~}Vt5S}~tvsn&fZ(+hs{P8jC@o`8pKD~=GtP1VxNl|9qO$OURO{iJ>Iy$c|plikc7@o_f4H|NYP&p*kTZcgX{BdIc?y5gGsLbEm2fJY`q4)bpZN0kwraSK=XeyI^p!ddNn9hpTUS+eIkf%XbM;$ndq8&wAmG%Hb`X3VWfjz3`li!LZIHQq+H(;AUJHpRbWMoMcTbYiUIBjL)l_^@1{pTNMkgko2vaFZ);y!OPEf;JY;>$N>AWHo){#+}__oU+SVNAK@?}2FLF;(8`ax&uef9%P;yB)r%CZ#aIh4VR1t>_+RgWRIOkRSW6Q#%*Y27#*#l2aqjIgdF`1D4;l3Hgo@<Vb>IjrmY^{r;6t+!?)gWK^`{DXOth#UPdWIpIxpU9O=7f1VP>Pt9|yuMlGN>|RlZ4xYPf$or%pktC?!(@#RL>gS|^>By@%6Y4_Z6A8Y}tFX=sqIMQR0P>^^McFK9pKrWhS^wBhP46P*PfojCS^{E#I&ZBh4vp+xqKZ;N^EeS9yD6sk2#g7J5R`bmZ<OM%{F_SY>Fc263B~=fRIc2Ao?@9+ZjzOI{+L0kBiH)k;c-Bq8dkJRSl9JhIek}JJMb6lw4PD$Ep=T+BsnJf>yj>+6I<GaiWo|MKvRA+y0Hz}67^;zLXDnimp)|?c=xf@{b$axODwpGMu@giCs6I_aD@)@HBHriZ);m6(>b0EJgu&6ikayA8cNeWu!5R`ej1d$2}<gbps6|Bn?TTj^Yc>)z^&$hIXr|aI;HX1UgV!PNgXOHIY5#U#A7q%I1?N}uk-oT@Nydcf5=7#bNF{_Ax$%?<FkdOy_QN?0-S5j<t`^C6_3lFDbpKIE3N%`Ilohl|J187XH;-)>D;qcB|xu_7fFl>)Qi!$^8Z5a7YE?(hGodrWR_&|$5|~6anc(R2c%|&_kZW>1zQ1E&R#C!o3yyn%%W#%4KC0Zz3OHcQrito<E94%ll(mzV&yo0K99t+xkFx{(AyiDE|yHTHs*?|S(0(8_Q&MM>cwHi&oRunn6>mf4S8IfKq)^T^Qu^@cc4({FjJxDETU>6Rq1OHNtIWZVo4<a^T_k$6U|D@;Jlp2$#s>qW+A8+!&8{Ll2F+>0+~Dxv1WPf$5Lba-63a~umms|^$KmHa!sH!ekY*>bkP5Fz^3Dr{$rv&l(Ff<8AuF^>PC1GqnH#SgajG`fB;?PF&(Lq*C7qolWeG7{%!bEg+TujbB%M15Os^rh@pU;#o=KJPMy=JjpAoOuJk&nBl^1n>pcUy2(6yM$!kb1*MFM>;!%{IoSl;ws5w-$m&{cg{x6fpQ5U6l4LcXIrz_fL3Ts4rdB*TP^JO0px9(^8^1gq^j4N70AADt~$k`RtNMc&eBIh~olGZ06qpOON>m5}Z$K3jq?NP1!M;bAySv4d03x=9T43BHDwWR<p&U+Ym6NR5x?pJR0t2bRISsGsDDoadGxhQ}QL)ZMCoMHR55dC9HsKAX~-r2fg#XF9ws%eL6WNxmQGfxt-f(h^~W&A7*mU>N~Ci>M75wmTGre+r{f$T1W2x0_aF`S%@<rad=F{M6vtHO`zneaJiF1{9Ucvb@od%J0e0^edd9jU~(xH=%2)%gjV1+}AO)fhcxwc3<N*DlUNf69j?GJBQSqkcb>N?7_klO1p0Rl$;(A)xhV;S)-;_`54N+DN3Cyh^1Ya3{?`pQAGG*7FhVY`fo`;TP%jjKdA2bS?H`6Qk;8z(@6CN2MX<g71C8BF=HlKr<(6Qr*7iYF#tq9<WHBgw{!x7I<k;EN^AFb4c^hh7Y1+8)tL^C)nM^O~Z7CiU)M<nc?*qia<=>Jv2mgn}OPU{ef8qc{Z&^J9!(&Zyh;ljHno#c4i8_=5JBzMEK3T5pj4|y6?$vI*dB++R(Q(xY=)O6B9RoM&Mo?az{&R>WP@z@$El`7F)^-eg@Ob=TK_nQXkiwdy!??#*w-MR13(cxrUW*lIB&hZ1eXk%W=m@_j)&Yqg@hupZu*D$K=Z9DE>l)niS1vJoFnYT6CyCLdjf7r-X-9jI)N{RA6UsH6fAOlG$GeFzdH|a)7*s_b^4fMwlGKF{^DUIn@h{K{l112^v_<d2a5|ErfN7&CFe+VUrNVpYip2uBV-8GuR#rfGSsfQ9D$ClSKSRE9K&RtKRn=!}_X>C%%mJ$pKrosGBjyF+EPUU}NrqI1>2_Tgn`QduAUO#R<*?Q)?`7Wa8xozZCQuYhEix=!INc9Hzy(`3X@mI`2_-2F-bL;aOv%!dS(lzi!*J<kCZYX{E&U^5jx#X_;!|t<D3wp2FL%*{L4Cr7ubXBbaT~ifDunlsIC`Zc<OL<wx>E@(0Z5j7>vD10n6Sn%?)}R%Yiz3w<Z=8xDr1a}ji>YCm-ZV;}H0s(VQ;<z`8K-w8#;!BK%@W=zN`HWkYRy6Hi>tTeHg)o=@>CJT$ChG6X&fPAPdR$^Xb=BuPAcx?u$()U#X&=zn_^fyD1$o(Vk=zki{rH90eyS#~%xK)}MNBB0JJynj1?2S4Ef9HKKUiDkaWVX7!TcO<JhvanaJy?B0RB94TvMdf2{s3wOEgeOMIEE45CYKWik_rHm`_$OA1((7ybhyaTkq}SMVOF5Xjx3t@;S)!12<O&aa)MXo{qCq=DE_;vx+cX6`RD^=7TOZnh6hGlPvJN`3Z(kaSw6QX*2?Q`9;OzZO>G$oE>6nHDYtrC0BZ=3(hvdwlMHmMQ$oY4W5=%IpkE*=NvvxsAXFAqM(x})Eifxim0+sx0J@a!=aYNu^?tnD6Qw-Kq!;+X!M|aE;0^9PA9{;4HCDtY7o?NmtC|3Tg8X=f^b^pG=*<;xc=XS7+o{qSzILIqz(svUqOvs4C#*EbA^I4x0e)=cS`{;Ll>+yJZ;ASJ#8mz{#OkHAOz8g3qO3<NtjW{3OGYMJr1gt_%0k966mSEo-aU7Jm0B*gT^q~a1zw(A+>hoWQG#n~ErrTDxYO2;l+MVJS{6Z<@3@QqvZxdK4Y^huHV^ZgERRi$;0pFiN!SRY{+PKJSe5m}V8116X@E<Bw)l6YGwOb!*AsT3JOh_TCGMvY8R2z6_|8J9Af_+$dY@Ye0`t?F|0<C92xsF5T+zs&3xyV<eRcpd<;ry~E$4K$ta*Lqk*|8wHLEY_fTK^}U`r}BR8%f0OzYM?t$W2}J-wGzoVUifHxPF`c==hXh=u)o^d*u>V-$&*qY!fDHY@MKLRK)I5HwBB68F4MTnvOli&zU->Wu^uyyH`}cMuvJk6HW@J|>i1P3CA+-3&R|LON$T{CKKl;(vYNrL`}y6Te-1N{nOLbNGooA3i<b>Go%&`Q{_+q`6Q>0c3@5eL7>ZG@}0R7TWmKduQ}opt;a^!Xnk#!UBy;Oc1@6a<HkJLMw3EhxD&?8A~RcwO9ZdDG0_c<04736F|_k&GS`&^ydPPEz{!*U2}WUKGOpOu55PDjLv7&O$`JL^HC1>!8(kX8MeOWuKr{RK|W_b*pY$coa_gT<jCT!%OhMY4oT2kynyi&x>>+hu&$DAQ2&)*;U3T!xtIgp%x}veN`7c?MXPx<GMw;=vnY%+0Ic#>NAO9@Q|rSZ5jr4$Ru{h@2&qF7#Zvj#w|Y41ed7V}MNie#Ei@Eb;;*}WgzH|0dcr)lO(bS<rGuqNdK9?r8A&x8gSCk9B498rWAIywtE0GWK0+7{Zwkh5{X+#$A<11D<zTcd8<dKRhhsGm9NPnL96bz^1bsB#zhWVeQ4{loz2u+&EViyU*C+I%AfBl-&*7~zJZ&x3eVW5wSH6&4<ixVUVG4QHZ2-HewbJD#wEkAUsS+$CSicCMr;~u5KSnB#0v5T-R<itN${#`y`P!)Nbu<(!pijp^1b-wU@SD|!;3#U$RZ}I85~Y~UVKhmE5Y^5)hySahfz!A)D^p}~an}u2mo_Iqi}-KwVoEisCNC1CQSNO~A*q`oB~|Dkz8qY-n#EcU-DcZsiK>g6mg7POM{X`Hyp1lIQ3hEL;YRMo=8e@U=1|M66~PCI-^;4@1-|k2@B9{E@4rK;m|`byOJm8n{B!ME{t{b-H(eYV7?P>IQTi`aN$xYm%v6(rdO}d!tTB(GV6X(iJG4zRYN${``{}^8Raf9W3$q-$gK-^~nGFyTl7W}&i-v^%L6m*4EDXUfoew``y$?959^iI4e1Z5J((J^|3~(lanDO`7bmE@=^SvIV9sT!T_iL*!Df8D%IFHh&m!Mi9K38%HoRZXh)K-H<q1>58CqGpY8-kcJs2^Pn;nG=Rk%4^r{E$NcwM{wKo24VzNVI;vq9jbL4V(#Vt?(uEii@;UH+=&^!^%A*pG?YX=Sb2NW@f^UF_3Rl5Aj)N(8q@|iFC*jv-<O9eT!`5J6wkSQ*7420M;a$YcBkxekO@dX8+IVPpH#fNX|8qPW<3+aJs`53Ul^SqNReDlWquE(Z!9F2KrQ<+<X1o0SYK~iibg}(^K{G{X?waS6n!zMV4a$?;aDY7xO}Aht!6O%cZ3QF+k+z;!ju!6>}hgQCIoGw{goY*5b4uD3HUMkR%dcpEp=$Whvg2k9T5=#M>Hs51ra&odK8D_*JzCo>tdkLr7e5^9dbTEs<vN6jJVe=3~Sl&)VnBY{>Eu+C#wqR@sX9zA=9>T<P72vz9<HGt&41PunSBhvnnH40FsRi(~?yrV$&qqA?61vv&^&M>aPsR8<m)$IB7Q=qk^pNo-)qCcNaFXy1rraT(eH;jIrXK?3o1G{EF$oAdCxa6NW*6s^-QgCfi!bw7rW&6kYjJO&VG;k&>k`-V8aF!A(tG4df*XpU-ZPND)9bDHPP38F>hGH_VT><`cSU+BE(9CnN!V#P4<O^?&V>E`O?sOPcS+=BOswV6v9g~(b^(r`bEsAu9@1)*B(EN$+b2=4-3yCtvlU;6Lg5>U0ck5NZtXx#>y&KvN!iYo60=dlO1h_<(`%q)5s(i&X(t$I$(g$KVIdZ<yu&<{^T86N=#yhqQejDe*VJiJsvME34Os0ZrQAVw!lhAZL|YGZv{X@*cT_S{E$gd3`7a-UXqa(f@u2uAEj`_I=O@2@<dE-x>@w2@rS$$gVL@~5@AGBMR&$f7aVqR^rZOQj%+0H+PlvPF_5lB(BU0k|l8n;r9-*9(frd|grJH(!PQs`Z@Z0Zv1<a7MgHu1HkLoMoGgJ{h|`vN@Yq$9DX1?=?`A4o9b0SC?C}9v6)>L;FzE#t?%j%<zv!n5Dx>SwLRimVg3J9vAO>A=yS71ou$;id?!mC~OSAg(j3}g+<hEM{!FsI{e8z=93(cp_0Q0uie!vV!1;?tF93xZCoLbI7Xra=1ETlMZkSrHTShT?4qiqW=Il1@aq;W4JIz@X=?CFVJERQ`^A5JnT9D#r2{_#>?io%L?-G56CyY{vy-GMm<9(WtmHX?P+%&A`YT+iDqTinbUc(vqb&>w&g{}%ojjgtup6Z&2uvfGD6QWJelDI-fG)<rWl+A#ui~lbxCEV(gG5l^4zDdnlNMKP6<rSxLV{<X*}nogNuZ($6KtYi`mt#q#|li7YwMuj-Zz~4w83U0&<Fa$(ftg9qai1zwQzod|7TLHg>6#s6eJWPzbN9HR%}?Rds&$&w~aWx3M9Y{VbS7556o{+Lm@x|#2HU#G3^#6On<&GCXQQJdS5Xrly}9Wl6O<mxd+4$Lusjm!I4WA@(?{i{UWtphXQ?S6*!a!CPS^bgrr66J!Umo-0rIqXxGMj&h-q3z|abdgK<&L<SWVkYzl_NGq@8O<}U;A{x_(hm&b$<=FWn@#}E!LGWe+6TK{PbRWtyXUYXy~Vaq|9vg_CaR9Ec_ZLbB$mExlEeijBs%G7(P+LE*q-_x**q5^Q|7?qg0yUF0k_y<>)XA&xM%ZFquPBY%y)jxDeW(3qX!J@9@-<GpP$%t((d4Hd%jgqM10v-ozMo?~I#MArR7<}*4rDMgA!F;>>KXf$gX-(*2t}D<8B`we0F=ocCiGh65-;VE>3z17qi71FgvbC2r<Uys<ZXsLBLl6~RRu7u(s_nl=1(>C@sf0R&jkmwNGeJ4yX^PP!W-caf>7}FCI=0^q345`)ddY9qrX)9;ff2PA4N$*q@xJLcZkdK=@;2f%x4HaRc8UjB*=UF@JJx9`m&hi;zstxKr@#nzGq>G>;S7cw^n%>-PCx^^hzHV-{DTXQNjdg_2yE=~%)8ikwR?_JMs3MlcMEAp?>Ro6!x6l~SwK#f*IA!xytRxD2{`rJV_5L)7vQ@Pw|XX5mDcVlQ8?U((KT>}<Ao1u(?9g;M|r8`x?6Li(mkjvO|2dc1`+25H`lNk;Myh>HmCmqyhlSg+6RRWF+J}YJ5@0ky57G*L=FGtZ6(g@gt}>OIgKv%Zn>-#hCbau7i&pIJg|qCE2=oAv+tMAcicE6*MWqi`gG!Nct-$VpO!Tlr<H!uq$hF(Y>`<)Q=KJyFIx@S_$}=9RIyzJ`IqLHJ3A;_s3e-d8*&E-&`3<H*Nqhdo+u{Nr(^;T0-%j)4rsvZ5-+1Eroc;2@>4^yxgy+>5Iw8qCu73HCgPR|@yeB)jmj6D+YSSM>9wXK2z<)4*ihgru-wfi)bVy07pKJn`&y7CNNw1QA(pge2CUs`@U)#43QeP-p6q$H8vPZ&o7gBm?rjKJN0N}$a|Rt2%90poDmNAfFU5!A>9mOJ046AFIG)-LwjAr2wX5Aeih&FUa5nFo(y?!?oNh8Q$+R<k3ku2``2VOF=6B*WNV7xr{l6_oU{dMru^P#HWD4fsIe9>KuG9GJn1)qL+1ePlbJ`X%y0-2GPnjEN31#qHC0A}k9&pVb2zihp0N6>t)tIt=hTQ9=o_uNE$id@y*WR0-8>S!Abt2WGBU?QNSguMyYPGHIdwMYj3xzvrzIhjj$1d2D7^U!re+haBE{!JB5}p6*c`1r`N<7y{qNYFG;GBFd@+&4y+dP_KXU`-#FOJPrBK^8TNLHU{byea{q}BeuseM91KNq9@=98N6pU{p*h;*n)+x^0><O)X6+F)x);`h3KvO+*cI$$z#M(FoNr^NCAX8k*~1Fro6XWygC7CZzq8Hb)<Gg`%W^8X;ZQ-eK`Kzu5DT=*zuKQ!h^3S~<1V}kyuA-KexZw7ra4`zb=@sRm+Y;+knTCz;Nx)oZf;2S$b2z{(=A&m2;)ADKeuUBvr%CDhK57Js#G<8FHc%a^(`q2I%)}i~*xVg@2w+l^|(&~&jlowS6jX${L{UJFLW}ti=pyEj0u+hxhF)7g2I~LerY@191#B4TekHnr9hH3aJ?9YFu&Sd1aEVPUHM~-ped6(i|W4Em6;R#`QW4M*!ryd%`Uhw+^+N!<OG$%N@kH3d@DRbT!M|7qJ$S(2S5C4OqY`=O9$fA||uh09Ffabat^sE50psrKvQ33C7^RVede{&<E3nS}TBN;TbXHtKJr=l{%HZ%$qQB4GBFLLMQ`N>S413Vd?vKwdVEq2~!EL4}3GF*`SU()AxbkPuuY}`hz8N;k+ixSn4BICMdlg71m^w10AR}EA~2+zu&4xB4=sfosYhqP(`0}4$Bad^<|`zwQWW!|z;)&Z$<Y|g2slp$Y}HM6{X0P?!SkUZ3$t?Do#GbYLmRYaO}^Do$LxPRiogU=a7ZXeue#Eo6bv}N}D5ad!BwY(j9G05boN5Kv+8#t)VF8TeBRlD41g5ONrnhyk=VsEyhrKVs_<bbqr?nd8cbSu7}v=(B>PzdE$Xcmr{m^^ULG$C7zz+lgv`^R(^Ao|nm=)i*kwCW81Zn?3}@7oF)Bs{1X6Wy2G+XUD6XoZz9Pyt~iq_4b7l#ffq_Mn~jDpr>2jIVh^N(oA;%FU$<4K280INXTN1`;pp`J2dIm*TN-Zb_#Si>x}=K+O|jIgB<mtN|w1ZM!&^;#!1VdmPH)0Us8qH^}^4S#HqlND5M<y3bt^NAA-Tdlno{rZl9@MNmZrkW2{~=J^yt$?<#^FlT;7YRx^h&3$|*BESTA^L5W13wi>w*AWEkcv}j~{}Tq6*Y3=?0rkpwdnI>%a>y)Fzj1y&s4^aqJ$oqMW|L!!d+&w^Garop9Tl$Tlx~{Iju337j;GNpg(mm$E(^3@(yX9fTm1QO)Ilt=yV3?Q-=mBD(^_R!)T#zkt50H1<P$wRAWnAo``x^%GNa$Cm^yn+)s}kiz&K~zdWX#%g7xuYFcEQCT=!K$7PzYK4AH9Sb{8|%1+pmr8^B->)i4?Ywe(2<WLA!Qy^KEX(KHOY8(2Wh)=_*;2~rfRtOeVZJhjtRPcduBA@MO#NB$cNm=to2<xiJiT4>qE76%eb?kr=XAE&Cxx}!vwQ9>d2UP_frP}!_Cz9lYJ`|ZkiC|R=EHg!hqI%LI@()ttYsN9(|`xIq325NaD)b|5gbuX)V<HsMEs|?i8Gut_DTnMkEwOr+3K(jdtO6ot}r^4-SoOFI7Ltl7+-L=;L8t0G-D!so^{WfA+C4qv}ip?)lu-M2%?qc`;c>`l7^ZC2(AhO-g&386Q5-_IEh?gDM$IEiF=dOf5_YnPzB~#!;mVH0MKe7(%?2eC@uT8s!0-zZoY#IGEadQ_4K}mIR=o6wkSDmg54m90Ox@3sN3G~U%yA~b2C_j|DipFa--zN8b;=}>J^eX4`<ysz$oRcMq<L#oJ%%6k*R;>bpYGVb=5}Zz=PTD4X1V#0jlYb+$JFQTD1G;$q`U$xzGD!Ny=g)t}o<JuA<ghK!|CW-PV!&og1a4$-#O*~vk^sRItDe|W^;^`@$FS`#uSDq?Ay)12S*?dX1bOJ0im!_nLi=`Q^b%e>1UM(4G*eg=qXp=}2>#R^uBT(TW_oOeyVhe#Rr3OE`AnAHY3$R-uj->nz9$Z7gO<(<ZBSxHgKoAX`BqbJhE0DAMq=W8*sip+(#Ufb&eG$C(A-^|oDPI#tyFZ15%8Py000v-K3m$eqc;>H<rMyh;>pah5RJRACrvjuHtmTGXSl4fI2GYem5+dY5eng8c7ThQD4T1?b2ZhXUM~TO^(j3N_ZW6f88G7P)xdemM_Haf?QO$xYJ+AMXTu)qPL~r$<bi2j1riig&<~Er36zFCNbMxz#nttwR*KK)V}0lqw#}7+-24wnwiSZfc@`vpmEixUS|3B|ClE&#fR#A7=*qHwoxx+jO}wx92lA&yUa-9_!`ReZj&^c%G~Y9DwHsZAnzqo79JYf{oOyLsqy#gz3vfn(&~53VYPBHsJ+Y|YVXrpy0Zk+GFq=KA*U<YW?G@ph&^NT{H~3GeZuHGg<$gp_rlMRcy^T<AZG@0Kn<J&M+S@npm*5ElR=modSIv_8;i29Dk|ks>A#MH_=hGA$zH~_Wsh1O_h%v@s{-x4hR#VQKeztBw9>jL*_ER0q#$N)+=m|In8ZyN5-UsAme)M}{!EXk--m>u$xZSYinHi%|$(<-wFHRDM{?`{s!L~?6-T(roNDzy#s6+=5wraA?R*udX?;^w-wi~An3AQDUNQW1_gpNeKyz?H}^*QVRe1U3KoSp>;)A2aw?jvJcV<3-u!pe~<;jU9twNL>?W?mD#lDZ$-7zcw|vD+(VI4gDB4cEZCC;>*|I>dWl0efG#{^X+)n{w$Uj_4&iwYYBgmulUslSm`-1R1pkAYHEpTmoVsJ%_TsHC#D8?kA^D79yp(fb!A9r#w>Phw9eCjxJbn0~34)GV}FrZK2~l60wQOGVYsE1eS4CL?}ds$01%Yt`xcrr>jRz8H?qmamgX5wOlD*`7abKas9i0<@lLuwx3X<6oA-sRA9W`xvoLBa47Q=#Fmwh`Epg`XSfbxsqhTi){?_hvRQN?Sjv7KrSy+*WLbZH-_vv8PARsxEKZqV7IsK;5H~9wPB`2sy(Q3UivPq7sQ0RgoB7(G?b++Vaimc7`DYNYfm;R?O7V{gvTNC3WC~2~MvjLq^9{}NMW(601A@Vym-WTbYP>BFq!<gseOB5o2oLynD$hwK+;s7yug+?H#@(L`i<}m5t9!DSl1?6rJ1x|#O#`Qbb)Hk@LJJ%_QBFwk!>iQqFW8jrur)vIAG!lRgfSyS?+8S_SX_{SDde@5_<R;{am#Q#EozX|gBGC;?mU2Dv)Oeb`DFuU*%(Rr?{Kt>Hqp<d896?52#6`0>EA|+8ErOAr*JLEq+iPsvnRre?*dgQvAK0gA=dRXu40NUZ>22S7Ejo4X45kwuhUW=7H>JqJ25vGKnEAkIn0?Ma3!H7;A!;`;&4#%9`<O}*c(LfSh9Q=pBog5SRV2xi_3C;pWpdJK$%14=Jd5kh51xOXwCX0G7-Eta*!fOul1;33I0DFGwsxWNs}&6VJ9mD+nU+K?t!@vxEAPpw%E)5GS$ou*nFz<C%(B^^o>`oMMYzMICl)b@NoCu+m)3dCC;8<QwaTLe*hn?BGjO-#wLxxq94J$`lf7s4Bc6LPq2X=6v0S--$)^!jnRm+&#o+cv(Y5sM%!Iq^eK+%P)%c^RbysHkQdu9n#zUWpp~Jufs6pkC_uoRZ%L<kk4Z}##n~5IA1!SU#$M*f<n0lgH5FyrGt^aT4cUG#R~BT@O4_rjEI#_A(ne#tZHPx=!c<UcKdTmk0Z(5>ojd$|ze?MgqU5nOmgS!JidjSIy748PP5Pf-TAov-xO+tgzuxm^2E1TaM!Zc@e;%NN3gHg>b>JZ21fR~9%E0cmMO=O!<_+iw{_(eXn2S|Uh#8n3)DQ28P3qo;l7J{})T+kjCSTE}nwm9xHsJC2EVrMyQ4y;Aq`iLFzRplu^B8P_=KE~r#{8_9UL=us17JW}As*0PY6sI$hzPMT^HM?2;$gm=oY@JVO7{9^cxzLG@{v@)=q%J+-_6Y%$r9TLdh~RrH+c7XhZeFHf~2cGxki(B55ob%%-<~DpSUcpFQCH^3Y}nUb%&U2yG0{r0Ww8`XWtprK@_F!>SX4C+pL2F2uhIHxjThB*ZLZNtLWgQM1oRYk&4p8UL!3%R~?@)rT`jq)^soR6Q61N;_)4?g8Xk5?1%Irzp8jp+lyFkIwI%~TlybKW_$W*!k=GiMgIBbsk)A$w}fg(nvou}6hq9mju;}Z8)@`P8|fK;202!ZRnBb2!6#_I>->LfhtNCxuHHARXv_~B^N&KVxi2SG8s?xjrK*SKstxcX@+I$T3;#R+CmB`N--p*v)pKUf&-{m*p78NJ<Tp{<jb^OYknMe92{`Zwj8d*~0JFBmHQNCO%5uoKIwoIiGr@>?^<+zCQb)5q`rDTG(TbON8c|!kKs5_`FM&bkhHEG6Ev(SpsIU)`VoNJ)%qSX~5I$$VqWwA7o>H0CA~SZcfc<ryc!|8*;*7QA>)+DK$L$iEXJrm%D{L<YL%Z_?#0CbcePbe$5Z~Vg4_a{AJ07F&;Tk8Avih+vX~f}u5QbHDiBOWZbare@&1HHh?V~JJ;lPT0FZ#%rNq^VD7l7$`dMCYTMShO!pfh|}37`%KAkKHo*Z4hTpEHyShj)TB5Zp%O<a&_{g=^5FhTER_&cYjTfmdf|H2onp(Ou)!Ia`H)t9DnfjCXevs*#O5y^gEmPyVXUno`f671Yb>3!WI9Ag|XHzr)jrj8(6@!fukmqsVn8(n(kcn_>KYvlPK)FUGj|V=<tr{RXGrTM{dZ2eChXTX-<{7$Q4!6<tPb=`GD)j`Ix&wH(_2$D?+Ui1T4bWbkpQ^&RTN7s1CCyGP69NjfNb)XvN(EYQBI*)Zn<qPbub>soPKT<SH{pmHtkLLE`EBg;tKB1rMHq;zsiFl;f!D{Bn?HV`5jCslrlglKxf*2>EZDyZ&fR<8iI-etVrzq5{I6(aVbLWW!5UJ4=C6cF=m^<wJ;9&EjBAcJ_c)BOAEDRAO9hJ-`UFGeorg&^N^{sfBWKH80QI}=Ug<I6~D)K8BTMJDCZ7s~mc(&Qa4V)fy6;jw#kFqLU9l)$t(ALK8?7$~^54J5@OcRf($@}Hx(GWV8k)^xur$+U{lPKiWGIjoe5)w?B2+&XT%{$d3G*Gm+qabYmF_(tSwM-8W!AW>}{MqPpoY$T0m-IFa^g*qL>h2tM`O;QU5&I&!!3imXFlhz^#BJU+QjMpIlK<RL+rD<`t@)sl|Z6fOL>cT>K?H{4stYUU9PMD`m&AQNK`OD6p>8$SZVlhzb9s&kHYNKM{Sou%)U*FCG$MjtHo$vDVHCv65OUkLoyp0eJ5@UHw&rvbsZ<IMWph38B2~64NuOysk$F)zQe4eguaGPJKHx8bVC&<U`np^YLmkK*m&wp3Y|8ms{1=e;D+*1!@WR!<kLO@onW;w>T{z5aup$ierpE+*N`8zrPQ@6~PkLP3^voTbCKV6}a)=Wi{b0(8^I(#JqJdVt705h4KbTrTio^zHl^s{JNdywLS`NE1wEO6eJMOcg1<1h4|VgWk9Y)!4j;_dSa@K(H=0=*BZQ}sjGez6if{*^56LV9B4>L3f^)c{06fRWLF8k$3M+8xVtG(=0q-?oHxQ{`?KsvbUXGuavkLQ>k<5EU79DHL-UL9aFvts}gWOczY=l0YE=;j{D#0{0SsvawjDt@HDw@S}G!9erfR=g>p)a3}%0o)E$7LI}VdHkGKrj-{16=N73lM!SPD-PJm@d&r@&@Y5TwFetIK|K5kPN!KV@XFC+_$7qNhtnK_F_kCt^HCdqy#w&d|%Vgl2%FxdSb|T;-H$_{`j_l&1a{fP7ShM*xi|H%Bd_)oM>RvbOfafVufsFRF`RE#s<||2n9@?b167AVYgqXrOL)lZh6ejsW9_B|tG)btqoZHC&nH*m>5ew2l49T+()XOI0ebngi0|iRQSfxr^yK9PGNXgfFb{(1^Y#onN1gYCgQ#d>8mlNCD5I1?dPKeOhB^Z?u-5C!eZPsrmNJJ12j9E&zdj(*QRBC`o8inN*m$xjc)CYwUK0c%{kV0$ry`s6WI$*!mYFx3!os?z^Z_^@T3K}{ice_5@Q<0wJUnS!Yg#>QX1QQnhCp|8_q7<>jabuqMRmDIfV%fQ=S=JmpV&b2=y)|GS`kz$432p=!2aK1@L*E4+7-TV?!I&!?Fsugi4G-9yKEmTmveG74r+LC3;oskcjjy?do6-1Eg9R;+^~5&-K@b5esX<D9%`28%!pOJ+A`|h!BGBW>>&{=WzL+d{xXTgbA7=`*ZkI}3I*JTUf2sh@(dg=$o@kza_$z+DC#p_9Sr2NlH@q*5WLY5uCJeF-qO_pLtun|};`ei?8S<;DU4_L&8yS;s7}U^`equvF5NRL;-mCV4<(BLW1idRNrwU1W^??o9tfax8swt~B2wvX2RCkP$p~b$-52S$inz$_nJ>%G`F6A!iX;tOehB!yD-7l}8ghERwqC_ipCqm_rsZ^w!SL%-?6q@KAd$3lk0bDVGxPp@vzq1*x)2X<}m?Yevb(M)d#W#@tPoEfs!<FXcm|dWeE{~tGo%jKRU)n7lN4+Dcts#@(jFq1wB!AZLaja!-Tmy4xR%}}C!)-EExMEQQhi1v5T0=OG{e&*~lSSUYQN`Ed@cfy`jn)v|1?Xi<O9Afl6f|?m8fd%hhYztJ!e}zFx)L7`q-4Kk&+sEN?zENxI|hp>MWx68C$A&%)QXaZ#V&qhGif~e@G&0k{oEm>G;qkb9Bk)Z_8kW5?;-e`Qa$`_vf>7Y#W~OI+ozM+{$Jla4$nO%K){K^^^f~K5vM0TZ@UemIoKJY0p-Rk?OEJ5Z_@tei`icZALeN-%5F_PFI?<KLeFR8%qXc=UmR%jAerWbI!Psi3?ueOgfZ!+^4+zCY}4S$LNV<9FIt-clS4bx2K{eT)*}97nn3l_mqmMcM>B0i1}ZgW=`4;gdvO39@E0)x(;%l4cOTX>Z$c;W@2c5%7cPCH6dQepG&6vsb~Pyth)H`fxn0Q-Un(YzSqUK~+isM;YpFDI0qzh@Zv!$&m%DW`UbJB+|0x$pvnnv@IegjT8+<X&H6P7t8*}`PKeE8{wM4Yqketb3`n8DqnU{NC$%(3xT6G$9PwtEb6zd~@30py`GHd_JBl#Wy^(y`|EQM}X{qD)x^(hxr8;-CElNI}qutF4ij1I7r2=&eqvw6DSAwfBlRS?M%la@5n^Iochp{;cgNRCK<U3-|75U$e7EsC<NFx1OPY0m=Nwsn76O8YC8O~k*GrW{3Vl^y+KKD<1FEv@*pn~Wdvv5NJrm7{PtK@!iKQaft9%r{PQLL~yL|6h*xN^huwEO!!#iwL}`*&iVmqkGiyLiI;rVlk;+d>J)iYM=0vP-Y@df_YY_2?xMDYl6w*74X#Zzf|Upp<qhxVQz3BFmj)d-iM5`x1tK)0!_4H=u5#C$|Kha7#*fOf-&>g<)Opx|MK2S)}xi&bHmC&`m;dbZFVJM-aK1KSp!;?2Qg!6z^B2vrg!{T){cWy)HYJuzWGF<ZG&U#AAd=06sRZ9kf_swrAe*DN)(`=kZ1d&PKoc0*z3)u<s6{HCX}2{g%sMMyJpXNGPFk4fJ81o;d*)@5?UpT^0X&!7cS>t&$$fg{@Sfns1j&iUAuRDEL_0s?of&Vq!UXLB}ywlCe@KDXOR4hAG#xKXStawD;u6g`>uQN=aMvkIkN{!6>J&@I<A0!8>t&W6_C`*wFm&Jl4d|L!d}a07&xa4jPA2wLfs7(!Y6Q*eB>UQZ|)^*eNO6yHtwj%1?#&1oYUF5w6JyK35qjXY4lkIt^I=M?~RVr(xu-Fu%R%KDf$z7Zy%pvAzs9MU2%fPTPNSy9G&Nlhr8`ehEc55J4L%fzs2HR<&(XO@>Tc3WNucJzoVX(C4H)Pat`M%FBUWxC`+;joRQ4qpN1*tR8lDHmAO;OQec<85(tzfxkHuQN#Uag>1?yRi1FBh<Y?viSuEkl*8G^^$+zb)2GyWU4$2OAp3rw^%9jPvv*kE&CyuUgXD=%~@HP?**Y!iT14*%Wxih)ssmv+JTjUdaQ+9-!jjikDuR2)}EM@&!px{+aT=>;lm6_SU+@M6wm=T68JvkadT|9DF=y}u_psMHqlruH1;V9|0Rz%CBlNilPe(Kp~o2Fngu7=1h1yZR-`f7q(TTZ~iApWc#r2)>g#NUbi8^o>nSz}DO4d@?izbGepUEE8t4T4S_<}o*W=GijDC0$Z%UlIb?_;bRRKK!WFrGbW9ceX>pmdb4K(s$}0{xT659OUz~(nr<U%71vCYx`z&vjD<FYEAC*twL6*=u3>|x@hvhOu6iA2<>#fI%!i2<zpi42NN|+)QgSu;L^G-CxV*M0n(yc&ke0+>>V%>jK{|Yp)=3^&wd7CTPJ1Fu<mTi5GTm{q0bC<<eA9kbP(W1#LMW>WV&WyT|RWjo|z(|ivC^YO$h#;=Vi^qa7m{SDcXNZ@ss6BBc3CxnFb)1Ek>E$sKmS>Is}<iJz>8gKuaR!M}DZ-y3MIMN_{^mR;#>nQAgn7s?amxEJ!%2_UTYrS&0OjO5R2!^z!qu$v)k6*>USJX-`Ig_1IFDM7cg)5c6y?kTscbcV8h1phHb_Zb$TaS-Z-KnzPQ(BB^LN+UxWTiYR-0TvU^2TO@WtpD|0{gIQT6$5{vi&&F#=p3A!r58GfJK@VZFLA4z|!ourB-49qY9Hd7ZF@L&wjBHmKOk$ZO-|O~*n&gCL{8lgDh{=jPD<8dl@HjyFw%gSXq<gnW3{ur7q|rzwT(M3vxf30sR}0Z%gsez`EZh2ZC{;(8M$6h(p3v7YIzdTyHa}6;?nPZ!2!9!jyX$WB=K5KRjS~J?rTUoP4hF_#Ewn%F!7e$Ox<E1wD~qLHl4SHXwL`8+o&m1%=S!)HzW5799U&#wgC-Q=0jH=e2QQclaRxJlBmkxgFz{QEG-}=_ya=&jxxb9)=zF+g0f?=7kYFM@DjDU5%UsuMrmt5v`wJ$IrRoa9Pptt$;^k@5f%8UQe$kv>HixzE_V-P84rT&k**|Z%PLBSk^^!kvxj%}zbnKySgKp9Gtp&cuv7@@8b;ceIRw}AmwCo0MD&5g;i}uwuR0wa96044#Ro+a@=3^$4B@yOZI3(ZXv5c4X4mO7!KN_)5^sfTZ&9ENd0Hcm8SI6-W<anjbx`^lG#lR56M1f2o?*Zold{Wr%e9mqJ5c=huA;bYLNQ6^}iK=|!LC@84lh*<+?-=^@RS$)GZ<p5@-3Xv|UR*x&{rH9GzZ1)GAd;jzVzb|996apk7Sps{a+3DfvD!=!CGJSHmYKU{*T36XVSw~e{k<xSZdbBIBj2)SZ{V@dVy}1K(c^=BxBJuCA9}_7ta(V$M}i4Y(cY4#R}1I9A)Jv=w~nz#J5Vz6LKig_th}2ByJ1N+X;=I+4#P+-xd_k?3P0h4h&J<O>?K~R$`7-|s{D!F{Jcd&nG!GZ@65%sYA9L>ovra|Gu%6=JTYR@4Y~}b4Snen6%}#|CSdki_}+s#5!f=L|CvLhLeX}kk3-98^|zX=Uk(UN6}xpSt4&gEi~J2zu}r(NKTP?Q47J>RyDT_T>z^?-F2%Y3!K-44K8i4iQkO9M8=Bf1ZX;_S?IN57vY}7<qKaC0tNdUnGrrNBjC}r3N5J+`KQf2hsbG5B{5J~@H8P!2C6i=x9JRUU`%mwKPXB&WYGI*(;1@-&A{n3-s(H^A<1uYANHl??Y__&A`CVRdL~Oa{EyMg5>b(V^ELbUoIW_xeCz|P`UW9EnK&K;XXFbl_;M*~J7QSnizu5AyZrUe7V0+*-s`MEUNR<Rn?5x8gBO2&f8@Laftsi!KvtUm^yPxK&klKtD*Xye7sV@|HBYc5z1j$HK)@iCm6=_Ola6@ly=1o82a%!z%qy#)GmJB4p-><h4(I`Q)1ou0gSrr{=-n3iZ8MESX1I&VgRcx-)v6;Kc^)OF|>NxCNe`;~^{RW{G0e)FBhHpok%oIxN)L9T=<)ZF%-1>9;?p*1xB+@49MId18)eUTuKuH$}sbegMhj2}A-Mdbd{#DproVYalVaT}}Zs;Va%rk>`yrvoV-o{s5=BqhvPbW2En=v9%!01S?avYzmIgn-)i#)ZCyN?`$wAt1h!lV}*F#3kDbZ!|c9GK~)7^-Q<46qCcc1mN}N#(^&N(#{=)2>3}jG548vJ*ssfI+sHcVlj%r;aD}ah5Gk-J8KE{HSp^QJ=brqp$noSnhej^c5Ja94c_KV&&MTyMWpJIcxY)z}AQQy#Lx!ILv~y&^+e{@q($<T+=Y(W=jL}bWW9}2RADL;%<T^6}-0L^5}ZEs9+=T8X0&cxt4L!#0!r-<4f>-I0v**w&tdFzITfs2QS(_2}0?6+1xl&N16KiO?#EiA>vb@&nolIZ<~)j?imdy{g;3kRHZm#O>v#?^xG<8Locf9Di5u5_|DNH`QZwkj|*YqxtYIoito~e0S_~;4^E8pOmL(!DT8*UlJaSUq>vrxHOt`b|FF~Fm{k_|?P_9%xwV)_8Hcto^6v`Mb+yKtv43lCaDo2R;qrzc%4{@MOKS+coF|IW30iC5uY`~m_P0)iG#tw%9G4C@ejMTxr^0pXO|ZBpUk~F&>eiL^#~<_|%6TD`wmy!KoGikeAaq>d(JOleE*JM^njk=Mfa6U@Nmtr3jp8x;!&HSLC@F_p1k7^`T2(?B`a!~ZB<(C0B@Z+r`{%|)E^OF5Sig)h?9)PYcl&_ri#;@f2?%J)vuYOdy8y<u@N5u;796c`MK<jmvR6aIo~)T99vsNf>;;sK`?QntEXbHa$nlHpsHRCx9tfXB?(}-{1T%0f_>vMwk!F8kpbXq!T@c?=XHm`6fb$eO6TCAvl_Hd(2Ko1WJ?W5bc9`6ZK#$LC!5rDF6wNb6Dy#ChWeF<xn5MB5XG4J6atcK8cG_wak6G_sVrqXT0#=I=_fK{Xl4fAwX3e$ZX>)9l-am3$II}9RxI;|Z=yxmCD^^tXzxA7DcP@x+@Deqz`XA`=fmKycW|83ZVHGl`kRucTRO*0fP+b;pv&l~o7v<Je=YyEgJhmZi-6g~%aK13G>KoP+0-@*eiHTfVjUZ)%=ueP;Mr?A!-NdDeM%3_5AL{s_viu{LyMjFu^*3k6GJ<U?r-Kly7@>h}5IeyD0~MSO2kKD*eo9Y6@~5NEv}?S2tngV-x?OdBq3YE5`x~~z+?hId#1n)6S_>AHQ9(;F?YC*%E?lkcry9w?d{T(Ye0otHTWpKylX1A($EV#4#mDhT<oMVD{`nC{kyBaHoa!HyjJ|S21w<&<oQJIVt(|<Yq-)bqJ0W^e$#W$dWS8Mu8p_alsQ=PdGgeuKA8Hq6M$iY-?%MofILDOK^tzvewV4!(0UW_gHQ!waEkCR*TIC;v!gAUAo4=}7J#dBa;6wgHKn<ANM&_D}Y=P@D*ixdx+27ef5KFMCcMj?P`5muMHQN<V4NWJy%$<NssqgJQI7gcAxC^N3g=S_G?`N}h%uye*W>deDR~7;X$3_!%zo80pWcp-YTbGWMznF|%MdP^#b-zlJ%9NEG)7e})EYsQ5X!?Upij!UrQ-P{U+O9MzrUU2d$0PR<H5mA0Ff$2Yo;X<b5(NtP2fKz1sD{Pl!4CGFd(d80<w8iwmJMFDE9E-R4sU31Q|h=`lA~w_EE68YsAk5&Udzi~#DouuVdOklNF~(z$Ot925&J$W(e&Hd&lr*=5J$#%#Z;5fQHtrjrjWms#150TT-|R!VA&*@6Qlsi_mXvV2!~Jo+&8CZ)>p%M_EQOKZ9?>(oOBxs8PS1CfF$?An+z!Eh4InF-BN#ppyFhXB<u*<<J3fk?As|x0X_6(AJtQ6o78WAt9+v&XNf$4?HT}8TYV;>N9`mgoXr_ENzQ2>Gz>{;3|F%CInBg5DE3rA<+!mqvKrCuSVsgCG{fjtst9Jew^B`xPy1myP%WQ^P7f|r#agnAPg4ns4s*$GP>jh42&Y9sT(6!7D*RY;IfIyvfQ#Ylx*>d;*EyAg;)&rHP?jlsfumqA8pmeuKuC$HvK&SfbL{>|0gSx;LXCF<I#-tgX4=&hA3hIzM$<;plk}V;4A#xM&CWd7MTbYii|WMon@0?Qe`0<vEoCT(KF+e%pvY$i>yn#uUjwlWoxh($Gz=_`LLakwEiPD8b1Rz7D?bcUNJgn-(T#JGjUj7WVQ~AG8W|M1yg!CmY<GERBSUV~X}i9!B_1lS|62WDMseFR&<?=ukxPKPRq@Vd5Hkp#13$|WaheQj85sqsSa$SzvwA>|Wv&q(7qSq6e{-f&u9OSMT1iFrm(Fy)<L;ajvK1_qQ4EBO0_yP~Cj^$4F$Z~6>T<;}fQS`1em180ZJyQ}@rIP8e(KCM*;4B0Xgc*dYLFw}z#xC%g={)Ki)wzz;W;0AC}&?@Ip=T#!gIr?`1@Dw_`fUi9OAej&&A2!{59khA?xNnx#-uDLmLXfb71$UfAPDNKYUS|v5rX?tocj3Utbh<HJjk~$a2R#HsG5$b9kZPI>m)Pjd{G@Z^QxEiC5AJ?FCv?n6LJY0q4!A>Gs?p0!O!Nv_3P)_4C}@s8)c&#0fiTiDTb(yU$X1Oog!dn{-K_bi{=?m(S^gVegmZuXm$IEf<dH<C_g;@$5S0Oon4=b5Dlp82y%MJ$NV`Iw2m$>Nb%s<8vrgWGu+o(m5Lxbg?ehbevbv-M*WWc!*{OXxph7e%Pk*Ew0pc&#^RYUL>`JK`Gstw{{4YZ5#qK^lShvw5)sQcl`XxsGSQCB_=-!S4D6k#t%4q9_eBUB;p2NA0<9$Jh()*<h_$A=d^J8sWk%)ydzNG<`&Uu^4E-R%=sA7ZNM_(B+MG~_42_yTPwObU>ep$9(n*%^XVy>vEUSk<f8}tm=)qa1(L)jy#5~$m`7245CfAHi{Bqne^E7PBf4#EP$ty!dE-CHQPDaf+-)1`Mc7e}#Bj5tJEvx7&P4^}yy!Hm$1vj4<dI6`wh1Z=I1#(U{(FE*qXv`{+cY^;B|rp`BODLPZz=Y+()PsfoCQ#3QDQVeWm&-fg077-hJdl6ksu9U?YkUa^b{$;yn9YZ+6ClTE^$-OG8=KFURJ_Lr=CpPquL<xN)m#cXwu--rJpiXgZj}J?MIfap}>Lwm3&)muEWp=E)Ehw!B=6mpL)>p&((;G9wULO!_oS$vyAs9&UMdL9{vCpA~UaUof_WN_*)=iu2JseiWjfdX*kqiwC8Xg0UQyz0$THP#RRIr(=-ZU-{JdHZZ=^tI$Tbh?Ex4vXK<QE9E3|Z$0e=bMYNEY)%e=TS~qa3C(Cog8n+;$lGdzGbOAF&42^)xQW7=|cxgz`D)K3*48LQu#2)C_>>EPDfxnP4@`0MPf${%Vq0U~>9$|3zsY5{=Lds*q!9lAdbDsimq2BHMQ=Q{%pC<3tL=3>C(s&|kW8;}`DE7tm(!*;$xI#)1?sNbf2$Sq*rn{UgAi_ekx?t-WJ|$9)I+ZRd)sK?BA2ZEaUQCaawuFc~le_Lua}!Ps&Q0|wO5@n;6TtK`*!I;ltBR%a5w4etC2bbtu)FTx80i^=-g7yu0UJQKVTmYoRLSRrw&=euckfKcmFS{Ugrt-3g<QU;`0pdPOjTT4t%p%j(#(eD@v8kbZT|+JAz<Do5LlZzv%}dqr=m)8$nSw@5X{s=g3=hntbJIP3@)Wy2}ht*^Dsl_+8#od$#LQT2ipuOjTk>Ww-|lUaJlZMV+T6ccB=RJs>`Nuha3!s^RXznx%3=UrY)^EShJs68DlG?8=jXuqwR6bi21SYvoDX2cB~%s$dx|v*NakVXiy7T&Kd(1GDE$ttOx-+vjg2IQe6&CQpXj)YTA$N&vv>)JjBE_Hs*kftwVN~%unTrs?8fNskzrcv4R(W!`rGMW~Tu&9Gqqk=YShzz*{~Nfs*8@#6F)#GmrLC#wP*~KlEYP(a!SKL*c1|ITXUQ=pBA>{8&AaL)eX75W#{IDOu}umjU@f(2)^ACU|RjSy^&VXTPwfsJ?(5qv;E+&gsvj<JbIgaOh^`{jTYobKjOdSh*2`1{&^yi&pfJB;PHH@UWFD^x%b*)3m6uUX3^63D0UMAo!()2&(O_{3ho<i-rBo(O%B(A7_qUgV`oNS^aKsp;-<90YZJr9h}3du)>3U2kSNZ2<|ukvZH$JE)0f)mMUQAdUyaXQ<YF>YbJ+qJl{S`3v9)KX>U8l?$DFd&Wn2S!pVz29fIlS0^>`Ump`oe&Jcf^@Oe2DrflqlUCjVeY(L9@#i2RwaQN1mONkSX&u9HJkTqU@x=cI%xA}Vm`Wt!l6LAWwlEdbhPu_mB*z<oo#~r2dT1tUFj<>bSLzRve0$PJZP7BLR7WrZ`D@5Pz$i664S#JOb?n`qnXZ%h!*eP_ZVEClQVWdbT34*6cXpXxQtU1sib~Y3^d=6@RTx+&u4%ojDS&=Som3*oOze?2F!Yn{QW$CUE3b0{j;DaLdM73ZNbsB<bg<{FHkk$MOgLL#|5U>=lr#zHqV^`QzvEx-m)6@*Q8!0;Cd$au*4|voyOS(HpZ;9r<O61anDt|48Z~2FC(%=WCp4n4u&_&;YW?a%nbgE1+BK!{WYA<G(;}*l~K%ddGDi>8TCTPUn7{f6~bUo!JKC=JUp)?{D1>(QMw!yH<3coEcIpn><5jiEzZ<5{^PK((30z^b?)wrUuT<)hGRDl?o_)Q}|P&??+P`$)frC!kd(1#C!D{1}EnXvRfxmikBw>{KeR+}#9j+76wUX<kOVH7z#;*0S<)^9K?a}=QUog<wBNS$|Cn(uU49F2ZPqWGPR$GKM@%s3@#h*ikPsYpX!KoNO&Z_3Pnz%&bS<5MspB|wVtNW+I-)Mr11Z+MccE>6^j(5YITG4Qc~WIr@-%IlT%aaM{=@+{$^--g*3Hfv{;A|-VT*^?1Py?O0LRPFNppI6J@yA{wmY#dym%=3b7((Ft#a6`>}s9zmlas(`tFfQ;%7=?m$3$%>gA<V2HR~4S-K4Z?M#3dgRy!uAMoz6VTx23mirU3axvz6&fJV{XpdJz^LF|?6QZv7ep3f)F@dSnJL2}$D&^3_Amwf%EYo7Ok38%VrvHtKRn&$#j}fJ+HQ1uW<Qxj_(E&k5I_KHIcyX0tYHYdA@YPdpi~Cyrv8DYi0-rKokb@u5FEjCj&(K8DNC#(#U#s5q{11oIOY?foJaFq?INP4d`=oexzPIv=Uh72-T_aoG0#?$f;oC-M->0!?0u30HY1&wwB7+IO?av<<eirjiGbD@S={PME~^Zy`1UJ39eU8#i=3ELJ{z!3-wZFzs?SBJ4LkBxL0+$e;{wn7%KlK#)4PcUvW#MyiG+URosX7pAP|D%Pf176f=a`(#))nM=-V(Mav!{-g%DLWv;aMsP-8+LGPoI&W<7oF1!Js^ev2fsF+h7c5<^DOaQK`aMFaJOFVe=+>plM-NeK2DnYIe5cHJrf51G;qNh;qbv_G>l!bb>~<e?fsgN*TS=bAWgQTH#*8@fTs&U!>b8zW*U?=%z$C7)1{W=qc#JeBa^{N(3uA0|+ifT|aWIM1GoYzyw}6g8i`u^~0%9ahU_HLxhEA*vHh+JoY|4;PaJ|&R)oM7Y($9$RtTJkA27Vdbtl|P6VW#qMPhebRzb<HKi<TG3QQBZ2i||HLV67si(%Uo&h}iJRB)J<D9N1H88d8M-2Rlm(G(9fW`PI&w64*Mx+D4Qugk5eQcN=hJEs<bhJXy$oZ7@o%3m!EcWI-bnXRg<Er10(W(@mn0hOQns@Q7`N&_L4Y+-n_N70V%qs215JHKsfezoWpYVW++W+yf}=n~qI9hLSsmsl|yR2Z|&Rp}1<ELbQ+&Sk6r81%p+1QBV&z8OvY(mrcEVRqF>1l>!BUAw6F5y%|I{px#8jFZrsFW1W4<v*L#bcDaA9U?<J5ED0e12WyMt8gbTH4<mE=2!u8$a^nO?Z3dCT=seEwNpZYmJU<!YgIzXLXUzvZ#Vs^9Z^qq8ajX@1r~CBa6Itqx0o#3b&iT$?M=Uup@_rBFzw)L1q;4ujER5;saW%XhWPENYZ>AI6E@vV!m>XI(OQ9W|H-+e<rP#L}*+bN%0gi!*5VoL|S-D@sABJ|+r6W4ptY}ea^nZ3%{@8EV9<5sKUScu3E0aH>)X1X~;&t;g-ZjtrZ+w2$WONfQ3s*kr5+Hp5i75`&_dw#UD@!gy=rF|<9>2!5gtD)f8>xB`&IU><g^<387Z&=Gf92c)Hdf_~fhnISRXs~D>bam6&@oP#8~#;(<2OB786K=PM(+1s1l|^KH|5~*EiC6;jSY-HJ~1PZVEriI+)EF<M1=<@a&vnnNLT9GP9$7o@H@*5j5FnD#lg9S1F`GKf5C0j-nF-OGCW>_S#sXW%{kX}n5Sav%eIC(p&g9Iu<}ppz@kHauXEyh1K%vwhKyoT`IHC3YDY|1#a$B<jW9c{&8O{+3+u$i+w&GI7aSZt*yoiW=VhIj)}JvLV<+y~JBJAO+iRIoYYb<u%@sVytxm6#QuVs`IDsBP@2a+<S{tb^+8EuVwEZtCYha0)j(>J1cVOZQJQ|h~_<Bz@X=mfNH^FKh7n}{pepx~KdP_TeOiEqWi26YfD{JR(<Scp;7H2JaA~c77d%gx>yKPOoiDl9?91m|m#d{F>Y^dr8PBTJZKBF{U?&ftUHoSad>lqvh@_1_}m4AJZmzMQO3d7z29e0W3!Jz-D9c0b))4cy2YA06d&Q>S_l0Z?uC|l)rDVP3oF|!U5U8U95p%#@!TYWAw-q!hk*rWE_a=9>Cr2$o*#4dZq;zm1cFDdGFtw>b)K1N(=IitWVQ=Sv4Jr0!|j<BlqL5G@Tq+5?e{eP$cwTdh^%5RhMwn_!Ymn!wGhz37+19CQrR^-273<Gn77g}X-Af=1u6trtFXw^vuKbG8{l<jpO%W<p{CUpcI{!k2vEIAD&6!5~2Q!Ftpu(YA3n1Wv&ca&rb6gl~jQJtM}b_`uxTcAQ%kP5A_blnp)qeM|LY{*uiiBML}p?vjpjx9{a2|>M6vmMX!FO;f;0tXB*e{|N*>Z<z85ZvrpE=`Lj#(Fqy0iMiqt{wq}ls8Av?g3L6O=lg=u=`YKk+I0RCf)4?5xcEJYA5h1vbm{%{+rJ%@j~9DvtI#m&mirclp&z<uYE9KPdSIX3{?y|k+L}@5gltK{4wkW;yN??=6{3B%w=3wB@O2V7e3Z#IsT$dGFTCgDQxUFYiyMeO>4w_MBnOLVX6+Ypw&<6c?nlP=4gEZ!mPFK?H}@{OS3(R7Yu2Rk*oRihLT`s8TMa2^^WK@HaS#Oq-8HoQNX44n^-y^OCw`A=O0Q$-YD1akzU|9DWd_1`OP{hvls5$>XP0??;<^zx{0U7)u$E+c`Z!W%*}cQdR$@NCEL|lMDF)hSf&gG-yq7KtL2BxYfE>coS!ML+h4jIsEm^^Ae=Mp+lH0U7st7!)T@QCcBci5l<OoqeooKb(^Gfp^jq#3S1?@#@o7D6Qd$;uTrI}H)ccE#mWG23nh{hO_u}O*M0_`Bv1BxTt?NW$#>nmS>VI(-+G%)}XP-*LJWv`gPW1F?dXd4MH!M(ITj8xbw@hm{j?9~JUUd>aMX&-`Ch^!7==crP($~zj4Pgf~VWR)O4)amO>cxfQ?}3O*x{L}^YtL|g{OB-9+rGIkj>~zmBa|>8J-E|WceG>oT_hx|1)`{f?@tWu(U=0X6dkA?$!{a*EctFuR`bhEFp-c3NR`ZemiHI8ASGR}n6Yuz{rK0F&zyBtd8^NQ-}+7_Y^(R${D<c1{?~e1q@_!X42(swLD|y<oGZA2<!t7ZLrO%s5=)s|`0}jqMYYpaQGwO)2STQjDOa6wVi7!#yguF?VtLN%*)b3vO4k`YBDhytIay`APFe^;Aly{knc1->ik1%5a55ugPD`XuM?g)NWV>NB-&{@R83Nop(xp&kTNF}9CV9A(i9rJlgr1+R!>Z)D!9qZ7GF$_k8b<P()Yr!V#4AK;6W~J7F1UCqoI&_<E~)bQUx2$5Zlq=mp6<SbE8X)OxL_sAQza>GzW+O5rnC1X(AAdj>I>-W5aDiUKY?uC$+ft0-2A`z-nOLX8996AoA8(P?$Ky-hbD%RX=`aaVSjwbx|(Ls>j%IRL&vcffE95B1SW@KjUYEJFzlX~R;eoo8h2e=T^vCoG->Q;QWsmDgxUa@W1QRFVN0#037e(Xr44kWyr;(Q#cM~SZ?5K1CKPkk;I;1@Ro2aus<USh7!kHAP9Z+DKIU9ltLK=2&FL(kOd>No;Su&H$m<8#6D$y|toZV?!5AtC+H14XGDo(?GKOy-5m*juZqs3cko&3Zubx$Q%*IG7Wis!O&{Lr{uP%YY5JT?V?2#@4%m&d~IuvbsdnvyAVvWFKY)vvT=Wdo<nRPr(&!M>Dy_o3?lYcUAYn7d}p_1&aVg4q@YK)ltgYC0Zgi_#D$Q?5++WRZMv^yI7FiBC3Q^gjg*^M@+(pnoo?{Fh~NaFp!<gdCfu6^Hgp~QB^z5Jg#mocr5_t})hCCS(<AGmDhzywUXq*?e_pQuJ4@r_lP*cyx8Cm-dh<2t5{Yj`ChnGV^L4Cu6a+smKt8h8+Z%})+^Xj?j02&h^t&zKGiLT;!Tl<@)EL4vjZM^^`tuEdxC4W`|O4Hw88qv`iKAs={LCNvbyVuLvs<JSLQx9?|(d-{G3fHOBqp?+7kW0DQ*w(#6ev2RR@bAjUyUw6{;vTX2N`5$c~mYh_hdBkNmu58`24Umb%2~K8J97sDlFUL#CwQ^G$U&V7zgT}Tp|H>NR_4ga9ub4?{WoM*}%xc`EXl1EsqCX_~8E#Act7Pg?e&SJv8C)$8^xY@JKOe@k{Xa1c84aA%LOl+@%|Kh@GP@5p%SLSUXP#n8tN{J@tdX&|?J4b%2<uSHF^(gHV_laz*62yF2{;aXJ8A2qHxm#W2t|E9GlS+Tya~+qt*T#1|Lw7?hILjt{CZ&an>U_cL`&`o(&Kw*Q)6%|Y{k8R+GRzTO9b)f`yBwxcYIK>`a3d!P4u@2(yh?HokYnuUIUrf><ac+*{N@dZUxwp-Emrn8*|J%bsz2eO^}WRJyDdCb<mRO^T)b|09l!b^<2e=Cf0|KQpC@0HtRN4h%`n;Fy@|63(9&cs1Hy<LmBQ%P%DAM@)@Z=Ad$MvR2YMOkXJAUfWYvYZcY3JFVvJ)JfugNpC3R+W_*m{r7qH@3tP6W^eatejIoIh+(B!*z87*yb~Rnw%aC9lsFWvf%Flyr>o9slcf@M-FYuQU+FR`&a3Cb^q<#ARJIm|;hRUGfE3iL0{#Oa_m9vcSj=VZ5fEaz5NCpO#3t$T9F$71^OEo1pMhC)QR}!dum+q3Jo8UnFnil)$bV#aT)i6;_^dNhM(eEc+1Y<fm+=(Wl3UD(lX2&P`KhX14euOxcJYzT?M<q)T*yv+7cjPFTT3y{sDCW%>X~YW_-Urcf=FBcJ3B=MpNPHs<{lSxHD+0q{z$?V(Pj?f}lDhXzVe!b?<xfa~JHF;>U-mW9s60wywJ^BQ1?~lD+2<_RJ{;+kj?2B}tr*vxk>_ncLBM)jm1_GgNfG`W??+|FcbjD8kkNzjJM*1>r6B6{$-f5xc5*YgIEG2wAB^Z}#BPK~c9-%;I;sK>79A|AgK7ZR%L!x<ob;%Ay+3d+^Ov&0CUdk%ro97a=T{ho6OEprtR>5cwd8<pi}d9$j`=(dH_~}bLVymp;IwZAb{w<qJFSVqBA&?alZ=*EJZoj$79VSUw+4~`-1b;?TN#;brD8EMcGU6;$q(KHpniVWiD_2IM&KljBI^WxW<VXyh!V>_U8}SEvThB&zoAT*%<&B0evr@6EUk5cMxj>GRM>qGK~I(+HQ(lzwSPLu*buBji>S3~&$z{KP<B3bq=~GjJ2M~ePl(ovM?ohh8CrXO8Xl~aa%cIF+}b}SWhoq1^UP*h{XhUEOInjxB5o};9^JqezzlnoDgu0BTD<Yi4%cV!u2P|KH*FZ#)|@Z{Xw+Xp+qjaiVWaM-vKXpR3$Qi&tra2YN`Bf~FVa*#6?}U+XUyGBIjQR<_01k%i9vBCBqw+Xxh^PC>fu}SF8HR)G#Zux){ZRsg;9S)tA$5(9Pj476%rJf$@{b6u_6V_jTmhhlKE}vOrbDVyB~DS!BogQ;yb)GPP3oLOI8y#^nqQ^`k0ug$8N^MgqH8Vx!xCkWR(mWr@^-*Y!jM^F7x`43#|2m{MQg}jUErMl+U?}cWfT-oY)w<?~1V3X2Rs^3zM7<Y03zC?A!8D4w{JIRt<yhVAHHt$w>FI0Lf<Bz^nT!YeT9)ZsoBW)j$~bps0En+wl5p6wptMt<lsUjcQA$3A#EzmboKjDngzEK}ynPglp1c6aI~vKP}s*gmX7h!A>{-_);Iv^!qP=qX{B_aSCtLj%{2}E%zvdY}n?<mIc|XTGaI+rlrAaaPI=;uT+L+fB$Ig2!3fd5JfI91zK<W&~n_KQ3YrsmsDJGT`|P)(MhS05cgMUN>s-u2`gkp4$yoY-mdXU5*OY?4fK=s49uPzv+DL6B3G1$isA<Cl9RC1GAh4*nCE^cu+w#`UDpV{u=tcX`HQ#wZW?4xE6SjLd%#JqgJWg=X1wc2pwP}TIrCZZ|Nf1&LLq)A&p)l^{}7Z9N(m%nUd6|;VoGqg$58cXnt^E`5;nCLHVlt84~6rtcxphC$jHVMQE|ff*?^Jim;PBYH&PjcgEd(QTf}YYr_mcJ&^j+|`;tc`fiaMd8HNGs+NaD`gU~17cBH@{FJfj<x-=10F#O&6(1JGZHAfJoNsD_)o~D!O1o5Em#F;RVcve`6yI@N$m-=|%`^7zc0uf^QH+xhI%BIFmfARriHp{TS>K95txjJy*33C;roS3#ZE@hydxYb2{1tc!dK`*qtuL8$!Nu6&6(j1SLiue2Wy8k2-JwFnc&JY74)`|)v6YelL5MGHyckV<zC@C3L`@rt=F80>LlHlxP{-r4(C@UBAx%LT)kowBEVuVFsDuWR7X&G78#z7tt4ue00B&-e<BIc}YwB{K8>T;fv^BQI|Gwe5T+(2_x?|j2TlynbMd=I_D^D<if7!7`!4u7p8BO!y%FI5BN1{wT+3;_R+mYk%D4Q=wTrwHdD%+%Uhb*nSLs^XQ;%7|EcOFNBmVx{D#RA_!AtjG}b`w89DEx9L%MNn*6RH9#g87W)nM|jFf$0c(IJ7zMHS+y*0d!dk#>0-`2{cPTyPZps}D0CNhIXPbm_i?0fEczKoSaeU1Wl%#W$6feZXgY1J$W|`L_G8t_G0$Qwu|>J|0E`OG?u#U)qDXCZ^hU7rgyP4}_>?i!4za0b`uf$JPvI9MTHm&1-<o)t?32^Ns~|Fi!|$R!5vbqgG^a591)4sont4C_T~%FMnMup0d{g{`YGZQB`*v{=hTEZEu}^LHDYo$EDCl$W479l0tox(W$I_>H)y~Hes}L0`w&-VDRFqX&KU%0EaC5j9dT63w2sMiV`7bf#IPr{G#TF}7Uc(O`Ap@|&F?e1iBQt8Y1t%%tTU*V8c4QBpx)s8VR4eu%8@(Hf5Fa`>ELQ)Qf`{@l!c$Ia@;a($qqOR6g%J`qWI^TU>Qdk{kan;3fnpT0Eb=Dn0Ku}^a4vHz8M$?>g*eypuNbybnDb-WKbC*>Dnn-CUpuy+V=E{89S8-Hx4oMGCB<{R{NTPegVw1d>R|sj<B>wwq`L-?4#x-}$?f}+>V!>N5ZM@&iQqKj3AH3uK*3>CM!(0@Jy<3i=yg^9-&tlS>>^(eo76bZLL%)~A0Wh&0<s3JVj7;f$hr*SEPqi;2eJoERO0<jPz1_X1g+-!Nf4QGh=1`1-LL-e$O1uXHj+|~gmt5n=;{$4m0SftMHfZz6B>oq*4_ZEUAVpZspm&zHhogVFMluK!@XqQn$|}9F+wc8vJon+#pxaK94KYJz1Kd@t}4Y5&bF;L9~a6-^-`*m*_B{af}V<o(|HU!j*~sA@~%biq!h7^U3Ty17Y%<WAdXIX5PI_dBHEZNNjFMdB+J(uUOW`WT_9e0t@Abdi#x!YwMms7xKVrX_DsYthTbmlYmn%P{}m*h{6Ss^pB8|cJnCmFyWA1CDD8FdtwJCWG{GAk4)<KSzj<~xfpfquAQw1tz*za2(u8JHf&1JpniNf_Z>We%!~Z)?B-X|^PGu)sbPQ!(=uX^KvL=h)wRI?$#93f&lc6BI+|{*mi(X)+>G3Uqa8fesM)FuUi1UEXv^bZkB38`9%iQM60)yw9FXa3z(WEgXu!2~IL20{uKR!EokU;?DrW2Z@Fn@>gJ32U9mGp6df4R2KN8XEkh|12{qzddFkIhGwH1DU!Uy8Xgs-XhYj3l8}h<2P)25M;X1%1*eOs7&kdwG4_9BuN|@k%yzDk4e`Y?yq^Cw+4A+*poeHIw1Y^braa_wD&srDUwVWdaJ?Vk&NFYC=BuRKeYR)-I5UAE9d&1$<wrkrU8|PdyFc+ze4(RO5nnDmN2m=5-ZSfhngFflJ8GzPHK_;g1?`;+f=h&E)~1%D0+x&q(Wj1*?ZnS{M_vQd{GGyJg^7I8OfaLofG9v6Ll|j|MqbM4bH2GhAP{Y>3bDKdsAaxg)e!syV!};L7ulXqWP=8|I>;k_R?K1Ub7IZcH(i#g`%-9;`B;XOE$mk;wR`+lzBVp2#1-i03)8b#tBleXXYwvrw*CVsliL$;M#d(MkA|uaTjLISCKbOE<>vzo%H&`o4lq)3eF#lh2L{mKjiJ8@vI)f^0X3co0unaVUYx7;*{Tc_dY^oJTi9NXT#TOfm9ktYuUOOS6?#7=%rWT0O<SV)<bo$bYeu*obQwjkt>XhFSk4B!JwSs&>KO0^??=guvwj!+)N!eV6ZidV|3=VuTElbJNwb1?Lc^utGNvw?|{Nk{G&r5_ndRLoR~K=aH8@`q_ggKv^_i*`>Vpv})2j!iy5E9R7cFV>ll2U>e|d2pq|xGmbdDkYwAa!%Ds7>8f`Gt-CdyyGUpt@i{QR23Mi@7s7jz6~>VarYt2s0(r&uBcI;iEUp?jmA2N?ZCSMIVQ-Vy-nv{Y?^wnJaxn1OQnE6<PTf7z3G(FkV_u3AkmALAw0%=|PDr!Seg8?Sa7&REEa@k{nUgeh&IbO)E$S!&KUY{OVcI7fp!<d3<f;dL3Z%bNfMfFk1#`-@=x5#ALh4$3zm6043AvBG`gs7SwiCCDIILGqVI1OJMbpwtf#~gg#oK=FkkvRoaM4hi*No6wKZIdutNCDGE8?49qCrF`%~pqwfnOOYAx`GSIk!kTV4yu{PMyWJtVNc<(t<)dffIpBDeO6~^Pp9q3OycXez>p6^=;$YU^l1-RFe@G6_bzlX4dt&(%I5$90NE!V)k!YSrn>Q9NxWQv%cw`mQy@Kt~L4!h5N&8@;-u8ul?#=-w&%D$I_?mL8{+eF8TWsHzBRy_ciu2BDY}#@M2w+$thC|Q-oL@o_+EnN^e8&ve%(TC~oD-VDJELTjqe0QUJZ^to9TeU+C3_;f6<??83VbmAlSCVz%~AAWhW)Ntb|mAna<l^&jCfDei<6>*_kE9rz=Oc?|vJp7w`OjI8Zj4cU}3dmxZq|HIWXkHtmjfJ+=$0M%2ozzMfc?+`ThM!QBZRk@bjae21AWQ=;5C75UP{@q*YVsW*;Q}!2w4(&Z%v#7*}V%_qVUDtvrwNW{9G+WImxqV2#;&KrHy@F}nf%C@cWhI>bYEQ)i%p32j)ZmajH?Uvwc1Tvd>EzpHXzo-Bb~Xo3?QV^My<<p#6$nJIT0@p@TxnH5XBys(@~6ajECF$T%#6S4?%fT;Uh=qIQmR>pKPsC%^+eiaUS*_N78y20H_P%oLcp7>Y<pk0B+=%WeET}IV{21vBCqG^?`DFwOHj*rmEKc#RE*-r(9Y^PxIN};WOzh10nc|!=WBuSs$iTL7txmxKw0UQ^|hIJU?-yG@KCBG=mMxiKtero2E9`P!LglFr{I^M*snEzhK$L8LR?yLa=I&DajZbN#HhppfrL>Ng2h}lT~pY;JP?qtX8*>j9DYl7_W?!iVwx^g6ySb3V0NOV<kK5yG=MST0;JKho4Bc*{&5zOOj)s~nIIZGDVSYdnV|D>O)EFz%^VSPH$&MHxXw>9U70Y$prSTindYnQV+}BM<e&9*&U;0q%-MepR@%yB++Ex(-SP0$V>hS<^bWdA^%P+OiYi<U_J{rExFcEu4cW<XL$J6^)5U{mo-q*a3DiLncOG^F`tJlzyp*RYsLWPd7pxR#NMRAGiJehRTl?V-P9@*HM;fL5MJ9kn0@*!k@B~OTGw4I(n@lr}^e!nKDi|6PH!RuYjHZrinJux(DIGzPn`Us0yaczJjz;1WQ3x#eK0>u}&oz5sE2!RB1_4}oY8GbQ{*i;N1Jd>pVU$`8akElvp$XbRR(~cV31#DzF(7^B*k5_y9lpG2Rq`%{|6R2LoST3wjjZSLgqei@J<%c-!=}!bEkse@9(rEJ8V9$kcC@v}u2_mLW?Z(}E`y}}9mTag>9Qb^ZR_Sz1GGnzJ?wg&eO0~y(m!~fie84==8X?qjy*UTl?HDeQ!7bMWapOc@#-HsIP#JOSC4Jpsk5rw4^s|tbyOGi20I8FN~;g#h8Eww1%w*Mg2@9=Y)G2t&Y%)#W;{KKQ3?ptTj;G-tLDqj1a`E-tw6>S+=a@Id0gkzv;0F+Dy_SaiQQa3Kd`y(ZX&zQpfgz558)9+(xvc$_`lFV2jQmph%cPmtB0LWtBqAbg?wSpO0w7Ud7+!_LUN=BMwLHXn)bc)CKcAkeA}TV1+!(A&Y|J6<Ou;t3<cT3K~zI#c1$F70*Htpy&k0*M1gfzpLrB5wP&l+Y8V@$Csc!SC;9Ch>sAg%sG~RO^AlCqvY=Z3_kd1s5txaHcl=09z~T7nrEspP_nAe!PjDme3_La%C{D0IEeH2J$29NL-sLpgpHF6v<PLkT`usH{L@+oeWb;3__ruDfcf<|fITG@Bh`{&(iIi3X1DI?NtO!Eve>xL)P`eIk!@ghcvK9;{Cl}H3bQTzAz^m64I7vuUaE-CZJ#VInxek^L{eWo@Yhk-;ofR&vpCqob`$zsbYK?ubr3HnyHG#N@+w#-tRI1h={%-nTzS3m7G+B+ocr7!^R7&6pn6|mbyW4V>ZFs=F<m7W4`1*P%u}@zA@ya-G<c6dkHQ!dz({=o)<=lf8n+ViXK^qzGAuG4;+WXDjv_N_ec?mXz7~HL{YnWFwp^RT9Vu^9Q)7;*Lk!t3o!g6EAP*^NIZ%X3b_L>Gd(ixJ{e}X6Ojt)2PWe9(-`EpexyK@J`+aB2#HjfN3{rKtNOf!N0&&yP>njuVJk*Y_nE?hsX+B|zomartE{FIHHnR%PhPCoGkzYWlW6s?h^|5QkYbrQT@=#%@ryQ>X{J(dg;=z}trF-(pS)R&QknEo-0=+yA70LSBh$zHgoFBJ<$9t=6NAS2hjXV+-7XrLl2dHfmrxWRwetXzFaI;2&gE-}<Ta(Z$yQlPoH1AFqFmYVk$f1R!A9p#=S2iihe?9g&l0`%mvfsO@!Fu_n?X>=bQ2w4#AN41j&@3^8sUwCsqou;h6E+U~a(zaf_IP7TZnmnk)sREm8?O{zIm^wH4yDq~q3&r^#IEzjAg-q_v_|l<>hz6>0K4L+X|G5G0+OT~RsD;p<4}>UkNT$__JIf!C<Q{&t-wT+P9m!b=)Hj={TH5fpv#Wmz_sBsWrCo9%$Qjd*NFVJlqH3y&m=+)|dPb5b2aN~$ck6}CpLTnDObWQ1`bJhfPHQ7J>0OFIec$S`O*K0dcw<PYoJE@V$@`j-V1)cJ)2b_H$Y~zi=T^bLkb~C8)(+Xlnl=yHWaBPfZ|r)N^gOtchlOrs!Wii3sDyH{66c$47j)f9n)%$DtYUU2e3FGfzHL!3fNuw9G24H8Ypi{LPfaodZ%b<(#*RGS26DQ>0_M%n>_(I7lO%uGyn6ic`gW+)Z`pPUZrk^9S1g3H$;=+6{(?;7DA2U|`1qU;sg*C;Q@TJ3({z>`hIM3K`U|3#sox3jo0q}bsGNG)7I7CTfW43E&xx9={v>=Ca}YVm#i0Lusu>NN;*Md5i2p61`eb-n_h=Q()VZB|A!3OFH+G$Nf)M>+-pm}XIpGmTlWPmQcuburRcVP!tla&N=oRS!c>0OHcV~9&%69}`v=h<SEF|3DZopfthY$F1)WC0?q|i`;-($iWe&AAi;Fn%j(jM%yeLyVcx?syb=!IQ{ZP;nx<xL!{z+_D4Krz#SF1SQt+}n38gGf?MFW;#+B0$=jGKWM&MkbtYyZ!#Pco{?%7xQP?9OlWd4`}i4`hr9pR15vXF-;Ggo)Ur-ntrV*dbGJnRl>q}E&Cm=6lHk-ATWiSS}Y|oDpda8FWBT{VQmUu;EcxpbnJ6PxIIF)<8V&w7avJ~A8tt#e*+z*!n2m-<mg8q8_*~R!<l`Y%#aJ6^lmSuaQjxYY;^*yXCE@=k8c?d&znrVD^BHQ*hIaJVoF+X{masU-%d$7?idE!Jkffy%&`Uk9`D4V#ppED0UI1QzufY%^V(mLA9K$o$?K+ZO}$xPLWO22u&n1Aeo+RL%dT<&(s#c+gC%JBXzou3o(%@F0?ANUBjvSXoZ&D;Yj~=`(*GbRcAMJ7OnNVqP;dY&v?8N;`OU&dC@rKre<BP=Yk5F6lUuCaM&jet_Adv$@7k<6iwYq`!BQ+5MsP+!T=i+#V>q0)vHpZJk3H{12j^L_UNQg-rrZ=?#A4V*err&<MB5B*oDt-;3h>eD0GZloeh6yjr4`x1WnqR*=5}#g5u|qlxFsL@!v6DqvA;unpFYjI;qIZ~EJ3#_D5AvKmHU36K&fu2@El2@p9#q{!r37?^xyRb&RPo16#5DHCyCT!5TPhAKuNq5!jIYp7T4e$CqK{mxP`}H_}84C{oVaXCb>UROVlyB<DQehaMKmw;1P3I`NZK+hEXa}4{Z;J>IwI@2m+E>Z;<^D9l6md0~)L}dYXwL9{W~BNgbIyWSS-%Th{$#Ax2}o+F0c7jEx?~snA%Nt+lBNIO-vw<`AB}cvZt4sUy4iJu8)|NqN;|6LsGy;)pUOV=28yw=Tax2sRMrEFd=5AT%w*z55*#;rBGv(R@haTEfvv_-q9nO}^2=H0>6T%5+Z$Ys!P;F~#nVQ`!_6fX;}g=Ief}eo7B8Y>z&1C!-mh=`MSuTy1OGdYbo4A#1|EX}`rT<ZzTw>&u0OJX|8i&SU7<r<`H1Gist;s!`fHsWiO0l1|_IBZp6Bo+~jIo6J)KCk=cQAbu&kxj;vPN4MR-me^jNieP={Z7?j-MLnTOZ9KQA&D){@IUll7l81;r?T69h6$1Z~iE|vAj;{m=sc#*#yN5l|)Bldh&Z2HrE`ABV-84ST&;mX&)ZO;lGQz!(0ZX2n*4s*lPY9Ob7bi0YJP}+pC*>T>W>)%TY82>s(<DO?5+;MhGo5r7&UsOQiHW{-$U|*k!j9t3wCgce^;MKLSSAUi{o#`xh6Jj^UwAi)LMsne(tl#AQ$)?-=}HsR-z=Ph$H%yN7m{TA>6<b2K{VCKq2k0-Ci>)2BJM*I8e>koP!W3e0bY$irEO~3>CE-{0y^!dzG}l-3$EbF1kPLu?GSOL5g!P(_jcPagUDLH5f1!tJrc1%*uj}OquLl|#8h%Hg~Jx73t(!)ZdL|cK=7TFEoX(g_?AM@Nq@<7m+&hJELyXOjuouy?g}FLyg|()N(`0cQJnw7erH=(6wX7tYzL$V2hkE-LM8kjcQX9k<Y66Db_=|}!*^RUJ&p%u=pMy~ej)2nh+mLTzQfVa7>X2|KTP>0r!6BxZP&n7*jAn<(%fcoU9kqJYB{z|Q4YQl${li5Ofb{baL-KSz@nm1DI>2&Q*Mz6)&#1BOB6$KK;YWWRzOj)?-nChWcZjeo#?}3v{7%^FmXC4hLwHi=Zwdwa);q-t=g^VJ1`WNH4|`lXPtH~O3PU9;n{nxgf~xMiaIC)dlz`#cL@H(+ZMSM8R}6#(Zcj(P`TB0_dd-`-uxJmJIu+WuPSe2<*h31oKS53o?_L~Oqm0*%QtxSCK)FbZmN(+s1Xi^3%kCkdZ1}`Ywl)RbAoA%lvv<~VVs=uBL|;1!1n)iTL%5x{&yw5`!5T9L?S-$l(*lO*DAKNSZ#;zoH-s13QM+&K=p%?Gpx{~Q?xhO?CN({Az`n+Qa5{X_dfwX;(YmsPgI}p^1QCiWjYdx_qRZBQ;PV2=v`Ra!<b(qT8tEeNL*BD=0Oj>j5+x@1<;Air_o6uHP<G_!Jrl=B*aQM8HQ~7gjaBC>-w_K?&+Hjl7FX>KK~V7KYF)U-t^<>LR6KRn(ei6JPy^7l?v326c~%_dh@poonWZ_FUiM7zw26$rf2S1>vFTPHIE^tbi;SSeLQQr0YVrxN0&Q4nA{z~ENeNL!vu^QMLn_u)8_$=xdkjTKGiFXF+(ZWFR2C?oaj`J-XrfDCS2p-NQD&!=%hrsB6v>=#ZN&tZE@jhD8+f7+A?zfVoefuJ|wf_Rs0Cl!J<sxrjkF4djfe2TQ>XchP5*>+l?L*A<AufjcNwo_2MqEo!roL+UJ_UpZHfeNZ{O`Bi}kJ{EH7#I6`8~-pvon@GmtzU4q_{<Gn*J6!a^Y5eYrCzIW!gybg6EO%(%7%J=OClf72z{}}ozi2{g0&44=hxU{(?ssu<^9)_kL4;O`tT>mfmG``6Jywrw_3|qpx2WDig)fEx?A<BpS^Q$r!PSH2}%hEm4iA{T%Lln!MiOX~qmgRN5bdB4#Ygc;_=#~h$KHL)1+|R*F!acwGrt4R<B#UTke?}x;dSm59l*%sJj<Ua;|I<YhY2pKi=ozfp^o%+M_rq;H)GTHuw!gRl4c7>63GJ3;^CsB^-8*~X-Z;1Q<RYIz)1hSeA%p8{js?z+`P_XVS~PB-AOQCZ!R*)bZ>5SPikTi!-u0NBsQLWxLmft?r{e|B2a7j&lRY9+6@_f*R}8<Vj;=hmb9=H%Rwl6+k`FC!>3dR6Py!||m;rvbZZR;8FH1hGw^)8ofkXU-xdEhdzD5RNkAiKl`~)`AFIR6TzP9%vGS;D~w|S_JU*#^-VsyNfKfdk8gr4iGM`Z;JT(I9gc+;f{HGt_cOjLO09Vrtzk67p;GEB6qA7^@MHQud*s0nWEEfmLsXk7BS`reW+rm`O#6}-gG5mC;4B;3tWOCYwyLHq3^fN<2=*Sg=J<fTjfLK>^5+dz0IB%BYy<iC09Ib(2ct8a6l(i_|YKfiR0^jwKk3BFI?%NSlQmjW`G&5U7vbx!mdaVYLjO0G+6?!v1*EwLWP*`?yvtI31c{}?v@=adH7IjMzuI58>FN0l@+qByBH;6}zhB7R3v+ZZLeOCo7$=(=1I%dbO@F%+<wi^so9PjplC6%$9WZElvWnSs3YpIV?M{SCEc3rmHnxz#7v28;04ZJ3V(dFu8xCahDqK{V=rlEW^5i`DB`+ZHcU!-WV~pLPXNUJ&P!+%O(yaJwWKjkb31aofXvyK{q$dn9mky>Wn<=7S6V%zApf%FNN0e6uaEsspQ~iM=)+*Ul7Hw%L3^ypab!y%PWx|9sut{a*tT<^lLl#~6_n|4C_CYp3Dq78ruoP!0z4P^3Srd{!`AU5KUMzMR9x>gVhns9~%Onpb3WIWrFzGTM8HxAt^uj#J;%603eXuyQhtXVrN|@X-{x@%Bjf5)cPn)e(-GQo;BK{A1@&FrmH`IG|?9lRMz*c$ggw0~ga!*xE#+Oep64rOQY+XHx8@ioro5k&9^9H|k_aIc&K;&I-P=@~ggAg=MTv_9$1H@fuXF2F0n&nG_*;PtNtc0A!5$C862yVE3|Hi-i{G!TZ1C6**<WlzCYNEiwPjS|^!Uwa~>ceM9@rAaG<jk$S>=XmuwJ8f00)@J1*{_X8A#Q+;h#H-Dyzeo^>}L!xfEwotdk(ZCQ;QlwdA3%hLRg#L)G-nvn7t@mSljFptw5z+uBDP3t%UJ}M$vqJcuynCijmQ|OvRCFNLQF&#yrpGLCMzETg;x<5je`3QSAm`{jV1{Bkk0LQNQ!xOMM1!MF$TO;l-az<|1lolR%VpH6D<nLCgb9HBQe~gbro_?!PtBq(sBTUr`Hw745p8-lbl_S1PHP-W?7+-d^9xxCT4y-JOza>cTl?|hE(kh(pO$KGBi=-&yGvu*2|!EOwBbL+^T@cP26WSn-u`@65hX<_CIZj#N#ei+J=e@*vgK70GpSK;O$?#H!&Z$+>Bo@VN+&YMlt$OM?gB3^2S>h=$HZWcHk@T*!Odd>;`ws2wa{ndxZyCh^inR@jB+UOm{f1T)KG~%s@zn0(3b*>PV0Epk;qV6Wz%^A4}}PGw@xdOb5Ru-){&{rNGuIi+$K}my?2HjsG2^xOJ71nqQwc{V1q;Rvz(l2S+olmQbY``(8@S(cr?$ec#B-Rzv`;X!C>x4NzWESb8+joJNI{_Ef<0_K$$`q)tC;>Dbs3rS&qz!^i1j?m%hPIbEGmeT>%W_-5}pUBGm`(e0%TQ;zD@}w?82FSI&0>hGq9dz9X{O5||y+Yk`750ri&_xfz%S3qN7z1Slm_pTZ`LmV1qa;p`3E)TGUzB#T64b>0<DUBH5);qPe-cE#9PLv7-QR2q>lpRFNlioj=RUt9J(>|2oMzVcl=MiU+!f<LY)<6%(bT`nK{0_vTy9}gs+spNxg_@e37LW9!fhh=Sk;yInghNo~o6n0!cVg%H%ZQ_pi?D*tVj2UzJ;cjI@m&ymhg;6)HUmIaESeolI0c_fTu$Y#CmK~xolZhOX7)Zp|feu7j_gk9jcPv7DeBbAM2;>RrU7|78TtHr8qU-M$4!I&;lN6}I*cHW%Jt_w_6s@gMXwYOkERl>*$wc<FH=OIuwZ1o{kQ7LF)X|aDQq}bJO-i#=-B|C4eHN+Zwuo-5vQN1@ZKD+W2I{Qxnk={Xwb%yl8p-^4kz^;rznYTnRIP;(nI4n@I@SkUnKL>mIcF(eWHCB;XHLhxU$&WC<zDW)q}z@j^vErxNeJt!4KIsC{?vfX`N-6d=F$-3=&Kh&0Lv6DU>}*{hs935mWjetWezL9%Tel`yVTMxEuE@MT9VDf9N_hmMm{9V{%f+EU$31$1UiQfh(|Opl$$sgWIxhy*_+wpGHCxy7Ok$UoI<05R`q+`w1AtKHvgr|F$S)nYJJP=N(IozglQG?1~>NQY>2?qYRgU|`Y;t3%0^&NYsn+WGh_?a@9v{WztMUN%9|HaW@d^K*b4WZRS{DeM(*OI7%TssJ4SbD#PB+ctvqYDB|U`+zZ+G5`1n8~6g^RbLct4<vO_C9+6w0zXdIS2Z{>q%X=WTXN3ELq!02T2j20BJkamC4K2N_fbl$T@uFtL}Xu?JP5;vIXH=~INnnXiWflOvFdI1P2`EBXI<Q=9UZt4B73#0-_fJcYw^A2YrzS(7_?06{Y;v`(Mgx?r>Au%z#$du%Wf1&L*-U1X*B#W8MHz0^`pdDnue>;LmpZIjmOPQy9>sLJw*irbQ3vJ6ZQo_}PLuyNNISxaM97?hMG#0@{yk!qaqMHZ+x&hdk&;fdaA{QEyLEm|8KKROH9N1Tku=cET>jp>xX5+@&1X8)*yRcCaeWH8SvisWz)sNPgWbJ;WFf8|*-aiLf;K5)<*iCCKJ1nAK#F=EgA(K1#cu1KuKF^wt37!E1`@*0<4-jTUjH+Maa%BIq9~z?T12kx*7b5^<8v_a%FynJEF$In!ao75O+vnv%2jZ5#NB`+Yqb|gq0BU|tz__g6kx!9)zd>KVUC~`uAiq08((3uMfzkF9#$20kp{wGHjt-865uv@aQ#BNh;L#-dRB#&MJVu&%J#~kSN*4F0>06pKUU+DHUTRSgZRiua<0BK8CjnGX2<C)jnb{EHY>LDB0F22j7;RKhA9j4M`~#mX(R#idr&LpQ6~HvPIf7tOWa`TH<-PvK(4quQy)DhO*>|M~)uccRefZTJ0yM^SHj+;i`ApgOF7)PyssPR|h|y1X1j+RPZSdw2Q+qs9k=G~}H54bW0!<Ppi=jJ#D$2gN<(_w~Ou<7!`@+v=Wo3?F&Y+^Z%K_*LJ}$%{xDT-{+Tj*4|IEp2qdC*g_dTtlh&<)x*@X?q<rKD1<~i`}+cn96hJ$<+dk#cIPSou{PAe9Mvm^;9UUv^X#_nVr2K_y%Z%GuW(IJLt+hZ8ZG5_Br*5q;CF`KxCBw;rP<ceMUjqa77NneU<Q(M&BSJ9XqnvQZMuq{TunGu%?R~U0xTX6in`lM#nxH+wa9i5lmp8F?hNV~Y<)oWu(!aeJIw$l+w)n2#vXaGAYkwy|1!%F+70K389ABOQD_)GeS&YIlRwCWKV8bWH=@BIdqk*8F-u;?{_S0bgIMOqrv_?--xg!8vIijhENk|A%(BHtkQg;SP??5u1T;$JL%!?<lXT<CUW1#YzcRgrn#b_b}WdH9A+hb(P}4Vd3pY{Afo)l{syGSHsTP&0@w1TK>n7;ymMh;zhY;Gj}D_AczlZwIJ6b<vv<)XPUBXoe$$B)*0IY`?BY2xb+^A4SWt>3uDqULPlJ$WakNyy#5Ni_h#TYRF>D9GvXUQyQ@9zb28X1M0lyOa~U8AAbJT6nGYW-9x6Oi%<!}IZusyyHObxQPAK1+|(P1kFF9N?O01}_%jB7hcQnT2?i>-9r8>#e>+d)a2?2r2I3TJngB7-4Kif%HzF!!R`GW$X9%dw3*A-^nq&$Gz%2}`C(!p9g0n1M2}1Kv5z=V`{e4>~wir8;QnX|jyL6Xto78p$J7LL=6uIQi(iavA1XU;m4+ZY$^*+bc#6{?U<=6@Jk<@N0>?bJ-Uq7l`Hp4+o`b(Byq5f~UxPNlku1E$0r*6#X8n4~8=<g9hP;1L}sQq<#=>au22?M-ca~Zy+b+xxfJSHWX%I6g2qPYu6V>f<zrq5L<WQj!b^hz}`bMj}r$e*Tp4!3%i{y5ORW<pGDEBcjkzk^}H0Y6LupgeB)7V*bi4~eoQ-A$}-LFRKF#vUg+tJSgC)VxR-NS%!5MATVM^3VPe16~UiI(-mtq4l5$h_(H-BEI`{5I7?@;RdW{+;Sk9U;#70)A}eh?V&(Uyx^HugZM~BKjF*&s$E*ETf5ex%x5t@!Ehn<W=icH6>a=mk_UHd`%&g^VW!lKeeK93^zEWke$K4D^llUVBcY*QK~Tp3RYm-$0O<b!{kq-U24`3Mssj_4(2MTHiCw4Qm?F(JhT#Pg5TlGqVoR0whMoZ7B8;>bfC4}2P6KsruC(f5E(~{-3CzCXEAYm~fM48fr*0Vxmr%C6imC)HsxAPRC!Oe{kSM9Ee;lwJ!?xmHmV{Ml9aqy{e2j6v8H>0ykAp1;P~3UA(wm}4kH={~MTqQ<M}~+Qa~ajR4}AG;yn|7Le7BCZR4VPN<;0g54Xm8G_qd@*(MGwMZcs7qG+6l25S^&B01BqxDrBs~#;Un5lXX+nHk56RA|;X?Hx%i`*bhbx(^zD{-2z6-Yj2um0qm^+kf50(%^3Vkz8ZpEcIDa~5kA@Or`89iG^lWTd+Oi^E#lb{C4PGr3^isYFU)5FR`K#yF!Y~j;|11IaY|`1D5c?O<<F?CuoMS2o;E|j(oR%V9P#i4X0Q(Twrc<kvHTvK*eM?TM8&iZe?veGnXL{fbbl^ZD&ZF;aECJq4xVc?*i#_<!7+wpC4Sipp41#l+lQQxR0r**0}=fN@PEiwP#GB!IQLK^<K^iQu$_lQWnm2gn_x(FqL|labWE)-O5SJRk;y5|5FsD9%Vc#Z_W%liF(Na{qn@yITi`o&eRYM*!A{4tWji)|hpG|ZH~_F7LT!>^ToV4S&jQly(o!JKc*F!iauj}bquuCwTVjkd5hgX{IyP#gWOIVXXl8M$a&!}?eWjIk3-DJmV0_!u#2#D`a|;H2k8OVqkZ(L1^FtwTS(wzbjsb?ngZcGgZRnHHPb>aeS^~aBEZcegdac|PBU<;oA{<K>X)Y%Tdq-~?e#IdyPQ)3V6fkJ^>G+VIf|beP7P7#AJ=i>jOy)qDRyT5jTW#@CipIabQddZg=NRCZPo$Iw?PMHcxmc7mx##{k#-4H}Ixra+S!zdIs-x@%b1-`zf8z<4LGw0T*z?k8ZoZ4E>Q0Z6SzYuYlmjPeEY8qq-;6ZbCp0!s?MRBp=WaT@G=0726Xc7QIq?#+VpwlWCc+XC?IOW!1h6^CN>fVL<fH5Io#Pw|Ld^e5hiI^&?&tqb4c2+Je9bUpon$9UhSIvalMCfV5%^_%ds|r%=WY^f8cAxXT__YKf@zu4n*w7bW^q*MN>X&F4(AL3^}T<bN1^<zDq=i)5`VIxt(!^dw~4~Zeo6qHW(?Yk`!^1U5=sd>j{gM-dF#pB9k%Z31;DNY+uz~wW=aETo&9d;6eeWCXC^<7E0$`v&o<NE5TG>&<NMjHzc^pgx&D-7^NI!C1`zo7iyUnldH;Oz<p>1Yh~=i*ifQCwSs4b<IXXTI-GtX5qO%R>b0Flt`G<wu>+Q#ze1m_jWRyN_zFFYuM(6}Ax|c?5pLR4U32v`$UEwiUtQ7B6ZTv6jlnt0Qn8htxpsL;%sT63|%&U&w4C@@)!x3VEe+8MTcCrhCg$1yyBNKyFW{Z@w;_aO2jM}2*%}4ZhaY*lerO}}2TJLr@3C3)$u1eEKI~*(v51n^b4!jmH-~riZ%|oc-3PWlL_%<zX_*Oj7Qhr&~MEP)5`R`>$!uNWA_XM|lQ7fr<r0c{oyb=z;BTBFN<uw%yP6s0dGZ$jumg3`&N|dy*UXqfqRsAW!eWl8^bm;PQ;Q1BweQe*i!j$}Z+V7uTNGWy#EE=JP#Bj|<m21BULYF0987!O^k>HKv2Q1b39uyHul~;4)qY3bHE_fk0bH9BP5h5U@-{B_|mk|fCwaWF_`jrvv9r4M>$NdXSx8=64Z~r|#-?;DHfSQz@Sjr1Z4eNoje{V7T<r3bY2@Mvh{3jlk={$!&t&Xy>b*TZmtqUF`TGsbK6Yd7TJ?!0lOaAd`ouBTT$7Tz75UV$cMSG<s28I5Zj5b^BaZhi-LvTbX3Iz(+WbpUe>KH0&3MMTBrc-uDsrpPNoG)?BjolSRqt%^Y=n$&*?}xynD;(DZ4uKZQ7B;J1vkhR>Oyph%P3&we`ErH6=<4oSjbYX5p2lkfrwyaaq@yEse=G=}_bQ#T!#t>)qoUTQ$4z9C(Sp!&P7zaHPX{79D{%ntQW|c5*IC9W7S73?<MQ7|PawgGS*t7%fxXPKOQi3NSe}Y9B|AIFIn~i_aEi6a?kX;|X|>_UJnECUihZz)7avp7-4IKKQ^k01Mk*lh;Qw@}5Zp<)x|kDJq3>T`5OFklbh~4T=i_L|kQiRx)p2b;3a?s7`4`A?Me#S(se%wsq`bB!;b65@K*qEFk|iYa*dBep^@3RCf79;<?vij9!78jmHu-<8-8HoGI2jWRlZ%%kN|(K0Db($98aqdp#?!Zg_DXVXdt?+wj7G#9Cy2JN6mb~oxR5TN@0sfH9LWw0*<;U5XS-PHuD&jrD;Y+fnsekH!gVwR#tJQUy`QSSogMAPj;F%q5DYdfl{OP=yxZMG_;zdXcNAeEz?a9NSUE4Wt^+V;(Yg>JL~Vzuz=P`g-BH9g_y_`QAkVtXbJ7GpEVe~y5^D<NNvo!l*Jw#Z6E0-Y`%3?Kp7|-U(-}BJY0EtQ8pvlPJ3UNsqI{1jS&lR8f%P1a7e-9?KbWv%BZ{x8y@rrl1s$Q!sZ`aQ+@H@xn;yt?ct8k$v<-u|$(gt*>$L$(bn~CEnv7nG<ItDm@Wa(9SGHfFbU+$_?HS^c%Hh@94k89?`-G}lSNKe%kFZtvM4~*3@P-8W%^_vUl^Yngg#!G=oZR(C%!?j!$7atvEheM%0b3OX@QZ3p_#B@gR+Y=w@`=Mz$t2On+iP-A&)QYhOXuxiB)%?4TDiOi{_ilA|J~Ke#IFsW@+41aSg}TZWNrQ%3Tj+7KD#d)g{~1loVKHzc9=5c_U|L3qx7)uV01YGp<+)5(&LJrm#*y9EBW3aO$hGxzig8()PL)$26cYodZJsfb8>+?cR=*fFJs=Yx1c>!rKI<GF@7#XB-xwFYa!#X6^aS*oC|)>nBi_NSD6wm-JkD?l!8u>@&@$<C8NVBY=1ZuyHiz0&ZSPi(Agc)k590hid@`cmtSKD6*p)TUgvb){bz^HFG^vX@*NG13~iLa;<#GLBl8DLm3!i57%b#CZC^1!*H5px=yl2j6(Sips@hWxoiC4x#5N~~-vq}}0F5~=;viH*Ly~l7uKwFN&6eZf9^CE>H^Or=R+~3*j|P20)|NtWi=JX;ey&U7FCKDdYY+}d7_-!YDBZEtEqEr(u?q&m=D@UV?p2<^PiD`i_5oGj!>5+7G188FOroH{SN1pcH%9u?pk6CF??++QV54M|psh!yti2b+ZIef^KYRAeG@|)xBO-epmqm3)^v=EUbB@^BI-UREdWX}z=5og_P+J0Vw!jC$gQ)#S^AFxe{ZezfaWcS(W{0bTT<Cv4r{>B*7{W6`r$!`yiv=w34NA<<vZ7b(uugWJR1xNcV!*7!i%KQ7<U}wznelg4esvY&oE$|I`ea&I$xnGE7|0cP_n~_Y+bGD6#*gpZ5n<L2{L3F(S(d~1G}vxjXPvE3X5l`}I-W-ekR;Ula(xr;DEbV+)=MQ|G?IB^01R&Y$zrc|5U_IE*d2)EgT6Oh7-Uw^rKO;!XM~?n3c#=4?$}3ydD6IkbwS%-|KvSvlcJuI#WyIU>kcdB+>A1aQ$f`{QC9qI?rR`U-p9>Mz}d#N-fOaYIG@HyzltQ+;$?FM^_8qoVOZ)V4gkwiiebFXxLLvJ74H~|dU;k(`+{&G_r|23Nw_01$(0{dDJ4vzPTfoE2k)6|m_?{w?Wc|tYh6b{mCdi^YfRSE@|nv311|A8vU8+2)lH?UHf`zQ!psajbi%bg@}N~Htj*};3yGU`J#JN99*yU_tm6G!rHcH9`~fKQ%2jY7gYPtkdsA$WDZ6#5ruS%+jUDYBc@wCAvSV~*yIk@~QCBBi>9M3}16h9<5w55`gLris9xaLk$qHAE^iTIi38BO|#cOVhQP9)8W`tfgaJ!%Vhcx=yprp(l`Z0|O=_0L{&g0-?^c+T9+2|I-59S~hXPPkkzAPeTd)Qjcog+bjL_cm#&mXf14qylICHZCfF%Kx5o)|H|!8Id6guJX&3p&9%<M1S=YW!l;^ntg5*09g2K@(cTgQC;3DgT#r*)ZEktccwXpC)Snq8<L5ZXeYnU+1c7UP%i%Hm6VX4t&Wo*5DiIvb+I(_F2;C)&ez#mWQ5_I}<8aO$Dy^LV%pUhjlL~jwqncPjm?L{`F^qCFR3hzw~m0AT7rIe5FSZ+VtY1L@wKo>&8eZjk~Hf6Fma>jV%)QC#oVnSy;p;a_cV$7hb}h%f&BGsYcxEJ8AfYMzVI%X>1s7AUOc>Pi1b;Z+VK0`}xC&+6DCB>QIL3Fz*VB+cU6$GVU8j1E9n&@?>K7&Zm0MjP$}*wt~G2rbf1pRC;^++~ZjI{`~eOR|`mv>{TS2=t1&Jl$g<t!{Z8C?oTq{1o*%W!kuyuAgaih;WC**=K8zT3ggBl9!z=KX`0S`xF$y{f#2f}yM}1Ulkaa4`}>Qicl9Q~^t?TaQ1XU3;~hGgjRN0=Q5w&-;2HmRN(FPS1-oymK)Txm7Dlo&BfLngmo@oU3Ra{&tQSOZlj~MYgo(`v(?CI#AnkuO3y8~1vNMGZ3c396Tl7vn*@U|E|Gd!8TaWGcxaKC5cEcg9f1bTN0|_)|Z{{NwU`_r;l>FHu16;t2mTNAZV>!>Zlnbz|)U<$rgmc_JuV)u!;wn!3=31GtF*x29+$8>3By-nrPlVppi>TzfaFW*DAXv>d>8-Tcnod4_0CmWo=B8|Ha`JREKS}Vd+t+%_!1@$)CBEB60YqMrpdXoA&*;RSG1CO*=;ld-nXW~QeuRdcn^}>UM&H^A{LuD%1V;^&X9n6zCE9KC?Wt=+KBf4zTmysV543a*)&ZQ$1O-qD14~0%?nRqyfrtFpv$Of(1$IYkzH_QME?RxR<S2Q2pY$4HuE`whQ$L8bDv7;>blO^th353<LOpME)~#uia=sYSoA<CIilrWg20e3TJlvFuaaVh^snr3n>BEhj=x$Qz(CAweLHD@ENGVLi-%w3cA+az89vSbz7^xEDzj&M565N5v{_zDHPuwFc*;m?#N+eW>0@yn?@L>n6T4m}HqIztrWNeKvZe3e269?x=sQVOGFU`014|gt~ajp#}pV?;dh33CpU#ersXF*^dpeA;eCQzsil@Jd*QNqM0_@4=j3}RbzGllvAkg}Np=&RTQQSo?khMy~CPyNnHP39dTcDvHKd`A0<@($0ULD;)_4@RG{8kJ9%9*o?^%E+r-|5}?J06wY8kmakz+6$gPw?H$On%0Nhd>rxg*IqibTv|ol0v)QJF7?i`UxQ0A7TqM?E}AprXI*MHC8j|s`0Gr5y0j|Axd2#RIFgsc7E=A>X3|W33H9R+FG_1LNvJJqq(lw^|G*&zq^~mPwL3nep9=S82Ll2Ik3v8%NH-C)I1_7YJWcesV^b#x??cEBQkE{q)I1(cLyCjC<fCtr<kjS~Je;(n|8DX#_JQK4)B2A$;7Z@9GRNxvz^YV%MaNto?zcX1;<g!!AFoHIMXT4TFFh&%?=Tv|5LDK*XDCK!u|+kln**tgeQ&gwK(IL9IBofU4Ae-O_N6&I#?TX*9Mw8S{>=J^^0_qy%IoBOR$kcpw9P!79}1nMK$yCJ<Pq|&Fr+Ri!y2SLags^96$biLayeiJvMG`6o-;1VnVXHa25KHYZm0@@k$R_f_SD0huZ|lrj*WFc)LiL3$ulLSGkCbx7TZ$Z%g7W>(l#D;2mA$n3tW@Q7=Iecq7IqqaMQ!D4K6i`+1+?gR^;%7mDL)Wjfnj7)U-QPeJWq;TrI5YgYV(<NR=%LEJJ>z@RMAnAM;AKZkakA2`9CMU8=>wnKb>o+iV`=K$4lCJDLDiOK*0Wx9@5$Q{~iUT9W3zQc2pIG>JOTy85`fb2-|p7OF8)4b@#)#dmxOIxM1y|0B!1xRA0}&m!-|vZ0~$rFaU0YnW|vjZUnQ$Y^gWwDSuV8I0p}*K33v^+NbJbmHT3nJp{nf97hFs4iUlnB_AC(_Z5<u5H$tD=?SYjw1CgRPdIN_+`)S@&@mF0UUEj$zO+4d9Y(cXGP5mFQm>Myu<PPDxenbJx+>n`K`{s1NBCEeGzm1VBu^D(7Uk<sAg!BtLq1lnpIoi*rzJvM7}6%e-m2<3+ao?(oImTQSB(&hL_z|UfbDIk+vA^ZpT~1u;V&P`>#|y-KPY;!mOAI_5<x}`!AzsS@5=;V&B{OGjsq$XF|PJ7UVP!JeU&$+es*#y~eH4WD)+J=atf<V>m}?ICRv{=cmJT@(zG%$VIG^vc}atVx^EAz9(PwNx6sXs0$D4`D2S42;gE;wuO8}mFS<ot;)c8Y=k6xeGs@BFO%`BqJ~NbdQT|-`-gzsD^ihPXb>RkDPv=+kSrj4ZeDT5T{CiX!W_uWOH}8UAxlo2w_i7Wa&ODEn$4}!u*K4+0EiX4>8|RAJG~^h)|n)=)vXAQIMTum-j4N6IwPwn=40T>{{{yMDnWeWkiR0)wy4rIelXN*-5Q{g!iww0l6H9o&dd6pRE*h@43m!CgC~@9Kz508%7FheYM}jj*>(XuP@H`A?`I2kFztMcIqnbrWL?E&a(^t<9CuTX&NGYH81oz1oh+uh3G}sW=kj3b^8Ge_F;x5t{lLF3VgYs$>AE*$HQFae?VB;dK?`0rPY)Y^mlPD7a!vcC2b<qjLv%}&=ql~<RQD9Ftdxlwlia4n7n8_|y}2lBBpI}+D*6%$x;z1~7{&$QI6e=FQ$qmuk;H7miWNTi(U0zd*zhH+Z<t;w_$YtB6=aZ5(ZIqtuis51BXIA?vVW>7U$XSKu)7<EleB901w$eq(>p(Ph@np=5Uxe6JgF%4K-PyF82qA?pq^29gke1lXrb|s<!YJg^xs`vgCg#6O2+kEVh4wX{@~S6=#IA1L5$9<EFtDixz}9J6!Stc|Eh#T^FN&Y1ixnxbc43Z{p63Tumig*)(EXq&233?Han=k3X@T{F{y!xES??__Uv*NJOT2^#6=sDWjoP}gXXznq!ENorfPCdx%+<Ikc-xUz~#hW!2P6#O$UI2Jbx<JiMf(dz=Re+AtioDeioj&^n~BWu7N}h{bG&DIpN@Ooh~fPFfVt<{hjqb>tQeFRydZmYN{ihZBC?uC8e#?#KR;`6VnE51qDw;jtVH6g=kCX95azyR51*fF01M|*#qnQ>9~oH+qM;sC$%IeDXZHiupGD54SwWxRue%iTK{<q%A*^1$SeZ~g1litRoA{`+zm=zfuvmC-wQ&(2b&oy94iY1;nB@f#i&*R<C*%cyX}tyNde)^c+a=p8JTn-+ydN!$M$vXbYE<uLe$T?1=a)qwQ&||gbp`6AXdT>c^%tRz7q9$b>P^r|FHvp%q4&Z_|gxpy$jJpS{g<O9c<Hx2nS=a99Nb{^ozY*+w|A!(Ywuqab*^_c~v8@un>x()B1NP25bei54~f@uU(>BWvr^inYzRo|J+%H`Y;41+y+-Gvm13ecCU-%52s=5Gi|^^DkSla{(g??l<$fzU)mA(2QX<SF<jra8Gf^Bq=mTTG%v7@M*@Xa^M@}_4iNDS**Z%VWZcP>UpKn5+dqouat+lMVR29KKzE@LS^;PYlE;k@yFEzJ8grXhg0m#btd@dGn5z2l4McH&x<%L>S!<>~cm^2$QW|c|Y{HdlC!84;;<&UKIowArG3<YN!2Gn!MEHmQ&v^D*46N9p3~40Mf`J{)|NmVo$}ve4X-(iLy<$)STbZvL@VaLhyvn}kk(DLNNo?@Dq+d^0Nk05leZ?bYs7Y{;A4&`mCuel+GFNpw&lkZ|4zzQr;3y2}#ZbGhQt&1IrLC}=T$C1WJXAw^pvN_Y%%5sZmZ9tRCBR{~t5*49P-Z8NF6TNGs%)F?+yk_~P;XazQhm?jKKb=L3G&X*IPLb5yE`0=W6kxy&v=|5{xG`dsubN}sT6joGCDEzfh@_*Xxa1(9xvqyOXt1P=+X-J{R;Cso-Hgy)$B*&w|Il*mY}3_ChnRGkqJad-Wb%#G7S&dM4%b(#pom8`Y&BkV{V6dV!6#-G!;DnTlr>O=hx8QfI_5ka-C%DMxn-A<Yrq|6Qq%GxTPBNFp!i&R%M)*#$5N^?n4#fj4O$WTgszk#a0p<iDBSo&gJo#iM4M+?W0;z(S8^CNlRB4XZ15TpjNLlM6MN!var7MrfV*dIS-c&F3i=;Yv3r>iRdU^IW{{s2J6~tLRN~ls812yxz7cY(vuPflO&9l=O1^SGf~^DP_g|hc=c4L%gfnd63xP*k8x1wg`DeJx*0W`JNLzNoZ@&OsBap+L^7et4j+*PSXlBIvh!H=!F$>%Qj3PL6%R7^=&|>X)?*Pdt2!k$XsUEfSWv(<^_Q_a$luQPf2e5EX`xbitq|Y*;78>Umyq<n2;Xo?KpGt@8b8~-rT1KHxpC$x!~@B*lMDYKP^$0HYdEpB#LBCBtBtPLr~`5iTZ%B6YX5f&rpqmObQm7#jv|yof5SEg-WlM;T@ih_$_c8G<Yz!{S1?(uS3$9GM0na_ED(rHW`*QI%>6bx?+~wndQLcw_JCadQ|6D59&FPvQ&ykcwN91vy`(_!E8#vBMjnby>2QPV@>O}@F9E&1nuQp|{!%-Vh)ZBX>r~H~(yBpC_5!?qt}bXi;IY-uxfNL)Yq)m5y1(<DBf{>Km&30#Z0}l>yzp{K80LdRTS2&9LZptZ)xJqZ?rBAS4aJ-OO@r^hV8$T5k&*PCyL|Gr0Q2t(qX0;MexY2I>1U0R1<5S$gi|y0u&It@ftlpjC!`>9Twjf+BfdmC$;$~oCem>L4yGvZ{p6<HRY?lRw!VF&oBlLDfgAbdJm8EfZ?6YR_-vCdwZT}ykWKGVRXrasL^u?hN<cuBOKsgKd~Ou7en4LpxY%gEhX+zRRKw1veOetOkL_-LVepo<thKI3|5WQP<Mzx_Wc=d7KbzRz(P3Bm#lKlY<}c}E)?!K!Kg&lawy7@_V;jqj*Pv4O2~fki%j4fE#!@p89?c0>!hQaY`v~_i`oZ|^AfZI;p6uIkN#6ZMvrkWKJs@-Quvt!w;)BjXkA@Tikj8n*@e-%{iIID)ZWoTAXrAa%6s`AWA~VbYnI^1#DU}1twKx9iy36G<?f6feunmIUX_x@8e=Iv@>X~C#^m!Icd<~yT5Cp-W{Sjc!B1tiCEZVgaBeYJ+i1ONv+E}u#-<j33TgUievn8Re4Puk6QIY8jAY31`baw8*nWxtw93w2+7s=`W+}S;FL{+{56~u0G<%^Sq4hTq-AInNC%V@;?g{a%JZvvQ~Ib6ypl{Y7oWMv2am91k1UmOm&0;@Xz7^5y7)7~ZuFlqrgM@JN0Q)Ob@+S`n(^dl3JB|r_%(?QoP%4qv7))CB?Bx%&uGzX-a@nAQMU0FO}9)Gqart~jWfWRB$^v-_)eiR@ldk^pE4<N<l^BKA_hQO#6d{-{7+27KG%yg#-tOpOw_UYGVtgpNsuOBkGCd}q({|U%xw;5zE6rv>cf$+ep>>oB<%>x(Xs9(Ll$R(-Gf)%<Bgg?fUzD`?*Z1~}$RK%weo{=w+&hSAmCO#mgFtz9!STVwEerf#)3;?jz<JO_2d(7S*L>giqGAF$)1zUg`)WG46X^z-OF6i>tdwmE^$?7)TU9hNtChqquQ#RG(6!n}~Z0c4iEf>c@DW!$kQin6#Sxn3cXI9DuFc2%~EG4i2;O~>luymWd+T1t;KsI80a<)Yalz{tcdQmTt)>f(Wte}*^BA5$BMX}@2ic!Vdd~Ki`v?|E*KbBL|IqBIE+*azvXaXzvtnS&xmRZO_4Za7Ce8>&XKAU4_=4rq^uvSz$V9E(AeU-K7Q{K~wwU3Z|xE)KeTSu)sSnH&CP)}(Lx{FoblD>GYgW*1V2k<=6x)+?ezQaHDQ*16WXVcfAew6qqAq_S0WmGJ7YXO2A*j6B=1|3aJ9{Yz87q$)*O=5c&|0sTgXp%Jrt*TUKo0wf(;(110#(`zPqdtxhpvnXLl0NCrT2bJJ=w4_9`{XLk2=IBEmt#*VA;RILIk(avzCeGbDh-XxDe$kYSwAxynIk8oTzoCawh~EEem1UwgP2MDj!rA@L-nJq<dUhiN6-g;Ibk{sPswElF;Up7Ey-(HXhX=OcfmFq4egZoV4!RI_trs>edkHU=k?%&kR%?sNGkE3YfR%Ss27OY8pG5)|5K>0mo_;$o(ZsJj(Q3PGg3zpvsaA1xFvzg$fe$3d_`#>*YIh%{7Tn+!KO%G>88JX-yU`Kqq4>u<`FuZ_f5{s$<v#D)pO9;k&sK!gbJ<5WpH%85W;5{_$<j4{MNXBpY$AaxzGRY%tOC}n7&emI`FH$;d(m8s?HK!7BF+b<1}BZ%oU6~%OfbG`uz|2MP9nk;H34_W-Zf$hO^E)_MD*F<qZRss9%}mPVOv!>#4T6@D6~wn3ZxJ)$Mi*PA#HsPSU4_{<EW(Yn)UYB2#=D@-uvx#U>9UDvPM|({7erNP`_7mFFVE2T}Yjz%~MRZq&6ch~bYSVlCe(CdJ%F_|o0B|BR(k&%i1gv3T4Qc0r<OSHT)?<_P!9S6uO@wz33ruj$IqR^Zo>&(jBOB#hX$$G9d+lmryV*e8qoFGax&cLY?6-gRLDAr|v5iuM@g)7r(GCGr8uREx44I}LU6URS!b#C;`}jnPQ96p-3nVWgOekHtHvU`D9y7eqKkKd9DcYD5OJ8CAy7jgdTz2EMS?dvoZ(Z361}2BO02#k`JhLj&}C5!>{eoQWzS=%_HuWC5@|?2CJyb`Jr+P6ic!_(F^OHv{Wv&Kxj@!{VJ*Qt)Ul{5jSaIT>U{h7u!p#8pPjN2aV^#4_Yh$!$v%OEVY%8nA&RbaL1ru23dG{7s@&dDpLEOQLD!*L6UkcoX0Lmi;&ATM&{fRuoYVMDgaWWV~Y;fgZ_w_pcw{b=nUAZ>&!^ZWE_B<O<X=L6G(z)DwEVRpQd(r1}2_Uo|W>cY_GN{tnhk3k&esp*I%N*x^UBd2@V^u!Zwo2#z5vCAo_C;f!Eff_wr(u1{h6$&LFAtbC^ghxPA0u+8Eb;XFKWBsl6~0L?h?(Rl74s*sm}#40wgF^VQg`3JN8%jiLvv3R?(MRxs@%^57H^ZW#_UBgj8*}JBwGS#4V6ufmcZxVZUiXYId(Ia@MdJr4@3nwzABoEi(?0l3vnA1>*$IY%i>oPmb<KLk0?H(nmv?2;Is;y`%dyr?~;h_J!0I9?Qvk4G&b<N}o12V4^rX7T@NrGoFqtWy|IQ;iWWfqE4P9)rDL_4-1p8E`5INE#}NnQVzua1QL5^7)w(_~K<9tdat4u-pj&2$`(%EGoBH&0}R26ZD7%}=#PB1O4u!?~D(T&oELv=;S@qCD#rVKRNq8F0Cx(({?1nWHa&fE(c{uBMFX%nhNDQFPG2_)4-+*_WJ<j9;dXPlmtKOZ;Tk^Pve1(uN+fPlu$;#H+!*Kef%Wa`oPRRrTRp)2MZ2Wg*K49H>By2B52e3B4Af>it1_3NQe9&*WY!&&_x<7@(l2T;W6SBI*tW_I5yhSW*Y0_ueS&xrrzo@)0w=sUl}Sto}IS25&6bznm8`ta}X0O4krA-y79liJR4y!f+M2@oR2v5hJ)V76{Xup_i0|_#R>dG?uP=AOZC<@Mc1c4h6KtpVZ;s^}*3^K=P(H7^H~fEwq(^fvyb3h~6aOE;osQ@)~j^>a&UhM>jfi*wfvC1pW$;!WE>)$V3?Ml^?-SSW|<Dr(-~ch0m1gv`!(5_K8jigd&j+9OsB}p3(s?NHW^LGT=g2NYR3bmSp+T6`%E48yL{BW`9ozxR6Khgaw4g952h5@%0$4OwMcIpUsm2HT}zC**5^5c3@uay{7Uo2y&z|!ssl}3}o1{IO{beCsIbKk0ziN=?2hU7+_{glGP<nN+NqSD!#kR`Q_kw&FcrdrfVPe&2E&C>71B_c`b4FmBH~RsS4j0mdl(~st<I_4h4nK7!hksj%NEU?O)W{*_@Fv?10&>{v$t}Bk(qyO^i<tX4}De+ja?1jtqlTKJFkxmI;?wo^Pc*Ds}J%-Pm*YWfJbc7le(k2B173K-D=N(7jJo4~9Uh#g~GbWgQS0|A_wQvTC4KZMb1Ckdr)LoTGh>b*r<73&<6Z<O}QMEHl=~#XDO&JqFc;o6J5@JilJidb9RA+jZ2EoE12uz)Ct!v9x7O`qm(c*@9W`(53=Hrz0ENCP#!Ln41n|hp+798V%Rt|BzMN29%^zHucjZ8>azPR|AN_zCl4s1@IGSRCD7JwnU$7>Pl-2)1e{`CQdghw>43(X|o0JzH6V4N_gJ^dT<Qe3UwTX<Ll8@9M3Bptk3S;u6&`Xu0o8d4%wNG?!uCh9Vt}h_8OD)wRM?1SPv;}faB~fs3(Ge1?_|mTi10UbjIRa+4QQ#Mm130D5GDxVu;<uk`jH?9BaI49kxOM!aZR`ob>d2VdP@sshs3me@>XC%2Sv|IX2rjZtgLyfEuQO-+1=OlP<{7E|;B;)g#1?%X7QFPyu%bq|dRMzN5UQA!xvTbMyk$)3B>ShwXwEu&}aieV;ZXdlwG#N(rMjANM4JmI!#je<ToQZ2Z13t#oEGK)&m8eRKBmN};<Lt+YG;TBbb^h(=4ha4w#$1eHgx{*sv4H|R2ycNYBE0%?IFwfFFRy+Gj3k1ZTjWXYGpsmU9>-1}}cS$Lp{_x7|HpVAN{5pACl>2*TPj>QGs44U-)RQ?Ed(9u%R1Av_!^t))_PUK>3-wdspx(Qx`&o<qjjY`pMWhks^71vZLay=1Ej%UMwfc1SxVCCBJgnaq_j^il+V-nuGuE$MrMB?k5p04s~GJ4kHHfy!K;q_T8wcc8N2`Qx^u`^Hr2DS_+;;j3-`UqaKD2(cfUfY9U{haHO_9$0oE0WN<c?vgGPAJXf<~ff^fy*5X=BwXvHqIc5r{Wkdz=;g`bv-d2W8&K>Vep}tM`xDbwzUfOMn&vC!jQnNVAcez*7AoSq<RQ_w!UkY-xD-C1AJl>s-B=5RXe0~oqxEhel8M8f**#twb43~F>nk1>V>^?XJgw}b0T(F^c~8qFqv7lh{fBs$Tt6rT$MUg6m@iP*U4I9e;-v+sm0Bh{4B(Vc<eF2zve=M0V{9G1Sbxw%>F+LC_t!?j-XV?XcJrUetYM#73U=lCxE4std?z6NqQ(WIX^LFEe;glaF!+Q!X_l@4>&t^U95W}e29(|^x(rpJfdqMHzc5@x=;1X%6^rtSziki{bF$FG2fm#Mk8|Vv!DUCm#_G~_0!9z+kd(W2p{Chz63@{InA(pNg*{46!d^b@*hMTL_@AMblz#DZtB6aW-9O?eQ4}Vhr(T^D{^1t22Q57yd{bzZK|K<K5%llo#STYE*GoW8(&d!kg$+K#I(NqRg}knnXQG!|0s?e*Z8C1S4;LV`V#l;ebe5edl<zun{7&M{`UqCvbQNPWCgBCgq*(MLwN{2QCt%=2yJTz66BKqoXmmsVc=8CFF$c}b(Q*n%8@*~EJrabxd!yScbS<pniNqKUSTiSGhn4l%C!)yxXwr5vQ?$s=!Pow$9POWsuwE$>=?R4CAX*%V8?~Li80wNk~uwodq7Lx<m9f9AFj?2f<uQ5ulI#z?o5imxl8RLBkmxrZlOJ%W{{vxs$5l@!m%kR7eA><9=L%szFvJXG3~P-{hgFF$Vj1D)V2mxeDXP#<;^Sq>-_^ffeif*l3#u$g6@?__JGF8!e)YO)DKJ&rP0$_{`Qq(iEaBoUkM`H#2>`i7S@!UyZv-JT0yJV4S~1P7%Y~Ic<uw9SC<A~4i=Q)K9H|{T&dyn;4496gp#3|AF<=RcGI}$O1T;P-;(PKxfx1&R{D3L-J3MhdvjyM=`R+p1d>J&?nvYHgoY_mRCQns61}aZcU#evvj>==NrA1aAw_D|;LhQ2WZD<fx}2%0Fq2CbdK^@@!>8|nva>as-pS7BL#h{Geu<HjTzN_{==oF!KDxrLF6B&iQ;W4T`?!bCzcenF4Ph`rljXa536280Y&}R>u<?1>f;F0h7bpzCe|kolk770YfGDykd>*gWe8txhF7j-XiWAimMN#P|p5-y&v?on{=&Rx*H;tos^`ux0^J*?`;}#^(sTE?h6qeK&ak?hs*9dQIXd1_wddJT?0b+FgQc8{Yz#7F#Sej&1w+A!=W+=3TJ~Te;HnZs65$!sj4gt@!uG>Kl42JE&hVNWIy_+$cu6;R^0sL;BcIO!Emj>nx=K~%Nd7}F%)vJvI=_s>ukKdo6Ip1Nap+OFNnd4y2+g7ByTq7Eb7hwFZ$Ui59cmueYl_jdPC+Z|6A@dXqDCwPGvG|s|=VW#iok*25sT+W{*UMCS)G4D7Xb+WJfLq5nX)g51Kp3MLM3Z+KRkJc^gR4_Uu0D0hF>;c9cpU90NJ8<d;0q3QwCy<WEUSIDp&lp0%=Oamvp}PzurSW<NEQ#^JRX!WyXm}Lwy3C0p+s}h`>e7)Uh!i7FhkDO&*C?#XOwRf$V=m~h7FxM0Wu4wewK)Ri7B1%#3l;7eVzGHo<sYH5KsEg>!EFZQkfX4S{xCjqZxN#ROwc52%|v_NR+sEWJ!zC#e&7nynAuW_{gAKen~QPiR82ST2@V&BIhnKNA>5_G~>r_PX`{iN^c0nwix&2DxPE&G}fBYsM4R|!7FfN-%Ay)EbL@;q)s<sm`+6@5DOA^U*p@nDsoL#oxgK1K<v(>TTJ+v(wag_T4lo3@3bkUIxd^FAns6IC@Z4=;3q1>=4Fx)b(PK)w>=ICZ34q(V+F8(J*jHpImUUD`W#-7qPh}a%8C5)nl=z!SIG<?rdBSGJo8bXjbjTWS|uBtabBiCFwPxm<|J3CbCEV}Z*=gHS{$V(Dro`Vs`hL7i|H~rJ^vvIs!#Ly$z`s*TGcQG!a581kLeI#ib=p`d_~?<DNwxJ5MZ}Hr$G{}wp8LvRLd8;14HAaWz=CucRy6p1|*I5#Nu(C()Tcd=u!P6)vFL?#uSO*Q8+<AzW27g_G7p)B22S0DqFQCYwhOAx@zMELM<zb?MP+|U(f%cO$TlBlG=;bv;vCk6#KG1&h>PC7TQ42ATz&r@-@aWGViYP1&6KDm?oZgFG|%AVD|7Ut1=CxStfiO4lR^Y3=f2nTG@t#(b*di`&gG2OHV<WrH$gJal4PPL1^iM8xpx2BCg(bbRW|q?SuWPfO^_3rOR0Ug<}UA^+`z_!AHP&fKr!rsPH*p!dF6XXwZiE6<R%pg`le7mAX&1T+iV>yp=ICYO%T6ta5?%#x-U;v@ug_1PS5#NK}7EW<<{L_pAB-a&f*Z=5N6j&Ow0V?Uyh^Z8xu+IX#@&*q21QmU85DC^O=!$<Fh>VOB{#G!q8w1WZ%2oY5Mgxeu2*%d8s(UCEU?^y~V7y6=Tzy%|K+#>nDgd`UMs(qEBW@#jW*W)Is2UqUD{O0u$4Af)H)a$4x-P>OQkdZ(mX=@m1w+~WTUa(XeA7VvPhoSe5>GS>a(;5)B$5)QnV+^YHlcF)H$9n?8DQl4~b=V`Bap<hgX=6Kl(KPNm8Q~bf#FMXc8M|Sl3vy?>D27FBFkiap8?aHNEIVh++0HRv`2(1YLxug|*X!u`$PCtVCx{ha#ICX=TdEo)r|6LAhGxV|t4~41yx1X4_a3zhoZ?91aTwpzEhNBfTH$X=06<1NMq#}NxC?<CmJHlY$^Gd@U!OTZVN)ufZPh+B~pVkU}o7+Xk)~B(c{N_f{s9MomRH2^;;rxd1pN>oY&FsnL=T*f@2S7(nt#BoW30&y#w3EHKIhW_3&>HLX`<zhu3!qgXN0;)pl~pr=8j!UzJT{ubOuVb>4`jD5>@NLaj>D%GRy4pudD*=up3WEu(apG3`cse6*~ZmnGb;NcufeRqyP}5PgB~HncQ$Ws83&Hz&q6n)kn%td+F)vF2g5q}39|Hk(%4NAsk^x#lBS%B4Xku{9j1Fa?xZ{bj1$(3H$w9m>qC^yp~U5>K!?iz=Z;gUNh0ACc9%i>MQ_^>Ms>wDB%LN%Ht0&~BLJ4Pi)7VwVTq%DtrfKiBYh4C`Hy%AQ})XART6Bu3VsvqpnmY6B>g1!8LZr(bQlIyN8qITZf=PJXlPn!)OK&seeeV&gY-j{8QbvC`Zw){k)&Hox$EV70a%Izfw>-};$>rqtf;Ti2#Y-UmN|*zP4S!%;x|c2{>1!&_ISi>i7C(!7NyD6vdJ6V8Mb0o7H0a?Fb6wP2pq!%eMlT<gPqrz+sda=4AtO@Jxp(1adAqOkEE4uSr7W7QLsFTGZ^M|c=C&NeGD4Z@GHpgE3X%v>Typ<8>CLB&U#CWy(D%RpWmM5zfrsM07sIeB9M+RMl(8wA|;>i-|lch3Tu|Y4E9n*%2!XH$-D@*tOiu!*XKB$7~ZdZ;3Geluio;ED`xS`BzAK(3HTw6Gt*Oc!Vqng5LRG{kj$9zi35m5uY&ps-PRz!k-2s9y?9%&hoEoq*m(6uatsQLIqt%WOb0>i-NQ8KTp)=<Xj>n^*Z9lMT$8VU_j1<gRd|}_r2w(>K%6YHl6f8sf*Lx4#dJIr20Taz41yVQUP#crSP2(Drzj=axzFFkPLvRu5I}lnoi3T`q#IyQpSkT2&JYxW;@x_b)LLI@ML`&c>T~MSV`qo*_39R`e=d>ODvVSEL+$1Vxe$s7S)P24H&r&KT`r|dn2n=r0hH1Z_LOm&<lwK4dCGGUjK*;y;Si@7wo_H&JP6=_QCk(ScqD0MI>4q)KpPV;JZmafcp&fVqp+uy0Y<Kx+q(8b@B|eB908@7ea;owqXSYrv?h8vXCOUBr&Mv8uON*OOQg->_^HM`Tpq#1NAY0$QE4r%f;d&d7B$kd_Ausqr8*>r&zvy=*u1f%jsz(%bchtJ>P;CY+zmF*seW9ngPRH@hwl}Bbn>sHVs|&oDI}}T@C#K=7pDDqSuv!p<>~)le4*7zkgTs@gzuoN!{6Rt3)NHJAj&Ju>YT20m^{%O2(l`yyCT_HaSyDWphikJzeG_@yu?^YI5-YkjnU5~F{kcBl&KPxr(S<jx>x%Uu0YE>?W75hgh>E6_x5QzCW7$JFcZ648nBC<*%MdLQX3gH>w(xDHG&MbXu{4H0DjY09dbl|BK=?d?}813{b6>4hYV_s@Az<$W5yAiw&fwKl0!qfG8%H82gW>b1hTM>Yb0d6<s!7(glEl`{N*`4kkLZDHbSV+kyvN)@)oJze>z^UtA<%)#HD<`L9V(Z2l+E#9z_%o%^7w<TKFN{)eh5y3|zMDyH_!=>6-F3Uf-i(yyNW)%D2Mc$SFnGDnC=tu03U+D%vH6fv%55Bl56WOSsW4<ha#&gOYoq&dLO@eP;SMeEPdb;xsdK<D?7FX~_0{SWVQ%@#OP94$!AHON&(2u9ha?T5X0Vthdd5MuE9h38)Ra)_nq+y62U^W<mypjxU#Q5Dvix<f^b_Xi}kiX$;_y?M?KSIqc+3gqge-pXD^ec`WvGOrCSoxr$a+^9{~lXFPC~mmdilu4Sn9Wl&Z9b!i@8{NF#;EE4mOD<%#t5!3R9Zl{Ej%=uX9s`YpK=!9~c0-#0G;X>6#cWvqFj+M&(Kk4AKCAxgcuF&Mf6(Y8%N{#Kq3Ri?ps*Fjt#yR4iqtt}JyGcC=!+XWOHHK$bzDZy1=hHQ&zc}{?@WSma{y})g<-|nA;0F_U>K_PupdKARyRBR?Pc1_Gnfk-s(@eGc7#69FQa!N4%HDvxACo&V*Knxar;+9Td>hMf{GE4a0BBB=v&7gO)$p{0`ri#8zvueOi?1nOAF$CVpD%Z<8;ZLT5Q9xBhP+u!-Aht;QuM)D@zI@uyTGQ9d&)g~YjPJ6-9CzJ<w!oFBk*TdLxJ&y_n7)wXo0rrfKzh_qu0f<p^=&Scn}hK_K-G+02|^)4C?C_O>vIm1ZAzab;gYt&hN)K2W8m5<mbS(IEvQlnBSmb_wqyT7IKt5d!}VC@tt6G`Z~y}I)$B05?=Z4qZ4v3_YB#U)qzApho9{h{&DiI1Yo`k9~m{mh;^*32p_}%4x|+CPiwj9O53x|19)HjR(9_<9gXKb*T@KcL($1iXBc&JhuqsLl8?KWZLI&S2<Rb({F~^pB<nH)@{9@J1K=}2IJ94cggWlf8<(n?`2FAELndPz3;{y=?J-#9?7B4#qhEwqweMBP?WrKv`?W(B4_yGS$%`b7CY0ziDMn;_pdth+^~KwU&FF7XJy0F@2AkSz^FjQ_!sa;8h=#FEs9R6w$tG*jqYfQ?qZrHw)@DKAJxe2*#&xE{1hv!)nMK+A9NY2K=bwD_PU>{2&%9quXRL5vp%x$jqP^b7A{p$ciw`wY(ZvZ`&5XphnC_LHx{^Qe`r=peBS(IXYCywY)-P@ijwd>!y$1bmSgDJtcuX@`Ksj$G8nP6o5eytK{60XMn3Ld`i|WG);k!FyAjsMnH1f9+wk&X;Q+NX0lQ+5@-|g;`Gp}g<K}mb=QW*GfEZVFTFr&_(XU<vbdXN>9PEnn`_)QEgcm7>QmYSzzPk<dRv4pIShxI&=w4k*ZjHPztxhwRuL?3$t+(N82T6NB5D-EyR<rc!~g6ltDNDNZZI{8Y%iQdj?bAS_y=}*Kv&^46<*=Ua`K}ik|9>6Jbv<GK_D=|4<S<`V#^6z0>p5e3W`3s$i3H2si5*oRm61%rOhWQScIc&OHX?A@pRhn^+Qa5}DPFrVCyf()r%t1xHWKZrWJ%Juw+`IcXU(Q@lovpYKaQ(9m0DPkOhyM3aW3ZR)gfd|xyY&^7QJL4VpieY<abSV>C_uI<0u(350Up4QVRE9gB(8qTD~LNUoET;XlUeo?E&V@|n<DKDcJSvmkV^0pWx6u*x3Wo^HmtFOfIDk1ILesi69BJ0c}d1PbIj3WP2#rk9#o>eCNhPFmAh}QEd`x>ESieh24E5^BaAZo-5LdTg(;$POf8*_%@)S!Sxa+30iC%CBsQgkDnp|P%NIie1Ur6r=RTiAw|^}Oj=WfxhYvVmeIy1>;J`QJBy{KU7L3Cz*%<@R?Z^vsQ;2}V`+MN>jYi(JlOY;(eK+qN_z<PhjI=qyW{9S6WxZw4=0WxydvY(5mPjZ5JYBx33fVtU)Z6IN!hndYI}0)7&2y1uq!Us4FF+uwDY8-FSL~zCK`pjG6=Ctn)8GAw5Muho-D4%%Qo{=`9h3?Db~CMckrm**;VuRH8r6^Xwe{k{TpxG7HVY0e^F5EzQ#ECEHCaPFu)maz1DIk3U1^0~qy_dAmCb#~nLFgPu@oz>$52!)B3mO3`vWa93V9gZU?^~r`@Y%+g;u?Pv<Zw`7X9>15IxmPCYT5^XdwY=hY2tlJ<cIhdX*Hy(~8X40T`gVGtux;#YH`<y+?4?%k&ayHceYSUJ<-AGaOjxU!YKk8Xx;T5(Od04yX=LRL*)|{}s7}L(78xg(2p-M9aXpDNrnXHqDw*)ljfI#Q2h%l~l?r#N`hKV5LU>Q>A>+L}glmv7c?k3!wx>TL!!iS891^bw&91qzVVxVY2kXb;P<KEyJf?E(3F}aT1kNR=I_9Z#`_M62FWFoq>xM8T?=@s!Qmuo%(BA??~6VdPPpA@FP_)Wpi$(Xa4mh)*xTDn+X99DYcMLdQMF#$BDCKq9rO4EpXxY{>uu6#1ssrXvqRPs_i2~>KuW^&D5PLt{nvPxhj!9xaxu?9Bh8`7RhNjB+r=uCHG<0za^*HzmuHtr~Ltu6S?GrbPRYK)Elb}6P5ggRAH^#YCUmQg+Y5P4G1!1ajR5RMzn1%;mH1Jgar(-3^yqGvq^{CT>)Ts+4FG6PFg}$Yz1Mb0FsHDx$|Nyf>;i71fHA}y8uu`N}19Tt~YknVTPsSnOK9CxNS0<ETBDo&gla`A@poM<xmt#6VvNX{cAv?M*S8Lj=w)=Za!v=Hb==zYcfFQ*l6J#iOhBVPj-L!bt2{>n9&x9%*aeCjCCvjNl|qE>ta_fN9f$7%eqB4yGuYxxtH=`k*fv^-y~aoVFvF@b{o-@NW8OIgm|-BqP1>_vOl|!-KTZjIqS$Ws#nP2j<;qk6J^wik`2vNSEVuBjP^agh8ln}hZM$RK3Joq;VoY6+ckB-O|f;!=2$!<3HY_4$(F^<4yjsE*F?lTusaiKv9AVAsXG5@tH%DQH*JC-MAPH#?(;hFJZz$E7Yi-cUEm7A{xB|`4$hcqYiZpAtml_kP(F#UtCmO($BYiQwEz^aLZxx?uMvZIe#f<~;W0p<{&)aELX+gv5t<tco0@&F=U&<~9%&ne=LTf95{%X2b8ZK40I()63B<>DALu(snVeyBZ{#3URPemsRz%@ybmr}+4U^Vfd84Lb?<ejTUaKU4DL(EgZ@HlO3hyIOM{<n0{UEWSxdc4C?a~~J@C2RkKp9NuPnZ%=-PZjDvGI~ILC4OlZ1&ZQEF$vZlj@>^8Bf(lo5xCj@~=N=jpC4{YSQyNxCTA&Fx<;8M=<6z<%+LZ7SA)ZF6Tnt^rYPoY8poRzgppbapeKHvhsg6ywY(I1rPK<`B!h*)IaZu^44{5X}t4?diDH{>G;5d6g2R+uc6laT^3r}acw|B;8Dl#>~&%(q%)RR-I8M4fm1RAG}tWL@j2kCRp_Yr%}<^%`14pp&hZss6PSYKK@Oe>^~le16uT^?rU}g-PhAh3*QhyW9yc!_oJ8>wUzis<sbyR05G<8)mUO2i*ov-qdnM^w-1snjFA=+xb2xOD4fs*F>F6A+W$|HWzua2_N^IQ!b6Zq4RFpv{U5{&l(LeKF+l@4B;cawP;15fcz4vDo=B2|^J)PAl;84QT`EUanC&oK#Q`IE*2uJ2LfPAxA9B>&w%?^_U{QpyZ7s5xnw9?J7IXFyM_1Q~FFYwVt3SU9vawtQOGL^MXtJ&a3Pnl?B3Abrib?@>ZE6>*~c(>Q@o4%UMkMx)w$+5te<}HVo`+kJWsC7`nSVd0G1Q(LqjYXekFhV59VsaTjG*^k}=dY@;c3r*`Zv4?R=a<a9CuYNC=Ais<cdYtae6Du(C&3MmeDNC6n)NgR1(f>qA1Gj`U!vYb`+SRbarjer8HZ=X!9dr$Qm@`7AcQqwTq)*O1Cw9pzI2T9$u@fCUxTjIm1x?ujdT4MDOOg(Lar-C2`R;-9acn6H|iFL9iGkBJIq_;oD0&*q2!8KEJH02TfwQl$}9h(H+D)II;m)$E<MZ@I)eQ?8D@iJTe(1<&#5jdGQ@`?0#><KUu!T#o<u#RL!y{2I}`#qgrGLumZ7^oiFNG4=9t}sjTZ&f%!V7tlcMrk`?a9br>J0<0!6yAZ+|cJl=<$Ub!iCug*v!8ag9|y-c~B7x^+l;8hL*6v{x$X50ltS@-ij?ftkJ>0Tio7R;ibjQb~}al9pXRF(~h8DUuhcOp*T&Yd0AoIRj+6bGyt+3^O!m#MAS8pPI@mMp#W4XFb{SZAvYJK9D^@@6wslt>;0cj9_M$oX4lq<4xn#rV}aY(kC+`A8veoc(F!VF6eIRIn#2DR`jpnwKWl$DTFl7hU9jhxK$ovqD!x2-x(kU&M;yn0)H%iFdgSX72s$_D8@xevLpMekoUHu`?VY)U+BZm`I~Fq`eHv&;j#?op=B05{|zr=`>>!eKnz?boesxABb?|rl+DfMZn~6@i?8&v``eU+YKhq95Tr9be2^TtLJswi|I3qan1@Z3)aFE`VUopQ7wA^@)c0jvdTs2K{x)CEwh}bhcF|JQ^+Rsx4IyjJ7$jQCWzwbfnLFHhz4x`}2Jvy<w@&`_FCNWZak$kAFpt$J@>)g33}4F7HFfSR)dZtAEvs?iqimf%fuf=uJ-FlnCA;+iRf|RL!pLU=0vFj>&=*I&v=5jFR#z7KWVR{64bHA(r85tL(6lXlY<l0I=BBKjM71S>Lq1bRoF`3^SX7#+vNHQ(zF+=<8}LVtvZmNsm+q$15VeXS`TTsj*z&1+RvMaIQ%nFKIjCRF++uBW=?rxAmngG!&3w1?mns2uJzPulV*t`|*ikVJ^j74{DSKmD>JgB`%*8U|7!cNLbhdvEuF#7k2RY@4&QH|vE|M87+sP*$c;e(Xcnp#)t2$h}yLWXJG-K{*yNb-YF;Gh5oDB7*9_%wTqG6LL=NAZ7tV!^_hQ!d&+p;iw2bwHh`g`)O2c7vD1I0Je4yPhWxLW){z_jmvCSUMxJ+$kzHC1J>gV%p`0=ttn-y#74Zi2gX3lyEvtLsW9?WNuj6<VuyUX|U?cqkhmMK6gd9hAXNdM`72qFQSEuivWJVpddnB6$P7PCUl79xf!lFFLpJw#o<)nM2Bq@hWy9#uC||buqXeCx3wY;CA7Ma)u6Ivj%|)#<(3cc+3lr1E3Gf*V;v%cjBq>-^tJD*WCujJSDJSEXz(934AW_vw<xlJ$AoIM3csEYS?W_1Eb-&y$)1$!UNyv+dKvtII-jxFonH?t3&Xvz`Ainw@H#iQ7nb{ttDXpB`EdyEx7?{dr~TMkg_UBHhts*T2L<syInUrr|SZCvIaWaiTZV|)=8G?gRE%h$&n=eRt6LYP*ajfL~~zFQd0Jid7jQD<t!=Lr8}v_O(b2RekeC~PBE}E!1H$(tQ^@@TAK<99k85f7t=4Qn&IN#VG5Z?7t>m7-_4PAqY)Lwb(iZ#ynchyfU3x1j<bffkcbXs1n8!u)g;v2f-hQh`gH=mkb)rV+yzQBoNy;5S;DPZA9Q|_$uB{uo!mcbMonhU3S^*c;jxUttAx(t_>&cQWAtf%FG2M`NIK_y6WKkOo_GlgiNdft`l_cgT>UbOoYD8jLxP~^L%ZUM8!tCPOkI9CAR^VpR69lkyABWO<{a{i-*J^-8yCFj7|3J9@EOB08KwklTiUD53&T)c5dkotmQ}hU0^cf4rSh;q{%b6ZiEF`vq#r64>V45q`O+1Ahl1s#I1`HJHCyC72ij7q4?R!_1lvo-t~Fwcj=LI4;NFc`Ww2NFlc=CcId!R|A9DF;IB}1<*(en?EOPo#CxSrgDa2q6yuScfYp4uZ$-3(c7;=I1<f5#~J8d6GZ95*DQ>D1qm$adyykp6jrf((1;U%-Pr9xn}`iEJC>wy~`bK)wHK!Di37+45XS~YQYU=NlKF!wq&lc-;}!ODbpGGpNSPsU-Ov$<Qz9VO0Oc?dzU6EQ;`ZfnalWISmN6!88qOI|sq74*9nzgkcN+Iev(en8An%$n^kNITWI%v(@=3-??|QIv-*yRo~ZI;y<#Y9|t8)M@U-MDUi(NXo@}p!N(J^Ve26!xPSUU(&d!LS~xnUw~i(o-4y+^~n+E`j6f$V6%Miw;Dm{lMtaJrzjq5_K6c6<rc^mktO?~Jr<%*dc&&92dcr1vk90qP3N}@qCt*NU((tFCiqCDu^H@H5M$&v>rVNGE>@#fO2eRKTKHyI>L>JCKnQgM2YTbO%JJUug_QqIWmQJ?JZ~Xw_1OlgN+|RA51s5oziK-(;;f7IUm;lM-sY!nwr@Tan*(5*P`JVJq>usH9W1nA`z<P$=`vSPZ?2XZSZ$M4nmfS?lSU3NoF*B04>&7XqN{{Ckc$JvYq*T2=#2}CaLF;<Gw^mSF5`l#EwVht5l?M^fu-VfGDqh*1X+jBGChz@nz~hsDB5wF_W3N=0%{GG3x4MGdx4yhopNcZ2eVJ2mEH{&**6`s^SyBAm{FD=eKj^g`=GM#-_1?klV?*Id&2STc0c9Cu;+pu*KwiE4hSN$QF=Wtnjm|c2ah3BQ^;uTN_vuci<y<ezT#~xe}^pfEyy%f5_PSH?mlqWsS8nI-Q5c~&D&EUn1|WXnX^1FnXG(jfW)~JPn&3dcv(n>Jnof|DXQyc<X}{hPA|pWz)nIzwOrpL*GlCYxyZ&&1MP;>YGyMNcXjC$^a%PDoqe{YZY_XG2ET7A6w*>O3R%lT;dC+rM|A^Iez(6j@6B8}&9z)FJCns4S{U%mC=+k|F<ySeZV#*OEqxjLx~h>^MDa5;u^+0#@M;h#Wq`O-{+=oatH%iAv-7D3!+&Z+eWiZ+ghaj68Mki57(m_Mu6e{36veAtF8Or*UPi766H6xo#?<7Dd`YtN?e1eB4{g|*b&c%0t=%jIDb*M6OvxWlOZJm|FENmBqk*Ai0D&c_@BRSd>~`B5(6HKg*G^Lm%|u@RE4%-I|IrJqcG9aC@~e2KGc8`;nF7))b)q<T=Fq!7)%kxt#~xM0%TZ@!Oy0i**w+<Y?GEuWwI0)%NR!sllAY?>d-5KHLqw|$t*EcdT)MRmSI^4s8RgL;C)d8n4*GqT>|$N6rL-<6Wu<E#dGa?Uab+$6kjd^n{_p8ZjIta0MUhFzQe1ZgYQChN<e8|kM3~QV4j<E0n99%(>|7p!nOR(~PYMPJrU!f#IRfN#B<2JJ3~MD=-Zh+R91}Y0RvNdiFbG=JT<&N20Xo|?_9p(%wN~FW8MX*(!8lpxaA+B8z%+`QkGyYM?+z`Xu+%@4D9`YT=&=$#A{!>jGTA*8yt7?rm^doicHC00$F4>K>#l!5nx!Fq<HE6Gs)2lDe(CxrArgD6m04|U4|G7+8`2#_i+DtP*%Yh*Bm~X18c4w6cKQW*IP{S!aFwMDcG^bUN$h=>4~00k{`w;kcJlux6m^<Rbri)PkiIlbF=)1B1^O1}s}$RT|As3)0Kf%!QRKEAAn4q(r^wYVc$m8tGgWQc<rlI!RX?!rqNMB*lr@55p=?+lp;99oJ0}nDyMw0_jBxIPi4YVXOq@I-XCu1OjAhiPHFU~5m>9T4g?CBjbv3a^aktLw)sc`q%__ZncU11uy;k<7ef!9e;X>dzKW#3+->8zDSQ#r74$PPdAY{z`cOpa@F+y@u9?>W!f7M*Fcq7n9RE@*Z1J#w8><njwl1bZ~0UtNR^RsCh?G$CIU-dfHr@53gZ(n7picCx#-(v9XgRoTBdV(w+Aekdx6+X9V5q0UC2F<{GtWDQ3s&G`=Uf=?YIGuvfV&}UHxQ6}4#-&x-x&!NkR!xCs#|)Y$LG2r>&(bFPP+s2|NrDH)nU}jlQ|5<CunFJV{eB4wu~Clz_TfH;8d_839QlxwL;3;z2o=dEyAWTXGrP}3XNCM;<kDAxpp7M8FE7l?{Fdm=6cC>!CSEfGyN(Lnpo5}G;O1$|SY;}YlSsTg6+=bC$=qdEHuX*jQx}eV`oEcRVu(|k(&;f!UoJtU&YN}%$A4d~^f1_RnZp~;j}NL(%s4FK{GwL-y(JPYLrNIOByxP&bcGW_QWi<JQwkTnX3lX)Yfw|be=|*T5nl-WYlxS1%HYj4x}d@(ro}V;exi1iXToD@4+&_#x8n;L-5e&J2&#R0S@3}#K13p7LtbOHL3`$zU*adUTz$VFtNDS+U;>P9Hoqp^=S+r;Ic<{?%q$_0`H6Yaim)831kcF72v~P<)c~6?a;>HAKqwX6{13vuk+PSjC^S1k<iSQl;m84Mdq@gM;K4%q_L?~Zz?er;Uf8=TWoW$QuCYjehj-Qj0EHv65AXkr|8n0gsk`@NMiZITe20v3Xui3Dy3xsEdecRu*A5vAa(z<`{W@Y~hC1mAg-6$+QPd_4;3d0Isw8s1A(1?6)dX01$ij}|9r~!Z*bT^mszqAVi|!-`;D5VCtP`l>88r77AC=w?Bc(m7aPbob-N@Ce6-b^etaueX1Z0+uhg<X{Skwj>N_t6H8W%dZW&ck9@$8miGV8-<w#D-@Q}4w5uO(>gZa*PD^#lB_w)Ru|&wn*PIvzl6MW#|F$sA}kbFg!eVphD6M~?w#Qath|-Ru#CJ@g7TOb%S&&W%x&InvIMaQ~dcQ+swP$OBi(j>WXFv&AZv9OMXe4a}H2))ETb*6?mH!*HH9gvUOYV^t!MbZgLytdEV-JS#<D3iRD6O?<a#_H3c<Tp()_srrD<_$Pb_-&0vfVc=<DfQ5f`5`-}x{bq42WQ?00v`8Cgom8_(tRd&|+%YA!{d$dP83lG_m4go(z$coUJ5|5k+IOV;Z1cDla@hF^89s25PVk6lFZyFGs_Egj7~<a^^ppF8%P3mc&?(_KuD+djn9DNR4+h7_lt<lXhZwLlHJxV;n2IeNtj{5gwt;>OG?mqs@@WfyHj?%SLG;gcXiKc3kvsx<J$lso+Q6W(<N{S&CT4L^<fo=U2!J4e8`lz29|N9&+jpJz7eF-36<Ho*)WqMi&%g<Zs^9E$r1B<L<mVjq%XQCaA_T$xrN635wm^zvfs536hdHx|4SZro7rW`aO&u9@{MICQPpIDHclD?OkkmAf#MVU3hhS)y$fx6Q=@1Bz5lIiFa{`9xjTkH$0P)17?{EFOCn?`^ZB297GLlW}Iqpq|XKL@yyG9p3WOnc=0TqF*sA2@Aj-sjr%f+ujX)(T!v4%D1AO{>rL!3t>5x7coopy~c23f3tF<lvaRYBw#+(%(^Nnm*n=TIq>S2Jj#Q{*DinTrRn;prrxi&-qO(SdJE-A@*uG)E-oHOm3P*o?VKriQ(NrurkLnrHy3Fpc9$5<<tsDq-x{i*^*!B?aoM{U;H7p7OV1hv<Up7icH*#|n|xyjz70=NH<UQyIJ8WSFPT^21&_>n@yH<jR5GOJzaa8heQdHkN{3>E55KCnPB>uhwi~C3!QAKKt8l?EW|Wsc#BsN#KG?VDXcnWe&9DhD+kQ<FjBMWd4D$u<nYAXNZhc(N;*FmWeGjxOcnOmtuP`k}{l>&xYY(jjAxR(uiEwioi-zaOeHd0OXF+X#pE4@HrMnlX;z(Z9VG7S^oS#8wwrv6@%U*DLFI-_QJ+mycXXojMcXOWT<uUd#FWh=86{pP?DgsE_|KNT3)^5h`gN(MOoCOtQ(1@4DG;`iC-Xf-&@l92ndvS->H-S?1tdt?kN=B9OC;b%L1&)3L&8PNn<2(&eX!3^@>jdAo#4$;YFnsBLb7JI7W6fKof0&1>$Y_D`z@BzDsaFy$fszoOZ9^P?PYe?TT=(I1I338<HFDnRPMGMNIVf$PEb{BL~0BSs4M?2g*TGdTFPuGHUhcgdcj*kcj79+eU_Ek~D*kRRKxN)tU<@uwr>T$|q6TRDEN8p(a{h!816K^>BZH1N8-os|@VbNlF`XSZBaEK=xmi8p9nV>1kL^5!uW5RJk7nI&;Cqv#nK3U}y-?uz?!&d|_6r$Qd(-hLO2<_oXrWn!MoSG>F|YXD`wZviYTvxL$Ps&6&y-D5eQRAfpbRULe^aY;^a&Qz>Q3q_>4Z!s4UW+^=b|?WCC&xjDi`s;~PL=V@~&YRe`W6)Mw#%d}jL8?N&cF|zHll3w}E?OhWOE$;Av=ocI<d-AN@_PQAVbf3k_UNQSUR+I51rh})S5b$@0{k36BJp+yeSmFH2BE@MKm|-sYU9`3pg>*0DCvdjr?{dHDPGi|#hivv;4ZPvuGGTVi0x0X1gtE?RL@K5`;%8AncpNrr_;_(#zzI!B@(R$v#w<gpAhzHb0NGE!COF4y8cT`WN8y&^S<3jO%*E*?HWw5l7fXnN>l|Ynj{)h|hAI)0K|=TXU{gp}t(Nd95pVczUgasX_^J7^ZKE_oP<opJJ2{0BEP>w4l(omZ4mZAxc=%Mg14VIU@NLzhopui!W!%;-cIZ9&z-dv1Fm6T*V4FVtIk&~^77$Lfrpylg>UT2iB56usB@<0?+wBUVE52Mb9gVzx`<g_#k^Np>&u-EZdUd=fj%D6=uA}af+WvM_%6>GVHBWP}+Fl2HwPU`TSwX>KYL^mDor&^{p-0=XN2YZ43d&O{?XpqWFQJasMe={^r2iwHrskjA@!Ctvni~Z7&F#i+hkY6{FdD2(fSxc^cb|$Nl`lDc2qD-}wM%t?d_E>q)oPr-qTE*(bQb|g(+eh}zPZ_{(h+{6P9P`53;%M9&#@&XVRVjOo!KLmhhaZPSnRcGT$~mi+f!AKK|gP%pLE8Zj3?_&x)rZR(~mW819Snn0f4(C75h|H*yu1;hns=pK=%BIYBffe1+t{EFmCTZN#1C7v^0J>0c7<>q?pi4sBx+5=E&`_n9O~g_oDhZ-a0thOu;5s`D3bx+*uz_a^MNwvt@w}bUf=r;A4af_M|o4JReOc`UWRJeS~rvR()C)ER&KTvT$%STr?Me(J?w|Oh1L{o9hi}(VrbAfXH>xRGWZDDWn;#u#Zhr8e+!r0$~29!f6Zh(eA=j+IX@2S6zdY?zKX=7FCgZI5HE;A#4OWn5xpc*!n(hrpS`C3o0*1`8?LxiL-NvrI*}^F$0g=HplzyDyk)E1wO`6;)lULPA*$kwV3Vzxn*-|JU}JMo3yH&d(~Zq))@zP>=X<1u<mSoYnH<BK&n-AN3rK~ldmhThyy~BX9j@#hmob1^T~GbiDKDl8}izEoUAFr^tV?P&PeA%&^!E%rVU_%LV%XW1t+Jo!wJ62o_YpP_qdb7T(@zdR<bEG$xe*Psmlnbta^JN^#iD&?20?=+ASC5BwSzhTZw+gRo+kqUTmFNJ3uLrhZnKM17~VK+%V?W`-aAvC=^~v!LJ8`@odRG<k9$9){cMiOM$qFS!^r86Z2y8p1!T|@d^ou2RQng3mUS(I9{34dfTWl_nYE254lO=<IehyHE7B}iZ=-t7A;mdyb_+du0-ojcDg!De~%mxlQn&2ZG-I?S$lZdC5bf4gRDx7RWOEK&g##fu<Q-x;u0E){<e=PAe#C)`_Us*C^lPqB+;z#u)QD!W^z3(mjwa!^VZ+ehO(*NS?F8>HjTZYoSElobnKpb7Jpz&lWoavI3J}aRT=o?l_sWswPIG89b{7qM0RYY{RK6PZfDE&b8%~aMcJH+tVe!OWSJR!;T>*knrgNCnUb(0t~d(fRnem<&NUpPF>KNCPw1im%6IBl-4{3&O7=NIp;#mM>kqewsX{fNzCkwZMg!wZEr<p3_S?pE1C3Z*jsvfuD1*}FufAhst(7h~D;}dg|N9%heq8JL?XiwBSU~tg;}d(Zy$x*Cev_+&XY|a*bZ2B18JH{ZYgr3>5$jhK8fY%)zLg5g93T3oBe3i46mQ7JD3Wo>E^8iO_Bj)eT~bH^s}r#+C4YXrXJ^%3*h2}$tlU78cP7QR4JmOj<{F#=#)11q$`j9DcD1loE77|$ayCkRYH#er9_BOy`wSr|ef@u&2-RX=PXg40VDkJIQVZVVG0oym^ZAEw(KFUb9%JWK*fJ5RaRKT_O-@%<a$kJMbWy4<Mvz@%CxoZk^qX;RuavL4kgk_AxTg%9(3t_HQ$f_Rvl%Zry*`q9iZWWsfsyD!i}8P+Aq0Q5ixNDM-$xD*KH<O9Ispd3B?W2Su1MLRR<u0zGkV94v3j`Y)CW%&H3o2aHLxr?1TW8BscVrqeRb=>IIg8&$hz35Y(*UER#=|L7Dh*W7<Ba9+s4mbIGi)YUVU^xZ9!%vEuq{wo+0mxhv`0l=*t{95~szXo4I*HUyM2dq&}%cC(r@|kv7!?fck4vJ;Oi?WfKWP(~K7siM?d2!hw}X#GJps_q8VRC?B>m%YoHf$k~z>`ZCl-QtX>ztntO4n)Kt|&N*JWR-@x(SOjfCk~!C<2I;V=HH<DnmQs^!d*~KLjFFE*d~}X^+^iU6XB1kJMP(VFScqJ<U`BONT+F@(m}cYf%@~Ow(#0l5WQ)&6(S4^%4)aeS#>elE5TM|_rsQ6MZk$2twvs@27l|l4wV<S?96MRp`x>BNE@KXsLs>YDXZqG3l&H0TFz(Vn;&mh*_dI{O#6%j!+E{o%6aYaSC9%1lkLqtVF?|=%IUIu{Q<@-F;OiaZh%8;TgU~fl-qSa>>IDP_i@SI}mnr%hj|skxB;#}E+urCqJX%6&XvhkOZV{NuYCr0XT)-LG$MxKkpURfY#Yz4FK4OtRC$uJwlHAY2B%4IblLtSG(z?ttj#H{+a6B?7&G@w|6)vs0bayh8Ys3U?5|23PNC`CSzx!#;35w*IE!*|10f`#V3ExAF6igCZx$4jt9EF&ys88o<q0L6TGd8B+ts_$2>Jq=r9<f}GR1SQHcsbU&xe#$Hf?4$0Y#t(sa2+qDW>>ou$DRjmx^mq@!hlZt_i5DORNRGX^{;vK>Ou;k(-_s%;d*1^k!<1I@ky?Wcp<A)Jkzlp0Zs{2-_CLTr;2dT`gMT;i1|O12H?8;*Zm);f|>Aa8)gLCvUVE`P8C%m8)+uKk#Q^?aRL`hvaQ@Z$EvzEn<|B@(Kff@COriLI+_OFhVrj4qS*MR1#uM9`b>1tm^pIH!IHbTu44X+=r~G`{W9R#W(i1l1w7{|8tsGQZSIFizBni3F8S=D0}XcYTH&KFAOke-cn0WXIq-|OioJcG&B7-MLP<Aod?*mI%95M|s7~rL*H=)|NPyV-J&$Jnx?w|Xs%&IyceNO_?rGsNuO0+}IitQQ(RUR1G)YJt=NwPs`IEpX#I?<>J_{u^w@~EhU02#G?^)7C5l(m0<afHwPoP<a2ms(iIMV3cZug@&$(r3q!Y%st;C?^g4*%vM-@hpDZ@gmhemWJmISGO~K-Z@uf!VU`x^OD_KeN=#;0K6Px+sNKngqW3f5q<`le8a&*eje}9_kS~`*;D=5HW)XpVuM^fMR*{@^OW0vXVy|$p-;dvNEl(b#%2Xuo?%oK+C{jG>3R=`$`td8le)Iqt8DN(cv!Mu2kdz8w?T`AKHfxspDpqp42iM(BdI7D9_nII(NtII6uLpR>4{|)f=|kwqkE6r<7YkNWz2|OuI?av%D58qt$$CNyu85p^(V><FV`!k(8%4$RX`E6LDR@wZev6Ymi#d3be6b{}^T0^q;$qR>G-*cx-Zz+KED__sTREh_jdEydpP*tpA@-9Cz?7u0E`ss+7DDJib_Ja}-RTDq#F-$@0%G+`Ya+KS{#c+?7>_75&!smsS<ZMJnSO1$w)a@IIU%cijokz_|VZ#y|oMC);oD+g%S55ZI_av%D(EAs3Pkngkc!U*h%^DUWvt$P0~m$U}HWt35fQuwXd4qbf7^_(gZ5{{`57#McAyoU~>V*y5*2g3{nfuW^!jZM$}A!#l!noAr7B(2<SEQ)+9j?se4#2tPS3sO41+idpczVqo^<ll<zpsY+aSMZ5ew?r7ySJvoROnPMFl9>1@BefxJ=nlDyoCckU%pcZ;*cT<lZ0se4zf#eQd#S&S+L_<6pheH!<v4HJ(jU<+U?E+_Ipmy=olngT(2{doPA3^29hrC=N$GIUNvWQct-k4A7y)Q2B#kN*}%`uFWg}J)}rsJ{}EJ+x#1p!Ls34CS7%eW`J_=~Sg)jB`H`G)k<(*y<7tPC*gDl|OI`i<0)BH^~h{o_ipZX*c#TNy}q)+R2n1}LRtPgPJ63NDZsM+i)7;hGv~$^u<L)iYDVdsnTCxE7OSlwbg2H%#jH1-nhDAAu+KCwta9ys1Ph7fR^J)h;sW4N1X=6+9$1zdkx%a{Oos&c1n9T>yy|hJPw<N<<3qw5;^Sm9*e}vTA)j^;`KW)GI?_*)j=72Tu|RtUyq?ldpQZ85cZDFlb3dnaa?lq$>y+U;sW&3OD-H#IPW2txe5(EI<j}8iOXZm!O`F!Atu*W$y+t5Q-d5?9;V>OzgB4ZKyExU;gJQJ1-6W-d5LV(~=o+v1!aJ{hCb(BUP;c2M-8QRm8XqJ8dv&fTJh`Q5#`NE!%w`Do!%bJoDV|@sZJHw(A%>fJ%ct<%eX9K2DG~NMO0TC_Zz&9Tv{B#?fR`s%b3*%f@CFF#386m}>I1#sy;zTjb?2cWi~OrHc^B1G)711Vg3wCZ|BU2@=D?x#v%8fLZv<bBxQljr<R(HrK$Jgp03nvGN=pREVEP1VV?2;X$-axjng?dBg57;zWS39`U*c=Vk86z!`z`@NlM}Nq6z*B63v}n7iUZ^{;^l%nGyqqLu~thyFV1&`vaA8CqB_)N|6tbGEv-@F(fB)KCdCLW)?P{E8LnBw#!mXCon8Rj%IXnVBiz8dQ4qxmF7KYrM(j#|+MgkO)xnV~890@O|MYq$XgsET%SUyA2;59i@TKd<XzY>0<GJ>l=WB+BTJazxUz;HwXsxkQqAcwQsRtq22v{1$_8>igK{?^)A}~x0o6;4-uX{<pgg>XN3ux*gU5@m%us7bg7giLxj0Vo$ooSpdWsd0$L&iCdq#?jI7>Zb$?{amnEM|^SnvA8XHalXudTsy@@0B$qFuI9CN!~<O3xV0xvt$gXg4a;^8ZNO*q>#y%)X7hm7hsgP9`>{G8fk_RSt*+BnEN!JUKg@@Fp_@&!ewx1#duKuW9X$wpysi2{0-UtDu*+lHn5;Y`-O2nA;@js%c;zc~Y_u)WC>@Jy7%cokF_AC!#BO9+(y#CL6F+qB5NREm`XJ#loeO0I=#hhf+I`cBCpKE*dxt!)9ONaeYunoB5mT)R-F!Y_eK#x?!>^|||P8C+>uKh9V(Ksf8cL>J<3Z=|W@=Eibf0kHpB|B9siDdykqk9G=2m6z^%N&2==yKn8u5NsFe@uV)s%^hOV$HbVAM-W__zfr?R)gR}}ifNHOh=V(;qEco*YteuhDBCeR`VNVI3X4LLV4~hoDRne-9D4`qNF{f9?v0w6;xm7t9AF5#2in=4=i8<#uqCY?>5Fy^vKM_<{P^d05RIF=?_I5PM1Qpyv%IggUoW-beqxNxgZ@}b1_#uN{*O?VV+~pf5L-4M^*5~&g!X=1>~YQR&NpFN%a1&oD#9bp3G-u+EHQr~^86=2$y;vM#Cwkv4AqG=PUumno;S<obc5SE>g+Xhhue5KT=Wdmaqoyut%yefsxfLB&&B}<oCvzW52j?a1H)Qnz9~qRw%D%<KeCq!9-l54E*5y};?pnvI*xMri<{3*aXxBUvh|Jv8&NS1{!Z#2oB`M;-+kqtk6P!ZW>;t~vCIf%#b6EOmrAK*vE{U4tan(Xg}evXo3;;Cy7BS}i3#SZ4~a!+*8aqEbq(aQk<^*GUAqLu8A-N}1WeD3>re!nK6Y^%|9Cr)8t~1<+dsBK`NH{z+_3L1t9vRWH{*cZaI2Bp0neft-k$Z6@&Ju9S|S>px^eVCK7%e5M*HJ@zHzXM<O#hY3@Ef)IN6^+KZdcj&f}nH<U#0si&lB?8NWC2uKDnwIeMJ=iY3wG*u9uG1g#aRPUQyynnatknj~E>H%?>%SG0w;V<|NoOI~DP@CzGrHosSu{+R)|KzppLaY{WL%!es%$2}u95WES&wv;o|DM@iIM8AdXx76uF9M7Z?b!m2e8W24nFvJ5v2)lRxiv3Dkj`vE)R({p+KpW70`JB;NVS)%5Zmg5y^{F#JOVFl&5K`oEfa8?k(pn~@omTNQ^-l?hOGYT2S{&!&Bq-^?M?9y-Xq#FP<HCCa(Ei9t;UbDaTDoCL9)Qj|Df)wst9>9oV;ZUVrZ?}n$fvT=K$iBYpt9o~|LA24s?0wG9zFCwK1KW(;!X?2C^@h@<e_ZF`~ZjaOnoJUL%l*ADFB5$dK0eQ$mi<=f3UkFY6Wa6OfN@pvZ0tMqtdN317?@iLfko}ixnAaIPvCzJh&LeG3B7+58mZ8eNcxr`Fz0(ttyT@X6(2d^|X`=n>-~!TYXpZk+d`6nC9z_yse`gSZ?|^%S_HO!E|qN98H=7{2(T6?RECOa;TnH{)X{a${vtSXVjE3$Kf(Q{F0%f>+aRn+bXsLHz%eE;JygdSHy%1V*%VgM6#GmYzOOwFiBaBq_I0LKD!-98`W`qsiel7qo^QCL7Q;@FUCUF<L70sMg-&I?Xn${-%dFg^*J#fD`~UK-Fq#kgZm(3T_O{p#H?<?IeS?CD-1`AlsCKaT~X?AZOl0?Wud1sHcXOfH6yG_e!a5@dI78|c9%_n`1JCRv0K|Bfkz(8ivcPEyzM}B<UilnEVZ4OIMfh}Qp_V8N}x0kn^<1aU#fY%4A1zSF)nSi(-rZdr>|&CQ4Ms3W*+k0c5u3^y;$4*9+SClc~ZqF9$W)(c<M40Ac4nTUItl?^*l)UF9edfz&ONYY0)EpCpMnmbYlN)&lv(;Fy6U<gnyoUZGuLa<h(i~Yd}uIC@U%Xo*r!;_s*6!#MAn|k!8i%rNuq>n``d7^TOri_K9rm&-K?>q?oFhodHWABtY}jl2gHyKp$Py^4DSvSh-@tmT(@(;AnYjYSUXJ04xlDfx`SF&gLWADs^a07@6Dc2t2S$Vw8`58Lfjfj?BolvSd$MPMkMivgd8NiM|j%8{QSX-&8BG*BcirtmoqzfKeZYJn_#W;Hf?<1rX{VES)KW^JvjR<}ODaRyLiMI7X{XFwZ|=z7TA9_JI4$pRM*zaDs5^h!LTyijB@AR_hkQ#fXY;SxPd6yzAHShh3vTYuBr2r&#@F!(OjyDYsFi`(bZzEzO<#<{Y4KXJ6dxtL3<s@D;LR$6L%>?mig`SAwMr^O{6+7vc|f4%sTZ=ilv-8e%S{12#5kLe`F~zfdStmAyr^4N*;y5Jzm6=Tt=SB-Og<K%i;-$)ECR;~9@fQKqB>v*JoQFSEE|rQS+$T}j7suXdRxB7jGZPKI}SZz1v8-$z@5DI`rxv6q~P^^(kqnXb?widNcWo_|K0lb1Z48A913zg)D7wX1e5wLl4$%C_+xaQpT|=YnHcO?*J7gLh$z&S?YCc-(SvS+If*U}~akpb6OHz(Zf1GKU>o97j10XVgYpqIa4+Rr<oegsFb34sHeumqL5)`y{y0BZ77^`Uartn>{_fN5Swo6qvWopWK3E0T`rqJHD&OVDfC?Qa!zqJX9UdcyDvjyq1~{$-bSG@llT)Di~+`j}lbdh9MwqM5kUg)Ayo}-|e~+>~$P7SZFF5SgaXW4!n`V2?db6-BBqJjC~>7pt>GSs1}<AEB6Bu<mSq9U}l8qpi#nYYA~9=GzY5)v80(%A*Ao^!J_*<Y(^Q>7L>0R+tl7E%Y39X3%JPJ`;?My36le5Ze)yqf+V$!pBX6k!atibDbDU>xJBa?JSU*v=w@SCdTDJ?Z>P9!7!$z1laa5aZ1di_A%x8R)Kpp*zI#&pnSUn{gaItWkb)}2T-|>N(iwLypA5o;(MxEmFp$(_Yjr*)R-;;saVrr9G+wv$=tArO7}5M;y)#}yrnJdYFp|>&_w#Z3StagVEf9_r&iE+IcNQmQ*b&0MVDk=^E?%oN_})(Mn1R9;71U%cwkxT1weplvkBLA=<rt3*t+n-{-|bPg8;o66I`mQPp#@5TFIw$jcgWtvA~PknMJ1?$_2$eY=VpZ)hmhf9!=0D8q1j?C=B)-ZTkDssC;3XM#!6x*sHQ>`8HGcUj>C3ym^Q~h1FhA?%+0G6GXEhd;0K-xa6=;o1CHTvO1yOdnLGG6dx_HXe0<uClw?tbkvia`ayBtEg@f2f@=tXQ+STdmed={_a<Bs*`|f1_-K;7NhTmHcX8?JxX7qnHUG`M>H$9nPf$v!j3zMV$Z=~l*nKR<JD_{uN&Lv(_0dz>jK1$^duJ85$9~a;ANLQSa!|I@1oIwh(Q1wxXj}$2d&qL?;9}fYUXI7blTYg3=lM5kll)+=v#A2T?ub@#P-|HY(2T*pZ>A9l|wu&(PwUA&A2m4_%gj>X$N5#)yb=eCY$=49izKqDMhTuDc9r;Y8HaO23&!gxahw1t+=qOViu=;RWEl0MBMD@a|aw0o|4}a`P1bLMaCg2FkuI=CTsgLA8_l<5vB?$oH_?&lQN>*q_Y41OeUsvR^l%_DI@htT$@cG1DD~5D1X;Ta??Nj;259XlOs__8p2zI;WM%4Vf3f8Luc4!%aSB!wkU5zA(@F1gkg0a7c5Dv&mh{3;D;^R8nYRtV%^%}}BNU3K3rw=+wx0HOb;d~QTifAd)SYuoQnkpND5^CVG`98$Kua`rhyL$<D&utN0iZDtjmPsd`>(ixekNx%=#8SQhpuqlQ5EwowkDm^pTXAX=gN`t}hg&cw`aD$>7gmM@PoDS3ps)R=a<J+0lu~Ac%W~f-dF$7l_Mq{)5HTi-7FWhF>1Kz7;&*QlB*W7xO15{#n&8n<CWpB?s6*#V8c?fJGOK+&NCoa3W)nUmz)Qzg`TvvDB7=SSa@3^PE~KE|2j)zdxZ&becyEbSPgu1p#_~2C*q&%|UNFsi!F?IZG+dsaS3#BwFZEo(bbrP8z*~=`iA^CaV|ZNsp4<x{jz$AO=W(PR1A&=NrdZ8e4JUAE@#rEM#lXse{$1}<3lC0y<&4*RxWd{&!^=<Qf4iptjL#cMV%=$z>U<j(+$J%6(qB^De*qj_lo4vU-s03rK{`crDR)K>%f1>FLg6b8E`LS0RS%$>lGZV8I~qRQhI7*Q1rMhyJejXlj{ELVOsKbyv|&;qM16}0MTz*SW=U&hzRp&l_|o$Sh6c%MP=JXIib&YYDUHFmZt36A;3>9cbt1`m?Rb|9bHVnMN>g2Zrr@k0J`fu!OGoOsshApuuG(vSI|NSurVCT!*W*ss7I+xj+)GKAPxckCl~zw6dPi(k&&dol-N+6<a7mT>V(IdLya?c!o?-&Mc3mH|<g|9HD=DmJru!U5@ty_c{HssEBsz<p$jTtWQkX6v0!uQx+EvUjbm2|Ja$ncz<=8ULt7h9GE&=qB;Mi;_A{^wue-UX=<h!opWrN=pQHyGB)7wXZu*|>QTGJMk%&EY=Lyr9*NpT*8e3P>zyRXtRVwjhNL~wu)i-!EKUixn<fNYT}+O-lA2s6`*P%7T18m2|nqaSxW(4D3;=*SPw$AOP7q0VYuCST>jnCY~>))HJ*ebSAtA_BkharV6yLZzu*^ffQ%bYNTZw2fAG4pRo;5<LYAA^0_5mVk3U6HsHtQaUAib7t>CLG8PmPAU;I?Y~PysXIDYiLq!#xAy7T&xKr7*k5anVd}Vv;vY?;CtPE=KiZB*0FIfU+fwqKbVU~X=MDiNNZ>2-wC$+|H_<n2%kW9jRBeVt+GQ3MV6M}B=jlQ!Tch2N70h`8$d&|8tcP9v-A&3{={l(5<)oA|<J%&ruhPef_Lqvkhq4N?a6%Yg<^n3o@6Vstj^q<e6iT^I3<yt@J$)t2!s?p>o2#<PLxY%(Q;wBEr2-kN#=g&RF?HR}<k)7~&H2j++uzC)0^?>@FDv?Vhu=ed0A&I=YQ1$gRj2bm2U`riZ3e35#aI}Aq6V&+M2Dsm79nDSQLupa!pJELW{wPC77RME1s#Y+PLew&gJMj&L*RQT@%7FSCPTBF0{}tzDJ$&5F)MQ{j1g{<0hs&<VnRz6Iw-Ai%S)ztq}cZ%W&+G%ek~P}>qY>A-s^LW%C^>zz4Yu>CnkuLFkcRXBB<op{=Zt9B2bcpe9xgG^JKGO0kV8H4f{yjRLK9quiCCI^9jGjIFFw5xQIAUcc2Hr2f5GLL9AI<egs0GUKS-l(M*mCsKju-FN0Gv2~;2005kNt%8&oG(ZChIm%)O2D3xGkN-)PDNyBn>5J{Q!`9Lqfq#2@b?S_;iA@Y9GES*J26mhpe{FR{v2#(N-tIDgQoNn}ad?ttH+=)cTp1Zey6kH&3>>8wI6KoJlGI^}UL=g%+jW(Pmu-9k>=M_c!+<jeIAL1yUDIxvR+p&`_<k5~ENZ~CUN{ctiIkoCsAhcL~{GSFE4VW&#yOZIHdD-|gd6l6UA1$v1{!abNIR9ooCli`f?0|!DN|-&4*@(5EmT<bajx<kzK8aZHJ-Ev8ia(1=k*N)%m}Tfah)2awhs=^TdPc)(f>y<Rm0hH*c?1adB9#xe()Z9VpEpAez_PF+(e7w(EE8y2`1%uCBIHm6EQTU<dAA!1T63T<=0l_Cn-X;z^UPpbT6czH7UPCxVRjw(U1<suG1_7=-?5X)pL-<hY)FNr*UjFOS4TxF+HIY+Rg8OHEeo_WGj4niEW%^Ee!hzxGz1CEOH}GXW*9hUG+c|0NephWGjwwhVS1Y>=@JIiXFfFW?c<~i3U)AI?UBIDx$OyahgbFKRyKT>7fi2%I>pCBDiQ3$N}1bYxz0t#v+&O;gRqoJayZ9Lh*R9L6&@F<xRWK*a9FpMCXDkNq0Q3(8%I^JeU`CqD`8ZP_RF!qqb^XN6DE5m{<oy5+C}5evH~izCwDd9MxJc<F}gPaLMlZqxEpVF$R28cg%hOgzK!szdATV$dZdv{08DIYD2tG8Gb_9uT6n<7fT_zJBOqwrE?$m0_m<%9`mV@@fmFodVxdx`P!U1pBl?8pkq({AQ>$K&){Z}DTi<px)GbVK*A0kM?w$k+H~j~dtiHI!p^H~gHnQD5HF+0hJ6AR<B8?M7emqdwZG*JR(D`g+J}WJ(Nv``pe6P$~i%*S_6d^5Nd)Of&B7ht<yPK#oL-@NIn5S6VC{eygQf8wygycLI>3$>V_8msj1ybi_o)kC}nVbr#MXI^xmu9_4*9R`IX0Hnq+ihks7f;`Hps0GN#b4lV8WpB}L_u@}nZiRO;_tQ`NX`*|0zkE~*uT#^Y(t7nP2(OAUbLl6DYU^`oreeWfc=0nb2xk5af;JJ{1obmc;S!^x8`qj_mbQe!CvV~?VZktM1@SXG-h=SsxT7c?=x#w3*3>DUN0U;uH|ZaN$}K-(2a{Zi>VRF@G0mbbE5O_=Ip|H@wE~i>6r4N0&2g$S@Rz@2?jz#`m1nPQ<fmky1S!)>SI_+hlJn@tM5!Fo*KMDG~rm14P?2m4l2jlZ<05>PKN*F_r8EQ9D;Y*4`?;@kKmS;=<&+?<K0YQOhxi;LBVAI^vlBJRMbvlNns<raM&l?zR5xp?8ZG(7G=Yf4!|7Uzy47u;y|16yM;;8y=QA|lXWO<`6N`5tXQX>g;8A%Xga&0*X6c-483|s`zt&c^<^&|Vbw$}0-V_sWv3|XkBiIRh9m=p%n!$2Rq5ZEp@{aC7Zgy;qObo3Onnehk6+gL+|!I%^J6b_layLeQUw+G7A&&IbU}FFq*v>R3bcP&PUP=X>|_f@3UE9X43P~7%bnyZnw<qN$=jrM;FQZLUKr;G1W)r$p6^;m(J?6HF@h747RlsR4+Ekyy)N+77AOAk>wjXzl{1e3jfTpmtXP$zw*x>Nw@szGfwhz92Um#$pDCREBZN+)#VyyT;Gwx6l5`<NPBMfAkus=kDUq_s<isxB`KZ~7=5jWuB5rZVYjX?n7GmeHOFfS%41O?b#<#PWMV*KCgizNi@}3k4tF7U6UrCE)jY?6!z})q0E-8~tN6BjKgZ<}R=r#+PQW0It_P?CT0*{d6&7NXPkqhI+{LJ95Z+^r%K-LNTEdB=TfI$rX;_RPU$oeA8SPc#%bhyx*x2XgwaGzG(Z`c9t^HhQt4o8J9f(e|}N0GkS<hz3l-oGDQoTPuH_s+s<UTp&GNEZ#oJ^6CprWC61LcrtlKjuGE3!*ChKCpB=b6ezeNEpM-1F$n*WnxR4=$o3Dps{kc3l<#FKjVt9b;hg?cWLt;mw#;PI{t1+Q7_MP&CGmYf8IoE$P*6AKb)TrroX$Krh<;$tFMpsAbUR^Pd_`y?+~4D?@kNE5sd!J3RFa7%QNsBoi|Mc1dy#)tcelbH0n>j<L7nKcp!|Czf>eHZ3X2abWk89cyaLX<#p=8Pa3o7&8Ow=&S?T`8)JPYg$O2XiW?2ONk@+63*D*_<!K(gs+_jH<j&ybxQ^NXHb&K9hnLHBuB<8IH)>u~TmJmI`PB4<NQwd2v^OK659Udp7gcw!jhSEzIjouNR#ye`#Iim5#oApH*3D#_MHv92RZ46hc53FCtSsmm_}Z&=w^-`duypGbrUrL$dWfz$u5czr*-473m7X7Xjr*fWnA7<%impIz_Yq3?Je~NW9{e4-K#0>8B>1T<br}_XGTn}JS0#`Om+nhf4)|tt?<}hBprb5nfhx<jg-YmuFR8EG@JL|SGT{GxM@>dTGo+QE#CZCZIhS^3(UGc#>yat@k&6b->hZuP3?-Ns0?zsB+e@y=f3=d><>@r^#)Xva3iH+bW4oG`jkib^)VaABzo<L4yr~$nYZv_;y{*D`)c0a)2z%p?>^%tiq=B(mNlP)r$*vcDI2M;o2moJ+#J>C*>}!3C?@vD))b5Yyr0!@s2#Zj<(+1X&M`J)Z`<E$So7TU4?j(d1Sx&rE7hTgHHFv<y(;hgWAxz|dsQc`?dx&A?00<mO__-O{w1(j^{3@V)%9%a7@+t$t>!Ge+@ppdIp!9pbkK(w1m?*@gx|8r`$3<VH%@$^yMO1!iY(iV@yvH-DE#fzFcPEV6h$ga>`)IfKy&%Dr`cLk;OA}2Mu)@o&jPZ_RW8}LwcnpVg5x51Z%^qTM>Kev*+m3Oh9`pzwkU@jQZ3Mh+HmmgxdqB%pT9%0XG4|g4I>5go_cVwg8X8W(5Ns|HA;bIV%Hq;!p5kMM*T8aZuE>PVNUBq%9Fr{v6>%(f+9Gb4em|Qd*X3<|1IsBXz2o_#+#6+epR*@pbh0@O!t^D@9iw*R2?-L2<L8a48m9z9?Hw0Qq|wYk)ZFm3n;he%mC=08_CC4o&?<9i$ek==mG~bR2Wlf=hWJwQfD9lES1yemPw9iIxtrgXI;Myx1--ZmD;FKS7xCtO0L3xvBes3m+UkfHEtj}!%qqPH6gomprF*A);MR!HFkmKo%;G<%>VzMrzX2+N@l~1?y_cjHP^s8Oz+1^7q4lG{Gf4kZIq+>-(l!iefKO7?icI(xaNRD~1xcw1tSfJ;^gC~}fK(RF9cS9X4tx$Af#;@laM|wOH!aI8e35ND#8U+blET{h=1{Y^W&kY~$I25B$xpw^#gl)_=BbS!CuD19%xIBN{E6q(w+?1Y-yP5NsH1IrE9Lr6hA|)Q-p~3Pszuuc1b{E#E-`vvj|gujqg9?U;&oSJLm7Fai5t**cCb$B-+4#%^k5ancQ|(cXS3nB+;ve$Zz_gNQV@X%uQQ78|Lu&RaqVP(Eh_CgZw<UG^ZtKhMr!<DvcJ?W9Wi=5OK8uJb_Vo;z!-d^1HWiAOC(v{`(9TGI42bo)$dIn)PFYYnI^L$1D<yiTil*&N}P0ZN>ITvp7nXa%+;4z0abDgfkb;tjnlA{rV%a=QNp3od&H+=r$raE?w>)OGGO*3Uc+j=lgz+)KOIHaR`dZ^LWg90&FZISl|1fW=#^pg)-8HUKO2G(zd9ztcDoujCyXvfWF@ti?&|*>LtLW>EfLvU!4HLsMn_SYh=|1hdBG`JluU4a=<{_LM8Wx^Xb@JL%&P_~qlJha_Rt4NWCM&Am`%ECF45QB;<9tHLpmX7{Dz*?w2gauG*GE1Py!8n+%#f=H3)`TOVHcl-N(c!OP-p@W>Zqy$$B3m731ts<-G-$^OHq<_ue-chBRU|`$z^;>0Lg7I)$~I+&M}Yaz6ohF<6`1A$LkBdfOg&05ldN$sgS~Y1!x@VL<%z8T6jqqyd?*%m<;KIOJu3E}jIKLyNizP>hH<-o>3n<RlJEeKtIhY{Mi#DuTU$=qkk2x9(OrqUG)1lfCBUdt3XLJX221d1sH-8qZ&6c~E0ViVjLyWm9}6GYMew^heRZIby9P_dOmy&|Jb(Jl{WX{jxQB#rvRXnd~B#*t_T(R^pM(5jSn&5;XxxQ`UBn5RAY$Bucn&-2qEINYdlaw;3sqk4g<V1@NwR6<$V9F3g#K-=WU|0-9VQ+T4OscAtoD_2Tey!Db|VVrCHc(QnZn)%QwkV5DqLGXcShbl0fVrpUw*K28hIu~(Z|YP&J|O&+7!i2?tGh~v*-2&I_WmQSL(qaoD>&cXVE`%*wmnjRhHTAH1Hh%*ceO7MaD^}(>$p4w52HH9jC-vt~%>zf1IF#M*fCa8jrh0^4e3FM6cL37ZNU_wrN5AfU?M{)8u3GV}Vn|Bul+X+P$8|+g#n^0X6PgmFB(1L1oi3h)@QCtra`j!!?|IL~@%6m{&F8fo@gYpxuvZalBDU!Ybp#vp`(Mgum8{1z-HlN{TXCkH>vA1_UnL25kZh&!%69I+XvmGPI<`KOZPzhnemY2>ldrVhN{w`@`HkhqNs9xZn`Soj+r6oKETin#~+k&$dE?LKVB`PVhH{P@{9AfMxG`u|t_5+xKTE~orFltg25u!TqYD>hF?TzBL{vLiUKlRbp%;337U9!(xNI4#W(xUnH00E=0s~%{U`1SDp^uCzw@{kohTvIzCpgx|;fY5*>jK2nJ?+n26_Z^g8XUX)vkCzH?Sv-?bITS{UdZ66-l+4YV%{&`WS7$170MXyPO@XH1JWH9+68iSG{XUgcuL~C1gw_qYI-D(G<<+LCy9P<I8sEoQOK-4K0+8C8D$K6uT}B!FpG#h&`4mHQexCISz7O3?bfhwRasPN9HN>RCXp$8HRr|pXnzk6LtgbN}3D*AiAy<*d&t4uwjLaEzLn9|gxk1vY{r|N2|MWWYW-A$*hYfjkixkzK(Fk9;r_LYXLIpXZ_9-XAaBeY@oaA;ly+&{2<3NpTvXf7Y<%NX|nrj^sQtueB*^1(I>%d{iQY**bt>5x~zejf#L6<R|yhfGVxvvLqBH__cj6@#3<O;Vc{<j-^;$9M8+9a&N0X<lUy;?RWx`h4q2t5(3JG??_tTbS7(f_KV;g}EEXvCPsm_FO8>&|0Y_%DCQTQ|#Oyev6E-Kzi-{B1~BcYotU$xJV;KbN12pbxf%&kq3_)q82it@DW;+yU6nFqZwPy2i<lN-JXQ15PabY}|YT)C&^K3aSwpC;RC(1vTq1p-Ne}f5&>mm|;@QrI!yQxd#3)H|HD#KF%<d+JG(mjhC%gI3hyi&;GcLTj@}lX##r)y^bh|Qb>NcUXer4$r{=F-n?Pjfo;YS$kL~_uTMaYyyA_K9qD$L!;6SCgc?#O@yoaw;a6iB2Bjo`wH6IR+Xt?OHvxd(9ss#v8zzgofhK-R*<y)ki2@lkWBO@jE#jrwnz<`qj8j)#_^Sm>4ioA;CTS{2`mg4?^tQE2w1N=S2qH)=McJ-)^7nL|>ZEvlASBWM{}Cg}XU$iQbV@JdQIklM4eP;gAMH4nG_Y|Om$-e1K&8D`_ogp_cr1}2+KMBFfvFBT{=ssr;y)jd6u{M$ITWoUURs)ooAL5wtce~Gd|QrQM+aGJ(<#ou_4Q1)CtSe_X<<t*hATcog0p2)jqy?=r=yHwrBbqev%BvMhuF?>?P?HGIe*vwz|U=i&^<j|e5`2vi#rY}G|j{&!tlDkD3kW^GFmcl)5>&rKVl`@^$0q<lRS#XRnTTxVpQ4RJ{31a$0al|Zo#4T!YF0e(3sYUgcHxbYNW_Y3paJR_3UpBb9JmkJ*ZZ^T+uYzMIT--WYUJL73u#az&yv?esufjhh1~&RSyUmhlvGH-A2{<xc&{ZHC($|G(lj3>YBjT=7&%n%~~2iqq3`fjx`8IS50kVk-WJ}6S<Jk&*2h>0q$WVs#wGJ`id#UcE$E@O8iO7fDzLCIj@6z>t}(`Bz(TOAHNtM;ieX(WAAAtnj?lHV4l;8>PVXX3s-2i$0ueS@&V`{o$=@23n(4>yx3^~+rL0i$c377*^`7uq}2=BJP&sZgJQtys~9!Ei)$(%Rfy+ll!P7^2bT$Iazaq*GPUt_J!6g)cRvb0x#2%^k0{XgKT;vf6-XoqCBhvyZ=$F$5_yKtBfqoUL-WJzY*YYNFA8<amfm_!d&RVC9RUx|hUxzhSpHS9mDyce6#+S{@@dc7!S*fYlO0WBS;-^Qh|Hd?Qn-rSAk*gmLJ5|qm4VVCR7!xA%JfX`cWif|N2~Nj%7x@d$eyqsjpwt?V<F!e+7+%f_Y^Y1k}=9DIw2i@=POypyywDT3kAa&{4s3Ooh=z25yKUlC7-?<`|j5;+`~f$Mh7LIwR6|DC9EV;xn#7K812Vp2-)0L3@J7Vp6X84pijRg$MD8}RI@CcZ@SEGUHWXnp=HWJK!V;uQoaLkO~3F`H|xa>Qf%q`V;xrlPZiw|_2wmw3<pa5X<p)^%6D9#MU>X4cmtTsc%souVHwT$+VX1wII?lEL62h3Xj!zGGozVg55xK6{Q|Y^^4aP=^R+n;s)Ry4HD9dk;=mrQ7Yz#=lK*QkZ~7l?TJl%3hc7+2uVhn+=v!2rb3yQ(cAi1x55YVzAbgXdn$0avpDvhliJ>#2nQqtO2Cpo4B5`rOGM7T-TRv6o-zGrd`%GhAozUxFIc3OO&X$mO1+&MB_H&EX*%6uRWJw)pg43T^89R$QqX2boXs~n_4iv8fVkjD5TxMmNiELeY`eVpkB?c1HYbC-e(k5@XXoX6Yf69+7ve6YfC_c4_j#M<FhtDVwd4o3chv!&*&n(;}<R+lb3x9~_yk<<0u$?Xx1!n*%&aGVdy4#RsczsNWf?0rf_~PV8W%gzX26$mQYmCmy?QauJ6^-NMl6jorUK?2QV=BVzdPFF7sp^|$7VN%DPa+O0B!g}({9qu$h-0uDbW*_2Ds)aL`J<#YtpPeQd|AP>iz!%TPB9?eqi6>6VCWeF)z(|)Sg?#c3%6H+84RBBXQQzMdMkmHl(JwS3$S>mw@F|nq48(jP$e&H7ain+zT7(>hZ-r=%5pW<7w%3z&xL)d12i}E72vctm}{+1&^!U0-G~c1GaD0!_K3ENhLKq<P-+3&ekMs6S@Z^&{bX++UfXbgjoynG%Hwoz?QBN#9$fFibh}+X_R|W~J0MsYuavU2k8yuWg6CekcOkE))~cItQP`F#uB3B^&lXLugD<KuWI+x#g2^>T&r_^)E&oh+H7-RrUzSR?KwA2@fTdDzLVrTmB<<PzC4Gmw9#MqfK}L-b;*($rTl7xA4k$%79<VS*UQ-pc^1f?!oa8_dTD+lV<A5}Hc2;;b$Tes^%XlMMia7|btAAFZs#5jKGYBj%5`<-=W&KfbJfYyzYEte*=oD!D=0}uu1Sh)sc8M+QD1aq>yBjO5P!4u&eo_*-tVg2r1IzgzpxGQai-<`(QGfEjcT4*3{CC*U**sA-gnpXg)9@}nb!xyEUt9lYSUNSOp<|FhU6%*#9j1~NwEzde<o5z4ZK6-zq_M0b<Io1-nuu75Man*T+=;2(l~59U#p}s@rXiZ>?(KF?O{01C(+Ec3ie-_pV$_}#W;Xsgmb{w}RZPvk7(j@buYnszOr(4SsiPN=RPR00fa}*9secp0-_JLxmqipHk2T^_L0iLMi}t5?LksfH7EB(q$1iO_&=Up~>Ls&3iLM?>(`H(A=|Yh5q<?KJrn{NUMJJ?POFHo*R=s$6%p*=|X@6BWd?YoTdd}fG()qX7RG1W(l`gOx7rTBsBjCorZ##2@p89t`w(6W~^#?w}`-he4Gxn#FmNKix3s~EcHQ<^`>D}-aN%z<LpwM_=?i!Eg#htZ7cLT$n!)7QuTLt4dH8kS(QyFwch7&~|x*wt?C3&L}8F&KrdXE0cR3!8hHDXG0f=9!)etqeF$t)yt^>`L81_AR){{m9M1CGA57#k{^;LFZ&>P=)&uR1Pwdu_XnRMi~T5Wxl)5`HKzN3LpPJ#v3PG(|p6Km+onDWwA|Kr8G8bOomJg*o55PMaP5U~|h=FE`%rzwSLrC9R0;I*uQGjr7fV<!+fIfWRp!1YI+J#ciql)S!H~2DrUWh;wMvffdW*`=bHkT9!^6H<7kz;S{skO#%+GO2?P>?hfE4xcJ_v1R3*}sFlmunt5M<-A<p(W^9K(BEi&6G{z9IjT7C(&wQ>dnVE{t^^!oTJ8<?gOzw=&nT1WnX4pVxhS#&ioXuR^N$ZC3?9h27+Q<AyG9^b{Xuyecd9HbJk|1ae-^oEG<sZZ?I~6pW-P&52LIucLX}st2MtRR+$}=R#w1c*vxREqZzagB1i^(L1IV*zQ)Hsc8z8+);>0{&O6-e3H$_NP7MV^LQvu;FZ6!ypq)9#U^zaBR_tY<0f4TR64D9!I^V3i&j@9pL={ieo?B2iPCmF-9S?R3NP1$a+9#UNcE%B39{7yO)=4BmLwTs(Pgp=2^3-loLYxHqON{RA%^)UeP<(}?SE=p3_8Vnhtz)Pu10H^@EI6`#i5$6lL~33BdDhGS&oAg!h!N+$dYq|Ldke)#u*q!Nn^Iq2isJg7~KayTGNLzFpk{ofMg9ln!JB+ZM+VhLZR^_y$Yl6?vZ$8uF+MiZ<&mozo~X}DT@qxu`NS|^C$Od&oVqj-mfU_KzkQ|Q;xFu}DrW0f+t(ms5xocjJVSg2J!;4D{%=@k<J#CzL!LK0e93tOZIYN5@mNd<3-^A7dLf%<VYD~*V`h9OFnrY9`^VzLN9#pJc*j34nDgb64p1D)L-IAkCbX)zJm843J~f20zZN~sRLgZ{#vh$}+CT>3($<pV`7EbwO-V0tqGCD5QU>@j+hiqY!MbImCvCk)GgjEaFfLbl7eu<mX2ybS@RBQqB+^*zrCbi@dTtpum&Hk_dy#$ZncRfAoQ8>r@Vxu5f!+?AvkNFGB*%Ss65=zdvLj8X#INLVP2`TH^d-{u7UM*gHmOeLPVpnLcCbk}cAU%aC`eNIPr4eRta;bpju?mli{BwCXWM3th0RzOkKyZT9#wC8?Z+!@?y7$8)Imt!qk`yw2s^NY{(SdmGIQ<@jeOZm)Og`yu7ElMOky*P`Mp?@nB+3zhXs}dBBU{hU{_v^wSj|$TE?N#Zm!Z<ZUz^%3Di*{?qBlWbytXYFmA83eCS|)ayY!#h(bVQeg!#&7CEWp{6I9@6?*45}`s-yM}#IDN}uVBC&y(|;h`NEGroDUeyrge2HDvkP2gV{9jy)9X|_BFvwo}wyzgGgh~aiVZKhyq0-CV~1-u;SPVVp@98ubO5Mj^_#|_11izZ=6_QlY=M!)BcWk$q5|-UAUCPxfIe7?O(J!zbFnEN!Tr&^eYdOR@;t!N^X95N$cez@-pKJH7kVPC$-JJJ~H^s5dunY+Pk2amrz4<mFoPG+|d}<GD?c5U|mGxk00dC90vW-_FR#)Lzb{z19CRF^G**q9p@Yei>=jqO*I#5FeHdUYqkCmpBFU3!?l1!tX4f491n>jF1GcksNFrLiq2<z5;-}beS^1FtMHmY;nxq%mgk0qq;Hx+@h(f~`KeMm%9L};lIHKt!gqVUW+pgfa>cU@(U3l78=TA<$GK^GWu*Cjb;{?;o_-oq85t<<o}JPmhFIfXmoR@uq;O?W)8yB#`p-oli#fYBoKT%jNQPjiW0+CC?#9}>#-b0jYEUkig?c*kY6ew(-np(A5GPnOO#IDLUymzM8;)c31{`G_=q`dBxdh}Hf|Yz@069!7ompmQA;uTwq3x8)w$;;)u(W+*4`ogX;P)N`7R1HWC+PfWtXmw-Wd&>tmi<2jI%;RA^|YY`HBImE=A8={WdX6O_fBxw9^CH8L%WE=*ukRrjLlLyO*dM8&L&`GQq^}`)^mkFIYs<8T^sa=gfA<GQ-{eozzxKv_9gueXN%~2oMfFe2v#7#OS}p=S8!riQE1=k)S9dgo@M~ouLZ5k5fjmoi$55C*UEb8k_01f=--m^CYWFjYl!5Pmj+re!zNDf>)NJxdQbSanD`I<WX#R-dJ7Rk?9-&rZ_IH+F}c&QhSfroflCMI$Pq!i14kiI9HmcEMR*l*^}G$Fa`<{bhAg7%D+3&kq$w7eZ}E`*WtL&SN}S?k6%R&!&@P*ROo}Cw;kdfdlg7=l<JXLaW`I$BFQP&wc<GF9TUEZ+&n$ZRb8;=81@<kW_A|H*`P(^vM}j=zw~V@^`Y7g;9Xntw^PwViboHb}zTKK|3B1*?(IlDB`)#~K@e|Z3tKcT!w+#@qW)xxZlK#0|rRi52)VaOx^L3O2O31Z1NZNwF2u)Nq<0XB-uMT=G8;2m3b|af}I>O4YbW|9((YkoHA?jeq`YB*GYC-g4b9PjI+D3jsjc52y+nO68G|Xco?-LMy#<_#57ZhKY84|zSy!V|{jPIVE>`^ILdC{2Y14Q4Tq2Xzp3OjCkk#oY3dMj-79i^URM@Nu-u|VUKICNo_)0Xf#r!Po|1nt?=^#&zm(4o2K1C`aVrE~=+HywEC85AEK_T_kbb(=9vt}BVb@(G*BibK>c)XvE%OYS;Zqz+%}IOPZ+q!hxGiOw5USA_;;-fLTTXVqe*j{zJ+_hQkP1+&%82m!ZMF+wzP=)-oC`}9}VH`f9ftEw(Egr2E7QtsJTt5;bIbN86()>%e0#Y1`02DdK1n;A@oJ6B8<KPI*t-*3kLkoN6%G#O@ok9kQp9f%5oD@MpV905^*DneQB4`bo300k!4d~Cl_$&Ib*8~bn-cmc)a{w0=ro2l{NIlS$#lh&D$QcJnD9}RM7X@=2;`u8@A=QU21k0^6(<a2wzx|!~Q<L4`Ig)8aQZ9iLr)vhnWGE$hOAo8Vyh9z;5X=aoIox0s{X8KqJylG?sbTp}$#&G@zsop%HSR`y#u)dN>pkwe*NlQpF(pNOMe|=Nxi6f+-(-g<;1>}S8_XmgPl~YwWoA?t+-ef?R7^bNNYhOZYC9n)j!m)cz-m?{SVrwvR3C!ufhz|HAPsfown!UieE!OSTWKjyG9(;ZH%~yP-3-xh;0X9z<y$-~mYcVX2%B>8Eg=maXLeO8AIx6Sx<`!+W$fIN|x;P;Oi;_B2QZ&bT_XR6y?@fsOZ*GDCw`oMu`%H%&tta_pM5?FpHt`d%`p@?!oJYEyJ;r4;lfVd$eHAPlDj(afdQ%sBoc{OD`v=}4$N^PA43|f?;CQlK`3$T~HlUe(23qxJx(oWZx=BFG;NQs|z%^@Ir8ymd|9@XBs6<OHxxB%x>H3l8&m!u#L{@8BKh}^GM6j5ZS#9G%3~oH~FghNtof*(uZ7}T}V>zJ!s3=}_>CS$A9^IN`;I@Kl95?YgmR(WM%KDA(45IP{H3QamUV4L4ase=wNLVxmU<l)aDLcVV1TJ!81-r%|kJD%U|EZ%gMu7V5`?JLj+hVLvF>2KApjmJaT&z!LGIra1L{@GezZ`7G(M*=shyIj8whHXx26lIegs1znsX_YTB;;`Mehbx4a&T>TAUtV@nCNab+6=2|(d=s9BI3}+-1?D1TN%gt8PtZ2n_}W_LV6s9BF{k}p0S(Qnvsp6!kJ3dgl0znWX2P=UzxaTa?ore_Iy%kHkO&%AK_ofP0XrPY<IAd4$st|X*@0wdCw=ymS_^YAP8OQ&yFFBCo)UZRlGdufH>&J{z6)X>Zm)MIO>ruV;T_>9k-u(E9z*@xH4MRnBwe(trbpJ^#Hq@Zo3ArVk@Oz73Y!;NZQ7MGHz}33#j=4C{FFus@Y&9USW&}RgnP@L(-)i=n<%Z0vb~Kb|vY`bH1!Ra_-IM<6t9AZbI%qOo)?3R{L~wk&r4~ktvid=o(*Q1DBjtu4MjzctqD%UNUy~#-1kB-~lhW%jNt6p_YG7v%hm7iu!7hY>{{d;?{Z3Lrh(Y95cWA*1LXpk)}K6ZWg&(pw()VvFo2Q-P=%1{JG4vNtv##XC%+uKsi{qFSmRY{e??(16*yAX$3zDR`+NEVF_0&g*1L+|J`W^D)t?LfE78S`<=W+6~6!NUrm*g^gm0I&Jy{(?0>3<++?g}wW-XsD-g73U8U~lVRsy;i%_rQ&3<!;>d=kp^arF>X^?7JT}qcEetEYKlg)v7QEQ|PZ^q+l0;jV6vPt@4nm)PGQoL+&Pt5eP=?B#ovhL=keX$C_x`=0tL5@@*^!ur<U-ad-EhOPhsvmmg$HqcHc_5vo^>*Dd_!P`ax=v=I!knJ^<clNkoa<FkRu0K-MXkvlPd4LbC!j>|T<ISLNQlSFnh4taG-Hr@;>w-{pwpnfL1kwWZJS0eRJ@~QT$*mjO-Yzow*0WcDaqPNIT@5~uY*9_E`6OD%(K}iPBTk&8|QPV_4=?qZ8&xDgf?AK6%Nk-XWlNiv{X0~g7I+GPz<)FQ-H>pgV6Byc9jK|T-9^*Z1?m+5I0UPLWVfhYOA}%&4~3{Fc2wYgfIA#388#pT0vGEwyaW2#_A*gkF+#HD3y!)xHOPxM9}*oQ#={k#ZC2ZB`nv3`pWy)hZIuX8CjvZ75r=8z8d-A8xAcxPd(u59d{<Ad}noU3X2`fC=!ZDWp=!x8Db>PssmVR*oS@`n=`=mqjMBLscKo9FmP%b8$B2dwwV|Xw0?(-<Tn!&HZvbO_7WcEZs8%B8{PWeLN}$Uz|iKy5@Y<1r*of{ppKiiawKaJBP><&4s=^f23UL+jjKEvd{L-TyHOL;NXJO@U&7)vcya}YAc-|znIq;YeCuiS^SUz>Wf2jJ^1+M?3B?a_cFi@bO8>J5lv$Qv>3FT-EXgDxQPdVZ{Eswz-?}H;)Y#33)y3o{G}RJvW`Rwv4&jS4oGsdwR1+<b1edWQ2#*9(riGsh&nSYq*OU<$tV!{7gxM`$H&Tc|EL|BL!mdQCBQS~jSh}r5ZJifR?=TrfTo07Xsf}xON0cT~2Furx_Dm&!=jDu{apraJFjvXg;)Jnj*9X9`(0!ov{)$pSD|l+h%{S5`Vs=X+vq*zvAexYq0ZTP2k7pHB3u+{QI!YjKb_cqN1dr`6KwQ)9HvpvNB&^U$Kj+z&TedIR?ihVHy>46wrMDHMiiSZc6*JjRGrV8`8<ok&+&(2RhE<>oR)t0aQIZ<L33!!vA7H<TfZr@AJ)}OvVFY(T6phW(bdHKp-!<hmlROJ!aAy~fE>l#C;qV=yNd19aN<p^HxIO5Js#a?ANV`4RIp0Rjni%??&nek~(IM*(_gKeeN;hqmEQPXh5;V-F>1v*@-CkJgVo->4*U$VxHm9mg5_(oY`=LIC)^6%Pbx4Bw1OpU6dCij}CeoU@x(GTrfXrx5tOA+FO54Pe%kAvZ{Z=Vt;mSQiuP^Roq$}Zxu<#dHY6C-Tn_{6I<O-pYs^01cvw?1Yl{YfAPj*0yGFwJFTXAWD?dtxQI@B?_R$(}D`GJz!3er+b{sYoId>N)jQ*~G{zR2(*6LIha+DcJ|Y^azsCj%_j4UdZMKzqr+?)_QW=x8#*{Q5pI+23v<TZk(gkD*XMS*C#y_s!I|xTd~xEse(1MA0LOtGU3g3S~@RK>V&LVS@8_3Ner)(kwv!Imp9v0Stns9Em4R)D$zxsDt=7=*k684$AIYpd-eANgsw!LVPJ!SRaCs@MZDco|3d7T0tyq)ZgcLS{^&O8Vm*sD)?n~9$>Lf7aAHK&H}?ZpD5T}I<}VhZ{-T1Nc+Y{OrFklFhG^QV8jtF<eSEYtAi-c@6S-NOk&t~Da$VekR1KfxVO0luYiV0CXu@zFLIrj+0^KQfZd6#TM&bb(SYW<j-KJDUmiHwkgLIarJtSO0BGcvCx2;0zr#G>n-j#YGEE7${}$m<fR!kUQ*|wnjdX{#M00{T7_e(5rENLTmlERr1;30X4VH-r)x<nO93zqZ+uiptyIKd^dNCG0gmWfL%0~ChY;W_pN4_K}9&K6%P{#eIp{ALF4fd+mvoQlIb}M9aMK|Zz-F+p+(@eHA2eVOo`%+$R!j4%LHW;Ugd*f7Xqs4p0r%3@b3N7gsH&=Wy7U1Cq5-jX=4w<oc;+<E?a1R2K+GCkceKvE>j}Jehj-yh@+YO~8^jw9R)16A0fCy|x#RPEse%aCycjJ1>a>vEUJ_ho>9Iw&YQjmN00u<K$8MT8flO(UBf=N{{AEUWkcab4pbO&;!1AuVZIJREzLl!vgi#t{lg;fAinnw2g4~E~MfOURO@cgEcC2<)6pUfA*aF=-5R8W;OvF1r(#8VRc7`-PUoHu+31Zj&)PP#TM&Qe#67(nEUMyZp_a$ezOMH-sRvb4~y#Do**JSoV=BBIHL-UeSf*Adj^&SX)z;Jid(IvefF1QoES*jzQm)t!$TvyPo_JpWK)c6sXX3%h#SMtFi|B4wj<<^w>$U9|JA6NAHcPmlLMyO0bFmN`%22`05Dg6e{Ts&UaKde2<jttO-)z19@T225BYr0LnmSCJnwrgaJlOo%!8@`UVa7jsb61r-~Y*cWJ8wozEWbKWbZ0lwR-+1?0;(wES(R4gE4WDEu^+F@LCD@Qz8;A1C>xfPV+r(K3}D=Wu0N8vY6pPZ%c`g&7?7i##jk)ci;LpP09i)G2QCT)SjxAB}}o#%Eammrzq7JYpuqzN<h_TZK*-CU-{KElwgqZYPX`Si>1;G~yC-q2_Vu3Lr3|Ay?kQb+s!>T9qp%r?YhD*wI-(H(8Oek+z%CtKel_7f!&skNErtH`5SZV#H$^tt)JJh&XWH(KQNe6`AJ{kIb(ZIOR-f;Bz<jurJw#O(>kt4p}1^s-Xz1l7Z}!3QidDL4Gk50M(T%qndXTN$u6JZz@YsO92vgqhu5?LH4-!Px^Tzo9RA`0oS>^>c~1V(HcAietgjgIWF;S$mmX2i2=<0dtQU0K<i%#@L+n;C})~h6SSGjLav)dufU@4kYtMg9m*TJ)I;#Ceo#(#$VhPhI07{gNu;iFN{T)U9uGWfgGH@E9lVY2I*o$o(3$pb{s8Q_>p+c=N(Azl4JW_yD7;9r>>7?tcrwOwG}Y%SuOc)1?ww61v1*9PA|v7yF=1}%*@=mTEQ&I%|)6Jl*-k2>?(V@R?2QRJ=lox?J*Vh^tqq4h`yS+?+Vk_iM@as)rh+E^Q$3zjX+FE&Ea?$SJc&xMCYjKX>b`Qs=sf(7n;;)5mytFGJ--V)?Dej0}9B1bD%!}Tjs4)TE>thXf)$xFdmZym&V7RLfOMMX_T&@Sb4`IxLfihczhbm*U4ILoVhkMF_A?KqPf%*3nH4a6AfZ=GF=Wrs2JLExF`^nQcVcU{h5z~_#J^Oul=YrwXfjnGGhJa%)Wg}ySc0FQDs6_Ob>DY>lK5f6eFt>W-HJQhQ)G_^p(EMn-5o}dtc&pzgD33FB`B&N*b2jqq`9F)l3=fg(SA5w#F_<s({m!I2OkY`pPLI#EW8<^!X{=p2N&Q5k1OvSLf$&_(9E>6xQ_x{-KS(X59(G?n?DZK#Eod1iqF<O=YDIp;GVz6ec>w+Q%Or+<X{LIKsg!3|vBG+F=zAjw$hLum*Tg%#5V^#ccn%hjO=zu~sv1-7$ERMAy478B0oC@5phGMoiE_ytMuW82~_mJk7DdZI@A?_!!nJbNNV(z$0s9W{jRt+IlX)6wGPsza)&&=yTYyk6_U4fAZHX9u(|10rIFosM>&A7SN{vg*cr9;dJZnH~<@2ewL9apka}~P%*Rv<3zupRVKOmm7;KjLi{x3*2G}n<J_xgV8TM+Y5I`8ig<!7H;qLI#$Y>)Yf!SS#&S$IK92qFixZXZd*dXfdF}FrE;g2sr7I=&O6I<{mJ!RX>yRXw_tYIPj}UFs?4_VLiB|=2IaV-XrqtCV5IKto@-?%1`?PgC+tdU<Ne!HT1)Wj(T|IjBohd@$I?FT<V5~LBXac}DRK^HK$V=O)R4Kj6;4`p5o8;=m>h@1-A%D+*$IeR7+!v|p-2oC8XtI$t+4A}%)1!83E|#)*Ey@lM=!<5$FChNipzv#VnvFLMZ8Q6R$(~pp(1QrdJuW2F@yii%;${*3v`08GOHR@EZ=P^_mKn_@j9wcM->4*H(IB@m4i>m6+NK-6=3oYyOV}F{TK6o6x}Or)CE6LyCs|+knKIHU6P3JScu%vys@~oO37gWj{p*7cA|uTG4VQfNae%_;?KBH#|3fZ!REk}#SNR~LoEpJ{OmX<cqF34H(lk`J!+N@>`1{X{NRU*YQi2DBnO92|gHIM(00;8ztvCO+XHr!8&SXawIT*C#S>MW4P#H2N=)#a-GGy=?LqcZJUH7b`+4qUr+0lI=TNa#{v4$+=d90cs0q`0eGaXSlNxJ@6E8vGjAVWC!M)`l?Q+2vh4Lu8!S}%v3B=edvH7#C807S8oVKQ*G%^36FoS!-E1|0?N{ru~XD$8Ste4VQ!Wg*O;XX2mlMjwr#oswMr^@&P6ni05?@Ggm2*kXMPf8Cg_Zl=GJm;`(oJ^T)N16QLbX0qY;BOZLiTHf15vP0!x)XPwNHLycsNpYosKCEOcZUGQHic@3+2|Iqax9G&Z1?AnrW>E)z7I?QGd&x>`5LC3?8COm3U|6^O=coI1wmL3Ok}ftBcoOhZaKg&d(X1vf?`AzAtNL456sIJ9%dB{KseEx2g7n^Oobh=2Y*Z@VT08xLN2jdw!RUP5SzsGaAJ8aT)5ErqH>z9g=C=)o%QzCWr*U|145OiPZSu_>x76@t-__eQ4mbegnVvtD<?36c&H$zwDhsA}zAI@MqBNONq2Zm#qSi+{5Rah;QzH!Ak3|7;{(i$oCrx*I2p)HT0h|#cn$@!&Wym;m4FDCIC~vzj-O`7p<SmoC>LJ?{;NDVd(HWP9O91$(gI7ORzLzb2(`nXc<lc7$ThOcU9BWQFRxn5rG-R-QRd;$OBo=zbj?XjtFeqvdC!Huk#+2}eegyR2<e@A`pfJw0yf}rAud8awX#ijBH${Mgr(Vj=0Lk54Z8oJ<CD5WSt>avynq5}v$re^2FmVEnuA`G=Tst_G9CDxk62(5VqGdsbL*3u_3ogkvhRANY&*iCcdIJpm^r;Kd%HG3NtACfmE#MMrXmMNK6(&}|<zj#9KE8hZf;s6<h8SoWj5T9wk)OA5nDP>ngdd+}g~T9>6z?Lw0x|+!6bY@ZqT2#B=7B3A^2JIa7nE2?^~e$p&UG5RN_d10LsL=lHW<jX<S2`MMa3j2y;O-Ri8Tb0aTK)i>9#(b>LJ@t;R)TU50yBa65>(_UC5R|c5gSv(tPu`Iy}3}%x5}9-}P-tVNAm^72Bl#FCZbBu<}-#`?!=J1<3x#ju=O>LmG*15JOAc;uBD-m?RzUofzj*md7y&<QBk7>;#*Npp+NP;$|}&PkRbeqm!AdOMxWcF-Q&b>VF5g*<YfsxE|HZ5wDB&Dw!aQ3ZGhG>UpcAf#tHo9F0YRT)!7qQq{w*1jr*NIsFtOX-Q)N@H2SnJ$-(U0{<xAAG;@E2806Has%V7DOfVipVOtsL!2PMly2E)VAdf4kSF(5PLCwRkrfFiX=jSL&Vk!qt}0w`>s_8ks&He&{Zn4cVm2C2V@qx+4Lm)6E3s!NX6Plm1F)8w@Q7Z15S^kYnpVnVkx07kg<99!m_eK*CHfcrecR;91CJ148}_4i>Vms#Kv@WCn~me^<O%je*Oj$xr#9A-mX?u~=_=G_jFY6FXbDrlORAvjYdZGk!#Ba9;Jd$?eyYZSLvX}=Yq<P_wSi!3krBdpA+0LmXw<30`n^?>>7g`0j>)`P-L0KzoVC*r3kx~*jO@mhZlJGY%$P!qK-a2}S9$EJ(p{@kNbpVb<eb{}=49v++c*lKR_U5{Rt~zJe&JY1(_e*ok-p)K&!q^Y;rXk=Sjd4MOzu9+TydHBlBEgy4Zy(T=3UmlFT4*qX2HyRKfC;*mqvG<5k=N6#^<_Co4XhCiH2UEvWjy`?f95$O`!drVgT59wrUZeHzpfa=<IPw@tEq|Wa0B<%YO4fSOUn;pA(u%`v`OKwJveuf~lpCrsWBSnl1I(0|EX=Xd0_Oj5-kMH}Jr?4KIxPw&yzukZQQ=Gv}TU_J#Qq4I*(k+@++Lc+}T8ulaG^i4=PFSRG*sx^PW6IJ%5#Z3$Ln>jIfxZrW!NAtJx9xm<z6gjma%jKbt6NFSQ_l)y<1_12dI-7S#0b1q&{%lZWlTZBK}2D!ZW*8}w1f3PnxC@mrpPXWAAHp<=8{rIHa9Ih!}ZRX_5EV9=^_}@s^-daEpU6?ug{f@8}pm}ZP;xi^Fez_9cArCEhz|BI!k{1wSahikn-;D;GT^t(YsqbNx9LxFyHxlK<urxrf)J_*{pI;J##?P%Rkl+GRp^+}y3uCfqK5Y%aiK=d8#1+XZ;a`qxrVikRr@6*dpSWQS+Y3{W4s0`O^m8#F%_6NNmClU)?^Pl85c;{YkUP#&V>y6HC5qRTc5vA1QHX@LHO$?U06RX##=z_xMuK<^#+UcR)#y&PEz5cTK<VGoR2QH)jOJ9;USt{WF9;`?*5$&$KMAu?SV3%{v+7H5SZoROAv$fi3GJ9m*cM0nT2Vx>gblu$LFMgC{*!Ev^jEdOsYiGkZ*tD|wKGoF)DJv@PR2>DE;DY>x1ME^7PWzPZo<6;1<oP%)Wd@f{MqNo+10NiVn9AyLJAF6+YzCPW79gRdc>C{@PML)TpI+5&t^7Xpfm&hkAj-@#EKjAdEmDLFd{Cz$F*Pz1o*c8G$H`$C6ar^`pvD=a$$@!c1!fLXy&L9utO+R$+6eE5bmP*O|Nfrzna^3qFJmh+>#Tc*<7l`3?X7<Qdsbb3JBmI^V4tZ*UAUQWHcwSJBs8$0Ywf2QE+#^J4@A=Wc2lC5g))_AJbAZ>Y&P)+pc9zDds-4%3*T9cb2HS!Vw_?r7Q1cHn&fwM$IRf4}6ZwQ=}qQW;exRY1pp|xolvoTC48rRyk?-CDXUU0;%nea2WLZ)Z&;9FGnf)vZ0?mva30|GGUC0?@sb_ocs47XYLxXG47e#HO3K#BRp2m-st>RA{v!Apb)fpvwIs&&FM}o+p?Yq4?K4YDO?TyhkX&ly-cqb{nW$u@5U{kAvBkZv0iUo)ETdT&;u1KL=p_kOE*FiO8PWel{SK{hVdK`ygdjX7ws0B8wV9x&nWvb9GgAyvQ+X)q`SO@i!-uc)|cH()i#U>2I&`39=EUj_=jY!LkG-^^c5^Am43Aax5ZdVMh)?$JWT)(7E&lov=_-yW_+1pD}GVlIqDx5esT|kEvGWWX5LX>!-F>l0>=!29h7}C5i0TYbY^xuA?>xS{98xN;BOeM#4(ETo^KSYTJR-W1H`(EF&rB?KKcMly>=WoZx92oht9Al5+evnh!@!Qn~p1bsE-~qzO3@qnymH^qWe;iTv?WK4xCidPN~5x@NpwclH}G6DU`<-oqn`N0QSqMGd+hKu0r`W1pAc++3L$3p{!vZFvIw=Vp;1NH6LgT@pw%`(d=oZ-@t@ZbYkcrElC)H`IcV`mx`V9tKR`XQ0vUDhpyGm<7-eyv!a!7L68G?7eGDQkChTF(%L@?fD2AJJULihLcCmK`u@WDqrEZ+<1MXk9qxXb$FU03omSbwWm%E6Qe(3#+LVCT5kKxfcq-jUly0Cb(P;R#*i`*;LGz5xMY2(Nh~c-eEYupX^}?$JK8(zjqA6z#bVY*HDL9`wrpG=JBs_2P{#<xkB;IX>3Ew=X!i3|`z#P4+#y?kO-`@4cpGx_7=ya9#=!CkqkGZ13(B6q>R!!K^^fRRN{S~I<>!20x-@(`5tmemJl$bIslZmC{4K7e<3Zh&1oJ0JHnCLiFZ7_p517hX#Mcx~?nPFlSDsrj)^CwuD&$Ry94*`rdEJ&H-9$#nAp9UeQv|1YBZ+no;_dLDZg}e>Bl}9A&fm!Oitz{0wq}LRh+!m{QL0T0Br(DHN{A$CYCqg=IpPrO~gpOFEG}CvC?5>~x>+bH3KbN?WhrgaIYln2qqE!2f{RmMmG;Py_qc9wB9E-&IUC^6GEhUZT%ZQ8Bph0{Wm8s53hvGm+BBjmY8RoHBVR$MuQG|mS-g!;ptWDWC`*3QhsAtj#f}ce)<q|)=D4KUg^MvrzDaXBk62Jix9uS(9hA0|BCjM;PC|cOaEu=X+EDDS?Lm1gkRoAKAG1e1G&<`40eV#m_+%^a>c(?Jtf|w2ZIgs?PIZ2g|vuRmW$}-?z0(7RQqRXC`2Vyo2juEVrWg~Kx<9v5gLHfr;7#yooKMy?Erqr{_ao~8*%8^zVHwIyE!9k{S+i-2TL%nyzAo2Kxye0tbRz&jXRc)lZ@)_mgQ9xYV!$skRiP6}NXy*!o%IzqjQ<=C#(FY@TOP6o95I+BdEDHa^Q4{m96j6ji&};FRo@ecOXmZF@g>XV8=2<Ii?Z9MVWq)>%$lbbHs%h*ha{H79-80a2jabe60bkWV&%0wb#<4J0{eWJ2Z{Mew((RGyq;7YXw_gvghDpMM1+Y|0{uD|wL;k2l2pR6QN|9Q8;0|*~DcKnBS+C}-+tJAf*5j=g$YG!%j)GF~C-aco6Pj{d3ZMZZXHN+k%M#}OSRt-mcUAlLZ+r_}CrE@aaEX%4aKRjhSP^;C(<x8UZ5i*bD9{rl%?=zIZM|o3t~oNsCn6`?X!M)P4TLrmT+&o!*$DPkS0TTnr$uYSRRLLN2?Tj$3RN&u5B&~T{&YAC>oK}Zw<KM-LBeZ4)KiSiqzdihjb!o-g><4xu9du5tG}S^(}$UP1I#vk4B;~_B;up?(m~N=JNCQO*`wkrW~B})-d}Ctd!W+>wCcSUzd_IVKDuNX>Gx9uo9t|kd!|)Dg%_!smG`Gr@H{2}^I93Si<{=Y)(4uKxS$7f@k59aQimgJFC5<FFRU7%R-6_|Rf2;e2$VK>hxp9`VT6Acr^&h_O*s2Krkv9TA^Q&^k`&PqGKvUd|N6YAL4AjC2$63vRI$pUeS1MbYo#6}7H&%6<cO@h${fZBGSzP9>fjX*y}>Gq63-@&@xq;pYHKUbAZ`KOa#3S)Qc6p?W$~MpSb{{V8z~_{v1qUP@u~arxvina1>!(v(_PzuV{}34UfvpEhYGus&QxZAbj*&P1%(x`E)tszdsbjNHQWJbL3+|NQGY^LPd0`Xuh=XF&^-xKz>!tfVO(|DL*TH5hdnc{nZ^PbK$5j5mkQ6bC;tIB!~-Id^sbin$GOUzq(K36x;H!Thc)}#mTewDFruNr?<6;o=QCN30@m8jUl*kS7A>ggKiH@25ioIyMMGDI*6}lem;<eH+Cu7<gnc?DoTGqC_Tvq*0Si+i82+l)OykI^5qe9*pvAS>P!lF=!>9+E8arM+_*i%j>o9J#MTuLqm@vVjdn?-Uv6`EN&Q;!^Wzq;4*f!&-L{7p02YWZN&k)oLE=zI#PpZs}sGi6F(*cd$HcGM$QjrB4QzKJ?!h#mC((EvRr>SzBi6Whe^4+31z_e1IYm6jdV(1Q&F`N?*CAzF25cp!`5hz?ztIa}_1aA$fh=xAiQ_MOCgI~+raVxfNTO>aVVg?>r#K#pEke=m#u$X#w1__0)qy+l6^nx7>9uqVpQ`pCY8}>QuNA4}#pG-p+S7{pJU9gB}p{Z&_lM<_xLE?jRlqq*0Na&<QBd&#bmkCoA(}M=Wcd&K8oeogl%tri%G=cF)I6$g@9{E3{(#3~|YBk%7pJ9Ol`>GGw%Jxhb#0Tt+0X<GpEV5}XX^&QrG{R)LR_xo_kU}sIxIg40wmJWayHj7^(t}tBiY?MG^k9(%vMm+dY+AB>`X+a}p2%Od?A2@eF-eIMwTSa#2T#1VKR^Ehs-@==k0%wiMQPxq3k29Z;=cYjyx{Nw?omGf+c{Jl7}nE^EHw%t_KrUod`JqIkD~jeohRmxGT*>FZmOVSsg~jqGx}Ya7!1n_mD4Vat40>car;A@b0L;A^q8!IWup#K?*AuY?z-s*rJJ(VW*nfSGGV4vv&$JQ$L{Q=cXxJ|X&qw*UD$@Vjq+-W`VTWNQ?}-WfrhsrAZlUmmoB`z{H~6sE#kq}@FK`#Gog2&dk9U%)b>UX=wq5=07LUVS?~R+h%s9@;+|~ZJQ}pg(QK}s8sF>Gc*d@S7V1=-D;lr5bG<!_1@U14Y9NR4*?+x1?rzM_&6XHaLxUBl+Kk2@laM4s=8N4rbx#cwG%lQKE1QlkFP6PCIS+G_CQuLUh{t^%(V4w1hmy;aGf2Hs3}Zy?2a6^XnL$pFg1H8Bx}wCZIa%z@Cv-zBP*#&jGS4339{fM4y-p0Eo~+2_`^6|_y8S(nu)jL?e3<D`0q}3UXBsQ!|I|wXgi1mZxg>BiP`DH|2~hhvT%4$sVV5rb!5O}`h*zT*;W!H*wG;I|+7HQ&gY;XslT_6fqq`fit0DrPUg`#W#*?!q_?<89$%)^Hy>#n70Ar%<bQGk<G(4BtC$Ec9_EBO<<bxhJ)9Po6Lro5U>p;5D<5CUkkh-cVv>jrHtJS3P&@|fb_87dLAF(h=lWR9zcTcfV<j$dy>isyT7%b+7LR!uOHExtcPLktaFvf*=Dj>b1I<Y;UV>;3aZT+Ir#nhGDV8oBl4?1HeVJ|Q`2iwh*p0wmh6;7>OdGaNLs=763+$zex8|*@sZxHNa<c;_E8W&+ek#!x%MZ>N}_?7GX^qo#ej0<FWC;@;@Cip&?fy7m-*kGXDW|T|lZz!lXKB{R~WU<VcwEx67-lSQK@SO87`LdJIojh-O)j4$}Ptj_Qy3quhhSbHRsxS@=45fu!s?M0(L*oM14~8&<pWiVtBLB#l7J3$Y<2z%_&z$z`a^I6MyBBBkHEgR)x_#TGW6ry_SZ}Y`J8n~XAf@MVUgx_XQCotQ8wO3Y{Rvgh{7DO(xj7mkkT`%EfCI+u^wwBNl4wvUfHeDmlpe2;Zf7Ktix35Z#Ij?YKC-Ym)*rA0EDu*J9&=O-Cz6mSr9-24xsMr2e1bC9V?Y&c*MyR6gL^yCG}0Ie91iQ5y3X~LLR3UP<OvasYvqde#2{=%$4xw^NXy0S$0!R~35E?LdDkEI0t5IV5ROU%x~vYlD1@=*(5tx}#0$NeC6rsZHD6Fq*y8!T^-i~hBD;WoOky44Gzv=BJ&lC1aJvJw;e#Kj3@Gs0Gm@addq)XpGsv2N$kr(Xr)ak3s6N>5u;xP8aJM@EkLTj*t(5Y3$z!ObTb&E8?Wg;Wtt8S}1ik`WTCQ~r0VrQDfug?X!C<Bt`Xjxn!_^eist<9YXD}uO6yq=h@e*M!Mvc$hm(qap^_v^}-|}dnlc+6EZVxdDO8$@aoG<0g55%^|{ZmV-{o!_pA>D5x=BYHcf*J~JB?%yCeM`hG%Oe2Pz@z7wE-FoI@4NH!&Yf?-k67~ike+utxJ&kB@U}N*<H6`(Q=yXQit?`ihrL^%Yao~IQV0@_@Q8Bt+Op?;H^{VLJUIStn~YBWAIO!Vq0tf17wk$Mnp;#vR{c+vt#l2gddxrRFvYoUx#;f8_y}IXR5pC{I>SZU_a`(aHm)*!Ulw~HHVOi&=zHo=I=(T)suvj3IQ*?$-tF9qCz1vH0jxdesNie9r5xz7*J9qH4Y1Sqed%+MwkKrK#a#vV-2alm5S1B&PWSO9LBT-?MS`-34!56<AyR&B&xGGLrxV@e)t3=-ii&!Cjc;S=`B$^{_E`jwi0nQ1LmY0lPPne2S4V^SQObr)5r&ku@2o4Ftq=^#6|R7Yid#7DG%n*19Bb@$$qKH!;VNVM!uSn|=R$P#-0aM`JV1ycq2wiN{<Ib~lF)U0*bsVwZ;Lq@YeyP;eA@tfu=NDiXKf^$s)Q3^%L<?5)2o@0k<iKk5HX1H^Fsacj{l&s0;>5sp+iI%v6n4f9nR({bqx7R(K1_T#S=7sb#06gE31cgW1RsQSGrB+?Ziwa0MTJ9xQ`F?ZUzz#@--fzZHepxYrUgJmLU<U)hW@y5PIp3P@`6<w)ZX%OGW!Izxzv5_o%zCZ&rZC{jPvLtFYN+M|paJ(7?$YeSlm@dm5XNX{PutZ1J?lFK(a!%wz^u{+r}gXm^?pEi+Pq-PlOL*Wq<`h-R!Xa<aA_1+^gAe32eJ9$mUdLGPckBFsvDa!{Nt*z9+4XMa8@`87hePBU5kDcR!}3`O?ODh^Vb<oe;f-;RAvvu$#PuItc;w*s&({_cRF*bGe|gCYaCeUZgwMZ6PG?3Gh}DgLKudG^RT!>_WnU)bra;JqRx^s|_NjevAPZSBG*oXZdnkc<RMb1$|GYZ~(Kpk|5gqvrk^wirnvT|aA8!dET*8G`7#kFu*E)$1^LagtIk$c3nGd>m7-oSp4%HwEM;$h)P~Vew72?Vu5lkM~&4<ZRYbMaD1lcslIMtWY#Q=c}&g{Rs%Ps#XZND#*<VzPT(R<b@Y{<~38ixlR?T3IVqlND*+`f=h=s1k33!Ukzx%DriI&G~fG+5tNY+Zs}a~o5R<-N74LXp$>_X+B*NC8x$9mk@AlJ`Qe<0nw1rNCKefI1Ge@HkI@N`#ESSXd`#{S7oKu|$p(EHoSi>E*H3dG2SI8{6DI^#6~a1lamMOJRq5+Vfraxg9lzxiQTUFIIABdYIcv}hy7JNTIZ=UUIj4!(8nYmD?9s~&hpxPHs33eJ1!MR=LKx9}!sZE@+u|=vOOf)%-iqyE(6%GSWNnNqLTPQ*WGS^(gg^Umi%<AZs6iS0P?_^Tl`bZ#1>~j1WdD}E-?%f>d!imucgnshzcZ)K^0^3X9fAbDPP9)lz?E6KXYmxsl5+<nDrS<LB01_GTbLX^xMddRSA@cSovchw8zO#7#6g}wo2_J8G}|2bjih=Xl9B-+5QrCkgXUn}YhxVMy`^q{Zuj|rM3^B}q^*MNm5SFxRoLBg1Lz=HY8dRb##FQ6^2<e@7X3!8ff=?QfqrtVTT<nfTcA^C;!z`W%C5-7rQ>w^|BCFIP_$p2fN=Ab?wZBsnNgW7`;lcI^gU!y^$+V2n(^doFbin1f~de$l{9ZJeN)8zCGJ)|8|SA@r#t&1*dE0{F8wne2n>dTlPfvC9}}IY?~Yl4ui-N72dnNtu~UpiGX}R1!9GkMS28u!sp>e}V}bt;%})H%jFk=>7+Gz!j@Ca$HB<Dc(h~o#5GMEj^pfNtR)cKmb>)4>^30*I+s3OW85}`a`^SMmK<(ckY<Dtp#i`f7oG1vp&$+w7LvPQkJ2d)kpXNDQ+eMuG_kmm8&xEmG+Sd0@?}T?Xb|;z7drVB6rb`B|)PE;<MtrcRP}*YI++8W^r;!%JtL?4+Pu{A`Z<C?+h4uDlP1n{l0SFTGW&#fpx0$LhwE?ebc1zSpaa#2C4VA89h#Idyu}A(erjaJQ<tUA8^+bdRU<J5dA~ij&2eCV*_}D}s0Xy(sg_<P^2-PaMjtPD;Nvq$CXL)mCs}yR*PHJ-m#P!lUQB-+%D8dqsf}P>9I+57_@SiPk9S1ad^TjHty)D|x5;+|rYJL7*Dalqe4>!;f9W>?>e^<?!Rzya94Zdtz>}rT7KGOPnmV-qCjb0VyzRT%_cE^Qy(z&+;$GJIE<micI47tAOA%I64XyR>x@a>`y21N>a%x#ui%x|%$1VtPeNC5%7G}t!Fl9oxhxO&Uh+Gsp&8p5oU%}p}L?04nix2KPK*5+T5sWuq-a*D)aiM$XB+x2BC$yPa`QH;+DHY#0*30Q0_&jgY>mxATAvxW}kZ@dO;2)Oyv@o1s77=72Vug0n?s7pZ%K~W-;+p|1A8m0y|W1Tz{D3LUNPx3Uw+^Xn_Vcnc!iNb;pH;<<yVx+-`u;@E1g-V#7U*~@9ZjV&WrU1TDRw0idz0^uYkXUvCQ#w((KVMjQ5OvamnoHQR(`1V-oTInG$u($+zVf*OF}L?9KptAtFEe9W_VCj7a9ehBRr(EdEr%Y9Y?B&u*=nP=g7JNI#c09|#0P5@e=NXYLe=o*d&}=RHc+$8%ulD6N35X(@pi&y=S+3&#c`8AQs>*FKLTKkkbu42l^FpF`m`{(Vb$bjGSSo<L;r!maWm7Hm^&iB{*7C!xT5M8&@N#x#14gH3V!wIIoi2QRp9%g5+Xe}knqZ{bsVSBG9G^mJYrKsGs7)oF^h(uZQZJML$ET{#&T!{ADUAV%%9*)Qy>Uou#3m9AociV_OO8p5H@@%j@c6KCl#89M`&H@?V;;Gpiz!TVOorJx6mc1R+=;PJ?VsOmps}5mx{@gftIf=e<D+2&IHgsp$wTsIpx2~?5`-Psh;GW4iH5Ekfgvk841b=_r@vG>1y~3(-#WxH+K3Z<;k{0UPCD2R{@UovcUAR`SWi0KI$3Jz)pHNF(r+Pj2pOpH=w?8iPvgYWh^yb_p&v>N3Y-@bi;3w%||Yc4(kv$<cQ9CZVW%M_nl1A!htcygi~o!t`Ne?6@qRjuBUeSV*reE4nt^Lw$|>}0Zu9WS|0MH=0*MRDJKq}SVx%nDt)FayNlib#o#a<lgop5KcR*nbxeuJ&f-oU>fEZm7JmDYFW@h@GC#(>OiC<9qi_V4PW@7&kDC48I_pa+7t7B0$;gs)zC5b`ytYKu1<6gLp@nbejlwOyyHP+uN?N0K3e;I03bSA6N2ga;d^hmK2+qLEISO)b?uB>UodYNt)01G`NG<+sDT~wI&k{}ldd#TL?^&K(jd$M7Y`XnDUd+>_7$#inj*T@@4j|cqRI5ds`*yc1wc<B5)`@JkPjM>Pi-f|97t>Go47!@tpV!t+=Y<y+*8^`0kCACO(96Bl{t+hym{GT}EUd$f1s$?m`7on33|MiFMymhFomGCNZSIB{XUtZ@o4C7<e8u1{1JBdi($rMZr+1KWZZ=mLB)@EkNn&3Q&RicE+8CcjoT3stYbjipX;N)mO;h&>4`1bbv_|Z2l9vPrDiR=zMxX9iw}+?|#0pQ5E>Fax681&AM$2j!!UjL-{yDV0#NA*={Rs5>Y(0^M-U#1fb5e&l7eEk%H!F17NpabSO%Iwt%o<yvo0PF$A)_=oK#wjM>{)$?90ng)w0Gm%rB%@)E`Gy8u)<|U3$I=yw3&d(9y!x<ocOOCmd{#>t?nQx8rfYyKDE&J@Pq0z=M_BtpE?`WF&d(2cNC?o#>-zqs5Jsv2dN{~g(aD+tzNBF#roO1pX-<@#vqNh;g}trJdG-Jfbg3Nx!9Zp3o1YLkF~bF=vG2<;_JLLEVZ5SWXU)L@M|q6W7o60c!V_R_dq1wpllCGEHxK%kBX5zjQ&5IQqPaYLX1mIfTdH#WC!@nxnNnTN&)X?gwY|=>F9+=bsG2I3$mvi0l|zx-bS;jy7N~5{_h_kgmsk4+`<<zll7iPv-DJB;yU1D-Z31uX8F#KnEB{^$kSy{>f``f{64!n)n>0enOt58&F|bnx18-Hdg#{nb5TxGqi6r%u`aDN{O(?)vPLqRiyzeQ-xX4-<?b>sngYXT0?@QH1suZ`Nl!hd5FUlm5~)M`o@02xMvU9O4dq<}X?ohvBF3#aMW+s8Nm-jO<3;>c2*FHe5QoHjxWtiPa7RJ~IZ)(B?QmtkZhU$7{Kl!5PaKNxn)vz^rpa;Q9@^KhgJ16CNM-sA@NRn!Phu*Xwk87zJUhWSjNDUL(&gS5{S=+pw2M=M`o~JlsOMjo9)Kb`Ng~IbK}2VE<fMfL@g)@<54&o`BOr65tOus%G-EB*I^dFr-TKDk{}9v0oa(p@AH3_aAc;~mebV(*?$T7OesA$2V@}M}$FlHjHs*cR&1YqX@qR4HbuAN%7K4p}S8XZ+o;h=*Q}&FLj$NytzsS*0P;zUS92dXMqOSI%V)7=3k;Ivq)cVEMoFs^1gEZfikqIqd7F+?t_3Ve$zkexC2aoM+D=lAKMGU8crpX+}%mHjQuJJ(PH3B5je3{a)2_`4GA<;@g-7H6=BEN}OW}T+yt5sJiyf*TR1Glxm8}Sw0lI$6)Ca_l77uHAZU^^)VlXXl<WCH&Wjs~Xle$?x@UFy<&4%Q|V^iveCn3bJdf%Klc<}2GbXT4dUe(%5)x-1*(rZ`rCo++9y>xcq*`;5vE_S9=V9r+*0g5k2-TURVtpO1)TQu?e0CbR=WFwyNwt5GE3Yw;{A8RxA>H<zjWfCoqlxbR!KCorj$RCVSGu65><d53)CX7;fl?_!UF&stCvpDfyyS>tx5DG>*$==uR7Xu-{CFe0e2ngK{CwA4Fj){wj8G&P5Pz6ajc5PlKKkicl6!o#$%##2GtqdT>PVyB5;Q`P^5Mj%aeyk6^GVCuE_bmh-DmNelPntG^-b8DhGqzTO^uiyIsmu?giXps1$U+nM6?Wy$3;L6g!%6!f0aq(NaATj-sdA!iVad^b>TSdNOr!`bolFUHJLEC*i2Dn9<*D=tkHh+f#iEbRBLS)<<9XN%({EasY4)tpWTmJ=WJIFHa;Em_IbP;B8E%{{!j9llRSrPj$?^v`8zy}ew{#M39*4ar>02VR%ow#c3dh>YC5I>J&SIYR2H!5jI5w>%!-1~feeW-_H7uDA<f2;5m<s8`Gi^nc0M*zYJb?Irfv&$EjzgnzH5|U{#V85L23#|ACp3!_T=mP+lOdG9_t({7ic^<mk1+m$!Bt^ruz5Ef-?)<wHWaf}Y<*o6N45aZ#p9l{rO2ho@2{|-;dS8MWI%3kP`vC0b?DTx}V=8#;z&%&rDrUK=s7zY{XYU~+$!)IZ-O+2f(nM87pfGPEi#Nm`8&5_aEv}T^N-2+l#3!bZM67COp{s`gEi*;OvnKU%3|iqu18+f0LkFXp7|)3{r4&C<rn00g#;9yv_p$xrekumwop;cTE}0hkcqMO=E2Y9jFO`4H7+Rv(rp~a#n?K>7hZvTk)*PPRT9lvGH=+kf>*+!zq8FwK>zZMNkjALp`@~}Lkav)sN$i88g*s>kkGnCCb&boDL?8Lk<pH*tmjm?Q<-bzc_5p!Z+BD+qO7k*VkOT#}yxlpPPv$`vKz#$Tv2ww(PTrP!{Yl+7w}7Hac!H0Wex!<_@DK9d&_9UvU#q91%~kN95mUL$dLz)P)&;fcq+T2uCwxCmc|(YgA%m}}#44j0nqH`8&l5|zD6E3%KE|QEOco8IR7R;cfn5w1%qfHKN0VYpkr={o@{3E#yFG=tE!7$zsb?K#`kl5-g~!pN{%8(_RBNgY*V5j5AG&mD&z&lIyk6wA_HbbUeSB;8iJ4bZ&wiY}9XjGwyl{XErgH^dA@%nFoj6+5h{-sKwF@xf(lZ|X<<!9tPK416AkW?3_9f{$Y7)id?7!``xkV-KG2qD;gJtJvv|Y}`yy<nA*19b{sA<b<KxV)>0i3@xFt+Tn{qU}zm_G^vH4GU>{Q#u3l4pYwrLWmnP;QD=WruK9#r}XxgV9yP9t(GP5yS%5I5*Lg3=I{}msF3W+O4rh!E7jmQU{%K9gq<Z`S#Bk5Qp$)WAOujKfa4bG-h0WKo`nRDpYJE=G)D$SjI+&?=*7XR&&gQ!Po?>`<isv+u_wUQqMiP{p(ZK>C*EwK^!*Da*>T~v?)+gtoSweR?9j)zbtR@ZXJ{{65?wXkuq{9a<l_Y86^P^sY<`C|0-yTrF$ux>dMGiJxUe}pyoEsB~VwF@0ti6bN9*rKkIVH|2{9EbQw#`B6JTXOZMJ*qHw*v0?&%!g?hZdS%EHsft!^xBCK}=liIF3XJ|2&!>)z3+Y^^5F~Y79w0D2R46#JRIxzL|Qj@SkW_7@l&zh+*6eAC>QT4=L2m_`{%v(#sl{d4~IeyYPAeTX1J(-#B6yrN7n#M%53<CZ4fi$gn&6b(u(hWjH`R}Vr|0Xz@&T(#1+38_?u>;>PgSgQL^Jq$Z8dHP!=R>W^LK#^+mdzt?X3TMADAzWQ-n=!5)OQZH>Rz%hn)D<#8&Pz=LN<(SpK0V6iO|kDuVCRB34o9Bvg|9~ETXmTz^Gw9!GrUxuGX*DI(2U#c>DOonZEy5eNn1PmcAVv*EEgE4sPoW%EprbXhu_R6xX%@_!bzI>vrzEIkr?~Y5YvfCyct<-x<Os?eSIg`Z#ed>L^tNwm}F3PHOxz4bdwrGHA)1pT}+0HNMldvufY9&DG*`LE;Ie|BdORv$X<#k1d|m*1BQ#(}%_F%+_!6SsGgZIluoq1=^v!<c?hA@;GdPNN4*_@n^TfCuFq)@5J7kxC0c1uR(cK<fMXUhe6hmSRO6MoH>jggm9g6QEE_(H#n|7cSI9}b2#~fO|$~x^W~ZD(>>nfWfYUN=hKM93l)yo$4i30^Vl1bDlWxXb5970))V7lg?Dhwi-471g5c!&j$O!Jl65uEE9GWEpYRon#h9liN1dDad;WDfO%YJ#+94Mmj$R<+H6DcvIXAPBQr7(AXL+JLH#Gj6T=)qv_Pr{obPw$+l~sr)N^_y-5LY{9KWx5Zfhtrc>Q{`rwit(Ga8m=7hKl)3ymdJq2o@)dz6nw&s*Lv)H1=fBF!$+c3|?}4nnfbyY)5$GyUg^ms3`O$8!xpov11KWTCwPK4P4|fYUI1f2;xcLD9QWV7J?yl2b7%P@jB(P&=k<EwMZDJK#p6I^C?732UqhS$<?MSsQTS=!x|(@>~TIPArb?xH(MF97ro4uqof^D_%Lo)8e@~+Ez~@)JU1gJNF@QSxP@p@(6FXc<YY%7{QJP?AdKxRI?0<cfkyz~Ns(&Lrn4EXzr8<&kDj3ok{UVV*U%Pc%Eog9+vYL1UaK(ylWrwAFAkkgO65oy;(=(h)hHIW^BT2`24kXik98xUm44!SX)bz=>$#Cyl>g@+JUY8Qre1ktjq=Km%l)Ey{FGkwB%eS81X!-h=xaeCx0IBZ)RUXz@LYjQF1m#TUo3QQ(5l~QSCKDznAY0giM{8C24{@CqBclZ#FSLqP$D=QHF!`CjNewQK4Oih;$VbDYE8c4m~a$3AvnL67&5hk_!Q%q*lp{ZElr_F4lvnH2^hHLRy@9?|5^7KeiLi4(oZk%SLOI{4(5q`SX0*Jsvv_ry@~Z~>zt#~O}{J4K5Sn6L?k4F%sNYWW^7;lk%GrbdLb+K6Y1b?2{te!W3cXR%AXP#O(+-RbZx*(yy{*8XLe_~x^?@p%Ef9b{=aNszI#YDA}qkaCVx=BQ<?!5eVdAK#x`KL62L3?3fXgyuE*(D=|&H@4PP^13J>^`zF0al3Z$JP(*)Q4m^R`PZ`A@#eFM?$Cn^T843|}M)vGWYtkUWg!ulF|LV0%>2F0hmxcPy?h(wyXIrYuYVUNiPg@3lD`waHT%{qCv%Uca$KP5w6t6Tyi%092-Y(<?3eG}S9OBy#V;4ZG8C&gb78<-z|Fd}y7<LI$A2I^5CWd1I}M2Nb>|8*f#VZe#_7*#iZt)6(y^JC6lB@k8YX#v($)3NQFWLX|72X{T$&gvts*f&!wBu{cB;6$H6@ye5N&Y@lpym}`)wI?`Fd3EZiqytTBOBFRAebH-7H*=M7pL$Vii#mOp2`kOW7cY*#qNmNvP6eE?q+~*0+brqwb)T~qy^$ANMpr5wzv;XJXkYVrnzPW?4>oUbdH%nRnModzFgmwMHz$N^IyN->XLLhDy+}C%jf1ZV%pmD+9uy3y`KAHgIR)QA1cGm#N5;dni1v3;C4|qnX`wiqCcx|0Z%?&yl5v;A^pH_Dpi6RJ23VFor~Q0G7)Q`8;sR0Rh=f3A1Plk}z%P1e^<;_6B!7X2EN&-|Z+cvNVByf$T_!ZOJgH-(vwHP3AU)dV1?NwT|9R)^u7(KM{+Mw%rP843>v+RX(=6*)b8(P#3|nx2jqrWqP~!Me6M|*su)hMLD2!G$Uj=o}g?Zwpfn=I7(C1sq-kJZ%b)(tGUV(H=+<ACfe^o<8&lPfjoe1e-4{G4{?`2gD;Y;lp=HanUw8x<g_1y>dx796u{wfK{`O*(|v<9~^vjYo$G^ZSw4Ml*gfDH@)Z}RPx+Ku!*nQ@I~23RF#H4vYY(x07f>Mv<Xt6{R6<}!3$4<x?VMw?DvMRsHk`=Y;=g69(l+ntg#rncK3G|5OON75z%leba%NqB0t&ppAh{B=@q1-L{(X-$|g{tws4(;%1CZLwCaNs~M@P|UU1>7lnKb!)X@P(LAQzzg;JG4S`_J*F&UuQsf6JzvDZyMJ%17;1>+<c1PEa9Kf^nStIoRIE_6yN91{yz1H!Bl^q16X($YJsG*&$_ha<oW7)@(XPuM)eQ;T?x!JdE!}#`&Tw_Zrr?8h+Y}Cs=>-R?te4Ls^gHBq3q0TXQxj+0DKrqwq~j=eGl3)sQODzaOs`)wA_MhU^zUEfRMvxxC$lR3RyR6<r_#hUdGUp|pX&XzyAq8gzV?$4WloMY)QLj@nhi%uZ|LODp%hS~7$T)(7M_`Q!I*ay2NTKcFarc3yY$8_RBq)&a}8kiK`CXtVB4ontXRK;?C5K!{E981Zk3AiW3&L2QD*(0`C>4d9l}?;JE0QGznur4%hUZ9FX8&l9K+ji60(e~)?WXoerdh0@a=}Qw?z(hhkI!pNGh|vv#`4uA>}R7CAbH&bh!Or^Fv!T9Ud#cbrFWu^mhQyF#DvGJuCGV5nFx3&Um>k51aTX7^*M*y0k_0Sf4AT1C@33(iRBM_1qCpQdiakf0IU=SPtQJbDBxUnewgxbHk(P5~;aVJmwAYdv33bS#+f1AizW(3hj0(=WDS{c3gr@mvcB@!CKC*JA-=NGS_~68qm7HM^<aNCUo9UPE`1)zJS<Xue#ho>QtVnFC|WKDsox1siA^d6upn;&VkJKzS0E?`GIMeJvAG($0ww#30hH{d90dY%Y^^PbIUoK2_DP?vGaaW&(>P>MdIF%^S~p6-it(Yr4vCW1&L3}YvtakoKl<)g3}C5&`E1D1Wzo0JK?9DAyHkujMY;3E0Y_q?Ww~t2TipgEJ?wip+Ao&`x|O}sx}}o6^<7$3jcz#%ONDS$nLIQNxV+kz{f_U!-SvlwSK8*5{+;&3xLXl8CA)zhQTio|AWVS@^JpO$pn2Xd)dJ>sh0P(sXj7~!@G77k$-zy^i~}AGoI^Qy$kTMeB0%Xts`freZr>P*YJ5m?&=a=_4ZY|Ks=@dLD8|+?XS&ZY8GD#_F2nvE%)>#n67ks-V6Bns|YRHz$&_N{Bkamb+hy=OTQ5@OBtR+Of<DMwi?g9b~yaQDm`zYn4b`iLF^9P#cg5O9R);)ICDN?|2$k-Y7R~P)Jzy4m$}(A$LN58TA@ktVrf)bnS9fQpW+4|U2-{j^0;zm(z>loHy;JR#p8V1%k`aEGfA^Hm^ic>PQq>?XV;ug3LfT#Jdv|-Mn1;A=xQ#vvdbdFaVP(yKY0Yvg1${1G9@IkN%3bN%68eIk20>A0tx8JrqMRGwdz6+P}zkhYFiUXqT_B$Mzz!aIWJ+~N!sQpHZ}DtuzTz|0zs_Cf~U@2nf^|YN<=FOFrqqV_T-^?)C6CgMWcl}1(kjJN+RvWvwErGmVQ&F`06UVFPV@SphdRYK|sNEy?5sE<yU8F7h+>JYURZ&WGY;UeDbRR#;A+*Hk<)~9$ZXyb|4PlV!_*2f{aYovzg4>#l{u+Rv_&3ChK`ScMPCwWQ02ype|Y|KtT&XmqWsnhX~b=1;J2lBi~f>T-rEO{$dwfCII|$uR>sjGyj9i>y|p$GU@yVeV_eo7i)VbPVU%78S+sCJd*JLIxK4~I`<^J^?*|U5tO8nlQvAP1i9ZpIv^=J2T{rC>Pn4#S)&1i;b|;Pao`ZJxHi`YK~_u{$8#+bMJ6pYTt;FpSiN*tNWk)Bo4?gBT>p;)i;`scq5OM(y%Jli<h1{m&`xhj5|1ipEr)Br1KHf%ut!&!M^ik!efJqj(*>2S-cRN~NVQzAs*fXMC#mrs+W(mM`yJ{z($q2^H{{bq{(sCDX%9`u;ozPLlNOrsoAs9LG#g0(Ch1N6hX3aJ#0s<Tfm;@=;IOR3Aj*Vq`a<o~oqUQ6(xFkX)qBn1<lE4@^2_ywulO*YWnO}Ra&KOC(O+UVlLcpVdai_75Fx+FdI2H_KitT%2@bzLDgi(@#wUmWyqf<Gi*xB<#>?Z0vbM>?+Z1JL3u?uje^}f|Qe72x>$QoU^(#VT1Il>CxUzsEIWLLchP?7yD@IEK%{pz>-nW--p?H-0gz$szApDE9bd85Dv5(V<XdcY3Z5&V|!Z#+7RX*<+kj3V8YUw!u7RW*;3T@nb0EsbN4X=30%t@_o*-DueH%O#V%e`%uSU5pls*=5b0cfQ#eXkYHt~R6GJ3D)(vvt_ROUa`>4G?pS8s=0{n4Q_NTF-h~FT^?GqWKMH>AYtRE=UH-Q)3*g$D$k((1kbv!7MM?IRQ}-52tga|6Hr$U2F4a_q+iCsYbp&_Hb&-2*Zm*;yuOund6YPMz#ZpAjI<)yyu3zh}x^+^0XJnmPeoYp>y@pU>L$Fa6@6(JMX|hADwkK6-B<5R62*k_M<gUIp!ZjS`mgX60?Blj=Df!C^&`yIrPE?__p1i|21r|Caxuy$3-@>T$etH04KP5w}1hO5H88b^;tb=TfE=1$HB3FaG+=6Pzlh$LO;OSKX-Fa;6`Sgz$gn^XAMZ<f>+KhWjM!u1LFtePrg*1m*bWdSQwZO$igm3Q{dQIohxc$eeYZMWF8BpKQx}aIJz74gixr9Xt*{?!!upt-!caG+_7+HSvRZ7S7|iil9wsx`$G@;W@y&Sv(7ezFup*gOD~uV%48>cy-QmuKkowV<UQfzJY93gB2yqI{Mr);tgt8(Jg0uEk%a`PC_x9~zi#p??)S70`}e=I&1HBUdC*2kb_jrExTbfQrGHP#y2B|YWqq@tMWWq`<a9wx#<Y1`xtIENWNmQoD$~Wn*yf#}QYxLk5#{25(Y7<Ex}#C{RLaH;jcU#<En+Y54a9NMd!a`m<O>W4A3=)Ik%~+D+drii1-t4}{+51cjM`s#sAQ1RTGjSwk(_y(m_wh)?gU-hyjhpR%<Z8f%j&qmni=FFHGF==Yb)Mn`7*culzSI;3b&zgIbU;DCSi=OzO+lz7t}Pi1K`%q_=w)L=GdXtj4$Ie^|tg5$jN3wwGi9g(Z_#!Yy$}sQggG&!t>r5u-F#hBAaw6*-sTfmE_D!T}2!^huno;x#=$De22710SZ*tnNkcM5SNFS51m?q7NKtD{Y+f&NOtdnB8LN3=}8+RzP`XDiMa$^PYCOX37}d%)qEAj@d)t(-JlG<pDEOMCPHH&@;`N)NTTC`I%g^g*BLwh1v@Q^U(!JI^?d4>$htGIUE@p!)9?8;AE}6{rwptGLS7>89O*r2rRwPaDruryw4i11M>(e72`T_u<|jJ%RBW@#iJOF|vzu{*u0FXx2l(JVR?Rfmq0AGc{~5R<O9EP<b3$j9?T+RsIZv~T+^M9Pd@60TNG=%+ZSIrRUse<DlEGy1w)Nrs7w+&-k*al7UWPS2+qz|0tqFc5iQIz<H}0WQ2wtH!%AAh)PAhKk!5|)8<(wH@ML%sUk$aXUsX>j!?y6d7&EFQ@mJtB%rTS0EJE37loc>7-Fkn+oNw?ewbFzuJq=K14&6=9mI_~u$zPZo?4Cu&T##06eO>G<^LOk!)1)8^S;fD^ObMSz+`s1fu6`f#&B<Lkb+0diQ)*XaxrIn4pX5N%h!_kA^Q-KrOWFR_@dW*lUCJTTl#F(=gv?7iZO3hD2{j|gtF>3g;xAGSm1AJR&XP*^&Qqaex_a|0FSLO^JlI8JfF`&*7bdI?U!@5CRl6no3KO|NIEudngbD$KQ-so@1(VB!F-{;@O_uTUeVzN`=^3=T2WnZ)=W5$6W8KTw`FlZdU>qh<JU2SQVcJ2*3m5|Nu*s#shHTzgcle6fRSU6c;>moOVj=^}FCT(jK3WTJuKD<n&-J%|$vB*R?o)p_IN`}Xg_h1RLhA1LWlU3_%VC9~LltgvywYgeNeoySxxFhzUsmg`plk6H61pc>jGb(d0CS2>j5a}of><gP*UxJ2Y9h--S)HXVc<OZRMyDcgwbVgu*b;tOVpKMa%XQ&?CG07HFW*b)`+^sArT4~|zvl(299VZSyXzrDs>vs(kG~6_l#_d}nrw`dD8|%MLn&?NN1x?PMSe68kJ^`lwPA8m)a{1lc_T3R?ivC7+iT^>^=q@jpf;%?7UdL=MK-_-4(G|hE^E94olmim^K5GvwirWir6}fV*Ru04;)iyee3S-u<@AEQ8_&389cOz<8Ona3umc>;x^CrzleyBPAaE=KQ6LmKGZ{#N_Wz3iL2wOe&ylN7W{Xxh02gAw?`=39xrSsh{0(s3`e)<~0H`XAtB`!<!zvBqL$y#x3L2QSB)h6blJl|sg#4;y{pafJbNa^nAiYJ-wAN682q{UY0t54*@!?M;c{yWvI`l_~(6TeV4)pDkOOaoV{OS1G3?x|*D42&NY>u`rKRT;|yOkBW1(o@_T&<@?}Frbv|$0H}M?tGUX?3=)VQrF>7OC^IpMlEGu>7@f9ke%+y10T5eFyR|}Fmg!-8Kt?gj+BVjXS<jOSt>LcU=*|@W{zpK!O+k{je;Wdgw7Bt`Z*}PP>3k7G%p?Z9%-41)4-&u2#jp&g%Ucp=)m9}hZmX*;_x5OR!Egy(?S+)YSWDverX$Ln`Fz0-juAd$8Vtiw70@Du)zU;Z?^I*>yP5*b$&Qd0+HM~cQUXT)SUYqgIEvaLJhXBZJZZSMId{_h+0#R78<+dOw57?=j!ZA6r~yGe16MRmlEdV4VUHXQhRH9dxsW~$R7xlj_CFzt)T(Ea_JQB9W#xmgAlb(rU^q582if+*5dOKVEkq5)w5sUY7q11MEwB^d15`;8l6vrKaAnrGOmUq_>I?E6V7fa=09WTgt99mAX}0Cn0DUdb^{Pp;d5qQXRWb0bJGM)0lUhsqZepXz^?1T<siSz`h0g<!bVg0Ipqp(5<i&PDm6eN$K0i61+<I#KPx16I8^GeE>o?0AugaH)Nsd}BO3`||4Etn6JWB0gVPH=1F3~*yJO7)dXh{)wiX)&y+M892IZtp9hJNKB?b|0%5>P!mjKlU8yv-!*47VQomk?HnR*p9A(eNtKL7BfSuW{uYyZXrx1Y0ZY1^qUmnQ+_Rdw1>pdaLR9G1%c024W5QO2rb!*)o74g#Km!XPRC32@3LQ;-|ZmYKZLQ>MFRSz0zJ<Qa`xj&<_cnsit_EE3t&GS=C-@zQKCtB;qFA$r8&ZsW9mW7^~0;~DY^LnzC$*O(lgz0MUj3;PoB$;UxSPW~&(mic={HYyKj_?&K{RnooQC|69|>^kZRrWd0wTy9P8EAAj7i{1jGlq}gevlRp0gFRv)Qb|5K``=a*H21n^ZZMzwO*-W9`mA0ZS-^+OFAnThH49uc3ks%GYFxZsrwm;ahY}KM+7XCGtO$X##>A&lng9RJm5}aUFlrTW;+641(>G<Fg$B(H=~-`+Jf%goQy3_$8WO71tNGf<$X3>tVO*Xq2uMZKXp~3;YzjT-QAw>HCMuWqWO;Fg{wWDA)aB&mFDQLjxr1PB6^{Sh&fW4WWOA1sed7!Y62##JAN5i%RCFhYS)w?R-B+C04kNuz*$(i_*Qu(9UEyqIMAVNP`!ZZhCU?bZO@bdw<sGV3%Y^}LlZs_=NKi@shgxIHU_nkZnp`T1oZ*`f&Z`vK-e;4=(!wK~VkP{>6IP%f5uf7?Y2!}p;2mmGul_|*0GIP^&yf`{4k?Pf>K?<MAx$n<HD~No%j<iRHgV<Q2Sg13_p(wyuX4J9>hs<f={C>%VGg-D)Ht2=w_^S_(vWbSYV(^y*^f9EZK$%!Rq1Hmb=2#s*kCPwgLQQvMfm_BJ(p*%8o|Ry8o8%V`zuJB1fmhyQ~@F*XFEjeF}8>)Ms$6nV9BiRg&G}LvO=$uI?JBGYjgBF`R>*zHnz3;`zqoSOY+chO7%vv$ItFx*(CIzJc6TA1rRrNYuoIP0hI5mRve;&neFIXRTSvD0&)#)r8E}b?DJm_Mz?6qK&qcL86c2rTeYhw14823VntNsew(i-8X5fcZ9#3h+Q~%~sRP*wQ4oomAvx?1huOWZHw-Z|A!OWJS>c9u=@!3ge!irm6@3mQK5JeH_Jj24SYkP(&4#u8+oqH$3DLAtr0$%=zjTkVH1?z(2=D|C^l3b5N5u3M_D`8=E{NlJr~z}B&IatM0eyRC`gp=BXA)JU1e0zgRAI87`Ifn3*(Tt;spvfykzu#$9iL{DNu1BZ#n0g{mH%AeM6{aio+fyyi*_K$T0*0)Nq@(J9C-Olr8rXE)yfZv6c3R6n?@iSzKI>Ik*{PNa}W+t9>tyoy)8Z))Sz$?E#C3}Jx@_+<wyWZnqZ#Lz~ea4ZD0F|KnLZl0)Rh=2TBd_1GUJ3RC!U{b?RIaJfH>y)B;%wR=~w^@CPby@^_$Xj81dAwj^UZVX3|eqi9M<v7u;E=lFn9Zo3ugDeCJQ9SY-F%ArJKDmn>j<H#S>Zvv(6l^kxF6i6AHXAaPUi4vh?j*x_?Tm3}D=boLqb+nd5?AzqrpqEk*my~3JkctLs(*y`ap?2^mw!kzpms4O2!r)KJ_5B&U{ap>DRud6zQNX4MBX_`51+<$39dAVWjTJVVgNnV;52eKsxk(Qkw*r_Y`OaLmdCmyRZseFhQk`9>nxn=D1dI$!7k=IyK!OOcbU04oGBL1D-X?O{pgR#=6rVJ+PRCICZ`TiN;<UG}F&ln83e6^pp%A7e=EHGd^hFsA^XsR;;;q<<+HFJmX?zrKTI}vQ(KpXsG~;`(ouTECs-6KEg8px&CLT;o(iWV0W5b?_`-;MO>%i`m-7XwNoMMqxldN6M^N}5KcV*vcY2Nlv2DUG0SQP%;@2FlW?ZK)U2pl7Nee8u_YF~g;`3ORJtO^oNRTszXUqH?a<#v}x$xSkBU5k)ROH*7WkH6Bs7BXkV(c#pB>J{(0QxjEtE2<ERp$*nZ)ERK65f{%WvJdiqgJ)q%wcxy{1#2i9k8L}~H|htiq((q}Es6`tk3C@gj|QonYrLwW?4V6D97o#t3<ze35Tch-zqBC%49ii`A$@>q7l-QsA1{5l^{BAvyVap#(*eQ07K`omy_~&;;99nfg^H}xru&9i3J$f{<5w=GaPoKC4kyG8mFVXyHk_(`FK^{vZ8c3FT0B)`fwO0NjrXaA11YN#jX4*Vg7Zn(#w5U@P6@N~86zO|aI3@r0hbs4>?&E@c|Dv_C0YyjCc$2h&!(EY5mxD8(0#kRU1!~o+XwjM`otZH$bs^rGH;a^a{Kj<Hu{LmzMjC?nCraivi&=AwP}p3yvH;ES9P?kFm5A7NvxavK2*cmcnhmCVJ?BZ^QNN}&&K5*BWwfK+%GqZjFO&g2ZFlHVfV>SrS5oHQDt&RUR$1QS3{)k=d=ioc@rJFn;0cp?nq#Y>%i3Jy?Qb%r3xf;;rKi+n*F}>=CeUg_ITpL{t38-u!<*q`D1QK@*m9xm)`{>^BWZAtam+QJP{Ux49fAZkR2kS1{&=^PPs7ikf+sB71xAE@PLC{1Ls)giYOU?5LdiYq&w<r+p+*~0zBYpO>J*&<+)7N4BSX`7{x?`H?>P!&iSog>@%nttM8(Hn8_z}btc&~d??4(7u6Ejkla@!(KiENOSAZu9DABxlja7NK)R0TT!Mf@O6xT<HLDdNvDoqdAIPW^K*;(C7Ko@Zp|hv-muBp_g9^1K#Ma?o44Ni_gzaWuxTQJuWp*D&S6jJ|+gcm(&p7K!+XXK}mG!<j9|dI_t2#mGI}$5GOm%tay;HhWLo!yWoA`O~a+8=$!C69`ui3sL%A(%TWQd36rwzRR?M2z58G(y_b`vT>uRLuE4$u*fXWPA*!hDc?&bJ^`D_r|nf?WIo<QKDKXK))H-hBI1iV~qQ>zt+^LrdsR4c{1w?|l;0(_YlRPML8#QM?lpQ-}(-T&1uH=ik#IzB`}&itKRnJ!IYXG=WM$CdFex1DA@V8`SnhAHMloDI}!j37Z&>_E=V!x-p9n5O7Enj>IS=TFhGp`$!lz*pj-(25ZOg4VkM+CRH7#Q$1U$fsrccNs6N9urnloDY&(BtZUZW*Uy!<@g*)2c}#Oe#jn<+4oVJW6OP`+F*kRJ>(NZ&y7&5YLua9EKX7{7`M3h6;%{Z}B(oKpKMBmHF)UM+C`JhwU(YN(TU=2UN@`0_n0|3H4F@!@T{yjurcXaPA(x4@^s(0I+PN0p1CeB>Ko)g%DNn!V9_^>|u)3uE$R)mTpgPLb|L6Ju<pLbqClc3cfYMu}W&aLfobBQvtm<>fX6}njS?$mfY~FkHM--PujW8F*AI&1^LAgw4&bpq3tXX5vE~DX=61N})A)$*BGozsSjI8~nG2g)(6flU?=Z!3j^VeI|-Mfdks^?EG3A5Q)fP=uP4`^5px9^6{PqdOJi*V>09p5k)7zVoJ1ePkQda5NRFCBoAVv#>a*f=Q~Sk*(BlMru~fjdj3?j3)HQx22N?oaipN~R2@MTr}-02V};{$mQhA&8dbi>m@3DoOX%i5A+l+CyCOLcT7ZT;K|DhB9W8p97!QCLF&zkT)_V);595_s%Id;M80!zzx+w*cp#X^dfo~{K5?eurRVU`ZeP)eTe}$>WbC7NqJ^5<Rr}GfMr&i|LR+F?-|*%lK;PTEj(sx4mSY=!VJk13?n0LAS^>JV8?eJ!+}e`;CZqkJ93qy5{<OSjj9akxf!4X6?_54p0oiu*%`Y8_Ba_L@Z=fK<KXV%Z%0vGKNk?Pyyc(l)A22Sxv_gvq`1|p1fwy%1C|cU(HDBI5V944{qYP}j7a1t8nl4^Hd6@oEQhzXLS)Ye?dw~zTQxezRUX3weG-h}**?QXisT~tit?PreEe^(=(_|zMm(SBi9OvT>5uDh&}carG5&=#MRX}}OBI2ZeYm=S#4*H=x3o*9Fi`S4N6jTqBHF%_^Vx?%<xIv2v1<Z#Jp|ao%U1LZpNlN#rf+y9>Oxck{G^rxO+}HbhcIeH8BZR+0xnR&Wxrs#f@yO6Q^pb!`}~1?ilFLsX_WXD9)`Mun0_Wo1SWu%ym-P4A2CqQwRUi8x$fkf^awtT>w1XrfbQY%c|o6@IOo#Zuz-lq$geSr#UqEqy>A{$gDUG|?$?fC`JB*#BO)D4w$#IA1h`hgA;EA`3umOnD&Gx1j;M2750!CTuJ0(RC*@8Yb=i2r-v>r<gaS(Ru?gitq7jXC+<qpcLP&Ofm@&Zb$H>bq#^;)V@x%!&hE{^OSx-Q+hj_Sk3BOc+bffZt^mDY3D;$NndlIoY#0uOBC8>ZXQy4Qek+9|z10`5lSafBQy)lY=eya1Gc<)k_(7go?Wc=bG+N(s{LL1ZZ@%)`z7p3csP&zY@)Hv-ny)gurCvT}yYH3`#oEP%l>4wbduNhUMockTYZj7^>c#mOE3}g+WU=m=;Mpc0&&GUQUAP$IB$F5_r9OB$9ix2PLkiIKDPp5WB^UB$W`5ljc{hUg~;Q`f6XtmXtM)TKR?Qyu@qbPd0=KWh0P!-J1Ig};=Yd=|lDr1<T%t>+LS5XEiH1llv|B|TRioOX3)!)!yYG4TtO(PBWxrW(f2K#GMk>)+0rCW>Z@@%Fks(0i5!8!ey8m(4^5>5AYzcprs8-ttL+aMQV4f6vKuGGl0roQ@%b&9bM;hB#TuQ*nS=)+owyvR<YZT)5QX1Qxe@rgg4HWP1EyjzF>%^HdhKS#4nruP%X4XL!{dN4&wbGNR@l{!2cb&3dV31TG-?KjmQOjUty_382r`yo5R&0)F|sM99_O${{s27!b6TsftqaMzo?d<!qEA0|y?^(IH_`XXUXvQdZ9{vocNOZLJPjLu=D&=4#@lfny2zuK^~y2sxmHMbUJF;mbZ>z7|9^tkc!fVPFv-B<Sl#bOkHS{J=n1ln#?{r-X!*wiuFAW&yPXRG^`)(6Ku?BaRaw(=w?u3qws;ECM;OctiQw$<UL=y0+pm(=zzcYvzgq;63G8!e=Xe;SH*EprY*!#Q~VG;%@;v$f?YGvT!}wwMxQ%kYCFTmfm$fy05upG<vVpJQ?85M@39l#B`ua79V6!L;MyH`wVMYQyc7aO?>gU`ld>4~|q1Pj(KcYc}|5xr#t9H|x)Q6W1_1tEp;9&!fqihvWXn$3U}fO3Wp;rHY*{#AWB2BO0lQgg5?KbIe*lOrIJo9$|GmXYi6dVJh@!(j9uUF&D{c3Ad7kBhYs~q^<k9r&T8hUY-d{29<))_R1}qefdlrhxV*5=C!YcklU?)*M;5q87EkySqR>#B(S`L5{B)=SNj=nE{WFCvMPm62SVAo(`>0S28ryOQ0J;dNUGI!ZlZwH*zpG!eGr)HH0*W@O`(X4Q*1vVP4LD2<oQ-?)G0=A7!hXLI1^mPiS?qBBaLWY3H<BCyimq~Czj2Fnll#JOz%^Q(#TKhR&75;KL5qHym!5u_@8*i|6l*S#9{|KUiE_u9;i$WMq`d2VUGU-B<3X-^Ok0oD)m*%8FP>qa<*<2ypj&FzlXz#+H*`}ju#J?!?%qeM-HlvK5F&Bf|eu0_qg&ozFolkT)@40-25fl?Hx4-+FdZlJbM+9Dsp<`H-H+HX!vKMgdXych(3~1?bYRd$}?VDrCu<;trE)hJP-PGAU%TLZBg+3e%kI-{i&BD0(uR9xCUvga1?xdgdl!En8lKVUF46nFpRKC6fxAiuQ(dT!A$A1$({%{P8C8yp4kYImcpo#j|+eIGk9Fpo2aP_R(GaDN+_=TdP%#$IO>R@FiLF!Cx@iqjxH<p0#-(bgPPqeDqt{gH-3w=ylJX5tC*<Ms(>ai@pWx^hmi*S1uQkVJDW@#=K!CF3>a_z(yiU=@j={qL(Kt#6Gs+fXD%R|@XkKK#vgH3==t{H-<r&KZ?Gv3$1%6>gboIu)Mh7F-Gf!vv<nBdq`MdT*KutJ1iqY69n@<k-Pt;-GU<zA3yWc&LI~k)FBP@J5Uw%MRZnYr)@Yp<iZiGGt+~_f5$~2DxfVV(r5S3Wt>;>VU~^+#>F(zUw>>fHjx1^pZ3ORA6MD<BVg>$Jbnf*>+Tk+*6s}4z)%{HZe3(Lxv(e<)@LK_LJmMt1&buPIfc8lg{Q4Qi()T6+>>Ie;cym4E-_@|bcdX+0Ee{S@o%t@Bx-|xS8Uv{MUhkm1(iMVW`$)TK^5W6Ydo?U;|5q(uPV9=3Sr(S^Eh0&yJG%^D*{u_i{4Ub%%Y-FPhxI<Iaji0re7)zC*?>O{Q%B<za(uDzr;_K)=`6$4cxRUef^GtC_q?)eaL|t(H%(WJ%ujE!7nwe{Y$a$iQUK<X<ozxr>!;#OwMHOWOUf69aR@6Ex0P=^4x<CW&K@jfQ<sL&#&?T)gOy|>QD>ia_%&l}ywvIG5~V^9L3_(d&}j~y0B>Vnk0B<>jFnZ;{u8qZqcQ$P2O;-$T!<+jY;per5iGAW#ajqRxGbNbr=xCUC8|$YyLkqci|u^BHuruQ%7BfmiM=GL)Y42zUVtAg#}PQ##6Q>yc<&Zv548}Oew>c1c5S2E)@YW~)v~dN;+{VU1eRQ`PlpmkTlx$3!p#1d!wiDWoKk<9Lo5M5-S=TsommQcKa9B&j#Q}pi-gukWWw6~Qdejn?`~(ZjUt2tDhV^B<xp1IOk+utBb-S0>5sEPmpp1acch(j84q-;+j+8QI|k|2fl&)w*5XgIS@MK>KYzb&IKbfI$cAAn)CM2uB9&G`5fUV3D+*^te|pqYt%v|vWOE4`m#Tw%>-2RF8G*J13sQUhR{dev#KHXF575jC)6zOsi?4JA9OL?WTR81l?(d{V97&<YyRWozmCoXsf%j@407;a`#{!YNUl4eSWpN$&O=Klw`53gmgjzV=r5Ha7)ipYc+r5ljA__)Fa|0994-dEg5Q44%G|fPY?SB~cc@rr?l|1JX8}$@~x}PakRal{1d=iQ2{|%u^J}Af{01|_29kmPtqJRneVD_pU3QalGiv2Lm%@_)jEV+jG5-Rdr-ht=?H6zzLPObAn0F5JQGv~8o^h7b#aI?nGkMT_J;IS$H6d?CEQBgLtkIq4E12~=&=Hd=~R?Vf4Am;jhQFmK)cv;JqXI@hVSO7>)pQUF%Qc=L+AGTzGBmq9FFvFDCn?sP*+W?d*EG&Xf_{yBo#s&9}z><Rg_>y`bF<TdWzI-w%*ry+B4hUvVvSjP^sJ7e?i~Nr2^?-=J(*uQeNkpP`Tt;UXF~>%3ui*W<mFptBsd94+-@)U|+gV07eKNa;-R#*6*Y@HmjWi4|gpET8Fg2qecsRnwO4*85uTOZXEh^x7F&%TRzK!O$$g2=LqW_Q@2_BCj=M87Orl;}da9p`w%`du8Kj!~>9_fsFFTOXv0{zkL2o{4f%E>#@Fk{j9-oX*MJP->shyH6_g2ccL(i@rVpCf|7dURXnr{J912<MAwKN}R-R}7MYkPN1-Fa96EV|(>g)(Hqp8_$Rw(_Ve1=O?WfFXLci#}j$blirteq>EEa74~=)pB61{K}r=h^`yDV84?d5WwFWJG~8EAAHS1@?j2mLE$C~sSAU`%|C4wqJ05ynYIBsTPWhD4uQ8dd$b`W{e#Fa=j5IjVNYZrfKilfDTc^ep<9obSDRNaia36ydGjTiG+xzb5sE(|>xv2JlGsW<N=GA407+@wuNz4<cD{JJJPJ?&L)(7VBo9HISJ5~h#YdtQJ<eIjeq3~}ylh~%;ApO3B`Do*Dq<r@G@38r_x{K{axN&8)10M=5^{;f|PRlUGO?#Sr$W8E}+5M{siV((>M8Y%|r8Gt4(mU)<@99NstTw$EbK4#n?_?<RiIT28dY;MivKb@_^E!W3!FDMR0s4|I6&2lD@ee>yUqL7SQIna|m&PSOCrk>yMJDq!h&wj<3(yhJkt%vSf{tfnt)Ew&#Do;i@15gaFVeOidr9~dn$u7f1`uHOu|GRK2r8>Z%n}`doWC$`9<eQ9gv(DLlfY&KxPQ1}!1wQ=L|RB4*K4LxkveJBvFUG;1fm+H><g^pz=`{qw;Yg1hT|?U(5aCk3BFo@wWA+Y$lCn11@=cXijR6`-@93*<|zb)Z|M4KSEKYfy*6bOmic4zku~vOGqqUcH%DS+*>x>0UJQ<P&Uzh!rIYvg(54n9<PJNd)adHGAbALzY;hAGhp<~pysf6a8|kK8t2m^cETnfaU=&^X6a)*RZp#{{>UmWbw_2I2H@EgTXtNjf_~{u2u)d1LvBphj>pL#&lm8&!spWoKRI)j@-g77tVKX`;Rb~5jAO06mE&fXfC^@tYo_Mt^zbtah;ti|OLe%u?>tEQ?^Kul~I!2ty+JFM;@LzK&gYO`l+_ONzS-v5@%WYncvG2M=aA)sTG1IQq+M0ZEl`5UAec&D3XQo|cM0TX?U0en;j9mCKtL;#P2N8%y9+K=~0at>7IrwHPjD~4sOE>eco7udy_m>(aNIJ|aW;`+jEWD+FDBvwL30=MJNw>af$2G4!@ZB5pa#+8=c?xw(vdS~0g*7F;tdN%MGSFQ9@-F4vuGHwg<W=-PREZ(#4jM)pIMCvF#~4|u^>wNYR`Um>de<1kcJI1{N=ROD*zGfxVobd@Ms8O)tM2e$Z_PRf&Wj+J_gI!BI<>2q3GmsbOZ}*#-Ki@`Ky`be@Y~Ia3e%kH<w*ZncN-_yRHbmfECLteNY#eEfA$#zC)(lKUE;#~0=|%bZgQs-8$3Ib;D&5_xewd0+~~x3Y9VnAuJXhalPlXq4~6ZKvxoZt`*LcYQB3641Fb@Q%dumtN}f}%vctOh<|BAzRa>#MRhitC5LV{0;4$ea2ZE&PHR+=~GJftq=&IT$=eu=FwPatx_3yj-X|;i^rG3m49TnEJ(omoUy`-PMTq`P!ZK+ZG<NQGw5xny=K=%wQ?EfA*$Pd$CJZZznt61REm;bqH<VUTG{pvx(B$!NTz9={kIXdJjIlk^Q315e0W3_-4LutdH1cq^fg-r%CLx*X49yOeQ_)6BozNhg~eIyeuXm~4OqpyrOLS7txm>T7EkCCvTg<<DjXu*08KCs-oJDR)#jq0@-m{e}Og75Oma0ElChM2y)+pK1If8B8Vf?ioy4rh;yv!0GEk$b$S6&ETv^7;(cKK>tV${*bKDh##onJ3vJ+8&{8LS(H(@Q>-Uwu!!Wqi(Ooxxe2$+8wPMDyfzp9E9m7nMyJ$9zn%%Vpp}zg_`0#GhC(T9yF)O_G5NK?FFZ4oDAfb(g(Zd+Yt*5)!+$&#H{%XrMoi%6m=C*8?MmHyG;O|PIYR=zTb7%Mf!KWIkntT(v-gzx<e&ugR8fG10r+V5{lAU!FZF_=CPQ50OREcaU;*c6@>Rk_NH0@T}*6;eDoE&I=T@GD7I=U!gU4uTKrtGF=N~1Badvl7V92xu-@be-|?u#XxTjW0~%nFz9`%Clp=`<SKq{+cx8ylZ+#*^p=iH(_XrxRl`F%XxB<hH?h_2hRZ)ZbQecAzIJar9!Vck9eJR^`7i3tb6vf-N?8*e~ptAiEWYc-$_CTC>tZL!3iO#4Tu5{pB4)A7l89UFK1KKSB(?=k!Q0|%hv7Z8eO}Zoi_J(zGtvV=G#e>q3EAbJMq<cmoUBn|Wjsoopx_LL(9%!E-R77?}z#MQ#mI*||cIUAeHJ>X-J0*fdTO8JR+gx84lDKXtD{xSI4QK<tsY2&8Q=PT#-uQDNFW0^`rWcZ=clEJA17InxTjURNm|}Ivz_|CwqaGD0un<BjREXXm;&kkf9#>CAcR@EOmO2^0P`;D_NflLSv0EA;x1R`7EF`+QS8VqJc#1?I#^+W2Pc+~0<+}g~n5YKO*uR{p{Fdz`E0lO<{;4QjZkqP%abs{{(x)XPE>1ox5}2x-a%6iECWx-A!$io~cv6cQG+Y}{dK2A-RE+k9LEZD(d&Z5?;}ezDO^c)!TnMmp`!SJK%1;xxJ$^Jgw;|!%(olAvt3@QNH<iGHQfJ?(Btnn@Dd{6JnU{4yn%L!CXANK4W6{8Gl~jXO3K?Mr3Ga8eL&Mv`=YbKr_<)e#HPn8CmpeC*Fap8D{~Ds2g%EFlPah16e!NV!vP5DL({5WS(~)M~Pnz1`DR(Ot2b8xP`wlvKS2i*>8QP8@7IT_E$aHqlh2{=?`zKrZiiRj}qKg2ThA`2E4YC>|+{YyAy12U|U-DtqC$6&nKplrxZW7;2+vk53BM4<VHgW9qG-F)@Em0Pv(w=dV%*}*4jS7kk*;}|E072iA5U9ZFD{xFWZ5IT9&dh!Oi$QZbO!;eU^<81uI9mW%nzz;BY&1nm?aAAwB(Xa%VCMv$Fu<7$7c><1`y<)|2)sCGGQ@rhLh;rKWj7VAw54tU&vU9Wsc$$`O-vUQB)j<WOb;IFqg+tZ#fo|TDm~ZBAZwh=&VY|i>uLT-t@gxO7QT8s9R;Kybp%*E<TxqR$R`q6zVB;5^ivP&TWMJ)lAs!#Y<V{#M;Ju}BiU2<7sKm|{W00T;Deg1JWh!kFDDWYE_8~G_DC-4tgC0Wf&({NyW_qHsLA%~rWWCnR61;D*_B^2DsN&(Ax<VvrbR8!86*XocCTVGR7J1<$a$<7I^<0}fd}&cW@-uEMF`w6Af0TG5i=yH0W6L_)BTqxn<{8P)u{-lg96@ER+lBrZV@2Bt3*p?Jai;;L#3^zj1lr$4&bUN2;r$tnZYO1B{noXao!V_s07hBoITgy<B|2^ypK?knwRx3{q{y?WqIUs=C*F=dWTFW8I@_;IIZb|l48(ib^3hSRAQMw@1j*a_oFsiBSBcBtV`YPfQpS*iu7%X4KjjL@yyRjcZ>Op@FqYSvlV?zkvYeoEO9Rrwu~hNl=K~CVPl{@$nBB|z^*$D%cymj|H6sdkSeaGBAD$$v~&L+y&Rj$7zB*yxDhgUsIO1?8cTzkSlQT~2qY&GGf1HpLL4N`Yfa{ev`O5V?`Q5j3%oapNqf$1??<gO+XG{Gv;ckCaXz&=zTKnf%kTZCJB}6Jp)mv0E+fNQkn;RxABGr6Up#QL6e$wN#j*;RT3ei;Z;q)E?CRjZ8x-cl#LQw^?XmI&oL^8-+?r{jj<N4gCYPo=Xyka(`~%sB2$ip7bmzH-O$bT#a@smnmvyZ^yvu8<?OZH-Ifw*jmO3#xJ6pxz_gr)e4u!F+3FPUa-1mw;r<(J++83~J>5(7Y@OEvcplKxhD{HE~CaXvwoEuCX8P4VN4N5Mj34=RS3kXuCVe`F3KM>67lZ85uy!wUQ+5!-I^9RdklEr;bbv*@NX!UpYbOdc-lT}q}&q;lD=8i?3xa3*tDMuj36&8ot@&E*e{r9=}ksMb+?c#|0ma4?T^1!{Dz{#Rf?4^}W*x~Gt7EHvkE=8L1^gmf^6k6*i`9l6h5a#-m1}}~#nF%i;G5S`ZkoVsL)h)WzUQS}$4EJO|bc`h2bXX-UOw?Xv%|D1AR{3E^k>!7vneJM>24RFo($!NVQ1z2<4#UaqR<HnK9^|rQF=%uYSb$dc-@z@(*&HZ97?zJ^)T<vqH6W;>?2VTCe&xqiZhR&D)L~oJ2gAr@LXXtgz2^)twIDT1w}o@{n;!uHaB>kJahz8xQxx@C-Ga%O_C~%1YY5>gx9YXT<&oM1p@m!zqFG<s;5-p1ZTImVrR8SuXY)JkII1hled$=3<J`*S4XQoMU_%2gDcW$4beSW3jNwiFFb&Bxj$74s_y`9CP0N2?MFm@x(~>b#mfwcyu}+bzN3q3=j@XD|FnA7osuOKQE(1W3f$Dg8tS6cqUh0<;VRX`HXU+#$i8pZa_XS8nQW=9Db*r!e?x@M{SJB=t_;aUgu1C#|@|fuD*!?$lS$3+$_&8wQ{W5!l6PmJkKW5uqkFpxzvsn`=1bB=Hov8qX7#~qY=$W7_vC|cX)bbi25_7K0Ee|k`^dqp7*}NbH)(%fhwu<)=b5IUXpzdPeKUwde3i$LL#b6AlJ1;vYm50NSBFOBQY6WC~k;}UxNWO>hv^uDmPwfStlr;C?41wdf7hnF6(JTS@P=e@5n}EezEo8G7UO#b6_f|vNn7K0wdo{DQg7z;eH-Qa-^>;%TZ7u*Q^Z9u6Xugp=H>ZdDOGxP|-uV$~R+a<;%vGlTw}V){3Zh0q!`&ES$xgzHUpNBH7F~<Ui?^ZDL>vYhH5%U(5?tyt(eaCLpfLl8F4venHto~KmtMR>k7{$zVZ<z;f-68VXWv&hY;H*2k9nJP$u(YjEeYnRMR%Hln$%UXO!>ls=!^{BmRLuVDdvO86(%%VObZ&KKP%&p!Gr&<rxoWoklba;vqIBnf4cLb(4ADTit5Id0D3a(CMY#0ip0mWN<8^8E)-N)-m*Ph=~u2wG3cR3xdPK1$LNE&CtK#xDeK3fPJ4{HK1tJCf7)(ltbt6`APKW$l!Q`s!^Sk(YGj+l(uY<puLS5dqtJ@i!ue9uy>jRr9snjlJdhGzgGV)6qdbKLYv44n*-)<IX}4J6Q4hD97PxXus5q}S#|WLf2da7tRM<uEoC28ecBs1LuVem?=q|UE4H@V(_Q-F}KW*s~XHUMI-E?(EU~91+y@za7U)yEb8PXB^{Q^%%|3UY5fWbK<*U)SQ_)rjRp{6iG`5HYs3IOI7!QwOzihKT?9u=oTF?rA{XIHI#ul=eYeJqqcS_gJ>l}L7^8%1mI@3v34#^Fqis8=K!Ugw2S;&wamA^!z+ydHfYtfEP>vz8uhlXmh>gp&kpis*M=D%vhhY=%=hq@!&cS;Y>o?amvX%7IM-(%zkfSicN)%eZVjck@Uwbm%FoOQp>|Qt-9hxRxeZe*kOp_jg7Y<3M9!sV8T0%o1U@O2VXpjPQFM=QTvES912jJelT&spZC)yWrA%A?tZ~Gx(vq%r>8@10Wt(GsASKiDys{Mm}k*HZ8YHGENl<sP)MuuBg<fNev3T)D(If51gt{+JHCJ^Sl#;<MVudobLN)%-?DgVGpn}qQl=E$WkK_G`V1z9H@r(MW`ux!-O`P07f;|*2`Pi4zOy^sZ{qin>d8o;{t(eb)PjOgoWZWPzrP5qO8_$R@^bTDcE>d*|PXpFUKtE-SquSBFT!SuK?h?J?UiO>8XS$L@}xxXZ>RQL4sw-2rY(QmrEID;Aey9qbca>+Q~159RQx!Hoz~gWGIcF0(2{xHQ-Wbs*r6=2d^IA1=h}Q>A&j>Buk}t#B6$U1PL#M8}@rQ^(M>JC2a_q9>Bwf0qG*tI7>~@=`#WMO7sA<VQ@zp0Y=tlhhHz6t50Eu#sDBCC)OTsgYY6qNSB0bs<ti2hK-7p`8*VTW0d4Zk-2CCvNFbAX*}h|Q-u~W`5kG+uo*>&Trol0?zAqw{is%H`6t+n2#pj*{@#E%u9&xjCx}?<7iK@uum*?_sW0bq$s$ywz{x4!Sv8{=iOr2)h^4dF3Q_E&Uh_=2B23EMEJAS(L=Y9hgAyolR~t_PIT8Y=0S?ex;0J1SO^*>cmSR;>gEO0N9|Y1BNGWtNWlOd-w!iBO_+HwFbDT6wEXO5^KvZ*piZqC%)nyym70Q{W(_zb-T!w`R+BY?R__<lpw)~q9w1TuA;0+X_4eFBNJoThw>eyf2+O62^WmA7ua7QF1BbqqzRIQu-`4)so`W>Qq?M&Gdt==V&o%fF^B2sgv`-z%XiV@18Sr;{L%Ml}MkPLFtyhO%~RHINt5y;|tmqILh+#<r))g0|J>rSRz@q>6zDDX&0RhNHU``@z2ri!0p>U_{<@HC~mr_B>!5m}&+fN;kW!8V4_2pi2@fk!~|OFU7W$^hR4b&J9}!Jwc7DOE!lDna_LKDIlb(i(<i<Kj0+pe;0Ra9J8z8^eaor9f&fik92r6tuQPjAp!5lGda;ugbtkBUI7pU%NlUn0g4`L>bQ?wlj;PvL$&%P!iDxl>R#}Pu!23ud5LoQ{H2ygfk9!^A`*<5`ocF7fj8zaw$^pta;wcubAD6ji(+vd}-`_$aB~i?_7OaVRUk4{Vi7ML(Fb8bg_n>Tll|rQ`bOMB<paIvEOl>1{C8-MRJ4uRaNT<EO40xDl&gw3~|tL{T<4}h>g+UO3C)LNxUBY#j{AcZ3^3=?l292fKpvnzrxX=bayfO$3%09?<jy6P;I<Q{k1<0@DP;(<8)Kj6d3Z$maM1CJ8vBOSWm<ADbx2<mWUQK(2V8pOHs(VXtnd61aQzCvAt=DavTs4eNcPXe*8Yxzv&oMMggD#tmNshQi)3dmM$keHaXSYquuUezu)^u@kyOh>Zq81f;?byvbB#~mR|f#UA|DY<DIV@_HiED9+-25twd4ILwWaxg9l-Pdt-===E#pKueqACl#4UBu3ekqxE=^$`G={0gpr@iu`1+eg?q7E(gZj%h4jMvKT{b2q5K0adE2i`ZidbO2&mMWm0oh)ScaXs3SG1So_>Go|Db$hN|fe_{te)MjRZINab6;5;X5_pp5|mw-Iz&`OX0i{9z|3)xDYAi8~B*GFoYlRoH~*ui;0n3)icXL4HTQcMsqg5>Lvq((T_bo)fo%-SMR=MH5c&O0FMY+_;>PadBGDlGb%lQjVt;X*+W==1n9S0_>=kv0hnM$U(66BR0n=5{uLXP8<Cadlqjlta{Z0(cTh8r^@}w>m^<AD>g-5CU0qAC3Ni9+vrTlIKD``UL8de?ne=HNbHQ+U8OhMW&t~3|N>|<lBy6lM!}J#jkRi$)2s^_f{lvb#gvnw<)?Mw=Gknk4k{w+`YM^fbc-uoNDBaQ6TG1byZlci7XL|1qzhqI7;Eqxj3RCI-%0n6-bWJc$tK5M^v?)`u;*q*6B_gd;Cqtb5$`}c1kqna3y-5<+NvMxJJ^X=yaa@eDR_E0AE5Q3%ulj=$$3WiJd8dsRa1Y}E*y-nP_j!{3Z>y^2%t8cU*Dwa7vmVY+1?qk<(W~*uXR|}Dge~t})|IMkm^do3DCK9Z4QPrUonuc_NN6-2t@_l9x6&ulS9)_rbnTF*;IB;KP4K%$1nWw9_m=y6%3lS$fQ+bxQIfu``*>$~vM4{nz2BS8#}W}F_g7h)*g<b>=-^-s;$O{=I5TBz10Wm3VC)?ayM6E04|(@8SRV^ZvVW~riB;2P0<vW#GP=bh_RMHUyySHf!$PcGBdx~rwe_{Rb*y24t}am*Sm0D1R8RYjK#1ZHzzNSDCs6MO6UjNw;P@R``!s7L36A+q1Y&N?-s{H9phngEaG-g5Es_5Yf+xMgMSNrL#|w?C4P3DicY1$ZY~8G0nHqdGIT(ZTiI0Q8WawY4mOtOZZF(Ap2L~V5cD3S-u!v2S%;P-OWs6v|y7X4CNZp)XJ8HSD;a*~e2V9$+3-AJ9>=t(11_}?<3ly>E9PBAK>rqCJF?^&yP*t$OLs{CKkf|c-PQ(=L@b+b!PnQDeZ~F=ICxzyk6EoL30EH4<<S(RqUv6orts$>7Xz6Hg67^k&BBlB7VYkWhNAwAXI+dc`AEf6v>Fbh=mhwf7;{A*<5BFYVm}yFzjjavdC0^wDI^l70yfnC{q~FE+t*474>DHu@>4nQ{F7LjCKojuklab`G`)$@aGWSadU$)9KDELeGS@8YFwqkjzWMv_4L1X7>>_HKyv@`+~tef*R1%1KziTp%$5|&@l#nT!%%v|c_?5P<R9>v33PePt5)MMg~lzS>6qRj1!6DgMk&1J8cC&%&{6^w;c&_gIMz{E-wCLjy-IhMCI7Q!$9Wh5w``72**dx5e#qeqBfzv*v;nDl+Q5WFN!4c=1XI#-Qg#)T%@Y$6@iKB<8v5W|GDq{W=j9Du42y6Z_+Yvm>tG<R7U$=!%Q2-gPZ*+*TO5%`k^D<`mGOq>q}a|GQj9R<K8k!}9@kQqJ~Xv+gf-27~&?HK}CVGFqn;jj`s-BA7(2n@xV`tCO~23{c?IO=Wn6SXDadTVP<CIxS%)Wr~e`cOT5YpfQ;a6QbPM}dIS@jYs-IfmDHL0Ctxtk|RP9;?)59Uim>`001(YXk~(`w=t0EbFXb*KWy=a_gDDNZbnWgE{}kh;C%>U{f<JfAgZDQ$ZPBq+n)(@k1oTK^M)trAAA7{`18B{G^mMd&T(1?*On0AfMJG^^+`>-Z>Mcytqp2u&>3ORaBfW<_FD03Ov(P`Z9|6z|tP%f4Q>6q7a|;i)m2Qg(>%M{TmksvA#xDPQ`-UYJ>8PiGl=VcSf)IS>3iv;y*3k-a&7#PZcRniuttbTg1278`$5inCS@jHXk0FBh4JAAq%c?vgQgaPDXTfr##`f<oL-Q)%Q@Y-t$ao%?W~$n1aX>HGq$qIoOIbhGeVBbYWK$xCtTeed~-L0=U?-4_S~bs{D}0pTs}y7pSss8+iHnSNfWS3?XJ+du5^`Aq6(EYS@$J7R$)C2{yj;W~dI9|IW;yBMHL`_n9u+05$#owxo>xK8puocOh&##msAmWqfbAQPU;sD`gO#GUR%6MF0Q5sZ!J9$EAH)?FedC5@I|IePi$(M`t+!1lD$>9x5&pHHOEZkS%0Lv7?UbLKUAEm(xiY)l}Ln4PBp1%dtA-mJ7(6fgFWGD1jlqPN12W&0{FiQ0M|Q{9St{R%O!ogaE0`=DvGB5U(zu9Y<mv+$#qAL3?gZofe*QR+)k<FiH$h2fwE|ef=y3E_}%hYxsROLI3Vlsv3vzA_D;cgS-5%uZZf9Ty;EY{y)|EADMLTbQ05cXoV%$eZhflq)RNkqe~2LM8-Z4Mc>dHmN<>FwuC>`VJ|wmY8q}Ja^X`p;N2(1I2;~EpyJ-!ALyIQy8q_6W~mrM%}w36YLxbZjpk$!y!7}-66de!8jX|3Qpg>&{nrY#F>B(mT5$~aE4jRA91zMh0_S07d=SUKA)%+Ab2*03x07oH88#&Bc|{+YdwLWNv}7Q@wy_6MOE}=%L~W|mPtgcb9Pc?UvB~r>>kbd>7;{zC7(R%O2dVHNEL))0Qs(>eWqjJJ-Qv?zZh+Om?OioKGDnbXO2A3tiu8fou_PgiL!7Zs18cSQlJPlNFmrlD^d{V(?G{SyDC1EoLdfDy*}gBFGez4OY45;xiuB@wcShkA7}vF=c0pLQ%?ewM4Ya3Los4{ja>`SH%m<Z#5FlP)o}nvU;7QMWezN>{sA%$2Dj?n<^r|jq^5-Cp!OIZM;E+v-*EmEyO}IZHk(TOz8^h=+DWY%S9r|c4d>k4!0Y^MMYWbeKd$EIc7|bN}qz@o~wg$|mabE8$3;T|<SCjSkyjI~ggnk{2^6#iV^vQ*hr1{Oe`5V;!tnsEdR|wB@q9C1t7k<*m`l^bgB`O`>4fpL?2|5LC`m9E}1_~oCBVVosXpr{q4FTLQfAv;`yjR*m!K>Lqkn4UfW5pilQ3*JZUyHc89JZS=F;YEjiZFrs`_sU}l`dPGqT88;SR)pB>H~K5OWr)th`u7ao)3Bi<Sloc%fbM6M{n=gu$GJGkk|_Ff`)xCw`bbQ6WQMon59d|HE7SkL)lm<2h=;>@XyVXdfi&~eaC%J`MA-#{p2anx*rV*3gFbYN_U{i`#CT(QlU}ju%MbK5Wee_ijvXev01=Sh#VjH2yw<VK%~)TNh>WXC|De1mL?F4cI_J=jlZib#IMs48sTKP0doY2nG42IRvMVNP>a;Ofx@K5R52o~rKn|61ep}Wp-~JcAYfH-AKl2p&a$aw(o!(WGP4YNUe9VF)Ss>Hf}1MQp7X-$Qu^j}!b0oftosG4)Tat3dkghME8a=n&(CK$=UBcCo=;<<_Q#Rx#5RsxoqH55v@uh;`kduJe8Ex&L>W3+I8IO0gdbF(z8huZtfIUuV~E<4cr0QtYS<<CE6G?xmZEyR3B7+V@y0C4xcsLf934zvo2rk72GbOB_+lX)s8`$@&5rcnZXhbg`K?N;R0$);Y8S#>^mq{Tq}T$Z#RHe+7!`JKN*dR0HB7?yA_+ek#Lg6Ps;X2$)7B{X6j^|EU6G3KyC~sAQu$*KZfthpYAcXd@}Vs`Urqm@)H&58DiEk^XPDsZ&5_(uSUv*Z?`L{sAEhzae_K+UK!)re@mTWVnw*dLQ&`?s9iz->hb@%3JPptnzoGz5jim*a)~_w`IkWC2tXtzokqOo$s(&cn#zu*>$95#$+kMX*$ZPe8s$^kh*FX*Z$~9mgZ`M~0=u*C+|F7<<y4ginUBd6zppam!2jRtVC2_o18Yxc9>J*dB*5Sp0`xUi0aHqOJ+S>$at|^SudSUr8w%MPs+?4v)NS~-?Cej5soQk9OoWbSrub5|=<ntO%2;o+pz)J?tr7%C}a>Tz8>%HAq?7x5Jt5D5>o4(X3L40iypS(OBxW#4S>Td9e*3m*711E`Q3NAmY4wv*mb<>+(F$`Ci>uCAk)Yes_#F6|=hFn?cU9w(!b?%|Wk_PW_8!T2b-L6n9_28d_nw#@HaG~jlLc&5<<)HkWIQj11axlvs@5gbHqa~*hB5S=jbH#e|2k85fBiU<ssfjEjF=8Re#^*(;K|OCOK=BSFwj!!Wt=!Pr5I5QkNJNHZ?>5d%N3dMYkD*jDov58tpwtb`y8_14RU8f23@#`n)JLKD9qq0*G~}lYBm{SIBk?ulKaD==m3NR14GE=_^EU<+?~L2RGeoMo#Jw;{0VX!+<9P-fWOAMCG0*q4(N3k#gX}C9XXPEpoUM2ITD75977=6*HUFw4usl)ov{b?YeJDREhY1#B%#`|Z(%2{IAH=y1Jo#k9-7?~+Zj4CaEtQLTImYwpzo+fFal58Io4(NDN|0EM?c1|-V=&hupOj2(;Q$KIiQ805WMIDAUF7d-sJw(4oh9#eG^5VNtc(ah23W6|A_mV64K9W2nTWb6>;8?|Cs4sTT7XCvB_?jbU_93*qG-ffL>AbXIMso6N`!0D$ez+8IMezr%-rq9d8+;=;6zpg*Dq82*kLge-RQmj-B;;>aP6bnOQM4e$n!61SP<XD>@rcZkBu`YYE5w*ALuo|!L3Dt5})|Q7t@n$2bz!%Yg!=tCxG{3z3HCS?Xo>o{e`4bw&33dSV)CNbZs8+7^^iV3&+9EI7E@=2_3#arQU-HdQ9Fe_8^4=kPMR6)+<}QWihSeY&qtgiP=8F`_5oq^$CxRoLq4HzRZtE&?jl?z*0MUL1CKCZ{kz5=?O6*@rVgq=)&kP$Tp=vBv7M3To(YhD`DjD(zWIKNz3`dr&J-M$3UkDJBLI%k_nwcC9KRw=xg8^wxMJNh`Kw{tIhK@Z$%Mpuw0xFnK|I~Aa9fkB=^)7f)y=oK^|vE7=OmXJsNb1>Ty0rL;k?cXlq_ur{;GK<Jm2JR3VqMy}TnuZ-ZY#OXLp`D_C;`oUE&t_YFs=aFxiRAj-E6a#nBAj91w}u_}PhIC_@`%tI4JiL_$2I#lRLL`Z-H{KWI}K&fBGaNMPd)wfh<!tD;&JY3lzk%&s~kGMeu-_S?Eenaw~pYy&%?wE)Q6eSDHL0o^Jffr|@c_-?Yo~_K~wN(I^=BoL!-*orIeF6T(2|*4MW7SUt5iDXtQGkg%q7OO;ZKhPH4KdoiphT87?MTWIjE%<#AD*&Yb&Sw&=TUWwYm4Tp5{75b&O8g08Hw#ol&bAH3pV-|&JgK$?vc1sssP)$I46PyrTxO%w`HqStr;r2obW+vKB;BW3(eHR5LK!uEzC}1mI?bZcHB6mGHP-e5?6@@k|H2%zTQZ#y-|FoLV{XuNX`(`aVQVfi>{^1{_JOCnCQJVoi70|;7jbql}ksLp*C+U4u+Ur^dmBI(+<nGOT<`}3Me$Hkv{Z>JKf^6`Ii*xr2S~P+#-pg>y-sWB|&P<0~KKp!^5ik9wda2|6NmQcfsRu<1ZqF61xuG<8MBHcb^hzf`8U*|I%k5XdbR0rH!!py*_;m8-u;xoG6&bm=l1z%N-uoI9I!Wd_0jSMd}-R9--qm{k^SQ2@=>&*eQ3S7h*Fr$|buU;8$C4QBR^1HN_XyXIi}W2mfx8N1KVUW-z7vXM1VnP6!hNtS9O2i}d}LXpGQbbI1dFxi}o7YZV~fM$hNT?}_0ej4qVLJWF}aFqKc3MPu7iU9sjNHG5ZqPh7V+OAF>kHC&r`!4Y=IvZ~#D6dN*4(Rv|N3Ni&2i>~3~bxNm$o$rShzp_rh{zAr=*kU&x(R0CKKuxlY!*$o?x~EIt4dpjyuB|A1s8K16KP5-MYoP$l&3(7pw;-$Vwwja^lGXwqsp9Qc+?7<0{5#gL51Ph3s!A-TJG8G6A%QX5QUCcM^~cR;sEbM|TS*mErPN^#gGsH(zp+kXN%cGJJRe~$D4<KepN$8eeb3Cc<yQS8(nmjk@N|>XJha@}b7$&kfOuO{Fz+G-pm$Xrv;dkMt(+8-vlx9;8kG5V&}vTY$^TfLnv-ZxAf+QI0{5h$N5_rJ>xh4)!b@USh!5`t-Pkb5e=@L1*rG8O;}Yr*T_T_i&b$0zMU$vQnwT~4j%Dx;cVw{EqplOowoc!~@`)NqU29X4Kz?!W%K8-zo7_S{W|W_b?MqT|cYXTSezABaluj3Fo9z2o<L=LINEvQ*-Is2%3e*HuIx(T#T>k$>JY`wtv7dKLrNhaN=Ru?h>a5vV3xKwu<tTm+PbGASJ0>isR!U9(9bB29jlh^+l>_7IXs-*8w|hyICGp92IgWrrTs%j}ZE`h*jih6<tO_#_U>FSLqZMqh)<YBR&t~K5n&9hB(fOq6@MY}kY3usyvi(hUuY{`bowrK>n9=Qj@}JBNN>oOK^F?WT7V=FFA$dnERdkIFiHqh8!Jgi8O@KPr*Y_5_SxIv=xGoLciw^>@lMf2VtK4kM-2^-SXSH9kA7Gu4VBwC#+BUfXbhc{&DmqbtZvNm7|B+})Von@p%rf}$pEmLsjUc@?uT`~xJg&80yLb<PYtk6;Ia2m3Dpz)4;U2?<$`>l05|<lxbK{BB93AEMRg}Ll!jWxhf1q0ys-uj6(tiLJqbXoy=VV1yZFO}k5kM$iBAQ!P`D-?fh2k2Rq8Du4cAZ=rPEHdqJD1OE2wZTi+~-2mT46A^#hSnK<N(2Cm=-A2QEtvlk`zuJbmcVVtj}<zvI2h#Oip}`ib^G^A7BjJ6H{F00&ti)N;JW%pR1(15Q;^xBb_Mrts#zOz!T9_q%@-VC&$kjD}TJ(Mo)@&;BOQYq^yiN2qx9XTAaz?NyE>$f$<@<SciqN{{n0tAf9ZK9k-N$Xh{WzC2p|C`_*(R-wOOLSMw87$!3@#PHby}aJuyWTz%IbjT4R+8<2<$l0UvLvVSVfYlpr%#OVU(yXwAptANxp5qa0$LNj3a;$iOSQhac$I0N0XZH2t8975dl4Fc5I&)%w(IHuj?43ZkUo(YH|_bRKMyp!X28D3`$QAn39+q_wn>I%v`(%fclEIG%?nbP<L*GE=5SB(FBqjmo62{h>>;W$mg_H;<%=q-TnS>SgulTnKO8cZU`GBjxJO;vNut4?%HD$AK>)4;BUD|pGBax>!c7XiG-{OYK(T!(^c+ItbMN$xMxaT75#SfJ#A6`^potrb`}aWNh+-I^nZs(;abh2(bvQ9I||0{*Etve@1^@_FpvbGbhtuv^&_p5Z=%+;a|$wC;h)K_AJJ=N480oECW$*AoWczdLRPDqnk;2G{??rVM6yF{%OoN7p_W1-!KX8p~R-lSo6Y<Dzbj%|enje<Ty?rEU?of=L%V(j?30dMV`}KA+ndU_zi3GYRCGVQ<C8JYqlR>zD?cI^%k6AmMft%cp$m>ntLnzZZdyhh9no^T!cAx?78x>@c&i_P}lv=jBHtdZ&qlFp-aOA|R|h;Z=A^f`>GeIJwS(o=mXk>47N}LtNBU`))8g4?_;-lUyL~Kryv{DD+wW5Y3Lh#2buQWhDJGm~BvTY#<(njKASF|BkgbZhAq;e`9*k)x{-5$#NuUCWR?Wa%xHQ=&Cf8bV-JL0lYUUgTbZ%pgieGp`7r?Ri8Q|7t8W7hDd92eX#Q@gz>6Xu7={pC#c`)KJy8wA+EQH9008IG5oe8C(#rsTI09!$7F%d&g6lowtCiCdRHwc)rE%AAbB@&6-+&bC!aW$+y<08`BDd+`h3j~aTGE%y;6Ax_0xeA!2oQd)R2Ct?Cj^ErjG!>GsmDOxP%Nr?7cuz=d0^TNgJ#pGU0f%BzqL~rh(V6aGqw98RHW2z&ja4Fh^zZxH0N^ovpz$`dwOYI(F#J@e1#VhvrqB|IN;Whg9=tQa3lFpb_F>sj_D*{Dp9M+$#8?1V267LrABIXLy_ADl(LKc62)FwdRI>4NeGAhgE8xW`OFbmqE9S=@*nRH(mh12a@Umq~xHwkj0Gvr?I~vq?;F2{-QT?LJ}c|2ez~%;T;3Rp*q%t<&?ADk4<03qnFY$2^Mi<pegqSzWt!@z$>zrr>$eMTetPU3tm878+NZiuaW*yly9c4KrY8YPzmy3=TXiJ**S_rYoM`q5$IP)PPcU_`tez-wA_?|q)c)_%Lr*+Q=2;*gie>3(sY1n`W(qUx~ZtbjD`+leJ(UB1TOcnC}y%0`;dnJ1yS(x?AOw#HQMcfiSMd6_1w;>w^F@PXZXM6#(zewud%_eglqj}F8P2)0**;t;?Xn9Y3NDQMBgB}-&a@Q2i_1jX=cJkf<5K~`kT0pi{v0`eQ~}%A=@6dF)FBA1m5B$YQF#K7Hm?Y6~$<*SjgaKVN_nBOwK1ql5joT_;u%k<nkrMJU%FAo0LHoQ;XrzfZOBpny#x8`jT{)&bZ(X#ic01nKf_vW@1s!%8PFs_m#desMnF!?yH0!>-@`g$$3sqURxn#3DM9;VG?At*s>2w-pU!!;-T;|bi}1eh0{971nskh8Zk*^q&Y^ovXJZ0BQTPVd-aF}!*KCUNCv`U^LvW91VVsYGEIs3M~gRw&=aRKMjS!>3Y7KuXb5)5(%IVE0^3>a>927e99IlUH7YI<(E7Nwr4cPq&mOgYxw<82LMBd@R&^EAUPbzAQoPox?A`Tqf56w_t;oO1&{b~ISg$DO|3R5h-%Lf4S`oUT7Vdr!=K&jfwidm>S_{F~2jfFznTR$71~0y}h<OZWrPp#ptLbV%*$U4a;^SG3`FP(ZljhC%H*kamYDthcl(xt$r2=~Y(~*MIcOkDKDhVB(;9y}>VTaA9jOex0v+~x}0_{zvuK4A0uIWM?^1J-4w+6SJRQl6JxX%qQ<C3~hV}U?o+)wgzq!zd?)7`P~sBwgA#rXsR4EfZr*B0;}a6w5tG7o6Qgqvp^3G8XtvxC{{$ts+V-#GsScJteJ0!t1Ejq>Lm!fn?w>$`;GTlSxhK(7>II@BPk2O=uG73EnOXUDr_1IsaW4=Prz;t)cX({Y7c1m$Fn8vp6ZP*Glc1G))zz(WWv3aVM^q7Y(fL#ED5U-O?B&C#D=A?4HsJJ(9g)wxWVDMdO)jVlD7o2`*TzDZSy7kO@}1L<9v?=KB)_rI)KL8ki`3Ma=o*(*`Lh%PuG=aMs}MwQ$4ui{0o*$cFWWIxRgNe$jmGgK5~b`+eJT$|X*i<H9xknI&N0?6973##4=WQ;?St_TatF#k4eQ-V^%vr$_rD|>yG?G~xwx5-=X%z~)9EPCpHFq`&Ud1j{ylVA|W5q$c+F)n2ZXmZdAYL+ilqO9ppMXZq-u){@+Y7mN2vETj`;WFeyJnsuZ?RyWb%y9=ru*@w5LO<e%Om8C}2bddCnl{KSBO9NcVTJZPuCAi*K8-MQPv!+4_Wgo2duLKI$x<=37g=G&7maaFC+WbV9A8Hq1gQx`SA0`_N(*t>AnNpKU}MjfuT=g<)|OwwS$jM-Am27Q6A0oAIY&r=f1eb+och&0_F!}=$662UNekUEG;GIUh*zwOmTc@WKUTn>M)y)gah&<kfKll;y8^x`nT((EjF^lcikthKKf)cSXHU+(b4{O>3CxuiN<^r@o6?F-`+wfs59zuT`G%7>uH;CNo;wq~KqxWwEfXh~lCZjSJ(FK~(=sMpmmxtCK=JFw^uLxa-m>lpDM=UbAuQ=mU(Smzq@jH~()l_)QbXrU#E9>FHB(uZRu$<~c4A;6^yDf<9CnSZjcCZ8?}21(C9caQ29W+CRiJ<5oBx{cZ@5N%j+AB%_|-uf41RwYGELD4nLNo`BS<)Dn_hnf=(NUW0$xE*0xkAOM<}=Oa>7qL-ZpQusb3v0jn@@OyT}0K;QNUCmNp+igTZ6l)cFlz(J~J=B_rZd++-~@M(b~wg?Z;jhi~K$tSnz@+45e{ao*g_80vk-);(&8Dn{p2H5Cd`VL?B$MvBFH5!xpP7rt({O)V;Raf;K}>nC+gif^L4_6pC;&w0P!%4)gT^k+g842|my!BhGyzHP4j#|i$)bo$|y&K(2haqcS0u3WUz=+g4o1Fy*m5yvP`QJf!^Aqxolt+3QkxqN)&hQ+oq4VP5ta^<d1K}b0|OC(brI%Nuil~Zv>K`*+k<@D287cE$Rm?kv9II@hn#><k8=?C7^KN<RIX4bM6L)tCOD@mN07l0M<MP{V*5jHwvnJR{htQ%GBz2-T2T3Uz&(O1Pi#1|QUc3~pv;aBQsb|?<iN;C0Bze{4a<0KyOl^lM5pHePVE+4BECRR$l%lViaNCiG=S)5j_l0EgB#^l-u<#;LDSZm}3{||q}`1CCfxE+-ZfHypD98*u6-j%75e(u8X%m2Zv1{xA#G(P8^MD5RYG5s@pwWwqSGEEK#UNJ2am<ccf0`kF@5w0nGAs=oO1vJBF46p?)3;+qtZbgKSvTa{_Wq+G$W%C#QE)7(Mv~J;>27t<$oc)%_2&1xAokes`k_Zq#ikOGHI*iru&2u|l17vIS+@hXhnTQFxIx~OCDEV3efPgqK5FD!wc|Ob80ZZ$>vz<CYTNwZu3JM+}4*L=T4U~KWXUd|&gEZOVjiIQe_I|R6M6W;AD{nvcHJ321<ZN~g8vc~WfX|uxvSXfngj>xhu)3|gL#1@xSp3S}+!)fuYX5>>wwY!DcnYW^R<O%gma^vdbXtE!?DMLbU!y)4*al)I`lvh{h7IP_zyOuQ&&lbl<P&gjTO;7HIp{o{&+;L>tJRwChiv#a#u@7%APjwynzr8dr1f9(5aDlgGQfEQ@++bg@63Dne9<Cu$D)s*P<tB_Qv60v8;z#~8%}Rr(nv*Q#svku#Rt^!<DsU2OYFoH%eFll+=@7afIdno84lO`>jM9|xaH*knP-P{(tW|7juuywE(**NWGE%^==6AT;HJxOS80>v1xsi4VUtoZ$Aa4HkC^cnHI@u#->khsf!WSP#q=&@Ehu~FV{M689ScOf(^U(ulJ#{KGSLWVjVn*70gWP2kE!4G-+%vfJ15f^aeX0Z1lDv&lD1+^Wm?8Z3Nqw{0t|!JageS_XQa{sdAJ?r2AdLyo7A4wPvJIEF&4|{1z{ur>dcX9G|e-7Q#R!NaZbjZFG=Ag`=(`=3x@se*Vh%D_Y?OwTI(7eHM|2zDb6<q*2KuS9%mhK$s}N*y6y^r{{h7q6dO$Y)9^T_*b%Jxuhx$@n2EM~GEk6g8hVe}9WA_<Ggd-mx~rk0BAEguywO>S;cJW?tCiL&R%&4b+=Wmtabd}MKZVl=vU|A*^C1}c)T&p=iknge^ddPxC%@*a5luSFDvpsyGu@c*#VqpoBP*kPI`@ZKiQAuG=7U2JD2OouD^$<(1-#t50$A+xbH~-(ppq}Abb+~85Q&CQ(T0ZTTMjU86g2kbWEIr*kihL9R!M~`U};$W0IHWtN&#2yAZnlz)Labtq>(2S${N>pKClVeROzhZF15la=#Io(Ala-YFa_VY2k4;^Y{5^cJn~m{Uw9A4&O@i{%lD(WQnPdV%ZLc056FlBQQ5Jv669A9bAfMj3OI<|Z4-%%b&_!=PRAp@d5Ix>aQN<KZJm7SdMgQuGT-;6-C0azb>WwH@{396gL1^^-$%%D%5v?{gUWEOv~w)qy<>5ipMtGzSsHB{Mu0zQlkdWn(D7gO;Cc?=G&xkD!0ym{9G8K+O<||slEwP?M%2h4WJ@N_nS3)4wh-N?)z6mNq!miTAZnbK?SX!{2YY&qwM1cc-g4_iuV1dgCU@v~=HoE(dZoK1CrgUJ&4r-CFdHvk&x={tl^S|e=edFr)E5YT(ZRS4V&HI-@<3j~ywJK&X1jp4em?sskIC3)3rrG-nk^6FWq8P~lvnOYwsM4(bGJ){|Hr|6o-)x9yiH>Enl4+Lk7(B5P2J%^`PdO>yNe7+3Pg=x*n)aeYN$lNWWB}X6QS&uwNr3sFTH+HI+*&Eu<Y{*dR#9lV?y7;JMQ|N%G$qBpvqoJWW4(XUjcbf#cShP>)d_2VKMRu3LPvy9rRk77&XA#k&!mLWN@^1QCy9bx9F!0Zo)TZ;|@c=PiK2yZB#K<RUP4>FTJGMuFclfoZw#J3*0j#P)p;L&T&dtHxiAq6#3Bc1HzG}n>+>KSLSif_Z~^AUzE~d4sQ?V82gplXi!zjq)v;zl55V*<{=!zJ6%LjMGD`{<})>4-Otj>+%Fo?-aB_LRVB4a*`ffa1EtDY$M_Hb33!^LClA<@wAK%l@9f*xT1{XwLQlty=#^)6jb=H!RmvBVvb~o$35@K<y3L4d#lDrzuCV&GK`}A95|fwhk;ECWU+o%1C%%~is?J1^Wu+$cNl)gPfcwp#Q+ko*g9vjFI9px;sImf>guzjP9tl8<rr&4@#gcd3EXX-^gZJz%<4f{D-;4fD6qpIZ?hrai>V@Rb5k1Lix!@FrxCq!FK2H)`8zG)Lp9eK@DjihEO_ps&VqJl1-Rw}@RY?%ig?5{p$-tw=ekw)aEp=QA>AgS3ml{X{!&tfd<rm+l64x3XyUB|f)pf72FAMESWNHVr`;|Oe>$c#s%FWqgYp_Bi>y4y#h9up*hSwJ6QVzx*kP|?&`<5Iic7{J9b*#Is#vEnzLF(6Gr$S_nPrn^}E+-C$5YIh%NBQt?@a1{5gMz8s>7JVfTAOD{?ItTq?Pfb6X7qDiq?b_v#fQ!V=eB@A{IDv#Zt7Oofj-Y+SUil~?i4hJTLaK1Ym>N?w+zVk><Wp`p_AR@3#maf>lCLin>~H~W3t<=*=5%`HaaiHhI9)10$x<2up@7UuVzomUb=L~prSeJu;HY3#6b5|)0<bF)*VFV;+bsp(xE&RhoSbz4!iGj*k<#!HG&jy16@4Z43%69333U}1*$M&B>3SL-5KeS{q8BY0i7qq|5;1<`7QuNb-k?%2cvY21Jq?|=ymae;x0ubUBhu-wCdcLqK&;$Vmbr<-*>QU+iEc!SXv8+YAA8g^kJ6EZLrR-?s{UIMTRUH%7}GU0pXjFlG%~8BAvDG^%z<hW=6rudC3wZ<}7rbtj;^>VG0^Q`Yx6zeNWW0>33MRU8l(Xqmi1pV$Fh)J2m$*LcEn*@T~@M#>#9W(C<!m{5bG0et|*PIur3vk^b>15c6q!1m7(P=wF>|pzN<BS(qtsCkjc7-V0)2H~3~fwFPDs&}?_0Z*tb)u|H?1il4dR+Zb54uaZ-xEqe8IhhUE(e<n^NJ{R&IE^Yq5#$wF;_sc>=Ch*1-Tl;zRRF|xhEb^f|FVQfu7aWL07bs3?sPP*`%ygy)316`8gV>ux^)7LB#4V4U{-_l*3u(NUxWJo5U?!39^p*>?$>Zzhzw}PKM-Vp5pX+UmAm}@{pLgK})WEtO*VY3?{p3KPzy}h^PI=d=Fi@LyvFM8R5YzGKp{O$)uMXIx1B^Dxb7fs1dX)zUl0|hUpR%4Ib7(Z)7g+XF!%IGw>Az##=Zgvw`-nQF8Q_L5BM16L_+%^QA@x$8i#I$I0*Cq0ucuQ=n<C~U;6+i0&E0D@Lh||`JjPAwHXs8ajjtR*E33Z?z=Y3tyY+i_t)f9i<RK%;K30%i_;?|O7EF2*$w+>WgmB@x%KG3k+@|}jkhm(r>A1}2-?W9M$-71)v(O0eRKz5N^YkUTyjYz8g%h8*RhK<EfsO%UXH`|Rj$<zG++6(f1C3Sizby2B;6Z1*`B4V<e_0X@#!ae;8`iD-2;KTwvhI~|Jm&`~mG+pdONM)4UMV1|2qm-Mr++|8wp>Caz?$(nbreopJ3KXN6(us{7_A5M6!-Mu_Busih3KXBB931`tJn|rd8|2#kKAbBx6YOdGu7})#2ts5-Ay5XD-tie@Fp3(41e3HUE|$8zNzN(Iv)h?YbBD($k{=1Zw|!^VY3=H7Hz$uX&I<eCQ11uv~F(_gO8Tq)G{WTKI#x7N!*Ikn+@3eU`t7W9fCRXP`E}YJ3?1`prq2s!)JuQT3=&<$fa@DM;Fstei)9;y<Z%%zaDNx@fLNuWK?&iX@T=C8KUxi&MK&w-9_`n`H&C(;hPbz(FMGnBNP(%4EaH)Kgi&Tnd5naD^$D~)ZH}}=sh{O7Aj@S^Ai2Bqb?hkh=Um{QdxrTuPvKRa&GO!a|8#z=~3O$UyuO{KJ@J7`i`3vE$|IvhIVi2ev=i_b8@`7#gB#)gs;_^fe~esmSCJ+&L?2<;o-5L-aaQgeY3%2GN%dMPGLZOwvN#r$7XkS9$&~*^sUj^Ro~w>@(HkIc+pnBy-c8=DE=Ut$XTxyBw-S?F@M54yR^-<->iv9y>SoDU&IvfqS->4f2!!TtOEU#1|qZ2=X~HX7j^!go+qY}L`htg0ltifrBrzs)tYa_^;HWRf;a?w$^u)JOi%_bv@&W>sg`|a=oPW6T!+dxv2TryrJmm%r+{=+OBeuNLd`_m^>8Lt(R8>5ObL&vcj@}SH+z(Cc`cX%Aof_HX=DN_*wetO@ehXny>h5^E_<g2uSWT-*Uh_dtZ&2$*787Z{iM!Ik3)%pnHaGvDRih$RAgx;6a-eZ;okp(r#OMg1(1zRX$C%4^FKI@fk_PI0stL)Wo$*XAG=!01U40e|FGc)Z}Uw8M8#RbHDJug;-aU+gzC{j`E|r^qEos`35pjNIql*X&Bwrkw>h)09u+IMNEBb*v!2E{c;k67JGm94i?mu}tjR_?VTb9Cqnz=Eb~)1!ffgBCI3?-pah>z15yOAQ1+2czHL-%K>Yl}PxFijOqHWx?K+unn+g;QPl4R#RA{!!&mhInq+?%UH0O4Q_kOVSi2$qr5*0G(!OTZH}AwS%TLJJs|VqmPSP5Jj=T203BnhiiJ+rP9HMP#nEp71ed-IC&%1BD!RZx^b9duqAm=9*_8<eGJd{*$=lku|3Eckx6qCN{p6YZeqO06cNU$c^g5)(FJcW2;ov@WzT4mxnqOIR29X?jh?C%A2>pTXzrdszAr*>y6O_(ZYU9tp%6OZ>pwGR~W<-Yr*d$f#9ry^sf9Pb(SVlM`f@G*sOZ#JfK|l_cy?k(Low<`Rq_^8SDAt6BbVSyCFqbLUNhVKa#|O@{gt7*od*vdeva|onQ1x$DdQ+0<`MUxdNE#^Hecd$F*Di8Ht<|^7A5Qm+dE9@HZ2JO0l(8Fuo<)k$R7aJxGe0Uit85dy$ci)!$$|zDMsLxA=jJ2I5w>BLGldt7Z}vD+ZF*9}xjd4v$jN@KK(qGgX|klBlNq&jCD)I0lCFz)Rcfo+3huBEDgRL>NoQ$%c%Qv%P3T__rxcDg9G%ve=0?@ps}sO9*8f!jwN-;XJu7FOiwQSm+I(L?oz^Z8H9ddr@n^dQ6Ra{el_t8z}kvb;lHy+wRa^ovxmXONI<EuG3VGERrkyF7^6RpF!z)U2)JsPN(_FbZT2iAfWchp6+4`u==xwQbn|A<!*uG*JRIDAbOVb8=MoPp89!1<!p4l4nB4cJHRyEG&UulSxmJa4zjjyd+qABn5IZYnp^sMkAJ<<RS_}6w~K+eq=7b|j%=cERrbal7uTpoU;=V8;9H-HL9WM;PtA3KrNGHpBvAQfU=qRgRS)G3waC%X(TK95WfbfD@Q?s%6fGMP?m6GO0pU|UiR`cNA#nmOq9Qbdh;Ze_X*rn|B<OL05{BWV`uqNxk$G-RiU+nQ-jm$*K1Ek^vJ~P;WH?arSDnX3D9D++r_)qT;aHo$DX&7)-#{qSY|22LG&Y`=R)T#bdh%f*25`Z<JzCW>y;G$B;P6&&F5ShH8?bnzqiR1Ugq*5s;gxj(|4t?luC?>8ei;RVQgmiX0Pnwukad^7OoJm8coc$0vW@2)WXE<|U4UQYdaGDv&&j$;x^x{{6M+o%iLz_qbq+#=<%qX_yhYf4=6*!xQ+qP<F{?Hyda&W4gTHM;I8!NQ_p<EB3k@NZIhbOxln%84IZ5D!?SKJbC!nEh-cT_f;=n4Uo>K8Fr7fua6nv{{k{lbo9{{ysRTJ;19+@GT;(x!CuRjxw?&Yj~<ugeDd}O0j!-De;Q_1lPR1mSr89euys(!J7T@Cno=xxG63hV;=0MVhMq19tvKWg-ZGF`j=A`{)y{Ai)K9PeIu6t{^HD|Q(Pm0?P@Ww3skBWnw|OFM<D_E3U-56a!lrK0<w;2K5`2|f&YAZWt;M&=>ow{a#&JZ|f%r>y5jsTP}6zeFsG^Tnvt3_y6Vv?*jn&-3_P?x;;q=MW{~O|nACxvjbID7m@EQ&VcdciKQV3oFVR?qq!*<0TWn7r9Fi)NlQ?m=WTM;nd-;FdS!*UCgZuKUw}Q6U?(inB8x3aRjl-^DAN(&1kxuNP+$8;|^w;LwR7Kv{DG~M@Lm&)rEa(Ndh?||K`~QW9dWf_Pr^*bAlGYp5?r{0hd!;yP+{`=r%wRLMEW=c@B_^8w$P+rdDuC8v5GD$!!AJ#kl-a^OcRPgSB^R_LY1|Hc+_<F#xOc-647wN~1Ad$h15|Xy6wLXoX~H{B|$bONVDkCRd^h>Xj<OhTF)*NN|~*p?dM$Q|g+$i^#c3B$){FNDlXgFheFvbvi6w6On?L>F_W$+eXUE6!tdOE1kj|r0}+%SenD%Oe5V{+YhQD<A-5CF(9E|{qTSjfNeA3+=}-YIo#bx?9hhDPyfH8LD~@0wmDQtT%++z(eO;T!GmMow6w2IT0d0;-ElU^kJq!I--1g#ChnW7UHc?jaTRBH6byAf&f&<{X-}VJv{vQR7Pha3;oLu1m|LPSBAJg@SuK9Z9wo)pim^igYCAtw?MR0br)mM>R70$w-0&%_Y|KO0fi26k;09D;IuYQCLpGT!`}A4?z;>&y+zrU$g>;Q8MR*U%_C9w0|883D^gH&~Qp$&fg8!&$M!Z;~`%wviCcvjU5?2U#Hml&lPw<sPf67!?^<77poBegJxV3{7qh2ax_Vj`7N1idppa1?ls*!O^07`3-rD&V(n6r}(3^9(xfSX*d{P@_bn%eH<sYz*~I0)dxgL~6@Aza2@>7Oyu7#;BsA17Qsc)HPkMwqNI=a2#EH^3=0-=O|?Or82wW^yj2IhCI=KTe=~9L7og@5pDlb>F2Hf=PJSjFtg7)e$qM{;Hv$V255XDUKE~@Ww)??^*(0bxc=RE3`cOHQm^O@FkH;WLF!WE4okcqUp{_ErvEj!uMh9n#1uUS@3!P>y8i!awnIEj@oM5hkdBDI%CkV`Jsrt6n^-Sj(pwFG)(R+T1;Lb%_~?!8D}QzU52G6DfEA|ZVVY>*+W0wDrTPCKD5nfSMc^R$FCkOqrxf4!F|m!c%lNR*Q4;*W{QH!k$|SwrftMHPD_A!55*?EhK4=S>6XWVqZLHkho}|7sQx3>teFUVe3@qrf7s_d4P@n>tHb<py=Xp)w4MW+PnL+c%Tug2d%-su2%ts%3F0%bujnBdqBZS5=Yzl~7Y42zrEv8_Qv^AGqw&GSCDpPrM4N`kSwO*AvQj!EvVFxQ&3Wc1bMQFgz9I@Vo4T)n0B1<^U1u`e?V{snX7EGFPn#M=P6{!#S2tk6hc)Mu@A;dstloy=k&0zXpo9E;HqZDw`3vKA6g0{gCQ=`bJ-PLm!Q?Wg9=&#wu2qyXzFAhuKhJ@{b28U@x^o4OnPWEd%g?O3AbAS5aK6R4*d-2AngYhbL3z83^eb)qce@3P_0oqU8<-kqs*^!fP*k&jxK$fMjQ`*q!Npl2@5Uoi53(`USA;MsZIegZ1XmdeCyr?Q1vlkIvk!Ef5zZQG`oO}3mTgZvMv*YM_&D+Kla?==w(TN7doFc+I4>WFaO0x_Y<6sh0RN}$#=-`}sV}7u_C{Vu3h7F0J{eeq$PlV|n|^MpSVGZKs8bGz*C`5;<a@38ch45SV%dJF=MU7>kcU*X3ilh1YrBuThq8Xj=g|))TPM%$#pg%K!-i~itk@<SNF}|Hj0^CaRMKQd8joWFzABCBlW|7_ppIzfe%{(Vb(;&x%&7oeIFyqjJsxPXMuNBD`jp&S4HN`X!ArI^e+*Vm6oEQp3Zqh)SNjlGsE&XhNfFx|O_b4}Yf5f<lw8re>pU{j=FRT@Zp0NqsyO};7T)9ru9EnzU@GQ%&Wv3-!@Fq3@>#vCwN|>j%-j;*4|2x`#m^!7*GCmpZ&=|OB&XqRVlfDf(1guL9v3pJFZK0MEP5Iy55F@=+3jit>_+-_;mwsLK>ArsH}(sUFGsLrHtVkpi*g=cnwSS=*>OyEG3{m$-;Nh%wbR>F(}(TKo_07}3C5`IJ)zh*8E$g#LIuk%NT`rJQw>FALGpt2v|B_dVVl?!iSJ}+3q)5P3h4yr#%YYX{jD_}6fzc|PW9@F`XX5iU{m9DBKA*8mTDn`g7>r9#EU(|u9uCarNd8WG6%{|b{e67oT{SdpUtDs28l^e!|cL<Rw{$eOFC0iP&r)Z5wYMxy-OP1EYHVTxBZrIE_MfAopyVQacZkJ_F&@@iB6B`J9z@*CS&O%Ilw<3Mao#lLO+`(MEmk3(VGe$5nvX>RH%MBVmo1!&BR>jC&T}0Lra0e;(<?XF5?Suz7lhW;xaKbO{z2ial<o{Xo%@LhM~dga^J(FOQe#Y6*2vS)Fia~l-G8WoT7iLr}aw!CcJ|zk7^&*SjjKfl7*^N9yqkf;*qjFtgCs_hM)CB;MwHzl&u^9OX>*ZX@=^OSVIQP=Plq=rPSKv@Uu)8uoyRu2}E$hO&=`fJ+KBNn1lzgmAL=IMG<m6Z6Lkt&4ArA98Mg@^O4O2jheJJST9wA1hj3+=a<T@nc+KrIBz>bvs2?hiOPh=sN{vC=?R*~+RBe06lu`bko8FGlAO`6q1CDjffuS|nAQJes<{+9U!enRC5mMHq(AX~jGJ0^rzj{{DA0!m-7VgsTvF`ns2?}$?$7S}Z66J5O>Skop{Bi>LDHQ8c4TGUtdBrhM!~LOL=U;xxvQl&C@OV6vxRm(Ohm3e&s11s<9I{-tJTSslMc^8VuFI7%uf)9KB}B+mx?b!BmJrnW?qg&9hG{f_W%gW-ex}ROAgVmPL9Gvh|5!eDZAP9_6n1qv@&+2DeMXETjfJy6LgryZ~x7-v!<TQw>X{n92^)FF4^_gSkw4%+}X|bC=|6V?Nvs`Uc3~#2Wt7q)@>y+9>dOex26J1-idXbX1;b%AA@}f`^J7xE?jB~hM0A(arK&m<01_wK{*IrVZg&^*4n<8ai#njC$5Y_i#EdkHd{d(tCStJTou`VOhCpqqPowBJd*am$H)cTj1k=D|7yd~V$?;I{=3t2^mFCdWXE;g59m2fktR^NS%O0^osA<dlPSM)A-yzX>CAx|Mw9KA6`k>N4Pt3T&*=!=p|#!If`)t+4pl8<rF%D+cIP=vd=cKp^FT)^1=&hY7gHx@t6ter3q{KLc=Z~)R>;Jub>LkN!5E*oFT9)9Vo)YhvDNILCNT+vtHQP)(O{SUC^H2-;%h|RB0ZEd`%kjQGcvKgiGd$TiYmkAScPM7Ad9Q8hG$pD3c+i7Vzh|JVd6C|b1q};ML~dwIy`?CPwCYo48eHsUiT$e-;lZ>ISZRMNng;ADU9;QVE@$2WEFt&(d~Cad?&Dv*yMvB43W4^D6J$8vXk)AtAIR$j_~C7Vi4uFrB!%WA{nI*S|g`2W&3B<k2PK=Cyk-S)4we5F_cQHkkw=66mR`KboFrv^Y**o7|)WA<MomI7etD0Qb-5Aq8~YI`?$em(G7@QF8A=2bzwuPAp;jn8i2my`!g_@u#$3u*+@F^HHK8+1*dEfy|>Mkn_3^o`d2~1I5nzdLoMfU!&^$%dI-$U`-p?&#K8l90D@@TQcJ-I^%px$8%lfq0}t%gS4@(Cw2`H5Wov02dHv$50O2!}Hz?}RWmo6EJL(DQp{=?#ymWbhksG3_%j<F%qQn9k$A+2W1CBp^M>4N$+rR~K3*a<$vdqngjL>o!mKls0iipmyyVxtIy=y!Q5WjE@kJ#a-oh-fCTGC=IG8@$U&j4jii?<Dtq?Pg5jchUFY#f21=Gh?s6_cBqHCVk9!y>_FoleSPtW)rf;e_kGJeYG+r-#x7uaS?F^K5{kYsGMMbYqIsH>F#XAP)vYDi%+bgCSu?<|0cnErI9IkQE^ZW$&;dwYcf!ZJ)z(nlx9c1$$-J-?T9^-A19M{gEnmq~%K509-lkZQ$OYXVhJH+(c<0JLL}RnrcYfe}nELYB~G`Bx0}sY$s;Wzk=^7GA1-C#E8e=HE8@-;mR<U$agmsLC-2Kz8hM-!8B2cn$;6sUMY3m4gdXLF1G(STO>6Xh6fLWyG(g-byXB99kOz{Il>2!Ga2T8R<*v{L)!r6`$EaAnD2}2K7osV&|ICgO$X{08RcaslUgvj!{j%c^mwpFmNQ47U}82?X9RBBQRM1!hBRgLnZ%-?d6LjqpTIg+CXy6Z_5b)?RZUR=pQpOrcp}8;jAJGRGxuFI<)XpY?c@uPyt1tSvJUU*yXEHUj}elym4m=wa46wAX>GF*Uea6@qI3NwKd>Tr&luo=b8;yn&HU0%Q#`dhGIL7hc;~?NZSjkfJ};c8wAEaEVEzRcxa=>ZP+p8%mt;GXDS;5r@-NMFymq03+nq4i-elxVWIq~SJ#_FvVR-N@E}=O}#b2oMo(r6bOP*(M=`S7FI$&wvzgz)j+d`)-`DUWVkUeMdDtQyUTtbOTeoP^x>jT2uA&)?8#DUuGF3RO(+d07pW|cyFSBvO3ZGGGZ`{_!m^8}+ZXnxsV|N4}pBwJ{xzA)NE79*yWoyGnTK`-O-{z1V(V=UiTz@CfJVNV|q;{UoHC%>g~&4aET^tIm|GCG<K5fYfZlk+_ZyUuaqaH*8mg?wa14%l_@v~$>!Q5p6~Vm3~)`6bASBx%gx{8d#}Yo8<)ifKxg(swmaywYOmtzZ`Xzn&5DO|5Py<!b@%+)c-ASN4(Vk<~f4Q074#NR8-qRfl;hs)FQ$=ecS~-r&xA=*Lr4vv(5Or%!4P(a1#WV=)}kt?*TPe=(ir6jTpS%+U%+;RyYm9`loFN%79?Hwx<rbpQ2zb<!xYD-~PZ^Yy+txBZMZ373eLwDUfByyeIj3$oUiHPi@3U<ywSXLah;Jy;IvsP*0O;Iyh>LvY4<4cVrc{*HZ)a+FoI$9I^Tow&M)R2IqzLWYoa90&RvVO@5ukTs<JhY^9Cyst$W1*Tvv5=`(=gE{g`!WaXDmzm$JzPDG9lk6Rz>DMCJnUBoFoT?@UsX8nzt~wxz73GLvlr(!EdLZy<$u^#`3J-M8E@I~7vFWfHAB`J1G)MRn7?=cb!*FJN$$_zjOWtX<;zl=XpPr9Cy94q?fzh>9gCNVwZsby3VKlE_wpK-sTP6C4k&a$+J4)WQ))sdn<60Ks{pRamRr+kA`5QZ?(kD5(z|YN4^R;)J*p6e{r$5I-nETS^-5V*M&s5njhpvyXg#BB^YPqOoTh{sei;Yc@P6(o^WuSvJq7<laoU*~0d*@y*g-OPzv{Jb;<^8Al{jphzM^kki&0To>$@cy5uSA4s#AIk<(dZm4=ewb;`#+Vfwkfv!AV9T{*{=@@aYnxuLxH8tkd}uWE&fRIsKPF9<alA$IH%(+ZTWn7wj(vKYW*00(-gxdlDC4;Db2x(8|TO_+i3|j7NY!<PE_Z)Eckq*X{M4ug2iPZd^myt6hou}Kj(@d6i!=_ujqlMqk?Hbu}nL~UWCdj*ggM?k?3LmjHuA{xy*x3NQU2<wPt3}YmJ+b#v|uFr@f#`Hb*zRU)TJJG47Fy{$F7<6^f_{sy-Jj*{I116}knj55pq7SU<vB0#8&kvMpuUWeBi{O5=Lg$DGa1#~A+vyhODs0|zg2?2HY~UDn@VqO}LJM0!RXy*!gS-eYYdzBz!4hzoWvg8u{HtR~(zWpS@hMDts?9py~&bQ~_+BFRJ$lZAp0V~0%b2PAAFyG)+%&Df<c=zn(IT9rVT59v-8K<0>PA6}b4fZw4$*2-47YCm;mU6G2)lQ<|#OxAxr<L5db!y2%0lv(fRTQuCWg*C3#v&hr{Q6-S!P7Ampv95AVpH7wjyBF3wmF|N5YV}$5>;B17hHcekXm<Ref3Gx)jCWAe<htB3p(yYIDRXw;l}C=SS+%<3apiy1X%p(=>&VZ%KVfm&Ovy8`f#!Glk!ng_j&`5q+~+!rD&+WJn&QDdv;%e<7vBPq{cm1Srp>`aU@;h$`twQ}0&O+E1Hz<e+OYO&TZnm@afx$8A|&16%8#rr$QlJ$alJh<ZjhC0@{Y+*;6z}Tdi6T$5gMDy4F5MDo62+^FF=c#7x9~hcQd_PH5tkS|IK7*(+h|uL)(1@Wtz=JI*pfN+-|NR4ox^$_#yv!=k5I<1JiSqMA^pq14sjd&d>t!=h+GA)q6nxV~6498M%-GVKVr*CVQwei0D0;p1|nB`Cykn_--TAGJ?WirOQ9Qt*z0;x`J~hTO3<3Nf%w>WW5K~JE%T11{c!|@Bw$F!8+%Fl;N+@1@7zoBA&m*5y>J>VNunBUyBI~`Lt;>woQv;6GD3BilCke7YaX)bFe>E#b`Ve1KByh<_JW;y<izj!)1wk_(utAzy^AOkvlHRBV37PT<sC)S(GYHx#pu%A;e7kv`gt7-}9d!okgP?mBvbsAT3J>u=@<OrApraa%KM&V4UHM7&Lh}ev@*JDA<S;p}+dG&RZ=nF?+ZrC9rUkT}kdA4?T)J<Wq{h-4n~GoOPJ|?2zaaZY~``EUR1_tS76X{=fX_o&9t|WscVjO$Ik}pXy=)w`TNRP{X(ASJFG<D3@qdm0d>}MVI+zZkp4yink6Ltx(S%OW5+fclod@A1ZQsjLbAat|iF1@hnufxbcrTVaIgaoP*Qej8O%R7KLN}@x}ZIv|u6K%%tPgUGmv29wzE?P5E^@7uOOBYtX;jK37Y?t9I3hLE+vO8m=h*B+|=%H%`r!u?knV;g3kg$+G&X@6^E8O}YB`N<o>rD+>K$wUJG<7h*?Simq`j+3E65YI+Vm{P}YWq}ki3cg@DIl5gf*?cFm;ng{>@N&O{qjX=x+4(p-g-mQ*L$|qNeF(m7K3Wd-j3IhU5oA4@8@HTw}qDuktmm|d><K(NI{E+kdcor@KEN92G6QOP92-u4-gsZjTRQ)*t+C@nleV4r0LY4hVE`u%f4}~ux&m!&!#%Fj^{tu=aFip}&azKM-f{rMPSU%>JtGfp+d-DnRA4H3M;;!z_5-irXLgIkG`NVOm)tOC@p&!BJaLnpvT@3AM*WL`qu1Hv$i_A-nc4AsQbN>XTRK_yMtaM=soWT2+YJDn4e(ST>#n8*s#R|Qv09|Ao`<L2dwS;`VY!^OT2MLMTA1tWq;%x8ng`_UO1K9kc(NqnuYdwwL|5i6$jl(scH)Jo>jYClOI3=xYu10kB<F-{=X1({|%?)$hF=t22l?WhG3FB7nnHP@4Qu-hcb(C15H=v+V*4_iWH~H$O@ue`2bR1Gju;C0LEmPvlcuCS6ZI`XiF)HhK3|mNG$o^L7$2+<#To*>iDO*Qj<X-jYDSGetF1Z7OnP*aHY*bcw*BdU`Q+(PxS;Kv2Wg8tb(0s?qa`%Kq-hO;8oB9tPXb~Bp^^P_rbp(HIOC}PHDMXvXB#%0Cv@=m|4p_;AEEZp6gC~HFT=PXMVz4x7exW}DUUrn{CjdU?Nd6pPB=lDodIkb$d!Lo(EugCzh%!2#DA1T^Bb?TW$j)P8AKowp=jNQqERV<aevW0>_f~>(YV2#zp70Nr00fZ#sav{0uI{WI&sU}VdG0OqQ5X=<<>u)oYH|67M-}53cv7sw_mVfvh2g1-yzBEIr^>~j{+zA&p0N-QCz%Vs>qGthxvqaT5`JZ)OX6^xENFOHjZeSggtkFhR+QS3=nM4Q@?`HT4}{MSlJ1&dAs_-zJa8U|yO~xYJHzXeJMxd`l%y|a5kX;)xSQ@-dDQip5X@ZlRv(v3@X(h=mD@6us6E(UqR~qXwcmeMBvQZe6F3aRE4;sUBm^1u7ug^_ct|uNec?wlewCqNb#&MIquC!s1_qDpNx(loh^Zz-_Bp(dV2}(0>*n$dsNsN8OS^%=Fp7kd<uq!&d5di>y?jYJXZu}qMS&b!gth=q_c&f^qknEXrJ6gjiE{c3`VpSuWU8%#Kuc#b=Y5HEV&s(}c@r~J{NhfVcy1!S;xC`w2E)i#u=g0k*s#yNzrnMdk!Eg@sDQ##sw5L+xtM%MyxOAT2pE>a8zuRK6rvh$0nuHZ;g_^S!00xWqen-v!UR8r_(Et@HYhZLyE0#c-dK?MS1M{T1SZEW+ECv*ZiZrhFE^7J+5A)UrKb|$lmH>L0~8RJ=z=NrjC|e4eZUb_9CPt4dv)Tw)!=erE2jtz{}jy5BSeUnk5%;7-JjC(-0s(x|EHsDrw&_kGsItE8WUSHS1cl>nd2U3{j!VYrr~@w{zNvl=WBx>t_wQmpx_VE-PX`?G3Lf-c)3`FJkirip1Ft=-I}5#xC0hiu}MyL>nMn;tX0f5kX}}8d`dSx7{<4?6{VZN=CZCu0=hm^V3T(tfeP*S2%_S%PYe>xTeOyG?kg)RJu-I{R&PU5`NcVg69(Z!88R|;Ph*I^)uBsY5Wm@fZOwa1mMVCmyNv8zoNqgO$_wC+Ix;GqfeyF?4-PCx6=11#t<cGM4#D<PE-<}t!G&51?|RjDx08Mz9}p<rl|ctjef*Ua@akVyJFACtpHTgJ##m4XORhR~fxQskl$@Jy`NtZBpI$2$MzLMonL87moqAPN!U$0K^Mk=E1)@4XQZZ|M<lg|jZ}<TsT=pz2J(>yW^Tr)p-sz;p@^M3fX`FP$3man@(S~+$Z!5du_jgH8hhf)zcRta#Z}4~X5qvsvbZlEd=q*4=bNF*;vUjl<YHOTecOQs%fH3mjYFy}9ItM@m*4Ua{AcOb*!B}qY1QgqBQT=k!3`SzT79Ta6_sO!|4Otxfnkkx#oK-=L!Y*Nf=^-@Nzu6#KpiHcT1IiN0IWW~rnmq4diMo>?CR6&%Kr$@Nony%9zRva{`6AS^id#^nO_c6W2%ZIC07{1r>Y+C~EkA9F*Fb+$sKMg=zw~t6kE9QaylLyK=8_}VVWI32S~df{j7_@3QdYKa=0$jJWrYTaAG&$2Ai>uPcZRjFWyr!9LwsAB&U)`xG@N@W9s}`rnKwitaZ{TuA7D;lt?czL`qMGaJAY__h92~v6lEL&Zg{k3-C#wYsEpjhZDxY6qqNJ&4Bct0e_ZwKbJrZJA7u{m+jW_46|B(=Pb>b7_cmFszd(!?=4z}{pUfO_ua4245Uwo%0RWQ_ea<U_ZVm}k33D;?i|n%wV{7P>XfY<GB|Msc`>3OBr;FjSWKOOAvKspvQ?DPgGPgZOM7tlzOiE_nfch|FN98bItjgodpvtgg&)P+F>TE8f8Ecm&)i}CDCa+y)(Y++19kQtWYk?CAD8ileg+MqfJ?a~u>sC6-U%67PO4svJaJZWr)TFr)sNtA>alDNduZX|>-*{cm92Sw((3|RY@-)W%gzHOJwW@qMJya`*7H1b$iKiC$wDEb|#*=TmcehbLwizizWVfdENw0jk`dF{v|1cYHh+!aWmg*g=rjxouRSq(w@XiEE^6rvz0qs7C5?1P2-ODLHf2J+qxgQe}WT8D`b9$~?^$)6G7Wg_;NMy`C7)Qog3f2CT{Tjy)6i954;TGD0g^%k&`$v35(TV$5^CJ5Ip*(=h<gV|KK)al9LvDxop;CPmk>9$e+R7~HUDbr)LF7nMb%r72!bwO3Ksmj6u=q*>!N~1yLLWHqbtnS58t74ajg&z;l?0ER0d-Jp84;70>2Ave{3V_PFQ2HiV9aN%Hb2eG7WO1<bf|5N)kfN?AuA~Nc0;ugnR>Bvl09swz$|WM<GYEgPS13IP6bY71>G3BoIBrG+rt+>mI%el{`SA)4X)imo;7lviG<F|Q$MiMfRcPmhyNwGh=2@s5Nt<DnCEmP0$lLWVG@k=09q+WXb??NM}&!rb>vq$ZqjNuDNOx+-!L^tFj}*qHV;9SNVxlLm2H6ulN$&2`vQG=L!3n?`DQ?D$bXXNp2Lr0+%l@CN|5r|zW`@4bjv}sv5gW06n#q#^#j1PfEh5Jpdjw+0xZ&zVoz`gT-sJdn=EiIrRXw5$9Qp<DbUy3%zWGBc><6B4P82o+Khi%EfH%qZ8|fz?soSbtn5;q6M6IMG+2c<@rt&{?J@=#t;I7yP-*C&mm3_Nw3x{!9y2fd03?RV?WELFZ{txW<I^A}k!V;OcdZp=F@d=8aMH(aw1ctpD4*H}xk6$v4tjC!q5WS1Hj5cdSOfTQY`Ec>(R9kjJ?6cvPmw*Dn2T8c6W-X7Jr^2M<azV6BP+Go&h+7{kAmc{g@`B0jd@#*VRv{~!#nYPawQ++OKSS-y<o+tv~6H0G5%kY?h9P6Hm<|yRj5ndE!y$?f$T3d&za0QNI~%524rI=IHdEa16fVD4;5FLv=>a8EnMCvdzJ(}8p|yM%fz1Sg6a++s^ZBt+51tYgZ9R&9O{Bb#QO1?oh~d(=NFv*N}|!R=5IeN!BHXi6#Z{m|J1~9{r><p2kMt2mDnbvT@(825=~Q{l7?ZgvHj{Y2u8c7gPmvd@cK<ohf%)KvBpdy)DNvXD#dljRXp7Zu-4<5?*3%BUuAw-Zn3EyU1$fFA?+^;Yuqwb2dw<s`~xLO9WhoVa(wX#=m-B?3#Zk%z>GaJippk)`n-HT-R)DOp2b%XxijT3L<KKhG3WyCUupLf=Zfxq2-5}w76lrv#e53Zuz2tAJ{*&0#Y#ldU}sF+m-{#v#47D(4uFv(;6sZraWI3<a&K8PDs<7rKPqTf#_}*4n+Rt~Bks?3r<(=w5F7+HEy}sjk#+k$p(?vy0)M7wT%2o{so>BGX?)l`9N_ow1(wkmGY-*M+p-?M$5g{yis6=@VVD-JcFIWeS&-NPve7@4ObI3Pd-+mj2)(niwFz1NM=Y5VzyMF86a$jDy7RBwC<Gf7b5NiwokTpnepUbFM*7){4dy#Um~xp-8pQo()1eIoyZ*POm;ZRj%6Rl^>dO%cl0wCu-kOBYqIt_<x_#lmU=d5+X@)$@?KSnB=qcuupAulvdCf4JJGwUQO4|TR=UNSCE}II38(M6RELe3G>u=Bx@;DA4RhF3G$7A?9jy||3AQi{$1=Js!;iLJfmZ-9-IYSzwzE&vpD@uFbYvx<FN3M-m?f#{_&{)kV{kY|(Nm09m=gn0YIWv&*8$82=B`ro%c-K?Pb2#H0eE$AqyxMC626>8E4q16^5Dc-dM)Ld9>4p|f^pB1xHDN6c>q1DUC{XKoPAGn=a3JsfnINsdD1)gmuJFs+xEptyC;&uD&gIw^^Bh>Z&t?sO<VrX!bu=lIJsK$IoXkYBCu0c48u&A%ZW&yqrCNDVEPIaR%enwVdcSY_$SDeST+}(RtN}25rik0bP^6v}!UVm6+N~26_k2`jy2b;OO2v0fwAWCdVM^y?@qLQrsB^-<O<&NzY9x;uEyzJE&XT*bT!F!uxmu(BKxAs~Pzaoo?z1?ob54CMP&tncDr!hx0Zlb6EK|g5|DCx%Rg1>1+5s#^R-?!y+kcVw_i4Frt?M9#W%+#TH>)YG($Qwc<v_&~7Osl6EdOJ%Gh+INg><!ztzGP;N?d%p4#&k>za5uA1NEo3Ngj6uT=u}oPv`ytO!oY_p-ZX@ZY9|CeJg-g(0ZH<8zl)$QG3979X3dDI6!|r=+#4dcu(FKdJ4UMJ%^v6KWPb@0k|)*GUOnK*VDUfdkEJ~WheN0Ct{5y>zrl>uw5P)*%6bPddB>5@hk+61Le1%MF*mf;NUXrOJWJB6RwOAw%RIg`@wA=gz`Z%H9;Jp>!!tW^R2P|{4B7h2*^y!A@rhd;`nx@rEyn-N#FU16rY8=uMnEu$(wHx(|UpjLnCKK(8+!wc}~6q$ZoD&n<ITf7#RvPX$G_<D#q^~C{=dY?~x|$GpCbiG|LAa^M-e^qmNZsx6fwCM6NBU`!l#U=PPr~7;sYRWljyc4Jr1JI5Z2zpJ`l8yuWg?W;DLDMjJs=*#QZ!zay<Gy>K(Bh+HT0hj?&N0h0lZEu<olH~*v}25gF6E1yxw%(`mO0OPFP>VfY-;p{MYVgBm(MMlV42xy_5!KUXp6zP>Z|IL?K3k@pYww#Uz8AbRvlwOA|oS=NZ*T}E(<%Y<;UMWJ&GJFAs)O;EM6~;E6A`lh}C`(-P($wn!VRXEHiGYcuzXMel$iZ4F5wwbW9S^}UJm1>_wgnZ**)E(vlU6q|p%62Cr`#>?qIGa5@mSpbAsDR_9ZG6o#>V0oA7==LpA5KkqnW$WLo(0^=OyAu1ey?ebpysBy!mv-vw_yxA`u~q+JDE|_+n~m5__MsSWb(<)7EG9D)()G=)ZxgsclX5u{JCVa*omZU><oiZ|m$WJB6c7IBM_lf$5}*(j<<xxtHZl(N+}y?0F<D(WXI~&u)EWxa>VFRSAkCZK@=V33E{BpVGp-2e10mV643pc_T6(sC?>H`C3&tE#si&A!BnR+C|63eu6|kq40qX>r%;nm=AT`FWr{|{7+cwtwIfbWo3De&5DiL5QbnVdeIyO5*0as#keR3f67Cv;A)guUZDfctr(XLN!%FlNgp9M<GvRUJ$gzYowFfmoxikjRX#1ijjmjYXe1U|xyF#k0pxwyF1>C3%q#BN2m0xZm~w}4(PZDmU<j`!OTqCRzdy@g(<lC`Z4pQ-<*BL>HW@IGOtVYj(E#2Lvei#t!cvA}+>wgq-o!YpLQ(&Q(H+=Z?N2>1Nj-KgUqg3r{enVuP!vTS_x05KQF)o{1sl3Gp68L3A3K8<Wf`~@Yp=c)HGkKgTU2{qlF{7|#eAJVeuo?NE9v2zn=7ugM04)#a;Gj=+mz>0*VCZ_o>?Vfu+41eS#d4cdMk&JSdENV-`^_-GdaB5!2bGseh7f+x$*XLS~Q$eQj(jPM@uFF8$fD8>qXmL{fnEhJU^1CI9XR5#bK7q_TjAy#js3k{brN2>}@A~F$~Q;yZ6w)b`j4y20}#L6E}6lzg`{#p018h%zm1D^v~4Fh$<n9vqBUz2IpW@BdqC!x{LkzhE$3ylS^J4ynEYEJr>9I`bCQQs|PMr)on2F%ftdj;iZWj!dhN!UN)g0VIanUI_orj1m{A~$G^$0lN(%T%M+u3=Z6H3M@JcZ1z*-48JgClKV$iS5%_YO*tV9-6l~-)l+JJ~rDC+Pe5uBib6diJ{-EB%AbeuJjyA-YD%d$n#c3!H0oby^_|?)Q>M8><%*_F;Y;H635}dl+3y=QPQ;<Uzk*oC373loPOU{OT7aS25|IH4v1sC_Y2dv|LuG~YACU5s61|;q_3%3OPOK!9Ry6dO}tEDX+q)M1n@v$G3M!NklUfxrqYzk^*S)GvqrnN2{mfKo5#3sjEh2h=hMu;m{y+QxG5YgLDOZZ?$ori#GlO4h4(blJBeeR3lQza(eUOY`ubpcE4{K_w-Tt@u_<T^$FBl@SFd;fT3q7~Mn_=JBGvBy?I6ri^^)pI0z2?nzT5ekKW;6kh4npkp|sj93Qs!JtXu4ooGaWyw-XWs{#4%W>^Qtq}fDB{(QS+uehl<^Q^p@_sPAW<?1q#R9_O4$PvI#|WHzE>kd<t1_ov6fDT*s43jIzoI@M8#_}my17#S#dmbEi6cq<gDyRr-|~SP~)f8Q%9tgzwAlx`&g|q+$&E8xa#5cIJR!se(Ae#1D;U9kBQ_&UQcI4?Wn;V#{YSC#ba2tQA)^POV+l>8@p)&V7P+JaP>o08SCQNvNw0kEhk<2SGf<yOmJ~<Nv@Nq^aXc=hg@7W^XHRndjLV8vrriLA83*dd?Iqb$YVm!d#FFiKz^D@jF7NfnUS(S)?_7tAO!~jbnmL-X;$W7%kM5L`9;Gv<&-+cyp6I$Bfa|H62OI2=RQwY-kRiZ3w$BoQ&wabeLX(!wesIs0tu~zH@*yEPMj*_<4JBX8m3aZq<%1KdbthK>j@05p0>pPVROkyw1ZswN|KrGT^@3apc#wH;{q`rKCHC1T1tgG_;cVUvp#cdblo`R1!lfjJ~47qe-T7H`90^0Z~sl~y$;2Lp#=oY*99op_V&*3>haQBzLB%*T)V3v{sF%t*1o_7Q@K}@uSz`_Q;d6;0j7GQg)nTa(*Oy0{q6LI{E4>P-U+(X`<!j$Kmv;X-)b(Q`P>P+NYZpS%XFB=wX?O*!wqJ#45%1qWLoooqlxr8<rlAX_T{(nXZ8H)XZmQF3PDI&c~Ivy`y~7C7crr#R^UsNDMdGPhmxmyJvb9>+l1jH0zHY_t{u|WCYWI$wo8a(UxlnQ>tWF}Og=P0O<MK|)H&}W=I8Z;8B&+7Hgi-??q-EN+9Jncx}bO{N)!17pIcU%)0=1vb8|;tKMpdFJ=(1@EqFCjr_VQ+0PU~+3Em7AoqBg*5Y5sLA}}j+%iK=3AtnwyQlRS6_d(l%|G|9D{lMF|bCr;qp1^Ve^1hPkp{V?K{j=EXj2J+R2HZdZ-|bBc6D9c6NS-*h(SGUOPT$23_^t6p@t<+I>v{_Ga1PZ@gZo&SSu>f?crOL}0*ia_yKJ&ot&NGh079F<l=2n`b;Ja@k)elay?FSm9zpMT;N(!To*eiQck$AYH7|53S`Mt3^V)bRx<}M31Q9CqzW_fi02{6pabZcTq(I`nsR^tzzabUG0z0WKuqgWpP-U}MRiT{x`OO}=OzdqiI)n40U3M!F&%w~2gR760D~i&Am6D+X>=r^ksM3lCR{<IactLE7o=)2XsvO>WayR;>OTt^3smC)k(Y7^7kgUOHkPPM*Qe1-p+9rr8796Hg4)6=)FLmL^)BJeXWc&b-=bg}!(U-#$5o+nkc#(zttLBlyqO=}ZVa$UOzyb`aMUh~STy^sjk1`th==w@?C81xu4<6nl^`xrcavM@?sTWGYbB#+iM^;S|n8?s;`2VxvT}HlcPEFb0NEmVtJaJlJJko&z=qoTPrxKtq?g`*{R-t76-QM{<z_Be8a*VB3i(JIojUzxm6_Btn35if6=lx%MNDyvJaml#+IQ@5hTazWkcqy0^h7aJ)`EH~_M;b<V`i1QF@n)~&T7XYu1*QLVvKuKc9snzlQGBOIrvSa@Zz53*7Bi~Sf7j5dxuCpA?z$*9o65Bu)8-tapf+p~srsiOnb6k0$SWlD>iPZ7akZJypq8=KdL3ZVZ&1e3#yC--vngdSr|CUCL|`-!OpmCJW&E#+@?Ut^s|IgnI&>#8zzEt!{Q@hM$I%p&_R5q(fQbm8!79z6@D#Roi~Bq7#fZN0gJZVM`913`ZEvTAl^fZn_ZJaaPVi5vlP&mM!CU3^!&~`B;y{K`5ENekJs!Y&IZaEs5AA)ltbf6SSTB~Douo_$mm<4|1P`r@XTkHYG38(7!$!hd9K)d{`5|2Tb}8NN_=po&JB*4Zo43;2iIyN!FyYCp255i!%=9o?Mp7#Z2CIlw#UntBmyRm~@(J&WG6(ch<*2G<8>CVy0=xu26E`StLk#SV@6=iyX<a_;bW`h<8!<m50H5XmVNee>r^87p-9>lFgPvO&HwedfZalJ|+xqgjup3C9mPKsNz%3L=KLKW2;LW$22PF6NV~L3t#mLgravxx<l0~>GJeHdmpEy2CQ^0el3ViFwBPXo(0aCz+boHc*^XXX81(zW=3yTCBV!3h1+`X`%4}${BrWY3n&Uu(nYeHzY!y>Q;=Z?9vP2rZ@gpY>i?=8(t6c(~359!$N#H2LQM&d$%xQYiMvs-MFGukDhvyB%LJTv%^)NpW!UNnbl#O#T!L}6h-w182^k@~IOn^v|bs+x&Je4yU=gtaP$ejrpb1%#OO@jAd2v1y8^%(nE4^xB-gh4-Sqk1ZN2nP^>sHOg<iy$C1KO$V9mp&c-3_C$8g+@LuWrA`}9H0wCS-}6p3lQ<}UtYve697o2jGg4bdh!-t6w93*q9Z)c-#@n%5{xh|(%M8zJ>^c2x=DmA^d_B(QT|cv$lN-VGx1G&7sX5|>mA(q_rd<zw1CF0oXGpK5&a2KVAKAJQ(d#I6xw3Hbkw*r?L^;Ukua}jaf^+T5ZI+!w5dUa?H~jn!l=?!yH`1duUyokKGrW?0p6v>H^+p~-G{g-1;{`LH6iYFt=>+}C7lmVuNx|f-1O)>90oh>eohba_k=_9R61P}4wS5g?e9gq#muAfmY!4OeHG@cpm^&y~d%jk!E+Ugu=o_jFyFDFUw6FD<l-L;{59u1zfmQ5t%M?lP9uhHp(gKrqVGjA)j;C#TI$H1|lv?!xn|rsn1$6SU6U{csd7{EBp>WD#>cExeg=HBtu$^VgtN5}fWP13cbnjvXQt(0UL35Wps<mBwCuib%y2U1mZp*ujwEi6tW6+ymOCAI7)CU;YWmbxDCnx`uS4&B9Ps6q(xm}y^_6-%Q*fW<_^b<#Y2cTte9>>xWTOrVQ^);UDXg}tG_l2mW6qDSATljZSvUzR~-^>6xW$%{K6Y@De(zpsNldMx${qx4sPzAn8-O7l(J$FIY2!Tc?ldW%3++ZiP)1dKj#b`6`I8rMdZZX~0FsYdw{tT#Eyx3Be-d*Qmkwk&wLd0pw=6Nh|Uk#C9BVaBo5|t>!m7ty3>$8-05wa|;@e7M+?f8$7^5B7-L|1rPMCTyd3iyo=ku#Q_$SB=pLqh?vn>6{r%^B|~ldCs;U|&XIfTAdmFAZpW89$t8t)$S~e`F=FcgH@n>wHsV&`793Q4!4{r?_lP|2E7qp;2+P|8Ua%GveDsYZS6}->Mek9~Y-W6J3k_EGA+RLFz)l^&%UnBUZ!=k=F=N(K9!#iS;CxP4pKhfBx!cS^z0H2KK$aQE4}^{>Q`v^zk4yzFyqSY!*{u@cTjR8k(d|z#f}Jg-0Ho=tmsxSNH*3X`F^UA^I_=taOPrB%WpZarctq9RKmfPd%A8@!`1lJm>fv6vJn~_dF8UTYP!<*;#Tw&SuehnW*DSFgun<NmW(bV;*s$97H3{Ovucd{6M!a3zj>}%zGcnDL_D4ozXwewMQ(GHzP6>o2fE$?k4)!TK@5G%#p&(Ma*{aVE-?7P>QHSocW?8zCmS@1av2reQ`45BAJ{y2-iO?ituT^2Qtu2y80m+MpiiUO5XWkz%kNxVoRFnE7#PC<PTUM`j>!4wx2IICRt6{WRN0T%p~8x-NzYoQ=z6UcQK{>rR8{qe?mj~;Yh3y6?EWjixk~O7od2aS?Cih(kcYbcr$=U4%t;A1o2{Q?2yL75pIj`nTgUL$K-^MHy{$zqyZTYDE~KC3-+Q+kn=6NPjY{=#YzANu!D0xs)ntPOKp@!Hg!PQrzIPbrT@QNSy%TdV!H|dB`BOP!9;zW;7KLIJ!`AJ^Dr9Vd<`o=xIt=5+~#FWqy030&--umKoR>ZGd9AH%)es=Z{6AQTkN!UO84tX=Y>JK5~T_sojMS%Zl~zB&9w+`UjF0LxI!;E+t<|Qbx#(ayWi**S6TC}1vs!zP#H3v=rIoBAizzL!ys**d$5!{a}~C2nI}`%yP=1kT)$Fr(o#9JVXJI?i6)9MyCbsG&IMB?uYgUZ-P5a}eZ)u%<$4+*W`$j{*ztWmO1DzTNi!d=C4-RJBBd7fH|5RQ#}h$SRtV^^T`$_=@V7xaqHd^%8FxKH8b{FTd?l?wh@Ksd7idW{WPbR~gBNaaloLTE=w=$~0I01P(Y9ezVJOmt`dTZA4*9ZTBQ=-Ave3|6@!i}w!gQ_s(#1=PHDO2EiosfcJl~Nl<?8{mz0_@^R`-Iy+r#m7V_c*X1-;g?1~!WZd&GQR4fyP(TvL?@9ceg1TptDqnj35%9DV6@8GIJFxzx`DO`B`x&{>Mhr<%_+a;oIug9igTe_L%FEv2!GIN3a4Eg}fc6!=xF&Az?CYoF<BEuFp?ixbSHM-mZN!R&qQ_Z1yNhCcJ;OSI-WBc`!uR5bnw;#AM{us=uEV`^u;PlM%)B#HZRQ$&|*j9B!my;N$3H-rx=9kG?7B(+lk1bg~(4#vYBa<K-&qXBzSZ-R#QTLtLEw@bxYwc2qB>=uw(zMovD-k&jd;54IpZF<g;@RZGO6Nj&iq$akb5wIxUxZ<b1TBbKG*6EYF8V2O-jnKjnP54=(4vp=9u9ZU4T){|^88%&Vz9|YejGib;M&xw^g<coiVB9K#_f1B79KY0F!^{mKfNHMVLZHqBhQ3|pc`zxm{a3qu-~9_AM9L%H_EhqAZ+S{E_Lo!(r^uMBz^swi)h)zUT5|B=8p<|IL?2A>N5UupDtDmOU1m&4GXH}<YmINZL|@l5Hw9bbWZ1!h^HLb_c!bp0cmAthVF63C`?=r{lLv)}r^r{GYKb)E>v&h>%rG}Q6+-jyzSfosM*zW07UII?K=03Z7Q2Uh`b8~jkprVk8GX2#t!#n-A2efE#Svu0g$@w9k7d=Ak)|ZZ5_x_`!`*UA=X)y7-jl3@_t_GKTXE4XBh)AB!fP_K^e!*z>Gm|R0zB?cLs^TZhy#&IyW6i4?Ug4@?1TZH4D&0=-2^yq9?*HfC<yu9un;Kwk}$KU{nAMt@tE8YWmA%VRmy&c6Kbd!v1s>-w{fsH*xS?x<%C+UO}J5KfHQpgyGWEL-Qm>D!wlrDGXhMnL8wBoTKx_{tKpM{U-8Ai3uUXa?g%wD&%75gSpk?Tr%msX`^mCJBNJcUgQZzNW10^*h(Yp+>ftLE0xPRQ4?7ENw-wTKRmd9>T&3ipj!Y2bYLhM)ff6XhF}8w_Wyi*oOOCZg!_-4mJn)%7EQIuh1jM!W6oqQy(kzIt9>xfH_Y?y@XGiA!dI#lNQB?Ez>j50^YXo~oid*z_zG&~}e*Z_9kN-Qa@93Q}cCEnxGpr$l?-wRZeT{v0G~(X!x7gG1Q9YcygA;|>PmHM`LUJVnPM1}<Z<IRW#lGe@TCEXWp9O^Fci|O4NJ)4O^WYp-A;OsHY&rM_>afv?GB(s(j&bj0%5NL8(y_I7{i#dI+r>9H<b0n&JAV)(IHqG)?j{xSe?vRe6Fn8oR1~?zL5RS81dV1(kl@}iWMcq8392*pm~eC8Ro#1~k2Zkr@?ZFJE)-#b{v*$wQ*`4UR}N6^W>M*Yq-sGsG_S;k&4=YM5(va)hZD-@!SWk30j%2*m+a{{Y^282O1TgH2#3q;{EY8y5xQrNCfaW)KY19-hmcB=M!*coSOEuQ`kc6{qEG>}@d}A)0vhk6j)+U49#bSVS!BVBJ}cJ1Pp%HP$C2-zCD3c^W}r$TBmUW?btD#HKndjsqsx&&SRiKea!Yb4yfZh9YI?BcWMb<;s*t<?4yw}3t1;3*z|t{+lCQgnOT<fMA=ALkCx`leTIBNDse*qQbNANT-mGrg_0)Ii52Fm!tnWVp(ZmtR@Z@N>Hee@E^wOGolKA>iZV(eO?|zYcUy2!EQCux46^b2=8B~b%pozl7TqmEcRH|hSX@k{8?0_{-kiWX*FGAjI1&&nahb&q4@XIOhVve$Uu{grmM}wP-NI%aHjG#Z~l(}y(6@w+R$K909%wop~Tgmx_w6PYtc}3|CpnuYB9X|e`)D0se%W?4GR*ncl0(cRo$nIHp2xOn*I?n}jx<)$NOxYw1L1|sS+tkven2f@Zq`;M^y;$y@1EOqAD06iMb-%0KbIiue{|CYZ25WDD54oG0%dUW>X>$Us7RILG{&+o<`Dhnwwfa4z_<$bTeT+at>cU9_w&#v3E_VCf5yg+}m^z!ijE2Nk<bkqM)1STJZCOp-<M}_FxOR}wl1Tz`jtI{4rDo>*$@G9_D7>*CW~*rWjS|TA^z^cPKCEMQ%Q$dr-uNrEJcS=VGUI`|ef>#J@rS+Xf6kf&kJb6!|C2koWge&dM`()<L18-wCBmT&HyD@bo0CR<?av&x+TnQ5HhhNK51(RnKmkKyo@b-`YEq5~q$Jb`4aOONLi8-ty0gP%j83*NNZhN{NU6HOCVw?1TQrP*_AL=N$_M_Vb*pK3XIVMkq?C=f-1CJdhpT3`r`s<#ALCg>v^2FTDWheoww|S=IgI-5U_#nII?dy6ju%)-CT^8v@6Fr>;TvI+0+EcK0JQm9%{!ya9*HA9G{k^s940g56mm!LyyEOTd~wDwb=H(oB)N-Ssfi$%O3cp~n;`?v8W6dTb|S%7=P9}^1WN|Zm;<L|x<jtR1Rk&b4?dwkDEBd!y*zFwCHvlzXwBB+)OWfbdyGrJ)|q|jq(VY`YmtM<1rr~6pz+wPJ?m_bSfGW{`xbN9{QvDn?$Q4u1YyJ|9kA$`JpCQRKjg^lz_&Wixx@z=baK$A<6%6_R>7;|3LrXm3E4ikB*B2fGiU{1{s`IG(%%}X|7Ra+Xg`@LVrKU{JJ?CBAx$KL_vqVk)1p=Uok2Z0zek;t3M}M+DWTPqFF$C)2{Elkr`8{sw_0L`{@FPbAoxRJGc^q;hqD?jGaI}z^H@xbq~PUL!-gQ27mhxzVC1hJU-nxSV%|kXXcei$mwN@Ja1dPGm99sKtT}ctTM5J6BP&=~F)sB^(}=TWw|uU+jYoWooLXx-$g@D#N2@i)Scz!4wKH*&>Ji;j1R&`-{uZPOkv__6o$cWuk?$JP1UYJJj_zrtBWXQ}@&niy&&AK9OlU!^5A<Yu4#hHy0Hzwt>ePR!Tn|-VO;|JQH?HsbTP1TwotTymUy|H{Yyx9Gcf#*dsb&u=9C~FS8t(;#!4Z;VDmdMh5-MAP*(hwr)`hLyJ`NY4zVxh%YKi9K%PZ*TM)}SlGGtN|XtyXKH4~)sILDeOT~P-0dQS|!9wo+0OBDug!Qrrx4DQXPymkxNlmWe9is<u4o`YOneji^;WrPSVYD>lAxnzHU*A~q=J&pOSSJ|REvGrU>o|Xt(WQvy!VOwy|SLKi8zYM`^b8Cyj6Dr`Y4y>ossB_d5jQ&IkCht`DOZO$F<cNs6SuOZSJjV;V9`3tLAy9w=-!Cgn2kyNAAb36i9jW5C>AG(1ivyOubO~I8{;Ct?fomBFjC9xERLQxmDk@5_zYr-|JkxmK)f(p3OI`_bn`T>|7KBM*=cOND-sh}&z%4*s7Xaqo^U#5+-Jd=Py3%4vV~k|E4);POd{`2`lfg#AyXmoCnp$GG#q`F}g1sIyUQSM6oUs*#$aS$Z$J^X5<0(OT=ezjL9-r~*K&;kv*Yezvl3Dw;gr#Y2Ye>u$5Ad}Baqrp%-DF6pX}N{;c!^_8xY1u<!GsIK2b{2A3^bNmU#m~}N4$gkK(v818f7q^b)0w6G1)?jo_t%*o5Os#_=|g@Dlv^6g&2F-dT+jX^SAWe2g)a;NDgiEL-G%+-OML8ZBhVlTk%EFU4Y3&HRi-db6t~8YWT3*x|ega54wm~mqOWWJ)3b`=2Lqbp^dsDf8bvW$zKFPnzEY}4-VQisD#GPsRl<n76?cA>@0ng@%v2jrJu9C-bKWp+^NnT_#&}y&}0H&T<<@FdF*0htu4LG*o}{=NWc}-H}WBLx-f2xJH?BxMQgb6+CN_7y*mSpZszS76=(qIQkQj&8`*;OD8R$&y@Vo1PrQO%g<C{+#FQz9re4V#udYxW;kP?E)wWaGO(~#*NBuo=%(-|O1HB!apyb7%V25ptxxP`XeZz5BWfDQWdzQ_DER-wuL-}rm(D=g$k6m3`@XQTqey^a*xUly}`tl8TKiu#*O%E`G(WCtYXo$)qe9n)u#U{K9H%%nlx<U)S3?y$D1BY=C5cZOy;2YpTOU*J9kmLiyO5|=qLRF8m1t>e^N^CyKpW<cc^{rxW=u{po{=a{c{BFwO^6=I0z&(22$vL^>im;n_QQ{dggpNW;OK`!PvOxVxs)|_A<hdC~5}IwSfqO%H6KRbF^*gdMudkw)-Z<6MsyM}9S?qj$?DZgh=5Hns7_ab^`WDVk3w=Pu!H7%~Po7Bn<XhVwh=f}v1|0qyyh>MAy3kb#BMF<CXNS;Hzn;&9xnV|gf`+>-rO_dhe-|K5El>&Rru>Ss>@tGeqd)i3>R90w0fGFb`3H+*?_gb3GF$^fqj0B*lfh4<g^j#L+;OT?*CoAG?k70_alY9*68E8Di%5he>IO+>*H*4WQYMPYtRXvE70z*xS)!78G(s`*KVi1%A1{S9DDevA@oFu?oz;KPCSW{!FJ{G(u{r}3V2=0uV{sMRuzoD$EXj#Sr5U*CFC8>~Msy@X@9?t<Hu9IfdtG;!Qw8xf`c=IxN6Ee@W%Y-**)_bZ%|XIau6Z6fuSvy^`uF(&+@we$0ad32OQ!}Vn(XLJR_BQ6Z&725n!JW3l%o&FbTq<^ko0Mb3t$wsn~RH&?(Lx!RvTjwAOq^^%E0&WeOuQNgVx{y9N}WJ#?^jV3=oP1080)K4Vk#oYj@w56q!|W)6{%l*ukab3x$@w-Dv&su_N1TM;x0lG&MCqm86rY3q~O(%I8o9Wg50EvLi%Xwwg8mCPU4;c!~n?DD_<q%vXR;??@aRKE?LtMqYVpLS$W3^LQ2^3E+!;DE;}DOgq;NPpweqIIdSy;CE?muR<oRU--833Xv~Ekce{Fjk9$}r7dRyinL?1BRuAq6hlM)5r<2(aV04zHD}>Ba@WEYTN{Z7_Xegn`5LAa>pAaani{yXPwk<vj__E>8z^L4k_CCM!V%YflD)+eo6y418W<|?(o7d0oLrn#k3xdJPs)ms6l`{0Y#zwg1>gau%9P;w+D0K-As$j?$B4%Di>cN?*fh=&zl-&;3c$zzXMx;cw6aCrOvDc%+g^G=5M_g9dvTesC@?MV@9m%3XkZIoRLctbIoXe5wp;#I1IxRrmVgUmMs-hKYMRhX#rL~N+ue6A>%g#GLBQo?0Q>)B+y0is)YXkc`{_|v$L#AAhMFLK&BE<)BQ{7KPthH`A13fz@A1T04_0r=+$xe39Bg|H0a7KLhe&Rzs%NtNP=o$yrUypxgl2`YD3`fZ6dU@W%`L3Sy@~5P*1=QWa3+_Bf$cWSg|PHe{q5f06@MPw3^M)&>6ZLwbgWOpyuK`lx*yjhhNv^)`x}0Tn?131ni-rqF$6jWt5`G*yj?kLB_M9<R92Tg_?;O6Pr9?Xz$2nViy0@TMa<(R8|@o^ETaGb;PIO(092ZW00F$#2!f&u$ZZaUvBYQl0ssI200dcD\')).decode()\nassert hashlib.sha256(_s14_body.encode()).hexdigest() == \'33eadd5bec8ee4598b15704b5c75ddc7441e43ea88e4e51524ce76fbb540b51a\'\n_s14_file = "<s14ab-body-33eadd5bec8e>"\nlinecache.cache[_s14_file] = (len(_s14_body), None, _s14_body.splitlines(True), _s14_file)\nexec(compile(_s14_body, _s14_file, "exec"), globals())\n_s14_lab = lab\ndef lab(deadline, weights, state):\n    return _s14_lab(deadline, weights, state)\n')]
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
