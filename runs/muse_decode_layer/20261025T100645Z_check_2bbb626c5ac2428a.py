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
    if x.shape[0] in (32,) and tuple(w.shape)==(4608,8192):
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


# ---- h200_lab wrapper: payloads s14stage3_payload 211eb22387d6fca0 on base S14 f1c77ce6 ----
_LAB_PAYLOADS = [('s14stage3_payload', '211eb22387d6fca0', 'import base64, lzma, hashlib, linecache\n_s14_body = lzma.decompress(base64.b85decode(\'{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;Wq{E{#^hOXc5f@^K)lHO<*fof4pu)l+@0?H16~xz2zWQERsfZ<fy2LtAQbcBL?s6oo!lM(iMCJW7=Sg+{9CgJrD@~vz@ApcNE?{z1hQ5hf(pjQdBs{0thh{t1|ZS8wP*Nhf>g$CGGbW`G=0M$UVkRS#78y{7Bp4)082L(mtw0b(fQ&`fyuL=Zuack$V^k001HIo()@$=taD;yN{debycx%zrmhxO4jSXDU!gQC_QV_AgT0uzkYP2qE^C6Cba`mJBqwd()q9o3*;E&G%4+}#IFo~@|C&N`9q3uC^LK`k3Ol|s!>i3T&GW4Yx5>FY9D<jdB~K|+yBE~KP`zwZ|~u0cUpUdrmv#{zI>-63g!oG7EO#vd)R%{oPe7)z8!w=(Hc+r^_2$%xTo(lk33RZ^EsuK@Q3I-b9jm+;22CYY*qbO*yZ*@Wtoa)^b3vtaL9Ol#+~bAu85Gz?mW8Wrxuwj{ynP_pr<F4K9gG1%}Zw*SbncHfO?!1#6RB8)Eyu=C!<>4o^j&wE9cnuj$3!@)a%t_Lhwc2EY;rMca5mGr995591EZ{l(6|Lfa5N;r`$7o*}+-#E}*>br$o&&AG0dLPkKG=V;CksTrk(cwccg~1wrN5c}$f_9$#>1oX~IPR9IONGStW7^tx^qyyHi`s^McQ?BIG0w9OazDK)x(3y;k`R&krBlFn`f&x05<yYL@LR{+ap_nZ%0;#0r2aEkq+2yI-dx8H9@bq&3-`BXH2b?;rrPBGlp3V)e(I?KwgX0>?YIzO{lS=z{|nYJxEuA7yLN0}73rC@To*58%BTsgrOLw?v5;$=XE$3Ae}Hdf@gpUt+AKy316mIP>jD*nbyryNe?!0ow7;^Jhc2m_gnASYK@iUjXUj*^;*``-}`5{9yfM&f-?5)GH|xS41~xs$d!Jyws<u5D|`otHdxV>GC*QmzHYaZ9l#Pao#yOX+~L>o8A__S4lx+KaW(^%Ri)w%HbE8wOQZ?s*K*SrVVuw>yal?CJ;}#W6nwAUS7A0EXI+B+4!iqh@fD3JbPh_wR&Ro=TGYD=Q{$ZkaV7L8QRwY5KEq-Nc0St>S=Le>#-NjRn6v&}3H0M1s1jC5flbXMtS$F)yuV7=E{1wDhKWqk+UZ{YMVGl{AD#cF~n>e#up&SC6OU2}Ng-SiOS;ryzAPXGhF5utNGp+qj!g@161f4L-r``%@$gylxawIvf(?2j%pi{2esOy0Y+gc~i;(okw(1ix%iZ*L_sE3uMqRs~wHBr~@Yel+nXh9|DNK&7|XuRhIrG1m%?wQbxERsZP#>Fan4KAR~?OxqF-n2C2lGTY+_K$}MsaA7FBOZzd~}WRw1%&{_q9ts+*lL$SqW$(f=mo+Rm7k}MSah1dfIQYQ0wMZ|MNkSuuomK{_M5E@(ww0!3vW=wdthhfAzJ8n|Df)Ux9s#_utLBk_ypl~o#U-)U8I7K4h#}zsOOOQXXZHOW2o^^`_2*UorcDRiaK&w{F8Wdk#;dD=CgHvEW+ld@+;Y$B%A{C#+`8(mUm!==2<u$3KyoRkQWOKM_dAdOafGO$W3x4e0k8O>hffJ;bUWK;MSKRIK1Hx)=3cDII-pI^|!fubZLuY>U>{tj}#8gWEcAV+!!Fav+-D?h7Ht?^X6js^P+Cw!DgBJeZXAc3pFW;K9bP^iV-cH&R`ZxOhR0gMWp%?A%sbE1Myvn-57x{S>-xk081+J#y;#2mKI~uSgZ32MfD~8UgM-UXCKz`ZP;X`Jt5VagS-nplL2rxy^3*B+m04}yvI8our!MlIL+`JUx>A2X1<J-%W_ZroO)8_H|M#jZjy$2PrjR%gx|EcuFs(FL?BOE9@#-E3~Ell!0WDwIpTe!Rq5j&El|3zx_{-iuLrATQ)8c=oB1;Nyc{T`qzkjJ=Ux5rOVH(`eV)ffr~(iuL+)}8R0tOViI3~=vtJhTZ=+x_qAOl`&M7-VmBZ6|d8aUSia=&`4dEA&o<y$GC}NLTB3GD{lN;}$UYz#7I0C8oiLDJ+iCeS;)n&{nVlvdZh<AAF#F&6whUOzlY4ZJcsBL3gt<%C>biEVQtw{fcyv#_1FRrAnHTeU^&gmB_z%gGB1|`EG~nEK6*D;+2J+g{^kEK?2R%<FcBhZrmCGKxNBl)c4)z7R5W8-8!N<7|mG@5m8FP>DWphK%sSklwk&#L8Y7gp(+h5f)oq<v<T2@_nc`}fZT!lidzVTrAH>HSX3h(YjcBsk(h7Lf%+ORg#do0&Zwn%vjK4}Hfm4xOe98Iq>d|0LkczrHU|jSwsB#nTT#1SJBRn$pK<DPQL~qw_bCOu-KWgknj7B}J^3$E)H3j4jmUWlyWvTEZUAvs8;b%hX&a@Gi>vQ$Fd|>kP2t>58jGCHcJTaXx-YMS>1KQ-qdRq*<PV78<S#S@2Mwi^#85I(8JF+$MQ+QNiMF3TYrqOdOM?l70=~ed0hZ`!qTpQ$`vBS(Ifwc#;fYuR?K^n(P}a_nH;I|MBrq&>9;Er5sSMKZfI2SRPiaL7O@ElaW&aLoeJ$ay6A{AE`xYGWNCa65rH}yKJAj3F@$R02IUKJKOP^i8>LAAY=mpKjX1(N0MaP*mteB=-h)-=OFH<h`hdW@tp-decpWok9$weL-S{t!`qOw-m0a5U#NS*V?jH8-Qr5?pLncC)imxzmeE$nl~e`fboWmu-EFNSSuZRoWhN6uV!J)7;`JJsO7Db$UNrsOE*$M8hPO_DKmw7@^;j@uPSlURK=rE09L;)9dx*eM=|kd1C(ujAam9k5VQ9sve>_%?x2{7j;-Xr<2pr88fgqtB>oc~W$hsgQmXVG^Jf_<JdqYD##a0B_EWB<Gmb_yz_7pL|8&yLTZ<y}yi>-uQx7qqUejy(P)Vy{aUOAqqv6^c6ira9h^m0+3D(#RmO2xe_ee+9zRb%|!A1`s$b7vIS>$43<Q(yP;y;{8>R)9{u#Dumh~9?ibAKSl9~JR7^@qRkGhGP&d3N@23<<eGLw9j8Bxb%dOl&Q4m#aY<n2`_hd&wnEPuCppJ&I>mv0UsADy)0-LZzM0^-bMmDuD*nu9Rfqr9|cFwo#8VT!HaSP90xRl9j+c)wa*O}fsQ#xHhLggK8^eG$7$YUCi@sM1QN|ZIVen$GZ1gM7U0kbhCio3h$g?>#-=wbRB-95SW9lZqg3qHNP$hf8*flRNwN=-$L>)Xad!=1|H`h2rg^DbO*m>qFP73TxzFL)i`#X7@2@zN!A)$Ulku`ChR?YDXbDgie;0cDPVwac@zkz?cB>?t+#=hBfN1RYXpr#$kaD1m<^kq*nXu@?<%#j==FH~#BO?BUaLL)rtI#%e;=T=`$qZpX0MCUj??3AqwSNCE>W>K#USOimPt#ze$9osa=7y~%HTkI`P6GGY}}HNy_R9a4==a1&#`L}^ELKwaua(#m3|P}Qf0!|U>4cL+6t7Ig}w5s`r0xw!o=X~0TtzI*Mu7xn3J-&f;w_=`@Fmb+P~8$!&zBR;bR(39z@7+eX2-ThWjNa@X+D~<1^8y29H6L^N94onW(9ltbQg(W6Q{lzizWz;fx|1bE0D_8=-LLdyNQJ6-f^gyv?K?vH7w=!kawWG*jg6R8h`FD-<CX;MB5L$IMj0A)B^vnN1oBG?$rAQHIt))0c%^a3zT^mYsl8;_E9uQd|h{B(b9T4lGnWSwP8o@=S_hgKgMFJU=LI`}htOIDZIDI{tGA0WJkBV#YI(IDa9z2=?E!w*$TH1s>X%gBeyXFQ?@TS$YTLQ;DKz+K4ZTUGo?!^`z4;Cs@Mdg?B1E6h3jk{=@GX{D@eKw?rT;0H`L~3}9g1Ut)L|l7<y5@<6Q1Z%{lvV;<y{LM&b!!`hV^i+27NiIMFS>^Kq`u}{@I5>zqR@wF-M2ceLc4q&=dQ^`Xp=LFbUSQRQO^AM`Wf2JH$cxJm6ZlMPN6s|!7v(Tgf@!9xrcrhmA={-eVW3UuxAQGP~1DP5y<yB6JBn~s)roXbLep=StpG_NJSEM+r8D7(s2746A}tIR@Z&V%F*L=7B0{Eok>!*=S5T*k=nqON8uIMB9H7yuezwUQTQWN>rEP$j91N=cdzy9lx-(1t9_M)lzqAM-084nGDNi3w?ji~>-<AtM2tEKly$T5Ecx5pbMd8tt{&`@!0-nl^#7{1)RJo~)O5eKb4lyv;&+~-j?x`O(I&H^6OEh~hYn295Mzz$Nba`x&PQ4uDcP(^vTmwTiuFIv{JW0ahZjoi*(?agS>9_13SXRP=7KYdq?EmwcIZaT?S((??T0%|iva@G24Ks~zYKy|FeTk@+xNuX<>OBsPU?wha6=LR8i4s)qleK+;`V<5|GveZwS{EHn4mj!WZ(b2rtNQ|Uqpm95NNCBylm6iGq^ONec9tCwQHehQR$=#@+N0kPUb#8#p>5pC;aB*-STm-tk-1bLQ$llGc!9sg=wO&-8H1nb?{=~#*Xa`U)5dh`^jKkEEe-W#OAjN2DLHm_4L0QGe^hVEgx^Y`3rg#zlJu+?#I*g<8*eb!fsDx8wWDh&J2}=$G@s69#Hb!HwMD@K5Uq5E&e4Tpv=HvW58&Ze0jv11R%0@rZ&bovOpZ!n21=ToVNh$b$>?=zYYrdsjFC3Gi;x_mW?I0?n`vWpqk9kTcn?yJMGdx8PO2YJ(IU$Z_O;9;6yEL1Q6G%T%QN<Dq+ql_Xn*=6q)63+lYy?Pn0d+gV4l-!ZKa7w-X;_t*1sB8~C2e!S&&)74do)F>sd8QmteZDJ-BH-6#d*{&rHw=>1#q_5Xb4>}&bq4Vs++7Uj|^@=bbWSJ5|sSCg*kK-oolzICk~iWGEZW<z<YThfk#3m*l(70mY1mA`r?Xnt6&<PT;`DC_k*Qn9F`q^lwOY|lx4kO>m0CFM<&P1nn=St~nRo)GJ~05qb!C8@MHbiIc~L~0_Iflx$W4$0-r=(Y>Eb#ZhWkqP)xx-8JaSX3(IIGKSqNdON;M7bI)fU6sk5tL0}W`b8mihaJO0`#hgfT+`TmcD8(+RBSWvc~gHJ7N;*?LVRY17C9kPZw^epcKW};~oRypuP)!v0S#Rn{`$)!D9ST-tzOGF;Cqes;x<cksAc~(D9T4x1gSyQFHbY>)Mwx`gU$kFmnja&@4>0LU2eJ$(<j(3=DS;DiM=K2~OK4t=fST%BRvW?4l4I8RVLKY|9e;tes8)#TJaZK0ekB1c5wSQ?>0H9;rbWA19vW^_X1jaJ6lyY_Qfjh1px0DdD2%1%>)U>^xPdle*+iVK7`I_nH0D*mCcSVN9zYOATOg^JBS|AEOD8&CY!MP+Ha??M#~lK<2X2mL`+DTE!*Ee*Fi*4>*s*V(Ek?e)~Vw!hvYy&$2H;0gp(;z+hVj0Yy=&reVk76bwUg+0yl0)CwhqJH$}%r`2nnpd1JoEg1HXc^$MuIS!_J5Z!=n3+Mfc7bz**s%LyDU$7t*$n=9loQlR_Wo;ZpP*2Q|mh1KFL{eiqG;w2@Q@<5!JjmAJL#)j7V$!X@ps#H|{+`Unk*OFbCS3>-=%Oqt9#D)YbA$?J?2xQ2maL(N!e}N}z+`0IbyJEYwq$cSzu~;)rx>O2TXcF#7#*TuzyG}4n1Qp*5f-7*VGX$hIO0O+fnDfvtKn3$k3j8BYnQMI?Zwl28@uP2e{7C=b3ULw`4vN)YLkOZ@8EbW)-To48oqg4{H&T~>E*F>UwBiOF&2>Y!K40W*qk>jgEK~-Wlj-vBK?A}07z)#^DZ7Rt@|lI)P;g~%P9H$!QhI`W{X6(p=xWNsG*OQuXwuPQij3&KSl(|1|qPPijO*H)<Sj3DX8i5bynAJFZimTl85T!`E&?{b>y}Z$ZM71jwR)~q@#J#MHO32y!VGWb&y>WHIIz}+QG<C$Z(5QrYWP%ZzOxv7=a94E<b|vZ9TvAbJQmKkpkcXB8t_pz|iwYT^|P~*P6XM7CTLn`z>#+8u{qubCOM^^{sP=5IcI(!0=vv3hkOdW*J;o+c!l=#Mf;yPUC$oY@p6ZU%~Sl)W>cc{Yw2FAIfGl6FfHKl~6*sPSdJWCLnFzIDe(Q%|O$%<`{~@$_JZrmz~DM{^DNeM!0JF-<d>djx7@RGVWn6$Tv%Lq&FQ0-u?O04bRD5eI_Bjb%mmENdZUv8|T4QdSA~M(bETLW_1cN$CWjXG-Xl6ur?&2-_l7r=&1(vURE-l<=3LURcX)I1BPM)?P5gmOZeJviMe@=jAPcSlxTi?J$9MeJ_Qr`Yg8n(qUl<xN6y-2Y8YNBDyceuHgD4`%-F0^cF~f*)WIF2bDb*R$iv}{3Zf35=&D=b6)IiVz;WK?jNihJjg^Gq>G;XN8D(YCWEy_wS17w?m50Fw*Cjx}hb<p>Kcjfe$&4U`N*m`pey==jA6&mxXOrATf;Q+MLjvgP*_Ci21wkw;31S-)C!)<KV?K^ooisWZi(3p{V?M!FdV++OUDw}Hv8MT94UltDF8iM+xldxtihEStK;6SicLEDfN1XN|I?7(Y+g8mgV+X&GP***Ix}+PLKw_N~+dLA}AG`G!Io{%qP9IjAC<rXCa6aELx_6r0k^YYP!}z`hY3{xV@@oD`|7QE5D10T#^MWgk{8y3uK)TO94RRD~c~4aN>%J?J)WyYQHLVjp{`Zic7NWo<p!>Gw7&Y54H9*SYG_k}p2j#q@u5R9tcAf+nYQt}5i*8l8;+-tWK|d>LFE!}W0>H6B92DT3!@-l-|BV6hF!7LSpE3H+(fap2m?X>P8{JnAD6qtRgn=j0^}h&^{R4BZ6Dp@_YLJfh!d4SaA0tD5-H!ayL8Kt&0iRJVg&vYoFdwap+x}^g4}=2OYl=@ncLRT7O~pUit`Kqy^&OyC%((9V_YT7mRw}-pwyS0<xqH8^jAap(y`8{CTCmu>^6|O^tpdByCaydrh;|EeGzn|$(g=MmPE$=5s|7*}xpIHsXrQSud~ANo_8SB{Ot~+pv_T;g-LZf3tr}Dse~9aEIMmcEaux)8O6f1~R+Wll6BG!<8UgAd=j+aW{#ycRz$vprPkCVgC>X8};Dv>T^`e`wEmt16Z%riblrhkJ;hoPq56nZo^ynpHZ#;vaI2E8E5z#s0I1p0q_H3P1@<AKMD|mC+8Dnml{P<#Wx=CS(Q^rofnQ~H|@z<THr~d+y1*~K|K+Vm~vdVJWW`jZ$&<!l^b4JQ*4+X*`vK}dA<H{;NA}{X_=d?09&W&*pQHHt`a#wFX@Cc0RUO88VC-kWVFh6KTX@N$6MM1mqf3Qq8q_aCqL1aOXZ1p#}PQY^XvSThj4(oXYC=y`v@fLnA_Z0cgb1tc?mvz!#|GF0$lgiyr_{RafjStEAccOc+aSJUm$=CFJ9jNrNzg|y)DsR9<XpHScRr_OWP7}T)$6AOM>KIFQ4a|_tMsKo6m59iS>C4R=+`jcR|Ev~!CHj=e&bkDT70@?NJD<xfZW*c&4VmjXY|(f1E78`sjb^nVM!s*Ewb0*coKdM@4eb8(b6_YiAru7WXQx0tKN?t`uPR6Ewcc0Ot*O5usxTI`Xq^U|PRF!D)?<a)>_q~YN!GvWdyOK_4p`T~M9A?7XXOm}D;ButChxukxi0%o;_Hda?HM6apT2fD_d2YQ;t}o3RBt%lg0sY(;m<o~fY|QAjn!p9*!q)oAYuEm_!DQc(JWZ5aYHE%i+tIu7T}Nwf;0vW-ATe>MM#Ec<&hBP$x+R0#Ntx~Px2S-lBp$IlyLhsd^6iYg;pHZ+<~}VQ@$>>md}g8XL6Fwy{X>c)I3VVfr9S}RtzMFK1u>3WkOv&&`jA+)@otQRwYpHPH!n&UZiEy)Z&h?Ot*leYY7Cam%5#jnoH?@iVNA9<z27YfWM!jI>M%K^r3#Wh?0*gVIwtMrmUw9t4`G_Rs#ao$)Lrlvi>L4G-+)16tT!CZ;FkFH_!J<KiaO2iL?Zp{AJvnKu^5K)&U~1(%v^?rtCqhTn4NGpC?Zi?7J3^M0scCpTQOORrVhCudLnw=6$JGeeE22<&24DbF>Hz$RiT-r3N)Iq@38f4xt+G<TdHX_p*074boZ$;gxuzQjG%rlmrhpq<=Q_DTtxnB24VLT#(K|&la2DuNj(N2U`bgOj(G5KBuu|1>7=;2^IZqhY8ZB*~O7R6j%+FM=zSf)jNrLrq3VpK#k`?mbE~q@h-z;++DSft8AR5)V|i({>LjRD}<5GqKYOnsIo8BhpF8n#vXm_UMnVz@?Iv6LDYuf@zdRj+$2PM#Umf1`BV19u@1G$p+Gf7w)tL$*-Zq~gwj8O-a6L>RD8CX)BYmpn8^Rb;C3=TPX>Qi$I%E5+!7P^c8RxI^$A)KJnsL=@pZK~Ohy98`I4UYuX<nVA=81kHbNn)!J*PYj%nF{D@wx<X$q24E0lyr`UZd|m(prD#==ilPhb7kQjR)7l{qw33VxX^hs=Q*G<!sEx&GeNme(>|YN466DbOGf0_PD^0pF^4f);O1J=DE(bZ+=v!08|1unhWr?qsxYHAxlF4t3X?HA7^0J%9xaAdUyhcm7_hF&pN;E1pvSp*C>XLWCj1zpRSvYS|y>%Km)@ul=0&o(6M87d!yEPa(qr4@@rR@7|$emXh>=Mj-QLu%!|HmDf$|!7Q<tbo&)60D@e-UMbl{M)6XeA<+Dh2Q$auCe<8J7`Gx5j3dyc2b1kg9KLaeQz4nn<)3ox9k!yDAOVqw<DH4;D6AFOuR>QDFy$!hpF}4$EDGuvg-b3+pm5p~ug8;i0+qgAm-k3UybffHSuXie>SIa1Kj_2tRVFm*<oK7---%+<&8b#a)U|$;J|(bvmZk?SEs8gC(x9RyDqtMu?6M#xdUl4Uwei!o3w#2}Uas(Uczmb$|FhIsk&v9_IdBcZ#yH46t4(uJ{~YZiqlH>BOB+XV-O7y-Ii9)z=`%|lBw+0nF$gC=ZxXEg;!IR1#^J?IMZNK4UXJ2Typj;dc8zN<8rocha2LeA2&dgfm~aIWY)$;=nP~y|FjFTJnO;c=y~XKsUS*u9rFnqz+iI>ZY>u_{xVH7dT^5&${&=Ww8pR)DzN$oTQ(Q=nHM^cKLLjvLLl<?3?2cdLT)+tsVJ~&s5jt>rBvi}5v|6kiQ+eZlIJQRcu*J&4MQ}t<j3{_~@{oCTg3&J>0lIT2B}H>jv=vDd&VFKHJ)Ah&`PnOiobmA%QZKQRs3J%@MFM)Jj%mBXm321vkP2AT@U~JLRg;UQ<^5`1`&yQy@1yB=I(csy7c;@D#SBJOlA{Vb6LdFaH`E39h?^+<KxUfrv=JOJd{x?JiB3!XMRchopbEMp?wwV&8(bB*EfdcbldqrA<R*k!NG^^a0uzqhh(J*$SU<IX7TzIeFC^}<6l1}XBX^~6jNWDJ2D$D>WT1FLA{zy-VpZgqCmj$77k-t9SBx@)if1MZHyr=9BxpCbnPN4)=nLqIt~|V=%dXnMB1lX5-hVbn_98i>-Yd|mPw788XoMZJ?+oCe2(Uw-=&hNmIecw*8TY{(OrL2HlzLdFfKE&{_Ff8u-@^_k4D4jowWwJu9-_fyy#*viV;3SP6XHfWbwR*wK9(3JD|KhpUrf3p7u9Snj-dImBaKWENs6uo&<R&MTh46ybIaMuAgqx(<S}~Mz1#1yOGfs$Cx#+E`!K^1m=l>Pj884ZcmVdJIwU_r^A;i{HmC6*{*mVf%6;%67|Onk7nWMw9@J`WE0^_gS`rZktBkzq9F40O%?VXMA44pcS<~_+oxub6gj8|h`lj0U?JH+Hqtihr^|V>MI<<t+aaKVkl-p_y_PwP*X3SvfnF91rptQK=C}I{LNc5rq=%MREX<59oU{cP7{!N(g;jE2JNEg|8X2z%5n_UguCS3=1?3woycu`Upv}&HvOfY0-zl8!%o|%!?!!i4D$CwKWdzKdutWKL!_6Q9Rmg|;V7&^gO)hi6oRY&3|%=y1kQQ|1G;Rsi&9NLNor5hi>L;q>Aa^1*st~-Wi+8DVIQTQ+Jg+W0l!dIdbodtSzbynf=xg0kYOXQ|X*@DSVEbQbtLY$83RXMHe!Kz1xNjyN6Lck{rwrsjjmJAK87zlR6H3>CwEK8sc_C*}WGNCwHnF%CgOXtnpH2|ghnb!5pZ66p=+KybRSSi_*SH;C|BVxw*t0?6O@?sZR7G3J5M9m~Ub!rGOj%^6O<Mjp(e+GS;CLVgXhqHFucJ)iy&xGc8v(DcwZ(6i~HI#?!4|CzPTBaDtCc=Rfj>MWYOqZJ@R))*U&+NzLK6A}fMGc~Ub>hvT56l=^T>fFCtsRV#8xes7?B31TDS=B_Ok@u%NH&A<<85+Uh@Yf;ZDP9@LZR>QhiM;ZltMgjDQA58!(z5kt2;x~i7fQwOwNoT-74dV)jD*Fmf|k({4wYi@w>~hZk0x{fFer2TNGEpI;i9*8`v>*Kh6L!2(*sp*>a|%8Ki@QWM{>h@MJr9T22nP4<09H`vA-4uwpeBKq_N~pPu!<+68$A1};|#M_1CEiB28;_ntrdgfXQ^lrtOWcObg`d_5Vg=aUR5(h`aRd|02dvpc)G{xhMl>r$wz;eC{2I)xnGi5N}JkiFD_4ja4^S<d9ZHITi!U*u6%Z4j?_N=9#ZQ(m|QTp(V_>%p=@GW<V-$9s@lfzZDFNIxs0xn>QooV#tX0pKCVL1Zzhsg1v~{|N2FBj43Y$!^z#xH=UfFFRW%6H%xjKN`N@$A*#j$XgV*eBT7}fF#%oLE`E^NU(&t!bdNJ3({JirMjLhJI$}MWbIi{uLCE)q|X*7Xeo1$=s7~uVruy#Lvdq>K;J*!?pKgxyI^rAAgX500M7DX=l{zLn6X771E)DwD4Z99eMc-u$E1ELKndK`!%op6ejOO^)i&@q<bPbV1eWim#2px%7ABKLD_Du&7K8_K4wOW^jZP+1-<Q@UaeT<RuN_LS1J|2l48Kmdw~d;MJ#6Q!kT0D#>9Z;s$;tigjsZjp<H3XtZm@vAy}O=B#53EW*~rg(-;a%<q_$>SOXl#Zg9{TSaVmH64G!odyF-saa+v132@!u8&CQzFZS?3<=Fqq+3=N{RKKOV)a^N<hj_-1~rR~I(efo9k<e*b{E<*=Rw1)`QUg1wDk04GI&+B2qrB9003EN@m3`lvTmxSOea%(>n{3kCeF#r<((4L`i7p9(FKbhMW2wo)du)Fh)UQgm)QIB5J(2?s%P~-#)1nBfKH+r9pT0Y4z`k)w@hgv!c2stzS98^C0jxB3wE5a#p3@;*VUNz;qr(ikNiWPuie+i?6#U!C>kjBR<6n7aoB>?5=<1Uj}TQM7@T~8dE5FQcTE&M&XPBl)P_!#2SAIr<Fjd&C0SAh($4o?=Yyi#953$9vPn4RJb>1q50eMQFww%18T8_G{d!P#%<>Bu3E94-L9kuAhyVes%4$W#yx#NO9_!H88<Ds&uOx9Iv4c0tZ5ur1#9ye=usL*V(L<-e4zVz_^^bO(g1^aiC1b7pwMtSZN1{$U)NYr4b;R?dTOAh;);noY|T_vV9>FUVjm^w46#UQA>7MQVD^+*0NGV^sm-$M+p#8Gu2Mw24OiO8?FpRXFAGNr*-;%Rt>GA6`O5!q5EKl?usEe2&=J#0&%z^+(R5QqptVyrl^Gq?OBy$<~t68=;IpR{V@35+i&_Aa+nSH<lcT{^b41ToAjPKXLW~?gs}(eHWwG(08`%=q|}-)t|b|>%gpYJ*OWU9>&x9)H6k+CC-?uo=`jPG$uwuy{FNuLamnsjq!Tt$zNiLO=GEyf9}1S^lXk%SY)myEj7@MtbB6cJ061P;W7=7V~hJ1t?{LbF~retfY?OP5u@AWs}`}FE0Y326><l!-Fz-%Ud@0WI6Taw&~1P8DXe77qUa*e*_UU)`E;H&4)@;Jx3wZglNyR^Hj9I9nyR7Ba!j7HATM)5<)<@CdAc(tNN{PJ!+{SRF3Ft}3)yDkwPa8#bVYGltb`V7bgcr_&pV?r#OXRwnAd&IKZfA1WKQM|5rF*SKrAm(O(8U!1`YK+ax!h3xT2qmIH$uN@DLpzvMxhtw#lYU6Wj4yBg+;Ev+-zX?4q7W9&KA8+KPgO0N__J>A40N9OA*!D*~(O_tmWAWvRJ%J1>kWXK3Hv_WqxM3-Cb3OuE-1GX$U%%S+m9hD!U6fa_O^rCc}ZP+HTw)t80aG;{8=oYtO34;Ekg8m`dZ*Z|$8gk?_UrJ~LpNVUhsO7zIyS%D&`w{SX&rP|L0F->8IHC@CXR`JG1eCgYhS_lTq6S6PH1-x!?1<>XMTd<q#Mdw;yFdq4z!De*6kLH$_N+ky4rwBORnNH><B6vo6;6a$i#VSd-bFlSl6-QP>)XYdzqY1j&09AjQRiwf><wO8!mx(QRfGD=%GN6L<T=;IIm6Z0`x(naLhu0*?r=VW!_=p&AS!y#`o~~jtCBd+4%02oKkciGNyG+;Fs=V_j_qia;#!}48ueR=>WHW9CN|N(!3Xr?T<vvd<Q$<Z%#6|TL$L_Te6Jk4JnkdhI@?LyC?Cd=1l{2LL3feT`sH3Tr(4rH5SfoPS#MA4L2U7>FQGBHVxKHjkEB-ehDhG-`6b^B*j7FpJwnEg1i2nH7bLIMqGzzoi3XI~^YUY5cPx~VC0F+$kB3Gl}HQ)C#og0D%2;1K6m{m8L^`CoP$Hs9#`y@z`O>+gI3q#h3Re{WqYlirk4Q<ittwxxfsfo2rCRoFJfFbUe>4xn{qHTG30bsSkoxenV*8gn6XG{Aol0btnsEIyp)l>Pfb!@QypRPU#?OwA3-%VFtQB*Gdrht~M3Mlq|$2Hxh#CuoL3J{MNuQJd;pK|duMh(dDb@#Z{mn0^@pi)>y3^`)Kf9@NH>v0<dZUz?Ut!YQ{H{<&DD6dxZ?n8y5AW{e-F0X4y99UBK=z~*fAjOY!TEADYh2ix65Vl^+2Sp;N<o%3Rvb;+4UQIMP?GO0AWTtK%#!qSiC+!e0^$5TNustqS_}-Fq&kv&>e2CsL8Hb7Si%sT!jd4a~rI3Wp<<u?#J-|9_4zV?A1%U?=Xv5bhIbI7qLJRm<lkT;7Tl1#~<j``y#b}>=;8cX@NJIn0&m6-@L*O3|SP>&>q&4%T_LSk?KuvuLCs+w(8mgJt&;BEScy7?6v@8_^Z=D)&4Cgzvi9?RGr(gN~h)_&=5XfmsDKvGZ5ll3M*D!sFO~Vf2ksS%C#LZ|^nnx(Y`*Yp+2_%rns&<O@tdB}|h&0GJ{2^g?hq)~*F{A>@#}s0eZr%Wpcn#v(2yaAJW-Ly-Rb8u<v+6J;r$Yg#!biwMdW6kro<}}Z0&~l;%+JUFV1I5qVJ`_&9<+zj)fS5zwMAbD8MS!y@%dj9A+P>ATsMwK`2IoeHCvPIx^D}XdgOo;Ir=zTwsqW-KRlTy90`Kq!@r6kPgHU$&?uQ*OPhH=EKopVjUMOqU49JXWdF&-_TXI-Zif5Fo>?WM(gHM6?Hp+3Vs-!jfV`6?K$B~Sh&-RI24TvI!!{V=6>tQ?4;!OZ3GH0trQ<Z%HjkxtN+RubM%lkSl-maZG9F1G^ycBbU917k_O+>r*XU{LE5>4Tsw2-L{Z*c}{bg(T3x|byush5pixm^mVk>AlNd0$J1N@suq_Kb_*-oB9J#Z@P7<!}F&^g2fRU7zV3c^#vo)pHD5AspePo_rZ`%&*krvE><aPmbd(ne-Q4GI3*?^WgDirB_y#MTDO%!bgkuU~%5#!)($E7SMN8e}f#Q_uD?c!kE>6gu>om`B7(Mt4}4ien~N=z{gK=4f0F(QWgC{TD=7x6{g2WAtFy9IE{g>GefxD^$muY?^Z=KJ<t*<yZndf8O+SJg5am2tBW3MzX`GN7M1Qc0$H0d_=ICDigZSPNYBXSsBhhVO=i#f^e?8R{_o85Izo$%LZ_dgD;ZHlWtKB8O~EK+4526-B6o4=F^4kPK{Q{6s&z=?OH6Dv;}Ytz@!-WaHer`cNT&|ubPMo9taAReh(F1w)VTYo{wmZb6J)a1a%fWnFM-9+8iO3`$g{%jU5I<nF}iQ1TDOO6RU{pb^#LzRCf6%?-$rWhb}DncXxOD=ho`cJ*tT`kS-WIrdvr2Ch5d?^~tj$`{U41sJVL$aNF)Q;Q*)<10StGF1dz~RhgLix*81tr(^Hrdzn9PUer**LwQ#bWA*rbj@1Z5Ar7V%0n`t^l|?W|wg1j#M*wKPHZud%@DT<L$M5s^O%od`Oab!o1_&M%Sfn9S{J8z1?4-Gouos~Otu1bct~0g<;a0$ZXCU!plKVWxHy?#iqYAexFog(_fRVQMs_MSAsEKT5d0UJgGGZaueO$ItKWr}{Yr}2~f^k{@&!hfGQu`yJq}%eGENt^f{DgL|4jmD*u^Q>VVhdHdkRO?4wa$;iT>6*a>4sKOS{P*8Qv0@)^!T^t%F5m>rCH0tol`I(*dmm&#wSQEVCY4PdQ{omX(uAsLdVNfMxuG41#uDZv~RO5&fQ(Lx`*63FE22Un#;K@YJS3s8OlD4C;zLV-zsBoyR87l$UL+>z3i`~rlsc;#*~B!B@OY)bxH7@)=BZ-k4Tw>{(TvInCjjW(4a+3D$3dO6*4l-r*b7VC0Tgp*>;Bu)RU1|g)hYzre=esYQ$alavj9fC75YSDsC+>u{s|nHG4y9VLs2+b8z%r7C{wXu3k98zHZ+$k*X+kobPH6jL${1$g)S?bsPbXCKB}n2=qZ?2-mlctTAu*X7|0|P%8`=%s!i=1kOW0o<q2x_EAS5K5gzhEs)ioe`X-Pr8!%`02ik29A(s^Dl;CTmP9YT1*>N27ar(^nkuF(hre_)vW#m{IRRT_Pwy8}u&(qFi~!j<p)IuJTx=nR?CIdvB?8t|M>t4Zb%6=$ig5Tg#FcJwcgESa<a^zoCyTQ+FQ168SfB&I@*;8XV}KgUQ{F2FN^N&nEaUxC4!`m_JQ3T{)rTO|vy62t3Pu|k4$gQ2q>0}^yVnTt!PU8NUIzGad2>{P!+UYkAyQYJ@*65C_@*n*ukQF#Q4%!5Zs}{6R#lpxo7@7icG<x^oI7xsH)R&_Fjb|?3?8yEUef&M#T;rmmTT!3YtS9R_BZv7S9K_C)vtXp9cCd0QFqTLvu+hsy<|f-kvq-%JghWdK4prkRv*;F@F4SLIpti79(bueLCA0~92dx7U*gtASy>v8tQmG5l<-Q@&;mO#Lt&e{INnZW^}n(x3fDH-_L>}$Fa8F<b@rG<(-`4nB<<sAz<zm+Fj1qIoEk<Ci}Kkq%m+&VS3H3rOe-TG2&Lu#sF%yNI~UWFcN*8C<O_z%TasoFNu!wrmd%swoXAFK+FvpJ&bOsez7FbFZ65-8`Hs=o)17tZn)YpICGnTF<9tlfsQ;@Xm0p~=8yx-DFR{+c4}pEY8+Au;(~YE^l`65hs#96P9yiKG@I$Dp9}a~_s2s{AGtaIzWjLbdz4jBiWp4$l96RA%9w|a@GUH;{L}tc|Cl^1PT|I?&84OQ51f)0hgc=osV8?$8@;B9lMSO6aI;~%S)QG)Z6q36;vCR<Z9@=Mwkc!m%1(S=|2f&|_md4OvgBDK!HH7*bf|$58&D{><2bJvM9y*_07}{7~pHMglqmk)Z)+ts`<W^$Z)6Y&+FE9cMK<rD3*yx)GWO@av-anW36&DH?{)3z<5~UJA61ydt_unfWatNxMw2zQZa(3XrdD#2EQaKW<=o7kW9Md<8(!SNYUd3WE7q?V-#Mfb-!JD(sM^Q7`eE68ffNiJn1PT&GiZ8}fT+rE2-t=f}G(Q@4!B?;WET+6S+e37&SyE-Zx71~cYF^*VU3_%sc>-Ot_PtaZF9n6{cV#NN%AS<RqgWRIjI2VmVXauf2{BRIQG$?Hfw#E4d&^3Mu%-B0$!|s#p;=0%6ao-uHJ4(qoIf;2Kinxp)loupP~tg$(umMV*rsuJ-Xks@hO%d=SyD_`%1y+={}5%tauSvEW!vFsrW9+0q>+vzYxGmUuQz{R^cW_lu;_iR#q)y)EUXurv-n&RzC+|EGFA6qvY{Ny4R$%VX789ljCH$lR3`1s5}ko<&K(4jm&;hrQ6IgU2o05QcBXXpKBh*EkmRyUk6u$39F<;WN{4BHnnY0IYDzn&&8a|<XUN@2jM`KxA$$0rx+fsa>eMu{$6`}oS8`xzoQ$i_HNn}tp^3%f$u7>_3!OlL8iQ{zuMVycg*fT`Q5zjj8iEAtd#Q5L1l43KHa-1*XWVAq&I%&Rwt%+jg(~lev(M_|G#{6A(8{z3o{~+J`jNVf8kNO?qPO-al^@~ZrpIboPFW|)|5xH_^N=+pxXwO`cgW=Ry(B5Lk#cCK^5NeM_G0&8Y8wlUPL&~YiWVR4H<%hjMxOq><;on2@AtxZ%8js;aIm4+S+5YCv<6tg^3BO0S^rwOijSs<SQ9|wIk<YdW*D8iQDh(#3Ex%_?!dEaDD(@4Q)`vBb(uoCyYMD#(|jb0|M9Y?q=ZPtJtz#iKO3l<41|c@Cfdr2cS8c7Vao1*T^YGU0J_g`T_z1lV8eGUj;}8U9B9T3@F<`=n+*&Qfu*HU5AJK&O*!YmT6NqcIN}Z}vI{v-1Kg3^;06*3)-yt;7io6^1we}kbn3AOuZ}Q%Hd(~{%9*%5t&-xRA;7EZ=@2b=0zrs#%TuTnO#NifgDXOwB}@hI;FRO%hZ*y%{En}ES{A|QLCA5J=F*K6*N=PGYP2mG3@?xQ@n70GM_wC5@P03JL2=F0BJV3Zw?-vzyFDEy1>`2q87-8-dbyk@rM=f*%-+LR9$3(!bOOCN`=8P~Wfq<;>g%B12~=AqMT%;!JK*8(-CX{zh!#At1gwCo)exo5=CH}4Zbk9tu`fw(md&HX``Q$oCGM0b+OOPo7{|kpTf)<U6HOQ~VpYKvtmkAp%zas#QzQ&Sam2|)Vkz^?bW9hWII<%3b}JM*sK`RAVsQA)Gex?XXYWF8veP|5w?EPvHlpY<sA11U2P*M>Zb%p@ykdCaTu8pZjKz5oE)B>mvw#n9+sL4^$w5rUvwUc#uCzX*Bs}2Ws6x%`Az$Fj1<eELQ4F#C6Y@1aH+CGf-1+PpV&0n{M6<D}jgtC?G<|BKJHGKi8<~ki4$px@<OqE<!^~BUUt@`OQrFkhU14xxibJ=lSbAwoEp3XnE=}t7{76;bhWkp5$Qcq&kwZ;K>J8g)QP=USTu(1mn!(qTiakyrSpSK-N`$G!nP!ED3CJ3y{g7*dqi><6;JQv2vW}#EN(502)y8s)A|x<Oh@8UjT)0nOX(!s0a_jzP#CZ&8@x6AzNAm5>CU^H>(Olk*=@qw!oMQpHIK6k5my=4D68tVPg5+XMH>&p=FzVN%33KBBhVnm)gYyioMMaG~`)2R5@V?8>>-1qTt@YAkoD6(x{_)n0vd#waZxW2y*w1wZsc`r!cMisY??v?7kjYoiCy8i=hX(uoG4wWZ{<X9c-phUB;sMb(-9k0su$qb(G2SJcn>7lwigwp<LLc~eLClx_3*!`?6sEY&*kf#4NQR?I;!7d7Q1dro%(MRnCzG1i+L+eSC~<Cv`8ghn+dzl){1XbJ4fY-Fr_~tDmlup?+V11FEyZOiF+nLyg=FoxenwzIzhO&V_@GM7Ei(Q$^fk+ECpSGsgn8wb?yhM>!P5oq*$wo9X#7WDyG%4!r*ITWmB_jk{3*}C@=_+iCTBHfl}IEq)3N@~<kHo1F;+izmMl%F_r}K;@Q?)xyPWm&m}O{U-8s#p|5WHDfBVJ4GTu)%dbeaYnBX|lNLXhLz^JWL4D|Mh@^p^7SSGU*YE|D0Oj>IE7b<YGtu>=F?UEP86(*<J-69ywqmSR5hfd9=H2{fAcHz9qbM{l!dX9H#6201?xI;R)7uUp~8thaEX{e}B`T41|>0t{ja>st?n~l!RD0Z^Nh0l&vGS<a{{cEb8L~AlpB$Rt1yXA-&cwtmqZTMkw$Y?tlr$@T_Mv|1VbN%;19!%M<ZNTregw3i?FXAv1n|RV%kqw14DQ&Zg)C0^AS-+C2V<aE!krjq%M4KU%n4=cB>?9r=*YS@<AF;X#fLAOAbV*ox<9<Q$c(2y0Jl4*oH_Ry7o^*PDrTrFrC3mrP1Nr=O+`?Uh3IM_^V?NANQ_U+RpO&;9t5@q+J)agnaaXL4_*4St)Ac324Kz4U-v*sv8BAhYd7l31XE5-=<&RQFfkj;R>I#t{NLb37D<{g$;lwhME^z-}gwk_{SWp9hiR#}bSR&is9c67A{eB?%`%k|uippJ3g6(-Y)$cL$vg)%%Z~^B<;BATYwTUeJxU5aA+dih*FzTEYalj8mA@;(_R2q;5mQRx+5!(%nVe#aRNbGq0J(<@~g{;##xhQe_k$VI-6pZ`g`{@04%uYskN~K_Q%@*M#=4%fz|BLA|v*e?0mf*!tzR;qB=HF-wPZF+u^3DFy^uT^!m@mt%v0+68O@9+Kd_t0&29bm9j@sZtBpe!foB7!OOSRs}DdVWh=(aUEWCKsT>#bO(@$}su5F|SDe=FS=Vyf@xy^QS@!686zEvm(Yb=cOwbg&xzNN2}}*K$Cp-3BKS1@G9fT&Hr_{FMp<U!+1?uz*#Un{v@)Y@^FOQEO255;~f-%2|+fOxa^EWWRaW*-Y1m_2OZ|^q32xlM>(1(O@gGI0T8VZG8;^={Mi8zTIj#o?p=xx}OV6LJSSi)7T951UF)0oezQGv8J3-Hb66&>G|i==wzj-VA@|HUBI4v(0;qqYLFNUOd<IxkA@VG;V?C=>Nv3Sl7W^%*jd##O(4(4ov=oYX`o~f7KQ?HL#Olsp87oQ<~ucO8VOU1VYqN`;!C`&l*=I|n`0flq~vH8_0Afpg8*q$K})G_5jcWgb@(U`NPTh@L_e5B$FH=88pwLh9ncha6>{Z#G(KBenx*_V0wrxm8NGlr6IwezD`d#T5dg8BiP$ERTb@?Fx4sd&xw`2fMMCx#E41Ksuim}po$|ta#r?zt*i*oZxG|W;_!*e9%=9=<5Ov4dXdOv>m~V4<k}$`)7T>q$Egzb7*5B8zo}gmtY4+z1XB0CMAKw=JkD`?O<>4L`;RdN6lO@OQi%W$`4N~(SlETo%QTl0|WaDX9eEe+PIIw&YEr;I-p<AO8*Z*FEMEdoEl$M*#F-+Ij{F8v=+imCh;KC>Zax|by*<e8hIzI;pNfq_nG;O8x(O8;whuWpUT%rY2;4cx>t@c^*;vyL1`0h3Ts0rumj_<l3+09Ge4tb(;vz9e>_d{_!7-U*5wFwR!zcBFTChSPhIm3coEI(SlYovUbU3bv=rz>nR*mwn=pjsP)GT?c>qH6R9B9FH?yhKGhg4k|R2YM>n=oxl*T^Mtee8eR#UARV)%a^^~I}pV)VjI=ssgMKh)X%`JuW1?nq+{l%8YMW-N*D!z(L6_Hyo7+Q!g9CwL3<W6lhD-Z<wM<(U}i~j!q{n)c&fs<|C85GUi<FSUd6c^Ox6cm(Jwi8FU);!L}W{RS9Hy7IG#;SrdSR*u8Mv7dOWKExZDiB3|jwFXc1jj@<c1v`vXU4JPAhY#QRHeZ6F<8m_`mDr!nbnZp%*M2$~ez2A$f7S|Xg6tFr-ig$2IOrmF*Iw|uBw)El-cZWd&t38<5=_(UF<7~ih7$;3rt?yJ?QKKAehG_jk@V{Z|(UJOlh)J%m+yCw)naKc%F<<okOYnPB_jYl<+ylsYW|AMC<PiZ3vG{@{GdpF~xOk+VZpwLpS{Gi4?neSvrzwQj&o;$Q$btSshK%z9Pg6)K+zdIv)(NS}MYKvzUZrAUg=^U$k`<tJ}d607@q>kCz1R`884mN+eZL`I-qt7-J>iKTenje5gsAuT1CF@jCC))x>`GQ=`9KfR}s<RfRyPGg2Z?Pd8NP7+E960b(URV|Lu~e_-Wyw6Ug_BB`#^|<jrdQHj`B(ImlD>F(Iv1SX+PcO@{l~v@@&v2ZOUKI<M^KNV2y4489?StVHxFcbdFp@!{jHf66$7L*>g)EA!T6=j?@&ALy>hVPosg2=p41*kWr)?32#oojAFg}=*D}8S5EwUBHKMhAz5ZD9&$EjByS}}MTJYr1%1U1{3Jyl7vTWD+*8$xm30q=fF~UI}?7AYIXmFJG4N$KQbrsc);wWvD%=szy{A5V>er5tq9hX`j(Nim<qc*1xrKTj&y=CJ%!R)aL><I6=v`L-{{Ke)gr|y&?1=wWJ?UFF+cRuFQE*(+#VY&Wp-#NHrXUkd&rV4XUq>D{QH$*AOB&dgR&<tRrW{fb0!QLcx?ziGzSMgG393Wwjiz!`rUcVt&${A9&{sa(a?BERzG9J#dm=8m#J~oGk0mh;;OL%w&YO$!AK8>lZ<606~^|uaKIE=``$CwWbbZe>te1N1tslt;X<*!fQ@$wdx1K-`4SH*hwDu;%8X}QKr&K=r_3#g=+{EPB~2OA_o0$rZV=0CR%1(yq+4Oy>tSwDGm=i1<ebsSXvQQ`;uh5XJJYbd9u;)}r_P@;SUC>8b443s^PAHGI7^H7iQ*q7CrZ$2G@zEHU5f8GsE2f}%``D%7vxqMWbBfJqD&*VB^+-Xk9e!6ghYLG`-!7ESE9YEiP7bO~OoBS+y92liHssA{ILHnl@2|#%2fCC#N(LUQ0b#bKG=s<13*bm%TkIF7>N8#q&grEO#(Iwn*S%!^|G=>vVKz8xZlmTi=-V0bjWpG=iX#67CX>MDwq;DCg-kl(GQozhMhu_lA^m<3*_yv2yj$8)f#fPI%lI*B@J=Fx&Ljg*tA+wAZ&x8#RxUh%t3m9K=0lnT)lwyWfd?Fs6VSEVKEp<^+e2_pMk=Wb?!i9{+i0V&><LSeL`KeVwka+C=n1JOrF&gpV|6&8v*;Ztwm-hq*2b+?i#b@`g90K83gr2B#ByvVzit+Qv^+T7{vy?^P%<-YZUjLG;wWlGEU@nZFh=6HW7B7CBd0LGTcF**C;sQFHZMydroOYkt%|yp@voaJsqqi6R`#3mVq|u-WRELT}+k=1dryU(_`TR2@)F+|S?vLcYZs*+C=37NRL_O&og=3E)iFE_@LhOSjRl%66&I6&sdGe2v^j+@QZE~xUuJ1~hSCcw#>$w-ja#l%@WC(9J==^gd3WuuXp3nEmb-nXkhxf(bGUBziz$=c;>OvbuPh9FSzE;};C1bB;!L+E@n_R!Y-y0jlY-)T1`=&i9PW>_7ca)Frmes8C#7No}pkH~wVly(d((RY)%Dwf;D{GJ>jL9<Zw~gd0gZ`;oPj_9L@JuRV>8&N0PKeD#5|pox`68BA;Wb1_RRVUt*l5fA9)3RWU-vi*al2zPcXPQdPVG|009)|_lQ_hByTe3}CY!qb18GVww1qvR03#B6&JhWU`<RUF-xNz2mp6nG5r-lOcNc<BIiNn4qq@G`7P&($IGMM&0;aR3TTGAH_Bhvl=&DT!%nFNhC|Sj^GA+B@Q~X50fW<e3DmABvb>wruruts0S39Pm>HaM5u|*ec>0vE>77nF0Xlsa_UxDxXIy1c*`JlQYY0Ws1hQ&PKJSl3VB?#uK9Cf`)eg(ptb)ye;9Km-3)w`j;o$1?A1M30!It4*qTETJB(SlUtne~k6QOCR@7PaQTG|%7hrQ&y>pByn#K!HE)hcEMk&^>`PcD_uPPSM(=UrOFrt8z(n4W@iaq=aEJ(#)J=nJdZ<7ma@P>T-w;8+Q77+m~-DuwT-Ii+rCcVlf5b=K=)K0;=J+(xB{gfe8@9x9m@$IK$nH<WIs{ld_WJUoI)i$zI+$M8bz37>!5mL@K-D-J#Y1Z%-W?n20I1A8b`A#V2y^_M*JK1$WQn>wlpVm1TM-%a6!LLh<<gA4^HgUYKtLIWzE^e$5Eh=qxaUUrufQW+r*ha%vj?<speYvivGmSQRHpt|&9!qGIINIG1syVh>_;tT)l@L1(oukvY)!rq9DaBg|lQbx-cw>n*NBI|3H@M)nx=9g@}TUpcMQWM>Y2Itz=Rrp!DK16;~5PRmXm*-Hq%&M>ky%?-r&Me%L>Cp_U?O@WRO9uq3#UdGA2)~Wk>42(CGkMG1U&m&(6z|mk2$VurtpDh{>8Frv6<OgI%_t9V*kUuWzyOMtlPT}vaY6{TS73kOr0?lxx%mNJTvrxN%fQR|(dCR~aJozgKn4!!I>n_B1abD%*X9Ev+rMFiYp=1Z5Z~^VPHC^5EyZkd4tuV7tm3x;m$XQ4jK=eQ}&A#wwIlU!pU@DU!4sj{YdCCicb(69l?1^#<7ce(#$1aogJ|I&SKndRo)C%(_hyMV5WRYohwB(sI&c?^>T#obefB&FEW9tTL{@Y6VDwh)3J;nV>Z|9l%GMdi}`^kR_A1lMiF_OoZA?SF$RD5L`a*vQ0OKMI&9c=;4bfD80@1-qN-`t}~AJ*e{^U4&W7>B127@e-#9pCt)kv>(Np{K1{D}4{zzyBK)-xTRV3ICmf`pW*?PTH_-dq^GBS%-tjnuK{gLrT`bud{D|)@Ni{0LX#oAG|+F@m&u_@B75(6-UclZY0l$Dw7>ilxO#YVp#nSV@txQ`Ihx}vz~liIQcDs5T7~8lszPAAx(U<>|5nHGFbQYZnZCPFBo@Vob$K}@&Z8|Hfn&219CB!t)Ao}Gq9IEtRN=;`-NLU@$eeX<zrg?3QSb6`@Q}*YoHIh#9<fj?U-GXUIZBHq{$=iy(w4-XGFuW)I3euW21#UVHsPA1Xb$AnRHAvuu~-GdSW_!MCM+<GNxzc!Kf7SMqFo*>FB=TpN*&XA0NYVy9Ou$MQzRt_YJjcuzK?#0a^7xx{tnp3fG%QBdmALVd*crCzxPK!y=M5L!s*~w-D#KtU*@A>T1-#K=ls|bTpF5&blhNFMF{?%YHkYOimyj7MYB$?I7r|m=YhYfQtlIWBfS?03(c&#Xc6sPVJz?{IRx=goCS8<VF(w6)WwOqeNtdO;p@6CZtpYdv5RNj3_>L-5y1n`8Bay2hg$2#RG9OS{f8gn31PZ`aUgTi*w;qhFZ|A+!<XFH!jgbAx;ste(TFq^m6aN;uLKsGh$#r>@b*n=IL$)k0<-vlRANp(es~suWtAe)t8JGNBJ93s^Cl+_SN^IERn+DmEGD^Ah>WmX?OvGiqrt1ELY}Dby!(#IN_zp-SE2xMG*PTBaM;{f!;yekTr`y_FzD^5WH+BTvb#-YDsCOW<$;dyEqsDYRAqGPbZQFjG>m`bikD{w}4dW`ki|J`>#yz5A?Vz2$U680=dQCwQYs<tkt_0H0b0?6KP5Oh1L;{i^T6%*#DBQvY`Hq4acIW$ECf`Grha5uC<&nK%;o5?U7b#*8kLd>k7ZsjdG07%(~|=@yr!_oS9<+PiPkfsXt#N2<<x+BR?V0(F~mebcTev6Ylq*8Zeg0ZuNCuiO6k?Ww}^@lYpo~{%IHIJ}^_98Dy$fMgP#37&B?<jw)G3C3XF^=U0G&cw4ppPH9#}o}F2XJupcEm^2_OJguP-tk7IZv=@kI^^>U4{-T?s#&o>kDL}k6J3~?<qnuqb<9-LjO#Kk`HQZ#jvDV}G!9I{rzxb^*eV@xB(UXM$r5Uv(^3vroE1HwfQD7d>Y_aWv!`4sh8--#tBDr7mWyh`v?t^%eawIti1s2Cr0Or^JlRN}0U%biXpN^Ngv7|Vn&N;z4Gn=7c!nUDDYjG@I)5pKQIUXIOGBN!&RlpSlDEs`ad$u>aB7ru53D#*n$8BvIa@~>*nWgB>0^@$JHV)tWy*vV8<{+Ok2`|`ZO0Y7$*EhKq>F4+!b;*#Ql@Tf$9%H0kE}%N7W_1ee^fO=p#$f>=OH<hX^+gWXAqKmQY&hNdhM%x1SS1CKCade@V}6C{46V|Ta6@WYz@e1u%x*kzR{ZP5NxxN4-_$ZYq9S=h;W*f3@(R90C!1o~19czW&glJ(;WBa#R7m55eV<*BaucQ+1o|Ejf;tY@H}PARRWqnuqtp6PKKu?KUXtXD9Bge~oiQC!OEfTFTKPD&ze=6-d9T=fjm2)>NfhWrBOFFm9TO>o`z&R=-YK9raR5U73Ee2dAp*aexC?2CK{&VOY}aRg`V(}ShA2aQB)r<-Fy&8{KaGVNJbK%*r3nR2ba<)SQN9jyfU#QcgJ;Ex4~uA4ys*sL+XL>_T;%ETRDUgZ30gCt67X5de)@O`&G1CkhD|w-(HXyzB}eXzMj`@DyG8zYfR8&tO7kz)EST0SkZEqqTzn~7gS)3Pq{WV{dag7JEgsun*ZJ@sO-ijHgw&^A-`(B>gte7gp$)}3_ZCr+2e?o1wj3%{LrI==4QOpYWQ8<7#kcg79!AM3af2EjxSor<F`diN-82;pn%wdwHg-pp^JFDuFl{v~F7P16UO?u>K{URG!I)py_}98n-~Rj;u(!ismVOs~6cSKQ(Z16Nxf66YO!AGK+UW|D`D@*$+5>tU;m=$s-OS2pLEhM22jnt6#p>N2SlgF9Rf7yJ!;U<W@i;25Jqk!9o0aHA5$y<dV}G6}KB=51*Vn+l0ya`5)rj_)0l)J);sjQb_9b^A_2u~^toTC}+~0IKEek>C{{uG-;v%|*eGETe`%OYj8Yh|znYs?c?w~AKAKlzI^xY$SGi(l`e)09!S}Z0`vrX3LB6+RMx~TD?LXA`pM8OIlsgc2r7ecG$fiE7dh++YdQ1pa@IV64e#3shT(wzm2*X6;$Y_%SqS@HSBXB!JeY?!?>ZFRfB#<*4pn|9B2!t>CTIpvwnXJ^r!YC|=$#^&2aLlaEA-7g6bo-QD3C!54=_NVO|px2o8#YqP(h?LFzv>D7~4u1Je-AG`Lp+g>cp2~{!XU$$QE$-?#qGWb8`I5S?2DSv;5g<5WV5m@2AFf1KZ$a_mNjCUlE1*@K*4V&IG>xn!Cj4OCUuhBNuP)D`*n>MVJFI{RbM)}=yJS~exg*Cx#X4*I*g*XN_BP+j=iXS*tq2TmFRku<3th5}oG--|ZP>ZpnMfh;JFZEo1>Ml2s)zw38%4$JQb=JJyPO04_&gk|SV8w{=CUt)nDVotefedJGdouSn-Gj88}9^kS|$hjzR6f>NrtXdy(-I;WH+ds>Wr|too23xMLM&^Ju4$YX7AJK*n+ujV-5Of8|BrfLyvse_&nJ7B*e34yY7JXQNQCj2nJzGXL@b?k<k)GIel?|_E5eXgIjn6n!g^0>+loNcSs!GB@75l=0n0)#%|={!u4dAcnkZ<JOll=K<nZ;u_O3i3cV~=9y!M`1T67Sq>>q$i#39&Hj`Q}=7>4V%RX_vBrbmvkIO8KwQe2SZKXZnNzQpO9=-<;huk5B`KTl4qS22_+S)coE82kLS943aFPElnZlHO1bB-T!AiZGQQjj_=#$o|f0v{ZP{Wwp`JsBsEk+lxN3VB6?fIr!~fx#)hLxrZmv@p394QaUILHaAWV2KnEX)?WjX+M0KjlRUtY971c906da6HvoJ5|;6ZKC^h9sYGgk5^&Bp;)%3Gkk(nqUyuO$^9_<K&ui^}^tUPDh2-4ZF??GU9Yb{9wd-OIxV_cAg1Hg`ss~7EArdiB#|$z4mg4~2$jgK<E>rHOd=xQ$WCMXau1CLdV-xL?t!^A&PzZ@^zd4F$)(c33a8nHYcH!lxhfN(*<PLKL-!H6ncIptW3!hIz+#s}hABe9%h=KQi28B=R@y5(h-GHQJZhe3qH?HVH3te5@!PpJ*oN}UN7JUO)MgjhnQJkzwi|NEgXgRiZeWV>FIPlA2DSOzzrCN=BR{%=8pG@mNjUP#{X(4Q>-v<rGF>;fzO`pY@*ayMG=OlbIbvCfQXh*D_Ri7g$O+m&$MngFIb@35lX|Aiga$>xc3k+p>oBAf|9ppBlADb{A3bK7@5PZ50&Qly8aB6}WcJdJ&KA|+*eD~I@<~B8M#tul=$j%^UF5E>xcH4*F>Sf#PtwLTQx?dxa{~OC{o&Zfq$e>|GC*w4=mON`caq9wRvMMNkqu87l$YAePAFTBi|NSuJlQ@&dTE>$g7JWsd;%C_%WAQ_1HLV}d;^FEM&5bc8gSohRC*0)C8wE|Bh~>L9_8m%C(n#z8)_rK4%psym&5MK{W1A(LC@rXem4YAh1HKTDAP0y)oZcvkZnOI7-R@(cSq4ze|9K?LBPL3lFQz>nfieufklGp3GVAimV^T`$<u0xEdYaVNX^e;=E!E)oLUl^u7V6OVofWJVEcF(gTAEg#kA~`I8h^*5GUpo+qDK>r*Vd>=0l$XS8q%+YZ^1!RD*F*!2_H*LA3_sEv`x#^cQwefWW(o$nwUck`-vW-`S_r?zctqYMVe*U?7`ULUKn+IjL_NE)b!H<H3*cAm@p;xc$p1ApXwm^?<uw8h+^Weti@|TTe!7E=#*sAT&J9dn*+FrRSVzU0@z~6W+63TPWHh4^uwb@oW1!v8^|lT+qaB{Z%|vMz3MGQH=9s7CVoH`g<&Cjma>3FN5^evH^>Agrn-_{oXg}kBu~PCr!-pvf-U&9`ewZIE}ihT7CEqWVA6JTlRY*5GNvmA*%9ON8+mo&s&u#=(TSw`?f0Ty{$S7JKpo6>$vIB2>qL6xuJB=@$royKnp*}<TaT!XE<k&n!kET0qdTzRf01VBE!XM8_CfF~9E^hTiegf99&O|II?b_rg09i^dQfbsz|ViXSiG$WP>}V*GJ0=rN?{its)`hct&bod%#BDEE<>*H$tx-D3%b0H^cLRHg^isNL0Z7Ycpli`-(?G(_WW=rgq6Y5MXUZP8(PmTSPhMioyf5@@9^y5(*kWa^*L%4$tR(K*N>Vo`;Jj)_MIl@u^$mia>xxv*~T!dU2%>w=Kv<E3jM5wa0t}Hp@zMY*!UU(cd<+)bMj1nvs3fW6k!RqKnM6NvFue%Sr?5FJqlR(RR2s2<MK{!FofPUjOXShr-6Ct=qH0-`TXFvA88WR{Z7&p$@zv3G8;7Gu1f*0!3L@dX(M`5xJ~((GqZr(LG-;kwniLSQBD{09vI-%atS!s&*aeU+>RG7$5M+`Wt9^<zs3r{vzl3U9L;pDMrktA1lq+QOvq%jIM{tvRk2gRfQr=)a^YR)Ew1{td0E#;81_1u67Op2wH<3aX2dHjfV%cBy|RdJD^F#bc>^z-Frv3X?{qKB1$@|_cVfeqwQ*9`66`hpH#s$xP?H`eGV?qsgdS4pq+iL)JUZ;qtEh#9A>x`#_g%;H;%0hp`d3S(|9rki4P)-6A7OD@Q$oHpG_?dCEHrriZo>(Q9)gd56?<#kunh75MousKLJ7u2fWWt6AU3I`))aN3K15`B%;1dxK<X15R|lEo*O%P`<-|GQ+dM_vQoO};Z5m)CB@m0aOA0ywvs9=h-<jJA!BQt>E!_D%Q+=48l=KJVuI(L)yRJr%8rZ)TZ%|*9E{;ub@gUk75zUZwHEbTK`6vupCdqYq>i&H;c1e(~BmvKk)_G+VWOQ!;)d@i<bQJ>SFGg-a_FsxUyyQAbWe49gdKSxiX9`aBGq0nY(3%WRH(rIUCHpWPKy?E(_6jQ+VzzQi7#`$4`1WqJk5_2f3g@SN7?uOw7R)I;s~Oaus1hTHV<ON(lfaQYheyj0vSTlSJD8Z$-q<Vtzgi02d@zu#>DQN2t>}^p)CvDM0&{@<WoQKYC1>=7?+Tl#z%J1)(!}NX)ew4tJdJ6*;SD3s^i#g1j0#%rX?}AF8p(}sufQD=cH>2pVr>f~AXLf{1D*GIKV+J^tr&3CRJc4*KC6wTg6;J6yqZR62F*IJPZn$Q^vH={K(_KLW}$t~C~%N@v4|Cc_l#3~mf2m>Z2kY-a@>nuB{b$0^mW}?*}VaQyd+A^U&@A=5P)5j2X{7#z08p)NQ5XUxh`sY8*~}^abDcHoMQ<rEwurghcKWRvA%k_2c6!c;X@VsIa^K4ry{c?-N7WJ`#%V`U@Du%XhtNmgU&KWZv!*H#-KG_G)^9&ZSc9n3K>a^zHDTA=U|{reAJvYw}*aFV0DboGLlaHQZc^c-!!E{doMbNCYlQDiw@PJe~M>>u5V|G8icj^`z6mgLyTvUei0xEzFSnu@w7nt2R!pl)GoGPq!y5`gmp3<U7q@f=NU!8w!al%7V$=D0-wesjY#+?IQMgNT1P#JMcq~!1?XW=a2%%|wNUWDYxOI-K$MoGwv+Au8iL`yW{7VD;AL#Gmt2{v+c?owLXg5#-`@Wij{ag~jjpx9^@3Gt=}eddn8N``gavM}3;`k+mfUy5sf|?Wl_w0<q|x6NEYr2PICcdF*6*=q!K1bNx;IT>^jl$ZG^JRi-5;SgMtTv~`G$iMNK%=a7@Y&jx?`>%INO1zqMtD<TJdw?Lim3EnM62mD(q5ahKZ(G_;Ks*R^Est%`;pmrzdkf9MG<b^AMcBc)uyAAK#UI?+$t^Kfw3ZmcB4X2DwN(!7nbTb}0=IxBS30)^R#4TP{ka1I+3|X7+=n<h)GNSvcDLN@u)qS~NABR3FxEe_4cF8p7V>3p!%w9Zse5F-9I4*X1pw{;C8m21)lwVUrhA<yGvvtdU@)_w<yam{`uJZ_+QHm5*Os&e;=Z(TI<`nSq<bnwI8#{iMGF$!$Uf-Hw^*zCNQMOtN|o)s8I*UBk6W35}Dv`xdvxbyV`-BzIr5pot0+C8P2`)H15XxmEZG`vDdr8VO=Qqt`mc)J-@xc3;29G*^0@VDs3wIq86bp#bVXcT~!_A*Xhcc_svLhRYG=Po4AY2>fiQRspIRO4Tu@T@V7nA$FkLMcKUv!xV$Ass}_s<WdcGkE155Mmz(8R?B)2uVl-ysHaHET@*TgO1@ItgvSmTRJW6%yu*kgU9e$OtS~A4gt;ZdXV!~hk|lBrsUEr^UB>ZnHy~uDdD#z01*s5s>guPv27D6e{lcb>I#jSbs1vNr{<?k%K@$WWkIf=aFuo7IM?LJcN<ThZXwI~QKAe&=uoo)Z0fj}<2kxGW15mzhMJb_lm)5+2#cdl_Y(h2UqUX#Bm_gF_apd`Aqh$cXxg+1W0Xy?(#3Wp~qqMqJ9Z_YFA&4%~>&{Cbgtj8+0v#LcWlwu;dS9*W$6737_B9ia`y~JTb8pJ2qYsV0NR4)c(u7CSwI-qV@m&lXS>sfRwcQ;ItR*+OYs)(Jp|i4e6gyde_9ghq*|Ufy?W2~I@mKzQ>v$zZDhC4Sw+!lcb2ZaN_{RzX<b&1&$s4%OcGOKk6w@ffpXz>d&3aCX9%In_47Ii_=h0CM)g+5-&AEf~t%ZQYdq8_S;Pu`iMR`DO`6Nj06=-Z<pHK^^rr(K2N6fRONRrNTVN)kLqJ~hfj<nNGhJ|3vLYg#ueSfSn&)d@rhF{yfIwJq47&DEX$&44jA>$)oB`APimWa*eDknL2<*J*JiDeC;B|>1d6J!(&RDJ_zpYP;EEG=-tjo)U!NufRhn|i`^L;;Itg9o!3EkhJp3<tyfX^^T>^1&BiJ<8KokT8v8RCoF|Ko3+IDf^Ip#}k<LUl056H`~j535+B5t*DaQ)p{?v=&{;fRPHLCk|@QN$5qRjkS`Y52027MIOFu64FU>mP2Wqj`xVGm%+XbJXEE_1kdt`A4Tog2a(7|?Cop&y?^)Sd#v)TW3e!niolJF16l_ma&o1GDLB-9eB%-_Q2Fe`ro1-!UM~kXz;D~<OiuHM+u4lmjEH@)tTx#m}>_i9?lwuHbV#{BXW-0H5X)DbRwax-Nw&Srf!N;|xhXBhGX+SIf*|0$Dilv}eYOEI8SWZgl=~Pjnx9dd^*y}p^(9@)#S%&QTymNy6XiDAOM+2{YRCgBz_|wVW8figjQ-wMZwqGx0?nbw+4~?Lt$To-+VPa-N*G-BX0RgMu3U6+h(T(lwJ4DH0d=&FtPOxLifMyGPi1Egm1Ikb@ly|2^Mpd@;<~MziE{OwsrErWb!yTg_^gQl=4x#fUD5x9B{K)2Sh{YZbPud8IPlrJ$=msrk{|y!Ji@%iVx5KmBPr^oCa7&ULbakjgcwC_AA)#z%F`il>b(_ym72-LPBOqd$05xPKhX$!t%R73hmqORO1e(`1jGVk%oq7VRi@LyB;&uP;v6X;kM<S-jUk^$agS2c{(=>O$bS#1o7~if!-vVs2ti(TE!~Ifpf(t?arY*KQ=E-*lZ)Zq^w{gqfJq;ALj$9OPPmook?HfWStIx*sP;WsmCUUs-<J}#iO4$zDKjL>=-i>0lj@>WK2o{d7;{DHKb+#F$#F9^RB){;8+(&9XZ7DY((g;AW`;EUxz<?<m+mq6U$)EEQSnb^$J4@_iuSPfNx!+8*!sXojv>0t6hE|V_dL<xUquy)p(MEZS>N0)>eC{ej)dQ!hmKFKS8h&Yq;X9%Mn2xJKd@eMEXJjZ;M!DPBEe|sXNxQ?Mna78n9o)M~a4G0?C&bQ%%P5<jzm@q=Y^|faBt&Aa&ec^gUS=)OD{~!ygupU%(?HRgFb|(`lg?0hRt3dUz!9%hf!9+D|1`n>z0DD&N~^>cuFP~xp%)I#bL1B~E?i}0fs9eri!lyhW$);Y!b;@2t`Z4~NUc&7BgTY60YfI<HC4ym+|SVkb0leSSGax`wSd)2MoE2VZx_+U`{!`oD7P6)vTcSXkyM{XO>|Z2VZ|I(`3E1}$plK|%qE#8<=hf#`x&|kC>NT7ne4sfSpY*8U~Rc2LY!&UT0;gOY6S!ym59aqGUeD1VCMiCUVno@G3mAP8bS6Or5-ZBy8tyd_yazQ;}+y1MzcJ#SGb1_i{=UW_L_YXKSF2Aa8{=it;6@v*~r;Y`+A$-ao@!Y6?jLkn{(l0<Demz+I5YEz4qs*?C^v0a2DKp0S9C>FUaNWAwTUgZ`LVXK7L!mTg&^Bd3bdqBJgaF;(_9(<l9HuEc-QZOy5d5#w3sI*`af-9&7zlvC2bG&NenLz;s}Z-OqDhju@K=+i_4BnB0Re%LfmS*h8eFztYc7axialu$FBnG^YJWrav-b`^`)f+gX9f1H!vq%Gr_?LOKI#-iq6J0a8IH!|gbjIpmwff+~}05Hw;!!{H_(7j!8x*S2(#{!phP4s1;ejnLqL<zXlG#5RCZUGWfc2%2R$F$Io^i;7ynuA&MfRw;m7qC)w0q}a=?8531`yWt}42lh<d)8+EM5nLe-;=JPKoxSx44#+2)-x?L^r-AlZ2+z}=aT77bYV)Ga{nY^!>OU5^Jmn?bstjTtQHiSx^X4_klYqf}g$114f!ja(%sO9_dvA1Y<taD5<YdJ4!{VG1d7X=OT#cNCNTi7G7fw9*wsADzsJR~uUYU;=5o{XI9s-IB=K4!l-|$+Zs0wMwaEclw+jBsr|J_Vlcqy9#c8`7Vw?J^TDycO*4Wg-o)QmyWtQhQmZ9o9rcZa~x+u#thA0vxXU)-fSKl<y(!d*NTywl-hs>26kJg1{R?Adj8l9R&lR~_UMId`#FiuHL*Nq8^uH3rc8koSk?y=|Dsv3Xp*i_*c3U!)pUM!m4oC_6opsmnq;LL^7ReZlTiFOqqP=S;C+Js^>t(BtuDIK?5$DthaKS}x>vZ%LDp1+uh+9s0jS<_&8GcV>@5KVOrQspN!egS(#&uP|){<61M1-Ai2o21QpRv0}8uzXE8OjTTp2SyC}w$p$os(cy$FFN7FqFj{sKOLQG=rawhhqB@OkV#jEoUNlQrVk{hc-5VdOf8Bnubvz+$&@qTm(p`3GcqGgPJ8oX2GTu+<%V9K5N(bGU+L|T(m;8x>5@}9u&h-Y(OMe?q+HbIPO~mwsydZcHCbREl#bz4T773t|xT+26C}f~^QP~mrGt9o-W%DlE%|a6bzp`rp(9mHY{sqp)N)%D!eq{xdg7HWAIMF*#Pcbp6xMRw=%o~;mXZEMZMG-Z-3Se5JAI}PC7%td@FGZrf1i_L8AML7-6;g`|c}ngs9a4teTgc7B`^~K+BGpYkYw*~br@4CTyjO3WGjM-3xc9}o%PkA3fTyN`e;s0rmI2Jf>LJBS(mv>*5NRFM*es(BG|LT1&oFo!vT2b4%3+KF;<+MKn4IobpB82Nb2WG;Q)mhjDgC@@VePyxt;!(X5YEis8Jem{IGu}HDde!z^|5_2#j!Ep;_ZJV+bBK08w8hGK@85S4@D=iDe$SO(at^?<91&c%36ZGyR2UT^giB@v23Q3<VPQ4`Y>|lru`O1E?$X>yH)nNBHi0n8s3<ttRO)tp)7gf=-UxLqD#^|-JX1h&B&5uOvaO*CcFPO?<MolwUW?$=Q&0Tg~(8#a*9CGf~+$(Yy7nx<wnNTbVbg$%QpQ>HNE}Cf8OXm#X2lkoGbNuW4!NJ!Kcf#b6_qG5=)%OTzH^w`0kF}dwIH@(+M7&c$V#0=oEuna2cYIWEamR@;|@zU9&pC0E0yV?|63;`+LJ~@^DsmdsaV0x?u64BhZbZ^uYQ(5u}5P`h>~_xx|u>7^K0KGaaG;Y+EB$P=j$_BM2wt#bofCRiQg*SpF;vYWoUg!Z~;2d-#F)ZT^=T@r2hBES$m9XF2N@Ta((*4soc|14!|5d{yO*1y_>&|3_(XVDE=E(E$uTP=lAsXE44-*x%Iyb)P-$O0YvA?u4uQXyU!_Az$)Eq8X*KjE_Uxb)w|6H>k|Duw&k>HQeHOIHC>Vxtj`?yjWDZ9|F0$hTC_D`g38*t7=s(yEqg>=+lR8;*y#!k7oDtQyb$VuiR~G?+(DKp<XwF2_E-3I9mYq&=meM=wGx<x{HYy5~%tz%LtYzh#B6aGgQohu)iyHtUiY?2R=Z&AYjDdLW%|w${di-;$ecA3ilBVo2AS0h9+3XBXmAnMQpK}^j^K_T~s+6%#}Huz7#CXZ=_%C_;G9SUtf_>jwHKGko27clUlCK^5)+%%p0%W*j-G%rkLKt0yZ~@Qjo(%1#iNKSRO-}Y8m4>U2OP=0Lkck$uLmbFBm(xJqx*q-Iu%WvT$Kr_b)z!`HcK0i4S#5$*lN-F%%?V7ToE0YK2Jh(s()plC-5)iJm(idxgyv>u?LjHboD7qD`MQ@O0DlA^%)xaTgTGj2azw5ttrC>s#eyB^s>DKLt2BcfeD^>R$U0TZn`Jo^Z!!R+KPKOf?%BQfDCWg5G1?&XoCqw>Fq5p0k?qeeTgaS6)b)Qn2BB8hT^-wgtX40`;ABEUtIPz>K{)auI;k$^3hH#rPGK9Z!ATXR3m?q`;7^AjJt9Q17&Hn9s8HEiL1|`F<1}_aP^}<Kxc_B=qP7L$Lpo(k^o4cYR>Q(qxb)T1yGY`ybIV(8jtZLu6yn(6g>4>aE`H!<h_H)`k!rZ3b(QFxhQna)(OKfnQM1dpo|?yOogz!4Rm7?Q-GH3Uy3&4~$qt0>Z4`wD>Cq?YM_xa@kCe{oeXdA-k;nh0>>SlP?}AOJ~LKdBx$$&meQwS~pAPO7(=?L^<vA@I>~;fsB_Cpm-y=5cl9GIi+n3m3|nu84xeZz~?^A52>VW4`iwmgK7Kg@a?-zH{<}P3ed@fndU?0^~ywF-wZz=$iKMULWw--hi7!7b@QA1X>$o|Dw#mp$qWiPzc5r(n|3N(5;ObyhkxLF;#0$w>0A*JK_&req*cPOFaUc6*sO`w{7Ug;!Vye=j<Gv<YthiSW&~dGE80G~)@XWdATVQ77VAR7y0X+yC#O7=IEDE4y$Kk^hAbAcBMzj`$t?Y)h!UO_(~|$yG;Vd^?O^#9h<|5#=8rZ45H^Pvz>^m6;L8t65ORcr(##s=IzDC|{o|dfG@h2_6;+0Ja>s*<CFj0r+e4n6Zl7l)e@f6D)p6_GDTCJ7UE7U;x$WgVyL`X_81%GOU4&+hI7DhphX@Sj9({g3>duAa7}U5>8CcfFlAJd*$9ld-X)kXNyA;{}$>a+sUcH8xLqV6^;|j|Z6m_1ky5}w?pNiw>(=ly$AK8X%aTU(P{n@=|VN#nE%*BDgi4UuJ&$3D%CDfqoxB<Ex^yfh?dtTk8)n71#e1lrlhGbrD7G9@`GA{t%3irvVr4Y$g!6p}3=t@B`>@jVCW+A*HS>AVNCh!f$fSLnzxa1&P(dG^ujHiUM!czFY57ate<}#}w&<Po7-fs!tv28d2`huIB9PrR=mWy^e^{w<(4oAdfTsJvWdmC^M)ZC!&!VXOgG=WkuHku(Bj4NYpXkl5ZAv~V)h1DO8XXAvyb_ATiM+)V0Om>8xUbvplb;+hHMvkG{$21MjxL&wxw(J%pZ)|0!@@hEKI!s~lRc|yUgJ|G;o%AKK@SSCKV|iBdv2l~w7rZ3JP|h`1umIACtlPh1_eyn4baL$WJ>0=VKp9`6NZ++ii$xo#UIKrbC=nP%hx-zhRe&0BV5Lb>V<BtBWP6IZo$PHMitvx_79*FU_smhXZx*r#zBQBZXG08sn+&S!+v@Pak?SL>T`-Kho755uk1vO~L6+0Y`#q6jC;jQM6QcOfftlab(WVCVR+cO4uG6e##P{{v)a(NaiIcY_562KwtZKac^TkB9IwmE%yN-Dnm;|k)auA?US$HN%sHpsGRZF|Akebd%C-uudg8;L?DfOIQ-vvsP!Wv5D<MGb5NZ{M}FO!cdoPr7~DB3xbOx9jsG*mf?F87z!8C|*lyU6IDNK^LTEPIil2eZ@~a$<LprCHV?@oQ7cXePBs!RJUG8RRYPPX8Z_)e5(pJP;>cFXx>a*RCEsrxp?#bam$dEAy3}AqtnqxfjB$H91};`WqDZR+3k}G}~H2P^yyo{E6x-z=F`3$vh3ZM<|V_;5|bikvDGu`wMUag#v9dM47)q+$9)#oz8vMonAB0t3OzEcJsgNrcH`L-cWBha(vbSMAy~M(4mKY@G8hTjI<(ltr?xpn(z{S2I#{u8|1P^;)bE&YBtTA!GDZvE)MSJ_U=UBX^1Yt6u^N^^B@6*7&D7*0Jt)M=G_I2qhjG2v2YIgq-MVMUm6A-?*E`ptw=fYxTZ9xVl(I4nXZNUpwUu2+MM=RQm5_i^s=QnaBVuO`;MiRSz0<u>;5`DrE(ek8KS8hW2mh%Ag^d^EB$CSScvDDUY@55!E7-FP#Af6l}}({1wxl8(YzBN%aeExu}E0WQY{kQ;+B!wK4Z5CS(=&U?KFhv4S_0G-A>Kd8mF;`gRVLy7Hs|$XMy5q2<)Jn@*eYJ`eBd5r}RTay$?+JP-QI{n}8+tHq|AU??p0Wm86@itPgPJN?yEql=9bNwqFr?iVVy{f3GxB?|;O!x|(QbO8e#Ls=U9<6hwPTL!N-%O5@TSdMQazvhV;z0PejvF$4mt6X7XK5$@MIrF-_LB;{1~n9ZgL6uY^7@1Nm~mh#7zVGCQ5{Sb5e-C+=8q65DNBT_uv5;N;vK3_kJL)Z9UJAMRXgsR4NFd{mbj$3G$@3sJ|mMflzLxq@p4t?nbtXh(C2Imw=Hm&BR)yiZ)c$gSwj!a(TgQ+2<4kK~$3ov(Fe#d9rn6u9hAt)s^D;fZpDxfb*YN1<)>~TxZt4~SgQV4j3=iFq`Jaq%5MF*4ADq<RTvMhXo^jx!pd?isq7OT4y1w(9@jNxd+y$0nE3#>{VB-Iibm&Jb3?`}`!F9QC~Fq?ndIT;TEk$LpE%+{UthL8ivrnrq4>w~ncGV9DZxPlq=Xi4g*eO03&BZmC=4To2@ZW<e0Dp*?USAqXWOibf7B^(Y&H@|<N?D3m!NsAKCZL%E%Q?Da{f9P(}lnQ;S@^CH|QWj50<hp!taj>u>!rj9aqYGc6r<Vz8CBscpTQKzW@v1b^l@;DLE^2ufK#BY*<>skXCH4nLmr(uA9vvD$4EV>1Q5<N)lq#5T^6rdDYtcU^D(u0l8tjI8$<HbX+B8|Ab*7vbHXC~-7mOa*8AHf@W`X)OAL9+r1&H(lLH_b^+Nl1WWOM&I%*VTrC10vvdn4rN23N8TjzP?OrF#rS?%9-7W8f%&oIi`r*d&>WpyP2H+`-{~${K?;R!@`ge}>%f>S}VIVvKE+x@Bi$dZI{Mh{bj;_XIvR4rvdt^zjKQCD#B7|5v|z^mOjz&=#P-=Lcvck5FQ^O+HsI$l-{cNZxjnk+e0W@y~+u(VTAz{$8qx6uVlw$QZ^#SpBboQH{pzU?iH<@8u98F(41_7CT4VX%wQ1bW3_VXe|z{09>5YLMK~DrulE+h>|hp^xQKm>a2g%Us%%aV}%$7wZ3-pP91$~mQIMXE<`y@u`xcC<7npj%zVrgu)mGj*qpGSv;@!K?;iJ7%*tL75q8;%e8RR{9>-q@qe15=B8>E^BpVXaSc?CbLHS|Njl{Z?Zz_@(iHzqS2>0u!p<0+FVbVF>tAqtQhhW`lc;HfOs%?=McV8}U^G_O_aNf*OX4aL#?9tZQtl$Aa5rWe*18{T58jt1do+J*NZvL1X-PY|wj6x$*HI6titM~cdoGvNa#C{ZPtb)+E1aZ!n??cay6>BD5fZ05oNt4daV3LXlK;=ur9n{|tZ?R<I!rkY$q+tLwH3&UISJf<c5J%`)FS7x2mEk}Yfey(gPdt06C|N(VKTQ1d7BQ2O?jzEOM`-KdZE)^39Pctg$u9dQs)UXQ!&hq5yZtN?w7hhwvZu4^aWlIhqX&Gj)YCB)OMIgDJe#LWsIm=In7<Gi92(8IPqyGZj3w|8B@npVrp>`OlY0#ZZT|#<OJo(;O<p+ZMfPC39O78T^)o#31{jG!Tl}hn6V8BU)f`h&AIa@;Y@oEUL*mUo+9EB*Qf;s_VdJC1ZJhVyYwWLM#D#MbtLS{wx0Wv7*?+9oC{sM<wH4oEYPY=GP7Y+oFTcMn!P(R2ppCiQB^DFymFXF!P0#|{m|&5WhA$!+R5eU#H?msfyYZJECk+o^!DFf*D)sc9LHfBa8%tBY(BH5oC^cr_`HBd;YP7YQ3Et+Al;~$2k0Qwk%R_&1yk6#ufVmnhG<-8wR9rkl;)dHUv!q*oE#e!EdDoUAldt2Bc(i7p%e>a%S6u97sGk{bFwH_F6F}0*E5I~$qOY?Xs6mdl+ZhOqwFf~i{tKu8d0e*G;lsfa%DsXL*jlf}1jx_tcp8F{ALu=6#X+lfM0xjp8hOR>I2YaIk{|)qgLRD2-SLXO(07hSjxBKVs-t(ah&N;*ksd<a5ht+)$)S(L4<oGfVqhGcRvmT8^Xc}9W-g;!K}98>OG*9~Tt38c1Vb{bP^xP@qq3*Vkq+KxqZ_xn{VC>aT!D?36~06pC<60bRrP}7YsPE-l$f@>`;q>&jPX&y4JI#{PHi2m%eL?$fbD$5tPpl<l4~?7?WmSwQxy0Wre?J6f`!5Oxl#!Kir$yzwo;tR(Dtz-_Goxm9kTpL?*G^j&NdF7M|(?h8KA08&3`*yl)zANLg^)A>UNqtKthCwwV5DF5v*5AwKQbXpZFlee(}w2_ZY5-`D9#_k<nj3^F>Av0~Rv$r>_NkXjlYXVg&jSVAm0|HbB}9x1(T(KSJGaaKGHQw0Ih7@_xhbckg#cmcu5iP8Q<q2Y6>&24nBQ>@J@-Sy;#Io^C_+Q<dD&ZFBzOd&|o)J=jgbZ#^jOms#hcO?bDoz+<ea<yY6~->xIc3)|7;r3i*%3I>WXg;)EwQ#d5zi?8T~bmYl?)8LnYDZ%Ia92+(VM^~19!O>Elr!A|TOsT+e)kAbLy7GB=LT;Q_scwdr35tP{g}MJ%ZzM%FI75+j6`VqPU*&42{ra%2=eYHiRd0Q!uq04XO&Fm-cb8T<ulN{?E~OLKjHKoPWe2ynh*jH{Qklzg(EG-r6{g{AA%KzYwBc7q30CD~y&^8!JhGU#c{PD!MyI$8uY51|8|X}i-&7w=erf!c+NWdBQmduyFwIQ6!3K&C6U_~|Aln;z>HtZSv$GHFjNW%Jb8M$y$#hx;FjsvFKU9imUhlodQjm(wnTpR7wf$~Dsc8VWS`(+3#}MH(lqnSGmkvx~**;WQJqPU@t&e}gC`7#AS0iWdemt;yrzu!d<R%L)JA+J%5J5ReJqmj<L?py{OkFsD=%iCMm1uk!*ZdDXn{ALmy3CX+FwX##{-+)P=GqND;;WxmdGA7Zh)=g6H700b;ut&R=W)j-OxXbIv{6q9Y6+Zh#i`#+>t1ccc_U=S57;gT^HxB~f!SBBu>UmJ1lDl+;<a!8=f=p_Gmsmoww9pv0Emrlbu23=$<#f)(1Z&>GP6c@95FC9bjLj#+HbR7nHl$&65THHN8%TA!mx_mfJzP+X}9@#nOgTGutmfQJxTFPd_K*yhdc`@88E`E>sZphyXt^>r$T@LO24C95*vY~D}Hs!$qml?h@vELh{yrM2seNPB?gKCE_DAt`3z{G9jnTD9UZ+gPj;IM`D?R(Zehx$242mSr0=u%s?Sf(A0GgKyQG(1UVbPku^)6x2$*dxwnQ}-Qm8<ugOACx8`ozw?J*GMik3j|O-6iYYYrCemv+jS+v6TVXv(lzSx?(Yfx11LTY#cEeu>*YiUW`HXZVr@&#`i#Buc0Mw3#1pSuaqzgA<XhXRA<1PQ&UHjYDv-1aEHaqXFSAzLx`gsR#1I+ECQb#B0(!6?Xj~;i@KGoBE~4J1?6QbR4#7p>|s7!T<>89Old}$83&H+n>=5jLCzfu2M{RQH?h(kR2fplByJNUAmCKvFXBqzg{mwQ8Ex;==u4dKybX`ar`B-IEjotdVA&Z_8R4f*5U$JG@%W&kN429-%tXO_>m%{k3F`q((}{BK|E0nF-7=`57|KU0oof1*X0Zh1i?5*$TQb?6UpV#8`@xw6yo65RVWeW2DaAymN~=?4UcBlHf#c7b?G%Jf&@g`@Uih)oHTX+P{-OnLnyo8g+GvtW3|b?^mAVuQa@2Blv~!g3_Zlc&h||Wch1Xl*%c>EZpl?c`garJ(^*0M=s$kzp{NNpZFPr)4rIC#0UPYs+V2d(mU%#4HYI^AUCo+5uGBQGt1Su*d}UW)d{Ixs#yj;Yc7jwS$|hcung^J?O%Aq-I>{^77Q_oGq|LxCkSKE%Zf_&ztrq8ku{6KjfQ&Riq{A6?%Cx_4p188|w$eX77s9<W!-JqB=L|%}M<Fb*Ne=JT%>5J5eCRH<R=hwDOji*1(Lo{wSZuAMpsCZGZ8|eChH>*~SHdRd&|884RNofPv3B?&G%78<p7H<~CJw2H|94E_$jIzPh!MR&LU$xOK98#t!^t9Afh|widl2wNYLCTLsfz4g)Tj-BqDf7|>V7*Ok6E&<5v&90f2eC$%cYHg8))zFTFIA<a(mVY6hAqqpCA3AG`{U3m9~t<1+Vq!IhR4+6Aoa1|9I&(J|K`AQOU)zmuAKXM`E`*n#6$jl#K_ljA=H9Nxyo^g4RPz$zgq}ZNsZx$)veTvu}}uCbrq*uE@sJ+*AvT0R%h}yFD^1u{#1Glq5@f>NrwJP^m)(t(O2|SX{?0<8Fvd<3+<yF%Yd~J-q$cwap0e_VVSp?Jv_wc?2-c-S@S=q+(9M&K{vr>E?3+ng(4uD0JZj;eaMSxEMA?f9o}R#j^{&QQNXOB-m0C-TM}CCqKn1PW`+HjZSj>$p-Vzsd7po&uEUN&AeNsm*7bCu-OSPgNY#Aqb5FvVM-T8{rj@YX8%46NyJU>FhIAAQ`;>7?h=9<%2&OPiL|DVl-e#1@ia=Rfe%^Y?T}UiFD*AKw}^ELBB1hW>hNu4n*ae_iDc=y#u>u%-<lL1kcmJLAA4U}grHkZ?;|2w7i2;QCIg)jlz5TBxIs`yY|aZ$ZMkC$&|W|nhXYHlY=_G@sbD|zLPta&I)lA|ym78iZlfU{y^X;5a8~RMws#N!aTu<S(tQ?QWts**3%ReO1_+hVsda{?lqbczx0y|WdDj~T)42i=C72)~wt!%XT<jxk<YJwQ4(~S{;*j`QERjLJAJ}jdFym3he|tjJshq+>5Qb&?N)+hZ(GKmpjB%aKH(&)`IB9W-n<gm^>Dc?3nJ68B@kw{qqhqV>Rj5de1h(X;5;IML&}s-Ud-QfpwrKA5?LY{bl6N~XA(!y+LL^!JoD0N$k_yT5mn*V%iYj!DHjr$5=DL()SgZ7i9ysKVU3U%>K9{#t(jHb^)5^K4G0>1SL!M_X#y=Ez#gCCN9<b>(_&M=Lh9j#+<pNp<y|&cN|4*4~HnW>dp-Q?#t~i)l48DcinKFh}D3$l)^-Oopg+Sl;ox9;gw;tJMzlO~%G#-={&1?j5-NBI>K?l}{IZhvA;AfIt+Kl;pda6-hkESYdssBEtjq9*qyy1-#7a*`XF({E-nHrYFB_%B9^Wk$`e>+~oIk|J4Pd#2FY|ep`=#izlD~|z2WG2@NPFWvx*-w{LUZ%N;Z}{M=pQ(oqklZu?rNPsY;F1x`*caC)>uL(wP70|`tLDYgeow&fg4yFn)1!widF~&KD;)!DG6&epG@n_*!CV!`X{PQ^o`A(dbgz6Y<pY7mu*Xz;=M{A=PQp6)pzPXFoJ6g&ZDTFrahkav5yqb~s>$WlCyu_u;C(q^kzY-Ng`Mw&j|2KVY!=!23zPqI6TJO`ZmSlVL5F5SAUz4o$2*L3NufWZ4e+_%YR;V7Y!3}PbH1??R|@aZi8YOZ>zM%D$d>z5$<{Re)w>IpsDh-K2GWONpD{{h%$z`v=IHpxL}RCe&|jWLbxpK%>}(vZZRmxzY3iSU!G^E88u->eR|>14NXm9y+ozmQ0PdF!BGv4mI#vx6arfiU)fCh)!0G<;Lu~2NY<M~1vM<Ip1aaJ8{nvC4t+6nY4&7#z1H#h9l9hRK7h}@MG1uT{r7zHlh8<^l<km=m0Xuu)1+Q^0Jc5UDsXyN9IHE;5{^|3S9=6_Ftr0F3mDs5#1wE?DtAD3I7JtMJu;qRuN6UacrUpj^$-~KV_@E0xx_b@CIet&Wf|nQ8Sq{y?c=;;`xX~>vM4`~(2wrg`qV?H#D|VZJmEw0`__LZoT}yIdlsBHNIVt<mFQ_i|?G7IqeD84|TL(EIc$*NR+JFROZy;MrqSLmh$h5(r)nQV<H;#SkE{-mD%&F^P+l2lfD|f|T;*R~z^bpt`n<l6ETCl5+X5Ul*=MtVzE2p;|+Bw^D1W+{34fx)MbW45c9K^z7z0687MM%zt(!Bq^2`^e{vbdPp)^%m433x8SaIa^&VMD~WbDtbQ0;BN!APf!S?5c8j#py+qQG(#$5a^g$WkHab*#h)G@CmC^j*0nMn-S7QUwzxX@bB*)H>`?#_|i5y-qo;%My@oy-w`z2y+(F3@9lKr{|^X}P2OfcrIMsx@Z_Tc!S*Xs;JYyhfYQ4pNQh6A77v~`aorDkwEZD;qfE(_4;vtIq49e&8lyaMc1*@^1b0Nlynv6&C}F8-q9#e!ZMw<c%}N#O$+p(0r=+lV%b&I!(id9WDVN+AnW{5S#xTa}H4Y~F*w!kiWftc~-+%Rj2^|ewhp4*VCJDRc=`|)rO$S<}o(gJwWPx@Gq`y01ie|_w@Bf&DS|0$1Ik1ZmsVD}Ol4Ki1PpyZl{=YjIN9(4dbvb{AuRmuL4p&T(mRu_Cft-Uh%e$v5&FK@ZiqO>%v}PRwPrP?E&5y5v_MB9ZS8_`B*Kv$h#D}DuVGEP@F_I(4A45z2J;|j%TGv(IG(8Q32AzN>k9u$qKw4Q)aB-otZNhHu=R^OXRJwCsM?`%n#Gu(&8}}L*6)rK<P?b>rvL%(w(si{}sD1bx;>TfLxZH7B-G_mk2itNI*=hH(@jvZ%5X{Bzakjm@-{WezSI9LO5qAx^TZGDTmPyxxY7;^MHZ&>}Y#t7$rge7>$R*pS&&lnG0hiKr*YDg>=GWp}f0Oy-ps|fyY|ly-Ul1w4zTBL=G-`~eDSgr-wt(3gfHA+<0D+IA#N>)K%AG(8Z%^X9Ra(<Th;HYH7gN2yn9Jq^DN|cu)^nj<@+VUq_Vf1=-{yYxCno}Hd^cO$@?`4_KiZ=eZ@%t+FWD(++U^yb5did?1}A_4&RK!l$eazo(Yq4%;^#fQy*YBlsi4t>_1GJ7%{k8u5Q$t{AyF8deZP>n(VGpq-cuUo-wqo6u6Q>e3s5l^NeF&UPTZ1EY!v_vAE>ZkGE<7o9{9a2?lR?1=mYKa&R9j2;M0W|j=hi5fCfEJGxK9+iuC5_1$urXGII;21Q8+d?rhN4y*Lh=PE;d^@npJsu8X|svpjgL-yMj}uC16h<iL@BH-Rn=ltMkA$d>CL-kbfp_|TG^E1UAc=FiF-55uSpXv>l3p~8@r0R$pu(qNpYR}iT^utn#cgs5hYTlyv{oBhHDnULzXB{<ViC+nsn*@R1NJtFE3F__m&{i<BI5v06||K3C~CFK=)bm@{jH$e%qOPh4INtfGoR-8hEdR;5*`GIrv=#OuXn}?a;hmd-V?40hGOLitlF<#Z?Y$i_>Yf8qmP<APR2Tq703Z3Mk7691DjII4*BU+PP9xgeSeFXgtifbCetwsv1M>bB7UX0xExCv-?S7hXat`vvwWtIIE%|6u*IUmn}Z#9@mB-9K)6IBZOWnK_doR9%oHL?z&xh$T`e5xGFzSf;8@$^bqFaFgf!-nf?sl;CJ23gBXdOmZ>HKGQJsnUrkxq!KsqStEpGuf&e*-5JT!6MP9u80hWO@t?ulhpAK&pB_kTIV_c(H8Be*F-XpF=i#W0HPqYOfA^?`1quJtJuv+#Cr79H^r>VkuH6D@<5mUM36!w;VQP40o5xFkZ6;K7~0<8{mlBOqSGK>e_~{SuPR_5@!+m&vh<yAJY;C5b~^PNaCYM<)60|Rh$<jtde0RbOficb@l=^r+_&G>seuhCJaNZ=zhqc~hy7!sePnGZ&4<L)7+pJFb{YBfZ$=~9WU@o8s=T&PK;Ic2A_+xcG&oXgaJX%4HchCkHePOCAa7=nH&DP)*YV3rMT4Y=<&^AssrbnWQ|nFwNqFiYjBr;Jv9<Z(*b5dv>mpay7lR;9K(=+9&xm(*C*;6yMO-pDn_d~_dE?0+E%nkK?-GGd`rLjJagD314f&Qj-`Vhw3$?pN^2$OC=Sr~3!A)S@hwc1h(AR)T&fek$!<1zqGDdqV9FX@EzBJY7JyLTXu~DEbuFk9SFF=UOuB(^V^CIflt<r&@oHR*}%1(rhQJH$IxnOTPiMixuVw6tgig}Kg=wKjw5V*L)0C=&tL}!K?wT>onE5r<1S?n~)_kLB&l(pW2&rtqX9oRvtG(z4AAvUS;q3{WQIx%sAq<MmXFh;Au3|wrY>h^P-{&kuB>ikXwi_E9Tb9CTL5<&e5E}*wjSq<NRSb;Sx)1m-}N^(*~3l2G7cXl84X8eFa+MRNjf$Ee;WhI)9z!^r|==1IS(&!CNAL*jdFUQd;L}M=cWx8;^^hIF%w3p&AEVd1O5{Tgz&Ey$qNw6^4WrE^{m;e&}#~@vgpkb~AJ-2C*^bfBjK83m{)(oyMy%$!f!8%Oz_hy_=_~N>%Dn^<8vlB!Wg|%VyQkoQ(`W^H?A_+BEr^f|49~=+oMah~KYdD?H=+ty6KZyylI3$}n<P`nOEV&oL0T8fVV#yitA&o#wkp6l#?m9J4t`eK8Jv`@ia7m$k<TnC_Y4RI$>#E4GxRl*5sC5T8PKacULm1TW(JxzYW&YwT+G%MSp3uchCf7%z>G2q-UDnL_!rdK&t>OR(Fop(oe-!~bssVummpVKh<ev}|oxYS;+;ern9hy+Xf*tQ_85QK0OfI?qB!tr5t?9TXZ4((>wrV4W-Cs!62Qq`{d-P9D+1u5qH%|;2$*Wst3DY5Dt`cy&%W<(5>cOiiq*1w^#G|Y-ZWE`y8?uxU+f>XlN3L$-uJyoi5%9~|s|Llu3}Ho{wvm{sH*)VAeW~@-E;Ff-7nKF>%{Izhoqzjv#2>GFunF%iVzbumwV$w52?kpV-c_|@Suyj5!q_?}!FGv}6W{@Hiv+0a{SVtqahAG~h-gUUBx|7-)qPr(hr`fZ@BOUH{k$m2yf3???33PUyK<?0FYThcknKvUqkG2y^=*(NCYDahX=6S>UM}R|Ow7C+H!g^L5kM=jt%q(uKf|DsNW9-}MPiUgYKd%c@_oG|p5|Nn-}~6N*~;02IPQ8WxY%0iOwHSyQ&`+Z5gSrHO1Uw)kNvM5)1Pn{5AIADQx{g0YK}`AWUXy+x+28+JoG61lquRn&KsAeOmlzB?aYTE@$5$h_z{Wt-aP#DHJR>hbr=<JGy3t4ll|wF&gCL#T$0kjZguidrglx1e0?JS!b)v}VDa+b=hz{WD6@X0xDwTvHsKK@UIdE)h2OFPP!aYxPaCj4hqNEsem1~gjTY>ryZcc4`C!S-(^jt%X{M)Lx+VC-maSjzt*}PEYS5n~NzhWukwQ45?`tUlYYEc21FLMiT^>C+YMogefloO%5PaXuOMjWnq8vY6H~PC}@t|8AS3e#Mi0_j9y#rX$$E<lbG2PU1JpO72M1UEySk6$XKN|i>t0-P=HXQe;hVd@tckd$$t#$S^;-rHe<;P4kjuKi9S`L_UBDnPuVTCuf;);=Ak6v9t_Nf#~*m%-35%cs|7qsK>ihAf;{-9CY14&`?oON<lMUSP|ec%;Fl>aHGQobHx!xiwBosogP_z1Smdq&~tdTh46d0v&7UlaXORdC}TeVAd73w{cyCD9TVM^hg|w@#^SFWelsc-8vy;{O_Q#~3C_U-66oVPjyIoIjr<bllt0$8uccT58cu0t&5P2vBICZQk?9Tt7WTd;m>Nl@x;WB)`}|kMqSh6|wQO#es7nBD-%H!{GHyW@_<P?HdMEHezjhZYuGcUkX*t5^gmN#_|k^zjMp4#22?nScU{*P_rEV_2kQ>AcuV*8&T~`CU0ei98h{SE`N~@aN@z3Iegb888N%d=)tR;m^(7KnC7bmb>(%>Qx+MF{=meajtAz<#-_uGKfsfaNa)JTBHY8cw5WSSp6GsJW^nmTNgJquJa{p!5|LiEpmK7Mc#~L8ir_0=TeM#N%b_#zi6lXpeu`d=K>1+5h^d1T%og~b>^{ivvWN0PD~4YK)gn&CNjSz~vwqYK{$DW1GFnp3kWh5tZmSS>5W}s#NdU{%CYWrDK^vd~V-toyamXf_<=r$iP-^TR$H;B1?b>ENz=a6N0_>=^KT*quzrnS!NDF3p>4Sf%aejbvzk;D)3G&eP4Q%KG3NO=5*~tXuYBc6+*R313Jbq!Gx&pvY@>;Su?feu21Yoo<V9nh;q43;u6}UDd74Tp%i`rut=*RP&;Rr>}o%T62_|WE}QUF}x{_3Y<Qoha)dR_NTc5P>U9f0n<d!UGxizz7vEO9|EN7gzgL0=;&N{uvMA#(@!c{MiH(ymKmDV>YV7;gjK{8|r~jHn&Zej-P*Y9|>qV|w!ZExR@|y~5j6G7d<AAfoouSsiwwWF?~4qI!s96@4BvnK^j{xb<V;bLsnop#@cEO9^0WZE&dT*1%)kahlEs1AWxN-F&ixWV76RtY$dpfTw17K9MM_AZzYg>57Zi#tO>^O2G6n?nk0Ti9VMTfvwC%e^WtgSmxU4yqmSglVufMDFq?i`+LkH<a@)e!!Nb0L;?-^(fZZ3wOYg-JC+P%GV})%yOa&buurzih2=5gA7}?!-8(^ywzd_(&^}J~j#(ig%VCR7cf+(mdIxcz%}?tAhC%}LM$G=!p$Jom2>J5SWyeLi*G>kKPDl`*->-a%%R!vBY{>kQmi4Bsn#N4tLGp&q_`1bfwm230sCa>ayb7_NmhC37a6d)~zR`P}p}XN7RxK{HVtRX=pfP+$QeD-hNZZcNE{-cgk@H0+WV0{%Ewe0Ss5gHl+xHw*%fBv&8(FK951_uT=z4W?1p^%B#jFROA<MlrxcC9oMVU)r+Zeduz8*M>(b6LzKYUgD;Yjknz^ZSCS@*yhqOct3b7{>RoDI<}a^(y&nI*YZKZ$UX`n}>;4P^EKrXGA>zyr+7%!$s%<k2<NG_=Z%@Rbmx3c^e5`IYc0M?HFpyf2eBs+RM8j()p%;aa&Ua5LSy9tG<KWIVWP?~OaEIy7@Y6w4-frETJd>w;qLvblR8C1%#}sz9+NDl+ClCa(6naOD>)x6P`mWvCfy6DFoLE?uQ*kjFkEgM>#a)zfpjr9E+2)EDz)xukLN+K628e2``?@wz{<AH*Ku<5?rpy>D{F-%NrI0}DcZaSOlR2T$Cn$msI5=Cc&yqrU;{IKjqofUICnHn=PYV)b>+Q|E6s%K)oV`vV-9<q;<=2gt;rr|ru&?L5>eozQ?^cGpliN(}jymU`=tPnd1#OI@Cy0gIH)3aI0^JM}k00Ng_a>GJ+Cs{<awFyKa}*qv$4F;|(p=3tGziY!X|SKkML{SFxsM_qi=-m=RYnprJ^(vh;;1PU9O=Iy!vY`$}0Ar9(ACE#!05bNmi;~OJ?Z&bGAqp@aq>W5|>d68!m)Yx1*mU%erLY9und-ZO^wf7277->BW^<2<iMR7JoBn_GJtxuOk4lgUP6ru5npG7CP2=GyjY#zkMp62<z;+`w}81pxo`E`>o%Hh8oJ;p1OzdlB<%LC{j;5>ZFMI&HCUqnE575E~gWLUAn7duHg$}|gMzK<d|dnJcLjdc^@^P=MWP|dtA#$YIHz0jL31)SOAbQR1(Er<y$u#`<N357f_Ch}u>zt<uqIc2G9N|((_l%8%nWqdH)b_oI6m~_p_k-T|3n~?ZgXQ}}ZyHQUN_hAE>0mEk@Gw_jxaF(r<(McsHeDLXKq-u#xPk9|Rk6YY0jtjxjL!iC@UlxM4yS(DfiGk5c)KS9To41A#7WjL?%jN$Y@25kwhPVnNPe$I3In7A6zq_fLzzWs~g;|%tW1!94mI~&0sxLbJgP386eO#gDK{qIW_`8)0xW7`NC`nE3eb{SKB)OD?`zCPHF=79r`L0z*ID!VFzkTXV{N<eMUWckrW|7SS?D_Xj?en4AZxbW5o*gh30grCb+}+)q!fp-VhRK#FY42AwkK`vV$}`%rX|ZU%&K=?cibEm_<gi&UdVmUA&Z2}Jz2X2LDg!K{u9R(RNIiR1Qbz9kl_^1<BnHt-W;`+f^Udi#g=0xft+=se*)RfLe%3EdbOxD)LCD=-W{dR$ZA@d{1l1Rn(@3|i3vJH?i{H;P8$2WVo&iqY@L8Qp$oIT8-C{veXcYbD_5Av(Ey<ven1j6W%I#mJ`Dh_{GTXl&hK9ciVggu1r#p%(Ppheld-Ka=Tn48bT$1P(e*BYScckcJV>`yW@Oj?Mwlwq$Cc^E$w%w@1D(i@zkDr|eX?~M_EPcyPBL&aW`1bN;yu+;%0gg;D^J(O&&N6;54aFM#7Le?$i)%w%{GLP1Z4lIisYZEu!PcN=PVFR;!L`}I4J-^MvHwXJTfH_WvjgtZfl}zMVTy3zN>6?hu=$^}r%Jb`GdYgnmK{HMEMqJdoSSrT4h6UYaqq9nksq8ZW8M1jtJ@8nUv(#~G=-3)@Jz2Z1MH!6*XSQS^|U8IbJ=dm0dcu-Hl=?jtMd}SP+;p&$pDeiy=;plA~p2^Xf$SV7|EYc9Z*+PBoK(kOgRQc=nSPfyP#VFb9f4Lj)lg5I@~q;+Ejp_y3@lt;KS16J|wqwC98W|td1@MhaU9ML&AoQdam=mm;W7eC*)$?<?0@UpGbRZc5uxdJ3mv{YGo)C5A!>S!9?Ipza)23UW~I{8uj6>Q&QU+bY5#xhy8M>UrHt;3pEWaB4SEwRjB&xm3KrWbT!bP>}L}$cUt{cpuF*7FmSYnmF~0ZMdl+9+Ki91=csJNV2G_%Wg5RA5SPjWo{S{pRfqATlpV;FW$g<q55(F!0I<e3>0hd+{eG?;5!$!DEH6y4;%$~(5`l+;G*#Gpc}X(p-5U9!INFdO8sRK4`8C1}rETgWS|=fMJOSn!P|KadF&Iz8bH4VzWxKN}y_>V0cTLEv+2DV4AoE^bu<*)Opaw5mCF~@@DoEmzn*N&<gm_6YVX8BnRW4vd`^s$G;?LQHrPah9DG0&d4;jy|J|tohU0If~#(rp8A#?)uDL6W>`5~3vqbZ>gJnM1N2ZXsn>y^E7&d}cgrkGo-D3*a3SM~cpT51hU@lj$m_V<d4(db?4#P5Xn`}#{0j;$KsDq$Q%kx-EaU@k)W)BD~!A2#6U{wUuR65y18YVz0H^LX0%Iih7|hLwpWe@isc9QQzwh|wSvX_P@u_Py`@3=2$ArK8P(#2Mfrx(1M?QLc`}H;~lohn8n6yd}4P3)WvYWO|V$qu996?Rs@%EiN+Mm>KTpPl!UFMHIdWpr7L5J`E1CkMxcRNOzhM{L{RE2e9QuwF`kF@JmqH)sWE8_qx|0X7oeF1h8-cnTvJM3;HGwp!|}uQDR03pOl>oZCszlz8icY$phZ<J?ktAk(B{7Fwcg^Ks>0szaUA6fXH`u!slHHvLAFHl*rMJUuCPD<fcjr?iB(}FR(QP1^Y&`K?)^K2YKwh9ltVa&+7zdoANn1Sy`NT&uc8yd4(6Tir-hkd#^42kb%;JgO|Ye+ou%70KtljJ_R`uO0<U63}XD#ijXQ8V{m^qd7km5-(}XQ{F{<&kk<S(Jt5pT((v)mZ2!Xfno8*94a{O1+bC!Ur~upvi6t6B${Fb|Q+bVkM$xIH+!K7trDB@7b-u4i*`&|PXZswk$0mmvzY$@u#dtj*gD$W@TWo>H$hWacO&skFhtvlDE)7KSu{Z@#CIxXuN+wg!DzQ*)^3FiF?#=h$n;1V^aY(rf(y`m?()CyVA>7$mbJmrGIF+cwUl>;o1+&T<0FtN8>@Hf`@AgAP4y<+r35<1Ahnc#<7vyuI$tm_*O=!JijYXNcU|Q<2tp2@;cCcY~mvs>|^AtKj0jucN@<5l6evYPd+@$oa@0EjPoLU6z;L%zgf|lyLs(S_$Hl1GoleVzrx}|bIl5gWcEZz*=W$0s~a&NBT=>>L;j8P6z2zFp2ANZygZr!*7u_^?zp8Q=Ar2Ho$90{NKDdMpsMso^&$Buzq#E>e(<q^ytuOh~Wd;Y^GL;9UsB|HKGI94kJ*h)%Bocta&Z#hUawUaGixP+&S<;~$v<E!G}e~c)AcW@WJ(0E9T@-AFqA1c2rGVUW%8{Em&-)M;b1~U3RxJeI;iY+j7ut@|DCLDub5*^M>P&$7A^i`J|+Ly1_Xlul7pkvO4?MI$)=6T^k;5FYR_rp9hbHws_$rt49O~0*$)O^xnfKD$`fws56g?YxeG9pk?*0$d=t!HqZ^RGpodRAHGApCph8m2kEIh@HDjxae@%2aj+;6zaLFxvUH<b9Jyvutm?m^Z+ch58rX3P2V>S|vYa&)D(N9+O$`TKAP|^-hU{%!a9-no^?pms`dSoIfN_p0S<+mqjqbd|ci*0~WhdBg{GZ2z66m`_#)svWAxizLV%VR6NEHH14Y@&?}~L?E{#A%kEyWv*1ac0UTjk80`eB?4B(4$FKlM@3mu&)o;D2!gVu&CBU}G1iPKi&Yhwyq$Q;2hv~->PeS=p;}eAw-mhl@X7HQ-p}t|Xy8(12?2q4<nz8rUPKr%-$K}(;xP5;>99{BH=#9Mg>(Evu<;LWiY6Q6nbq+@J&a`P&)g$1eF|%RWWTDBQzorff{EZHyxhMh%8JA{6mOn+6y>FaPKQcT2(MtHM+na}oA|EZNf|C?+t#Gm^RfFi*K_r7_#^M(9vkC_85#+FbE)<HeJmzOV4=P7=VBgi+0fHtlzYOd;>!3nrgJ0%guKu+?<1yhCvpOOBw_leD^1POjRXjt!(;3Fwr39J>{}*iX0&cqO;(TJP@gT@EYy@8UUvwuBy*SK#X*}Aw3_HR%Sz%{6o}k#hsD6ml8VE+0q(u06LuPIZeH3jF#bu}zsSM2SWO0l>LATWZ2GlluF$q7%0D@CQvE!V={Y=ps`)B?QbOZkYxM&HP8%nLl*UhvOoC}9i=%u5o1SXDjR;oR56gYCB*3b`-E|ROK-{AeeP!J$}Ma}mD4oL<jaLv+X5lS59A?t)uM9S|Z$1swP0F!Ey$t?af+4a%a$%~hT0hK4Q&JOkSw%a$M#wVtg>5ZF8x#4oal)ji0%uy-{5I%3lO}F+=8WOS2JmsG(aU-8RYPfy%T!_<7OqS+5z#yr;LODoNYW}Qwd4h>ptwOx#oqV(_v*JcG2g5O(nv|yjf=i|d00!Igg8MbN41f<FE|#aMkBFa-Z*C9rnl-#mc)+RZ$Ba%JIo0$002918yI<AJ6q-8#D6|25xEK85uw=MDMF&MI46BQoh@YaRF|zOjuJNLp*l$b+9_ZR0pYFBaBfAt_ntYFvru_~yIY^d9&<c%jGwY*HO95q;4U>QQ_Pm`J*J{dTEC3~Z3v6r!@(|@C!SzzGydFKQY`^_UhVK@5F3~c+PX7=Z@C};`l*M4*otEz~*P`D^3e`n>OI<r$z1;mQ)sDs*^ji;l3Hnd6o-CuQEm`j67$VEq$y^2VYr2f#=M!5Vh(rCV!45T@UGsJDm22ZQ`<;9Ke&p%cy3G10=D_7aLVYzv586y{c~96LEIX{DfUKjgz3%;}@&O2->9=R|=w15W!IXsKd`6|=Ff4Gd`*{8;5uXdbO5>=uyT4`;5JY!|-Gf_#zpu@Dp2tRjpvMya^_sx5DX2WFC{oaHw#nI9+-Q5}M$}607;luCY@n>wiGM!f9;n4!lg+3Z!4N5QxWSn%m0AK56)$hH#xvSX^SfPdoKBOzfw&e)$~h$pahccm>zo)#l<Pw6AXc&Weg9RL!J)Mu2>ZXILU=Pq@3zPFlucFMXj0OV!4_nWgt5V2)|HX|yX|4MYwHp^gHma%@Tjm?MYjYLHKl>}`l+1`plNjyvsG6<hb&++a|2w-aIlIp4qRb&ZIGF+k_R<ra*dC0yV2??rA`eC>%WB7*1Dv3;O1Gf$^!8sRC~==;-fxt+uT5<lAr#-V0UhXM1Z7yZ`JquJ#5-%s|ngY{W}vc@l;#o9Rld4ve`5knJCcg0Bi<nFLr0-UprKSeuJ4Vnxw+>eVm4F@}p=nU;K9)ByOKBCcJCWU09zvmoQ1^uNshUy_-Cvw%+puwB27in9?eLN-&%Vo|zQ~J5cVRv3+Hf_?SjQY+B&KwjXb6K*leqM%!tr?y|UBxe|Z~9${Tr(P3mpfzDJ5#po#nZ_{R|3-X%q!D7Fp7-SezlOI)AzqCa>vsz(<+_lfTC+v1k&ny<a#VX=92-;t-3}3q#^e?ft8udK<VN!Z3Hv_o7#%HVg$OLq{h6!UNMucVlsGa5LZbC6Om?gCT5g0+(MwHtcF5RzJ9cSwglhsvSF>k5hF^!{4RW1Xl%}hx%mZW+*GbMkvYdhTjvC?atS(M!C0uj^r3IAAH8>6$8Oug+~q<tbND7+-&(0WTX;R2Mq?WI{cQhx$S4^0TC_KT}+Wl;j|Z2i0(@*%AjhO;*y-Fr{$L)gLMh-tpal-Mk%&%t=O>xv0UE~EwK^oykJmUJXgG6l~FMqOXRE1E3<X>n~X3<{K3S}RVXxB;DHA@eMj1{fmYjpLoGQaWlG0StLCpV|!avC5=XwP;(;Bp^bBECKMAYKuri`>jB|5y5Zrt&-f+iQ~nAAbJ=fDkC$o3mNrI(M19AxHI&uX2(SKLR2pKa}C<i1PpRC5Pb1oz)9&2&IEO!h?5sl7A7l^eqGE_gFfv+TE~^f^I}Xrs-#kR&{QXKL>l25{$IQ*hWkMhjF$Fa=q<+4h#3LX$F1*-Atf4jDtg_0oH=uMi1whqQ6#5*;49nxNfS-lbRop=2|}rw8s8CE2-V~PWKbCMjH-VdhwQ*R(25u~(A;3Vm^r08A(cqwBPgQW4PR{q*kWlKDj6^J1lD2HYZ$C0;UasW{p2V8n1tO~@SLRgKpUdRF{meV>Vh(T(4G;AD*9Ww#pq=>sk779Qj9?70RUEzF)dd078zx>lsLPC@`8Ym)MuP*v$S-Yq>lIaC;jK`yV~byEVdwdX4YOMFyV@VQZsdqO&2+VpP<il?yyE&?+7hN%q%x{#ugz|A~JLlx-drs0Qio;t6D3Jq~yra5q~FJG8SWs=cf?D!%sNBB}HAXC*+1TK2!33a+mbSjCQ(eWPmvXdB5D_CYG4LZ30K@ZECP1q<-I8*1#}m{D}d)pMsdoX?~el9s;Prw+rl?nQdI71wIH7{Nz5l)p$uYYhI60w)h{`;A5K?N2@96eDSSK_gq#|dhMPGeC896j;@Fl7$2BEBilj*k)Cvk1hvBL90`<{?wCuEh<l7}p!ba#B>4nFfknGcRBk;V2tvuJ7SLQxXu?EvQ%6(Uuu1bwXyCAM4>nEf0@)}STam0##K2*BFXr!ss>0t?{h?wjKxfpobi|yqn**w(NDbAG#JW~gw-9_(VkH$0KU_s&L4sMxiooC0;jZGRy#NQ&pQxuvYsFe7(tF<pe~qeaLfMg=B!h%?h&z777^SW$3Z?N9C+$mLj8`&1gz5LR+|S163q?kAG3BV~c><3bI)?fx5tDX8wd>lrl%J(pcrEotl9}g-9pwcYs-yW^Q?`}mm*H*_T*q(R2_wVtVqo?zV({I7EF1){GrV0A?w#pADd%>S)2Xnc$pYVLRlZua$6*rcoO>w|T@|@TBEy(pWS<Typl+{~;C|xR#(JThq{}@sjT2K*$~v>Rh{~zhsE8iE)&8XT-88?6rj|^iytM@`TA^#HXeog5p;Ka4b$y{KoN>nJZm&ss1+%vPIfCoPw?SybuAe_uSALd}G)t+eQjXyV7;=tcH~RoO{Vb&i!FA|bKDvIfWfLOksfu+rADs(W<iZZy*Y&8SzEJ<EMzs(n`>2l&+#i9GzJ%O`lkwi0+9?O%28yA1=)Mo3x;tPi>phWyN-p}rE2q(fXwO}5q0>RRxzFIG8?SG>@=sq=DbOkUPUf%n<Is+c@;GWM;GlRk#j%a0bhbb6)Q<7%FaUa8uAagI?)GXY;XZX^iuBeQ0#{C?Esm+U=;Y$YJD5=w9mK-URZ9I^LGGzfy$XHVweT^?k8^Fl2X0877Z+sQwjwfAkkUh$Tx62cT$&fQd%1Z@?rfj9|A-#<BQ=+*)hbKcOnPn=+;X%@p%sgY%3W`!ZEeWo5YC&D{J_sMOzl<L#cbG4y`D7S3#vD{;Xr9rVi~sEHM7S}yz<brkVk`F09U4vsT&8ykqJ=sFWzq=^c=?}WP>Q;#rUlFY*J{f9KlsTxd<2gxkdwt{~QS1T267fY6$b`qo$4{Q4CB`?v&_mw%!14t7)DfOtZh3p>jJUAISr<q7L3kDA!KR(Xst6RYyF+qmT@L<lFWJM8(~_qV(i&;9?GKga5wa5d!7>ZCm)==(L6*z2x_Ruio|0(Bzw-D1)KSGC~bt&nNoExLSR(9V4YS1Lz<Pg0foq8;>AafXGv}mOhGUlgJI5Y_CmR5pJ!MDUM_f_;4##wFED+z)jMKV+;@l`gmJgU?kFmS&k#eN2eD~(<VnOasX`2`A(Inb(wNp=>MddWZN!~nVy^5VZXn%MQgnCh~`lmbl_rhyXN9wg8qjOERyK~JuT6GQHJ8`i?jBy_Rl1@=n8EkE^}x0%hyFctI=0wjj=-3Lp%CwErAUI!RTeCE_8fXkp0sz@gL3DRFi)?sp(mp-)M-nZ`MhjUfBu#;w_k)ICYZD2dpJQpXq&x5?L>Yc4W%W$${Zit8(C5nRgXjIRe<l|98rYI@H0o>~W6*Hw^WCKi{2;C_=P7Dk5j)z;21nEj@gH*2lk6^sq+=lt_42Npc4(VF7G+Lq-yoC4HKv={7k(5WB_ULXtd@DuS?#>Suum6av}#?T4VOVG+w6UI>1pPn{F%==A}~m>@(BdI96xP3j2`Uzp(+_l#}-KhF3TaG@g)WZ$hT$$TKfsZ?5aIt?jz{@_qQYeH#m>$i+F^-d`b2c{kHq!R;CtJ&*p)^BC_nE$}r3qmEQ4p$QrJ9mB5T@&$9*|@oS++K0ii3F3S#O2_Vb&d@v-wj*quEP_fIS)>27PMzPWAgxYzjYK^Z{(k*6*plQT!EGri?>XMmqFHNuS{K68H($f&Rq}WtL0?@l3%CklRl?FuC!$1^d2#qsZU|rs0NXfU`PVf#&(#nDn83^=sU-U_xhP2UHSd!b9eYnCJhl~o{v{RERP+A8T+fhIdNH`(taJU@(Z`-p1n!z85m8H7=Uoers&GQyE)V8^hP}Vu#oNl$|Dsx!vNkTF56J1sAik-ClYE&?-H7IGD=ToZh&HSzIXjpv*#SxBXbFOx2fni7%O`wcru<!CqcrV!+TF3)e6dl)M3eNtC8gBLVmo&krqB7;5u4r{&5P~X*m!I?_WZg{=Rt4ADaD)-B`Bqg2G47Qi<U7$B3h!FJb;_ANUT$k!CT?GERUB83^7rIsinGBOqmv_9y<S301C8&{L8$qK>^|Wd+2~b|r{wC+U$zNbegV7u$20dBed$Y$)OjPzvQSKDD9>MRx)%pj7h%P7Rlwxl7v~1bBT#iMLBUU&{Q)<^E;%W(4L0;WQ!3xSK~hsmc4zpwRM4oxbD~$s(r-z+z3)>nKxjV>=X)>uR!J_WUU0eoM38{EDDP4I?Ez5(<ZbmLb!z-I-lNlPc6G<BCSiP)p6S9q$u66mzW(V~!9w{VJDN25WRe1|;p7FwWta%{OS6=ky+&1DvewEc!^NIvu%;H*nTdJedN&qhE8FhPWTH-WWViwyqM7?X%E!C|j8`*nzi9_ory%1&fvYhr``9gIh4@pkd62Kc&>+sx_%CE2rCj>U!qII+{YrStfYY@okP88jD=;PIaNW*r5uy7zRR`Er_63in}F{m~h!x39{;!hfcQz5Wjc*#-j+3eBal#!m*AgpeK3l-)nd88ES4S@mJmqF+IOzRX-i@C}j!};mImuBqiy!{1m!#YPjSrct*IZ?qyN$j2{f+oSj9Y6CDQVtl-d@b|@d|^PXaOAv`78u7La>&$#NOTSzI|O)?ic+x&(*!2P$+j}P$iK^nMKbBGWLwkD!M3j&Cyo@wE4T2fMiLhVkl$QY3~3CYziC*_77P<=0IfURrZ9cm^Jl}nr?G%6*9hW-P$B3}AtC91@R+(-pi9tXls&dppz?v91e`W@eaI_}tX(DF3lSdleY8#;6~c1wY5+yJC@$N(XH6eKZyV%Jw05C@t?1+HT@%Mwpp#V3<)RuqiO#zst%fJP6r#PyIN5k*z$F4R!wpB6;{!?b=JObox8Z6ZCHRxa8}KgV=ncKh9k175Er{R6(9?sM%`NlD>E3W@RnPOHf&dFc+bJwu!{o4+#E$as}>-kGiPexY~+gG<;-{b^$Orfb~3mqXk`{3Sq=@$PpBrp$CN65D-?vf|7!&0r`_6`LNZ>4yEEupe&>^Bd~S$GJmA9?wPfP7(YUlx2-+XMoMX@WJtXPJ-mhA~scII0-EDA)Mk3nr*wqHRek@JKXj<$DyG=-#cU-NHHP|&7*&?vA{4Lqk;SI4`Vp(Wkw8i^g6S3#802C=lkZs(mSm=t5uSe0x@xz|54`_B}RaSIk6yKGtX(exa8x+Nis&ZNiJ0*%5g?g;3=Z|raoWv)93z<oz)_`?L%OcI773jW~Grn3xiG36o^#mDy42Nrr{4aF-R|Z<92B3Jf#@H?x<M|xmk9*j?ep7yT%<EyYx&ZMQx;eS|V8H@Rcum!E)iC@PYm;tVAN8iw<h2qRwDZ6scG*`J1gWR6ENTfx6{p2PP|cCj8)xVcVO$*{K&PiD9S72L*n;)0WCT<pX|>=vl7U(UQuoj|r68B>C20zLlZdAOgIV!N%Cd<1|-KX4`1%pGls7l-DKa4v#u4+qEK}w~2D1Sw%JIhmh8G4C!ru169r!!^8hoL$CxT&E{iNr9z9iw8$`2k)+P)4^R`1mWtP?A7H5l|0e=5qqN>gV6T20iUo&XABXrJXZV(3W)<6{P6(U3bNgXIL;)SrSi|jF#IQ<}O*%^WRIc;%Z+u;-BEF18{(><aUn|bcy*|oJ^7e#gK80@vQ7q|Fn6p|v%p-^dTh-g;xI3V!tJr))yt=PjlXHFyUD%g~R;f8sE3r*-c|e$p2=F!Pwjn_spDMG6DL@<nM!$A7y^1sI93)L)_qkt&Uqz3MjAL|%gz12^+!us&2s@_Po-T)o!83zu4Ak1C(e4CsHg|Z%`UZITsb4ET(uw!1iwf)%3JMf4`)8Hni+ynTIviu(p|K*dyQ2D|rk%8cv6PZQ)y;-n#SUUCIWtFrWgwfR0LOio527Yh2%|IWF@L?2jU^s2MFSfMZF<Uf<SGh0gU=%gN;(~|B8)6SG2Lw0xOu8|uNoWsLHF3rnLz8<uZ@;+mAUjNWvk+Pm~Qxd8>%;{qdusai%#Nevy1bHQTM?>1t2t0fYq7m&-X*MtO5{l`LzJH($+`@^;DsRA;$)oDA`gGvxg&6I4t4^wA#apA{zmXOFk|9lVJ@doEt|S^Lg4-U9JdIW^}-`X8a@NDT}Rc!DS1?&-+-o5eQzX1$-aDq%ZAJx>%f;i!v;@WM0lt(*l9uEbDq2#f5+ck#EoKfkqSA1LB278frs6tZ?09#e4_>saj+}FTqPs#SpcDcFaNU!9Lhk&IXxG>)Hii{qo@v@1eDReC@}YPbaDsPbRNIgt%2Yz&fFvw?&D_E^yK_mjb|f$`Bo#;D+2Sh7}Av2~9gqb?3GN`mxofMI&#Eo(M7bR&jIc%<=SDZD*G@fTOET<N!)z4t9ViBZtVfM2Xy??14W)ux;27#a`LJc6k|vcBI85hC?VPAarIm*EvqLDQBb>>)hG0TnqX?9)i``kM4PBWN;r=ax}aH79391$Qf}cxx<R#b679XY03zyN1Jpele{kq_8!Tt8^5!X@H}ZQ)}2~x(HY5LgjB-6Q*hIXM@QhjnKre$#6N+A@494ibNi0`Di$dqN62#`u$t6inn|uU+dB5t-?qDW$X6tFXsmA%R4NjjKk<uaFG7}%EbxS*klZLKT@B{07)xG#LJcrYA~ruX(_9xTCv%L$Z1!GZJCs8fsr@iQHTF|ye_{#%zPo$*olyKTA1hn3V$+0_(y?O$=E9<IN=AZZPIR)^S0sMp+c6W7;3&FWgEw3SjjKC^%YyKdi-$1e8w|xS8>V#1Sr)~<E(^njM5+2SNG~Oy$jx-k6Na4VN%5s`R;uGO^RH!hzuM8RwlgIZ>5uy)5GNOG#AjqPIz2Ho9GW0Gw-b)HN}sbNy%sY2tPr2x98Ki@hz~4CT6$kGvL)x17^Bo!9C6rX6lP-Hu94Omqe;jO8fu;tp<lWV1?^0;WT6BQB_J{s?sh=6{(nXZU7KO&XIi}dj>f2hzKY1j`skrVlngZkNVmROa@ms+>QOJPU)fSe!L9+RV(k;Ubdn!>|2n|tk}LI@7=2faisy~|ktN2T^am2>vb>LxPDZIv=WEgH%|i+ljtkMif?*)7rY!Y*(_iYo-1IbR@o=E@vm>PRNZnwXY7(ZB5%E`({jCWfVH9<Oc``~skIj8RgKTuTTYWCYBksUtLpZV4Um&CzA$Tvyk7Xwcx>OY8(~+8f<)Y6__pkIj&#l>bqcrJ3s<{;o-bG|?-B%ee{OAz#zO<!&A31%MKr<)cjB;hyJ6;1OP=VS7GogCug1x5x2S@Nw#3M1&6ZeldFX|s$E6Z<UYMU)JsSbiL<Noj#+?+_l5bb0~S7BuZVg2P+bP1(c(zhk9je1<e+dXOKu}FT`sDS@n=22~RCN=*P2X7SFLaoA{_U!{So_|LD-9d<uyAUvrf3cYmr4B+?uARwmm0d>L-|9_!%^Z>zeu(b3JwC%abuf^~StohRH95kjJN6P8YFDV!PCzw^dAHLkC;I2Hj+))(CPq0$|CY=tNeJfO1VS*T?da#a)AHrqzG(cmm-D=5+*Dn#xK2KI{W3joPNP5#X{FJ!+!wNm<HGue9W1uQoJHKCWE^^kSSR%!-LiP%gDXb9X_D^9+fb%j3YQ3+Yrt>x)DiJ$*KY=tzR@FmdHFw_@HTu4X11J7Ws!nolW7=Q>;;^^8RI63-pLQi@(!Sk&qn4lf29{iKtUQI+9;c-3|liIEf-?f<~om6L4=@Ee;}MXa8AixPBm6PgCX2F=4dE0L47wEkq^g1Ae5Y6v$||V56di*r$1BQEQg*C04Tu(BlpVI=yY3*hAAf~r?W>xye`XNdT#uzk|n?vEF^RH4<8Pnf-w>KrlYUh8M8+%w~*-xgfG%p$v+ScwP+C(Aq1+a@h5cAi}4fwF*gbf6)8EX())t%bj^e@km3P{odM~zTrznYRV~FYTF!@QD)Xv;sAK*?ZB}g1?c*e`ujw~FN@pc;aZsT!VQzg!lnj|N_|DMnLIsS)Bg-hrZ4q|jbp!N0La;22egg;#1rCsfyCKkpR3bwLL9EK5A5W9`X9%#bQwqL=<BNOP1frz?68Ff!h=m<53TJ5x<y#*3djk^e$e4q^x5aek6_giDD1m`d>{xVo*!$?6su5#qY!T+XPyjzf5s^+@_9TZyUIn#F^nhV%xUhx`Opt<HXDq9pMe14NLMF{ksHr3~;}R_2@m+taauJ2HK(_u4_a1bS&`)RZuL2v%z`}4Ixb5Xd`~mF00Y?aZ*7{kHR6W9)_^2Zr<hgeIOV~B#q?YYQx}q>sD<;p8j?p5yIJi88Rn6Y!K+GjsNHR!4i?lM2u6YppUA1>Mo@EkZ+f@m8<njQn^6@}pSVwpVO%_Q9SUwI%kyi0Mium30e4x(@Nf<h1A&Q5p;{bFHIQ*ECnqZ}A0gay%`v_gnkd~M)SqnKj2L}KNNs=|}Qmn(=GH4~;=1g2?)7yaJjcMqB_)XyQ2UDqXHd)G(Y<#?NbXrc8w&;Rl&1qk@s|NIo6hO%TH!Fbt9)W3TTx3wT(`broO)U6%pCZDmVb7R23oR(xAq-RA+-6WQFF8W%5i5P3TN(s6Wq5^di0u@Sbl`gZmZq?dkF_fXJH3W+DIN3C7m+N5cZ?u2O|YM(gsHB|?ICJlq+KR6S;O>nz(yiYKU9@pUC_*^XkujaUTI_npXyQLyelUyV6sSkJr4pkRdC&T?Z4yJ%34ZN$4FtloArswjtOk>rYF(61+6%OC*WZ71_!>C#CpRTHYyQX=e!}&P8ZeIE$`q{thU}wVA~9A`!(>IkoEG&^3$eJ?0^akjfo<OCrw?P4W{HhU9uWT9GOb8sgqN(ugb^yjBjn(v*8YGq(j{8NSCLne*_X{FGLYXs3qQ9`$XF@4Vh#%XiXur0z9)v#PO+;!%j<A=S-;Xj4P@QA#{FL+r<|nb$vxWi;jo$W}up8ayvfO5a?j#263Rcy1IBa5S8V@g@HhB9p{<g@CC>_h>fXxrPrJ4fcZoj(`(d8qCb$ujI`{CLdnW!i90~=J{;e<X!TcFLw7lfl)BT8NXMd{dT(Kgc6<^$fhxTxRFKucn9WV$TV|YZQNtb5;A=F#cJ$20Z5qxjWd(JFFwE}q_!tmTEfZWLiYp5Z8ya_)0et7|f_y<&Q5HWD@IGSZiIZe+!^lgn0^IO4KlKv{3t8<P&}-Hn1CaQ?#Mi!n5^F}Ib5<(y1o|ez%0W9o5}Ng+M9|osI3V8>pcjwgN2h6Jty!B#?}PnmEF^6S+t0l8Mg1CrklF|%po!yGO+PrvT_s6oZBj7;O7Bro!p}mwG46NUyhr!CO9jUR$)C`Dn91~^=LJUa`K!l_D>y4WN0_aueR736pf3+3wr(YkvPdoFlEH<J77zSRvE^$7gnN5(!>|OqmRr!4j&Y5?)*a!W<TNbCTn~FV9TuPB@DiN!z8V?<B6%)}Imp*dJ!QZQat6$OK@#Sj1D$QIEQ@-&q!a>U6M6nEVu^YoOOScG&Y>*!HUHJCTiJh$*_rb~(By^|j=mSxkJg5qd=h|RGb-_>2`eyOt;)l12fxz`&eYAQiCjhOhQ-Q`!6X_cmTRZYa$^&l9=wQb;R_E9wy+Kf<(NxL+SRIBS0Bwwjp*Gp6quDvN<jWUxD2nSsn3X^fYsV5^|i;6MWszXnYLgjj{+kwB&%)O)6)|?DQsG3qO-&nCxw!qtda%+z5e&L(_O-@vj`iHe8ar#;I;U+A(=z8O`-~3#91}_$?j42U~1Ig3-7vST)q-=n{@%>LSMpP(IbTK0w!)U)jz0ZKzq`5^+cTC%U%URh;y7f0z{Bd5NE#>|Ayb(9{L=}to-76#XC=-;%q<fsZfjkh4<|G!`Ze<BPY6!L?7*QO*8@xkeV?N+v3C1(0gt|`z@ldn%pduPQ6C*7a&!n;Z`=g2#^MI&nwS8+S9n3VIWH4&Xh)7H%~?eu?1VTC42$pILth&0_4j3Vf7(Z&)ch$uB^SnMK6r5AC>Zej5Mu{t)$NVkm*!qmjKYsKKF*zY#6NKy-m5|af{qREY|-fj9#QH_PgJ#2)z0f-A_2pU{<7dO{06tP1)}W6eWN>8xX;5CiT(qUYW=qwu{1alA*iLwv>E5JAkJBBEf^M-O`=2XA^rXUu)?IctFA;QM1_s`e3xfDOloB_Q)VZmN?8>J!N!pcgxKJ!qI$!Op=ElPlD(*7-AFh79yb)Mr$&GG0K}FFCK2pKQZp@-mgAFO!HhOI=vGDDy}-YPJG!Wv_zRNEN)LqL2A7++s%ED4Xfud9zs{#@bY0=5{!AkXS)MxT-l0jK$#~2to!Zmz%=a0msBAu3brBJD+?@l&~Te8Q<3d}I1U)gB4sgRpnJ3V;uXUS&?<1SX5l2Wj+|y9XQ{J9naDK_Pd0-6AN$mz%3OcBm{y45`3pU_YG%|z8Y3(axrbBYGG@kVowTlDbW_CsO*4B{e$>kZI7D9G;jS5l(kPMz(N?TaT`oM`^(%lSJm(oRhcjRb0~87mx;2V2*GF#c?U@bADl+PD)j=k-3*P4fwS{&E_5VBDf=NjYgBf^ggJ+@ms}Q~XKGI8P4Ukkx$*)rla&a`?WyW8|&@s#n1Z$5gQ9#0PbN<<FK?rJ{r&D@*k4ZK&+~S}saQz+Nv0iF-=>&;15dF^Sl|Ycg7sAWO83KyRgPT%Sy;ZkdA4`1>R~w|~h(NQ-0in{cByf;}R^88LUh*p8H0z?E&;7?B{c?d;y*A_tgKzMOMXAS%?=rKAa@o2OG#na|6Vy?5V#SzBveodew>L5m9ZUk>irIwI_G4KZZlx$7EjOTHbFU3kor#UkPw~BUk@@r7om4t#;MwIhjBUH+uf&AueZ@Rjd;?HwV_GXMq0t}i89v%e%+6}EId!uU&_$OZi4JwlA5{$nRpc`hHgCD{=V$hftKZE_v~kwU-y^Yu81v7VxC?SN^Uu=LwIbuX-60$7_PmkAdr$e5EU6YVe$@of2Ru<YdbySdl%9Rt-EO9b<?-w*O8bK1g*nLT9KCB{N_#r0449IC6Su_M+!kzlXATI>m@PM@Cgrdm8wPm$@IeSoOV`S2<^U&XfCG)CC!Z`l*R;v-YdH$B6=mt|#mS}zbz`rSQj+U>kge};%lrN?&WznDK5aEA^4&^*t<Y`kRG_JO-W^IRmO~jMGwj(SW+G?@ty;I8%&Fy#FOD0?Y2zIW7$M+F25_$OZ_c%I!m_lg*{X`1LGZ6<0*HlD{<U10Om2;bZ!SJN73dDh&&-I~+s}Dw5_4%D{(ZF2Y%gKFEss*aWu48IMwpFoKN>V)pC@yl9md}RT}=s^388IT2JGFfaCG{$Z&n&+r=4Bk4A(aohH9(tFVJR>SOz&og?=$ccZ|}fI_;q`Y3-IwU7|U9Q!@)T4Y}j8l9vb6nJv5Kvf1XhOMI==U3&B4sQ^oa1y;FT!&C5wbkA4<Vw9EY%VrziB!HJ7hXXr~wIYz?g~$y>5svtv1xN*tYfpk|XsMzG(@H1gMkK=@8!j#No}Z*ixk(_0GgtLw1HLLmN<ila^^szwB~G3NzjF+wMlB7BubVguWx|~`YCct1ZF$$K(FRn4r&)#G9Kbq!7Se66nl8H7fZt!@Pun3{ToF5VMNt3OjxdBH1V0=hJV29=oqy%qQY!#4_2Q2y<e1eZE%IH}7d$6Vl{b3n@AQH>IF749FvOmaRwME283M}^o^}|U{RLikW`JuAr<)+aXfLNl2pKJaxg(|pds8TsVe6%ER`Krl5`jOZMe^b$_C?H?H@~BR^+HZ%I8kUorXq<n-l+U{YsTcDVXkJQo>EN6vw9qpOGL)ac1vXO4!s4g1GqLGf~6rS!k+je!8>pBamA7OdjXtsKtR*cgLFYMpYMNL+|!|g1NDQ3Lc`Z0gd|V>=3!-cM1Ai^8F6Q$Ed&pEo$=Djzd2j0OHLwHDJh!soj5Ke(bLyDEE13QT{ulVQ-EYds)SkJfW9qN3*uC~%yRN1FG<K=EA9FM)PxB==&Qgaz>cNO4cy>z&ed8}Jf8f>G8HcuBbqCdMLjWV3PU76(_x|{jUt}lr7i9Y#gFN&)LDX}NR4UU=;#xAB=HfNVdYMqB29sKS$e@R!irB#u`g@JC}>MuuJ1a~SZ_edv&H-`!H;&?Tp6E>XZ>YTeQw;;Rg=|>*1g<Y{CcTP^nsPjN17FtQ=w!b!1~&d$gK<wx2tvd07rx%UHwynb~e?Ea%*brUjUB`?EpBP*s^2GtM-uJ_6du#U_|gJ)j1yVoT6!WQ8>t|s&Dm%<p^1PTYaUVm|)tcF-<DRh2mPu)yP9ct-RB$3O5(gTVA}<EJXcTWaj)MMRWxse4PUSP69<XOTyqrTcFdu6f`c7$_cx(=0PH{1V-Jf3@5xuz7p>G$3j!fnctazUyjnZhvVxf3<@eksGTFt-mYy~U6i5P<Np_tjhuL-8eLvzr2HZfUDR~wxLb7v{qjNTWhT`RlPb6f+pI?GYPS2JqkP3|t;S!{tyC~Dz|GB#D+j=AKga{Dv(%-J8!}REJq9kZk!&Lmh&CbpTADQV&@kQ9--FIdzS%UiX4VIhdBptsWC#5mCr|{WSOm0OouYx(YY}lS&nVKX1!jlB9WpO;75|+rK1``DevS_9IV{0oQ+;$0CI|U0wt%hGRBJo<<Wwp?18ujs^_3c%UR5)zDp5)O^L*^=83Phq)yg{1z%+3y$3J7(u>&&&MnWE1%?*1=OLDR{pVuUOff4Ra!dH2yjldYdv}lQ>{=ew$^Lhu4Ft7~hzpl_LN@M8AaHa^%?`4V`u>n%*u;}wJdg3>B6_ijYYo8=h`!+H#thH=X^XfbtBk4#KMwc$Jjv^Xo9*<Qz2&}ANYdx6Cdx2rd@NOG(%uB0DMS}@57EYqJ77@ahM=ma)B15L+3Tj?k(YKzR8^nMWzj?^~!FY&Qe|N~J^<~k%5&<^Bk$#NtVtKRP><m79(HU3TnI-!uGp@fB;xo;qjVFzO^io*d2x2#0)g1b;p^0N+P&l2g5C6?81j*18ioMy|obzR+Y?oQ2*xp)y7dld2tcO!3U3Q_^uu+&DE-o`T1+9s=?5*RUzz&&47d72G>T-C%W2)b|x^%h*j>5Z3Z&L!Bq;47#RIZMQq*COtmE>OT48s@R5byWBhP0?4n^K=<`=c^nH>;XRasf!}k?{Yl@3=AQfIrEk96h9Qn$f$CYEc1&IP!Y=y>nwr)6g`;NSLdo_w>J(Z2R;aHcwJ%wt49Zd!zwUD2FJ9-_8PCqTEE|ZVI2Ns)&p%#rrn`qAQGhV$oj#?D|nKO4Do%WfmcCBCT!4yiqX4%X0`0q2cDdpY1gi$tB)<v0zJjq3>o>k%~5v^DJJIEe1?x-&AOIWdKy~O)^qzeq>2@0bdDZQ7tQOtIemRC$nl1KbJx>DA)x3VOG>6B%i8rBwxFqq1{Jm0lOm?%&o-IBKq-9Ut)jR!_MwdtMt7wyw+L#EA3-cm+al(PN1o0_@%3(z@M2%z~1fdrZN%nqefnQj)V$PpaMd}G2pfC17VRJK#z!7n77xa2mi~+lC<%I7?xhyBW7QnTxYUDq*lDNKNZ8P=eg@)bnG_RQ_(^0SMIhw_2S4{+&T-625P5wqq4Jxqo5=Nh$#}y3snOE6s3`u2|sW|7-?i18R;J0Wdm2Frsxjb_z=P<e~h2xMm0BPVN?f|5~(V2)SFA^MwbLyKc49fz7*zD5Zk$7MGE4wO?L4`_%X~=C$g^CM_~kev_Ts8b^3`WXLilc`vZyKy}~Z7TAkhhFufaRqz9szt2ogJ+c_Wz+w3Cc<~1JW%#E;tW*NQWEjD)eB18u1s_CeP^$=myedB0ToBnU-J&b1<#4eM?=6Ag0Dj@VCJLh30o!L-DknZ|r#MRrUjHu9fd>`el9)9Ubr6=zuD>6^O{0aj*65+;3$C#KS4-Inn>)XU<-x+_qKZ&8v0a(PFCo9pV0zF`XW|>L7$XWoIPNzJgH?)p&TlWnSeOglwvD5IZm}N-lIdkI-!-cN7dl+;OKh|P8-NJpD@f8G+&0O*xt<$n=C!>D}&tLbqZLqayBB*zlg4w-~%gK&!cxqyr0Q<}Epdo))J95xy_hd9kO0CnKz3_;ga~H3>?6f-bmJ+W!1pJ|TxXye>+G`An?JOZj(GcswEF?#A-F9N@RdsSnV5fP|E(|Hm&Q6Z8W~&LPM8?sUH{_cn3=4Sm(*@Xo&L$jHWFl4x2r`Rx1%T~}t~OJz#jx&(>Z=T+d3;P6nAa#T=C1}PBt|4XQ(()uv$nIkj@k{mKdtUNid|jS`Bi{&?#w%QU@_1`Vh!efHE7D1WFutTR7F?pgAsc^fx2W}x=$AG?;)Cy=qK>ns6yx&0^m2s*fAF?x)U(M06=zn`>-QP=9qSeF-ywh9%AA;p$D(IfU^8D*!MRU;i(DUmSHX;Op~`?8#j`F2m-o78)9p}HdPr5P*C-r00~%90(o7YQ!=6s%>FQ1wzmPc3uCSCkUNt>P#Vt;U_AOuo?Gk`?~cEODSjttd*V7(#n>n9V3F9@vy2Si!>SD_QTt%tMyuH;YvYgC_)#yscSfGPJQf_T>S_^P9A+-^m7F+<40Gx~3_E_B4i@i`l>BxuPtVN%?=DAoSS%Jl9^bS68`~rn%qhFl??l`BPj7f@RmG7ADz}6u)t@v+&mSw}t2y>G&i?U&z>gEkqY>9;>lvMBQT9Y$+|gH);8TUc5x?)pi>i-v7?E#im0&^uu_>)=KtX|dJYv{DH0%5mA)WM#RaxL73Ed3&KAMb&pP-TYP(o|w;HYv}=%b2;={ZVBYHFj+rl_Tov`u;CcYFrtM=|}#Z3}M9Z<}n>Y^fn;hO7-47%%nkjh~mg_5G33y{2l~WX|aS2_a}Qv<`<7xu8ubgKx}-5q9B)uB%;uL(J|J)r1aaI^)6TPv(^V?U8e;?>iR`2%f0OGL>Zfe%0A>iM)r^o7pUY9IfmKh`ji0go4JLB7QS~1DZr`G7#@yN_6f-xmP?1WZ^RMWvfow&t1Y!zKC6#6aLYX#wr%`-68fZH)R>Qgv{@p-R{8;qfL>>s6EQBY-9=zu$kZ4TRll0SSS}v;>>NpGJvcanpf=Vo^DA_KUY)_l%I$285n=Byy-ZAVe`Fzg-M?c_SAaBY{ifX%|o@=Kc$iuk{iqkQsT}y9chtAEEYi^jEZGtO2G47QN66nMA2A7qk`#e1s2kYgRwJQTHaBWc~Mbivop@W@vfPxQLOuD{RsMLM2+=97zu8&Rg2KA;esB6XmNH|$6GDk3oXq_N;!jyZ1=)T+Qt%wb}SA7!U^N;&))-;{i+hWh=Ezd4LVUJTVQg>88O8JGGg4m7q`=u7oz3KER`7gO8lb`d#3yJJ$~Eynhn^9t>#i0#3x;M_WlG-LG!kM>%D4>Z7S^Z5$Vux2O{P)ZD+z4<>LlLC9&j!1O#Ei7mI3Q59No&X6;P_YzOLSUg?>mcP2NiGF>b_@;gggVa^31C)M_EF!*4YCItb~g=|@}Q~||ZO!{ARm7%yS9kcxYG3qd{#wW|ze~|B<7Ay#0_GSP~5f2R-8d307kOqn_t)bm*<?FwjFDW)EEhN#}0XQf?fwG$5;K?0gUIR^F&p5uvdTkXOGPTbvmqsX~Qkq-;&F1sD;{l`81XHh2Mh`hu$w#>g=cv$kxU7@KX{PeukbUL$u{6~hooHZCGe6+T7b}treHJ||9yE=d&JI_q#aQqCAh^$emP1_7;643CBK*2cuD05QnfdsaM8EKfhx(Os;;1YnGT^4V-9h@5bl2XG)H~$0#5ejo8tz4Z@gb*?oD^QaQksKv!1>=Z|6WmlQ>eG$KFu(k(fzVG{_IYn2jTC1^6p=wATyOSkqBFr)3q7;89WzZH!VfZfx)Fx<2AT8oIB>j$>|(+5TwrC#PT#20?T$6TnXObKlm}reR6Y67zGDtnQAdxHU6Nfk2jK!@hYkzUzh25#LrmsftT{!7}TRpT-%QZp^M`=Y|Bx>szl4yD6Q0vIKz)Ye01K*&({mbY|$Ydom-dWU0GCxxttG=$L#KbKl1pxreVP*I~edBqUYuJLyl9&uPn^Tk2bUT#v$s8gptpVip-`C6X-E>N9)|xzstxiqQPIV$qVHYZD}?PfX88<jt?6Gz#`XU2ho|gA10%|vLYX`+dVb4Xoa@oYV4j>r!kqUU!Wx{>?`C&+8pa6DUBnAtz99k+N?mWj-;eX^(}_U3u0VWhI2>}lB1sgv{<J-xqq)@)Ifrgwi_L5^iMsFz~EyWrL5CYyHo8@=J>SOD<s*NP88Y3;|^9$bCD%iIF0%o{Adn(u(3rB&V~%rj<SS?z7h{-kPK;#3eI~mS;WA@f;ht;yG(!quFD>VPZEPM`r2QlT0)}V|3IDt;OQ&kaRF*fKSOB^-d2uwzRg(6DA?o<%1m15ZI++zALx+D`bVga>z49Nr%_gBSBuBP?z`8Z-%T}xZF?p=_IVRiONQ9%M6jAS-8LgLH1rhQ-9TJ6<fWDzZbJcbs41FqwP~-*BOQD0Q>&^M<*LRrzx|4=VDGvz*MZ=G#Vkprq1xxCG2^>dfj5ANea#v!&SS?nS@|TdjvX-5#+$a=pX?{~sb(GZ)8;+rO682654KCB6`0`DPWRr({gr)ibGBoOgl`qPw}Dp>t==#)r(TC6&1;8_bi#FfROVH(8j2=Dn8c6#1U(Rl&T@sKDUN<?yLY;Incv_|-`a`&)-r^!C$0hM(CAyN>WX)hU7jp04Fvi8_5Yh_LFDWSeh8~~4l&KU2v@n}vkpg7u+-i9VDVVnWF{oE?VE}pssj-xMqXh`HT0Z6ABINq-qnXzs<qJTS8fdiQ@kQ7KC0my6(4`AQ~W5K6aPV}6^rN-HCfTEYp6YDbH4r1Gv|1~y<p1zJ<1cukR;qcI}M6sYm`S(yhONtXvONJo(Gz=U<|r4)zwk+MJWE&=dkPr&4718HSy_=Zdsf2QX-Itz{{dJBz6*HSjRRwS=kREzlZ!%fcBi^Ob7qB^IzM$jULUwnQinSgZahEOpjgYLD?<M^y0E<eHVi#ZW}?S3LgOm0gIl@SUQ4jtgY87*!Bv8eA34?OeRyD7(%Z}Y>fknE(Rs!zfW=KhG9Q#n%IZLlgT)azEY1(Nm`J}oz|t5O4`u<ix1F%ytMVA|8li9rK!1h>~h}TZjYaAwPvvV7NCo~(SrMN8f6hjSXKX^<x)*dZaySD0mj}wS@%36iMp^%h96Z!kfo<8x>w>ML<3?G3@mx4x>T;-mA))9HMA2$)Hr}&hvkEsGqp0+>13(4yW4!Tz`JQ>Cdwl4v(ClzlO_x=)&2w{Ux$FHB8z;Rv7H&%#&FPEr{T)q%+4L1@coyO0%5|sQv=~qc3bw6IY0eBTitHA33}YSUz|#gt74->hs29jQ+krQwycPbA@v8u8P9&qUal{ZXJbiyg=srq@Fp8M6dVW*@dS6UQ}s9$3b}I$WcmT8Xe(DJ-P$buNTnf!@agvu+bJdWPcZ)b*Ohcv#Brrodj62|_&2PDKlbu~_OD2Y`}2|<-=>j+B*0r0x{JO{ZvkkJ*PxIx&+|Va0XBl(<CYiw!Zhr*ibXfPMnKBS<^U=Fyv1Gh7+wBz%D=>HVpp9BB4fLwr}{;I6)K$fMYL(Y;^|>7(3=!0L(+R^033vkOTIQCs6$4#U;55;7+gHUhrTf{jU&OgNos^?X;XA-1nYk3Bh1869r{FO+YZ8g2idIigL~!hVzt>?I32&j0!fqS0Y#3?5z->mKweCinxD(FhlHX_*+RoH#eRrgckzo`AZkU9Qp`x;c6WBXW*!iD!(X{G;c;rluDS#m355%LUmNf2y5%A!YD4UNKo=XUO!cmIs*9rcgO1v^Gg$^H>pArT*-)0nC2*bgW>yvhfCmm?uCf^{fX~0$T$2=^jl^-#8p$E&i=!84l=x46li9hN?$e8^Q;zdm246ByhMi2};;nN%oVE$mx=*yH6=f4aN(!ymdhvv*CY2}n92s?a(5eVr7l8c`g-x@qwA?C3SzJIo#md32t;D2Ve}^oBn^!>^>oKCUFmE2hU%C>96IP$PzVEM_<*QlG*QfnD`A^Ez!8n$UUu6xy{q=k_dgZpw78qBsJh%Q-=?4Q!jD6OeRF&xG<`df)Paa-R3pK1M$aN!R$&ZcnDdq9dZ+W#5^>yXK&uHse#w&nu=^twyso={Lz>|Wy|K4b+#2m0DcVFBK%&Nz$IB~q4?Vi#%A}~M1GTa>d^Os>UR&GI~csJHuftTu7ZPE|hz|dgttmm^({#rlC@ct@TDLWWk!_rGeTBG<yLG@u@r0+aHO5Y*)ZY&Ecd_N~uxW588v!B)NipHe!71|X&zsz5oV_Ggfy+yOZT;vaEM@oufxPho!+XVz9$=a9?K7pxQlm!tjH=8DZ*B?v)x+AH4C0KOk3sM%;4e*jPs>{s|;W=kViop22O}((4WiyaGgKe@Ep$EYAz($?EeR@>DuC;GLoNVJ-dVQ#{u+y&NiRQt=Yc0fV@wSY1jZMxVCSt46(bDOfdD--RBqQfb{}}Je<Ggy4w7BcS$QHo^+fA4x{#LrC<}{EouB={Txi%xLInQf;lev>w#xZq_ITve7aKyDn28%rmH)q*tNFZO%^bdi)ir6^{nA4rl2F2!;sCK^kn+IE$h59!}B>IKU;irV~8Yp!5OeJCn9_2{nc|Z;I_T%XQ(2{eU2bwFdqsMzd!#m}U{X#N}Mb+8_Kk5=R#8@`=TM}@#%&L1HO%jA^cM>YhK)I|0aoiT3G!%pEVdCJ9?`a&;GYoTy-5^Su(%5BKW}oA6s7>d8U&-fgJ~Q`|-?lS=D#!cs{3{TE<j-1;3+mT5wWVZg&xyEDZdZc$CG@oi;a1QlCe>;I1iiV$@FJ!Gu6S{$NL4&tAhcdzix154FSO>*5Ra9jBo|R}>K|ItG)W?w*us33!ECh;`Nb92`(KohZ@zJz*C`|2g!iJxF_2J=95gKTYP$x|w)35aUHmL2<clGs24Sgkqr!$3_f$G7*`;G5#ParKsRW9-l~b$bYQcTixO|!Fs{a8^gp+&E&;TFAy*Vo4@xVrI<I#isTI1=tm740`gSVkl@tEJlSCfDTtPxu@U(SB~Ur4g8k6xcrwe1*)X9j~~BJppEA>PaWV;_D3K8la}EZ33VfQkHd8!Po7$r#|{e1XRa><C1G0&&?Cx^(g-XNPQc--ct$Iftpj!Jv)QV#0eK>@hw@4u&>)($a=AKm<=eT-Q93qvYydfJ?~%vAn->;OnZ~fc^rH0g0x<c!A{WUKlQItbBx=WhXM>XnQq+na8}ovY1>^DseGvoC@Su`Z(dtg~MtEnMW7&=IP`mRscg~ril_lG@M!)4vwyg&<@)i@B`PZeo5iuh8#spEA)!(IKTf0Cw!zt+j{aeFZev@v3=3|VOi$8u=n9lzef~JM357i*NO~M+}5AKxq8ZNFgRo{;~Sf!hM=@~<&3wVq@UwA7tbr%QCO)$Y@~Ir$<fG^ViC-@EwxvLG#w?f*>G5Fk*+AG0e1QJ&;z>9o48vPu9@lmBOTP`@Cx4e#ilLEmF}5g5>ZgX#rF&B-rQnEUFfJo0g*=bv*l(%a5MC%9G#`hR;sA|$UR;rl;d<g244AUoL9A?o6WgV0_rGjhCmf-RmF^)7(clayWn`C41Xf}<b{O+A1mNNTi_J=ib<@YVv3bYVIin>A1{Fi0oLR=2;Ir)2l3jivDxa`;^%Sc2WSmj1vfk-CbOs)C%|vk2x>TlICX{GXIh*CAl049gVRA!>Sa(Nz5C?L<E?+sWC8m2TjvKT;mc>G1$Y7OSHvH+c)(rJy*{z2Rl8rqK<@joVJGDcu2dgNkPYJ!J*Gj2+;!ZbpcPdx<aB)1=v=0U(Y)^v`TFz5P3|e#Mo)C}1W#P><9B5NV{H*!QR-pz(fRe7At;Jy>dG~V?Yw!(O3j|ihLe?O3OO_j&LQ=DZ6R3S#iu%ek}^HRedeWR(n*K7^>z;eZCx|{b^nid*5N#gQEa(L`fRq8OVl#s?MQTf$cqW>omI22>eIF*+|+m-NzIin^C2r^_w52#xxUO8;Q5bE$3#>67q*Ge)Pr?3U6kpvYXlFsru->HBY<T#KH3{x-`4L!50>`NHV}!d@=-J_nGGN}D4i)oiK)#t7t9Z?5-l#!ck6Sxb~0oX7DXgk)&}3g7}UP-ddMLJl*KhBaj1&u><VH&VoHqpFk~7#-`0gC-k1e=42*1uJd2?uB&#F~ACx`CG!<qD=c~{6_V+rQtLeVisI3s=F(A3{rnaf?>DSkiC6I`(rDbCnQ!?XkQmNFqNSLvU6p{J8z$Y#qZ;b`l<l5w1Uxi7m!w=RVJg(-artyj$f4Vo|g$u{&k`(p+hePw#!7_;fVhxIzHbm-npz*mFg_-BPgmoEOOUM2`_!<fKN?CUdp7bRjv;nHE>F2o`!AwO@9CIwilW&jC)C5WwbH%QCEkl_0Uo4<9Y9`%x0=7w4lu3oVB*wgOylHVbZv$Sp$A?$`mRY=%lCS8dRQz<uPH21wa5(bi(VPLNYd&@bM_C<ZaYptf0og8qV^pg)NUTR`?L^c!Auibp$kqBjdiC>9vG7pMYV~EHjpx*X;wxwt;u(d3e)8M;d@*?G#$uJnNNl_xg#yVcfPsz~CK%PdoQlYnf^ArV6dopiDV6?#U|70h&4{v(p*8Xar?NlBK8cCAOr0+PUk1h3|B4g+szvJ=YP#l4L6CBWXGD`Ll*1G5OKffN-}+gL&1BrUR5S^!hzu@pVr~66L-&|4*C4zGLcvN8Nk4ANb86}ey>z>d)TqM3BSeL9<l;{YfYu3r+uslxfwUj{xYQ;q_C)x0`cxA=!=eyyFjv>MR*3fZ>-iLaK}snFQ6F<@`h8|mk*1xnATQ#v4(~RU_8QqZ&T_D(w}rN`K_ALS0bB)dn;3LT@d5}DWM6)m3eF94yF*i<eiRYq@G9&w8cW*$b=I?sH}h3)`vPc<)c$X6vq``^DM?~Z9$}JfNeImTKWkgh69HQ=kNcCw)IWx{<YA%h>UoYikB#vs=Mq!Q&$M>xZ{n&0z$MrO;;ur$gP0lALrmot>Y3|&r4AcXn~X_>k!-R%p&QKa<FL<~08A|54BbC+pwAolaM5t0;xEdR$6`9g`_R#7Og%rXcM`zYZ=9<m$n-Y>!#PKOM3|?R>|j_b3<WTT5C!wB{aq%Dqs={SJ`$rMWob}z;6W`I{JqwfIEYlztNkueY~DpZnIKo-@Zvk1_+$v?9qcGCw_1xmE3n<`eI#yqC7yLBTZeDcd9V1q_t~z|ezbuB!nMQiy%YyXY?2+s{qxYX{aS#%-uOt1&(0<C4Q}}a?mshG7DkCGa2g+;Ddru8HY09`2j1j!drdKFO~qR!hv{(Zkp_wkYK4+?pN=h^jmKU$7km_^<$=&OMgH9BR?>VW_Y-`Mj&5RFP4D@1pZqN1^M@V*Vncvm(<$XJ4*B$-ENB|RxLT%>)~jNE#T?EYugAT!eA>S==JfqaEq3t`DEYn`{N{Q}7=&IDIgny#+<TZaj0q7xIIm)Lo6DoFZxK&+S*j~`b3E!M8Fa%i1l9?3aBa@{Y%=dYggASX@@_Cl&$+mP5|C2sg}NV(WS{s~<5u`%nim&36}JweG_d@Z2DlYli($dA6Pz83T|4>vY60nv@_p^y(nIL_D8j^my&gAnt?R!D08yK{fqEE5+~R91B=Bf9Nf?iP_*ca+|8SR^6wpR`pIcs`R~pl<p`P(Pk>rEkP~EJquu@x36)6luBsh*TQb(rlI}>A_Kv5AJDKog0c^Q0hI7e*J0va9Ar;O`Lt2NRQu>^4Wx+WpDgj7KMCZ$H!FfYC{fx2h5vV@{te&QUQL^qG3^J|#;RDe{4#X3#)D}Q1)bDN{!z2LNyuey^8WtsE?gtP)~-J7f0SjBYoNNosaS6tU+I%0skquSKC%@?61!<M_?LChoWErZfoxsF=wd2{_Y<*JQA8HkIkO4-dVui=+n305|&VSnko?Yf&JO(=mBqPEh%gMtbUu8dT;K4hY|y}c^}pI%mQ?GX#3>u)ey)L7JxpA0#ZnyJwT+>mtEuUQvyCn{Yuyv-JFH7U%ktO55lDX^(8?c(8ftT!ZTwQGQ^$Ht(-)M2_VO{YJYB$+p+u!1LekJ@!ES+3bzq8J8bI7F^+_I$ysnQ*YX&aChD_$6$)?uh<TnOd-1g3E1yW+z<5TB20xnP;+=``z`8tUY@%{w=}7DlgG|54j+2Fbx_lDu-9*mzQ30y+h8ms_*R9bXt^jW!=2V4FL67Ma}m<WcnN?7sg4uFyGctRCH$|y8=TyDQ+={Zk-Mf)-jC3$`klgfW${Ym-U2)<hX}nZ6bODF*CF8#Lvv_MH#mFYx6&&t3D}E+2uclHsh6X`ybA0VUEiRPbL9Dhi=V<LC0o|*4XKO6wXyQx<<l;C(WcA!IIS&OFrgqx#M5Xm+yD(DU<oy;O;2>`ysxu3<`3Vl^pGa<tfNVjC;^kRlE9=hTx0n^08v`b;6dACfc%8o|m<9#bKO!E4!naOc*+yK<6A!v0&E}zSTrK@uZG0oJYiHkU}PVS>>ymfq5=De$L+Q8A7UF+!$WT1I%gkW)r*0?Wx=)Yj1r$Sqpc5N>Q;b7NDhfa8?f-%D9P7uU-tP>&?BI_5Q+e6*AngIv3JqG>TU_c>m(QNb+pNK?)>j!x{uuLNFN>`#2uPHz&3qXfAN^(UKH3f}K=E|6dTP-zgChK+P;AwQb96Dyd-JG)_t42$M466dy7t(oqDs#?)mTw!aq!LWG?YN|N}c)hVp{RxHhQM5g#ud;+?Bg$Bx8ra3~MtHFwvH<?K<H3Voy=w9~uRbV2M$U;h1h`ufV{=#DhaKsV;MC;fIm9fNbOK3t1Bt!NA`3|Fl8gj`Q81RKnl6F*gG;kJM&TD_Px3&i6*T)<`ZrlCBXBFui?Gw)6$e7xd(xw>9jeOk*nSH}bCBlQBJ<U>K_ieP0;O|~!kfT%4k`To}ifmVj)#K<dz5h;BxVcT)I6IfY=}r$urCy_A=xo7x&HnA{cpADl^C>!rkV@xn;EuP1>CIH_&GNuOo-6;3uax`q9oNoB!J%49SAFvS70aW8j0H5ld@{A|cy@Rir;xRUhh3O!%qQv|ra^(dea8Hd0w5!K3(IP|p;+9MZsCr^Km3}gSm=V<b6PhpFIhpg^~{CO6~T=a?$ai06A~jRV5+vEa_{TR?;^O>!;1hZytp*W*SqmGlS!1zZ~s@U_Nn0J2-0+#q{(4@?c^6K;^-G}-3PN_qPfUxa<9ZUoM&rAA=Sl!`ik_-rmXEcL!`h(Zc;Ee9$iYMbsJ1*4R~|b7t`{;#(Q}XfR}!LqDjz@#hQGY@%{ga;d&I5j9gY-RS;4o!z;j6q}94IB!z=|rK@D3Q~C5M3wd%=KjiMH7%U%2`~5)8&ujGt?j!7GQV`WBxVt~`Iqq%UcfCL9KWvlwubb%Cz|?%Rh1A1x{T~1pWMjCk58r{A0YylulQF?43-k6~(VUIJ9U$W_2>8i*{J;~2`T`ViclW8o<D}w+yLm6Sv8z>BIS=C_CG8wq>i<SedEO}&<4^ao2qM)&B$Yzz(hz*QE9$_aw4Etka_{Fa66a^Wg)-o1kLqqdB-eXL4rCg#%n3l*JFd$6NCq~)C4mvG__ipaA255Bd<l66k>jKl`mK7Wa&DPAceeulxI>Xq#&FguRy~Z{XSQ|g5%~$=R=7)Mo0HjaS6|1+$KO1VcbU~AYJMUgj@}t)JW+2e1mauM0X1-&i_J+`SA()V?%v9zy%PI3U(*GdY%D@k+QF0Jhm9MCzs&@pmS+6X;uqlCMAY5?8|T6ujDzCF8DwN@u#4$-YQG(INebQ-5QgDphJv&9BqBMk>32_2l}RZeb$Sm&a_z9W%3Yg!488!c86H!&Fgl!uRS-~n^R}&zuo8kGrL*;YIt<KVi8w-?S2B4wPg+R2YBZj&FmvJ5$NPHj$KjvTucEdar0rFJJ%z_c%Bz(XmuuyljU1!W@z(Mj9FEZSHYE?BU1s&8XzctPe}wNT=W)>_{Q5)fLihoDoIEChh3BkLM!H;$gKEjD05mrcd~DQVMFj%~#B->w4LihOD_TgxhRJ^_l{>kOawp@08tdTH^>9b&;?<jd@QuFJ@hoTO`G#HjXSvk_(~Nhfvd!Qkk-ebQINO44PN>w{C9Y&p_me^WQTxkBS~J;z4rDUL<(0AzaajuOljmdjC=;S%HJ<d!j-4#lP^vnW_LiWLfQH=-Chd-o>?(F7b#gN9uf}_Lhr09)@XO@Tw_-(M3z6pfhs@9v-{6V7rN5`VQ47x_BaJ^%povxhUM-?0<<GYYIFV}?iGm^&9C2kC4x9SpL%!t#W!hZnS1j0H6r?~(DUF_{l{}&N*!!7!|MTW}8x!by&`XwOAC@8)7@ICq>p*FPE_)r0we$Y?tLAVTC17<c1RCb_lUF|4mt0GSQNE#c&?%stQ_}EjxTBj%2CKyv<z<Gh`j0!@f?fm;D*?Y%&2BGr@1Ba^6aatz#f%Z`ge^*ggvyFDL~2gF!%JZ24}o1uS(}-0waiiMg#@=VbS|dwJXbWffUXtX6^sM#BJ-rcl^2xke#u96TCU3~K$Qcf7&0nFB)^wXPueh|Bcy%9jh7bulGEdIb2it&|L?XrH5sro-8K?@tP0M#v09qM7e!W?0peu-jVtFwZ6(Fm;oJL><Bjh0kO~Jm0rF)R*}!pxc&uB1pY+!WxE|FdeYkGEdqURAX-joyZrXekN2>!%>ngSC_`7OWHR<bJA2{8HaQ2=1v!h!JW(iMBY7>h>#IXNP;>yEA`JjmE#++rlo7Rju-${qtrcWF|47wwnBnkOiGtrSQ%KTRQWPk=JIEz(P9LU;7>N^6Abr%Ka5me3FY@X)8W58T%s2P!i=Jf5fud<dWSN{XOc>mYRnqBzvCt_SLbuFdC{5V06*pR9BxQe&WQFZ}HBmb^b0AxllOY<B=xSQ-6k$soi-hYnl-WPvwvx2$rrb^@HXFiikJxQIaeWmP8{7p}O1S3Ics3niQ5Q`6bF{<D6mJ=*qmFCj0&d0J6j1xCLZ9K4rw(wl7uEaImdAy4g?AwTk;lo+SJ-aNYCh`z9nQGJ~C;PAqp#x&`EaL`y-p`^TCv|x{&oAx$3!m5<8MzYZmGx+xqpdm$CjqcIFT?hOQOe!3tjVA7`NFRCcJOJ=*b)4r8iap$ZkbKdxJ9}|(4=ZlwuwTA*?2ZgDc@Qp2JIhN*uC*hbZqG+xaY1wNh?dR_ss6~@xe=_<#M}0IvS>S&&0eAXnF2s0om4w7q0X`iqbm7bIe=BfvEpMm<RQoMJiKa>$K5n8KAg7D*blL*)Oz7wiQh{gN>dGYpDvC9oDp^%lnkT9B5`EW2x?BYno9h02DLJ7MnTjKVtJ|g?eL6%#KwlXgenqCw9IAHzhDrGg+p7@iy>({rBqPO8H4kt864t)8C{v!nSzU$Z;_kGqfvVT8qg?n=s(dWZ)*%pZx0(Qz2*Jz(Upz)O@$~tiWS}{e`xDe?xTjTs`#F4VsLMdahi$0o$a_DBPC|mLoy`v$Qek6Y9H1gPH#<?fEkXhPq3{+b|2re^LoLrFnH|y$X#pu;v*0*NBBm<x#()B4p}W|AP{%mebTExnz?-{=>V;px%Nmw_VQ4O}LO$-P~wLk&LN<M@xe0G;G2lv>3wE{geizq~CB4k!76tmJu>Me-He=7Fp_q^$GY1llKN58D%%e8vvTN(h0`;uCrvQ2WN|@0J>9ms9riAfcv;%1^BIoHS&Xz0*!|QL`=|riTl5vs^f^faGJ>D%vDAV6$4cx-2*N=R>Q#E<N`=P20U71qcvOjwJ6->k7{6%?k0n{hSkGlA;yb^3;HYO&_1ah(0w_KtIkh-qHH&*@Ke?!a~<1-&yiI(iA=!F(RWu&1`$l;I0cBD6cp#Q94^%bSVwWV6^U_%#7R}4hs)|Ga?3lv%M~xLyx4GR(a$&WI!sCn!ZBv3l%ptcL%Hda+=$QgWo}DAVg59uZUFlviH6!!nq~zWp=DuUoq@hyYyva!R$Px*Cpl2-<FX2JSc46HXCSeOwhU}|)!X@SpV^!-F^3Q`AMC#t6<d)t)vNgAre$}Gxh!VM`=zEC2`v|q6)alyH=irttaCS_1!L#1woG!}O*nJAs@%wxy4CDQKVUi%@PJn;HXWnTJdqRrM!u)Qs32RLIge4dg&2}$K^y=zddt1p#KkoPr@;b2<IzA3pdf1&(XW96_Es<yW?C%Y?)P*&b%Vq@IfT=?D27FBk~VN90L~WMw#Q9lk>!-i-;~65j({4IO`<g+LDG#;6x(<2t;iJQCZ7~&IJI1UQ1FwGu>uKMr3u3_W+W<|c-WpSw=l3$*;_v;X;n1KKKU%tfoY+xYHm8Lt9+xyNu82=a4W1ExKxAa^UV241o@QMSK<yI^U~_DSB|$&y;eEy&*4@E7xqh}Eir1>(`E~)MC5>&2=IM#d+r~Rp1eG##nS)rRIK}e1V{|j+VuY=3n~rzuwYcNyWhJzbk#3sV<sO?9D%8X@3y^^cnG9_0#ng>7(6k;&%OU@^wzMqp3U?J%}lN+SAzy2^qxs8`zbd!2<#_uEOj2d+gaIK?m_X@ZJuELrfl>>8ONej9XeUDVZ4=TGFUN97wz(mIRk*|OY~3a`NjBfhforeD%CcNDwjYS)-|E0Y>*LoZq$Nww1|LuB7;}%-ybPq$l1Rz9WV~mk4cPtNzz9fXQy690BU`7JX)t6R%lWYY|2<8!Fd{LlgsFh9`}Ip5e&+<+%s6cQ6BfTjOJNxV@N0t;`>hcwO>0!W#)=nu~SK7YKnkeK!&SuDRXyZY$5ByZw&Fyisc`A-$|n;&9mdNTJ+Qc#wUPcT;GOyDNY<kPX!5NA0hMu$aX$#6ZS+Ei2(rNBf8Z7Tio%kk?^q9-vW26DW1=Fm3YpRCny>ML=q95t8~*?jGLK?Ey0}U@_^}E#tS!r$yI@j|7UsC{{nsCE@j1XR*%n=!l)M3@{x!6G7cC>V;oT>tcagx)vwn>R~qH*Fx7~&OAvwQsQ{>uY^eMvTRnn2><g4}n=J2~ruoHQ!@6Rw0f8HH@L;7L0FBcezyZrcyDW*xK3+f*5I*5jFZ+v$ev99(Z$(<|Vsa3uOw`lso<gf%uQOd39ZV-;w@nc!nDj%=z#*O6-<#4#JHK48tc9bybFp;B9FGQs-)npi-6&sCO`clx<*)O_W_4G&M#KaWSM+I^$1V0hCb9~F*o3|cYq#9+Sh7hA3N1flx=*2qyv8wJG4hrc%(mK>btB-&Df!R>F<=9^2I0a4<oNcDO9oXDL37&kZWK#ELRR_(bCAoOua5gdR5sIFapG;Qcz>7Xghf`8vTgW{v}<pW1?vP|Y9_jX^#+AFH><vEx2nShBx^u&y+Cdfby;s~JFBKw&KG7)I#&ph2Z>X&^T-zVrAKLy9vh3kCg^#&5f9KhrPk(yUk$%QBt6lN072NN1<YN`XKUYJ;NE0ChC>>3N{4BtXHh|Dr!bwzQZupgLmH;_Z==*1`R})Ut*)%h8Yxfmmbwq#e-#+Ee6(*=@;)n$Q0kgW*GdT!wfE^V$ua^+J=JT(St-x)LRfPpKNt$1LxDNd=5f!laKQR_$JJ|s#!@$3*F%GmqZ-W{qPjv#st%6a`kAPizgD)bhL<K(XusDCo@+z(M}9%H?aubM4<Y6h+dzA7+Kjk>gSVGdBUCH@E7*EZwJi`X(hYCyOsDe=gz3fB9`f)AT5a&*0A^5QR`26i?sTuCh#t>qlX*A&mMA0)Jj*E*%v&Z5mYB%U@3-ZGok+N%7@oNDAMqV2?t#}lRf0H_>zGDU09T18qPi|I&EP@aeQhS~AaK?vH4MHq9hkz}Iev(BtgsmbZ5`UczVfN8!D93~9D^vao*rr02c)S6f3qR5)Q^5o&owFik*@3hczBPl-12i5XlW7762#z2YGI|$*%rAOLQjv|z}K;N!EStif%RSB$ZQ;JjY1j9S$VBhhlSYY^aLp9W^YoShW@A(+D^#%_xaWN2b{~iO0Aa`(x4WMlGTq-(7e;;=K<jHQg$SI8r?Vuqb#%Y^F9JcxBkOk!PZqGN)%|s1_ti;1GES@VD`a?3raf{P=GElOL)i8?}&)jp(dSsMFw<&PhS!G{q?chC}MIVr>as_+-&*3o(KPRl+H}aSZ-2wx-Lc?s-Qy|vog)I*b<>xgpC$ph`~;U&&;zE65os7dZz6TcQteO(3N~5{gw-3q)4sq4LXA&W8-A?B~vGiDJplB=0aOvrz=5+k(}qLN=SRrS|0;Z8vrr2{;0kHIVs3UEGXj;6{;2MjsuRpYYedRxKJYsbGkw`+)f2c$de#tLD7#!w3`q}Y<iKQh3w0l7&dFrcOCb|G@l_W54RrxTW)-IfoOoZ9G9qDHQNk^58Np7Rb+OG^?ttP6T{eg1@BB(5?H3>zQ3}$$)pRgv*xABHMLQ=8K@iOU9yWOs_2s4$RCCG+j5kh9&IsFqqzKv;YCk_%j<LIYD_3ZScWjL_%RZCL@rYh2IKou=!=H=Jb$W7)T`=(2qR(S8dvA<ILhQHB}B{BBRk$o0RZGM#vES+f~47t@OcP4*YpU<1*7xAgV;iH$zz(fwZ;rKz>DG1-y-vJLhIobu*k_q&oJmesySFMzeYY@XiCuw5RlEk21-*N5vac2epQ__hE%U{S*mCyBOte(cmPWu#DOKm@BRRo-!VCMW1!dMnbUI_wE$u0{W5CH0wdATV1iy)t_b#RI<pdl*gL#P*O|E9-633EwSe&CSjtvh!c_tp7e#VST1vTDsK#ljb8A^ppdohh2&j#psL0%Ax=kj{4Y&bpCt5HAoWU&GeEM68i?IDxSBXb?Efz~WV^x+g2)_bh^^xCp<-ZOzeO1ipqWR~4ilz9|4d^J(<<-tPmg=&|!Y@gA!5a>f>X#Cm!^&kI^dy+*@1bQ|rS_MQ@!veH{MOiSqT99qak`oC?bRcw>76b6W6;oKqN$``q@*!`<0kPn1B)=UeQFZ$luR`Fg+Su?2%?@+T?25=rN2(eC~f$6UJy+xP)TViRO{lelLnFm+F>b^ahf|KG=<s&d3yp}b9p5@6!4zy<CP6TFTHw2jt>m>k%Fx2C+0%3P`aONe*hbN37mD39De4GTnY5O>Yr9>mYDN~V0Xtvq|<;bc(y$FHNyK=pxvz})-9^Hf+yhi&@3Sss8ppxgS-u(9Wl~JJQ8V+hvQs!WaMm8JU^Ik<x?E4CymT3bHqj7!t`Js>|r{8v5`I*`5F%E2iK3*R0DK;MvCgHTOSDN$3vgQb^PDb*8ohino!ab9X(k=bQmJ`;7vDu3pbxvcCYDMBVSbAwkoQPVlAe^;{#(E>qlVc#Q&K{wpIt!d2V><PX9FS*a+A)P*aYFZ4Tih`H?~_m%TsTOZ|KfdHdZac8+NvH5lk8tciEg8n>MCM>u`-#iRdw)e9QOZ7~7YjYU6V7+F><P~Uw`vbPt+6UwNlLrx?n>+iKWQ6-){r3s$;wvjjhCw1A1gWfbC44Ze<Xl4;>y*Cx*SfuJoq|)G-s~%qdA7WYtyd-)~MMl{pnth*Z;7@mi()*ea>Z^D<!K6nEOf~n#@PL1XDvSaT5Yh<XeR#Ft<xfgxH_22lE~gdkg$=Q^^w<{8xM#H`+jZVde2B_Ea7QQz>f#p5mM&Q-Wg<DFY1z@5QdPHqGql_)3DY=#i**d;{*~;Imf79olu&S~uXU+fHPbFES+w)o9q{%$Zdm+qrO!KYTJvrld(CuxAqq8#)zjg-CLV%=Bt%2xk&@Vx+&!9ff2r#2rn|J`>|5e0GK0WDyZtw~lg)SRa$#?0FZ7;CSQpvQYDO_@qUh>?WoQi8$a40<c4zu&hrCVny>(lbSG^TCCcaJGFc9W0-`O_J<~4t*T)&fexezBvU)XHV_t3R2#AhRO8K~fp@v}{H<sFT>C3iMT@6zmU(Tj;pWv{~ZU~Q`w^K3;RjUl)Ov>}i^hOW)@4BcTsE@^e}#KE8^W|F_*b@xvUT2!^Rv$a=)YosXLbW1Flsr1Y&rBs=r<RE&Qpj$2Q3Vk=}R#13tXBWF**=fk*$#@qrwG~Cr{>dD=eUSt7zEn>EB@lR{NavQ?OZNIoM$Hi*+|T(i>bKrK<m*jr!mUT*@O(~m*nDUrD<W$t#if`>deVkee$~rGW0q8K5ir0#)a`ll!s_nJ44L0qFxP;TW7lSD9qshzloz#GqX-7d>b&qRAoQuKGNiTTQFmDBOA;6w&04qF<lKbteud>hzb(n4uFrY%J9zTMG3!*Dt-b^bdNYENwM#M&cwe%GY!i6>S_%F8KvURT6?`u1MnPnxCQ)jJ34?pC(;N|sNTEM|5sc3w?^Mx<<7S_Q!>QGGGPo&MX^G5z(FOpkSbtWb{K%d&XVg4_V+;XQ;a5^Us<gsJkJcm8&#uj<lSG&tJPe#Ml$@z|K-QMgA#Ia|9d&V$*y+2PWvZFa|GJ)TBX}Ehw_aVFAr3tfAIaEl%~+~QB0FBmn=UuUxhAl$|F5UYIZi~syRW*!W$RY?86u)1-AcKmk7=*~n?~5J%hb2<D^IvY`)IsnJH-a3jYW_=Xuf1rNy}yjjr|Hv?R(gRT2PB+u=P9h)UO3?&-b59tntj{otqo4uS;J<0yiR57lXSf8CRwS^;??kKtXD2^4J>1^6f0lYYMm&nL9fRXb#R5SCn$t;YYc}#1Py?c!c;<vv(0xjLSL!UeNl_S>Mqj_oLH1W<+*X0sCfqIOx1@kG;;0@IWRr9Hh-$%yY|os04vxX}0K#Ee8Ms-F8fMZ(OJX?Q{ol_1ho1=`o%1$g^@Tpf$~9Qb40vJt)$&m~~MH)L<p1fbn_`1aTIlq{7|PUVACVD<$0zZQRu+O0dl~M5AhPX8CNULK^y6uV-&ZM2*MJ#=2*s2lbv#sm2&jYf_Cdrcme~AVZy6z$GJP>fM{MS#^A+V{)(&LH8b&*XL1QLI4X<%j*M2#_an0Gt`xTU9C6coI^J$WrIQd<I8(M>^k20tT$Y_y#GpAR$A6nW0Q^m)jEmTuI(=5GEXjOP`U;;ly$4B_u9z`G!wlg--t}|g8OC4A-qM2Q7YtxkgQfB3jhf8Z22^Z5AOF3_7{`)cu^Z~t?h1+GiO@}iZ8%B+Kr($GuD>Khk*G4w;J<NEuZq~jD$<SH%VU?*wrjA7>yH7QQ}+r-F5^%f}v9m3fm2!T`|?(uW;-R&1UB{5LMI$!l0tI)Iv#QF{zsS=}nOg6|!f>?b7-3Yhujb?AjDIMYgMrd5M-&`tH1P6;f;{O<csSmowjNWli>mWhK*#g4j8K;$yFiDF1>Dz4`7JlOFFLRlf+K^R;9oZi*^^K3=hTS9=pwUlAx)#GrcXa@Z7mfgkzFbClbw^?VJyJ9M<Cgl)dHL~wnJMo#ph>T+sN>hn>u>EQsFn3E^R;`hbd7Y4nkOxX}NWWD<uV{l*<_3^~_iWy6Y#n@J?>42JANte0IIiImxil{Hd2Cj+~VCt}eKT{Xftp+cK>TH;LQrYI+_^uXwP(dgpUehc*Tb|m_3QoD<>PVR5g8&GxB?<CJ;~R74|2@<9ti4a%&}&A&d*lH8*GPCFwgfgbR)tS6IWT{TANm?Qk`XTDs-M25=}rGwrvl)9*xI0JrW5>Ce@maAW4WNUB|fqK8p(1vD+RA_4<)}Kvyse^RP?U$7}A*}T_OA~GysTYwdzU{quYFcG77g|{1qoni|t-q{kKgzo!Bhf8s7a~W^tu|pQz}+Q~-iO*3h>^N59JENOC)_b2^~_Fgmhzg+qYE@8@Y+7MV-2k5@W-VEv6Z2%d|&%~&oekERC?9hG$uksbR@xuqpBrciQF;96EW*LMg_kEf0~K8i)kV0=4*h7!SfY&pN>&R5N#;C%Mq7k8s~a}~H!Ab(o^&{ckmx4tiG{Ut@<CG(G}#}~YQFCe(!oLrOP;C<1;t`Fz5#-5I5J&V^=Lie;dGpUA8xCON4zVC(4)$2JkE06zgR9cL;i7H0m-)sd&Q*J~wzFA(-oSof0jrC#XeZMW`T6s@~ClzXI$$vR^;L#j8tH}3QR-%L(3}IGh$Z+_ko8}?X_Zk<g6B26B*ier^nX%1@LK1W<N<eYzD+Y!R(Qc<fWseWo1Lr-IQhUV;CekK9O3-8ruU;9RMDh%^mQN$%4N@j&ry%8U8Y1p=Bn-w=&w#Y^$M=6^*7*>+PJDZGVZEYJnRtH_2DNfr0Kl>@o#4IrT?ZP#=aROQRNEK+K#c2T)VDThv4peuQf1k>a{2wAz&oyXghqNRz0v5gwQwgy%(F+6Xm9#```*Bsv!R)cS;P6-iCSZB6WMj7%JG`#5_czL@$aY;q;xXkyLXwWX{0CdwH8HS$?6LfK1-Oy!}473hkz9>m>q12l^pA>tXe#{IF&L)uy^sXoruXQKg<0NC-gu7x;tE_9X!IU`E%<d5g85d;p5r9(y-R42L+e(uwV!@=PWU2+<-Ph!_ZJ6>4!gRxn{vOoz3-(vJ7VigMXHP>(<$Ks&9)wtw)F0>u&sFUhRL+wgbgxLdXgt<2`of1-*B|)ppq5E)+#c%xgS4Eth`jiTA8Tg(#=Hq-5HXl@$KLEdmu-J2s@XsvZJ5rFh!<>Y!-JtR}t$m^0ZcTuz*8ByDC&RGXrK8}2sZ*ci#W+&otsGc~tuA-R6e5(>~ZK^(iS5S*oCD?9`ehan!6##L4|co|3cjuu@IMpiYjAH%tCK+FKOR+Z|rXLWIc`eEsV($sdQGQ1s#gA55p(-djrELWqjN8N8s54RL}3yB~sk^Q25L_4m#!i|$tr&d9O{R%c2i7X6r2r|{z90|Ns2*reLa|X$)70!ioAS#NU$JtTAGclTm?5=VA3j`>|$@g%aRhb7g$shdEWAOz9PQKw3kcjQhfrLYJ^xAV%bxN<-#0hz)q9jAi<C@<IAPrIWqq!ycph7A-r&4z>qSWndMw4?N5>Y-UmBNbU31?;TBQf1>SQN-JBqim_2Clur6S+cGk{huMdbeDC(MgR1Q%d;1p~Mh=CvX0TCSUq4Roy%j6MtylBPIxbp7?k=E<AeGMEL+CM9bJ)jpl>&o$CSo{qo{JT9yN)LVH_etNPD|4}qDJ;dk2J@fX=YTMGNTUbkV>dVGa2i4I7AC&ZhAu_ZNQcy&?z?f#n})5yD~o~!N!U8{7SeZ%FO1Hi_&lh~lP>|P=3X>Au#q637oBvtCVbDgIO0)Uhwq%MY3;-%*>c@OnUL{@lFoP#|zU}|9?yy7n#={9SfaI<+&=j#<Y{$5JisI&xF<UeJGw-Id~Cjj2!Ppl1ya(cOs$xEdUz{+}j$W`f^8Lxfr45!tpEt_+RrgKMZ7onzM(wMinO6H_fZ}ohdoAq~hzy}ln_$Q)y*&_;Lmf)b`%R<>eo1(ma9UIE!VpFjh{$Lcrr(>IPZtah_Dw|LE)lKnr?j%FLxkJ`QcT>fU{=l`iR@OB0>a|oB3U%BB#KgZJRM;1J>srH?cA+WO%~sXh(Culif{bh_F&PMiAyMW!t&Pb#+$r)qHV0yQ^kv07e97)2Ne9SKj^18^3P=WPpbvLZ_g_p@J$%n|bRB*dT0c3cFa*o?1)PU{%1#9(i|~#RAKqi=l-cqln~5`s4o8gxTQh0Dgfif~oaMcbcZP)D%f!BTt37AO?9WjdxsJTBS2v^Xo5aL1$RGgtI*g-6qtOL6jdSG^A>Yjjqs+|=8nx1s(Sg4PYYdR#zj$pIH8_C2vN`+R3|w+iU?6!wM>ar$QLv(c$%&oXCs_u%4+29ByXP(1nvJiV+KlSD&1d!9+4RyG_>|30B&E-}-^ch0{&e7dP|@#5BN;Lb!eLBRU!;g>;v>Ms9^gx!;n%d+{{K>`fbOlINdL{fgVJAO7URNB(W**-8WoI4$!+bs5;h^jlw4({bzSHNSe$XHve?F|YiTtX*YRRgvJI>imY8Dj5Ie}jG0Lrn%L3c}GA0<v?Qu|^YAtA~w1(Y~jlI0KQz=3jHR}e*8ybjWqF;q|GH}kH5|2k?@%oD&N6u5c!2zd-1}*Ox0351}^)S>hhI0eg1>t}bK3bXCTE=)y^x<r!3zhKnf{xPB;mCD1$<gX}4tdeY8jgh~CtA&PbVN^iCI&Yw|9>9TDq=yRC>>fK?ajfRZ_Y9SL5FvNO<VE!6dnaL*~z?=tl>i^|L_{-LkQLGl_bTm>1FPpAM%gtbItQn{&ST3%pk*N|BlHbdR?c?U6Oa`!EBC8!MgDR0r0)ca{H)5qjPn0JJGgpg0jJRa0-%Wu-7k1FAyqw7#-f*zC?P%)2+gjkaS4B<?ShCEA*>r61uG^^)M=$?a|69z=_x6s~&-Iyuu69l@vg=u&kWZ@2Vkae)(Lw^31gw^?21rn$>5@0x$%>a(<!Zl!rVyIvW?~s4EISFKQ8?cRqwqLj)lc9F9f{3s+}ZW?!ATqNk~6JI~0(y7)8T^D;vR%t}eVDf3p?VTlVN7n01r<;S12?v51fe4oo-(11dYC--}cSwb=^SQ-SZl-jf7K~B2s!Ey9=SA9zkX8f3+yw)<xXaSuUE<kAq7Cg4t?Px^ZJ)T`F(YzI%Q!eK}G8an-GpnwSxexONpNe&O79$%cA|~RXvV<C4<U>WmB{#o2FnqqEc{cENY`Vx^h{y{XZ>s*_dhwHa@{A5}tmrjYO^OTzV?OKVSI!4Io&KC$K-UZg<dF)tOR_4+oI#3xMQ|@m!;0*3#)0$i2uq9yQvsF`W8{}*mGXit5}UnIPqN|}@B&C&?M`~+F2lD0$85JyxngC|(S(jLgkEYQ2T#qKr;r4fkyd?gwsuUa&hq9t3ng?^$0RH{fq()z@DR*|K(4n2=>K6wxB*@weL6+ukkvBJqNyeVR9|Gp0dC{q`9GU0{8W$C!tB~VV;^3&W|ucLn1W`*VXId(F%fQk6;_>iGb<U;FZMDhd%ORAuy`K$P!rPvj?i<kXt8G38@8s6ZX=F5s{#FEpYP~Td+q$W_E+%#bKA5Hj{4ghE$qUtem-N*$P5CD^xUc3oc_&np!p_~kZU?lE!}GZCVhPO^*8|QUsvU!kUg_4ltnDbRfG%=NVWIQNSvG)PuGWhr94(RO}q)9A;);rZvAG1z*n;i$d&8*!DaCcn@{)Kvh_@{6`Mo*R%(=hLhl?-h_vO@4}=(+dLpauOnFg2&5!8(b<wb%?#^@UoEAL0S~9B);Nc~MSikZ|bEvN@-mAB>7Ko*}d1v_EvbSs;0gM3iaMsqmT2~Py#)#~c#;#VrkAeAF2m%Hi{-GP7af=rx*t_Id5pZG7OJ-<Serg!(dTW3Tu$<?k1rnJ7VQz9ECG@*rxDJL%7n%blWKBsC*$h!uC0@Gki)(yb2&}J&L=O(s{9apQ>5tc^$t08#$NO6SoP>L?01xgqgN@`dYO4;gKOw%QN}J5Wc5OTy#=eaQ<hi_-?S=iwn`D{AQ3gRWvb+Aoh^{3thbA9<7FcsT7~62?Uk*N_xXs$RWF8pzwG<cM=h&b9N_I)PRz4FSuxv?}R1gi9vKwKo`5s$xOB>C$)qxTfy$KaIrK?5d)mxOd*Me6WIB1qUTX<$hNz$)d)0rFERHwlUf1O_vwObPV1h=q93kp85cAX#}sO`B-1pNL2Najk>F^LsAR}PgBB&`^Q4STZ4;5ST78F(9~oW{V?Eb<x65--Z1mP#stNRrVKEUtS&Cj?6LcxamunVyM;PP|VSEdedBO@yM+7hY*=KIjqoT6<75@5<gc0B&Eq1k7d_bCfiCn>=LkTCA+qtByOoV@$-mCcAI5KUlU6DF~T^*LG2BgVS_ekJkV{U?Dl|cR?{6fw-VdL%X~%Z8Sa~C=y%SZl!jzbMaP)J%1T8$*_SlIrtyCuXQ8y8rd@dd3weNEKR5<jlGEx9FPnqcD0cb>>(}kjb!#HxacsG+w8jzfGMk3?7z_c7cY3Ea_X;{kXWv&N4<NR-TEk<Kx^a!AuG5jilNZkJ63>`K>;^xX<J&2K2bX>;K|Qv1RLxA$1D&N{se14!r)+Mhr`+#it$0}DB9m3%{!7uvf@fK!iKe3Xh7^aL$Q;0+IV>{kj<Xoc~;uTL<*@ZUj)x^wE$!+f`~TppgGslNH49?mJrsLK|THF_1G5Bu3FsSlnqa3s-DuR9hb<ue5yx52PG}V$Wzus9`p?vIRN$cGoewuEK_FB?n=G@kK3<vvWK>{fLhov*>P3qlpzh?1Q_htjZ#a{G>-$`Z?~Jr6D&rL6FdsCda0~V-pWcblv;8c7vs?&GWu@6!AR2g@(wZFXUs3$Ga`8Yi@QWiWhkA9qq%(@Y6IX?kF(DOcpaMpF|%<@bmqqXTVT-(zLNie9&f7XTc{$D66h(%2lpZUf{`~=l$>neB0hOi<(GTh#|2=9CRgF#>}Rc0V+`E^NOxj&gZ5wqAuBk30Kf~Ae*3!OvXAy-PVPpPKOP_HIhdI10tY}V(wnl{ai}BhZbnI~kWJY`{j;T_itC#C-Ct*N!xi5fUT!D<)`J}}We^^BMz@Z~^}D)Mlt(ch$~ceSPOV0Bou%@CMGa3jc*?v&VTog*e%~eq@O3`<OD`NNPHtNP*DYI+<_JF`14(ry)rqZIc>SNs9fypEU@lUGpYPjpW)B+o*}jE+D+tM1`GaV_6!paOsMhz%$!fI7eZv3SwW7(4aq%~6s_B>nY|*OkRlc$*cK^gg<lhQ5`@gi+Lj~YsQW^X>hIO;B24#B``k`u6U){dF`BKeTMjjF)pjpw|rViA!Ynwk3WC=!)Rf2Kv4TrD?^Y`!4yK8*3@Um+^L?#882H<X*Mr90v3O!h*gjSB+55`(@$|oiQ!cb~m%Nwg9>0qhe%MNJrz2Q1;eNXv>T?-a7q?$I0{-1nAD@Hd(-$V1^T(@FW{G^6J85+i2Y{S>N;m$CJ9*TV#@w#WFaSh)Y4aO;3!pz#2^+Gk=xOarfNJf_#?JwDli<EiWrE{@f^kkJC<tn6^I*bGq6UAI2u&rEjAX3j?OdPA3S1cLHeIlM0vv-ShU3P}VRHE|%-5|x50&!<%oHEU%{I(KDa7*7M{hjZSF2^_zkS~SASDJm8Y9i)_qGaz7$9;_7m*MsfB7C|_{6zia5sXN~6N$%0!pSvIN($E;R-eW1Rj>|*<UT+l^Vw)?LSQTQO0rt@*juKAqBY><F1+mv9S&BqLGy&c-9C#MHCPtX_(MN~<m5MgK=7u){`NPZlay`&s1um`wFOfWu(xj8-i~}u<l~m4vl89P0XbfaU)6EIbN}jh>rw6)n3mVrZ}+++Y36_6Q}bT>JXVPJtUzC{o<(8(Cw*OiFw#R}OJsIJ1Eb=dJ!)R2q00tNc}4~q*L&Us8TU_i?{sVnz?h8L5cg0qVb30xoYYwc1pJm@4S2361y2AqoWP=9m^e=8FTF8kMb7?q1`hpyDF8gy_L2{jCjr{MHQSIjh@>EHtf`8WR9elJUx+_YG@f*o?f-v8Fbou%whxi*Jy<*dl*4cglX?_!1Zk>DltvnppIES`K>5RHv)GvcLICCa>H79eqq7@O0<_bp6Wfc4qVYRCXCYV>)LyD2())YjfF1TZB{anUB6%gNxWcZ>r({4(2)X9dlioMXv5%<W{F~pv3y$Gz#rZq1r4@>JjG!1V`1GT?L2`S_TDo)8L}lOm8Q)HzR-s8*E6^r{{#JCKHX!iT)~_*dNQKA@05br}N)N^j%OjE$yH|aQ=i?%Zs&J!B(|Q$in~C{JCU5mSX4>az#B!K%;VfV-T_PJ!9P@o|r};#_stLf=K4fRJhi!Oe9~x68xLp|-?~3?<HzL|0=bfZ^tE|N-EZy6>x)fw=K*xc+5Z8*Y{h&)SKmzY2Wh0L8Cy@L}+Zpy(sd_c@$MPR5SWi5&=Fp&{j}{Z@i(ZN5`6QOiTGVF2{%Ge#gF>!4{!N7@Wyj=$!LXC-6|?yYa@HrdkLEXW=s!>)sKK{*MNXXX{gs!PpiOU`6EtRn2N=Gp?NFXgSF(@W@2~8^5<yHWDFjw;c#ZZ~zjzpelAhSIVh2-uv0_HKKqxHG>#1=&+#!`AeJ8_amWg95!l`Axlj9@Lw+RyI6(Lr7g)X*`*Ly*}0_>Jy{=n5V<zd!w(tX#bQrnyc8)lyf<^AgOQ&wv<(&Ehp`Xt7tfB}G9ZXu}MwUXaMYAafyT`R9ztl-~!b3ZzU%vT1S_jkXNqEo)K%s7q0S)S}`Ss&O%nD{1B06Ziyr$K`!GjmdK>=Ch;imp3+KT(URR$6Y9A!CWx@o-n{={5D|wQmQKzdutXp@j6V)O3{X$J^N>l&oyQo;bcHAlY*JIz?|W^xtJAp+imvpkp&em}g?(*@GU^wOi!pfo}z!)d$`wK)^XqUb(3Z+Q7l!;B3a3F?{hAqVeUx^~>H0td)NK$SL!PZQShsy61-~O!xuV)YJ;BjU6WP;%|5x@g9H<b6Cq)5|99hZZzC+OY%Eg{Z)&=7V`QnXOJ?=n|Av)G7%K)JSz1<p8!Y%i^hvej;C(yPn;dNeZT4w{|SL!Y1u8C;`9N8miFX?S4uH<;S(Dn@F`{uV>9iBFW*SNtae#CnFZ8iMm?A-d^=J{laOv)=lxc6Mgqv`66x??+wmsFXERAVoC=6q(oJmt2+l{Sf+>@RNu-wS*0G+N7cpfspq5n0>jA~kLwa^sn8DTAueW#X03}Y$>93FrjvJp6jZQ~HiasL}vgc^U?J$eKVTp+}9MunYaec1^LF2nA!sKqAIf;ez99$t+^pWL;S-r52pz`biwy+4{$v~|{{^Xs+41NW-$5(70vOAF}A!D>%t7D3@SMXtK`cIqm%d#owk3|-c%K<%|mhcaVl8VSK_eoo}7o%7d__%7qwzt%aiV{sIY$eNdLY@HPdFX@Sisy@1g)v<T+~QdpAYZ({f<bFRD&vb5ZD6pl<(xqNl87%<;2bREE&wu*R}>REqIYX=rRu?NvW=ZeXFl*k`aunmFy6wIN`P>P6o4-RCQP&(!R9(a_pvQSTjAZ<Zt%sMUg}t=D|DBP`{gKd&@rsKC(aN?VHRe=47~`EtwS@cTJI%tu!g}O@&NI-;I>^alJzdt<YJz*`*bBdp%uddc-zqqa2$&{N6{*#_XIQ}9QnEZ+Nd;eGjVQSzRR!iZ)yH0!&#tl#*z!<N(tnt!jfKG^B~fkwoGw&OIqijHW}4s+%#*0gC!<Yjn371U`&ZS!a`TKFpmX3pw`uUMsNN@(+>;WGqinQh=bI^GhgTYv~X=cXwO*b)J2hFPr5aiL6+-mK%oH^{P8^t3<tuQuCa)FGY&28pE1n1I-8(-6G_1O8_4x=oVzzkV}9e>6yTBJWGz}j5-6l1PPW!)An|fNBwA&8d25o!pj(!scL(vISQutQD9d>s-5jDB^6q)odZS+$hdmmJcEZWnf%c}f#e&^M){+Qnn{npGjgGO#FI6swr4hhHA`jq#r{9Fgd)Rs!ZDc30cRE!)_ucKzDd<L-N25R?OFaqEqPkH12`NNL73d+;HTHR_HmG~{<y)60j2K(G2l=-X)-%<HC3$3fWC}$jMPXfCm}lMknH?yZN9lxR)mAlkvLdfX9nTSdZE?nD+?SEHDse<3O*<jK#9}`c;arOP6u4WYIl0S(@JS_$1}nO@<p^jDwgIUB)yEM=^c#mKe_m-)`Rqy@VH}H))qilx=wcxZMQdr!?g^<ycT5M-^3-7jLjGZ7;x4#hSe<-ZH##@Yp}~4yu1Ly_#_M4v9(yb)q)pzL;7I?|?BK|Re!jd<R@8ud<0ZHq0C5jQS5}{kH?DNRJe31_Y49CoaO;SYHZP6X_8<4%Qr|boF;6*5q!;nE0#yd&^qY7#{&C++aOW9I%;E59j9(M!Kn43M;UInH_=(8LTC&~jaK2GD7LuU2m#26oaKFabBL9yvPS{$*ZyAdP;!T0OSPt|CcqsYnTTuQ${^p-?B_QIp8i_dkKHKn@Smj9hWo?mS#vqn>BEHQqOsEq3eU$TUfXPxmxxmdsGS1;*5TaSHObw&)ppi8zjvQz}tojOtb!{GA+^ZHN=mJ?eV8bFEsgDm|#lyca`yRGD%CP9715ML48(_GNx%BUPg8~mV+auFCq-W$t4HG^lKlQaS4_~x?{A@QjE@jv4!>R^M=qiE@nvPulPsx1)s!aJu&J4c{8OsdM(1t=velR{wgLq_+jiV*;N)<(sT_}tw-%$@GRrnty%?8rh_K{U_kyh)y^Rz<mJnx{32<FRqwjH&y>k};17X$%$QvJ>nVo82t9m#x8EqWUgz>BwX3HVcAyd{XEvQB2W<Y>-0snQC?=1Jg9oxMj&vHEEr<5H_k%NW3AO|KgLa~@+~Zi;WJ#ibXK>p7;p*AmD+qx3r^k5WDamJUT!g&T{EGe6hr^UxO_zhD-yVuT^J0*Z?M4MBIvlPEwGtHh2&8Xjh&yZ4sHoMEdMg;vjs3rV;|XI)_CCYDY*{$Ivv_SNty=HlihO&ZxdtQ@t>NC}ku+ayG|Xu$dd=nVaRvZx9PW5X0sT%|eCk0R|yr~bp?pI7byq|2*jdg7vOQ#01nd1gLq*~N@W^`N@J>w~-~+2aGjVwoI}`DED*KRTGg1aKsz0<qUm2S6!61W3q#V=pl!ZYZUN$xpnf|HFJq98q#(%o{z0(Fy&ofUKQ-7$YzKe7RrAZ<h#?&&i{`AzRaHSYw-0sQ=M;P8S&ypTtxmMESW4T$#v$vZLxx>x_+0)ngDL-ZANGO!bPhbhwlg%^)u`Dx?yxL^^f0QJL51lwA1#ck#`<<u2}>YA^@k{gL1L3Z`{ce*u@uVsc}CiwW;U81*7}Mn`W<eazbuvaA#&7WW>3Oz&eWZh`_ZqqeO0bDAZ#XMmuqP!iSq){f&=mpb;Z;eZJ7Ys3B8Z0EwDQ#Cmdte_LOIXg0B%`KXGvN2vGauAHpFAGM23va|mhEkkg4N%r1Z8yU%<RO=7_4sepINxJ$UKkQ9Zk>wI;d8DE&$65))g{JN?;DApY44%8x_^8x?jnQ}Nh&^e@VuN<Xh{l<Ux@FsWB<+4jnUIXInC1+NWSzv9dZbtux_7QQ?{9Kj9R^Xa;!Z@=ukQ8!$7wgj`=H!DPI!>QoyBg)H(%=mNc@=5Yt-GhB2K2*)-inv=ofIEly?l?U0LrSh81vRk~>#{655}l;m~TBQJ+Psj3V2YcV`Xf;+s-p)~qFRsIL;4O+I`$%{Xu54gc%()G3JVusvHc&U-HcDU;ysf6Y2Y=x=RN6b3Odbo-ym=X#?{0m1k-DS16+wdwFm1STDnEGp(Ekjso-*r?HVMp+?eMXs92qB2QGg<}YRZPi*gnDiIF;}i+K-avdlT~>r=+UV5lFs?s@%K|CtCv(==C2|XGKd6|_G6DHf{D2fiQ)0HSe3vDh{p!Iq#r85s6Fm0e8!1YgwG=#_@F{}*c?#zJxR1(ebz^|{nM1C))9R_pg<161@o+73=x7epsH9^8yz1dLm&EQ;ZwXFQzHtgyq3^DP<gKY$c66|Z|_yJXSiy%4iZxPI_P*a0fVtGdF${XiI7H~mkQnHEy4q{D6Mpgjpr&OHyM!FMi8+blP9&V=?wUg54#lBnLI#64a6fgX;}4K^F^ua#2oD_sLecU@gn!A7O7$jXb<NaBm-~+p_LI-t(tbz1LJ4N{9!r)46Sz;Mj<Nht*kr`y538D)bN*axva2IT)N*OUg;+VQkT~F)}RNOB`apuuD|hD|8cm7<qzK7FldA|#34b`{)zWJn}Ml?k<SSjqE-A^?o8)2<VzQCQ&yEA?`Qro`ep&@EK_9^O^9YZJGT5<aFJaP9<+`uq|qK^0mPbX9@2_YQ@FyG{bdmLR@z!I@ltH3Gn$dy3EOyfTp=$7zmGc>v=uj4uSzCP_G%3Bf0{mXfKy=d=^>CfIu4pn{S_PMykDIbW?^8J%_IpiY3S1Orgj@J{@t}cT%_|6`~lIgb`~$l^9ITRD2b?*$(l;0sjdD(CP-GN2DEAA`NYdrML;5m6fT)9G1H((1m^Gi1j3CD+-=`U&cQXZvcjU$nfYvO(P#??qO~Lc3X*h*0RJ|3hthf7xfKwIMYLni9zYW@Q}#KIdA^f)dB749!OTvgix$7#v!$V#rM4-*R^Y@%xuQkaP+DUN{v0Y}7k(p(O+F@qZS;V6KEag2axvYe6o0=-Fg_+5*1xN9Kk)arLIe0V^Hq}aAPP^_=7J4drJ6gdNUaAQs@5bm@DRw<QJrAF2ZmXqBhKOPr(!0B(&E~TlW>r>PLd5NYSti7qK(_~hB;F7-a3)UKd~CJ$(->-@$qEc!7>c78`n5)>wpBsto8s_g{PV`hG8hq<rM4%`djjp65sp3uAXX5B_)H`<hv12_+)X`e(y0(%&6S2hELbn?;J&Db<d~XInP0OfaRplt%uHYo#`I4j}C1c5GxQ3*<}%O1*l=&&4w|w7sTZK5XFMYz0z)D-<sFM{_o6Zj>mAX0Geym#bElS)9FH4@V_(}dcv`Vw(k_F!dxMk+YBm9z)$v=M%QJt-zkWzO6LonvPn0ega5e=r2!Dq!?f_heqm&wTtJ>prL3;F1V^r;+X4ZhnQoT$!l`YT5Frexpjy4C>-V)&$H!}OcO-I)&C6vj8QSg)q1o3v_?FW>vkL)QpZu>a(*1?N5t<!d=RYN#_Vd2cjOBkW@k$`~r(NxrF?v>ASfz!BlU7fpgEIxN{N4=LgHEIoDV02DPFD<AZm?5Ha;1kfFA%KW^Ghm-QGGzdD4#gh`VN97glljzyzykLPay}XTxG;Tz~h&ygPfGN=1|ncwv_Ik5LCtsM&2*%6yzi#awyX!(|>TUY4%w4id13i<8$8>$&blNC1B0)qW#g7Q@*Y5rsQIu**BFrt@gjfW1key^uvG=<Hy^cDkmigRo*;}KS<GikAu6D5xjD?(K}g?H>%8O<uKq+wUr~oGYG2d6Y$%nn%*eYiNnW_Xs&QIuwMLrpj~#p<sqsgD}|^)w{tZ&+!@*19rj=$a<>=lE+A@c1cw9H8afe-FR5U=j)QX@N<!C~rx&{ij|kYIow~7Z-`HbBClTDBH6oWJFq#qNJf?DxF-j~I<0UKrkP{-;^A@Vikm3~OxDxBqFba8TK8F0$7gk(sbF-q|OViaWM>pA-yvSEvdE`f-5TQEsLO6haG&3Xmf=W$h%M))iHLjm(BMa~UIme{OUkiVPd9M@5wk3<l$|_t@s0Bpqx<14t9uXnzAuKLg4Qr8cCA^X4>E^`#88VCJ=1<Zp=j$zu4@4Y#9L)|SK1P0Zi8i1yH1K=#903<x$n6~Nb0(=(zmSTrm=nNesMzGAGlOfJd1brqs(j8R9V8HJ!%Nbt4evx#M%0*H30!cbYPO|<#<JGr?BuSV(RFs0TdJ&I5pDJD8CrY(*CU#e%J+xIn8pZd>j6UVy;h7d{T`h6!O-t{@*<n2Wc-5zwzsE_TjQ2uMYr0IuRu7}v;<t=Q=`QKO`ZNANQhtbwE7%FU2>3LMO3Bbkloq=YwJpCkOB$h&UmaInBI%pLr*qC-nd4FLi3T~yhT!!^O=_xMX`K^X*Qj=2D;|ZiE53wi-`bIPcP;$L>QT5T8ve>puH#svG@`QHGzar(3ZjYcGYkQABq=`PMfv%+y><Z({qx7hxlvC<VAmNk>!W^t*po@SPi7FKR*un@3z<ErDRY*p<^gu!aPmn^z=)FMJtnoHT)J?=>c?p<X;jIN7*}c9afF;Dp*2`D6b-*ygfC2_(f9k7DX8OswV&Fr4wHVQxlLVL#y%4^R3Gjjw!Tcx%6>veBL`XX#XjAq&kJ&4*-S2pAVmJ+lg3we+x(Yf7#+s@RkfD101bW#b%QoH0%_Yv_REmhNx7hnF**TMTmo9-ZHms@6YnX5SWAD<hSjK&72Dn$HdeX-N{cbUqp<Tx-}KfdmU)t?|aC(w-k)W(IcjA`p%a}=pr@n9F6)K+#KL&;A!l2XZS}1*^?2UkFhzX79Xit81}ftj|FS<*!S;H&v=_?3_kEN#wvcg5VatpC@{sAA#xZ%NlJYhAa8X05nE&@2!h?d6`lO{X8dZZL{|cm{}AVH@d<9%^6Z(vs3v*do>$a|RYV&KXGKay4((ht!zyc+FH#&%H!KZCK;?5fek%)QRavq9nq5=_&8T%Q<Gb#hiv@Z*2u=aX%h$L5k*BeL?JZTU0|(3{y2=+ai^a$`|AP{Oa=@!9L;iBwSeIo0oTFGNe1ULvyZJkk3>fH7{^3yrOHn$QQNoxO{n<7cn0{G~n4!bkk2koy?F>DdF0O%i6s2AD&=@EGPn#l}TvC6GVe;|Uw0rQN_DaF7Y8Gv0CQ|>d@NbxT-x6up3HFoCbH)kmF|PdcJep>xLiG8gt9P!}_)zPea#>F@r#(=r7k-e)NY)x1M9X*PjMb%o!N?T$qj4y#{#hVkZ7jw{fn+duRIJNn&raitb6g_H`oz?a79#Ay18H;1)O?|i6=E3$8n5BUkgA0ol9#lQKWfx~Y+{aRb;81rlp<yl*_n6C*Pd-8jhX*RobuKgEuldOh|W0^DEM)4ASrtnB~_(Za@>fVO%1SrP|k2?q$W<p*x^=POJ|(wfTAegN8`5r#Oh4#^9As3fG;H#0f1{3lZ@Kh4;m<T_aZXkm=xaGFXE2j+?wRdy>-ZDnC0h=>ZmBY05atPt`OIx>PMXb(;<|xh9y(>=K{MKykNJ{Jdw0?Ct%GS2i@hF{L8H*N;zaf4cAnxYOjdFP{tO}2i3~ngN{*^2S^K>^f{8u%0k4~z%+kRH^*?{f_)X?>|(Xrk{=`7;fgAbo>qIJuob^alAZcdl~49YTQF_<hNak);fqN!0Qt13TM#@%y=s#Cf<&?Smva-3lhrOv(x*<cLzat0?RCMJqhl^f<{#`Wo6)!N$VRQ#Is*h~t+J@%kgvXtl}tczPwNt^kAN}hdr}HQjw@Me5O8}6*{M%j#ku_EcUeL|yu&T$9N(~eb2&drGD1H;C+Hr-w~?oTvv-virJ%^AxsMqi<$s{5>4qZSBsW&wNc(xk1?quDSWI<0x6EvoiiW6Zun+e4-=(GW8O2NCu8TEIJ`6kL1lLn-&zEhB{)NZx;|VdBn2RR=kIxw6Giy`PTf8Ghp-@z^wb@O@QnCd+#r|E{NZ7f5YfQt1<Nrz1BTuql0~@D33=em(Zk93D{K_#^jjWO8)yykUWwZ8X4$C|Dr75J}7nvA-dRFDp3L;a*K}`!4FJz;)S8VwQZ-UgBDNaFCux5R9xs;jplMLCo{y0letC5TNoF(Bp*^G~~v-HT3edtPdTqkrEoC`ZH;O#NA!ELeWcGo0-THJPXWAEv066nO{=R+$tSOAuqAs@D=DZe3#<AnU0)BV%K!aIDGBYZ)Y@Z6c`#JcJI>e~0|Ck>ks^Iu+FsNUg3#X1?o^-x+sd55u>qxM@OeqcKt@{O}tL(s=tr9U(4y1jEEvX3s726PkblkUhUAsEUj=^SEr+pmg1?nn58!DZbpzu{+%CCUf`a!^y?0(*T#I~`XW#q<Dmk{<?G7&ZkYTw7K!U_)ETns39RpObb!iFHfFPbROqltM+03%Jz!!+i2s-~>RBMa0(96Z$WocfBstnt*<_cf24|P6CkVnpT_yo(D2X!e)u?%5y|p1!uo*^4r`evdiNaQv!o)yZ)V{&LRL`RRT5Yacz{(W9j}uRiC&HMaS%B^<H^PC^(x&!c-(f^C<4j$w0k>3Xpz|fOE*e1Ijkq?8kBexLEWjXT|y*WfjS)6#=t1*dL;-3fe@w4v~oJRgHcH5JqTC?bF(Brylg=6s=^?zql*AZ>4rrj_Y);pw-3kZ;GRbh3m?5$;69Whlea6LTTJ4AfP7Ti#@-25F&_)m16cN6RtHbeC9}2Spgp-jeUcKE4N+l2wRmP;u(9`qWY*bJepqErz2G77oqep{wRW3=gh7`UgBDjt^^xCRp5}vPAe@+vSaLs+9?m<K$AX2x7nSGPc(Wt)x4+7s^c7YS~QQ%E_m&)O;7J;#T;@sI6CeRAJ~h6`yPE*Ech25^g$O;Evy8esfXiYpXJorC|B0Wm6UL6yd+@Zz}NeW^OU1<6(IOn+8bOrl6)duJwCIVP1>>HxUaUX`aB1CmR4$wgE90{di-4L!F#dDj?SqgLB+5EGz-AuedMs(+QnFzg*1SViz+@#5umiEQNtn#Kee!NBHcIs$*tPTRhi0etyA*ROrT$oVD;*kP!%Krj)OA5y&yeLP12Ys^<piuw#`pYph=EMp-9@9Idl23J{oCT<zShdVG(1``eD)elDbvda-3-Wrr0);X=XufPQ?`d>G;xk+-3y%PYyzE!B2U%TPU#TgxU)l;otHn)Y90x#;&UL?X^)e9r@2*#v(ML&xRx0ia`Q1yl3}^F7IXFNN)#iy6^Q_Wg<&Ks3n-Z`U-)P=BihX33{A*tLorjbBYcjt|SMHuldm~#KQGA?`*i3Lf{u@;@zk?DfRG7^X^>^D&%_n<rr1=sz;SJ)#*AYtEtO38MiXO`dX&@72an0rf8QB^o5og5Iv(>n$vxgOu@B;c73=WTE+4&kT_h8d6)FMU-#+E652hhDRf&5U~8?a$gY;fF(|0^1!bYk7cAKukynLagtj0;H}>*l<f6jPLDVNhdJ3h;bfSb<!DYpq*p!~)u$qX-F9&+Yl!~hI9VZ~&SlMVhC5wbE#WAgERF$vA$v<8p1V;8bxJ$!u2*eg!`_=lp_dv!{PgR4z^SDBwmX|*g<;t6i_%#~~S%}O3Ahiy%B#{J22%))oeDu^gs=*OZ@D%dRU}d2lu~kd)A36fJ2q`oK6M1N4UKV@^iH#S1xv}{yy0f(u13{CnP6cKFv)}R~^-}CgXBf#F(3kYAl-J$uG4zmj3;zOyiHE;zBT#eqBn{UZDe&Ewvu+k$^%-u$2Okt<6ep0-PL4x>FOh>K#A=M&di@~pGwB_bDf0my^GmQf(A6T#7fM=hJMM=v@F(j|X##?rN5`d|$e^~v-jh!)diz2`08_G$1D{sa&zXR&^1$DPYrz<a>zryOUUu>rQW#ab048X6x5MVyXN;(E)jPNd*=959<GEzBkrSyYSUb?&4&iLb7IT;is~s=6`EpWD2&5VZF%zP9S1X+fBs}S_12+iYhR_3tkVzmT81!>}$4bX{-*y%kt3%DsHG&f<!6JaNJbjop^8a+ii)pP#0OU-LPS><vG%mE{3#mQ^LzJlNebgjpW`f0o*YTpB;T@C-7dX2lCstp{-VcyRz)}Z?y2D-%$>C8*+@T)-JZl$|RPNrtL&}p-!kQ@{jKs&n){g`TeO;qDre|*xX}a-f@~&~|Icpl5p5Uqxfa2Ew%U>&ockOO<X37jjIeo@fx8KKiVhC;%c!Ezy_7CO$=ry-1`xBaNp!h9k*nkzd)VYu)i;(A&ff$Pfj92&ONUXbf(bUX>U#o<~lDIvbK>c;1zngr*kU+MnHwbu4S(=?wkL>~)F=5h=pwG^3T5uJX@BQX?$;<uH3qw`|)}p`CQmT;v3wEX1tf?%`*e*wutQ|$dpz3!D9lGvYlfO41W5jm%()Jn@Jx;mGd~;{$<>e@yjeY+ULszfhjZ|1Y2#Wg)YLqJqIq*aNZ<4`!D?@WtR%E0WDG^!v!~;)(=4}M67H`Va`Au(HC;5J`Im=5oH)Z@A94aZqkl4D}36qG#H8W^rQXOn}QDYRghlgWxyJ+z7wulF<S~~Mq*3j0xa)$vmJ?Tk1&CbPuQX|gFOQmYRlU2YdS~TKJQEwUEG~J5x$+UINfd*vcczxu(Dht6Mig;0myL@4^I@!IP&K)5n$`}thkZxr5m>qx>*Q{>vLZ(ac`6RsT#xs9{(B$!YHe3LvtdC%ds72S!SZBIyvyau_6(=<OzV*8+n2q(jmlu54nE55e{0wu);YxL@6+;z!S1C`(N#rS<D35IcyS^(92i*=~6i@`6ZY1W%t+q;o6u91e8XxX9z#;Ar<(6J>Z>I7oZl9}KkYf=s2f??K2w2kiU*h>7&{+31So<EIj#Z;W84-^unp&b}8ybv6%8@$wfF1bVqZ6s|^noZyBpN#d($PB(sCKYQMck-c0GW%uA-A?+Q0(ECM7Wy$m>`ivB%U+b6$7R!);+_6^Z|@p-u84HPS4MnqaE2@n@#tte#h5?Q98>gbE}xO@CdAq+Wb27!sir{6)Iw+Y~WI%vOra|WZJejAAjZ74$L4zCdeL4+x0YGJ9H&0{r=37UaHsoEx1w%?;9g>6oM(gFT$Hv$3KinA04}J*X;dUOL_2IxlfsBqt3w%6^Nj{&%okk(i=YQg)ho?@%wop`?O=gF2j59+H-%4osAY?K7EXDP|mRCpMf4xOZP#AW()=Yq=Nfd7s&GZlk$gonr7T=fg3c+d<J+)!OGZULmkd##z86j?bluz4M@fm<EV2fM@wVM^{6bP4^TrhuBshU85SEV4&XtKba5$<Oa;;!P*sC7Lhg7=zdGM-Prb!68IgN3a#gz?gggh~g&rCr{Kk=&0mks9Xx3D)nT&X{eWud=F^>{xS_1ZxkSk(M86)?g(O=5)@Vt5;5xETptpo<0D>cVUf_pt92lGPh{wWoyvA+w9nLgptBdzC!H|rR&u4oc3R#d_aGP+IoW*_$Et#BWYu>mcwSIqyH{fn!t@`l|W&hu$267}UW73Zr}7(m#t+dB;=ZVV`zGX^2YAdxL*6LY}wt`gT`Oup2Lu;pXF(<@_Y@?%F?#1FKW##ENdJ>-*K#!^|L5ekC#Rg29fjC`n6?D=k#@@7tX>{69sC^n%>U+?q>uALh$I|M4!K=88}r$XSE7XCI1B=2g$RQQ)gbj_F;XiePYEbWkR%5dwX(P)FB;e{f@34<7S;vd-62)XT_*8Up!9QcnDi1}fcE%oGW<F+@Y;ye|9hU80otdO`dt6Dit(j>o+A-O$1sa${hkELCe_&yKb>O;J#2CUX?B$hHx^*>Ge=yXGfjLQ8$^Ri*_{8OuxnAU4}z1d#1+SK(Bo%L&~=QhsLa!WbFze9yf<jzd)=Zo_#F+1NFF2%@(^eiNQ%k-`CiAA?3)UK>s;D9S!2`Y^Zp)T+nZP0pYXkj3ivI$^kcyuB<r0_JbH8i}{%HrgwN#Z+)a`M3MxBN*Jj?G0U5+&l;L=ZdEVS~{pN|$WW7o*N^oZ^A`{f0u6j(ik-rx{iX4uxlb7Xs>s%jwVNeFI<PMK=yNx5A)EeG~W_W?JDx(*e>$lDEehbs<iIJALm@!rfyEmd^UKsNY~h|EgM5vo6=MHxhOSyTlgP3-KN3p@L!EN}gOJmTDHb-uzW37unV~&eI5VZW<grf(@{qwM`PE9t<<L|D7A^>mBE}POWANBw#P7c<o)0{isE|E(}hsVlyYssK_1@T;P3nYVP0Rx(7zelrT?Y)fvoh8tUQorN_%c+k4=A(Ol(SMY(s`uez+oV}Xyh3Vh}5x+T4n7%5~0VTSfB1!#bWyz@VkNG5#>jbUBz_`&{D^`S#Ph<egjcdNAreb3Sc#d)-)2)S)Rm{q9-rCANPnm)@4G!XLoVjG?9wH^ZY)8M{$9d?bimG_gvuhd{v&sLs8H?-6Ht6q!Cq+-=*7~Br8_>t(P(V2N13l|-##L$Xqkb)!*rxdp$n45h+gQh9zS_P7&^Ku%2v?Y*%b^At()xv+_{O-0X7%ukY<CjSQuw7gHa=^xO5rc(^u+i9(!i(M|O2{pct~0b0O#N=NZ@`2d5(9Sp1#SwJwh2+d*o<Erg9CJY0|3N4Rz1S<ShV`2MQrSw^^dcFRhkj_rFA3&JW>E(>>}ISMakg<_Iui+A0akJe8qwVcwxBfQA{{vyBbZ6Z>I=DDX16DWy+Mk6L_rQ+($z!(#%~c64H?aUwqMXO!k+wgv%Gm973hM6bj-8(q@qh7c#DrQMF5|L)~oD%kf5jYwughs})+hPQ=$fs8w95US^-#?V{U%El=JyTE==z#~Sb-LyOy%$gXp$RrA=<AMkjS27^O|N30lSW!{S34EFHAT}vT>?D!y@p&fc$*89!ajbUOYd=9CtW_}bK_&99fhN$6M0}1IPpX4=<6aFc~(~Mz-aAZE-2iqnxND<j<`c}q>dH0NV7aq^3x;|v>^|;g2v17-(L^x9yt?YYt>|D5NhCQsMJ}hB6BXwgdyMgH-P?zypCtL(%<yC3utWmiCxfy;TAxV`IBtY-_=SNhV^oa+O{XP%p%9wCg1Rx9uU$iB?M~H8m;tvX2=th2pcRAV*4PP;}JI!Ss3W1bmMbKvH9_(3Fqcg-2@WzZIvh^@p=3|u3B!R|?-K5qzPvLy`=_*<Qx6gY&P6|;?Q=EVX{M9K&_jc925gTZ71l7b&8qE*S5&v|5xMfxns@zH9Q(YmitT9?A-a{bxN#Fe*z!l)0da|DoUi6MLyGFWoljuy<-i)J-KQpg%@g)f$A6r4CSPEhS2osOmA>C=t$AT_6d=hNweXI`JrdfTWNfAgVGFm(FFey*;HFV1vaoh>+$DL5jUSf&uEeV`YJI6bY0bW5&90uUD0}tzqYGB=`$vB%OiX<qId2eM^z{7+G^{#o}fx6@XKwBXy5Mnvra$7gX#l@*5<#8(%KDS+jMj-7mSOd!_vdBwYltR}Bz_$d8j*+KXGKphAv#Z#xOaJiEsQXnWcLp>nFsv-EpZhYYjLB9U;dEqN#rsMZRxfuQmgfjXwWhp6l&S|Zs1(Gq+D?e?<$<^;Zz$}Vm;sjKs}BSRh0IEBiobxRK5?A?o-7XS3`1yO`Xmz0^^cAP1(<?u&37^IWN&N$pCX+v(<9<phBmOU1wAq#Nw0{9qwj1(jICWH`FhhzT+nD05Wh(FlxXWUThE-&X_YPkFh8i-u)o<K)4_$8nEiD4<%i)1%u~ot1sgzq`r5CPRzZnz%9}O`zc)J5zkDh0<J+p<N)1cdp%AS`_5d9ra??EG+k9~W!}c8*?Tb4HbKWFV5L|ruqCtOeHadJl+-(?wu~&jv>v#ngO9fRUhvLK@==_pcUGcXP03ypZqv~YZi{bupxs%7t`J%`%LbwKf7Tr^uxOQG`Bc<dn9||ouL(#GyW{e&1mzklQn<rlyL%(CO6-&Crdo%4td68q$POZib(JKcfJ37X6nOWXdL6D;+Zw{o_>xdiSs+zec^;l+KRB?QJ7>~KY9m|qt)l;O^s=YJ(jUbuz1riNaYgQA>tZG}+rAo!m?J99RQ(44ss2``P4<vGBV`YON1S@+ONnb0(OoC6{`}c|mA?h4_h&Jy)7xNst#L`9H3FJWW$QO8Ik`I;IA`t5+#(3S{eD_*q*^TW>k)x_S+)J<GPqGq{SA|lmUK|0dHURB!J5fk%n;Y@Z+hYixvqxi1MHL+|+f6vZU9Rno&=^*wtDQDLEkI0reGP%<mL1kr)}2CvE0hKMP)R>Ve#9^TC`T{V4Ts=##kKk<9JP23TL#pIQQ+P?59aqwW=^D{C4oGUb#u#hy1&Du{FO6lRKU8N^sd*9-(}Vr)QpjjhOFU$RK0D~x2Dq<Zyc79+@V@a$?Ot2vyU9shCw-=ibVOYs|+~0DJULLH{C-PWmmmFHdC0svE#i3sB}ki+#N2sx{>ox<{b89>*bGu>W)hY+gmfHLa;KFwL?f9QI3F4Gkxlmv08=2WCg*vcjj!Nm;(`wJ7v$2x&mIg&=mKrno_zdzeP$?XlADPzbSaR&3rn`w23;s(YCa79M<!0Fpi4=Jau(oh=kJFx#G>g?JKrYuTEds2EoxTI?&t(^?;Y1gd;quqHi9~WRlFB$+j+68mY|K0(h?&Wb7{mvrvD-mD$#<r@U|ptx8tG(x!4bt9BRGysD<0kIq~n#wkuEv=jZ<{Uo5l6KgYqeaMH#{nQV+OlpkC^b2l?V}_5sMy>IL7uGMyTOg+bd@znPzv>D7R6~FSW%r)(@<1dFLy^sAt9ywIfn=8s>c3=5+4jQ7h0(+aXO3xB1uX6~r3Z|T^`j`l=9F#KkODxcSn$T30Z2_J(6{X5w8w{gp|%ga84ipp1Mqb2NLN&KZU$j+_vZu?W}~6JkyS)=Dldw~o3AV1eE(v}8pb!%U79j4>MaGOl;d)l&y0K+_*8{468b}5zk8mksV`vhkp9Yq3EtQ1bqkEEoqY0f8gu(NVV*4vR5AR-Qz)?$3&E(p#gYaNvLEYvtlqILar$C=sfgJ;XdR3=wvS)+oaIZbwzaMKVd7>lhX=sXe;39(wP7w7&jpE$Wu(V!nF+%F2E#7Seyx;#PWHy<J_n|?S>eY}uu+sTAqfIeo4C6dg1R8eTle#ZVD8LpJ)b!pgEdL(2LBy*ltvbn_g;q#5_pUAxT4`MbLb`pVH%)Sgo+ws>C*QkoSLQbVzg=Z%{uf2BPoqNtC~2k#d88>h%_K@6(W=THbrNaaVT&pm}NFb!_dn?vNP0V_w+<&r%rLq+TH6bW{X>fV3)$Qv&*Ws7e1uv#LV&#xcXMRH~nn2FCkrGf+YSQguNkks}L<VMv4s569|^yP%+bOI;LA!=3RV5r<2>Nu94=5M!dQ^dypI&SF;<y@2K@*fTg_+uD)X==Jc~J@K+AL$Qd@h^BIhHV0sM_A;TYA@%GFnzss1ojNf=EM4DXeY$}`|e)0%Pv|mmJhji!Vy`P)G@k%vRSna$u)6zEO_=#S)^7oQVzARJ33d~-dy#;w^b*cb;Wz?{G@Ho!hlj9zp&pn#E9lG^EAII43vX8DZIF@nc)wNO=Gb&aK&^7;a5|x;NGw=)yUt;d+m*GcCiiyG;{`Y`jnGRg%s|2ztgFGc*3{{@e_{DeU_3Nty(=V-d^AXXW4mp0<0E&W7Q>%%p?&(Xp%{s<bR$$?wI~G_A)H}9mC5rnMUucD`l<J?e4RwucAr2P@D4E+#NKJvS)z1g9Fuo`>zK3Yk`I{(Y7~O}XSc{x`_4_4LkJ4%`_CnZRhkUS9m>`{L0MOTjkAkRrOsYvFI0H9p)8ZQ5dselA>eS|+4qj#{y^NI;XKHeF1q6Cws|Z#%_@5J+FiSZCUAJPod|b6?crGgyu=k#MP-d%kn7k&(EZAWokjE*Rl3_qyVr650Uks98WBMY#p<&wn=-S9^G9g#EV2J-YbY_n(?mfad&dR)=FhUkYgNarbc+<Z8Ex)*)$8(-43*wP#Vii@fY+g=ZKOyZf_09Mw{02=!FZLnJG*x9p2!wvKs!d9ra-kyva$O++3nFL@)z`t*;|*b-1mJr`g~W=y-sE18b$7O8vfBd+BFC=RV*X4qP!}iXzWY_!0r#qA$QiTD_maLiK(7?f<dh48nf>pVjUt~DK7zHgoT9eqcMt8Ue={@8pQ>B3`J~>T%Y-c&9w-3M#we5oB;zAqk2@})^Fg;Eq1$<JeQTx<+XcD&kON<}J;r#rBVc3<SQi0*7#~1UsFP*8y#QAAGPF-(K!c<hHCa0rwCSMp&}=Ep+Ddj@Fi$4ap3$R-G2Xx6iZg1~gNlFl6hAc`lB0kBaFG&X9k~BnH>`P4_*Y5Q+^0F;bNR$akw@xJDAxR*oueOIJTQIlCjh)Xm{1-<Q2yeEz3vp;x8*P(b=<BA^7nd@tIsr`wRHU*6~brtmNb{}wFuJNm8)R&t``@aR?vnDDi%zF+YNPyR(^}&;rg-orE$SsWYW$xxev-qqMDE$8CsaE;~g7wWtnYT2pDhf6F~Y0t)6|)U8Z{T6?T5ogBnW`TbEaP&CtsK<=sVY<Vrlm(x7@hjCZL#$tF5J9h&HXd1n1i;6TR3P{tBxjVLjY=z2O_Gd$E}0RvrKDc9aPksq#TmlRmbw&!AQ&YC;cZXub)F8n=HMENc4<%5k^C7)z5MEOq;0JE(W_?~#fvll-r&r7LD#$BY?J0j7TkIKKkz=<t8k$UEjG>;s&I)85sgV86(s%w)NsR(M98Uk6bqu`(OzBKvP00e8XwO9C?;WRO^&B0?}LvIwQlHGiod&>~)JO5TW;RzNK5<c6!X1RRAjr6pCrm%J1F(jxVyI!p@fSbB6n)*GjiSmndkU0Haxg&Fo(lr!RjIY^HcGV)js4F@P<!~sbP0#rrI0~?CAHeFqporx$CMp-F$y1<)j|-xqwh_KsMN><#c_#qnu*?%^<b4`fOw?-!Uhczat^+t4e*FO#WNs!^1F<xPyEa4JbhD1)pI8P@jT_<YM}MZPg%Bw7rW(eA!nwd8Xg1#h^E?qk8g5$43NVS-6$b!4opGPeX!#y;JFU{=moeTcf9;F8_keq=g^Ihz*<RGB$T2W4dKBkiVrfqlD0+vNxmU6cCkx$s_jmyrC>IyN$_4gupH5nx)@@{@`;L)_ylRc&OLF72c#8wlXE}!X3j7EGfHK8|*?^(u1zo-Tyf&LnI)Thogofy79<s5^=BN4o(*$V=2g5#q^viiW2?P21mC^K$aWlK1lR(P|ZS!J&vvbR`rB+S?v&LUxQ7D2IEAqUe9?goC;?DtqYr|7Q>QC(R-;;TqM>dFSr=9e>=b12%Q5k}Mw>^Ir|H#S43-#>ctKzY2yG{;W)b}qn000pG)tf9pijGmIwwQ}jp0jN9bt68TJB*l?@`$;KTN>N$Wt~PyF|-$br46qBl=9O*$llvVl*pPS`YtIy>S`zJ-X$I?0{NAW!a$$fgzTDDlHqB~<EVn_fW(u{`2><HRDcHfoO++jT>;#y&pv4bf!H7{Op8_+UZJPy8^av@7(3{dLXF&AO4CNzCfw%5Cwr&l?r!c&Qfw+9x2{i>pO`?pV*L>Esc4Qx0pHGmGHFc>5<j$UHB4s+q9x3!L4Jej!DZHq04=5OH1RCHa4Z|FDc`I;_&kdc(+qOSBE4*f=PRi4g=N9iv?!ZVa7>Q;_1;N^#eQ$4l+?j3>ck}HV=U>i6E&2l2ol~x{|4y?g>`GR&3xV2y!8~f3DTOy_GBne+@dQrrXyoF1*j=$WQ;P@`GI@J%RBtrgWX?4)(`IxCZ$sf5coaGPF6Z&xo}qO1W3gnfZ|gN$0_JPOp7J&t4ss*V{bm|VP<3wEn=JyELcO;^f+FCme#`C{}N0ge2XZYx^TImzq~Ml)KsV+2{logQ0}p_R;ztL`tMF0W}-zk2Q;OL)*g=a)pKRMDICl5b(?eiPNYK6KFZcrPL`~xFmee5QT3Gex)m2s%JMHJCkx=%BgnF!f5MS`z=X(1pFt+z==|3se~UJ(fa{uP^q6*B>}fg@NB-TC25N=^OkBN94-ys41-)!02Z2cuJB^Q1QlU<N$9HLQNvB_7Y-zNV5iqvZW8wRuzD}h(S2QHu%Uc;jdQlKVLSQ1GG!le%q({6o{br58`X8hqIi#=ez@h0P<V4aFE1KslMvZM`OUrNel)K}M{+huP<suyQ%`)nBSy#u0K$yUc`+&*8W{V{iOX)XV!!aAuRWCqM%|S3w?LQ1xW%b6Y<Gx2Aptd+*-eVA#NSH58{>EM>$L2(BeE9x=Dz@h;qpd7cW~_vvj1*w14@4qvY^g<Go7EA3F{P0Fs+-M&ARH2eUu(Xn|2`TF^t;YnZm4L4qU95n22Bqb*E>xjck)93u4W(iID*x+4L=!hApERAK$A`it?uc3#+TFJ482WA918x&t344B26}OFe9-(^nCf!IEG3Ekvx+wtM{h#KO6EP9V-QO{-WhrNoYeEKxh812nHjl%_NUUQgfi3E834DYL_O)cqvD+iMc9oSz@ABUfy`l+tzDP9OuRMoz#X+BE8pm*`3a1Qs^MSk?+ssKH}PW5M9GsGSQF6<*x%D;?|$NG-FI(2AHeaWNqSL1pfZ>VyuSaf*dTO(tF`(5qxU&8Q5}8W`rqIDGaOhdPWcBkZxLQ@IVVB|xLm))#8KCH#6Pa(j>s_&nWh%UR!XNNGvmitfwWEgRGQ3+UYdd=%GDv{I5PBRh=F{eRM4JkNY8amPG*1@8EHKl0a0u$;z0)Bh|N!#a_)3mk|=P08dlIcov9?rw<r8#6{V+m08ysaza0c4FqG12uuennMc!j`JW@bPoU1sqBPGKHq{5A|?Y=)JpRx!#e~E#(Odl-hk^Az)RxPN?1=RZv!aWf00!t^*;~eNl;P)d4%rz^(l8+7uv$HJnSc6@Zz=dz8nZksfutJ?Jhh<jyls|J_H9d6UDspIQ8H%($Wr%WvVuzpkTv5ff21(s#3|)o;7-1Jp-3f(t;R)0d**AWY{O=6VQu|4pFST<yKnu=)^-x=kq~~r&pq$^+b$50)iu16|=KBUi9m`C~yqYW0dj%4+%_=r}Rj>-X18MgX=XsgzN$;p7FYsm-Cp@HhUtM&Y4X)~t^>z?|{&`olQv|f%RD-KWoo7D|Afp7yrY!Rvjd=SJ51`;8H9AB1Wq3>-p7r+qA(PD&^NMT)#&sy${61uN?WuXyC9sgonipJEf;*A5%^Wn4C>hAdWNJ<`M7|_VvzNrN`8lK<`k_Jhp*>p9og$W=nhxGgT}0^1wFt0h_mm_vRE3pta(P4D+=e%m%~7jDD+wZg#>)@6rKYKU8=_?D^t2K=CXDRge17@&YsY<gL-*41$#P&f%f2BvkDbv3_SLQW-B~?JG81-Pb=u10B;C0A4%{cjx|9m69jUy``AkxE9JdKv4@!kDl?2WJN4iQYt80qu#anWnf>t-zpd>hAb*`kbYu%CQX%!<~E(KL6yhMnCSoV|16n}v)GVge-6+D}pzl}Ctt6S!*8s>bS!K9m9crTSuaG~m9J&e`#P@$si6ot%9oIwWh)csfYQ>_?c)%qfW2=NX8H=CKBW-Ff6D@uNm*RC>}Q0jBuDEigV{j4>@mzhZ5<K3{&;q5XeP7eu!Xh3MI#I?dU7w1PPTL<<q7cTfnuT^(fRIOfF5_**S8ty8K1)0JTw>onft9G*wjuE&1=8v9sRy;SoiaXyz9q#)IBFY}#D=$U>x}kUSGpRk}4CKyR$;prGQ_AqShaUTt>Uz0#ZvBqRZA)uCM)w-^eMZP*r7&n<4a=J-H6)O~kjzsoPYPGho>=j+VF=-!0<Ye-lehGu3X^z}S6^UnRO+fyR_GE==D%y=t~e*ERQYhSt`s3m%6Ov;_1y(yIxVn3JT2(eJH9Mgce=l#iBBySmSp`Q^1QxX!{rto9oK;x{8>2D7S&H<nSe`#Lz@DgiIJHbg#hiKJY20Twpqvbm(-7IA;0`3WOnMar;sepy78!yaSU09PzT*7#3R#uMZ#`Uz)0w46+=DxsPMGDP>OjSDY8=1Fa&21Ni~M44k6IUvrocv?T0BgOfm?Jui=BVUQpY%e*B+ejzSb?QNyh8W<v;l3qq&E*3GuriC3_N_{>VP;1ef$rNv@OfO0!2n5RI5e1ktSGBo-nJt5(&6b}0csvEYBQKjYjzoWBS5D!j|f!od5k?zp-e^rz5?mL8o*wtdMQQvrF)4SqWg{4jvMw5k!c@-w&c+3kV2mvO8Z+;RoQ;i)(r_*k(3|ibxjdY*#_kj0La#`b=rJ`wS!?%5N&;B1;9h$G9n`h!!Icr{hBNV?i<A4<HU(Xk@x{cz2tvyrw2-G*3QjBdMbPlQ!I^ia2EW~Q)%vBcBXHgUVeDEI4tQu=`E^^<1iy&<YD;%+e3>wuHRu3uue4x?C5eIS#8lnmSHA9yM+YR`8Mwk8DSR7q;oyM(@O_krDpTcbSzZS4B@r<%&aw-fs#4x`UL^dDt*q`%fH;|Z(9B{3BI5Y|^dL9KySY4#tDXKC(k}gVqbPdG~q2t-3x+m%s1f&A~Gs$2V#)yJ<la+~`9k3{><W4%x1`}99ypb3OKs3^+AiT_#<hA_(2m$%`79s$B5jKJ;p2+m?jpLQ7yMn50T$WvEYK;LFD(aG;Aqt4ufd;PFjUpIKzpB=qRUGvxu`hTxlxh|v;82U-iG!HK>ZIT7@j`f!3^)yn8U4WZy>%YPt+0=y-2vgEKyKIq9$Q4{>6q^0r9w>Vw{-%Fpc%Ch^li6Dl>U=u(dnS1o_!&AO6)Ha`RxT*1Ly`_KCj*wfT7qO8WKm54Yu~_Pbz_(AB1R75H@bh!+03xg^#nyp5=P8We%Wi8KHxA!D94T1NH2!YENxlqCGZfLvsec%k#*awGyd}TfAX}f|GCv4f2hBQT%Q&i4J?g=X1kDNv6KdVIC(NJQ-&ou4w@)M^4SS<e_hmH*ts1Ll&w`)}==g8ZsiqW`U~jFu=o}vn*%`+ud^hML3{{Mg(+*oClO^$n7J<g{6k+#``ha0M|dZjYKudlKi3i60eJo>02ghY*M~5elR~SR65Mr<*yY}MeR<%#m02dUtPs=zY`hXhg+b<cWV4e$GYK|U)BR8Xm<l9_ldv9xCoASsI>-65N_F}fIJ;;T8OCo@u2SK>|;ueJu>nxGxkXek{G2i=XkjT<B*mK0+N)HPImn)Zi#bSp_pfs;my3O=w@4ZT9FI0UpWvaHzR<$YA02>0%>!Tvx_nYqcEnPn}jcg4|Jw#PI_&b|2+KIW|KO2w9|Z|i>uTwIBP0bId)$OtMLL7LGQuqUxku%^oF0oLZa94TRQ4KJNsXRx73HLR9CblAPskrrKTkUo9Z0ejV@B}?)j#b10wdm0iJAf7>aDdLT*QUn1H}TOuwRSkRa>Ox)HddYC9NRvv{|oA@Xuty#c(k1%5?jE@>~6J`6gBk(Bl<s#_Xj)huC)aD;v2p*Mh4ZC+>}yMptj!BOPYJCCJh+ztioou*+Z0eg&_mD@U~C_rK4>(T}jHoV5>ZiuKQj8&Nu-P98gDgfaV+_R6g*bYw>rf)<I3%c^-BZ#p)L6iz0vk^q#R*$|S4d`C}`~=dQv)q;#M{dAUq7FBMZJ+&<Aq7A}1UAW~5!et6ET;A0C$I;pQn}jzZW@0UnbT}%*S)R_VyKW1&Jm}S1!xp^=KAmo7h-~r^|oey^Oa-4IY3+OSjx=*8$^RF|Cv;k&=O{5e@26V9nO*qu5^}EU;RB-m^}(>+@4lQ-bOp*bPmTJ>ADX(tOZ~5!PDuQI`$YR(LTQWC<Al!G*tU(`%|4vL3o~6TDJU}xqboS7~Bw1NlY4XS$x;Z90XZgg3tbm1aq4|J7{7Al*Gp<J<p2Cct9C=+RqXjK_~bN_eq^Az;r5`li)=?*Y?Ndy15?bXKX1LnqZHypWIruG?aE0F*$98o<Mhj9$e2&5eCE&d{r@bN7EVSf{zt_o2eDeVOzQ6mR5imi<Oz3J|c0BYVkFp|Bz&gJL}dPd8J~fAYNL9j<^wh7VMH_wUhQWkI-QfZl%nPWMTPg%j1QlXFrUw<^%dRr|Pn#tFtrO*4C(1IA(fp79{A?HM(JQ=Drj-K1?E5i}tp0F~auc=fS+wnBR`h;6ENlsijSz`%Hz=;1syRSvQ2IUNIC^QDzemVt{?a)nx`skYzw|q5OGmtQ6S|q;9yq%5#RYrw#uTf*MUnJ3Awo^@c)X&I6V6K;SYCK5N-YCC<LK<XB$`E6%oa=7N`k-$&`eS=6@_a4luZ*QHKCBxe`-vF-pFX@-8u+Z-gUG-bExAW~5B%LMBicsxMS)y&NEvZ^}b-~Pi_KlG0TzJCFSerd#p&+d3VTg!tmLF_~i%vlJt{;7LoUbm7NJd=7z%M;U1M>r#Cyf7crXRZ-q{yT6ts0<oB6e*SwMPZW?M4aSm`$)#{jjtX+31;2(-w==!Lsr43Us`<6YSffZjJOjkcs~6+;`<{roXas(g;Z(kZIC8g6vi*%FLwFr%y<Lo*3XHRX}882!Iiwt_XyUjLvd{_aReP_2q&OJOo0i@P74~kjN{jh8};6^Sww1OSZN^#Q;sEfi_WBBzGeeck%Bb|04{R27G3FKyR*W?exMi?8`J^_$JQ*t5X?vW&#BQ0epNpGN$?{*ym19wBMtIbX-Y-ZnlRM1x5>_l9fleu5$Yo!DE-ZK6Q6~lz&tq66$AgD9-BV9GoC}(dqOo<-aYiDAhlDuT{w2O`H=SGO`l-Rh}1}v8WK~V)$3qPvceFvTnQ<3@T;7(#2oklCt@-BJ+${7OM`t8nyC3K;1|DntW2wR73*uK8d~l&{Ds6oG~2S0gdS930kxKYLV8y;Bji};n2x`P7>dt`n}hN%17v=)L5|Nq<5<|jKoz*TE_<PXB{WZ!<xk(B3D(-@SPYLnjRR<r-bdf=SNSpr)Dj1>3leKBs;B>N<;n6F&Wn^r*wpgTq8;x_uB@4pIo^a>TNhLX(AsQ*#wMnpxr7|%G55PKj5?j;*0+jNoj-e=byuqLFMX=!QSW85$-CtHSM^{_9BgZn%Ec0f$hz`al@}FdKEdbXjF45yiQwg<o7^@zH<D%N!PEr7BQon`LnJyLmdOsxLyMZpkdLMmu2zvqI5~)kwQ-n*=h9MwRZMSs=dJJXk|7_mp|(I{D}O(4V(x3HHNMLIA0#L=(weC@m=h?WLsWc&pCqlhMfoe0Z*4ejf<AH}Ibo5M9fn?kgEP--M~#*g^tz(m7uBR84)38J#gL93#D|N*81Ns$T58=@4@gk5jv$;*po{UAzeRyE^K$oe%)}?CBrsw4sopsk>EJ2zme~Y^36;8Sl2%=>!}rwt*&bjU1hxJI;@>mE<x`e2zX`({sfZLkSuysM_iM?3`#cA6T}<Jt=aTvbmI0C_DW<H^n$pXT=tCZMMI3UGYB%=?<9|E>uvzsGEtIdz#E-q&n7gb80i+0<yu8}(?%+iaEtj8c$*eg_$)VHPrZsh&U7f&NqATCSe^ns~jVf!aDCizPAAzT+gHof{9}+52h@mwD3$mEa$RsG{X6f{9{RH(mA^Js(iEswj)lZC0#tPIsQ*>0_WI76I=%!LF(ZA6Jc@y=D=Q>wuYBIA+i1%D4N@Lap@KB%2^L>Wh<NSCM7m8saJo|6zj5FLi9hektY<z00HrM8ajT!)x{}uYz)JL2Z@wLUGPlwk{>l)MAwRKauz^M?jTrgE7XP<6xDbFX;6UyVr0<mXs&>9gn+|yFE-lTR+*5)w|ye%y$dU>-K;``*GItxo~ga0@B0+KDj(E~q2pAQCTHk|KY(Fb}HTESeBBrT;z=|_F)7`yS(lB<UIpJhO<dqNN}Q6U9CyT5O76g-0A=7Z)XxujPgXXPZIkcGy7AbD>h0(Du--|ad7I`nP%Yle~jPYr3Ai?!~TF#p%C;`7<F`Fvv(IVww{o<&CEXbT_GLgK?)uOCVV$r`rcxhE}J+{aAZi02LOt8ymd#1|hdhW5}4K3Fi!0Q|P=kH)fqTi<v@C#-+FJ29!y=QXu%*b<u)Vq^svAUBqvqpU`Zv1n4tcEyMScvVA%Q$Tl$s#4v5OERpVV_@f=`A`^iZag-4O;D;!0S9l-U1;W(DM{jKUp6|dPqQn^a&MqA6phpKs!SAvkZ#ljC~HzNEXp0loOiJI;c4@(+_Mo`ou8df6F7sEYEA7_g{nK~Y5_z3BvILGad49|f&WG`W0~Q%UN8z#TtgJUji95WB$-JhL(2>oj}2g_iWsOI@s4Y96wUus)9nTahPZA<zJKBSzr~Jx4I;da<_fz|c=CaG<$)XcAWneWj)j6#qg)i?Aj*wSh>aYd&&0b}0ZP+}LC6Gtc0VL%Ezs;u`lVFnW(8^h7}4&3#W)8UtKqqyUQww`DO?V4#qUYjv-t;6OVuSd!C{oBe?*k23@U_7>7~8cF}b6DF4jcgC)3DS;gR_V5GTr(^hY4bT}eJ$<xH@YH&X8e{l%UcT9n`)*~Vz^?=kV3{dxirm{T&Ug<5)fJhE(=gvb5WKI;QEjM+dyem(cLBF}TzakSId?DL<%2N6q;-oQ0hs<-$dBq0Q7&fs<-u_4{r@fo^Zkb}!1wbcD=u>TVu8=v`9YW&X_`ouEy&d3v2I)?GN;Zud5!gX&AZtYN&ak0DxSwBSBG+~R$bk~le8i`#nNL@RTKaq`)vC1<d9^QBi0FC4G&6u$tG`~(ZVK!{NLS9&>bP@Bua~krD`IHMU91qO@1d{w=v#$j~$h9lW@;ww@h~_6<uT4M|>!y)MjawvOOsFED#;W}vZTxv;tBGpSDbGF0rZ`i=j2g!DzH&zKp9$7hQ)e^4vr-q$qj*0Q)M;fo>RZfO-!z#9ey8reLIVGjk%TyXCn)1y@()ERm!5@+ngU0BomfK9wzns*pmR8Jo^5SOsaVK>&#<N9S9I~m(|KlAD+Q7_0U8%QvLeEH|BjbA6xv7h3pzTZz3yOma_ugGcU}MW<?oyhV7WaIaS--I)?4h`6q;drdhP)yTVg|q*FA~7vrwK1_8O{6i&;>mp;U$dOz^n>U^tV~1i&+><nDch!it(!7fj(?hRQNLPPW9sRT^|~o^7pk{k9Twm}!;Zf{kt5HY~)FKo=@g7DwPR@L{$a57Fc^kXTw$+lzEz;)#`hO+=p6vfT#!4`Pq=Y`468xIAr`1lF~yy>rX^xkfRx>XbycrlbC6>0-TTgdEJI523DQmR?@w5rQ&a^S>U}IPfYHP*j+N19m^mk0AjMbTyXTEmME^3;FhJk~iZIqxPw{&i1I=qX{nIr9yk2h|UOI9Ln*qyQSdi<kY+&C-SrN&IIJ8?e1g1uyLiBd6BUf8*fILrKfcwaL|&kqgABlnJ$pP=K?=sysDaG1INYXk=ut<;Ugt^mf`as&bBlo293zdoUou?$xS`jap=y0M9=`Z^l)`i5x_nwH^BmpXEQc_MGr}I%Jzn{3YO<(NbR+&HCttD2_EPBvh+WLGCkUhcBUG!gBnZFAiIC>$R0CU!LKM5_s|sv$)2u-eA!l@j_gPTr<9z#3gx=Hi6wh{US^|xC$Rb84(3NW+8P+r%Dw6IJj5g6unD~N0Vx2!q~0mhW$V>wq!ggInI9x<&cK^7*kS-@;-|bB+}A|@s%8Fn+0lvEs)oZHH?NzzMP}xDVEt31=;TYtN1VZRXqvDLv;shj2&QFlB!InLXc8=gK=Iq(I=bA)P=>}Jya?~PAHUU57O>`$YwK2Cp;H+t-+tN$_nD{&Eg)SFP2+Ux1h+OJpgL#~*;IYQ!-1TXTGu|EbJ<ou@F|;sX9T6Jg9SFC02un7R8@7qDzzI*-et>;%U3tDq5T{keSbTxXm2Qc){&V+2}4g{?67nC75$!+4g_IrQg3H?2N2Zh)vAOK1#&FY12RwAAbo^ui+A_q*V59rz)>3#$qfGS73bd+{+;#V^8wIoi-~x(VKo9-kphk}2rYynhSEj*!Y5zs6sh)Fa{G43C?|D1LN2tvW5yY=PdZbgc1GwaJn2AWOcW$nxATSKsWddjXJ-ls2E7`liU1JZF<U19b2n4Ol(n9vnR^$GKTHsSb_s&;$YrT(I{<36cGJBJs>Sgr2>>+i6BUqH-Dtc60qS-Xjp)PQae?TnD6Q~8>>I^)Tku8n+bWcw0Lw$La*-Fr>?PXT;#Kea6B+W^U5^2km?=4QJ=RAZUf#q?W&2D)y!39VGDEYu1l^X%H313eX!0ri^WK}J@U3ok89wl4^o?0M79p%|)Jd>ksW^|m&1;1!d3lIsitHU`JX4`NCzzm5UmKmJA;~m8x&-qk85^UX7Z%J18-letK8D7!tbo)dPy_GSxa=9P*aD#9V#E9Z$IIc1aN^J3%0Qx{S9l^G+}|U>9^$hl{i0)kZZn$OMU=t&P%T~&5Sqgjc6cu}k<#{rNVwY)x1B@d*E{+Ue)wMN2wS$$bZF1+E`?<$vPNTWiEKy&w(LG)M&~OJ<oE<12${DgIeh5qym_HFgI_1#_}WvCgKC>;%W41CqnL;0&D7=dxOXywo1^*?iK=?1a`p7O)iaL6_()+DAAq-whZUvDM%+#0zKpohx7~kv#P~oU@2QwoczivWYREDcPqsHz^rGr~WqbLp5c{vgc<^1|We%x<6Ky@)HPR!D4R9HSS4=-T1x)KqJu%ukk>^w3K`8j=gydH2sE?yo)NC%G9TANMvLgp2l+>~wfmPRf3C{97bU0q<me!%sS@*Xv!vgq+gL<)%@i34b{}=pAY#hX9vP5QfytZ2%PhoiiV!Rq<`wXw6$}m04?<?V)i`9x$yoE|7u!WRLmBuYM_v2M@CKO*=uRULFBF*vbubTT!BXo7F=3B^yyLh6KaS%WDw~}*Op+9{zFY%da;wGmg^{*H}{$IZ^XObLza-v3Qwb1d4c#!w&)5H)g7r>iZu3#~CcF4~uexf!(3ju^haV)enpdQ@|x=!bny6t@J76w4x0$;b-#77`{^LM*A#a}0_s~K+0Ep>%_nH}DX1ZCNGlfm#n>i{|SW7GlV*X|9TG_n~dq6`XP5#Ho51L?uUdgVJr3=CW{iTb9RPPXcC_v6WlCiS;fX<b=?!B-0(4u<gtl_&o(EkJHTHNE{uwOLu6^nZi7{-h=9T-|N(Qm^Yfc5L0;gqqfE$#JQ-qjgMTwiQuQ3k8%cZt)pOYV_qht(k~kpg$Xwbv!HXHcu;4dLPSa=ywY$Bc)*yO*B_rXSN!Zp!6~70?-)hUtPZ&up`dU;<N0%P;iP0ADx?K_^F|Pke~SSHC~>YXb$BFiZyP)0-Q)MR(Jsu^lBP<^(MYLEH%U<EN_n}A+hCO07S5yyiV-yd4kdrJ7y)ntqP(?ND87eR-H(x0bGt0VW79+10Vju1BA$<Mdl8eoHcly!Py0(Y6DQjW5d3~tbU1rqak1%aS53z+k)rFM>$msAiZ_h?1jF$xolX*W~DN~sI4Z&odLE05x-2{D>@GvxpkkVJQ1Ifl-wIWH32@aKBY_27l>w6Q&8yaFbz<nWQJBASg&jJeeyuhs%ZJ64Gk`N0u%Ws-v{>gGeeL@pR_rY_RA9lFu6y!9JuY^I6*hY<9azsSLDI=wkCQdS42c3%VRYwAi-j_l`CC!PPbh&TGt;zl+x4pR&V!{xmXN(1Jx8>;B(`@W(hOTqs}FSBixTrQr<d4SKzvThR#q%u{p0%6bk90G~24U3`dw_So*v4$-lm9cS|Q099Yq}347J0y5i(xw2(ZzDTA%HYG(7jTfzmjA-Av=g#W=ARZO4_Of)N`9^8}}x*3+uADVa)fD~ofQ8fAgb2Pk2%%RI#zyCGIWo4^MYc#F~*`GL!z~B^=l&erdd-&ZL_e=6d)wrU{_og8Km@z7s3x}O3XqKmFn)if*7Rhu*iiS~O8$LCcE+rR%<1-WfksLbVeCyrVzl>McHVlNAob!cP5J(KsuC{rwyly3je&`y3nxx&LVUps-z|-#x+-ju_T2;D15kw!cE*&3&tfbu`Ix<bjKL-?}HPM3B^@94iz6C_h)=J0e{8x<Gq;E7x-eH#_e>MTVvJcl~%o+mU^^>T_#h>vxlx1Va$E%{tMqHIwBzT}6x1HjcfGul|TTR;BnBLglS%W$&mj>2|H#H@SYoczo^cmP38UsLbI@46qbssjj>M8!@Nq_k>W}HWyaEqUM!37&?2IyQN(X@ar#E?cglr@av??{XMEnq)|)GXrRHBiM@32&BpDv(uJM8!{uNh>Vbh1^|6TX4736gO^Qr@yWU0uxDU%b~6FmREuhcocF`o%SHVV3hgJG$)6PBX`|-J6r}Tceuur5b(I%F?z2!wULG~I){#dG<-*J;T}Ec<KsRIo<~71|Hagz^|O22nQH=0{=F}5FVeZJ*2#ZAnX+fG8hLM~&yL0Xw2Jlo6QA*5wgORrPzQzhv(4F30C-Zg^b;yQ`Iwes>I-HfLuof<U9Iv@d{$=dE|h19pbgediaJwl7Le8BBF%PkZ(gvyJUwYU?qYJAInmlh?T9h~p#FH86Q)luG+(;AdnasPYJxdc^&oP+uFv+um!McMYqFci{ISDDu>{1g03m$B(3dzK-n?^pwv}Hy3y&*n3rS0vDY!D@!7Fsvq&z84vev2k-znAI{-|?wuk)7-j`TbIS<|Rn3oUa1{1t-MTRSax?Hl+?dEVcEVi0jh+f(9I3IR%VvmdbV$-C3cRVcrF%;s)&QoMP|3<Es)GAd;BR3Yln^4IfMxEsB64@UM?Vk;wdAO2gb<X^vL(tzO&590jKKJhV8P-<eT^fM1A`197Jf)gQy@cVJYV8<g)KkW8xkMY-Wjs-;3Rgd$`G~<6p-wIH|^1*7d#hYG!=gZ>?{N^NmRyq@8yOAV8y6XLRGj7|P!?g<;#<_HpQUp^7EjSf9HS3W3hgVfw8S1$EHbdtAGanmxF+bUgh~oG-!)E&OFqoJ4Je93cqdU}6Q-wt!qvA3FNf}|Fpe!24-+1mw&p))hPP0`^DrMgwZ?gd#i1MMV?t9zzN9E)uOM;Q}%B&=GzW|{Ku^|Yvqnw`9WR1zme`O<t&b>jnTa!ygk(uYxo)~y?ahQY%^>9_)Mhu^7kEG)18E$-GZ@s)f?*}6ASw}rn?724NC>F4@(lG@*QKt{K@p^sa4^swi+zPGdUrjX27%ru)@qMjzh4<{>*47Z%Ovp{IE>4F4k_2DUEV}h<EVO(GxIz!hkZ=&A`Wb@Wo?b}j*coxn?cXJpYj^84PKEj#q@6;O8l!4M_XI&Unvu>$V2b}fZ!GeXSDMXhHO=t<{LNx(ux!;Xt4F4GWlcYAKS0n}$)6t*Pn86IoWr;M6R9%0(2@+>qQ;JA1GM5fRB&lE-2i#z^8VsmyiGUcK~k&g5NIG5WI?<XU@9j7)6WFJh!X_FtsT~0!sD3uftC%%B($?KS>QX|CQ4>;{GV$S?%dL3oc&(e699#{=leEDagzMHo^`|qe0R*EJ_jRXS9G&(w=kBj`r06jtMWKK)1ST8{6}rSa)srik#EI-jab5DP*24$;GYVM@3$+5K)PO3!@0-Z>FZq9KT#n&S)b)fZMPP2pfrmeZSAl@B9uQ!?CiZ&aQR^o87r#jfCDf$Eg0Z)pF-_L>EY(@n<_A$VneCf!R?G&y_;G+Q{nr+t9%OFip31i+&MFE`RzBF`C)w4{&3*IxH=_3u~#4?g*t_@_jkkH5eIo7b(`}|SymPX3A^ZMm9_Lq@_>W1<l`7%lrV-^bIQ8|u7%$IYp#5um2l72=!T|s4x)%rNSCPP`y<>y(j`^X-k+fy>WEC~OyGjv^6LDN&w0WNM>g|dAc!FWuGxARG0qugiK=asA85L)+!sj14wyHq3vQ4nv}I|}1`27(9=M6wy%s_QUVm!gganEzL|BfyMS%PT6L1$T0dx$t^679${pdm|-&!`7V*S5WcM>6JX5KD#OyEcn7e>1MMW~zD&?bghEL42^K_C{omI<o>piZ<Zl)2T>2b{{lVd@Jb!|HQHxkNwi-pKqNBjh^Ks@qI71O9&PxNG4wBTO;vEd{H5lI3XC)xX-jj%41~DffIixg|h_mEAdz<Tfft1TiwHIMv3-ehAKOz#L~@gVWOdBgo$<FfMN5_7&O=gA0AsiL@LKTWnc3er`L3vYi>~1T*-z@OkB~p(6~YNmLIajQ`dGI`*O?NU9@b$CshJVwnajI;P9-4<Q7o9X!wn=(xx%U$3tErdUk4gBwk(nn(ko_r+b~va}NrPBx>(PTG=m-ma^W*#@>N{%^pR!c+va_Pko{vXkUp*?J!vZN2%v83PA_$ao?Rl7>%oIu~<Xzbn`HmO}!?=saYU2+&pz>nLEKrW!-Z1#HPuIZwS4QQ+=8Ft9B0_`oJ!Qp7U$V9}f!Xj1mwDLuZq4AS7)DUzu^q89YqFR?7(lWC}}H{lsSY^4qmx%=4!+jn+3xaFY3hZwl3(m2!7GgP+?f~NwreX`Y<Mk1oet!96+u(9>{e0Uu$b56>>2RK1tX!hUk!gT<G;#y_l(E??(#x5Jf=4`2aWubX*-Euly-1}*&gp?HpvNxf1IneWbm`J_`@T~>AX8sQR73j-f0ei8f<$~+`@-yO4RDk3Wg*;<jP?y<lEcnMMyyT{G<|XmG0*25^ayTUE;i#iz_U!L@pLrY%Oat})q#Ot8j4YjlGCU*`R1u;bvGZX`Iq0#urFE}@wrzOCNO685k(oqynLADo@AL3_vQkH1?{G(I4<U+pIHuCo=(vX#EEBWyzaGsZ;uD)AxkF{Y^B<u?WX~|NZ?x*m1s0RJaU7O=Os4?^g)v4n)&x!elThWJYO;fXD~D2f({HOP(`l6);3TOYY?ql3!^#I-+A+Eb8-9mJWp2H0j*N>YyKeVlx&5Ap4Ss+Y`A~HjH}G^PM3$hHjA`YJc8AmI^$S>P!LH$_VHxl)Sa(un6Fz@w5#Za`^T)yl6i#;)wn#CGTp<LGJ;Ca+8TXHaIqqpSUebS0c7%#<X$IqnwLOHfKvVNAip*mGajHS8b%^|uykT$d!ogI)JFvr7#%C3a+hF|)RSBDyEX>EF%IDPRn0#C%8Wk=-*8uL5M!I^m2Vm{?cK!$LB~iwnJl^(=kaH7lXV~veD@xIwOn8hdnpcfK3?>%9mKZbMK+%KLIScq8(e(zws2^S$mskGRQT}+JhrTIrdgYNLZNT%q2a?SF!+S21bA>QpfKM%3&BtxZ+EpmaHUj@gr>jm3&VyQhNzCF(wORq)H9~64!&^(~4>=+ET%jMY^Z7Hh;wjy{5ftMZ`3D<yMEn|<MXiXv^xaYS&P1vQ?wU4N4QG%a&xMe)<^zHclSTI$VzwzPf!4^R(hvc9ak}ojD2vdJ$4xGkqJ!i<hxB$)3pW*|65-LlFXfIQ9|UzsPC1z0pHDk2>~@uxWy-kt^01lmlP8dj*D*SuZN_N=8NWu-WAS3JvygLR_t&uz*nhZeLAd?X128CdA}Cb~`haOm)&0dgdGukyDn9d=?}LI;O{qDCsNr*F+IwiGaMnq|%(~R(-f0X3nFRWs#SHI}XhLeS#PbSVZFs`j$-K>=kXrIgFc!Ky<wfell^qGzWdHzZ@I=Z^cHc`H5b%t`w?WltKt$dZOaL94%z7Q-R7yoO7*trpDEf=9Bk>;mkiBCB-NOk-gg|uP)1cSBDMN$vlCPD8qMVsB$RzgG=ir%iAd+v~Mf55?6hZ+X-bVFp{JOb5FPe&ke5@pgQ{lVxt>@pkout*wUzUVE26YRJW2Chzj;`tB-nd9_h8Mk!pnfO7xGDYtb#9H6)7(EGA#*25E=E55E_R@ivd2nu6pyANAmHdGx}fUVrHmc1n}AD_=S6fgp<#bj$81T3RmRl!uIoqs1bP5_+Wb5!V|ifCQ$ih&Su35nQSAM4oX$DUFG=kZrW+@A_gTWUzGc@6amAvpOZE@z8Dhr0s5)s3>+fd>yrkZ!5N6!9F;#T9mRQDabyW=CB&5Gc1=CZFxq(X^c>;wez(?-RWtBl(mgs7thf8Y75_RHKQ%U*UlQT)Xn>r+2(i%+Z)z;LV*8FgYqk~OZ3EHHWdu@aSq^Dt}gK4jPWqORoZGcK@YH-5}pl77+X?l0BEf|RcjE;M_fzp#uW!c4OMhr#S)DjXq`W&V>8yx2*{iYI@$^|`*RvD(T<g|~1>YtQGYyisQ{}0tLL6Hvi$xh8MS4K(Ns9`hZ+r56)9ep1xqUEF{k#1W6+lu`<olB5armfh`WUcxhnk*m;TF<-N2ugS~-bXH;dZ{8}=$poLp2-dsbT6O6U$kxj9-wtS$-19w<E<THy$s)}5X#L{vY*zgfBgYzm7n?YGUjA0ZhQ?Z-GI?*>Q7!h7P|@jeF*@c^f9D|E`Tps#FeX=_<|oh<|KDK4@F}yU+%6#6;|)|A=|O%deTs?gRtj%O-%tIK2xQYN};P&5b1^g;d+ElM16$i{eZS7i3jQGX)9xjjNKZ3$7v~Q=ZP4cu`-Qz)b?T_I}H6_u-pPn6VVO{$zyZi8=22?c#8m76N*GN!~Of1_w;kzR3y=du#0ZWHqa`DV;zDvjZ`CGz9f5)xQ!#jn#m7|%(h!)f)4d&dO@s^U}#WJN^IMHS+ObS+(u)nDIjtKPLTKe_xnAQ(VXR8CXewz`Avo4n{cSL>1bJizR~>$LR3^C0;buvHQkd*<y=IRy7lFh>F~aOte~AP2lj{CYT}~!U}XvF{sn=jPuGr$bC;Fm(ajqBmg|FV9bm9&`HGxa8Aam4?7FdMMl$1jc5GVzKnn-fG%mM&mxv!LKPLJbO3}^FA>0=Cg!)G*MC0^g?;-wv`$_&%I|P2TbDP18Rl6uJSzo(i>+%GFdY`;tN_E1;GO&!;Py?~QQgFakG3}Qj`&5^lNI-}ZIgU}T?}O0O2-YWtEcqiForDrxd{S!3sc)iKJ;m#AMAtX3h%ZzArYk5z8!Wy4TF+BU41?Hy3!~Wrwg;{0qBvf}W}a8o<;JxiJ}C}tA3S!MODw|ZOd26`vz#)S1OYOysu&5Lc*H6hIPwOeX_qBqWq$Yl$FW%}irSSa`N*FYKkmywz3@KgP#X$8C}QWTo$YHHcoIc(qvPDd-wi(}i`xIFT+ySh2fc6t!FqXPu=G}59v$IYBCQzJ(JmmOo^;iyM={-cLSp~&aPcMwHTC7atsx-T6jcigz#$`zA?KEimLn_h1W}ae0l-6$N~Zb*SB6L2LBB3>;7@`3-RsnPUeVm?KNv~G?yfNpKNPO|oiRsgqn#l{0ZZhnLVHrzJR}um-^!Ybk5quNeHL0l({JWm8{aOnX(b!R7@V0m1?@~h`;1Z)#Fu~K)#Qk55zn=3GZb+lZ9{WaDRXq!4j8<gYfnfD!dud}LX}52oDXv(2*f;?zFa^^B|9Mt))3um^Mj&BXEXp4>?)<Cht5x9Qny@)!teqByd^KH;L`}dAV1OVXgW#Y+RQc@xDgDWPmvsN6Mf@;r~eTno(q*+CWo_~6USL_r3kp4F(H+ATdG}g16vaole00(3i3+>viT6|FWtM(fOE^ODf!?J>9wO|)D!&i1X2YGFfhK@5}KuTP&ubxG?Y*l|D!L8u>zta#Ku$On$4h}r;P;qJmZ_t)nsM^$oIMc+<<A;_~*HbNQvT>4?_?7-{eo-26cj$QZYSsE-8e$O1@E4aI%oO2@N~R&poaa`?cj^W8V~X_J~`pM$6ow+LjDiMiM|d6qlJq&dJfT&QPWOyG^LB<|-XuRs^=j9vh}4V6Ocot_^tCXr>VG3GJ`Tby=Fxw}JId=xM8a$Sy_ZO8MGLoT_h#-l1F`u2#{T2kn2CB63#wstr%H1Gb|a9u1&dBT4!HQif|EB}Z{zd%@9Q`8p{K7Yt0ztI8JRGPGMwRRz+T1X}ltakWjD$(85pt<n1`!*uuWZr1u}doc8W4A*cV9@j&{_6u@HP<`wk8CgfU%(R-x(Kjlg<2Ouwm1eYh#7Y&n1uKPZVN8~(^|B5|0Xe<KHWv&x{YUM}-o4r)WhZ?xE2O_LD_pSP{8o@O>@cEHaE5kUgd{6G(kGzbmG-Le7}BUY*j|jHwzUUF>jc^eJRdHsKkEz-t>lQI=dnvROQO!Z1`XJ`Bn;ynDEP6RE9<p$QbBgnffol=JvTx-Ry29Ek&;|tW=nfr{?t<VhWkP5rRa!s*Rs5KscL++TXjE2D%y-wj<}L6DkBXq?<?c?T;P6$dX%#jn)D<Nn<$oLfc|o+#zzDZS~Y=23M9j=aDlnfk<0BiM>0>BqN+Y2LoIL_;7&Z^dAnqg(J=WWPRhgV`uR%z?5xQ8jfWwY3b_Jx*o*>Y@@LV){HQtkDiTKRSoy4oaw{Rl^8k_q<+R5Yv@=C;kwY&z>Wg>nX=Nvkq)wRd)xQJ2A9)I{#&n1SLTJS7v+X#@h`eN(DqW@dU3(?SPc^yqWv|}_;TcO0r{8ga-qs14{S=I@zGskUv6PdKdlF@`)?SE)YvuzmflB*ioaOCcx;apg5WjqCKGQ?qiqTU1Er<%DmY=LLnY!CQ7-l|^DMCU8r}msuPqDI)a`v~6zek~FBeu1QugAL-c4w@-1nx>N&HZyw_)sJ2Ksc(u2iZ?Y58#xTdze@!%na5K0_`p(w#+!MI;@Z5O>GP9TSsaGk8pCd)FgkS|78g8#N)Bv1`QAIMT~%##iH=_<jC^MJ7Ofg4*dFGLf+iRA4i{Q2u!uKfd3l?zqBKHfY7Sw@p*fBa}eze+`mZD;|okpbra*qu~0%17(y#dLuGp#Uj^ze(?R;1R0-53mH}f!pYk$fSs>qx4jSGJ78sh|3iVndx9o?Jl<Y0Y)&m?&p!D>BG@>e{@LJ)LPNHG383#HAH-lOa;<Y?b284P2aW0D5k*(T(25x#X_%yz6-PQlXpFVn$CPXOmMefroyE@Ze%d$2>)lzbb1Vt<_a{eAw*%|saXRsK1<ENL#`U=%7MkVO2DN8Whd7brQ#BM@2bydogbxxoL48!)isJ(fKTHbG=@EWN_Zk)Bneu^H>0e(=-^D<(`Z2yzz2KJg=vx%^eM|CP5m0QtmcMWIvEB5*jYF%yHOUx%|=CemcN;VHn`14_qAWai3jf^y{yyswf(-?s0_AR9qol&dOalO(DdNl4(!O>-cg5}w5NN;cfmn-ls&446d_c$patvJFDGC9dDjVgcJB64fwy>yzUtUO+b^vk*rLev+*tyaaEv(Sc4^?m6B-VIFWz^huczK^Nb+W-37-@75~73PPZ##`okSO%))2DlZX=11aKS9@EcOMGeUn^cgHFW!n)3z0y1@e3WZK-m^eb`gnl*YwC?w4Xi8E;OY_yOoICXj%w-Y?wOwx9&BYgL+pgJA$21?di(`UI`A=V1zeSxG<T&PEt%z=(YB<z3lHnDv-j^c%_!!V`-&52RnW8<nE=eA+Pyc+!eoN?YIb$AtD-{%NTAvUBxA6R*aYO$*U|PAM^t&Fd+q~BO)>%Y{|~S8K%#MUsQ35I@1DT&Etr6OzZ0y`aVK5A(_+#^tw0BZ#-gY$b0lMR*~0^2IH4rZfW9QPLevKP4KgDRxCkvntK&Ty7F1(=Y}tfeeZwDVYLh(O-`K*BAQQZCPf#1ZRfzpAYsiUPdRbVc93ZwzzY_@1?0M1deC@!Jqztm*<<kw)tajp^*Fl9TkS1*82D<&G&D?DX1|eN$PZlG2PP^H5_G~)W)>zRhfO5s=k+LIEAN{mG5gX}w9d_sgp#+}bL@HqD2N@+03C2R$Z)qQr8sdw2T0&`XWRGV;Y^rk9MlIDfab&PVk_}*c#o&zZ$kH^e!*zLr^*BtCV!@c*l~X)OI6#B)dv^(Rz#yjQnTGo+fwZs!mMhC=uqjW;h`mGm4DKUB%)-+goHPa$j7^ZMI1s?l5An{WCb60>Wd0IDcSCDzvwYVx@N1rwemdlQcD#Tp%DodS!v}54<&hgqjOEZ?Tr&C^7Fp&I(s$ph8XGrDVI>VoZG#1KIgx@T#<UJ831}=K_HQz64+cF<5BAqO3iPq(GXPGfhBdmR;)k<iU}ju?ksV<C&J7RF$~hseB9d%4=Ap9ge1{$Z^De(^PGuDx?2IimxtP+w~@%P#uf#ZsS-$%R^9sHE+2sRie^|Z&*sWy3p?$E_^f8!Y=)}Ikco>L=9OV$lFH0_lVbA07xHOPGE*R9+#k`A<19*hwGyo{{Qb?Yj1~9MSsUEI1KP6Jx;Ql{X-z4d=mk`khyYpL^koY2^V4<#9r4v2V`G&ZxK=SU0YVN=B-xtUY^{)SKD@rLr3&k7M5FNOP~6;Q1@$&amWs{~Z=cBrLzxs3xb-duPj4&)8dm|P#tp~Y`iRxle>b@(j;qC)TAL&Je;?h-9C`Z~>_fgb!RU$;;JRi|r53=u_5cmtl%FAe<3@q9fqP&AHAUK3t9v`KMSOrAZ})Aon%6U%<2}D60Ext-$495dWJrR}20BP(W=s!oEz)esjM=>MMf!OFLmx)A)Lt&CQ%Zs>Ag2_`$0$y?QQbcGRI<q>ralH!>k!B{tMWhMII!K}LEd9FRAyoB7A7%TXp|%;f}(v`uAx1u357Fk+mk7wisx6^e2$|8#*M|3)oDt?F$8gfTLkaK(>D6$8%xpr%<(YL1-lBT4F0BW^Qg0=vySMddM+ubP#;$fzneMg&~Y=U?Ut=fCq9z=A<I`ejXHWcGnT6<m}A3~q=Bt5bziw#7y?g<QXyq8C7aOJESq$RId=cF75h*5|E%;p@Qn|Gf32BF7zVQ`izOSLe=V}DSM)9Fu_58wmBr*RP+b6PDKiuBXFo?6NCzaNHx_%+4=6fo`gj3qL<C@7*c=Dd>!x8t^F=?Ng8)qn!5)KvY|C-}$~O?G5nn)<f@3YP=zld~RLXw;4yeageHM2Sg6U}Oh2~Tlw<mL~m4Vlv(hlPT(AGWDO>FK_=@bv3?s9;UhEq%rOCG2op;uA85UneC$0(->LYH*trz^Z`-dw71weBnWGPBOJ4TUW&$RAXMA+6*E>fwa?jv&9amiK{Kd^;?3`ILDj#%POALO$b3VQ7hmkk)7F0B|^;m4Ju|giT2qyLM$h8k$+p>6&GoWo-wXEM>N)_iOp~mVR0}BmHd6f_pcdE<sa7TI|3G$q66DuInO)8M$(U*QdeYcR?Sx1ogKHd;$y}L?3GK!|~uM==6h+D3ah~`NjnM)#uYfXikGawCeNys7^yymr3Tks5%~JPH3zQQS3ucL%8R3=#fq77wWkULS5UTCSU1b;>Bx#35N`I!CW}jr$Btn{YGP_ke8uE%OUOKxLZK>1fR0kidvJ&+KQS$c6zR;FnZIecQ=+DXJ`|&xeEvnLd*KB4yEKBA+*5Rd8fMc*>1f>L_LrcUaFdL{ZFH+-Xt5|NTn#M!tzOS1>%B#u@@_BGuW4i0LrB;vpSW!$ElwE*pUcK6YEo+Ae(z>WNQ^|aMC(V{q(~^Cnx<~Goyu>Lyai1Faf3($nJ+|nN~A;a%p&WJ*I&nZ9)h5R95I~7+6!tw20WDcP5kzu2r(n=l#^8p$`k>vvNq$ufi}y762V8R%FYq1OPS>QK(_|OMiGT&Kine7qgwtT0rE^iszkpIB@}a2Gx>{7xwB9Y2%wN!30RXyb^a`Gc?){1<~)zjU(+D3(!0su|NU{aQ>qN2BLew!s>c-)VJ_~7esO(nv>E5`>mMT?}Wa3z@Z}R72U0<PpXkTlg~ndy+uhz7L~4oaM8~CDT2;~I3hCjMj4fOvgl@|v;Xk0xtPIEz7nJ%Ckj7CH*TcdZ4rLj=vBz2Kig}|LhjZ5=%)0F_2IqSH-x=A2<UNP877!^*X^L#9h{fdiF@pr7SrAmZ9X_F4-j3mqpc!DM08=~J+B!Jco@*}T%Cz#l^+UPA*q@B+g&!(Nve+Wzde^D<!jbzcXE&Ips2jZozql@%`=&rl3mv;HV5USbqpR1WZzuU0M4>?k}YX*<-d;NXH_d8NTc+C4<9U@Ospd$;oueAEs5tBuc)*iMa4TYUpX38H><|O_zm6U0mqHl;K@UuB=j&K@@wAZ*Gryd<N3x-2H5#3<i#x_jr<-vLZij$U`IEYS8h3nTBuOnbb+zPmCEWJ0@N6Xz$mN#+$|k`K4k~RG1D+Gxw0-k0iwFw7&OtP`Xy4%yQ~ZKgV~{XDI3&Dxi$$wNWyYV2n{VQPzr|#S1D8hEg4jgg6hHXh@Y3BXvBvI>p1<E5$eO%#a3U|1tkm#C5+&=t0we_J9MR`I!OK;03fOGe50qyNVKFu=iqb|hqgEk>kvJSyCY~sLj*%8Fs0Oz)PB0|2}4Ccv#$r#a@I|Yn@__~%b(~e@qVQ+a2{gy`{C%O6l-Z08xKn(=~_AUGx^n(Sp%YNa@MK=;b8<jKfUbT-yu8i>IsM4vsEe*Uh?k8yz}i|f5hPl{?tzkj|u`!5M#Jf#Mvyt+I%Eu`t{jN&r&MsVwt$^EHnXX>R!yDh`mS4Ij|PW)Wz@~yhsf%@stKl?_O}V1zSpWxwf6`#EU9-pd3@Rq_Vk4X!ihm1E|K9FUi=RVbcAuo)#{q4l4@L+mwSnp~$&Swg-|oDtZoGHv4m#SSBm|HF%f@@V1iIO?g(IX$mD&4gpL>0NPuqGz(!T*LCF9qZRFCOm3SrkDRbh1$FJicSXj9MbiDd#Ca1WV<vSOm$m^#!s4HsOXPKJtR{00odHjOHfo4ldqtWwzhproXp;ZcSCXvCM7&Y~TWu|IEomb!K9&KfX>}9xhN8mDo0`w2??DKw(KwwD7wt?nJ9C}!KkX8e>>bFcpxzv9$$fWNd4R3|KGlU9{DB{y(%$~JN3bc8%dgcg@4yH3d(Y3DJE&sgyeE@?K8EiN_BATa$g>ZMb&7T0y~WDiQKVvpj#{f(wkeIh)RIUN7awRB4)dqYMRqYFJ$sgxXlsGXT7vf@L{&8WZ^q49p^qN`CDmj%n00cSSe~RyHY_3>9X2zwXXeVULLr5)Bu)LXq@FGX^lf7IPcv?{bdMOF_L&@zDpi)Kkao~`9bW)8pddg5<<d6+bY%8)oV`&Xo9d3UcK5t+2yq7a?%LOKxCi`B(ynWP(XcP%#u4wmQ)-w!nE^vf#6S3pPf9xb0_Yd}DnBCU9_Ov|7a!CMjC2#w%3LV8oz#a5UEAKjDO^h?#7=1;CbG^Y-!f5z%GdCJlI8t+pflkE%FtDjVz`%5JF`w2j}#@?t&~WoaGRq9lPHYA`XeHqE=D6KpNqI6?N3P*qcEJGScL4x<H!S&RB#d4h*dDtd8XlJC#`4XsaQA#;QB73RDaPvII$LFZvid`#IsiXPCnSB9nrij?1V3Bn-p-pM?Na^eqZM>A+k>X7X-7bl>EzN$B}xaH=)6K7;csAfBfWVn@~qFRXUo7vdWBxITsnc;6YOC`%<GndPNxG;8hy%;z34D1sqU*X(M)3{&m9GZYj|;N$ObYz^2Q_OeA+`_GH68+wqe@YR%}#^3GPAsj)4IBu{5C+9i&+yxj{40yd1_^AdM)C+0z3!8pjDK@+1MNyCkEw@3XEGN1}pV6h54fDt4r;dxN6G|cRC`%T7_`2lp~fPWLf9|Y}nZ76Bow$So2oJ^o@C1O<IxF{Aublx}?7jKMrg(@i%42xb|LVgGEL?h={N3M2slhA#Ul_U6~?t+&&ztJ(YQ6004llFF`cz$sLMrqY7w@|CdLXU+jH4+Q-pc61>Z1>j?To8ttzlAE0j@^@DMu^!Xg{Ga!_Cv$B#Un_aYd}_b+2VUU5suV-PpzxX(OTW<2DHMYITj@TF)CM#1@ros#`o+)N0VY#!)q|UAz729JX^>yi7f+Wi+YboZ*#^jN>M<4Zfn~s63ZJd6?_VR%=Q8OPC+>b-wv`LVZS+aO>%HH;b`e9xf|ry<@nGsX^t5&JMOe8s`K`TTsNd~k)QYxwj4NuvsmBjtW=9f1WL!QeClOYr=IF9y_vYZsTy&hqPVsUp&#JPpqDkSq7H(~%H|Emkbo|uIfcE8$}dwId2)v`!gx4n%+$fl0L4@L-Xg#f)*J^FIeA=A=sB}0xj<-zhHC~$o<7@X=SwJ1N8*qDqVz8)0GRAH*zMiBvc`D-v;qcQsih%W&@EVbGT+ouPiq9WZI2wVM2CxhU)IsC*=9*e(?VDsXv_mcs7*Hd76j_T;NzFA+t+<AQF_BUJ3XqVKE3OI15M_819o4;mVP=iJ_W4o7rqe%b$TP)W5<(Y44GOA)V#4gZ}o7&u;x+vXdSp_WP)jv!hV)zNmR`7s!VeWM)xK<&1Tl{-a;^k;9Xwy|DFOcdog3ESr%M-#+|Es*4#py*Tx7b@xAL`5gYSwI{}okR%N&ChRhC$M~D_Ujom3iQD*9JYsH=Oybzgqp7PqFIwkk24BS4fgj_jf+Vp;y?p#R&Kt<U?sM2?TK#MG<G-tab>BLP?aXH8*>|Uz$9ow^$v(G;#PwvH_v4kp0atQh#RUOLGP!!u4&c(eePsAaf-exUJV`$lxf|<0XWzxGxo}fXBGQ=X+aV~hu75H(ilT!`$&KCzmRYjd^z`qYpxx1(eo#=`t{WYL7r@$+DdbHo-QJLCM$B{zuJ>z0FAAg5J6H}ggW6{y$L;}IzhqW!9BBczU`M9#OqJ&8qpVn><a2dsbH{Q8zb(oUj(C=XfPt!CDq+Lh)^Nt(>Skr*M6-2ousH-$7?moD`FB+1{bQrnk{ItS0R79I=UsX^ylC13W5cleqYfMmvw$w?v;w3efumZMOXct`t5Ob0uzKN3xC(GBQPvV@5^}<6&qgDpeBIFJS$w8@xYgsD?(Acy%o?fSJZigQ~fEI_5n@&Kbr2A5^LMjY18MI>^23WjxoAVXGBBhk#@mo(cd9|w4Q8TrdaW%K@BCW$RuUwji?xVO?tEq@mZN95@AtRpe&5tKQ1a;w8O+-k?MXue`qf-@XKKgFkGJMF|U3#ECzU-^hI=0@K&Q8&7b{FKy<{Ja|`G|l*50{xJrrDlUcMVPng4^S?j!?5d*{*#l74#7i`Qn<easaYLia_boJ*mVgh!J)dVjFE(_GuSuOafp_-3<cw(3>|-ZfN&&Cd01!t?Gp{&ip7|d`q(UI7B=!VT^*HLpPK{3UG=DZsG7~ljR-eHbe}_^sRNyMnuD9>0K&@!IP#C2F21924t4;COQaDTCnRBLyQyO8(nJ20I;5JG-s;JQHY+gIZXQ=uzOAoRx4uh8dMlJte8qDdJxjxc}fVX6_NizDT~FZ)?UPQ$BE37b7Hhqt9ihtE(6^_&wU!HyXa;cPdjpKzHAcKL`?_CH>0|%qF!=!X71G{Cbr1nI$%AsFZ*n>a>)%6=vbfw-D&Fg>6gm+bOl~`2xP~bG(Id9@*^p);lMX{x&S4^PA9X|ffUIbt*gPkIZem8jsmnA+Qd5E6kE8#;rs_JmY(of)1s=s`R-DlR^TiOYEE3_?jmTKAMWFAxfhlm2g1=-?JA{iveP!)MRc`|pO$0OVpY>Ytky>R_b>hW?nQ?wq{%S<C3!5>p}$6|@sxhF9fD5JCl8hm(Fvz_JSwlqPKe^@RiA5q3})+-OKr(2mc};FQfad)m+gNe{haI`2xYN7@aaq$esMcyxKH+eLEZ#~!tc-PW5#A}6rFsJucLJu*e1nhcX+!_znn%#KQvQheeCdOP5+P-i4mu3MI=f!euC_RX5n*+6%D$-IDg+zEJCd>s#7b&VGi5;Uvo)Lh7yJ&$8Nyp;S~nOf$xRG(&g~PpL7lU_B!$@xquTo?0`f05cU9<51+L)vFb3JftEFZPX>tlj8J0-?(t!?XF<eX=9<h#eZms7h9OsUG;ob2o=L%Dq#h{6f}1ccL?`${wWGwuIsU|oOo?mxIq=sGKntES)!Da$j13Yo8z^$uEDD%Em-YXQi;S6FEQ-j$hqwPD7hD;DlR3hUzpPe2do+`f<u#r1SA~)Lwh&9MtkzDaJek1=)!{{(`ehkRm;!jA1~rjZCUK7fj+8L@{*1WzsAev)9QhKj2DEnSB^AyLGws>~4jaC2vQqW4n(y*P^8rxqbl8>IeD|>_xxI(|$G7+q80aQ-QhTnoY-MuP;+Q{g?JcJ9D(TDQg%OT}b1o;@lC6ICh_=-I`YC{geV!z$m7gb!@c;WSPX!a9CZ=Q^y02J28{WS&a^M&1(}u$2?s<$$yl$7hT<1<V;QN1}R^ZoC^#|2VI?O%g5-FHAZW^f*rhld-Jfh<Thv%uA?74R>xjcI2q^!*d4E!NU1OLqzhu4(l)wqaW-k~MeLImHUms~6eQ+5svPZyLoFAg@*JYszd%XQK_MA@+yLP9tQE~I$tI?4Ki2iFatRw&`<PJqQ*#o%D)$zT>YkhrD@ArVv>J>EJCH6<3CRpg%<W(TpHX7ALyQ@E#o|4p@32-}e#g=)(R(3@te%_rr?Rck^W#wc1akF3Lfia$#AdjnLiikns0;4~1qCS{ZhtZP)Gg-NL?KPK@MDGdM2FC!WVDURH@YCTa)0ESpKgFFKR3tw5lhP+KKu6|>^3-+4>O)VzN+@#v#+O56_+J|kGegm6jLQJm`vZU&y1^`!rS)y9ALJob}h<Gq{!q&qy@;W9e#q{3l5MNaCBq7gO<zU^wNxBY}-kM<#r${vM*q0p*o}9Duv9LWJF8B=TDU1q3?1t+K^U=tc?n9>TM5d3ASUZ;!Io_Z_;7Z<0t*^yKdO54rCY!Pb*e!Zx8gpo$0_!R<Ljk%cPRUV;_$cxjfMF(~^s228olq4YVSh>xq~Fu%ivvtiO85ClHoZl|9)Ca(UTh>YfIk9!3Qe+0m$V^$U}X?KujJw1Q*Z<NnZVYclSJreCg?KZxiAiSv{_V4U%7SV%0X95`g<K}+j_+}ZpaX#2%G3m41&?X%Raimb*>(JxfMXzF|#_V3WmscV^VTwbzd&e!rBsLmmze}s(z4Pjmli_+0nCJ)ejM9eI~M*dV9J{=??(WXm1bNi*+8<Zm?qskht9Ihf_YbQ}uV8Drs5f8x=R#W#L^NW|NDTpnR<J3TfAoX{M166wB3*pIE%=4pXhN6~DD$CS9Dq;Jl<kwat93oe1JuR{3i7$C@U!HHJo$(1il@q()brE`-gO{s2evFO$dCE7HS;9+9sA!PXHPn;0G%S)edFEfbqo;Asw8cE3futfIxpx~oCl((!ekl`yf_du&OZb?5;Lt)x?6-t$$?X86n<xL<1Ih8=SUlJ6%i1}BkghH2`4#kVlP6v*u=NzXp7ozHL>W~K2VIDw_Je0=yJ@r5@V9Rkl4o1j`VgwX^)0~!RDSRJpB@g_QJcEU%xVB=_yu#TK<pUz@(@2<4hqTSG`A-4*lHG)9OC_1=8MePjN&iUVG4qB{A^EonOqn4+YA(X-qG6d~jGL2-)O)s#xwX?w6!xkwF%A?b-I|EdlntrMN1i_$X1-<YlO<;QNb4|J^tltRV%NGhF%T})o>DF@pq;f2yH)@7|UYb!7)=_!JLeRk`(qO$7zxL~%A=RKENvy`*k2)d6`F<f~^hv{QXSm>s{nu3EAo31`IHY3+Vw^eL&RfcOPwBV)I3vxYp09-u;5ItQ!~!v`xX5HX=3N@ntbaEi+)5#7PJe^+InrKR1tlFj(^dXjr%r(y)lYQhbkl`u7@_3E?y+3&-~9sC%%UrGVpmA7jJRmr`Rc;;@-Rn($4Zvi^jaduSI>2DS7Yu(TEno&fY(G5Nfgc5vaeGw1wjY}SCDx!$mb?RJbSb5nXcNizZE4;>aT3}HgRT*lmYhtXfnsMk)}*yNut5H^UQQH^(AQKoRj)2ewx*wE3RS)bO87=ZuR<{;`Xla3X~HDS<X&OI9xPnuf(>|G7>_ZL_hw6LYnK$AhWZ^h}R|dbU}eYLn%cI*OmI81%Ul=bIIGBHS+4GFrhmsd>zmIoB%o$IZX?2Yz{T6tp1qgqO~%=l=B(q*p<mEH-z^#$kUHK@H;-4+;Pmk*lkm-s8t{Mdd1r*O2pI%Z?asQ#HUS<TSM8XKjk4@Ng4k2KLTd2YfDxR`wafTg?GY2@<N)Z;nHcbibQQh%PNRlvU7`tIrB$IX20`RtP4f|cGH6tQKR`kcq@c#>B(!tdk>+)1O)dYfeByjd$uy@%t!ZZ&`c}A#)=73O)y~sw5s%o=);V`IqNbwCJcw7PG+6kY@U^`cJ^ID0`6oNmPHc+Z2_hsz5lOke`b++XZXwIh}DV0rwag`(>LmjTeSalc>W3T+@~4K-mBO%ix`tA2q}w!XH<(hla6Fd&z*FagkBz$hH~VAL1ROn-hy<;)Y;2ApmZ8R8WT^iwQGmN*ZUciC3DSBhbvm;!qQ(gAc@Ymp0xitGS84RTni1JpBhIVo&Hh2?#NBA_YsGAA<UA0M5_aaay{8mZDet8qI@pYod$%c9`RTKJSTvgZ}W-fVdBvqr_K>i4F;X*`o=eHf#X6QU#iB>D^*pZ9zM4~Li&m|o-X#xi-`m1Lu&>TrZ}4F*zQ7CE(+~(@5N+PuOkp9$-JU7A%MP+hU=-0$Y!La2q?{gU_7ME^;lL%H<}xG6)yazRY?2@XjtM6R`X5I^0U-BbjVw_YMPi-CeLe3!5hB~=`>ceCc7E;zj;KSvQbD^JcsfO-IAp&r9j#j;RlsYZWm-!`i#n?Ub_wfVTs#`9x%Q<Gv5OWCWQMVA&+3%s>eN)hX|vq2K=q5w?n{IXQFatFb6ekkAMaR(2<0}_xzqXbT|}z7$skGm~7|1HPEt|L*0e=e&DxHU`9zm;F3;V^?%1l`0O@rJv#;pTB~3p=)-Vm)R-+=ymXzgGqLrrX5XD1ci8jDX#rg1F5^hnd$jk6CJPto8%KOn5Y`BfLq_~W<(p?;NSD_)RcbXiJ5@Rkb|q<lX`JKMWw*@R_nv|x3r7DSQ#O3x$>Hjwh<%t}ilg)Dw3y{5?xOGdJe?F+xPH)sPc$`X^)+)&qQOU=Nn-}%8c}PJa94(w3CZOqE>(d_B0IcP?}5P|NLkShrw<1Sqd)L(%D}+s=q6Ud%mfNYxVT8z6dC8>u}Bl)ysN*?qD53_5SxZOdH1c&=vA{!_5CsJWJZ#Z=U&9vZK=&B(ev4^^N6*9bPk-k>_9-W0~y)7?ug!zcPmQ*7!zY|Q{EqIRK8YlX#XguZ<tx#vtKkNUq7w<W?Sk~-hZU+BXr^>#x<--bl*v4P9AcdJ&05{5+UN@k?Q~9enS~2E*nAhI<J}mru{aJA^DkBWx&1L%lGV~PkUv2IfbXF*mYsIA1{}&Bci88FIWlIBC;5eZ8}`qmczi-q&%J-<%3mGvvU7WpEr-RjCEYeAsWsemb^yz^f_ih1^nw(qg@kvW>xdh{)4t!9}HktSfgB~`#i_di@CYbsg2Z<fNt0TwYA~Yz~tbVC^uTkf&5q$N|gg~wB=~WEiKCv?=Xn4y1)xnZvhkt^v5+>b`Xma?!!~ymvz_RcY&{p;28<5Mr#;o6Zz}EbM8=|8!k5$3>4U7&f=z(DLATg76`+XEurigwp?p5yJg$+1xD&kkBF${jf=q(BFsA~7w_AeXNH_DcRI-SQAE|eD6h_#4qPOIfy!<rY{=AbSNJZ;?qWyPSK3@?d~}xSEp|y`J`2xVt+De@_7Q>wL-Ipa^isFOFDOf4_Bp6zI3NT-i5p$zc_<%#g}=P$>=!sxp;3saJ>}^(E+>XWI|a`}T&%}k=KXAP0|hs~hs*|phxMNe53@JdfoqQuSIl!I8;tm8Fa4+?m#Q@qs|JUB2QR&k18We<a<gv{=6J3LRLp?c@)HST!9n{eu>+~67$+oV|7#S8*z6wRfCf%VL*0y*X;F=aSF2FY0HM5_kkZoG)45DAwO<aUiF>Gq7bGR9LZfW(uhsb(@+`(^yO?xF<9FS<M`MxU<<bi4geJ#kapucp25}TnkvxQrCJyv$Zc&9oHAci48DzTL#*Yeaj&d+K{EK27#bUBHb7RIb8Gd{G&Uv{xXOFgAH8d9T*xJkuLbe^G@MxK826X;pxh@+!>w_(sMulsY79NE-Xh?5nwVBX}oa4{@#kJ~e>(0$d-w^fMki<*uI3MIDdqhX%^I&9!I{JI)VsdXCxaM7=#akBtMkhgu2N3lpK0fAdB<4>YMdEifg{AeuJ`|*pK}`}qXr-w_D`}k#cJDb|lR>=nK>#YXH9Hz^6A!sy0$v~|*XQ?uau-ZrjnS-3^OJGm+<o*N1}|%CJ_Rl{`UoWQong=Lc}mwy)3M<sOQ9e+OUNw;Ln&})KF3GVIfsfZlyDj1!RMCNkCehm(PO++e(?#QUFZ&H^V+f8tS-Y`g@3hM>cSCMe9bI_MgZU>TS!AS@nV?VPvDZL3n~~@;OHHdu#TGgNqmBB|6oNG*5m2MxnyAm7*Vq@XL9@Ijhv-6MXaUtf7q8>E$o(NFmCP`!r2iSBGi8kqOdb)-Xcg%oEfCT;$r_0ra4#7FwSKdBB7Z5jQ~i$4o5|Mkz=e+)5qe)F|P$R-NKi^;7kGv9I5^82eNsiS|hJEpsPo*%!Z4Uqw{#=n8wnlVdxw@kq*9<dYMB<f6FK%Ia?3?m?P7!75}gvhhc=UW3su69g#)x1h~)RV1N8+ih5SGPu$T8l36<JScf!mPM|cd2TK`T@i*NG-v_Iv-1YM}qHzDQM1gb!PIgBSk$?q9(FUZ;KNvYhRiTR_{LssSRW54SEtDGMkADk7P$qWfv{Kq3#G^H?)Q>CQq($BF%x0UM>vH(%DKHL#e>zD+D*!0wDu~)n#eMP$LHH$-{*7>0-w}|~3|FQeL^Vik463%0F9el+PXm_Id_8Qwomxr50`p7s=j|y0vfOr7h32tN*TT_d+0F1Uvd_%X%m&omQ@*XKPI80wF~1OZGp&<CTtg`;Ws5v{;(gLmLPA{$P7gRQ*${u1R9lync1!5q6Y#&b8+oFlYJ7%@kDJx=t&Me`(5`C%^mj%PFJ{A?MG_qf<F?IIvYkw{miWD7Wg6wD2$1m%EMggsW?ngS$~B~SCy=DD&;_P;qBMFHh+-IrfB)`J9BR|efw=VgV+|$N_q+(bb57|oqmQj2if@zR+GYdUSms*Q`Z!rlkT^)5{U&=D&ZSG7h^#DJQGw?AOfvaTKTcGDPKD0s2rS`EWiSM~`qb`7YQ@YjWe&m>=zU{wZ*a>;%B&4?gCjeqK0#`Q5mjPST6?r-xsk*da*hmYi5h5{=kn|*_<{h6|5P2mvyG_&$WkLz293cke6ixN;C}S;;FZm5DxB_o_T8X{iK-KPxf0@Lg&6;+_vUq5T7;2MO((UqJ_5SS$O*&Ff$U}^kGQN=vz9PU=TR^lA+xskUeW~qbcKw%Fo!g`C;)9aC8C;f(~B+nzUej$PA`p<O4Mfd_auK0m`XHrt_wOLL{}p$SWN67g@^=**eky=3%m!z;NYl1ki|!$#$*LDoa}^&+*?=(O#beKrkgJBR2z@sUYAyGk~>@}E;>d!r+;SBhzHmA_Gmds3L*7Keok1Mn)|On)AL&u&E)VFpSud_oh^Mc#&p0zDKPI+)u@H)gV{+VK!%Qc4-4&<2a6sm6%Zg3CIy@%bKMJsUR|I*Ee3p2uAj(u-<_Ev$A{FTWY7X3x?QHfutmN`xgA$p56#nMDr$L0?67XL-qF`<?SSTZN4b6!GZ-A1NKR&Y)R=X{zW62d#p8Dd#h$%Ls=aNdQPd~W-KtbW^HmT*MO;us9Cn;ha2(G`TQ<cZ%?Bx+b5-^E140C7`Iu;q&WPrrw|~s4J|?9=vGfHs?tb|k?#g>n+_bE%Ypda9w@*b1-;j4nIDA|s=Y5)+pP|h)zgOW#6~6%Y7k#8|0aErimZ|2Y8>2i2j~y+sp4zCs<F!3rf+jB|R${$jIi2yysU*q_LPU~@G<3DkG@jWKaK+<EK05qtEv>Ja?*6I_z=}dgz>N`evB6q3a`fl&kL{uo?+^Nxsxyt5hjcPy7<Lo<N}WYbjKGH^r;ohQ&xEeUJRs_X1K^>Td|;u)1vA6`{4+<GmHC6M&WEc>v&19N+>{C#<v9^mR<SB(<E_2@$Pr>sl8hv^QiG*&mO@`;Er~PYX@0FVi0|@H-&TKgVOvqm)U(jl0_;Kj#-E^LZr%Dj<XP1g9>pakK5?RmrS{_t&~6IB8E2KN5{GICqv8VJxYzB#%FJwJxW$@;Fflb|l2Pif&%pRa!4Y+@G4R85Ds!8@HSYSmg_gVkM*y6icl;d_6Yc+`YOp48LMxCI)Y*zW*NaqLR2b}s<Em_nhBdy`Zo3C&;`Ol6w0kWiXCl+Y4!Ey`8~(Hsmo<oxBPkiFx?nx0)@24``m5nRQ7WTE0hlY>Yw+R$^Uolm7C3&`gWinDuL{!ouv~)UW-x+;96>q}1*zKx+^8Y-NgV4r+(KNa84|gzJ*Ro^Hy)vwvY5g9C@hWaJRkV3e{JC%u5=vGR&^+{^r7W@3CM2m`EO&YslQiL?nin?5h0M%gwN^9_fD9Gp<&anLFxNw`q?|=jy6+&d&ziCEPRufIV!nWF(A$+$w(_hWQaK~I$WA-(!_iP9?>^`(FT%=?(ik5DFt(FtZDDry&6%BQ2X8I(Izlu5r{4D3w*N+XH@C4^<lF<Ct!TfZIv>J7#hv`=s6M6?)5R+`mYuTZphcA)71nv(79BgI*}~V2E~!P%CuJP12o`5Rl};M?3rD)Wd28s%(BeQMtnXyXM*1BVdbiKV}+RksH2|w*~$-N6i~>+PFb$W;i|IVWac<&q1@O@N=K`NSGw8A)6{rKUy{jy{ZTLoRTxo2pNuKCoUMTusYRBrK6GhQF-BJf{Mul%*gg|-?Q~go@;TF_3>V<-q&F0_(vEc`e`Oi?`bnrp;|DPZy5kn2f7&}zkZKi3&l2KI8()r`4RBlj6og7o;R_z|L2A_Ht}on#&oDpn&!eGw=K`~QLMeS<xXVDcSyU#58gOC5^k%k8O>@3Q9G`0A+(hl=Z4iU8rFVKqO5a@^Q)d^6Me%AgLMBiuMei|+DEFoY)Ob@}tsK(+6Hk0Ehe5toOBd6_^!s6n2o&D4fCI1Z@HbZF%LwChN_7$y$0trYrt2pdum0(Jugk)}4>r$2l2-ZS>XsG`=}v~S4ev~B3_&#T{w#D*ZC6I#bxxX<8~qjTBjF+W{bo4kzy027-zdso+F^Q?dxD<wAz@c-Bcg>3VPC1E&dUwAVt!7$TOM{_HXY7AZm~du{BYkGq5`@8IP3eJq%)gXD_7sTM^y8<V{d9E&S)$e_g_jwAlBOLp5_L^6$@VU?wdj8I&R^kAEkG_8Ljk+oQY0%b5tgbd`8(*2#`sVNbgdd=52`*|Je#-@J#`&P&RS(Q@mfYJ&x|F>s8hv%BqYqr)|va!lEZ*cG0-|ccrp?Xs{d~X1C|+;HkmXeLtxSjlZ;qwgVn#`?c9H?~{3ij>e!WfQ24{yTo7D{YU%t&4&enN{(7%r7yABeL9N3>PA|U2c<>zGV$eI%hyr7BGDTCLlC5Ma#>~htmv>k>micdK%WZT!z?Ncv_an{?Qb)}W;O4@g~22PKEc&sD|%bY&M4X}S^PhpBYU8CA77Xkpx#*(rXJI+3<}16d9gQPvapb*i>>Eh{BbwQwXy!{3O@y59-THvZ-J=AR5Ub&8#EDQPrZu*^b%4Fdkv(kWymt8ow_@bi~Z+y^z-Jm-y4EV*>keOd&`X@S6`Zm?$BKq)YWHNBt>fBc>a(t#0?UQE0?0uyYf(zvTJhA6LMQU391|_2|dAqXvWdHs;wapLqsh1`b|9G!_K!Ge56$BY_@>~8hTWx7Nx%X>6Dp}C|zOCDIu(|ozTz|?GsensnA#4cr|MUZaq(9v2r?-8!)^LJJ8n~#uK*Gm+BtbhKg98W4ye2<=fNKyhQGGK1#`?dhT85x8o5`vi^Y0e_5jszaJLtIKmeL%x|5rK6~Voy<mR<3s-SprC1!a8^{07wJ8}R0<6^nuHbhB{I($1@+I1vd2AXmAsT<hxJspi2BP?^XF6AFG6p2~QQ|O;n%AA?5*`y5U^mV$qU)L;pxlI{VzM1SAk1^beD`@^=bH``v+*rj)~!cU61`;YJ&_I7^5{l{(_;pcwrwJ(A`lN%d>y9OI5ji}lP2rnaW8LrWGQrpUS^~Cv*z`XIc;nzLEvGAh22O<rC;Z-cbqXACX;7?P%UH37Y5z2i@g~~c&*P+Y#nYx2Sun_Z`98o*u(u@0N4G8$~rYK?MSikeh$7w65GCuJDT1iPKkzG4Zj(x0%AZVrxV^;ZaYn!Ez}Yk7m~mX?yV0|x#42Yk*zor)+ZPP4ui|qER-jAn?%(-d$*dED^ce+B9U3(GL+%#>p>z2JDV|+VAM6d&^vUszy}w(e*3rpN|Pe<AEYAPC<UNf9lcw!2q9_SQV%lIXaz-#`fINjEQE+ppd-`^48i=Sn?qORgGtL>4@OQ^u?r@nRFwqOy(pas*0raIHK$2@XQb$}Y6A#TFNe#fa<~D6DDeuQM4!N(YX|d@g0_*T@x@{I9IZg!IO0f*NPvg0WLYlIa1CQoyu>Te^ic9AmPzECSc`{<{)4Sz2q8BLAwwoz$nzMVncZ8UOOKK-8Bk0zAkiV9^}YX5a4$b-8Qt!u9~y;1S&ze3<T_A6C?|QT=GF3SZ-CpNbB0G#&yEGQpNncJF2Z*^96515JImk%1-C^a3)d5Dl~1Zh2w>*WQ)_)uitiZs)Re4k`nfUySr2P!^4cBATXSAp{=tFVFOJnCPUFS=Zfw!VHqSo3Meem;nvTCC`sv%&g&$?Yxf23t8?+ZUQ;g+HbCuG2DV6NG{U7AkE8a{X%|~OjAz;7H>|byZX%23rk&-$w5rNDfGlNiHT5Ud|YV{9I6xnM!Y4-_M8Q3xnwdA!5SEP+Xz^(xznR7bPj`D`FjeF*_<Y3qu;TFHBG3B4uIoAO;_Nt=SkL5;G`yqVWRxE_@vC?6Y^cOKCixMpBMAa3yv-O6f8GlF$_x?s%s)ijC%2Xtew3#Gt0_xS_>}taSNyNNf6B@^VizfUfUAL6oN>KLz?W%K64jG$aPbR{-nW7+QxK|*QN1I34?1BhBth1=x&3#)(sMo-RVlv9`>Q15@ADYjP{_IB5J<PN}GAB~Cn)u>8w&Qb;Y_7CynLeh9T*0oZ;Rp$VzS{kKAK)C!b=_TNTd~aNTer{KBI8U`8wLKYHUaKzSfa&!>`EIt@N$FBbi?DAdF<W)e*!3%-B|fT%?ayWRLgj*WdSW=$|BKL!v&+XI90GT5D12JC7Ny1S=MYJ+7?Yke{6j0Vjp`?Qf1wLIB0;a1+XR+CXsE11c0ss-P+=Apo;{&W?pFz01OF~6*@`xU3hYo{L93WN@eaUc#jXd1o?$P^Mm=kdj`inhLFHAg!z)b=jL_ARm-hry`c2<(|o5I@+qo>u#xEV7DW9KQ%}brsq^GVvXQ^#bOL_}wJ2J)tiF)jF*iIV|L|iABe@%Bx&<klW`Ji2oo~vp6-o}~rO3_~hRzCG=Vb&E?9geU)rWd&|I3w7CKUYL8VKw&%yH|ggWc%wG)X{*#NObFVNq3WYR6o;{8G*<hsR!-zyxE!a-?$C$IxZSkA=W=JcUIY8&iwj?aAMKzv2n5bn(;h<*OX_gE_W93p;L9+{eM%<2id*DmdE%N1Ra6i4%)!24<D=oLkI<&vf(zLvX<AR`$3v{3dMh|4~e}H@UIN@APn<?Ja_sOvIpLT&ViLrk=!{)0}>R&^>a4NrRqpX{aIl9Z)65M)=EX9Q6!^EIf;E=%USV6+L>+tZFGMMgb$PKzqH1JhFB0(1&a`XUsPK>!;OKTu$1~pRoNIwsDwy`7aG|_*qw876M&M4vS4mAP3?@2h#1rH~!R`r%wUIDkD5X)A~s|)D?ltY*4@eJD}7W3Nj!%0l-+y75%$8RO+2v=!Xq)yT54}fM2t@qTWukQVXTwi_#6I{Jg}ySmF$f{lqgW)h*aa={R2HMg&H-i$W5pwa?{!RRGg~7_k%t*uO*vlmwV8>1ZQTd3@$wT*5LrVTb_ep}QqoBD^o{3d-vFw>@Pjkwk2?F#YR{-p4nr*+>zQ&a3d-<M)gSvlNY{UjYCo`s6_KmU4z*?P{%PPmDhx(cGJc!M!k?F%jCsQ*|%nE*EWOmU$A;8gfBVqi(2r&OJs+g5*QkVDoG$Dc@QIRB&SlAdboI70Vxd=NN9o+8%<T5vO)kHoK~G4huc*BG^fu8LQ3|n8Rw+q8xb;OW_qX{md#s_B`~pB`z&9rNru=94nKQ7D@QlBlWxO8Y9cdlnqz8hWkNz)v=zP5uuA!cHvCA*Mip^&(iHCTC7G2&H?AfWP6-tC6!pMcn0S2e-DG`a5~MHs{eVQXLGxP&_@DBP*7rPN8B>UOItXSy8;4E7@zn6v5bnCI=w&8gL*FHma6ZjGM$zg?stLrONij#L*WL@j~!kdND+=BcEw*IeD|Gt##n|m#*FeLeiInFBlYkZE<w$G*5V|K&2A3SOajZn`xFQh2(S9ImuGoYJ>m{?b3qN2_<5RHjV`eSW*Z6ahzlov$O!MFFmtO>%W@aR9nXp`4nHWjoC~I@p$g>2ziK>_KQ+#R6YD;kwczy#JZFqA4PJkp^M($<NHfN4yoS`K2Dwc)7f4$>2PnfGsRns<m0vgNnE*e$+nW;x3-EnHdwoba;<svI6N^vdL)tp;B);<iU=TV1>hN)c{cB!Ny~KCYs?yNTfQI&&88>NREh{GTkc*lR0Kmf+xs)y}-%FvzsK99r2!9!@8%s5Ze52(=A|;)qdw)68BNZj;V#^L+qF%JdA$A}A-Q+b#YD*WMTvWBO^B??|HrfFoo`xYO+uNyVU<QPF7Jf$510mDV5O~GZC<t&g3VV<<9k&_Yjn47SQkX10a^GSy$j%)NJ^@H)py^{yrNFLutZQk?G(E!7cL)n(E!WLL+<I!YGr;&^3k7s2Gi7jYcH7PnI?ztX1BVVWS%;397#T%n4uX}SXp<Y7KNPMIt2;J|5kjIu3j>m#FNPJZZrIELFD)7WBdm1a8l#Tl%W=HdqrGsgaxcbKD*K)27epW_ZwY5yW%jIM9Zm^9AEb-<RaU?+IGr9qt*BgEga-&rwWF9OUV(7$>^Vp9+v?I0<J7@-`_dSCYFSA@sx?jsWjXbLlF4y9xEDt{M7tL+qx~ZFCWwq-@QlE%Y~i7gfFX!%UAllP6=rC!(*n-j^tY$8EDtXDSXf_M+Rd>k_u4E<{o*B$^6j~w@8FZo^scc=7{fiWog^ltFl6VmzOxOyb9^QjMpjQ(XW4AR^k*Y)h9i|qx?<&L@1h6y4dfv;Q$)W7$TuTyYlR9#nbjpv?);J=mNlMf!0ua&ilb)n_iH@G)oSnz`yd}v0#p|+)BtL;ej<~uXnj-u<$*R767!Q~BJ!1BXyqQzpdNjYFp&7J_|V0vixBKckuIWW(=X`;`FD+`)xJS4`%VMZspluvgYb*h_I{ixs)|W{hB>Fbxo|f!5l#ZpbN7YP4_5SX1=*pDmAP}`a{{Xjw$K6*c)dHSuN>g2WVyO4B+HpbNiS+qaY0eK(u<_(xOPOZ<!A{`(lMV#`cYi5X+h94)VgxweU_Oaj!^QRAV-XFba6h8StJt$U$3r{V5eE96jJk;g|A^<g=-};6vIj1AwuWX+ps8>=*sroC+`hJQukK2DZz#FNiZDI*jTC-UC3${1a2{MNNDBDk?#FnragDo^YSm+Ho2Cyc4bi=a+CRFtZtBA1LiHqPpZdo*)9AD9;K@k#Flq^a{}sX-$HA_@`Pv^MMBS7h<r*92-#CYxl}-X+sBy!ZATmdty_6!mI=(F5|V;s?jwkt5%^u-GfoJUD@TMsU19Z;{qR=%eBfY!^@6)B2ij*}GUPnJq5k3S#sOfsVcPYOGgQzihQQB<m?52ldgV6HumkAi{OLz|Oni_aPyTr1#nFM+AYJp!B0BvjS8>y9InQnI01^ic!lLC#+m6bdS)iA95Cl(ZhG)o%@BB3HOKs#?PpRUxXkzygZHS&DpX8Bxk#d;Mc*!CK<{Lq$QCvtZy)wR_hi2gE9C-V7B%Ghm)tpPw<KG}O&0-v!*E<%w54U5TQ~j$YdJ(r95^A0?WIJ&&iAMUZVulU=qP$89i$(rW74Ta>3$0Rk<8^jIj&MNRbrj$q0b--f7MU3-lQKQDK~HHwZz<6+ZQ%{g7p;9k4|oJ!a?3u^dI=iLCc8I<#s4C7{075t$WT^Yj`+9a+P8p>KCwzdi<BNq71ew_v-i0HjFK=j^H_cMb^Sl59)>0GcTA+PQ+-1Yv)D<Bo-$beCGsL+?)M=zG{$oc74chFzVTJ$XT6$(@LLm@VmiWy#J$&?D6J05VgnY=M_1K68W5X`Xfq7R>V8Y@Us?#dK*TUZ=5XU8(Ubu_6Y0T~1}-VX1_HCb_kxc+To(as%mdw;)`@3;kKdb4F)0VFiXfqJxYOh7-G|>h{KVBdTs=;2FitU?(!V&XJMMYqPxD0tB4yrYnH5<FkVTJ#L5BH<mB63K`me7+J<qg;kE0<+@pw;(z^V`3NOX%pwq|hEpi_e{QvFIju0fvGI%#KCzmO8CVTt+<Em>^D(&gC!&xuL*aDZ>^Fqs1|i&3sYtPZ*A&2O%bFfOI)yQBZItwlX#M2C3Oek`<Tja;(6JZv5h5n{l?7`ah3GI|336Ft!H(ZV!nYM_R#_|96M4p?g5<Eq&OoSRv<SPvOs6G47|ZLTI1FycU}{XGxcO70t>eX-QAkS^^<<JpkI!pHxg0Aclg9l2<aW(l!3?eNLy6-?y)*YyR1VxeXCOgeXi$U3IMBZ;;Nw*aUyV)^XJG;y!(d@@~6O0{*(R3m}LrFaFzuX$;i)D$feiZNqb=$@G>j}H-M!)$#zr@E_MkhDIi?s1aqum?4R`@|t&)6+<6YL8TZ#xG6fnyn}s)^Dn$N-UVAGtWIjiClLae^OOJLn$kgH?%i%0t`RX3*aV$!K-VJo-%oj%sHm|E1QRINtNz=BS*?RgAFHnjF^DIGN$T9#3Kbd;X$yryF>hDUsIVQOCR*ny;5Zzovf4c_MQFomWJCh@ugT|jQ2Pe5^W<*+ZtSC4Hp}^oi0zfzRUPBd7azK!kH#n$iN7IG!^(34y>}h#WG8kNz$7pPHBaU(yFi(2gthzN}iIixTUymRuAhJX%oLYyVd?eoHg<?#(UPvNpT2Yoe*{BdJ7OlsbppMZ-B4KUmK82_vUw{3SX{*yvTdWivYUIz8qE)IY{I8y^9`3`?n_B2jG(`Ost+M0bg7*Q#8sL_G-*!_>+F25BO9lgd-#@Oo3L-8wolZq@X6CCK80)KGpCs1EtpY$`9fli$H2n06NQB$qfrDbdlJ3O|7Y_ff{u~n-RG~L4JiNr#QBL7u(A{AdzzdxWvT9WIulLj_Iw)YkMq+WM^A0ygW$1IQdER@5UhUy1Rv7K}Sy&qAP*F-JtP~*<g{9_Ch;>MMI58rwPd)cul|+&XF|I^7e&@R}w+L4&D?r-Wk9)dvrxIru{!r{*Qa9M5oK5-O9opO?MUpyM_tS0W%fB?EolrO)R!av1wtbj8yAP#ZvQW$*W$TTzg1ZTk6qM;hf3g-7&&SxO7235g6G2EV1^xHBo~YC4BT9%WKXFPyk87#~@KXksb(iE>)XM|C>+D>0SRQ|8YQQR?xXJqD;q79p$A4p0VzIV&+UY6j_&>0zHo|^vrW>4c&xn<nHq?mYKu==fOQjPd?O_k3WuY)H2f1l4*tSosOjf=UkBfmCEedx9%fXux$T?*3etS@51)eJ?8ffRxCGCgYG<Z<(>)P?Ce@bSKv;VYMt(zN(fz?px{shN`_%NcUq%nk_D@Xt$&Jb>*(9EQ@O0#yYIhA>Tg}SNRH<*cG26ssbD6Lzt2G1mrV5kC$&__$%b2w@hc*A%%eNdOL)?DlpM11gYg~stzYxR9CCH|+YBHI+sS@j*qx!N4vdb7oovCP@1g2{(ilCS#I!f~JjDn5h}jC?RSTvPfA-Xdjc!_E-M2GL*rxXq?SVr7#i<q_#hwIL7WinsYJkIfmrZUyQWD?NmxtMBjC9I`r00MhE}KG)L2MGq!R`mVkLHf?VCTTg06#R#SbV}K_NEm*6?EYIdvU`7$?o5r2Y&sTmq0t*oQ)QQhEW@(DQ}HVs-f<{d}5TJK|?LzYGytcx3<rqg;~o^jVa-Bo0pOI@}E%h=xueryxr;I@N;tRFnvi6uRI6F_J(2|F=@tMMTAeY;r%o`r4HE-4aSyI%FFVq9<fe&-F`NQYZ~$;-oqeVjN7egQBZE}ro9;)|3bh@5U54Z!mk6+U#NT(r{5?yxvwsif7fNLk_54eW88_=2277lDPXCUoqbpWmzX8Sm7=057Y79{0_O*qP`v9OKAI1@aULW~rk$qblnjN@?uoC31bvqn<F+JE*i-M4=Ul@0frlSA13I2kon=q|?v7MS$pKm6`B17<M_^fqBpkXph%BC&58Q4DML522B)awcUzA(?_ZB1$vfH{HT2lCj<bLZ!!q0@1dXN}@bKsnkJl7i}BWmzy5q{wHHGekBg-x(~qi*pL*>Ih)1Gx62>{mD#(7QBi6zNnY>gNTov{l;z?SM-df@x@Zx*)B|1b?fn9YIrrV-(KlvP3M8P3wi|hmc|^Oc<7cMGWn8NY)t&-WE?3Fli6DGH2RIox1C5J)z;&GbZ!#(RprDrXSZ&5Zuf_QRuOW?t1&TE@)a<+$+Qxyb3Pp$v~z<*deaDUG`qVdMY!Dg)}e!UgMm!LoU(}WRa<i0*bE^<JhcGpSYI5<pGMK{Dav^8ve3{POK={N_B6Bjs*e0x#(^nm>8!gF1Fw67F#oNkcT1g$|i$rO|*T~-N*lE^LK(4c&(3Ad!k=L8m50^>Cw_|b~xSLI=y_xAIjtvB#PzVTp8CsZX|x=fwE3J3HQ!+$+_H_{gsXUOx9&$VSNOXaQw?)#Z#voAfxk``9b;znR95_?tin#C<GcUC(PJD7cJd^T5<ptB;)e2%k9=oF71*9Yk>g$M-A_-m5=zs-|0*?X3av@3%9oSR6o5%*da`&Bb)5@<%&Yo6*!Zu_tBu-MXPZVkVmK>%UE!d<$0BIt;HXhwlhHBO~d+_uejutUk-Bo+rQpyD9yVZIDL}g=F{Kna9|E$a%3I@6=dPvIYei*DqXVv&UrpSz<-0Q?=c$$wwtVfc-W|=rutkN^KxAp1X9z_%L#BMN8bUFmcgK*lF<q_Odn;C^6H!@EN+JVOPS1Zz$=62NymVqJb_rIvVmun*hN5P+^rBkgD{zM88@^1I!(CwHF)zrqe1bOYmIw2Q?U~zlkY?@Q65<YVZ|9osqOi{Jov_w?F*NV)cv5)06S@!f(Q9N6qoXIG+|7tYjGwPv_y3?@C0#+ydmPg^MIreYdAK8&tAhlT16-T57Org2)g#>W-fgB=>e~tGu-2p*{!uEi1C1&<0~!SLUOx8y6{*)#iYW;tiHj{m$DFR6et_mAEt3wZwl+!)8reiIX3kFn|Rchin9`xxEHK_N0{7za2+KUQHS_paW+NQ(=0zC+4g44-u6+Mc;G?Sf(TOXLe#Z(^~cQAg=yM2WUt{`Zn*Dc1oktE&Bg3ErnAeY<$y&J|5X%SLtKD>S=k_KHK}*$JEa0K8*{js5;bVGQGE(}7)GhPYlTcQj4uB#6$$lZZx6&$@BeEA|FGhA*Niq|Y{?7@criwsaT6*AydO~!c}pn!nv%kax#5U%+p`{a5W`3Y|HSe(G>|Vtl>0C88{I^bo~l#ww{A1)Jf|!0ei<azKPP<&-B!*!PT_hNRo$w@F<MAIGdw!yt$Z_=$Kxi+*$*N0-c&GKF)HQp*@q)Ef}o$uz3RoJBuV@r0<z&7?Pw&t(Y2iBk_KCZf$9W0Dmd^tiveRYj4*k*sjr^O&1nUXuoxH)7PQ`R+%$=KI2mrRYO?PFM9~kelMB0=klf+ey}|~@UiEJsZk})GMcE^`w$Y;@uIt1u=b0$opdM8sCdIKy83*~YYx7jll6aw7FH8irRIkn?=qVu#8ldTdbr%NfpgI_L4+m^tOsY^qF`!V~E74N!&C<}akBWFqq}T^cI(RnqoRO_|jQsrmUGKMteEVJrd%TrQD?@08C(n}AV5ld0=JES43OTQ>GOGEtQ3IQ2n|&>qJQI5OT5VK*v(A<~nx=JaW3^uSpZ_2Wv48GmOPoW3+rsKOEy-fR&f6k8a>wZm#hSBbjs5!bQ&(M7mBBby$J<pn37-ofeRal+h9kM2udFgQAfAAVLl<~|w+=QW=wb;xKwB%&em9{?`b(iuuK{3Tu-pZ<Z!Ln@hZ>)sZzkl>eg^X$4Lqw`*u3&MmUy(lRXD>@h$vhzW^o0tTm_g6P`z;uFhp|hL7VXWqo>NYRR<+Co-|>a<vpgA(|aeNPR%ltOXNIgT7|UB;JXehJ*z^FA(TAk(E(4zJ$cvRr^GY!E(;CGKNeG^8SuERc0!&y+1!11X`&P97S<VTL9&@Y-ytF~DQYQzgI_C82Ub2=N}7g)7z%A9O<oRbVtTOjsSzZ)%nyjs#IX^`8^X1^pHicBse=xQ60SMzWYaz-%E<oGnP)(2@lX!{Ys{pAlXdQ$X+<W-ikkA~v_3$IJ+*A~BtL$#)bYyxV!8>>=kk{(<3-MOHzP0jxobO!*m_QVy}2OhVZc_E_mU|Nq#s$I#XNMo1C#0CVtr#-gg~6|Tki#*($HN7`zqjBGM0bWn3>Ez8uQHO+)u@+9l!3-9Q9ZY%$~#nu?kZU4ozQJeeTJLWA;>nYo(f!n{*sw2~HCFCcY<#H@rzx4Ux!UN)|G1Q0aAWfPXAc;t!Egx%{JQget{A%Ww&N9IjT9fzf<q7_poc5Ppouvb^TbCX2CDZx_ATy!calR}K_-_$vG2WTl@hgzm-rJNIMqHJLaW0N7EJ<9&2lEO?T^xH;qWmD*HX$95XxH~^Z>okoBSl_q7bWhSryp5VH4lyWmCdYw?BpVdj(3=(tZO@tns^&&1M?$|i4=s-8&K_}7oAB4j6698li-^t~DRLjrcfn|3}2H!`Z{BF-uu6J>ANezJw)rtbWy;9uRLqNNTSjJORCqC))+<-7}Mh{uXIZT&Pkl(3%8wt#YaTW}i%g0?P@Wm_ehlE?rkay}|3L_nEbcN%nAikWg5Lh4EGZSm|uC;~BNa;b_q!m-7PSTyku;<k%5!J|C8oPI)n>UFu@2rW1AU1F#u_kaV&Lzg<f0c^uGT9`G){^+aOhRo~as}LaMw4H8j18f1)ycnghs~veb>s`OM1z@V$5h=OjI)@+?F&ZQCIgRK84XxVnuY}8Kw-~h9KYSo__XoXl=q&3Z;Mg#OLvHmQPB4a+zRm}w_&6E*Uk8y{!Eb(j2IiaRFYx!n}$pk1Q%S=3VS%4;?dX+r0a<_XufvBsH|MMO(WqXZfsOHD{-y$y6G<Lu3{t7UYGO5eKS4TnMhPFhI`N5RIkO_0}3yqyKD5IYDdE}HnV$4#%ltTJP7X;2v`}yj{d22!thvLCOP^RZkKyP@OCqBkwu`*L4w>kRM;KtG@!=oDY5D3D)1|!18IhOt_*zf@JTuyrQ-}S8OnW=tz!z90b0FKAJpCsBBxyvV;F+oN_C%C&FDH|T}S&{GyLPOF$*dJt$BHEJaYARvLYhSkfgGJqk3I3;%1$~?zpB#m1?7`+$hhHxJG&`OuER0w(AKt*I-u@*RCKWx5Tbtdv3cAu6D<kw$_n5)886BvD=>{YYHh-08^RLh@4#;NYE_en3+jL>b(&hyrK{9ZZufC8VRqq!6G&&HF{Wj(=!XxEXVyL3S`_hcLA2@IFUo^=mT=U_uNS8;#eOGS0EE+grfNAY!D*cmz#G5qp!NpphL&r4Odf$9e<{Vd(#~x-}v(-jId;>3S>Po=bbKe)*BT~{OGIPW1|d|#m)WcP~mY_tJ7SmYNpK-V2-|#5F%7r`QH^O>x$JUJ3ZBSiRaieXP}iKbJDl}n#g0OD$as&um!JHEC+UYe+YH6RWM(X>;7W8pT!{ZZ;878Xv|p;lrB7lNW~c-se6`8-P;q%vBxMab5bVeaTDr6LWP`5W57%WgAtEg#TJc<;>PdpQ&~}ZxVqGRqDgij^aOVYZ@sj{p-H_!zt)>~d=s^<#<5M^$xv3Proj{Zs=sGxZS^N8tdzR(CM36d5!l@91jYn?b4!%ERwQvn;ihrb_%;|~Aw}lnLdZMVK#^JT<m*5c6aDS52!pv5bCi=<eB#Y6j}7q>?0<9j;8q-^;q>(5mo3s)wH&NH&%Zl88%Yc2$|C+1Cwu+UND_@8VpGyO0=DnE+S}TLpZSlQp+?o7wzluE=k|Rp2tf?KJH@hgYefLiA7F<98$v=Sc@b&lf)x#J_k#ajcv*t?SU%yL-mb&216jHY%sG&fedCyYB*-vlp&c-|nTymuFcd4fcV(UH%|w{p@VjF^G7I3xNG8{2BA{OBYpnKG&Kp^rKq@(xiFw?<4`Q#1$kO4M=M1?h1^BQncrjw0Q!jc$8A$vNnwF8{=6gIWHxwz~Hz_czuc;|}O9gp;<wb^6-Jwlm4&uo@>}$s1h7Hxx$w`0fGMg>cM@=wxF7A?2AEf1wdihmvuce-%B|wRp=tA7PzsfZ|lvAzJBkqU7fI+wsq&WV4oL>Nby7}QcZOuA{7JNUr9~Y*iTi8R6-3}Kp6QqluNy@nt8-_Jn#Ko;kTH{1?5?`Jgw5Ogz<a6z)xoOe$j=ACwDWWsHq(9Wuq4F*+-IX`k6tVsSysZi!CvbOh4FcnUIEu`{MFFm{kLJ*e^)I=;yBq4;yKskNcioz-_@SDwFq`7hM8oBpA98$|7`-D-39L|JJ*_H|Je42_wS^c1Um1=CmLh&Q1XBeL;X&^Y*z7^_OiU>~5x~V5PpYZ*N0|jK^X|QI>9w2qa@%r17%w|he~%R}o9~nP_3|QSa@ku#^V~t-JF3Nh6W1@td)blZtD$s9nKN%oAv9|7dsAulwin<-`%@B>3qocK#m4MvC#>QeqZh_~@nw90!ZE0xDnqj}FxPpwMzrg`@YCm+WST>}{Bwxfc+d8><mf)~6sYeqD>BVb?$R6DBbtY$#BAFG%Kw7^z#^Ep+dMKXe53_rtQKCV>l~^9B&Zpd(ak&zlUi|Rp_b3ZHk%(>5W0Wu!T9nl_GW?RX~4t;MAACH=Kr3dDmA^uU}_SDiqNK%L1ks4ih3#|b^<qT6F9+o?6lX4dZw3v24;Q#wPkfOr;P-rD0QU&(%8^|muZVTGVm#4H^_V7Xwpgu($F9@m~cdlT*5Kt-B){K37oH@pk1I_w5lJxU2vetxZh*Bt=1>i`4=DJF0wuS%br>-XrTef#)!X^mYPf_{5lv7v-0DNDP<xx3aV*#U@32RSE_7_WS*WB9ZR-=0|ur(01}1R7Gpmdi9?-V6?sZ_(u?<rSqiJ;or4|DLNiWXE+a~K<7rGE(V88lmSNtz9(GP@erVR-0O)`TyV(5pH_qn)LaYKdX_@ai*{zPTj85nfEo*%y#A9W$mGeh1pL(~h5cV9kI@#If<M+c+7UBr)FzB({o`FPDM~ErU-)tzh)*l;>N&}|igt}fT5LoT{@l(jf*Lj5Kai~O|XvpJ`bLmV(0{+xrCUgn+_H0nzPNe7W+qKaTqEmO$mEuCz$#pup=J*MUf9v#;P3sFi1fZm>4!a?DTn)@cShL$G@Rb0};#Xz-0GQ~ym#=HsYP1&t`QIJm)?|wvT?-h7#Ncx5V@qlomr0;~F{+yK;?=R<=!^jYT6N`MdqM7@-d;6%2}Ms^?<CTf46bH-0ItE4`SStiy7Inxt%7Tskg1_!ztAHLUI!6pN)Go6>%Q`71}=oIX?a?{zH*CZP$VPQIFv!W-tY3uo#NiplngWAR@a*+>=&a}H!`KTPLDkqHmW(rWW4kvtXs<Q>$saU05|MoAJ@%`K~N#xYJiL}Y?|{h`e7ERoK$VRpr?i!dl>-kd^^%s3yAEO+Gs?UjU1N9mOc$04f58yXqeoCwFr;^jju32#sHM5kB>Ib=#hQJIJ5P9-w{Ej<@UG`WW)<8=(<#OtgXBQ@N<83%cM&=Sy;Uh*)4S;kAkSe#2S0D0J;NU>H?{#ZC_Y!AaZ8SnukakB^+pcK|P#%CBpPr=OKzk1#o_jMLK5fhg$0w&SF&^m)awb0_g~hZhjokeqm_``wf_wT@{17fcvG=D^`>KZL(GqB?67k8D8Bz^rXeYO}Lyks!psngTk1AuW~E>N;|^)geF_i4Es`|P-sgt*n?K&<8fTBV~>XbS)p3@e!n&rSv=b7rLE|d0tWiRSR9Q6B77W<p7={raswNynN&Y3_itUwFTh|Jlw4RlQM}a~`lV)*BUR`T3}bqnqUt79P>=r<Xv=1w$?5?VMiJ4_s=Dd!D|&*9(51kcfkYwdCcL{-2mCQ3^_ued(^j9{&RumvV>{x5!#9r(2*OQ$#353YE-m4;Ik6hXV*n&CJwY{dBOPb1Tz#Zp1k1Bj@B<iacSFCC{+Nyfa&5bYhI5)HVSpd~ML;-aGqrfzrP>cYvH8Ky@L001)k7WR?<utQ|AZc~s5XJulrA&4c$8yjs+rYWGMcA;<a=0&mgtLLUTuG*Mt7iD=e|Z==efB!2^>DCr*7=mQHiBYNG45H)ZIUZQRT&thnlf3D}UnC{eqPI%kac4ek&Qrv6fhbsnYbwt#`>dMT2GiC>YFOA8mbpRSHv6+hdnJO3JVnL{9f&Rj%0vU)!dI9d$}2zecEQ0Ub{OCKB{b?6@+&QY<(j`n{H86{&nbWqoljieeEnbBO$<>xig~d{efY81=T49CjoN5T6jyN~JU3<i5_*apQ};oL7u*eAz}lqDk>gW}CoZK_cs6i#u8lRW&%W-}{`|6n;pl@_eiS{a!3paV#Z%Qi!(asvk>Xagz~YHpH`V_#IVbP42d50FTm*_?VHKVAEI7JkDLfpeCnZXG@?4D*+arDqC2y2!;!b!GNhq6YAMslNoJpmbsmPs>_WY@FQGko*lg@yh1=@6k3jRAN5iPNcKP!FB2Cmi7AkbqC*%H-rS9N=O?vdw>=95%$fC{Cbq8ZP^EFN;{4W!i{f5c?*{oZZP$;ky?&$YaMxf800i&*W^f1NyHrtYumsaa<Wn^^7PH7idt)Vj$(2#E`rT-j`dwqa#(1I=6?hYru-E-+@>MUSUUV4RE{Mw_le-J-hD0q4B$G$Va;)6m$ccZjPdpQNAU)WT@MycOtiyGNa8>sKdFR55LK01bCw@qTMy>vZ%D*^|B~Jcouhw2WPAXH!yVbWaXWZ8m{omuGCbD83giqd|w4x+t!rWck`%=k_QX97tFKaLKqZB%K5DgNh%}2&O?yWYpdJ9f}0$cP9SWZ=2r7VSOvIqA5N$5w{$cHkmti@I;A8R04L7wz9zsjHlWcWMJbMkle{nn3(3ed}q4XA|C?$Bi>yc6_YM?qF^oBmLEXiY&(1WT_nzMCpzDlgA^P;FQM@$opP=Jnb{K`E8Ih708ZU;`F%ZbM^P@WH)fyH2PHoP(h3&S7!7eb;6)UA$q`aX3hGPL7IWgL$Ol8_3#p8PbPs@n*SkW$P4!F`;}_uw80bOMhU?X_{2kymfO&5lGtFz-_X~Ns9RKXz+jrf>uqsKKox8-l?k#m}sFh#^2W$r!6LXMBuYlM<$(l;%spw5TP3{7#ayRCdHN(o<!7ZS7j@fHloZ&Qm6gOT*KOI?u2$%WGumum_fnuF41<Fw01eCjXC3Q=@9}J>XJhhMtztbNFeAIaNO#f%L423c(r)<h2eRETNdT9P`!7P|H5hJ-28+a<i2soQ&&a=;@^cF=+$sMfMDqeAmN)YR}ZO-yWUo~+Oa($Z#bmBMBo3?nPtCXN+85=FZUur0|ditPKewlh<N=SQ`MlFK-aw@Jo83*62@_Je>Zp1-#%AQK%pyMZe8Ff7(1joTS3+wB%XSL-TxLD%;YxBcaNrn5O>!_nG%F^p-#FH*GK+|9o;$@A#W|SuQXfn^du+aU*jS0(vh-uMZxHKal%)r%?z7lRe>0ZFQA=QMBf)lRs}JmMoig70uv=8m0&*?nSq$pH!h#(Z?}v1qlQF>5sx@C?Uxrf_d-f!&NO?IaiEFVVhhUol3q&H{KK}B+gvQxW;2%-0F{7L_*VGNT!im%bx5b4O!UR%@bchHI5IIuS5gz)fH$)$LlsM{bFzcq^WA+SWQu}6OU|p~^Rb1?ncT<dcgp;z{W{}0e99&+X?%9QVOQ<i{_73;1@wO1)1GLO3K!<-e<J!|TMs}Q1K_~q57LVj03=`#pcgrtawL|KNAperHLJ~RMpiHeMkBCQK)qXZRD%3yqFx;{P<OjYAus}6i>m51ug<zeZb^^lVnsK&!8?ruk?!ySF2jA$QSqHt^-3zWt!hq-oW}Nw9V<a;!$MQ7YdSJjPQ=$GguAlsmi{8|#=YQ>?k7kn6KsrL$h=m2&9SG(f%bl~eHOIXQ2YASxL;0BBd*Nsts4CD5a&QmqfU<7Zi1^uNZ=zBIA1h8;wkW|ur5q{AvR8B!K~6mfn+G4xr26>9e7qS_ayc?Ib=3-kTx@{ge5p0QhhpCLt>O0z5l~}=e%2hMZ`nu&uT6G?U44kAu^M-%?lW#Kq9~x4V8MdQNTW@RSnNkIncRcWQwJ6CHlaBL^URHNE9*M?=9jDM#^(Ao9ULr0se+p`$Wvc`fAj}&dkA~$ORvwbypum*DsW`7W&GJY~l%yn$!RcCjghA)K&j_?IccCKdxK9u4L4hWAx*NEc1PF7W`r#I;c)-Ru;KE`MF8JxW;xug9!2o<i8mFGKz!M;%wTmNKjaYJq5W#6bC-nrKE!Z`LJ;e`P2zz!>7VKaTWrJs;Whm7_rbIk2XX6Zbh)l87Hlk>xq_#Rl#ul>tcMfUG>$eN~%Rz?Ijxwoqd8HG(0z~y?o2rG#j+Q4!otW-(Jj+<#lx*?J9l7kM!a&z>z0UXQewrLa@p-L!w*<Ueeo;7lt!CwTl0T7u1YpkK-?da)1-6_xoAu#AU)3nj>MdIpsBnqPg9we>S*KGjQc<j!f2Gbl&rE|6MFs%T@%~c`|Zs&m$<c;DobYar`$d2qh-vr*a57jU5^bn&mzHyeeE6VT^5N;?-UJWWxwO8|_(MR)F?4LhIS|ob|7pNs~wyfb(X72|#KUvDl9h1i0V@x-)15m&lE;UoioAO=w|0HvhAHr9=_)iWOYcPV$82wbCT`yBPNOmwn;sXo$N&lgD9%pjqM%!(UzzgoalJ208yayeP}tRL%HBeKgpvWbP@?pIJ})2|5uWe~wAdRL*2K`Aulau9^%cZnm|Q4U=WSlCo8y5vY7qB#X2y1Ag4pATnjqJi5s>FShJ^I{LE>WgCRKuX9-PM!e9ryV{UT3SiIrXlD(}#i&xCRia{kg*bo0Ki9fK#RB&3Y?oB1VQpb$P%@6n(N!^F=-iilnCg<F#M3(Ob*VgMs~o3sO?_Y|m68|^WbR=5azk-yVl1!%$6yjZ%_ONIgl#?&v-v@}l(wLC(c!u0y+naJ`3GL)*=*a048TNzPm#A45EY?02(UiZbe-e>c!$s42RYmK2U5<<P1bKu8wJ*q;dB|evb<!&@`DPI9BowXKRmn7>m-|}rS9O}>H2{|c-zc-uOxYyZ4(#v<<(<Kq5vu8FF~8P%Tm1*qHd8`!ABMSocOwiF44+SjKk40D<g|hZ~i|E#EsMaR3x2T4<YvM6I`*3-jRL=gQQpFMPW8dZd<T|Z_j%+WQ$B2HQ{Xt=`~-H_yZVH?N_4okAk;cC^j=1+1+}VRtv}uQ1#IEP1v?qGdq-ID)dT@n5<oMEiZwu)?|vOCZhSz5Y16yLNP5-E$JkHd5v^yE|A3@IOhV(%i9G{F-{8&)yt6_3}O+FN7b{<EoF_9D+eP0Zwu>1=Ep3hX|s<e1Qa&Kdae3SLf$$BPyr!mi?h49kc&^_zwM)4N3JjBuxP4KPKzCsg&U2hK6xgcX97D^VedoJHkV~V{;3J=wle$7^l5>FJd5XkXi9ISS1y`=JxK48VtRsHr1gD?K=3FGkw*3aDcr!vsD3oi`i6H~ae1gT5F~SDTbH-<d^-5=Nud_SKz37GED+c`jVQ8`t^-TR69ITTI2FDC&6Z&*vNu1e|AA7*$rN~Ilsa+XS~6VAO>kb#p`1HcUO?C^wbjk5Q>lDm!Y?W4`e&kr*-!jE(8u2Cq{vl6#ecnXi0JIz^W#6OaEc%~+7HH1vTWuF#@C=ol|jeEAsr9CUxxHz;(FunNukGQ?R$AS(1v;Fh74j(U~*Fkh22$g^sk5s1U7oY9XH~u$^tmFHeQh3-y$;k;!uh?aYT30)C6c*0|^x8^pr01Y+cQs#E1ix`-w@vRj2H7*CCWUE3SA86SX7jdhNKuNC!^`hFiQYz_;8bR4G|Jnnsw*<b(V-ti{uRO2-Qr05Y}(dm#_Kg^Qq6X#sHa`!$TTbB7#9(X2@B9TK1$k@17E+oMUH`Wxu(k)9F7vtT*-0Gsi@zE^AbRGf11X4y(_?ouVA+o1aiAFv7%2bz@@!;k<|QMOT?TD{7`9|I#fT~JHDRA1*NPyMwllc#k7OQ`tvq+L_~9kXN!9D_|IQ}?@|(7yGf`i)l$*$n&2hS)+>Ihmo^8o;qejSPO(9GnHR^p`zQH*lF>7tPRFnS_hsmCwn=N2jzj@2RmANHHb$dEt`=%q`gdbLg8|LS3VJNW>fJ%2FwqGnu>Lj`w!HEr&GBP@$sr7JK7V!#0ip$s(%2sAo#Fp!r`I_O`HoL#(H8(s&OhGt$wzGYR&vHQ#O~O#3LYezCKQs`_vXYLpXE-799lf{0dGqPvrhnI0{cwj$hGMe{IIkIFz!R9V~o2S%++Hc%*+FE9JZItMF0WCKKex|x7;X;6YV{12I}62J6s3HjnWa%-6oA>ZExDMwm;zYcwp{Knab^j=r-IZ(d%_?g_Aoc7oWE=yl4UUn#UG&BByi8aPrz=6*E*O%2pk)VL%-YK*ue4+bl6|}t*y#rMvg>)Ix*dS3_L-m+2m-FrgF$F`l1h`2Z=)3;Tzmbio2wG=<v??^^5`&S@IKK|UC^i&<+BigLZ7%l_@6iFfv4rC=vmUS`)xVk{h~wS+w5F27wMxg@TFE{J)C?rf2(=`HTVF}eMWAUhQU>azDAB%pZlpAelv~L0x}3pXHSM<As#51>T|C%L-6hMzmhaKQ|3_o@^LuEoKDIB6-63#NR!dp;X(|L;fP))ss)KOL5CuM$pT5qr(IDs^X!t+W@5>?0GU!Xo=NR9kzZ-0F=Nx?nn3xcH^|H(5^j0M3#033S`zK6U+7^_$>zfO4jxeHRCfP(#<WInOe@uTzQss}lCG&c_<hX_OMvICxs~cmS4%#+YTZRMiYXKN-{U*=@dT<M8^oGUwnt=1rL>~4=E%g|kT5i>O(b{F-ohWx6o3w~0U&T07qPrxM$3s(AD`z<b7tN7XSi<kuVi1g1vN@TrnM))b;ZpQD%eDfYt5$H@gr1E@6Yu!8R&gep9N<U>-mmeplUz|(w!W`uL5wA)!1q_q3mMm*!^xdpKo$^rXND`DKOW=|NY?)*sH!!#wmdS1;qIw~Sg8pfjouKfOb3gcTD#yv#icaWczP_H4*Ic+k@1_3$N{DfnkK10h@w#;aF4W;CK&M(en1=&p!r;x_hEAk4l6&`W&M|Ozs|T0*l{@h5H7|I_*RLJ09k>9>-k$O;)hy|*<RWX{3FNI2ZL?D+3gxSazy`|z~db%18i;i;mJ)pfYYscZ){_rnH$JNN`e=214ZrHZ(s+qMhp=nMuNa69_jhO`7HO`a&5T^l3GFDnc1G92PS;tIU@JjT&209R^&f^SlQcla6dg4YAi*^d3OegZ)}-C&Ka^K+H}k*4rzu)N<h`X28uL$CN^Mjs-ZYc3bQ!HFpH{jRKNjU3T=f6FOeJ2Ct_av4t=?7iIz^-5!tSuq~z7mw|<>1FFyYxdar<0$R|KNZkQ#x)qWs=#2k>*w5D9e90sKiwHsNhZY_I?9iZ(mVryttPV2FkpATcu4!boQG9b{S!bZJ7J`qeTOBZ($d~+nesCXWirV^dH_A%n6?5D$DE8Mm48Hw*fvg<A&J*FPU-!$Ne|H;C=Aq(frDu3*Xnfu<sLy!8TUdnsa?!7S`FxRzT<Jnane~5p?-Y6VHZ@IV;Dc^+IBCg3+_N`hao69>fm5SuC86QAaW7ptfIZ{o{H}V8;rBxH=wX|MY>yoTtXqNq(TBp}~Q7;!Khg%fa1gPZJ4e2kh4re9AbJiQ*Nh2pLr@ze}@IZWd!PKEeoQGR4yjj?7_~5DOn@$QVg<{^%eJC5WV=Wm_3he(Kk^<e^m)wEzoT1tqDQVCGWX6kEv!HI<G+s~*uQO_A4W5BbS79e#I2BX$fW73-VybsQvtwdQtB?~+ps_0v5Qk^O`4HR_+QHKl;dP}e@XTv4hyzypcQp(Wg<b{RwgV_8!^}q7S?+;^xeDG7e*ZBgGySwFe84Y8dM83M0EbvJ%pTJMHKc4`Ix}pNBr-?OJ;*?R)u9DK&jyHXB7$0yF)M{XqH&h%gwR4aQ5;MrOo0+5TLJ;2XcSYcR%GCOjco$1UQy52ay7i8@Ixh?$(z~(v(Kwr(aO2|=v^MlI^BKYc`))yJ!MMHstgq=8{N1M%mQcnjCA|;SLFTId=w5vJNI{I>0#?Exq0I)A-(j3FWR@Sg}DM(Z^4Hu+UQb4hTM{*V$n;zBKo-({pDCYuzIK$vK&jLx=0p#5$vH`=r(1Gy#Y|<X`DnHk3`>J+wYPe?P_d<>8K%%l<Yw?>qg1W^a;ubTwg|olgIwYE>YIFf}sjl#{J>o?S51Xo}B0Gr6_au%usH{RO{4-(k#6sIgKT5d&W0e(G+6NW>DTs<YH?{sZCdvnLmgjp90^1QuBbO9z*P(iD-DXblZ9?oJMo)c-eov?kVQpS8E9)bq#)oEH(v=lLxS6Ti^JVLzo%cWE_b)LC+!;6rTVis%Jc?(!;MB$D$X8sXS4)PKOO!#ESX4hI8XLaa>gdX#*GMC)9&<2pG8r8a2Y9#C#%yYkARRmCr*t6Su@*aJ_pqY*dKw3C1SHi;-MD>aOh^No(1jz?*(JrCf=CoLdV%GxGoct8eMlia$N5ic$3+`uGiP9h@I)K_0X5$YL%lg$dmbxTI;9ZLi80flslRMP^V|(SW=|LGy6`W>WZ<<q778hxnV-GB(dZ^1TYPuc-uCpmI*lRH~v-lDqoSf@iazI+IL(d3ol)?W-y&PD|%Ig3+c>uEv8<QIoAZn?+hAufMk<o1pQ9XY^mkSY>fOyGR=}^7Ag>o-`gqPGRqr79F;S>`_H!SjXE(q`0dHHl|7#_7O-OY57)xn6LhBb=l^|3J?zwyBzVH#pX;{o0=jBW42dB;}dH!mL9PA51w>chWmU(EhH(vQHQH{5s)4G`&^(L-|W@&YbZrUGeAI2-d6~Xq(m2Z*jB<66QWQi-uw~b=<MjxzT(y<1-uVLRf=E`f%?ec*&&&7+arQ9uQpcoZnGW3hQ#iZP|_(BT$k}K=2k??pFy+1ykU8ZKbJ<vI&N%Z7Nf5tKFA*1w2<5n36cafJiQfZC-OvPcR+TE+1j;iS>52q&2mXVSh6Npz&xk-h$w`)wkL6$Q1Hb!lXnD4D)4l55&uQqK4L6nw(r6=zmt&tsU_B5%g+(p8=Bayu4iH&C{&enBu&wBK418|PPkacl#-j4Zi&+tofccUNb|CkUG_mmK>AQJeT(+?2ySrKT6G6FW<$ReS=2yPsc85Z^eehBU-_u}CvJnLA;#R!)!F-$xs;m(bCR#IjKV^>RhJ9I2g*lNUzThzU+8SD>)=8N(i78&O{gY#P?sf(C1aX^!kAks=rd|i<e-7PYxvS8h>_>UMnpKo-fKP6rMy>8m!=~2ddv$FE_ET$KUh2fJAT2a;HeR+>M<Yq{ld~#+7@d1gaU`heLwR34*AuvFNJV}n^^Dsu5SU)W92lxHgD(-1Aqql&{Z!}9eU7~2%;R$c>mOfU^l`8_4c@6X-4jXW2*O}2N#dOl}|aF(Ip+c9BioqOOb@z>|L_sR%9KJ5E-K6e+*trn2UbC!TPTl0bJuKB<)!*D-JJd)r90%SsR*|Wz&!q4vbb0&P~);bONj!Ja)YKF1_0NZ`Zz!6R_wZZOIXcfY0w*0l%!YrrF^i(ty)2gg)0Hfju&iFbo#>uYkBf@2Bcz5QU-f$Ex`sYsG9gOdqk-S=p_$Hd&vKw+-L|!0TfrRPP%J(xKzcS~S4lrwk5k%@i&bID(YqDNKTiV0VwQn5Dd!=Gd@RJW-N6`DP`o%STbOuTBMT%LC5*PZNhmET5&>L0^Eudjj^ccmdBA2A-zsE{Mxxz{`I2Tf**TNThy2)HtBoDMBQ6)UPN!L6N4hF8ZTAw%75VI9!@Ck6-xr#`3Px2Y?&mYj@A8tcjZ?FRlb<R8Y5z{s!ya_h&7mW)LrDkTFun5IMJ@{OZvgk!|r?+2hF+SQ7<fKGg}!3#&-rwZdzH)OdIN#T&XipGK4Ngpa;TcQv~k3cIr$^P9pqJk}>Zn9<z_PjLL_!-ot5)5BMxuh8fhRYAn4_)^abrKVYT`+#&dO9BAqG;kqFsOL=dZwAv}vpqwvU2}*;lam~6@jxLY+G>!tPs8n;h`*vrYbs1J_FK>ux%jsDWE$s9U3U!?z~MyK5!zp*jYMgq;swFb*mKWp30}dM#A%=yt(l)qM(xfarL`*PoegFPOO@ImNu9pOS-9V=VLpY>veEtdTRGyJG2y&EgUnd2hdb+65N{<Ki6LN&6580J@A+tNAr7%#CP>d>B-CE^wc()A0e1!Ck?QB{%k~%;rNLIG4xvzKDJ5RMvAZvJaa@PwjGs*asHMg~#N&4@=ok5e9`X^a1E|6UXlr&z{|EW;(dM|2Sa!9*KMU9EB2|?=rIY%fb7<EfauJv5F38B%8?Tv>7PCxy<qjR%G?mG74Up~>MQVgv$j!huOj0Nudb%Q@y_R94?7+Pht5xngpY+D~Gfm|{vIH)EkBkn4uD$6DItjtnQ`zVQYJ&VfX&&Hw{dl;Vh(|I}Fv-f~Oh^q_$*eL|`2-(KUa)c$JIFr^ki_jb$~7WX2VQ-GrQR(Kd)=mKhF9IjsHJ7XHJ*ji!jam54n|o3M8zOS=S0^&<wSu>*0AFW;sC(@g&zPe%VDpAt#878D9Kt1ckWEN1<2Nze(W3+-vVRmCHCJtTx}U|N0SSKEgwsPAc|#ui52{_6Hr8a40FpX+bgPref&x1m+*{o%;l_&Nr<XSm5G4e;Zf{!p&@)6YhnO>0q$j}p+=C&fhPX%p*wyBcUdkf>eGEpGY~F3&N(EX^cCkh<AaEOtcm%*zR{mf5(3K@>7kaUKV$OCBIM?smW^uN3DBDoivnfXf#%IXMC{dG(F~=2ScU|_oj{D@XHIIlFs11YZme-V+sXsAK^_c%(A0LhCpXbZ-5&Vc;+MXW5s>6xqVsmMa*>9dA6_D0oPVjYl>3vcL+eAA9f*8{vfAL^413cHR)+V<T|n8Fe&pNitUFm?`S{%fIpSGB;eS_N%G6=nwFYDRpE}evB&Z@=Q6~?<FC4&g$Aif-RHkZ)JSU|ZgT|Hf8||$ZO_6z!5Ko0u2{AVV=pSi&Qn9Sux^Jl1L`W$PVvL}OJ`!aIiIJ>UsjYQA-)-Q4(9Myo^p#^#?;MSiH$-)|u9d!o`DQ@O6@Hjh+TQcgOPoPJ8SBm}1(ejg-<xPsfA=OpC53C5@K}7(JlTS7_HPMJatkR&Ip6z*r-#P_8fh|Q?+&uMQ(w3u!)P0jz_!W$4dm59bT7y=8tUCiLYtRTqOA>AUSn9{2)o&F_r)OjMF=C&InehU>eVIuoakeMRVC1))_8u^k#bFcpEiVwwU6!ggashwKUFVOZ3M2e*^(0tj6y<oeyBm@^Pio~NZ^m^BDTlbfDU6C@E6)^{>A_t;6-i8k3WQ7!$+5F!v-tJ^Yd7#>X6z(?U=~*MoSO1<?t<g8A=pmWa;N44VyU(X=jb`=6wildKZbBF_jK{n*<d+FjB@cGN1+E%Z@V$L+)h;7TBcQ>IQb?C`xOCo=&CxX3qh?x&C;r_wyfs?jeEWolyVKhHCd^PCO`Q#kYL6)rKMKOG+M2G&c*5C1|Gdz>l+0C^USU?PY9{Y3Iiy%DjPgO5wEVweJ06At#E<r5~dX%XuXY`TzMoz`Mz~@RDTTcvBXnC9s}_|HR6Sq{>FqMM+{_@R7OZvzHP3h><YbH4qBb>--{?7IZ|zpdm@6$SWlLi-ek!1E!S{8&PB4xU~=o2P^f*s@we)84zD(QT!IjVrrw;BQXi+QG5D}N$#oEs&MdaQUXC$ZAWxM`_nx}o#+mFd#(CkYG9t0K7NlB(Uqll@6icN21NaD;$MyK!j5kd@OhTe7~<w$d@}EG7mLTOM&u73XRS@3o!<8hd9oD6tB-3Y3c73Z!18Jvj6cyUVNVHa7qv-NSE+;9melp)XuCN(6U=Usb&e|iS3-{?Yp~Q7rmOq>6k|!eI}q>dw2PwCR_zMZkPEy;!}`ySQD{2*47NTFB9(O=l<*N*rH1vz(FrH@KZw$~c4XR4bal)PzfwNbp?m`fii}(|UU)$!R-O5v^HJ_ZK@Yy7CBZHDz*T|BZF%ta&psl6@pC<>3%=<#NC6S2I4<3N6V5ld*^0jLZ-RP}uOkWIFituuk7DDb;WT#x!&p|VPUaA%p~0f<VZ)NEkL7otp8FiYSR(o@hlcISDK_v$kte41@xNDGp~XnRUEN7Y7;F94m#Cw&)VB8ts5YKf@s${&K{v<viXVQBVVwh~XvYcOU#_|i|7wh}%VyN>GfDc9mSj&l^*~oG+--@=w!>!Us@UjYGBwQ+ljB?~`?&SUs=b_c01peE$*5ymz6anD1?@??KfdD)&EQAWp)fX*oP1r~`f~m3dIC3XO+y}iWN??0QR<~`n=492QKHcp7?#p;jJ>~S)M@AI!NVpq8%AAuD8{NXLx^{E`{Zj}uUf>@jesZo?^)r$@q7eZl|2f<<-8hV9lPpNyQ)sQ-EmV(le_~I1i|4?IWx?T^A{Ce=KBHh{#&<p?w*GCO`}_FJNGvOa_ejk8%+}J>4#N2;by3`s#>948OgUJoi&^QoDV|8Bql^|D_X`?xh@>pdNj4ki`15WmxxRuj@(0Ss{#d@&qUP0Kl6o@g5q;+_!$F#$^FdnH5z+!88&sCs$-mP{B-s6Rz{2kBy$0A{rXMVGi#Q}>b1`DIf%)L?zpln8LS|5crapDK95a;ttt%z=tXH?zNmFoR$yZ3qo_&Aam0dr?uYY3e13Hz`m_SSf@?8W&sRe&maD_GAblf=n01AHio78rB~?4yi#CE6!lF%P2to$9U3pM@h@$e{!bR5(kPIj;|IBTxMlxtD&6)`r?e>DfQjggL;O6}vf-FD5C@sLdwcm8C$S2o1Rha58aI-r<qA369@NZhkV++k~7XfF=Ra=0Rk3<x^#=rY3W|c2ZyEqYV-|xIO53w$VdUq(R&8h~c7%ph&5hFAY8Y(LBN@>m?L`oUkbgE5SzCoy_;(2g^6m+w<wgwzE^}sG;$(4;_LpS5e_93A5Z}YPKl2)OFE^U~guVV_qMb8`rc+uPBA3U;ko@#x`+0Vs?<9cCCL7Q!R4OTsNj_nD_>Hq)!OLXq($baPs(Z+G(fwcXZ<yjkz8KMf7ANhHXGpCU;htB!*eoXZkku&$9eK*Z20SjT+Ny_>=x;?M~?p43oNqcV=GdiqYcNw)+WS$$IJ`0?g5oSt_Du4T7ulOVn;6B9QB=t8cWeT@OYi<3_qJ$0j_(c8wlIGF14Et)~Q?;NCERq1hC55m8BV!zxU){qjPNxK`ONgqXWGeVh=HL~6_1OG~tS|gL*Obp4=erHL2;Vg<)ICk`j`Z^U)A6sef%2W!vCH@2uQW_*OIeZVpu^uA>?l4#Xha`RbE}Z(i`%>o91IgTE10ff&|nh;B;D76&2GjRKPw+T(|y&z70UL<@C(=WmbGpvIck;V!%Jg06$ThY3Ev%(vH>O!-mBHNtzj?@s^T7UeO5x)s=UzyOn^WHKh9(N1Q_A#Ukvg*Qt+7b^4^nuR*EO+ownEly(Qx@GIdE#Dzk-0XyJrkNJ8@nwMMk3Y}bFkzVkf4E$j2Pc*m4Elc}!OMina<uvuq0is>{MwKRZSZOy;h4dtNjs2A})7EZqJAF(p?U^+0JcrKCvsb@etQ0v!ptTbgSPR`jqlrA0dP}4b>h&RbzWEnlW2jy-V^it8?P^)>Sz#3X1Y7ktI=QU4g{3*g{Sek^db9GC_KnR4<yuSQwd1bwv5ukQI^Pn%`jEB1+UjSR}j8clpxWw-=i$H{huhS$)0t>LhCJMz0<;b4`Qk26VtiiVHDWI2cw!r&zI^+-ID(r!EGVat&`tsS;-N31`8lDhF^t1#<^`!J#popKI%omtMS0#pz4JFiR@7m$7APv{OG7{Jkehj|Qo*=c(?xFOb33qQKoq&=B-VmR<v;>FgYo+1WeP~|^>b;8~u?6p`MHs|j+FrLEpw%*9Rc`C>$G7=P6EW0(xQ`kuwogqM&Do#;GH3M?S<ola6hqw*K<X^E8Wn)yd+kFjAJ)_~d|$*D0)|W<KV1b;)S{;avPku6-*LJ3*}!}Fkr>+1YQ%*mC$e(&zG8AJSan)Iy;oPdJ=3<j#Zhi7A-h%!=#8l-PA9=kG8ut&Vx;CSKQ2G~g81=onD-a#v+Mkmo0HO#fXwX36Y!H|^lhfwMD^C<!MvGCZ`7220_sl8cqBalNl`cGKoUI<qah@PMhG=@(gnz(1v{<IjwVQe0=VP5{pxAPDpPVyvC*Q_3Uvq_n|$}7g1j;5TY~|(mw}M49`@k^v9_C1?d1d=nI;Z<I9^c!pJU-!u-qha(0}d^LbZ~>h<uOCvM_LHB)|x=+v#katHK>nuz1%qE1dAlTNpO|2)V0GgN?Br7{pza6Nl8v+qSwqmDFi>BcL5AX3$>)TnEV6^1kwXbEr|qS4_e*fC~6?OX^wnR~O`3*^9!}m;UTyQo=5z>y>W=B-uEO7FOjK8X?MDmPdd+#&q?wflr}&?1Co+`RE29i{@@s5l%0z?yWAqkNmPR%Kl^qsl-&qltqWbA?yt@`z2LyE167uJJrZw@iuiRJ@zkLh`QiN@wNwOqep+AA(5Fn)8%Ldb|k0n$DO)rZ90F0P!MiIr4v><uzr;{6H}CCf7t1>i=IlNSEPhcjJ*dy^p9tEL}!a!!RYk(;q19m?%-a``x<z^OtGhTM{P=&?v_h<8I1WEqr?kCRa4^-XuJ~_*HY=Ffl9ZI0>Z{~%*m((tA%j*@bQ%d59Cn4wVBtrJ0>)zrj->b$T?9Ag^UF!%9q89^@6&Bz*~X8GwV1-EH#gS5r0VHd=wdCU2?y4e`gHVzb3ai#UP0D7Zl_cXk~iwZ>-ED#?-8BGg8r%4`C|J8*EJ*k^CS1rPTmM5@NPr*z9dNZXX0LSQJW}HAY(VF9`WzA&G&3g7dxL)jqV&UV;dLAqE|8w(DC28DczsD<34q+<_M84u%y_c(^9%U~(_+mbDDzH7O;W&j<ftxlS7E&3A~dG+h>Fd^&x4y1{0-6s=(;qbtA{>Grwdcjum$wC1Blq&I3{lM{1w>UckOe8!$21QxJ&LfeTI=d0A?V<Zyf;?z87$OQ3kB_%({eg)Y1IZ;!h*;q)=yfyOP*DAjkPQ3HobQp#{%H}MfBAFDHjm9Zbu`W4mt$cWK2K&CZ(Ds__Z?NQ%%#6<xdyq{tU)4{IMKPBpm6~m68QWp#+5O{%_7*u@qguOULlA}VBwNB8W+>Dv>A?`>;OQzgHcFI&22_kt7Q0-=+XtJan1+v)xt6c?3^=Y=4epPylu=8f*f@SG3sh#AFVa4k?2crOE2@&;)Q)|Ray9i5+cMQl<cb6FIG8g>>p*tZ6rWQgl1(D4)j8&}w+xMhR97P&@z^=1@+G3wg4<+905Me|O@8eINi<>9T{wNG-wXT`tB5}_9|B0F*8L3BKMTy;W^{|0^hfm$Xz(!U1T`s^x(zcP8uz7r`fVRB@eg#vuP^ybXP%VzhE5d(vK9k49(UavbSO`-B=FTp4=}mxlwc8DnL0l1%Y*_;H-ijo-cLQ~A>EbW3cKYB!{BKAll)nCMYC9v5^l05$sfX0dM8%nh-8LGuc{#*%>~JaS7qz?b`NTDCq|vi&WDUGF(zL=Fe)$uGeLo4lK9(iQExW|*R#}kNjPBJ8Hsie$7@KCb?sX+8bozh#!W)_;USxS)aKz6ILzdjBfvH|qrBY``R@sSc%+ESRPK$4{Dm{~JPHxsdkkk-G0{efw?ESkVt%STQF1M+13HQv$V~9;GX%%#uq&D$Eg9GI?;DY;A-JK8fhno3B!$nmzZPHVLV6cxCBudW0&>|&JSZhNHgWY@_2IH(BO4iJaZRTwwW}Oa2in=&nJ+U|PO*Z~m?`e*Xy_dNz|Vp@Df^<>d}Ko^X!8vMcC#hnJcjX7e47_;S_3ax;#HYNXde^)_;f<3`(<ObH6cAk+7RiOF$}R9&75$c;GJU*ZQgO`U2YVENEN0PD0F0jn?r;<Wk;I8tm;)rLHn^ak1cN*bqV;Q0-t4CKjS4&(68#Ts8fBu=M%l^kU#ZMsl~UlHHRBN!-%na;kAXw2hI_!<@1DQVSVVhQt@;3>puwu?2R_=)??vk{e@&OkS72Pg%8HX>#g7JJlGiRH>%@nUFpb0k!t>qH?&XoJ|chu?dfqnO9*)t{D?7dM8FX^|7r6AIx8)=B@yk~<y5M{c=~*F@gU~k%ma42-y>%nl>e81kWX8W<0!o%_}27<a#~f%)9aKIrp2|l?~LHd_0p?X?uz=xbjXi501y{r0?uu$@cl<u|4{LP!S=L}&I+XOc}HNdb>LEXHwVu1^vl?Moi%%{hdmi>UG;ViJjt;Y7uxf<bJnJAJ-^Li2XPZ}vn*g89@Y!xhd<~=&3r?ez6ORr+?__ZvwmUfbysd3PAB&a6Iq3r4VNAWs`hl<R-PIR=y_XnoX7AG+s--H9OaD+g9QN`WKru9gX@g^Pp*SbB)AuNT@PctQseqy5{oG!L~x_$j({p9&4n_m7-LImh2?}l)gFR*>asmfNuDPZ5qEUn<S;f{P_e|*GYrTUGgZ`&VSSb4BUU0qa{3DI;7&w!P~`Ehexau3jgzdY5@62wiu|BR0bs$WQT9hPI4tN(GP9X9-Xke!X_;(>4~_#g+f3`+cB6r`v$IEpqHiZUM=<!!+Jpr-5KE%X5Hd1b(f4`KJ-(o8is8D7Q4J_kB)J91mQ!IPNN+gE>33yzXx<ZpVlMM21#Kl#g4gDMZswk@9Ysyulmbyq+SOTc3C)C3kgW}n*xSnSNhfT?Zr@IPVIXZPptot7<ldLJ&?h-2X3{YrCl~dm{fvXR;qi$f=X-EUkn>Yr6cYKoBkFRJtuVXt&k>8)+;(iAO+DGCui1{4%)bE=V$&!tT#|)I5aH+}7Lous*H67SlKyvukFm;*zn$WiA<VrUx{*UdN$KoajmPhhY<|v5O{lG?9lzuj?U^IBe)Wfn0G!q%X$VjbZSf@_Ko8>>07)}BFJqe&^sd@|QjJKIP&sK!orJz{=q_?ue*P*t;Lf+egvPnE<rGfQg4&k5w56)zYF8~GPB$1xV-RQD|Eo)~oD={h3Vi6fZdHHY?M4A<OWav4PEf^Xsg|2F>($xLUTXXQ52twty)lbf!Mu>%pKDdqmgnpt>`&slELcH_kb)*7Sd(0E77*=Cq71u}3Xq2NkFZiPiz=?u#4OfbJXRZX$vUAT3&cU-OjF(sy6|a7xBA$eYF=V{K}aE8fo#O;?HhVrZ8?%Spdcw4YA8Ey>SIFzY&{M(zN|Xd;rVQ~k$=_$V4G2-e{}wK1KkmKT}?&zeaeg|EhI?*C|L`J#9Vb_Az3c9;b{>pbFp3$4YPWryIy_pbEw!`x(VN4a)Jf@Rc@4WRYH2ETEaNx%*`9_Eu1>F5jsZu9fX<b+zLjl{M&;GcZi<5LLXE{1Cnb@HU@Pbw%`>0h+?M^Ab~wLDk-r6LJW``&8uQ}=bANiirZn4R)(ka>nUjGIt(q5B6K|~68ts%VcjfHrmgno^c;Zj^{Fj&K5k`A;~K3un7@5to6WI^%&^TeW_5!(?0pbTDjr@nOa+Hq=DlpBG0qNr&Q**T=1e1s2uS87oy6&Xf5Wyk1@cY}fM~GjMfPJW*6LWJ!jrqV*y0nj`JZACg7ckB8H)PJm<oRS?OeG#iq3i3420PY@1hQH%GzBT2KY>MYq>s|au1PJ4i>dk9_u*0fPNr#WrzVG5FFfNrfLi7O7%xr8C~0Wz)5!pqK}!|zCXr1fH_^(PSNW~^RPp}mrxvzx?yxF{N@hRQ$t1`0XcP9^k(>)iwV~3>FEUp-VH^J@MgeTov<})rQ_TKC->&VPK7~=q%5*3CJdn`CgB^$68U*!g<t;I&}N2Cu9n`Z=7e3X%azwF{j5tN;J60eH51_E*o1?JJ5W9+C{Tjl%ml(nM?6<TdIkO|JL`Rd&tMmQKo?!NQcV;SB6PdDNj*)ON_A#-d=60REmO7I-R5T<`$Swu?3ivz*pX!nGt**c#3F{B9}_i2tx;Jtvo{&hEr%sLwcYTJKR_9#nWKyGURPVXkhB-xEaT!Nw3oUb)DDEeM2Om%Tx8Til{PtMfFX$(4$%cVx;4vSVDrdT&~df!$X<Bxgn=}urEdj(^v_NLA=9;|cFT|wsw*9rU*8LeQ)h1<&>w_!vriDEdo<IVf4B11Fta;9aa$HH*3Lbb!&6c_Fd!exP$aWO-7?Gk<yzz&kZ<7~ZNnZVR`=pe&kH$%RQA&t{Y;kxo8DYo1!uDFc-5$5RCHv7V(&IJu#UyBFrN}|GVUP(FmV94ts-Cx3~Xz^G9An-jqJ)tW%9VJDE<AHmF+F30oxWTcDi|qA&x8>d9I^_@UAZQDd9Mr{&NQB%CkSBlwD&s+b9VF`$g3`sq&*&{uA|#TJ}aWAEvs1BSk3T8|}Q%eYObWo-p6HEcups^axr!O>)n%n0rEoGpUDeM4EG_vZSlpnZ)gO@)l{Bdh^X_-V-yBWUoy_nAyP$#Osc-?4@%dGFsqQaql$2f;7eDn@V@?2t+K<a??fE85PZj0jrWf@rx{#CncPN>h~~CCL&;hH3}Q-IR;xiKNHDH0$W1dlr!jS9{XL4JEaY?Pph0JGqRFQmVSZ?e=Q==JoYl!LtqT?S$FDfI3thQW3mpn;F$}vAED3Q)+EV8(C(E7g>+*h)%NR6^@ag?xoQ-sA?WdkKPM?Z{eisvnzd@L(yq-39MSgKUaIJtnVdqaRiL8d8gU(VhQyFlERH&HIk~bh08xN;TekPL+psPLEWTko24D^?QnPA2q1OFwsO*l>bn@#Hk3-#=3tlz~j8rV)RXFwPG#hFu(L>7J{rz2CUK&A#SOJY@cE=v3^eN)^n<wnRPC=tz=jz+x9%yu$h<KK@j%w6-=W0|#RYZ$m+@Q|}?m_uAB(|FfKW~cJD)lh{?PYXZ_X;2IG94I#9*}Kto0sPhgC>T5yWw@I!>=1<|2SB<JXtA}Ya|Mf!+?=co;Kj#Qp*AS2neMTtMbYS)F8NMP0q}fW2<0i7Yu7l4#n2?5WG4^lE`aH@d8dL&DpGx!l*PiVCt@(l&Ry-&4#U7sa&fmBd$u&&`NJCtyGT7IqTojZ7E(%g(r83JQe%v@iY`gS}G6><O$r#jygq|V%G^J-~IH|2&ki4;SA48>t`>98qOsRA{oXcb4XKkP2dUP5IDc*8SbmA{*ufPXYYcNX(imAk)5YFAq*>*jp14TPMp_`h!I?(S*E=-J^>F1>%oj-EyT$0N_n8&!Jf^PtjEEEJDk!p)(dr^zUH{$NISGpf9r#H(A<tG)&((0UIoQ0Ua-zD#07bKo2yx0k<J!jTVzE!MMEqu3$NN_g@HQ|`Db1e=`I|)Zaj-;AHR@M7(jNmMEZ&m(<?Y7qht%}%Wtv@k(L-HAR|loVc&18>FM!)m<kn7W7g{q{F_AgriI)|RHM)L=Avz`IO<-oHj*dx3GQ&gyF<T;eW=WA+=<H62kLQI<av|&bB;tw6Yp%Iwl?G>(64u!woNv8E1TnY;V$`)Jj=1w)xfD+exOZt4^U{CH#UnVczXG~10*)PwPVK8=K)EQrWAnLZ)pMNF2UUxaL=;_s9wc+Rq<AZ3dS<Uof?krIG~)DRl5=V42OWqJ^h%MlwTX8U>P}mmjdcxw;NdZRNMo+^uYN6u6sBEC%Zai2K6*;VIUFlxyB!}oV;+b2GiO`2Eig76wN-I1&TuAcLb^CeXFd%%QgXR;3-rNDKd$Qs*lXKZ%^3rYrbz6wk|a3eghqA#csC5g?w^B*6%Fl++}<mhMFOI?XMu5qzkxWE^)v%UM}YhmBE|s#2U#!kDU(vMdHKD*Gq7sRGbGK6zKf$+vK2)NLh{ipPI>&Y#{#JkC0a+e!PIACGlyae=Jsk3)|rC$wR?K{7Mew&A|%wQy#x*HU1bTxJ%HMJ3K=lP_%!rHLl2^g2QsSwm)?@59g$UJqTD7Hja&~Tr+et9!n+!mqCjV)PGX#Gr&31-`T&7T>P{C9p9oFci4iUv8i%eI)Sh)X<|#$_>A8Q41kG%*(eaHL`}BwG3djFuwCH3;ro&Su@y)ei&q$i;M>K#B8#d!x>2k*26=t~3EVwhk_cC}FoFE-vmQhX_!Qx602DGBgz0stmJ56W;-kgawH;N@OA2S?#e9ia$gA61d`_EoMB-(krvzY$1s%BFYw)HJR4NKZGCdmzWUA-MLEgh*klJ=9xLDeKJDJGXd<0MV+~RAx#2(XPGB#d6#7>MS0-c7j<YIP8Mwhs|GXnnsk{RU{U2J}sJUVCg;s+@$#0f5h34X_D?g2h?6*{)Vn~|r%r)w1|-igTqvB4}Rx`foiX>POj+%~bUO(9^-YA8Im!N#{T*NXTR<vh@~t^N3kUAXLGzAte%UE{iu3}(7eP0-S7`Sc|sE%#ZK1zMd_49yP9t~jNBYXKx6E0>GS?LEnwKV|fHEQ1gVjjtQr&Q)?RuP2z(nzy>ZawYMS=eXp{!C?Gw0Po;G&qszkMz<jzd#=W$T=OjPPdp}Z_GVuHs~LKO4|Kd06}Q7&=&{6n*;Qo8YX<UXS?l(JSkd=Dd~$rh#Q#nXlgMcF3*mR2*FmTKI3wY@n>*3l6x;Xf(-aLt;+%m%(9t|n)`5=C`MLTj=w^0PnAi2AND?%XU=gMo7|VbvSPTP8UBY1NC<BkhQr5~j83mzG$HT;I8Hma%Gw_QN4X25Ko(8RCm7*Y@9ClkKn{l4pGby^)1>h#NKMbc|1Py^Ct)|hA|FbkU?N;E5aU}(-!qHHHWehHfed^u9RAkroOm#np*~vx7=}EPOHk;-r%4EsAo~-t%Q@;M7^xpo#?AZ)s*3Matc4JmZU}{G4*v}+eJ5;98dn8>ODkckCL`MF@dTe|=I6;036uHAhM`=n`t<N>aT)vJzw#;L#jH@~ml)STOH<+}D5ahL+K;tS#SEvZv!qAfYc!ZGUGUvZ?s%Add0Kho^Io(RJ`zR&5ML$nUuf)2;PJk_I2qTbH1S<Vji8PG}lm8M`G<~E;dPIq#NNh_q@6TO8&dA)Svt$Fou^k1!ptS1r8etm(#=zOAY_BIiJljE*Dspp6xxE(;(Fb#yZzEh4;`G#w!FTHCJ*U`KRx(=NW~K?JeE~M<%*X?gxvCbMg&{YwVRHv1x&O^)XF|^Pn6agzAJhOFPv5H4F)gDbA-dKAl{rZCofqZ|)y}gLtP(QnH<re611u2q0f^6(>n*6ua*22dT#co&#HxU^{Prrpe+G{MrOdBLtah#G|JwfQl48?)o9Px&9zGYqWc4_zwREMXAxcTGJrkj(7N8|I@-XMJeN~y2!1|eO4YI}V#Mb_Wp+%<Yn6OjHIRz^Gl^5kR?!KKg-VUV|hU0OSNkWh|`X1Hk4`Q~Qb3hUr82*}hXu4wSgEf22U{-Tg+Ro3w4Uu?T_QymmqI_Tfs7!XpL_<z#70gR?1cIKtb@N|L0maLNtM5h|`W10WyaV2DO^1R&RK?FZ9glK>^DE~mb{;=~Y_}%#tm$ze33&d2mCt;QLRTNieVc9N89DQCT~$63_1=yjBLIl04Dx?dd~lro(YHMoUxoao1V<r@UJMWGJO|qLWqKwnd3(P%PnQC0Wm2vR1(r=wg&0>sEKQPfaXA?C!Zz1EYMl`iFPL)-(Q!egTIwTYvw4bKW<ul;y><Tm!<1yemKp8yY;3`A2YJcAWx3QU3j58|)209F*8sW(Ve9k%+xEI45rb}hPq1T)RKY$?p~mm2U%G?tigwnykxEHw*^4cwX3#uD@dq@{W8v#eCwWGLO-NCy7VEYEhq5F=I~HHxXA)0_@&Qm?a+@z0%os~5cn71zl9xH>N2WxL!d`x$9Tg^ugQ;=SQ<b^kVa1YD<wO7u^UFBzl{<PrW_e-gycV!hh^h%s_KkL_B;-F^0z~Y~4J30+HAGhO1!{8PfNvhpACMJ{elErjOH7#-Fqrm=<m$EE2Z6=Go&kr~Rc)IeYnygy>VzF#H5g{D7%Kw7+URzykN20nw$C-Dz|>-;*78du4?;9Y?T;eH1;4@j!RQ&bmZ6}@lIx>Axm)MppAvzLKPh4$yyKW@kfSeXN}ToUnHhML|4>Vw*OJaP-Fj}n{xn(inP^_uXGK+l#tH0+)q!l<KwvhHI*|va)$m|LO4|vBF``rinXBpbdzFfXaLiUqXPG%`Laq!jj9F?(LoL}(3<jFpXHQMp1j0TXjk>E$sXs^A=Ck`I{^hr+t27n4fb7cmfci<SN5{Kiq(fI=Z9A9#s>~KV4+`%)19uD2p{e-JX!*La$e0f-<R<R3K~sipS$n^wu)s%*)g_eVl7QA*)JB=@_dkwaR-!bw!t2t}WnidRXz^;HEi!sumX=eq@Ri_N-{VeQCvX?zwBXuVdSQAA{H9}g_lEyC_W&6gF=4o@*b*=BW92_j;F}LvvUD)*k)+?A$zq*Hufwr0qsiRJ{kTJ&`>Nw~zxwtjuE#dH?$!VNzpUZNl$Y$SE28#2w*WUf;(=t98GRu`-e)&9;05eX-&50+rMkZTS`Og(B-6puBHD$zT^?v)CDG2rA^*v1Pw<6a-D<k~nh?$p&q2n)d#=9sfVCTkje#%Es|S{?xdoj$3n&!CWYFNR;!Tn4%8D1(XSHJb(XZndDVqOLC9VjyF8jp0C}Icy+;qBAJ#$pKgb!WsEkKHGeTWPvla*sJ4g`csN;ipl7f5@o`k-^$CE+K_=e!s+$L%<k^tQ6cixYHR>UBT%Gx69B=_p-b4^dbPueY8dcX@iM*;7BKuS$}jGedDD@HwaA@M>p12ApOuC)-^DiaIzIsIyq>?*5Fg;C?Wx5+IYaBnbgdb0GAqTlWdRwAumqEPCr(N+SG7!<-I3PdnzYconI~7bd)Z`ojS;zA9JHOo<cElq~z<;8d-In2V87c8h{#nhj1uQuBQL5pY4$Lw_6>>Q-nU?<x3-2Q39V?x?*9Dka9Gibtcn11W^o#QSdn?SR+-uS8xtw0+dDY0?Pl2$5{_O%(7Zimupt^d<Ap*|B8n@;>41YUU~9pV56-97Fq+tskg|3=1}DMtL8O<b-Ypa`SE61V^uJn#nk{1tYos)5u<MMyj1+*uTCDm{x@Qr^Rm@-44SXO+3}U$ORGLg{v2oV+Fq9Z^-j`4mbulASg!z3E8Kx_R%6eit_)*%`0irC-;*3)<pE#bn&^>2sQR`BG07Zqh?f+ui=g~)htT3+$1G_4bj`I-lc+PyQlKe&kG6bS-n)Q8^~M$>jrsm1QpNFbZYhmKp~Q0q>G1~Ra*2m7YKW-z6jUFi-0JM508Oy`kU+$h~%uma<t6*VP0G+7QCj!XAT|wh?V<Z?_*U-5M>FD^}s#?wrezBOrmVH2W9R8B9Cv5H5m{q>rl;q3AU?4GYSj;KBLRj`+77Y$b7=P3=b%V1&|?p>V9E;tB*?BYx^%8uiHiem-gvvO9=J(dx{TCKsUmys5mKCb{%mXN3ScXRmAd6WZknUH%nYS`i#Wlr<^VOU+~h7SojzyHNS@J_Yl*tM%}h;=UAjNt?_YrqJr2wSA;$@otBI2pwfSwCJsL;XYzW#x^8r?KE)?$aDlBR|BGr+xQ(H<u7Gl56ipRXtG!jMaom|?-5UcLkpF@W6W-{IdCBw;pl^cKBx_%{6QOQjE9F@3_pkQv{#SF6^LGqiHc6*ghvEqjF*UVI1)%!@C!GJ%5>+|`9zFA_!9-%aWmdHcuhHF87mm4aD^~fYNKyX`(5*(zGt!9gQd(C`qTQq3%+k<BB_t!+X|DvJ1u7{@Nt7B}!<uivdX9xto@E)gZe(J}9jnws?Sg*g6mI}sFvp(Ej00MshcFD@#gvb-dBsb+jlm6dE!yAukj(q^yS=G}ClM+Zw}WC@-g(4u6Ny`JOiid{=AU2AV-2u^2(pT~wG9#T<8{%kkWA8H<(2237;GK&vgF0bdiij2TMEmSsNxKLw4-S@wB%XYdrHuWcU2}KanK-=E%zY+P*!<9N}Fy&o4yNt+wv}u3%{Dpu#T;8*na}PqVoH^0fq7Y9_8GTQlLVGw&s^qm}UgYBqV#&d#W}E(ow*A6kYHzCu6+U7pIZN{h5?r)KKc2lJydpeTx!Z{VhP_QdgZ#gF3<do*&N|9;YC-!P?Bx?hpy!)BC0u)Vq!;7VnADaM7Ed@AebG409K6#N&^l4c`T4qFSTx)dhQt;aX82+~sngI61udq1zI_r@9i62Lsd6jcZIC$eY!SHBio9sx&rQpwB%s;CqbMiD>H^?TL+{WyDK*Gzv&c@Kyt!D7M@=mn|I;#vSDK-o4$rbAkFy$+}2yjqXFK95xkBqYyqXz1Hr!>ddB4VO3qsMVR#qBZf;(vc`9qDx@EQL}J<-{eb-0kCv729w`@5Ng=JB;-$PPRD69s{r0<?%=fF*lP~~#H;=x<c7)Wq@cWJ3pXu%p*K=@%#t~XVQwmvMoR%;Wtedhy%s?)ys!qDHY#AZfx6k2*L#oXdA}uWWp@)z~s6}-5=JkwW0(SWBIz5gedijYXOTw7<Fbb5p2XT1tV!F!h3Sm)4Akc~H);CB7R6&keykyAKa79c%rb^Q8kHK~R@HDVSt`)uqt9nCEndrsL+~4EjdNo|-Q5<|PgF9$9%9DvfKQ_vVm$TJr2Kpph72F&^HQtgMGPdOo^sV0~h$T9^x4*-6OGcb_|3Ogw4Oj)W|HP&AR6`8d3WR_GlTV+0ZYqkSyHPdUM73MwJSOV_p!k;HmCjX6<^6h|8o&BFCUbpA^snS1CAjovok8B10^puH)tsz(SVbrhWS%eL9RvrT#JURDNbv7)FbpD1GSwHL;?a`5ilBFpvphB$w_^I%bt^47;<%#~p|!c7FG*GD|I4W@h0r4TcOZo$a;hyuim1eK*PdvpwFM&ln*2lSj_O&Rk1g>kZ<qxf!{!b_`cBfwe;r3@U}k_lUycUX%M?fv3a_hFfh42a^R53#T2>(O-!=0(b3S7C!^@_bcwlPy-7BkdQ_7JgjYq-tP>v47tq*<qFp!z8E)RSEk3AJ5-tm%=#apR5UHtB;3l?Fsd>|xKJS#q!nZ(76(Czpn-0cqmNDw5pup|)zP(`+&e03Q@Ys>jr_2zFTKXv!X*+Fe$p(@A=+A!K~7dz)8-0$6OC8?3Fe`B(pLV6Q62+$^P<9FAKB4Ho~oEXiJtp+e2kCozWYB%wn>Q8Hv;|P}_^c086g{wJyFj$>5+@xTnL|p4{0U6tXi*<wB;Bs$DYjy|=7-!7IqC_RWAnxgh;9N|5#0!na27?3V?L6;&yV9_`&^w1$d-U;9uDNT;X?Bw5$jqM8*e<1Z>bh#u_B)xIC8>O0qPxXtkWbRX<@wB<43$OJIwhFGM_cTQ!7ojs2VLlATyzP7-t+Iza`+k<1r^{niUSfC;~L+2IWar#uzE(oGcaR6z!0-aOX<w0AX!&Gm%HzEsw>YK5}xrizX^0^ryi%ZAuo%kTuDnI;xzEw3Wwrvhmb!tIrb$-CMc!m*7_^+7CMky8tVU*krouOMiKnO4=4t_<<EJi<%sO4RhNO&UG@gxBxQL%an4hL*gTqTZ<GhMrq9oN1ZFMJ{vY3c+alk0Ia3ZKNlqIDa#NFYaEj}Om@@An6gc7%rM9c<X&(SvUj~$4*9=cBQ*+Ykc9xtLh6-2ItE?lH8gODXc2tg_;X%NJaNe^|-T}qi!t%{xnwL|eapzUR7&45mtPV`f$S761a>=W+bPdc^!69rDGgtl-@_PAc2N$cT$$O7gtoxaN>b;566>x;q1d+V5>O{NNmB`Xp$9m9esl-S<Y#1pGB!HVz!0W#c?3*2}b7oXyQT+7|Sl~hO8e?~oss2ddyRS6FzGn!Nn$x{`z(iq?ER8x<){OF2xqG@<bGG{ZmN2Et%7fyAKQK0+KwKj=-w4S}s1K{8;i!-~$J8&qx!nnj^qz|;fkhu~((j3KMBfRLvOR_QhkP;_$Tr6`T9=%8PF^^Z1LBLQrwcpO&>C;@Tw0jnEkd^uu=`<v{Sqk3Xr&G@DNjA!C5YBoWs8zpL<12&MRD9nvse!##b)KSP+pR)VPq?~bM<qYC)NP;3^EN@Z3#^khphi_d$TP4=W~~|4)yR_8WofI;vLX<0_>Y9kX}4csNBui)Fw`2Be0n<;@fR`g`glas|0iLrm*n+@9RcqhpN5Fp?_1b)dwt4MJCBp0?@i7AO0V^^fgm=cRlyBzu*n?fuVx%o?2aM@81X*ttkcvSCbF$XJfHGQEZWh_u(;+`q;$1LAN_!q@CKORK$HT$CsUGu?K2+WeX21g(5uvs!HrA^<DP~i?^jhvUN^5#jD{6N2oD*TNjL@#|=e|JLf_JBTNk_cT7XJG%sqWh&yHKa-i@1eUqXy<><WA4geBfAX~azmqRyEkwSLQu9|pvF493=w6m)=xs7xXcK(LBKY8&$GEOZulWu4vuj<Jhe!UB(o?-5k5S?roe_Bsm9N8Hm8b%|ifNHW;S3)9wPBM54X%AmB%y4oGP`sEAIO3*L_MqimGHg?lSc8|7|3&rU5w~{N^b7dB)<{gYWu1zXEE29J{{8ZZIjBmWzLfa?vP3_N8D-MwBQ6zYa%7$--J53Pw}+QAoJd5kfqy2<TpUfzl+eF>!1k^Dt2Ov0y_Nx31j@z@*2{Cm=o&!w088h>Lmx_gMhm5PLmHI9v{@6`BRO}?wn#nC{D1F+jXsIW8-R2d`{q-(vws<L6Mv~+2kBdnF1AdX5Ww!n`MuLhhk@}*g+xNFdSFn)YFet_^)j`i>{DQa-v3SwCt-v%2azFFX{DE^Bb($Yi=4hbUkA1`m=wA1^WUx2hl%BE9fh1523AKyvF#@ONY<Yy==0_GgmIACjAjDQ&Ga!GjNqNr$faw+-_9Gt^NJjl(qj4C9bd0tiI*1RqBr@@1Ia4kF63v6Jt2o1&2v~V)5{?~DZy5|I?e~5vR7Mv(=#{j?7(%;vQHJQSF=~u882D@H<RXS@B@%q5+PUwVe}2Sa7+qc9s*8n9kT%!h&M8k{9t(3Ka>s5TZ}UXxemoZ6ur`bYdRYcopFpQMHURfQ(kh^a_9o1s2zY~a3<5jkf4CBt`Q#W=?v$7AbHr%RuNYLx_7?k_C&8j-=$strDuJIa=@EihGoHMq}I-JzvyMaX-skb6okqb?}{C2``1;X=kl^se~o9zX}|hill`itD^II$0PH+HkH4F_rUhWjp@~z5#?o(jJcU%#Si#nQkOhkOho7~*yDLP+tX}GJspwkSRq-1VsI<DF9S?H5><-nv0O%F2&otia@>26k;qQpVcw{`qH3>lqxO0dzM%vUd*Yd*?REN&TWSBRDrsWeG`i{2|a+qWt>?EqUJ9_1(`60~Y<-=8al3OUUdtmWrQC*@}oppaUTB%<|IG#^HZ-x~nHy}$59?yCh%eilmn|>ys`N3}2#%jU+z|@9+>4ak-Fe(kIrSD;W3SIDUAzkBi9)*50yztD>n(|wRI2unWIjaM|QD9EvE%{oSt0q!y*U={=sp1_HWQF*7NfKQda=9V%*&`gp2dCb3H+HlnDd2!c<9SNDaQ@ZGUiBfc`<GNBjd5HA(b{60`n2$vGxLh7;==B#QcIJK>$-*|o_q58<<nbA7*4LF-mX<%oNzjJ@jtZN-^gcJ1IS}Mo9f`D`?(Mc7IjEoL}6?`o|4Lux{+AXu{^WAW*TJTQ^akv)*}M>Sth{wo~h_92`^C^kyRKCXt}*}rc=uWU@^BR{{6Fq2yX6g0t`h2POCKyP&hGZ>tS<A1d80E?+)i3t_<1%iz5slAQl&{#dV}2Xn+9Kl8zjL$gySSN%lBhJVklbcc&G~PVRT*|MM@ET;m|!2FT;>A%y!R6l>2e6ZpA3q**}?2oBsGAO=76|2wvQPj%47+i|bq^AJcPc6mQm7U!fOP%Cj;gSaU_QP>xBqG^c8Gj^z$jrFtgce6$#^jWwjIm}V|ia~4OD)fN}+Veo8Tzyf$_J6=Nr!vERC(f{D=5^lzMf1Sc|21X6BVQ!~-d8M9AUxoJhgz}8!<x?NlQ`qd=0p2oSC`&qZpi_l#<0(!;KiQ`2<y=d{-1sf9Y?Ng@!pxR4VHDPqlZ-a0+6{*&Dp7&s{4EoxP;2e7&!v?IcNPLq(xXqSb|ME^&%ud?#DFBa1Dotopu#^)@m?@F=caGM&R1=0iF^pap_qgXQuCMW-uEmFkvF>G%Gq|M*Hw0P$sQ4LIXTk2f)zdeL|2wJ_yAQBiOs~wI>k#TR5bV{m}Fy&%?o8t3Bg3`C%T#%k$AH>j%*#x6V$X#&~~m6;+F%VQhsM$)n-fmD-Q2k4EfOxCwfgAQs4;k7+%+w$Wwr{U94Y!Lh4n_Iv=?`<vgrsi5gkgw+m0f%!mhj`iJ#_=CQ1_6f;fO*;&v6rZDz#i0aqM7ea0;7a$A2LMhrdH>{q_j{?S_^J6<1A9jSB9w<vMF4>f@<N&}Zqlo9m-7m!ke7VLHG38wD9j3>>M8o-BZ&+lGRZ>0Hg{3U+xyO_+m~JR6w#0v$V1?jPAH0{ks8=4q;e_ei-oMUsUbognd3bR<6<6moXq8I#v)zJhz-4Hrqf$AQNgcRiP*jyB?qvkqBoF|eg`tKePR~vy>zy0iM^B0PQi_;q6P`qWJ0N<Ul`bzcj`uL48w|>+AXqJ1*$}jPt@YpyP|FdEx>%zyG#`$AIs9t_Qu3c2b2p!T*Q7fJ)*l=dv7zlYy|ms-h882|HHENdd$$Uoxl%Tw2I>o-=VPEyk&(&aB2`@c@NM=G#<Q>QZqF!F8A!sn~N3M%<!s^t5NS}VLr%h`ylDNH7@OeJ1hr#INLAl!~ltz(3+p@tJo{4trW3C7g2(F6mb!p`>}!U-)Xr_4g@pZ$nZ<gIU`l4qb*w6!Z&EhRI`X&^M)?uf_0SqdIxf#=-hqGpCT3*OT@1+?3&<g{6641p`Mk<@-S$Zu<<0>>f)7hG+2`0CRObo@z|xe=m=6fQ45zByR)uXbLA-+Q;TQDFG8wF5%No<*RM*^=R3z6Z6!-4r~am$mUgsU97*caO+sjYEVwqyqi}#7NVmv8i<_|Cu`wJNBO$ACQpIygk6rMurIn;J^bxi}=G=<e(NMdLa2a>U^)n4_k6(!kt6pW<=zCd1!mXt*CxqsVV>9*@3)@r88lOMj+jXn9IN-n?KeR{@9WGfPFbO(VZ*RQ!ug5Qp*C2i2T<<)~{klH5XA)woeR5%+1Brb5;342i)X(6wm8?y4jy+dlsFev<YTtkV>w<-s$h~w~PyNcV0>T2&HWEu}Nb*@NiRR@pSrzJB0{53E%QnBtTgMIpUatqX_SmKRnX}mb45p|%yXLCqW73nve-<afBbPacH=CE<a;sc2zHAa9Hmg`ogz%9`^LtDjl%V$0ltSVvil~CKeh%$$i`vtNN!w1}svH??BcFwSFjb<>peD(oT;$OOoeU9#8-`WZzaHiXIgQO!@??~>nvl%_T1wR1dxODzk%{(W!PE)p6sHxzV3ECR?+B%IQb~Ba7l<R!369xf@X4$B<V`iw*4<YXw-VWRDhW7fQ2{jEsm3a<9Qx3oPDD~1KtLtkWn<bU=Y9LO+9Q9+Xzx#^uD1IqBV+C53Sj?rrnQ9x8VxBOap5nYTsqd@u0q8vz&e({xqRl8rdHYgP=X;7Y!=htC21CCi%yk}Go8on7T6_vG1?2h9=@K>c@VxY6tr9aQj~&u+{(IROxdLHge@F9-YnV+GNlU1>i-XpwdmPIc(){|D*(tw^Q4Ohrc?qSG@f8Cx^nm@3QG14ip4hf%|X_RIcAXC6G|nf{Jv83-69a59zc(blb~v`C555e<-b7Cxv$Uv9z|V$C$v|vl;J3D^ap0++B@EE>Jbyq=?fK=EYU=;(bG0Hz(i|jtEhWeeP^3*=Op)w?i!&Z9;1Y!u$2nY<wQheF@1;q>VYkVKzkm|bs}REmj9e?Gt%L8SaRlyjx+MN8+Vr5a8)!CtbYvz%*SW^w$Q3&hUU7iBX=D9{GhThPC9;=r7zMCtxn-k1u+c??%X2T4+I7^D@6{B!iww%y)YHvn>ybG#bpg=uq}+#3bra}rtoQ~>x1YKKJm8{5O(;7Ha-O<aZXn(GD-k$4n3g5I~7|RF{Dc%+lAL&_5i1Yiu{iiXm8p}FkAu!hm=FH1Wn7Nyr?SHcf2X~1MyF)a7(CC0I;Xqt3HH9*jD>r=Iyy+W;4V1sR`PdNwW-srqsP0tYJ4wRT2w+i(XGaSM}&eKov1cy|n&b;w{&;JhKaWXRhB1?{103Waxj0^nc=s)586|Vy?bL<v`%N<0w{UBDhvtScnCABBSI!rn9k+&72Z0sq!Z+l=`0Uw6_bg%eSZzQrIZ`Smbb5UJ#R1C$^rxmwX~m06fgsg`La>@Mxazstm0)oM1Nlzh*oS;|opG3_$i>6d(=}A*$tEbovSj)W&;GS{RJ8w^7;|i?x5;7E0Lf=Q)~z${;EogV4ujyuV$`*BdAlyse?MDKd})+)0B9C1cST5ZiEJWh*KZvfu%gnI++++a@s_e@@_(rT~HtJpD(fO1yOZVUh~eAL}mQdsiS-xM7+E=4mRiS|a)QE~wZEFj%7P@=_U7W$rNoY<~}H1XwOHiB8Rh4qtqr?>1~NG+DW@Bm;Q|3*0t!l3buPp6r;HjeYZWXOcuRf77qbp%ea}+FXV>asq&e{~ixeIyG)Rg$<@K5O|V&0!EWp@H2}Aj1~UntH2kMjo0}3tU;xsd&rv-ZfxR{RG}Fw3?_yup*{ekfKrcUzRTPCpGht>x`o8HlIhRwvWEJrhROZc`ocdDCJ%w6rcsf9Jexk<(2NF%W(2-wb*EhrF17CT-0ezTZ(uBjZHx#`8_A+w7;^5pB^H4!4QV`|ttFe26OAe#j%~*G*<72PTcBO&dE7x%TLKztv6z3J;Z!8l3NaAAg^pNdTJ=LT@A4_6M@+qR>{$MBhDQka8h-{)@k2@A`8ApMM1<F;3_ee9a`VP6%Z<&v(UHxPL?#X1AQpi^1;Dp?@ld27qt!xsLer9PF0s%-Wr4v!8sDd3E(gb;N>m~2!2ivkJur)=<FbN~(l5tmwkx)(CP%ryJ+@Q#qcm5evsn?B`pf(=1I(p)gu8N7XzlT<Dg3Y_FYk6@PNwR^&7smL^ONHhMk`p3cN)-F_AO_g**~>h#lo9M&{q;8i{|^^BEA+@q0Ax))QHdy9Xf~1D*`A=;q*67r0<-XYl)%!l$pNtd2nlm&(JaD+2|%f^)x5(Vjr?!+|%ICE=8BD7h4S@q_9;Lxm)AP7`zut8-%=D*>ik4xPt1=S%>?ovdfdvfz*nMfiK%)s+DzEDu57eWG+N;fI8F0Z=tb}bqH@i9(;<D+Gtpl8+er|L%*Le8%~09XzTDWXj`683`ufDsWl-J^r;FF7uggU*ykoBQgoB`LC;gB_o;5Ji*XS=W3QWE<eYY4Baoq=sXqh0@#uc08^`aFzwOdMk3AWayb=uyinXLUS|pNd!&=<FtqR~KO2N~~X%?Ct@r6MRxp(QeQjqa4$T80s$r<i2f`u3M8No(mo9aS-qinQCI9sL6tR|H~&ZgZ~nl~+DA}vgLgL}kGU{loilR(wKiy6>@+e=Ew%4#2La`Qy?k6!b3j`8av!nRbfmgZdvOP(E-0dE=Fi!zi2m)Q{0IX>V{TAP@1%I0(*sB=@PFXYegJZ=Qy6+GL6^d%y3h-=HA&d>h4Sd>FSdk%>rac@9oVh}C>?)(Dq)0LjIt1dCUIbcs0eXlvl#t;i9#!rgeqqmTQ-rr9^(WbNFNT#w8)$|cBd+#V7#|y`Plu1QWs~=E!k#$|)<`Ty;HJ0rQucN96MgMHEwI`#<V>=0cJNyfAk0~_6KfA;?1gyyw>o+VUT}JN0tBmuFi8oK(omhzO(xfR<fz?Zv%J6J5!OeLQy<XRP*>joa7<;t7Ei6_VEKtcl>w!U+_aB#Lr&FL!pYM!^F`=?e^vBncqV`GZSdU}Z`>cXOQ!~XJ+DOBS$HY|HtGpOo`<G7Xu-oe%J7<g{!U~l!<Q6a2u#@5RwZMH~b(1=O8p?;)9)w-j(hwytZ*5qq5fG1|KE-L=rOnap>)WKr(>Tx`>JEH%%E=wUr^B=doES|9gNOr<DiG2<AmnT=l#|F-xTl;5ekP<D!-%~|8zpOqoM;EvYu5HAQyNZu7KtGtjM>A)nfR!kSyry-Irc1(=ob354G~>G$<Gsb7~ZWcN4V?gRw~R?Lw<#sW|aLEmnD{1_)HYKk#^S_vYr&qH#0uMb;tF|<rId@5yn7a=hyTn{d^w^`wU&}Rhlz04mYB<ak?*6Q!O+TwH0zS#{1#wQ85ReD(r)ht}Q45k;oNt)*bET1_0f&Wa&tFu*rg6X<eJ2icyLI#Cp{N?0CF2ZW}h<mELEG2dF1Ga=-kT#dbVqvCx^KEPIl+=nu|T^%wAjj;t=DAi$Xy(pF_<H$+|E^KpU{%6vKCSxP0%P65c6Qh@$qE<M{0yWOo7DgNz+6S^=?S|9j_+p3ZAGdP6iXE4a`dAHC(ZvE6%VYJxBwx|3}(_AC7nT0^v^uNF=O8pLCMZw0;#C>XRkPbr=g4gRl@kD!Idb-Pu8KGzY?K0DM-F^9FL#P9`l<{n<VywzfgRcbrlL&2Z#s(*z)KtzKLUk2-i$bu3P%Vxjot<>e8p=r-XUpv@Pm?gFVFpq5RCz)BcbcNkzG3uIRus8^{50Ujj|i``)scSxHC9eyl?OaR2ps}LL8TBZ?mn&=W)HD1HTG0=rpssq77*&>=<1$)A}v%67D%GuW@E`=;0Q=W3oNNnjW1z;*yv<THdV*<n-$%plNM;o&UqJ;x+nbX;KZkl9eU=MeRR~8ogxfZoB6gTyVO>l7d@OGb_X*!wqg6bn=&Pt_yMuPOmePNb97vNh1t$(9SZ8&{C_7yZl~5d;v@{z!^+MLCdBh3)O#ihsVV2bsz5HO(sx7ABZYSRa{HRviM23hUhAF*74Adpn11brssB`^L2v<}w#8s$F$49qN#(QW^JD9uhzAwYJm%HKwZqep7Q_A_4xgv{bgaL6S6Z=Zdj4x#lcRtm1QF?hePxb)a2DYkk&2A6p+US<B1v$acR@DyfNEgkC~~OuEHylh*>x?{LJjB1#IE!wbsW%P;~j!Q?8G93+hcj5D41;^@;U}|Aao#`H;>UKkLErvfi)yuVpkNBygO$J^a5_AFdhkLSl%z|eXB%C{ME8!zEhGbqj0A-2lcY-GQ=$+``xSIgPLoIc!&%Ri7O<=ZXG(Q;}YpH86F?XoTiF2I5pN~7GbT+tBTLxLo6nVSz#<!HMc!hgI(~dv4^fy5g*|K{X2nDUP=XyjOPnOzNH|B6;1mx9nU|f5EfpS)WBKwiRT#ni^Jv!IH&?XlyPuO(wPeTZWC*3U`DB%hll;!8|xf;cv^yFreSvWjPFJ|!B>1CQy-$DJ<M~f;}j^r6j8q>)Uv7C2En;_sr>I;16GvmTPRMyd=w$ILpQq;sFte018F0^7mRR=8_lN$=Bdh{0iFrRG@yy|)PXOoI_0s?OswP80#aUPbqD2^Rz)j=h*GwJa3}TKPD}G_^!a_P-Me+==?_cAT%LuvjpcQslx6T>H-y8N4($j{v1PS(r3!JQx26nke!atu#EOm&!k&JnjDmEC$zy7KMCW^FMI_FEUhZ|G+X3}ccmy?~A;D(Nm2@pm#~FcOM9T_^as%Uyw(x!53phVx+TDmtL$5%*IjlH%R=jS~$k7cHN1bJv5R)eGA$@ZU#BUW$gO2=BXp2r*J2`W;uJgluJ(xYN^74Tf-SBI)i2+#WTv{O;R!sij%kRSu$DMap8M#=mSXzMwBVq)~Q1u`}>Cz%$QuKUlQ7;C74;a-eTMT;^2tlAu%;<czYjs0V|3#c1ob=eygxybb&8^28j2fQYI7Su|7@|@N3d9-aqZn29D5D|q{5!~r)F)7k=)Rf7Jn7&x*TJsPwJ2Wf990h^4*DnUj?9thpoKYJW!6J|e5Z`e;v0qcR44lT=IqdCE{Vj8VE{_L;)(CYFq!l7S6K|EA6@NBvpnv29|AnyJ7JrmUudZ!3G#D{WtGk9qAg8{Ft*}zhXT3GbgdB3L_|&GwJGfa){7*=v^t_ypf^9LBPkb!gXmP-ncEZEkD-G7R|ok#(a~HDW7-edxlPk=j$v|lZv)E@SJIhk@)|#13VzX8sjts^{sBI!Xq;-{c!dxdJCD*cGIN-4ITa$X7vy<=;~`9O25Q;DnrokYemb9+S{ew<4buh_3eO%KpuLA)=gy3_G?U&;{{&9Ymi5?}zE6W{LF^$ALfTkwX!k>+yY+&Jv>N<s?k+z5Cdrd~7|s#?jPfU`b@!F=pP;1KN`C@@9Gr_}_+u&>7=IwwV33}iUV#Q{4C9YX+oA^(mQbe2cA4vPHV@oPxBuAFUHJg(<RZ#5fxs*EO2WV??%L}P2&jNSIW+za9fW$aY=={BH^JMVjXDY;#9P9+(j)gK7Aw58M0aVzFnxXb0WJ4>Ks}Y$7Z7kkP%zf-DE~-Ks~BBU-<FT;9yrOSI4<#;?oH}yph}r|enGr+PQBA1y79sKu%0*KsQ7rQ;*X>!vQ^%=Vi!`_yE@KV4ocFh8PWT@p?ugD$Blmuh>}02y4DMgzWgJGu9-Q?SY>Pi@5{7$Igu}K<^TWybMz1Xs*?B$00E}f2!f&uHB3V$vBYQl0ssI200dcD\')).decode()\nassert hashlib.sha256(_s14_body.encode()).hexdigest() == \'e08e0999b8ceddbce104e921cc10a8291590384e32c1b3a1002eebf37b12451a\'\n_s14_file = "<s14ab-body-e08e0999b8ce>"\nlinecache.cache[_s14_file] = (len(_s14_body), None, _s14_body.splitlines(True), _s14_file)\nexec(compile(_s14_body, _s14_file, "exec"), globals())\n_s14_lab = lab\ndef lab(deadline, weights, state):\n    return _s14_lab(deadline, weights, state)\n')]
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
