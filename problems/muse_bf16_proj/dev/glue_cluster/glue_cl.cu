// Cluster glue for the decode layer (P7): one thread-block cluster of C CTAs per token. Each CTA reads
// only its 1/C slice of the row (S6's Triton glue has every CTA read the whole row, so at S split
// partials one SM's L2 bandwidth bounds it), and the row's norm partials meet through distributed
// shared memory: each CTA pushes its partial sums into every CTA of the cluster, one cluster barrier,
// then every CTA finishes its slice from the same C partials in the same order.
//
// boundary_cl: stream' = alpha * stream + beta * rmsnorm(branch [* postw]); out = rmsnorm(stream') * prew.
//   ROUNDS 2: exact two-pass form (|stream'|^2 summed after the branch norm is known).
//   ROUNDS 1: one barrier; |stream'|^2 = sum (a s)^2 + 2 r sum (a s)(b br) + r^2 sum (b br)^2, r = branch inv.
// route_cl: x = rmsnorm(pooler) * prew (BF16); logits = sum of router hi + lo rows; sigmoid(logit/sqrt(32)),
//   top-8 of sigmoid + bias (ties to the lower expert), weights = picked sigmoids / their sum, local ids.
// Both release dependents at entry, load their weight vectors, then griddepcontrol.wait before any
// read of the producer's output; launched with programmatic stream serialization.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdexcept>

namespace cg = cooperative_groups;

namespace glue_cl {

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

__device__ __forceinline__ float4 ld4(const void* base, int f32, long long off) {
  if (f32) return *reinterpret_cast<const float4*>(static_cast<const float*>(base) + off);
  const uint2 v = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(base) + off);
  return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                     __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
}

// Sum S split partials at one float4 position, all loads in flight (groups of 8).
__device__ __forceinline__ float4 sum_splits(const void* P, int f32, int S, long long ps, long long off) {
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s0 = 0; s0 < S; s0 += 8) {
    float4 z[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) z[u] = (s0 + u < S) ? ld4(P, f32, (s0 + u) * ps + off) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      a.x += z[u].x; a.y += z[u].y; a.z += z[u].z; a.w += z[u].w;
    }
  }
  return a;
}

template <int K, int NT>
__device__ __forceinline__ void block_sum(float (&v)[K], float* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int k = 0; k < K; ++k)
#pragma unroll
    for (int d = 16; d; d >>= 1) v[k] += __shfl_xor_sync(0xffffffffu, v[k], d);
  if (lane == 0)
#pragma unroll
    for (int k = 0; k < K; ++k) red[warp * K + k] = v[k];
  __syncthreads();
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NT / 32; ++w) s += red[w * K + k];
    v[k] = s;
  }
}

// Push this CTA's K partials into slot [rank] of every CTA's `slots` [C][K], then one cluster barrier;
// returns the C partials summed in rank order (identical in every CTA).
template <int K, int C>
__device__ __forceinline__ void cluster_sum(cg::cluster_group& cl, float (&v)[K], float* slots) {
  const int rank = (int)cl.block_rank();
  if (threadIdx.x < C) {
    float* dst = cl.map_shared_rank(slots, (int)threadIdx.x);
#pragma unroll
    for (int k = 0; k < K; ++k) dst[rank * K + k] = v[k];
  }
  cl.sync();
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float s = 0.f;
#pragma unroll
    for (int r = 0; r < C; ++r) s += slots[r * K + k];
    v[k] = s;
  }
}

constexpr int HID = 8192;

// grid (C, T), cluster (C, 1, 1), NT threads; each thread owns HID / (C * NT) consecutive float4 groups.
template <int C, int NT, int ROUNDS>
__global__ void __launch_bounds__(NT) boundary_cl(const void* __restrict__ P, int pf32, int S, long long ps, long long rs,
                                                  int NB, const float* __restrict__ stream, float* __restrict__ sout,
                                                  __nv_bfloat16* __restrict__ out, const float* __restrict__ postw,
                                                  const float* __restrict__ alpha, const float* __restrict__ beta,
                                                  const float* __restrict__ prew) {
  constexpr int G = HID / (4 * C * NT);  // float4 groups per thread
  static_assert(G >= 1 && HID % (4 * C * NT) == 0, "slice");
  __shared__ float red[(NT / 32) * 4];
  __shared__ float slots[2][C * 4];
  pdl_release();
  cg::cluster_group cl = cg::this_cluster();
  const int rank = (int)cl.block_rank(), t = blockIdx.y, tid = threadIdx.x;
  float4 a[G], b[G], w[G], q[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const int j = (rank * NT * G + g * NT + tid) * 4;
    a[g] = *reinterpret_cast<const float4*>(alpha + j);
    b[g] = *reinterpret_cast<const float4*>(beta + j);
    w[g] = *reinterpret_cast<const float4*>(prew + j);
    q[g] = postw ? *reinterpret_cast<const float4*>(postw + j) : make_float4(1.f, 1.f, 1.f, 1.f);
  }
  pdl_wait();
  float4 s[G], br[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const int j = (rank * NT * G + g * NT + tid) * 4;
    s[g] = *reinterpret_cast<const float4*>(stream + (long long)t * HID + j);
    br[g] = j < NB ? sum_splits(P, pf32, S, ps, (long long)t * rs + j) : make_float4(0.f, 0.f, 0.f, 0.f);
    br[g].x *= q[g].x; br[g].y *= q[g].y; br[g].z *= q[g].z; br[g].w *= q[g].w;
  }
  float inv_br, inv;
  if (ROUNDS == 1) {
    float v[4] = {0.f, 0.f, 0.f, 0.f};  // sum br^2, (a s)^2, (a s)(b br), (b br)^2
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float bb[4] = {br[g].x, br[g].y, br[g].z, br[g].w}, ss[4] = {s[g].x, s[g].y, s[g].z, s[g].w};
      const float aa[4] = {a[g].x, a[g].y, a[g].z, a[g].w}, be[4] = {b[g].x, b[g].y, b[g].z, b[g].w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float x = aa[k] * ss[k], y = be[k] * bb[k];
        v[0] += bb[k] * bb[k]; v[1] += x * x; v[2] += x * y; v[3] += y * y;
      }
    }
    block_sum<4, NT>(v, red);
    cluster_sum<4, C>(cl, v, slots[0]);
    inv_br = rsqrtf(v[0] / HID + 1e-8f);
    inv = rsqrtf((v[1] + 2.f * inv_br * v[2] + inv_br * inv_br * v[3]) / HID + 1e-5f);
#pragma unroll
    for (int g = 0; g < G; ++g) {
      s[g].x = a[g].x * s[g].x + b[g].x * (br[g].x * inv_br); s[g].y = a[g].y * s[g].y + b[g].y * (br[g].y * inv_br);
      s[g].z = a[g].z * s[g].z + b[g].z * (br[g].z * inv_br); s[g].w = a[g].w * s[g].w + b[g].w * (br[g].w * inv_br);
    }
  } else {
    float v[1] = {0.f};
#pragma unroll
    for (int g = 0; g < G; ++g) v[0] += br[g].x * br[g].x + br[g].y * br[g].y + br[g].z * br[g].z + br[g].w * br[g].w;
    block_sum<1, NT>(v, red);
    cluster_sum<1, C>(cl, v, slots[0]);
    inv_br = rsqrtf(v[0] / HID + 1e-8f);
    float u[1] = {0.f};
#pragma unroll
    for (int g = 0; g < G; ++g) {
      s[g].x = a[g].x * s[g].x + b[g].x * (br[g].x * inv_br); s[g].y = a[g].y * s[g].y + b[g].y * (br[g].y * inv_br);
      s[g].z = a[g].z * s[g].z + b[g].z * (br[g].z * inv_br); s[g].w = a[g].w * s[g].w + b[g].w * (br[g].w * inv_br);
      u[0] += s[g].x * s[g].x + s[g].y * s[g].y + s[g].z * s[g].z + s[g].w * s[g].w;
    }
    __syncthreads();  // red reuse
    block_sum<1, NT>(u, red);
    cluster_sum<1, C>(cl, u, slots[1]);
    inv = rsqrtf(u[0] / HID + 1e-5f);
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const int j = (rank * NT * G + g * NT + tid) * 4;
    *reinterpret_cast<float4*>(sout + (long long)t * HID + j) = s[g];
    __nv_bfloat162 o0 = __floats2bfloat162_rn(s[g].x * inv * w[g].x, s[g].y * inv * w[g].y);
    __nv_bfloat162 o1 = __floats2bfloat162_rn(s[g].z * inv * w[g].z, s[g].w * inv * w[g].w);
    uint2 ov = make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));
    *reinterpret_cast<uint2*>(out + (long long)t * HID + j) = ov;
  }
}

// MoERouteMeta v1 (problems/muse_decode_layer/dev/nt1_branches/route_meta.h): per-call routing description
// for the small-M MoE (tokens <= 32), written by route_cl's last-arriving cluster.
struct alignas(128) MoERouteMeta {
  int32_t version, tokens, local_start, pairs;
  int32_t count[32];
  int32_t offset[33];
  int32_t blocks8[32];
  int32_t units13[33], units2[33];
  int32_t token_local[32];
  int32_t pair_of[256];  // sorted local position -> token*8+slot, -1 unused
  int32_t pos_of[256];   // token*8+slot -> sorted local position, -1 remote
};
static_assert(sizeof(MoERouteMeta) == 2944, "MoERouteMeta v1");

// nt1_branches/route_meta.cu.inc's make_route_meta body for one 256-thread CTA, reading local ids (-1 remote).
__device__ __forceinline__ void write_route_meta(const int* __restrict__ lids, int M, int start, MoERouteMeta* dst) {
  __shared__ int n[32], wc[8][32], off[32];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  if (t < 32) n[t] = 0;
  dst->pair_of[t] = -1;
  dst->pos_of[t] = -1;
  if (t < 32) dst->token_local[t] = 0;
  __syncthreads();
  const int e = t < M * 8 ? __ldcg(lids + t) : -1;
  if (e >= 0) atomicAdd(n + e, 1);
  __syncthreads();
  if (warp == 0) {
    const int count = n[lane], nb = (count + 7) / 8;
    int p = count, u1 = nb * 512, u2 = nb * 256;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(~0u, p, d), b = __shfl_up_sync(~0u, u1, d), c = __shfl_up_sync(~0u, u2, d);
      if (lane >= d) { p += a; u1 += b; u2 += c; }
    }
    off[lane] = p - count;
    dst->count[lane] = count; dst->offset[lane] = p - count; dst->blocks8[lane] = nb;
    dst->units13[lane + 1] = u1; dst->units2[lane + 1] = u2;
    if (lane == 31) { dst->pairs = p; dst->offset[32] = p; }
    if (lane == 0) { dst->version = 1; dst->tokens = M; dst->local_start = start; dst->units13[0] = dst->units2[0] = 0; }
  }
  wc[warp][lane] = 0;
  __syncthreads();
  const unsigned mk = __match_any_sync(~0u, e);
  const int rnk = __popc(mk & ((1u << lane) - 1));
  if (e >= 0 && rnk == 0) wc[warp][e] = __popc(mk);
  __syncthreads();
  if (t < 32) {
    int sum = off[t];
    for (int w = 0; w < 8; ++w) { const int v = wc[w][t]; wc[w][t] = sum; sum += v; }
  }
  __syncthreads();
  if (e >= 0) { const int pos = wc[warp][e] + rnk; dst->pair_of[pos] = t; dst->pos_of[t] = pos; }
  if (t < M) {
    int cnt = 0;
#pragma unroll
    for (int q = 0; q < 8; ++q) cnt += __ldcg(lids + t * 8 + q) >= 0;
    dst->token_local[t] = cnt;
  }
}

constexpr int DP = 4096;   // pooler width
constexpr int NE = 256;    // experts

// Order-preserving integer image of a float (S6's key: larger float -> larger key; never 0 for a non-NaN).
__device__ __forceinline__ uint32_t route_ord(float biased) {
  const uint32_t bits = __float_as_uint(biased);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// grid (C, T), cluster (C, 1, 1), NT threads. Pooler: HID-style float4 slices of DP / C columns.
// Logits: CTA rank owns experts [rank * NE / C, ...); its sigmoids and keys go to rank 0's smem.
template <int C, int NT>
__global__ void __launch_bounds__(NT) route_cl(const void* __restrict__ P, int pf32, int S, long long ps, long long rs,
                                               const float* __restrict__ bias, const float* __restrict__ prew,
                                               const int* __restrict__ lstart, __nv_bfloat16* __restrict__ X,
                                               int* __restrict__ ids, float* __restrict__ wts, int* __restrict__ lids,
                                               MoERouteMeta* __restrict__ meta, unsigned* __restrict__ arrive) {
  constexpr int GP = DP / (4 * C * NT);   // pooler float4 groups per thread (0: some threads idle)
  constexpr int EP = NE / C;              // experts per CTA
  static_assert(DP % (4 * C) == 0 && NE % C == 0 && EP <= NT, "slice");
  constexpr int QT = DP / (4 * C);        // pooler float4 groups per CTA
  __shared__ float red[(NT / 32)];
  __shared__ float slots[C];
  __shared__ float score_all[NE];
  __shared__ uint32_t ord_all[NE];
  pdl_release();
  cg::cluster_group cl = cg::this_cluster();
  const int rank = (int)cl.block_rank(), t = blockIdx.y, tid = threadIdx.x;
  constexpr int GQ = (QT + NT - 1) / NT;
  float4 w[GQ];
#pragma unroll
  for (int g = 0; g < GQ; ++g) {
    const int q = g * NT + tid;
    w[g] = q < QT ? *reinterpret_cast<const float4*>(prew + (rank * QT + q) * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  const int e = rank * EP + tid;
  const float bs = tid < EP ? bias[e] : 0.f;
  const int start = *lstart;
  pdl_wait();
  float4 p[GQ];
  float v[1] = {0.f};
#pragma unroll
  for (int g = 0; g < GQ; ++g) {
    const int q = g * NT + tid;
    p[g] = q < QT ? sum_splits(P, pf32, S, ps, (long long)t * rs + (rank * QT + q) * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    v[0] += p[g].x * p[g].x + p[g].y * p[g].y + p[g].z * p[g].z + p[g].w * p[g].w;
  }
  if (tid < EP) {  // logit: per split hi + lo, accumulated over splits (S6 order)
    const long long base = (long long)t * rs + DP + e;
    float logit = 0.f;
    for (int s0 = 0; s0 < S; s0 += 8) {
      float hi[8], lo[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        hi[u] = lo[u] = 0.f;
        if (s0 + u < S) {
          const long long o = (s0 + u) * ps + base;
          if (pf32) { hi[u] = static_cast<const float*>(P)[o]; lo[u] = static_cast<const float*>(P)[o + NE]; }
          else { hi[u] = __bfloat162float(static_cast<const __nv_bfloat16*>(P)[o]); lo[u] = __bfloat162float(static_cast<const __nv_bfloat16*>(P)[o + NE]); }
        }
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) logit += hi[u] + lo[u];
    }
    const float sc = 1.f / (1.f + __expf(-logit * 0.1767766952966369f));
    float* rs_ = cl.map_shared_rank(score_all, 0);
    uint32_t* ro = cl.map_shared_rank(ord_all, 0);
    rs_[e] = sc;
    ro[e] = route_ord(sc + bs);
  }
  block_sum<1, NT>(v, red);
  cluster_sum<1, C>(cl, v, slots);   // its barrier also publishes the scores and keys to rank 0
  const float inv = rsqrtf(v[0] / DP + 1e-5f);
#pragma unroll
  for (int g = 0; g < GQ; ++g) {
    const int q = g * NT + tid;
    if (q < QT) {
      __nv_bfloat162 o0 = __floats2bfloat162_rn(p[g].x * inv * w[g].x, p[g].y * inv * w[g].y);
      __nv_bfloat162 o1 = __floats2bfloat162_rn(p[g].z * inv * w[g].z, p[g].w * inv * w[g].w);
      *reinterpret_cast<uint2*>(X + (long long)t * DP + (rank * QT + q) * 4) =
          make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));
    }
  }
  if (rank == 0 && tid < 32) {  // top-8 over 256: 8 per lane; per round redux max of the key, ties to min id
    const int lane = tid;
    uint32_t k[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) k[i] = ord_all[lane * 8 + i];
    int pick[8];
    float sc[8], tot = 0.f;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      uint32_t bo = k[0];
      int bi = 0;
#pragma unroll
      for (int i = 1; i < 8; ++i)
        if (k[i] > bo) { bo = k[i]; bi = i; }
      const uint32_t top = __reduce_max_sync(0xffffffffu, bo);
      const int ex = (int)__reduce_min_sync(0xffffffffu, bo == top ? (uint32_t)(lane * 8 + bi) : 0xffffffffu);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (lane * 8 + i == ex) k[i] = 0u;
      pick[r] = ex;
      sc[r] = score_all[ex];
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) tot += sc[r];
    if (lane < 8) {
      int ex = pick[0];
      float s = sc[0];
#pragma unroll
      for (int r = 1; r < 8; ++r) if (lane == r) { ex = pick[r]; s = sc[r]; }
      ids[t * 8 + lane] = ex;
      wts[t * 8 + lane] = s / (tot + 1e-15f);
      const int local = (ex - start + 256) % 256;
      lids[t * 8 + lane] = local < 32 ? local : -1;
    }
  }
  if (meta != nullptr && rank == 0) {  // last-arriving token's rank-0 CTA writes MoERouteMeta (no CTA waits)
    __shared__ int last;
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      last = atomicAdd(arrive, 1u) == (unsigned)(gridDim.y - 1);
      if (last) __threadfence();
    }
    __syncthreads();
    if (last) {
      write_route_meta(lids, (int)gridDim.y, start, meta);
      if (tid == 0) *arrive = 0u;
    }
  }
}

template <typename K>
static void set_attrs(K kernel) {
  static bool done = false;
  if (!done) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    done = true;
  }
}

template <typename K, typename... A>
static cudaError_t launch(K kernel, int C, int NT, int T, cudaStream_t s, A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C, T, 1);
  cfg.blockDim = dim3(NT, 1, 1);
  cfg.stream = s;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 2;
  return cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace glue_cl

#define GLUE_B(C_, NT_, R_)                                                                                   \
  if (C == C_ && NT == NT_ && rounds == R_) {                                                                \
    auto k = glue_cl::boundary_cl<C_, NT_, R_>;                                                              \
    glue_cl::set_attrs(k);                                                                                   \
    e = glue_cl::launch(k, C, NT, (int)T, s, (const void*)P, (int)pf32, (int)S, (long long)ps, (long long)rs, \
                        (int)NB, (const float*)stream, (float*)sout, (__nv_bfloat16*)out, (const float*)postw,   \
                        (const float*)alpha, (const float*)beta, (const float*)prew);                          \
  }

void boundary_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t NB, int64_t stream, int64_t sout,
                 int64_t out, int64_t postw, int64_t alpha, int64_t beta, int64_t prew, int64_t T, int64_t C, int64_t NT,
                 int64_t rounds, int64_t stream_handle) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_handle);
  cudaError_t e = cudaErrorInvalidValue;
  GLUE_B(16, 128, 1) GLUE_B(16, 128, 2) GLUE_B(8, 256, 1) GLUE_B(8, 256, 2) GLUE_B(8, 128, 1) GLUE_B(8, 128, 2)
  GLUE_B(4, 256, 1) GLUE_B(4, 256, 2) GLUE_B(4, 512, 1) GLUE_B(4, 512, 2) GLUE_B(2, 512, 1) GLUE_B(2, 512, 2)
  GLUE_B(2, 256, 1) GLUE_B(2, 256, 2)
  if (e != cudaSuccess) {
    cudaGetLastError();  // a rejected launch (e.g. cluster size) must not poison later launches
    throw std::runtime_error(cudaGetErrorString(e));
  }
}

#define GLUE_R(C_, NT_)                                                                                        \
  if (C == C_ && NT == NT_) {                                                                                 \
    auto k = glue_cl::route_cl<C_, NT_>;                                                                      \
    glue_cl::set_attrs(k);                                                                                    \
    e = glue_cl::launch(k, C, NT, (int)T, s, (const void*)P, (int)pf32, (int)S, (long long)ps, (long long)rs, \
                        (const float*)bias, (const float*)prew, (const int*)lstart, (__nv_bfloat16*)X, (int*)ids, \
                        (float*)wts, (int*)lids, (glue_cl::MoERouteMeta*)meta, (unsigned*)arrive);             \
  }

void route_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t bias, int64_t prew, int64_t lstart,
              int64_t X, int64_t ids, int64_t wts, int64_t lids, int64_t meta, int64_t arrive, int64_t T, int64_t C,
              int64_t NT, int64_t stream_handle) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_handle);
  cudaError_t e = cudaErrorInvalidValue;
  if (meta && (NT != 256 || T > 32)) throw std::runtime_error("route_cl metadata: NT 256 and T <= 32");
  GLUE_R(16, 128) GLUE_R(8, 128) GLUE_R(8, 256) GLUE_R(4, 256) GLUE_R(2, 256)
  if (e != cudaSuccess) {
    cudaGetLastError();  // a rejected launch (e.g. cluster size) must not poison later launches
    throw std::runtime_error(cudaGetErrorString(e));
  }
}
