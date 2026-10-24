#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Standalone ABI: no PyTorch headers, no architecture-specific host branch.
// The exact cp.async + mma.sync path also executes on the local sm_120.
namespace stream_proj {
using B = __nv_bfloat16;
__device__ __forceinline__ uint4 load16(const B* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void cp16(B* dst, const B* src, bool valid) {
  unsigned a = (unsigned)__cvta_generic_to_shared(dst);
  int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :: "r"(a), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mma(float* d, unsigned a0, unsigned a1,
                                   unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void output(void* y, size_t i, float v, bool fp32) {
  if (fp32) ((float*)y)[i] = v;
  else ((B*)y)[i] = __float2bfloat16_rn(v);
}

// No spinning, no extra launch: every split publishes its own disjoint tile;
// the last CTA reduces in FP32. The private counter returns to zero before exit.
// scratch/counters belong to this call site and cannot be shared concurrently.
template<int R, int T, int THREADS>
__device__ __forceinline__ void finish(float* partial, unsigned* counters, void* y,
    int M, int N, int splits, int nt, int mt, bool fp32) {
  __shared__ int last;
  const int tile = blockIdx.z * nt + blockIdx.x;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(counters + tile, 1u) == (unsigned)(splits - 1);
  __syncthreads();
  if (last) {
    for (int e = threadIdx.x; e < R * T; e += THREADS) {
      const int r = e % R, t = e / R;
      if (blockIdx.x * R + r < N && blockIdx.z * T + t < M) {
        float sum = 0.f;
        for (int s = 0; s < splits; ++s)
          sum += ((volatile float*)partial)[((size_t)tile * splits + s) * R * T + e];
        output(y, (size_t)(blockIdx.z * T + t) * N + blockIdx.x * R + r, sum, fp32);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicExch(counters + tile, 0u);
  }
}

// A single lane-preserving shuffle creates a true dependency on every
// outstanding weight vector. The returned FP32 value is exactly +0, but
// ptxas cannot sink the vector loads to the first FMA that uses each one.
// Unlike a bare barrier, this survives instruction scheduling in SASS.
template<int U, int ROWS>
__device__ __forceinline__ float ring_dependency(uint4 (&w)[ROWS][U]) {
  unsigned hash = 0;
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int u = 0; u < U; ++u) hash ^= w[r][u].x;
  unsigned same;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 31, -1;"
      : "=r"(same) : "r"(hash), "r"(threadIdx.x & 31));
  return __uint_as_float(same ^ hash);
}

template<int U, int WARPS, int ROWS>
__global__ __launch_bounds__(WARPS * 32)
void scalar(const B* __restrict__ w, const B* __restrict__ x, void* y,
            float* partial, unsigned* counters, int M, int N, int K, int splits, bool fp32) {
  constexpr int R = WARPS * ROWS;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * R + warp * ROWS;
  const int ks = K / splits, begin = blockIdx.y * ks, end = begin + ks;
  const int tok = blockIdx.z;
  float a[ROWS][8] = {};
  for (int base = begin; base < end; base += U * 256) {
    // Issue the entire register ring before consuming it. U=8 is 4 KiB of
    // outstanding weight requests per warp, before adding another resident CTA.
    uint4 xv[U], wv[ROWS][U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int k = base + u * 256 + lane * 8;
      xv[u] = k < end ? *reinterpret_cast<const uint4*>(x + (size_t)tok * K + k) : uint4{};
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
        wv[r][u] = k < end && row + r < N ? load16(w + (size_t)(row + r) * K + k) : uint4{};
    }
    const float dependency = ring_dependency(wv);
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j) a[r][j] += dependency;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const B* xx = reinterpret_cast<const B*>(&xv[u]);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const B* ww = reinterpret_cast<const B*>(&wv[r][u]);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          a[r][j] = fmaf(__bfloat162float(ww[j]), __bfloat162float(xx[j]), a[r][j]);
      }
    }
  }
  const int nt = (N + R - 1) / R, tile = tok * nt + blockIdx.x;
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) v += a[r][j];
#pragma unroll
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    if (lane == 0 && row + r < N) {
      if (splits == 1) output(y, (size_t)tok * N + row + r, v, fp32);
      else partial[((size_t)tile * splits + blockIdx.y) * R + warp * ROWS + r] = v;
    }
  }
  if (splits > 1) finish<R, 1, WARPS * 32>(partial, counters, y, M, N, splits, nt, M, fp32);
}

template<int R, int T, int BK, int STAGES, int WARPS>
__global__ __launch_bounds__(WARPS * 32)
void tensor(const B* __restrict__ w, const B* __restrict__ x, void* y,
            float* partial, unsigned* counters, int M, int N, int K, int splits, bool fp32) {
  constexpr int STRIDE = BK + 8, STAGE = (R + T) * STRIDE;
  constexpr int CELLS = (R / 16) * (T / 8) / WARPS;
  extern __shared__ __align__(16) B smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, l = lane & 3;
  const int row0 = blockIdx.x * R, tok0 = blockIdx.z * T;
  const int ks = K / splits, begin = blockIdx.y * ks, end = begin + ks;
  const int tiles = (ks + BK - 1) / BK;
  auto load = [&](int tile) {
    B* s = smem + (tile % STAGES) * STAGE;
    const int kb = begin + tile * BK;
    // 16-byte cp.async; padded shared rows avoid bank conflicts for the MMA
    // fragment's eight independent rows (four lanes per row).
    for (int v = tid; v < (R + T) * BK / 8; v += WARPS * 32) {
      const int r = v / (BK / 8), c = (v % (BK / 8)) * 8;
      const bool is_w = r < R;
      const int global_r = is_w ? row0 + r : tok0 + r - R;
      const bool valid = kb + c < end && global_r < (is_w ? N : M);
      const B* src = valid ? (is_w ? w : x) + (size_t)global_r * K + kb + c : x;
      cp16(s + r * STRIDE + c, src, valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  float a[CELLS][4] = {};
#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) load(st);
  for (int tile = 0; tile < tiles; ++tile) {
    load(tile + STAGES - 1);
    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 1) : "memory");
    __syncthreads();
    B* s = smem + (tile % STAGES) * STAGE;
#pragma unroll
    for (int k = 0; k < BK; k += 16) {
#pragma unroll
      for (int c = 0; c < CELLS; ++c) {
        const int cell = warp + c * WARPS;
        const int rn = (cell / (T / 8)) * 16, tm = (cell % (T / 8)) * 8;
        const B* wa = s + (rn + g) * STRIDE + k + l * 2;
        const B* xb = s + (R + tm + g) * STRIDE + k + l * 2;
        unsigned a0 = *reinterpret_cast<const unsigned*>(wa);
        unsigned a1 = *reinterpret_cast<const unsigned*>(wa + 8 * STRIDE);
        unsigned a2 = *reinterpret_cast<const unsigned*>(wa + 8);
        unsigned a3 = *reinterpret_cast<const unsigned*>(wa + 8 * STRIDE + 8);
        unsigned b0 = *reinterpret_cast<const unsigned*>(xb);
        unsigned b1 = *reinterpret_cast<const unsigned*>(xb + 8);
        mma(a[c], a0, a1, a2, a3, b0, b1);
      }
    }
    __syncthreads();
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  const int nt = (N + R - 1) / R, mt = (M + T - 1) / T;
  const int tile_id = blockIdx.z * nt + blockIdx.x;
#pragma unroll
  for (int c = 0; c < CELLS; ++c) {
    const int cell = warp + c * WARPS;
    const int rn = (cell / (T / 8)) * 16 + g, tm = (cell % (T / 8)) * 8 + l * 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = rn + (i / 2) * 8, t = tm + (i & 1);
      if (row0 + r < N && tok0 + t < M) {
        if (splits == 1) output(y, (size_t)(tok0 + t) * N + row0 + r, a[c][i], fp32);
        else partial[((size_t)tile_id * splits + blockIdx.y) * R * T + t * R + r] = a[c][i];
      }
    }
  }
  if (splits > 1) finish<R, T, WARPS * 32>(partial, counters, y, M, N, splits, nt, mt, fp32);
}

template<int U, int W, int R>
int launch_scalar(const B* w, const B* x, void* y, float* p, unsigned* c,
                  int M, int N, int K, int splits, bool fp32, cudaStream_t stream) {
  scalar<U,W,R><<<dim3((N + W*R - 1)/(W*R), splits, M), W*32, 0, stream>>>
      (w,x,y,p,c,M,N,K,splits,fp32);
  return (int)cudaGetLastError();
}
template<int R, int T, int BK, int S, int W>
int launch_tensor(const B* w, const B* x, void* y, float* p, unsigned* c,
                  int M, int N, int K, int splits, bool fp32, cudaStream_t stream) {
#ifdef SP_LOCAL_SHARED_CAP
  // Local Blackwell has 99 KiB/CTA; H200 has 227 KiB. Execute the identical
  // fragments/reducer/glue with fewer stages locally instead of skipping it.
  constexpr int fit = SP_LOCAL_SHARED_CAP / ((R + T) * (BK + 8) * sizeof(B));
  constexpr int stages = S < fit ? S : fit;
#else
  constexpr int stages = S;
#endif
  static_assert(stages > 0);
  constexpr int bytes = stages * (R + T) * (BK + 8) * sizeof(B);
  auto kernel = tensor<R,T,BK,stages,W>;
  auto e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (e != cudaSuccess) return (int)e;
  kernel<<<dim3((N + R - 1)/R, splits, (M + T - 1)/T), W*32, bytes, stream>>>
      (w,x,y,p,c,M,N,K,splits,fp32);
  return (int)cudaGetLastError();
}
} // namespace stream_proj

// Same-source read control: isolate request throughput from arithmetic and
// output/reduction costs, with the identical cold-L2 graph/event protocol.
__global__ void sp_read_kernel(const uint4* w, unsigned* sink, size_t vectors) {
  const size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = (size_t)gridDim.x * blockDim.x;
  unsigned sum = 0;
  for (size_t b = first; b < vectors; b += step * 8) {
    uint4 v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const size_t i = b + u * step;
      v[u] = i < vectors ? stream_proj::load16((const stream_proj::B*)(w + i)) : uint4{};
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) sum ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  // Every vector contributes; each thread has a private sink element.
  sink[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = sum;
}
extern "C" int sp_read(const void* w, void* sink, size_t bytes, int sms, void* stream) {
  sp_read_kernel<<<sms, 512, 0, (cudaStream_t)stream>>>((const uint4*)w, (unsigned*)sink, bytes/16);
  return (int)cudaGetLastError();
}

extern "C" int sp_launch(const void* w, const void* x, void* y, void* p, void* c,
                         int M, int N, int K, int config, int splits, int fp32, void* stream) {
  using namespace stream_proj;
  if (K % (splits * 8)) return (int)cudaErrorInvalidValue;
#define ARG (const B*)w,(const B*)x,y,(float*)p,(unsigned*)c,M,N,K,splits,(bool)fp32,(cudaStream_t)stream
  switch(config) {
    case 0: return launch_scalar<4,4,1>(ARG);
    case 1: return launch_scalar<8,4,1>(ARG);
    case 2: return launch_scalar<16,4,1>(ARG);
    case 3: return launch_scalar<8,8,1>(ARG);
    case 4: return launch_scalar<4,4,2>(ARG);
    case 5: return launch_scalar<8,4,2>(ARG);
    case 10: return launch_tensor<32,16,128,3,4>(ARG);
    case 11: return launch_tensor<32,16,256,3,4>(ARG);
    case 12: return launch_tensor<64,8,256,3,4>(ARG);
    case 13: return launch_tensor<64,16,256,3,8>(ARG);
    case 14: return launch_tensor<32,32,256,3,8>(ARG);
    case 15: return launch_tensor<32,16,256,4,4>(ARG);
    case 16: return launch_tensor<64,16,128,4,8>(ARG);
    case 20: return launch_tensor<32,64,128,3,8>(ARG);
    case 21: return launch_tensor<64,64,128,2,8>(ARG);
    case 22: return launch_tensor<32,128,128,3,8>(ARG);
    case 23: return launch_tensor<64,128,128,2,8>(ARG);
    case 24: return launch_tensor<32,64,256,3,8>(ARG);
    case 25: return launch_tensor<32,128,256,2,8>(ARG);
    default: return (int)cudaErrorInvalidValue;
  }
#undef ARG
}
