#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
namespace cg = cooperative_groups;

#include <cuda.h>
#include <cudaTypedefs.h>
#ifndef MP_SPIN_LIMIT
#define MP_SPIN_LIMIT (1u << 26)
#endif
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define MP_SM90A 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define MP_TMA 1  // TMA, mbarriers, clusters: sm_90 and later (wgmma itself needs sm_90a)
#endif

namespace muse_wg {
__device__ __forceinline__ void expect_only(uint64_t* bar, uint32_t bytes) {
  const uint32_t address = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
      :: "r"(address), "r"(bytes) : "memory");
}


__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

#ifdef MP_TMA
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su32(bar)) : "memory");
}
// Bounded: a pipeline bug yields wrong numbers (caught by tests) instead of a hung GPU.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t done = 0;
  uint32_t tries = 0;
  do {
    if (false) {
#ifdef MP_DEBUG
      printf("mbar_wait timeout block (%d,%d,%d) thread %d parity %u\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, parity);
#endif
      break;
    }
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done) : "r"(su32(bar)), "r"(parity) : "memory");
  } while (!done);
}
// Same box into this CTA and every CTA in mask (same smem offsets, each one's own mbarrier).
__device__ __forceinline__ void tma_load_2d_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                               uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(su32(bar)), "r"(c0), "r"(c1), "h"(mask)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(uint64_t* bar, uint32_t cta_rank) {
  asm volatile(
      "{\n .reg .b32 ra;\n mapa.shared::cluster.u32 ra, %0, %1;\n"
      " mbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\n}\n"
      ::"r"(su32(bar)), "r"(cta_rank) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(su32(bar)), "r"(c0), "r"(c1) : "memory");
}
template <int N_>
__device__ __forceinline__ void acc_fence(float (&d)[N_]) {
#pragma unroll
  for (int i = 0; i < N_; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
#endif
#ifdef MP_SM90A
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N_>
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N_) : "memory"); }
// K-major operand, 128-byte swizzle, 8-row atoms 1024 bytes apart.
__device__ __forceinline__ uint64_t make_desc(const void* p) {
  const uint64_t a = su32(p);
  return ((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wgmma_m64n8(float (&d)[4], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %6, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n16(float (&d)[8], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n32(float (&d)[16], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n64(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n128(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n256(float (&d)[128], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "l"(da), "l"(db), "r"(scale_d));
}

#elif defined(MP_TMA)
// Off sm_90a: the same swizzled tiles through ldmatrix + mma.sync (validates the pipeline).
__device__ __forceinline__ void ldsm_x4(const void* p, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(su32(p)));
}
__device__ __forceinline__ void ldsm_x2(const void* p, uint32_t& b0, uint32_t& b1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(su32(p)));
}
__device__ __forceinline__ void mma_emu(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

// out[M, N] = X[M, K] @ W[N, K]^T on Hopper: TMA tiles (128-byte swizzle) of BR = 128*NSUB weight
// rows and BT tokens per 64-wide K step, ST stages; warpgroups 0-1 run wgmma (weights on its M
// side, tokens on N), warpgroup 2's first thread issues the TMA loads. K is split over the
// cluster's CTAs (blockIdx.y) and reduced through DSMEM, 64 tokens per pass.
// MC = 2: the CTA pair (blockIdx.x, blockIdx.x ^ 1) in a cluster shares one activation tile, each
// loading half of it and multicasting to both; a stage is refilled only after both pairs' consumers
// release it.
template <int BT, int NSUB, int NCW, int ST, int MC, bool OUTF32>
__global__ void __launch_bounds__(128 * NCW + 32) wg_kernel(const __grid_constant__ CUtensorMap tmW,
                                                    const __grid_constant__ CUtensorMap tmX,
                                                    void* __restrict__ Out, int M, int N, int nkb, int kx, int pipe, int pdl, int kgroup) {
#ifdef MP_TMA
  constexpr int BR = 64 * NSUB * NCW;
  constexpr int A_BYTES = BR * 64 * 2;
  constexpr int B_BYTES = BT * 64 * 2;
  constexpr int SLICE_BYTES = A_BYTES + B_BYTES;
  const int STAGE_BYTES = SLICE_BYTES * kgroup;
  constexpr int NACC = BT / 2;
  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem = (unsigned char*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * STAGE_BYTES);
  uint64_t* empty = full + ST;
  const int wg = threadIdx.x >> 7;

  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int groups_n = (N / BR) / MC;
  const int jobs = groups_n * ((M + BT - 1) / BT);
  for (int job = blockIdx.x / MC; job < jobs; job += gridDim.x / MC) {
  const int row_cta = (job % groups_n) * MC * BR + (blockIdx.x % MC) * BR;
  const int tok0 = (job / groups_n) * BT;
  const int ks = gridDim.y;
  const int kb0 = (int)((long long)blockIdx.y * nkb / ks);
  const int kb1 = (int)((long long)(blockIdx.y + 1) * nkb / ks);
  const int nsteps = (kb1 - kb0) / kgroup;
  const bool clustered = MC > 1 || ks > 1;
  const int cx = (int)(blockIdx.x % MC);         // position in the multicast pair
  const int my_rank = cx + MC * (int)blockIdx.y;  // rank in the (MC, ks) cluster
  if (threadIdx.x == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(&full[s], 1); expect_only(&full[s], STAGE_BYTES);
      mbar_init(&empty[s], NCW * MC);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  if (MC > 1)
    cg::this_cluster().sync();  // partners' barriers must exist before multicast or remote arrive
  else
    __syncthreads();

  float acc[NSUB][NACC];
#pragma unroll
  for (int s = 0; s < NSUB; ++s)
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[s][i] = 0.f;

  if (wg == NCW) {
    if (threadIdx.x == 128 * NCW) {
      for (int s = 0; s < nsteps; ++s) {
        const int stage = s % ST;
        const uint32_t ph = (s / ST) & 1;
        if (s >= ST) mbar_wait(&empty[stage], ph ^ 1);
        mbar_arrive(&full[stage]);
        for (int kg = 0; kg < kgroup; ++kg) {
        unsigned char* sa = smem + stage * STAGE_BYTES + kg * SLICE_BYTES;
        const int kc = (kb0 + s * kgroup + kg) * 64;
        tma_load_2d(sa, &tmW, &full[stage], kc, row_cta);
        if (s == 0 && pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
        if (MC == 1) {
          tma_load_2d(sa + A_BYTES, &tmX, &full[stage], kc % kx, tok0);
        } else {
#ifdef MP_SM90A
          const uint16_t mask = (uint16_t)(((1u << MC) - 1) << (MC * blockIdx.y));
          tma_load_2d_mc(sa + A_BYTES + cx * (BT / MC) * 128, &tmX, &full[stage], kc % kx, tok0 + cx * (BT / MC),
                         mask);
#else
          // Portable transport stand-in: native multicast is optimized for sm90a.
          // Keep identical barriers, cluster scheduling and arithmetic locally.
          for (int part = 0; part < MC; ++part)
            tma_load_2d(sa + A_BYTES + part * (BT / MC) * 128, &tmX, &full[stage], kc % kx,
                        tok0 + part * (BT / MC));
#endif
        }
        }  // K transaction group
      }
    }
  } else {
    for (int s = 0; s < nsteps; ++s) {
      const int stage = s % ST;
      const uint32_t ph = (s / ST) & 1;
      mbar_wait(&full[stage], ph);
      unsigned char* stage_base = smem + stage * STAGE_BYTES;
#ifdef MP_SM90A
#pragma unroll
      for (int sub = 0; sub < NSUB; ++sub) acc_fence(acc[sub]);
      wgmma_fence();
      for (int kg = 0; kg < kgroup; ++kg) {
        unsigned char* sa = stage_base + kg * SLICE_BYTES;
      const uint64_t db = make_desc(sa + A_BYTES);

#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) {
          const uint64_t da = make_desc(sa + (wg * NSUB + sub) * 64 * 128);
          if constexpr (BT == 8) wgmma_m64n8(acc[sub], da + 2 * kk, db + 2 * kk, 1);
          else if constexpr (BT == 16) wgmma_m64n16(acc[sub], da + 2 * kk, db + 2 * kk, 1);
          else if constexpr (BT == 32) wgmma_m64n32(acc[sub], da + 2 * kk, db + 2 * kk, 1);
          else if constexpr (BT == 64) wgmma_m64n64(acc[sub], da + 2 * kk, db + 2 * kk, 1);
          else if constexpr (BT == 128) wgmma_m64n128(acc[sub], da + 2 * kk, db + 2 * kk, 1);
          else if constexpr (BT == 256) wgmma_m64n256(acc[sub], da + 2 * kk, db + 2 * kk, 1);
        }

      }
      wgmma_commit();
      if (pipe) wgmma_wait<1>(); else wgmma_wait<0>();
#pragma unroll
      for (int sub = 0; sub < NSUB; ++sub) acc_fence(acc[sub]);
#else
      for (int kg = 0; kg < kgroup; ++kg) {
        unsigned char* sa = stage_base + kg * SLICE_BYTES;

      {
        const int wq = (threadIdx.x >> 5) & 3, ln = threadIdx.x & 31;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
          for (int sub = 0; sub < NSUB; ++sub) {
            const int arow = (wg * NSUB + sub) * 64 + wq * 16 + (ln & 15);
            const int achunk = kk * 2 + (ln >> 4);
            uint32_t a0, a1, a2, a3;
            ldsm_x4(sa + arow * 128 + ((achunk ^ (arow & 7)) << 4), a0, a1, a2, a3);
#pragma unroll
            for (int j = 0; j < BT / 8; ++j) {
              const int brow = j * 8 + (ln & 7);
              const int bchunk = kk * 2 + ((ln >> 3) & 1);
              uint32_t b0, b1;
              ldsm_x2(sa + A_BYTES + brow * 128 + ((bchunk ^ (brow & 7)) << 4), b0, b1);
              mma_emu(&acc[sub][4 * j], a0, a1, a2, a3, b0, b1);
            }
          }
        // mma.sync warps finish independently: sync the warpgroup before releasing the stage.
        asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory");
      }

      }
#endif
      if ((threadIdx.x & 127) == 0 && (!pipe || s > 0)) {
        const int release_stage = pipe ? (s + ST - 1) % ST : stage;
        expect_only(&full[release_stage], STAGE_BYTES / NCW);
        mbar_arrive(&empty[release_stage]);
        if (MC > 1) mbar_arrive_remote(&empty[release_stage], (uint32_t)(my_rank ^ 1));
      }
    }
#ifdef MP_SM90A
    wgmma_wait<0>();
#endif
    if (pipe && nsteps > 0 && (threadIdx.x & 127) == 0) {
      const int last = (nsteps - 1) % ST;
      expect_only(&full[last], STAGE_BYTES / NCW);
      mbar_arrive(&empty[last]);
      if (MC > 1) mbar_arrive_remote(&empty[last], (uint32_t)(my_rank ^ 1));
    }
  }
  __syncthreads();  // pipeline buffers are free: reuse them as the reduction tile

  constexpr int S = BR + 4;
  constexpr int PT = BT < 64 ? BT : 64;  // tokens per reduction pass
  float* sred = reinterpret_cast<float*>(smem);
  const int warp_in_wg = (threadIdx.x >> 5) & 3;
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  auto store = [&](int tok, int row, float v) {
    const size_t idx = (size_t)(tok0 + tok) * N + row_cta + row;
    if (OUTF32)
      reinterpret_cast<float*>(Out)[idx] = v;
    else
      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
  };
#pragma unroll
  for (int pass = 0; pass < BT / PT; ++pass) {
    if (wg < NCW) {
#pragma unroll
      for (int sub = 0; sub < NSUB; ++sub) {
        const int row = (wg * NSUB + sub) * 64 + warp_in_wg * 16 + g;
#pragma unroll
        for (int ii = 0; ii < PT / 8; ++ii) {
          const int i = pass * (PT / 8) + ii;
          float* p = sred + (size_t)(ii * 8 + 2 * t) * S + row;
          p[0] = acc[sub][4 * i + 0];
          p[S] = acc[sub][4 * i + 1];
          p[8] = acc[sub][4 * i + 2];
          p[S + 8] = acc[sub][4 * i + 3];
        }
      }
    }
    const int ptok = pass * PT;
    const int E = PT * BR;
    if (ks == 1) {
      __syncthreads();
      for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const int tok = e / BR, row = e - tok * BR;
        if (tok0 + ptok + tok < M) store(ptok + tok, row, sred[tok * S + row]);
      }
      __syncthreads();
    } else {
      cg::cluster_group cluster = cg::this_cluster();
      cluster.sync();
      const int slice = (int)blockIdx.y;
      const int per = (E + ks - 1) / ks;
      const int e0 = slice * per;
      const int e1 = min(E, e0 + per);
      for (int e = e0 + (int)threadIdx.x; e < e1; e += blockDim.x) {
        const int tok = e / BR, row = e - tok * BR;
        if (tok0 + ptok + tok >= M) continue;
        float v = 0.f;
        for (int q = 0; q < ks; ++q) v += cluster.map_shared_rank(sred, cx + MC * q)[tok * S + row];
        store(ptok + tok, row, v);
      }
      cluster.sync();
    }
  }
  if (MC > 1 && ks == 1) cg::this_cluster().sync();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(su32(&full[s])) : "memory");
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(su32(&empty[s])) : "memory");
    }
  }
  }  // next persistent logical tile (all cluster partners use the same schedule)
#endif
}

static PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// 2D BF16 tensor map over a row-major [rows, cols] matrix, box [box_rows, 64], 128-byte swizzle.
static CUtensorMap make_map(const void* ptr, int rows, int cols, int box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides,
                           box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

template <int BT, int NSUB, int NCW, int ST, int MC>
void wg_launch_t(const torch::Tensor& W, const torch::Tensor& X, torch::Tensor& Out, int N, int ks, int waves, int pipe, int pdl, int kgroup) {
  const int M = X.size(0), KX = X.size(1), KW = W.size(1);
  constexpr int BR = 64 * NSUB * NCW;
  TORCH_CHECK(N % BR == 0 && KW % 64 == 0 && KX % 64 == 0, "wg: bad shape");
  const bool f32 = Out.scalar_type() == at::kFloat;
  auto kern = f32 ? wg_kernel<BT, NSUB, NCW, ST, MC, true> : wg_kernel<BT, NSUB, NCW, ST, MC, false>;
  TORCH_CHECK((N / BR) % MC == 0 && BT % (8 * MC) == 0, "wg: multicast pair needs an even tile count");
  TORCH_CHECK(kgroup == 1 || kgroup == 2 || kgroup == 4, "K group");
  TORCH_CHECK((KW/64) % (ks*kgroup) == 0, "K partition/group");
  const size_t stage = (size_t)(BR + BT) * 64 * 2 * kgroup;
  size_t smem = ST * stage + 2 * ST * 8 + 1024;
  const size_t red = (size_t)64 * (BR + 4) * 4 + 1024;
  if (red > smem) smem = red;
  static int optin = 0;
  static bool attr_set[2] = {false, false};
  if (!attr_set[f32]) {  // once per instance, before any graph capture (prepare runs every config)
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr_set[f32] = true;
  }
  TORCH_CHECK(smem <= (size_t)optin, "wg: too much shared memory: ", smem, " > ", optin);
  CUtensorMap mw = make_map(W.data_ptr(), N, KW, BR);
  CUtensorMap mx = make_map(X.data_ptr(), M, KX, BT / MC);
  cudaLaunchConfig_t cfg = {};
  int dev = 0, sms = 0;
  cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int logical = ((N / BR) / MC) * ((M + BT - 1) / BT);
  const int groups = waves ? min(logical, max(1, sms * waves / (MC * ks))) : logical;
  cfg.gridDim = dim3(groups * MC, ks, 1);
  cfg.blockDim = dim3(128 * NCW + 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  cfg.numAttrs = 0;
  if (ks > 1 || MC > 1) {
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = MC;
    attr[0].val.clusterDim.y = ks;
    attr[0].val.clusterDim.z = 1;
    cfg.numAttrs = 1;
  }
  if (pdl) {
    attr[cfg.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[cfg.numAttrs].val.programmaticStreamSerializationAllowed = 1;
    ++cfg.numAttrs;
  }
  cfg.attrs = attr;
  auto err = cudaLaunchKernelEx(&cfg, kern, mw, mx, Out.data_ptr(), M, N, KW / 64, KX, pipe, pdl, kgroup);
  TORCH_CHECK(err == cudaSuccess, "wg launch: ", cudaGetErrorString(err));
}

}  // namespace muse_wg

// out[M, N] = X[M, KX] @ W[:N, :KW]^T with the X column index taken mod KX (KW = 2*KX: the
// router's [hi | lo]). Hopper (sm_90a) only.
void wg_gemm(torch::Tensor W, torch::Tensor X, torch::Tensor Out, int64_t N, int64_t bt, int64_t nsub,
             int64_t ncw, int64_t st, int64_t ks, int64_t mc, int64_t waves, int64_t pipe, bool pdl, int64_t kgroup) {
  using namespace muse_wg;
#define WG_CASE(BT_, NSUB_, NCW_, ST_, MC_)                                          \
  if (bt == BT_ && nsub == NSUB_ && ncw == NCW_ && st == ST_ && mc == MC_) {         \
    wg_launch_t<BT_, NSUB_, NCW_, ST_, MC_>(W, X, Out, (int)N, (int)ks, (int)waves, (int)pipe, (int)pdl, (int)kgroup);             \
    return;                                                                          \
  }
  WG_CASE(32, 1, 1, 2, 1) WG_CASE(32, 1, 1, 2, 2) WG_CASE(64, 1, 1, 2, 1) WG_CASE(64, 1, 1, 2, 2) WG_CASE(64, 1, 1, 3, 1) WG_CASE(64, 1, 1, 3, 2) WG_CASE(128, 1, 1, 2, 1) WG_CASE(128, 1, 1, 2, 2) WG_CASE(128, 1, 1, 3, 1) WG_CASE(128, 1, 1, 3, 2) WG_CASE(64, 1, 2, 2, 1) WG_CASE(64, 1, 2, 2, 2) WG_CASE(256, 1, 1, 2, 1) WG_CASE(256, 1, 1, 2, 2) WG_CASE(64, 1, 1, 4, 1) WG_CASE(64, 1, 1, 4, 2) WG_CASE(64, 1, 1, 5, 1) WG_CASE(64, 1, 1, 5, 2) WG_CASE(64, 1, 1, 6, 1) WG_CASE(64, 1, 1, 6, 2) WG_CASE(128, 1, 1, 4, 1) WG_CASE(128, 1, 1, 4, 2) WG_CASE(128, 1, 1, 6, 1) WG_CASE(128, 1, 1, 6, 2) WG_CASE(256, 1, 1, 3, 1) WG_CASE(256, 1, 1, 3, 2) WG_CASE(256, 1, 1, 4, 1) WG_CASE(256, 1, 1, 4, 2) WG_CASE(256, 1, 1, 5, 1) WG_CASE(256, 1, 1, 5, 2) WG_CASE(64, 1, 2, 4, 1) WG_CASE(64, 1, 2, 4, 2) WG_CASE(128, 1, 2, 3, 1) WG_CASE(128, 1, 2, 3, 2) WG_CASE(128, 1, 2, 4, 1) WG_CASE(128, 1, 2, 4, 2) WG_CASE(256, 1, 2, 3, 1)
  TORCH_CHECK(false, "wg: no instance for bt=", bt, " nsub=", nsub, " ncw=", ncw, " st=", st, " mc=", mc);
}
