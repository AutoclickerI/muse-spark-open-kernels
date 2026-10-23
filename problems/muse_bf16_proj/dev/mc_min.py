import torch
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
namespace cg = cooperative_groups;
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__global__ void __cluster_dims__(2, 1, 1) mc_kernel(const __grid_constant__ CUtensorMap map, float* out, int use_mc) {
  __shared__ __align__(1024) unsigned char buf[4096];
  __shared__ __align__(8) uint64_t bar;
  const int rank = cg::this_cluster().block_rank();
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) buf[i] = 0;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(su32(&bar)));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  cg::this_cluster().sync();
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(&bar)), "r"(4096) : "memory");
    if (use_mc) {
      // each CTA loads 16 rows (2 KB) of a 32x64 bf16 box and multicasts to both
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
        ::"r"(su32(buf + rank * 2048)), "l"((uint64_t)&map), "r"(su32(&bar)), "r"(0), "r"(rank * 16), "h"((uint16_t)3) : "memory");
    } else {
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        ::"r"(su32(buf)), "l"((uint64_t)&map), "r"(su32(&bar)), "r"(0), "r"(0) : "memory");
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        ::"r"(su32(buf + 2048)), "l"((uint64_t)&map), "r"(su32(&bar)), "r"(0), "r"(16) : "memory");
    }
  }
  uint32_t done = 0, tries = 0;
  while (!done && ++tries < 1000000) {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(su32(&bar)) : "memory");
  }
  __syncthreads();
  float s = 0.f;
  for (int i = threadIdx.x; i < 2048; i += blockDim.x) s += __bfloat162float(reinterpret_cast<__nv_bfloat16*>(buf)[i]);
  atomicAdd(&out[rank * 2], s);
  if (threadIdx.x == 0) out[rank * 2 + 1] = done ? 1.f : -1.f;
  cg::this_cluster().sync();
}
void run(torch::Tensor x, torch::Tensor out, int64_t use_mc) {
  CUtensorMap m;
  cuuint64_t dims[2] = {64, 32}; cuuint64_t strides[1] = {128}; cuuint32_t box[2] = {64, (cuuint32_t)(use_mc ? 16 : 16)}; cuuint32_t es[2] = {1, 1};
  void* fn = nullptr; cudaDriverEntryPointQueryResult q;
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
  auto r = ((PFN_cuTensorMapEncodeTiled_v12000)fn)(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), dims, strides, box, es,
     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == 0);
  mc_kernel<<<2, 128, 0, at::cuda::getCurrentCUDAStream()>>>(m, out.data_ptr<float>(), (int)use_mc);
}
'''
mod = load_inline(name="mcmin3", cpp_sources="void run(torch::Tensor x, torch::Tensor out, int64_t use_mc);", cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-std=c++20"])
x = torch.ones(32, 64, device="cuda").bfloat16()
for mc in (0, 1):
    out = torch.zeros(4, device="cuda")
    mod.run(x, out, mc); torch.cuda.synchronize()
    print("use_mc", mc, "sum/flag per CTA:", out.tolist(), "(expect 2048, 1 each)")
