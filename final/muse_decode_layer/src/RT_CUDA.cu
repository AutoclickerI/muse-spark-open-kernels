#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdexcept>

__device__ __forceinline__ float4 rt_load4(const float* p){float4 v;asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"l"(p):"memory");return v;}
__device__ __forceinline__ float rt_load1(const float* p){float v;asm volatile("ld.global.cg.f32 %0,[%1];":"=f"(v):"l"(p):"memory");return v;}
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
  if (f32) return rt_load4(static_cast<const float*>(base)+off);
  uint2 v;const void* p=static_cast<const __nv_bfloat16*>(base)+off;asm volatile("ld.global.cg.v2.u32 {%0,%1},[%2];":"=r"(v.x),"=r"(v.y):"l"(p):"memory");
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
template <int C, int NT, int ROUNDS, bool POST, int MODE>
__global__ void __launch_bounds__(NT) boundary_tail(const void* __restrict__ P, int pf32, int S, long long ps, long long rs,
                                                  int NB, const float* __restrict__ stream, float* sout,
                                                  __nv_bfloat16* __restrict__ out, const float* __restrict__ postw,
                                                  const float* __restrict__ alpha, const float* __restrict__ beta,
                                                  const float* __restrict__ prew, const float* __restrict__ a2, float* __restrict__ tail, float* __restrict__ stats) {
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
    s[g] = rt_load4(stream + (long long)t * HID + j);
    br[g] = j < NB ? sum_splits(P, pf32, S, ps, (long long)t * rs + j) : make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr(POST && (MODE&1))if(rank>0)s[g]=rt_load4(tail+(long long)t*HID+j);
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
        const float x = (POST && (MODE&1) && rank>0) ? ss[k] : aa[k] * ss[k], y = be[k] * bb[k];
        if(!(POST && (MODE&2) && rank>0)){v[0] += bb[k] * bb[k];v[1] += x*x;v[2] += x*y;v[3] += y*y;}
      }
    }
    if constexpr(POST && (MODE&2)){
      if(rank==0)block_sum<4,NT>(v,red);
      else {v[0]=0.f;v[1]=rt_load1(stats+(long long)t*C+rank);v[2]=v[3]=0.f;}
    }else block_sum<4, NT>(v, red);
    cluster_sum<4, C>(cl, v, slots[0]);
    inv_br = rsqrtf(v[0] / HID + 1e-8f);
    inv = rsqrtf((v[1] + 2.f * inv_br * v[2] + inv_br * inv_br * v[3]) / HID + 1e-5f);
#pragma unroll
    for (int g = 0; g < G; ++g) {
      if(!(POST && (MODE&1) && rank>0)){
      s[g].x = a[g].x * s[g].x + b[g].x * (br[g].x * inv_br); s[g].y = a[g].y * s[g].y + b[g].y * (br[g].y * inv_br);
      s[g].z = a[g].z * s[g].z + b[g].z * (br[g].z * inv_br); s[g].w = a[g].w * s[g].w + b[g].w * (br[g].w * inv_br);
      }
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
      if(!(POST && (MODE&1) && rank>0)){
      s[g].x = a[g].x * s[g].x + b[g].x * (br[g].x * inv_br); s[g].y = a[g].y * s[g].y + b[g].y * (br[g].y * inv_br);
      s[g].z = a[g].z * s[g].z + b[g].z * (br[g].z * inv_br); s[g].w = a[g].w * s[g].w + b[g].w * (br[g].w * inv_br);
      }
      u[0] += s[g].x * s[g].x + s[g].y * s[g].y + s[g].z * s[g].z + s[g].w * s[g].w;
    }
    __syncthreads();  // red reuse
    block_sum<1, NT>(u, red);
    cluster_sum<1, C>(cl, u, slots[1]);
    inv = rsqrtf(u[0] / HID + 1e-5f);
  }
  if constexpr(!POST){
    float sq[1]={0.f};
#pragma unroll
    for(int g=0;g<G;++g){
      int j=(rank*NT*G+g*NT+tid)*4;
      if(rank>0){float4 a=rt_load4(a2+j),z=s[g];z.x*=a.x;z.y*=a.y;z.z*=a.z;z.w*=a.w;
        if constexpr(MODE&1)*reinterpret_cast<float4*>(tail+(long long)t*HID+j)=z;
        if constexpr(MODE&2)sq[0]+=z.x*z.x+z.y*z.y+z.z*z.z+z.w*z.w;
      }
    }
    if constexpr(MODE&2){
      __syncthreads();block_sum<1,NT>(sq,red);
      if(tid==0)stats[(long long)t*C+rank]=sq[0];
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const int j = (rank * NT * G + g * NT + tid) * 4;
    if(!(POST && (MODE&1) && rank>0))*reinterpret_cast<float4*>(sout + (long long)t * HID + j) = s[g];
    __nv_bfloat162 o0 = __floats2bfloat162_rn(s[g].x * inv * w[g].x, s[g].y * inv * w[g].y);
    __nv_bfloat162 o1 = __floats2bfloat162_rn(s[g].z * inv * w[g].z, s[g].w * inv * w[g].w);
    uint2 ov = make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));
    *reinterpret_cast<uint2*>(out + (long long)t * HID + j) = ov;
  }
}


} // namespace glue_cl
template<bool POST,int MODE>void rt_launch(std::vector<int64_t>v,int T,cudaStream_t stream){
 auto kernel=glue_cl::boundary_tail<8,256,1,POST,MODE>;
 cudaFuncSetAttribute(kernel,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(8,T);cfg.blockDim=dim3(256);cfg.stream=stream;
 cudaLaunchAttribute a[2]{};a[0].id=cudaLaunchAttributeClusterDimension;a[0].val.clusterDim.x=8;a[0].val.clusterDim.y=1;a[0].val.clusterDim.z=1;
 a[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;a[1].val.programmaticStreamSerializationAllowed=1;
 cfg.attrs=a;cfg.numAttrs=2;
 auto e=cudaLaunchKernelEx(&cfg,kernel,(const void*)v[0],(int)v[1],(int)v[2],(long long)v[3],(long long)v[4],(int)v[5],
  (const float*)v[6],(float*)v[7],(__nv_bfloat16*)v[8],(const float*)v[9],(const float*)v[10],(const float*)v[11],(const float*)v[12],(const float*)v[13],(float*)v[14],(float*)v[15]);
 if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
void rt_boundary(std::vector<int64_t>v,int64_t T,int64_t post,int64_t mode,int64_t stream){
#define RT(M) if(mode==M){if(post)rt_launch<true,M>(v,T,(cudaStream_t)stream);else rt_launch<false,M>(v,T,(cudaStream_t)stream);return;}
 RT(1) RT(2) RT(3)
#undef RT
 throw std::runtime_error("rt mode");
}
std::vector<int64_t> rt_info(int64_t post,int64_t mode){cudaFuncAttributes a{};
#define RT(M) if(mode==M){if(post)cudaFuncGetAttributes(&a,glue_cl::boundary_tail<8,256,1,true,M>);else cudaFuncGetAttributes(&a,glue_cl::boundary_tail<8,256,1,false,M>);}
 RT(1) RT(2) RT(3)
#undef RT
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes};
}
