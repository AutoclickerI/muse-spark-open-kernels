#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <stdexcept>
struct SCRegion {const char* p;uint64_t groups,stride,head;};
struct SCParams {SCRegion r[16];int n;unsigned* sink;};
__device__ __forceinline__ void sc_bulk(const char* p,unsigned n){
 asm volatile("cp.async.bulk.prefetch.L2.global [%0],%1;"::"l"(p),"r"(n):"memory");
}
template<int MODE>__global__ void sc_prefetch(SCParams p){
 // Preserve the transitive producer dependency: independent reads issue now,
 // but this intermediary cannot finish before its own producer completes.
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
 uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x,step=gridDim.x*blockDim.x;
 unsigned sum=0;
 for(int d=0;d<p.n;++d){
  SCRegion r=p.r[d];
  if constexpr(MODE==0){
   uint64_t width=r.head/16,total=r.groups*width;
   for(uint64_t k=tid;k<total;k+=step){
    uint4 v;const char* src=r.p+(k/width)*r.stride+(k%width)*16;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(src):"memory");
    sum^=v.x^v.y^v.z^v.w;
   }
  }else if constexpr(MODE==1){
   uint64_t width=r.head/128,total=r.groups*width;
   for(uint64_t k=tid;k<total;k+=step){
    const char* src=r.p+(k/width)*r.stride+(k%width)*128;
    asm volatile("prefetch.global.L2 [%0];"::"l"(src):"memory");
   }
  }else{
   for(uint64_t g=tid;g<r.groups;g+=step)
    for(uint64_t off=0;off<r.head;off+=65536)
     sc_bulk(r.p+g*r.stride+off,(unsigned)min((uint64_t)65536,r.head-off));
  }
 }
 if constexpr(MODE==0)p.sink[tid]=sum;
 asm volatile("griddepcontrol.wait;":::"memory");
}
void sc_run(std::vector<int64_t> args,int64_t sink,int64_t mode,int64_t stream){
 if(args.size()%4 || args.size()>64)throw std::runtime_error("SC descriptors");
 SCParams p{};p.n=args.size()/4;p.sink=(unsigned*)sink;
 for(int i=0;i<p.n;++i){p.r[i]={(const char*)args[4*i],(uint64_t)args[4*i+1],(uint64_t)args[4*i+2],(uint64_t)args[4*i+3]};
  if((args[4*i]&15)||(p.r[i].stride&15)||(p.r[i].head&15))throw std::runtime_error("SC alignment");}
 cudaLaunchConfig_t c{};c.gridDim=dim3(mode==0?8:2);c.blockDim=dim3(128);c.stream=(cudaStream_t)stream;
 cudaLaunchAttribute a{};a.id=cudaLaunchAttributeProgrammaticStreamSerialization;a.val.programmaticStreamSerializationAllowed=1;c.attrs=&a;c.numAttrs=1;
 cudaError_t e;
 if(mode==0)e=cudaLaunchKernelEx(&c,sc_prefetch<0>,p);
 else if(mode==1)e=cudaLaunchKernelEx(&c,sc_prefetch<1>,p);
 else e=cudaLaunchKernelEx(&c,sc_prefetch<2>,p);
 if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
