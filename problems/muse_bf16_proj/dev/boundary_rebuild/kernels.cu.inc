#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdexcept>
namespace br {
__device__ __forceinline__ void release() {
#if __CUDA_ARCH__ >= 900
 asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void wait() {
#if __CUDA_ARCH__ >= 900
 asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ float4 load4(const float* p) {
 float4 v;asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"l"(p):"memory");return v;
}
__device__ __forceinline__ float load1(const float* p){float x;asm volatile("ld.global.ca.f32 %0,[%1];":"=f"(x):"l"(p):"memory");return x;}
__device__ __forceinline__ int loadi(const int* p){int x;asm volatile("ld.global.ca.u32 %0,[%1];":"=r"(x):"l"(p):"memory");return x;}
__device__ __forceinline__ void ready(const int* p) {
 int v;do {asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");}while(v!=1);
}
__device__ __forceinline__ float4 branch4(const void* P,int f32,int S,long long ps,long long off,const int* flags,int col,int tiles) {
 float4 v=make_float4(0,0,0,0);
 for(int s=0;s<S;++s){
  if(flags){
   // Sixteen adjacent threads own one producer's 64-column tile. One acquire
   // per group, then an explicit warp barrier publishes readiness to its peers.
   if((threadIdx.x&15)==0)ready(flags+s*tiles+col/64);
   __syncwarp((threadIdx.x&16)?0xffff0000u:0x0000ffffu);
  }
  float4 z;
  if(f32){const void* ptr=(const float*)P+s*ps+off;asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(z.x),"=f"(z.y),"=f"(z.z),"=f"(z.w):"l"(ptr):"memory");}
  else{uint2 a;const void* ptr=(const __nv_bfloat16*)P+s*ps+off;asm volatile("ld.global.cg.v2.u32 {%0,%1},[%2];":"=r"(a.x),"=r"(a.y):"l"(ptr):"memory");z=make_float4(__uint_as_float(a.x<<16),__uint_as_float(a.x&0xffff0000),__uint_as_float(a.y<<16),__uint_as_float(a.y&0xffff0000));}
  v.x+=z.x;v.y+=z.y;v.z+=z.z;v.w+=z.w;
 }
 return v;
}
template<int K,int NT>
__device__ __forceinline__ void reduce(float (&v)[K],float* red){
 const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
#pragma unroll
 for(int k=0;k<K;++k){
#pragma unroll
  for(int d=16;d;d>>=1)v[k]+=__shfl_xor_sync(~0u,v[k],d);
  if(lane==0)red[warp*K+k]=v[k];
 }
 __syncthreads();
 if(warp==0){
#pragma unroll
  for(int k=0;k<K;++k){float a=lane<NT/32?red[lane*K+k]:0.f;
#pragma unroll
   for(int d=16;d;d>>=1)a+=__shfl_xor_sync(~0u,a,d);
   if(lane==0)red[k]=a;
  }
 }
 __syncthreads();
#pragma unroll
 for(int k=0;k<K;++k)v[k]=red[k];
 __syncthreads(); // next reduction may reuse red immediately
}
__device__ __forceinline__ void store(__nv_bfloat16* out,float* sout,int at,float4 v,float inv,const float* prew){
 *reinterpret_cast<float4*>(sout+at)=v;float4 w=load4(prew+(at&8191));
 __nv_bfloat162 a=__floats2bfloat162_rn(v.x*inv*w.x,v.y*inv*w.y),b=__floats2bfloat162_rn(v.z*inv*w.z,v.w*inv*w.w);
 *reinterpret_cast<uint2*>(out+at)=make_uint2(*reinterpret_cast<uint32_t*>(&a),*reinterpret_cast<uint32_t*>(&b));
}
template<int MODE,int NT>
__global__ __launch_bounds__(NT) void boundary(const void* __restrict__ P,int f32,int S,long long ps,long long rs,int NB,
 const float* __restrict__ stream,float* __restrict__ sout,__nv_bfloat16* __restrict__ out,const float* __restrict__ post,
 const float* __restrict__ alpha,const float* __restrict__ beta,const float* __restrict__ prew,const int* __restrict__ flags){
 constexpr int G=8192/(4*NT);const int t=blockIdx.x,tid=threadIdx.x;__shared__ float red[4*(NT/32)];
 release();if(!flags)wait();
 float4 x[G],y[G];float v[4]={0,0,0,0};
#pragma unroll
 for(int g=0;g<G;++g){
  const int j=(g*NT+tid)*4;float4 b=j<NB?branch4(P,f32,S,ps,(long long)t*rs+j,flags,j,8192/64):make_float4(0,0,0,0);
  if(post){float4 q=load4(post+j);b.x*=q.x;b.y*=q.y;b.z*=q.z;b.w*=q.w;}
  y[g]=b;v[0]+=b.x*b.x+b.y*b.y+b.z*b.z+b.w*b.w;
  if constexpr(MODE==1){
   const float4 a=load4(alpha+j),be=load4(beta+j),s=load4(stream+t*8192+j);
   x[g]=make_float4(a.x*s.x,a.y*s.y,a.z*s.z,a.w*s.w);y[g]=make_float4(be.x*b.x,be.y*b.y,be.z*b.z,be.w*b.w);
   v[1]+=x[g].x*x[g].x+x[g].y*x[g].y+x[g].z*x[g].z+x[g].w*x[g].w;
   v[2]+=x[g].x*y[g].x+x[g].y*y[g].y+x[g].z*y[g].z+x[g].w*y[g].w;
   v[3]+=y[g].x*y[g].x+y[g].y*y[g].y+y[g].z*y[g].z+y[g].w*y[g].w;
  }
 }
 float inv,r;
 if constexpr(MODE==1){
  reduce<4,NT>(v,red);r=rsqrtf(v[0]/8192.f+1.e-8f);inv=rsqrtf((v[1]+2.f*r*v[2]+r*r*v[3])/8192.f+1.e-5f);
#pragma unroll
  for(int g=0;g<G;++g)x[g]=make_float4(x[g].x+r*y[g].x,x[g].y+r*y[g].y,x[g].z+r*y[g].z,x[g].w+r*y[g].w);
 }else{
  float a[1]={v[0]};reduce<1,NT>(a,red);r=rsqrtf(a[0]/8192.f+1.e-8f);a[0]=0;
#pragma unroll
  for(int g=0;g<G;++g){
   int j=(g*NT+tid)*4;float4 al=load4(alpha+j),be=load4(beta+j),s=load4(stream+t*8192+j),b=y[g];
   x[g]=make_float4(al.x*s.x+be.x*(b.x*r),al.y*s.y+be.y*(b.y*r),al.z*s.z+be.z*(b.z*r),al.w*s.w+be.w*(b.w*r));
   a[0]+=x[g].x*x[g].x+x[g].y*x[g].y+x[g].z*x[g].z+x[g].w*x[g].w;
  }
  reduce<1,NT>(a,red);inv=rsqrtf(a[0]/8192.f+1.e-5f);
 }
#pragma unroll
 for(int g=0;g<G;++g)store(out,sout,t*8192+(g*NT+tid)*4,x[g],inv,prew);
}
__device__ __forceinline__ unsigned order(float x){unsigned b=__float_as_uint(x);return b&0x80000000?~b:b^0x80000000;}
template<int NT>
__global__ __launch_bounds__(NT) void route(const void* __restrict__ P,int f32,int S,long long ps,long long rs,const float* __restrict__ bias,
 const float* __restrict__ prew,const int* __restrict__ lstart,__nv_bfloat16* __restrict__ X,int* __restrict__ ids,float* __restrict__ wts,int* __restrict__ lids){
 constexpr int G=4096/(4*NT);const int t=blockIdx.x,tid=threadIdx.x;__shared__ float red[NT/32],scores[256];__shared__ unsigned keys[256];
 release();const int start=loadi(lstart);const float bs=tid<256?load1(bias+tid):0.f;wait();float4 p[G];float v[1]={0};
#pragma unroll
 for(int g=0;g<G;++g){int j=(g*NT+tid)*4;p[g]=branch4(P,f32,S,ps,(long long)t*rs+j,nullptr,j,0);float4 a=p[g];v[0]+=a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w;}
 if(tid<256){float z=0;
  for(int s=0;s<S;++s){long long off=s*ps+(long long)t*rs+4096+tid;float a,b;if(f32){a=((const float*)P)[off];b=((const float*)P)[off+256];}else{a=__bfloat162float(((const __nv_bfloat16*)P)[off]);b=__bfloat162float(((const __nv_bfloat16*)P)[off+256]);}z+=a+b;}
  float sc=1.f/(1.f+__expf(-z*0.1767766952966369f));scores[tid]=sc;keys[tid]=order(sc+bs);
 }
 reduce<1,NT>(v,red);float inv=rsqrtf(v[0]/4096.f+1.e-5f);
#pragma unroll
 for(int g=0;g<G;++g){int j=(g*NT+tid)*4;float4 a=p[g],w=load4(prew+j);__nv_bfloat162 x=__floats2bfloat162_rn(a.x*inv*w.x,a.y*inv*w.y),y=__floats2bfloat162_rn(a.z*inv*w.z,a.w*inv*w.w);*reinterpret_cast<uint2*>(X+t*4096+j)=make_uint2(*reinterpret_cast<uint32_t*>(&x),*reinterpret_cast<uint32_t*>(&y));}
 if(tid<32){
  unsigned k[8];int pick[8];float sc[8];
#pragma unroll
  for(int i=0;i<8;++i)k[i]=keys[tid*8+i];
#pragma unroll
  for(int r=0;r<8;++r){unsigned top=k[0];int best=0;
#pragma unroll
   for(int i=1;i<8;++i)if(k[i]>top){top=k[i];best=i;}
   unsigned val=__reduce_max_sync(~0u,top);int e=__reduce_min_sync(~0u,top==val?tid*8+best:0xffffffffu);pick[r]=e;sc[r]=scores[e];
#pragma unroll
   for(int i=0;i<8;++i)if(tid*8+i==e)k[i]=0;
  }
  float total=0;
#pragma unroll
  for(int i=0;i<8;++i)total+=sc[i];
  if(tid<8){int e=pick[tid];ids[t*8+tid]=e;wts[t*8+tid]=sc[tid]/(total+1.e-15f);int local=(e-start+256)%256;lids[t*8+tid]=local<32?local:-1;}
 }
}
template<class K,class... A>void launch(K k,int T,int NT,int serial,cudaStream_t stream,A...args){cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(T);cfg.blockDim=dim3(NT);cfg.stream=stream;cudaLaunchAttribute a{};a.id=cudaLaunchAttributeProgrammaticStreamSerialization;a.val.programmaticStreamSerializationAllowed=!serial;cfg.attrs=&a;cfg.numAttrs=1;auto e=cudaLaunchKernelEx(&cfg,k,args...);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}
void boundary(int64_t P,int64_t f32,int64_t S,int64_t ps,int64_t rs,int64_t NB,int64_t stream,int64_t sout,int64_t out,int64_t post,int64_t alpha,int64_t beta,int64_t prew,int64_t T,int64_t mode,int64_t nt,int64_t flags,int64_t serial,int64_t handle){
#define B(M,N) if(mode==M && nt==N){br::launch(br::boundary<M,N>,T,N,serial,(cudaStream_t)handle,(void*)P,(int)f32,(int)S,(long long)ps,(long long)rs,(int)NB,(float*)stream,(float*)sout,(__nv_bfloat16*)out,(float*)post,(float*)alpha,(float*)beta,(float*)prew,(int*)flags);return;}
 B(1,256) B(1,512) B(2,256) B(2,512)
 throw std::runtime_error("boundary geometry");
}
void route(int64_t P,int64_t f32,int64_t S,int64_t ps,int64_t rs,int64_t bias,int64_t prew,int64_t start,int64_t X,int64_t ids,int64_t wts,int64_t lids,int64_t T,int64_t nt,int64_t handle){
#define R(N) if(nt==N){br::launch(br::route<N>,T,N,0,(cudaStream_t)handle,(void*)P,(int)f32,(int)S,(long long)ps,(long long)rs,(float*)bias,(float*)prew,(int*)start,(__nv_bfloat16*)X,(int*)ids,(float*)wts,(int*)lids);return;}
 R(256) R(512)
 throw std::runtime_error("route geometry");
}
