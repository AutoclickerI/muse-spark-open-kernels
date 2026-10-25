// A compact expert-major work list, built inside the live route CTA.
namespace tail {
TAIL_ROUTE
// MODE0 is a compact static schedule. MODE1 claims one full expert/channel tile.
// NT1 remains responsible for <=8-token experts; no ready-spin or grid barrier.
template<int MODE,int S,bool F>__global__ void __launch_bounds__(256,1) kernel(
 const int*Q,const unsigned char*SC,const float*GS,const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,int*work,unsigned scale_base){
 constexpr int MT=F?64:32;extern __shared__ __align__(1024) unsigned char memory[];
 float*red=(float*)memory+(threadIdx.x>>7)*256;__shared__ int ticket;
 int total=work[2]*MT;
 for(int base=blockIdx.x;;base+=gridDim.x){
  int u;
  if constexpr(MODE){if(threadIdx.x==0)ticket=atomicAdd(work+(F?0:1),1);__syncthreads();u=ticket;}else u=base;
  if(u>=total)break;
  int e=work[3+u/MT],mt=(u%MT)*2+(threadIdx.x>>7),n=C[e],row=n>256?(n/256)*256:0,rem=n-row;
#define TAIL_TILE(NT) if constexpr(S==3)qt3::tile<NT,F,1,S>(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS,e,row,mt,nullptr,nullptr,scale_base,red);else oc::tile<NT,F,1,S>(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS,e,row,mt,nullptr,nullptr,scale_base,red)
  if(rem<=16){TAIL_TILE(16);}else if(rem<=32){TAIL_TILE(32);}else{TAIL_TILE(64);}
#undef TAIL_TILE
  // The tile's final all-WG barrier completes norm scratch before another tile.
 }
}
}
void tail_route(std::vector<int64_t>v,int64_t work,int64_t M,int64_t start,int64_t stream){
 tail::route<true><<<1,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],M,32,start,(int*)work);BR_CUDA_CHECK(cudaGetLastError());
}
template<int MODE,int S,bool F>std::vector<int64_t> tail_resource(){cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncSetAttribute(tail::kernel<MODE,S,F>,cudaFuncAttributeMaxDynamicSharedMemorySize,oc::dyn<S,64>()));BR_CUDA_CHECK(cudaFuncGetAttributes(&a,tail::kernel<MODE,S,F>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,tail::kernel<MODE,S,F>,256,oc::dyn<S,64>()));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,oc::dyn<S,64>(),n};}
std::vector<int64_t> tail_info(int64_t mode,int64_t s,int64_t f){
#define TAIL_INFO(A,S) if(mode==A&&s==S)return f?tail_resource<A,S,true>():tail_resource<A,S,false>();
 TAIL_INFO(0,2) TAIL_INFO(1,2) TAIL_INFO(1,3)
#undef TAIL_INFO
 BR_REQUIRE(false);return {};
}
void tail_run(std::vector<int64_t>v,int64_t work,int64_t mode,int64_t s,int64_t first,int64_t M,int64_t grid,int64_t stream){
#define TAIL_RUN(A,S) if(mode==A&&s==S){if(first)tail::kernel<A,S,true><<<grid,256,oc::dyn<S,64>(),(cudaStream_t)stream>>>(BR_ARGS(v),M,(int*)work,0x70007000u);else tail::kernel<A,S,false><<<grid,256,oc::dyn<S,64>(),(cudaStream_t)stream>>>(BR_ARGS(v),M,(int*)work,0x70007000u);BR_CUDA_CHECK(cudaGetLastError());return;}
 TAIL_RUN(0,2) TAIL_RUN(1,2) TAIL_RUN(1,3)
#undef TAIL_RUN
 BR_REQUIRE(false);
}
