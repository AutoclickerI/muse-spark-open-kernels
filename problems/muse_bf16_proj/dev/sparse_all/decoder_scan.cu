#include <cuda_runtime.h>
#include <stdint.h>
// The builder expands both reader implementations under unique names.
DC_SCALAR
DC_VECTOR
DC_VIEW
template<int MODE> __global__ void scan_k(const void*raw,const void*bank,unsigned*out,int units){
  int lane=threadIdx.x&31,u=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(u>=units)return;
  int4 words[3];
  if constexpr(MODE==0){
    const int4*p=(const int4*)raw+u*96+lane;
    words[0]=p[0];words[1]=p[32];words[2]=p[64];
  }else if constexpr(MODE==1)ds_read(bank,u,lane,words);
  else if constexpr(MODE==2)dv_read(bank,u,lane,words);
  else if constexpr(MODE==3)dh_read(bank,u,lane,words);
  else{
    unsigned ext;int rot,fmt;const unsigned char*p=ds_address(bank,u,lane,ext,rot,fmt);
    words[0]=*(const int4*)(p+lane*16);words[1]=*(const int4*)(p+512+lane*16);
    const unsigned*c=(const unsigned*)(p+1024);
    words[2]=make_int4(ds_u32(c+lane),ds_u32(c+32+lane),ds_u32(c+64+lane),ext);
  }
  unsigned v=0;
  #pragma unroll
  for(int i=0;i<12;++i){
    unsigned z=((const unsigned*)words)[i]*(1664525u+1013904223u*(i+1));
    // Nonlinear per-word mixing prevents XOR cancellation from removing the
    // codeword reconstruction whose cost this scanner is meant to attribute.
    v^=__funnelshift_r(z,z,(i*7)&31);
  }
  #pragma unroll
  for(int d=16;d;d>>=1)v^=__shfl_xor_sync(0xffffffffu,v,d);
  if(lane==0)out[u]=v;
}
void scan(uint64_t raw,uint64_t bank,uint64_t out,int units,int mode,uint64_t stream){
  #define RUN(M) scan_k<M><<<(units+7)/8,256,0,(cudaStream_t)stream>>>((const void*)raw,(const void*)bank,(unsigned*)out,units)
  if(mode==0){RUN(0);}else if(mode==1){RUN(1);}else if(mode==2){RUN(2);}else if(mode==3){RUN(3);}else{RUN(4);}
  #undef RUN
}
int resources(int mode,int query){
  cudaFuncAttributes a{};int active=0;
  #define GET(M) cudaFuncGetAttributes(&a,scan_k<M>);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,scan_k<M>,256,0)
  if(mode==0){GET(0);}else if(mode==1){GET(1);}else if(mode==2){GET(2);}else if(mode==3){GET(3);}else{GET(4);}
  #undef GET
  return query==2?a.localSizeBytes:query==1?active:a.numRegs;
}
