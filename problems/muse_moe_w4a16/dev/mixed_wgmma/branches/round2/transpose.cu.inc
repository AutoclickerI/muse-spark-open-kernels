// Lossless activation-fragment layout; packed NVFP4 banks remain unchanged.
namespace rt {
using mwbr::smaddr;using mwbr::dq;using mwbr::keep;
RT_MMA
__device__ __align__(16) const unsigned zero_fragment[512]={};
__device__ __forceinline__ size_t ax(int e,int group,int k16,int t,int groups){
 return (((size_t)e*groups+group)*256+k16)*128*4+t*4;
}
__global__ void pack(const __nv_bfloat16*X,const int*C,const int*SLOTS,unsigned*A,int M,int groups){
 int e=blockIdx.x/(groups*32),g=blockIdx.x/32%groups,kh=blockIdx.x%32,t=threadIdx.x,w=t>>5,l=t&31;
 int r=g*64+w*16+(l>>2),src0=r<C[e]?SLOTS[e*M+r]/8:0,src1=r+8<C[e]?SLOTS[e*M+r+8]/8:0;
 if(C[e]<=16||g*64>=C[e])return;
 #pragma unroll
 for(int b=0;b<8;++b){int k=kh*128+b*16+(l&3)*2;uint4 v{};
  if(r<C[e]){v.x=__ldg((const unsigned*)(X+(size_t)src0*4096+k));v.z=__ldg((const unsigned*)(X+(size_t)src0*4096+k+8));}
  if(r+8<C[e]){v.y=__ldg((const unsigned*)(X+(size_t)src1*4096+k));v.w=__ldg((const unsigned*)(X+(size_t)src1*4096+k+8));}
  *(uint4*)(A+ax(e,g,kh*8+b,t,groups))=v;
 }
}

template<int NC,int CAP,bool F>__global__ void __launch_bounds__(NC==128?256:128,1) kernel(
 const int*Q,const unsigned char*SC,const float*GS,const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,const unsigned*A,unsigned*A2){
 constexpr int WG=NC/64,NG=(CAP/64+WG-1)/WG,MT=(F?8192:4096)/NC,N=F?8192:4096;
 extern __shared__ __align__(1024) unsigned char storage[];auto*B=(__nv_bfloat16*)storage;
 int tid=threadIdx.x,t=tid&127,w=t>>5,l=t&31,wg=tid>>7,groups=(M+63)/64;
 for(int u=blockIdx.x;u<32*MT;u+=gridDim.x){
  int e=ORDER[u/MT],mt=u%MT,count=C[e],off=OFF[e];
  if(count<=16)continue;
  for(int row=0;row<count;row+=256){int rem=min(count-row,256);
   if(rem>CAP||(CAP==128?rem<=64:CAP==256?rem<=128:false))continue;
   float acc[NG][NC/2]={};unsigned af[2][16]={};
   uint64_t desc0=((uint64_t)(smaddr(B)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);
   #pragma unroll 1
   for(int kh0=0;kh0<64;kh0+=2){
    #pragma unroll
    for(int slot=0;slot<2;++slot){int kh=kh0+slot;
    // All WGs have retired the older B slot before any writer reaches it.
    __syncthreads();
    const int*qp=Q+(((unsigned)e*(N/16)+mt*(NC/16)+wg*4+w)*128+kh*2+(l>>4))*64+(l&15)*4;
    uint4 q=__ldg((const uint4*)qp);
    const unsigned char*sp=SC+(((unsigned)e*(N/16)+mt*(NC/16)+wg*4+w)*32+kh/2)*128+(l>>2)*16+(kh&1)*8;
    uint2 sv{};if((l&3)==0)sv=__ldg((const uint2*)sp);sv.x=__shfl_sync(~0u,sv.x,l&~3);sv.y=__shfl_sync(~0u,sv.y,l&~3);unsigned d[16];dq(q.x,q.y,sv.x,d,0x70007000u);dq(q.z,q.w,sv.y,d+8,0x70007000u);
    #pragma unroll
    for(int j=0;j<16;++j){int r=wg*64+w*16+(l>>2)+(j&1)*8,k=(l&3)*2+((j>>1)&1)*8+(j>>2)*16;
     *(unsigned*)(B+slot*NC*64+r*64+(k^((r&7)*8)))=d[j];
    }
    __syncthreads();
#if BR_HOPPER
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif
    #pragma unroll
    for(int g=0;g<NG;++g){int group=row/64+wg+g*WG;
     {int aidx=(slot*NG+g)&1;
      #pragma unroll
      for(int b=0;b<4;++b){const unsigned*ap=group*64<count?A+ax(e,group,kh*4+b,t,groups):zero_fragment+t*4;
       uint4 v=__ldg((const uint4*)ap);af[aidx][b*4]=v.x;af[aidx][b*4+1]=v.y;af[aidx][b*4+2]=v.z;af[aidx][b*4+3]=v.w;}
      keep(af[aidx]);
#if BR_HOPPER
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
      #pragma unroll
      for(int b=0;b<4;++b)mm<NC>(acc[g],af[aidx]+b*4,B+slot*NC*64,b,desc0+slot*NC*8);
     }
#if BR_HOPPER
     // Empty groups preserve the retirement distance for inactive token groups.
     asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 1;":::"memory");
#endif
     keep(af[((slot*NG+g)&1)^1]);
    }
   }}
#if BR_HOPPER
   asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
   keep(af[0]);keep(af[1]);
   #pragma unroll
   for(int g=0;g<NG;++g){int group=row/64+wg+g*WG,r0=group*64+w*16+(l>>2);
    if constexpr(F){float gg=GS[e]*256.f,gu=GS[e+32]*256.f;
     #pragma unroll
     for(int i=0;i<NC/8;i+=2){unsigned lo=0,hi=0;
      #pragma unroll
      for(int j=0;j<4;++j){float a=__bfloat162float(__float2bfloat16_rn(acc[g][4*i+j]*gg)),b=__bfloat162float(__float2bfloat16_rn(acc[g][4*(i+1)+j]*gu));
       __nv_bfloat16 h=__float2bfloat16_rn(a/(1.f+__expf(-a))*b);unsigned bits=*(unsigned short*)&h;
       if(j<2)lo|=bits<<(16*(j&1));else hi|=bits<<(16*(j&1));
      }
      int ch=mt*(NC/2)+(i/2)*8+(l&3)*2,at=w*32+(l>>2)*4+(l&3),word=(ch&8)?2:0;
      if(group*64<count)*(uint2*)(A2+ax(e,group,ch/16,at,groups)+word)=make_uint2(lo,hi);
     }
    }else{float low=0,high=0,gs=GS[e]*256.f;
     #pragma unroll
     for(int j=0;j<NC/2;++j){int r=r0+((j>>1)&1)*8,ch=mt*NC+(j>>2)*8+(l&3)*2+(j&1);
      float z=acc[g][j]*gs*(1.f+__bfloat162float(PNW[ch]));if(r<count)Y[(size_t)(off+r)*4096+ch]=z;
      if(j&2)high+=z*z;else low+=z*z;
     }
     low+=__shfl_xor_sync(~0u,low,1);low+=__shfl_xor_sync(~0u,low,2);high+=__shfl_xor_sync(~0u,high,1);high+=__shfl_xor_sync(~0u,high,2);
     if((l&3)==0){if(r0<count){SS[(size_t)(off+r0)*64+mt*(NC/64)]=low;if constexpr(NC==128)SS[(size_t)(off+r0)*64+mt*2+1]=0;}
      if(r0+8<count){SS[(size_t)(off+r0+8)*64+mt*(NC/64)]=high;if constexpr(NC==128)SS[(size_t)(off+r0+8)*64+mt*2+1]=0;}}
    }
   }
   __syncthreads();
  }
 }
}
}
void rt_pack(std::vector<int64_t>v,int64_t a,int64_t M,int64_t stream){int groups=(M+63)/64;rt::pack<<<32*groups*32,128,0,(cudaStream_t)stream>>>((__nv_bfloat16*)v[3],(int*)v[4],(int*)v[8],(unsigned*)a,M,groups);BR_CUDA_CHECK(cudaGetLastError());}
template<int N,int C,bool F>std::vector<int64_t> rt_resource(){cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncGetAttributes(&a,rt::kernel<N,C,F>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,rt::kernel<N,C,F>,N*2,N*64*2*2));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,N*64*2*2,n};}
std::vector<int64_t> rt_info(int64_t n,int64_t c,int64_t f){
#define RT_INFO(N,C) if(n==N&&c==C)return f?rt_resource<N,C,true>():rt_resource<N,C,false>();
 RT_INFO(64,64) RT_INFO(64,128) RT_INFO(64,256) RT_INFO(128,64) RT_INFO(128,128) RT_INFO(128,256)
#undef RT_INFO
 BR_REQUIRE(false);return {};
}
void rt_run(std::vector<int64_t>v,int64_t a,int64_t a2,int64_t nc,int64_t cap,int64_t first,int64_t M,int64_t grid,int64_t stream){
#define RT_RUN(N,C) if(nc==N&&cap==C){if(first)rt::kernel<N,C,true><<<grid,N*2,N*64*4,(cudaStream_t)stream>>>(BR_ARGS(v),M,(unsigned*)a,(unsigned*)a2);else rt::kernel<N,C,false><<<grid,N*2,N*64*4,(cudaStream_t)stream>>>(BR_ARGS(v),M,(unsigned*)a,(unsigned*)a2);BR_CUDA_CHECK(cudaGetLastError());return;}
 RT_RUN(64,64) RT_RUN(64,128) RT_RUN(64,256) RT_RUN(128,64) RT_RUN(128,128) RT_RUN(128,256)
#undef RT_RUN
 BR_REQUIRE(false);
}
