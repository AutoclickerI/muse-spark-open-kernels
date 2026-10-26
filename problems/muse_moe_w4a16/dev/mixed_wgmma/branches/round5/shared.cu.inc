// Two N32 token groups share a raw weight packet; SS also shares its decode.
namespace tokpair {
using mwbr::smaddr;using mwbr::dq;using mwbr::keep;
constexpr int S=2,QB=4096,SB=512,NT=32,KT=128,BASE=10240,XB=32768,AB=32768;
template<bool SHARE>__device__ __forceinline__ void group(float*c,const unsigned*a,const __nv_bfloat16*xs,uint64_t ad,uint64_t bd){
 #pragma unroll
 for(int b=0;b<4;++b){
  if constexpr(!SHARE)mwbr::mma<32>(c,a+b*4,xs,b,bd);
  else {
#if BR_HOPPER
   uint64_t aa=ad+b*2,bb=bd+b*2;
   asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},%16,%17,p,1,1,0,0;}"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]):"l"(aa),"l"(bb));
#else
   int t=threadIdx.x&127,l=t&31,w=t>>5;unsigned ar[4];
   const unsigned*as=(const unsigned*)__cvta_shared_to_generic((ad&0x3fff)*16);
   #pragma unroll
   for(int j=0;j<4;++j){int r=w*16+(l>>2)+(j&1)*8,k=b*16+(l&3)*2+(j>>1)*8;ar[j]=as[(r*64+(k^((r&7)*8)))/2];}
   mwbr::mma<32>(c,ar,xs,b,bd);
#endif
  }
 }
#if BR_HOPPER
 asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 1;":::"memory");
#endif
}

template<bool F,bool SHARE>__global__ void __launch_bounds__(256,2) kernel(
 const int*Q,const unsigned char*SC,const float*GS,const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M){
 extern __shared__ __align__(1024) unsigned char sm[];
 auto*qs=sm;auto*sf=sm+S*QB;auto*xs=(__nv_bfloat16*)(sm+BASE);auto*as=(__nv_bfloat16*)(sm+BASE+XB);
 float*red=(float*)sm;int t=threadIdx.x&127,w=t>>5,l=t&31,wg=threadIdx.x>>7;
 constexpr int N=F?8192:4096,MT=N/64;
 for(int u=blockIdx.x;u<32*MT;u+=gridDim.x){int e=ORDER[u/MT],mt=u%MT,count=C[e],offset=OFF[e];if(count<=16)continue;
  for(int row0=0;row0<count;row0+=64){int row=row0+wg*32;float acc[16]={};unsigned af[2][16]={};int src[2];
   #pragma unroll
   for(int j=0;j<2;++j){int r=row+t/8+j*16;src[j]=r<count?(F?SLOTS[e*M+r]/8:offset+r):-1;}
   const int*qg=Q+((unsigned)e*(N/16)+mt*4+(t>>4))*128*64+(t&15)*4;
   const unsigned char*sg=SC+(((unsigned)e*(N/16)+mt*4+(t-64)/8)*128*32+((t-64)%8)*16);
   const __nv_bfloat16*xg[2];
   #pragma unroll
   for(int j=0;j<2;++j)xg[j]=X+(size_t)max(0,src[j])*4096+(t&7)*8;
   auto issue=[&](int st){
    if(wg==0&&t<64){unsigned dest=smaddr(qs+st*QB+(t>>4)*512+(t&15)*16);
     #pragma unroll
     for(int g=0;g<4;++g)asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dest+(g>>1)*2048+(g&1)*256),"l"(qg+g*64):"memory");qg+=256;}
    if(wg==0&&t>=64&&t<96){asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(smaddr(sf+st*SB+(t-64)*16)),"l"(sg):"memory");sg+=128;}
    #pragma unroll
    for(int h=0;h<2;++h){
     #pragma unroll
     for(int j=0;j<2;++j){int r=t/8+j*16,k=(t&7)*8;auto*d=xs+(st*2+wg)*NT*KT+h*NT*64+r*64+(k^((r&7)*8));mwbr::async_copy(d,xg[j]+h*64,16,src[j]>=0);}
    }
    #pragma unroll
    for(int j=0;j<2;++j)xg[j]+=KT;
    asm volatile("cp.async.commit_group;":::"memory");
   };
   uint64_t xd=((uint64_t)(smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);
   uint64_t ad=((uint64_t)(smaddr(as)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);
   issue(0);
   for(int k0=0;k0<32;k0+=S){
    #pragma unroll
    for(int st=0;st<S;++st){int k=k0+st;asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
     if constexpr(SHARE)if(wg==0){uint4 sv=*(const uint4*)(sf+st*SB+(w*8+(l>>2))*16);
      #pragma unroll
      for(int sub=0;sub<2;++sub){uint4 q=*(const uint4*)(qs+st*QB+sub*2048+t*16);dq(q.x,q.y,sub?sv.z:sv.x,af[sub],0x70007000u);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,0x70007000u);
       if constexpr(SHARE){
        #pragma unroll
        for(int b=0;b<4;++b){
         #pragma unroll
         for(int j=0;j<4;++j){int r=w*16+(l>>2)+(j&1)*8,z=b*16+(l&3)*2+(j>>1)*8;*(unsigned*)(as+(st*2+sub)*64*64+r*64+(z^((r&7)*8)))=af[sub][4*b+j];}
        }
       }
      }
     }
     if constexpr(SHARE)__syncthreads();
#if BR_HOPPER
     asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif
     #pragma unroll
     for(int sub=0;sub<2;++sub){
      if constexpr(!SHARE){uint4 sv=*(const uint4*)(sf+st*SB+(w*8+(l>>2))*16),q=*(const uint4*)(qs+st*QB+sub*2048+t*16);dq(q.x,q.y,sub?sv.z:sv.x,af[sub],0x70007000u);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,0x70007000u);}
      if constexpr(!SHARE)keep(af[sub]);int xi=(st*2+wg)*NT*KT+sub*NT*64;
#if BR_HOPPER
      asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
      group<SHARE>(acc,af[sub],xs+xi,ad+(st*2+sub)*512,xd+xi/8);if constexpr(!SHARE)keep(af[sub^1]);
      if(sub==0&&k+1<32)issue((st+1)%S);
     }
    }
   }
#if BR_HOPPER
   asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
   if constexpr(!SHARE){keep(af[0]);keep(af[1]);}
   if constexpr(F){float gg=GS[e]*256.f,gu=GS[e+32]*256.f;int ch=mt*32+w*8+(l>>2);
    #pragma unroll
    for(int j=0;j<16;++j)if(((j>>1)&1)==0){int r=row+(l&3)*2+(j&1)+(j>>2)*8;float g=__bfloat162float(__float2bfloat16_rn(acc[j]*gg)),v=__bfloat162float(__float2bfloat16_rn(acc[j+2]*gu));if(r<count)ACT[(size_t)(offset+r)*4096+ch]=__float2bfloat16_rn(g/(1.f+__expf(-g))*v);}
   }else{float gs=GS[e]*256.f;
    #pragma unroll
    for(int j=0;j<16;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8,ch=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float z=acc[j]*gs*(1.f+__bfloat162float(PNW[ch]));acc[j]=z;if(r<count)Y[(size_t)(offset+r)*4096+ch]=z;}
    #pragma unroll
    for(int i=0;i<4;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2],b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];
     #pragma unroll
     for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}if(l<4){red[(wg*4+w)*32+i*8+2*l]=a;red[(wg*4+w)*32+i*8+2*l+1]=b;}}
    __syncthreads();if(t<32&&row+t<count){float sum=0;
     #pragma unroll
     for(int j=0;j<4;++j)sum+=red[(wg*4+j)*32+t];SS[(size_t)(offset+row+t)*64+mt]=sum;}
   }
   __syncthreads();
  }
 }
}
}
template<bool F,bool SHARE>std::vector<int64_t> tp_resource(){int bytes=tokpair::BASE+tokpair::XB+(SHARE?tokpair::AB:0);BR_CUDA_CHECK(cudaFuncSetAttribute(tokpair::kernel<F,SHARE>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncGetAttributes(&a,tokpair::kernel<F,SHARE>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,tokpair::kernel<F,SHARE>,256,bytes));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,n};}
std::vector<int64_t> tp_info(int64_t first,int64_t share){if(first)return share?tp_resource<true,true>():tp_resource<true,false>();return share?tp_resource<false,true>():tp_resource<false,false>();}
void tp_run(std::vector<int64_t>v,int64_t first,int64_t share,int64_t grid,int64_t M,int64_t stream){
#define TP_GO(F,S) tokpair::kernel<F,S><<<grid,256,tokpair::BASE+tokpair::XB+(S?tokpair::AB:0),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M)
 if(first){if(share){TP_GO(true,true);}else{TP_GO(true,false);}}else{if(share){TP_GO(false,true);}else{TP_GO(false,false);}}
#undef TP_GO
 BR_CUDA_CHECK(cudaGetLastError());
}
