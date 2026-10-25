// WG0: one TMA warp + three decode warps; WG1/2: independent RS consumers.
namespace ws {
using mwbr::smaddr;using mwbr::dq;using mwbr::keep;
WS_MMA
constexpr int R=2,PB=9216,AB=32768,KT=128;
__device__ __forceinline__ void wait(unsigned b,unsigned phase){
 asm volatile("{.reg .pred p; WS_WAIT%=:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;@!p bra WS_WAIT%=;}"::"r"(b),"r"(phase):"memory");
}
__device__ __forceinline__ void arrive(unsigned b){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(b):"memory");}
template<int ID,int N>__device__ __forceinline__ void sync(){asm volatile("bar.sync %0,%1;"::"n"(ID),"n"(N):"memory");}

template<int MAX,bool F,class Fn>__device__ __forceinline__ void jobs(const int*C,const int*ORDER,uint64_t*ep,int pipe,Fn fn){
 constexpr int MT=F?64:32;int seq=0;
 for(int u=blockIdx.x;u<32*MT;u+=gridDim.x){int e=ORDER[u/MT],mt=u%MT,n=C[e];
  if(n<=16)continue;
  for(int row=0;row<n;row+=256){int rem=min(n-row,256);if(rem<=MAX-64||rem>MAX)continue;
   if(!pipe)wait(smaddr(ep),(seq/32)&1);
   fn(e,mt,row,rem,n,seq);seq+=32;
  }
 }
}

template<int MAX,int NT,bool F,int S>__device__ __forceinline__ void consume(
 const float*GS,const int*OFF,const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*HP,
 int e,int mt,int row,int count,int firstseq,unsigned*afsm,__nv_bfloat16*xs,
 uint64_t*empty,uint64_t*afull,uint64_t*xfull){
 int cid=(threadIdx.x-128)/128,t=threadIdx.x&127,w=t>>5,l=t&31,offset=OFF[e];mt=mt*2+cid;
 float acc[NT/2]={};unsigned af[2][16]={};
 uint64_t desc0=((uint64_t)(smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);
 for(int k0=0;k0<32;k0+=S){
  #pragma unroll
  for(int st=0;st<S;++st){int k=k0+st,seq=firstseq+k;
   wait(smaddr(afull+st),(seq/S)&1);wait(smaddr(xfull+st),(seq/S)&1);
   #pragma unroll
   for(int sub=0;sub<2;++sub){
    #pragma unroll
    for(int j=0;j<4;++j){uint4 v=*(const uint4*)(afsm+(st*AB+cid*16384+sub*8192+t*64+j*16)/4);
     af[sub][4*j]=v.x;af[sub][4*j+1]=v.y;af[sub][4*j+2]=v.z;af[sub][4*j+3]=v.w;
    }
    keep(af[sub]);
#if BR_HOPPER
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
    #pragma unroll
    for(int b=0;b<4;++b)mm<NT>(acc,af[sub]+b*4,xs+st*MAX*KT+sub*NT*64,b,desc0+(st*MAX*KT+sub*NT*64)/8);
#if BR_HOPPER
    asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 1;":::"memory");
#endif
    keep(af[sub^1]);
    if constexpr(S>1)if(sub==0&&k>0){if(cid==0)sync<4,128>();else sync<5,128>();if(t==0)arrive(smaddr(empty+(st+S-1)%S));}
   }
   if constexpr(S==1){
#if BR_HOPPER
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
    if(cid==0)sync<4,128>();else sync<5,128>();if(t==0)arrive(smaddr(empty+st));
   }
  }
 }
#if BR_HOPPER
 asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
 keep(af[0]);keep(af[1]);
 if constexpr(S>1){if(cid==0)sync<4,128>();else sync<5,128>();if(t==0)arrive(smaddr(empty+(firstseq+31)%S));}
 // Stage release precedes epilogue: producer/decode may stream the next tile.
 if constexpr(F){float gg=GS[e]*256.f,gu=GS[e+32]*256.f;int ch=mt*32+w*8+(l>>2);
  #pragma unroll
  for(int j=0;j<NT/2;++j)if(((j>>1)&1)==0){int r=row+(l&3)*2+(j&1)+(j>>2)*8;
   float g=__bfloat162float(__float2bfloat16_rn(acc[j]*gg)),u=__bfloat162float(__float2bfloat16_rn(acc[j+2]*gu));
   if(r<count)ACT[(size_t)(offset+r)*4096+ch]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);
  }
 }else{float gs=GS[e]*256.f;
  #pragma unroll
  for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8,ch=mt*64+w*16+(l>>2)+((j>>1)&1)*8;
   float z=acc[j]*gs*(1.f+__bfloat162float(PNW[ch]));acc[j]=z;if(r<count)Y[(size_t)(offset+r)*4096+ch]=z;
  }
  #pragma unroll
  for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2],b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];
   #pragma unroll
   for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}
   if(l<4){int r=row+i*8+2*l;if(r<count)HP[(size_t)(offset+r)*256+mt*4+w]=a;if(r+1<count)HP[(size_t)(offset+r+1)*256+mt*4+w]=b;}
  }
 }
}

template<int MAX,bool F>__global__ void __launch_bounds__(384,1) kernel(
 const int*Q,const unsigned char*SC,const float*GS,const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,const unsigned char*pack,float*HP,int pipe,const __grid_constant__ CUtensorMap maplo,const __grid_constant__ CUtensorMap maphi){
 constexpr int S=BR_HOPPER?2:(MAX<=64?2:1),MT=F?64:32;
 extern __shared__ __align__(1024) unsigned char sm[];
 unsigned*afsm=(unsigned*)sm;auto*xs=(__nv_bfloat16*)(sm+S*AB);auto*raw=sm+S*(AB+MAX*KT*2);
 __shared__ __align__(8) uint64_t bars[2*R+3*S+1];
 auto*rf=bars;auto*re=rf+R;auto*af= re+R;auto*xf=af+S;auto*empty=xf+S;auto*ep=empty+S;
 if(threadIdx.x==0){
  for(int i=0;i<2*R+3*S+1;++i)asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(smaddr(bars+i)),"r"(i>=2*R+2*S?2:1):"memory");
  if(!pipe){arrive(smaddr(ep));arrive(smaddr(ep));}
  for(int i=0;i<R;++i)arrive(smaddr(re+i));
  for(int i=0;i<S;++i){arrive(smaddr(empty+i));arrive(smaddr(empty+i));}
#if BR_HOPPER
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
#endif
 }
 __syncthreads();int tid=threadIdx.x,wg=__shfl_sync(~0u,tid/128,0);
 if(wg==0){
#if BR_HOPPER
  asm volatile("setmaxnreg.dec.sync.aligned.u32 88;":::"memory");
#endif
 if(tid<32){
  jobs<MAX,F>(C,ORDER,ep,pipe,[&](int e,int mt,int row,int rem,int n,int first){int nt=rem<=MAX-32?MAX-32:MAX;
   for(int k=0;k<32;++k){int seq=first+k,rs=seq%R,st=seq%S;wait(smaddr(re+rs),(seq/R)&1);wait(smaddr(empty+st),(seq/S)&1);
#if BR_HOPPER
    if(tid==0){const unsigned char*p=pack+(((size_t)e*MT+mt)*32+k)*PB;const CUtensorMap*map=rem<=MAX-32?&maplo:&maphi;
     asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(smaddr(rf+rs)),"r"(PB):"memory");
     asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smaddr(raw+rs*PB)),"l"(p),"r"(PB),"r"(smaddr(rf+rs)):"memory");
     asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(smaddr(xf+st)),"r"(nt*KT*2):"memory");
     #pragma unroll
     for(int h=0;h<2;++h)asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
      ::"r"(smaddr(xs+st*MAX*KT+h*nt*64)),"l"(map),"r"(k*KT+h*64),"r"(OFF[e]+row),"r"(smaddr(xf+st)):"memory");
    }
#else
    // TMA stand-in: raw packet read globally; cooperative cp.async for X.
    for(int t=tid;t<128;t+=32)for(int j=0;j<nt/16;++j){int r=t/8+j*16,z=(t&7)*8;
     for(int h=0;h<2;++h)mwbr::async_copy(xs+st*MAX*KT+h*nt*64+r*64+(z^((r&7)*8)),X+(size_t)(OFF[e]+row+r)*4096+k*KT+h*64+z,16,row+r<n);
    }
    asm volatile("cp.async.commit_group;cp.async.wait_group 0;":::"memory");__syncwarp();if(tid==0){arrive(smaddr(rf+rs));arrive(smaddr(xf+st));}
#endif
   }
  });
 }else{int dt=tid-32;
  jobs<MAX,F>(C,ORDER,ep,pipe,[&](int e,int mt,int row,int rem,int n,int first){
   for(int k=0;k<32;++k){int seq=first+k,st=seq%S,rs=seq%R;wait(smaddr(rf+rs),(seq/R)&1);wait(smaddr(empty+st),(seq/S)&1);
#if BR_HOPPER
    const unsigned char*p=raw+rs*PB;
#else
    const unsigned char*p=pack+(((size_t)e*MT+mt)*32+k)*PB;
#endif
    for(int t=dt;t<128;t+=96){int w=t>>5,l=t&31;
     #pragma unroll
     for(int out=0;out<2;++out){uint4 sv=*(const uint4*)(p+out*4608+4096+(w*8+(l>>2))*16);
      #pragma unroll
      for(int sub=0;sub<2;++sub){uint4 q=*(const uint4*)(p+out*4608+sub*2048+t*16);unsigned a[16];dq(q.x,q.y,sub?sv.z:sv.x,a,0x70007000u);dq(q.z,q.w,sub?sv.w:sv.y,a+8,0x70007000u);
       #pragma unroll
       for(int j=0;j<4;++j)*(uint4*)(afsm+(st*AB+out*16384+sub*8192+t*64+j*16)/4)=make_uint4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]);
      }
     }
    }
    sync<3,96>();if(dt==0){arrive(smaddr(af+st));arrive(smaddr(re+rs));}
   }
  });
 }
 }else{
#if BR_HOPPER
  asm volatile("setmaxnreg.inc.sync.aligned.u32 208;":::"memory");
#endif
  jobs<MAX,F>(C,ORDER,ep,pipe,[&](int e,int mt,int row,int rem,int n,int first){
   if(rem<=MAX-32)consume<MAX,MAX-32,F,S>(GS,OFF,PNW,ACT,Y,HP,e,mt,row,n,first,afsm,xs,empty,af,xf);
   else consume<MAX,MAX,F,S>(GS,OFF,PNW,ACT,Y,HP,e,mt,row,n,first,afsm,xs,empty,af,xf);
   if(!pipe)if((tid&127)==0)arrive(smaddr(ep));
  });
 }
 __syncthreads();
 if(tid==0)for(int i=0;i<2*R+3*S+1;++i)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(smaddr(bars+i)):"memory");
}
__global__ void gather(const __nv_bfloat16*x,__nv_bfloat16*gx,const int*C,const int*OFF,const int*SLOTS,int M){
 int z=blockIdx.x;if(z>=OFF[32])return;int e=0;
 for(int d=16;d;d>>=1)if(OFF[e+d]<=z)e+=d;
 int r=z-OFF[e];if(r>=C[e]||C[e]<=16)return;int src=SLOTS[e*M+r]/8;
 for(int k=threadIdx.x*8;k<4096;k+=128*8)*(uint4*)(gx+(size_t)z*4096+k)=*(const uint4*)(x+(size_t)src*4096+k);
}
WS_FINISH
}
template<int N,bool F>std::vector<int64_t> ws_resource(){int dev,major;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));int s=major==9?2:(N<=64?2:1),bytes=s*(ws::AB+N*256)+(major==9?ws::R*ws::PB:0);cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncSetAttribute(ws::kernel<N,F>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));BR_CUDA_CHECK(cudaFuncGetAttributes(&a,ws::kernel<N,F>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,ws::kernel<N,F>,384,bytes));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,n};}
std::vector<int64_t> ws_info(int64_t n,int64_t f,int64_t pipe){
#define WS_INFO(N) if(n==N)return f?ws_resource<N,true>():ws_resource<N,false>();
 WS_INFO(64) WS_INFO(128) WS_INFO(192) WS_INFO(256)
#undef WS_INFO
 BR_REQUIRE(false);return {};
}
template<int N,bool F>void ws_launch(std::vector<int64_t>&v,int64_t packet,int64_t hp,int M,int grid,int bytes,cudaStream_t stream,int pipe){CUtensorMap maps[2]{};int dev,major;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));if(major==9){maps[0]=oc_map((void*)v[3],M*8+32*128,N-32);maps[1]=oc_map((void*)v[3],M*8+32*128,N);}ws::kernel<N,F><<<grid,384,bytes,stream>>>(BR_ARGS(v),M,(unsigned char*)packet,(float*)hp,pipe,maps[0],maps[1]);BR_CUDA_CHECK(cudaGetLastError());}
void ws_run(std::vector<int64_t>v,int64_t packet,int64_t hp,int64_t n,int64_t pipe,int64_t first,int64_t M,int64_t grid,int64_t stream){
 int dev,major;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));
#define WS_RUN(N) if(n==N){int s=major==9?2:(N<=64?2:1),bytes=s*(ws::AB+N*256)+(major==9?ws::R*ws::PB:0);if(first)ws_launch<N,true>(v,packet,hp,M,grid,bytes,(cudaStream_t)stream,pipe);else ws_launch<N,false>(v,packet,hp,M,grid,bytes,(cudaStream_t)stream,pipe);return;}
 WS_RUN(64) WS_RUN(128) WS_RUN(192) WS_RUN(256)
#undef WS_RUN
 BR_REQUIRE(false);
}
void ws_finish(std::vector<int64_t>v,int64_t c,int64_t hp,int64_t M,int64_t start,int64_t stream){ws::finish<4><<<M*4,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(float*)v[2],(int*)v[3],(float*)v[4],(float*)v[5],(__nv_bfloat16*)v[6],M,start,(int*)c,(float*)hp);BR_CUDA_CHECK(cudaGetLastError());}

void ws_gather(std::vector<int64_t>v,int64_t gx,int64_t M,int64_t stream){ws::gather<<<M*8+32*32,128,0,(cudaStream_t)stream>>>((__nv_bfloat16*)v[3],(__nv_bfloat16*)gx,(int*)v[4],(int*)v[5],(int*)v[8],M);BR_CUDA_CHECK(cudaGetLastError());}
