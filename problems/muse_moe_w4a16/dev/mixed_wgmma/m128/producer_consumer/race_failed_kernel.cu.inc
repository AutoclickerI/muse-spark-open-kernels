namespace pc {
__device__ __forceinline__ void pbar(){asm volatile("bar.sync 1,128;":::"memory");}
__device__ __forceinline__ void cbar(){asm volatile("bar.sync 2,128;":::"memory");}
__device__ __forceinline__ void init_bar(uint64_t*b){asm volatile("mbarrier.init.shared::cta.b64 [%0],4;"::"r"(smaddr(b)):"memory");}
__device__ __forceinline__ void arrive(uint64_t*b){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(smaddr(b)):"memory");}
__device__ __forceinline__ void wait_bar(uint64_t*b,unsigned phase){
 unsigned ok;
 do{asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2;selp.u32 %0,1,0,p;}" :"=r"(ok):"r"(smaddr(b)),"r"(phase):"memory");}while(!ok);
}

template<int CAP,int S>struct Layout{
 static constexpr int RAW=3072,A=RAW,B=A+8192,STRIDE=B+CAP*128;
 static constexpr int BYTES=S*STRIDE;
};

// One producer warpgroup owns all packed reads, BF16 scatter, and activation
// copies. Raw, A, and B have independent slots in the same ring. A is converted
// exactly once per expert/channel/K64 tile, outside every token-subgroup loop.
template<int CAP,int WIDTH,int S>
__device__ __forceinline__ void produce(const int*Q,const unsigned char*SF,const __nv_bfloat16*X,
 const int*SLOTS,int M,int e,int offset,int count,int mt,int N,unsigned char*sm,uint64_t*full,uint64_t*empty){
 using L=Layout<CAP,S>;int t=threadIdx.x,w=t>>5,l=t&31;
 int src[WIDTH/16];
 #pragma unroll
 for(int j=0;j<WIDTH/16;++j){int r=t/8+j*16;src[j]=r<count?(N==8192?SLOTS[e*M+r]/8:offset+r):-1;}
 auto copy=[&](int step){
  int st=step%S;unsigned ph=(step/S)&1;
  // Empty barriers start pending phase0: phase1 denotes the unused slot.
  wait_bar(empty+st,ph^1);unsigned char*slot=sm+st*L::STRIDE;
  unsigned char*q=slot,*sf=slot+2048;auto*as=(__nv_bfloat16*)(slot+L::A);auto*bs=(__nv_bfloat16*)(slot+L::B);
  #pragma unroll
  for(int g=0;g<2;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+step*2+g;
   if(l<16)async_copy(q+(g*128+w*32)*8+l*16,Q+z*64+l*4,16,1);
   if(l<8)async_copy(sf+g*128+w*32+l*4,SF+z*32+l*4,4,1);
  }
  #pragma unroll
  for(int j=0;j<WIDTH/16;++j){int r=t/8+j*16,k=(t&7)*8;
   async_copy(bs+r*64+(k^((r&7)*8)),X+(size_t)max(src[j],0)*4096+step*64+k,16,src[j]>=0);
  }
  asm volatile("cp.async.commit_group;":::"memory");
 };
 #pragma unroll
 for(int step=0;step<S-1;++step)copy(step);
 #pragma unroll 1
 for(int step=0;step<64;++step){
  if(step+S-1<64)copy(step+S-1);
  if(step==63)asm volatile("cp.async.wait_group 0;":::"memory");
  else if constexpr(S==2)asm volatile("cp.async.wait_group 1;":::"memory");
  else if(step==62)asm volatile("cp.async.wait_group 1;":::"memory");
  else asm volatile("cp.async.wait_group 2;":::"memory");
  // Q and scale words used by this warp were copied by this warp. The sole
  // producer-group barrier is after decode, when all A/B writers publish.
  __syncwarp();int st=step%S;unsigned char*slot=sm+st*L::STRIDE;
  unsigned char*q=slot,*sf=slot+2048;auto*as=(__nv_bfloat16*)(slot+L::A);
  unsigned af[16];
  #pragma unroll
  for(int g=0;g<2;++g){uint2 v=*(const uint2*)(q+(g*128+t)*8);unsigned s=*(const unsigned*)(sf+g*128+w*32+(l>>2)*4);dq(v.x,v.y,s,af+g*8);}
  #pragma unroll
  for(int atom=0;atom<4;++atom){
   #pragma unroll
   for(int j=0;j<4;++j){int r=w*16+(l>>2)+(j&1)*8,k=atom*16+(l&3)*2+(j>>1)*8;
    *(unsigned*)(as+r*64+(k^((r&7)*8)))=af[atom*4+j];}
  }
  // Each producer warp publishes its own generic/cp.async writes. Four
  // arrivals complete full only after all four warps reach this point.
  asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncwarp();if(l==0)arrive(full+st);
 }
}

template<int CAP,int WIDTH,int NLIMIT,int S>
__device__ __forceinline__ void consume(unsigned char*sm,uint64_t*full,uint64_t*empty,float*acc){
 using L=Layout<CAP,S>;int t=threadIdx.x&127;
 #pragma unroll 1
 for(int step=0;step<64;++step){
  int st=step%S;unsigned ph=(step/S)&1;wait_bar(full+st,ph);
  auto*as=(__nv_bfloat16*)(sm+st*L::STRIDE+L::A);auto*bs=(__nv_bfloat16*)(sm+st*L::STRIDE+L::B);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
  #pragma unroll
  for(int group=0;group<(WIDTH+NLIMIT-1)/NLIMIT;++group){constexpr int NW=WIDTH<NLIMIT?WIDTH:NLIMIT;
   #pragma unroll
   for(int atom=0;atom<4;++atom)ss_mma<NW>(acc+group*(NLIMIT/2),as,bs+group*NLIMIT*64,atom);
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
  // Both shared descriptors remain live until wait0. The producer can fill
  // other slots while this group executes, but never overwrite this one.
  __syncwarp();if((t&31)==0)arrive(empty+st);
 }
}

template<int CAP,int NLIMIT,int S,bool FIRST>
__global__ void __launch_bounds__(256,2) heavy(const int*Q,const unsigned char*SF,const float*GS,
 const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M){
 using L=Layout<CAP,S>;
 extern __shared__ __align__(1024) unsigned char raw[];
 unsigned char*sm=(unsigned char*)(((uintptr_t)raw+1023)&~(uintptr_t)1023);
 __shared__ uint64_t full[S],empty[S];
 __shared__ int epilogue_meta[2];
 __shared__ __nv_bfloat16 epi[FIRST?32*CAP:1];__shared__ float rowsum[FIRST?1:4*CAP];
 constexpr int N=FIRST?8192:4096,MT=N/64;int tx=threadIdx.x,t=tx&127,w=t>>5,l=t&31;
 int groups=TASK[32];
 for(int unit=blockIdx.x;unit<groups*MT;unit+=gridDim.x){
  int group=unit%groups,mt=unit/groups,ix=0;
  #pragma unroll
  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;
  int e=ORDER[ix],count=C[e],offset=OFF[e];
  if(tx==0){
   epilogue_meta[0]=count;epilogue_meta[1]=offset;
   #pragma unroll
   for(int s=0;s<S;++s){init_bar(full+s);init_bar(empty+s);}
   asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  __syncthreads();
  // Select N outside the K loop. On M128 each touched expert has at most128
  // distinct token routes, so there is exactly one task for all its rows.
  if(tx<128){
   if(count<=16)produce<CAP,16,S>(Q,SF,X,SLOTS,M,e,offset,count,mt,N,sm,full,empty);
   else if(count<=32)produce<CAP,32,S>(Q,SF,X,SLOTS,M,e,offset,count,mt,N,sm,full,empty);
   else if constexpr(CAP>=64){
    if(count<=64)produce<CAP,64,S>(Q,SF,X,SLOTS,M,e,offset,count,mt,N,sm,full,empty);
    else produce<CAP,128,S>(Q,SF,X,SLOTS,M,e,offset,count,mt,N,sm,full,empty);
   }
  }else{
   float acc[CAP/2]={};
   if(count<=16)consume<CAP,16,NLIMIT,S>(sm,full,empty,acc);
   else if(count<=32)consume<CAP,32,NLIMIT,S>(sm,full,empty,acc);
   else if constexpr(CAP>=64){
    if(count<=64)consume<CAP,64,NLIMIT,S>(sm,full,empty,acc);
    else consume<CAP,128,NLIMIT,S>(sm,full,empty,acc);
   }
   // The output row metadata is needed only after the K loop. Explicit
   // reloads prevent ptxas from spilling it across the live accumulators.
   asm volatile("ld.shared.u32 %0,[%2];ld.shared.u32 %1,[%3];":"=r"(count),"=r"(offset):"r"(smaddr(epilogue_meta)),"r"(smaddr(epilogue_meta+1)):"memory");
   if constexpr(FIRST){
    #pragma unroll
    for(int j=0;j<CAP/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8,c=w*16+(l>>2)+((j>>1)&1)*8;
     if(c>=32)epi[(c-32)*CAP+r]=__float2bfloat16_rn(acc[j]*GS[e+32]*256.f);}
    cbar();
    #pragma unroll
    for(int j=0;j<CAP/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8,c=w*16+(l>>2)+((j>>1)&1)*8;
     if(c<32&&r<count){float g=__bfloat162float(__float2bfloat16_rn(acc[j]*GS[e]*256.f));float u=__bfloat162float(epi[c*CAP+r]);
      ACT[(size_t)(offset+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);}}
   }else{
    #pragma unroll
    for(int j=0;j<CAP/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8,c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;
     float z=acc[j]*GS[e]*256.f*(1.f+__bfloat162float(PNW[c]));acc[j]=z;if(r<count)Y[(size_t)(offset+r)*4096+c]=z;}
    #pragma unroll
    for(int i=0;i<CAP/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2],b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];
     #pragma unroll
     for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}
     if(l<4){rowsum[w*CAP+i*8+2*l]=a;rowsum[w*CAP+i*8+2*l+1]=b;}}
    cbar();
    for(int r=t;r<CAP;r+=128)if(r<count){float sum=0;
     #pragma unroll
     for(int ww=0;ww<4;++ww)sum+=rowsum[ww*CAP+r];SS[(size_t)(offset+r)*64+mt]=sum;}
   }
  }
  // All producers and consumers finish this expert/channel before barriers
  // are reinitialized for the persistent CTA's next task.
  __syncthreads();
  if(tx==0){
   #pragma unroll
   for(int s=0;s<S;++s){
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];mbarrier.inval.shared::cta.b64 [%1];"::"r"(smaddr(full+s)),"r"(smaddr(empty+s)):"memory");
   }
  }
 }
}
}

#define PC_CASES(F) F(0,32,2) F(1,64,2) F(2,128,2) F(3,64,3)
int64_t pc_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
 #define INIT(ID,NL,S) {auto init=[&]<int CAP,bool F>(){WX_CUDA_CHECK(cudaFuncSetAttribute(pc::heavy<CAP,NL,S,F>,cudaFuncAttributeMaxDynamicSharedMemorySize,pc::Layout<CAP,S>::BYTES+1023));};init.template operator()<32,true>();init.template operator()<32,false>();init.template operator()<128,true>();init.template operator()<128,false>();}
 PC_CASES(INIT)
 #undef INIT
 return opt;
}
std::vector<int64_t> pc_info(int64_t which,int64_t cap,int64_t first){
 #define INFO(ID,NL,S) case ID:{auto get=[&]<int CAP,bool F>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,pc::heavy<CAP,NL,S,F>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,pc::heavy<CAP,NL,S,F>,256,pc::Layout<CAP,S>::BYTES+1023));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,pc::Layout<CAP,S>::BYTES+1023,blocks};};if(cap==32){if(first)return get.template operator()<32,true>();return get.template operator()<32,false>();}if(first)return get.template operator()<128,true>();return get.template operator()<128,false>();}
 switch(which){PC_CASES(INFO)default:WX_CHECK(false);}
 #undef INFO
}
void pc_run(std::vector<int64_t>v,int64_t first,int64_t which,int64_t grid,int64_t M,int64_t stream){
 #define RUN(ID,NL,S) case ID:{auto go=[&]<int CAP,bool F>(){pc::heavy<CAP,NL,S,F><<<grid,256,pc::Layout<CAP,S>::BYTES+1023,(cudaStream_t)stream>>>( \
 (int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[14],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],M);};if(M<=32){if(first)go.template operator()<32,true>();else go.template operator()<32,false>();}else{if(first)go.template operator()<128,true>();else go.template operator()<128,false>();}break;}
 switch(which){PC_CASES(RUN)default:WX_CHECK(false);}
 #undef RUN
 WX_CUDA_CHECK(cudaGetLastError());
}
