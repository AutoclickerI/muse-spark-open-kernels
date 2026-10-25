namespace mwbr {
struct UqArgs {
 const int*q;const unsigned char*sf;const float*gs;const __nv_bfloat16*x;
 const int*c,*off,*order,*task,*slots;const __nv_bfloat16*pnw;
 __nv_bfloat16*act;float*y,*ss;
};
// q state: head13, ready-mask, live-experts, head2[32], done13[32],
// done2[32], head of separate PHASE W2 queue. Route resets all 100 words.
__device__ __forceinline__ int uq_acquire(const int*p){int v;
 asm volatile("ld.acquire.gpu.global.s32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}

__device__ __forceinline__ void uq_do(UqArgs a,int e,int row,int mt,int M,bool first){
#define UQ_TILE(NT) uq_tile<NT>(a.q,a.sf,a.gs,a.x,a.c,a.off,a.order,a.task,a.slots,a.pnw,a.act,a.y,a.ss,M,0x70007000u,e,row,mt,first)
 int rem=a.c[e]-row;
 if(rem<=8){UQ_TILE(8);}else if(rem<=16){UQ_TILE(16);}else{UQ_TILE(32);}
#undef UQ_TILE
}

template<int MODE>__global__ void __launch_bounds__(128,2) uq_queue(UqArgs a,UqArgs b,int*st,int M){
 __shared__ int work[5];
 const int tid=threadIdx.x,groups=a.task[32];
 const int total=groups*(MODE==1?64:128);
 while(true){
  if(tid==0){
   int e=-1,row=0,mt=0,pass=MODE==1,unit=-1;
   if constexpr(MODE==2){
    // Scan ready experts in a CTA-dependent order, spreading atomic contention.
    unsigned mask=(unsigned)uq_acquire(st+1);
    if(mask){int rot=blockIdx.x&31;unsigned rm=__funnelshift_r(mask,mask,rot);int r=__ffs(rm)-1;e=(r+rot)&31;
     int n2=((a.c[e]+31)/32)*64,ix=atomicAdd(st+3+e,1);
     if(ix<n2){row=(ix/64)*32;mt=ix%64;pass=1;unit=0;}
     else{atomicAnd((unsigned*)(st+1),~(1u<<e));e=-1;}
    }
   }
   if(unit<0){
    int*head=MODE==1?st+99:st;
    // CAS avoids millions of overshoot atomics while waiting for a producer.
    int ix=uq_acquire(head);
    if(ix<total){ix=atomicAdd(head,1);if(ix<total)unit=ix;}
    if(unit>=0){
     int group=unit%groups;mt=unit/groups;int j=0;
     #pragma unroll
     for(int d=16;d;d>>=1)if(a.task[j+d]<=group)j+=d;
     e=a.order[j];row=(group-a.task[j])*32;pass=MODE==1;
    }else if(MODE!=2||uq_acquire(st+2)==0)e=-2;
   }
   work[0]=e;work[1]=row;work[2]=mt;work[3]=pass;
  }
  __syncthreads();
  int e=work[0],row=work[1],mt=work[2],pass=work[3];
  if(e==-2)break;
  if(e<0){
#if __CUDA_ARCH__>=700
   __nanosleep(64);
#endif
   continue;
  }
  if constexpr(MODE==0)uq_do(a,e,row,mt,M,true);
  else if constexpr(MODE==1)uq_do(b,e,row,mt,M,false);
  else{
   UqArgs selected=pass?b:a;uq_do(selected,e,row,mt,M,!pass);
   // All writers release their own stores before CTA 0 publishes the expert.
   __threadfence();__syncthreads();
   if(tid==0){
    int nb=(a.c[e]+31)/32;
    if(pass){if(atomicAdd(st+67+e,1)==nb*64-1)atomicSub(st+2,1);}
    else{if(atomicAdd(st+35+e,1)==nb*128-1)atomicOr((unsigned*)(st+1),1u<<e);}
   }
  }
  __syncthreads();
 }
}
}

void uq_run(std::vector<int64_t>v,std::vector<int64_t>w,int64_t state,int64_t mode,int64_t grid,int64_t M,int64_t stream){
 BR_REQUIRE(v.size()==13&&w.size()==13);
 auto args=[](std::vector<int64_t>&p){return mwbr::UqArgs{(int*)p[0],(unsigned char*)p[1],(float*)p[2],(__nv_bfloat16*)p[3],(int*)p[4],(int*)p[5],(int*)p[6],(int*)p[7],(int*)p[8],(__nv_bfloat16*)p[9],(__nv_bfloat16*)p[10],(float*)p[11],(float*)p[12]};};
 auto a=args(v),b=args(w);
#define UQ_GO(K) case K: mwbr::uq_queue<K><<<grid,128,mwbr::smem_bytes<32>(),(cudaStream_t)stream>>>(a,b,(int*)state,M);break;
 switch(mode){UQ_GO(0) UQ_GO(1) UQ_GO(2) default:BR_REQUIRE(false);}
#undef UQ_GO
 BR_CUDA_CHECK(cudaGetLastError());
}
std::vector<int64_t> uq_info(int64_t mode){
 cudaFuncAttributes a{};int n=0;
#define UQ_INFO(K) case K:BR_CUDA_CHECK(cudaFuncSetAttribute(mwbr::uq_queue<K>,cudaFuncAttributeMaxDynamicSharedMemorySize,mwbr::smem_bytes<32>()));BR_CUDA_CHECK(cudaFuncGetAttributes(&a,mwbr::uq_queue<K>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwbr::uq_queue<K>,128,mwbr::smem_bytes<32>()));break;
 switch(mode){UQ_INFO(0) UQ_INFO(1) UQ_INFO(2) default:BR_REQUIRE(false);}
#undef UQ_INFO
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)mwbr::smem_bytes<32>(),n};
}
