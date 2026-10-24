// One native Hopper warpgroup owns Q's 64 rows; four gather warps own the ring.
#ifndef USE_WGMMA
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
#define USE_WGMMA 1
#else
#define USE_WGMMA 0
#endif
#endif
DI void consumer_bar(){asm volatile("bar.sync 1,128;":::"memory");}
DI void mb_init(uint64_t* p,int n){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(su(p)),"r"(n):"memory");}
DI void mb_arrive(uint64_t* p){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(su(p)):"memory");}
DI void cp_arrive(uint64_t* p){asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"::"r"(su(p)):"memory");}
DI void mb_wait(uint64_t* p,int phase){uint32_t ok;do{asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2;selp.b32 %0,1,0,p;}":"=r"(ok):"r"(su(p)),"r"(phase):"memory");}while(!ok);}
DI uint64_t descriptor(const uint8_t* p){return (uint64_t)(su(p)>>4)|(1ull<<16)|(32ull<<32)|(2ull<<62);}
DI void proxy_fence(){asm volatile("fence.proxy.async.shared::cta;":::"memory");}
DI void wg_fence(){
#if USE_WGMMA
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
}
DI void wg_wait(){
#if USE_WGMMA
 asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
}
DI void protect(float (&a)[8][4]){
#pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<4;e++)asm volatile("":"+f"(a[n][e])::"memory");
}
WG_WRAPPERS
// The transpose is shared by all 64 queries and preserves the E4M3 bytes.
DI void transpose_v64(const uint8_t* V,uint8_t* VT){
 const int tid=threadIdx.x,t=tid&3;
#pragma unroll
 for(int i=0;i<8;i++){
  const int group=tid/4+i*32,key=group/16*4,dim=group%16*4;
  const uint32_t x=word(V,key+t,dim);
  const int base=threadIdx.x&28;
  const uint32_t a=__shfl_sync(0xffffffff,x,base),b=__shfl_sync(0xffffffff,x,base+1);
  const uint32_t c=__shfl_sync(0xffffffff,x,base+2),d=__shfl_sync(0xffffffff,x,base+3);
  const uint32_t ab=__byte_perm(a,b,t*0x11+0x40),cd=__byte_perm(c,d,t*0x11+0x40);
  *(uint32_t*)(VT+off(dim+t,key))=__byte_perm(ab,cd,0x5410);
 }
}
struct WGState{float o[8][4],m[2],l[2];};
// The portable shim has the exact same 64-row accumulator coordinates.
DI void qk_group(float (&s)[8][4],const uint8_t* Q,const uint8_t* K){
#pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<4;e++)s[n][e]=0;
 protect(s);proxy_fence();wg_fence();
#pragma unroll
 for(int kk=0;kk<2;kk++){
#if USE_WGMMA
  wg8(s,descriptor(Q+32*kk),descriptor(K+32*kk));
#else
  const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
  const uint32_t a[4]={word(Q,w*16+g,32*kk+4*t),word(Q,w*16+g+8,32*kk+4*t),word(Q,w*16+g,32*kk+16+4*t),word(Q,w*16+g+8,32*kk+16+4*t)};
#pragma unroll
  for(int n=0;n<8;n++)mma8(s[n],a,word(K,n*8+g,32*kk+4*t),word(K,n*8+g,32*kk+16+4*t));
#endif
 }
 wg_wait();protect(s);
}
DI void pv_group(float (&o)[8][4],const uint8_t* P,const uint8_t* VT){
 protect(o);proxy_fence();wg_fence();
#pragma unroll
 for(int kk=0;kk<2;kk++){
#if USE_WGMMA
  wg8(o,descriptor(P+32*kk),descriptor(VT+32*kk));
#else
  const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
  const uint32_t a[4]={word(P,w*16+g,32*kk+4*t),word(P,w*16+g+8,32*kk+4*t),word(P,w*16+g,32*kk+16+4*t),word(P,w*16+g+8,32*kk+16+4*t)};
#pragma unroll
  for(int n=0;n<8;n++)mma8(o[n],a,word(VT,n*8+g,32*kk+4*t),word(VT,n*8+g,32*kk+16+4*t));
#endif
 }
 wg_wait();protect(o);
}
template<bool MASK,bool CLOCK>
DI void native_tile(WGState& st,const uint8_t* Q,const uint8_t* K,const uint8_t* V,uint8_t* P,uint8_t* VT,int kp,int hi,int seq,float scale,Trace& trace){
 unsigned long long ts=0;if constexpr(CLOCK)ts=tick();
 transpose_v64(V,VT);consumer_bar();
 if constexpr(CLOCK){const auto now=tick();trace.c[2]+=now-ts;ts=now;}
 float s[8][4];qk_group(s,Q,K);
 if constexpr(CLOCK){const auto now=tick();trace.c[3]+=now-ts;ts=now;}
 const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
#pragma unroll
 for(int rh=0;rh<2;rh++){
  float mx=-INFINITY;
#pragma unroll
  for(int n=0;n<8;n++){
#pragma unroll
   for(int e=0;e<2;e++){
    float x=s[n][2*rh+e]*scale;
    if constexpr(MASK){const int key=kp+8*n+2*t+e;if(key>=hi||key>seq+2*warp+rh)x=-INFINITY;}
    s[n][2*rh+e]=x;mx=fmaxf(mx,x);
   }
  }
  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
  const float nm=fmaxf(st.m[rh],mx),safe=nm==-INFINITY?0.f:nm;
  const float alpha=st.m[rh]==-INFINITY?0.f:ex(st.m[rh]-safe);
  float sum=0;
#pragma unroll
  for(int n=0;n<8;n++){
   const float2 p=exp2pair(s[n][2*rh]-safe,s[n][2*rh+1]-safe);
   s[n][2*rh]=p.x;s[n][2*rh+1]=p.y;sum+=p.x+p.y;
  }
  sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);
  st.m[rh]=nm;st.l[rh]=st.l[rh]*alpha+sum;
#pragma unroll
  for(int n=0;n<8;n++){st.o[n][2*rh]*=alpha;st.o[n][2*rh+1]*=alpha;}
 }
 if constexpr(CLOCK){const auto now=tick();trace.c[4]+=now-ts;ts=now;}
#pragma unroll
 for(int n=0;n<8;n++){
  *(uint16_t*)(P+off(warp*16+g,8*n+2*t))=fp8pair(s[n][0]*256.f,s[n][1]*256.f);
  *(uint16_t*)(P+off(warp*16+g+8,8*n+2*t))=fp8pair(s[n][2]*256.f,s[n][3]*256.f);
 }
 consumer_bar();
 if constexpr(CLOCK){const auto now=tick();trace.c[5]+=now-ts;ts=now;}
 pv_group(st.o,P,VT);
 if constexpr(CLOCK)trace.c[6]+=tick()-ts;
}
template<int ST,bool CLOCK>
__global__ __launch_bounds__(256,2)void native_verify8(VP p){
 extern __shared__ __align__(1024) uint8_t sm[];
 uint8_t* K=sm;uint8_t* V=K+ST*4096;uint8_t* Q=V+ST*4096;
 uint8_t* NK=Q+4096;uint8_t* NV=NK+4096;uint8_t* P=NV+4096;uint8_t* VT=P+4096;
 __shared__ __align__(16)uint64_t full[ST],empty[ST];
 const int tid=threadIdx.x,warp=tid>>5,lane=tid&31;
 const int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];const bool last=sp==p.ns-1;
 const int chunk=((seq+p.ns-1)/p.ns+63)&~63;
 const int begin=min(seq,sp*chunk),end=min(seq,begin+chunk),ntiles=(end-begin+63)/64;
 if(tid<ST){mb_init(&full[tid],128);mb_init(&empty[tid],128);}
 __syncthreads();proxy_fence();
 if(tid>=128){
  const int ptid=tid-128;Trace trace{};unsigned long long start=0;if constexpr(CLOCK)start=tick();
  for(int it=0;it<ntiles;it++){
   unsigned long long ts=0;if constexpr(CLOCK)ts=tick();
   const int slot=it%ST;
   if(it>=ST)mb_wait(&empty[slot],((it/ST)-1)&1);
   if constexpr(CLOCK){const auto now=tick();trace.c[1]+=now-ts;ts=now;}
#pragma unroll
   for(int rr=0;rr<2;rr++){
    const int row=ptid/4+32*rr,col=16*(ptid&3),key=begin+it*64+row;
    const bool valid=key<end;
    const int physical=valid?p.pt[(int64_t)b*p.pages+(key>>p.lg)]*p.ps+(key&(p.ps-1)):0;
    const int64_t src=(int64_t)physical*p.HK*64+h*64+col;
    cp(K+slot*4096+off(row,col),p.kc+src,valid);cp(V+slot*4096+off(row,col),p.vc+src,valid);
   }
   commit();cp_arrive(&full[slot]);wait<ST-1>();
   if constexpr(CLOCK)trace.c[0]+=tick()-ts;
  }
  wait<0>();
  if constexpr(CLOCK){if(lane==0){auto* dst=p.clocks+((((int64_t)b*p.HK+h)*p.ns+sp)*8+warp)*10;for(int i=0;i<7;i++)dst[i]=trace.c[i];dst[7]=0;dst[8]=tick()-start;dst[9]=ntiles;}}
  return;
 }
 Trace trace{};unsigned long long start=0,init=0;if constexpr(CLOCK)start=tick();
#pragma unroll
 for(int i=0;i<16;i++){
  const int row=warp+4*i,j=row/8,head=h*8+row%8;
  const float2 q=input_pair(p,b*8+j,head*64+2*lane);
  const float n=rsqrtf(wsum(q.x*q.x+q.y*q.y)/64.f+1.e-5f);
  *(uint16_t*)(Q+off(row,2*lane))=fp8pair(q.x*n,q.y*n);
 }
 if(last){
  for(int i=tid;i<8192/16;i+=128)*(uint4*)(NK+16*i)=make_uint4(0,0,0,0);
  consumer_bar();
#pragma unroll
  for(int i=0;i<4;i++){
   const int r=warp+4*i,j=r/2,isv=r&1;
   float2 x=input_pair(p,b*8+j,(p.HQ+isv*p.HK+h)*64+2*lane);
   if(!isv){const float n=rsqrtf(wsum(x.x*x.x+x.y*x.y)/64.f+1.e-5f);x.x*=n;x.y*=n;}
   const uint16_t f=fp8pair(x.x/(*(isv?p.vs:p.ks)),x.y/(*(isv?p.vs:p.ks)));
   *(uint16_t*)((isv?p.vc:p.kc)+p.loc[b*8+j]*p.HK*64+h*64+2*lane)=f;
   *(uint16_t*)((isv?NV:NK)+off(j,2*lane))=f;
  }
 }
 consumer_bar();WGState st;
#pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<4;e++)st.o[n][e]=0;
 st.m[0]=st.m[1]=-INFINITY;st.l[0]=st.l[1]=0;
 if constexpr(CLOCK)init=tick()-start;
 for(int it=0;it<ntiles;it++){
  unsigned long long ts=0;if constexpr(CLOCK)ts=tick();
  mb_wait(&full[it%ST],(it/ST)&1);consumer_bar();
  if constexpr(CLOCK)trace.c[1]+=tick()-ts;
  const int key=begin+it*64;
  if(key+64<=end)native_tile<false,CLOCK>(st,Q,K+(it%ST)*4096,V+(it%ST)*4096,P,VT,key,end,seq,p.scale*(*p.ks),trace);
  else native_tile<true,CLOCK>(st,Q,K+(it%ST)*4096,V+(it%ST)*4096,P,VT,key,end,seq,p.scale*(*p.ks),trace);
  consumer_bar();mb_arrive(&empty[it%ST]);
 }
 if(last)native_tile<true,CLOCK>(st,Q,NK,NV,P,VT,seq,seq+8,seq,p.scale*(*p.ks),trace);
 const int g=lane>>2,t=lane&3;
 float* out=p.partial+(((int64_t)b*p.HK+h)*p.ns+sp)*64*68;
#pragma unroll
 for(int rh=0;rh<2;rh++){
  const int row=warp*16+g+8*rh;
#pragma unroll
  for(int d=0;d<8;d++)*(float2*)(out+row*68+8*d+2*t)=make_float2(st.o[d][2*rh],st.o[d][2*rh+1]);
  if(t==0)*(float2*)(out+row*68+64)=make_float2(st.m[rh],st.l[rh]);
 }
 if constexpr(CLOCK){if(lane==0){auto* dst=p.clocks+((((int64_t)b*p.HK+h)*p.ns+sp)*8+warp)*10;for(int i=0;i<7;i++)dst[i]=trace.c[i];dst[7]=init;dst[8]=tick()-start;dst[9]=ntiles;}}
}
