DI void rescale(float (&O)[8][4],const float (&alpha)[2]) {
#pragma unroll
  for(int n=0;n<8;n++) {O[n][0]*=alpha[0];O[n][1]*=alpha[0];O[n][2]*=alpha[1];O[n][3]*=alpha[1];}
}
// One CTA owns one (request, KV head, GQA subgroup, window split).
// All splits form one hardware cluster, with no device counters or metadata pass.
DI uint32_t ca(const void* p,int rank) {uint32_t v;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(v):"r"(su(p)),"r"(rank));return v;}
DI float2 cload(const float* p,int rank) {float2 v;uint32_t a=ca(p,rank);asm volatile("ld.shared::cluster.v2.f32 {%0,%1},[%2];":"=f"(v.x),"=f"(v.y):"r"(a):"memory");return v;}
DI void cbar() {asm volatile("barrier.cluster.arrive.release.aligned;barrier.cluster.wait.acquire.aligned;":::"memory");}

template<int N> DI void guard(float (&x)[N][4]) {
#pragma unroll
  for(int j=0;j<N;j++)for(int e=0;e<4;e++)asm volatile("":"+f"(x[j][e])::"memory");
}
template<int N> DI void guardp(uint32_t (&x)[N]) {
#pragma unroll
  for(int j=0;j<N;j++)asm volatile("":"+r"(x[j])::"memory");
}
// One warp transposes 16 keys. Independent 64-key slabs keep the native RS-P layout unchanged.
DI void trans64(const uint8_t* vr,uint8_t* vt,int pw,int lane) {
  const int mi=lane>>3,rr=lane&7,t=lane&3,cc=lane>>2;
#pragma unroll
  for(int dp=0;dp<2;dp++) {
    const int key=16*pw+((mi&1)<<3)+rr,dg=2*dp+(mi>>1);
    uint32_t r0,r1,r2,r3;ldsm4t(r0,r1,r2,r3,su(vr+off(key,16*dg)));
    const int d0=32*dp+2*cc,d1=d0+16,col=16*pw+4*t;
    *(uint32_t*)(vt+off(d0,col))=prmt(r0,r1,0x6420);
    *(uint32_t*)(vt+off(d0+1,col))=prmt(r0,r1,0x7531);
    *(uint32_t*)(vt+off(d1,col))=prmt(r2,r3,0x6420);
    *(uint32_t*)(vt+off(d1+1,col))=prmt(r2,r3,0x7531);
  }
}

template<int L,int QT>
DI void prep(const VP& p,uint8_t* Q,uint8_t* KN,uint8_t* VN,float2* csbuf,int b,int kh,int sub,int seq,bool last) {
  constexpr int H=8/QT,NR=L*H;
  const int tid=threadIdx.x,w=tid>>5,lane=tid&31,r=tid>>2,c=tid&3;
  // Historical KV prefetch has already issued. Projection values are consumed only after this wait.
  asm volatile("griddepcontrol.wait;":::"memory");
  constexpr int ROUNDS=(NR+31)/32;
  int tk[ROUNDS*8],col[ROUNDS*8];bool vd[ROUNDS*8];float2 x[ROUNDS*8];
#pragma unroll
  for(int j=0;j<ROUNDS;j++)for(int i=0;i<8;i++) {
    const int row=r+32*j,token=row/H,head=kh*8+sub*H+row%H,a=j*8+i;
    vd[a]=row<NR;tk[a]=b*L+(vd[a]?token:0);col[a]=head*64+c*16+2*i;
  }
  input_pairs<ROUNDS*8>(p,tk,col,vd,x);
  for(int i=tid;i<L*32;i+=128) {
    float sn,cs;__sincosf((float)p.positions[b*L+(i>>5)]*p.inv_freq[i&31],&sn,&cs);
    csbuf[i]=make_float2(cs,sn);
  }
  __syncthreads();
#pragma unroll
  for(int j=0;j<ROUNDS;j++) {
    float z=0.f;
#pragma unroll
    for(int i=0;i<8;i++)z+=x[j*8+i].x*x[j*8+i].x+x[j*8+i].y*x[j*8+i].y;
    z+=__shfl_xor_sync(0xffffffffu,z,1);z+=__shfl_xor_sync(0xffffffffu,z,2);
    const float norm=rsqrtf(z*(1.f/64.f)+1e-5f);const int row=r+32*j;
    const int token=min((int)row/H,L-1);
#pragma unroll
    for(int i=0;i<8;i++) {
      const float2 cs=csbuf[token*32+c*8+i],v=x[j*8+i];
      const float a=v.x*norm,z=v.y*norm;
      if(row<NR)*(uint16_t*)(Q+off(row,c*16+2*i))=fp8pair(a*cs.x-z*cs.y,a*cs.y+z*cs.x);
    }
  }
  // Remaining padded Q rows and virtual V columns must be zero, including FP8 NaN bit patterns.
  for(int i=tid;i<4096/16;i+=128) {
    if(i*16>=NR*64)*(uint4*)(Q+i*16)=make_uint4(0,0,0,0);
    *(uint4*)(KN+i*16)=make_uint4(0,0,0,0);
    *(uint4*)(VN+i*16)=make_uint4(0,0,0,0);
  }
  __syncthreads();
  int kt[4],kc[4];bool kv[4];float2 val[4];
#pragma unroll
  for(int i=0;i<2;i++) {
    const int tok=2*w+i;kv[i]=kv[i+2]=tok<L;
    kt[i]=kt[i+2]=b*L+(tok<L?tok:0);
    kc[i]=(p.HQ+kh)*64+2*lane;kc[i+2]=(p.HQ+p.HK+kh)*64+2*lane;
  }
  input_pairs<4>(p,kt,kc,kv,val);
#pragma unroll
  for(int i=0;i<2;i++) {
    const int tok=2*w+i;
    if(tok<L) {
      const float2 v=val[i+2],k=val[i],cs=csbuf[tok*32+lane];
      const float norm=rsqrtf(wsum(k.x*k.x+k.y*k.y)*(1.f/64.f)+1e-5f);
      const float a=k.x*norm,z=k.y*norm;
      const uint16_t k8=fp8pair((a*cs.x-z*cs.y)/(*p.ks),(a*cs.y+z*cs.x)/(*p.ks));
      const uint16_t v8=fp8pair(v.x/(*p.vs),v.y/(*p.vs));
      *(uint16_t*)(KN+off(tok,2*lane))=k8;
      const int pos=4*(tok>>1)+(tok&1);
      VN[off(2*lane,pos)]=(uint8_t)v8;VN[off(2*lane+1,pos)]=(uint8_t)(v8>>8);
      if(last && sub==0) {
        const int64_t loc=p.loc[b*L+tok];
        *(uint16_t*)(p.kc+(loc*p.HK+kh)*64+2*lane)=k8;
        *(uint16_t*)(p.vc+(loc*p.HK+kh)*64+2*lane)=v8;
      }
    }
  }
  __syncthreads();proxy_fence();
}

template<int L,int QT,int NK>
DI void prob(float (&s)[NK/8][4],float (&m)[2],float (&ls)[2],float (&alpha)[2],uint32_t (&pk)[NK/8],float sc,int key0,int seq,int end,int window) {
  constexpr int H=8/QT,N=NK/8;
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,t=lane&3,g=lane>>2;
  if(warp*16>=L*H) {
    alpha[0]=alpha[1]=1.f;
#pragma unroll
    for(int i=0;i<NK/8;i++)pk[i]=0;
    return;
  }
#pragma unroll
  for(int rh=0;rh<2;rh++) {
    const int row=warp*16+g+8*rh,tok=row/H;
    const bool live=row<L*H;
    float mx=-INFINITY;
#pragma unroll
    for(int n=0;n<N;n++)for(int e=0;e<2;e++) {
      const int key=key0+8*n+2*t+e;
      const bool valid=live && key<end && key<=seq+tok && key>=seq+tok-window+1;
      const float v=valid?s[n][2*rh+e]*sc:-INFINITY;
      s[n][2*rh+e]=v;mx=fmaxf(mx,v);
    }
    mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,2));
    const float mm=fmaxf(m[rh],mx),safe=mm==-INFINITY?0.f:mm;
    alpha[rh]=ex2(m[rh]-safe);m[rh]=mm;
    float total=0.f;
#pragma unroll
    for(int n=0;n<N;n++)for(int e=0;e<2;e++) {
      const float v=ex2(s[n][2*rh+e]-safe+8.f);s[n][2*rh+e]=v;total+=v;
    }
    ls[rh]=ls[rh]*alpha[rh]+total;
  }
#pragma unroll
  for(int kk=0;kk<NK/32;kk++)for(int hf=0;hf<2;hf++) {
    const int n0=4*kk+2*hf;
    pk[4*kk+2*hf]=pack4(s[n0][0],s[n0][1],s[n0+1][0],s[n0+1][1]);
    pk[4*kk+2*hf+1]=pack4(s[n0][2],s[n0][3],s[n0+1][2],s[n0+1][3]);
  }
}

template<int NK> DI void multiply_v(float (&O)[8][4],const uint32_t (&pk)[NK/8],const uint8_t* vt) {
#if USE_WGMMA
#pragma unroll
  for(int j=0;j<NK/32;j++) {
    uint64_t va=desc(su(vt+(j/2)*4096))+(j%2)*2;
    pin(va);wg_rs(O,pk[j*4],pk[j*4+1],pk[j*4+2],pk[j*4+3],va);
  }
#else
#pragma unroll
  for(int j=0;j<NK/64;j++) {
    uint32_t pp[8];
#pragma unroll
    for(int k=0;k<8;k++)pp[k]=pk[j*8+k];
    pv(O,pp,vt+j*4096);
  }
#endif
}
// N8 QK for the block's own keys; one K32 PV, rather than a full padded N64 tile.
template<int L,int QT>
DI void virtual_tile(float (&O)[8][4],float (&m)[2],float (&ls)[2],const uint8_t* Q,const uint8_t* KN,const uint8_t* VN,float sc,int seq) {
  float s[1][4];guard(s);wg_fence();qk8(s,Q,KN);wg_commit();wg_wait<0>();guard(s);
  uint32_t pk[4]={};float alpha[2];
  constexpr int H=8/QT;
  const int w=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
#pragma unroll
  for(int rh=0;rh<2;rh++) {
    const int row=w*16+g+rh*8,tok=row/H;
    float v0=(row<L*H && 2*t<=tok && 2*t<L)?s[0][2*rh]*sc:-INFINITY;
    float v1=(row<L*H && 2*t+1<=tok && 2*t+1<L)?s[0][2*rh+1]*sc:-INFINITY;
    float mx=fmaxf(v0,v1);mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,2));
    const float mm=fmaxf(m[rh],mx),safe=mm==-INFINITY?0.f:mm;
    alpha[rh]=ex2(m[rh]-safe);m[rh]=mm;
    v0=ex2(v0-safe+8.f);v1=ex2(v1-safe+8.f);
    ls[rh]=ls[rh]*alpha[rh]+v0+v1;
    pk[rh]=(uint32_t)fp8pair(v0,v1);
  }
  rescale(O,alpha);protect(O);
#if USE_WGMMA
  wg_fence();uint64_t va=desc(su(VN));pin(va);wg_rs(O,pk[0],pk[1],pk[2],pk[3],va);wg_commit();wg_wait<0>();protect(O);
#else
  uint32_t p8[8]={pk[0],pk[1],pk[2],pk[3],0,0,0,0};pv(O,p8,VN);
#endif
}

template<int L,int QT,int NK> constexpr int cluster_smem() {return 4096+5*NK*64+8192+L*32*8;}

template<int L,int QT,int NK>
__global__ __launch_bounds__(128,2) void sliding_main(VP p) {
  asm volatile("griddepcontrol.launch_dependents;":::"memory");
  extern __shared__ __align__(1024) uint8_t sm[];
  constexpr int KB=NK*64,H=8/QT,NR=L*H;
  uint8_t* Q=sm;uint8_t* K=Q+4096;uint8_t* V=K+2*KB;uint8_t* VT=V+2*KB;
  uint8_t* KN=VT+KB;uint8_t* VN=KN+4096;float2* csbuf=(float2*)(VN+4096);
  const int split=blockIdx.x,sub=blockIdx.y%QT,kh=blockIdx.y/QT,b=blockIdx.z;
  const int tid=threadIdx.x,w=tid>>5,lane=tid&31;
  const int seq=p.seq[b],lo=max(0,seq-p.window+1),nc=seq-lo;
  const int chunk=((nc+p.ns-1)/p.ns+NK-1)/NK*NK,begin=min(seq,lo+split*chunk),end=min(seq,begin+chunk);
  const int nt=(end-begin+NK-1)/NK;
  auto issue=[&](int tile,int slot) {
#pragma unroll
    for(int j=0;j<NK/32;j++) {
      const int row=tid/4+32*j,key=begin+tile*NK+row,clamped=min(key,max(seq-1,0));
      const int page=p.pt[(int64_t)b*p.pages+(clamped>>p.lg)];
      const int physical=page*p.ps+(clamped&(p.ps-1));
      const int64_t offg=((int64_t)physical*p.HK+kh)*64+16*(tid&3);
      const bool valid=tile<nt && key<end;
      cp16(K+slot*KB+off(row,16*(tid&3)),p.kc+offg,valid);
      cp16(V+slot*KB+off(row,16*(tid&3)),p.vc+offg,valid);
    }
    commit();
  };
  issue(0,0);issue(1,1);
  prep<L,QT>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==p.ns-1);
  float O[8][4]={},m[2]={-INFINITY,-INFINITY},ls[2]={},alpha[2];
  const float sc=p.sc*(*p.ks);
  for(int it=0;it<nt;it++) {
    cp_wait<1>();__syncthreads();
#pragma unroll
    for(int j=0;j<NK/64;j++)trans64(V+(it%2)*KB+j*4096,VT+j*4096,w,lane);
    __syncthreads();proxy_fence();
    float s[NK/8][4];uint32_t pk[NK/8];guard(s);wg_fence();
    if constexpr(NK==64)qk64(s,Q,K+(it%2)*KB);else qk128(s,Q,K+(it%2)*KB);
    wg_commit();wg_wait<0>();guard(s);
    prob<L,QT,NK>(s,m,ls,alpha,pk,sc,begin+it*NK,seq,end,p.window);
    rescale(O,alpha);protect(O);guardp(pk);wg_fence();multiply_v<NK>(O,pk,VT);wg_commit();wg_wait<0>();protect(O);guardp(pk);
    __syncthreads();issue(it+2,it%2);
  }
  cp_wait<0>();__syncthreads();
  if(split==p.ns-1)virtual_tile<L,QT>(O,m,ls,Q,KN,VN,sc,seq);
  // All asynchronous use has drained; stage storage becomes per-row FP32 partials.
  __syncthreads();float* part=(float*)K;
  const int g=lane>>2,t=lane&3;
#pragma unroll
  for(int rh=0;rh<2;rh++) {
    const int row=w*16+g+8*rh;
    float den=ls[rh];den+=__shfl_xor_sync(0xffffffffu,den,1);den+=__shfl_xor_sync(0xffffffffu,den,2);
    float* pp=part+row*68;
#pragma unroll
    for(int n=0;n<8;n++)*(float2*)(pp+8*n+2*t)=make_float2(O[n][2*rh],O[n][2*rh+1]);
    if(t==0)*(float2*)(pp+64)=make_float2(m[rh],den);
  }
  __syncthreads();cbar();
  const int rank=split%p.cs;
  for(int j=w;rank+j*p.cs<NR;j+=4) {
    const int r=rank+j*p.cs,token=r/H,head=kh*8+sub*H+r%H;
    float2 ml[16],ov[16];
#pragma unroll
    for(int c=0;c<16;c++)if(c<p.cs) {ml[c]=cload(part+r*68+64,c);ov[c]=cload(part+r*68+2*lane,c);}
    float mx=-INFINITY;
#pragma unroll
    for(int c=0;c<16;c++)if(c<p.cs)mx=fmaxf(mx,ml[c].x);
    float x=0.f,y=0.f,den=0.f;
#pragma unroll
    for(int c=0;c<16;c++)if(c<p.cs) {const float z=ex2(ml[c].x-mx);x=fmaf(z,ov[c].x,x);y=fmaf(z,ov[c].y,y);den=fmaf(z,ml[c].y,den);}
    const float scl=(*p.vs)/den;x*=scl;y*=scl;
    const float norm=rsqrtf(wsum(x*x+y*y)*(1.f/64.f)+1e-5f);
    float2 gate=make_float2(0.f,0.f);
    if(p.gate) {int tk[1]={b*L+token},col[1]={(p.HQ+2*p.HK+head)*64+2*lane};bool valid[1]={true};float2 gv[1];input_pairs<1>(p,tk,col,valid,gv);gate=gv[0];}
    *(__nv_bfloat162*)(p.out+((int64_t)(b*L+token)*p.HQ+head)*64+2*lane)=p.gate?
      __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate.x)),y*norm/(1.f+__expf(-gate.y))):__floats2bfloat162_rn(x*norm,y*norm);
  }
  cbar();
}

template<int L,int QT,int NK> void attrs() {
  cudaFuncSetAttribute(sliding_main<L,QT,NK>,cudaFuncAttributeMaxDynamicSharedMemorySize,cluster_smem<L,QT,NK>());
  cudaFuncSetAttribute(sliding_main<L,QT,NK>,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
}
template<int L,int QT,int NK> std::vector<int64_t> resource(int cs) {
  attrs<L,QT,NK>();cudaFuncAttributes a;cudaFuncGetAttributes(&a,sliding_main<L,QT,NK>);
  int blocks=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,sliding_main<L,QT,NK>,128,cluster_smem<L,QT,NK>());
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(cs);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=cluster_smem<L,QT,NK>();
  cudaLaunchAttribute at={};at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim.x=cs;at.val.clusterDim.y=1;at.val.clusterDim.z=1;cfg.attrs=&at;cfg.numAttrs=1;
  int clusters=0;if(cudaOccupancyMaxActiveClusters(&clusters,sliding_main<L,QT,NK>,&cfg)!=cudaSuccess){cudaGetLastError();clusters=0;}
  return {a.numRegs,(int64_t)a.localSizeBytes,cluster_smem<L,QT,NK>(),blocks,clusters};
}
template<int L,int QT,int NK> void run(VP p,cudaStream_t stream) {
  attrs<L,QT,NK>();cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(p.ns,p.HK*QT,p.B);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=cluster_smem<L,QT,NK>();cfg.stream=stream;
  cudaLaunchAttribute at[2]={};at[0].id=cudaLaunchAttributeClusterDimension;at[0].val.clusterDim.x=p.cs;at[0].val.clusterDim.y=1;at[0].val.clusterDim.z=1;
  at[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;at[1].val.programmaticStreamSerializationAllowed=1;cfg.attrs=at;cfg.numAttrs=2;
  auto e=cudaLaunchKernelEx(&cfg,sliding_main<L,QT,NK>,p);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace

std::vector<int64_t> sliding_resources(int64_t L,int64_t qt,int64_t nk,int64_t cs) {
#define RES(L,Q) if(nk==64)return resource<L,Q,64>(cs);else return resource<L,Q,128>(cs)
  if(L==1){if(qt==4){RES(1,4);}else{RES(1,1);}}
  else {if(qt==4){RES(8,4);}else{RES(8,1);}}
#undef RES
}
void sliding_forward(int64_t qkv,int64_t positions,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t inv,int64_t out,
 int64_t B,int64_t L,int64_t HQ,int64_t HK,int64_t stride,int64_t pages,int64_t ps,int64_t window,int64_t splits,int64_t split_stride,int64_t qt,int64_t nk,int64_t cs,int64_t width,double sc,int64_t stream) {
  VP p={};p.qkv=(const void*)qkv;p.positions=(const int64_t*)positions;p.loc=(const int64_t*)loc;p.pt=(const int*)pt;p.seq=(const int*)seq;
  p.kc=(uint8_t*)kc;p.vc=(uint8_t*)vc;p.ks=(const float*)ks;p.vs=(const float*)vs;p.inv_freq=(const float*)inv;p.out=(__nv_bfloat16*)out;
  p.B=B;p.L=L;p.HQ=HQ;p.HK=HK;p.stride=stride;p.pages=pages;p.ps=ps;p.window=window;p.proj_splits=splits;p.split_stride=split_stride;p.qt=qt;p.ns=p.cs=cs;
  p.gate=width>(HQ+2*HK)*64;p.sc=sc;while((1<<p.lg)<ps)p.lg++;
#define GO(L,Q) if(nk==64)run<L,Q,64>(p,(cudaStream_t)stream);else run<L,Q,128>(p,(cudaStream_t)stream)
  if(L==1){if(qt==4){GO(1,4);}else{GO(1,1);}}
  else {if(qt==4){GO(8,4);}else{GO(8,1);}}
#undef GO
  auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
