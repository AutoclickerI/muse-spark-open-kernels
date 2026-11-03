

template<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>
__global__ __launch_bounds__(128*WK,CAP) void m32w_gemm(
    const __grid_constant__ CUtensorMap tx,const uint4* __restrict__ w,
    const uint32_t* __restrict__ esc,__nv_bfloat16* out,int m,int n,int k,int split,uint32_t t0,uint32_t t1) {
  extern __shared__ __align__(1024) uint8_t sm[];
  constexpr int RW=R/16,NT=128*WK,RP=R+4,SB=BN*128,WS=(Q*(SB+8)+1023)/1024*1024,NL=C12?3:4;
  int tid=threadIdx.x,wg=tid/128,warp=tid%128/32,lane=tid%32;
  uint32_t base0=addr(sm),base=base0+wg*WS,full=base+Q*SB;
  int rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;
  int kb0=(k/64)*(rank*WK+wg)/(split*WK),nkb=(k/64)*(rank*WK+wg+1)/(split*WK)-kb0;
  uint64_t pw,px;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(pw));
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(px));
  if(tid%128==0) {
    for(int s=0;s<Q;s++)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  auto loadx=[&](int i) {
    if(tid%128==0 && i<nkb) {
      int s=i%Q;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
      tma(base+s*SB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);
    }
  };
  #pragma unroll
  for(int s=0;s<Q;s++)loadx(s);
  uint32_t e0=0xffffffffu,e1=0xffffffffu,nxt=0xffffffffu;int ep=0;
  if(C12 && warp<RW) {
    const uint32_t* el=esc+((nt*split*WK+rank*WK+wg)*RW+warp)*64;
    e0=el[lane];e1=el[lane+32];nxt=__shfl_sync(0xffffffffu,e0,0);
  }
  uint4 raw[PF][NL]={};
  auto load=[&](int slot,int i) {
    if(i<nkb && warp<RW) {
      const uint4* src=w+((long)(nt*(k/64)+kb0+i)*RW+warp)*NL*32+lane;
      #pragma unroll
      for(int l=0;l<NL;l++)raw[slot][l]=ldw(src+l*32,pw);
    }
  };
  #pragma unroll
  for(int p=0;p<PF;p++)load(p,p);
  float acc[BN/2];
  #pragma unroll
  for(int j=0;j<BN/2;j++)asm volatile("mov.b32 %0,0;":"=f"(acc[j]));
  #pragma unroll 1
  for(int ib=0;ib<nkb;ib+=PF) {
    #pragma unroll
    for(int p=0;p<PF;p++) {
      int i=ib+p;if(i>=nkb)break;
      uint32_t a[16];
      if constexpr(C12) {
        dec(raw[p][0].x,raw[p][0].y,raw[p][2].x,t0,t1,a);
        dec(raw[p][0].z,raw[p][0].w,raw[p][2].y,t0,t1,a+4);
        dec(raw[p][1].x,raw[p][1].y,raw[p][2].z,t0,t1,a+8);
        dec(raw[p][1].z,raw[p][1].w,raw[p][2].w,t0,t1,a+12);
      } else {
        #pragma unroll
        for(int l=0;l<4;l++) {
          a[4*l]=raw[p][l].x;a[4*l+1]=raw[p][l].y;a[4*l+2]=raw[p][l].z;a[4*l+3]=raw[p][l].w;
        }
      }
      load(p,i+PF);
      if constexpr(C12) {
        while((nxt>>18)==uint32_t(i)) {
          if(lane==int((nxt>>13)&31u))patch(a,nxt);
          ++ep;
          uint32_t ev=ep<32?e0:e1;
          nxt=__shfl_sync(0xffffffffu,ev,ep&31);
          if(ep>=64)nxt=0xffffffffu;
        }
      }
      int s=i%Q;
      waitbar(full+s*8,(i/Q)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");
      if constexpr(BN==16)mma16(acc,a,desc(base+s*SB));
      if constexpr(BN==32)mma32(acc,a,desc(base+s*SB));
      if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));
      #pragma unroll
      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");
      loadx(i+Q);
    }
  }
  __syncthreads();
  if(tid%128==0) {
    for(int s=0;s<Q;s++)asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(full+s*8) : "memory");
  }
  __syncthreads();
  int row=warp*16+lane/4,col=2*(lane%4);
  #pragma unroll
  for(int j=0;j<BN/8;j++) {
    #pragma unroll
    for(int h=0;h<2;h++) {
      if(warp>=RW)continue;
      ((float*)sm)[(wg*BN+col+8*j)*RP+row+8*h]=acc[4*j+2*h];
      ((float*)sm)[(wg*BN+col+8*j+1)*RP+row+8*h]=acc[4*j+2*h+1];
    }
  }
  __syncthreads();
  if(split>1)cluster_sync();
  for(int q=tid+rank*NT;q<BN*(R/4);q+=NT*split) {
    int token=q/(R/4),row=(q%(R/4))*4;
    if(mt*BN+token>=m)continue;
    float4 sum=make_float4(0,0,0,0);
    for(int pp=0;pp<split*WK;pp++) {
      int s=pp/WK;
      uint32_t remote=base0+((pp%WK*BN+token)*RP+row)*4;
      if(split>1)asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(remote):"r"(remote),"r"(s));
      float4 v;
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(remote):"memory");
      sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
    }
    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);
    *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);
  }
  __syncthreads();
  if(split>1)cluster_sync();
}

