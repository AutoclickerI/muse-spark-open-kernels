// K-major QK: M=64 cached keys, N=8 GQA queries. The cache/gather ring
// and final FP16-P PV are unchanged. All128 threads issue each native WG.
DEVI int q8off(int r,int c){return r*64+(((c>>4)^((r>>1)&3))<<4)+(c&15);}
DEVI uint64_t q8desc(const void*p){return (uint64_t)(smem_u32(p)>>4)|(1ull<<16)|(32ull<<32)|(2ull<<62);}
DEVI uint32_t kword(const uint8_t* p,int row,int h,int col){return *(const uint32_t*)(p+row*ROWB+(((4*h+(col>>4))^swz(row))<<4)+(col&15));}
DEVI void key_query(float (&s)[4],const uint8_t* K,const uint8_t* Q,int h){
 const int lane=threadIdx.x&31,r=16*(threadIdx.x>>5)+(lane>>2),t=lane&3;
#pragma unroll
 for(int e=0;e<4;e++)s[e]=0.f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900
 asm volatile("" : "+f"(s[0]),"+f"(s[1]),"+f"(s[2]),"+f"(s[3]) :: "memory");
#pragma unroll
 for(int kk=0;kk<2;kk++){
  const uint32_t a0=kword(K,r,h,32*kk+4*t),a1=kword(K,r+8,h,32*kk+4*t);
  const uint32_t a2=kword(K,r,h,32*kk+16+4*t),a3=kword(K,r+8,h,32*kk+16+4*t);
  const uint64_t b=q8desc(Q)+2*kk;
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  asm volatile("{.reg .pred p;setp.ne.b32 p,%9,0;wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},%8,p,1,1;}"
   :"+f"(s[0]),"+f"(s[1]),"+f"(s[2]),"+f"(s[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(b),"r"(kk));
 }
 asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;" ::: "memory");
 asm volatile("" : "+f"(s[0]),"+f"(s[1]),"+f"(s[2]),"+f"(s[3]) :: "memory");
#else
 // Explicit local stand-in consumes exact E4M3 Q/K and the same score layout.
#pragma unroll
 for(int e=0;e<4;e++){
  float acc=0.f;
  const int kr=r+8*(e>>1),qr=2*t+(e&1);
#pragma unroll
  for(int c=0;c<64;c++){
   const uint8_t kb=K[kr*ROWB+(((4*h+(c>>4))^swz(kr))<<4)+(c&15)];
   const uint8_t qb=Q[q8off(qr,c)];
   const float x=__half2float(__nv_cvt_fp8_to_halfraw(kb,__NV_E4M3));
   const float y=__half2float(__nv_cvt_fp8_to_halfraw(qb,__NV_E4M3));
   acc=fmaf(x,y,acc);
  }
  s[e]=acc;
 }
#endif
}
