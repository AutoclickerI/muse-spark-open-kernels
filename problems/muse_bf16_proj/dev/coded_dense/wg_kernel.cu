// Register-A WGMMA over lossless codes. Shared X: [k64][padded tokens][64], TMA swizzle128.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define CD_WG_NATIVE 1
#else
#define CD_WG_NATIVE 0
#endif

template<int MT> __device__ __forceinline__ void wg_atom(float (&d)[MT/2], const uint32_t (&a)[4],
                                                       const __nv_bfloat16* xs, int j) {
#if CD_WG_NATIVE
  const uint64_t desc=(uint64_t)(smem_u32(xs+j*16)>>4) | ((uint64_t)64<<32) | ((uint64_t)1<<62);
  WG_ATOMS
#else
  // Exact same ownership/activation layout; arithmetic stand-in for local sm_120 only.
  const int lane=threadIdx.x&31, g=lane>>2, t=lane&3;
#pragma unroll
  for(int nt=0;nt<MT/8;++nt) {
    const int row=nt*8+g, col=j*16+2*t;
    const int off=row*64+(col^((row&7)*8));
    const uint32_t b0=*(const uint32_t*)(xs+off);
    const uint32_t b1=*(const uint32_t*)(xs+row*64+((col+8)^((row&7)*8)));
    mma16816(*reinterpret_cast<float(*)[4]>(d+nt*4),a,b0,b1);
  }
#endif
}
template<int MT> __device__ __forceinline__ void wg_protect(float (&d)[MT/2]) {
#pragma unroll
  for(int i=0;i<MT/2;++i)asm volatile("":"+f"(d[i])::"memory");
}
template<int Q> __device__ __forceinline__ void wg_wait() {
#if CD_WG_NATIVE
  asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(Q):"memory");
#endif
}
__device__ __forceinline__ void wg_fence() {
#if CD_WG_NATIVE
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
}
__device__ __forceinline__ void wg_commit() {
#if CD_WG_NATIVE
  asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
#endif
}
__device__ __forceinline__ void wg_keep(const uint32_t (&a)[4][4]) {
#pragma unroll
  for(int j=0;j<4;++j)asm volatile(""::"r"(a[j][0]),"r"(a[j][1]),"r"(a[j][2]),"r"(a[j][3]):"memory");
}
__device__ __forceinline__ void tma_x(uint32_t dst,const CUtensorMap* map,int col,uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
    ::"r"(dst),"l"(map),"r"(col),"r"(0),"r"(bar):"memory");
}

template<int MT,int WB>
__global__ void __launch_bounds__(256) wg_gemv(const uint4* __restrict__ Wc,const float* __restrict__ scales,
 const int* __restrict__ eoff,const int2* __restrict__ esc,const __nv_bfloat16* X,int x_stride,int M,
 float* P,int N,int KB,int S,int rs,int ks,int prew,const __grid_constant__ CUtensorMap xm) {
  extern __shared__ __align__(1024) unsigned char sm[];
  auto xs=reinterpret_cast<__nv_bfloat16*>(sm);
  const int lane=threadIdx.x&31, warp=threadIdx.x>>5,g=lane>>2,t=lane&3,nw=blockDim.x>>5;
  const int s=blockIdx.y,kb0=s*KB/S,kb1=(s+1)*KB/S,nb=kb1-kb0;
  const int r=(blockIdx.x*nw+warp), step=ks*96;
  const uint4* wp=Wc+((size_t)r*rs+(size_t)kb0*ks)*96+lane;
  const float sc=ld_f32(scales+r);
  const int ea=ld_i32(eoff+(size_t)r*KB+kb0),eb=ld_i32(eoff+(size_t)r*KB+kb1);
  uint4 buf[2][3];
  pdl_release();
  if(!prew)pdl_wait();
#pragma unroll
  for(int p=0;p<2;++p)
#pragma unroll
    for(int q=0;q<3;++q)buf[p][q]=ld_w_if(wp+p*step+q*32,p<nb);
  constexpr int CH=4;
  const int nch=(nb+CH-1)/CH;
  const uint32_t bar0=smem_u32(sm+nb*MT*128);
  if(threadIdx.x==0) {
    for(int c=0;c<nch;++c)mbar_init(bar0+c*8,1);
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  __syncthreads();
  if(prew)pdl_wait();
  if(threadIdx.x==0) {
    for(int c=0;c<nch;++c) {
      const uint32_t bar=bar0+c*8;
      mbar_expect(bar,min(CH,nb-c*CH)*MT*128);
      for(int b=c*CH;b<min(nb,(c+1)*CH);++b)
        tma_x(smem_u32(sm+b*MT*128),&xm,(kb0+b)*64,bar);
    }
  }
  float acc[MT/2];
#pragma unroll
  for(int i=0;i<MT/2;++i)acc[i]=0.f;
  uint32_t A[WB][4][4];
  // Even block count is required by the host ABI. Both unrolled operands have independent lifetimes.
  for(int b=0;b<nb;b+=2) {
    if((b%CH)==0)mbar_wait0(bar0+(b/CH)*8);
#pragma unroll
    for(int p=0;p<2;++p) {
      const int z=WB==1?0:p;
      if(b+p>=WB) {
        wg_wait<WB-1>();
        wg_keep(A[z]);
      }
#pragma unroll
      for(int j=0;j<4;++j)
        decode_step(word(buf[p][j>>1],(j&1)*2),word(buf[p][j>>1],(j&1)*2+1),word(buf[p][2],j),A[z][j]);
#pragma unroll
      for(int q=0;q<3;++q)buf[p][q]=ld_w_if(wp+(b+p+2)*step+q*32,b+p+2<nb);
      wg_protect<MT>(acc);
      wg_fence();
#pragma unroll
      for(int j=0;j<4;++j)wg_atom<MT>(acc,A[z][j],xs+(size_t)(b+p)*MT*64,j);
      wg_commit();
    }
  }
  wg_wait<0>();
#pragma unroll
  for(int z=0;z<WB;++z)wg_keep(A[z]);
  wg_protect<MT>(acc);
#pragma unroll
  for(int i=0;i<MT/2;++i)acc[i]*=sc;
  // Escapes retain their original BF16 bits; only the code's virtual-K address is permuted back.
  const float wd=sc*3.0517578125e-05f;
  for(int e=ea;e<eb;++e) {
    const int2 en=ld_i64(esc+e);
    const int rl=en.x>>16,v=en.x&65535;
    if((rl&7)!=g)continue;
    const int k=(v&~63)+((v>>2)&3)*16+((v>>4)&3)*2+((v>>1)&1)*8+(v&1);
    const float w=__uint_as_float((uint32_t)en.y<<16);
#pragma unroll
    for(int nt=0;nt<MT/8;++nt)
#pragma unroll
      for(int q=0;q<2;++q) {
        const int tok=nt*8+t*2+q;
        if(tok<M) {
          const int kk=k-kb0*64, row=tok;
          const int off=(kk/64)*MT*64+row*64+((kk%64)^((row&7)*8));
          const float xv=__bfloat162float(xs[off]);
          if(rl&8)acc[nt*4+2+q]=fmaf(-wd,xv,fmaf(w,xv,acc[nt*4+2+q]));
          else acc[nt*4+q]=fmaf(-wd,xv,fmaf(w,xv,acc[nt*4+q]));
        }
      }
  }
#pragma unroll
  for(int nt=0;nt<MT/8;++nt)
#pragma unroll
    for(int q=0;q<2;++q) {
      const int tok=nt*8+t*2+q;
      if(tok<M) {
        float* out=P+(size_t)s*M*N+(size_t)tok*N+r*16+g;
        out[0]=acc[nt*4+q];out[8]=acc[nt*4+2+q];
      }
    }
}
