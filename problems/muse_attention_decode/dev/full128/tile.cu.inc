__global__ void full128_tile_test(const uint8_t* qp,const uint8_t* kp,const uint8_t* vp,float* out,float sc,int masked) {
  extern __shared__ __align__(1024) uint8_t shared[];
  uint8_t *Q=shared,*K=Q+8192,*VR=K+16384,*VT=VR+16384;
  const int tid=threadIdx.x,w=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
  for(int i=tid;i<2*128*4;i+=128) {
    const int h=i/(128*4),r=(i/4)%128,c=i%4;
    *(uint4*)(K+h*8192+off(r,16*c))=((const uint4*)kp)[i];
    *(uint4*)(VR+h*8192+off(r,16*c))=((const uint4*)vp)[i];
  }
  for(int i=tid;i<2*64*4;i+=128) {
    const int h=i/(64*4),r=(i/4)%64,c=i%4;
    *(uint4*)(Q+h*4096+off(r,16*c))=((const uint4*)qp)[i];
  }
  __syncthreads();transpose_group(VR,VT,w,lane);__syncthreads();proxy_fence();
  for(int h=0;h<2;h++) {
    float sx[16][4],O[8][4]={},m[2]={-INFINITY,-INFINITY},l[2]={},alpha[2];uint32_t pk[16];
    protect(sx);wg_fence();qk128(sx,Q+h*4096,K+h*8192);wg_commit();wg_wait<0>();protect(sx);
    int lo[2]={0,0},hi[2]={96+2*w,97+2*w};
    if(masked)softmax<16,true>(sx,m,l,alpha,pk,sc,0,lo,hi,true);
    else softmax<16,false>(sx,m,l,alpha,pk,sc,0,lo,hi,true);
    protect(pk);protect(O);wg_fence();pv128(O,pk,VT+h*8192);wg_commit();wg_wait<0>();protect(O);
#pragma unroll
    for(int rh=0;rh<2;rh++) {
      float den=l[rh];den+=__shfl_xor_sync(0xffffffffu,den,1);den+=__shfl_xor_sync(0xffffffffu,den,2);
      const int row=16*w+g+8*rh;
#pragma unroll
      for(int n=0;n<8;n++)*(float2*)(out+(h*64+row)*64+8*n+2*t)=make_float2(O[n][2*rh]/den,O[n][2*rh+1]/den);
    }
  }
}
void full128_tile_forward(int64_t q,int64_t k,int64_t v,int64_t out,double sc,int64_t masked,int64_t stream) {
  cudaFuncSetAttribute(full128_tile_test,cudaFuncAttributeMaxDynamicSharedMemorySize,57344);
  full128_tile_test<<<1,128,57344,(cudaStream_t)stream>>>((uint8_t*)q,(uint8_t*)k,(uint8_t*)v,(float*)out,sc,masked);
  auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
