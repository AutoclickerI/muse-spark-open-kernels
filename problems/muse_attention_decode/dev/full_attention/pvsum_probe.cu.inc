
__global__ void pv_den_test(const uint8_t*P,const float*A,float*out){
 extern __shared__ __align__(1024) uint8_t sm[];
 int tid=threadIdx.x,lane=tid&31,g=lane>>2,t=lane&3,w=tid>>5;
 if(tid<32)*(uint4*)(sm+CLKPAD+24576+16*tid)=make_uint4(0x38383838,0x38383838,0x38383838,0x38383838);
 proxy_fence();__syncthreads();
 float d[4]={};
#pragma unroll
 for(int step=0;step<2;step++){
  if(step){d[0]*=A[16*w+g];d[1]*=A[16*w+g];d[2]*=A[16*w+g+8];d[3]*=A[16*w+g+8];}
  uint32_t allp[16];
#pragma unroll
  for(int j=0;j<4;j++){
#pragma unroll
   for(int i=0;i<4;i++){
    int row=16*w+g+8*(i&1),col=32*j+16*(i>>1)+2*t;
    auto p=P+step*64*128+row*128+col;
    allp[4*j+i]=uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[8])<<16)|(uint32_t(p[9])<<24);
   }
  }
  protect(allp);
  for(int i=0;i<4;i++)asm volatile("":"+f"(d[i])::"memory");
  wg_fence();
#pragma unroll
  for(int j=0;j<4;j++)pv_den(d,allp+4*j);
  wg_commit();wg_wait<0>();
  protect(allp);
  for(int i=0;i<4;i++)asm volatile("":"+f"(d[i])::"memory");
 }
 for(int i=0;i<4;i++)out[4*tid+i]=d[i];
}
void pv_den_probe(int64_t p,int64_t a,int64_t y,int64_t stream){
 pv_den_test<<<1,128,CLKPAD+25600,(cudaStream_t)stream>>>((uint8_t*)p,(float*)a,(float*)y);
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
