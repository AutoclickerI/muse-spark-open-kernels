namespace {
__global__ void kmajor_probe_kernel(const uint8_t*K,const uint8_t*Q,float*out){
 __shared__ __align__(1024) uint8_t qs[512];
 for(int i=threadIdx.x;i<512;i+=128)qs[q8off(i/64,i%64)]=Q[i];
 asm volatile("fence.proxy.async.shared::cta;" ::: "memory");__syncthreads();
 float x[4];key_query(x,K,qs,0);int r=16*(threadIdx.x>>5)+((threadIdx.x&31)>>2),c=2*(threadIdx.x&3);
 out[r*8+c]=x[0];out[r*8+c+1]=x[1];out[(r+8)*8+c]=x[2];out[(r+8)*8+c+1]=x[3];
}
}
void kmajor_probe(int64_t k,int64_t q,int64_t out,int64_t stream){kmajor_probe_kernel<<<1,128,0,(cudaStream_t)stream>>>((const uint8_t*)k,(const uint8_t*)q,(float*)out);auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
