
#include <cuda_runtime.h>
#include <vector>
__global__ void mx_meta_kernel(const int*C,const int*ORDER,int*T13,int*T2,float*SS,int cap,int h13,int n13,int h2,int n2){
 int t=threadIdx.x,idx=blockIdx.x*blockDim.x+t;
 for(int i=idx;i<cap*64;i+=gridDim.x*blockDim.x)SS[i]=0.f;
 if(blockIdx.x==0&&t<32){int n=C[ORDER[t]],a=n>h13?(n+n13-1)/n13:0,b=n>h2?(n+n2-1)/n2:0;
  for(int d=1;d<32;d*=2){int aa=__shfl_up_sync(~0u,a,d),bb=__shfl_up_sync(~0u,b,d);if(t>=d){a+=aa;b+=bb;}}
  T13[t+1]=a;T2[t+1]=b;if(t==0){T13[0]=0;T2[0]=0;}
 }
}
void mx_meta(std::vector<int64_t>v,int64_t cap,int64_t h13,int64_t n13,int64_t h2,int64_t n2,int64_t stream){mx_meta_kernel<<<128,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(float*)v[4],cap,h13,n13,h2,n2);}
