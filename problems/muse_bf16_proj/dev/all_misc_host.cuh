extern "C" int all_gemv(const void* w,const void* x,void* out,int N,int K,int variant,void* stream_ptr){
  auto W=(const __nv_bfloat16*)w;auto X=(const __nv_bfloat16*)x;auto Y=(__nv_bfloat16*)out;
  auto stream=(cudaStream_t)stream_ptr;int rows=N>=8192 ? 2 : 1;
  if(variant==0)all_gemv0::gemv_kernel<<<(N+7)/8,256,K*2,stream>>>(W,X,Y,N,K);
  else if(variant==1)all_gemv1::gemv_kernel<<<(N+8*rows-1)/(8*rows),256,K*2,stream>>>(W,X,Y,N,K);
  else if(variant==2)all_gemv2::gemv_kernel<<<(N+16*rows-1)/(16*rows),512,K*2,stream>>>(W,X,Y,N,K);
  else if(variant==3){
    if(N==2304 && K==8192)all_gemv3::gemv_kernel_t<2304,8192,1,8><<<288,256,0,stream>>>(W,X,Y);
    else if(N==4096 && K==8192)all_gemv3::gemv_kernel_t<4096,8192,1,8><<<512,256,0,stream>>>(W,X,Y);
    else if(N==8192 && K==1024)all_gemv3::gemv_kernel_t<8192,1024,2,8><<<512,256,0,stream>>>(W,X,Y);
    else if(N==1024 && K==4096)all_gemv3::gemv_kernel_t<1024,4096,1,8><<<128,256,0,stream>>>(W,X,Y);
    else return lean_fail("static gemv shape",-1);
  }else return lean_fail("gemv variant",-1);
  return (int)cudaGetLastError();
}
template<int BT,int KT>
int all_tc_launch(const void* w,const void* x,void* out,int M,int N,int K,cudaStream_t stream){
  auto kernel=all_tc::tc_slab_kernel<BT,KT>;
  constexpr size_t shared=4*(16+BT)*KT+8*BT*20*4;
  static bool initialized=false;
  if(!initialized){LEAN_CUDA(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));initialized=true;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(N/16,1,(M+BT-1)/BT);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=shared;cfg.stream=stream;
  return (int)cudaLaunchKernelEx(&cfg,kernel,(const __nv_bfloat16*)w,(const __nv_bfloat16*)x,(__nv_bfloat16*)out,M,N,K);
}
extern "C" int all_tc_run(const void* w,const void* x,void* out,int M,int N,int K,int bt,int kt,void* stream){
#define ALL_TC(BT,KT) if(bt==BT && kt==KT)return all_tc_launch<BT,KT>(w,x,out,M,N,K,(cudaStream_t)stream);
  ALL_TC(16,128) ALL_TC(16,256) ALL_TC(32,128) ALL_TC(32,256)
  return lean_fail("tc specialization",-1);
}
extern "C" int all_m16(const void* w,const void* x,void* out,int N,int K,void* stream){
  all_scalar::m16_kernel<<<(N+7)/8,256,16*256*2,(cudaStream_t)stream>>>((const __nv_bfloat16*)w,(const __nv_bfloat16*)x,(__nv_bfloat16*)out,N,K);
  return (int)cudaGetLastError();
}
