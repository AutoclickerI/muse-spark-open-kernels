// Owned host ABI for the reviewed Lee v56 kernels; shape choices match that pin.
// Changes: expose only the four selected families/flags, check launch errors.
void lee_attention(torch::Tensor qkv,torch::Tensor loc,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,
 torch::Tensor wo,torch::Tensor wml,torch::Tensor cnt,torch::Tensor out,torch::Tensor invf,torch::Tensor empty,
 int64_t kind,int64_t window,int64_t ns,int64_t x,double theta,double qk_factor,double sm_scale,double kv_scale,int64_t nmgcap){
 auto st=at::cuda::getCurrentCUDAStream();const int B=seq.size(0),L=qkv.size(1);
 if(kind==0){run_fdec(qkv,loc,kc,vc,pt,seq,wo,wml,cnt,out,1,window,theta,qk_factor,sm_scale*kv_scale,kv_scale,ns,empty,invf,1,window?5:(B==1?7:5));}
 else{
  A a{};a.kc=(const uint8_t*)kc.data_ptr();a.vc=(const uint8_t*)vc.data_ptr();a.pt=pt.data_ptr<int>();a.seq=seq.data_ptr<int>();a.wo=wo.data_ptr<float>();a.wml=wml.data_ptr<float>();
  a.B=B;a.L=L;a.hq=16;a.hkv=2;a.ps=1;a.maxp=pt.size(1);a.window=window;a.nsplit=ns;a.rows=L*8;a.qk=sm_scale*kv_scale*qk_factor*LOG2E;
  WF f{(const __nv_bfloat16*)qkv.data_ptr(),loc.data_ptr<int64_t>(),invf.data_ptr<float>(),cnt.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),(float)theta,(float)(1./kv_scale),(float)(1./kv_scale),(float)kv_scale,1,nullptr,1};
  int dev,sms;cudaGetDevice(&dev);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
#if WGEMU
  constexpr int nl=1;
#else
  constexpr int nl=2;
#endif
  if(kind==1){
   f.nmg=2;const int smd=nl*(DRS*16384+32768+4096+2048+2048)+256+nl*1024+1024;
   adf<nl,64>(B*ns/nl,smd,st,a,f);
  }else if(kind==2){
   const int smw=RSW*16384+8192+32768+4096+1024+1024;f.nmg=min(8,1<<(31-__builtin_clz((unsigned)ns)));
   cudaFuncSetAttribute(attnw39<true,true,true,8>,cudaFuncAttributeMaxDynamicSharedMemorySize,smw);
   attnw39<true,true,true,8><<<B*ns,256,smw,st>>>(a,f);
  }else{
   TORCH_CHECK(kind==3,"unknown Lee family");
   f.nmg=min((B*ns<=nl*sms)?(int)nmgcap:8,1<<(31-__builtin_clz((unsigned)ns)));
   const int smp=nl*(PRS*16384+8192+32768+2048)+1024+256+1024;
   const int grid=(x&256)?sms:B*ns/nl;
   if(x==68)apf<nl,68>(grid,smp,st,a,f);
   else if(x==32836)apf<nl,32836>(grid,smp,st,a,f);
   else {TORCH_CHECK(x==324,"unsupported Lee flag");apf<nl,324>(grid,smp,st,a,f);}
  }
 }
 const auto e=cudaGetLastError();TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
}
