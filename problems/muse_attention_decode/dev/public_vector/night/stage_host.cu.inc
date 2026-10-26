template<int NL,bool POLY> static void stage_call(int grid,int sm,cudaStream_t st,const A&a,const WF&f,int x){
 if(x==68)apf<NL,68|(POLY?2048:0)>(grid,sm,st,a,f);
 else {TORCH_CHECK(x==324,"reviewed verify flags");apf<NL,324|(POLY?2048:0)>(grid,sm,st,a,f);}
}
void lee_attention(torch::Tensor qkv,torch::Tensor loc,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,
 torch::Tensor wo,torch::Tensor wml,torch::Tensor cnt,torch::Tensor out,torch::Tensor invf,torch::Tensor empty,
 int64_t mode,int64_t window,int64_t ns,int64_t x,double theta,double qk_factor,double sm_scale,double kv_scale,int64_t cap){
 auto st=at::cuda::getCurrentCUDAStream();const int B=seq.size(0),L=qkv.size(1)/B;TORCH_CHECK(L==8,"verify only");
 A a{};a.kc=(const uint8_t*)kc.data_ptr();a.vc=(const uint8_t*)vc.data_ptr();a.pt=pt.data_ptr<int>();a.seq=seq.data_ptr<int>();a.wo=wo.data_ptr<float>();a.wml=wml.data_ptr<float>();
 a.B=B;a.L=L;a.hq=16;a.hkv=2;a.ps=1;a.maxp=pt.size(1);a.window=window;a.nsplit=ns;a.rows=64;a.qk=sm_scale*kv_scale*qk_factor*LOG2E;
 WF f{(const __nv_bfloat16*)qkv.data_ptr(),loc.data_ptr<int64_t>(),invf.data_ptr<float>(),cnt.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),(float)theta,(float)(1./kv_scale),(float)(1./kv_scale),(float)kv_scale,1,nullptr,1};
 f.qv=view(qkv);int dev,sms;cudaGetDevice(&dev);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
 f.nmg=min((B*ns<=2*sms)?(int)cap:8,1<<(31-__builtin_clz((unsigned)ns)));
#if WGEMU
 const int nl=1;
#else
 const int nl=mode==4?1:2;
#endif
 const int sm=nl*(PRS*16384+8192+32768+2048)+1024+256+1024;
 const int grid=(x&256)?(2*sms/nl):B*ns/nl;
 if(nl==1){
#if WGEMU
  if(mode==5)stage_call<1,true>(grid,sm,st,a,f,x);else
#endif
  stage_call<1,false>(grid,sm,st,a,f,x);
 }
 else if(mode==5)stage_call<2,true>(grid,sm,st,a,f,x);
 else stage_call<2,false>(grid,sm,st,a,f,x);
 const auto e=cudaGetLastError();TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
}
template<int NL,int X> static std::vector<int64_t> stage_res(){
 const int sm=NL*(PRS*16384+8192+32768+2048)+1024+256+1024;
 cudaFuncSetAttribute(attnp<NL,X>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);
 cudaFuncAttributes a{};cudaFuncGetAttributes(&a,attnp<NL,X>);int blocks=0;
 cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,attnp<NL,X>,128*(2*NL+1),sm);
 return {a.numRegs,(int64_t)a.localSizeBytes,sm,blocks};
}
std::vector<int64_t> stage_resources(int64_t mode,int64_t x){
#if WGEMU
 if(mode==5)return x==68?stage_res<1,2116>():stage_res<1,2372>();
 return x==68?stage_res<1,68>():stage_res<1,324>();
#else
 if(mode==4)return x==68?stage_res<1,68>():stage_res<1,324>();
 if(mode==5)return x==68?stage_res<2,2116>():stage_res<2,2372>();
 return x==68?stage_res<2,68>():stage_res<2,324>();
#endif
}
