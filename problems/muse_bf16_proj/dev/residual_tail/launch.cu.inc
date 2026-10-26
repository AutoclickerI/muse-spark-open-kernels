template<bool POST,int MODE>void rt_launch(std::vector<int64_t>v,int T,cudaStream_t stream){
 auto kernel=glue_cl::boundary_tail<8,256,1,POST,MODE>;
 cudaFuncSetAttribute(kernel,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(8,T);cfg.blockDim=dim3(256);cfg.stream=stream;
 cudaLaunchAttribute a[2]{};a[0].id=cudaLaunchAttributeClusterDimension;a[0].val.clusterDim.x=8;a[0].val.clusterDim.y=1;a[0].val.clusterDim.z=1;
 a[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;a[1].val.programmaticStreamSerializationAllowed=1;
 cfg.attrs=a;cfg.numAttrs=2;
 auto e=cudaLaunchKernelEx(&cfg,kernel,(const void*)v[0],(int)v[1],(int)v[2],(long long)v[3],(long long)v[4],(int)v[5],
  (const float*)v[6],(float*)v[7],(__nv_bfloat16*)v[8],(const float*)v[9],(const float*)v[10],(const float*)v[11],(const float*)v[12],(const float*)v[13],(float*)v[14],(float*)v[15]);
 if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
void rt_boundary(std::vector<int64_t>v,int64_t T,int64_t post,int64_t mode,int64_t stream){
#define RT(M) if(mode==M){if(post)rt_launch<true,M>(v,T,(cudaStream_t)stream);else rt_launch<false,M>(v,T,(cudaStream_t)stream);return;}
 RT(1) RT(2) RT(3)
#undef RT
 throw std::runtime_error("rt mode");
}
std::vector<int64_t> rt_info(int64_t post,int64_t mode){cudaFuncAttributes a{};
#define RT(M) if(mode==M){if(post)cudaFuncGetAttributes(&a,glue_cl::boundary_tail<8,256,1,true,M>);else cudaFuncGetAttributes(&a,glue_cl::boundary_tail<8,256,1,false,M>);}
 RT(1) RT(2) RT(3)
#undef RT
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes};
}
