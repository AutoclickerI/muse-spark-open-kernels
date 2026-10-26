static CUtensorMap probe_map(void*ptr,uint64_t rows,int nt){
 static PFN_cuTensorMapEncodeTiled_v12000 enc=nullptr;
 if(!enc){cudaDriverEntryPointQueryResult q;BR_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",(void**)&enc,cudaEnableDefault,&q));BR_REQUIRE(q==cudaDriverEntryPointSuccess&&enc);}
 CUtensorMap m;cuuint64_t dims[2]={4096,rows},strides[1]={8192};cuuint32_t box[2]={64,(unsigned)nt},es[2]={1,1};
 BR_REQUIRE(enc(&m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS);return m;
}

template<int K,bool F>int probe_bytes(){int dev,major;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));return K<2?93184:(major==9?102400:76800);}
template<int K,bool F>std::vector<int64_t> probe_resource(){auto fn=probe::kernel<K,F>;int bytes=probe_bytes<K,F>();cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));BR_CUDA_CHECK(cudaFuncGetAttributes(&a,fn));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,fn,256,bytes));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,n};}
template<int K,bool F>void probe_launch(std::vector<int64_t>&v,int M,cudaStream_t stream){int dev,major,sms;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev));int grid=sms*4;CUtensorMap m32{},m64{};
 if constexpr(K>=2){m32=probe_map((void*)v[3],M*8+32*128,32);m64=probe_map((void*)v[3],M*8+32*128,64);}
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=probe_bytes<K,F>();cfg.stream=stream;cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=K>=2?2:1;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;cfg.attrs=&attr;cfg.numAttrs=1;
 if(major==9){BR_CUDA_CHECK(cudaLaunchKernelEx(&cfg,probe::kernel<K,F>,BR_ARGS(v),M,m32,m64));}else{probe::kernel<K,F><<<grid,256,probe_bytes<K,F>(),stream>>>(BR_ARGS(v),M,m32,m64);BR_CUDA_CHECK(cudaGetLastError());}}
std::vector<int64_t> probe_info(int64_t k,int64_t f){
#define PR_INFO(K) if(k==K)return f?probe_resource<K,true>():probe_resource<K,false>();
 PR_INFO(0) PR_INFO(1) PR_INFO(2) PR_INFO(3)
#undef PR_INFO
 BR_REQUIRE(false);return {};}
void probe_run(std::vector<int64_t>v,int64_t k,int64_t f,int64_t M,int64_t stream){
#define PR_GO(K) if(k==K){if(f)probe_launch<K,true>(v,M,(cudaStream_t)stream);else probe_launch<K,false>(v,M,(cudaStream_t)stream);return;}
 PR_GO(0) PR_GO(1) PR_GO(2) PR_GO(3)
#undef PR_GO
 BR_REQUIRE(false);}
