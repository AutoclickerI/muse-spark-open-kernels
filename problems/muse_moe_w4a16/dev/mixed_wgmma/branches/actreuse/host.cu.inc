static CUtensorMap ax_map(void*ptr,uint64_t rows,int nt){
 static PFN_cuTensorMapEncodeTiled_v12000 enc=nullptr;
 if(!enc){cudaDriverEntryPointQueryResult q;BR_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",(void**)&enc,cudaEnableDefault,&q));BR_REQUIRE(q==cudaDriverEntryPointSuccess&&enc);}
 CUtensorMap m;cuuint64_t dims[2]={4096,rows},strides[1]={8192};cuuint32_t box[2]={64,(unsigned)nt},es[2]={1,1};
 BR_REQUIRE(enc(&m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS);return m;
}
void ax_gather(std::vector<int64_t>v,int64_t gx,int64_t M,int64_t stream){ax::gather<<<32*((M+15)/16),256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)v[3],(__nv_bfloat16*)gx,(const int*)v[4],(const int*)v[5],(const int*)v[8],M);BR_CUDA_CHECK(cudaGetLastError());}
template<int K,int CS,bool F>int ax_bytes(){int dev,major;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));return major==9?ax::bytes<K==6?3:4,K==5?32:64,K>=5?2:1>():ax::bytes<K==6?2:3,K==5?32:64,K>=5?2:1>();}
template<int K,int CS,bool F>std::vector<int64_t> ax_resource(){auto fn=ax::kernel<K,CS,F>;int bytes=ax_bytes<K,CS,F>();cudaFuncAttributes a{};int n;
 BR_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));BR_CUDA_CHECK(cudaFuncGetAttributes(&a,fn));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,fn,256,bytes));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,n};}
template<int K,int CS,bool F>void ax_launch(std::vector<int64_t>&v,int M,cudaStream_t stream){
 int dev,major,sms;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev));int grid=sms*4;
 CUtensorMap m32{},m64{};if constexpr(K>0&&K<5){m32=ax_map((void*)v[3],M*8+32*128,32);m64=ax_map((void*)v[3],M*8+32*128,64);}
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=ax_bytes<K,CS,F>();cfg.stream=stream;
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=CS;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;cfg.attrs=&attr;cfg.numAttrs=1;
 if(major==9){BR_CUDA_CHECK(cudaLaunchKernelEx(&cfg,ax::kernel<K,CS,F>,BR_ARGS(v),M,m32,m64));}
 else{ax::kernel<K,CS,F><<<grid,256,ax_bytes<K,CS,F>(),stream>>>(BR_ARGS(v),M,m32,m64);BR_CUDA_CHECK(cudaGetLastError());}
}
#define AX_CASES(F) F(0,1) F(1,1) F(2,2) F(4,4) F(5,1) F(6,1)
std::vector<int64_t> ax_info(int64_t k,int64_t f){
#define AX_INFO(K,C) if(k==K)return f?ax_resource<K,C,true>():ax_resource<K,C,false>();
 AX_CASES(AX_INFO)
#undef AX_INFO
 BR_REQUIRE(false);return {};}
void ax_run(std::vector<int64_t>v,int64_t kind,int64_t first,int64_t M,int64_t stream){
#define AX_GO(K,C) if(kind==K){if(first)ax_launch<K,C,true>(v,M,(cudaStream_t)stream);else ax_launch<K,C,false>(v,M,(cudaStream_t)stream);return;}
 AX_CASES(AX_GO)
#undef AX_GO
 BR_REQUIRE(false);}
