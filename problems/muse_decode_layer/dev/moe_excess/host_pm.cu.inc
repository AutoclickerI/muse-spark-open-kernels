struct XLResource{int original,matched,threads,original_resident;};
template<int FAMILY,bool F>XLResource xl_resource(int M,int stages){
 int original,threads=FAMILY==1?128:256,dev,smemSM,n,major;cudaFuncAttributes a{};
 BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&smemSM,cudaDevAttrMaxSharedMemoryPerMultiprocessor,dev));
 if constexpr(FAMILY==2){original=pmz::smem_bytes(1,stages,M);auto k=pmz::phase<1,F,true,F,8>;BR_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,original));BR_CUDA_CHECK(cudaFuncGetAttributes(&a,k));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,threads,original));}
 int matched=std::max(original+2048,(smemSM/(n+1)+1023)&~1023);
 return {original,matched,threads,n};
}
#define XL_HARGS(v) (const int*)v[0],(const unsigned char*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(const __nv_bfloat16*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12]
#define XL_PARGS(v) (const __nv_bfloat16*)v[0],(const __nv_bfloat16*)v[1],(const uint8_t*)v[2],(const uint8_t*)v[3],(const float*)v[4],(const float*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11]

template<int FAMILY,bool F,int MODE,bool HIST>std::vector<int64_t> xl_launch(std::vector<int64_t>&v,int M,int stages,int grid,cudaStream_t st,uint64_t*trace,bool launch){
 XLResource r=xl_resource<FAMILY,F>(M,stages);cudaFuncAttributes a{};int resident;const int xl_dyn=MODE==-2?r.original:r.matched;
 auto config=[&](auto k){BR_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,xl_dyn));BR_CUDA_CHECK(cudaFuncGetAttributes(&a,k));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k,r.threads,xl_dyn));};
 if constexpr(MODE==7){throw std::runtime_error("stream runs in main module");}
 else if constexpr(MODE==-1||MODE==-2){
 if constexpr(FAMILY==2){auto k=pmz::phase<1,F,true,F,8>;config(k);if(launch)k<<<grid,256,xl_dyn,st>>>(XL_PARGS(v),M,stages);}}
 else{
 if constexpr(FAMILY==2){auto k=pmz::xphase<1,F,true,F,8,MODE,HIST>;config(k);if(launch)k<<<grid,256,xl_dyn,st>>>(XL_PARGS(v),M,stages,trace);}}
 if(launch)BR_CUDA_CHECK(cudaGetLastError());
 return {r.original,xl_dyn,r.threads,r.original_resident,resident,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes};
}
template<int FAMILY,bool F>std::vector<int64_t> xl_modes(std::vector<int64_t>&v,int mode,int M,int stages,int grid,cudaStream_t st,uint64_t*trace,bool launch,bool hist){
 if(hist)return xl_launch<FAMILY,F,4,true>(v,M,stages,grid,st,trace,launch);
 #define X(K) if(mode==K)return xl_launch<FAMILY,F,K,false>(v,M,stages,grid,st,trace,launch);
 X(-2) X(-1) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
 #undef X
 throw std::runtime_error("unknown ladder mode");
}
std::vector<int64_t> xl_run(std::vector<int64_t>v,int64_t family,int64_t first,int64_t mode,int64_t M,int64_t stages,int64_t grid,int64_t stream,int64_t trace,bool launch,bool hist){
 #define X(F) if(family==F){if(first)return xl_modes<F,true>(v,mode,M,stages,grid,(cudaStream_t)stream,(uint64_t*)trace,launch,hist);else return xl_modes<F,false>(v,mode,M,stages,grid,(cudaStream_t)stream,(uint64_t*)trace,launch,hist);}
 X(2)
 #undef X
 throw std::runtime_error("unknown ladder family");
}
