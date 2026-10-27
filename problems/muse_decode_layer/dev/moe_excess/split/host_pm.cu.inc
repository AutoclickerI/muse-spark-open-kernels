// Shared reservation kinds: 0 original, 1 original+2 KiB scratch, 2 force one CTA/SM,
// 3 exact ring (DEPTH 3 heavy fits three CTAs/SM), 4 force two CTAs/SM.
static int xl_smem(int kind,int original,int smemSM){
 if(kind==0||kind==3)return original;
 if(kind==1)return original+2048;
 if(kind==2)return std::max(original+2048,(smemSM/2+1023)&~1023);
 return std::max(original+2048,(smemSM/3+1023)&~1023);
}
#define XL_HARGS(v) (const int*)v[0],(const unsigned char*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(const __nv_bfloat16*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12]
#define XL_PARGS(v) (const __nv_bfloat16*)v[0],(const __nv_bfloat16*)v[1],(const uint8_t*)v[2],(const uint8_t*)v[3],(const float*)v[4],(const float*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11]

template<int FAMILY,bool F,int MODE,bool HIST,int DEPTH>std::vector<int64_t> xl_launch(std::vector<int64_t>&v,int M,int stages,int grid,cudaStream_t st,uint64_t*trace,bool launch,int kind){
 int dev,smemSM,major,original,threads=FAMILY==1?128:256,resident;cudaFuncAttributes a{};
 BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&smemSM,cudaDevAttrMaxSharedMemoryPerMultiprocessor,dev));
 if constexpr(FAMILY==2)original=pmz::smem_bytes(1,stages,M);
 const int dyn=xl_smem(kind,original,smemSM);
 auto config=[&](auto k){BR_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn));BR_CUDA_CHECK(cudaFuncGetAttributes(&a,k));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k,threads,dyn));};
 if constexpr(MODE==7){throw std::runtime_error("stream runs in main module");}
 else if constexpr(MODE==-2){
 if constexpr(FAMILY==2){auto k=pmz::phase<1,F,true,F,8>;config(k);if(launch)k<<<grid,256,dyn,st>>>(XL_PARGS(v),M,stages);}}
 else{
 if constexpr(FAMILY==2){auto k=pmz::xphase<1,F,true,F,8,MODE,HIST>;config(k);if(launch)k<<<grid,256,dyn,st>>>(XL_PARGS(v),M,stages,trace);}}
 if(launch)BR_CUDA_CHECK(cudaGetLastError());
 return {original,dyn,threads,resident,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes};
}
template<int FAMILY,bool F>std::vector<int64_t> xl_modes(std::vector<int64_t>&v,int mode,int depth,int M,int stages,int grid,cudaStream_t st,uint64_t*trace,bool launch,bool hist,int kind){
 if(hist&&mode==4)return xl_launch<FAMILY,F,4,true,4>(v,M,stages,grid,st,trace,launch,kind);
 if(hist&&mode==8)return xl_launch<FAMILY,F,8,true,4>(v,M,stages,grid,st,trace,launch,kind);
 if(hist)throw std::runtime_error("unknown hist mode");
 if(depth!=4)throw std::runtime_error("depth");
 #define X(K) if(mode==K)return xl_launch<FAMILY,F,K,false,4>(v,M,stages,grid,st,trace,launch,kind);
 X(-2) X(0) X(8) X(9) X(12) X(13)
 #undef X
 throw std::runtime_error("unknown split mode");
}
std::vector<int64_t> xl_run(std::vector<int64_t>v,int64_t family,int64_t first,int64_t mode,int64_t depth,int64_t M,int64_t stages,int64_t grid,int64_t stream,int64_t trace,bool launch,bool hist,int64_t kind){
 #define X(F) if(family==F){if(first)return xl_modes<F,true>(v,mode,depth,M,stages,grid,(cudaStream_t)stream,(uint64_t*)trace,launch,hist,kind);else return xl_modes<F,false>(v,mode,depth,M,stages,grid,(cudaStream_t)stream,(uint64_t*)trace,launch,hist,kind);}
 X(2)
 #undef X
 throw std::runtime_error("unknown split family");
}
