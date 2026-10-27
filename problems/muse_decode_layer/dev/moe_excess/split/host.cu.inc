namespace xlad {
template<int FAMILY,bool F>__global__ void stream(const uint8_t*Q,const uint8_t*SC,const float*GS,const uint8_t*PNW,const int*C,uint64_t*out){
 extern __shared__ unsigned char hold[];
 __shared__ int experts[32],ne;
 constexpr int N=F?8192:4096,QC=FAMILY==2?(N/16*36864/16384):(N*2048/16384),SCN=FAMILY==2?0:N*256/16384,EC=QC+SCN,TH=FAMILY==1?128:256;
 int t=threadIdx.x;if(t==0){int n=0;for(int e=0;e<32;e++){int c=C[e];if((FAMILY==0&&c>16)||(FAMILY==1&&c>8&&c<=16)||(FAMILY==2&&c>0&&c<=8))experts[n++]=e;}ne=n;}__syncthreads();
 uint32_t a=0,b=0,c=0,d=0;
 for(int u=blockIdx.x;u<ne*EC;u+=gridDim.x){int ix=u/EC,ch=u%EC,e=experts[ix];const uint8_t*p=ch<QC?Q+(size_t)e*QC*16384+ch*16384:SC+(size_t)e*SCN*16384+(ch-QC)*16384;
 #pragma unroll
 for(int j=0;j<1024/TH;j++){const uint8_t*ptr=p+(t+j*TH)*16;uint4 v;asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(ptr):"memory");a^=v.x;b^=v.y;c^=v.z;d^=v.w;}}
 if(blockIdx.x==0){for(int i=t;i<ne;i+=TH){int e=experts[i];unsigned v;asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(v):"l"(GS+e):"memory");a^=v;if constexpr(F){asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(v):"l"(GS+e+32):"memory");b^=v;}}
 if constexpr(FAMILY==0&&!F)for(int i=t;i<512;i+=TH){uint4 v;asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(PNW+i*16):"memory");a^=v.x;b^=v.y;c^=v.z;d^=v.w;}}
 a^=b^c^d;a=__reduce_xor_sync(0xffffffffu,a);if((t&31)==0)out[blockIdx.x*8+(t>>5)]=a;
}
}
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
 if constexpr(FAMILY==0)original=major==9?chunk::dyn<DEPTH,64>():chunk::dyn<3,64>();
 if constexpr(FAMILY==1)original=mwbr::smem_bytes<16>();
 if constexpr(FAMILY==2)original=pmz::smem_bytes(1,stages,M);
 const int dyn=xl_smem(kind,original,smemSM);
 auto config=[&](auto k){BR_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn));BR_CUDA_CHECK(cudaFuncGetAttributes(&a,k));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,k,threads,dyn));};
 if constexpr(MODE==7){auto k=xlad::stream<FAMILY,F>;config(k);if(launch){const uint8_t*q=(uint8_t*)v[FAMILY==2?(F?2:3):0];const uint8_t*sc=(uint8_t*)v[1];const float*gs=(float*)v[FAMILY==2?(F?4:5):2];const uint8_t*nw=(uint8_t*)v[FAMILY==2?1:9];
 k<<<grid,threads,dyn,st>>>(q,sc,gs,nw,(int*)v[FAMILY==2?6:4],trace);}}
 else if constexpr(MODE==-2){
 if constexpr(FAMILY==0){auto k=chunk::kernel<4,F>;config(k);if(launch)k<<<grid,256,dyn,st>>>(XL_HARGS(v),M);}
 if constexpr(FAMILY==1){auto k=mwbr::gemm<16,F,F?7:6,1,8,16>;config(k);if(launch)k<<<grid,128,dyn,st>>>(XL_HARGS(v),M,0x70007000u);}
 if constexpr(FAMILY==2){auto k=pmz::phase<1,F,true,F,8>;config(k);if(launch)k<<<grid,256,dyn,st>>>(XL_PARGS(v),M,stages);}}
 else{
 if constexpr(FAMILY==0){auto k=chunk::xkernel<DEPTH,F,MODE,HIST>;config(k);if(launch)k<<<grid,256,dyn,st>>>(XL_HARGS(v),M,trace);}
 if constexpr(FAMILY==1){auto k=mwbr::xgemm<16,F,F?7:6,1,8,16,MODE,HIST>;config(k);if(launch)k<<<grid,128,dyn,st>>>(XL_HARGS(v),M,0x70007000u,trace);}
 if constexpr(FAMILY==2){auto k=pmz::xphase<1,F,true,F,8,MODE,HIST>;config(k);if(launch)k<<<grid,256,dyn,st>>>(XL_PARGS(v),M,stages,trace);}}
 if(launch)BR_CUDA_CHECK(cudaGetLastError());
 return {original,dyn,threads,resident,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes};
}
template<int FAMILY,bool F>std::vector<int64_t> xl_modes(std::vector<int64_t>&v,int mode,int depth,int M,int stages,int grid,cudaStream_t st,uint64_t*trace,bool launch,bool hist,int kind){
 if(hist&&mode==4)return xl_launch<FAMILY,F,4,true,4>(v,M,stages,grid,st,trace,launch,kind);
 if(hist&&mode==8)return xl_launch<FAMILY,F,8,true,4>(v,M,stages,grid,st,trace,launch,kind);
 if(hist)throw std::runtime_error("unknown hist mode");
 if constexpr(FAMILY==0)if(depth==3){if(mode==8)return xl_launch<FAMILY,F,8,false,3>(v,M,stages,grid,st,trace,launch,kind);throw std::runtime_error("depth 3 is mode 8 only");}
 if(depth!=4)throw std::runtime_error("depth");
 #define X(K) if(mode==K)return xl_launch<FAMILY,F,K,false,4>(v,M,stages,grid,st,trace,launch,kind);
 X(-2) X(0) X(7) X(8) X(9) X(12) X(13)
 if constexpr(FAMILY==0){X(10) X(11)}
 #undef X
 throw std::runtime_error("unknown split mode");
}
std::vector<int64_t> xl_run(std::vector<int64_t>v,int64_t family,int64_t first,int64_t mode,int64_t depth,int64_t M,int64_t stages,int64_t grid,int64_t stream,int64_t trace,bool launch,bool hist,int64_t kind){
 #define X(F) if(family==F){if(first)return xl_modes<F,true>(v,mode,depth,M,stages,grid,(cudaStream_t)stream,(uint64_t*)trace,launch,hist,kind);else return xl_modes<F,false>(v,mode,depth,M,stages,grid,(cudaStream_t)stream,(uint64_t*)trace,launch,hist,kind);}
 X(0) X(1) X(2)
 #undef X
 throw std::runtime_error("unknown split family");
}
