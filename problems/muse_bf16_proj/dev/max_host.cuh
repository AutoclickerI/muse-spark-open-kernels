template<int NT,int TR,int U,int NA,int ST,int LD>
int max_tile_launch(const void* w,const void* lo,const void* x,void* out,int M,int N,int K,int wn,int ks,cudaStream_t stream){
  constexpr int BT=8*NT;int BR=wn*TR*16;
  if(N%BR || K%ks || (K/ks)%(32*U) || wn>8 || ks>16)return lean_fail("tile shape",-1);
  auto kernel=muse_tile::tile_kernel<NT,TR,U,NA,ST,LD>;
  size_t shared=muse_tile::tile_smem<NT,TR,U,NA,ST,LD>(wn);
  static bool init=false;if(!init){LEAN_CUDA(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,96*1024));LEAN_CUDA(cudaFuncSetAttribute(kernel,cudaFuncAttributeNonPortableClusterSizeAllowed,1));init=true;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(N/BR,ks,(M+BT-1)/BT);cfg.blockDim=dim3(wn*32);cfg.dynamicSmemBytes=shared;cfg.stream=stream;
  cudaLaunchAttribute attr={};if(ks>1){attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=1;attr.val.clusterDim.y=ks;attr.val.clusterDim.z=1;cfg.attrs=&attr;cfg.numAttrs=1;}
  return (int)cudaLaunchKernelEx(&cfg,kernel,(const __nv_bfloat16*)w,(const __nv_bfloat16*)lo,(const __nv_bfloat16*)x,out,M,N,K,K/ks,wn);
}
extern "C" int max_tile(const void* w,const void* lo,const void* x,void* out,int M,int N,int K,
                        int nt,int tr,int u,int wn,int ks,int split2,int st,int ld,void* stream){
  int na=split2 ? 2 : 1;
#define MAX_TILE(NT,TR,U,NA,ST,LD) if(nt==NT && tr==TR && u==U && na==NA && st==ST && ld==LD)return max_tile_launch<NT,TR,U,NA,ST,LD>(w,lo,x,out,M,N,K,wn,ks,(cudaStream_t)stream);
  MAX_TILE_CASES
  return lean_fail("tile specialization",-1);
}
template<int BT,int NS,int NC,int ST,int MC,bool OF>
int max_wide_launch(const void* w,const void* x,void* out,int M,int N,int K,int KW,int ks,cudaStream_t stream){
  constexpr int BR=64*NS*NC;
  if(N%BR || (N/BR)%MC || K%64 || KW%64 || ks*MC>16 || BT%(8*MC))return lean_fail("wide shape",-1);
  auto kernel=muse_wide::wg_kernel<BT,NS,NC,ST,MC,OF>;
  size_t shared=std::max((size_t)ST*(BR+BT)*128+16*ST+1024,(size_t)64*(BR+4)*4+1024);
  static bool init=false;if(!init){LEAN_CUDA(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));LEAN_CUDA(cudaFuncSetAttribute(kernel,cudaFuncAttributeNonPortableClusterSizeAllowed,1));init=true;}
  CUtensorMap mw,mx;int err=lean_map(&mw,w,N,KW,BR);if(err)return err;err=lean_map(&mx,x,M,K,BT/MC);if(err)return err;
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(N/BR,ks,(M+BT-1)/BT);cfg.blockDim=dim3(128*(NC+1));cfg.dynamicSmemBytes=shared;cfg.stream=stream;
  cudaLaunchAttribute attr={};if(ks>1 || MC>1){attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=MC;attr.val.clusterDim.y=ks;attr.val.clusterDim.z=1;cfg.attrs=&attr;cfg.numAttrs=1;}
  return (int)cudaLaunchKernelEx(&cfg,kernel,mw,mx,out,M,N,KW/64,K);
}
extern "C" int max_wide(const void* w,const void* x,void* out,int M,int N,int K,int KW,int of,
                        int bt,int ns,int nc,int st,int ks,int mc,void* stream){
#define MAX_WIDE(BT,NS,NC,ST,MC,OF) if(bt==BT && ns==NS && nc==NC && st==ST && mc==MC && of==OF)return max_wide_launch<BT,NS,NC,ST,MC,OF>(w,x,out,M,N,K,KW,ks,(cudaStream_t)stream);
  MAX_WIDE_CASES
  return lean_fail("wide specialization",-1);
}
