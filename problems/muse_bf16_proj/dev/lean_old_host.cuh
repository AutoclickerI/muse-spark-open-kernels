// Only the receipt-selected legacy SM90 kernels are instantiated.
template<int BT,int BK,int ST,bool OF>
int lean_old_launch(const void* w,const void* x,void* out,float* work,unsigned* counts,
                    int M,int N,int K,int KW,int mode,int split,int64_t work_elems,int64_t count_elems,cudaStream_t stream){
  if(N%64 || K%BK || KW%BK || K<1 || split<1)return lean_fail("old shape",-1);
  int nkb=KW/BK,tiles=(N/64)*((M+BT-1)/BT),total=tiles*nkb;
  if(mode!=2 && (split>16 || split>nkb))return lean_fail("old split",-1);
  int target=std::min(split,tiles*16),chunk=(total+target-1)/target,blocks=mode==2 ? (total+chunk-1)/chunk : tiles;
  if(work_elems<(int64_t)tiles*17*BT*64 || count_elems<tiles)return lean_fail("old workspace",-1);
  auto kern=muse_legacy::hop_kernel<BT,BK,ST,OF>;
  size_t shared=std::max((size_t)ST*(64+BT)*BK*2+ST*16,(size_t)BT*68*4);
  int device=0,optin=0;LEAN_CUDA(cudaGetDevice(&device));LEAN_CUDA(cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
  if(shared>(size_t)optin)return lean_fail("old shared memory",-1);
  static bool initialized=false;if(!initialized){LEAN_CUDA(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));LEAN_CUDA(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));initialized=true;}
  CUtensorMap mw,mx;int err=lean_map(&mw,w,N,KW,64);if(err)return err;err=lean_map(&mx,x,M,K,BT);if(err)return err;
  cudaLaunchConfig_t c={};c.gridDim=dim3(blocks,mode==2 ? 1 : split);c.blockDim=dim3(256);c.dynamicSmemBytes=shared;c.stream=stream;
  cudaLaunchAttribute attr={};if(mode==1 && split>1){attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=1;attr.val.clusterDim.y=split;attr.val.clusterDim.z=1;c.attrs=&attr;c.numAttrs=1;}
  auto e=cudaLaunchKernelEx(&c,kern,mw,mx,out,work,counts,M,N,nkb,K,mode,split,chunk,total);
  return e==cudaSuccess ? 0 : lean_fail("old launch",(int)e);
}
extern "C" int lean_old(const void* w,const void* x,void* out,void* work,void* counts,
                       int M,int N,int K,int KW,int of,int bt,int bk,int st,int mode,int split,
                       int64_t work_elems,int64_t count_elems,void* stream){
#define LOLD(BT,BK,ST,OF) if(bt==BT && bk==BK && st==ST && of==OF)return lean_old_launch<BT,BK,ST,OF>(w,x,out,(float*)work,(unsigned*)counts,M,N,K,KW,mode,split,work_elems,count_elems,(cudaStream_t)stream);
  LEAN_OLD_CASES
  return lean_fail("old specialization",-1);
}
