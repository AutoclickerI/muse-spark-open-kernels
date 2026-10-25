#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/util/packed_stride.hpp>

static thread_local void* vd_function=nullptr;
namespace vd {
using namespace cute;
using BF=cutlass::bfloat16_t;
template<int BN,int BK,int CX,int CY,bool STREAM,typename Out>
struct Gemm {
  using Tile=Shape<Int<STREAM?128:64>,Int<BN>,Int<BK>>;
  using Cluster=Shape<Int<CX>,Int<CY>,_1>;
  using Acc=float;
  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Acc,Acc,Out,cutlass::layout::ColumnMajor,128/cutlass::sizeof_bits<Out>::value,
    Out,cutlass::layout::ColumnMajor,128/cutlass::sizeof_bits<Out>::value,
    std::conditional_t<STREAM,cutlass::epilogue::TmaWarpSpecializedCooperative,cutlass::epilogue::TmaWarpSpecialized>>::CollectiveOp;
  using Main=typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    BF,cutlass::layout::RowMajor,8,BF,cutlass::layout::ColumnMajor,8,
    Acc,Tile,Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    std::conditional_t<STREAM,cutlass::gemm::KernelTmaWarpSpecializedCooperative,cutlass::gemm::KernelTmaWarpSpecializedPingpong>>::CollectiveOp;
  using Scheduler=std::conditional_t<STREAM,cutlass::gemm::StreamKScheduler,cutlass::gemm::PersistentScheduler>;
  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,Scheduler>;
  using Device=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static typename Device::Arguments args(const void* w,const void* x,void* y,int m,int n,int k,int mode){
    using SA=typename Kernel::StrideA;using SB=typename Kernel::StrideB;using SC=typename Kernel::StrideC;using SD=typename Kernel::StrideD;
    auto sa=cutlass::make_cute_packed_stride(SA{},make_shape(n,k,1));
    auto sb=cutlass::make_cute_packed_stride(SB{},make_shape(m,k,1));
    auto sc=cutlass::make_cute_packed_stride(SC{},make_shape(n,m,1));
    auto sd=cutlass::make_cute_packed_stride(SD{},make_shape(n,m,1));
    cutlass::KernelHardwareInfo hw{};cudaGetDevice(&hw.device_id);
    hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
    typename Device::Arguments a{cutlass::gemm::GemmUniversalMode::kGemm,{n,m,k,1},
      {(const BF*)w,sa,(const BF*)x,sb},{{1.f,0.f},(const Out*)y,sc,(Out*)y,sd},hw};
    a.scheduler.max_swizzle_size=1;
    if constexpr(STREAM){
      using D=typename Kernel::TileScheduler::DecompositionMode;
      a.scheduler.decomposition_mode=mode==1?D::StreamK:D::Heuristic;
    }
    return a;
  }
  static int run(const void* w,const void* x,void* y,void* workspace,size_t capacity,int m,int n,int k,int mode,cudaStream_t stream,size_t* needed,bool launch){
    auto a=args(w,x,y,m,n,k,mode);
    if(!Kernel::can_implement(a))return 100;
    *needed=Kernel::get_workspace_size(a);if(!launch)return 0;if(*needed>capacity)return -2;
    auto status=Kernel::initialize_workspace(a,workspace,stream);
    if(status!=cutlass::Status::kSuccess)return 200+(int)status;
    auto params=Kernel::to_underlying_arguments(a,workspace);
    auto grid=Kernel::get_grid_shape(params);auto block=Kernel::get_block_shape();
    constexpr int bytes=Kernel::SharedStorageSize;
    auto f=(CUfunction)vd_function;
    CUresult e=cuFuncSetAttribute(f,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,bytes);if(e)return 1000+e;
    CUlaunchConfig config={};config.gridDimX=grid.x;config.gridDimY=grid.y;config.gridDimZ=grid.z;
    config.blockDimX=block.x;config.blockDimY=block.y;config.blockDimZ=block.z;
    config.sharedMemBytes=bytes;config.hStream=(CUstream)stream;
    CUlaunchAttribute attr={};
    if(CX*CY>1){attr.id=CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;attr.value.clusterDim.x=CX;attr.value.clusterDim.y=CY;attr.value.clusterDim.z=1;config.attrs=&attr;config.numAttrs=1;}
    void* args[]={&params};e=cuLaunchKernelEx(&config,f,args,nullptr);return e?1000+e:0;
  }
};
}
extern "C" int vd_cutlass(const void* w,const void* x,void* y,void* workspace,size_t capacity,int m,int n,int k,int code,int fp32,int mode,void* stream,size_t* needed,int launch,void* function){
vd_function=function;
#define CALL(N,K,X,Y,S) if(fp32)return vd::Gemm<N,K,X,Y,S,float>::run(w,x,y,workspace,capacity,m,n,k,mode,(cudaStream_t)stream,needed,launch);else return vd::Gemm<N,K,X,Y,S,vd::BF>::run(w,x,y,workspace,capacity,m,n,k,mode,(cudaStream_t)stream,needed,launch)
  if(code==0){CALL(32,128,1,1,true);}
  if(code==1){CALL(64,128,1,2,false);}
  if(code==2){CALL(64,64,2,1,true);}
  return -1;
#undef CALL
}
