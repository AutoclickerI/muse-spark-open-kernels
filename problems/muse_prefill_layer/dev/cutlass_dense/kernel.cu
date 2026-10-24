// Hopper CUTLASS 3.x collective API, shipped in the judge image.
// __global__ entry point: cutlass::device_kernel<Kernel> below.
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cute/tensor.hpp>
#include <cstdint>
#include <exception>
#include <string>
#include <stdexcept>
#include <type_traits>
using namespace cute;
namespace vgcut {
using BF=cutlass::bfloat16_t;
using Row=cutlass::layout::RowMajor;
using Col=cutlass::layout::ColumnMajor;
using Mode=cutlass::gemm::kernel::detail::DecompositionMode;
using Reduction=cutlass::gemm::kernel::detail::ReductionMode;
using Raster=cutlass::gemm::kernel::detail::RasterOrderOptions;
static thread_local std::string error;
inline void check(cutlass::Status s) {
  if (s!=cutlass::Status::kSuccess) throw std::runtime_error(cutlassGetStatusString(s));
}
struct Base {
  size_t bytes=0; int stages=0,threads=0,smem=0,gx=0,gy=0,gz=0;
  virtual void run(void*,const void*,void*,void*,cudaStream_t)=0;
  virtual ~Base()=default;
};
template<int BN,int BK,int CM,int CN,int NS,bool Swap,bool FP32>
struct Plan:Base {
  using D=std::conditional_t<FP32,float,BF>;
  using LD=std::conditional_t<Swap,Col,Row>;
  using Tile=Shape<_128,Int<BN>,Int<BK>>;
  using Cluster=Shape<Int<CM>,Int<CN>,_1>;
  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,float,float,
    D,LD,128/cutlass::sizeof_bits<D>::value,D,LD,128/cutlass::sizeof_bits<D>::value,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  using Stages=std::conditional_t<NS==0,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
    cutlass::gemm::collective::StageCount<NS>>;
  using Main=typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    BF,Row,8,BF,Col,8,float,Tile,Cluster,Stages,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,cutlass::gemm::StreamKScheduler>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  typename Gemm::Arguments args{};
  Gemm gemm;
  Plan(int m,int n,int k,int splits,int mode,int sms,int raster,int swizzle,int reduction) {
    int mm=Swap?n:m,nn=Swap?m:n;
    args.mode=cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape=make_shape(mm,nn,k,1);
    args.mainloop.dA=make_stride(int64_t(k),_1{},int64_t(0));
    args.mainloop.dB=make_stride(int64_t(k),_1{},int64_t(0));
    if constexpr(Swap) {
      args.epilogue.dC=make_stride(_1{},int64_t(n),int64_t(0));
      args.epilogue.dD=make_stride(_1{},int64_t(n),int64_t(0));
    } else {
      args.epilogue.dC=make_stride(int64_t(n),_1{},int64_t(0));
      args.epilogue.dD=make_stride(int64_t(n),_1{},int64_t(0));
    }
    args.epilogue.thread.alpha=1.f;
    args.epilogue.thread.beta=0.f;
    cudaGetDevice(&args.hw_info.device_id);
    args.hw_info.sm_count=sms;
    args.scheduler.splits=splits;
    args.scheduler.decomposition_mode=static_cast<Mode>(mode);
    args.scheduler.reduction_mode=static_cast<Reduction>(reduction);
    args.scheduler.raster_order=static_cast<Raster>(raster);
    args.scheduler.max_swizzle_size=swizzle;
    bytes=Gemm::get_workspace_size(args);
    stages=Gemm::kStages;threads=Kernel::MaxThreadsPerBlock;smem=Kernel::SharedStorageSize;
    cudaError_t err=cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
    if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));
  }
  void run(void* x,const void* w,void* out,void* workspace,cudaStream_t stream)override {
    args.mainloop.ptr_A=reinterpret_cast<const BF*>(Swap?w:x);
    args.mainloop.ptr_B=reinterpret_cast<const BF*>(Swap?x:w);
    args.epilogue.ptr_C=nullptr;
    args.epilogue.ptr_D=reinterpret_cast<D*>(out);
    check(Gemm::can_implement(args));
    // Include required reduction-counter reset in EVERY timed invocation / graph replay.
    check(Kernel::initialize_workspace(args,workspace,stream));
    check(gemm.update(args,workspace));
    check(gemm.run(stream));
  }
};
template<bool F>Base* create(int id,int m,int n,int k,int sk,int mode,int sms,int raster,int sw,int red) {
#define V(ID,N,K,CM,CN,S,SWAP) case ID:return new Plan<N,K,CM,CN,S,SWAP,F>(m,n,k,sk,mode,sms,raster,sw,red)
  switch(id){
    V(0,128,64,1,1,0,false);
    V(1,128,64,1,2,0,false);
    V(2,64,64,1,1,0,false);
    V(3,256,64,1,1,0,false);
    V(4,128,128,1,1,0,false);
    V(5,128,64,1,1,3,false);
    V(6,128,64,1,1,0,true);
    V(7,64,64,2,1,0,true);
    default:throw std::runtime_error("unknown CUTLASS tile");
  }
#undef V
}
}
extern "C" const char* vgc_error(){return vgcut::error.c_str();}
extern "C" void* vgc_create(int id,int fp32,int m,int n,int k,int sk,int mode,int sms,int raster,int sw,int red){
  try{return fp32?vgcut::create<true>(id,m,n,k,sk,mode,sms,raster,sw,red):vgcut::create<false>(id,m,n,k,sk,mode,sms,raster,sw,red);}
  catch(const std::exception& e){vgcut::error=e.what();return nullptr;}
}
extern "C" size_t vgc_workspace(void* p){return reinterpret_cast<vgcut::Base*>(p)->bytes;}
extern "C" int vgc_info(void* p,int index){auto* q=reinterpret_cast<vgcut::Base*>(p);return index==0?q->stages:index==1?q->threads:q->smem;}
extern "C" void vgc_destroy(void* p){delete reinterpret_cast<vgcut::Base*>(p);}
extern "C" int vgc_run(void* p,void* x,const void* w,void* y,void* ws,cudaStream_t stream){
  try{reinterpret_cast<vgcut::Base*>(p)->run(x,w,y,ws,stream);return 0;}
  catch(const std::exception& e){vgcut::error=e.what();return -1;}
}
