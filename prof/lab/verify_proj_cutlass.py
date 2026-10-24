"""Source-only Hopper CUTLASS split-K/Stream-K projections. No input-value cache."""
import ctypes as _vgct
import hashlib as _vghash
import os as _vgos
import time as _vgtime
import torch

_VGC_MODULE = None
VGC_TILES = (
    (128,128,64,1,1,0,False), (128,128,64,1,2,0,False),
    (128,64,64,1,1,0,False), (128,256,64,1,1,0,False),
    (128,128,128,1,1,0,False), (128,128,64,1,1,3,False),
    (128,128,64,1,1,0,True), (128,64,64,2,1,0,True),
)

class VgcModule:
    def __init__(self, path):
        self.lib = _vgct.CDLL(path)
        p, i, z = _vgct.c_void_p, _vgct.c_int, _vgct.c_size_t
        self.lib.vgc_error.restype = _vgct.c_char_p
        self.lib.vgc_create.argtypes = [i]*11
        self.lib.vgc_create.restype = p
        self.lib.vgc_workspace.argtypes = [p]
        self.lib.vgc_workspace.restype = z
        self.lib.vgc_info.argtypes = [p,i]
        self.lib.vgc_info.restype = i
        self.lib.vgc_destroy.argtypes = [p]
        self.lib.vgc_run.argtypes = [p]*6
        self.lib.vgc_run.restype = i

    def error(self):
        return RuntimeError(self.lib.vgc_error().decode())

    def plan(self, m, n, k, dtype, recipe, device):
        return VgcPlan(self, m, n, k, dtype, recipe, device)

class VgcPlan:
    def __init__(self, module, m, n, k, dtype, recipe, device):
        # recipe: tile, decomposition(0=heuristic,1=DP,2=splitK,3=streamK),
        # splits, active SMs, raster(0=auto,1=M,2=N), swizzle, atomic reduction.
        self.module, self.handle = module, None
        tile, mode, splits, sms, raster, swizzle, atomic = recipe
        if not 0 <= tile < len(VGC_TILES) or dtype not in (torch.bfloat16,torch.float32):
            raise ValueError('invalid CUTLASS tile/dtype')
        if mode not in (0,1,2,3) or splits < 1 or (splits != 1 and mode not in (0,2)):
            raise ValueError('invalid CUTLASS split decomposition')
        if sms < 1 or raster not in (0,1,2) or swizzle not in (1,2,4,8) or atomic not in (0,1):
            raise ValueError('invalid CUTLASS scheduling recipe')
        handle = module.lib.vgc_create(tile, int(dtype==torch.float32),m,n,k,
                                       splits,mode,sms,raster,swizzle,atomic)
        if not handle:
            raise module.error()
        self.handle, self.shape, self.dtype, self.recipe = handle,(m,n,k),dtype,recipe
        try:
            size = module.lib.vgc_workspace(handle)
            self.workspace = torch.empty(max(128,size),device=device,dtype=torch.uint8)
            self.info = tuple(module.lib.vgc_info(handle,i) for i in range(3))
        except Exception:
            self.close()
            raise

    def run(self, x, w, out):
        if self.handle is None:
            raise RuntimeError('closed CUTLASS plan')
        m,n,k = self.shape
        if (x.shape!=(m,k) or w.shape!=(n,k) or out.shape!=(m,n) or
                x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16 or out.dtype!=self.dtype or
                not x.is_contiguous() or not w.is_contiguous() or not out.is_contiguous() or
                x.device!=w.device or x.device!=out.device or x.device!=self.workspace.device):
            raise ValueError('CUTLASS shape/layout mismatch')
        err = self.module.lib.vgc_run(self.handle,x.data_ptr(),w.data_ptr(),out.data_ptr(),
                                     self.workspace.data_ptr(),torch.cuda.current_stream(x.device).cuda_stream)
        if err:
            raise self.module.error()
        return out

    def close(self):
        if self.handle is not None:
            self.module.lib.vgc_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def vgc_module():
    global _VGC_MODULE
    if _VGC_MODULE is not None:
        return _VGC_MODULE
    if torch.cuda.get_device_capability() != (9,0):
        raise RuntimeError('CUTLASS Hopper projection requires sm_90a')
    from torch.utils.cpp_extension import load_inline
    previous = _vgos.environ.get('TORCH_CUDA_ARCH_LIST')
    _vgos.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    try:
        path = load_inline(
            name='muse_verify_cutlass_'+_vghash.sha256(VGC_SOURCE.encode()).hexdigest()[:12],
            cpp_sources='',cuda_sources=VGC_SOURCE,is_python_module=False,no_implicit_headers=True,
            extra_include_paths=['/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include'],
            extra_cuda_cflags=['-O3','-DNDEBUG','-std=c++17','--expt-relaxed-constexpr'],
            extra_ldflags=['-lcuda'],verbose=False)
    finally:
        if previous is None:
            _vgos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
        else:
            _vgos.environ['TORCH_CUDA_ARCH_LIST'] = previous
    _VGC_MODULE = VgcModule(path)
    return _VGC_MODULE


def vgc_choices(m,n,k,sms=132):
    # Round-robin order ensures every tile, cluster and operand family is reached
    # before tuning extra raster, grid-size and reduction settings.
    for mode,sk in ((3,1),(2,2),(2,4),(2,8),(1,1),(0,1)):
        for tile in range(len(VGC_TILES)):
            yield (tile,mode,sk,sms,0,1,0)
    for tile in range(len(VGC_TILES)):
        for mode,sk in ((3,1),(2,4)):
            for raster in (1,2):
                yield (tile,mode,sk,sms,raster,1,0)
            for use_sms in (max(8,sms//2//2*2),max(8,(sms//4*3)//2*2)):
                yield (tile,mode,sk,use_sms,0,1,0)
            yield (tile,mode,sk,sms,0,1,1)
        yield (tile,2,16,sms,0,1,0)
        yield (tile,3,1,sms,2,4,0)

VGC_SOURCE = (
    '// Hopper CUTLASS 3.x collective API, shipped in the judge image.\n// __global__ entry point: cut'
    'lass::device_kernel<Kernel> below.\n#include <cuda_runtime.h>\n#include <cutlass/cutlass.h>\n#inclu'
    'de <cutlass/bfloat16.h>\n#include <cutlass/gemm/collective/collective_builder.hpp>\n#include <cutl'
    'ass/epilogue/collective/collective_builder.hpp>\n#include <cutlass/gemm/device/gemm_universal_ada'
    'pter.h>\n#include <cutlass/gemm/kernel/gemm_universal.hpp>\n#include <cutlass/gemm/kernel/tile_sch'
    'eduler.hpp>\n#include <cute/tensor.hpp>\n#include <cstdint>\n#include <exception>\n#include <string>'
    '\n#include <stdexcept>\n#include <type_traits>\nusing namespace cute;\nnamespace vgcut {\nusing BF=cu'
    'tlass::bfloat16_t;\nusing Row=cutlass::layout::RowMajor;\nusing Col=cutlass::layout::ColumnMajor;\n'
    'using Mode=cutlass::gemm::kernel::detail::DecompositionMode;\nusing Reduction=cutlass::gemm::kern'
    'el::detail::ReductionMode;\nusing Raster=cutlass::gemm::kernel::detail::RasterOrderOptions;\nstati'
    'c thread_local std::string error;\ninline void check(cutlass::Status s) {\n  if (s!=cutlass::Statu'
    's::kSuccess) throw std::runtime_error(cutlassGetStatusString(s));\n}\nstruct Base {\n  size_t bytes'
    '=0; int stages=0,threads=0,smem=0,gx=0,gy=0,gz=0;\n  virtual void run(void*,const void*,void*,voi'
    'd*,cudaStream_t)=0;\n  virtual ~Base()=default;\n};\ntemplate<int BN,int BK,int CM,int CN,int NS,bo'
    'ol Swap,bool FP32>\nstruct Plan:Base {\n  using D=std::conditional_t<FP32,float,BF>;\n  using LD=st'
    'd::conditional_t<Swap,Col,Row>;\n  using Tile=Shape<_128,Int<BN>,Int<BK>>;\n  using Cluster=Shape<'
    'Int<CM>,Int<CN>,_1>;\n  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<\n    '
    'cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,\n    cutlass::epilogue::collecti'
    've::EpilogueTileAuto,float,float,\n    D,LD,128/cutlass::sizeof_bits<D>::value,D,LD,128/cutlass::'
    'sizeof_bits<D>::value,\n    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;\n  us'
    'ing Stages=std::conditional_t<NS==0,\n    cutlass::gemm::collective::StageCountAutoCarveout<sizeo'
    'f(typename Epi::SharedStorage)>,\n    cutlass::gemm::collective::StageCount<NS>>;\n  using Main=ty'
    'pename cutlass::gemm::collective::CollectiveBuilder<\n    cutlass::arch::Sm90,cutlass::arch::OpCl'
    'assTensorOp,\n    BF,Row,8,BF,Col,8,float,Tile,Cluster,Stages,\n    cutlass::gemm::KernelTmaWarpSp'
    'ecializedCooperative>::CollectiveOp;\n  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<i'
    'nt,int,int,int>,Main,Epi,cutlass::gemm::StreamKScheduler>;\n  using Gemm=cutlass::gemm::device::G'
    'emmUniversalAdapter<Kernel>;\n  typename Gemm::Arguments args{};\n  Gemm gemm;\n  Plan(int m,int n,'
    'int k,int splits,int mode,int sms,int raster,int swizzle,int reduction) {\n    int mm=Swap?n:m,nn'
    '=Swap?m:n;\n    args.mode=cutlass::gemm::GemmUniversalMode::kGemm;\n    args.problem_shape=make_sh'
    'ape(mm,nn,k,1);\n    args.mainloop.dA=make_stride(int64_t(k),_1{},int64_t(0));\n    args.mainloop.'
    'dB=make_stride(int64_t(k),_1{},int64_t(0));\n    if constexpr(Swap) {\n      args.epilogue.dC=make'
    '_stride(_1{},int64_t(n),int64_t(0));\n      args.epilogue.dD=make_stride(_1{},int64_t(n),int64_t('
    '0));\n    } else {\n      args.epilogue.dC=make_stride(int64_t(n),_1{},int64_t(0));\n      args.epi'
    'logue.dD=make_stride(int64_t(n),_1{},int64_t(0));\n    }\n    args.epilogue.thread.alpha=1.f;\n    '
    'args.epilogue.thread.beta=0.f;\n    cudaGetDevice(&args.hw_info.device_id);\n    args.hw_info.sm_c'
    'ount=sms;\n    args.scheduler.splits=splits;\n    args.scheduler.decomposition_mode=static_cast<Mo'
    'de>(mode);\n    args.scheduler.reduction_mode=static_cast<Reduction>(reduction);\n    args.schedul'
    'er.raster_order=static_cast<Raster>(raster);\n    args.scheduler.max_swizzle_size=swizzle;\n    by'
    'tes=Gemm::get_workspace_size(args);\n    stages=Gemm::kStages;threads=Kernel::MaxThreadsPerBlock;'
    'smem=Kernel::SharedStorageSize;\n    cudaError_t err=cudaFuncSetAttribute(cutlass::device_kernel<'
    'Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\n    if(err!=cudaSuccess)throw std::ru'
    'ntime_error(cudaGetErrorString(err));\n  }\n  void run(void* x,const void* w,void* out,void* works'
    'pace,cudaStream_t stream)override {\n    args.mainloop.ptr_A=reinterpret_cast<const BF*>(Swap?w:x'
    ');\n    args.mainloop.ptr_B=reinterpret_cast<const BF*>(Swap?x:w);\n    args.epilogue.ptr_C=nullpt'
    'r;\n    args.epilogue.ptr_D=reinterpret_cast<D*>(out);\n    check(Gemm::can_implement(args));\n    '
    '// Include required reduction-counter reset in EVERY timed invocation / graph replay.\n    check('
    'Kernel::initialize_workspace(args,workspace,stream));\n    check(gemm.update(args,workspace));\n  '
    '  check(gemm.run(stream));\n  }\n};\ntemplate<bool F>Base* create(int id,int m,int n,int k,int sk,i'
    'nt mode,int sms,int raster,int sw,int red) {\n#define V(ID,N,K,CM,CN,S,SWAP) case ID:return new P'
    'lan<N,K,CM,CN,S,SWAP,F>(m,n,k,sk,mode,sms,raster,sw,red)\n  switch(id){\n    V(0,128,64,1,1,0,fals'
    'e);\n    V(1,128,64,1,2,0,false);\n    V(2,64,64,1,1,0,false);\n    V(3,256,64,1,1,0,false);\n    V('
    '4,128,128,1,1,0,false);\n    V(5,128,64,1,1,3,false);\n    V(6,128,64,1,1,0,true);\n    V(7,64,64,2'
    ',1,0,true);\n    default:throw std::runtime_error("unknown CUTLASS tile");\n  }\n#undef V\n}\n}\nexter'
    'n "C" const char* vgc_error(){return vgcut::error.c_str();}\nextern "C" void* vgc_create(int id,i'
    'nt fp32,int m,int n,int k,int sk,int mode,int sms,int raster,int sw,int red){\n  try{return fp32?'
    'vgcut::create<true>(id,m,n,k,sk,mode,sms,raster,sw,red):vgcut::create<false>(id,m,n,k,sk,mode,sm'
    's,raster,sw,red);}\n  catch(const std::exception& e){vgcut::error=e.what();return nullptr;}\n}\next'
    'ern "C" size_t vgc_workspace(void* p){return reinterpret_cast<vgcut::Base*>(p)->bytes;}\nextern "'
    'C" int vgc_info(void* p,int index){auto* q=reinterpret_cast<vgcut::Base*>(p);return index==0?q->'
    'stages:index==1?q->threads:q->smem;}\nextern "C" void vgc_destroy(void* p){delete reinterpret_cas'
    't<vgcut::Base*>(p);}\nextern "C" int vgc_run(void* p,void* x,const void* w,void* y,void* ws,cudaS'
    'tream_t stream){\n  try{reinterpret_cast<vgcut::Base*>(p)->run(x,w,y,ws,stream);return 0;}\n  catc'
    'h(const std::exception& e){vgcut::error=e.what();return -1;}\n}\n'
)

import ctypes as ct
import hashlib,os,time,statistics
import torch
_pg_lib=None


class _PgNative:
    def __init__(self, path):
        self.lib = ct.CDLL(path)
        P, I, S = ct.c_void_p, ct.c_int, ct.c_size_t
        self.lib.lean_error_string.restype = ct.c_char_p
        self.lib.pg_algo_bytes.restype = I
        self.algo_bytes = self.lib.pg_algo_bytes()
        self.lib.all_lt_options.argtypes = [I] * 6 + [S, P, I, ct.POINTER(I), ct.POINTER(I)]
        self.lib.all_lt_run.argtypes = [P] * 5 + [I] * 8 + [S, P]

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.lean_error_string().decode())

    def options(self, m, n, k, nn, fp32, workspace):
        codes = torch.empty((65536, self.algo_bytes), dtype=torch.uint8)
        count, heuristics = ct.c_int(), ct.c_int()
        self.check(self.lib.all_lt_options(n, m, k, n if nn else k, nn, int(fp32),
                    workspace, codes.data_ptr(), len(codes), ct.byref(count), ct.byref(heuristics)))
        return codes[:count.value].clone(), heuristics.value

    def run(self, x, w, out, workspace, codes, index, nn, fp32):
        if not 0 <= index < len(codes):
            raise ValueError('unprepared Lt ordinal')
        self.check(self.lib.all_lt_run(w.data_ptr(), x.data_ptr(), out.data_ptr(),
                    workspace.data_ptr(), codes.data_ptr(), out.shape[1], x.shape[0], x.shape[1],
                    w.stride(0), nn, int(fp32), index, 0, workspace.numel(),
                    torch.cuda.current_stream(x.device).cuda_stream))

def _pg_native(arch=None):
    global _pg_lib
    if _pg_lib is not None and arch is None:
        return _pg_lib
    from torch.utils.cpp_extension import load_inline
    target = arch or ('9.0a' if torch.cuda.get_device_capability()[0] == 9 else '12.0')
    old = os.environ.get('TORCH_CUDA_ARCH_LIST')
    os.environ['TORCH_CUDA_ARCH_LIST'] = target
    try:
        path = load_inline(name='muse_prefill_lt_' + hashlib.sha1(PG_RAW.encode()).hexdigest()[:12] + target.replace('.', '_'),
                           cpp_sources='', cuda_sources=PG_RAW, functions=None,
                           is_python_module=False, no_implicit_headers=True,
                           extra_cuda_cflags=['-O3', '-std=c++17'], extra_ldflags=['-lcublasLt'], verbose=False)
    finally:
        if old is None: os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        else: os.environ['TORCH_CUDA_ARCH_LIST'] = old
    result = _PgNative(path)
    if arch is None: _pg_lib = result
    return result

class _PgGraph:
    active = 0

    def __init__(self, fn):
        # Destroying a graph on another live graph's stream can invalidate
        # cuBLAS workspace pointers. Give every capture its own native stream.
        self.runtime = torch.cuda.cudart()
        handle = ct.c_void_p()
        status = self.runtime.cudaStreamCreate(ct.addressof(handle))
        if int(status): raise RuntimeError('cudaStreamCreate failed')
        self.handle = handle.value
        self.stream = torch.cuda.ExternalStream(self.handle)
        self.graph = torch.cuda.CUDAGraph()
        type(self).active += 1
        try:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream): fn()
            self.stream.synchronize()
            with torch.cuda.graph(self.graph, stream=self.stream): self.out = fn()
            self.graph.replay()
            self.stream.synchronize()
        except Exception:
            self.close()
            raise

    def close(self):
        if self.handle is not None:
            self.graph.reset()
            self.runtime.cudaStreamDestroy(self.handle)
            self.handle = None
            type(self).active -= 1

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

PG_RAW = r'''

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstring>
#include <string>
#include <initializer_list>
static thread_local std::string pg_error;
extern "C" const char* lean_error_string(){return pg_error.c_str();}
static int lean_fail(const char* msg,int code){pg_error=std::string(msg)+": "+std::to_string(code);return code ? code : -1;}
#define LEAN_LT(expr) do {auto s=(expr);if(s!=CUBLAS_STATUS_SUCCESS)return lean_fail(#expr,(int)s);} while(0)
static thread_local cublasLtHandle_t lean_lt_handle=nullptr;
static int lean_lt_init(){if(!lean_lt_handle){LEAN_LT(cublasLtCreate(&lean_lt_handle));}return 0;}
extern "C" int pg_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}
// All returned cuBLASLt heuristics, plus checked explicit split-K configurations.
#include <cstdlib>
struct AllLtStorage {void* rows=nullptr;void* algorithms=nullptr;~AllLtStorage(){std::free(rows);std::free(algorithms);}};
struct AllLtLayout {
  cublasLtMatmulDesc_t op=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  int init(int n,int m,int k,int stride,int nn,int kind) {
    cudaDataType_t in=kind==2 ? CUDA_R_32F : CUDA_R_16BF;
    cudaDataType_t out=kind ? CUDA_R_32F : CUDA_R_16BF;
    LEAN_LT(cublasLtMatmulDescCreate(&op,kind==2 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;
    LEAN_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    LEAN_LT(cublasLtMatrixLayoutCreate(&a,in,nn ? n : k,nn ? k : n,stride));
    LEAN_LT(cublasLtMatrixLayoutCreate(&b,in,k,m,k));
    LEAN_LT(cublasLtMatrixLayoutCreate(&c,out,n,m,n));
    return 0;
  }
  ~AllLtLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
};
extern "C" int all_lt_options(int n,int m,int k,int stride,int nn,int kind,size_t workspace,
                              void* codes,int capacity,int* count,int* heuristic_count) {
  int err=lean_lt_init();if(err)return err;
  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;
  cublasLtMatmulPreference_t pref;LEAN_LT(cublasLtMatmulPreferenceCreate(&pref));
  auto s=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));
  if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt preference",(int)s);}
  int requested=64,got=0;
  AllLtStorage storage;
  storage.rows=std::calloc(capacity,sizeof(cublasLtMatmulHeuristicResult_t));
  storage.algorithms=std::calloc(capacity,sizeof(cublasLtMatmulAlgo_t));
  if(!storage.rows || !storage.algorithms){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt host storage",-1);}
  auto rows=(cublasLtMatmulHeuristicResult_t*)storage.rows;
  auto valid=(cublasLtMatmulAlgo_t*)storage.algorithms;
  int valid_count=0;bool overflow=false;
  do {

    s=cublasLtMatmulAlgoGetHeuristic(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,pref,requested,rows,&got);
    if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristics",(int)s);}
    if(got<requested)break;
    requested*=2;
    if(requested>capacity){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristic capacity",-1);}
  }while(true);
  cublasLtMatmulPreferenceDestroy(pref);

  auto add=[&](const cublasLtMatmulAlgo_t& algo) {
    for(int i=0;i<valid_count;++i)if(std::memcmp(valid+i,&algo,sizeof(algo))==0)return;
    cublasLtMatmulHeuristicResult_t checked={};
    auto status=cublasLtMatmulAlgoCheck(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,&algo,&checked);
    if(status==CUBLAS_STATUS_SUCCESS && checked.state==CUBLAS_STATUS_SUCCESS && checked.workspaceSize<=workspace)
      {if(valid_count==capacity)overflow=true;else valid[valid_count++]=algo;}
  };
  for(int i=0;i<got;++i)if(rows[i].state==CUBLAS_STATUS_SUCCESS && rows[i].workspaceSize<=workspace)add(rows[i].algo);
  *heuristic_count=valid_count;
  // Preserve every returned heuristic; add supported swizzles and FP32 split-K reductions.
  const int base_count=valid_count;
  for(int bi=0;bi<base_count;++bi) {
    const auto base=valid[bi];
    int32_t split_support=0;uint32_t mask=0,swizzle=0;size_t written=0;
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,&split_support,sizeof(split_support),&written));
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,&mask,sizeof(mask),&written));
    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,&swizzle,sizeof(swizzle),&written));
    for(uint32_t sw=0;sw<=swizzle;++sw) {
      auto plain=base;
      if(cublasLtMatmulAlgoConfigSetAttribute(&plain,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,&sw,sizeof(sw))!=CUBLAS_STATUS_SUCCESS)continue;
      add(plain);
      if(split_support && (mask&CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {
        for(uint32_t split:{2u,4u,8u,16u,32u}) {
          auto algo=plain;uint32_t scheme=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS)continue;
          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&scheme,sizeof(scheme))!=CUBLAS_STATUS_SUCCESS)continue;
          add(algo);
        }
      }
    }
  }
  if(overflow)return lean_fail("all Lt option capacity",-1);
  for(int i=0;i<valid_count;++i)std::memcpy((char*)codes+i*sizeof(cublasLtMatmulAlgo_t),&valid[i],sizeof(cublasLtMatmulAlgo_t));
  *count=valid_count;return 0;
}
extern "C" int all_lt_run(const void* w,const void* x,void* out,void* workspace,const void* codes,
                          int n,int m,int k,int stride,int nn,int kind,int index,int accumulate,
                          size_t workspace_bytes,void* stream) {
  int err=lean_lt_init();if(err)return err;
  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;
  cublasLtMatmulAlgo_t algo;std::memcpy(&algo,(const char*)codes+index*sizeof(algo),sizeof(algo));
  float alpha=1.f,beta=accumulate ? 1.f : 0.f;
  auto status=cublasLtMatmul(lean_lt_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&algo,
                            workspace,workspace_bytes,(cudaStream_t)stream);
  return status==CUBLAS_STATUS_SUCCESS ? 0 : lean_fail("all Lt run",(int)status);
}

'''

def vp_pair(old,new,x,banks,flush,repeats=20):
    times=[[],[]];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for r in range(repeats):
        x.copy_(banks[r%len(banks)])
        for side in ((0,1) if r%2==0 else (1,0)):
            flush.sum()
            a.record();(old if side==0 else new).graph.replay();b.record();b.synchronize()
            times[side].append(float(a.elapsed_time(b))*1000)
    ratios=[((times[1][i]/times[0][i])*(times[1][i+1]/times[0][i+1]))**.5 for i in range(0,repeats,2)]
    return statistics.median(times[0]),statistics.median(times[1]),statistics.median(ratios)


def vp_check(out,expected):
    rel=float((out.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12))
    if not bool(torch.isfinite(out).all()) or rel>.01:raise ValueError(f'FP32 self-reference rel={rel:.5g}')
    return rel


import triton
import triton.language as tl

@triton.jit
def _l2_prefetch(ptrs):
    return tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [ptrs],
                                     dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _pdl_start(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _pdl_wait(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_wait()


@triton.jit
def _vgc_gemv_kernel(X, W, P, T, x_stride,
                 N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr, TB: tl.constexpr, PREFETCH: tl.constexpr, USE_PDL: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    _pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    k0 = ps * KS
    if PREFETCH:
        lines = tl.arange(0, KS // 64)
        _l2_prefetch(W + n[:, None] * K + k0 + lines[None, :] * 64)
    _pdl_wait(USE_PDL)
    t = tl.program_id(2) * TB + tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for k in range(k0, k0 + KS, BK):
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)



def vgc_gemv(x,w,dtype):
    m,k=x.shape;n=w.shape[0]
    ks=512 if k==1024 else 2048 if n==4608 or k==4096 else 1024
    p=torch.empty((k//ks,m,n),device=x.device,dtype=torch.float32)
    _vgc_gemv_kernel[(n//64,k//ks,triton.cdiv(m,64))](x,w,p,m,x.stride(0),N=n,K=k,KS=ks,
        BN=64,BK=128,TB=64,PREFETCH=False,USE_PDL=False,
        num_warps=4,num_stages=4)
    return p.sum(0).to(dtype)

# Every optional failure is diagnostic; the scored host is unchanged.
VGC_LT_RECIPES = {(128,'qkv_gate'):(1,23),(128,'pool_router'):(0,0),
                  (128,'unpooler'):(1,0),(128,'o'):(1,0),
                  (256,'qkv_gate'):(0,0),(256,'pool_router'):(0,0),(256,'unpooler'):(0,0)}

def lab(deadline,weights,state=None):
    begin=time.monotonic();deadline=min(deadline,begin+225)
    rows=[]
    try:module=vgc_module()
    except Exception as exc:return [f'VGC_REJECT module {type(exc).__name__}:{str(exc)[-300:]}']
    compile_s=time.monotonic()-begin
    try:lt=_pg_native()
    except Exception as exc:
        lt=None;rows.append(f'VGC_REJECT lt {type(exc).__name__}:{str(exc)[:120]}')
    device=weights['qkv_gate'].device
    sms=torch.cuda.get_device_properties(device).multi_processor_count
    router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
    ws={name:weights[name] for name in ('qkv_gate','o','unpooler')}
    ws['pool_router']=torch.cat((weights['pooler'],hi,lo),0).contiguous()
    workspace=torch.empty(4<<30,device=device,dtype=torch.uint8) if lt is not None else None
    flush=torch.zeros((1<<30)//4,device=device,dtype=torch.float32)
    generator=torch.Generator(device=device).manual_seed(8374021)
    rows.append(f'VGC_META sm={sms} compile_s={compile_s:.2f} cold1GiB paired20 fresh3; FP32 pool+router singleGEMM; counter reset timed')
    keys=[(m,name) for m in (256,128) for name in ('qkv_gate','pool_router','unpooler','o')]
    for pos,(m,name) in enumerate(keys):
        now=time.monotonic();stop=now+max(0,deadline-now-3)/(len(keys)-pos)
        if now>=deadline-4:rows.append(f'VGC_BUDGET {m} {name}');continue
        old=None;torch_graph=None;lt_graph=None
        try:
            w=ws[name];n,k=w.shape;dtype=torch.float32 if name=='pool_router' else torch.bfloat16
            banks=[torch.randn(m,k,device=device,dtype=torch.bfloat16,generator=generator) for _ in range(2)]
            probe=torch.zeros_like(banks[0]);probe[:,127]=.5;banks.append(probe)
            x=banks[0].clone();refs=[b.float()@w.float().T for b in banks]
            torch_graph=_PgGraph(lambda:torch.mm(x,w.T,out_dtype=torch.float32) if dtype==torch.float32 else torch.mm(x,w.T))
            for b,ref in zip(banks,refs):x.copy_(b);torch_graph.graph.replay();vp_check(torch_graph.out,ref)
            old=torch_graph;control='cublas';lib_row='none'
            if lt is not None and (m,name) in VGC_LT_RECIPES:
                try:
                    nn,ordinal=VGC_LT_RECIPES[m,name]
                    weight=w.T.contiguous() if nn else w
                    codes,_=lt.options(m,n,k,nn,dtype==torch.float32,workspace.numel())
                    if ordinal>=len(codes):raise ValueError('H200 Lt ordinal unavailable')
                    out=torch.empty((m,n),device=device,dtype=dtype)
                    def run_lt():
                        lt.run(x,weight,out,workspace,codes,ordinal,nn,dtype==torch.float32)
                        return out
                    lt_graph=_PgGraph(run_lt)
                    for b,ref in zip(banks,refs):
                        x.copy_(b);lt_graph.graph.replay();vp_check(lt_graph.out,ref)
                    t0,t1,r=vp_pair(torch_graph,lt_graph,x,banks,flush,20)
                    lib_row=f'{t0:.2f}/{t1:.2f}/{r:.4f}'
                    if r<1:old=lt_graph;control=f'lt{nn}.{ordinal}'
                except Exception as exc:
                    rows.append(f'VGC_REJECT lt {m} {name} {type(exc).__name__}:{str(exc)[:90]}')
            gemv='rejected'
            try:
                with _PgGraph(lambda:vgc_gemv(x,w,dtype)) as graph:
                    for b,ref in zip(banks,refs):x.copy_(b);graph.graph.replay();vp_check(graph.out,ref)
                    t0,t1,ratio=vp_pair(old,graph,x,banks,flush,20)
                    gemv=f'{t0:.2f}/{t1:.2f}/{ratio:.4f}'
            except Exception as exc:
                rows.append(f'VGC_REJECT gemv {m} {name} {type(exc).__name__}:{str(exc)[:90]}')
            candidates=[];bad=0;reasons=set();families={};attempts=0
            coarse_stop=now+.70*max(0,stop-now)
            for recipe in vgc_choices(m,n,k,sms):
                if time.monotonic()>coarse_stop:break
                plan=None;graph=None;attempts+=1
                try:
                    plan=module.plan(m,n,k,dtype,recipe,device)
                    out=torch.empty((m,n),device=device,dtype=dtype)
                    graph=_PgGraph(lambda:plan.run(x,w,out))
                    error=0.
                    for b,ref in zip(banks,refs):
                        x.copy_(b);old.graph.replay();graph.graph.replay()
                        error=max(error,vp_check(graph.out,ref),vp_check(graph.out,old.out))
                    t0,t1,ratio=vp_pair(old,graph,x,banks,flush,4)
                    candidates.append((ratio,recipe,error,plan.info,plan.workspace.numel()))
                    # Record best tile/layout and each scheduler family, including losses.
                    family=(recipe[0],recipe[1])
                    if family not in families or ratio<families[family][0]:families[family]=(ratio,t1)
                except Exception as exc:
                    bad+=1;reason=f'{type(exc).__name__}:{str(exc).splitlines()[0][:100]}'
                    if reason not in reasons and len(reasons)<2:rows.append(f'VGC_REJECT {m} {name} c={recipe} {reason}')
                    reasons.add(reason)
                finally:
                    if graph is not None:graph.close()
                    if plan is not None:plan.close()
            finals=sorted(candidates)[:3]
            for mode in (1,2,3):
                family=[r for r in candidates if r[1][1]==mode]
                if family and min(family) not in finals:finals.append(min(family))
            best=(1.,None,None);final_rows=[]
            for _,recipe,error,info,size in finals:
                if time.monotonic()>min(stop,deadline-2):break
                plan=None
                try:
                    plan=module.plan(m,n,k,dtype,recipe,device)
                    out=torch.empty((m,n),device=device,dtype=dtype)
                    with _PgGraph(lambda:plan.run(x,w,out)) as graph:
                        t0,t1,ratio=vp_pair(old,graph,x,banks,flush,20)
                        for b,ref in zip(banks,refs):
                            x.copy_(b);old.graph.replay();graph.graph.replay()
                            vp_check(graph.out,ref);vp_check(graph.out,old.out)
                        final_rows.append((recipe,round(t0,3),round(t1,3),round(ratio,5)))
                        if ratio<best[0]:best=(ratio,recipe,(t0,t1))
                except Exception as exc:
                    bad+=1;rows.append(f'VGC_REJECT final {m} {name} {type(exc).__name__}:{str(exc)[:90]}')
                finally:
                    if plan is not None:plan.close()
            rows.append(f'VGC_CONTROLS {m} {name} selected={control} torch/Lt/r={lib_row} control/GEMV/r={gemv}')
            # Compact losses too: one best result per tile and per decomposition.
            tile_best={a:min(v for (t,_),v in families.items() if t==a) for a in range(8)
                       if any(t==a for t,_ in families)}
            mode_best={b:min(v for (_,d),v in families.items() if d==b) for b in range(4)
                       if any(d==b for _,d in families)}
            fam=' '.join(f't{a}:{v[1]:.1f}/{v[0]:.3f}' for a,v in tile_best.items())
            fam+=' '+' '.join(f'd{b}:{v[1]:.1f}/{v[0]:.3f}' for b,v in mode_best.items())
            rows.append(f'VGC_FAMILIES {m} {name} {fam}')
            rows.append(f'VGC_BEST {m} {name} pick={best[1]} us={best[2]} ratio={best[0]:.5f} tried={attempts} rejected={bad} finals={final_rows}')
        except Exception as exc:
            rows.append(f'VGC_REJECT shape {m} {name} {type(exc).__name__}:{str(exc)[:140]}')
        finally:
            if lt_graph is not None:lt_graph.close()
            if torch_graph is not None:torch_graph.close()
    rows.append(f'VGC_END elapsed={time.monotonic()-begin:.2f} left={deadline-time.monotonic():.2f}')
    return rows
