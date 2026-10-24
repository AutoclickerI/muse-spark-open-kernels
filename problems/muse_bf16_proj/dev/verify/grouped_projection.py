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

VG_SOURCE='#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <cooperative_groups.h>\n#include <stdint.h>\nnamespace cg = cooperative_groups;\n\n#include <cuda.h>\n#include <cudaTypedefs.h>\n#ifndef MP_SPIN_LIMIT\n#define MP_SPIN_LIMIT (1u << 26)\n#endif\n#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define MP_SM90A 1\n#endif\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n#define MP_TMA 1  // TMA, mbarriers, clusters: sm_90 and later (wgmma itself needs sm_90a)\n#endif\n\nnamespace muse_wg {\n__device__ __forceinline__ void expect_only(uint64_t* bar, uint32_t bytes) {\n  const uint32_t address = (uint32_t)__cvta_generic_to_shared(bar);\n  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"\n      :: "r"(address), "r"(bytes) : "memory");\n}\n\n\n__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n\n#ifdef MP_TMA\n__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(bar)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su32(bar)) : "memory");\n}\n// Bounded: a pipeline bug yields wrong numbers (caught by tests) instead of a hung GPU.\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  uint32_t done = 0;\n  uint32_t tries = 0;\n  do {\n    if (false) {\n#ifdef MP_DEBUG\n      printf("mbar_wait timeout block (%d,%d,%d) thread %d parity %u\\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, parity);\n#endif\n      break;\n    }\n    asm volatile(\n        "{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n        : "=r"(done) : "r"(su32(bar)), "r"(parity) : "memory");\n  } while (!done);\n}\n// Same box into this CTA and every CTA in mask (same smem offsets, each one\'s own mbarrier).\n__device__ __forceinline__ void tma_load_2d_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,\n                                               uint16_t mask) {\n  asm volatile(\n      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"\n      " [%0], [%1, {%3, %4}], [%2], %5;"\n      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(su32(bar)), "r"(c0), "r"(c1), "h"(mask)\n      : "memory");\n}\n__device__ __forceinline__ void mbar_arrive_remote(uint64_t* bar, uint32_t cta_rank) {\n  asm volatile(\n      "{\\n .reg .b32 ra;\\n mapa.shared::cluster.u32 ra, %0, %1;\\n"\n      " mbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\\n}\\n"\n      ::"r"(su32(bar)), "r"(cta_rank) : "memory");\n}\n__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {\n  asm volatile(\n      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"\n      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(su32(bar)), "r"(c0), "r"(c1) : "memory");\n}\ntemplate <int N_>\n__device__ __forceinline__ void acc_fence(float (&d)[N_]) {\n#pragma unroll\n  for (int i = 0; i < N_; ++i) asm volatile("" : "+f"(d[i])::"memory");\n}\n#endif\n#ifdef MP_SM90A\n__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\n__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\ntemplate <int N_>\n__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N_) : "memory"); }\n// K-major operand, 128-byte swizzle, 8-row atoms 1024 bytes apart.\n__device__ __forceinline__ uint64_t make_desc(const void* p) {\n  const uint64_t a = su32(p);\n  return ((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\n}\n__device__ __forceinline__ void wgmma_m64n8(float (&d)[4], uint64_t da, uint64_t db, int scale_d) {\n  asm volatile("{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %6, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n      : "l"(da), "l"(db), "r"(scale_d));\n}\n__device__ __forceinline__ void wgmma_m64n16(float (&d)[8], uint64_t da, uint64_t db, int scale_d) {\n  asm volatile("{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %10, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, 0, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])\n      : "l"(da), "l"(db), "r"(scale_d));\n}\n__device__ __forceinline__ void wgmma_m64n32(float (&d)[16], uint64_t da, uint64_t db, int scale_d) {\n  asm volatile("{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %18, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, p, 1, 1, 0, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])\n      : "l"(da), "l"(db), "r"(scale_d));\n}\n__device__ __forceinline__ void wgmma_m64n64(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {\n  asm volatile("{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %34, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n      : "l"(da), "l"(db), "r"(scale_d));\n}\n__device__ __forceinline__ void wgmma_m64n128(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {\n  asm volatile("{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %66, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n      : "l"(da), "l"(db), "r"(scale_d));\n}\n__device__ __forceinline__ void wgmma_m64n256(float (&d)[128], uint64_t da, uint64_t db, int scale_d) {\n  asm volatile("{\\n"\n      ".reg .pred p;\\n"\n      "setp.ne.b32 p, %130, 0;\\n"\n      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0;\\n"\n      "}\\n"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])\n      : "l"(da), "l"(db), "r"(scale_d));\n}\n\n#elif defined(MP_TMA)\n// Off sm_90a: the same swizzled tiles through ldmatrix + mma.sync (validates the pipeline).\n__device__ __forceinline__ void ldsm_x4(const void* p, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {\n  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"\n               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(su32(p)));\n}\n__device__ __forceinline__ void ldsm_x2(const void* p, uint32_t& b0, uint32_t& b1) {\n  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(su32(p)));\n}\n__device__ __forceinline__ void mma_emu(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                        uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n#endif\n\n// out[M, N] = X[M, K] @ W[N, K]^T on Hopper: TMA tiles (128-byte swizzle) of BR = 128*NSUB weight\n// rows and BT tokens per 64-wide K step, ST stages; warpgroups 0-1 run wgmma (weights on its M\n// side, tokens on N), warpgroup 2\'s first thread issues the TMA loads. K is split over the\n// cluster\'s CTAs (blockIdx.y) and reduced through DSMEM, 64 tokens per pass.\n// MC = 2: the CTA pair (blockIdx.x, blockIdx.x ^ 1) in a cluster shares one activation tile, each\n// loading half of it and multicasting to both; a stage is refilled only after both pairs\' consumers\n// release it.\ntemplate <int BT, int NSUB, int NCW, int ST, int MC, bool OUTF32>\n__global__ void __launch_bounds__(128 * NCW + 32) wg_kernel(const __grid_constant__ CUtensorMap tmW,\n                                                    const __grid_constant__ CUtensorMap tmX,\n                                                    void* __restrict__ Out, int M, int N, int nkb, int kx, int pipe, int pdl, int kgroup) {\n#ifdef MP_TMA\n  constexpr int BR = 64 * NSUB * NCW;\n  constexpr int A_BYTES = BR * 64 * 2;\n  constexpr int B_BYTES = BT * 64 * 2;\n  constexpr int SLICE_BYTES = A_BYTES + B_BYTES;\n  const int STAGE_BYTES = SLICE_BYTES * kgroup;\n  constexpr int NACC = BT / 2;\n  extern __shared__ unsigned char smem_raw[];\n  unsigned char* smem = (unsigned char*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);\n  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * STAGE_BYTES);\n  uint64_t* empty = full + ST;\n  const int wg = threadIdx.x >> 7;\n\n  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  const int groups_n = (N / BR) / MC;\n  const int jobs = groups_n * ((M + BT - 1) / BT);\n  for (int job = blockIdx.x / MC; job < jobs; job += gridDim.x / MC) {\n  const int row_cta = (job % groups_n) * MC * BR + (blockIdx.x % MC) * BR;\n  const int tok0 = (job / groups_n) * BT;\n  const int ks = gridDim.y;\n  const int kb0 = (int)((long long)blockIdx.y * nkb / ks);\n  const int kb1 = (int)((long long)(blockIdx.y + 1) * nkb / ks);\n  const int nsteps = (kb1 - kb0) / kgroup;\n  const bool clustered = MC > 1 || ks > 1;\n  const int cx = (int)(blockIdx.x % MC);         // position in the multicast pair\n  const int my_rank = cx + MC * (int)blockIdx.y;  // rank in the (MC, ks) cluster\n  if (threadIdx.x == 0) {\n    for (int s = 0; s < ST; ++s) {\n      mbar_init(&full[s], 1); expect_only(&full[s], STAGE_BYTES);\n      mbar_init(&empty[s], NCW * MC);\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n  }\n  if (MC > 1)\n    cg::this_cluster().sync();  // partners\' barriers must exist before multicast or remote arrive\n  else\n    __syncthreads();\n\n  float acc[NSUB][NACC];\n#pragma unroll\n  for (int s = 0; s < NSUB; ++s)\n#pragma unroll\n    for (int i = 0; i < NACC; ++i) acc[s][i] = 0.f;\n\n  if (wg == NCW) {\n    if (threadIdx.x == 128 * NCW) {\n      for (int s = 0; s < nsteps; ++s) {\n        const int stage = s % ST;\n        const uint32_t ph = (s / ST) & 1;\n        if (s >= ST) mbar_wait(&empty[stage], ph ^ 1);\n        mbar_arrive(&full[stage]);\n        for (int kg = 0; kg < kgroup; ++kg) {\n        unsigned char* sa = smem + stage * STAGE_BYTES + kg * SLICE_BYTES;\n        const int kc = (kb0 + s * kgroup + kg) * 64;\n        tma_load_2d(sa, &tmW, &full[stage], kc, row_cta);\n        if (s == 0 && pdl) asm volatile("griddepcontrol.wait;" ::: "memory");\n        if (MC == 1) {\n          tma_load_2d(sa + A_BYTES, &tmX, &full[stage], kc % kx, tok0);\n        } else {\n#ifdef MP_SM90A\n          const uint16_t mask = (uint16_t)(((1u << MC) - 1) << (MC * blockIdx.y));\n          tma_load_2d_mc(sa + A_BYTES + cx * (BT / MC) * 128, &tmX, &full[stage], kc % kx, tok0 + cx * (BT / MC),\n                         mask);\n#else\n          // Portable transport stand-in: native multicast is optimized for sm90a.\n          // Keep identical barriers, cluster scheduling and arithmetic locally.\n          for (int part = 0; part < MC; ++part)\n            tma_load_2d(sa + A_BYTES + part * (BT / MC) * 128, &tmX, &full[stage], kc % kx,\n                        tok0 + part * (BT / MC));\n#endif\n        }\n        }  // K transaction group\n      }\n    }\n  } else {\n    for (int s = 0; s < nsteps; ++s) {\n      const int stage = s % ST;\n      const uint32_t ph = (s / ST) & 1;\n      mbar_wait(&full[stage], ph);\n      unsigned char* stage_base = smem + stage * STAGE_BYTES;\n#ifdef MP_SM90A\n#pragma unroll\n      for (int sub = 0; sub < NSUB; ++sub) acc_fence(acc[sub]);\n      wgmma_fence();\n      for (int kg = 0; kg < kgroup; ++kg) {\n        unsigned char* sa = stage_base + kg * SLICE_BYTES;\n      const uint64_t db = make_desc(sa + A_BYTES);\n\n#pragma unroll\n      for (int kk = 0; kk < 4; ++kk)\n#pragma unroll\n        for (int sub = 0; sub < NSUB; ++sub) {\n          const uint64_t da = make_desc(sa + (wg * NSUB + sub) * 64 * 128);\n          if constexpr (BT == 8) wgmma_m64n8(acc[sub], da + 2 * kk, db + 2 * kk, 1);\n          else if constexpr (BT == 16) wgmma_m64n16(acc[sub], da + 2 * kk, db + 2 * kk, 1);\n          else if constexpr (BT == 32) wgmma_m64n32(acc[sub], da + 2 * kk, db + 2 * kk, 1);\n          else if constexpr (BT == 64) wgmma_m64n64(acc[sub], da + 2 * kk, db + 2 * kk, 1);\n          else if constexpr (BT == 128) wgmma_m64n128(acc[sub], da + 2 * kk, db + 2 * kk, 1);\n          else if constexpr (BT == 256) wgmma_m64n256(acc[sub], da + 2 * kk, db + 2 * kk, 1);\n        }\n\n      }\n      wgmma_commit();\n      if (pipe) wgmma_wait<1>(); else wgmma_wait<0>();\n#pragma unroll\n      for (int sub = 0; sub < NSUB; ++sub) acc_fence(acc[sub]);\n#else\n      for (int kg = 0; kg < kgroup; ++kg) {\n        unsigned char* sa = stage_base + kg * SLICE_BYTES;\n\n      {\n        const int wq = (threadIdx.x >> 5) & 3, ln = threadIdx.x & 31;\n#pragma unroll\n        for (int kk = 0; kk < 4; ++kk)\n#pragma unroll\n          for (int sub = 0; sub < NSUB; ++sub) {\n            const int arow = (wg * NSUB + sub) * 64 + wq * 16 + (ln & 15);\n            const int achunk = kk * 2 + (ln >> 4);\n            uint32_t a0, a1, a2, a3;\n            ldsm_x4(sa + arow * 128 + ((achunk ^ (arow & 7)) << 4), a0, a1, a2, a3);\n#pragma unroll\n            for (int j = 0; j < BT / 8; ++j) {\n              const int brow = j * 8 + (ln & 7);\n              const int bchunk = kk * 2 + ((ln >> 3) & 1);\n              uint32_t b0, b1;\n              ldsm_x2(sa + A_BYTES + brow * 128 + ((bchunk ^ (brow & 7)) << 4), b0, b1);\n              mma_emu(&acc[sub][4 * j], a0, a1, a2, a3, b0, b1);\n            }\n          }\n        // mma.sync warps finish independently: sync the warpgroup before releasing the stage.\n        asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory");\n      }\n\n      }\n#endif\n      if ((threadIdx.x & 127) == 0 && (!pipe || s > 0)) {\n        const int release_stage = pipe ? (s + ST - 1) % ST : stage;\n        expect_only(&full[release_stage], STAGE_BYTES / NCW);\n        mbar_arrive(&empty[release_stage]);\n        if (MC > 1) mbar_arrive_remote(&empty[release_stage], (uint32_t)(my_rank ^ 1));\n      }\n    }\n#ifdef MP_SM90A\n    wgmma_wait<0>();\n#endif\n    if (pipe && nsteps > 0 && (threadIdx.x & 127) == 0) {\n      const int last = (nsteps - 1) % ST;\n      expect_only(&full[last], STAGE_BYTES / NCW);\n      mbar_arrive(&empty[last]);\n      if (MC > 1) mbar_arrive_remote(&empty[last], (uint32_t)(my_rank ^ 1));\n    }\n  }\n  __syncthreads();  // pipeline buffers are free: reuse them as the reduction tile\n\n  constexpr int S = BR + 4;\n  constexpr int PT = BT < 64 ? BT : 64;  // tokens per reduction pass\n  float* sred = reinterpret_cast<float*>(smem);\n  const int warp_in_wg = (threadIdx.x >> 5) & 3;\n  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;\n  auto store = [&](int tok, int row, float v) {\n    const size_t idx = (size_t)(tok0 + tok) * N + row_cta + row;\n    if (OUTF32)\n      reinterpret_cast<float*>(Out)[idx] = v;\n    else\n      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);\n  };\n#pragma unroll\n  for (int pass = 0; pass < BT / PT; ++pass) {\n    if (wg < NCW) {\n#pragma unroll\n      for (int sub = 0; sub < NSUB; ++sub) {\n        const int row = (wg * NSUB + sub) * 64 + warp_in_wg * 16 + g;\n#pragma unroll\n        for (int ii = 0; ii < PT / 8; ++ii) {\n          const int i = pass * (PT / 8) + ii;\n          float* p = sred + (size_t)(ii * 8 + 2 * t) * S + row;\n          p[0] = acc[sub][4 * i + 0];\n          p[S] = acc[sub][4 * i + 1];\n          p[8] = acc[sub][4 * i + 2];\n          p[S + 8] = acc[sub][4 * i + 3];\n        }\n      }\n    }\n    const int ptok = pass * PT;\n    const int E = PT * BR;\n    if (ks == 1) {\n      __syncthreads();\n      for (int e = threadIdx.x; e < E; e += blockDim.x) {\n        const int tok = e / BR, row = e - tok * BR;\n        if (tok0 + ptok + tok < M) store(ptok + tok, row, sred[tok * S + row]);\n      }\n      __syncthreads();\n    } else {\n      cg::cluster_group cluster = cg::this_cluster();\n      cluster.sync();\n      const int slice = (int)blockIdx.y;\n      const int per = (E + ks - 1) / ks;\n      const int e0 = slice * per;\n      const int e1 = min(E, e0 + per);\n      for (int e = e0 + (int)threadIdx.x; e < e1; e += blockDim.x) {\n        const int tok = e / BR, row = e - tok * BR;\n        if (tok0 + ptok + tok >= M) continue;\n        float v = 0.f;\n        for (int q = 0; q < ks; ++q) v += cluster.map_shared_rank(sred, cx + MC * q)[tok * S + row];\n        store(ptok + tok, row, v);\n      }\n      cluster.sync();\n    }\n  }\n  if (MC > 1 && ks == 1) cg::this_cluster().sync();\n  if (threadIdx.x == 0) {\n#pragma unroll\n    for (int s = 0; s < ST; ++s) {\n      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(su32(&full[s])) : "memory");\n      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(su32(&empty[s])) : "memory");\n    }\n  }\n  }  // next persistent logical tile (all cluster partners use the same schedule)\n#endif\n}\n\nstatic PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {\n  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;\n  if (!fn) {\n    cudaDriverEntryPointQueryResult q;\n    void* p = nullptr;\n    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);\n    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled unavailable");\n    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);\n  }\n  return fn;\n}\n\n// 2D BF16 tensor map over a row-major [rows, cols] matrix, box [box_rows, 64], 128-byte swizzle.\nstatic CUtensorMap make_map(const void* ptr, int rows, int cols, int box_rows) {\n  CUtensorMap m;\n  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};\n  cuuint64_t strides[1] = {(cuuint64_t)cols * 2};\n  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};\n  cuuint32_t es[2] = {1, 1};\n  CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides,\n                           box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,\n                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);\n  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);\n  return m;\n}\n\ntemplate <int BT, int NSUB, int NCW, int ST, int MC>\nvoid wg_launch_t(const torch::Tensor& W, const torch::Tensor& X, torch::Tensor& Out, int N, int ks, int waves, int pipe, int pdl, int kgroup) {\n  const int M = X.size(0), KX = X.size(1), KW = W.size(1);\n  constexpr int BR = 64 * NSUB * NCW;\n  TORCH_CHECK(N % BR == 0 && KW % 64 == 0 && KX % 64 == 0, "wg: bad shape");\n  const bool f32 = Out.scalar_type() == at::kFloat;\n  auto kern = f32 ? wg_kernel<BT, NSUB, NCW, ST, MC, true> : wg_kernel<BT, NSUB, NCW, ST, MC, false>;\n  TORCH_CHECK((N / BR) % MC == 0 && BT % (8 * MC) == 0, "wg: multicast pair needs an even tile count");\n  TORCH_CHECK(kgroup == 1 || kgroup == 2 || kgroup == 4, "K group");\n  TORCH_CHECK((KW/64) % (ks*kgroup) == 0, "K partition/group");\n  const size_t stage = (size_t)(BR + BT) * 64 * 2 * kgroup;\n  size_t smem = ST * stage + 2 * ST * 8 + 1024;\n  const size_t red = (size_t)64 * (BR + 4) * 4 + 1024;\n  if (red > smem) smem = red;\n  static int optin = 0;\n  static bool attr_set[2] = {false, false};\n  if (!attr_set[f32]) {  // once per instance, before any graph capture (prepare runs every config)\n    int dev = 0;\n    cudaGetDevice(&dev);\n    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);\n    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n    attr_set[f32] = true;\n  }\n  TORCH_CHECK(smem <= (size_t)optin, "wg: too much shared memory: ", smem, " > ", optin);\n  CUtensorMap mw = make_map(W.data_ptr(), N, KW, BR);\n  CUtensorMap mx = make_map(X.data_ptr(), M, KX, BT / MC);\n  cudaLaunchConfig_t cfg = {};\n  int dev = 0, sms = 0;\n  cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);\n  const int logical = ((N / BR) / MC) * ((M + BT - 1) / BT);\n  const int groups = waves ? min(logical, max(1, sms * waves / (MC * ks))) : logical;\n  cfg.gridDim = dim3(groups * MC, ks, 1);\n  cfg.blockDim = dim3(128 * NCW + 32);\n  cfg.dynamicSmemBytes = smem;\n  cfg.stream = at::cuda::getCurrentCUDAStream();\n  cudaLaunchAttribute attr[2];\n  cfg.numAttrs = 0;\n  if (ks > 1 || MC > 1) {\n    attr[0].id = cudaLaunchAttributeClusterDimension;\n    attr[0].val.clusterDim.x = MC;\n    attr[0].val.clusterDim.y = ks;\n    attr[0].val.clusterDim.z = 1;\n    cfg.numAttrs = 1;\n  }\n  if (pdl) {\n    attr[cfg.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n    attr[cfg.numAttrs].val.programmaticStreamSerializationAllowed = 1;\n    ++cfg.numAttrs;\n  }\n  cfg.attrs = attr;\n  auto err = cudaLaunchKernelEx(&cfg, kern, mw, mx, Out.data_ptr(), M, N, KW / 64, KX, pipe, pdl, kgroup);\n  TORCH_CHECK(err == cudaSuccess, "wg launch: ", cudaGetErrorString(err));\n}\n\n}  // namespace muse_wg\n\n// out[M, N] = X[M, KX] @ W[:N, :KW]^T with the X column index taken mod KX (KW = 2*KX: the\n// router\'s [hi | lo]). Hopper (sm_90a) only.\nvoid wg_gemm(torch::Tensor W, torch::Tensor X, torch::Tensor Out, int64_t N, int64_t bt, int64_t nsub,\n             int64_t ncw, int64_t st, int64_t ks, int64_t mc, int64_t waves, int64_t pipe, bool pdl, int64_t kgroup) {\n  using namespace muse_wg;\n#define WG_CASE(BT_, NSUB_, NCW_, ST_, MC_)                                          \\\n  if (bt == BT_ && nsub == NSUB_ && ncw == NCW_ && st == ST_ && mc == MC_) {         \\\n    wg_launch_t<BT_, NSUB_, NCW_, ST_, MC_>(W, X, Out, (int)N, (int)ks, (int)waves, (int)pipe, (int)pdl, (int)kgroup);             \\\n    return;                                                                          \\\n  }\n  WG_CASE(32, 1, 1, 2, 1) WG_CASE(32, 1, 1, 2, 2) WG_CASE(64, 1, 1, 2, 1) WG_CASE(64, 1, 1, 2, 2) WG_CASE(64, 1, 1, 3, 1) WG_CASE(64, 1, 1, 3, 2) WG_CASE(128, 1, 1, 2, 1) WG_CASE(128, 1, 1, 2, 2) WG_CASE(128, 1, 1, 3, 1) WG_CASE(128, 1, 1, 3, 2) WG_CASE(64, 1, 2, 2, 1) WG_CASE(64, 1, 2, 2, 2) WG_CASE(256, 1, 1, 2, 1) WG_CASE(256, 1, 1, 2, 2) WG_CASE(64, 1, 1, 4, 1) WG_CASE(64, 1, 1, 4, 2) WG_CASE(64, 1, 1, 5, 1) WG_CASE(64, 1, 1, 5, 2) WG_CASE(64, 1, 1, 6, 1) WG_CASE(64, 1, 1, 6, 2) WG_CASE(128, 1, 1, 4, 1) WG_CASE(128, 1, 1, 4, 2) WG_CASE(128, 1, 1, 6, 1) WG_CASE(128, 1, 1, 6, 2) WG_CASE(256, 1, 1, 3, 1) WG_CASE(256, 1, 1, 3, 2) WG_CASE(256, 1, 1, 4, 1) WG_CASE(256, 1, 1, 4, 2) WG_CASE(256, 1, 1, 5, 1) WG_CASE(256, 1, 1, 5, 2) WG_CASE(64, 1, 2, 4, 1) WG_CASE(64, 1, 2, 4, 2) WG_CASE(128, 1, 2, 3, 1) WG_CASE(128, 1, 2, 3, 2) WG_CASE(128, 1, 2, 4, 1) WG_CASE(128, 1, 2, 4, 2) WG_CASE(256, 1, 2, 3, 1)\n  TORCH_CHECK(false, "wg: no instance for bt=", bt, " nsub=", nsub, " ncw=", ncw, " st=", st, " mc=", mc);\n}\n'
VG_INSTANCES=[(32, 1, 1, 2, 1), (32, 1, 1, 2, 2), (64, 1, 1, 2, 1), (64, 1, 1, 2, 2), (64, 1, 1, 3, 1), (64, 1, 1, 3, 2), (128, 1, 1, 2, 1), (128, 1, 1, 2, 2), (128, 1, 1, 3, 1), (128, 1, 1, 3, 2), (64, 1, 2, 2, 1), (64, 1, 2, 2, 2), (256, 1, 1, 2, 1), (256, 1, 1, 2, 2), (64, 1, 1, 4, 1), (64, 1, 1, 4, 2), (64, 1, 1, 5, 1), (64, 1, 1, 5, 2), (64, 1, 1, 6, 1), (64, 1, 1, 6, 2), (128, 1, 1, 4, 1), (128, 1, 1, 4, 2), (128, 1, 1, 6, 1), (128, 1, 1, 6, 2), (256, 1, 1, 3, 1), (256, 1, 1, 3, 2), (256, 1, 1, 4, 1), (256, 1, 1, 4, 2), (256, 1, 1, 5, 1), (256, 1, 1, 5, 2), (64, 1, 2, 4, 1), (64, 1, 2, 4, 2), (128, 1, 2, 3, 1), (128, 1, 2, 3, 2), (128, 1, 2, 4, 1), (128, 1, 2, 4, 2), (256, 1, 2, 3, 1)]
_vg_tma_mods={}


def vg_tma_module(arch=None):
    from torch.utils.cpp_extension import load_inline
    target=arch or ('9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0')
    tag=hashlib.sha256(VG_SOURCE.encode()).hexdigest()[:12]+target.replace('.','_')
    if tag not in _vg_tma_mods:
        previous=os.environ.get('TORCH_CUDA_ARCH_LIST');os.environ['TORCH_CUDA_ARCH_LIST']=target
        try:
            _vg_tma_mods[tag]=load_inline(name='muse_verify_grouped_'+tag,
                cpp_sources='''#include <torch/extension.h>
void wg_gemm(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,int64_t);''',
                cuda_sources=VG_SOURCE,functions=['wg_gemm'],
                extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','--split-compile=16'],verbose=False)
        finally:
            if previous is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:os.environ['TORCH_CUDA_ARCH_LIST']=previous
    return _vg_tma_mods[tag]


def vg_tma(x,w,choice,out_dtype=torch.bfloat16,pdl=False,mod=None):
    bt,nsub,ncw,st,mc,ks,waves,pipe,kgroup=choice
    out=torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=out_dtype)
    native=vg_tma_module() if mod is None else mod
    native.wg_gemm(w,x,out,w.shape[0],bt,nsub,ncw,st,ks,mc,waves,pipe,pdl,kgroup)
    return out


def vg_tma_choices(m,n,k,limit=227328):
    choices=[]
    for kgroup in (2,4,1):
        for bt,ns,wg,st,mc in VG_INSTANCES:
            br=64*ns*wg;shared=max(st*(br+bt)*128*kgroup+16*st+1024,64*(br+4)*4+1024)
            if bt>m*2 or n%br or (n//br)%mc or shared>limit:continue
            for ks in (1,2,4):
                if ks*mc>8 or (k//64)%(ks*kgroup):continue
                for waves in (0,1,2):choices.append((bt,ns,wg,st,mc,ks,waves,1,kgroup))
    return choices
