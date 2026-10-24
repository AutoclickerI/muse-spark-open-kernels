"""PDL dense projection drop-in; constructs only before graph capture."""
PP_CUDA='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\n// Each producer writes the decode layer\'s [split,token,row] FP32 ABI.\n// Only immutable weights may be read before griddepcontrol.wait.\nnamespace pp {\nusing B = __nv_bfloat16;\n__device__ __forceinline__ void start(bool pdl) {\n  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n}\n__device__ __forceinline__ void wait(bool pdl) {\n  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");\n}\n__device__ __forceinline__ unsigned shared(const void* p) {\n  return (unsigned)__cvta_generic_to_shared(p);\n}\n__device__ __forceinline__ void cp16(B* d, const B* s, bool valid) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"\n      :: "r"(shared(d)), "l"(s), "r"(valid ? 16 : 0) : "memory");\n}\n__device__ __forceinline__ void mma(float* d, uint4 a, uint4 b, uint4 x) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\n      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n      : "r"(a.x), "r"(b.x), "r"(a.y), "r"(b.y), "r"(x.x), "r"(x.y));\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\n      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n      : "r"(a.z), "r"(b.z), "r"(a.w), "r"(b.w), "r"(x.z), "r"(x.w));\n}\n\ntemplate<int R, int T, int BK, int STAGES, bool EARLY, bool PACKED=false>\n__global__ __launch_bounds__(128)\nvoid bulk_ring(const B* __restrict__ w, const B* __restrict__ x,\n               float* __restrict__ p, int M, int N, int K, int KS, bool pdl) {\n  constexpr int STRIDE = BK + 32;\n  constexpr int CELLS = (R / 16) * (T / 8) / 4;\n  extern __shared__ __align__(16) unsigned char raw[];\n  B* sw = reinterpret_cast<B*>(raw);\n  B* sx = sw + STAGES * R * STRIDE;\n  __shared__ __align__(8) uint64_t bars[STAGES];\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int rn = blockIdx.x * R, tm = blockIdx.z * T;\n  const int begin = blockIdx.y * KS, tiles = KS / BK;\n  start(pdl);\n  if (tid < STAGES)\n    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(shared(bars + tid)) : "memory");\n  __syncthreads();\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n  __syncthreads();\n  auto issue = [&](int tile) {\n    const int slot = tile % STAGES;\n    if (tid == 0) {\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"\n          :: "r"(shared(bars + slot)), "r"(R * (PACKED ? STRIDE : BK) * 2) : "memory");\n      if constexpr (PACKED) {\n        const B* src = w + ((size_t)blockIdx.x * (K/BK) + begin/BK + tile) * R * STRIDE;\n        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n            :: "r"(shared(sw + slot * R * STRIDE)), "l"(src),\n               "r"(R * STRIDE * 2), "r"(shared(bars + slot)) : "memory");\n      } else {\n      // A row is contiguous and original BF16. The entire request group is\n      // issued before waiting for the dependent activation producer.\n#pragma unroll\n      for (int r = 0; r < R; ++r) {\n        const B* src = w + (size_t)(rn + r) * K + begin + tile * BK;\n        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\n            :: "r"(shared(sw + (slot * R + r) * STRIDE)), "l"(src),\n               "r"(BK * 2), "r"(shared(bars + slot)) : "memory");\n      }\n      }\n    }\n  };\n  if (!EARLY) wait(pdl);\n#pragma unroll\n  for (int st = 0; st < STAGES; ++st) if (st < tiles) issue(st);\n  if (EARLY) wait(pdl);\n  float acc[CELLS][4] = {};\n  for (int tile = 0; tile < tiles; ++tile) {\n    const int slot = tile % STAGES;\n    // Activation accesses start strictly after wait. No activation is copied\n    // by the initial weight prologue.\n    for (int v = tid; v < T * BK / 8; v += 128) {\n      int t = v / (BK / 8), k = (v % (BK / 8)) * 8;\n      cp16(sx + t * STRIDE + k,\n           x + (size_t)(tm + (tm + t < M ? t : 0)) * K + begin + tile * BK + k,\n           tm + t < M);\n    }\n    asm volatile("cp.async.commit_group;\\ncp.async.wait_group 0;" ::: "memory");\n    const unsigned bar = shared(bars + slot), phase = (tile / STAGES) & 1;\n    asm volatile("{ .reg .pred done;\\nPP_WAIT: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\\n@!done bra PP_WAIT; }"\n        :: "r"(bar), "r"(phase) : "memory");\n    __syncthreads();\n#pragma unroll\n    for (int k = 0; k < BK; k += 32) {\n#pragma unroll\n      for (int c = 0; c < CELLS; ++c) {\n        const int cell = warp + c * 4;\n        const int r = (cell / (T / 8)) * 16 + lane / 4;\n        const int t = (cell % (T / 8)) * 8 + lane / 4;\n        const int kk = k + (lane % 4) * 8;\n        uint4 a = *reinterpret_cast<const uint4*>(sw + (slot * R + r) * STRIDE + kk);\n        uint4 b = *reinterpret_cast<const uint4*>(sw + (slot * R + r + 8) * STRIDE + kk);\n        uint4 xv = *reinterpret_cast<const uint4*>(sx + t * STRIDE + kk);\n        mma(acc[c], a, b, xv);\n      }\n    }\n    __syncthreads();\n    if (tile + STAGES < tiles) issue(tile + STAGES);\n  }\n#pragma unroll\n  for (int c = 0; c < CELLS; ++c) {\n    const int cell = warp + c * 4;\n    const int r = rn + (cell / (T / 8)) * 16 + lane / 4;\n    const int t = tm + (cell % (T / 8)) * 8 + (lane % 4) * 2;\n#pragma unroll\n    for (int i = 0; i < 4; ++i)\n      if (t + (i & 1) < M)\n        p[((size_t)blockIdx.y * M + t + (i & 1)) * N + r + (i / 2) * 8] = acc[c][i];\n  }\n}\n\n__device__ __forceinline__ uint4 ld16(const B* p) {\n  uint4 v;\n  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"\n      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");\n  return v;\n}\n// 512-thread CTAs, 16 independent row warps, 32/64 KiB of issued weight\n// vectors per CTA. More resident warps and independent bytes than the old\n// 128-thread GEMV. Splits make QKV/pool grids close to the 528-CTA stream.\ntemplate<int U, bool EARLY>\n__global__ __launch_bounds__(512)\nvoid wide_gemv(const B* __restrict__ w, const B* __restrict__ x,\n               float* __restrict__ p, int M, int N, int K, int KS, bool pdl) {\n  const int lane = threadIdx.x & 31, row = blockIdx.x * 16 + threadIdx.x / 32;\n  const int tok = blockIdx.z, begin = blockIdx.y * KS;\n  uint4 wv[U];\n  float acc[8] = {};\n  start(pdl);\n  if (!EARLY) wait(pdl);\n#pragma unroll\n  for (int u = 0; u < U; ++u)\n    wv[u] = ld16(w + (size_t)row * K + begin + u * 256 + lane * 8);\n  if (EARLY) wait(pdl);\n  for (int base = begin; base < begin + KS; base += U * 256) {\n#pragma unroll\n    for (int u = 0; u < U; ++u) {\n      const uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)tok * K + base + u * 256 + lane * 8);\n      const B* xx = reinterpret_cast<const B*>(&xv);\n      const B* ww = reinterpret_cast<const B*>(&wv[u]);\n#pragma unroll\n      for (int j = 0; j < 8; ++j) acc[j] = fmaf(__bfloat162float(ww[j]), __bfloat162float(xx[j]), acc[j]);\n    }\n    if (base + U * 256 < begin + KS) {\n#pragma unroll\n      for (int u = 0; u < U; ++u)\n        wv[u] = ld16(w + (size_t)row * K + base + U * 256 + u * 256 + lane * 8);\n    }\n  }\n  float sum = 0.f;\n#pragma unroll\n  for (int j = 0; j < 8; ++j) sum += acc[j];\n#pragma unroll\n  for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);\n  if (lane == 0) p[((size_t)blockIdx.y * M + tok) * N + row] = sum;\n}\n\n__global__ __launch_bounds__(512)\nvoid read_stream(const B* w, unsigned* out, size_t vectors) {\n  size_t first = (size_t)blockIdx.x * 512 * 4 + threadIdx.x;\n  unsigned hash = 0;\n  for (size_t base = first; base < vectors; base += (size_t)gridDim.x * 512 * 4) {\n    uint4 v[4];\n#pragma unroll\n    for (int u = 0; u < 4; ++u) v[u] = base + u * 512 < vectors ? ld16(w + (base + u * 512) * 8) : uint4{};\n#pragma unroll\n    for (int u = 0; u < 4; ++u) hash ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;\n  }\n  out[blockIdx.x * 512 + threadIdx.x] = hash;\n}\n\ntemplate<int R, int T, int BK, int S, bool EARLY, bool PACKED=false>\ncudaError_t launch_ring(const B* w, const B* x, float* p, int M, int N,\n                       int K, int KS, bool pdl, cudaStream_t stream) {\n  constexpr int bytes = (S * R + T) * (BK + 32) * 2;\n  auto fn = bulk_ring<R,T,BK,S,EARLY,PACKED>;\n  auto e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);\n  if (e != cudaSuccess) return e;\n  cudaLaunchAttribute attr{};\n  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;\n  attr.val.programmaticStreamSerializationAllowed = pdl;\n  cudaLaunchConfig_t cfg{};\n  cfg.gridDim = dim3(N/R,K/KS,(M+T-1)/T); cfg.blockDim = 128;\n  cfg.dynamicSmemBytes = bytes; cfg.stream = stream; cfg.attrs = &attr; cfg.numAttrs = 1;\n  return cudaLaunchKernelEx(&cfg, fn, w, x, p, M, N, K, KS, pdl);\n}\ntemplate<int U, bool EARLY>\ncudaError_t launch_wide(const B* w, const B* x, float* p, int M, int N,\n                       int K, int KS, bool pdl, cudaStream_t stream) {\n  cudaLaunchAttribute attr{};\n  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;\n  attr.val.programmaticStreamSerializationAllowed = pdl;\n  cudaLaunchConfig_t cfg{};\n  cfg.gridDim = dim3(N/16,K/KS,M); cfg.blockDim = 512;\n  cfg.stream = stream; cfg.attrs = &attr; cfg.numAttrs = 1;\n  return cudaLaunchKernelEx(&cfg, wide_gemv<U,EARLY>, w, x, p, M, N, K, KS, pdl);\n}\n} // namespace pp\n\nextern "C" int pp_run(const void* w, const void* x, void* p, int M, int N,\n                       int K, int KS, int config, int early, int pdl, void* stream) {\n  using namespace pp;\n#define PP_ARGS (const B*)w, (const B*)x, (float*)p, M, N, K, KS, bool(pdl), (cudaStream_t)stream\n#define PP_RING(R,T,BK,S) (early ? launch_ring<R,T,BK,S,true>(PP_ARGS) : launch_ring<R,T,BK,S,false>(PP_ARGS))\n#define PP_PACK(R,T,BK,S) (early ? launch_ring<R,T,BK,S,true,true>(PP_ARGS) : launch_ring<R,T,BK,S,false,true>(PP_ARGS))\n#define PP_WIDE(U) (early ? launch_wide<U,true>(PP_ARGS) : launch_wide<U,false>(PP_ARGS))\n  cudaError_t e = cudaErrorInvalidValue;\n  switch (config) {\n    case 0: e = PP_RING(32,16,256,3); break;\n    case 1: e = PP_RING(32,32,256,4); break;\n    case 2: e = PP_RING(64,8,128,4); break;\n    case 3: e = PP_WIDE(4); break;\n    case 4: e = PP_WIDE(8); break;\n    case 5: e = PP_PACK(32,16,256,3); break;\n    case 6: e = PP_PACK(32,32,256,4); break;\n    case 7: e = PP_PACK(64,8,128,4); break;\n  }\n  if (e != cudaSuccess) return (int)e;\n  return (int)cudaGetLastError();\n}\nextern "C" int pp_read(const void* w, void* out, size_t bytes, int ctas, void* stream) {\n  pp::read_stream<<<ctas,512,0,(cudaStream_t)stream>>>((const pp::B*)w,(unsigned*)out,bytes/16);\n  return (int)cudaGetLastError();\n}\n'
import ctypes as _ppct
import hashlib as _pphash
import os as _ppos

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline as _pp_load

PP_CONFIGS = {0:(32,16,256,3), 1:(32,32,256,4), 2:(64,8,128,4),
              3:(16,1,1024,1), 4:(16,1,2048,1), 5:(32,16,256,3),
              6:(32,32,256,4), 7:(64,8,128,4)}
_PP_LIBRARY = None
_PP_PACKED = {}


def pp_library():
    global _PP_LIBRARY
    if _PP_LIBRARY is None:
        old = _ppos.environ.get('TORCH_CUDA_ARCH_LIST')
        _ppos.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        try:
            path = _pp_load(name='muse_pdl_dense_'+_pphash.sha256(PP_CUDA.encode()).hexdigest()[:16],
                cpp_sources='',cuda_sources=PP_CUDA,functions=None,is_python_module=False,
                no_implicit_headers=True,extra_cuda_cflags=['-O3','-std=c++17','-lineinfo','--ptxas-options=-v'],
                verbose=False)
        finally:
            if old is None:_ppos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_ppos.environ['TORCH_CUDA_ARCH_LIST']=old
        _PP_LIBRARY = _ppct.CDLL(path)
        _PP_LIBRARY.pp_run.argtypes=[_ppct.c_void_p]*3+[_ppct.c_int]*7+[_ppct.c_void_p]
        _PP_LIBRARY.pp_run.restype=_ppct.c_int
        _PP_LIBRARY.pp_read.argtypes=[_ppct.c_void_p]*2+[_ppct.c_size_t,_ppct.c_int,_ppct.c_void_p]
        _PP_LIBRARY.pp_read.restype=_ppct.c_int
    return _PP_LIBRARY


class PPDense:
    def __init__(self,w,tokens,config,ks,early=True,pdl=True):
        r,t,bk,stages=PP_CONFIGS[config]
        if w.dtype!=torch.bfloat16 or not w.is_contiguous() or w.shape[0]%r or w.shape[1]%ks or ks%bk:
            raise ValueError('invalid dense weight geometry')
        self.lib=pp_library();self.w=w;self.tokens=tokens
        self.bank=w
        if config>=5:
            key=(w.data_ptr(),tuple(w.shape),r,bk)
            if key not in _PP_PACKED:
                tiled=w.reshape(w.shape[0]//r,r,w.shape[1]//bk,bk).permute(0,2,1,3).contiguous()
                _PP_PACKED[key]=(w,torch.nn.functional.pad(tiled,(0,32)))
            self.bank=_PP_PACKED[key][1]
        self.config=config;self.ks=ks;self.early=bool(early);self.pdl=bool(pdl)
        self.splits=w.shape[1]//ks
        self.out=torch.empty((self.splits,tokens,w.shape[0]),device=w.device,dtype=torch.float32)
        self.ctas=(w.shape[0]//r)*self.splits*triton.cdiv(tokens,t)
        self.flight_per_cta=r*(bk+32 if config>=5 else bk)*2*stages

    def __call__(self,x):
        if x.dtype!=torch.bfloat16 or x.shape!=(self.tokens,self.w.shape[1]) or not x.is_contiguous():
            raise ValueError('dense activation ABI changed')
        err=self.lib.pp_run(self.bank.data_ptr(),x.data_ptr(),self.out.data_ptr(),self.tokens,
            self.w.shape[0],self.w.shape[1],self.ks,self.config,int(self.early),int(self.pdl),
            torch.cuda.current_stream(x.device).cuda_stream)
        if err:raise RuntimeError(f'PDL dense CUDA={err} config={self.config} KS={self.ks}')
        return self.out,self.splits


@triton.jit
def pp_bulk_hint(PTR,SIZE:tl.constexpr):
    return tl.inline_asm_elementwise(
        'cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0;',
        '=r,l,r',[PTR,SIZE],dtype=tl.int32,is_pure=False,pack=1)


@triton.jit
def pp_hint_gemv(X,W,P,T,x_stride,N:tl.constexpr,K:tl.constexpr,KS:tl.constexpr,
                 BN:tl.constexpr,BK:tl.constexpr,TB:tl.constexpr,DEPTH:tl.constexpr,USE_PDL:tl.constexpr):
    pn=tl.program_id(0);ps=tl.program_id(1)
    if USE_PDL:tl.extra.cuda.gdc_launch_dependents()
    n=pn*BN+tl.arange(0,BN);k0=ps*KS
    if DEPTH:
        pp_bulk_hint(W+n*K+k0,min(KS,BK*DEPTH)*2)
    if USE_PDL:tl.extra.cuda.gdc_wait()
    t=tl.arange(0,TB);kk=tl.arange(0,BK);acc=tl.zeros((BN,TB),tl.float32)
    for k in range(k0,k0+KS,BK):
        w=tl.load(W+n[:,None]*K+(k+kk)[None,:])
        x=tl.load(X+t[:,None]*x_stride+(k+kk)[None,:],mask=t[:,None]<T,other=0.)
        acc=tl.dot(w,tl.trans(x),acc)
    tl.store(P+ps*(T*N)+t[None,:]*N+n[:,None],acc,mask=t[None,:]<T)


def pp_hint_project(m,x,w,name,state,depth,retain_dispatch):
    choice=state['proj_plan'].get((x.shape[0],name),'custom')
    if retain_dispatch and choice!='custom':return m.project(x,w,name,state,out_dtype=torch.float32 if name=='pool_router' else torch.bfloat16)
    tokens,k=x.shape;n=w.shape[0]
    cfg=m._GEMV_CHOICES.get((tokens,n,k))
    if not isinstance(cfg,tuple):cfg=(*m.GEMV_CFG[n,k],False)
    bn,ks,bk,_=cfg
    p=torch.empty((k//ks,tokens,n),device=x.device,dtype=torch.float32)
    pp_hint_gemv[(n//bn,k//ks)](x,w,p,tokens,x.stride(0),N=n,K=k,KS=ks,BN=bn,BK=bk,
        TB=max(16,triton.next_power_of_2(tokens)),DEPTH=depth,USE_PDL=True,
        launch_pdl=True,num_warps=4,num_stages=4)
    return p,k//ks


@triton.jit
def pp_bf16_boundary(P,Y,S:tl.constexpr,COUNT:tl.constexpr,B:tl.constexpr,USE_PDL:tl.constexpr):
    if USE_PDL:tl.extra.cuda.gdc_launch_dependents()
    if USE_PDL:tl.extra.cuda.gdc_wait()
    i=tl.program_id(0)*B+tl.arange(0,B)
    acc=tl.full((B,),0.,tl.float32)
    for s in range(S):acc+=tl.load(P+s*COUNT+i,i<COUNT,0.)
    tl.store(Y+i,acc,i<COUNT)


def pp_preserve_boundary(m,x,name,state,value):
    # Baseline library producers return BF16. Preserve that quantization
    # boundary rather than silently feed higher precision to later norms.
    choice=state['proj_plan'].get((x.shape[0],name),'custom')
    if name=='pool_router' or choice=='custom':return value
    p,s=value
    out=torch.empty((1,x.shape[0],state[name].shape[0]),device=x.device,dtype=torch.bfloat16)
    count=out.numel()
    pp_bf16_boundary[(triton.cdiv(count,256),)](p,out,S=s,COUNT=count,B=256,USE_PDL=True,
        launch_pdl=True,num_warps=4)
    return out,1
