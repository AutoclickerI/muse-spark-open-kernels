# okc: problem=muse_bf16_proj
"""Projection portfolio: measured small-M kernels and synthetic cold-L2 graph autotuning.
Hopper: TMA/WGMMA, direct/cluster/last-CTA split-K and balanced Stream-K.
Router: each K tile loads hi and lo together and updates the same FP32 accumulators.
Router and qkv verify paths always launch one fused kernel with an in-kernel merge.
"""

import math

import os

from typing import TypedDict

import torch

import torch.nn.functional as F

class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor

class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str

class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config

INSTANCES = [(1, 1, 4, 1, 1, 0), (1, 2, 4, 1, 1, 0),
             (2, 1, 4, 1, 1, 0), (2, 2, 4, 1, 1, 0),
             (4, 1, 4, 1, 1, 0), (8, 1, 2, 1, 1, 0),
             (1, 1, 4, 2, 1, 0), (2, 1, 4, 2, 1, 0),
             (4, 1, 4, 2, 0, 0)]

_MOD = None

SKINNY_MAX_M = 64

def _nt(m: int) -> int:
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8

CONFIGS = {
    ("qkv_gate", 1): (1, 2, 1, 8, 1, 0),
    ("qkv_gate", 2): (2, 4, 1, 8, 1, 1),
    ("qkv_gate", 4): (1, 4, 1, 8, 1, 1),
    ("qkv_gate", 8): (1, 2, 2, 4, 2, 1),
    ("o", 1): (1, 2, 1, 8, 1, 0),
    ("o", 2): (1, 2, 1, 8, 1, 0),
    ("o", 4): (1, 4, 8, 1, 1, 1),
    ("o", 8): (1, 2, 8, 1, 1, 1),
    ("pooler", 1): (1, 2, 2, 4, 1, 1),
    ("pooler", 2): (1, 4, 2, 4, 1, 1),
    ("pooler", 4): (2, 2, 1, 8, 1, 1),
    ("pooler", 8): (1, 2, 8, 1, 4, 1),
    ("unpooler", 1): (1, 2, 1, 8, 1, 1),
    ("unpooler", 2): (1, 4, 2, 4, 1, 0),
    ("unpooler", 4): (1, 4, 2, 4, 1, 1),
    ("unpooler", 8): (1, 2, 2, 4, 1, 1),
    ("router", 1): (1, 2, 1, 8, 8, 1),
    ("router", 2): (1, 2, 1, 8, 8, 1),
    ("router", 4): (1, 2, 1, 8, 4, 1),
    ("router", 8): (1, 2, 1, 8, 4, 0),
}

TUNED_CONFIGS = {
    ("qkv_gate", 1): (1, 4, 4, 1, 8, 1),
    ("qkv_gate", 8): (2, 4, 1, 8, 1, 1),
    ("qkv_gate", 16): (1, 4, 4, 1, 8, 1),
    ("qkv_gate", 32): (1, 2, 4, 1, 8, 1),
    ("qkv_gate", 64): (1, 2, 2, 4, 1, 1),
    ("o", 1): (1, 4, 1, 4, 1, 1),
    ("o", 8): (1, 2, 1, 4, 1, 1),
    ("o", 16): (1, 4, 2, 4, 1, 0),
    ("o", 32): (1, 4, 2, 4, 1, 0),
    ("o", 64): (1, 2, 2, 4, 1, 1),
    ("pooler", 1): (1, 4, 1, 4, 1, 1),
    ("pooler", 8): (1, 4, 1, 4, 1, 1),
    ("pooler", 16): (2, 4, 1, 8, 1, 1),
    ("pooler", 32): (1, 4, 2, 4, 1, 1),
    ("pooler", 64): (1, 2, 2, 4, 1, 1),
    ("unpooler", 1): (1, 4, 1, 8, 1, 1),
    ("unpooler", 8): (1, 4, 1, 8, 1, 1),
    ("unpooler", 16): (1, 2, 2, 4, 1, 1),
    ("unpooler", 32): (1, 2, 2, 4, 1, 1),
    ("unpooler", 64): (1, 2, 2, 4, 1, 1),
    ("router", 1): (1, 4, 2, 4, 8, 1),
    ("router", 8): (1, 4, 2, 4, 8, 1),
    ("router", 16): (1, 4, 1, 8, 4, 1),
    ("router", 32): (1, 4, 1, 8, 4, 0),
}

def _prepare_small(weights: Weights):
    _module()
    state = {name: w.contiguous() for name, w in weights.items() if name != "router"}
    router = weights["router"].float().contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["mod"] = _module()
    return state

def skinny_gemm(x: torch.Tensor, w: torch.Tensor, n: int, layer: str, w_lo: torch.Tensor | None = None,
                out_dtype=torch.bfloat16) -> torch.Tensor:
    m = x.shape[0]
    nt = _nt(min(m, SKINNY_MAX_M))
    key_m = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 64
    tr, u, wn, wk, ks, xp = TUNED_CONFIGS.get((layer, key_m), CONFIGS[(layer, nt)])
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    _module().skinny(w, w if w_lo is None else w_lo, x, out, n, nt, tr, u, wn, wk, ks, w_lo is not None, xp, 0)
    return out

def _small(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    # The skinny kernel is useful only for the measured small-M winners.  Keep
    # cuBLAS for the shapes where its larger GEMM tiles are faster, and use the
    # standard split-BF16 router path once M is beyond the skinny regime.
    if layer == "router":
        if m <= 32:
            return skinny_gemm(x, state["router_hi"], n, layer, state["router_lo"], torch.float32)
        hi_t = state["router_hi_t"]
        lo_t = state["router_lo_t"]
        acc = torch.mm(x, hi_t, out_dtype=torch.float32)
        acc += torch.mm(x, lo_t, out_dtype=torch.float32)
        return acc
    use_skinny = ((layer == "qkv_gate" and m <= 16)
                  or (layer == "o" and m == 1)
                  or (layer == "pooler" and m <= 16)
                  or (layer == "unpooler" and m <= 8))
    if use_skinny:
        return skinny_gemm(x, state[layer], n, layer)
    return F.linear(x, state[layer][:n])

def _hopper():
    return torch.cuda.get_device_capability() == (9,0)

def _launch_hop(x, state, layer, n, config, scratch):
    out=torch.empty((x.shape[0],n),dtype=torch.float32 if layer=='router' else torch.bfloat16,device=x.device)
    if config[0]=='lt_nn':
        key=(layer,x.shape[0],n,x.shape[1])
        state['mod'].lt_nn_run(state['lt_transposed'][layer],x,out,state['lt_workspace'],state['lt_nn_algorithms'][key],config[1])
        return out
    if config[0]=='lt':
        key=(layer,x.shape[0],n,x.shape[1])
        state['mod'].lt_run(state[layer],x,out,state['lt_workspace'],state['lt_algorithms'][key],config[1])
        return out
    bt,bk,st,mode,split,threads=config
    w=state['router_pair'] if layer=='router' else state[layer]
    state['mod'].hopper(w,x,out,*scratch,bt,bk,st,mode,split,threads)
    return out

class _TuneGraph:
    """Own a native stream: graph destruction must not clear another graph's cuBLAS workspace."""
    def __init__(self):
        import ctypes
        handle=ctypes.c_void_p()
        self.runtime=torch.cuda.cudart()
        err=self.runtime.cudaStreamCreate(ctypes.addressof(handle))
        if int(err)!=0:
            raise RuntimeError(f'cudaStreamCreate failed: {err}')
        self.handle=handle.value
        self.stream=torch.cuda.ExternalStream(self.handle)
        self.graph=torch.cuda.CUDAGraph()

    def replay(self):
        self.graph.replay()

    def __del__(self):
        if hasattr(self,'handle'):
            self.graph.reset()
            self.runtime.cudaStreamDestroy(self.handle)

def _graph(fn):
    graph=_TuneGraph()
    graph.stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(graph.stream):
        fn()
    graph.stream.synchronize()
    with torch.cuda.graph(graph.graph,stream=graph.stream):
        out=fn()
    graph.replay()
    return graph,out

def _cold_times(graphs, flush, repeats=3):
    """Paired graphs with a fresh cache flush before each sample; returns microseconds."""
    samples=[[] for _ in graphs]
    for r in range(repeats):
        for i in (range(len(graphs)) if r%2==0 else reversed(range(len(graphs)))):
            flush.add_(1)  # Read and overwrite more than L2 before every timed replay.
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record(); graphs[i].replay(); b.record()
            try:
                b.synchronize()
            except RuntimeError as exc:
                raise RuntimeError(f'portfolio graph timing failed at graph {i}/{len(graphs)}, repetition {r}') from exc
            samples[i].append(a.elapsed_time(b)*1000)
    import statistics
    return [statistics.median(v) for v in samples]

def _shared_bytes(layer,bt,bk,st):
    return max(st*(64*(2 if layer=='router' else 1)+bt)*bk*2+st*16,bt*68*4)

def _variants(m,n,k,sm_count):
    layer='router' if n==256 else 'qkv_gate' if n==2304 else None
    yield from FUSED_CONFIGS.get((layer,m),())

def prepare(weights: Weights):
    import time
    started=time.monotonic()
    state=_prepare_small(weights)
    state['choices']={}
    state['scratch']={}
    state['lt_algorithms']={}
    state['lt_nn_algorithms']={}
    state['lt_transposed']={layer:w.T.contiguous() for layer,w in weights.items() if layer not in ('router','qkv_gate')}
    state['router_pair']=torch.cat([state['router_hi'],state['router_lo']],dim=1).contiguous()
    state['lt_workspace']=torch.empty(32*1024*1024,device=state['router_hi'].device,dtype=torch.uint8)
    flush=torch.zeros(128*1024*1024,device=state['router_hi'].device,dtype=torch.uint8)
    props=torch.cuda.get_device_properties(flush.device)
    # One portfolio check measures all paths on synthetic activations; none are derived from data.
    for layer in ('router','qkv_gate','pooler','unpooler','o'):
        n,k=weights[layer].shape
        for m in (128,256):
            x=torch.randn((m,k),device=flush.device,dtype=torch.bfloat16)
            data={'x':x,'config':{'layer':layer,'n':n,'k':k,'m':m}}
            expected=x.double() @ weights[layer].double().T
            base_graph,base_out=_graph(lambda: _small(data,state))
            max_tiles=(n//64)*((m+63)//64)
            scratch=(torch.empty(m*n*17,device=x.device,dtype=torch.float32),
                     torch.zeros(max_tiles,device=x.device,dtype=torch.uint32))
            trials=[]
            configs=list(_variants(m,n,k,props.multi_processor_count))
            if layer not in ('router','qkv_gate'):
                codes=state['mod'].lt_options(state[layer],x,base_out)
                state['lt_algorithms'][(layer,m,n,k)]=codes
                configs += [('lt',i) for i in range(codes.shape[0])]
                nn_codes=state['mod'].lt_nn_options(state['lt_transposed'][layer],x,base_out)
                state['lt_nn_algorithms'][(layer,m,n,k)]=nn_codes
                configs += [('lt_nn',i) for i in range(nn_codes.shape[0])]
            for config in configs:
                if config[0] not in ('lt','lt_nn'):
                    bt,bk,st,mode,split,threads=config
                    smem=_shared_bytes(layer,bt,bk,st)
                    if smem>props.shared_memory_per_block_optin:
                        continue
                g,out=_graph(lambda c=config: _launch_hop(x,state,layer,n,c,scratch))
                # Catch pipeline/merge errors in every specialization before choosing by speed.
                error=(out.double()-expected).norm()/expected.norm()
                limit=1e-4 if layer=='router' else 1e-2
                if not bool(torch.isfinite(out).all()) or float(error)>limit:
                    raise RuntimeError(f'projection portfolio wrong output: {layer} {m} {config} error {float(error)}')
                us=_cold_times([g],flush,3)[0]
                trials.append((us,config))
            trials.sort(key=lambda row:row[0])
            contenders=trials[:3]
            finalist_graphs=[]
            finalist_outputs=[]
            for _,config in contenders:
                g,out=_graph(lambda c=config: _launch_hop(x,state,layer,n,c,scratch))
                finalist_graphs.append(g)
                finalist_outputs.append(out)
            paired=_cold_times([base_graph]+finalist_graphs,flush,9)
            winner=min(range(len(paired)),key=paired.__getitem__)
            selected=contenders[winner-1][1] if winner else None
            if selected is not None:
                state['choices'][(layer,m,n,k)]=selected
                state['scratch'][(layer,m,n,k)]=scratch if selected[0] not in ('lt','lt_nn') else None
            print(f'FUSED_SELECT {layer} m={m} incumbent_us={paired[0]:.3f} chosen={selected} chosen_us={paired[winner]:.3f} top={trials[:5]}',flush=True)
    print(f'FUSED_SELECT prepare_s={time.monotonic()-started:.2f}',flush=True)
    return state

def kernel_fn(data: Inputs,state)->torch.Tensor:
    x=data['x']; layer=data['config']['layer']; n=data['config']['n']
    key=(layer,x.shape[0],n,x.shape[1])
    config=state['choices'].get(key)
    if config is not None:
        return _launch_hop(x,state,layer,n,config,state['scratch'][key])
    return _small(data,state)

FUSED_CONFIGS = {('router', 128): [(64, 64, 4, 1, 16, 160), (64, 64, 4, 1, 16, 128), (64, 64, 4, 1, 8, 160), (128, 64, 3, 3, 32, 160)], ('router', 256): [(64, 64, 4, 1, 8, 128), (64, 64, 4, 1, 8, 160), (64, 64, 4, 1, 8, 256), (256, 64, 2, 3, 32, 160)], ('qkv_gate', 128): [(64, 64, 4, 1, 2, 160), (64, 64, 4, 1, 2, 256), (128, 64, 3, 1, 4, 160)], ('qkv_gate', 256): [(128, 64, 4, 1, 2, 256), (128, 128, 2, 1, 2, 256), (128, 64, 4, 1, 2, 160)]}

# Embedded in the standalone candidate by build_fused_lean.py.
class _LeanModule:
    def __init__(self,path):
        import ctypes as ct
        self.ct=ct;self.lib=ct.CDLL(path)
        P,I,L=ct.c_void_p,ct.c_int,ct.c_int64
        self.lib.lean_error_string.restype=ct.c_char_p
        self.lib.lean_algo_bytes.restype=I
        self.lib.lean_skinny.argtypes=[P]*4+[I]*12+[P]
        self.lib.lean_pair.argtypes=[P]*5+[I]*11+[L,L,P]
        self.lib.lean_lt_options.argtypes=[I]*5+[P,ct.POINTER(I)]
        self.lib.lean_lt_run.argtypes=[P]*5+[I]*6+[ct.c_size_t,P]
        self.algo_bytes=self.lib.lean_algo_bytes()

    def _check(self,status,context):
        if status:
            error=self.lib.lean_error_string().decode()
            raise RuntimeError(f'{context}: native status {status}: {error}')

    def skinny(self,w,lo,x,out,n,nt,tr,u,wn,wk,ks,split2,xp,ld):
        stream=torch.cuda.current_stream(x.device).cuda_stream
        status=self.lib.lean_skinny(w.data_ptr(),lo.data_ptr(),x.data_ptr(),out.data_ptr(),
                                   x.shape[0],n,x.shape[1],nt,tr,u,wn,wk,ks,int(split2),xp,ld,stream)
        self._check(status,'skinny')

    def hopper(self,w,x,out,work,counts,bt,bk,st,mode,split,threads):
        stream=torch.cuda.current_stream(x.device).cuda_stream
        status=self.lib.lean_pair(w.data_ptr(),x.data_ptr(),out.data_ptr(),work.data_ptr(),counts.data_ptr(),
                                 x.shape[0],out.shape[1],x.shape[1],w.shape[1],int(out.dtype==torch.float32),
                                 bt,bk,st,mode,split,threads,work.numel(),counts.numel(),stream)
        self._check(status,f'pair {(bt,bk,st,mode,split,threads)}')

    def _options(self,w,x,out,nn):
        count=self.ct.c_int()
        codes=torch.empty((32,self.algo_bytes),dtype=torch.uint8,device='cpu')
        self._check(self.lib.lean_lt_options(out.shape[1],x.shape[0],x.shape[1],w.stride(0),nn,codes.data_ptr(),self.ct.byref(count)),'Lt options')
        return codes[:count.value]

    def lt_options(self,w,x,out):
        return self._options(w,x,out,0)

    def lt_nn_options(self,w,x,out):
        return self._options(w,x,out,1)

    def _lt_run(self,w,x,out,work,codes,index,nn):
        if not 0<=index<codes.shape[0]:raise RuntimeError('Lt algorithm index outside prepared table')
        stream=torch.cuda.current_stream(x.device).cuda_stream
        status=self.lib.lean_lt_run(w.data_ptr(),x.data_ptr(),out.data_ptr(),work.data_ptr(),codes.data_ptr(),
                                   out.shape[1],x.shape[0],x.shape[1],w.stride(0),nn,index,work.numel()*work.element_size(),stream)
        self._check(status,'Lt matmul')

    def lt_run(self,w,x,out,work,codes,index):
        self._lt_run(w,x,out,work,codes,index,0)

    def lt_nn_run(self,w,x,out,work,codes,index):
        self._lt_run(w,x,out,work,codes,index,1)


def _module():
    global _MOD
    if _MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        arch='9.0a' if _hopper() else '12.0'
        old=os.environ.get('TORCH_CUDA_ARCH_LIST')
        os.environ['TORCH_CUDA_ARCH_LIST']=arch
        tag=hashlib.sha1(RAW_CUDA.encode()).hexdigest()[:10]
        try:
            path=load_inline(name=f'muse_proj_lean_{tag}_{arch.replace(chr(46),chr(95))}',
                             cpp_sources='',cuda_sources=RAW_CUDA,functions=None,
                             is_python_module=False,no_implicit_headers=True,
                             extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','-Xptxas=-v'],
                             extra_ldflags=['-lcublasLt'],verbose=False)
        finally:
            if old is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:os.environ['TORCH_CUDA_ARCH_LIST']=old
        _MOD=_LeanModule(path)
    return _MOD

RAW_CUDA = r'''// Lightweight host ABI for the unchanged CUDA kernels. No Torch C++ headers.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cublasLt.h>
#include <stdint.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
namespace cg=cooperative_groups;
static thread_local char lean_error[512];
extern "C" const char* lean_error_string(){return lean_error;}
static int lean_fail(const char* where,int code){std::snprintf(lean_error,sizeof(lean_error),"%s: status %d",where,code);return code ? code : -1;}
#define LEAN_CUDA(call) do{auto e=(call);if(e!=cudaSuccess)return lean_fail(#call,(int)e);}while(0)
#define LEAN_LT(call) do{auto e=(call);if(e!=CUBLAS_STATUS_SUCCESS)return lean_fail(#call,(int)e);}while(0)



#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>



namespace muse_proj {

// Weight stream: read once, keep out of L1. LD 0: 256B L2 prefetch, 1: none, 2: 128B.
template <int LD>
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  if (LD == 0)
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else if (LD == 1)
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else
    asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Activations: shared by the CTA's warps through L1.
__device__ __forceinline__ uint4 ld_act(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One CTA: BR = WN*TR*16 weight rows x BT = 8*NT tokens x KB of K (its cluster rank's slice).
// Warp (wn, wk): TR row tiles of 16 x the wk-th 1/WK of the CTA's K slice, in steps of 32*U.
// Lane (g = lane/4, t = lane%4) loads 16 bytes at k = k0 + 8t of rows g and g+8 (and of
// token g for activations); the first mma takes k-slots {8t..8t+3}, the second {8t+4..8t+7}.
// NA = 2: a second weight matrix (the router's lo half) at the same offsets, same accumulators.
// XP: activations prefetched one batch ahead with the weights (else loaded at compute time).
template <int NT, int TR, int U, int NA, int XP, int LD>
__global__ void __launch_bounds__(256) skinny_kernel(
    const __nv_bfloat16* __restrict__ W0, const __nv_bfloat16* __restrict__ W1,
    const __nv_bfloat16* __restrict__ X, void* __restrict__ Out, int M, int N, int K, int KB,
    int WN, int WK) {
  constexpr bool OUTF32 = NA == 2;
  constexpr int XU = XP ? U : 1;
  constexpr int XN = XP ? NT : 1;
  extern __shared__ float sred[];
  constexpr int BT = 8 * NT;
  const int BR = WN * TR * 16;
  const int S = BR + 4;  // padded token stride of the reduction tile (bank spread)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wn = warp % WN, wk = warp / WN;
  const int row_cta = blockIdx.x * BR;
  const int rowbase = row_cta + wn * TR * 16;
  const int tok0 = blockIdx.z * BT;
  const int KW = KB / WK;
  const int kbeg = blockIdx.y * KB + wk * KW;
  const int nsteps = KW / (32 * U);

  float acc[TR][NT][4];
#pragma unroll
  for (int r = 0; r < TR; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.f;

  const size_t off = (size_t)(rowbase + g) * K + kbeg + 8 * t;
  const __nv_bfloat16* wp0 = W0 + off;
  const __nv_bfloat16* wp1 = W1 + off;
  const size_t rs8 = (size_t)8 * K;
  const size_t rs16 = (size_t)16 * K;
  const __nv_bfloat16* xp[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    int tok = tok0 + j * 8 + g;
    tok = tok < M ? tok : M - 1;  // padded tokens read a real row; their outputs are dropped
    xp[j] = X + (size_t)tok * K + kbeg + 8 * t;
  }

  uint4 a0[U][NA][TR][2], a1[U][NA][TR][2];
  uint4 b0[XU][XN], b1[XU][XN];

  auto load = [&](uint4 (&a)[U][NA][TR][2], uint4 (&b)[XU][XN], int s) {
    const int koff = s * 32 * U;
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int na = 0; na < NA; ++na)
#pragma unroll
        for (int r = 0; r < TR; ++r) {
          const __nv_bfloat16* p = (na == 0 ? wp0 : wp1) + r * rs16 + koff + u * 32;
          a[u][na][r][0] = ld_stream<LD>(p);
          a[u][na][r][1] = ld_stream<LD>(p + rs8);
        }
    if (XP) {
#pragma unroll
      for (int u = 0; u < XU; ++u)
#pragma unroll
        for (int j = 0; j < XN; ++j) b[u][j] = ld_act(xp[j] + koff + u * 32);
    }
  };
  auto compute = [&](uint4 (&a)[U][NA][TR][2], uint4 (&bp)[XU][XN], int s) {
    const int koff = s * 32 * U;
    uint4 bl[U][NT];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        if (XP)
          bl[u][j] = bp[u < XU ? u : 0][j < XN ? j : 0];
        else
          bl[u][j] = ld_act(xp[j] + koff + u * 32);
      }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int na = 0; na < NA; ++na)
#pragma unroll
        for (int r = 0; r < TR; ++r)
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const uint4& lo = a[u][na][r][0];
            const uint4& hi = a[u][na][r][1];
            mma16816(acc[r][j], lo.x, hi.x, lo.y, hi.y, bl[u][j].x, bl[u][j].y);
            mma16816(acc[r][j], lo.z, hi.z, lo.w, hi.w, bl[u][j].z, bl[u][j].w);
          }
  };

  if (nsteps > 0) load(a0, b0, 0);
  for (int s = 0; s < nsteps; s += 2) {
    if (s + 1 < nsteps) load(a1, b1, s + 1);
    compute(a0, b0, s);
    if (s + 1 < nsteps) {
      if (s + 2 < nsteps) load(a0, b0, s + 2);
      compute(a1, b1, s + 1);
    }
  }

  // Warp partials -> shared tile [WK][BT][S] (token-major so the output stores coalesce).
#pragma unroll
  for (int r = 0; r < TR; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int row = wn * TR * 16 + r * 16 + g;
      const int tok = j * 8 + 2 * t;
      float* p = sred + (size_t)(wk * BT + tok) * S + row;
      p[0] = acc[r][j][0];
      p[S] = acc[r][j][1];
      p[8] = acc[r][j][2];
      p[S + 8] = acc[r][j][3];
    }
  __syncthreads();
  const int E = BT * BR;
  if (WK > 1) {
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
      const int tok = e / BR, row = e - tok * BR;
      float v = 0.f;
      for (int w = 0; w < WK; ++w) v += sred[(w * BT + tok) * S + row];
      sred[tok * S + row] = v;
    }
    __syncthreads();
  }

  auto store = [&](int tok, int row, float v) {
    const size_t idx = (size_t)(tok0 + tok) * N + row_cta + row;
    if (OUTF32)
      reinterpret_cast<float*>(Out)[idx] = v;
    else
      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
  };

  const int KS = gridDim.y;
  if (KS == 1) {
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
      const int tok = e / BR, row = e - tok * BR;
      if (tok0 + tok < M) store(tok, row, sred[tok * S + row]);
    }
  } else {
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    const int rank = (int)cluster.block_rank();
    const int per = (E + KS - 1) / KS;
    const int e0 = rank * per;
    const int e1 = min(E, e0 + per);
    for (int e = e0 + (int)threadIdx.x; e < e1; e += blockDim.x) {
      const int tok = e / BR, row = e - tok * BR;
      if (tok0 + tok >= M) continue;
      float v = 0.f;
      for (int q = 0; q < KS; ++q) v += cluster.map_shared_rank(sred, q)[tok * S + row];
      store(tok, row, v);
    }
    cluster.sync();  // keep this CTA's shared tile alive until every rank has read it
  }
}


} // namespace muse_proj



#include <cuda.h>
#include <cudaTypedefs.h>
#ifndef MP_SPIN_LIMIT
#define MP_SPIN_LIMIT (1u << 26)
#endif
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define MP_SM90A 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define MP_TMA 1  // TMA, mbarriers, clusters: sm_90 and later (wgmma itself needs sm_90a)
#endif

namespace muse_pair {

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

#ifdef MP_TMA
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su32(bar)) : "memory");
}
// Bounded: a pipeline bug yields wrong numbers (caught by tests) instead of a hung GPU.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t done = 0;
  uint32_t tries = 0;
  do {
    if (++tries > (uint32_t)MP_SPIN_LIMIT) {
#ifdef MP_DEBUG
      printf("mbar_wait timeout block (%d,%d,%d) thread %d parity %u\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, parity);
#endif
      asm volatile("trap;");
    }
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done) : "r"(su32(bar)), "r"(parity) : "memory");
  } while (!done);
}
// Same box into this CTA and every CTA in mask (same smem offsets, each one's own mbarrier).
__device__ __forceinline__ void tma_load_2d_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                               uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(su32(bar)), "r"(c0), "r"(c1), "h"(mask)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(uint64_t* bar, uint32_t cta_rank) {
  asm volatile(
      "{\n .reg .b32 ra;\n mapa.shared::cluster.u32 ra, %0, %1;\n"
      " mbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\n}\n"
      ::"r"(su32(bar)), "r"(cta_rank) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(su32(bar)), "r"(c0), "r"(c1) : "memory");
}
template <int N_>
__device__ __forceinline__ void acc_fence(float (&d)[N_]) {
#pragma unroll
  for (int i = 0; i < N_; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
#endif
#ifdef MP_SM90A
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N_>
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N_) : "memory"); }
// K-major operand, 128-byte swizzle, 8-row atoms 1024 bytes apart.
__device__ __forceinline__ uint64_t make_desc(const void* p, int bk) {
  const uint64_t a = su32(p);
  return ((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wgmma_m64n64(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n128(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n256(float (&d)[128], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "l"(da), "l"(db), "r"(scale_d));
}

#elif defined(MP_TMA)
// Off sm_90a: the same swizzled tiles through ldmatrix + mma.sync (validates the pipeline).
__device__ __forceinline__ void ldsm_x4(const void* p, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(su32(p)));
}
__device__ __forceinline__ void ldsm_x2(const void* p, uint32_t& b0, uint32_t& b1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(su32(p)));
}
__device__ __forceinline__ void mma_emu(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif


// Each block owns a contiguous K interval. No CTA waits for another CTA to be scheduled.
// Mode 0: split-K + last-CTA merge; mode 1: DSMEM cluster; mode 2: balanced Stream-K.
template <int BT, int BK, int ST, bool OUTF32, int THREADS, bool HIER>
__global__ void __launch_bounds__(THREADS) hop_kernel(
    const __grid_constant__ CUtensorMap tmW, const __grid_constant__ CUtensorMap tmX,
    void* __restrict__ Out, volatile float* __restrict__ Work, unsigned* __restrict__ Counts,
    int M, int N, int nkb, int kx, int mode, int split, int chunk, int total) {
#ifdef MP_TMA
  // A 128-byte TMA swizzle admits at most 64 BF16 elements in its inner box.
  // BK=128 groups two legal K64 boxes under one pipeline barrier.
  constexpr int BR = 64, AB = BR * 64 * 2, XB = AB * (OUTF32 ? 2 : 1);
  constexpr int CB = (BR * (OUTF32 ? 2 : 1) + BT) * 64 * 2, SB = CB * (BK/64);
  extern __shared__ __align__(1024) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * SB);
  uint64_t* empty = full + ST;
  __shared__ unsigned last;
  const int tid = threadIdx.x, consumer = tid < 128;
  const int begin = mode == 2 ? blockIdx.x * chunk : 0;
  const int end = mode == 2 ? min(total, begin + chunk) : 1;
  int pos = begin;
  while (pos < end) {
    const int tile = mode == 2 ? pos / nkb : (int)blockIdx.x;
    const int first = mode == 2 ? tile * nkb / chunk : 0;
    const int parts = mode == 2 ? ((tile + 1) * nkb - 1) / chunk - first + 1 : split;
    const int slot = mode == 2 ? (int)blockIdx.x - first : (int)blockIdx.y;
    const int kb0 = mode == 2 ? pos - tile * nkb : slot * nkb / split;
    const int kb1 = mode == 2 ? min(nkb, end - tile * nkb) : (slot + 1) * nkb / split;
    const int steps = kb1 - kb0;
    const int row0 = (tile % (N / BR)) * BR, tok0 = (tile / (N / BR)) * BT;
    if (tid == 0) {
      for (int s = 0; s < ST; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    float acc[BT / 2];
#pragma unroll
    for (int i = 0; i < BT / 2; ++i) acc[i] = 0.f;
    // One warpgroup owns both the TMA ring and WGMMA; no idle producer warpgroup.
    auto prefetch = [&](int s) {
      int stage=s%ST, kc=(kb0+s)*BK;
      unsigned char* a=smem+stage*SB;
      mbar_expect_tx(full+stage,SB);
#pragma unroll
      for (int chunk_k=0;chunk_k<BK/64;++chunk_k) {
        tma_load_2d(a+chunk_k*CB,&tmW,full+stage,kc+chunk_k*64,row0);
        if constexpr(OUTF32) tma_load_2d(a+chunk_k*CB+AB,&tmW,full+stage,kc+chunk_k*64+kx,row0);
        tma_load_2d(a+chunk_k*CB+XB,&tmX,full+stage,kc+chunk_k*64,tok0);
      }
    };
    if constexpr(THREADS==128) { if(tid==0) for(int s=0;s<ST && s<steps;++s) prefetch(s); }
    if (THREADS>128 && tid == 128) {
      for (int s = 0; s < steps; ++s) {
        int stage = s % ST;
        if (s >= ST) mbar_wait(empty + stage, ((s / ST) & 1) ^ 1);
        unsigned char* a = smem + stage * SB;
        mbar_expect_tx(full + stage, SB);
        int kc = (kb0 + s) * BK;
#pragma unroll
        for (int chunk_k=0; chunk_k<BK/64; ++chunk_k) {
          tma_load_2d(a+chunk_k*CB, &tmW, full + stage, kc+chunk_k*64, row0);
          if constexpr (OUTF32) tma_load_2d(a+chunk_k*CB+AB, &tmW, full + stage, kc+chunk_k*64+kx, row0);
          tma_load_2d(a+chunk_k*CB+XB, &tmX, full + stage, kc+chunk_k*64, tok0);
        }
      }
    } else if (consumer) {
      for (int s = 0; s < steps; ++s) {
        int stage = s % ST;
        mbar_wait(full + stage, (s / ST) & 1);
        unsigned char* a = smem + stage * SB;
#ifdef MP_SM90A
        acc_fence(acc);
        wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
          const uint64_t db = make_desc(a+(kk/4)*CB+XB,64);
          { const uint64_t da = make_desc(a+(kk/4)*CB,64); if constexpr(BT==64) wgmma_m64n64(acc,da+2*(kk%4),db+2*(kk%4),1); else if constexpr(BT==128) wgmma_m64n128(acc,da+2*(kk%4),db+2*(kk%4),1); else if constexpr(BT==256) wgmma_m64n256(acc,da+2*(kk%4),db+2*(kk%4),1); }
          if constexpr (OUTF32) { const uint64_t da = make_desc(a+(kk/4)*CB+AB,64); if constexpr(BT==64) wgmma_m64n64(acc,da+2*(kk%4),db+2*(kk%4),1); else if constexpr(BT==128) wgmma_m64n128(acc,da+2*(kk%4),db+2*(kk%4),1); else if constexpr(BT==256) wgmma_m64n256(acc,da+2*(kk%4),db+2*(kk%4),1); }
        }
        wgmma_commit();
        wgmma_wait<0>();
        acc_fence(acc);
#else
        const int warp = tid >> 5, ln = tid & 31;
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
          unsigned char* chunk_a=a+(kk/4)*CB;
          int ar = warp * 16 + (ln & 15), ac = (kk%4) * 2 + (ln >> 4);
          uint32_t a0,a1,a2,a3,lo0,lo1,lo2,lo3;
          ldsm_x4(chunk_a + ar * 128 + ((ac ^ (ar & 7)) << 4), a0,a1,a2,a3);
          if constexpr (OUTF32) ldsm_x4(chunk_a + AB + ar * 128 + ((ac ^ (ar & 7)) << 4),lo0,lo1,lo2,lo3);
#pragma unroll
          for (int j = 0; j < BT / 8; ++j) {
            int br = j * 8 + (ln & 7), bc = (kk%4) * 2 + ((ln >> 3) & 1);
            uint32_t b0,b1;
            ldsm_x2(chunk_a + XB + br * 128 + ((bc ^ (br & 7)) << 4), b0,b1);
            mma_emu(&acc[4*j],a0,a1,a2,a3,b0,b1);
            if constexpr (OUTF32) mma_emu(&acc[4*j],lo0,lo1,lo2,lo3,b0,b1);
          }
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
#endif
        if constexpr(THREADS==128) {
          if(tid==0 && s+ST<steps) prefetch(s+ST);
        } else {
          if (tid == 0) mbar_arrive(empty + stage);
        }
      }
    }
    __syncthreads();
    auto store = [&](int token, int row, float v) {
      if (tok0 + token < M) {
        size_t idx = (size_t)(tok0 + token) * N + row0 + row;
        if constexpr (OUTF32) reinterpret_cast<float*>(Out)[idx] = v;
        else reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);
      }
    };
    // Same WGMMA accumulator layout as mma.sync emulation.
    if (consumer) {
      int warp = tid >> 5, lane = tid & 31, row = warp * 16 + (lane >> 2), col = 2 * (lane & 3);
#pragma unroll
      for (int j = 0; j < BT / 8; ++j) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = row + (q >= 2 ? 8 : 0), t = j * 8 + col + (q & 1);
          if (parts == 1) store(t, r, acc[j*4+q]);
          else if (mode == 1 || mode == 3) reinterpret_cast<float*>(smem)[t * 68 + r] = acc[j*4+q];
          else Work[((size_t)tile * 17 + slot) * BT * BR + t * BR + r] = acc[j*4+q];
        }
      }
    }
    if (parts > 1) {
      if constexpr (HIER) {
        // Hierarchical split-K: each 16-CTA cluster reduces its K interval in DSMEM.
        // Only the last completed cluster merges the 2/4 global partials, distributed
        // over all 16 resident CTAs. There is no cross-cluster spin or second kernel.
        constexpr int CK=16;
        auto cluster=cg::this_cluster();
        const int rank=slot%CK, group=slot/CK, groups=parts/CK;
        cluster.sync();
        for(int e=rank*THREADS+tid;e<BT*BR;e+=CK*THREADS) {
          int t=e/BR,r=e%BR; float v=0.f;
          for(int p=0;p<CK;++p) v+=cluster.map_shared_rank(reinterpret_cast<float*>(smem),p)[t*68+r];
          Work[((size_t)tile*17+group)*BT*BR+e]=v;
        }
        __threadfence();
        cluster.sync();
        if(rank==0 && tid==0) last=atomicAdd(Counts+tile,1u)==(unsigned)(groups-1);
        cluster.sync();
        const bool elected=*cluster.map_shared_rank(&last,0)!=0;
        if(elected) {
          for(int e=rank*THREADS+tid;e<BT*BR;e+=CK*THREADS) {
            float v=0.f;
            for(int p=0;p<groups;++p) v+=Work[((size_t)tile*17+p)*BT*BR+e];
            store(e/BR,e%BR,v);
          }
        }
        cluster.sync();
        if(elected && rank==0 && tid==0) atomicExch(Counts+tile,0u);
      } else if (mode == 1) {
        auto cluster = cg::this_cluster();
        cluster.sync();
        for (int e = slot * THREADS + tid; e < BT * BR; e += parts * THREADS) {
          int t = e / BR, r = e % BR;
          float v = 0.f;
          for (int p = 0; p < parts; ++p) v += cluster.map_shared_rank(reinterpret_cast<float*>(smem), p)[t*68+r];
          store(t,r,v);
        }
        cluster.sync();
      } else {
        // Release each writer's global stores before publishing this block's ticket.
        __threadfence();
        __syncthreads();
        if (tid == 0) last = atomicAdd(Counts + tile, 1u) == (unsigned)(parts - 1);
        __syncthreads();
        if (last) {
          for (int e = tid; e < BT * BR; e += THREADS) {
            float v = 0.f;
            for (int p = 0; p < parts; ++p) v += Work[((size_t)tile*17+p)*BT*BR+e];
            store(e/BR,e%BR,v);
          }
          __syncthreads();
          if (tid == 0) atomicExch(Counts + tile, 0u);
        }
      }
    }
    __syncthreads();
    pos = mode == 2 ? tile * nkb + kb1 : end;
  }
#endif
}


} // namespace muse_pair


template<int NT,int TR,int U,int NA,int XP,int LD>
int lean_skinny_launch(const __nv_bfloat16* w,const __nv_bfloat16* lo,const __nv_bfloat16* x,void* out,
                      int M,int N,int K,int wn,int wk,int ks,cudaStream_t stream){
  constexpr int BT=NT*8;int BR=wn*TR*16;
  if(N%BR || ks<1 || K%ks || (K/ks)%(wk*32*U) || wn*wk>8)return lean_fail("skinny shape",-1);
  size_t shared=(size_t)wk*BT*(BR+4)*4;if(shared>48*1024)return lean_fail("skinny shared",-1);
  cudaLaunchConfig_t c={};c.gridDim=dim3(N/BR,ks,(M+BT-1)/BT);c.blockDim=dim3(wn*wk*32);c.dynamicSmemBytes=shared;c.stream=stream;
  cudaLaunchAttribute attr={};if(ks>1){attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=1;attr.val.clusterDim.y=ks;attr.val.clusterDim.z=1;c.attrs=&attr;c.numAttrs=1;}
  return (int)cudaLaunchKernelEx(&c,muse_proj::skinny_kernel<NT,TR,U,NA,XP,LD>,w,lo,x,out,M,N,K,K/ks,wn,wk);
}
extern "C" int lean_skinny(const void* w,const void* lo,const void* x,void* out,
                          int M,int N,int K,int nt,int tr,int u,int wn,int wk,int ks,int split2,int xp,int ld,void* stream){
  int na=split2 ? 2 : 1;
#define LSKIN(NT,TR,U,NA,XP,LD) if(nt==NT && tr==TR && u==U && na==NA && xp==XP && ld==LD)return lean_skinny_launch<NT,TR,U,NA,XP,LD>((const __nv_bfloat16*)w,(const __nv_bfloat16*)lo,(const __nv_bfloat16*)x,out,M,N,K,wn,wk,ks,(cudaStream_t)stream);
  LSKIN(1,1,4,1,1,0) LSKIN(1,2,4,1,1,0) LSKIN(2,1,4,1,1,0) LSKIN(2,2,4,1,1,0) LSKIN(4,1,4,1,1,0) LSKIN(8,1,2,1,1,0) LSKIN(1,1,4,2,1,0) LSKIN(2,1,4,2,1,0) LSKIN(4,1,4,2,0,0)
  return lean_fail("skinny specialization",-1);
}

static int lean_map(CUtensorMap* m,const void* ptr,int rows,int cols,int box_rows){
  static PFN_cuTensorMapEncodeTiled_v12000 fn=nullptr;
  if(!fn){void* p=nullptr;cudaDriverEntryPointQueryResult q;LEAN_CUDA(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&p,cudaEnableDefault,&q));if(!p)return lean_fail("tensor map function",-1);fn=(PFN_cuTensorMapEncodeTiled_v12000)p;}
  cuuint64_t dims[2]={(cuuint64_t)cols,(cuuint64_t)rows},strides[1]={(cuuint64_t)cols*2};
  cuuint32_t box[2]={64,(cuuint32_t)box_rows},es[2]={1,1};
  auto e=fn(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,const_cast<void*>(ptr),dims,strides,box,es,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return e==CUDA_SUCCESS ? 0 : lean_fail("tensor map encode",(int)e);
}
template<int BT,int BK,int ST,int THREADS,bool OF,bool HH>
int lean_pair_launch(const void* w,const void* x,void* out,float* work,unsigned* counts,
                     int M,int N,int K,int KW,int mode,int split,int64_t work_elems,int64_t count_elems,cudaStream_t stream){
  if(N%64 || K%BK || KW%BK || K<1 || split<1)return lean_fail("pair shape",-1);
  int nkb=K/BK,tiles=(N/64)*((M+BT-1)/BT),total=tiles*nkb;
  if(mode!=2 && (split>64 || split>nkb || (mode==1 && split>16) || (mode==3 && split%16)))return lean_fail("pair split",-1);
  int target=std::min(split,tiles*16),chunk=(total+target-1)/target,blocks=mode==2 ? (total+chunk-1)/chunk : tiles;
  if(work_elems<(int64_t)tiles*17*BT*64 || count_elems<tiles)return lean_fail("pair workspace",-1);
  auto kern=muse_pair::hop_kernel<BT,BK,ST,OF,THREADS,HH>;
  size_t shared=std::max((size_t)ST*(64*(OF ? 2 : 1)+BT)*BK*2+ST*16,(size_t)BT*68*4);
  int device=0,optin=0;LEAN_CUDA(cudaGetDevice(&device));LEAN_CUDA(cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
  if(shared>(size_t)optin)return lean_fail("pair shared memory",-1);
  static bool initialized=false;if(!initialized){LEAN_CUDA(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));LEAN_CUDA(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));initialized=true;}
  CUtensorMap mw,mx;int err=lean_map(&mw,w,N,KW,64);if(err)return err;err=lean_map(&mx,x,M,K,BT);if(err)return err;
  cudaLaunchConfig_t c={};c.gridDim=dim3(blocks,mode==2 ? 1 : split);c.blockDim=dim3(THREADS);c.dynamicSmemBytes=shared;c.stream=stream;
  cudaLaunchAttribute attr={};if((mode==1 || mode==3) && split>1){attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=1;attr.val.clusterDim.y=mode==3 ? 16 : split;attr.val.clusterDim.z=1;c.attrs=&attr;c.numAttrs=1;}
  auto e=cudaLaunchKernelEx(&c,kern,mw,mx,out,work,counts,M,N,nkb,K,mode,split,chunk,total);
  return e==cudaSuccess ? 0 : lean_fail("pair launch",(int)e);
}
extern "C" int lean_pair(const void* w,const void* x,void* out,void* work,void* counts,
                        int M,int N,int K,int KW,int of,int bt,int bk,int st,int mode,int split,int threads,
                        int64_t work_elems,int64_t count_elems,void* stream){
#define LPAIR(BT,BK,ST,TH,OF,HH) if(bt==BT && bk==BK && st==ST && threads==TH && of==OF && (mode==3)==HH)return lean_pair_launch<BT,BK,ST,TH,OF,HH>(w,x,out,(float*)work,(unsigned*)counts,M,N,K,KW,mode,split,work_elems,count_elems,(cudaStream_t)stream);
  LPAIR(64,64,4,128,true,false) LPAIR(64,64,4,160,false,false) LPAIR(64,64,4,160,true,false) LPAIR(64,64,4,256,false,false) LPAIR(64,64,4,256,true,false) LPAIR(128,64,3,160,false,false) LPAIR(128,64,3,160,true,true) LPAIR(128,64,4,160,false,false) LPAIR(128,64,4,256,false,false) LPAIR(128,128,2,256,false,false) LPAIR(256,64,2,160,true,true)
  return lean_fail("pair specialization",-1);
}

struct LeanLayout{
  cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  int init(int n,int m,int k,int stride,int nn){
    LEAN_LT(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;
    LEAN_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    LEAN_LT(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,nn ? n : k,nn ? k : n,stride));
    LEAN_LT(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,k));LEAN_LT(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n));return 0;
  }
  ~LeanLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
};
static cublasLtHandle_t lean_lt_handle=nullptr;
static int lean_lt_init(){if(!lean_lt_handle)LEAN_LT(cublasLtCreate(&lean_lt_handle));return 0;}
extern "C" int lean_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}
extern "C" int lean_lt_options(int n,int m,int k,int stride,int nn,void* codes,int* count){
  int err=lean_lt_init();if(err)return err;LeanLayout l;err=l.init(n,m,k,stride,nn);if(err)return err;
  cublasLtMatmulPreference_t pref;LEAN_LT(cublasLtMatmulPreferenceCreate(&pref));size_t bytes=32*1024*1024;
  auto status=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes));
  if(status!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("Lt preference",(int)status);}
  cublasLtMatmulHeuristicResult_t rows[32];status=cublasLtMatmulAlgoGetHeuristic(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,pref,32,rows,count);
  cublasLtMatmulPreferenceDestroy(pref);if(status!=CUBLAS_STATUS_SUCCESS)return lean_fail("Lt heuristics",(int)status);
  for(int i=0;i<*count;++i){if(rows[i].state!=CUBLAS_STATUS_SUCCESS || rows[i].workspaceSize>bytes)return lean_fail("Lt heuristic state",-1);std::memcpy((char*)codes+i*sizeof(cublasLtMatmulAlgo_t),&rows[i].algo,sizeof(cublasLtMatmulAlgo_t));}
  return 0;
}
extern "C" int lean_lt_run(const void* w,const void* x,void* out,void* work,const void* codes,
                          int n,int m,int k,int stride,int nn,int index,size_t work_bytes,void* stream){
  if(work_bytes<32*1024*1024)return lean_fail("Lt workspace",-1);
  int err=lean_lt_init();if(err)return err;LeanLayout l;err=l.init(n,m,k,stride,nn);if(err)return err;
  cublasLtMatmulAlgo_t algo;std::memcpy(&algo,(const char*)codes+index*sizeof(algo),sizeof(algo));
  float alpha=1.f,beta=0.f;
  auto status=cublasLtMatmul(lean_lt_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&algo,work,work_bytes,(cudaStream_t)stream);
  return status==CUBLAS_STATUS_SUCCESS ? 0 : lean_fail("Lt matmul",(int)status);
}
'''

import types as _df_types
_df_proj=_df_types.ModuleType("embedded_muse_proj")
exec('"""Local BF16 Muse o/pooler/unpooler GEMMs from H200 receipt f3786341611f4f1f.\n\nConstruct MuseProjection(weight, kind) after weight loading, before CUDA graph\ncapture. Pass the exact local weight, including a TP row slice when applicable.\nCalls return only this rank\'s GEMM result: the caller keeps its all-reduce,\nall-gather, residual and norm operations. No model parameters are replaced.\n\nSmall-M schedules and 32 MiB cuBLASLt TN/NN selection match the receipt. New\n512-row pooler / 1024-row unpooler shapes tune their actual dimensions; they\nhave no standalone H200 speedup evidence. Plans are cached by shape/device,\nthen checked against torch for each new weight. Unseen M uses the ordinary\nGEMM or the masked small-M schedule, without online tuning or host sync.\n"""\n\nfrom __future__ import annotations\n\nimport ctypes as ct\nfrom functools import lru_cache\nfrom itertools import chain\nimport hashlib\nimport logging\nimport os\nimport statistics\nimport time\n\nimport torch\nimport torch.nn.functional as F\n\nlogger = logging.getLogger(__name__)\nVERIFY_TOKENS = (8, 16, 32, 64, 128, 256)\nWORKSPACE_BYTES = 32 << 20\n_SHAPES = {\n    "o": ((8192, 1024),),\n    "pooler": ((4096, 8192), (512, 8192)),\n    "unpooler": ((8192, 4096), (1024, 4096)),\n}\n# Ordinals refer to 32 returned heuristics and a 32 MiB workspace, exactly as\n# in f3786341611f4f1f. They compete with cuBLAS and every returned heuristic.\n_RECEIPT = {\n    ("pooler", 128): ("lt_nn", 1),\n    ("pooler", 256): ("lt_nn", 0),\n    ("unpooler", 128): ("lt_nn", 0),\n    ("unpooler", 256): ("lt", 0),\n    ("o", 128): ("lt_nn", 0),\n    ("o", 256): ("cublas",),\n}\n_PLAN_CACHE = {}\n\n\ndef projection_supported(x, weight, kind):\n    """H200 integration gate; a direct instance also runs native sm_120 tests."""\n    return (\n        kind in _SHAPES\n        and x.is_cuda\n        and weight.is_cuda\n        and x.device == weight.device\n        and x.ndim == weight.ndim == 2\n        and tuple(weight.shape) in _SHAPES[kind]\n        and x.shape[1] == weight.shape[1]\n        and x.dtype == weight.dtype == torch.bfloat16\n        and x.is_contiguous()\n        and weight.is_contiguous()\n        and 0 < x.shape[0] <= 256\n        and torch.cuda.get_device_capability(x.device) == (9, 0)\n    )\n\n\nclass _Module:\n    def __init__(self, path):\n        self.lib = ct.CDLL(path)\n        p, i = ct.c_void_p, ct.c_int\n        self.lib.lean_error_string.restype = ct.c_char_p\n        self.lib.lean_algo_bytes.restype = i\n        self.lib.lean_skinny.argtypes = [p] * 4 + [i] * 12 + [p]\n        self.lib.lean_lt_options.argtypes = [i] * 5 + [p, ct.POINTER(i)]\n        self.lib.lean_lt_run.argtypes = [p] * 5 + [i] * 6 + [ct.c_size_t, p]\n        self.algo_bytes = self.lib.lean_algo_bytes()\n\n    def check(self, status):\n        if status:\n            raise RuntimeError(self.lib.lean_error_string().decode())\n\n    def skinny(self, x, weight, out, config):\n        nt, tr, u, wn, wk, ks, xp = config\n        self.check(self.lib.lean_skinny(\n            weight.data_ptr(), weight.data_ptr(), x.data_ptr(), out.data_ptr(),\n            x.shape[0], weight.shape[0], x.shape[1], nt, tr, u, wn, wk, ks,\n            0, xp, 0, torch.cuda.current_stream(x.device).cuda_stream,\n        ))\n\n    def options(self, m, n, k, nn):\n        count = ct.c_int()\n        codes = torch.empty((32, self.algo_bytes), dtype=torch.uint8, device="cpu")\n        self.check(self.lib.lean_lt_options(\n            n, m, k, n if nn else k, nn, codes.data_ptr(), ct.byref(count),\n        ))\n        return codes[:count.value].clone()\n\n    def lt(self, x, weight, out, workspace, codes, index, nn):\n        if not 0 <= index < codes.shape[0]:\n            raise ValueError("cuBLASLt ordinal outside the prepared heuristic table")\n        self.check(self.lib.lean_lt_run(\n            weight.data_ptr(), x.data_ptr(), out.data_ptr(), workspace.data_ptr(),\n            codes.data_ptr(), out.shape[1], x.shape[0], x.shape[1],\n            weight.stride(0), nn, index, workspace.numel(),\n            torch.cuda.current_stream(x.device).cuda_stream,\n        ))\n\n\n@lru_cache(None)\ndef _module(arch=None):\n    from torch.utils.cpp_extension import load_inline\n\n    if torch.cuda.is_current_stream_capturing():\n        raise RuntimeError("construct MuseProjection after loading, before graph capture")\n    capability = torch.cuda.get_device_capability()\n    target = arch or ("9.0a" if capability == (9, 0) else f"{capability[0]}.{capability[1]}")\n    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")\n    os.environ["TORCH_CUDA_ARCH_LIST"] = target\n    try:\n        tag = hashlib.sha256(_CUDA_SOURCE.encode()).hexdigest()[:12]\n        path = load_inline(\n            name=f"muse_dense_{tag}_{target.replace(\'.\', \'_\')}",\n            cpp_sources="", cuda_sources=_CUDA_SOURCE, functions=None,\n            is_python_module=False, no_implicit_headers=True,\n            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr"],\n            extra_ldflags=["-lcublasLt"], verbose=False,\n        )\n    finally:\n        if previous is None:\n            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)\n        else:\n            os.environ["TORCH_CUDA_ARCH_LIST"] = previous\n    return _Module(path)\n\n\n@lru_cache(None)\ndef _options(device_index, m, n, k, nn):\n    with torch.cuda.device(device_index):\n        return _module().options(m, n, k, nn)\n\n\ndef _version(weight):\n    try:\n        return weight._version\n    except RuntimeError:\n        # SGLang\'s inference tensors may not carry version counters. Weights\n        # must remain immutable after preparation, as with other packed weights.\n        return None\n\n\ndef _agrees(out, expected):\n    if out.shape != expected.shape or not bool(torch.isfinite(out).all()):\n        return False\n    delta, ref = out.float() - expected.float(), expected.float()\n    if float(delta.norm() / ref.norm().clamp_min(1e-12)) > .01:\n        return False\n    norms = ref.norm(dim=1)\n    return float((delta.norm(dim=1) / norms.clamp_min(norms.mean() * .1 + 1e-12)).max()) <= .05\n\n\nclass _Graph:\n    """Own its stream: destruction cannot clear another graph\'s BLAS workspace."""\n    active_streams = 0\n\n    def __init__(self, fn):\n        self.runtime = torch.cuda.cudart()\n        handle = ct.c_void_p()\n        if int(self.runtime.cudaStreamCreate(ct.addressof(handle))):\n            raise RuntimeError("projection tuning stream creation failed")\n        self.handle = handle.value\n        type(self).active_streams += 1\n        self.stream = torch.cuda.ExternalStream(self.handle)\n        self.graph = torch.cuda.CUDAGraph()\n        try:\n            self.stream.wait_stream(torch.cuda.current_stream())\n            with torch.cuda.stream(self.stream):\n                fn()\n            self.stream.synchronize()\n            with torch.cuda.graph(self.graph, stream=self.stream):\n                self.out = fn()\n            self.graph.replay()\n        except BaseException:\n            self.close()\n            raise\n\n    def close(self):\n        if getattr(self, "handle", None) is not None:\n            self.graph.reset()\n            self.graph = self.out = None\n            self.runtime.cudaStreamDestroy(self.handle)\n            self.handle = None\n            type(self).active_streams -= 1\n\n    def __enter__(self):\n        return self\n\n    def __exit__(self, *exc):\n        self.close()\n\n    def __del__(self):\n        self.close()\n\n\ndef _times(graphs, x, banks, flush, repeats):\n    samples = [[] for _ in graphs]\n    for rep in range(repeats):\n        x.copy_(banks[(rep // 2) % len(banks)])\n        order = range(len(graphs)) if rep % 2 == 0 else reversed(range(len(graphs)))\n        for index in order:\n            flush.add_(1)\n            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)\n            a.record()\n            graphs[index].graph.replay()\n            b.record()\n            b.synchronize()\n            samples[index].append(a.elapsed_time(b) * 1000)\n    return [statistics.median(values) for values in samples]\n\n\nclass MuseProjection:\n    """Prepared local projection; prepare/timing never occurs in __call__.\n\n    Shape plans are shared across layers; NN weight copies are instance-local.\n    Temporary Lt workspaces use the current stream\'s allocator, so concurrent\n    graph/eager calls do not race on shared mutable scratch. Set tokens=() to\n    construct without tuning, then call prepare() during the loader\'s warmup.\n    """\n    def __init__(self, weight, kind, *, tokens=VERIFY_TOKENS, budget_s=15):\n        if torch.cuda.is_current_stream_capturing():\n            raise RuntimeError("MuseProjection preparation must precede graph capture")\n        if kind not in _SHAPES or tuple(weight.shape) not in _SHAPES[kind]:\n            raise ValueError(f"unsupported TP8 {kind} projection shape {tuple(weight.shape)}")\n        if not weight.is_cuda or weight.dtype != torch.bfloat16 or not weight.is_contiguous():\n            raise ValueError("MuseProjection requires a contiguous CUDA BF16 weight")\n        self.source, self.weight, self.kind = weight, weight.detach(), kind\n        self.version = _version(weight)\n        self.transposed = None\n        self.plans, self.codes, self.timings = {}, {}, {}\n        with torch.cuda.device(weight.device):\n            self.mod = _module()\n        self.prepare(tokens, budget_s=budget_s)\n\n    def _skin_config(self, m):\n        nt = 1 if m <= 8 else 2\n        if self.kind == "pooler" and m > 8:\n            return nt, 2, 4, 1, 8, 1, 1\n        return nt, 1, 4, 1, 8 if self.kind == "unpooler" else 4, 1, 1\n\n    def _base(self, m):\n        n = self.weight.shape[0]\n        if 0 < m <= 16 and self.kind == "pooler" and n == 4096:\n            return "skin",\n        if 0 < m <= 8 and self.kind == "unpooler" and n == 1024:\n            return "skin",\n        if m == 1 and self.kind == "o":\n            return "skin",\n        return "cublas",\n\n    def _full_shape(self):\n        return self.weight.shape[0] == _SHAPES[self.kind][0][0]\n\n    def _ensure(self, m, config):\n        if config[0] not in ("lt", "lt_nn"):\n            return\n        nn = int(config[0] == "lt_nn")\n        if (m, nn) not in self.codes:\n            n, k = self.weight.shape\n            self.codes[(m, nn)] = _options(self.weight.device.index, m, n, k, nn)\n        if config[1] >= self.codes[(m, nn)].shape[0]:\n            raise ValueError(f"receipt heuristic unavailable: {config}")\n        if nn and self.transposed is None:\n            self.transposed = self.weight.T.contiguous()\n\n    def _run(self, x, config):\n        if config[0] == "cublas":\n            return F.linear(x, self.weight)\n        out = torch.empty((x.shape[0], self.weight.shape[0]), device=x.device, dtype=x.dtype)\n        if config[0] == "skin":\n            schedule = list(self._skin_config(x.shape[0]))\n            if len(config) > 1:\n                schedule[5] = config[1]\n            self.mod.skinny(x, self.weight, out, tuple(schedule))\n        elif config[0] in ("lt", "lt_nn"):\n            nn = int(config[0] == "lt_nn")\n            workspace = torch.empty(WORKSPACE_BYTES, device=x.device, dtype=torch.uint8)\n            w = self.transposed if nn else self.weight\n            self.mod.lt(x, w, out, workspace, self.codes[(x.shape[0], nn)], config[1], nn)\n        else:\n            raise ValueError(f"unknown projection variant {config}")\n        return out\n\n    def _variants(self, m):\n        base = self._base(m)\n        yield base\n        # Receipt small-M dispatch is already measured. Shards introduced by\n        # serving are different shapes and compete against cuBLAS locally.\n        if (self._full_shape() and m < 128) or (self.kind == "unpooler" and m <= 8):\n            return\n        yield "cublas",\n        if m <= 16:\n            yield "skin",\n            # New serving shards have only 16-64 output CTAs. The same receipt\n            # kernel\'s in-cluster K merge can fill more of H200\'s 132 SMs.\n            schedule = self._skin_config(m)\n            for split in (2, 4, 8):\n                if self.weight.shape[1] % (split * schedule[4] * 32 * schedule[2]) == 0:\n                    yield "skin", split\n        if self._full_shape() and (self.kind, m) in _RECEIPT:\n            yield _RECEIPT[(self.kind, m)]\n        for nn in (0, 1):\n            tag = "lt_nn" if nn else "lt"\n            try:\n                self._ensure(m, (tag, 0))\n                for index in range(self.codes[(m, nn)].shape[0]):\n                    yield tag, index\n            except Exception as exc:\n                logger.warning("MUSE_PROJ_REJECT role=%s m=%s layout=%s: %s", self.kind, m, tag, exc)\n\n    def prepare(self, tokens=VERIFY_TOKENS, *, budget_s=15):\n        """Select once per actual shape; rejects never abort a trial sweep."""\n        if torch.cuda.is_current_stream_capturing():\n            raise RuntimeError("call MuseProjection.prepare before graph capture")\n        deadline = time.monotonic() + max(0, budget_s)\n        flush = None\n        for m in dict.fromkeys(tokens):\n            if not 0 < m <= 256:\n                raise ValueError("prepare token counts must be in 1..256")\n            self.plans[m] = self._base(m)\n            key = (self.weight.device.index, self.kind, tuple(self.weight.shape), m)\n            x = torch.randn((m, self.weight.shape[1]), device=self.weight.device, dtype=torch.bfloat16)\n            if key in _PLAN_CACHE:\n                config = _PLAN_CACHE[key]\n                try:\n                    self._ensure(m, config)\n                    if not _agrees(self._run(x, config), F.linear(x, self.weight)):\n                        raise ValueError("cached choice exceeds 1% torch guard")\n                    self.plans[m] = config\n                    continue\n                except Exception as exc:\n                    logger.warning("MUSE_PROJ_REJECT cached role=%s m=%s: %s", self.kind, m, exc)\n            if time.monotonic() >= deadline:\n                continue\n            # Exact small-M receipt schedules need no redundant benchmarking.\n            configs = iter(self._variants(m))\n            base = next(configs)\n            if (self._full_shape() and m < 128) or (self.kind == "unpooler" and m <= 8):\n                try:\n                    if not _agrees(self._run(x, base), F.linear(x, self.weight)):\n                        raise ValueError("receipt choice exceeds 1% torch guard")\n                    self.plans[m] = _PLAN_CACHE[key] = base\n                except Exception as exc:\n                    self.plans[m] = ("cublas",)\n                    logger.warning("MUSE_PROJ_REJECT receipt role=%s m=%s: %s", self.kind, m, exc)\n                continue\n            if flush is None:\n                flush = torch.zeros(128 << 20, device=x.device, dtype=torch.uint8)\n            banks = [torch.randn_like(x) for _ in range(3)]\n            banks[2][:, ::257] *= 32\n            with _Graph(lambda: self._run(x, self._base(m))) as baseline:\n                trials, seen = [], set()\n                for config in chain((base,), configs):\n                    if time.monotonic() >= deadline:\n                        break\n                    if config in seen:\n                        continue\n                    seen.add(config)\n                    try:\n                        self._ensure(m, config)\n                        with _Graph(lambda c=config: self._run(x, c)) as graph:\n                            for bank in banks:\n                                x.copy_(bank)\n                                graph.graph.replay()\n                                if not _agrees(graph.out, F.linear(x, self.weight)):\n                                    raise ValueError("variant exceeds 1% torch guard")\n                            us = _times([baseline, graph], x, banks, flush, 2)[1]\n                            trials.append((us, config))\n                    except Exception as exc:\n                        logger.warning("MUSE_PROJ_REJECT role=%s m=%s config=%s: %s", self.kind, m, config, exc)\n                finalists = sorted(trials, key=lambda row: row[0])[:3]\n                receipt = _RECEIPT.get((self.kind, m)) if self._full_shape() else None\n                keep = [self._base(m)] + [config for _, config in finalists]\n                if receipt in {config for _, config in trials} and receipt not in keep:\n                    keep.append(receipt)\n                scores = []\n                for config in dict.fromkeys(keep):\n                    if time.monotonic() >= deadline:\n                        break\n                    try:\n                        self._ensure(m, config)\n                        with _Graph(lambda c=config: self._run(x, c)) as graph:\n                            for bank in banks:\n                                x.copy_(bank)\n                                graph.graph.replay()\n                                if not _agrees(graph.out, F.linear(x, self.weight)):\n                                    raise ValueError("finalist exceeds 1% torch guard")\n                            base_us, us = _times([baseline, graph], x, banks, flush, 12)\n                            scores.append((us / base_us, config, us))\n                    except Exception as exc:\n                        logger.warning("MUSE_PROJ_REJECT finalist role=%s m=%s config=%s: %s", self.kind, m, config, exc)\n                if scores:\n                    ratio, config, us = min(scores, key=lambda row: row[0])\n                    self.plans[m] = _PLAN_CACHE[key] = config\n                    self.timings[m] = {"config": config, "event_us": us, "ratio_to_base": ratio}\n                    logger.info("MUSE_PROJ_PICK role=%s n=%s m=%s %s", self.kind, self.weight.shape[0], m, self.timings[m])\n        # Keep an extra weight layout only while a selected plan uses it.\n        if not any(config[0] == "lt_nn" for config in self.plans.values()):\n            self.transposed = None\n        return self\n\n    def __call__(self, x):\n        if _version(self.source) != self.version:\n            raise RuntimeError("Muse projection weight changed: construct a new prepared instance")\n        if (x.ndim != 2 or x.shape[1] != self.weight.shape[1]\n                or x.device != self.weight.device or x.dtype != torch.bfloat16):\n            raise ValueError("projection requires CUDA BF16 [tokens, local K] input")\n        x = x.contiguous()\n        if not x.shape[0]:\n            return x.new_empty((0, self.weight.shape[0]))\n        return self._run(x, self.plans.get(x.shape[0], self._base(x.shape[0])))\n\n\n_CUDA_SOURCE = r"""// Lightweight host ABI for the unchanged CUDA kernels. No Torch C++ headers.\n#include <cuda_runtime.h>\n#include <cuda.h>\n#include <cudaTypedefs.h>\n#include <cuda_bf16.h>\n#include <cooperative_groups.h>\n#include <cublasLt.h>\n#include <stdint.h>\n#include <algorithm>\n#include <cstring>\n#include <cstdio>\nnamespace cg=cooperative_groups;\nstatic thread_local char lean_error[512];\nextern "C" const char* lean_error_string(){return lean_error;}\nstatic int lean_fail(const char* where,int code){std::snprintf(lean_error,sizeof(lean_error),"%s: status %d",where,code);return code ? code : -1;}\n#define LEAN_CUDA(call) do{auto e=(call);if(e!=cudaSuccess)return lean_fail(#call,(int)e);}while(0)\n#define LEAN_LT(call) do{auto e=(call);if(e!=CUBLAS_STATUS_SUCCESS)return lean_fail(#call,(int)e);}while(0)\nnamespace muse_proj {\n\n// Weight stream: read once, keep out of L1. LD 0: 256B L2 prefetch, 1: none, 2: 128B.\ntemplate <int LD>\n__device__ __forceinline__ uint4 ld_stream(const void* p) {\n  uint4 r;\n  if (LD == 0)\n    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"\n                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  else if (LD == 1)\n    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"\n                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  else\n    asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"\n                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  return r;\n}\n\n// Activations: shared by the CTA\'s warps through L1.\n__device__ __forceinline__ uint4 ld_act(const void* p) {\n  uint4 r;\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));\n  return r;\n}\n\n__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,\n                                         uint32_t a3, uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "\n      "{%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n\n// One CTA: BR = WN*TR*16 weight rows x BT = 8*NT tokens x KB of K (its cluster rank\'s slice).\n// Warp (wn, wk): TR row tiles of 16 x the wk-th 1/WK of the CTA\'s K slice, in steps of 32*U.\n// Lane (g = lane/4, t = lane%4) loads 16 bytes at k = k0 + 8t of rows g and g+8 (and of\n// token g for activations); the first mma takes k-slots {8t..8t+3}, the second {8t+4..8t+7}.\n// NA = 2: a second weight matrix (the router\'s lo half) at the same offsets, same accumulators.\n// XP: activations prefetched one batch ahead with the weights (else loaded at compute time).\ntemplate <int NT, int TR, int U, int NA, int XP, int LD>\n__global__ void __launch_bounds__(256) skinny_kernel(\n    const __nv_bfloat16* __restrict__ W0, const __nv_bfloat16* __restrict__ W1,\n    const __nv_bfloat16* __restrict__ X, void* __restrict__ Out, int M, int N, int K, int KB,\n    int WN, int WK) {\n  constexpr bool OUTF32 = NA == 2;\n  constexpr int XU = XP ? U : 1;\n  constexpr int XN = XP ? NT : 1;\n  extern __shared__ float sred[];\n  constexpr int BT = 8 * NT;\n  const int BR = WN * TR * 16;\n  const int S = BR + 4;  // padded token stride of the reduction tile (bank spread)\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int g = lane >> 2, t = lane & 3;\n  const int wn = warp % WN, wk = warp / WN;\n  const int row_cta = blockIdx.x * BR;\n  const int rowbase = row_cta + wn * TR * 16;\n  const int tok0 = blockIdx.z * BT;\n  const int KW = KB / WK;\n  const int kbeg = blockIdx.y * KB + wk * KW;\n  const int nsteps = KW / (32 * U);\n\n  float acc[TR][NT][4];\n#pragma unroll\n  for (int r = 0; r < TR; ++r)\n#pragma unroll\n    for (int j = 0; j < NT; ++j)\n#pragma unroll\n      for (int q = 0; q < 4; ++q) acc[r][j][q] = 0.f;\n\n  const size_t off = (size_t)(rowbase + g) * K + kbeg + 8 * t;\n  const __nv_bfloat16* wp0 = W0 + off;\n  const __nv_bfloat16* wp1 = W1 + off;\n  const size_t rs8 = (size_t)8 * K;\n  const size_t rs16 = (size_t)16 * K;\n  const __nv_bfloat16* xp[NT];\n#pragma unroll\n  for (int j = 0; j < NT; ++j) {\n    int tok = tok0 + j * 8 + g;\n    tok = tok < M ? tok : M - 1;  // padded tokens read a real row; their outputs are dropped\n    xp[j] = X + (size_t)tok * K + kbeg + 8 * t;\n  }\n\n  uint4 a0[U][NA][TR][2], a1[U][NA][TR][2];\n  uint4 b0[XU][XN], b1[XU][XN];\n\n  auto load = [&](uint4 (&a)[U][NA][TR][2], uint4 (&b)[XU][XN], int s) {\n    const int koff = s * 32 * U;\n#pragma unroll\n    for (int u = 0; u < U; ++u)\n#pragma unroll\n      for (int na = 0; na < NA; ++na)\n#pragma unroll\n        for (int r = 0; r < TR; ++r) {\n          const __nv_bfloat16* p = (na == 0 ? wp0 : wp1) + r * rs16 + koff + u * 32;\n          a[u][na][r][0] = ld_stream<LD>(p);\n          a[u][na][r][1] = ld_stream<LD>(p + rs8);\n        }\n    if (XP) {\n#pragma unroll\n      for (int u = 0; u < XU; ++u)\n#pragma unroll\n        for (int j = 0; j < XN; ++j) b[u][j] = ld_act(xp[j] + koff + u * 32);\n    }\n  };\n  auto compute = [&](uint4 (&a)[U][NA][TR][2], uint4 (&bp)[XU][XN], int s) {\n    const int koff = s * 32 * U;\n    uint4 bl[U][NT];\n#pragma unroll\n    for (int u = 0; u < U; ++u)\n#pragma unroll\n      for (int j = 0; j < NT; ++j) {\n        if (XP)\n          bl[u][j] = bp[u < XU ? u : 0][j < XN ? j : 0];\n        else\n          bl[u][j] = ld_act(xp[j] + koff + u * 32);\n      }\n#pragma unroll\n    for (int u = 0; u < U; ++u)\n#pragma unroll\n      for (int na = 0; na < NA; ++na)\n#pragma unroll\n        for (int r = 0; r < TR; ++r)\n#pragma unroll\n          for (int j = 0; j < NT; ++j) {\n            const uint4& lo = a[u][na][r][0];\n            const uint4& hi = a[u][na][r][1];\n            mma16816(acc[r][j], lo.x, hi.x, lo.y, hi.y, bl[u][j].x, bl[u][j].y);\n            mma16816(acc[r][j], lo.z, hi.z, lo.w, hi.w, bl[u][j].z, bl[u][j].w);\n          }\n  };\n\n  if (nsteps > 0) load(a0, b0, 0);\n  for (int s = 0; s < nsteps; s += 2) {\n    if (s + 1 < nsteps) load(a1, b1, s + 1);\n    compute(a0, b0, s);\n    if (s + 1 < nsteps) {\n      if (s + 2 < nsteps) load(a0, b0, s + 2);\n      compute(a1, b1, s + 1);\n    }\n  }\n\n  // Warp partials -> shared tile [WK][BT][S] (token-major so the output stores coalesce).\n#pragma unroll\n  for (int r = 0; r < TR; ++r)\n#pragma unroll\n    for (int j = 0; j < NT; ++j) {\n      const int row = wn * TR * 16 + r * 16 + g;\n      const int tok = j * 8 + 2 * t;\n      float* p = sred + (size_t)(wk * BT + tok) * S + row;\n      p[0] = acc[r][j][0];\n      p[S] = acc[r][j][1];\n      p[8] = acc[r][j][2];\n      p[S + 8] = acc[r][j][3];\n    }\n  __syncthreads();\n  const int E = BT * BR;\n  if (WK > 1) {\n    for (int e = threadIdx.x; e < E; e += blockDim.x) {\n      const int tok = e / BR, row = e - tok * BR;\n      float v = 0.f;\n      for (int w = 0; w < WK; ++w) v += sred[(w * BT + tok) * S + row];\n      sred[tok * S + row] = v;\n    }\n    __syncthreads();\n  }\n\n  auto store = [&](int tok, int row, float v) {\n    const size_t idx = (size_t)(tok0 + tok) * N + row_cta + row;\n    if (OUTF32)\n      reinterpret_cast<float*>(Out)[idx] = v;\n    else\n      reinterpret_cast<__nv_bfloat16*>(Out)[idx] = __float2bfloat16_rn(v);\n  };\n\n  const int KS = gridDim.y;\n  if (KS == 1) {\n    for (int e = threadIdx.x; e < E; e += blockDim.x) {\n      const int tok = e / BR, row = e - tok * BR;\n      if (tok0 + tok < M) store(tok, row, sred[tok * S + row]);\n    }\n  } else {\n    cg::cluster_group cluster = cg::this_cluster();\n    cluster.sync();\n    const int rank = (int)cluster.block_rank();\n    const int per = (E + KS - 1) / KS;\n    const int e0 = rank * per;\n    const int e1 = min(E, e0 + per);\n    for (int e = e0 + (int)threadIdx.x; e < e1; e += blockDim.x) {\n      const int tok = e / BR, row = e - tok * BR;\n      if (tok0 + tok >= M) continue;\n      float v = 0.f;\n      for (int q = 0; q < KS; ++q) v += cluster.map_shared_rank(sred, q)[tok * S + row];\n      store(tok, row, v);\n    }\n    cluster.sync();  // keep this CTA\'s shared tile alive until every rank has read it\n  }\n}\n\n\n} // namespace muse_proj\n\n\n\n#include <cuda.h>\n#include <cudaTypedefs.h>\n#ifndef MP_SPIN_LIMIT\n#define MP_SPIN_LIMIT (1u << 26)\n#endif\n#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n#define MP_SM90A 1\n#endif\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n#define MP_TMA 1  // TMA, mbarriers, clusters: sm_90 and later (wgmma itself needs sm_90a)\n#endif\n\ntemplate<int NT,int TR,int U,int NA,int XP,int LD>\nint lean_skinny_launch(const __nv_bfloat16* w,const __nv_bfloat16* lo,const __nv_bfloat16* x,void* out,\n                      int M,int N,int K,int wn,int wk,int ks,cudaStream_t stream){\n  constexpr int BT=NT*8;int BR=wn*TR*16;\n  if(N%BR || ks<1 || K%ks || (K/ks)%(wk*32*U) || wn*wk>8)return lean_fail("skinny shape",-1);\n  size_t shared=(size_t)wk*BT*(BR+4)*4;if(shared>48*1024)return lean_fail("skinny shared",-1);\n  cudaLaunchConfig_t c={};c.gridDim=dim3(N/BR,ks,(M+BT-1)/BT);c.blockDim=dim3(wn*wk*32);c.dynamicSmemBytes=shared;c.stream=stream;\n  cudaLaunchAttribute attr={};if(ks>1){attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=1;attr.val.clusterDim.y=ks;attr.val.clusterDim.z=1;c.attrs=&attr;c.numAttrs=1;}\n  return (int)cudaLaunchKernelEx(&c,muse_proj::skinny_kernel<NT,TR,U,NA,XP,LD>,w,lo,x,out,M,N,K,K/ks,wn,wk);\n}\nextern "C" int lean_skinny(const void* w,const void* lo,const void* x,void* out,\n                          int M,int N,int K,int nt,int tr,int u,int wn,int wk,int ks,int split2,int xp,int ld,void* stream){\n  int na=split2 ? 2 : 1;\n#define LSKIN(NT,TR,U,NA,XP,LD) if(nt==NT && tr==TR && u==U && na==NA && xp==XP && ld==LD)return lean_skinny_launch<NT,TR,U,NA,XP,LD>((const __nv_bfloat16*)w,(const __nv_bfloat16*)lo,(const __nv_bfloat16*)x,out,M,N,K,wn,wk,ks,(cudaStream_t)stream);\n    LSKIN(1,1,4,1,1,0)\n  LSKIN(1,2,4,1,1,0)\n  LSKIN(2,1,4,1,1,0)\n  LSKIN(2,2,4,1,1,0)\n  return lean_fail("skinny specialization",-1);\n}\n\nstruct LeanLayout{\n  cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;\n  int init(int n,int m,int k,int stride,int nn){\n    LEAN_LT(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;\n    LEAN_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));\n    LEAN_LT(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,nn ? n : k,nn ? k : n,stride));\n    LEAN_LT(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,k));LEAN_LT(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n));return 0;\n  }\n  ~LeanLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}\n};\nstatic cublasLtHandle_t lean_lt_handle=nullptr;\nstatic int lean_lt_init(){if(!lean_lt_handle)LEAN_LT(cublasLtCreate(&lean_lt_handle));return 0;}\nextern "C" int lean_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}\nextern "C" int lean_lt_options(int n,int m,int k,int stride,int nn,void* codes,int* count){\n  int err=lean_lt_init();if(err)return err;LeanLayout l;err=l.init(n,m,k,stride,nn);if(err)return err;\n  cublasLtMatmulPreference_t pref;LEAN_LT(cublasLtMatmulPreferenceCreate(&pref));size_t bytes=32*1024*1024;\n  auto status=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes));\n  if(status!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("Lt preference",(int)status);}\n  cublasLtMatmulHeuristicResult_t rows[32];status=cublasLtMatmulAlgoGetHeuristic(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,pref,32,rows,count);\n  cublasLtMatmulPreferenceDestroy(pref);if(status!=CUBLAS_STATUS_SUCCESS)return lean_fail("Lt heuristics",(int)status);\n  for(int i=0;i<*count;++i){if(rows[i].state!=CUBLAS_STATUS_SUCCESS || rows[i].workspaceSize>bytes)return lean_fail("Lt heuristic state",-1);std::memcpy((char*)codes+i*sizeof(cublasLtMatmulAlgo_t),&rows[i].algo,sizeof(cublasLtMatmulAlgo_t));}\n  return 0;\n}\nextern "C" int lean_lt_run(const void* w,const void* x,void* out,void* work,const void* codes,\n                          int n,int m,int k,int stride,int nn,int index,size_t work_bytes,void* stream){\n  if(work_bytes<32*1024*1024)return lean_fail("Lt workspace",-1);\n  int err=lean_lt_init();if(err)return err;LeanLayout l;err=l.init(n,m,k,stride,nn);if(err)return err;\n  cublasLtMatmulAlgo_t algo;std::memcpy(&algo,(const char*)codes+index*sizeof(algo),sizeof(algo));\n  float alpha=1.f,beta=0.f;\n  auto status=cublasLtMatmul(lean_lt_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&algo,work,work_bytes,(cudaStream_t)stream);\n  return status==CUBLAS_STATUS_SUCCESS ? 0 : lean_fail("Lt matmul",(int)status);\n}\n"""\n',_df_proj.__dict__)

"""DFlash2 local compute, preserving released weights and stock rounding.

The serving owner wires these helpers; no collective, drafter architecture or
sampling policy changes here. DFlashLinear is prepared after loading and before
capture. It admits a new GEMM only when it is bit-exact to stock on fresh probes.
"""

import logging
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import cuda as _df_cuda


_df_log = logging.getLogger(__name__)
_df_plan_cache = {}


@triton.jit
def _df_unary_kernel(X, Y, N: tl.constexpr, SCALE: tl.constexpr, CAP: tl.constexpr,
                     B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    v = tl.load(X + i, i < N, 0).to(tl.float32)
    v = v * SCALE
    if CAP > 0:
        v = v * (1.0 / CAP)
        v = _df_cuda.libdevice.tanh(v)
        v = v * CAP
    tl.store(Y + i, v, i < N)


def dflash_unary(logits, scale=0.1767766952966369, softcap=20.0):
    """Stock float -> scale -> divide -> tanh -> multiply, in one launch."""
    x = logits.contiguous()
    out = x if x.dtype == torch.float32 else torch.empty_like(x, dtype=torch.float32)
    if x.numel():
        _df_unary_kernel[(triton.cdiv(x.numel(), 256),)](
            x, out, x.numel(), scale, 0.0 if softcap is None else softcap,
            256, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _df_conv_kernel(X, D, W, O, T: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
                    DT: tl.constexpr, DS: tl.constexpr, DG: tl.constexpr,
                    BLOCK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    t, h = i // H, i % H
    g = h // G
    x0 = tl.load(X + i, t < T, 0).to(tl.float32)
    x1 = tl.load(X + (t - 1) * H + h, (t < T) & (t % BLOCK > 0), 0).to(tl.float32)
    d0 = tl.load(D + t * DT + g * DG, t < T, 0).to(tl.float32)
    d1 = tl.load(D + t * DT + DS + g * DG, t < T, 0).to(tl.float32)
    c0 = tl.load(W + h).to(tl.float32) + d0
    c1 = tl.load(W + H + h).to(tl.float32) + d1
    # Stock Inductor contracts the first multiply into the final add. BF16
    # coefficient/product temporaries remain FP32 in that compiled function.
    result = tl.fma(c0, x0, c1 * x1)
    tl.store(O + i, result, t < T)



def dflash_conv2(hidden, delta, base, block_size=8, group_size=16):
    """Two-tap released grouped convolution; accepts strided [T,2,H/16] deltas.

    Match stock Inductor's FP32 intermediates, first-product FMA and BF16 store.
    """
    if hidden.dtype != torch.bfloat16 or delta.dtype != torch.bfloat16 or base.dtype != torch.bfloat16:
        raise ValueError('DFlash conv2 expects the released BF16 tensors')
    t, h = hidden.shape
    if tuple(delta.shape) != (t, 2, h // group_size) or tuple(base.shape) != (2, h):
        raise ValueError('DFlash conv2 shape mismatch')
    x, w = hidden.contiguous(), base.contiguous()
    out = torch.empty_like(x)
    if t:
        _df_conv_kernel[(triton.cdiv(t * h, 256),)](
            x, delta, w, out, t, h, group_size, *delta.stride(), block_size,
            256, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _df_selector_pack_kernel(A, B, IDS, H, ANCHOR, P, Q,
                             C: tl.constexpr, L: tl.constexpr, K: tl.constexpr,
                             R: tl.constexpr, V: tl.constexpr, BATCH: tl.constexpr):
    row = tl.program_id(0)
    r = tl.arange(0, BATCH)
    request, slot, candidate = row // (L * K), row // K % L, row % K
    current = tl.load(IDS + row)
    if slot == 0:
        pred = tl.load(ANCHOR + request)
    else:
        pred = tl.load(IDS + row - K)
    av = tl.load(A + pred * R + r, r < R, 0).to(tl.float32)
    hv = tl.load(H + (request * L + slot) * R + r, r < R, 0).to(tl.float32)
    bv = tl.load(B + current * R + r, r < R, 0)
    tl.store(P + row * R + r, av * hv, r < R)
    tl.store(Q + row * R + r, bv, r < R)


def dflash_selector_inputs(predecessor_table, successor_table, candidate_ids, hidden, anchors):
    """Fuse codebook gathers and BF16 predecessor*hidden before the stock bmm."""
    c, slots, k = candidate_ids.shape
    rank = hidden.shape[-1]
    pred = torch.empty((c * slots, k, rank), device=hidden.device, dtype=hidden.dtype)
    succ = torch.empty_like(pred)
    if c:
        _df_selector_pack_kernel[(c * slots * k,)](
            predecessor_table, successor_table, candidate_ids.contiguous(), hidden.contiguous(),
            anchors.contiguous(), pred, succ, c, slots, k, rank,
            predecessor_table.shape[0], triton.next_power_of_2(rank), num_warps=4)
    return pred, succ


def dflash_selector_scores(predecessor_table, successor_table, candidate_ids, unary_logits, hidden, anchors):
    """Keep stock BF16 bmm and FP32 unary addition; replace only its input glue."""
    pred, succ = dflash_selector_inputs(predecessor_table, successor_table, candidate_ids, hidden, anchors)
    c, slots, k = candidate_ids.shape
    pair = torch.bmm(pred, succ.transpose(1, 2)).view(c, slots, k, k)
    return unary_logits[:, :, None] + pair


class DFlashLinear:
    """Arbitrary local released BF16 drafter weight, with exact-stock admission.

    Targets context FC, QKV/o, gate/up/down, convolution coefficient projections,
    and the draft-head projection. Keep the strided 3D selector-hidden bmm
    on stock; flattening it changes BF16 rounding. The caller retains collectives.
    """
    def __init__(self, weight, *, tokens=(8, 32, 128, 256), budget_s=10, stock_nn=False):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('prepare DFlashLinear before capture')
        if not weight.is_cuda or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            raise ValueError('DFlashLinear needs contiguous CUDA BF16 weights')
        self.stock_nn = stock_nn
        self.weight = weight.detach().T.contiguous() if stock_nn else weight.detach()
        self.source, self.version = weight, _df_proj._version(weight)
        self.transpose = weight.detach() if stock_nn else None
        self.codes, self.plans, self.timings, self.rejections = {}, {}, {}, []
        self.mod = _df_proj._module()
        self.prepare(tokens=tokens, budget_s=budget_s)

    def _df_ensure(self, m, config):
        if config[0] not in ('lt', 'lt_nn'):
            return
        nn = int(config[0] == 'lt_nn')
        if (m, nn) not in self.codes:
            n, k = self.weight.shape
            self.codes[(m, nn)] = _df_proj._options(self.weight.device.index, m, n, k, nn)
        if config[1] >= self.codes[(m, nn)].shape[0]:
            raise ValueError('unavailable Lt algorithm')
        if nn and self.transpose is None:
            self.transpose = self.weight.T.contiguous()

    def _df_run(self, x, config):
        if config[0] == 'stock':
            return self._df_stock(x)
        out = torch.empty((x.shape[0], self.weight.shape[0]), device=x.device, dtype=torch.bfloat16)
        if config[0] == 'skin':
            nt = 1 if x.shape[0] <= 8 else 2
            self.mod.skinny(x, self.weight, out, (nt, 1, 4, 1, 4, config[1], 1))
        else:
            nn = int(config[0] == 'lt_nn')
            work = torch.empty(32 << 20, device=x.device, dtype=torch.uint8)
            self.mod.lt(x, self.transpose if nn else self.weight, out, work,
                        self.codes[(x.shape[0], nn)], config[1], nn)
        return out

    def _df_stock(self, x):
        return torch.mm(x, self.source) if self.stock_nn else F.linear(x, self.weight)

    def _df_variants(self, m):
        yield ('stock',)
        n, k = self.weight.shape
        if m <= 32 and n % 16 == 0:
            for split in (1, 2, 4, 8):
                if k % (split * 512) == 0:
                    yield ('skin', split)
        for nn in (0, 1):
            tag = 'lt_nn' if nn else 'lt'
            try:
                self._df_ensure(m, (tag, 0))
                for index in range(self.codes[(m, nn)].shape[0]):
                    yield (tag, index)
            except Exception as exc:
                _df_log.warning('DFLASH_REJECT layout=%s m=%d: %s', tag, m, exc)

    def prepare(self, *, tokens=(8, 32, 128, 256), budget_s=10):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('DFlash tuning must precede capture')
        deadline = time.monotonic() + budget_s
        flush = None
        for m in tokens:
            self.plans[m] = ('stock',)
            key = (self.weight.device.index, tuple(self.weight.shape), m, self.stock_nn)
            x = torch.randn((m, self.weight.shape[1]), device=self.weight.device, dtype=torch.bfloat16)
            if key in _df_plan_cache:
                candidate = _df_plan_cache[key]
                try:
                    self._df_ensure(m, candidate)
                    if not torch.equal(self._df_run(x, candidate), self._df_stock(x)):
                        raise ValueError('cached algorithm differs from deterministic stock')
                    self.plans[m] = candidate
                    continue
                except Exception as exc:
                    _df_log.warning('DFLASH_REJECT cache m=%d: %s', m, exc)
            if time.monotonic() >= deadline:
                continue
            if flush is None:
                flush = torch.zeros(128 << 20, device=x.device, dtype=torch.uint8)
            banks = [torch.randn_like(x) for _ in range(3)]
            banks[-1].zero_(); banks[-1][:, 127] = .5
            trials = []
            with _df_proj._Graph(lambda: self._df_stock(x)) as baseline:
                for config in self._df_variants(m):
                    if time.monotonic() >= deadline:
                        break
                    try:
                        self._df_ensure(m, config)
                        with _df_proj._Graph(lambda c=config: self._df_run(x, c)) as graph:
                            for fresh in banks:
                                x.copy_(fresh); graph.graph.replay()
                                if not torch.equal(graph.out, self._df_stock(x)):
                                    raise ValueError('not exact to deterministic stock')
                            base_us, us = _df_proj._times([baseline, graph], x, banks, flush, 4)
                            trials.append((us / base_us, config))
                    except Exception as exc:
                        self.rejections.append((m, config, str(exc)))
                        _df_log.warning('DFLASH_REJECT shape=%s m=%d config=%s: %s', tuple(self.weight.shape), m, config, exc)
            # Recheck global finalists against stock with >=20 paired cold
            # replays; a four-sample coarse win must not determine serving.
            finalists=[('stock',)]+[config for _,config in sorted(trials)[:3]]
            paired=[]
            with _df_proj._Graph(lambda:self._df_stock(x)) as baseline:
                for config in dict.fromkeys(finalists):
                    if time.monotonic()>=deadline:
                        break
                    try:
                        self._df_ensure(m,config)
                        with _df_proj._Graph(lambda c=config:self._df_run(x,c)) as graph:
                            for fresh in banks:
                                x.copy_(fresh);graph.graph.replay()
                                if not torch.equal(graph.out,self._df_stock(x)):
                                    raise ValueError('finalist differs from stock')
                            base_us,us=_df_proj._times([baseline,graph],x,banks,flush,20)
                            paired.append((us/base_us,config))
                    except Exception as exc:
                        self.rejections.append((m,config,str(exc)))
                        _df_log.warning('DFLASH_REJECT finalist m=%d config=%s: %s',m,config,exc)
            if paired:
                ratio,config=min(paired,key=lambda row:row[0])
                self.plans[m]=_df_plan_cache[key]=config
                self.timings[m]=dict(config=config,paired_ratio=ratio,pairs=20)
        if not any(v[0] == 'lt_nn' for v in self.plans.values()):
            self.transpose = None
        return self

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != self.weight.shape[1] or x.dtype != torch.bfloat16:
            raise ValueError('DFlashLinear supports only the local 2D BF16 GEMM')
        if self.version is not None and _df_proj._version(self.source) != self.version:
            raise RuntimeError('drafter weight changed after preparation')
        return self._df_run(x.contiguous(), self.plans.get(x.shape[0], ('stock',)))


class DFlashConv2:
    """Drop-in local prepare/finish math for DFlashGroupedConv's two taps.

    Keep the original module's parameters/registration. Bind this helper after
    loading its kernel_projection.weight and base_kernel; no weights change.
    """
    def __init__(self, projection_weight, base_kernel, *, block_size=8, tokens=(8,32,128,256), budget_s=10):
        h = projection_weight.shape[1]
        if tuple(base_kernel.shape) != (2,2,h) or projection_weight.shape[0] != h // 4:
            raise ValueError('released DFlash two-tap/group16 convolution shape required')
        self.base = base_kernel
        self.block_size = block_size
        self.projection = DFlashLinear(projection_weight, tokens=tokens, budget_s=budget_s)

    def prepare(self, x):
        coefficients = self.projection(x).view(x.shape[0], 2, 2, x.shape[1] // 16)
        return dflash_conv2(x, coefficients[:,0], self.base[0], self.block_size), coefficients[:,1]

    def finish(self, x, coefficients):
        return dflash_conv2(x, coefficients, self.base[1], self.block_size)

@torch.compile(dynamic=True)
def _df_stock_grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, 0)
        out = out + coefficients[:, tap] * shifted
    return out.flatten(-2)

@torch.compile(dynamic=True)
def _df_stock_score_edges(
    *,
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    keys = successor_table[candidate_ids]
    # Concatenate the ids and look them up once. Concatenating the looked-up rows
    # instead moves a [b, slots, k, rank] float tensor where this moves one id per
    # candidate, and it costs a second gather for the anchor.
    predecessor_ids = torch.cat(
        [anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]],
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], keys
    )

_DF_MODULE_SHA='ecd7348666f1bf34d2ec0bd73084d7fa32b2d8ad812c331478fbf448f3bac65a'

# Diagnostic-only prepare extension. No diagnostic state reaches kernel_fn.
_df_scored_prepare = prepare


def _df_diag_equal(a, b):
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_df_diag_equal(x,y) for x,y in zip(a,b))
    return torch.equal(a,b)


def _df_diag_pair(stock, fast, mutate, flush, deadline):
    import statistics,time
    with _df_proj._Graph(stock) as a, _df_proj._Graph(fast) as b:
        old,new=[],[]
        for rep in range(20):
            if time.monotonic() >= deadline:
                raise TimeoutError('20 complete pairs would exceed diagnostic budget')
            mutate()
            for index in ((0,1) if rep%2==0 else (1,0)):
                flush.add_(1)
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();(a if index==0 else b).graph.replay();end.record();end.synchronize()
                (old if index==0 else new).append(begin.elapsed_time(end)*1000)
            if not _df_diag_equal(a.out,b.out):
                raise ValueError('fresh-input result differs from deterministic stock; rejected')
        return statistics.median(old),statistics.median(new)


def _df_run_diagnostics(deadline, lines):
    import time
    device=torch.device('cuda')
    flush=torch.zeros(128<<20,device=device,dtype=torch.uint8)
    # Local TP8 shapes. Context-KV is already a stacked NN GEMM in stock.
    specs=(('conv_coeff',2048,8192,8,10,False),('gate_up',7168,8192,8,5,False),
           ('down',8192,3584,8,5,False),('context_fc',1024,40960,8,1,False),
           ('qkv',1280,8192,8,5,False),('o',8192,1024,8,5,False),
           ('context_kv',1280,8192,8,1,True),
           ('draft_head',25256,8192,7,1,False))
    for name,n,k,mult,count,nn in specs:
        if time.monotonic()+10>=deadline:
            lines.append(f'DFB_SKIP {name} budget');continue
        try:
            canonical=torch.randn((n,k),device=device,dtype=torch.bfloat16)/k**.5
            w=canonical.T.contiguous() if nn else canonical
            ms=tuple(c*mult for c in (1,4,16,32))
            op=DFlashLinear(w,tokens=ms,budget_s=min(8,max(0,deadline-time.monotonic()-10)),stock_nn=nn)
            for c,m in zip((1,4,16,32),ms):
                if time.monotonic()+2>=deadline:
                    lines.append(f'DFB_SKIP {name} c={c} budget');continue
                try:
                    x=torch.randn((m,k),device=device,dtype=torch.bfloat16)
                    old,new=_df_diag_pair(lambda:op._df_stock(x),lambda:op(x),
                                          lambda:x.copy_(torch.randn_like(x)),flush,deadline)
                    chosen=op.plans.get(m,('stock',))
                    credit=0.0 if chosen==('stock',) else count*(old-new)/1000
                    rejects=sum(r[0]==m for r in op.rejections)
                    lines.append(f'DFB linear {name} c={c} m={m} n={n} k={k} old_us={old:.3f} new_us={new:.3f} pairs=20 choice={chosen} rejects={rejects} save_ms={credit:.6f}')
                except Exception as exc:
                    lines.append(f'DFB_REJECT {name} c={c}: {str(exc).splitlines()[0][:100]}')
            del op,w,canonical
        except Exception as exc:
            lines.append(f'DFB_REJECT {name}: {str(exc).splitlines()[0][:100]}')
    for c in (1,4,16,32):
        for label in ('unary','selector_hidden','conv','selector'):
            if time.monotonic()+10>=deadline:
                lines.append(f'DFB_SKIP {label} c={c} budget');continue
            try:
                if label=='unary':
                    # TP8 candidate_topk gathers FP32 values: float() is a view,
                    # not a fifth cast kernel. Keep independent mutable inputs.
                    x=torch.randn((c*7,16),device=device,dtype=torch.float32)
                    fast_x=x.clone()
                    stock=lambda:x.float().mul_(.1767766952966369).div_(20).tanh_().mul_(20)
                    fast=lambda:dflash_unary(fast_x)
                    def mutate():
                        fresh=torch.randn_like(x)*20;x.copy_(fresh);fast_x.copy_(fresh)
                    count=1
                elif label=='selector_hidden':
                    x=torch.randn((c,8,8192),device=device,dtype=torch.bfloat16)
                    view=x[:,1:]
                    w=torch.randn((256,8192),device=device,dtype=torch.bfloat16)/8192**.5
                    # Stock is a strided bmm at C>1. Keep its arithmetic; a
                    # flattened mm is not an exact replacement.
                    stock=lambda:F.linear(view,w)
                    fast=stock
                    mutate=lambda:x.copy_(torch.randn_like(x))
                    count=0
                elif label=='conv':
                    x=torch.randn((c*8,8192),device=device,dtype=torch.bfloat16)
                    full=torch.randn((c*8,2,2,512),device=device,dtype=torch.bfloat16)
                    delta=full[:,1];w=torch.randn((2,8192),device=device,dtype=torch.bfloat16)
                    # The actual model marks these dimensions static before tracing.
                    torch._dynamo.mark_static(x,1);torch._dynamo.mark_static(delta,1);torch._dynamo.mark_static(delta,2)
                    stock=lambda:_df_stock_grouped_conv(x,delta,w,8,512,16,2)
                    fast=lambda:dflash_conv2(x,delta,w)
                    mutate=lambda:(x.copy_(torch.randn_like(x)),full.copy_(torch.randn_like(full)))
                    count=20
                else:
                    v=202048
                    a=torch.randn((v,256),device=device,dtype=torch.bfloat16)
                    b=torch.randn_like(a);ids=torch.randint(v,(c,7,16),device=device)
                    h=torch.randn((c,7,256),device=device,dtype=torch.bfloat16)
                    unary=torch.randn((c,7,16),device=device);anchor=torch.randint(v,(c,),device=device)
                    for tensor in (ids,unary,h):
                        torch._dynamo.mark_static(tensor,1);torch._dynamo.mark_static(tensor,2)
                    stock=lambda:_df_stock_score_edges(predecessor_table=a,successor_table=b,candidate_ids=ids,
                                                       unary_logits=unary,hidden=h,anchor_token_ids=anchor,top_k=16)
                    fast=lambda:dflash_selector_scores(a,b,ids,unary,h,anchor)
                    mutate=lambda:(ids.random_(0,v),h.copy_(torch.randn_like(h)),unary.normal_())
                    count=1
                old,new=_df_diag_pair(stock,fast,mutate,flush,deadline)
                lines.append(f'DFB glue {label} c={c} old_us={old:.3f} new_us={new:.3f} pairs=20 count={count} save_ms={count*(old-new)/1000:.6f}')
            except Exception as exc:
                lines.append(f'DFB_REJECT {label} c={c}: {str(exc).splitlines()[0][:100]}')


def prepare(weights):
    import time
    started=time.monotonic()
    state=_df_scored_prepare(weights)
    lines=[]
    try:
        _df_run_diagnostics(started+225,lines)
    except Exception as exc:
        lines.append(f'DFB_REJECT diagnostic: {str(exc).splitlines()[0][:120]}')
    # Keep every compact result in the final 16K receipt tail, after compilation,
    # trial rejections and the scored dispatch's verbose prepare output.
    print('DFB_BEGIN raw graph CUDA-event medians; cold128MiB; 20 order-swapped pairs; exact-stock guard',flush=True)
    for line in lines:print(line,flush=True)
    print(f'DFB_END prepare_s={time.monotonic()-started:.3f} base=f3786341611f4f1f module_sha={_DF_MODULE_SHA}',flush=True)
    return state
