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


# ---- h200_lab wrapper: payloads moe_m128_producer_consumer 8e4d7be1588a7036 on base f3786341611f4f1f ----
_LAB_PAYLOADS = [('moe_m128_producer_consumer', '8e4d7be1588a7036', 'import torch\nimport torch as rv_torch\nimport torch\nimport torch as rv_torch\nimport torch as hv_torch\nimport torch as rr_torch\nimport math as fx_math\nimport statistics as fx_statistics\nimport time as fx_time\nfrom typing import Callable as fx_Callable\nimport torch as fx_torch\nfx_HIDDEN = 8192\nfx_D_EXPERT = 4096\nfx_N_EXPERTS = 256\nfx_LOCAL_EXPERTS = 32\nfx_TOP_K = 8\nfx_ROUTER_MULTIPLIER = 0.1767766952966369\nfx_RMS_EPS = 1e-05\nfx_STREAM_SEED = 5354\nfx_STREAM_MEAN = 0.7\nfx_STREAM_SHARED = 0.9\nfx_STREAM_RANK = 4\nfx_DRAFT_SHARE = 0.6\n\ndef fx__rms(x: fx_torch.Tensor) -> fx_torch.Tensor:\n    xf = x.float()\n    return xf * fx_torch.rsqrt(xf.square().mean(-1, keepdim=True) + fx_RMS_EPS)\n\ndef fx__basis(device: fx_torch.device) -> tuple[fx_torch.Tensor, fx_torch.Tensor]:\n    gen = fx_torch.Generator().manual_seed(fx_STREAM_SEED)\n    mean = fx_torch.randn((fx_HIDDEN,), generator=gen).to(device)\n    shared = fx_torch.randn((fx_STREAM_RANK, fx_HIDDEN), generator=gen).to(device)\n    return (mean, shared)\n\ndef fx_make_fixture(weights: dict[str, fx_torch.Tensor], tokens: int, seed: int, *, mode: str=\'real\', q_len: int | None=None) -> dict:\n    """Return an expert-call fixture with real router or controlled route IDs.\n\n    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.\n    Pass it explicitly for other loads. Draft sharing applies within every verify block.\n    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps\n    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every\n    slot to a uniformly chosen local expert, matching older synthetic tuners.\n    """\n    device = weights[\'router\'].device\n    gen = fx_torch.Generator(device=device).manual_seed(seed)\n    mean, shared = fx__basis(device)\n    coef = fx_torch.randn((tokens, fx_STREAM_RANK), generator=gen, device=device)\n    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)\n    if q_len < 1 or tokens % q_len:\n        raise ValueError(f\'tokens={tokens} must be divisible by q_len={q_len}\')\n    scale = fx_torch.ones((tokens // q_len, q_len, 1), device=device)\n    if 1 < q_len <= 8:\n        scale[:, 1:] = fx_DRAFT_SHARE\n    scale = scale.reshape(tokens, 1)\n    stream = scale * (fx_STREAM_MEAN * mean + fx_math.sqrt(fx_STREAM_SHARED / fx_STREAM_RANK) * (coef @ shared)) + (1 - fx_STREAM_SHARED * scale.square()).sqrt() * fx_torch.randn((tokens, fx_HIDDEN), generator=gen, device=device)\n    h = fx__rms(stream) * (1.0 + weights[\'pre_ffn_norm\'].float())\n    router = weights[\'router\'].float()\n    biased = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER) + weights[\'router_bias\'].float()\n    scores = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER)\n    global_ids = biased.topk(fx_TOP_K, dim=-1).indices.to(fx_torch.int32)\n    picked = scores.gather(1, global_ids.long())\n    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)\n    ranges = fx_torch.bincount((global_ids // fx_LOCAL_EXPERTS).flatten(), minlength=fx_N_EXPERTS // fx_LOCAL_EXPERTS)\n    local_start = fx_LOCAL_EXPERTS * int(ranges.argmax())\n    local = ((global_ids - local_start) % fx_N_EXPERTS).to(fx_torch.int32)\n    local_mask = local < fx_LOCAL_EXPERTS\n    if mode == \'uniform\':\n        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)\n    elif mode == \'matched_uniform\':\n        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)\n        lids = fx_torch.where(local_mask, lids, fx_torch.full_like(lids, -1))\n    elif mode == \'real\':\n        lids = fx_torch.where(local_mask, local, fx_torch.full_like(local, -1))\n    else:\n        raise ValueError(mode)\n    x = h @ weights[\'pooler\'].float().t()\n    x = (fx__rms(x) * (1.0 + weights[\'pre_expert_norm\'].float())).to(fx_torch.bfloat16)\n    tw = topk_weights if mode != \'uniform\' else fx_torch.rand((tokens, fx_TOP_K), generator=gen, device=device)\n    if mode == \'uniform\':\n        tw = tw / tw.sum(-1, keepdim=True)\n    cfg = {\'local_count\': 32, \'d_hidden\': fx_D_EXPERT, \'d_expert\': fx_D_EXPERT, \'local_start\': 0}\n    return {\'hidden_states\': x, \'topk_ids\': lids, \'topk_weights\': tw.float(), \'config\': cfg, \'local_start\': local_start, \'global_ids\': global_ids, \'local_mask\': local_mask}\nimport math as rg_math\nimport functools as rg_functools\nimport torch as rg_torch\nrg_HIDDEN = 8192\nrg_Q_SIZE = 1024\nrg_QKV_SIZE = 2304\nrg_D_EXPERT = 4096\nrg_LOCAL_EXPERTS = 32\nrg_N_EXPERTS = 256\nrg_STREAM_SEED = 5354\nrg_STREAM_RANK = 4\nrg_ROUTER_STD = 0.045\nrg_ROUTER_ALIGN = 0.07\nrg_ROUTER_BIAS_STD = 0.055\nrg_BLOCK = 16\nrg_E4M3_MAX = 448.0\nrg_SCALE_LOG2_MEAN = 7.0\nrg_SCALE_LOG2_STD = 0.4\n\ndef rg_random_weights(rows: int, k: int, *, generator: rg_torch.Generator, device: str | rg_torch.device, leading: tuple[int, ...]=(), wide_scales: bool=False):\n    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.\n\n    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)\n    clamped to [8, 448]; the global scale\n    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448\n    instead.\n    """\n    shape = (*leading, rows, k // 2)\n    packed = rg_torch.randint(0, 256, shape, dtype=rg_torch.uint8, generator=generator, device=device)\n    scale_shape = (*leading, rows, k // rg_BLOCK)\n    if wide_scales:\n        u = rg_torch.rand(scale_shape, generator=generator, device=device)\n        sf = rg_torch.exp2(-6 + u * (rg_math.log2(rg_E4M3_MAX) + 6)).to(rg_torch.float8_e4m3fn)\n        mean_sq = float(sf.float().pow(2).mean())\n    else:\n        z = rg_torch.randn(scale_shape, generator=generator, device=device)\n        sf = rg_torch.exp2(rg_SCALE_LOG2_MEAN + rg_SCALE_LOG2_STD * z).clamp(8.0, rg_E4M3_MAX).to(rg_torch.float8_e4m3fn)\n        mean_sq = 4.0 ** rg_SCALE_LOG2_MEAN * rg_math.exp(2 * (rg_math.log(2) * rg_SCALE_LOG2_STD) ** 2)\n    gs = rg_torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=rg_torch.float32, device=device)\n    return (packed, sf, gs)\n\ndef rg_generate_weights(seed: int, device: str=\'cuda\', router_seed: int | None=None):\n    """The layer\'s weights from `seed`. With router_seed, the router and its bias come from that\n    seed instead."""\n    gen = rg_torch.Generator(device=device).manual_seed(seed)\n\n    def dense(n, k, dtype=rg_torch.bfloat16):\n        return (rg_torch.randn((n, k), generator=gen, device=device) / rg_math.sqrt(k)).to(dtype)\n\n    def vec(n, std):\n        return (std * rg_torch.randn((n,), generator=gen, device=device)).to(rg_torch.bfloat16)\n    w13, w13_sf, gs13 = rg_random_weights(2 * rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))\n    w2, w2_sf, gs2 = rg_random_weights(rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))\n    gs13 = gs13 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))\n    gs2 = gs2 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))\n    o, attn_gate, pre_ffn = (dense(rg_HIDDEN, rg_Q_SIZE), vec(rg_HIDDEN, 0.3), vec(rg_HIDDEN, 0.1))\n    weights = {\'qkv_gate\': dense(rg_QKV_SIZE, rg_HIDDEN), \'o\': o, \'attn_residual_gate\': attn_gate, \'pre_ffn_norm\': pre_ffn, **rg__router(gen, device, pre_ffn), \'pooler\': dense(rg_D_EXPERT, rg_HIDDEN), \'pre_expert_norm\': vec(rg_D_EXPERT, 0.1), \'w13\': w13, \'w13_sf\': w13_sf, \'w13_gs\': gs13, \'w2\': w2, \'w2_sf\': w2_sf, \'w2_gs\': gs2, \'post_expert_norm\': vec(rg_D_EXPERT, 0.1), \'unpooler\': dense(rg_HIDDEN, rg_D_EXPERT), \'post_ffn_norm\': vec(rg_HIDDEN, 0.1), \'ffn_residual_gate\': vec(rg_HIDDEN, 0.3), \'next_pre_attn_norm\': vec(rg_HIDDEN, 0.1)}\n    if router_seed is not None:\n        weights.update(rg__router(rg_torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))\n    return weights\n\n@rg_functools.cache\ndef rg__stream_basis(device: str):\n    """The residual stream\'s mean direction [8192] and shared directions [STREAM_RANK, 8192]."""\n    gen = rg_torch.Generator().manual_seed(rg_STREAM_SEED)\n    mean = rg_torch.randn((rg_HIDDEN,), generator=gen)\n    shared = rg_torch.randn((rg_STREAM_RANK, rg_HIDDEN), generator=gen)\n    return (mean.to(device), shared.to(device))\n\ndef rg__router(gen, device, pre_ffn):\n    """FP32 router rows, each with a component along the stream\'s mean direction, and the router\n    bias. pre_ffn is unused."""\n    mean, _ = rg__stream_basis(str(device))\n    w = rg_ROUTER_STD * rg_torch.randn((rg_N_EXPERTS, rg_HIDDEN), generator=gen, device=device)\n    align = rg_ROUTER_ALIGN * rg_torch.randn((rg_N_EXPERTS, 1), generator=gen, device=device)\n    rows = w + align * mean[None, :] / rg_math.sqrt(rg_HIDDEN)\n    bias = rg_ROUTER_BIAS_STD * rg_torch.randn((rg_N_EXPERTS,), generator=gen, device=device)\n    return {\'router\': rows.contiguous(), \'router_bias\': bias.contiguous()}\nCAPTURED_R0 = \'eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK\'\nimport base64\nimport statistics\nimport time\nimport zlib\nimport torch\nimport triton\nimport triton.language as tl\nimport ctypes as pf_ct\nimport hashlib as pf_hashlib\nimport logging as pf_logging\nimport math as pf_math\nimport os as pf_os\nimport statistics as pf_statistics\nimport time as pf_time\nimport torch as pf_torch\nimport torch.nn.functional as pf_F\nimport triton as pf_triton\nimport triton.language as pf_tl\nimport json as pf_json\nimport os as pf_os\nfrom typing import TypedDict as pf_TypedDict\nimport torch as pf_torch\npf__saved_flag = pf_os.environ.get(\'HUMMING_DISABLE_PARALLEL_BUILD\')\npf_os.environ[\'HUMMING_DISABLE_PARALLEL_BUILD\'] = \'1\'\ntry:\n    from humming import ops as pf_humming_ops\n    from humming.config import GemmType as pf_GemmType, WeightScale2Type as pf_WeightScale2Type\n    from humming.kernel.humming import HummingKernel as pf_HummingKernel\n    from humming.layer import HummingMethod as pf_HummingMethod\n    from humming.schema.modelopt import ModeloptNvfp4WeightSchema as pf_ModeloptNvfp4WeightSchema\n    from humming.tune import get_heuristics_config as pf_get_heuristics_config\nfinally:\n    if pf__saved_flag is None:\n        pf_os.environ.pop(\'HUMMING_DISABLE_PARALLEL_BUILD\', None)\n    else:\n        pf_os.environ[\'HUMMING_DISABLE_PARALLEL_BUILD\'] = pf__saved_flag\nimport ctypes as pf_ct\nimport hashlib as pf_hashlib\nimport logging as pf_logging\nimport os as pf_os\nimport statistics as pf_statistics\nimport time as pf_time\nimport torch as pf_torch\nimport torch.nn.functional as pf_F\nimport triton as pf_triton\nimport triton.language as pf_tl\n\nclass MWGraph:\n\n    def __init__(self, fn, data):\n        self.data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}\n        fn(self.data)\n        torch.cuda.synchronize()\n        self.graph = torch.cuda.CUDAGraph()\n        with torch.cuda.graph(self.graph):\n            self.out = fn(self.data)\n\n    def refresh(self, data):\n        for k in (\'hidden_states\', \'topk_ids\', \'topk_weights\'):\n            self.data[k].copy_(data[k])\n\n    def replay(self):\n        self.graph.replay()\n        return self.out\nimport base64\nimport hashlib\nimport statistics\nimport time\nimport zlib\n\ndef dc_weights(seed, device):\n    w = rg_generate_weights(seed, device=device)\n    gen = torch.Generator(device=device).manual_seed(seed + 1000003)\n    w[\'w13_gs_gate\'] = w[\'w13_gs\'].clone()\n    w[\'w13_gs_up\'] = w[\'w13_gs\'] * torch.exp(0.55 * torch.randn(32, generator=gen, device=device))\n    w[\'post_norm_w\'] = w[\'post_expert_norm\']\n    assert bool((w[\'w13_gs_gate\'] != w[\'w13_gs_up\']).all())\n    return w\n\ndef dc_oracles(ds, w):\n    """Independent FP32 scratch oracle; no expanded bank survives this function."""\n    old = torch.backends.cuda.matmul.allow_tf32\n    torch.backends.cuda.matmul.allow_tf32 = False\n    try:\n        wants = [torch.zeros_like(d[\'hidden_states\'], dtype=torch.float32) for d in ds]\n        lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=w[\'w13\'].device)\n        for e in range(32):\n            active = [torch.where(d[\'topk_ids\'] == e) for d in ds]\n            if not any((t.numel() for t, _ in active)):\n                continue\n\n            def dq(key):\n                q = w[key][e]\n                c = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()\n                return lut[c & 7] * torch.where(c & 8 != 0, -1.0, 1.0) * w[key + \'_sf\'][e].float().repeat_interleave(16, -1)\n            w13 = dq(\'w13\')\n            w2 = dq(\'w2\') * w[\'w2_gs\'][e]\n            for d, out, (t, j) in zip(ds, wants, active):\n                if t.numel() == 0:\n                    continue\n                h = d[\'hidden_states\'][t].float() @ w13.t()\n                g = h[:, :4096] * w[\'w13_gs_gate\'][e]\n                u = h[:, 4096:] * w[\'w13_gs_up\'][e]\n                a = (torch.nn.functional.silu(g) * u).bfloat16()\n                y = a.float() @ w2.t()\n                y *= 1 + w[\'post_norm_w\'].float()\n                y *= torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-08)\n                out.index_add_(0, t, y * d[\'topk_weights\'][t, j, None])\n        return [out.bfloat16() for out in wants]\n    finally:\n        torch.backends.cuda.matmul.allow_tf32 = old\n\ndef dc_fixtures(w, m, r0=False):\n    ds = [fx_make_fixture(w, m, 9000 + m + 7919 * j) for j in range(3)]\n    if r0:\n        raw = zlib.decompress(base64.b64decode(CAPTURED_R0))\n        ms = (8, 16, 32, 128, 256)\n        offset = 8 * sum(ms[:ms.index(m)])\n        ids = torch.tensor(list(raw[offset:offset + m * 8]), device=w[\'w13\'].device, dtype=torch.int32).view(m, 8)\n        ids = torch.where(ids < 32, ids, -1)\n        ds = [dict(d, topk_ids=ids.clone()) for d in ds]\n    return ds\nimport functools\nimport tilelang\nimport tilelang.language as T\nimport torch\nimport triton\nimport triton.language as tl\nimport functools\nimport hashlib\nfrom pathlib import Path\nimport torch\nimport triton\nimport triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\nimport statistics\nimport time\nimport functools\nimport tilelang\nimport tilelang.language as T\nimport torch\nimport triton\nimport triton.language as tl\nimport functools\nimport hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\nimport statistics\nimport time\nimport functools\nimport hashlib\nfrom torch.utils.cpp_extension import load_inline\nimport time\nimport statistics\nimport functools\nimport gc\nfrom torch.utils.cpp_extension import load_inline\nimport statistics\nimport time\nimport functools\nimport gc\nfrom torch.utils.cpp_extension import load_inline\nimport statistics\nimport time\nimport functools\nimport gc\nfrom torch.utils.cpp_extension import load_inline\nimport statistics\nimport time\n\ndef pm_error(got, want):\n    if not bool(torch.isfinite(got).all()):\n        return (float(\'inf\'), float(\'inf\'))\n    delta = got.float() - want.float()\n    den = want.float().norm(dim=-1)\n    return (float(delta.norm() / want.float().norm().clamp_min(1e-12)), float((delta.norm(dim=-1) / den.clamp_min(den.mean() * 0.1 + 1e-12)).max()))\n"""H200770809cf admitted H_H M128: NT1<=16, WGMMA32>16 in both phases."""\nimport functools\nimport hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\ndef m128_hh_mw_repack_planes(q):\n    """A permutation of all four bits of every nibble, matching RS register ownership.\n\n    Two words encode the 16 codes each lane needs for M16/K32. The extraction\n    table is the incumbent V9 bit-plane bijection, used here with WGMMA layout.\n    """\n    e, n, k2 = q.shape\n    k = k2 * 2\n    dst = torch.empty((e, n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int32)\n    slots = ((\'A\', 0), (\'A\', 4), (\'A\', 10), (\'B\', 2), (\'B\', 6), (\'B\', 12), (\'W\', 24), (\'W\', 30))\n    mask = 1613783088\n    for expert in range(e):\n        codes = torch.stack((q[expert] & 15, q[expert] >> 4), -1).reshape(n // 16, 2, 8, k // 32, 2, 2, 4, 2)\n        codes = codes.permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 32, 32, 8, 2).to(torch.int64)\n        words = torch.zeros((n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int64)\n        for slot, (src, rot) in enumerate(slots):\n            for half in (0, 1):\n                for bit, dbit in enumerate((6, 7, 8, 15)):\n                    p = (dbit + 16 * half + rot) % 32\n                    word = (0 if mask >> p & 1 else 1) if src == \'W\' else 0 if src == \'A\' else 1\n                    words[..., word] |= (codes[..., slot, half] >> bit & 1) << p\n        dst[expert] = words.to(torch.int32)\n    return dst\n\ndef m128_hh_issue_prepare(w):\n    bank = {}\n    for key in (\'w13\', \'w2\'):\n        q, sf = (w[key], w[key + \'_sf\'])\n        e, n, k16 = sf.shape\n        planes = []\n        scales = []\n        for expert in range(e):\n            code, neg, zero = m128_hh_encode_scales(sf[expert])\n            folded = q[expert].reshape(n, k16, 8) ^ neg.to(torch.uint8)[..., None] * 136\n            folded = torch.where(zero[..., None], torch.zeros_like(folded), folded)\n            planes.append(m128_hh_mw_repack_planes(folded.reshape(1, n, -1)))\n            scales.append(code)\n        bank[key] = torch.cat(planes)\n        sfbytes = torch.stack(scales)\n        bank[key + \'_sf\'] = sfbytes.reshape(e, n // 16, 2, 8, k16 // 2, 2).permute(0, 1, 4, 3, 5, 2).contiguous()\n    bank[\'w13_gs\'] = torch.cat((w.get(\'w13_gs_gate\', w[\'w13_gs\']), w.get(\'w13_gs_up\', w[\'w13_gs\']))).float().contiguous()\n    bank[\'w2_gs\'] = w[\'w2_gs\'].float().contiguous()\n    bank[\'pnw\'] = w[\'post_norm_w\'].contiguous()\n    ng = bank[\'w13\'].shape[1]\n    rr = torch.arange(ng // 2, device=w[\'w13\'].device).reshape(-1, 2)\n    order = torch.cat((rr, rr + ng // 2), 1).flatten()\n    bank[\'w13\'] = bank[\'w13\'].index_select(1, order).contiguous()\n    bank[\'w13_sf\'] = bank[\'w13_sf\'].index_select(1, order).contiguous()\n    return bank\nM128_HH_H = 4096\nM128_HH_NLOC = 32\nM128_HH_TILE_BYTES = 36864\nM128_HH_NT1, M128_HH_NT2 = (512, 256)\nM128_HH_NCNT = 64\nM128_HH_DQ_SLOTS = ((\'A\', 0), (\'A\', 4), (\'A\', 10), (\'B\', 2), (\'B\', 6), (\'B\', 12), (\'W\', 24), (\'W\', 30))\nM128_HH_LA_MASK = 1613783088\n\ndef m128_hh_bit_table():\n    """(slot, half, code bit, word, bit position) for the kernel\'s dq8 decode."""\n    dbit = {3: 15, 2: 8, 1: 7, 0: 6}\n    la = {p for p in range(32) if M128_HH_LA_MASK >> p & 1}\n    table, used = ([], {\'A\': set(), \'B\': set()})\n    for k, (src, rot) in enumerate(M128_HH_DQ_SLOTS):\n        for h in (0, 1):\n            for b in range(4):\n                p = (dbit[b] + 16 * h + rot) % 32\n                w = src if src != \'W\' else \'A\' if p in la else \'B\'\n                assert p not in used[w]\n                used[w].add(p)\n                table.append((k, h, b, w, p))\n    assert len(used[\'A\']) == 32 and len(used[\'B\']) == 32\n    return table\nM128_HH_TABLE = m128_hh_bit_table()\n\ndef m128_hh_encode_scales(sf: torch.Tensor):\n    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),\n    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for\n    every nonzero finite E4M3 value (subnormals included)."""\n    s = sf.float()\n    if not torch.isfinite(s).all():\n        raise ValueError(\'non-finite NVFP4 block scale\')\n    a = s.abs()\n    nz = a > 0\n    m, e = torch.frexp(torch.where(nz, a, torch.ones_like(a)))\n    mant = ((m * 2 - 1) * 8).round().to(torch.int32)\n    e5 = (e - 1 + 21).to(torch.int32)\n    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and (int(mant.max()) <= 7)\n    code = torch.where(nz, e5 << 3 | mant, torch.zeros_like(e5))\n    back = torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)\n    assert torch.equal(torch.where(nz, back, torch.zeros_like(back)), a)\n    return (code.to(torch.uint8), s < 0, ~nz)\n\ndef m128_hh_repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:\n    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""\n    R, K = (wq.shape[0], wq.shape[1] * 2)\n    assert K == 4096\n    T = rowmap.shape[0]\n    codes = torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)\n    u, neg, zero = m128_hh_encode_scales(sf)\n    codes = torch.where(neg[..., None], codes ^ 8, codes)\n    codes = torch.where(zero[..., None], torch.zeros_like(codes), codes)\n    codes = codes.reshape(R, K)[rowmap.reshape(-1)]\n    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)\n    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)\n    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)\n    for k, h, b, w, p in M128_HH_TABLE:\n        words[0 if w == \'A\' else 1] |= (ct[..., k, h] >> b & 1) << p\n    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()\n    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)\n    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)\n    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)\n    return torch.cat((wbytes, sbytes), 1).contiguous()\n\ndef m128_hh_issue_nt_prepare(weights):\n    dev = weights[\'w13\'].device\n    t = torch.arange(M128_HH_NT1, device=dev)\n    i = torch.arange(8, device=dev)\n    rowmap1 = torch.cat((8 * t[:, None] + i, M128_HH_H + 8 * t[:, None] + i), 1)\n    rowmap2 = torch.arange(M128_HH_H, device=dev).view(M128_HH_NT2, 16)\n    w13p = torch.empty((M128_HH_NLOC, M128_HH_NT1, M128_HH_TILE_BYTES), dtype=torch.uint8, device=dev)\n    w2p = torch.empty((M128_HH_NLOC, M128_HH_NT2, M128_HH_TILE_BYTES), dtype=torch.uint8, device=dev)\n    for e in range(M128_HH_NLOC):\n        w13p[e] = m128_hh_repack(weights[\'w13\'][e], weights[\'w13_sf\'][e], rowmap1)\n        w2p[e] = m128_hh_repack(weights[\'w2\'][e], weights[\'w2_sf\'][e], rowmap2)\n    mod = m128_hh_pm_module()\n    props = torch.cuda.get_device_properties(dev)\n    torch.cuda.synchronize()\n    return {\'w13p\': w13p, \'w2p\': w2p, \'gs13\': weights[\'w13_gs\'][:M128_HH_NLOC].float().contiguous(), \'gs2\': weights[\'w2_gs\'][:M128_HH_NLOC].float().contiguous(), \'pnw\': weights[\'post_norm_w\'].contiguous(), \'cnt\': torch.zeros((M128_HH_NCNT,), dtype=torch.int32, device=dev), \'grid\': props.multi_processor_count, \'mod\': mod}\n\ndef m128_hh_issue_word_bank(bank):\n    word = dict(bank)\n    for key in (\'w13_sf\', \'w2_sf\'):\n        word[key] = ((bank[key].to(torch.int32) << 4) + 28672).to(torch.int16)\n    return word\nM128_HH_PM_CUDA = \'#include <cuda_runtime.h>\\n#include <vector>\\n#include <stdexcept>\\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\\n// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\\n\\n\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace idist {\\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\\n unsigned dst=smaddr(d);\\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\\n}\\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\\n}\\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned long long raw,unsigned*out){\\n unsigned w,mask=0x81c081c0u;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));\\n unsigned s0=__byte_perm((unsigned)raw,0u,0x1010u);\\n unsigned s1=__byte_perm((unsigned)raw,0u,0x3232u);\\n unsigned s2=__byte_perm((unsigned)(raw>>32),0u,0x1010u);\\n unsigned s3=__byte_perm((unsigned)(raw>>32),0u,0x3232u);\\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\\n}\\n__device__ __forceinline__ void keep(const unsigned*a){\\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\\n}\\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\\n d+=b*2;\\n if constexpr(NT==16){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else if constexpr(NT==32){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else{\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }\\n#else\\n int lane=threadIdx.x&31;\\n #pragma unroll\\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\\n  float*p=c+4*j;\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n }\\n#endif\\n}\\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\\n#pragma unroll\\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n}\\ntemplate<int NT,int KT>__device__ __forceinline__ void load_stage(\\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\\n int t=threadIdx.x,w=t>>5,l=t&31;\\n #pragma unroll\\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\\n  if(l<16)async_copy(q+(g*128+w*32)*8+l*16,Q+z*64+l*4,16,1);\\n  if(l<8)async_copy(s+g*256+w*64+l*8,S+z*64+l*8,8,1);\\n }\\n #pragma unroll\\n for(int c=0;c<KT/64;++c){\\n  #pragma unroll\\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\\n  }\\n }\\n asm volatile("cp.async.commit_group;":::"memory");\\n}\\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\\n else if constexpr(S>3){\\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\\n  else if constexpr(S>4){\\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\\n   else if constexpr(S>5){\\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\\n    else if constexpr(S>6){\\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\\n     else if constexpr(S>7){\\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n}\\n__device__ __forceinline__ int publish(int*p){\\n int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;\\n}\\n__global__ void route_init(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,\\n int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){\\n int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;\\n for(int i=all;i<clear_count;i+=step)state[i]=0;\\n for(int i=all;i<M*4096/8;i+=step)reinterpret_cast<uint4*>(out)[i]=make_uint4(0,0,0,0);\\n if(blockIdx.x)return;\\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\\n for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;\\n  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}\\n }\\n for(int token=t;token<M;token+=256){int n=0;\\n  #pragma unroll\\n  for(int j=0;j<8;++j){int e=ids[token*8+j]-start;n+=(e>=0&&e<32);}\\n  state[clear_count+token]=n;\\n }\\n __syncthreads();\\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\\n  int scan=pad;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\\n  int key=n*32+31-t;\\n  #pragma unroll\\n  for(int k=2;k<=32;k*=2){\\n   #pragma unroll\\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\\n   }\\n  }\\n  // Descending keys after reversing the ascending bitonic result.\\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\\n }\\n}\\ntemplate<int NT,int KT,int S,int SPLIT,bool FIRST,int MODE>\\n__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,\\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\\n const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,\\n float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start,bool profiling){\\n constexpr int QB=64*KT/2,SB=64*KT/8,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;\\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\\n extern __shared__ __align__(128) unsigned char rawsm[];\\n unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);\\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);\\n __shared__ int last,flags[NT];\\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\\n int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;\\n int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;\\n \\n int total=TASK[32]*MT*SPLIT;\\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\\n  uint64_t tick=profiling?clock64():0;uint64_t times[6]={};bool islast=false;\\n  int task=unit/SPLIT,split=unit%SPLIT,group=task/MT,mt=task%MT,ix=0;\\n  #pragma unroll\\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\\n  int src[NT/16];\\n  #pragma unroll\\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\\n  float acc[NT/2]={};unsigned af[2][16]={};\\n  #pragma unroll\\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\\n  #pragma unroll\\n  for(int i=0;i<min(S-1,NS);++i)\\n   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\\n  // K64 has one MMA group per stage. Alternate register-A banks across\\n  // stages so wait1 retires a bank before its next conversion overwrites it.\\n  // K128/K256 already alternate within each stage; their schedule is unchanged.\\n  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");\\n  #pragma unroll 1\\n  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){\\n   #pragma unroll\\n   for(int ksub=0;ksub<(KT==64?2:1);++ksub){\\n   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\\n#endif\\n   #pragma unroll 1\\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\\n    #pragma unroll\\n    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;\\n     #pragma unroll\\n     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\\n      unsigned long long sf=*(const unsigned long long*)(ss+st*SB+z*256+w*64+(l>>2)*8);dq(q.x,q.y,sf,af[reg]+g*8);}\\n     keep(af[reg]);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\\n#endif\\n     mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);keep(af[reg^1]);\\n     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;\\n      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);}\\n    }\\n   }\\n   }\\n  }\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n  keep(af[0]);keep(af[1]);\\n  if(profiling){uint64_t now=clock64();times[0]=now-tick;tick=now;}\\n  if constexpr(SPLIT>1){\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];\\n   }\\n   __threadfence();__syncthreads();\\n   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();\\n   if(!last)continue;\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;\\n    if(r<count){\\n     #pragma unroll\\n     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];\\n    }\\n    acc[j]=sum;\\n   }\\n  }\\n  if constexpr(FIRST){\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\\n   }\\n   __syncthreads();\\n   #pragma unroll\\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\\n   }\\n  }else{\\n   float gs=GS[e]*256.f;\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\\n   }\\n   #pragma unroll\\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\\n    #pragma unroll\\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\\n   }\\n   __syncthreads();\\n   if(t<NT&&row+t<count){float sum=0;\\n    #pragma unroll\\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\\n   }\\n   if constexpr(MODE!=1){\\n   if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\\n   __threadfence();__syncthreads();\\n   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();\\n   if(profiling){uint64_t now=clock64();times[2]=now-tick;tick=now;}\\n   islast=last;\\n   if(last){\\n    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];\\n     #pragma unroll\\n     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\\n     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);\\n    }\\n    __threadfence();__syncthreads();\\n    if(profiling){uint64_t now=clock64();times[3]=now-tick;tick=now;}\\n    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;\\n      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}\\n    __syncthreads();\\n    if(profiling){uint64_t now=clock64();times[4]=now-tick;tick=now;}\\n    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;\\n     if constexpr(MODE==2){\\n     int base[8];float alpha[8];\\n     #pragma unroll\\n     for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;base[j]=-1;alpha[j]=0;\\n      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}\\n     for(int c=l*4;c<4096;c+=128){float4 sum=make_float4(0,0,0,0);\\n      #pragma unroll\\n      for(int j=0;j<8;++j)if(base[j]>=0){float4 z=*(const float4*)(Y+(size_t)base[j]*4096+c);float a=alpha[j];sum.x+=z.x*a;sum.y+=z.y*a;sum.z+=z.z*a;sum.w+=z.w*a;}\\n      __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\\n      *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\\n     }\\n     }else{\\n     for(int c=l;c<4096;c+=32){float sum=0;\\n      #pragma unroll\\n      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;\\n       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}\\n      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);\\n     }\\n     }\\n    }\\n   }\\n   }else if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\\n  }\\n  __syncthreads();\\n  if(profiling&&t==0){uint64_t*p=(uint64_t*)PART+(size_t)unit*8;\\n   for(int i=0;i<5;++i)p[i]=times[i];p[5]=clock64()-tick;p[6]=islast;p[7]=clock64();}\\n }\\n}\\n#define FX_CASES(F) \\\\\\n F(0,16,128,4,1) F(1,32,128,4,1)\\n}\\nint64_t idist_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\\n #define INIT(ID,NT,KT,S,SK) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/8)+1023)&~1023)+S*NT*KT*2;auto set=[&]<bool F,int MODE>(){WX_CUDA_CHECK(cudaFuncSetAttribute(idist::fused<NT,KT,S,SK,F,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<true,1>();set.template operator()<false,1>();if constexpr(ID==0)set.template operator()<false,0>();}\\n FX_CASES(INIT)\\n #undef INIT\\n return opt;\\n}\\nstd::vector<int64_t> idist_info(int64_t which,int64_t first,int64_t mode){\\n #define INFO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/8)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F,int MODE>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,idist::fused<NT,KT,S,SK,F,MODE>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,idist::fused<NT,KT,S,SK,F,MODE>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(first)return info.template operator()<true,1>();if(mode==1)return info.template operator()<false,1>();if constexpr(ID==0){WX_CHECK(mode==0);return info.template operator()<false,0>();}else WX_CHECK(false,"legacy control exists only forNT16");}\\n switch(which){FX_CASES(INFO) default:WX_CHECK(false,"unknown fixed configuration");}\\n #undef INFO\\n}\\nvoid idist_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear_count,int64_t stream){\\n idist::route_init<<<128,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear_count);WX_CUDA_CHECK(cudaGetLastError());\\n}\\nvoid idist_run(std::vector<int64_t> v,int64_t first,int64_t which,int64_t grid,int64_t start,int64_t mode,int64_t profiling,int64_t M,int64_t cap,int64_t stream_value){\\n WX_CHECK(v.size()==20);auto stream=(cudaStream_t)stream_value;\\n #define GO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/8)+1023)&~1023)+S*NT*KT*2; \\\\\\n auto launch=[&]<bool FIRST,int MODE>(){idist::fused<NT,KT,S,SK,FIRST,MODE><<<grid,128,bytes,stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(const __nv_bfloat16*)(void*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(const __nv_bfloat16*)(void*)v[14],(float*)v[10],(__nv_bfloat16*)(void*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)(void*)v[19],M,cap,start,profiling);};if(first)launch.template operator()<true,1>();else if(mode==1)launch.template operator()<false,1>();else{WX_CHECK(mode==0&&which==0);if constexpr(ID==0)launch.template operator()<false,0>();}break;}\\n switch(which){FX_CASES(GO) default:WX_CHECK(false,"unknown fixed configuration");}\\n #undef GO\\n WX_CUDA_CHECK(cudaGetLastError());\\n}\\nnamespace idist {\\n__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\\n int tok=blockIdx.x,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\\n __syncthreads();int pp[8];float aa[8];\\n #pragma unroll\\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\\n for(int c=t*4;c<4096;c+=1024){float4 sum=make_float4(0,0,0,0);\\n  #pragma unroll\\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\\n }\\n}\\n}\\nvoid idist_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){idist::finish<<<M,256,0,(cudaStream_t)stream>>>((int*)v[5],(int*)v[11],(float*)v[12],(int*)v[13],(float*)v[15],(float*)v[16],(__nv_bfloat16*)(void*)v[19],M,start);WX_CUDA_CHECK(cudaGetLastError());}\\n\\n#include <cuda_runtime.h>\\n#include <vector>\\n#include <stdexcept>\\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\\n\\n\\n\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace pmnt {\\n\\nconstexpr int HD = 4096;\\nconstexpr int NLOC = 32;\\nconstexpr int TILE_BYTES = 36864;\\nconstexpr int TILE_W_BYTES = 32768;\\nconstexpr int NT1 = 512;\\nconstexpr int NT2 = 256;\\nconstexpr int NCW = 8;                 // consumer warps\\nconstexpr int NCT = NCW * 32;          // consumer threads\\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\\nconstexpr uint32_t LA_MASK = 0x60306030u;\\n// persistent counters\\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\\n\\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\\n}\\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\\n}\\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\\n  uint32_t ok;\\n  asm volatile("{\\\\n .reg .pred p;\\\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\\\n selp.u32 %0, 1, 0, p;\\\\n}\\\\n"\\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\\n  return ok != 0;\\n}\\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\\n  while (!mbar_try(b, parity)) {}\\n}\\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\\n}\\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\\n}\\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\\n}\\n__device__ __forceinline__ void bar_consumers() {\\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\\n}\\n__device__ __forceinline__ int ld_acquire(const int* p) {\\n  int v;\\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\\n  return v;\\n}\\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\\n  uint32_t d;\\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\\n  return d;\\n}\\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\\n                                         uint32_t b0, uint32_t b1) {\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\\n}\\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\\n  o[0] = A & DQ_MASK;\\n  o[1] = rotr(A, 4) & DQ_MASK;\\n  o[2] = rotr(A, 10) & DQ_MASK;\\n  o[3] = rotr(B, 2) & DQ_MASK;\\n  o[4] = rotr(B, 6) & DQ_MASK;\\n  o[5] = rotr(B, 12) & DQ_MASK;\\n  uint32_t W;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(W):"r"(A),"r"(B));\\n  o[6] = rotr(W, 24) & DQ_MASK;\\n  o[7] = rotr(W, 30) & DQ_MASK;\\n}\\n\\nstruct PrefetchState {\\n  volatile int index, stage, phase, pass;\\n  volatile int e, nb, m, mb, me;\\n  volatile uint64_t pol_first, pol_normal;\\n};\\n\\nstruct Route {        // shared-memory routing summary\\n  int n[NLOC];        // pairs per local expert\\n  int off[NLOC];      // first sorted position\\n  int nblk[NLOC];     // token blocks\\n  int us1[NLOC + 1];  // phase-1 unit prefix\\n  int us2[NLOC + 1];  // phase-2 unit prefix\\n  int c1, c2;         // m-chunk sizes\\n  int run[NLOC];\\n  int wc[NCW][NLOC];\\n  PrefetchState pre;\\n};\\n\\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\\n                                            int& e, int& nb, int& m) {\\n  e = 0;\\n#pragma unroll\\n  for (int d = 16; d > 0; d >>= 1) {\\n    if (us[e + d] <= u) e += d;\\n  }\\n  const int rel = u - us[e];\\n  const int nbk = nblk[e];\\n  const int nfull = ntile / cc;\\n  const int span = cc * nbk;\\n  if (rel < nfull * span) {\\n    const int mc = rel / span;\\n    const int r2 = rel - mc * span;\\n    nb = r2 / cc;\\n    m = mc * cc + (r2 - nb * cc);\\n  } else {\\n    const int L = ntile - nfull * cc;\\n    const int r3 = rel - nfull * span;\\n    nb = r3 / L;\\n    m = nfull * cc + (r3 - nb * L);\\n  }\\n}\\n\\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\\n                                               const int* nblk, int cc, int ntile) {\\n  if (++m == me) {\\n    if (++nb == nblk[e]) {\\n      nb = 0;\\n      mb += cc;\\n      if (mb >= ntile) {\\n        ++e;\\n        while (e < NLOC && nblk[e] == 0) ++e;\\n        mb = 0;\\n      }\\n      me = min(ntile, mb + cc);\\n    }\\n    m = mb;\\n  }\\n}\\n\\n\\n__device__ __forceinline__ void prefetch_one(\\n    Route* rt, int total, int first, int a2, int nstage,\\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\\n    const uint8_t* w13p, const uint8_t* w2p) {\\n  auto& p = rt->pre;\\n  int idx = p.index;\\n  if (idx >= total) return;\\n  int st=p.stage, ph=p.phase, pass=p.pass;\\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\\n  const int ntile=pass?NT2:NT1;\\n  const int cc=pass?rt->c2:rt->c1;\\n  const uint8_t* base=pass?w2p:w13p;\\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\\n  mbar_wait(&emptyb[st], ph^1u);\\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\\n  ++idx;\\n  if (idx < total) {\\n    if (idx == first) {\\n      pass=1;\\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\\n      mb=(m/rt->c2)*rt->c2;\\n      me=min(NT2,mb+rt->c2);\\n    } else {\\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\\n    }\\n  }\\n  if (++st == nstage) { st=0; ph^=1; }\\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\\n}\\n\\ntemplate<int NT,bool FIRST,bool SMALL>\\n__global__ void __launch_bounds__(NTHR,2) phase(\\n const __nv_bfloat16*x,const __nv_bfloat16*pnw,const uint8_t*w13p,const uint8_t*w2p,\\n const float*gs13,const float*gs2,const int*C,const int*OFF,const int*SLOTS,\\n __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){\\n constexpr int NB=8*NT,RED=NT*4;\\n extern __shared__ __align__(128) uint8_t smem[];\\n uint8_t*stages=smem;float*red=(float*)(smem+nstage*TILE_BYTES);\\n uint64_t*fullb=(uint64_t*)(red+2*NCW*32*RED),*emptyb=fullb+nstage;\\n Route*rt=(Route*)(emptyb+nstage);\\n const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;\\n if(tid==0){for(int z=0;z<nstage;++z){mbar_init(fullb+z,1);mbar_init(emptyb+z,NCW);}\\n  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\\n if(warp==0){int n=C[lane];int nb=(n+NB-1)/NB;if constexpr(SMALL)if(n>16)nb=0;\\n  rt->n[lane]=n;rt->off[lane]=OFF[lane];rt->nblk[lane]=nb;\\n  int u1=nb*NT1,u2=nb*NT2;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,u1,d),b=__shfl_up_sync(~0u,u2,d);if(lane>=d){u1+=a;u2+=b;}}\\n  rt->us1[lane+1]=u1;rt->us2[lane+1]=u2;\\n  if(lane==0){rt->us1[0]=rt->us2[0]=0;}\\n  int U1=__shfl_sync(~0u,u1,31),U2=__shfl_sync(~0u,u2,31);\\n  if(lane==0){rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));}\\n }\\n __syncthreads();\\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\\n\\n  const int first_count=FIRST?b1-a1:0,total_count=FIRST?first_count:b2-a2;\\n  if (tid == 0) {\\n    uint64_t pf,pn;\\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\\n    if (total_count) {\\n      int e,nb,m;\\n      const int pass=first_count?0:1;\\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\\n    }\\n    for (int i=0;i<nstage;++i)\\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n  }\\n  __syncthreads();\\n\\n  const int q = lane & 3;\\n  uint32_t Bf[NT][8][8];\\n  int st = 0;\\n  uint32_t ph = 0;\\n  int buf = 0;\\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\\n  float fw[2];   // (1 + post_norm_w) for this thread\\\'s phase-2 epilogue row, refreshed per tile\\n\\n  for (int pass = FIRST?0:1; pass < (FIRST?1:2); ++pass) {\\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\\n    const int* us = pass ? rt->us2 : rt->us1;\\n    const int cc = pass ? rt->c2 : rt->c1;\\n    const int ntile = pass ? NT2 : NT1;\\n    int e=0, nb=0, m=0, mb=0, me=0;\\n    if (ua < ub) {\\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\\n      mb = (m / cc) * cc;\\n      me = min(ntile, mb + cc);\\n    }\\n    for (int u = ua; u < ub; ++u) {\\n      const int ne = rt->n[e], off = rt->off[e];\\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\\n        cur_pass=pass;cur_e=e;cur_nb=nb;seg_tiles=0;\\n#pragma unroll\\n        for (int j = 0; j < NT; ++j) {\\n          const int idx = nb * NB + j * 8 + (lane >> 2);\\n          const __nv_bfloat16* row = nullptr;\\n          if (idx < ne) {\\n            const int pos = off + idx;\\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(SLOTS[e*M+idx] >> 3) * HD;\\n          }\\n#pragma unroll\\n          for (int g = 0; g < 8; ++g) {\\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\\n            if (row) {\\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\\n            }\\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\\n          }\\n        }\\n      }\\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\\n\\n      mbar_wait(&fullb[st], ph);\\n      const uint8_t* tile = stages + st * TILE_BYTES;\\n      float acc[NT][4];\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\\n      {\\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\\n#pragma unroll\\n        for (int g = 0; g < 8; ++g) {\\n          const uint4 wv = wq[g * 32];\\n          const uint32_t sw = scw[g >> 1];\\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\\n          uint32_t o[16];\\n          dq8(wv.x, wv.y, o);\\n          dq8(wv.z, wv.w, o + 8);\\n#pragma unroll\\n          for (int s = 0; s < 4; ++s) {\\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\\n#pragma unroll\\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\\n          }\\n        }\\n      }\\n      __syncwarp();\\n      if (lane == 0) mbar_arrive(&emptyb[st]);\\n      if (++st == nstage) { st = 0; ph ^= 1u; }\\n\\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\\n      bar_consumers();\\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n      const float* rb = red + buf * NCW * 32 * RED;\\n      if (pass == 0) {\\n        const float sc = gs13[e] * 256.f;\\n        if (tid < 8 * NB) {\\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\\n          float G1 = 0.f, U = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) {\\n            G1 += rb[(w * 32 + ls) * RED + rg];\\n            U += rb[(w * 32 + ls) * RED + rg + 2];\\n          }\\n          const int idx = nb * NB + n;\\n          if (idx < ne) {\\n            G1 *= sc;\\n            U *= gs13[e+32]*256.f;\\n            G1=__bfloat162float(__float2bfloat16_rn(G1));\\n            U=__bfloat162float(__float2bfloat16_rn(U));\\n            const float h = G1 / (1.f + __expf(-G1)) * U;\\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\\n          }\\n        }\\n      } else {\\n        const float sc = gs2[e] * 256.f;\\n        if (tid < 16 * NB) {\\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\\n          float v = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\\n          v *= sc;\\n          const int idx = nb * NB + n;\\n          const bool ok = idx < ne;\\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v*fw[0];\\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\\n#pragma unroll\\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\\n          if (ok && i == 0) atomicAdd(ss + (size_t)(off+idx)*64+(m>>2), t2);\\n        }\\n      }\\n      buf ^= 1;\\n      ++seg_tiles;\\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\\n    }\\n  }\\n}\\n\\nsize_t smem_bytes(int nt, int nstage, int M) {\\n  const int npairs = M * 8;\\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\\n}\\n\\n}  // namespace\\n\\n\\n\\nint64_t pm_nt_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\\n auto set=[&]<bool F,bool S>(){cudaFuncAttributes a{};WX_CUDA_CHECK(cudaFuncGetAttributes(&a,pmnt::phase<1,F,S>));WX_CUDA_CHECK(cudaFuncSetAttribute(pmnt::phase<1,F,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));};\\n set.template operator()<true,false>();set.template operator()<false,false>();set.template operator()<true,true>();set.template operator()<false,true>();return opt;}\\nint64_t pm_nt_smem(int64_t stages,int64_t M){return pmnt::smem_bytes(1,stages,M);}\\nint64_t pm_nt_resident(int64_t stages,int64_t M,int64_t first,int64_t small){int n;\\n auto get=[&]<bool F,bool S>(){WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmnt::phase<1,F,S>,pmnt::NTHR,pmnt::smem_bytes(1,stages,M)));};\\n if(first){if(small)get.template operator()<true,true>();else get.template operator()<true,false>();}else{if(small)get.template operator()<false,true>();else get.template operator()<false,false>();}return n;}\\nvoid pm_nt_run(std::vector<int64_t>v,int64_t first,int64_t small,int64_t M,int64_t stages,int64_t grid,int64_t stream){\\n auto launch=[&]<bool F,bool S>(){pmnt::phase<1,F,S><<<grid,pmnt::NTHR,pmnt::smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\\n (__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(uint8_t*)v[2],(uint8_t*)v[3],(float*)v[4],(float*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11],M,stages);};\\n if(first){if(small)launch.template operator()<true,true>();else launch.template operator()<true,false>();}else{if(small)launch.template operator()<false,true>();else launch.template operator()<false,false>();}WX_CUDA_CHECK(cudaGetLastError());}\\n\\n\\n#include <cuda_runtime.h>\\n#include <vector>\\n__global__ void pm_meta_kernel(const int*C,const int*ORDER,int*T32,int*T128,int*T16,float*SS,int cap){\\n int t=threadIdx.x,idx=blockIdx.x*blockDim.x+t;\\n for(int i=idx;i<cap*64;i+=gridDim.x*blockDim.x)SS[i]=0.f;\\n if(blockIdx.x==0&&t<32){int n=C[ORDER[t]],v32=n>16?(n+31)/32:0,v128=(n+127)/128,v16=(n+15)/16;\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v32,d),b=__shfl_up_sync(~0u,v128,d),c=__shfl_up_sync(~0u,v16,d);if(t>=d){v32+=a;v128+=b;v16+=c;}}\\n  T32[t+1]=v32;T128[t+1]=v128;T16[t+1]=v16;if(t==0){T32[0]=0;T128[0]=0;T16[0]=0;}\\n }\\n}\\nvoid pm_meta(std::vector<int64_t>v,int64_t cap,int64_t stream){pm_meta_kernel<<<128,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(float*)v[5],cap);}\\n\'\nM128_HH_PM_CPP = \'#include <pybind11/pybind11.h>\\n#include <pybind11/stl.h>\\n#include <vector>\\n#include <cstdint>\\nint64_t idist_init();std::vector<int64_t> idist_info(int64_t,int64_t,int64_t);void idist_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void idist_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void idist_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);\\nint64_t pm_nt_init();int64_t pm_nt_smem(int64_t,int64_t);int64_t pm_nt_resident(int64_t,int64_t,int64_t,int64_t);void pm_nt_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void pm_meta(std::vector<int64_t>,int64_t,int64_t);\\n\'\nM128_HH_PM_FUNCTIONS = [\'idist_init\', \'idist_info\', \'idist_route\', \'idist_run\', \'idist_finish\', \'pm_nt_init\', \'pm_nt_smem\', \'pm_nt_resident\', \'pm_nt_run\', \'pm_meta\']\nM128_HH_ISSUE_OPTS = {}\n\n@functools.lru_cache(None)\ndef m128_hh_pm_module():\n    arch = \'90a\' if torch.cuda.get_device_capability() == (9, 0) else \'120\'\n    mod = load_inline(name=\'m128_phase_mix_\' + hashlib.sha256(M128_HH_PM_CUDA.encode()).hexdigest()[:12], cpp_sources=M128_HH_PM_CPP, cuda_sources=M128_HH_PM_CUDA, functions=M128_HH_PM_FUNCTIONS, no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=[\'-O3\', \'-std=c++20\', \'-lineinfo\', \'-gencode=arch=compute_\' + arch + \',code=sm_\' + arch], verbose=False)\n    for p in (\'idist\', \'pm_nt\'):\n        M128_HH_ISSUE_OPTS[p] = getattr(mod, p + \'_init\')()\n    return mod\n\ndef m128_hh_pm_plan(data, base, word, native):\n    x = data[\'hidden_states\']\n    m = x.shape[0]\n    dev = x.device\n    nt = 32\n    cap = m * 8 + 32 * 128\n    c = torch.empty(32, device=dev, dtype=torch.int32)\n    off = torch.empty(33, device=dev, dtype=torch.int32)\n    order = torch.empty_like(c)\n    tasks = [torch.empty_like(off) for _ in range(4)]\n    slots = torch.empty((32, m), device=dev, dtype=torch.int32)\n    ids = data[\'topk_ids\'].int().contiguous()\n    tw = data[\'topk_weights\'].float().contiguous()\n    pos = torch.empty_like(ids)\n    clear = cap // nt * 193 + m\n    state = torch.empty(clear + m, device=dev, dtype=torch.int32)\n    act = torch.empty((cap, 4096), device=dev, dtype=torch.bfloat16)\n    part = torch.empty(1, device=dev)\n    y = torch.empty((cap, 4096), device=dev)\n    ss = torch.empty((cap, 64), device=dev)\n    inv = torch.empty(cap, device=dev)\n    out = torch.empty_like(x)\n    common = [c, off, order, tasks[0], slots, act]\n    tail = [ids, tw, pos, base[\'pnw\'], y, ss, inv, state, out]\n\n    def vector(bank, phase, task):\n        v = [bank[\'w13\' if phase else \'w2\'], bank[\'w13_sf\' if phase else \'w2_sf\'], bank[\'w13_gs\' if phase else \'w2_gs\'], x if phase else act] + common + [part] + tail\n        v[7] = task\n        return v\n    word13 = vector(word, 1, tasks[0])\n    word2 = vector(word, 0, tasks[0])\n    big13 = vector(word, 1, tasks[1])\n    big2 = vector(word, 0, tasks[1])\n    nv = [x, base[\'pnw\'], native[\'w13p\'], native[\'w2p\'], base[\'w13_gs\'], base[\'w2_gs\'], c, off, slots, act, y, ss]\n    vectors = [word13, word2, big13, big2, nv]\n    ptrs = [[v.data_ptr() for v in row] for row in vectors]\n    rp = [v.data_ptr() for v in (ids, c, off, order, tasks[0], slots, pos, state, out)]\n    mp = [v.data_ptr() for v in (c, order, tasks[1], tasks[2], tasks[3], ss)]\n    mod = m128_hh_pm_module()\n    start = data[\'config\'].get(\'local_start\', 0)\n    owners = (vectors, ids, tw, tasks)\n    stages = 2\n    while stages > 1 and mod.pm_nt_smem(stages, m) > M128_HH_ISSUE_OPTS[\'pm_nt\']:\n        stages -= 1\n\n    def stream():\n        return torch.cuda.current_stream().cuda_stream\n\n    def route():\n        _ = owners\n        mod.idist_route(rp, m, nt, start, clear, stream())\n        mod.pm_meta(mp, cap, stream())\n\n    def phase(which):\n        _ = owners\n        small = 1\n        grid = native[\'grid\'] * mod.pm_nt_resident(stages, m, which, small)\n        mod.pm_nt_run(ptrs[4], which, small, m, stages, grid, stream())\n        mod.idist_run(ptrs[2 if which else 3], which, 1, 1056, start, 1, 0, m, cap, stream())\n\n    def w13():\n        phase(1)\n\n    def w2():\n        phase(0)\n\n    def finish():\n        _ = owners\n        mod.idist_finish(ptrs[1], m, start, stream())\n        return out\n    return dict(steps=(route, w13, w2, finish), out=out, owners=owners, v13=word13, v2=word2, tasks=tasks)\n\ndef m128_hh_pm_call(data, base, word, native):\n    for fn in m128_hh_pm_plan(data, base, word, native)[\'steps\']:\n        out = fn()\n    return out\n\ndef hh_prepare(weights):\n    w=dict(weights,post_norm_w=weights.get(\'post_expert_norm\',weights.get(\'post_norm_w\')))\n    base=m128_hh_issue_prepare(w)\n    return dict(base=base,word=m128_hh_issue_word_bank(base),native=m128_hh_issue_nt_prepare(w))\n\ndef hh_experts(x,scores,local_ids,state,local_start=0):\n    assert x.shape[0]==128, \'H_H cell is selected only for M128\'\n    data=dict(hidden_states=x,topk_ids=local_ids,topk_weights=scores,config=dict(local_start=local_start))\n    return m128_hh_pm_call(data,state[\'base\'],state[\'word\'],state[\'native\'])\n\nimport torch\nimport hashlib\nfrom torch.utils.cpp_extension import load_inline\nNT1_SOURCES={\'lop3\': \'\\n#define C10_CUDA_CHECK(expr) do{cudaError_t e=(expr);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)\\n#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())\\n\\n#include <cuda_runtime.h>\\n#include <stdexcept>\\n#include <vector>\\n\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace {\\n\\nconstexpr int HD = 4096;\\nconstexpr int NLOC = 32;\\nconstexpr int TILE_BYTES = 36864;\\nconstexpr int TILE_W_BYTES = 32768;\\nconstexpr int NT1 = 512;\\nconstexpr int NT2 = 256;\\nconstexpr int NCW = 8;                 // consumer warps\\nconstexpr int NCT = NCW * 32;          // consumer threads\\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\\nconstexpr uint32_t LA_MASK = 0x60306030u;\\n// persistent counters\\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\\n\\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\\n}\\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\\n}\\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\\n  uint32_t ok;\\n  asm volatile("{\\\\n .reg .pred p;\\\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\\\n selp.u32 %0, 1, 0, p;\\\\n}\\\\n"\\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\\n  return ok != 0;\\n}\\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\\n  while (!mbar_try(b, parity)) {}\\n}\\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\\n}\\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\\n}\\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\\n}\\n__device__ __forceinline__ void bar_consumers() {\\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\\n}\\n__device__ __forceinline__ int ld_acquire(const int* p) {\\n  int v;\\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\\n  return v;\\n}\\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\\n  uint32_t d;\\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\\n  return d;\\n}\\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\\n                                         uint32_t b0, uint32_t b1) {\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\\n}\\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\\n  o[0] = A & DQ_MASK;\\n  o[1] = rotr(A, 4) & DQ_MASK;\\n  o[2] = rotr(A, 10) & DQ_MASK;\\n  o[3] = rotr(B, 2) & DQ_MASK;\\n  o[4] = rotr(B, 6) & DQ_MASK;\\n  o[5] = rotr(B, 12) & DQ_MASK;\\n  uint32_t W; asm("lop3.b32 %0,%1,%2,%3,0xe4;":"=r"(W):"r"(A),"r"(B),"n"(LA_MASK));\\n  o[6] = rotr(W, 24) & DQ_MASK;\\n  o[7] = rotr(W, 30) & DQ_MASK;\\n}\\n\\nstruct PrefetchState {\\n  volatile int index, stage, phase, pass;\\n  volatile int e, nb, m, mb, me;\\n  volatile uint64_t pol_first, pol_normal;\\n};\\n\\nstruct Route {        // shared-memory routing summary\\n  int n[NLOC];        // pairs per local expert\\n  int off[NLOC];      // first sorted position\\n  int nblk[NLOC];     // token blocks\\n  int us1[NLOC + 1];  // phase-1 unit prefix\\n  int us2[NLOC + 1];  // phase-2 unit prefix\\n  int c1, c2;         // m-chunk sizes\\n  int run[NLOC];\\n  int wc[NCW][NLOC];\\n  PrefetchState pre;\\n};\\n\\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\\n                                            int& e, int& nb, int& m) {\\n  e = 0;\\n#pragma unroll\\n  for (int d = 16; d > 0; d >>= 1) {\\n    if (us[e + d] <= u) e += d;\\n  }\\n  const int rel = u - us[e];\\n  const int nbk = nblk[e];\\n  const int nfull = ntile / cc;\\n  const int span = cc * nbk;\\n  if (rel < nfull * span) {\\n    const int mc = rel / span;\\n    const int r2 = rel - mc * span;\\n    nb = r2 / cc;\\n    m = mc * cc + (r2 - nb * cc);\\n  } else {\\n    const int L = ntile - nfull * cc;\\n    const int r3 = rel - nfull * span;\\n    nb = r3 / L;\\n    m = nfull * cc + (r3 - nb * L);\\n  }\\n}\\n\\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\\n                                               const int* nblk, int cc, int ntile) {\\n  if (++m == me) {\\n    if (++nb == nblk[e]) {\\n      nb = 0;\\n      mb += cc;\\n      if (mb >= ntile) {\\n        ++e;\\n        while (e < NLOC && nblk[e] == 0) ++e;\\n        mb = 0;\\n      }\\n      me = min(ntile, mb + cc);\\n    }\\n    m = mb;\\n  }\\n}\\n\\n\\n__device__ __forceinline__ void prefetch_one(\\n    Route* rt, int total, int first, int a2, int nstage,\\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\\n    const uint8_t* w13p, const uint8_t* w2p) {\\n  auto& p = rt->pre;\\n  int idx = p.index;\\n  if (idx >= total) return;\\n  int st=p.stage, ph=p.phase, pass=p.pass;\\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\\n  const int ntile=pass?NT2:NT1;\\n  const int cc=pass?rt->c2:rt->c1;\\n  const uint8_t* base=pass?w2p:w13p;\\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\\n  mbar_wait(&emptyb[st], ph^1u);\\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\\n  ++idx;\\n  if (idx < total) {\\n    if (idx == first) {\\n      pass=1;\\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\\n      mb=(m/rt->c2)*rt->c2;\\n      me=min(NT2,mb+rt->c2);\\n    } else {\\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\\n    }\\n  }\\n  if (++st == nstage) { st=0; ph^=1; }\\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\\n}\\n\\ntemplate <int NT>\\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\\n    int M, int local_start, int nstage) {\\n  constexpr int NB = 8 * NT;\\n  constexpr int RED = NT * 4;\\n  const int npairs = M * 8;\\n  extern __shared__ __align__(128) uint8_t smem[];\\n  uint8_t* stages = smem;\\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\\n  uint64_t* emptyb = fullb + nstage;\\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\\n\\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\\n  const int G = gridDim.x, c = blockIdx.x;\\n  int bar_gen0 = 0;\\n\\n  // ---------------- routing: counts ----------------\\n  if (tid < NLOC) rt->n[tid] = 0;\\n  if (tid == 0) {\\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\\n    bar_gen0 = ld_acquire(cnt + C_GEN);\\n  }\\n  __syncthreads();\\n  for (int i = tid; i < npairs; i += NTHR) {\\n    int e = __ldg(ids + i) - local_start;\\n    if (e < 0 || e >= NLOC) e = -1;\\n    eflat[i] = static_cast<int8_t>(e);\\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\\n  }\\n  __syncthreads();\\n  if (warp == 0) {\\n    const int n = rt->n[lane];\\n    const int nb = (n + NB - 1) / NB;\\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\\n#pragma unroll\\n    for (int d = 1; d < 32; d <<= 1) {\\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\\n    }\\n    rt->off[lane] = off - n;\\n    rt->run[lane] = off - n;\\n    rt->nblk[lane] = nb;\\n    rt->us1[lane + 1] = u1;\\n    rt->us2[lane + 1] = u2;\\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\\n    if (lane == 0) {\\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\\n    }\\n  }\\n  __syncthreads();\\n\\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\\n\\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\\n  if (tid == 0) {\\n    uint64_t pf,pn;\\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\\n    if (total_count) {\\n      int e,nb,m;\\n      const int pass=first_count?0:1;\\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\\n    }\\n    for (int i=0;i<nstage;++i)\\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n  }\\n  __syncthreads();\\n\\n  // ---------------- consumers: deterministic token lists ----------------\\n  for (int base = 0; base < npairs; base += NCT) {\\n    rt->wc[warp][lane] = 0;\\n    bar_consumers();\\n    const int i = base + tid;\\n    const int e = i < npairs ? eflat[i] : -1;\\n    const unsigned mk = __match_any_sync(~0u, e);\\n    const int rank = __popc(mk & ((1u << lane) - 1u));\\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\\n    bar_consumers();\\n    if (tid < NLOC) {\\n      int s = rt->run[tid];\\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\\n      rt->run[tid] = s;\\n    }\\n    bar_consumers();\\n    if (i < npairs) {\\n      if (e >= 0) {\\n        const int pos = rt->wc[warp][e] + rank;\\n        pos_of[i] = static_cast<int16_t>(pos);\\n        pair_of[pos] = static_cast<int16_t>(i);\\n      } else {\\n        pos_of[i] = -1;\\n      }\\n    }\\n  }\\n  // zero this CTA\\\'s slice of the sum-of-squares accumulators, then publish\\n  {\\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\\n    __threadfence();\\n    bar_consumers();\\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\\n  }\\n\\n  const int q = lane & 3;\\n  uint32_t Bf[NT][8][8];\\n  int st = 0;\\n  uint32_t ph = 0;\\n  int buf = 0;\\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\\n  float fw[2];   // (1 + post_norm_w) for this thread\\\'s phase-2 epilogue row, refreshed per tile\\n\\n  for (int pass = 0; pass < 2; ++pass) {\\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\\n    const int* us = pass ? rt->us2 : rt->us1;\\n    const int cc = pass ? rt->c2 : rt->c1;\\n    const int ntile = pass ? NT2 : NT1;\\n    int e=0, nb=0, m=0, mb=0, me=0;\\n    if (ua < ub) {\\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\\n      mb = (m / cc) * cc;\\n      me = min(ntile, mb + cc);\\n    }\\n    for (int u = ua; u < ub; ++u) {\\n      const int ne = rt->n[e], off = rt->off[e];\\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\\n          __threadfence();\\n          bar_consumers();\\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\\n        }\\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\\n        if (pass == 1) {\\n          const int target = NT1 * rt->nblk[e];\\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\\n        }\\n#pragma unroll\\n        for (int j = 0; j < NT; ++j) {\\n          const int idx = nb * NB + j * 8 + (lane >> 2);\\n          const __nv_bfloat16* row = nullptr;\\n          if (idx < ne) {\\n            const int pos = off + idx;\\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\\n          }\\n#pragma unroll\\n          for (int g = 0; g < 8; ++g) {\\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\\n            if (row) {\\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\\n            }\\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\\n          }\\n        }\\n      }\\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\\n\\n      mbar_wait(&fullb[st], ph);\\n      const uint8_t* tile = stages + st * TILE_BYTES;\\n      float acc[NT][4];\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\\n      {\\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\\n#pragma unroll\\n        for (int g = 0; g < 8; ++g) {\\n          const uint4 wv = wq[g * 32];\\n          const uint32_t sw = scw[g >> 1];\\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\\n          uint32_t o[16];\\n          dq8(wv.x, wv.y, o);\\n          dq8(wv.z, wv.w, o + 8);\\n#pragma unroll\\n          for (int s = 0; s < 4; ++s) {\\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\\n#pragma unroll\\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\\n          }\\n        }\\n      }\\n      __syncwarp();\\n      if (lane == 0) mbar_arrive(&emptyb[st]);\\n      if (++st == nstage) { st = 0; ph ^= 1u; }\\n\\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\\n      bar_consumers();\\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n      const float* rb = red + buf * NCW * 32 * RED;\\n      if (pass == 0) {\\n        const float sc = gs13[e] * 256.f;\\n        if (tid < 8 * NB) {\\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\\n          float G1 = 0.f, U = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) {\\n            G1 += rb[(w * 32 + ls) * RED + rg];\\n            U += rb[(w * 32 + ls) * RED + rg + 2];\\n          }\\n          const int idx = nb * NB + n;\\n          if (idx < ne) {\\n            G1 *= sc;\\n            U *= sc;\\n            const float h = G1 / (1.f + __expf(-G1)) * U;\\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\\n          }\\n        }\\n      } else {\\n        const float sc = gs2[e] * 256.f;\\n        if (tid < 16 * NB) {\\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\\n          float v = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\\n          v *= sc;\\n          const int idx = nb * NB + n;\\n          const bool ok = idx < ne;\\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\\n#pragma unroll\\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\\n        }\\n      }\\n      buf ^= 1;\\n      ++seg_tiles;\\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\\n    }\\n  }\\n  if (cur_pass == 0 && seg_tiles > 0) {\\n    __threadfence();\\n    bar_consumers();\\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\\n  }\\n\\n  // ---------------- grid barrier ----------------\\n  __threadfence();\\n  bar_consumers();\\n  if (tid == 0) {\\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\\n      cnt[C_ZERO] = 0;\\n      cnt[C_BAR] = 0;\\n      __threadfence();\\n      atomicAdd(cnt + C_GEN, 1);\\n    } else {\\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\\n    }\\n  }\\n  bar_consumers();\\n\\n  // ---------------- combine: item = (token, half of the columns) ----------------\\n  for (int it = c; it < 2 * M; it += G) {\\n    const int t = it >> 1;\\n    const int c0 = (it & 1) * 2048 + tid * 8;\\n    float f[8];\\n    {\\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\\n#pragma unroll\\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\\n    }\\n    float acc[8];\\n#pragma unroll\\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\\n#pragma unroll\\n    for (int j = 0; j < 8; ++j) {\\n      const int pos = pos_of[t * 8 + j];\\n      if (pos >= 0) {\\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\\n      }\\n    }\\n    uint4 ov;\\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\\n#pragma unroll\\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\\n  }\\n}\\n\\nsize_t smem_bytes(int nt, int nstage, int M) {\\n  const int npairs = M * 8;\\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\\n}\\n\\n}  // namespace\\n\\nint64_t max_smem_attr() {\\n  int dev, optin;\\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\\n  return optin;\\n}\\n\\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\\n  int n;\\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\\n  return n;\\n}\\n\\nvoid run(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stages,int64_t grid,int64_t stream){\\n moe_fused<1><<<grid,NTHR,smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\\n (const __nv_bfloat16*)v[0],(const int*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],\\n (const uint8_t*)v[4],(const uint8_t*)v[5],(const float*)v[6],(const float*)v[7],\\n (__nv_bfloat16*)v[8],(float*)v[9],(float*)v[10],(int*)v[11],(__nv_bfloat16*)v[12],M,start,stages);\\n C10_CUDA_KERNEL_LAUNCH_CHECK();\\n}\\n\'}\nH = 4096\n\nNLOC = 32\n\nTOPK = 8\n\nTILE_BYTES = 36864\n\nNT1, NT2 = 512, 256\n\nNCNT = 64\n\n_DQ_SLOTS = (("A", 0), ("A", 4), ("A", 10), ("B", 2), ("B", 6), ("B", 12), ("W", 24), ("W", 30))\n\n_LA_MASK = 0x60306030\n\ndef _bit_table():\n    """(slot, half, code bit, word, bit position) for the kernel\'s dq8 decode."""\n    dbit = {3: 15, 2: 8, 1: 7, 0: 6}\n    la = {p for p in range(32) if (_LA_MASK >> p) & 1}\n    table, used = [], {"A": set(), "B": set()}\n    for k, (src, rot) in enumerate(_DQ_SLOTS):\n        for h in (0, 1):\n            for b in range(4):\n                p = (dbit[b] + 16 * h + rot) % 32\n                w = src if src != "W" else ("A" if p in la else "B")\n                assert p not in used[w]\n                used[w].add(p)\n                table.append((k, h, b, w, p))\n    assert len(used["A"]) == 32 and len(used["B"]) == 32\n    return table\n\n_TABLE = _bit_table()\n\ndef _encode_scales(sf: torch.Tensor):\n    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),\n    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for\n    every nonzero finite E4M3 value (subnormals included)."""\n    s = sf.float()\n    if not torch.isfinite(s).all():\n        raise ValueError("non-finite NVFP4 block scale")\n    a = s.abs()\n    nz = a > 0\n    m, e = torch.frexp(torch.where(nz, a, torch.ones_like(a)))\n    mant = ((m * 2 - 1) * 8).round().to(torch.int32)\n    e5 = (e - 1 + 21).to(torch.int32)\n    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and int(mant.max()) <= 7\n    code = torch.where(nz, (e5 << 3) | mant, torch.zeros_like(e5))\n    back = torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)\n    assert torch.equal(torch.where(nz, back, torch.zeros_like(back)), a)\n    return code.to(torch.uint8), s < 0, ~nz\n\ndef _repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:\n    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""\n    R, K = wq.shape[0], wq.shape[1] * 2\n    assert K == 4096\n    T = rowmap.shape[0]\n    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)\n    u, neg, zero = _encode_scales(sf)\n    codes = torch.where(neg[..., None], codes ^ 8, codes)                  # sign moved into the codes\n    codes = torch.where(zero[..., None], torch.zeros_like(codes), codes)   # 0 * anything stays 0\n    codes = codes.reshape(R, K)[rowmap.reshape(-1)]\n    # k = 512 w + 64 g + 16 q + 4 (2 p + sl) + 2 j + h ; tile row = 8 ri + rr\n    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                           # T ri rr w g q p sl j h\n    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)\n    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)\n    for k, h, b, w, p in _TABLE:\n        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p\n    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()                    # T w g lane p {A,B}\n    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)\n    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)                       # T ri rr w g q\n    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)                  # T w (rr q) g ri\n    return torch.cat((wbytes, sbytes), 1).contiguous()\n\ndef _raw_repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:\n    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""\n    R, K = wq.shape[0], wq.shape[1] * 2\n    assert K == 4096\n    T = rowmap.shape[0]\n    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)\n    u = sf.contiguous().view(torch.uint8)  # exact stored E4M3 bytes\n    codes = codes.reshape(R, K)[rowmap.reshape(-1)]\n    # k = 512 w + 64 g + 16 q + 4 (2 p + sl) + 2 j + h ; tile row = 8 ri + rr\n    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                           # T ri rr w g q p sl j h\n    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)\n    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)\n    for k, h, b, w, p in _TABLE:\n        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p\n    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()                    # T w g lane p {A,B}\n    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)\n    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)                       # T ri rr w g q\n    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)                  # T w (rr q) g ri\n    return torch.cat((wbytes, sbytes), 1).contiguous()\nNT1_CPP = \'\'\'#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <stdint.h>\nint64_t max_smem_attr();int64_t smem_needed(int64_t,int64_t,int64_t);\nint64_t resident_blocks(int64_t,int64_t,int64_t);\nvoid run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);\n\'\'\'\n\nNT1_MODULES = {}\n\nNT1_HOPPER = torch.cuda.get_device_capability() == (9, 0)\n\ndef nt1_module(kind):\n    if kind not in NT1_MODULES:\n        source = NT1_SOURCES[kind]\n        tag = hashlib.sha256(source.encode()).hexdigest()[:12]\n        mod = load_inline(name=\'muse_nt1_limit_\' + kind + tag, cpp_sources=NT1_CPP, cuda_sources=source,\n            functions=[\'max_smem_attr\', \'smem_needed\', \'resident_blocks\', \'run\'],\n            no_implicit_headers=True, with_pytorch_error_handling=False,\n            extra_cuda_cflags=[\'-O3\', \'-std=c++20\', \'-lineinfo\',\n                *([\'-gencode=arch=compute_90a,code=sm_90a\'] if NT1_HOPPER else [])], verbose=False)\n        mod.smem_optin = mod.max_smem_attr()\n        NT1_MODULES[kind] = mod\n    return NT1_MODULES[kind]\n\ndef nt1_call(data, state, kind=\'recode\', stages=2, grid=264):\n    x = data[\'hidden_states\'].contiguous()\n    ids = data[\'topk_ids\'].int().contiguous(); tw = data[\'topk_weights\'].float().contiguous()\n    m = x.shape[0]; p = m * 8; dev = x.device\n    if not 1<=m<=32:raise ValueError(\'small-M specialization expects 1..32 tokens\')\n    mod = nt1_module(kind)\n    if mod.smem_needed(1, stages, m) > mod.smem_optin:\n        raise ValueError(f\'{kind} stages{stages} shared memory exceeds device capacity\')\n    resident = mod.resident_blocks(1, stages, m)\n    if resident < 1:\n        raise ValueError(\'no resident CTA for NT1\')\n    grid = min(grid, state[\'sms\'] * resident)\n    bank = state[\'banks\'][kind]\n    inter = torch.empty((p, 4096), device=dev, dtype=torch.bfloat16)\n    y = torch.empty((p, 4096), device=dev)\n    ss = torch.empty((p,), device=dev)\n    cnt = state[\'counters\'][kind]\n    out = torch.empty_like(x)\n    tensors = [x, ids, tw, state[\'pnw\'], bank[\'w13p\'], bank[\'w2p\'], state[\'gs13\'], state[\'gs2\'], inter, y, ss]\n    tensors += [cnt, out]\n    if kind==\'split2\':\n        parts=torch.empty((p,2,8192),device=dev)\n        tensors += [parts,state[\'tickets\']]\n    mod.run([v.data_ptr() for v in tensors], m, data[\'config\'][\'local_start\'], stages, grid,\n            torch.cuda.current_stream(dev).cuda_stream)\n    return out\n\ndef prepare_kind(weights,kind=\'lop3\'):\n    if kind not in NT1_SOURCES:raise ValueError(kind)\n    nt1_module(kind);device=weights[\'w13\'].device\n    rows=torch.arange(512,device=device)[:,None]*8+torch.arange(8,device=device)\n    maps={\'w13\':torch.cat((rows,rows+4096),1),\'w2\':torch.arange(4096,device=device).view(256,16)}\n    bank={}\n    for name,rowmap in maps.items():\n        repack=_raw_repack if kind==\'raw\' else _repack\n        value=torch.stack([repack(weights[name][e],weights[name+\'_sf\'][e],rowmap) for e in range(32)])\n        if kind==\'split2\':\n            tiles=rowmap.shape[0]\n            w=value[...,:32768].reshape(32,tiles,2,16384).permute(0,2,1,3)\n            sf=value[...,32768:].reshape(32,tiles,2,2048).permute(0,2,1,3)\n            value=torch.cat((w,sf),-1).contiguous().reshape(32,tiles*2,18432)\n        bank[name+\'p\']=value\n    return dict(banks={kind:bank},counters={kind:torch.zeros(64,device=device,dtype=torch.int32)},\n        tickets=torch.zeros(256*768,device=device,dtype=torch.int32) if kind==\'split2\' else None,\n        gs13=weights[\'w13_gs\'].float().contiguous(),gs2=weights[\'w2_gs\'].float().contiguous(),\n        pnw=weights.get(\'post_norm_w\',weights.get(\'post_expert_norm\')).contiguous(),\n        sms=torch.cuda.get_device_properties(device).multi_processor_count)\n\n\ndef lop3_bind(recoded_bank):\n    nt1_module(\'lop3\')\n    return dict(banks={\'lop3\':{k:recoded_bank[k] for k in (\'w13p\',\'w2p\')}},\n        counters={\'lop3\':recoded_bank[\'cnt\']},gs13=recoded_bank[\'gs13\'],gs2=recoded_bank[\'gs2\'],\n        pnw=recoded_bank[\'pnw\'],sms=recoded_bank[\'grid\'])\n\n\ndef lop3_experts(x,wts,lids,state):\n    return nt1_call(dict(hidden_states=x,topk_ids=lids,topk_weights=wts,config={\'local_start\':0}),state,\'lop3\')\n\nimport gc\nimport time\nimport statistics\nPC_CUDA=\'#include <cuda_runtime.h>\\n#include <vector>\\n#include <stdexcept>\\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\\n// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\\n\\n\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace pc {\\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\\n unsigned dst=smaddr(d);\\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\\n}\\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\\n}\\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out){\\n unsigned w,mask=0x81c081c0u;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));\\n unsigned s0=__byte_perm(raw,0u,0x4040u)*16u+0x70007000u;\\n unsigned s1=__byte_perm(raw,0u,0x4141u)*16u+0x70007000u;\\n unsigned s2=__byte_perm(raw,0u,0x4242u)*16u+0x70007000u;\\n unsigned s3=__byte_perm(raw,0u,0x4343u)*16u+0x70007000u;\\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\\n}\\n__device__ __forceinline__ void keep(const unsigned*a){\\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\\n}\\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\\n d+=b*2;\\n if constexpr(NT==16){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else if constexpr(NT==32){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else if constexpr(NT==64){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else{\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},{%64,%65,%66,%67},%68,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]),"+f"(c[32]),"+f"(c[33]),"+f"(c[34]),"+f"(c[35]),"+f"(c[36]),"+f"(c[37]),"+f"(c[38]),"+f"(c[39]),"+f"(c[40]),"+f"(c[41]),"+f"(c[42]),"+f"(c[43]),"+f"(c[44]),"+f"(c[45]),"+f"(c[46]),"+f"(c[47]),"+f"(c[48]),"+f"(c[49]),"+f"(c[50]),"+f"(c[51]),"+f"(c[52]),"+f"(c[53]),"+f"(c[54]),"+f"(c[55]),"+f"(c[56]),"+f"(c[57]),"+f"(c[58]),"+f"(c[59]),"+f"(c[60]),"+f"(c[61]),"+f"(c[62]),"+f"(c[63])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }\\n#else\\n int lane=threadIdx.x&31;\\n #pragma unroll\\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\\n  float*p=c+4*j;\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n }\\n#endif\\n}\\ntemplate<int NW>__device__ __forceinline__ void ss_mma(float*c,const __nv_bfloat16*as,const __nv_bfloat16*bs,int atom){\\n\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n uint64_t da=(uint64_t)((smaddr(as)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\\n uint64_t db=(uint64_t)((smaddr(bs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);da+=atom*2;db+=atom*2;\\n if constexpr(NW==16){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1,0,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\\n :"l"(da),"l"(db));\\n }\\n else if constexpr(NW==32){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},%16,%17,p,1,1,0,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\\n :"l"(da),"l"(db));\\n }\\n else if constexpr(NW==64){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\\n :"l"(da),"l"(db));\\n }\\n else if constexpr(NW==128){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},%64,%65,p,1,1,0,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]),"+f"(c[32]),"+f"(c[33]),"+f"(c[34]),"+f"(c[35]),"+f"(c[36]),"+f"(c[37]),"+f"(c[38]),"+f"(c[39]),"+f"(c[40]),"+f"(c[41]),"+f"(c[42]),"+f"(c[43]),"+f"(c[44]),"+f"(c[45]),"+f"(c[46]),"+f"(c[47]),"+f"(c[48]),"+f"(c[49]),"+f"(c[50]),"+f"(c[51]),"+f"(c[52]),"+f"(c[53]),"+f"(c[54]),"+f"(c[55]),"+f"(c[56]),"+f"(c[57]),"+f"(c[58]),"+f"(c[59]),"+f"(c[60]),"+f"(c[61]),"+f"(c[62]),"+f"(c[63])\\n :"l"(da),"l"(db));\\n }\\n#else\\n int t=threadIdx.x&127,w=t>>5,l=t&31;unsigned a[4];\\n #pragma unroll\\n for(int j=0;j<4;++j){int r=w*16+(l>>2)+(j&1)*8,k=atom*16+(l&3)*2+(j>>1)*8;\\n  a[j]=*(const unsigned*)(as+r*64+(k^((r&7)*8)));}\\n mma<NW>(c,a,bs,atom);\\n#endif\\n}\\n}\\nnamespace pc {\\n__device__ __forceinline__ void pbar(){asm volatile("bar.sync 1,128;":::"memory");}\\n__device__ __forceinline__ void cbar(){asm volatile("bar.sync 2,128;":::"memory");}\\n__device__ __forceinline__ void init_bar(uint64_t*b){asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(smaddr(b)):"memory");}\\n__device__ __forceinline__ void arrive(uint64_t*b){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(smaddr(b)):"memory");}\\n__device__ __forceinline__ void wait_bar(uint64_t*b,unsigned phase){\\n unsigned ok;\\n do{asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2;selp.u32 %0,1,0,p;}" :"=r"(ok):"r"(smaddr(b)),"r"(phase):"memory");}while(!ok);\\n}\\n\\ntemplate<int CAP,int S>struct Layout{\\n static constexpr int RAW=3072,A=RAW,B=A+8192,STRIDE=B+CAP*128;\\n static constexpr int BYTES=S*STRIDE;\\n};\\n\\n// One producer warpgroup owns all packed reads, BF16 scatter, and activation\\n// copies. Raw, A, and B have independent slots in the same ring. A is converted\\n// exactly once per expert/channel/K64 tile, outside every token-subgroup loop.\\ntemplate<int CAP,int WIDTH,int S>\\n__device__ __forceinline__ void produce(const int*Q,const unsigned char*SF,const __nv_bfloat16*X,\\n const int*SLOTS,int M,int e,int offset,int count,int mt,int N,unsigned char*sm,uint64_t*full,uint64_t*empty){\\n using L=Layout<CAP,S>;int t=threadIdx.x,w=t>>5,l=t&31;\\n int src[WIDTH/16];\\n #pragma unroll\\n for(int j=0;j<WIDTH/16;++j){int r=t/8+j*16;src[j]=r<count?(N==8192?SLOTS[e*M+r]/8:offset+r):-1;}\\n auto copy=[&](int step){\\n  int st=step%S;unsigned ph=(step/S)&1;\\n  // Empty barriers start pending phase0: phase1 denotes the unused slot.\\n  wait_bar(empty+st,ph^1);unsigned char*slot=sm+st*L::STRIDE;\\n  unsigned char*q=slot,*sf=slot+2048;auto*as=(__nv_bfloat16*)(slot+L::A);auto*bs=(__nv_bfloat16*)(slot+L::B);\\n  #pragma unroll\\n  for(int g=0;g<2;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+step*2+g;\\n   if(l<16)async_copy(q+(g*128+w*32)*8+l*16,Q+z*64+l*4,16,1);\\n   if(l<8)async_copy(sf+g*128+w*32+l*4,SF+z*32+l*4,4,1);\\n  }\\n  #pragma unroll\\n  for(int j=0;j<WIDTH/16;++j){int r=t/8+j*16,k=(t&7)*8;\\n   async_copy(bs+r*64+(k^((r&7)*8)),X+(size_t)max(src[j],0)*4096+step*64+k,16,src[j]>=0);\\n  }\\n  asm volatile("cp.async.commit_group;":::"memory");\\n };\\n #pragma unroll\\n for(int step=0;step<S-1;++step)copy(step);\\n #pragma unroll 1\\n for(int step=0;step<64;++step){\\n  if(step+S-1<64)copy(step+S-1);\\n  if(step==63)asm volatile("cp.async.wait_group 0;":::"memory");\\n  else if constexpr(S==2)asm volatile("cp.async.wait_group 1;":::"memory");\\n  else if(step==62)asm volatile("cp.async.wait_group 1;":::"memory");\\n  else asm volatile("cp.async.wait_group 2;":::"memory");\\n  // Q and scale words used by this warp were copied by this warp. The sole\\n  // producer-group barrier is after decode, when all A/B writers publish.\\n  __syncwarp();int st=step%S;unsigned char*slot=sm+st*L::STRIDE;\\n  unsigned char*q=slot,*sf=slot+2048;auto*as=(__nv_bfloat16*)(slot+L::A);\\n  unsigned af[16];\\n  #pragma unroll\\n  for(int g=0;g<2;++g){uint2 v=*(const uint2*)(q+(g*128+t)*8);unsigned s=*(const unsigned*)(sf+g*128+w*32+(l>>2)*4);dq(v.x,v.y,s,af+g*8);}\\n  #pragma unroll\\n  for(int atom=0;atom<4;++atom){\\n   #pragma unroll\\n   for(int j=0;j<4;++j){int r=w*16+(l>>2)+(j&1)*8,k=atom*16+(l&3)*2+(j>>1)*8;\\n    *(unsigned*)(as+r*64+(k^((r&7)*8)))=af[atom*4+j];}\\n  }\\n  // All writers publish their generic shared writes to the async proxy before\\n  // the leader releases full. No producer crosses a stage-reuse edge early.\\n  asm volatile("fence.proxy.async.shared::cta;":::"memory");pbar();if(t==0)arrive(full+st);\\n }\\n}\\n\\ntemplate<int CAP,int WIDTH,int NLIMIT,int S>\\n__device__ __forceinline__ void consume(unsigned char*sm,uint64_t*full,uint64_t*empty,float*acc){\\n using L=Layout<CAP,S>;int t=threadIdx.x&127;\\n #pragma unroll 1\\n for(int step=0;step<64;++step){\\n  int st=step%S;unsigned ph=(step/S)&1;wait_bar(full+st,ph);\\n  auto*as=(__nv_bfloat16*)(sm+st*L::STRIDE+L::A);auto*bs=(__nv_bfloat16*)(sm+st*L::STRIDE+L::B);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n  asm volatile("wgmma.fence.sync.aligned;":::"memory");\\n#endif\\n  #pragma unroll\\n  for(int group=0;group<(WIDTH+NLIMIT-1)/NLIMIT;++group){constexpr int NW=WIDTH<NLIMIT?WIDTH:NLIMIT;\\n   #pragma unroll\\n   for(int atom=0;atom<4;++atom)ss_mma<NW>(acc+group*(NLIMIT/2),as,bs+group*NLIMIT*64,atom);\\n  }\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n  asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n  // Both shared descriptors remain live until wait0. The producer can fill\\n  // other slots while this group executes, but never overwrite this one.\\n  cbar();if(t==0)arrive(empty+st);\\n }\\n}\\n\\ntemplate<int CAP,int NLIMIT,int S,bool FIRST>\\n__global__ void __launch_bounds__(256,2) heavy(const int*Q,const unsigned char*SF,const float*GS,\\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\\n const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M){\\n using L=Layout<CAP,S>;\\n extern __shared__ __align__(1024) unsigned char raw[];\\n unsigned char*sm=(unsigned char*)(((uintptr_t)raw+1023)&~(uintptr_t)1023);\\n __shared__ uint64_t full[S],empty[S];\\n __shared__ __nv_bfloat16 epi[FIRST?32*CAP:1];__shared__ float rowsum[FIRST?1:4*CAP];\\n constexpr int N=FIRST?8192:4096,MT=N/64;int tx=threadIdx.x,t=tx&127,w=t>>5,l=t&31;\\n int groups=TASK[32];\\n for(int unit=blockIdx.x;unit<groups*MT;unit+=gridDim.x){\\n  int group=unit%groups,mt=unit/groups,ix=0;\\n  #pragma unroll\\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\\n  int e=ORDER[ix],count=C[e],offset=OFF[e];\\n  if(tx==0){\\n   #pragma unroll\\n   for(int s=0;s<S;++s){init_bar(full+s);init_bar(empty+s);}\\n   asm volatile("fence.mbarrier_init.release.cluster;":::"memory");\\n  }\\n  __syncthreads();\\n  // Select N outside the K loop. On M128 each touched expert has at most128\\n  // distinct token routes, so there is exactly one task for all its rows.\\n  if(tx<128){\\n   if(count<=16)produce<CAP,16,S>(Q,SF,X,SLOTS,M,e,offset,count,mt,N,sm,full,empty);\\n   else if(count<=32)produce<CAP,32,S>(Q,SF,X,SLOTS,M,e,offset,count,mt,N,sm,full,empty);\\n   else if constexpr(CAP>=64){\\n    if(count<=64)produce<CAP,64,S>(Q,SF,X,SLOTS,M,e,offset,count,mt,N,sm,full,empty);\\n    else produce<CAP,128,S>(Q,SF,X,SLOTS,M,e,offset,count,mt,N,sm,full,empty);\\n   }\\n  }else{\\n   float acc[CAP/2]={};\\n   if(count<=16)consume<CAP,16,NLIMIT,S>(sm,full,empty,acc);\\n   else if(count<=32)consume<CAP,32,NLIMIT,S>(sm,full,empty,acc);\\n   else if constexpr(CAP>=64){\\n    if(count<=64)consume<CAP,64,NLIMIT,S>(sm,full,empty,acc);\\n    else consume<CAP,128,NLIMIT,S>(sm,full,empty,acc);\\n   }\\n   if constexpr(FIRST){\\n    #pragma unroll\\n    for(int j=0;j<CAP/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8,c=w*16+(l>>2)+((j>>1)&1)*8;\\n     if(c>=32)epi[(c-32)*CAP+r]=__float2bfloat16_rn(acc[j]*GS[e+32]*256.f);}\\n    cbar();\\n    #pragma unroll\\n    for(int j=0;j<CAP/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8,c=w*16+(l>>2)+((j>>1)&1)*8;\\n     if(c<32&&r<count){float g=__bfloat162float(__float2bfloat16_rn(acc[j]*GS[e]*256.f));float u=__bfloat162float(epi[c*CAP+r]);\\n      ACT[(size_t)(offset+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);}}\\n   }else{\\n    #pragma unroll\\n    for(int j=0;j<CAP/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8,c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n     float z=acc[j]*GS[e]*256.f*(1.f+__bfloat162float(PNW[c]));acc[j]=z;if(r<count)Y[(size_t)(offset+r)*4096+c]=z;}\\n    #pragma unroll\\n    for(int i=0;i<CAP/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2],b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\\n     #pragma unroll\\n     for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\\n     if(l<4){rowsum[w*CAP+i*8+2*l]=a;rowsum[w*CAP+i*8+2*l+1]=b;}}\\n    cbar();\\n    for(int r=t;r<CAP;r+=128)if(r<count){float sum=0;\\n     #pragma unroll\\n     for(int ww=0;ww<4;++ww)sum+=rowsum[ww*CAP+r];SS[(size_t)(offset+r)*64+mt]=sum;}\\n   }\\n  }\\n  // All producers and consumers finish this expert/channel before barriers\\n  // are reinitialized for the persistent CTA\\\'s next task.\\n  __syncthreads();\\n }\\n}\\n}\\n\\n#define PC_CASES(F) F(0,32,2) F(1,64,2) F(2,128,2) F(3,64,3)\\nint64_t pc_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\\n #define INIT(ID,NL,S) {auto init=[&]<int CAP,bool F>(){WX_CUDA_CHECK(cudaFuncSetAttribute(pc::heavy<CAP,NL,S,F>,cudaFuncAttributeMaxDynamicSharedMemorySize,pc::Layout<CAP,S>::BYTES+1023));};init.template operator()<32,true>();init.template operator()<32,false>();init.template operator()<128,true>();init.template operator()<128,false>();}\\n PC_CASES(INIT)\\n #undef INIT\\n return opt;\\n}\\nstd::vector<int64_t> pc_info(int64_t which,int64_t cap,int64_t first){\\n #define INFO(ID,NL,S) case ID:{auto get=[&]<int CAP,bool F>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,pc::heavy<CAP,NL,S,F>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,pc::heavy<CAP,NL,S,F>,256,pc::Layout<CAP,S>::BYTES+1023));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,pc::Layout<CAP,S>::BYTES+1023,blocks};};if(cap==32){if(first)return get.template operator()<32,true>();return get.template operator()<32,false>();}if(first)return get.template operator()<128,true>();return get.template operator()<128,false>();}\\n switch(which){PC_CASES(INFO)default:WX_CHECK(false);}\\n #undef INFO\\n}\\nvoid pc_run(std::vector<int64_t>v,int64_t first,int64_t which,int64_t grid,int64_t M,int64_t stream){\\n #define RUN(ID,NL,S) case ID:{auto go=[&]<int CAP,bool F>(){pc::heavy<CAP,NL,S,F><<<grid,256,pc::Layout<CAP,S>::BYTES+1023,(cudaStream_t)stream>>>( \\\\\\n (int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[14],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],M);};if(M<=32){if(first)go.template operator()<32,true>();else go.template operator()<32,false>();}else{if(first)go.template operator()<128,true>();else go.template operator()<128,false>();}break;}\\n switch(which){PC_CASES(RUN)default:WX_CHECK(false);}\\n #undef RUN\\n WX_CUDA_CHECK(cudaGetLastError());\\n}\\n\\n__global__ void pc_meta_kernel(const int*C,const int*ORDER,int*T,int*T32,float*SS,int cap,int cut){\\n int t=threadIdx.x,all=blockIdx.x*blockDim.x+t;\\n for(int i=all;i<cap*64;i+=gridDim.x*blockDim.x)SS[i]=0.f;\\n if(blockIdx.x==0&&t<32){int n=C[ORDER[t]],v=n>cut,b=v?(n+31)/32:0;\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),c=__shfl_up_sync(~0u,b,d);if(t>=d){v+=a;b+=c;}}\\n  T[t+1]=v;T32[t+1]=b;if(t==0){T[0]=0;T32[0]=0;}}\\n}\\nvoid pc_meta(std::vector<int64_t>v,int64_t cap,int64_t cut,int64_t stream){pc_meta_kernel<<<128,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(float*)v[4],cap,cut);WX_CUDA_CHECK(cudaGetLastError());}\\n\'\nPC_CPP=\'#include <pybind11/pybind11.h>\\n#include <pybind11/stl.h>\\n#include <vector>\\n#include <cstdint>\\nint64_t pc_init();std::vector<int64_t> pc_info(int64_t,int64_t,int64_t);void pc_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void pc_meta(std::vector<int64_t>,int64_t,int64_t,int64_t);\\n\'\nPC_ROWS=[]\nPC_SMALL=[]\n\n\n@functools.lru_cache(None)\ndef pc_module():\n    arch=\'90a\' if torch.cuda.get_device_capability()==(9,0) else \'120\'\n    mod=load_inline(name=\'m128_pc_\'+hashlib.sha256(PC_CUDA.encode()).hexdigest()[:12],cpp_sources=PC_CPP,cuda_sources=PC_CUDA,\n        functions=[\'pc_init\',\'pc_info\',\'pc_run\',\'pc_meta\'],no_implicit_headers=True,with_pytorch_error_handling=False,\n        extra_cuda_cflags=[\'-O3\',\'-std=c++20\',\'-lineinfo\',\'-gencode=arch=compute_\'+arch+\',code=sm_\'+arch],verbose=False)\n    mod.pc_init();return mod\n\n\ndef pc_plan(data,state,which=1,grid=1056,mask=3,cut=16):\n    x=data[\'hidden_states\'];m=x.shape[0];dev=x.device\n    if not 1<=m<=128:raise ValueError(\'producer capacity covers M1..128\')\n    base=state[\'base\'];word=state[\'word\'];native=state[\'native\'];cap=m*8+32*128\n    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(c)\n    tasks=[torch.empty_like(off) for _ in range(3)];slots=torch.empty((32,m),device=dev,dtype=torch.int32)\n    ids=data[\'topk_ids\'].int().contiguous();tw=data[\'topk_weights\'].float().contiguous();pos=torch.empty_like(ids)\n    clear=(cap//32)*193+m;counts=torch.empty(clear+m,device=dev,dtype=torch.int32)\n    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);part=torch.empty(1,device=dev)\n    y=torch.empty((cap,4096),device=dev);ss=torch.empty((cap,64),device=dev);inv=torch.empty(cap,device=dev);out=torch.empty_like(x)\n    common=[c,off,order,tasks[0],slots,act];tail=[ids,tw,pos,base[\'pnw\'],y,ss,inv,counts,out]\n    def vector(bank,first,task):\n        v=[bank[\'w13\' if first else \'w2\'],bank[\'w13_sf\' if first else \'w2_sf\'],bank[\'w13_gs\' if first else \'w2_gs\'],x if first else act]+common+[part]+tail\n        v[7]=task;return v\n    v13=vector(base,1,tasks[1]);v2=vector(base,0,tasks[1]);h13=vector(word,1,tasks[2]);h2=vector(word,0,tasks[2])\n    nv=[x,base[\'pnw\'],native[\'w13p\'],native[\'w2p\'],base[\'w13_gs\'],base[\'w2_gs\'],c,off,slots,act,y,ss]\n    ptrs=[[v.data_ptr() for v in row] for row in (v13,v2,h13,h2,nv)]\n    rp=[v.data_ptr() for v in (ids,c,off,order,tasks[0],slots,pos,counts,out)]\n    mp=[v.data_ptr() for v in (c,order,tasks[1],tasks[2],ss)]\n    old=m128_hh_pm_module();mod=pc_module();start=data[\'config\'].get(\'local_start\',0);owners=(v13,v2,h13,h2,nv,tasks,ids,tw)\n    stages=2\n    while stages>1 and old.pm_nt_smem(stages,m)>M128_HH_ISSUE_OPTS[\'pm_nt\']:stages-=1\n    ngrids={f:native[\'grid\']*old.pm_nt_resident(stages,m,f,1) for f in (0,1)}\n    def stream():return torch.cuda.current_stream().cuda_stream\n    def route():\n        _=owners\n        old.idist_route(rp,m,32,start,clear,stream());mod.pc_meta(mp,cap,cut,stream())\n    def light(first):\n        _=owners\n        if cut:old.pm_nt_run(ptrs[4],first,1,m,stages,ngrids[first],stream())\n    def heavy(first):\n        _=owners\n        if mask&(1 if first else 2):mod.pc_run(ptrs[0 if first else 1],first,which,grid,m,stream())\n        else:old.idist_run(ptrs[2 if first else 3],first,1,1056,start,1,0,m,cap,stream())\n    def a_light():light(1)\n    def a_heavy():heavy(1)\n    def b_light():light(0)\n    def b_heavy():heavy(0)\n    def first():a_light();a_heavy()\n    def second():b_light();b_heavy()\n    def finish():\n        _=owners\n        old.idist_finish(ptrs[3],m,start,stream());return out\n    return dict(steps=(route,first,second,finish),components=(route,a_light,a_heavy,b_light,b_heavy,finish),owners=owners,v13=v13,v2=v2,tasks=tasks,out=out)\n\n\ndef pc_call(data,state,which,grid,mask=3,cut=16):\n    for fn in pc_plan(data,state,which,grid,mask,cut)[\'steps\']:out=fn()\n    return out\n\n\nPC_CHOICES=((\'HH_route\',1,1056,0),(\'P32s2\',0,1056,3),(\'P64s2\',1,1056,3),(\'P128s2\',2,1056,3),\n    (\'P64s3\',3,1056,3),(\'P64g528\',1,528,3),(\'P64g2112\',1,2112,3),\n    (\'A32s2\',0,1056,1),(\'A64s2\',1,1056,1),(\'A128s2\',2,1056,1),(\'A64s3\',3,1056,1),\n    (\'A64g528\',1,528,1),(\'A64g2112\',1,2112,1),(\'B64s2\',1,1056,2))\nPC_NAMES=(\'HH_admitted\',)+tuple(v[0] for v in PC_CHOICES)\n\n\ndef pc_measure(w,state,ds,flush,small=False):\n    wants=dc_oracles(ds,w)\n    if small:\n        tiny=lop3_bind(state[\'native\'])\n        graphs=[MWGraph(lambda d:lop3_experts(d[\'hidden_states\'],d[\'topk_weights\'],d[\'topk_ids\'],tiny),ds[0])]\n        choices=((\'P64s2\',1,1056,3),(\'P64s3\',3,1056,3));names=(\'NT1lop\',)+tuple(c[0] for c in choices)\n    else:\n        graphs=[MWGraph(lambda d:hh_experts(d[\'hidden_states\'],d[\'topk_weights\'],d[\'topk_ids\'],state),ds[0])];choices=PC_CHOICES;names=PC_NAMES\n    graphs += [MWGraph(lambda d,c=c,g=g,p=p:pc_call(d,state,c,g,p,0 if small else 16),ds[0]) for name,c,g,p in choices]\n    errors=[[0.,0.] for g in graphs];records=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)\n    for index,(data,want) in enumerate(zip(ds,wants)):\n        for j,g in enumerate(graphs):\n            g.refresh(data);out=g.replay();torch.cuda.synchronize();e,r=pm_error(out,want);errors[j]=[max(e,errors[j][0]),max(r,errors[j][1])]\n            if e>.01 or r>.05:raise RuntimeError(\'pre FP32 \'+str((data[\'hidden_states\'].shape[0],names[j],index,e,r)))\n        samples=[[] for g in graphs];noise=[]\n        for repeat in range(6):\n            order=list(range(len(graphs)))+[0];shift=repeat%len(order);order=order[shift:]+order[:shift]\n            if repeat%2:order.reverse()\n            aa=[]\n            for j in order:\n                flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();us=a.elapsed_time(b)*1000.\n                samples[j].append(us)\n                if j==0:aa.append(us)\n            noise.append(aa[1]-aa[0])\n        for j,g in enumerate(graphs):\n            e,r=pm_error(g.out,want);errors[j]=[max(e,errors[j][0]),max(r,errors[j][1])]\n            if e>.01 or r>.05:raise RuntimeError(\'post FP32 \'+str((names[j],index,e,r)))\n        ids=data[\'topk_ids\'];counts=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32).cpu().tolist()\n        records.append(dict(M=data[\'hidden_states\'].shape[0],fixture=index,names=names,us=[statistics.median(v) for v in samples],samples=samples,errors=[list(v) for v in errors],aa=noise,counts=counts,\n            heavy_experts=sum(c>16 for c in counts),HH_heavy_groups=sum((c+31)//32 for c in counts if c>16)))\n    return records\n\n\ndef pc_phases(data,state,which,grid,mask,flush):\n    plan=pc_plan(data,state,which,grid,mask);steps=plan[\'components\'];result=[]\n    for f in steps:f()\n    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)\n    for index,fn in enumerate(steps):\n        graph=torch.cuda.CUDAGraph()\n        with torch.cuda.graph(graph):fn()\n        samples=[]\n        for _ in range(3):\n            for prep in steps[:index]:prep()\n            flush.zero_();a.record();graph.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000.)\n        result.append(statistics.median(samples))\n    return result\n\n\ndef lab(deadline,weights,state=None):\n    global PC_ROWS,PC_SMALL\n    PC_ROWS=[];PC_SMALL=[];started=time.monotonic();dev=next(v.device for v in weights.values() if torch.is_tensor(v))\n    lines=[\'META corrected PC:128 producer/128 WGMMA consumer threads,2/3-slot full/empty mbarriers; each expert/channel/K64 packed tile decoded once, shared by every N subgroup; exact u8 recode; no prepared expanded expert weights.\',\n        \'META12 public synthetic MoE/router seeds x3 identical routes; M128 vs exact H_H, M16/32 vs NT1lop;6 cold128MiB AB/BA,A:A,independent FP32 before/after. ORDER \'+\',\'.join(PC_NAMES)]\n    if deadline-time.monotonic()<50:return lines+[\'INCOMPLETE before_compile\']\n    m128_hh_pm_module();mod=pc_module();nt1_module(\'lop3\')\n    lines.append(f\'BUILD s={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}\')\n    for cfg in (0,1,2,3):lines.append(f\'RES cfg{cfg} M128 W13/W2=\'+str([mod.pc_info(cfg,128,f) for f in (1,0)]))\n    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8);phases=[]\n    for seed in range(12):\n        if deadline-time.monotonic()<12:lines.append(f\'INCOMPLETE seed={seed}\');break\n        w=dc_weights(5353+7919*seed,dev);w[\'w13_gs_gate\']=w[\'w13_gs\'];w[\'w13_gs_up\']=w[\'w13_gs\']\n        bank=hh_prepare(w);ds=dc_fixtures(w,128);rows=pc_measure(w,bank,ds,flush);PC_ROWS.extend(dict(seed=seed,**v) for v in rows)\n        if seed==0:\n            for name,c,g,p in PC_CHOICES:\n                if p==3 or p==0:phases.append(\'PH \'+name+\' r/Al/Ah/Bl/Bh/f=\'+\',\'.join(f\'{v:.1f}\' for v in pc_phases(ds[0],bank,c,g,p,flush)))\n        for M in (16,32):PC_SMALL.extend(dict(seed=seed,**v) for v in pc_measure(w,bank,dc_fixtures(w,M),flush,True))\n        lines.append(\'S\'+str(seed)+\' us=\'+\',\'.join(f\'{statistics.mean(v["us"][j] for v in rows):.1f}\' for j in range(len(PC_NAMES))))\n        del bank,w,ds;gc.collect()\n    lines+=phases\n    for j,name in enumerate(PC_NAMES):\n        if PC_ROWS:\n            mean=statistics.mean(v[\'us\'][j] for v in PC_ROWS);ratio=statistics.mean(v[\'us\'][j]/v[\'us\'][0] for v in PC_ROWS)\n            e=max(v[\'errors\'][j][0] for v in PC_ROWS);r=max(v[\'errors\'][j][1] for v in PC_ROWS)\n            lines.append(f\'MEAN {name} n={len(PC_ROWS)}/36 us={mean:.2f} vsHH={ratio:.4f} L2={e:.5f} row={r:.5f} PASS\')\n    for M in (16,32):\n        rows=[v for v in PC_SMALL if v[\'M\']==M]\n        for j,name in enumerate((\'NT1lop\',\'P64s2\',\'P64s3\')):\n            if rows:lines.append(f\'SMALL M{M} {name} n={len(rows)}/36 us={statistics.mean(v["us"][j] for v in rows):.2f} ratio={statistics.mean(v["us"][j]/v["us"][0] for v in rows):.4f} L2={max(v["errors"][j][0] for v in rows):.5f} row={max(v["errors"][j][1] for v in rows):.5f} PASS\')\n    for label,rows in ((\'M128\',PC_ROWS),(\'M16_32\',PC_SMALL)):\n        if rows:\n            noise=[x for v in rows for x in v[\'aa\']];lines.append(f\'AA {label} delta/abs/max_us={statistics.mean(noise):.3f}/{statistics.mean(abs(v) for v in noise):.3f}/{max(abs(v) for v in noise):.3f}\')\n    lines.append(f\'END M128={len(PC_ROWS)}/36 small={len(PC_SMALL)}/72 lab_s={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}\')\n    return lines\n')]
_lab_base_prepare = prepare
_lab_ran = False


def _lab_run_one(name, src, deadline, weights, state):
    import inspect as _inspect, linecache as _linecache
    fname = f"<h200_lab {name}>"  # in linecache, so inspect.getsource (triton.jit) finds the source
    _linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns = dict(globals())
    exec(compile(src, fname, "exec"), ns)
    fn = ns["lab"]
    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)


def prepare(weights):
    global _lab_ran
    import time as _t
    _start = _t.monotonic()
    state = _lab_base_prepare(weights)
    if not _lab_ran:
        _lab_ran = True
        import traceback as _tb
        _end = _start + 285
        for _i, (_name, _sha, _src) in enumerate(_LAB_PAYLOADS):
            _t0 = _t.monotonic()
            _deadline = _t0 + max(0.0, _end - _t0) / (len(_LAB_PAYLOADS) - _i)
            _lines = []
            try:
                _lines = [str(x) for x in (_lab_run_one(_name, _src, _deadline, weights, state) or [])]
                torch.cuda.synchronize()  # inside the try: a sticky CUDA error still prints this payload's lines
                torch.cuda.empty_cache()
            except BaseException as _e:  # the error is the result; a sticky CUDA error still fails the run after
                _lines += [f"LAB_ERROR {type(_e).__name__}: {_e}"] + _tb.format_exc().splitlines()[-8:]
            # Over the cap, keep the first two lines (META) and the last lines (summaries, errors) and drop the middle.
            _cap = 9000 // len(_LAB_PAYLOADS)
            _head, _tail, _n = _lines[:2], [], sum(len(_l) + 1 for _l in _lines[:2]) + 40
            if sum(len(_l) + 1 for _l in _lines) > _cap:
                for _l in reversed(_lines[2:]):
                    if _n + len(_l) + 1 > _cap:
                        break
                    _tail.insert(0, _l)
                    _n += len(_l) + 1
                _out = _head + [f"LAB_TRUNCATED {len(_lines) - 2 - len(_tail)} middle lines"] + _tail
            else:
                _out = _lines
            print(f"LAB_BEGIN {_name} {_sha}", flush=True)
            for _l in _out:
                print(_l, flush=True)
            print(f"LAB_END lab_s={_t.monotonic() - _t0:.1f} prepare_s={_t.monotonic() - _start:.1f}", flush=True)
    return state
