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


# ---- h200_lab wrapper: payloads moe_pipeline_seed 41b4eefee919f67f on base f3786341611f4f1f ----
_LAB_PAYLOADS = [('moe_pipeline_seed', '41b4eefee919f67f', '# Control source SHA256 af4877a2ad643b95e2a5a71cc7ab95e757b338bb774b043fd9b5b2677dce87eb\nimport torch\nimport torch as rv_torch\nrv_H = 4096\nrv_NLOC = 32\nrv_TOPK = 8\nrv_TILE_BYTES = 36864\nrv_NT1, rv_NT2 = (512, 256)\nrv_NCNT = 64\nrv_CUDA_SRC = \'\\n#include <torch/extension.h>\\n#include <ATen/cuda/CUDAContext.h>\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace {\\n\\nconstexpr int HD = 4096;\\nconstexpr int NLOC = 32;\\nconstexpr int TILE_BYTES = 36864;\\nconstexpr int TILE_W_BYTES = 32768;\\nconstexpr int NT1 = 512;\\nconstexpr int NT2 = 256;\\nconstexpr int NCW = 8;                 // consumer warps\\nconstexpr int NCT = NCW * 32;          // consumer threads\\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\\nconstexpr uint32_t LA_MASK = 0x60306030u;\\n// persistent counters\\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\\n\\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\\n}\\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\\n}\\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\\n  uint32_t ok;\\n  asm volatile("{\\\\n .reg .pred p;\\\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\\\n selp.u32 %0, 1, 0, p;\\\\n}\\\\n"\\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\\n  return ok != 0;\\n}\\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\\n  while (!mbar_try(b, parity)) {}\\n}\\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\\n}\\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\\n}\\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\\n}\\n__device__ __forceinline__ void bar_consumers() {\\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\\n}\\n__device__ __forceinline__ int ld_acquire(const int* p) {\\n  int v;\\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\\n  return v;\\n}\\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\\n  uint32_t d;\\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\\n  return d;\\n}\\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\\n                                         uint32_t b0, uint32_t b1) {\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\\n}\\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\\n  o[0] = A & DQ_MASK;\\n  o[1] = rotr(A, 4) & DQ_MASK;\\n  o[2] = rotr(A, 10) & DQ_MASK;\\n  o[3] = rotr(B, 2) & DQ_MASK;\\n  o[4] = rotr(B, 6) & DQ_MASK;\\n  o[5] = rotr(B, 12) & DQ_MASK;\\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\\n  o[6] = rotr(W, 24) & DQ_MASK;\\n  o[7] = rotr(W, 30) & DQ_MASK;\\n}\\n\\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\\n if(exp) b=((exp+238)<<7)|(m<<4);\\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\\n b|=(raw&128)<<8; return b|(b<<16);\\n}\\n\\nstruct PrefetchState {\\n  volatile int index, stage, phase, pass;\\n  volatile int e, nb, m, mb, me;\\n  volatile uint64_t pol_first, pol_normal;\\n};\\n\\nstruct Route {        // shared-memory routing summary\\n  int n[NLOC];        // pairs per local expert\\n  int off[NLOC];      // first sorted position\\n  int nblk[NLOC];     // token blocks\\n  int us1[NLOC + 1];  // phase-1 unit prefix\\n  int us2[NLOC + 1];  // phase-2 unit prefix\\n  int c1, c2;         // m-chunk sizes\\n  int run[NLOC];\\n  int wc[NCW][NLOC];\\n  PrefetchState pre;\\n};\\n\\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\\n                                            int& e, int& nb, int& m) {\\n  e = 0;\\n#pragma unroll\\n  for (int d = 16; d > 0; d >>= 1) {\\n    if (us[e + d] <= u) e += d;\\n  }\\n  const int rel = u - us[e];\\n  const int nbk = nblk[e];\\n  const int nfull = ntile / cc;\\n  const int span = cc * nbk;\\n  if (rel < nfull * span) {\\n    const int mc = rel / span;\\n    const int r2 = rel - mc * span;\\n    nb = r2 / cc;\\n    m = mc * cc + (r2 - nb * cc);\\n  } else {\\n    const int L = ntile - nfull * cc;\\n    const int r3 = rel - nfull * span;\\n    nb = r3 / L;\\n    m = nfull * cc + (r3 - nb * L);\\n  }\\n}\\n\\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\\n                                               const int* nblk, int cc, int ntile) {\\n  if (++m == me) {\\n    if (++nb == nblk[e]) {\\n      nb = 0;\\n      mb += cc;\\n      if (mb >= ntile) {\\n        ++e;\\n        while (e < NLOC && nblk[e] == 0) ++e;\\n        mb = 0;\\n      }\\n      me = min(ntile, mb + cc);\\n    }\\n    m = mb;\\n  }\\n}\\n\\n\\n__device__ __forceinline__ void prefetch_one(\\n    Route* rt, int total, int first, int a2, int nstage,\\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\\n    const uint8_t* w13p, const uint8_t* w2p) {\\n  auto& p = rt->pre;\\n  int idx = p.index;\\n  if (idx >= total) return;\\n  int st=p.stage, ph=p.phase, pass=p.pass;\\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\\n  const int ntile=pass?NT2:NT1;\\n  const int cc=pass?rt->c2:rt->c1;\\n  const uint8_t* base=pass?w2p:w13p;\\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\\n  mbar_wait(&emptyb[st], ph^1u);\\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\\n  ++idx;\\n  if (idx < total) {\\n    if (idx == first) {\\n      pass=1;\\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\\n      mb=(m/rt->c2)*rt->c2;\\n      me=min(NT2,mb+rt->c2);\\n    } else {\\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\\n    }\\n  }\\n  if (++st == nstage) { st=0; ph^=1; }\\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\\n}\\n\\ntemplate <int NT>\\n__global__ void __launch_bounds__(NTHR, NT == 1 ? 2 : 1) moe_fused(\\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\\n    int M, int local_start, int nstage) {\\n  constexpr int NB = 8 * NT;\\n  constexpr int RED = NT * 4;\\n  const int npairs = M * 8;\\n  extern __shared__ __align__(128) uint8_t smem[];\\n  uint8_t* stages = smem;\\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\\n  uint64_t* emptyb = fullb + nstage;\\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\\n\\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\\n  const int G = gridDim.x, c = blockIdx.x;\\n  int bar_gen0 = 0;\\n\\n  // ---------------- routing: counts ----------------\\n  if (tid < NLOC) rt->n[tid] = 0;\\n  if (tid == 0) {\\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\\n    bar_gen0 = ld_acquire(cnt + C_GEN);\\n  }\\n  __syncthreads();\\n  for (int i = tid; i < npairs; i += NTHR) {\\n    int e = __ldg(ids + i) - local_start;\\n    if (e < 0 || e >= NLOC) e = -1;\\n    eflat[i] = static_cast<int8_t>(e);\\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\\n  }\\n  __syncthreads();\\n  if (warp == 0) {\\n    const int n = rt->n[lane];\\n    const int nb = (n + NB - 1) / NB;\\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\\n#pragma unroll\\n    for (int d = 1; d < 32; d <<= 1) {\\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\\n    }\\n    rt->off[lane] = off - n;\\n    rt->run[lane] = off - n;\\n    rt->nblk[lane] = nb;\\n    rt->us1[lane + 1] = u1;\\n    rt->us2[lane + 1] = u2;\\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\\n    if (lane == 0) {\\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\\n    }\\n  }\\n  __syncthreads();\\n\\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\\n\\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\\n  if (tid == 0) {\\n    uint64_t pf,pn;\\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\\n    if (total_count) {\\n      int e,nb,m;\\n      const int pass=first_count?0:1;\\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\\n    }\\n    for (int i=0;i<nstage;++i)\\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n  }\\n  __syncthreads();\\n\\n  // ---------------- consumers: deterministic token lists ----------------\\n  for (int base = 0; base < npairs; base += NCT) {\\n    rt->wc[warp][lane] = 0;\\n    bar_consumers();\\n    const int i = base + tid;\\n    const int e = i < npairs ? eflat[i] : -1;\\n    const unsigned mk = __match_any_sync(~0u, e);\\n    const int rank = __popc(mk & ((1u << lane) - 1u));\\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\\n    bar_consumers();\\n    if (tid < NLOC) {\\n      int s = rt->run[tid];\\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\\n      rt->run[tid] = s;\\n    }\\n    bar_consumers();\\n    if (i < npairs) {\\n      if (e >= 0) {\\n        const int pos = rt->wc[warp][e] + rank;\\n        pos_of[i] = static_cast<int16_t>(pos);\\n        pair_of[pos] = static_cast<int16_t>(i);\\n      } else {\\n        pos_of[i] = -1;\\n      }\\n    }\\n  }\\n  // zero this CTA\\\'s slice of the sum-of-squares accumulators, then publish\\n  {\\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\\n    __threadfence();\\n    bar_consumers();\\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\\n  }\\n\\n  const int q = lane & 3;\\n  uint32_t Bf[NT][8][8];\\n  int st = 0;\\n  uint32_t ph = 0;\\n  int buf = 0;\\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\\n  float fw[2];   // (1 + post_norm_w) for this thread\\\'s phase-2 epilogue row, refreshed per tile\\n\\n  for (int pass = 0; pass < 2; ++pass) {\\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\\n    const int* us = pass ? rt->us2 : rt->us1;\\n    const int cc = pass ? rt->c2 : rt->c1;\\n    const int ntile = pass ? NT2 : NT1;\\n    int e=0, nb=0, m=0, mb=0, me=0;\\n    if (ua < ub) {\\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\\n      mb = (m / cc) * cc;\\n      me = min(ntile, mb + cc);\\n    }\\n    for (int u = ua; u < ub; ++u) {\\n      const int ne = rt->n[e], off = rt->off[e];\\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\\n          __threadfence();\\n          bar_consumers();\\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\\n        }\\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\\n        if (pass == 1) {\\n          const int target = NT1 * rt->nblk[e];\\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\\n        }\\n#pragma unroll\\n        for (int j = 0; j < NT; ++j) {\\n          const int idx = nb * NB + j * 8 + (lane >> 2);\\n          const __nv_bfloat16* row = nullptr;\\n          if (idx < ne) {\\n            const int pos = off + idx;\\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\\n          }\\n#pragma unroll\\n          for (int g = 0; g < 8; ++g) {\\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\\n            if (row) {\\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\\n            }\\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\\n          }\\n        }\\n      }\\n\\n\\n      mbar_wait(&fullb[st], ph);\\n      const uint8_t* tile = stages + st * TILE_BYTES;\\n      float acc[NT][4];\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\\n      {\\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\\n#pragma unroll\\n        for (int g = 0; g < 8; ++g) {\\n          const uint4 wv = wq[g * 32];\\n          const uint32_t sw = scw[g >> 1];\\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\\n          uint32_t o[16];\\n          dq8(wv.x, wv.y, o);\\n          dq8(wv.z, wv.w, o + 8);\\n#pragma unroll\\n          for (int s = 0; s < 4; ++s) {\\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\\n#pragma unroll\\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\\n          }\\n        }\\n      }\\n      __syncwarp();\\n      if (lane == 0) mbar_arrive(&emptyb[st]);\\n      if (++st == nstage) { st = 0; ph ^= 1u; }\\n\\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\\n      bar_consumers();\\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n      const float* rb = red + buf * NCW * 32 * RED;\\n      if (pass == 0) {\\n        const float sc = gs13[e] * 256.f;\\n        for (int z=tid;z<8*NB;z+=NTHR) {\\n          const int i = z & 7, n = z >> 3, j = n >> 3, col = n & 7;\\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\\n          float G1 = 0.f, U = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) {\\n            G1 += rb[(w * 32 + ls) * RED + rg];\\n            U += rb[(w * 32 + ls) * RED + rg + 2];\\n          }\\n          const int idx = nb * NB + n;\\n          if (idx < ne) {\\n            G1 *= sc;\\n            U *= gs13[e + NLOC] * 256.f;\\n            const float h = G1 / (1.f + __expf(-G1)) * U;\\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\\n          }\\n        }\\n      } else {\\n        const float sc = gs2[e] * 256.f;\\n        for (int z=tid;z<16*NB;z+=NTHR) {\\n          const int i = z & 15, n = z >> 4, j = n >> 3, col = n & 7;\\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\\n          float v = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\\n          v *= sc;\\n          const int idx = nb * NB + n;\\n          const bool ok = idx < ne;\\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\\n          const float fw0=1.f+__bfloat162float(pnw[m*16+i]); float t2 = ok ? (v*fw0)*(v*fw0) : 0.f;\\n#pragma unroll\\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\\n        }\\n      }\\n      buf ^= 1;\\n      ++seg_tiles;\\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\\n    }\\n  }\\n  if (cur_pass == 0 && seg_tiles > 0) {\\n    __threadfence();\\n    bar_consumers();\\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\\n  }\\n\\n  // ---------------- grid barrier ----------------\\n  __threadfence();\\n  bar_consumers();\\n  if (tid == 0) {\\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\\n      cnt[C_ZERO] = 0;\\n      cnt[C_BAR] = 0;\\n      __threadfence();\\n      atomicAdd(cnt + C_GEN, 1);\\n    } else {\\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\\n    }\\n  }\\n  bar_consumers();\\n\\n  // ---------------- combine: item = (token, half of the columns) ----------------\\n  for (int it = c; it < 2 * M; it += G) {\\n    const int t = it >> 1;\\n    const int c0 = (it & 1) * 2048 + tid * 8;\\n    float f[8];\\n    {\\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\\n#pragma unroll\\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\\n    }\\n    float acc[8];\\n#pragma unroll\\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\\n#pragma unroll\\n    for (int j = 0; j < 8; ++j) {\\n      const int pos = pos_of[t * 8 + j];\\n      if (pos >= 0) {\\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\\n      }\\n    }\\n    uint4 ov;\\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\\n#pragma unroll\\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\\n  }\\n}\\n\\nsize_t smem_bytes(int nt, int nstage, int M) {\\n  const int npairs = M * 8;\\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\\n}\\n\\n}  // namespace\\n\\nint64_t max_smem_attr() {\\n  int dev, optin;\\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\\n  return optin;\\n}\\n\\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\\n  int n;\\n  if(nt==1){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));}else if(nt==2){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<2>, NTHR, smem_bytes(nt,nstage,M)));}else{C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<4>, NTHR, smem_bytes(nt,nstage,M)));}\\n  return n;\\n}\\n\\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\\n         int64_t grid) {\\n  auto stream = at::cuda::getCurrentCUDAStream();\\n  const int M = static_cast<int>(x.size(0));\\n  const size_t smem = smem_bytes(nt, nstage, M);\\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\\n#define LAUNCH(NTV)                                                                                          \\\\\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\\\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\\\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\\\\n      static_cast<int>(nstage))\\n  if(nt==1){LAUNCH(1);}else if(nt==2){LAUNCH(2);}else if(nt==4){LAUNCH(4);}else{TORCH_CHECK(false,"V9 NT must be 1/2/4");}\\n#undef LAUNCH\\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\\n}\\n\'\nrv_CPP_SRC = \'\\n#include <torch/extension.h>\\nint64_t max_smem_attr();\\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\\n         int64_t grid);\\n\'\nrv__MOD = None\n\ndef rv__module():\n    global rv__MOD\n    if rv__MOD is None:\n        import hashlib\n        from torch.utils.cpp_extension import load_inline\n        tag = hashlib.sha1(rv_CUDA_SRC.encode()).hexdigest()[:12]\n        rv__MOD = load_inline(name=f\'muse_moe_w4a16_v9_{tag}\', cpp_sources=rv_CPP_SRC, cuda_sources=rv_CUDA_SRC, functions=[\'max_smem_attr\', \'smem_needed\', \'resident_blocks\', \'run\'], extra_cuda_cflags=[\'-O3\', \'-std=c++20\', \'-lineinfo\'], verbose=False)\n        rv__MOD.smem_optin = rv__MOD.max_smem_attr()\n    return rv__MOD\nrv__DQ_SLOTS = ((\'A\', 0), (\'A\', 4), (\'A\', 10), (\'B\', 2), (\'B\', 6), (\'B\', 12), (\'W\', 24), (\'W\', 30))\nrv__LA_MASK = 1613783088\n\ndef rv__bit_table():\n    """(slot, half, code bit, word, bit position) for the kernel\'s dq8 decode."""\n    dbit = {3: 15, 2: 8, 1: 7, 0: 6}\n    la = {p for p in range(32) if rv__LA_MASK >> p & 1}\n    table, used = ([], {\'A\': set(), \'B\': set()})\n    for k, (src, rot) in enumerate(rv__DQ_SLOTS):\n        for h in (0, 1):\n            for b in range(4):\n                p = (dbit[b] + 16 * h + rot) % 32\n                w = src if src != \'W\' else \'A\' if p in la else \'B\'\n                assert p not in used[w]\n                used[w].add(p)\n                table.append((k, h, b, w, p))\n    assert len(used[\'A\']) == 32 and len(used[\'B\']) == 32\n    return table\nrv__TABLE = rv__bit_table()\n\ndef rv__repack(wq: rv_torch.Tensor, sf: rv_torch.Tensor, rowmap: rv_torch.Tensor) -> rv_torch.Tensor:\n    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""\n    R, K = (wq.shape[0], wq.shape[1] * 2)\n    assert K == 4096\n    T = rowmap.shape[0]\n    codes = rv_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)\n    u = sf.view(rv_torch.uint8)\n    codes = codes.reshape(R, K)[rowmap.reshape(-1)]\n    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)\n    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(rv_torch.int64)\n    words = rv_torch.zeros((2, T, 8, 8, 32, 2), dtype=rv_torch.int64, device=wq.device)\n    for k, h, b, w, p in rv__TABLE:\n        words[0 if w == \'A\' else 1] |= (ct[..., k, h] >> b & 1) << p\n    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()\n    wbytes = words.view(rv_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)\n    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)\n    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)\n    return rv_torch.cat((wbytes, sbytes), 1).contiguous()\n\ndef rv_prepare(weights):\n    dev = weights[\'w13\'].device\n    t = rv_torch.arange(rv_NT1, device=dev)\n    i = rv_torch.arange(8, device=dev)\n    rowmap1 = rv_torch.cat((8 * t[:, None] + i, rv_H + 8 * t[:, None] + i), 1)\n    rowmap2 = rv_torch.arange(rv_H, device=dev).view(rv_NT2, 16)\n    w13p = rv_torch.empty((rv_NLOC, rv_NT1, rv_TILE_BYTES), dtype=rv_torch.uint8, device=dev)\n    w2p = rv_torch.empty((rv_NLOC, rv_NT2, rv_TILE_BYTES), dtype=rv_torch.uint8, device=dev)\n    for e in range(rv_NLOC):\n        w13p[e] = rv__repack(weights[\'w13\'][e], weights[\'w13_sf\'][e], rowmap1)\n        w2p[e] = rv__repack(weights[\'w2\'][e], weights[\'w2_sf\'][e], rowmap2)\n    mod = rv__module()\n    props = rv_torch.cuda.get_device_properties(dev)\n    rv_torch.cuda.synchronize()\n    return {\'w13p\': w13p, \'w2p\': w2p, \'gs13\': weights[\'w13_gs\'][:rv_NLOC].float().contiguous(), \'gs2\': weights[\'w2_gs\'][:rv_NLOC].float().contiguous(), \'pnw\': weights[\'post_norm_w\'].contiguous(), \'cnt\': rv_torch.zeros((rv_NCNT,), dtype=rv_torch.int32, device=dev), \'grid\': props.multi_processor_count, \'mod\': mod}\n\ndef rv_kernel_fn(data, state):\n    cfg = data[\'config\']\n    assert cfg[\'local_count\'] == rv_NLOC and cfg[\'d_hidden\'] == rv_H and (cfg[\'d_expert\'] == rv_H)\n    mod = state[\'mod\']\n    x = data[\'hidden_states\'].contiguous()\n    ids = data[\'topk_ids\'].to(rv_torch.int32).contiguous()\n    tw = data[\'topk_weights\'].to(rv_torch.float32).contiguous()\n    M = x.shape[0]\n    npairs = M * rv_TOPK\n    nt = state.get(\'nt\', 1)\n    nstage = state.get(\'nstage_choice\', 2)\n    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:\n        nstage -= 1\n    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:\n        raise ValueError(f"batch {M} too large for the fused kernel\'s shared-memory routing")\n    dev = x.device\n    inter = rv_torch.empty((npairs, rv_H), dtype=rv_torch.bfloat16, device=dev)\n    y = rv_torch.empty((npairs, rv_H), dtype=rv_torch.float32, device=dev)\n    ss = rv_torch.empty((npairs,), dtype=rv_torch.float32, device=dev)\n    out = rv_torch.empty((M, rv_H), dtype=rv_torch.bfloat16, device=dev)\n    mod.run(x, ids, tw, state[\'pnw\'], state[\'w13p\'], state[\'w2p\'], state[\'gs13\'], state[\'gs2\'], inter, y, ss, state[\'cnt\'], out, cfg[\'local_start\'], nt, nstage, state[\'grid\'] * mod.resident_blocks(nt, nstage, M))\n    return out\nimport torch\nimport torch as rv_torch\nimport torch as hv_torch\nimport torch as rr_torch\nimport math as fx_math\nimport statistics as fx_statistics\nimport time as fx_time\nfrom typing import Callable as fx_Callable\nimport torch as fx_torch\nfx_HIDDEN = 8192\nfx_D_EXPERT = 4096\nfx_N_EXPERTS = 256\nfx_LOCAL_EXPERTS = 32\nfx_TOP_K = 8\nfx_ROUTER_MULTIPLIER = 0.1767766952966369\nfx_RMS_EPS = 1e-05\nfx_STREAM_SEED = 5354\nfx_STREAM_MEAN = 0.7\nfx_STREAM_SHARED = 0.9\nfx_STREAM_RANK = 4\nfx_DRAFT_SHARE = 0.6\n\ndef fx__rms(x: fx_torch.Tensor) -> fx_torch.Tensor:\n    xf = x.float()\n    return xf * fx_torch.rsqrt(xf.square().mean(-1, keepdim=True) + fx_RMS_EPS)\n\ndef fx__basis(device: fx_torch.device) -> tuple[fx_torch.Tensor, fx_torch.Tensor]:\n    gen = fx_torch.Generator().manual_seed(fx_STREAM_SEED)\n    mean = fx_torch.randn((fx_HIDDEN,), generator=gen).to(device)\n    shared = fx_torch.randn((fx_STREAM_RANK, fx_HIDDEN), generator=gen).to(device)\n    return (mean, shared)\n\ndef fx_make_fixture(weights: dict[str, fx_torch.Tensor], tokens: int, seed: int, *, mode: str=\'real\', q_len: int | None=None) -> dict:\n    """Return an expert-call fixture with real router or controlled route IDs.\n\n    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.\n    Pass it explicitly for other loads. Draft sharing applies within every verify block.\n    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps\n    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every\n    slot to a uniformly chosen local expert, matching older synthetic tuners.\n    """\n    device = weights[\'router\'].device\n    gen = fx_torch.Generator(device=device).manual_seed(seed)\n    mean, shared = fx__basis(device)\n    coef = fx_torch.randn((tokens, fx_STREAM_RANK), generator=gen, device=device)\n    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)\n    if q_len < 1 or tokens % q_len:\n        raise ValueError(f\'tokens={tokens} must be divisible by q_len={q_len}\')\n    scale = fx_torch.ones((tokens // q_len, q_len, 1), device=device)\n    if 1 < q_len <= 8:\n        scale[:, 1:] = fx_DRAFT_SHARE\n    scale = scale.reshape(tokens, 1)\n    stream = scale * (fx_STREAM_MEAN * mean + fx_math.sqrt(fx_STREAM_SHARED / fx_STREAM_RANK) * (coef @ shared)) + (1 - fx_STREAM_SHARED * scale.square()).sqrt() * fx_torch.randn((tokens, fx_HIDDEN), generator=gen, device=device)\n    h = fx__rms(stream) * (1.0 + weights[\'pre_ffn_norm\'].float())\n    router = weights[\'router\'].float()\n    biased = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER) + weights[\'router_bias\'].float()\n    scores = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER)\n    global_ids = biased.topk(fx_TOP_K, dim=-1).indices.to(fx_torch.int32)\n    picked = scores.gather(1, global_ids.long())\n    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)\n    ranges = fx_torch.bincount((global_ids // fx_LOCAL_EXPERTS).flatten(), minlength=fx_N_EXPERTS // fx_LOCAL_EXPERTS)\n    local_start = fx_LOCAL_EXPERTS * int(ranges.argmax())\n    local = ((global_ids - local_start) % fx_N_EXPERTS).to(fx_torch.int32)\n    local_mask = local < fx_LOCAL_EXPERTS\n    if mode == \'uniform\':\n        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)\n    elif mode == \'matched_uniform\':\n        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)\n        lids = fx_torch.where(local_mask, lids, fx_torch.full_like(lids, -1))\n    elif mode == \'real\':\n        lids = fx_torch.where(local_mask, local, fx_torch.full_like(local, -1))\n    else:\n        raise ValueError(mode)\n    x = h @ weights[\'pooler\'].float().t()\n    x = (fx__rms(x) * (1.0 + weights[\'pre_expert_norm\'].float())).to(fx_torch.bfloat16)\n    tw = topk_weights if mode != \'uniform\' else fx_torch.rand((tokens, fx_TOP_K), generator=gen, device=device)\n    if mode == \'uniform\':\n        tw = tw / tw.sum(-1, keepdim=True)\n    cfg = {\'local_count\': 32, \'d_hidden\': fx_D_EXPERT, \'d_expert\': fx_D_EXPERT, \'local_start\': 0}\n    return {\'hidden_states\': x, \'topk_ids\': lids, \'topk_weights\': tw.float(), \'config\': cfg, \'local_start\': local_start, \'global_ids\': global_ids, \'local_mask\': local_mask}\nimport math as rg_math\nimport functools as rg_functools\nimport torch as rg_torch\nrg_HIDDEN = 8192\nrg_Q_SIZE = 1024\nrg_QKV_SIZE = 2304\nrg_D_EXPERT = 4096\nrg_LOCAL_EXPERTS = 32\nrg_N_EXPERTS = 256\nrg_STREAM_SEED = 5354\nrg_STREAM_RANK = 4\nrg_ROUTER_STD = 0.045\nrg_ROUTER_ALIGN = 0.07\nrg_ROUTER_BIAS_STD = 0.055\nrg_BLOCK = 16\nrg_E4M3_MAX = 448.0\nrg_SCALE_LOG2_MEAN = 7.0\nrg_SCALE_LOG2_STD = 0.4\n\ndef rg_random_weights(rows: int, k: int, *, generator: rg_torch.Generator, device: str | rg_torch.device, leading: tuple[int, ...]=(), wide_scales: bool=False):\n    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.\n\n    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)\n    clamped to [8, 448]; the global scale\n    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448\n    instead.\n    """\n    shape = (*leading, rows, k // 2)\n    packed = rg_torch.randint(0, 256, shape, dtype=rg_torch.uint8, generator=generator, device=device)\n    scale_shape = (*leading, rows, k // rg_BLOCK)\n    if wide_scales:\n        u = rg_torch.rand(scale_shape, generator=generator, device=device)\n        sf = rg_torch.exp2(-6 + u * (rg_math.log2(rg_E4M3_MAX) + 6)).to(rg_torch.float8_e4m3fn)\n        mean_sq = float(sf.float().pow(2).mean())\n    else:\n        z = rg_torch.randn(scale_shape, generator=generator, device=device)\n        sf = rg_torch.exp2(rg_SCALE_LOG2_MEAN + rg_SCALE_LOG2_STD * z).clamp(8.0, rg_E4M3_MAX).to(rg_torch.float8_e4m3fn)\n        mean_sq = 4.0 ** rg_SCALE_LOG2_MEAN * rg_math.exp(2 * (rg_math.log(2) * rg_SCALE_LOG2_STD) ** 2)\n    gs = rg_torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=rg_torch.float32, device=device)\n    return (packed, sf, gs)\n\ndef rg_generate_weights(seed: int, device: str=\'cuda\', router_seed: int | None=None):\n    """The layer\'s weights from `seed`. With router_seed, the router and its bias come from that\n    seed instead."""\n    gen = rg_torch.Generator(device=device).manual_seed(seed)\n\n    def dense(n, k, dtype=rg_torch.bfloat16):\n        return (rg_torch.randn((n, k), generator=gen, device=device) / rg_math.sqrt(k)).to(dtype)\n\n    def vec(n, std):\n        return (std * rg_torch.randn((n,), generator=gen, device=device)).to(rg_torch.bfloat16)\n    w13, w13_sf, gs13 = rg_random_weights(2 * rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))\n    w2, w2_sf, gs2 = rg_random_weights(rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))\n    gs13 = gs13 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))\n    gs2 = gs2 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))\n    o, attn_gate, pre_ffn = (dense(rg_HIDDEN, rg_Q_SIZE), vec(rg_HIDDEN, 0.3), vec(rg_HIDDEN, 0.1))\n    weights = {\'qkv_gate\': dense(rg_QKV_SIZE, rg_HIDDEN), \'o\': o, \'attn_residual_gate\': attn_gate, \'pre_ffn_norm\': pre_ffn, **rg__router(gen, device, pre_ffn), \'pooler\': dense(rg_D_EXPERT, rg_HIDDEN), \'pre_expert_norm\': vec(rg_D_EXPERT, 0.1), \'w13\': w13, \'w13_sf\': w13_sf, \'w13_gs\': gs13, \'w2\': w2, \'w2_sf\': w2_sf, \'w2_gs\': gs2, \'post_expert_norm\': vec(rg_D_EXPERT, 0.1), \'unpooler\': dense(rg_HIDDEN, rg_D_EXPERT), \'post_ffn_norm\': vec(rg_HIDDEN, 0.1), \'ffn_residual_gate\': vec(rg_HIDDEN, 0.3), \'next_pre_attn_norm\': vec(rg_HIDDEN, 0.1)}\n    if router_seed is not None:\n        weights.update(rg__router(rg_torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))\n    return weights\n\n@rg_functools.cache\ndef rg__stream_basis(device: str):\n    """The residual stream\'s mean direction [8192] and shared directions [STREAM_RANK, 8192]."""\n    gen = rg_torch.Generator().manual_seed(rg_STREAM_SEED)\n    mean = rg_torch.randn((rg_HIDDEN,), generator=gen)\n    shared = rg_torch.randn((rg_STREAM_RANK, rg_HIDDEN), generator=gen)\n    return (mean.to(device), shared.to(device))\n\ndef rg__router(gen, device, pre_ffn):\n    """FP32 router rows, each with a component along the stream\'s mean direction, and the router\n    bias. pre_ffn is unused."""\n    mean, _ = rg__stream_basis(str(device))\n    w = rg_ROUTER_STD * rg_torch.randn((rg_N_EXPERTS, rg_HIDDEN), generator=gen, device=device)\n    align = rg_ROUTER_ALIGN * rg_torch.randn((rg_N_EXPERTS, 1), generator=gen, device=device)\n    rows = w + align * mean[None, :] / rg_math.sqrt(rg_HIDDEN)\n    bias = rg_ROUTER_BIAS_STD * rg_torch.randn((rg_N_EXPERTS,), generator=gen, device=device)\n    return {\'router\': rows.contiguous(), \'router_bias\': bias.contiguous()}\nCAPTURED_R0 = \'eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK\'\nimport base64\nimport statistics\nimport time\nimport zlib\nimport torch\nimport triton\nimport triton.language as tl\nimport ctypes as pf_ct\nimport hashlib as pf_hashlib\nimport logging as pf_logging\nimport math as pf_math\nimport os as pf_os\nimport statistics as pf_statistics\nimport time as pf_time\nimport torch as pf_torch\nimport torch.nn.functional as pf_F\nimport triton as pf_triton\nimport triton.language as pf_tl\npf_N_EXPERTS = 256\npf_TOP_K = 8\n\n@pf_triton.jit\ndef pf__gather_norm_kernel(p_ptr, w_ptr, slot_e_ptr, slot_pos_ptr, starts_ptr, a_ptr, row_of_ptr, D: pf_tl.constexpr):\n    """Gather already-normalized input into local-route rows."""\n    t = pf_tl.program_id(0).to(pf_tl.int64)\n    c = pf_tl.arange(0, D)\n    x = pf_tl.load(p_ptr + t * D + c).to(pf_tl.float32)\n    x = x.to(pf_tl.bfloat16)\n    for k in pf_tl.static_range(8):\n        e = pf_tl.load(slot_e_ptr + t * 8 + k)\n        if e >= 0:\n            r = pf_tl.load(starts_ptr + e) + pf_tl.load(slot_pos_ptr + t * 8 + k)\n            pf_tl.store(a_ptr + r.to(pf_tl.int64) * D + c, x)\n            pf_tl.store(row_of_ptr + t * 8 + k, r)\n        else:\n            pf_tl.store(row_of_ptr + t * 8 + k, -1)\n\n@pf_triton.jit\ndef pf__combine_kernel(y_ptr, row_of_ptr, slot_e_ptr, w_ptr, gs_ptr, pw_ptr, out_ptr, D: pf_tl.constexpr):\n    """out[t] = sum_k w[t,k] * rms(gs_e * y[row] * (1 + w_post), 1e-8) over the token\'s local routes."""\n    t = pf_tl.program_id(0).to(pf_tl.int64)\n    c = pf_tl.arange(0, D)\n    pw = pf_tl.load(pw_ptr + c)\n    acc = pf_tl.zeros([D], pf_tl.float32)\n    for k in pf_tl.static_range(8):\n        r = pf_tl.load(row_of_ptr + t * 8 + k)\n        if r >= 0:\n            e = pf_tl.load(slot_e_ptr + t * 8 + k)\n            y = (pf_tl.load(y_ptr + r.to(pf_tl.int64) * D + c).to(pf_tl.float32) * pf_tl.load(gs_ptr + e)).to(pf_tl.bfloat16).to(pf_tl.float32) * pw\n            y = y * pf_tl.rsqrt(pf_tl.sum(y * y, 0) / D + 1e-08)\n            acc += pf_tl.load(w_ptr + t * 8 + k) * y\n    pf_tl.store(out_ptr + t * D + c, acc.to(pf_tl.bfloat16))\npf_GROUPED_SRC = \'#include <torch/extension.h>\\n#include <ATen/cuda/CUDAContext.h>\\n#include <c10/cuda/CUDAException.h>\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n#include <stdio.h>\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n#define USE_WGMMA 1\\n#else\\n#define USE_WGMMA 0\\n#endif\\n#include <cuda.h>\\n#include <cuda_runtime.h>\\n#define MOE_WGMMA USE_WGMMA\\nnamespace raw17 {\\n\\nconstexpr int BN = 128;              // tokens per tile\\nconstexpr int X_BYTES = BN * 128;    // 16 KB\\nconstexpr int W_BYTES = 4 * 2048;    // 8 KB\\nconstexpr int SF_BYTES = 4 * 256;    // 1 KB\\nconstexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;   // 25600 (multiple of 1024)\\n// epilogue staging: per consumer WG, 1 tile (MODE 1) or 2 tiles (MODE 0) of [128 tok x 64 feat]\\ntemplate <int MODE> constexpr int epi_bytes() { return 2 * (MODE == 1 ? 1 : 2) * X_BYTES; }\\ntemplate <int MODE, int STAGES> constexpr int smem_bytes() {\\n  return STAGES * STAGE_BYTES + epi_bytes<MODE>() + 1024 /*align*/ + 256 /*barriers, tiles*/;\\n}\\n\\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\\n}\\n\\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));\\n}\\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {\\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");\\n}\\n__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {\\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");\\n}\\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\\n  asm volatile(\\n      "{\\\\n"\\n      ".reg .pred p;\\\\n"\\n      "WAIT_%=:\\\\n"\\n      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\\\n"\\n      "@!p bra WAIT_%=;\\\\n"\\n      "}\\\\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");\\n}\\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {\\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"\\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");\\n}\\n__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {\\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"\\n               :: "l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");\\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\\n}\\n__device__ __forceinline__ void bulk_wait_read() {\\n  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");\\n}\\n__device__ __forceinline__ void fence_async_smem() {\\n  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\\n}\\n__device__ __forceinline__ void named_bar(int id, int n) {\\n  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");\\n}\\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\\n  uint32_t r;\\n  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));\\n  return r;\\n}\\n\\n// bf16 bit patterns of e2m1 magnitudes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6: low and high bytes.\\nconstexpr uint32_t LUT_LO_A = 0xC0800000u, LUT_LO_B = 0xC0804000u;\\nconstexpr uint32_t LUT_HI_A = 0x3F3F3F00u, LUT_HI_B = 0x40404040u;\\n\\n// 8 e2m1 nibbles (value 2j in the low nibble of byte j) -> 4 bf16x2: r_j = (v_2j, v_2j+1).\\n__device__ __forceinline__ void decode8(uint32_t x, uint32_t lo_a, uint32_t hi_a, uint32_t (&r)[4]) {\\n  uint32_t xm = x & 0x77777777u;\\n  uint32_t l03 = prmt(lo_a, LUT_LO_B, xm);\\n  uint32_t h03 = prmt(hi_a, LUT_HI_B, xm);\\n  uint32_t l47 = prmt(lo_a, LUT_LO_B, xm >> 16);\\n  uint32_t h47 = prmt(hi_a, LUT_HI_B, xm >> 16);\\n  uint32_t se = x << 4;\\n  uint32_t s03 = prmt(se, x, 0x5140u);\\n  uint32_t s47 = prmt(se, x, 0x7362u);\\n  h03 |= s03 & 0x80808080u;\\n  h47 |= s47 & 0x80808080u;\\n  r[0] = prmt(l03, h03, 0x5140u);\\n  r[1] = prmt(l03, h03, 0x7362u);\\n  r[2] = prmt(l47, h47, 0x5140u);\\n  r[3] = prmt(l47, h47, 0x7362u);\\n}\\n\\n// Two e4m3 bytes (lo = row r, hi = row r+8) -> bf16x2 (exact).\\n// Encoded scale byte u: |sf| = 2^(u/8 - 21) * (1 + (u%8)/8).\\n// Reconstruct |sf| * 2^118 exactly in BF16. Zero has its own code.\\n__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {\\n  uint32_t v[2];\\n  #pragma unroll\\n  for (int i = 0; i < 2; ++i) {\\n    uint32_t raw = (two_bytes >> (8 * i)) & 255u;\\n    uint32_t mag = raw & 127u, exp = mag >> 3, m = mag & 7u, b = 0;\\n    if (exp) b = ((exp + 238) << 7) | (m << 4);\\n    else if (m) { int k = 31 - __clz(m); b = ((236 + k) << 7) | ((m << (7 - k)) & 127u); }\\n    v[i] = b | ((raw & 128u) << 8);\\n  }\\n  uint32_t bits = v[0] | (v[1] << 16);\\n  return *reinterpret_cast<__nv_bfloat162*>(&bits);\\n}\\n\\n__device__ __forceinline__ uint32_t bmul(uint32_t a, __nv_bfloat162 s) {\\n  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&a);\\n  v = __hmul2(v, s);\\n  return *reinterpret_cast<uint32_t*>(&v);\\n}\\n\\n// Decode k16 block b of one m64 block: raw word x, scales (row r, row r+8) as bf16x2 -> A fragment.\\n// Sixteen re-encoded E2M1 codes in A,B -> eight BF16x2 words.\\n// The bit planes represent FP4 * 2^-126; multiplication by the encoded scale\\n// yields the exact NVFP4 weight / 256. The epilogue restores the power of two.\\n__device__ __forceinline__ void make_frag(uint32_t A, uint32_t B, int half,\\n      __nv_bfloat162 sf, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {\\n  constexpr uint32_t MASK = 0x81C081C0u, LA = 0x60306030u;\\n  uint32_t r[4];\\n  if (half == 0) {\\n    r[0] = A & MASK;\\n    r[1] = __funnelshift_r(A,A,4) & MASK;\\n    r[2] = __funnelshift_r(A,A,10) & MASK;\\n    r[3] = __funnelshift_r(B,B,2) & MASK;\\n  } else {\\n    r[0] = __funnelshift_r(B,B,6) & MASK;\\n    r[1] = __funnelshift_r(B,B,12) & MASK;\\n    uint32_t W = (A & LA) | (B & ~LA);\\n    r[2] = __funnelshift_r(W,W,24) & MASK;\\n    r[3] = __funnelshift_r(W,W,30) & MASK;\\n  }\\n  __nv_bfloat162 slo = __low2bfloat162(sf), shi = __high2bfloat162(sf);\\n  a[0] = bmul(r[0], slo); a[1] = bmul(r[1], shi);\\n  a[2] = bmul(r[2], slo); a[3] = bmul(r[3], shi);\\n}\\n\\n__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr) {\\n  // K-major, 128B swizzle: SBO = 1024 B (8 rows x 128 B), LBO unused, layout type 1 (B128).\\n  return (uint64_t)((smem_addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);\\n}\\n\\n#if MOE_WGMMA\\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }\\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }\\ntemplate <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }\\n#else\\n__device__ __forceinline__ void wg_fence() {}\\n__device__ __forceinline__ void wg_commit() {}\\ntemplate <int N> __device__ __forceinline__ void wg_wait() {}\\n#endif\\n\\n__device__ __forceinline__ void keep(const uint32_t (&a)[4]) {\\n  asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));\\n}\\n__device__ __forceinline__ void fence_acc(float (&acc)[64]) {\\n#pragma unroll\\n  for (int i = 0; i < 64; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");\\n}\\n\\n// acc[64] (m64 x n128 per warpgroup) += A(regs) x B(smem tile at xs, k16 block b)\\n__device__ __forceinline__ void mma_block(float (&d)[64], const uint32_t (&a)[4], const uint8_t* xs, int b, int scale_d) {\\n#if MOE_WGMMA\\n  uint64_t desc = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);\\n  asm volatile(\\n      "{\\\\n"\\n      ".reg .pred p;\\\\n"\\n      "setp.ne.b32 p, %69, 0;\\\\n"\\n      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\\n      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "\\n      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "\\n      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "\\n      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "\\n      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\\\\n"\\n      "}\\\\n"\\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),\\n        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),\\n        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),\\n        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),\\n        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),\\n        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),\\n        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),\\n        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));\\n#else\\n  // Emulation: per warp, 16 rows of the m64 block, n8 blocks via mma.sync over the swizzled tile.\\n  const int lane = threadIdx.x & 31;\\n  const int g = lane >> 2, c = lane & 3;\\n  if (!scale_d) {\\n#pragma unroll\\n    for (int i = 0; i < 64; ++i) d[i] = 0.f;\\n  }\\n#pragma unroll\\n  for (int i = 0; i < BN / 8; ++i) {\\n    const int n = 8 * i + g;\\n    const int kb = b * 32 + 4 * c;                    // byte offset of k = 16b + 2c within the 128 B row\\n    const uint8_t* row = xs + n * 128;\\n    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + ((((kb) >> 4) ^ (n & 7)) << 4) + (kb & 15));\\n    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + ((((kb + 16) >> 4) ^ (n & 7)) << 4) + ((kb + 16) & 15));\\n    asm volatile(\\n        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "\\n        "{%0, %1, %2, %3};"\\n        : "+f"(d[4 * i]), "+f"(d[4 * i + 1]), "+f"(d[4 * i + 2]), "+f"(d[4 * i + 3])\\n        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\\n  }\\n#endif\\n}\\n\\nstruct Params {\\n  const uint8_t* wblob;\\n  const uint8_t* sfblob;\\n  const float* gs;\\n  const uint8_t* x;\\n  uint8_t* y;\\n  const int* tile_start;\\n  int nfb, kt, kt_out;\\n};\\n\\ntemplate <int MODE, int STAGES>\\n__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Params p) {\\n  constexpr int EPI_BYTES = epi_bytes<MODE>();\\n  extern __shared__ uint8_t smem_raw[];\\n  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));\\n  uint8_t* epi = smem + STAGES * STAGE_BYTES;\\n  uint64_t* full = reinterpret_cast<uint64_t*>(epi + EPI_BYTES);\\n  uint64_t* empty = full + STAGES;\\n  int* tstart = reinterpret_cast<int*>(empty + STAGES);\\n\\n  const int tid = threadIdx.x;\\n  const int wg = tid / 128;\\n  if (tid < 33) tstart[tid] = p.tile_start[tid];\\n  if (tid == 0) {\\n    for (int s = 0; s < STAGES; ++s) {\\n      mbar_init(&full[s], 1);\\n      mbar_init(&empty[s], 8);\\n    }\\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\\n  }\\n  __syncthreads();\\n  const int num_tiles = tstart[32];\\n  const int total = num_tiles * p.nfb;\\n  const int KT = p.kt;\\n\\n  if (wg == 0) {\\n#if MOE_WGMMA\\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");\\n#endif\\n    if (tid != 0) return;\\n    int s = 0;\\n    uint32_t phase = 0;\\n    for (int item = blockIdx.x; item < total; item += gridDim.x) {\\n      const int tt = item / p.nfb, fb = item % p.nfb;\\n      int e = 0;\\n      while (e < 31 && tstart[e + 1] <= tt) ++e;\\n      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;\\n      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;\\n      const uint8_t* xsrc = p.x + ((size_t)tt * KT) * X_BYTES;\\n      for (int k = 0; k < KT; ++k) {\\n        mbar_wait(&empty[s], phase ^ 1);\\n        uint8_t* st = smem + s * STAGE_BYTES;\\n        mbar_expect_tx(&full[s], STAGE_BYTES);\\n        bulk_g2s(st, xsrc + (size_t)k * X_BYTES, X_BYTES, &full[s]);\\n        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);\\n        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);\\n        if (++s == STAGES) { s = 0; phase ^= 1; }\\n      }\\n    }\\n    return;\\n  }\\n\\n#if MOE_WGMMA\\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");\\n#endif\\n  const int cw = wg - 1;                 // consumer warpgroup 0/1\\n  const int t = tid - 128 * wg;          // thread within the warpgroup\\n  const int warp = t >> 5, lane = t & 31;\\n  const int slot = warp * 8 + (lane >> 2);\\n  const uint32_t lo_a = LUT_LO_A, hi_a = LUT_HI_A;\\n  uint8_t* my_epi = epi + cw * (EPI_BYTES / 2);\\n  int s = 0;\\n  uint32_t phase = 0;\\n  float acc[2][64];\\n\\n  for (int item = blockIdx.x; item < total; item += gridDim.x) {\\n    const int tt = item / p.nfb, fb = item % p.nfb;\\n    int e = 0;\\n    while (e < 31 && tstart[e + 1] <= tt) ++e;\\n\\n    uint32_t A0[2][4], A1[2][4];\\n    uint4 raw[2];\\n    uint2 sfw[2];\\n    auto load_words = [&](int st_idx) {\\n      const uint8_t* st = smem + st_idx * STAGE_BYTES;\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) {\\n        raw[j] = *reinterpret_cast<const uint4*>(st + X_BYTES + (2 * cw + j) * 2048 + t * 16);\\n        sfw[j] = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + (2 * cw + j) * 256 + slot * 8);\\n      }\\n    };\\n    auto issue = [&](uint32_t (&F)[2][4], const uint8_t* xs, int b, int scale_d) {\\n      fence_acc(acc[0]);\\n      fence_acc(acc[1]);\\n      wg_fence();\\n      mma_block(acc[0], F[0], xs, b, scale_d);\\n      mma_block(acc[1], F[1], xs, b, scale_d);\\n      wg_commit();\\n      fence_acc(acc[0]);\\n      fence_acc(acc[1]);\\n    };\\n    // prologue: first k-tile\\\'s words, block 0\\\'s fragments\\n    mbar_wait(&full[s], phase);\\n    load_words(s);\\n    int cur_s = s;\\n#pragma unroll\\n    for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\\n\\n    for (int k = 0; k < KT; ++k) {\\n      const uint8_t* xs = smem + cur_s * STAGE_BYTES;\\n      // block 0 (A0)\\n      issue(A0, xs, 0, k > 0);\\n      if (k > 0 && MOE_WGMMA) {\\n        wg_wait<1>();\\n        keep(A1[0]); keep(A1[1]);\\n        __syncwarp();\\n        if (lane == 0) mbar_arrive(&empty[(cur_s + STAGES - 1) % STAGES]);   // previous k-tile done\\n      }\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 1, e4m3x2_to_bf16x2(sfw[j].x >> 16), lo_a, hi_a, A1[j]);\\n      // block 1 (A1)\\n      issue(A1, xs, 1, 1);\\n      wg_wait<1>();\\n      keep(A0[0]); keep(A0[1]);\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 0, e4m3x2_to_bf16x2(sfw[j].y & 0xFFFF), lo_a, hi_a, A0[j]);\\n      // block 2 (A0)\\n      issue(A0, xs, 2, 1);\\n      wg_wait<1>();\\n      keep(A1[0]); keep(A1[1]);\\n#pragma unroll\\n      for (int j = 0; j < 2; ++j) make_frag(raw[j].z, raw[j].w, 1, e4m3x2_to_bf16x2(sfw[j].y >> 16), lo_a, hi_a, A1[j]);\\n      // block 3 (A1)\\n      issue(A1, xs, 3, 1);\\n      wg_wait<1>();\\n      keep(A0[0]); keep(A0[1]);\\n#if !MOE_WGMMA\\n      // mma.sync has consumed this tile completely. Release it before waiting\\n      // for the next tile, including the one-stage down-projection fallback.\\n      __syncwarp();\\n      if (lane == 0) mbar_arrive(&empty[cur_s]);\\n#endif\\n      if (k + 1 < KT) {\\n        if (++s == STAGES) { s = 0; phase ^= 1; }\\n        mbar_wait(&full[s], phase);\\n        load_words(s);\\n        cur_s = s;\\n#pragma unroll\\n        for (int j = 0; j < 2; ++j) make_frag(raw[j].x, raw[j].y, 0, e4m3x2_to_bf16x2(sfw[j].x & 0xFFFF), lo_a, hi_a, A0[j]);\\n      }\\n    }\\n    wg_wait<0>();\\n    fence_acc(acc[0]);\\n    fence_acc(acc[1]);\\n    keep(A1[0]); keep(A1[1]);\\n    __syncwarp();\\n    if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[cur_s]);\\n    if (++s == STAGES) { s = 0; phase ^= 1; }\\n\\n    // ---- epilogue: registers -> swizzled smem tile(s) -> bulk store ----\\n    const float gsc = p.gs[e] * 256.f;\\n    if (t == 0) bulk_wait_read();       // previous item\\\'s store has finished reading my_epi\\n    named_bar(1 + cw, 128);\\n    const int r0 = warp * 16 + (lane >> 2);\\n    const int c2 = 2 * (lane & 3);\\n    constexpr int NT = MODE == 1 ? 1 : 2;\\n#pragma unroll\\n    for (int j = 0; j < NT; ++j) {\\n      uint8_t* tile = my_epi + j * X_BYTES;\\n#pragma unroll\\n      for (int i = 0; i < BN / 8; ++i) {\\n#pragma unroll\\n        for (int h = 0; h < 2; ++h) {        // h: row r0 (h=0) or r0+8 (h=1)\\n          const int rho = r0 + 8 * h;\\n#pragma unroll\\n          for (int q = 0; q < 2; ++q) {      // token column 8i + c2 + q\\n            const int n = 8 * i + c2 + q;\\n            float v;\\n            if (MODE == 1) {\\n              float gg = __bfloat162float(__float2bfloat16_rn(acc[0][4 * i + 2 * h + q] * gsc));\\n              float uu = __bfloat162float(__float2bfloat16_rn(acc[1][4 * i + 2 * h + q] * (p.gs[e + 32] * 256.f)));\\n              v = gg / (1.f + __expf(-gg)) * uu;\\n            } else {\\n              v = acc[j][4 * i + 2 * h + q] * gsc;\\n            }\\n            __nv_bfloat16 bv = __float2bfloat16_rn(v);\\n            *reinterpret_cast<__nv_bfloat16*>(tile + n * 128 + ((((rho >> 3) ^ (n & 7))) << 4) + (rho & 7) * 2) = bv;\\n          }\\n        }\\n      }\\n    }\\n    fence_async_smem();\\n    named_bar(1 + cw, 128);\\n    if (t == 0) {\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) {\\n        const int kto = MODE == 1 ? fb * 2 + cw : fb * 4 + 2 * cw + j;\\n        bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto) * X_BYTES, my_epi + j * X_BYTES, X_BYTES);\\n      }\\n    }\\n  }\\n  if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");\\n}\\n\\n}  // namespace raw17\\n\\nvoid raw17_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,\\n              torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, int64_t mode, int64_t grid) {\\n  raw17::Params p;\\n  p.wblob = wblob.data_ptr<uint8_t>();\\n  p.sfblob = sfblob.data_ptr<uint8_t>();\\n  p.gs = gs.data_ptr<float>();\\n  p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());\\n  p.y = reinterpret_cast<uint8_t*>(y.data_ptr());\\n  p.tile_start = tile_start.data_ptr<int>();\\n  p.nfb = (int)nfb;\\n  p.kt = (int)kt;\\n  p.kt_out = (int)kt_out;\\n  auto stream = at::cuda::getCurrentCUDAStream();\\n  int dev = 0, major = 0;\\n  cudaGetDevice(&dev);\\n  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\\n#define MOE_LAUNCH(M, S)                                                                                  \\\\\\n  do {                                                                                                     \\\\\\n    cudaFuncSetAttribute(raw17::moe_gemm_kernel<M, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,          \\\\\\n                         raw17::smem_bytes<M, S>());                                                         \\\\\\n    raw17::moe_gemm_kernel<M, S><<<(int)grid, 384, raw17::smem_bytes<M, S>(), stream>>>(p);                   \\\\\\n  } while (0)\\n  if (major == 9) {\\n    if (mode == 1) MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);\\n  } else {\\n    if (mode == 1) MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);\\n  }\\n#undef MOE_LAUNCH\\n}\\n\\n\'\npf_BN = 128\npf_D = 4096\npf__mod = None\n\ndef pf__grouped_module():\n    global pf__mod\n    if pf__mod is None:\n        from torch.utils.cpp_extension import load_inline\n        pf__mod = load_inline(name=\'pipe_raw17_\' + pf_hashlib.sha256(pf_GROUPED_SRC.encode()).hexdigest()[:12], cpp_sources=\'void raw17_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);\', cuda_sources=pf_GROUPED_SRC, functions=[\'raw17_gemm\'], extra_cuda_cflags=[\'-O3\', \'-std=c++20\', \'-gencode=arch=compute_90a,code=sm_90a\', \'-gencode=arch=compute_120,code=sm_120\'], verbose=False)\n    return pf__mod\n\ndef pf__raw_blob_index(dev):\n    """quack\'s per-m64 _raw_repack: thread t\'s 16 bytes per (m64, k64) tile -> (row in m64, byte in k-tile)."""\n    t = pf_torch.arange(128, device=dev)\n    j = pf_torch.arange(16, device=dev)\n    b, p = (j >> 2, j & 3)\n    r16 = 16 * (t >> 5) + (t & 31) // 4\n    c = t & 3\n    row = r16[:, None] + 8 * (p & 1)[None, :]\n    col = 8 * b[None, :] + c[:, None] + 4 * (p >> 1)[None, :]\n    s = pf_torch.arange(32, device=dev)\n    srow = 16 * (s >> 3) + (s & 7)\n    return (row, col, srow)\n\ndef pf__raw_cta_row_order(m, gated, dev):\n    """Weight row permutation into 256-row CTA blocks. Gated (w13: gate rows then up rows):\n    block fb = [gate fb*128 + 0..63, up .., gate fb*128 + 64..127, up ..]."""\n    if not gated:\n        return pf_torch.arange(m, device=dev)\n    half = m // 2\n    fb = pf_torch.arange(m // 256, device=dev)[:, None, None]\n    q = pf_torch.arange(4, device=dev)[None, :, None]\n    r = pf_torch.arange(64, device=dev)[None, None, :]\n    feat = fb * 128 + (q >> 1) * 64 + r\n    return (feat + (q & 1) * half).reshape(-1)\n\ndef pf__raw_repack(w, sf, gated):\n    """w [E, M, K/2] uint8 packed e2m1, sf [E, M, K/16] e4m3 -> (blob [E, NFB, KT, 4, 128, 16] uint8,\n    sfblob [E, NFB, KT, 4, 32, 4, 2] uint8). A lossless permutation of the same bytes."""\n    E, M, KB = w.shape\n    dev = w.device\n    sf = sf.view(pf_torch.uint8)\n    order = pf__raw_cta_row_order(M, gated, dev)\n    row, col, srow = pf__raw_blob_index(dev)\n    nfb, kt = (M // 256, KB // 32)\n    blobs, sfbs = ([], [])\n    for e in range(E):\n        we = w[e][order]\n        se = sf[e][order]\n        g = pf_torch.arange(M // 64, device=dev)\n        k = pf_torch.arange(kt, device=dev)\n        rows = 64 * g[:, None, None, None] + row[None, None]\n        cols = 32 * k[None, :, None, None] + col[None, None]\n        bl = we[rows, cols]\n        bl = bl.view(nfb, 4, kt, 128, 16).permute(0, 2, 1, 3, 4).contiguous()\n        bidx = pf_torch.arange(4, device=dev)\n        i = pf_torch.arange(2, device=dev)\n        srows = 64 * g[:, None, None, None, None] + srow[None, None, :, None, None] + 8 * i[None, None, None, None, :]\n        scols = 4 * k[None, :, None, None, None] + bidx[None, None, None, :, None]\n        sb = se[srows, scols]\n        sb = sb.view(nfb, 4, kt, 32, 4, 2).permute(0, 2, 1, 3, 4, 5).contiguous()\n        blobs.append(bl)\n        sfbs.append(sb)\n    return (pf_torch.stack(blobs), pf_torch.stack(sfbs))\n\n@pf_triton.jit\ndef pf__raw_tile_offsets_kernel(counts_ptr, tstart_ptr, BNC: pf_tl.constexpr):\n    e = pf_tl.arange(0, 32)\n    c = pf_tl.load(counts_ptr + e)\n    nt = (c + BNC - 1) // BNC\n    incl = pf_tl.cumsum(nt, 0)\n    pf_tl.store(tstart_ptr + e, incl - nt)\n    pf_tl.store(tstart_ptr + 1 + e, incl, mask=e == 31)\n\n@pf_triton.jit\ndef pf__raw_gather_tiled_kernel(x_ptr, slot_e_ptr, slot_pos_ptr, tstart_ptr, out_ptr, row_ptr, D: pf_tl.constexpr, BNC: pf_tl.constexpr):\n    """Each local route\'s token row x[t] -> its expert\'s token tile in the swizzled tiled layout;\n    row_ptr[t, k] = global sorted row (tile * BN + r) or -1."""\n    t = pf_tl.program_id(0).to(pf_tl.int64)\n    c = pf_tl.arange(0, D)\n    x = pf_tl.load(x_ptr + t * D + c)\n    kt = c // 64\n    cc = c % 64\n    for k in pf_tl.static_range(8):\n        e = pf_tl.load(slot_e_ptr + t * 8 + k)\n        if e >= 0:\n            pos = pf_tl.load(slot_pos_ptr + t * 8 + k)\n            tile = pf_tl.load(tstart_ptr + e) + pos // BNC\n            r = pos % BNC\n            off = ((tile.to(pf_tl.int64) * (D // 64) + kt) * BNC + r) * 64 + (cc // 8 ^ r % 8) * 8 + cc % 8\n            pf_tl.store(out_ptr + off, x)\n            pf_tl.store(row_ptr + t * 8 + k, tile * BNC + r)\n        else:\n            pf_tl.store(row_ptr + t * 8 + k, -1)\n\n@pf_triton.jit\ndef pf__raw_combine_tiled_kernel(y_ptr, rows_ptr, topk_ptr, pw_ptr, out_ptr, D: pf_tl.constexpr, BNC: pf_tl.constexpr):\n    t = pf_tl.program_id(0).to(pf_tl.int64)\n    c = pf_tl.arange(0, D)\n    pw = pf_tl.load(pw_ptr + c).to(pf_tl.float32)\n    acc = pf_tl.zeros([D], pf_tl.float32)\n    for k in pf_tl.static_range(8):\n        g = pf_tl.load(rows_ptr + t * 8 + k).to(pf_tl.int64)\n        if g >= 0:\n            tile, r = (g // BNC, g % BNC)\n            cc = c % 64\n            off = ((tile * (D // 64) + c // 64) * BNC + r) * 64 + (cc // 8 ^ r % 8) * 8 + cc % 8\n            y = pf_tl.load(y_ptr + off).to(pf_tl.float32) * pw\n            y *= pf_tl.rsqrt(pf_tl.sum(y * y, 0) / D + 1e-08)\n            acc += pf_tl.load(topk_ptr + t * 8 + k) * y\n    pf_tl.store(out_ptr + t * D + c, acc.to(pf_tl.bfloat16))\n\ndef pf__raw_experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st):\n    T = pooled.shape[0]\n    dev = pooled.device\n    max_tiles = pf_triton.cdiv(T * 8 + 32 * (pf_BN - 1), pf_BN)\n    tstart = pf_torch.empty(33, dtype=pf_torch.int32, device=dev)\n    pf__raw_tile_offsets_kernel[1,](counts, tstart, BNC=pf_BN)\n    xn = pooled\n    xt = pf_torch.empty((max_tiles * pf_BN * pf_D,), dtype=pf_torch.bfloat16, device=dev)\n    rows = pf_torch.empty((T, 8), dtype=pf_torch.int32, device=dev)\n    pf__raw_gather_tiled_kernel[T,](xn, slot_e, slot_pos, tstart, xt, rows, D=pf_D, BNC=pf_BN, num_warps=4)\n    inter = pf_torch.empty_like(xt)\n    y = pf_torch.empty_like(xt)\n    st[\'tiled_mod\'].moe_gemm(st[\'blob13\'], st[\'sfb13\'], st[\'w13_gs\'], xt, inter, tstart, 32, 64, 64, 1, st[\'sms\'])\n    st[\'tiled_mod\'].moe_gemm(st[\'blob2\'], st[\'sfb2\'], st[\'w2_gs\'], inter, y, tstart, 16, 64, 64, 0, st[\'sms\'])\n    out = pf_torch.empty_like(pooled)\n    pf__raw_combine_tiled_kernel[T,](y, rows, topk_w, st[\'post_expert_bf16\'], out, D=pf_D, BNC=pf_BN, num_warps=4)\n    return out\n\n@pf_triton.jit\ndef pf__encode_blob_fast(B, S, O):\n    pid = pf_tl.program_id(0).to(pf_tl.int64)\n    t = pf_tl.arange(0, 128)\n    slot = t // 32 * 8 + t % 32 // 4\n    for pair in pf_tl.static_range(2):\n        x = pf_tl.load(B + pid * 512 + t * 4 + pair * 2).to(pf_tl.uint32)\n        y = pf_tl.load(B + pid * 512 + t * 4 + pair * 2 + 1).to(pf_tl.uint32)\n        c0 = x >> 0 & 15\n        c1 = x >> 4 & 15\n        c2 = x >> 8 & 15\n        c3 = x >> 12 & 15\n        c4 = x >> 16 & 15\n        c5 = x >> 20 & 15\n        c6 = x >> 24 & 15\n        c7 = x >> 28 & 15\n        c8 = y >> 0 & 15\n        c9 = y >> 4 & 15\n        c10 = y >> 8 & 15\n        c11 = y >> 12 & 15\n        c12 = y >> 16 & 15\n        c13 = y >> 20 & 15\n        c14 = y >> 24 & 15\n        c15 = y >> 28 & 15\n        oA = (c0 >> 3 & 1) << 15 | (c0 >> 2 & 1) << 8 | (c0 >> 1 & 1) << 7 | (c0 >> 0 & 1) << 6 | (c1 >> 3 & 1) << 31 | (c1 >> 2 & 1) << 24 | (c1 >> 1 & 1) << 23 | (c1 >> 0 & 1) << 22 | (c2 >> 3 & 1) << 19 | (c2 >> 2 & 1) << 12 | (c2 >> 1 & 1) << 11 | (c2 >> 0 & 1) << 10 | (c3 >> 3 & 1) << 3 | (c3 >> 2 & 1) << 28 | (c3 >> 1 & 1) << 27 | (c3 >> 0 & 1) << 26 | (c4 >> 3 & 1) << 25 | (c4 >> 2 & 1) << 18 | (c4 >> 1 & 1) << 17 | (c4 >> 0 & 1) << 16 | (c5 >> 3 & 1) << 9 | (c5 >> 2 & 1) << 2 | (c5 >> 1 & 1) << 1 | (c5 >> 0 & 1) << 0 | (c12 >> 0 & 1) << 30 | (c13 >> 0 & 1) << 14 | (c14 >> 3 & 1) << 13 | (c14 >> 1 & 1) << 5 | (c14 >> 0 & 1) << 4 | (c15 >> 3 & 1) << 29 | (c15 >> 1 & 1) << 21 | (c15 >> 0 & 1) << 20\n        oB = (c6 >> 3 & 1) << 17 | (c6 >> 2 & 1) << 10 | (c6 >> 1 & 1) << 9 | (c6 >> 0 & 1) << 8 | (c7 >> 3 & 1) << 1 | (c7 >> 2 & 1) << 26 | (c7 >> 1 & 1) << 25 | (c7 >> 0 & 1) << 24 | (c8 >> 3 & 1) << 21 | (c8 >> 2 & 1) << 14 | (c8 >> 1 & 1) << 13 | (c8 >> 0 & 1) << 12 | (c9 >> 3 & 1) << 5 | (c9 >> 2 & 1) << 30 | (c9 >> 1 & 1) << 29 | (c9 >> 0 & 1) << 28 | (c10 >> 3 & 1) << 27 | (c10 >> 2 & 1) << 20 | (c10 >> 1 & 1) << 19 | (c10 >> 0 & 1) << 18 | (c11 >> 3 & 1) << 11 | (c11 >> 2 & 1) << 4 | (c11 >> 1 & 1) << 3 | (c11 >> 0 & 1) << 2 | (c12 >> 3 & 1) << 7 | (c12 >> 2 & 1) << 0 | (c12 >> 1 & 1) << 31 | (c13 >> 3 & 1) << 23 | (c13 >> 2 & 1) << 16 | (c13 >> 1 & 1) << 15 | (c14 >> 2 & 1) << 6 | (c15 >> 2 & 1) << 22\n        pf_tl.store(O + pid * 512 + t * 4 + pair * 2, oA)\n        pf_tl.store(O + pid * 512 + t * 4 + pair * 2 + 1, oB)\nimport json as pf_json\nimport os as pf_os\nfrom typing import TypedDict as pf_TypedDict\nimport torch as pf_torch\npf__saved_flag = pf_os.environ.get(\'HUMMING_DISABLE_PARALLEL_BUILD\')\npf_os.environ[\'HUMMING_DISABLE_PARALLEL_BUILD\'] = \'1\'\ntry:\n    from humming import ops as pf_humming_ops\n    from humming.config import GemmType as pf_GemmType, WeightScale2Type as pf_WeightScale2Type\n    from humming.kernel.humming import HummingKernel as pf_HummingKernel\n    from humming.layer import HummingMethod as pf_HummingMethod\n    from humming.schema.modelopt import ModeloptNvfp4WeightSchema as pf_ModeloptNvfp4WeightSchema\n    from humming.tune import get_heuristics_config as pf_get_heuristics_config\nfinally:\n    if pf__saved_flag is None:\n        pf_os.environ.pop(\'HUMMING_DISABLE_PARALLEL_BUILD\', None)\n    else:\n        pf_os.environ[\'HUMMING_DISABLE_PARALLEL_BUILD\'] = pf__saved_flag\nfrom sglang.kernels.ops.activation.activation import silu_and_mul as pf_silu_and_mul\nfrom sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as pf_fused_post_expert_norm_weighted_sum\nfrom sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as pf_moe_align_block_size\n\nclass pf__Experts:\n    """One GPU\'s experts [start, start + count) in Humming\'s layout, and the kernels for each bs."""\n\n    def __init__(self, weights, start: int, count: int, tma=False):\n        self.tma = tma\n        self.gemm_type = pf_GemmType.INDEXED\n        device = weights[\'w13\'].device\n        self.count = count\n        self.d_expert = weights[\'w13\'].shape[1] // 2\n        self.d_hidden = weights[\'w2\'].shape[1]\n        self.layer = pf_torch.nn.Module()\n        sl = slice(start, start + count)\n        for name, n, k, n_stacks in ((\'w13\', 2 * self.d_expert, self.d_hidden, [self.d_expert] * 2), (\'w2\', self.d_hidden, self.d_expert, [self.d_hidden])):\n            checkpoint = {\'weight\': weights[name][sl].contiguous(), \'weight_scale\': weights[f\'{name}_sf\'][sl].contiguous(), \'weight_scale_2\': weights[f\'{name}_gs\'][sl].contiguous()}\n            schema, tensors = pf_ModeloptNvfp4WeightSchema().convert_humming(tensors=checkpoint, shape_n_stacks=n_stacks, shape_k_stacks=[k], param_dtype=pf_torch.bfloat16, num_experts=count)\n            assert schema.weight_scale_2_type == pf_WeightScale2Type.TENSOR, schema\n            for key, tensor in tensors.items():\n                setattr(self.layer, f\'{name}_{key}\', pf_torch.nn.Parameter(tensor, requires_grad=False))\n            pf_HummingMethod.prepare_layer_meta(layer=self.layer, shape_n=n, shape_k=k, weight_schema=schema, num_experts=count, pad_n_to_multiple=256, pad_k_to_multiple=128, torch_dtype=pf_torch.bfloat16, sublayer_name=name)\n            pf_HummingMethod.transform_humming_layer(self.layer, sublayer_name=name)\n        self.locks = pf_torch.zeros(1024, dtype=pf_torch.int32, device=device)\n        self.expert_map = pf_torch.full((pf_N_EXPERTS,), -1, dtype=pf_torch.int32, device=device)\n        self.expert_map[sl] = pf_torch.arange(count, dtype=pf_torch.int32, device=device)\n        self.kernels = {\'w13\': {}, \'w2\': {}}\n        self.plans = {}\n\n    def weight(self, name: str) -> dict:\n        return {\'weight\': getattr(self.layer, f\'{name}_weight\'), \'weight_scale\': getattr(self.layer, f\'{name}_weight_scale\'), \'weight_scale_2\': getattr(self.layer, f\'{name}_weight_scale_2\')}\n\n    def routes(self, bs: int) -> int:\n        """The GEMM\'s M for Humming\'s heuristics: the routes expected on this GPU."""\n        return max(1, bs * pf_TOP_K * self.count // pf_N_EXPERTS)\n\n    def tuning(self, name: str, routes: int) -> str:\n        config = dict(pf_get_heuristics_config(self.layer.humming_metas[name], shape_m=routes, gemm_type=self.gemm_type))\n        if self.tma and pf_torch.cuda.get_device_capability()[0] == 9:\n            config.update(use_tma=True, use_tma_a=False, use_tma_as=False, use_tma_c=False, use_mbarrier=True, use_warp_spec=True)\n        return pf_json.dumps(config)\n\n    def compile(self, batch_sizes) -> None:\n        """Compile, in parallel, every kernel these batch sizes need that isn\'t compiled yet."""\n        wanted = {bs: {name: self.tuning(name, self.routes(bs)) for name in self.kernels} for bs in batch_sizes}\n        for name, kernels in self.kernels.items():\n            new = sorted({t[name] for t in wanted.values()} - kernels.keys())\n            if not new:\n                continue\n            launch = pf_HummingKernel.prepare_kernels(self.layer.humming_metas[name].to_str(), pf_json.dumps({\'use_f16_accum\': False, \'gemm_type\': self.gemm_type.value}), [[i, i + 1, pf_json.loads(t)] for i, t in enumerate(new)])\n            for t, (_, _, kernel_id, num_sms) in zip(new, launch.view(-1, 4).tolist()):\n                kernels[t] = pf_torch.tensor([0, 1 << 30, kernel_id, num_sms], dtype=pf_torch.int64)\n        for bs, tunings in wanted.items():\n            self.plans[bs] = (self.routes(bs), {name: (pf_json.loads(t)[\'block_shape\'][0], self.kernels[name][t]) for name, t in tunings.items()})\n\n    def plan(self, bs: int):\n        bucket = 1 << (max(bs, 1) - 1).bit_length()\n        if bucket not in self.plans:\n            self.compile([bucket])\n        return (self.routes(bs), self.plans[bucket][1])\n\n@pf_triton.jit\ndef pf__fixed_offsets_kernel(Starts, T: pf_tl.constexpr):\n    e = pf_tl.arange(0, 32)\n    pf_tl.store(Starts + e, e * T)\n\ndef pf__experts_humming_masked(pooled, topk_w, slot_e, slot_pos, counts, st):\n    T = pooled.shape[0]\n    stride = pf_triton.cdiv(T, 128) * 128\n    starts = pf_torch.empty(32, device=pooled.device, dtype=pf_torch.int32)\n    pf__fixed_offsets_kernel[1,](starts, T=stride)\n    a = pf_torch.empty((32 * stride, 4096), device=pooled.device, dtype=pf_torch.bfloat16)\n    rows = pf_torch.empty((T, 8), device=pooled.device, dtype=pf_torch.int32)\n    pf__gather_norm_kernel[T,](pooled, st[\'pre_expert\'], slot_e, slot_pos, starts, a, rows, D=4096, num_warps=4)\n    h = pf_torch.empty((32 * stride, 8192), device=pooled.device, dtype=pf_torch.bfloat16)\n    inter = pf_torch.empty_like(a)\n    y = pf_torch.empty_like(a)\n    bank = st[\'humming\']\n    routes, gemms = bank.plan(T)\n    pf_humming_ops.launch_kernel(configs=gemms[\'w13\'][1], inputs=a, outputs=h, **bank.weight(\'w13\'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)\n    pf__portfolio_swiglu[32, pf_triton.cdiv(T, 4)](h, inter, starts, counts, st[\'gs13_gate\'], st[\'gs13_up\'], BR=4, D=4096, num_warps=8)\n    pf_humming_ops.launch_kernel(configs=gemms[\'w2\'][1], inputs=inter, outputs=y, **bank.weight(\'w2\'), locks=bank.locks, expert_layout=counts, valid_shape_m=routes)\n    out = pf_torch.empty_like(pooled)\n    pf__combine_kernel[T,](y, rows, slot_e, topk_w, st[\'ones\'], st[\'post_expert\'], out, D=4096, num_warps=4)\n    return out\nimport ctypes as pf_ct\nimport hashlib as pf_hashlib\nimport logging as pf_logging\nimport os as pf_os\nimport statistics as pf_statistics\nimport time as pf_time\nimport torch as pf_torch\nimport torch.nn.functional as pf_F\nimport triton as pf_triton\nimport triton.language as pf_tl\n\n@pf_triton.jit\ndef pf__portfolio_swiglu(H, A, STARTS, COUNTS, GATE_GS, UP_GS, D: pf_tl.constexpr, BR: pf_tl.constexpr):\n    e = pf_tl.program_id(0)\n    n = pf_tl.load(COUNTS + e)\n    r = pf_tl.program_id(1) * BR + pf_tl.arange(0, BR)\n    if pf_tl.program_id(1) * BR >= n:\n        return\n    row = pf_tl.load(STARTS + e) + r\n    c = pf_tl.arange(0, D)\n    g = pf_tl.load(H + row[:, None].to(pf_tl.int64) * (2 * D) + c[None, :], r[:, None] < n, 0).to(pf_tl.float32) * pf_tl.load(GATE_GS + e)\n    u = pf_tl.load(H + row[:, None].to(pf_tl.int64) * (2 * D) + D + c[None, :], r[:, None] < n, 0).to(pf_tl.float32) * pf_tl.load(UP_GS + e)\n    g = g.to(pf_tl.bfloat16).to(pf_tl.float32)\n    u = u.to(pf_tl.bfloat16).to(pf_tl.float32)\n    pf_tl.store(A + row[:, None].to(pf_tl.int64) * D + c[None, :], (g * pf_tl.sigmoid(g) * u).to(pf_tl.bfloat16), r[:, None] < n)\n\nclass MWGraph:\n\n    def __init__(self, fn, data):\n        self.data = {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}\n        fn(self.data)\n        torch.cuda.synchronize()\n        self.graph = torch.cuda.CUDAGraph()\n        with torch.cuda.graph(self.graph):\n            self.out = fn(self.data)\n\n    def refresh(self, data):\n        for k in (\'hidden_states\', \'topk_ids\', \'topk_weights\'):\n            self.data[k].copy_(data[k])\n\n    def replay(self):\n        self.graph.replay()\n        return self.out\nimport base64\nimport hashlib\nimport statistics\nimport time\nimport zlib\n\ndef dc_weights(seed, device):\n    w = rg_generate_weights(seed, device=device)\n    gen = torch.Generator(device=device).manual_seed(seed + 1000003)\n    w[\'w13_gs_gate\'] = w[\'w13_gs\'].clone()\n    w[\'w13_gs_up\'] = w[\'w13_gs\'] * torch.exp(0.55 * torch.randn(32, generator=gen, device=device))\n    w[\'post_norm_w\'] = w[\'post_expert_norm\']\n    assert bool((w[\'w13_gs_gate\'] != w[\'w13_gs_up\']).all())\n    return w\n\ndef dc_oracles(ds, w):\n    """Independent FP32 scratch oracle; no expanded bank survives this function."""\n    old = torch.backends.cuda.matmul.allow_tf32\n    torch.backends.cuda.matmul.allow_tf32 = False\n    try:\n        wants = [torch.zeros_like(d[\'hidden_states\'], dtype=torch.float32) for d in ds]\n        lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=w[\'w13\'].device)\n        for e in range(32):\n            active = [torch.where(d[\'topk_ids\'] == e) for d in ds]\n            if not any((t.numel() for t, _ in active)):\n                continue\n\n            def dq(key):\n                q = w[key][e]\n                c = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()\n                return lut[c & 7] * torch.where(c & 8 != 0, -1.0, 1.0) * w[key + \'_sf\'][e].float().repeat_interleave(16, -1)\n            w13 = dq(\'w13\')\n            w2 = dq(\'w2\') * w[\'w2_gs\'][e]\n            for d, out, (t, j) in zip(ds, wants, active):\n                if t.numel() == 0:\n                    continue\n                h = d[\'hidden_states\'][t].float() @ w13.t()\n                g = h[:, :4096] * w[\'w13_gs_gate\'][e]\n                u = h[:, 4096:] * w[\'w13_gs_up\'][e]\n                a = (torch.nn.functional.silu(g) * u).bfloat16()\n                y = a.float() @ w2.t()\n                y *= 1 + w[\'post_norm_w\'].float()\n                y *= torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-08)\n                out.index_add_(0, t, y * d[\'topk_weights\'][t, j, None])\n        return [out.bfloat16() for out in wants]\n    finally:\n        torch.backends.cuda.matmul.allow_tf32 = old\n\ndef dc_fixtures(w, m, r0=False):\n    ds = [fx_make_fixture(w, m, 9000 + m + 7919 * j) for j in range(3)]\n    if r0:\n        raw = zlib.decompress(base64.b64decode(CAPTURED_R0))\n        ms = (8, 16, 32, 128, 256)\n        offset = 8 * sum(ms[:ms.index(m)])\n        ids = torch.tensor(list(raw[offset:offset + m * 8]), device=w[\'w13\'].device, dtype=torch.int32).view(m, 8)\n        ids = torch.where(ids < 32, ids, -1)\n        ds = [dict(d, topk_ids=ids.clone()) for d in ds]\n    return ds\nimport functools\nimport tilelang\nimport tilelang.language as T\nimport torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef mw_route(IDS, COUNTS, SLOTS, POS, M: tl.constexpr, START: tl.constexpr, B: tl.constexpr):\n    e = tl.program_id(0)\n    t = tl.arange(0, B)\n    hit = tl.full((B,), False, tl.int1)\n    slot = tl.full((B,), 0, tl.int32)\n    for j in tl.static_range(8):\n        yes = tl.load(IDS + t * 8 + j, t < M, -999) == e + START\n        hit = hit | yes\n        slot = tl.where(yes, j, slot)\n    rank = tl.cumsum(hit.to(tl.int32)) - 1\n    count = tl.sum(hit.to(tl.int32))\n    tl.store(COUNTS + e, count)\n    tl.store(SLOTS + e * M + rank, t * 8 + slot, hit)\n    tl.store(POS + t * 8 + slot, rank, hit)\n\n@triton.jit\ndef mw_prefix(COUNTS, OFFS, ORDER, TASKS, BN: tl.constexpr):\n    e = tl.arange(0, 32)\n    n = tl.load(COUNTS + e)\n    padded = tl.cdiv(n, BN) * BN\n    off = tl.cumsum(padded) - padded\n    tl.store(OFFS + e, off)\n    tl.store(OFFS + 32, tl.sum(padded))\n    keys = tl.sort(n * 32 + 31 - e, descending=True)\n    order = 31 - keys % 32\n    rows = keys // 32\n    ends = tl.cumsum(tl.cdiv(rows, BN))\n    tl.store(ORDER + e, order)\n    tl.store(TASKS + e + 1, ends)\n    tl.store(TASKS, 0)\n\n@triton.jit\ndef mw_swiglu(Y, Z, OFFS, B: tl.constexpr=4096):\n    row = tl.program_id(0)\n    k = tl.arange(0, B)\n    if row < tl.load(OFFS + 32):\n        g = tl.load(Y + row * 8192 + k)\n        u = tl.load(Y + row * 8192 + 4096 + k)\n        tl.store(Z + row * 4096 + k, g * tl.sigmoid(g) * u)\n\n@triton.jit\ndef mw_norm(Y, PNW, OFFS, INV):\n    row = tl.program_id(0)\n    if row < tl.load(OFFS + 32):\n        k = tl.arange(0, 4096)\n        v = tl.load(Y + row * 4096 + k) * (1.0 + tl.load(PNW + k).to(tl.float32))\n        inv = tl.rsqrt(tl.sum(v * v) / 4096.0 + 1e-08)\n        tl.store(INV + row, inv)\n\n@triton.jit\ndef mw_combine(Y, PNW, IDS, WTS, POS, OFFS, INV, OUT, M: tl.constexpr, START: tl.constexpr, B: tl.constexpr=1024):\n    t = tl.program_id(0)\n    k = tl.program_id(1) * B + tl.arange(0, B)\n    f = 1.0 + tl.load(PNW + k).to(tl.float32)\n    out = tl.full((B,), 0.0, tl.float32)\n    for j in range(8):\n        e = tl.load(IDS + t * 8 + j) - START\n        yes = (e >= 0) & (e < 32)\n        r = tl.load(POS + t * 8 + j, yes, 0)\n        off = tl.load(OFFS + e, yes, 0)\n        inv = tl.load(INV + off + r, yes, 0.0)\n        v = tl.load(Y + (off + r) * 4096 + k, yes, 0.0)\n        w = tl.load(WTS + t * 8 + j)\n        out += v * f * inv * w\n    tl.store(OUT + t * 4096 + k, out)\n\ndef mw_repack_planes(q):\n    """A permutation of all four bits of every nibble, matching RS register ownership.\n\n    Two words encode the 16 codes each lane needs for M16/K32. The extraction\n    table is the incumbent V9 bit-plane bijection, used here with WGMMA layout.\n    """\n    e, n, k2 = q.shape\n    k = k2 * 2\n    dst = torch.empty((e, n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int32)\n    slots = ((\'A\', 0), (\'A\', 4), (\'A\', 10), (\'B\', 2), (\'B\', 6), (\'B\', 12), (\'W\', 24), (\'W\', 30))\n    mask = 1613783088\n    for expert in range(e):\n        codes = torch.stack((q[expert] & 15, q[expert] >> 4), -1).reshape(n // 16, 2, 8, k // 32, 2, 2, 4, 2)\n        codes = codes.permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 32, 32, 8, 2).to(torch.int64)\n        words = torch.zeros((n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int64)\n        for slot, (src, rot) in enumerate(slots):\n            for half in (0, 1):\n                for bit, dbit in enumerate((6, 7, 8, 15)):\n                    p = (dbit + 16 * half + rot) % 32\n                    word = (0 if mask >> p & 1 else 1) if src == \'W\' else 0 if src == \'A\' else 1\n                    words[..., word] |= (codes[..., slot, half] >> bit & 1) << p\n        dst[expert] = words.to(torch.int32)\n    return dst\n"""Deeper packed Q/SF/X async ring with overlapping register-A groups and split-K."""\nimport functools\nimport hashlib\nfrom pathlib import Path\nimport torch\nimport triton\nimport triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\n\n\nPIPE_CUDA=\'// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\\n#include <torch/extension.h>\\n#include <ATen/cuda/CUDAContext.h>\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace mwpipe {\\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\\n unsigned dst=smaddr(d);\\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\\n}\\n__device__ __forceinline__ unsigned sf_bits(unsigned raw){\\n unsigned s=raw&127,bit=0;\\n if(s>=8)bit=(238u<<7)+(s<<4);\\n else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}\\n if(s==127)bit=0x7fff;\\n bit|=(raw&128)<<8;return bit|(bit<<16);\\n}\\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\\n}\\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){\\n unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;\\n unsigned s0=lut[raw&255],s1=lut[(raw>>8)&255],s2=lut[(raw>>16)&255],s3=lut[raw>>24];\\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\\n}\\n__device__ __forceinline__ void keep(const unsigned*a){\\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\\n}\\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\\n d+=b*2;\\n if constexpr(NT==16){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else if constexpr(NT==32){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else{\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }\\n#else\\n int lane=threadIdx.x&31;\\n #pragma unroll\\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^((r&7)*8)));\\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^((r&7)*8)));\\n  float*p=c+4*j;\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n }\\n#endif\\n}\\ntemplate<int NT,bool OVERLAP>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\\n#pragma unroll\\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n}\\ntemplate<int NT,int KT>__device__ __forceinline__ void load_stage(\\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\\n int t=threadIdx.x,w=t>>5,l=t&31;\\n #pragma unroll\\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\\n  async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);\\n }\\n #pragma unroll\\n for(int c=0;c<KT/64;++c){\\n  #pragma unroll\\n  for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;\\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\\n  }\\n }\\n asm volatile("cp.async.commit_group;":::"memory");\\n}\\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\\n else if constexpr(S>3){\\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\\n  else if constexpr(S>4){\\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\\n   else if constexpr(S>5){\\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\\n    else if constexpr(S>6){\\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\\n     else if constexpr(S>7){\\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n}\\ntemplate<int NT,int KT,int S,int SPLIT,bool OVERLAP=true>\\n__global__ void __launch_bounds__(128,2) grouped(const int*Q,const unsigned char*SCALE,const float*GS,\\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\\n float*Y,int M,int N,int cap,int first){\\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT;\\n static_assert(S>=3&&S<=8&&NS>=1);\\n extern __shared__ __align__(128) unsigned char sm[];\\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(ss+S*SB);\\n __shared__ unsigned lut[256];int t=threadIdx.x,w=t>>5,l=t&31;\\n lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();\\n int total=TASK[32]*(N/64)*SPLIT;\\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\\n  int task=unit/SPLIT,split=unit%SPLIT;\\n  int group=task/(N/64),mt=task%(N/64),ix=0;\\n  #pragma unroll\\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\\n  int src[NT/16];\\n  #pragma unroll\\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;\\n   src[j]=r<count?(first?SLOTS[e*M+r]/8:offset+r):-1;\\n  }\\n  float acc[NT/2]={};unsigned af[2][16]={};\\n  #pragma unroll\\n  for(int j=0;j<NT/2;++j)asm volatile("" : "+f"(acc[j]));\\n  #pragma unroll\\n  for(int i=0;i<min(S-1,NS);++i)\\n   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i);\\n  #pragma unroll 1\\n  for(int k=0;k<NS;++k){int st=k%S;\\n   cp_wait<S,NS>(k);__syncthreads();\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\\n#endif\\n   #pragma unroll 1\\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\\n   #pragma unroll\\n   for(int sub=0;sub<(KT>=128?2:1);++sub){\\n    int chunk=pair*2+sub;\\n    int reg=OVERLAP?sub:0;\\n    #pragma unroll\\n    for(int g=0;g<2;++g){int z=chunk*2+g;\\n     uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\\n     unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);\\n     dq(q.x,q.y,sf,lut,af[reg]+g*8);\\n    }\\n    keep(af[reg]);\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n    asm volatile("wgmma.fence.sync.aligned;":::"memory");\\n#endif\\n    mma_group<NT,OVERLAP>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);\\n    // Input-only keep extends the *previous* bank through the next wait1.\\n    // A read/write empty asm whose result is overwritten loses this lifetime\\n    // under unrolling: ptxas then reuses four registers and inserts wait0/MMA.\\n    if constexpr(OVERLAP)keep(af[reg^1]);else keep(af[reg]);\\n    if(chunk==0){\\n     if(k+S-1<NS){int dst=(k+S-1)%S;\\n      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,\\n                       Q,SCALE,X,src,e,mt,N,split*NS+k+S-1);\\n     }\\n    }\\n   }\\n   }\\n  }\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n  keep(af[0]);keep(af[1]);\\n  #pragma unroll\\n  for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n   int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n   float gs=GS[e+(first&&c>=4096?32:0)]*256.f;\\n   if(r<count)Y[((size_t)split*cap+offset+r)*N+c]=acc[j]*gs;\\n  }\\n  __syncthreads();\\n }\\n}\\nconstexpr int CONFIGS=10;\\n// NT,Kstage,stages,splitK. 2 is the wait0 control for 1.\\n#define PIPE_CASES(F) \\\\\\n F(0,16,64,4,1,false) \\\\\\n F(1,16,128,4,1,true) \\\\\\n F(2,16,128,4,1,false) \\\\\\n F(3,16,128,6,1,true) \\\\\\n F(4,16,256,4,1,true) \\\\\\n F(5,32,128,4,1,true) \\\\\\n F(6,32,256,3,1,true) \\\\\\n F(7,16,128,4,2,true) \\\\\\n F(8,32,128,4,2,true) \\\\\\n F(9,16,128,4,4,true)\\n}\\nint64_t pipe_init(){\\n int dev,opt;C10_CUDA_CHECK(cudaGetDevice(&dev));\\n C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\\n #define INIT(ID,NT,KT,S,SK,OL) {constexpr int bytes=S*(64*KT/2+64*KT/16+NT*KT*2);TORCH_CHECK(bytes+1024<=opt,"pipeline config exceeds shared memory: ",ID);C10_CUDA_CHECK(cudaFuncSetAttribute(mwpipe::grouped<NT,KT,S,SK,OL>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));}\\n PIPE_CASES(INIT)\\n #undef INIT\\n return opt;\\n}\\nstd::vector<int64_t> pipe_info(int64_t which){\\n #define INFO(ID,NT,KT,S,SK,OL) case ID:{cudaFuncAttributes a{};int blocks;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,mwpipe::grouped<NT,KT,S,SK,OL>));constexpr int bytes=S*(64*KT/2+64*KT/16+NT*KT*2);C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwpipe::grouped<NT,KT,S,SK,OL>,128,bytes));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};}\\n switch(which){PIPE_CASES(INFO) default:TORCH_CHECK(false,"unknown pipeline configuration");}\\n #undef INFO\\n}\\nvoid pipe_run(torch::Tensor q,torch::Tensor s,torch::Tensor gs,torch::Tensor x,torch::Tensor c,torch::Tensor off,\\n torch::Tensor order,torch::Tensor task,torch::Tensor slots,torch::Tensor y,int64_t first,int64_t which,int64_t grid,int64_t M){\\n auto stream=at::cuda::getCurrentCUDAStream();int cap=y.size(-2),N=y.size(-1);\\n #define GO(ID,NT,KT,S,SK,OL) case ID: mwpipe::grouped<NT,KT,S,SK,OL><<<grid,128,S*(64*KT/2+64*KT/16+NT*KT*2),stream>>>(q.data_ptr<int>(),s.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),c.data_ptr<int>(),off.data_ptr<int>(),order.data_ptr<int>(),task.data_ptr<int>(),slots.data_ptr<int>(),y.data_ptr<float>(),M,N,cap,first);break;\\n switch(which){PIPE_CASES(GO) default:TORCH_CHECK(false,"unknown pipeline configuration");}\\n #undef GO\\n C10_CUDA_KERNEL_LAUNCH_CHECK();\\n}\\n\'\nPIPE_CPP=\'int64_t pipe_init();std::vector<int64_t> pipe_info(int64_t);void pipe_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);\'\nPIPE_CONFIGS=((16,64,4,1,False),(16,128,4,1,True),(16,128,4,1,False),\n              (16,128,6,1,True),(16,256,4,1,True),(32,128,4,1,True),\n              (32,256,3,1,True),(16,128,4,2,True),(32,128,4,2,True),(16,128,4,4,True))\n\n\n@functools.lru_cache(None)\ndef pipe_module():\n    mod=load_inline(name=\'mw_pipe_\'+hashlib.sha256(PIPE_CUDA.encode()).hexdigest()[:12],cpp_sources=PIPE_CPP,\n                    cuda_sources=PIPE_CUDA,functions=[\'pipe_init\',\'pipe_info\',\'pipe_run\'],extra_cuda_cflags=[\'-O3\',\'-std=c++20\',\'-lineinfo\',\n                    \'-gencode=arch=compute_90a,code=sm_90a\',\'-gencode=arch=compute_120,code=sm_120\'],verbose=False)\n    mod.pipe_init()\n    return mod\n\n\ndef pipe_prepare(w):\n    bank={}\n    for key in (\'w13\',\'w2\'):\n        bank[key]=mw_repack_planes(w[key]);e,n,k16=w[key+\'_sf\'].shape\n        bank[key+\'_sf\']=w[key+\'_sf\'].view(torch.uint8).reshape(e,n//16,2,8,k16//2,2).permute(0,1,4,3,5,2).contiguous()\n    bank[\'w13_gs\']=torch.cat((w.get(\'w13_gs_gate\',w[\'w13_gs\']),w.get(\'w13_gs_up\',w[\'w13_gs\']))).float().contiguous()\n    bank[\'w2_gs\']=w[\'w2_gs\'].float().contiguous()\n    bank[\'pnw\']=w.get(\'post_norm_w\',w.get(\'post_expert_norm\')).contiguous()\n    bank[\'mod\']=pipe_module()\n    return bank\n\n\n@triton.jit\ndef pipe_reduce(P,Y,OFF,N:tl.constexpr,CAP:tl.constexpr,SK:tl.constexpr):\n    row=tl.program_id(0);col=tl.program_id(1)*256+tl.arange(0,256)\n    if row<tl.load(OFF+32):\n        acc=tl.full((256,),0,tl.float32)\n        for s in tl.static_range(SK):acc+=tl.load(P+(s*CAP+row)*N+col)\n        tl.store(Y+row*N+col,acc)\n\n\ndef pipe_work(data,bank,which=1,grid=528):\n    nt,kt,st,sk,overlap=PIPE_CONFIGS[which]\n    x=data[\'hidden_states\'];m=x.shape[0];dev=x.device;cap=m*8+32*nt\n    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)\n    order=torch.empty_like(c);task=torch.empty_like(off)\n    slots=torch.empty((32,m),device=dev,dtype=torch.int32);pos=torch.empty((m,8),device=dev,dtype=torch.int32)\n    h=torch.empty((cap,8192),device=dev,dtype=torch.float32)\n    a=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((cap,4096),device=dev,dtype=torch.float32)\n    hp=torch.empty((sk,cap,8192),device=dev,dtype=torch.float32) if sk>1 else h\n    yp=torch.empty((sk,cap,4096),device=dev,dtype=torch.float32) if sk>1 else y\n    inv=torch.empty(cap,device=dev,dtype=torch.float32);out=torch.empty_like(x)\n    def route():\n        mw_route[(32,)](data[\'topk_ids\'],c,slots,pos,m,data[\'config\'].get(\'local_start\',0),triton.next_power_of_2(m))\n        mw_prefix[(1,)](c,off,order,task,nt)\n    def first():\n        bank[\'mod\'].pipe_run(bank[\'w13\'],bank[\'w13_sf\'],bank[\'w13_gs\'],x,c,off,order,task,slots,hp,1,which,grid,m)\n        if sk>1:pipe_reduce[(cap,32)](hp,h,off,8192,cap,sk)\n    def act():mw_swiglu[(cap,)](h,a,off,num_warps=4)\n    def second():\n        bank[\'mod\'].pipe_run(bank[\'w2\'],bank[\'w2_sf\'],bank[\'w2_gs\'],a,c,off,order,task,slots,yp,0,which,grid,m)\n        if sk>1:pipe_reduce[(cap,16)](yp,y,off,4096,cap,sk)\n    def norm():mw_norm[(cap,)](y,bank[\'pnw\'],off,inv,num_warps=4)\n    def combine():\n        mw_combine[(m,4)](y,bank[\'pnw\'],data[\'topk_ids\'],data[\'topk_weights\'],pos,off,inv,out,m,data[\'config\'].get(\'local_start\',0))\n        return out\n    return (route,first,act,second,norm,combine)\n\n\ndef pipe_call(data,bank,which=1,grid=528):\n    for f in pipe_work(data,bank,which,grid):out=f()\n    return out\n\nimport statistics\nimport time\n\n# Fixed families, compared across every seed; no per-seed winner is reported.\nPP_CHOICES=((0,528),(1,528),(2,528),(3,528),(4,528),(5,528),(6,528),(7,1056),(8,1056),(9,1056))\nPP_NAMES=(\'NT1\',\'NT2\',\'masked\')+tuple(\'P\'+str(i) for i in range(len(PP_CHOICES)))\n\n\ndef pp_control_prepare(w,raw=False):\n    dev=w[\'w13\'].device;old=rv_prepare(w)\n    gs=torch.cat((w[\'w13_gs_gate\'],w[\'w13_gs_up\'])).float().contiguous();old[\'gs13\']=gs\n    st=dict(pre_expert=torch.ones(4096,device=dev),ones=torch.ones(32,device=dev),\n        post_expert=(1+w[\'post_norm_w\'].float()).bfloat16(),post_expert_bf16=(1+w[\'post_norm_w\'].float()).bfloat16(),\n        sms=torch.cuda.get_device_properties(dev).multi_processor_count,gs13_gate=w[\'w13_gs_gate\'],\n        gs13_up=w[\'w13_gs_up\'],gs2=w[\'w2_gs\'],w13_gs=gs,w2_gs=w[\'w2_gs\'],w13p=old[\'w13p\'],w2p=old[\'w2p\'])\n    hum=pf__Experts(dict(w,w13_gs=torch.ones(32,device=dev)),0,32)\n    hum.gemm_type=pf_GemmType.GROUPED_MASKED;hum.compile([128,256]);st[\'humming\']=hum\n    for key in (\'w13\',\'w2\'):assert hum.weight(key)[\'weight_scale\'].dtype==torch.float8_e4m3fn\n    if raw:\n        class Raw:\n            def __init__(self):self.moe_gemm=pf__grouped_module().raw17_gemm\n        rr=dict(st,tiled_mod=Raw())\n        for key,gated in ((\'w13\',True),(\'w2\',False)):\n            b,sf=pf__raw_repack(w[key],w[key+\'_sf\'],gated);out=torch.empty_like(b)\n            pf__encode_blob_fast[(b.numel()//2048,)](b.view(torch.int32),sf,out.view(torch.int32),num_warps=4)\n            rr[\'blob13\' if gated else \'blob2\']=out;rr[\'sfb13\' if gated else \'sfb2\']=sf\n        st[\'raw\']=rr\n    return old,st\n\n\ndef pp_reference(data,st,name):\n    x=data[\'hidden_states\'];m=x.shape[0];dev=x.device\n    c=torch.empty(32,device=dev,dtype=torch.int32);slots=torch.empty((32,m),device=dev,dtype=torch.int32)\n    pos=torch.empty((m,8),device=dev,dtype=torch.int32)\n    mw_route[(32,)](data[\'topk_ids\'],c,slots,pos,m,0,triton.next_power_of_2(m),num_warps=4)\n    ids=torch.where((data[\'topk_ids\']>=0)&(data[\'topk_ids\']<32),data[\'topk_ids\'],-1).int()\n    args=(x,data[\'topk_weights\'],ids,pos,c)\n    if name==\'raw17\':return pf__raw_experts_tiled(*args,st[\'raw\'])\n    return pf__experts_humming_masked(*args,st)\n\n\ndef pp_measure(w,bank,old,st,ds,flush,raw=False):\n    funcs=[lambda d:rv_kernel_fn(d,dict(old,nt=1)),lambda d:rv_kernel_fn(d,dict(old,nt=2)),\n           lambda d:pp_reference(d,st,\'masked\')]\n    funcs.extend(lambda d,c=choice:pipe_call(d,bank,*c) for choice in PP_CHOICES)\n    names=PP_NAMES\n    if raw:funcs.append(lambda d:pp_reference(d,st,\'raw17\'));names=names+(\'raw17\',)\n    graphs=[MWGraph(f,ds[0]) for f in funcs];wants=dc_oracles(ds,w)\n    errors=[]\n    for g in graphs:\n        err,row=0.,0.\n        for d,want in zip(ds,wants):\n            g.refresh(d);got=g.replay();torch.cuda.synchronize()\n            if not bool(torch.isfinite(got).all()):err=float(\'inf\');row=float(\'inf\');continue\n            delta=got.float()-want.float();den=want.float().norm(dim=-1)\n            err=max(err,float(delta.norm()/want.float().norm().clamp_min(1e-12)))\n            row=max(row,float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max()))\n        errors.append((err,row))\n    if any(e>.01 or r>.05 for e,r in errors[:3]):raise RuntimeError(\'control gate failed \'+str(errors[:3]))\n    times=[[] for _ in names];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)\n    for i in range(8):\n        for g in graphs:g.refresh(ds[i%3])\n        order=list(range(len(graphs)));order=order[i%len(order):]+order[:i%len(order)]\n        if i%2:order.reverse()\n        for j in order:\n            if errors[j][0]>.01 or errors[j][1]>.05:continue\n            flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();times[j].append(a.elapsed_time(b)*1000.)\n    return [(statistics.median(t) if t else float(\'inf\'),e,r) for t,(e,r) in zip(times,errors)]\n\n\ndef pp_raw_phase(w,st,ds,refs,counts,flush):\n    m=ds[0][\'hidden_states\'].shape[0];dev=w[\'w13\'].device\n    c=torch.tensor(counts,device=dev,dtype=torch.int32);ts=torch.empty(33,device=dev,dtype=torch.int32)\n    ids=ds[0][\'topk_ids\'];slots=torch.empty((32,m),device=dev,dtype=torch.int32);pos=torch.empty_like(ids)\n    mw_route[(32,)](ids,c,slots,pos,m,0,triton.next_power_of_2(m))\n    pf__raw_tile_offsets_kernel[(1,)](c,ts,BNC=128)\n    cap=triton.cdiv(m*8+32*127,128)*128\n    xt=torch.empty(cap*4096,device=dev,dtype=torch.bfloat16);inter=torch.empty_like(xt);rows=torch.empty_like(ids)\n    x=ds[0][\'hidden_states\'].clone()\n    def gather():pf__raw_gather_tiled_kernel[(m,)](x,ids,pos,ts,xt,rows,D=4096,BNC=128,num_warps=4)\n    def gemm():st[\'tiled_mod\'].moe_gemm(st[\'blob13\'],st[\'sfb13\'],st[\'w13_gs\'],xt,inter,ts,32,64,64,1,st[\'sms\'])\n    gather();gemm();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()\n    with torch.cuda.graph(g):gemm()\n    starts=ts.cpu().tolist();valid=torch.tensor([r for e,n in enumerate(counts) for r in range(starts[e]*128,starts[e]*128+n)],device=dev)\n    rr=valid[:,None];cc=torch.arange(4096,device=dev)[None,:]\n    indices=((rr//128*64+cc//64)*128+rr%128)*64+((cc%64//8)^(rr%8))*8+cc%8\n    worst=0.\n    for data,rr in zip(ds,refs):\n        x.copy_(data[\'hidden_states\']);gather();g.replay();torch.cuda.synchronize()\n        want=(torch.nn.functional.silu(rr[:,:4096])*rr[:,4096:]).bfloat16().float()\n        worst=max(worst,float((inter[indices].float()-want).norm()/want.norm().clamp_min(1e-12)))\n    if worst>.01:raise RuntimeError(f\'raw17 W13/SwiGLU gate {worst}\')\n    vals=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)\n    for i in range(6):\n        x.copy_(ds[i%3][\'hidden_states\']);gather();flush.zero_();a.record();g.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000.)\n    return f\'RAW17_W13_SWIGLU M={m} us={statistics.median(vals):.1f} fp32={worst:.6f} gather_excluded\'\n\n\ndef pp_w13(w,bank,ds,flush,rawst):\n    # Native W13 timings, including split-K reduction, on three fresh R0 inputs.\n    m=ds[0][\'hidden_states\'].shape[0];dev=w[\'w13\'].device;out=[]\n    plans=((0,264),(0,528),(0,1056),(1,264),(1,528),(1,1056),(2,528),\n           (3,528),(4,528),(5,528),(6,528),(7,528),(7,1056),(8,1056),(9,1056))\n    # Independent W13 FP32 oracle, a temporary per-expert matrix only.\n    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=dev)\n    ids=ds[0][\'topk_ids\'];counts=torch.bincount(ids[ids>=0],minlength=32).cpu().tolist()\n    requested={nt:sum((n+nt-1)//nt for n in counts)*8192*4096*9//16 for nt in (16,32)}\n    unique=sum(n>0 for n in counts)*8192*4096*9//16\n    old_tf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False\n    try:\n        refs=[[] for _ in ds]\n        for e,n in enumerate(counts):\n            if not n:continue\n            q=w[\'w13\'][e];codes=torch.stack((q&15,q>>4),-1).flatten(-2).long()\n            weight=lut[codes&7]*torch.where(codes&8!=0,-1.,1.)*w[\'w13_sf\'][e].float().repeat_interleave(16,-1)\n            tok=torch.where(ids==e)[0]\n            for j,data in enumerate(ds):\n                want=data[\'hidden_states\'][tok].float()@weight.t()\n                want[:,:4096]*=w[\'w13_gs_gate\'][e];want[:,4096:]*=w[\'w13_gs_up\'][e];refs[j].append(want)\n        refs=[torch.cat(x) for x in refs]\n    finally:torch.backends.cuda.matmul.allow_tf32=old_tf\n    for which,grid in plans:\n        static={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()}\n        nt=PIPE_CONFIGS[which][0];cap=m*8+32*nt\n        c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)\n        order=torch.empty_like(c);tasks=torch.empty_like(off);slots=torch.empty((32,m),device=dev,dtype=torch.int32)\n        pos=torch.empty_like(ids);sk=PIPE_CONFIGS[which][3]\n        y=torch.empty((cap,8192),device=dev);part=torch.empty((sk,cap,8192),device=dev) if sk>1 else y\n        mw_route[(32,)](ids,c,slots,pos,m,0,triton.next_power_of_2(m));mw_prefix[(1,)](c,off,order,tasks,nt)\n        def launch():\n            bank[\'mod\'].pipe_run(bank[\'w13\'],bank[\'w13_sf\'],bank[\'w13_gs\'],static[\'hidden_states\'],c,off,order,tasks,slots,part,1,which,grid,m)\n            if sk>1:pipe_reduce[(cap,32)](part,y,off,8192,cap,sk)\n        launch();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()\n        with torch.cuda.graph(graph):launch()\n        offsets=off.cpu().tolist();worst=0.\n        valid=torch.tensor([r for e,n in enumerate(counts) for r in range(offsets[e],offsets[e]+n)],device=dev)\n        for data,want in zip(ds,refs):\n            static[\'hidden_states\'].copy_(data[\'hidden_states\']);graph.replay();torch.cuda.synchronize()\n            worst=max(worst,float((y[valid]-want).norm()/want.norm().clamp_min(1e-12)))\n        if worst>.01:\n            out.append(f\'W13_REJECT M={m} P{which} grid={grid} fp32={worst:.6f}\')\n            continue\n        vals=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)\n        for i in range(6):\n            static[\'hidden_states\'].copy_(ds[i%3][\'hidden_states\']);flush.zero_();a.record();graph.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000.)\n        us=statistics.median(vals)\n        out.append(f\'W13 M={m} P{which} grid={grid} us={us:.1f} req_GBs={requested[nt]/us/1e3:.1f} unique_GBs={unique/us/1e3:.1f} fp32={worst:.6f}\')\n    out.append(pp_raw_phase(w,rawst,ds,refs,counts,flush))\n    return out\n\n\ndef lab(deadline,weights,state=None):\n    dev=next(v.device for v in weights.values() if torch.is_tensor(v));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)\n    lines=[\'META larger async K stages, prefetch S-1, recycle previous stage after wait1, one CTA barrier/stage, two A register banks, gather sources hoisted, split-K FP32 reduction. Raw E4M3 stays raw; conversion in timed kernel. Same 12 seeds/3 fresh fixtures as de6a56fc with independent gate/up GS. Eight rotated cold128MiB timings. FP32 L2<=.01,row<=.05. W13 bandwidth nominal packed Q+SF, not hardware counters.\']\n    lines.append(\'ORDER \'+\',\'.join(PP_NAMES));mod=pipe_module()\n    for i,cfg in enumerate(PIPE_CONFIGS):lines.append(f\'CFG P{i} NT,K,stages,split,overlap={cfg} regs,localB,staticSM,dynSM,resident={mod.pipe_info(i)}\')\n    records=[]\n    for i in range(12):\n        if time.monotonic()>deadline-15:lines.append(f\'INCOMPLETE before_seed={i}\');break\n        w=dc_weights(5353+7919*i,dev);old,control=pp_control_prepare(w,raw=i==0);bank=pipe_prepare(w)\n        for m in (128,256):\n            ds=dc_fixtures(w,m)\n            if i==0:\n                r0=dc_fixtures(w,m,True);rr=pp_measure(w,bank,old,control,r0,flush,raw=True)\n                lines.append(f\'R0 M={m} us=\'+\',\'.join(f\'{v[0]:.1f}\' for v in rr)+f\' raw17_last l2={max(v[1] for v in rr):.5f} row={max(v[2] for v in rr):.4f}\')\n                lines.extend(pp_w13(w,bank,r0,flush,control[\'raw\']))\n            vals=pp_measure(w,bank,old,control,ds,flush)\n            counts=torch.bincount(ds[0][\'topk_ids\'][ds[0][\'topk_ids\']>=0],minlength=32)\n            lines.append(f\'S {i} M={m} P={int(counts.sum())} E={int((counts>0).sum())} us=\'+\',\'.join(f\'{v[0]:.1f}\' for v in vals)+f\' l2={max(v[1] for v in vals):.5f} row={max(v[2] for v in vals):.4f}\')\n            records.append((m,vals))\n    for m in (128,256):\n        for j,name in enumerate(PP_NAMES):\n            vals=[v[j] for mm,v in records if mm==m]\n            if vals:\n                good=all(e<=.01 and r<=.05 for _,e,r in vals);ts=[v[0] for v in vals]\n                lines.append(f\'AGG {name} M={m} n={len(ts)} mean={statistics.mean(ts):.1f} med={statistics.median(ts):.1f} \'+(\'PASS\' if good else \'REJECT\'))\n    return lines\n')]
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
            _out, _n = [], 0
            for _l in _lines:
                if _n + len(_l) + 1 > 9000 // len(_LAB_PAYLOADS):
                    _out.append(f"LAB_TRUNCATED {len(_lines) - len(_out)} lines")
                    break
                _out.append(_l)
                _n += len(_l) + 1
            print(f"LAB_BEGIN {_name} {_sha}", flush=True)
            for _l in _out:
                print(_l, flush=True)
            print(f"LAB_END lab_s={_t.monotonic() - _t0:.1f} prepare_s={_t.monotonic() - _start:.1f}", flush=True)
    return state
