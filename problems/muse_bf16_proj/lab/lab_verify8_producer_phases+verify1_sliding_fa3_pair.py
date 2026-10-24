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


# ---- h200_lab wrapper: payloads verify8_producer_phases 6b8ece021449b4ca, verify1_sliding_fa3_pair a652ee6400e5c27b on base f3786341611f4f1f ----
_LAB_PAYLOADS = [('verify8_producer_phases', '6b8ece021449b4ca', 'import math, time\nimport torch\nfrom torch.utils.cpp_extension import load_inline\nROW_FACTOR=5.0\nROW_NORM_FLOOR=.1\ndef relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:\n    """|a - b| / |b|."""\n    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))\n\n\ndef query_precision_bound(e4m3_err: float, factor: float, floor: float, cap: float, slack: float = 0.0) -> float:\n    """Attention\'s bound for one case: `factor` times the reference\'s own error with queries\n    rounded to FP8 E4M3 (`e4m3_err`), plus `slack`, within [floor, cap]."""\n    return min(cap, max(floor, factor * e4m3_err + slack))\n\n\ndef compare(output, ref: torch.Tensor, tol: float) -> tuple[bool, str]:\n    if not isinstance(output, torch.Tensor):\n        return False, f"expected a tensor, got {type(output).__name__}"\n    if output.shape != ref.shape or output.dtype != ref.dtype:\n        return False, f"expected {tuple(ref.shape)} {ref.dtype}, got {tuple(output.shape)} {output.dtype}"\n    out = output.float()\n    if not torch.isfinite(out).all():\n        return False, "output contains NaN or Inf"\n    r = ref.float()\n    diff = out - r\n    err = float(diff.norm() / r.norm().clamp(min=1e-12))\n    if err > tol:\n        return False, f"relative L2 error {err:.3e} > {tol:g}"\n\n    rows_diff = diff.reshape(-1, r.shape[-1]).norm(dim=-1)\n    rows_ref = r.reshape(-1, r.shape[-1]).norm(dim=-1)\n    floor = (ROW_NORM_FLOOR * rows_ref.mean()).clamp(min=1e-12)\n    row_err = rows_diff / torch.maximum(rows_ref, floor)\n    worst = int(row_err.argmax())\n    worst_err = float(row_err[worst])\n    if worst_err > ROW_FACTOR * tol:\n        index = tuple(int(i) for i in torch.unravel_index(torch.tensor(worst), r.shape[:-1])) if r.dim() > 1 else ()\n        return False, (f"relative L2 error {err:.3e}, but row {index} is off by {worst_err:.3e}"\n                       f" > {ROW_FACTOR:g} x {tol:g}")\n    return True, f"relative L2 error {err:.3e} (worst row {worst_err:.3e})"\n\nHEAD_DIM = 64\n\nHEADS = {"tp8": (16, 2), "dp": (128, 16)}  # (query heads, key/value heads) on one GPU\n\nSLIDING_WINDOW = 2048\n\nROPE_THETA = 500_000.0\n\nQK_SCALE_FACTOR = 5.4730064863875\n\nSM_SCALE = HEAD_DIM ** -0.5\n\nRMS_EPS = 1e-5\n\nPAGE = 64\n\nFP8 = torch.float8_e4m3fn\n\nK_SCALE = V_SCALE = 1.0\n\nREL_L2_TOL = 0.15\n\nFP8_QUERY_FACTOR = 2.5\n\nFP8_QUERY_FLOOR = 0.03\n\nFP8_QUERY_SLACK = 0.015\n\nQUERY_DTYPE = torch.bfloat16\n\nCACHE_TOL = 8e-2\n\nSOFT_C2 = {"sliding": (0.3, 0.75), "full": (0.5, 0.62)}\n\nCOHERENT_HEAD = 2048\n\nCOHERENT_ALPHA = 1.0\n\ndef _coherent_tail(v_cache, k_cache, q, seq_lens, page_table, layer, seed, page, scale):\n    """Add the shared direction to the values of each request\'s low-attention keys."""\n    hkv, hq = v_cache.shape[-2], q.shape[-2]\n    group = hq // hkv\n    tail_gen = torch.Generator(device=v_cache.device).manual_seed(seed + 0xC0DE)\n    u = torch.randn((hkv, HEAD_DIM), generator=tail_gen, device=v_cache.device)\n    u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))\n    qn = _rms(q) * QK_SCALE_FACTOR  # [B, L, Hq, 64], as the reference attends (RoPE on sliding)\n    if layer == "sliding":\n        pos = seq_lens.long()[:, None] + torch.arange(q.shape[1], device=q.device)\n        qn = _rope(qn, pos, ROPE_THETA)\n    k_flat = k_cache.view(-1, hkv, HEAD_DIM)\n    v_flat = v_cache.view(-1, hkv, HEAD_DIM)\n    for b in range(len(seq_lens)):\n        seq = int(seq_lens[b])\n        first = max(0, seq - SLIDING_WINDOW + 1) if layer == "sliding" else 0\n        past = torch.arange(first, seq, device=q.device)\n        rows = page_table[b].long()[past // page] * page + past % page\n        keys = _dequant(k_flat[rows], scale)  # [S, Hkv, 64]\n        # Each key\'s best logit over its KV head\'s query heads and the block\'s tokens.\n        logits = torch.einsum("lkgd,skd->kgls", qn[b].view(-1, hkv, group, HEAD_DIM), keys)\n        best = logits.flatten(1, 2).amax(1)  # [Hkv, S]\n        if best.shape[1] <= COHERENT_HEAD:\n            continue\n        tail = best < best.topk(COHERENT_HEAD, -1).values[:, -1:]  # [Hkv, S]\n        vals = _dequant(v_flat[rows], scale) + COHERENT_ALPHA * tail.T[..., None] * u\n        v_flat[rows] = _quant(vals, scale)\n\nAGENTIC_SPREAD = 0.075\n\ndef _longest(ctx: int, spread: str) -> int:\n    """The longest length _lengths can draw."""\n    if spread == "fixed":\n        return ctx\n    if spread == "agentic":\n        return math.ceil(ctx * (1 + AGENTIC_SPREAD))\n    return 4 * ctx\n\ndef _lengths(gen, batch: int, ctx: int, spread: str, device) -> torch.Tensor:\n    if spread == "fixed":\n        return torch.full((batch,), ctx, dtype=torch.int64, device=device)\n    if spread == "agentic":\n        u = torch.rand((batch,), generator=gen, device=device)\n        return (ctx * (1 + AGENTIC_SPREAD * (2 * u - 1))).long().clamp(1, _longest(ctx, spread))\n    # Log-normal around ctx: many short requests, a long tail.\n    z = torch.randn((batch,), generator=gen, device=device)\n    return (ctx * torch.exp(z - 0.5)).clamp(16, 4 * ctx).long()\n\ndef _shortest_share(spread: str) -> float:\n    """A lower bound on mean length / ctx, for the speed-of-light bound."""\n    return {"fixed": 1.0, "agentic": 1 - AGENTIC_SPREAD}.get(spread, 0.5)\n\ndef _quant(x, scale):\n    return (x.float() / scale).to(FP8)\n\ndef _dequant(x, scale):\n    return x.float() * scale\n\nCACHE_CHUNK = 1 << 20  # cache rows generated at a time\n\ndef _fill_cache(gen, shape, device, scale=K_SCALE, seed=0, dist="normal", layer="full"):\n    """FP8 key and value caches of `shape`: N(0, 1) entries, soft keys for dist="soft" and\n    "coherent" (see SOFT_C2)."""\n    hkv = shape[-2]\n    k_cache = torch.empty(shape, dtype=FP8, device=device)\n    v_cache = torch.empty(shape, dtype=FP8, device=device)\n    mix = None\n    if dist in ("soft", "coherent"):\n        # Its own generator, so the other inputs match the same seed\'s normal case.\n        soft = torch.Generator(device=device).manual_seed(seed + 0x50F7)\n        u = torch.randn((hkv, HEAD_DIM), generator=soft, device=device)\n        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))\n        lo, hi = SOFT_C2[layer]\n        c2 = lo + (hi - lo) * torch.rand((hkv, 1), generator=soft, device=device)\n        mix = (c2.sqrt() * u, (1 - c2).sqrt())\n    for dst, keys in ((k_cache, True), (v_cache, False)):\n        flat = dst.view(-1, hkv, HEAD_DIM)\n        for at in range(0, flat.shape[0], CACHE_CHUNK):\n            n = min(CACHE_CHUNK, flat.shape[0] - at)\n            x = torch.randn((n, hkv, HEAD_DIM), generator=gen, device=device)\n            if keys and mix is not None:\n                x = mix[0] + mix[1] * x\n            flat[at:at + n] = _quant(x, scale)\n    return k_cache, v_cache\n\ndef generate_input(batch: int, ctx: int, heads: str, layer: str, seed: int, q_len: int = 1,\n                   spread: str = "fixed", dist: str = "normal", page: int = PAGE,\n                   kv_scale: float = K_SCALE, device: str = "cuda", **_):\n    """layer: "sliding" (2,048-token window, RoPE) or "full" (every cached token, no RoPE).\n    q_len: 1 for a decode step, 8 for a speculative-decoding verify block (causal inside it).\n    dist: "normal", "soft" (see SOFT_C2), "coherent" (see COHERENT_HEAD), or "edges": keys\n    aligned with each KV head\'s queries at the window edges, relative to the block\'s first\n    token (sliding: the oldest position inside the window and the newest just outside it;\n    full: position 0)."""\n    gen = torch.Generator(device=device).manual_seed(seed)\n    hq, hkv = HEADS[heads]\n    seq_lens = _lengths(gen, batch, ctx, spread, device)\n    pages_per = (seq_lens + q_len + page - 1) // page  # room for the new tokens\n    # The pool and page table are sized for the longest possible length, so every\n    # repetition\'s inputs have the same shapes (one CUDA graph is replayed on fresh inputs).\n    longest = _longest(ctx, spread)\n    max_pages = (longest + q_len + page - 1) // page\n    total = batch * max_pages\n    # Physical pages are scattered.\n    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)\n    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)\n    offsets = torch.cumsum(pages_per, 0) - pages_per\n    for b in range(batch):\n        n, o = int(pages_per[b]), int(offsets[b])\n        page_table[b, :n] = perm[o:o + n]\n\n    # Cached keys are post-norm (unit RMS per head); values are projections, scale ~1.\n    shape = (total, page, hkv, HEAD_DIM)\n    k_cache, v_cache = _fill_cache(gen, shape, device, kv_scale, seed, dist, layer)\n    q = (2 * torch.randn((batch, q_len, hq, HEAD_DIM), generator=gen, device=device)).to(torch.bfloat16)\n    if dist == "coherent":\n        _coherent_tail(v_cache, k_cache, q, seq_lens, page_table, layer, seed, page, kv_scale)\n    if dist == "edges":\n        # Each KV head\'s key points along its group\'s first query head of the block\'s first\n        # token, as the kernel will see it (normalized, then RoPE on sliding layers).\n        qn = _rms(q[:, 0])\n        if layer == "sliding":\n            qn = _rope(qn, seq_lens, ROPE_THETA)\n        planted = _quant(qn[:, :: hq // hkv], kv_scale)  # [B, Hkv, 64]\n        k_flat = k_cache.view(-1, hkv, HEAD_DIM)\n        for b in range(batch):\n            p = int(seq_lens[b])\n            spots = [p - SLIDING_WINDOW + 1, p - SLIDING_WINDOW] if layer == "sliding" else [0]\n            for t in spots:\n                if 0 <= t < p:\n                    k_flat[int(page_table[b, t // page]) * page + t % page] = planted[b]\n    k_new = (2 * torch.randn((batch, q_len, hkv, HEAD_DIM), generator=gen, device=device)).to(torch.bfloat16)\n    v_new = torch.randn((batch, q_len, hkv, HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)\n    gate = torch.randn((batch, q_len, hq * HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)\n    # Fused projection output, per token: q | k | v | gate.\n    qkv = torch.cat([q.flatten(2), k_new.flatten(2), v_new.flatten(2), gate], -1)\n    # Step metadata: each new token\'s position and flat cache row, per-request cache lengths\n    # (including the new tokens) and query offsets, and on full-attention decode steps FA3\'s\n    # tile schedule.\n    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)\n    pos = positions.view(batch, q_len)\n    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)\n    cache_seqlens = (seq_lens + q_len).to(torch.int32)\n    cu_seqlens_q = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=device)\n    return {\n        "qkv": qkv,\n        "positions": positions,\n        "cache_loc": cache_loc,\n        "cache_seqlens": cache_seqlens,\n        "cu_seqlens_q": cu_seqlens_q,\n        "scheduler_metadata": _fa3_scheduler_metadata(cache_seqlens, cu_seqlens_q, hq, hkv, page, q_len, layer),\n        "k_cache": k_cache,\n        "v_cache": v_cache,\n        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),\n        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),\n        "page_table": page_table,\n        "seq_lens": seq_lens.to(torch.int32),\n        "config": {\n            "batch": batch,\n            "q_len": q_len,\n            "num_q_heads": hq,\n            "num_kv_heads": hkv,\n            "head_dim": HEAD_DIM,\n            "page_size": page,\n            "window": SLIDING_WINDOW if layer == "sliding" else 0,\n            "rope_theta": ROPE_THETA if layer == "sliding" else 0.0,\n            "qk_scale_factor": QK_SCALE_FACTOR,\n            "sm_scale": SM_SCALE,\n            "kv_scale": kv_scale,\n        },\n    }\n\ndef split_qkv(data):\n    """q [B, L, Hq, 64], k and v [B, L, Hkv, 64], gate [B, L, Hq*64] from the fused input."""\n    cfg = data["config"]\n    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]\n    batch, length = cfg["batch"], cfg["q_len"]\n    q, k, v, gate = data["qkv"].split([hq * d, hkv * d, hkv * d, hq * d], -1)\n    return (q.reshape(batch, length, hq, d), k.reshape(batch, length, hkv, d), v.reshape(batch, length, hkv, d),\n            gate)\n\ndef _rms(x, eps=RMS_EPS):\n    x = x.float()\n    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)\n\ndef _rope(x, pos, theta):\n    """Interleaved (GPT-J) RoPE: pairs (2i, 2i+1) rotate by pos * theta^(-2i/64).\n    x [..., heads, 64] with pos broadcast over everything but the last two dims."""\n    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=x.device).float() / HEAD_DIM))\n    ang = pos.float()[..., None] * inv_freq\n    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)\n    even, odd = x[..., 0::2], x[..., 1::2]\n    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)\n\ndef _rows(data, positions):\n    """Flat cache rows of [B, L] positions."""\n    size = data["config"]["page_size"]\n    page = torch.gather(data["page_table"].long(), 1, positions // size)\n    return page * size + positions % size\n\ndef new_slots(data) -> torch.Tensor:\n    """[B, L] flat cache rows where the block\'s new tokens go."""\n    pos = data["seq_lens"].long()[:, None] + torch.arange(data["config"]["q_len"], device=data["seq_lens"].device)\n    return _rows(data, pos)\n\ndef ref_kernel(data, query_dtype=None, new_kv=None):\n    """Returns (output [B, L, Hq*64], new keys and values [B, L, Hkv, 64] as stored: FP8).\n    query_dtype: the precision queries are rounded to (default QUERY_DTYPE). With new_kv\n    (FP8 keys and values [B, L, Hkv, 64]), the block attends those as its own entries."""\n    cfg = data["config"]\n    batch, length = cfg["batch"], cfg["q_len"]\n    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]\n    q_in, k_in, v_in, gate = split_qkv(data)\n    pos = data["seq_lens"].long()[:, None] + torch.arange(length, device=q_in.device)\n    q = _rms(q_in) * cfg["qk_scale_factor"]\n    k = _rms(k_in)\n    if cfg["rope_theta"]:\n        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])\n    # FP8 rounds the normalized query, before qk_scale_factor.\n    query_dtype = query_dtype or QUERY_DTYPE\n    if query_dtype == torch.bfloat16:\n        q = q.to(torch.bfloat16).float()\n    else:\n        q = (q / cfg["qk_scale_factor"]).to(query_dtype).float() * cfg["qk_scale_factor"]\n    k_new, v_new = _quant(k, data["k_scale"]), _quant(v_in, data["v_scale"])\n    if new_kv is not None:\n        k_new, v_new = new_kv\n    k_flat = data["k_cache"].view(-1, hkv, HEAD_DIM)\n    v_flat = data["v_cache"].view(-1, hkv, HEAD_DIM)\n    group = hq // hkv\n\n    out = torch.empty((batch, length, hq, HEAD_DIM), device=q.device)\n    for b in range(batch):\n        seq = int(data["seq_lens"][b])\n        past = torch.arange(seq, device=q.device)\n        size = cfg["page_size"]\n        past_rows = data["page_table"][b].long()[past // size] * size + past % size\n        keys_past = _dequant(k_flat[past_rows], data["k_scale"])\n        vals_past = _dequant(v_flat[past_rows], data["v_scale"])\n        keys = torch.cat([keys_past, _dequant(k_new[b], data["k_scale"])])  # positions 0 .. seq+L-1\n        vals = torch.cat([vals_past, _dequant(v_new[b], data["v_scale"])])\n        for j in range(length):\n            p = seq + j\n            start = max(0, p - cfg["window"] + 1) if cfg["window"] else 0\n            kb, vb = keys[start:p + 1], vals[start:p + 1]\n            att = torch.einsum("kgd,tkd->kgt", q[b, j].view(hkv, group, HEAD_DIM), kb) * cfg["sm_scale"]\n            out[b, j] = torch.einsum("kgt,tkd->kgd", att.softmax(-1), vb).reshape(hq, HEAD_DIM)\n    out = _rms(out) * torch.sigmoid(gate.float()).view(batch, length, hq, HEAD_DIM)\n    return out.reshape(batch, length, hq * HEAD_DIM), k_new, v_new\n\ndef collect(data, output) -> torch.Tensor:\n    """Runs in the submission\'s process after a step: the output followed by the new cache\n    entries, as raw bytes (see unpack)."""\n    cfg = data["config"]\n    shape = (cfg["batch"], cfg["q_len"], cfg["num_q_heads"] * HEAD_DIM)\n    if type(output) is not torch.Tensor or output.dtype != torch.bfloat16 or tuple(output.shape) != shape:\n        raise ValueError(f"expected a bfloat16 tensor of shape {shape}")\n    slots = new_slots(data)\n    hkv = cfg["num_kv_heads"]\n    k = data["k_cache"].view(-1, hkv, HEAD_DIM)[slots]\n    v = data["v_cache"].view(-1, hkv, HEAD_DIM)[slots]\n    return torch.cat([p.contiguous().view(torch.uint8).reshape(-1) for p in (output, k, v)])\n\ndef unpack(raw: torch.Tensor, cfg: dict):\n    b, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]\n    shapes = [((b, length, hq * HEAD_DIM), torch.bfloat16), ((b, length, hkv, HEAD_DIM), FP8),\n              ((b, length, hkv, HEAD_DIM), FP8)]\n    sizes = [math.prod(shape) * dtype.itemsize for shape, dtype in shapes]\n    if raw.dtype != torch.uint8 or raw.numel() != sum(sizes):\n        return None\n    out, at = [], 0\n    for (shape, dtype), n in zip(shapes, sizes):\n        out.append(raw[at:at + n].view(dtype).view(shape))\n        at += n\n    return out\n\ndef check_implementation(data, raw):\n    parts = unpack(raw, data["config"])\n    if parts is None:\n        return False, "malformed output"\n    out, k, v = parts\n    _, ref_k, ref_v = ref_kernel(data)\n    for name, got, want, scale in (("key", k, ref_k, data["k_scale"]), ("value", v, ref_v, data["v_scale"])):\n        ok, kv_msg = compare(_dequant(got, scale), _dequant(want, scale), CACHE_TOL)\n        if not ok:\n            return False, f"new {name} cache entries: {kv_msg}"\n    # The output is checked against the reference attending the submission\'s own new cache\n    # entries (checked above), since FP8 rounding of those entries varies by implementation.\n    ref = ref_kernel(data, new_kv=(k, v))[0].to(torch.bfloat16)\n    e4m3 = relative_l2(ref_kernel(data, torch.float8_e4m3fn, new_kv=(k, v))[0].to(torch.bfloat16), ref)\n    tol = query_precision_bound(e4m3, FP8_QUERY_FACTOR, FP8_QUERY_FLOOR, REL_L2_TOL, FP8_QUERY_SLACK)\n    ok, msg = compare(out, ref, tol)\n    msg = (f"{msg} [bound {tol:.3f}: {FP8_QUERY_FACTOR:g} x the reference\'s {e4m3:.3f} with FP8 E4M3 queries"\n           f" + {FP8_QUERY_SLACK:g}]")\n    return ok, msg\n\n\ndef _fa3_scheduler_metadata(*args,**kwargs): return None\n\n# The gate above is frozen from the task; this caches its three FP32 passes\n# per input only inside the untimed lab. Timed kernels never read these values.\ndef gate_prepare(data):\n    c=data[\'config\'];slots=new_slots(data);hk=c[\'num_kv_heads\']\n    actual=tuple(data[name].view(-1,hk,64)[slots].clone() for name in (\'k_cache\',\'v_cache\'))\n    _,want_k,want_v=ref_kernel(data)\n    want=ref_kernel(data,new_kv=actual)[0].to(torch.bfloat16)\n    fp8=ref_kernel(data,torch.float8_e4m3fn,new_kv=actual)[0].to(torch.bfloat16)\n    tol=query_precision_bound(relative_l2(fp8,want),FP8_QUERY_FACTOR,FP8_QUERY_FLOOR,REL_L2_TOL,FP8_QUERY_SLACK)\n    return dict(out=want,tol=tol,kv=(want_k,want_v),slots=slots)\n\n\ndef gate_cached(data,out,expected):\n    c=data[\'config\'];hk=c[\'num_kv_heads\']\n    for i,name in enumerate((\'k_cache\',\'v_cache\')):\n        actual=data[name].view(-1,hk,64)[expected[\'slots\']]\n        scale=data[\'k_scale\' if i==0 else \'v_scale\']\n        ok,msg=compare(_dequant(actual,scale),_dequant(expected[\'kv\'][i],scale),CACHE_TOL)\n        if not ok:return False,\'cache \'+msg,float(\'inf\')\n    out=out.view_as(expected[\'out\'])\n    ok,msg=compare(out,expected[\'out\'],expected[\'tol\'])\n    return ok,msg,relative_l2(out,expected[\'out\'])\n\n\ndef timed(fn,data,flush,rounds=5):\n    fn(data);torch.cuda.synchronize()\n    graph=torch.cuda.CUDAGraph()\n    with torch.cuda.graph(graph):\n        out=fn(data)\n    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)\n    values=[]\n    for _ in range(rounds):\n        flush.zero_();start.record();graph.replay();end.record();end.synchronize()\n        values.append(start.elapsed_time(end)*1000.)\n    return sorted(values)[len(values)//2],out,graph\n\nimport triton\nimport triton.language as tl\nfrom sgl_kernel.flash_attn import flash_attn_with_kvcache\nHQ,HKV,HD=16,2,64\nQ_SIZE=1024\nPDL=True\n@triton.jit\ndef _pdl_start(USE: tl.constexpr):\n    if USE:\n        tl.extra.cuda.gdc_launch_dependents()\n\n\n@triton.jit\ndef _pdl_wait(USE: tl.constexpr):\n    if USE:\n        tl.extra.cuda.gdc_wait()\n\n\ndef _launch(kernel, grid, *args, use_pdl=None, **kw):\n    enabled = PDL if use_pdl is None else bool(use_pdl)\n    if enabled:\n        kw["launch_pdl"] = True\n    return kernel[grid](*args, USE_PDL=enabled, **kw)\n\n\n@triton.jit\ndef _qk_prep_kernel(P, S: tl.constexpr, ps_stride, POS, LOC, KC, VC, KSCALE, VSCALE, INV_FREQ,\n                    QOUT, GATE, q_mult, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,\n                    ROPE: tl.constexpr, Q_FP8: tl.constexpr, USE_PDL: tl.constexpr):\n    t = tl.program_id(0)\n    _pdl_start(USE_PDL)\n    cache_row=tl.load(LOC+t)\n    kscale=tl.load(KSCALE);vscale=tl.load(VSCALE)\n    if ROPE:\n        half=tl.arange(0,D//2)\n        angle=tl.load(POS+t).to(tl.float32)*tl.load(INV_FREQ+half)\n        rope_cos=tl.cos(angle)[None,:];rope_sin=tl.sin(angle)[None,:]\n    _pdl_wait(USE_PDL)\n    W: tl.constexpr = (2 * HQ + 2 * HKV) * D\n    base = P + t * W\n    hq = tl.arange(0, HQ)\n    hk = tl.arange(0, HKV)\n    d = tl.arange(0, D)\n    qo = hq[:, None] * D + d[None, :]\n    ko = HQ * D + hk[:, None] * D + d[None, :]\n    vo = (HQ + HKV) * D + hk[:, None] * D + d[None, :]\n    go = (HQ + 2 * HKV) * D + qo\n    q = tl.zeros((HQ, D), tl.float32)\n    k = tl.zeros((HKV, D), tl.float32)\n    v = tl.zeros((HKV, D), tl.float32)\n    g = tl.zeros((HQ, D), tl.float32)\n    for s in tl.static_range(S):\n        q += tl.load(base + s * ps_stride + qo).to(tl.float32)\n        k += tl.load(base + s * ps_stride + ko).to(tl.float32)\n        v += tl.load(base + s * ps_stride + vo).to(tl.float32)\n        g += tl.load(base + s * ps_stride + go).to(tl.float32)\n    tl.store(GATE + t * (HQ * D) + qo, g)\n    q = q * tl.rsqrt(tl.sum(q * q, 1) / D + eps)[:, None]\n    k = k * tl.rsqrt(tl.sum(k * k, 1) / D + eps)[:, None]\n    if ROPE:\n        c=rope_cos;sn=rope_sin\n        qe, qd = tl.split(tl.reshape(q, (HQ, D // 2, 2)))\n        q = tl.reshape(tl.join(qe * c - qd * sn, qe * sn + qd * c), (HQ, D))\n        ke, kd = tl.split(tl.reshape(k, (HKV, D // 2, 2)))\n        k = tl.reshape(tl.join(ke * c - kd * sn, ke * sn + kd * c), (HKV, D))\n    row = cache_row\n    kv_off = row * (HKV * D) + hk[:, None] * D + d[None, :]\n    tl.store(KC + kv_off, (k / kscale).to(tl.float8e4nv))\n    tl.store(VC + kv_off, (v / vscale).to(tl.float8e4nv))\n    if Q_FP8:\n        tl.store(QOUT + t * (HQ * D) + qo, q.to(tl.float8e4nv))\n    else:\n        tl.store(QOUT + t * (HQ * D) + qo, (q * q_mult).to(tl.bfloat16))\n\n\ndef qk_prep(qkv, data, state, q_fp8: bool):\n    P, S = qkv\n    cfg = data["config"]\n    tokens = cfg["tokens"]\n    q = torch.empty((tokens, HQ, HD), device=P.device, dtype=FP8 if q_fp8 else torch.bfloat16)\n    gate = torch.empty((tokens, Q_SIZE), device=P.device, dtype=torch.float32)\n    _launch(_qk_prep_kernel, (tokens,),\n            P, S, P.stride(0), data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"],\n            data["k_scale"], data["v_scale"], state["inv_freq"], q, gate, cfg["qk_scale_factor"], RMS_EPS,\n            HQ=HQ, HKV=HKV, D=HD, ROPE=bool(cfg["rope_theta"]), Q_FP8=q_fp8, num_warps=4)\n    return q, gate\n\n\n@triton.jit\ndef _norm_gate_kernel(X, GATE, OUT, eps, HQ: tl.constexpr, D: tl.constexpr, USE_PDL: tl.constexpr):\n    t = tl.program_id(0)\n    _pdl_start(USE_PDL)\n    h = tl.arange(0, HQ)\n    d = tl.arange(0, D)\n    off = h[:, None] * D + d[None, :]\n    _pdl_wait(USE_PDL)\n    o = tl.load(X + t * (HQ * D) + off).to(tl.float32)\n    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]\n    g = tl.load(GATE + t * (HQ * D) + off)\n    tl.store(OUT + t * (HQ * D) + off, (o * tl.sigmoid(g)).to(tl.bfloat16))\n\n\ndef _attention_fa3_public(q, gate, data, state):\n    from sgl_kernel.flash_attn import flash_attn_with_kvcache\n    cfg = data[\'config\']\n    batch, L, tokens = (cfg[\'batch\'], cfg[\'q_len\'], cfg[\'tokens\'])\n    window = cfg[\'window\']\n    descale = {}\n    if cfg[\'kv_scale\'] != 1.0:\n        descale = dict(k_descale=data[\'k_scale\'].reshape(1, 1).expand(batch, HKV), v_descale=data[\'v_scale\'].reshape(1, 1).expand(batch, HKV))\n    out = flash_attn_with_kvcache(q=q, k_cache=data[\'k_cache\'], v_cache=data[\'v_cache\'], page_table=data[\'page_table\'], cache_seqlens=data[\'cache_seqlens\'], cu_seqlens_q=data[\'cu_seqlens_q\'], max_seqlen_q=L, softmax_scale=cfg[\'sm_scale\'] * cfg[\'qk_scale_factor\'], causal=True, window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data[\'scheduler_metadata\'], **descale)\n    if isinstance(out, tuple):\n        out = out[0]\n    o = torch.empty((tokens, Q_SIZE), device=q.device, dtype=torch.bfloat16)\n    _launch(_norm_gate_kernel, (tokens,), out, gate, o, RMS_EPS, HQ=HQ, D=HD, num_warps=4)\n    return o\n\n\n@triton.jit\ndef _best7_old_qk_prep_kernel(P, S: tl.constexpr, ps_stride, POS, LOC, KC, VC, KSCALE, VSCALE, INV_FREQ, QOUT, GATE, q_mult, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr, Q_FP8: tl.constexpr, USE_PDL: tl.constexpr):\n    t = tl.program_id(0)\n    _pdl_start(USE_PDL)\n    _pdl_wait(USE_PDL)\n    W: tl.constexpr = (2 * HQ + 2 * HKV) * D\n    base = P + t * W\n    hq = tl.arange(0, HQ)\n    hk = tl.arange(0, HKV)\n    d = tl.arange(0, D)\n    qo = hq[:, None] * D + d[None, :]\n    ko = HQ * D + hk[:, None] * D + d[None, :]\n    vo = (HQ + HKV) * D + hk[:, None] * D + d[None, :]\n    go = (HQ + 2 * HKV) * D + qo\n    q = tl.zeros((HQ, D), tl.float32)\n    k = tl.zeros((HKV, D), tl.float32)\n    v = tl.zeros((HKV, D), tl.float32)\n    g = tl.zeros((HQ, D), tl.float32)\n    for s in tl.static_range(S):\n        q += tl.load(base + s * ps_stride + qo).to(tl.float32)\n        k += tl.load(base + s * ps_stride + ko).to(tl.float32)\n        v += tl.load(base + s * ps_stride + vo).to(tl.float32)\n        g += tl.load(base + s * ps_stride + go).to(tl.float32)\n    tl.store(GATE + t * (HQ * D) + qo, g)\n    q = q * tl.rsqrt(tl.sum(q * q, 1) / D + eps)[:, None]\n    k = k * tl.rsqrt(tl.sum(k * k, 1) / D + eps)[:, None]\n    if ROPE:\n        pos = tl.load(POS + t).to(tl.float32)\n        half = tl.arange(0, D // 2)\n        ang = pos * tl.load(INV_FREQ + half)\n        c = tl.cos(ang)[None, :]\n        sn = tl.sin(ang)[None, :]\n        qe, qd = tl.split(tl.reshape(q, (HQ, D // 2, 2)))\n        q = tl.reshape(tl.join(qe * c - qd * sn, qe * sn + qd * c), (HQ, D))\n        ke, kd = tl.split(tl.reshape(k, (HKV, D // 2, 2)))\n        k = tl.reshape(tl.join(ke * c - kd * sn, ke * sn + kd * c), (HKV, D))\n    row = tl.load(LOC + t)\n    kv_off = row * (HKV * D) + hk[:, None] * D + d[None, :]\n    tl.store(KC + kv_off, (k / tl.load(KSCALE)).to(tl.float8e4nv))\n    tl.store(VC + kv_off, (v / tl.load(VSCALE)).to(tl.float8e4nv))\n    if Q_FP8:\n        tl.store(QOUT + t * (HQ * D) + qo, q.to(tl.float8e4nv))\n    else:\n        tl.store(QOUT + t * (HQ * D) + qo, (q * q_mult).to(tl.bfloat16))\n\n\ndef _best7_old_qk_prep(qkv, data, state, q_fp8: bool):\n    P, S = qkv\n    cfg = data[\'config\']\n    tokens = cfg[\'tokens\']\n    q = torch.empty((tokens, HQ, HD), device=P.device, dtype=FP8 if q_fp8 else torch.bfloat16)\n    gate = torch.empty((tokens, Q_SIZE), device=P.device, dtype=torch.float32)\n    _launch(_best7_old_qk_prep_kernel, (tokens,), P, S, P.stride(0), data[\'positions\'], data[\'cache_loc\'], data[\'k_cache\'], data[\'v_cache\'], data[\'k_scale\'], data[\'v_scale\'], state[\'inv_freq\'], q, gate, cfg[\'qk_scale_factor\'], RMS_EPS, HQ=HQ, HKV=HKV, D=HD, ROPE=bool(cfg[\'rope_theta\']), Q_FP8=q_fp8, num_warps=4)\n    return (q, gate)\n\n\nPHASE_CUDA=\'#include <cuda_runtime.h>\\n#include <cuda_fp16.h>\\n#include <cuda_bf16.h>\\n#include <cuda_fp8.h>\\n#include <cstdint>\\n#include <stdexcept>\\n#include <vector>\\n#define DI __device__ __forceinline__\\nnamespace {\\nconstexpr int D=64, ROWS=64, PR=68, THREADS=128;\\nstruct VP {\\n const __nv_bfloat16* qkv; const int64_t* loc; const int* pt; const int* seq;\\n uint8_t* kc; uint8_t* vc; const float* ks; const float* vs;\\n float* partial; __nv_bfloat16* out;\\n int B,HQ,HK,stride,pages,ps,lg,ns,proj_splits; float scale;\\n unsigned long long* clocks;\\n};\\nDI uint32_t su(const void* x){return (uint32_t)__cvta_generic_to_shared(x);}\\nDI void cp(void* dst,const void* src,bool valid){\\n asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(su(dst)),"l"(src),"r"(valid?16:0):"memory");\\n}\\nDI void commit(){asm volatile("cp.async.commit_group;":::"memory");}\\ntemplate<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}\\nDI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}\\nDI float2 exp2pair(float x,float y){\\n __half2 h=__floats2half2_rn(x,y);uint32_t bits=*(uint32_t*)&h,out;\\n asm("ex2.approx.f16x2 %0,%1;":"=r"(out):"r"(bits));\\n return __half22float2(*(__half2*)&out);\\n}\\nDI float wsum(float x){\\n#pragma unroll\\n for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);\\n return x;\\n}\\nDI uint16_t fp8pair(float x,float y){return __nv_cvt_float2_to_fp8x2(make_float2(x,y),__NV_SATFINITE,__NV_E4M3);}\\nDI int off(int r,int c){return r*64+(((c>>4)^(r&3))*16)+(c&15);}\\nDI uint32_t word(const uint8_t* p,int r,int c){return *(const uint32_t*)(p+off(r,c));}\\nDI float2 input_pair(const VP& p,int token,int col){\\n if(!p.proj_splits)return __bfloat1622float2(*(const __nv_bfloat162*)(p.qkv+(int64_t)token*p.stride+col));\\n const float* raw=(const float*)p.qkv;float2 sum=make_float2(0,0);\\n for(int s=0;s<p.proj_splits;s++){\\n  const float2 x=*(const float2*)(raw+((int64_t)s*p.B*8+token)*p.stride+col);\\n  sum.x+=x.x;sum.y+=x.y;\\n }\\n return __bfloat1622float2(__floats2bfloat162_rn(sum.x,sum.y));\\n}\\nDI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){\\n asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "\\n "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n}\\nstruct State {\\n uint32_t q[2][4];\\n float o[8][4];\\n float m[2],l[2];\\n};\\n// Convert an 8-key QK accumulator panel into two rows of FP8 pairs. No shared store.\\nDI uint32_t probpack(float (&s)[4]){\\n return (uint32_t)fp8pair(s[0]*256.f,s[1]*256.f)|((uint32_t)fp8pair(s[2]*256.f,s[3]*256.f)<<16);\\n}\\n// One CTA transposes V once per stage, amortized over the four query warps.\\n// Four lanes transpose each 4x4 byte panel using two PRMT steps.\\ntemplate<int N>\\nDI void transpose_v(const uint8_t* V,uint8_t* VT){\\n const int tid=threadIdx.x,t=tid&3;\\n#pragma unroll\\n for(int i=0;i<N/8;i++){\\n  const int group=tid/4+i*32;\\n  const int key=(group/16)*4,dim=(group%16)*4;\\n  const uint32_t x=word(V,key+t,dim);\\n  const uint32_t a=__shfl_sync(0xffffffff,x,(threadIdx.x&28));\\n  const uint32_t b=__shfl_sync(0xffffffff,x,(threadIdx.x&28)+1);\\n  const uint32_t c=__shfl_sync(0xffffffff,x,(threadIdx.x&28)+2);\\n  const uint32_t d=__shfl_sync(0xffffffff,x,(threadIdx.x&28)+3);\\n  const uint32_t ab=__byte_perm(a,b,t*0x11+0x40);\\n  const uint32_t cd=__byte_perm(c,d,t*0x11+0x40);\\n  const uint32_t out=__byte_perm(ab,cd,0x5410);\\n  *(uint32_t*)(VT+(dim+t)*N+((key^((((dim+t)&7)*16)&(N-1)))))=out;\\n }\\n}\\nstruct Trace { unsigned long long c[7]; };\\nDI unsigned long long tick(){unsigned long long x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}\\ntemplate<int N,bool MASK,int VN,bool CLOCK>\\nDI void tile(State& st,const uint8_t* K,const uint8_t* V,int kp,int hi,int seq,float scale,int vkey,Trace& trace){\\n unsigned long long ts=0;if constexpr(CLOCK)ts=tick();\\n constexpr int NT=N/8;\\n const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,warp=threadIdx.x>>5;\\n float s[NT][4];\\n#pragma unroll\\n for(int n=0;n<NT;n++){\\n#pragma unroll\\n  for(int i=0;i<4;i++)s[n][i]=0;\\n#pragma unroll\\n  for(int k=0;k<2;k++){\\n   const uint32_t b0=word(K,n*8+g,k*32+t*4);\\n   const uint32_t b1=word(K,n*8+g,k*32+16+t*4);\\n   mma8(s[n],st.q[k],b0,b1);\\n  }\\n }\\n if constexpr(CLOCK){const auto now=tick();trace.c[3]+=now-ts;ts=now;}\\n // Four lanes jointly own a query row. Both online reductions and P stay in registers.\\n#pragma unroll\\n for(int rh=0;rh<2;rh++){\\n  float mx=-INFINITY;\\n#pragma unroll\\n  for(int n=0;n<NT;n++){\\n#pragma unroll\\n   for(int e=0;e<2;e++){\\n    float x=s[n][2*rh+e]*scale;\\n    if constexpr(MASK){\\n     const int key=kp+8*n+2*t+e;\\n     if(key>=hi || key>seq+2*warp+rh)x=-INFINITY;\\n    }\\n    s[n][2*rh+e]=x;mx=fmaxf(mx,x);\\n   }\\n  }\\n  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));\\n  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));\\n  const float nm=fmaxf(st.m[rh],mx),use=nm==-INFINITY?0.f:nm;\\n  const float alpha=st.m[rh]==-INFINITY?0.f:ex(st.m[rh]-use);\\n  float sum=0;\\n#pragma unroll\\n  for(int n=0;n<NT;n++){\\n   const float2 prob=exp2pair(s[n][2*rh]-use,s[n][2*rh+1]-use);\\n   const float a=prob.x,b=prob.y;\\n   s[n][2*rh]=a;s[n][2*rh+1]=b;sum+=a+b;\\n  }\\n  sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);\\n  st.l[rh]=st.l[rh]*alpha+sum;st.m[rh]=nm;\\n#pragma unroll\\n  for(int d=0;d<8;d++){st.o[d][2*rh]*=alpha;st.o[d][2*rh+1]*=alpha;}\\n }\\n if constexpr(CLOCK){const auto now=tick();trace.c[4]+=now-ts;ts=now;}\\n uint32_t packed[NT];\\n#pragma unroll\\n for(int n=0;n<NT;n++)packed[n]=probpack(s[n]);\\n if constexpr(CLOCK){const auto now=tick();trace.c[5]+=now-ts;ts=now;}\\n#pragma unroll\\n for(int k=0;k<N/32;k++){\\n  uint32_t pa[4];\\n#pragma unroll\\n  for(int half=0;half<2;half++){\\n   const int src=g*4+2*(t&1);\\n   const uint32_t a0=__shfl_sync(0xffffffff,packed[4*k+2*half],src);\\n   const uint32_t a1=__shfl_sync(0xffffffff,packed[4*k+2*half],src+1);\\n   const uint32_t b0=__shfl_sync(0xffffffff,packed[4*k+2*half+1],src);\\n   const uint32_t b1=__shfl_sync(0xffffffff,packed[4*k+2*half+1],src+1);\\n   const uint32_t x=t<2?a0:b0,y=t<2?a1:b1;\\n   pa[2*half]=(x&0xffffu)|(y<<16);\\n   pa[2*half+1]=(x>>16)|(y&0xffff0000u);\\n  }\\n#pragma unroll\\n  for(int d=0;d<4;d++){\\n   const int col=16*d+g,kk=32*k+4*t+vkey;\\n   const int perm=((col&7)*16)&(VN-1);\\n   const uint32_t b00=*(const uint32_t*)(V+col*VN+(kk^perm));\\n   const uint32_t b01=*(const uint32_t*)(V+col*VN+((kk+16)^perm));\\n   const uint32_t b10=*(const uint32_t*)(V+(col+8)*VN+(kk^perm));\\n   const uint32_t b11=*(const uint32_t*)(V+(col+8)*VN+((kk+16)^perm));\\n   mma8(st.o[2*d],pa,b00,b01);\\n   mma8(st.o[2*d+1],pa,b10,b11);\\n  }\\n }\\n if constexpr(CLOCK)trace.c[6]+=tick()-ts;\\n}\\ntemplate<int ST,int KT,bool CLOCK>\\n__global__ __launch_bounds__(128,2) void verify8_main(VP p){\\n Trace trace{};unsigned long long start=0,ts=0,init=0;if constexpr(CLOCK)start=tick();\\n constexpr int KB=KT*64;\\n extern __shared__ __align__(128) uint8_t sm[];\\n uint8_t* K=sm;uint8_t* V=K+ST*KB;uint8_t* Q=V+ST*KB;\\n uint8_t* NK=Q+4096;uint8_t* NV=NK+2048;uint8_t* VT=NV+2048;\\n const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;\\n const int h=blockIdx.x,split=blockIdx.y,b=blockIdx.z,seq=p.seq[b];\\n const bool last=split==p.ns-1;\\n const int chunk=((seq+p.ns-1)/p.ns+KT-1)&~(KT-1);\\n const int begin=min(seq,split*chunk),end=min(seq,begin+chunk),tiles=(end-begin+KT-1)/KT;\\n const int* pt=p.pt+(int64_t)b*p.pages;\\n constexpr int NR=KT/32;\\n int rows[NR];\\n auto rowids=[&](int it){\\n#pragma unroll\\n  for(int rr=0;rr<NR;rr++){\\n   const int key=begin+it*KT+tid/4+32*rr;\\n   rows[rr]=(it<tiles && key<end)?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):-1;\\n  }\\n };\\n auto issue=[&](int slot){\\n#pragma unroll\\n  for(int rr=0;rr<NR;rr++){\\n   const int row=tid/4+32*rr,col=16*(tid&3),r=rows[rr];\\n   const int64_t src=(int64_t)max(r,0)*p.HK*64+h*64+col;\\n   cp(K+slot*KB+off(row,col),p.kc+src,r>=0);\\n   cp(V+slot*KB+off(row,col),p.vc+src,r>=0);\\n  }\\n };\\n#pragma unroll\\n for(int s=0;s<ST-1;s++){rowids(s);issue(s);commit();}\\n rowids(ST-1);\\n#pragma unroll\\n for(int i=0;i<16;i++){\\n  const int row=warp+4*i,j=row/8,head=h*8+row%8;\\n  const float2 q=input_pair(p,b*8+j,head*64+2*lane);\\n  const float r=rsqrtf(wsum(q.x*q.x+q.y*q.y)/64.f+1.e-5f);\\n  *(uint16_t*)(Q+row*64+2*lane)=fp8pair(q.x*r,q.y*r);\\n }\\n if(last){\\n  for(int i=tid;i<4096/16;i+=128)*(uint4*)(NK+i*16)=make_uint4(0,0,0,0);\\n  __syncthreads();\\n#pragma unroll\\n  for(int i=0;i<4;i++){\\n   const int rr=warp+4*i,j=rr/2,isv=rr&1;\\n   float2 x=input_pair(p,b*8+j,(p.HQ+isv*p.HK+h)*64+2*lane);\\n   if(!isv){const float r=rsqrtf(wsum(x.x*x.x+x.y*x.y)/64.f+1.e-5f);x.x*=r;x.y*=r;}\\n   const float sc=*(isv?p.vs:p.ks);\\n   const uint16_t f=fp8pair(x.x/sc,x.y/sc);\\n   *(uint16_t*)((isv?p.vc:p.kc)+p.loc[b*8+j]*p.HK*64+h*64+2*lane)=f;\\n   *(uint16_t*)((isv?NV:NK)+off(j,2*lane))=f;\\n  }\\n }\\n __syncthreads();State st;\\n#pragma unroll\\n for(int k=0;k<2;k++){\\n  const int qr=warp*16+g;\\n  st.q[k][0]=*(const uint32_t*)(Q+qr*64+32*k+4*t);\\n  st.q[k][1]=*(const uint32_t*)(Q+(qr+8)*64+32*k+4*t);\\n  st.q[k][2]=*(const uint32_t*)(Q+qr*64+32*k+16+4*t);\\n  st.q[k][3]=*(const uint32_t*)(Q+(qr+8)*64+32*k+16+4*t);\\n }\\n st.m[0]=st.m[1]=-INFINITY;st.l[0]=st.l[1]=0;\\n#pragma unroll\\n for(int d=0;d<8;d++)for(int e=0;e<4;e++)st.o[d][e]=0;\\n const float scale=p.scale*(*p.ks);\\n if constexpr(CLOCK)init=tick()-start;\\n for(int it=0;it<tiles;it++){\\n  if constexpr(CLOCK)ts=tick();\\n  wait<ST-2>();__syncthreads();\\n  if constexpr(CLOCK){const auto now=tick();trace.c[1]+=now-ts;ts=now;}\\n  issue((it+ST-1)%ST);commit();rowids(it+ST);\\n  if constexpr(CLOCK){const auto now=tick();trace.c[0]+=now-ts;ts=now;}\\n  transpose_v<KT>(V+(it%ST)*KB,VT);__syncthreads();\\n  if constexpr(CLOCK)trace.c[2]+=tick()-ts;\\n  // Register footprint stays at 64 scores while a handoff stages 128 keys.\\n  // Subtiles are independent warp work; no barrier or KV reload between them.\\n#pragma unroll 1\\n  for(int sub=0;sub<KT/64;sub++){\\n   const int key=begin+it*KT+sub*64;\\n   const uint8_t* kp=K+(it%ST)*KB+sub*4096;\\n   const uint8_t* vp=VT;\\n   if(key+64<=end)tile<64,false,KT,CLOCK>(st,kp,vp,key,end,seq,scale,sub*64,trace);\\n   else tile<64,true,KT,CLOCK>(st,kp,vp,key,end,seq,scale,sub*64,trace);\\n  }\\n }\\n wait<0>();__syncthreads();\\n if(last){transpose_v<32>(NV,VT);__syncthreads();tile<32,true,32,CLOCK>(st,NK,VT,seq,seq+8,seq,scale,0,trace);}\\n float* out=p.partial+(((int64_t)b*p.HK+h)*p.ns+split)*ROWS*PR;\\n#pragma unroll\\n for(int rh=0;rh<2;rh++){\\n  const int row=warp*16+g+8*rh;\\n#pragma unroll\\n  for(int d=0;d<8;d++)*(float2*)(out+row*PR+8*d+2*t)=make_float2(st.o[d][2*rh],st.o[d][2*rh+1]);\\n  if(t==0)*(float2*)(out+row*PR+64)=make_float2(st.m[rh],st.l[rh]);\\n }\\n if constexpr(CLOCK){\\n  const unsigned long long elapsed=tick()-start;\\n  if(lane==0){\\n   auto* dst=p.clocks+((((int64_t)b*p.HK+h)*p.ns+split)*4+warp)*10;\\n#pragma unroll\\n   for(int k=0;k<7;k++)dst[k]=trace.c[k];\\n   dst[7]=init;dst[8]=elapsed;dst[9]=tiles;\\n  }\\n }\\n\\n}\\n// Each warp owns one query/head row. Grid size is B*HK*16 independent CTAs.\\n__global__ __launch_bounds__(128,4) void verify8_combine(VP p){\\n const int warp=threadIdx.x>>5,lane=threadIdx.x&31;\\n const int row=blockIdx.x*4+warp,h=blockIdx.y,b=blockIdx.z;\\n const float* base=p.partial+((int64_t)b*p.HK+h)*p.ns*ROWS*PR+row*PR;\\n float mx=-INFINITY;\\n for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,base[(int64_t)s*ROWS*PR+64]);\\n#pragma unroll\\n for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));\\n float total=0;\\n for(int s=lane;s<p.ns;s+=32){const float2 ml=*(const float2*)(base+(int64_t)s*ROWS*PR+64);total+=ml.y*ex(ml.x-mx);}\\n total=wsum(total);\\n float x=0,y=0;\\n for(int first=0;first<p.ns;first+=8){\\n  float2 ov[8],ml[8];\\n#pragma unroll\\n  for(int k=0;k<8;k++)if(first+k<p.ns){\\n   const float* pp=base+(int64_t)(first+k)*ROWS*PR;\\n   ov[k]=*(const float2*)(pp+2*lane);ml[k]=*(const float2*)(pp+64);\\n  }\\n#pragma unroll\\n  for(int k=0;k<8;k++)if(first+k<p.ns){const float a=ex(ml[k].x-mx);x=fmaf(a,ov[k].x,x);y=fmaf(a,ov[k].y,y);}\\n }\\n const float sc=(*p.vs)/(total*256.f);x*=sc;y*=sc;\\n const float norm=rsqrtf(wsum(x*x+y*y)/64.f+1.e-5f);\\n const int j=row/8,head=h*8+row%8;\\n const float2 gate=input_pair(p,b*8+j,(p.HQ+2*p.HK+head)*64+2*lane);\\n *(__nv_bfloat162*)(p.out+((int64_t)(b*8+j)*p.HQ+head)*64+2*lane)=\\n __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate.x)),y*norm/(1.f+__expf(-gate.y)));\\n}\\ntemplate<int ST,int KT,bool CLOCK>void launch(VP p,cudaStream_t stream){\\n constexpr int smem=ST*KT*128+8192+KT*64;\\n cudaFuncSetAttribute(verify8_main<ST,KT,CLOCK>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\\n verify8_main<ST,KT,CLOCK><<<dim3(p.HK,p.ns,p.B),128,smem,stream>>>(p);\\n verify8_combine<<<dim3(16,p.HK,p.B),128,0,stream>>>(p);\\n}\\n}\\nvoid verify8_forward(int64_t qkv,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,\\n int64_t ks,int64_t vs,int64_t parts,int64_t out,int64_t B,int64_t HQ,int64_t HK,int64_t stride,\\n int64_t pages,int64_t ps,int64_t ns,int64_t proj_splits,double scale,int64_t stages,int64_t kt,int64_t profile,int64_t clocks,int64_t stream){\\n VP p{(const __nv_bfloat16*)qkv,(const int64_t*)loc,(const int*)pt,(const int*)seq,(uint8_t*)kc,(uint8_t*)vc,\\n (const float*)ks,(const float*)vs,(float*)parts,(__nv_bfloat16*)out,(int)B,(int)HQ,(int)HK,(int)stride,(int)pages,\\n (int)ps,0,(int)ns,(int)proj_splits,(float)scale,(unsigned long long*)clocks};\\n while((1<<p.lg)<ps)p.lg++;\\n const auto s=(cudaStream_t)stream;\\n if(profile){ if(kt==64){if(stages==3)launch<3,64,true>(p,s);else launch<4,64,true>(p,s);}\\n else {if(stages==3)launch<3,128,true>(p,s);else launch<4,128,true>(p,s);}}else{ if(kt==64){if(stages==3)launch<3,64,false>(p,s);else launch<4,64,false>(p,s);}\\n else {if(stages==3)launch<3,128,false>(p,s);else launch<4,128,false>(p,s);}}\\n const auto err=cudaGetLastError();if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));\\n}\\n\\nnamespace {\\ntemplate<int ST,int KT,bool C>std::vector<int64_t> resources(){\\n cudaFuncAttributes a;cudaFuncGetAttributes(&a,verify8_main<ST,KT,C>);\\n constexpr int smem=ST*KT*128+8192+KT*64;\\n cudaFuncSetAttribute(verify8_main<ST,KT,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\\n int active=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,verify8_main<ST,KT,C>,128,smem);\\n return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,active};\\n}\\n}\\nstd::vector<int64_t> verify8_resources(int64_t stages,int64_t kt,int64_t profile){\\n#define RES(S,K) if(stages==S && kt==K){return profile?resources<S,K,true>():resources<S,K,false>();}\\n RES(3,64) RES(4,64) RES(3,128) RES(4,128)\\n#undef RES\\n throw std::runtime_error("bad resource setting");\\n}\\n\'\nPHASE_CPP=\'#include <cstdint>\\n#include <vector>\\nvoid verify8_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t,int64_t,int64_t,int64_t);\\nstd::vector<int64_t> verify8_resources(int64_t,int64_t,int64_t);\\n\'\nWIDE_CUDA=\'#include <cuda_runtime.h>\\n#include <cuda_fp16.h>\\n#include <cuda_bf16.h>\\n#include <cuda_fp8.h>\\n#include <cstdint>\\n#include <stdexcept>\\n#include <vector>\\n#define DI __device__ __forceinline__\\nnamespace {\\nconstexpr int D=64, ROWS=64, PR=68, THREADS=128;\\nstruct VP {\\n const __nv_bfloat16* qkv; const int64_t* loc; const int* pt; const int* seq;\\n uint8_t* kc; uint8_t* vc; const float* ks; const float* vs;\\n float* partial; __nv_bfloat16* out;\\n int B,HQ,HK,stride,pages,ps,lg,ns,proj_splits; float scale;\\n unsigned long long* clocks;\\n};\\nDI uint32_t su(const void* x){return (uint32_t)__cvta_generic_to_shared(x);}\\nDI void cp(void* dst,const void* src,bool valid){\\n asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(su(dst)),"l"(src),"r"(valid?16:0):"memory");\\n}\\nDI void commit(){asm volatile("cp.async.commit_group;":::"memory");}\\ntemplate<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}\\nDI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}\\nDI float2 exp2pair(float x,float y){\\n __half2 h=__floats2half2_rn(x,y);uint32_t bits=*(uint32_t*)&h,out;\\n asm("ex2.approx.f16x2 %0,%1;":"=r"(out):"r"(bits));\\n return __half22float2(*(__half2*)&out);\\n}\\nDI float wsum(float x){\\n#pragma unroll\\n for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);\\n return x;\\n}\\nDI uint16_t fp8pair(float x,float y){return __nv_cvt_float2_to_fp8x2(make_float2(x,y),__NV_SATFINITE,__NV_E4M3);}\\nDI int off(int r,int c){return r*64+(((c>>4)^(r&3))*16)+(c&15);}\\nDI uint32_t word(const uint8_t* p,int r,int c){return *(const uint32_t*)(p+off(r,c));}\\nDI float2 input_pair(const VP& p,int token,int col){\\n if(!p.proj_splits)return __bfloat1622float2(*(const __nv_bfloat162*)(p.qkv+(int64_t)token*p.stride+col));\\n const float* raw=(const float*)p.qkv;float2 sum=make_float2(0,0);\\n for(int s=0;s<p.proj_splits;s++){\\n  const float2 x=*(const float2*)(raw+((int64_t)s*p.B*8+token)*p.stride+col);\\n  sum.x+=x.x;sum.y+=x.y;\\n }\\n return __bfloat1622float2(__floats2bfloat162_rn(sum.x,sum.y));\\n}\\nDI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){\\n asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "\\n "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n}\\nstruct State {\\n uint32_t q[4][4];\\n float o[8][4];\\n float m[2],l[2];\\n};\\n// Convert an 8-key QK accumulator panel into two rows of FP8 pairs. No shared store.\\nDI uint32_t probpack(float (&s)[4]){\\n return (uint32_t)fp8pair(s[0]*256.f,s[1]*256.f)|((uint32_t)fp8pair(s[2]*256.f,s[3]*256.f)<<16);\\n}\\nstruct Trace { unsigned long long c[7]; };\\nDI unsigned long long tick(){unsigned long long x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}\\nDI uint32_t widen8(uint16_t x){uint32_t y;asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(y):"h"(x));return y;}\\nDI int off16(int r,int c){return r*128+(((c>>3)^(r&7))*16)+((c&7)*2);}\\nDI uint32_t word16(const uint8_t* p,int r,int c){return *(const uint32_t*)(p+off16(r,c));}\\ntemplate<int N>DI void widen_stage(const uint8_t* K,const uint8_t* V,uint8_t* KH,uint8_t* VH){\\n const int tid=threadIdx.x;\\n for(int i=tid;i<N*32;i+=128){\\n  const int r=i/32,c=(i%32)*2;\\n  const uint32_t k=widen8(*(const uint16_t*)(K+off(r,c)));\\n  const uint32_t v=widen8(*(const uint16_t*)(V+off(r,c)));\\n  *(uint32_t*)(KH+off16(r,c))=k;\\n  const int perm0=((c&7)*8)&(N-1),perm1=(((c+1)&7)*8)&(N-1);\\n  *(uint16_t*)(VH+2*(c*N+(r^perm0)))=(uint16_t)v;\\n  *(uint16_t*)(VH+2*((c+1)*N+(r^perm1)))=(uint16_t)(v>>16);\\n }\\n}\\nDI void mma16(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){\\n asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "\\n "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n}\\nDI uint32_t pack16(float x,float y){__half2 h=__floats2half2_rn(x,y);return *(uint32_t*)&h;}\\ntemplate<int N,bool MASK,int VN,bool CLOCK>\\nDI void tile(State& st,const uint8_t* K,const uint8_t* V,int kp,int hi,int seq,float scale,int vkey,Trace& trace){\\n unsigned long long ts=0;if constexpr(CLOCK)ts=tick();\\n constexpr int NT=N/8;\\n const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,warp=threadIdx.x>>5;\\n float s[NT][4];\\n#pragma unroll\\n for(int n=0;n<NT;n++){\\n#pragma unroll\\n  for(int i=0;i<4;i++)s[n][i]=0;\\n#pragma unroll\\n  for(int k=0;k<4;k++){\\n   const uint32_t b0=word16(K,n*8+g,k*16+t*2);\\n   const uint32_t b1=word16(K,n*8+g,k*16+8+t*2);\\n   mma16(s[n],st.q[k],b0,b1);\\n  }\\n }\\n if constexpr(CLOCK){const auto now=tick();trace.c[3]+=now-ts;ts=now;}\\n // Four lanes jointly own a query row. Both online reductions and P stay in registers.\\n#pragma unroll\\n for(int rh=0;rh<2;rh++){\\n  float mx=-INFINITY;\\n#pragma unroll\\n  for(int n=0;n<NT;n++){\\n#pragma unroll\\n   for(int e=0;e<2;e++){\\n    float x=s[n][2*rh+e]*scale;\\n    if constexpr(MASK){\\n     const int key=kp+8*n+2*t+e;\\n     if(key>=hi || key>seq+2*warp+rh)x=-INFINITY;\\n    }\\n    s[n][2*rh+e]=x;mx=fmaxf(mx,x);\\n   }\\n  }\\n  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));\\n  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));\\n  const float nm=fmaxf(st.m[rh],mx),use=nm==-INFINITY?0.f:nm;\\n  const float alpha=st.m[rh]==-INFINITY?0.f:ex(st.m[rh]-use);\\n  float sum=0;\\n#pragma unroll\\n  for(int n=0;n<NT;n++){\\n   const float2 prob=exp2pair(s[n][2*rh]-use,s[n][2*rh+1]-use);\\n   const float a=prob.x,b=prob.y;\\n   s[n][2*rh]=a;s[n][2*rh+1]=b;sum+=a+b;\\n  }\\n  sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);\\n  st.l[rh]=st.l[rh]*alpha+sum;st.m[rh]=nm;\\n#pragma unroll\\n  for(int d=0;d<8;d++){st.o[d][2*rh]*=alpha;st.o[d][2*rh+1]*=alpha;}\\n }\\n if constexpr(CLOCK){const auto now=tick();trace.c[4]+=now-ts;ts=now;}\\n uint32_t packed[NT][2];\\n#pragma unroll\\n for(int n=0;n<NT;n++){\\n  packed[n][0]=pack16(s[n][0],s[n][1]);packed[n][1]=pack16(s[n][2],s[n][3]);\\n }\\n if constexpr(CLOCK){const auto now=tick();trace.c[5]+=now-ts;ts=now;}\\n#pragma unroll\\n for(int k=0;k<N/16;k++){\\n  const uint32_t pa[4]={packed[2*k][0],packed[2*k][1],packed[2*k+1][0],packed[2*k+1][1]};\\n#pragma unroll\\n  for(int d=0;d<8;d++){\\n   const int col=8*d+g,kk=16*k+2*t+vkey,perm=((col&7)*8)&(VN-1);\\n   const uint32_t b0=*(const uint32_t*)(V+2*(col*VN+(kk^perm)));\\n   const uint32_t b1=*(const uint32_t*)(V+2*(col*VN+((kk+8)^perm)));\\n   mma16(st.o[d],pa,b0,b1);\\n  }\\n }\\n if constexpr(CLOCK)trace.c[6]+=tick()-ts;\\n}\\ntemplate<int ST,int KT,bool CLOCK>\\n__global__ __launch_bounds__(128,2) void verify8_main(VP p){\\n Trace trace{};unsigned long long start=0,ts=0,init=0;if constexpr(CLOCK)start=tick();\\n constexpr int KB=KT*64;\\n extern __shared__ __align__(128) uint8_t sm[];\\n uint8_t* K=sm;uint8_t* V=K+ST*KB;uint8_t* Q=V+ST*KB;\\n constexpr int QRES=KT*128>8192?KT*128:8192;uint8_t* NK=Q+QRES;uint8_t* NV=NK+2048;uint8_t* KH=Q;uint8_t* VT=NV+2048;\\n const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;\\n const int h=blockIdx.x,split=blockIdx.y,b=blockIdx.z,seq=p.seq[b];\\n const bool last=split==p.ns-1;\\n const int chunk=((seq+p.ns-1)/p.ns+KT-1)&~(KT-1);\\n const int begin=min(seq,split*chunk),end=min(seq,begin+chunk),tiles=(end-begin+KT-1)/KT;\\n const int* pt=p.pt+(int64_t)b*p.pages;\\n constexpr int NR=KT/32;\\n int rows[NR];\\n auto rowids=[&](int it){\\n#pragma unroll\\n  for(int rr=0;rr<NR;rr++){\\n   const int key=begin+it*KT+tid/4+32*rr;\\n   rows[rr]=(it<tiles && key<end)?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):-1;\\n  }\\n };\\n auto issue=[&](int slot){\\n#pragma unroll\\n  for(int rr=0;rr<NR;rr++){\\n   const int row=tid/4+32*rr,col=16*(tid&3),r=rows[rr];\\n   const int64_t src=(int64_t)max(r,0)*p.HK*64+h*64+col;\\n   cp(K+slot*KB+off(row,col),p.kc+src,r>=0);\\n   cp(V+slot*KB+off(row,col),p.vc+src,r>=0);\\n  }\\n };\\n#pragma unroll\\n for(int s=0;s<ST-1;s++){rowids(s);issue(s);commit();}\\n rowids(ST-1);\\n#pragma unroll\\n for(int i=0;i<16;i++){\\n  const int row=warp+4*i,j=row/8,head=h*8+row%8;\\n  const float2 q=input_pair(p,b*8+j,head*64+2*lane);\\n  const float r=rsqrtf(wsum(q.x*q.x+q.y*q.y)/64.f+1.e-5f);\\n  *(uint32_t*)(Q+row*128+4*lane)=widen8(fp8pair(q.x*r,q.y*r));\\n }\\n if(last){\\n  for(int i=tid;i<4096/16;i+=128)*(uint4*)(NK+i*16)=make_uint4(0,0,0,0);\\n  __syncthreads();\\n#pragma unroll\\n  for(int i=0;i<4;i++){\\n   const int rr=warp+4*i,j=rr/2,isv=rr&1;\\n   float2 x=input_pair(p,b*8+j,(p.HQ+isv*p.HK+h)*64+2*lane);\\n   if(!isv){const float r=rsqrtf(wsum(x.x*x.x+x.y*x.y)/64.f+1.e-5f);x.x*=r;x.y*=r;}\\n   const float sc=*(isv?p.vs:p.ks);\\n   const uint16_t f=fp8pair(x.x/sc,x.y/sc);\\n   *(uint16_t*)((isv?p.vc:p.kc)+p.loc[b*8+j]*p.HK*64+h*64+2*lane)=f;\\n   *(uint16_t*)((isv?NV:NK)+off(j,2*lane))=f;\\n  }\\n }\\n __syncthreads();State st;\\n#pragma unroll\\n for(int k=0;k<4;k++){\\n  const int qr=warp*16+g;\\n  st.q[k][0]=*(const uint32_t*)(Q+qr*128+2*(16*k+2*t));\\n  st.q[k][1]=*(const uint32_t*)(Q+(qr+8)*128+2*(16*k+2*t));\\n  st.q[k][2]=*(const uint32_t*)(Q+qr*128+2*(16*k+8+2*t));\\n  st.q[k][3]=*(const uint32_t*)(Q+(qr+8)*128+2*(16*k+8+2*t));\\n }\\n st.m[0]=st.m[1]=-INFINITY;st.l[0]=st.l[1]=0;\\n#pragma unroll\\n for(int d=0;d<8;d++)for(int e=0;e<4;e++)st.o[d][e]=0;\\n const float scale=p.scale*(*p.ks);\\n if constexpr(CLOCK)init=tick()-start;\\n for(int it=0;it<tiles;it++){\\n  if constexpr(CLOCK)ts=tick();\\n  wait<ST-2>();__syncthreads();\\n  if constexpr(CLOCK){const auto now=tick();trace.c[1]+=now-ts;ts=now;}\\n  issue((it+ST-1)%ST);commit();rowids(it+ST);\\n  if constexpr(CLOCK){const auto now=tick();trace.c[0]+=now-ts;ts=now;}\\n  widen_stage<KT>(K+(it%ST)*KB,V+(it%ST)*KB,KH,VT);__syncthreads();\\n  if constexpr(CLOCK)trace.c[2]+=tick()-ts;\\n  // Register footprint stays at 64 scores while a handoff stages 128 keys.\\n  // Subtiles are independent warp work; no barrier or KV reload between them.\\n#pragma unroll 1\\n  for(int sub=0;sub<KT/64;sub++){\\n   const int key=begin+it*KT+sub*64;\\n   const uint8_t* kp=KH+sub*8192;\\n   const uint8_t* vp=VT;\\n   if(key+64<=end)tile<64,false,KT,CLOCK>(st,kp,vp,key,end,seq,scale,sub*64,trace);\\n   else tile<64,true,KT,CLOCK>(st,kp,vp,key,end,seq,scale,sub*64,trace);\\n  }\\n }\\n wait<0>();__syncthreads();\\n if(last){widen_stage<32>(NK,NV,KH,VT);__syncthreads();tile<32,true,32,CLOCK>(st,KH,VT,seq,seq+8,seq,scale,0,trace);}\\n float* out=p.partial+(((int64_t)b*p.HK+h)*p.ns+split)*ROWS*PR;\\n#pragma unroll\\n for(int rh=0;rh<2;rh++){\\n  const int row=warp*16+g+8*rh;\\n#pragma unroll\\n  for(int d=0;d<8;d++)*(float2*)(out+row*PR+8*d+2*t)=make_float2(st.o[d][2*rh],st.o[d][2*rh+1]);\\n  if(t==0)*(float2*)(out+row*PR+64)=make_float2(st.m[rh],st.l[rh]);\\n }\\n if constexpr(CLOCK){\\n  const unsigned long long elapsed=tick()-start;\\n  if(lane==0){\\n   auto* dst=p.clocks+((((int64_t)b*p.HK+h)*p.ns+split)*4+warp)*10;\\n#pragma unroll\\n   for(int k=0;k<7;k++)dst[k]=trace.c[k];\\n   dst[7]=init;dst[8]=elapsed;dst[9]=tiles;\\n  }\\n }\\n\\n}\\n// Each warp owns one query/head row. Grid size is B*HK*16 independent CTAs.\\n__global__ __launch_bounds__(128,4) void verify8_combine(VP p){\\n const int warp=threadIdx.x>>5,lane=threadIdx.x&31;\\n const int row=blockIdx.x*4+warp,h=blockIdx.y,b=blockIdx.z;\\n const float* base=p.partial+((int64_t)b*p.HK+h)*p.ns*ROWS*PR+row*PR;\\n float mx=-INFINITY;\\n for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,base[(int64_t)s*ROWS*PR+64]);\\n#pragma unroll\\n for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));\\n float total=0;\\n for(int s=lane;s<p.ns;s+=32){const float2 ml=*(const float2*)(base+(int64_t)s*ROWS*PR+64);total+=ml.y*ex(ml.x-mx);}\\n total=wsum(total);\\n float x=0,y=0;\\n for(int first=0;first<p.ns;first+=8){\\n  float2 ov[8],ml[8];\\n#pragma unroll\\n  for(int k=0;k<8;k++)if(first+k<p.ns){\\n   const float* pp=base+(int64_t)(first+k)*ROWS*PR;\\n   ov[k]=*(const float2*)(pp+2*lane);ml[k]=*(const float2*)(pp+64);\\n  }\\n#pragma unroll\\n  for(int k=0;k<8;k++)if(first+k<p.ns){const float a=ex(ml[k].x-mx);x=fmaf(a,ov[k].x,x);y=fmaf(a,ov[k].y,y);}\\n }\\n const float sc=(*p.vs)/total;x*=sc;y*=sc;\\n const float norm=rsqrtf(wsum(x*x+y*y)/64.f+1.e-5f);\\n const int j=row/8,head=h*8+row%8;\\n const float2 gate=input_pair(p,b*8+j,(p.HQ+2*p.HK+head)*64+2*lane);\\n *(__nv_bfloat162*)(p.out+((int64_t)(b*8+j)*p.HQ+head)*64+2*lane)=\\n __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate.x)),y*norm/(1.f+__expf(-gate.y)));\\n}\\ntemplate<int ST,int KT,bool CLOCK>void launch(VP p,cudaStream_t stream){\\n constexpr int smem=ST*KT*128+(KT*128>8192?KT*128:8192)+4096+KT*128;\\n cudaFuncSetAttribute(verify8_main<ST,KT,CLOCK>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\\n verify8_main<ST,KT,CLOCK><<<dim3(p.HK,p.ns,p.B),128,smem,stream>>>(p);\\n verify8_combine<<<dim3(16,p.HK,p.B),128,0,stream>>>(p);\\n}\\n}\\nvoid verify8_forward(int64_t qkv,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,\\n int64_t ks,int64_t vs,int64_t parts,int64_t out,int64_t B,int64_t HQ,int64_t HK,int64_t stride,\\n int64_t pages,int64_t ps,int64_t ns,int64_t proj_splits,double scale,int64_t stages,int64_t kt,int64_t profile,int64_t clocks,int64_t stream){\\n VP p{(const __nv_bfloat16*)qkv,(const int64_t*)loc,(const int*)pt,(const int*)seq,(uint8_t*)kc,(uint8_t*)vc,\\n (const float*)ks,(const float*)vs,(float*)parts,(__nv_bfloat16*)out,(int)B,(int)HQ,(int)HK,(int)stride,(int)pages,\\n (int)ps,0,(int)ns,(int)proj_splits,(float)scale,(unsigned long long*)clocks};\\n while((1<<p.lg)<ps)p.lg++;\\n const auto s=(cudaStream_t)stream;\\n if(profile){ if(kt==64){if(stages==3)launch<3,64,true>(p,s);else launch<4,64,true>(p,s);}\\n else {if(stages==3)launch<3,128,true>(p,s);else launch<4,128,true>(p,s);}}else{ if(kt==64){if(stages==3)launch<3,64,false>(p,s);else launch<4,64,false>(p,s);}\\n else {if(stages==3)launch<3,128,false>(p,s);else launch<4,128,false>(p,s);}}\\n const auto err=cudaGetLastError();if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));\\n}\\n\\nnamespace {\\ntemplate<int ST,int KT,bool C>std::vector<int64_t> resources(){\\n cudaFuncAttributes a;cudaFuncGetAttributes(&a,verify8_main<ST,KT,C>);\\n constexpr int smem=ST*KT*128+(KT*128>8192?KT*128:8192)+4096+KT*128;\\n cudaFuncSetAttribute(verify8_main<ST,KT,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\\n int active=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,verify8_main<ST,KT,C>,128,smem);\\n return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,active};\\n}\\n}\\nstd::vector<int64_t> verify8_resources(int64_t stages,int64_t kt,int64_t profile){\\n#define RES(S,K) if(stages==S && kt==K){return profile?resources<S,K,true>():resources<S,K,false>();}\\n RES(3,64) RES(4,64) RES(3,128) RES(4,128)\\n#undef RES\\n throw std::runtime_error("bad resource setting");\\n}\\n\'\n\n_EXT_PHASE={}\nPHASE_NAMES=(\'issue\',\'wait\',\'transpose\',\'qk\',\'softmax\',\'p_convert\',\'pv_layout\')\n\ndef phase_ext(wide=False):\n    global _EXT_PHASE\n    if wide not in _EXT_PHASE:\n        import hashlib\n        major,minor=torch.cuda.get_device_capability();arch=f\'{major}{minor}\'\n        source=WIDE_CUDA if wide else PHASE_CUDA\n        _EXT_PHASE[wide]=load_inline(name=f\'verify8_phase_{hashlib.sha256(source.encode()).hexdigest()[:12]}_{arch}\',\n            cpp_sources=PHASE_CPP,cuda_sources=source,functions=[\'verify8_forward\',\'verify8_resources\'],\n            extra_cuda_cflags=[\'-O3\',f\'-gencode=arch=compute_{arch},code=sm_{arch}\',\'-lineinfo\'],verbose=False)\n    return _EXT_PHASE[wide]\n\n\ndef phase_run(data,setting,profile,wide=False):\n    cfg=data[\'config\'];B=cfg[\'batch\'];HQ=cfg[\'num_q_heads\'];HK=cfg[\'num_kv_heads\'];ns,st,kt=setting\n    parts=torch.empty((B*HK*ns*64*68,),device=\'cuda\',dtype=torch.float32)\n    out=torch.empty((B,8,1024),device=\'cuda\',dtype=torch.bfloat16)\n    clocks=torch.empty((B*HK*ns,4,10),device=\'cuda\',dtype=torch.uint64) if profile else torch.empty(1,device=\'cuda\',dtype=torch.uint64)\n    phase_ext(wide).verify8_forward(data[\'qkv\'].data_ptr(),data[\'cache_loc\'].data_ptr(),data[\'page_table\'].data_ptr(),\n        data[\'seq_lens\'].data_ptr(),data[\'k_cache\'].data_ptr(),data[\'v_cache\'].data_ptr(),\n        data[\'k_scale\'].data_ptr(),data[\'v_scale\'].data_ptr(),parts.data_ptr(),out.data_ptr(),\n        B,HQ,HK,data[\'qkv\'].shape[-1],data[\'page_table\'].shape[1],1,ns,0,\n        cfg[\'sm_scale\']*cfg[\'qk_scale_factor\']*1.4426950408889634,st,kt,int(profile),clocks.data_ptr(),\n        torch.cuda.current_stream().cuda_stream)\n    return out,clocks\n\n\n\ndef phase_fa3(data):\n    raw=data[\'qkv\'].view(1,-1,2304)\n    q,g=qk_prep((raw,1),data,{\'inv_freq\':None},q_fp8=True)\n    return _attention_fa3_public(q,g,data,{})\n\n\ndef original_phase_lab(deadline,weights):\n    import json\n    lines=[\'META measured be19da94 producer; synthetic task inputs; clock cycles per stage; PV includes P layout shuffles\']\n    if deadline-time.monotonic()<25:return lines+[\'SKIP budget<25s before compile\']\n    ext=phase_ext();sms=torch.cuda.get_device_properties(\'cuda\').multi_processor_count\n    lines.append(f\'META sms={sms} clocks instrument every warp; compare plain/profile overhead and resource limits\')\n    flush=torch.empty(128<<20,dtype=torch.uint8,device=\'cuda\')\n    for B in (16,32):\n        if deadline-time.monotonic()<12:lines.append(f\'SKIP b{B} budget<12s\');break\n        data=generate_input(B,80000,\'tp8\',\'full\',741+B,q_len=8,page=1,spread=\'agentic\',dist=\'coherent\')\n        data[\'config\'][\'tokens\']=B*8\n        control_us,control,control_graph=timed(phase_fa3,data,flush)\n        control_expected=gate_prepare(data);cok,cmsg,cerr=gate_cached(data,control,control_expected)\n        lines.append(f\'FA3 B={B} us={control_us:.2f} gate={"P" if cok else "F"} err={cerr:.6f}\')\n        if not cok:lines.append(\'GATE_ERROR \'+cmsg)\n        base,_=phase_run(data,(17,3,128),False);torch.cuda.synchronize()\n        expected=gate_prepare(data);logical=float(data[\'seq_lens\'].sum())*2*2*64\n        for wide,setting in ((False,(17,3,128)),):\n            if deadline-time.monotonic()<4:lines.append(f\'SKIP b{B} remaining configs budget<4s\');break\n            ns,st,kt=setting\n            def plain(d):return phase_run(d,setting,False,wide)[0]\n            us,out,g=timed(plain,data,flush)\n            ok,msg,err=gate_cached(data,out,expected)\n            traced,clocks=phase_run(data,setting,True,wide);torch.cuda.synchronize()\n            equal=bool(torch.equal(out,traced));tok,tmsg,terr=gate_cached(data,traced,expected)\n            def traced_fn(d):return phase_run(d,setting,True,wide)[0]\n            instrument_us,_,tg=timed(traced_fn,data,flush,rounds=3)\n            values=clocks.cpu().to(torch.float64).reshape(-1,10);counts=values[:,9].clamp_min(1)\n            samples=values[:,:7]/counts[:,None]\n            med=samples.median(0).values.tolist();p90=torch.quantile(samples,.9,dim=0).tolist()\n            resources=[phase_ext(wide).verify8_resources(st,kt,i) for i in (0,1)]\n            longest=int(data[\'seq_lens\'].max());chunk=((longest+ns-1)//ns+kt-1)//kt*kt\n            row=dict(B=B,wide=wide,setting=setting,us=round(us,2),tb_s=round(logical/us/1.e6,3),gate=bool(ok and tok and equal),\n                error=round(err,6),trace_equal=equal,trace_us=round(instrument_us,2),grid=B*2*ns,\n                cta_kv_max_bytes=chunk*128,resources=resources,cta_cycles=[round(values[:,7].median().item()),round(torch.quantile(values[:,7],.9).item()),round(values[:,8].median().item()),round(torch.quantile(values[:,8],.9).item())],\n                cycles={k:[round(a),round(b)] for k,a,b in zip(PHASE_NAMES,med,p90)})\n            lines.append(\'ROW \'+json.dumps(row,separators=(\',\',\':\')))\n            if not row[\'gate\']:lines.append(\'GATE_ERROR \'+msg+\'; \'+tmsg)\n        # Graphs and closures leave scope on the next iteration; no captured name is deleted.\n    return lines\n\nNATIVE_CUDA=\'#include <cuda_runtime.h>\\n#include <cuda_fp16.h>\\n#include <cuda_bf16.h>\\n#include <cuda_fp8.h>\\n#include <cstdint>\\n#include <stdexcept>\\n#include <vector>\\n#define DI __device__ __forceinline__\\nnamespace {\\nconstexpr int D=64, ROWS=64, PR=68, THREADS=128;\\nstruct VP {\\n const __nv_bfloat16* qkv; const int64_t* loc; const int* pt; const int* seq;\\n uint8_t* kc; uint8_t* vc; const float* ks; const float* vs;\\n float* partial; __nv_bfloat16* out;\\n int B,HQ,HK,stride,pages,ps,lg,ns,proj_splits; float scale;\\n unsigned long long* clocks;\\n};\\nDI uint32_t su(const void* x){return (uint32_t)__cvta_generic_to_shared(x);}\\nDI void cp(void* dst,const void* src,bool valid){\\n asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(su(dst)),"l"(src),"r"(valid?16:0):"memory");\\n}\\nDI void commit(){asm volatile("cp.async.commit_group;":::"memory");}\\ntemplate<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}\\nDI float ex(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}\\nDI float2 exp2pair(float x,float y){\\n __half2 h=__floats2half2_rn(x,y);uint32_t bits=*(uint32_t*)&h,out;\\n asm("ex2.approx.f16x2 %0,%1;":"=r"(out):"r"(bits));\\n return __half22float2(*(__half2*)&out);\\n}\\nDI float wsum(float x){\\n#pragma unroll\\n for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffff,x,d);\\n return x;\\n}\\nDI uint16_t fp8pair(float x,float y){return __nv_cvt_float2_to_fp8x2(make_float2(x,y),__NV_SATFINITE,__NV_E4M3);}\\nDI int off(int r,int c){return r*64+(((c>>4)^((r>>1)&3))*16)+(c&15);}\\nDI uint32_t word(const uint8_t* p,int r,int c){return *(const uint32_t*)(p+off(r,c));}\\nDI float2 input_pair(const VP& p,int token,int col){\\n if(!p.proj_splits)return __bfloat1622float2(*(const __nv_bfloat162*)(p.qkv+(int64_t)token*p.stride+col));\\n const float* raw=(const float*)p.qkv;float2 sum=make_float2(0,0);\\n for(int s=0;s<p.proj_splits;s++){\\n  const float2 x=*(const float2*)(raw+((int64_t)s*p.B*8+token)*p.stride+col);\\n  sum.x+=x.x;sum.y+=x.y;\\n }\\n return __bfloat1622float2(__floats2bfloat162_rn(sum.x,sum.y));\\n}\\nDI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){\\n asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "\\n "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n}\\nstruct State {\\n uint32_t q[2][4];\\n float o[8][4];\\n float m[2],l[2];\\n};\\n// Convert an 8-key QK accumulator panel into two rows of FP8 pairs. No shared store.\\nDI uint32_t probpack(float (&s)[4]){\\n return (uint32_t)fp8pair(s[0]*256.f,s[1]*256.f)|((uint32_t)fp8pair(s[2]*256.f,s[3]*256.f)<<16);\\n}\\nstruct Trace { uint32_t c[7]; };\\nDI unsigned long long tick(){unsigned long long x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}\\n// One native Hopper warpgroup owns Q\\\'s 64 rows; four gather warps own the ring.\\n#ifndef USE_WGMMA\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900\\n#define USE_WGMMA 1\\n#else\\n#define USE_WGMMA 0\\n#endif\\n#endif\\nDI void consumer_bar(){asm volatile("bar.sync 1,128;":::"memory");}\\nDI void mb_init(uint64_t* p,int n){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(su(p)),"r"(n):"memory");}\\nDI void mb_arrive(uint64_t* p){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(su(p)):"memory");}\\nDI void cp_arrive(uint64_t* p){asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"::"r"(su(p)):"memory");}\\nDI void mb_wait(uint64_t* p,int phase){uint32_t ok;do{asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2;selp.b32 %0,1,0,p;}":"=r"(ok):"r"(su(p)),"r"(phase):"memory");}while(!ok);}\\nDI uint64_t descriptor(const uint8_t* p){return (uint64_t)(su(p)>>4)|(1ull<<16)|(32ull<<32)|(2ull<<62);}\\nDI void proxy_fence(){asm volatile("fence.proxy.async.shared::cta;":::"memory");}\\nDI void wg_fence(){\\n#if USE_WGMMA\\n asm volatile("wgmma.fence.sync.aligned;":::"memory");\\n#endif\\n}\\nDI void wg_wait(){\\n#if USE_WGMMA\\n asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n}\\nDI void protect(float (&a)[8][4]){\\n#pragma unroll\\n for(int n=0;n<8;n++)for(int e=0;e<4;e++)asm volatile("":"+f"(a[n][e])::"memory");\\n}\\nDI void wg8(float (&d)[8][4],uint64_t a,uint64_t b){\\n#if USE_WGMMA\\n asm volatile("{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1;}":"+f"(d[0][0]),"+f"(d[0][1]),"+f"(d[0][2]),"+f"(d[0][3]),"+f"(d[1][0]),"+f"(d[1][1]),"+f"(d[1][2]),"+f"(d[1][3]),"+f"(d[2][0]),"+f"(d[2][1]),"+f"(d[2][2]),"+f"(d[2][3]),"+f"(d[3][0]),"+f"(d[3][1]),"+f"(d[3][2]),"+f"(d[3][3]),"+f"(d[4][0]),"+f"(d[4][1]),"+f"(d[4][2]),"+f"(d[4][3]),"+f"(d[5][0]),"+f"(d[5][1]),"+f"(d[5][2]),"+f"(d[5][3]),"+f"(d[6][0]),"+f"(d[6][1]),"+f"(d[6][2]),"+f"(d[6][3]),"+f"(d[7][0]),"+f"(d[7][1]),"+f"(d[7][2]),"+f"(d[7][3]):"l"(a),"l"(b));\\n#endif\\n}\\n\\n// The transpose is shared by all 64 queries and preserves the E4M3 bytes.\\nDI void transpose_v64(const uint8_t* V,uint8_t* VT){\\n const int tid=threadIdx.x,t=tid&3;\\n#pragma unroll\\n for(int i=0;i<8;i++){\\n  const int group=tid/4+i*32,key=group/16*4,dim=group%16*4;\\n  const uint32_t x=word(V,key+t,dim);\\n  const int base=threadIdx.x&28;\\n  const uint32_t a=__shfl_sync(0xffffffff,x,base),b=__shfl_sync(0xffffffff,x,base+1);\\n  const uint32_t c=__shfl_sync(0xffffffff,x,base+2),d=__shfl_sync(0xffffffff,x,base+3);\\n  const uint32_t ab=__byte_perm(a,b,t*0x11+0x40),cd=__byte_perm(c,d,t*0x11+0x40);\\n  *(uint32_t*)(VT+off(dim+t,key))=__byte_perm(ab,cd,0x5410);\\n }\\n}\\nstruct WGState{float o[8][4],m[2],l[2];};\\n// The portable shim has the exact same 64-row accumulator coordinates.\\nDI void qk_group(float (&s)[8][4],const uint8_t* Q,const uint8_t* K){\\n#pragma unroll\\n for(int n=0;n<8;n++)for(int e=0;e<4;e++)s[n][e]=0;\\n protect(s);proxy_fence();wg_fence();\\n#pragma unroll\\n for(int kk=0;kk<2;kk++){\\n#if USE_WGMMA\\n  wg8(s,descriptor(Q+32*kk),descriptor(K+32*kk));\\n#else\\n  const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;\\n  const uint32_t a[4]={word(Q,w*16+g,32*kk+4*t),word(Q,w*16+g+8,32*kk+4*t),word(Q,w*16+g,32*kk+16+4*t),word(Q,w*16+g+8,32*kk+16+4*t)};\\n#pragma unroll\\n  for(int n=0;n<8;n++)mma8(s[n],a,word(K,n*8+g,32*kk+4*t),word(K,n*8+g,32*kk+16+4*t));\\n#endif\\n }\\n wg_wait();protect(s);\\n}\\nDI void pv_group(float (&o)[8][4],const uint8_t* P,const uint8_t* VT){\\n protect(o);proxy_fence();wg_fence();\\n#pragma unroll\\n for(int kk=0;kk<2;kk++){\\n#if USE_WGMMA\\n  wg8(o,descriptor(P+32*kk),descriptor(VT+32*kk));\\n#else\\n  const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;\\n  const uint32_t a[4]={word(P,w*16+g,32*kk+4*t),word(P,w*16+g+8,32*kk+4*t),word(P,w*16+g,32*kk+16+4*t),word(P,w*16+g+8,32*kk+16+4*t)};\\n#pragma unroll\\n  for(int n=0;n<8;n++)mma8(o[n],a,word(VT,n*8+g,32*kk+4*t),word(VT,n*8+g,32*kk+16+4*t));\\n#endif\\n }\\n wg_wait();protect(o);\\n}\\ntemplate<bool MASK,bool CLOCK>\\nDI void native_tile(WGState& st,const uint8_t* Q,const uint8_t* K,const uint8_t* V,uint8_t* P,uint8_t* VT,int kp,int hi,int seq,float scale,Trace& trace){\\n unsigned long long ts=0;if constexpr(CLOCK)ts=tick();\\n transpose_v64(V,VT);consumer_bar();\\n if constexpr(CLOCK){const auto now=tick();trace.c[2]+=now-ts;ts=now;}\\n float s[8][4];qk_group(s,Q,K);\\n if constexpr(CLOCK){const auto now=tick();trace.c[3]+=now-ts;ts=now;}\\n const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;\\n#pragma unroll\\n for(int rh=0;rh<2;rh++){\\n  float mx=-INFINITY;\\n#pragma unroll\\n  for(int n=0;n<8;n++){\\n#pragma unroll\\n   for(int e=0;e<2;e++){\\n    float x=s[n][2*rh+e]*scale;\\n    if constexpr(MASK){const int key=kp+8*n+2*t+e;if(key>=hi||key>seq+2*warp+rh)x=-INFINITY;}\\n    s[n][2*rh+e]=x;mx=fmaxf(mx,x);\\n   }\\n  }\\n  mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));\\n  const float nm=fmaxf(st.m[rh],mx),safe=nm==-INFINITY?0.f:nm;\\n  const float alpha=st.m[rh]==-INFINITY?0.f:ex(st.m[rh]-safe);\\n  float sum=0;\\n#pragma unroll\\n  for(int n=0;n<8;n++){\\n   const float2 p=exp2pair(s[n][2*rh]-safe,s[n][2*rh+1]-safe);\\n   s[n][2*rh]=p.x;s[n][2*rh+1]=p.y;sum+=p.x+p.y;\\n  }\\n  sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);\\n  st.m[rh]=nm;st.l[rh]=st.l[rh]*alpha+sum;\\n#pragma unroll\\n  for(int n=0;n<8;n++){st.o[n][2*rh]*=alpha;st.o[n][2*rh+1]*=alpha;}\\n }\\n if constexpr(CLOCK){const auto now=tick();trace.c[4]+=now-ts;ts=now;}\\n#pragma unroll\\n for(int n=0;n<8;n++){\\n  *(uint16_t*)(P+off(warp*16+g,8*n+2*t))=fp8pair(s[n][0]*256.f,s[n][1]*256.f);\\n  *(uint16_t*)(P+off(warp*16+g+8,8*n+2*t))=fp8pair(s[n][2]*256.f,s[n][3]*256.f);\\n }\\n consumer_bar();\\n if constexpr(CLOCK){const auto now=tick();trace.c[5]+=now-ts;ts=now;}\\n pv_group(st.o,P,VT);\\n if constexpr(CLOCK)trace.c[6]+=tick()-ts;\\n}\\ntemplate<int ST,bool CLOCK>\\n__global__ __launch_bounds__(256,2)void native_verify8(VP p){\\n extern __shared__ __align__(1024) uint8_t sm[];\\n uint8_t* K=sm;uint8_t* V=K+ST*4096;uint8_t* Q=V+ST*4096;\\n uint8_t* NK=Q+4096;uint8_t* NV=NK+4096;uint8_t* P=NV+4096;uint8_t* VT=P+4096;\\n __shared__ __align__(16)uint64_t full[ST],empty[ST];\\n const int tid=threadIdx.x,warp=tid>>5,lane=tid&31;\\n const int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];const bool last=sp==p.ns-1;\\n const int chunk=((seq+p.ns-1)/p.ns+63)&~63;\\n const int begin=min(seq,sp*chunk),end=min(seq,begin+chunk),ntiles=(end-begin+63)/64;\\n if(tid<ST){mb_init(&full[tid],128);mb_init(&empty[tid],128);}\\n __syncthreads();proxy_fence();\\n if(tid>=128){\\n  const int ptid=tid-128;Trace trace{};unsigned long long start=0;if constexpr(CLOCK)start=tick();\\n  for(int it=0;it<ntiles;it++){\\n   unsigned long long ts=0;if constexpr(CLOCK)ts=tick();\\n   const int slot=it%ST;\\n   if(it>=ST)mb_wait(&empty[slot],((it/ST)-1)&1);\\n   if constexpr(CLOCK){const auto now=tick();trace.c[1]+=now-ts;ts=now;}\\n#pragma unroll\\n   for(int rr=0;rr<2;rr++){\\n    const int row=ptid/4+32*rr,col=16*(ptid&3),key=begin+it*64+row;\\n    const bool valid=key<end;\\n    const int physical=valid?p.pt[(int64_t)b*p.pages+(key>>p.lg)]*p.ps+(key&(p.ps-1)):0;\\n    const int64_t src=(int64_t)physical*p.HK*64+h*64+col;\\n    cp(K+slot*4096+off(row,col),p.kc+src,valid);cp(V+slot*4096+off(row,col),p.vc+src,valid);\\n   }\\n   commit();cp_arrive(&full[slot]);wait<ST-1>();\\n   if constexpr(CLOCK)trace.c[0]+=tick()-ts;\\n  }\\n  wait<0>();\\n  if constexpr(CLOCK){if(lane==0){auto* dst=p.clocks+((((int64_t)b*p.HK+h)*p.ns+sp)*8+warp)*10;for(int i=0;i<7;i++)dst[i]=trace.c[i];dst[7]=0;dst[8]=tick()-start;dst[9]=ntiles;}}\\n  return;\\n }\\n Trace trace{};unsigned long long start=0,init=0;if constexpr(CLOCK)start=tick();\\n#pragma unroll\\n for(int i=0;i<16;i++){\\n  const int row=warp+4*i,j=row/8,head=h*8+row%8;\\n  const float2 q=input_pair(p,b*8+j,head*64+2*lane);\\n  const float n=rsqrtf(wsum(q.x*q.x+q.y*q.y)/64.f+1.e-5f);\\n  *(uint16_t*)(Q+off(row,2*lane))=fp8pair(q.x*n,q.y*n);\\n }\\n if(last){\\n  for(int i=tid;i<8192/16;i+=128)*(uint4*)(NK+16*i)=make_uint4(0,0,0,0);\\n  consumer_bar();\\n#pragma unroll\\n  for(int i=0;i<4;i++){\\n   const int r=warp+4*i,j=r/2,isv=r&1;\\n   float2 x=input_pair(p,b*8+j,(p.HQ+isv*p.HK+h)*64+2*lane);\\n   if(!isv){const float n=rsqrtf(wsum(x.x*x.x+x.y*x.y)/64.f+1.e-5f);x.x*=n;x.y*=n;}\\n   const uint16_t f=fp8pair(x.x/(*(isv?p.vs:p.ks)),x.y/(*(isv?p.vs:p.ks)));\\n   *(uint16_t*)((isv?p.vc:p.kc)+p.loc[b*8+j]*p.HK*64+h*64+2*lane)=f;\\n   *(uint16_t*)((isv?NV:NK)+off(j,2*lane))=f;\\n  }\\n }\\n consumer_bar();WGState st;\\n#pragma unroll\\n for(int n=0;n<8;n++)for(int e=0;e<4;e++)st.o[n][e]=0;\\n st.m[0]=st.m[1]=-INFINITY;st.l[0]=st.l[1]=0;\\n if constexpr(CLOCK)init=tick()-start;\\n for(int it=0;it<ntiles;it++){\\n  unsigned long long ts=0;if constexpr(CLOCK)ts=tick();\\n  mb_wait(&full[it%ST],(it/ST)&1);consumer_bar();\\n  if constexpr(CLOCK)trace.c[1]+=tick()-ts;\\n  const int key=begin+it*64;\\n  if(key+64<=end)native_tile<false,CLOCK>(st,Q,K+(it%ST)*4096,V+(it%ST)*4096,P,VT,key,end,seq,p.scale*(*p.ks),trace);\\n  else native_tile<true,CLOCK>(st,Q,K+(it%ST)*4096,V+(it%ST)*4096,P,VT,key,end,seq,p.scale*(*p.ks),trace);\\n  consumer_bar();mb_arrive(&empty[it%ST]);\\n }\\n if(last)native_tile<true,CLOCK>(st,Q,NK,NV,P,VT,seq,seq+8,seq,p.scale*(*p.ks),trace);\\n const int g=lane>>2,t=lane&3;\\n float* out=p.partial+(((int64_t)b*p.HK+h)*p.ns+sp)*64*68;\\n#pragma unroll\\n for(int rh=0;rh<2;rh++){\\n  const int row=warp*16+g+8*rh;\\n#pragma unroll\\n  for(int d=0;d<8;d++)*(float2*)(out+row*68+8*d+2*t)=make_float2(st.o[d][2*rh],st.o[d][2*rh+1]);\\n  if(t==0)*(float2*)(out+row*68+64)=make_float2(st.m[rh],st.l[rh]);\\n }\\n if constexpr(CLOCK){if(lane==0){auto* dst=p.clocks+((((int64_t)b*p.HK+h)*p.ns+sp)*8+warp)*10;for(int i=0;i<7;i++)dst[i]=trace.c[i];dst[7]=init;dst[8]=tick()-start;dst[9]=ntiles;}}\\n}\\n// Each warp owns one query/head row. Grid size is B*HK*16 independent CTAs.\\n__global__ __launch_bounds__(128,4) void verify8_combine(VP p){\\n const int warp=threadIdx.x>>5,lane=threadIdx.x&31;\\n const int row=blockIdx.x*4+warp,h=blockIdx.y,b=blockIdx.z;\\n const float* base=p.partial+((int64_t)b*p.HK+h)*p.ns*ROWS*PR+row*PR;\\n float mx=-INFINITY;\\n for(int s=lane;s<p.ns;s+=32)mx=fmaxf(mx,base[(int64_t)s*ROWS*PR+64]);\\n#pragma unroll\\n for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));\\n float total=0;\\n for(int s=lane;s<p.ns;s+=32){const float2 ml=*(const float2*)(base+(int64_t)s*ROWS*PR+64);total+=ml.y*ex(ml.x-mx);}\\n total=wsum(total);\\n float x=0,y=0;\\n for(int first=0;first<p.ns;first+=8){\\n  float2 ov[8],ml[8];\\n#pragma unroll\\n  for(int k=0;k<8;k++)if(first+k<p.ns){\\n   const float* pp=base+(int64_t)(first+k)*ROWS*PR;\\n   ov[k]=*(const float2*)(pp+2*lane);ml[k]=*(const float2*)(pp+64);\\n  }\\n#pragma unroll\\n  for(int k=0;k<8;k++)if(first+k<p.ns){const float a=ex(ml[k].x-mx);x=fmaf(a,ov[k].x,x);y=fmaf(a,ov[k].y,y);}\\n }\\n const float sc=(*p.vs)/(total*256.f);x*=sc;y*=sc;\\n const float norm=rsqrtf(wsum(x*x+y*y)/64.f+1.e-5f);\\n const int j=row/8,head=h*8+row%8;\\n const float2 gate=input_pair(p,b*8+j,(p.HQ+2*p.HK+head)*64+2*lane);\\n *(__nv_bfloat162*)(p.out+((int64_t)(b*8+j)*p.HQ+head)*64+2*lane)=\\n __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate.x)),y*norm/(1.f+__expf(-gate.y)));\\n}\\n\\ntemplate<int ST,bool C>void native_launch(VP p,cudaStream_t stream){\\n constexpr int smem=ST*8192+20480;\\n cudaFuncSetAttribute(native_verify8<ST,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\\n native_verify8<ST,C><<<dim3(p.ns,p.HK,p.B),256,smem,stream>>>(p);\\n verify8_combine<<<dim3(16,p.HK,p.B),128,0,stream>>>(p);\\n}\\n}\\nvoid native_forward(int64_t qkv,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,\\n int64_t ks,int64_t vs,int64_t parts,int64_t out,int64_t B,int64_t HQ,int64_t HK,int64_t stride,\\n int64_t pages,int64_t ps,int64_t ns,int64_t proj_splits,double scale,int64_t stages,int64_t profile,int64_t clocks,int64_t stream){\\n VP p{(const __nv_bfloat16*)qkv,(const int64_t*)loc,(const int*)pt,(const int*)seq,(uint8_t*)kc,(uint8_t*)vc,\\n (const float*)ks,(const float*)vs,(float*)parts,(__nv_bfloat16*)out,(int)B,(int)HQ,(int)HK,(int)stride,(int)pages,\\n (int)ps,0,(int)ns,(int)proj_splits,(float)scale,(unsigned long long*)clocks};\\n while((1<<p.lg)<ps)p.lg++;\\n auto stream_ptr=(cudaStream_t)stream;\\n if(stages==4){if(profile)native_launch<4,true>(p,stream_ptr);else native_launch<4,false>(p,stream_ptr);}\\n else if(stages==8){if(profile)native_launch<8,true>(p,stream_ptr);else native_launch<8,false>(p,stream_ptr);}\\n else throw std::runtime_error("native stages must be 4 or 8");\\n auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));\\n}\\nnamespace {\\ntemplate<int ST,bool C>std::vector<int64_t> native_resource(){\\n cudaFuncAttributes a;cudaFuncGetAttributes(&a,native_verify8<ST,C>);\\n constexpr int smem=ST*8192+20480;\\n cudaFuncSetAttribute(native_verify8<ST,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\\n int active=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,native_verify8<ST,C>,256,smem);\\n return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,active};\\n}\\n}\\nstd::vector<int64_t> native_resources(int64_t stages,int64_t clocks){\\n if(stages==4)return clocks?native_resource<4,true>():native_resource<4,false>();\\n return clocks?native_resource<8,true>():native_resource<8,false>();\\n}\\n\'\nNATIVE_CPP=\'#include <cstdint>\\n#include <vector>\\nvoid native_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t,int64_t,int64_t);\\nstd::vector<int64_t> native_resources(int64_t,int64_t);\\n\'\nimport hashlib\n\n_NATIVE_EXT={}\n\ndef native_ext(force_hopper=False):\n    # force_hopper exercises the Hopper Python dispatcher locally; the CUDA\n    # shim preserves its stage/barrier/layout work on the sm120 screen GPU.\n    major,minor=torch.cuda.get_device_capability()\n    arch=\'90a\' if major==9 else f\'{major}{minor}\'\n    key=(arch,bool(force_hopper))\n    if key not in _NATIVE_EXT:\n        _NATIVE_EXT[key]=load_inline(name=f\'verify8_native_{hashlib.sha256(NATIVE_CUDA.encode()).hexdigest()[:12]}_{arch}\',\n            cpp_sources=NATIVE_CPP,cuda_sources=NATIVE_CUDA,functions=[\'native_forward\',\'native_resources\'],\n            extra_cuda_cflags=[\'-O3\',f\'-gencode=arch=compute_{arch},code=sm_{arch}\',\'-lineinfo\'],verbose=False)\n    return _NATIVE_EXT[key]\n\n\ndef native_attention(qkv,data,setting=(17,4),clocks=False,force_hopper=False):\n    cfg=data[\'config\'];B=cfg[\'batch\'];HQ=cfg[\'num_q_heads\'];HK=cfg[\'num_kv_heads\']\n    assert cfg[\'q_len\']==8 and not cfg[\'window\'] and HQ==8*HK and cfg[\'head_dim\']==64\n    count=0\n    if isinstance(qkv,tuple):\n        raw,count=qkv\n        if raw.dtype==torch.bfloat16:\n            assert count==1\n            count=0\n        else:assert raw.dtype==torch.float32\n    else:raw=qkv\n    assert raw.is_contiguous()\n    ns,st=setting;assert 1<=ns<=128 and st in (4,8)\n    parts=torch.empty((B*HK*ns*64*68,),device=raw.device,dtype=torch.float32)\n    out=torch.empty((B,8,HQ*64),device=raw.device,dtype=torch.bfloat16)\n    trace=torch.empty((B*HK*ns,8,10) if clocks else (1,),device=raw.device,dtype=torch.uint64)\n    native_ext(force_hopper).native_forward(raw.data_ptr(),data[\'cache_loc\'].data_ptr(),data[\'page_table\'].data_ptr(),\n        data[\'seq_lens\'].data_ptr(),data[\'k_cache\'].data_ptr(),data[\'v_cache\'].data_ptr(),\n        data[\'k_scale\'].data_ptr(),data[\'v_scale\'].data_ptr(),parts.data_ptr(),out.data_ptr(),\n        B,HQ,HK,raw.shape[-1],data[\'page_table\'].shape[1],cfg[\'page_size\'],ns,count,\n        cfg[\'sm_scale\']*cfg[\'qk_scale_factor\']*1.4426950408889634,st,int(clocks),trace.data_ptr(),\n        torch.cuda.current_stream(raw.device).cuda_stream)\n    return out,trace\n\n\ndef attention_verify8_native(qkv,data,setting=(17,4)):\n    cfg=dict(data[\'config\'],num_q_heads=16,num_kv_heads=2,head_dim=64)\n    return native_attention(qkv,dict(data,config=cfg),setting,force_hopper=True)[0].view(cfg[\'batch\']*8,1024)\n\n\ndef native_lab(deadline,weights):\n    import json\n    lines=[\'META native FP8 WGMMA Q-major M64; parallel combine; same-input FA3; no incumbent selection\',\n           \'META clocks: consumer wait/transpose/QK/softmax/P/PV plus producer issue/backpressure; resource and plain/clock wall times\']\n    if deadline-time.monotonic()<20:return lines+[\'SKIP native compile budget<20s\']\n    native_ext(True)\n    flush=torch.empty(128<<20,dtype=torch.uint8,device=\'cuda\')\n    for B in (16,32):\n        if deadline-time.monotonic()<8:lines.append(f\'SKIP b{B} budget<8s\');break\n        data=generate_input(B,80000,\'tp8\',\'full\',741+B,q_len=8,page=1,spread=\'agentic\',dist=\'coherent\');data[\'config\'][\'tokens\']=B*8\n        def control(d):\n            q,g=qk_prep((d[\'qkv\'].view(1,-1,2304),1),d,{\'inv_freq\':None},q_fp8=True)\n            return _attention_fa3_public(q,g,d,{})\n        us,actual,graph=timed(control,data,flush)\n        expected=gate_prepare(data);ok,msg,error=gate_cached(data,actual,expected)\n        lines.append(f\'FA3 B={B} us={us:.2f} gate={"P" if ok else "F"} err={error:.6f}\')\n        if not ok:lines.append(\'GATE_ERROR \'+msg)\n        logical=float(data[\'seq_lens\'].sum())*256\n        for setting in ((9,4),(17,4),(17,8),(33,4)):\n            if deadline-time.monotonic()<3:lines.append(f\'SKIP b{B} native tail budget<3s\');break\n            def call(d):return native_attention(d[\'qkv\'],d,setting,force_hopper=True)[0]\n            us,out,g=timed(call,data,flush)\n            ok,msg,error=gate_cached(data,out,expected)\n            trace_out,trace=native_attention(data[\'qkv\'],data,setting,clocks=True,force_hopper=True);torch.cuda.synchronize()\n            tok,tmsg,terr=gate_cached(data,trace_out,expected);equal=bool(torch.equal(out,trace_out))\n            values=trace.cpu().to(torch.float64);count=values[:,:,9].clamp_min(1)\n            consumer=values[:,:4,:7]/count[:,:4,None]\n            producer=values[:,4:,:2]/count[:,4:,None]\n            cm=consumer.reshape(-1,7).median(0).values.tolist();pm=producer.reshape(-1,2).median(0).values.tolist()\n            def trace_call(d):return native_attention(d[\'qkv\'],d,setting,clocks=True,force_hopper=True)[0]\n            trace_us,_,trace_graph=timed(trace_call,data,flush,rounds=3)\n            resources=[native_ext(True).native_resources(setting[1],i) for i in (0,1)]\n            row=dict(B=B,setting=setting,us=round(us,2),tb_s=round(logical/us/1.e6,3),gate=bool(ok and tok and equal),error=round(error,6),trace_equal=equal,\n                grid=B*2*setting[0],trace_us=round(trace_us,2),resources=resources,consumer_cycles=[round(x) for x in cm],producer_cycles=[round(x) for x in pm])\n            lines.append(\'NATIVE \'+json.dumps(row,separators=(\',\',\':\')))\n            if not row[\'gate\']:lines.append(\'GATE_ERROR \'+msg+\'; \'+tmsg)\n    return lines\n\n\ndef lab(deadline,weights):\n    start=time.monotonic()\n    # Old phases use a small fixed share; the pair gets its own equal share\n    # from h200_lab. No kernel result is read outside this prepare lab.\n    first=original_phase_lab(min(deadline,start+35.),weights)\n    second=native_lab(deadline,weights)\n    return first+second\n'), ('verify1_sliding_fa3_pair', 'a652ee6400e5c27b', 'import math, time\nimport torch\nfrom torch.utils.cpp_extension import load_inline\nROW_FACTOR=5.0\nROW_NORM_FLOOR=.1\ndef relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:\n    """|a - b| / |b|."""\n    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))\n\n\ndef query_precision_bound(e4m3_err: float, factor: float, floor: float, cap: float, slack: float = 0.0) -> float:\n    """Attention\'s bound for one case: `factor` times the reference\'s own error with queries\n    rounded to FP8 E4M3 (`e4m3_err`), plus `slack`, within [floor, cap]."""\n    return min(cap, max(floor, factor * e4m3_err + slack))\n\n\ndef compare(output, ref: torch.Tensor, tol: float) -> tuple[bool, str]:\n    if not isinstance(output, torch.Tensor):\n        return False, f"expected a tensor, got {type(output).__name__}"\n    if output.shape != ref.shape or output.dtype != ref.dtype:\n        return False, f"expected {tuple(ref.shape)} {ref.dtype}, got {tuple(output.shape)} {output.dtype}"\n    out = output.float()\n    if not torch.isfinite(out).all():\n        return False, "output contains NaN or Inf"\n    r = ref.float()\n    diff = out - r\n    err = float(diff.norm() / r.norm().clamp(min=1e-12))\n    if err > tol:\n        return False, f"relative L2 error {err:.3e} > {tol:g}"\n\n    rows_diff = diff.reshape(-1, r.shape[-1]).norm(dim=-1)\n    rows_ref = r.reshape(-1, r.shape[-1]).norm(dim=-1)\n    floor = (ROW_NORM_FLOOR * rows_ref.mean()).clamp(min=1e-12)\n    row_err = rows_diff / torch.maximum(rows_ref, floor)\n    worst = int(row_err.argmax())\n    worst_err = float(row_err[worst])\n    if worst_err > ROW_FACTOR * tol:\n        index = tuple(int(i) for i in torch.unravel_index(torch.tensor(worst), r.shape[:-1])) if r.dim() > 1 else ()\n        return False, (f"relative L2 error {err:.3e}, but row {index} is off by {worst_err:.3e}"\n                       f" > {ROW_FACTOR:g} x {tol:g}")\n    return True, f"relative L2 error {err:.3e} (worst row {worst_err:.3e})"\n\nHEAD_DIM = 64\n\nHEADS = {"tp8": (16, 2), "dp": (128, 16)}  # (query heads, key/value heads) on one GPU\n\nSLIDING_WINDOW = 2048\n\nROPE_THETA = 500_000.0\n\nQK_SCALE_FACTOR = 5.4730064863875\n\nSM_SCALE = HEAD_DIM ** -0.5\n\nRMS_EPS = 1e-5\n\nPAGE = 64\n\nFP8 = torch.float8_e4m3fn\n\nK_SCALE = V_SCALE = 1.0\n\nREL_L2_TOL = 0.15\n\nFP8_QUERY_FACTOR = 2.5\n\nFP8_QUERY_FLOOR = 0.03\n\nFP8_QUERY_SLACK = 0.015\n\nQUERY_DTYPE = torch.bfloat16\n\nCACHE_TOL = 8e-2\n\nSOFT_C2 = {"sliding": (0.3, 0.75), "full": (0.5, 0.62)}\n\nCOHERENT_HEAD = 2048\n\nCOHERENT_ALPHA = 1.0\n\ndef _coherent_tail(v_cache, k_cache, q, seq_lens, page_table, layer, seed, page, scale):\n    """Add the shared direction to the values of each request\'s low-attention keys."""\n    hkv, hq = v_cache.shape[-2], q.shape[-2]\n    group = hq // hkv\n    tail_gen = torch.Generator(device=v_cache.device).manual_seed(seed + 0xC0DE)\n    u = torch.randn((hkv, HEAD_DIM), generator=tail_gen, device=v_cache.device)\n    u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))\n    qn = _rms(q) * QK_SCALE_FACTOR  # [B, L, Hq, 64], as the reference attends (RoPE on sliding)\n    if layer == "sliding":\n        pos = seq_lens.long()[:, None] + torch.arange(q.shape[1], device=q.device)\n        qn = _rope(qn, pos, ROPE_THETA)\n    k_flat = k_cache.view(-1, hkv, HEAD_DIM)\n    v_flat = v_cache.view(-1, hkv, HEAD_DIM)\n    for b in range(len(seq_lens)):\n        seq = int(seq_lens[b])\n        first = max(0, seq - SLIDING_WINDOW + 1) if layer == "sliding" else 0\n        past = torch.arange(first, seq, device=q.device)\n        rows = page_table[b].long()[past // page] * page + past % page\n        keys = _dequant(k_flat[rows], scale)  # [S, Hkv, 64]\n        # Each key\'s best logit over its KV head\'s query heads and the block\'s tokens.\n        logits = torch.einsum("lkgd,skd->kgls", qn[b].view(-1, hkv, group, HEAD_DIM), keys)\n        best = logits.flatten(1, 2).amax(1)  # [Hkv, S]\n        if best.shape[1] <= COHERENT_HEAD:\n            continue\n        tail = best < best.topk(COHERENT_HEAD, -1).values[:, -1:]  # [Hkv, S]\n        vals = _dequant(v_flat[rows], scale) + COHERENT_ALPHA * tail.T[..., None] * u\n        v_flat[rows] = _quant(vals, scale)\n\nAGENTIC_SPREAD = 0.075\n\ndef _longest(ctx: int, spread: str) -> int:\n    """The longest length _lengths can draw."""\n    if spread == "fixed":\n        return ctx\n    if spread == "agentic":\n        return math.ceil(ctx * (1 + AGENTIC_SPREAD))\n    return 4 * ctx\n\ndef _lengths(gen, batch: int, ctx: int, spread: str, device) -> torch.Tensor:\n    if spread == "fixed":\n        return torch.full((batch,), ctx, dtype=torch.int64, device=device)\n    if spread == "agentic":\n        u = torch.rand((batch,), generator=gen, device=device)\n        return (ctx * (1 + AGENTIC_SPREAD * (2 * u - 1))).long().clamp(1, _longest(ctx, spread))\n    # Log-normal around ctx: many short requests, a long tail.\n    z = torch.randn((batch,), generator=gen, device=device)\n    return (ctx * torch.exp(z - 0.5)).clamp(16, 4 * ctx).long()\n\ndef _shortest_share(spread: str) -> float:\n    """A lower bound on mean length / ctx, for the speed-of-light bound."""\n    return {"fixed": 1.0, "agentic": 1 - AGENTIC_SPREAD}.get(spread, 0.5)\n\ndef _quant(x, scale):\n    return (x.float() / scale).to(FP8)\n\ndef _dequant(x, scale):\n    return x.float() * scale\n\nCACHE_CHUNK = 1 << 20  # cache rows generated at a time\n\ndef _fill_cache(gen, shape, device, scale=K_SCALE, seed=0, dist="normal", layer="full"):\n    """FP8 key and value caches of `shape`: N(0, 1) entries, soft keys for dist="soft" and\n    "coherent" (see SOFT_C2)."""\n    hkv = shape[-2]\n    k_cache = torch.empty(shape, dtype=FP8, device=device)\n    v_cache = torch.empty(shape, dtype=FP8, device=device)\n    mix = None\n    if dist in ("soft", "coherent"):\n        # Its own generator, so the other inputs match the same seed\'s normal case.\n        soft = torch.Generator(device=device).manual_seed(seed + 0x50F7)\n        u = torch.randn((hkv, HEAD_DIM), generator=soft, device=device)\n        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))\n        lo, hi = SOFT_C2[layer]\n        c2 = lo + (hi - lo) * torch.rand((hkv, 1), generator=soft, device=device)\n        mix = (c2.sqrt() * u, (1 - c2).sqrt())\n    for dst, keys in ((k_cache, True), (v_cache, False)):\n        flat = dst.view(-1, hkv, HEAD_DIM)\n        for at in range(0, flat.shape[0], CACHE_CHUNK):\n            n = min(CACHE_CHUNK, flat.shape[0] - at)\n            x = torch.randn((n, hkv, HEAD_DIM), generator=gen, device=device)\n            if keys and mix is not None:\n                x = mix[0] + mix[1] * x\n            flat[at:at + n] = _quant(x, scale)\n    return k_cache, v_cache\n\ndef generate_input(batch: int, ctx: int, heads: str, layer: str, seed: int, q_len: int = 1,\n                   spread: str = "fixed", dist: str = "normal", page: int = PAGE,\n                   kv_scale: float = K_SCALE, device: str = "cuda", **_):\n    """layer: "sliding" (2,048-token window, RoPE) or "full" (every cached token, no RoPE).\n    q_len: 1 for a decode step, 8 for a speculative-decoding verify block (causal inside it).\n    dist: "normal", "soft" (see SOFT_C2), "coherent" (see COHERENT_HEAD), or "edges": keys\n    aligned with each KV head\'s queries at the window edges, relative to the block\'s first\n    token (sliding: the oldest position inside the window and the newest just outside it;\n    full: position 0)."""\n    gen = torch.Generator(device=device).manual_seed(seed)\n    hq, hkv = HEADS[heads]\n    seq_lens = _lengths(gen, batch, ctx, spread, device)\n    pages_per = (seq_lens + q_len + page - 1) // page  # room for the new tokens\n    # The pool and page table are sized for the longest possible length, so every\n    # repetition\'s inputs have the same shapes (one CUDA graph is replayed on fresh inputs).\n    longest = _longest(ctx, spread)\n    max_pages = (longest + q_len + page - 1) // page\n    total = batch * max_pages\n    # Physical pages are scattered.\n    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)\n    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)\n    offsets = torch.cumsum(pages_per, 0) - pages_per\n    for b in range(batch):\n        n, o = int(pages_per[b]), int(offsets[b])\n        page_table[b, :n] = perm[o:o + n]\n\n    # Cached keys are post-norm (unit RMS per head); values are projections, scale ~1.\n    shape = (total, page, hkv, HEAD_DIM)\n    k_cache, v_cache = _fill_cache(gen, shape, device, kv_scale, seed, dist, layer)\n    q = (2 * torch.randn((batch, q_len, hq, HEAD_DIM), generator=gen, device=device)).to(torch.bfloat16)\n    if dist == "coherent":\n        _coherent_tail(v_cache, k_cache, q, seq_lens, page_table, layer, seed, page, kv_scale)\n    if dist == "edges":\n        # Each KV head\'s key points along its group\'s first query head of the block\'s first\n        # token, as the kernel will see it (normalized, then RoPE on sliding layers).\n        qn = _rms(q[:, 0])\n        if layer == "sliding":\n            qn = _rope(qn, seq_lens, ROPE_THETA)\n        planted = _quant(qn[:, :: hq // hkv], kv_scale)  # [B, Hkv, 64]\n        k_flat = k_cache.view(-1, hkv, HEAD_DIM)\n        for b in range(batch):\n            p = int(seq_lens[b])\n            spots = [p - SLIDING_WINDOW + 1, p - SLIDING_WINDOW] if layer == "sliding" else [0]\n            for t in spots:\n                if 0 <= t < p:\n                    k_flat[int(page_table[b, t // page]) * page + t % page] = planted[b]\n    k_new = (2 * torch.randn((batch, q_len, hkv, HEAD_DIM), generator=gen, device=device)).to(torch.bfloat16)\n    v_new = torch.randn((batch, q_len, hkv, HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)\n    gate = torch.randn((batch, q_len, hq * HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)\n    # Fused projection output, per token: q | k | v | gate.\n    qkv = torch.cat([q.flatten(2), k_new.flatten(2), v_new.flatten(2), gate], -1)\n    # Step metadata: each new token\'s position and flat cache row, per-request cache lengths\n    # (including the new tokens) and query offsets, and on full-attention decode steps FA3\'s\n    # tile schedule.\n    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)\n    pos = positions.view(batch, q_len)\n    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)\n    cache_seqlens = (seq_lens + q_len).to(torch.int32)\n    cu_seqlens_q = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=device)\n    return {\n        "qkv": qkv,\n        "positions": positions,\n        "cache_loc": cache_loc,\n        "cache_seqlens": cache_seqlens,\n        "cu_seqlens_q": cu_seqlens_q,\n        "scheduler_metadata": _fa3_scheduler_metadata(cache_seqlens, cu_seqlens_q, hq, hkv, page, q_len, layer),\n        "k_cache": k_cache,\n        "v_cache": v_cache,\n        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),\n        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),\n        "page_table": page_table,\n        "seq_lens": seq_lens.to(torch.int32),\n        "config": {\n            "batch": batch,\n            "q_len": q_len,\n            "num_q_heads": hq,\n            "num_kv_heads": hkv,\n            "head_dim": HEAD_DIM,\n            "page_size": page,\n            "window": SLIDING_WINDOW if layer == "sliding" else 0,\n            "rope_theta": ROPE_THETA if layer == "sliding" else 0.0,\n            "qk_scale_factor": QK_SCALE_FACTOR,\n            "sm_scale": SM_SCALE,\n            "kv_scale": kv_scale,\n        },\n    }\n\ndef split_qkv(data):\n    """q [B, L, Hq, 64], k and v [B, L, Hkv, 64], gate [B, L, Hq*64] from the fused input."""\n    cfg = data["config"]\n    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]\n    batch, length = cfg["batch"], cfg["q_len"]\n    q, k, v, gate = data["qkv"].split([hq * d, hkv * d, hkv * d, hq * d], -1)\n    return (q.reshape(batch, length, hq, d), k.reshape(batch, length, hkv, d), v.reshape(batch, length, hkv, d),\n            gate)\n\ndef _rms(x, eps=RMS_EPS):\n    x = x.float()\n    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)\n\ndef _rope(x, pos, theta):\n    """Interleaved (GPT-J) RoPE: pairs (2i, 2i+1) rotate by pos * theta^(-2i/64).\n    x [..., heads, 64] with pos broadcast over everything but the last two dims."""\n    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=x.device).float() / HEAD_DIM))\n    ang = pos.float()[..., None] * inv_freq\n    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)\n    even, odd = x[..., 0::2], x[..., 1::2]\n    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)\n\ndef _rows(data, positions):\n    """Flat cache rows of [B, L] positions."""\n    size = data["config"]["page_size"]\n    page = torch.gather(data["page_table"].long(), 1, positions // size)\n    return page * size + positions % size\n\ndef new_slots(data) -> torch.Tensor:\n    """[B, L] flat cache rows where the block\'s new tokens go."""\n    pos = data["seq_lens"].long()[:, None] + torch.arange(data["config"]["q_len"], device=data["seq_lens"].device)\n    return _rows(data, pos)\n\ndef ref_kernel(data, query_dtype=None, new_kv=None):\n    """Returns (output [B, L, Hq*64], new keys and values [B, L, Hkv, 64] as stored: FP8).\n    query_dtype: the precision queries are rounded to (default QUERY_DTYPE). With new_kv\n    (FP8 keys and values [B, L, Hkv, 64]), the block attends those as its own entries."""\n    cfg = data["config"]\n    batch, length = cfg["batch"], cfg["q_len"]\n    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]\n    q_in, k_in, v_in, gate = split_qkv(data)\n    pos = data["seq_lens"].long()[:, None] + torch.arange(length, device=q_in.device)\n    q = _rms(q_in) * cfg["qk_scale_factor"]\n    k = _rms(k_in)\n    if cfg["rope_theta"]:\n        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])\n    # FP8 rounds the normalized query, before qk_scale_factor.\n    query_dtype = query_dtype or QUERY_DTYPE\n    if query_dtype == torch.bfloat16:\n        q = q.to(torch.bfloat16).float()\n    else:\n        q = (q / cfg["qk_scale_factor"]).to(query_dtype).float() * cfg["qk_scale_factor"]\n    k_new, v_new = _quant(k, data["k_scale"]), _quant(v_in, data["v_scale"])\n    if new_kv is not None:\n        k_new, v_new = new_kv\n    k_flat = data["k_cache"].view(-1, hkv, HEAD_DIM)\n    v_flat = data["v_cache"].view(-1, hkv, HEAD_DIM)\n    group = hq // hkv\n\n    out = torch.empty((batch, length, hq, HEAD_DIM), device=q.device)\n    for b in range(batch):\n        seq = int(data["seq_lens"][b])\n        past = torch.arange(seq, device=q.device)\n        size = cfg["page_size"]\n        past_rows = data["page_table"][b].long()[past // size] * size + past % size\n        keys_past = _dequant(k_flat[past_rows], data["k_scale"])\n        vals_past = _dequant(v_flat[past_rows], data["v_scale"])\n        keys = torch.cat([keys_past, _dequant(k_new[b], data["k_scale"])])  # positions 0 .. seq+L-1\n        vals = torch.cat([vals_past, _dequant(v_new[b], data["v_scale"])])\n        for j in range(length):\n            p = seq + j\n            start = max(0, p - cfg["window"] + 1) if cfg["window"] else 0\n            kb, vb = keys[start:p + 1], vals[start:p + 1]\n            att = torch.einsum("kgd,tkd->kgt", q[b, j].view(hkv, group, HEAD_DIM), kb) * cfg["sm_scale"]\n            out[b, j] = torch.einsum("kgt,tkd->kgd", att.softmax(-1), vb).reshape(hq, HEAD_DIM)\n    out = _rms(out) * torch.sigmoid(gate.float()).view(batch, length, hq, HEAD_DIM)\n    return out.reshape(batch, length, hq * HEAD_DIM), k_new, v_new\n\ndef collect(data, output) -> torch.Tensor:\n    """Runs in the submission\'s process after a step: the output followed by the new cache\n    entries, as raw bytes (see unpack)."""\n    cfg = data["config"]\n    shape = (cfg["batch"], cfg["q_len"], cfg["num_q_heads"] * HEAD_DIM)\n    if type(output) is not torch.Tensor or output.dtype != torch.bfloat16 or tuple(output.shape) != shape:\n        raise ValueError(f"expected a bfloat16 tensor of shape {shape}")\n    slots = new_slots(data)\n    hkv = cfg["num_kv_heads"]\n    k = data["k_cache"].view(-1, hkv, HEAD_DIM)[slots]\n    v = data["v_cache"].view(-1, hkv, HEAD_DIM)[slots]\n    return torch.cat([p.contiguous().view(torch.uint8).reshape(-1) for p in (output, k, v)])\n\ndef unpack(raw: torch.Tensor, cfg: dict):\n    b, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]\n    shapes = [((b, length, hq * HEAD_DIM), torch.bfloat16), ((b, length, hkv, HEAD_DIM), FP8),\n              ((b, length, hkv, HEAD_DIM), FP8)]\n    sizes = [math.prod(shape) * dtype.itemsize for shape, dtype in shapes]\n    if raw.dtype != torch.uint8 or raw.numel() != sum(sizes):\n        return None\n    out, at = [], 0\n    for (shape, dtype), n in zip(shapes, sizes):\n        out.append(raw[at:at + n].view(dtype).view(shape))\n        at += n\n    return out\n\ndef check_implementation(data, raw):\n    parts = unpack(raw, data["config"])\n    if parts is None:\n        return False, "malformed output"\n    out, k, v = parts\n    _, ref_k, ref_v = ref_kernel(data)\n    for name, got, want, scale in (("key", k, ref_k, data["k_scale"]), ("value", v, ref_v, data["v_scale"])):\n        ok, kv_msg = compare(_dequant(got, scale), _dequant(want, scale), CACHE_TOL)\n        if not ok:\n            return False, f"new {name} cache entries: {kv_msg}"\n    # The output is checked against the reference attending the submission\'s own new cache\n    # entries (checked above), since FP8 rounding of those entries varies by implementation.\n    ref = ref_kernel(data, new_kv=(k, v))[0].to(torch.bfloat16)\n    e4m3 = relative_l2(ref_kernel(data, torch.float8_e4m3fn, new_kv=(k, v))[0].to(torch.bfloat16), ref)\n    tol = query_precision_bound(e4m3, FP8_QUERY_FACTOR, FP8_QUERY_FLOOR, REL_L2_TOL, FP8_QUERY_SLACK)\n    ok, msg = compare(out, ref, tol)\n    msg = (f"{msg} [bound {tol:.3f}: {FP8_QUERY_FACTOR:g} x the reference\'s {e4m3:.3f} with FP8 E4M3 queries"\n           f" + {FP8_QUERY_SLACK:g}]")\n    return ok, msg\n\n\ndef _fa3_scheduler_metadata(*args,**kwargs): return None\n\n# The gate above is frozen from the task; this caches its three FP32 passes\n# per input only inside the untimed lab. Timed kernels never read these values.\ndef gate_prepare(data):\n    c=data[\'config\'];slots=new_slots(data);hk=c[\'num_kv_heads\']\n    actual=tuple(data[name].view(-1,hk,64)[slots].clone() for name in (\'k_cache\',\'v_cache\'))\n    _,want_k,want_v=ref_kernel(data)\n    want=ref_kernel(data,new_kv=actual)[0].to(torch.bfloat16)\n    fp8=ref_kernel(data,torch.float8_e4m3fn,new_kv=actual)[0].to(torch.bfloat16)\n    tol=query_precision_bound(relative_l2(fp8,want),FP8_QUERY_FACTOR,FP8_QUERY_FLOOR,REL_L2_TOL,FP8_QUERY_SLACK)\n    return dict(out=want,tol=tol,kv=(want_k,want_v),slots=slots)\n\n\ndef gate_cached(data,out,expected):\n    c=data[\'config\'];hk=c[\'num_kv_heads\']\n    for i,name in enumerate((\'k_cache\',\'v_cache\')):\n        actual=data[name].view(-1,hk,64)[expected[\'slots\']]\n        scale=data[\'k_scale\' if i==0 else \'v_scale\']\n        ok,msg=compare(_dequant(actual,scale),_dequant(expected[\'kv\'][i],scale),CACHE_TOL)\n        if not ok:return False,\'cache \'+msg,float(\'inf\')\n    out=out.view_as(expected[\'out\'])\n    ok,msg=compare(out,expected[\'out\'],expected[\'tol\'])\n    return ok,msg,relative_l2(out,expected[\'out\'])\n\n\ndef timed(fn,data,flush,rounds=5):\n    fn(data);torch.cuda.synchronize()\n    graph=torch.cuda.CUDAGraph()\n    with torch.cuda.graph(graph):\n        out=fn(data)\n    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)\n    values=[]\n    for _ in range(rounds):\n        flush.zero_();start.record();graph.replay();end.record();end.synchronize()\n        values.append(start.elapsed_time(end)*1000.)\n    return sorted(values)[len(values)//2],out,graph\n\nimport triton\nimport triton.language as tl\nfrom sgl_kernel.flash_attn import flash_attn_with_kvcache\nHQ,HKV,HD=16,2,64\nQ_SIZE=1024\nPDL=True\n"""Verify-1 sliding option: H200 attention kernel win; whole-layer comparison required."""\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n\n_INV_FREQ = {}\n\ndef _inv_freq(theta, device):\n    key = (float(theta), str(device))\n    if key not in _INV_FREQ:\n        _INV_FREQ[key] = (1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()\n    return _INV_FREQ[key]\n\n\nWINDOW_CUDA = \'\\n#include <cuda_runtime.h>\\n#include <cuda_fp16.h>\\n#include <cuda_bf16.h>\\n#include <cuda_fp8.h>\\n#include <cstdint>\\n#include <stdexcept>\\n#define DEVI __device__ __forceinline__\\nnamespace {\\nconstexpr int ROWB=64;\\nDEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }\\n\\nDEVI void cp_async16(uint32_t dst, const void* src, int src_bytes) {\\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\\\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");\\n}\\nDEVI void cp_commit() { asm volatile("cp.async.commit_group;\\\\n" ::: "memory"); }\\ntemplate <int N>\\nDEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\\\\n" ::"n"(N) : "memory"); }\\n\\nDEVI uint32_t cvt_e4m3x2(uint32_t v) {\\n  uint32_t r;\\n  uint16_t h = static_cast<uint16_t>(v & 0xffffu);\\n  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\\\\n" : "=r"(r) : "h"(h));\\n  return r;\\n}\\n\\nDEVI uint32_t pack_h2(float lo, float hi) {\\n  __half2 h = __floats2half2_rn(lo, hi);\\n  return *reinterpret_cast<uint32_t*>(&h);\\n}\\n\\nDEVI float ex2(float x) {\\n  float y;\\n  asm("ex2.approx.ftz.f32 %0, %1;\\\\n" : "=f"(y) : "f"(x));\\n  return y;\\n}\\n\\nDEVI void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {\\n  asm volatile(\\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "\\n      "{%0,%1,%2,%3};\\\\n"\\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\\n}\\n\\nDEVI uint32_t cluster_addr(const void* p, int rank) {\\n  uint32_t r;\\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\\\\n" : "=r"(r) : "r"(smem_u32(p)), "r"(rank));\\n  return r;\\n}\\nDEVI float ld_cluster_f32(uint32_t a) {\\n  float v;\\n  asm volatile("ld.shared::cluster.f32 %0, [%1];\\\\n" : "=f"(v) : "r"(a) : "memory");\\n  return v;\\n}\\nDEVI float2 ld_cluster_f32x2(uint32_t a) {\\n  float2 v;\\n  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];\\\\n" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");\\n  return v;\\n}\\nDEVI void cluster_sync_all() {\\n  asm volatile("barrier.cluster.arrive.release.aligned;\\\\nbarrier.cluster.wait.acquire.aligned;\\\\n" ::: "memory");\\n}\\n\\n// 16-byte chunk swizzle of a 128-byte token row: conflict-free K (LDS.128) and V (LDS.64) reads.\\nDEVI int swz(int t) { return t & 3; }\\n\\nDEVI float warp_sum(float v) {\\n#pragma unroll\\n  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\\n  return v;\\n}\\n\\ntemplate <int L>\\nstruct WarpState {\\n  static constexpr int MTW = (L == 1) ? 1 : 2;  // 16-row m-tiles per warp\\n  uint32_t qa[MTW][4][4];                        // Q A-fragments, 4 k-steps of 16 dims\\n  float o[MTW][8][4];                            // O accumulators, 8 n-tiles of 8 dims\\n  float m[MTW][2];                               // running max (log2 domain, scaled)\\n  float l[MTW][2];                               // running sum (thread-partial)\\n};\\n\\n// One tile of keys for one warp: tokens tb .. tb + 8*NTC - 1 of the smem tile.\\n// Rows of m-tile mt: (L==1) row g = query head g (rows 8..15 padding);\\n//                    (L==8) row g -> token 2*mta, row g+8 -> token 2*mta+1 (query head g).\\ntemplate <int L, int NTC>\\nDEVI void compute_tile(WarpState<L>& st, const uint8_t* Ks, const uint8_t* Vs, int tb, int h, int wh,\\n                       int kp0, int k_hi, int seq, int window, bool need_mask, float scale_log2, int nq, bool same_query) {\\n  constexpr int MTW = WarpState<L>::MTW;\\n  constexpr int KK = NTC / 2;\\n  const int lane = threadIdx.x & 31;\\n  const int g = lane >> 2, q4 = lane & 3;\\n\\n  float s[MTW][NTC][4];\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt)\\n#pragma unroll\\n    for (int n = 0; n < NTC; ++n)\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) s[mt][n][e] = 0.f;\\n\\n  // S = Q K^T. B fragment (K^T, k = dims, n = tokens): thread holds token g of the n-tile,\\n  // dims 16*q4 + 4*ks .. +3 for k-step ks (a dim permutation shared with Q\\\'s fragments).\\n#pragma unroll\\n  for (int n = 0; n < NTC; ++n) {\\n    const int t = tb + 8 * n + g;\\n    const uint4 kw = *reinterpret_cast<const uint4*>(Ks + t * ROWB + (((4 * h + q4) ^ swz(t)) << 4));\\n    const uint32_t w[4] = {kw.x, kw.y, kw.z, kw.w};\\n#pragma unroll\\n    for (int ks = 0; ks < 4; ++ks) {\\n      const uint32_t b0 = cvt_e4m3x2(w[ks]);\\n      const uint32_t b1 = cvt_e4m3x2(w[ks] >> 16);\\n#pragma unroll\\n      for (int mt = 0; mt < MTW; ++mt) mma16816(s[mt][n], st.qa[mt][ks], b0, b1);\\n    }\\n  }\\n\\n  if (need_mask) {\\n#pragma unroll\\n    for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n      for (int e = 0; e < 4; ++e) {\\n        const int j = same_query ? 0 : g;\\n        const int qp = seq + j;\\n        const int lo = window ? (qp - window + 1) : -2147483647;\\n        const int hi = min(k_hi, qp + 1);\\n#pragma unroll\\n        for (int n = 0; n < NTC; ++n) {\\n          const int kp = kp0 + tb + 8 * n + 2 * q4 + (e & 1);\\n          if (kp < lo || kp >= hi || j >= nq) s[mt][n][e] = -INFINITY;\\n        }\\n      }\\n    }\\n  }\\n\\n  // Online softmax per row (rows g and g+8 of each m-tile; L==1 has only row g).\\n  uint32_t pa[MTW][KK][4];\\n#pragma unroll\\n  for (int mt = 0; mt < MTW; ++mt) {\\n#pragma unroll\\n    for (int rh = 0; rh < 2; ++rh) {\\n      if (L == 1 && rh == 1) continue;\\n      float mx = -INFINITY;\\n#pragma unroll\\n      for (int n = 0; n < NTC; ++n) mx = fmaxf(mx, fmaxf(s[mt][n][2 * rh], s[mt][n][2 * rh + 1]));\\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));\\n      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));\\n      const float m_new = fmaxf(st.m[mt][rh], mx * scale_log2);\\n      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;\\n      const float corr = ex2(st.m[mt][rh] - m_use);\\n      st.m[mt][rh] = m_new;\\n      st.l[mt][rh] *= corr;\\n#pragma unroll\\n      for (int jn = 0; jn < 8; ++jn) {\\n        st.o[mt][jn][2 * rh] *= corr;\\n        st.o[mt][jn][2 * rh + 1] *= corr;\\n      }\\n      float sum = 0.f;\\n#pragma unroll\\n      for (int n = 0; n < NTC; ++n) {\\n        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));\\n        const float p1 = ex2(fmaf(s[mt][n][2 * rh + 1], scale_log2, -m_use));\\n        s[mt][n][2 * rh] = p0;\\n        s[mt][n][2 * rh + 1] = p1;\\n        sum += p0 + p1;\\n      }\\n      st.l[mt][rh] += sum;\\n    }\\n#pragma unroll\\n    for (int kk = 0; kk < KK; ++kk) {\\n      pa[mt][kk][0] = pack_h2(s[mt][2 * kk][0], s[mt][2 * kk][1]);\\n      pa[mt][kk][2] = pack_h2(s[mt][2 * kk + 1][0], s[mt][2 * kk + 1][1]);\\n      if (L == 1) {\\n        pa[mt][kk][1] = 0u;\\n        pa[mt][kk][3] = 0u;\\n      } else {\\n        pa[mt][kk][1] = pack_h2(s[mt][2 * kk][2], s[mt][2 * kk][3]);\\n        pa[mt][kk][3] = pack_h2(s[mt][2 * kk + 1][2], s[mt][2 * kk + 1][3]);\\n      }\\n    }\\n  }\\n\\n  // O += P V. B fragment (V, k = tokens, n = dims): n-tile jn column g is dim 8*g + jn, so a\\n  // thread reads 8 contiguous bytes (dims 8g..8g+7) of four tokens and byte-transposes them.\\n#pragma unroll\\n  for (int kk = 0; kk < KK; ++kk) {\\n    const int t0 = tb + 16 * kk + 2 * q4;\\n    const int coff = ((4 * h + (g >> 1))) ;\\n    const uint2 va = *reinterpret_cast<const uint2*>(Vs + t0 * ROWB + ((coff ^ swz(t0)) << 4) + ((g & 1) << 3));\\n    const uint2 vb = *reinterpret_cast<const uint2*>(Vs + (t0 + 1) * ROWB + ((coff ^ swz(t0 + 1)) << 4) + ((g & 1) << 3));\\n    const uint2 vc = *reinterpret_cast<const uint2*>(Vs + (t0 + 8) * ROWB + ((coff ^ swz(t0 + 8)) << 4) + ((g & 1) << 3));\\n    const uint2 vd = *reinterpret_cast<const uint2*>(Vs + (t0 + 9) * ROWB + ((coff ^ swz(t0 + 9)) << 4) + ((g & 1) << 3));\\n#pragma unroll\\n    for (int jj = 0; jj < 2; ++jj) {\\n      const uint32_t a = jj ? va.y : va.x, b = jj ? vb.y : vb.x;\\n      const uint32_t c = jj ? vc.y : vc.x, d = jj ? vd.y : vd.x;\\n      const uint32_t ab_lo = __byte_perm(a, b, 0x5140), ab_hi = __byte_perm(a, b, 0x7362);\\n      const uint32_t cd_lo = __byte_perm(c, d, 0x5140), cd_hi = __byte_perm(c, d, 0x7362);\\n      const uint32_t b0[4] = {cvt_e4m3x2(ab_lo), cvt_e4m3x2(ab_lo >> 16), cvt_e4m3x2(ab_hi), cvt_e4m3x2(ab_hi >> 16)};\\n      const uint32_t b1[4] = {cvt_e4m3x2(cd_lo), cvt_e4m3x2(cd_lo >> 16), cvt_e4m3x2(cd_hi), cvt_e4m3x2(cd_hi >> 16)};\\n#pragma unroll\\n      for (int x = 0; x < 4; ++x)\\n#pragma unroll\\n        for (int mt = 0; mt < MTW; ++mt) mma16816(st.o[mt][4 * jj + x], pa[mt][kk], b0[x], b1[x]);\\n    }\\n  }\\n}\\n\\nstruct WP {\\n const __nv_bfloat16* qkv; const int64_t *pos,*loc; const int *pt,*seq;\\n uint8_t *kc,*vc; const float *ks,*vs,*freq; __nv_bfloat16* out;\\n float *parts; int* counters;\\n int L,HQ,HK,stride,pages,ps,window,ns,proj_splits,tokens; float qscale,smscale;\\n};\\n// Projection partials are reduced and rounded to BF16 at the same boundary as stock.\\nDEVI float2 qkv_pair(const WP& p, int token, int column) {\\n if(!p.proj_splits) return __bfloat1622float2(*(const __nv_bfloat162*)(p.qkv+(int64_t)token*p.stride+column));\\n const float* raw=(const float*)p.qkv;\\n float2 sum=make_float2(0,0);\\n for(int s=0;s<p.proj_splits;s++) {\\n   const float2 x=*(const float2*)(raw+((int64_t)s*p.tokens+token)*p.stride+column);\\n   sum.x+=x.x;sum.y+=x.y;\\n }\\n return __bfloat1622float2(__floats2bfloat162_rn(sum.x,sum.y));\\n}\\ntemplate<int L, bool SHARE=false>\\n__global__ __launch_bounds__(128,2) void window_mma(WP p) {\\n constexpr int SZ=256*64, STAGES=2;\\n extern __shared__ __align__(128) uint8_t shared[];\\n uint8_t* K=shared; uint8_t* V=K+STAGES*SZ;\\n uint8_t* NK=V+STAGES*SZ; uint8_t* NV=NK+16*64;\\n __half* Q=(__half*)(NV+16*64);\\n const int tid=threadIdx.x, warp=tid/32, lane=tid%32, g=lane/4, q4=lane%4;\\n const int split=blockIdx.x,h=blockIdx.y,b=blockIdx.z,hk=SHARE?h:h/8;\\n constexpr int R=SHARE?8:L;\\n const int groups=SHARE?p.HK:p.HQ;\\n const int seq=p.seq[b], lo=max(0,seq-p.window+1), count=seq-lo;\\n const int chunk=((count+p.ns-1)/p.ns+63)&~63;\\n const int begin=min(seq,lo+split*chunk), end=min(seq,begin+chunk);\\n const int tiles=(end-begin+255)/256;\\n auto issue=[&](int tile,int slot) {\\n#pragma unroll\\n   for(int i=0;i<8;i++) {\\n     const int ix=tid+i*128, t=ix/4, col=ix%4, key=begin+tile*256+t;\\n     const bool valid=tile<tiles && key<end;\\n     const int row=valid ? p.pt[(int64_t)b*p.pages+key/p.ps]*p.ps+key%p.ps : 0;\\n     const int off=slot*SZ+t*64+((col^swz(t))*16);\\n     const int64_t src=(int64_t)row*p.HK*64+hk*64+col*16;\\n     cp_async16(smem_u32(K+off),p.kc+src,valid?16:0);\\n     cp_async16(smem_u32(V+off),p.vc+src,valid?16:0);\\n   }\\n   cp_commit();\\n };\\n issue(0,0);\\n const float ks=*p.ks, vs=*p.vs;\\n#pragma unroll\\n for(int i=0;i<2;i++) {\\n   const int t=warp+4*i;\\n   float qx=0,qy=0,kx=0,ky=0,vx=0,vy=0;\\n   float cs=1,sn=0;\\n   if(t<R) {\\n     const int qt=SHARE?0:t, qh=SHARE?h*8+t:h;\\n     const auto qr=qkv_pair(p,b*L+qt,qh*64+2*lane);\\n     const float qn=rsqrtf(warp_sum(qr.x*qr.x+qr.y*qr.y)/64+1.e-5f)*p.qscale;\\n     if(p.freq) sincosf((float)p.pos[b*L+qt]*p.freq[lane],&sn,&cs);\\n     qx=__bfloat162float(__float2bfloat16_rn((qr.x*cs-qr.y*sn)*qn));\\n     qy=__bfloat162float(__float2bfloat16_rn((qr.x*sn+qr.y*cs)*qn));\\n   }\\n   if(t<L) {\\n     const auto kr=qkv_pair(p,b*L+t,(p.HQ+hk)*64+2*lane);\\n     const auto vr=qkv_pair(p,b*L+t,(p.HQ+p.HK+hk)*64+2*lane);\\n     const float kn=rsqrtf(warp_sum(kr.x*kr.x+kr.y*kr.y)/64+1.e-5f);\\n     kx=(kr.x*cs-kr.y*sn)*kn/ks; ky=(kr.x*sn+kr.y*cs)*kn/ks;\\n     vx=vr.x/vs; vy=vr.y/vs;\\n   }\\n   *(__half2*)(Q+t*64+2*lane)=__floats2half2_rn(qx,qy);\\n   const auto k8=__nv_cvt_float2_to_fp8x2(make_float2(kx,ky),__NV_SATFINITE,__NV_E4M3);\\n   const auto v8=__nv_cvt_float2_to_fp8x2(make_float2(vx,vy),__NV_SATFINITE,__NV_E4M3);\\n   const int off=t*64+(((lane/8)^swz(t))*16)+(lane%8)*2;\\n   *(unsigned short*)(NK+off)=k8; *(unsigned short*)(NV+off)=v8;\\n   // Pad up to 16 rows for the short new-token MMA.\\n   const int off2=(t+8)*64+(((lane/8)^swz(t+8))*16)+(lane%8)*2;\\n   *(unsigned short*)(NK+off2)=0; *(unsigned short*)(NV+off2)=0;\\n   if(t<L && split==0 && (SHARE || h%8==0)) {\\n     const int64_t dst=p.loc[b*L+t]*p.HK*64+hk*64+2*lane;\\n     *(unsigned short*)(p.kc+dst)=k8; *(unsigned short*)(p.vc+dst)=v8;\\n   }\\n }\\n __syncthreads();\\n WarpState<1> st;\\n#pragma unroll\\n for(int k=0;k<4;k++) {\\n   const uint2 x=*(const uint2*)(Q+g*64+16*q4+4*k);\\n   st.qa[0][k][0]=x.x; st.qa[0][k][2]=x.y;\\n   st.qa[0][k][1]=st.qa[0][k][3]=0;\\n }\\n st.m[0][0]=st.m[0][1]=-INFINITY; st.l[0][0]=st.l[0][1]=0;\\n#pragma unroll\\n for(int n=0;n<8;n++)\\n#pragma unroll\\n for(int e=0;e<4;e++) st.o[0][n][e]=0;\\n for(int tile=0;tile<tiles;tile++) {\\n   cp_wait<0>(); __syncthreads();\\n   if(tile+1<tiles) issue(tile+1,(tile+1)%2);\\n   const int kp=begin+tile*256;\\n   compute_tile<1,8>(st,K+(tile%2)*SZ,V+(tile%2)*SZ,64*warp,0,0,kp,end,seq,p.window,\\n                     (kp<seq+L-p.window || kp+256>end),p.smscale*ks*1.4426950408889634f,R,SHARE);\\n }\\n cp_wait<0>(); __syncthreads();\\n if(split==p.ns-1 && warp==0)\\n   compute_tile<1,2>(st,NK,NV,0,0,0,seq,seq+L,seq,p.window,true,p.smscale*ks*1.4426950408889634f,R,SHARE);\\n float den=st.l[0][0]; den+=__shfl_xor_sync(0xffffffff,den,1); den+=__shfl_xor_sync(0xffffffff,den,2);\\n // The pipeline is drained; reuse it for the four warp partials.\\n __syncthreads();\\n float* sp=(float*)shared;\\n float* dst=sp+(warp*8+g)*68;\\n float4* d4=(float4*)(dst+16*q4);\\n d4[0]=make_float4(st.o[0][0][0],st.o[0][1][0],st.o[0][2][0],st.o[0][3][0]);\\n d4[1]=make_float4(st.o[0][4][0],st.o[0][5][0],st.o[0][6][0],st.o[0][7][0]);\\n d4[2]=make_float4(st.o[0][0][1],st.o[0][1][1],st.o[0][2][1],st.o[0][3][1]);\\n d4[3]=make_float4(st.o[0][4][1],st.o[0][5][1],st.o[0][6][1],st.o[0][7][1]);\\n if(q4==0) {dst[64]=st.m[0][0];dst[65]=den;}\\n __syncthreads();\\n float results[2][2];\\n#pragma unroll\\n for(int i=0;i<2;i++) {\\n   const int j=warp+4*i;\\n   if(j<R) {\\n     float mx=-INFINITY;\\n#pragma unroll\\n     for(int w=0;w<4;w++) mx=fmaxf(mx,sp[(w*8+j)*68+64]);\\n     float total=0,x=0,y=0;\\n#pragma unroll\\n     for(int w=0;w<4;w++) {\\n       const float* pp=sp+(w*8+j)*68;\\n       const float ww=ex2(pp[64]-(mx == -INFINITY ? 0.f : mx));\\n       const float2 oo=*(const float2*)(pp+2*lane);\\n       x=fmaf(ww,oo.x,x);y=fmaf(ww,oo.y,y);total=fmaf(ww,pp[65],total);\\n     }\\n     if(p.ns==1) {results[i][0]=x*vs/total;results[i][1]=y*vs/total;}\\n     else {\\n       float* pp=p.parts+(((int64_t)b*groups+h)*p.ns+split)*R*68+j*68;\\n       *(float2*)(pp+2*lane)=make_float2(x,y);\\n       if(lane==0) *(float2*)(pp+64)=make_float2(mx,total);\\n     }\\n   }\\n }\\n if(p.ns>1) {\\n   __threadfence(); __syncthreads();\\n   __shared__ int last;\\n   if(tid==0) last=(atomicAdd(p.counters+b*groups+h,1)==p.ns-1);\\n   __syncthreads();\\n   if(!last) return;\\n#pragma unroll\\n   for(int i=0;i<2;i++) {\\n     const int j=warp+4*i;\\n     if(j<R) {\\n       const float* base=p.parts+((int64_t)b*groups+h)*p.ns*R*68+j*68;\\n       float2 ml[16], ov[16];\\n#pragma unroll\\n       for(int s=0;s<16;s++) if(s<p.ns) {\\n         const float* pp=base+s*R*68;\\n         ml[s]=__ldcg((const float2*)(pp+64));\\n         ov[s]=__ldcg((const float2*)(pp+2*lane));\\n       }\\n       float mx=-INFINITY;\\n#pragma unroll\\n       for(int s=0;s<16;s++) if(s<p.ns) mx=fmaxf(mx,ml[s].x);\\n       float total=0,x=0,y=0;\\n#pragma unroll\\n       for(int s=0;s<16;s++) if(s<p.ns) {\\n         const float ww=ex2(ml[s].x-mx);\\n         x=fmaf(ww,ov[s].x,x);y=fmaf(ww,ov[s].y,y);total=fmaf(ww,ml[s].y,total);\\n       }\\n       results[i][0]=x*vs/total;results[i][1]=y*vs/total;\\n     }\\n   }\\n }\\n#pragma unroll\\n for(int i=0;i<2;i++) {\\n   const int j=warp+4*i;\\n   if(j<R) {\\n     const float x=results[i][0],y=results[i][1];\\n     const float norm=rsqrtf(warp_sum(x*x+y*y)/64+1.e-5f);\\n     const int qt=SHARE?0:j,qh=SHARE?h*8+j:h;\\n     const auto gate=qkv_pair(p,b*L+qt,(p.HQ+2*p.HK+qh)*64+2*lane);\\n     *(__nv_bfloat162*)(p.out+(int64_t)(b*L+qt)*p.HQ*64+qh*64+2*lane)=\\n        __floats2bfloat162_rn(x*norm/(1+__expf(-gate.x)),y*norm/(1+__expf(-gate.y)));\\n   }\\n }\\n __syncthreads(); if(tid==0 && p.ns>1) atomicExch(p.counters+b*groups+h,0);\\n}\\n}\\nvoid window_forward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,\\n int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t out,int64_t parts,int64_t counters,\\n int64_t B,int64_t L,int64_t HQ,int64_t HK,int64_t stride,int64_t pages,int64_t ps,int64_t window,\\n int64_t ns,int64_t proj_splits,double qscale,double smscale,int64_t stream) {\\n WP p{(const __nv_bfloat16*)qkv,(const int64_t*)pos,(const int64_t*)loc,(const int*)pt,(const int*)seq,\\n (uint8_t*)kc,(uint8_t*)vc,(const float*)ks,(const float*)vs,(const float*)freq,(__nv_bfloat16*)out,\\n (float*)parts,(int*)counters,(int)L,(int)HQ,(int)HK,(int)stride,(int)pages,(int)ps,(int)window,(int)ns,(int)proj_splits,(int)(B*L),(float)qscale,(float)smscale};\\n constexpr int smem=2*2*256*64+2*16*64+8*64*2;\\n cudaError_t err;\\n if(L==1 && B>1) {cudaFuncSetAttribute(window_mma<1,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\\n   window_mma<1,true><<<dim3(ns,HK,B),128,smem,(cudaStream_t)stream>>>(p);}\\n else if(L==1) {cudaFuncSetAttribute(window_mma<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\\n   window_mma<1><<<dim3(ns,HQ,B),128,smem,(cudaStream_t)stream>>>(p);}\\n else {cudaFuncSetAttribute(window_mma<8>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\\n   window_mma<8><<<dim3(ns,HQ,B),128,smem,(cudaStream_t)stream>>>(p);}\\n\\n err=cudaGetLastError();if(err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));\\n}\\n\'\nWINDOW_CPP = \'\\n#include <cstdint>\\nvoid window_forward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,\\n int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t out,int64_t parts,int64_t counters,\\n int64_t B,int64_t L,int64_t HQ,int64_t HK,int64_t stride,int64_t pages,int64_t ps,int64_t window,\\n int64_t ns,int64_t proj_splits,double qscale,double smscale,int64_t stream);\\n\'\n_WINDOW_EXT = None\n\n\ndef _window_ext():\n    global _WINDOW_EXT\n    if _WINDOW_EXT is None:\n        major, minor = torch.cuda.get_device_capability()\n        arch = f\'{major}{minor}\'\n        _WINDOW_EXT = load_inline(\n            name=f\'muse_window_mma_sm{arch}\', cpp_sources=WINDOW_CPP, cuda_sources=WINDOW_CUDA,\n            functions=[\'window_forward\'],\n            extra_cuda_cflags=[\'-O3\', f\'-gencode=arch=compute_{arch},code=sm_{arch}\', \'-lineinfo\'],\n            verbose=False)\n    return _WINDOW_EXT\n\n\ndef sliding_attention(raw, data, splits=0):\n    """Layer adapter: BF16 projected QKV plus the attention task\'s cache/config fields."""\n    cfg = data[\'config\']\n    B, L, HQ, HK = cfg[\'batch\'], cfg[\'q_len\'], cfg[\'num_q_heads\'], cfg[\'num_kv_heads\']\n    assert cfg[\'window\'] and cfg[\'window\'] <= 2048\n    out = torch.empty((B, L, HQ*64), dtype=torch.bfloat16, device=raw.device)\n    freq = _inv_freq(cfg[\'rope_theta\'], raw.device) if cfg[\'rope_theta\'] else None\n    ns = 8 if B == 1 else (min(16, max(1, 132 // (B*HK))) if L == 1 else 1)\n    parts = torch.empty((B*HQ*ns*L*68 if ns>1 else 1,),dtype=torch.float32,device=raw.device)\n    counters = _window_counters(B*HQ, raw.device)\n    _window_ext().window_forward(\n        raw.data_ptr(), data[\'positions\'].data_ptr(), data[\'cache_loc\'].data_ptr(),\n        data[\'page_table\'].data_ptr(), data[\'seq_lens\'].data_ptr(),\n        data[\'k_cache\'].data_ptr(), data[\'v_cache\'].data_ptr(),\n        data[\'k_scale\'].data_ptr(), data[\'v_scale\'].data_ptr(),\n        freq.data_ptr() if freq is not None else 0, out.data_ptr(), parts.data_ptr(), counters.data_ptr(),\n        B, L, HQ, HK, raw.shape[-1], data[\'page_table\'].shape[1], cfg[\'page_size\'],\n        cfg[\'window\'], ns, splits, cfg[\'qk_scale_factor\'], cfg[\'sm_scale\'],\n        torch.cuda.current_stream(raw.device).cuda_stream)\n    return out\n\n\n\n_WINDOW_COUNTERS = {}\ndef _window_counters(n, device):\n    key = str(device)\n    if key not in _WINDOW_COUNTERS:\n        _WINDOW_COUNTERS[key] = torch.zeros(1 << 16, dtype=torch.int32, device=device)\n    assert n <= (1 << 16)\n    return _WINDOW_COUNTERS[key]\n\n\ndef attention_sliding(raw, data):\n    splits = 0\n    if isinstance(raw, tuple):\n        partial, count = raw\n        assert partial.is_contiguous()\n        if partial.dtype == torch.float32:\n            splits = count\n        else:\n            assert partial.dtype == torch.bfloat16 and count == 1\n        raw = partial\n    cfg = dict(data[\'config\'], num_q_heads=16, num_kv_heads=2)\n    metadata = dict(data, config=cfg)\n    return sliding_attention(raw, metadata, splits=splits).view(cfg[\'batch\']*cfg[\'q_len\'], 1024)\n\n\nif torch.cuda.is_available():\n    _window_counters(1 << 16, torch.device(\'cuda\', torch.cuda.current_device()))\n\n\ndef attention_verify1_sliding(qkv, data):\n    cfg = data[\'config\']\n    if (cfg[\'batch\'], cfg[\'q_len\'], cfg[\'window\']) != (1, 8, 2048):\n        raise ValueError(\'verify1 sliding pick accepts only batch=1, q_len=8, window=2048\')\n    return attention_sliding(qkv, data)\n\n@triton.jit\ndef _pdl_start(USE: tl.constexpr):\n    if USE:\n        tl.extra.cuda.gdc_launch_dependents()\n\n\n@triton.jit\ndef _pdl_wait(USE: tl.constexpr):\n    if USE:\n        tl.extra.cuda.gdc_wait()\n\n\ndef _launch(kernel, grid, *args, use_pdl=None, **kw):\n    enabled = PDL if use_pdl is None else bool(use_pdl)\n    if enabled:\n        kw["launch_pdl"] = True\n    return kernel[grid](*args, USE_PDL=enabled, **kw)\n\n\n@triton.jit\ndef _qk_prep_kernel(P, S: tl.constexpr, ps_stride, POS, LOC, KC, VC, KSCALE, VSCALE, INV_FREQ,\n                    QOUT, GATE, q_mult, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,\n                    ROPE: tl.constexpr, Q_FP8: tl.constexpr, USE_PDL: tl.constexpr):\n    t = tl.program_id(0)\n    _pdl_start(USE_PDL)\n    cache_row=tl.load(LOC+t)\n    kscale=tl.load(KSCALE);vscale=tl.load(VSCALE)\n    if ROPE:\n        half=tl.arange(0,D//2)\n        angle=tl.load(POS+t).to(tl.float32)*tl.load(INV_FREQ+half)\n        rope_cos=tl.cos(angle)[None,:];rope_sin=tl.sin(angle)[None,:]\n    _pdl_wait(USE_PDL)\n    W: tl.constexpr = (2 * HQ + 2 * HKV) * D\n    base = P + t * W\n    hq = tl.arange(0, HQ)\n    hk = tl.arange(0, HKV)\n    d = tl.arange(0, D)\n    qo = hq[:, None] * D + d[None, :]\n    ko = HQ * D + hk[:, None] * D + d[None, :]\n    vo = (HQ + HKV) * D + hk[:, None] * D + d[None, :]\n    go = (HQ + 2 * HKV) * D + qo\n    q = tl.zeros((HQ, D), tl.float32)\n    k = tl.zeros((HKV, D), tl.float32)\n    v = tl.zeros((HKV, D), tl.float32)\n    g = tl.zeros((HQ, D), tl.float32)\n    for s in tl.static_range(S):\n        q += tl.load(base + s * ps_stride + qo).to(tl.float32)\n        k += tl.load(base + s * ps_stride + ko).to(tl.float32)\n        v += tl.load(base + s * ps_stride + vo).to(tl.float32)\n        g += tl.load(base + s * ps_stride + go).to(tl.float32)\n    tl.store(GATE + t * (HQ * D) + qo, g)\n    q = q * tl.rsqrt(tl.sum(q * q, 1) / D + eps)[:, None]\n    k = k * tl.rsqrt(tl.sum(k * k, 1) / D + eps)[:, None]\n    if ROPE:\n        c=rope_cos;sn=rope_sin\n        qe, qd = tl.split(tl.reshape(q, (HQ, D // 2, 2)))\n        q = tl.reshape(tl.join(qe * c - qd * sn, qe * sn + qd * c), (HQ, D))\n        ke, kd = tl.split(tl.reshape(k, (HKV, D // 2, 2)))\n        k = tl.reshape(tl.join(ke * c - kd * sn, ke * sn + kd * c), (HKV, D))\n    row = cache_row\n    kv_off = row * (HKV * D) + hk[:, None] * D + d[None, :]\n    tl.store(KC + kv_off, (k / kscale).to(tl.float8e4nv))\n    tl.store(VC + kv_off, (v / vscale).to(tl.float8e4nv))\n    if Q_FP8:\n        tl.store(QOUT + t * (HQ * D) + qo, q.to(tl.float8e4nv))\n    else:\n        tl.store(QOUT + t * (HQ * D) + qo, (q * q_mult).to(tl.bfloat16))\n\n\ndef qk_prep(qkv, data, state, q_fp8: bool):\n    P, S = qkv\n    cfg = data["config"]\n    tokens = cfg["tokens"]\n    q = torch.empty((tokens, HQ, HD), device=P.device, dtype=FP8 if q_fp8 else torch.bfloat16)\n    gate = torch.empty((tokens, Q_SIZE), device=P.device, dtype=torch.float32)\n    _launch(_qk_prep_kernel, (tokens,),\n            P, S, P.stride(0), data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"],\n            data["k_scale"], data["v_scale"], state["inv_freq"], q, gate, cfg["qk_scale_factor"], RMS_EPS,\n            HQ=HQ, HKV=HKV, D=HD, ROPE=bool(cfg["rope_theta"]), Q_FP8=q_fp8, num_warps=4)\n    return q, gate\n\n\n@triton.jit\ndef _norm_gate_kernel(X, GATE, OUT, eps, HQ: tl.constexpr, D: tl.constexpr, USE_PDL: tl.constexpr):\n    t = tl.program_id(0)\n    _pdl_start(USE_PDL)\n    h = tl.arange(0, HQ)\n    d = tl.arange(0, D)\n    off = h[:, None] * D + d[None, :]\n    _pdl_wait(USE_PDL)\n    o = tl.load(X + t * (HQ * D) + off).to(tl.float32)\n    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]\n    g = tl.load(GATE + t * (HQ * D) + off)\n    tl.store(OUT + t * (HQ * D) + off, (o * tl.sigmoid(g)).to(tl.bfloat16))\n\n\ndef _attention_fa3_public(q, gate, data, state):\n    from sgl_kernel.flash_attn import flash_attn_with_kvcache\n    cfg = data[\'config\']\n    batch, L, tokens = (cfg[\'batch\'], cfg[\'q_len\'], cfg[\'tokens\'])\n    window = cfg[\'window\']\n    descale = {}\n    if cfg[\'kv_scale\'] != 1.0:\n        descale = dict(k_descale=data[\'k_scale\'].reshape(1, 1).expand(batch, HKV), v_descale=data[\'v_scale\'].reshape(1, 1).expand(batch, HKV))\n    out = flash_attn_with_kvcache(q=q, k_cache=data[\'k_cache\'], v_cache=data[\'v_cache\'], page_table=data[\'page_table\'], cache_seqlens=data[\'cache_seqlens\'], cu_seqlens_q=data[\'cu_seqlens_q\'], max_seqlen_q=L, softmax_scale=cfg[\'sm_scale\'] * cfg[\'qk_scale_factor\'], causal=True, window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data[\'scheduler_metadata\'], **descale)\n    if isinstance(out, tuple):\n        out = out[0]\n    o = torch.empty((tokens, Q_SIZE), device=q.device, dtype=torch.bfloat16)\n    _launch(_norm_gate_kernel, (tokens,), out, gate, o, RMS_EPS, HQ=HQ, D=HD, num_warps=4)\n    return o\n\n\n@triton.jit\ndef _best7_old_qk_prep_kernel(P, S: tl.constexpr, ps_stride, POS, LOC, KC, VC, KSCALE, VSCALE, INV_FREQ, QOUT, GATE, q_mult, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr, Q_FP8: tl.constexpr, USE_PDL: tl.constexpr):\n    t = tl.program_id(0)\n    _pdl_start(USE_PDL)\n    _pdl_wait(USE_PDL)\n    W: tl.constexpr = (2 * HQ + 2 * HKV) * D\n    base = P + t * W\n    hq = tl.arange(0, HQ)\n    hk = tl.arange(0, HKV)\n    d = tl.arange(0, D)\n    qo = hq[:, None] * D + d[None, :]\n    ko = HQ * D + hk[:, None] * D + d[None, :]\n    vo = (HQ + HKV) * D + hk[:, None] * D + d[None, :]\n    go = (HQ + 2 * HKV) * D + qo\n    q = tl.zeros((HQ, D), tl.float32)\n    k = tl.zeros((HKV, D), tl.float32)\n    v = tl.zeros((HKV, D), tl.float32)\n    g = tl.zeros((HQ, D), tl.float32)\n    for s in tl.static_range(S):\n        q += tl.load(base + s * ps_stride + qo).to(tl.float32)\n        k += tl.load(base + s * ps_stride + ko).to(tl.float32)\n        v += tl.load(base + s * ps_stride + vo).to(tl.float32)\n        g += tl.load(base + s * ps_stride + go).to(tl.float32)\n    tl.store(GATE + t * (HQ * D) + qo, g)\n    q = q * tl.rsqrt(tl.sum(q * q, 1) / D + eps)[:, None]\n    k = k * tl.rsqrt(tl.sum(k * k, 1) / D + eps)[:, None]\n    if ROPE:\n        pos = tl.load(POS + t).to(tl.float32)\n        half = tl.arange(0, D // 2)\n        ang = pos * tl.load(INV_FREQ + half)\n        c = tl.cos(ang)[None, :]\n        sn = tl.sin(ang)[None, :]\n        qe, qd = tl.split(tl.reshape(q, (HQ, D // 2, 2)))\n        q = tl.reshape(tl.join(qe * c - qd * sn, qe * sn + qd * c), (HQ, D))\n        ke, kd = tl.split(tl.reshape(k, (HKV, D // 2, 2)))\n        k = tl.reshape(tl.join(ke * c - kd * sn, ke * sn + kd * c), (HKV, D))\n    row = tl.load(LOC + t)\n    kv_off = row * (HKV * D) + hk[:, None] * D + d[None, :]\n    tl.store(KC + kv_off, (k / tl.load(KSCALE)).to(tl.float8e4nv))\n    tl.store(VC + kv_off, (v / tl.load(VSCALE)).to(tl.float8e4nv))\n    if Q_FP8:\n        tl.store(QOUT + t * (HQ * D) + qo, q.to(tl.float8e4nv))\n    else:\n        tl.store(QOUT + t * (HQ * D) + qo, (q * q_mult).to(tl.bfloat16))\n\n\ndef _best7_old_qk_prep(qkv, data, state, q_fp8: bool):\n    P, S = qkv\n    cfg = data[\'config\']\n    tokens = cfg[\'tokens\']\n    q = torch.empty((tokens, HQ, HD), device=P.device, dtype=FP8 if q_fp8 else torch.bfloat16)\n    gate = torch.empty((tokens, Q_SIZE), device=P.device, dtype=torch.float32)\n    _launch(_best7_old_qk_prep_kernel, (tokens,), P, S, P.stride(0), data[\'positions\'], data[\'cache_loc\'], data[\'k_cache\'], data[\'v_cache\'], data[\'k_scale\'], data[\'v_scale\'], state[\'inv_freq\'], q, gate, cfg[\'qk_scale_factor\'], RMS_EPS, HQ=HQ, HKV=HKV, D=HD, ROPE=bool(cfg[\'rope_theta\']), Q_FP8=q_fp8, num_warps=4)\n    return (q, gate)\n\n\n\ndef layer_fa3(qkv,data,state,glue=\'current\'):\n    # Both current and best5 glue converge on the exact public FA3 call.\n    q,gate=(qk_prep if glue==\'current\' else _best7_old_qk_prep)(qkv,data,state,q_fp8=True)\n    return _attention_fa3_public(q,gate,data,state)\n\n\ndef lab(deadline,weights):\n    lines=[\'META verify1 sliding: exact guarded adapter vs decode public FA3 incl qk_prep and norm_gate; no projection GEMV\',\n           \'META q_len=8 B=1 ctx=80000 window2048 page1; PDL=True control; BF16 and FP32 partial ABIs\']\n    if deadline-time.monotonic()<25:return lines+[\'SKIP budget<25s before compile\']\n    _window_ext()\n    data=generate_input(1,80000,\'tp8\',\'sliding\',931,q_len=8,page=1,dist=\'soft\')\n    data[\'config\'][\'tokens\']=8\n    theta=data[\'config\'][\'rope_theta\']\n    state={\'inv_freq\':1./(theta**(torch.arange(0,64,2,device=\'cuda\').float()/64.))}\n    flush=torch.empty(128<<20,dtype=torch.uint8,device=\'cuda\')\n    for count in (0,1,4):\n        if deadline-time.monotonic()<8:lines.append(f\'SKIP partials{count} budget<8s\');break\n        raw=data[\'qkv\'].view(8,2304)\n        if count:\n            raw=torch.randn((count,8,2304),device=\'cuda\',dtype=torch.float32)\n            data[\'qkv\']=raw.sum(0).to(torch.bfloat16).view(1,8,2304)\n            qkv=(raw,count)\n        else:qkv=(raw.unsqueeze(0),1)\n        first=attention_verify1_sliding(qkv,data);torch.cuda.synchronize()\n        expected=gate_prepare(data)\n        # Fresh values arrive through the same captured projection-buffer ABI.\n        for name in (\'single\',\'fa3_current\',\'fa3_best5\'):\n            if deadline-time.monotonic()<3:lines.append(\'SKIP paired tail budget<3s\');break\n            def call(d):\n                if name==\'single\':return attention_verify1_sliding(qkv,d)\n                return layer_fa3(qkv,d,state,glue=\'current\' if name==\'fa3_current\' else \'best5\')\n            us,out,g=timed(call,data,flush,rounds=7)\n            expected=gate_prepare(data)\n            ok,msg,err=gate_cached(data,out,expected)\n            # Modify values, then verify the captured graph used them.\n            saved=raw.clone()\n            raw.add_(.015625)\n            data[\'qkv\']=(raw.sum(0) if count else raw).to(torch.bfloat16).view(1,8,2304)\n            g.replay();torch.cuda.synchronize();fresh=gate_prepare(data)\n            fok,fmsg,ferr=gate_cached(data,out,fresh)\n            lines.append(f\'ROW partials={count} kernel={name} us={us:.2f} gate={"P" if ok and fok else "F"} err={err:.6f} fresh_err={ferr:.6f} graph=P\')\n            if not(ok and fok):lines.append(\'GATE_ERROR \'+msg+\'; \'+fmsg)\n            raw.copy_(saved)\n            data[\'qkv\']=(raw.sum(0) if count else raw).to(torch.bfloat16).view(1,8,2304)\n    return lines\n')]
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
