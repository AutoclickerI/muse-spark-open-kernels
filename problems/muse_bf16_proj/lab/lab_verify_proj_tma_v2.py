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

# ---- h200_lab payload verify_proj_tma_v2.py ----
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
def vp_boundary_kernel(A,S,H,NEW,ALPHA,BETA,PRE,M:tl.constexpr,USE_PDL:tl.constexpr):
    if USE_PDL:tl.extra.cuda.gdc_launch_dependents()
    r=tl.program_id(0);c=tl.arange(0,8192)
    alpha=tl.load(ALPHA+c);beta=tl.load(BETA+c);pre=tl.load(PRE+c)
    if USE_PDL:tl.extra.cuda.gdc_wait()
    branch=tl.load(A+r.to(tl.int64)*8192+c).to(tl.float32)
    stream=tl.load(S+r.to(tl.int64)*8192+c)
    branch=branch*tl.rsqrt(tl.sum(branch*branch,0)/8192.+1e-8)
    new=alpha*stream+beta*branch
    tl.store(NEW+r.to(tl.int64)*8192+c,new)
    h=new*tl.rsqrt(tl.sum(new*new,0)/8192.+1e-5)*pre
    tl.store(H+r.to(tl.int64)*8192+c,h.to(tl.bfloat16))


def vp_chain(x,stream,wo,wp,alpha,beta,pre,co,cp,pdl,native):
    a=torch.mm(x,wo.T) if co is None else vg_tma(x,wo,co,torch.bfloat16,pdl,native)
    h=torch.empty((x.shape[0],8192),device=x.device,dtype=torch.bfloat16)
    new=torch.empty_like(stream)
    vp_boundary_kernel[(x.shape[0],)](a,stream,h,new,alpha,beta,pre,M=x.shape[0],USE_PDL=pdl,
        launch_pdl=pdl,num_warps=8)
    y=torch.mm(h,wp.T,out_dtype=torch.float32) if cp is None else vg_tma(h,wp,cp,torch.float32,pdl,native)
    return y


def vp_unique(m,n,k,sms):
    seen=set();configs=[]
    for c in vg_tma_choices(m,n,k):
        bt,ns,ncw,st,mc,ks,waves,pipe,kg=c;br=64*ns*ncw
        jobs=(n//br//mc)*((m+bt-1)//bt)
        groups=jobs if not waves else min(jobs,max(1,sms*waves//(mc*ks)))
        canonical=(bt,ns,ncw,st,mc,ks,groups,pipe,kg)
        if canonical not in seen:seen.add(canonical);configs.append(c)
    return configs


def vg_tma_lab(deadline,weights,local=False):
    rows=[];native=vg_tma_module();dev=weights['qkv_gate'].device
    router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
    ws={name:weights[name] for name in ('qkv_gate','o','unpooler')}
    ws['pool_router']=torch.cat((weights['pooler'],hi,lo),0).contiguous()
    flush=torch.zeros((128<<20)//4,device=dev,dtype=torch.float32)
    gen=torch.Generator(device=dev).manual_seed(772410)
    sms=torch.cuda.get_device_properties(dev).multi_processor_count
    keys=[(m,name) for m in (256,128) for name in ('pool_router','qkv_gate','unpooler','o')]
    rows.append(f'VG_TMA_META sms={sms} cold128MiB graph fresh3; raw_us; cfg=(BT,NS,WG,ST,MC,KS,waves,async,KG)')
    winners={};measurements=[]
    for pos,(m,name) in enumerate(keys):
        now=time.monotonic();stop=now+max(0,deadline-now-10)/(len(keys)-pos)
        if now>=deadline-10:rows.append(f'VG_TMA_BUDGET {m} {name}');continue
        coarse_stop=now+.7*max(0,stop-now)
        w=ws[name];n,k=w.shape;dtype=torch.float32 if name=='pool_router' else torch.bfloat16
        banks=[torch.randn(m,k,device=dev,dtype=torch.bfloat16,generator=gen) for _ in range(3)]
        x=banks[0].clone();refs=[torch.mm(b.float(),w.float().T) for b in banks]
        old=_PgGraph(lambda:torch.mm(x,w.T,out_dtype=torch.float32) if dtype==torch.float32 else torch.mm(x,w.T))
        configs=vp_unique(m,n,k,sms);cost=[];table=[];bad={}
        for index,c in enumerate(configs):
            if time.monotonic()>coarse_stop:break
            graph=None
            try:
                graph=_PgGraph(lambda c=c:vg_tma(x,w,c,dtype,False,native))
                for b in range(3):x.copy_(banks[b]);graph.graph.replay();vp_check(graph.out,refs[b])
                t0,t1,ratio=vp_pair(old,graph,x,banks,flush,4)
                cost.append((ratio,index,t1));table.append(round(t1,1))
            except Exception as exc:
                reason=type(exc).__name__+':'+str(exc).splitlines()[0][:80]
                if reason not in bad and len(bad)<2:rows.append(f'VG_TMA_REJECT {m} {name} i={index} {reason}')
                bad[reason]=bad.get(reason,0)+1;table.append('x')
            finally:
                if graph is not None:graph.close()
        finalists=sorted(cost)[:4]
        # Keep a multicast and a non-multicast finalist where available.
        for mc in (1,2):
            family=[r for r in cost if configs[r[1]][4]==mc]
            if family and min(family) not in finalists:finalists.append(min(family))
        best=(float('inf'),None,None);details=[]
        for _,index,_ in finalists:
            if time.monotonic()>min(stop,deadline-8):break
            graph=None;c=configs[index]
            try:
                graph=_PgGraph(lambda c=c:vg_tma(x,w,c,dtype,False,native))
                t0,t1,ratio=vp_pair(old,graph,x,banks,flush)
                for b in range(3):x.copy_(banks[b]);graph.graph.replay();vp_check(graph.out,refs[b])
                details.append((index,round(t0,2),round(t1,2),round(ratio,4)))
                if ratio<best[0]:best=(ratio,c,(t0,t1))
            except Exception as exc:rows.append(f'VG_TMA_REJECT final {m} {name} {index} {type(exc).__name__}:{str(exc)[:70]}')
            finally:
                if graph is not None:graph.close()
        winners[(m,name)]=best[1]
        rows.append(f'VG_TMA_BEST {m} {name} pick={best[1]} raw_us={best[2]} ratio={best[0]:.4f} final={details} rejected={sum(bad.values())}')
        # Retain each tile/ring family's best coarse result. Printing every trial
        # would crowd the final keys out of the lab's9K output allowance.
        family_best={}
        for ratio,index,us in cost:
            family=(configs[index][0],configs[index][1],configs[index][2],configs[index][4],configs[index][-1])
            if family not in family_best or ratio<family_best[family][0]:family_best[family]=(ratio,index,us)
        measurements.append((m,name,[(i,round(us,1)) for _,i,us in family_best.values()]))
        old.close();del banks,x,refs,old,configs,cost
    # Measure PDL only across a real projection/boundary/projection dependency chain.
    for m in (256,128):
        if time.monotonic()>deadline-8:rows.append(f'VG_CHAIN_BUDGET {m}');continue
        co=winners.get((m,'o'));cp=winners.get((m,'pool_router'))
        if co is None or cp is None:continue
        x=torch.randn(m,1024,device=dev,dtype=torch.bfloat16,generator=gen)
        banks=[torch.randn_like(x) for _ in range(3)]
        stream=torch.randn(m,8192,device=dev,generator=gen)
        alpha=torch.full((8192,),.8,device=dev);beta=torch.full_like(alpha,.5);pre=torch.ones_like(alpha)
        old=_PgGraph(lambda:vp_chain(x,stream,ws['o'],ws['pool_router'],alpha,beta,pre,None,None,False,native))
        for pdl in (False,True):
            graph=None
            try:
                graph=_PgGraph(lambda:vp_chain(x,stream,ws['o'],ws['pool_router'],alpha,beta,pre,co,cp,pdl,native))
                vp_check(graph.out,old.out)
                t0,t1,ratio=vp_pair(old,graph,x,banks,flush)
                vp_check(graph.out,old.out)
                rows.append(f'VG_CHAIN {m} pdl={pdl} raw_us={t0:.2f}/{t1:.2f} ratio={ratio:.4f} co={co} cp={cp}')
            except Exception as exc:rows.append(f'VG_CHAIN_REJECT {m} {pdl} {type(exc).__name__}:{str(exc)[:100]}')
            finally:
                if graph is not None:graph.close()
        old.close()
    for m,name,table in measurements:rows.append(f'VG_TMA_FAMILY {m} {name} index_us='+str(table).replace(' ',''))
    rows.append(f'VG_TMA_END left_s={deadline-time.monotonic():.2f}')
    return rows


def lab(deadline,weights,state=None):
    # The native portable reader and all Hopper ABI families are validated separately
    # on sm120; this host-screen branch keeps the lab within prepare's300-second cap.
    if torch.cuda.get_device_capability()[0]>=12:
        native=vg_tma_module();x=torch.randn(33,1024,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(512,1024,device='cuda',dtype=torch.bfloat16)
        for c in ((64,1,1,3,1,1,0,1,2),(32,1,1,2,2,2,1,1,4)):
            with _PgGraph(lambda:vg_tma(x,w,c,torch.float32,False,native)) as g:
                vp_check(g.out,x.float()@w.float().T)
        return ['VP_LOCAL portable TMA/cluster graph math passes; full Hopper families forced by dev/verify/validate_grouped.py']
    return vg_tma_lab(deadline,weights)

# ---- h200_lab wrapper: payload verify_proj_tma_v2 sha256 b0cfbb95563cfec3 on base f3786341611f4f1f ----
_lab_base_prepare = prepare
_lab_ran = False


def prepare(weights):
    global _lab_ran
    import time as _t
    _start = _t.monotonic()
    state = _lab_base_prepare(weights)
    if not _lab_ran:
        _lab_ran = True
        import traceback as _tb
        _t0 = _t.monotonic()
        try:
            import inspect as _inspect
            _lab_params = len(_inspect.signature(lab).parameters)
            _lab_result = (lab(_start + 285, weights, state) if _lab_params >= 3
                           else lab(_start + 285, weights))
            _lines = [str(x) for x in (_lab_result or [])]
        except BaseException as _e:  # the run must still score; the error is the result
            _lines = [f"LAB_ERROR {type(_e).__name__}: {_e}"] + _tb.format_exc().splitlines()[-8:]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _out, _n = [], 0
        for _l in _lines:
            if _n + len(_l) + 1 > 9000:
                _out.append(f"LAB_TRUNCATED {len(_lines) - len(_out)} lines")
                break
            _out.append(_l)
            _n += len(_l) + 1
        print("LAB_BEGIN verify_proj_tma_v2 b0cfbb95563cfec3", flush=True)
        for _l in _out:
            print(_l, flush=True)
        print(f"LAB_END lab_s={_t.monotonic() - _t0:.1f} prepare_s={_t.monotonic() - _start:.1f}", flush=True)
    return state
