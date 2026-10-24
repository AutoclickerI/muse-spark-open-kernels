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

# ---- h200_lab payload verify_projection_glue.py ----
import torch
import triton
import triton.language as tl


@triton.jit
def _gb_publish(WARPS:tl.constexpr):
    tl.inline_asm_elementwise('fence.acq_rel.gpu; mov.u32 $0,$1;', '=r,r',
        [tl.arange(0,WARPS*32)],dtype=tl.int32,is_pure=False,pack=1)
    tl.debug_barrier()


@triton.jit(do_not_specialize=["M"])
def _gb_boundary_gemv(X,W,STREAM,ALPHA,BETA,POST,PRE,P,CNT,OUT,SOUT,
                      M,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,
                      BK:tl.constexpr,KS:tl.constexpr,BT:tl.constexpr,TB:tl.constexpr,
                      POSTW:tl.constexpr,WARPS:tl.constexpr):
    # Token tiles are adjacent: CTAs computing the same weights share L2 lines.
    MT=tl.cdiv(M,BT)
    pid=tl.program_id(0);pt=pid%MT;pn=pid//MT;ps=tl.program_id(1)
    n=pn*BN+tl.arange(0,BN);r=tl.arange(0,TB);t=pt*BT+r
    kk=tl.arange(0,BK);acc=tl.zeros((BN,TB),tl.float32)
    for off in range(ps*KS,(ps+1)*KS,BK):
        a=tl.load(X+t[:,None]*K+off+kk[None,:],mask=(t[:,None]<M)&(r[:,None]<BT),other=0.)
        w=tl.load(W+n[:,None]*K+off+kk[None,:])
        acc=tl.dot(w,tl.trans(a),acc)
    tl.store(P+ps*M*N+t[None,:]*N+n[:,None],acc,mask=(t[None,:]<M)&(r[None,:]<BT))
    _gb_publish(WARPS)
    previous=tl.atomic_add(CNT+pt,1,sem='acq_rel')
    if previous==(N//BN)*(K//KS)-1:
        _gb_publish(WARPS)
        if BT==1:
            h=tl.arange(0,8192)
            aa=tl.load(ALPHA+h);bb=tl.load(BETA+h);pw=tl.load(PRE+h)
            if POSTW:post=tl.load(POST+h)
            # A last CTA handles one small token tile; no grid spin or residency assumption.
            for rr in range(BT):
                tok=pt*BT+rr
                f=tl.zeros((8192,),tl.float32)
                for s in tl.static_range(K//KS):
                    f+=tl.load(P+s*M*N+tok*N+h,mask=(h<N)&(tok<M),other=0.,cache_modifier='.cg')
                f=f.to(tl.bfloat16).to(tl.float32)
                if POSTW:f=f*post
                norm=tl.rsqrt(tl.sum(f*f,0)/8192+1.e-8)
                st=tl.load(STREAM+tok*8192+h,mask=tok<M,other=0.)
                st=aa*st+bb*f*norm
                rh=tl.rsqrt(tl.sum(st*st,0)/8192+1.e-5)
                tl.store(SOUT+tok*8192+h,st,mask=tok<M)
                tl.store(OUT+tok*8192+h,(st*rh*pw).to(tl.bfloat16),mask=tok<M)
        else:
            CH:tl.constexpr=512
            rr=tl.arange(0,triton.next_power_of_2(BT));tok=pt*BT+rr
            d=tl.arange(0,CH)
            sb=tl.zeros((triton.next_power_of_2(BT),),tl.float32)
            sa=tl.zeros((triton.next_power_of_2(BT),),tl.float32)
            sab=tl.zeros((triton.next_power_of_2(BT),),tl.float32)
            sbb=tl.zeros((triton.next_power_of_2(BT),),tl.float32)
            for block in range(8192//CH):
                col=block*CH+d
                v=tl.zeros((triton.next_power_of_2(BT),CH),tl.float32)
                for split in tl.static_range(K//KS):
                    v+=tl.load(P+split*M*N+tok[:,None]*N+col[None,:],
                        mask=(tok[:,None]<M)&(rr[:,None]<BT)&(col[None,:]<N),other=0.,cache_modifier='.cg')
                v=v.to(tl.bfloat16).to(tl.float32)
                if POSTW:v*=tl.load(POST+col)[None,:]
                carry=tl.load(ALPHA+col)[None,:]*tl.load(STREAM+tok[:,None]*8192+col[None,:],mask=(tok[:,None]<M)&(rr[:,None]<BT),other=0.)
                branch=tl.load(BETA+col)[None,:]*v
                sb+=tl.sum(v*v,1);sa+=tl.sum(carry*carry,1)
                sab+=tl.sum(carry*branch,1);sbb+=tl.sum(branch*branch,1)
            rb=tl.rsqrt(sb/8192+1.e-8)
            rh=tl.rsqrt((sa+2*rb*sab+rb*rb*sbb)/8192+1.e-5)
            for block in range(8192//CH):
                col=block*CH+d
                v=tl.zeros((triton.next_power_of_2(BT),CH),tl.float32)
                for split in tl.static_range(K//KS):
                    v+=tl.load(P+split*M*N+tok[:,None]*N+col[None,:],
                        mask=(tok[:,None]<M)&(rr[:,None]<BT)&(col[None,:]<N),other=0.,cache_modifier='.cg')
                v=v.to(tl.bfloat16).to(tl.float32)
                if POSTW:v*=tl.load(POST+col)[None,:]
                carry=tl.load(ALPHA+col)[None,:]*tl.load(STREAM+tok[:,None]*8192+col[None,:],mask=(tok[:,None]<M)&(rr[:,None]<BT),other=0.)
                result=carry+tl.load(BETA+col)[None,:]*v*rb[:,None]
                tl.store(SOUT+tok[:,None]*8192+col[None,:],result,mask=(tok[:,None]<M)&(rr[:,None]<BT))
                y=result*rh[:,None]*tl.load(PRE+col)[None,:]
                tl.store(OUT+tok[:,None]*8192+col[None,:],y.to(tl.bfloat16),mask=(tok[:,None]<M)&(rr[:,None]<BT))
        _gb_publish(WARPS)
        tl.atomic_xchg(CNT+pt,0,sem='release')


@triton.jit(do_not_specialize=["M"])
def _gb_pool_route(X,W,PRE,BIAS,LSTART,P,RP,CNT,Y,IDS,WTS,LIDS,
                   M,BN:tl.constexpr,BK:tl.constexpr,
                   KS:tl.constexpr,BT:tl.constexpr,TB:tl.constexpr,WARPS:tl.constexpr):
    MT=tl.cdiv(M,BT)
    pid=tl.program_id(0);pt=pid%MT;pn=pid//MT;ps=tl.program_id(1)
    r=tl.arange(0,TB);t=pt*BT+r;kk=tl.arange(0,BK)
    if pn<4096//BN:
        n=pn*BN+tl.arange(0,BN)
        acc=tl.zeros((BN,TB),tl.float32)
        for off in range(ps*KS,(ps+1)*KS,BK):
            a=tl.load(X+t[:,None]*8192+off+kk[None,:],mask=(t[:,None]<M)&(r[:,None]<BT),other=0.)
            w=tl.load(W+n[:,None]*8192+off+kk[None,:])
            acc=tl.dot(w,tl.trans(a),acc)
        tl.store(P+ps*M*4096+t[None,:]*4096+n[:,None],acc,mask=(t[None,:]<M)&(r[None,:]<BT))
        _gb_publish(WARPS)
        prev=tl.atomic_add(CNT+pt*2,1,sem='acq_rel')
        if prev==(4096//BN)*(8192//KS)-1:
            _gb_publish(WARPS)
            if BT==1:
                d=tl.arange(0,4096);pre=tl.load(PRE+d)
                for rr in range(BT):
                    tok=pt*BT+rr
                    p=tl.zeros((4096,),tl.float32)
                    for s in tl.static_range(8192//KS):
                        p+=tl.load(P+s*M*4096+tok*4096+d,mask=tok<M,other=0.,cache_modifier='.cg')
                    # Stock pooler output is BF16 before its pre-expert RMSNorm.
                    p=p.to(tl.bfloat16).to(tl.float32)
                    p=p*tl.rsqrt(tl.sum(p*p,0)/4096+1.e-5)*pre
                    tl.store(Y+tok*4096+d,p.to(tl.bfloat16),mask=tok<M)
            else:
                CH:tl.constexpr=512
                rr=tl.arange(0,triton.next_power_of_2(BT));tok=pt*BT+rr;col=tl.arange(0,CH)
                squares=tl.zeros((triton.next_power_of_2(BT),),tl.float32)
                for block in range(4096//CH):
                    d=block*CH+col
                    val=tl.zeros((triton.next_power_of_2(BT),CH),tl.float32)
                    for split in tl.static_range(8192//KS):
                        val+=tl.load(P+split*M*4096+tok[:,None]*4096+d[None,:],mask=(tok[:,None]<M)&(rr[:,None]<BT),other=0.,cache_modifier='.cg')
                    val=val.to(tl.bfloat16).to(tl.float32)
                    squares+=tl.sum(val*val,1)
                scale=tl.rsqrt(squares/4096+1.e-5)
                for block in range(4096//CH):
                    d=block*CH+col
                    val=tl.zeros((triton.next_power_of_2(BT),CH),tl.float32)
                    for split in tl.static_range(8192//KS):
                        val+=tl.load(P+split*M*4096+tok[:,None]*4096+d[None,:],mask=(tok[:,None]<M)&(rr[:,None]<BT),other=0.,cache_modifier='.cg')
                    val=val.to(tl.bfloat16).to(tl.float32)
                    val=val*scale[:,None]*tl.load(PRE+d)[None,:]
                    tl.store(Y+tok[:,None]*4096+d[None,:],val.to(tl.bfloat16),mask=(tok[:,None]<M)&(rr[:,None]<BT))
            _gb_publish(WARPS)
            tl.atomic_xchg(CNT+pt*2,0,sem='release')
    else:
        # Tiled router rows: hi and lo products share FP32 accumulators.
        e=(pn-4096//BN)*BN+tl.arange(0,BN)
        racc=tl.zeros((BN,TB),tl.float32)
        for off in range(ps*KS,(ps+1)*KS,BK):
            a=tl.load(X+t[:,None]*8192+off+kk[None,:],mask=(t[:,None]<M)&(r[:,None]<BT),other=0.)
            hi=tl.load(W+(4096+e[:,None])*8192+off+kk[None,:])
            lo=tl.load(W+(4352+e[:,None])*8192+off+kk[None,:])
            racc=tl.dot(hi,tl.trans(a),racc)
            racc=tl.dot(lo,tl.trans(a),racc)
        tl.store(RP+ps*M*256+t[None,:]*256+e[:,None],racc,mask=(t[None,:]<M)&(r[None,:]<BT))
        _gb_publish(WARPS)
        prev=tl.atomic_add(CNT+pt*2+1,1,sem='acq_rel')
        if prev==(256//BN)*(8192//KS)-1:
            _gb_publish(WARPS)
            efull=tl.arange(0,256)
            logits=tl.zeros((TB,256),tl.float32)
            for s in tl.static_range(8192//KS):
                logits+=tl.load(RP+s*M*256+t[:,None]*256+efull[None,:],mask=(t[:,None]<M)&(r[:,None]<BT),other=0.,cache_modifier='.cg')
            score=tl.sigmoid(logits*0.1767766952966369)
            biased=score+tl.load(BIAS+efull)[None,:]
            kr=tl.arange(0,8)
            ids=tl.zeros((TB,8),tl.int32);picked=tl.zeros((TB,8),tl.float32)
            for i in tl.static_range(8):
                which=tl.argmax(biased,1)
                value=tl.sum(tl.where(efull[None,:]==which[:,None],score,0.),1)
                ids=tl.where(kr[None,:]==i,which[:,None],ids)
                picked=tl.where(kr[None,:]==i,value[:,None],picked)
                biased=tl.where(efull[None,:]==which[:,None],-float('inf'),biased)
            picked=picked/(tl.sum(picked,1)[:,None]+1.e-15)
            lid=(ids-tl.load(LSTART)+256)%256
            mask=(t[:,None]<M)&(r[:,None]<BT)
            tl.store(IDS+t[:,None]*8+kr[None,:],ids,mask=mask)
            tl.store(WTS+t[:,None]*8+kr[None,:],picked,mask=mask)
            tl.store(LIDS+t[:,None]*8+kr[None,:],tl.where(lid<32,lid,-1),mask=mask)
            _gb_publish(WARPS)
            tl.atomic_xchg(CNT+pt*2+1,0,sem='release')

import time
import statistics

GB_TOKENS=(1,8,16,32,128,256)


def gb_choices(m,phase):
    for bt in tuple(dict.fromkeys((min(m,8),min(m,16),min(m,4)))):
        for bn in (64,32):
            for splits in ((2,1,4) if phase=='b1' else (8,4,2) if phase=='pool' else (8,4,2,1)):
                yield bt,bn,splits,64 if phase=='pool' else 128,4


class GbPlan:
    def __init__(self,state,m,phase,choice):
        self.m,self.phase,self.choice=m,phase,choice
        bt,bn,splits,bk,warps=choice
        if phase not in ('b1','pool','b2') or bt<1 or bt>32 or bn not in (32,64) or splits not in (1,2,4,8,16):
            raise ValueError('invalid glue plan')
        if bk not in (32,64,128) or warps not in (4,8):raise ValueError('invalid glue launch')
        device=state['o'].device
        self.n=4608 if phase=='pool' else 8192 if phase=='b1' or m>8 else 1024
        self.k=8192 if phase=='pool' else 1024 if phase=='b1' else 4096
        if self.k%splits or (self.k//splits)%bk:raise ValueError('invalid split granule')
        self.part=torch.empty((splits,m,4096 if phase=='pool' else self.n),device=device,dtype=torch.float32)
        self.rpart=torch.empty((splits,m,256),device=device,dtype=torch.float32) if phase=='pool' else self.part
        # Only synchronization state is prepared. All input-dependent outputs are overwritten each call.
        self.counter=torch.zeros((triton.cdiv(m,bt)*(2 if phase=='pool' else 1),),device=device,dtype=torch.int32)
        self.state=state

    def run(self,x,stream=None,data=None):
        st=self.state;m=self.m;bt,bn,splits,bk,warps=self.choice;tb=max(16,triton.next_power_of_2(bt))
        if x.shape!=(m,self.k) or x.dtype!=torch.bfloat16 or not x.is_contiguous():raise ValueError('glue input layout')
        mt=triton.cdiv(m,bt);dev=x.device
        if self.phase=='pool':
            if data is None:raise ValueError('routing requires local_start')
            y=torch.empty((m,4096),device=dev,dtype=torch.bfloat16)
            ids=torch.empty((m,8),device=dev,dtype=torch.int32)
            weights=torch.empty((m,8),device=dev,dtype=torch.float32);lids=torch.empty_like(ids)
            _gb_pool_route[((4096//bn+256//bn)*mt,splits)](x,st['pool_router'],st['pre_expert_w'],
                st['router_bias'],data['local_start'],self.part,self.rpart,self.counter,y,ids,weights,lids,
                M=m,BN=bn,BK=bk,KS=8192//splits,BT=bt,TB=tb,WARPS=warps,
                num_warps=warps,num_stages=2,enable_fp_fusion=False)
            return y,ids,weights,lids
        if stream is None or stream.shape!=(m,8192) or stream.dtype!=torch.float32:raise ValueError('glue stream layout')
        b2=self.phase=='b2';name=('unpooler_slice' if m<=8 else 'unpooler') if b2 else 'o'
        a,b,pw=('alpha2','beta2','next_pre_attn_w') if b2 else ('alpha1','beta1','pre_ffn_w')
        out=torch.empty((m,8192),device=dev,dtype=torch.bfloat16);sout=torch.empty_like(stream)
        _gb_boundary_gemv[(self.n//bn*mt,splits)](x,st[name],stream,st[a],st[b],st['post_ffn_w'],st[pw],
            self.part,self.counter,out,sout,M=m,N=self.n,K=self.k,BN=bn,BK=bk,KS=self.k//splits,
            BT=bt,TB=tb,POSTW=b2,WARPS=warps,num_warps=warps,num_stages=2,enable_fp_fusion=False)
        return out,sout

# Synthetic untimed math guards. Never invoked by runtime projection calls.
def gb_state(weights):
    dev=weights['o'].device
    gen=torch.Generator(device=dev).manual_seed(838291)
    def vector(name,scale):
        return weights[name] if name in weights else (torch.randn(4096 if name=='pre_expert_norm' else 8192,
            device=dev,generator=gen)*scale).bfloat16()
    def coefficients(gate):
        gate=gate.float()/0.3;b=torch.sigmoid(gate)
        return torch.sqrt(torch.clamp(torch.sigmoid(-gate)*(1+b),min=1.e-3)),b
    a1,b1=coefficients(vector('attn_residual_gate',.3));a2,b2=coefficients(vector('ffn_residual_gate',.3))
    hi=weights['router'].bfloat16();lo=(weights['router'].float()-hi.float()).bfloat16()
    state={k:weights[k].contiguous() for k in ('qkv_gate','o','unpooler')}
    state.update(pool_router=torch.cat((weights['pooler'],hi,lo),0).contiguous(),
        unpooler_slice=weights['unpooler'][:1024].contiguous(),alpha1=a1.contiguous(),beta1=b1.contiguous(),
        alpha2=a2.contiguous(),beta2=b2.contiguous(),pre_ffn_w=1+vector('pre_ffn_norm',.1).float(),
        pre_expert_w=1+vector('pre_expert_norm',.1).float(),post_ffn_w=1+vector('post_ffn_norm',.1).float(),
        next_pre_attn_w=1+vector('next_pre_attn_norm',.1).float(),router_bias=(weights['router_bias'].float().contiguous()
            if 'router_bias' in weights else torch.randn(256,device=dev,generator=gen)*.035))
    return state


def gb_reference(phase,x,stream,state,data):
    if phase=='pool':
        p=x.float()@state['pool_router'].float().T
        pool=p[:,:4096].bfloat16().float()
        pooled=(pool*torch.rsqrt(pool.square().mean(-1,keepdim=True)+1.e-5)*state['pre_expert_w']).bfloat16()
        scores=torch.sigmoid((p[:,4096:4352]+p[:,4352:])*0.1767766952966369)
        biased=scores+state['router_bias'];ids=biased.topk(8,dim=-1).indices.to(torch.int32)
        w=scores.gather(1,ids.long());w=w/(w.sum(-1,keepdim=True)+1.e-15)
        lid=(ids-data['local_start']+256)%256;lid=torch.where(lid<32,lid,-1)
        return (pooled,ids,w,lid),{'scores':scores,'biased':biased}
    b2=phase=='b2';name=('unpooler_slice' if x.shape[0]<=8 else 'unpooler') if b2 else 'o'
    f=(x.float()@state[name].float().T).bfloat16().float()
    if f.shape[1]<8192:f=torch.nn.functional.pad(f,(0,8192-f.shape[1]))
    if b2:f=f*state['post_ffn_w']
    a,b,pre=('alpha2','beta2','next_pre_attn_w') if b2 else ('alpha1','beta1','pre_ffn_w')
    st=state[a]*stream+state[b]*f*torch.rsqrt(f.square().mean(-1,keepdim=True)+1.e-8)
    hidden=(st*torch.rsqrt(st.square().mean(-1,keepdim=True)+1.e-5)*state[pre]).bfloat16()
    return (hidden,st),{}


def gb_rel(actual,expected,limit=.01):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or not bool(torch.isfinite(actual).all()):
        raise ValueError('glue dtype/shape/nonfinite')
    rel=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1.e-12))
    if rel>limit:raise ValueError(f'glue rel={rel:.6f}>{limit}')
    return rel


def gb_check(phase,actual,expected,meta,stream,state,data):
    err=gb_rel(actual[0],expected[0])
    if phase!='pool':
        err=max(err,gb_rel(actual[1],expected[1]))
        # Residual carry must not hide errors in the branch contribution.
        alpha=state['alpha2' if phase=='b2' else 'alpha1']
        err=max(err,gb_rel(actual[1]-alpha*stream,expected[1]-alpha*stream))
        return err
    ids=actual[1].long();scores=meta['scores'];biased=meta['biased']
    if bool((ids<0).any()) or bool((ids>=256).any()):raise ValueError('route id range')
    sorted_ids=ids.sort(-1).values
    if bool((sorted_ids[:,1:]==sorted_ids[:,:-1]).any()):raise ValueError('duplicate route')
    best=biased.topk(8,dim=-1);ninth=biased.topk(9,dim=-1).values[:,-1:]
    if float((best.values[:,-1:]-biased.gather(1,ids)).max())>.05:raise ValueError('public routing gap')
    start=int(data['local_start'].item());local=(best.indices-start)%256<32
    kept=(best.indices[:,:,None]==ids[:,None,:]).any(-1)
    if bool((local&~kept&(best.values-ninth>.03)).any()):raise ValueError('public required local expert')
    own_local=(ids-start)%256<32
    if int(own_local.sum())<int(local.sum())*.92-4:raise ValueError('public local route count')
    own_distinct=ids[own_local].unique().numel();ref_distinct=best.indices[local].unique().numel()
    if own_distinct<ref_distinct*.85-2:raise ValueError('public distinct local experts')
    wanted=scores.gather(1,ids);wanted=wanted/(wanted.sum(-1,keepdim=True)+1.e-15)
    err=max(err,gb_rel(actual[2],wanted))
    lid=(ids-start+256)%256;lid=torch.where(lid<32,lid,-1).to(torch.int32)
    if not torch.equal(actual[3],lid):raise ValueError('local expert map')
    return err

import ctypes as ct
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


def gb_pair(old,new,inputs,banks,flush,repeats=20):
    vals=[[],[]];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for r in range(repeats):
        for target,fresh in zip(inputs,banks[r%len(banks)]):target.copy_(fresh)
        for side in ((0,1) if r%2==0 else (1,0)):
            flush.sum();a.record();(old if side==0 else new).graph.replay();b.record();b.synchronize()
            vals[side].append(float(a.elapsed_time(b))*1000)
    ratios=[((vals[1][r]/vals[0][r])*(vals[1][r+1]/vals[0][r+1]))**.5 for r in range(0,repeats,2)]
    return statistics.median(vals[0]),statistics.median(vals[1]),statistics.median(ratios)


def gb_callbacks(state,project_fn,boundary_fn,route_fn,o_pool_fn):
    def workspace(m,dev):
        raw=(torch.zeros if m<=8 else torch.empty)((15*m,),device=dev,dtype=torch.int32)
        return raw[3*m:].view(torch.float32).view(3,m,4),raw[:3*m].view(3,m)
    def b1(x,stream,data):
        m=x.shape[0];sf,si=workspace(m,x.device)
        p=project_fn(x,state['o'],'o',state)
        return boundary_fn(p,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],
            (sf[0],si[0]),use_pdl=state.get('pdl_plan',{}).get((m,'b1'),True))
    def pool(x,stream,data):
        m=x.shape[0];sf,si=workspace(m,x.device)
        p=project_fn(x,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
        return route_fn(p,state,data,(sf[1],si[1]))
    def b2_inner(x,stream,sf,si):
        m=x.shape[0];name='unpooler_slice' if m<=8 else 'unpooler'
        p=project_fn(x,state[name],name,state)
        return boundary_fn(p,stream,state['alpha2'],state['beta2'],state['next_pre_attn_w'],
            (sf[2],si[2]),postw=state['post_ffn_w'],
            use_pdl=state.get('pdl_plan',{}).get((m,'b2'),True))
    def b2(x,stream,data):
        sf,si=workspace(x.shape[0],x.device)
        return b2_inner(x,stream,sf,si)
    def prefix_inner(x,stream,data,sf,si):
        m=x.shape[0];kind,pdl=state.get('prefix_plan',{}).get(m,('regular',True))
        if kind=='fused':
            counters=torch.zeros((triton.cdiv(m,32),),device=x.device,dtype=torch.int32)
            pr,ss=o_pool_fn(x,stream,state,counters,use_pdl=pdl)
        else:
            a=project_fn(x,state['o'],'o',state)
            h,ss=boundary_fn(a,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],
                (sf[0],si[0]),use_pdl=pdl)
            pr=project_fn(h,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
        return route_fn(pr,state,data,(sf[1],si[1]))+(ss,)
    def prefix(x,stream,data):
        sf,si=workspace(x.shape[0],x.device)
        return prefix_inner(x,stream,data,sf,si)
    def chain(x,stream,moe,data):
        # Exactly one shared synchronization allocation/reset, as kernel_fn uses.
        sf,si=workspace(x.shape[0],x.device)
        front=prefix_inner(x,stream,data,sf,si)
        return front+b2_inner(moe,front[4],sf,si)
    return {'b1':b1,'pool':pool,'b2':b2,'prefix':prefix,'chain':chain}


def _gb_apply(bank,phase,x,stream,data):
    plan=bank['plans'].get((x.shape[0],phase))
    return plan.run(x,stream,data) if plan is not None else bank['callbacks'][phase](x,stream,data)


def gb_prefix(attn,stream,state,data):
    bank=state['_gb_glue']
    if not bank['prefix'].get(attn.shape[0],False):return bank['callbacks']['prefix'](attn,stream,data)
    h,stream1=_gb_apply(bank,'b1',attn,stream,data)
    return _gb_apply(bank,'pool',h,stream1,data)+(stream1,)


def gb_tail(moe,stream,state):
    return _gb_apply(state['_gb_glue'],'b2',moe,stream,None)


def gb_prepare(state,callbacks,deadline,*,tokens=GB_TOKENS,recipes=None):
    """Bounded per-shape choice; return compact lines to print after all other tuning.

    callbacks are the current cand_r3_best b1/pool/b2/prefix chains. recipes=None
    screens the registry; an H200-positive dict[(M,phase)]=[choice5] makes the port short.
    Optional failures reject and log. No input-derived result is retained in state.
    """
    begin=time.monotonic();deadline=min(deadline,begin+225)
    bank={'callbacks':callbacks,'plans':{},'prefix':{},'enabled':{},'trials':{}}
    state['_gb_glue']=bank;rows=[]
    if recipes is not None and not recipes:return ['GB_END no H200-positive recipes; incumbent retained']
    try:
        dev=state['o'].device;generator=torch.Generator(device=dev).manual_seed(2889121)
        flush=torch.zeros((1<<30)//4,device=dev,dtype=torch.float32)
        keys=[(m,p) for m in tokens for p in ('b1','pool','b2')]
        stage_stop=begin+.80*max(0,deadline-begin)
        for pos,(m,phase) in enumerate(keys):
            now=time.monotonic();stop=now+max(0,stage_stop-now-2)/(len(keys)-pos)
            if now>=stage_stop-2:rows.append(f'GB_BUDGET {m} {phase}');continue
            graph=None
            try:
                k=8192 if phase=='pool' else 1024 if phase=='b1' else 4096
                x=torch.randn(m,k,device=dev,dtype=torch.bfloat16,generator=generator)
                stream=torch.randn(m,8192,device=dev,generator=generator)
                start=torch.zeros((),device=dev,dtype=torch.int32);data={'local_start':start}
                inputs=(x,stream,start);banks=[];refs=[]
                for seed in range(3):
                    xx=torch.randn(x.shape,device=dev,dtype=x.dtype,generator=generator)
                    ss=torch.randn(stream.shape,device=dev,generator=generator)
                    if seed==2:xx.zero_();xx[:,127]=.5;ss[:,::97]*=10
                    ls=torch.tensor((seed*37)%256,device=dev,dtype=torch.int32)
                    banks.append((xx,ss,ls));refs.append(gb_reference(phase,xx,ss,state,{'local_start':ls}))
                oldfn=lambda:callbacks[phase](x,stream,data)
                graph=_PgGraph(oldfn)
                for fresh,(ref,meta) in zip(banks,refs):
                    for a,b in zip(inputs,fresh):a.copy_(b)
                    graph.graph.replay();gb_check(phase,graph.out,ref,meta,stream,state,data)
                entries=[];rejections=0;reasons=set();coarse_stop=now+.58*max(0,stop-now)
                choices=list(gb_choices(m,phase)) if recipes is None else recipes.get((m,phase),[])
                for c in choices:
                    if time.monotonic()>coarse_stop:break
                    try:
                        plan=GbPlan(state,m,phase,c)
                        with _PgGraph(lambda:plan.run(x,stream,data)) as new:
                            error=0.
                            for fresh,(ref,meta) in zip(banks,refs):
                                for a,b in zip(inputs,fresh):a.copy_(b)
                                graph.graph.replay();new.graph.replay()
                                error=max(error,gb_check(phase,new.out,ref,meta,stream,state,data),
                                    gb_rel(new.out[0],graph.out[0]))
                                if phase!='pool':gb_rel(new.out[1]-state['alpha2' if phase=='b2' else 'alpha1']*stream,
                                                        graph.out[1]-state['alpha2' if phase=='b2' else 'alpha1']*stream)
                            t0,t1,ratio=gb_pair(graph,new,inputs,banks,flush,4)
                            entries.append((ratio,c,error))
                    except Exception as exc:
                        rejections+=1;reason=f'{type(exc).__name__}:{str(exc).splitlines()[0][:100]}'
                        if reason not in reasons and len(reasons)<2:rows.append(f'GB_REJECT {m} {phase} c={c} {reason}')
                        reasons.add(reason)
                best=float('inf');winner=None;final=[]
                for _,c,error in sorted(entries)[:3]:
                    if time.monotonic()>min(stop,deadline-2):break
                    try:
                        plan=GbPlan(state,m,phase,c)
                        with _PgGraph(lambda:plan.run(x,stream,data)) as new:
                            t0,t1,ratio=gb_pair(graph,new,inputs,banks,flush,20)
                            for fresh,(ref,meta) in zip(banks,refs):
                                for a,b in zip(inputs,fresh):a.copy_(b)
                                new.graph.replay();gb_check(phase,new.out,ref,meta,stream,state,data)
                        final.append((c,round(t0,3),round(t1,3),round(ratio,5)))
                        if ratio<best:best=ratio;winner=plan
                    except Exception as exc:
                        rejections+=1;rows.append(f'GB_REJECT final {m} {phase} {type(exc).__name__}:{str(exc)[:95]}')
                if winner is not None:bank['plans'][m,phase]=winner
                bank['trials'][m,phase]=final
                rows.append(f'GB_STAGE {m} {phase} choice={winner.choice if winner else None} ratio={best:.5f} n={len(entries)} rejected={rejections} final={sorted(final,key=lambda r:r[3])[:2]}')
            except Exception as exc:
                rows.append(f'GB_REJECT incumbent {m} {phase} {type(exc).__name__}:{str(exc).splitlines()[0][:125]}')
            finally:
                if graph is not None:graph.close()
        for m in tokens:
            if time.monotonic()>=deadline-1:rows.append(f'GB_BUDGET prefix {m}');continue
            if not any((m,p) in bank['plans'] for p in ('b1','pool')):
                rows.append(f'GB_PREFIX {m} incumbent no_stage_win');continue
            try:
                x=torch.randn(m,1024,device=dev,dtype=torch.bfloat16,generator=generator)
                stream=torch.randn(m,8192,device=dev,generator=generator)
                ls=torch.zeros((),device=dev,dtype=torch.int32);data={'local_start':ls}
                banks=[(torch.randn_like(x),torch.randn_like(stream),torch.tensor(j*17,device=dev,dtype=torch.int32)) for j in range(3)]
                def fused():
                    h,ss=_gb_apply(bank,'b1',x,stream,data)
                    return _gb_apply(bank,'pool',h,ss,data)+(ss,)
                with _PgGraph(lambda:callbacks['prefix'](x,stream,data)) as old, _PgGraph(fused) as new:
                    for fresh in banks:
                        for a,b in zip((x,stream,ls),fresh):a.copy_(b)
                        old.graph.replay();new.graph.replay()
                        gb_rel(new.out[0],old.out[0]);gb_rel(new.out[4]-state['alpha1']*stream,old.out[4]-state['alpha1']*stream)
                        (href,sref),_=gb_reference('b1',x,stream,state,data)
                        ref,meta=gb_reference('pool',href,sref,state,data)
                        gb_check('pool',new.out[:4],ref,meta,sref,state,data)
                    t0,t1,ratio=gb_pair(old,new,(x,stream,ls),banks,flush,20)
                    bank['prefix'][m]=ratio<1.
                    rows.append(f'GB_PREFIX {m} enabled={ratio<1.} us={t0:.3f}/{t1:.3f} ratio={ratio:.5f}')
            except Exception as exc:rows.append(f'GB_REJECT prefix {m} {type(exc).__name__}:{str(exc)[:125]}')
        for m in tokens:
            if time.monotonic()>=deadline-1:rows.append(f'GB_BUDGET chain {m}');continue
            if not bank['prefix'].get(m,False) and (m,'b2') not in bank['plans'] and not any((m,p) in bank['plans'] for p in ('b1','pool')):
                rows.append(f'GB_CHAIN {m} enabled=False no_valid_fusion');continue
            tail_plan=bank['plans'].get((m,'b2'));best=1.;winner=None;results=[]
            try:
                x=torch.randn(m,1024,device=dev,dtype=torch.bfloat16,generator=generator)
                stream=torch.randn(m,8192,device=dev,generator=generator)
                moe=torch.randn(m,4096,device=dev,dtype=torch.bfloat16,generator=generator)
                ls=torch.zeros((),device=dev,dtype=torch.int32);data={'local_start':ls}
                inputs=(x,stream,moe,ls)
                banks=[(torch.randn_like(x),torch.randn_like(stream),torch.randn_like(moe),
                        torch.tensor(j*17,device=dev,dtype=torch.int32)) for j in range(3)]
                def fused_chain():
                    front=gb_prefix(x,stream,state,data)
                    return front+gb_tail(moe,front[4],state)
                with _PgGraph(lambda:callbacks['chain'](x,stream,moe,data)) as old:
                    for front_on,tail_on in ((False,True),(True,True),(True,False)):
                        if time.monotonic()>=deadline-1:break
                        if tail_on and tail_plan is None:continue
                        if front_on and not any((m,p) in bank['plans'] for p in ('b1','pool')):continue
                        bank['prefix'][m]=front_on
                        if tail_on:bank['plans'][m,'b2']=tail_plan
                        else:bank['plans'].pop((m,'b2'),None)
                        try:
                            with _PgGraph(fused_chain) as new:
                                for fresh in banks:
                                    for aa,bb in zip(inputs,fresh):aa.copy_(bb)
                                    old.graph.replay();new.graph.replay()
                                    for i in (0,4,5,6):gb_rel(new.out[i],old.out[i])
                                    carry=state['alpha1']*state['alpha2']*stream
                                    gb_rel(new.out[6]-carry,old.out[6]-carry)
                                    (href,sref),_=gb_reference('b1',x,stream,state,data)
                                    ref,meta=gb_reference('pool',href,sref,state,data)
                                    gb_check('pool',new.out[:4],ref,meta,sref,state,data)
                                t0,t1,ratio=gb_pair(old,new,inputs,banks,flush,20)
                                results.append((int(front_on),int(tail_on),round(t0,3),round(t1,3),round(ratio,5)))
                                if ratio<best:best=ratio;winner=(front_on,tail_on)
                        except Exception as exc:rows.append(f'GB_REJECT chain {m} {front_on}/{tail_on} {type(exc).__name__}:{str(exc)[:100]}')
                rows.append(f'GB_CHAIN {m} enabled={winner is not None} choice={winner} ratio={best:.5f} all={results}')
            except Exception as exc:rows.append(f'GB_REJECT chain {m} {type(exc).__name__}:{str(exc)[:125]}')
            bank['enabled'][m]=winner is not None
            bank['prefix'][m]=bool(winner and winner[0])
            if winner and winner[1]:bank['plans'][m,'b2']=tail_plan
            else:bank['plans'].pop((m,'b2'),None)
            if not (winner and winner[0]):
                for phase in ('b1','pool'):bank['plans'].pop((m,phase),None)
    except Exception as exc:
        bank['plans'].clear();bank['prefix'].clear();bank['enabled'].clear()
        rows.append(f'GB_REJECT setup {type(exc).__name__}:{str(exc)[:150]}')
    rows.append(f'GB_END elapsed_s={time.monotonic()-begin:.3f} left_s={deadline-time.monotonic():.3f} picks={len(bank["plans"])} prefix={bank["prefix"]} enabled={bank["enabled"]}')
    return rows


def gb_wrap_kernel(original,project_fn,attention_fn,experts_fn):
    """Wrap cand_r3_best without changing attention/MoE or unselected shapes."""
    def kernel(data,state):
        m=data['config']['tokens'];bank=state.get('_gb_glue')
        if bank is None or not bank['enabled'].get(m,False):return original(data,state)
        qkv=project_fn(data['attn_input'],state['qkv_gate'],'qkv_gate',state)
        attn=attention_fn(qkv,data,state)
        x,ids,wts,lids,stream=gb_prefix(attn,data['stream'],state,data)
        moe=experts_fn(x,wts,lids,state)
        hidden,stream2=gb_tail(moe,stream,state)
        return hidden,stream2,ids
    return kernel

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
import ctypes as ct
import time,statistics
_gbc_HIDDEN = 8192
_gbc_D_EXPERT = 4096
_gbc_HQ, _gbc_HKV, _gbc_HD = (16, 2, 64)
_gbc_Q_SIZE, _gbc_KV_SIZE = (_gbc_HQ * _gbc_HD, _gbc_HKV * _gbc_HD)
_gbc_QKV_SIZE = 2 * _gbc_Q_SIZE + 2 * _gbc_KV_SIZE
_gbc_N_EXPERTS = 256
_gbc_TOP_K = 8
_gbc_LOCAL_EXPERTS = 32
_gbc_UNPOOL_SLICE = _gbc_HIDDEN // 8
_gbc_SLICED_MAX_TOKENS = 8
_gbc_RMS_EPS = 1e-05
_gbc_POST_EPS = 1e-08
_gbc_ROUTER_MULT = 0.1767766952966369
_gbc_ROPE_THETA = 500000.0
_gbc_FP8 = torch.float8_e4m3fn
_gbc_GEMV_MAX_T = 64
_gbc_PDL = True
_gbc_PREFETCH = True

@triton.jit
def _gbc__l2_prefetch(ptrs):
    return tl.inline_asm_elementwise('prefetch.global.L2 [$1]; mov.u32 $0, 0;', '=r,l', [ptrs], dtype=tl.int32, is_pure=False, pack=1)

@triton.jit
def _gbc__pdl_start(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_launch_dependents()

@triton.jit
def _gbc__pdl_wait(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_wait()

def _gbc__launch(kernel, grid, *args, use_pdl=None, **kw):
    enabled = _gbc_PDL if use_pdl is None else bool(use_pdl)
    if enabled:
        kw['launch_pdl'] = True
    return kernel[grid](*args, USE_PDL=enabled, **kw)

@triton.jit
def _gbc__gemv_kernel(X, W, P, T, x_stride, N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr, PREFETCH: tl.constexpr, USE_PDL: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    _gbc__pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    k0 = ps * KS
    if PREFETCH:
        lines = tl.arange(0, KS // 64)
        _gbc__l2_prefetch(W + n[:, None] * K + k0 + lines[None, :] * 64)
    _gbc__pdl_wait(USE_PDL)
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for k in range(k0, k0 + KS, BK):
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)
_gbc_GEMV_CFG = {(_gbc_QKV_SIZE, _gbc_HIDDEN): (64, 1024, 128), (_gbc_HIDDEN, _gbc_Q_SIZE): (64, 512, 128), (_gbc_D_EXPERT + 2 * _gbc_N_EXPERTS, _gbc_HIDDEN): (64, 2048, 128), (_gbc_UNPOOL_SLICE, _gbc_D_EXPERT): (64, 512, 128), (_gbc_HIDDEN, _gbc_D_EXPERT): (64, 2048, 128)}

def _gbc_gemv(x, w, out_dtype=torch.bfloat16):
    """x [T, K] bf16, w [N, K] bf16 -> (P [S, T, N], S): FP32 split-K partials, or for T above
    GEMV_MAX_T one cuBLAS product (S = 1, out_dtype)."""
    T, K = x.shape
    N = w.shape[0]
    if T > _gbc_GEMV_MAX_T:
        if out_dtype == torch.float32:
            out = torch.mm(x, w.t(), out_dtype=torch.float32)
        else:
            out = F.linear(x, w)
        return (out.unsqueeze(0), 1)
    choice = _gbc__GEMV_CHOICES.get((T, N, K))
    if choice == 'cublas':
        y = torch.mm(x, w.t(), out_dtype=torch.float32) if out_dtype == torch.float32 else F.linear(x, w)
        return (y.unsqueeze(0), 1)
    BN, KS, BK, pf = choice if choice is not None else (*_gbc_GEMV_CFG[N, K], _gbc_PREFETCH)
    S = K // KS
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    TB = max(16, triton.next_power_of_2(T))
    _gbc__launch(_gbc__gemv_kernel, (N // BN, S), x, w, P, T, x.stride(0), N=N, K=K, KS=KS, BN=BN, BK=BK, TB=TB, PREFETCH=pf, num_warps=4, num_stages=4)
    return (P, S)

@triton.jit
def _gbc__qk_prep_kernel(P, S: tl.constexpr, ps_stride, POS, LOC, KC, VC, KSCALE, VSCALE, INV_FREQ, QOUT, GATE, q_mult, eps, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr, Q_FP8: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    _gbc__pdl_start(USE_PDL)
    _gbc__pdl_wait(USE_PDL)
    W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    base = P + t * W
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    d = tl.arange(0, D)
    qo = hq[:, None] * D + d[None, :]
    ko = HQ * D + hk[:, None] * D + d[None, :]
    vo = (HQ + HKV) * D + hk[:, None] * D + d[None, :]
    go = (HQ + 2 * HKV) * D + qo
    q = tl.zeros((HQ, D), tl.float32)
    k = tl.zeros((HKV, D), tl.float32)
    v = tl.zeros((HKV, D), tl.float32)
    g = tl.zeros((HQ, D), tl.float32)
    for s in tl.static_range(S):
        q += tl.load(base + s * ps_stride + qo).to(tl.float32)
        k += tl.load(base + s * ps_stride + ko).to(tl.float32)
        v += tl.load(base + s * ps_stride + vo).to(tl.float32)
        g += tl.load(base + s * ps_stride + go).to(tl.float32)
    tl.store(GATE + t * (HQ * D) + qo, g)
    q = q * tl.rsqrt(tl.sum(q * q, 1) / D + eps)[:, None]
    k = k * tl.rsqrt(tl.sum(k * k, 1) / D + eps)[:, None]
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        half = tl.arange(0, D // 2)
        ang = pos * tl.load(INV_FREQ + half)
        c = tl.cos(ang)[None, :]
        sn = tl.sin(ang)[None, :]
        qe, qd = tl.split(tl.reshape(q, (HQ, D // 2, 2)))
        q = tl.reshape(tl.join(qe * c - qd * sn, qe * sn + qd * c), (HQ, D))
        ke, kd = tl.split(tl.reshape(k, (HKV, D // 2, 2)))
        k = tl.reshape(tl.join(ke * c - kd * sn, ke * sn + kd * c), (HKV, D))
    row = tl.load(LOC + t)
    kv_off = row * (HKV * D) + hk[:, None] * D + d[None, :]
    tl.store(KC + kv_off, (k / tl.load(KSCALE)).to(tl.float8e4nv))
    tl.store(VC + kv_off, (v / tl.load(VSCALE)).to(tl.float8e4nv))
    if Q_FP8:
        tl.store(QOUT + t * (HQ * D) + qo, q.to(tl.float8e4nv))
    else:
        tl.store(QOUT + t * (HQ * D) + qo, (q * q_mult).to(tl.bfloat16))

def _gbc_qk_prep(qkv, data, state, q_fp8: bool):
    P, S = qkv
    cfg = data['config']
    tokens = cfg['tokens']
    q = torch.empty((tokens, _gbc_HQ, _gbc_HD), device=P.device, dtype=_gbc_FP8 if q_fp8 else torch.bfloat16)
    gate = torch.empty((tokens, _gbc_Q_SIZE), device=P.device, dtype=torch.float32)
    _gbc__launch(_gbc__qk_prep_kernel, (tokens,), P, S, P.stride(0), data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], state['inv_freq'], q, gate, cfg['qk_scale_factor'], _gbc_RMS_EPS, HQ=_gbc_HQ, HKV=_gbc_HKV, D=_gbc_HD, ROPE=bool(cfg['rope_theta']), Q_FP8=q_fp8, num_warps=4)
    return (q, gate)

@triton.jit
def _gbc__group_barrier(CNT, n):
    """Spin until n programs have arrived at counter CNT (zeroed per call; the n programs must be
    co-resident, which holds for the small grids that use it)."""
    tl.atomic_add(CNT, 1, sem='release')
    while tl.atomic_add(CNT, 0, sem='acquire') < n:
        pass

@triton.jit
def _gbc__boundary_kernel(BR, S: tl.constexpr, bs_stride, br_stride, STREAM, STREAM_OUT, OUT, POSTW, ALPHA, BETA, PREW, SUMS, CNT, post_eps, pre_eps, NB: tl.constexpr, H: tl.constexpr, C: tl.constexpr, HAS_POSTW: tl.constexpr, USE_PDL: tl.constexpr):
    """Token t, chunk c of C. With C > 1 the chunks of a token meet at a barrier to share the four
    sums the two norms need: sum b^2, then sum (a s + b r_b)^2 = A + 2 r_b B + r_b^2 Cc."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    _gbc__pdl_start(USE_PDL)
    CH: tl.constexpr = H // C
    offs = c * CH + tl.arange(0, CH)
    alpha = tl.load(ALPHA + offs)
    beta = tl.load(BETA + offs)
    prew = tl.load(PREW + offs)
    if HAS_POSTW:
        postw = tl.load(POSTW + offs)
    _gbc__pdl_wait(USE_PDL)
    st = tl.load(STREAM + t * H + offs)
    br = tl.zeros((CH,), tl.float32)
    for s in tl.static_range(S):
        br += tl.load(BR + s * bs_stride + t * br_stride + offs, mask=offs < NB, other=0.0).to(tl.float32)
    if HAS_POSTW:
        br = br * postw
    if C == 1:
        br = br * tl.rsqrt(tl.sum(br * br, 0) / H + post_eps)
        st = alpha * st + beta * br
        rs = tl.rsqrt(tl.sum(st * st, 0) / H + pre_eps)
    else:
        a_s = alpha * st
        b_b = beta * br
        tl.atomic_add(SUMS + t * 4 + 0, tl.sum(br * br, 0))
        tl.atomic_add(SUMS + t * 4 + 1, tl.sum(a_s * a_s, 0))
        tl.atomic_add(SUMS + t * 4 + 2, tl.sum(a_s * b_b, 0))
        tl.atomic_add(SUMS + t * 4 + 3, tl.sum(b_b * b_b, 0))
        _gbc__group_barrier(CNT + t, C)
        sbb = tl.atomic_add(SUMS + t * 4 + 0, 0.0, sem='acquire')
        sss = tl.atomic_add(SUMS + t * 4 + 1, 0.0, sem='acquire')
        ssb = tl.atomic_add(SUMS + t * 4 + 2, 0.0, sem='acquire')
        sbb2 = tl.atomic_add(SUMS + t * 4 + 3, 0.0, sem='acquire')
        rb = tl.rsqrt(sbb / H + post_eps)
        st = a_s + b_b * rb
        rs = tl.rsqrt((sss + 2.0 * rb * ssb + rb * rb * sbb2) / H + pre_eps)
    tl.store(STREAM_OUT + t * H + offs, st)
    tl.store(OUT + t * H + offs, (st * rs * prew).to(tl.bfloat16))

def _gbc__chunks(tokens):
    return 16 if tokens <= 8 else 1

def _gbc_boundary(branch, stream, alpha, beta, prew, sync, postw=None, use_pdl=None):
    P, S = branch
    tokens = stream.shape[0]
    C = _gbc__chunks(tokens)
    s_out = torch.empty_like(stream)
    out = torch.empty((tokens, _gbc_HIDDEN), device=stream.device, dtype=torch.bfloat16)
    sums, cnt = sync
    _gbc__launch(_gbc__boundary_kernel, (tokens, C), P, S, P.stride(0), P.stride(1), stream, s_out, out, postw if postw is not None else alpha, alpha, beta, prew, sums, cnt, _gbc_POST_EPS, _gbc_RMS_EPS, NB=P.shape[2], H=_gbc_HIDDEN, C=C, HAS_POSTW=postw is not None, num_warps=8 if C == 1 else 4, use_pdl=use_pdl)
    return (out, s_out)

@triton.jit
def _gbc__route_norm_kernel(PR, S: tl.constexpr, ps_stride, pr_stride, BIAS, PREW, LSTART, X, IDS, WTS, LIDS, SUMS, CNT, mult, eps, D: tl.constexpr, NE: tl.constexpr, K: tl.constexpr, NLOCAL: tl.constexpr, C: tl.constexpr, USE_PDL: tl.constexpr):
    """Token t. Program 0: sigmoid scores from the hi+lo logits, biased top-8, renormalized
    weights, local ids (-1 elsewhere). Programs 1..C: rmsnorm(p) * (1 + w) over D / C each."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    _gbc__pdl_start(USE_PDL)
    if c == 0:
        e = tl.arange(0, NE)
        bias = tl.load(BIAS + e)
        lstart = tl.load(LSTART)
        _gbc__pdl_wait(USE_PDL)
        logit = tl.zeros((NE,), tl.float32)
        for s in tl.static_range(S):
            row = PR + s * ps_stride + t * pr_stride
            logit += tl.load(row + D + e).to(tl.float32) + tl.load(row + D + NE + e).to(tl.float32)
        score = tl.sigmoid(logit * mult)
        biased = score + bias
        ids = tl.zeros((K,), tl.int32)
        sk = tl.zeros((K,), tl.float32)
        kr = tl.arange(0, K)
        for k in tl.static_range(K):
            idx = tl.argmax(biased, 0)
            sel = e == idx
            ids = tl.where(kr == k, idx, ids)
            sk = tl.where(kr == k, tl.sum(tl.where(sel, score, 0.0), 0), sk)
            biased = tl.where(sel, -float('inf'), biased)
        tl.store(IDS + t * K + kr, ids)
        tl.store(WTS + t * K + kr, sk / (tl.sum(sk, 0) + 1e-15))
        lid = (ids - lstart + NE) % NE
        tl.store(LIDS + t * K + kr, tl.where(lid < NLOCAL, lid, -1))
    else:
        CH: tl.constexpr = D // C
        offs = (c - 1) * CH + tl.arange(0, CH)
        prew = tl.load(PREW + offs)
        _gbc__pdl_wait(USE_PDL)
        p = tl.zeros((CH,), tl.float32)
        for s in tl.static_range(S):
            p += tl.load(PR + s * ps_stride + t * pr_stride + offs).to(tl.float32)
        if C == 1:
            ss = tl.sum(p * p, 0)
        else:
            tl.atomic_add(SUMS + t, tl.sum(p * p, 0))
            _gbc__group_barrier(CNT + t, C)
            ss = tl.atomic_add(SUMS + t, 0.0, sem='acquire')
        tl.store(X + t * D + offs, (p * tl.rsqrt(ss / D + eps) * prew).to(tl.bfloat16))

def _gbc_route_and_norm(pr, state, data, sync):
    P, S = pr
    tokens = P.shape[1]
    C = _gbc__chunks(tokens)
    dev = P.device
    x = torch.empty((tokens, _gbc_D_EXPERT), device=dev, dtype=torch.bfloat16)
    ids = torch.empty((tokens, _gbc_TOP_K), device=dev, dtype=torch.int32)
    wts = torch.empty((tokens, _gbc_TOP_K), device=dev, dtype=torch.float32)
    lids = torch.empty((tokens, _gbc_TOP_K), device=dev, dtype=torch.int32)
    sums, cnt = sync
    _gbc__launch(_gbc__route_norm_kernel, (tokens, 1 + C), P, S, P.stride(0), P.stride(1), state['router_bias'], state['pre_expert_w'], data['local_start'], x, ids, wts, lids, sums, cnt, _gbc_ROUTER_MULT, _gbc_RMS_EPS, D=_gbc_D_EXPERT, NE=_gbc_N_EXPERTS, K=_gbc_TOP_K, NLOCAL=_gbc_LOCAL_EXPERTS, C=C, num_warps=4 if C > 1 else 8)
    return (x, ids, wts, lids)
_gbc_H = 4096
_gbc__GEMV_CHOICES = {}
_gbc_LT_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cublasLt.h>\n#include <cstring>\nnamespace decode_lt {\nstatic cublasLtHandle_t handle() {\n  static cublasLtHandle_t h=[] { cublasLtHandle_t p; TORCH_CHECK(cublasLtCreate(&p)==CUBLAS_STATUS_SUCCESS,"cublasLtCreate"); return p; }();\n  return h;\n}\nstruct Layout {\n  cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a,b,c;\n  Layout(const torch::Tensor& w,const torch::Tensor& x,const torch::Tensor& y) {\n    int64_t n=y.size(1),m=x.size(0),k=x.size(1);\n    TORCH_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"Lt desc");\n    cublasOperation_t ta=CUBLAS_OP_T;\n    TORCH_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))==CUBLAS_STATUS_SUCCESS,"Lt transpose");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,w.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt A layout");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,x.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt B layout");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n)==CUBLAS_STATUS_SUCCESS,"Lt C layout");\n  }\n  ~Layout(){cublasLtMatmulDescDestroy(op);cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(c);}\n};\nconstexpr int CAP=32; constexpr size_t WORK_BYTES=32*1024*1024;\n}\ntorch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out) {\n  using namespace decode_lt; Layout layout(W,X,Out); cublasLtMatmulPreference_t pref;\n  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS,"Lt preference");\n  size_t workspace=WORK_BYTES;\n  TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace))==CUBLAS_STATUS_SUCCESS,"Lt workspace");\n  cublasLtMatmulHeuristicResult_t results[CAP]; int count=0;\n  auto status=cublasLtMatmulAlgoGetHeuristic(handle(),layout.op,layout.a,layout.b,layout.c,layout.c,pref,CAP,results,&count);\n  cublasLtMatmulPreferenceDestroy(pref); TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt heuristics");\n  auto codes=torch::empty({count,(int64_t)sizeof(cublasLtMatmulAlgo_t)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU));\n  for(int i=0;i<count;++i){TORCH_CHECK(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=WORK_BYTES,"Lt invalid heuristic");std::memcpy(codes.data_ptr<uint8_t>()+i*sizeof(cublasLtMatmulAlgo_t),&results[i].algo,sizeof(cublasLtMatmulAlgo_t));}\n  return codes;\n}\nvoid lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index) {\n  using namespace decode_lt; TORCH_CHECK(index>=0 && index<Algos.size(0) && Work.nbytes()>=WORK_BYTES,"Lt invalid algorithm");\n  Layout layout(W,X,Out); cublasLtMatmulAlgo_t algo; std::memcpy(&algo,Algos.data_ptr<uint8_t>()+index*sizeof(algo),sizeof(algo));\n  const float alpha=1.f,beta=0.f;\n  auto status=cublasLtMatmul(handle(),layout.op,&alpha,W.data_ptr(),layout.a,X.data_ptr(),layout.b,&beta,Out.data_ptr(),layout.c,Out.data_ptr(),layout.c,&algo,Work.data_ptr(),WORK_BYTES,at::cuda::getCurrentCUDAStream());\n  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt matmul");\n}\n'
_gbc_LT_CPP_SRC = '\ntorch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out);\nvoid lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index);\n'
_gbc__LT_MOD = None

def _gbc__lt_module():
    global _gbc__LT_MOD
    if _gbc__LT_MOD is None:
        from torch.utils.cpp_extension import load_inline
        import hashlib
        tag = hashlib.sha1(_gbc_LT_CUDA_SRC.encode()).hexdigest()[:12]
        _gbc__LT_MOD = load_inline(name=f'muse_decode_lt_{tag}', cpp_sources=_gbc_LT_CPP_SRC, cuda_sources=_gbc_LT_CUDA_SRC, functions=['lt_options', 'lt_run'], extra_cuda_cflags=['-O3', '-std=c++20'], extra_ldflags=['-lcublasLt'], verbose=False)
    return _gbc__LT_MOD

def _gbc__dense_project(x, w, out_dtype):
    if out_dtype == torch.float32:
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    return F.linear(x, w)

def _gbc__project_variant(x, w, name, state, variant, out_dtype=torch.bfloat16):
    if isinstance(variant, tuple) and variant[0] == 'short_lt':
        _, nn, ordinal = variant
        bank = state['measured_lt']
        weight, codes = bank['layouts'][x.shape[0], name, nn]
        out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=out_dtype)
        bank['mod'].run(x, weight, out, bank['workspace'], codes, ordinal, nn, out_dtype == torch.float32)
        return (out.unsqueeze(0), 1)
    if variant == 'custom':
        return _gbc_gemv(x, w, out_dtype=out_dtype)
    if variant == 'cublas':
        return (_gbc__dense_project(x, w, out_dtype).unsqueeze(0), 1)
    mod = state['lt_mod']
    out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
    mod.lt_run(w, x, out, state['lt_workspace'], state['lt_algos'][name, x.shape[0]], int(variant[1]))
    return (out.unsqueeze(0), 1)

def _gbc_project(x, w, name, state, out_dtype=torch.bfloat16):
    variant = state.get('proj_plan', {}).get((x.shape[0], name), 'custom')
    return _gbc__project_variant(x, w, name, state, variant, out_dtype=out_dtype)

def _gbc__materialize(value):
    p, s = value
    return p.sum(0) if s > 1 else p[0]

def _gbc__measure_graph(fn, flush, repeats=3):
    for _ in range(2):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(repeats):
        flush.sum()
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        vals.append(a.elapsed_time(b) * 1000.0)
    del graph, out
    return float(__import__('statistics').median(vals))
_gbc__TUNE_TOKENS = (1, 8, 16, 32, 128, 256)

def _gbc__fa3_on_device(device):
    return torch.cuda.get_device_capability(device) == (9, 0)

def _gbc__tune_projections(state, flush, device):
    plan = {}
    if _gbc__fa3_on_device(device):
        try:
            state['lt_mod'] = _gbc__lt_module()
            state['lt_workspace'] = torch.empty((32 << 20,), dtype=torch.uint8, device=device)
        except Exception as exc:
            print(f'PROJ_LT_REJECT module reason={str(exc).splitlines()[0][:100]}', flush=True)
            state['lt_mod'] = None
    names = ('qkv_gate', 'o', 'pool_router', 'unpooler_slice', 'unpooler')
    for t in _gbc__TUNE_TOKENS:
        for name in names:
            if name == 'unpooler_slice' and t > 8 or (name == 'unpooler' and t <= 8):
                continue
            if name == 'pool_router' and t <= 32:
                plan[t, name] = 'custom'
                print(f'PROJ_TUNE t={t} name={name} best=custom receipt=V9_union', flush=True)
                continue
            w = state[name]
            out_dtype = torch.float32 if name == 'pool_router' else torch.bfloat16
            x = torch.randn((t, w.shape[1]), device=device, dtype=torch.bfloat16)
            state['_proj_stream'] = torch.randn((t, _gbc_HIDDEN), device=device, dtype=torch.float32)
            state['_proj_data'] = {'positions': torch.arange(t, device=device, dtype=torch.int64), 'cache_loc': torch.arange(t, device=device, dtype=torch.int64), 'k_cache': torch.empty((t, 1, _gbc_HKV, _gbc_HD), device=device, dtype=_gbc_FP8), 'v_cache': torch.empty((t, 1, _gbc_HKV, _gbc_HD), device=device, dtype=_gbc_FP8), 'k_scale': torch.ones((), device=device), 'v_scale': torch.ones((), device=device), 'local_start': torch.zeros((), device=device, dtype=torch.int32), 'config': dict(tokens=t, rope_theta=_gbc_ROPE_THETA, qk_scale_factor=5.4730064863875)}
            variants = ['custom', 'cublas']
            if _gbc__fa3_on_device(device) and state.get('lt_mod') is not None and (name != 'pool_router'):
                try:
                    out = torch.empty((t, w.shape[0]), device=device, dtype=torch.bfloat16)
                    state.setdefault('lt_algos', {})[name, t] = state['lt_mod'].lt_options(w, x, out)
                    variants += [('lt', i) for i in range(int(state['lt_algos'][name, t].shape[0]))]
                except Exception as exc:
                    print(f'PROJ_LT_REJECT name={name} t={t} reason={str(exc).splitlines()[0][:100]}', flush=True)
            scores = []
            want = None
            for variant in variants:
                fn = lambda variant=variant: _gbc__materialize(_gbc__project_variant(x, w, name, state, variant, out_dtype))
                try:
                    y = fn()
                    torch.cuda.synchronize()
                    want = y.float() if want is None else want
                    rel = ((y.float() - want).norm() / want.norm().clamp_min(1e-12)).item()
                    if not torch.isfinite(y).all().item() or rel > 0.01:
                        print(f'PROJ_REJECT name={name} t={t} variant={variant} rel={rel:.4g}', flush=True)
                        continue
                    scores.append((variant, _gbc__measure_graph(lambda variant=variant: _gbc__projection_consumer(x, w, name, state, variant, out_dtype), flush, repeats=3)))
                except Exception as exc:
                    print(f'PROJ_REJECT name={name} t={t} variant={variant} reason={str(exc).splitlines()[0][:100]}', flush=True)
            if not scores:
                print(f'PROJ_REJECT name={name} t={t} reason=no_valid_variant fallback=custom', flush=True)
                plan[t, name] = 'custom'
                continue
            best = min(scores, key=lambda z: z[1])
            plan[t, name] = best[0]
            print(f'PROJ_TUNE t={t} name={name} best={best[0]} us={best[1]:.2f} all={scores}', flush=True)
    return plan

def _gbc__tune_pdl(state, flush, device):
    plan = {}
    for t in _gbc__TUNE_TOKENS:
        stream = torch.randn((t, _gbc_HIDDEN), device=device, dtype=torch.float32)
        for phase in ('b1', 'b2'):
            n = _gbc_HIDDEN if phase == 'b1' or t > _gbc_SLICED_MAX_TOKENS else _gbc_UNPOOL_SLICE
            branch = (torch.randn((1, t, n), device=device, dtype=torch.float32), 1)
            scores = []
            for enabled in (False, True):

                def fn(enabled=enabled):
                    sums = torch.zeros((t, 4), device=device, dtype=torch.float32)
                    cnt = torch.zeros((t,), device=device, dtype=torch.int32)
                    if phase == 'b1':
                        return _gbc_boundary(branch, stream, state['alpha1'], state['beta1'], state['pre_ffn_w'], (sums, cnt), use_pdl=enabled)
                    return _gbc_boundary(branch, stream, state['alpha2'], state['beta2'], state['next_pre_attn_w'], (sums, cnt), postw=state['post_ffn_w'], use_pdl=enabled)
                try:
                    scores.append((enabled, _gbc__measure_graph(fn, flush, repeats=3)))
                except Exception as exc:
                    print(f'PDL_REJECT t={t} phase={phase} pdl={enabled} reason={str(exc).splitlines()[0][:100]}', flush=True)
            if not scores:
                plan[t, phase] = False
                print(f'PDL_TUNE t={t} phase={phase} fallback=False', flush=True)
            else:
                best = min(scores, key=lambda z: z[1])
                plan[t, phase] = best[0]
                print(f'PDL_TUNE t={t} phase={phase} best={best[0]} us={best[1]:.2f} all={scores}', flush=True)
    return plan

@triton.jit
def _gbc__publish_fence():
    tl.inline_asm_elementwise('fence.acq_rel.gpu; mov.u32 $0, $1;', '=r,r', [tl.arange(0, 128)], dtype=tl.int32, is_pure=False, pack=1)
    tl.debug_barrier()

@triton.jit
def _gbc__o_stats_kernel(X, W, STREAM, ALPHA, BETA, A, STATS, COEFF, CNT, T: tl.constexpr, TB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, USE_PDL: tl.constexpr):
    pn = tl.program_id(0)
    pt = tl.program_id(1)
    _gbc__pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    t = pt * TB + tl.arange(0, TB)
    ti = tl.arange(0, TB)
    k = tl.arange(0, BK)
    alpha = tl.load(ALPHA + n)
    beta = tl.load(BETA + n)
    lines = tl.arange(0, 1024 // 64)
    _gbc__l2_prefetch(W + n[:, None] * 1024 + lines[None, :] * 64)
    _gbc__pdl_wait(USE_PDL)
    acc = tl.zeros((BN, TB), tl.float32)
    for off in range(0, 1024, BK):
        w = tl.load(W + n[:, None] * 1024 + off + k[None, :])
        x = tl.load(X + t[:, None] * 1024 + off + k[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    a = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(A + t[None, :] * 8192 + n[:, None], a.to(tl.bfloat16), mask=t[None, :] < T)
    s = tl.load(STREAM + t[None, :] * 8192 + n[:, None], mask=t[None, :] < T, other=0.0)
    ast = alpha[:, None] * s
    bb = beta[:, None] * a
    NTILES: tl.constexpr = 8192 // BN
    base = pt * 4 * NTILES * TB
    tl.store(STATS + base + (0 * NTILES + pn) * TB + ti, tl.sum(a * a, 0))
    tl.store(STATS + base + (1 * NTILES + pn) * TB + ti, tl.sum(ast * ast, 0))
    tl.store(STATS + base + (2 * NTILES + pn) * TB + ti, tl.sum(ast * bb, 0))
    tl.store(STATS + base + (3 * NTILES + pn) * TB + ti, tl.sum(bb * bb, 0))
    _gbc__publish_fence()
    previous = tl.atomic_add(CNT + pt, 1, sem='acq_rel')
    if previous == NTILES - 1:
        r = tl.arange(0, NTILES)
        sb = tl.sum(tl.load(STATS + base + (0 * NTILES + r[:, None]) * TB + ti[None, :]), 0)
        sa = tl.sum(tl.load(STATS + base + (1 * NTILES + r[:, None]) * TB + ti[None, :]), 0)
        sab = tl.sum(tl.load(STATS + base + (2 * NTILES + r[:, None]) * TB + ti[None, :]), 0)
        sbb = tl.sum(tl.load(STATS + base + (3 * NTILES + r[:, None]) * TB + ti[None, :]), 0)
        rb = tl.rsqrt(sb / 8192 + 1e-08)
        rh = tl.rsqrt((sa + 2 * rb * sab + rb * rb * sbb) / 8192 + 1e-05)
        tl.store(COEFF + t, rb, mask=t < T)
        tl.store(COEFF + T + t, rh, mask=t < T)

@triton.jit
def _gbc__pool_boundary_kernel(A, COEFF, STREAM, STREAM1, ALPHA, BETA, PREW, W, P, T: tl.constexpr, TB: tl.constexpr, BN: tl.constexpr, KS: tl.constexpr, BK: tl.constexpr, USE_PDL: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    _gbc__pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    t = pt * TB + tl.arange(0, TB)
    k = tl.arange(0, BK)
    k0 = ps * KS
    lines = tl.arange(0, KS // 64)
    _gbc__l2_prefetch(W + n[:, None] * 8192 + k0 + lines[None, :] * 64)
    _gbc__pdl_wait(USE_PDL)
    rb = tl.load(COEFF + t, mask=t < T, other=0.0)
    rh = tl.load(COEFF + T + t, mask=t < T, other=0.0)
    acc = tl.zeros((BN, TB), tl.float32)
    for off in range(k0, k0 + KS, BK):
        kk = off + k
        alpha = tl.load(ALPHA + kk)
        beta = tl.load(BETA + kk)
        prew = tl.load(PREW + kk)
        a = tl.load(A + t[:, None] * 8192 + kk[None, :], mask=t[:, None] < T, other=0.0).to(tl.float32)
        s = tl.load(STREAM + t[:, None] * 8192 + kk[None, :], mask=t[:, None] < T, other=0.0)
        st = alpha[None, :] * s + beta[None, :] * a * rb[:, None]
        x = (st * rh[:, None] * prew[None, :]).to(tl.bfloat16)
        if pn == 0:
            tl.store(STREAM1 + t[:, None] * 8192 + kk[None, :], st, mask=t[:, None] < T)
        w = tl.load(W + n[:, None] * 8192 + kk[None, :])
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * T * 4608 + t[None, :] * 4608 + n[:, None], acc, mask=t[None, :] < T)

def _gbc_o_and_pool(attn, stream, state, counter, use_pdl=True):
    t = attn.shape[0]
    dev = attn.device
    tb = min(32, max(16, triton.next_power_of_2(t)))
    bn = 64
    tiles = triton.cdiv(t, tb)
    a = torch.empty((t, 8192), device=dev, dtype=torch.bfloat16)
    stats = torch.empty((tiles, 4, 8192 // bn, tb), device=dev, dtype=torch.float32)
    coeff = torch.empty((2, t), device=dev, dtype=torch.float32)
    stream1 = torch.empty_like(stream)
    pr = torch.empty((4, t, 4608), device=dev, dtype=torch.float32)
    _gbc__launch(_gbc__o_stats_kernel, (8192 // bn, tiles), attn, state['o'], stream, state['alpha1'], state['beta1'], a, stats, coeff, counter, T=t, TB=tb, BN=bn, BK=128, num_warps=4, num_stages=2, use_pdl=use_pdl)
    _gbc__launch(_gbc__pool_boundary_kernel, (4608 // 64, 4, tiles), a, coeff, stream, stream1, state['alpha1'], state['beta1'], state['pre_ffn_w'], state['pool_router'], pr, T=t, TB=tb, BN=64, KS=2048, BK=128, num_warps=4, num_stages=2, use_pdl=use_pdl)
    return ((pr, 4), stream1)

def _gbc__prefix_variant(attn, stream, state, data, kind, use_pdl):
    t = attn.shape[0]
    dev = attn.device
    if kind == 'fused':
        counter = torch.zeros((triton.cdiv(t, 32),), device=dev, dtype=torch.int32)
        pr, stream1 = _gbc_o_and_pool(attn, stream, state, counter, use_pdl=use_pdl)
    else:
        sync = torch.zeros((15 * t,), device=dev, dtype=torch.int32)
        si = sync[:3 * t].view(3, t)
        sf = sync[3 * t:].view(torch.float32).view(3, t, 4)
        a = _gbc_project(attn, state['o'], 'o', state)
        h, stream1 = _gbc_boundary(a, stream, state['alpha1'], state['beta1'], state['pre_ffn_w'], (sf[0], si[0]), use_pdl=use_pdl)
        pr = _gbc_project(h, state['pool_router'], 'pool_router', state, out_dtype=torch.float32)
    sync2 = torch.zeros((15 * t,), device=dev, dtype=torch.int32)
    si2 = sync2[:3 * t].view(3, t)
    sf2 = sync2[3 * t:].view(torch.float32).view(3, t, 4)
    return _gbc_route_and_norm(pr, state, data, (sf2[1], si2[1])) + (stream1,)

def _gbc__projection_consumer(x, w, name, state, variant, out_dtype):
    """Time the projection with its real reduction/norm consumer."""
    p = _gbc__project_variant(x, w, name, state, variant, out_dtype)
    t = x.shape[0]
    if name == 'qkv_gate':
        return _gbc_qk_prep(p, state['_proj_data'], state, q_fp8=True)
    sync = (torch.zeros((t, 4), device=x.device), torch.zeros((t,), device=x.device, dtype=torch.int32))
    if name == 'pool_router':
        return _gbc_route_and_norm(p, state, state['_proj_data'], sync)
    if name == 'o':
        return _gbc_boundary(p, state['_proj_stream'], state['alpha1'], state['beta1'], state['pre_ffn_w'], sync)
    return _gbc_boundary(p, state['_proj_stream'], state['alpha2'], state['beta2'], state['next_pre_attn_w'], sync, postw=state['post_ffn_w'])

def _gbc__seed_v9_gemv_choices(state):
    _gbc__GEMV_CHOICES.clear()
    for t in (1, 8, 16, 32):
        for name in ('qkv_gate', 'o', 'pool_router', 'unpooler_slice', 'unpooler'):
            if name == 'unpooler_slice' and t > 8 or (name == 'unpooler' and t <= 8):
                continue
            n, k = state[name].shape
            if name == 'qkv_gate':
                choice = (32, 1024, 128, False) if t <= 16 else (64, 512, 128, False)
            elif name == 'pool_router':
                choice = (32, 1024, 128, False) if t <= 16 else (64, 512, 128, False)
            elif name == 'unpooler':
                choice = 'cublas'
            else:
                choice = (64, 512, 128, False)
            _gbc__GEMV_CHOICES[t, n, k] = choice

def _gbc__tune_prefix(state, flush, device):
    plan = {}
    for t in _gbc__TUNE_TOKENS:
        stream = torch.randn((t, _gbc_HIDDEN), device=device, dtype=torch.float32)
        attn = torch.randn((t, _gbc_Q_SIZE), device=device, dtype=torch.bfloat16)
        data = {'local_start': torch.zeros((), device=device, dtype=torch.int32), 'config': {'tokens': t}}
        pdl = state['pdl_plan'].get((t, 'b1'), True)
        regular = lambda: _gbc__prefix_variant(attn, stream, state, data, 'regular', pdl)
        ref = regular()
        scores = [(_gbc__measure_graph(regular, flush), ('regular', pdl))]
        for enabled in (False, True):
            fn = lambda enabled=enabled: _gbc__prefix_variant(attn, stream, state, data, 'fused', enabled)
            try:
                for seed in range(2):
                    if seed:
                        attn.copy_(torch.randn_like(attn))
                        stream.copy_(torch.randn_like(stream))
                        ref = regular()
                    got = fn()
                    for i, (a, b) in enumerate(zip(got, ref)):
                        if not a.is_floating_point():
                            if not torch.equal(a, b):
                                raise ValueError(f'route ids differ at field {i}')
                        elif not torch.isfinite(a).all() or float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)) > 0.01:
                            raise ValueError(f'prefix field {i} exceeds 1% control guard')
                scores.append((_gbc__measure_graph(fn, flush), ('fused', enabled)))
            except Exception as exc:
                print(f'PREFIX_REJECT t={t} pdl={enabled} reason={str(exc).splitlines()[0][:130]}', flush=True)
        us, choice = min(scores, key=lambda x: x[0])
        plan[t] = choice
        print(f'PREFIX_TUNE t={t} best={choice} us={us:.3f} all={scores}', flush=True)
    return plan

class _gbc__PgNative:

    def __init__(self, path):
        self.lib = ct.CDLL(path)
        P, I, S = (ct.c_void_p, ct.c_int, ct.c_size_t)
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
        count, heuristics = (ct.c_int(), ct.c_int())
        self.check(self.lib.all_lt_options(n, m, k, n if nn else k, nn, int(fp32), workspace, codes.data_ptr(), len(codes), ct.byref(count), ct.byref(heuristics)))
        return (codes[:count.value].clone(), heuristics.value)

    def run(self, x, w, out, workspace, codes, index, nn, fp32):
        if not 0 <= index < len(codes):
            raise ValueError('unprepared Lt ordinal')
        self.check(self.lib.all_lt_run(w.data_ptr(), x.data_ptr(), out.data_ptr(), workspace.data_ptr(), codes.data_ptr(), out.shape[1], x.shape[0], x.shape[1], w.stride(0), nn, int(fp32), index, 0, workspace.numel(), torch.cuda.current_stream(x.device).cuda_stream))
_gbc_PG_RAW = '\n\n#include <cuda_runtime.h>\n#include <cublasLt.h>\n#include <cstring>\n#include <string>\n#include <initializer_list>\nstatic thread_local std::string pg_error;\nextern "C" const char* lean_error_string(){return pg_error.c_str();}\nstatic int lean_fail(const char* msg,int code){pg_error=std::string(msg)+": "+std::to_string(code);return code ? code : -1;}\n#define LEAN_LT(expr) do {auto s=(expr);if(s!=CUBLAS_STATUS_SUCCESS)return lean_fail(#expr,(int)s);} while(0)\nstatic thread_local cublasLtHandle_t lean_lt_handle=nullptr;\nstatic int lean_lt_init(){if(!lean_lt_handle){LEAN_LT(cublasLtCreate(&lean_lt_handle));}return 0;}\nextern "C" int pg_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}\n// All returned cuBLASLt heuristics, plus checked explicit split-K configurations.\n#include <cstdlib>\nstruct AllLtStorage {void* rows=nullptr;void* algorithms=nullptr;~AllLtStorage(){std::free(rows);std::free(algorithms);}};\nstruct AllLtLayout {\n  cublasLtMatmulDesc_t op=nullptr;\n  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;\n  int init(int n,int m,int k,int stride,int nn,int kind) {\n    cudaDataType_t in=kind==2 ? CUDA_R_32F : CUDA_R_16BF;\n    cudaDataType_t out=kind ? CUDA_R_32F : CUDA_R_16BF;\n    LEAN_LT(cublasLtMatmulDescCreate(&op,kind==2 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,CUDA_R_32F));\n    cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;\n    LEAN_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));\n    LEAN_LT(cublasLtMatrixLayoutCreate(&a,in,nn ? n : k,nn ? k : n,stride));\n    LEAN_LT(cublasLtMatrixLayoutCreate(&b,in,k,m,k));\n    LEAN_LT(cublasLtMatrixLayoutCreate(&c,out,n,m,n));\n    return 0;\n  }\n  ~AllLtLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}\n};\nextern "C" int all_lt_options(int n,int m,int k,int stride,int nn,int kind,size_t workspace,\n                              void* codes,int capacity,int* count,int* heuristic_count) {\n  int err=lean_lt_init();if(err)return err;\n  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;\n  cublasLtMatmulPreference_t pref;LEAN_LT(cublasLtMatmulPreferenceCreate(&pref));\n  auto s=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));\n  if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt preference",(int)s);}\n  int requested=64,got=0;\n  AllLtStorage storage;\n  storage.rows=std::calloc(capacity,sizeof(cublasLtMatmulHeuristicResult_t));\n  storage.algorithms=std::calloc(capacity,sizeof(cublasLtMatmulAlgo_t));\n  if(!storage.rows || !storage.algorithms){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt host storage",-1);}\n  auto rows=(cublasLtMatmulHeuristicResult_t*)storage.rows;\n  auto valid=(cublasLtMatmulAlgo_t*)storage.algorithms;\n  int valid_count=0;bool overflow=false;\n  do {\n\n    s=cublasLtMatmulAlgoGetHeuristic(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,pref,requested,rows,&got);\n    if(s!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristics",(int)s);}\n    if(got<requested)break;\n    requested*=2;\n    if(requested>capacity){cublasLtMatmulPreferenceDestroy(pref);return lean_fail("all Lt heuristic capacity",-1);}\n  }while(true);\n  cublasLtMatmulPreferenceDestroy(pref);\n\n  auto add=[&](const cublasLtMatmulAlgo_t& algo) {\n    for(int i=0;i<valid_count;++i)if(std::memcmp(valid+i,&algo,sizeof(algo))==0)return;\n    cublasLtMatmulHeuristicResult_t checked={};\n    auto status=cublasLtMatmulAlgoCheck(lean_lt_handle,l.op,l.a,l.b,l.c,l.c,&algo,&checked);\n    if(status==CUBLAS_STATUS_SUCCESS && checked.state==CUBLAS_STATUS_SUCCESS && checked.workspaceSize<=workspace)\n      {if(valid_count==capacity)overflow=true;else valid[valid_count++]=algo;}\n  };\n  for(int i=0;i<got;++i)if(rows[i].state==CUBLAS_STATUS_SUCCESS && rows[i].workspaceSize<=workspace)add(rows[i].algo);\n  *heuristic_count=valid_count;\n  // Preserve every returned heuristic; add supported swizzles and FP32 split-K reductions.\n  const int base_count=valid_count;\n  for(int bi=0;bi<base_count;++bi) {\n    const auto base=valid[bi];\n    int32_t split_support=0;uint32_t mask=0,swizzle=0;size_t written=0;\n    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,&split_support,sizeof(split_support),&written));\n    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,&mask,sizeof(mask),&written));\n    LEAN_LT(cublasLtMatmulAlgoCapGetAttribute(&base,CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,&swizzle,sizeof(swizzle),&written));\n    for(uint32_t sw=0;sw<=swizzle;++sw) {\n      auto plain=base;\n      if(cublasLtMatmulAlgoConfigSetAttribute(&plain,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,&sw,sizeof(sw))!=CUBLAS_STATUS_SUCCESS)continue;\n      add(plain);\n      if(split_support && (mask&CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {\n        for(uint32_t split:{2u,4u,8u,16u,32u}) {\n          auto algo=plain;uint32_t scheme=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;\n          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS)continue;\n          if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&scheme,sizeof(scheme))!=CUBLAS_STATUS_SUCCESS)continue;\n          add(algo);\n        }\n      }\n    }\n  }\n  if(overflow)return lean_fail("all Lt option capacity",-1);\n  for(int i=0;i<valid_count;++i)std::memcpy((char*)codes+i*sizeof(cublasLtMatmulAlgo_t),&valid[i],sizeof(cublasLtMatmulAlgo_t));\n  *count=valid_count;return 0;\n}\nextern "C" int all_lt_run(const void* w,const void* x,void* out,void* workspace,const void* codes,\n                          int n,int m,int k,int stride,int nn,int kind,int index,int accumulate,\n                          size_t workspace_bytes,void* stream) {\n  int err=lean_lt_init();if(err)return err;\n  AllLtLayout l;err=l.init(n,m,k,stride,nn,kind);if(err)return err;\n  cublasLtMatmulAlgo_t algo;std::memcpy(&algo,(const char*)codes+index*sizeof(algo),sizeof(algo));\n  float alpha=1.f,beta=accumulate ? 1.f : 0.f;\n  auto status=cublasLtMatmul(lean_lt_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&algo,\n                            workspace,workspace_bytes,(cudaStream_t)stream);\n  return status==CUBLAS_STATUS_SUCCESS ? 0 : lean_fail("all Lt run",(int)status);\n}\n\n'
_gbc__MEASURED_LT = {(256, 'pool_router'): (0, 0), (256, 'qkv_gate'): (0, 0), (256, 'unpooler'): (0, 0), (128, 'pool_router'): (0, 0), (128, 'qkv_gate'): (1, 23), (128, 'unpooler'): (1, 0), (128, 'o'): (1, 0)}

def _gbc__tune_measured_lt(state, flush, device):
    if not _gbc__fa3_on_device(device):
        return
    import hashlib
    import statistics
    major, minor = torch.cuda.get_device_capability(device)
    arch = '90a' if major == 9 else f'{major}{minor}'
    tag = hashlib.sha256(_gbc_PG_RAW.encode()).hexdigest()[:12]
    path = load_inline(name=f'muse_decode_measured_lt_{tag}_{arch}', cpp_sources='', cuda_sources=_gbc_PG_RAW, functions=None, is_python_module=False, no_implicit_headers=True, extra_cuda_cflags=['-O3', '-std=c++20', f'-gencode=arch=compute_{arch},code=sm_{arch}'], extra_ldflags=['-lcublasLt'], verbose=False)
    bank = {'mod': _gbc__PgNative(path), 'layouts': {}, 'workspace': torch.empty(4 << 30, device=device, dtype=torch.uint8)}
    state['measured_lt'] = bank
    for (t, name), (nn, ordinal) in _gbc__MEASURED_LT.items():
        w = state[name]
        n, k = w.shape
        dtype = torch.float32 if name == 'pool_router' else torch.bfloat16
        x = torch.randn((t, k), device=device, dtype=torch.bfloat16)
        codes, _ = bank['mod'].options(t, n, k, nn, dtype == torch.float32, bank['workspace'].numel())
        if ordinal >= len(codes):
            print(f'MEASURED_LT_REJECT t={t} name={name} nn={nn} ordinal={ordinal} unavailable', flush=True)
            continue
        key = (t, name, nn)
        bank['layouts'][key] = (w.T.contiguous() if nn else w, codes)
        choice = ('short_lt', nn, ordinal)
        control = state['proj_plan'][t, name]
        try:
            for _ in range(3):
                x.copy_(torch.randn_like(x))
                actual = _gbc__materialize(_gbc__project_variant(x, w, name, state, choice, dtype)).float()
                want = torch.mm(x.float(), w.float().T)
                if not torch.isfinite(actual).all() or float((actual - want).norm() / want.norm().clamp_min(1e-12)) > 0.01:
                    raise ValueError('measured Lt recipe exceeds 1% FP32 guard')
            state['_proj_stream'] = torch.randn((t, _gbc_HIDDEN), device=device, dtype=torch.float32)
            state['_proj_data'] = {'positions': torch.arange(t, device=device, dtype=torch.int64), 'cache_loc': torch.arange(t, device=device, dtype=torch.int64), 'k_cache': torch.empty((t, 1, _gbc_HKV, _gbc_HD), device=device, dtype=_gbc_FP8), 'v_cache': torch.empty((t, 1, _gbc_HKV, _gbc_HD), device=device, dtype=_gbc_FP8), 'k_scale': torch.ones((), device=device), 'v_scale': torch.ones((), device=device), 'local_start': torch.zeros((), device=device, dtype=torch.int32), 'config': dict(tokens=t, rope_theta=_gbc_ROPE_THETA, qk_scale_factor=5.4730064863875)}
            old = lambda: _gbc__projection_consumer(x, w, name, state, control, dtype)
            new = lambda: _gbc__projection_consumer(x, w, name, state, choice, dtype)
            ratios = []
            for repeat in range(4):
                x.copy_(torch.randn_like(x))
                if repeat % 2:
                    after = _gbc__measure_graph(new, flush)
                    before = _gbc__measure_graph(old, flush)
                else:
                    before = _gbc__measure_graph(old, flush)
                    after = _gbc__measure_graph(new, flush)
                ratios.append(after / before)
            ratio = statistics.median(ratios)
            if ratio < 1.0:
                state['proj_plan'][t, name] = choice
            print(f"MEASURED_LT t={t} name={name} recipe={choice} paired={ratio:.5f} best={state['proj_plan'][t, name]}", flush=True)
        except Exception as exc:
            print(f'MEASURED_LT_REJECT t={t} name={name} reason={str(exc).splitlines()[0][:120]}', flush=True)
    keep = {(t, name, choice[1]) for (t, name), choice in state['proj_plan'].items() if isinstance(choice, tuple) and choice[0] == 'short_lt'}
    bank['layouts'] = {key: value for key, value in bank['layouts'].items() if key in keep}
    if not keep:
        state.pop('measured_lt')

def lab(deadline,weights,state=None):
    started=time.monotonic();deadline=min(deadline,started+230)
    st=gb_state(weights);dev=weights['o'].device
    st['inv_freq']=(1./(500000.**(torch.arange(0,64,2,device=dev).float()/64))).contiguous()
    _gbc__seed_v9_gemv_choices(st)
    flush=torch.zeros((1<<30)//4,device=dev,dtype=torch.float32)
    lines=['GB_META current=cand_r3_best_812f5f36 dense=proj_host norm_gates=synthetic_model_distribution cold1GiB graph paired20 fresh3; route=sigmoid_biased_top8; BF16_boundaries_preserved']
    # Original best's own projection, Lt, PDL and prefix selection, without MoE/attention setup.
    try:
        st['proj_plan']=_gbc__tune_projections(st,flush,dev)
        _gbc__tune_measured_lt(st,flush,dev)
        st['pdl_plan']=_gbc__tune_pdl(st,flush,dev)
        st['prefix_plan']=_gbc__tune_prefix(st,flush,dev)
    except Exception as exc:
        # Missing optional incumbent setup cannot produce a misleading comparison.
        return lines+[f'GB_REJECT control_setup {type(exc).__name__}:{str(exc)[-180:]}']
    callbacks=gb_callbacks(st,_gbc_project,_gbc_boundary,_gbc_route_and_norm,_gbc_o_and_pool)
    lines.append(f'GB_CONTROL_SETUP elapsed_s={time.monotonic()-started:.3f}')
    for m in GB_TOKENS:
        names=('o','pool_router','unpooler_slice' if m<=8 else 'unpooler')
        lines.append(f'GB_CONTROL {m} proj={[st["proj_plan"].get((m,n),"custom") for n in names]} prefix={st["prefix_plan"].get(m)}')
    lines+=gb_prepare(st,callbacks,deadline)
    lines.append(f'GB_LAB_END elapsed_s={time.monotonic()-started:.3f}')
    errors=[r for r in lines if r.startswith('GB_REJECT')]
    compact=[r for r in lines if not r.startswith('GB_REJECT')]
    compact += [r[:180] for r in errors[:6]]
    if errors:compact.append(f'GB_REJECT_COUNT {len(errors)} sampled={min(6,len(errors))}')
    return compact

# ---- h200_lab wrapper: payload verify_projection_glue sha256 34143ac6134efe1f on base f3786341611f4f1f ----
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
        print("LAB_BEGIN verify_projection_glue 34143ac6134efe1f", flush=True)
        for _l in _out:
            print(_l, flush=True)
        print(f"LAB_END lab_s={_t.monotonic() - _t0:.1f} prepare_s={_t.monotonic() - _start:.1f}", flush=True)
    return state
