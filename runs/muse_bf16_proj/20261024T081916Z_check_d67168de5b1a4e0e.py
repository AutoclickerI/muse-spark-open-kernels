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

# ---- h200_lab payload prefill_dense_cutlass.py ----
import ctypes as ct
import torch,time,statistics,json,hashlib,os
import triton
import triton.language as tl
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

VGC_SOURCE = '// Hopper CUTLASS 3.x collective API, shipped in the judge image.\n// __global__ entry point: cutlass::device_kernel<Kernel> below.\n#include <cuda_runtime.h>\n#include <cutlass/cutlass.h>\n#include <cutlass/bfloat16.h>\n#include <cutlass/gemm/collective/collective_builder.hpp>\n#include <cutlass/epilogue/collective/collective_builder.hpp>\n#include <cutlass/gemm/device/gemm_universal_adapter.h>\n#include <cutlass/gemm/kernel/gemm_universal.hpp>\n#include <cutlass/gemm/kernel/tile_scheduler.hpp>\n#include <cute/tensor.hpp>\n#include <cstdint>\n#include <exception>\n#include <string>\n#include <stdexcept>\n#include <type_traits>\nusing namespace cute;\nnamespace vgcut {\nusing BF=cutlass::bfloat16_t;\nusing Row=cutlass::layout::RowMajor;\nusing Col=cutlass::layout::ColumnMajor;\nusing Mode=cutlass::gemm::kernel::detail::DecompositionMode;\nusing Reduction=cutlass::gemm::kernel::detail::ReductionMode;\nusing Raster=cutlass::gemm::kernel::detail::RasterOrderOptions;\nstatic thread_local std::string error;\ninline void check(cutlass::Status s) {\n  if (s!=cutlass::Status::kSuccess) throw std::runtime_error(cutlassGetStatusString(s));\n}\nstruct Base {\n  size_t bytes=0; int stages=0,threads=0,smem=0,gx=0,gy=0,gz=0;\n  virtual void run(void*,const void*,void*,void*,cudaStream_t)=0;\n  virtual ~Base()=default;\n};\ntemplate<int BN,int BK,int CM,int CN,int NS,bool Swap,bool FP32>\nstruct Plan:Base {\n  using D=std::conditional_t<FP32,float,BF>;\n  using LD=std::conditional_t<Swap,Col,Row>;\n  using Tile=Shape<_128,Int<BN>,Int<BK>>;\n  using Cluster=Shape<Int<CM>,Int<CN>,_1>;\n  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<\n    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,\n    cutlass::epilogue::collective::EpilogueTileAuto,float,float,\n    D,LD,128/cutlass::sizeof_bits<D>::value,D,LD,128/cutlass::sizeof_bits<D>::value,\n    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;\n  using Stages=std::conditional_t<NS==0,\n    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,\n    cutlass::gemm::collective::StageCount<NS>>;\n  using Main=typename cutlass::gemm::collective::CollectiveBuilder<\n    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,\n    BF,Row,8,BF,Col,8,float,Tile,Cluster,Stages,\n    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;\n  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,cutlass::gemm::StreamKScheduler>;\n  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;\n  typename Gemm::Arguments args{};\n  Gemm gemm;\n  Plan(int m,int n,int k,int splits,int mode,int sms,int raster,int swizzle,int reduction) {\n    int mm=Swap?n:m,nn=Swap?m:n;\n    args.mode=cutlass::gemm::GemmUniversalMode::kGemm;\n    args.problem_shape=make_shape(mm,nn,k,1);\n    args.mainloop.dA=make_stride(int64_t(k),_1{},int64_t(0));\n    args.mainloop.dB=make_stride(int64_t(k),_1{},int64_t(0));\n    if constexpr(Swap) {\n      args.epilogue.dC=make_stride(_1{},int64_t(n),int64_t(0));\n      args.epilogue.dD=make_stride(_1{},int64_t(n),int64_t(0));\n    } else {\n      args.epilogue.dC=make_stride(int64_t(n),_1{},int64_t(0));\n      args.epilogue.dD=make_stride(int64_t(n),_1{},int64_t(0));\n    }\n    args.epilogue.thread.alpha=1.f;\n    args.epilogue.thread.beta=0.f;\n    cudaGetDevice(&args.hw_info.device_id);\n    args.hw_info.sm_count=sms;\n    args.scheduler.splits=splits;\n    args.scheduler.decomposition_mode=static_cast<Mode>(mode);\n    args.scheduler.reduction_mode=static_cast<Reduction>(reduction);\n    args.scheduler.raster_order=static_cast<Raster>(raster);\n    args.scheduler.max_swizzle_size=swizzle;\n    bytes=Gemm::get_workspace_size(args);\n    stages=Gemm::kStages;threads=Kernel::MaxThreadsPerBlock;smem=Kernel::SharedStorageSize;\n    cudaError_t err=cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);\n    if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));\n  }\n  void run(void* x,const void* w,void* out,void* workspace,cudaStream_t stream)override {\n    args.mainloop.ptr_A=reinterpret_cast<const BF*>(Swap?w:x);\n    args.mainloop.ptr_B=reinterpret_cast<const BF*>(Swap?x:w);\n    args.epilogue.ptr_C=nullptr;\n    args.epilogue.ptr_D=reinterpret_cast<D*>(out);\n    check(Gemm::can_implement(args));\n    // Include required reduction-counter reset in EVERY timed invocation / graph replay.\n    check(Kernel::initialize_workspace(args,workspace,stream));\n    check(gemm.update(args,workspace));\n    check(gemm.run(stream));\n  }\n};\ntemplate<bool F>Base* create(int id,int m,int n,int k,int sk,int mode,int sms,int raster,int sw,int red) {\n#define V(ID,N,K,CM,CN,S,SWAP) case ID:return new Plan<N,K,CM,CN,S,SWAP,F>(m,n,k,sk,mode,sms,raster,sw,red)\n  switch(id){\n    V(0,128,64,1,1,0,false);\n    V(1,128,64,1,2,0,false);\n    V(2,64,64,1,1,0,false);\n    V(3,256,64,1,1,0,false);\n    V(4,128,128,1,1,0,false);\n    V(5,128,64,1,1,3,false);\n    V(6,128,64,1,1,0,true);\n    V(7,64,64,2,1,0,true);\n    default:throw std::runtime_error("unknown CUTLASS tile");\n  }\n#undef V\n}\n}\nextern "C" const char* vgc_error(){return vgcut::error.c_str();}\nextern "C" void* vgc_create(int id,int fp32,int m,int n,int k,int sk,int mode,int sms,int raster,int sw,int red){\n  try{return fp32?vgcut::create<true>(id,m,n,k,sk,mode,sms,raster,sw,red):vgcut::create<false>(id,m,n,k,sk,mode,sms,raster,sw,red);}\n  catch(const std::exception& e){vgcut::error=e.what();return nullptr;}\n}\nextern "C" size_t vgc_workspace(void* p){return reinterpret_cast<vgcut::Base*>(p)->bytes;}\nextern "C" int vgc_info(void* p,int index){auto* q=reinterpret_cast<vgcut::Base*>(p);return index==0?q->stages:index==1?q->threads:q->smem;}\nextern "C" void vgc_destroy(void* p){delete reinterpret_cast<vgcut::Base*>(p);}\nextern "C" int vgc_run(void* p,void* x,const void* w,void* y,void* ws,cudaStream_t stream){\n  try{reinterpret_cast<vgcut::Base*>(p)->run(x,w,y,ws,stream);return 0;}\n  catch(const std::exception& e){vgcut::error=e.what();return -1;}\n}\n'

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

# H200-positive recipes only. Empty until the matching lab receipt lands.
PFC_RECIPES = {}
PFC_SHAPES = {'pooler': (4096,8192), 'unpooler': (8192,4096),
              'qkv': (2304,8192), 'o': (8192,1024), 'router_cat': (512,8192)}


def pfc_choices(m,n,k,sms=132):
    # Cover every tile/multicast family before refinements. Large M favors DP;
    # narrow N also gets explicit split-K16/32, with all reductions timed.
    for mode,split in ((1,1),(3,1),(2,2),(2,4),(2,8),(0,1)):
        for tile in range(len(VGC_TILES)):
            yield (tile,mode,split,sms,0,1,0)
    for recipe in vgc_choices(m,n,k,sms):
        if recipe[4:]!=(0,1,0) or recipe[3]!=sms or recipe[2]==16:
            yield recipe
    if n<=2304:
        for tile in range(len(VGC_TILES)):
            yield (tile,2,32,sms,0,1,0)


def pfc_check(out,reference,router=False):
    a,b=out.float(),reference.float()
    diff=a-b
    rel=float(diff.norm()/b.norm().clamp_min(1e-12))
    norms=b.norm(dim=1)
    row=float((diff.norm(dim=1)/norms.clamp_min(norms.mean()*.1).clamp_min(1e-12)).max())
    if not bool(torch.isfinite(a).all()) or rel>(5e-5 if router else .008) or row>(2e-4 if router else .016):
        raise ValueError(f'FP32 gate rel={rel:.6g} worst_row={row:.6g}')
    return rel,row


def pfc_pair(old,new,x,banks,flush,repeats=20,deadline=float('inf')):
    samples=[[],[]]
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for r in range(repeats):
        if time.monotonic()>=deadline:raise TimeoutError('CUTLASS paired timing budget')
        x.copy_(banks[(r//2)%len(banks)])
        for side in ((0,1) if r%2==0 else (1,0)):
            flush.sum()
            a.record();(old if side==0 else new).graph.replay();b.record();b.synchronize()
            samples[side].append(float(a.elapsed_time(b))*1000)
    ratios=[(samples[1][j]*samples[1][j+1]/samples[0][j]/samples[0][j+1])**.5
            for j in range(0,repeats,2)]
    return statistics.median(samples[0]),statistics.median(samples[1]),statistics.median(ratios)


def pfc_banks(m,k,device):
    generator=torch.Generator(device=device).manual_seed(479183+m+k)
    banks=[torch.randn(m,k,device=device,dtype=torch.bfloat16,generator=generator) for _ in range(2)]
    probe=torch.zeros_like(banks[0]);probe[:,127%k]=.5;banks.append(probe)
    return banks


def pfc_reference(x,w):
    # Independent full-FP32 oracle, even if a caller enabled TF32 globally.
    previous=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:return x.float()@w.float().T
    finally:torch.backends.cuda.matmul.allow_tf32=previous


def pfc_prepare(st,deadline,recipes=None):
    """Re-gate and retime only receipt-approved recipes; reject, never raise.

    Runs after the inherited stage tuners and before the dense plan/group tuner.
    Thus the latter sees the selected CUTLASS product as another incumbent, and
    includes routing/hi+lo consumption when it compares mixed/separate groups.
    """
    st['_pfc_plans']={};st['_pfc_notes']=[]
    rows=st['_pfc_notes']
    recipes=PFC_RECIPES if recipes is None else recipes
    keys=[(m,key) for m in (16384,753,3012) for key in PFC_SHAPES if recipes.get((m,key))]
    if not keys:
        rows.append('PFC_INSTALL no H200-positive recipes; incumbent retained')
        return
    try:
        if deadline-time.monotonic()<100:
            rows.append('PFC_INSTALL_BUDGET compile; incumbent retained');return
        module=vgc_module()
        device=st['qkv'].device
        flush=torch.zeros((1<<30)//4,device=device,dtype=torch.float32)
        for pos,(m,key) in enumerate(keys):
            now=time.monotonic();stop=now+max(0,deadline-now-2)/(len(keys)-pos)
            if now>=deadline-2:rows.append(f'PFC_INSTALL_BUDGET {m} {key}');continue
            winner=None;old=None;best=1.;old_choice=st['dense_choices'].get((key,m),('cublas',))
            try:
                w=st[key];n,k=w.shape;dtype=torch.float32 if key=='router_cat' else torch.bfloat16
                if tuple(w.shape)!=PFC_SHAPES[key]:raise ValueError('unexpected prefill projection shape')
                banks=pfc_banks(m,k,device);x=banks[0].clone()
                refs=[pfc_reference(fresh,w) for fresh in banks]
                old=_PgGraph(lambda:_score_dense_run(x,st,key,old_choice))
                for fresh,ref in zip(banks,refs):
                    x.copy_(fresh);old.graph.replay();pfc_check(old.out,ref,key=='router_cat')
                for recipe in recipes[m,key]:
                    if time.monotonic()>=min(stop,deadline-2):break
                    plan=None;keep=False
                    try:
                        plan=module.plan(m,n,k,dtype,recipe,device)
                        def run():
                            out=torch.empty((m,n),device=device,dtype=dtype)
                            return plan.run(x,w,out)
                        with _PgGraph(run) as graph:
                            for fresh,ref in zip(banks,refs):
                                x.copy_(fresh);old.graph.replay();graph.graph.replay()
                                pfc_check(graph.out,ref,key=='router_cat')
                                pfc_check(graph.out,old.out,key=='router_cat')
                            t0,t1,ratio=pfc_pair(old,graph,x,banks,flush,20,min(stop,deadline-2))
                            for fresh,ref in zip(banks,refs):
                                x.copy_(fresh);graph.graph.replay();pfc_check(graph.out,ref,key=='router_cat')
                        rows.append(f'PFC_INSTALL_TRIAL {m} {key} c={recipe} us={t0:.3f}/{t1:.3f} r={ratio:.5f}')
                        if ratio<best:
                            if winner is not None:winner.close()
                            winner=plan;best=ratio;keep=True
                    except Exception as exc:
                        rows.append(f'PFC_INSTALL_REJECT {m} {key} {type(exc).__name__}:{str(exc)[:130]}')
                    finally:
                        if plan is not None and not keep:plan.close()
                if winner is not None:
                    recipe=tuple(winner.recipe)
                    st['_pfc_plans'][key,m,recipe]=winner
                    st['dense_choices'][key,m]=('cutlass',recipe)
                rows.append(f'PFC_INSTALL_PICK {m} {key} c={winner.recipe if winner else None} r={best:.5f}')
            except Exception as exc:
                if winner is not None:winner.close()
                rows.append(f'PFC_INSTALL_REJECT shape {m} {key} {type(exc).__name__}:{str(exc)[:130]}')
            finally:
                if old is not None:old.close()
    except Exception as exc:
        rows.append(f'PFC_INSTALL_REJECT setup {type(exc).__name__}:{str(exc)[:160]}')


def pfc_run(x,st,key,recipe):
    plan=st['_pfc_plans'][key,x.shape[0],tuple(recipe)]
    out=torch.empty((x.shape[0],st[key].shape[0]),device=x.device,dtype=plan.dtype)
    return plan.run(x,st[key],out)

@triton.jit
def _dense_persistent(A,B,C,M,N:tl.constexpr,K:tl.constexpr,
                      BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GRID:tl.constexpr):
    mi=tl.cdiv(M,BM);ni=tl.cdiv(N,BN)
    for tile in range(tl.program_id(0),mi*ni,GRID):
        # Group eight row tiles for weight reuse, including the partial last group.
        group=tile//(8*ni);first=group*8;gs=tl.minimum(mi-first,8)
        within=tile%(8*ni);pm=first+within%gs;pn=within//gs
        rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
        acc=tl.zeros((BM,BN),tl.float32)
        for kg in range(tl.cdiv(K,BK)):
            kk=kg*BK+rk
            av=tl.load(A+rm[:,None].to(tl.int64)*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0)
            bv=tl.load(B+rn[None,:].to(tl.int64)*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0)
            acc+=tl.dot(av,bv)
        tl.store(C+rm[:,None].to(tl.int64)*N+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<N))

_DENSE_TRITON={'triton64':(64,128,64,4,3),'triton128':(128,128,64,8,3),'triton256':(128,256,64,8,3)}
def _score_dense_run(x,st,key,choice):
    w=st[key];fp32=key=='router_cat';name=choice[0]
    if name=='cublas':return torch.mm(x,w.t(),**({'out_dtype':torch.float32} if fp32 else {}))
    if name=='cublas_nn':return torch.mm(x,st['dense_transposed'][key],**({'out_dtype':torch.float32} if fp32 else {}))
    out=torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=torch.float32 if fp32 else torch.bfloat16)
    if name=='lt':
        codes=st['lt_algorithms'][(key,x.shape[0])]
        st['grouped_mod'].lt_run(w,x,out,st['lt_workspace'],codes,choice[1])
    elif name=='dg':
        import deep_gemm
        deep_gemm.bf16_gemm_nt(x,w,out)
    else:
        bm,bn,bk,warps,stages=_DENSE_TRITON[name]
        _dense_persistent[(st['sms'],)](x,w,out,x.shape[0],N=w.shape[0],K=w.shape[1],BM=bm,BN=bn,BK=bk,
                                      GRID=st['sms'],num_warps=warps,num_stages=stages)
    return out

PFC_LT_SOURCE = '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cublasLt.h>\n#include <cstring>\nnamespace muse_lt {\nstatic cublasLtHandle_t handle() {\n  static cublasLtHandle_t h=[] { cublasLtHandle_t p; TORCH_CHECK(cublasLtCreate(&p)==CUBLAS_STATUS_SUCCESS,"cublasLtCreate"); return p; }();\n  return h;\n}\nstruct Layout {\n  cublasLtMatmulDesc_t op;\n  cublasLtMatrixLayout_t a,b,c;\n  Layout(const torch::Tensor& w,const torch::Tensor& x,const torch::Tensor& y) {\n    int64_t n=y.size(1),m=x.size(0),k=x.size(1);\n    TORCH_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"Lt desc");\n    cublasOperation_t ta=CUBLAS_OP_T;\n    TORCH_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))==CUBLAS_STATUS_SUCCESS,"Lt transpose");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,w.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt A layout");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,x.stride(0))==CUBLAS_STATUS_SUCCESS,"Lt B layout");\n    TORCH_CHECK(cublasLtMatrixLayoutCreate(&c,(y.scalar_type()==at::kFloat ? CUDA_R_32F : CUDA_R_16BF),n,m,n)==CUBLAS_STATUS_SUCCESS,"Lt C layout");\n  }\n  ~Layout() { cublasLtMatmulDescDestroy(op); cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c); }\n};\nconstexpr int CAP=32;\nconstexpr size_t WORK_BYTES=256*1024*1024;\n}\ntorch::Tensor lt_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out) {\n  using namespace muse_lt;\n  Layout layout(W,X,Out);\n  cublasLtMatmulPreference_t pref;\n  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS,"Lt preference");\n  size_t workspace=WORK_BYTES;\n  TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace))==CUBLAS_STATUS_SUCCESS,"Lt workspace preference");\n  cublasLtMatmulHeuristicResult_t results[CAP]; int count=0;\n  auto status=cublasLtMatmulAlgoGetHeuristic(handle(),layout.op,layout.a,layout.b,layout.c,layout.c,pref,CAP,results,&count);\n  cublasLtMatmulPreferenceDestroy(pref);\n  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt heuristics: ",int(status));\n  auto codes=torch::empty({count,(int64_t)sizeof(cublasLtMatmulAlgo_t)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU));\n  for(int i=0;i<count;++i) {\n    TORCH_CHECK(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=WORK_BYTES,"Lt invalid heuristic");\n    std::memcpy(codes.data_ptr<uint8_t>()+i*sizeof(cublasLtMatmulAlgo_t),&results[i].algo,sizeof(cublasLtMatmulAlgo_t));\n  }\n  return codes;\n}\nvoid lt_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index) {\n  using namespace muse_lt;\n  TORCH_CHECK(index>=0 && index<Algos.size(0) && Work.nbytes()>=WORK_BYTES,"Lt invalid algorithm/workspace");\n  Layout layout(W,X,Out);\n  cublasLtMatmulAlgo_t algo;\n  std::memcpy(&algo,Algos.data_ptr<uint8_t>()+index*sizeof(algo),sizeof(algo));\n  const float alpha=1.f,beta=0.f;\n  auto status=cublasLtMatmul(handle(),layout.op,&alpha,W.data_ptr(),layout.a,X.data_ptr(),layout.b,&beta,\n                            Out.data_ptr(),layout.c,Out.data_ptr(),layout.c,&algo,Work.data_ptr(),WORK_BYTES,\n                            at::cuda::getCurrentCUDAStream());\n  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Lt matmul: ",int(status));\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("lt_options",&lt_options);m.def("lt_run",&lt_run);}\n'
PFC_INCUMBENTS = {('pooler', 16384): ('lt', 1), ('unpooler', 16384): ('lt', 6), ('qkv', 16384): ('lt', 1), ('o', 16384): ('lt', 6), ('router_cat', 16384): ('lt', 0), ('pooler', 753): ('dg',), ('unpooler', 753): ('dg',), ('qkv', 753): ('cublas',), ('o', 753): ('cublas',), ('router_cat', 753): ('lt', 6), ('pooler', 3012): ('lt', 1), ('unpooler', 3012): ('cublas',), ('qkv', 3012): ('lt', 2), ('o', 3012): ('dg',), ('router_cat', 3012): ('cublas',)}

def pfc_lt_module():
    from torch.utils.cpp_extension import load_inline
    previous=os.environ.get('TORCH_CUDA_ARCH_LIST')
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
    try:
        return load_inline(name='prefill_exact_lt32_'+hashlib.sha256(PFC_LT_SOURCE.encode()).hexdigest()[:12],
            cpp_sources='',cuda_sources=PFC_LT_SOURCE,no_implicit_headers=True,
            extra_cuda_cflags=['-O3','-std=c++20'],extra_ldflags=['-lcublasLt'],verbose=False)
    finally:
        if previous is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
        else:os.environ['TORCH_CUDA_ARCH_LIST']=previous


def lab(deadline,weights,state=None):
    begin=time.monotonic()
    # Wrapper deadline is from BEFORE host prepare: never reset that clock.
    deadline=min(deadline,begin+270)
    rows=[];rejects=[]
    try:module=vgc_module();lt=pfc_lt_module()
    except Exception as exc:return [f'PFC_REJECT compile {type(exc).__name__}:{str(exc)[-350:]}']
    compile_s=time.monotonic()-begin
    device=weights['qkv_gate'].device
    hi=weights['router'].bfloat16();lo=(weights['router']-hi.float()).bfloat16()
    st={'qkv':weights['qkv_gate'],'o':weights['o'],'pooler':weights['pooler'],
        'unpooler':weights['unpooler'],'router_cat':torch.cat((hi,lo),0).contiguous()}
    st['dense_transposed']={key:st[key].T.contiguous() for key in PFC_SHAPES}
    st['sms']=torch.cuda.get_device_properties(device).multi_processor_count
    st['grouped_mod']=lt;st['lt_algorithms']={}
    st['lt_workspace']=torch.empty(256<<20,device=device,dtype=torch.uint8)
    flush=torch.zeros((1<<30)//4,device=device,dtype=torch.float32)
    rows.append(f'PFC_META compile_s={compile_s:.3f} sms={st["sms"]} incumbent=0d738799802340fd; exact Lt32/256MiB; cold1GiB, fresh3, paired20; TFLOPS=2MNK/us/1e6; peak=989')
    # Allocate time to every shape, with the largest absolute lever first.
    keys=[(m,key) for m in (16384,753,3012) for key in PFC_SHAPES]
    shares={(m,key):(2 if key in ('pooler','unpooler') else 1) for m,key in keys}
    left=sum(shares.values())
    for m,key in keys:
        now=time.monotonic();stop=now+max(0,deadline-now-3)*shares[m,key]/left
        left-=shares[m,key]
        if now>=deadline-3:rows.append(f'PFC_BUDGET {m} {key}');continue
        old=None;cu=None
        try:
            w=st[key];n,k=w.shape;router=key=='router_cat';dtype=torch.float32 if router else torch.bfloat16
            banks=pfc_banks(m,k,device);x=banks[0].clone()
            refs=[pfc_reference(fresh,w) for fresh in banks]
            router_refs=[pfc_reference(fresh,weights['router']) for fresh in banks] if router else None
            cu=_PgGraph(lambda:_score_dense_run(x,st,key,('cublas',)))
            choice=PFC_INCUMBENTS[key,m]
            if choice[0]=='lt':st['lt_algorithms'][key,m]=lt.lt_options(w,x,cu.out)
            old=_PgGraph(lambda:_score_dense_run(x,st,key,choice))
            for fresh,ref in zip(banks,refs):
                x.copy_(fresh);cu.graph.replay();old.graph.replay()
                pfc_check(cu.out,ref,router);pfc_check(old.out,ref,router)
            cu_us,inc_us,_=pfc_pair(cu,old,x,banks,flush,20,min(stop,deadline-2))
            attempts=0;bad=0;trials=[];tile_best={};mode_best={}
            coarse_stop=time.monotonic()+.72*max(0,stop-time.monotonic())
            for recipe in pfc_choices(m,n,k,st['sms']):
                if time.monotonic()>=coarse_stop:break
                plan=None;attempts+=1
                try:
                    plan=module.plan(m,n,k,dtype,recipe,device)
                    def run():
                        out=torch.empty((m,n),device=device,dtype=dtype)
                        return plan.run(x,w,out)
                    with _PgGraph(run) as graph:
                        error=0.;worst=0.
                        for i,(fresh,ref) in enumerate(zip(banks,refs)):
                            x.copy_(fresh);graph.graph.replay()
                            rel,row=pfc_check(graph.out,ref,router);error=max(error,rel);worst=max(worst,row)
                            if router:
                                pfc_check(graph.out[:,:256]+graph.out[:,256:],router_refs[i],True)
                        t0,t1,ratio=pfc_pair(old,graph,x,banks,flush,4,min(stop,deadline-2))
                    trials.append((ratio,recipe,error,worst,plan.workspace.numel(),plan.info))
                    for mapping,family in ((tile_best,recipe[0]),(mode_best,recipe[1])):
                        if family not in mapping or t1<mapping[family]:mapping[family]=t1
                except Exception as exc:
                    bad+=1
                    if len(rejects)<4:rejects.append(f'PFC_REJECT {m} {key} {recipe} {type(exc).__name__}:{str(exc).splitlines()[0][:100]}')
                finally:
                    if plan is not None:plan.close()
            finals=sorted(trials)[:2]
            for mode in (1,2,3):
                family=[row for row in trials if row[1][1]==mode]
                if family and min(family) not in finals:finals.append(min(family))
            best=None;final_rows=[]
            for _,recipe,error,worst,size,info in finals:
                if time.monotonic()>=min(stop,deadline-2):break
                plan=None
                try:
                    plan=module.plan(m,n,k,dtype,recipe,device)
                    def run_final():
                        out=torch.empty((m,n),device=device,dtype=dtype)
                        return plan.run(x,w,out)
                    with _PgGraph(run_final) as graph:
                        t0,t1,ratio=pfc_pair(old,graph,x,banks,flush,20,min(stop,deadline-2))
                        for i,(fresh,ref) in enumerate(zip(banks,refs)):
                            x.copy_(fresh);graph.graph.replay();pfc_check(graph.out,ref,router)
                            if router:pfc_check(graph.out[:,:256]+graph.out[:,256:],router_refs[i],True)
                    final_rows.append([list(recipe),round(t0,3),round(t1,3),round(ratio,5)])
                    if best is None or ratio<best[0]:best=(ratio,recipe,t0,t1,error,worst,size,info)
                except Exception as exc:
                    bad+=1
                    if len(rejects)<4:rejects.append(f'PFC_REJECT final {m} {key} {type(exc).__name__}:{str(exc)[:100]}')
                finally:
                    if plan is not None:plan.close()
            result={'m':m,'key':key,'cu':round(cu_us,3),'inc':round(inc_us,3),'c':choice,
                    'pick':list(best[1]) if best and best[0]<1 else None,
                    'best':([round(best[2],3),round(best[3],3),round(best[0],5),
                             round(2*m*n*k/best[3]/1e6,2),float(f'{best[4]:.3g}'),float(f'{best[5]:.3g}'),
                             best[6],list(best[7])] if best else None),
                    'tile_us':[round(tile_best[t],2) if t in tile_best else None for t in range(8)],
                    'mode_us':[round(mode_best[t],2) if t in mode_best else None for t in range(4)],
                    'finals':final_rows,'tried':attempts,'rejected':bad}
            rows.append('PFC_ROW '+json.dumps(result,separators=(',',':')))
        except Exception as exc:
            rows.append(f'PFC_REJECT shape {m} {key} {type(exc).__name__}:{str(exc)[:140]}')
        finally:
            if old is not None:old.close()
            if cu is not None:cu.close()
    # Preserve the 15 measurement rows within the wrapper's 9000-char output cap.
    rows.extend(rejects)
    rows.append(f'PFC_END elapsed_s={time.monotonic()-begin:.3f} left_s={deadline-time.monotonic():.3f}')
    return rows

# ---- h200_lab wrapper: payload prefill_dense_cutlass sha256 5320418352787009 on base f3786341611f4f1f ----
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
        print("LAB_BEGIN prefill_dense_cutlass 5320418352787009", flush=True)
        for _l in _out:
            print(_l, flush=True)
        print(f"LAB_END lab_s={_t.monotonic() - _t0:.1f} prepare_s={_t.monotonic() - _start:.1f}", flush=True)
    return state
