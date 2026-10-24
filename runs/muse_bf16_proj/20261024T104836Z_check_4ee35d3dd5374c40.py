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

# ---- h200_lab payload moe_vector_wgmma_seed.py ----
"""Grouped NVFP4/BF16 RS WGMMA. All weight conversion is inside the timed GEMMs.

Banks contain the original E2M1 nibbles and E4M3 bytes. Weights occupy MMA M;
routed tokens occupy N. A GPU prefix schedule splits hot experts along N.
"""
import functools

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl


MW_PRELUDE = r'''
#include <cuda_bf16.h>
#include <cutlass/bfloat16.h>
__device__ __forceinline__ cutlass::bfloat16_t mw_decode(unsigned q, unsigned s, int odd) {
  unsigned c=(q>>(4*odd))&15u, m=c&7u;
  // Two byte permutations implement the exact E2M1 -> BF16 lookup.
  unsigned lo=__byte_perm(0xc0800000u,0xc0804000u,m)&255u;
  unsigned hi=__byte_perm(0x3f3f3f00u,0x40404040u,m)&255u;
  unsigned vb=lo|(hi<<8)|((c&8u)<<12);
  unsigned sm=s&127u, sb;
  if(sm>=8u) sb=(120u<<7)+(sm<<4);
  else if(sm) { unsigned k=31u-__clz(sm);sb=((118u+k)<<7)|((sm<<(7u-k))&127u); }
  else sb=0;
  sb|=(s&128u)<<8;
  return cutlass::bfloat16_t::bitcast(__bfloat16_as_ushort(
                __hmul(__ushort_as_bfloat16((unsigned short)vb),
                       __ushort_as_bfloat16((unsigned short)sb))));
}
__device__ __forceinline__ cutlass::bfloat16_t mw_plane(unsigned a,unsigned b,unsigned raw,int slot,int half) {
  unsigned v;
  if(slot==0)v=a;
  else if(slot==1)v=__funnelshift_r(a,a,4);
  else if(slot==2)v=__funnelshift_r(a,a,10);
  else if(slot==3)v=__funnelshift_r(b,b,2);
  else if(slot==4)v=__funnelshift_r(b,b,6);
  else if(slot==5)v=__funnelshift_r(b,b,12);
  else {unsigned w=(a&0x60306030u)|(b&~0x60306030u);
        v=__funnelshift_r(w,w,slot==6?24:30);}
  v&=0x81c081c0u; // two exact BF16 values, each scaled by 2^-126
  unsigned s=raw&127u,sb;
  if(s>=8u)sb=(238u<<7)+(s<<4);
  else if(s){unsigned k=31u-__clz(s);sb=((236u+k)<<7)|((s<<(7u-k))&127u);}
  else sb=0;
  sb|=(raw&128u)<<8;
  unsigned sf=sb|(sb<<16), out;
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(out):"r"(v),"r"(sf));
  return cutlass::bfloat16_t::bitcast((unsigned short)(out>>(half*16)));
}
'''


@triton.jit
def mw_route(IDS, COUNTS, SLOTS, POS, M: tl.constexpr, START: tl.constexpr, B: tl.constexpr):
    e=tl.program_id(0)
    t=tl.arange(0,B)
    hit=tl.full((B,),False,tl.int1)
    slot=tl.full((B,),0,tl.int32)
    for j in tl.static_range(8):
        yes=tl.load(IDS+t*8+j,t<M,-999)==e+START
        hit=hit|yes
        slot=tl.where(yes,j,slot)
    rank=tl.cumsum(hit.to(tl.int32))-1
    count=tl.sum(hit.to(tl.int32))
    tl.store(COUNTS+e,count)
    tl.store(SLOTS+e*M+rank,t*8+slot,hit)
    tl.store(POS+t*8+slot,rank,hit)


@triton.jit
def mw_prefix(COUNTS, OFFS, ORDER, TASKS, BN: tl.constexpr):
    e=tl.arange(0,32)
    n=tl.load(COUNTS+e)
    padded=tl.cdiv(n,BN)*BN
    off=tl.cumsum(padded)-padded
    tl.store(OFFS+e,off)
    tl.store(OFFS+32,tl.sum(padded))
    # Stable hot-first ordering. Every expert is retained, including unseen ones.
    keys=tl.sort(n*32+31-e,descending=True)
    order=31-keys%32
    rows=keys//32
    ends=tl.cumsum(tl.cdiv(rows,BN))
    tl.store(ORDER+e,order)
    tl.store(TASKS+e+1,ends)
    tl.store(TASKS,0)


def mw_prim(M, N, K, BM=64, BN=16, BK=128, STAGES=2, GRID=264, FIRST=True, RS=True, PLANES=False):
    CAP=M*8+32*BN
    QSHAPE=(32,N//16,K//32,32,2) if PLANES else (32,N,K//2)
    QDTYPE='int32' if PLANES else 'uint8'

    @T.prim_func
    def main(Q:T.Tensor(QSHAPE,QDTYPE),
             SF:T.Tensor((32,N,K//16),'uint8'),
             GS:T.Tensor((32,),'float32'),
             X:T.Tensor((M if FIRST else CAP,K),'bfloat16'),
             COUNTS:T.Tensor((32,),'int32'),
             OFFS:T.Tensor((33,),'int32'),
             ORDER:T.Tensor((32,),'int32'),
             TASKS:T.Tensor((33,),'int32'),
             SLOTS:T.Tensor((32,M),'int32'),
             Y:T.Tensor((CAP,N),'float32')):
        with T.Kernel(GRID,threads=128,prelude=MW_PRELUDE) as pid:
            if PLANES:
                qs=T.alloc_shared((BM//16,BK//32,32,2),'int32')
            else:
                qs=T.alloc_shared((BM,BK//2),'uint8')
            ss=T.alloc_shared((BM,BK//16),'uint8')
            bs=T.alloc_shared((BN,BK),'bfloat16')
            if RS:
                af=T.alloc_fragment((BM,BK),'bfloat16')
            else:
                af=T.alloc_shared((BM,BK),'bfloat16')
            acc=T.alloc_fragment((BM,BN),'float32')
            ix=T.alloc_var('int32')
            total=TASKS[32]*(N//BM)
            for wave in T.serial(T.ceildiv(total,GRID)):
                task=wave*GRID+pid
                if task<total:
                    group=task//(N//BM)
                    tile=task%(N//BM)
                    ix=0
                    for step in T.unroll(5):
                        d=16>>step
                        if TASKS[ix+d]<=group:
                            ix=ix+d
                    e=ORDER[ix]
                    row=(group-TASKS[ix])*BN
                    count=COUNTS[e]
                    offset=OFFS[e]
                    T.clear(acc)
                    for kk in T.Pipelined(K//BK,num_stages=STAGES):
                        if PLANES:
                            T.copy(Q[e,tile*(BM//16),kk*(BK//32),0,0],qs)
                        else:
                            T.copy(Q[e,tile*BM,kk*(BK//2)],qs)
                        T.copy(SF[e,tile*BM,kk*(BK//16)],ss)
                        for j,k in T.Parallel(BN,BK):
                            if row+j<count:
                                if FIRST:
                                    p=SLOTS[e,row+j]
                                    bs[j,k]=X[p//8,kk*BK+k]
                                else:
                                    bs[j,k]=X[offset+row+j,kk*BK+k]
                            else:
                                bs[j,k]=T.cast(0,'bfloat16')
                        for i,k in T.Parallel(BM,BK):
                            if PLANES:
                                lane=(i%8)*4+(k%8)//2
                                slot=(k%32)//16*4+(k%16)//8*2+(i%16)//8
                                af[i,k]=T.call_pure_extern('bfloat16','mw_plane',
                                    T.cast(qs[i//16,k//32,lane,0],'uint32'),
                                    T.cast(qs[i//16,k//32,lane,1],'uint32'),
                                    T.cast(ss[i,k//16],'uint32'),slot,k%2)
                            else:
                                af[i,k]=T.call_pure_extern('bfloat16','mw_decode',
                                    T.cast(qs[i,k//2],'uint32'),T.cast(ss[i,k//16],'uint32'),k%2)
                        T.gemm(af,bs,acc,transpose_B=True)
                    for i,j in T.Parallel(BM,BN):
                        if row+j<count:
                            Y[offset+row+j,tile*BM+i]=acc[i,j]*(GS[e]*(256. if PLANES else 1.))
    return main


@functools.lru_cache(None)
def mw_compile(M,N,K=4096,BM=64,BN=16,BK=128,STAGES=2,GRID=264,FIRST=True,RS=True,target='auto',PLANES=False):
    spec=target if target=='auto' else {'kind':'cuda','arch':target}
    return tilelang.compile(mw_prim(M,N,K,BM,BN,BK,STAGES,GRID,FIRST,RS,PLANES),target=spec,
                            pass_configs={'tl.disable_warp_specialized':True})


@triton.jit
def mw_swiglu(Y, Z, OFFS, B:tl.constexpr=4096):
    row=tl.program_id(0)
    k=tl.arange(0,B)
    if row<tl.load(OFFS+32):
        g=tl.load(Y+row*8192+k)
        u=tl.load(Y+row*8192+4096+k)
        tl.store(Z+row*4096+k,g*tl.sigmoid(g)*u)


@triton.jit
def mw_norm(Y,PNW,OFFS,INV):
    row=tl.program_id(0)
    if row<tl.load(OFFS+32):
        k=tl.arange(0,4096)
        v=tl.load(Y+row*4096+k)*(1.+tl.load(PNW+k).to(tl.float32))
        inv=tl.rsqrt(tl.sum(v*v)/4096.+1.e-8)
        tl.store(INV+row,inv)


@triton.jit
def mw_combine(Y,PNW,IDS,WTS,POS,OFFS,INV,OUT,M:tl.constexpr,START:tl.constexpr,B:tl.constexpr=1024):
    t=tl.program_id(0)
    k=tl.program_id(1)*B+tl.arange(0,B)
    f=1.+tl.load(PNW+k).to(tl.float32)
    out=tl.full((B,),0.,tl.float32)
    for j in range(8):
        e=tl.load(IDS+t*8+j)-START
        yes=(e>=0)&(e<32)
        r=tl.load(POS+t*8+j,yes,0)
        off=tl.load(OFFS+e,yes,0)
        inv=tl.load(INV+off+r,yes,0.)
        v=tl.load(Y+(off+r)*4096+k,yes,0.)
        w=tl.load(WTS+t*8+j)
        out+=v*f*inv*w
    tl.store(OUT+t*4096+k,out)


def mw_prepare(weights):
    return {k:weights[k].contiguous() for k in ('w13','w2','w13_gs','w2_gs')}|{
        'w13_sf':weights['w13_sf'].view(torch.uint8).contiguous(),
        'w2_sf':weights['w2_sf'].view(torch.uint8).contiguous(),
        'pnw':weights.get('post_norm_w',weights.get('post_expert_norm')).contiguous()}


def mw_repack_planes(q):
    """A permutation of all four bits of every nibble, matching RS register ownership.

    Two words encode the 16 codes each lane needs for M16/K32. The extraction
    table is the incumbent V9 bit-plane bijection, used here with WGMMA layout.
    """
    e,n,k2=q.shape;k=k2*2
    dst=torch.empty((e,n//16,k//32,32,2),device=q.device,dtype=torch.int32)
    slots=(('A',0),('A',4),('A',10),('B',2),('B',6),('B',12),('W',24),('W',30))
    mask=0x60306030
    for expert in range(e):
        codes=torch.stack((q[expert]&15,q[expert]>>4),-1).reshape(n//16,2,8,k//32,2,2,4,2)
        codes=codes.permute(0,3,2,6,4,5,1,7).reshape(n//16,k//32,32,8,2).to(torch.int64)
        words=torch.zeros((n//16,k//32,32,2),device=q.device,dtype=torch.int64)
        for slot,(src,rot) in enumerate(slots):
            for half in (0,1):
                for bit,dbit in enumerate((6,7,8,15)):
                    p=(dbit+16*half+rot)%32
                    word=(0 if (mask>>p)&1 else 1) if src=='W' else (0 if src=='A' else 1)
                    words[...,word]|=((codes[...,slot,half]>>bit)&1)<<p
        dst[expert]=words.to(torch.int32)
    return dst


def mw_add_planes(bank):
    for name in ('w13','w2'):bank[name+'_planes']=mw_repack_planes(bank[name])
    return bank


def mw_work(data,bank,choice=(64,16,128,2,264,True),target='auto'):
    bm,bn,bk,stages,grid,rs=choice[:6]
    planes=len(choice)>6 and choice[6]
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*bn
    counts=torch.empty(32,device=dev,dtype=torch.int32)
    offs=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(counts)
    tasks=torch.empty(33,device=dev,dtype=torch.int32)
    slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    pos=torch.empty((m,8),device=dev,dtype=torch.int32)
    h=torch.empty((cap,8192),device=dev,dtype=torch.float32)
    z=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16)
    y=torch.empty((cap,4096),device=dev,dtype=torch.float32)
    inv=torch.empty(cap,device=dev,dtype=torch.float32)
    out=torch.empty_like(x)
    first=mw_compile(m,8192,4096,bm,bn,bk,stages,grid,True,rs,target,planes)
    second=mw_compile(m,4096,4096,bm,bn,bk,stages,grid,False,rs,target,planes)
    start=data['config'].get('local_start',0)
    def route():
        mw_route[(32,)](data['topk_ids'],counts,slots,pos,m,start,triton.next_power_of_2(m),num_warps=4 if m<2048 else 16)
        mw_prefix[(1,)](counts,offs,order,tasks,bn)
    def w13():
        first(bank['w13_planes' if planes else 'w13'],bank['w13_sf'],bank['w13_gs'],x,counts,offs,order,tasks,slots,h)
    def act():
        mw_swiglu[(cap,)](h,z,offs,num_warps=4)
    def w2():
        second(bank['w2_planes' if planes else 'w2'],bank['w2_sf'],bank['w2_gs'],z,counts,offs,order,tasks,slots,y)
    def norm():
        mw_norm[(cap,)](y,bank['pnw'],offs,inv,num_warps=4)
    def combine():
        mw_combine[(m,4)](y,bank['pnw'],data['topk_ids'],data['topk_weights'],pos,offs,inv,out,m,start)
        return out
    return (route,w13,act,w2,norm,combine)


def mw_call(data,bank,choice=(64,16,128,2,264,True)):
    for phase in mw_work(data,bank,choice):
        out=phase()
    return out


"""NVFP4 BF16 MoE: eight math warps per CTA, thread 0 streams the next weight tile.
A fully resident occupancy-sized grid runs both GEMMs, SwiGLU, expert norm and reduction.
All scheduling metadata and activations are recomputed each call. Weights remain lossless
packed NVFP4; BF16 activations use FP32 accumulation.
"""
import torch as ctrl_torch
ctrl_H = 4096
ctrl_NLOC = 32
ctrl_TOPK = 8
ctrl_TILE_BYTES = 36864
ctrl_NT1, ctrl_NT2 = (512, 256)
ctrl_NCNT = 64
ctrl_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\n      static_cast<int>(nstage))\n  LAUNCH(1);\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
ctrl_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid);\n'
ctrl__MOD = None

def ctrl__module():
    global ctrl__MOD
    if ctrl__MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(ctrl_CUDA_SRC.encode()).hexdigest()[:12]
        ctrl__MOD = load_inline(name=f'muse_moe_w4a16_v9_{tag}', cpp_sources=ctrl_CPP_SRC, cuda_sources=ctrl_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
        ctrl__MOD.smem_optin = ctrl__MOD.max_smem_attr()
    return ctrl__MOD
ctrl__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
ctrl__LA_MASK = 1613783088

def ctrl__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if ctrl__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(ctrl__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
ctrl__TABLE = ctrl__bit_table()

def ctrl__encode_scales(sf: ctrl_torch.Tensor):
    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),
    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for
    every nonzero finite E4M3 value (subnormals included)."""
    s = sf.float()
    if not ctrl_torch.isfinite(s).all():
        raise ValueError('non-finite NVFP4 block scale')
    a = s.abs()
    nz = a > 0
    m, e = ctrl_torch.frexp(ctrl_torch.where(nz, a, ctrl_torch.ones_like(a)))
    mant = ((m * 2 - 1) * 8).round().to(ctrl_torch.int32)
    e5 = (e - 1 + 21).to(ctrl_torch.int32)
    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and (int(mant.max()) <= 7)
    code = ctrl_torch.where(nz, e5 << 3 | mant, ctrl_torch.zeros_like(e5))
    back = ctrl_torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)
    assert ctrl_torch.equal(ctrl_torch.where(nz, back, ctrl_torch.zeros_like(back)), a)
    return (code.to(ctrl_torch.uint8), s < 0, ~nz)

def ctrl__repack(wq: ctrl_torch.Tensor, sf: ctrl_torch.Tensor, rowmap: ctrl_torch.Tensor) -> ctrl_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = ctrl_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u, neg, zero = ctrl__encode_scales(sf)
    codes = ctrl_torch.where(neg[..., None], codes ^ 8, codes)
    codes = ctrl_torch.where(zero[..., None], ctrl_torch.zeros_like(codes), codes)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(ctrl_torch.int64)
    words = ctrl_torch.zeros((2, T, 8, 8, 32, 2), dtype=ctrl_torch.int64, device=wq.device)
    for k, h, b, w, p in ctrl__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(ctrl_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return ctrl_torch.cat((wbytes, sbytes), 1).contiguous()

def ctrl_prepare(weights):
    dev = weights['w13'].device
    t = ctrl_torch.arange(ctrl_NT1, device=dev)
    i = ctrl_torch.arange(8, device=dev)
    rowmap1 = ctrl_torch.cat((8 * t[:, None] + i, ctrl_H + 8 * t[:, None] + i), 1)
    rowmap2 = ctrl_torch.arange(ctrl_H, device=dev).view(ctrl_NT2, 16)
    w13p = ctrl_torch.empty((ctrl_NLOC, ctrl_NT1, ctrl_TILE_BYTES), dtype=ctrl_torch.uint8, device=dev)
    w2p = ctrl_torch.empty((ctrl_NLOC, ctrl_NT2, ctrl_TILE_BYTES), dtype=ctrl_torch.uint8, device=dev)
    for e in range(ctrl_NLOC):
        w13p[e] = ctrl__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = ctrl__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = ctrl__module()
    props = ctrl_torch.cuda.get_device_properties(dev)
    ctrl_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:ctrl_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:ctrl_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': ctrl_torch.zeros((ctrl_NCNT,), dtype=ctrl_torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}

def ctrl_kernel_fn(data, state):
    cfg = data['config']
    assert cfg['local_count'] == ctrl_NLOC and cfg['d_hidden'] == ctrl_H and (cfg['d_expert'] == ctrl_H)
    mod = state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(ctrl_torch.int32).contiguous()
    tw = data['topk_weights'].to(ctrl_torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * ctrl_TOPK
    nt = 1
    nstage = 2
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = ctrl_torch.empty((npairs, ctrl_H), dtype=ctrl_torch.bfloat16, device=dev)
    y = ctrl_torch.empty((npairs, ctrl_H), dtype=ctrl_torch.float32, device=dev)
    ss = ctrl_torch.empty((npairs,), dtype=ctrl_torch.float32, device=dev)
    out = ctrl_torch.empty((M, ctrl_H), dtype=ctrl_torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, state['cnt'], out, cfg['local_start'], nt, nstage, state['grid'] * mod.resident_blocks(nt, nstage, M))
    return out


"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""
from typing import TypedDict as marlin_TypedDict
import types as marlin_types
import torch as marlin_torch
from sgl_kernel.scalar_type import scalar_types as marlin_scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul as marlin_silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm as marlin_moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as marlin_fused_post_expert_norm_weighted_sum
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as marlin_moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace as marlin_marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin as marlin_prepare_moe_nvfp4_layer_for_marlin

class marlin_Weights(marlin_TypedDict):
    w13: marlin_torch.Tensor
    w13_sf: marlin_torch.Tensor
    w13_gs: marlin_torch.Tensor
    w2: marlin_torch.Tensor
    w2_sf: marlin_torch.Tensor
    w2_gs: marlin_torch.Tensor
    post_norm_w: marlin_torch.Tensor

class marlin_Config(marlin_TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str

class marlin_Inputs(marlin_TypedDict):
    hidden_states: marlin_torch.Tensor
    topk_ids: marlin_torch.Tensor
    topk_weights: marlin_torch.Tensor
    config: marlin_Config
marlin_N_EXPERTS = 256
marlin_TOP_K = 8
marlin_BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
marlin_POST_NORM_EPS = 1e-08

def marlin__marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = marlin_types.SimpleNamespace(quant_config=marlin_types.SimpleNamespace(group_size=16), moe_runner_config=marlin_types.SimpleNamespace(is_gated=True), intermediate_size_per_partition=weights['w13'].shape[1] // 2, params_dtype=marlin_torch.bfloat16, w13_weight=weights['w13'][sl].contiguous(), w2_weight=weights['w2'][sl].contiguous(), w13_weight_scale=weights['w13_sf'][sl].contiguous(), w2_weight_scale=weights['w2_sf'][sl].contiguous(), w13_weight_scale_2=weights['w13_gs'][sl].contiguous(), w2_weight_scale_2=weights['w2_gs'][sl].contiguous())
    marlin_prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer

def marlin_prepare(weights):
    device = weights['w13'].device
    banks = {}
    for start, count in ((0, 32),):
        layer = marlin__marlin_layer(weights, start, count)
        expert_map = None
        if count != marlin_N_EXPERTS:
            expert_map = marlin_torch.full((marlin_N_EXPERTS,), -1, dtype=marlin_torch.int32, device=device)
            expert_map[start:start + count] = marlin_torch.arange(count, dtype=marlin_torch.int32, device=device)
        banks[start, count] = {'w13': layer.w13_weight.data, 'w13_s': layer.w13_weight_scale.data, 'w13_gs': layer.w13_weight_scale_2.data, 'w2': layer.w2_weight.data, 'w2_s': layer.w2_weight_scale.data, 'w2_gs': layer.w2_weight_scale_2.data, 'expert_map': expert_map, 'count': count, 'size_n': layer.intermediate_size_per_partition}
    marlin_torch.cuda.synchronize()
    return {'banks': banks, 'workspace': marlin_marlin_make_workspace(device, max_blocks_per_sm=4), 'norm_w': (weights['post_norm_w'].float() + 1.0).to(marlin_torch.bfloat16)}

def marlin_kernel_fn(data, state):
    cfg = data['config']
    bank = state['banks'][cfg['local_start'], cfg['local_count']]
    x = data['hidden_states'].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank['size_n']
    gemm1_n = 2 * size_n
    expert_map = bank['expert_map']
    local_ids = data['topk_ids'].to(marlin_torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data['topk_weights'].to(marlin_torch.float32).contiguous()
    for block_size_m in marlin_BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * marlin_TOP_K / bank['count'] / block_size_m < 0.9:
            break
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token
        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = marlin_moe_align_block_size(local_ids, block_size_m, marlin_N_EXPERTS if expert_map is not None else bank['count'])
    cache2 = marlin_torch.empty((num_tokens * marlin_TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = marlin_torch.empty((num_tokens * marlin_TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[:num_tokens * marlin_TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[:num_tokens * marlin_TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None, b_q_type=marlin_scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    ws = state['workspace']
    cache1 = marlin_moe_wna16_marlin_gemm(x, cache1, bank['w13'], None, bank['w13_s'], bank['w13_gs'], None, None, None, ws, sorted_ids, block_experts, n_padded, topk_weights, top_k=marlin_TOP_K, size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    marlin_silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = marlin_moe_wna16_marlin_gemm(cache2, cache3, bank['w2'], None, bank['w2_s'], bank['w2_gs'], None, None, None, ws, sorted_ids, block_experts, n_padded, topk_weights, top_k=1, size_m=num_tokens * marlin_TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, marlin_TOP_K, size_k)
    return marlin_fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['norm_w'], eps=marlin_POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids)


import math as fx_math
import statistics as fx_statistics
import time as fx_time
from typing import Callable as fx_Callable
import torch as fx_torch
fx_HIDDEN = 8192
fx_D_EXPERT = 4096
fx_N_EXPERTS = 256
fx_LOCAL_EXPERTS = 32
fx_TOP_K = 8
fx_ROUTER_MULTIPLIER = 0.1767766952966369
fx_RMS_EPS = 1e-05
fx_STREAM_SEED = 5354
fx_STREAM_MEAN = 0.7
fx_STREAM_SHARED = 0.9
fx_STREAM_RANK = 4
fx_DRAFT_SHARE = 0.6

def fx__rms(x: fx_torch.Tensor) -> fx_torch.Tensor:
    xf = x.float()
    return xf * fx_torch.rsqrt(xf.square().mean(-1, keepdim=True) + fx_RMS_EPS)

def fx__basis(device: fx_torch.device) -> tuple[fx_torch.Tensor, fx_torch.Tensor]:
    gen = fx_torch.Generator().manual_seed(fx_STREAM_SEED)
    mean = fx_torch.randn((fx_HIDDEN,), generator=gen).to(device)
    shared = fx_torch.randn((fx_STREAM_RANK, fx_HIDDEN), generator=gen).to(device)
    return (mean, shared)

def fx_make_fixture(weights: dict[str, fx_torch.Tensor], tokens: int, seed: int, *, mode: str='real', q_len: int | None=None) -> dict:
    """Return an expert-call fixture with real router or controlled route IDs.

    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.
    Pass it explicitly for other loads. Draft sharing applies within every verify block.
    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps
    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every
    slot to a uniformly chosen local expert, matching older synthetic tuners.
    """
    device = weights['router'].device
    gen = fx_torch.Generator(device=device).manual_seed(seed)
    mean, shared = fx__basis(device)
    coef = fx_torch.randn((tokens, fx_STREAM_RANK), generator=gen, device=device)
    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)
    if q_len < 1 or tokens % q_len:
        raise ValueError(f'tokens={tokens} must be divisible by q_len={q_len}')
    scale = fx_torch.ones((tokens // q_len, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = fx_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = scale * (fx_STREAM_MEAN * mean + fx_math.sqrt(fx_STREAM_SHARED / fx_STREAM_RANK) * (coef @ shared)) + (1 - fx_STREAM_SHARED * scale.square()).sqrt() * fx_torch.randn((tokens, fx_HIDDEN), generator=gen, device=device)
    h = fx__rms(stream) * (1.0 + weights['pre_ffn_norm'].float())
    router = weights['router'].float()
    biased = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER) + weights['router_bias'].float()
    scores = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER)
    global_ids = biased.topk(fx_TOP_K, dim=-1).indices.to(fx_torch.int32)
    picked = scores.gather(1, global_ids.long())
    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)
    ranges = fx_torch.bincount((global_ids // fx_LOCAL_EXPERTS).flatten(), minlength=fx_N_EXPERTS // fx_LOCAL_EXPERTS)
    local_start = fx_LOCAL_EXPERTS * int(ranges.argmax())
    local = ((global_ids - local_start) % fx_N_EXPERTS).to(fx_torch.int32)
    local_mask = local < fx_LOCAL_EXPERTS
    if mode == 'uniform':
        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)
    elif mode == 'matched_uniform':
        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)
        lids = fx_torch.where(local_mask, lids, fx_torch.full_like(lids, -1))
    elif mode == 'real':
        lids = fx_torch.where(local_mask, local, fx_torch.full_like(local, -1))
    else:
        raise ValueError(mode)
    x = h @ weights['pooler'].float().t()
    x = (fx__rms(x) * (1.0 + weights['pre_expert_norm'].float())).to(fx_torch.bfloat16)
    tw = topk_weights if mode != 'uniform' else fx_torch.rand((tokens, fx_TOP_K), generator=gen, device=device)
    if mode == 'uniform':
        tw = tw / tw.sum(-1, keepdim=True)
    cfg = {'local_count': 32, 'd_hidden': fx_D_EXPERT, 'd_expert': fx_D_EXPERT, 'local_start': 0}
    return {'hidden_states': x, 'topk_ids': lids, 'topk_weights': tw.float(), 'config': cfg, 'local_start': local_start, 'global_ids': global_ids, 'local_mask': local_mask}

def fx_route_summary(fixture: dict) -> dict[str, float]:
    ids = fixture['topk_ids']
    local = ids[ids >= 0]
    counts = fx_torch.bincount(local, minlength=fx_LOCAL_EXPERTS)
    return {'pairs': float(local.numel()), 'distinct': float((counts > 0).sum()), 'max_expert': float(counts.max()), 'mean_nonzero': float(counts[counts > 0].float().mean())}

# Public generator sha256 23e177b87aee70c0d4990515989b95bff2972270cdac8ab967445b21fa2f5163
import math as rg_math
import functools as rg_functools
import torch as rg_torch
rg_HIDDEN = 8192
rg_HEAD_DIM = 64
rg_HQ = 16
rg_HKV = 2
rg_Q_SIZE = 1024
rg_KV_SIZE = 128
rg_QKV_SIZE = 2304
rg_D_EXPERT = 4096
rg_LOCAL_EXPERTS = 32
rg_N_EXPERTS = 256
rg_STREAM_SEED = 5354
rg_STREAM_RANK = 4
rg_ROUTER_STD = 0.045
rg_ROUTER_ALIGN = 0.07
rg_ROUTER_BIAS_STD = 0.055
rg_BLOCK = 16
rg_E4M3_MAX = 448.0
rg_SCALE_LOG2_MEAN = 7.0
rg_SCALE_LOG2_STD = 0.4

def rg_random_weights(rows: int, k: int, *, generator: rg_torch.Generator, device: str | rg_torch.device, leading: tuple[int, ...]=(), wide_scales: bool=False):
    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.

    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)
    clamped to [8, 448]; the global scale
    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448
    instead.
    """
    shape = (*leading, rows, k // 2)
    packed = rg_torch.randint(0, 256, shape, dtype=rg_torch.uint8, generator=generator, device=device)
    scale_shape = (*leading, rows, k // rg_BLOCK)
    if wide_scales:
        u = rg_torch.rand(scale_shape, generator=generator, device=device)
        sf = rg_torch.exp2(-6 + u * (rg_math.log2(rg_E4M3_MAX) + 6)).to(rg_torch.float8_e4m3fn)
        mean_sq = float(sf.float().pow(2).mean())
    else:
        z = rg_torch.randn(scale_shape, generator=generator, device=device)
        sf = rg_torch.exp2(rg_SCALE_LOG2_MEAN + rg_SCALE_LOG2_STD * z).clamp(8.0, rg_E4M3_MAX).to(rg_torch.float8_e4m3fn)
        mean_sq = 4.0 ** rg_SCALE_LOG2_MEAN * rg_math.exp(2 * (rg_math.log(2) * rg_SCALE_LOG2_STD) ** 2)
    gs = rg_torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=rg_torch.float32, device=device)
    return (packed, sf, gs)

def rg_generate_weights(seed: int, device: str='cuda', router_seed: int | None=None):
    """The layer's weights from `seed`. With router_seed, the router and its bias come from that
    seed instead."""
    gen = rg_torch.Generator(device=device).manual_seed(seed)

    def dense(n, k, dtype=rg_torch.bfloat16):
        return (rg_torch.randn((n, k), generator=gen, device=device) / rg_math.sqrt(k)).to(dtype)

    def vec(n, std):
        return (std * rg_torch.randn((n,), generator=gen, device=device)).to(rg_torch.bfloat16)
    w13, w13_sf, gs13 = rg_random_weights(2 * rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))
    w2, w2_sf, gs2 = rg_random_weights(rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))
    gs13 = gs13 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))
    gs2 = gs2 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))
    o, attn_gate, pre_ffn = (dense(rg_HIDDEN, rg_Q_SIZE), vec(rg_HIDDEN, 0.3), vec(rg_HIDDEN, 0.1))
    weights = {'qkv_gate': dense(rg_QKV_SIZE, rg_HIDDEN), 'o': o, 'attn_residual_gate': attn_gate, 'pre_ffn_norm': pre_ffn, **rg__router(gen, device, pre_ffn), 'pooler': dense(rg_D_EXPERT, rg_HIDDEN), 'pre_expert_norm': vec(rg_D_EXPERT, 0.1), 'w13': w13, 'w13_sf': w13_sf, 'w13_gs': gs13, 'w2': w2, 'w2_sf': w2_sf, 'w2_gs': gs2, 'post_expert_norm': vec(rg_D_EXPERT, 0.1), 'unpooler': dense(rg_HIDDEN, rg_D_EXPERT), 'post_ffn_norm': vec(rg_HIDDEN, 0.1), 'ffn_residual_gate': vec(rg_HIDDEN, 0.3), 'next_pre_attn_norm': vec(rg_HIDDEN, 0.1)}
    if router_seed is not None:
        weights.update(rg__router(rg_torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))
    return weights

@rg_functools.cache
def rg__stream_basis(device: str):
    """The residual stream's mean direction [8192] and shared directions [STREAM_RANK, 8192]."""
    gen = rg_torch.Generator().manual_seed(rg_STREAM_SEED)
    mean = rg_torch.randn((rg_HIDDEN,), generator=gen)
    shared = rg_torch.randn((rg_STREAM_RANK, rg_HIDDEN), generator=gen)
    return (mean.to(device), shared.to(device))

def rg__router(gen, device, pre_ffn):
    """FP32 router rows, each with a component along the stream's mean direction, and the router
    bias. pre_ffn is unused."""
    mean, _ = rg__stream_basis(str(device))
    w = rg_ROUTER_STD * rg_torch.randn((rg_N_EXPERTS, rg_HIDDEN), generator=gen, device=device)
    align = rg_ROUTER_ALIGN * rg_torch.randn((rg_N_EXPERTS, 1), generator=gen, device=device)
    rows = w + align * mean[None, :] / rg_math.sqrt(rg_HIDDEN)
    bias = rg_ROUTER_BIAS_STD * rg_torch.randn((rg_N_EXPERTS,), generator=gen, device=device)
    return {'router': rows.contiguous(), 'router_bias': bias.contiguous()}

CAPTURED_R0='eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK'

import base64
import hashlib
import statistics
import time
import zlib
import torch


def _reuse_fp32_impl(data,weights):
    """Independent agreement oracle; scratch weights die here and never enter returned state."""
    x=data['hidden_states'];ids=data['topk_ids'];tw=data['topk_weights'];start=data['config']['local_start']
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=x.device)
    out=torch.zeros_like(x,dtype=torch.float32)
    for e in range(32):
        t,j=torch.where(ids==e+start)
        if t.numel()==0:continue
        def dq(name):
            q=weights[name][e]
            code=torch.stack((q&15,q>>4),-1).flatten(-2).long()
            return lut[code&7]*torch.where((code&8)!=0,-1.,1.)*weights[name+'_sf'][e].float().repeat_interleave(16,-1)*weights[name+'_gs'][e]
        a=x[t].float()@dq('w13').t()
        a=(torch.nn.functional.silu(a[:,:4096])*a[:,4096:]).bfloat16()
        y=a.float()@dq('w2').t()
        y*=1+weights['post_norm_w'].float()
        y*=torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-8)
        out.index_add_(0,t,y*tw[t,j,None])
    return out.bfloat16()


def reuse_relative(got,want):
    return float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))


def reuse_fp32(data,weights):
    old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        return _reuse_fp32_impl(data,weights)
    finally:
        torch.backends.cuda.matmul.allow_tf32=old


"""Exact captured core IDs, public decode weights/router for fresh x and prefill rows."""
import base64
import statistics
import time
import zlib

MW_LAB_PLANES=True

class MWGraph:
    def __init__(self,fn,data):
        self.data={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
        fn(self.data);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=fn(self.data)
    def refresh(self,data):
        for k in ('hidden_states','topk_ids','topk_weights'):self.data[k].copy_(data[k])
    def replay(self):
        self.graph.replay()
        return self.out


def mw_pair(old,new,ds,flush):
    aa=[];bb=[];rr=[]
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for j in range(6):
        old.refresh(ds[j%len(ds)]);new.refresh(ds[j%len(ds)])
        pair=[]
        for g in ((old,new) if j%2==0 else (new,old)):
            flush.zero_();a.record();g.replay();b.record();b.synchronize()
            pair.append(a.elapsed_time(b)*1000.)
        x,y=pair if j%2==0 else pair[::-1]
        aa.append(x);bb.append(y);rr.append(y/x)
    return statistics.median(aa),statistics.median(bb),statistics.median(rr)


def mw_gate(graph,ds,wants):
    worst=0.;row=0.
    for d,want in zip(ds,wants):
        graph.refresh(d);got=graph.replay();torch.cuda.synchronize()
        diff=got.float()-want.float()
        err=float(diff.norm()/want.float().norm().clamp_min(1.e-12))
        den=want.float().norm(dim=-1)
        erow=float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1).clamp_min(1.e-12)).max())
        if not bool(torch.isfinite(got).all()):err=float('inf')
        worst=max(worst,err);row=max(row,erow)
    return worst,row


def mw_phase_times(phases,flush):
    times=[]
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for f in phases:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):f()
        vals=[]
        for _ in range(4):
            flush.zero_();a.record();graph.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000.)
        times.append(statistics.median(vals))
    return ','.join(f'{t:.1f}' for t in times)


import torch
import torch as rv_torch
rv_H = 4096
rv_NLOC = 32
rv_TOPK = 8
rv_TILE_BYTES = 36864
rv_NT1, rv_NT2 = (512, 256)
rv_NCNT = 64
rv_CUDA_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, NT == 1 ? 2 : 1) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = __ldg(ids + i) - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);\n          const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        for (int z=tid;z<8*NB;z+=NTHR) {\n          const int i = z & 7, n = z >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        for (int z=tid;z<16*NB;z+=NTHR) {\n          const int i = z & 15, n = z >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          const float fw0=1.f+__bfloat162float(pnw[m*16+i]); float t2 = ok ? (v*fw0)*(v*fw0) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\n}\n\n}  // namespace\n\nint64_t max_smem_attr() {\n  int dev, optin;\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\n  return optin;\n}\n\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\n  int n;\n  if(nt==1){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));}else if(nt==2){C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<2>, NTHR, smem_bytes(nt,nstage,M)));}else{C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<4>, NTHR, smem_bytes(nt,nstage,M)));}\n  return n;\n}\n\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  const int M = static_cast<int>(x.size(0));\n  const size_t smem = smem_bytes(nt, nstage, M);\n  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto* pp = reinterpret_cast<const __nv_bfloat16*>(pnw.data_ptr());\n  auto* ip = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());\n  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n#define LAUNCH(NTV)                                                                                          \\\n  moe_fused<NTV><<<grid, NTHR, smem, stream>>>(xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp,               \\\n      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip,     \\\n      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start),     \\\n      static_cast<int>(nstage))\n  if(nt==1){LAUNCH(1);}else if(nt==2){LAUNCH(2);}else if(nt==4){LAUNCH(4);}else{TORCH_CHECK(false,"V9 NT must be 1/2/4");}\n#undef LAUNCH\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
rv_CPP_SRC = '\n#include <torch/extension.h>\nint64_t max_smem_attr();\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M);\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M);\nvoid run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor pnw, torch::Tensor w13p,\n         torch::Tensor w2p, torch::Tensor gs13, torch::Tensor gs2, torch::Tensor inter, torch::Tensor y,\n         torch::Tensor ss, torch::Tensor cnt, torch::Tensor out, int64_t local_start, int64_t nt, int64_t nstage,\n         int64_t grid);\n'
rv__MOD = None

def rv__module():
    global rv__MOD
    if rv__MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1(rv_CUDA_SRC.encode()).hexdigest()[:12]
        rv__MOD = load_inline(name=f'muse_moe_w4a16_v9_{tag}', cpp_sources=rv_CPP_SRC, cuda_sources=rv_CUDA_SRC, functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
        rv__MOD.smem_optin = rv__MOD.max_smem_attr()
    return rv__MOD
rv__DQ_SLOTS = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
rv__LA_MASK = 1613783088

def rv__bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if rv__LA_MASK >> p & 1}
    table, used = ([], {'A': set(), 'B': set()})
    for k, (src, rot) in enumerate(rv__DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != 'W' else 'A' if p in la else 'B'
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used['A']) == 32 and len(used['B']) == 32
    return table
rv__TABLE = rv__bit_table()

def rv__repack(wq: rv_torch.Tensor, sf: rv_torch.Tensor, rowmap: rv_torch.Tensor) -> rv_torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = (wq.shape[0], wq.shape[1] * 2)
    assert K == 4096
    T = rowmap.shape[0]
    codes = rv_torch.stack((wq & 15, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.view(rv_torch.uint8)
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(rv_torch.int64)
    words = rv_torch.zeros((2, T, 8, 8, 32, 2), dtype=rv_torch.int64, device=wq.device)
    for k, h, b, w, p in rv__TABLE:
        words[0 if w == 'A' else 1] |= (ct[..., k, h] >> b & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()
    wbytes = words.view(rv_torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)
    return rv_torch.cat((wbytes, sbytes), 1).contiguous()

def rv_prepare(weights):
    dev = weights['w13'].device
    t = rv_torch.arange(rv_NT1, device=dev)
    i = rv_torch.arange(8, device=dev)
    rowmap1 = rv_torch.cat((8 * t[:, None] + i, rv_H + 8 * t[:, None] + i), 1)
    rowmap2 = rv_torch.arange(rv_H, device=dev).view(rv_NT2, 16)
    w13p = rv_torch.empty((rv_NLOC, rv_NT1, rv_TILE_BYTES), dtype=rv_torch.uint8, device=dev)
    w2p = rv_torch.empty((rv_NLOC, rv_NT2, rv_TILE_BYTES), dtype=rv_torch.uint8, device=dev)
    for e in range(rv_NLOC):
        w13p[e] = rv__repack(weights['w13'][e], weights['w13_sf'][e], rowmap1)
        w2p[e] = rv__repack(weights['w2'][e], weights['w2_sf'][e], rowmap2)
    mod = rv__module()
    props = rv_torch.cuda.get_device_properties(dev)
    rv_torch.cuda.synchronize()
    return {'w13p': w13p, 'w2p': w2p, 'gs13': weights['w13_gs'][:rv_NLOC].float().contiguous(), 'gs2': weights['w2_gs'][:rv_NLOC].float().contiguous(), 'pnw': weights['post_norm_w'].contiguous(), 'cnt': rv_torch.zeros((rv_NCNT,), dtype=rv_torch.int32, device=dev), 'grid': props.multi_processor_count, 'mod': mod}

def rv_kernel_fn(data, state):
    cfg = data['config']
    assert cfg['local_count'] == rv_NLOC and cfg['d_hidden'] == rv_H and (cfg['d_expert'] == rv_H)
    mod = state['mod']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(rv_torch.int32).contiguous()
    tw = data['topk_weights'].to(rv_torch.float32).contiguous()
    M = x.shape[0]
    npairs = M * rv_TOPK
    nt = state.get('nt', 1)
    nstage = state.get('nstage_choice', 2)
    while nstage > 1 and mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        nstage -= 1
    if mod.smem_needed(nt, nstage, M) > mod.smem_optin:
        raise ValueError(f"batch {M} too large for the fused kernel's shared-memory routing")
    dev = x.device
    inter = rv_torch.empty((npairs, rv_H), dtype=rv_torch.bfloat16, device=dev)
    y = rv_torch.empty((npairs, rv_H), dtype=rv_torch.float32, device=dev)
    ss = rv_torch.empty((npairs,), dtype=rv_torch.float32, device=dev)
    out = rv_torch.empty((M, rv_H), dtype=rv_torch.bfloat16, device=dev)
    mod.run(x, ids, tw, state['pnw'], state['w13p'], state['w2p'], state['gs13'], state['gs2'], inter, y, ss, state['cnt'], out, cfg['local_start'], nt, nstage, state['grid'] * mod.resident_blocks(nt, nstage, M))
    return out
import torch as hv_torch
import torch as rr_torch
import math as fx_math
import statistics as fx_statistics
import time as fx_time
from typing import Callable as fx_Callable
import torch as fx_torch
import math as rg_math
import functools as rg_functools
import torch as rg_torch
import base64
import statistics
import time
import zlib
import torch
"""Packed-fragment WGMMA producer; portable MMA executes identical loads/layout/dequant locally."""
import functools
import hashlib
from pathlib import Path

import torch
import triton
from torch.utils.cpp_extension import load_inline



VECTOR_CUDA='// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace mwvec {\nconstexpr int BK=64,THREADS=128;\nconstexpr int QB=64*BK/2,SB=64*BK/16;\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned sf_bits(unsigned raw){\n unsigned s=raw&127,bit=0;\n if(s>=8)bit=(238u<<7)+(s<<4);\n else if(s){unsigned k=31-__clz(s);bit=((236+k)<<7)|((s<<(7-k))&127);}\n if(s==127)bit=0x7fff;\n bit|=(raw&128)<<8;return bit|(bit<<16);\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\n}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,const unsigned*lut,unsigned*out){\n unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;\n unsigned s0=lut[raw&255],s1=lut[(raw>>8)&255],s2=lut[(raw>>16)&255],s3=lut[raw>>24];\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==32){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^((r&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^((r&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT>__device__ __forceinline__ void load_stage(unsigned char*q,unsigned char*s,__nv_bfloat16*bs,\n const int*Q,const unsigned char*S,const __nv_bfloat16*X,const int*slots,int e,int mt,int row,int count,int offset,int M,int N,int kk,int first){\n int t=threadIdx.x,w=t>>5,l=t&31;\n #pragma unroll\n for(int g=0;g<2;++g){size_t tile=((size_t)e*(N/16)+mt*4+w)*128+kk*2+g;\n  async_copy(q+(g*128+t)*8,Q+tile*64+l*2,8,1);\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+tile*32+l*4,4,1);\n }\n #pragma unroll\n for(int j=0;j<NT*BK/8/128;++j){int i=t+j*128,r=i/8,k=(i&7)*8,valid=row+r<count;\n  int src=valid?(first?slots[e*M+row+r]/8:offset+row+r):0;\n  async_copy(bs+r*BK+(k^((r&7)*8)),X+(size_t)src*4096+kk*BK+k,16,valid);\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int NT>__global__ void __launch_bounds__(128,2) grouped(const int*Q,const unsigned char*S,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,float*Y,int M,int N,int first){\n extern __shared__ __align__(128) unsigned char sm[];\n unsigned char*qs=sm,*ss=qs+2*QB;auto*bs=(__nv_bfloat16*)(ss+2*SB);\n __shared__ unsigned lut[256];int t=threadIdx.x,w=t>>5,l=t&31;\n lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();\n int total=TASK[32]*(N/64);\n for(int task=blockIdx.x;task<total;task+=gridDim.x){\n  int group=task/(N/64),mt=task%(N/64),ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\n  float acc[NT/2]={};unsigned af[16];\n  load_stage<NT>(qs,ss,bs,Q,S,X,SLOTS,e,mt,row,count,offset,M,N,0,first);\n  load_stage<NT>(qs+QB,ss+SB,bs+NT*BK,Q,S,X,SLOTS,e,mt,row,count,offset,M,N,1,first);\n  for(int kk=0;kk<64;++kk){int stage=kk&1;\n   if(kk==63)asm volatile("cp.async.wait_group 0;":::"memory");\n   else asm volatile("cp.async.wait_group 1;":::"memory");\n   __syncthreads();\n   #pragma unroll\n   for(int g=0;g<2;++g){uint2 q=*(const uint2*)(qs+stage*QB+(g*128+t)*8);\n    unsigned sf=*(const unsigned*)(ss+stage*SB+g*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,lut,af+g*8);}\n   #pragma unroll\n   for(int j=0;j<16;++j)asm volatile("" : "+r"(af[j]));\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("fence.proxy.async.shared::cta;wgmma.fence.sync.aligned;":::"memory");\n#endif\n   #pragma unroll\n   for(int b=0;b<4;++b)mma<NT>(acc,af+b*4,bs+stage*NT*BK,b);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n   #pragma unroll\n   for(int j=0;j<16;++j)asm volatile("" : "+r"(af[j]));\n   __syncthreads();\n   if(kk+2<64)load_stage<NT>(qs+stage*QB,ss+stage*SB,bs+stage*NT*BK,Q,S,X,SLOTS,e,mt,row,count,offset,M,N,kk+2,first);\n  }\n  float gs=GS[e]*256.f;\n  #pragma unroll\n  for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n   int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n   if(r<count)Y[(size_t)(offset+r)*N+c]=acc[j]*gs;\n  }\n  __syncthreads();\n }\n}\n}\nvoid vec_run(torch::Tensor q,torch::Tensor s,torch::Tensor gs,torch::Tensor x,torch::Tensor c,torch::Tensor off,\n torch::Tensor order,torch::Tensor task,torch::Tensor slots,torch::Tensor y,int64_t first,int64_t nt,int64_t grid,int64_t M){\n auto stream=at::cuda::getCurrentCUDAStream();int N=y.size(1);\n #define GO(NT) mwvec::grouped<NT><<<grid,128,2*mwvec::QB+2*mwvec::SB+2*NT*mwvec::BK*2,stream>>>(q.data_ptr<int>(),s.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),c.data_ptr<int>(),off.data_ptr<int>(),order.data_ptr<int>(),task.data_ptr<int>(),slots.data_ptr<int>(),y.data_ptr<float>(),M,N,first)\n if(nt==16){GO(16);}else if(nt==32){GO(32);}else if(nt==64){GO(64);}else TORCH_CHECK(false,"NT must be16/32/64");\n #undef GO\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
VECTOR_CPP='void vec_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);'


@functools.lru_cache(None)
def vec_module():
    return load_inline(name='mw_vec_'+hashlib.sha256(VECTOR_CUDA.encode()).hexdigest()[:12],cpp_sources=VECTOR_CPP,
                       cuda_sources=VECTOR_CUDA,functions=['vec_run'],extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',
                           '-gencode=arch=compute_90a,code=sm_90a','-gencode=arch=compute_120,code=sm_120'],verbose=False)


def vec_prepare(w):
    bank={}
    for name in ('w13','w2'):
        bank[name]=mw_repack_planes(w[name])
        e,n,k16=w[name+'_sf'].shape
        bank[name+'_sf']=w[name+'_sf'].view(torch.uint8).reshape(e,n//16,2,8,k16//2,2).permute(0,1,4,3,5,2).contiguous()
        bank[name+'_gs']=w[name+'_gs'].float().contiguous()
    bank['pnw']=w.get('post_norm_w',w.get('post_expert_norm')).contiguous()
    bank['mod']=vec_module()
    return bank


def vec_work(data,bank,nt=32,grid=264):
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*nt
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
    order=torch.empty_like(c);task=torch.empty_like(off)
    slots=torch.empty((32,m),device=dev,dtype=torch.int32);pos=torch.empty((m,8),device=dev,dtype=torch.int32)
    h=torch.empty((cap,8192),device=dev,dtype=torch.float32)
    a=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((cap,4096),device=dev,dtype=torch.float32)
    inv=torch.empty(cap,device=dev,dtype=torch.float32);out=torch.empty_like(x)
    def route():
        mw_route[(32,)](data['topk_ids'],c,slots,pos,m,data['config'].get('local_start',0),triton.next_power_of_2(m))
        mw_prefix[(1,)](c,off,order,task,nt)
    def first():bank['mod'].vec_run(bank['w13'],bank['w13_sf'],bank['w13_gs'],x,c,off,order,task,slots,h,1,nt,grid,m)
    def act():mw_swiglu[(cap,)](h,a,off,num_warps=4)
    def second():bank['mod'].vec_run(bank['w2'],bank['w2_sf'],bank['w2_gs'],a,c,off,order,task,slots,y,0,nt,grid,m)
    def norm():mw_norm[(cap,)](y,bank['pnw'],off,inv,num_warps=4)
    def combine():
        mw_combine[(m,4)](y,bank['pnw'],data['topk_ids'],data['topk_weights'],pos,off,inv,out,m,data['config'].get('local_start',0))
        return out
    return (route,first,act,second,norm,combine)


def vec_call(data,bank,nt=32,grid=264):
    for f in vec_work(data,bank,nt,grid):out=f()
    return out

"""H200 fixed-producer test: 12 public seeds and R0; no per-seed winner selection."""


class VecGraph:
    def __init__(self,fn,data):
        self.data={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
        fn(self.data);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=fn(self.data)
        self.graph.replay();torch.cuda.synchronize()

    def refresh(self,data):
        for key in ('hidden_states','topk_ids','topk_weights'):self.data[key].copy_(data[key])


def vector_measure(w,bank,old,datasets,flush):
    names=('V9NT1','V9NT2','Vec16','Vec32','Vec64')
    graphs=[]
    for nt in (1,2):
        st=dict(old,nt=nt,nstage_choice=2)
        graphs.append(VecGraph(lambda d,s=st:rv_kernel_fn(d,s),datasets[0]))
    for nt in (16,32,64):graphs.append(VecGraph(lambda d,n=nt:vec_call(d,bank,n,264),datasets[0]))
    worst=[0.]*len(names);rows=[0.]*len(names);times=[[] for _ in names]
    for data in datasets:
        want=reuse_fp32(data,w);den=want.float().norm().clamp_min(1e-12)
        rowden=want.float().norm(dim=-1);rowden=rowden.clamp_min(rowden.mean()*.1+1e-12)
        for j,g in enumerate(graphs):
            g.refresh(data);g.graph.replay();torch.cuda.synchronize();got=g.out.float()
            if not bool(torch.isfinite(got).all()):worst[j]=float('inf');rows[j]=float('inf');continue
            worst[j]=max(worst[j],float((got-want.float()).norm()/den))
            rows[j]=max(rows[j],float(((got-want.float()).norm(dim=-1)/rowden).max()))
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(8):
        for g in graphs:g.refresh(datasets[i%3])
        order=list(range(len(names)));order=order[i%len(order):]+order[:i%len(order)]
        if i%2:order.reverse()
        for j in order:
            flush.zero_();a.record();graphs[j].graph.replay();b.record();b.synchronize()
            times[j].append(a.elapsed_time(b)*1000.)
    return [(statistics.median(t),e,r) for t,e,r in zip(times,worst,rows)]


def vector_phase(data,bank,flush):
    results=[]
    for nt in (16,32,64):
        fs=vec_work(data,bank,nt,264)
        for f in fs:f()
        row=[]
        for f in fs:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):f()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);ts=[]
            for _ in range(6):
                flush.zero_();a.record();graph.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000.)
            row.append(statistics.median(ts))
        results.append(f'PHASE NT={nt} route,w13,act,w2,norm,combine='+','.join(f'{t:.1f}' for t in row))
    return results


def lab(deadline,weights,state=None):
    dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META packed register fragments, CTA-local format LUT from raw E4M3, two-stage cp.async Q/SF/X, BK64, no spills in sm90a. ORDER V9NT1,V9NT2,Vec16,Vec32,Vec64. 12 public generator seeds; 3 fresh real-router fixtures/seed/M,8 rotated cold128MiB graph timings. FP32 L2<=.01,row<=.05. R0 exact IDs uses fresh public weights, not original decode-host weights. Phases diagnostic, do not sum.']
    raw=zlib.decompress(base64.b64decode(CAPTURED_R0));ms=(8,16,32,128,256)
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8);records=[]
    for i in range(12):
        if time.monotonic()>deadline-15:lines.append(f'INCOMPLETE before_seed={i}');break
        seed=5353+7919*i;w=rg_generate_weights(seed,device=dev);w['post_norm_w']=w['post_expert_norm']
        old=rv_prepare(w);bank=vec_prepare(w)
        for m in (128,256):
            ds=[fx_make_fixture(w,m,9000+m+7919*j) for j in range(3)]
            if i==0:
                off=8*sum(ms[:ms.index(m)])
                ids=torch.tensor(list(raw[off:off+m*8]),device=dev,dtype=torch.int32).view(m,8)
                ids=torch.where(ids<32,ids,-1)
                r0=[dict(d,topk_ids=ids.clone()) for d in ds]
                vals=vector_measure(w,bank,old,r0,flush)
                lines.append(f'R0 M={m} us='+','.join(f'{v[0]:.1f}' for v in vals)+' l2='+','.join(f'{v[1]:.5f}' for v in vals)+' row='+','.join(f'{v[2]:.4f}' for v in vals))
                lines.extend(f'M={m} '+p for p in vector_phase(r0[0],bank,flush))
            vals=vector_measure(w,bank,old,ds,flush)
            counts=torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids']>=0],minlength=32)
            lines.append(f'S seed={seed} M={m} P={int(counts.sum())} E={int((counts>0).sum())} hot={int(counts.max())} us='+','.join(f'{v[0]:.1f}' for v in vals)+' l2='+','.join(f'{v[1]:.5f}' for v in vals)+' row='+','.join(f'{v[2]:.4f}' for v in vals))
            records.append((m,vals))
        # Graphs and scratch live only during each measurement; no closure name is deleted.
    names=('V9NT1','V9NT2','Vec16','Vec32','Vec64')
    for m in (128,256):
        for j,name in enumerate(names):
            vals=[v[j] for mm,v in records if mm==m]
            if not vals:continue
            good=all(v[1]<=.01 and v[2]<=.05 for v in vals)
            times=[v[0] for v in vals]
            lines.append(f'AGG {name} M={m} seeds={len(vals)} mean={statistics.mean(times):.1f} median={statistics.median(times):.1f} min={min(times):.1f} max={max(times):.1f} '+('PASS' if good else 'REJECT'))
    return lines

# ---- h200_lab wrapper: payload moe_vector_wgmma_seed sha256 de53e58b5071116b on base f3786341611f4f1f ----
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
        print("LAB_BEGIN moe_vector_wgmma_seed de53e58b5071116b", flush=True)
        for _l in _out:
            print(_l, flush=True)
        print(f"LAB_END lab_s={_t.monotonic() - _t0:.1f} prepare_s={_t.monotonic() - _start:.1f}", flush=True)
    return state
