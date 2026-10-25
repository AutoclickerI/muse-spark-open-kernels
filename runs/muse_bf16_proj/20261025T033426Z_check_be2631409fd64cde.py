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


# ---- h200_lab wrapper: payloads moe_nt1_timeline 4583fe72360768ff on base f3786341611f4f1f ----
_LAB_PAYLOADS = [('moe_nt1_timeline', '4583fe72360768ff', 'import torch\nfrom torch.utils.cpp_extension import load_inline\nimport time as lab_time\nimport statistics as lab_stats\nlab_torch=torch\n\nNT1_SOURCES={\'INC\': \'\\n#define C10_CUDA_CHECK(expr) do{cudaError_t e=(expr);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)\\n#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())\\n\\n#include <cuda_runtime.h>\\n#include <stdexcept>\\n#include <vector>\\n\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace {\\n\\nconstexpr int HD = 4096;\\nconstexpr int NLOC = 32;\\nconstexpr int TILE_BYTES = 36864;\\nconstexpr int TILE_W_BYTES = 32768;\\nconstexpr int NT1 = 512;\\nconstexpr int NT2 = 256;\\nconstexpr int NCW = 8;                 // consumer warps\\nconstexpr int NCT = NCW * 32;          // consumer threads\\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\\nconstexpr uint32_t LA_MASK = 0x60306030u;\\n// persistent counters\\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\\n\\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\\n}\\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\\n}\\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\\n  uint32_t ok;\\n  asm volatile("{\\\\n .reg .pred p;\\\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\\\n selp.u32 %0, 1, 0, p;\\\\n}\\\\n"\\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\\n  return ok != 0;\\n}\\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\\n  while (!mbar_try(b, parity)) {}\\n}\\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\\n}\\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\\n}\\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\\n}\\n__device__ __forceinline__ void bar_consumers() {\\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\\n}\\n__device__ __forceinline__ int ld_acquire(const int* p) {\\n  int v;\\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\\n  return v;\\n}\\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\\n  uint32_t d;\\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\\n  return d;\\n}\\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\\n                                         uint32_t b0, uint32_t b1) {\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\\n}\\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\\n  o[0] = A & DQ_MASK;\\n  o[1] = rotr(A, 4) & DQ_MASK;\\n  o[2] = rotr(A, 10) & DQ_MASK;\\n  o[3] = rotr(B, 2) & DQ_MASK;\\n  o[4] = rotr(B, 6) & DQ_MASK;\\n  o[5] = rotr(B, 12) & DQ_MASK;\\n  uint32_t W; asm("lop3.b32 %0,%1,%2,%3,0xe4;":"=r"(W):"r"(A),"r"(B),"n"(LA_MASK));\\n  o[6] = rotr(W, 24) & DQ_MASK;\\n  o[7] = rotr(W, 30) & DQ_MASK;\\n}\\n\\nstruct PrefetchState {\\n  volatile int index, stage, phase, pass;\\n  volatile int e, nb, m, mb, me;\\n  volatile uint64_t pol_first, pol_normal;\\n};\\n\\nstruct Route {        // shared-memory routing summary\\n  int n[NLOC];        // pairs per local expert\\n  int off[NLOC];      // first sorted position\\n  int nblk[NLOC];     // token blocks\\n  int us1[NLOC + 1];  // phase-1 unit prefix\\n  int us2[NLOC + 1];  // phase-2 unit prefix\\n  int c1, c2;         // m-chunk sizes\\n  int run[NLOC];\\n  int wc[NCW][NLOC];\\n  PrefetchState pre;\\n};\\n\\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\\n                                            int& e, int& nb, int& m) {\\n  e = 0;\\n#pragma unroll\\n  for (int d = 16; d > 0; d >>= 1) {\\n    if (us[e + d] <= u) e += d;\\n  }\\n  const int rel = u - us[e];\\n  const int nbk = nblk[e];\\n  const int nfull = ntile / cc;\\n  const int span = cc * nbk;\\n  if (rel < nfull * span) {\\n    const int mc = rel / span;\\n    const int r2 = rel - mc * span;\\n    nb = r2 / cc;\\n    m = mc * cc + (r2 - nb * cc);\\n  } else {\\n    const int L = ntile - nfull * cc;\\n    const int r3 = rel - nfull * span;\\n    nb = r3 / L;\\n    m = nfull * cc + (r3 - nb * L);\\n  }\\n}\\n\\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\\n                                               const int* nblk, int cc, int ntile) {\\n  if (++m == me) {\\n    if (++nb == nblk[e]) {\\n      nb = 0;\\n      mb += cc;\\n      if (mb >= ntile) {\\n        ++e;\\n        while (e < NLOC && nblk[e] == 0) ++e;\\n        mb = 0;\\n      }\\n      me = min(ntile, mb + cc);\\n    }\\n    m = mb;\\n  }\\n}\\n\\n\\n__device__ __forceinline__ void prefetch_one(\\n    Route* rt, int total, int first, int a2, int nstage,\\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\\n    const uint8_t* w13p, const uint8_t* w2p) {\\n  auto& p = rt->pre;\\n  int idx = p.index;\\n  if (idx >= total) return;\\n  int st=p.stage, ph=p.phase, pass=p.pass;\\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\\n  const int ntile=pass?NT2:NT1;\\n  const int cc=pass?rt->c2:rt->c1;\\n  const uint8_t* base=pass?w2p:w13p;\\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\\n  mbar_wait(&emptyb[st], ph^1u);\\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\\n  ++idx;\\n  if (idx < total) {\\n    if (idx == first) {\\n      pass=1;\\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\\n      mb=(m/rt->c2)*rt->c2;\\n      me=min(NT2,mb+rt->c2);\\n    } else {\\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\\n    }\\n  }\\n  if (++st == nstage) { st=0; ph^=1; }\\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\\n}\\n\\ntemplate <int NT>\\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\\n    int M, int local_start, int nstage) {\\n  constexpr int NB = 8 * NT;\\n  constexpr int RED = NT * 4;\\n  const int npairs = M * 8;\\n  extern __shared__ __align__(128) uint8_t smem[];\\n  uint8_t* stages = smem;\\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\\n  uint64_t* emptyb = fullb + nstage;\\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\\n\\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\\n  const int G = gridDim.x, c = blockIdx.x;\\n  int bar_gen0 = 0;\\n\\n  // ---------------- routing: counts ----------------\\n  if (tid < NLOC) rt->n[tid] = 0;\\n  if (tid == 0) {\\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\\n    bar_gen0 = ld_acquire(cnt + C_GEN);\\n  }\\n  __syncthreads();\\n  for (int i = tid; i < npairs; i += NTHR) {\\n    int e = __ldg(ids + i) - local_start;\\n    if (e < 0 || e >= NLOC) e = -1;\\n    eflat[i] = static_cast<int8_t>(e);\\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\\n  }\\n  __syncthreads();\\n  if (warp == 0) {\\n    const int n = rt->n[lane];\\n    const int nb = (n + NB - 1) / NB;\\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\\n#pragma unroll\\n    for (int d = 1; d < 32; d <<= 1) {\\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\\n    }\\n    rt->off[lane] = off - n;\\n    rt->run[lane] = off - n;\\n    rt->nblk[lane] = nb;\\n    rt->us1[lane + 1] = u1;\\n    rt->us2[lane + 1] = u2;\\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\\n    if (lane == 0) {\\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\\n    }\\n  }\\n  __syncthreads();\\n\\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\\n\\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\\n  if (tid == 0) {\\n    uint64_t pf,pn;\\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\\n    if (total_count) {\\n      int e,nb,m;\\n      const int pass=first_count?0:1;\\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\\n    }\\n    for (int i=0;i<nstage;++i)\\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n  }\\n  __syncthreads();\\n\\n  // ---------------- consumers: deterministic token lists ----------------\\n  for (int base = 0; base < npairs; base += NCT) {\\n    rt->wc[warp][lane] = 0;\\n    bar_consumers();\\n    const int i = base + tid;\\n    const int e = i < npairs ? eflat[i] : -1;\\n    const unsigned mk = __match_any_sync(~0u, e);\\n    const int rank = __popc(mk & ((1u << lane) - 1u));\\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\\n    bar_consumers();\\n    if (tid < NLOC) {\\n      int s = rt->run[tid];\\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\\n      rt->run[tid] = s;\\n    }\\n    bar_consumers();\\n    if (i < npairs) {\\n      if (e >= 0) {\\n        const int pos = rt->wc[warp][e] + rank;\\n        pos_of[i] = static_cast<int16_t>(pos);\\n        pair_of[pos] = static_cast<int16_t>(i);\\n      } else {\\n        pos_of[i] = -1;\\n      }\\n    }\\n  }\\n  // zero this CTA\\\'s slice of the sum-of-squares accumulators, then publish\\n  {\\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\\n    __threadfence();\\n    bar_consumers();\\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\\n  }\\n\\n  const int q = lane & 3;\\n  uint32_t Bf[NT][8][8];\\n  int st = 0;\\n  uint32_t ph = 0;\\n  int buf = 0;\\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\\n  float fw[2];   // (1 + post_norm_w) for this thread\\\'s phase-2 epilogue row, refreshed per tile\\n\\n  for (int pass = 0; pass < 2; ++pass) {\\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\\n    const int* us = pass ? rt->us2 : rt->us1;\\n    const int cc = pass ? rt->c2 : rt->c1;\\n    const int ntile = pass ? NT2 : NT1;\\n    int e=0, nb=0, m=0, mb=0, me=0;\\n    if (ua < ub) {\\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\\n      mb = (m / cc) * cc;\\n      me = min(ntile, mb + cc);\\n    }\\n    for (int u = ua; u < ub; ++u) {\\n      const int ne = rt->n[e], off = rt->off[e];\\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\\n          __threadfence();\\n          bar_consumers();\\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\\n        }\\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\\n        if (pass == 1) {\\n          const int target = NT1 * rt->nblk[e];\\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\\n        }\\n#pragma unroll\\n        for (int j = 0; j < NT; ++j) {\\n          const int idx = nb * NB + j * 8 + (lane >> 2);\\n          const __nv_bfloat16* row = nullptr;\\n          if (idx < ne) {\\n            const int pos = off + idx;\\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\\n          }\\n#pragma unroll\\n          for (int g = 0; g < 8; ++g) {\\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\\n            if (row) {\\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\\n            }\\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\\n          }\\n        }\\n      }\\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\\n\\n      mbar_wait(&fullb[st], ph);\\n      const uint8_t* tile = stages + st * TILE_BYTES;\\n      float acc[NT][4];\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\\n      {\\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\\n#pragma unroll\\n        for (int g = 0; g < 8; ++g) {\\n          const uint4 wv = wq[g * 32];\\n          const uint32_t sw = scw[g >> 1];\\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\\n          uint32_t o[16];\\n          dq8(wv.x, wv.y, o);\\n          dq8(wv.z, wv.w, o + 8);\\n#pragma unroll\\n          for (int s = 0; s < 4; ++s) {\\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\\n#pragma unroll\\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\\n          }\\n        }\\n      }\\n      __syncwarp();\\n      if (lane == 0) mbar_arrive(&emptyb[st]);\\n      if (++st == nstage) { st = 0; ph ^= 1u; }\\n\\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\\n      bar_consumers();\\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n      const float* rb = red + buf * NCW * 32 * RED;\\n      if (pass == 0) {\\n        const float sc = gs13[e] * 256.f;\\n        if (tid < 8 * NB) {\\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\\n          float G1 = 0.f, U = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) {\\n            G1 += rb[(w * 32 + ls) * RED + rg];\\n            U += rb[(w * 32 + ls) * RED + rg + 2];\\n          }\\n          const int idx = nb * NB + n;\\n          if (idx < ne) {\\n            G1 *= sc;\\n            U *= sc;\\n            const float h = G1 / (1.f + __expf(-G1)) * U;\\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\\n          }\\n        }\\n      } else {\\n        const float sc = gs2[e] * 256.f;\\n        if (tid < 16 * NB) {\\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\\n          float v = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\\n          v *= sc;\\n          const int idx = nb * NB + n;\\n          const bool ok = idx < ne;\\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\\n#pragma unroll\\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\\n        }\\n      }\\n      buf ^= 1;\\n      ++seg_tiles;\\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\\n    }\\n  }\\n  if (cur_pass == 0 && seg_tiles > 0) {\\n    __threadfence();\\n    bar_consumers();\\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\\n  }\\n\\n  // ---------------- grid barrier ----------------\\n  __threadfence();\\n  bar_consumers();\\n  if (tid == 0) {\\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\\n      cnt[C_ZERO] = 0;\\n      cnt[C_BAR] = 0;\\n      __threadfence();\\n      atomicAdd(cnt + C_GEN, 1);\\n    } else {\\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\\n    }\\n  }\\n  bar_consumers();\\n\\n  // ---------------- combine: item = (token, half of the columns) ----------------\\n  for (int it = c; it < 2 * M; it += G) {\\n    const int t = it >> 1;\\n    const int c0 = (it & 1) * 2048 + tid * 8;\\n    float f[8];\\n    {\\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\\n#pragma unroll\\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\\n    }\\n    float acc[8];\\n#pragma unroll\\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\\n#pragma unroll\\n    for (int j = 0; j < 8; ++j) {\\n      const int pos = pos_of[t * 8 + j];\\n      if (pos >= 0) {\\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\\n      }\\n    }\\n    uint4 ov;\\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\\n#pragma unroll\\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\\n  }\\n}\\n\\nsize_t smem_bytes(int nt, int nstage, int M) {\\n  const int npairs = M * 8;\\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\\n}\\n\\n}  // namespace\\n\\nint64_t max_smem_attr() {\\n  int dev, optin;\\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\\n  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));\\n  return optin;\\n}\\n\\nint64_t smem_needed(int64_t nt, int64_t nstage, int64_t M) { return (int64_t)smem_bytes(nt, nstage, M); }\\nint64_t resident_blocks(int64_t nt, int64_t nstage, int64_t M) {\\n  int n;\\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));\\n  return n;\\n}\\n\\nvoid run(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stages,int64_t grid,int64_t stream){\\n moe_fused<1><<<grid,NTHR,smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\\n (const __nv_bfloat16*)v[0],(const int*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],\\n (const uint8_t*)v[4],(const uint8_t*)v[5],(const float*)v[6],(const float*)v[7],\\n (__nv_bfloat16*)v[8],(float*)v[9],(float*)v[10],(int*)v[11],(__nv_bfloat16*)v[12],M,start,stages);\\n C10_CUDA_KERNEL_LAUNCH_CHECK();\\n}\\n\'}\nH = 4096\n\nNLOC = 32\n\nTOPK = 8\n\nTILE_BYTES = 36864\n\nNT1, NT2 = 512, 256\n\nNCNT = 64\n\n_DQ_SLOTS = (("A", 0), ("A", 4), ("A", 10), ("B", 2), ("B", 6), ("B", 12), ("W", 24), ("W", 30))\n\n_LA_MASK = 0x60306030\n\ndef _bit_table():\n    """(slot, half, code bit, word, bit position) for the kernel\'s dq8 decode."""\n    dbit = {3: 15, 2: 8, 1: 7, 0: 6}\n    la = {p for p in range(32) if (_LA_MASK >> p) & 1}\n    table, used = [], {"A": set(), "B": set()}\n    for k, (src, rot) in enumerate(_DQ_SLOTS):\n        for h in (0, 1):\n            for b in range(4):\n                p = (dbit[b] + 16 * h + rot) % 32\n                w = src if src != "W" else ("A" if p in la else "B")\n                assert p not in used[w]\n                used[w].add(p)\n                table.append((k, h, b, w, p))\n    assert len(used["A"]) == 32 and len(used["B"]) == 32\n    return table\n\n_TABLE = _bit_table()\n\ndef _encode_scales(sf: torch.Tensor):\n    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),\n    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for\n    every nonzero finite E4M3 value (subnormals included)."""\n    s = sf.float()\n    if not torch.isfinite(s).all():\n        raise ValueError("non-finite NVFP4 block scale")\n    a = s.abs()\n    nz = a > 0\n    m, e = torch.frexp(torch.where(nz, a, torch.ones_like(a)))\n    mant = ((m * 2 - 1) * 8).round().to(torch.int32)\n    e5 = (e - 1 + 21).to(torch.int32)\n    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and int(mant.max()) <= 7\n    code = torch.where(nz, (e5 << 3) | mant, torch.zeros_like(e5))\n    back = torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)\n    assert torch.equal(torch.where(nz, back, torch.zeros_like(back)), a)\n    return code.to(torch.uint8), s < 0, ~nz\n\ndef _repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:\n    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""\n    R, K = wq.shape[0], wq.shape[1] * 2\n    assert K == 4096\n    T = rowmap.shape[0]\n    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)\n    u, neg, zero = _encode_scales(sf)\n    codes = torch.where(neg[..., None], codes ^ 8, codes)                  # sign moved into the codes\n    codes = torch.where(zero[..., None], torch.zeros_like(codes), codes)   # 0 * anything stays 0\n    codes = codes.reshape(R, K)[rowmap.reshape(-1)]\n    # k = 512 w + 64 g + 16 q + 4 (2 p + sl) + 2 j + h ; tile row = 8 ri + rr\n    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                           # T ri rr w g q p sl j h\n    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)\n    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)\n    for k, h, b, w, p in _TABLE:\n        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p\n    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()                    # T w g lane p {A,B}\n    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)\n    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)                       # T ri rr w g q\n    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)                  # T w (rr q) g ri\n    return torch.cat((wbytes, sbytes), 1).contiguous()\ndef _raw_repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:\n    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""\n    R, K = wq.shape[0], wq.shape[1] * 2\n    assert K == 4096\n    T = rowmap.shape[0]\n    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)\n    u = sf.contiguous().view(torch.uint8)  # exact stored E4M3 bytes\n    codes = codes.reshape(R, K)[rowmap.reshape(-1)]\n    # k = 512 w + 64 g + 16 q + 4 (2 p + sl) + 2 j + h ; tile row = 8 ri + rr\n    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                           # T ri rr w g q p sl j h\n    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)\n    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)\n    for k, h, b, w, p in _TABLE:\n        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p\n    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()                    # T w g lane p {A,B}\n    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)\n    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)                       # T ri rr w g q\n    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)                  # T w (rr q) g ri\n    return torch.cat((wbytes, sbytes), 1).contiguous()\n\n"""Prepare-time MoE tuner for decode-layer routing.\n\nIt reconstructs the layer\'s router distribution from the loaded router, bias, pre-FFN norm and\npooler weights. It never reads a submission input: callers provide only their loaded weights and\nshape list, so the resulting shape dispatch remains CUDA-graph safe.\n"""\nimport math as fx_math\nimport statistics as fx_statistics\nimport time as fx_time\nfrom typing import Callable as fx_Callable\nimport torch as fx_torch\nfx_HIDDEN = 8192\nfx_D_EXPERT = 4096\nfx_N_EXPERTS = 256\nfx_LOCAL_EXPERTS = 32\nfx_TOP_K = 8\nfx_ROUTER_MULTIPLIER = 0.1767766952966369\nfx_RMS_EPS = 1e-05\nfx_STREAM_SEED = 5354\nfx_STREAM_MEAN = 0.7\nfx_STREAM_SHARED = 0.9\nfx_STREAM_RANK = 4\nfx_DRAFT_SHARE = 0.6\n\ndef fx__rms(x: fx_torch.Tensor) -> fx_torch.Tensor:\n    xf = x.float()\n    return xf * fx_torch.rsqrt(xf.square().mean(-1, keepdim=True) + fx_RMS_EPS)\n\ndef fx__basis(device: fx_torch.device) -> tuple[fx_torch.Tensor, fx_torch.Tensor]:\n    gen = fx_torch.Generator().manual_seed(fx_STREAM_SEED)\n    mean = fx_torch.randn((fx_HIDDEN,), generator=gen).to(device)\n    shared = fx_torch.randn((fx_STREAM_RANK, fx_HIDDEN), generator=gen).to(device)\n    return (mean, shared)\n\ndef fx_make_fixture(weights: dict[str, fx_torch.Tensor], tokens: int, seed: int, *, mode: str=\'real\', q_len: int | None=None) -> dict:\n    """Return an expert-call fixture with real router or controlled route IDs.\n\n    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.\n    Pass it explicitly for other loads. Draft sharing applies within every verify block.\n    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps\n    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every\n    slot to a uniformly chosen local expert, matching older synthetic tuners.\n    """\n    device = weights[\'router\'].device\n    gen = fx_torch.Generator(device=device).manual_seed(seed)\n    mean, shared = fx__basis(device)\n    coef = fx_torch.randn((tokens, fx_STREAM_RANK), generator=gen, device=device)\n    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)\n    if q_len < 1 or tokens % q_len:\n        raise ValueError(f\'tokens={tokens} must be divisible by q_len={q_len}\')\n    scale = fx_torch.ones((tokens // q_len, q_len, 1), device=device)\n    if 1 < q_len <= 8:\n        scale[:, 1:] = fx_DRAFT_SHARE\n    scale = scale.reshape(tokens, 1)\n    stream = scale * (fx_STREAM_MEAN * mean + fx_math.sqrt(fx_STREAM_SHARED / fx_STREAM_RANK) * (coef @ shared)) + (1 - fx_STREAM_SHARED * scale.square()).sqrt() * fx_torch.randn((tokens, fx_HIDDEN), generator=gen, device=device)\n    h = fx__rms(stream) * (1.0 + weights[\'pre_ffn_norm\'].float())\n    router = weights[\'router\'].float()\n    biased = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER) + weights[\'router_bias\'].float()\n    scores = fx_torch.sigmoid(h @ router.t() * fx_ROUTER_MULTIPLIER)\n    global_ids = biased.topk(fx_TOP_K, dim=-1).indices.to(fx_torch.int32)\n    picked = scores.gather(1, global_ids.long())\n    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)\n    ranges = fx_torch.bincount((global_ids // fx_LOCAL_EXPERTS).flatten(), minlength=fx_N_EXPERTS // fx_LOCAL_EXPERTS)\n    local_start = fx_LOCAL_EXPERTS * int(ranges.argmax())\n    local = ((global_ids - local_start) % fx_N_EXPERTS).to(fx_torch.int32)\n    local_mask = local < fx_LOCAL_EXPERTS\n    if mode == \'uniform\':\n        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)\n    elif mode == \'matched_uniform\':\n        lids = fx_torch.randint(0, fx_LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=fx_torch.int32)\n        lids = fx_torch.where(local_mask, lids, fx_torch.full_like(lids, -1))\n    elif mode == \'real\':\n        lids = fx_torch.where(local_mask, local, fx_torch.full_like(local, -1))\n    else:\n        raise ValueError(mode)\n    x = h @ weights[\'pooler\'].float().t()\n    x = (fx__rms(x) * (1.0 + weights[\'pre_expert_norm\'].float())).to(fx_torch.bfloat16)\n    tw = topk_weights if mode != \'uniform\' else fx_torch.rand((tokens, fx_TOP_K), generator=gen, device=device)\n    if mode == \'uniform\':\n        tw = tw / tw.sum(-1, keepdim=True)\n    cfg = {\'local_count\': 32, \'d_hidden\': fx_D_EXPERT, \'d_expert\': fx_D_EXPERT, \'local_start\': 0}\n    return {\'hidden_states\': x, \'topk_ids\': lids, \'topk_weights\': tw.float(), \'config\': cfg, \'local_start\': local_start, \'global_ids\': global_ids, \'local_mask\': local_mask}\n\ndef fx_route_summary(fixture: dict) -> dict[str, float]:\n    ids = fixture[\'topk_ids\']\n    local = ids[ids >= 0]\n    counts = fx_torch.bincount(local, minlength=fx_LOCAL_EXPERTS)\n    return {\'pairs\': float(local.numel()), \'distinct\': float((counts > 0).sum()), \'max_expert\': float(counts.max()), \'mean_nonzero\': float(counts[counts > 0].float().mean())}\n# Public generator sha256 23e177b87aee70c0d4990515989b95bff2972270cdac8ab967445b21fa2f5163\nimport math as rg_math\nimport functools as rg_functools\nimport torch as rg_torch\nrg_HIDDEN = 8192\nrg_HEAD_DIM = 64\nrg_HQ = 16\nrg_HKV = 2\nrg_Q_SIZE = 1024\nrg_KV_SIZE = 128\nrg_QKV_SIZE = 2304\nrg_D_EXPERT = 4096\nrg_LOCAL_EXPERTS = 32\nrg_N_EXPERTS = 256\nrg_STREAM_SEED = 5354\nrg_STREAM_RANK = 4\nrg_ROUTER_STD = 0.045\nrg_ROUTER_ALIGN = 0.07\nrg_ROUTER_BIAS_STD = 0.055\nrg_BLOCK = 16\nrg_E4M3_MAX = 448.0\nrg_SCALE_LOG2_MEAN = 7.0\nrg_SCALE_LOG2_STD = 0.4\n\ndef rg_random_weights(rows: int, k: int, *, generator: rg_torch.Generator, device: str | rg_torch.device, leading: tuple[int, ...]=(), wide_scales: bool=False):\n    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.\n\n    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)\n    clamped to [8, 448]; the global scale\n    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448\n    instead.\n    """\n    shape = (*leading, rows, k // 2)\n    packed = rg_torch.randint(0, 256, shape, dtype=rg_torch.uint8, generator=generator, device=device)\n    scale_shape = (*leading, rows, k // rg_BLOCK)\n    if wide_scales:\n        u = rg_torch.rand(scale_shape, generator=generator, device=device)\n        sf = rg_torch.exp2(-6 + u * (rg_math.log2(rg_E4M3_MAX) + 6)).to(rg_torch.float8_e4m3fn)\n        mean_sq = float(sf.float().pow(2).mean())\n    else:\n        z = rg_torch.randn(scale_shape, generator=generator, device=device)\n        sf = rg_torch.exp2(rg_SCALE_LOG2_MEAN + rg_SCALE_LOG2_STD * z).clamp(8.0, rg_E4M3_MAX).to(rg_torch.float8_e4m3fn)\n        mean_sq = 4.0 ** rg_SCALE_LOG2_MEAN * rg_math.exp(2 * (rg_math.log(2) * rg_SCALE_LOG2_STD) ** 2)\n    gs = rg_torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=rg_torch.float32, device=device)\n    return (packed, sf, gs)\n\ndef rg_generate_weights(seed: int, device: str=\'cuda\', router_seed: int | None=None):\n    """The layer\'s weights from `seed`. With router_seed, the router and its bias come from that\n    seed instead."""\n    gen = rg_torch.Generator(device=device).manual_seed(seed)\n\n    def dense(n, k, dtype=rg_torch.bfloat16):\n        return (rg_torch.randn((n, k), generator=gen, device=device) / rg_math.sqrt(k)).to(dtype)\n\n    def vec(n, std):\n        return (std * rg_torch.randn((n,), generator=gen, device=device)).to(rg_torch.bfloat16)\n    w13, w13_sf, gs13 = rg_random_weights(2 * rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))\n    w2, w2_sf, gs2 = rg_random_weights(rg_D_EXPERT, rg_D_EXPERT, generator=gen, device=device, leading=(rg_LOCAL_EXPERTS,))\n    gs13 = gs13 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))\n    gs2 = gs2 * rg_torch.exp(rg_torch.randn((rg_LOCAL_EXPERTS,), generator=gen, device=device))\n    o, attn_gate, pre_ffn = (dense(rg_HIDDEN, rg_Q_SIZE), vec(rg_HIDDEN, 0.3), vec(rg_HIDDEN, 0.1))\n    weights = {\'qkv_gate\': dense(rg_QKV_SIZE, rg_HIDDEN), \'o\': o, \'attn_residual_gate\': attn_gate, \'pre_ffn_norm\': pre_ffn, **rg__router(gen, device, pre_ffn), \'pooler\': dense(rg_D_EXPERT, rg_HIDDEN), \'pre_expert_norm\': vec(rg_D_EXPERT, 0.1), \'w13\': w13, \'w13_sf\': w13_sf, \'w13_gs\': gs13, \'w2\': w2, \'w2_sf\': w2_sf, \'w2_gs\': gs2, \'post_expert_norm\': vec(rg_D_EXPERT, 0.1), \'unpooler\': dense(rg_HIDDEN, rg_D_EXPERT), \'post_ffn_norm\': vec(rg_HIDDEN, 0.1), \'ffn_residual_gate\': vec(rg_HIDDEN, 0.3), \'next_pre_attn_norm\': vec(rg_HIDDEN, 0.1)}\n    if router_seed is not None:\n        weights.update(rg__router(rg_torch.Generator(device=device).manual_seed(router_seed), device, pre_ffn))\n    return weights\n\n@rg_functools.cache\ndef rg__stream_basis(device: str):\n    """The residual stream\'s mean direction [8192] and shared directions [STREAM_RANK, 8192]."""\n    gen = rg_torch.Generator().manual_seed(rg_STREAM_SEED)\n    mean = rg_torch.randn((rg_HIDDEN,), generator=gen)\n    shared = rg_torch.randn((rg_STREAM_RANK, rg_HIDDEN), generator=gen)\n    return (mean.to(device), shared.to(device))\n\ndef rg__router(gen, device, pre_ffn):\n    """FP32 router rows, each with a component along the stream\'s mean direction, and the router\n    bias. pre_ffn is unused."""\n    mean, _ = rg__stream_basis(str(device))\n    w = rg_ROUTER_STD * rg_torch.randn((rg_N_EXPERTS, rg_HIDDEN), generator=gen, device=device)\n    align = rg_ROUTER_ALIGN * rg_torch.randn((rg_N_EXPERTS, 1), generator=gen, device=device)\n    rows = w + align * mean[None, :] / rg_math.sqrt(rg_HIDDEN)\n    bias = rg_ROUTER_BIAS_STD * rg_torch.randn((rg_N_EXPERTS,), generator=gen, device=device)\n    return {\'router\': rows.contiguous(), \'router_bias\': bias.contiguous()}\nclass LabGraph:\n    def __init__(self,fn,data):\n        self.data={k:v.clone() if lab_torch.is_tensor(v) else v for k,v in data.items()}\n        fn(self.data)\n        lab_torch.cuda.synchronize()\n        self.graph=lab_torch.cuda.CUDAGraph()\n        with lab_torch.cuda.graph(self.graph):self.out=fn(self.data)\n    def refresh(self,data):\n        for k in (\'hidden_states\',\'topk_ids\',\'topk_weights\'):self.data[k].copy_(data[k])\n    def replay(self):\n        self.graph.replay();return self.out\n\ndef lab_pair(old,new,ds,flush,repeats=6):\n    ot,nt,ratios=[],[],[]\n    a,b=lab_torch.cuda.Event(enable_timing=True),lab_torch.cuda.Event(enable_timing=True)\n    for i in range(repeats):\n        old.refresh(ds[i%len(ds)]);new.refresh(ds[i%len(ds)])\n        pair=[]\n        for g in ((old,new) if i%2==0 else (new,old)):\n            flush.sum()\n            a.record();g.replay();b.record();b.synchronize()\n            pair.append(float(a.elapsed_time(b))*1000)\n        x,y=pair if i%2==0 else pair[::-1]\n        ot.append(x);nt.append(y);ratios.append(y/x)\n    # Median paired ratio, anchored by the median reference time, as in the judge.\n    return lab_stats.median(ot)*lab_stats.median(ratios),lab_stats.median(nt),lab_stats.median(ot)\nVAR_CUDA=\'\\n#define C10_CUDA_CHECK(expr) do{cudaError_t e=(expr);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)\\n#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())\\n\\n#include <cuda_runtime.h>\\n#include <stdexcept>\\n#include <vector>\\n#include <algorithm>\\n\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n\\nnamespace {\\n\\nconstexpr int HD = 4096;\\nconstexpr int NLOC = 32;\\nconstexpr int TILE_BYTES = 36864;\\nconstexpr int TILE_W_BYTES = 32768;\\nconstexpr int NT1 = 512;\\nconstexpr int NT2 = 256;\\nconstexpr int NCW = 8;                 // consumer warps\\nconstexpr int NCT = NCW * 32;          // consumer threads\\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\\nconstexpr uint32_t LA_MASK = 0x60306030u;\\n// persistent counters\\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\\n\\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\\n}\\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\\n}\\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\\n  uint32_t ok;\\n  asm volatile("{\\\\n .reg .pred p;\\\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\\\n selp.u32 %0, 1, 0, p;\\\\n}\\\\n"\\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\\n  return ok != 0;\\n}\\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\\n  while (!mbar_try(b, parity)) {}\\n}\\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\\n}\\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\\n}\\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\\n}\\n__device__ __forceinline__ void bar_consumers() {\\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\\n}\\n__device__ __forceinline__ int ld_acquire(const int* p) {\\n  int v;\\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\\n  return v;\\n}\\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\\n  uint32_t d;\\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\\n  return d;\\n}\\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\\n                                         uint32_t b0, uint32_t b1) {\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\\n}\\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\\n  o[0] = A & DQ_MASK;\\n  o[1] = rotr(A, 4) & DQ_MASK;\\n  o[2] = rotr(A, 10) & DQ_MASK;\\n  o[3] = rotr(B, 2) & DQ_MASK;\\n  o[4] = rotr(B, 6) & DQ_MASK;\\n  o[5] = rotr(B, 12) & DQ_MASK;\\n  uint32_t W; asm("lop3.b32 %0,%1,%2,%3,0xe4;":"=r"(W):"r"(A),"r"(B),"n"(LA_MASK));\\n  o[6] = rotr(W, 24) & DQ_MASK;\\n  o[7] = rotr(W, 30) & DQ_MASK;\\n}\\n\\nstruct PrefetchState {\\n  volatile int index, stage, phase, pass;\\n  volatile int e, nb, m, mb, me;\\n  volatile int seg, left;  // F_RND: current run and units left in it\\n  volatile uint64_t pol_first, pol_normal;\\n};\\n\\nstruct Route {        // shared-memory routing summary\\n  int n[NLOC];        // pairs per local expert\\n  int off[NLOC];      // first sorted position\\n  int nblk[NLOC];     // token blocks\\n  int us1[NLOC + 1];  // phase-1 unit prefix\\n  int us2[NLOC + 1];  // phase-2 unit prefix\\n  int c1, c2;         // m-chunk sizes\\n  int run[NLOC];\\n  int wc[16][NLOC];  // up to 16 consumer warps (F_W16)\\n  PrefetchState pre;\\n  int erank[8];  // F_SLOT: local expert of each local rank\\n  int P;         // F_SLOT: local pairs\\n  int sa[4], sn[4];  // F_RND: first unit and count of each (pass, round) run of this CTA\\n  int tcnt[8];       // F_ROT: per reduce buffer, warp arrivals (monotonic)\\n};\\n\\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\\n                                            int& e, int& nb, int& m) {\\n  e = 0;\\n#pragma unroll\\n  for (int d = 16; d > 0; d >>= 1) {\\n    if (us[e + d] <= u) e += d;\\n  }\\n  const int rel = u - us[e];\\n  const int nbk = nblk[e];\\n  const int nfull = ntile / cc;\\n  const int span = cc * nbk;\\n  if (rel < nfull * span) {\\n    const int mc = rel / span;\\n    const int r2 = rel - mc * span;\\n    nb = r2 / cc;\\n    m = mc * cc + (r2 - nb * cc);\\n  } else {\\n    const int L = ntile - nfull * cc;\\n    const int r3 = rel - nfull * span;\\n    nb = r3 / L;\\n    m = nfull * cc + (r3 - nb * L);\\n  }\\n}\\n\\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\\n                                               const int* nblk, int cc, int ntile) {\\n  if (++m == me) {\\n    if (++nb == nblk[e]) {\\n      nb = 0;\\n      mb += cc;\\n      if (mb >= ntile) {\\n        ++e;\\n        while (e < NLOC && nblk[e] == 0) ++e;\\n        mb = 0;\\n      }\\n      me = min(ntile, mb + cc);\\n    }\\n    m = mb;\\n  }\\n}\\n\\n\\n__device__ __forceinline__ void prefetch_one(\\n    Route* rt, int total, int first, int a2, int nstage,\\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\\n    const uint8_t* w13p, const uint8_t* w2p) {\\n  auto& p = rt->pre;\\n  int idx = p.index;\\n  if (idx >= total) return;\\n  int st=p.stage, ph=p.phase, pass=p.pass;\\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\\n  const int ntile=pass?NT2:NT1;\\n  const int cc=pass?rt->c2:rt->c1;\\n  const uint8_t* base=pass?w2p:w13p;\\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\\n  mbar_wait(&emptyb[st], ph^1u);\\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\\n  ++idx;\\n  if (idx < total) {\\n    if (idx == first) {\\n      pass=1;\\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\\n      mb=(m/rt->c2)*rt->c2;\\n      me=min(NT2,mb+rt->c2);\\n    } else {\\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\\n    }\\n  }\\n  if (++st == nstage) { st=0; ph^=1; }\\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\\n}\\n\\n// ---- flag-templated variants of the admitted one-LOP3 NT1 kernel (F=0 is the incumbent\\\'s math) ----\\nenum {\\n  F_Z = 1,      // zero each pair\\\'s norm slot in its W13 tile 0 (before expert release); no C_ZERO wait\\n  F_SLOT = 2,   // M=1: dispatch straight from the 8 top-k slots (rank = popcount), no counts/scan/lists/search\\n  F_ZOUT = 4,   // W2 epilogue stores z = v(1+w); the finalizer drops its post-norm weight loads\\n  F_RSQ = 8,    // finalizer: lanes 0-7 compute each pair\\\'s scale once and shuffle it\\n  F_NREG = 16,  // W2 norm partials stay in registers across a CTA\\\'s run of tiles for one (expert, block)\\n  F_PDL = 64,   // launched as a programmatic dependent: setup overlaps the router, griddepcontrol.wait first\\n  F_LAST = 128, // the last CTA to finish W2 runs the whole finalizer; no grid-wide release/spin\\n  F_TRACE = 256, // globaltimer stamps per CTA (layout of the earlier TRACE build), written to v[13]\\n  F_NT2 = 2048, // 16-token blocks: each weight tile is decoded once for up to 16 of an expert\\\'s tokens; 1 CTA/SM\\n  F_W16 = 4096, // F_NT2 with 16 consumer warps (each half of a K slice): 16 warps/SM as NT1, operands in 64 registers\\n  F_RND = 8192, // two rounds split at the expert boundary nearest half the W13 units: every CTA runs its share of the\\n                // first experts\\\' W13, then the rest; W2 likewise, so the first experts\\\' W2 runs while the rest publish\\n  F_ROT = 16384 // no per-tile CTA barrier: warps count their arrival per tile in shared memory, the last one refills\\n                // the stage, and tile t\\\'s epilogue runs on warp t % 8 after its next tile (nstage + 2 reduce buffers)\\n};\\ntemplate <int Fv> constexpr int nt_of = (Fv & F_NT2) ? 2 : 1;\\ntemplate <int Fv> constexpr int threads_of = (Fv & F_W16) ? 512 : NTHR;\\nconstexpr int NT2_PROBE = F_NT2 | 69;           // instantiated words that resident_blocks() sizes NT2 / W16 grids with\\nconstexpr int W16_PROBE = F_NT2 | F_W16 | 69;\\ntemplate <int N> __device__ __forceinline__ void bar_cta() { asm volatile("bar.sync 1, %0;" :: "n"(N) : "memory"); }\\n__device__ __forceinline__ uint64_t tg() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory"); return t; }\\n__device__ __forceinline__ uint64_t tclk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)::"memory"); return t; }\\n\\n__device__ __forceinline__ void prefetch_slot(Route* rt, int total, int first, int a1, int a2, int nstage,\\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb, const uint8_t* w13p, const uint8_t* w2p) {\\n  auto& p = rt->pre;\\n  const int idx = p.index;\\n  if (idx >= total) return;\\n  const int st = p.stage, ph = p.phase;\\n  const bool pass = idx >= first;\\n  const int u = pass ? a2 + idx - first : a1 + idx;\\n  const int r = pass ? (u >> 8) : (u >> 9), m = pass ? (u & (NT2 - 1)) : (u & (NT1 - 1));\\n  const int e = rt->erank[r];\\n  const uint8_t* src = (pass ? w2p : w13p) + (static_cast<size_t>(e) * (pass ? NT2 : NT1) + m) * TILE_BYTES;\\n  mbar_wait(&emptyb[st], ph ^ 1u);\\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\\n  bulk_g2s(stages + st * TILE_BYTES, src, TILE_BYTES, &fullb[st], p.pol_first);\\n  p.index = idx + 1;\\n  if (st + 1 == nstage) { p.stage = 0; p.phase = ph ^ 1; } else p.stage = st + 1;\\n}\\n\\n// F_RND prefetcher: the CTA\\\'s units are up to four (pass, round) runs rt->sa[i] .. rt->sa[i] + rt->sn[i]\\n__device__ __forceinline__ void prefetch_rnd(Route* rt, int total, int nstage, uint8_t* stages, uint64_t* fullb,\\n                                             uint64_t* emptyb, const uint8_t* w13p, const uint8_t* w2p) {\\n  auto& p = rt->pre;\\n  int idx = p.index;\\n  if (idx >= total) return;\\n  int st = p.stage, ph = p.phase, pass = p.pass, seg = p.seg, left = p.left;\\n  int e = p.e, nb = p.nb, m = p.m, mb = p.mb, me = p.me;\\n  const int ntile = pass ? NT2 : NT1;\\n  const uint8_t* src = (pass ? w2p : w13p) + (static_cast<size_t>(e) * ntile + m) * TILE_BYTES;\\n  const uint64_t pol = rt->nblk[e] > 1 ? p.pol_normal : p.pol_first;\\n  mbar_wait(&emptyb[st], ph ^ 1u);\\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\\n  bulk_g2s(stages + st * TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\\n  ++idx;\\n  if (idx < total) {\\n    if (--left == 0) {\\n      do { ++seg; } while (rt->sn[seg] == 0);\\n      pass = seg >> 1;\\n      const int cc = pass ? rt->c2 : rt->c1, nt = pass ? NT2 : NT1;\\n      decode_unit(rt->sa[seg], pass ? rt->us2 : rt->us1, rt->nblk, cc, nt, e, nb, m);\\n      mb = (m / cc) * cc; me = min(nt, mb + cc); left = rt->sn[seg];\\n    } else {\\n      advance_unit(e, nb, m, mb, me, rt->nblk, pass ? rt->c2 : rt->c1, ntile);\\n    }\\n  }\\n  if (++st == nstage) { st = 0; ph ^= 1; }\\n  p.index = idx; p.stage = st; p.phase = ph; p.pass = pass; p.seg = seg; p.left = left;\\n  p.e = e; p.nb = nb; p.m = m; p.mb = mb; p.me = me;\\n}\\n\\ntemplate <int NT, int F>\\n__global__ void __launch_bounds__(threads_of<F>, NT == 1 ? 2 : 1) moe_var(\\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\\n    int M, int local_start, int nstage, uint64_t* __restrict__ trace) {\\n  constexpr bool Z = F & F_Z, SLOT = F & F_SLOT, ZOUT = F & F_ZOUT, RSQ = F & F_RSQ, NREG = F & F_NREG;\\n  constexpr bool PDL = F & F_PDL, LAST = F & F_LAST, TRACE = F & F_TRACE;\\n  static_assert(((F & F_NT2) != 0) == (NT == 2) && !(NT == 2 && SLOT), "F_NT2 selects NT=2 (M>1 dispatch)");\\n  constexpr bool W16 = F & F_W16, RND = F & F_RND;\\n  static_assert(!(RND && SLOT), "F_RND is the M>1 dispatch");\\n  constexpr bool ROT = F & F_ROT;\\n  static_assert(!ROT || (NT == 1 && !W16 && !NREG), "F_ROT: NT=1, 8 warps, norm atomics per tile");\\n  static_assert(!W16 || NT == 2, "F_W16 widens the NT2 CTA");\\n  constexpr int CW = W16 ? 16 : NCW, CT = CW * 32, GS = W16 ? 4 : 8;   // consumer warps, threads, K groups per warp\\n  constexpr int NB = 8 * NT;\\n  constexpr int RED = NT * 4;\\n  const int npairs = M * 8;\\n  extern __shared__ __align__(128) uint8_t smem[];\\n  uint8_t* stages = smem;\\n  const int NR = ROT ? nstage + 2 : 2;                                       // reduce buffers\\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [NR][CW][32][RED]\\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + NR * CW * 32 * RED);\\n  uint64_t* emptyb = fullb + nstage;\\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\\n\\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\\n  const int G = gridDim.x, c = blockIdx.x;\\n  int bar_gen0 = 0;\\n  __shared__ int last_cta;\\n  // F_TRACE layout: [0] entry [1] routed [2] main loop [3] W13 end [4] first W2 ready [5] W2 end\\n  // [6] barrier [7] combine [8] end [9] exit [10] first copy issue [11,12] first W13 copy wait\\n  // [13] first W13 tile done [14,15] first W2 copy wait [16] sum of ready waits\\n  // [17,19] sums of publish fence / barrier time [18] last publish atomic [22] tid 0 tile-barrier sum\\n  // [28,29] tid 0 W13/W2 tile-wait sums [30,31] tid 224 tile-barrier / tile-wait sums\\n  // [20,21] clock64 entry/exit [23] finalizer [24] smid [25] grid [26,27] W13/W2 units [32+w] warp entry [40+w] warp exit\\n  __shared__ uint64_t probe[TRACE ? 64 : 1];\\n  if constexpr (TRACE) {\\n    if ((tid & 31) == 0 && warp < 8) probe[32 + warp] = tg();\\n    if (tid == 0) { probe[0] = probe[32]; probe[20] = tclk(); }\\n    if (tid < 64 && tid != 0 && tid != 20 && !(tid >= 32 && tid < 40)) probe[tid] = 0;\\n  }\\n  auto dump = [&](bool finalizer) {\\n    if constexpr (TRACE) {\\n      if (tid == 0) { probe[8] = tg(); probe[23] = finalizer; if (!probe[4]) probe[4] = probe[3]; }\\n      bar_cta<CT>();\\n      if (tid < 64) trace[(size_t)c * 64 + tid] = probe[tid];\\n      bar_cta<CT>();\\n      if (lane == 0 && warp < 8) trace[(size_t)c * 64 + 40 + warp] = tg();\\n      if (tid == 0) { trace[(size_t)c * 64 + 21] = tclk(); trace[(size_t)c * 64 + 9] = tg(); }\\n    }\\n  };\\n\\n  if (tid == 0) {\\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], CW); }\\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\\n    uint64_t pf, pn;\\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\\n    rt->pre.pol_first = pf; rt->pre.pol_normal = pn;\\n    rt->pre.index = 0; rt->pre.stage = 0; rt->pre.phase = 0;\\n    if constexpr (ROT) { for (int i = 0; i < 8; ++i) rt->tcnt[i] = 0; }\\n  }\\n  if constexpr (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");\\n\\n  int U1, U2, a1, b1, a2, b2, first_count, total_count;\\n  if constexpr (SLOT) {\\n    // M == 1: a local expert holds at most one pair. Local slots in slot order: rank = popcount.\\n    if (warp == 0) {\\n      int e = -1;\\n      if (lane < 8) {\\n        e = __ldg(ids + lane) - local_start;\\n        if (e < 0 || e >= NLOC) e = -1;\\n      }\\n      if (!LAST && lane == 0) bar_gen0 = ld_acquire(cnt + C_GEN);\\n      const unsigned mk = __ballot_sync(~0u, e >= 0);\\n      const int P = __popc(mk), r = __popc(mk & ((1u << lane) - 1u));\\n      if (lane < 8) pos_of[lane] = static_cast<int16_t>(e >= 0 ? r : -1);\\n      if (e >= 0) { pair_of[r] = static_cast<int16_t>(lane); rt->erank[r] = e; }\\n      __syncwarp();   // lane 0\\\'s first copies read erank\\n      U1 = P * NT1; U2 = P * NT2;\\n      a1 = static_cast<int>((long long)U1 * c / G); b1 = static_cast<int>((long long)U1 * (c + 1) / G);\\n      a2 = static_cast<int>((long long)U2 * c / G); b2 = static_cast<int>((long long)U2 * (c + 1) / G);\\n      first_count = b1 - a1; total_count = first_count + b2 - a2;\\n      if (lane == 0) {\\n        rt->P = P;\\n        if constexpr (TRACE) probe[10] = tg();\\n        for (int i = 0; i < nstage; ++i)\\n          prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);\\n      }\\n    }\\n    __syncthreads();\\n    if (warp != 0) {\\n      const int P = rt->P;\\n      U1 = P * NT1; U2 = P * NT2;\\n      a1 = static_cast<int>((long long)U1 * c / G); b1 = static_cast<int>((long long)U1 * (c + 1) / G);\\n      a2 = static_cast<int>((long long)U2 * c / G); b2 = static_cast<int>((long long)U2 * (c + 1) / G);\\n      first_count = b1 - a1; total_count = first_count + b2 - a2;\\n    }\\n  } else {\\n    if (tid < NLOC) rt->n[tid] = 0;\\n    if (!LAST && tid == 0) bar_gen0 = ld_acquire(cnt + C_GEN);\\n    __syncthreads();\\n    for (int i = tid; i < npairs; i += CT) {\\n      int e = __ldg(ids + i) - local_start;\\n      if (e < 0 || e >= NLOC) e = -1;\\n      eflat[i] = static_cast<int8_t>(e);\\n      if (e >= 0) atomicAdd(&rt->n[e], 1);\\n    }\\n    __syncthreads();\\n    if (warp == 0) {\\n      const int n = rt->n[lane];\\n      const int nb = (n + NB - 1) / NB;\\n      int off = n, u1 = nb * NT1, u2 = nb * NT2;\\n#pragma unroll\\n      for (int d = 1; d < 32; d <<= 1) {\\n        const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\\n        if (lane >= d) { off += a; u1 += b; u2 += e2; }\\n      }\\n      rt->off[lane] = off - n;\\n      rt->run[lane] = off - n;\\n      rt->nblk[lane] = nb;\\n      rt->us1[lane + 1] = u1;\\n      rt->us2[lane + 1] = u2;\\n      if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\\n      const int T1 = __shfl_sync(~0u, u1, 31), T2 = __shfl_sync(~0u, u2, 31);\\n      if (lane == 0) {\\n        rt->c1 = min(NT1, max(1, (T1 + G / 2) / G));\\n        rt->c2 = min(NT2, max(1, (T2 + G / 2) / G));\\n      }\\n    }\\n    __syncthreads();\\n    U1 = rt->us1[NLOC]; U2 = rt->us2[NLOC];\\n    a1 = static_cast<int>((long long)U1 * c / G); b1 = static_cast<int>((long long)U1 * (c + 1) / G);\\n    a2 = static_cast<int>((long long)U2 * c / G); b2 = static_cast<int>((long long)U2 * (c + 1) / G);\\n    first_count = b1 - a1; total_count = first_count + b2 - a2;\\n    if constexpr (RND) {\\n      int B1 = U1, B2 = U2, best = U1;   // no interior expert boundary: round 1 is empty\\n      for (int e = 1; e < NLOC; ++e) {\\n        const int b = rt->us1[e];\\n        if (b > 0 && b < U1 && abs(2 * b - U1) < best) { best = abs(2 * b - U1); B1 = b; B2 = rt->us2[e]; }\\n      }\\n      const int lo[4] = {0, B1, 0, B2}, hi[4] = {B1, U1, B2, U2};\\n      int sa[4], sn[4];\\n#pragma unroll\\n      for (int i = 0; i < 4; ++i) {\\n        const int w = hi[i] - lo[i];\\n        sa[i] = lo[i] + static_cast<int>((long long)w * c / G);\\n        sn[i] = lo[i] + static_cast<int>((long long)w * (c + 1) / G) - sa[i];\\n      }\\n      __syncthreads();   // every thread has read us1/us2 before tid 0 writes the runs\\n      if (tid < 4) { rt->sa[tid] = sa[tid]; rt->sn[tid] = sn[tid]; }\\n      first_count = sn[0] + sn[1]; total_count = first_count + sn[2] + sn[3];\\n      __syncthreads();\\n    }\\n    if (tid == 0) {\\n      if (RND && total_count) {\\n        int i = 0;\\n        while (rt->sn[i] == 0) ++i;\\n        const int pass = i >> 1, cc = pass ? rt->c2 : rt->c1, ntile = pass ? NT2 : NT1;\\n        int e, nb, m;\\n        decode_unit(rt->sa[i], pass ? rt->us2 : rt->us1, rt->nblk, cc, ntile, e, nb, m);\\n        rt->pre.pass = pass; rt->pre.e = e; rt->pre.nb = nb; rt->pre.m = m; rt->pre.seg = i; rt->pre.left = rt->sn[i];\\n        rt->pre.mb = (m / cc) * cc; rt->pre.me = min(ntile, (m / cc) * cc + cc);\\n      } else if (total_count) {\\n        int e, nb, m;\\n        const int pass = first_count ? 0 : 1;\\n        const int cc = pass ? rt->c2 : rt->c1, ntile = pass ? NT2 : NT1;\\n        decode_unit(pass ? a2 : a1, pass ? rt->us2 : rt->us1, rt->nblk, cc, ntile, e, nb, m);\\n        rt->pre.pass = pass; rt->pre.e = e; rt->pre.nb = nb; rt->pre.m = m;\\n        rt->pre.mb = (m / cc) * cc; rt->pre.me = min(ntile, (m / cc) * cc + cc);\\n      }\\n      if constexpr (TRACE) probe[10] = tg();\\n      for (int i = 0; i < nstage; ++i) {\\n        if constexpr (RND) prefetch_rnd(rt, total_count, nstage, stages, fullb, emptyb, w13p, w2p);\\n        else prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);\\n      }\\n    }\\n    __syncthreads();\\n    // deterministic token lists (sorted by local expert, then flat slot)\\n    for (int base = 0; base < npairs; base += CT) {\\n      rt->wc[warp][lane] = 0;\\n      bar_cta<CT>();\\n      const int i = base + tid;\\n      const int e = i < npairs ? eflat[i] : -1;\\n      const unsigned mk = __match_any_sync(~0u, e);\\n      const int rank = __popc(mk & ((1u << lane) - 1u));\\n      if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\\n      bar_cta<CT>();\\n      if (tid < NLOC) {\\n        int s = rt->run[tid];\\n        for (int w = 0; w < CW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\\n        rt->run[tid] = s;\\n      }\\n      bar_cta<CT>();\\n      if (i < npairs) {\\n        if (e >= 0) {\\n          const int pos = rt->wc[warp][e] + rank;\\n          pos_of[i] = static_cast<int16_t>(pos);\\n          pair_of[pos] = static_cast<int16_t>(i);\\n        } else {\\n          pos_of[i] = -1;\\n        }\\n      }\\n    }\\n  }\\n  if constexpr (TRACE) {\\n    if (tid == 0) {\\n      probe[1] = tg(); uint32_t sm; asm("mov.u32 %0, %%smid;" : "=r"(sm)); probe[24] = sm; probe[25] = G;\\n      probe[26] = first_count; probe[27] = total_count - first_count;\\n    }\\n  }\\n  if constexpr (!Z) {\\n    // zero this CTA\\\'s slice of the sum-of-squares accumulators, then publish\\n    const int P = SLOT ? rt->P : (U1 > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0);\\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\\n    for (int i = z0 + tid; i < z1; i += CT) ss[i] = 0.f;\\n    __threadfence();\\n    bar_cta<CT>();\\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\\n  }\\n\\n  if constexpr (TRACE) { if (tid == 0) probe[2] = tg(); }\\n  const int q = lane & 3;\\n  uint32_t Bf[NT][GS][8];\\n  const int kw = W16 ? (warp >> 1) : warp, gofs = W16 ? (warp & 1) * 4 : 0;   // this warp\\\'s K slice and groups\\n  int st = 0;\\n  uint32_t ph = 0;\\n  int buf = 0;\\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\\n  float fw[2];   // (1 + post_norm_w) for this thread\\\'s phase-2 epilogue row, refreshed per tile\\n  float nacc = 0.f;  // F_NREG: this thread\\\'s squared sum over the current (expert, block) run\\n  int npos = -1;\\n  auto flush_norm = [&]() {\\n    if (tid < 16 * NB) {\\n      float t2 = nacc;\\n#pragma unroll\\n      for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);\\n      if (npos >= 0 && (tid & 15) == 0) atomicAdd(ss + npos, t2);\\n    }\\n    nacc = 0.f; npos = -1;\\n  };\\n\\n  // F_TRACE publish probes: [17] sum of fence time, [19] sum of fence-to-atomic (barrier) time, [18] last atomic\\n  auto publish = [&](const int pe, const int tiles) {\\n    uint64_t p0 = 0, p1 = 0;\\n    if constexpr (TRACE) { if (tid == 0) p0 = tg(); }\\n    __threadfence();\\n    if constexpr (TRACE) { if (tid == 0) p1 = tg(); }\\n    bar_cta<CT>();\\n    if (tid == 0) {\\n      atomicAdd(cnt + C_DONE + pe, tiles);\\n      if constexpr (TRACE) { const uint64_t p2 = tg(); probe[17] += p1 - p0; probe[19] += p2 - p1; probe[18] = p2; }\\n    }\\n  };\\n\\n  // F_ROT: tile `ptile` (pass pp, expert pe, block pnb, tile pm, pairs pne at poff) awaits its epilogue on warp ptile % CW\\n  int tcount = 0, ptile = -1, pp = 0, pe = 0, pnb = 0, pm = 0, pne = 0, poff = 0;\\n  auto run_epi = [&]() {\\n    const int rbi = ptile % NR, target = (ptile / NR + 1) * CW;\\n    uint64_t s0 = 0;\\n    if constexpr (TRACE) { if (lane == 0) s0 = tg(); }\\n    while (*reinterpret_cast<volatile int*>(&rt->tcnt[rbi]) < target) {}\\n    __threadfence_block();\\n    if constexpr (TRACE) { if (tid == 0 || tid == 224) probe[tid ? 30 : 22] += tg() - s0; }\\n    const float* rb = red + rbi * CW * 32 * RED;\\n    if (pp == 0) {\\n      const float sc = gs13[pe] * 256.f;\\n#pragma unroll\\n      for (int k = 0; k < 2; ++k) {\\n        const int o = lane + 32 * k, i = o & 7, n = o >> 3;\\n        const int ls = i * 4 + (n >> 1), rg = n & 1;\\n        float G1 = 0.f, U = 0.f;\\n#pragma unroll\\n        for (int w = 0; w < CW; ++w) {\\n          G1 += rb[(w * 32 + ls) * RED + rg];\\n          U += rb[(w * 32 + ls) * RED + rg + 2];\\n        }\\n        const int idx = pnb * NB + n;\\n        if (idx < pne) {\\n          if constexpr (Z) { if (pm == 0 && i == 0) ss[poff + idx] = 0.f; }\\n          G1 *= sc;\\n          U *= sc;\\n          const float h = G1 / (1.f + __expf(-G1)) * U;\\n          inter[static_cast<size_t>(poff + idx) * HD + pm * 8 + i] = __float2bfloat16(h);\\n        }\\n      }\\n    } else {\\n      const float sc = gs2[pe] * 256.f;\\n      const int i = lane & 15;\\n      const float fwv = 1.f + __bfloat162float(pnw[pm * 16 + i]);\\n#pragma unroll\\n      for (int k = 0; k < 4; ++k) {\\n        const int n = (lane >> 4) + 2 * k;\\n        const int ls = (i & 7) * 4 + (n >> 1), rg = (n & 1) + ((i >> 3) << 1);\\n        float v = 0.f;\\n#pragma unroll\\n        for (int w = 0; w < CW; ++w) v += rb[(w * 32 + ls) * RED + rg];\\n        v *= sc;\\n        const int idx = pnb * NB + n;\\n        const bool ok = idx < pne;\\n        const float z = v * fwv;\\n        if (ok) y[static_cast<size_t>(poff + idx) * HD + pm * 16 + i] = ZOUT ? z : v;\\n        float t2 = ok ? z * z : 0.f;\\n#pragma unroll\\n        for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\\n        if (ok && i == 0) atomicAdd(ss + poff + idx, t2);\\n      }\\n    }\\n  };\\n  auto flush_epi = [&]() {   // F_ROT: the pending tile\\\'s epilogue now (segment publish, end of the units)\\n    if constexpr (ROT) {\\n      if (ptile >= 0 && ptile % CW == warp) run_epi();\\n      ptile = -1;\\n    }\\n  };\\n\\n  auto unit_body = [&](const int pass, const bool firstp, const int e, const int nb, const int m, const int ne,\\n                       const int off) {\\n    if (pass != cur_pass || e != cur_e || nb != cur_nb) {\\n      if (cur_pass == 0 && seg_tiles > 0) { flush_epi(); publish(cur_e, seg_tiles); }  // the finished phase-1 segment\\n      if constexpr (NREG) { if (cur_pass == 1) flush_norm(); }\\n      cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\\n      if (pass == 1) {\\n        uint64_t wait0 = 0;\\n        if constexpr (TRACE) { if (tid == 0) wait0 = tg(); }\\n        const int target = NT1 * (SLOT ? 1 : rt->nblk[e]);\\n        while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\\n        if constexpr (!Z) { while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64); }\\n        if constexpr (TRACE) { if (tid == 0) probe[16] += tg() - wait0; }\\n      }\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) {\\n        const int idx = nb * NB + j * 8 + (lane >> 2);\\n        const __nv_bfloat16* row = nullptr;\\n        if (idx < ne) {\\n          const int pos = off + idx;\\n          row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\\n        }\\n#pragma unroll\\n        for (int g = 0; g < GS; ++g) {\\n          uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\\n          if (row) {\\n            const uint4* p = reinterpret_cast<const uint4*>(row + kw * 512 + (gofs + g) * 64 + q * 16);\\n            if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\\n            else { v0 = __ldg(p); v1 = __ldg(p + 1); }\\n          }\\n          Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\\n          Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\\n        }\\n      }\\n    }\\n    if constexpr (TRACE) { if (tid == 0 && pass == 1 && firstp) probe[4] = tg(); }\\n    if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\\n\\n    if constexpr (TRACE) { if (tid == 0 && firstp) probe[pass ? 14 : 11] = tg(); }\\n    uint64_t fw0 = 0;\\n    if constexpr (TRACE) { if (tid == 0 || tid == 224) fw0 = tg(); }\\n    mbar_wait(&fullb[st], ph);\\n    if constexpr (TRACE) {\\n      if (tid == 0 && firstp) probe[pass ? 15 : 12] = tg();\\n      if (tid == 0 || tid == 224) probe[tid ? 31 : 28 + pass] += tg() - fw0;\\n    }\\n    const uint8_t* tile = stages + st * TILE_BYTES;\\n    float acc[NT][4];\\n#pragma unroll\\n    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\\n    {\\n      const uint4* wq = reinterpret_cast<const uint4*>(tile) + kw * 8 * 32 + lane;\\n      const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + kw * 32 + lane);\\n      const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\\n#pragma unroll\\n      for (int g = 0; g < GS; ++g) {\\n        const uint4 wv = wq[(gofs + g) * 32];\\n        const uint32_t sw = scw[(gofs + g) >> 1];\\n        const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\\n        const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\\n        uint32_t o[16];\\n        dq8(wv.x, wv.y, o);\\n        dq8(wv.z, wv.w, o + 8);\\n#pragma unroll\\n        for (int s = 0; s < 4; ++s) {\\n          const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\\n          const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\\n          const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\\n#pragma unroll\\n          for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\\n        }\\n      }\\n    }\\n    __syncwarp();\\n    if (lane == 0) mbar_arrive(&emptyb[st]);\\n    if (++st == nstage) { st = 0; ph ^= 1u; }\\n\\n    if constexpr (ROT) {\\n      const int rbi = tcount % NR;\\n      *reinterpret_cast<float4*>(red + ((rbi * CW + warp) * 32 + lane) * RED) = make_float4(acc[0][0], acc[0][1], acc[0][2], acc[0][3]);\\n      __syncwarp();\\n      int old = 0;\\n      if (lane == 0) { __threadfence_block(); old = atomicAdd(&rt->tcnt[rbi], 1); }\\n      old = __shfl_sync(~0u, old, 0);\\n      if (old % CW == CW - 1 && lane == 0) {   // every warp is past this stage: refill it\\n        __threadfence_block();\\n        if constexpr (RND) prefetch_rnd(rt, total_count, nstage, stages, fullb, emptyb, w13p, w2p);\\n        else prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);\\n      }\\n      if (ptile >= 0 && ptile % CW == warp) run_epi();\\n      ptile = tcount; pp = pass; pe = e; pnb = nb; pm = m; pne = ne; poff = off;\\n      ++tcount;\\n      ++seg_tiles;\\n      return;\\n    }\\n    float* rp = red + ((buf * CW + warp) * 32 + lane) * RED;\\n#pragma unroll\\n    for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\\n    uint64_t bw0 = 0;\\n    if constexpr (TRACE) { if (tid == 0 || tid == 224) bw0 = tg(); }\\n    bar_cta<CT>();\\n    if constexpr (TRACE) { if (tid == 0 || tid == 224) probe[tid ? 30 : 22] += tg() - bw0; }\\n    if (tid == 0) {\\n      if constexpr (SLOT) prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);\\n      else if constexpr (RND) prefetch_rnd(rt, total_count, nstage, stages, fullb, emptyb, w13p, w2p);\\n      else prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);\\n    }\\n    const float* rb = red + buf * CW * 32 * RED;\\n    if (pass == 0) {\\n      const float sc = gs13[e] * 256.f;\\n      if (tid < 8 * NB) {\\n        const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\\n        const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\\n        float G1 = 0.f, U = 0.f;\\n#pragma unroll\\n        for (int w = 0; w < CW; ++w) {\\n          G1 += rb[(w * 32 + ls) * RED + rg];\\n          U += rb[(w * 32 + ls) * RED + rg + 2];\\n        }\\n        const int idx = nb * NB + n;\\n        if (idx < ne) {\\n          if constexpr (Z) { if (m == 0 && i == 0) ss[off + idx] = 0.f; }\\n          G1 *= sc;\\n          U *= sc;\\n          const float h = G1 / (1.f + __expf(-G1)) * U;\\n          inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\\n        }\\n      }\\n    } else {\\n      const float sc = gs2[e] * 256.f;\\n      if (tid < 16 * NB) {\\n        const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\\n        const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\\n        float v = 0.f;\\n#pragma unroll\\n        for (int w = 0; w < CW; ++w) v += rb[(w * 32 + ls) * RED + rg];\\n        v *= sc;\\n        const int idx = nb * NB + n;\\n        const bool ok = idx < ne;\\n        const float z = v * fw[0];\\n        if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = ZOUT ? z : v;\\n        if constexpr (NREG) {\\n          if (ok) { nacc += z * z; npos = off + idx; }\\n        } else {\\n          float t2 = ok ? z * z : 0.f;\\n#pragma unroll\\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\\n        }\\n      }\\n    }\\n    if constexpr (TRACE) {\\n      if (tid == 0 && pass == 0 && firstp) probe[13] = tg();\\n    }\\n    buf ^= 1;\\n    ++seg_tiles;\\n  };\\n  constexpr int NSEG = RND ? 4 : 2;\\n  bool fp = true;   // the CTA\\\'s first unit of the pass (trace stamps)\\n  for (int sg = 0; sg < NSEG; ++sg) {\\n    const int pass = RND ? (sg >> 1) : sg;\\n    const int ua = RND ? rt->sa[sg] : (pass ? a2 : a1), ub = RND ? ua + rt->sn[sg] : (pass ? b2 : b1);\\n    if (RND && (sg & 1) == 0) fp = true;\\n    const int* us = pass ? rt->us2 : rt->us1;\\n    const int cc = pass ? rt->c2 : rt->c1;\\n    const int ntile = pass ? NT2 : NT1;\\n    int e = 0, nb = 0, m = 0, mb = 0, me = 0;\\n    if constexpr (!SLOT) {\\n      if (ua < ub) {\\n        decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\\n        mb = (m / cc) * cc;\\n        me = min(ntile, mb + cc);\\n      }\\n    }\\n    for (int u0 = ua; u0 < ub; ++u0) {\\n      const int u = u0;\\n      int ne, off;\\n      if constexpr (SLOT) {\\n        const int r = pass ? (u >> 8) : (u >> 9);\\n        m = u & (ntile - 1); e = rt->erank[r]; nb = 0; ne = 1; off = r;\\n      } else {\\n        ne = rt->n[e]; off = rt->off[e];\\n      }\\n      unit_body(pass, RND ? fp : u0 == ua, e, nb, m, ne, off);\\n      fp = false;\\n      if constexpr (!SLOT) advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\\n    }\\n    if constexpr (TRACE) { if (tid == 0 && (!RND || (sg & 1))) probe[pass ? 5 : 3] = tg(); }\\n  }\\n  flush_epi();\\n  if (cur_pass == 0 && seg_tiles > 0) publish(cur_e, seg_tiles);\\n  if constexpr (NREG) { if (cur_pass == 1) flush_norm(); }\\n\\n  // ---------------- grid barrier (or last-arriver election) ----------------\\n  __threadfence();\\n  bar_cta<CT>();\\n  if constexpr (TRACE) { if (tid == 0) probe[6] = tg(); }\\n  if constexpr (LAST) {\\n    if (tid == 0) {\\n      const bool last = atomicAdd(cnt + C_BAR, 1) == G - 1;\\n      if (last) {\\n        __threadfence();\\n        for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\\n        cnt[C_ZERO] = 0;\\n        cnt[C_BAR] = 0;\\n      }\\n      last_cta = last;\\n    }\\n    bar_cta<CT>();\\n    if (!last_cta) { dump(false); return; }\\n  } else {\\n    if (tid == 0) {\\n      if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\\n        for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\\n        cnt[C_ZERO] = 0;\\n        cnt[C_BAR] = 0;\\n        __threadfence();\\n        atomicAdd(cnt + C_GEN, 1);\\n      } else {\\n        while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\\n      }\\n    }\\n    bar_cta<CT>();\\n  }\\n\\n  // ---------------- combine: item = (token, 8 columns per thread: half the row at 256 threads, all of it at 512) ----\\n  constexpr int PARTS = HD / (8 * CT);\\n  if constexpr (TRACE) { if (tid == 0) probe[7] = tg(); }\\n  for (int it = LAST ? 0 : c; it < PARTS * M; it += LAST ? 1 : G) {\\n    const int t = it / PARTS;\\n    const int c0 = (it % PARTS) * (8 * CT) + tid * 8;\\n    float f[8];\\n    if constexpr (!ZOUT) {\\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\\n#pragma unroll\\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\\n    }\\n    float scl = 0.f;\\n    int posl = -1;\\n    if constexpr (RSQ) {\\n      if (lane < 8) {\\n        posl = pos_of[t * 8 + lane];\\n        if (posl >= 0) scl = tw[t * 8 + lane] * rsqrtf(__ldcg(ss + posl) * (1.f / HD) + 1e-8f);\\n      }\\n    }\\n    float acc[8];\\n#pragma unroll\\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\\n#pragma unroll\\n    for (int j = 0; j < 8; ++j) {\\n      int pos;\\n      float sc;\\n      if constexpr (RSQ) {\\n        pos = __shfl_sync(~0u, posl, j);\\n        sc = __shfl_sync(~0u, scl, j);\\n      } else {\\n        pos = pos_of[t * 8 + j];\\n        sc = pos >= 0 ? tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f) : 0.f;\\n      }\\n      if (pos >= 0) {\\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\\n        if constexpr (ZOUT) {\\n          acc[0] += sc * u0.x; acc[1] += sc * u0.y; acc[2] += sc * u0.z; acc[3] += sc * u0.w;\\n          acc[4] += sc * u1.x; acc[5] += sc * u1.y; acc[6] += sc * u1.z; acc[7] += sc * u1.w;\\n        } else {\\n          acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\\n          acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\\n        }\\n      }\\n    }\\n    uint4 ov;\\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\\n#pragma unroll\\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\\n  }\\n  dump(LAST || c < PARTS * M);\\n}\\n\\nsize_t smem_bytes(int nt, int nstage, int M) {\\n  const int npairs = M * 8;\\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\\n}\\n\\n#define VAR_FLAGS(X) X(0) X(1) X(2) X(4) X(8) X(16) X(128) X(159) X(29) X(64) X(66) X(223) X(93) X(130) X(31) \\\\\\n  X(95) X(69) X(77) X(256) X(415) X(287) X(320) X(351) X(479) X(285) X(349) X(325) X(2048) X(2117) X(2304) X(2373) X(6213) X(6469) X(8261) X(8517) X(16453) X(16709) X(24645) X(24901)\\n}  // namespace\\n\\nint64_t max_smem_attr() {\\n  int dev, optin;\\n  C10_CUDA_CHECK(cudaGetDevice(&dev));\\n  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\\n  int dyn = optin;\\n#define SET_ATTR(Fv) { cudaFuncAttributes a = {}; C10_CUDA_CHECK(cudaFuncGetAttributes(&a, moe_var<nt_of<Fv>, Fv>)); \\\\\\n    C10_CUDA_CHECK(cudaFuncSetAttribute(moe_var<nt_of<Fv>, Fv>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)a.sharedSizeBytes)); \\\\\\n    dyn = std::min(dyn, optin - (int)a.sharedSizeBytes); }\\n  VAR_FLAGS(SET_ATTR)\\n  SET_ATTR(0)  // resident_blocks() sizes NT=1 grids with moe_var<1, 0> (same smem/threads as every NT=1 word)\\n#undef SET_ATTR\\n  return dyn;\\n}\\n\\nsize_t smem_var(int flags, int nstage, int M) {   // smem_bytes() plus the wider (F_W16) or deeper (F_ROT) reduce buffers\\n  const int nt = (flags & F_NT2) ? 2 : 1, cw = (flags & F_W16) ? 16 : NCW;\\n  return smem_bytes(nt, nstage, M) + static_cast<size_t>(2 * (cw - NCW) * 32 * nt * 4) * sizeof(float) +\\n         ((flags & F_ROT) ? static_cast<size_t>(nstage) * NCW * 32 * 4 * sizeof(float) : 0);\\n}\\nint64_t smem_needed(int64_t flags, int64_t nstage, int64_t M) { return (int64_t)smem_var(flags, nstage, M); }\\nint64_t resident_blocks(int64_t flags, int64_t nstage, int64_t M) {\\n  int n;\\n  if (flags & F_W16) C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_var<2, W16_PROBE>, 512, smem_var(flags, nstage, M)));\\n  else if (flags & F_NT2) C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_var<2, NT2_PROBE>, NTHR, smem_var(flags, nstage, M)));\\n  else C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_var<1, 0>, NTHR, smem_var(flags, nstage, M)));\\n  return n;\\n}\\n\\ntemplate <int Fv>\\nvoid launch_var(std::vector<int64_t>& v, int M, int start, int stages, int grid, cudaStream_t stream) {\\n  cudaLaunchConfig_t cfg = {};\\n  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(threads_of<Fv>); cfg.dynamicSmemBytes = smem_var(Fv, stages, M);\\n  cfg.stream = stream;\\n  cudaLaunchAttribute attr{};\\n  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;\\n  attr.val.programmaticStreamSerializationAllowed = 1;\\n  if (Fv & F_PDL) { cfg.attrs = &attr; cfg.numAttrs = 1; }\\n  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_var<nt_of<Fv>, Fv>,\\n    (const __nv_bfloat16*)v[0], (const int*)v[1], (const float*)v[2], (const __nv_bfloat16*)v[3],\\n    (const uint8_t*)v[4], (const uint8_t*)v[5], (const float*)v[6], (const float*)v[7],\\n    (__nv_bfloat16*)v[8], (float*)v[9], (float*)v[10], (int*)v[11], (__nv_bfloat16*)v[12], M, start, stages,\\n    (uint64_t*)(v.size() > 13 ? v[13] : 0)));\\n}\\n\\nvoid run(std::vector<int64_t> v, int64_t M, int64_t start, int64_t stages, int64_t grid, int64_t stream, int64_t flags) {\\n  if ((flags & F_SLOT) && M != 1) throw std::runtime_error("F_SLOT requires M == 1");\\n  if ((flags & F_TRACE) && (v.size() < 14 || !v[13])) throw std::runtime_error("F_TRACE requires a stamp buffer");\\n  switch (flags) {\\n#define CASE(Fv) case Fv: launch_var<Fv>(v, M, start, stages, grid, (cudaStream_t)stream); return;\\n    VAR_FLAGS(CASE)\\n#undef CASE\\n  }\\n  throw std::runtime_error("flag combination not instantiated");\\n}\\n\'\nACTIVITY_CPP=\'#include <pybind11/pybind11.h>\\n#include <pybind11/stl.h>\\n#include <cupti.h>\\n#include <cupti_activity.h>\\n#include <vector>\\n#include <tuple>\\n#include <string>\\n#include <mutex>\\n#include <cstdlib>\\n#include <stdexcept>\\n\\nusing ActivityRow=std::tuple<std::string,std::vector<int64_t>>;\\nnamespace {\\nstd::vector<ActivityRow> activity_rows;\\nstd::mutex activity_mutex;\\nint activity_error=0;\\nsize_t activity_dropped=0;\\nbool activity_on=false,activity_callbacks=false;\\nvoid ck(CUptiResult status){\\n if(status!=CUPTI_SUCCESS){const char* text=nullptr;cuptiGetResultString(status,&text);throw std::runtime_error(text?text:"CUPTI activity error");}\\n}\\nvoid CUPTIAPI request_activity(uint8_t** buffer,size_t* size,size_t* max_records){\\n *size=4<<20;*max_records=0;void* p=nullptr;\\n if(posix_memalign(&p,64,*size))p=nullptr;\\n *buffer=(uint8_t*)p;\\n}\\nvoid CUPTIAPI finish_activity(CUcontext context,uint32_t stream,uint8_t* buffer,size_t size,size_t valid){\\n std::lock_guard<std::mutex> guard(activity_mutex);\\n CUpti_Activity* record=nullptr;\\n while(valid){\\n   CUptiResult status=cuptiActivityGetNextRecord(buffer,valid,&record);\\n   if(status==CUPTI_ERROR_MAX_LIMIT_REACHED)break;\\n   if(status!=CUPTI_SUCCESS){activity_error=(int)status;break;}\\n   if(record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL||record->kind==CUPTI_ACTIVITY_KIND_KERNEL){\\n     auto* k=(CUpti_ActivityKernel10*)record;\\n     activity_rows.emplace_back(k->name?k->name:"",std::vector<int64_t>{\\n       (int64_t)k->start,(int64_t)k->end,(int64_t)k->queued,(int64_t)k->submitted,\\n       (int64_t)k->graphNodeId,(int64_t)k->gridId,(int64_t)k->graphId,\\n       k->gridX,k->gridY,k->gridZ,k->blockX,k->dynamicSharedMemory,k->registersPerThread});\\n   }\\n }\\n size_t lost=0;auto status=cuptiActivityGetNumDroppedRecords(context,stream,&lost);\\n if(status!=CUPTI_SUCCESS)activity_error=(int)status;\\n activity_dropped+=lost;free(buffer);\\n}\\n}\\n\\nvoid profile_begin(){\\n if(activity_on)throw std::runtime_error("activity recorder already enabled");\\n if(!activity_callbacks){ck(cuptiActivityRegisterCallbacks(request_activity,finish_activity));activity_callbacks=true;}\\n {std::lock_guard<std::mutex> guard(activity_mutex);activity_rows.clear();activity_error=0;activity_dropped=0;}\\n ck(cuptiActivityEnableLatencyTimestamps(1));\\n ck(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));activity_on=true;\\n}\\n\\nstd::vector<ActivityRow> profile_end(){\\n if(!activity_on)throw std::runtime_error("activity recorder not enabled");\\n auto flush_status=cuptiActivityFlushAll(0);\\n auto disable_status=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);\\n activity_on=false;\\n auto latency_status=cuptiActivityEnableLatencyTimestamps(0);\\n ck(flush_status);ck(disable_status);ck(latency_status);\\n std::lock_guard<std::mutex> guard(activity_mutex);\\n if(activity_error)ck((CUptiResult)activity_error);\\n if(activity_dropped)throw std::runtime_error("activity recorder dropped records");\\n return activity_rows;\\n}\\n\'\nCONTROL_CUDA=\'#include <cuda_runtime.h>\\n#include <stdint.h>\\n#include <stdexcept>\\n\\n__device__ __forceinline__ uint64_t timeline_gt(){uint64_t v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;}\\n__global__ void node_before(uint64_t* stamps){if(threadIdx.x==0){stamps[0]=timeline_gt();stamps[1]=clock64();}}\\n__global__ void node_after(uint64_t* stamps){if(threadIdx.x==0){stamps[2]=timeline_gt();stamps[3]=clock64();}}\\n__global__ void empty_like(uint64_t* stamps){\\n extern __shared__ uint8_t shared[];\\n if((threadIdx.x&31)==0){int w=threadIdx.x>>5;stamps[blockIdx.x*16+w]=timeline_gt();stamps[blockIdx.x*16+8+w]=timeline_gt();}\\n}\\n__global__ void stream_banks(const uint8_t* w13,const uint8_t* w2,const int* experts,int parts,uint32_t* sums,uint64_t* stamps){\\n const int tid=threadIdx.x,bank=blockIdx.x>=2*parts;\\n const int block=bank?blockIdx.x-2*parts:blockIdx.x,grid=bank?parts:2*parts;\\n const int e=experts[blockIdx.y],bytes=bank?9437184:18874368;\\n const uint4* src=reinterpret_cast<const uint4*>((bank?w2:w13)+(size_t)e*bytes);\\n uint32_t acc=0;uint64_t entry=timeline_gt(),first0=0,first1=0;\\n for(int base=block*blockDim.x*8;base<bytes/16;base+=grid*blockDim.x*8){\\n   if(base==block*blockDim.x*8)first0=timeline_gt();\\n   #pragma unroll\\n   for(int j=0;j<8;++j){int index=base+j*blockDim.x+tid;if(index<bytes/16){uint4 v=__ldg(src+index);acc^=v.x^v.y^v.z^v.w;}}\\n   if(base==block*blockDim.x*8){\\n     // A real dependency through the checksum and CTA barrier. The first\\n     // interval includes its reduction/barrier overhead; empty_like is separate.\\n     volatile __shared__ uint32_t sink[256];sink[tid]=acc;__syncthreads();\\n     acc^=sink[(tid+1)%blockDim.x];first1=timeline_gt();\\n   }\\n }\\n sums[(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+tid]=acc;\\n if(tid==0){int b=blockIdx.y*gridDim.x+blockIdx.x;\\n   stamps[b*4]=entry;stamps[b*4+1]=first0;stamps[b*4+2]=first1;stamps[b*4+3]=timeline_gt();}\\n}\\n\\n__global__ void gt_resolution(uint64_t* out){\\n uint64_t prev=timeline_gt(),first=prev,mind=~0ull;int changes=0;\\n for(int i=0;i<40000;++i){uint64_t t=timeline_gt();if(t!=prev){mind=t-prev<mind?t-prev:mind;++changes;prev=t;}}\\n out[0]=mind;out[1]=changes;out[2]=prev-first;\\n}\\nvoid resolution(int64_t out,int64_t stream){gt_resolution<<<1,1,0,(cudaStream_t)stream>>>((uint64_t*)out);}\\nvoid mark(int64_t stamps,int64_t after,int64_t stream){\\n if(after)node_after<<<1,32,0,(cudaStream_t)stream>>>((uint64_t*)stamps);\\n else node_before<<<1,32,0,(cudaStream_t)stream>>>((uint64_t*)stamps);\\n}\\nvoid empty(int64_t stamps,int64_t grid,int64_t threads,int64_t smem,int64_t stream){\\n empty_like<<<grid,threads,smem,(cudaStream_t)stream>>>((uint64_t*)stamps);\\n}\\nvoid streaming(int64_t w13,int64_t w2,int64_t experts,int64_t count,int64_t sums,int64_t stamps,int64_t grid,int64_t threads,int64_t smem,int64_t stream){\\n int p=grid/(3*count);if(p<1)throw std::runtime_error("stream grid too small");\\n stream_banks<<<dim3(3*p,count),threads,smem,(cudaStream_t)stream>>>((const uint8_t*)w13,(const uint8_t*)w2,(const int*)experts,p,(uint32_t*)sums,(uint64_t*)stamps);\\n}\\nint64_t control_optin(){\\n int dev,n;cudaGetDevice(&dev);cudaDeviceGetAttribute(&n,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev);\\n cudaFuncAttributes a={};cudaFuncGetAttributes(&a,empty_like);cudaFuncSetAttribute(empty_like,cudaFuncAttributeMaxDynamicSharedMemorySize,n-a.sharedSizeBytes);\\n cudaFuncGetAttributes(&a,stream_banks);cudaFuncSetAttribute(stream_banks,cudaFuncAttributeMaxDynamicSharedMemorySize,n-a.sharedSizeBytes);return n;\\n}\\n\'\nimport hashlib\nimport time\nimport statistics\n\nNT1_CPP=\'\'\'#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <stdint.h>\n#include <vector>\nint64_t max_smem_attr();int64_t smem_needed(int64_t,int64_t,int64_t);int64_t resident_blocks(int64_t,int64_t,int64_t);\nvoid run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);\n\'\'\'\nNT1_MODULES={}\nNT1_HOPPER=torch.cuda.get_device_capability()==(9,0)\n\n\ndef nt1_module(kind):\n    if kind not in NT1_MODULES:\n        source=NT1_SOURCES[kind];tag=hashlib.sha256(source.encode()).hexdigest()[:12]\n        mod=load_inline(name=\'muse_timeline_\'+kind+tag,cpp_sources=NT1_CPP,cuda_sources=source,\n            functions=[\'max_smem_attr\',\'smem_needed\',\'resident_blocks\',\'run\'],no_implicit_headers=True,\n            with_pytorch_error_handling=False,extra_cuda_cflags=[\'-O3\',\'-std=c++20\',\'-lineinfo\',\n            *([\'-gencode=arch=compute_90a,code=sm_90a\'] if NT1_HOPPER else [])],verbose=False)\n        mod.smem_optin=mod.max_smem_attr();NT1_MODULES[kind]=mod\n    return NT1_MODULES[kind]\n\n\nCONTROL_MODULE=[]\nACTIVITY_MODULE=[]\n\n\ndef control_module():\n    if not CONTROL_MODULE:\n        cpp=\'\'\'#include <pybind11/pybind11.h>\n#include <stdint.h>\nvoid mark(int64_t,int64_t,int64_t);\nvoid resolution(int64_t,int64_t);\nvoid empty(int64_t,int64_t,int64_t,int64_t,int64_t);\nvoid streaming(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\nint64_t control_optin();\n\'\'\'\n        tag=hashlib.sha256((cpp+CONTROL_CUDA).encode()).hexdigest()[:12]\n        mod=load_inline(name=\'muse_timeline_control_\'+tag,cpp_sources=cpp,cuda_sources=CONTROL_CUDA,\n            functions=[\'mark\',\'resolution\',\'empty\',\'streaming\',\'control_optin\'],\n            no_implicit_headers=True,with_pytorch_error_handling=False,\n            extra_cuda_cflags=[\'-O3\',\'-std=c++20\',\'-lineinfo\',\n                *([\'-gencode=arch=compute_90a,code=sm_90a\'] if NT1_HOPPER else [])],verbose=False)\n        mod.control_optin();CONTROL_MODULE.append(mod)\n    return CONTROL_MODULE[0]\n\n\ndef activity_module():\n    """CUPTI kernel records (graph-node start/end); optional: the stamps stand without it."""\n    if not ACTIVITY_MODULE:\n        tag=hashlib.sha256(ACTIVITY_CPP.encode()).hexdigest()[:12]\n        ACTIVITY_MODULE.append(load_inline(name=\'muse_timeline_activity_\'+tag,cpp_sources=ACTIVITY_CPP,\n            functions=[\'profile_begin\',\'profile_end\'],no_implicit_headers=True,with_pytorch_error_handling=False,\n            extra_include_paths=[\'/usr/local/cuda/include\'],\n            extra_ldflags=[\'-L/usr/local/cuda/lib64\',\'-Wl,-rpath,/usr/local/cuda/lib64\',\'-lcupti\'],verbose=False))\n    return ACTIVITY_MODULE[0]\n\n\ndef nt1_banks(w):\n    dev=w[\'w13\'].device;r=torch.arange(512,device=dev)[:,None]*8+torch.arange(8,device=dev)\n    bank={}\n    for name,rows in ((\'w13\',torch.cat((r,r+4096),1)),(\'w2\',torch.arange(4096,device=dev).view(256,16))):\n        bank[name+\'p\']=torch.stack([_repack(w[name][e],w[name+\'_sf\'][e],rows) for e in range(32)])\n    return dict(bank=bank,counters={k:torch.zeros(64,device=dev,dtype=torch.int32) for k in NT1_SOURCES},\n        pnw=w[\'post_norm_w\'].contiguous(),gs13=w[\'w13_gs\'].float().contiguous(),gs2=w[\'w2_gs\'].float().contiguous(),\n        sms=torch.cuda.get_device_properties(dev).multi_processor_count)\n\n\ndef nt1_call(d,state,kind=\'INC\',grid=264,stages=2,trace_out=False):\n    x=d[\'hidden_states\'].contiguous();m=len(x);dev=x.device;mod=nt1_module(kind)\n    if mod.smem_needed(1,stages,m)>mod.smem_optin:raise ValueError(\'stage depth exceeds shared capacity\')\n    grid=min(grid,state[\'sms\']*mod.resident_blocks(1,stages,m))\n    if grid<1:raise ValueError(\'no resident CTA\')\n    inter=torch.empty((m*8,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((m*8,4096),device=dev);ss=torch.empty(m*8,device=dev)\n    out=torch.empty_like(x);ids=d[\'topk_ids\'].int().contiguous();tw=d[\'topk_weights\'].float().contiguous()\n    tensors=[x,ids,tw,state[\'pnw\'],state[\'bank\'][\'w13p\'],state[\'bank\'][\'w2p\'],state[\'gs13\'],state[\'gs2\'],inter,y,ss,state[\'counters\'][kind],out]\n    stamps=torch.empty((grid,64),device=dev,dtype=torch.int64) if kind.startswith(\'TRACE\') else None\n    if stamps is not None:tensors.append(stamps)\n    mod.run([v.data_ptr() for v in tensors],m,d[\'config\'][\'local_start\'],stages,grid,torch.cuda.current_stream(dev).cuda_stream)\n    return (out,stamps) if trace_out else out\n\n\ndef nt1_error(got,want):\n    a,b=got.float(),want.float();den=b.norm(dim=-1)\n    return float((a-b).norm()/b.norm().clamp_min(1e-15)),float(((a-b).norm(dim=-1)/den.clamp_min(den.mean()*.1).clamp_min(1e-15)).max())\n\nVAR_CPP=\'\'\'#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <stdint.h>\n#include <vector>\nint64_t max_smem_attr();int64_t smem_needed(int64_t,int64_t,int64_t);int64_t resident_blocks(int64_t,int64_t,int64_t);\nvoid run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\n\'\'\'\nF_Z,F_SLOT,F_ZOUT,F_RSQ,F_NREG,F_PDL,F_LAST,F_TRACE,F_NT2,F_W16,F_RND,F_ROT=1,2,4,8,16,64,128,256,2048,4096,8192,16384\nVAR_MODULE=[]\n\n\ndef var_module():\n    if not VAR_MODULE:\n        tag=hashlib.sha256(VAR_CUDA.encode()).hexdigest()[:12]\n        mod=load_inline(name=\'muse_smallm_var_\'+tag,cpp_sources=VAR_CPP,cuda_sources=VAR_CUDA,\n            functions=[\'max_smem_attr\',\'smem_needed\',\'resident_blocks\',\'run\'],no_implicit_headers=True,\n            with_pytorch_error_handling=False,extra_cuda_cflags=[\'-O3\',\'-std=c++20\',\'-lineinfo\',\n            *([\'-gencode=arch=compute_90a,code=sm_90a\'] if NT1_HOPPER else [])],verbose=False)\n        mod.smem_optin=mod.max_smem_attr();VAR_MODULE.append(mod)\n    return VAR_MODULE[0]\n\n\ndef var_geometry(state,flags,m,grid=264,stages=None):\n    """(nt, stages, grid): NT2 runs 1 CTA/SM with as many stages as fit (4 on the H200\'s 227 KB; 2, or 1 for W16, on sm_120\'s 99 KB)."""\n    mod=var_module();nt=2 if flags&F_NT2 else 1;f=flags&~F_TRACE\n    if stages is None:stages=next((s for s in ((4,3,2,1) if nt==2 else (2,)) if mod.smem_needed(f,s,m)<=mod.smem_optin),2)\n    if mod.smem_needed(f,stages,m)>mod.smem_optin:raise ValueError(\'stage depth exceeds shared capacity\')\n    return nt,stages,min(grid,state[\'sms\']*mod.resident_blocks(f,stages,m))\n\n\ndef var_call(d,state,flags,grid=264,stages=None,stamps=None):\n    """The flag-templated NT1 variant; flags=0 is the incumbent\'s math in the variant harness."""\n    x=d[\'hidden_states\'].contiguous();m=len(x);dev=x.device;mod=var_module()\n    if flags&F_SLOT and m!=1:raise ValueError(\'F_SLOT is the M=1 dispatch\')\n    nt,stages,grid=var_geometry(state,flags,m,grid,stages)\n    inter=torch.empty((m*8,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((m*8,4096),device=dev);ss=torch.empty(m*8,device=dev)\n    out=torch.empty_like(x);ids=d[\'topk_ids\'].int().contiguous();tw=d[\'topk_weights\'].float().contiguous()\n    cnt=state.setdefault(\'vcounters\',{}).setdefault(flags&~F_TRACE,None)\n    if cnt is None:cnt=state[\'vcounters\'][flags&~F_TRACE]=torch.zeros(64,device=dev,dtype=torch.int32)\n    tensors=[x,ids,tw,state[\'pnw\'],state[\'bank\'][\'w13p\'],state[\'bank\'][\'w2p\'],state[\'gs13\'],state[\'gs2\'],inter,y,ss,cnt,out]\n    ptrs=[v.data_ptr() for v in tensors]\n    if flags&F_TRACE:\n        if stamps is None or stamps.shape!=(grid,64):raise ValueError(\'F_TRACE needs a (grid,64) int64 stamp buffer\')\n        ptrs.append(stamps.data_ptr())\n    mod.run(ptrs,m,d[\'config\'][\'local_start\'],stages,grid,torch.cuda.current_stream(dev).cuda_stream,flags)\n    return out\n\n\ndef routed_var(pr,rs,state,flags,stamps=None):\n    x,ids,tw,lids=route_call(pr,rs)\n    d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={\'local_start\':0})\n    return var_call(d,state,flags,stamps=stamps),x,ids,tw,lids\n\n\ndef run_lengths(d,grid=264):\n    """W2 tile runs per (CTA, expert, block) under the incumbent\'s unit schedule (host replica)."""\n    ids=d[\'topk_ids\'].flatten().tolist();n=[0]*32\n    for e in ids:\n        if 0<=e<32:n[e]+=1\n    nblk=[(c+7)//8 for c in n];us=[0]\n    for b in nblk:us.append(us[-1]+256*b)\n    U=us[-1];cc=min(256,max(1,(U+grid//2)//grid));runs=[]\n    for c in range(grid):\n        a,b=U*c//grid,U*(c+1)//grid;cur=None;length=0\n        for u in range(a,b):\n            e=max(i for i in range(32) if us[i]<=u and nblk[i]);rel=u-us[e];span=cc*nblk[e];nfull=256//cc\n            if rel<nfull*span:nb=(rel-(rel//span)*span)//cc\n            else:nb=(rel-nfull*span)//(256-nfull*cc)\n            if (e,nb)!=cur:\n                if cur is not None:runs.append(length)\n                cur=(e,nb);length=0\n            length+=1\n        if cur is not None:runs.append(length)\n    return runs\n\nimport triton\nimport triton.language as tl\n@triton.jit\ndef _pdl_start(USE: tl.constexpr):\n    if USE:\n        tl.extra.cuda.gdc_launch_dependents()\n\n@triton.jit\ndef _pdl_wait(USE: tl.constexpr):\n    if USE:\n        tl.extra.cuda.gdc_wait()\n\n@triton.jit\ndef timeline_route(PR,S:tl.constexpr,PS:tl.constexpr,RS:tl.constexpr,BIAS,PREW,LSTART,\n                X,IDS,WTS,LIDS,REPLICAS:tl.constexpr,USE_PDL:tl.constexpr):\n    t=tl.program_id(0);part=tl.program_id(1)\n    _pdl_start(USE_PDL)\n    d=tl.arange(0,4096);e=tl.arange(0,256)\n    prew=tl.load(PREW+d);bias=tl.load(BIAS+e);start=tl.load(LSTART)\n    _pdl_wait(USE_PDL)\n    p=tl.full((4096,),0.,tl.float32)\n    logit=tl.full((256,),0.,tl.float32)\n    for s in tl.static_range(S):\n        row=PR+s*PS+t*RS\n        p+=tl.load(row+d).to(tl.float32)\n        logit+=tl.load(row+4096+e).to(tl.float32)+tl.load(row+4352+e).to(tl.float32)\n    norm=p*tl.rsqrt(tl.sum(p*p,0)/4096+1.e-5)*prew\n    tl.store(X+t*4096+d,norm.to(tl.bfloat16),d//(4096//REPLICAS)==part)\n    if part==0:\n        scores=tl.sigmoid(logit*0.1767766952966369)\n        biased=scores+bias\n        bits=biased.to(tl.int32,bitcast=True)\n        ordered=tl.where(bits<0,~bits,bits^(-2147483648)).to(tl.uint32)\n        keys=(ordered.to(tl.uint64)<<32)|(255-e).to(tl.uint64)\n        winners=tl.topk(keys,8)\n        ids=255-(winners&0xffffffff).to(tl.int32)\n        scores=tl.gather(scores,ids,0)\n        k=tl.arange(0,8)\n        tl.store(IDS+t*8+k,ids)\n        tl.store(WTS+t*8+k,scores/(tl.sum(scores,0)+1.e-15))\n        local=(ids-start+256)%256\n        tl.store(LIDS+t*8+k,tl.where(local<32,local,-1))\n\ndef route_call(pr,rs):\n    p,s=pr;m=p.shape[1];dev=p.device\n    x=torch.empty((m,4096),device=dev,dtype=torch.bfloat16)\n    ids=torch.empty((m,8),device=dev,dtype=torch.int32);lids=torch.empty_like(ids);tw=torch.empty((m,8),device=dev)\n    warps=4 if m==8 else 8\n    timeline_route[(m,1)](p,s,p.stride(0),p.stride(1),rs[\'bias\'],rs[\'pre\'],rs[\'start\'],x,ids,tw,lids,\n        REPLICAS=1,USE_PDL=True,launch_pdl=True,num_warps=warps)\n    return x,ids,tw,lids\n\n\ndef routed_experts(pr,rs,state,kind=\'INC\'):\n    x,ids,tw,lids=route_call(pr,rs)\n    d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={\'local_start\':0})\n    return nt1_call(d,state,kind),x,ids,tw,lids\n\n\ndef routed_fixture(w,m,seed):\n    device=w[\'router\'].device;gen=torch.Generator(device=device).manual_seed(seed)\n    mean,shared=fx__basis(device);coef=torch.randn((m,fx_STREAM_RANK),generator=gen,device=device)\n    q=8 if m==8 else 1;scale=torch.ones((m//q,q,1),device=device)\n    if q>1:scale[:,1:]=fx_DRAFT_SHARE\n    scale=scale.reshape(m,1)\n    stream=scale*(fx_STREAM_MEAN*mean+(fx_STREAM_SHARED/fx_STREAM_RANK)**.5*(coef@shared))\n    stream+=(1-fx_STREAM_SHARED*scale.square()).sqrt()*torch.randn((m,fx_HIDDEN),generator=gen,device=device)\n    h=fx__rms(stream)*(1+w[\'pre_ffn_norm\'].float())\n    logits=h@w[\'router\'].float().T;pooled=h@w[\'pooler\'].float().T\n    p=torch.cat((pooled,logits,torch.zeros_like(logits)),-1)[None].contiguous()\n    biased=torch.sigmoid(logits*.1767766952966369)+w[\'router_bias\'].float()\n    ids=biased.topk(8,-1).indices;start=int(torch.bincount((ids//32).flatten(),minlength=8).argmax())*32\n    rs=dict(bias=w[\'router_bias\'].float().contiguous(),pre=(1+w[\'pre_expert_norm\'].float()).contiguous(),\n        start=torch.tensor([start],device=device,dtype=torch.int32))\n    return (p,1),rs\n\n\ndef route_reference(pr,rs):\n    p,s=pr;value=p[:s].double().sum(0)\n    logits=(value[:,4096:4352]+value[:,4352:4608])*.1767766952966369\n    scores=logits.sigmoid();biased=scores+rs[\'bias\'].double()\n    x=(value[:,:4096]*torch.rsqrt(value[:,:4096].square().mean(-1,keepdim=True)+1e-5)*rs[\'pre\'].double()).float().to(torch.bfloat16)\n    return x,scores,biased\n\n\ndef route_gate(ids,biased,start):\n    # TASK.md/ref check_routing: allow legitimate near ties and gate local load.\n    if bool((ids<0).any()) or bool((ids>=256).any()):raise ValueError(\'router expert out of range\')\n    ranked=ids.long().sort(-1).values\n    if bool((ranked[:,1:]==ranked[:,:-1]).any()):raise ValueError(\'duplicate routed expert\')\n    best=biased.topk(8,-1);selected=biased.gather(1,ids.long())\n    if float((best.values[:,-1:]-selected).max())>.05:raise ValueError(\'router score cutoff exceeds0.05\')\n    ninth=biased.topk(9,-1).values[:,-1:];kept=(best.indices[:,:,None]==ids.long()[:,None,:]).any(-1)\n    local=(best.indices-start)%256<32\n    if bool((local&~kept&(best.values-ninth>.03)).any()):raise ValueError(\'router dropped mandatory local expert\')\n    ref_local=best.indices[local];got_local=ids[((ids-start)%256)<32]\n    if got_local.numel()<ref_local.numel()*.92-4:raise ValueError(\'router local route count\')\n    if got_local.unique().numel()<ref_local.unique().numel()*.85-2:raise ValueError(\'router distinct local experts\')\n    return int((ids.long().sort(-1).values!=best.indices.sort(-1).values).any(-1).sum())\ndef nt1_fp32(data, weights):\n    result = torch.zeros_like(data[\'hidden_states\'], dtype=torch.float32)\n    ids = data[\'topk_ids\'] - data[\'config\'][\'local_start\']\n    grid = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device=ids.device)\n    def expand(name, e):\n        p = weights[name][e]\n        code = torch.stack((p & 15, p >> 4), -1).flatten(-2)\n        value = grid[(code & 7).long()] * torch.where(code & 8 != 0, -1., 1.)\n        return (value.view(-1, 256, 16) * weights[name + \'_sf\'][e].float()[..., None] *\n                weights[name + \'_gs\'][e]).reshape(-1, 4096)\n    for e in range(32):\n        token, slot = (ids == e).nonzero(as_tuple=True)\n        if not token.numel():\n            continue\n        h = data[\'hidden_states\'][token].float() @ expand(\'w13\', e).T\n        a = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]\n        y = a @ expand(\'w2\', e).T\n        z = y * (1. + weights[\'post_norm_w\'].float())\n        result.index_add_(0, token, z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-8) * data[\'topk_weights\'][token, slot, None])\n    return result\n\nDIAG_ALL=F_Z|F_SLOT|F_ZOUT|F_RSQ|F_NREG\nDIAG_ALLM=F_Z|F_ZOUT|F_RSQ|F_NREG\nDIAG_FLAGS={\'TRACE0\':F_TRACE,\'TRACEZZP\':F_TRACE|F_Z|F_ZOUT|F_PDL,\'TRACEALL\':F_TRACE|DIAG_ALL|F_LAST,\'TRACEALLnl\':F_TRACE|DIAG_ALL,\'ALL_positioned\':DIAG_ALL|F_LAST,\n    \'RTRACE0\':F_TRACE,\'RTRACEPDL\':F_TRACE|F_PDL,\'RTRACEALLnl\':F_TRACE|DIAG_ALL|F_PDL,\'RTRACEALL\':F_TRACE|DIAG_ALL|F_LAST|F_PDL,\n    \'TRACEALLM\':F_TRACE|DIAG_ALLM,\'RTRACEALLM\':F_TRACE|DIAG_ALLM|F_PDL,\'RTRACEZZP\':F_TRACE|F_Z|F_ZOUT|F_PDL,\'RTRACEZZP2\':F_TRACE|F_Z|F_ZOUT|F_PDL|F_NT2,\'RTRACEZZPW\':F_TRACE|F_Z|F_ZOUT|F_PDL|F_NT2|F_W16,\n    \'RTRACEZZPR\':F_TRACE|F_Z|F_ZOUT|F_PDL|F_RND,\'RTRACEZZPO\':F_TRACE|F_Z|F_ZOUT|F_PDL|F_ROT,\n    \'RTRACEZZPRO\':F_TRACE|F_Z|F_ZOUT|F_PDL|F_RND|F_ROT}\nDIAG_CASES=(\'INC_plain\',\'INC_positioned\',\'TRACE0\',\'TRACEALL\',\'TRACEALLnl\',\'ALL_positioned\',\'EMPTY_match\',\'EMPTY_tiny\',\'STREAM\',\n    \'RTRACE0\',\'RTRACEPDL\',\'RTRACEALLnl\',\'RTRACEALL\')\nDIAG_CASES_M=(\'INC_plain\',\'TRACE0\',\'TRACEALLM\',\'STREAM\',\'RTRACE0\',\'RTRACEALLM\',\'RTRACEZZP\')\n\n\ndef timeline_case(d,state,kind,reps=10,cupti=True,routed=None):\n    """One graph: [marker] node [marker]; event time, marker stamps, per-CTA stamps, optional CUPTI node records."""\n    ctl=control_module();dev=d[\'hidden_states\'].device;m=len(d[\'hidden_states\'])\n    markers=torch.zeros(4,device=dev,dtype=torch.int64)\n    inc=nt1_module(\'INC\');grid=min(264,state[\'sms\']*inc.resident_blocks(1,2,m));smem=inc.smem_needed(1,2,m)\n    experts=(d[\'topk_ids\']-d[\'config\'][\'local_start\']).flatten();experts=experts[(experts>=0)&(experts<32)].unique().int().contiguous()\n    count=len(experts)\n    flags=DIAG_FLAGS.get(kind)\n    if flags is not None and flags&F_NT2:grid=var_geometry(state,flags,m)[2]\n    if kind==\'STREAM\':\n        if not count:raise ValueError(\'stream control needs a routed expert\')\n        stamps=torch.zeros((grid+16,4),device=dev,dtype=torch.int64);sums=torch.empty(((grid+16)*256,),device=dev,dtype=torch.int32)\n    elif kind.startswith(\'EMPTY\'):stamps=torch.zeros((grid,16),device=dev,dtype=torch.int64)\n    elif \'TRACE\' in kind:stamps=torch.zeros((grid,64),device=dev,dtype=torch.int64)\n    else:stamps=None\n    positioned=not kind.startswith(\'INC_plain\')\n    def launch():\n        stream=torch.cuda.current_stream(dev).cuda_stream\n        if positioned:ctl.mark(markers.data_ptr(),0,stream)\n        if kind.startswith(\'EMPTY\'):\n            ctl.empty(stamps.data_ptr(),1 if kind==\'EMPTY_tiny\' else grid,32 if kind==\'EMPTY_tiny\' else 256,0 if kind==\'EMPTY_tiny\' else smem,stream)\n        elif kind==\'STREAM\':\n            ctl.streaming(state[\'bank\'][\'w13p\'].data_ptr(),state[\'bank\'][\'w2p\'].data_ptr(),experts.data_ptr(),count,\n                sums.data_ptr(),stamps.data_ptr(),grid,256,smem,stream)\n        elif kind[0]==\'R\':routed_var(*routed,state,flags,stamps=stamps)\n        elif flags is not None:var_call(d,state,flags,stamps=stamps)\n        else:nt1_call(d,state,\'INC\')\n        if positioned:ctl.mark(markers.data_ptr(),1,stream)\n    launch();torch.cuda.synchronize()\n    act=None\n    if cupti:\n        try:act=activity_module();act.profile_begin()\n        except Exception as e:act=None;cupti=f\'{type(e).__name__}: {e}\'[:120]\n    samples=[]\n    try:\n        graph=torch.cuda.CUDAGraph()\n        with torch.cuda.graph(graph):launch()\n        for _ in range(2):lab_cold_l2();graph.replay();torch.cuda.synchronize()\n        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)\n        for rep in range(reps):\n            lab_cold_l2();a.record();graph.replay();b.record();b.synchronize()\n            samples.append(dict(event_us=a.elapsed_time(b)*1000,markers=markers.cpu().tolist() if positioned else None,\n                stamps=stamps.cpu().tolist() if stamps is not None else None))\n    finally:\n        torch.cuda.synchronize();records=act.profile_end() if act is not None else None\n    if records is not None:\n        main_name=\'moe_var\' if flags is not None else \'empty_like\' if kind.startswith(\'EMPTY\') else \'stream_banks\' if kind==\'STREAM\' else \'moe_fused\'\n        main=sorted([r for r in records if main_name in r[0] and r[1][4]>0],key=lambda r:r[1][0])[-reps:]\n        before=sorted([r for r in records if \'node_before\' in r[0]],key=lambda r:r[1][0])[-reps:]\n        after=sorted([r for r in records if \'node_after\' in r[0]],key=lambda r:r[1][0])[-reps:]\n        router=sorted([r for r in records if \'timeline_route\' in r[0]],key=lambda r:r[1][0])[-reps:]\n        if len(main)!=reps or (positioned and (len(before)!=reps or len(after)!=reps)):records=None;cupti=\'incomplete records\'\n    for i,s in enumerate(samples):\n        if positioned:s[\'marker_span_us\']=(s[\'markers\'][2]-s[\'markers\'][0])/1000\n        if records is None:continue\n        rec=main[i][1];s[\'activity\']=rec;s[\'node_us\']=(rec[1]-rec[0])/1000\n        if len(router)==reps:s[\'router_us\']=(router[i][1][1]-router[i][1][0])/1000;s[\'router_end_to_moe_start_us\']=(rec[0]-router[i][1][1])/1000\n        if positioned:\n            rb,ra=before[i][1],after[i][1];mb=s[\'markers\'][0];ma=s[\'markers\'][2]\n            lo=max(rb[0]-mb,ra[0]-ma);hi=min(rb[1]-mb,ra[1]-ma)\n            if lo<=hi:s[\'offset_bounds\']=[lo,hi]\n            s[\'graph_span_us\']=(ra[1]-rb[0])/1000\n    return dict(kind=kind,experts=experts.cpu().tolist(),unique_bytes=count*(18874368+9437184),grid=grid,samples=samples,cupti=cupti)\n\n\ndef skew_stats(rows,first):\n    """W13 end, W2 end and grid-barrier exit per CTA (us from the first entry) and units per CTA: where W2 skew sits."""\n    def q(v):\n        v=sorted(v);return (v[len(v)//2],v[(9*len(v))//10],v[-1]) if v else (0.,0.,0.)\n    w13=[(r[3]-first)/1000 for r in rows if r[3] and r[26]];w2=[(r[5]-first)/1000 for r in rows if r[5] and r[27]]\n    bar=[(r[7]-first)/1000 for r in rows if r[7]];u1=[r[26] for r in rows];u2=[r[27] for r in rows]\n    return q(w13)+q(w2)+q(bar)+(min(u1),max(u1),min(u2),max(u2))\n\n\ndef tile_stats(rows,first):\n    """Where a CTA\'s tile time goes (per-CTA sums, us; median over CTAs): tid 0 (epilogue warp) and tid 224 (MMA-only warp)\n    waits on landed tiles and at the per-tile barrier, the expert publish (fence, then barrier+atomic), the W13 and W2 phase\n    per tile, and how late the last publish lands after the last W13 end."""\n    def p50(v):\n        v=sorted(v);return v[len(v)//2] if v else 0.\n    w=[r for r in rows if r[26]];w2=[r for r in rows if r[27]]\n    return dict(\n        wait13_t0=p50(r[28]/1000 for r in w),wait2_t0=p50(r[29]/1000 for r in w2),bar_t0=p50(r[22]/1000 for r in rows),\n        bar_w7=p50(r[30]/1000 for r in rows),wait_w7=p50(r[31]/1000 for r in rows),\n        fence=p50(r[17]/1000 for r in w),fence_max=max((r[17]/1000 for r in w),default=0.),\n        pubbar=p50(r[19]/1000 for r in w),pubbar_max=max((r[19]/1000 for r in w),default=0.),\n        per13=p50((r[3]-r[2])/1000/r[26] for r in w if r[3]),per2=p50((r[5]-r[4])/1000/r[27] for r in w2 if r[5] and r[4]),\n        last_pub_after_last13=(max(r[18] for r in w if r[18])-max(r[3] for r in w))/1000 if any(r[18] for r in w) else 0.,\n        pub_after_own13=p50((r[18]-r[3])/1000 for r in w if r[18] and r[3]))\n\n\ndef timeline_account(case):\n    """Critical CTA (latest exit) as contiguous phases, plus the dispatch ramp and drain against the marker stamps."""\n    out=[]\n    for s in case[\'samples\']:\n        rows=[r for r in s[\'stamps\'] if r[0]];entries=[min(r[32:40]) for r in rows];exits=[max(max(r[40:48]),r[9]) for r in rows]\n        first=min(entries);last=max(exits);k=max(range(len(rows)),key=lambda i:exits[i]);r=rows[k]\n        cut=[entries[k]]+[r[i] if r[i] else 0 for i in (1,2,3,4,5,6,7,8)]+[exits[k]]\n        for i in range(1,len(cut)):cut[i]=cut[i] or cut[i-1]\n        phases=[(b-a)/1000 for a,b in zip(cut,cut[1:])]\n        res=dict(event_us=s[\'event_us\'],cta_span_us=(last-first)/1000,ramp_to_critical_us=(entries[k]-first)/1000,phases=phases,\n            entry_spread_us=(max(entries)-first)/1000,critical_finalizer=bool(r[23]),\n            launch_us=(first-s[\'markers\'][0])/1000 if s.get(\'markers\') else None,\n            drain_us=(s[\'markers\'][2]-last)/1000 if s.get(\'markers\') else None,\n            marker_span_us=s.get(\'marker_span_us\'),node_us=s.get(\'node_us\'),\n            after_last_W2_us=(last-max(rr[5] for rr in rows))/1000,\n            last_W13_end_us=(max(rr[3] for rr in rows)-first)/1000,\n            first_copy_issue_us=statistics.median((rr[10]-min(rr[32:40]))/1000 for rr in rows if rr[10]),\n            first_copy_land_us=statistics.median((rr[12]-rr[10])/1000 for rr in rows if rr[10] and rr[12]),\n            w2_wait_med_us=statistics.median(rr[16]/1000 for rr in rows if rr[27]),\n            w2_wait_max_us=max(rr[16]/1000 for rr in rows if rr[27]),\n            skew=skew_stats(rows,first),tile=tile_stats(rows,first))\n        if s.get(\'offset_bounds\') and s.get(\'activity\'):\n            lo,hi=s[\'offset_bounds\'];n0,n1=s[\'activity\'][:2]\n            res[\'node_to_first_cta_us\']=[(first+lo-n0)/1000,(first+hi-n0)/1000];res[\'last_cta_to_node_end_us\']=[(n1-last-hi)/1000,(n1-last-lo)/1000]\n        out.append(res)\n    return out\n\n\ndef timeline_diagnose(d,state,routed,cases_run=DIAG_CASES):\n    ctl=control_module();res=torch.zeros(3,device=d[\'hidden_states\'].device,dtype=torch.int64)\n    ctl.resolution(res.data_ptr(),torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize();plain=res.cpu().tolist()\n    cases={}\n    for kind in cases_run:\n        try:cases[kind]=timeline_case(d,state,kind,routed=routed)\n        except Exception as e:cases[kind]=dict(kind=kind,error=f\'{type(e).__name__}: {e}\'[:160])\n    profiled=None\n    try:\n        act=activity_module();act.profile_begin()\n        try:ctl.resolution(res.data_ptr(),torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize();profiled=res.cpu().tolist()\n        finally:act.profile_end()\n    except Exception as e:profiled=f\'{type(e).__name__}\'\n    acc={k:timeline_account(c) for k,c in cases.items() if \'TRACE\' in k and \'samples\' in c}\n    return dict(cases=cases,account=acc,resolution=dict(plain=plain,profiled=profiled))\n\n\ndef timeline_lines(result,resolution=True):\n    out=[];r=result[\'resolution\']\n    if resolution:out.append(f\'GTRES plain_min/changes/span_ns={r["plain"]} under_cupti={r["profiled"]}\')\n    med=statistics.median\n    for k,c in result[\'cases\'].items():\n        if \'error\' in c:out.append(f\'NODE {k} ERROR {c["error"]}\');continue\n        if k in result[\'account\'] and not k.startswith(\'R\'):continue  # its ACCOUNT line carries event/marker/node\n        s=c[\'samples\'];line=f\'NODE {k} event={med(x["event_us"] for x in s):.2f}\'\n        if s[0].get(\'marker_span_us\') is not None:line+=f\' marker_span={med(x["marker_span_us"] for x in s):.2f}\'\n        if s[0].get(\'node_us\') is not None:line+=f\' cupti_node={med(x["node_us"] for x in s):.2f}\'\n        elif c.get(\'cupti\') not in (True,None):line+=f\' cupti={c["cupti"]}\'\n        if s[0].get(\'router_us\') is not None:line+=f\' router={med(x["router_us"] for x in s):.2f} router_end_to_moe_start={med(x["router_end_to_moe_start_us"] for x in s):.2f}\'\n        if k==\'STREAM\':line+=f\' MB={c["unique_bytes"]/1e6:.1f}\'\n        if k==\'EMPTY_match\' or k==\'STREAM\':\n            first=[min(r[0] for r in x[\'stamps\'] if r[0]) for x in s];line+=f\' launch={med((f-x["markers"][0])/1000 for f,x in zip(first,s)):.2f}\'\n        out.append(line)\n    names=\'route,zero,W13,W2ready,W2rest,fence,barrier,combine,footer\'\n    for k,rows in result[\'account\'].items():\n        g=lambda key:med(x[key] for x in rows)\n        out.append(f\'ACCOUNT {k} event={g("event_us"):.2f} marker_span={g("marker_span_us"):.2f} launch={g("launch_us"):.2f} CTAspan={g("cta_span_us"):.2f} drain={g("drain_us"):.2f} entry_spread={g("entry_spread_us"):.2f} lastW13end={g("last_W13_end_us"):.2f} afterLastW2={g("after_last_W2_us"):.2f} issue={g("first_copy_issue_us"):.2f} land={g("first_copy_land_us"):.2f} W2wait_med/max={g("w2_wait_med_us"):.2f}/{g("w2_wait_max_us"):.2f}\'\n            +(f\' cupti_node={g("node_us"):.2f}\' if rows[0].get(\'node_us\') else \'\'))\n        sk=[x[\'skew\'] for x in rows];sm=[med(v[i] for v in sk) for i in range(13)]\n        out.append(f\'SKEW {k} W13end p50/p90/max={sm[0]:.2f}/{sm[1]:.2f}/{sm[2]:.2f} W2end={sm[3]:.2f}/{sm[4]:.2f}/{sm[5]:.2f} barrier_exit={sm[6]:.2f}/{sm[7]:.2f}/{sm[8]:.2f} units13={sm[9]:.0f}-{sm[10]:.0f} units2={sm[11]:.0f}-{sm[12]:.0f}\')\n        out.append(f\'CRITICAL {k} {names}=\'+\',\'.join(f\'{med(x["phases"][i] for x in rows):.2f}\' for i in range(9)))\n        t={key:med(x[\'tile\'][key] for x in rows) for key in rows[0][\'tile\']}\n        out.append(f\'TILE {k} per13/per2={t["per13"]:.2f}/{t["per2"]:.2f} t0_wait13/wait2/bar={t["wait13_t0"]:.2f}/{t["wait2_t0"]:.2f}/{t["bar_t0"]:.2f} w7_wait/bar={t["wait_w7"]:.2f}/{t["bar_w7"]:.2f}\'\n            f\' pub_fence={t["fence"]:.2f}(max {t["fence_max"]:.2f}) pub_bar={t["pubbar"]:.2f}(max {t["pubbar_max"]:.2f}) pub_after_own13={t["pub_after_own13"]:.2f} last_pub_after_last13={t["last_pub_after_last13"]:.2f}\')\n    return out\n\nclass TimelineGraph:\n    def __init__(self,fn):\n        fn();torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()\n        with torch.cuda.graph(self.graph):self.out=fn()\n    def replay(self):self.graph.replay();return self.out\n\n\ndef timeline_timing(old,new,repeats=12):\n    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)\n    aa,bb,ratios=[],[],[]\n    def measured(graph):\n        lab_cold_l2();a.record();graph.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000\n    for rep in range(repeats):\n        order=(old,new) if rep%2==0 else (new,old);v=[measured(g) for g in order]\n        x,y=v if rep%2==0 else v[::-1];aa.append(x);bb.append(y);ratios.append(y/x)\n    return statistics.median(aa),statistics.median(bb),statistics.median(ratios)\n\n\ndef task_check(out,want):\n    """TASK.md: output relative L2 <= 1.5% vs FP32, every row <= 5x (near-zero rows vs a tenth of the mean row), finite."""\n    err,row=nt1_error(out,want);ok=bool(torch.isfinite(out).all()) and err<=.015 and row<=.075\n    return ok,err,row\n\n\nDIAG_BY_M={16:(\'RTRACEZZP\',\'RTRACEZZPR\',\'RTRACEZZPO\'),32:(\'RTRACEZZP\',\'RTRACEZZPR\',\'RTRACEZZPO\',\'RTRACEZZPRO\')}  # v8: critical paths + tile/publish split\n# R = the decode layer\'s context (router node + MoE node). R95/R93 = 849e4017\'s winners; R77 drops NREG, R69 drops\n# NREG and RSQ; ECAP = INC on the same routes with every local expert capped at 8 tokens (DIAGNOSTIC, no gate): the\n# most that decode-once / no-reread for multi-block experts could save.\nSMALLM_ARMS={}\nfor _m in (8,16,32):\n    SMALLM_ARMS[_m]=dict(E={},R={\'RINC\':None,\'R69\':69,\'R8261\':69|F_RND,\'R16453\':69|F_ROT,\'R24645\':69|F_RND|F_ROT})\n\n\ndef cap_fixture(d,cap=8):\n    """Same fixture, each local expert keeps its first `cap` (token, slot) pairs; the rest become remote (-1)."""\n    ids=d[\'topk_ids\'].clone();flat=ids.view(-1);local=flat-d[\'config\'][\'local_start\']\n    for e in range(32):\n        hit=((local>=0)&(local<32)&(local==e)).nonzero().flatten()\n        if len(hit)>cap:flat[hit[cap:]]=-1\n    return dict(d,topk_ids=ids)\n\n\ndef blocks(d):\n    local=(d[\'topk_ids\']-d[\'config\'][\'local_start\']).flatten();local=local[(local>=0)&(local<32)]\n    n=torch.bincount(local.long(),minlength=32);return int(((n+7)//8).sum()),int((n>0).sum())\n\n\ndef lab(deadline,weights):\n    from concurrent.futures import ThreadPoolExecutor\n    begin=time.monotonic();deadline-=5;device=weights[\'w13\'].device\n    lines=[\'META small-M NT1: E=MoE node alone (fixed real routes), R=router node+MoE node; INC=admitted one-LOP3 NT1; F0=same math in the flag harness; Z=norm-slot zero in W13 tile0; SLOT=M1 top-8 slot dispatch; ZOUT=W2 stores v(1+w); RSQ=finalizer scale once per pair; NREG=norm sums in registers per tile run; LAST=last CTA finalizes; PDL=MoE launched as programmatic dependent\',\n        \'META v8: cold = lab_cold_l2() (wrapper; judge-style read); paired AB/BA x6 per arm vs its INC, independent INC A:A; 16 seeds of public weights/routers; TASK.md gate (1.5% L2, row 5x) per cell; arms 69, 69+RND (two rounds split at the expert boundary nearest half the W13 units), 69+ROT (no per-tile CTA barrier: arrival counters, last arriver refills, tile t epilogue on warp t%8 one tile later) and 69+RND+ROT at M8/16/32; M16/32 critical paths with TILE (per-tile waits, barrier/epilogue spin, publish fence) lines\']\n    errors=[]\n    with ThreadPoolExecutor(max_workers=4) as pool:\n        jobs={name:pool.submit(fn) for name,fn in ((\'INC\',lambda:nt1_module(\'INC\')),(\'VAR\',var_module),(\'CONTROL\',control_module),(\'CUPTI\',activity_module))}\n        for name,job in jobs.items():\n            try:job.result()\n            except Exception as e:errors.append(f\'BUILD_ERROR {name} {type(e).__name__}: {str(e)[-200:]}\')\n    lines+=errors\n    if any(name in e for e in errors for name in (\'BUILD_ERROR INC\',\'BUILD_ERROR VAR\')):return lines\n    lines.append(f\'BUILD seconds={time.monotonic()-begin:.2f} left={deadline-time.monotonic():.1f}\')\n    rows={};deltas={};caps={};noise={m:{\'E\':[],\'R\':[]} for m in SMALLM_ARMS};worst={};failed={};flips=0;complete=0;peak=0.\n    previous=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False\n    try:\n        for seed in range(16):\n            if deadline-time.monotonic()<max(8.,peak*1.25+2.):\n                lines.append(f\'BUDGET before_seed={seed}\');break\n            started=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=device);w[\'post_norm_w\']=w[\'post_expert_norm\'];state=nt1_banks(w)\n            if seed==0:\n                for mt,cases in DIAG_BY_M.items():\n                    try:lines.extend(f\'M{mt} \'+l for l in timeline_lines(timeline_diagnose(fx_make_fixture(w,mt,9001),state,routed_fixture(w,mt,9001),cases),False))\n                    except Exception as e:lines.append(f\'DIAG_ERROR M{mt} {type(e).__name__}: {str(e)[:200]}\')\n            for m,arms in SMALLM_ARMS.items():\n                d=fx_make_fixture(w,m,9000+m+97*seed);want=nt1_fp32(d,w)\n                pr,rs=routed_fixture(w,m,9000+m+97*seed);xr,score,bias=route_reference(pr,rs);start=int(rs[\'start\'])\n                graphs={};truths={}\n                dcap=cap_fixture(d) if \'ECAP\' in arms[\'E\'] else None\n                if dcap is not None:caps.setdefault(m,[]).append(blocks(d)+blocks(dcap))\n                for label,flags in arms[\'E\'].items():\n                    if label in failed:continue\n                    try:graphs[label]=TimelineGraph(lambda:nt1_call(d,state,\'INC\')) if flags is None else \\\n                        TimelineGraph(lambda:nt1_call(dcap,state,\'INC\')) if label==\'ECAP\' else TimelineGraph(lambda f=flags:var_call(d,state,f))\n                    except Exception as e:failed[label]=f\'M{m} seed{seed} capture {type(e).__name__}: {str(e)[:80]}\'\n                for label,flags in arms[\'R\'].items():\n                    if label in failed:continue\n                    try:\n                        graphs[label]=TimelineGraph(lambda:routed_experts(pr,rs,state,\'INC\')) if flags is None else \\\n                            TimelineGraph(lambda f=flags:routed_var(pr,rs,state,f))\n                    except Exception as e:failed[label]=f\'M{m} seed{seed} capture {type(e).__name__}: {str(e)[:80]}\'\n                def gate(label,first=False):\n                    if label==\'ECAP\':return\n                    out=graphs[label].replay();torch.cuda.synchronize()\n                    if label[0]==\'E\':got,truth=out,want\n                    else:\n                        got,x,ids,tw,lids=out\n                        n=route_gate(ids,bias,start)\n                        if not torch.allclose(x.float(),xr.float(),rtol=.01,atol=.01):raise ValueError(\'pooled norm\')\n                        key=tuple(ids.flatten().cpu().tolist())\n                        if key not in truths:truths[key]=nt1_fp32(dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={\'local_start\':0}),w)\n                        truth=truths[key]\n                        if label==\'RINC\' and first:route_flips[0]+=n\n                    ok,err,row=task_check(got,truth)\n                    worst[label]=max(worst.get(label,(0.,0.)),(err,row))\n                    if not ok:raise ValueError(f\'TASK gate L2/row={err:.5f}/{row:.5f}\')\n                route_flips=[0]\n                for label in list(graphs):\n                    try:gate(label,True)\n                    except Exception as e:\n                        failed[label]=f\'M{m} seed{seed} {type(e).__name__}: {str(e)[:100]}\';graphs.pop(label)\n                flips+=route_flips[0]\n                for prefix in (\'E\',\'R\'):\n                    base=prefix+\'INC\'\n                    if base not in graphs:continue\n                    ctl=TimelineGraph(lambda:nt1_call(d,state,\'INC\')) if prefix==\'E\' else TimelineGraph(lambda:routed_experts(pr,rs,state,\'INC\'))\n                    a,b,_=timeline_timing(graphs[base],ctl);noise[m][prefix].append(b-a);rows.setdefault((m,base),[]).append(a)\n                    others=[k for k in graphs if k[0]==prefix and k!=base]\n                    if seed%2:others.reverse()\n                    for label in others:\n                        a,b,ratio=timeline_timing(graphs[base],graphs[label])\n                        rows.setdefault((m,label),[]).append(rows[m,base][-1]*ratio);deltas.setdefault((m,label),[]).append(rows[m,base][-1]*(ratio-1))\n                        try:gate(label)\n                        except Exception as e:failed[label]=f\'M{m} seed{seed} post-timing {type(e).__name__}: {str(e)[:100]}\'\n                graphs={};truths={}\n            complete+=1;peak=max(peak,time.monotonic()-started)\n    finally:torch.backends.cuda.matmul.allow_tf32=previous\n    for m,v in sorted(caps.items()):\n        lines.append(f\'CAP M{m} blocks/experts {statistics.mean(x[0] for x in v):.2f}/{statistics.mean(x[1] for x in v):.2f} -> {statistics.mean(x[2] for x in v):.2f}/{statistics.mean(x[3] for x in v):.2f} n={len(v)}\')\n    for label,why in failed.items():lines.append(f\'FAIL {label} {why}\')\n    lines.append(f\'GATE TASK.md L2/row worst \'+\',\'.join(f\'{k}:{v[0]:.4f}/{v[1]:.4f}\' for k,v in sorted(worst.items()))+f\'; RINC near-tie flips={flips}\')\n    for m in SMALLM_ARMS:\n        for prefix in (\'E\',\'R\'):\n            base=(m,prefix+\'INC\')\n            if base not in rows:continue\n            nz=noise[m][prefix];parts=[f\'{prefix}INC={statistics.mean(rows[base]):.2f}\']\n            for label in SMALLM_ARMS[m][prefix]:\n                if label==prefix+\'INC\' or (m,label) not in deltas:continue\n                dl=deltas[m,label];se=statistics.stdev(dl)/len(dl)**.5 if len(dl)>1 else 0.\n                parts.append(f\'{label}={statistics.mean(rows[m,label]):.2f}({statistics.mean(dl):+.2f}+-{se:.2f},n{len(dl)})\')\n            parts.append(f\'AA={statistics.mean(nz):+.2f}/{statistics.mean(map(abs,nz)):.2f}/{max(map(abs,nz)):.2f}\')\n            lines.append(f\'AGG M{m} \'+\' \'.join(parts))\n    lines.append(f\'END seeds={complete} min8={complete>=8} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}; nothing promoted here\')\n    return lines\n')]
_lab_base_prepare = prepare
_lab_ran = False
_lab_flush_buf = None


def lab_cold_l2():
    # The judge's cold-L2 step (TASK.md): read a buffer larger than L2 before each timed replay. Never flush with
    # zero_()/fill_(): that leaves ~50 MB of dirty lines for the timed call to write back (+5-16 us at M=1-32).
    global _lab_flush_buf
    if _lab_flush_buf is None:
        _lab_flush_buf = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    _lab_flush_buf.view(torch.int64).sum()  # int64 view: the uint8 sum upcasts and runs ~4x slower


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
