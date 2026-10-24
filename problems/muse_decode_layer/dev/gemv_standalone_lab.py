"""Standalone decode GEMV bandwidth probe for tools/h200_lab.py."""
import ctypes as _sg_ct
import hashlib as _sg_hashlib
import os as _sg_os
import statistics as _sg_statistics
import time as _sg_time

import torch
from torch.utils.cpp_extension import load_inline as _sg_load_inline


_SG_LT_CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cstring>
namespace sglt {
static cublasLtHandle_t handle(){static cublasLtHandle_t h=[](){cublasLtHandle_t p;TORCH_CHECK(cublasLtCreate(&p)==CUBLAS_STATUS_SUCCESS,"lt create");return p;}();return h;}
struct Layout {
 cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a,b,c;
 Layout(torch::Tensor w,torch::Tensor x,torch::Tensor y,bool fp32){
  TORCH_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"lt desc");
  cublasOperation_t ta=CUBLAS_OP_T;TORCH_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))==CUBLAS_STATUS_SUCCESS,"lt trans");
  auto in=CUDA_R_16BF;auto out=fp32?CUDA_R_32F:CUDA_R_16BF;
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&a,in,w.size(1),w.size(0),w.stride(0))==CUBLAS_STATUS_SUCCESS,"lt A");
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&b,in,x.size(1),x.size(0),x.stride(0))==CUBLAS_STATUS_SUCCESS,"lt B");
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&c,out,y.size(1),y.size(0),y.stride(0))==CUBLAS_STATUS_SUCCESS,"lt C");
 }
 ~Layout(){cublasLtMatmulDescDestroy(op);cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(c);}
};
torch::Tensor options(torch::Tensor w,torch::Tensor x,torch::Tensor y,bool fp32){
 Layout l(w,x,y,fp32);cublasLtMatmulPreference_t p;TORCH_CHECK(cublasLtMatmulPreferenceCreate(&p)==CUBLAS_STATUS_SUCCESS,"lt pref");size_t bytes=64ull<<20;
 TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(p,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes))==CUBLAS_STATUS_SUCCESS,"lt ws");
 cublasLtMatmulHeuristicResult_t h[32];int count=0;auto s=cublasLtMatmulAlgoGetHeuristic(handle(),l.op,l.a,l.b,l.c,l.c,p,32,h,&count);cublasLtMatmulPreferenceDestroy(p);TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"lt heuristics");
 auto out=torch::empty({count,(int64_t)sizeof(cublasLtMatmulAlgo_t)},torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
 for(int i=0;i<count;++i)TORCH_CHECK(h[i].state==CUBLAS_STATUS_SUCCESS,"lt heuristic state");
 for(int i=0;i<count;++i)std::memcpy(out.data_ptr<uint8_t>()+i*sizeof(cublasLtMatmulAlgo_t),&h[i].algo,sizeof(cublasLtMatmulAlgo_t));return out;
}
void run(torch::Tensor w,torch::Tensor x,torch::Tensor y,torch::Tensor work,torch::Tensor codes,int64_t index,bool fp32){
 TORCH_CHECK(index>=0&&index<codes.size(0),"lt index");Layout l(w,x,y,fp32);cublasLtMatmulAlgo_t a;std::memcpy(&a,codes.data_ptr<uint8_t>()+index*sizeof(a),sizeof(a));float al=1.f,be=0.f;
 auto s=cublasLtMatmul(handle(),l.op,&al,w.data_ptr(),l.a,x.data_ptr(),l.b,&be,y.data_ptr(),l.c,y.data_ptr(),l.c,&a,work.data_ptr(),work.nbytes(),at::cuda::getCurrentCUDAStream());TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"lt run");
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("options",&sglt::options);m.def("run",&sglt::run);}
'''
_SG_LT = None


def _sg_lt():
    global _SG_LT
    if _SG_LT is not None:
        return _SG_LT
    target = "9.0a" if torch.cuda.get_device_capability()[0] == 9 else "12.0"
    old = _sg_os.environ.get("TORCH_CUDA_ARCH_LIST")
    _sg_os.environ["TORCH_CUDA_ARCH_LIST"] = target
    try:
        tag = _sg_hashlib.sha1(_SG_LT_CUDA.encode()).hexdigest()[:12]
        _SG_LT = _sg_load_inline("muse_decode_standalone_lt_" + tag, cpp_sources="", cuda_sources=_SG_LT_CUDA,
                                 functions=None, extra_cuda_cflags=["-O3", "-std=c++20"],
                                 extra_ldflags=["-lcublasLt"], verbose=False)
    finally:
        if old is None:
            _sg_os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            _sg_os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _SG_LT


def _sg_ref(x, w, fp32):
    return torch.mm(x, w.t(), out_dtype=torch.float32) if fp32 else torch.nn.functional.linear(x, w)


def _sg_materialize(value):
    partial, splits = value
    return partial.sum(0) if splits > 1 else partial[0]


def _sg_custom(x, w, fp32, bn, ks, bk):
    n, k = w.shape
    s = k // ks
    p = torch.empty((s, x.shape[0], n), device=x.device, dtype=torch.float32)
    tb = max(16, 1 << (x.shape[0] - 1).bit_length())
    _gemv_kernel[(n // bn, s)](x, w, p, x.shape[0], x.stride(0), N=n, K=k, KS=ks, BN=bn, BK=bk, TB=tb,
                               PREFETCH=False, USE_PDL=False, num_warps=4, num_stages=4)
    out = p.sum(0)
    return out if fp32 else out.to(torch.bfloat16)


def _sg_graph(fn):
    for _ in range(2):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return g, out


def _sg_bench(g, flush, repeats=20):
    vals = []
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        flush.add_(1); a.record(); g.replay(); b.record(); b.synchronize(); vals.append(a.elapsed_time(b) * 1000.0)
    return _sg_statistics.median(vals)


def _sg_cta(n, k, bn, ks):
    cta = (n // bn) * (k // ks)
    return cta, (cta + 131) // 132


def lab(deadline, weights, state):
    del weights
    device = state["qkv_gate"].device
    flush = torch.empty(256 << 20, device=device, dtype=torch.uint8)
    lt = _sg_lt()
    workspace = torch.empty(64 << 20, device=device, dtype=torch.uint8)
    roles = (
        ("qkv", state["qkv_gate"], False, (64, 1024, 128), (32, 512, 128)),
        ("o", state["o"], False, (64, 512, 128), (32, 256, 128)),
        ("pool_router", state["pool_router"], True, (64, 2048, 128), (32, 1024, 128)),
        ("unpool_slice", state["unpooler_slice"], False, (64, 512, 128), (32, 256, 128)),
        ("unpool_full", state["unpooler"], False, (64, 2048, 128), (32, 512, 128)),
    )
    lines = ["SG_META sms=132 L2_MB=50 cold_L2=256MiB graph_replays=20"]
    for name, w, fp32, default, more in roles:
        for m in (1, 8, 16):
            if _sg_time.monotonic() >= deadline - 8:
                lines.append("SG_BUDGET")
                return lines
            if name == "unpool_slice" and m > 8:
                continue
            if name == "unpool_full" and m <= 8:
                continue
            x = torch.randn((m, w.shape[1]), device=device, dtype=torch.bfloat16)
            banks = [torch.randn_like(x) for _ in range(3)]
            # Keep one fixed graph input, replacing it between cold replays.
            codes = lt.options(w, x, torch.empty((m, w.shape[0]), device=device, dtype=torch.float32 if fp32 else torch.bfloat16), 1 if fp32 else 0)
            variants = []
            variants.append(("cublas", lambda: _sg_ref(x, w, fp32), ("library", "library")))
            old_choice = _GEMV_CHOICES.get((m, w.shape[0], w.shape[1]))
            _GEMV_CHOICES[(m, w.shape[0], w.shape[1])] = None
            variants.append(("gemv", lambda: _sg_materialize(gemv(x, w, out_dtype=torch.float32 if fp32 else torch.bfloat16)), _sg_cta(w.shape[0], w.shape[1], default[0], default[1])))
            if old_choice is None:
                _GEMV_CHOICES.pop((m, w.shape[0], w.shape[1]), None)
            else:
                _GEMV_CHOICES[(m, w.shape[0], w.shape[1])] = old_choice
            variants.append(("more_cta", lambda: _sg_custom(x, w, fp32, more[0], more[1], more[2]), _sg_cta(w.shape[0], w.shape[1], more[0], more[1])))
            for idx in range(min(4, int(codes.shape[0]))):
                out = torch.empty((m, w.shape[0]), device=device, dtype=torch.float32 if fp32 else torch.bfloat16)
                variants.append((f"lt{idx}", lambda idx=idx, out=out: (lt.run(w, x, out, workspace, codes, idx, bool(fp32)) or out), ("library", "library")))
            results = []
            reference = _sg_ref(x, w, fp32)
            for tag, fn, meta in variants:
                try:
                    g, out = _sg_graph(fn)
                    rel = float((out.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-8))
                    if rel > 1e-2 or not torch.isfinite(out).all():
                        raise RuntimeError(f"relative output mismatch rel={rel:.3e}")
                    us = _sg_bench(g, flush)
                    cta, waves = meta if isinstance(meta[0], int) else (meta[0], meta[1])
                    bytes_read = w.numel() * 2
                    rate = bytes_read / us / 1e6
                    results.append((tag, us, rate, cta, waves))
                    g.reset()
                except Exception as exc:
                    lines.append(f"SG_REJECT role={name} m={m} variant={tag} {type(exc).__name__}: {str(exc)[:100]}")
            if results:
                details = " ".join(f"{tag}:{us:.2f}us/{rate:.2f}TB/s/cta={cta}/waves={waves}" for tag, us, rate, cta, waves in results)
                lines.append(f"SG_SHAPE role={name} m={m} bytes={w.numel()*2/1e6:.2f} {details}")
    lines.append("SG_DONE")
    return lines
