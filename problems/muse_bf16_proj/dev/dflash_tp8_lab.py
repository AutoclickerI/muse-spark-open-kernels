"""H200 TP8 DFlash dense portfolio probe, appended by tools/h200_lab.py.

The proj host supplies its already-built cuBLASLt wrapper in ``state['mod']``.
This payload keeps all drafter weights synthetic because the proj host owns only
the projection bank; dimensions and transposes are the serving model's real
per-rank shapes.
"""
import statistics as _df_statistics
import time as _df_time

import torch


_DFLASH_ROLES = (
    # name, N, K, number of draft rows per request, stock uses X @ W
    ("qkv", 1280, 8192, 8, False),
    ("o", 8192, 1024, 8, False),
    ("gate_up", 7168, 8192, 8, False),
    ("down", 8192, 3584, 8, False),
    ("conv_coeff", 2048, 8192, 8, False),
    ("context_fc", 1024, 40960, 8, False),
    ("context_kv", 1280, 8192, 8, True),
    # DFlash emits seven predictions per draft request; retain this real shape.
    ("draft_head", 25256, 8192, 7, False),
)


_DF_PG_RAW = r'''
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstring>
static thread_local cublasLtHandle_t pg_handle=nullptr;
static int pg_init(){return pg_handle ? 0 : (int)cublasLtCreate(&pg_handle);}
extern "C" int pg_algo_bytes(){return sizeof(cublasLtMatmulAlgo_t);}
struct PgLayout {
  cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  int init(int n,int m,int k,int stride,int nn){
    if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)!=CUBLAS_STATUS_SUCCESS)return 1;
    cublasOperation_t ta=nn ? CUBLAS_OP_N : CUBLAS_OP_T;
    if(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))!=CUBLAS_STATUS_SUCCESS)return 2;
    if(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,nn?n:k,nn?k:n,stride)!=CUBLAS_STATUS_SUCCESS)return 3;
    if(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,k)!=CUBLAS_STATUS_SUCCESS)return 4;
    if(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n)!=CUBLAS_STATUS_SUCCESS)return 5;
    return 0;
  }
  ~PgLayout(){if(op)cublasLtMatmulDescDestroy(op);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
};
extern "C" int pg_options(int n,int m,int k,int stride,int nn,size_t bytes,void* codes,int cap,int* count){
  int e=pg_init();if(e)return e;PgLayout l;if((e=l.init(n,m,k,stride,nn)))return e;
  cublasLtMatmulPreference_t p=nullptr;if(cublasLtMatmulPreferenceCreate(&p)!=CUBLAS_STATUS_SUCCESS)return 6;
  if(cublasLtMatmulPreferenceSetAttribute(p,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes))!=CUBLAS_STATUS_SUCCESS)return 7;
  cublasLtMatmulHeuristicResult_t h[64];int got=0;
  auto s=cublasLtMatmulAlgoGetHeuristic(pg_handle,l.op,l.a,l.b,l.c,l.c,p,64,h,&got);
  cublasLtMatmulPreferenceDestroy(p);if(s!=CUBLAS_STATUS_SUCCESS)return (int)s;
  int valid=0;for(int i=0;i<got && valid<cap;++i)if(h[i].state==CUBLAS_STATUS_SUCCESS && h[i].workspaceSize<=bytes){std::memcpy((char*)codes+valid*sizeof(cublasLtMatmulAlgo_t),&h[i].algo,sizeof(cublasLtMatmulAlgo_t));++valid;}
  *count=valid;return 0;
}
extern "C" int pg_run(const void* w,const void* x,void* out,void* work,const void* codes,int n,int m,int k,int stride,int nn,int index,size_t bytes,void* stream){
  int e=pg_init();if(e)return e;PgLayout l;if((e=l.init(n,m,k,stride,nn)))return e;cublasLtMatmulAlgo_t a;std::memcpy(&a,(const char*)codes+index*sizeof(a),sizeof(a));float alpha=1.f,beta=0.f;
  auto s=cublasLtMatmul(pg_handle,l.op,&alpha,w,l.a,x,l.b,&beta,out,l.c,out,l.c,&a,work,bytes,(cudaStream_t)stream);return s==CUBLAS_STATUS_SUCCESS?0:(int)s;
}
'''


_DF_PG = None


def _df_pg():
    global _DF_PG
    if _DF_PG is not None:
        return _DF_PG
    import ctypes as _df_ctypes
    import hashlib as _df_hashlib
    import os as _df_os
    from torch.utils.cpp_extension import load_inline as _df_load_inline
    target = "9.0a" if torch.cuda.get_device_capability()[0] == 9 else "12.0"
    old = _df_os.environ.get("TORCH_CUDA_ARCH_LIST")
    _df_os.environ["TORCH_CUDA_ARCH_LIST"] = target
    try:
        path = _df_load_inline(
            name="muse_dflash_pg_" + _df_hashlib.sha1(_DF_PG_RAW.encode()).hexdigest()[:12],
            cpp_sources="", cuda_sources=_DF_PG_RAW, functions=None,
            is_python_module=False, no_implicit_headers=True,
            extra_cuda_cflags=["-O3", "-std=c++17"], extra_ldflags=["-lcublasLt"], verbose=False)
    finally:
        if old is None:
            _df_os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            _df_os.environ["TORCH_CUDA_ARCH_LIST"] = old
    lib = _df_ctypes.CDLL(path)
    P, I, S = _df_ctypes.c_void_p, _df_ctypes.c_int, _df_ctypes.c_size_t
    lib.pg_algo_bytes.restype = I
    lib.pg_options.argtypes = [I] * 5 + [S, P, I, _df_ctypes.POINTER(I)]
    lib.pg_run.argtypes = [P] * 5 + [I] * 6 + [S, P]
    class _Lt:
        def options(self, w, x, out, nn, workspace):
            codes = torch.empty((64, lib.pg_algo_bytes()), dtype=torch.uint8)
            count = _df_ctypes.c_int()
            status = lib.pg_options(out.shape[1], x.shape[0], x.shape[1], w.stride(0), nn,
                                    workspace, codes.data_ptr(), 64, _df_ctypes.byref(count))
            if status:
                raise RuntimeError(f"pg options failed {status}")
            return codes[:count.value].clone()
        def run(self, w, x, out, work, codes, index, nn):
            status = lib.pg_run(w.data_ptr(), x.data_ptr(), out.data_ptr(), work.data_ptr(), codes.data_ptr(),
                                out.shape[1], x.shape[0], x.shape[1], w.stride(0), nn, index,
                                work.numel(), torch.cuda.current_stream(x.device).cuda_stream)
            if status:
                raise RuntimeError(f"pg run failed {status}")
    _DF_PG = _Lt()
    return _DF_PG


def _df_stock(x, w, stock_nn):
    return torch.mm(x, w) if stock_nn else torch.nn.functional.linear(x, w)


def _df_event(fn, flush, banks, x):
    """Cold-L2 median for one graph-safe eager call."""
    values = []
    for i, fresh in enumerate(banks):
        x.copy_(fresh)
        flush.add_(1)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop) * 1000.0)
    return _df_statistics.median(values)


def _df_skin(mod, x, w, out, role, m):
    """Portable GEMV family; invalid tile combinations are rejected by caller."""
    skin_fn = globals().get("skinny_gemm")
    if skin_fn is not None:
        layer = "o" if role == "o" else "unpooler" if role == "down" else "qkv_gate"
        out.copy_(skin_fn(x, w, out.shape[1], layer))
        return
    # These are the same ABI fields as the proj portfolio's small-M kernel.
    nt = 1 if m <= 8 else 2
    tr, u, wn, wk, ks, xp, ld = 1, 1, 1, 1, 1, 0, 0
    mod.skinny(w, w, x, out, out.shape[1], nt, tr, u, wn, wk, ks, False, xp, ld)


def _df_lt(mod, w, x, out, workspace, codes, index, nn, ltmod=None):
    if ltmod is not None:
        ltmod.run(w, x, out, workspace, codes, index, nn)
        return
    if hasattr(mod, "all_lt"):
        mod.all_lt(w, x, out, workspace, codes, index, nn, 0)
    elif nn:
        mod.lt_nn_run(w, x, out, workspace, codes, index)
    else:
        mod.lt_run(w, x, out, workspace, codes, index)


def _df_options(mod, w, x, out, nn, workspace, ltmod=None):
    if ltmod is not None:
        return ltmod.options(w, x, out, nn, workspace)
    if hasattr(mod, "all_options"):
        value = mod.all_options(w, x, out, nn, 0, workspace)
        return value[0] if isinstance(value, tuple) else value
    return mod.lt_nn_options(w, x, out) if nn else mod.lt_options(w, x, out)


def _df_measure(deadline, mod, role, n, k, mult, stock_nn, workspace, flush, ltmod=None):
    rows = (8, 32, 128, 256) if mult == 8 else (7, 28, 112, 224)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / (k ** 0.5)
    # NN is the serving context_kv layout: the released weight is [K,N].
    op_w = w.t().contiguous() if stock_nn else w
    result = []
    for m in rows:
        if _df_time.monotonic() >= deadline:
            break
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        banks = [torch.randn_like(x) for _ in range(3)]
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        # Clone the operands: a bad Lt layout must not be allowed to mutate the
        # source and then compare against a post-mutation stock call.
        expected = _df_stock(x.clone(), op_w.clone() if stock_nn else w.clone(), stock_nn)
        # Build all heuristic choices on the exact serving shape. The host's
        # native wrapper checks every code and the calls below remain graph-safe.
        codes = _df_options(mod, op_w if stock_nn else w, x, out,
                            int(stock_nn), workspace.numel(), ltmod)
        trials = [("stock", _df_event(lambda: _df_stock(x, op_w if stock_nn else w, stock_nn),
                                       flush, banks, x))]
        for idx in range(int(codes.shape[0])):
            if _df_time.monotonic() >= deadline:
                break
            try:
                def run_lt(index=idx):
                    _df_lt(mod, op_w if stock_nn else w, x, out, workspace, codes,
                            index, int(stock_nn), ltmod)
                run_lt()
                if not torch.equal(out, expected):
                    rel = float((out.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-6))
                    raise RuntimeError(f"Lt output differs from stock rel={rel:.3e}")
                us = _df_event(run_lt, flush, banks, x)
                trials.append((f"lt{idx}", us))
            except Exception as exc:
                if idx == 0:
                    print(f"DFLASH_REJECT role={role} m={m} lt_family {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        # The small-M kernel uses the released row-major layout only.
        if not stock_nn and m <= 32 and n % 16 == 0:
            try:
                def run_skin():
                    _df_skin(mod, x, w, out, role, m)
                run_skin()
                if not torch.equal(out, expected):
                    rel = float((out.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-6))
                    raise RuntimeError(f"skin output differs from stock rel={rel:.3e}")
                trials.append(("skin", _df_event(run_skin, flush, banks, x)))
            except Exception as exc:
                print(f"DFLASH_REJECT role={role} m={m} skin {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        trials.sort(key=lambda item: item[1])
        # Recheck the incumbent and three finalists with twenty paired calls.
        finalists = trials[:4]
        paired = []
        for tag, _ in finalists:
            if _df_time.monotonic() >= deadline:
                break
            fn = (lambda: _df_stock(x, op_w if stock_nn else w, stock_nn)) if tag == "stock" else None
            if tag.startswith("lt"):
                idx = int(tag[2:])
                fn = lambda idx=idx: _df_lt(mod, op_w if stock_nn else w, x, out, workspace, codes, idx, int(stock_nn), ltmod)
            if tag == "skin":
                fn = lambda: _df_skin(mod, x, w, out, role, m)
            if fn is None:
                continue
            # Fresh inputs and cold-L2 flush mirror the serving diagnostic.
            vals = []
            for rep in range(20):
                x.copy_(banks[rep % len(banks)])
                flush.add_(1)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); fn(); b.record(); b.synchronize()
                vals.append(a.elapsed_time(b) * 1000.0)
            paired.append((tag, _df_statistics.median(vals)))
        best = min(paired, key=lambda item: item[1]) if paired else ("stock", trials[0][1])
        base = next((v for t, v in paired if t == "stock"), trials[0][1])
        result.append(f"DFLASH_PICK role={role} m={m} best={best[0]} us={best[1]:.3f} stock_us={base:.3f} ratio={best[1]/base:.4f} lt={len(codes)}")

    # FC concatenation is a real serving path: compare cat+GEMM against a
    # prepared buffer and the same selected Lt family without changing math.
    if role == "context_fc" and _df_time.monotonic() < deadline:
        m = 32
        left = torch.randn((m, k // 2), device="cuda", dtype=torch.bfloat16)
        right = torch.randn_like(left)
        cat = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
        banks = [(torch.randn_like(left), torch.randn_like(right)) for _ in range(3)]
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        def stock_cat():
            return torch.nn.functional.linear(torch.cat((left, right), dim=-1), w)
        def fused_cat():
            cat[:, :k // 2].copy_(left); cat[:, k // 2:].copy_(right)
            return torch.nn.functional.linear(cat, w)
        stock_cat(); fused_cat()
        svals=[]; fvals=[]
        for rep in range(20):
            left.copy_(banks[rep % 3][0]); right.copy_(banks[rep % 3][1]); flush.add_(1)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record(); stock_cat(); b.record(); b.synchronize(); svals.append(a.elapsed_time(b)*1000.0)
            flush.add_(1); a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record(); fused_cat(); b.record(); b.synchronize(); fvals.append(a.elapsed_time(b)*1000.0)
        result.append(f"DFLASH_CONCAT m=32 stock_us={_df_statistics.median(svals):.3f} fused_buffer_us={_df_statistics.median(fvals):.3f} ratio={_df_statistics.median(fvals)/_df_statistics.median(svals):.4f}")
    return result


def lab(deadline, weights, state):
    """Measure all serving TP8 drafter dense calls in one proj-host run."""
    del weights
    started = _df_time.monotonic()
    mod = state["mod"]
    workspace = torch.empty(512 << 20, device="cuda", dtype=torch.uint8)
    flush = torch.zeros(128 << 20, device="cuda", dtype=torch.uint8)
    lines = []
    try:
        ltmod = _df_pg()
        lines.append("DFLASH_LT_READY heuristics=cuBLASLt")
    except Exception as exc:
        ltmod = None
        lines.append(f"DFLASH_LT_REJECT {type(exc).__name__}: {str(exc)[:160]}")
    for role, n, k, mult, stock_nn in _DFLASH_ROLES:
        if _df_time.monotonic() >= deadline:
            lines.append(f"DFLASH_BUDGET role={role}")
            break
        try:
            lines.extend(_df_measure(deadline, mod, role, n, k, mult, stock_nn, workspace, flush, ltmod))
        except Exception as exc:
            lines.append(f"DFLASH_ERROR role={role} {type(exc).__name__}: {str(exc)[:160]}")
    # The receipt tail retains this compact summary even if the host's tuner is verbose.
    lines.append(f"DFLASH_DONE elapsed_s={_df_time.monotonic()-started:.1f} roles={len(_DFLASH_ROLES)}")
    return lines
