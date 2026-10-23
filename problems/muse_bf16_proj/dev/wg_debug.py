import os, sys, torch
from torch.utils.cpp_extension import load_inline
sys.path.insert(0, "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj/dev")
import importlib.util
spec = importlib.util.spec_from_file_location("c", "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj/cand_wg2.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
src = c.CUDA_SRC.replace("MP_INSTANCES", "") + c.wg_source([(8, 1, 1, 4), (128, 1, 2, 2)])
mod = load_inline(name="wgdbg3", cpp_sources=c.CPP_SRC, cuda_sources=src, functions=["wg_gemm"],
                  extra_cuda_cflags=["-O3", "-std=c++20", "-DMP_SPIN_LIMIT=20000", "-DMP_DEBUG"], verbose=False)
for (n, k, m, bt, nsub, ncw, st, ks) in [(64, 256, 1, 8, 1, 1, 4, 1), (2304, 8192, 1, 8, 1, 1, 4, 1), (2304, 8192, 8, 8, 1, 1, 4, 4), (4096, 8192, 128, 128, 1, 2, 2, 2)]:
    w = torch.randn(n, k, device="cuda").bfloat16()
    x = torch.randn(m, k, device="cuda").bfloat16()
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    mod.wg_gemm(w, x, out, n, bt, nsub, ncw, st, ks)
    torch.cuda.synchronize()
    ref = x.float() @ w.float().T
    print((n, k, m, bt, nsub, ncw, st, ks), "err", ((out.float() - ref).norm() / ref.norm()).item(), flush=True)
