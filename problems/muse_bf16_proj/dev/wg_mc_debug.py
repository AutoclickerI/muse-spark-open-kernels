import sys, torch, importlib.util
from torch.utils.cpp_extension import load_inline
spec = importlib.util.spec_from_file_location("c", "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj/dev/_wg2_local.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
src = c.CUDA_SRC.replace("MP_INSTANCES", "") + c.wg_source([(32, 1, 1, 4, 2), (32, 1, 1, 4, 1), (64, 1, 1, 4, 2), (16, 1, 1, 4, 2)])
mod = load_inline(name="wgdbg_mc4", cpp_sources=c.CPP_SRC, cuda_sources=src, functions=["wg_gemm"],
                  extra_cuda_cflags=["-O3", "-std=c++20", "-DMP_SPIN_LIMIT=200000", "-DMP_DEBUG", "-lineinfo"], verbose=False)
mc = int(sys.argv[1]); ks = int(sys.argv[2])
n, k, m = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
w = torch.randn(n, k, device="cuda").bfloat16(); x = torch.randn(m, k, device="cuda").bfloat16()
out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
mod.wg_gemm(w, x, out, n, int(sys.argv[6]), 1, 1, 4, ks, mc); torch.cuda.synchronize()
ref = x.float() @ w.float().T
print("mc", mc, "ks", ks, "n", n, "k", k, "err", ((out.float() - ref).norm() / ref.norm()).item())
rows = ((out.float() - ref).norm(dim=0) / ref.norm(dim=0)).view(-1, 64).max(dim=1).values
print("per 64-row tile err:", [round(v, 3) for v in rows.tolist()])
toks = ((out.float() - ref).norm(dim=1) / ref.norm(dim=1))
print("per token err:", [round(v, 2) for v in toks.tolist()])
