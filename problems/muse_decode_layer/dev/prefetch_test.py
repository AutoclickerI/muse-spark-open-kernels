import torch, triton, triton.language as tl

@triton.jit
def _pf(ptrs):
    return tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [ptrs],
                                     dtype=tl.int32, is_pure=False, pack=1)

@triton.jit
def k(W, OUT, N: tl.constexpr):
    pid = tl.program_id(0)
    lines = tl.arange(0, 64)
    _pf(W + pid * 64 * 64 + lines * 64)
    tl.extra.cuda.gdc_wait()
    x = tl.load(W + pid * 4096 + tl.arange(0, 4096))
    tl.store(OUT + pid, tl.sum(x.to(tl.float32), 0))

w = torch.randn(64 * 4096, device="cuda", dtype=torch.bfloat16)
o = torch.empty(64, device="cuda")
k[(64,)](w, o, N=1, launch_pdl=True)
torch.cuda.synchronize()
print(torch.allclose(o, w.float().view(64, 4096).sum(1), rtol=1e-3, atol=1e-2))
asm = k.cache[0] if hasattr(k, "cache") else None
import glob
