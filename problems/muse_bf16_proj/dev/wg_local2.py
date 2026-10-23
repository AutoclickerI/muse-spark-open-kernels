import sys, torch, importlib.util
from torch.utils.cpp_extension import load_inline
spec = importlib.util.spec_from_file_location("c", "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj/dev/_wg2_local.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
src = c.CUDA_SRC.replace("MP_INSTANCES", "") + c.wg_source(c.WG_INSTANCES)
mod = load_inline(name="wgdbg_mc1", cpp_sources=c.CPP_SRC, cuda_sources=src, functions=["wg_gemm"],
                  extra_cuda_cflags=["-O3", "-std=c++20", "-DMP_SPIN_LIMIT=200000", "-DMP_DEBUG"], verbose=False)
sys.path.insert(0, "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj")
import reference as ref
w = ref.generate_weights(ref.WEIGHTS_SEED)
router = w["router"].float()
hi = router.bfloat16(); lo = (router - hi.float()).bfloat16()
wr = torch.cat([hi, lo], 1).contiguous()
bad = 0
only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
for layer, (n, k) in {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "router": (256, 8192)}.items():
    if only and layer not in only: continue
    for m in (1, 5, 8, 9, 16, 31, 32, 33, 64, 65, 128, 130, 256):
        nn = 1024 if layer == "unpooler" and m <= 8 else n
        bt = c._bt(m)
        x = torch.randn(m, k, device="cuda").bfloat16()
        if layer == "router":
            W, odt, refv, tol = wr, torch.float32, (x.float() @ router.T), 1e-4
        else:
            W, odt, tol = w[layer], torch.bfloat16, 5e-3
            refv = x.float() @ W[:nn].float().T
        cnt = 0
        for (b, nsub, ncw, stg, mc) in c.WG_INSTANCES:
            if b != bt: continue
            br = 64 * nsub * ncw
            if nn % (br * mc): continue
            if stg * (br + bt) * 128 + 2048 > 99 * 1024: continue
            for ks in (1, 2, 4, 8):
                if (W.shape[1] // 64) < ks or ks * mc > 8: continue
                out = torch.full((m, nn), float("nan"), device="cuda", dtype=odt)
                try:
                    mod.wg_gemm(W, x, out, nn, bt, nsub, ncw, stg, ks, mc); torch.cuda.synchronize()
                except Exception as e:
                    print("ERR", layer, m, (nsub, ncw, stg, ks, mc), str(e)[:100], flush=True); bad += 1; continue
                err = ((out.float() - refv).norm() / refv.norm()).item()
                cnt += 1
                if not (err < tol and torch.isfinite(out).all().item()):
                    bad += 1; print("BAD", layer, m, (bt, nsub, ncw, stg, ks, mc), f"{err:.2e}", flush=True)
        print("done", layer, m, cnt, flush=True)
print("bad:", bad)
