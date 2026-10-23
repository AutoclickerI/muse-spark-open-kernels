"""Local (sm_120) functional test of the TMA pipeline with the mma.sync emulation."""
import sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("c", sys.argv[1]); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
sys.path.insert(0, "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj")
import reference as ref
w = ref.generate_weights(ref.WEIGHTS_SEED)
st = c.prepare({k: v.clone() for k, v in w.items()})
mod = c._module()
wr = torch.cat([st["router_hi"], st["router_lo"]], dim=1).contiguous()
bad = 0
for layer, (n, k) in {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "router": (256, 8192)}.items():
    for m in (1, 5, 8, 9, 16, 31, 32, 33, 64, 65, 128, 130, 256):
        nn = 1024 if layer == "unpooler" and m <= 8 else n
        bt = c._bt(m)
        x = torch.randn(m, k, device="cuda").bfloat16()
        if layer == "router":
            W, odt, refv, tol = wr, torch.float32, x.double() @ w["router"].double().T, 1e-4
        else:
            W, odt, refv, tol = st[layer], torch.bfloat16, x.double() @ st[layer][:nn].double().T, 5e-3
        for (b, nsub, ncw, stg) in c.WG_INSTANCES:
            if b != bt: continue
            br = 64 * nsub * ncw
            if nn % br: continue
            smem = stg * (br + bt) * 128 + 2048
            if smem > 99 * 1024: stg = max(2, (99 * 1024 - 2048) // ((br + bt) * 128))
            if (b, nsub, ncw, stg) not in [tuple(i) for i in c.WG_INSTANCES]: continue
            for ks in (1, 2, 4, 8):
                if (W.shape[1] // 64) < ks: continue
                out = torch.full((m, nn), float("nan"), device="cuda", dtype=odt)
                try:
                    mod.wg_gemm(W, x, out, nn, bt, nsub, ncw, stg, ks); torch.cuda.synchronize()
                except Exception as e:
                    print("ERR", layer, m, (nsub, ncw, stg, ks), str(e)[:100]); bad += 1; continue
                err = ((out.double() - refv).norm() / refv.norm()).item()
                ok = err < tol and torch.isfinite(out).all().item()
                if not ok:
                    bad += 1; print("BAD", layer, m, (bt, nsub, ncw, stg, ks), f"{err:.2e}")
print("bad:", bad)
