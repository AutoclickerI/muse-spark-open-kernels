import sys, torch, importlib.util, statistics as st
spec = importlib.util.spec_from_file_location("c", "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj/dev/_wg2_local.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
spec2 = importlib.util.spec_from_file_location("t", "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj/dev/sweep4_tail.py")
exec(open("/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj/dev/sweep4_tail.py").read().split("_prepare_base = prepare")[0], c.__dict__)
sys.path.insert(0, "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj")
import reference as ref
w = ref.generate_weights(ref.WEIGHTS_SEED)
stt = c.prepare(w)
mod = c._module()
T = c._Timer()
for layer, n, k in (("pooler", 4096, 8192), ("qkv_gate", 2304, 8192), ("o", 8192, 1024), ("unpooler", 1024, 4096)):
    for m in (1, 32):
        x = torch.randn(m, k, device="cuda").bfloat16()
        W = stt[layer]
        tcb = T.time(c._graph(lambda: torch.nn.functional.linear(x, W[:n])))
        tsk = T.time(c._graph(lambda: c.skinny_gemm(x, W, n, layer)))
        res = []
        bt = c._bt(m)
        for (b, nsub, ncw, stg) in c.WG_INSTANCES:
            if b != bt or n % (64 * nsub * ncw) or stg * (64*nsub*ncw + bt) * 128 + 2048 > 99 * 1024: continue
            for ks in (1, 2, 4, 8):
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                res.append((T.time(c._graph(lambda: mod.wg_gemm(W, x, out, n, bt, nsub, ncw, stg, ks))), (nsub, ncw, stg, ks)))
        res.sort()
        print(f"{layer} m{m}: cublas {tcb:.1f} skinny {tsk:.1f} wg-emu best {res[0][0]:.1f} {res[0][1]}  {res[1][0]:.1f} {res[1][1]}")
