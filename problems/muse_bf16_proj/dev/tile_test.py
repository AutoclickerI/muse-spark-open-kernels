import sys, importlib.util, torch, statistics as st
sys.path.insert(0, "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj")
import reference as ref
spec = importlib.util.spec_from_file_location("c", sys.argv[1]); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
w = ref.generate_weights(ref.WEIGHTS_SEED)
stt = c.prepare({k: v.clone() for k, v in w.items()})
mod = c._module()
T = c._Timer()
cases = [("pooler", 128, (2, 2, 8, 8, 4)), ("pooler", 128, (1, 2, 8, 4, 4)), ("unpooler", 128, (2, 2, 8, 4, 4)),
         ("qkv_gate", 128, (1, 2, 4, 4, 4)), ("o", 128, (1, 2, 8, 2, 4)), ("o", 128, (2, 2, 8, 4, 3)),
         ("pooler", 256, (2, 2, 8, 8, 4)), ("pooler", 33, (1, 2, 8, 4, 4)), ("router", 256, (1, 2, 4, 8, 4)),
         ("router", 128, (1, 2, 4, 8, 4)), ("unpooler", 130, (2, 2, 8, 4, 4)), ("qkv_gate", 64, (1, 2, 8, 4, 3))]
for layer, m, (tr, u, wn, ks, stg) in cases:
    d = ref.generate_input(m, layer, 77 + m, "outliers")
    x, n = d["x"], d["config"]["n"]
    nt = 16 if m > 64 else 8 if m > 32 else 4
    split = layer == "router"
    W0 = stt["router_hi"] if split else stt[layer]
    W1 = stt["router_lo"] if split else W0
    out = torch.empty(m, n, device="cuda", dtype=torch.float32 if split else torch.bfloat16)
    fn = lambda: mod.tile(W0, W1, x, out, n, nt, tr, u, wn, ks, split, stg, 0)
    fn(); torch.cuda.synchronize()
    ok, msg = ref.check_implementation(d, out, w)
    t = T.time(c._graph(fn))
    if split:
        hi_t, lo_t = W0.t().contiguous(), W1.t().contiguous()
        def rf():
            a = torch.mm(x, hi_t, out_dtype=torch.float32); a += torch.mm(x, lo_t, out_dtype=torch.float32); return a
    else:
        rf = lambda: torch.nn.functional.linear(x, W0[:n])
    tr_ = T.time(c._graph(rf))
    print(f"{'PASS' if ok else 'FAIL'} {layer} m{m} cfg{(tr,u,wn,ks,stg)}: tile {t:.2f} us  cublas {tr_:.2f} us  {msg[:60]}")
