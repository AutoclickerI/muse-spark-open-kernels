import sys, importlib.util, time, torch
pdir = "/home/jihoon/competitions/muse_kernel/problems/muse_bf16_proj"
sys.path.insert(0, pdir)
import reference as ref
spec = importlib.util.spec_from_file_location("cand", sys.argv[1]); cand = importlib.util.module_from_spec(spec); spec.loader.exec_module(cand)
w = ref.generate_weights(ref.WEIGHTS_SEED)
t0 = time.time(); st = cand.prepare({k: v.clone() for k, v in w.items()}); torch.cuda.synchronize(); print("prepare", round(time.time()-t0,1), "s")
ms = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1,2,5,7,8,9,15,16,17,31,32,33,48,63,64,65,128,130,256]
bad = 0
for layer in ["qkv_gate","o","pooler","unpooler","router"]:
    for m in ms:
        for dist in ["normal","outliers"]:
            d = ref.generate_input(m, layer, 1234+m, dist)
            out = cand.kernel_fn(d, st); torch.cuda.synchronize()
            ok, msg = ref.check_implementation(d, out, w)
            if not ok: bad += 1
            if not ok or (dist=="normal" and m in (1,8,32,256)): print("PASS" if ok else "FAIL", layer, m, dist, msg)
print("failures:", bad)
