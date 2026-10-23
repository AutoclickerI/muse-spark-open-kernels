"""Run cand_v3 with the attention agent's kernel (problems/muse_attention_decode/<file>) as the
attention stage; qkv from cuBLAS (complete BF16). Tests + a few timings, local only."""
import sys, importlib.util, time, statistics
import torch
repo = "/home/jihoon/competitions/muse_kernel"
pdir = repo + "/problems/muse_decode_layer"
sys.path[:0] = [pdir]
spec = importlib.util.spec_from_file_location("reference", pdir + "/reference.py")
ref = importlib.util.module_from_spec(spec); sys.modules["reference"] = ref; spec.loader.exec_module(ref)
ref._fa3_scheduler_metadata = lambda *a, **k: None
spec = importlib.util.spec_from_file_location("cand", pdir + "/cand_v3.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
t0 = time.time()
spec = importlib.util.spec_from_file_location("attn", repo + "/problems/muse_attention_decode/" + sys.argv[1])
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
A._ext()
print(f"attention ext build+load: {time.time() - t0:.1f} s")

def stage_attention(qkv, data, state):
    P, S = qkv
    cfg = data["config"]
    qkv_bf16 = (P.sum(0) if S > 1 else P[0]).to(torch.bfloat16)
    d2 = dict(data); d2["qkv"] = qkv_bf16.view(cfg["batch"], cfg["q_len"], -1)
    return A.kernel_fn(d2).view(cfg["tokens"], -1)
c.stage_attention = stage_attention
w = ref.generate_weights(ref.WEIGHTS_SEED)
state = c.prepare({k: v.clone() for k, v in w.items()})
import re
tests = [l for l in open(pdir + "/TASK.md").read().split("```")[3].strip().splitlines()]
for i, line in enumerate(tests):
    case = {k: (int(v) if v.strip().isdigit() else (float(v) if v.replace('.', '').isdigit() else v.strip()))
            for k, v in (p.strip().split(" ", 1) for p in line.split(","))}
    case["seed"] = 4242 + i
    case.update(ref.case_fields(case, w))
    data = ref.generate_input(**case)
    out = c.kernel_fn(data, state)
    ok, msg = ref.check_implementation(data, ref.collect(data, out), w)
    print("PASS" if ok else "FAIL", line, msg[:90])
