"""Execute every compiled kernel, split reducer and tail on the local GPU.

cp.async and mma.sync execute on both GPUs. Configurations exceeding sm_120's
99 KiB/CTA use fewer ring stages locally; the fragment/reduction/glue is real.
Full-sized cold timings belong to the judge, not this correctness screen.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("payload", nargs="?", type=Path, default=HERE.parent / "streaming_gemv_lab.py")
parser.add_argument("--output", type=Path, default=HERE / "validation.json")
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("stream_proj", args.payload)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
start = time.monotonic()
checked, max_rel, graphs = 0, 0., 0
for m in (1, 7, 8, 16, 32, 33, 128, 130, 256):
    # 257 rows exercise all row tails and all counter/tile layouts. K=896
    # exercises zero-filled K tails in the 256-wide ring at every split.
    w = torch.randn((257, 896), device="cuda", dtype=torch.bfloat16)
    x = mod.sp_inputs(m, 896, 718, w.device)
    ref = torch.mm(x, w.t(), out_dtype=torch.float32)
    for cfg in mod.SP_CONFIGS:
        if cfg < 10 and m > 8:
            continue
        for splits in (1, 2, 4):
            op = mod.StreamingProjection(w, m, cfg, splits, fp32=True)
            rel = mod.sp_check(op(x), ref, x)
            max_rel = max(max_rel, rel)
            checked += 1
            if bool(op.counters.any()):
                raise RuntimeError(f"counter failed {m}/{cfg}/{splits}")
            if splits == 4:
                graph, out = mod.sp_graph(lambda: op(x))
                for seed in (31, 43, 79):
                    x.copy_(mod.sp_inputs(m, 896, seed, w.device))
                    graph.replay()
                    mod.sp_check(out, torch.mm(x, w.t(), out_dtype=torch.float32), x)
                if bool(op.counters.any()):
                    raise RuntimeError("counter failed graph replay")
                graph.reset()
                graphs += 1
                # Restore reference after replay mutated the captured input.
                x.copy_(mod.sp_inputs(m, 896, 718, w.device))
    print(f"TAIL_PASS m={m} checked={checked} graphs={graphs}", flush=True)

# Real decode pool+router: independent FP32 weights, exact hi/lo layout.
router = torch.randn((256, 8192), device="cuda", dtype=torch.float32) / 8192**.5
hi = router.to(torch.bfloat16)
lo = (router - hi.float()).to(torch.bfloat16)
w = torch.cat((torch.randn((4096, 8192), device="cuda", dtype=torch.bfloat16), hi, lo))
for m in (1, 7, 8, 16, 32, 128, 256):
    x = mod.sp_inputs(m, 8192, 41, w.device)
    ref = torch.mm(x, w.t(), out_dtype=torch.float32)
    for cfg, splits in mod.sp_variants(m, 8192):
        op = mod.StreamingProjection(w, m, cfg, splits, fp32=True)
        max_rel = max(max_rel, mod.sp_check(op(x), ref, x, router))
        checked += 1
    print(f"ROUTER_PASS m={m} checked={checked}", flush=True)

result = dict(device=torch.cuda.get_device_name(), local_shared_cap=98304, checked=checked, graphs=graphs,
              max_fp32_rel=max_rel, elapsed_s=time.monotonic()-start,
              source_sha256=hashlib.sha256(mod.SP_CUDA.encode()).hexdigest(),
              payload_sha256=hashlib.sha256(args.payload.read_bytes()).hexdigest())
print(json.dumps(result), flush=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
