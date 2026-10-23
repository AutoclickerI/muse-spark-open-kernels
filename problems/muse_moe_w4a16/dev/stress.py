"""Additional local correctness: wide weight scales, nearby batches, hot and empty routing."""
import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import reference as R
from nvfp4 import random_weights

spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
gen = torch.Generator(device="cuda").manual_seed(729533)
weights = {}
for name, rows in (("w13", 8192), ("w2", 4096)):
    q, sf, gs = random_weights(rows, 4096, generator=gen, device="cuda", leading=(32,), wide_scales=True)
    weights[name], weights[name + "_sf"] = q, sf
    weights[name + "_gs"] = gs.expand(32).contiguous()
weights["post_norm_w"] = (torch.randn(4096, generator=gen, device="cuda") * 0.1).bfloat16()
state = candidate.prepare(weights)
torch.backends.cuda.matmul.allow_tf32 = False

for bs, dist in ((2, "normal"), (9, "skewed"), (65, "normal"), (300, "hot"), (17, "empty")):
    data = R.generate_input(bs, "ep8", 52025 + bs, dist="skewed" if dist == "skewed" else "normal")
    if dist == "hot":
        data["topk_ids"] = torch.arange(8, dtype=torch.int32, device="cuda")[None, :].expand(bs, 8).contiguous()
    if dist == "empty":
        data["topk_ids"].add_(32).remainder_(224).add_(32)
    candidate.kernel_fn(data, state)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        candidate.kernel_fn(data, state)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = candidate.kernel_fn(data, state)
    for seed in (0, 1):
        data["hidden_states"].normal_(generator=gen)
        graph.replay()
        torch.cuda.synchronize()
        if dist == "empty":
            ok, msg = bool((out == 0).all()), "all-remote routes return zeros"
        else:
            ok, msg = R.check_implementation(data, out, weights)
        print(f"bs={bs} dist={dist} replay={seed}: {ok} {msg}", flush=True)
        if not ok:
            raise RuntimeError("additional correctness test failed")
    del graph, out, data
    torch.cuda.empty_cache()
