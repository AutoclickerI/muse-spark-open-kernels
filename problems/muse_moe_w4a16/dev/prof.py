"""Per-kernel GPU times of a candidate's kernel_fn (graph replays, cold L2), via torch.profiler.
    CUDA_VISIBLE_DEVICES=1 tools/img python problems/muse_moe_w4a16/dev/prof.py <cand.py> [bs ...]"""
import importlib.util, sys, statistics
from pathlib import Path
import torch
P = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(P)]
import reference as R

def load(path):
    spec = importlib.util.spec_from_file_location("cand", path)
    m = importlib.util.module_from_spec(spec); sys.modules["cand"] = m; spec.loader.exec_module(m); return m

cand = load(sys.argv[1])
bss = [int(a) for a in sys.argv[2:]] or [1, 8, 16, 32, 128, 256]
w = R.generate_weights(R.WEIGHTS_SEED)
state = cand.prepare(w)
flush = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
for bs in bss:
    data = R.generate_input(bs, "ep8", 1234 + bs)
    out = cand.kernel_fn(data, state); torch.cuda.synchronize()
    ok, msg = R.check_implementation(data, out, w)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): cand.kernel_fn(data, state)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cand.kernel_fn(data, state)
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ts = []
    for _ in range(20):
        flush.sum(); ev[0].record(); g.replay(); ev[1].record(); ev[1].synchronize(); ts.append(ev[0].elapsed_time(ev[1]) * 1e3)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            flush.sum(); g.replay()
        torch.cuda.synchronize()
    ka = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and "reduce_kernel" not in e.name:
            ka.setdefault(e.name[:60], []).append(e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total)
    ids = data["topk_ids"].flatten().long(); c = torch.bincount(ids, minlength=256)[:32]
    print(f"bs {bs}: {'ok' if ok else 'FAIL ' + msg}; graph replay median {statistics.median(ts):.1f} us; touched {int((c>0).sum())} max/expert {int(c.max())} pairs {int(c.sum())}")
    for k, v in ka.items():
        print(f"    {statistics.median(v):8.1f} us  x{len(v)//10}  {k}")
