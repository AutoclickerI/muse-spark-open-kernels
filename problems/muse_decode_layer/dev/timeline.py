"""Kernel timeline (start offset, duration, stream) of one graph replay. usage: timeline.py cand.py case"""
import sys, importlib.util, os
import torch
pdir = "/home/jihoon/competitions/muse_kernel/problems/muse_decode_layer"
sys.path[:0] = [pdir]
spec = importlib.util.spec_from_file_location("reference", pdir + "/reference.py")
ref = importlib.util.module_from_spec(spec); sys.modules["reference"] = ref; spec.loader.exec_module(ref)
ref._fa3_scheduler_metadata = lambda *a, **k: None
spec = importlib.util.spec_from_file_location("cand", sys.argv[1])
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
if os.environ.get("NOPDL"): c.PDL = False
w = ref.generate_weights(ref.WEIGHTS_SEED)
state = c.prepare({k: v.clone() for k, v in w.items()})
name = sys.argv[2]
kind, rest = name[0], name[1:]
batch = int(rest[:-1])
case = dict(batch=batch, ctx=80000, layer="sliding" if rest[-1] == "s" else "full", page=1, dist="soft", seed=77,
            q_len=8 if kind == "v" else 1, spread="agentic" if batch > 1 else "fixed")
case.update(ref.case_fields(case, w))
data = ref.generate_input(**case)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2): c.kernel_fn(data, state)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = c.kernel_fn(data, state)
flush = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
for _ in range(5): flush.sum(); g.replay()
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    flush.sum(); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
evs = [e for e in evs if "reduce_kernel" not in e.name and "direct_copy" not in e.name]
t0 = min(e.time_range.start for e in evs)
end = 0
for e in sorted(evs, key=lambda e: e.time_range.start):
    st = e.time_range.start - t0
    print(f"{st:8.1f} +{e.time_range.elapsed_us():7.1f}  gap {st - end:6.1f}  {e.name[:70]}")
    end = max(end, e.time_range.end - t0)
print(f"end {end:.1f} us")
