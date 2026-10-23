"""Per-kernel GPU times of a candidate's graph replay on chosen shapes (local GPU).
usage: prof.py <cand.py> [case ...]   case like 'b1s' 'b32f' 'v32s' (b=decode, v=verify; s/f layer)"""
import sys, importlib.util, statistics, collections
import torch
repo = "/home/jihoon/competitions/muse_kernel"
pdir = repo + "/problems/muse_decode_layer"
sys.path[:0] = [pdir]
spec = importlib.util.spec_from_file_location("reference", pdir + "/reference.py")
ref = importlib.util.module_from_spec(spec); sys.modules["reference"] = ref; spec.loader.exec_module(ref)
ref._fa3_scheduler_metadata = lambda *a, **k: None
spec = importlib.util.spec_from_file_location("cand", sys.argv[1])
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
import os
if os.environ.get("NOPDL"): c.PDL = False
w = ref.generate_weights(ref.WEIGHTS_SEED)
state = c.prepare({k: v.clone() for k, v in w.items()})
cases = sys.argv[2:] or ["b1s", "b32f", "v32s"]
flush = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
for name in cases:
    kind, rest = name[0], name[1:]
    layer = "sliding" if rest[-1] == "s" else "full"
    batch = int(rest[:-1])
    case = dict(batch=batch, ctx=80000, layer=layer, page=1, dist="soft", seed=77,
                q_len=8 if kind == "v" else 1, spread="agentic" if batch > 1 else "fixed")
    case.update(ref.case_fields(case, w))
    data = ref.generate_input(**case)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            c.kernel_fn(data, state)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = c.kernel_fn(data, state)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(20):
        flush.sum(); st.record(); g.replay(); en.record(); en.synchronize(); ts.append(st.elapsed_time(en) * 1e3)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            flush.sum(); g.replay()
        torch.cuda.synchronize()
    agg = collections.defaultdict(list)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and "reduce_kernel" not in e.name and "direct_copy" not in e.name:
            agg[e.name].append(e.device_time)
    print(f"== {name}: replay median {statistics.median(ts):.1f} us (incl ~4us replay overhead)")
    tot = 0
    for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        per = sum(v) / 5
        tot += per
        print(f"   {per:8.1f} us  x{len(v)//5}  {k[:90]}")
    print(f"   {tot:8.1f} us  total kernel time")
