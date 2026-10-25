"""Local (sm_120) screen of a lab payload: synthetic proj weights at the real shapes, then lab()."""
import runpy, sys, time
import torch

path = sys.argv[1]
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(5)
def r(*s, dt=torch.bfloat16, scale=0.02):
    return (torch.randn(*s, device=dev, generator=g) * scale).to(dt)
weights = {"qkv_gate": r(2304, 8192), "o": r(8192, 1024), "pooler": r(4096, 8192),
           "unpooler": r(8192, 4096), "router": r(256, 8192, dt=torch.float32)}
_cold = []
def lab_cold_l2():  # same as tools/h200_lab.py: read 1 GiB before each timed replay
    if not _cold:
        _cold.append(torch.empty(1 << 30, dtype=torch.uint8, device=dev))
    _cold[0].sum()
ns = runpy.run_path(path, init_globals={"lab_cold_l2": lab_cold_l2})
t0 = time.monotonic()
out = ns["lab"](t0 + budget, weights)
print("\n".join(out))
