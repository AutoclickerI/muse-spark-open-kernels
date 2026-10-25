"""Exercise the unsent paired payload, with explicit local-only K16 shared-capacity stand-in."""
import linecache
import time
from pathlib import Path
import torch
import k16_standin

R = Path(__file__).resolve().parents[4]
buf = []


def flush():
    if not buf:
        buf.append(torch.empty(1 << 30, device='cuda', dtype=torch.uint8))
    buf[0].view(torch.int64).sum()


text = (R / 'prof/lab/moe_nt1_pipe_rebuild.py').read_text()
name = '<pipe rebuild lab>'
linecache.cache[name] = (len(text), None, text.splitlines(True), name)
n = {'__name__': 'h200lab', 'lab_cold_l2': flush}
exec(compile(text, name, 'exec'), n)
calls = k16_standin.install(n)
print('LOCAL sm120 GPU0, K16 native one-stage math checks original two-stage ABI; no H200 timing.', flush=True)
lines = n['lab'](time.monotonic() + 280, None)
for line in lines:
    print(line, flush=True)
assert any('END seeds=32' in line for line in lines)
assert not any(line.startswith(('FAIL ', 'NODE_ERROR ', 'BUDGET ')) for line in lines)
assert sum(len(line) + 1 for line in lines) < 8750
assert calls
print('LOCAL_K16_FORCED_ABI', len(calls), flush=True)
