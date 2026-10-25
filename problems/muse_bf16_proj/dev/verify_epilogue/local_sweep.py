"""Run the exact sweep wrapper locally with the judge's wide READ flush."""
import runpy
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
g = torch.Generator(device='cuda').manual_seed(5)
weights = {name: (torch.randn(n, k, device='cuda', generator=g) * .02).bfloat16()
           for name, n, k in (('o', 8192, 1024), ('unpooler', 8192, 4096))}
buf = torch.empty(1 << 30, device='cuda', dtype=torch.uint8)
payload = sys.argv[1] if len(sys.argv)>1 else 'lab_verify_sweep.py'
ns = runpy.run_path(str(HERE / payload), init_globals={'lab_cold_l2': lambda: buf.view(torch.int64).sum()})
for line in ns['lab'](time.monotonic() + 240, weights):
    print(line, flush=True)
