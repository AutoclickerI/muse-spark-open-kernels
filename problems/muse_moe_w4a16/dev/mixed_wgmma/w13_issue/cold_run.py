"""Exercise the exact lab in a fresh native build directory within a shared-run budget."""
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[4]
os.environ['TORCH_EXTENSIONS_DIR'] = str(H / 'quad_cold_cache')
os.environ['CCACHE_DISABLE'] = '1'
import torch

payload = R / 'prof/lab/moe_w13_issue_m256.py'
ns = runpy.run_path(str(payload))
start = time.monotonic()
lines = ns['lab'](start + 78., {'device': torch.empty(1, device='cuda')})
result = dict(sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
              budget_s=78., used_s=time.monotonic() - start,
              compiler_cache_disabled=True, lines=lines)
(H / 'cold.json').write_text(json.dumps(result, indent=2) + '\n')
print('\n'.join(lines))
assert all('BUDGET' not in line and 'REJECT' not in line for line in lines)
assert sum(line.startswith('S') and ' P=' in line for line in lines) == 12
assert sum(line.startswith('AGG ') and 'n=12 ' in line and line.endswith('PASS') for line in lines) == 12
assert sum(line.startswith('PH ') for line in lines) == 8
assert result['used_s'] < 78.
assert sum(len(line) + 1 for line in lines) <= 4500, 'Batched lab output would be truncated'
