"""Screen the exact self-contained whole-layer payload on the local GPU."""
import json
import runpy
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / 'problems/muse_decode_layer'))
import reference

weights = reference.generate_weights(5353, device='cuda')
buf = torch.empty(1 << 30, device='cuda', dtype=torch.uint8)
payload = Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'lab_verify_layer.py'
ns = runpy.run_path(str(payload), init_globals={'lab_cold_l2': lambda: buf.view(torch.int64).sum()})
lines = ns['lab'](time.monotonic() + 260, weights)
print('\n'.join(lines), flush=True)
(payload.parent / ('issue_layer_validation.json' if 'GI_CASES' in ns else 'layer_validation.json')).write_text(json.dumps(dict(lines=lines), indent=2) + '\n')
count=len(ns.get('GI_CASES',ns.get('VE_CASES',())))
assert any(line.startswith(f'END shapes={count}/{count} ') for line in lines), lines[-5:]
assert not any('FAIL' in line or 'INCOMPLETE' in line for line in lines), [line for line in lines if 'FAIL' in line or 'INCOMPLETE' in line]
