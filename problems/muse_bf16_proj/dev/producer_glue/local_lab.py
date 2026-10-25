"""Run the exact paired/timeline payload with the selected S11 setup locally."""
import json
from pathlib import Path
import runpy
import sys
import time

import torch

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO/'problems/muse_decode_layer'))
import reference

weights=reference.generate_weights(5353,device='cuda')
buf=torch.empty(1<<30,device='cuda',dtype=torch.uint8)
mode='hybrid' if '--hybrid' in sys.argv else 'stats' if '--stats' in sys.argv else 'glue'
payload=HERE/f'lab_producer_{mode}.py'
ns=runpy.run_path(str(payload),init_globals={'lab_cold_l2':lambda:buf.view(torch.int64).sum()})
lines=ns['lab'](time.monotonic()+260,weights)
print('\n'.join(lines),flush=True)
g=ns['lab'].__globals__
(HERE/(mode+'_lab_validation.json' if mode!='glue' else 'lab_validation.json')).write_text(json.dumps(dict(lines=lines,profiles=g['PG_RESULTS']),indent=2)+'\n')
assert any(l.startswith('END shapes=8/8 timelines=8 ') for l in lines)
assert not any('GATE=' in l or l.startswith(('FAIL','INCOMPLETE')) for l in lines)
print('PASS payload chars',sum(len(l)+1 for l in lines))
