"""Run the exact packed-bank lab in a fresh build directory under its shared budget."""
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

H=Path(__file__).resolve().parent;R=H.parents[4]
os.environ['TORCH_EXTENSIONS_DIR']=str(H/'cold_cache')
os.environ['CCACHE_DISABLE']='1'
import torch

m=runpy.run_path(str(R/'prof/lab/moe_m256_mix.py'));fn=m['lab'];start=time.monotonic()
lines=fn(start+78.,dict(device=torch.empty(1,device='cuda')));used=time.monotonic()-start
payload=R/'prof/lab/moe_m256_mix.py'
report=dict(sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),budget_s=78.,used_s=used,
            compiler_cache_disabled=True,lines=lines,rows=fn.__globals__['MX_ROWS'],phases=fn.__globals__['MX_PHASES'])
(H/'cold.json').write_text(json.dumps(report,indent=2)+'\n');print('\n'.join(lines),flush=True)
assert used<78. and not any('BUDGET' in line for line in lines)
assert len(report['rows'])==36 and len(report['phases'])==84
assert sum(len(line)+1 for line in lines)<=4500
print('MIX_COLD_PASS',used,sum(len(line)+1 for line in lines),flush=True)
