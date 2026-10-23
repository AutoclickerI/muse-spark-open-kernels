"""Check diagnostic isolation, failures, expiry, and the frozen scored prefix."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

import torch

ROOT=Path(__file__).resolve().parents[3]
path=ROOT/'problems/muse_bf16_proj/cand_dflash_diag.py'
base=(ROOT/'problems/muse_bf16_proj/cand_fused_select.py').read_text()
source=path.read_text();assert source.startswith(base)
assert hashlib.sha256(base.encode()).hexdigest()=='10d02fa32b572b1649f5e0c63e5f9841567835dc3b83ffe17cd7d843429117e9'
assert not any(isinstance(n,ast.ImportFrom) and n.module and 'muse_proj' in n.module for n in ast.parse(source).body)
spec=importlib.util.spec_from_file_location('dflash_diag_test',path)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
old=c._df_run_diagnostics;sentinel=object()
c._df_scored_prepare=lambda weights:sentinel
def bad(*args):raise TypeError('injected diagnostic failure')
c._df_run_diagnostics=bad
out=io.StringIO()
with contextlib.redirect_stdout(out):result=c.prepare({})
assert result is sentinel and 'DFB_REJECT diagnostic:' in out.getvalue()
c._df_run_diagnostics=old
lines=[];c._df_run_diagnostics(time.monotonic()-1,lines)
assert lines and all('SKIP' in line for line in lines)
x=torch.randn((8,),device='cuda')
flush=torch.zeros(128<<20,device='cuda',dtype=torch.uint8)
try:
    c._df_diag_pair(lambda:x*2,lambda:x*3,lambda:x.normal_(),flush,time.monotonic()+10)
except ValueError:pass
else:raise AssertionError('incorrect diagnostic variant was not rejected')
assert c._df_proj._Graph.active_streams==0
print(json.dumps(dict(ok=True,scored_prefix_sha256=hashlib.sha256(base.encode()).hexdigest(),
                     candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
                     injected_failure_preserved_scored_state=True,expired_cases=len(lines),
                     numerical_failure_rejected=True,active_tuning_streams=0)),flush=True)
