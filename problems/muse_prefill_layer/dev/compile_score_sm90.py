"""Compile the complete score portfolio for SM90a without executing it locally."""
import importlib.util,os,time
from pathlib import Path
from torch.utils import cpp_extension
p=Path('problems/muse_prefill_layer/cand_score_portfolio.py')
s=importlib.util.spec_from_file_location('score_compile',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
# The capability affects only extension construction here, never kernel execution.
m.torch.cuda.get_device_capability=lambda *a,**k:(9,0)
original=cpp_extension.load_inline

def build(**kw):
 kw['name']+='_compile_sm90';kw['verbose']=True
 return original(**kw)
cpp_extension.load_inline=build
start=time.monotonic();m._grouped_module()
print('SM90a complete expert/Lt source compile PASS, seconds',time.monotonic()-start,flush=True)
