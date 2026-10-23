from pathlib import Path
import importlib.util,time
from torch.utils import cpp_extension
p=Path('problems/muse_prefill_layer/cand_dense_fusion_portfolio.py');s=importlib.util.spec_from_file_location('mixed_compile90',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.torch.cuda.get_device_capability=lambda *a,**k:(9,0)
original=cpp_extension.load_inline
def build(**kw):
 kw['name']+='_compile_sm90';kw['verbose']=True
 return original(**kw)
cpp_extension.load_inline=build
start=time.monotonic();m._dense_tma_module()
print('SM90a mixed-output TMA WGMMA compile PASS',time.monotonic()-start,flush=True)
