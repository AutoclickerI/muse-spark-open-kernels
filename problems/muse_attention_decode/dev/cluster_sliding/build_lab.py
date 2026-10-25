"""Freeze S11's original timeline and the single-launch sliding variant for a paired layer lab."""
import ast
import hashlib
import re
from pathlib import Path
import build
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
TL=ROOT/'problems/muse_decode_layer/dev/timeline'
s11=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text()
assert hashlib.sha256(s11.encode()).hexdigest().startswith('acda6c62')
cuda=build.source()
cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
src=(HERE/'driver.py').read_text(); tree=ast.parse(src);lines=src.splitlines(True)
body='\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in tree.body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('_MOD','_PLANS','_FREQ') for t in n.targets)))
for name in ('CUDA','CPP','_MOD','_PLANS','_FREQ','module','plan','attention'):
    body=re.sub(r'\b'+name+r'\b','cs_'+name,body)
payload=('import gc, hashlib, json, linecache, math, statistics, time, types, lzma, base64\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
    +'TL_S11='+repr(s11)+'\nTL_ACTIVITY='+repr((TL/'activity.cpp').read_text())+'\ncs_CUDA='+repr(cuda)+'\ncs_CPP='+repr(cpp)+'\n'
    +body+'\n'+(TL/'selected_prepare.py').read_text()+'\n'+(TL/'lab_body.py').read_text().split('def lab(deadline, weights, state=None):')[0]+'\n'+(HERE/'lab.py.inc').read_text())
compile(payload,'cluster_sliding_timeline.py','exec')
p=ROOT/'prof/lab/cluster_sliding_timeline.py';p.write_text(payload)
print(p,hashlib.sha256(payload.encode()).hexdigest())
