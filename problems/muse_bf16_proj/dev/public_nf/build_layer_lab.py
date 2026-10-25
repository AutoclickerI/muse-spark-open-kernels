"""Package the reproducible S15 candidate and independent public-distribution fixtures."""
import ast,base64,lzma,zlib
from pathlib import Path
H=Path(__file__).resolve().parent
# Existing same-input paired helpers, extracted as source without executing a payload.
p=H.parents[2]/'muse_decode_layer/dev/s13/s13finalab_payload.py'
t=ast.parse(p.read_text());n=next(n for n in t.body if isinstance(n,ast.Assign) and any(getattr(x,'id','')=='_s13_body' for x in n.targets))
s=eval(compile(ast.Expression(n.value),'<read own lab helpers>','eval'),{'lzma':lzma,'base64':base64})
t=ast.parse(s);wanted={'ab_label','ab_copy','ab_graph','ab_tensors','ab_gate','ab_time','AB_SEEDS','AB_CASES','AB_LOADS'}
nodes=[]
for n in t.body:
 name=n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
 if name in wanted:nodes.append(ast.unparse(n))
body='import torch,math,statistics,time\n'+'\n\n'.join(nodes)+'\n'+(H/'layer_weights.py.inc').read_text()+'\n'+(H/'layer_lab.py.inc').read_text()
candidate=(H/'cand_nf_s15.py').read_text()
packed=base64.b64encode(zlib.compress(candidate.encode(),9)).decode()
s='import base64,zlib\nLAYER_SOURCE=zlib.decompress(base64.b64decode('+repr(packed)+')).decode()\n'+body
(H/'s15_layer_lab.py').write_text(s)
print('whole-layer payload',len(s))
