"""Held-out paired confirmation on the exact frozen conditional candidate."""
import ast
import base64
import hashlib
import lzma
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO/'tools'))
import h200_lab

candidate=(HERE/'d1_trial.py').read_text()
baseline=(REPO/'problems/muse_decode_layer/cand_r3_best7_s13_final.py').read_text()
assert hashlib.sha256(baseline.encode()).hexdigest().startswith('958f1258')
helpers=(HERE/'lab_body_generated.py').read_text()
tree=ast.parse(helpers)
names=('ab_copy','ab_graph','ab_tensors','ab_gate','ab_time','ab_summary')
prefix='import ast,base64,gc,hashlib,json,linecache,lzma,math,statistics,time\nimport torch\n'
prefix+='GP_BASE='+repr(baseline)+'\nGP_CANDIDATE='+repr(candidate)+'\n'
prefix+='\n\n'.join(ast.get_source_segment(helpers,n) for n in tree.body
    if isinstance(n,ast.FunctionDef) and n.name in names)+'\n'
body=prefix+(HERE/'confirm_body.py.inc').read_text();compile(body,'gp_confirm','exec')
tag=hashlib.sha256(body.encode()).hexdigest()
bootstrap='import base64,lzma,linecache,hashlib\n_GP_CONFIRM=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(body.encode(),preset=9)).decode())+')).decode()\n'
bootstrap+='assert hashlib.sha256(_GP_CONFIRM.encode()).hexdigest()=='+repr(tag)+'\n_GP_CONFIRM_FILE="<gp-confirm-'+tag[:12]+'>"\n'
bootstrap+='linecache.cache[_GP_CONFIRM_FILE]=(len(_GP_CONFIRM),None,_GP_CONFIRM.splitlines(True),_GP_CONFIRM_FILE)\nexec(compile(_GP_CONFIRM,_GP_CONFIRM_FILE,"exec"),globals())\n_GP_CONFIRM_LAB=lab\ndef lab(deadline,weights,state):\n    return _GP_CONFIRM_LAB(deadline,weights,state)\n'
payload=HERE/'lab_confirm.py';payload.write_text(bootstrap)
wrapped=candidate.rstrip()+'\n\n'+h200_lab.WRAPPER.format(names='P7 frozen d1 confirmation',base='conditional S13 '+hashlib.sha256(candidate.encode()).hexdigest()[:8],
    payloads=[h200_lab.embed(payload)],budget=300-h200_lab.MARGIN_S,limit=h200_lab.LIMIT)
out=HERE/'confirm_trial.py';out.write_text(wrapped)
print('payload',hashlib.sha256(bootstrap.encode()).hexdigest(),'body',tag)
print(out,len(wrapped),hashlib.sha256(wrapped.encode()).hexdigest())
