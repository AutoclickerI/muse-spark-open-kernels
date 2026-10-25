"""One independent reset-queued decode payload, ten arms and eight seeds per full shape."""
import ast
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
reset=R/'problems/muse_bf16_proj/lab/lab_softmax_mix_lab.py'
assert hashlib.sha256(reset.read_bytes()).hexdigest().startswith('03086413')
p=next(ast.literal_eval(n.value)[0][2] for n in ast.parse(reset.read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_LAB_PAYLOADS')
# Recover only the lab's own public synthetic inputs/reference arithmetic; no judge import.
prefix=p[:p.index('def check_probe(')]
tree=ast.parse(prefix);lines=prefix.splitlines(True)
for n in sorted([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('timed','check_implementation')],key=lambda n:n.lineno,reverse=True):lines[n.lineno-1:n.end_lineno]=[]
prefix=''.join(lines)
assert prefix.count('CACHE_CHUNK = 1 << 20')==1
prefix=prefix.replace('CACHE_CHUNK = 1 << 20','CACHE_CHUNK = 1 << 13')
check=(H/'check.py').read_text();node=next(n for n in ast.parse(check).body if isinstance(n,ast.FunctionDef) and n.name=='probe');helper=''.join(check.splitlines(True)[node.lineno-1:node.end_lineno])
manifest=json.loads((H/'manifest.json').read_text());sources={r['name']:(H/(r['name']+'.py')).read_text() for r in manifest['variants']}
s=prefix+'\n'+helper+'\nSOURCES='+repr(sources)+'\n'+(H/'lab.py.inc').read_text();compile(s,'decode_softmax_lab.py','exec');(H/'decode_softmax_lab.py').write_text(s)
(H/'inputs.py').write_text(prefix)
print('Decode payload',len(s.encode()),hashlib.sha256(s.encode()).hexdigest())
