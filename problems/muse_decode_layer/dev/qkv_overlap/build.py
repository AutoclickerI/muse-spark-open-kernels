from pathlib import Path
import ast,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
source=(ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py').read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_best7_run')
fn='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
fn=fn.replace('def _best7_run(', 'def _kg_best7_run(')
old="    qkv=_best7_project(data['attn_input'],'qkv_gate',state,plan)\n    if prefetch is not None:prefetch.begin('front')\n    attn=_best7_attention(qkv,data,state,plan)"
new="    attn=state['kg_ops'][data['config']['batch'],data['config']['window']](data['attn_input'],data)"
assert old in fn;fn=fn.replace(old,new)
cell=(HERE/'runtime.py').read_text()+'\n'+fn+'\n'+(HERE/'dropin.py.inc').read_text()
(HERE/'best7_kv_group_cell.py').write_text(cell)
(HERE/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),cell_sha256=hashlib.sha256(cell.encode()).hexdigest(),enabled=[]),indent=2)+'\n')
print(hashlib.sha256(cell.encode()).hexdigest())
