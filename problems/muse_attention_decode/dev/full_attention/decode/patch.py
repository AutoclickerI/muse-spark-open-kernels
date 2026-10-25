"""A selected full-decode arm on exact S14; generation is ready, promotion waits on paired receipts."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
manifest=json.loads((H/'manifest.json').read_text())['variants']
ap=argparse.ArgumentParser();ap.add_argument('arm',choices=[x['name'] for x in manifest if x['name']!='ref']);a=ap.parse_args()
p=R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py';old=p.read_text();assert hashlib.sha256(old.encode()).hexdigest().startswith('f1c77ce6')
drv=ast.parse((H/(a.arm+'.py')).read_text())
names={n.name for n in drv.body if isinstance(n,ast.FunctionDef)}|{n.targets[0].id for n in drv.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
rename={n:'_dcd_'+n.lstrip('_') for n in names}
class CellNames(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id in rename:n.id=rename[n.id]
  return n
 def visit_FunctionDef(self,n):
  self.generic_visit(n);n.name=rename.get(n.name,n.name);return n
 def visit_Global(self,n):n.names=[rename.get(x,x) for x in n.names];return n
cell=ast.unparse(CellNames().visit(drv))+'\n'
lines=old.splitlines(True);n=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='_best7_attention')
body=''.join(lines[n.lineno-1:n.end_lineno]);mark="    cfg=data['config'];choice=plan['attention']\n";assert body.count(mark)==1
body=body.replace(mark,mark+"    if cfg['q_len']==1 and not cfg['window'] and cfg['batch'] in (16,32):\n        return _dcd_attention(data,qkv=qkv).view(cfg['batch'],Q_SIZE)\n")
lines[n.lineno-1:n.end_lineno]=[body];new=''.join(lines)+'\n\n# Selected full decode16/32 cell; no other attention dispatch changes.\n'+cell
compile(new,'s14_decode.py','exec');out=H/('s14_'+a.arm+'.py');out.write_text(new)
patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+str(p.relative_to(R)),tofile='b/'+str(p.relative_to(R))))
pp=H/('s14_'+a.arm+'.patch');pp.write_text(patch)
proof=dict(arm=a.arm,base_sha256=hashlib.sha256(old.encode()).hexdigest(),candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),promotion=False)
(H/('s14_'+a.arm+'_identity.json')).write_text(json.dumps(proof,indent=2)+'\n');print(proof)
