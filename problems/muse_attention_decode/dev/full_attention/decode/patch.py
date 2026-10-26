"""Emit only selected full-decode cells onto exact S15 (whole-layer admission pending)."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
manifest=json.loads((H/'manifest.json').read_text())['variants']
ap=argparse.ArgumentParser();ap.add_argument('arm',choices=[x['name'] for x in manifest if x['name']!='ref']+['gather'])
ap.add_argument('--batches',type=int,nargs='+',default=[16,32],choices=[16,32]);a=ap.parse_args()
p=R/'problems/muse_decode_layer/cand_r3_best7_s15.py';old=p.read_text();assert hashlib.sha256(old.encode()).hexdigest().startswith('60254883')
driver=(H.parent/'gather/driver.py') if a.arm=='gather' else H/(a.arm+'.py')
drv=ast.parse(driver.read_text());four_warp=any(isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CUDA' for n in drv.body)
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
branch=f"    if cfg['q_len']==1 and not cfg['window'] and cfg['batch'] in {tuple(a.batches)!r}:\n"
if four_warp:
 # The attention-board driver reads data['qkv'] in plan_for(). The decode-layer
 # ABI supplies projection partials separately. Preserve the original owned PDL
 # sum/cast and give the driver its actual raw BF16 operand for plan/allocation.
 branch+="        raw=_attn_port_raw(qkv,data)\n        return _dcd_attention(dict(data,qkv=raw)).view(cfg['batch'],Q_SIZE)\n"
else:branch+="        return _dcd_attention(dict(data,qkv=qkv[0]),qkv=qkv).view(cfg['batch'],Q_SIZE)\n"
body=body.replace(mark,mark+branch)
lines[n.lineno-1:n.end_lineno]=[body];new=''.join(lines)+'\n\n# Selected full-decode cell; all other S15 dispatch is retained.\n'+cell
compile(new,'s15_decode.py','exec');outdir=H/'out';outdir.mkdir(exist_ok=True)
tag=a.arm+'_'+'_'.join(map(str,a.batches));out=outdir/('s15_'+tag+'.py');out.write_text(new)
patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+str(p.relative_to(R)),tofile='b/'+str(p.relative_to(R))))
pp=outdir/('s15_'+tag+'.patch');pp.write_text(patch)
proof=dict(arm=a.arm,batches=a.batches,base_sha256=hashlib.sha256(old.encode()).hexdigest(),candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),promotion=False)
(outdir/('s15_'+tag+'_identity.json')).write_text(json.dumps(proof,indent=2)+'\n');print(proof)
