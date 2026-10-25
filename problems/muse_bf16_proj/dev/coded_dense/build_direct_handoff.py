"""Package a645a795's direct BF16 qkv3 (and independently removable O) rows onto S13."""
import ast
import runpy
from pathlib import Path
H=Path(__file__).resolve().parent
runpy.run_path(str(H/'build_direct.py'))
src=(H/'direct_body.py').read_text().split('\n"""Lossless register-A WGMMA experiment;',1)[0]
tree=ast.parse(src)
names={}
for n in tree.body:
 if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith(('cd_','_cd_')):
  names[n.name]=n.name.replace('cd_','dcp_',1)
 if isinstance(n,ast.Assign):
  for t in n.targets:
   if isinstance(t,ast.Name) and t.id.startswith(('CD_','_CD_')):names[t.id]=t.id.replace('CD_','DCP_',1)
class Rename(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id in names:n.id=names[n.id]
  return n
 def visit_FunctionDef(self,n):
  self.generic_visit(n)
  if n.name in names:n.name=names[n.name]
  return n
body=ast.unparse(Rename().visit(tree))+'\n'
# Only T=1 in these handoffs. Decode is identical to the H200-tested function.
old='DCP_INSTANCES = tuple(((nt, rt, pipe, v) for v in (3, 4) for nt in (1, 2, 4) for rt in (1, 2) for pipe in (2, 4)))'
assert body.count(old)==1
body=body.replace(old,'DCP_INSTANCES = ((1, 1, 2, 3),)')
scaffold='''
_DCP_ROWS = __ROWS__

def _dcp_prepare(state):
    state['direct_coded']={}
    for t,name in _DCP_ROWS:
        if name=='qkv3':state[name]=state['qkv_gate'][:QKV3_SIZE]
        w=state[name]
        c=dcp_encode(w,kmajor=True)
        if not torch.equal(dcp_decode(c).view(torch.int16),w.view(torch.int16)):
            raise RuntimeError(f'direct {name}: bit-exact decode failed')
        state['direct_coded'][name]=c

def _dcp_force(state):
    for t,name in _DCP_ROWS:state['proj_plan'][(t,name)]='custom'

def _dcp_project(x,name,state):
    NW,S=_DCP_ROWS[(x.shape[0],name)]
    return dcp_gemv(x,state['direct_coded'][name],S,NW=NW,RT=1,PIPE=2,prew=int(name!='o'),pdl=PDL,V=3)
'''
template='''"""Direct BF16 coded row handoff from H200 a645a795307b4c77; additive onto S13/S12.
Default qkv3 T1: 7.55 vs 8.09 us, paired 0.9346. O is a separate optional layer ablation.
"""
import ast
import runpy
import re
import sys
import hashlib
from pathlib import Path
BODY = __BODY__
SCAFFOLD = __SCAFFOLD__
ROWS = __ROWS__

def patch(src):
    if '_DCP_ROWS = ' in src:
        match=re.search(r'^_DCP_ROWS = (.+)$',src,re.M)
        rows=ast.literal_eval(match.group(1));assert not (set(rows)&set(ROWS))
        rows.update(ROWS);src=src[:match.start(1)]+repr(rows)+src[match.end(1):]
    else:
        marker='# Projection portfolio:';assert src.count(marker)==1
        src=src.replace(marker,BODY+'\\n'+SCAFFOLD.replace('__ROWS__',repr(ROWS))+'\\n'+marker,1)
        old='def _project_variant(x, w, name, state, variant, out_dtype=torch.bfloat16):\\n';assert src.count(old)==1
        src=src.replace(old,old+"    if (x.shape[0],name) in _DCP_ROWS:\\n        return _dcp_project(x,name,state)\\n",1)
        match=re.search(r'with ThreadPoolExecutor\\(max_workers=(\\d+)\\) as native_pool:',src);assert match
        src=src[:match.start()]+f'with ThreadPoolExecutor(max_workers={int(match.group(1))+1}) as native_pool:\\n        dcp_future=native_pool.submit(dcp_module)'+src[match.end():]
        old='        state=_prepare_weights(weights)\\n';assert src.count(old)==1
        src=src.replace(old,old+'        dcp_future.result();_dcp_prepare(state)\\n',1)
        old='    _gate_split_prepare(state,device)\\n';assert src.count(old)==1
        src=src.replace(old,old+'    _dcp_force(state)\\n',1)
    if (1,'o') in ROWS:
        assert src.count("{'o': ('stream', 6, 1)}")==2
        src=src.replace("{'o': ('stream', 6, 1)}",'{}')
    compile(src,'<direct row>','exec')
    return src

if __name__=='__main__':
    base,out=map(Path,sys.argv[1:3]);out.write_text(patch(base.read_text()));print(out,hashlib.sha256(out.read_bytes()).hexdigest())
'''
for name,rows in [('qkv3',{(1,'qkv3'):(4,32)}),('o',{(1,'o'):(8,8)})]:
 text=template.replace('\nROWS = __ROWS__\n','\nROWS = '+repr(rows)+'\n').replace('__BODY__',repr(body)).replace('__SCAFFOLD__',repr(scaffold))
 (H/f'patch_direct_{name}.py').write_text(text)
print('Built independent qkv3 and O handoffs; O admission still needs S11 stream comparison')
