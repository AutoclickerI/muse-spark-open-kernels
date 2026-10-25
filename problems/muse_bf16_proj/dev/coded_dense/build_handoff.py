"""Freeze measured c446fdab CP and 04d89c7d WGMMA into one S12 -> S13 integration patch."""
import ast
import hashlib
import runpy
from pathlib import Path
H=Path(__file__).resolve().parent
runpy.run_path(str(H/'build_cpall.py'))
runpy.run_path(str(H/'build_wg.py'))
# SOURCES in this exact screened/sent payload pins the measured cp.async kernel.
tree=ast.parse((H/'cpall_lab.py').read_text())
cp=next(ast.literal_eval(n.value)['cpall'] for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SOURCES')
assert hashlib.sha256(cp.encode()).hexdigest() == '1c1efc81eac5cad536358025f40389019449cac518ac2c56e9d3b02ef2bd3916'
wg=(H/'wg_frozen.py').read_text().replace('from coded import CD_CUDA, cd_encode, cd_decode\n','')
body=cp+'\n'+wg
rows={(1,'qkv_gate'):('cp',8,22,1,True),(1,'pool_router'):('cp',8,11,1,True),
      (16,'qkv_gate'):('wg',8,16,2,True),(16,'unpooler'):('wg',8,4,2,True),(32,'unpooler'):('wg',8,4,2,True)}
scaffold=r'''
CD_ROWS = __ROWS__  # (family,NW,S,WB,kmajor). Only H200 stage winners; layer admission belongs to integration.

def _cd_compile():
    cd_module()
    if any(cfg[0]=='wg' for cfg in CD_ROWS.values()):wg_module()

def _cd_prepare(state):
    state['coded_dense']={}
    for (t,name),(family,NW,S,WB,km) in CD_ROWS.items():
        key=(family,name,km)
        if key in state['coded_dense']:continue
        w=state[name]
        c=(wg_encode if family=='wg' else cd_encode)(w,kmajor=km)
        decoded=(wg_decode if family=='wg' else cd_decode)(c)
        if not torch.equal(decoded.view(torch.int16),w.view(torch.int16)):
            raise RuntimeError(f'coded {key}: bit-exact decode failed')
        state['coded_dense'][key]=c

def _cd_force(state):
    for t,name in CD_ROWS:state['proj_plan'][(t,name)]='custom'

def _cd_project(x,name,state):
    family,NW,S,WB,km=CD_ROWS[(x.shape[0],name)]
    c=state['coded_dense'][(family,name,km)]
    if family=='cp':return cd_gemv(x,c,S,NW=NW,RT=1,PIPE=2,prew=int(name!='o'),pdl=PDL,V=3,C=1)
    return wg_gemv(x,c,S,NW=NW,WB=WB,prew=int(name!='o'),pdl=PDL)
'''
template=r'''"""S13 dense coding handoff. Applies onto S12's existing coded block (screened base a92b6619).
Measured kernels: CP c446fdab439b44e5, WG 04d89c7dc7344c08. No whole-layer promotion claimed.
Use patch(src, rows=...) for independent ablations. Default rows preserve S12's T8 exclusion.
"""
import hashlib
import re
import sys
from pathlib import Path
BODY = __BODY__
ROWS = __ROWS__
SCAFFOLD = __SCAFFOLD__

def patch(src,rows=None):
    rows=ROWS if rows is None else rows
    start=src.index('\n\n# ---- P10 coded dense GEMVs (')
    end=src.index('# Projection portfolio:',start)
    instances=tuple(sorted({(1 if t<=8 else 2 if t<=16 else 4,1,2,3) for (t,_),v in rows.items() if v[0]=='cp'}))
    body=BODY
    old='CD_INSTANCES = tuple((nt, rt, pipe, v) for v in (3, 4) for nt in (1, 2, 4) for rt in (1, 2) for pipe in (2, 4))'
    assert body.count(old)==1
    body=body.replace(old,f'CD_INSTANCES = {instances!r}')
    block='\n\n# ---- P10 coded dense GEMVs (S13 measured CP + register-A WGMMA) ----\n'+body+'\n'+SCAFFOLD.replace('__ROWS__',repr(rows))+'\n'
    src=src[:start]+block+src[end:]
    start=src.index('    cfg = CD_ROWS.get((x.shape[0], name))\n',src.index('def _project_variant('))
    end=src.index("    if isinstance(variant,tuple) and variant[0]=='vdsk':",start)
    src=src[:start]+"    if (x.shape[0], name) in CD_ROWS:\n        return _cd_project(x,name,state)\n"+src[end:]
    assert src.count('cd_future=native_pool.submit(cd_module)')==1
    src=src.replace('cd_future=native_pool.submit(cd_module)','cd_future=native_pool.submit(_cd_compile)')
    if (1,'o') in rows:
        assert src.count("{'o': ('stream', 6, 1)}")==2
        src=src.replace("{'o': ('stream', 6, 1)}",'{}')
    compile(src,'<S13 dense>','exec')
    return src

if __name__=='__main__':
    base,out=map(Path,sys.argv[1:3]);out.write_text(patch(base.read_text()));print(out,hashlib.sha256(out.read_bytes()).hexdigest())
'''
scaffold=scaffold.replace('__ROWS__','__'+'ROWS'+'__')
template=template.replace('__BODY__',repr(body)).replace('__SCAFFOLD__',repr(scaffold)).replace('\nROWS = __ROWS__\n','\nROWS = '+repr(rows)+'\n')
(H/'patch_coded_s13.py').write_text(template)
print(H/'patch_coded_s13.py')
