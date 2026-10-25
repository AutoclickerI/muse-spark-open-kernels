"""Batch layer admission and geometry/decoder probes, sharing the exact embedded code."""
import ast
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'s15_layer_lab.py').read_text();pool=(H/'pool_q16_lab.py').read_text()
lines=pool.splitlines(keepends=True);skip=set()
for n in ast.parse(pool).body:
 if isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('PORT_SOURCE','S13_SOURCE') for t in n.targets):skip.update(range(n.lineno-1,n.end_lineno))
pool=''.join(l for i,l in enumerate(lines) if i not in skip)
base=(H.parents[2]/'muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
s+='\nNF_BASE_CHARS='+str(len(base))+'\nNF_TUNE_SOURCE='+repr(pool)+'\n'
s+='''
_s15_layer_lab=lab

def lab(deadline,weights,state=None):
    lines=_s15_layer_lab(deadline-100,weights,state)
    if time.monotonic()>deadline-40:return lines+['SKIP pool/q16: insufficient remaining lab budget']
    import linecache
    file='<nf-pool-q16-batch>'
    linecache.cache[file]=(len(NF_TUNE_SOURCE),None,NF_TUNE_SOURCE.splitlines(True),file)
    ns=dict(globals())
    ns['S13_SOURCE']=LAYER_SOURCE[:NF_BASE_CHARS]
    start=LAYER_SOURCE.index('"""Layer adapter for Nathen Fernandes')
    end=LAYER_SOURCE.index('# Selected by paired H200 producer+consumer')
    ns['PORT_SOURCE']=LAYER_SOURCE[start:end]
    exec(compile(NF_TUNE_SOURCE,file,'exec'),ns)
    return lines+ns['lab'](deadline,weights)
'''
compile(s,'<S15 batch>','exec');(H/'s15_batch_lab.py').write_text(s);print('batch bytes',len(s))
