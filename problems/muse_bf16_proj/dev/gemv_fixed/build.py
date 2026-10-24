"""Freeze best7_scale, its recorded dense choices, and instrumented own kernels."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
source=ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py';base=source.read_text()
receipt=ROOT/'problems/muse_decode_layer/profiles/cand_r3_best6.1b80ffc037cc454c.log'
choices={}
for line in receipt.read_text().splitlines():
    if not line.startswith('BEST_SHAPE '):continue
    b,q,_=map(int,re.search(r'b=(\d+) q=(\d+) window=(\d+)',line).groups())
    if b*q>32:continue
    for role,choice in ast.literal_eval(line.split(' proj=',1)[1].split(' prefix=',1)[0]).items():
        key=(b*q,role)
        if key in choices:assert choices[key]==choice
        choices[key]=choice
assert len(choices)==16

tree=ast.parse(base);traces=[]
for name in ('_gemv_kernel','_boundary_fast','_route_fast','_qk_prep_kernel'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    text=ast.get_source_segment(base,node)
    # This is a separate source copy compiled before execution; no loaded module
    # is edited. Arithmetic, geometry and PDL remain the baseline's own.
    text=text.replace('def '+name+'(', 'def gd'+name+'(STAMPS, ')
    text=text.replace('_pdl_start','gd_start').replace('_pdl_wait','gd_wait').replace('_l2_prefetch','gd_prefetch')
    text=text.replace('    gd_start(USE_PDL)','    gd_mark(STAMPS,0)\n    gd_start(USE_PDL)',1)
    text=text.replace('    gd_wait(USE_PDL)','    gd_mark(STAMPS,2)\n    gd_wait(USE_PDL)\n    gd_mark(STAMPS,3)',1)
    if name=='_gemv_kernel':
        text=text.replace('    for k in range(k0, k0 + KS, BK):','    gd_mark(STAMPS,1)\n    for k in range(k0, k0 + KS, BK):',1)
        text=text.replace('    tl.store(P +','    gd_mark(STAMPS,4)\n    gd_mark(STAMPS,5)\n    tl.store(P +',1)
    elif name=='_boundary_fast':
        text=text.replace('    for s in tl.static_range(S):','    gd_mark(STAMPS,1)\n    for s in tl.static_range(S):',1)
        text=text.replace('    if HAS_POSTW:br*=postw','    gd_mark(STAMPS,4)\n    if HAS_POSTW:br*=postw',1)
        text=text.replace('    tl.store(SOUT','    gd_mark(STAMPS,5)\n    tl.store(SOUT',1)
    elif name=='_route_fast':
        text=text.replace('    for s in tl.static_range(S):','    gd_mark(STAMPS,1)\n    for s in tl.static_range(S):',1)
        text=text.replace('    norm=p*','    gd_mark(STAMPS,4)\n    norm=p*',1)
        text=text.replace('    tl.store(X','    gd_mark(STAMPS,5)\n    tl.store(X',1)
    else:
        text=text.replace('    for s in tl.static_range(S):','    gd_mark(STAMPS,1)\n    for s in tl.static_range(S):',1)
        text=text.replace('    tl.store(GATE','    gd_mark(STAMPS,4)\n    gd_mark(STAMPS,5)\n    tl.store(GATE',1)
    text+='\n    tl.debug_barrier()\n    gd_mark(STAMPS,6)\n'
    for phase in range(7):assert 'gd_mark(STAMPS,'+str(phase)+')' in text,(name,phase)
    for phase in range(7):text=text.replace('gd_mark(STAMPS,'+str(phase)+')',f'gc{phase},gg{phase},gs{phase}=gd_read()')
    text+='    gd_save(STAMPS,'+','.join(f'gc{i},gg{i}' for i in range(7))+',gs6)\n'
    traces.append('@triton.jit\n'+text)

runtime='GF_CUDA='+repr((HERE/'kernel.cu').read_text())+'\n'+(HERE/'runtime.py.inc').read_text()
(HERE/'projection.py').write_text('"""Unsplit register-MMA dense projection cell."""\n'+runtime)
body='GF_BASE_SOURCE='+repr(base)+'\nGF_PROJ='+repr(choices)+'\n'+runtime+'\n'
body+=(HERE/'diagnostic.py.inc').read_text()+'\n'+'\n\n'.join(traces)+'\n'+(HERE/'lab.py.inc').read_text()
sys.path.insert(0,str(ROOT/'tools'));from okc_run import rejected_words
assert not rejected_words(body),rejected_words(body)
out=ROOT/'prof/lab/dense_gemv_fixed.py';out.write_text(body)
manifest=dict(payload=str(out.relative_to(ROOT)),payload_sha256=hashlib.sha256(body.encode()).hexdigest(),
 baseline=str(source.relative_to(ROOT)),baseline_sha256=hashlib.sha256(base.encode()).hexdigest(),
 kernel_sha256=hashlib.sha256((HERE/'kernel.cu').read_bytes()).hexdigest(),
 projection_sha256=hashlib.sha256((HERE/'projection.py').read_bytes()).hexdigest(),receipt=receipt.name,
 choices={f'{t}:{r}':c for (t,r),c in choices.items()})
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out,manifest['payload_sha256'])
