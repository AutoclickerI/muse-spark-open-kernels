"""Freeze the held-out decode-1 full preload win onto S13 final."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
rows=tuple((1,1,0,role) for role in ('b1','route','b2'))
source=(HERE/'patch.py').read_text();tree=ast.parse(source)
functions='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body
    if isinstance(n,ast.FunctionDef) and n.name in ('cuda_source','patch'))
functions=functions.replace("(HERE/'runtime.py.inc').read_text()",'RUNTIME')
functions=functions.replace('def patch(src,rows=()):','def patch(src):\n    rows='+repr(rows))
base=(REPO/'problems/muse_decode_layer/cand_r3_best7_s13_final.py').read_text()
base_sha=hashlib.sha256(base.encode()).hexdigest()
frozen='"""H200 0110a781 + 1a880faf: decode-1 full glue preload win; S13 final only."""\nimport ast,hashlib\nBASE_SHA='+repr(base_sha)+'\nRUNTIME='+repr((HERE/'runtime.py.inc').read_text())+'\n\n'+functions+'\n'
path=HERE/'patch_d1.py';path.write_text(frozen)
ns={};exec(compile(frozen,str(path),'exec'),ns);out=ns['patch'](base)
candidate=HERE/'d1_trial.py';candidate.write_text(out)
confirmation=json.loads((HERE/'h200_1a880faf747d4c1b.json').read_text())
by_case={r['case']:r for r in confirmation['cells']}
full=by_case['d1F']['paired']['B'];sliding=by_case['d1S']['paired']['B']
derived_load=(46*sliding[3]+16*full[3]*full[0])/(46*sliding[3]+16*full[3])
manifest=dict(base='problems/muse_decode_layer/cand_r3_best7_s13_final.py',base_sha256=base_sha,
    patch=str(path.relative_to(REPO)),patch_sha256=hashlib.sha256(frozen.encode()).hexdigest(),
    candidate=str(candidate.relative_to(REPO)),candidate_sha256=hashlib.sha256(out.encode()).hexdigest(),
    rows=rows,receipts=['0110a781a74e4261','1a880faf747d4c1b'],paired_full_ratio=full[0],paired_full_se=full[1],
    full_control_a2_ratio=by_case['d1F']['paired']['A2'][0],
    derived_decode1_full_only_ratio=derived_load,derived_six_load_speedup=derived_load**(-1/6),
    status='paired full-shape win; full-only candidate local validation pending; not promoted')
local={}
for mode in ('normal','forced'):
    file=HERE/f'full_only_{mode}.json'
    if file.exists():
        check=json.loads(file.read_text())
        if check['sha256']==manifest['candidate_sha256']:
            local[mode]=dict(sha256=check['sha256'],tests=len(check['tests']),shapes=len(check['shapes']),
                static=len(check['static']),ok=all(v['ok'] for key in ('tests','shapes','static') for v in check[key]))
manifest['local']=local
if len(local)==2 and all(v['ok'] for v in local.values()):
    manifest['status']='paired full-shape win and full-only local candidate verified; not promoted'
manifest['remaining_verification']='fresh full-only six-load H200 check or integration composition; prepare-host quota exhausted at last observation'
(HERE/'handoff.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
