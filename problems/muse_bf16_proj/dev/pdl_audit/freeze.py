"""Freeze the exact H200 metadata-only winner as a standalone S14 source patch."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
src=(HERE/'patch.py').read_text();tree=ast.parse(src)
definitions=[]
for n in tree.body:
    if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ('BASE_SHA','CUDA_HELPERS','TRITON_HELPERS'):
        definitions.append(ast.get_source_segment(src,n))
    elif isinstance(n,ast.FunctionDef) and n.name in ('once','glue_source','native','triton_source','patch'):
        code=ast.get_source_segment(src,n)
        if n.name=='triton_source':
            a=code.index("    if name in ('_gemv_kernel','_gb_kernel','_gi_kernel'):")
            b=code.index("    if name=='_attn_split_kernel':",a)
            code=code[:a]+code[b:]
        definitions.append(code)
frozen='"""S14 metadata preload patch; paired integration verification required."""\nimport ast,hashlib\n\n'+'\n\n'.join(definitions)+'\n'
p=HERE/'patch_metadata_s14.py';p.write_text(frozen);ns={};exec(compile(frozen,str(p),'exec'),ns)
base=(REPO/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text();candidate=ns['patch'](base)
assert candidate==(HERE/'meta_trial.py').read_text(),'freeze differs from the measured candidate'
sha=hashlib.sha256(candidate.encode()).hexdigest()
job=sys.argv[1] if len(sys.argv)>1 else '0f74021a2c53445e'
report=json.loads((HERE/('h200_'+job+'.json')).read_text());assert report['all_gates_clean_16_seeds']
assert report['variants']['M']==sha
checks={}
for mode in ('normal','forced'):
    d=json.loads((HERE/('meta_'+mode+'.json')).read_text());assert d['sha256']==sha
    checks[mode]={k:len(d[k]) for k in ('tests','shapes','static')};assert all(v['ok'] for k in checks[mode] for v in d[k])
# Verify that the promoted source retains every projection kernel/function exactly.
def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
old,new=functions(base),functions(candidate)
projection_names=('_gemv_kernel','_gb_kernel','_gi_kernel','_gate_split_kernel','_vdsk_gemm','gemv','_gb_gemv','_best7_project')
assert all(old[n]==new[n] for n in projection_names)
manifest=dict(base='problems/muse_decode_layer/cand_r3_best7_s14_final.py',base_sha256=ns['BASE_SHA'],
    patch=str(p.relative_to(REPO)),patch_sha256=hashlib.sha256(frozen.encode()).hexdigest(),
    candidate='problems/muse_bf16_proj/dev/pdl_audit/meta_trial.py',candidate_sha256=sha,
    h200_job=job,arm='M',six_load_speedup=report['score_statistics']['M'],
    control_speedup=report['score_statistics']['A2'],loads=report['loads'],local=checks,
    projection_functions_unchanged=list(projection_names),
    status='provisional paired gain; shared-JIT confirmation quota-blocked; not applied to incumbent',
    limitation='unchanged JIT kernels used separate device handles; small gain with verify32 regression; larger glue target remains unmet',
    confirmation=dict(payload='problems/muse_bf16_proj/dev/pdl_audit/lab_pdl_shared_followup.py',
        payload_sha256=hashlib.sha256((HERE/'lab_pdl_shared_followup.py').read_bytes()).hexdigest(),
        status='not accepted: no H200 quota; own reset-wait sender cancelled',next_quota='2026-10-26 09:00 KST'),
    legacy_patch='problems/muse_bf16_proj/dev/pdl_audit/patch_d1_s14.py',legacy_patch_sha256=hashlib.sha256((HERE/'patch_d1_s14.py').read_bytes()).hexdigest())
(HERE/'handoff.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
