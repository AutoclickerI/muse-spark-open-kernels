"""Publish exact measured arms and integration limits to P4."""
import ast,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
manifest=json.loads((H/'manifest.json').read_text());result=json.loads((H/'result.json').read_text())
controls=('A2','G','LATE','KV_LATE')
nominees={a:r for a,r in result['arms'].items() if a in manifest['arms'] and a not in controls and r.get('G0')}
base=(H/'out/base.py').read_text();patches={}
def literal(source,key):
    return next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==key)
for arm,key in (('H2','P3B_CUDA'),('B1','GLUE_CL_CUDA')):
    source=(H/(arm+'_trial.py')).read_text();before=literal(base,key);after=literal(source,key)
    filename=arm+'_'+key+'.patch';(H/filename).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+key,tofile='b/'+key,n=0)))
    patches[arm]=dict(cuda_patch=filename,base_cuda_sha256=hashlib.sha256(before.encode()).hexdigest(),changed_cuda_sha256=hashlib.sha256(after.encode()).hexdigest())
source=(H/'B1_trial.py').read_text();names=('_boundary_fast','_boundary_late')
get=lambda text:'\n\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names)
filename='B1_TRITON.patch';(H/filename).write_text(''.join(difflib.unified_diff(get(base).splitlines(True),get(source).splitlines(True),fromfile='a/boundary_jit',tofile='b/boundary_jit',n=0)))
patches['B1'].update(triton_patch=filename,triton_helper='glue_triton.py.inc',runtime_dispatch='runtime.py.inc::_best7_boundary, EW_BOUNDARIES=(b1,) only; retain rt_workspace path')
handoff=dict(status='complete',base='S18',base_expanded_sha256=manifest['base_sha256'],receipt=result['receipt'],
    result='problems/muse_bf16_proj/dev/stream_continuity/extend/result.json',nominees=nominees,
    arm_sources={a:manifest['arms'][a] for a in nominees},all_boundaries=result['arms']['ALL'],editable=result['arms']['EDITABLE'],
    review_patches=patches,
    controls={a:result['arms'][a] for a in controls},TASK_checks=result['TASK_checks'],failures=result['failures'],
    control_diagnostics={a:result['arms'][a] for a in ('H2_vs_G','B1_vs_G')},
    caution='H2 andB1 clear raw G0, but zero-GPU-change G also has CI>1. Both paired contrasts againstG span1. Keep raw gate unchanged; require P4 fresh S19 composition to establish a deployable gain.',
    recovery_metric='ln(R)/ln(1.0855676); signed in result; slower recipes recover0 positive share',
    integration=dict(source_builder='problems/muse_bf16_proj/dev/stream_continuity/extend/build.py',
        asserted_transforms='problems/muse_bf16_proj/dev/stream_continuity/extend/patch.py',
        runtime='problems/muse_bf16_proj/dev/stream_continuity/extend/runtime.py.inc',
        prepared_state='rebind only the selected role functions/modules; retain incumbent wp/esc/TMA-map tensor allocations; never globally replace every NF role',
        wide_launch='OL first-stage changes require actual PDL cudaLaunchKernelEx and a private ol_launch_t configuration guard; cross-DSO GNU-unique configured flag reproduced at48KiB, fixed marker eager/graph passes at64KiB',
        readiness='N13 publishes IDs at route completion; entry admission+spin can deadlock in ALL. Scratch flags are reset each timed call. Keep the late GDC before activation reads.',
        baseline_limit='P4 S19 already dropped the original O-only W; these fresh paired results are against S18. Any nominee needs an S19 composition/leave-out G0 check.',
        opaque='ALL and per-role Q/O/P/U use owned BF16 readers where incumbent is opaque; EDITABLE preserves those incumbent libraries. KV uses native VQ8 instead of opaque FA3 at v16/v32S. LATE controls expose replacement cost.'),
    build_manifest_sha256=hashlib.sha256((H/'manifest.json').read_bytes()).hexdigest(),
    no_promotion_without='TASK pass and lower95 six-load paired bound>1; no per-load veto; no independent replay count inflation')
(H/'P4_HANDOFF.json').write_text(json.dumps(handoff,indent=2)+'\n');print('P4 nominees',list(nominees))
