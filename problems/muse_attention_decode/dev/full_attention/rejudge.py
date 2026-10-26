"""Nominate every measured per-shape point improvement; P4 confirms six-load R."""
import base64,difflib,hashlib,json,lzma,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=R/'problems/muse_decode_layer';out=H/'rejudgment';out.mkdir(exist_ok=True)
base=(P/'cand_r3_best7_s15.py').read_text();rows=[]
for lane,full,groups,job in [('pdl_release','E',['attention','moe','projections','glue'],'f88fda4aac144374'),('library_chain','L',['o','pool128','unpool128','v1full','v32slide'],'f22020d25d234be6')]:
 d=json.loads((P/'dev'/lane/'reset_result.json').read_text());r=d['result']
 # Frozen summaries expose seed SE only after46/16 load aggregation. State
 # missing per-shape uncertainty honestly; never manufacture covariance.
 for shape,v in r['shapes'].items():
  for g in groups:
   off='no_'+g;ratio=v['us'][full]/v['us'][off]
   active=(g=='moe' or (g=='attention' and shape.startswith(('16x1','32x1'))) or
       (g=='projections' and shape.startswith('16x8')) or (g=='o' and shape.startswith(('16x8','32x8'))) or
       (g in ('pool128','unpool128') and shape.startswith('16x8')) or
       (g=='v1full' and shape=='1x8f') or (g=='v32slide' and shape=='32x8s'))
   if not active:continue
   if ratio>=1 or g=='glue':continue
   load=shape[:-1];s=r['loads'][load]
   rows.append(dict(lane=lane,piece=g,shape=shape,ratio_on_over_off=ratio,se=None,se_reason='Frozen receipt omitted per-shape seed samples and SE; load mean SE below is not a per-piece paired SE.',
    full_load_ratio=s[full][0],full_load_SE=s[full][1],off_load_ratio=s[off][0],off_load_SE=s[off][1],seeds=8,job=job,
    scope_caveat='An unchanged shape may vary by graph/order noise; nominate executable changes only at shapes they affect.'))
 # Exact original full-minus-one-on differences for all groups, including
 # grouped MoE (small-M/V9 and large) and O256 previously omitted as uncertain.
 import importlib.util
 sp=importlib.util.spec_from_file_location('rejudge_'+lane,P/'dev'/lane/'build.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 for g in groups:
  if g=='glue':continue
  src=m.compose(without=tuple(x for x in groups if x!=g));src=src[0] if isinstance(src,tuple) else src
  patch=''.join(difflib.unified_diff(base.splitlines(True),src.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15.py'))
  file=out/(lane+'_'+g+'.patch');file.write_text(patch)
  subprocess.run(['git','apply','--check',str(file)],cwd=R,check=True)
  for row in rows:
   if row['lane']==lane and row['piece']==g:row.update(patch=str(file.relative_to(R)),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),candidate_sha256=hashlib.sha256(src.encode()).hexdigest())
near=json.loads((P/'dev/nearby/P4_HANDOFF.json').read_text());near['status']='Nominate whole guarded S15n to P4 now; unchanged scored dispatch, native TASK receipt pending';near['point_estimate_rule']='No per-load/A:A veto for nomination; P4 composes and applies held-out six-load R gate.';(P/'dev/nearby/P4_HANDOFF.json').write_text(json.dumps(near,indent=2)+'\n')
result=dict(policy='Every measured shape point ratio<1 is a nomination, with uncertainty; only P4 applies the held-out paired six-load R lower-CI gate. TASK.md only.',rows=rows,nearby=near,
 changed_verdicts=['Whole guarded S15n handed off now, not withheld for per-load spread. Native TASK result remains required correctness evidence.',
 'O256 and grouped small-M release cells with point benefit are nominated despite A:A overlap.',
 'VDSK qkv release v16 nomination retained: six-load marginal below1 is inapplicable to its one supported T128 cell.',
 'Previously losing whole compositions do not veto improved individual shape cells.',
 'Both reset softmax batches still nominate no changed arm: all point ratios exceed1 at both target shapes.'])
(out/'P4_HANDOFF.json').write_text(json.dumps(result,indent=2)+'\n');print('nominations',len(rows))
