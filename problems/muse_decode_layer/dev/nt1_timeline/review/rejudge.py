"""Export all valid point-estimate wins; only P4's held-out six-load R gate promotes."""
import hashlib,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent;R=D.parents[4]
report=json.loads((P/'h200_results.json').read_text());rebuild=json.loads((P/'rebuild_results.json').read_text());rows=report['P4_point_estimate_candidates']
for row in rebuild['rows']:
 if row['decision']=='nominate to P4':rows.append(dict(job=rebuild['job'],batch='rebuild',control='S14',**row))
for row in rows:
 row['shapes']=[[1,8,w] for w in (True,False)] if row['M']==8 else [[row['M'],1,w] for w in (True,False)]
 row['handoff']=True
 row['SE']=row.get('se');row['seeds']=row.get('n')
 # Legacy A:A summaries remain evidence; nothing here vetoes a shape by them.
 row.pop('clears_summary_control',None)
arm_kinds={'EPBASE':'base','VEC':'vec','PROD':'producer','NORM':'norm','PAIR':'pair','TREE':'tree','FOUR':'four','LOCAL':'local','FOUR_LOCAL':'four_local','FLIGHT':'flight','ILP':'ilp','K16':'k16','LUT':'lut'}
patches={}
sys.path.insert(0,str(P));import patch_s14
out=R/'runs/screen/p3a_point_patches';out.mkdir(parents=True,exist_ok=True)
for arm in sorted({r['arm'] for r in rows}):
 if arm not in arm_kinds:continue
 path=out/(arm.lower()+'.py');patch_s14.build(arm_kinds[arm],path)
 meta=json.loads(path.with_suffix('.json').read_text());target=P/'review'/('point_'+arm.lower()+'.patch');target.write_bytes(path.with_suffix('.patch').read_bytes());patches[arm]=dict(meta,patch=str(target.relative_to(R)))
direct=json.loads((P/'ilp_vs_pair_results.json').read_text());assert direct['official_passed']
result=dict(direct_PAIR_receipt=direct,gate='Every valid TASK-gated per-shape point estimate below1 nominates; SE and A:A are descriptive. P4 composes and confirms six-load R on held-out seeds.',rows=rows,patches=patches,changed_verdicts=['ILP M8 now nominates: .99055 SE .00181 n32.','M1 PAIR .99890 SE .00232 n16 and NORM .99839 SE .00213 n16 now nominate despite A:A.','TREE direct gains over PAIR at M8/M16 (.99904/.99811) now nominate; TREE versus S14 nominates M8/M16/M32.','LOCAL versus S14 nominates M8/M16/M32 even though it loses to PAIR; alternative, not a combined claim.','PROD and NORM point wins are retained as alternatives to PAIR; EPBASE control-alias point wins are exported for completeness/deduplication.','FLIGHT/K16/LUT and ONCE families remain slower; PDIA remains deliberately incorrect and ineligible.'])
review_path=D/'results.json'
if review_path.exists():
 review=json.loads(review_path.read_text());assert review['measured'] and not review['invalid']
 specs=json.loads((D/'manifest.json').read_text());new=[]
 for v in review['summary']:
  if not v['point_nominee']:continue
  # BLOCK M1/M8 call the unchanged S15 control; retain the noise in the receipt,
  # without presenting an identical control path as a new P4 implementation.
  alias=v['arm']=='BLOCK' and v['M'] in (1,8)
  spec=specs[v['arm']];v=dict(v,job=review['job'],batch='S15 review',handoff=not alias,
      shapes=[[1,8,w] for w in (True,False)] if v['M']==8 else [[v['M'],1,w] for w in (True,False)],
      exact_patch=spec,seed_recipe=dict(weights='175353+7919*seed',routing='179000+M+97*seed'),
      control_alias=alias)
  assert hashlib.sha256((R/spec['patch']).read_bytes()).hexdigest()==spec['patch_sha256']
  assert hashlib.sha256((R/spec['cell']).read_bytes()).hexdigest()==spec['cell_sha256']
  new.append(v)
 result['review_receipt']=review['job'];result['review_rows']=new
 result['review_patches']={arm:specs[arm] for arm in sorted({v['arm'] for v in new})}
 result['rows']+=new
 result['unmeasured_review_receipt']=dict(job='1da2153a0f5543b9',reason='Wrapper dropped GATE and timing archive; no stage verdict')
 result['review_source_audit']=review.get('source_audit')
 result['changed_verdicts']+=['Original S15 review1da2153a0f5543b9 is unmeasured; bounded retryb57d691d689046a8 establishes valid per-arm results.',
   'BLOCK M16/M32 nominates; M1/M8 are unchanged control aliases, not a new kernel win.',
   'SHARD2/4/8 M1 directly beat PAIR by point estimate; at M8+ they lose to PAIR but all remain alternatives versus S15.',
   'CANCEL and PACK5 have no point win at any measured M; close only these exact implementations.']
 if (D/'combo/queue.json').exists():
  result['unmeasured_composition_followup']=dict(queue=json.loads((D/'combo/queue.json').read_text()),manifest=json.loads((D/'combo/manifest.json').read_text()),status='Unmeasured composition; no additive gain or P4 promotion claimed')
(P/'point_nominations.json').write_text(json.dumps(result,indent=2)+'\n')
h=json.loads((P/'p4_ilp_handoff.json').read_text());h.update(measured_winner_M=[8,16,32],nominate_rows=[[b,q,w] for b,q in ((1,8),(16,1),(32,1)) for w in (True,False)],M8='point-estimate nominee: ratio .99055 SE .00181 n32; no A:A veto',policy='Per-shape point winners nominate; P4 composes and confirms held-out six-load R.');h.update(comparison_to_PAIR_verified=True,direct_PAIR_receipt=direct['job'],direct_comparison=direct['rows'],direct_ranking='PAIR is faster at M8/M16/M32; ILP is retained as an alternative versus S14')
(P/'p4_ilp_handoff.json').write_text(json.dumps(h,indent=2)+'\n')
print('\n'.join(result['changed_verdicts']))
