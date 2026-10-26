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
  cm=json.loads((D/'combo/manifest.json').read_text());cq=json.loads((D/'combo/queue.json').read_text())
  if (D/'combo/results.json').exists():
   cr=json.loads((D/'combo/results.json').read_text());assert cr['measured'] and not cr['invalid'] and cr['source_audit']['verified']
   new=[]
   for v in cr['summary']:
    if not v['point_nominee']:continue
    arm=v['arm'];spec={k:cm[k] for k in ('base_sha256','candidate','candidate_sha256','patch','patch_sha256')} if arm=='COMBO' else specs['SHARD2' if arm=='BEST' and v['M']==1 else 'PAIR']
    assert hashlib.sha256((R/spec['patch']).read_bytes()).hexdigest()==spec['patch_sha256']
    new.append(dict(v,job=cr['job'],batch='S15 composition',handoff=True,exact_patch=spec,
      shapes=[[1,8,w] for w in (True,False)] if v['M']==8 else [[v['M'],1,w] for w in (True,False)],
      seed_recipe=dict(weights='275353+7919*seed',routing='279000+M+97*seed')))
   result['rows']+=new;result['composition_rows']=new
   result['composition_followup']=dict(queue=cq,manifest=cm,source_audit=cr['source_audit'],status='Measured point nominee; P4 held-out six-load R confirmation remains required')
  else:
   result['unmeasured_composition_followup']=dict(queue=cq,manifest=cm,status='Unmeasured composition; no additive gain or P4 promotion claimed')
 if (D/'next_issue/queue.json').exists():
  follow=dict(queue=json.loads((D/'next_issue/queue.json').read_text()),manifest=str((D/'next_issue/manifest.json').relative_to(R)),admission=str((D/'next_issue/admission.json').relative_to(R)))
  if (D/'next_issue/results.json').exists():
   nr=json.loads((D/'next_issue/results.json').read_text());assert nr['measured'] and not nr['invalid'] and nr['source_audit']['verified'];nm=json.loads((D/'next_issue/manifest.json').read_text());new=[]
   for v in nr['summary']:
    if not v['point_nominee'] or v['arm']=='S15':continue
    spec={k:nm[v['arm']][k] for k in ('base_sha256','candidate','candidate_sha256','patch','patch_sha256')}
    assert hashlib.sha256((R/spec['patch']).read_bytes()).hexdigest()==spec['patch_sha256']
    new.append(dict(v,job=nr['job'],batch='S15 instruction cuts',handoff=True,exact_patch=spec,
      shapes=[[1,8,w] for w in (True,False)] if v['M']==8 else [[v['M'],1,w] for w in (True,False)],seed_recipe=dict(weights='375353+7919*seed',routing='379000+M+97*seed')))
   result['rows']+=new;result['instruction_rows']=new;follow.update(status='Measured; M1 FAST/INDEX direct point winners, M8+ alternatives versus S15 only',source_audit=nr['source_audit'])
  else:follow['status']='FAST/INDEX/FLUSH1 are independently screened but H200-unmeasured'
  result['next_instruction_followup']=follow
 structural=D.parent/'structural'
 if (structural/'results.json').exists():
  sr=json.loads((structural/'results.json').read_text());assert sr['measured'] and not sr['invalid'] and sr['source_audit']['verified'];sm=json.loads((structural/'manifest.json').read_text());new=[]
  for v in sr['summary']:
   if not v['point_nominee'] or v['arm']=='S15':continue
   spec={k:sm[v['arm']][k] for k in ('base_sha256','candidate','candidate_sha256','patch','patch_sha256')}
   assert hashlib.sha256((R/spec['patch']).read_bytes()).hexdigest()==spec['patch_sha256']
   new.append(dict(v,job=sr['job'],batch='structural depth-phase',handoff=True,exact_patch=spec,shapes=[[1,8,w] for w in (True,False)] if v['M']==8 else [[v['M'],1,w] for w in (True,False)],seed_recipe=dict(weights='475353+7919*seed',routing='479000+M+97*seed')))
  result['rows']+=new;result['structural_rows']=new;result['structural_followup']=dict(queue=json.loads((structural/'queue.json').read_text()),source_audit=sr['source_audit'],status='All recipes lose versus c49;SPLIT2M32 versus S15 remains a point alternative for P4')
dq=P/'structural/dequant'
if (dq/'results.json').exists():
 dr=json.loads((dq/'results.json').read_text());assert dr['measured'] and not dr['invalid'] and dr['source_audit']['verified'];dm=json.loads((dq/'manifest.json').read_text());new=[]
 for v in dr['summary']:
  if not v['point_nominee'] or v['arm']=='S15':continue
  spec={k:dm[v['arm']][k] for k in ('base_sha256','candidate','candidate_sha256','patch','patch_sha256')}
  assert hashlib.sha256((R/spec['patch']).read_bytes()).hexdigest()==spec['patch_sha256']
  new.append(dict(v,job=dr['job'],batch='structural exact byte-pair decoder',handoff=True,exact_patch=spec,shapes=[[1,8,w] for w in (True,False)] if v['M']==8 else [[v['M'],1,w] for w in (True,False)],seed_recipe=dict(weights='575353+7919*seed',routing='579000+M+97*seed')))
 result['rows']+=new;result['dequant_rows']=new;result['dequant_followup']=dict(queue=json.loads((dq/'queue.json').read_text()),source_audit=dr['source_audit'],status='Measured;every valid point winner exported;no A:A or per-shape CI veto;P4 confirms whole-layer R')
occ=P/'structural/occupancy'
if (occ/'results.json').exists():
 orr=json.loads((occ/'results.json').read_text());assert orr['measured'] and not orr['invalid'] and orr['source_audit']['verified'];om=json.loads((occ/'manifest.json').read_text());new=[]
 for v in orr['summary']:
  if not v['point_nominee'] or v['arm']=='S15':continue
  spec={k:om[v['arm']][k] for k in ('base_sha256','candidate','candidate_sha256','patch','patch_sha256')};assert hashlib.sha256((R/spec['patch']).read_bytes()).hexdigest()==spec['patch_sha256']
  new.append(dict(v,job=orr['job'],batch='three raw slots at two CTAs',handoff=True,exact_patch=spec,shapes=[[1,8,w] for w in (True,False)] if v['M']==8 else [[v['M'],1,w] for w in (True,False)],seed_recipe=dict(weights='675353+7919*seed',routing='679000+M+97*seed')))
 result['rows']+=new;result['occupancy_rows']=new;result['occupancy_followup']=dict(queue=json.loads((occ/'queue.json').read_text()),source_audit=orr['source_audit'],status='Measured;every valid point winner exported. Per-shape CI andA:A describe uncertainty;P4 confirms held-out six-load R')
ov=P/'structural/overlap'
if (ov/'results.json').exists():
 vr=json.loads((ov/'results.json').read_text());assert vr['measured'] and not vr['invalid'] and vr['source_audit']['verified'];vm=json.loads((ov/'manifest.json').read_text());new=[]
 for v in vr['summary']:
  if not v['point_nominee'] or v['arm']=='S15':continue
  spec={k:vm[v['arm']][k] for k in ('base_sha256','candidate','candidate_sha256','patch','patch_sha256')};assert hashlib.sha256((R/spec['patch']).read_bytes()).hexdigest()==spec['patch_sha256']
  new.append(dict(v,job=vr['job'],batch='dedicated TMA issue andcompute overlap',handoff=True,exact_patch=spec,shapes=[[1,8,w] for w in (True,False)] if v['M']==8 else [[v['M'],1,w] for w in (True,False)],seed_recipe=dict(weights='775353+7919*seed',routing='779000+M+97*seed')))
 result['rows']+=new;result['overlap_rows']=new;result['overlap_followup']=dict(queue=json.loads((ov/'queue.json').read_text()),source_audit=vr['source_audit'],clocks='problems/muse_decode_layer/dev/nt1_timeline/structural/overlap/clock_results.json',status='Measured;all valid point wins exported with exact patch/uncertainty. Only P4 held-out six-loadR promotes')
ip=P/'structural/intpipe'
if (ip/'results.json').exists():
 ir=json.loads((ip/'results.json').read_text());assert ir['measured'] and not ir['invalid'] and ir['source_audit']['verified'];im=json.loads((ip/'manifest.json').read_text());new=[]
 for v in ir['summary']:
  if not v['point_nominee'] or v['control_alias']:continue
  arm=v['arm'];spec={k:im[arm][k] for k in ('base','base_sha256','candidate','candidate_sha256','patch','patch_sha256','shipped','shipped_sha256')}
  assert hashlib.sha256((R/spec['patch']).read_bytes()).hexdigest()==spec['patch_sha256']
  new.append(dict(v,job=ir['job'],batch='S19 integer/address issue',handoff=True,exact_patch=spec,alias_diagnostic=next((dict(ratio=z['ratio'],SE=z['SE'],CI95=z['CI95'],note='Identical AFFINE M1 control alias;point gain can reflect paired timing bias,not implementation work') for z in ir['summary'] if z['control_alias'] and z['M']==v['M']),None),shapes=[[1,8,w] for w in (True,False)] if v['M']==8 else [[v['M'],1,w] for w in (True,False)],seed_recipe=dict(weights='975353+7919*seed',routing='979000+M+97*seed')))
 result['rows']+=new;result['intpipe_rows']=new;result['intpipe_followup']=dict(queue=json.loads((ip/'queue.json').read_text()),source_audit=ir['source_audit'],base='S19 62ea5bf4',status='All valid non-alias point wins versus S19;only P4 held-out six-load R promotes;A/B bounded forms unproven,not measured losses')
 (ip/'P4_HANDOFF.json').write_text(json.dumps(dict(base='S19 62ea5bf4',job=ir['job'],rows=new,source_audit=ir['source_audit'],gate=result['gate']),indent=2)+'\n')
dep=P/'structural/dependency'
if (dep/'results.json').exists():
 er=json.loads((dep/'results.json').read_text());assert er['measured'] and not er['invalid'] and er['source_audit']['verified'];em=json.loads((dep/'manifest.json').read_text());new=[]
 for v in er['summary']:
  if not v['point_nominee']:continue
  spec=em[v['arm']]
  for field in ('candidate','shipped','patch'):
   assert hashlib.sha256((R/spec[field]).read_bytes()).hexdigest()==spec[field+'_sha256']
  new.append(dict(v,job=er['job'],batch='S19 producer-owned W13-W2 slices',handoff=True,exact_patch=spec,shapes=[[1,8,w] for w in (True,False)] if v['M']==8 else [[v['M'],1,w] for w in (True,False)],seed_recipe=dict(weights='1175353+7919*seed',routing='1179000+M+97*seed')))
 result['rows']+=new;result['dependency_rows']=new;result['dependency_followup']=dict(queue=json.loads((dep/'queue.json').read_text()),source_audit=er['source_audit'],base='S19 62ea5bf4',oracle=er['oracle'],costs=str((dep/'results.json').relative_to(R)),status='All valid point wins versus S19 exported;oracle and NORED ineligible;only P4 held-out six-load R promotes' if new else 'No correct point winner versus S19;oracle and NORED ineligible;only the exact measured recipes close')
 (dep/'P4_HANDOFF.json').write_text(json.dumps(dict(base='S19 62ea5bf4',job=er['job'],rows=new,source_audit=er['source_audit'],gate=result['gate'],oracle=er['oracle'],material=er['material'],cost_evidence=str((dep/'results.json').relative_to(R))),indent=2)+'\n')
 if (dep/'sizing.json').exists():
  sz=json.loads((dep/'sizing.json').read_text());assert sz['inputs_sha256'][str((dep/'results.json').relative_to(R))]==hashlib.sha256((dep/'results.json').read_bytes()).hexdigest()
  info=dict(status=sz['status'],report=str((dep/'SIZING.md').relative_to(R)),model=str((dep/'sizing.json').relative_to(R)),free_wait_model_R=sz['score_model']['free_wait_removal_R'],common_added_cost_budget_us_for_R1p01=sz['score_model']['common_added_cost_us_for_R1p01'],new_H200_jobs=sz['new_H200_jobs'],note='Only dependency/DSM forms scoped;lane active via matched S19 ladder;G0 unchanged;favorable C8 sensitivity unverified')
  result['dependency_followup'].update(status=sz['status'],sizing=info)
  hand=json.loads((dep/'P4_HANDOFF.json').read_text());hand['sizing']=info;(dep/'P4_HANDOFF.json').write_text(json.dumps(hand,indent=2)+'\n')
lad=P/'structural/ladder'
if (lad/'results.json').exists():
 lr=json.loads((lad/'results.json').read_text());assert lr['measured'] and not lr['invalid'] and lr['source_audit']['verified'];lm=json.loads((lad/'early/manifest.json').read_text());new=[]
 spec=lm['EARLY']
 for field in ('candidate','shipped','patch'):
  assert hashlib.sha256((R/spec[field]).read_bytes()).hexdigest()==spec[field+'_sha256']
 for v in lr['candidate']:
  if not v['point_nominee'] or v['control_alias']:continue
  new.append(dict(v,job=lr['job'],batch='S19 matched residual / EARLY slot release',handoff=True,exact_patch=spec,shapes=[[1,8,w] for w in (True,False)] if v['M']==8 else [[v['M'],1,w] for w in (True,False)],seed_recipe=dict(weights='1375353+7919*seed',routing='1379000+M+97*seed')))
 result['rows']+=new;result['ladder_rows']=new
 info=dict(base='S19 62ea5bf4',job=lr['job'],status='active;EARLY slot-release placement loses at M8/16/32;activation/MMA scheduling and M32 load-side gap remain material',source_audit=lr['source_audit'],report=str((lad/'REPORT.md').relative_to(R)),ceilings=json.loads((lad/'ceiling.json').read_text()),tested=lr['candidate'],next='Pack-free BF16 operand-layout oracle,then correct timed packing only if exposed benefit supports cost;keep two CTAs',whole_layer_R='Not measured for EARLY;no stage point winner')
 result['ladder_followup']=info
 (lad/'P4_HANDOFF.json').write_text(json.dumps(dict(**info,rows=new,gate=result['gate'],exact_tested_patch=spec,validation=str((lad/'early/validation.json').relative_to(R))),indent=2)+'\n')
operand=P/'structural/operand'
if (operand/'P4_HANDOFF.json').exists():
 hand=json.loads((operand/'P4_HANDOFF.json').read_text());assert hand['rows']==[]
 result['operand_followup']=hand
(P/'point_nominations.json').write_text(json.dumps(result,indent=2)+'\n')
h=json.loads((P/'p4_ilp_handoff.json').read_text());h.update(measured_winner_M=[8,16,32],nominate_rows=[[b,q,w] for b,q in ((1,8),(16,1),(32,1)) for w in (True,False)],M8='point-estimate nominee: ratio .99055 SE .00181 n32; no A:A veto',policy='Per-shape point winners nominate; P4 composes and confirms held-out six-load R.');h.update(comparison_to_PAIR_verified=True,direct_PAIR_receipt=direct['job'],direct_comparison=direct['rows'],direct_ranking='PAIR is faster at M8/M16/M32; ILP is retained as an alternative versus S14')
(P/'p4_ilp_handoff.json').write_text(json.dumps(h,indent=2)+'\n')
print('\n'.join(result['changed_verdicts']))
