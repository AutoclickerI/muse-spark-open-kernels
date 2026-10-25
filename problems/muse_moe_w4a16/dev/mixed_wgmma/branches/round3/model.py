"""Falsifiable schedule estimates; residual allocations are not phase counters."""
import importlib.util
import json
from pathlib import Path
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('round2_model',H.parent/'round2/model.py');r2=importlib.util.module_from_spec(spec);spec.loader.exec_module(r2)
out={}
for m,q in r2.out.items():
 o=q['observed'];a=q['arms']['CTA2'];point=a['estimated_whole_same_overlap_us'];best=a['estimated_whole_full_overlap_us'];cs=r2.counts[m]
 unique=q['weight_stream_4TBs_us'];heavyW=q['unique_weight_MB']/4;lightW=unique-heavyW
 # Logical activation service rates from 3e8996f8 are L2-inclusive, not HBM.
 rates={128:(10.289,8.645),256:(13.610,12.798)}[m]
 heavyX=sum(b/r for b,r in zip(a['X_MB'],rates))
 # NT1's cached-row refresh: ~24 MB in 3e8996f8. N16 has 192 channel tiles.
 n16=sum(8<n<=16 for n in cs);lightX_MB=24+n16*16*4096*2*192/1e6;lightX=lightX_MB/4.5
 hs=a['estimated_heavy_service_us'];ls=sum(o['light']);residualH=hs-heavyW-heavyX;residualL=ls-lightW-lightX
 raw_shell=o['control']-sum(o['heavy'])-ls;shell=max(16,raw_shell);overlap=hs+ls+shell-point
 components=dict(unique_weights_4TBs_us=unique,heavy_activation_L2_service_us=heavyX,light_activation_L2_service_us=lightX,
  heavy_stage_decode_epilogue_residual_us=residualH,light_stage_decode_epilogue_residual_us=residualL,
  route_finish_and_unattributed_tail_us=shell,overlap_credit_us=overlap,launch_topology_increment_assumed_us=0,
  launch_tail_limitation='Launch/topology and terminal CTA tail are not independently measured. They remain inside shell/residual; zero means no assumed incremental launch win.')
 assert abs(unique+heavyX+lightX+residualH+residualL+shell-overlap-point)<1e-6
 gap=point-best
 arms={
  'S14':dict(point_us=o['control'],source='8ee003db observed mean; contextual182.4/256.4 uses an earlier receipt'),
  'CTA2':dict(point_us=point,design_bound_us=best),
  'CTA2S3':dict(point_us=point,design_bound_us=best,assumption='No extra S3 credit before the attribution receipt'),
  'TAIL_STATIC':dict(point_us=point-.25*gap,design_bound_us=best,assumption='Compact list/fused N16 removes one quarter of the remaining scheduling deficit'),
  'TAIL_QUEUE':dict(point_us=point-.50*gap,design_bound_us=best,assumption='Expert-major claiming removes one half of the scheduling deficit'),
  'TAIL_QUEUE_S3':dict(point_us=point-.50*gap,design_bound_us=best,assumption='Same queue recovery; no extra S3 credit before measurement')}
 out[m]=dict(target_us=q['target_us'],components=components,arms=arms,scheduling_deficit_us=gap,
  stream_weight_floor_4_58TBs_us=q['all_unique_weight_MB']/4.58,
  limitation='Scenario estimates, not causal attribution: fixed-fixture phase/count data plus whole-MoE mean control from8ee003db. ACT rates borrow3e8996f8. Residual combines decode, issue, synchronization, epilogue and tail; cannot isolate these from existing receipts. Queue recovery fractions are explicit hypotheses tested by paired runs.')
(H/'model.json').write_text(json.dumps(out,indent=2)+'\n')
for m,q in out.items():
 print(m,q['components'])
 print([(n,round(a['point_us'],1),round(a.get('design_bound_us',a['point_us']),1)) for n,a in q['arms'].items()])
