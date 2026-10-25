"""Heavy-only pipeline bound from the prior explicit ledger; no light credit."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
old=json.loads((H.parent/'round3/model.json').read_text());out={}
for m,q in old.items():
 c=q['components'];prior=q['arms']['CTA2']['point_us'];heavyW={128:113.246208/4,256:198.180864/4}[int(m)];light={128:130.03,256:114.35}[int(m)];shell=c['route_finish_and_unattributed_tail_us']
 # Hypothesis: TMA weight stream covers decode, MMA issue and cached X service.
 # Four us budgets fill/drain, exposed terminal work and the larger norm reduction.
 gather={128:9.02,256:8.96}[int(m)];h=heavyW+4
 # Resource audit: 64,512 entry registers leave no room for an NT1 CTA on the
 # same SM. Spatial/tail overlap is possible, but carrying the old overlap as
 # a point prediction is unjustified. Record it only as an optimistic scenario.
 credit=0;optimistic_credit=min(c['overlap_credit_us'],h,light);pipe=h+light+shell+gather
 # Barrier arm deliberately exposes ~one tenth of the old heavy residual.
 hb=h+.1*c['heavy_stage_decode_epilogue_residual_us'];bar=hb+light+shell+gather
 out[m]=dict(target_us=q['target_us'],CTA2_point_us=prior,heavy_weight_stream_us=heavyW,
  heavy_residual_targeted_us=c['heavy_stage_decode_epilogue_residual_us'],light_residual_untouched_us=c['light_stage_decode_epilogue_residual_us'],
  gather_service_us=gather,fill_drain_allowance_us=4,light_service_unchanged_us=light,route_finish_tail_us=shell,overlap_credit_point_us=credit,
  optimistic_spatial_overlap_credit_us=optimistic_credit,optimistic_spatial_overlap_prediction_us=pipe-optimistic_credit,
  WS_PIPE_prediction_us=pipe,WS_TILE_BARRIER_prediction_us=bar,remaining_above_target_us=pipe-q['target_us'],
  full_overlap_bound_us=q['arms']['CTA2']['design_bound_us'],
  assumption='Decode/epilogue plus cached X service hide under the heavy unique-weight stream. This is a hypothesis, not a consequence of specialization. No light residual removed. Register admission64,512 prevents NT1 CTA co-residency; zero interbranch overlap is the point estimate. Old overlap survives only as an optimistic spatial/tail scenario. Barrier-arm10% exposed residual is an unmeasured sensitivity.')
(H/'model.json').write_text(json.dumps(out,indent=2)+'\n')
for m,q in out.items():print(m,q['WS_PIPE_prediction_us'],q['WS_TILE_BARRIER_prediction_us'],q['remaining_above_target_us'])
