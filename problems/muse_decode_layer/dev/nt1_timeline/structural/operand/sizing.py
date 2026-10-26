"""Pre-receipt work sizing; never substitute sm120 timings for H200 measurements."""
import hashlib,json,math,statistics,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import promotion
read=lambda p:json.loads(p.read_text());old=read(D.parent/'ladder/results.json');counts=read(D/'counts.json');layer={int(k):v for k,v in old['modeled_layer_us'].items()}
def stats(xs):
 n=len(xs);mean=statistics.mean(xs);se=statistics.stdev(xs)/n**.5;e=promotion.T95[n-1]*se
 return dict(mean=mean,CI95=[mean-e,mean+e],SE=se,values=xs)
def model(ds):return math.exp(sum(-math.log1p(-max(0,v)/layer[m]) for m,v in ds.items())/6)
padding=[]
for seed,M,nn in old['seed_level_record']['work_counts']:
 blocks=sum((n+7)//8 for n in nn);issued=blocks*768*8*32;useful=sum(nn)*768*32
 padding.append(dict(seed=seed,M=M,token_counts=nn,issued_warp_HMMA=issued,useful_equivalent_warp_HMMA=useful,useful_fraction=useful/issued,nominal_wasted_tensor_service_us=(issued-useful)*4096/989e6))
summary=[]
for M in (1,8,16,32):
 v=[x for x in padding if x['M']==M]
 summary.append(dict(M=M,useful_fraction=stats([x['useful_fraction'] for x in v]),issued_warp_HMMA=stats([x['issued_warp_HMMA'] for x in v]),useful_equivalent_warp_HMMA=stats([x['useful_equivalent_warp_HMMA'] for x in v]),nominal_wasted_tensor_service_us=stats([x['nominal_wasted_tensor_service_us'] for x in v])))
ep={m:next(v['mean'] for v in old['contrasts'] if v['M']==m and v['term']=='epilogue_NOWAIT') for m in (8,16,32)}
pad={v['M']:v['nominal_wasted_tensor_service_us']['mean'] for v in summary if v['M']!=1}
s21=read(R/'problems/muse_decode_layer/dev/s21/manifest.json');shipped=R/s21['shipped_source'];assert hashlib.sha256(shipped.read_bytes()).hexdigest()==s21['shipped_sha256']
prod=dict(base='S21',expanded_sha256=s21['sha256'],shipped=str(shipped.relative_to(R)),shipped_sha256=s21['shipped_sha256'],protocol=s21['measurement_protocol'],criterion='TASK.md only;whole production graph times across all12 shapes;T=46*t_sliding+16*t_full;G0 held-out six-load R lower bound>1',selection_scope='Oracles are selection data only;do not nominate a correct leaf from per-call times',next='Choose the largest buildable measured oracle term,include all timed packing and producer costs,local normal/forced glue plus native leaf screens,then one fresh S21 paired production-graph lab under promotion.evaluate',status='awaiting oracle receipt;no production build or G0 result yet')
(D/'production_plan.json').write_text(json.dumps(prod,indent=2)+'\n')
out=dict(status='H200 oracle timing pending;counts and resource sizing only',input_receipt=old['job'],input_sha256=hashlib.sha256((D.parent/'ladder/results.json').read_bytes()).hexdigest(),padding=padding,padding_summary=summary,
 ceilings=dict(padding_nominal_service_M8_32_R=model(pad),padding_model='Wasted source-derived HMMA*4096FLOP/989TFLOP/s,all wall credit;nominal service scenario,not an exposed-time measurement or removable hardware lower bound',activation_reference_M8_32_R=read(D.parent/'ladder/ceiling.json')['activation_reference_M8_32_ceiling_R'],register_lifetime_unallocated_upper_R=read(D.parent/'ladder/ceiling.json')['MMA_activation_M8_32_ceiling_R'],occupancy_only_R=1.,epilogue_reference_M8_32_R=model(ep)),
 resource_claims=dict(shared_pattern_bank_bytes=8192,alignment_reserve_bytes=128,shared_pattern_note='Synthetic two-plane vector fragment pattern repeated across K groups;no real activation pack timed;oracle only',real_full_Bf_cache_bytes=65536,two_raw_slots_full_cache_M32_shared_bytes=85136+65536,one_raw_slot_full_cache_M32_shared_bytes=85136-36864+65536,one_raw_slot_full_cache_admitted=False,reason='Real64KiB cache plus two raw slots fails2CTA;one slot fits113808B but must pay repeat LDS/shared publication and its measured old one-slot penalty;await the FRAG_JIT bound before building'),
 occupancy='All existing oracle layouts fit2CTA,zero spills. BASE126regs,MMA-only L2 127,epilogue L3 128 atM8-32. Dropping epilogue changes no CTA residency;lower registers may change scheduling but no occupancy-only gain is sized',
 convert='S19 already reads BF16 bits directly into uint4/Bf;no BF16 conversion exists in this fill;conversion-removal ceiling is zero',
 closed_forms='Read moe_public_ports P4_HANDOFF/MECHANISMS and FINDINGS mixed_wgmma. No donor kernel or shared decoded-weight/WGMMA producer-consumer family copied. Public whole ports and matched hybrids had no winner.',
 caveat='Old MATH changes BASE register allocation126->47 atM8-32 as well as raw-weight/activation loading;max(MATH,STREAM) is not a proved critical-path roof. Computation/operand work is ranked first;the independent M32 load contrast is not declared impossible',
 new_H200_measurements=None,whole_layer_G0=None)
(D/'sizing.json').write_text(json.dumps(out,indent=2)+'\n')
print('NOMINAL_PADDING_R',out['ceilings']['padding_nominal_service_M8_32_R'],'EPILOGUE_REFERENCE_R',out['ceilings']['epilogue_reference_M8_32_R'])
