"""Exact SASS classes plus bounded per-pipe issue model;not hardware saturation counters."""
import collections,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_intpipe';cells=json.loads((D/'cells.json').read_text());counts=json.loads((D/'counts.json').read_text())['cells'];rows={}
def ops(text):return dict(collections.Counter(m[1].split('.')[0] for m in re.finditer(r'/\*[0-9a-f]+\*/\s+(?:@!?(?:U?P)\d+\s+)?([A-Z][A-Z0-9_.]*)',text)))
classes={'INT':['SHF','LOP3','LEA','IMAD','IADD3','VIADD','ISETP','SEL','IABS','VIADDMNMX','VIMNMX','PLOP3','POPC'], 'UNIFORM':['UIADD3','ULOP3','USHF','UISETP','UIMAD','ULEA','USEL','VOTEU','R2UR','UMOV'],'FMA':['FADD','FMUL','FFMA','HMUL2','HFMA2','HADD2','HMAX2','HMIN2'],'TENSOR':['HMMA'],'PRMT':['PRMT'],'SHARED':['LDS','STS']}
for name,v in cells.items():
 flag=95 if name.endswith('0') else 69;obj=O/(name.lower()+'.o');sass=subprocess.check_output(['cuobjdump','--dump-sass',str(obj)],text=True);obj.with_suffix('.sass').write_text(sass)
 body=next(b for f,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S) if f'moe_varILi1ELi{flag}E' in f);o=ops(body);c=counts[name];assert c['cuda_sha256']==v['cuda_sha256'];assert o['HMMA']==32 and o['HMUL2']==128;assert c['spill_stores']==c['spill_loads']==0;assert all(t['resident']==2 for t in c['rows'] if t['depth']==2)
 rows[name]=dict(cuda_sha256=v['cuda_sha256'],flag=flag,registers=c['registers'],stack_bytes=c['stack_bytes'],spill_loads=0,spill_stores=0,ops=o,classes={k:sum(o.get(x,0) for x in values) for k,values in classes.items()})
# Shared decoder body remains exactS19:the new arms only change B-fill/copy ownership.
for i in range(3):
 def core(cu):a=cu.index('    const uint8_t* tile =');b=cu.index('    __syncwarp();',a);return cu[a:b]
 a=core((O/f'base{i}.cu').read_text())
 for arm in ('affine','private'):assert core((O/f'{arm}{i}.cu').read_text())==a
# Existing measured operational envelopes:INT/PRMT1.96,BF16HMUL2 3.57,HMMA .658.
# Units:SM cycles per16 requested weights perlane (warp issues512values).
rho={1:1.,8:1.,16:1.2009803921568627,32:1.5905172413793103};core16=dict(INT=273/16,PRMT=1.,FMA=8.,TENSOR=2.,weight_scale_LDS=.5625);rate=dict(INT=1.96,PRMT=1.96,FMA=3.57,TENSOR=.658,weight_scale_LDS=.25)
model=[]
for m,r in rho.items():
 demand={k:v*r/rate[k] for k,v in core16.items()};model.append(dict(M=m,logical_reload_factor=r,cycles_per_unique16=demand,cycles_per_unique_weight={k:v/16 for k,v in demand.items()},largest_issue_class=max(demand,key=demand.get),saturation='unmeasured;model ranks pressure,not scheduler utilization'))
old=json.loads((D.parent.parent/'review/next_issue/sass_cost.json').read_text());archive=R/'runs/screen/p3a_next_issue';correct={};old_sass={}
for tag in ('BASE','INDEX'):
 cu=archive/('cost_'+tag.lower()+'.cu');assert hashlib.sha256(cu.read_bytes()).hexdigest()==old['full_kernels'][tag]['cuda_sha256']
 path=cu.with_suffix('.sass');text=path.read_text();part=next(b for f,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',text,re.S) if 'moe_varILi1ELi69E' in f);correct[tag]=ops(part);old_sass[tag]=hashlib.sha256(text.encode()).hexdigest()
b,ix=correct['BASE'],correct['INDEX'];delta={k:ix.get(k,0)-b.get(k,0) for k in set(ix)|set(b)};delta={k:v for k,v in delta.items() if v}
report=dict(receipt_envelopes='16c180784f414576;per-warp clocks20ef2355a317497f observer tax retained',source_base='S19 62ea5bf4',rows=rows,core_per_requested16=core16,rate_warp_instructions_per_SM_cycle=rate,core_model_unchanged_in_address_arms=model,
 addressing='Per-segment/address-control gains require dynamic segment/copy counts. Full image counts below include divergent/exception paths and are not normalized as executed work.',
 INDEX=dict(receipt='d7fb2d8bddae4b8c',marked_fill_integer_W13=[31,13],marked_fill_integer_W2=[26,8],full_kernel_delta=delta,new_uniform_vote_sites=2,new_LEA_sites=33,new_UIMAD_sites=53,new_UIADD3_sites=delta['UIADD3'],archived_full_ops=correct,archived_sass_sha256=old_sass,parser_fix='Counts predicated @UP instructions;the prior parser omitted them',
 explanation='INDEX removes per-fill row branches but changes uniform scheduling throughout the kernel:two new VOTEU sites and many uniform dispatch/address operations. No extra loads,spills or occupancy loss;122 versus125 registers. Added static sites identify the tradeoff,not its unique runtime cause. AFFINE does not reuse its branch hoist;M1 preserves existing S19 INDEX unchanged.'),
 marker_correction='The exact457-instruction marker is273INT(17LEA+112SHF+144LOP3),16PRMT,128HMUL2,32HMMA,8LDS;the initial scaleLDS is outside. Prior416decode/32MMA/9LDS bucket label mixed that scope. Per-group model charges scaleLDS separately.',conclusions='INT is the largest modeled issue demand at every M,not proven saturated. A/B exactness/layout premises are bounded local results in premises.json;unimplemented forms are unmeasured. Address arms retain the457-instruction math body and twoCTAs.')
(D/'sass_model.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v['classes'] for k,v in rows.items()},indent=2))
