"""Normalize marked SASS to unique packed bytes and enumerate reduction-bank requests."""
import collections,functools,json,re
from pathlib import Path
DEV=Path(__file__).resolve().parent
r=json.loads((DEV/'issue_counts.json').read_text());rows=r['probes']
def math(n):return next(x for x in rows if f'p_mathILi{n}E' in x['name'] and x['segment']==1)
def epi(p,role,v):return next(x for x in rows if f'p_epiILi{p}ELi{role}ELb{int(v)}E' in x['name'])
result={'clock_assumptions_GHz':[1.41,1.83,1.98],'budget_per16_weights_per_lane':{},'math':{},'epilogue':{},'banks':{}}
for ghz in result['clock_assumptions_GHz']:result['budget_per16_weights_per_lane'][str(ghz)]=4*132*ghz*1e9/(4e12/(9*32))
for n in (1,2,3,4):
 x=math(n);result['math'][n]={'instructions_per_warp_tile':x['instructions'],'per16_weights_per_lane':x['instructions']/16,'dequant_scale':416/16,'MMA':x['ops'].get('HMMA',0)/16,'activation_global_loads':x['ops'].get('LDG',0)/16,'packed_weight_shared_loads':8/16,'probe_spill_instructions':x['ops'].get('STL',0)+x['ops'].get('LDL',0),'other':(x['instructions']-416-x['ops'].get('HMMA',0)-x['ops'].get('LDG',0)-8)/16}
for pas,nroles in ((0,2),(1,4)):
 old=[epi(pas,w,False) for w in range(nroles)];new=epi(pas,0,True)
 result['epilogue'][pas]={'scalar_warp_role_instructions':[x['instructions'] for x in old],'scalar_CTA_instructions':sum(x['instructions'] for x in old),'vector_CTA_instructions':new['instructions'],'scalar_per16':sum(x['instructions'] for x in old)/128,'vector_per16':new['instructions']/128,'scalar_CTA_shared_load_instructions':sum(x['ops'].get('LDS',0) for x in old),'vector_CTA_shared_load_instructions':new['ops'].get('LDS',0),'scalar_CTA_atomic_warp_instructions':sum(x['ops'].get('REDG',0) for x in old),'vector_CTA_atomic_warp_instructions':new['ops'].get('REDG',0),'scalar_CTA_SHFL':sum(x['ops'].get('SHFL',0) for x in old),'vector_CTA_SHFL':new['ops'].get('SHFL',0),'scalar_BF16_output_converts':sum(x['ops'].get('F2FP',0) for x in old),'vector_BF16_output_converts':new['ops'].get('F2FP',0)}
 for role in range(nroles):
  banks=collections.defaultdict(set)
  for lane in range(32):
   tid=role*32+lane
   if pas==0:i=tid&7;n=tid>>3;ls=i*4+(n>>1);rg=n&1
   else:i=tid&15;n=tid>>4;ls=(i&7)*4+(n>>1);rg=(n&1)+((i>>3)<<1)
   word=ls*4+rg;banks[word%32].add(word)
  result['banks'][f'pass{pas}_scalar_role{role}']={'banks_used':len(banks),'max_distinct_words_per_bank':max(map(len,banks.values()))}
 # A128-bit instruction splits into four128B requests, each8 adjacent lanes owns32 distinct words.
 counts=[]
 for start in range(0,32,8):
  banks=collections.defaultdict(set)
  for lane in range(start,start+8):
   for j in range(4):word=lane*4+j;banks[word%32].add(word)
  counts.append(max(map(len,banks.values())))
 result['banks'][f'pass{pas}_vector_requests']={'128B_requests_per_LDS128':4,'conflict_degree_per_request':counts,'scalar_wavefronts_per_CTA_model':128,'vector_wavefronts_per_CTA_model':32}
# A branch-aware envelope for one transferring producer call, with no barrier retry. Predicates are not
# symbolically correlated; these are instruction-path bounds, not measured steady-state counts.
sass=(DEV/'issue_probe.sass').read_text()
name,body=next((n,b) for n,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S) if 'p_prefetch' in n)
ins=[]
for line in body.splitlines():
 m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;',line)
 if m:ins.append((int(m[1],16),m[2]))
marks=[i for i,(_,op) in enumerate(ins) if 'CS2R' in op and 'SR_CLOCK' in op];a,b=marks[0]+1,marks[1];lookup={pc:i for i,(pc,op) in enumerate(ins)}
@functools.lru_cache(None)
def walk(i,copied):
 if i==b:return (0,0) if copied else None
 if i<a or i>b:return None
 pc,op=ins[i];copied=copied or 'UBLKCP' in op
 if 'BPT' in op or op.startswith('EXIT'):return None
 dest=[i+1]
 m=re.search(r'\bBRA\s+(0x[0-9a-f]+)',op)
 if m:
  j=lookup.get(int(m[1],16),-1)
  dest=([i+1] if op.startswith('@') else [])+([j] if j>i else [])
 vals=[v for j in dest if (v:=walk(j,copied)) is not None]
 if not vals:return None
 return 1+min(v[0] for v in vals),1+max(v[1] for v in vals)
result['producer']={'full_static_instruction_set':next(x['instructions'] for x in rows if 'p_prefetch' in x['name']),'one_copy_no_retry_path_bounds':walk(a,False),'normalization_groups_per_tile':128,'caveat':'bounds may include infeasible combinations of predicates; retries are unbounded; changing pass is rarer than steady advance'}
# Routing classes reconstructed from H200 f58cc7a6 phase tile counts /512:123 experts,198 blocks/16seeds.
classes={1:81,2:17,3:17,4:8};experts=sum(classes.values());blocks=sum(n*v for n,v in classes.items())
old_epi=(result['epilogue'][0]['scalar_per16']*2+result['epilogue'][1]['scalar_per16'])/3
vec_epi=(result['epilogue'][0]['vector_per16']*2+result['epilogue'][1]['vector_per16'])/3
store_barrier=7*8/128
result['routing_classes']={'nb_to_expert_count':classes,'seeds':16,'experts':experts,'blocks':blocks,'mean_blocks_per_expert':blocks/experts}
result['weighted_model']={'S14_math_per_unique16':math(1)['instructions']/16*blocks/experts,'ONCE_math_probe_per_unique16':sum(classes[n]*math(n)['instructions']/16 for n in classes)/experts,'S14_scalar_epi_per_unique16':old_epi*blocks/experts,'ONCE_scalar_epi_per_unique16':old_epi*blocks/experts,'S14_vector_epi_per_unique16':vec_epi*blocks/experts,'partial_store_barrier_per_requested16':store_barrier,'activation_cache_fill_once_per_segment_per16':46/16,'ONCE_extra_B0_shared_loads_not_in_probe_per16':8/16,'producer_no_retry_per_requested16_bounds':[x/128 for x in result['producer']['one_copy_no_retry_path_bounds']]}
result['limits']=['valid dense lane path; math probes omit value masks and outer scheduler','math probe NB3/NB4 spills differ from the nonspilling full ONCE kernel; shown explicitly','epilogue static sets include exceptional division branches but not their callees; normal finite-path instruction model only','S14 B0 global loads occur once per expert/block segment, amortize46 instructions by the actual run length','four-way bank conflicts and dependent waits are latency costs; instruction count does not measure stall cycles','1.98GHz is an assumption; next H200 trace reports clock64/globaltimer ratio','atomics count warp instructions separately from active scalar atomic operations']
pairs={}
for pas in (0,1):
 rr=[next(x for x in rows if f'p_pairILi{pas}ELi{role}E' in x['name']) for role in (0,1)]
 count=sum(x['instructions'] for x in rr)-(2 if pas==1 else 0)
 pairs[pas]={'per_role_static_instructions':[x['instructions'] for x in rr],'per_CTA_after_observation_store_discount':count,'per16_weights_per_lane':count/128,'CTA_shared_load_instructions':sum(x['ops'].get('LDS',0) for x in rr),'CTA_LDS64':sum(sum(v for k,v in x['memory_width_ops'].items() if k.startswith('LDS.64')) for x in rr),'synthetic_norm_observation_STG_discount':2 if pas==1 else 0}
 result['banks'][f'pass{pas}_pair_requests']={'LDS64_writers':2,'128B_requests_per_instruction':2,'conflict_degree':1,'reduction_wavefronts_per_CTA':2*8*2}
result['pair_epilogue']=pairs
flush=[x for x in rows if 'p_norm_flush' in x['name']]
result['pair_norm_flush']={'CTA_instructions_per_run':sum(x['instructions'] for x in flush),'CTA_SHFL_per_run':sum(x['ops'].get('SHFL',0) for x in flush),'CTA_atomic_warp_instructions_per_run':sum(x['ops'].get('REDG',0) for x in flush),'physical_scalar_atomics_per_full8token_run':8}
result['pair_partial_store_barrier']={'per_warp':next(x['instructions'] for x in rows if 'p_store_pair' in x['name']),'per_CTA':8*next(x['instructions'] for x in rows if 'p_store_pair' in x['name'])}
result['weighted_model']['scale_shared_load_per_requested16_excluded_from_mark']=1/16
result['limits']+=['SCV scale vector is loaded before math marker: add1 shared-vector instruction per warp tile','PAIR norm probe uses one synthetic store/role to expose the persistent register value; discounted','The epilogue scales with output tile count, not K bytes per tile; it is a fixed K-split reduction at fixed output geometry','shared-wavefront counts are an address-model estimate, not a measured bank-conflict counter']
result['round2_epilogue']={}
for label,code,nroles in (('TREE',0,2),('FOUR',1,4)):
 result['round2_epilogue'][label]={}
 for pas in (0,1):
  rr=[next(x for x in rows if f'p_next{code}ILi{pas}ELi{role}E' in x['name']) for role in range(nroles)]
  n=sum(x['instructions'] for x in rr)-(nroles if pas==1 else 0)
  result['round2_epilogue'][label][pas]={'per_role_static_instructions':[x['instructions'] for x in rr],'CTA_after_observation_store_discount':n,'per16_weights_per_lane':n/128,'CTA_LDS64':sum(sum(v for k,v in x['memory_width_ops'].items() if k.startswith('LDS.64')) for x in rr),'caveat':'static finite path set includes guarded inactive-lane branches, not a dynamic issue count'}
# SASS-to-pipe mapping is not measured. This conservative envelope groups decoder integer/address ops.
intops=sum(math(1)['ops'].get(k,0) for k in ('LOP3','SHF','LEA','PRMT','IMAD'))/16
result['integer_envelope']={'S14_math_integer_address_per_requested16':intops,'S14_M32_per_unique16':intops*blocks/experts,'hypothetical_2warp_issue_budget_per16_at_clock':{str(c):v/2 for c,v in [(c,result['budget_per16_weights_per_lane'][str(c)]) for c in result['clock_assumptions_GHz']]},'caveat':'Hopper has64 INT32 cores and4 schedulers; exact routing/throughput of each SASS opcode is not measured; excludes outer integer work'}
(DEV/'issue_budget.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
