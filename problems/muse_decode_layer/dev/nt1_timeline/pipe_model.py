"""Operational SASS classes and Little-law envelopes; undocumented port limits stay unmeasured."""
import json
from pathlib import Path
D=Path(__file__).resolve().parent
r=json.loads((D/'issue_counts.json').read_text());math=next(x for x in r['probes'] if 'p_mathILi1E' in x['name'] and x['segment']==1)
rhos={8:1.,16:1.2009803921568627,32:1.5905172413793103};clock=1.98;cy=132*clock*1e9/(4e12/(9*32))
epi=[x for x in r['probes'] if 'p_epiILi' in x['name'] and 'Lb0E' in x['name']]
def epi_ops(ops):
 return sum(sum(x['ops'].get(op,0) for op in ops)*(2 if 'p_epiILi0E' in x['name'] else 1) for x in epi)/384
classes=[('integer_L0',sum(math['ops'].get(k,0) for k in ('SHF','LOP3','LEA','IMAD'))/16+epi_ops(('SHF','LOP3','LEA','IMAD','IADD3')),2,'64 scalar integer results/SM/clock;LEA/LOP3/SHF routing is an operational grouping,not a published SASS port map'),
 ('FP32_epi',epi_ops(('FADD','FMUL','FFMA')),4,'normal finite-path epilogue arithmetic;rare divide helper excluded'),
 ('SFU_epi',epi_ops(('MUFU',)),.5,'16 scalar results/SM/clock'),
 ('F2FP_BF16_epi',epi_ops(('F2FP',)),.25,'Packed two-result conversion conservatively charged against16 scalar conversions/SM/clock;not a published SASS-port rate'),
 ('SHFL_epi',epi_ops(('SHFL',)),1,'32 scalar results/SM/clock'),
 ('PRMT',math['ops']['PRMT']/16,None,'No public same-op H200 throughput limit;lab measures independent and dependent chains'),
 ('HMUL2_BF16',math['ops']['HMUL2']/16,4,'NVIDIA f16x2 table gives128 packed results;BF16-specific effective rate measured'),
 ('HMMA16816',math['ops']['HMMA']/16,1,'Dense BF16 TC upper bound4096FLOP/SM/clock /4096FLOP per instruction;mma.sync effective rate measured,not WGMMA peak assumed'),
 ('LDS128_weight_and_scale',.5625,.25,'512 bytes/full warp instruction,128B/SM/clock shared-bank envelope'),
 ('shared_bank_service_bytes_total',448.,128,'288 weight/scale bytes+128 bank-service reduction bytes+32 partial-store bytes per requested group;address model'),
 ('activation_LDG128_fill',1.,None,'Per segment,divide byrun length;512B/fullwarp load. Not part of457-instruction math marker')]
rows=[]
for name,amount,limit,caveat in classes:
 rows.append(dict(class_=name,per_requested16=amount,per_unique16={m:amount*r for m,r in rhos.items()},max_warp_instr_or_bytes_per_SM_cycle=limit,budget_at_4TBs=limit*cy if limit else None,load_M32_at_target=amount*rhos[32]/(limit*cy) if limit else None,caveat=caveat))
latencies=[.25,.5,.75,1.,1.5,2.]
little=[{'latency_us_assumed':v,'unique_4TB_required_B_per_SM':4e6/132*v,'M32_all_reloads_cold_required_B_per_SM':4e6/132*v*rhos[32]} for v in latencies]
result={'base':r['S14_sha256'],'clock_assumed_GHz':clock,'cycles_per_unique_group_at4TBs':cy,'rows':rows,'rhos_from_local32seeds':rhos,
 'little_law':{'stage_bytes':36864,'capacity_2CTA_2stages_B':147456,'one_active_consumed_slot_excluded_lookahead_B':73728,'target_unique_B_per_SM_us':4e6/132,'latency_supported_by_lookahead_us':73728/(4e6/132),'M32_all_reloads_cold_latency_supported_us':73728/(4e6/132*rhos[32]),'illustrative_not_measured_H200':little,'measured_H200_latency':None},
 'limits':['58.7us is ONCE math/activation;S14 instruction normalization is a different schedule','Integer,PRMT,FP32,BF16,LSU pipelines may share execution resources;isolated peak loads cannot simply be added','Published math limits are upper envelopes,not empirical same-SASS opcode limits','Stage capacity is not a measured time-average outstanding count;the lab varies depth and measures TMA completion underload','Repeated block factor is logical TMA traffic;L2 hits mean it is not the HBM byte multiplier'],
 'primary_sources':['https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-best-practices-guide/index.html#arithmetic-instructions','https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#shared-memory-5','https://arxiv.org/html/2501.12084v2']}
(D/'pipe_model.json').write_text(json.dumps(result,indent=2)+'\n')
for x in rows:print(x['class_'],x['per_requested16'],x['per_unique16'][32],x['max_warp_instr_or_bytes_per_SM_cycle'],x['budget_at_4TBs'],x['load_M32_at_target'])
print(result['little_law'])
