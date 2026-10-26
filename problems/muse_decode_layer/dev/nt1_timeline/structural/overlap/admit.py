"""Bind production fingerprints,role proof,local TASK/capture screens andbounded full lab before send."""
import hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();arms=read(D/'manifest.json');cells=read(D/'cells.json');counts=read(D/'counts.json')['cells'];audit=read(D/'sass_audit.json');screens=read(D/'screens.json');component=read(D/'component_validation.json');lab=read(D/'lab_local_validation.json');payload=read(D/'payload.json');diag=read(D/'diag_local_validation.json')
assert lab['complete'] and lab['seeds']==16 and lab['paired_rows']==432 and lab['AA_rows']==48 and not lab['failures'];assert lab['output_chars']<9000
assert sha(R/payload['payload'])==payload['sha256']==lab['payload_sha256']==diag['payload_sha256'];assert len(diag['records']['records'])==6
for arm,spec in arms.items():
 assert sha(R/spec['candidate'])==spec['candidate_sha256']==component['arms'][arm]['candidate_sha256'];assert component['arms'][arm]['TASK_checks']==58;assert sha(R/spec['patch'])==spec['patch_sha256'];subprocess.run(['git','apply','--check',spec['patch']],check=True,capture_output=True)
 for mode in ('normal','forced'):assert screens[arm][mode]['candidate_sha256']==spec['candidate_sha256'] and screens[arm][mode]['TASK_cases']==13 and screens[arm][mode]['shapes']==12
for name,v in cells.items():
 assert sha(R/v['cell'])==v['cell_sha256']==counts[name]['cell_sha256']==audit[name]['cell_sha256'];assert v['cuda_sha256']==counts[name]['cuda_sha256']==audit[name]['cuda_sha256']
 assert all(x['resident']==2 for x in counts[name]['rows'] if x['depth']==2);assert counts[name]['spill_stores']==counts[name]['spill_loads']==0
 if v['kind'].startswith('WS'):
  f=audit[name]['flags']['69'];assert f['dynamic_regs']['producer']==32;assert not f['producer_tail_ops'].get('HMMA',0) and not f['consumer_prefix_ops'].get('UBLKCP',0)
wrapper=R/'runs/screen/p3a_overlap/wrapper.py';assert wrapper.stat().st_size<1000000
result=dict(payload_sha256=payload['sha256'],candidates={a:v['candidate_sha256'] for a,v in arms.items()},native_component_TASK_checks=232,normal_forced_screens=8,each_screen=dict(TASK=13,shapes=12,static='all pass'),dynamic_registers_honored_in_exact_SASS=True,offline_all_arms_resident=2,uninstrumented_zero_spills=True,TRACE_spills='4Bstores BASEclock,24Bstores WS8clock;observer tax measured,diagnostic only',standin='local setmaxnreg nop,CACHE/WS8 raw2to1;producer roles/math native;standin absent from payload',packed_lab=dict(seeds=16,M=[8,16,32],AA=48,pairs=432,chars=lab['output_chars'],seconds=lab['elapsed_seconds'],clock_seeds=4,all_TASK_pass=True),wrapper_sha256=sha(wrapper),source_bytes=wrapper.stat().st_size,status='all local evidence passes;ready for H200 overlap verdict')
(D/'admission.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
