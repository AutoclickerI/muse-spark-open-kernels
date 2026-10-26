"""Bind exact candidates,resource screen,TASK/capture evidence and packed receipt before send."""
import hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();arms=read(D/'manifest.json');cells=read(D/'cells.json');counts=read(D/'counts.json')['cells'];screens=read(D/'screens.json');component=read(D/'component_validation.json');lab=read(D/'lab_local_validation.json');payload=read(D/'payload.json')
assert lab['complete'] and lab['seeds']==16 and lab['paired_rows']==624 and lab['AA_rows']==48 and not lab['failures'];assert sha(R/payload['payload'])==payload['sha256']==lab['payload_sha256']
for arm,spec in arms.items():
 assert sha(R/spec['candidate'])==spec['candidate_sha256']==component['arms'][arm]['candidate_sha256'];assert component['arms'][arm]['TASK_checks']==58
 assert sha(R/spec['patch'])==spec['patch_sha256'];subprocess.run(['git','apply','--check',spec['patch']],check=True,capture_output=True)
 for block in (0,1):
  key=spec['kind']+str(block);v=cells[key];c=counts[key];assert sha(R/v['cell'])==v['cell_sha256']==c['cell_sha256'];assert v['cuda_sha256']==c['cuda_sha256'];assert c['spill_stores']==c['spill_loads']==0
  assert all(x['resident']==2 for x in c['rows'] if x['depth']==3)
 deep=spec['kind']+'3'
 for mode in ('normal','forced'):assert screens[deep][mode]['candidate_sha256']==arms[deep]['candidate_sha256'] and screens[deep][mode]['TASK_cases']==13 and screens[deep][mode]['shapes']==12
wrapper=R/'runs/screen/p3a_occupancy/wrapper.py';assert wrapper.stat().st_size<1000000
result=dict(payload_sha256=payload['sha256'],candidates={a:v['candidate_sha256'] for a,v in arms.items()},native_component_TASK_checks=348,whole_layer_normal_forced_screens=6,each_screen=dict(TASK=13,shapes=12,static='all pass'),two_slot_controls='Same CUDA as corresponding three-slot candidate;all6selectors native component/capture-gated',offline_H200_all_three_slot_layouts_resident=2,registers={k:v['registers'] for k,v in counts.items()},standin='local native compact-scratch math with explicit3to2 raw depth;actual3slots H200,absent from payload',packed_lab=dict(seeds=16,M=[8,16,32],AA=48,pairs=624,chars=lab['output_chars'],seconds=lab['elapsed_seconds'],all_TASK_pass=True),wrapper_sha256=sha(wrapper),source_bytes=wrapper.stat().st_size,status='all screens pass;ready for H200 depth-at-two-CTA verdict')
(D/'admission.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
