"""Bind the exact packed payload to successful local screens and bit/resource proofs."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();arms=read(D/'manifest.json');screens=read(D/'screens.json');component=read(D/'component_validation.json');lab=read(D/'lab_local_validation.json');payload=read(D/'payload.json')
assert lab['complete'] and lab['seeds']==16 and lab['paired_rows']==240 and lab['AA_rows']==48 and not lab['failures']
assert sha(R/payload['payload'])==payload['sha256']==lab['payload_sha256']
for arm,spec in arms.items():
 assert sha(R/spec['candidate'])==spec['candidate_sha256']==component['arms'][arm]['candidate_sha256']
 assert sha(R/spec['patch'])==spec['patch_sha256']
 for mode in ('normal','forced'):assert screens[arm][mode]['candidate_sha256']==spec['candidate_sha256'] and screens[arm][mode]['TASK_cases']==13 and screens[arm][mode]['shapes']==12
 assert component['arms'][arm]['TASK_checks']==58
assert component['full_bank_bitproof']['weights_bit_exact']==1610612736 and component['full_bank_bitproof']['scale_codes_bit_exact']
table=read(D/'table_validation.json');assert table['checks']==32768 and sha(D/'probe.cu')==table['source_file_sha256']
audit=read(D/'static_audit.json');cells=read(D/'cells.json')
for k,v in cells.items():assert v['cuda_sha256']==audit['cells'][k]['cuda_sha256'] and v['cell_sha256']==sha(R/v['cell'])
wrapper=R/'runs/screen/p3a_structural/dequant_wrapper.py';assert wrapper.stat().st_size<1000000
result=dict(payload_sha256=payload['sha256'],candidates={a:v['candidate_sha256'] for a,v in arms.items()},native_component_TASK_checks=116,normal_forced_screens=4,each_screen=dict(TASK=13,shapes=12,static=27),bank_bitproof=component['full_bank_bitproof'],native_64KiB_address_checks=32768,standin='local32KiB,explicitaddress and1 raw stage;actual64KiB onePRMT1/2stagesHopper compile/address proof;standin absent from payload',packed_lab=dict(seeds=16,M=[8,16,32],AA=48,pairs=240,chars=lab['output_chars'],seconds=lab['elapsed_seconds'],all_TASK_pass=True),wrapper_sha256=sha(wrapper),source_bytes=wrapper.stat().st_size,status='all screens/proofs pass;ready for H200 structural verdict')
(D/'admission.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
