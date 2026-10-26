"""Strict local admission;one H200 packet after exact source/native/glue checks pass."""
import hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_dependency';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cells=read(D/'cells.json');counts=read(D/'counts.json')['cells'];manifest=read(D/'manifest.json');native=read(D/'component_validation.json');lab=read(D/'lab_local_validation.json');payload=read(D/'payload.json')
assert native['all_pass'] and native['TASK_checks']==675 and not native['standins'];assert lab['complete'] and lab['paired']==240 and lab['oracle']==lab['AA']==48 and not lab['checks']['failures'];assert lab['payload_sha256']==payload['sha256']==sha(R/payload['payload']);assert lab['expanded_sha256']==payload['expanded_sha256'];assert lab['packet_bytes']<=8744
for name,v in cells.items():
 assert sha(R/v['cell'])==v['cell_sha256']==native['cells'][name]==counts[name]['cell_sha256'];assert v['cuda_sha256']==counts[name]['cuda_sha256']
 for part in counts[name]['parts'].values():
  for row in part['rows']:assert row['resident']==2 and row['spill_stores']==row['spill_loads']==0
for arm,v in manifest.items():
 assert sha(R/v['candidate'])==v['candidate_sha256'];assert sha(R/v['shipped'])==v['shipped_sha256'];assert sha(R/v['patch'])==v['patch_sha256'];subprocess.run(['git','apply','--check',v['patch']],capture_output=True,check=True)
 t=read(O/(arm.lower()+'_static.json'));assert t['sha256']==v['shipped_sha256'] and all(x['ok'] for x in t['static'])
for name in ('a128_changed','a128d_changed','payload_static','transport'):
 t=read(O/(name+'.json'));assert all(v['ok'] for v in t['static'])
screens={}
for arm,mode in (('A128','normal'),('P128','forced'),('P128D','forced')):
 p=O/(arm.lower()+'_'+mode);t=read(p/'local.json');assert t['sha256']==manifest[arm]['shipped_sha256'] and len(t['tests'])==13 and len(t['shapes'])==12 and all(v['ok'] for k in ('tests','shapes','static') for v in t[k]);calls=read(p/'calls.json')['calls'];leaf=manifest[arm]['prefix']+'_experts';assert calls[leaf]['eager'] and calls[leaf]['captured'];screens[arm]=dict(mode=mode,TASK=13,shapes=12,leaf_calls=calls[leaf])
r=dict(status='All local gates pass;ready for one MoE check',payload_sha256=payload['sha256'],native_TASK_checks=675,native_standins=False,full_S19_screens=screens,all_offline_resident=2,all_zero_spills=True,packed_local=lab,oracle='NOWAIT/NORED/NOREDD are deliberately incorrect and ineligible',decision='All16 oracle pairs first;builds only if M16 or M32 mean NOWAIT/BASE<.95;all valid point wins to P4')
(D/'admission.json').write_text(json.dumps(r,indent=2)+'\n');print('ADMITTED',lab['packet_bytes'])
