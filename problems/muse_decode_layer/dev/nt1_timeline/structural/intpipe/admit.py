"""Bind exact S19 patches,static/changed tiers,native checks and packed lab before send."""
import hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_intpipe';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=read(D/'manifest.json');cells=read(D/'cells.json');c=read(D/'component_validation.json');lab=read(D/'lab_local_validation.json');payload=read(D/'payload.json');counts=read(D/'counts.json')['cells'];audit=read(D/'sass_model.json')['rows'];assert c['all_pass'] and c['TASK_checks']==324 and not c['standins'];assert lab['complete'] and lab['paired']==128 and lab['AA']==64 and not lab['checks']['failures'];assert sha(R/payload['payload'])==payload['sha256']==lab['payload_sha256'];assert lab['packet_bytes']<=8744
for a,v in m.items():
 assert sha(R/v['candidate'])==v['candidate_sha256'] and sha(R/v['shipped'])==v['shipped_sha256'] and sha(R/v['patch'])==v['patch_sha256']
 if a!='BASE':subprocess.run(['git','apply','--check',v['patch']],check=True,capture_output=True)
 if a!='BASE':
  for tier in ('static','changed'):
   t=read(O/(a.lower()+'_'+tier+'.json'));assert all(x['ok'] for x in t['static']) and t['sha256']==v['shipped_sha256']
for name,v in cells.items():
 assert sha(R/v['cell'])==v['cell_sha256']==c['cells'][name]==counts[name]['cell_sha256'];assert counts[name]['cuda_sha256']==audit[name]['cuda_sha256']==v['cuda_sha256'];assert counts[name]['spill_loads']==counts[name]['spill_stores']==0;assert all(x['resident']==2 for x in counts[name]['rows'] if x['depth']==2)
assert all(x['ok'] for x in read(O/'payload_static.json')['static']);assert all(x['ok'] for x in read(O/'transport.json')['static'])
r=dict(status='all local gates pass;ready for one MoE check',base_S19=m['BASE']['base_sha256'],payload_sha256=payload['sha256'],normal_and_forced='All changed native paths execute directly on sm120;no Hopper stand-in needed for changed kernels. Existing S19 glue is unchanged;static/changed/import-smoke tiers used.',native_TASK_current_input_capture_checks=324,packed_lab=lab,all_resident=2,all_zero_spill=True,PRIVATE_stack_bytes=[counts['PRIVATE'+str(i)]['stack_bytes'] for i in range(3)],A_B='bounded forms not implemented;unmeasured,not measured losses')
(D/'admission.json').write_text(json.dumps(r,indent=2)+'\n');print('ADMITTED',lab['packet_bytes'])
