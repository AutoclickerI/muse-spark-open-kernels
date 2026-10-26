"""Seal the locally screened portfolio before its diagnostic-completion event."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((H/p).read_text())
m=read('manifest.json');lab=read('lab_manifest.json')
assert m['base_sha256']==lab['base_sha256']=='4b8c803db080f4e0f2a69e4c666ee91573fc6ffde238d2e63db4223f83f84d27'
assert set(m['arms'])==set(lab['arms'])=={'INC','CHAIN','STAGE','LANE1','SOFTMAX','OCHAIN'}
assert sha(H/'out/p5b_night.py')==lab['payload_sha256']
assert sha(H/'out/wrapper.py')==lab['wrapper_sha256']
payload=read('payload_screen.json');assert payload['sha256']==lab['payload_sha256'] and all(x['ok'] for x in payload['static'])
candidate=read('candidate_validation.json');assert len(candidate)==12
for n,s in m['arms'].items():
 assert sha(H/('out/'+n+'.py'))==s['shipped_sha256']
 assert sha(H/(n+'.patch'))==s['patch_sha256']
 for mode in ('normal','forced'):
  rows=[x for x in candidate if x['arm']==n and x['mode']==mode]
  assert len(rows)==1 and rows[0]['passed'] and rows[0]['sha256']==s['shipped_sha256']
  assert rows[0]['TASK']==13 and rows[0]['production_graphs']==12
for mode in ('normal','forced','normal_protocol'):
 s=read(mode+'_screen.json');assert s['complete'] and not s['failures'] and not s['native_claim']
 assert s['payload_sha256']==lab['payload_sha256'] and s['nearby_TASK']==260
 assert s['TASK']==s['expected_TASK']==(3712 if mode=='normal_protocol' else 116)
 assert s['tail_with_4000_suffix_pass'] and s['wire_bytes']<=11700
 assert (s['seed_count'],s['rounds'])==((64,16) if mode=='normal_protocol' else (2,2))
pruned=read('pruned_screen.json');assert pruned['complete'] and pruned['TASK']==96 and pruned['nearby_TASK']==260
native=read('native_compile.json');assert len(native)==3 and all(x['passed'] and x['entry_release'] and x['dependency_wait'] for x in native)
assert native[0]['source_sha256']==m['stage_cuda_sha256'] and len(native[0]['registers'])==6
stage=read('stage_validation.json');assert stage['stage_cuda_sha256']==m['stage_cuda_sha256']
assert len(stage['cases'])==18 and all(x['bit_exact'] and x['gate_after_overwrite_exact'] for x in stage['cases'])
assert sum(x['cross_request_CTAs'] for x in stage['cases'])==106 and sum(x['single_active_lane_CTAs'] for x in stage['cases'])==2
sass=read('sass_validation.json');assert sass['stage_cuda_sha256']==m['stage_cuda_sha256']
policy=read('prune_validation.json');assert all(policy.values())
parent=json.loads((H.parent/'queue.json').read_text());assert parent['pid']==3831549 and parent['priority']==2 and parent['not_before']==1793059200
waiting=json.loads((R/'runs/waiting/3831549.json').read_text());assert waiting['sha256']==parent['sha256']
assert sha(R/parent['file'])==parent['sha256']
fixed=[p for p in H.iterdir() if p.suffix in ('.py','.inc','.patch')]
fixed += [H/'manifest.json',H/'out/bundle.json',H.parent/'analyze.py',H.parent/'lab_manifest.json',H.parent.parent/'public_hybrid/codec.py.inc']
fixed += [R/('problems/muse_decode_layer/dev/'+p) for p in ('s19/build_lab.py','s19/graph_helpers.py.inc','s21/timing.py.inc','s21/lab.py.inc','s21/evidence.py','nearby/cases.py')]
fixed += [R/('problems/muse_decode_layer/'+p) for p in ('checks.py','nvfp4.py','reference.py')]
fixed += [R/('tools/'+p) for p in ('h200_lab.py','lab_packet.py','promotion.py')]
sealed={str(p.relative_to(R)):sha(p) for p in sorted(set(fixed))}
out=dict(passed=True,base_sha256=m['base_sha256'],parent=parent,priority_lane='P5b',priority_at_admission=2,
 candidate_hashes={n:s['sha256'] for n,s in m['arms'].items()},sealed_files=sealed,
 screens={'exact_candidates':candidate,'batch':[read(x+'_screen.json') for x in ('normal','forced','normal_protocol')],
          'pruned_portfolio':pruned,'stage_cases':18,'cross_request_CTAs':106,'single_active_lane_CTAs':2,'sm90a_units':3},
 decision='Wait for diagnostic completion, prune only universally resolved inherited recipe losses, send one whole-graph paired check via h200_lab --lane P5b.',
 native_status='Unmeasured; local Hopper stand-ins do not establish native TASK or performance. Verify kernels still spill; native stage_resources records occupancy.')
(H/'admission.json').write_text(json.dumps(out,indent=2)+'\n');print('ADMITTED',sha(H/'admission.json'),len(sealed),'sealed files')
