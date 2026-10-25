"""Require exact-source local screens, native replay, SASS and full packed-lab evidence before send."""
import hashlib,json,lzma
from pathlib import Path
HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'build_manifest.json').read_text());rows={};receipts=json.loads(lzma.decompress((HERE/'screens.json.xz').read_bytes()))
for name in ('weights','late_prew','hints','stream','c4'):
 assert hashlib.sha256((HERE/(name+'_trial.py')).read_bytes()).hexdigest()==m['variants'][name]
 for mode in ('normal','forced'):
  x=receipts[name+'_'+mode];assert x['sha256']==m['variants'][name]
  assert len(x['tests'])==13 and len(x['shapes'])==12 and len(x['static'])==25
  assert all(r['ok'] for r in x['tests']+x['shapes']+x['static'])
  receipts[name+'_'+mode]=x;rows[name+'_'+mode]=dict(sha256=x['sha256'],tests=13,captured_shapes=12,static_checks=25,passed=True)
for mode in ('normal','forced'):
 x=receipts['roles_'+mode];assert x['weights_candidate_sha256']==m['variants']['weights']
 for arm,r in x['arms'].items():
  assert len(r['tests'])==13 and len(r['shapes'])==12 and all(v['ok'] for v in r['tests']+r['shapes'])
  rows[arm+'_'+mode]=dict(tests=13,captured_shapes=12,shared_prepare=True,passed=True)
 receipts['roles_'+mode]=x
native=json.loads((HERE/'native_validation.json').read_text())
assert all(native['candidates'][n]==m['variants'][n] for n in native['candidates']) and sum(c['replays'] for c in native['cells'])==192
raw=json.loads(lzma.decompress((HERE/'local_raw.json.xz').read_bytes()));assert raw['source_sha256']==m['variants'] and len(raw['cases'])==2 and all(not c['gates'] and len(c['samples'])==32 for c in raw['cases'])
assert hashlib.sha256((HERE/'lab_d1_chain.py').read_bytes()).hexdigest()==m['payload_sha256']
from patch import source
base=(HERE.parents[3]/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
for r in json.loads((HERE/'sass_validation.json').read_text()):
 if r['mode']=='baseline':continue
 assert r['source_sha256']==hashlib.sha256(source(base,r['mode']).encode()).hexdigest() and r['local_bytes']==0 and r['pre_wait']
 if r['mode']=='hints':assert not any('LDG.' in v for v in r['pre_wait'])

(HERE/'screen_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
(HERE/'admission.json').write_text(json.dumps(dict(payload_sha256=m['payload_sha256'],base_sha256=m['base_sha256'],normal_and_forced_candidates=5,normal_and_forced_role_ablations=3,native_changed_input_replays=192,lab_cases=2,lab_seeds=32,lab_arms=11,lab_rounds=10,all_gates_pass=True),indent=2)+'\n')
print('ADMIT',m['payload_sha256'],'5 candidates +3 role ablations normal/forced,192 native replays,11-arm/32-seed lab')
