"""Verify exact-source screens, replay/publication safety and receipt budget before dispatch."""
import ast,hashlib,json,lzma
from pathlib import Path
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'build_manifest.json').read_text());receipts=json.loads(lzma.decompress((HERE/'screens.json.xz').read_bytes()));summaries={}
for name,sha in manifest['variants'].items():
 assert hashlib.sha256((HERE/(name+'_trial.py')).read_bytes()).hexdigest()==sha
 for mode in ('normal','forced'):
  x=receipts[name+'_'+mode];assert x['sha256']==sha
  assert len(x['tests'])==13 and len(x['shapes'])==12 and len(x['static'])==25 and all(r['ok'] for r in x['tests']+x['shapes']+x['static'])
  receipts[name+'_'+mode]=x;summaries[name+'_'+mode]=dict(sha256=sha,tests=13,captured_shapes=12,static_checks=25,passed=True)
for mode in ('normal','forced'):
 x=receipts['roles_'+mode];assert x['source_sha256']==manifest['variants']['exact']
 for name,r in x['arms'].items():
  assert len(r['tests'])==13 and len(r['shapes'])==12 and all(v['ok'] for v in r['tests']+r['shapes'])
  summaries[name+'_'+mode]=dict(tests=13,captured_shapes=12,shared_prepare=True,passed=True)
 receipts['roles_'+mode]=x
cu=(HERE/'kernels.cu.inc').read_text();native=json.loads((HERE/'native_validation.json').read_text());sass=json.loads((HERE/'sass_validation.json').read_text())
assert hashlib.sha256(cu.encode()).hexdigest()==native['native_source_sha256']==sass['native_source_sha256']
assert all(v['local_bytes']==0 for v in sass['native']) and len(native['cells'])==16 and sum(v['replays'] for v in native['cells'])==256
assert sum(v['replays'] for v in native['publication'])==128
raw=json.loads(lzma.decompress((HERE/'local_raw.json.xz').read_bytes()));assert raw['source_sha256']==manifest['variants'] and len(raw['cases'])==8
assert all(not c['gates'] and len(c['samples'])==24 and len(c['samples'][0])==10 for c in raw['cases'])
assert hashlib.sha256((HERE/'lab_boundary_rebuild.py').read_bytes()).hexdigest()==manifest['payload_sha256']
validation=json.loads((HERE/'batch_validation.json').read_text());assert validation['archive_chars']<8800 and validation['all_gates_pass']
(HERE/'screen_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
(HERE/'admission.json').write_text(json.dumps(dict(payload_sha256=manifest['payload_sha256'],base_sha256=manifest['base_sha256'],normal_and_forced_candidates=6,normal_and_forced_role_ablations=2,native_math_replays=256,publication_stale_flag_replays=128,lab_shapes=8,lab_seeds=24,lab_arms=10,lab_rounds=8,all_gates_pass=True),indent=2)+'\n')
print('ADMIT',manifest['payload_sha256'],'6 candidates+2 roles normal/forced;384 native replays;8x24x10x8 lab')
