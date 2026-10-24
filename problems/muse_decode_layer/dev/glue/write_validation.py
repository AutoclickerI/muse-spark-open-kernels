"""Write a small SHA-pinned handoff from completed local screens."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
cand=ROOT/'problems/muse_decode_layer/cand_r3_best5_glue.py';sha=hashlib.sha256(cand.read_bytes()).hexdigest()
rows={}
for mode in ('normal','forced'):
 p=ROOT/f'problems/muse_decode_layer/dev/glue_{mode}_local.json';x=json.loads(p.read_text())
 assert x['sha256']==sha,(mode,x['sha256'],sha)
 tests=x['tests'];graphs=x['shapes'];static=x['static']
 assert len(tests)==13 and all(t['ok'] for t in tests)
 assert len(graphs)==12 and all(t['ok'] for t in graphs)
 assert len(static)==13 and all(t['ok'] for t in static)
 rows[mode]=dict(tests_passed=len(tests),graphs_passed=len(graphs),static_passed=len(static),prepare_s=x['prepare_s'],
                 gpu=x['gpu'],cc=x['cc'])
resources=json.loads((HERE/'hopper_resources.json').read_text())
rows['hopper_glue_compile']=dict(specializations=len(resources),max_shared_bytes=max(r['shared'] for r in resources))
rows['lab_smoke']=dict(rows=len(json.loads((HERE/'lab_valid_local.json').read_text())),fresh_gate_seeds_per_geometry=3)
rows['sha256']=sha;rows['base_sha256']='a651af25cb44550932f497b57eb82fb32c0fbe07c1f21a8592bd0982f937c150'
rows['h200_verified']=False;rows['candidate_sent']=False;rows['glue_target_us_per_layer']=8;rows['glue_target_verified']=False
rows['lab_queue_key']='bd66426465d8b072';rows['lab_source_sha256']=hashlib.sha256((ROOT/'prof/lab/decode_glue.py').read_bytes()).hexdigest()
(HERE/'validation.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
