"""Freeze the exact best6 portfolio and best5 glue in one standalone payload."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
NEW=ROOT/'problems/muse_decode_layer/cand_r3_best6.py'
OLD=ROOT/'problems/muse_decode_layer/cand_r3_best5.py'
new=NEW.read_text();old=OLD.read_text()
assert hashlib.sha256(new.encode()).hexdigest()=='4516cb979dac623a6b6a2347be54cb5ecb416be23d7631aeae092a403beb9528'
assert hashlib.sha256(old.encode()).hexdigest()=='a651af25cb44550932f497b57eb82fb32c0fbe07c1f21a8592bd0982f937c150'
names={'_chunks','_group_barrier','_boundary_kernel','boundary','_route_norm_kernel','route_and_norm','_qk_prep_kernel','qk_prep'}
nodes=[n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes}==names
control=ast.unparse(ast.Module(nodes,type_ignores=[]))+'\n'
body='''"""Paired decode stages on complete public inputs; no host state is modified."""\n'''
body+='DP_BEST6_SOURCE='+repr(new)+'\nDP_BEST5_GLUE_SOURCE='+repr(control)+'\n'
log=(ROOT/'problems/muse_decode_layer/profiles/cand_r3_best5_glue.7370b18221964412.log').read_text()
geometry=[ast.literal_eval(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith('GLUE_GEOMETRY ')]
assert len(geometry)==18
body+='DP_GLUE_PLAN='+repr([(row['t'],row['phase'],row['selected']) for row in geometry])+'\n'
body+=(HERE/'driver.py.inc').read_text()
out=ROOT/'prof/lab/decode_paired.py';out.write_text(body)
manifest=dict(payload=str(out.relative_to(ROOT)),sha256=hashlib.sha256(body.encode()).hexdigest(),
              best6_sha256=hashlib.sha256(new.encode()).hexdigest(),best5_sha256=hashlib.sha256(old.encode()).hexdigest(),
              old_glue_functions=sorted(names),seeds=[172903,180822,188741])
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out,manifest['sha256'])
