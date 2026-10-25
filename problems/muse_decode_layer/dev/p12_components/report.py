"""Pin every board screen to the exact candidate bytes before queueing."""
import hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
manifest=json.loads((H/'manifest.json').read_text());rows=[]
for item in manifest['candidates']:
 for mode in ('normal','forced'):
  p=H/'screen'/item['role']/mode/'result.json';d=json.loads(p.read_text())
  assert d['sha256']==item['sha256'],(p,d['sha256'])
  checks={k:dict(count=len(d[k]),passed=sum(x['ok'] for x in d[k])) for k in ('tests','shapes','static')}
  assert all(x['count']==x['passed'] for x in checks.values()),(item['role'],mode,checks)
  rows.append(dict(role=item['role'],mode=mode,sha256=d['sha256'],checks=checks,prepare_s=d['prepare_s'],calls=json.loads(p.with_name('calls.json').read_text())))
(H/'admission.json').write_text(json.dumps(dict(base_sha256=manifest['base_sha256'],screens=rows),indent=2)+'\n')
for row in rows:print(row['role'],row['mode'],row['checks'])
