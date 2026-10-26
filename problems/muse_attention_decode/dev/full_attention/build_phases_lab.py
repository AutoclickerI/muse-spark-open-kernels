"""Pack a stage-only S19 attention comparison. All trial values recomputed in call."""
import base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'phases'
b=json.loads((o/'base_bundle.json').read_text());b['adapter']=(H/'attention_adapter.py.inc').read_text();b['activity']=(R/'problems/muse_decode_layer/dev/timeline/activity.cpp').read_text()
for n in ('Aq','Av','P','C3','T8','T16','E25','E50'):b[n]=(o/(n+'.cu')).read_text()
p=R/'problems/muse_attention_decode'
ref='\n'.join(l for l in (p/'reference.py').read_text().splitlines() if not l.startswith('from checks import'))
b['checker']='\n'.join(l for s in ((p/'checks.py').read_text(),ref) for l in s.splitlines() if not l.startswith('from __future__ import'))+'\nCACHE_CHUNK=8192\n'
blob=base64.b85encode(lzma.compress(json.dumps(b,separators=(',',':')).encode(),preset=9)).decode()
s='import base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types\nimport torch\n_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(blob)+')))\n'+(H/'phases_lab.py.inc').read_text()
(o/'phases_attention.py').write_text(s)
meta=dict(base_sha256=hashlib.sha256(b['source'].encode()).hexdigest(),expanded_base='62ea5bf4',payload_sha256=hashlib.sha256(s.encode()).hexdigest(),native=json.loads((o/'manifest.json').read_text()),seeds=[4671259+7919*i for i in range(8)],rounds=8,official_TASK=78,scored_TASK=280,stress_TASK=140)
(o/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['payload_sha256'])
