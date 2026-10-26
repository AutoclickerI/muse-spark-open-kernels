"""Pack independent FA3 schedules and parallel merge families on exact S19."""
import base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'chain'
b=json.loads((H/'phases/base_bundle.json').read_text());b['adapter']=(H/'attention_adapter.py.inc').read_text()
a="partial_port_rows=ns['_PP_ROWS'],vq8_rows=ns['_VQ8_ROWS'],attention_plan={})"
z="partial_port_rows=ns['_PP_ROWS'],vq8_rows=ns['_VQ8_ROWS'],attention_plan={},r2_rows=ns['_R2_SELECTION'],wr_rows=ns['_WINDOW_SELECTION'])"
assert b['adapter'].count(a)==1;b['adapter']=b['adapter'].replace(a,z)
b['sched']=(H/'chain_sched.py.inc').read_text();b['bridge']=(o/'fa.cpp').read_text();b['reduce']=(o/'reduce.cu').read_text()
for n in ('N','G','F'):b[n]=(o/(n+'.cu')).read_text()
b['activity']=(R/'problems/muse_decode_layer/dev/timeline/activity.cpp').read_text()
p=R/'problems/muse_attention_decode';ref='\n'.join(l for l in (p/'reference.py').read_text().splitlines() if not l.startswith('from checks import'))
b['checker']='\n'.join(l for s in ((p/'checks.py').read_text(),ref) for l in s.splitlines() if not l.startswith('from __future__ import'))+'\nCACHE_CHUNK=8192\n'
blob=base64.b85encode(lzma.compress(json.dumps(b,separators=(',',':')).encode(),preset=9)).decode()
s='import base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types\nimport torch\n_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(blob)+')))\n'+(H/'chain_lab.py.inc').read_text()
(o/'chain_attention.py').write_text(s);meta=dict(base_sha256=hashlib.sha256(b['source'].encode()).hexdigest(),payload_sha256=hashlib.sha256(s.encode()).hexdigest(),seeds=[5671243+7919*i for i in range(8)],rounds=8,official_TASK=112,stress_TASK=228,scored_TASK=456,partial_TASK=132,native={n:hashlib.sha256(b[n].encode()).hexdigest() for n in ('N','G','F','bridge','reduce')})
(o/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['payload_sha256'])
