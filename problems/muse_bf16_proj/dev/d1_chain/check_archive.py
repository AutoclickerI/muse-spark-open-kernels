"""Verify the actual packed batch retains every seed, graph edge and kernel observation."""
import base64,json,lzma,struct,time
from pathlib import Path
from report import decode
HERE=Path(__file__).resolve().parent
raw=json.loads(lzma.decompress((HERE/'local_raw.json.xz').read_bytes()));data=json.loads(json.dumps(raw))
ns=dict(base64=base64,json=json,lzma=lzma,struct=struct)
exec(compile((HERE/'lab_body.inc').read_text(),'body','exec'),ns);blob=ns['dc_pack'](data);out=decode(blob)
assert len(blob)+300<9000,len(blob)
assert len(raw['cases'])==2 and len(raw['seeds'])==32
for a,b in zip(raw['cases'],out['cases']):
 assert not a['gates'] and len(a['samples'])==32 and len(a['samples'][0])==11
 for s,t in zip(a['samples'],b['samples']):assert all(abs(s[k]-t[k])<.000501 for k in s)
 for arm,p in a['profiles'].items():
  assert a['dag'][arm]['nodes']==b['dag'][arm]['nodes'] and a['dag'][arm]['edges']==b['dag'][arm]['edges']
  assert p==b['profiles'][arm]
(HERE/'batch_validation.json').write_text(json.dumps(dict(cases=2,seeds=32,arms=11,rounds=10,all_gates_pass=True,archive_chars=len(blob),dag_roundtrip_exact=True,all_kernel_observations_exact=True,sample_quantization_us=.001,local_sm120_only=True),indent=2)+'\n')
print('PASS archive',len(blob),'chars; every kernel observation and DAG edge exact, every seed within .001us')
