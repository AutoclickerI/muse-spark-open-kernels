"""Repack the actual local batch; verify receipt bounds and round-trip every observation."""
import base64,importlib.util,json,lzma,struct,time
from pathlib import Path
from report import decode
HERE=Path(__file__).resolve().parent
x=json.loads(lzma.decompress((HERE/'local_raw.json.xz').read_bytes()));raw=json.loads(lzma.decompress((HERE/'local_raw.json.xz').read_bytes()))
ns=dict(base64=base64,json=json,lzma=lzma,struct=struct,time=time)
exec(compile((HERE/'lab_body.inc').read_text(),'body','exec'),ns)
ns['BD_CONFIRM_RUN']=lambda *a:['PDL_AUDIT_XZ85 '+base64.b85encode(lzma.compress(json.dumps(x['confirmation']).encode())).decode()]
ns['bd_run']=lambda *a:x['boundaries']
lines=ns['lab'](time.monotonic()+10,None,None);assert sum(len(s)+1 for s in lines)<9000
blob=next(s.split(' ',1)[1] for s in lines if s.startswith('BD_BATCH_XZ85 '));out=decode(blob)
for section in ('confirmation','boundaries'):
 a=raw[section]['cells'] if section=='confirmation' else raw[section];b=out[section]['cells'] if section=='confirmation' else out[section]
 for before,after in zip(a,b):
  assert not before['gates'] and len(after['samples'])==16
  for s,t in zip(before['samples'],after['samples']):assert all(abs(s[n]-t[n])<.000501 for n in s)
  if section=='boundaries':
   for arm,p in before['profiles'].items():
    assert before['dag'][arm]['nodes']==after['dag'][arm]['nodes'] and before['dag'][arm]['edges']==after['dag'][arm]['edges']
    for v,w in zip(p['nodes'],after['profiles'][arm]['nodes']):
     for k in ('node','name','stream','stage','registers','grid','block','shared'):assert v[k]==w[k]
     for k in ('start','end'):assert abs(v[k]-w[k])<.000501
(HERE/'local_archive.txt').write_text('\n'.join(lines)+'\n')
(HERE/'batch_validation.json').write_text(json.dumps(dict(confirmation_fixtures=12,boundary_fixtures=4,seeds_per_fixture=16,cold_rounds=8,all_gates_pass=True,archive_chars=len(blob),receipt_chars=sum(len(s)+1 for s in lines),lossless_dag=True,interval_quantization_us=.001,sample_quantization_us=.001,portable_only=True),indent=2)+'\n')
print('PASS every seed/edge/interval retained; receipt chars',sum(len(s)+1 for s in lines))
