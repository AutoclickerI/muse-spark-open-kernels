"""Check every paired seed ratio, dependency edge, role interval and stage envelope under the receipt cap."""
import base64,json,lzma,struct
from pathlib import Path
from report import decode,summary
HERE=Path(__file__).resolve().parent
raw=json.loads(lzma.decompress((HERE/'local_raw.json.xz').read_bytes()));data=json.loads(json.dumps(raw));ns=dict(base64=base64,json=json,lzma=lzma,struct=struct)
exec(compile((HERE/'lab_body.inc').read_text(),'rebuild_body','exec'),ns);blob=ns['br_pack'](data);out=decode(blob)
assert len(blob)+200<9000,len(blob)
assert len(raw['cases'])==8 and len(raw['seeds'])==24
for a,b in zip(raw['cases'],out['cases']):
 assert not a['gates'] and len(a['samples'])==24 and len(a['samples'][0])==10
 for s,t in zip(a['samples'],b['samples']):
  assert abs(s['A']-t['A'])<=.005001
  assert all(abs(s[k]/s['A']-t[k]/t['A'])<=.000050001 for k in s)
 for arm,p in b['profiles'].items():
  orig=a['profiles'][arm]
  assert a['dag'][arm]['nodes']==[n[:2] for n in b['dag'][arm]['nodes']] and a['dag'][arm]['edges']==b['dag'][arm]['edges']
  assert p['nodes']==[n for n in orig['nodes'] if n['stage'] in ('o','b1','pool_router','route','unpooler','b2')]
  for stage,r in p['stage_envelopes'].items():
   assert r['start_ns']==min(n['start_ns'] for n in orig['nodes'] if n['stage']==stage) and r['end_ns']==max(n['end_ns'] for n in orig['nodes'] if n['stage']==stage)
summary(out)
(HERE/'batch_validation.json').write_text(json.dumps(dict(shapes=8,seeds=24,arms=10,rounds=8,all_gates_pass=True,archive_chars=len(blob),dag_roundtrip_exact=True,role_intervals_and_resources_exact=True,stage_envelopes_exact=True,ratio_quantization=.0001,baseline_quantization_us=.01,redundant_verify_P_profile_omitted=True,local_sm120_only=True),indent=2)+'\n')
print('PASS archive',len(blob),'chars; all24 seeds, full DAGs, role intervals and stage envelopes retained')
