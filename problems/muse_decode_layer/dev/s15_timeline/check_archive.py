"""Prove packed timestamps/DAG/accounting/work models round-trip within their stated resolution."""
import argparse,base64,hashlib,json,lzma,statistics,struct
from pathlib import Path
from analysis import chain_account,stage_name
from decode import decode,aggregate,table
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--forced',action='store_true');a=p.parse_args()
raw=json.loads(lzma.decompress((HERE/('forced_raw.json.xz' if a.forced else 'local_raw.json.xz')).read_bytes()));data=json.loads(json.dumps(raw));ns=dict(base64=base64,hashlib=hashlib,json=json,lzma=lzma,statistics=statistics,struct=struct)
ns['stage_name']=stage_name
exec(compile((HERE/'lab_body.inc').read_text(),'body','exec'),ns);blob=ns['t15_pack'](data);out=decode(blob)
assert len(blob)+200<9000,len(blob)
assert len(out['cases'])==12
for before,after in zip(raw['cases'],out['cases']):
 assert before['schedule_identical'] and not before['gates'] and len(before['seeds'])==3
 assert before['dag']==after['dag']
 for x,y in zip(before['trace'],after['trace']):
  assert x['name'].replace('void ','').replace('(anonymous namespace)::','').split('(')[0][:16]==y['name'].rsplit('@',1)[0]
  assert stage_name(x['role'],x['name'])==y['role']
  assert all(x[k]==y[k] for k in x if k not in ('name','role')),(x,y)
 for x,y in zip(before['seeds'],after['seeds']):
  assert x['work']==y['work'],('work reconstruction mismatch',before['case'],x['work'],y['work'])
  for k,v in x['account']['critical_us'].items():assert abs(v-y['account']['critical_us'][k])<1e-8
  assert abs(x['account']['wall_us']-y['account']['wall_us'])<1e-8
  assert x['account']['joins']==y['account']['joins']
  assert x['all_stamps_monotone'] and x['max_accounting_error']<.02
 assert abs(chain_account(after['trace'],after['dag'])['wall_us']-chain_account(before['trace'],before['dag'])['wall_us'])<1e-8
report=aggregate(out);text=table(report);assert all(name in text for name in ('attention_sliding','attention_full','moe_small','moe_heavy','moe_light','launch_gaps'))
result=dict(s15_sha256=raw['s15_sha256'],shapes=12,seeds_per_shape=3,profiles_per_seed=3,all_schedule_signatures_identical=True,all_output_gates_pass=True,all_stamps_monotone=True,max_accounting_error=max(s['max_accounting_error'] for c in raw['cases'] for s in c['seeds']),archive_chars=len(blob),timestamp_and_dag_roundtrip_exact=True,logical_work_roundtrip_exact=True,critical_accounting_roundtrip_exact=True,local_sm120_only=True)
(HERE/('forced_validation.json' if a.forced else 'local_validation.json')).write_text(json.dumps(result,indent=2)+'\n');(HERE/('forced_receipt.txt' if a.forced else 'local_receipt.txt')).write_text('META packed actual local '+('forced' if a.forced else 'normal')+' timeline\nS15_TIMELINE_XZ85 '+blob+'\nEND local sm120 evidence\n')
print('PASS',json.dumps(result))
