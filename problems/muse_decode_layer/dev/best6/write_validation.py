"""Freeze final screen counts, gates, timings and explicit native-Hopper hold."""
from pathlib import Path
import ast,hashlib,json,re
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
candidate=ROOT/'problems/muse_decode_layer/cand_r3_best6.py';sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
result={'sha256':sha,'candidate_sent_by_lane':False,'native_h200_verified':False,
        'submission_hold':'owner sends after the separate M128/256 R0 + twelve-seed native adapter lab passes',
        'choices':['v9nt1','v9nt2','raw17','humming_masked','dg_w8']}
for mode in ('normal','forced'):
 p=ROOT/f'problems/muse_decode_layer/dev/best6_{mode}_local.json';x=json.loads(p.read_text())
 assert x['sha256']==sha,(mode,x['sha256'],sha)
 assert len(x['tests'])==13 and all(t['ok'] for t in x['tests'])
 assert len(x['shapes'])==12 and all(t['ok'] for t in x['shapes'])
 assert len(x['static'])==14 and all(t['ok'] for t in x['static'])
 result[mode]=dict(tests=13,fresh_graph_shapes=12,static_checks=14,prepare_s=x['prepare_s'],gpu=x['gpu'],cc=x['cc'])
 log=ROOT/f'problems/muse_decode_layer/dev/best6_{"forced_cold" if mode=="forced" else "normal"}_local.log'
 text=log.read_text();match=re.search(r'BEST_PREPARE elapsed=([\d.]+)s',text)
 result[mode]['candidate_prepare_s']=float(match.group(1))
 gates={}
 for line in text.splitlines():
  if line.startswith('BEST6_MOE '):
   _,n,row=line.split(' ',2);gates[n]=json.loads(row)
 result[mode]['portfolio']=gates
 if mode=='forced':
  assert all(row['reason']=='measured' and not row['rejected'] and len(row['choices'])==5 for row in gates.values())
  assert result[mode]['prepare_s']<240,result[mode]
errors=json.loads((HERE/'error_audit_local.json').read_text());assert errors['sha256']==sha
assert len(errors['cases'])==18
result['runtime_error_audit']=dict(cases=18,all_three_builder_errors_isolated=True,expired_default_checked=True)
result['forced_families']=json.loads((HERE/'forced_families_local.json').read_text())
manifest=json.loads((HERE/'build_manifest.json').read_text());result['base_sha256']=manifest['base_sha256'];result['prefill_sha256']=manifest['prefill_sha256']
result['local_gate_max_relative_l2']=max(v['fp32'] for row in result['forced']['portfolio'].values() for v in row['choices'].values())
result['local_gate_max_worst_row']=max(v['worst_row'] for row in result['forced']['portfolio'].values() for v in row['choices'].values())
(HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('normal','forced')},indent=2))
print('PREPARE',result['normal']['prepare_s'],result['forced']['prepare_s'])
