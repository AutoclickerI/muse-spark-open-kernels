"""Bind runtime occupancy, role clocks and the exact submitted packet to the verdict."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5]
read=lambda p:json.loads(p.read_text())
q=read(D/'queue.json');r=read(D/'results.json');c=read(D/'counts.json')['cells'];d=read(D/'clock_results.json')
assert r['measured'] and not r['invalid'] and r['source_audit']['verified']
assert hashlib.sha256((R/q['source']).read_bytes()).hexdigest()==q['sha256']
runtime=[]
for m,base,arms,traces in r['resources']:
 assert base==[2,125,0,{8:84176,16:84496,32:85136}[m],0]
 for name,info in arms:
  count=c[name.upper()+str(int(m>8))];row=next(v for v in count['rows'] if v['M']==m and v['depth']==2)
  expected=[2,count['registers'],0,row['dynamic_smem'],0]
  assert info==expected,(m,name,info,expected)
  runtime.append(dict(M=m,arm=name,info=info,offline_match=True))
 for name,info in traces:
  assert info[0]==2
  assert info[2]==(8 if name=='clock' else 24),(m,name,info)
for v in d['records']:
 w=v['warp_rows'];assert len(w)==(8 if v['kind']=='BASE' else 9)
 assert len({row[7] for row in w[:8]})==1
 tiles=w[0][7]
 if v['kind']=='BASE':
  assert abs(w[0][9]+w[7][9]-tiles)<=1
  assert all(row[9]==0 for row in w[1:7])
 else:
  assert all(row[9]==0 for row in w[:8])
  assert w[8][7]==w[8][8]==0 and abs(w[8][9]-tiles)<=1
  assert w[8][3]==w[8][6]==0
out=dict(job=q['job'],all_uninstrumented_resource_rows_match_offline=True,runtime=runtime,
 clock_role_counts_pass=True,clock_local_bytes=dict(BASE=8,WS8=24),
 observer_tax={str(v['M'])+'_'+v['kind']:[v['observer_ratio'],v['observer_SE']] for v in d['records']},
 limits='Clocks are instrumented residence,not issue utilization or exclusive wall-time attribution;event stamps are issue endpoints.')
(D/'resource_validation.json').write_text(json.dumps(out,indent=2)+'\n')
q.update(state='complete16seeds/allTASK/source-packet-resource audit;all arms lose BASE;WS4 S15 alternatives to P4',receipt=q['source'].replace('.py','.json'))
(D/'queue.json').write_text(json.dumps(q,indent=2)+'\n')
a=read(D/'admission.json');a.update(status=q['state'],receipt=q['receipt'],submitted_wrapper_sha256=q['sha256'],submitted_source_bytes=(R/q['source']).stat().st_size)
(D/'admission.json').write_text(json.dumps(a,indent=2)+'\n')
print(json.dumps(out,indent=2))
