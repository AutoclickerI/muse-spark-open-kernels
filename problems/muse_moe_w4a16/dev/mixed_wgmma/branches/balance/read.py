"""Reconstruct independent-seed CIs and the ordered, conditional oracle ledger."""
import argparse
import base64
import json
import math
from pathlib import Path
import re
import statistics
import struct
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args()
r=json.loads(a.receipt.read_text());tail='\n'.join(r['lines']) if 'lines' in r else r['result'].get('log_tail','')
assert 'LAB_ERROR' not in tail and 'LAB_TRUNCATED' not in tail
gate=re.search(r'GATE maxL2/row=([\d.eE+-]+)/([\d.eE+-]+) END seeds=(\d+)',tail)
assert gate and float(gate[1])<=.015 and float(gate[2])<=.075,'Unmeasured: missing or failed TASK gates'
arc=json.loads(re.search(r'^SEEDS (.*)$',tail,re.M)[1]);ns=arc['n'];assert ns==int(gate[3]) and ns>=4
values=struct.unpack('<'+'f'*(2*ns*28),base64.b85decode(arc['data']))
samples=[[[values[(mi*ns+s)*28+j] for j in range(28)] for s in range(ns)] for mi in range(2)]
tcrit={3:3.182446,4:2.776445,5:2.570582,6:2.446912,7:2.364624,8:2.306004,9:2.262157,10:2.228139,11:2.200985,12:2.178813,13:2.160369,14:2.144787,15:2.131450,16:2.119905,23:2.068658}.get(ns-1,1.96)
def summary(xs):
 mean=statistics.mean(xs);se=statistics.stdev(xs)/math.sqrt(len(xs))
 return dict(mean=mean,SE=se,CI95=[mean-tcrit*se,mean+tcrit*se],seeds=len(xs),samples=xs)
traffic=json.loads(re.search(r'^BYTES (.*)$',tail,re.M)[1]);result=dict(receipt=str(a.receipt),native_H200='result' in r,gate=dict(L2=float(gate[1]),row=float(gate[2])),arms=[],oracle=[],AA={},clocks=[l for l in tail.splitlines() if l.startswith('CLOCK ')],traffic=traffic)
for mi,m in enumerate((128,256)):
 rows=samples[mi]
 for i,name in enumerate(arc['names']):
  if not i:continue
  spec=dict(M=m,arm=name,vs_S15=summary([v[i] for v in rows]))
  if i>=2:spec['vs_CH4']=summary([v[7+i-2] for v in rows])
  spec['P4_point_nomination']=result['native_H200'] and i>=2 and min(spec['vs_S15']['mean'],spec['vs_CH4']['mean'])<1
  spec['P4_current_component_point_gain']=result['native_H200'] and i>=2 and spec['vs_CH4']['mean']<1
  result['arms'].append(spec)
 diffs=[v[13]-v[12] for v in rows];result['AA'][m]=dict(signed_mean_us=statistics.mean(diffs),absolute_mean_us=statistics.mean(abs(x) for x in diffs),maximum_absolute_seed_mean_us=max(abs(x) for x in diffs),ratio=summary([v[13]/v[12] for v in rows]))
 floors=[v[23]/4 for v in rows]
 # All components are conditional differences between matched read requests.
 # Cache reuse, changed issue/occupancy and oracle shell costs stay in them.
 for name,idx in (('S15',(0,1,2,3,7)),('CH4',(0,4,5,6,8))):
  o=[[v[14+j] for j in idx] for v in rows]
  components={'model_floor_4TB':summary(floors),'unique_stream':summary([v[0] for v in o]),'stream_minus_4TB_model':summary([v[0]-f for v,f in zip(o,floors)]),'repeated_weights_given_stream':summary([v[1]-v[0] for v in o]),'valid_X_given_repeats':summary([v[2]-v[1] for v in o]),'split_topology_given_requests':summary([v[3]-v[2] for v in o]),'full_math_shell_given_split':summary([v[4]-v[3] for v in o]),'whole':summary([v[4] for v in o]),'total_gap_above_4TB_model':summary([v[4]-f for v,f in zip(o,floors)])}
  result['oracle'].append(dict(M=m,control=name,floor_4TB_us=statistics.mean(floors),components=components,bytes_MB={n:summary([v[23+j] for v in rows]) for j,n in enumerate(('uniqueW','W32','X32','W64','X64'))},interpretation='Ordered interventions, not separable causal stalls. Repeated/X requests are logical, not measured DRAM bytes. B-X includes launch and branch fragmentation. R-B includes math, waits, real memory schedule, epilogue, route and finish. Different cache reuse and checksum output traffic are not subtracted.'))
(H/('results.json' if result['native_H200'] else 'local_results.json')).write_text(json.dumps(result,indent=2)+'\n')
for x in result['arms']:print(x['M'],x['arm'],{k:(v['mean'],v['CI95']) for k,v in x.items() if k.startswith('vs_')})
for x in result['oracle']:print('ORACLE',x['M'],x['control'],{k:round(v['mean'],3) for k,v in x['components'].items()})
