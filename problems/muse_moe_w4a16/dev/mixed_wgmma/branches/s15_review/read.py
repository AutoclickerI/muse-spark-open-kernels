"""Recompute shape point nominations and SE from retained independent seed means."""
import argparse
import base64
import json
import math
from pathlib import Path
import re
import statistics
import struct
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--output',type=Path,default=H/'results.json');a=p.parse_args()
receipt=json.loads(a.receipt.read_text());tail='\n'.join(receipt['lines']) if 'lines' in receipt else receipt['result'].get('log_tail','')
gate=re.search(r'GATE L2/row=([\d.eE+-]+)/([\d.eE+-]+) END seeds=(\d+)',tail)
assert gate and float(gate[1])<=.015 and float(gate[2])<=.075 and 'LAB_ERROR' not in tail and 'LAB_TRUNCATED' not in tail,'Unmeasured: missing/failed gates or incomplete transport'
archive=json.loads(re.search(r'^SEEDS (.*)$',tail,re.M)[1]);names=archive['names'];ns=archive['seeds'];na=len(names)
assert ns==int(gate[3])
values=struct.unpack('<'+'d'*(2*ns*(na+2)),base64.b85decode(archive['data']))
ratios=[[[values[(mi*ns+seed)*na+i] for i in range(na)] for seed in range(ns)] for mi in range(2)]
offset=2*ns*na;aa=[[[values[offset+(mi*ns+seed)*2+i] for i in (0,1)] for seed in range(ns)] for mi in range(2)]
native='result' in receipt
results=[]
for mi,m in enumerate((128,256)):
 for i,name in enumerate(names):
  rs=[r[i] for r in ratios[mi]];mean=statistics.mean(rs);se=statistics.stdev(rs)/math.sqrt(ns) if ns>1 else None
  results.append(dict(M=m,arm=name,ratio=mean,SE=se,seeds=ns,fixtures_per_seed=3,nominate_to_P4=native and name!='S15' and mean<1,seed_ratios=rs))
out=dict(receipt=str(a.receipt),native_H200=native,gate=dict(L2=float(gate[1]),row=float(gate[2])),results=results,AA_us=aa,decision='Every qualified per-shape point gain is nominated; A:A is uncertainty, not a veto. P4 held-out six-load geomean promotes.')
out['readiness_vs_matched_grid']=[]
for m in (128,256):
 ready=next(r for r in results if r['M']==m and r['arm']=='READY')
 serial=next(r for r in results if r['M']==m and r['arm']=='READY_SERIAL')
 rs=[x/y for x,y in zip(ready['seed_ratios'],serial['seed_ratios'])]
 out['readiness_vs_matched_grid'].append(dict(M=m,ratio=statistics.mean(rs),SE=statistics.stdev(rs)/math.sqrt(ns) if ns>1 else None,seeds=ns,meaning='Ratio of same-seed S15-normalized means; exploratory matched-grid contrast, not direct READY/SERIAL ABBA timing.'))
a.output.write_text(json.dumps(out,indent=2)+'\n')
for r in results:print(r['M'],r['arm'],'ratio',r['ratio'],'SE',r['SE'],'seeds',ns,'P4',r['nominate_to_P4'])
