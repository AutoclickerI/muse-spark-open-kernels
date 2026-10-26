"""Descriptive per-seed depth3/depth2 ratios normalized to separate paired BASE timings."""
import json,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;r=json.loads((D/'results.json').read_text());rec=r['seed_level_record'];out=[]
for kind in ('SINGLE','HALF','DOUBLE'):
 for m in (8,16,32):
  vals=[]
  for seed in range(16):
   def value(depth):return next(v[6]/1e6 for v in rec['rows'] if v[0]==seed and v[1]==m and rec['arm_names'][v[2]]==kind+str(depth) and v[3]==0)
   vals.append(value(3)/value(2))
  mean=statistics.mean(vals);se=statistics.stdev(vals)/4;out.append(dict(kind=kind,M=m,ratio=mean,SE=se,CI95=[mean-2.131449545559323*se,mean+2.131449545559323*se],seeds=16,seed_ratios=vals))
result=dict(job=r['job'],kind='BASE-normalized depth contrast;not directly timed depth3/depth2 pairs',rows=out)
(D/'depth_contrast.json').write_text(json.dumps(result,indent=2)+'\n')
for v in out:print(v['kind'],v['M'],round(v['ratio'],6),[round(x,6) for x in v['CI95']])
