"""Preserve held-out per-seed spread and six-load paired confirmation."""
import base64
import json
import lzma
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
job=sys.argv[1]
receipt,=REPO.glob(f'runs/*/*_{job}.json')
raw=json.loads(receipt.read_text());lines=raw['result']['log_tail'].splitlines()
encoded=next(l.split(' ',1)[1] for l in lines if l.startswith('GP_CONFIRM_XZ85 '))
x=json.loads(lzma.decompress(base64.b85decode(encoded)))
valid=len(x['cells'])==12 and all(r['seeds']==16 and not r['gates'] for r in x['cells'])
by_case={r['case']:r for r in x['cells']};spread={}
if valid:
    for arm in ('A2','B'):
        seed_speedups=[]
        for i in range(16):
            ratios=[]
            for load in ('d1','d16','d32','v1','v16','v32'):
                a,b=(by_case[load+w]['samples'][i] for w in ('S','F'))
                ratios.append((46*a[arm]+16*b[arm])/(46*a['A']+16*b['A']))
            seed_speedups.append(math.prod(ratios)**(-1/6))
        spread[arm]=dict(mean=statistics.mean(seed_speedups),se=statistics.stdev(seed_speedups)/4,
            min=min(seed_speedups),max=max(seed_speedups),samples=seed_speedups)
out=dict(job=job,valid=valid,base_sha256='958f1258',candidate_sha256='08e82efd',
    loads=x['loads'],six_load_speedup=math.prod(v['B'] for v in x['loads'].values())**(-1/6),
    seed_speedups=spread,cells=x['cells'],receipt=str(receipt.relative_to(REPO)))
path=HERE/f'h200_{job}.json';path.write_text(json.dumps(out,indent=2)+'\n')
print('valid',valid)
for k,v in out['loads'].items():print(k,' '.join(f'{n}={r:.7f}' for n,r in v.items()))
print('six-load speedup',out['six_load_speedup'])
for n,v in spread.items():print('spread',n,{k:x for k,x in v.items() if k!='samples'})
print(path)
