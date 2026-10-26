"""Preserve S15 audit gates and ratios without inventing missing seed covariance."""
import argparse
import base64
import hashlib
import json
import lzma
import math
import re
import statistics
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]


def parse(path):
    receipt=json.loads(path.read_text());tail=receipt.get('result',{}).get('log_tail','')
    found=[json.loads(x) for x in re.findall(r'^RESULT (\{.*\})$',tail,re.M)]
    found += [json.loads(lzma.decompress(base64.b85decode(x))) for x in re.findall(r'^RESULT_XZ85 (\S+)$',tail,re.M)]
    found=[x for x in found if 'loads' in x and 'shapes' in x]
    if len(found)!=1:raise ValueError(('expected one audit result',len(found),tail[-1200:]))
    result=found[0];loads=result['loads'];arms=sorted(set.intersection(*(set(x) for x in loads.values()))) if loads else []
    point={name:math.exp(-sum(math.log(v[name][0]) for v in loads.values())/len(loads)) for name in arms}
    controls={load:dict(mean=row['A2'][0],se=row['A2'][1],min=row['A2'][2],max=row['A2'][3],
        unity_inside_seed_range=row['A2'][2]<=1<=row['A2'][3]) for load,row in loads.items()}
    paired={}
    seed_loads=result.get('seed_load_us')
    if seed_loads is None and 'seed_us' in result:
        v=result['seed_us'];seed_loads={}
        for b,l in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
            seed_loads[f'{b}x{l}']=[{name:(46*s[name]+16*f[name])/62 for name in s}
                for s,f in zip(v[f'{b}x{l}s'],v[f'{b}x{l}f'])]
    if seed_loads is not None:
        assert len(seed_loads)==6 and all(len(v)==8 for v in seed_loads.values())
        for name in arms:
            logs=[sum(math.log(v[i]['A']/v[i][name]) for v in seed_loads.values())/6 for i in range(8)]
            m=statistics.mean(logs);se=statistics.stdev(logs)/math.sqrt(8)
            paired[name]=dict(R=math.exp(m),log_SE=se,lower95=math.exp(m-2.364624251*se),
                upper95=math.exp(m+2.364624251*se),seed_R=list(map(math.exp,logs)))
    return dict(receipt=str(path.relative_to(R)),receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        host_status=receipt.get('status'),host_passed=receipt.get('result',{}).get('passed'),result=result,
        six_load_inverse_geomean_of_mean_ratios=point,
        controls=controls,
        unmeasured_loads_due_to_control_drift=[load for load,c in controls.items() if not c['unity_inside_seed_range']],
        paired_six_load_score=paired,
        score_confidence=('Paired eight-seed six-load log-R interval uses Student t(7), two-sided95%; inspect TASK and A:A before promotion.' if paired else
            'Unavailable: this frozen payload omitted per-seed times. Mean ratios are descriptive; held-out per-seed six-load confirmation is required.'),
        correctness_passed=bool(result['complete'] and not result['errors'] and not result['failures']),
        measured_loads=[load for load,c in controls.items() if c['unity_inside_seed_range']],
        measured=bool(result['complete'] and not result['errors'] and not result['failures']
                      and all(c['unity_inside_seed_range'] for c in controls.values())))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    d=parse(a.receipt.resolve())
    if a.out:a.out.write_text(json.dumps(d,indent=2)+'\n')
    print('measured',d['measured'],'gates',d['result']['failures'],'errors',d['result']['errors'])
    for load,row in d['result']['loads'].items():print(load,json.dumps(row,sort_keys=True))
    print('DESCRIPTIVE score ratios',d['six_load_inverse_geomean_of_mean_ratios'])
    if d['paired_six_load_score']:print('PAIRED held-out score',json.dumps(d['paired_six_load_score'],sort_keys=True))
    print(d['score_confidence'])


if __name__=='__main__':main()
