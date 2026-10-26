"""Nearby/scored SGLang-relative speedup, grouped by the same nearest B bucket.

All 53 nearby cases are included. Each case is assigned to its nearest scored B
at the same q_len; ties choose the smaller B, matching S15n. Sliding/full weights
are 46/16. This is a diagnostic aggregation, not the official six-load score.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from cases import cases
H=Path(__file__).resolve().parent;R=H.parents[3]


def parse(path):
    rec=json.loads(path.read_text());r=rec.get('result',{})
    if 'rows' not in r:
        matches=[json.loads(x) for x in re.findall(r'^RESULT (\{.*\})$',r.get('log_tail',''),re.M)]
        matches=[x for x in matches if x.get('shape_count')==65]
        if len(matches)!=1:raise ValueError('expected one 65-shape result')
        r=matches[0]
    assert not r['baseline_error'],r['baseline_error']
    assert len(r['rows'])==65 and r['complete'] and not r['errors'],r.get('errors')
    specs={c['label']:c for c in cases()};rows={}
    for row in r['rows']:
        assert len(row)==7 and row[-1],row
        label,old,new,base,r0,r1,ok=row
        assert min(old,new,base)>0,row
        rows[label]=dict(s15=old,s15n=new,sglang=base,paired_n_over_a=[r0,r1])
    loads={};per_case=[]
    for q in (1,8):
        for b in (1,16,32):
            scored=[n for n,c in specs.items() if n.startswith('scored-') and c['q_len']==q and c['batch']==b]
            near=[n for n,c in specs.items() if not n.startswith('scored-') and c['q_len']==q
                  and min((1,16,32),key=lambda anchor:abs(c['batch']-anchor))==b]
            def speed(names,arm):
                w=lambda n:46 if specs[n]['layer']=='sliding' else 16
                return sum(w(n)*rows[n]['sglang'] for n in names)/sum(w(n)*rows[n][arm] for n in names)
            aa=[x for n in scored for x in rows[n]['paired_n_over_a']]
            cell=dict(scored_cases=scored,nearby_cases=near,dispatch_identical_control=dict(
                mean=sum(aa)/len(aa),min=min(aa),max=max(aa)),arms={})
            for arm in ('s15','s15n'):
                s,n=speed(scored,arm),speed(near,arm)
                cell['arms'][arm]=dict(scored_speedup=s,nearby_speedup=n,nearby_over_scored=n/s)
            loads[f'{"decode" if q==1 else "verify"} {b}']=cell
    for name,c in specs.items():
        if name.startswith('scored-'):continue
        b=min((1,16,32),key=lambda anchor:abs(c['batch']-anchor));q=c['q_len']
        label=f'scored-{"d" if q==1 else "v"}{b}-{c["layer"][0]}'
        per_case.append(dict(label=name,anchor=label,**{arm:(rows[name]['sglang']/rows[name][arm])/
            (rows[label]['sglang']/rows[label][arm]) for arm in ('s15','s15n')}))
    return dict(job=rec.get('id',path.stem.split('_')[-1]),receipt=str(path.relative_to(R)),
        receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),complete=True,loads=loads,per_case=per_case,
        caveats=['nearby aggregate uses all53 cases grouped by nearest scored batch; context/page axes retained',
                 'native lab gates are schema/finite only; native TASK correctness still needs an independent check',
                 'scored S15/S15n rows are an identical-dispatch control, not an independently prepared A:A arm'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    r=parse(a.receipt.resolve())
    if a.out:a.out.write_text(json.dumps(r,indent=2)+'\n')
    print('| load | S15 scored | S15 nearby | nearby/scored | S15n scored | S15n nearby | nearby/scored | identical dispatch min/max |')
    print('|---|---:|---:|---:|---:|---:|---:|---|')
    for name,c in r['loads'].items():
        a,n=c['arms']['s15'],c['arms']['s15n'];aa=c['dispatch_identical_control']
        print('| '+name+' | '+' | '.join(f'{v:.4f}' for d in (a,n) for v in d.values())+f' | {aa["min"]:.4f}/{aa["max"]:.4f} |')


if __name__=='__main__':main()
