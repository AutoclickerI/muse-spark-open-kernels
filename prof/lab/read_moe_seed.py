#!/usr/bin/env python3
"""Render the native seed lab's landed receipt. Never queries the judge."""
import argparse
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]


def read(ref):
    p=Path(ref)
    if not p.exists():
        hits=sorted(ROOT.glob(f'runs/*/*_{ref}.json'))
        if not hits:raise FileNotFoundError(f'No landed receipt for {ref}')
        p=hits[-1]
    doc=json.loads(p.read_text());tail=doc.get('result',{}).get('log_tail','')
    if 'LAB_ERROR' in tail or 'LAB_END' not in tail:raise ValueError('Failed or incomplete lab')
    agg=[];r0=[]
    for line in tail.splitlines():
        if m:=re.match(r'AGG (\S+) M=(\d+) seeds=(\d+) mean=([\d.]+) min=([\d.]+) max=([\d.]+) max/min=([\d.]+) max/mean=([\d.]+)',line):
            name,M,n,mean,lo,hi,spread,tailratio=m.groups();agg.append(dict(kernel=name,M=int(M),seeds=int(n),mean_us=float(mean),min_us=float(lo),max_us=float(hi),max_min=float(spread),max_mean=float(tailratio)))
        if m:=re.match(r'T (\S+) M=(\d+) R0 us=([\d.]+) ratio=([\d.]+) fp32=([\d.]+) (PASS|REJECT)',line):
            name,M,us,ratio,err,status=m.groups();r0.append(dict(kernel=name,M=int(M),us=float(us),ratio=float(ratio),fp32=float(err),passed=status=='PASS'))
    return dict(job=doc.get('id'),receipt=str(p),aggregate=agg,R0=r0,budget_lines=[l for l in tail.splitlines() if l.startswith('BUDGET')])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('receipt');ap.add_argument('--json',type=Path);a=ap.parse_args();r=read(a.receipt)
    print('| Kernel | M | Seeds | Mean us | Max us | Max/min | Max/mean |\n|---|---:|---:|---:|---:|---:|---:|')
    for x in r['aggregate']:print(f'| {x["kernel"]} | {x["M"]} | {x["seeds"]} | {x["mean_us"]:.1f} | {x["max_us"]:.1f} | {x["max_min"]:.3f} | {x["max_mean"]:.3f} |')
    for x in r['R0']:print('R0',x)
    if a.json:a.json.write_text(json.dumps(r,indent=2)+'\n')
