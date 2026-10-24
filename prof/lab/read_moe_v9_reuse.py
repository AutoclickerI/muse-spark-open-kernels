#!/usr/bin/env python3
"""Read landed V9-reuse receipts once; make a table and agreement-gated winners. No API calls."""
import argparse
import hashlib
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ROW=re.compile(r'^T (\S+) M=(\d+) (R0|G) us=([\d.]+) V9=([\d.]+) ratio=([\d.]+) fp32=([\d.e+-]+) (PASS|REJECT)$',re.M)


def read(ref):
    p=Path(ref)
    if not p.exists():
        hits=sorted(ROOT.glob(f'runs/*/*_{ref}.json'))
        if not hits:raise FileNotFoundError(f'No landed receipt for {ref}; do not poll the judge')
        p=hits[-1]
    receipt=json.loads(p.read_text());tail=receipt['result'].get('log_tail','')
    begin=re.search(r'^LAB_BEGIN (\S+) (\S+)',tail,re.M)
    if not begin or 'LAB_END' not in tail or 'LAB_ERROR' in tail:raise ValueError('Missing or failed lab payload')
    job=receipt.get('id') or re.search(r'_([0-9a-f]{16})\.json$',str(p)).group(1)
    rows=[dict(kernel=k,M=int(m),routing=r,us=float(us),v9_us=float(v),ratio=float(ratio),fp32=float(error),passed=status=='PASS') for k,m,r,us,v,ratio,error,status in ROW.findall(tail)]
    if not rows:raise ValueError('No timing rows')
    winners={}
    for m in (128,256):
        for routing in ('R0','G'):
            good=[r for r in rows if r['M']==m and r['routing']==routing and r['passed']]
            if good:
                best=min(good,key=lambda r:r['ratio'])
                winners[f'{m}:{routing}']=best if best['ratio']<1 else dict(kernel='V9',M=m,routing=routing,us=best['v9_us'])
    return dict(job=job,payload=begin[1],payload_sha16=begin[2],receipt=str(p.relative_to(ROOT)),receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=rows,winners=winners)


def table(results):
    lines=['| Job | Kernel | M | Routing | us | V9 us | Ratio | FP32 rel |','|---|---|---:|---|---:|---:|---:|---:|']
    for result in results:
        for r in result['rows']:
            lines.append(f'| {result["job"][:8]} | {r["kernel"]} | {r["M"]} | {r["routing"]} | {r["us"]:.1f} | {r["v9_us"]:.1f} | {r["ratio"]:.4f} | {r["fp32"]:.5f} |')
    return '\n'.join(lines)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('receipts',nargs='+');ap.add_argument('--json',type=Path);args=ap.parse_args()
    results=[read(r) for r in args.receipts]
    print(table(results))
    for r in results:print('WINNERS',r['job'],json.dumps(r['winners'],sort_keys=True))
    if args.json:args.json.write_text(json.dumps(results,indent=2)+'\n')
