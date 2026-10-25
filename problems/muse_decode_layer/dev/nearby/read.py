"""Decode the exact audit receipt and rank loss of baseline-relative speedup."""
import argparse
import json
import re
from pathlib import Path
from cases import cases


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args()
    raw=json.loads(a.receipt.read_text())
    if 'result' in raw and isinstance(raw['result'],dict) and 'rows' in raw['result']:d=raw['result']
    else:
        tail=raw['result'].get('log_tail','')
        found=re.findall(r'^RESULT (\{.*\})$',tail,re.M)
        matching=[json.loads(s) for s in found if '"shape_count":65' in s]
        if len(matching)!=1:raise ValueError('expected one 65-shape nearby result')
        d=matching[0]
    print('complete:',d['complete'],'baseline prepared:',not d['baseline_error'],'errors:',d['errors'])
    specs={c['label']:c for c in cases()};rows={r[0]:r for r in d['rows']}
    print('| case | S15 us | S15n us | SGLang us | N/S15 | SGLang/S15n |')
    print('|---|---:|---:|---:|---:|---:|')
    ranked=[]
    for row in d['rows']:
        if len(row)!=7:raise ValueError('expected S15/S15n/SGLang three-way row')
        name,old,new,base,r0,r1,ok=row
        ranked.append(((r0+r1)/2,name,old,new,base,ok))
    for ratio,name,old,new,base,ok in sorted(ranked):
        print(f'| {name} | {old:.2f} | {new:.2f} | {base:.2f} | {ratio:.5f} | {base/new:.4f} |'+(' FAIL' if not ok else ''))
    print('Local sm120 timings do not predict H200. No review cutoff is assumed; read TASK gates separately.')


if __name__=='__main__':main()
