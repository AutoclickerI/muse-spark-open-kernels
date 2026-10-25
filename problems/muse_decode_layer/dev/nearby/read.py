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
    print('| case | S15 us | SGLang us | baseline/S15 | scored controls | nearby/control speedup |')
    print('|---|---:|---:|---:|---|---|')
    ranked=[]
    for name,row in rows.items():
        c=specs[name];q=c['q_len'];b=c['batch'];layer=c['layer'][0]
        bs=(1,16,32);below=max((x for x in bs if x<=b),default=1);above=min((x for x in bs if x>=b),default=32)
        controls=[f'scored-{"d" if q==1 else "v"}{x}-{layer}' for x in sorted({below,above})]
        ratios=[row[3]/rows[k][3] for k in controls if len(row)==6 and k in rows and len(rows[k])==6]
        ranked.append((min(ratios,default=1),name,row,controls,ratios))
    for _,name,row,controls,ratios in sorted(ranked):
        if len(row)!=6:
            print('|',name,'| baseline missing |');continue
        print(f'| {name} | {row[1]:.2f} | {row[2]:.2f} | {row[3]:.4f} | {", ".join(controls)} | '+
              ', '.join(f'{v:.4f}' for v in ratios)+' |')
    print('Local sm120 timings do not predict H200. No review cutoff is assumed; read TASK gates separately.')


if __name__=='__main__':main()
