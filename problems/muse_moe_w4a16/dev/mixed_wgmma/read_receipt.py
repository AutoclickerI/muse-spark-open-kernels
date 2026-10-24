"""Render landed H200 lab rows. Reads local receipts only; never calls the judge API."""
import argparse
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[4]
ROW=re.compile(r'^T (\S+) M=(\d+) us=(\S+) (V9|Marlin)=(\S+) ratio=(\S+) fp32=(\S+) row=(\S+) (PASS|REJECT)$',re.M)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',nargs='+',help='Local receipt path or judge job ID')
    args=ap.parse_args()
    for ref in args.receipt:
        path=Path(ref)
        if not path.exists():
            hits=sorted(ROOT.glob(f'runs/*/*_{ref}.json'))
            if not hits:
                print(f'{ref}: no local receipt; pending (no API request made)')
                continue
            path=hits[-1]
        receipt=json.loads(path.read_text());result=receipt['result']
        assert 'H200' in str(result.get('gpu','')),result.get('gpu')
        tail=result.get('log_tail','')
        assert re.search(r'^LAB_BEGIN moe_mixed_wgmma(?:_planes|_prefill)? ',tail,re.M),f'No mixed WGMMA lab in {path}'
        print(f"Job {receipt['id']}; GPU {result['gpu']}; host status {result.get('status')}")
        print('\n| M | Variant | us | Control | control us | speedup | FP32 L2 | row | gate |')
        print('|---:|---|---:|---|---:|---:|---:|---:|---|')
        winners={}
        rows=sorted(ROW.findall(tail),key=lambda r:(int(r[1]),r[0]))
        for label,m,us,base,bu,ratio,err,row,gate in rows:
            m=int(m);us,bu,ratio,err,row=map(float,(us,bu,ratio,err,row))
            valid=gate=='PASS' and all(map(math.isfinite,(us,bu,ratio,err,row))) and err<=.01 and row<=.05
            print(f'| {m} | {label} | {us:.1f} | {base} | {bu:.1f} | {1/ratio:.3f}x | {err:.5f} | {row:.5f} | {"PASS" if valid else "REJECT"} |')
            if valid and label.startswith(('RS','SS','PRS','plane')) and (m not in winners or us<winners[m][0]):winners[m]=(us,label)
        print('\nFP32-admitted new producer minima: '+str(winners))
        for line in tail.splitlines():
            if line.startswith(('ERROR ','LAB_ERROR','BUDGET ','ROUTE ','END ')):print(line)
        missing={128,256,512,753,3012,16384}-{int(r[1]) for r in rows if r[0]!='Marlin'}
        if missing:print('Unmeasured M: '+','.join(map(str,sorted(missing))))


if __name__=='__main__':main()
