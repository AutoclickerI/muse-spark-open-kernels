"""Actual bank reader end-to-end; margin plus 30-request stock-spread gate."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/data2/jihoon/muse_kernel/harness')


def main():
    p=argparse.ArgumentParser();p.add_argument('--tree',type=Path,required=True);p.add_argument('--prefix',required=True)
    a=p.parse_args();out=DATA/'results'/f'{a.prefix}-evidence';out.mkdir(exist_ok=True)
    completed=[]
    for tp in (1,2):
        for kind,roles in [('margin',['stock','reader']),('random',['stock-a','stock-b','stock-c','reader'])]:
            reference=f'{a.prefix}-random-stock-a-tp{tp}'
            for role in roles:
                label=f'{a.prefix}-{kind}-{role}-tp{tp}'
                cmd=[sys.executable,'serving/tests/wide_bank_harness.py','--tree','stock' if role.startswith('stock') else str(a.tree),
                     '--weights',kind,'--label',label,'--tp',str(tp),'--gpus','0' if tp==1 else '0,1',
                     '--levels','1,4,16,32' if kind=='margin' else '1,4',
                     '--convs','1:2,4:8,16:16,32:32' if kind=='margin' else '1:2,4:8','--turns','3']
                if kind=='random' and role!='stock-a':cmd+=['--agreement-reference',reference]
                print('START',label,flush=True)
                existing=DATA/'results'/label/'meta.json'
                if existing.exists():
                    meta=json.loads(existing.read_text())
                    assert not meta.get('failure') and not meta['validation_errors'],label
                    assert meta['provenance']==meta['provenance_end'],label
                    records=[json.loads(s) for s in (existing.parent/'requests.jsonl').read_text().splitlines()]
                    assert len(records)==(174 if kind=='margin' else 30)
                    assert all(r['n_out']==220 and len(r['output_ids'])==220 for r in records)
                else:
                    with (DATA/'logs'/f'{label}.log').open('w') as log:
                        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                completed.append(label);(out/'completed.json').write_text(json.dumps(completed,indent=2)+'\n')
                print('DONE',label,flush=True)
            if kind=='margin':
                subprocess.run([sys.executable,'serving/harness/compare.py',f'{a.prefix}-margin-stock-tp{tp}',
                    f'{a.prefix}-margin-reader-tp{tp}','--require-exact','--json',str(out/f'margin_tp{tp}.json')],
                    cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
            else:
                stocks=[f'{a.prefix}-random-stock-{v}-tp{tp}' for v in ('a','b','c')]
                patch=f'{a.prefix}-random-reader-tp{tp}'
                cmd=[sys.executable,'serving/harness/agreement_metric.py','--patch',patch,
                     '--forced-patch',str(DATA/'results'/patch/'forced'),'--json',str(out/f'random_gate_tp{tp}.json')]
                for stock in stocks:cmd+=['--stock',stock,'--forced-stock',str(DATA/'results'/stock/'forced')]
                with (out/f'random_gate_tp{tp}.log').open('w') as log:
                    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('WIDE REAL READER ALL LOCAL AGREEMENT GATES PASS',flush=True)


if __name__=='__main__':main()
