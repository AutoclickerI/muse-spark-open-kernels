"""Alternating stock/patch brackets, p50 AND p90, full unchanged workload."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/data2/jihoon/muse_kernel/harness')
sys.path.insert(0,str(ROOT/'serving/harness'))
import compare


def run(label,tree,tp,native):
    existing=DATA/'results'/label/'meta.json'
    if existing.exists():
        meta=json.loads(existing.read_text())
        assert meta.get('provenance_end')==meta['provenance'] and not meta.get('failure')
        assert not meta['validation_errors']
        cmd=meta['server_cmd']
        assert float(cmd[cmd.index('--mem-fraction-static')+1])<=0.3
        print('REUSE VERIFIED',label,flush=True)
        return
    command=[sys.executable,'serving/tests/gpu0_ttft_harness.py','--with-logprob',
             '--tree',str(tree),'--weights','margin','--label',label,'--tp',str(tp),
             '--gpus','0' if tp==1 else '0,1','--levels','1,4,16,32',
             '--convs','1:2,4:8,16:16,32:32','--turns','3']
    if native: command.append('--native')
    print('START',label,flush=True)
    with (DATA/'logs'/f'{label}.log').open('w') as out:
        subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    meta=json.loads((DATA/'results'/label/'meta.json').read_text())
    assert not meta.get('failure') and not meta['validation_errors'],label
    assert meta['provenance']==meta['provenance_end'],label
    cmd=meta['server_cmd']
    assert float(cmd[cmd.index('--mem-fraction-static')+1])<=0.3
    print('COMPLETE',label,flush=True)


def paired(stock,patch):
    ma,a=compare.load(stock); mb,b=compare.load(patch)
    result=compare.compare(ma,a,mb,b)
    assert not result['problems'],result['problems']
    assert len(a)==len(b)==174 and all(a[k]['output_ids']==b[k]['output_ids'] for k in a)
    rows={}
    for c in (1,4,16,32):
        av=[v['ttft'] for v in a.values() if v['level']==c]
        bv=[v['ttft'] for v in b.values() if v['level']==c]
        rows[c]={'p50':statistics.median(bv)/statistics.median(av),
                 'p90':compare.quantile(bv,.9)/compare.quantile(av,.9)}
    return dict(stock=stock,patch=patch,exact=174,ratios=rows,
                margin_pass=all(v<=1.5 for row in rows.values() for v in row.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--tree',required=True,type=Path)
    p.add_argument('--prefix',required=True);args=p.parse_args()
    out=DATA/'results'/f'{args.prefix}-evidence';out.mkdir(exist_ok=True)
    completed=[];results=[]
    for tp in (1,2):
        labels={}
        # Each patch has an immediately preceding/following stock bracket.
        for role,native in [('stock-a',True),('native',True),('stock-b',True),
                            ('portable',False),('stock-c',True)]:
            label=f'{args.prefix}-{role}-tp{tp}'
            run(label,'stock' if role.startswith('stock') else args.tree,tp,native)
            labels[role]=label;completed.append(label)
            (out/'completed.json').write_text(json.dumps(completed,indent=2)+'\n')
        for mode,before,after in [('native','stock-a','stock-b'),('portable','stock-b','stock-c')]:
            for stock in (before,after):
                item=paired(labels[stock],labels[mode]);results.append(item)
                print(json.dumps(item),flush=True)
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('ALL AGREEMENT GATES PASS; TTFT MARGIN',all(r['margin_pass'] for r in results),flush=True)


if __name__=='__main__': main()
