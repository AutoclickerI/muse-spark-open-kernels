"""Land the full H200 archive and paired per-load/per-seed uncertainty."""
import base64
import json
import lzma
import math
from pathlib import Path
import re
import statistics
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def stats(values):
    return dict(mean=statistics.mean(values),se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0.,n=len(values))


def main():
    arg=sys.argv[1];p=Path(arg)
    if not p.exists():p=sorted(REPO.glob('runs/*/*_'+arg+'.json'))[-1]
    receipt=json.loads(p.read_text());tail=receipt['result'].get('log_tail','')
    match=re.search(r'PDL_AUDIT_XZ85 (\S+)',tail)
    assert match,'no complete PDL archive in receipt: '+tail[-1500:]
    data=json.loads(lzma.decompress(base64.b85decode(match[1])))
    cells={c['case']:c for c in data['cells']}
    good=all(not c['gates'] and c['seeds']==16 for c in cells.values()) and len(cells)==12
    data['all_gates_clean_16_seeds']=good
    names=[name for name in next(iter(cells.values()))['samples'][0] if name!='A']
    load_samples={};scores={name:[] for name in names}
    for load in ('d1','d16','d32','v1','v16','v32'):
        if not all(load+k in cells for k in ('S','F')):continue
        a,b=cells[load+'S']['samples'],cells[load+'F']['samples']
        load_samples[load]=[{name:(46*s[name]+16*f[name])/(46*s['A']+16*f['A']) for name in names}
            for s,f in zip(a,b)]
        print('LOAD',load, {name:stats([s[name] for s in load_samples[load]]) for name in names})
    if len(load_samples)==6:
        for i in range(min(map(len,load_samples.values()))):
            for name in scores:scores[name].append(math.prod(v[i][name] for v in load_samples.values())**(-1/6))
    data['per_seed_loads']=load_samples;data['per_seed_scores']=scores
    data['score_statistics']={k:stats(v) for k,v in scores.items() if v}
    data['job']=receipt.get('job',arg);data['receipt']=str(p.relative_to(REPO))
    data['host']='muse_attention_decode';data['host_result_status']=receipt['result'].get('status')
    out=HERE/('h200_'+str(data['job'])+'.json');out.write_text(json.dumps(data,indent=2)+'\n')
    print('GATES',good,'SCORE',data['score_statistics']);print(out)


if __name__=='__main__':main()
