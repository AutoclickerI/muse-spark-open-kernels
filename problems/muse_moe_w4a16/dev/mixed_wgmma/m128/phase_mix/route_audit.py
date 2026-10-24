"""Recompute route work counts; distinguish counts from measured bandwidth."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    source=ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/seed_sweep.json';data=json.loads(source.read_text());rows=[]
    for x in data['rows']:
        if x['M']!=128:continue
        c=x['counts'];n8=sum((n+7)//8 for n in c);n32=sum((n+31)//32 for n in c)
        nh=sum((n+7)//8 if n<=16 else (n+31)//32 for n in c)
        p8=n8*8;ph=sum(((n+7)//8*8 if n<=16 else (n+31)//32*32) for n in c)
        rows.append(dict(seed=x['seed'],case=x['case'],counts=c,tokens=sum(c),NB8=n8,NB32=n32,mix_groups=nh,
            group_reduction=1-nh/n8,padding_extra_vs8=ph/p8-1,padding_vs_actual=ph/sum(c)-1,padded8=p8,padded32=n32*32,padded_mix=ph))
    summary={k:dict(mean=statistics.mean(v[k] for v in rows),min=min(v[k] for v in rows),max=max(v[k] for v in rows))
             for k in ('group_reduction','padding_extra_vs8','padding_vs_actual')}
    result=dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        distinct_seeds=len(set(r['seed'] for r in rows)),row_count=len(rows),rows=rows,summary=summary,
        aggregate_group_reduction=1-sum(r['mix_groups'] for r in rows)/sum(r['NB8'] for r in rows),
        aggregate_extra_padding=sum(r['padded_mix'] for r in rows)/sum(r['padded8'] for r in rows)-1,
        interpretation='Review range35-50% does not hold for every row; mean35.4%, range25.8-48.6. Padding extra is relative to eight-token padded work, not actual tokens. Neither metric measures HBM traffic or wall time.')
    (HERE/'route_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
