"""Whole sliding-layer seed CIs and producer-context clocks; no stage or per-load significance veto."""
import argparse,base64,json,lzma
from pathlib import Path
from analyze import stats
H=Path(__file__).resolve().parent

def analyze(path,global_merge=False):
    receipt=json.loads(path.read_text());marker='SLIDING_MERGE_XZ85' if global_merge else 'SLIDING_EXP_XZ85'
    blob=next(s.split(' ',1)[1] for s in receipt['result']['log_tail'].splitlines() if s.startswith(marker+' '))
    d=json.loads(lzma.decompress(base64.b85decode(blob)));rows=[];candidate='GLOBAL' if global_merge else 'EXP'
    for r in d['results']:
        n=len(r['seeds']);assert all(v==n for v in r['TASK'].values()) and all(c['TASK'] for c in r['clocks'])
        v=r['times_ns'];ratios={a+'/'+b:stats([x/y for x,y in zip(v[a],v[b])]) for a,b in [('INC2','INC'),('RAW','INC'),(candidate,'INC'),(candidate,'RAW'),('A','INC')]}
        changed={'RAW':(r['B'],r['L'])==(16,8),'A':(r['B'],r['L'])!=(16,8),candidate:True}
        nominations=[dict(arm=a,control='INC',**ratios[a+'/INC']) for a in (candidate,'RAW','A') if changed[a] and ratios[a+'/INC']['mean']<1]
        rows.append(dict(B=r['B'],L=r['L'],seed_ids=r['seeds'],ratios=ratios,point_nominations=nominations,
            source_scope='Exact S15 other stages. INC=D8d16/D4d32/FA3v16/RAW2v32. Ratios are whole sliding-layer times, not attention-only or six-load score.',
            critical=r['critical'],bounds=r['bounds'],resources=r['resources'],clocks=r['clocks']))
    out=dict(job=receipt['id'],receipt=str(path),elapsed_s=d['elapsed_s'],TASK_complete=True,layer_TASK=sum(sum(r['TASK'].values()) for r in d['results']),producer_attention_TASK=sum(len(r['clocks']) for r in d['results']),
             rows=rows,sources=d['sources'],s15_sha256=d['s15_sha256'],gate='Every changed per-shape point below1 to P4 with uncertainty; only P4 held-out six-load G0 confirms a composition.')
    (H/('merge_result.json' if global_merge else 'exp_result.json')).write_text(json.dumps(out,indent=2)+'\n')
    for r in rows:
        print(r['B'],r['L'],'control',r['ratios']['INC2/INC'],'candidate',r['ratios'][candidate+'/INC'],'nominations',r['point_nominations'])
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--global-merge',action='store_true');a=p.parse_args();analyze(a.receipt,a.global_merge)
