"""Decode joint samples from complete P10 H200 archives; keep failed runs explicit."""
import base64,hashlib,json,math,re,statistics,sys,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def interval(values):
    logs=[math.log(x) for x in values];mean=statistics.mean(logs)
    # All current structural batches have sixteen independent input seeds.
    assert len(logs)==16
    half=2.13145*statistics.stdev(logs)/math.sqrt(len(logs))
    return [math.exp(mean),math.exp(mean-half),math.exp(mean+half)]
def read(job,kind):
    files=list((R/'runs').glob('muse_*/*'+job+'.json'));assert len(files)==1,files
    p=files[0];raw=json.loads(p.read_text());tail=raw['result'].get('log_tail','')
    pattern='p10_projection_followup' if kind=='followup' else 'p10_qkv_dispatch'
    body=re.search(r'^LAB_BEGIN '+pattern+r' .*?\n(.*?)^LAB_END',tail,re.M|re.S)
    assert body is not None,'Missing lab, not a measured loss'
    body=body[1];assert not any(x in body for x in ('LAB_ERROR','INCOMPLETE','TRUNCATED'))
    end='END complete=o1:3,large:20' if kind=='followup' else 'END complete=24/24'
    assert end in body,'Incomplete batch, not a measured loss'
    records=json.loads(zlib.decompress(base64.b85decode(re.search(r'^DATA (.+)',body,re.M)[1])))
    grouped={}
    for row in records:grouped.setdefault((row['t'],row['role']),[]).append(row)
    winners=[]
    for (t,role),rows in grouped.items():
        control=next(x for x in rows if x['arm']=='A')
        for row in rows:
            ratios=[b/a for a,b in zip(control['us'],row['us'])]
            row['ratio']=interval(ratios);row['control_us']=control['us']
            row['mean_us']=statistics.mean(row['us'])
            # Every valid point win is a nomination. Only P4 judges G0.
            row['point_nomination']=row['arm'] not in ('A','AA') and row['ratio'][0]<1
            if row['point_nomination']:winners.append(row)
            print(t,role,row['arm'],' '.join(f'{x:.6f}' for x in row['ratio']))
    out=dict(job=raw['id'],receipt=str(p.relative_to(R)),receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),valid=True,outer_host_passed=raw['result']['passed'],records=records,point_nominations=winners,ci='geomean and descriptive Student-t15 interval over 16 paired log ratios; stage selection, not whole-layer G0',limits=['one independent synthetic weight bank','shape points selected on these seeds need held-out P4 confirmation','A:A reported without a per-shape veto'])
    (H/(kind+'_results.json')).write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':
    read('646822ab3d164be8','followup')
    if len(sys.argv)>1:read(sys.argv[1],'qkv')
