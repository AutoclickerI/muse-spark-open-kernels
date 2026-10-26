from pathlib import Path
import re,json,datetime
H=Path(__file__).resolve().parent;R=H.parents[3];ad=json.loads((H/'reduce_admission.json').read_text())
import sys
sys.path.insert(0,str(R/'problems/muse_decode_layer/dev/p12_components'))
from own_receipts import receipts
ledger=receipts();hits=[x for x in ledger if x['sha256']==ad['wrapper_sha256']]
out=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='pending',rows=[])
if hits:
    row=hits[-1];f=R/Path(row['source']).with_suffix('.json');r=json.loads(f.read_text()).get('result') or {};tail=r.get('log_tail','')
    match=re.search(r'^LAB_BEGIN p12_projection_reduce_lab .*?\n(.*?)(?:^LAB_END.*?$|\Z)',tail,re.M|re.S);body=match[1] if match else tail
    out.update(job=row['job'],receipt=str(f.relative_to(R)),error_lines=[x for x in body.splitlines() if any(y in x for y in ('LAB_ERROR','INCOMPLETE','TRUNCATED','Error'))])
    for m in re.finditer(r'^R (\d+):(\S+) A=([\d.]+) B=([\d.]+) P=([\d.]+) r=([\d.]+)\[([\d.]+),([\d.]+)\] p/a=([\d.]+) aa=([\d.]+) e=([\deE+.-]+) probe=([\deE+.-]+)',body,re.M):
        t,role,*v=m.groups();a,b,p,ratio,lo,hi,pr,aa,e,probe=map(float,v)
        out['rows'].append(dict(t=int(t),role=role,A_us=a,B_us=b,producer_us=p,ratio=ratio,ratio_min=lo,ratio_max=hi,producer_ratio=pr,aa_max=aa,error=e,probe=probe))
    valid=len(out['rows'])==30 and 'END complete=30/30' in body and not out['error_lines'];out['status']='measured' if valid else 'unmeasured'
    out['component_candidate']=valid and any(r['ratio']<1-max(.02,2*r['aa_max']) for r in out['rows']) and all(r['ratio']<=1+max(.02,r['aa_max']) for r in out['rows'])
(H/'reduce_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out.get('job'),len(out['rows']),out.get('component_candidate'))
