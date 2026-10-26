from pathlib import Path
import re,json,datetime
H=Path(__file__).resolve().parent;R=H.parents[3];wrapper=json.loads((H/'point_retry_admission.json').read_text())['wrapper_sha256']
ledger=[json.loads(l) for l in (R/'runs/ledger.jsonl').read_text().rsplit('\n',1)[0].splitlines()];hits=[x for x in ledger if x['sha256']==wrapper]
out=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='pending',rows=[])
if hits:
    row=hits[-1];f=R/Path(row['source']).with_suffix('.json');r=json.loads(f.read_text()).get('result') or {};tail=r.get('log_tail','')
    m=re.search(r'^LAB_BEGIN p10_point_retry .*?\n(.*?)(?:^LAB_END.*?$|\Z)',tail,re.M|re.S);body=m[1] if m else tail
    out.update(job=row['job'],receipt=str(f.relative_to(R)),error_lines=[x for x in body.splitlines() if any(y in x for y in ('LAB_ERROR','INCOMPLETE','TRUNCATED','Error'))])
    for m in re.finditer(r'^POINT (\d+):(\S+) A=([\d.]+) scalar=([\d.]+) group=([\d.]+) rS=([\d.]+) rG=([\d.]+)\[([\d.]+),([\d.]+)\] aa=([\d.]+) e=([\deE+.-]+) p=([\deE+.-]+)',body,re.M):
        t,role,*v=m.groups();a,s,b,rs,ratio,lo,hi,aa,e,probe=map(float,v)
        out['rows'].append(dict(t=int(t),role=role,A_us=a,scalar_us=s,B_us=b,scalar_ratio=rs,ratio=ratio,ratio_min=lo,ratio_max=hi,aa_max=aa,error=e,probe=probe))
    out['status']='measured' if len(out['rows'])==20 and 'POINT_END complete=20/20' in body and not out['error_lines'] else 'unmeasured'
    out['route_gate']=next((x for x in body.splitlines() if x.startswith('POINT_ROUTE ')),None)
(H/'point_retry_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out.get('job'),len(out['rows']))
