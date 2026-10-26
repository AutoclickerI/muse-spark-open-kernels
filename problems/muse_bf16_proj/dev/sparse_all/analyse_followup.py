"""Native and grouped sections have independent completion/correctness gates."""
from pathlib import Path
import re,json,datetime
H=Path(__file__).resolve().parent;R=H.parents[3];sha=json.loads((H/'combined_manifest.json').read_text())['payload_sha256']
wrapper=json.loads((H/'combined_admission.json').read_text())['wrapper_sha256']
ledger=[json.loads(l) for l in (R/'runs/ledger.jsonl').read_text().rsplit('\n',1)[0].splitlines()];hits=[x for x in ledger if x['sha256']==wrapper]
out=dict(payload_sha256=sha,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='pending',native=[],point=[],scans=[])
if hits:
    row=hits[-1];file=R/Path(row['source']).with_suffix('.json');d=json.loads(file.read_text());r=d.get('result') or {};tail=r.get('log_tail','')
    found=re.search(r'^LAB_BEGIN p10_coding_structural .*?\n(.*?)(?:^LAB_END.*?$|\Z)',tail,re.M|re.S);body=found[1] if found else tail
    out.update(job=row['job'],receipt=str(file.relative_to(R)),host_passed=bool(r.get('passed')),error_lines=[x for x in body.splitlines() if any(y in x for y in ('ERROR','Error','INCOMPLETE','TRUNCATED','Traceback'))])
    native_complete='END complete=10/10' in body and 'LAB_TRUNCATED' not in body
    point_complete='POINT_END complete=20/20' in body and 'POINT_ERROR' not in body and 'LAB_TRUNCATED' not in body
    for match in re.finditer(r'^R (\d+):(\S+) A=([\d.]+) scalar=([\d.]+) vector=([\d.]+) view=([\d.]+) rS=([\d.]+) rV=([\d.]+) rH=([\d.]+)\[([\d.]+),([\d.]+)\] aa=([\d.]+) e=([\deE+.-]+) p=([\deE+.-]+)',body,re.M):
        t,role,*v=match.groups();a,s,b,h,rs,rv,rh,lo,hi,aa,err,probe=map(float,v)
        # Only the hoisted arm reports its full per-seed range. Vector/scalar
        # medians are recorded; promote neither from an unreported range.
        out['native'].append(dict(t=int(t),role=role,A_us=a,scalar_us=s,vector_us=b,view_us=h,scalar_ratio=rs,vector_ratio=rv,ratio=rh,ratio_min=lo,ratio_max=hi,aa_max=aa,error=err,probe=probe,
            valid=native_complete,followup_candidate=native_complete and rh<1-max(.02,2*aa) and hi<=1+aa))
    for match in re.finditer(r'^POINT (\d+):(\S+) A=([\d.]+) scalar=([\d.]+) group=([\d.]+) rS=([\d.]+) rG=([\d.]+)\[([\d.]+),([\d.]+)\] aa=([\d.]+) e=([\deE+.-]+) p=([\deE+.-]+)',body,re.M):
        t,role,*v=match.groups();a,s,b,rs,rg,lo,hi,aa,err,probe=map(float,v)
        out['point'].append(dict(t=int(t),role=role,A_us=a,scalar_us=s,B_us=b,scalar_ratio=rs,ratio=rg,ratio_min=lo,ratio_max=hi,aa_max=aa,error=err,probe=probe,
            valid=point_complete,followup_candidate=point_complete and rg<1-max(.02,2*aa) and hi<=1+aa))
    out['scans']=[x for x in body.splitlines() if x.startswith(('SCAN ','REGS '))]
    out['native_valid']=native_complete and len(out['native'])==10
    out['point_valid']=point_complete and len(out['point'])==20
    out['status']='measured' if out['native_valid'] and out['point_valid'] else 'partially measured' if out['native_valid'] or out['point_valid'] else 'unmeasured'
(H/'followup_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out.get('job'),len(out['native']),len(out['point']))
for row in out['native']+out['point']:
    if row['followup_candidate']:print('P4_CANDIDATE',row)
