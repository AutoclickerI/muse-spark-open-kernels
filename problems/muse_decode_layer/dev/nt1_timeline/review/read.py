"""Read complete review H200 archive; nominate point winners and retain uncertainty."""
import argparse,json,math,re,statistics
from pathlib import Path
from receipt import review_unpack
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args();text=a.receipt.read_text();host=None
if a.receipt.suffix=='.json':
 host=json.loads(text);text=host['result']['log_tail']
blobs=re.findall(r'^P3A_REVIEW_XZ85 (\S+)$',text,re.M);assert len(blobs)==1
r=review_unpack(blobs[0]);invalid=[]
if host is not None and not host['result'].get('passed'):invalid.append('official host failed')
if 'S15' in r['failures']:invalid.append('S15 control failed')
if 'S15' not in r['gates'] or r['gates']['S15'][0]>.015 or r['gates']['S15'][1]>.075:invalid.append('S15 TASK gate failed or absent')
summary=[]
for m in (1,8,16,32):
 aa=[(b-a)/1000 for seed,mm,a,b,ratio in r['AA_rows'] if mm==m]
 for control in (0,r['arm_names'].index('PAIR')):
  for idx,arm in enumerate(r['arm_names']):
   vals=[row for row in r['rows'] if len(row)==7 and row[1]==m and row[2]==idx and row[3]==control]
   if not vals:continue
   rr=[v[6]/1e6 for v in vals];mean=statistics.mean(rr);se=statistics.stdev(rr)/len(rr)**.5 if len(rr)>1 else None
   row_invalid=list(invalid)
   for target in (arm,r['arm_names'][control]):
    error=r['failures'].get(target,'')
    if error and not error.startswith('node '):row_invalid.append(target+': '+error)
    if target not in r['gates'] or r['gates'][target][0]>.015 or r['gates'][target][1]>.075:row_invalid.append(target+' TASK gate failed or absent')
   if not aa:row_invalid.append('A:A control absent')
   summary.append(dict(M=m,arm=arm,control=r['arm_names'][control],ratio=mean,SE=se,seeds=len(rr),AA_mean_abs_us=statistics.mean(map(abs,aa)) if aa else None,AA_max_abs_us=max(map(abs,aa)) if aa else None,point_nominee=host is not None and mean<1 and not row_invalid,invalid=row_invalid))
out=dict(job=host['id'] if host else 'LOCAL_ONLY',invalid=invalid,seed_level_record=r,summary=summary,promotion='P4 held-out six-load R lower confidence bound >1; per-shape uncertainty is descriptive')
a.receipt.with_suffix('.summary.json').write_text(json.dumps(out,indent=2)+'\n')
for row in summary:print(row)
