"""Read complete review H200 archive; nominate point winners and retain uncertainty."""
import argparse,json,math,re,statistics,sys
from pathlib import Path
from receipt import review_unpack
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parents[4]/'tools'))
from lab_packet import read_packets,result_lines
p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args();text=a.receipt.read_text();host=None
if a.receipt.suffix=='.json':
 host=json.loads(text);tail=host['result']['log_tail'];packets=read_packets(tail);text='\n'.join(result_lines(tail))
blobs=re.findall(r'^P3A_REVIEW_XZ85 (\S+)$',text,re.M)
if len(blobs)!=1:
 out=dict(job=host['id'] if host else 'LOCAL_ONLY',measured=False,invalid=['Seed-level timing/correctness archive absent; '+('wrapper truncated middle lines' if 'LAB_TRUNCATED' in text else 'missing archive')],summary=[],all_arms='unmeasured; no winner or loss can be inferred from official host pass alone')
 a.receipt.with_suffix('.summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));sys.exit(0)
r=review_unpack(blobs[0]);invalid=[]
if host is not None and not host['result'].get('passed'):invalid.append('official host failed')
if host is not None:
 for packet in packets:
  if packet['packet'].get('complete') is False or packet['packet'].get('transport_errors'):invalid.append('lab packet incomplete or transport failed')
if 'S15' in r['failures']:invalid.append('S15 control failed')
if 'S15' not in r['gates'] or r['gates']['S15'][0]>.015 or r['gates']['S15'][1]>.075:invalid.append('S15 TASK gate failed or absent')
summary=[]
for m in (1,8,16,32):
 aa=[(b-a)/1000 for seed,mm,a,b,ratio in r['AA_rows'] if mm==m]
 for control in sorted({row[3] for row in r['rows'] if len(row)==7}):
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
   seeds=[v[0] for v in vals]
   if sorted(seeds)!=list(range(r['requested_seeds'])):row_invalid.append('incomplete or duplicate seed pairs')
   if len(aa)!=r['requested_seeds']:row_invalid.append('incomplete A:A pairs')
   aar=[row[4]/1e6 for row in r['AA_rows'] if row[1]==m]
   summary.append(dict(M=m,arm=arm,control=r['arm_names'][control],ratio=mean,SE=se,seeds=len(rr),seed_indices=seeds,AA_mean_ratio=statistics.mean(aar) if aar else None,AA_SE=statistics.stdev(aar)/len(aar)**.5 if len(aar)>1 else None,AA_mean_abs_us=statistics.mean(map(abs,aa)) if aa else None,AA_max_abs_us=max(map(abs,aa)) if aa else None,point_nominee=host is not None and mean<1 and not row_invalid,invalid=row_invalid))
out=dict(job=host['id'] if host else 'LOCAL_ONLY',measured=not invalid,invalid=invalid,seed_level_record=r,summary=summary,promotion='P4 held-out six-load R lower confidence bound >1; per-shape uncertainty is descriptive')
a.receipt.with_suffix('.summary.json').write_text(json.dumps(out,indent=2)+'\n')
for row in summary:print(row)
