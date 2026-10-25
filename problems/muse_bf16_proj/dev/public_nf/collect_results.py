"""Extract this lane's lab evidence from durable judge receipts; never calls the judge."""
import json,re
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
JOBS=['bdc1f5a233d5405e','08fff0542ea1462b','a1ed67b7a9f04854','f4d1a1950d18449c','c9e69082a6854d92','84ddc084a57644ec','e3c637d10a584a0f']
records=[];pairs=[]
pat=re.compile(r'R t=(\d+) (\w+) A=([\d.]+) B=([\d.]+) r=([\d.]+)\[([\d.]+),([\d.]+)\] aa=([\d.]+) e=([^ ]+) p=([^ ]+)(?: NS4/bal2=([\d.]+))?')
for job in JOBS:
 files=list((R/'runs').glob('*/*_check_'+job+'.json'))
 if not files:records.append(dict(job=job,status='pending'));continue
 p=files[0];d=json.loads(p.read_text());result=d.get('result') or {}
 logpath=Path(d.get('output',{}).get('log',''));text=logpath.read_text() if logpath.is_file() else result.get('log_tail','')
 blocks=re.findall(r'LAB_BEGIN .*?LAB_END[^\n]*',text,re.S)
 lab='\n'.join(blocks)
 trace='\n'.join(l for l in text.splitlines() if l.startswith('NFTRACE') or 'stack smashing' in l)[-1400:]
 records.append(dict(job=job,receipt=str(p.relative_to(R)),status=d['status'],host_passed=result.get('passed'),host_speedup=result.get('speedup'),error=result.get('error'),lab=lab,trace_tail=trace))
 for line in lab.splitlines():
  m=pat.fullmatch(line)
  if not m:continue
  t,n,a,b,r,lo,hi,aa,e,probe,ns4=m.groups()
  pairs.append(dict(job=job,t=int(t),role=n,ours_us=float(a),nf_us=float(b),ratio=float(r),seed_min=float(lo),seed_max=float(hi),aa_max=float(aa),relative_l2=float(e),onehot_l2=float(probe),ns4_over_bal2=float(ns4) if ns4 else None))
(H/'h200_results.json').write_text(json.dumps(records,indent=2)+'\n')
if pairs:(H/'paired_stages.json').write_text(json.dumps(pairs,indent=2)+'\n')
for row in records:print(row['job'],row['status'],len(row.get('lab','')),row.get('host_speedup'))
print('paired rows',len(pairs))
