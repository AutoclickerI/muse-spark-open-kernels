"""Exact packed structural lab with test-only local depth stand-in."""
import hashlib,json,linecache,time
from pathlib import Path
import torch
import local_depth
D=Path(__file__).resolve().parent;R=D.parents[4];p=R/'prof/lab/moe_p3a_structural.py';s=p.read_text();f='<structural local packed>';linecache.cache[f]=(len(s),None,s.splitlines(True),f);buf=[]
def cold():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
n={'__name__':'structural_lab','lab_cold_l2':cold};exec(compile(s,f,'exec'),n);calls=local_depth.install(n);lines=n['lab'](time.monotonic()+350,None)
for line in lines:print(line,flush=True)
packed=next(l.split(' ',1)[1] for l in lines if l.startswith('P3A_REVIEW_XZ85 '));r=n['review_unpack'](packed);assert r['seeds']==16 and not r['failures'],r
assert len(r['AA_rows'])==48 and len(r['rows'])==528
for arm in r['arm_names'][1:]:assert sum(v[2]==r['arm_names'].index(arm) and v[3]==0 for v in r['rows'])==48
assert sum(len(l)+1 for l in lines)<9000
(D/'lab_local_validation.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),seeds=16,M=[8,16,32],AA_rows=48,paired_rows=528,gates=r['gates'],failures=r['failures'],elapsed_seconds=r['elapsed_seconds'],output_chars=sum(len(l)+1 for l in lines),complete=True,depth_standin_calls=len(calls),requested_depths=sorted({v['requested_depth'] for v in calls}),executed_local_depth=2,native_split_math=True,archive=packed),indent=2)+'\n');print('LOCAL_COMPLETE',sum(len(l)+1 for l in lines),r['elapsed_seconds'],flush=True)
