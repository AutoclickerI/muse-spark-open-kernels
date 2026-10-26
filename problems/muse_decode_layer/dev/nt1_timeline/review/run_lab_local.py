"""Exercise exact complete packed lab with native small-M math on GPU0."""
import argparse,base64,hashlib,json,linecache,lzma,time
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[4]
parser=argparse.ArgumentParser();parser.add_argument('--payload',type=Path,default=R/'prof/lab/moe_p3a_review.py');parser.add_argument('--output',type=Path,default=D/'lab_local_validation.json');args=parser.parse_args()
p=args.payload;text=p.read_text();name='<P3a S15 review lab>';linecache.cache[name]=(len(text),None,text.splitlines(True),name)
buf=[]
def flush():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
n={'__name__':'h200lab','lab_cold_l2':flush};exec(compile(text,name,'exec'),n)
lines=n['lab'](time.monotonic()+280,None)
for line in lines:print(line,flush=True)
archive=next(l.split(' ',1)[1] for l in lines if l.startswith('P3A_REVIEW_XZ85 '));data=n['review_unpack'](archive)
assert data['seeds']==16 and not data['failures'],data
assert len(data['AA_rows'])==64
for name in n['REVIEW_ARMS']:
 idx=data['arm_names'].index(name);assert sum(len(r)==7 and r[2]==idx and r[3]==0 for r in data['rows'])==64
assert sum(len(l)+1 for l in lines)<9000,'complete archive exceeds receipt cap'
args.output.write_text(json.dumps(dict(payload_sha256=hashlib.sha256(text.encode()).hexdigest(),device='GPU0 sm120 native review math; local timing only',seeds=16,AA_rows=64,complete=True,output_chars=sum(len(l)+1 for l in lines),gates=data['gates'],failures=data['failures'],elapsed_seconds=data['elapsed_seconds'],archive=archive),indent=2)+'\n');print('LOCAL_COMPLETE',len(lines),sum(len(l)+1 for l in lines),data['elapsed_seconds'],flush=True)
