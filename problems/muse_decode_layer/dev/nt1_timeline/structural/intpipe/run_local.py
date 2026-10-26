"""Exact packed lab execution and checksummed packet bound,local timing is not H200 evidence."""
import hashlib,json,linecache,sys,time
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import lab_packet
p=R/'prof/lab/moe_p3a_intpipe.py';s=p.read_text();f='<intpipe-packed>';linecache.cache[f]=(len(s),None,s.splitlines(True),f);buf=[]
def cold():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
n={'__name__':'intpipe_local','lab_cold_l2':cold};exec(compile(s,f,'exec'),n);r=n['lab'](time.monotonic()+240,None)['packet'];sha=hashlib.sha256(s.encode()).hexdigest();wire=lab_packet.encode_packet('moe_p3a_intpipe',sha[:16],r);assert r['complete'] and len(wire)+1<=8744,(r['seeds'],r['checks'],len(wire));assert lab_packet.read_packets(wire)[0]['packet']==r
(D/'lab_local_validation.json').write_text(json.dumps(dict(payload_sha256=sha,complete=True,seeds=r['seeds'],paired=len(r['paired']),AA=len(r['AA']),checks=r['checks'],packet_bytes=len(wire)+1,elapsed_seconds=r['elapsed_seconds'],standins=False),indent=2)+'\n');print('ALL_PASS',len(r['paired']),len(r['AA']),len(wire)+1,r['elapsed_seconds'],flush=True)
