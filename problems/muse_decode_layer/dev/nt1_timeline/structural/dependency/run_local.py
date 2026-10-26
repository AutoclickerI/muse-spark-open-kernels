"""Full packed local execution;force material branch coverage,never use local timings as H200 evidence."""
import hashlib,json,linecache,sys,time
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import lab_packet
p=R/'prof/lab/moe_p3a_dependency.py';s=p.read_text();f='<dependency-packed>';linecache.cache[f]=(len(s),None,s.splitlines(True),f);buf=[]
def cold():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
n={'__name__':'dependency_local','lab_cold_l2':cold};exec(compile(s,f,'exec'),n);policy=n['dep_material'];n['dep_material']=lambda rows:True
r=n['lab'](time.monotonic()+600,None)['packet'];sha=hashlib.sha256(s.encode()).hexdigest();wire=lab_packet.encode_packet('moe_p3a_dependency',sha[:16],r);assert r['complete'] and not r['checks']['failures'],(r['seeds'],r['checks']);assert len(r['paired'])==240 and len(r['oracle'])==len(r['AA'])==48;assert len(wire)+1<=8744,len(wire);assert lab_packet.read_packets(wire)[0]['packet']==r
(R/'runs/screen/p3a_dependency/local_packet.json').write_text(json.dumps(r,indent=2)+'\n');(D/'lab_local_validation.json').write_text(json.dumps(dict(payload_sha256=sha,expanded_sha256=n['hashlib'].sha256(n['_source'].encode()).hexdigest(),complete=True,seeds=r['seeds'],paired=len(r['paired']),oracle=len(r['oracle']),AA=len(r['AA']),profiles=len(r['profiles']),checks=r['checks'],packet_bytes=len(wire)+1,elapsed_seconds=r['elapsed_seconds'],native_math_no_standins=True,material_policy='Local coverage forces branch;original policy result='+str(policy(r['oracle']))+';H200 uses original policy'),indent=2)+'\n');print('ALL_PASS',len(r['paired']),len(r['oracle']),len(r['AA']),len(wire)+1,r['elapsed_seconds'],flush=True)
