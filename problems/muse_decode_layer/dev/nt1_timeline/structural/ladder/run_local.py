import hashlib,json,linecache,sys,time
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import lab_packet
parts=[];buf=[]
def cold():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
for part in (0,1):
 p=R/f'prof/lab/moe_p3a_ladder{part}.py';s=p.read_text();f=f'<ladder-local-{part}>';linecache.cache[f]=(len(s),None,s.splitlines(True),f);n={'__name__':'ladder_local','lab_cold_l2':cold};exec(compile(s,f,'exec'),n);outer=n['lab'](time.monotonic()+500,None)['packet']
 assert all(k in outer for k in ('identities','seeds','paired','checks','complete'))
 import lzma,base64
 r=json.loads(lzma.decompress(base64.b85decode(outer['paired']['data'])))
 if part==0:r.update(complete=outer['complete'],checks=outer['checks'],seeds=outer['seeds'])
 wire=lab_packet.encode_packet('moe_p3a_ladder'+str(part),hashlib.sha256(s.encode()).hexdigest()[:16],outer)
 (R/f'runs/screen/p3a_ladder/local_packet{part}.json').write_text(json.dumps(r,indent=2)+'\n')
 print('PART',part,'complete',outer['complete'],'bytes',len(wire)+1,flush=True)
 assert outer['complete'] and not outer['checks']['failures'] and lab_packet.read_packets(wire)[0]['packet']==outer
 assert len(wire)+1<=n['LAB_RESOURCES']['output_bytes'],(part,len(wire)+1)
 parts.append(dict(part=part,wire_bytes=len(wire)+1,payload_sha256=hashlib.sha256(s.encode()).hexdigest(),checks=outer['checks'],rows=len(r['round_delta32_remainder_ns']) if part==0 else 4,native_math=True,schema_keys_verified=True))
(D/'local_validation.json').write_text(json.dumps(parts,indent=2)+'\n')
