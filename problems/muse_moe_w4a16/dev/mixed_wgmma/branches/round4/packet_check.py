"""Inverse packet conversion proves only byte permutation; no expanded values."""
import hashlib
import json
from pathlib import Path
import runpy
import torch
H=Path(__file__).resolve().parent
s=runpy.run_path(str(H/'ws_payload.py'),init_globals={'lab_cold_l2':lambda:None})
results=[]
for e,ng in ((1,8),(2,16),(3,24)):
 q=torch.randint(-(2**31),2**31,(e,ng,128,64),device='cuda',dtype=torch.int32);sf=torch.randint(0,256,(e,ng,32,8,16),device='cuda',dtype=torch.uint8)
 p=s['ws_packet'](q,sf);mt=ng//4
 p64=p.reshape(e,mt//2,32,2,4608).permute(0,1,3,2,4).contiguous().reshape(e,mt,32,4608)
 qi=p64[...,:4096].contiguous().view(torch.int32).reshape(e,mt,32,2,4,2,16,4).permute(0,1,4,2,3,5,6,7).contiguous().reshape(q.shape)
 si=p64[...,4096:].contiguous().reshape(e,mt,32,4,8,16).permute(0,1,3,2,4,5).contiguous().reshape(sf.shape)
 assert torch.equal(q,qi) and torch.equal(sf,si)
 assert p.numel()==q.numel()*4+sf.numel() and p.data_ptr()%16==0
 results.append(dict(experts=e,groups=ng,bytes=p.numel(),bit_equal=True))
out=dict(payload_sha256=hashlib.sha256((H/'ws_payload.py').read_bytes()).hexdigest(),cases=results,meaning='All packed code and scale bytes preserved, no precision expansion; metadata aligns each9216B TMA packet.')
(H/'packet.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
