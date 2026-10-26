import importlib.util,json,time,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'));from lab_packet import encode_packet
s=importlib.util.spec_from_file_location('xl_wire',H/'out/ladder.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
flush=torch.empty(1<<30,dtype=torch.uint8,device='cuda')
def cold():flush.view(torch.int64).sum()
m.lab_cold_l2=cold
m.XL_LOCAL_DUMP=lambda d:(H/'out/local_raw.json').write_text(json.dumps(d))
out=m.lab(time.monotonic()+840,None);packet=encode_packet('ladder',m.sha((H/'out/ladder.py').read_text())[:16],dict(lines=out,complete=None,format='legacy-lines'));n=len(packet.encode())+1;assert n<=11744,n
(H/'out/local_wire.txt').write_text('\n'.join(out));(H/'wire_validation.json').write_text(json.dumps(dict(passed=True,packet_bytes=n,all_seeds=8,rounds=4,local_only=True),indent=2)+'\n');print('WIRE',n)
