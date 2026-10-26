"""Native oracle replay, correctness-bearing controls, exact packet size; no H200 timing claims."""
import base64,hashlib,json,linecache,lzma,sys,time
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import lab_packet
buf=[]
def cold():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
p=R/'prof/lab/moe_p3a_operand.py';s=p.read_text();f='<operand-native-local>';linecache.cache[f]=(len(s),None,s.splitlines(True),f);ns={'__name__':'operand_local','lab_cold_l2':cold};exec(compile(s,f,'exec'),ns)
outer=ns['lab'](time.monotonic()+600,None)['packet'];raw=json.loads(lzma.decompress(base64.b85decode(outer['paired']['data'])))
wire=lab_packet.encode_packet(p.stem,hashlib.sha256(s.encode()).hexdigest()[:16],outer)
(R/'runs/screen/p3a_operand/local_packet.json').write_text(json.dumps(dict(outer=outer,decoded=raw),indent=2)+'\n')
assert outer['complete'] and not outer['checks']['failures'],outer['checks']
assert lab_packet.read_packets(wire)[0]['packet']==outer and len(wire.encode())+1<=ns['LAB_RESOURCES']['output_bytes'],len(wire.encode())+1
assert len(raw['round_delta32_remainder_ns'])==32 and len(raw['telemetry'])==32 and len(raw['resources'])==60
out=dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),wire_bytes=len(wire.encode())+1,all_native=True,complete=outer['complete'],checks=outer['checks'],round_samples=1920,resource_rows=60,rows=32,static_path='runs/screen/p3a_operand/static.json',timing_scope='sm120 transport/correctness only;not H200 oracle timings')
(D/'local_validation.json').write_text(json.dumps(out,indent=2)+'\n');print('OPERAND_NATIVE_PASS',out,flush=True)
