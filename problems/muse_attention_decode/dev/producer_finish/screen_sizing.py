"""Exact sizing payload, forced S19 Hopper glue; real finish math on GPU0."""
import hashlib,importlib.util,json,linecache,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')]
import torch,pack_sizing,lab_packet
s=pack_sizing.build();ns={};exec(compile(s,str(H/'out/sizing.py'),'exec'),ns)
sp=importlib.util.spec_from_file_location('pf_s19_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');scr=importlib.util.module_from_spec(sp);sp.loader.exec_module(scr)
original=ns['pf_host']
def host():
    m=original();scr.install(m,True);return m
ns['pf_host']=host;ns['SEEDS']=ns['SEEDS'][:1];ns['ROUNDS']=2
buf=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:buf.view(torch.int64).sum()
out=ns['lab'](time.monotonic()+3600,None);p=out['packet'];assert p['complete'] and len(p['paired'])==12
chars=len(lab_packet.encode_packet('sizing',hashlib.sha256(s.encode()).hexdigest()[:16],p).encode())+1
r=dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),checks=p['checks'],case_count=12,packet_bytes_one_seed=chars,native_finish_sha256=p['identities']['finish_sha256'],standins='Unchanged S19 forced Hopper stand-ins; finish CUDA executes as shipped. Public attention TASK uses actual ready Q/cache; full-QKV gate is materialized only in the untimed checker, QKV3 uses a zero-gate attention fixture. Whole-layer A/A2 keep the real concurrent gate GEMV.')
(H/'admission.json').write_text(json.dumps(r,indent=2)+'\n');(H/'out/local_packet.json').write_text(json.dumps(p));print('PASS',r)
