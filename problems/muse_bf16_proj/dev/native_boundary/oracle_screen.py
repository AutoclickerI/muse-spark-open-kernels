"""Full oracle wiring/TASK rehearsal with explicit local Hopper arithmetic stand-ins."""
import importlib.util,json,time,hashlib,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
screen=load('nb_s19_oracle_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');path=Path(sys.argv[1]);p=load('nb_oracle_payload',path);p.NB_LOCAL_INSTALL=lambda m:screen.install(m,True)
f=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:f.view(torch.int64).sum();raws=[];p.NB_LOCAL_ARCHIVE=lambda d:raws.append(d);lines=p.lab(time.monotonic()+1000,None);d=raws[0];(H/'generated'/('local_'+p.NB_SELECTED+'_oracle.json')).write_text(json.dumps(d));assert d['complete'],d['failure'];sys.path.insert(0,str(R/'tools'));from lab_packet import encode_packet
wire=len(encode_packet('oracle_payload','0'*64,dict(lines=lines,complete=None,format='legacy-lines')))+1;assert wire<=11744,wire
(H/('oracle_'+p.NB_SELECTED+'_validation.json')).write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=12,TASK=d['checks'],per_round_times_retained=True,actual_wrapper_bytes=wire,limits='Only local stand-in arithmetic; not native H200 timing'),indent=2)+'\n');print('PASS',d['checks'],'TASK',wire,'wire')
