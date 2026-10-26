"""All native-dispatch arms and strict TASK contracts, with explicit local TMA stand-in."""
import importlib.util,json,time,hashlib,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
screen=load('nb_s19_tune_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');portable=load('nb_tune_portable',H.parent/'dense_all/portable.py');p=load('nb_tune_payload',H/'generated/tune_payload.py');calls=[]
def install(m):
 screen.install(m,True)
 def project(x,w,variant,chain=False):
  _,cfg,s=variant;bt,bw,bk,ns,nwarps,tma=cfg;assert w.shape[0]%bw==0 and w.shape[1]%bk==0;calls.append((tuple(x.shape),tuple(w.shape),cfg,s,chain));out=torch.zeros((s,x.shape[0],w.shape[0]),device=x.device);portable.mm(x,w,out[0]);return out,s
 m._vdsk_project=project
p.NB_LOCAL_INSTALL=install;f=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:f.view(torch.int64).sum();raws=[];p.NB_LOCAL_ARCHIVE=lambda d:raws.append(d);lines=p.lab(time.monotonic()+1000,None);raw=raws[0];(H/'generated/local_tune.json').write_text(json.dumps(raw));assert raw['complete'],raw['failure'];sys.path.insert(0,str(R/'tools'));from lab_packet import encode_packet,read_packets
wire=len(encode_packet('tune_payload','0'*64,dict(lines=lines,complete=None,format='legacy-lines')))+1;assert wire<=13744,wire
(H/'tune_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'generated/tune_payload.py').read_bytes()).hexdigest(),layer_TASK=raw['checks'],projection_router_TASK=raw['projection_checks'],host_dispatches=len(calls),per_round_times_retained=True,actual_wrapper_bytes=wire,limits='Portable GEMM stands in for Hopper TMA; exact native body is unchanged from S19'),indent=2)+'\n');print('PASS',raw['checks'],raw['projection_checks'],wire)
