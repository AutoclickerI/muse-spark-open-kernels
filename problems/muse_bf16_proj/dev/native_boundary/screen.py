"""Exact sizing payload rehearsal; S19 Hopper arithmetic uses explicit local stand-ins."""
import importlib.util,json,time,hashlib,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
screen=load('nb_s19_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');p=load('nb_sizing_payload',H/'generated/sizing_payload.py');p.NB_LOCAL_INSTALL=lambda m:screen.install(m,True)
f=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:f.view(torch.int64).sum();p.NB_LOCAL_ARCHIVE=lambda d:(H/'generated/local_sizing.json').write_text(json.dumps(d));lines=p.lab(time.monotonic()+1000,None);d=json.loads((H/'generated/local_sizing.json').read_text());assert d['complete'],d['failure'];print('PASS',d['checks'],'TASK',len(lines[0]),'archive chars')
(H/'sizing_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'generated/sizing_payload.py').read_bytes()).hexdigest(),rows=len(d['rows']),TASK=d['checks'],per_round_times_retained=True,native_kernels='stand-ins on sm120; CUPTI mapping executes'),indent=2)+'\n')
