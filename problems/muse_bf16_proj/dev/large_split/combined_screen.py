"""Rehearse the exact single-prepare packet, including all C++ dispatches."""
import importlib.util,hashlib,json,time,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
p=load('p10_joint_local',H/'generated/combined_payload.py');oracle=load('p10_oracle_local_screen',H/'screen.py');candidate=load('p10_candidate_local_screen',H/'ss_screen.py');pin=hashlib.sha256((H/'generated/combined_payload.py').read_bytes()).hexdigest()
def install(m):oracle.install(m);candidate.install(m,layer=False)
p.LS_LOCAL_INSTALL=install;p.COMBINED_LOCAL_ARCHIVE=lambda d:(H/'generated/combined_local_raw.json').write_text(json.dumps(d));buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:buf.view(torch.int64).sum();lines=p.lab(time.monotonic()+3600,None);print('\n'.join(lines));d=json.loads((H/'generated/combined_local_raw.json').read_text());assert d['complete'],d;assert pin==hashlib.sha256((H/'generated/combined_payload.py').read_bytes()).hexdigest();(H/'combined_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=pin,oracle_shapes=len(d['oracle']['rows']),candidate_shapes=len(d['candidate']['rows']),oracle_dispatches=len(oracle.dispatches),candidate_dispatches=len(candidate.dispatches),oracle_and_candidate_seeds=[8,8],encoded_bytes=sum(len(s)+1 for s in lines)),indent=2)+'\n')
