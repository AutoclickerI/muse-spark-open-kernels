"""Rehearse every new whole-layer dispatch; actual NF math, explicit TMA stand-in."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
local=load('p10_public_local',H/'screen.py');local.s19=load('p10_s21_standins',R/'problems/muse_decode_layer/dev/s21/screen.py');p=load('p10_public_whole',H/'out/public_whole.py')
def install(m):
 local.install(m);m._pp_pool_tma_early=m._cbf_skinny_gemm_tma_tiled_ppdl;original=m.pp_pool_project
 def project(x,bank,pool,name):
  for (role,bk),wp in bank['tiled'].items():m._pp_local_lookup[wp.data_ptr(),bank['roles'][role].shape[0]]=bank['roles'][role]
  return original(x,bank,pool,name)
 m.pp_pool_project=project
p.PP_LOCAL_INSTALL=install;flush=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:flush.view(torch.int64).sum();archive=[];p.PP_LOCAL_ARCHIVE=archive.append;lines=p.lab(time.monotonic()+1500,None);raw=archive[0];(H/'out/whole_local_raw.json').write_text(json.dumps(raw));assert raw['complete'],raw['fatal'];assert all(v=='ready' for v in raw['pool_status'].values()),raw['failures']
sys.path.insert(0,str(R/'tools'));from lab_packet import encode_packet,read_packets
wire=encode_packet('public_whole','0'*16,dict(lines=lines,complete=None,format='legacy-lines'));assert len(wire)+1<=13744,len(wire);assert len(read_packets((wire+'\n'+'suffix\n'*285+'X'*256)[-16000:]))==1
(H/'whole_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'out/public_whole.py').read_bytes()).hexdigest(),cells=len(raw['rows']),layer_TASK=raw['checks'],projection_router_TASK=raw['projection_checks'],host_dispatches=len(local.calls),wire_bytes=len(wire)+1,two_k_suffix_guard=True,limits='Actual NF pool arithmetic including changed PDL scheduling; explicit TMA/Hopper stand-ins preserve ABI and launch dispatch. Native H200 receipt needed.'),indent=2)+'\n');print('PASS',raw['checks'],raw['projection_checks'],len(wire)+1)
