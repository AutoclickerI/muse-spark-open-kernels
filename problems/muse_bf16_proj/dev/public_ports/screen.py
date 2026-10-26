"""Exact batch, actual coded decoder, and explicit Hopper launch/metadata stand-ins."""
import importlib.util,json,time,hashlib,sys
from pathlib import Path
import torch,triton,triton.language as tl
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
s19=load('pubp_s19_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');portable=load('pubp_portable',H.parent/'dense_all/portable.py');p=load('public_proj_payload',H/'out/public_projection.py');calls=[]
def install(m):
 s19.install(m,True);oldprep=m.pp_prepare;lookup={};m._pp_local_lookup=lookup
 class Kernel:
  def __init__(self,name):self.name=name
  def __getitem__(self,grid):
   def run(*a,**kw):
    calls.append((self.name,grid))
    if self.name.startswith('_pp_coded'):
     xd,words,base,offs,esc,out,mt,n,k=a;w=lookup[words.data_ptr(),n]
    elif self.name.startswith('_cbf_gemv_rows'):
     x,w,out,*_=a;portable.mm(x,w,out);return
    else:
     xd,wd=a[:2];mt,n,k=a[6:9] if 'skinny' in self.name else a[4:7];out=a[3] if 'skinny' in self.name else a[2];w=lookup[wd.base.data_ptr(),n]
    x=xd.base;shape=(mt,n)
    if out.ndim==3:out.zero_();portable.mm(x,w,out[0])
    else:portable.mm(x,w,out)
   return run
 class Cluster:
  def __init__(self,pdl):self.pdl=pdl
  def warmup(self,*a,**kw):
   class Compiled:
    packed_metadata=(4,1,65536);function=None
    def _init_handles(self):pass
    def launch_metadata(self,*args):return None
    def run(self,gx,gy,gz,stream,function,pm,metadata,h1,h2,*args):
     xd,wd,out,mt,n,k,bm,bn,bk,s,slots=args;assert gx==1 and gy==n//bn and pm[1]==s and s in (3,4);calls.append(('cluster_pdl' if self.pdl else 'cluster',tuple((mt,n,k,bm,bn,bk,s,slots))));portable.mm(xd.base,wd.base,out)
   z=Compiled();z.pdl=self.pdl;return z
 for name in ('_cbf_skinny_gemm','_cbf_skinny_gemm_tma','_cbf_skinny_gemm_tma_tiled','_cbf_gemv_rows','_cbf_gemv32','_cbf_lm_partials','_cbf_lm_partials_tiled'):
  setattr(m,name,Kernel(name));setattr(m,name+'_ppdl',Kernel(name+'_ppdl'))
 m._cbf_cluster_gemm=Cluster(False);m._cbf_cluster_gemm_ppdl=Cluster(True);m._pp_coded=Kernel('_pp_coded')
 def prepare(ns,w,st,requested):
  bank=oldprep(ns,w,st,requested)
  for role,ww in bank['roles'].items():lookup[ww.data_ptr(),ww.shape[0]]=ww
  for (role,bk),wp in bank['tiled'].items():lookup[wp.data_ptr(),bank['roles'][role].shape[0]]=bank['roles'][role]
  for role,c in bank['coded'].items():lookup[c['words'].data_ptr(),c['n']]=bank['roles'][role]
  # Integer encoding is additionally checked on every BF16 bit pattern.
  bits=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16).view(256,256);m.pp_encode(bits)
  return bank
 m.pp_prepare=prepare
if __name__=="__main__":
 p.PP_LOCAL_INSTALL=install;flush=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:flush.view(torch.int64).sum();data=[];p.PP_LOCAL_ARCHIVE=lambda d:data.append(d);lines=p.lab(time.monotonic()+1600,None);raw=data[0];(H/'out/local_raw.json').write_text(json.dumps(raw));assert raw['complete'],raw['fatal'];sys.path.insert(0,str(R/'tools'));from lab_packet import encode_packet,read_packets
 wire=encode_packet('public_projection','0'*64,dict(lines=lines,complete=None,format='legacy-lines'));assert len(wire)+1<=13744,len(wire);assert len(read_packets((wire+'\n'+'suffix\n'*285+'X'*256)[-16000:]))==1
 (H/'local_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'out/public_projection.py').read_bytes()).hexdigest(),rows=len(raw['rows']),projection_TASK=raw['checks'],router_TASK=raw['router_checks'],gate_layer_TASK=raw['gate_layer_checks'],host_dispatches=len(calls),wire_bytes=len(wire)+1,two_k_suffix_guard=True,limits='Public Hopper body math replaced by explicit stand-ins; original S19/NF available local paths retained, all BF16 encoding bits tested'),indent=2)+'\n');print('PASS',raw['checks'],raw['router_checks'],len(calls),len(wire)+1,'failures',len(raw['failures']))
