"""Exact packed oracle rehearsal, explicit Hopper stand-ins and host launch ABI."""
import importlib.util,hashlib,json,time,types,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
p=load('ls_payload',H/'generated/payload.py');screen=load('ls_s19_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');portable=load('ls_portable',H.parent/'dense_all/portable.py');payload_sha=hashlib.sha256((H/'generated/payload.py').read_bytes()).hexdigest();dispatches=[]
def install(m):
 screen.install(m,True);m.LS_LOCAL_MM=portable.mm;original=m.load_inline
 def compile(name,cpp_sources='',cuda_sources='',**kw):
  if name.startswith('p10_ls_'):
   s=cuda_sources;a=s.index('{',s.index('__global__'));b=s.index('\nusing EncodeFn=',a);s=s[:a]+'{}\n'+s[b:];kw['extra_cuda_cflags']=[f.replace('compute_90a','compute_120').replace('sm_90a','sm_120') for f in kw['extra_cuda_cflags']];cuda_sources=s
  return original(name=name,cpp_sources=cpp_sources,cuda_sources=cuda_sources,**kw)
 m.load_inline=compile;prepare=m.ls_prepare
 def prep(ns,w):
  bank=prepare(ns,w)
  for t in (128,256):
   for role in ('qkv_gate','o','pool_router','unpooler'):
    bn,s=m.LS_CONFIG[t,role];idx=0 if bn==128 else 1;k=bank['roles'][role].shape[1];n=bank['roles'][role].shape[0];x=torch.empty(t,k,device='cuda',dtype=torch.bfloat16);out=torch.empty(s,t,n,device='cuda');sink=torch.empty(1,device='cuda',dtype=torch.int32)
    for mode,mod in bank['modules'].items():
     rc=mod.launch(idx,x.data_ptr(),bank['packed'][role].data_ptr(),sink.data_ptr(),out.data_ptr(),t,n,k,s,0,torch.cuda.current_stream().cuda_stream);assert rc==0,(t,role,mode,rc);dispatches.append((t,role,mode))
  torch.cuda.synchronize();return bank
 m.ls_prepare=prep
if __name__=='__main__':
 p.LS_LOCAL_INSTALL=install;p.LS_LOCAL_ARCHIVE=lambda d:(H/'generated/local_raw.json').write_text(json.dumps(d));buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:buf.view(torch.int64).sum();lines=p.lab(time.monotonic()+3600,None);print('\n'.join(lines));d=json.loads((H/'generated/local_raw.json').read_text());assert d['complete'],d['failure'];assert payload_sha==hashlib.sha256((H/'generated/payload.py').read_bytes()).hexdigest();(H/'lab_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=payload_sha,shapes=8,seeds=8,host_dispatches=len(dispatches),rows=len(d['rows']),limits='Native math replaced by portable BF16 GEMM; synthetic oracles are not promoted or checked as true GEMMs'),indent=2)+'\n')
