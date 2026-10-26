"""Screen native C++ dispatch and all SS arms with explicit local device stand-ins."""
import importlib.util,hashlib,json,time,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
s19=load('ss_s19_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');portable=load('ss_portable',H.parent/'dense_all/portable.py');dispatches=[]
def install(m,layer=True):
 if layer:s19.install(m,True)
 m.SS_LOCAL_MM=portable.mm;original=m.load_inline
 def compile(name,cpp_sources='',cuda_sources='',**kw):
  if name.startswith('p10_shared_shared_'):
   s=cuda_sources;a=s.index('{',s.index('__global__'));b=s.index('\nusing EncodeFn=',a);s=s[:a]+'{}\n'+s[b:];s=s.replace('cfg.dynamicSmemBytes=smem;','cfg.dynamicSmemBytes=0;').replace('cudaFuncAttributeMaxDynamicSharedMemorySize,smem','cudaFuncAttributeMaxDynamicSharedMemorySize,0').replace('f,BT*2,smem','f,BT*2,0');cuda_sources=s
   kw['extra_cuda_cflags']=[f.replace('compute_90a','compute_120').replace('sm_90a','sm_120') for f in kw['extra_cuda_cflags']]
  return original(name=name,cpp_sources=cpp_sources,cuda_sources=cuda_sources,**kw)
 m.load_inline=compile;old=m.ss_prepare
 def prepare(st):
  bank=old(st)
  for i,c in enumerate(m.SS_CONFIGS):
   t=c[0]
   for role in ('qkv_gate','o','pool_router','unpooler'):
    w=st[role];n,k=w.shape;x=torch.empty(t,k,device='cuda',dtype=torch.bfloat16)
    for s in (1,2,4,8):
     out=torch.empty(s,t,n,device='cuda');rc=bank['module'].launch(i,x.data_ptr(),w.data_ptr(),out.data_ptr(),t,n,k,s,torch.cuda.current_stream().cuda_stream);assert rc==0,(i,role,s,rc);dispatches.append((i,role,s))
  torch.cuda.synchronize();return bank
 m.ss_prepare=prepare
if __name__=='__main__':
 p=load('ss_payload',H/'generated/ss_payload.py');pin=hashlib.sha256((H/'generated/ss_payload.py').read_bytes()).hexdigest();p.SS_LOCAL_INSTALL=install;p.SS_LOCAL_ARCHIVE=lambda d:(H/'generated/ss_local_raw.json').write_text(json.dumps(d));buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:buf.view(torch.int64).sum();lines=p.lab(time.monotonic()+3600,None);print('\n'.join(lines));d=json.loads((H/'generated/ss_local_raw.json').read_text());assert d['complete'],d['failure'];assert pin==hashlib.sha256((H/'generated/ss_payload.py').read_bytes()).hexdigest();(H/'ss_lab_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=pin,shapes=8,seeds=8,arms_per_shape=26,host_dispatches=len(dispatches),limits='Hopper bodies explicitly replaced by portable BF16 GEMM; exact C++ arguments/guards/TMA encoding tested, dynamic smem disabled only locally'),indent=2)+'\n')
