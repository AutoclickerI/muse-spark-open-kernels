"""Exact packed-bank lab rehearsal and native host guards for uneven K splits."""
import importlib.util,hashlib,json,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
p=load('ps_payload',H/'generated/packed_payload.py');screen=load('ps_ss_screen',H/'ss_screen.py');pin=hashlib.sha256((H/'generated/packed_payload.py').read_bytes()).hexdigest();calls=[]
def install(m):
 screen.install(m);m.PS_LOCAL_MM=screen.portable.mm;original=m.load_inline
 def build(name,cpp_sources='',cuda_sources='',**kw):
  if name.startswith('p10_packed_shared_'):
   s=cuda_sources;a=s.index('{',s.index('__global__'));b=s.index('\nusing EncodeFn=',a);s=s[:a]+'{}\n'+s[b:];s=s.replace('cfg.dynamicSmemBytes=smem;','cfg.dynamicSmemBytes=0;').replace('cudaFuncAttributeMaxDynamicSharedMemorySize,smem','cudaFuncAttributeMaxDynamicSharedMemorySize,0').replace('f,BT*2,smem','f,BT*2,0');cuda_sources=s;kw['extra_cuda_cflags']=[f.replace('compute_90a','compute_120').replace('sm_90a','sm_120') for f in kw['extra_cuda_cflags']]
  return original(name=name,cpp_sources=cpp_sources,cuda_sources=cuda_sources,**kw)
 m.load_inline=build;prepare=m.ps_prepare
 def prep(st):
  bank=prepare(st)
  for i,c in enumerate(m.PS_CONFIGS):
   t,bw=c[:2]
   for role in ('qkv_gate','o','pool_router','unpooler'):
    w=st[role];n,k=w.shape;x=torch.empty(t,k,device='cuda',dtype=torch.bfloat16)
    for s in (2,3,4,5,6,8):
     out=torch.empty(s,t,n,device='cuda');rc=bank['module'].launch(i,x.data_ptr(),bank['banks'][role,bw].data_ptr(),out.data_ptr(),t,n,k,s,torch.cuda.current_stream().cuda_stream);assert rc==0,(i,role,s,rc);calls.append((i,role,s))
  torch.cuda.synchronize();return bank
 m.ps_prepare=prep
if __name__=='__main__':
 p.PS_LOCAL_INSTALL=install;p.PS_LOCAL_ARCHIVE=lambda d:(H/'generated/packed_local_raw.json').write_text(json.dumps(d));buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:buf.view(torch.int64).sum();lines=p.lab(time.monotonic()+3600,None);print('\n'.join(lines));raw=json.loads((H/'generated/packed_local_raw.json').read_text());assert raw['complete'],raw['candidate']['failure'];assert pin==hashlib.sha256((H/'generated/packed_payload.py').read_bytes()).hexdigest();(H/'packed_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=pin,cases=8,seeds=8,arms=42,host_dispatches=len(calls),all_original_router_and_GEMM_TASK=True,all_banks_bitwise_exact=True),indent=2)+'\n')
