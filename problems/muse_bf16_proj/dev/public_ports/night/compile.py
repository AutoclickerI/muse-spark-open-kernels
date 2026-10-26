"""Compile all changed kernels for sm90a and prove PDL/TMA admission and PTX order."""
import hashlib,importlib.util,inspect,json,re,subprocess,sys
from pathlib import Path
import torch,triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import mangle_type
H=Path(__file__).resolve().parent;O=H/'out'
spec=importlib.util.spec_from_file_location('ng_native_source',O/'source.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);original=m._nf_vendor;insts0=[]
class Captured(Exception):pass
def vendor():
 ns=original()
 def build(insts,label):
  insts0.extend(insts);(O/'native.cu').write_text(ns['v18s_cuda_source'](insts));raise Captured
 ns['v18s_build']=build;return ns
m._nf_vendor=vendor
roles={r:torch.empty((n,k),device='cuda',dtype=torch.bfloat16) for r,n,k in [('qkv_gate',2304,8192),('qkv3',1280,8192),('o',8192,1024),('pool_router',4608,8192),('unpooler',8192,4096)]}
try:m.ng_prepare(vars(m),roles)
except Captured:pass
with (O/'nvcc.log').open('w') as log:subprocess.run(['nvcc','-arch=sm_90a','-cubin','-lineinfo','-Xptxas=-v',str(O/'native.cu'),'-o',str(O/'native.cubin')],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run(['nvcc','-arch=sm_90a','-ptx','-lineinfo',str(O/'native.cu'),'-o',str(O/'native.ptx')],check=True)
ptx=(O/'native.ptx').read_text();log=(O/'nvcc.log').read_text();units=[]
for body in ptx.split('.visible .entry ')[1:]:
 name=body.split('(')[0].strip()
 if not name.startswith('_Z9v18s_gemv'):continue
 params=list(map(int,re.findall(r'Li(\d+)E',name)));body=body.split('// .globl',1)[0];wait=body.find('griddepcontrol.wait');ldg=body.find('ld.global.nc.L1::no_allocate.v4.s32');assert wait>ldg>=0;assert body.find('ld.global.nc.v4.u32')>wait
 prop=log.split('Function properties for '+name+'\n',1)[1].split('ptxas info    : Compile time',1)[0];units.append(dict(inst=params,regs=int(re.search(r'Used (\d+) registers',prop).group(1)),zero_spill='0 bytes spill stores, 0 bytes spill loads' in prop,first_global_load_before_wait=True,source='NVCCsm90a'))
assert len(units)==9
cluster=[]
class Cross:
 def warmup(self,*args,**kw):
  fn=m._ng_cluster_original;bound=inspect.signature(fn.fn).bind_partial(*args,**{k:v for k,v in kw.items() if k in fn.arg_names});bound.apply_defaults();sig={};const={};attrs={}
  for i,n in enumerate(fn.arg_names):
   x=bound.arguments[n]
   if i in fn.constexprs:const[n]=x
   else:
    sig[n]=mangle_type(x)
    if isinstance(x,torch.Tensor) or isinstance(x,int) and x%16==0:attrs[(i,)]=[['tt.divisibility',16]]
  options={k:v for k,v in kw.items() if k not in fn.arg_names and k!='grid'};k=triton.compile(ASTSource(fn,sig,const,attrs),target=GPUTarget('cuda',90,32),options=options);ptx=k.asm['ptx'];wait=ptx.find('griddepcontrol.wait');tma_before=re.findall(r'cp\.async\.bulk\.tensor[^\n]*?\], \[(%rd\d+),',ptx[:wait]);tma_all=re.findall(r'cp\.async\.bulk\.tensor[^\n]*?\], \[(%rd\d+),',ptx);assert len(set(tma_before))==1 and len(set(tma_all))==2,(tma_before,tma_all);assert k.metadata.launch_pdl is True;assert 49152<k.metadata.shared<=227328
  (O/'cluster.ptx').write_text(ptx);(O/'cluster.cubin').write_bytes(k.asm['cubin']);cluster.append(dict(shared=k.metadata.shared,launch_pdl=k.metadata.launch_pdl,constants=const,weight_TMA_before_wait=True,activation_TMA_after_wait=True,prewait_TMA_descriptors=list(set(tma_before)),all_TMA_descriptors=list(set(tma_all)),cubin_sha256=hashlib.sha256(k.asm['cubin']).hexdigest()))
  class Run:
   launch_pdl=True
   def __call__(self,*a):pass
  class Compiled:
   packed_metadata=(k.metadata.num_warps,1,k.metadata.shared);run=Run();function=None
   def _init_handles(self):pass
   def launch_metadata(self,*a):return None
  return Compiled()
m._ng_cluster_original=m._ng_cluster_pool;m._ng_cluster_pool=Cross();m.ng_tma_pool(torch.empty((32,8192),device='cuda',dtype=torch.bfloat16),dict(roles=roles,tma_w=torch.empty((4608*8192//128,128),device='cuda',dtype=torch.bfloat16),cluster={}))
out=dict(passed=all(u['zero_spill'] for u in units),NF=units,cluster=cluster,NF_source_sha256=hashlib.sha256((O/'native.cu').read_bytes()).hexdigest(),NF_cubin_sha256=hashlib.sha256((O/'native.cubin').read_bytes()).hexdigest(),limits='Actual sm90a compile/admission and PTX issue order; native H200 arithmetic, performance and prepare remain unmeasured.')
(H/'native_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
