"""Cross-compile changed donor TMA schedules and inspect actual issue/wait order."""
import importlib.util,json,sys,re
from pathlib import Path
import torch
from triton.tools.tensor_descriptor import TensorDescriptor
H=Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
cross=load('p10_reuse_native_compiler',H/'compile.py');m=load('p10_pool_native_source',H/'out/whole_source.py');rows=[]
for t in (16,32):
 for name,(bk,st,early) in m.PP_POOL_TMA.items():
  x=torch.empty((t,8192),device='cuda',dtype=torch.bfloat16);w=torch.empty((4608*8192//bk,bk),device='cuda',dtype=torch.bfloat16);out=torch.empty((t,4608),device='cuda');xd=TensorDescriptor.from_tensor(x,[t,bk]);wd=TensorDescriptor.from_tensor(w,[64,bk]);fn=m._pp_pool_tma_early if early else m._cbf_skinny_gemm_tma_tiled_ppdl
  c=cross.Cross(fn).go((xd,wd,wd,out,out,out,t,4608,8192),dict(BLOCK_M=t,BLOCK_N=64,BLOCK_K=bk,SPLIT_K=1,HILO=False,OUT_F32=True,num_warps=4,num_stages=st)).kernel
  ptx=c.asm['ptx'];cp=ptx.find('cp.async.bulk.tensor');wait=ptx.find('griddepcontrol.wait');assert cp>=0 and wait>=0
  before=set(re.findall(r'cp\.async\.bulk\.tensor[^\n]*?\], \[(%rd\d+),',ptx[:wait]));all_desc=set(re.findall(r'cp\.async\.bulk\.tensor[^\n]*?\], \[(%rd\d+),',ptx));assert len(before)==int(early) and len(all_desc)==2,(t,name,before,all_desc)
  rows.append(dict(t=t,arm=name,shared=c.metadata.shared,first_TMA_before_wait=cp<wait,requested_early=early,activation_TMA_after_wait=True,prewait_descriptors=sorted(before),all_descriptors=sorted(all_desc),cubin_sha256=cross.hashlib.sha256(c.asm['cubin']).hexdigest()))
out=dict(passed=all(r['shared']<=227328 for r in rows),units=rows,limits='sm90 cross-compile, issue-order inspection; actual native correctness/timing remain remote.')
(H/'pool_native_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
