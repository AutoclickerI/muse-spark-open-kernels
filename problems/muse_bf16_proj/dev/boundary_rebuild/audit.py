"""Offline sm90a compile: native resources and published-unpool ordering/pipeline."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
import torch,triton
from triton.backends.compiler import GPUTarget
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'d1_chain'))
from audit_native import compile_one,load
m=load(HERE/'publish_trial.py');folder=HERE/'native'/hashlib.sha256(m.BR_CUDA.encode()).hexdigest()[:12];folder.mkdir(parents=True,exist_ok=True)
r=compile_one(('boundary_rebuild',m.BR_CUDA),folder);assert r['ok']
usage=(folder/'boundary_rebuild.resources').read_text();native=[]
for b in usage.split('Function ')[1:]:
 if not ('boundaryILi' in b or 'routeILi' in b):continue
 native.append(dict(symbol=b.splitlines()[0].strip(),registers=int(re.search(r'REG:(\d+)',b)[1]),local_bytes=int(re.search(r'LOCAL:(\d+)',b)[1])))
assert native and all(r['local_bytes']==0 for r in native),native
old=triton.runtime.driver.active.get_current_target;triton.runtime.driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
x=torch.empty(1,device='cuda',dtype=torch.bfloat16);p=torch.empty(1,device='cuda');flags=torch.empty(1,device='cuda',dtype=torch.int32);rows=[]
try:
 for t in (16,32):
  for name in ('_gb_kernel','_br_publish_kernel'):
   k=getattr(m,name);v=dict(X=x,W=x,P=p,FLAGS=flags,T=t,x_stride=4096,N=8192,K=4096,S=2,BN=64,BK=128,TB=t,NS=4,USE_PDL=True)
   c=k.warmup(*[v[a] for a in k.arg_names],grid=(1,1,1),num_warps=4,launch_pdl=True)
   path=folder/(name+str(t));path.with_suffix('.ptx').write_text(c.asm['ptx']);cub=path.with_suffix('.cubin')
   subprocess.run(['ptxas','-arch=sm_90a','-O3',str(path.with_suffix('.ptx')),'-o',str(cub)],check=True,capture_output=True)
   sass=subprocess.check_output(['nvdisasm','-g',str(cub)],text=True);path.with_suffix('.sass').write_text(sass);u=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(cub)],text=True)
   ins=[s.strip() for s in sass.splitlines() if re.search(r'/\*[0-9a-f]+\*/',s) and ';' in s];release=next(i for i,s in enumerate(ins) if 'PREEXIT' in s)
   row=dict(t=t,name=name,registers=int(re.search(r'REG:(\d+)',u)[1]),local_bytes=int(re.search(r'LOCAL:(\d+)',u)[1]),shared=c.metadata.shared,ldgsts=sum('LDGSTS' in s for s in ins),mma=sum('HGMMA' in s for s in ins),before_release=[s for s in ins[:release] if 'STG' in s or 'MEMBAR' in s],cubin_sha256=hashlib.sha256(cub.read_bytes()).hexdigest());rows.append(row)
   if name=='_br_publish_kernel':assert row['before_release'] and row['local_bytes']==0
  a,b=rows[-2:];assert a['ldgsts']==b['ldgsts'] and a['mma']==b['mma'] and a['shared']==b['shared']
finally:triton.runtime.driver.active.get_current_target=old
(HERE/'sass_validation.json').write_text(json.dumps(dict(native=native,publication=rows,native_source_sha256=hashlib.sha256(m.BR_CUDA.encode()).hexdigest()),indent=2)+'\n')
print('PASS native no spills, publisher reset precedes release, async GEMV pipeline intact')
