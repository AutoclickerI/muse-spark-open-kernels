"""Check exact unpool tile hints preserve the sm90a asynchronous pipeline."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
import torch,triton
from triton.backends.compiler import GPUTarget
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'pdl_audit'))
from audit_native import load,compile_one
m=load(HERE/'release_hint_trial.py');folder=HERE/'native';folder.mkdir(exist_ok=True)
old=triton.runtime.driver.active.get_current_target;triton.runtime.driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
x=torch.empty(1,device='cuda',dtype=torch.bfloat16);p=torch.empty(1,device='cuda');rows=[]
try:
 for t in (16,32):
  for name in ('_gb_kernel','_bd_unpool_kernel'):
   k=getattr(m,name);vals=dict(X=x,W=x,P=p,T=t,x_stride=4096,N=8192,K=4096,S=2,BN=64,BK=128,TB=max(16,t),NS=4,USE_PDL=True)
   c=k.warmup(*[vals[a] for a in k.arg_names],grid=(1,1,1),num_warps=4,launch_pdl=True)
   path=folder/(name+str(t));path.with_suffix('.ptx').write_text(c.asm['ptx']);cub=path.with_suffix('.cubin')
   subprocess.run(['ptxas','-arch=sm_90a','-O3',str(path.with_suffix('.ptx')),'-o',str(cub)],check=True,capture_output=True)
   sass=subprocess.check_output(['nvdisasm','-g',str(cub)],text=True);path.with_suffix('.sass').write_text(sass)
   usage=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(cub)],text=True)
   ins=[s for s in sass.splitlines() if re.search(r'/\*[0-9a-f]+\*/',s) and ';' in s];wait=next(i for i,s in enumerate(ins) if 'ACQBULK' in s)
   rows.append(dict(t=t,name=name,registers=int(re.search(r'REG:(\d+)',usage)[1]),shared=c.metadata.shared,ldgsts=sum('LDGSTS' in s for s in ins),mma=sum('HMMA' in s or 'HGMMA' in s for s in ins),pre_wait=[s.strip() for s in ins[:wait] if 'CCTL' in s],cubin_sha256=hashlib.sha256(cub.read_bytes()).hexdigest()))
  a,b=rows[-2:];assert all(a[k]==b[k] for k in ('registers','shared','ldgsts','mma')),(a,b);assert b['pre_wait'] and b['mma']>0,b
finally:triton.runtime.driver.active.get_current_target=old
r=compile_one(('SMALLM_VAR_CUDA',m.SMALLM_VAR_CUDA),folder)
assert r['ok'] and all(k['releases'] and k['releases'][0]<k['waits'][0] for k in r['kernels'] if k['waits'])
(HERE/'sass_validation.json').write_text(json.dumps(dict(unpool=rows,moe=r),indent=2)+'\n');print('PASS unpool pipeline unchanged; all MoE releases precede wait',len(r['kernels']))
