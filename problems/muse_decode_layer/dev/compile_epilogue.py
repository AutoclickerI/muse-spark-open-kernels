import importlib.util,json,hashlib
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
p=Path(__file__).resolve().parents[1];out=p/'dev/sm90a';out.mkdir(exist_ok=True)
f=p/'cand_epilogue.py'
sp=importlib.util.spec_from_file_location('epilogue_static',f);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows=[]
for t in (1,8,16,32):
 tb=max(16,triton.next_power_of_2(t))
 jobs=[('o_stats',m._o_stats_kernel,dict(X='*bf16',W='*bf16',STREAM='*fp32',ALPHA='*fp32',BETA='*fp32',A='*bf16',STATS='*fp32',COEFF='*fp32',CNT='*i32'),dict(T=t,TB=tb,BN=64,BK=128,USE_PDL=True)),
  ('pool_boundary',m._pool_boundary_kernel,dict(A='*bf16',COEFF='*fp32',STREAM='*fp32',STREAM1='*fp32',ALPHA='*fp32',BETA='*fp32',PREW='*fp32',W='*bf16',P='*fp32'),dict(T=t,TB=tb,BN=64,KS=2048,BK=128,USE_PDL=True))]
 for label,fn,sig,const in jobs:
  name=f'{label}_t{t}'
  attrs={(fn.arg_names.index(a),):[('tt.divisibility',16)] for a in sig}
  k=triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=GPUTarget('cuda',90,32),options={'num_warps':4,'num_stages':3,'launch_pdl':True})
  for ext in ('ptx','cubin'):
   value=k.asm[ext];(out/(name+'.'+ext)).write_bytes(value if isinstance(value,bytes) else value.encode())
  md=k.metadata._asdict();rows.append({'name':name,'metadata':md});print(name,md,flush=True)
(out/'epilogue_manifest.json').write_text(json.dumps({'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'kernels':rows},indent=2,default=str))
