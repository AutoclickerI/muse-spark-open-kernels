"""Compile changed dictionary-held hybrid CUDA with the shipped native macros."""
import ast,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parents[3]/'tools'))
from local_eval import hopper_compile
s=(H/'generated/s22.py').read_text();sources=next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_HY_SOURCES');code='';count=0
for name,source in sources.items():
 for n in ast.parse(source).body:
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CUDA':
   key='HY_'+name.upper();code+=key+'='+repr(ast.literal_eval(n.value))+"\nload_inline(cuda_sources="+key+",extra_cuda_cflags=['-O3','-lineinfo','-DRSW=4','-DPRS=4','-DDRS=4','-DRBF=5'])\n";count+=1
p=H/'generated/hybrid_native_units.py';p.write_text(code);checks=hopper_compile(p);assert len(checks)==count>0;(H/'generated/native_compile.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks);assert all(c['ok'] for c in checks)
