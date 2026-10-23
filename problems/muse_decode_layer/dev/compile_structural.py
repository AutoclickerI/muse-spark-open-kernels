"""Compile the exact device templates for sm_90a, without attempting Hopper execution."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]
out=p/'dev'/(sys.argv[2] if len(sys.argv)>2 else 'sm90a');out.mkdir(exist_ok=True)
f=p/(sys.argv[1] if len(sys.argv)>1 else 'cand_chain6.py')
src=f.read_text();tree=ast.parse(src)
constants={t.id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) for t in n.targets if isinstance(t,ast.Name)}
manifest={'source':str(f),'sha256':hashlib.sha256(src.encode()).hexdigest(),'templates':{}}
for name,key in [('dense','PERSIST_CUDA'),('moe','mv4_CUDA_SRC'),('attention','CUDA_SRC')]:
 cu=constants[key].replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
 if name=='dense':
  cu=cu[:cu.index('template <int NT, bool PREFIX>\nvoid launch')]
  cu+='\n'+''.join(f'template __global__ void dense_island<{nt},{pre}>(Params);\n' for nt in (1,2,4) for pre in ('true','false'))+'}\n'
 elif name=='moe':
  cu=cu[:cu.index('size_t smem_bytes(')]
  cu+='\ntemplate __global__ void moe_fused<1>(const __nv_bfloat16*,const int*,const float*,const __nv_bfloat16*,const uint8_t*,const uint8_t*,const float*,const float*,__nv_bfloat16*,float*,float*,int*,__nv_bfloat16*,int,int,int,bool);\n}\n'
 else:
  cu=cu[:cu.index('template <int L, int STAGES>\nvoid launch_t')]
  cu+='\ntemplate __global__ void attn_main<1,4>(Params);\ntemplate __global__ void attn_combine<1>(Params);\n}\n'
 path=out/(name+'.cu');path.write_text(cu)
 cmd=['nvcc','-cubin','-arch=sm_90a','-O3','-std=c++20','--expt-relaxed-constexpr','-lineinfo','-Xptxas=-v',str(path),'-o',str(out/(name+'.cubin'))]
 proc=subprocess.run(cmd,capture_output=True,text=True)
 (out/(name+'.compile.log')).write_text(proc.stdout+proc.stderr)
 print(name,proc.returncode,flush=True)
 manifest['templates'][name]={'status':proc.returncode,'sha256':hashlib.sha256(cu.encode()).hexdigest()}
 if proc.returncode: print(proc.stderr[-3000:])
 else:
  res=subprocess.run(['cuobjdump','--dump-resource-usage',str(out/(name+'.cubin'))],capture_output=True,text=True)
  (out/(name+'.resources.txt')).write_text(res.stdout)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if any(x['status'] for x in manifest['templates'].values()):sys.exit(1)
