"""Extract device source without importing candidates; instantiate original device templates."""
import ast, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'prof'/'sass';out.mkdir(exist_ok=True)
paths={'moe_v6':'problems/muse_moe_w4a16/cand_v6.py','moe':'problems/muse_moe_w4a16/cand_v4.py','attention':'problems/muse_attention_decode/cand_v2.py','proj':'problems/muse_bf16_proj/cand_skinny.py'}
manifest={}
for name,rel in paths.items():
 p=root/rel;txt=p.read_text(); tree=ast.parse(txt)
 src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
 src=src.replace('#include <torch/extension.h>','#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>','')
 if name.startswith('moe'):
  src=src[:src.index('size_t smem_bytes(')]
  sig='template __global__ void moe_fused<{nt}>(const __nv_bfloat16*, const int*, const float*, const __nv_bfloat16*, const uint8_t*, const uint8_t*, const float*, const float*, __nv_bfloat16*, float*, float*, int*, __nv_bfloat16*, int, int, int);\n'
  src+='\n'+''.join(sig.format(nt=i) for i in (1,2))+'\n}\n'
 elif name=='attention':
  src=src[:src.index('template <int L, int STAGES>\nconstexpr int smem_bytes()')]
  src+='\ntemplate __global__ void attn_main<1,4>(const Params);\ntemplate __global__ void attn_main<8,3>(const Params);\n}\n'
 else:
  src=src[:src.index('template <int NT, int TR, int U, int NA, int XP, int LD>\nvoid launch_t')]
  # Compile every template reachable from current CONFIGS, including router and NT8.
  configs=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIGS' for t in n.targets))
  instances=sorted({(nt,v[0],v[1],2 if layer=='router' else 1,v[5],0) for (layer,nt),v in configs.items()})
  sig='template __global__ void skinny_kernel<{v}>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, void*, int, int, int, int, int, int);\n'
  src+='\n'+''.join(sig.format(v=','.join(map(str,x))) for x in instances)+'\n}\n'
 (out/f'{name}.cu').write_text(src)
 manifest[name]={'source':rel,'sha256':hashlib.sha256(txt.encode()).hexdigest(),'extracted_sha256':hashlib.sha256(src.encode()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
