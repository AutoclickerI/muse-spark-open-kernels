from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from vendor import vendor
root=Path(__file__).resolve().parents[1]
s=(root/'cand_moe4_attn.py').read_text()
p=vendor(root.parent/'muse_bf16_proj/cand_skinny.py','pr_')
p=p.replace('constexpr bool OUTF32 = NA == 2;', 'constexpr bool OUTF32 = true;')
p=p.replace('const size_t idx = (size_t)(tok0 + tok) * N + row_cta + row;', 'const size_t idx = ((size_t)blockIdx.y * M + tok0 + tok) * N + row_cta + row;')
a=p.index('  const int KS = gridDim.y;');b=p.index('\n}\n\ntemplate <int NT',a)
p=p[:a]+'''  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    const int tok = e / BR, row = e - tok * BR;
    if (tok0 + tok < M) store(tok, row, sred[tok * S + row]);
  }'''+p[b:]
a=p.index('  if (ks > 1) {');b=p.index('\n  auto err = cudaLaunchKernelEx',a)
p=p[:a]+p[b:]
s+='\n\n# Direct-register BF16 MMA streaming, with split reduction in consumers.\n'+p
s+='''

pr_INSTANCES = [(1,1,4,1,1,0), (1,2,4,1,1,0), (2,1,4,1,1,0),
                (2,2,4,1,1,0), (4,1,2,1,1,0), (4,1,4,1,1,0)]
_gemv_before_native = gemv
_prepare_before_native = prepare

def gemv(x, w, out_dtype=torch.bfloat16):
    t,k = x.shape
    if t > 32:
        return _gemv_before_native(x,w,out_dtype)
    n = w.shape[0]
    nt = 1 if t <= 8 else 2 if t <= 16 else 4
    if n == QKV_SIZE:
        tr,u,wn,wk,ks,xp = (2,4,1,8,1,1) if t > 1 and t <= 8 else (1,4,4,1,8,1) if t <=16 else (1,2,4,1,8,1)
    elif n == D_EXPERT + 2*N_EXPERTS:
        tr,u,wn,wk,ks,xp = (1,4,1,4,1,1) if t <=8 else (2,4,1,8,1,1) if t<=16 else (1,4,2,4,1,1)
    elif k == Q_SIZE:
        tr,u,wn,wk,ks,xp = (1,4,1,4,1,1) if t<=16 else (1,4,2,4,1,1)
    else:
        tr,u,wn,wk,ks,xp = (1,4,1,8,1,1) if t<=16 else (1,2,2,4,1,1)
    out = torch.empty((ks,t,n), device=x.device, dtype=torch.float32)
    pr__module().skinny(w,w,x,out,n,nt,tr,u,wn,wk,ks,False,xp,0)
    return out,ks


def prepare(weights):
    pr__module()
    return _prepare_before_native(weights)
'''
(root/'cand_native.py').write_text(s)
