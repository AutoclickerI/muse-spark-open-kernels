"""Generate sparse loads inside our exact winning Triton loop/schedule bodies."""
import ast
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];tree=ast.parse((R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text())
names={'_gb_kernel','_gi_kernel','_gemv_kernel','_gate_split_kernel','_vdsk_gemm'}
out=[]
for n in tree.body:
 if not isinstance(n,ast.FunctionDef) or n.name not in names:continue
 n.name='_dc'+n.name
 for arg in n.args.args:
  if arg.arg=='K':arg.annotation=ast.Attribute(ast.Name('tl',ast.Load()),'constexpr',ast.Load())
 s=ast.unparse(n)
 s=s.replace('w = tl.load(W + n[:, None] * K + (k + kk)[None, :])','w = dc_load(W,n[:,None],(k+kk)[None,:],K)')
 s=s.replace("w = tl.load(W + n[:, None] * K + (kb * BK + kk)[None, :], cache_modifier='.cg' if CG else '')",'w = dc_load(W,n[:,None],(kb*BK+kk)[None,:],K)')
 s=s.replace('w = w_desc.load([pw * BW, kk * BK])','w = dc_load(W,rw[:,None],(kk*BK+rk)[None,:],K)')
 s=s.replace('w = tl.load(W + rw[:, None] * K + (kk * BK + rk)[None, :])','w = dc_load(W,rw[:,None],(kk*BK+rk)[None,:],K)')
 # Encoded weights have their own transport; there is no raw-BF16 prefetch.
 if n.name=='_dc_gemv_kernel':
  a=s.index('    if PREFETCH:');b=s.index('    _pdl_wait',a);s=s[:a]+s[b:]
 assert 'w = tl.load(W' not in s and 'w_desc.load' not in s,s
 compile(s,'<sparse triton>','exec');out.append(s)
(H/'triton.py.inc').write_text('\n\n'.join(out)+'\n')
