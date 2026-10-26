"""Adapt operator GEMMs to S19's actual sorted FP32 Y / 64-part SS ABI."""
import ast

def operator_abi(source):
 t=ast.parse(source);nodes=[]
 wanted={'_w13_item','_w13_mix','_w2_item','_w2_mix'}
 for n in t.body:
  if not isinstance(n,ast.FunctionDef) or n.name not in wanted:continue
  s=ast.get_source_segment(source,n)
  # get_source_segment omits decorators; append the original decorator.
  s='@gluon.jit\n'+s
  old=n.name
  if old.startswith('_w13'):
   s=s.replace('PF: gl.constexpr):','PF: gl.constexpr, ROWMAP, SS):').replace('PF: gl.constexpr = False):','PF: gl.constexpr = False, ROWMAP=None, SS=None):')
   s=s.replace('CHK=CHK, PF=PF)', 'CHK=CHK, PF=PF, ROWMAP=ROWMAP, SS=SS)')
   if old=='_w13_item':
    s=s.replace('optr = out_ptr + gl.expand_dims(pair_o, 0)', 'mapped = gl.load(ROWMAP + pair_o, mask=ro < r1, other=0)\n    optr = out_ptr + gl.expand_dims(mapped, 0)')
    s+='\n    if pid_n < 64:\n        gl.store(SS + mapped * 64 + pid_n, gl.zeros([BM],gl.float32,layout=gl.SliceLayout(0,mma)), mask=ro < r1)\n'
  else:
   s=s.replace('PF: gl.constexpr):','PF: gl.constexpr, ROWMAP, PNW, SS):').replace('PF: gl.constexpr = False):','PF: gl.constexpr = False, ROWMAP=None, PNW=None, SS=None):')
   s=s.replace('CHK=CHK, PF=PF)', 'CHK=CHK, PF=PF, ROWMAP=ROWMAP, PNW=PNW, SS=SS)')
   if old=='_w2_item':
    s=s.replace('x_off = pair * K', 'x_off = gl.load(ROWMAP + pair, mask=rv < r1, other=0) * K')
    s=s.replace('_xg_offsets(sorted_ptr, r0, r1, K, False, BM)', '_abi_xg_offsets(sorted_ptr, ROWMAP, r0, r1, K, BM)')
    s=s.replace('optr = out_ptr + gl.expand_dims(pair_o, 0)', 'mapped = gl.load(ROWMAP + pair_o, mask=ro < r1, other=0)\n    nw = gl.load(PNW + n0 + n_o).to(gl.float32) + 1.0\n    y = y * gl.expand_dims(nw, 1)\n    optr = out_ptr + gl.expand_dims(mapped, 0)')
    s=s.replace('gl.store(optr, y.to(gl.bfloat16),', 'gl.store(optr, y,')
    s+='\n    norm = gl.sum(y*y, 0)\n    gl.store(SS + mapped*64 + pid_n, norm, mask=ro < r1)\n'
  for name in wanted:s=s.replace(name,'_abi'+name)
  nodes.append(s)
 return '\n\n'.join(nodes)
