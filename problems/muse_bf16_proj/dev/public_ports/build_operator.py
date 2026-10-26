"""Reviewed kernel adaptations: FP32 fused-router stores and explicit PDL twin."""
import ast,json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;s=(H/'operatorstack.py.inc').read_text();original=s
# Tensor store type supplies the final conversion; BF16 stores are bit-identical.
# Fused pool/router instead keeps the original FP32 accumulator, never rounds it.
s=s.replace('tl.sum(accum, axis=1).to(tl.bfloat16)','tl.sum(accum, axis=1)').replace('tot.to(tl.bfloat16)','tot').replace('acc.to(tl.bfloat16)','acc')
raw=s;lines=s.splitlines(keepends=True);tree=ast.parse(s);names=[]
for n in reversed(tree.body):
 if not isinstance(n,ast.FunctionDef) or not n.decorator_list:continue
 names.append(n.name);first=n.body[0];i=first.end_lineno if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str) else first.lineno-1
 lines[i:i]=['    tl.extra.cuda.gdc_launch_dependents()\n','    tl.extra.cuda.gdc_wait()\n']
pdl=''.join(lines)
for name in names:pdl=pdl.replace('def '+name+'(', 'def '+name+'_ppdl(')
# Keep one import/constant/ASM-helper copy; duplicate only JIT definitions.
tree=ast.parse(pdl);pdl='\n\n'.join('@triton.jit\n'+ast.get_source_segment(pdl,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.decorator_list)
out='\n'.join(l.rstrip() for l in (raw+'\n'+pdl).splitlines())+'\n';compile(out,'<reviewed operator projection kernels>','exec');(H/'operator_kernels.py.inc').write_text(out)
(H/'adaptation.json').write_text(json.dumps(dict(reviewed_sha256=hashlib.sha256(original.encode()).hexdigest(),adapted_sha256=hashlib.sha256(out.encode()).hexdigest(),changes=['Implicit pointer-typed stores preserve FP32 fused pool/router; BF16 stores unchanged','Separate PDL twin releases/waits at entry; raw donor kernels retained separately','No donor prepare/dispatch/fallback/driver copied'],kernels=sorted(names)),indent=2)+'\n')
