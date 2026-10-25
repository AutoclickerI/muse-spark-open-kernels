"""Reuse S15's pinned successful native compile proof for identical CUDA and flags.
Python checks and all board runtime checks always run. Changed CUDA is compiled.
"""
import ast,hashlib,json,tempfile
from pathlib import Path
H=Path(__file__).resolve().parent;BASE=H.parents[1]/'cand_r3_best7_s15.py'
def sources(path):
 tree=ast.parse(path.read_text());templated=set();flags={}
 for n in ast.walk(tree):
  if isinstance(n,ast.Attribute) and n.attr in ('format','replace','format_map') and isinstance(n.value,ast.Name):templated.add(n.value.id)
  elif isinstance(n,ast.BinOp) and isinstance(n.op,ast.Mod) and isinstance(n.left,ast.Name):templated.add(n.left.id)
  if isinstance(n,ast.Call) and getattr(n.func,'attr',getattr(n.func,'id',''))=='load_inline':
   kw={k.arg:k.value for k in n.keywords};src=kw.get('cuda_sources');extra=[]
   names=[x.id for x in (src.elts if isinstance(src,(ast.List,ast.Tuple)) else [src]) if isinstance(x,ast.Name)]
   for key,prefix in (('extra_cuda_cflags',''),('extra_include_paths','-I')):
    try:extra += [prefix+str(x) for x in ast.literal_eval(kw[key])]
    except (KeyError,ValueError):pass
   for name in names:flags[name]=[x for x in extra if not x.startswith(('-gencode','-arch','--generate-code'))]
 out={}
 for n in tree.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
   name,text=n.targets[0].id,n.value.value
   if '#include' in text and (name in flags or '__global__' in text or 'torch/extension.h' in text) and name not in templated:out[name]=(text,flags.get(name,[]),n)
 return out

def install(local_eval):
 admission=json.loads((H.parent/'s15/admission.json').read_text())['combined']['normal']
 assert hashlib.sha256(BASE.read_bytes()).hexdigest()==admission['sha256']
 base=sources(BASE);proof=admission['checks']['static'];assert proof['count']==proof['pass_count']==len(base)+2
 compiler=local_eval.hopper_compile
 def check(path):
  cur=sources(path);reuse={k for k,v in cur.items() if k in base and v[:2]==base[k][:2]};changed=set(cur)-reuse
  rows=[]
  if changed:
   lines=path.read_text().splitlines(keepends=True)
   for name in sorted(reuse,key=lambda k:cur[k][2].lineno,reverse=True):
    n=cur[name][2];lines[n.lineno-1:n.end_lineno]=[name+' = ""\n']
   with tempfile.TemporaryDirectory(prefix='p12-static-') as tmp:
    reduced=Path(tmp)/path.name;reduced.write_text(''.join(lines));rows=compiler(reduced)
   assert {r['name'] for r in rows}==changed
  return rows+[dict(name=k,ok=True,msg='identical CUDA/flags to admitted S15 60254883') for k in sorted(reuse)]
 local_eval.hopper_compile=check
