"""Reuse the existing 28-case eager/graph ABI screen for the exact ready patch."""
import ast,base64,lzma
from pathlib import Path
H=Path(__file__).resolve().parent
def expanded_preview(raw):
 # The shared compile cache scans literal top-level sources. Give it an
 # expanded file, so its keys and local_eval's expanded-source scan agree.
 tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and getattr(n.value.func,'id','')=='exec')
 packed=next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_source' for t in n.targets))
 code=max((x.value for x in ast.walk(packed.value) if isinstance(x,ast.Constant) and isinstance(x.value,str)),key=len)
 return lzma.decompress(base64.b85decode(code)).decode()+'\n'+''.join(raw.splitlines(True)[node.end_lineno:])
s=(H.parent/'public_hybrid/screen_candidate.py').read_text()
s=s.replace("raw=(o/'candidate.py').read_text()","raw=(o/'all_F.py').read_text()")
needle="  if hasattr(m,'_p5b_merge_setup'):m._p5b_merge_setup()"
assert needle in s
s=s.replace(needle,"""  for n in sorted(m._PV_SOURCES):
   z=m._pv_module(n)
   if n.startswith('op_'):v.operator(z,a.forced)
   elif n.startswith('lee_'):v.lee(z)
   if hasattr(z,'setup'):z.setup('cuda')
"""+needle)
needle="  holder['m']=m"
assert needle in s
s=s.replace(needle,"""  def compile_expanded(path,_compile=ev.hopper_compile):
   expanded=path.with_name(path.stem+'_expanded_compile.py')
   expanded.write_text(expanded_preview(path.read_text()))
   return _compile(expanded)
  ev.hopper_compile=compile_expanded
"""+needle)
exec(compile(s,str(H/'screen_candidate.py'),'exec'),globals())
