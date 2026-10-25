"""Compile reviewed standalone router selections for static resource evidence only.
No upstream Python module is imported; AST closure builders emit CUDA into ignored files.
"""
import ast,concurrent.futures,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
EXP=Path('/home/jihoon/.tmp-live-20260919/runtime/claude-1016/-home-jihoon-competitions-muse-kernel/ae5ada91-d8f5-49d0-a518-22df30a603ba/scratchpad/ext_expanded/nf/submission_expanded.py')
t=ast.parse(EXP.read_text());defs={}
for n in t.body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef)):defs[n.name]=n
 elif isinstance(n,ast.Assign):
  for x in n.targets:
   if isinstance(x,ast.Name):defs[x.id]=n
needed={'build_router'};todo=list(needed)
while todo:
 for n in ast.walk(defs[todo.pop()]):
  if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in defs and n.id not in needed:needed.add(n.id);todo.append(n.id)
ids={id(defs[n]) for n in needed};nodes=[n for n in t.body if id(n) in ids];sources=[]
def capture(name,cpp_sources='',cuda_sources='',**kw):sources.append(('router_generic',cuda_sources))
ns={'hashlib':hashlib,'load_inline':capture};exec(compile(ast.Module(nodes,type_ignores=[]),'<reviewed-router-builders>','exec'),ns)
ns['build_router']([(32,16,4,1,2,1)])
sources.append(('routertail',ast.literal_eval(defs['rt__CUDA'].value)))
for tokens,source in ast.literal_eval(defs['rw_SOURCES'].value).items():sources.append(('routerwide'+str(tokens),source))
out=H/'native_original';out.mkdir(exist_ok=True)
def build(item):
 name,source=item;f=out/(name+'.cu');f.write_text(source)
 r=subprocess.run(['nvcc','-std=c++20','-O3','-arch=sm_90a','--cubin','-Xptxas=-v',str(f),'-o',str(f.with_suffix('.cubin'))],capture_output=True,text=True)
 f.with_suffix('.log').write_text(r.stdout+r.stderr)
 row=dict(name=name,sha256=hashlib.sha256(source.encode()).hexdigest(),ok=r.returncode==0,error=(r.stdout+r.stderr)[-1200:] if r.returncode else '')
 print(row,flush=True);return row
with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(build,sources))
(H/'router_compile.json').write_text(json.dumps(rows,indent=2)+'\n');assert all(x['ok'] for x in rows)
