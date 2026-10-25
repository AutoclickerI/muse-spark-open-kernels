"""Instantiate each final native geometry and compile sm90a, keeping resource evidence."""
import json,runpy,subprocess,hashlib,concurrent.futures
from pathlib import Path
H=Path(__file__).resolve().parent;p=runpy.run_path(str(H/'adapter.py'));out=H/'native';out.mkdir(exist_ok=True)
units=[]
for wr,wk,s,st in ((1,8,1,1),(2,8,1,1),(1,4,1,1),(1,4,4,2)):
 for sparse,vector in ((False,False),(True,False),(True,True)):
  ns=p['_nf_vendor']();src=ns['V18S_CUDA_TEMPLATE']
  if sparse:src=p['sparse_source'](src,'v18',vector)
  ns['V18S_CUDA_TEMPLATE']=src
  def capture(name,cpp_sources='',cuda_sources='',**kw):units.append((f'w{wr}k{wk}s{s}p{st}sp{int(sparse)}v{int(vector)}',cuda_sources))
  ns['load_inline']=capture
  cfg=(wr,wk,128//(wk*s),s)
  insts=[(1,mt,1,st,2 if s>1 else 0,0,wr*wk*32,0,4608,8192,*cfg,6) for mt in (1,2,4)]
  ns['v18s_build'](insts,'sp_native')
def build(item):
 name,s=item;f=out/(name+'.cu');f.write_text(s)
 r=subprocess.run(['nvcc','-std=c++20','-O3','-arch=sm_90a','--cubin','-Xptxas=-v',str(f),'-o',str(f.with_suffix('.cubin'))],capture_output=True,text=True)
 f.with_suffix('.log').write_text(r.stdout+r.stderr)
 import re
 regs=[int(x) for x in re.findall(r'Used (\d+) registers',r.stderr)];spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',r.stderr)
 row=dict(name=name,sha256=hashlib.sha256(s.encode()).hexdigest(),ok=r.returncode==0,regs=regs,stack_spills=spills,error=(r.stdout+r.stderr)[-1500:] if r.returncode else '')
 print(row,flush=True);return row
with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=list(pool.map(build,units))
(H/'native_compile.json').write_text(json.dumps(rows,indent=2)+'\n');assert all(r['ok'] for r in rows)
