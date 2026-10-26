"""Exact sm90a SASS/resources before any H200 run; no timing inference from static counts."""
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_operand'
cells=json.loads((D/'cells.json').read_text());modes=json.loads((D/'modes.json').read_text());names={v:k for k,v in modes.items()};rows=[]
occ=O/'occupancy'
subprocess.run(['g++','-O2','-I/usr/local/cuda/include',str(D/'occupancy.cpp'),'-o',str(occ)],check=True)
for i in range(3):
 p=O/f'opa{i}.cu';assert hashlib.sha256(p.read_bytes()).hexdigest()==cells[str(i)]['cuda_sha256']
 z=subprocess.run(['nvcc','-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-c',str(p),'-o',str(p.with_suffix('.o')),'-Xptxas=-v'],text=True,capture_output=True)
 log=z.stdout+z.stderr;p.with_suffix('.compile.log').write_text(log);z.check_returncode()
 sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass)
 for fn,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S):
  f=95 if i==0 else 69;op=re.search(r'op_varILi1ELi'+str(f)+r'ELi(\d+)E',fn);base=re.search(r'moe_varILi1ELi'+str(f)+r'E',fn);lad=re.search(r'lad_varILi1ELi'+str(f)+r'ELi([1234])ELb([01])E',fn)
  if lad and not ((lad[1] in '123' and lad[2]=='0') or (lad[1]=='4' and lad[2]=='1')):lad=None
  if not(op or base or lad):continue
  arm=names[int(op[1])] if op else 'BASE' if base else 'HIST' if lad[1]=='4' else 'L'+lad[1]
  v=re.search(r'Function properties for '+re.escape(fn)+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers([^\n]*)',log);assert v,fn
  stack,ss,sl,regs=map(int,v.groups()[:4]);sm=re.search(r'(\d+) bytes smem',v[5]);stat=int(sm[1]) if sm else 0
  ops=dict(collections.Counter(x.split('.')[0] for x in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',body)))
  for M in ([1] if i==0 else [8] if i==1 else [16,32]):
   dyn=2*36864+8192+32+1904+((M*8+15)&~15)+4*M*8+(8320 if arm in ('FRAG_SMEM','FRAG_JIT','BASE_PAD') else 0)
   resident=int(subprocess.check_output([str(occ),str(regs),str(dyn),str(stat)],text=True))
   rows.append(dict(M=M,arm=arm,registers=regs,stack=stack,spill_stores=ss,spill_loads=sl,shared=dyn,static=stat,resident=resident,ops=ops,cell_sha256=cells[str(i)]['cell_sha256'],cuda_sha256=cells[str(i)]['cuda_sha256']))
  print(i,arm,regs,stack,ss,sl,ops.get('HMMA',0),flush=True)
for M in (1,8,16,32):
 rr={v['arm']:v for v in rows if v['M']==M};assert len(rr)==14,(M,rr.keys())
 for arm,v in rr.items():assert v['ops'].get('HMMA',0)==(0 if arm=='L1' else 32),(M,arm)
qualified=sorted({v['arm'] for v in rows if all(z['resident']==2 and z['spill_loads']==0 and z['spill_stores']==0 for z in rows if z['arm']==v['arm'])})
assert all(a in qualified for a in ('BASE','BASE_PAD','PAD_CTRL','LAZY'))
(D/'counts.json').write_text(json.dumps(dict(rows=rows,qualified=qualified,dropped=sorted(set(v['arm'] for v in rows)-set(qualified)),scope='Static code/resource witnesses;dynamic padding counts come from the per-expert token histogram;not a saturation model'),indent=2)+'\n')
print('QUALIFIED',qualified,flush=True)
