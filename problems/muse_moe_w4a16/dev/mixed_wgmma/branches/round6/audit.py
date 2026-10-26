"""Admit repaired native kernels; permit only outer-loop metadata spills."""
import collections
import hashlib
import json
from pathlib import Path
import re
H=Path(__file__).resolve().parent
cu=(H/'repair.cu').read_text();log=Path('/tmp/p3b_r6_ptxas.log').read_text();sass=Path('/tmp/p3b_r6.sass').read_text()
assert not re.search(r'C751[239]|C7520|error',log)
rows=[];kernels=[];tail_shapes=[]
for block in log.split('Compiling entry function ')[1:]:
 z=re.search(r'chunk6kernelILi(\d)ELb(\d)',block.splitlines()[0])
 if not z:continue
 frame=list(map(int,re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',block).groups()));assert frame[0]<=8 and max(frame[1:])<=16
 regs=int(re.search(r'Used (\d+) registers',block)[1]);assert regs<=128
 smem={3:76800,4:102400}[int(z[1])];assert smem*2<=227328 and regs*512<=65536
 rows.append(dict(stages=int(z[1]),first=int(z[2]),registers=regs,stack_spill=frame,dynamic_smem=smem,two_CTA_resource_admission=True))
for f in sass.split('Function : ')[1:]:
 if 'tail6kernel' in f.splitlines()[0]:
  shapes=sorted(set(re.findall(r'HGMMA\.([^\s]+)',f)))
  for n in (16,32,64):assert f'64x{n}x16.F32.BF16' in shapes,(f.splitlines()[0],n,shapes)
  tail_shapes.append(dict(name=f.splitlines()[0].strip(),BF16_shapes=shapes))
 if 'chunk6kernel' not in f.splitlines()[0]:continue
 inst={int(pc,16):t.strip() for pc,t in re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',f)}
 loops=[]
 for pc,t in inst.items():
  z=re.search(r'\bBRA(?:\.[A-Z]+)* (0x[0-9a-f]+)',t)
  if z and int(z[1],16)<pc:
   a=int(z[1],16)
   if any('HGMMA' in text for pp,text in inst.items() if a<=pp<=pc):loops.append((a,pc))
 inner=[(a,b) for a,b in loops if not any(a<=x and y<=b and (a,b)!=(x,y) for x,y in loops)]
 assert len(inner)==2
 spills=[(pc,t) for pc,t in inst.items() if re.search(r'\b(?:LDL|STL)\b',t)]
 assert not any(a<=pc<=b for pc,t in spills for a,b in inner),'spill inside K loop'
 ops=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[UP]\d+\s+)?([A-Z][A-Z0-9_.]+)',f))
 assert 'WARPGROUP.DEPBAR.LE gsb0, 0x1' in f and 'BAR.SYNC' in f
 kernels.append(dict(name=f.splitlines()[0].strip(),K_loops=inner,outer_only_spills=spills,instructions=dict(ops)))
assert len(rows)==len(kernels)==4 and len(tail_shapes)==6
out=dict(cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),resources=rows,kernels=kernels,tail_native_shapes=tail_shapes,sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),admission='Up to8B local stack for outer expert/row metadata, no LDL/STL within either NT32/64 K loop; allowing the tiny outer spill buys2CTA register admission. Every tail specialization contains native BF16 N16/32/64, unlike the failed original.',limitation='S4 uses S3 only on sm120 because102400B exceeds its101376B opt-in limit. Native S4 and2CTA runtime admission need H200.')
(H/'resources.json').write_text(json.dumps(out,indent=2)+'\n');print(rows)
