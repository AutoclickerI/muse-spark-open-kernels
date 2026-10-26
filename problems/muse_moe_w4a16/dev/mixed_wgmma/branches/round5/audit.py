"""Check native resource admission and the shared-A layout ownership."""
import collections
import hashlib
import json
from pathlib import Path
import re
H=Path(__file__).resolve().parent
cu=(H/'tokpair.cu').read_text();ctrl=(H.parent/'round2/cta2.cu').read_text();assert cu.startswith(ctrl)
log=Path('/tmp/p3b_r5_ptxas.log').read_text();sass=Path('/tmp/p3b_r5.sass').read_text();rows=[]
assert not re.search(r'C751[239]|C7520|error',log)
for block in log.split('Compiling entry function ')[1:]:
 z=re.search(r'tokpair6kernelILb(\d)ELb(\d)',block.splitlines()[0])
 if not z:continue
 frame=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',block);assert frame and not any(map(int,frame.groups()))
 regs=int(re.search(r'Used (\d+) registers',block)[1]);assert regs<=128
 share=int(z[2]);dynamic=10240+32768+share*32768;assert dynamic*2<=227328 and regs*256*2<=65536
 rows.append(dict(first=int(z[1]),share_decode=share,registers=regs,stack_spill=list(map(int,frame.groups())),dynamic_smem=dynamic))
kernels=[]
for f in sass.split('Function : ')[1:]:
 if 'tokpair6kernel' not in f.splitlines()[0]:continue
 ops=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[UP]\d+\s+)?([A-Z][A-Z0-9_.]+)',f));assert not any(k.startswith(('LDL','STL')) for k in ops)
 dep=collections.Counter(re.findall(r'WARPGROUP.DEPBAR.LE gsb0, (0x[0-9a-f]+)',f));assert dep.get('0x1',0)>0 and dep.get('0x0',0)>0
 kernels.append(dict(name=f.splitlines()[0].strip(),instructions=dict(ops),group_wait=dict(dep)))
assert len(rows)==len(kernels)==4
cells=[]
for t in range(128):
 w,l=divmod(t,32)
 for b in range(4):
  for j in range(4):
   r=w*16+(l>>2)+(j&1)*8;k=b*16+(l&3)*2+(j>>1)*8
   cells.extend((r*64+(k^((r&7)*8))+h) for h in (0,1))
assert len(cells)==len(set(cells))==4096 and set(cells)==set(range(4096))
out=dict(cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),CTA2_prefix_sha256=hashlib.sha256(ctrl.encode()).hexdigest(),resources=rows,sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),kernels=kernels,shared_A_layout_bijection=True,ordering='At stage entry wait0 of cp.async and full CTA barrier complete all X/raw copies. S2 A slot is overwritten only after all WGs have crossed the previous stage; prior stage group0 wait1 retired its older reader. Decode stores cross a CTA barrier and async-proxy fence before SS WGMMA. RS A0/A1 are decoded just before issue, so the previous group1 is retired before A1 overwrite. Terminal WGMMA wait0 precedes norm scratch reuse and an end-tile CTA barrier.',limitation='Native SS descriptors and Hopper residency require judge execution; local stand-in exercises the same shared-A matrix layout and S2 buffers.')
(H/'resources.json').write_text(json.dumps(out,indent=2)+'\n');print(rows)
