"""Admit the optimized Hopper binary and record the source identity it proves."""
import collections
import hashlib
import json
from pathlib import Path
import re
H=Path(__file__).resolve().parent
cu=(H/'ws.cu').read_text();ctrl=(H.parent/'round2/cta2.cu').read_text();assert cu.startswith(ctrl)
log=Path('/tmp/p3b_r4_ws_ptxas.log').read_text();sass=Path('/tmp/p3b_r4_ws.sass').read_text();rows=[];kernels=[]
assert not re.search(r'C751[239]|C7520|C7602|error',log),'unadmitted WGMMA compilation'
for block in log.split('Compiling entry function ')[1:]:
 z=re.search(r'ws6kernelILi(\d+)ELb(\d)',block.splitlines()[0])
 if not z:continue
 frame=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',block);assert frame and not any(map(int,frame.groups()))
 regs=int(re.search(r'Used (\d+) registers',block)[1]);assert regs==168
 n=int(z[1]);dynamic=2*(32768+n*256)+2*9216;static=1024;assert dynamic+static<=227328
 rows.append(dict(max_tokens=n,first=int(z[2]),entry_registers=regs,stack_spill=list(map(int,frame.groups())),dynamic_smem=dynamic,static_smem=static))
for f in sass.split('Function : ')[1:]:
 if 'ws6kernel' not in f.splitlines()[0]:continue
 op=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[UP]\d+\s+)?([A-Z][A-Z0-9_.]+)',f));assert not any(k.startswith(('LDL','STL')) for k in op)
 dep=collections.Counter(re.findall(r'WARPGROUP.DEPBAR.LE gsb0, (0x[0-9a-f]+)',f));assert dep.get('0x1',0)>0
 assert 'USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xd0' in f and 'USETMAXREG.DEALLOC.CTAPOOL 0x58' in f
 assert op['UTMALDG.2D'] and op['UBLKCP.S.G'],op  # tensor X TMA and linear raw-weight TMA
 inst={int(pc,16):text.strip() for pc,text in re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',f)}
 start=next(pc for pc,t in inst.items() if 'USETMAXREG.DEALLOC.CTAPOOL 0x58' in t);todo=[start+16];seen=set()
 while todo:
  pc=todo.pop()
  if pc in seen or pc not in inst:continue
  seen.add(pc);t=inst[pc]
  assert 'USETMAXREG.TRY_ALLOC' not in t,'producer unexpectedly reaches consumer allocation'
  regs=list(map(int,re.findall(r'\bR(\d+)\b',t)));assert not regs or max(regs)<88,(pc,t)
  branch=re.search(r'\bBRA(?:\.[A-Z]+)* (0x[0-9a-f]+)',t)
  if branch:
   todo.append(int(branch[1],16))
   if t.startswith('@'):todo.append(pc+16)
  elif re.search(r'\b(EXIT|RET)\b',t):
   if t.startswith('@'):todo.append(pc+16)
  else:todo.append(pc+16)
 all_regs=list(map(int,re.findall(r'\bR(\d+)\b',f)));assert max(all_regs)<208
 producer_max=max([int(r) for pc in seen for r in re.findall(r'\bR(\d+)\b',inst[pc])],default=0)
 kernels.append(dict(name=f.splitlines()[0].strip(),instructions=dict(op),group_wait=dict(dep),producer_CFG_instructions=len(seen),producer_max_physical_register=producer_max,overall_max_physical_register=max(all_regs)))
assert len(rows)==len(kernels)==8
assert (88+208+208)*128==168*384==64512
out=dict(cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),CTA2_prefix_sha256=hashlib.sha256(ctrl.encode()).hexdigest(),control_prefix_equal=True,sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),register_budget=dict(producer_decode=88,consumer0=208,consumer1=208,entry=168,total=64512),resources=rows,kernels=kernels,limitation='Native TMA and dynamic register redistribution are compiled and audited, not executed on local sm120.')
(H/'resources.json').write_text(json.dumps(out,indent=2)+'\n');print(rows)
