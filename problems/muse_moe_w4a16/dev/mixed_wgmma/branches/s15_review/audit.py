"""Native static K-loop counts, resources, and source-bound readiness contracts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
H=Path(__file__).resolve().parent
out={}
for arm in ('chunk4','pack5','ready'):
 cu=(H/(arm+'.cu')).read_text()
 with tempfile.TemporaryDirectory(prefix='p3b-s15-audit-') as work:
  cubin=Path(work)/'kernel.cubin'
  built=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_90a','--cubin',str(H/(arm+'.cu')),'-o',str(cubin),'-Xptxas=-v'],capture_output=True,text=True,check=True)
  log=built.stdout+built.stderr
  sass=subprocess.check_output(['cuobjdump','-sass',str(cubin)],text=True)
 assert not re.search(r'error|C751[239]|C7520',log)
 resources=[];functions=[]
 for f in log.split('Compiling entry function ')[1:]:
  name=f.splitlines()[0]
  if 'gemmILi32' not in name and 'chunk6kernel' not in name:continue
  frame=list(map(int,re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',f).groups()))
  regs=int(re.search(r'Used (\d+) registers',f)[1]);resources.append(dict(name=name,registers=regs,stack_spills=frame))
  if 'gemmILi32' in name:assert frame==[0,0,0]
 for f in sass.split('Function : ')[1:]:
  name=f.splitlines()[0].strip()
  if 'gemmILi32' not in name:continue
  inst={int(pc,16):t.strip() for pc,t in re.findall(r'/\*([0-9a-f]+)\*/\s+([^\n]*?);',f)}
  loops=[]
  for pc,t in inst.items():
   z=re.search(r'\bBRA(?:\.[A-Z]+)* (0x[0-9a-f]+)',t)
   if z and int(z[1],16)<pc:
    a=int(z[1],16)
    if any('HGMMA' in text for pp,text in inst.items() if a<=pp<=pc):loops.append((a,pc))
  # Innermost math-bearing backedge. PACK5 includes both compressed/raw paths statically.
  inner=[(a,b) for a,b in loops if not any(a<=x and y<=b and (a,b)!=(x,y) for x,y in loops)]
  assert len(inner)==1,(arm,name,inner)
  a,b=inner[0];ops=Counter(re.sub(r'^@!?[UP]\d+\s+','',t).split()[0] for pc,t in inst.items() if a<=pc<=b)
  mma=sum(n for op,n in ops.items() if op.startswith('HGMMA'));stages=mma/8
  assert stages==4,(name,mma)
  integer=sum(n for op,n in ops.items() if op.startswith(('IADD','IMAD','UIADD','UIMAD','LEA','SHF','USHF','LOP','ULOP','PRMT','BFE','BFI','ISETP','UISETP','POPC')))
  functions.append(dict(name=name,backedge=[a,b],KT128_stages=stages,static_instructions_per_stage=sum(ops.values())/stages,static_integer_per_stage=integer/stages,HMUL2_per_stage=sum(n for op,n in ops.items() if op.startswith('HMUL2'))/stages,instructions=dict(ops),limitation='PACK5 totals include both raw and compressed forward branches, not measured executed path counts. No issue-port saturation or stall clocks are inferred.'))
 out[arm]=dict(cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),resources=resources,K_loops=functions)
# A publication is per expert/32-token block, not per W13 segment or expert.
ready=(H/'ready.cu').read_text()
assert '__threadfence();__syncthreads();if(t==0)atomicAdd(READY_COUNT+e*RB+row/32,1)' in ready
assert 'ld.global.acquire.gpu.u32' in ready and 'done<128' in ready
out['readiness_contract']=dict(producer_tiles_per_block=128,stores_before_device_fence=True,all_writers_join_before_atomic=True,consumer_acquire=True,reset='timed graph node before route fork; next replay waits for both streams',progress='At most one W2 CTA per H200 SM globally; one producer+consumer resource sum fits each H200 SM. Blocking every SM would require at least2 W2 CTAs perSM, exceeding the grid. Local stand-in halves the grid when one pair cannot fit.')
(H/'native_audit.json').write_text(json.dumps(out,indent=2)+'\n')
for arm,rows in out.items():
 if isinstance(rows,dict) and 'K_loops' in rows:
  print(arm,[(r['name'][:58],r['static_instructions_per_stage'],r['static_integer_per_stage'],r['HMUL2_per_stage']) for r in rows['K_loops']])
