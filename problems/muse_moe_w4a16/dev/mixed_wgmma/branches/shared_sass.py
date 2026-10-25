"""Summarize WGMMA waits and innermost 4-stage SASS loops, no retired-count claim.

Usage: cuobjdump -sass cubin > /tmp/p3b_shared_final.sass
       python3 shared_sass.py /tmp/p3b_shared_final.sass ptxas.log
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
H=Path(__file__).resolve().parent
s=Path(sys.argv[1]).read_text();out={}
for b in s.split('Function : ')[1:]:
 name=b.splitlines()[0]
 if 'uq_queue' not in name and 'gemmILi32' not in name:continue
 ins=[(int(a,16),op) for a,op in re.findall(r'/\*([0-9a-f]+)\*/\s+(.+?);',b)]
 loops=[]
 for pc,op in ins:
  z=re.search(r'BRA (0x[0-9a-f]+)',op)
  if z and int(z[1],16)<pc:
   lo=int(z[1],16);body=[v for a,v in ins if lo<=a<=pc]
   if sum('HGMMA' in v for v in body)==32:
    tile=next(re.search(r'HGMMA.64x(\d+)x16',v)[1] for v in body if 'HGMMA' in v)
    loops.append(dict(start=lo,end=pc,NT=int(tile),instructions_per_KT128=len(body)/4,
                      HGMMA_per_stage=8,HMUL2_per_stage=sum('HMUL2' in v for v in body)/4,
                      LDGSTS_per_stage=sum('LDGSTS' in v for v in body)/4,
                      local_memory_per_loop=sum('LDL' in v or 'STL' in v for v in body)))
 # The outer persistent loop can contain the same 32 MMAs; retain inner spans.
 loops=[x for x in loops if not any(y['start']>=x['start'] and y['end']<x['end'] for y in loops)]
 out[name.split('EEEv')[0]]=dict(depbar=dict(collections.Counter(re.findall(r'WARPGROUP.DEPBAR.LE gsb0, (0x[0-9a-f]+)',b))),loops=loops)
log=Path(sys.argv[2]).read_text();resources={}
for b in log.split('Compiling entry function ')[1:]:
 z=re.search(r'uq_queueILi(\d+)',b.splitlines()[0])
 if z:
  frame=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',b)
  sm=re.search(r'(\d+) bytes smem',b)
  resources[z[1]]=dict(registers=int(re.search(r'Used (\d+) registers',b)[1]),static_smem=int(sm[1]) if sm else 0,
                       dynamic_smem=52223,stack_spill_store_spill_load=[int(v) for v in frame.groups()])
meta=dict(cuda_sha256=hashlib.sha256((H/'cu_shared.cu').read_bytes()).hexdigest(),sass_sha256=hashlib.sha256(s.encode()).hexdigest(),kernels=out,resources=resources,
          note='No retired instruction claim. 4-stage inner-loop counts /4. Three NT variants per queue; group wait1 retained. Compiled sm90a; actual occupancy comes from H200 receipt.')
(H/'shared_sass.json').write_text(json.dumps(meta,indent=2)+'\n');print(resources)
