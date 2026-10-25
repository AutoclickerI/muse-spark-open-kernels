"""Exact index coverage for history splits and all query/token subgroup partitions."""
import json
from pathlib import Path
rows=[]
for length in (1,8):
 for qt in (1,2,4,8):
  heads=8//qt
  q=[(sub,token,head,token*heads+head) for sub in range(qt) for token in range(length) for head in range(heads)]
  assert len({(tok,sub*heads+h) for sub,tok,h,_ in q})==length*8
  for seq in (0,1,10,127,128,257,2047,2048,2305,80000):
   lo=max(0,seq-2048+1);nc=seq-lo
   for ns in (1,2,4,8,16):
    chunk=((nc+ns-1)//ns+127)//128*128
    keys=[k for split in range(ns) for k in range(min(seq,lo+split*chunk),min(seq,min(seq,lo+split*chunk)+chunk))]
    assert keys==list(range(lo,seq))
    for tok in range(length):
     want=list(range(max(0,seq+tok-2048+1),seq+tok+1))
     got=[k for k in keys if k>=seq+tok-2048+1]+list(range(seq,seq+tok+1))
     assert got==want
   rows.append(dict(L=length,qt=qt,seq=seq,queries=length*8))
p=Path(__file__).resolve().parent/'out/index_admission.json';p.write_text(json.dumps(dict(cases=len(rows),passed=True),indent=2)+'\n');print('PASS',len(rows),'query/history coverage cases')
