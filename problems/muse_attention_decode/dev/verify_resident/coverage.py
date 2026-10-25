"""Exact history/query ownership and lifetime-space bounds for each resident topology."""
import json
from pathlib import Path
rows=[]
for cs in (2,4,8,16):
 cap=2048//cs
 assert cap*128+1024>=64*68*4
 for seq in (0,1,10,127,128,129,257,2047,2048,2305,80000,300000):
  for window in (1024,2048):
   lo=max(0,seq-window+1);nc=seq-lo;chunk=((nc+cs-1)//cs+127)//128*128
   assert chunk<=cap
   owned=[list(range(min(seq,lo+s*chunk),min(seq,min(seq,lo+s*chunk)+chunk))) for s in range(cs)]
   assert [k for keys in owned for k in keys]==list(range(lo,seq))
   for tok in range(8):
    got=[k for keys in owned for k in keys if k>=seq+tok-window+1]+list(range(seq,seq+tok+1))
    assert got==list(range(max(0,seq+tok-window+1),seq+tok+1))
   assert sorted(r for rank in range(cs) for w in range(4) for j in range(w,64,4) if (r:=rank+j*cs)<64)==list(range(64))
   rows.append([cs,seq,window])
H=Path(__file__).resolve().parent;(H/'out/index_admission.json').write_text(json.dumps(dict(cases=len(rows),passed=True),indent=2)+'\n');print('PASS',len(rows),'history/query/lifetime cases')
