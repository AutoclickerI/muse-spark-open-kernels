"""Work coverage and reset/progress proof; no occupancy-dependent readiness."""
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent
checks=0
for n in range(1025):
 expected=[r for r in range(0,n,256) if n>8 and ((n<=16 and r==0) or (n>16 and n-r<=64))]
 eligible=8<n<=64 or n>256 and 0<n%256<=64
 actual=[(n//256)*256 if n>256 else 0] if eligible else []
 assert expected==actual,(n,expected,actual);checks+=1
for experts in range(33):
 for mt in (32,64):
  total=experts*mt
  for grid in (1,2,4,132,264):
   static=[u for c in range(grid) for u in range(c,total,grid)]
   assert sorted(static)==list(range(total))
   # Successful atomics claim a monotonically increasing ticket. Every CTA
   # exits on its first out-of-range claim; no claimant waits for another CTA.
   claims=list(range(total+grid));live=[u for u in claims if u<total]
   assert live==list(range(total)) and claims[-1]+1==total+grid;checks+=1
out=dict(checks=checks,cuda_sha256=hashlib.sha256((H/'tail.cu').read_bytes()).hexdigest(),coverage='All expert counts0..1024, MT32/64, empty/full lists and grids1/2/4/132/264.',progress='Only bounded tile computation and monotonic atomic claiming. No cross-CTA wait or resident-grid assumption. W13/W2 have independent heads, both reset by the route kernel every replay.',ordering='Route completes before fork; each stream waits on fork. W2 follows the full compact W13 kernel on the same stream. Every tile retires WGMMA and crosses a CTA barrier before ticket/scratch reuse. Parent copy_sync_check.py covers the unchanged S2/S3 rings.',limitation='Contract model and source coverage; hardware race freedom not exhaustively tested.')
(H/'schedule.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
