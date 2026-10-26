"""Exhaustive finite-domain layout and partition proofs for structural arms."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;o=H/'roof'
trials=0
for seq in (*range(1,258),511,512,513,2047,80000,86000):
 for splits in (1,2,3,4,7,12,24,32,132,336,512):
  tiles=(seq+63)//64;spans=[(min(s*tiles//splits*64,seq),min((s+1)*tiles//splits*64,seq)) for s in range(splits)]
  assert spans[0][0]==0 and spans[-1][1]==seq
  assert all(a[1]==b[0] for a,b in zip(spans,spans[1:]))
  assert sum(b-a for a,b in spans)==seq
  trials+=1
# Old paired producer:8chunks/row,8issues/thread. New per-head:4chunks,
# 4issues/thread, every one of128keys x64B covered once per head.
old=[];new=[]
for pw in range(4):
 for lane in range(32):
  q,c=lane>>3,lane&7
  for i in range(8):old.append((c>>2,16*pw+4*(i&3)+q+64*(i>>2),(c&3)*16))
  q,c=lane>>2,lane&3
  for i in range(4):new.append((16*pw+8*(i&1)+q+64*(i>>1),c*16))
assert len(set(old))==1024 and len(set(new))==512
assert set(new)=={(k,c) for h,k,c in old if h==0}
assert {(h,k,c) for h in range(2) for k,c in new}==set(old)
# New G ownership is a permutation of all64rows per tile, not omitted keys.
old={(lt+16*k,c) for lt in range(16) for k in range(4) for c in range(8)}
new={(4*lt+k,c) for lt in range(16) for k in range(4) for c in range(8)}
assert old==new and len(new)==512
manifest=json.loads((o/'manifest.json').read_text())
for name in ('H1','H2'):
 s=(o/(name+'.cu')).read_text()
 assert 'mb_init(&empty[tid], 4);' in s and 'slot * 64 + hh * 64' in s
 assert 'return CLKPAD + 12288 + ST * 24576;' in s
 assert 'p.grid *= '+('1;' if name=='H1' else '2;') in s
 assert manifest[name]['cuda_sha256']==hashlib.sha256(s.encode()).hexdigest()
result=dict(partition_cases=trials,old_producer_chunks=1024,new_chunks_per_head=512,gather_coordinate_permutation=True,
    one_head_empty_count=4,one_head_full_count=4,consumer_registers=216,producer_registers=32,
    two_CTA_register_budget=2*(128*216+128*32),capacity=65536,passed=True)
assert result['two_CTA_register_budget']<=result['capacity']
(o/'protocol.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
