"""Source-coupled least-completion witness for both-consumer shared-X reuse."""
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parent
old=(B/'round2/cta2s3.cu').read_text();new=(H/'repair.cu').read_text()
missing='if constexpr(S>1)if(sub==0){if(k+S-1<32)'
joined='if constexpr(S>1)if(sub==0){__syncthreads();if(k+S-1<32)'
assert old.count(missing)==1 and new.count(missing)==0 and new.count(joined)==3
assert new.count('if constexpr(S==1){__syncthreads();if(k+1<32)')==3
assert new.count('if constexpr(NT==16)mwbr::mma<16>')==3
witness=[];checks=0
for stages in (1,2,3,4):
 for k in range(1,32):
  # Both consumers have committed the old reader before stage-entry CTA sync.
  # Only WG0 has retired it when WG0 returns from its current wait1 (or wait0
  # for S1). PTX orders each WG's queue; it does not order WG1 completion by WG0.
  old_reader=k if stages==1 else k-1
  producer_slot=(k+stages-1)%stages
  assert producer_slot==old_reader%stages
  done=[old_reader,old_reader-1]
  assert done[0]>=old_reader and done[1]<old_reader
  if k==1:witness.append(dict(stages=stages,copy_slot=producer_slot,last_reader=old_reader,retired=done[:],unsafe_without_join=True))
  # Every WG reaches the new barrier after its own retirement wait. Therefore
  # the writer can cross only once BOTH consumer completion frontiers pass.
  done[1]=old_reader;assert min(done)>=old_reader;checks+=3
for n in range(17,1025):
 fixed=sum(32 if min(n-r,64)<=32 else 64 for r in range(0,n,64))
 assert fixed==((n+31)//32)*32  # Same padded math rows as S14, unlike round5.
 checks+=1
out=dict(checks=checks,old_source_sha256=hashlib.sha256(old.encode()).hexdigest(),new_source_sha256=hashlib.sha256(new.encode()).hexdigest(),witnesses=witness,source_occurrences=dict(original_missing_join=1,repaired_multi_slot_joins=3,repaired_single_slot_joins=3),meaning='Original CTA2/tail raw timing is unqualified despite passing numerical gates. TRANS decoded-B path already has the relevant all-WG frontier barrier. Round4 uses a two-arrival empty barrier. Round5 X slices are private per WG and its shared decoded-A slot retires before overwrite.',limitation='PTX least-completion counterexample, not an assertion that the observed H200 output was corrupted. No unproven family is closed by the raw timing.')
(H/'protocol.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
