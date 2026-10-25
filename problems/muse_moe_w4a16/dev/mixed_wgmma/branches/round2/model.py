"""Traffic and anchored service estimates, separating padded reads from unique bytes."""
import json
import math
from pathlib import Path
H=Path(__file__).resolve().parent
# Exact fixed fixtures printed by H200 8ee003db. The phase service measurements
# are isolated, not whole-layer critical-path increments.
counts={128:[1,2,0,17,0,8,0,0,0,0,0,20,0,0,22,51,5,1,1,0,10,1,4,0,0,2,11,4,0,0,6,0],256:[0,1,0,40,2,18,0,0,0,0,0,27,0,0,54,95,5,5,1,0,26,0,8,0,0,2,31,13,0,1,12,1]}
observed={128:dict(control=185.43,heavy=[43.38,34.56],light=[80.86,49.17],direct=[65.97,41.41]),256:dict(control=261.03,heavy=[81.89,47.71],light=[70.16,44.19],direct=[109.90,75.79])}
out={}
for m,cs in counts.items():
 hot=[n for n in cs if n>16];o=observed[m];tiles32=sum(math.ceil(n/32) for n in hot);tiles64=sum(math.ceil(n/64) for n in hot);pad32=tiles32*32;pad64=tiles64*64
 # All Q plus recoded uint8 scales, no widening in prepare.
 uniqueW=len(hot)*(8192+4096)*4096*(.5+1/16)
 q=dict(heavy_counts=hot,unique_weight_MB=uniqueW/1e6,control_weight_ratio=tiles32/len(hot),new_weight_ratio=1.,
  control_X_MB=[pad32*4096*2*(8192//64)/1e6,pad32*4096*2*(4096//64)/1e6],arms={})
 for name,nc,pad,smem in (('TRANS64',64,pad64,16384),('TRANS128',128,pad64,32768),('CTA2',128,pad32,0),('CTA2S3',128,pad32,0)):
  x=[pad*4096*2*(8192//nc),pad*4096*2*(4096//nc)]
  pack_read_write=2*pad64*4096*2 if name.startswith('TRANS') else 0
  # W13->W2 AF stores replace ACT stores; no extra ACT->AF repack is hidden.
  # Reuse the measured DIRECT service rate for each phase, with its padded
  # same-expert weights/activations. This transport-only estimate deliberately
  # retains the observed instruction cost; new STS/SHFL costs remain unknown.
  directX=[pad32*4096*2*(8192//64),pad32*4096*2*(4096//64)]
  weights=[uniqueW*2/3,uniqueW/3]
  services=[o['direct'][p]*(weights[p]+x[p])/(weights[p]+directX[p]) for p in (0,1)]
  h=sum(services)+pack_read_write/4.58e6;l=sum(o['light']);baseH=sum(o['heavy'])
  # These overlap scenarios are sensitivity, not fitted predictions.
  fixed_overlap=baseH+l+16-o['control']
  same_overlap=h+l+16-fixed_overlap
  full_overlap=max(h,l)+16
  q['arms'][name]=dict(X_MB=[v/1e6 for v in x],X_ratio_vs_control=sum(x)/(sum(q['control_X_MB'])*1e6),timed_pack_MB=pack_read_write/1e6,decoded_B_smem=smem,
   estimated_heavy_service_us=h,estimated_whole_same_overlap_us=same_overlap,estimated_whole_full_overlap_us=full_overlap,
   extra_transport_cost='Unmeasured decoded-B STS/SHFL/barriers for TRANS; CTA2 occupancy and issue effects unmeasured. Padding and timed pack included.')
 q['target_us']={128:132.6,256:158.5}[m];q['observed']=o;out[m]=q
(H/'model.json').write_text(json.dumps(out,indent=2)+'\n')
for m,q in out.items():print(m,'W',q['control_weight_ratio'],'->1',[(n,[round(x,2) for x in a['X_MB']],round(a['estimated_whole_same_overlap_us'],1),round(a['estimated_whole_full_overlap_us'],1)) for n,a in q['arms'].items()])
