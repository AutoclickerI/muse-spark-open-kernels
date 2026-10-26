"""Rank measured post-port cells; resource counterfactual, not a measured G0."""
import json,math,statistics,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'problems/muse_attention_decode'));import reference as ref
p=json.loads((H.parent/'public_ports/result.json').read_text());t=json.loads((R/'problems/muse_decode_layer/dev/s19_timeline/result/report.json').read_text());sh={s['case']:s for s in t['shapes']};rows=[];load_totals={}
for cell,samples in p['seed_ns'].items():
 b,l=map(int,cell[:-1].split('x'));w=cell[-1]=='s';name=('d' if l==1 else 'v')+str(b)+('S' if w else 'F');load=name[:-1];pick=p['selection'].get(cell,'A');weight=46 if w else 16
 times=[s[pick]/1000 for s in samples];targets=[];lengths=[]
 for seed in p['seeds']:
  g=torch.Generator(device='cuda').manual_seed(seed);lens=ref._lengths(g,b,80000,'agentic' if b>1 else 'fixed','cuda');lens=lens.cpu().tolist();lengths.append(lens)
  # Unique KV history plus appended causal block: a union of whole rows.
  n=sum(min(seq+l,2048+l-1) if w else seq+l for seq in lens);targets.append(1.3*n*256/4.8e6)
 old_wall=sh[name]['wall_us'];old_attn=sh[name]['rows']['attention']['critical_us'];picktime=statistics.mean(times);target=statistics.mean(targets);excess=statistics.mean(max(a-z,0) for a,z in zip(times,targets))
 wall=old_wall-old_attn+picktime;load_totals[load]=load_totals.get(load,0)+weight*wall
 rows.append(dict(cell=cell,name=name,load=load,pick=pick,weight=weight,time_us=picktime,target_us=target,excess_us=excess,weighted_us=weight*excess,weighted_time_us=weight*picktime,lengths=lengths,target_seed_us=targets))
for r in rows:r['R_if_removed']=(load_totals[r['load']]/(load_totals[r['load']]-r['weighted_us']))**(1/6)
rows.sort(key=lambda r:r['R_if_removed'],reverse=True)
out=dict(rows=rows,estimated_pick_load_us=load_totals,method='post-port component times26d5054f; uniqueKV actual-seed lengths at4.8TB/s x1.3; whole-layer non-attention from S19 CUPTI9c16c850. Replace component only; assumes savings transfer without dependency/overlap cost. Ranking scenario, not measurement or attainable promise.',receipts=['26d5054ffce94231','9c16c850f7b941a5'])
(H/'sizing.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r['name'],*(f'{r[k]:.4f}' for k in ('time_us','target_us','excess_us','weighted_us','R_if_removed')))
