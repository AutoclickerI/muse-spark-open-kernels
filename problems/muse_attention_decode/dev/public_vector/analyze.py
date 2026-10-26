"""Whole-load transfer, single-cell leave-outs and composed six-load G0 sizing."""
import argparse,base64,hashlib,json,lzma,math,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets
from promotion import interval
exec((H.parent/'public_hybrid/codec.py.inc').read_text())
ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args()
receipt=json.loads(a.receipt.read_text());records=read_packets(receipt['result'].get('log_tail',''))
record=next(r for r in records if r['name']=='in_graph');raw=record['packet']
raw=decode_rounds(raw);meta=json.loads((H/'lab_manifest.json').read_text())
assert raw['native'] and raw['complete'] and not raw['failures'],raw.get('failures')
assert raw['identities']==meta['identities'] and raw['seeds']==meta['seeds']
assert record['source_sha256']==meta['payload_sha256'][:16]
means={};control={};points=[]
for cell,rows in raw['round_ns'].items():
 assert len(rows)==len(raw['seeds'])
 names=list(rows[0]);assert all(set(r)==set(names) and all(len(v)==meta['rounds'] and min(v)>0 for v in r.values()) for r in rows)
 assert raw['checks'][cell]=={n:len(rows) for n in names}
 means[cell]=[{n:statistics.mean(v) for n,v in row.items()} for row in rows]
control={};selection={};percell=[];loads={};comb={n:[] for n in ('S','D','V','F')};aa_all=[]
for b,l,w in meta['cases']:
 cell=('d' if l==1 else 'v')+str(b)+('S' if w else 'F');other=cell[:-1]+('F' if w else 'S');weight,ow=(46,16) if w else (16,46)
 rows=means[cell];others=means[other];aa=[(weight*r['A2']+ow*o['A2'])/(weight*r['A']+ow*o['A']) for r,o in zip(rows,others)]
 control[cell]=interval(aa)
 for arm in rows[0]:
  if arm in ('A','A2'):continue
  # Exactly one cell changed. The other cell's A2 is the unchanged twin,
  # measured on the same seed; no component timing enters this estimate.
  ratios=[(weight*r[arm]+ow*o['A2'])/(weight*r['A']+ow*o['A']) for r,o in zip(rows,others)]
  point=dict(cell=cell,arm=arm,new_over_pick=interval(ratios,aa),seed_ratios=ratios)
  if arm=='S':point['port_over_S19']=interval([1/x for x in ratios],aa)
  percell.append(point)
 best=min([x for x in percell if x['cell']==cell],key=lambda x:x['new_over_pick']['R'],default=None)
 if best and best['new_over_pick']['R']<1:selection[cell]=best['arm']
for l in (1,8):
 for b in (1,16,32):
  s=('d' if l==1 else 'v')+str(b)+'S';f=s[:-1]+'F'
  loads[s[:-1]]={};aa=[(46*x['A2']+16*y['A2'])/(46*x['A']+16*y['A']) for x,y in zip(means[s],means[f])]
  aa_all.append(aa)
  for arm in comb:
   ratios=[]
   for x,y in zip(means[s],means[f]):
    # Native d1 has V only, identical to the combined fix for these cells.
    ns=arm if arm in x else 'V' if arm=='F' and 'V' in x else 'A2'
    nf=arm if arm in y else 'V' if arm=='F' and 'V' in y else 'A2'
    ratios.append((46*x[ns]+16*y[nf])/(46*x['A']+16*y['A']))
   loads[s[:-1]][arm]=interval(ratios,aa);comb[arm].append(ratios)
geo=lambda pairs:[math.exp(statistics.mean(math.log(p[i]) for p in pairs)) for i in range(len(raw['seeds']))]
aa_geo=geo(aa_all);six={n:interval([1/x for x in geo(v)],aa_geo) for n,v in comb.items()}
out=dict(job=receipt['id'],measured=True,controls=control,points=percell,loads=loads,six_load_old_over_new=six,selection=selection,seeds=raw['seeds'],TASK=raw['checks'],gate='Candidate points come only from production kernel_fn whole loads. P4 confirms a selected composition with disjoint held-out G0 seeds.',profiles=raw['profiles'])
(H/'result.json').write_text(json.dumps(out,indent=2)+'\n');(H/'receipt_data.json').write_text(json.dumps(raw,indent=2)+'\n');(H/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
for p in percell:print(p['cell'],p['arm'],p['new_over_pick']['R'],p['new_over_pick']['CI95'])
print('COMPOSED',six)
