MECH_PAIRS=(('PREPACK','PACK'),('EARLY3','PACK3'),('DUPCTA','DUPSYNC'),('MC2','DUPSYNC'),('PREMC2','MC2'))
def mechanism_lab(deadline,weights,state=None):
 import json,math,struct,base64
 started=time.monotonic();mechanism_build();rows=[];peak=0.;worst=[0.,0.];ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
 for seed in range(16):
  if deadline-time.monotonic()<max(12,peak*1.2+1):break
  tick=time.monotonic();w=dc_weights(1047293+65537*seed,'cuda');st=act_prepare(w)
  for m in (128,256):
   ds=dc_fixtures(w,m);wants=dc_oracles(ds,w);gs={a:MechanismGraph(ds[0],st,a) for a in MECH_NAMES};records=[]
   for fi,(d,want) in enumerate(zip(ds,wants)):
    gs['BASE'].refresh(d)
    for a,g in gs.items():
     if a!='BASE':g.refresh(d,gs['BASE'])
    errs=[op_error(g.replay(),want) for g in gs.values()];aa0,aa1=op_pair(gs['BASE'],gs['BASE'],ev);rs=[1.];us=[aa0]
    for arm in MECH_NAMES[1:]:
     a,b=op_pair(gs['BASE'],gs[arm],ev);rs.append(b/a);us.append(b)
    contrasts=[]
    for new,old in MECH_PAIRS:
     a,b=op_pair(gs[old],gs[new],ev);contrasts.append(b/a)
    for i,g in enumerate(gs.values()):errs[i]=tuple(max(a,b) for a,b in zip(errs[i],op_error(g.replay(),want)))
    assert all(e<=.015 and r<=.075 for e,r in errs),(seed,m,fi,errs)
    worst=[max(worst[0],max(e for e,r in errs)),max(worst[1],max(r for e,r in errs))];records.append(dict(rs=rs,us=us,contrasts=contrasts,aa=[aa0,aa1]))
   rows.append(dict(m=m,seed=seed,rs=[statistics.mean(r['rs'][i] for r in records)for i in range(9)],us=[statistics.mean(r['us'][i]for r in records)for i in range(9)],cs=[statistics.mean(r['contrasts'][i]for r in records)for i in range(5)],aa=[statistics.mean(r['aa'][i]for r in records)for i in (0,1)]));del gs
  peak=max(peak,time.monotonic()-tick);del st,w;torch.cuda.empty_cache()
 assert len(rows)>=8
 lines=['META diagnostic only:route frozen outside timing for all arms (atomic slots);all share current-fixture metadata.PREPACK/PREMC2 gather outside timing, cold flush before every replay, never nominatable. BASE=S18 95df42fe; PACK=current gather. PACK3/EARLY3 same Q3/93184B footprint, X earlier by one KT128 only. DUPSYNC/DUPCTA same2CTA cluster grid,private unicast copies,cluster vs local sync. MC2/DUPSYNC isolates multicast with sync fixed; unsafe literal MC2-local excluded.']
 for m in (128,256):
  rr=[r for r in rows if r['m']==m]
  for i,a in enumerate(MECH_NAMES):
   xs=[r['rs'][i]for r in rr];lines.append(f'M{m} {a} n={len(xs)} ratio={statistics.mean(xs):.6f} SE={statistics.stdev(xs)/math.sqrt(len(xs)):.6f} us={statistics.mean(r["us"][i]for r in rr):.3f} TASK=PASS')
  for i,(new,old) in enumerate(MECH_PAIRS):
   xs=[r['cs'][i]for r in rr];lines.append(f'PAIR M{m} {new}/{old} ratio={statistics.mean(xs):.6f} SE={statistics.stdev(xs)/math.sqrt(len(xs)):.6f}')
  aa=[r['aa'][1]-r['aa'][0]for r in rr];lines.append(f'AA M{m} signed/abs/max_us={statistics.mean(aa):.3f}/{statistics.mean(abs(x)for x in aa):.3f}/{max(abs(x)for x in aa):.3f}')
 vals=[v for m in (128,256)for r in rows if r['m']==m for v in r['rs']+r['cs']+r['aa']+r['us']]
 lines.append('SEEDS '+json.dumps(dict(names=MECH_NAMES,pairs=MECH_PAIRS,n=len(rows)//2,format='fp32 little endian[M,seed,ratios9,contrasts5,AA_us2,us9]',data=base64.b85encode(struct.pack('<'+'f'*len(vals),*vals)).decode()),separators=(',',':')))
 lines.append('RES '+str({a:[list(probe_module().probe_info(i,f))for f in (0,1)]for a,i in [('PACK3',0),('EARLY3',1),('DUPSYNC',2),('DUPCTA',3)]}))
 lines.append(f'GATE L2/row={worst[0]:.6f}/{worst[1]:.6f} END seeds={len(rows)//2} seconds={time.monotonic()-started:.1f}');assert len('\n'.join(lines))<9000
 return lines
