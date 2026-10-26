def act_lab(deadline,weights,state=None):
 import base64,json,math,struct
 started=time.monotonic();act_build();rows=[];peak=0.;worst=[0.,0.];ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
 for seed in range(16):
  if deadline-time.monotonic()<max(12,peak*1.2+1):break
  tick=time.monotonic();w=dc_weights(29173+104729*seed,'cuda');st=act_prepare(w)
  for m in (128,256):
   ds=dc_fixtures(w,m);wants=dc_oracles(ds,w);graphs=[MWGraph(lambda d,a=a:act_call(d,st,a),ds[0]) for a in ACT_NAMES];records=[];counts=[]
   for d,want in zip(ds,wants):
    counts.append(act_counts(d))
    for g in graphs:g.refresh(d)
    errs=[op_error(g.replay(),want) for g in graphs];aa0,aa1=op_pair(graphs[0],graphs[0],ev);rs=[1.];us=[aa0]
    for g in graphs[1:]:
     a,b=op_pair(graphs[0],g,ev);rs.append(b/a);us.append(b)
    for j,g in enumerate(graphs):errs[j]=tuple(max(a,b) for a,b in zip(errs[j],op_error(g.replay(),want)))
    assert all(e<=.015 and r<=.075 for e,r in errs),(seed,m,errs)
    worst=[max(worst[0],max(e for e,r in errs)),max(worst[1],max(r for e,r in errs))];records.append(dict(rs=rs,us=us,aa=[aa0,aa1]))
   rows.append(dict(m=m,seed=seed,rs=[statistics.mean(x['rs'][j] for x in records) for j in range(7)],us=[statistics.mean(x['us'][j] for x in records) for j in range(7)],aa=[statistics.mean(x['aa'][j] for x in records) for j in (0,1)],activation_MB=[statistics.mean(c['activation_MB'][a] for c in counts) for a in ACT_NAMES],weight_MB=[statistics.mean(c['weight_MB'][a] for c in counts) for a in ACT_NAMES]));del graphs
  peak=max(peak,time.monotonic()-tick);del st,w;torch.cuda.empty_cache()
 assert len(rows)>=8,'Insufficient seeds'
 lines=['META S18 95df42fe CHUNK4/P7 shell retained. PACK timed expert-X copy; TMA1 unicast matched MC2/MC4 output-channel sharing; W4N32/W4N64 two output tiles per WG with shared X, original/smaller token block. Six independent arms, all seeds, A:A, fresh cold pairs. Logical activation bytes include gather reads+writes and padded multicast payload. No HBM counter claim.']
 for m in (128,256):
  rr=[r for r in rows if r['m']==m]
  for j,a in enumerate(ACT_NAMES):
   xs=[r['rs'][j] for r in rr];mean=statistics.mean(xs);se=statistics.stdev(xs)/math.sqrt(len(xs));lines.append(f'M{m} {a} n={len(xs)} ratio={mean:.6f} SE={se:.6f} us={statistics.mean(r["us"][j] for r in rr):.3f} X_MB={statistics.mean(r["activation_MB"][j] for r in rr):.3f} W_MB={statistics.mean(r["weight_MB"][j] for r in rr):.3f} TASK=PASS')
  aa=[r['aa'][1]-r['aa'][0] for r in rr];lines.append(f'AA M{m} signed/abs/max_us={statistics.mean(aa):.3f}/{statistics.mean(abs(x) for x in aa):.3f}/{max(abs(x) for x in aa):.3f}')
 values=[v for m in (128,256) for r in rows if r['m']==m for v in r['rs']+r['aa']+r['activation_MB']+r['weight_MB']]
 lines.append('SEEDS '+json.dumps(dict(names=ACT_NAMES,n=len(rows)//2,format='little endian fp32 [M,seed,ratio(7),AA_us(2),activation_MB(7),weight_MB(7)]',data=base64.b85encode(struct.pack('<'+'f'*len(values),*values)).decode()),separators=(',',':')))
 mod=ACT_NS['ARMS']['activation_module']();lines.append('RES '+str({a:[list(mod.ax_info(k,f)) for f in (0,1)] for a,k in [('PACK',0),('TMA1',1),('MC2',2),('MC4',4),('W4N32',5),('W4N64',6)]}))
 lines.append(f'GATE L2/row={worst[0]:.6f}/{worst[1]:.6f} END seeds={len(rows)//2} seconds={time.monotonic()-started:.1f}');assert len('\n'.join(lines))<9000
 return lines
