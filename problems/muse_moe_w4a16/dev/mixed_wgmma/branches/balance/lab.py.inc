def balance_lab(deadline,weights,state=None):
 import json,math,struct,base64
 started=time.monotonic();balance_build();oracle_module();rows=[];peak=0.;worst=[0.,0.];clocks=[];traffic=[]
 ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
 oracle_names=('U','W32','X32','B32','W64','X64','B64','R15','R4')
 for seed in range(16):
  if deadline-time.monotonic()<max(12,peak*1.15+1):break
  tick=time.monotonic();w=dc_weights(18497+104729*seed,'cuda');st=balance_prepare(w)
  for m in (128,256):
   ds=dc_fixtures(w,m);wants=dc_oracles(ds,w)
   graphs=[MWGraph(lambda d,a=a:balance_call(d,st,a),ds[0]) for a in BAL_NAMES]
   records=[]
   for fi,(d,want) in enumerate(zip(ds,wants)):
    for g in graphs:g.refresh(d)
    errs=[op_error(g.replay(),want) for g in graphs]
    aa0,aa1=op_pair(graphs[0],graphs[0],ev);ratios=[1.];times=[aa0]
    for g in graphs[1:]:
     a,b=op_pair(graphs[0],g,ev);ratios.append(b/a);times.append(b)
    ratios4=[]
    for g in graphs[2:]:
     a,b=op_pair(graphs[1],g,ev);ratios4.append(b/a)
    for j,g in enumerate(graphs):errs[j]=tuple(max(a,b) for a,b in zip(errs[j],op_error(g.replay(),want)))
    assert all(e<=.015 and r<=.075 for e,r in errs),(seed,m,fi,errs)
    worst=[max(worst[0],max(e for e,r in errs)),max(worst[1],max(r for e,r in errs))]
    records.append(dict(rs=ratios,rs4=ratios4,us=times,aa=[aa0,aa1]))
   for g in graphs:g.refresh(ds[0])
   og=[oracle_graph(ds[0],st,*a) for a in ((0,32,False),(1,32,False),(2,32,False),(2,32,True),(1,64,False),(2,64,False),(2,64,True))]
   ous=[]
   for g in og:
    a,b=op_pair(graphs[0],g,ev);ous.append(b)
   # Same first fixture and cache protocol, with full math controls remeasured.
   a,b=op_pair(graphs[0],graphs[1],ev);ous.extend([a,b])
   if seed<4:
    for arm,control in (('S15',graphs[0]),('CH4',graphs[1])):
     cg=clock_plan(ds[0],st,arm);obs=[];marks=[]
     for _ in range(3):
      a,b=op_pair(control,cg,ev);obs.append(b/a);lab_cold_l2();cg.replay();torch.cuda.synchronize()
      marks.append([cg.marks['origin'].elapsed_time(cg.marks[k])*1000 for k in ('h13','h2','n13','n2','l13','l2','joined','end')])
     clocks.append(dict(m=m,arm=arm,observer=statistics.mean(obs),us=[statistics.mean(x[j] for x in marks) for j in range(8)]))
   if seed==0:
    counts=torch.stack([((ds[0]['topk_ids']-ds[0]['config'].get('local_start',0))==e).sum() for e in range(32)]).cpu().tolist()
    traffic.append(dict(m=m,counts=counts,unique_MB=sum(n>0 for n in counts)*28.311552,requests32=og[2].work,requests64=og[5].work))
   rows.append(dict(m=m,seed=seed,ratio=[statistics.mean(r['rs'][a] for r in records) for a in range(len(BAL_NAMES))],ratio4=[statistics.mean(r['rs4'][a] for r in records) for a in range(5)],us=[statistics.mean(r['us'][a] for r in records) for a in range(len(BAL_NAMES))],aa=[statistics.mean(r['aa'][a] for r in records) for a in (0,1)],oracle=ous,byte_MB=[og[0].work['weight_MB'],og[2].work['weight_MB'],og[2].work['valid_X_MB'],og[5].work['weight_MB'],og[5].work['valid_X_MB']]))
   del graphs,og
  peak=max(peak,time.monotonic()-tick);del st,w;torch.cuda.empty_cache()
 assert len(rows)>=8,'Insufficient independent seeds'
 lines=['META S1560254883/CHUNK4 controls; B8/4/2/0 transfers low-count experts from NT1 to N16, ONE N32/64 all-expert path. All point winners to P4. Oracle U/W/X/B excludes route+finish, R includes them; ordered interventions are not independent additive stall counters.']
 for m in (128,256):
  rr=[r for r in rows if r['m']==m]
  for a,name in enumerate(BAL_NAMES):
   rs=[r['ratio'][a] for r in rr];mean=statistics.mean(rs);se=statistics.stdev(rs)/math.sqrt(len(rs))
   extra=''
   if a>=2:
    s4=[r['ratio4'][a-2] for r in rr];extra=f' vsCH4={statistics.mean(s4):.6f} SE4={statistics.stdev(s4)/math.sqrt(len(s4)):.6f}'
   lines.append(f'M{m} {name} n={len(rs)} ratio={mean:.6f} SE={se:.6f}{extra} us={statistics.mean(r["us"][a] for r in rr):.3f} TASK=PASS')
  aa=[r['aa'][1]-r['aa'][0] for r in rr]
  lines.append(f'AA M{m} signed/abs/max_us={statistics.mean(aa):.3f}/{statistics.mean(abs(x) for x in aa):.3f}/{max(abs(x) for x in aa):.3f}')
  lines.append('ORACLE M'+str(m)+' '+json.dumps({n:round(statistics.mean(r['oracle'][i] for r in rr),3) for i,n in enumerate(oracle_names)},separators=(',',':')))
 for m in (128,256):
  for arm in ('S15','CH4'):
   cc=[r for r in clocks if r['m']==m and r['arm']==arm]
   lines.append(f'CLOCK M{m} {arm} n={len(cc)} observer_ratio={statistics.mean(r["observer"] for r in cc):.5f} ends_h13_h2_nt13_nt2_n1613_n162_join_finish_us='+str([round(statistics.mean(r['us'][j] for r in cc),2) for j in range(8)]))
 lines.append('BYTES '+json.dumps(traffic,separators=(',',':')))
 # Retain independent-seed samples for paired differences and CIs. Binary fp32
 # is transport only, not compute precision. Bound unaffected by timing entropy.
 values=[v for m in (128,256) for r in rows if r['m']==m for v in r['ratio']+r['ratio4']+r['aa']+r['oracle']+r['byte_MB']]
 lines.append('SEEDS '+json.dumps(dict(names=BAL_NAMES,oracles=oracle_names,n=len(rows)//2,format='little-endian float32 [M,seed,ratio_S15(7),ratio_CH4_Barms(5),AA_us(2),oracle_us(9),MB(uniqueW,W32,X32,W64,X64)]',data=base64.b85encode(struct.pack('<'+'f'*len(values),*values)).decode()),separators=(',',':')))
 lines.append('RES '+str(dict(S15=[list(nf_module().br_info(k)) for k in (26,21)],CH4=[list(BAL_NS['CH4']['p3b_module']().ch_info(4,f)) for f in (0,1)],BAL=[list(BAL_NS['BAL']['balance_module']().ch_info(4,f)) for f in (0,1)])))
 lines.append(f'GATE maxL2/row={worst[0]:.6f}/{worst[1]:.6f} END seeds={len(rows)//2} seconds={time.monotonic()-started:.1f}')
 assert sum(map(len,lines))+len(lines)<9000,('Receipt overflow',sum(map(len,lines)))
 return lines
