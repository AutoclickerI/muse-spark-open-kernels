REVIEW_NAMES=('S15','CHUNK3','CHUNK4','PACK5','READY_SERIAL','READY')
REVIEW_NS={}
def review_build():
 import linecache
 if REVIEW_NS:return
 mod=nf_module();NF_OPTS['pmz']=mod.pmz_init()
 REVIEW_NS['S15']=globals()
 for arm,cell in REVIEW_CELLS.items():
  ns=dict(globals());filename='<P3b-'+arm+'>';linecache.cache[filename]=(len(cell),None,cell.splitlines(True),filename)
  exec(compile(cell,filename,'exec'),ns);ns['p3b_module']();REVIEW_NS[arm]=ns

def review_prepare(w):
 w=dict(w,post_norm_w=w['post_expert_norm'])
 base=nf_perm(nf_wide(m128_hh_issue_prepare(nf_ilv(w))));native=m128_hh_issue_nt_prepare(w)
 states={}
 for arm in REVIEW_NAMES:
  state=dict(s5_hh_base=base,moe_bank=native,nf_streams=(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()))
  if arm!='S15':REVIEW_NS[arm]['p3b_bind'](state)
  states[arm]=state
 return states

def review_plan(data,states,arm):
 state=states[arm]
 if arm=='S15':return nf_plan(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'])
 ns=REVIEW_NS[arm]
 args=(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'])
 return ns['p3b_plan'](*args,*([state['p3b_packets']] if arm=='PACK5' else []))

def review_call(data,states,arm):
 p=review_plan(data,states,arm)
 for fn in p['steps']:out=fn()
 return out

def review_lab(deadline,weights,state=None):
 import json,math,struct,base64
 started=time.monotonic();review_build();rows=[];peak=0
 ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));worst=[0.,0.]
 for seed in range(24):
  if deadline-time.monotonic()<max(12,peak*1.2+1):break
  tick=time.monotonic();w=dc_weights(7717+7919*seed,'cuda');states=review_prepare(w)
  for m in (128,256):
   ds=dc_fixtures(w,m);wants=dc_oracles(ds,w)
   graphs=[MWGraph(lambda d,a=a:review_call(d,states,a),ds[0]) for a in REVIEW_NAMES]
   records=[]
   for fi,(d,want) in enumerate(zip(ds,wants)):
    for g in graphs:g.refresh(d)
    errs=[op_error(g.replay(),want) for g in graphs]
    aa0,aa1=op_pair(graphs[0],graphs[0],ev);rs=[1.];times=[aa0]
    for g in graphs[1:]:
     a,b=op_pair(graphs[0],g,ev);rs.append(b/a);times.append(b)
    for i,g in enumerate(graphs):errs[i]=tuple(max(a,b) for a,b in zip(errs[i],op_error(g.replay(),want)))
    assert all(e<=.015 and r<=.075 for e,r in errs),(seed,m,fi,errs)
    worst=[max(worst[0],max(e for e,r in errs)),max(worst[1],max(r for e,r in errs))]
    records.append(dict(rs=rs,us=times,aa=[aa0,aa1]))
   # Independent weight seeds are the replication unit, not the three correlated fixtures.
   rows.append(dict(m=m,seed=seed,ratio=[round(statistics.mean(r['rs'][a] for r in records),6) for a in range(len(REVIEW_NAMES))],us=[statistics.mean(r['us'][a] for r in records) for a in range(len(REVIEW_NAMES))],aa=[round(statistics.mean(r['aa'][a] for r in records),3) for a in (0,1)]))
   del graphs
  peak=max(peak,time.monotonic()-tick);del states,w;torch.cuda.empty_cache()
 lines=['META S15 NF 0a31e251 controls; independent heavy PACK5, token-block readiness plus matched W2-grid sequential control, R6 CHUNK3/4 rebase. All point improvements go to P4; A:A diagnostic; held-out whole-layer six-load R confirms.']
 for m in (128,256):
  rr=[r for r in rows if r['m']==m]
  for a,name in enumerate(REVIEW_NAMES):
   rs=[r['ratio'][a] for r in rr];mean=statistics.mean(rs);se=statistics.stdev(rs)/math.sqrt(len(rs)) if len(rs)>1 else float('nan')
   lines.append(f'M{m} {name} seeds={len(rs)} fixtures={3*len(rs)} ratio={mean:.6f} SE={se:.6f} us={statistics.mean(r["us"][a] for r in rr):.3f} TASK=PASS')
  aa=[r['aa'][1]-r['aa'][0] for r in rr]
  lines.append(f'AA M{m} signed/abs/max={statistics.mean(aa):.3f}/{statistics.mean(abs(x) for x in aa):.3f}/{max(abs(x) for x in aa):.3f}')
 # Fixed-width, uncompressed binary transport: its length cannot grow with H200 timing entropy.
 values=[v for m in (128,256) for r in rows if r['m']==m for v in r['ratio']]
 values += [v for m in (128,256) for r in rows if r['m']==m for v in r['aa']]
 archive=dict(names=REVIEW_NAMES,seeds=len(rows)//2,format='little-endian float64: ratios[2,seeds,arms], then AA_us[2,seeds,2]',data=base64.b85encode(struct.pack('<'+'d'*len(values),*values)).decode())
 lines.append('SEEDS '+json.dumps(archive,separators=(',',':')))
 lines.append('RES '+str({a:([list(REVIEW_NS[a]['p3b_module']().ch_info(3 if a=='CHUNK3' else 4,f)) for f in (0,1)] if a.startswith('CHUNK') else [list(REVIEW_NS[a]['p3b_module']().ready_info(f)) for f in (0,1)] if a=='READY' else [list((REVIEW_NS[a]['p3b_module']() if a!='S15' else nf_module()).br_info(k)) for k in (26,21)]) for a in REVIEW_NAMES}))
 lines.append(f'GATE L2/row={worst[0]:.6f}/{worst[1]:.6f} END seeds={len(rows)//2} seconds={time.monotonic()-started:.1f}')
 assert sum(map(len,lines))+len(lines)<9000,'Bounded per-seed archive exceeded receipt allocation'
 return lines
