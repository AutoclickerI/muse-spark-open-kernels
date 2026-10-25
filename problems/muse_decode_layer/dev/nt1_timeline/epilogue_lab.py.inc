EPI_ARMS={'EPBASE':'epib','VEC':'epiv','PROD':'epip','NORM':'epin','PAIR':'epit'}
ONCE_ARMS={'ONCE':0,'ONCE_VEC':2,'ONCE_PAIR':2}

def epilogue_routed(pr,rs,st,arm,trace=False,stamps=None):
 x,ids,tw,lids=route_call(pr,rs);m=x.shape[0];d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 f=95 if m==1 else 69
 if arm=='S14':out=smallm_experts(d,st,f)
 elif arm in EPI_ARMS:out=globals()[EPI_ARMS[arm]+'_experts'](d,st,f|256 if trace else f,stamps=stamps)
 else:out=(reuseh_experts if arm=='ONCE_PAIR' else reusee_experts)(d,st,ONCE_ARMS[arm],trace,stamps)
 return out,x,ids,tw,lids

def epi_nodes(g,arm):
 act=activity_module();act.profile_begin()
 try:
  for _ in range(3):lab_cold_l2();g.replay();torch.cuda.synchronize()
 finally:records=act.profile_end()
 name='moe_reuse' if arm in ONCE_ARMS else 'moe_var';rows=[v for k,v in records if name in k and v[4]>0][-3:]
 if len(rows)!=3:raise ValueError('CUPTI missing consumer '+arm)
 return statistics.median((v[1]-v[0])/1000 for v in rows)

def norm_atomic_model(ns,grid):
 import bisect
 nb=[(n+7)//8 for n in ns];us=[0]
 for b in nb:us.append(us[-1]+b*256)
 total=us[-1];cc=min(256,max(1,(total+grid//2)//grid));runs=0;atoms=0
 for c in range(grid):
  last=None
  for u in range(total*c//grid,total*(c+1)//grid):
   e=bisect.bisect_right(us,u)-1;rel=u-us[e];full=256//cc;span=cc*nb[e]
   if rel<full*span:block=(rel%span)//cc
   else:L=256-full*cc;block=(rel-full*span)//L
   if (e,block)!=last:runs+=1;atoms+=min(8,ns[e]-block*8);last=(e,block)
 return total,runs,256*sum(ns),atoms

def epilogue_trace(g,off,st,m,arm,pr,rs,w,truth):
 if arm in EPI_ARMS:
  p=EPI_ARMS[arm];grid=min(264,st['sms']*globals()[p+'_module']().epi_info(325,2,m)[0]);stamps=torch.zeros((grid,64),device='cuda',dtype=torch.int64)
 else:
  grid=(reuseh_geometry if arm=='ONCE_PAIR' else reusee_geometry)(st,m,ONCE_ARMS[arm],True)[0];stamps=torch.zeros((grid,1040),device='cuda',dtype=torch.int64)
 traced=TimelineGraph(lambda:epilogue_routed(pr,rs,st,arm,True,stamps));reuse_gate(traced.replay(),w,truth,pr,rs)
 a,b,r=timeline_timing(off,traced);traced.replay();torch.cuda.synchronize();rows=stamps.cpu().tolist()
 if arm in EPI_ARMS:
  live=[v for v in rows if v[8]>v[0]>0];crit=max(live,key=lambda v:v[8]);
  phase=[(crit[1]-crit[0])/1000,(crit[2]-crit[1])/1000,(crit[3]-crit[2])/1000,(crit[5]-crit[3])/1000,(crit[7]-crit[5])/1000,(crit[8]-crit[7])/1000]
  subs=[crit[j]/1000 for j in (52,53,54,55,56,57,60)]
  ghz=statistics.median((v[21]-v[20])/(v[9]-v[0]) for v in live if v[9]>v[0] and v[21]>v[20])
  return r,b-a,phase,subs,ghz
 phase,sums,_,_=phase_summary(stamps,arm,0)
 return r,b-a,phase,sums,0.

def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();deadline-=5;dev=torch.device('cuda')
 lines=['META P3a epilogue issue/latency test on exact S14 f1c77ce6 small-M flags95/69. EPBASE is scalar reduction with diagnostic hooks disabled; VEC preserves accumulator layout and gathers with8 LDS128 on warp0; PROD additionally moves TMA issue to warp7; NORM additionally aggregates norm partials in registers per expert/block run; PAIR instead stores two contiguous float2 planes and uses two writer warps, retaining independent epilogue warps at the same modeled32 shared wavefronts. Same packed bytes,dequant,MMA,Ksplit,2 raw stages; no widened prepare bank.',
 'META S14 and all arms paired AB/BA x6 read flush,16 synthetic weight/router seeds,M1/8/16/32; ONCE/ONCE_VEC/ONCE_PAIR onlyM8/16/32, identical decode-once math with scalar vs one/two-vector-writer/warp7 epilogues. Native TASK gate before/after,independent A:A,separate3sample CUPTI. M32 trace8seeds: store+barrier,TMA call,warp0 body,final norm flush. Producer overlaps body in PROD/NORM so those subintervals are not additive; observer/off measured. Clock64/globaltimer reports effective GHz.']
 modules=(smallm_module,epib_module,epiv_module,epip_module,epin_module,epit_module,reusee_module,reuseh_module,activity_module)
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in [pool.submit(fn) for fn in modules]:f.result()
 lines.append(f'BUILD seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
 times={};ratios={};deltas={};nodes={};bws={};aa={m:[] for m in (1,8,16,32)};trace={};worst={};fails={};atomic=[];done=0;peak=0.
 oldtf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
 try:
  for seed in range(16):
   if deadline-time.monotonic()<max(8,peak*1.2+2):lines.append(f'BUDGET before_seed={seed}');break
   start=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=dev);w['post_norm_w']=w['post_expert_norm'];base=nt1_banks(w)
   st={k:base[k] for k in ('pnw','gs13','gs2','sms')};st['banks']={'lop3':base['bank']};smallm_bind(st,{1:95,8:69,16:69,32:69})
   for prefix in EPI_ARMS.values():globals()[prefix+'_bind'](st,{1:95,8:69,16:325})
   reusee_bind(st);reuseh_bind(st)
   if seed==0:
    for arm,p in EPI_ARMS.items():lines.append(f'GEOM {arm} off/trace resident,regs,local,dynamic,static='+str([globals()[p+'_module']().epi_info(f,2,32) for f in (69,325)]))
    for mode in (0,2):lines.append(f'GEOM ONCE mode{mode} off/trace grid,regs,local,dynamic,resident='+str([reusee_geometry(st,32,mode,t) for t in (False,True)]))
   for m in (1,8,16,32):
    pr,rs=routed_fixture(w,m,9000+m+97*seed);truth={};graphs={};arms=['S14',*EPI_ARMS]+([] if m==1 else list(ONCE_ARMS))
    for arm in arms:
     if arm in fails:continue
     try:
      graphs[arm]=TimelineGraph(lambda arm=arm:epilogue_routed(pr,rs,st,arm));worst[arm]=max(worst.get(arm,(0.,0.)),reuse_gate(graphs[arm].replay(),w,truth,pr,rs))
     except Exception as e:fails[arm]=f'M{m} seed{seed} {type(e).__name__}: {str(e)[:110]}'
    if 'S14' not in graphs:continue
    local=graphs['S14'].replay()[4];counts=torch.bincount(local[local>=0].long(),minlength=32).cpu().tolist();unique=sum(n>0 for n in counts)*28311552
    ctrl=TimelineGraph(lambda:epilogue_routed(pr,rs,st,'S14'));a,b,_=timeline_timing(graphs['S14'],ctrl);aa[m].append(b-a);times.setdefault((m,'S14'),[]).append(a)
    order=[a for a in arms if a!='S14'];order.reverse() if seed%2 else None
    for arm in order:
     if arm in fails:continue
     a,b,r=timeline_timing(graphs['S14'],graphs[arm]);times.setdefault((m,arm),[]).append(b);ratios.setdefault((m,arm),[]).append(r);deltas.setdefault((m,arm),[]).append(b-a)
     worst[arm]=max(worst.get(arm,(0.,0.)),reuse_gate(graphs[arm].replay(),w,truth,pr,rs))
    for arm in ('S14','VEC','PROD','NORM','PAIR'):
     if arm in graphs and arm not in fails:
      try:v=epi_nodes(graphs[arm],arm);nodes.setdefault((m,arm),[]).append(v);bws.setdefault((m,arm),[]).append(unique/v/1e6)
      except Exception as e:lines.append(f'NODE_ERROR {arm} M{m} {str(e)[:80]}')
    if m==32:
     grid=min(264,st['sms']*epin_module().epi_info(69,2,m)[0]);atomic.append(norm_atomic_model(counts,grid))
     if seed<8:
      for arm in (*EPI_ARMS,*ONCE_ARMS):
       if arm in fails:continue
       try:trace.setdefault(arm,[]).append(epilogue_trace(None,graphs[arm],st,m,arm,pr,rs,w,truth))
       except Exception as e:lines.append(f'TRACE_ERROR {arm} {type(e).__name__} {str(e)[:100]}')
    graphs={};truth={}
   done+=1;peak=max(peak,time.monotonic()-start)
 finally:torch.backends.cuda.matmul.allow_tf32=oldtf
 for arm,why in fails.items():lines.append('FAIL '+arm+' '+why)
 lines.append('GATE '+','.join(f'{a}:{v[0]:.5f}/{v[1]:.5f}' for a,v in sorted(worst.items())))
 for m in (1,8,16,32):
  for arm in ('S14',*EPI_ARMS,*ONCE_ARMS):
   vals=times.get((m,arm),[])
   if not vals:continue
   rr=ratios.get((m,arm),[1.]*len(vals));line=f'PAIR M{m} {arm} us={statistics.mean(vals):.3f} r={statistics.mean(rr):.5f} se={statistics.stdev(rr)/len(rr)**.5 if len(rr)>1 else 0:.5f} wins={sum(v<1 for v in rr)}/{len(rr)}'
   if (m,arm) in deltas:line+=f' d_us={statistics.mean(deltas[m,arm]):+.3f}'
   if (m,arm) in nodes:line+=f' node={statistics.mean(nodes[m,arm]):.3f} TBs={statistics.mean(bws[m,arm]):.3f}'
   lines.append(line)
  if aa[m]:lines.append(f'AA M{m} abs={statistics.mean(map(abs,aa[m])):.3f} us')
 for arm,rs in trace.items():
  parts=[statistics.mean(r[2][j] for r in rs) for j in range(len(rs[0][2]))];subs=[statistics.mean(r[3][j] for r in rs) for j in range(len(rs[0][3]))]
  lines.append(f'TRACE M32 {arm} n={len(rs)} observer={statistics.mean(r[0] for r in rs):.4f} d_us={statistics.mean(r[1] for r in rs):+.3f} phase_us='+','.join(f'{v:.3f}' for v in parts)+' sub_us='+','.join(f'{v:.3f}' for v in subs)+f' eff_GHz={statistics.mean(r[4] for r in rs):.4f}')
 if atomic:lines.append('NORM_MODEL host_schedule mean W2tiles/runs/scalar_atomics_before/after='+','.join(f'{statistics.mean(r[j] for r in atomic):.2f}' for j in range(4))+'; simulation of exact S14 unit order,not a hardware counter')
 lines.append(f'END seeds={done} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}; promote only by paired whole-layer S14 ratios')
 return lines
