# S13's small-M cell is byte-identical to S11 (c023c946); the complete patch base is S13 958f1258.
PHASE_ARMS={'OLD':None,'BASE':0,'BUNDLE':2}
def phase_routed(pr,rs,state,arm,trace=False,stamps=None):
 x,ids,tw,lids=route_call(pr,rs);d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 if arm=='S13':out=smallm_experts(d,state,69)
 elif arm=='OLD':out=reuse_experts(d,state,0)
 else:out=reusep_experts(d,state,PHASE_ARMS[arm],trace,stamps)
 return out,x,ids,tw,lids

def phase_summary(stamps,arm,seed):
 rows=stamps.cpu().tolist();active=[r for r in rows if r[5]>r[0]>0]
 if not active:raise ValueError('missing phase stamps')
 crit=max(active,key=lambda r:r[5]);h=crit[:6]
 sums=[0.]*6;by={};counts={}
 for r in active:
  for k in range(128):
   q=r[16+k*8:24+k*8]
   if not q[0]:continue
   assert q[0]<=q[3]<=q[4]<=q[5]<=q[6],('clock order',arm,k,q)
   nb=q[7]&255;pas=(q[7]>>8)&255
   vals=[(q[3]-q[0])/1000,(q[4]-q[3])/1000,(q[5]-q[4])/1000,(q[6]-q[5])/1000]
   key=(pas,nb);by.setdefault(key,[]).append(vals);counts[key]=counts.get(key,0)+1
 for k in range(128):
  q=crit[16+k*8:24+k*8]
  if not q[0]:continue
  for j,v in enumerate((q[3]-q[0],q[4]-q[3],q[5]-q[4],q[6]-q[5])):sums[j]+=v/1000
  if q[1]:sums[4]+=(q[1]-q[0])/1000;sums[5]+=(q[2]-q[1])/1000
 # Whole-CTA partition is additive; unit sums are diagnostic subintervals of that same CTA.
 part=[(h[i+1]-h[i])/1000 for i in range(5)]
 groups={key:[statistics.mean(x[j] for x in vals) for j in range(4)] for key,vals in by.items()}
 return part,sums,groups,counts

def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();deadline-=5;device=torch.device('cuda')
 lines=['META P3a M32 decode-once cause: BASE reproduces ONCE math, SPEC block-count specialization + BUNDLE parallel multi-block reduction/epilogues; exact original ONCE retained as OLD control. Same raw packed S13/S11 bank, no K expansion in prepare. S13 final958f1258 small-M cell is byte-identical c023c946; flags69.',
 'META paired router+MoE graphs M8/16/32,16 synthetic weight/router seeds,read flush AB/BAx6,A:A,TASK gates before/after. TRACE separately compiled: globaltimer only at phase/unit boundaries,no per-g clocks.8B trace spills vs176B old; report observer/off ratios. Trace caches4 extra operand words in shared; same2 native H200 CTAs/SM. Only local BUNDLE TRACE uses1 stage to fit sm120. No phase-clock speedup is a promotion.']
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in [pool.submit(fn) for fn in (smallm_module,reuse_module,reusep_module)]:f.result()
 ratios={};deltas={};times={};aa={m:[] for m in (8,16,32)};fails={};worst={};ph={};obs={};group={};counts={};done=0;peak=0.
 oldtf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
 try:
  for seed in range(16):
   if deadline-time.monotonic()<max(8,peak*1.2+2):lines.append(f'BUDGET before_seed={seed}');break
   started=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=device);w['post_norm_w']=w['post_expert_norm'];base=nt1_banks(w)
   state={k:base[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':base['bank']};smallm_bind(state,{8:69,16:69,32:69});reuse_bind(state);reusep_bind(state)
   if seed==0:
    for mode in (0,2):
     for trace in (0,1):lines.append(f'GEOM mode={mode} trace={trace} grid,regs,localB,smemB,resident='+str(reusep_geometry(state,32,mode,trace)))
   for m in (8,16,32):
    pr,rs=routed_fixture(w,m,9000+m+97*seed);truths={};graphs={}
    for arm in ('S13',*PHASE_ARMS):
     if arm in fails:continue
     try:
      graphs[arm]=TimelineGraph(lambda arm=arm:phase_routed(pr,rs,state,arm));worst[arm]=max(worst.get(arm,(0.,0.)),reuse_gate(graphs[arm].replay(),w,truths,pr,rs))
     except Exception as exc:fails[arm]=f'M{m} seed{seed} {type(exc).__name__} {str(exc)[:120]}'
    if 'S13' not in graphs:continue
    control=TimelineGraph(lambda:phase_routed(pr,rs,state,'S13'));a,b,_=timeline_timing(graphs['S13'],control);aa[m].append(b-a);times.setdefault((m,'S13'),[]).append(a)
    arms=list(PHASE_ARMS)
    if seed%2:arms.reverse()
    for arm in arms:
     if arm in fails:continue
     a,b,r=timeline_timing(graphs['S13'],graphs[arm]);ratios.setdefault((m,arm),[]).append(r);deltas.setdefault((m,arm),[]).append(b-a);times.setdefault((m,arm),[]).append(b)
     worst[arm]=max(worst.get(arm,(0.,0.)),reuse_gate(graphs[arm].replay(),w,truths,pr,rs))
    if all(a in graphs for a in ('BASE','BUNDLE')):
     a,b,r=timeline_timing(graphs['BASE'],graphs['BUNDLE']);ratios.setdefault((m,'BUNDLE/BASE'),[]).append(r);deltas.setdefault((m,'BUNDLE/BASE'),[]).append(b-a)
    if m==32:
     for arm,mode in (('BASE',0),('BUNDLE',2)):
      if arm in fails:continue
      try:
       grid=reusep_geometry(state,m,mode,True)[0];stamps=torch.zeros((grid,1040),device=device,dtype=torch.int64)
       traced=TimelineGraph(lambda:phase_routed(pr,rs,state,arm,True,stamps));worst[arm+'T']=max(worst.get(arm+'T',(0.,0.)),reuse_gate(traced.replay(),w,truths,pr,rs))
       a,b,r=timeline_timing(graphs[arm],traced);obs.setdefault(arm,[]).append((r,b-a));traced.replay();torch.cuda.synchronize()
       part,sums,groups,ct=phase_summary(stamps,arm,seed);ph.setdefault(arm,[]).append((part,sums))
       for key,vals in groups.items():group.setdefault((arm,*key),[]).append(vals);counts[(arm,*key)]=counts.get((arm,*key),0)+ct[key]
      except Exception as exc:lines.append(f'TRACE_ERROR {arm} seed{seed} {type(exc).__name__} {str(exc)[:130]}')
    graphs={};truths={}
   done+=1;peak=max(peak,time.monotonic()-started)
 finally:torch.backends.cuda.matmul.allow_tf32=oldtf
 for arm,why in fails.items():lines.append('FAIL '+arm+' '+why)
 lines.append('GATE '+','.join(f'{k}:{v[0]:.5f}/{v[1]:.5f}' for k,v in sorted(worst.items())))
 for m in (8,16,32):
  lines.append(f'CTL M{m} S13_us={statistics.mean(times.get((m,"S13"),[0])):.3f} AA_abs_us={statistics.mean(map(abs,aa[m])) if aa[m] else 0:.3f}')
  for arm in (*PHASE_ARMS,'BUNDLE/BASE'):
   vals=ratios.get((m,arm),[])
   if vals:lines.append(f'PAIR M{m} {arm} r={statistics.mean(vals):.5f} se={statistics.stdev(vals)/len(vals)**.5 if len(vals)>1 else 0:.5f} d_us={statistics.mean(deltas[m,arm]):+.3f} wins={sum(v<1 for v in vals)}/{len(vals)}'+(f' us={statistics.mean(times[m,arm]):.3f}' if (m,arm) in times else ''))
 for arm,rs in ph.items():
  parts=[statistics.mean(r[0][j] for r in rs) for j in range(5)];sums=[statistics.mean(r[1][j] for r in rs) for j in range(6)]
  lines.append(f'PHASE M32 {arm} n={len(rs)} critical route/W13/W2/barrier/final_us='+','.join(f'{x:.3f}' for x in parts)+' critical unit_setup/wait/math/epi_us='+','.join(f'{x:.3f}' for x in sums[:4])+' setup_ready/cache_sub_us='+','.join(f'{x:.3f}' for x in sums[4:]))
  if arm in obs:lines.append(f'OBSERVER {arm} traced/off={statistics.mean(x[0] for x in obs[arm]):.5f} d_us={statistics.mean(x[1] for x in obs[arm]):+.3f}')
 for key,rs in sorted(group.items()):
  vals=[statistics.mean(r[j] for r in rs) for j in range(4)]
  lines.append(f'UNIT {key[0]} pass{key[1]} nb{key[2]} tiles={counts[key]} setup/wait/math/epi_us='+','.join(f'{x:.4f}' for x in vals))
 lines.append(f'END seeds={done} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}; no integration win without paired whole-layer confirmation')
 return lines
