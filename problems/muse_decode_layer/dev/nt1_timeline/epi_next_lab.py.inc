NEXT_ARMS={'PAIR':'epit','TREE':'epiu','FOUR':'epif','LOCAL':'epil','FOUR_LOCAL':'epix'}

def next_routed(pr,rs,st,arm,trace=False,stamps=None):
 x,ids,tw,lids=route_call(pr,rs);m=x.shape[0];d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 f=95 if m==1 else 69
 if arm=='S14':out=smallm_experts(d,st,f)
 else:out=globals()[NEXT_ARMS[arm]+'_experts'](d,st,f|256 if trace else f,stamps=stamps)
 return out,x,ids,tw,lids

def next_node(g):
 act=activity_module();act.profile_begin()
 try:
  for _ in range(3):lab_cold_l2();g.replay();torch.cuda.synchronize()
 finally:records=act.profile_end()
 rows=[v for k,v in records if 'moe_var' in k and v[4]>0][-3:]
 if len(rows)!=3:raise RuntimeError('CUPTI missing consumer')
 return statistics.median((v[1]-v[0])/1000 for v in rows)

def next_trace(off,st,m,arm,pr,rs,w,truth):
 p=NEXT_ARMS[arm];mod=globals()[p+'_module']();grid=min(264,st['sms']*mod.epi_info(325,2,m)[0]);stamps=torch.zeros((grid,64),device='cuda',dtype=torch.int64)
 trace=TimelineGraph(lambda:next_routed(pr,rs,st,arm,True,stamps));reuse_gate(trace.replay(),w,truth,pr,rs)
 a,b,r=timeline_timing(off,trace);trace.replay();torch.cuda.synchronize();rows=stamps.cpu().tolist()
 live=[v for v in rows if v[8]>v[0]>0];v=max(live,key=lambda x:x[8]);ghz=statistics.median((v[21]-v[20])/(v[9]-v[0]) for v in live if v[9]>v[0] and v[21]>v[20])
 phases=[(v[1]-v[0])/1000,(v[3]-v[2])/1000,(v[5]-v[3])/1000,(v[7]-v[5])/1000,(v[8]-v[7])/1000]
 subs=[v[i]/1000 for i in (52,53,54,55,56,57,60,61,62)]
 return r,b-a,phases,subs,ghz

def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();deadline-=5;dev=torch.device('cuda')
 lines=['META P3a second round independent of reset. Exact S14 f1c77ce6 and queued PAIR control; TREE balanced8-input FP32 add tree, FOUR4 half-active writer warps with same32 modeled shared read wavefronts, LOCAL CTA-local norm sums+coalesced final gather removes all global norm atomics, FOUR_LOCAL combines writer ownership+local sums. All same lossless packed bank, Ksplit8,2 raw stages,dequant/MMA. No skipped work or prepare expansion.',
 'META 32 weight/router seeds atM8/16/32, flags69,1GiB readflush,AB/BAx6,A:A; direct PAIR vs each new arm plusS14; TASK gates before/after; separate3sample CUPTI consumer nodes+unique byte TB/s.16seed phase traces atM8/16/32: route,W13,W2+ready,grid barrier,combine; sub52/53 store+bar,54/55 producer,56/57 writer0,60 norm flush,61 local export,62 final gather. Overlapped subclocks not additive. Observer/off and effective GHz measured. M1 preserved95 native tested locally.']
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in [pool.submit(fn) for fn in (smallm_module,epit_module,epiu_module,epif_module,epil_module,epix_module,activity_module)]:f.result()
 lines.append(f'BUILD seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
 data={};direct={};nodes={};aa={m:[] for m in (8,16,32)};traces={};worst={};fails={};models={m:[] for m in aa};route_classes={m:[0,0,0,0] for m in aa};peak=0.;done=0
 oldtf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
 try:
  for seed in range(32):
   if deadline-time.monotonic()<max(10,peak*1.2+2):lines.append(f'BUDGET before_seed={seed}');break
   start=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=dev);w['post_norm_w']=w['post_expert_norm'];base=nt1_banks(w)
   st={k:base[k] for k in ('pnw','gs13','gs2','sms')};st['banks']={'lop3':base['bank']};smallm_bind(st,{1:95,8:69,16:69,32:69})
   for p in NEXT_ARMS.values():globals()[p+'_bind'](st,{1:95,8:69,16:325})
   if seed==0:
    for arm,p in NEXT_ARMS.items():lines.append(f'GEOM {arm} off/trace resident,regs,local,dynamic,static='+str([globals()[p+'_module']().epi_info(f,2,32) for f in (69,325)]))
   for m in (8,16,32):
    pr,rs=routed_fixture(w,m,9000+m+97*seed);truth={};graphs={}
    for arm in ('S14',*NEXT_ARMS):
     if arm in fails:continue
     try:
      graphs[arm]=TimelineGraph(lambda arm=arm:next_routed(pr,rs,st,arm));worst[arm]=max(worst.get(arm,(0.,0.)),reuse_gate(graphs[arm].replay(),w,truth,pr,rs))
     except Exception as e:fails[arm]=f'M{m} seed{seed} {type(e).__name__}: {str(e)[:120]}'
    if 'S14' not in graphs or 'PAIR' not in graphs:continue
    local=graphs['S14'].replay()[4];counts=torch.bincount(local[local>=0].long(),minlength=32).cpu().tolist();unique=sum(n>0 for n in counts)*28311552
    grid=min(264,st['sms']*epit_module().epi_info(69,2,m)[0]);models[m].append(norm_atomic_model(counts,grid))
    for n in counts:
     if n:route_classes[m][(n+7)//8-1]+=1
    ctrl=TimelineGraph(lambda:next_routed(pr,rs,st,'S14'));a,b,_=timeline_timing(graphs['S14'],ctrl);aa[m].append(b-a)
    order=list(NEXT_ARMS);order.reverse() if seed%2 else None
    for arm in order:
     if arm in fails:continue
     a,b,r=timeline_timing(graphs['S14'],graphs[arm]);data.setdefault((m,arm),[]).append((a,b,r,b-a))
     if arm!='PAIR':
      x,y,ratio=timeline_timing(graphs['PAIR'],graphs[arm]);direct.setdefault((m,arm),[]).append((x,y,ratio,y-x))
     worst[arm]=max(worst.get(arm,(0.,0.)),reuse_gate(graphs[arm].replay(),w,truth,pr,rs))
    for arm,g in graphs.items():
     try:node=next_node(g);nodes.setdefault((m,arm),[]).append((node,unique/node/1e6))
     except Exception as e:lines.append(f'NODE_ERROR {arm} M{m} {str(e)[:80]}')
    if seed<16:
     for arm in NEXT_ARMS:
      if arm in fails:continue
      try:traces.setdefault((m,arm),[]).append(next_trace(graphs[arm],st,m,arm,pr,rs,w,truth))
      except Exception as e:lines.append(f'TRACE_ERROR {arm} M{m} {type(e).__name__} {str(e)[:100]}')
   done+=1;peak=max(peak,time.monotonic()-start)
 finally:torch.backends.cuda.matmul.allow_tf32=oldtf
 for arm,why in fails.items():lines.append('FAIL '+arm+' '+why)
 lines.append('GATE '+','.join(f'{a}:{v[0]:.5f}/{v[1]:.5f}' for a,v in sorted(worst.items())))
 for label,table in (('S14',data),('PAIR',direct)):
  for (m,arm),vals in sorted(table.items()):
   rr=[r[2] for r in vals];line=f'VS_{label} M{m} {arm} us={statistics.mean(r[1] for r in vals):.3f} ratio={statistics.mean(rr):.5f} se={statistics.stdev(rr)/len(rr)**.5 if len(rr)>1 else 0:.5f} wins={sum(r<1 for r in rr)}/{len(rr)} d_us={statistics.mean(r[3] for r in vals):+.3f}'
   lines.append(line)
 for (m,arm),vals in sorted(nodes.items()):lines.append(f'NODE M{m} {arm} us={statistics.mean(r[0] for r in vals):.3f} TBs={statistics.mean(r[1] for r in vals):.3f}')
 for m,vals in aa.items():
  if vals:lines.append(f'AA M{m} abs={statistics.mean(map(abs,vals)):.3f} us')
 for (m,arm),rows in sorted(traces.items()):
  phases=[statistics.mean(r[2][j] for r in rows) for j in range(5)];subs=[statistics.mean(r[3][j] for r in rows) for j in range(9)]
  lines.append(f'TRACE M{m} {arm} n={len(rows)} observer={statistics.mean(r[0] for r in rows):.4f} d_us={statistics.mean(r[1] for r in rows):+.3f} phase_us='+','.join(f'{v:.3f}' for v in phases)+' sub_us='+','.join(f'{v:.3f}' for v in subs)+f' eff_GHz={statistics.mean(r[4] for r in rows):.4f}')
 for m,vals in models.items():
  if vals:lines.append(f'MODEL M{m} W2tiles,runs,S14scalar_atomics,PAIRscalar_atomics='+','.join(f'{statistics.mean(r[j] for r in vals):.2f}' for j in range(4))+'; FOUR has2x PAIR scalar atoms/run,LOCAL0;LOCAL exports npairs*grid floats,finalizer gathers8*grid floats/CTA')
 for m,h in route_classes.items():
  experts=sum(h);blocks=sum((i+1)*v for i,v in enumerate(h));ratio=blocks/max(experts,1)
  lines.append(f'ISSUE_ROUTE M{m} experts_by_nb={h} repeated_blocks_per_unique={ratio:.5f} S14_math_per_unique16={28.5625*ratio:.5f} scalar_epi_per_unique16={(185*2+308)/384*ratio:.5f}; static issue model excludes outer work')
 lines.append(f'END seeds={done} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}; no promotion without paired whole-layer validation')
 return lines
