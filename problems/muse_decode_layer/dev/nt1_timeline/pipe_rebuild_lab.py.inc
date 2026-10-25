REBUILD_ARMS={'FLIGHT':'pfly','ILP':'pilp','K16':'pk16','LUT':'plut'}

def rebuild_routed(pr,rs,st,arm):
 x,ids,tw,lids=route_call(pr,rs)
 d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 out=smallm_experts(d,st,69) if arm=='S14' else globals()[REBUILD_ARMS[arm]+'_experts'](d,st,69)
 return out,x,ids,tw,lids

def rebuild_node(g):
 mod=activity_module();mod.profile_begin()
 try:
  for _ in range(3):lab_cold_l2();g.replay();torch.cuda.synchronize()
 finally:rows=mod.profile_end()
 vals=[(v[1]-v[0])/1000 for k,v in rows if 'moe_var' in k and v[4]>0][-3:]
 if len(vals)!=3:raise RuntimeError('missing consumer CUPTI records')
 return statistics.median(vals)

def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();deadline-=5
 lines=['META P3a conditional rebuild batch,UNSENT until pipe diagnostic isolates a bound. Exact S14 f1c77ce6 flags69,FLIGHT bulkL2 lookahead,ILP two MMA chains,K16 16warps/CTA64regs two H200 CTAs with timed shared B-cache,LUT fixed FP4 code-pair table and original timed scale math. Same precision and complete work;M1 remains original95 in integration patches.',
 'META 32 router/weight seeds,M8/16/32,readflush ABBAx6,A:A,TASK gates before/after,3sample CUPTI consumer nodes,unique-byte TB/s. All four responses fit one paired run;classification from the earlier diagnostic is a condition to send this batch,not a reason to promote any arm.']
 fns=[smallm_module,activity_module]+[globals()[p+'_module'] for p in REBUILD_ARMS.values()]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in [pool.submit(fn) for fn in fns]:f.result()
 lines.append(f'BUILD seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
 data={};nodes={};aa={m:[] for m in (8,16,32)};worst={};fails={};done=0;peak=0.
 for seed in range(32):
  if deadline-time.monotonic()<max(10,peak*1.2+2):lines.append(f'BUDGET before_seed={seed}');break
  start=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];base=nt1_banks(w)
  st={k:base[k] for k in ('pnw','gs13','gs2','sms')};st['banks']={'lop3':base['bank']};smallm_bind(st,{8:69})
  for p in REBUILD_ARMS.values():globals()[p+'_bind'](st,{8:69})
  if seed==0:
   for arm,p in REBUILD_ARMS.items():lines.append(f'GEOM {arm} resident,regs,local,dynamic,static='+str(globals()[p+'_module']().epi_info(69,2,32)))
  for m in aa:
   pr,rs=routed_fixture(w,m,9000+m+97*seed);truth={};graphs={}
   for arm in ('S14',*REBUILD_ARMS):
    if arm in fails:continue
    try:
     g=TimelineGraph(lambda arm=arm:rebuild_routed(pr,rs,st,arm));worst[arm]=max(worst.get(arm,(0.,0.)),reuse_gate(g.replay(),w,truth,pr,rs));graphs[arm]=g
    except Exception as e:fails[arm]=f'M{m} seed{seed} {type(e).__name__}: {str(e)[:140]}'
   if 'S14' not in graphs:continue
   local=graphs['S14'].replay()[4];unique=torch.unique(local[local>=0]).numel()*28311552
   ctrl=TimelineGraph(lambda:rebuild_routed(pr,rs,st,'S14'));a,b,_=timeline_timing(graphs['S14'],ctrl);aa[m].append(b-a)
   order=list(REBUILD_ARMS);order.reverse() if seed%2 else None
   for arm in order:
    if arm not in graphs:continue
    a,b,r=timeline_timing(graphs['S14'],graphs[arm]);data.setdefault((m,arm),[]).append((a,b,r,b-a))
    worst[arm]=max(worst[arm],reuse_gate(graphs[arm].replay(),w,truth,pr,rs))
   for arm,g in graphs.items():
    try:
     node=rebuild_node(g);nodes.setdefault((m,arm),[]).append((node,unique/node/1e6))
    except Exception as e:lines.append(f'NODE_ERROR M{m} {arm} {str(e)[:100]}')
  done+=1;peak=max(peak,time.monotonic()-start)
 for arm,why in fails.items():lines.append('FAIL '+arm+' '+why)
 lines.append('GATE '+','.join(f'{a}:{v[0]:.5f}/{v[1]:.5f}' for a,v in sorted(worst.items())))
 for (m,arm),vs in sorted(data.items()):
  rr=[v[2] for v in vs];lines.append(f'PAIR M{m} {arm} S14_us={statistics.mean(v[0] for v in vs):.3f} arm_us={statistics.mean(v[1] for v in vs):.3f} ratio={statistics.mean(rr):.5f} se={statistics.stdev(rr)/len(rr)**.5 if len(rr)>1 else 0:.5f} wins={sum(r<1 for r in rr)}/{len(rr)} delta_us={statistics.mean(v[3] for v in vs):+.3f}')
 for (m,arm),vs in sorted(nodes.items()):lines.append(f'NODE M{m} {arm} us={statistics.mean(v[0] for v in vs):.3f} unique_TBs={statistics.mean(v[1] for v in vs):.3f} n={len(vs)}')
 for m,vs in aa.items():
  if vs:lines.append(f'AA M{m} abs_us={statistics.mean(map(abs,vs)):.3f}')
 lines.append(f'END seeds={done} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f};promotion needs paired whole-layer evidence')
 return lines
