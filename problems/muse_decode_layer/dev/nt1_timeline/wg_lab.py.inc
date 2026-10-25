def wg_routed(pr,rs,state,arm):
 x,ids,tw,lids=route_call(pr,rs);d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 out=smallm_experts(d,state,69) if arm=='S11' else wg_experts(d,state,*WG_ARMS[arm])
 return out,x,ids,tw,lids
WG_ARMS={'W4G2':(4,2),'W4G4':(4,4),'W8G2':(8,2),'W8G4':(8,4)}

def wg_node(graph,arm):
 act=activity_module();act.profile_begin()
 try:
  for _ in range(3):lab_cold_l2();graph.replay();torch.cuda.synchronize()
 finally:rs=act.profile_end()
 name='moe_var' if arm=='S11' else 'moe_wg';rs=[r for r in rs if name in r[0] and r[1][4]>0][-3:]
 if len(rs)!=3:raise ValueError('incomplete CUPTI '+arm)
 return statistics.median((r[1][1]-r[1][0])/1000 for r in rs)

def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();deadline-=5;device=next(v.device for v in weights.values() if isinstance(v,torch.Tensor))
 lines=['META P3a structural RS WGMMA:64 output rows x adaptive8/16/32 expert-token columns; original S11 code bytes plus losslessly reordered raw u8 scale pairs; every expert weight decoded once for all token blocks. No K-split shared reduction; norm atomics4x fewer. Stages4/8, requested grid2/4 CTAs/SM capped by real residency. SM120 uses NF portable MMA stand-in; native H200 output must independently pass.',
  'META exact S11 acda6c62 flags69 comparator; router+MoE CUDA graphs paired AB/BA x6, read flush,16 weight/router seeds; TASK routing+output gate before/after; independent S11 A:A; separate3-sample CUPTI consumer-node times per seed for unique logical TB/s. W4G4/W8G4 request4 but launch bound3 may cap them at3 resident CTAs/SM.']
 with ThreadPoolExecutor(max_workers=3) as pool:
  for j in [pool.submit(f) for f in (smallm_module,wg_module,activity_module)]:j.result()
 lines.append(f'BUILD seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
 ratios={};deltas={};times={};nodes={};bws={};aa={m:[] for m in (8,16,32)};failed={};worst={};done=0;peak=0.
 oldtf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
 try:
  for seed in range(16):
   if deadline-time.monotonic()<max(8,peak*1.2+2):lines.append(f'BUDGET before_seed={seed}');break
   started=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=device);w['post_norm_w']=w['post_expert_norm'];base=nt1_banks(w)
   state={k:base[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':base['bank']};smallm_bind(state,{8:69,16:69,32:69});wg_bind(state)
   if seed==0:
    for name,params in WG_ARMS.items():lines.append('GEOM '+name+' grid,regs,localB,smemB,resident='+str(wg_geometry(state,32,*params)))
   for m in (8,16,32):
    pr,rs=routed_fixture(w,m,9000+m+97*seed);truths={};graphs={}
    x,ids,tw,lids=route_call(pr,rs);counts=torch.bincount(lids[lids>=0].long(),minlength=32)
    unique_bytes=int((counts>0).sum())*28311552
    for arm in ('S11',*WG_ARMS):
     if arm in failed:continue
     try:
      graphs[arm]=TimelineGraph(lambda arm=arm:wg_routed(pr,rs,state,arm))
      worst[arm]=max(worst.get(arm,(0.,0.)),reuse_gate(graphs[arm].replay(),w,truths,pr,rs))
     except Exception as exc:failed[arm]=f'M{m} seed{seed} {type(exc).__name__} {str(exc)[:140]}';graphs.pop(arm,None)
    if 'S11' not in graphs:continue
    ctl=TimelineGraph(lambda:wg_routed(pr,rs,state,'S11'));a,b,_=timeline_timing(graphs['S11'],ctl);aa[m].append(b-a);times.setdefault((m,'S11'),[]).append(a)
    arms=list(WG_ARMS)
    if seed%2:arms.reverse()
    for arm in arms:
     if arm not in graphs:continue
     a,b,r=timeline_timing(graphs['S11'],graphs[arm]);ratios.setdefault((m,arm),[]).append(r);deltas.setdefault((m,arm),[]).append(b-a);times.setdefault((m,arm),[]).append(b)
     try:worst[arm]=max(worst.get(arm,(0.,0.)),reuse_gate(graphs[arm].replay(),w,truths,pr,rs))
     except Exception as exc:failed[arm]=f'M{m} seed{seed} after {type(exc).__name__} {str(exc)[:140]}'
    for arm,g in graphs.items():
     try:
      node=wg_node(g,arm);nodes.setdefault((m,arm),[]).append(node);bws.setdefault((m,arm),[]).append(unique_bytes/node/1e6)
     except Exception as exc:lines.append(f'NODE_ERROR {arm} M{m} seed{seed} {str(exc)[:100]}')
    graphs={};truths={}
   done+=1;peak=max(peak,time.monotonic()-started)
 finally:torch.backends.cuda.matmul.allow_tf32=oldtf
 for name,why in failed.items():lines.append('FAIL '+name+' '+why)
 lines.append('GATE '+','.join(f'{k}:{v[0]:.5f}/{v[1]:.5f}' for k,v in sorted(worst.items())))
 for m in (8,16,32):
  for arm in ('S11',*WG_ARMS):
   ts=times.get((m,arm),[])
   if not ts:continue
   rs=ratios.get((m,arm),[1.]*len(ts));se=statistics.stdev(rs)/len(rs)**.5 if len(rs)>1 else 0.
   parts=[f'event_us={statistics.mean(ts):.3f}',f'r={statistics.mean(rs):.5f}',f'se={se:.5f}',f'n={len(rs)}',f'wins={sum(x<1 for x in rs)}/{len(rs)}']
   if (m,arm) in deltas:parts.append(f'd_us={statistics.mean(deltas[m,arm]):+.3f}')
   if (m,arm) in nodes:parts.append(f'node_us={statistics.mean(nodes[m,arm]):.3f} TBs={statistics.mean(bws[m,arm]):.3f}')
   lines.append(f'AGG M{m} {arm} '+' '.join(parts))
  if aa[m]:lines.append(f'AA M{m} d_us={statistics.mean(aa[m]):+.3f} abs={statistics.mean(map(abs,aa[m])):.3f}')
 lines.append(f'END seeds={done} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}; promote only after paired whole-layer test')
 return lines
