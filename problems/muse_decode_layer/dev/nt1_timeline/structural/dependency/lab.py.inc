DEP_ARMS={'A64':('dep',64,0),'A128':('dep',128,0),'P128':('dep',128,1),'A128D':('dpx',128,0),'P128D':('dpx',128,1),'NORED':('dep',128,2),'NOREDD':('dpx',128,2)}
def dep_call(pr,rs,state,arm):
 x,ids,tw,lids=route_call(pr,rs);M=x.shape[0];data=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 if arm in ('BASE','NOWAIT'):out=globals()['dp_'+arm.lower()+('1' if M==8 else '2')+'_experts'](data,state,69)
 else:
  prefix,S,mode=DEP_ARMS[arm];out=globals()[prefix+'_experts'](data,state,S,mode)
 return out,x,ids,tw,lids

def dep_material(rows):
 return any(statistics.mean(v[4]/1e6 for v in rows if v[1]==M)<.95 for M in (16,32))

def dep_profile(graph):
 mod=activity_module();mod.profile_begin()
 try:
  for _ in range(3):lab_cold_l2();graph.replay();torch.cuda.synchronize()
 finally:records=mod.profile_end()
 d={}
 for name,v in records:
  if v[4]<=0:continue
  key='main' if 'slice_mainILi' in name or 'moe_varILi1ELi69E' in name else 'zero' if 'slice_zeroE' in name else 'norm' if 'slice_normILi' in name else 'combine' if 'slice_combineE' in name else None
  if key:d.setdefault(key,[]).append((v[1]-v[0])/1000)
 assert len(d.get('main',[]))==3,d
 return [round(statistics.median(d.get(k,[0]))*1000) for k in ('main','zero','norm','combine')]

def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();deadline-=5;cases=[];oracle=[];AA=[];gates={};failures=[];paired=[];profiles=[];costs=[];resources=[];material=False;done=0;build_done=0;peak=0.;eligible=['A64','A128','P128','A128D','P128D'];names=['BASE','NOWAIT',*eligible,'NORED','NOREDD']
 fns=[dp_base1_module,dp_base2_module,dp_nowait1_module,dp_nowait2_module,dep_module,dpx_module,activity_module]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in [pool.submit(fn) for fn in fns]:f.result()
 compile_s=time.monotonic()-begin
 for M in (8,16,32):
  i='1' if M==8 else '2'
  for arm in ('BASE','NOWAIT'):resources.append([M,arm,globals()['dp_'+arm.lower()+i+'_module']().epi_info(69,2,M)])
  for arm,(prefix,S,mode) in DEP_ARMS.items():resources.append([M,arm,getattr(globals()[prefix+'_module'](),prefix+'_info')(S,mode)])
 # PhaseA:all16 oracle/control seeds first. Hold inputs/banks,not outputs for production.
 for seed in range(16):
  if deadline-time.monotonic()<max(25,peak*1.5+5):break
  started=time.monotonic();w=rg_generate_weights(1175353+7919*seed,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=nt1_banks(w);state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']}
  for name in ('dp_base1','dp_base2','dp_nowait1','dp_nowait2'):globals()[name+'_bind'](state,{8:69})
  case=dict(seed=seed,w=w,state=state,rows={})
  for M in (8,16,32):
   pr,rs=routed_fixture(w,M,1179000+M+97*seed);truth={};gb=TimelineGraph(lambda:dep_call(pr,rs,state,'BASE'));err=review_gate(gb.replay(),w,truth,pr,rs);gates['BASE']=[max(a,b) for a,b in zip(gates.get('BASE',[0,0]),err)]
   go=TimelineGraph(lambda:dep_call(pr,rs,state,'NOWAIT'))
   # Warm the deliberately wrong oracle's own scratch to finite prior-call data;outputs ignored.
   for _ in range(3):go.replay()
   torch.cuda.synchronize();ga=TimelineGraph(lambda:dep_call(pr,rs,state,'BASE'));a,b,ratio=timeline_timing(gb,ga);AA.append([seed,M,round(a*1000),round(b*1000),round(ratio*1e6)])
   a,b,ratio=timeline_timing(gb,go);oracle.append([seed,M,round(a*1000),round(b*1000),round(ratio*1e6)])
   err=review_gate(gb.replay(),w,truth,pr,rs);gates['BASE']=[max(a,b) for a,b in zip(gates['BASE'],err)]
   if seed<4:
    for arm,g in (('BASE',gb),('NOWAIT',go)):profiles.append([seed,M,names.index(arm),*dep_profile(g)])
   case['rows'][M]=(pr,rs,truth,gb)
  cases.append(case);done+=1;peak=max(peak,time.monotonic()-started)
 material=done==16 and dep_material(oracle)
 # No local-test override is shipped. Testdriver can replace dep_material for branch coverage.
 if material:
  for index,case in enumerate(cases):
   if deadline-time.monotonic()<max(25,peak*1.5+5):break
   started=time.monotonic();seed=case['seed'];w=case['w'];state=case['state'];dep_bind(state,64);dep_bind(state,128);dpx_bind(state,128)
   for M,(pr,rs,truth,gb) in case['rows'].items():
    graphs={}
    for arm in eligible:
     try:
      g=TimelineGraph(lambda arm=arm:dep_call(pr,rs,state,arm));err=review_gate(g.replay(),w,truth,pr,rs);graphs[arm]=g;gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0,0]),err)]
     except Exception as e:failures.append(dict(scope='arm',arm=arm,seed=seed,M=M,error=repr(e)))
    order=eligible if seed%2==0 else eligible[::-1]
    for arm in order:
     if arm not in graphs:continue
     a,b,ratio=timeline_timing(gb,graphs[arm]);err=review_gate(graphs[arm].replay(),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates[arm],err)];paired.append([seed,M,names.index(arm),round(ratio*1e6)])
    out,x,ids,tw,lids=gb.replay();counts=torch.bincount(lids[lids>=0].flatten(),minlength=32).cpu().tolist();p=sum(counts);nb=sum((v+7)//8 for v in counts);costs.append([seed,M,p,nb])
    if seed<4:
     for arm,g in graphs.items():profiles.append([seed,M,names.index(arm),*dep_profile(g)])
     for arm in ('NORED','NOREDD'):
      g=TimelineGraph(lambda arm=arm:dep_call(pr,rs,state,arm));profiles.append([seed,M,names.index(arm),*dep_profile(g)])
    err=review_gate(gb.replay(),w,truth,pr,rs);gates['BASE']=[max(a,b) for a,b in zip(gates['BASE'],err)]
   build_done+=1;peak=max(peak,time.monotonic()-started);cases[index]=None
 complete=done==16 and (not material or build_done==16)
 packet=dict(kind='P3A_DEPENDENCY',identities=DEP_IDENTITIES,seeds=dict(requested=16,oracle_complete=done,build_complete=build_done,weights='1175353+7919*seed',routing='1179000+M+97*seed'),paired=paired,oracle=oracle,AA=AA,profiles=profiles,cost_counts=costs,resources=resources,arm_names=names,checks=dict(gates=gates,failures=failures,oracle='NOWAIT/NORED/NOREDD deliberately ungated;outputs ignored;no nominations'),material=material,material_policy='Any M16/M32 mean NOWAIT/BASE<.95 afterall16seeds',complete=complete,compile_seconds=compile_s,elapsed_seconds=time.monotonic()-begin,profile_units='CUPTI kernel ns,median3 coldreplays perseed;separate from unprofiled paired timings',profile_columns=['main','zero','norm','combine'])
 return dict(packet=packet)
