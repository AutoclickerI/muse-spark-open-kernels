LAD_NAMES=['L0','L0S','L1','L2','L3','BASE','BSINK','MATH','STREAM','META','ACT','LOOP','EARLY','AA','HIST','ROUTER']
LAD_MODE={'L0':10,'L0S':0,'L1':1,'L2':2,'L3':3,'BSINK':9,'MATH':5,'STREAM':8,'META':6,'ACT':7,'HIST':4,'LOOP':11}
LAD_PROTOCOL=dict(name='palindromic-cohort-v1',rounds=4,seed_count=8,aggregation='arithmetic-mean-all-rounds',selective_retiming=False)

def lad_args(data,state,arm):
 M=data['hidden_states'].shape[0];i=0 if M==1 else 1 if M==8 else 2;prefix=('ear'+str(i)) if arm=='EARLY' and i else ('lad'+str(i));F=95 if M==1 else 69;mod=globals()[prefix+'_module']();dev=data['hidden_states'].device;p=M*8;grid=min(264,state['sms']*mod.epi_info(F,2,M)[0])
 inter=torch.full((p,4096),.01,device=dev,dtype=torch.bfloat16);y=torch.zeros((p,4096),device=dev);ss=torch.ones(p*2,device=dev);out=torch.empty_like(data['hidden_states']);cnt=torch.zeros_like(state[prefix+'_counters'][F]);trace=torch.zeros((grid,64),device=dev,dtype=torch.int64)
 bank=state['banks']['lop3'];tensors=[data['hidden_states'],data['topk_ids'],data['topk_weights'],state['pnw'],bank['w13p'],bank['w2p'],state['gs13'],state['cancel_meta'] if i==0 else state['gs2'],inter,y,ss,cnt,out,trace];ptrs=[v.data_ptr() for v in tensors];stream=torch.cuda.current_stream(dev).cuda_stream
 def run():
  stream=torch.cuda.current_stream(dev).cuda_stream
  if arm in ('BASE','AA','EARLY'):mod.run(ptrs,M,0,2,grid,stream,F)
  else:mod.lad_run(ptrs,M,0,2,grid,stream,F,LAD_MODE[arm],arm=='HIST')
  return out
 mode=LAD_MODE.get(arm,4);info=mod.epi_info(F,2,M) if arm in ('BASE','AA','EARLY') else mod.lad_info(M,F,mode,arm=='HIST')
 return dict(fn=run,tensors=tensors,trace=trace,info=list(info)+[grid])

def lad_hist(stamps):
 import collections
 rows=stamps.cpu().tolist();origin=min(v[0] for v in rows);hist=collections.Counter();sm={}
 for v in rows:
  tiles=int(v[8]+v[9]);experts=int(v[10]).bit_count();finish=round((v[7]-origin)/2048);hist[(tiles,experts,finish)]+=1
  s=sm.setdefault(int(v[1]),[0,0,0,0]);s[0]+=1;s[1]+=tiles;s[2]+=experts;s[3]=max(s[3],finish)
 starts=[v[0]-origin for v in rows];w2=[v[5]-origin for v in rows];ends=[v[7]-origin for v in rows]
 return dict(quantum_ns=2048,CTA_joint=[[a,b,c,n] for (a,b,c),n in sorted(hist.items())],SM_rows=[[k,*v] for k,v in sorted(sm.items())],endpoints_ns=[min(starts),max(starts),min(w2),statistics.median(w2),max(w2),min(ends),statistics.median(ends),max(ends)],phase_ns=[statistics.median(v[k]-v[0] for v in rows) for k in (2,4,5,6,7)],CTA_count=len(rows))

def lab(deadline,weights):
 import array,base64,json
 from concurrent.futures import ThreadPoolExecutor
 start=time.monotonic();deadline-=5;times=[];telemetry=[];hist=[];resources=[];gates={};failures=[];enq=[];done=[];work=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in [pool.submit(globals()['lad'+str(i)+'_module']) for i in range(3)]+[pool.submit(globals()['ear'+str(i)+'_module']) for i in (1,2)]:f.result()
 compile_s=time.monotonic()-start
 for seed in range(8):
  if deadline-time.monotonic()<30:break
  w=rg_generate_weights(1375353+7919*seed,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=nt1_banks(w);state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']}
  for i in range(3):globals()['lad'+str(i)+'_bind'](state,{1:95,8:69})
  for i in (1,2):globals()['ear'+str(i)+'_bind'](state,{8:69})
  for M in (1,8,16,32):
   pr,rs=routed_fixture(w,M,1379000+M+97*seed);x,ids,tw,lids=route_call(pr,rs);data=dict(hidden_states=x.contiguous(),topk_ids=lids.int().contiguous(),topk_weights=tw.float().contiguous());truth={};args={};graphs={}
   for arm in LAD_NAMES[:-1]:
    args[arm]=lad_args(data,state,arm);aa=args[arm];g=TimelineGraph(aa['fn']);graphs[arm]=g
    if seed==0:resources.append([M,arm,aa['info']])
    if arm in ('BASE','AA','BSINK','HIST','EARLY'):

     try:e=review_gate((g.replay(),x,ids,tw,lids),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0,0]),e)]
     except ValueError as e:
      if arm!='EARLY':raise
      failures.append(dict(scope='arm',arms=['EARLY'],seed=seed,M=M,reason=repr(e)))
   graphs['ROUTER']=TimelineGraph(lambda:route_call(pr,rs))
   orders=list(s21_orders(LAD_NAMES,LAD_PROTOCOL['rounds'],seed));raw={a:[] for a in LAD_NAMES};host={a:[] for a in LAD_NAMES};tel=[s21_telemetry()]
   for r,order in enumerate(orders):
    events=[]
    for arm in order:
     a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();t=time.perf_counter_ns();a.record();graphs[arm].replay();b.record();host[arm].append(round((time.perf_counter_ns()-t)/1000));events.append((arm,a,b))
    torch.cuda.synchronize()
    for arm,a,b in events:raw[arm].append(round(a.elapsed_time(b)*1e6))
    if r%2:tel.append(s21_telemetry())
   # All rounds,not selected medians. Integer ns retains the CUDA event values to1ns.
   flat=array.array('I',[v for a in LAD_NAMES for v in raw[a]]);times.append([seed,M,base64.b85encode(flat.tobytes()).decode()]);telemetry.append([seed,M,tel])
   severity=max(max(v)/statistics.mean(v) for v in raw.values());enq.append([severity,seed,M,[host[a] for a in ('BASE','AA','BSINK')]]);enq.sort(reverse=True);del enq[4:]
   h=lad_hist(args['HIST']['trace']);
   if True:
    import collections
    h.pop('SM_rows');c=collections.Counter()
    for t,e,f,n in h['CTA_joint']:c[(t,e)]+=n
    h['CTA_joint']=[[*k,n] for k,n in sorted(c.items())];h['joint_finish_retained']=False;h.pop('phase_ns')
   hist.append([seed,M,h]);counts=torch.bincount(lids[lids>=0].flatten(),minlength=32).cpu().tolist();work.append([seed,M,counts])
   for arm in ('BASE','AA','BSINK','HIST','EARLY'):

    try:e=review_gate((graphs[arm].replay(),x,ids,tw,lids),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0,0]),e)]
    except ValueError as e:
     if arm!='EARLY':raise
     failures.append(dict(scope='arm',arms=['EARLY'],seed=seed,M=M,reason=repr(e)))
   del args,graphs,truth
  if seed==0:
   for nm in (9,24):
    pr,rs=routed_fixture(w,nm,1379000+nm);x,ids,tw,lids=route_call(pr,rs);nd=dict(hidden_states=x.clone(),topk_ids=lids,topk_weights=tw,config={'local_start':0});ng=TimelineGraph(lambda:ear2_experts(nd,state,69))
    for scale in (1.,.75):
     nd['hidden_states'].mul_(scale);got=ng.replay();want=nt1_fp32(nd,w);ok,total,row=task_check(got,want)
     if not ok:failures.append(dict(scope='arm',arms=['EARLY'],seed=0,M=nm,reason='nearby TASK '+str((total,row))))
  done.append(seed)
 packet=dict(kind='P3A_LADDER',protocol=LAD_PROTOCOL,identities_sha256=hashlib.sha256(json.dumps(LAD_IDENTITIES,sort_keys=True).encode()).hexdigest(),seeds=done,complete=len(done)==8,arm_names=LAD_NAMES,round_ns_u32le_b85=times,telemetry=telemetry,host_enqueue_top4=enq,histograms=hist,resources=resources,work_counts=work,checks=dict(gates=gates,failures=failures,diagnostics='L0-L3/MATH/STREAM/META/ACT outputs ignored;never nominees'),clock_error=S21_TELEMETRY_ERROR,compile_seconds=compile_s,elapsed_seconds=time.monotonic()-start,sink='Common32volatileSTS128+scale sink inL0/1/2/3/BSINK;BSINK-BASE calibrates;L2 adds activation+MMA;ACT-META isolates standalone activation transport+sinks;MATH no rawglobal copies',hist_scope='All-seed CTA joint(tile count,expert count,finish256ns)histograms;seed0 perSM aggregates;minimal observer HIST timed in same cohort;no normal-kernel phase attribution without observer check')
 for key in ('sink','hist_scope'):packet.pop(key)
 out=[]
 for seed,M,b in packet.pop('round_ns_u32le_b85'):
  ar=array.array('I');ar.frombytes(base64.b85decode(b));last=ar[0]//32;delta=[last]
  for x in ar:delta.append(x//32-last);last=x//32
  out.append([seed,M,delta,[x%32 for x in ar]])
 packet['round_delta32_remainder_ns']=out
 import lzma
 health=packet['checks'];complete=packet['complete'];packet.pop('checks');packet.pop('complete');packet.pop('seeds')
 raw=json.dumps(packet,sort_keys=True,separators=(',',':')).encode()
 return dict(packet=dict(kind='P3A_LADDER',identities={'S19':LAD_IDENTITIES['base_S19'],'bundle':hashlib.sha256(json.dumps(LAD_IDENTITIES,sort_keys=True).encode()).hexdigest()},seeds=done,checks=health,complete=complete,paired=dict(codec='lzma-json-b85',data=base64.b85encode(lzma.compress(raw,preset=9)).decode())))
