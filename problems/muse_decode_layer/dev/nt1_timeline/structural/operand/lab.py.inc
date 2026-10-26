OP_NAMES=['BASE','AA','BASE_PAD','FRAG_REG','FRAG_SMEM','FRAG_JIT','ACC_CTRL','ACC_FREE','PAD_CTRL','PAD_USEFUL','L1','L2','L3','HIST','LAZY']
OP_MODE={'FRAG_SMEM':1,'FRAG_REG':2,'FRAG_JIT':3,'ACC_FREE':4,'ACC_CTRL':5,'PAD_USEFUL':6,'PAD_CTRL':7,'LAZY':9,'BASE_PAD':10}
OP_TASK=('BASE','AA','BASE_PAD','ACC_CTRL','PAD_CTRL','HIST','LAZY')
OP_PROTOCOL=dict(name='palindromic-cohort-v1',rounds=4,seed_count=8,aggregation='arithmetic-mean-all-rounds',selective_retiming=False)

def op_args(data,state,arm):
 M=data['hidden_states'].shape[0];i=0 if M==1 else 1 if M==8 else 2;prefix='opa'+str(i);F=95 if M==1 else 69;mod=globals()[prefix+'_module']();dev=data['hidden_states'].device;p=M*8;grid=min(264,state['sms']*mod.epi_info(F,2,M)[0])
 inter=torch.full((p,4096),.01,device=dev,dtype=torch.bfloat16);y=torch.zeros((p,4096),device=dev);ss=torch.ones(p*2,device=dev);out=torch.empty_like(data['hidden_states']);cnt=torch.zeros_like(state[prefix+'_counters'][F]);trace=torch.zeros((grid,64),device=dev,dtype=torch.int64)
 bank=state['banks']['lop3'];tensors=[data['hidden_states'],data['topk_ids'],data['topk_weights'],state['pnw'],bank['w13p'],bank['w2p'],state['gs13'],state['cancel_meta'] if i==0 else state['gs2'],inter,y,ss,cnt,out,trace];ptrs=[v.data_ptr() for v in tensors]
 def run():
  stream=torch.cuda.current_stream(dev).cuda_stream
  if arm in ('BASE','AA'):mod.run(ptrs,M,0,2,grid,stream,F)
  elif arm in OP_MODE:mod.op_run(ptrs,M,0,2,grid,stream,F,OP_MODE[arm])
  else:mod.lad_run(ptrs,M,0,2,grid,stream,F,4 if arm=='HIST' else int(arm[1:]),arm=='HIST')
  return out
 info=mod.epi_info(F,2,M) if arm in ('BASE','AA') else mod.op_info(M,F,OP_MODE[arm]) if arm in OP_MODE else mod.lad_info(M,F,4 if arm=='HIST' else int(arm[1:]),arm=='HIST')
 return dict(fn=run,tensors=tensors,trace=trace,info=list(info)+[grid])

def lab(deadline,weights):
 import array,base64,json,lzma,hashlib,collections
 from concurrent.futures import ThreadPoolExecutor
 started=time.monotonic();deadline-=5;times=[];telemetry=[];resources=[];gates={};failures=[];done=[];work=[];hist=[];host_top=[]
 assert set(OP_NAMES)-{'AA'}<=set(OP_QUALIFIED)
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in [pool.submit(globals()['opa'+str(i)+'_module']) for i in range(3)]:f.result()
 compile_s=time.monotonic()-started
 for seed in range(8):
  if deadline-time.monotonic()<30:break
  w=rg_generate_weights(1375353+7919*seed,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=nt1_banks(w);state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']}
  for i in range(3):globals()['opa'+str(i)+'_bind'](state,{1:95,8:69})
  for M in (1,8,16,32):
   pr,rs=routed_fixture(w,M,1379000+M+97*seed);x,ids,tw,lids=route_call(pr,rs);data=dict(hidden_states=x.contiguous(),topk_ids=lids.int().contiguous(),topk_weights=tw.float().contiguous());truth={};args={};graphs={}
   for arm in OP_NAMES:
    args[arm]=op_args(data,state,arm);graphs[arm]=TimelineGraph(args[arm]['fn'])
    if seed==0:resources.append([M,arm,args[arm]['info']])
    if arm in OP_TASK:
     try:
      e=review_gate((graphs[arm].replay(),x,ids,tw,lids),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0,0]),e)]
     except ValueError as exc:
      if arm!='LAZY':raise
      failures.append(dict(scope='arm',arms=['LAZY'],seed=seed,M=M,reason=repr(exc)))
   raw={a:[] for a in OP_NAMES};host={a:[] for a in OP_NAMES};tel=[s21_telemetry()]
   for rr,order in enumerate(s21_orders(OP_NAMES,OP_PROTOCOL['rounds'],seed)):
    events=[]
    for arm in order:
     a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();t=time.perf_counter_ns();a.record();graphs[arm].replay();b.record();host[arm].append(round((time.perf_counter_ns()-t)/1000));events.append((arm,a,b))
    torch.cuda.synchronize()
    for arm,a,b in events:raw[arm].append(round(a.elapsed_time(b)*1e6))
    if rr%2:tel.append(s21_telemetry())
   flat=[v for a in OP_NAMES for v in raw[a]];ticks=[v//32 for v in flat];delta=[ticks[0]]+[ticks[j]-ticks[j-1] for j in range(1,len(ticks))]
   times.append([seed,M,delta,[v%32 for v in flat]]);telemetry.append([seed,M,tel])
   severity=max(max(v)/statistics.mean(v) for v in raw.values());host_top.append([severity,seed,M,[host[a] for a in OP_NAMES]]);host_top.sort(reverse=True);del host_top[4:]
   counts=torch.bincount(lids[lids>=0].flatten(),minlength=32).cpu().tolist();work.append([seed,M,counts]);h=lad_hist(args['HIST']['trace'])
   if seed!=0:
    h.pop('SM_rows');c=collections.Counter()
    for t,e,f,n in h['CTA_joint']:c[(t,e)]+=n
    h['CTA_joint']=[[*k,n] for k,n in sorted(c.items())]
   else:
    h['SM_columns']=[list(v) for v in zip(*h.pop('SM_rows'))];h['CTA_joint_columns']=[list(v) for v in zip(*h.pop('CTA_joint'))]
   hist.append([seed,M,h])
   for arm in OP_TASK:
    try:
     e=review_gate((graphs[arm].replay(),x,ids,tw,lids),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0,0]),e)]
    except ValueError as exc:
     if arm!='LAZY':raise
     failures.append(dict(scope='arm',arms=['LAZY'],seed=seed,M=M,reason=repr(exc)))
   del args,graphs,truth
  done.append(seed)
 record=dict(protocol=OP_PROTOCOL,arm_names=OP_NAMES,round_delta32_remainder_ns=times,telemetry=telemetry,resources=resources,work_counts=work,histograms=hist,host_enqueue_top4=host_top,compile_seconds=compile_s,elapsed_seconds=time.monotonic()-started,clock_error=S21_TELEMETRY_ERROR)
 return dict(packet=dict(kind='P3A_OPERAND',identities=OP_IDENTITIES,seeds=done,complete=len(done)==8,paired=dict(codec='lzma-json-b85',data=base64.b85encode(lzma.compress(json.dumps(record,sort_keys=True,separators=(',',':')).encode(),preset=9)).decode()),checks=dict(gates=gates,failures=failures,diagnostics='FRAG*,ACC_FREE,PAD_USEFUL,L1-L3 ignored outputs;not production nominees')))
