# Lab-only: production kernel_fn graphs, unchanged rest-of-layer kernels.
def pv_module(name,source):
 f='<pv-'+name+'>';linecache.cache[f]=(len(source),None,source.splitlines(True),f)
 m=types.ModuleType(name);m.__file__=f;exec(compile(source,f,'exec'),m.__dict__);return m

def pv_clone(m):
 g=m.__dict__;v=dict(g)
 for k,f in g.items():
  if isinstance(f,types.FunctionType) and f.__globals__ is g:
   h=types.FunctionType(f.__code__,v,f.__name__,f.__defaults__,f.__closure__)
   h.__kwdefaults__=f.__kwdefaults__;h.__dict__.update(f.__dict__);v[k]=h
 return v

def pv_bind(base,old,arm,cell):
 v=pv_clone(base);original=v['_best7_attention']
 def attention(pair,data,state,plan):
  c=data['config'];key=c['batch'],c['q_len'],bool(c['window'])
  if key!=cell:return original(pair,data,state,plan)
  if arm=='S':
   st=dict(state,vq8_rows=old._VQ8_ROWS,partial_port_rows=old._PP_ROWS,cluster_rows=old._CLUSTER_ROWS,
           r2_rows=old._R2_SELECTION,wr_rows=old._WINDOW_SELECTION)
   return old._best7_attention(pair,data,st,plan)
  if key in ((1,1,True),(1,1,False)):
   return NATIVE['_best7_attention'](pair,data,state,plan)
  p=pair
  if arm=='D' and key not in ((32,1,True),(1,8,True)):
   q=MODULES['pack'].raw(pair,data);p=q.reshape(1,c['batch']*c['q_len'],2304),1
  return MODULES[('op_' if key==(1,8,True) else 'lee_')+arm].attention_pair(p,data).reshape(-1,1024)
 v['_best7_attention']=attention
 return v

def pv_orders(names,rounds,seed):
 for block in range(rounds//2):
  k=(block+seed)%len(names);order=names[k:]+names[:k]
  if (block+seed)&1:order=order[::-1]
  yield order;yield order[::-1]

def pv_times(graphs,rounds,seed):
 out={n:[] for n in graphs}
 for order in pv_orders(list(graphs),rounds,seed):
  events=[]
  for n in order:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   lab_cold_l2();a.record();graphs[n][0].replay();b.record();events.append((n,a,b))
  torch.cuda.synchronize()
  for n,a,b in events:out[n].append(round(a.elapsed_time(b)*1e6))
 return out

def pv_profile(graph):
 # Diagnostic only, never used to select timing samples or correctness criteria.
 try:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:graph.replay()
  torch.cuda.synchronize();events=sorted([e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA],key=lambda e:e.time_range.start)
  keys=[i for i,e in enumerate(events) if any(x in e.name for x in ('attnp<','attnd<','_fused_attn','_pack_kernel','sliding_main','attn_main','vq8_main','vq8_combine','row_reduce','flash_fwd'))]
  keep=sorted(set(j for i in keys for j in range(max(0,i-1),min(len(events),i+2))))
  return [[e.name,round(e.time_range.start*1000),round(e.time_range.elapsed_us()*1000)] for i,e in enumerate(events) if i in keep]
 except Exception as e:return {'error':type(e).__name__+':'+str(e)[:120]}

def lab(deadline,weights,state):
 global MODULES,NATIVE
 t0=time.monotonic();failures=[];checks={};rounds={};profiles={}
 base=pv_module('control',BUNDLE['base']);old=pv_module('s19',BUNDLE['s19'])
 MODULES={n:pv_module(n,s) for n,s in BUNDLE['modules'].items()}
 # Native vector arithmetic uses the original dispatch/launch geometry.
 NATIVE=pv_clone(base);NATIVE.update(_s16_global_C128_cs_CUDA=BUNDLE['native_cs'],PARTIAL_CUDA=BUNDLE['native_partial'],_s16_global_C128_cs__MOD={},_PP_MOD={})
 if hook:=globals().get('PV_LOCAL_INSTALL'):hook(base,old,MODULES,NATIVE)
 checker={};exec(compile(CHECKER,'<unchanged-TASK>','exec'),checker)
 if torch.cuda.get_device_capability()!=(9,0):checker['_fa3_scheduler_metadata']=lambda *a,**k:None
 # Candidate prepare is identical. Arm builds overlap it and never touch data.
 from concurrent.futures import ThreadPoolExecutor
 def build_new():
  for m in MODULES.values():
   if hasattr(m,'setup'):m.setup('cuda')
  NATIVE['_s16_global_C128_cs_module']();NATIVE['partial_port_ext'](False)
 with ThreadPoolExecutor(max_workers=2) as pool:
  fut=pool.submit(build_new)
  weights=checker['generate_weights'](WEIGHT_SEED,device='cuda',router_seed=ROUTER_SEED)
  ts=time.monotonic();state=base.prepare(weights);prepare_s=time.monotonic()-ts;fut.result()
 # S19 owns its native caches; load_inline reuses identical compiled objects by
 # source hash. Never borrow a cache whose hidden dependency might have changed.
 gen=base._ap_generate_input
 if torch.cuda.get_device_capability()!=(9,0):
  # The local stand-in only replaces Hopper scheduling metadata, as P4 screens do.
  gen=checker['generate_input']
 for key in CASES:
  b,l,w=key;cell=('d' if l==1 else 'v')+str(b)+('S' if w else 'F')
  port=(b,l,bool(w)) in PORTS;native=(b,l,bool(w)) in ((1,1,True),(1,1,False))
  arms=['A','A2']+(['S','D','V','F'] if port else ['V'] if native else [])
  make=lambda seed:gen(batch=b,q_len=l,ctx=CTX,layer='sliding' if w else 'full',spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
  data=make(SEEDS[0]);graphs={};rows=[];counts={n:0 for n in arms}
  for name in arms:
   v=base.__dict__ if name in ('A','A2') else pv_bind(base,old,name,(b,l,bool(w)))
   try:graphs[name]=ab_graph(lambda v=v:v['kernel_fn'](data,state))
   except Exception as e:failures.append([cell,name,'capture',type(e).__name__+':'+str(e)[:180]])
  for si,seed in enumerate(SEEDS):
   if time.monotonic()>deadline-8:failures.append([cell,'all','deadline']);break
   if si:ab_copy(data,make(seed))
   for graph,_ in graphs.values():graph.replay()
   torch.cuda.synchronize()
   ff,bf=checker['ref_front'],checker['ref_back'];fc={};bc={}
   def front(d,w,query_dtype=torch.bfloat16):
    if query_dtype not in fc:fc[query_dtype]=ff(d,w,query_dtype)
    return fc[query_dtype]
   def back(f,w,ids):
    k=id(f),tuple(ids.reshape(-1).tolist())
    if k not in bc:bc[k]=bf(f,w,ids)
    return bc[k]
   checker['ref_front'],checker['ref_back']=front,back
   try:
    for name,(graph,out) in graphs.items():
     graph.replay();torch.cuda.synchronize()
     ok,msg=checker['check_implementation'](data,checker['collect'](data,out),weights)
     counts[name]+=1
     if not ok:failures.append([cell,name,seed,'TASK:'+msg])
   finally:checker['ref_front'],checker['ref_back']=ff,bf
   fc=bc=None
   rows.append(pv_times(graphs,ROUNDS,si))
  rounds[cell]=rows;checks[cell]=counts
  if port and globals().get('PV_PROFILE',True):
   profiles[cell]={n:pv_profile(graphs[n][0]) for n in ('A','S','F') if n in graphs}
  graphs=data=None;gc.collect();torch.cuda.empty_cache()
 result=dict(identities=IDENTITIES,seeds=list(SEEDS),native=torch.cuda.get_device_capability()==(9,0),paired='whole production kernel_fn; load=46S+16F; cell leave-out uses A2 at the unchanged counterpart',checks=checks,complete=not failures and len(rounds)==12 and all(len(v)==len(SEEDS) for v in rounds.values()),failures=failures,round_ns=rounds,prepare_s=prepare_s,elapsed_s=time.monotonic()-t0,profiles=profiles,
  protocol=dict(rounds=ROUNDS,order='forward then exact reverse; rotated across seed/block; arithmetic mean of every round',seeds='fixed held-out, no adaptive stopping',flush='judge read-only >L2 before each replay'))
 if archive:=globals().get('PV_ARCHIVE'):archive(result)
 result=encode_rounds(result)
 # Exact compressed round vectors and kernel names; output checked before send.
 return {'packet':result,'logs':['P5b whole-graph TASK checks='+str(sum(sum(v.values()) for v in checks.values()))+' complete='+str(result['complete'])]}
