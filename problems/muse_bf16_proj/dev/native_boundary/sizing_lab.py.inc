def stage_name(role,name):return role
NB_CASES=tuple((b,q,w) for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for w in (2048,0))
def nb_load():
 p='<P10-native-S19>';linecache.cache[p]=(len(NB_BASE),None,NB_BASE.splitlines(True),p);m=types.ModuleType('nb_s19');m.__file__=p;exec(compile(NB_BASE,p,'exec'),m.__dict__)
 if globals().get('NB_LOCAL_INSTALL'):NB_LOCAL_INSTALL(m)
 return m.__dict__
def nb_clone(g):
 n=dict(g)
 for k,f in g.items():
  if isinstance(f,types.FunctionType) and f.__globals__ is g:
   z=types.FunctionType(f.__code__,n,f.__name__,f.__defaults__,f.__closure__);z.__kwdefaults__=f.__kwdefaults__;n[k]=z
 return n
def nb_time(graphs,seed_index,rounds=16):
 names=list(graphs);v={n:[] for n in names};events=[]
 for block in range(rounds//2):
  shift=(seed_index+block)%len(names);order=names[shift:]+names[:shift]
  if (seed_index+block)%2:order=order[::-1]
  for order in (order,order[::-1]):
   for n in order:
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();a.record();graphs[n][0].replay();b.record();events.append((n,a,b))
   torch.cuda.synchronize()
 for n,a,b in events:v[n].append(round(a.elapsed_time(b)*125000))
 return v

def nb_weight(role,t,st,m,io):
 size=lambda x:x.numel()*x.element_size()
 if role=='gate':return dict(bytes=size(st['gate_rows']),encoding='BF16',splits=1)
 w=st[role];encoding='BF16';wb=size(w)
 if role in ('qkv_gate','qkv3') and (t,w.shape[0]) in m['SGQ_SELECTED']:
  # SGQ bank is shared by split schedules; inspect the selected packed plan.
  p=st['sgq']['plans'][t,w.shape[0],m['SGQ_SELECTED'][t,w.shape[0]]];return dict(bytes=size(p['wp'])+size(p['esc']),encoding='SGQ12',splits=io[role][1][1])
 if role=='pool_router' and t in m['P10_POOL_TOKENS']:
  p=st['p10_pool_raw'];wb=size(p['wp'])+size(p['esc']);encoding='NF12 fused pool'
 elif (t,role) in m['NF_PORT_SELECTED']:
  p=st['nf_port']['plans'].get((t,role))
  if p is None:
   assert torch.cuda.get_device_capability()!=(9,0),'missing native coded bank'
   return dict(bytes=wb,encoding='LOCAL arithmetic stand-in; not H200 bank',splits=io[role][1][1])
  if 'wp' in p:wb=size(p['wp'])+size(p['esc']);encoding='NF12'
  elif 'w' in p:wb=size(p['w']);encoding='NF raw bits'
 elif (t,role) in m['CD_ROWS']:
  p=st['coded_dense'][role];wb=sum(size(p[k]) for k in ('coded','scales','eoff','esc'));encoding='CD lossless'
 return dict(bytes=wb,encoding=encoding,splits=io[role][1][1])

def nb_profile(graph,roles,dag,act):
 act.profile_begin();ev=[]
 for _ in range(2):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();a.record();graph.replay();b.record();ev.append((a,b))
 torch.cuda.synchronize();records=act.profile_end();by={}
 for name,r in records:
  if r[4] in roles:by.setdefault(r[4],[]).append((name,r))
 assert set(by)==set(roles) and all(len(v)==2 for v in by.values()),('activity records',len(by),len(roles))
 for v in by.values():v.sort(key=lambda x:x[1][0])
 out=[]
 for i in range(2):
  origin=min(v[i][1][0] for v in by.values());trace=[]
  for vals in by.values():
   name,r=vals[i];trace.append(dict(node=r[4],role=roles[r[4]],name=name,stream=r[13],start_ns=r[0]-origin,end_ns=r[1]-origin))
  a=chain_account(trace,dag);res={};critical={}
  for n in trace:res[n['role']]=res.get(n['role'],0)+n['end_ns']-n['start_ns']
  for n in a['path']:
   role=roles[n['node']];critical[role]=critical.get(role,0)+round(n['critical_us']*1000)
  scope={'qkv_gate','qkv3','gate','o','pool_router','unpooler','unpooler_slice','b1','b2','route'}
  res={k:v for k,v in res.items() if k in scope};critical={k:v for k,v in critical.items() if k in scope}
  out.append(dict(wall_ns=round(a['wall_us']*1000),residence_ns=res,critical_ns=critical,event_ticks=round(ev[i][0].elapsed_time(ev[i][1])*125000)))
 return out

def lab(deadline,weights,state=None):
 import contextlib,io,traceback
 start=time.monotonic();m=nb_load();task={}
 for s in NB_TASK:exec(compile(s,'<published layer TASK>','exec'),task)
 if torch.cuda.get_device_capability()!=(9,0):m['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None;task['_fa3_scheduler_metadata']=lambda *a,**kw:None
 weights=task['generate_weights'](21000311,router_seed=21000313);log=io.StringIO();t0=time.monotonic()
 with contextlib.redirect_stdout(log):st=m['prepare'](weights)
 prepare_s=time.monotonic()-t0;act=t15_activity();rows=[];failure=None;checks=0;seeds=[21000319+104729*i for i in range(8)]
 try:
  for b,q,w in NB_CASES:
   if time.monotonic()>deadline-15:raise TimeoutError(str((b,q,w)))
   make=lambda seed:m['_ap_generate_input'](batch=b,q_len=q,ctx=80000,layer='sliding' if w else 'full',spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
   data=make(seeds[0]);graphs={n:ab_graph(lambda:m['kernel_fn'](data,st)) for n in ('A','AA')};ns=nb_clone(m)
   # Record only existing graph metadata; no inserted event or device work.
   g,out,roles,ios,dag=t15_capture(ns['kernel_fn'],st,data,act,ns)
   weight={r:nb_weight(r,b*q,st,m,ios) for r in ios if r in ('qkv_gate','qkv3','gate','o','pool_router','unpooler','unpooler_slice')}
   values=[];profiles=[]
   for si,seed in enumerate(seeds):
    if seed!=seeds[0]:ab_copy(data,make(seed))
    for gg,oo in graphs.values():
     gg.replay();torch.cuda.synchronize();ok,msg=task['check_implementation'](data,task['collect'](data,oo),weights);checks+=1;assert ok,(b,q,w,seed,msg)
    values.append(nb_time(graphs,si));profiles.append(nb_profile(g,roles,dag,act))
   rows.append(dict(case=[b,q,w],weights=weight,times=values,profiles=profiles,nodes=len(roles)))
   graphs=g=data=None;gc.collect();torch.cuda.empty_cache()
 except Exception as e:failure=repr(e)+' | '+' | '.join(traceback.format_exc().splitlines()[-7:])
 raw=dict(rows=rows,seeds=seeds,rounds=16,event_unit_us=.008,prepare_s=prepare_s,checks=checks,complete=len(rows)==12 and failure is None,failure=failure,elapsed_s=time.monotonic()-start)
 if globals().get('NB_LOCAL_ARCHIVE'):NB_LOCAL_ARCHIVE(raw)
 blob=nb_encode(raw);assert nb_decode(blob)==raw
 return ['NATIVE_SIZE_XZ85 '+blob,'END complete='+str(raw['complete'])]
