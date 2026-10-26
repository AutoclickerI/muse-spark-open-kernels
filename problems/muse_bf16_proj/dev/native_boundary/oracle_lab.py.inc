NB_CASES=tuple((b,q,w) for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for w in (2048,0))
def nb_wrap(m,producer=None,consumer=None):
 n=nb_clone(m)
 if producer:
  old=n['_best7_project']
  def project(x,role,st,plan,out_dtype=torch.bfloat16):return producer(role)
  n['_best7_project']=project
  n['_gate_split_apply']=lambda *args:producer('gate')
 if consumer:
  n['_best7_boundary']=lambda branch,stream,st,plan,phase,sync=None:consumer(phase)
  n['_best7_route']=lambda *args,**kwargs:consumer('route')
 return n

def lab(deadline,weights,state=None):
 import contextlib,io,traceback
 start=time.monotonic();m=nb_load();task={}
 for s in NB_TASK:exec(compile(s,'<published layer TASK>','exec'),task)
 if torch.cuda.get_device_capability()!=(9,0):m['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None;task['_fa3_scheduler_metadata']=lambda *a,**kw:None
 weights=task['generate_weights'](22000311,router_seed=22000313);log=io.StringIO();t0=time.monotonic()
 with contextlib.redirect_stdout(log):st=m['prepare'](weights)
 prepare_s=time.monotonic()-t0;rows=[];checks=0;failure=None;seeds=[22000319+104729*i for i in range(8)]
 try:
  for b,q,w in NB_CASES:
   active=(b*q<=32)==(NB_SELECTED=='small_M');names=['A','AA']+(['P_READY','SINGLE_READY','C_READY','BOTH_READY'] if active else [])
   make=lambda seed:m['_ap_generate_input'](batch=b,q_len=q,ctx=80000,layer='sliding' if w else 'full',spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
   data=make(seeds[0]);m['kernel_fn'](data,st);torch.cuda.synchronize();tap=nb_clone(m);ready={};singles={}
   def wrap(name,label):
    old=tap[name]
    def call(*a,**kw):
     out=old(*a,**kw);ready[label(a)]=out;return out
    tap[name]=call
   wrap('_best7_project',lambda a:a[1]);wrap('_gate_split_apply',lambda a:'gate');wrap('_best7_boundary',lambda a:a[4]);wrap('_best7_route',lambda a:'route')
   refgraph,refout=ab_graph(lambda:tap['kernel_fn'](data,st))
   roles=set(ready)-{'b1','b2','route'}
   for role in roles:
    pair=ready[role]
    if role!='gate':singles[role]=(pair[0].sum(0,keepdim=True),1)
   def fill_single():
    for role,(p,s) in singles.items():p.copy_(ready[role][0].sum(0,keepdim=True))
   ns={a:m for a in ('A','AA')}
   if active:
    ns['P_READY']=nb_wrap(m,producer=lambda r:ready[r]);ns['SINGLE_READY']=nb_wrap(m,producer=lambda r:ready[r] if r=='gate' else singles[r]);ns['C_READY']=nb_wrap(m,consumer=lambda r:ready[r]);ns['BOTH_READY']=nb_wrap(m,producer=lambda r:ready[r],consumer=lambda r:ready[r])
   graphs={a:ab_graph(lambda a=a:ns[a]['kernel_fn'](data,st)) for a in names};values=[]
   for si,seed in enumerate(seeds):
    if time.monotonic()>deadline-10:raise TimeoutError(str((b,q,w,si)))
    if si:ab_copy(data,make(seed))
    refgraph.replay();torch.cuda.synchronize();fill_single();torch.cuda.synchronize()
    # All ready buffers are from this fresh seed's actual S19 execution. They
    # are lab-only deletion bounds and never production entry points.
    ff,bf=task['ref_front'],task['ref_back'];fc={};bc={}
    def front(d,wt,query_dtype=torch.bfloat16):
     if query_dtype not in fc:fc[query_dtype]=ff(d,wt,query_dtype)
     return fc[query_dtype]
    def back(f,wt,ids):
     key=id(f),tuple(ids.reshape(-1).tolist())
     if key not in bc:bc[key]=bf(f,wt,ids)
     return bc[key]
    task['ref_front'],task['ref_back']=front,back
    try:
     for a,(g,out) in graphs.items():
      g.replay();torch.cuda.synchronize();ok,msg=task['check_implementation'](data,task['collect'](data,out),weights);checks+=1;assert ok,(b,q,w,seed,a,msg)
    finally:task['ref_front'],task['ref_back']=ff,bf
    values.append(nb_time(graphs,si))
   rows.append(dict(case=[b,q,w],active=active,times=values,producer_roles=sorted(roles),original_splits={r:ready[r][1] for r in roles if r!='gate'}));graphs=refgraph=data=None;gc.collect();torch.cuda.empty_cache()
 except Exception as e:failure=repr(e)+' | '+' | '.join(traceback.format_exc().splitlines()[-8:])
 raw=dict(selected=NB_SELECTED,rows=rows,seeds=seeds,rounds=16,event_unit_us=.008,prepare_s=prepare_s,checks=checks,complete=len(rows)==12 and failure is None,failure=failure,elapsed_s=time.monotonic()-start,limits='Lab-only ready buffers from same-seed S19, never candidate. No added GPU nodes in timed graphs. Untouched remaining kernels/PDL; deletions change dependency frontier and are bounds, not exclusive wait cycles.')
 if globals().get('NB_LOCAL_ARCHIVE'):NB_LOCAL_ARCHIVE(raw)
 blob=nb_encode(raw);assert nb_decode(blob)==raw
 return ['NATIVE_ORACLE_XZ85 '+blob,'END complete='+str(raw['complete'])]
