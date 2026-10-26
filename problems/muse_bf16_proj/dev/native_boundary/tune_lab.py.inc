def lab(deadline,weights,state=None):
 import contextlib,io,traceback
 start=time.monotonic();m=nb_load();task={};check={}
 for s in NB_TASK:exec(compile(s,'<published layer TASK>','exec'),task)
 exec(compile(NB_CHECK,'<published projection TASK>','exec'),check)
 if torch.cuda.get_device_capability()!=(9,0):m['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None;task['_fa3_scheduler_metadata']=lambda *a,**kw:None
 weights=task['generate_weights'](23000311,router_seed=23000313);log=io.StringIO();t0=time.monotonic()
 with contextlib.redirect_stdout(log):st=m['prepare'](weights)
 prepare_s=time.monotonic()-t0;rows=[];failure=None;checks=0;projection_checks=0;seeds=[23000319+104729*i for i in range(8)];compile_start=time.monotonic()
 def invoke(arm,data):
  m['NB_POINTS']=NB_POINTS_ARMS.get(arm,{})
  return m['kernel_fn'](data,st)
 try:
  for t in (128,256):
   b=t//8;names=['A','AA']+[a for a,pts in NB_POINTS_ARMS.items() if any(k[0]==t for k in pts)]
   for w in (2048,0):
    make=lambda seed:m['_ap_generate_input'](batch=b,q_len=8,ctx=80000,layer='sliding' if w else 'full',spread='agentic',dist='soft',page=1,seed=seed,device='cuda')
    data=make(seeds[0]);graphs={a:ab_graph(lambda a=a:invoke(a,data)) for a in names};values=[]
    for si,seed in enumerate(seeds):
     if time.monotonic()>deadline-10:raise TimeoutError(str((t,w,si)))
     if si:ab_copy(data,make(seed))
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
       g.replay();torch.cuda.synchronize();ok,msg=task['check_implementation'](data,task['collect'](data,out),weights);checks+=1;assert ok,(t,w,seed,a,msg)
     finally:task['ref_front'],task['ref_back']=ff,bf
     values.append(nb_time(graphs,si))
    rows.append(dict(case=[b,8,w],times=values,arms=names));graphs=data=None;gc.collect();torch.cuda.empty_cache()
   # Independent random and one-column projection/router checks, all schedules.
   for arm,pts in NB_POINTS_ARMS.items():
    for (mt,role),(cfg,s) in pts.items():
     if mt!=t:continue
     ww=st[role];x=torch.randn((t,ww.shape[1]),device='cuda',dtype=torch.bfloat16)
     for seed in seeds:
      torch.manual_seed(seed);x.copy_(torch.randn_like(x))
      for onehot in (False,True):
       if onehot:x.zero_();x[:,seed%x.shape[1]]=16
       pair=m['_vdsk_project'](x,ww,('vdsk',cfg,s),chain=role!='qkv_gate');out=pair[0].sum(0);data=dict(x=x,config=dict(layer='qkv_gate',m=t,n=ww.shape[0],k=ww.shape[1]));ok,msg=check['check_implementation'](data,out.bfloat16(),dict(qkv_gate=ww));projection_checks+=1;assert ok,(arm,onehot,msg)
       if role=='pool_router':
        router=out[:,4096:4352]+out[:,4352:];ok,msg=check['check_implementation'](dict(x=x,config=dict(layer='router',m=t,n=256,k=8192)),router,weights);projection_checks+=1;assert ok,(arm,'router',onehot,msg)
 except Exception as e:failure=repr(e)+' | '+' | '.join(traceback.format_exc().splitlines()[-8:])
 raw=dict(rows=rows,seeds=seeds,rounds=16,event_unit_us=.008,prepare_s=prepare_s,compile_and_measure_s=time.monotonic()-compile_start,checks=checks,projection_checks=projection_checks,complete=len(rows)==4 and failure is None,failure=failure,elapsed_s=time.monotonic()-start)
 if globals().get('NB_LOCAL_ARCHIVE'):NB_LOCAL_ARCHIVE(raw)
 blob=nb_time_encode(raw);assert nb_time_decode(blob)==raw
 return ['NATIVE_TUNE_XZ85 '+blob,'END complete='+str(raw['complete'])]
