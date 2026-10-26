def whole_load():
 path='<P10 SS whole>';linecache.cache[path]=(len(WHOLE_SOURCE),None,WHOLE_SOURCE.splitlines(True),path);m=types.ModuleType('p10_ss_whole');m.__file__=path;exec(compile(WHOLE_SOURCE,path,'exec'),m.__dict__)
 if globals().get('WHOLE_LOCAL_INSTALL'):WHOLE_LOCAL_INSTALL(m)
 return m.__dict__

def lab(deadline,weights,state=None):
 import contextlib,io,gc,traceback
 start=time.monotonic();m=whole_load();task={}
 for i,s in enumerate(WHOLE_TASK):exec(compile(s,'<published layer TASK '+str(i)+'>','exec'),task)
 weights=task['generate_weights'](17000311,router_seed=17000313);t0=time.monotonic();log=io.StringIO()
 with contextlib.redirect_stdout(log):st=m['prepare'](weights)
 torch.cuda.synchronize();prepare_s=time.monotonic()-t0;arms=['A','AA']+list(m['SS_PORTFOLIOS']);seeds=[17000319+104729*i for i in range(16)];rows=[];failure=None;checks=0
 if torch.cuda.get_device_capability()!=(9,0):task['_fa3_scheduler_metadata']=lambda *a,**kw:None
 def invoke(arm,data):
  m['SS_ACTIVE']=arm not in ('A','AA')
  if m['SS_ACTIVE']:m['SS_SELECTED']=m['SS_PORTFOLIOS'][arm]
  return m['kernel_fn'](data,st)
 try:
  for b,q,w in AB_CASES:
   if time.monotonic()>deadline-15:raise TimeoutError(str((b,q,w)))
   make=lambda seed:task['generate_input'](batch=b,q_len=q,ctx=80000,layer='sliding' if w else 'full',spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
   data=make(seeds[0]);graphs={arm:ab_graph(lambda arm=arm:invoke(arm,data)) for arm in arms};values={arm:[] for arm in arms}
   for seed in seeds:
    if time.monotonic()>deadline-8:raise TimeoutError(str((b,q,w,seed)))
    if seed!=seeds[0]:ab_copy(data,make(seed))
    for g,out in graphs.values():g.replay()
    torch.cuda.synchronize();front0,back0=task['ref_front'],task['ref_back'];fc={};bc={}
    def front(d,wt,query_dtype=torch.bfloat16):
     if query_dtype not in fc:fc[query_dtype]=front0(d,wt,query_dtype)
     return fc[query_dtype]
    def back(f,wt,ids):
     key=id(f),tuple(ids.reshape(-1).tolist())
     if key not in bc:bc[key]=back0(f,wt,ids)
     return bc[key]
    task['ref_front'],task['ref_back']=front,back
    try:
     for arm,(g,out) in graphs.items():
      ok,msg=task['check_implementation'](data,task['collect'](data,out),weights);checks+=1;assert ok,(b,q,w,seed,arm,msg)
    finally:task['ref_front'],task['ref_back']=front0,back0
    ts=ab_time(graphs,12 if b*q<=8 else 8)
    for arm,v in ts.items():values[arm].append(round(statistics.median(v),4))
   rows.append(dict(case=[b,q,w],values=values));graphs=data=None;gc.collect();torch.cuda.empty_cache()
 except Exception as e:failure=repr(e)+' | '+' | '.join(traceback.format_exc().splitlines()[-6:])
 result=dict(rows=rows,seeds=seeds,arms=arms,complete=len(rows)==12 and failure is None and checks==12*16*len(arms),failure=failure,checks=checks,prepare_s=prepare_s,added_prepare_s=st['ss']['prepare_s'],resources=st['ss']['resources'],elapsed_s=time.monotonic()-start,weight_seed=17000311,router_seed=17000313)
 if globals().get('WHOLE_LOCAL_ARCHIVE'):WHOLE_LOCAL_ARCHIVE(result)
 blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode();assert len(blob)<8000,len(blob)
 return ['META P10 SS on frozen S19;16disjoint heldoutseeds all12shapes publicTASK originalrouter;all points and point winners judged by same G0','WHOLE_XZ85 '+blob,'END complete='+str(result['complete'])]
