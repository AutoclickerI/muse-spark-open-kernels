def lab(deadline,weights,state=None):
 import contextlib,io,traceback
 begin=time.monotonic();m=pp_module(PP_SOURCE);task={};check={}
 for s in PP_TASK:exec(compile(s,'<published layer TASK>','exec'),task)
 exec(compile(PP_CHECK,'<published projection TASK>','exec'),check)
 if torch.cuda.get_device_capability()!=(9,0):m['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None;task['_fa3_scheduler_metadata']=lambda *a,**kw:None
 weights=task['generate_weights'](32000311,router_seed=32000313);t0=time.monotonic();log=io.StringIO()
 with contextlib.redirect_stdout(log):st=m['prepare'](weights)
 base_prepare=time.monotonic()-t0
 requested={(t,r,p) for points in PW_POINTS.values() for t,r,p in points}
 requested|={(t,'pool_router','P') for t in (16,32)}
 with contextlib.redirect_stdout(log):bank=m['pp_prepare'](m,weights,st,requested);pool=m['pp_pool_prepare'](m,st)
 t0=time.monotonic();w=st['pool_router'];bank['tiled']['pool_router',128]=w.reshape(72,64,64,128).permute(0,2,1,3).contiguous().view(-1,128);torch.cuda.synchronize();extra_tile_s=time.monotonic()-t0
 old_project=m['_best7_project'];old_gate=m['_gate_split_apply'];current='A'
 def project(x,role,ss,plan,out_dtype=torch.bfloat16):
  if current in PW_POOL and role=='pool_router' and x.shape[0] in (16,32):return m['pp_pool_project'](x,bank,pool,current)
  provider=next((p for t,r,p in PW_POINTS.get(current,()) if t==x.shape[0] and r==role),None)
  return m['pp_project'](x,role,bank,provider) if provider else old_project(x,role,ss,plan,out_dtype)
 def gate(x,y,ss):
  provider=next((p for t,r,p in PW_POINTS.get(current,()) if t==x.shape[0] and r=='gate'),None)
  return m['pp_gate'](x,y,bank,provider) if provider else old_gate(x,y,ss)
 m['_best7_project']=project;m['_gate_split_apply']=gate
 seeds=PW_SEEDS;rows=[];checks=0;projection_checks=0;failures=[];fatal=None;pool_status={};pool_attrs={};warm_s={}
 # Warm every new code path and run strict original TASK checks, before captures.
 for arm in PW_POOL:
  try:
   t0=time.monotonic()
   for t in (16,32):
    x=torch.randn((t,8192),device='cuda',dtype=torch.bfloat16)
    for seed in seeds:
     torch.manual_seed(seed);x.copy_(torch.randn_like(x));original=x.clone()
     for onehot in (False,True):
      if onehot:x.zero_();x[:,seed%8192]=16
      out=m['pp_pool_project'](x,bank,pool,arm)[0].sum(0);torch.cuda.synchronize()
      for layer,ans in (('pooler',out[:,:4096].bfloat16()),('router',out[:,4096:4352]+out[:,4352:])):
       data=dict(x=x,config=dict(layer=layer,m=t,n=4096 if layer=='pooler' else 256,k=8192));ok,msg=check['check_implementation'](data,ans,weights);projection_checks+=1;assert ok,(arm,t,seed,onehot,layer,msg)
     x.copy_(original)
   warm_s[arm]=time.monotonic()-t0;pool_status[arm]='ready'
   if arm in m['PP_POOL_NF']:
    wk,stages,early=m['PP_POOL_NF'][arm];p=pool['packs'][wk];x=torch.empty((32,8192),device='cuda',dtype=torch.bfloat16);out=torch.empty((32,4608),device='cuda')
    args=(pool['indices'][arm,32],p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),32,4608,8192,*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream().cuda_stream)
    pool_attrs[arm]=[pool['mod'].v18s_launch(*args,q) for q in (2,1)]
  except Exception as e:
   torch.cuda.synchronize();pool_status[arm]='unmeasured:'+repr(e);failures.append(dict(arm=arm,reason=repr(e)))
 try:
  for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
   t=b*q;points=[a for a,pts in PW_POINTS.items() if any(mt==t for mt,r,p in pts)];arms=['A','AA']+points+([a for a in PW_POOL if pool_status[a]=='ready'] if t in (16,32) else [])
   for w in (2048,0):
    if time.monotonic()>deadline-20:raise TimeoutError(str((b,q,w)))
    make=lambda seed:m['_ap_generate_input'](batch=b,q_len=q,ctx=80000,layer='sliding' if w else 'full',spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
    data=make(seeds[0]);graphs={};values=[]
    for arm in arms:
     current=arm;graphs[arm]=pp_capture(lambda:m['kernel_fn'](data,st))
    for si,seed in enumerate(seeds):
     if time.monotonic()>deadline-10:raise TimeoutError(str((b,q,w,si)))
     if si:pp_copy(data,make(seed))
     ff,bf=task['ref_front'],task['ref_back'];fc={};bc={}
     def front(dd,ww,query_dtype=torch.bfloat16):
      if query_dtype not in fc:fc[query_dtype]=ff(dd,ww,query_dtype)
      return fc[query_dtype]
     def back(f,ww,ids):
      key=id(f),tuple(ids.reshape(-1).tolist())
      if key not in bc:bc[key]=bf(f,ww,ids)
      return bc[key]
     task['ref_front'],task['ref_back']=front,back
     try:
      for arm,(g,out) in graphs.items():
       g.replay();torch.cuda.synchronize();ok,msg=task['check_implementation'](data,task['collect'](data,out),weights);checks+=1;assert ok,(b,q,w,seed,arm,msg)
     finally:task['ref_front'],task['ref_back']=ff,bf
     values.append(pw_time(graphs,si))
    rows.append(dict(case=[b,q,w],times=values));graphs=data=None;gc.collect();torch.cuda.empty_cache()
 except Exception as e:fatal=repr(e)+' | '+' | '.join(traceback.format_exc().splitlines()[-8:])
 result=dict(rows=rows,seeds=seeds,rounds=8,event_unit_us=.008,checks=checks,projection_checks=projection_checks,failures=failures,fatal=fatal,complete=len(rows)==12 and fatal is None,pool_status=pool_status,pool_attrs=pool_attrs,base_prepare_s=base_prepare,added_prepare_s=bank['prepare_s']+pool['prepare_s']+extra_tile_s,prepare_times=dict(port=bank['times'],pool=pool['times'],extra_tile=extra_tile_s),warm_and_check_s=warm_s,elapsed_s=time.monotonic()-begin,limits='Selection-stage whole S21 native graphs, real PDL and native boundary. Every8 symmetric round retained; arithmetic means; original TASK only. New points need P4 held-out G0. Per-point unmodified loads use same paired A; no fabricated changes at those loads.')
 if globals().get('PP_LOCAL_ARCHIVE'):PP_LOCAL_ARCHIVE(result)
 blob=nb_time_encode(result);assert nb_time_decode(blob)==json.loads(json.dumps(result))
 return ['PUBLIC_WHOLE_XZ85 '+blob,'END complete='+str(result['complete'])]
