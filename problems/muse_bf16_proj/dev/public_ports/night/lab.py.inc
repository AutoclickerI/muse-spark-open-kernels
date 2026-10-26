def ng_time(graphs,seed_index):
 names=list(graphs);times={n:[] for n in names};events=[]
 # Complete cyclic positions, then the exact reverse; all2N rounds retained.
 for shift0 in range(len(names)):
  shift=(seed_index+shift0)%len(names);order=names[shift:]+names[:shift]
  if seed_index%2:order=order[::-1]
  for order in (order,order[::-1]):
   for name in order:
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();a.record();graphs[name][0].replay();b.record();events.append((name,a,b))
   torch.cuda.synchronize()
 for name,a,b in events:times[name].append(round(a.elapsed_time(b)*125000))
 return times

def lab(deadline,weights,state=None):
 import contextlib,io,traceback
 start=time.monotonic();m=pp_module(PP_SOURCE);task={};check={}
 for src in PP_TASK:exec(compile(src,'<published layer TASK>','exec'),task)
 exec(compile(PP_CHECK,'<published projection TASK>','exec'),check)
 if torch.cuda.get_device_capability()!=(9,0):m['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None;task['_fa3_scheduler_metadata']=lambda *a,**kw:None
 weights=task['generate_weights'](33000311,router_seed=33000313);t0=time.monotonic();log=io.StringIO()
 with contextlib.redirect_stdout(log):st=m['prepare'](weights)
 base_prepare=time.monotonic()-t0
 requested={(t,r,p) for t,r,p in NG_PARENT_POINTS};requested|={(t,'pool_router','P') for t,arm in NG_PARENT_POOL.items() if arm in m['PP_POOL_TMA']}
 with contextlib.redirect_stdout(log):bank=m['pp_prepare'](m,weights,st,requested);ng=m['ng_prepare'](m,st)
 pool=None
 if any(a in m['PP_POOL_NF'] for a in NG_PARENT_POOL.values()):
  with contextlib.redirect_stdout(log):pool=m['pp_pool_prepare'](m,st)
 if any(m['PP_POOL_TMA'].get(a,(0,))[0]==128 for a in NG_PARENT_POOL.values()):
  w=st['pool_router'];bank['tiled']['pool_router',128]=w.reshape(72,64,64,128).permute(0,2,1,3).contiguous().view(-1,128)
 old_project=m['_best7_project'];old_gate=m['_gate_split_apply'];current='A';current_t=0
 def project(x,role,ss,plan,out_dtype=torch.bfloat16):
  key=(x.shape[0],role);active=current in ('FULL','CODED_P32') or current.startswith('NO_')
  if active and key in NG_KEYS and key not in m['NG_GROUPS'].get(current[3:] if current.startswith('NO_') else '',()):return m['ng_project'](x,role,ng,pool32_coded=current=='CODED_P32')
  if current!='INC':
   if role=='pool_router' and x.shape[0] in NG_PARENT_POOL:return m['pp_pool_project'](x,bank,pool,NG_PARENT_POOL[x.shape[0]])
   provider=next((p for t,r,p in NG_PARENT_POINTS if (t,r)==key),None)
   if provider:return m['pp_project'](x,role,bank,provider)
  return old_project(x,role,ss,plan,out_dtype)
 def gate(x,y,ss):
  provider=next((p for t,r,p in NG_PARENT_POINTS if (t,r)==(x.shape[0],'gate')),None) if current!='INC' else None
  return m['pp_gate'](x,y,bank,provider) if provider else old_gate(x,y,ss)
 m['_best7_project']=project;m['_gate_split_apply']=gate
 rows=[];checks=0;strict=0;fatal=None;warm={};attrs={}
 try:
  # Actual projection/router TASK on every new point, random and one-column.
  for t,role in NG_KEYS:
   for coded in ((False,True) if (t,role)==(32,'pool_router') else (False,)):
    w=ng['roles'][role];n,k=w.shape;x=torch.empty((t,k),device=w.device,dtype=torch.bfloat16);t0=time.monotonic();m['ng_project'](x.zero_(),role,ng,pool32_coded=coded);torch.cuda.synchronize();warm[str((t,role))]=time.monotonic()-t0
    if (t,role) in ng['plans'] and ((t,role)!=(32,'pool_router') or coded):
     p=ng['plans'][t,role];out=torch.empty((t,n),device=x.device,dtype=torch.float32);args=(p['idx'],p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),t,n,k,*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream().cuda_stream);attrs[str((t,role))]=[p['fn'](*args,q) for q in (2,1)]
    for seed in NG_SEEDS:
     torch.manual_seed(seed);x.copy_(torch.randn_like(x))
     for onehot in (False,True):
      if onehot:x.zero_();x[:,seed%k]=16
      out=m['ng_project'](x,role,ng,pool32_coded=coded)[0].sum(0)
      if role=='pool_router':
       for layer,ans in (('pooler',out[:,:4096].bfloat16()),('router',out[:,4096:4352]+out[:,4352:])):
        data=dict(x=x,config=dict(layer=layer,m=t,n=4096 if layer=='pooler' else 256,k=k));ok,msg=check['check_implementation'](data,ans,weights);strict+=1;assert ok,(t,role,seed,onehot,layer,msg)
      else:
       data=dict(x=x,config=dict(layer='qkv_gate',m=t,n=n,k=k));ok,msg=check['check_implementation'](data,out.bfloat16(),dict(qkv_gate=w));strict+=1;assert ok,(t,role,seed,onehot,msg)
  for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
   current_t=b*q;groups=[g for g,keys in m['NG_GROUPS'].items() if any(t==current_t for t,r in keys)];names=['INC','A','AA','FULL']+(['CODED_P32'] if current_t==32 else [])+['NO_'+g for g in groups]
   for window in (2048,0):
    if time.monotonic()>deadline-20:raise TimeoutError(str((b,q,window)))
    make=lambda seed:m['_ap_generate_input'](batch=b,q_len=q,ctx=80000,layer='sliding' if window else 'full',spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
    data=make(NG_SEEDS[0]);graphs={};values=[]
    for name in names:current=name;graphs[name]=pp_capture(lambda:m['kernel_fn'](data,st))
    for si,seed in enumerate(NG_SEEDS):
     if time.monotonic()>deadline-10:raise TimeoutError(str((b,q,window,si)))
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
      for name,(graph,out) in graphs.items():
       graph.replay();torch.cuda.synchronize();ok,msg=task['check_implementation'](data,task['collect'](data,out),weights);checks+=1;assert ok,(b,q,window,seed,name,msg)
     finally:task['ref_front'],task['ref_back']=ff,bf
     values.append(ng_time(graphs,si))
    rows.append(dict(case=[b,q,window],times=values,rounds=2*len(names)));graphs=data=None;gc.collect();torch.cuda.empty_cache()
 except Exception as e:fatal=repr(e)+' | '+' | '.join(traceback.format_exc().splitlines()[-8:])
 raw=dict(rows=rows,seeds=NG_SEEDS,checks=checks,strict_checks=strict,fatal=fatal,complete=len(rows)==12 and fatal is None,parent=NG_PARENT_ID,parent_points=NG_PARENT_POINTS,parent_pool=NG_PARENT_POOL,base_prepare_s=base_prepare,port_prepare_s=bank['prepare_s'],parent_pool_prepare_s=pool['prepare_s'] if pool else 0,new_prepare_s=ng['prepare_s'],new_prepare_times=ng['times'],bank_bytes={str(k):p['wp'].numel()*p['wp'].element_size()+p['esc'].numel()*p['esc'].element_size() for k,p in ng['plans'].items()},tma_bank_bytes=ng['tma_w'].numel()*ng['tma_w'].element_size(),new_warm_s=warm,attrs=attrs,elapsed_s=time.monotonic()-start,limits='Native whole S21+selected3347492 portfolio, all cyclic/reverse2N rounds retained, fresh seeds. Leaveouts change one T/site group only. Untouched leaveout shapes are identical FULL graphs. Selection evidence, P4 held-out G0 still required.')
 if globals().get('PP_LOCAL_ARCHIVE'):PP_LOCAL_ARCHIVE(raw)
 blob=nb_time_encode(raw);assert nb_time_decode(blob)==json.loads(json.dumps(raw))
 return ['NIGHT_GAP_XZ85 '+blob,'END complete='+str(raw['complete'])]
