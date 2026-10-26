PP_SEEDS=tuple(31000319+104729*i for i in range(8))
PP_TIMES=(1,8,16,32,128,256)
PP_ROLES=('qkv_gate','qkv3','gate','o','pool_router','unpooler')
PP_PROVIDERS=('O','P','C','N')

def pp_copy(dst,src):
 if isinstance(dst,torch.Tensor):assert dst.shape==src.shape;dst.copy_(src)
 elif isinstance(dst,dict):
  for k in dst:pp_copy(dst[k],src[k])
 elif isinstance(dst,(tuple,list)):
  for a,b in zip(dst,src):pp_copy(a,b)

def pp_module(source):
 path='<P10 reviewed public projection ABI>';linecache.cache[path]=(len(source),None,source.splitlines(True),path);m=types.ModuleType('public_proj');m.__file__=path;exec(compile(source,path,'exec'),m.__dict__)
 if globals().get('PP_LOCAL_INSTALL'):PP_LOCAL_INSTALL(m)
 return m.__dict__

def pp_capture(fn):
 fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph(keep_graph=True)
 with torch.cuda.graph(g):out=fn()
 g.replay();torch.cuda.synchronize();return g,out

def pp_time(graphs,seed_index):
 names=list(graphs);values={n:[] for n in names};events=[]
 for block in range(2):
  shift=(seed_index+block)%len(names);order=names[shift:]+names[:shift]
  if (seed_index+block)%2:order=order[::-1]
  for order in (order,order[::-1]):
   for name in order:
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();a.record();graphs[name][0].replay();b.record();events.append((name,a,b))
   torch.cuda.synchronize()
 for name,a,b in events:values[name].append(round(a.elapsed_time(b)*125000))
 return values

def lab(deadline,weights,state=None):
 import io,contextlib,traceback
 start=time.monotonic();m=pp_module(PP_SOURCE);task={};check={}
 for s in PP_TASK:exec(compile(s,'<published layer fixture>','exec'),task)
 exec(compile(PP_CHECK,'<published projection TASK>','exec'),check)
 weights=task['generate_weights'](31000311,router_seed=31000313);log=io.StringIO();t0=time.monotonic()
 with contextlib.redirect_stdout(log):st=m['prepare'](weights)
 base_prepare=time.monotonic()-t0
 requested={(t,'unpooler_slice' if role=='unpooler' and t<=8 else role,p) for t in PP_TIMES for role in PP_ROLES for p in PP_PROVIDERS}
 with contextlib.redirect_stdout(log):bank=m['pp_prepare'](m,weights,st,requested)
 guard=load_inline(name='p10_pp_graph_'+hashlib.sha256(PP_GUARD.encode()).hexdigest()[:12],cpp_sources=PP_GUARD,functions=['graph_kernels'],extra_include_paths=['/usr/local/cuda/include'],extra_cflags=['-O2'],extra_ldflags=['-L/usr/local/cuda/lib64','-lcudart'],no_implicit_headers=True,with_pytorch_error_handling=False)
 empty=pp_capture(lambda:None);fixed=[pp_time({'A':empty,'AA':empty},i) for i in range(8)];rows=[];checks=0;router_checks=0;failures=[];fatal=None;nodes={};gate_layer_checks=0
 def materialize(pair):return pair[0].sum(0)
 def gate(data,out,ww,role):
  nonlocal checks,router_checks
  if role=='pool_router':
   data=dict(x=data['x'],config=dict(layer='pooler',m=data['x'].shape[0],n=4096,k=data['x'].shape[1]));ok,msg=check['check_implementation'](data,out[:,:4096].bfloat16(),weights);checks+=1
   if not ok:raise AssertionError('pooler '+msg)
   rd=dict(x=data['x'],config=dict(layer='router',m=data['x'].shape[0],n=256,k=data['x'].shape[1]));ok,msg=check['check_implementation'](rd,out[:,4096:4352]+out[:,4352:],weights);router_checks+=1
  else:ok,msg=check['check_implementation'](data,out.bfloat16(),dict(qkv_gate=ww));checks+=1
  if not ok:raise AssertionError(msg)
 try:
  for t in PP_TIMES:
   for logical in PP_ROLES:
    if time.monotonic()>deadline-20:raise TimeoutError(str((t,logical)))
    role='unpooler_slice' if logical=='unpooler' and t<=8 else logical;w=bank['roles'][role];n,k=w.shape;b,q=(t,1) if t in (1,16,32) else (t//8,8);plan=st['best7_plans'][b,q,2048];x=torch.randn((t,k),device='cuda',dtype=torch.bfloat16);res=torch.randn((t,8192),device='cuda');gy=torch.randn((t,1024),device='cuda',dtype=torch.bfloat16);data=dict(x=x,config=dict(layer='qkv_gate',m=t,n=n,k=k));local=dict(local_start=torch.zeros((),device='cuda',dtype=torch.int32));kind='gate_apply' if logical=='gate' and t in (1,16,32) else 'neighbour_S' if logical in ('o','pool_router','unpooler') else 'producer';inactive=logical=='gate' and t not in (1,16,32) or logical=='qkv3' and t not in (1,16,32) or logical=='qkv_gate' and t==1
    def producer(name):
     if name in ('A','AA'):
      if logical=='gate' or logical=='qkv3' and t not in (1,16,32):return torch.mm(x,w.T,out_dtype=torch.float32).unsqueeze(0),1
      return m['_best7_project'](x,role,st,plan,torch.float32 if role=='pool_router' else torch.bfloat16)
     return m['pp_project'](x,role,bank,name)
    def invoke(name):
     if kind=='gate_apply':return m['_gate_split_apply'](x,gy,st) if name in ('A','AA') else m['pp_gate'](x,gy,bank,name)
     p=producer(name)
     if logical=='pool_router':return m['_best7_route'](p,st,local,plan)
     if logical in ('o','unpooler'):return m['_best7_boundary'](p,res,st,plan,'b1' if logical=='o' else 'b2')
     return p
    names=['A','AA'];status={};g={}
    # Every arm passes the actual published projection/router checks before timing.
    # Unsupported shape/compile is explicit unmeasured evidence, never a loss.
    for name in ('A','AA',*PP_PROVIDERS):
     try:
      pair=producer(name);torch.cuda.synchronize();gate(data,materialize(pair),w,role);graph,out=pp_capture(lambda name=name:invoke(name));count=guard.graph_kernels(graph.raw_cuda_graph());assert count>0,'empty capture';g[name]=(graph,out);nodes[str((t,logical,name))]=count;status[name]='ready'
      if name not in names:names.append(name)
     except Exception as e:
      torch.cuda.synchronize();status[name]='unmeasured:'+repr(e);failures.append(dict(t=t,role=logical,arm=name,reason=repr(e)))
      if name in ('A','AA'):raise
    values=[];accepted=set(names)
    for si,seed in enumerate(PP_SEEDS):
     if time.monotonic()>deadline-10:raise TimeoutError(str((t,logical,si)))
     generator=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=generator).bfloat16());res.copy_(torch.randn(res.shape,device='cuda',generator=generator));gy.copy_(torch.randn(gy.shape,device='cuda',generator=generator).bfloat16());original=x.clone()
     for name in list(names):
      if name not in accepted:continue
      try:
       gate(data,materialize(producer(name)),w,role);x.zero_();x[:,seed%k]=16;gate(data,materialize(producer(name)),w,role);x.copy_(original)
      except Exception as e:
       torch.cuda.synchronize();x.copy_(original);accepted.remove(name);status[name]='unmeasured:TASK:'+repr(e);failures.append(dict(t=t,role=logical,arm=name,reason='TASK:'+repr(e)))
       if name in ('A','AA'):raise
     measured=pp_time({name:g[name] for name in names if name in accepted},si);values.append(measured)
    # Failed arm samples, if any, remain in the archive, but admission requires
    # all8seed pairs and all4rounds with every TASK check successful.
    rows.append(dict(t=t,role=logical,kind=kind,inactive=inactive,status=status,times=values,same_S19_NF=(t,role) in m['NF_PORT_SELECTED'],n=n,k=k));g=x=res=data=None;gc.collect();torch.cuda.empty_cache()
  # Actual published layer TASK verifies every active split-gate arm. This
  # checks the paid sigmoid/output ABI, not an invented component tolerance.
  if torch.cuda.get_device_capability()!=(9,0):m['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None;task['_fa3_scheduler_metadata']=lambda *a,**kw:None
  old_gate=m['_gate_split_apply']
  for b,q,window in ((1,1,2048),(1,1,0),(16,1,2048),(32,1,0)):
   t=b*q;row=next(r for r in rows if r['t']==t and r['role']=='gate');allowed=[a for a,z in row['status'].items() if z=='ready']
   make=lambda seed:m['_ap_generate_input'](batch=b,q_len=q,ctx=80000,layer='sliding' if window else 'full',spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
   d=make(PP_SEEDS[0]);graphs={}
   for arm in allowed:
    m['_gate_split_apply']=old_gate if arm in ('A','AA') else lambda xx,yy,ss,arm=arm:m['pp_gate'](xx,yy,bank,arm)
    graphs[arm]=pp_capture(lambda:m['kernel_fn'](d,st))
   m['_gate_split_apply']=old_gate
   for si,seed in enumerate(PP_SEEDS):
    if time.monotonic()>deadline-10:raise TimeoutError('gate layer TASK')
    if si:pp_copy(d,make(seed))
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
     for arm,(graph,out) in graphs.items():
      graph.replay();torch.cuda.synchronize();ok,msg=task['check_implementation'](d,task['collect'](d,out),weights);gate_layer_checks+=1
      if not ok:
       row['status'][arm]='unmeasured:layerTASK:'+msg;failures.append(dict(t=t,role='gate',arm=arm,reason='layerTASK:'+msg))
       if arm in ('A','AA'):raise AssertionError(msg)
    finally:task['ref_front'],task['ref_back']=ff,bf
   graphs=d=None;gc.collect();torch.cuda.empty_cache()
 except Exception as e:fatal=repr(e)+' | '+' | '.join(traceback.format_exc().splitlines()[-8:])
 result=dict(rows=rows,seeds=PP_SEEDS,rounds=4,event_unit_us=.008,fixed=fixed,checks=checks,router_checks=router_checks,gate_layer_checks=gate_layer_checks,failures=failures,fatal=fatal,complete=len(rows)==36 and fatal is None,base_prepare_s=base_prepare,added_prepare_s=bank['prepare_s'],prepare_times=bank['times'],prepare_errors=bank['errors'],nodes=nodes,elapsed_s=time.monotonic()-start,limits='Gate T1/16/32 includes paid application plus native layer TASK; other gate/inactive qkv shapes are diagnostic. Norm/route neighbours paid at sliding geometry. Whole-layer S/F G0 is P4 gate. Failed/unsupported arms are unmeasured. All rounds retained, no trimming.')
 if globals().get('PP_LOCAL_ARCHIVE'):PP_LOCAL_ARCHIVE(result)
 blob=nb_time_encode(result);assert nb_time_decode(blob)==json.loads(json.dumps(result))
 return ['PUBLIC_PROJ_XZ85 '+blob,'END complete='+str(result['complete'])]
