def ps_load():
 path='<P10 packed>';linecache.cache[path]=(len(PS_BASE),None,PS_BASE.splitlines(True),path);m=types.ModuleType('ls_packed');m.__file__=path;exec(compile(PS_BASE,path,'exec'),m.__dict__)
 if globals().get('PS_LOCAL_INSTALL'):PS_LOCAL_INSTALL(m)
 return m.__dict__

def lab(deadline,weights,state=None):
 import contextlib,io,gc,traceback
 start=time.monotonic();m=ps_load();task={};check={}
 for i,s in enumerate(SS_TASK):exec(compile(s,'<published layer '+str(i)+'>','exec'),task)
 exec(compile(SS_CHECK,'<published projection TASK>','exec'),check)
 weights=task['generate_weights'](18000311,router_seed=18000313);log=io.StringIO();t0=time.monotonic()
 with contextlib.redirect_stdout(log):st=m['prepare'](weights)
 base_prepare=time.monotonic()-t0;old=m['ss_prepare'](st);bank=m['ps_prepare'](st);rows=[];failure=None;seeds=[18000319+104729*i for i in range(8)]
 # Explicit roundtrip over every packed BF16 bit, not an approximate check.
 for (role,bw),wp in bank['banks'].items():
  w=st[role];n,k=w.shape;unpacked=wp.permute(1,2,0,3).reshape(n,k);assert torch.equal(unpacked.view(torch.int16),w.view(torch.int16)),('packed bits',role,bw)
 def capture(fn):
  fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  g.replay();torch.cuda.synchronize();return g,out
 def times(gs):
  values=[[] for g in gs]
  for r in range(8):
   events=[]
   for j in (range(len(gs)) if r%2==0 else range(len(gs)-1,-1,-1)):
    lab_cold_l2();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();gs[j].replay();b.record();events.append((j,a,b))
   torch.cuda.synchronize()
   for j,a,b in events:values[j].append(a.elapsed_time(b)*1000)
  return [statistics.median(v) for v in values]
 empty=times([capture(lambda:None)[0]])[0]
 try:
  for t in (128,256):
   for role in ('qkv_gate','o','pool_router','unpooler'):
    if time.monotonic()>deadline-25:raise TimeoutError(str((t,role)))
    w=st[role];n,k=w.shape;x=torch.randn(t,k,device='cuda',dtype=torch.bfloat16);res=torch.randn(t,8192,device='cuda');plan=st['best7_plans'][t//8,8,2048];data=dict(local_start=torch.zeros((),device='cuda',dtype=torch.int32));common=next(i for i,c in enumerate(m['SS_CONFIGS']) if c==(t,128,1,2,0))
    arms=[('A',),('AA',),('OLD',)]+[('U',common,s) for s in (2,3,4)]+[('P',i,s) for i,c in enumerate(m['PS_CONFIGS']) if c[0]==t for s in (2,3,4,5,6,8)]
    def run(a):
     if a[0] in ('A','AA'):return m['_best7_project'](x,role,st,plan,torch.float32 if role=='pool_router' else torch.bfloat16)
     if a[0]=='OLD':return m['ss_project'](x,w,old,*PS_STOCK[t,role])
     if a[0]=='U':return m['ss_project'](x,w,old,a[1],a[2])
     return m['ps_project'](x,w,role,bank,a[1],a[2])
    # Odd splits require the same safe partition guards in the unpacked twin.
    def consume(pair):return m['_best7_route'](pair,st,data,plan) if role=='pool_router' else m['_best7_boundary'](pair,res,st,plan,'b1' if role=='o' else 'b2')
    graphs=[capture(lambda a=a:run(a)) for a in arms];client=role!='qkv_gate'
    if client:graphs += [capture(lambda a=a:consume(run(a))) for a in arms]
    values=[[] for g in graphs];max_error=max_router=0.
    for seed in seeds:
     gen=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=gen).bfloat16());res.copy_(torch.randn(res.shape,device='cuda',generator=gen));d=dict(x=x,config=dict(layer='qkv_gate',m=t,n=n,k=k));wt=dict(qkv_gate=w)
     for a in arms:
      p,s=run(a);v=p.sum(0);out=v.bfloat16();ok,msg=check['check_implementation'](d,out,wt);assert ok,(t,role,a,seed,msg);max_error=max(max_error,check['relative_l2'](out,check['ref_kernel'](d,wt)))
      if role=='pool_router':
       rv=v[:,4096:4352]+v[:,4352:];rd=dict(x=x,config=dict(layer='router',m=t,n=256,k=k));ok,msg=check['check_implementation'](rd,rv,weights);assert ok,(t,a,seed,'FP32router',msg);max_router=max(max_router,check['relative_l2'](rv,check['ref_kernel'](rd,weights)))
     for j,v in enumerate(times([g for g,o in graphs])):values[j].append(round((v-empty)*125))
     x.zero_();x[:,seed%k]=16
     for a in arms:
      p,s=run(a);v=p.sum(0);ok,msg=check['check_implementation'](d,v.bfloat16(),wt);assert ok,(t,role,a,'column',msg)
      if role=='pool_router':
       rv=v[:,4096:4352]+v[:,4352:];rd=dict(x=x,config=dict(layer='router',m=t,n=256,k=k));ok,msg=check['check_implementation'](rd,rv,weights);assert ok,(t,a,'FP32 router column',msg)
    rows.append(dict(t=t,role=role,arms=[list(a) for a in arms],producer=values[:len(arms)],neighbour=values[len(arms):] if client else None,max_error=max_error,max_router_error=max_router));graphs=None;gc.collect();torch.cuda.empty_cache()
 except Exception as e:failure=repr(e)+' | '+' | '.join(traceback.format_exc().splitlines()[-6:])
 result=dict(rows=rows,seeds=seeds,unit_us=.008,complete=len(rows)==8 and failure is None,failure=failure,base_prepare_s=base_prepare,prepare_s=bank['prepare_s'],compile_s=bank['compile_s'],pack_s=bank['pack_s'],resources=bank['resources'],elapsed_s=time.monotonic()-start)
 archive=combined_metadata(dict(oracle={},candidate=result,complete=result['complete']))
 if globals().get('PS_LOCAL_ARCHIVE'):PS_LOCAL_ARCHIVE(archive)
 blob=combined_encode(archive);assert combined_decode(blob)==archive
 lines=['META P10 exact K-major TMA weights+uneven Ksplits;all8sites allarms8seeds8cold;originalFP32router and bitwise packing checks','PACKED_XZ85 '+blob,'END complete='+str(result['complete'])]
 # Check the real documented legacy packet schema, not an inner-blob proxy.
 import zlib
 packet=json.dumps(dict(lines=lines,complete=None,format='legacy-lines'),sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
 wire_bytes=len('LAB_PACKET packed_payload '+'0'*16+' '+'0'*64+' ')+len(base64.b85encode(zlib.compress(packet,9)))+1
 assert wire_bytes<=8744,('packet wire',wire_bytes)
 return lines
