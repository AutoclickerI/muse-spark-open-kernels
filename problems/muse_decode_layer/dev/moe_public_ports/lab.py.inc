LAB_RESOURCES={'compile_s':340,'run_s':470,'output_bytes':11744,'context_poison':'none'}
SEEDS=tuple(183528941+104729*i for i in range(8))
MS=(1,8,16,32,128,256)

def lab(deadline,weights,state=None):
 start=time.monotonic();checks={};fail=[];rounds={};phase_rounds={};nodecounts={}
 base=module('S19',BUNDLE['base']);op=module('operator',BUNDLE['operator']);lee=module('lee',BUNDLE['lee']);ck=module('TASK',BUNDLE['checker'])
 hook=globals().get('MOE_LOCAL_INSTALL')
 if hook:hook(base,op,lee)
 # All source compilation is independent. Each module has a source-specific
 # namespace to avoid weak C++ symbols sharing state across configurations.
 with ThreadPoolExecutor(7) as pool:
  futures={n:pool.submit(lee_module,k,f) for n,k,f in (('X1','X',('-DRPW=2',)),('X2','X',('-DNCW=16',)),('F1','F',('-DWPC=16','-DSMAX=3','-DMT2','-DLASTA','-DSMREQ=196')),('F2','F',('-DWPC=16','-DSMAX=3','-DMT2','-DSMREQ=196')))}
  fbase=[pool.submit(fn) for fn in (base.rvc_nfidx0_module,base.rvc_nbase1_module,base.rvc_nbase2_module,base.p3b_module,base.mt_module,base.mt_pmz_module)]
  w=ck.generate_weights(183528937);w={k:(v[:32].contiguous() if k.startswith(('w13','w2')) else v) for k,v in w.items()};sop=op.prepare(w);sl=lee.prepare_lee(w);sb=s19_prepare(base,w)
  modules={n:f.result() for n,f in futures.items()}
  for f in fbase:f.result()
 torch.cuda.synchronize();setup=time.monotonic()-start;print('MOE_PUBLIC_SETUP',round(setup,2),flush=True)
 e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 def timegraph(g):
  lab_cold_l2();e0.record();g.replay();e1.record();e1.synchronize();return round(e0.elapsed_time(e1)*1e6)
 empty=torch.cuda.CUDAGraph()
 with torch.cuda.graph(empty):pass
 fixed=statistics.mean(timegraph(empty) for _ in range(16))
 def make(d,all_phases=False):
  M=d['hidden_states'].shape[0];plans={'A':s19_plan(base,d,sb),'A2':s19_plan(base,d,sb),'O':op.operator_plan(d,sop)}
  for i in (1,2):
   plans['X'+str(i)]=lee.lee_plan(d,sl,modules['X'+str(i)])
   if M<=32:plans['F'+str(i)]=lee.lee_plan(d,sl,modules['F'+str(i)],'F')
   else:plans['S'+str(i)]=lee.lee_plan(d,sl,modules['X'+str(i)],split=True)
  return plans
 def verify(d,plans,label):
  ref=ck.ref_kernel(d,w)
  for name,p in plans.items():
   y=run_plan(p);torch.cuda.synchronize();ok,msg=ck.compare(y,ref,ck.REL_L2_TOL);checks[name]=checks.get(name,0)+1
   if not ok:fail.append((label,name,'eager',msg))
   g,y=capture(lambda p=p:run_plan(p));g.replay();g.replay();torch.cuda.synchronize();ok,msg=ck.compare(y,ref,ck.REL_L2_TOL);checks[name]+=1
   if not ok:fail.append((label,name,'graph',msg))
   del g
  assert not fail,('TASK',fail)
 # Official component TASK plus all requested ABI shapes and shifted local ids.
 for M,dist in ((4,'normal'),(33,'normal'),(257,'normal'),(128,'skewed'),(1,'normal')):
  # F's documented domain is <=32. X/O and S19's large plan cover nearby.
  d=ck.generate_input(bs=M,sharding='ep8',seed=183528919+M,dist=dist);plans=make(d);verify(d,plans,'official:'+str(M)+dist);plans=d=None;gc.collect();torch.cuda.empty_cache()
 for M in MS:
  samples=[];phases=[]
  for si,seed in enumerate(SEEDS):
   assert time.monotonic()<deadline-10,('deadline',M,si)
   d=ck.generate_input(bs=M,sharding='ep8',seed=seed,dist='normal');plans=make(d);ref=ck.ref_kernel(d,w);graphs={}
   for name,p in plans.items():
    g,y=capture(lambda p=p:run_plan(p));g.replay();torch.cuda.synchronize();ok,msg=ck.compare(y,ref,ck.REL_L2_TOL);checks[name]=checks.get(name,0)+1;assert ok,('TASK',M,seed,name,msg);graphs[name]=g
   names=tuple(plans);per={n:[] for n in names};orders=[]
   # Two cohorts, each exact forward/reverse. All arms share each seed,
   # cold-cache protocol and four rounds; every round survives in the packet.
   for c in range(2):
    k=(si+c)%len(names);order=names[k:]+names[:k]
    for seq in (order,tuple(reversed(order))):
     orders.append(seq)
     for n in seq:per[n].append(timegraph(graphs[n]))
   samples.append(per)
   # Isolated S19 and donor phases on ready fixtures. S19's small-M body
   # stays fused: phase denominators are deliberately absent there.
   pp={n:p for n,p in plans.items() if n in ('A','O') or n.startswith('S')}
   if M<=32:
    for i in (1,2):pp['S'+str(i)]=lee.lee_plan(d,sl,modules['X'+str(i)],split=True)
   pgs={};pv={}
   for name,p in pp.items():
    run_plan(p);torch.cuda.synchronize();pgs[name]=phase_graphs(p)
    # The explicit split arm must pass TASK too, even when it is diagnostic.
    out=run_plan(p);torch.cuda.synchronize();ok,msg=ck.compare(out,ref,ck.REL_L2_TOL);checks['phases:'+name]=checks.get('phases:'+name,0)+1;assert ok,('phase TASK',M,seed,name,msg)
    for stage,g in pgs[name].items():
     if stage in ('fused','body'):continue
     pv[name+':'+stage]=[]
   keys=tuple(pv)
   for c in range(2):
    k=(si+c)%max(1,len(keys));order=keys[k:]+keys[:k]
    for seq in (order,tuple(reversed(order))):
     for key in seq:
      n,stage=key.split(':')
      for label,fn in pp[n].get('phase_steps',pp[n]['steps']):
       if label==stage:break
       fn()
      pv[key].append(timegraph(pgs[n][stage]))
   # Confirm timed repeated stages leave valid results. W2 accumulators and
   # route counters are part of the port, never reset outside its timed call.
   for n,p in pp.items():
    out=run_plan(p);torch.cuda.synchronize();ok,msg=ck.compare(out,ref,ck.REL_L2_TOL);assert ok,('post phases TASK',M,seed,n,msg)
   phases.append(pv);plans=graphs=pgs=pp=d=ref=None;gc.collect();torch.cuda.empty_cache()
  rounds[str(M)]=samples;phase_rounds[str(M)]=phases;print('MOE_PUBLIC_TIMED',M,round(time.monotonic()-start,1),flush=True)
 result=dict(complete=True,seeds=SEEDS,weight_seed=183528937,checks=checks,failures=fail,fixed_ns=fixed,round_ns=rounds,phase_ns=phase_rounds,setup_s=setup,elapsed_s=time.monotonic()-start,identities={k:hashlib.sha256(v.encode()).hexdigest() for k,v in BUNDLE.items()},protocol='8 fresh input seeds, two rotated forward/reverse cohorts, 4 cold rounds; all integer ns kept; same seeds every arm; phase times isolated ready fixtures; small S19 fused phases N/A')
 binary=bytearray()
 def vint(v):
  while v>=128:binary.append((v&127)|128);v>>=7
  binary.append(v)
 for field in ('round_ns','phase_ns'):
  packed={}
  for M,rows in result[field].items():
   keys=list(rows[0]);packed[M]=dict(keys=keys,seeds=len(rows),rounds=4,offset=len(binary))
   for row in rows:
    assert list(row)==keys
    for key in keys:
     vs=row[key];assert len(vs)==4;vint(vs[0])
     for v in vs[1:]:
      d=v-vs[0];vint(2*d if d>=0 else -2*d-1)
  result[field]=packed
 meta=json.dumps(result,separators=(',',':')).encode();raw=struct.pack('<I',len(meta))+meta+binary
 blob=base64.b85encode(lzma.compress(raw,preset=9)).decode();assert len(blob)<11100,('packet',len(blob))
 return ['RESULT_BIN85 '+blob]
