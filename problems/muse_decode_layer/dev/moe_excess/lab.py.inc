LAB_RESOURCES={'compile_s':250,'run_s':430,'output_bytes':11744,'context_poison':'none'}
SEEDS=tuple(381522727+104729*i for i in range(8))
def lab(deadline,weights,state=None):
 started=time.monotonic();base=module('S19_large_excess',B['source']);ck=module('TASK_large_excess',B['checker']);layer=module('layer_fixture',B['layer']);rounds={};tele=[];hist=[];resources={};work=[];checks={};endchecks={}
 with ThreadPoolExecutor(4) as pool:
  fut=pool.submit(extension);fp=pool.submit(extension,True);fb=[pool.submit(fn) for fn in (base.p3b_module,base.mt_pmz_module,base.mt_module)]
  ext=fut.result();pm=fp.result()
  for f in fb:f.result()
 empty=torch.cuda.CUDAGraph()
 with torch.cuda.graph(empty):pass
 def timed(g):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();t=time.perf_counter_ns();a.record();g.replay();b.record();host=time.perf_counter_ns()-t;b.synchronize();return round(a.elapsed_time(b)*1e6),host
 fixed=statistics.mean(timed(empty)[0] for _ in range(16))
 for si,seed in enumerate(SEEDS):
  assert time.monotonic()<deadline-15,('deadline seed',si)
  w=layer.generate_weights(381522721+7919*si,device='cuda',router_seed=seed+17);w['post_norm_w']=w['post_expert_norm'];s=prepare_moe(base,w)
  for M in (128,256):
   d=fixture(layer,w,M,seed);p=make_plan(base,d,s);counts=torch.stack([(d['topk_ids']==e).sum() for e in range(32)]).cpu().tolist();work.append([si,M,counts]);truth=ck.ref_kernel(d,w)
   for phase in ('w13','w2'):
    label=f'{M}:{phase}';args={};graphs={};raw={n:[] for n in NAMES};enqueue={n:[] for n in ('BASE','AA')}
    # Exact ready fixture before every arm capture. W2 low uses atomic SS, so
    # reset from the W13 producer outside every isolated W2 measurement.
    for n in NAMES:
     for _,fn in p['steps']:fn()
     args[n]=cut(base,d,s,ext,pm,p,phase,n);graphs[n]=capture(args[n]['fn'])[0]
     if args[n]['resources'] is not None:resources[label+':'+n]=dict(info=args[n]['resources'],grids=args[n]['grids'])
     if n in ('BASE','AA','MATCH','SHADOW','L3','HIST'):
      if phase=='w2':dict(p['phase_steps'])['w13']()
      graphs[n].replay()
      if phase=='w13':dict(p['phase_steps'])['w2']()
      dict(p['phase_steps'])['finish']();torch.cuda.synchronize();ok,msg=ck.compare(p['out'],truth,.015);assert ok,('TASK',si,M,phase,n,msg);checks[n]=checks.get(n,0)+1
    t=[telemetry()]
    for cohort in range(2):
     shift=(si+cohort)%len(NAMES);order=NAMES[shift:]+NAMES[:shift]
     for seq in (order,tuple(reversed(order))):
      for n in seq:
       if phase=='w2':dict(p['phase_steps'])['w13']()
       ns,host=timed(graphs[n]);raw[n].append(ns)
       if n in enqueue:enqueue[n].append(host)
      t.append(telemetry())
    rounds.setdefault(label,[]).append(raw);tele.append([si,label,t,enqueue]);hist.append([si,label,histograms(args['HIST']['trace'],si==0)])
    for n in ('BASE','AA','MATCH','SHADOW','L3','HIST'):
     if phase=='w2':dict(p['phase_steps'])['w13']()
     graphs[n].replay()
     if phase=='w13':dict(p['phase_steps'])['w2']()
     dict(p['phase_steps'])['finish']();torch.cuda.synchronize();ok,msg=ck.compare(p['out'],truth,.015);assert ok,('POST TASK',si,M,phase,n,msg);endchecks[n]=endchecks.get(n,0)+1
    del graphs,args
   del p,d,truth
  del s,w;gc.collect();torch.cuda.empty_cache()
  print('LARGE_LADDER_SEED',si,'elapsed',round(time.monotonic()-started,1),flush=True)
 result=dict(complete=True,seeds=SEEDS,weight_seeds=[381522721+7919*i for i in range(8)],identities_sha256=sha(B['source']+B['cuda']+B['pmcuda']),round_ns=rounds,fixed_ns=fixed,telemetry=tele,telemetry_error=TELEMETRY_ERROR,histograms=hist,resources=resources,work_counts=work,checks=dict(before=checks,after=endchecks,diagnostics='L0/L1/L2/MATH/INIT/STREAM are excluded-work oracles; never candidates'),elapsed_s=time.monotonic()-started,protocol='8fresh TASK-layer seeds;4palindromic cold rounds;all rounds;fixed metadata;matched occupancy;diagnostics never promote')
 if hook:=globals().get('XL_LOCAL_DUMP'):hook(result)
 blob=encode_result(result);lines=['LADDER_RESULT_BIN85 '+blob]
 # Enforce the actual public transport limit, rather than an arbitrary bound
 # on the inner compressed blob. Header allowance exceeds this payload's.
 import zlib
 pkt=json.dumps(dict(lines=lines,complete=None,format='legacy-lines'),sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
 wire=len('LAB_PACKET large_moe_excess_ladder '+'0'*64+' '+'0'*64+' ')+len(base64.b85encode(zlib.compress(pkt,9)))+1
 assert wire<=LAB_RESOURCES['output_bytes'],('packet wire bytes',wire)
 return lines
