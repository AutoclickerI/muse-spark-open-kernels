LAB_RESOURCES={'compile_s':300,'run_s':400,'output_bytes':11744,'context_poison':'none'}
SEEDS=tuple(381522727+104729*i for i in range(8))
def lab(deadline,weights,state=None):
 started=time.monotonic();base=module('S22_large_split',B['source']);ck=module('TASK_large_split',B['checker']);layer=module('layer_fixture',B['layer']);rounds={};tele=[];hist=[];resources={};work=[];checks={};endchecks={};biteq={}
 with ThreadPoolExecutor(4) as pool:
  fut=pool.submit(extension);fp=pool.submit(extension,True);fb=[pool.submit(fn) for fn in (base.p3b_module,base.mt_pmz_module,base.mt_module)]
  ext=fut.result();pm=fp.result()
  for f in fb:f.result()
 compiled=time.monotonic()-started
 empty=torch.cuda.CUDAGraph()
 with torch.cuda.graph(empty):pass
 def timed(g):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();a.record();g.replay();b.record();b.synchronize();return round(a.elapsed_time(b)*1e6)
 fixed=statistics.mean(timed(empty) for _ in range(16))
 def task(n,g,phase,p,truth,si,M,ref):
  if phase=='w2':dict(p['phase_steps'])['w13']()
  g.replay()
  if phase=='w13':dict(p['phase_steps'])['w2']()
  dict(p['phase_steps'])['finish']();torch.cuda.synchronize();ok,msg=ck.compare(p['out'],truth,.015);assert ok,('TASK',si,M,phase,n,msg)
  out=p['out'].clone()
  return out if ref is None else bool(torch.equal(out,ref))
 for si,seed in enumerate(SEEDS):
  assert time.monotonic()<deadline-15,('deadline seed',si)
  w=layer.generate_weights(381522721+7919*si,device='cuda',router_seed=seed+17);w['post_norm_w']=w['post_expert_norm'];s=prepare_moe(base,w)
  for M in (128,256):
   d=fixture(layer,w,M,seed);p=make_plan(base,d,s);counts=torch.stack([(d['topk_ids']==e).sum() for e in range(32)]).cpu().tolist();work.append([si,M,counts]);truth=ck.ref_kernel(d,w)
   for phase in ('w13','w2'):
    label=f'{M}:{phase}';args={};graphs={};raw={n:[] for n in NAMES};ref=None
    for n in NAMES:
     for _,fn in p['steps']:fn()
     args[n]=cut(base,d,s,ext,pm,p,phase,n);graphs[n]=capture(args[n]['fn'])[0]
     if si==0:resources[label+':'+n]=args[n]['resources']
     if n in FULL:
      r=task(n,graphs[n],phase,p,truth,si,M,ref)
      if ref is None:ref=r
      else:biteq.setdefault(n,[]).append(int(r))
      checks[n]=checks.get(n,0)+1
    t=[telemetry()]
    for cohort in range(2):
     shift=(si+cohort)%len(NAMES);order=NAMES[shift:]+NAMES[:shift]
     for seq in (order,tuple(reversed(order))):
      for n in seq:
       if phase=='w2':dict(p['phase_steps'])['w13']()
       raw[n].append(timed(graphs[n])//32)
      t.append(telemetry())
    rounds.setdefault(label,[]).append([raw[n] for n in NAMES])
    tele.append([min(x[0] for x in t),min(x[1] for x in t),max(x[2] for x in t),max(x[3] for x in t)])
    for n in NAMES:
     if ARMS[n][0] is not None and ARMS[n][0][3]:hist.append(histograms(args[n]['trace']))
    for n in FULL:
     task(n,graphs[n],phase,p,truth,si,M,ref);endchecks[n]=endchecks.get(n,0)+1
    del graphs,args
   del p,d,truth
  del s,w;gc.collect();torch.cuda.empty_cache()
  print('SPLIT_SEED',ARMSET,si,'elapsed',round(time.monotonic()-started,1),flush=True)
 result=dict(complete=True,armset=ARMSET,arms=list(NAMES),seeds=SEEDS,weight_seeds=[381522721+7919*i for i in range(8)],identities_sha256=sha(B['source']+B['cuda']+B['pmcuda']),
  round_units32=rounds,fixed_ns=round(fixed),telemetry_min_sm_mem_max_temp_W=tele,telemetry_error=TELEMETRY_ERROR,histograms=hist,resources=resources,work_counts=work,
  checks=dict(before=checks,after=endchecks,bit_equal_to_BASE=biteq),compile_s=round(compiled,1),elapsed_s=round(time.monotonic()-started,1),
  protocol='8 fresh TASK-layer seeds; 4 palindromic cold rounds, every round kept (32 ns units); frozen metadata; loads-only arms never promote')
 if hook:=globals().get('XL_LOCAL_DUMP'):hook(result)
 blob=encode_result(result);lines=['SPLIT_RESULT_BIN85 '+blob]
 import zlib
 pkt=json.dumps(dict(lines=lines,complete=None,format='legacy-lines'),sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
 wire=len('LAB_PACKET moe_split_'+ARMSET+' '+'0'*64+' '+'0'*64+' ')+len(base64.b85encode(zlib.compress(pkt,9)))+1
 assert wire<=LAB_RESOURCES['output_bytes'],('packet wire bytes',wire)
 return lines
