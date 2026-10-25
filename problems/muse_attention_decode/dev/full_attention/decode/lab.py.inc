def lab(deadline,weights):
 import concurrent.futures,json,linecache,statistics
 started=time.monotonic();mods={}
 for name,src in SOURCES.items():
  fname='<decode_softmax_'+name+'>';linecache.cache[fname]=(len(src),None,src.splitlines(True),fname);m={};exec(compile(src,fname,'exec'),m);mods[name]=m
 with concurrent.futures.ThreadPoolExecutor(4) as pool:
  fs=[pool.submit(m['ext'],clock) for m in mods.values() for clock in (False,True)]
  for f in fs:f.result()
 def capture(fn):
  out=fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  g.replay();torch.cuda.synchronize();return g,out
 def timed(g):
  lab_cold_l2();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000.
 empty,_=capture(lambda:None);fixed=statistics.median(timed(empty) for _ in range(8))
 lines=['META S14 f1c77ce6 decode16/32 full; attn_main FMA/MUFU25/50%,FP16-P tensor-core row sums; native two-WG q1 reference/mix25/mix50/ping/ping25. Eight fresh seeds x8 rotating cold pairs, A:A.',
        'META same inputs and every key; attn_main arms use identical incumbent split/cluster plan; WG arms use frozen verify kernels at q_len1,live-row mask corrected in ping. Clocks perturb timing; component evidence only.']
 for name,m in mods.items():
  if 'CUDA' in m:lines.append('PROBE '+name+' '+json.dumps(probe(m),separators=(',',':')))
 names=['ref','A2']+[n for n in SOURCES if n!='ref'];arms=names[2:]
 for B in (16,32):
  ratios={n:[] for n in names[1:]};us={n:[] for n in names};errs=[];clocks=[];plan=None
  for seed in tuple(713971+7919*i for i in range(8)):
   assert time.monotonic()<deadline-8,('deadline before eight seeds complete',B,seed)
   d=generate_input(batch=B,ctx=80000,heads='tp8',layer='full',q_len=1,page=1,dist='coherent',spread='agentic',seed=seed)
   if plan is None:plan=mods['ref']['plan_for'](d)
   fn={n:(lambda n=n:mods['ref' if n=='A2' else n]['attention'](d,plan=plan)) for n in names}
   fn['ref']();want=gate_prepare(d);graphs={};out={}
   for n in names:
    graphs[n],out[n]=capture(fn[n]);ok,msg,err=gate_cached(d,out[n],want);assert ok,(B,n,seed,msg);errs.append(err)
   samples={n:[] for n in names}
   for j in range(8):
    order=names[j%len(names):]+names[:j%len(names)]
    if j&1:order=list(reversed(order))
    for n in order:samples[n].append(timed(graphs[n])-fixed)
   for n in names:us[n].append(statistics.median(samples[n]))
   for n in names[1:]:ratios[n].append(statistics.median(x/y for x,y in zip(samples[n],samples['ref'])))
   if seed==713971:
    for n,m in mods.items():
     rows=[];g,y=capture(lambda:m['attention'](d,plan=plan,clocks=rows));assert torch.equal(y,out[n]),(n,'clock math differs')
     wall=statistics.median(timed(g)-fixed for _ in range(4));a=rows[-1].double().cpu()
     if 'CUDA' in m:
      tiles=a[:,10].clamp_min(1);med=lambda i:round(float((a[:,i]/tiles).median()),1)
      calls=float(a[:,20].sum());changed=float(a[:,21].sum())
      r=dict(n=n,clock_us=round(wall,2),tile_cyc=med(12),wait=med(3),sync=med(4),page_issue=med(5),qk=med(6),smx=med(7),pv=med(8),max=med(15),alpha_rescale=med(16),exp_sum=med(17),pack_sum=med(18),max_static_fraction=round(1-changed/max(calls,1),4))
     else:
      sms=torch.cuda.get_device_properties(d['qkv'].device).multi_processor_count;a=a[:sms];a=a[a[:,10]!=-1];tiles=a[:,10].clamp_min(1);med=lambda i:round(float((a[:,i]/tiles).median()),1)
      r=dict(n=n,clock_us=round(wall,2),tile_cyc=med(5),smx=med(20),issue=med(22),qkw=med(19),pvw=med(21),publish=med(23),gather_issue=med(24))
     clocks.append(r);del g,y,rows
   del d,fn,graphs,out,want,samples;torch.cuda.empty_cache()
  desc=' '.join(f'{n}={statistics.mean(us[n]):.2f}/r{statistics.mean(ratios[n]):.5f}[{min(ratios[n]):.5f},{max(ratios[n]):.5f}]/se{statistics.stdev(ratios[n])/8**.5:.5f}' for n in names[1:])
  lines.append(f'R B{B} L1 A={statistics.mean(us["ref"]):.2f} plan={plan} empty={fixed:.2f} err={max(errs):.5g} '+desc)
  lines.append('CLOCK B'+str(B)+' '+json.dumps(clocks,separators=(',',':')))
 return lines+[f'END elapsed={time.monotonic()-started:.1f}s']
