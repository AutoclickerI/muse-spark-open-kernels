def lab(deadline,weights):
 import statistics,linecache,concurrent.futures,json
 start=time.monotonic();mods={}
 for name,src in SOURCES.items():
  f='<softmax_'+name+'>';linecache.cache[f]=(len(src),None,src.splitlines(True),f);m={};exec(compile(src,f,'exec'),m);m['SOFT_TC']=name=='tc25';mods[name]=m
 with concurrent.futures.ThreadPoolExecutor(len(mods)) as pool:
  fs=[pool.submit(m['vq8_ext']) for m in mods.values()]
  for f in fs:f.result()
  fs=[pool.submit(m['vq8_ext'],True) for n,m in mods.items() if n!='A']
  for f in fs:f.result()
 def capture(fn):
  fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  return g,out
 def timed(g):
  lab_cold_l2();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000.
 empty,_=capture(lambda:None);fixed=statistics.median(timed(empty) for _ in range(8))
 lines=['META S14 f1c77ce6 full verify softmax; exact incumbent c262e4c5 vs25/50% FP32 polynomial, WG stagger800 cycles, FP8-P tensor-core row sums. Three fresh seeds x16 rotating cold replays; stage6/min_tiles8.',
        'META all kernels read every key; mixed exponent is FP32, P stays E4M3. Native M64N8 P*ones sums use quantized P; its independent probe and TASK gates run before timing. No promotion.']
 for name,m in mods.items():
  if name!='A':lines.append('PROBE '+name+' '+json.dumps(check_probe(m,torch),separators=(',',':')))
 names=['A','A2']+[n for n in SOURCES if n!='A'];variants=names[2:]
 for B in (16,32):
  times={n:[] for n in names};ratios={n:[] for n in names[1:]};errs=[];clock=[]
  for seed in (61791,14357,92003):
   assert time.monotonic()<deadline-10,'deadline before all seeds complete'
   d=generate_input(batch=B,ctx=80000,heads='tp8',layer='full',q_len=8,page=1,dist='coherent',spread='agentic',seed=seed)
   fn={n:(lambda n=n:mods['A' if n=='A2' else n]['vq8_attention'](d,stages=6,min_tiles=8)) for n in names}
   fn['A']();want=gate_prepare(d);graphs={};outputs={}
   for n in names:
    outputs[n]=fn[n]();ok,msg,e=gate_cached(d,outputs[n],want);assert ok,(n,msg);errs.append(e)
    graphs[n],outputs[n]=capture(fn[n])
   per={n:[] for n in names}
   for r in range(16):
    order=names[r%len(names):]+names[:r%len(names)]
    if r&1:order=list(reversed(order))
    for n in order:per[n].append(timed(graphs[n])-fixed)
   for n in names:times[n].append(statistics.median(per[n]))
   for n in names[1:]:ratios[n].append(statistics.median(x/y for x,y in zip(per[n],per['A'])))
   if seed==61791:
    for n in variants:
     rows=[]
     def cf():return mods[n]['vq8_attention'](d,stages=6,min_tiles=8,clocks=rows)
     cg,cy=capture(cf);cg.replay();torch.cuda.synchronize()
     assert torch.equal(cy,outputs[n]),(n,'clock math differs')
     clk_us=statistics.median(timed(cg)-fixed for _ in range(4))
     act=rows[-1][:_soft_sms(d['qkv'].device)];act=act[act[:,10]!=-1].double().cpu();nt=act[:,10].clamp_min(1)
     med=lambda col:round(float((act[:,col]/nt).median()),1)
     calls=float(act[:,29].sum());taken=float(act[:,30].sum())
     clock.append(dict(n=n,us=round(clk_us,2),tile=round(float((act[:,5]/nt).median()),1),smx=med(20),max=med(25),exp_sum=med(26),pack=med(27),alpha_sum=med(28),rescale=med(31),rescale_calls=int(calls),rescale_taken=int(taken),skip=round(1-taken/max(calls,1),4),qkw=med(19),pvw=med(21),issue=med(22)))
     del rows,cg,cy
   desc=' '.join(f'{n}={statistics.mean(times[n]):.2f}/r{statistics.mean(ratios[n]):.5f}[{min(ratios[n]):.5f},{max(ratios[n]):.5f}]' for n in names[1:])
   del graphs,outputs,fn,d,want,per;torch.cuda.empty_cache()
  lines.append(f'R B{B} L8 A={statistics.mean(times["A"]):.2f} empty={fixed:.2f} err={max(errs):.5g} '+desc)
  lines.append('CLOCK B'+str(B)+' '+json.dumps(clock,separators=(',',':')))
 return lines+[f'END elapsed={time.monotonic()-start:.1f}s']

def _soft_sms(device):return torch.cuda.get_device_properties(device).multi_processor_count
