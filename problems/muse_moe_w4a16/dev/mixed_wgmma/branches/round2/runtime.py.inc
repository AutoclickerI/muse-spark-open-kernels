OP_MODS=[]
OP_ARM=0
OP_NAMES=('S14','TRANS64','TRANS128','CTA2','CTA2S3')

def op_build():
 if OP_MODS:return
 arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
 for ai,c in enumerate(OP_CUDA):
  fs=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run']+(['rt_pack','rt_run','rt_info'] if ai in (1,2) else ['oc_run','oc_info','oc_gather'] if ai>=3 else [])
  mod=load_inline(name='p3b_r2_'+hashlib.sha256(c.encode()).hexdigest()[:12],cpp_sources=OP_CPP[ai],cuda_sources=c,functions=fs,no_implicit_headers=True,with_pytorch_error_handling=False,
   extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
  mod.br_init();NF_OPTS['pmz']=mod.pmz_init();OP_MODS.append(mod)
  if ai>=3:[mod.oc_info(n,f,1) for n in (64,128,192,256) for f in (0,1)]

def nf_module():return OP_MODS[OP_ARM]

def op_prepare(w):
 w=dict(w,post_norm_w=w['post_expert_norm']);base=nf_perm(nf_wide(m128_hh_issue_prepare(nf_ilv(w))));native=m128_hh_issue_nt_prepare(w)
 return dict(base=base,native=native,streams=[(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()) if a==0 else ([torch.cuda.Stream() for _ in range(5)],torch.cuda.Event(),[torch.cuda.Event() for _ in range(5)]) for a in range(5)])

def op_plan(data,bank,arm):
 global OP_ARM
 OP_ARM=arm
 if arm==0:return nf_light_plan(data,bank['base'],bank['native'],bank['streams'][arm],4)
 return reuse_plan(data,bank['base'],bank['native'],bank['streams'][arm],arm)

def reuse_plan(data,base,native,streams,arm):
 x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*128;mod=nf_module();sms=torch.cuda.get_device_properties(dev).multi_processor_count
 c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(c);tasks=[torch.empty_like(off) for _ in range(3)];slots=torch.empty((32,m),device=dev,dtype=torch.int32)
 ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();pos=torch.empty_like(ids);start=data['config'].get('local_start',0)
 act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((cap,4096),device=dev);ss=torch.empty((cap,64),device=dev);out=torch.empty_like(x)
 h13=[base['w13'],base['w13_sf'],base['w13_gs'],x,c,off,order,tasks[1],slots,base['pnw'],act,y,ss]
 h2=[base['w2'],base['w2_sf'],base['w2_gs'],act,c,off,order,tasks[1],slots,base['pnw'],act,y,ss]
 nv=[x,base['pnw'],native['w13p'],native['w2p'],base['w13_gs'],base['w2_gs'],c,off,slots,act,y,ss]
 l13=list(h13);l2=list(h2);l13[7]=tasks[2];l2[7]=tasks[2]
 p13,p2,pn,pl13,pl2=([t.data_ptr() for t in v] for v in (h13,h2,nv,l13,l2))
 a1=torch.empty((32,(m+63)//64,256,128,4),device=dev,dtype=torch.int32) if arm in (1,2) else x
 a2=torch.empty_like(a1) if arm in (1,2) else x
 rp=[t.data_ptr() for t in (ids,c,off,order,tasks[0],slots,pos,tasks[1],tasks[2])];fp=[t.data_ptr() for t in (off,ids,tw,pos,y,ss,out)]
 sides,fork,joins=streams;stages=2
 while stages>1 and mod.pmz_smem(stages,m)>NF_OPTS['pmz']:stages-=1
 grids=[sms*mod.pmz_resident(stages,m,f) for f in (0,1)]
 owners=(h13,h2,nv,ids,tw,pos,tasks,a1,a2,l13,l2)
 def route():
  _=owners;mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)
 def body():
  _=owners;parent=torch.cuda.current_stream()
  if arm in (1,2):mod.rt_pack(p13,a1.data_ptr(),m,parent.cuda_stream)
  fork.record(parent)
  classes=(64,128,256) if arm in (1,2) else (64,128,192,256)
  for side,join,capn in zip(sides,joins,classes):
   side.wait_event(fork)
   with torch.cuda.stream(side):
    if arm in (1,2):
     mod.rt_run(p13,a1.data_ptr(),a2.data_ptr(),64 if arm==1 else 128,capn,1,m,sms*3,side.cuda_stream)
     mod.rt_run(p2,a2.data_ptr(),a2.data_ptr(),64 if arm==1 else 128,capn,0,m,sms*3,side.cuda_stream)
    else:
     mod.oc_run(p13,capn,1,1,sms*4,m,side.cuda_stream);mod.oc_run(p2,capn,0,1,sms*4,m,side.cuda_stream)
    join.record()
  sides[4].wait_event(fork)
  with torch.cuda.stream(sides[4]):mod.br_run(pl13,70,sms*4,m,sides[4].cuda_stream);mod.br_run(pl2,71,sms*4,m,sides[4].cuda_stream);joins[4].record()
  mod.pmz_run(pn,1,8,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)
  for join in list(joins[:len(classes)])+[joins[4]]:parent.wait_event(join)
 def finish():
  _=owners;mod.br_finish(fp,m,start,4,torch.cuda.current_stream().cuda_stream);return out
 return dict(steps=(route,body,finish),out=out,owners=owners)

def op_call(data,bank,arm):
 for f in op_plan(data,bank,arm)['steps']:out=f()
 return out

def op_error(o,w):
 d=o.float()-w.float();n=w.float().norm(dim=-1)
 return float(d.norm()/w.float().norm().clamp_min(1e-12)),float((d.norm(dim=-1)/n.clamp_min(n.mean()*.1+1e-12)).max())

def op_pair(a,b,ev):
 ts=[[],[]]
 for i in range(8):
  for j in ((0,1) if i%2==0 else (1,0)):
   lab_cold_l2();ev[0].record();(a,b)[j].replay();ev[1].record();ev[1].synchronize();ts[j].append(ev[0].elapsed_time(ev[1])*1000.)
 return statistics.median(ts[0]),statistics.median(ts[1])

def lab(deadline,weights,state=None):
 started=time.monotonic();op_build();rows=[];aa=[];peak=0;profiles=[];ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
 for seed in range(24):
  if deadline-time.monotonic()<max(12,peak*1.2+1):break
  tick=time.monotonic();w=dc_weights(5353+7919*seed,'cuda');bank=op_prepare(w)
  if seed==0:profiles=reuse_profiles(w,bank)
  for m in (128,256):
   ds=dc_fixtures(w,m);wants=dc_oracles(ds,w);graphs=[MWGraph(lambda d,a=a,bk=bank:op_call(d,bk,a),ds[0]) for a in range(5)]
   for fi,(d,want) in enumerate(zip(ds,wants)):
    for g in graphs:g.refresh(d)
    errors=[op_error(g.replay(),want) for g in graphs];ratios=[1.];base,repeat=op_pair(graphs[0],graphs[0],ev);aa.append(repeat-base);us=[base]
    for g in graphs[1:]:
     a,b=op_pair(graphs[0],g,ev);ratios.append(b/a);us.append(b)
    for i,g in enumerate(graphs):errors[i]=tuple(max(x,y) for x,y in zip(errors[i],op_error(g.replay(),want)))
    assert all(e<=.015 and r<=.075 for e,r in errors),(m,seed,fi,errors)
    rows.append(dict(m=m,seed=seed,fixture=fi,us=us,ratios=ratios,errors=errors))
   del graphs
  peak=max(peak,time.monotonic()-tick);del bank,w;torch.cuda.empty_cache()
 lines=['META S14 nf f1c77ce6 vs TRANS64/128 decoded-B shared within CTA, register A activation fragments, lossless timed pack and direct SwiGLU-to-W2 fragment stores; CTA2 native RS with two consumer WGs sharing one X slice. Packed weights decode only inside timed call. Light N16G4 retained. No clusters or global queues. Cold read AB/BA x8.']
 lines+=profiles
 for m in (128,256):
  rr=[r for r in rows if r['m']==m]
  if not rr:continue
  for a,name in enumerate(OP_NAMES):
   ratios=[r['ratios'][a] for r in rr];e=max(r['errors'][a][0] for r in rr);rw=max(r['errors'][a][1] for r in rr)
   lines.append(f'M{m} {name} n={len(rr)} us={statistics.mean(r["us"][a] for r in rr):.2f} pair={statistics.mean(ratios):.4f} lo/hi={min(ratios):.3f}/{max(ratios):.3f} L2/row={e:.6f}/{rw:.6f} TASK=PASS')
 lines.append('RES_TRANS '+str([(n,c,f,list(OP_MODS[1].rt_info(n,c,f))) for n in (64,128) for c in (64,128,256) for f in (0,1)]))
 lines.append('RES_CTA2 '+str([(a,n,f,list(OP_MODS[a].oc_info(n,f,1))) for a in (3,4) for n in (64,128,192,256) for f in (0,1)]))
 if aa:lines.append('AA signed/abs/max=%.3f/%.3f/%.3f'%(statistics.mean(aa),statistics.mean(abs(x) for x in aa),max(abs(x) for x in aa)))
 lines.append('END seeds=%d used=%.1f left=%.1f'%(len(rows)//6,time.monotonic()-started,deadline-time.monotonic()));return lines

def reuse_profiles(w,bank):
 """Fixed-fixture phase service; never treated as critical-path increments."""
 lines=[];ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
 def capture(fn):
  fn();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):fn()
  return g
 for m in (128,256):
  d=dc_fixtures(w,m)[0];plans=[];phases=[];packs=[];streams=[];sms=torch.cuda.get_device_properties(d['hidden_states'].device).multi_processor_count
  for arm in range(5):
   plan=op_plan(d,bank,arm)
   for fn in plan['steps']:fn()
   torch.cuda.synchronize();plans.append(plan);mod=OP_MODS[arm];h13,h2=plan['owners'][:2];p13,p2=([t.data_ptr() for t in z] for z in (h13,h2))
   sides=[torch.cuda.Stream() for _ in range(4)];fork=torch.cuda.Event();joins=[torch.cuda.Event() for _ in range(4)];streams.append((sides,fork,joins))
   a1=plan['owners'][7] if arm in (1,2) else None;a2=plan['owners'][8] if arm in (1,2) else None
   packs.append(capture(lambda mod=mod,p=p13,a=a1:mod.rt_pack(p,a.data_ptr(),m,torch.cuda.current_stream().cuda_stream)) if a1 is not None else None)
   pp=[]
   for first,p in ((1,p13),(0,p2)):
    def phase(arm=arm,mod=mod,p=p,first=first,a1=a1,a2=a2,sides=sides,fork=fork,joins=joins):
     parent=torch.cuda.current_stream()
     if arm==0:mod.br_run(p,26 if first else 21,1056 if m<=128 else 2112,m,parent.cuda_stream);return
     caps=(64,128,256) if arm in (1,2) else (64,128,192,256);fork.record(parent)
     for capn,side,join in zip(caps,sides,joins):
      side.wait_event(fork)
      with torch.cuda.stream(side):
       if arm in (1,2):mod.rt_run(p,(a1 if first else a2).data_ptr(),a2.data_ptr(),64 if arm==1 else 128,capn,first,m,sms*3,side.cuda_stream)
       else:mod.oc_run(p,capn,first,1,sms*4,m,side.cuda_stream)
       join.record()
     for join in joins[:len(caps)]:parent.wait_event(join)
    pp.append(capture(phase))
   phases.append(pp)
  for arm,name in enumerate(OP_NAMES):
   vals=[op_pair(g,g,ev)[0] for g in phases[arm]];pack=op_pair(packs[arm],packs[arm],ev)[0] if packs[arm] is not None else 0
   lines.append('PHASE M%d %s H13/H2/pack_us=%.2f/%.2f/%.2f'%(m,name,*vals,pack))
  del phases,plans,packs,streams
 return lines
