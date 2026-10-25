OP_MODS=[]
OP_ARM=0
OP_NAMES=('S14','CTA2','TAIL_STATIC','TAIL_QUEUE','TAIL_QUEUE_S3')

def op_build():
 if OP_MODS:return
 arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
 mod=load_inline(name='p3b_tail_'+hashlib.sha256(TAIL_CUDA.encode()).hexdigest()[:12],cpp_sources=TAIL_CPP,cuda_sources=TAIL_CUDA,
  functions=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run','oc_run','oc_info','oc_gather','tail_route','tail_run','tail_info'],
  no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
 mod.br_init();NF_OPTS['pmz']=mod.pmz_init()
 [mod.oc_info(n,f,1) for n in (64,128,192,256) for f in (0,1)]
 [mod.tail_info(a,s,f) for a,s in ((0,2),(1,2),(1,3)) for f in (0,1)]
 OP_MODS.extend([mod]*5)

def op_plan(data,bank,arm):
 global OP_ARM
 OP_ARM=arm
 if arm==0:return nf_light_plan(data,bank['base'],bank['native'],bank['streams'][arm],4)
 if arm==1:return reuse_plan(data,bank['base'],bank['native'],bank['streams'][arm],3)
 return tail_plan(data,bank['base'],bank['native'],bank['streams'][arm],arm)

def lab(deadline,weights,state=None):
 started=time.monotonic();op_build();rows=[];aa=[];peak=0;ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
 for seed in range(24):
  if deadline-time.monotonic()<max(12,peak*1.2+1):break
  tick=time.monotonic();w=dc_weights(5353+7919*seed,'cuda');bank=op_prepare(w)
  for m in (128,256):
   ds=dc_fixtures(w,m);wants=dc_oracles(ds,w);graphs=[MWGraph(lambda d,a=a,bk=bank:op_call(d,bk,a),ds[0]) for a in range(5)]
   for fi,(d,want) in enumerate(zip(ds,wants)):
    for g in graphs:g.refresh(d)
    errors=[op_error(g.replay(),want) for g in graphs];ratios=[1.];rcta=[0.];base,repeat=op_pair(graphs[0],graphs[0],ev);aa.append(repeat-base);us=[base]
    for g in graphs[1:]:
     a,b=op_pair(graphs[0],g,ev);ratios.append(b/a);us.append(b)
    rcta=[us[0]/us[1],1.]
    for g in graphs[2:]:
     a,b=op_pair(graphs[1],g,ev);rcta.append(b/a)
    for i,g in enumerate(graphs):errors[i]=tuple(max(x,y) for x,y in zip(errors[i],op_error(g.replay(),want)))
    assert all(e<=.015 and r<=.075 for e,r in errors),(m,seed,fi,errors)
    rows.append(dict(m=m,seed=seed,fixture=fi,us=us,ratios=ratios,rcta=rcta,errors=errors))
   del graphs
  peak=max(peak,time.monotonic()-tick);del bank,w;torch.cuda.empty_cache()
 lines=['META exact S14 and CTA2 controls vs compact expert-major N16+heavy-tail schedule. STATIC vs device QUEUE isolates work claiming; S3 applies only compact tail. NT1<=8 is unchanged. One route CTA builds list and clears independent W13/W2 counters. No spin or readiness DAG. All work timed. Cold read AB/BA x8, paired directly against BOTH controls.']
 for m in (128,256):
  rr=[r for r in rows if r['m']==m]
  if not rr:continue
  for a,name in enumerate(OP_NAMES):
   rs=[r['ratios'][a] for r in rr];rc=[r['rcta'][a] for r in rr];e=max(r['errors'][a][0] for r in rr);rw=max(r['errors'][a][1] for r in rr)
   lines.append(f'M{m} {name} n={len(rr)} us={statistics.mean(r["us"][a] for r in rr):.2f} pairS14={statistics.mean(rs):.4f} pairCTA2={statistics.mean(rc):.4f} lo/hi={min(rs):.3f}/{max(rs):.3f} L2/row={e:.6f}/{rw:.6f} TASK=PASS')
 lines.append('TAIL_RES '+str([(a,s,f,list(OP_MODS[0].tail_info(a,s,f))) for a,s in ((0,2),(1,2),(1,3)) for f in (0,1)]))
 if aa:lines.append('AA signed/abs/max=%.3f/%.3f/%.3f'%(statistics.mean(aa),statistics.mean(abs(x) for x in aa),max(abs(x) for x in aa)))
 lines.append('END seeds=%d used=%.1f left=%.1f'%(len(rows)//6,time.monotonic()-started,deadline-time.monotonic()));return lines
