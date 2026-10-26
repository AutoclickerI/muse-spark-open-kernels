OP_MODS=[]
OP_ARM=0
OP_NAMES=('S14','SAFE_CTA2S3','SAFE_STATIC','SAFE_QUEUE','SAFE_QUEUE_S3','CHUNK64_S3_2CTA','CHUNK64_S4_2CTA')
def op_build():
 if OP_MODS:return
 fs=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run','oc_run','oc_info','oc_gather','tail_run','tail_info','tail_route','ch_run','ch_info']
 if torch.cuda.get_device_capability()==(9,0):
  mod=load_inline(name='p3b_join_'+hashlib.sha256(SAFE_CUDA.encode()).hexdigest()[:12],cpp_sources=SAFE_CPP,cuda_sources=SAFE_CUDA,functions=fs,no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'])
 else:
  mod=load_inline(name='p3b_join_'+hashlib.sha256(SAFE_CUDA.encode()).hexdigest()[:12],cpp_sources=SAFE_CPP,cuda_sources=SAFE_CUDA,functions=fs,no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_120,code=sm_120'])
 mod.br_init();NF_OPTS['pmz']=mod.pmz_init();[mod.oc_info(n,f,1) for n in (64,128,192,256) for f in (0,1)];[mod.tail_info(a,s,f) for a,s in ((0,2),(1,2),(1,3)) for f in (0,1)];[mod.ch_info(s,f) for s in (3,4) for f in (0,1)];OP_MODS.extend([mod]*7)
def op_prepare(w):
 w=dict(w,post_norm_w=w['post_expert_norm']);base=nf_perm(nf_wide(m128_hh_issue_prepare(nf_ilv(w))));native=m128_hh_issue_nt_prepare(w)
 return dict(base=base,native=native,streams=[(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()) if a in (0,5,6) else ([torch.cuda.Stream() for _ in range(5)],torch.cuda.Event(),[torch.cuda.Event() for _ in range(5)]) for a in range(7)])
def op_plan(data,bank,arm):
 global OP_ARM
 OP_ARM=arm
 if arm==0:return nf_light_plan(data,bank['base'],bank['native'],bank['streams'][arm],4)
 if arm==1:return reuse_plan(data,bank['base'],bank['native'],bank['streams'][arm],3)
 if arm<=4:return tail_plan(data,bank['base'],bank['native'],bank['streams'][arm],arm)
 return chunk_plan(data,bank['base'],bank['native'],bank['streams'][arm],4,3 if arm==5 else 4)
def lab(deadline,weights,state=None):
 started=time.monotonic();op_build();rows=[];aa=[];peak=0;ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
 for seed in range(24):
  if deadline-time.monotonic()<max(12,peak*1.2+1):break
  tick=time.monotonic();w=dc_weights(5353+7919*seed,'cuda');bank=op_prepare(w)
  for m in (128,256):
   ds=dc_fixtures(w,m);wants=dc_oracles(ds,w);graphs=[MWGraph(lambda d,a=a,bk=bank:op_call(d,bk,a),ds[0]) for a in range(7)]
   for d,want in zip(ds,wants):
    for g in graphs:g.refresh(d)
    errors=[op_error(g.replay(),want) for g in graphs];a,b=op_pair(graphs[0],graphs[0],ev);aa.append(b-a);rs=[1.];us=[a]
    for g in graphs[1:]:
     a,b=op_pair(graphs[0],g,ev);rs.append(b/a);us.append(b)
    rc=[us[0]/us[1],1.]
    for g in graphs[2:]:
     a,b=op_pair(graphs[1],g,ev);rc.append(b/a)
    for i,g in enumerate(graphs):errors[i]=tuple(max(x,y) for x,y in zip(errors[i],op_error(g.replay(),want)))
    assert all(e<=.015 and r<=.075 for e,r in errors),(m,seed,errors)
    rows.append(dict(m=m,us=us,rs=rs,rc=rc,errors=errors))
   del graphs
  peak=max(peak,time.monotonic()-tick);del bank,w;torch.cuda.empty_cache()
 lines=['META shared-X retirement repair: both consumer WGs join after wait1 before copy reuse, plus S1 join after wait0. S14 and safeCTA2S3 direct controls vs safecompact STATIC/QUEUE/S3 and fixed64-token chunk S3/S4 with128-register two-CTA targets. No extra activation pack, cluster or producer role. Cold read ABBAx8, all TASK gates.']
 for m in (128,256):
  rr=[r for r in rows if r['m']==m]
  if not rr:continue
  for a,name in enumerate(OP_NAMES):
   rs=[r['rs'][a] for r in rr];e=max(r['errors'][a][0] for r in rr);rw=max(r['errors'][a][1] for r in rr)
   lines.append(f'M{m} {name} n={len(rr)} us={statistics.mean(r["us"][a] for r in rr):.2f} pairS14={statistics.mean(rs):.4f} pairCTA2={statistics.mean(r["rc"][a] for r in rr):.4f} lo/hi={min(rs):.3f}/{max(rs):.3f} L2/row={e:.6f}/{rw:.6f} TASK=PASS')
 lines.append('CH_RES '+str([(s,f,list(OP_MODS[0].ch_info(s,f))) for s in (3,4) for f in (0,1)]))
 if aa:lines.append('AA signed/abs/max=%.3f/%.3f/%.3f'%(statistics.mean(aa),statistics.mean(abs(x) for x in aa),max(abs(x) for x in aa)))
 lines.append('END seeds=%d used=%.1f left=%.1f'%(len(rows)//6,time.monotonic()-started,deadline-time.monotonic()));return lines
