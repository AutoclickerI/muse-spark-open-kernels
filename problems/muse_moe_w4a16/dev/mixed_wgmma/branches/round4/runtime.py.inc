OP_MODS=[]
OP_ARM=0
OP_NAMES=('S14','CTA2','WS_TILE_BARRIER','WS_PIPE')

def op_build():
 if OP_MODS:return
 fs=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run','oc_run','oc_info','oc_gather','ws_run','ws_info','ws_finish','ws_gather']
 if torch.cuda.get_device_capability()==(9,0):
  mod=load_inline(name='p3b_ws_'+hashlib.sha256(WS_CUDA.encode()).hexdigest()[:12],cpp_sources=WS_CPP,cuda_sources=WS_CUDA,functions=fs,
   no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'])
 else:
  mod=load_inline(name='p3b_ws_'+hashlib.sha256(WS_CUDA.encode()).hexdigest()[:12],cpp_sources=WS_CPP,cuda_sources=WS_CUDA,functions=fs,
   no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_120,code=sm_120'])
 mod.br_init();NF_OPTS['pmz']=mod.pmz_init();[mod.oc_info(n,f,1) for n in (64,128,192,256) for f in (0,1)];[mod.ws_info(n,f,p) for n in (64,128,192,256) for f in (0,1) for p in (0,1)];OP_MODS.extend([mod]*4)

def ws_packet(q,sf):
 e,ng=sf.shape[:2];mt=ng//4
 qq=q.reshape(e,mt,4,32,2,2,16,4).permute(0,1,3,4,2,5,6,7).contiguous().view(torch.uint8).reshape(e,mt,32,4096)
 ss=sf.reshape(e,mt,4,32,8,16).permute(0,1,3,2,4,5).contiguous().reshape(e,mt,32,512)
 return torch.cat((qq,ss),-1).reshape(e,mt//2,2,32,4608).permute(0,1,3,2,4).contiguous().reshape(e,mt//2,32,9216)

def op_prepare(w):
 w=dict(w,post_norm_w=w['post_expert_norm']);base=nf_perm(nf_wide(m128_hh_issue_prepare(nf_ilv(w))));native=m128_hh_issue_nt_prepare(w)
 base['packet13']=ws_packet(base['w13'],base['w13_sf']);base['packet2']=ws_packet(base['w2'],base['w2_sf'])
 return dict(base=base,native=native,streams=[(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()) if a==0 else ([torch.cuda.Stream() for _ in range(5)],torch.cuda.Event(),[torch.cuda.Event() for _ in range(5)]) for a in range(4)])

def op_plan(data,bank,arm):
 global OP_ARM
 OP_ARM=arm
 if arm==0:return nf_light_plan(data,bank['base'],bank['native'],bank['streams'][arm],4)
 if arm==1:return reuse_plan(data,bank['base'],bank['native'],bank['streams'][arm],3)
 return ws_plan(data,bank['base'],bank['native'],bank['streams'][arm],arm)

def lab(deadline,weights,state=None):
 started=time.monotonic();op_build();rows=[];aa=[];peak=0;ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
 for seed in range(24):
  if deadline-time.monotonic()<max(12,peak*1.2+1):break
  tick=time.monotonic();w=dc_weights(5353+7919*seed,'cuda');bank=op_prepare(w)
  for m in (128,256):
   ds=dc_fixtures(w,m);wants=dc_oracles(ds,w);graphs=[MWGraph(lambda d,a=a,bk=bank:op_call(d,bk,a),ds[0]) for a in range(4)]
   for fi,(d,want) in enumerate(zip(ds,wants)):
    for g in graphs:g.refresh(d)
    errors=[op_error(g.replay(),want) for g in graphs];rs=[1.];a,b=op_pair(graphs[0],graphs[0],ev);aa.append(b-a);us=[a]
    for g in graphs[1:]:
     a,b=op_pair(graphs[0],g,ev);rs.append(b/a);us.append(b)
    rc=[us[0]/us[1],1.]
    for g in graphs[2:]:
     a,b=op_pair(graphs[1],g,ev);rc.append(b/a)
    for i,g in enumerate(graphs):errors[i]=tuple(max(x,y) for x,y in zip(errors[i],op_error(g.replay(),want)))
    assert all(e<=.015 and r<=.075 for e,r in errors),(m,seed,fi,errors)
    rows.append(dict(m=m,us=us,rs=rs,rc=rc,errors=errors))
   del graphs
  peak=max(peak,time.monotonic()-tick);del bank,w;torch.cuda.empty_cache()
 lines=['META S14 / CTA2 vs WS_TILE_BARRIER and WS_PIPE: TMA packed-weight raw ring, three decode warps to BF16 A fragments, two RS-WGMMA WGs sharing X, release-before-epilogue and per-warp norm partials. Barrier arm blocks cross-tile stream until epilogues complete. Packed weights unchanged precision. Real output all arms. Cold read, direct ABBA x8 pairs against BOTH controls.']
 for m in (128,256):
  rr=[r for r in rows if r['m']==m]
  if not rr:continue
  for a,name in enumerate(OP_NAMES):
   e=max(r['errors'][a][0] for r in rr);rw=max(r['errors'][a][1] for r in rr);ratios=[r['rs'][a] for r in rr]
   lines.append(f'M{m} {name} n={len(rr)} us={statistics.mean(r["us"][a] for r in rr):.2f} pairS14={statistics.mean(ratios):.4f} pairCTA2={statistics.mean(r["rc"][a] for r in rr):.4f} lo/hi={min(ratios):.3f}/{max(ratios):.3f} L2/row={e:.6f}/{rw:.6f} TASK=PASS')
 lines.append('WS_RES '+str([(n,f,p,list(OP_MODS[0].ws_info(n,f,p))) for n in (64,128,192,256) for f in (0,1) for p in (0,1)]))
 if aa:lines.append('AA signed/abs/max=%.3f/%.3f/%.3f'%(statistics.mean(aa),statistics.mean(abs(x) for x in aa),max(abs(x) for x in aa)))
 lines.append('END seeds=%d used=%.1f left=%.1f'%(len(rows)//6,time.monotonic()-started,deadline-time.monotonic()));return lines
