# S19 lossless NF reader: prefetch the first codewords before activation readiness.
# OPT64 is new; original OPT6 arithmetic/escape patching and reduction are retained.
PP_POOL_NF={
 'CW8_WAIT':(8,1,False),'CW8_EARLY':(8,1,True),
 'CW8_S2':(8,2,True),'CW8_S2_WAIT':(8,2,False),
 'CW4_EARLY':(4,1,True),'CW4_S2':(4,2,True)}
PP_POOL_TMA={
 'TW256_S2':(256,2,False),'TW256_S2_E':(256,2,True),
 'TW128_S3':(128,3,False),'TW128_S3_E':(128,3,True)}

def pp_pool_prepare(m,state):
 start=time.monotonic();ns=m['_nf_vendor']()
 exec(compile(PP_GPU_PACK_SOURCE,'<existing GPU histogram>','exec'),ns)
 original=ns['v18s_cuda_source']
 def source(insts):
  s=original(insts);wait='  asm volatile("griddepcontrol.wait;" ::: "memory");'
  assert s.count(wait)==1
  s=s.replace(wait,'  if constexpr (!(OPT & 64)) '+wait.strip())
  anchor='    for (int s = 0; s < ST; ++s) load(s, s);'
  assert s.count(anchor)==1
  s=s.replace(anchor,anchor+'\n    if constexpr (OPT & 64) asm volatile("griddepcontrol.wait;" ::: "memory");')
  return s
 ns['v18s_cuda_source']=source;insts=[];indices={};packs={};times={}
 for name,(wk,st,early) in PP_POOL_NF.items():
  for t in (16,32):
   indices[name,t]=len(insts);insts.append((1,t//8,1,st,0,0,wk*32,0,4608,8192,1,wk,128//wk,1,6|(64 if early else 0)))
 t0=time.monotonic();mod=ns['v18s_build'](insts,'p10_pool_prefetch_'+hashlib.sha256(source(insts).encode()).hexdigest()[:12]);times['compile_NF']=time.monotonic()-t0
 t0=time.monotonic();packed,tables,stream=m['_nf_group_pack'](ns,state['pool_router'],'pool_router','v18');times['pack_NF']=time.monotonic()-t0
 for wk in (8,4):
  t0=time.monotonic();cfg=(1,wk,128//wk,1);p=stream(packed,1,*cfg)
  assert p is not None,'exact NF escape capacity'
  count=p['esc'].shape[0];tile=torch.arange(count,device=state['pool_router'].device)//wk;meta=torch.empty((count,34),device=tile.device,dtype=torch.int32);meta[:,:32]=p['esc'];meta[:,32:]=tables[tile].int();packs[wk]=dict(p,esc=meta,cfg=cfg);times['stream_NF_WK'+str(wk)]=time.monotonic()-t0
 torch.cuda.synchronize();return dict(mod=mod,packs=packs,indices=indices,prepare_s=time.monotonic()-start,times=times,insts=insts,source_sha256=hashlib.sha256(source(insts).encode()).hexdigest())

def pp_pool_project(x,bank,pool,name):
 t=x.shape[0];out=torch.empty((t,4608),device=x.device,dtype=torch.float32)
 if name in PP_POOL_NF:
  wk,st,early=PP_POOL_NF[name];p=pool['packs'][wk];rc=pool['mod'].v18s_launch(pool['indices'][name,t],p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),t,4608,8192,*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream().cuda_stream,0)
  if rc:raise RuntimeError('coded pool '+name+' '+str(rc))
 else:
  bk,st,early=PP_POOL_TMA[name];xd=TensorDescriptor.from_tensor(x,[t,bk]);wd=TensorDescriptor.from_tensor(bank['tiled']['pool_router',bk],[64,bk]);fn=_pp_pool_tma_early if early else _cbf_skinny_gemm_tma_tiled_ppdl
  fn[(1,72,1)](xd,wd,wd,out,out,out,t,4608,8192,BLOCK_M=t,BLOCK_N=64,BLOCK_K=bk,SPLIT_K=1,HILO=False,OUT_F32=True,num_warps=4,num_stages=st,launch_pdl=True)
 return out.unsqueeze(0),1
