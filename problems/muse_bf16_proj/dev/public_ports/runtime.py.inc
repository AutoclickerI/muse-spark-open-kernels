PP_LOCAL_MM=None
PP_ARCH=None

def pp_clone(g):
 n=dict(g)
 for key,f in g.items():
  if isinstance(f,types.FunctionType) and f.__globals__ is g:
   c=types.FunctionType(f.__code__,n,f.__name__,f.__defaults__,f.__closure__);c.__kwdefaults__=f.__kwdefaults__;n[key]=c
 return n

def pp_prepare(m,weights,state,requested):
 requested=set(requested);start=time.monotonic();roles={r:state[r] for r in ('qkv_gate','qkv3','o','pool_router','unpooler','unpooler_slice')};roles['gate']=state['gate_rows'];times={};coded={};tiled={};raw={};nf={};errors={}
 needed_c={('qkv_gate' if r in ('qkv_gate','qkv3','gate') else 'unpooler' if r=='unpooler_slice' else r) for t,r,p in requested if p=='C'}
 for role in sorted(needed_c):
  t0=time.monotonic();coded[role]=pp_encode(roles[role]);times['coded_'+role]=time.monotonic()-t0
 if 'qkv_gate' in coded:coded['qkv3']=pp_slice(coded['qkv_gate'],0,1280);coded['gate']=pp_slice(coded['qkv_gate'],1280,2304)
 if 'unpooler' in coded:coded['unpooler_slice']=pp_slice(coded['unpooler'],0,1024)
 needed_tiled={(r,256) for t,r,p in requested if p in ('O','P') and 1<t<=32 and r in ('pool_router','unpooler')}
 needed_tiled|={(r,128) for t,r,p in requested if p in ('O','P') and t==128 and r in ('qkv_gate','qkv3','gate')}
 for role,bk in sorted(needed_tiled):
  w=roles[role];t0=time.monotonic();n,k=w.shape;wp=w.reshape(n//64,64,k//bk,bk).permute(0,2,1,3).contiguous().view(-1,bk);torch.cuda.synchronize();tiled[role,bk]=wp;times['raw_tile_'+str((role,bk))]=time.monotonic()-t0
 # Exact keys prevent S19's selected-only shape dispatcher from hiding donor rows.
 ns=pp_clone(m);ns['_s15n_nf_key']=lambda t,name:(t,name);ns['NF_PORT_SELECTED']=set(ns['NF_SMALL'])|set(ns['NF_WIDE'])
 # The existing reviewed v18 packer histogram is kept on GPU, as already proved
 # in dense_all; this changes prepare only, not any timed donor arithmetic.
 if 'PP_GPU_PACK_SOURCE' in globals():
  old=ns['_nf_vendor']
  def vendor():
   v=old();exec(compile(PP_GPU_PACK_SOURCE,'<reviewed GPU-only histogram packer>','exec'),v);return v
  ns['_nf_vendor']=vendor
 groups={}
 available=set(ns['NF_SMALL'])|set(ns['NF_WIDE']);nf_requested={(t,r) for t,r,p in requested if p=='N'}
 for key in sorted(available&nf_requested):
  if key in m['NF_PORT_SELECTED'] and key in state['nf_port']['plans']:
   nf[key]=(m,state['nf_port'],key[1]);continue
  family=ns['NF_SMALL'][key][0] if key in ns['NF_SMALL'] else 'wide';groups.setdefault(family,set()).add(key)
 for family,keys in groups.items():
  t0=time.monotonic()
  try:
   bank=ns['nf_prepare'](weights,keys)
   for key in keys:nf[key]=(ns,bank,key[1])
  except Exception as e:
   for key in keys:errors['NF'+str(key)]=repr(e)
  times['NF_'+family]=time.monotonic()-t0
 # Generalised gate and qkv3@8 use the small selected templates' dynamic N.
 for role,rows,ts in (('gate',(1280,2304),(1,8,16,32)),('qkv3',(0,1280),(8,))):
  ts=tuple(t for t in ts if (t,role) in nf_requested)
  if not ts:continue
  t0=time.monotonic();ww=dict(weights,qkv_gate=weights['qkv_gate'][rows[0]:rows[1]].contiguous());keys={(t,'qkv_gate') for t in ts}
  try:
   bank=ns['nf_prepare'](ww,keys)
   for t in ts:nf[t,role]=(ns,bank,'qkv_gate')
  except Exception as e:
   for t in ts:errors['NF'+str((t,role))]=repr(e)
  times['NF_'+role]=time.monotonic()-t0
 torch.cuda.synchronize();return dict(roles=roles,coded=coded,tiled=tiled,nf=nf,errors=errors,times=times,cluster={},scratch={},prepare_s=time.monotonic()-start)

def pp_cfg(t,role):
 if t<=32:
  if role in ('qkv_gate','qkv3','gate'):return (64,256,3,4,4,max(16,t))
  if role=='unpooler_slice':return (64,256,4,4,4,16)
  return (64,256,1,4,4,max(16,t))
 return (64,128,3 if role in ('qkv_gate','qkv3','gate') else 1,4 if t==128 else 8,4,t)

def pp_cluster(x,w,out,cfg,bank,pdl):
 from triton import knobs
 bn,bk,s,nw,st,bm=cfg;t,k=x.shape;n=w.shape[0];xd=TensorDescriptor.from_tensor(x,[bm,bk]);wd=TensorDescriptor.from_tensor(w,[bn,bk]);grid=(1,n//bn,triton.cdiv(t,bm));slot=8 if (s-1)*bn*bm*4<=8192 else 16;args=(xd,wd,out,t,n,k,bm,bn,bk,s,slot);key=(pdl,cfg,t,n,k)
 if key not in bank['cluster']:
  fn=globals()['_cbf_cluster_gemm_ppdl' if pdl else '_cbf_cluster_gemm'];ck=fn.warmup(*args[:6],BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,SPLIT_K=s,SLOT_KB=slot,num_warps=nw,num_stages=st,grid=grid);ck._init_handles();pm=list(ck.packed_metadata);assert len(pm)==3 and pm[1]==1 and pm[2]>49152,pm;pm[1]=s;bank['cluster'][key]=(ck,tuple(pm))
 ck,pm=bank['cluster'][key];stream=torch.cuda.current_stream().cuda_stream;ck.run(*grid,stream,ck.function,pm,ck.launch_metadata(grid,stream,*args),knobs.runtime.launch_enter_hook,knobs.runtime.launch_exit_hook,*args)


def pp_project(x,role,bank,provider):
 t,k=x.shape;w=bank['roles'][role];n=w.shape[0]
 if PP_LOCAL_MM is not None:
  out=torch.empty((t,n),device=x.device,dtype=torch.float32 if role=='pool_router' or provider in ('C','N') else torch.bfloat16);PP_LOCAL_MM(x,w,out);return out.unsqueeze(0),1
 if provider=='N':
  if (t,role) not in bank['nf']:raise ValueError('No released NF shape '+str((t,role)))
  ns,st,name=bank['nf'][t,role];return ns['nf_project'](x,name,st)
 bn,bk,s,nw,st,bm=pp_cfg(t,role);pdl=provider=='P';dtype=torch.float32 if role=='pool_router' else torch.bfloat16
 if provider=='C':
  st=1;c=bank['coded'][role];out=torch.empty((s,t,n),device=x.device,dtype=torch.float32);xd=TensorDescriptor.from_tensor(x,[bm,bk]);_pp_coded[(triton.cdiv(t,bm),n//bn,s)](xd,c['words'],c['base'],c['offset'],c['values'],out,t,n,k,BM=bm,BN=bn,BK=bk,S=s,num_warps=nw,num_stages=st,launch_pdl=True);return out,s
 suffix='_ppdl' if pdl else ''
 if t==1:
  rows,tile,warps=(2,1024,4) if role=='o' else (1,2048,4) if role=='unpooler_slice' else (1,4096,16);out=torch.empty((t,n),device=x.device,dtype=dtype);fn=globals()['_cbf_gemv_rows'+suffix];fn[(triton.cdiv(n,rows),)](x,w,out,k,n,x.stride(1),w.stride(0),ROWS=rows,TILE_K=tile,num_warps=warps,num_stages=1,launch_pdl=pdl);return out.unsqueeze(0),1
 if t<=32 and role in ('qkv_gate','qkv3','gate','unpooler_slice'):
  out=torch.empty((t,n),device=x.device,dtype=dtype);pp_cluster(x,w,out,(bn,bk,s,nw,st,bm),bank,pdl);return out.unsqueeze(0),1
 if t<=32:
  out=torch.empty((t,n),device=x.device,dtype=dtype);xd=TensorDescriptor.from_tensor(x,[bm,bk]);wp=bank['tiled'][role,bk] if role!='o' else w;wd=TensorDescriptor.from_tensor(wp,[bn,bk]);fn=globals()[('_cbf_skinny_gemm_tma_tiled' if role!='o' else '_cbf_skinny_gemm_tma')+suffix];fn[(triton.cdiv(t,bm),n//bn,1)](xd,wd,wd,out,out,out,t,n,k,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,SPLIT_K=1,HILO=False,OUT_F32=dtype==torch.float32,num_warps=nw,num_stages=st,launch_pdl=pdl);return out.unsqueeze(0),1
 if t==128 and role in ('qkv_gate','qkv3','gate'):
  out=torch.empty((s,t,n),device=x.device,dtype=torch.float32);xd=TensorDescriptor.from_tensor(x,[bm,bk]);wd=TensorDescriptor.from_tensor(bank['tiled'][role,bk],[bn,bk]);fn=globals()['_cbf_lm_partials_tiled'+suffix];fn[(n//bn,s,1)](xd,wd,out,out,t,n,k,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,SPLIT_K=s,num_warps=nw,num_stages=st,launch_pdl=pdl);return out,s
 if t==128 and role=='pool_router':out=torch.mm(w,x.T,out_dtype=torch.float32).T.contiguous()
 else:out=torch.mm(x,w.T,out_dtype=torch.float32 if role=='pool_router' else torch.bfloat16)
 return out.unsqueeze(0),1

@triton.jit
def _pp_gate_apply(P,Y,O,T:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
 tl.extra.cuda.gdc_launch_dependents();tl.extra.cuda.gdc_wait();col=tl.program_id(0)*B+tl.arange(0,B);s=tl.arange(0,triton.next_power_of_2(S));v=tl.load(P+s[:,None]*T*1024+col[None,:],mask=(s[:,None]<S)&(col[None,:]<T*1024),other=0).to(tl.float32);gate=tl.sigmoid(tl.sum(v,0));y=tl.load(Y+col,mask=col<T*1024,other=0).to(tl.float32);tl.store(O+col,(y*gate).to(tl.bfloat16),mask=col<T*1024)

def pp_gate(x,y,bank,provider):
 p,s=pp_project(x,'gate',bank,provider);out=torch.empty_like(y);_pp_gate_apply[(triton.cdiv(y.numel(),256),)](p,y,out,x.shape[0],s,B=256,num_warps=4,launch_pdl=True);return out
