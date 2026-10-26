# Adapted from reviewed MIT operatorstack kernel; full notice in parent port_body.py.inc.
@triton.jit
def _ng_cluster_pool(x_desc, w_desc, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, SLOT_KB: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    pid_k = tl.program_id(0)  # rank in the cluster
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2)
    if SLOT_KB == 8:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 cbf_slots[8192]; mov.u32 $0, cbf_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    else:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 cbf_slots[16384]; mov.u32 $0, cbf_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    started = tl.inline_asm_elementwise("barrier.cluster.arrive.relaxed.aligned; mov.u32 $0, 0;", "=r,r", [pid_k],
                                        dtype=tl.int32, is_pure=False, pack=1)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    # Peel the first weight tile. The opaque zero returned after GDC makes all
    # activation descriptor coordinates depend on the wait, including pipelined
    # future iterations. A source-only wait inside the loop permits TMA hoisting.
    wt0 = w_desc.load([(pid_n * kb_total + kb0) * BLOCK_N, 0])
    first = tl.gather(tl.reshape(wt0.to(tl.uint16,bitcast=True),(BLOCK_N*BLOCK_K,)),tl.full((1,),0,tl.int32),axis=0)
    witness = tl.sum(first.to(tl.int32),0)
    ready = tl.inline_asm_elementwise("griddepcontrol.wait; and.b32 $0, $1, 0;", "=r,r", [witness], dtype=tl.int32, is_pure=False, pack=1)
    xm = tl.multiple_of(m0 + ready, BLOCK_M)
    xt0 = x_desc.load([xm, k]).T
    acc = tl.dot(wt0, xt0, tl.zeros((BLOCK_N, BLOCK_M),dtype=tl.float32))
    k += BLOCK_K
    for _ in range(kb0+1, kb1):
        wt = w_desc.load([(pid_n * kb_total + k // BLOCK_K) * BLOCK_N, 0])
        xt = x_desc.load([xm, k]).T
        acc = tl.dot(wt, xt, acc)
        k += BLOCK_K
    idx4 = (tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]) * 4
    TILE_BYTES: tl.constexpr = BLOCK_N * BLOCK_M * 4
    if pid_k != 0:
        tl.inline_asm_elementwise("st.shared.f32 [$1], $2; mov.u32 $0, 0;", "=r,r,f", [base + idx4, acc],
                                  dtype=tl.int32, is_pure=False, pack=1)
        tl.debug_barrier()
        ready = tl.inline_asm_elementwise("barrier.cluster.wait.aligned; mov.u32 $0, 0;", "=r,r", [pid_k + started],
                                          dtype=tl.int32, is_pure=False, pack=1)
        dst = base + (pid_k - 1) * TILE_BYTES + ready
        if TILE_BYTES == 4096:
            tl.inline_asm_elementwise(CBF_V4COPY2, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        else:
            tl.inline_asm_elementwise(CBF_V4COPY4, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned; mov.u32 $0, 0;", "=r,r", [pid_k],
                                  dtype=tl.int32, is_pure=False, pack=1)
    else:
        ready = tl.inline_asm_elementwise("barrier.cluster.wait.aligned; mov.u32 $0, 0;", "=r,r", [pid_k + started],
                                          dtype=tl.int32, is_pure=False, pack=1)
        done = tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned; "
                                         "barrier.cluster.wait.acquire.aligned; mov.u32 $0, 0;", "=r,r",
                                         [pid_k + ready], dtype=tl.int32, is_pure=False, pack=1)
        tot = acc
        for s in tl.static_range(1, SPLIT_K):
            tot += tl.inline_asm_elementwise("ld.shared.f32 $0, [$1];", "=f,r",
                                             [base + (s - 1) * TILE_BYTES + done + idx4], dtype=tl.float32,
                                             is_pure=False, pack=1)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
        tl.store(out_ptrs, tot, mask=offs_m[None, :] < M)

# Original MIT NF v18 exact payload/escape patcher, new shape-specific DSM/PDL schedule.
NG_CONFIG={
 (16,'qkv_gate'):(4,4,1),(16,'qkv3'):(4,4,1),
 (32,'qkv_gate'):(4,4,1),(32,'qkv3'):(4,4,1),
 (16,'o'):(1,4,2),(16,'pool_router'):(4,4,1),(32,'pool_router'):(4,4,1),
 (16,'unpooler'):(2,4,1),(32,'unpooler'):(2,4,1)}
NG_GROUPS={
 'Q16':[(16,'qkv_gate'),(16,'qkv3')],'Q32':[(32,'qkv_gate'),(32,'qkv3')],
 'O16':[(16,'o')],'P16':[(16,'pool_router')],'P32':[(32,'pool_router')],
 'U16':[(16,'unpooler')],'U32':[(32,'unpooler')]}

def ng_prepare(m,state):
 start=time.monotonic();ns=m['_nf_vendor']();exec(compile(PP_GPU_PACK_SOURCE,'<existing GPU NF histogram>','exec'),ns);original=ns['v18s_cuda_source']
 def source(insts):
  text=original(insts);wait='  asm volatile("griddepcontrol.wait;" ::: "memory");';assert text.count(wait)==1
  text=text.replace(wait,'');anchor='    for (int s = 0; s < ST; ++s) load(s, s);';assert text.count(anchor)==1
  return text.replace(anchor,anchor+'\n    asm volatile("griddepcontrol.wait;" ::: "memory");')
 ns['v18s_cuda_source']=source;insts=[];keys={};times={};roles={r:state[r] for r in ('qkv_gate','qkv3','o','pool_router','unpooler')}
 for (t,role),(s,wk,stages) in NG_CONFIG.items():
  n,k=roles[role].shape;assert k%(s*wk*64)==0;keys[t,role]=len(insts);insts.append((1,t//8,1,stages,3 if s>1 else 0,0,wk*32,0,n,k,1,wk,k//(s*wk*64),s,6))
 t0=time.monotonic();mod=ns['v18s_build'](insts,'p10_night_dsm_'+hashlib.sha256(source(insts).encode()).hexdigest()[:12]);times['compile_NF']=time.monotonic()-t0;plans={};streams={}
 for role in sorted({r for t,r in NG_CONFIG}):
  t0=time.monotonic();packed,tables,stream=m['_nf_group_pack'](ns,roles[role],role,'v18');times['pack_'+role]=time.monotonic()-t0
  for (t,r),(s,wk,stages) in NG_CONFIG.items():
   if r!=role:continue
   k=roles[r].shape[1];cfg=(1,wk,k//(s*wk*64),s);key=(role,cfg)
   if key not in streams:
    t0=time.monotonic();p=stream(packed,1,*cfg);assert p is not None,('NF exact escape capacity',key);count=p['esc'].shape[0];gw=torch.arange(count,device=roles[role].device);gx=roles[role].shape[0]//16;tile=(gw//wk)%gx;meta=torch.empty((count,34),device=gw.device,dtype=torch.int32);meta[:,:32]=p['esc'];meta[:,32:]=tables[tile].int();streams[key]=dict(p,esc=meta,cfg=cfg);times['stream_'+str(key)]=time.monotonic()-t0
   plans[t,role]=dict(streams[key],idx=keys[t,role],n=roles[r].shape[0],fn=mod.v18s_launch)
 t0=time.monotonic();w=roles['pool_router'];tma_w=w.reshape(72,64,64,128).permute(0,2,1,3).contiguous().view(-1,128);torch.cuda.synchronize();times['tile_pool128']=time.monotonic()-t0
 return dict(roles=roles,tma_w=tma_w,plans=plans,mod=mod,cluster={},prepare_s=time.monotonic()-start,times=times,insts=insts,source_sha256=hashlib.sha256(source(insts).encode()).hexdigest())

def ng_project(x,role,bank,pool32_coded=False):
 t,k=x.shape
 if (t,role)==(32,'pool_router') and not pool32_coded:return ng_tma_pool(x,bank)
 p=bank['plans'][t,role];out=torch.empty((t,p['n']),device=x.device,dtype=torch.float32);rc=p['fn'](p['idx'],p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),t,p['n'],k,*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream().cuda_stream,0)
 if rc:raise RuntimeError('night coded launch '+str((t,role,rc)))
 return out.unsqueeze(0),1

def ng_tma_pool(x,bank):
 from triton import knobs
 w=bank['roles']['pool_router'];t,k=x.shape;n=w.shape[0];bm,bn,bk,s,nw,st,slot=32,64,128,3,4,2,16
 xd=TensorDescriptor.from_tensor(x,[bm,bk]);wd=TensorDescriptor.from_tensor(bank['tma_w'],[bn,bk]);out=torch.empty((t,n),device=x.device,dtype=torch.float32);grid=(1,n//bn,1);args=(xd,wd,out,t,n,k,bm,bn,bk,s,slot)
 if 'pool32' not in bank['cluster']:
  ck=_ng_cluster_pool.warmup(*args[:6],BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,SPLIT_K=s,SLOT_KB=slot,num_warps=nw,num_stages=st,launch_pdl=True,grid=grid);ck._init_handles();assert ck.run.launch_pdl is True;pm=list(ck.packed_metadata);assert len(pm)==3 and pm[1]==1 and pm[2]>49152,pm;pm[1]=s;bank['cluster']['pool32']=(ck,tuple(pm))
 ck,pm=bank['cluster']['pool32'];stream=torch.cuda.current_stream().cuda_stream;ck.run(*grid,stream,ck.function,pm,ck.launch_metadata(grid,stream,*args),knobs.runtime.launch_enter_hook,knobs.runtime.launch_exit_hook,*args)
 return out.unsqueeze(0),1
