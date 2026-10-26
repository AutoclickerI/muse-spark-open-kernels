DA_TIMES={}
DA_LOCAL_MM=None
DA_SMALL=((4,4,False),(4,2,False),(8,1,False),(4,4,True))
DA_WIDE=((0,4,1,False),(0,2,1,False),(64,4,1,False),(128,4,2,False),(0,4,1,True),(0,8,1,False),(0,1,1,False))

def da_vendor():
    ns=_nf_vendor()
    # Keep histogram and all 121 window sums on the GPU. Only the selected
    # eight-byte decoding table crosses to the host for the launch ABI.
    exec(compile(DA_GPU_PACK_SOURCE,'<P10 GPU packing>','exec'),ns)
    return ns

def da_small_source(ns,early=False,gate=False):
    src=ns['V18S_CUDA_TEMPLATE']
    wait='  asm volatile("griddepcontrol.wait;" ::: "memory");\n'
    assert src.count(wait)==1
    if early or gate:src=src.replace(wait,'',1)
    if early and not gate:
        needle='    for (int s = 0; s < ST; ++s) load(s, s);'
        assert src.count(needle)==1
        initial='''    for(int s=0;s<ST;++s)for(int j=0;j<RT;++j)for(int l=0;l<NL;++l) {
      int4 v;const int4* p=wq+(size_t)s*sstride+(j*NL+l)*32;
      asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");w[s][j][l]=v;
    }
'''
        src=src.replace(needle,initial+wait)
    if gate:
        # Gate X is the already-ready layer input. Delay its Y dependency until
        # the complete dot product is ready, preserving overlap with attention.
        src=src.replace('using OutT = float;','using OutT = __nv_bfloat16;')
        src=src.replace('void v18s_store(', 'void da_base_store(')
        signature='uint32_t U0, uint32_t U1) {'
        position=src.index(signature,src.index('__global__'))
        src=src[:position]+src[position:].replace(signature,'uint32_t U0, uint32_t U1, const __nv_bfloat16* gate_y) {',1)
        needle='  OutT* out = reinterpret_cast<OutT*>(outp);'
        src=src.replace(needle,needle+'''
  auto v18s_store = [&](OutT* p,float v) {
    const long offset=p-out;
    da_base_store(p,__bfloat162float(gate_y[offset])/(1.f+__expf(-v)));
  };''')
        needle='    if constexpr (RO) {\n      if constexpr (RO_CODE)'
        assert src.count(needle)==1;src=src.replace(needle,wait+needle)
        src=src.replace('int64_t stream, int query) {','int64_t stream, int query, int64_t gate_y) {')
        src=src.replace('int64_t u01, int64_t stream, int query) {','int64_t u01, int64_t stream, int query, int64_t gate_y) {')
        needle='(uint32_t)((uint64_t)u01 >> 32));';assert src.count(needle)==1
        src=src.replace(needle,'(uint32_t)((uint64_t)u01 >> 32), reinterpret_cast<const __nv_bfloat16*>(gate_y));')
    return src

def da_wide_source(ns,instances,early=False):
    src=ns['dd_decodediet_tt_mw_cuda_source'](instances)
    # Use the already-reviewed register-A mapping for wider token MMA shapes.
    before=ns['dd_decodediet_tt_mw_CUDA_CORE']
    for bn in (128,256):
        src=src.replace(before,ns['dd_decodediet_tt_mw_mma'](bn)+'\n'+before,1)
        before=ns['dd_decodediet_tt_mw_CUDA_CORE']
    needle='      if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));'
    assert src.count(needle)==1
    src=src.replace(needle,needle+'\n      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));\n      if constexpr(BN==256)mma256(acc,a,desc(base+s*SB));')
    # Disjoint K partitions write FP32 partials for the existing fused consumer.
    # This avoids a full-output DSM allocation at large M.
    src=src.replace('__nv_bfloat16* out,int m,int n,int k,int split','float* out,int m,int n,int k,int split')
    src=src.replace('int rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;','int rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;\n  out+=(long)rank*m*n;')
    src=src.replace('if(split>1)cluster_sync();','')
    src=src.replace('for(int q=tid+rank*NT;q<BN*(R/4);q+=NT*split)','for(int q=tid;q<BN*(R/4);q+=NT)')
    src=src.replace('for(int pp=0;pp<split*WK;pp++)','for(int pp=0;pp<WK;pp++)')
    src=src.replace('if(split>1)asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(remote):"r"(remote),"r"(s));','')
    old='''    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);
    *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);'''
    assert src.count(old)==1
    src=src.replace(old,'    *(float4*)(out+(long)(mt*BN+token)*n+nt*R+row)=sum;')
    src=src.replace('if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}','')
    src=src.replace('(__nv_bfloat16*)out,m,n,k,split','(float*)out,m,n,k,split')
    # One table per 16 weight rows handles pool/router hi/lo origins exactly.
    old='const uint32_t* el=esc+((nt*split*WK+rank*WK+wg)*RW+warp)*64;'
    assert src.count(old)==1
    src=src.replace(old,old.replace('*64;','*258;')+'\n    t0=el[256];t1=el[257];')
    src=src.replace('uint32_t e0=0xffffffffu,e1=0xffffffffu,nxt=0xffffffffu;int ep=0;','uint32_t e0=0xffffffffu,e1=0xffffffffu,nxt=0xffffffffu;int ep=0;const uint32_t* escape_list=nullptr;')
    src=src.replace('    e0=el[lane];e1=el[lane+32];','    escape_list=el;e0=el[lane];e1=el[lane+32];')
    old='''          uint32_t ev=ep<32?e0:e1;
          nxt=__shfl_sync(0xffffffffu,ev,ep&31);
          if(ep>=64)nxt=0xffffffffu;'''
    assert src.count(old)==1
    src=src.replace(old,'''          if(ep>=64 && ep<256 && (ep&31)==0)e1=escape_list[ep+lane];
          uint32_t ev=ep<32?e0:e1;
          nxt=__shfl_sync(0xffffffffu,ev,ep&31);
          if(ep>=256)nxt=0xffffffffu;''')
    start=src.index('  extern __shared__',src.index('__global__'))
    src=src[:start]+'  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'+src[start:]
    loadx='  for(int s=0;s<Q;s++)loadx(s);';needle='  for(int p=0;p<PF;p++)load(p,p);'
    assert src.count(loadx)==1 and src.count(needle)==1
    if early:
        initial='''  for(int p=0;p<PF;++p)if(p<nkb && warp<RW) {
    const uint4* first=w+((long)(nt*(k/64)+kb0+p)*RW+warp)*NL*32+lane;
    for(int l=0;l<NL;++l) {
      uint4 v;asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(first+l*32):"memory");raw[p][l]=v;
    }
  }
'''
        src=src.replace(loadx,'',1);src=src.replace(needle,initial+'\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n  '+loadx)
    else:
        src=src.replace(loadx,'  asm volatile("griddepcontrol.wait;" ::: "memory");\n'+loadx)
    old='  return int(cudaLaunchKernelEx(&cfg,kern,tx,';assert src.count(old)==1
    src=src.replace(old,'  cudaLaunchAttribute pdl{};pdl.id=cudaLaunchAttributeProgrammaticStreamSerialization;pdl.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&pdl;cfg.numAttrs=1;\n'+old)
    # GNU-unique template-local ready flags are process-wide across DSOs.
    # Give the two bodies distinct symbols so each kernel receives its own
    # >48-KiB shared-memory opt-in (BN256 needs 69632 bytes).
    symbol='p10_daw_early' if early else 'p10_daw_normal'
    src=src.replace('m32w_gemm',symbol+'_gemm').replace('launch_t',symbol+'_launch_t')
    return src

def da_build_small(ns,early,gate=False,local_reduce=False):
    insts=[];index={}
    for t in (1,8,16,32):
        for k in ((8192,) if gate else (1024,4096,8192)):
            for arm,(wk,s,e) in enumerate(DA_SMALL):
                if e!=early:continue
                if local_reduce:s=1
                ku=k//(64*wk*s);inst=(1,max(1,t//8),1,1,2 if s>1 else 0,0,wk*32,0,0,k,1,wk,ku,s,6)
                if inst not in insts:insts.append(inst)
                index[t,k,arm]=(insts.index(inst),(1,wk,ku,s))
    old=ns['V18S_CUDA_TEMPLATE'];ns['V18S_CUDA_TEMPLATE']=da_small_source(ns,early,gate)
    src=ns['v18s_cuda_source'](insts);ns['V18S_CUDA_TEMPLATE']=old
    if gate:
        src=src.replace('stream, query);','stream, query, gate_y);')
    cpp=ns['V18S_CPP']
    if gate:cpp=cpp.replace('int query);','int query,int64_t gate_y);')
    tag=hashlib.sha256(src.encode()).hexdigest()[:12];begin=time.monotonic()
    mod=load_inline(name='p10_da_small_'+tag,cpp_sources=cpp,cuda_sources=src,functions=['v18s_launch','v18s_router_launch'],extra_cuda_cflags=['-O3',ns['_nf_arch_flag']()],no_implicit_headers=True,with_pytorch_error_handling=False)
    DA_TIMES['compile_small_'+str((early,gate))]=time.monotonic()-begin
    return mod,index,src

def da_prepare(weights,local_reduce=False,wide_standin=False):
    begin=time.monotonic();ns=da_vendor();roles={'qkv_gate':weights['qkv_gate'],'qkv3':weights['qkv_gate'][:1280].contiguous(),'gate':weights['qkv_gate'][1280:].contiguous(),'o':weights['o'],'pool_router':torch.cat((weights['pooler'],weights['router'].bfloat16(),(weights['router']-weights['router'].bfloat16().float()).bfloat16())),'unpooler':weights['unpooler'],'unpooler_slice':weights['unpooler'][:1024].contiguous()}
    st=dict(roles=roles,small={},wide={},modules={},ns=ns,wide_standin=wide_standin)
    small={}
    for early in (False,True):
        for gate in (False,True):
            mod,index,src=da_build_small(ns,early,gate,local_reduce);small[early,gate]=(mod,index);st['modules']['small',early,gate]=(mod,src)
    packed={}
    for name,w in roles.items():
        b=time.monotonic();pk,tables,stream=_nf_group_pack(ns,w,'pool_router' if name=='pool_router' else name,'v18');torch.cuda.synchronize();DA_TIMES['small_pack_'+name]=time.monotonic()-b
        for t in (1,8,16,32):
            actual='unpooler_slice' if name=='unpooler' and t<=8 else name
            if actual!=name or name=='unpooler_slice' and t>8:continue
            for arm,(wk,s,early) in enumerate(DA_SMALL):
                mod,index=small[early,False];idx,cfg=index[t,w.shape[1],arm];wr,wk,ku,s=cfg;key=name,cfg
                if key not in packed:
                    a=time.monotonic();sp=stream(pk,1,*cfg)
                    if sp is None:raise RuntimeError('DA small escape overflow '+str(key))
                    g=w.shape[0]//16*s*wk;gw=torch.arange(g,device=w.device);tile=(gw//wk)%(w.shape[0]//16)
                    meta=torch.empty((g,34),device=w.device,dtype=torch.int32);meta[:,:32]=sp['esc'];meta[:,32:]=tables[tile].to(torch.int32);packed[key]=dict(sp,esc=meta)
                    torch.cuda.synchronize();DA_TIMES['small_stream_'+str(key)]=time.monotonic()-a
                st['small'][t,name,arm]=dict(packed[key],idx=idx,cfg=cfg,fn=mod.v18s_launch)
                if name=='gate':
                    gm,gi=small[early,True];j,_=gi[t,w.shape[1],arm];st['small'][t,'gate_apply',arm]=dict(packed[key],idx=j,cfg=cfg,fn=gm.v18s_launch)
    insts=[];wi={}
    for t in (128,256):
        for arm,(bn,s,pf,early) in enumerate(DA_WIDE):
            inst=(bn or t,64,1,pf,2,True,2)
            if inst not in insts:insts.append(inst)
            wi[t,arm]=insts.index(inst)
    for early in (False,True):
        src=da_wide_source(ns,insts,early);tag=hashlib.sha256(src.encode()).hexdigest()[:12];a=time.monotonic()
        mod=None if wide_standin else load_inline(name='p10_da_wide_'+tag,cpp_sources=ns['dd_decodediet_tt_mw_CPP'],cuda_sources=src,functions=['launch','info'],extra_cuda_cflags=['-O3','-std=c++20',ns['_nf_arch_flag']()],extra_ldflags=['-lcuda'],no_implicit_headers=True,with_pytorch_error_handling=False)
        DA_TIMES['compile_wide_'+str(early)]=time.monotonic()-a;st['modules']['wide',early]=(mod,src)
    wide_packs={}
    for name,w in roles.items():
        if name=='unpooler_slice':continue
        a=time.monotonic();groups=(w[:4096],w[4096:4352],w[4352:]) if name=='pool_router' else (w,);base_packs=[]
        for group in groups:
            p=ns['dd_decodediet_tt_mw_pack_weight'](group.contiguous(),64,1,1,True)
            if p is None:raise RuntimeError('DA wide exact escape capacity '+name)
            base_packs.append(p)
        # The packed body is independent of split count. Re-encoding or storing
        # it for each geometry wasted both prepare time and memory.
        wp=torch.cat([p['w'] for p in base_packs]);torch.cuda.synchronize();DA_TIMES['wide_pack_'+name]=time.monotonic()-a
        for s in (1,2,4,8):
            a=time.monotonic();esc=[];metadata_requests=0
            for p in base_packs:
                old=p['esc'];g=torch.arange(old.shape[0],device=w.device)[:,None].expand_as(old);mask=old!=-1;e=old[mask].long();gg=g[mask].long()
                seq=e>>18;rank=seq//(p['k']//64//s);newseq=seq-rank*(p['k']//64//s);newg=(gg//4*s+rank)*4+gg%4
                order=torch.argsort(newg*16384+newseq);newg=newg[order];entry=((newseq<<18)|(e&((1<<18)-1)))[order].to(torch.int32)
                pos=torch.arange(e.numel(),device=w.device)-torch.searchsorted(newg,newg);counts=torch.bincount(newg,minlength=old.shape[0]*s)
                if int(counts.max())>256:raise RuntimeError('DA split escape capacity '+str((name,s)))
                metadata_requests+=int((((counts+31)//32).clamp(min=2)*128+8).sum())
                meta=torch.empty((old.shape[0]*s,258),device=w.device,dtype=torch.int32);meta[:,:256]=-1;meta[newg,pos]=entry;tab=p['table'];meta[:,256]=tab&0xffffffff;meta[:,257]=tab>>32;esc.append(meta)
            wide_packs[name,s]=dict(wp=wp,esc=torch.cat(esc),splits=s,n=w.shape[0],k=w.shape[1],metadata_requests=metadata_requests)
            torch.cuda.synchronize();DA_TIMES['wide_metadata_'+str((name,s))]=time.monotonic()-a
        for t in (128,256):
            for arm,(bn,s,pf,early) in enumerate(DA_WIDE):
                st['wide'][t,name,arm]=dict(wide_packs[name,s],idx=wi[t,arm],mod=st['modules']['wide',early][0])
    DA_TIMES['total']=time.monotonic()-begin;st['times']=dict(DA_TIMES);return st

def da_project(x,name,bank,arm,gate_y=None):
    t,k=x.shape;role='unpooler_slice' if name=='unpooler' and t<=8 else name
    if t<=32:
        cap=1 if t==1 else 8 if t<=8 else 16 if t<=16 else 32;p=bank['small'][cap,'gate_apply' if gate_y is not None else role,arm];n=bank['roles'][role].shape[0]
        out=torch.empty((t,n),device=x.device,dtype=torch.bfloat16 if gate_y is not None else torch.float32)
        args=(p['idx'],p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),t,n,k,*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream().cuda_stream,0)
        if gate_y is not None:args+= (gate_y.data_ptr(),)
        rc=p['fn'](*args)
        if rc:raise RuntimeError('DA small launch '+str((t,name,arm,rc)))
        return out if gate_y is not None else (out.unsqueeze(0),1)
    p=bank['wide'][t,role,arm];s=p['splits'];out=torch.empty((s,t,p['n']),device=x.device,dtype=torch.float32)
    if bank['wide_standin']:
        out.zero_();DA_LOCAL_MM(x,bank['roles'][role],out[0])
    else:
        rc=p['mod'].launch(p['idx'],x.data_ptr(),p['wp'].data_ptr(),p['esc'].data_ptr(),out.data_ptr(),t,p['n'],k,s,0,torch.cuda.current_stream().cuda_stream)
        if rc:raise RuntimeError('DA wide launch '+str((t,name,arm,rc)))
    return out,s

@triton.jit
def _da_reduce(P,O,T:tl.constexpr,N:tl.constexpr,S:tl.constexpr,PS:tl.constexpr,RS:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);r=i//N;c=i%N;v=tl.full((B,),0,tl.float32)
    for s in range(S):v+=tl.load(P+s*PS+r*RS+c,r<T,0).to(tl.float32)
    tl.store(O+i,v,i<T*N)

def da_materialize(pair):
    p,s=pair;t,n=p.shape[1:]
    if s==1 and p.dtype==torch.bfloat16:return p[0]
    out=torch.empty((t,n),device=p.device,dtype=torch.bfloat16)
    _da_reduce[(triton.cdiv(t*n,256),)](p,out,t,n,s,p.stride(0),p.stride(1),256,num_warps=4)
    return out
