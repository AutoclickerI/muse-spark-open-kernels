"""Asserted source transforms: one retained consumer-owned stage, no L2 hint kernels."""
import ast,re

def ew_patch_nf_small(src):
    wait='  asm volatile("griddepcontrol.wait;" ::: "memory");\n'
    if '    auto load = [&](int slot, int s) {' not in src:return src,'no small stage'
    assert src.count(wait)==1
    # Kernels with an activation cp.async prologue (XM) need the wait at that
    # prologue; those are qkv-only: their X is the external layer input.
    src=src.replace(wait,'',1)
    start=src.index('    auto load = [&](int slot, int s) {')
    end=src.index('    };',start)+len('    };')
    old=src[start:end];split=old.index('      if constexpr (!NOX')
    weight=old[:split].replace('auto load =','auto loadw =',1)+'    };'
    act='    auto loadx = [&](int slot,int s) {\n'+old[split:old.rindex('    };')]+'    };'
    new=weight+'\n'+act+'\n    auto load=[&](int slot,int s){loadw(slot,s);loadx(slot,s);};'
    src=src[:start]+new+src[end:]
    marker='    for (int s = 0; s < ST; ++s) load(s, s);'
    assert marker in src
    order=re.search(r'\b(\w+_order_point)\(m, outp\)',src).group(1)
    if 'v18d_native_gemv' in src:
        # QKV3's register stage increased79->90 registers (six->five CTAs).
        # A first-stage smem landing keeps the data out of live registers at
        # the acquire. Each thread consumes its own completed copies.
        first='''    __shared__ __align__(16) int4 ew_w[NTMAX*RT*NL];
#pragma unroll
    for(int j=0;j<RT;++j)
#pragma unroll
      for(int l=0;l<NL;++l){
        unsigned dst=(unsigned)__cvta_generic_to_shared(ew_w+(j*NL+l)*NTMAX+tid);
        const int4* ptr=wq+(j*NL+l)*32;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(dst),"l"(ptr):"memory");
      }
    asm volatile("cp.async.commit_group;":::"memory");
'''+wait+'''    asm volatile("cp.async.wait_group 0;":::"memory");
#pragma unroll
    for(int j=0;j<RT;++j)
#pragma unroll
      for(int l=0;l<NL;++l)w[0][j][l]=ew_w[(j*NL+l)*NTMAX+tid];
'''
    else:first='    loadw(0,0);\n    '+order+'(m,outp);\n'+wait
    src=src.replace(marker,first+'    loadx(0,0);\n#pragma unroll\n    for(int s=1;s<ST;++s)load(s,s);',1)
    return src,'one first weight stage before acquire'

def ew_patch_nf_mw(src):
    wait='  asm volatile("griddepcontrol.wait;" ::: "memory");\n';assert src.count(wait)==1
    src=src.replace(wait,'',1)
    # Delay all activation TMA; the first raw weight fragment is retained.
    first='  for(int s=0;s<Q;s++)loadx(s);';assert first in src;src=src.replace(first,'',1)
    mark='  for(int p=0;p<PF;p++)load(p,p);';assert mark in src
    # A named ordering branch pins global issue before ACQBULK as in W.
    helper='__device__ __noinline__ void ew_never(){asm volatile("trap;");}\n'
    src=helper+src
    src=src.replace(mark,'  load(0,0);\n  unsigned ew_clock;asm volatile("mov.u32 %0,%%clock;":"=r"(ew_clock));\n  if((ew_clock&(unsigned)m)==0x7fffffffu)ew_never();\n'+wait+'  for(int s=0;s<Q;s++)loadx(s);\n  for(int p=1;p<PF;p++)load(p,p);',1)
    return src

def ew_patch_nf_wide(src):
    wait='  asm volatile("griddepcontrol.wait;" ::: "memory");\n';assert src.count(wait)==1
    if 'auto load=[&](int i)' not in src:return src,'wide shape has a different transport'
    # The incumbent OL sources use <<<>>>; its old port injected GDC but
    # never enabled a dependent launch. Keep the exact launch geometry and
    # give each changed kernel its own host-template configuration guard.
    launch='  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);\n  return int(cudaGetLastError());'
    assert src.count(launch)==1
    src=src.replace(launch,'''  cudaLaunchConfig_t ew_cfg{};
  ew_cfg.gridDim=dim3(8192/R,(m+T-1)/T);ew_cfg.blockDim=dim3(128*TW+((PIPE&2)?128:0));
  ew_cfg.dynamicSmemBytes=SMEM;ew_cfg.stream=(cudaStream_t)stream;
  return int(nf_port_launch(&ew_cfg,fn,tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m));''',1)
    src=src.replace('cudaLaunchAttribute at[8];','cudaLaunchAttribute at[8]{};')
    for name in set(re.findall(r'\b\w*ol_(?:projection|launch_t)\b',src)):
        if name.endswith('launch_t'):
            # The vendor emits the switch cases after this template transform.
            # A macro renames those generated references as well as the guard.
            src='#define '+name+' '+name+'_ew\n'+src
        else:src=re.sub(r'\b'+name+r'\b',name+'_ew',src)
    src=src.replace(wait,'',1)
    # Weight and input TMA share one completion barrier. Issue the first weight
    # on that existing slot, wait, then issue X and all later complete slots.
    i=src.index('  auto load=[&](int i)');j=src.index('\n  };',i)+5;part=src[i:j]
    lines=part.splitlines();xline=next(l for l in lines if re.search(r'\b\w+_tma\(base\+s\*SB\+AB,&tx',l))
    part=part.replace('auto load=[&](int i)','auto load=[&](int i,bool ew_x=true,bool ew_w=true)')
    part=part.replace('    if constexpr(PACK==1)', '    if(ew_w)if constexpr(PACK==1)')
    part=part.replace(xline,'    if(ew_x)'+xline.strip())
    # expect the full stage only once, even when issuing W and X separately.
    part=part.replace('    asm volatile("mbarrier.arrive.expect_tx', '    if(ew_w)asm volatile("mbarrier.arrive.expect_tx')
    src=src[:i]+part+src[j:]
    mark='    for(int i=0;i<ST && i<IT;i++) load(i);';assert mark in src
    src=src.replace(mark,'    load(0,false,true);',1)
    mark='  __syncthreads();';loc=src.index(mark,src.index('auto load='))
    src=src[:loc]+mark+'\n'+wait+'  if(tid==((PIPE&2)?CT:0)){load(0,true,false);for(int i=1;i<ST && i<IT;i++)load(i);}\n'+src[loc+len(mark):]
    return src,'first weight TMA in existing smem slot'

def ew_patch_glue(src,phases=('b1','b2')):
    # Only boundary_cl's first per-thread vector group is staged. No full-row
    # register policy and no change to the reduction/cluster ownership.
    a=src.index('__global__ void __launch_bounds__(NT) boundary_cl');b=src.index('// MoERouteMeta',a) if '// MoERouteMeta' in src[a:] else src.index('template <typename K>',a)
    original=src[a:b];body=original
    mark='  float4 a[G], b[G], w[G], q[G];';assert mark in body
    body=body.replace(mark,'''  __shared__ __align__(16) float4 ew_stage[4][NT];
  const int ew_j=(rank*NT*G+tid)*4;
  auto ew_cp=[&](int slot,const float* p){asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"((unsigned)__cvta_generic_to_shared(&ew_stage[slot][tid])),"l"(p):"memory");};
  ew_cp(0,alpha+ew_j);ew_cp(1,beta+ew_j);ew_cp(2,prew+ew_j);
  if(postw)ew_cp(3,postw+ew_j);
  asm volatile("cp.async.commit_group;":::"memory");
  pdl_wait();
  asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
  float4 a[G], b[G], w[G], q[G];''')
    # Remove the later acquire, preserving the early one inserted above.
    at=body.index('  float4 a[G]');head,tail=body[:at],body[at:];tail=tail.replace('  pdl_wait();\n','',1)
    for slot,name in enumerate(('alpha','beta','prew','postw')):
        mark='gp_load4('+name+' + j)';assert tail.count(mark)==1
        tail=tail.replace(mark,f'(g==0?ew_stage[{slot}][tid]:gp_load4({name}+j))',1)
    changed='template <int C, int NT, int ROUNDS>\n'+(head+tail).replace(' boundary_cl(', ' ew_boundary_cl(',1)
    src=src[:a]+original+changed+src[b:]
    pred='true' if set(phases)=={'b1','b2'} else '!postw' if phases==('b1',) else 'postw'
    mark='glue_cl::boundary_cl<C_, NT_, R_>';assert src.count(mark)==1
    return src.replace(mark,'(('+pred+') ? glue_cl::ew_boundary_cl<C_, NT_, R_> : glue_cl::boundary_cl<C_, NT_, R_>)',1)

def ew_patch_attention(src):
    # attn_main's history excludes new tokens, so its first K/V stage can issue
    # into its existing smem before waiting for QKV; no added ring slots.
    if 'auto load_rows = ' not in src:return src,'already early producer or library'
    old='  asm volatile("griddepcontrol.wait;" ::: "memory");'
    a=src.index('__global__');p=src.index(old,a);src=src[:p]+'''  if(ntiles>0)issue(0,rows[0]);
  cp_commit();
'''+src[p:]
    src=src.replace('if (s < ntiles) issue(s, rows[s]);','if (s>0 && s < ntiles) issue(s, rows[s]);')
    return src,'first historical K/V stage before QKV acquire'

def ew_patch_publish_routes(src):
    mark='  if (meta != nullptr && rank == 0)'
    assert mark in src
    src=src.replace(mark,'''  if(meta==nullptr && arrive!=nullptr && rank==0){
    __syncthreads();
    if(tid==0)asm volatile("st.release.gpu.global.u32 [%0],1;"::"l"(arrive+t):"memory");
  }
'''+mark,1)
    # Ready-dependent addresses cannot be prefetched until every route CTA has
    # reached its publication point. Admission at entry can let spinning MoE
    # CTAs occupy resources still needed by the producer in a long PDL chain.
    a=src.index('__global__ void __launch_bounds__(NT) route_cl')
    b=src.index('template <typename K>',a)
    body=src[a:b];assert body.count('  pdl_release();')==1
    body=body.replace('  pdl_release();','',1)
    close=body.rindex('\n}')
    body=body[:close]+'\n  pdl_release();'+body[close:]
    return src[:a]+body+src[b:]

def ew_patch_nt1(src,w13=False,w2=False):
    if w13:
        mark='int M, int local_start, int nstage, uint64_t* __restrict__ trace) {';assert src.count(mark)==1
        src=src.replace(mark,mark.replace('trace) {','trace, const int* ew_ready) {'))
        mark='  if constexpr (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");';assert src.count(mark)==1
        src=src.replace(mark,'''  if constexpr(PDL){
    if(ew_ready){if(tid==0)for(int r=0;r<M;r++)while(ld_acquire(ew_ready+r)==0)__nanosleep(32);__syncthreads();}
    else asm volatile("griddepcontrol.wait;":::"memory");
  }''',1)
        mark='  if constexpr (TRACE) {\n    if (tid == 0) {\n      probe[1] = tg();';assert mark in src
        src=src.replace(mark,'  if constexpr(PDL)if(ew_ready)asm volatile("griddepcontrol.wait;":::"memory");\n'+mark,1)
        mark='(uint64_t*)(v.size() > 13 ? v[13] : 0)))';assert mark in src
        src=src.replace(mark,'(uint64_t*)(v.size() > 13 ? v[13] : 0), (const int*)(v.size()>14?v[14]:0)))')
    if w2:
        mark='    if (pass != cur_pass || e != cur_e || nb != cur_nb) {';assert mark in src
        src=src.replace(mark,'    if(pass==1)mbar_wait(&fullb[st],ph);\n'+mark,1)
        mark='    mbar_wait(&fullb[st], ph);';assert src.count(mark)==1
        src=src.replace(mark,'    if(pass==0)mbar_wait(&fullb[st], ph);',1)
    return src

def ew_patch_heavy(src,w13=False,w2=False):
    # Routing/count/order is completed by the existing fork event before the
    # heavy branch starts. W2 can therefore read it while W13 computes ACT.
    a=src.index('template<int NT,bool F,int CS,int S>__device__');b=src.index('template<int DEPTH,bool F>__global__',a)
    original_tile=src[a:b]
    part=original_tile
    i=part.index(' auto issue=[&](int k,int st){');j=part.index('\n };',i)+4
    old=part[i:j];split=old.index('  if(t<64){')
    inp=old[:split].replace('auto issue=','auto issuex=',1)+' };'
    weight=' auto issuew=[&](int k,int st){\n'+old[split:old.rindex('  asm volatile("cp.async.commit_group;')]+ ' };'
    new=inp+'\n'+weight+'\n auto issue=[&](int k,int st){issuex(k,st);issuew(k,st);asm volatile("cp.async.commit_group;":::"memory");};'
    part=part[:i]+new+part[j:]
    mark=' for(int p=0;p<(S>1?S-1:1);++p)issue(p,p);';assert mark in part
    condition='true' if w13 and w2 else 'F' if w13 else '!F'
    part=part.replace(mark,''' if constexpr('''+condition+'''){
  issuew(0,0);asm volatile("cp.async.commit_group;":::"memory");
  asm volatile("griddepcontrol.wait;":::"memory");
  issuex(0,0);asm volatile("cp.async.commit_group;":::"memory");
  for(int p=1;p<(S>1?S-1:1);++p)issue(p,p);
 }else{for(int p=0;p<(S>1?S-1:1);++p)issue(p,p);}''',1)
    part=part.replace('void tile(', 'void ew_tile(',1)
    src=src[:a]+original_tile+part+src[b:]
    selected='true' if w13 and w2 else 'F' if w13 else '!F'
    mark='#define CH_TILE(NT) tile<NT,F,1,S>'
    assert mark in src
    src=src.replace(mark,'#define CH_TILE(NT) ew_dispatch<NT,F,1,S>',1)
    # Keep the unselected phase compiled from the exact original tile body.
    dispatch='''template<int NT,bool F,int CS,int S>__device__ __forceinline__ void ew_dispatch(
 const int*Q,const unsigned char*SC,const float*GS,const __nv_bfloat16*X,const int*C,const int*OFF,const int*SLOTS,int M,const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int e,int row,int mt,uint64_t*bar,const CUtensorMap*map,unsigned scale_base,float*red){
 if constexpr('''+selected+''')ew_tile<NT,F,CS,S>(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS,e,row,mt,bar,map,scale_base,red);
 else tile<NT,F,CS,S>(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS,e,row,mt,bar,map,scale_base,red);
}
''';at=src.index('template<int DEPTH,bool F>__global__');src=src[:at]+dispatch+src[at:]
    a=src.index('template<int DEPTH,bool F>__global__');pos=src.index(' constexpr int S=',a)
    if w2:src=src[:pos]+' if constexpr(F)asm volatile("griddepcontrol.launch_dependents;":::"memory");\n'+src[pos:]
    mark='#define CH_GO(S,F) chunk::kernel<S,F><<<grid,256,bytes,(cudaStream_t)stream>>>(BR_ARGS(v),(int)M)';assert mark in src
    # The same dimensions/bytes and arguments; only the launch dependency type changes.
    replacement='''cudaLaunchConfig_t ec{};ec.gridDim=dim3(grid);ec.blockDim=dim3(256);ec.dynamicSmemBytes=bytes;ec.stream=(cudaStream_t)stream;
 cudaLaunchAttribute ea{};ea.id=cudaLaunchAttributeProgrammaticStreamSerialization;ea.val.programmaticStreamSerializationAllowed=1;ec.attrs=&ea;ec.numAttrs=1;
#define CH_GO(S,F) BR_CUDA_CHECK(cudaLaunchKernelEx(&ec,chunk::kernel<S,F>,BR_ARGS(v),(int)M))'''
    replacement=replacement.replace('ea.val.programmaticStreamSerializationAllowed=1;', 'ea.val.programmaticStreamSerializationAllowed=(first?'+str(int(bool(w13)))+':'+str(int(bool(w2)))+');')
    src=src.replace(mark,replacement,1)
    return src

def ew_patch_glue_triton(fn,phases=('b1','b2')):
    start=fn.index('    alpha=_pa_f32(ALPHA+j);beta=_pa_f32(BETA+j);prew=_pa_f32(PREW+j)')
    stop=fn.index('    _pdl_wait(USE_PDL)',start)+len('    _pdl_wait(USE_PDL)')
    new='''    ew_a,ew_b,ew_c,ew_d=_ew_immutable_stage(ALPHA,BETA,PREW,POSTW if HAS_POSTW else ALPHA)
    ew_idx=tl.minimum(j,127)
    alpha=tl.where(j<128,tl.gather(ew_a,ew_idx,0),tl.load(ALPHA+j,j>=128,0.))
    beta=tl.where(j<128,tl.gather(ew_b,ew_idx,0),tl.load(BETA+j,j>=128,0.))
    prew=tl.where(j<128,tl.gather(ew_c,ew_idx,0),tl.load(PREW+j,j>=128,0.))
    if HAS_POSTW:postw=tl.where(j<128,tl.gather(ew_d,ew_idx,0),tl.load(POSTW+j,j>=128,0.))'''
    old=fn[start:stop]
    if set(phases)!={'b1','b2'}:
        pred='not HAS_POSTW' if phases==('b1',) else 'HAS_POSTW'
        new='    if '+pred+':\n'+'\n'.join('    '+line for line in new.splitlines())+'\n    else:\n'+'\n'.join('    '+line for line in old.splitlines())
    return fn[:start]+new+fn[stop:]
