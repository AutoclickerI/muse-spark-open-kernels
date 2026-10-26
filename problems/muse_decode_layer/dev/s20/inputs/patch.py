"""Exact P7 nominated transforms, snapshotted for S20 reproducibility."""
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
