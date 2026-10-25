_dc_base_nf_vendor=_nf_vendor
_dc_base_nf_prepare=nf_prepare
_dc_base_nf_project=nf_project


def dc_nf_source(src,family):
    """Only replace native packed-W transport; preserve decoder/MMA/X/reducer."""
    if family=='mw':
        old='#pragma unroll\n      for(int l=0;l<NL;l++)raw[slot][l]=ldw(src+l*32,pw);'
        assert old in src
        src=src.replace(old,'dc_read(w,((nt*(k/64)+kb0+i)*RW+warp),lane,(int4*)raw[slot]);')
    elif family=='cp':
        a=src.index('#pragma unroll\n          for (int j=0; j<RT; ++j)',src.index('auto stage_weights'))
        b=src.index('\n        }',a)
        src=src[:a]+'''#pragma unroll
          for(int j=0;j<RT;++j) dc_stage(wp,(si*G+gw)*RT+j,lane,rawring+((slot*NW+warp)*RT+j)*NL*32+lane);'''+src[b:]
        # The shared third vector contains compact a/b/c plus its extension mask.
        old='          uint32_t A[RT][NA * 16];'
        assert old in src
        src=src.replace(old,'''          if constexpr(CP>0) {
#pragma unroll
            for(int j=0;j<RT;++j) {int4 z=w[s][j][2];dc_expand(z.x,z.y,z.z,z.w,((const unsigned*)wp)[4],1,&w[s][j][2]);}
          }
'''+old)
    else:
        pattern=r'#pragma unroll\s+for \(int l = 0; l < NL; \+\+l\)([^\n;]*?)w\[slot\]\[j\]\[l\] = \w+_ldw\(wq \+ \(size_t\)s \* sstride \+ \(j \* NL \+ l\) \* 32\);'
        def replace(m):
            guard=m[1].strip()
            return (guard+' ' if guard else '')+'dc_read(wp,(s*G+gw)*RT+j,lane,w[slot][j]);'
        src,count=re.subn(pattern,replace,src);assert count==1,(family,count)
    return DC_NATIVE+'\n'+src


def dc_nf_vendor():
    ns=_dc_base_nf_vendor()
    for name,family in (('v18d_SM_CUDA_TEMPLATE','d9'),('V18S_CUDA_TEMPLATE','v18'),('tt_v10_sm2_SMALLM2_CUDA_TEMPLATE','sm2'),
        ('tt_v12_smtrace1_SMTRACE_CUDA_TEMPLATE','u1'),('tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE','u8'),
        ('dd_decodediet_CPSSM_CUDA_TEMPLATE','cp'),('BUNDLE_QKV_CUDA','q8'),('tt_v10_m32x__Q_CUDA','q32'),
        ('b2q_tt_v10_m32x__Q_CUDA','q16'),('b2p_tt_v10_m32x__CUDA','p32')):
        ns[name]=dc_nf_source(ns[name],family)
    cpp,cuda,name=ns['mt_M16TRACE_SOURCES']['o'];ns['mt_M16TRACE_SOURCES']['o']=(cpp,dc_nf_source(cuda,'o16'),name)
    mw=ns['dd_decodediet_tt_mw_cuda_source']
    ns['dd_decodediet_tt_mw_cuda_source']=lambda insts:dc_nf_source(mw(insts),'mw')
    native_load=ns['load_inline']
    def unique_load(name,*a,**kw):
        source=kw.get('cuda_sources',a[1] if len(a)>1 else '')
        return native_load(name+'_dc_'+hashlib.sha256(source.encode()).hexdigest()[:10],*a,**kw)
    ns['load_inline']=unique_load
    return ns


def dc_nf_prepare(weights,keys):
    global _nf_vendor,load_inline
    # Run the reviewed preparation against changed transport, but do not warm a
    # 12-bit bank through its sparse reader before packing it.
    small=_nf_small_prepare
    previous=_nf_vendor;native_load=load_inline;_nf_vendor=dc_nf_vendor
    def unique_load(name,*a,**kw):
        source=kw.get('cuda_sources',a[1] if len(a)>1 else '')
        return native_load(name+'_all_'+hashlib.sha256(source.encode()).hexdigest()[:10],*a,**kw)
    load_inline=unique_load
    try:state=small(weights,keys)
    finally:_nf_vendor=previous;load_inline=native_load
    # Selected MW path is built explicitly: other raw wide plans remain controls.
    if (32,'unpooler') in keys and state['native']:
        ns=state['ns'];cuda=_port_cuda(ns['dd_decodediet_tt_mw_cuda_source']([(32,64,4,2,2,True,1)]),m32w=True)
        tag=hashlib.sha256(cuda.encode()).hexdigest()[:12]
        mod=load_inline(name='dc_mw_'+tag,cpp_sources=ns['dd_decodediet_tt_mw_CPP'],cuda_sources=cuda,functions=['launch','info'],extra_cuda_cflags=['-O3','-std=c++20',ns['_nf_arch_flag']()],no_implicit_headers=True,with_pytorch_error_handling=False)
        w=weights['unpooler'];pk=ns['dd_decodediet_tt_mw_pack_weight'](w,64,4,1,True);g=pk['esc'].shape[0]
        meta=torch.empty((g,66),device=w.device,dtype=torch.int32);meta[:,:64]=pk['esc'];tab=pk['table']&((1<<64)-1)
        meta[:,64]=(tab&0xffffffff)-(1<<32) if tab&0x80000000 else tab&0xffffffff
        top=tab>>32;meta[:,65]=top-(1<<32) if top&0x80000000 else top
        state['plans'][32,'unpooler']=dict(kind='mw',fn=mod.launch,idx=0,wp=pk['w'],esc=meta,n=w.shape[0])
    state['stats']={}
    for key,p in state['plans'].items():
        if key not in keys:continue
        family='mw' if p.get('kind')=='mw' else NF_SMALL[key][0]
        fmt=1 if family in ('cp','d9','q8','q16','q32','p32','mw') else 0
        p['wp'],stats=dc_pack(p['wp'],fmt,n=state['roles'][key[1]].shape[0],k=state['roles'][key[1]].shape[1])
        stats['metadata_bytes']=p['esc'].numel()*p['esc'].element_size();stats['total_bytes']+=stats['metadata_bytes'];stats['bits_per_weight']=stats['total_bytes']*8/stats['weights']
        state['stats'][key]=stats
    return state


_DC_CD_MODULE=None

def dc_cd_source():
    src=CD_CUDA
    for expression,unit in (('wp[rt] + p * 96 + q * 32','(r0+rt)*KB+kb0+p'),('wp[rt] + (bb + PIPE) * 96 + q * 32','(r0+rt)*KB+kb0+bb+PIPE')):
        old='#pragma unroll\n        for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w('+expression+');'
        if old in src:src=src.replace(old,'dc_read(Wc,'+unit+',lane,(int4*)buf[p][rt]);')
    old='#pragma unroll\n              for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w(wp[rt] + (bb + PIPE) * 96 + q * 32);'
    assert old in src
    src=src.replace(old,'dc_read(Wc,(r0+rt)*KB+kb0+bb+PIPE,lane,(int4*)buf[p][rt]);')
    old='#pragma unroll\n            for (int q = 0; q < 3; ++q) buf[p][rt][q] = ld_w_if(wp[rt] + (bb + PIPE) * 96 + q * 32, bb + PIPE < nb);'
    assert old in src
    src=src.replace(old,'if(bb+PIPE<nb)dc_read(Wc,(r0+rt)*KB+kb0+bb+PIPE,lane,(int4*)buf[p][rt]);')
    assert 'buf[p][rt][q] = ld_w(' not in src
    return DC_NATIVE+'\n'+src.replace('CDG_INSTANCES','CDG_CASE(1,1,2,3)')


def dc_cd_module():
    global _DC_CD_MODULE
    if _DC_CD_MODULE is None:
        src=dc_cd_source();arch='90a' if torch.cuda.get_device_capability()==(9,0) else ''.join(map(str,torch.cuda.get_device_capability()))
        _DC_CD_MODULE=load_inline(name='dc_ours_'+hashlib.sha256(src.encode()).hexdigest()[:12]+'_'+arch,cpp_sources=CD_CPP,cuda_sources=src,functions=['cd_gemv','cd_decode'],extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return _DC_CD_MODULE


def dc_cd_gemv(x,c,cfg):
    nw,rt,pipe,s,prew,v,cl=cfg;n,k=c['N'],c['K'];p=torch.empty((s,x.shape[0],n),device=x.device)
    dc_cd_module().cd_gemv(c['coded'].data_ptr(),c['scales'].data_ptr(),c['eoff'].data_ptr(),c['esc'].data_ptr(),x.data_ptr(),x.stride(0),x.shape[0],p.data_ptr(),n,k//64,s,nw,rt,pipe,0,prew,1,v,cl,torch.cuda.current_stream().cuda_stream)
    return p,s
