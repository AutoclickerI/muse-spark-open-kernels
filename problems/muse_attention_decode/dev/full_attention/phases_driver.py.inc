# P5b's measured full-attention selections, all other S19 calls unchanged.
_P5_PHASE_MODS={};_P5_PHASE_COUNTS={};_P5_PHASE_META={}
def _p5_phase_mod(role):
    if role not in _P5_PHASE_MODS:
        cu,cpp,functions=_P5_PHASE_SOURCES[role]
        arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
        _P5_PHASE_MODS[role]=load_inline(name='p5_phase_'+hashlib.sha256((cu+cpp).encode()).hexdigest()[:14]+'_'+arch,
            cpp_sources=cpp,cuda_sources=cu,functions=functions,extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return _P5_PHASE_MODS[role]

def _p5_phase_call(qkv,data,state,role):
    P,S=qkv;c=data['config'];B,L=c['batch'],c['q_len'];dev=P.device
    stream=torch.cuda.current_stream(dev).cuda_stream;pages=data['page_table'].shape[1];ps=c['page_size']
    sc=c['sm_scale']*c['qk_scale_factor']*1.4426950408889634
    out=torch.empty((B*L,Q_SIZE),device=dev,dtype=torch.bfloat16)
    # BEGIN M
    if role.startswith(('M','U')):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        ns=int(role[1:]);q,gate=qk_prep(qkv,data,state,q_fp8=True)
        descale={} if c['kv_scale']==1 else dict(k_descale=data['k_scale'].reshape(1,1).expand(B,HKV),v_descale=data['v_scale'].reshape(1,1).expand(B,HKV))
        a=flash_attn_with_kvcache(q=q,k_cache=data['k_cache'],v_cache=data['v_cache'],page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],max_seqlen_q=L,softmax_scale=c['sm_scale']*c['qk_scale_factor'],causal=True,window_size=(-1,-1),scheduler_metadata=_P5_PHASE_META[ns] if role.startswith('M') else None,num_splits=ns,**descale)
        if isinstance(a,tuple):a=a[0]
        _launch(_norm_gate_kernel,(B*L,),a,gate,out,RMS_EPS,HQ=HQ,D=HD,num_warps=4)
        return out
    # END M

    # BEGIN T
    raw=_attn_port_raw(qkv,data);mod=_p5_phase_mod(role);kvg=HKV//2
    occ=(max(1,mod.occupancy(1)),{i:mod.max_cluster(1,i) for i in range(1,17)})
    ns,cs=_attn_main_plan(B,kvg,pages*ps,L,0,_num_sms(dev),occ);rows=B*kvg*(ns//cs)*16
    po=torch.empty(rows*64,device=dev);ml=torch.empty(rows*2,device=dev);key=role,B,cs,str(dev)
    if key not in _P5_PHASE_COUNTS:_P5_PHASE_COUNTS[key]=torch.zeros(B*kvg*cs,device=dev,dtype=torch.int32)
    mod.attn_forward(raw.data_ptr(),*[data[k].data_ptr() for k in ('positions','cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')],0,po.data_ptr(),ml.data_ptr(),_P5_PHASE_COUNTS[key].data_ptr(),po.data_ptr(),1,out.data_ptr(),B,L,raw.stride(-2),HQ,HKV,pages,ps,0,ns,cs,sc,stream)
    return out
    # END T
_p5_phase_original_attention=_best7_attention
def _best7_attention(qkv,data,state,plan):
    c=data['config'];role=_P5_PHASE_SELECTION.get((c['batch'],c['q_len'])) if not c['window'] else None
    if role and not role.startswith('P'):return _p5_phase_call(qkv,data,state,role)
    return _p5_phase_original_attention(qkv,data,state,plan)
def _p5_phase_setup(device):
    for role in _P5_PHASE_SOURCES:_p5_phase_mod(role)
    for role in _P5_PHASE_SELECTION.values():
        if role.startswith('M'):
            ns=int(role[1:]);_P5_PHASE_META[ns]=torch.tensor([ns,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],device=device,dtype=torch.int32)
_p5_phase_original_prepare=prepare
def prepare(weights):
    _p5_phase_setup(weights['qkv_gate'].device)
    return _p5_phase_original_prepare(weights)
