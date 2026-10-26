_P5B_VQ_MOD={}
def _p5b_vq_module(role):
    if role not in _P5B_VQ_MOD:
        import hashlib
        src=_P5B_VQ_SOURCES[role]
        _P5B_VQ_MOD[role]=load_inline(name='p5b_vq_'+hashlib.sha256(src.encode()).hexdigest()[:14]+'_90a',cpp_sources=VQ8_CPP,cuda_sources=src,functions=['vq8_forward','vq8_resources'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a'])
    return _P5B_VQ_MOD[role]
_p5b_vq_original_attention=_best7_attention
def _best7_attention(qkv,data,state,plan):
    c=data['config'];arm=_P5B_VQ_SELECTION.get((c['batch'],c['q_len'],bool(c['window'])))
    if not arm:return _p5b_vq_original_attention(qkv,data,state,plan)
    P,S=qkv;b,l=c['batch'],c['q_len'];sms=torch.cuda.get_device_properties(P.device).multi_processor_count
    split=S if P.dtype==torch.float32 else 0;raw=P if split else P[0];ns=b*(HKV//2)
    out=torch.empty((b*l,1024),device=P.device,dtype=torch.bfloat16)
    part=torch.empty((max(sms,8*ns)+ns)*128*68+256,device=P.device)
    role='N' if arm in ('R','Rw4') else arm[0];cs=0 if role=='N' else int(arm[1:])
    tiles=8 if role=='N' else cs;grid=sms if role=='N' else cs*ns
    args=[raw.data_ptr()]+[data[k].data_ptr() for k in ('cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')]
    args+=[state['inv_freq'].data_ptr() if c['rope_theta'] else 0,part.data_ptr(),out.data_ptr(),b,l,16,2,raw.stride(-2),data['page_table'].shape[1],c['page_size'],0,split,tiles,grid,4,c['sm_scale']*c['qk_scale_factor']*1.4426950408889634,P.stride(0) if split else 0,1,torch.cuda.current_stream().cuda_stream]
    _p5b_vq_module(role).vq8_forward(*args)
    if role=='N':
        prefix=part[(sms+ns)*128*68:]
        _P5B_REDUCE.row_forward(part.data_ptr(),prefix.data_ptr(),0,0,raw.data_ptr(),0,data['v_scale'].data_ptr(),out.data_ptr(),b,l,16,2,0,raw.stride(-2),split,P.stride(0) if split else 0,0,4 if arm=='Rw4' else 1,torch.cuda.current_stream().cuda_stream)
    return out
_p5b_vq_original_prepare=prepare
def prepare(weights):
    for role in _P5B_VQ_SOURCES:_p5b_vq_module(role)
    return _p5b_vq_original_prepare(weights)
