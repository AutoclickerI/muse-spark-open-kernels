_P5B_BRIDGE=None;_P5B_REDUCE=None
def _p5b_merge_setup():
    global _P5B_BRIDGE,_P5B_REDUCE
    import hashlib
    import sgl_kernel.flash_ops
    _P5B_BRIDGE=load_inline(name='p5b_fa_bridge_'+hashlib.sha256(_P5B_BRIDGE_SRC.encode()).hexdigest()[:14],cpp_sources=_P5B_BRIDGE_SRC,functions=['fa_bind','fa_probe','fa_main'],extra_ldflags=['-ldl'],with_cuda=True)
    _P5B_BRIDGE.fa_bind(sgl_kernel.flash_ops.__file__)
    _P5B_REDUCE=load_inline(name='p5b_reduce_'+hashlib.sha256(_P5B_REDUCE_SRC.encode()).hexdigest()[:14]+'_90a',cpp_sources='void row_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);',cuda_sources=_P5B_REDUCE_SRC,functions=['row_forward'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'])
def _p5b_merge_main(q,data,s,fo,lse,oa,la,meta):
    c=data['config']
    _P5B_BRIDGE.fa_main(q,data['k_cache'],data['v_cache'],fo,lse,oa,la,data['cache_seqlens'],data['cu_seqlens_q'],data['page_table'],meta,data['k_scale'],data['v_scale'],c['q_len'],c['window'],s,c['sm_scale']*c['qk_scale_factor'],2 if c['kv_scale']==1 else 0,torch.cuda.current_stream().cuda_stream)
_p5b_merge_original_fa=_p5b_sched_fa
def _p5b_sched_fa(q,gate,data,split,pack):
    c=data['config'];b,l=c['batch'],c['q_len'];arm=_P5B_MERGE_SELECTION.get((b,l,bool(c['window'])))
    if not arm:return _p5b_merge_original_fa(q,gate,data,split,pack)
    ns=int(arm[1:].split('w')[0]);warps=4 if 'w4' in arm else 1
    effective_window=bool(c['window'] and c['window']<data['page_table'].shape[1]*c['page_size'])
    meta=_P5B_FA_META[b,l,effective_window,ns]
    fo=torch.empty((b*l,16,64),device=q.device,dtype=torch.bfloat16);lse=torch.empty((16,b*l),device=q.device)
    oa=torch.empty((ns,16,b*l,64),device=q.device);la=torch.empty((ns,16,b*l),device=q.device)
    _p5b_merge_main(q,data,ns,fo,lse,oa,la,meta)
    out=torch.empty((b*l,1024),device=q.device,dtype=torch.bfloat16)
    _P5B_REDUCE.row_forward(0,meta[-1:].data_ptr(),(fo if ns==1 else oa).data_ptr(),la.data_ptr(),0,gate.data_ptr(),data['v_scale'].data_ptr(),out.data_ptr(),b,l,16,2,ns,0,0,0,1,warps,torch.cuda.current_stream().cuda_stream)
    return out
_p5b_merge_original_prepare=prepare
def prepare(weights):
    _p5b_merge_setup()
    return _p5b_merge_original_prepare(weights)
