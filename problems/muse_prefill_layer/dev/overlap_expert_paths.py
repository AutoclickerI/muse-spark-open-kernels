# New schedule: overlap W2 expansion with gate/up GEMM and activation, with a complete
# CUDA-graph fork/join inside each invocation. No output or routing state crosses calls.
_OVERLAP_EXPERT_CHOICES=('overlap_dg','overlap_dg_hint_w8','overlap_fused128')


def _prepare_overlap(st):
    st['expert_expand_stream']=torch.cuda.Stream(priority=0)
    st['expert_expand_ready']=torch.cuda.Event()
    st['expert_expand_done']=torch.cuda.Event()
    # Initialize native event handles in prepare, before any graph capture.
    st['expert_expand_ready'].record()
    st['expert_expand_stream'].wait_event(st['expert_expand_ready'])
    with torch.cuda.stream(st['expert_expand_stream']):st['expert_expand_done'].record()
    torch.cuda.current_stream().wait_event(st['expert_expand_done'])


def _expand_w2_async(st,counts,w2):
    st['expert_expand_ready'].record()
    with torch.cuda.stream(st['expert_expand_stream']):
        st['expert_expand_stream'].wait_event(st['expert_expand_ready'])
        _dequant_kernel[(32,4096//8)](st['w2'],st['w2_sf'],w2,counts,ROWS=4096,K=4096,BR=8,BKB=512,num_warps=4)
        st['expert_expand_done'].record()


def _run_overlap_experts(pooled,probs,slot_e,slot_pos,counts,st,name):
    import deep_gemm
    fused=name=='overlap_fused128'
    T,D=pooled.shape;cap=T*8+32*128;dev=pooled.device
    starts=torch.empty(32,dtype=torch.int32,device=dev);psum=torch.empty_like(starts)
    _offsets_kernel[(1,)](counts,starts,psum,ALIGN=128)
    a=torch.empty((cap,D),dtype=torch.bfloat16,device=dev)
    rows=torch.empty((T,8),dtype=torch.int32,device=dev)
    _gather_norm_kernel[(T,)](pooled,st['pre_expert'],slot_e,slot_pos,starts,a,rows,D=4096,num_warps=8)
    w13=torch.empty((32,8192,4096),dtype=torch.bfloat16,device=dev)
    w2=torch.empty((32,4096,4096),dtype=torch.bfloat16,device=dev)
    q13=st['fused_w13_q'] if fused else st['w13']
    s13=st['fused_w13_sf'] if fused else st['w13_sf']
    _dequant_kernel[(32,8192//8)](q13,s13,w13,counts,ROWS=8192,K=4096,BR=8,BKB=512,num_warps=4)
    _expand_w2_async(st,counts,w2)
    inter=torch.empty_like(a);y=torch.empty_like(a)
    hint={} if name=='overlap_dg' else {'expected_m_for_psum_layout':max(1,triton.cdiv(2*T,32))}
    if fused:
        tiles=torch.empty(33,dtype=torch.int32,device=dev)
        _fused_plan[(1,)](counts,tiles,BM=128,num_warps=1)
        grid=st['sms']*2
        _fused_grouped_gemm[(grid,)](a,w13,s13,inter,inter,starts,counts,tiles,st['gs13_gate'],st['gs13_up'],
            CAP=cap,GATED=True,DIRECT=False,BM=128,FN=64,BK=64,SPLIT=1,GRID=grid,num_warps=8,num_stages=min(3,st['fused_max_stages']))
    else:
        h=torch.empty((cap,8192),dtype=torch.bfloat16,device=dev)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a,w13,h,psum,use_psum_layout=True,**hint)
        br=1 if name=='overlap_dg' else 4
        _portfolio_swiglu[(32,triton.cdiv(T,br))](h,inter,starts,counts,st['gs13_gate'],st['gs13_up'],D=4096,BR=br,num_warps=4 if br==1 else 8)
    torch.cuda.current_stream().wait_event(st['expert_expand_done'])
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(inter,w2,y,psum,use_psum_layout=True,**hint)
    out=torch.empty_like(pooled)
    _combine_kernel[(T,)](y,rows,slot_e,probs,st['gs2'],st['post_expert_bf16'],out,D=4096,num_warps=8)
    return out


_EXPERT_CHOICES=tuple(_OVERLAP_EXPERT_CHOICES)+_EXPERT_CHOICES
_score_expert_fused=_score_expert

def _score_expert(pooled,probs,slot_e,slot_pos,counts,st,name):
    if name in _OVERLAP_EXPERT_CHOICES:
        return _run_overlap_experts(pooled,probs,slot_e,slot_pos,counts,st,name)
    return _score_expert_fused(pooled,probs,slot_e,slot_pos,counts,st,name)
