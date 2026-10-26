# P5b deea291e: B1/L8/full has one packed query tile. All values here
# are schedule constants; FA3 still derives each key range from fresh lengths.
_P5B_M64_META=None
def _p5_phase_setup(device):
    global _P5B_M64_META
    _P5B_M64_META=torch.tensor([64,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],device=device,dtype=torch.int32)
_p5b_m64_original_attention=_attention_fa3_public
def _attention_fa3_public(q,gate,data,state):
    c=data['config']
    if (c['batch'],c['q_len'],c['window'])!=(1,8,0):return _p5b_m64_original_attention(q,gate,data,state)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    descale={} if c['kv_scale']==1.0 else dict(k_descale=data['k_scale'].reshape(1,1).expand(1,HKV),v_descale=data['v_scale'].reshape(1,1).expand(1,HKV))
    a=flash_attn_with_kvcache(q=q,k_cache=data['k_cache'],v_cache=data['v_cache'],page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],max_seqlen_q=8,softmax_scale=c['sm_scale']*c['qk_scale_factor'],causal=True,window_size=(-1,-1),scheduler_metadata=_P5B_M64_META,num_splits=64,**descale)
    if isinstance(a,tuple):a=a[0]
    o=torch.empty((8,Q_SIZE),device=q.device,dtype=torch.bfloat16)
    _launch(_norm_gate_kernel,(8,),a,gate,o,RMS_EPS,HQ=HQ,D=HD,num_warps=4)
    return o
_p5b_m64_original_prepare=prepare
def prepare(weights):
    _p5_phase_setup(weights['qkv_gate'].device)
    return _p5b_m64_original_prepare(weights)
