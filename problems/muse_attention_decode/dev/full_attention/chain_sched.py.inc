# Shape-only FA3 schedules. The bank contains no sequence-length or input values.
_P5B_FA_META={};_P5B_FA_STORAGE=None;_P5B_FA_SMS=132;_P5B_FA_PICKS={}
def _p5b_sched_init(device):
    global _P5B_FA_STORAGE,_P5B_FA_SMS
    _P5B_FA_SMS=torch.cuda.get_device_properties(device).multi_processor_count
    values=[];spans={}
    for b in range(1,129):
        br=(b+3)//4*4
        for l in (1,8):
            for sliding in (False,True):
                sort=not sliding;heads=sliding or l>1
                for s in (1,2,4,8,16,32,64,128):
                    v=[s]*b+[0]*(br-b)+[1]*b+[0]*(br-b)
                    if sort:v+=list(range(b))+[0]*(br-b)
                    if heads:v+=[1]*b+[0]*(br-b)
                    v+=[0];spans[b,l,sliding,s]=(len(values),len(v));values+=v
    _P5B_FA_STORAGE=torch.tensor(values,device=device,dtype=torch.int32)
    for k,(a,n) in spans.items():_P5B_FA_META[k]=_P5B_FA_STORAGE[a:a+n]
def _p5b_sched_split(c,requested=0,pack=None):
    b,l,w=c['batch'],c['q_len'],bool(c['window'])
    if requested:return requested
    if (b,l,w) in _P5B_FA_PICKS:return _P5B_FA_PICKS[b,l,w]
    n=max(1,_P5B_FA_SMS//(b*(c.get('num_q_heads',HQ) if pack is False else c.get('num_kv_heads',HKV))))
    return min(16 if w else 128,1<<(n.bit_length()-1))
def _p5b_sched_fa(q,gate,data,split,pack):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    c=data['config'];b,l=c['batch'],c['q_len'];w=bool(c['window'] and c['window']<data['page_table'].shape[1]*c['page_size'])
    meta=_P5B_FA_META[b,l,w,split]
    descale={} if c['kv_scale']==1. else dict(k_descale=data['k_scale'].reshape(1,1).expand(b,HKV),v_descale=data['v_scale'].reshape(1,1).expand(b,HKV))
    a=flash_attn_with_kvcache(q=q,k_cache=data['k_cache'],v_cache=data['v_cache'],page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],max_seqlen_q=l,softmax_scale=c['sm_scale']*c['qk_scale_factor'],causal=True,window_size=(c['window']-1,0) if w else (-1,-1),scheduler_metadata=meta,num_splits=split,pack_gqa=pack,**descale)
    if isinstance(a,tuple):a=a[0]
    out=torch.empty((b*l,Q_SIZE),device=q.device,dtype=torch.bfloat16)
    _launch(_norm_gate_kernel,(b*l,),a,gate,out,RMS_EPS,HQ=HQ,D=HD,num_warps=4)
    return out
_p5b_sched_original_public=_attention_fa3_public
_p5b_sched_original_call=_fa3_call
def _attention_fa3_public(q,gate,data,state):
    c=data['config'];s=_p5b_sched_split(c)
    if (c['batch'],c['q_len'],bool(c['window']),s) in _P5B_FA_META:return _p5b_sched_fa(q,gate,data,s,None)
    return _p5b_sched_original_public(q,gate,data,state)
def _fa3_call(q,gate,data,setting):
    c=data['config'];s=_p5b_sched_split(c,setting[0],setting[1])
    if (c['batch'],c['q_len'],bool(c['window']),s) in _P5B_FA_META:return _p5b_sched_fa(q,gate,data,s,setting[1])
    return _p5b_sched_original_call(q,gate,data,setting)
_p5b_sched_original_prepare=prepare
def prepare(weights):
    _p5b_sched_init(weights['qkv_gate'].device)
    return _p5b_sched_original_prepare(weights)
