# This postlude owns only the selected full-attention rows. The builder embeds
# the exact measured CUDA/CPP and S18+R8+D control before these definitions.
_P5_FIXED_MODS={};_P5_FIXED_COUNTS={}
def _p5_fixed_mod(role):
    if role not in _P5_FIXED_MODS:
        cu,cpp,funcs=_P5_FIXED_SOURCES[role]
        arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
        _P5_FIXED_MODS[role]=load_inline(name='p5_full_'+hashlib.sha256((cu+cpp).encode()).hexdigest()[:14]+'_'+arch,
            cpp_sources=cpp,cuda_sources=cu,functions=funcs,
            extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return _P5_FIXED_MODS[role]
def _p5_fixed_counts(key,count,device):
    if key not in _P5_FIXED_COUNTS:_P5_FIXED_COUNTS[key]=torch.zeros(count,device=device,dtype=torch.int32)
    return _P5_FIXED_COUNTS[key]
def _p5_fixed_attention(qkv,data,recipe):
    P,S=qkv;c=data['config'];B,L=c['batch'],c['q_len'];HQ_,HK_=c.get('num_q_heads',HQ),c.get('num_kv_heads',HKV)
    dev=P.device;raw=P[0] if P.dtype==torch.bfloat16 and P.dim()==3 else P;split=S if P.dtype==torch.float32 else 0
    out=torch.empty((B*L,HQ_*64),device=dev,dtype=torch.bfloat16);stream=torch.cuda.current_stream(dev).cuda_stream
    pages=data['page_table'].shape[1];ps=c['page_size'];sc=c['sm_scale']*c['qk_scale_factor']*1.4426950408889634
    common=[data[k].data_ptr() for k in ('positions','cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')]
    # Preserve S18's existing partial-sum cast before the q1 BF16 kernel.
    raw=_attn_port_raw(qkv,data);mod=_p5_fixed_mod(recipe);kvg=HK_//2
    occ=(max(1,mod.occupancy(1)),{i:mod.max_cluster(1,i) for i in range(1,17)})
    ns,cs=_attn_main_plan(B,kvg,pages*ps,L,0,_num_sms(dev),occ);rows=B*kvg*(ns//cs)*L*16
    po=torch.empty(rows*64,device=dev);ml=torch.empty(rows*2,device=dev);counter=_p5_fixed_counts((recipe,B,L,cs),B*kvg*cs,dev)
    mod.attn_forward(raw.data_ptr(),*common,0,po.data_ptr(),ml.data_ptr(),counter.data_ptr(),po.data_ptr(),1,out.data_ptr(),B,L,raw.stride(-2),HQ_,HK_,pages,ps,0,ns,cs,sc,stream)
    return out
_p5_fixed_base_attention=_best7_attention
def _best7_attention(qkv,data,state,plan):
    c=data['config'];recipe=_P5_FIXED_SELECTION.get((c['batch'],c['q_len'])) if not c['window'] else None
    if recipe:return _p5_fixed_attention(qkv,data,recipe)
    return _p5_fixed_base_attention(qkv,data,state,plan)
_p5_fixed_base_prepare=prepare
def prepare(weights):
    # Compile inside the caller's untimed prepare, before CUDA graph capture.
    for role in _P5_FIXED_SOURCES:_p5_fixed_mod(role)
    return _p5_fixed_base_prepare(weights)
