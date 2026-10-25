def prepare(weights):
    from concurrent.futures import ThreadPoolExecutor
    # Board supplies all256 experts; EP8 owns the first32 at original precision.
    w={k:(v[:32].contiguous() if k.startswith(('w13','w2')) else v.contiguous()) for k,v in weights.items()}
    w['post_expert_norm']=w['post_norm_w']
    with ThreadPoolExecutor(3) as pool:
        rec=pool.submit(_v9_recoded_module);pool.submit(smallm_module);nf=pool.submit(nf_module)
        raw=v9_prepare(w);bank=_prepare_recoded_bank(w,raw,rec.result());nf.result()
    state=dict(moe_bank=bank,moe_bank_raw=raw)
    _s4_prepare_banks(state,w)
    dev=w['w13'].device
    state['nf_streams']=(torch.cuda.Stream(device=dev),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(device=dev),torch.cuda.Event())
    return state


def kernel_fn(data,state):
    m=data['hidden_states'].shape[0]
    if m in SMALLM_FLAGS:return smallm_experts(data,state['s4_nt1_bank'],SMALLM_FLAGS[m])
    if m in (128,256):return nf_call(data,state)
    return v9_kernel_fn(data,state['moe_bank'])
