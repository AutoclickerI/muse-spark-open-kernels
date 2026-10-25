_P12_ATTN_STATE={}

def kernel_fn(data):
    cfg=dict(data['config']);b,l=cfg['batch'],cfg['q_len'];cfg['tokens']=b*l
    d=dict(data,config=cfg);pair=(data['qkv'].reshape(1,b*l,-1),1)
    # DP heads are a required nearby shape; S15's partial-input kernel is dynamic.
    if cfg['num_q_heads']!=16:
        return attention_partial_port(pair,d,_PP_FAST).reshape(b,l,-1)
    key=str(data['qkv'].device),cfg['rope_theta']
    if key not in _P12_ATTN_STATE:
        _P12_ATTN_STATE[key]=dict(inv_freq=_inv_freq(cfg['rope_theta'],data['qkv'].device),cluster_rows=_CLUSTER_ROWS,
            partial_port_rows=_PP_ROWS,vq8_rows=_VQ8_ROWS,attention_plan={})
    settings=BEST7_SETTINGS.get((b,l,cfg['window']),BEST7_SETTINGS['other'])
    glue=settings[0][0] if isinstance(settings[0],tuple) else settings[0]
    plan=dict(glue=glue,attention=settings[1])
    return _best7_attention(pair,d,_P12_ATTN_STATE[key],plan).reshape(b,l,-1)
