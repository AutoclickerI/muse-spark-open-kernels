
_ROOF_STATE={}
def kernel_fn(data):
    ns=globals()
    cfg=dict(data['config']);b,l=cfg['batch'],cfg['q_len'];cfg['tokens']=b*l
    d=dict(data,config=cfg);pair=(data['qkv'].reshape(1,b*l,-1),1)
    if cfg['num_q_heads']!=16:return ns['attention_partial_port'](pair,d,ns['_PP_FAST']).reshape(b,l,-1)
    key=str(data['qkv'].device),cfg['rope_theta']
    if key not in _ROOF_STATE:
        _ROOF_STATE[key]=dict(inv_freq=ns['_inv_freq'](cfg['rope_theta'],data['qkv'].device),cluster_rows=ns['_CLUSTER_ROWS'],
            partial_port_rows=ns['_PP_ROWS'],vq8_rows=ns['_VQ8_ROWS'],attention_plan={})
    settings=ns['BEST7_SETTINGS'].get((b,l,cfg['window']),ns['BEST7_SETTINGS']['other'])
    glue=settings[0][0] if isinstance(settings[0],tuple) else settings[0]
    plan=dict(glue=glue,attention=settings[1])
    return ns['_best7_attention'](pair,d,_ROOF_STATE[key],plan).reshape(b,l,-1)
