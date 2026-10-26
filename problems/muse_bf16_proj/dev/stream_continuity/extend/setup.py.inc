def ew_clone_native(old,new):
    if isinstance(old,torch.Tensor):
        assert old.shape==new.shape and old.dtype==new.dtype
        return old
    if isinstance(old,dict):return {k:ew_clone_native(v,new[k]) if k in new else v for k,v in old.items()}
    if isinstance(old,list):return [ew_clone_native(a,b) for a,b in zip(old,new)]
    if isinstance(old,tuple):return tuple(ew_clone_native(a,b) for a,b in zip(old,new))
    return new

def ew_spaces(weights,state,base):
    spaces={'A':base,'A2':base};states={'A':state,'A2':state}
    for arm in EW_ARMS[2:]:spaces[arm]=sc_namespace(arm,SC_SOURCES[arm],base)
    # Share every unchanged selected native module. Changed modules keep their
    # own container even when CUDA's source cache returns the same compiled lib.
    caches={'_RVC_NINDEX0_MODULE':'RVC_NINDEX0_CUDA','_RVC_NBASE1_MODULE':'RVC_NBASE1_CUDA','_RVC_NBASE2_MODULE':'RVC_NBASE2_CUDA','_P3B_MODULE':'P3B_CUDA','_GLUE_CL_EXT':'GLUE_CL_CUDA','_GP_MOD':'GP_CUDA','_RT_MOD':'RT_CUDA','_RT_FAST_MOD':'RT_FAST_CUDA','_PP_MOD':'PARTIAL_CUDA'}
    for arm,ns in spaces.items():
        if arm in ('A','A2'):continue
        ns['NF_OPTS']=dict(base['NF_OPTS'])
        for key,cu in caches.items():
            if key in base and ns.get(cu)==base.get(cu):
                v=base[key];ns[key]=list(v) if isinstance(v,list) else dict(v) if isinstance(v,dict) else v
    allns=spaces['ALL'];selected={key for key in base['NF_PORT_SELECTED'] if key!=(256,'o')}
    prepared=allns['nf_prepare'](weights,selected)
    tmp=dict(state);allns['p10_pool_prepare'](tmp)
    for arm in EW_ARMS[2:]:
        ns=spaces[arm];cfg=EW_CONFIG[arm];st=dict(state)
        if cfg.get('roles') and cfg.get('early',True):
            nf=dict(state['nf_port'],plans=dict(state['nf_port']['plans']))
            for key,p in prepared['plans'].items():
                if key[1] in cfg['roles']:nf['plans'][key]=ew_clone_native(nf['plans'][key],p)
            st['nf_port']=nf
            if 'pool_router' in cfg['roles']:st['p10_pool_raw']=ew_clone_native(state['p10_pool_raw'],tmp['p10_pool_raw'])
        states[arm]=st
        if cfg.get('roles'):ns['ew_dense_module']()
        if cfg.get('n13') or cfg.get('n2'):
            for name in ('rvc_nindex0_module','rvc_nbase1_module','rvc_nbase2_module'):ns[name]()
        if cfg.get('n13'):ns['_glue_cl_ext']();ns['gp_module']()
        if cfg.get('h13') or cfg.get('h2'):
            mod=ns['p3b_module']();mod.ch_info(4,0);mod.ch_info(4,1)
        if cfg.get('boundaries'):ns['_glue_cl_ext']()
        if cfg.get('kv'):ns['partial_port_ext']()
        ns['EW_ACTIVE']=True
    return spaces,states
