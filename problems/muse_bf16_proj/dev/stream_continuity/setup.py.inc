def sc_namespace(name,source,base=None):
    fname='<p7-continuity-'+name+'-'+hashlib.sha256(source.encode()).hexdigest()[:12]+'>'
    linecache.cache[fname]=(len(source),None,source.splitlines(True),fname)
    m=types.ModuleType('p7_cont_'+name);m.__file__=fname;exec(compile(source,fname,'exec'),m.__dict__)
    if globals().get('SC_LOCAL_INSTALL'):SC_LOCAL_INSTALL(m)
    ns=m.__dict__
    if base is None:return ns
    # Share an identical JIT dependency closure; never reuse changed code.
    names={k for k,v in base.items() if hasattr(v,'warmup') and isinstance(getattr(v,'src',None),str) and getattr(ns.get(k),'src',None)==v.src}
    deps={k:{n.id for n in ast.walk(ast.parse(base[k].src)) if isinstance(n,ast.Name) and hasattr(base.get(n.id),'warmup')} for k in names}
    while (kept:={k for k in names if deps[k]<=names})!=names:names=kept
    for k in names:ns[k]=base[k]
    # Only caches of identical native sources are shared. Mutable containers
    # themselves are copied, so native option initialization cannot mutate A.
    same=lambda key:ns.get(key)==base.get(key)
    caches={'_SMALLM_MODULE':'SMALLM_VAR_CUDA','_ATTN_PORT_EXT':'ATTN_PORT_CUDA_SRC',
            '_V9_MOD':'V9_CUDA_SRC','_V9_RECODED_MOD':'V9_CUDA_SRC','_GLUE_CL_EXT':'GLUE_CL_CUDA',
            '_GF_MOD':'GF_CUDA','_PARTIAL_EXT':'PARTIAL_CUDA','_VQ8_MOD':'VQ8_CUDA'}
    caches.update({'_s16_epi_NORM__EPIN_MODULE':'_s16_epi_NORM_EPIN_CUDA', '_s16_epi_PAIR__EPIT_MODULE':'_s16_epi_PAIR_EPIT_CUDA'})
    for k,source_key in caches.items():
        if k in base and same(source_key):
            v=base[k];ns[k]=list(v) if isinstance(v,list) else dict(v) if isinstance(v,dict) else v
    for k in ('_GEMV_CHOICES','_ATTENTION_PUBLIC_CHOICES'):
        if k in base:ns[k]=dict(base[k])
    return ns

def sc_prepare_spaces(weights,state,base):
    spaces={'A':base,'A2':base};states={'A':state,'A2':state}
    for arm in SC_ARMS[2:]:spaces[arm]=sc_namespace(arm,SC_SOURCES[arm],base)
    nf=state['nf_port'];shared=dict(state,sc_sink=torch.empty(1024,device=weights['o'].device,dtype=torch.int32))
    for arm in SC_ARMS[2:]:states[arm]=shared
    # All W arms reuse C's actual packed allocations. Only native launch
    # functions change, so a bank-address change cannot masquerade as overlap.
    if 'W' in spaces:
        w=spaces['W'];prepared=w['nf_prepare'](weights,{(1,'o'),(8,'o')})
        wn=dict(nf,plans=dict(nf['plans']))
        for key,p in prepared['plans'].items():
            original=nf['plans'][key];assert original['cfg']==p['cfg'] and original['idx']==p['idx']
            assert original['wp'].shape==p['wp'].shape and original['esc'].shape==p['esc'].shape
            wn['plans'][key]=dict(original,fn=p['fn'])
        states['W']=dict(shared,nf_port=wn)
        if 'X' in spaces:states['X']=dict(shared,nf_port=wn)
    # P and E share the same public tensor banks, with their own native code.
    for arm in ('E','P','X'):
        if arm in spaces:
            spaces[arm]['_s16_epi_NORM_epin_module']();spaces[arm]['_s16_epi_PAIR_epit_module']()
    for arm in SC_ARMS[2:]:
        ns=spaces[arm]
        if 'SC_ACTIVE' in ns:ns['sc_module']();ns['SC_ACTIVE']=True
    return spaces,states
