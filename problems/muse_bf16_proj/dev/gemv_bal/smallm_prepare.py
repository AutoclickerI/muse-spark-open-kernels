"""Materialize S11's fixed small-M dispatch without tuning unused large-M paths.

This is a lab comparator constructor, not a submission prepare replacement.
Its plans, stored weights and outputs are compared with full S11 prepare by
check_smallm_prepare.py before sending the lab.
"""
import concurrent.futures


def gi_prepare_smallm(ns,weights):
    dev=weights['qkv_gate'].device
    # The S11 small-M consumer reads only the recoded bank. Avoid constructing
    # unused raw V9, NT2, N_full, Humming and Lt banks for these fixed rows.
    def bank(w):
        base=dict(gs13=w['w13_gs'][:32].float().contiguous(),gs2=w['w2_gs'][:32].float().contiguous(),
                  pnw=w['post_norm_w'].contiguous(),grid=torch.cuda.get_device_properties(dev).multi_processor_count,
                  cnt=torch.zeros(ns['NCNT'],device=dev,dtype=torch.int32),mod=None)
        return ns['_prepare_recoded_bank'](w,base,None)
    ns['v9_prepare']=bank
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(ns[name]) for name in ('smallm_module','_glue_cl_ext','_layer_attn_ext')]
        if ns['_attn_port_hopper'](dev):futures.append(pool.submit(ns['_attn_port_ext']))
        st=ns['_prepare_weights'](weights)
        for f in futures:f.result()
    b=st['moe_bank']
    st['s4_nt1_bank']=dict(gs13=b['gs13'],gs2=b['gs2'],pnw=b['pnw'],sms=b['grid'],
                            banks={'lop3':dict(w13p=b['w13p'],w2p=b['w2p'])})
    ns['smallm_bind'](st['s4_nt1_bank'],ns['SMALLM_FLAGS'])
    ns['_seed_v9_gemv_choices'](st)
    st['proj_plan']={key:value for key,value in ns['PROJ_PINS'].items() if key[0]<=32}
    st['attention_plan']={};st['vq8_rows']={}
    st['best7_plans']={};st['best7_stream_ops']={};st['best7_prefetch_ops']={}
    ns['_gate_split_prepare'](st,dev)
    for key,(glue,attention,moe,projection) in ns['BEST7_SETTINGS'].items():
        if key!='other' and key[0]*key[1]>32:continue
        geometry={}
        if isinstance(glue,tuple):
            glue,b1,route,b2=glue;geometry=dict(b1=b1,route=route,b2=b2)
        projection={} if projection=='auto' else dict(projection)
        assert 'l2_prefetch' not in projection
        st['best7_plans'][key]=dict(glue=glue,geometry=geometry,attention=attention,moe=moe,
            projection=projection,prefetch=0,dense_chain=False)
        if key=='other':continue
        t=key[0]*key[1]
        for role,choice in projection.items():
            if isinstance(choice,tuple) and choice[0]=='stream':
                opkey=(t,role,choice[1],choice[2])
                if opkey not in st['best7_stream_ops']:
                    st['best7_stream_ops'][opkey]=ns['_best7_stream_op'](t,role,choice,st,dev)[0]
    torch.cuda.synchronize()
    return st


def ve_host(weights):
    fname='<p7_s11_fixed_smallm>'
    linecache.cache[fname]=(len(VE_HOST_SRC),None,VE_HOST_SRC.splitlines(True),fname)
    ns=dict(__name__='p7_s11_fixed_smallm')
    exec(compile(VE_HOST_SRC,fname,'exec'),ns)
    if torch.cuda.get_device_capability()!=(9,0):ns['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
    return ns,gi_prepare_smallm(ns,weights)
