def _s15n_nearest(t,knots):
    return min(knots,key=lambda k:(abs(t-k),k))


def _s15n_plan(cfg,state):
    b,l,w,t=cfg['batch'],cfg['q_len'],cfg['window'],cfg['tokens']
    key=(_s15n_nearest(b,(1,16,32)),l,w)
    plan=dict(state['best7_plans'][key])
    g=_s15n_nearest(t,_TUNE_TOKENS)
    gkey=(1,8,w) if g==8 else (g,1,w) if g<=32 else (g//8,8,w)
    plan['geometry']=state['best7_plans'][gkey]['geometry']
    plan['moe']='nt1_lop3' if t<=64 else 'hh_m128' if t<=128 else 'mixed_parallel_m256'
    return plan


def _s15n_nf_key(t,name):
    # Small templates enforce m<=MT*8; O32/MW32 mask arbitrary m. Original
    # unselected anchors remain projection choices, preserving the scored rows.
    caps={(1,'qkv3'):8,(32,'qkv3'):32,(32,'qkv_gate'):32,
          (1,'o'):8,(8,'o'):8,(16,'o'):16,(16,'unpooler'):16}
    anchors=[m for m in _TUNE_TOKENS if (name!='unpooler_slice' or m<=8)
             and (name!='unpooler' or m>8) and t<=caps.get((m,name),float('inf'))]
    key=(_s15n_nearest(t,anchors),name)
    return key if key in NF_PORT_SELECTED else None


def _s15n_projection(t,name,state):
    choices={m:v for (m,n),v in state['proj_plan'].items() if n==name}
    if name=='qkv3':
        choices={m:'custom' for m in (1,8,16,32)}
        choices[128]=('vdsk',*VDSK_SHORTLIST[128,'qkv_gate'][0])
    # Lt descriptors bind M. Pad only to a larger prepared M; if a call is
    # larger than every plan, process independent row chunks with the same role.
    valid=[m for m,v in choices.items() if (m>32 or t<=64)
           and (not isinstance(v,tuple) or v[0] not in ('lt','short_lt') or m>=t)]
    anchor=_s15n_nearest(t,valid) if valid else max(choices)
    return anchor,choices[anchor]
