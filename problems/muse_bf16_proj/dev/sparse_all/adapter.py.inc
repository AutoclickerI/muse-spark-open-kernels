DC_FORCE=False


def dc_role_weights(weights):
    hi=weights['router'].bfloat16();lo=(weights['router']-hi.float()).bfloat16()
    return dict(qkv_gate=weights['qkv_gate'],qkv3=weights['qkv_gate'][:1280].contiguous(),gate=weights['qkv_gate'][1280:].contiguous(),
        o=weights['o'],pooler=weights['pooler'],router_hi=hi,router_lo=lo,pool_router=torch.cat((weights['pooler'],hi,lo)),
        unpooler=weights['unpooler'],unpooler_slice=weights['unpooler'][:1024].contiguous())


def dc_triton_project(x,c,name):
    t,k=x.shape;n=c['N'];choice=GATE_SPLIT_QKV3 if name=='qkv3' else GB_ROWS.get((t,name))
    if name in ('pooler','router_hi','router_lo'):choice=GB_ROWS[t,'pool_router']
    if t==32 and name=='pool_router':
        s=6;p=torch.empty((s,t,n),device=x.device)
        _dc_gi_kernel[(n//64,s)](x,c['coded'],p,t,x.stride(0),n,k,s,BN=64,BK=128,TB=max(16,triton.next_power_of_2(t)),NS=2,TRANS=False,CG=False,num_warps=4,launch_pdl=True)
        return p,s
    if choice is not None:
        _,bn,s,ns=choice;p=torch.empty((s,t,n),device=x.device)
        _dc_gb_kernel[(n//bn,s)](x,c['coded'],p,t,x.stride(0),n,k,s,BN=bn,BK=128,TB=max(16,triton.next_power_of_2(t)),NS=ns,USE_PDL=True,num_warps=4,launch_pdl=True)
        return p,s
    bn,ks,bk=GEMV_CFG[(n,k)];s=k//ks;p=torch.empty((s,t,n),device=x.device)
    _dc_gemv_kernel[(n//bn,s)](x,c['coded'],p,t,x.stride(0),N=n,K=k,KS=ks,BN=bn,BK=bk,TB=max(16,triton.next_power_of_2(t)),PREFETCH=False,USE_PDL=True,num_warps=4,num_stages=4,launch_pdl=True)
    return p,s


def dc_gate_apply(x,y,c):
    t,k=x.shape;bn,s,ns=GATE_SPLIT_CFG;p,cnt=_gate_split_buffers(x.device,t);out=torch.empty_like(y)
    _dc_gate_split_kernel[(1024//bn,s)](x,c['coded'],p,cnt,y,out,t,x.stride(0),k,s,N=1024,BN=bn,BK=128,TB=max(16,triton.next_power_of_2(t)),NS=ns,num_warps=4,launch_pdl=True)
    return out


def dc_vdsk_project(x,c,direct=False):
    # Library internals cannot accept a code reader. This direct arm uses our
    # existing VDSK structure; the timed decode+library arm remains a comparison.
    if direct:cfg=(128,64,64,3,4,1);s=3 if c['K']==8192 else 2 if c['K']==4096 else 1
    else:_,cfg,s=PROJ_PINS[128,'qkv_gate']
    bt,bw,bk,ns,nw,tma=cfg;t,k=x.shape;n=c['N']
    out=torch.empty((s,t,n),device=x.device)
    xd=_VDSK_TD.from_tensor(x,[bt,bk]) if tma else None
    _dc_vdsk_gemm[(triton.cdiv(t,bt)*(n//bw),s)](x,c['coded'],out,xd,None,t,n,k,s,BT=bt,BW=bw,BK=bk,NS=ns,TMA=bool(tma),num_warps=nw,num_stages=ns)
    return out,s


def dc_prepare_roles(weights,keys=None,control=True):
    """Only dense paths; the layer's MoE/attention preparation is unnecessary here."""
    roles=dc_role_weights(weights);dev=weights['o'].device
    wanted=set(roles) if keys is None else set(keys)
    st=dict(roles=roles,coded={},sparse={},stats={},nf=None,nf_sparse=None,native=torch.cuda.get_device_capability()==(9,0))
    for name in sorted(wanted):
        c=cd_encode(roles[name]);st['coded'][name]=c;st['sparse'][name],st['stats'][name]=dc_encode_cd(c)
        st['stats'][name]['incumbent12_bytes']=c['coded'].numel()+c['eoff'].numel()*4+c['esc'].numel()*4+c['base'].numel()*4+c['scales'].numel()*4
    state={name:roles[name] for name in ('qkv_gate','qkv3','o','pool_router','unpooler','unpooler_slice')}
    _seed_v9_gemv_choices(state);_gate_split_prepare(state,dev)
    state['coded_dense']={name:st['coded'][name] for name in ('qkv_gate','pool_router') if name in st['coded']}
    state['proj_plan']=dict(PROJ_PINS);st['control']=state
    # Exact active NF kernels; qkv16 remains our balanced kernel as in S15.
    keys_nf={key for key in NF_PORT_SELECTED if key[1] in wanted and key!=(32,'o')}
    if control:
        keys_control=keys_nf|({(32,'o')} if 'o' in wanted else set())
        st['nf']=_dc_base_nf_prepare(weights,keys_control);state['nf_port']=st['nf']
    st['nf_sparse']=dc_nf_prepare(weights,keys_nf)
    if st['native'] or DC_FORCE:
        dc_prepare_library(st)
    if 'o' in wanted:dc_prepare_ol(st)
    return st


def dc_project(x,name,st,sparse):
    t=x.shape[0];c=st['sparse'][name];w=st['roles'][name]
    if not sparse:
        if name=='gate':
            raise ValueError('gate uses dc_gate_apply and its independent attention output')
        if name in ('pooler','router_hi','router_lo'):
            if t<=32:return _gb_gemv(x,w,GB_ROWS[t,'pool_router'])
            return torch.mm(x,w.T,out_dtype=torch.float32).unsqueeze(0),1
        if (t,name) in NF_PORT_SELECTED:return _dc_base_nf_project(x,name,st['nf'])
        if not st['native'] and not DC_FORCE and t>32:return torch.mm(x,w.T).unsqueeze(0),1
        return _nf_port_base_project(x,w,name,st['control'],PROJ_PINS.get((t,name),'custom'),torch.float32 if name=='pool_router' else torch.bfloat16)
    if name in ('pooler','router_hi','router_lo') and t<=32:return dc_triton_project(x,c,name)
    if (t,name) in CD_ROWS:return dc_cd_gemv(x,c,CD_ROWS[t,name])
    if (t,name) in NF_PORT_SELECTED:
        if (t,name)==(32,'o'):return dc_ol_project(x,st)
        return dc_nf_project(x,name,st)
    if t<=32:return dc_triton_project(x,c,name)
    if (t,name)==(128,'qkv_gate'):
        if st['native'] or DC_FORCE:return dc_vdsk_project(x,c)
        w=torch.empty_like(st['roles'][name]);dc_unpack(c,w)
        return torch.mm(x,w.T,out_dtype=torch.float32).unsqueeze(0),1
    # Opaque library kernels consume an exactly restored BF16 scratch inside this call.
    return dc_library_project(x,name,st)
