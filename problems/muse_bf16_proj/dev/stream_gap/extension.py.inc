SG_FORCE=False
SG_ARM='direct'
_sg_base_prepare=prepare
_sg_base_kernel=kernel_fn

def sg_build_codes(weights,bf16=False):
    ns=_nf_vendor();src=ns['V18S_CUDA_TEMPLATE']
    assert 'using OutT = float;' in src
    if bf16:src=src.replace('using OutT = float;','using OutT = __nv_bfloat16;')
    ns['V18S_CUDA_TEMPLATE']=src
    plans={};insts=[]
    for t in (1,8,16,32):
        for role in ('qkv_gate','o','pooler','unpooler'):
            w=weights[role][:1024] if role=='unpooler' and t<=8 else weights[role]
            n,k=w.shape;mt=max(1,t//8);cfg=(1,8,k//512,1)
            inst=(1,mt,1,1,0,0,256,0,n,k,*cfg,6);insts.append(inst);plans[t,role]=dict(w=w,cfg=cfg,inst=inst)
    insts=list(dict.fromkeys(insts));tag=hashlib.sha256(ns['v18s_cuda_source'](insts).encode()).hexdigest()[:12]
    mod=ns['v18s_build'](insts,'sg_direct_'+tag)
    packed={}
    for key,p in plans.items():
        w=p['w'];role=key[1];pkkey=(role,w.shape[0])
        if pkkey not in packed:
            pk,tables,stream=_nf_group_pack(ns,w,role,'v18');sp=stream(pk,1,*p['cfg']);assert sp is not None
            g=w.shape[0]//16*8;gw=torch.arange(g,device=w.device);tile=gw//8
            meta=torch.empty((g,34),device=w.device,dtype=torch.int32);meta[:,:32]=sp['esc'];meta[:,32:]=tables[tile].to(torch.int32)
            packed[pkkey]=dict(sp,esc=meta)
        p.update(packed[pkkey]);p.update(fn=mod.v18s_launch,idx=insts.index(p['inst']),bf16=bf16)
    return dict(plans=plans,mod=mod)

def sg_codes(x,role,bank):
    t=x.shape[0];cap=1 if t==1 else 8 if t<=8 else 16 if t<=16 else 32
    p=bank['plans'][cap,role];n,k=p['w'].shape
    out=torch.empty((x.shape[0],n),device=x.device,dtype=torch.bfloat16 if p['bf16'] else torch.float32)
    rc=p['fn'](p['idx'],p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),x.shape[0],n,k,*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream().cuda_stream,0)
    if rc:raise RuntimeError('SG coded launch '+str((role,x.shape[0],rc)))
    return out.unsqueeze(0),1

def sg_router_prepare(w):
    import linecache
    file='<sg_nf_router>';linecache.cache[file]=(len(SG_ROUTER_SOURCE),None,SG_ROUTER_SOURCE.splitlines(True),file)
    ns={'__name__':'sg_nf_router'};exec(compile(SG_ROUTER_SOURCE,file,'exec'),ns)
    st=ns['rt2_prepare_candidates']({'router':w},ns['rt2_ROUTER_CONFIGS']);st['small']={(16,8):ns['rt2_pack_small_router'](w,16,8)}
    return ns,st

def sg_router(x,bank):
    ns,st=bank;t=x.shape[0]
    if t==1:return ns['rt2_run_candidate'](x,st,-1)
    if t<=16:return ns['rt2_run_small_candidate'](x,st)
    return ns['rt2_run_candidate'](x,st,0 if t<=32 else 1 if t<=64 else 2 if t<=128 else 3)

def sg_prepare(weights):
    st=_sg_base_prepare(weights)
    st['sg_float']=sg_build_codes(weights,False);st['sg_direct']=sg_build_codes(weights,True);st['sg_router']=sg_router_prepare(weights['router'])
    # Launch attributes and driver function handles must exist before capture.
    for t in (1,8,16,32):
        for role in ('qkv_gate','o','pooler','unpooler'):
            x=torch.zeros((t,weights[role].shape[1]),device=weights[role].device,dtype=torch.bfloat16)
            sg_codes(x,role,st['sg_float']);sg_codes(x,role,st['sg_direct'])
    for t in (1,8,16,32,128,256):sg_router(torch.zeros((t,8192),device=weights['router'].device,dtype=torch.bfloat16),st['sg_router'])
    return st

def sg_run(data,st,arm):
    x=data['x'];t=x.shape[0];role=data['config']['layer']
    if arm=='A':return _sg_base_kernel(data,st)
    if role=='router':return sg_router(x,st['sg_router'])
    if t<=32:
        pair=sg_codes(x,role,st['sg_float' if arm=='float' else 'sg_direct'])
        return _p12_materialize(pair)
    return _sg_base_kernel(data,st)

prepare=sg_prepare

def kernel_fn(data,state):
    return sg_run(data,state,SG_ARM)
