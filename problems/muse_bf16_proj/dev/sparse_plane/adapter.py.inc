def sp_prepare(weights,wr=1,wk=8,split=1,st=1,sparse=False,vector=False):
    # Native kernel structure: one n8/16/32 warp MMA tile, local K reduction or DSM.
    # This standalone builder intentionally limits itself to generic V18 table decode.
    family='v18';cfg=(wr,wk,128//(wk*split),split)
    ns=_nf_vendor();src=ns['V18S_CUDA_TEMPLATE']
    if sparse:src=sparse_source(src,family,vector)
    ns['V18S_CUDA_TEMPLATE']=src
    insts=[(1,t//8 if t>=8 else 1,1,st,2 if split>1 else 0,0,wr*wk*32,0,4608,8192,*cfg,6) for t in (1,8,16,32)]
    unique=list(dict.fromkeys(insts));tag=hashlib.sha256(ns['v18s_cuda_source'](unique).encode()).hexdigest()[:12]
    mod=ns['v18s_build'](unique,'sparse_nf_'+tag)
    hi=weights['router'].bfloat16();w=torch.cat((weights['pooler'],hi,(weights['router']-hi.float()).bfloat16()))
    pk,tables,stream=_nf_group_pack(ns,w,'pool_router',family);sp=stream(pk,1,*cfg)
    if sp is None:raise RuntimeError('escape list capacity')
    gx=w.shape[0]//(wr*16);g=gx*split*wr*wk;gw=torch.arange(g,device=w.device)
    tile=((gw//(wr*wk))%gx)*wr+(gw%wr)
    meta=torch.empty((g,34),device=w.device,dtype=torch.int32);meta[:,:32]=sp['esc'];meta[:,32:]=tables[tile].to(torch.int32)
    # One common-window rotation per origin group. Each group differs only by a
    # power-of-two scale, so use per-group compression and concatenate after coding.
    plan=dict(sp,esc=meta,fn=mod.v18s_launch,idx=0,cfg=cfg,n=4608)
    stats=dict(weights=w.numel(),original_bytes=sp['wp'].numel(),packed_bytes=sp['wp'].numel(),bits_per_weight=12.)
    if sparse:plan,stats=sparse_pack(plan,family)
    return dict(plan=plan,w=w,mod=mod,insts=list(dict.fromkeys(insts)),stats=stats)


def sp_project(x,state):
    p=dict(state['plan']);t=x.shape[0];mt=t//8 if t>=8 else 1
    if p['cfg'][-1]>1 and torch.cuda.get_device_capability()!=(9,0):
        return torch.mm(x,state['w'].T,out_dtype=torch.float32).unsqueeze(0),1
    p['idx']=next(i for i,v in enumerate(state['insts']) if v[1]==mt)
    # Sparse generic kernels also execute with DSM=1 on sm120.
    out=torch.empty((t,4608),device=x.device,dtype=torch.float32)
    rc=p['fn'](p['idx'],p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),t,4608,8192,*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream().cuda_stream,0)
    if rc:raise RuntimeError('sparse NF launch '+str(rc))
    return out.unsqueeze(0),1
