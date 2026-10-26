# Geometry is selected by shape, never by activation values. All banks are exact
# BF16 codes, including the original per-tile exponent tables and escapes.
SGQ_CONFIGS=((4,1,2),(8,1,2),(4,2,1),(4,2,2),(4,4,1),(4,4,2))

def sgq_source(local_reduce=False):
    ns=_nf_vendor();insts=[];plans={}
    for t in (8,16):
        for n in (1280,2304):
            for arm,(wk,s,stages) in enumerate(SGQ_CONFIGS):
                if local_reduce:s=1
                cfg=(1,wk,8192//(64*wk*s),s)
                inst=(1,t//8,1,stages,2 if s>1 else 0,0,wk*32,0,n,8192,*cfg,6)
                if inst not in insts:insts.append(inst)
                plans[t,n,arm]=(insts.index(inst),cfg)
    return ns,insts,plans

def sgq_prepare(weights,local_reduce=False):
    ns,insts,plans=sgq_source(local_reduce);source=ns['v18s_cuda_source'](insts)
    tag=hashlib.sha256(source.encode()).hexdigest()[:12]
    mod=ns['v18s_build'](insts,'sgq_'+tag);banks={};packed={}
    for (t,n,arm),(idx,cfg) in plans.items():
        w=weights['qkv_gate'][:n].contiguous();wr,wk,ku,s=cfg;key=(n,cfg)
        if key not in packed:
            pk,tables,stream=_nf_group_pack(ns,w,'qkv_gate','v18');sp=stream(pk,1,*cfg)
            if sp is None:raise RuntimeError('SGQ exact escape capacity '+str(key))
            g=(n//16)*s*wk;gw=torch.arange(g,device=w.device);tile=(gw//wk)%(n//16)
            meta=torch.empty((g,34),device=w.device,dtype=torch.int32)
            meta[:,:32]=sp['esc'];meta[:,32:]=tables[tile].to(torch.int32)
            packed[key]=dict(sp,esc=meta)
        banks[t,n,arm]=dict(packed[key],idx=idx,cfg=cfg,fn=mod.v18s_launch)
    return dict(plans=banks,mod=mod,source_sha256=tag)

def sgq_project(x,n,arm,bank,query=0):
    p=bank['plans'][x.shape[0],n,arm]
    out=torch.empty((x.shape[0],n),device=x.device,dtype=torch.float32)
    rc=p['fn'](p['idx'],p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),x.shape[0],n,8192,*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream().cuda_stream,query)
    if query:return rc
    if rc:raise RuntimeError('SGQ launch '+str((x.shape[0],n,arm,rc)))
    return out.unsqueeze(0),1
