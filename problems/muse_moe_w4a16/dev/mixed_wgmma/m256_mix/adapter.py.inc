import functools
import hashlib
import time
import statistics
import torch
from torch.utils.cpp_extension import load_inline

MX_OPTS={}
MX_DEADLINE=float('inf')
# H2008483c0c6: vector mixed_parallel344.5,quad346.1,C1366.2us.
MX_DECODE=1
MX_FIRST=tuple((h,n) for h in (8,16,32) for n in (32,64))
MX_SECOND=((0,32),)+MX_FIRST


class MXBudget(RuntimeError):pass


def mx_need(seconds,where):
    if MX_DEADLINE-time.monotonic()<seconds:raise MXBudget(where)


@functools.lru_cache(None)
def mx_module():
    mx_need(40.,'native compile')
    arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
    mod=load_inline(name='m256_mix_'+hashlib.sha256(MX_CUDA.encode()).hexdigest()[:12],cpp_sources=MX_CPP,cuda_sources=MX_CUDA,functions=MX_FUNCTIONS,
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False)
    mod.rc_init();MX_OPTS['nt']=mod.mx_nt_init();mod.mx_opt_init();mod.mx_mixed_init();return mod


def mx_prepare(w):
    bank={}
    for key in ('w13','w2'):
        sf=w[key+'_sf'].view(torch.uint8);e,n,k16=sf.shape
        if bool(((sf&127)==127).any()):raise ValueError('nonfinite block scale')
        q=mw_repack_planes(w[key]);sf=sf.reshape(e,n//16,2,8,k16//2,2).permute(0,1,4,3,5,2).contiguous()
        if key=='w13':
            rr=torch.arange(n//32,device=q.device).reshape(-1,2);order=torch.cat((rr,rr+n//32),1).flatten()
            q=q.index_select(1,order).contiguous();sf=sf.index_select(1,order).contiguous()
        qr=torch.empty_like(q);sr=torch.empty_like(sf)
        rc_repack[(triton.cdiv(q.numel()//2,256),)](q,sf,qr,sr,q.numel()//2,RC_SIGNS,RC_ZEROS)
        bank[key]=qr;bank[key+'_sf']=sr
    bank['w13_gs']=torch.cat((w.get('w13_gs_gate',w['w13_gs']),w.get('w13_gs_up',w['w13_gs']))).float().contiguous()
    bank['w2_gs']=w['w2_gs'].float().contiguous();bank['pnw']=w.get('post_norm_w',w.get('post_expert_norm')).contiguous()
    dev=w['w13'].device;r=torch.arange(512,device=dev)[:,None];i=torch.arange(8,device=dev)
    maps=(torch.cat((r*8+i,r*8+i+4096),1),torch.arange(4096,device=dev).view(256,16))
    for key,rowmap in zip(('w13','w2'),maps):
        out=torch.empty((32,rowmap.shape[0],36864),device=dev,dtype=torch.uint8)
        for e in range(32):out[e]=m128_hh_repack(w[key][e],w[key+'_sf'][e],rowmap)
        bank[key+'_nt']=out
    bank['mod']=mx_module();bank['sms']=torch.cuda.get_device_properties(dev).multi_processor_count
    bank['mix_stream']=torch.cuda.Stream(device=dev);bank['mix_start']=torch.cuda.Event();bank['mix_done']=torch.cuda.Event()
    assert bank['w13_sf'].dtype==torch.uint8 and bank['w2_sf'].dtype==torch.uint8
    return bank


def mx_plan(data,bank,a=(16,32),b=(16,32),decoder=0,grid=2112,control=False):
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*(32 if control else 64)
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(c)
    task=torch.empty_like(off);t13=torch.empty_like(off);t2=torch.empty_like(off);slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();pos=torch.empty_like(ids)
    clear=(cap//32)*193+m;state=torch.empty(clear+m,device=dev,dtype=torch.int32)
    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);part=torch.empty(1,device=dev)
    y=torch.empty((cap,4096),device=dev);ss=torch.empty((cap,64),device=dev);inv=torch.empty(cap,device=dev);out=torch.empty_like(x)
    common=[c,off,order,task,slots,act];tail=[ids,tw,pos,bank['pnw'],y,ss,inv,state,out]
    v13=[bank['w13'],bank['w13_sf'],bank['w13_gs'],x]+common+[part]+tail
    v2=[bank['w2'],bank['w2_sf'],bank['w2_gs'],act]+common+[part]+tail
    if not control:v13[7]=t13;v2[7]=t2
    nv=[x,bank['pnw'],bank['w13_nt'],bank['w2_nt'],bank['w13_gs'],bank['w2_gs'],c,off,slots,act,y,ss]
    ptrs=[[v.data_ptr() for v in row] for row in (v13,v2,nv)]
    rp=[v.data_ptr() for v in (ids,c,off,order,task,slots,pos,state,out)];mp=[v.data_ptr() for v in (c,order,t13,t2,ss)]
    mod=bank['mod'];start=data['config'].get('local_start',0);owners=(v13,v2,nv,task,t13,t2,ids,tw)
    stages=2
    while stages>1 and mod.mx_nt_smem(stages,m)>MX_OPTS['nt']:stages-=1
    grids={f:bank['sms']*mod.mx_nt_resident(stages,m,f,1) for f in (0,1)}
    def stream():return torch.cuda.current_stream().cuda_stream
    def route():
        _=owners
        mod.rc_route(rp,m,32,start,clear,stream())
        if not control:mod.mx_meta(mp,cap,a[0],a[1],b[0],b[1],stream())
    def small(first,choice):
        _=owners
        if not control and choice[0]>0:mod.mx_nt_run(ptrs[2],first,1,m,stages,grids[first],choice[0],stream())
    def heavy(first,choice):
        _=owners
        if first and control==2:
            parent=torch.cuda.current_stream();bank['mix_start'].record(parent);bank['mix_stream'].wait_event(bank['mix_start'])
            with torch.cuda.stream(bank['mix_stream']):
                mod.mx_mixed_run(ptrs[0],0,grid,m,cap,start,bank['mix_stream'].cuda_stream);bank['mix_done'].record()
            mod.mx_mixed_run(ptrs[0],1,grid,m,cap,start,parent.cuda_stream);parent.wait_event(bank['mix_done'])
        elif first and decoder and not control:mod.mx_opt_run(ptrs[0],choice[1],decoder,grid,m,cap,start,stream())
        else:mod.rc_run(ptrs[0 if first else 1],first,1 if choice[1]==32 else 2,grid,start,1,0,m,cap,stream())
    def asmall():small(1,a)
    def aheavy():heavy(1,a)
    def bsmall():small(0,b)
    def bheavy():heavy(0,b)
    def first():asmall();aheavy()
    def second():bsmall();bheavy()
    def finish():
        _=owners
        mod.rc_finish(ptrs[1],m,start,stream());return out
    return dict(steps=(route,first,second,finish),components=(route,asmall,aheavy,bsmall,bheavy,finish),out=out,v13=v13,v2=v2,tasks=(t13,t2),owners=owners)


def mx_call(data,bank,a=(16,32),b=(16,32),decoder=0,grid=2112,control=False):
    for fn in mx_plan(data,bank,a,b,decoder,grid,control)['steps']:out=fn()
    return out


def mx_error(out,want):
    if not bool(torch.isfinite(out).all()):return float('inf'),float('inf')
    diff=out.float()-want.float();den=want.float().norm(dim=-1)
    return float(diff.norm()/want.float().norm().clamp_min(1e-12)),float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())
