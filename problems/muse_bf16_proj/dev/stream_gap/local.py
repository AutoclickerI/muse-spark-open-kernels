"""Local stand-ins only for the unchanged Hopper router and large VDSK launch."""
import torch

def install(m,forced=False):
    calls={};real_router_prepare=m.sg_router_prepare;real_router=m.sg_router
    class Mock:
        def __getattr__(self,name):
            def run(*args):
                x,w,registry=context['x'],context['w'],context['registry'];t=x.shape[0];stream=torch.cuda.current_stream().cuda_stream
                if name=='rt2_launch':assert args[2]==x.data_ptr() and args[4:]==(t,stream,0);outptr=args[3]
                elif name=='rt2_scalar':assert args[:2]==(w.data_ptr(),x.data_ptr()) and args[3:]==(t,stream,0);outptr=args[2]
                elif name=='rt2_small':assert args[1]==x.data_ptr() and args[3:]==(t,16,8,stream,0);outptr=args[2]
                else:raise AssertionError(name)
                out=registry[outptr]
                if name=='rt2_scalar':out.copy_(x.float()@w.T)
                else:
                    hi=w.bfloat16();lo=(w-hi.float()).bfloat16();out.copy_(torch.mm(x,hi.T,out_dtype=torch.float32)+torch.mm(x,lo.T,out_dtype=torch.float32))
                calls[name]=calls.get(name,0)+1;return 0
            return run
    context={}
    def prepare(w):
        # Original packers and exported launch wrappers are exercised. Only
        # Hopper device arithmetic is replaced, never the input/plan ABI.
        load=m.load_inline
        try:
            m.load_inline=lambda *a,**kw:Mock()
            # The router port owns an isolated imported loader.
            import linecache
            file='<sg_nf_router_local>';linecache.cache[file]=(len(m.SG_ROUTER_SOURCE),None,m.SG_ROUTER_SOURCE.splitlines(True),file)
            ns={'__name__':'sg_nf_router_local'};exec(compile(m.SG_ROUTER_SOURCE,file,'exec'),ns)
            ns['load_inline']=lambda *a,**kw:Mock()
            st=ns['rt2_prepare_candidates']({'router':w},ns['rt2_ROUTER_CONFIGS']);st['small']={(16,8):ns['rt2_pack_small_router'](w,16,8)}
            return ns,st
        finally:m.load_inline=load
    def router(x,bank):
        registry={};empty=torch.empty
        def track(*a,**kw):
            out=empty(*a,**kw);registry[out.data_ptr()]=out;return out
        context.update(x=x,w=bank[1]['weight'],registry=registry);torch.empty=track
        try:return real_router(x,bank)
        finally:torch.empty=empty
    m.sg_router_prepare=prepare;m.sg_router=router;m._P12_FORCE=forced
    if forced:
        class VD:
            def __getitem__(self,grid):
                def run(x,w,p,xd,wd,t,n,k,s,**kw):
                    assert p.shape==(s,t,n);p.zero_();p[0].copy_(torch.mm(x,w.T,out_dtype=torch.float32));calls['vdsk']=calls.get('vdsk',0)+1
                return run
        m._vdsk_gemm=VD()
    return calls
