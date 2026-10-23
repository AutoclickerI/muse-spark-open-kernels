# Embedded in cand_r3_proj.py. Only shape choices and weight layouts survive prepare.
import time as _dp_time
import statistics as _dp_stats
_DP_MOD=None


class _DPModule:
    def __init__(self,path):
        import ctypes as ct
        self.ct=ct;self.lib=ct.CDLL(path);P,I,L,S=ct.c_void_p,ct.c_int,ct.c_int64,ct.c_size_t
        self.lib.lean_error_string.restype=ct.c_char_p
        self.lib.lean_algo_bytes.restype=I
        self.lib.dp_skin.argtypes=[P]*4+[I]*16+[P]
        self.lib.lean_pair.argtypes=[P]*5+[I]*13+[L,L,P]
        self.lib.all_lt_options.argtypes=[I]*7+[S,P,I,ct.POINTER(I),ct.POINTER(I)]
        self.lib.all_lt_run.argtypes=[P]*5+[I]*9+[S,P]
        self.algo_bytes=self.lib.lean_algo_bytes()

    def _check(self,status,context):
        if status:raise RuntimeError(f'{context}: status {status}: {self.lib.lean_error_string().decode()}')

    def skin(self,w,lo,x,out,cfg,dual=False,mix=False,zero_tail=False):
        nt,tr,u,wn,wk,ks,xp,ld=cfg
        self._check(self.lib.dp_skin(w.data_ptr(),lo.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],out.shape[1],x.shape[1],
            nt,tr,u,wn,wk,ks,2 if dual else 1,xp,ld,int(out.dtype==torch.float32),int(mix),out.stride(0),int(zero_tail),
            torch.cuda.current_stream(x.device).cuda_stream),'projection skinny/mix')

    def hop(self,w,x,out,scratch,cfg,zero_tail=False):
        bt,bk,st,mode,split,threads=cfg;work,counts=scratch
        self._check(self.lib.lean_pair(w.data_ptr(),x.data_ptr(),out.data_ptr(),work.data_ptr(),counts.data_ptr(),
            x.shape[0],out.shape[1],x.shape[1],w.shape[1],int(out.dtype==torch.float32),bt,bk,st,mode,split,threads,
            out.stride(0),int(zero_tail),work.numel(),counts.numel(),torch.cuda.current_stream(x.device).cuda_stream),'projection TMA')

    def options(self,w,x,out,nn,workspace):
        count=self.ct.c_int();h=self.ct.c_int();cap=65536
        codes=torch.empty((cap,self.algo_bytes),dtype=torch.uint8,device='cpu')
        self._check(self.lib.all_lt_options(out.shape[1],x.shape[0],x.shape[1],w.stride(0),nn,int(out.dtype==torch.float32),out.stride(0),
            workspace.numel(),codes.data_ptr(),cap,self.ct.byref(count),self.ct.byref(h)),'projection Lt options')
        return codes[:count.value].clone()

    def lt(self,w,x,out,workspace,codes,index,nn):
        if not 0<=index<codes.shape[0]:raise RuntimeError('projection Lt index outside prepared table')
        self._check(self.lib.all_lt_run(w.data_ptr(),x.data_ptr(),out.data_ptr(),workspace.data_ptr(),codes.data_ptr(),
            out.shape[1],x.shape[0],x.shape[1],w.stride(0),nn,int(out.dtype==torch.float32),out.stride(0),index,0,
            workspace.numel(),torch.cuda.current_stream(x.device).cuda_stream),'projection Lt run')


def _dp_module():
    global _DP_MOD
    if _DP_MOD is None:
        import hashlib,os
        from torch.utils.cpp_extension import load_inline
        arch='9.0a' if IS_HOPPER else '12.0';old=os.environ.get('TORCH_CUDA_ARCH_LIST')
        os.environ['TORCH_CUDA_ARCH_LIST']=arch
        try:
            tag=hashlib.sha1(_DP_CUDA_SRC.encode()).hexdigest()[:12]
            path=load_inline(name=f'muse_decode_proj_{tag}_{arch.replace(chr(46),chr(95))}',cpp_sources='',cuda_sources=_DP_CUDA_SRC,
                functions=None,is_python_module=False,no_implicit_headers=True,
                extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','--split-compile=16','-Xptxas=--split-compile=16'],
                extra_ldflags=['-lcublasLt'],verbose=False)
        finally:
            if old is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:os.environ['TORCH_CUDA_ARCH_LIST']=old
        _DP_MOD=_DPModule(path)
    return _DP_MOD


class _DPGraph:
    def __init__(self):
        import ctypes
        self.runtime=torch.cuda.cudart();ptr=ctypes.c_void_p()
        result=self.runtime.cudaStreamCreate(ctypes.addressof(ptr))
        if int(result):raise RuntimeError(f'tuning stream create: {result}')
        self.handle=ptr.value;self.stream=torch.cuda.ExternalStream(self.handle);self.graph=torch.cuda.CUDAGraph()

    def replay(self):self.graph.replay()

    def __del__(self):
        if hasattr(self,'handle'):
            self.graph.reset();self.runtime.cudaStreamDestroy(self.handle)


def _measure_graph(fn,flush,repeats=3):
    graph=_DPGraph();graph.stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(graph.stream):fn()
    graph.stream.synchronize()
    with torch.cuda.graph(graph.graph,stream=graph.stream):out=fn()
    graph.replay();torch.cuda.synchronize()
    vals=[]
    for _ in range(repeats):
        flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000)
    return float(_dp_stats.median(vals))


def _dp_left(state,seconds=0):return _dp_time.monotonic()+seconds<state['_tune_deadline']


def _dp_reject(family,key,variant,exc):
    print(f'{family}_REJECT key={key} variant={variant} reason={str(exc).splitlines()[0][:180]}',flush=True)


def _dp_agree(value,control,limit=0.01):
    if isinstance(value,(tuple,list)):
        return len(value)==len(control) and all(_dp_agree(x,y,limit) for x,y in zip(value,control))
    if value.shape!=control.shape or not bool(torch.isfinite(value).all()):return False
    x,y=value.float(),control.float();den=y.norm().clamp_min(1e-12)
    if float((x-y).norm()/den)>limit:return False
    if x.ndim>=2:
        a=(x-y).reshape(x.shape[0],-1).norm(dim=1);b=y.reshape(y.shape[0],-1).norm(dim=1)
        if float((a/b.clamp_min(b.mean()*0.1+1e-12)).max())>5*limit:return False
    return True


def _dp_skin_cfg(name,t):
    layer={'pool_router':'pooler','unpooler_slice':'unpooler'}.get(name,name)
    nt=1 if t<=8 else 2 if t<=16 else 4 if t<=32 else 8
    bucket=t if t in (1,8,16,32) else 64
    tr,u,wn,wk,ks,xp=DP_SKIN_CONFIGS.get((layer,bucket),(1,2,2,4,1,0))
    return nt,tr,u,wn,wk,ks,xp,0


def _project_variant(x,w,name,state,variant,out_dtype=torch.bfloat16):
    if variant=='custom':return gemv(x,w,out_dtype=out_dtype)
    if variant=='cublas':return _dense_project(x,w,out_dtype).unsqueeze(0),1
    mod=state['dp_mod'];t=x.shape[0]
    out=torch.empty((t,w.shape[0]),device=x.device,dtype=out_dtype)
    tag=variant[0]
    if tag=='skin':mod.skin(w,w,x,out,variant[1])
    elif tag=='mix':mod.skin(state['dp_mix_hi'],state['dp_router_lo'],x,out,variant[1],dual=True,mix=True)
    elif tag=='hop':mod.hop(w,x,out,state['dp_scratch'][(t,name)],variant[1])
    elif tag=='lt':
        nn,index=variant[1:];weight=state['dp_transposed'][name] if nn else w
        mod.lt(weight,x,out,state['dp_workspace'],state['dp_codes'][(t,name,nn)],index,nn)
    elif tag=='pool_split':
        nn,index,router_kind,cfg=variant[1:]
        pool=out[:,:D_EXPERT];router=out[:,D_EXPERT:D_EXPERT+N_EXPERTS]
        weight=state['dp_pool_t'] if nn else state['dp_pool']
        mod.lt(weight,x,pool,state['dp_workspace'],state['dp_codes'][(t,'pool_only',nn)],index,nn)
        if router_kind=='skin':mod.skin(state['dp_router_hi'],state['dp_router_lo'],x,router,cfg,dual=True,zero_tail=True)
        else:mod.hop(state['dp_router_pair'],x,router,state['dp_scratch'][(t,'router')],cfg,zero_tail=True)
    else:raise RuntimeError(f'unknown projection path {variant}')
    return out.unsqueeze(0),1


def _dp_canonical(value,name):
    y=_materialize(value).float()
    if name=='pool_router':return y[:,:D_EXPERT],y[:,D_EXPERT:D_EXPERT+N_EXPERTS]+y[:,D_EXPERT+N_EXPERTS:]
    return y


def _dp_consumer(x,w,name,state,variant,stream,data):
    projected=_project_variant(x,w,name,state,variant,torch.float32 if name=='pool_router' else torch.bfloat16)
    t=x.shape[0];sync=(torch.zeros((t,4),device=x.device),torch.zeros((t,),device=x.device,dtype=torch.int32))
    if name=='qkv_gate':return qk_prep(projected,data,state,q_fp8=True)
    if name=='o':return boundary(projected,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],sync)
    if name=='pool_router':return route_and_norm(projected,state,data,sync)
    return boundary(projected,stream,state['alpha2'],state['beta2'],state['next_pre_attn_w'],sync,postw=state['post_ffn_w'])


def _dp_variants(state,name,t,w,x):
    yield 'cublas';yield 'custom'
    if state.get('dp_mod') is None:return
    cfg=_dp_skin_cfg(name,t)
    yield ('skin',cfg)
    if name=='pool_router':
        nt,u,xp=1 if t<=8 else 2 if t<=16 else 4 if t<=32 else 8,4 if t<=32 else 2,1 if t<=16 else 0
        for wn,wk,ks in ((1,4,1),(2,4,8),(1,8,4),(2,2,4)):
            if x.shape[1]%(wk*32*u*ks)==0:yield ('mix',(nt,1,u,wn,wk,ks,xp,0))
    else:
        for bt,bk,st,_,_,th in DP_PAIR_CONFIGS:
            if bt>max(64,t):continue
            shared=max(st*(64+bt)*bk*2+st*16,bt*68*4)
            if shared>torch.cuda.get_device_properties(x.device).shared_memory_per_block_optin:continue
            for mode,split in ((0,1),(0,4),(1,2),(1,4),(2,132)):
                yield ('hop',(bt,bk,st,mode,split,th))
    for nn in (0,1):
        for index in range(state['dp_codes'].get((t,name,nn),torch.empty(0)).shape[0]):yield ('lt',nn,index)
    if name=='pool_router':
        bucket=t if t in (1,8,16,32) else 64
        tr,u,wn,wk,ks,xp=DP_SKIN_CONFIGS.get(('router',bucket),(1,2,2,4,8,0))
        rskin=(1 if t<=8 else 2 if t<=16 else 4 if t<=32 else 8,tr,u,wn,wk,ks,xp,0)
        for nn in (0,1):
            # All pooler algorithms also compete as a strided-output combined path.
            for index in range(state['dp_codes'].get((t,'pool_only',nn),torch.empty(0)).shape[0]):
                yield ('pool_split',nn,index,'skin',rskin)
                if t>=128:yield ('pool_split',nn,index,'hop',(64,64,4,1,8,160 if t==128 else 256))


def _tune_projections(state,flush,device):
    plan={};state.update(dp_codes={},dp_scratch={},dp_transposed={})
    names=('qkv_gate','o','pool_router','unpooler_slice','unpooler')
    for t in _TUNE_TOKENS:
        for name in names:
            if name=='unpooler_slice' and t>8 or name=='unpooler' and t<=8:continue
            plan[(t,name)]='cublas'
    if not _dp_left(state,8):return plan
    try:
        state['dp_mod']=_dp_module()
        state['dp_workspace']=torch.empty(512<<20,device=device,dtype=torch.uint8)
        state['dp_transposed']={name:state[name].T.contiguous() for name in names}
        state['dp_pool']=state['pool_router'][:D_EXPERT].contiguous()
        state['dp_pool_t']=state['dp_pool'].T.contiguous()
        state['dp_router_hi']=state['pool_router'][D_EXPERT:D_EXPERT+N_EXPERTS].contiguous()
        state['dp_router_lo']=state['pool_router'][D_EXPERT+N_EXPERTS:].contiguous()
        state['dp_mix_hi']=state['pool_router'][:D_EXPERT+N_EXPERTS].contiguous()
        state['dp_router_pair']=torch.cat((state['dp_router_hi'],state['dp_router_lo']),1).contiguous()
    except Exception as exc:
        _dp_reject('PROJ_MODULE','all','native',exc);state['dp_mod']=None
    stop=min(state['_tune_deadline'],_dp_time.monotonic()+75)
    for t in _TUNE_TOKENS:
        for name in names:
            if (t,name) not in plan:continue
            if _dp_time.monotonic()>=stop:
                print('PROJ_BUDGET exhausted; remaining shapes retain validated cuBLAS',flush=True);return plan
            w=state[name];out_dtype=torch.float32 if name=='pool_router' else torch.bfloat16
            x=torch.randn((t,w.shape[1]),device=device,dtype=torch.bfloat16)
            stream=torch.randn((t,HIDDEN),device=device,dtype=torch.float32)
            data={'positions':torch.arange(t,device=device),'cache_loc':torch.arange(t,device=device),
                  'k_cache':torch.empty((t,1,HKV,HD),device=device,dtype=FP8),'v_cache':torch.empty((t,1,HKV,HD),device=device,dtype=FP8),
                  'k_scale':torch.ones((),device=device),'v_scale':torch.ones((),device=device),'local_start':torch.zeros((),device=device,dtype=torch.int32),
                  'config':dict(tokens=t,rope_theta=ROPE_THETA,qk_scale_factor=5.4730064863875)}
            if state.get('dp_mod') is not None:
                for label,n in ((name,w.shape[0]),)+( (('router',N_EXPERTS),) if name=='pool_router' else () ):
                    state['dp_scratch'][(t,label)]=(torch.empty(((t+63)//64)*64*n*17,device=device,dtype=torch.float32),torch.zeros(((t+63)//64)*(n//64),device=device,dtype=torch.uint32))
                for label,weight,n,kind in ((name,w,w.shape[0],out_dtype),)+( (('pool_only',state['dp_pool'],D_EXPERT,torch.float32),) if name=='pool_router' else () ):
                    out=torch.empty((t,w.shape[0]),device=device,dtype=kind)[:,:n]
                    for nn in (0,1):
                        try:
                            wt=(state['dp_pool_t'] if label=='pool_only' else state['dp_transposed'][name]) if nn else weight
                            state['dp_codes'][(t,label,nn)]=state['dp_mod'].options(wt,x,out,nn,state['dp_workspace'])
                        except Exception as exc:_dp_reject('PROJ_LT',(t,label),nn,exc)
            scores=[]
            for variant in _dp_variants(state,name,t,w,x):
                if _dp_time.monotonic()>=stop:break
                try:
                    # cuBLAS is the independent H200-proven control, not the first custom trial.
                    for seed in range(2):
                        if seed:x.copy_(torch.randn_like(x))
                        want=_dp_canonical(_project_variant(x,w,name,state,'cublas',out_dtype),name)
                        y=_dp_canonical(_project_variant(x,w,name,state,variant,out_dtype),name)
                        if not _dp_agree(y,want):raise ValueError('projection exceeds 1% proven-control guard')
                    fn=lambda v=variant:_dp_consumer(x,w,name,state,v,stream,data)
                    scores.append((variant,_measure_graph(fn,flush,repeats=2)))
                except Exception as exc:_dp_reject('PROJ',(t,name),variant,exc)
            if scores:
                finalists=sorted(scores,key=lambda q:q[1])[:3];paired=[]
                for variant,_ in finalists:
                    if _dp_time.monotonic()>=stop:break
                    try:
                        paired.append((variant,_measure_graph(lambda v=variant:_dp_consumer(x,w,name,state,v,stream,data),flush,repeats=5)))
                    except Exception as exc:_dp_reject('PROJ_FINAL',(t,name),variant,exc)
                if paired:plan[(t,name)]=min(paired,key=lambda q:q[1])[0]
            print(f'PROJ_TUNE t={t} name={name} best={plan[(t,name)]} consumer_us={min((v for _,v in scores),default=float("inf")):.3f} accepted={len(scores)}',flush=True)
    return plan


def _tune_fa3(flush,device,state):
    plan={}
    if not _fa3_on_device(device):return plan
    for b in (1,16,32):
        for w in (2048,0):
            if not _dp_left(state,5):return plan
            plan[(b,w)]=(0,None)
            try:
                q,gate,data=_fa3_synthetic(b,w,device)
                control=_fa3_call(q,gate,data,(0,None));scores=[]
            except Exception as exc:_dp_reject('FA3',(b,w),'control',exc);continue
            for setting in ((s,p) for s in _FA3_SPLITS for p in _FA3_PACK):
                if not _dp_left(state,2):break
                try:
                    out=_fa3_call(q,gate,data,setting)
                    if not _dp_agree(out,control):raise ValueError('exceeds 1% H200-proven FA3 control')
                    scores.append((_measure_graph(lambda s=setting:_fa3_call(q,gate,data,s),flush,3),setting))
                except Exception as exc:_dp_reject('FA3',(b,w),setting,exc)
            if scores:plan[(b,w)]=min(scores,key=lambda q:q[0])[1]
            print(f'FA3_TUNE batch={b} window={w} best={plan[(b,w)]} accepted={len(scores)}',flush=True)
            del q,gate,data,control
    return plan


def _tune_pdl(state,flush,device):
    plan={}
    for t in _TUNE_TOKENS:
        for phase in ('b1','b2'):
            if not _dp_left(state,2):return plan
            stream=torch.randn((t,HIDDEN),device=device,dtype=torch.float32)
            n=HIDDEN if phase=='b1' or t>8 else UNPOOL_SLICE
            branch=(torch.randn((1,t,n),device=device),1)
            def run(enabled):
                sync=(torch.zeros((t,4),device=device),torch.zeros(t,device=device,dtype=torch.int32))
                if phase=='b1':return boundary(branch,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],sync,use_pdl=enabled)
                return boundary(branch,stream,state['alpha2'],state['beta2'],state['next_pre_attn_w'],sync,postw=state['post_ffn_w'],use_pdl=enabled)
            scores=[]
            try:control=run(True)
            except Exception as exc:_dp_reject('PDL',(t,phase),'control',exc);continue
            for enabled in (True,False):
                try:
                    if not _dp_agree(run(enabled),control):raise ValueError('exceeds proven-control guard')
                    scores.append((_measure_graph(lambda e=enabled:run(e),flush,3),enabled))
                except Exception as exc:_dp_reject('PDL',(t,phase),enabled,exc)
            if scores:plan[(t,phase)]=min(scores,key=lambda q:q[0])[1]
    return plan


def _tune_moe(state,device,flush):
    # Full-layer H200 receipt 0e94268cf5654012: pure V9 = 1.427904x.
    # Isolated synthetic routing selected grouped/V7 paths which regressed the
    # actual layer (2ad75c19cb514f5c). Keep the measured layer control while
    # this candidate tests projection/FA3/PDL choices.
    return {m:'v9' for m in _TUNE_TOKENS}


def prepare(weights):
    started=_dp_time.monotonic();state=_BASE_PREPARE(weights);device=weights['qkv_gate'].device
    state['_tune_deadline']=started+240
    flush=torch.zeros(128<<20,device=device,dtype=torch.uint8)
    state['proj_plan']=_tune_projections(state,flush,device)
    state['moe_plan']=_tune_moe(state,device,flush)
    state['fa3_plan']=_tune_fa3(flush,device,state)
    state['pdl_plan']=_tune_pdl(state,flush,device)
    torch.cuda.synchronize(device)
    print(f'PROJ_LAYER prepare_s={_dp_time.monotonic()-started:.2f} proj={state["proj_plan"]} moe={state["moe_plan"]} fa3={state["fa3_plan"]} pdl={state["pdl_plan"]}',flush=True)
    return state
