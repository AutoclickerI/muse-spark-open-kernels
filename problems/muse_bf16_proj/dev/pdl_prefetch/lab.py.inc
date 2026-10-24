import gc
import linecache
import math
import statistics
import time

PP_LOCAL_INSTALL=None
PP_ROWS=[]
PP_MODULE=None
PP_CASES=tuple((b,q,w) for b,q in ((1,1),(16,1),(32,1),(1,8)) for w in (2048,0))
PP_SEEDS=(172903,180822,188741)
PP_WEIGHT_SEED=72140
PP_NAMES={'qkv_gate':'Q','o':'O','pool_router':'P','unpooler':'U','unpooler_slice':'U'}


class PPSpace:
    def __init__(self,ns):object.__setattr__(self,'ns',ns)
    def __getattr__(self,k):
        try:return self.ns[k]
        except KeyError:raise AttributeError(k) from None
    def __setattr__(self,k,v):self.ns[k]=v


def pp_module():
    filename='<dense_pdl_best7>'
    linecache.cache[filename]=(len(PP_BEST7_SOURCE),None,PP_BEST7_SOURCE.splitlines(True),filename)
    ns={'__name__':__name__,'__builtins__':__builtins__}
    exec(compile(PP_BEST7_SOURCE,filename,'exec'),ns)
    m=PPSpace(ns)
    if PP_LOCAL_INSTALL is not None:PP_LOCAL_INSTALL(m)
    return m


def pp_weights(m,host):
    # A per-name seed makes each synthetic tensor independent of host coverage.
    # The expert bank remains packed NVFP4; only the first EP8 shard is needed.
    device=next(v.device for v in host.values() if isinstance(v,torch.Tensor))
    bf=torch.bfloat16;fp=torch.float32;sf=torch.float8_e4m3fn
    specs={
        'qkv_gate':((2304,8192),bf),'o':((8192,1024),bf),
        'pooler':((4096,8192),bf),'unpooler':((8192,4096),bf),
        'router':((256,8192),fp),'router_bias':((256,),fp),
        'attn_residual_gate':((8192,),bf),'pre_ffn_norm':((8192,),bf),
        'pre_expert_norm':((4096,),bf),'post_expert_norm':((4096,),bf),
        'post_ffn_norm':((8192,),bf),'ffn_residual_gate':((8192,),bf),
        'next_pre_attn_norm':((8192,),bf),
        'w13':((32,8192,2048),torch.uint8),'w13_sf':((32,8192,256),sf),
        'w13_gs':((32,),fp),'w2':((32,4096,2048),torch.uint8),
        'w2_sf':((32,4096,256),sf),'w2_gs':((32,),fp)}
    out={};synthetic=[];provided=[]
    for name,(shape,dtype) in specs.items():
        source='post_norm_w' if name=='post_expert_norm' and name not in host and 'post_norm_w' in host else name
        if source in host:
            value=host[source]
            if name.startswith(('w13','w2')) and value.ndim==len(shape) and value.shape[0]>=32:
                value=value[:32]
            if tuple(value.shape)!=shape or value.dtype!=dtype or value.device!=device:
                raise ValueError(f'host tensor {source}: {tuple(value.shape)}/{value.dtype}, expected {shape}/{dtype}')
            out[name]=value;provided.append(name+('='+source if source!=name else ''))
            continue
        seed=PP_WEIGHT_SEED+int(_pphash.sha256(name.encode()).hexdigest()[:8],16)
        gen=torch.Generator(device=device).manual_seed(seed)
        if dtype==torch.uint8:
            value=torch.randint(0,256,shape,generator=gen,device=device,dtype=dtype)
        elif dtype==sf:
            value=torch.exp2(7.+.4*torch.randn(shape,generator=gen,device=device)).clamp(8.,448.).to(sf)
        elif name.endswith('_gs'):
            mean_sq=4.**7*math.exp(2*(math.log(2)*.4)**2)
            value=torch.full(shape,1./(8.6*mean_sq*4096)**.5,device=device,dtype=fp)
        elif name=='router':
            mean,_=m._ap_stream_basis(str(device))
            value=.045*torch.randn(shape,generator=gen,device=device)
            value+=.07*torch.randn((256,1),generator=gen,device=device)*mean[None,:]/8192**.5
        else:
            scale=shape[1]**-.5 if len(shape)==2 else (.055 if name=='router_bias' else .3 if name.endswith('residual_gate') else .1)
            value=(scale*torch.randn(shape,generator=gen,device=device)).to(dtype)
        out[name]=value.contiguous();synthetic.append(name)
    return out,synthetic,provided


def pp_state(m,weights):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        lt=pool.submit(m._lt_module);attn=pool.submit(m._attn_port_ext);layer=pool.submit(m._layer_attn_ext)
        state=m._prepare_weights(weights)
        state['lt_mod']=lt.result();attn.result();layer.result()
    m._seed_v9_gemv_choices(state)
    state.update(proj_plan={},attention_plan={},prefix_plan={},pdl_plan={},moe6_plan={},moe_nt_plan={},
                 best7_stream_ops={},best7_prefetch_ops={})
    for b,q,w,attn,proj in PP_CHOICES:
        state['proj_plan'].update({(b*q,n):c for n,c in proj.items()})
        state['attention_plan'][b,q,bool(w)]=attn
    state['lt_workspace']=torch.empty(32<<20,device=weights['o'].device,dtype=torch.uint8)
    state['lt_algos']={}
    for (tokens,name),choice in state['proj_plan'].items():
        if isinstance(choice,tuple) and choice[0]=='lt':
            x=torch.zeros((tokens,state[name].shape[1]),device=state[name].device,dtype=torch.bfloat16)
            y=torch.empty((tokens,state[name].shape[0]),device=x.device,dtype=torch.bfloat16)
            codes=state['lt_mod'].lt_options(state[name],x,y)
            if codes.shape[0]<=choice[1]:raise RuntimeError('baseline Lt ordinal unavailable')
            state['lt_algos'][name,tokens]=codes
    return state


def pp_plan(m,key):
    glue,attention,moe,projection=m.BEST7_SETTINGS[key]
    geometry={}
    if isinstance(glue,tuple):glue,b1,route,b2=glue;geometry=dict(b1=b1,route=route,b2=b2)
    if projection!='auto':raise RuntimeError('baseline gained a dense override: regenerate the experiment')
    return dict(glue=glue,geometry=geometry,attention=attention,moe=moe,projection={},prefetch=0,dense_chain=False)


def pp_spec(tokens,name,variant):
    ks=512 if tokens>=32 else 1024
    if variant=='R0':return 0,ks
    if variant=='R1':return 1,512
    if variant=='R2':return 2,ks
    if variant=='P0':return 5,ks
    if variant=='P1':return 6,512
    if variant=='P2':return 7,ks
    if variant=='W':return (3,1024) if name=='o' else (4,2048)
    raise ValueError(variant)


def pp_ops(state,tokens,variant,early=True):
    roles=('qkv_gate','o','pool_router','unpooler_slice' if tokens<=8 else 'unpooler')
    return {name:PPDense(state[name],tokens,*pp_spec(tokens,name,variant),early=early) for name in roles}


def pp_layer(m,data,state,plan,variant='B',ops=None,only=None,routes_out=None):
    tokens=data['config']['tokens'];trace={};sync=[None,None,None]
    if plan['glue']=='best5':
        raw=(torch.empty if m._best7_old_chunks(tokens)==1 else torch.zeros)((15*tokens,),device=data['stream'].device,dtype=torch.int32)
        si=raw[:3*tokens].view(3,tokens);sf=raw[3*tokens:].view(torch.float32).view(3,tokens,4)
        sync=[(sf[i],si[i]) for i in range(3)]
    def project(x,name):
        selected=variant if only is None or only==name else 'B'
        # Local admission: QKV BF16 tie rounding is amplified by q8 sliding FP8
        # attention beyond the .003 whole-layer bound. Keep that failed cell out
        # of compound trials; every standalone QKV kernel is still measured.
        if (data['config']['batch'],data['config']['q_len'],data['config']['window'])==(1,8,2048) and \
            name=='qkv_gate' and selected in ('A0','A4','R0','P0','P2'):selected='B'
        if selected=='B':value=m._best7_project(x,name,state,plan,out_dtype=torch.float32 if name=='pool_router' else torch.bfloat16)
        elif selected.startswith('H'):value=pp_hint_project(m,x,state[name],name,state,int(selected[1:]),True)
        elif selected.startswith('A'):value=pp_hint_project(m,x,state[name],name,state,int(selected[1:]),False)
        else:value=ops[name](x)
        if selected[0] in ('A','R','P','W'):value=pp_preserve_boundary(m,x,name,state,value)
        trace[name]=(x,value)
        return value
    qkv=project(data['attn_input'],'qkv_gate')
    attn=m._best7_attention(qkv,data,state,plan)
    a=project(attn,'o')
    h,s1=m._best7_boundary(a,data['stream'],state,plan,'b1',sync[0])
    pr=project(h,'pool_router')
    x,ids,wts,lids=m._best7_route(pr,state,data,plan,sync[1])
    if routes_out is not None:routes_out.extend((ids,wts,lids))
    # Keep the timed routing/norm kernel, but feed the experts one common route
    # snapshot per input stream. This is a fixed-workload transport experiment:
    # small rounding changes must not change the amount of expert work timed.
    if '_pp_routes' in data:ids,wts,lids=data['_pp_routes']
    moe=m._best7_expert(pr,x,wts,lids,state,plan)
    f=project(moe,'unpooler_slice' if tokens<=8 else 'unpooler')
    hidden,s2=m._best7_boundary(f,s1,state,plan,'b2',sync[2])
    return (hidden,s2,ids),trace


def pp_value(pair):
    p,s=pair
    return p.sum(0) if s>1 else p[0]


def pp_error(got,want):
    a,b=got.float(),want.float();d=(a-b).norm(dim=-1);n=b.norm(dim=-1)
    if not bool(torch.isfinite(a).all()):raise RuntimeError('nonfinite output')
    return float(d.norm()/n.norm().clamp_min(1.e-20)),float((d/n.clamp_min(.1*n.mean()).clamp_min(1.e-20)).max())


def pp_close(got,want,tol,label):
    error,row=pp_error(got,want)
    if error>tol or row>5*tol:raise RuntimeError(f'{label}: {error:.3g}/{row:.3g} bound={tol}')
    return error


def pp_gate(out,trace,base,weights,state):
    worst=0.
    for name,(x,partial) in trace.items():
        raw=pp_value(partial);reference=x.float()@state[name].float().T
        tol=.01 if partial[0].dtype==torch.bfloat16 else .0001
        worst=max(worst,pp_close(raw,reference,tol,name))
        if name=='pool_router':
            pp_close(raw[:,4096:4352]+raw[:,4352:4608],x.float()@weights['router'].float().T,.0001,'router FP32')
    if not torch.equal(out[2],base[2]):raise RuntimeError('whole-layer expert IDs differ')
    for got,ref in zip(out[:2],base[:2]):pp_close(got,ref,.003,'whole-layer')
    return worst


def pp_graph(fn):
    for _ in range(2):fn()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    graph.replay();torch.cuda.synchronize()
    return graph,out


def pp_refresh(data,bank):
    for name in ('attn_input','stream','local_start'):data[name].copy_(bank[name])
    if '_pp_routes' in data:
        for dst,src in zip(data['_pp_routes'],bank['routes']):dst.copy_(src)


def pp_pair(a,b,flush,rounds=6,data=None,banks=None):
    values=[[],[]];ea=torch.cuda.Event(enable_timing=True);eb=torch.cuda.Event(enable_timing=True)
    for rep in range(rounds):
        if data is not None:pp_refresh(data,banks[rep%len(banks)])
        for side in ((0,1) if rep%2==0 else (1,0)):
            flush.add_(1);ea.record();(a if side==0 else b).replay();eb.record();eb.synchronize()
            values[side].append(ea.elapsed_time(eb)*1000.)
    ratios=[b/a for a,b in zip(*values)]
    return dict(a=statistics.mean(values[0]),b=statistics.mean(values[1]),ratio=statistics.mean(ratios),
        seed_ratios=[statistics.mean(ratios[s::3]) for s in range(3)] if rounds>=3 else [],samples=values)


def pp_standalone(m,state,weights,flush,deadline,lines):
    gen=torch.Generator(device=weights['o'].device).manual_seed(72141)
    sink=torch.empty(528*512,device=weights['o'].device,dtype=torch.int32)
    for tokens in (1,16,32,8):
        for name in ('qkv_gate','o','pool_router','unpooler_slice' if tokens<=8 else 'unpooler'):
            if time.monotonic()>deadline-3:raise TimeoutError('standalone dense coverage incomplete')
            w=state[name];x=torch.randn((tokens,w.shape[1]),device=w.device,dtype=torch.bfloat16,generator=gen)
            # Include exact-column probes while retaining nontrivial M1 rows.
            if tokens>1:
                x[::8]=0;x[::8,17%w.shape[1]]=8
            baseline,bv=pp_graph(lambda:m.project(x,w,name,state,out_dtype=torch.float32 if name=='pool_router' else torch.bfloat16))
            def read():
                err=pp_library().pp_read(w.data_ptr(),sink.data_ptr(),w.numel()*2,528,torch.cuda.current_stream(w.device).cuda_stream)
                if err:raise RuntimeError('read control CUDA='+str(err))
                return sink
            rg,_=pp_graph(read);read_pair=pp_pair(rg,rg,flush)
            inputs=[x.clone()]+[torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=gen) for _ in range(2)]
            trials=[]
            for v in ('R0','P0','P1','W') if tokens==1 else ('R0','P0','P1','P2','W'):
                cfg,ks=pp_spec(tokens,name,v);op=PPDense(w,tokens,cfg,ks)
                x.copy_(inputs[0])
                graph,out=pp_graph(lambda:pp_preserve_boundary(m,x,name,state,op(x)))
                expected=x.float()@w.float().T
                tol=.01 if out[0].dtype==torch.bfloat16 else .0001
                pp_close(pp_value(out),expected,tol,'standalone producer ABI')
                for fresh in inputs[1:]:
                    x.copy_(fresh);graph.replay();expected=x.float()@w.float().T
                    pp_close(pp_value(out),expected,tol,'fresh standalone')
                    if name=='pool_router':
                        raw=pp_value(out);pp_close(raw[:,4096:4352]+raw[:,4352:],x.float()@weights['router'].float().T,.0001,'standalone router')
                x.copy_(inputs[0])
                pair=pp_pair(baseline,graph,flush,6)
                late=PPDense(w,tokens,cfg,ks,early=False)
                lg,_=pp_graph(lambda:pp_preserve_boundary(m,x,name,state,late(x)))
                control=pp_pair(lg,graph,flush,6)
                trials.append(dict(variant=v,config=cfg,ks=ks,ctas=op.ctas,flight=op.flight_per_cta,weight_bytes=op.bank.numel()*2,
                                   read_us=read_pair['a'],early_late=control['ratio'],**pair))
                graph.reset();lg.reset()
            best=min(trials,key=lambda r:r['ratio']);PP_ROWS.append(dict(kind='stage',tokens=tokens,role=name,trials=trials))
            lines.append(f'S {tokens} {PP_NAMES[name]} rd={read_pair["a"]:.2f} A={best["a"]:.2f} '+
                ' '.join(f'{r["variant"]}:{r["b"]:.2f}/{r["early_late"]:.3f}' for r in trials)+
                f' ct={best["ctas"]} kB={best["flight"]//1024}')
            baseline.reset();rg.reset()


def pp_full(m,state,weights,flush,deadline,lines):
    for case,key in enumerate(PP_CASES):
        if time.monotonic()>deadline-4:raise TimeoutError(f'whole-layer coverage incomplete at {key}')
        b,q,w=key;tokens=b*q;plan=pp_plan(m,key)
        data=m._ap_generate_input(batch=b,ctx=80000,layer='sliding' if w else 'full',q_len=q,
            spread='agentic' if b>1 else 'fixed',dist='soft',page=1,seed=PP_SEEDS[0],device=weights['o'].device)
        banks=[]
        for seed in PP_SEEDS:
            fresh=m._ap_generate_input(batch=b,ctx=257,layer='sliding' if w else 'full',q_len=q,
                spread='agentic' if b>1 else 'fixed',dist='soft',page=1,seed=seed,device=weights['o'].device)
            bank={k:fresh[k].clone() for k in ('attn_input','stream','local_start')}
            pp_refresh(data,bank);out,_=pp_layer(m,data,state,plan)
            bank['local_start'].fill_(32*int(torch.bincount((out[2].long()//32).flatten(),minlength=8).argmax()))
            pp_refresh(data,bank)
            routes=[];pp_layer(m,data,state,plan,routes_out=routes)
            bank['routes']=tuple(v.clone() for v in routes)
            banks.append(bank)
        data['_pp_routes']=tuple(v.clone() for v in banks[0]['routes'])
        base,(bv,bt)=pp_graph(lambda:pp_layer(m,data,state,plan))
        refs=[]
        for bank in banks:
            pp_refresh(data,bank);base.replay();refs.append(tuple(v.clone() for v in bv))
            pp_gate(bv,bt,refs[-1],weights,state)
        # Check the lab's traced baseline against the exact best7 run function.
        exact,ev=pp_graph(lambda:m._best7_run(data,state,plan,None))
        for bank,ref in zip(banks,refs):
            pp_refresh(data,bank);exact.replay()
            for actual,want in zip(ev[:2],ref[:2]):pp_close(actual,want,.003,'exact best7 baseline')
            if not torch.equal(ev[2],ref[2]):raise RuntimeError('baseline transcription changed IDs')
        noise=pp_pair(base,exact,flush,6,data,banks);exact.reset()
        trials=[];rejects=[]
        variants=('H1','H4','A0','A4','R0','P0','P1','W' if tokens==1 else 'P2')
        for variant in variants:
            graph=None
            try:
                ops=pp_ops(state,tokens,variant) if variant[0] in ('R','P','W') else None
                graph,(out,trace)=pp_graph(lambda:pp_layer(m,data,state,plan,variant,ops))
                worst=0.
                for bank,ref in zip(banks,refs):
                    pp_refresh(data,bank);graph.replay();worst=max(worst,pp_gate(out,trace,ref,weights,state))
                pair=pp_pair(base,graph,flush,6,data,banks)
                early_late=None
                if ops is not None:
                    late=pp_ops(state,tokens,variant,False)
                    lg,_=pp_graph(lambda:pp_layer(m,data,state,plan,variant,late))
                    early_late=pp_pair(lg,graph,flush,6,data,banks)['ratio'];lg.reset()
                trials.append(dict(variant=variant,fp32=worst,early_late=early_late,**pair))
            except Exception as exc:
                rejects.append(variant+':'+type(exc).__name__+':'+str(exc).splitlines()[0][:85])
            finally:
                if graph is not None:graph.reset()
        if not trials:raise RuntimeError('all full-layer variants failed: '+str(rejects))
        # Final pairs and single-stage ablations for the best early native path
        # (or hint path). A stage is promoted only from its whole-layer evidence.
        best=min(trials,key=lambda r:r['ratio']);variant=best['variant']
        ops=pp_ops(state,tokens,variant) if variant[0] in ('R','P','W') else None
        graph,(out,trace)=pp_graph(lambda:pp_layer(m,data,state,plan,variant,ops))
        final=pp_pair(base,graph,flush,24,data,banks)
        for bank,ref in zip(banks,refs):
            pp_refresh(data,bank);graph.replay();pp_gate(out,trace,ref,weights,state)
        graph.reset();ablations={};stage_trials={}
        native=[r for r in trials if r['variant'][0] in ('R','P','W')]
        ablate=list(dict.fromkeys([variant]+([min(native,key=lambda r:r['ratio'])['variant']] if native else [])))
        for name in ('qkv_gate','o','pool_router','unpooler_slice' if tokens<=8 else 'unpooler'):
            stage_trials[name]=[]
            for choice in ablate:
                stage_ops=pp_ops(state,tokens,choice) if choice[0] in ('R','P','W') else None
                ag,(av,at)=pp_graph(lambda:pp_layer(m,data,state,plan,choice,stage_ops,only=name))
                try:
                    for bank,ref in zip(banks,refs):pp_refresh(data,bank);ag.replay();pp_gate(av,at,ref,weights,state)
                    stage_trials[name].append(dict(variant=choice,**pp_pair(base,ag,flush,12,data,banks)))
                finally:ag.reset()
            ablations[name]=min(stage_trials[name],key=lambda r:r['ratio'])
        PP_ROWS.append(dict(kind='layer',key=key,trials=trials,rejections=rejects,selected=variant,
            final=final,ablations=ablations,stage_trials=stage_trials,noise=noise))
        grid=' '.join(f'{r["variant"]}:{r["ratio"]:.3f}'+(f'/{r["early_late"]:.3f}' if r['early_late'] is not None else '') for r in trials)
        lines.append(f'L {case} {final["a"]:.2f}/{final["b"]:.2f} '+
            f'{variant}={final["ratio"]:.4f} s='+','.join(f'{x:.3f}' for x in final['seed_ratios'])+
            f' aa={noise["ratio"]:.3f} '+grid)
        lines.append('AB '+str(case)+' '+','.join(f'{PP_NAMES[n]}:{r["variant"]}/{r["ratio"]:.4f}' for n,r in ablations.items()))
        if rejects:lines.append('PDL_REJ c='+str(case)+' '+';'.join(rejects))
        base.reset();data=fresh=refs=banks=trace=out=bt=av=at=None;gc.collect()


def lab(deadline,weights,host_state=None):
    global PP_ROWS,PP_MODULE
    PP_ROWS=[];started=time.monotonic();lines=[
        'META best7='+_pphash.sha256(PP_BEST7_SOURCE.encode()).hexdigest()[:16]+
        '; cold128MiB every AB/BA;24 pairs/3 fresh streams; raw us; FP32/router1e-4 BF16.01 layer.003 exact IDs; immutable W prologue; L cases=d1,d16,d32,v1 x sliding/full; AB=whole-layer per-role, Q/O/P/U',
        ''  # Tensor provenance stays in the wrapper's protected second line.
    ]
    old_tf=torch.backends.cuda.matmul.allow_tf32;old_bf=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    try:
        pp_library();m=pp_module();PP_MODULE=m;m._BEST5_DEADLINE=deadline
        weights,synthetic,provided=pp_weights(m,weights)
        lines[1]='META seed='+str(PP_WEIGHT_SEED)+' synthetic='+','.join(synthetic)+'; host='+','.join(provided)+\
            '; inputs=synthetic attn_input:BF16,stream:FP32,k/v_cache:E4M3,metadata; seeds='+','.join(map(str,PP_SEEDS))+\
            '; routes=frozen baseline IDs/weights/localIDs per stream (route kernel timed); first32 experts; c6 Q=B for A0/A4/R0/P0/P2:local layer>.003; S=all; H=hint A=all R=row P=packed W=512thread; /early:late'
        state=pp_state(m,weights);flush=torch.zeros(128<<20,device=weights['o'].device,dtype=torch.uint8)
        # Whole-layer coverage is primary; the dense bandwidth diagnosis follows
        # in the same prepare and uses the exact same weight bank and protocol.
        pp_full(m,state,weights,flush,deadline,lines)
        pp_standalone(m,state,weights,flush,deadline,lines)
        lines.append(f'PDL_END layers=8 stages=16 elapsed={time.monotonic()-started:.2f} left={deadline-time.monotonic():.2f}')
        return lines
    except Exception as exc:
        import traceback
        lines.append('PDL_ERROR '+type(exc).__name__+':'+str(exc)[:250]+' '+traceback.format_exc().splitlines()[-3])
        lines.append(f'PDL_INCOMPLETE rows={len(PP_ROWS)} elapsed={time.monotonic()-started:.2f}')
        return lines
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_tf;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old_bf
