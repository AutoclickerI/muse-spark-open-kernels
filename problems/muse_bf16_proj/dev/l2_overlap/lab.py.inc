import ast as _p7ast
import gc as _p7gc
import linecache as _p7linecache
import statistics as _p7stats
import time as _p7time

P7_LOCAL_INSTALL=None
P7_LOCAL_ROWS=[]


class P7Space:
    def __init__(self,ns):object.__setattr__(self,'ns',ns)
    def __getattr__(self,k):
        try:return self.ns[k]
        except KeyError:raise AttributeError(k) from None
    def __setattr__(self,k,v):self.ns[k]=v


def p7_module():
    filename='<p7 exact best6>'
    _p7linecache.cache[filename]=(len(P7_BEST6_SOURCE),None,P7_BEST6_SOURCE.splitlines(True),filename)
    ns={'__name__':__name__,'__builtins__':__builtins__}
    exec(compile(P7_BEST6_SOURCE,filename,'exec'),ns)
    m=P7Space(ns)
    if P7_LOCAL_INSTALL is not None:P7_LOCAL_INSTALL(m)
    return m


def p7_state(m,weights,deadline):
    # No best6 tuner is repeated; freeze the actual best6 receipt's choices.
    # Its raw E4M3 V9 bank is built from original packed weights, not the host's
    # differently packed/encoded bank. The host's returned state is untouched.
    from concurrent.futures import ThreadPoolExecutor
    m._BEST5_DEADLINE=deadline-3
    with ThreadPoolExecutor(max_workers=4) as pool:
        nt2=pool.submit(m._best5_nt2_module)
        lt=pool.submit(m._lt_module)
        attn=pool.submit(m._attn_port_ext)
        layer=pool.submit(m._layer_attn_ext)
        state=m._prepare_weights(weights)
        state['moe_nt2_mod']=nt2.result();state['lt_mod']=lt.result()
        attn.result();layer.result()
    m._seed_v9_gemv_choices(state)
    state.update(proj_plan={},attention_plan={},moe6_plan={1:'v9nt1',16:'v9nt1',32:'v9nt2'},
                 moe_nt_plan={1:1,16:1,32:2},pdl_plan={},prefix_plan={})
    for b,q,w,attn,projections in P7_RECEIPT_CHOICES:
        state['proj_plan'].update({(b*q,n):v for n,v in projections.items()})
        state['attention_plan'][b,q,bool(w)]=attn
    for tokens,phase,setting in P7_RECEIPT_GLUE:
        if phase=='route':m._GLUE_ROUTE[tokens]=setting
        else:m._GLUE_BOUNDARY[tokens,phase=='b2']=setting
    state['lt_workspace']=torch.empty(32<<20,device=weights['o'].device,dtype=torch.uint8)
    state['lt_algos']={}
    for (tokens,name),choice in state['proj_plan'].items():
        if isinstance(choice,tuple) and choice[0]=='lt':
            x=torch.zeros((tokens,state[name].shape[1]),device=state[name].device,dtype=torch.bfloat16)
            y=torch.empty((tokens,state[name].shape[0]),device=x.device,dtype=torch.bfloat16)
            codes=state['lt_mod'].lt_options(state[name],x,y)
            if codes.shape[0]<=choice[1]:raise RuntimeError('receipt Lt algorithm ordinal unavailable')
            state['lt_algos'][name,tokens]=codes
    # The receipt tail lost M1 boundary geometry. Run best6's own three-draw
    # geometry selector at M1 rather than claim an invented historical setting.
    saved=m._TUNE_TOKENS
    try:
        m._TUNE_TOKENS=(1,)
        m._tune_glue(state,torch.empty(128<<20,device=weights['o'].device,dtype=torch.uint8),weights['o'].device)
    finally:m._TUNE_TOKENS=saved
    return state


def p7_layer(m,data,state,prefetch=None):
    trace={};t=data['config']['tokens']
    qkv=m.project(data['attn_input'],state['qkv_gate'],'qkv_gate',state);trace['qkv']=(data['attn_input'],qkv)
    if prefetch is not None:prefetch.begin('front')
    attn=m.stage_attention(qkv,data,state);trace['attention']=(qkv,attn)
    if prefetch is not None:
        prefetch.join('front');prefetch.begin('pool')
    a=m.project(attn,state['o'],'o',state);trace['o']=(attn,a)
    h,s1=m.boundary(a,data['stream'],state['alpha1'],state['beta1'],state['pre_ffn_w']);trace['b1']=(a,data['stream'],h,s1)
    if prefetch is not None:prefetch.join('pool')
    pr=m.project(h,state['pool_router'],'pool_router',state,out_dtype=torch.float32);trace['pool_router']=(h,pr)
    x,ids,wts,lids=m.route_and_norm(pr,state,data);trace['route']=(pr,x,ids,wts,lids)
    if prefetch is not None:prefetch.begin('tail')
    moe=m._best6_experts(pr,x,wts,lids,state);trace['moe']=(x,ids,wts,lids,pr,moe)
    if prefetch is not None:prefetch.join('tail')
    name='unpooler_slice' if t<=m.SLICED_MAX_TOKENS else 'unpooler'
    f=m.project(moe,state[name],name,state);trace[name]=(moe,f)
    hidden,s2=m.boundary(f,s1,state['alpha2'],state['beta2'],state['next_pre_attn_w'],postw=state['post_ffn_w'])
    trace['b2']=(f,s1,hidden,s2)
    return (hidden,s2,ids),trace


def p7_value(value):
    p,s=value
    return p.float().sum(0) if s>1 else p[0].float()


def p7_relative(a,b):
    a,b=a.float(),b.float();d=(a-b).norm(dim=-1);n=b.norm(dim=-1)
    return float(d.norm()/n.norm().clamp_min(1.e-12)),float((d/n.clamp_min(.1*n.mean()).clamp_min(1.e-12)).max())


def p7_close(a,b,tol,label):
    if a.dtype in (torch.int32,torch.int64):
        if not torch.equal(a,b):raise RuntimeError(label+' expert IDs differ')
        return 0.
    error,row=p7_relative(a,b)
    if not bool(torch.isfinite(a).all()) or error>tol or row>tol*5:
        raise RuntimeError(f'{label} FP32 gate {error:.3e}/{row:.3e} bound={tol}')
    return error


def p7_oracle(m,trace,data,state,weights):
    want={};worst=0.
    for name in ('qkv','o','pool_router','unpooler_slice' if data['config']['tokens']<=8 else 'unpooler'):
        inp,partial=trace[name];bank=state['qkv_gate' if name=='qkv' else name]
        expected=inp.float()@bank.float().T
        # BF16 cuBLAS output may round; custom split-K is FP32 through its consumer.
        tol=.01 if partial[0].dtype==torch.bfloat16 else .0001
        worst=max(worst,p7_close(p7_value(partial),expected,tol,name))
        want[name]=(expected,tol)
    inp=trace['pool_router'][0];raw=p7_value(trace['pool_router'][1])
    logits=raw[:,4096:4352]+raw[:,4352:4608]
    expected=inp.float()@weights['router'].float().T
    p7_close(logits,expected,.0001,'router hi+lo')
    want['logits']=expected
    # Check route/norm from identical FP32 producer values. Exact A/B route IDs
    # are checked separately; the oracle checks each selected weight by ID,
    # avoiding a floating-point near-tie being mistaken for an address race.
    _,x,ids,wts,lids=trace['route'];scores=(logits*m.ROUTER_MULT).sigmoid()
    picked=scores.gather(1,ids.long());tw=picked/(picked.sum(-1,keepdim=True)+1.e-15)
    p7_close(wts,tw,.0001,'route weights')
    biased=scores+state['router_bias'];threshold=biased.topk(8,-1).values[:,-1:]
    if bool((biased.gather(1,ids.long())<threshold-1.e-5).any()):raise RuntimeError('route outside FP32 top-eight noise bound')
    expected_x=(m._best5_rms(raw[:,:4096])*state['pre_expert_w']).bfloat16()
    p7_close(x,expected_x,.01,'pre-expert norm')
    expert=dict(hidden_states=x,topk_ids=lids,topk_weights=wts,
                config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0))
    oracle=m._best5_fp32(expert,weights)
    if oracle is None:raise RuntimeError('deadline before FP32 expert oracle')
    p7_close(trace['moe'][-1],oracle,.01,'V9 expert')
    want['moe']=oracle
    for phase,post in (('b1',False),('b2',True)):
        partial,stream,hidden,sout=trace[phase];br=p7_value(partial)
        br=torch.nn.functional.pad(br,(0,8192-br.shape[1]))
        if post:br=br*state['post_ffn_w']
        br=br*torch.rsqrt(br.square().mean(-1,keepdim=True)+1.e-8)
        ss=state['alpha2' if post else 'alpha1']*stream+state['beta2' if post else 'beta1']*br
        hh=(m._best5_rms(ss)*state['next_pre_attn_w' if post else 'pre_ffn_w']).bfloat16()
        p7_close(sout,ss,.0001,phase+' residual');p7_close(hidden,hh,.01,phase+' norm')
    qkv,attn=trace['attention']
    m._attention_store_agrees(qkv,data)
    ref=m._attention_reference(qkv,data);rounded=m._attention_reference(qkv,data,True)
    ok,error,bound=m._attention_agrees(attn,ref,rounded)
    if not ok:raise RuntimeError(f'attention FP32 {error}/{bound}')
    return want,worst


def p7_gate(trace,want,out,reference):
    for name,(expected,tol) in ((n,v) for n,v in want.items() if n not in ('moe','logits')):
        p7_close(p7_value(trace[name][1]),expected,tol,name)
    raw=p7_value(trace['pool_router'][1]);p7_close(raw[:,4096:4352]+raw[:,4352:],want['logits'],.0001,'router')
    p7_close(trace['moe'][-1],want['moe'],.01,'experts')
    # This is a mixed-BF16 whole-layer bound, not the router's FP32 bound.
    # cuBLAS eager/graph reduction and V9 summation can move a BF16 rounding
    # boundary even in a no-prefetch twin. Raw/router gates above stay 1e-4.
    for i,(got,ref) in enumerate(zip(out,reference)):p7_close(got,ref,.003,'full layer '+str(i))


def p7_graph(fn):
    for _ in range(2):fn()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    g.replay();torch.cuda.synchronize()
    return g,out


def p7_refresh(data,bank):
    for k in ('attn_input','stream','local_start'):data[k].copy_(bank[k])


def p7_pair(a,b,data,banks,flush,rounds=24):
    vals=[[],[]];start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for rep in range(rounds):
        p7_refresh(data,banks[rep%3])
        for side in ((0,1) if rep%2==0 else (1,0)):
            flush.zero_();start.record();(a if side==0 else b).replay();end.record();end.synchronize()
            vals[side].append(start.elapsed_time(end)*1000.)
    seeds=[]
    for bank in range(3):
        ix=range(bank,rounds,3)
        ratios=[vals[1][i]/vals[0][i] for i in ix]
        seeds.append(_p7stats.mean(ratios))
    return dict(a=_p7stats.mean(vals[0]),b=_p7stats.mean(vals[1]),ratio=_p7stats.mean(seeds),seeds=seeds,samples=vals,rounds=rounds)


def p7_profile(graph,flush):
    # Activity timestamps observe the unchanged graph. No events are inserted
    # between dependent stages, which would serialize the baseline's PDL.
    flush.zero_();torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay();torch.cuda.synchronize()
    events=sorted((e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA),key=lambda e:e.time_range.start)
    sums={k:0. for k in ('qkv','attention','o','b1','pool','route','moe','unpool','b2','prefetch','other')}
    phase='qkv';boundaries=0;spans={};names=[]
    for e in events:
        name=e.name;names.append(name)
        if 'p7_l2' in name:label='prefetch'
        elif '_boundary_fast' in name:
            boundaries+=1;phase='b1' if boundaries==1 else 'b2';label=phase
        elif '_route_fast' in name:phase='route';label=phase
        elif 'moe_fused' in name:phase='moe';label=phase
        elif 'attn_' in name or 'Attn' in name:phase='attention';label=phase
        else:
            if phase=='attention':phase='o'
            elif phase=='b1':phase='pool'
            elif phase=='moe':phase='unpool'
            label=phase
        sums[label]+=e.device_time_total
        spans.setdefault(label,[]).append((e.time_range.start,e.time_range.end))
    if boundaries!=2 or not sums['moe'] or not sums['attention']:raise RuntimeError('activity stage classification incomplete: '+str(names[:4]))
    overlaps={}
    for stage in ('attention','moe'):
        overlaps[stage]=sum(max(0.,min(b,d)-max(a,c)) for a,b in spans.get('prefetch',[]) for c,d in spans.get(stage,[]))
    return dict(us=sums,prefetch_overlap_us=overlaps,kernels=names)


def lab(deadline,weights,host_state=None):
    global P7_LOCAL_ROWS
    started=_p7time.monotonic();rows=[];P7_LOCAL_ROWS=rows
    old_tf32=torch.backends.cuda.matmul.allow_tf32;old_bf=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    lines=['P7_META best6=4516cb97/1b80ffc0 frozen receipt dispatch; M1/16/32 q1 sliding/full; cold128MiB each A/B replay; final24 AB/BA rounds across3 refreshed public streams; all raw us',
           'P7_META normal L2 hints/loads, no set-aside; <=40MiB front, <=24MiB tail; P7_PICK = best recipe id A_us/B_us B:A ratio 3 seed ratios A:A delta_us; ST=Q,A,O,B1,P,R,E,U,B2,PF activity us']
    try:
        p7_l2_library();m=p7_module();state=p7_state(m,weights,deadline)
        dev=weights['o'].device;flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
        lines.append(f'P7_SETUP seconds={_p7time.monotonic()-started:.1f} left={deadline-_p7time.monotonic():.1f}')
        for case,(batch,window) in enumerate((b,w) for b in (1,16,32) for w in (2048,0)):
            if _p7time.monotonic()>deadline-5:lines.append(f'P7_INCOMPLETE case={case}');break
            data=m._ap_generate_input(batch=batch,ctx=80000,layer='sliding' if window else 'full',q_len=1,
                spread='agentic' if batch>1 else 'fixed',dist='soft',page=1,seed=172903,device=dev)
            banks=[];wants=[];refs=[];maxerr=0.
            for seed in (172903,180822,188741):
                fresh=m._ap_generate_input(batch=batch,ctx=257,layer='sliding' if window else 'full',q_len=1,
                    spread='agentic' if batch>1 else 'fixed',dist='soft',page=1,seed=seed,device=dev)
                bank={k:fresh[k].clone() for k in ('attn_input','stream','local_start')};p7_refresh(data,bank)
                output,trace=p7_layer(m,data,state)
                ids=output[2];start=32*int(torch.bincount((ids.long()//32).flatten(),minlength=8).argmax())
                bank['local_start'].fill_(start);p7_refresh(data,bank);output,trace=p7_layer(m,data,state)
                want,error=p7_oracle(m,trace,data,state,weights);maxerr=max(maxerr,error)
                banks.append(bank);wants.append(want);refs.append(tuple(v.clone() for v in output))
            base,base_out=p7_graph(lambda:m.kernel_fn(data,state))
            captured_refs=[]
            for bank,ref in zip(banks,refs):
                p7_refresh(data,bank);base.replay()
                for i,(got,expected) in enumerate(zip(base_out,ref)):
                    p7_close(got,expected,.003,'best6 replica '+str(i))
                captured_refs.append(tuple(v.clone() for v in base_out))
            # Compare captured A with captured B. Eager and graph cuBLAS calls
            # can choose different reduction paths before BF16 rounding.
            refs=captured_refs
            # Numerical A/A across two independently allocated graphs, before
            # any side stream exists. Retain this evidence if a gate fails.
            twin,twin_value=p7_graph(lambda:p7_layer(m,data,state))
            twin_error=[0.,0.]
            for bank,ref in zip(banks,refs):
                p7_refresh(data,bank);twin.replay()
                for i in (0,1):twin_error[i]=max(twin_error[i],p7_relative(twin_value[0][i],ref[i])[0])
            twin.reset()
            lines.append(f'P7_NUMERIC c={case} no_prefetch_twin_l2={twin_error[0]:.3e}/{twin_error[1]:.3e}')
            noise=p7_pair(base,base,data,banks,flush,12)
            choices=[];rejections=[]
            reserve=2.5*(6-case);coarse_end=_p7time.monotonic()+.55*max(0.,deadline-_p7time.monotonic()-reserve)/(6-case)
            for recipe in P7_PREFETCH_RECIPES:
                if _p7time.monotonic()>coarse_end:break
                graph=None
                try:
                    op=P7L2Prefetch(state,batch,recipe)
                    graph,(out,trace)=p7_graph(lambda:p7_layer(m,data,state,op))
                    for bank,want,ref in zip(banks,wants,refs):
                        p7_refresh(data,bank);graph.replay();p7_gate(trace,want,out,ref)
                    pair=p7_pair(base,graph,data,banks,flush,6)
                    choices.append((pair['ratio'],recipe))
                except Exception as exc:rejections.append(f'{recipe}:{type(exc).__name__}:{str(exc)[:70]}')
                finally:
                    if graph is not None:graph.reset()
            finals=[]
            for _,recipe in sorted(choices)[:2]:
                if _p7time.monotonic()>deadline-2:break
                op=P7L2Prefetch(state,batch,recipe);graph,(out,trace)=p7_graph(lambda:p7_layer(m,data,state,op))
                try:
                    pair=p7_pair(base,graph,data,banks,flush,24)
                    for bank,want,ref in zip(banks,wants,refs):
                        p7_refresh(data,bank);graph.replay();p7_gate(trace,want,out,ref)
                    finals.append((pair['ratio'],recipe,pair))
                    rows.append(dict(case=case,batch=batch,window=window,recipe=recipe,**pair))
                finally:graph.reset()
            if not finals:lines.append(f'P7_NO_PICK c={case} trials={len(choices)} rejects={rejections[:1]}');base.reset();continue
            ratio,recipe,pair=min(finals)
            lines.append(f'P7_PICK c={case} m={batch} w={window} r={recipe} us={pair["a"]:.3f}/{pair["b"]:.3f} ratio={ratio:.5f} seeds='+','.join(f'{x:.4f}' for x in pair['seeds'])+f' noise={noise["b"]-noise["a"]:.3f} n={len(choices)} rej={len(rejections)} fp32={maxerr:.2e}')
            # Profile only after the uninhibited whole-layer decision timing.
            if _p7time.monotonic()<deadline-2:
                op=P7L2Prefetch(state,batch,recipe);graph,(out,trace)=p7_graph(lambda:p7_layer(m,data,state,op))
                try:
                    p7_refresh(data,banks[0]);pa=p7_profile(base,flush);pb=p7_profile(graph,flush)
                    order=('qkv','attention','o','b1','pool','route','moe','unpool','b2','prefetch')
                    lines.append(f'P7_ST c={case} r={recipe} A='+','.join(f'{pa["us"][k]:.2f}' for k in order)+' B='+','.join(f'{pb["us"][k]:.2f}' for k in order)+f' overlapA/E={pb["prefetch_overlap_us"]["attention"]:.2f}/{pb["prefetch_overlap_us"]["moe"]:.2f}')
                    next(r for r in reversed(rows) if r['case']==case and r['recipe']==recipe)['profile']=dict(a=pa,b=pb)
                except Exception as exc:lines.append(f'P7_PROFILE_ERROR c={case} {type(exc).__name__}:{str(exc)[:100]}')
                finally:graph.reset()
            if rejections:lines.append(f'P7_REJECT c={case} '+rejections[0])
            base.reset();data=banks=wants=refs=trace=out=None;_p7gc.collect()
        lines.append(f'P7_END cases={len({r["case"] for r in rows})}/6 rows={len(rows)} elapsed={_p7time.monotonic()-started:.1f} left={deadline-_p7time.monotonic():.1f}')
        return lines
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_tf32;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old_bf
