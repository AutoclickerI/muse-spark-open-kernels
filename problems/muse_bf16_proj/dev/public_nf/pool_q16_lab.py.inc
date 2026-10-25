def lab(deadline,weights):
    import linecache
    start=time.monotonic();weights,kind=cl_weights(weights)
    def module(label,source):
        file='<nf_'+label+'>';linecache.cache[file]=(len(source),None,source.splitlines(True),file)
        ns={'__name__':label,'__file__':file};exec(compile(source,file,'exec'),ns);return ns
    old=module('S14_pool_control',S13_SOURCE)
    hi=weights['router'].bfloat16();w=torch.cat((weights['pooler'],hi,(weights['router']-hi.float()).bfloat16()))
    state=dict(pool_router=w,qkv_gate=weights['qkv_gate'],o=weights['o'],unpooler=weights['unpooler'],unpooler_slice=weights['unpooler'][:1024])
    old['_seed_v9_gemv_choices'](state);old['_gate_split_prepare'](state,weights['o'].device)
    old['cd_module'](((1,1,2,3),));old['_cd_prepare'](state)
    state['pre_expert_w']=torch.ones(4096,device='cuda');state['router_bias']=torch.zeros(256,device='cuda')
    data={'local_start':torch.zeros((),device='cuda',dtype=torch.int32)}
    states={};keys={(t,'pool_router') for t in (1,8,16,32)}
    for arm in ('original','rows16','rows32','rows32cp1'):
        source=PORT_SOURCE.replace("('sm',insts,2)","('sm',insts,1)") if arm=='rows32cp1' else PORT_SOURCE
        p=module('pool_'+arm,source)
        if arm!='original':
            wr=1 if arm=='rows16' else 2
            for t in (1,8):p['NF_SMALL'][t,'pool_router']=('v18',(wr,8,16,1),1,1,0,wr*256,(6,))
            for t in (16,32):p['NF_SMALL'][t,'pool_router']=('cp',(wr,4,32,1),t//8,2,0,wr*128,())
        states[arm]=(p,p['nf_prepare'](weights,keys))
    # Corrected qkv16 uses its own replicated-base packer, not qkv32's.
    q=module('q16_fix',PORT_SOURCE);qs=q['nf_prepare'](weights,{(16,'qkv_gate'),(16,'qkv3')})
    native=torch.cuda.get_device_capability()==(9,0)
    empty=statistics.median(_cl_times([_cl_graph(lambda:None)],8)[0]) if native else 0.
    lines=[f'META NF fused N4608 occupancy geometry tuning; {kind}; 8 seeds x12 cold ABBA; setup={time.monotonic()-start:.1f}s',
        'META original vs WR1/WK8(small),WR1/WK4(16/32); WR2 equivalents; CP1 vs CP2; corrected qkv16 packer']
    def query(p,st,t,name):
        if not native:return 'standin'
        v=st['plans'][t,name];x=torch.zeros((t,st['roles'][name].shape[1]),device='cuda',dtype=torch.bfloat16)
        out=torch.empty((t,v['n']),device='cuda',dtype=torch.float32)
        args=(v['idx'],v['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),v['esc'].data_ptr(),t,v['n'],x.shape[1],*v['cfg'],v['t01'],v['u01'],torch.cuda.current_stream().cuda_stream)
        return f'attr={v["fn"](*args,2)} active={v["fn"](*args,1)} cfg={v["cfg"]}'
    for t in (1,8,16,32):
        x=torch.randn((t,8192),device='cuda',dtype=torch.bfloat16)
        def baseline():return old['_project_variant'](x,w,'pool_router',state,'custom',torch.float32)
        def consume(v):return old['_glue_cl_route'](v,state,data,('cl',8,256,0)) if native else old['route_and_norm'](v,state,data)
        arms=[('S14',lambda:consume(baseline())),('AA',lambda:consume(baseline()))]
        for arm,(p,st) in states.items():
            arms.append((arm,lambda p=p,st=st:consume(p['nf_project'](x,'pool_router',st))))
            lines.append(f'INFO t={t} {arm} '+query(p,st,t,'pool_router'))
        graphs=[_cl_graph(fn) for _,fn in arms];times={n:[] for n,_ in arms};errs={n:0. for n in states}
        for seed in (61791,14357,92003,17453,51613,72577,35111,81901):
            g=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=g).bfloat16());x[-1]=0.;x[-1,(seed*131+t)%8192]=16.
            ref=x.double()@w.double().T
            for arm,(p,st) in states.items():
                out=p['nf_project'](x,'pool_router',st)[0].double().sum(0)
                err=float((out-ref).norm()/ref.norm());probe=float((out[-1]-ref[-1]).norm()/ref[-1].norm())
                assert err<3e-5 and probe<1e-5,(t,arm,err,probe);errs[arm]=max(errs[arm],err)
            vals=_cl_times(graphs,12)
            for j,(arm,_) in enumerate(arms):times[arm].append(statistics.median(vals[j])-empty)
        aa=max(abs(b/a-1) for a,b in zip(times['S14'],times['AA']))
        for arm in states:
            ratios=[b/a for a,b in zip(times['S14'],times[arm])]
            lines.append(f'POOL t={t} {arm} A={statistics.mean(times["S14"]):.2f} B={statistics.mean(times[arm]):.2f} r={statistics.mean(ratios):.5f}[{min(ratios):.5f},{max(ratios):.5f}] aa={aa:.5f} e={errs[arm]:.1e}')
    for name in ('qkv_gate','qkv3'):
        t=16;ww=qs['roles'][name];x=torch.randn((t,8192),device='cuda',dtype=torch.bfloat16)
        def baseline():return old['_project_variant'](x,ww,name,state,'custom',torch.bfloat16)
        consume=lambda pair:old['_materialize'](pair)
        graphs=[_cl_graph(lambda:consume(baseline())),_cl_graph(lambda:consume(baseline())),_cl_graph(lambda:consume(q['nf_project'](x,name,qs)))]
        a=[];b=[];aa=[];err=0.;probe=0.
        lines.append(f'INFO t=16 {name} '+query(q,qs,16,name))
        for seed in (61791,14357,92003,17453,51613,72577,35111,81901):
            g=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=g).bfloat16());x[-1]=0.;x[-1,(seed*131+t)%8192]=16.
            ref=x.double()@ww.double().T;out=q['nf_project'](x,name,qs)[0].double().sum(0)
            err=max(err,float((out-ref).norm()/ref.norm()));probe=max(probe,float((out[-1]-ref[-1]).norm()/ref[-1].norm()))
            assert err<3e-5 and probe<1e-5,(name,err,probe)
            v=_cl_times(graphs,12);a.append(statistics.median(v[0])-empty);b.append(statistics.median(v[2])-empty);aa.append((statistics.median(v[1])-empty)/a[-1])
        rr=[v/u for u,v in zip(a,b)]
        lines.append(f'Q16 {name} A={statistics.mean(a):.2f} B={statistics.mean(b):.2f} r={statistics.mean(rr):.5f}[{min(rr):.5f},{max(rr):.5f}] aa={max(abs(v-1) for v in aa):.5f} e={err:.1e} p={probe:.1e}')
    return lines+[f'END elapsed={time.monotonic()-start:.1f}s']
