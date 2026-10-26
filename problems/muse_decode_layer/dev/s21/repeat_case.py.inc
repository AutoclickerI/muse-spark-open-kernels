def s21_legacy_time(graphs,rounds):
    """Replay the archived rotation once; archive every round, never select a result."""
    names=list(graphs);times={n:[] for n in names};enqueue={n:[] for n in names};telemetry=[s21_telemetry()];orders=[]
    for r in range(rounds):
        order=names[r%len(names):]+names[:r%len(names)]
        if r%2:order=order[::-1]
        orders.append(order);events=[]
        for name in order:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            lab_cold_l2();host_start=time.perf_counter_ns();start.record();graphs[name][0].replay();end.record()
            enqueue[name].append((time.perf_counter_ns()-host_start)/1000.);events.append((name,start,end))
        torch.cuda.synchronize()
        for name,start,end in events:times[name].append(start.elapsed_time(end)*1000.)
        telemetry.append(s21_telemetry())
    return dict(round_us=times,host_enqueue_us=enqueue,orders=orders,telemetry=telemetry)

def s21_repeat_original_case(checker,base,deadline):
    """Exact old weights and seed, after primary confirmation; never feed G0."""
    global S21_TRACE_CONTEXT,S21_ROUND_COUNTS,S21_TOP_TRACES,S21_FLAGGED,S21_MEASURED
    import contextlib,io
    saved=(S21_TRACE_CONTEXT,S21_ROUND_COUNTS,S21_TOP_TRACES,S21_FLAGGED,S21_MEASURED)
    S21_TRACE_CONTEXT={};S21_ROUND_COUNTS={};S21_TOP_TRACES=[];S21_FLAGGED=0;S21_MEASURED=0
    started=time.monotonic();capture=io.StringIO();out=dict(complete=False,scope='diagnostic-only',seed=1730886002,weight_seed=1729315001,router_seed=1729315003,
        shape='v16S',protocols=['archived-rotation-8','palindromic-cohort-16'],primary_seeds_unchanged=True)
    weights=states=graphs=data=host=ns=None
    try:
        assert 1730886002 not in AB_SEEDS
        weights=checker['generate_weights'](1729315001,device='cuda',router_seed=1729315003)
        source=s16_source(S21_REPEAT_ARMS['S20'])
        path='<s21-original-S20-'+hashlib.sha256(source.encode()).hexdigest()[:12]+'>'
        linecache.cache[path]=(len(source+S19_HOST_HOOK),None,(source+S19_HOST_HOOK).splitlines(True),path)
        host=types.ModuleType('s21_original_S20');host.__file__=path
        exec(compile(source+S19_HOST_HOOK,path,'exec'),host.__dict__)
        if globals().get('S16_LOCAL_INSTALL'):S16_LOCAL_INSTALL(host)
        old=host.__dict__;ns={'A':old,'A2':old};out['sources']={'A':hashlib.sha256(source.encode()).hexdigest(),'A2':hashlib.sha256(source.encode()).hexdigest()}
        with contextlib.redirect_stdout(capture):
            prep_start=time.monotonic();prepared=old['_candidate_prepare'](weights);out['prepare_s']=time.monotonic()-prep_start
            states={'A':prepared,'A2':prepared}
            original=old['_S16_LAB_BASE_STATE']
            for name in ('INC','no_H2','no_B1'):
                text=s16_source(S21_REPEAT_ARMS[name]);out['sources'][name]=hashlib.sha256(text.encode()).hexdigest()
                arm,_=s16_namespace('repeat_'+name,text,old);ns[name]=arm
                arm['_vq8_module']();arm['nf_module']();arm['cd_module']();arm['smallm_module']();arm['_glue_cl_ext']();arm['partial_port_ext'](False)
                st=s16_copy_state(original);st['vq8_rows']=dict(arm['_VQ8_ROWS']);st['partial_port_rows']=arm['_PP_ROWS'];st['cluster_rows']=arm.get('_CLUSTER_ROWS',())
                if '_r2_prepare' in arm:st['r2_rows']=arm['_r2_prepare'](weights['qkv_gate'].device)
                states[name]=arm['_s16_prepare_delta'](weights,st)
        if time.monotonic()>deadline:raise RuntimeError('diagnostic setup exceeded lab deadline')
        data=old['_ap_generate_input'](batch=16,q_len=8,ctx=80000,layer='sliding',spread='agentic',dist='soft',page=1,seed=1730886002,device=weights['qkv_gate'].device)
        graphs={name:ab_graph(lambda n=n,st=states[name]:n['kernel_fn'](data,st)) for name,n in ns.items()}
        checks={}
        for name,(graph,output) in graphs.items():
            graph.replay();torch.cuda.synchronize();ok,msg=checker['check_implementation'](data,checker['collect'](data,output),weights)
            checks[name]=dict(passed=ok,message=msg)
        out['TASK']=checks
        out['legacy']=s21_legacy_time(graphs,8)
        S21_TRACE_CONTEXT.update(label='repeat_original_v16S',seed=1730886002,seed_index=15)
        values=ab_time(graphs,S21_PROTOCOL['rounds']);out['symmetric']=dict(round_us=values,trace=S21_TOP_TRACES[0])
        for key in ('legacy','symmetric'):
            values=out[key]['round_us'];avg={n:statistics.mean(v) for n,v in values.items()};out[key]['mean_us']=avg;out[key]['ratios_vs_S20']={n:v/avg['A'] for n,v in avg.items()}
        out['complete']=all(x['passed'] for x in checks.values())
    except Exception as exc:out['error']=type(exc).__name__+':'+str(exc)[:300]
    finally:
        out['elapsed_s']=time.monotonic()-started
        S21_TRACE_CONTEXT,S21_ROUND_COUNTS,S21_TOP_TRACES,S21_FLAGGED,S21_MEASURED=saved
        weights=states=graphs=data=host=ns=None;gc.collect();torch.cuda.empty_cache()
    return out
