GP_CONFIRM_RESULTS=[]


def gp_assignments(source):
    return {t.id:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
        if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}


def gp_functions(source):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}


def gp_baseline():
    name='<gp-s13-baseline>';linecache.cache[name]=(len(GP_BASE),None,GP_BASE.splitlines(True),name)
    ns=dict(__name__='gp_s13_baseline',__file__=name);exec(compile(GP_BASE,name,'exec'),ns)
    host=kernel_fn.__globals__;old,new=gp_assignments(GP_BASE),gp_assignments(GP_CANDIDATE)
    for key,value in old.items():
        if new.get(key)==value and key in host:ns[key]=host[key]
    old,new=gp_functions(GP_BASE),gp_functions(GP_CANDIDATE)
    for key,value in old.items():
        if key not in ('kernel_fn','_best7_run','prepare','_best7_boundary','_best7_route') and new.get(key)==value and key in host:
            ns[key]=host[key]
    assert ns['_GEMV_CHOICES']==host['_GEMV_CHOICES']
    return ns


def lab(deadline,weights,state):
    global GP_CONFIRM_RESULTS
    GP_CONFIRM_RESULTS=[];ns=gp_baseline();base=ns['kernel_fn'];candidate=kernel_fn
    seeds=tuple(942193+104729*i for i in range(16))
    lines=['META held-out frozen conditional S13 glue confirmation; B/A (<1 faster); decode1 only b1+route+b2 preload',
        'META 16 new seeds, 32 rotated cold READ rounds at d1,8 elsewhere; A2=A:A; host is exact frozen candidate 08e82efd']
    generator=ns['_ap_generate_input']
    if torch.cuda.get_device_capability()!=(9,0):
        # Clone only the public generator so its scheduler hook is portable.
        name='<gp-confirm-generator>';src=ast.get_source_segment(GP_BASE,next(n for n in ast.parse(GP_BASE).body
            if isinstance(n,ast.FunctionDef) and n.name=='_ap_generate_input'))
        ns['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
        linecache.cache[name]=(len(src),None,src.splitlines(True),name);exec(compile(src,name,'exec'),ns)
        generator=ns['_ap_generate_input']
    results={}
    for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        for w in (2048,0):
            label=('d' if q==1 else 'v')+str(b)+('S' if w else 'F')
            if time.monotonic()>deadline-8:lines.append('INCOMPLETE '+label);continue
            mk=lambda seed:generator(batch=b,ctx=80000,layer='sliding' if w else 'full',q_len=q,
                spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device=weights['o'].device)
            data=mk(seeds[0]);graphs={};gates={}
            for n,fn in (('A',base),('A2',base),('B',candidate)):
                try:graphs[n]=ab_graph(lambda fn=fn:fn(data,state))
                except Exception as exc:gates[n]=type(exc).__name__+':'+str(exc).replace('\n',' ')[:140]
            if 'A' not in graphs:lines.append('FAIL '+label+' '+repr(gates));continue
            samples=[];rounds=32 if (b,q)==(1,1) else 8
            for seed in seeds:
                if time.monotonic()>deadline-5:lines.append('INCOMPLETE '+label);break
                if seed!=seeds[0]:ab_copy(data,mk(seed))
                for graph,_ in graphs.values():graph.replay()
                torch.cuda.synchronize()
                for n,(_,out) in graphs.items():
                    if n!='A' and (why:=ab_gate(out,graphs['A'][1])):gates.setdefault(n,why)
                values=ab_time(graphs,rounds);samples.append({n:statistics.mean(v) for n,v in values.items()})
            summary=ab_summary(samples);results[(b,q,w)]=summary
            lines.append('PAIR '+label+' '+' '.join(f'{n}:{r:.6f}+/-{se:.6f}:{d:+.3f}' for n,(r,se,d,_,_) in summary.items())+
                (' GATE='+repr(gates) if gates else ''))
            GP_CONFIRM_RESULTS.append(dict(case=label,seeds=len(samples),rounds=rounds,gates=gates,
                paired=summary,samples=samples))
            del graphs,data;gc.collect();torch.cuda.empty_cache()
    loads={}
    for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        keys=[(b,q,w) for w in (2048,0)]
        if not all(k in results for k in keys):continue
        load=('d' if q==1 else 'v')+str(b);loads[load]={}
        for n in ('A2','B'):
            den=sum(wt*results[k][n][3] for k,wt in zip(keys,(46,16)))
            loads[load][n]=sum(wt*results[k][n][3]*results[k][n][0] for k,wt in zip(keys,(46,16)))/den
        lines.append('LOAD '+load+' '+' '.join(f'{n}={r:.7f}' for n,r in loads[load].items()))
    if len(loads)==6:
        lines.append('SCORE B/A speedup='+str(1/(math.prod(v['B'] for v in loads.values())**(1/6))))
    lines.append('END shapes='+str(len(results))+'/12')
    # Keep per-seed spread, but round far below the measurement uncertainty.
    records=json.loads(json.dumps(GP_CONFIRM_RESULTS),parse_float=lambda x:round(float(x),7))
    lines.append('GP_CONFIRM_XZ85 '+base64.b85encode(lzma.compress(json.dumps(dict(loads=loads,cells=records),separators=(',',':')).encode(),preset=9)).decode())
    return lines
