"""S15 whole-layer leave-one-out. r=off/host; >1 favors S15. All controls are timed."""
import ast
import gc
import hashlib
import json
import linecache
import threading
import types

S15_VARIANTS = _S15_BUNDLE['variants']
S15_HOST_SOURCE = _S15_BUNDLE['host']


def s15_restore(spec):
    lines = S15_HOST_SOURCE.splitlines(True)
    for start,end,text in reversed(spec['edits']):lines[start:end]=[text]
    source=''.join(lines)
    assert hashlib.sha256(source.encode()).hexdigest()==spec['sha256']
    return source


def s15_assignments(source):
    return {target.id:ast.dump(node) for node in ast.parse(source).body
            if isinstance(node,ast.Assign) for target in node.targets if isinstance(target,ast.Name)}


def s15_share_jit(base,target):
    # Share identical device functions only if their complete JIT dependency closure matches.
    names={name for name,value in base.items() if isinstance(getattr(value,'src',None),str)
           and hasattr(value,'warmup') and name in target and getattr(target[name],'src',None)==value.src}
    deps={name:{node.id for node in ast.walk(ast.parse(base[name].src)) if isinstance(node,ast.Name)
                and hasattr(base.get(node.id),'warmup')} for name in names}
    while True:
        kept={name for name in names if deps[name]<=names}
        if kept==names:break
        names=kept
    for name in names:target[name]=base[name]
    assert all(target[name] is base[name] for name in names)
    return len(names)


def s15_namespace(name,source,base):
    filename='<s15-'+name+'-'+hashlib.sha256(source.encode()).hexdigest()[:12]+'>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    module=types.ModuleType('s15_'+name);module.__file__=filename
    exec(compile(source,filename,'exec'),module.__dict__)
    if globals().get('S15_LOCAL_FORCE'):S15_LOCAL_FORCE(module)
    ns=module.__dict__
    host_assign,target_assign=s15_assignments(S15_HOST_SOURCE),s15_assignments(source)
    for key,code in target_assign.items():
        if key=='_VQ8_MOD' and ns['VQ8_CUDA']!=base['VQ8_CUDA']:continue
        if key not in ('_GEMV_CHOICES',) and host_assign.get(key)==code and key in base and not callable(base[key]):
            ns[key]=base[key]
    for key in ('GB_ROWS','GATE_SPLIT_QKV3','GATE_SPLIT_ROWS'):
        assert ns[key]==base[key],(name,key)
    ns['_GEMV_CHOICES']=dict(base['_GEMV_CHOICES'])
    reused=s15_share_jit(base,ns)
    if ns['VQ8_CUDA']!=base['VQ8_CUDA']:
        assert ns['_VQ8_MOD'] is None,(name,'changed VQ8 cache leaked')
    return ns,reused


def lab(deadline,weights,state):
    started=time.monotonic()
    base=kernel_fn.__globals__
    device=weights['qkv_gate'].device
    generator=_ap_generate_input
    if torch.cuda.get_device_capability(device)!=(9,0):
        generator=ab_namespace({'_ap_fa3_scheduler_metadata':lambda *a,**kw:None})['_ap_generate_input']
    ns={'A':base};states={'A':state};reused={};compiled={};errors={};threads={}
    def native(name,v):
        try:v['_vq8_module']();compiled[name]=True
        except Exception as exc:errors[name]=type(exc).__name__+':'+str(exc)[:100]
    for name,spec in S15_VARIANTS.items():
        source=s15_restore(spec)
        v,reused[name]=s15_namespace(name,source,base);ns[name]=v
        # No expert-layout or projection-plan change in S15. Preserve exact prepared banks,
        # counters, tuner choices and NF plans. Only attention dispatch rows differ by arm.
        st=dict(state)
        st['partial_port_rows']=v['_PP_ROWS']
        st['vq8_rows']=dict(v['_VQ8_ROWS'])
        st['cluster_rows']=v.get('_CLUSTER_ROWS',())
        assert v['BEST7_SETTINGS']==base['BEST7_SETTINGS'],name
        states[name]=st
        th=threading.Thread(target=native,args=(name,v),daemon=True);threads[name]=th;th.start()
    for name,thread in threads.items():
        thread.join(max(0.,deadline-time.monotonic()-30))
        if name not in compiled:errors.setdefault(name,'native compile incomplete')
    result=dict(host_sha256=hashlib.sha256(S15_HOST_SOURCE.encode()).hexdigest(),seeds=16,
                rounds=dict(small=32,other=8),shared_jit=reused,shapes={},loads={},failures=[],errors=errors)
    seed_times={}
    for b,q,w in AB_CASES:
        key=(b,q,w);label=ab_label(key);tokens=b*q
        if time.monotonic()>deadline-8:result['failures'].append(label+':deadline');continue
        make=lambda seed:generator(batch=b,q_len=q,ctx=80000,layer='sliding' if w else 'full',
                    spread='agentic' if b>1 else 'fixed',dist='soft',page=1,seed=seed,device=device)
        data=make(AB_SEEDS[0]);variants={'A':(base,state),'A2':(base,state)}
        variants.update({name:(ns[name],states[name]) for name in S15_VARIANTS if name in compiled})
        graphs={};gates={}
        for name,(v,st) in variants.items():
            try:graphs[name]=ab_graph(lambda v=v,st=st:v['kernel_fn'](data,st))
            except Exception as exc:gates[name]='capture:'+type(exc).__name__+':'+str(exc)[:70]
        if 'A' not in graphs:result['failures'].append(label+':host capture');continue
        per=[]
        for seed in AB_SEEDS:
            if time.monotonic()>deadline-4:result['failures'].append(label+':incomplete');break
            if seed!=AB_SEEDS[0]:ab_copy(data,make(seed))
            for graph,_ in graphs.values():graph.replay()
            torch.cuda.synchronize()
            for name,(_,output) in graphs.items():
                if name!='A' and (reason:=ab_gate(output,graphs['A'][1])):gates.setdefault(name,reason)
            times=ab_time(graphs,32 if tokens<=8 else 8)
            per.append({name:statistics.mean(values) for name,values in times.items()})
        if gates:result['failures'].append(label+':'+str(gates))
        if per:
            seed_times[key]=per
            shape={'n':len(per),'r':{name:round(statistics.mean(s[name]/s['A'] for s in per),6)
                    for name in per[0] if name!='A'}}
            aa=[s['A2']/s['A'] for s in per]
            shape['aa']=[round(min(aa),6),round(max(aa),6)]
            result['shapes'][label]=shape
        graphs=variants=data=None
        gc.collect();torch.cuda.empty_cache()
    for (b,q),label in AB_LOADS.items():
        keys=((b,q,2048),(b,q,0))
        if not all(key in seed_times for key in keys):continue
        paired=list(zip(*(seed_times[key] for key in keys)))
        if len(paired)!=16:continue
        aa=[(46*s['A2']+16*f['A2'])/(46*s['A']+16*f['A']) for s,f in paired]
        load={'aa':[round(min(aa),6),round(max(aa),6)],'r':{}}
        for name in paired[0][0]:
            if name=='A':continue
            rr=[(46*s[name]+16*f[name])/(46*s['A']+16*f['A']) for s,f in paired]
            normalized=[r/a for r,a in zip(rr,aa)]
            load['r'][name]=[round(statistics.mean(rr),6),round(statistics.stdev(rr)/math.sqrt(16),6),
                             round(statistics.mean(normalized),6),round(statistics.stdev(normalized)/math.sqrt(16),6)]
        result['loads'][label]=load
    result.update(elapsed=round(time.monotonic()-started,2),left=round(deadline-time.monotonic(),2))
    result['complete']=not result['failures'] and not errors and len(result['shapes'])==12 and len(result['loads'])==6 and all(
        s['n']==16 and len(s['r'])==len(S15_VARIANTS)+1 for s in result['shapes'].values())
    encoded=json.dumps(result,separators=(',',':'))
    assert len(encoded)<8500,len(encoded)
    return ['META S15 16 fresh paired seeds; r=off/host; cold read before every replay; all controls measured',
            'RESULT '+encoded]
