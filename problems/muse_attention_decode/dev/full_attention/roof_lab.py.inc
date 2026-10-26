SEEDS=tuple(1671223+7919*i for i in range(8))
CASES=tuple((b,l) for l in (1,8) for b in (1,16,32))

def module(name,source):
    f='<p5-roof-'+name+'>';linecache.cache[f]=(len(source),None,source.splitlines(True),f)
    m=types.ModuleType('p5_roof_'+name);m.__file__=f;exec(compile(source,f,'exec'),m.__dict__);return m

def lab(deadline,weights):
    begin=time.monotonic();fail=[];counts={'official':0,'scored':0,'profile':0}
    checker=module('public_TASK',_BUNDLE['checker']).__dict__
    if torch.cuda.get_device_capability()[0]!=9:checker['_fa3_scheduler_metadata']=lambda *a,**k:None
    spaces={};sources={}
    for name,spec in [('A',None),*_BUNDLE['arms'].items()]:
        src=_BUNDLE['base']
        if spec:
            lines=src.splitlines(True)
            for lo,hi,s in reversed(spec['edits']):lines[lo:hi]=[s]
            src=''.join(lines);assert hashlib.sha256(src.encode()).hexdigest()==spec['sha256']
        m=module(name,src+_BUNDLE['adapter'])
        if install:=globals().get('ROOF_LOCAL_INSTALL'):install(m)
        spaces[name]=m.__dict__;sources[name]=src
    spaces['A2']=spaces['A'];order=['A','A2','F','G','P','H1','H2']
    arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
    from torch.utils.cpp_extension import load_inline
    compiled={}
    def load(cu,cpp,functions,flags=()):
        key=hashlib.sha256((cu+cpp+repr(flags)).encode()).hexdigest()[:14]
        if key not in compiled:
            compiled[key]=load_inline(name='p5_roof_'+key+'_'+arch,
                cpp_sources='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n'+cpp,cuda_sources=cu,
                functions=functions,no_implicit_headers=True,with_pytorch_error_handling=False,
                extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}',*flags])
        return compiled[key]
    for name in ('A','F','G','P','H1','H2'):
        ns=spaces[name]
        ns['_ATTN_PORT_EXT']=load(ns['ATTN_PORT_CUDA_SRC'],ns['ATTN_PORT_CPP_SRC'],['attn_forward','occupancy','max_cluster'])
        ns['_PP_MOD']={False:load(ns['PARTIAL_CUDA'],ns['PARTIAL_CPP'],['attn_forward','occupancy','max_cluster'])}
        if arch=='90a':ns['_VQ8_MOD']=load(ns['VQ8_CUDA'],ns['VQ8_CPP'],['vq8_forward','vq8_resources'])
        else:ns['_vq8_module']()
    resources={name:spaces[name]['_vq8_module']().vq8_resources(4 if arch=='90a' else 2) for name in ('A','H1','H2')}
    e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    def graph(fn):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):out=fn()
        torch.cuda.current_stream().wait_stream(stream);g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        return g,out
    x=torch.zeros(1,device='cuda');empty,_=graph(lambda:x.add_(1))
    def replay(g):
        lab_cold_l2();e0.record();g.replay();e1.record();e1.synchronize();return e0.elapsed_time(e1)*1000
    fixed=statistics.median(replay(empty) for _ in range(16))
    raw_ref=checker['ref_kernel'];cache={}
    def cached_ref(data,query_dtype=None,new_kv=None):
        key=(query_dtype,None if new_kv is None else tuple(t.view(torch.uint8).cpu().numpy().tobytes() for t in new_kv))
        if key not in cache:cache[key]=raw_ref(data,query_dtype,new_kv)
        return cache[key]
    checker['ref_kernel']=cached_ref
    def check(data,y,label,group):
        ok,msg=checker['check_implementation'](data,checker['collect'](data,y));counts[group]+=1
        if not ok:fail.append(label+':'+msg)
    # Every official TASK fixture; every arm eagerly and after captured replay.
    official=[dict(batch=1,ctx=10,layer='sliding'),dict(batch=3,ctx=2047,layer='sliding'),
        dict(batch=5,ctx=5000,layer='sliding',q_len=8,spread='mixed',page=1),dict(batch=4,ctx=3000,layer='full'),
        dict(batch=8,ctx=8192,heads='dp',layer='full',spread='mixed'),dict(batch=2,ctx=100000,layer='full',q_len=8,page=1),
        dict(batch=4,ctx=6000,layer='sliding',dist='edges',page=1),dict(batch=4,ctx=6000,layer='full',q_len=8,dist='edges'),
        dict(batch=1,ctx=1048576,layer='full',q_len=8),dict(batch=3,ctx=4000,layer='sliding',q_len=8,page=1,kv_scale=.015625),
        dict(batch=2,ctx=9000,layer='full',kv_scale=.015625),dict(batch=4,ctx=80000,layer='full',spread='mixed',dist='soft'),
        dict(batch=4,ctx=6000,layer='sliding',q_len=8,dist='soft',page=1),dict(batch=2,ctx=20000,heads='dp',layer='full',q_len=8,dist='soft'),
        dict(batch=4,ctx=80000,layer='full',spread='mixed',dist='coherent',page=1),dict(batch=2,ctx=20000,heads='dp',layer='full',q_len=8,dist='coherent')]
    for i,spec in enumerate(official):
        data=checker['generate_input'](heads='tp8',seed=930117+i,**{k:v for k,v in spec.items() if k!='heads'}) if 'heads' not in spec else checker['generate_input'](seed=930117+i,**spec)
        cache.clear()
        for name in order:
            fn=lambda:spaces[name]['kernel_fn'](data)
            y=fn();torch.cuda.synchronize();check(data,y,f'official{i}:{name}:eager','official')
            g,y=graph(fn);g.replay();g.replay();torch.cuda.synchronize();check(data,y,f'official{i}:{name}:graph','official');del g,y
        del data;cache.clear();gc.collect();torch.cuda.empty_cache()
    assert counts['official']==224 and not fail,('official TASK',counts,fail)
    # Profile the actual S15 N128 code, not the old N64 reset source.
    profile_modules={};profiles={}
    if arch=='90a':
        for key in ('q1_clock','partial_clock'):
            profile_modules[key]=load(_BUNDLE['clocks'][key],_BUNDLE['partial_CPP' if key.startswith('partial') else 'CPP'],['attn_forward','occupancy','max_cluster'],['-DFULL_CLOCK=1'])
        profile_modules['vq_clocks']={name:load(spaces[name]['VQ8_CUDA'],_BUNDLE['VQ8_CPP'],['vq8_forward','vq8_resources'],['-DVQ8_CLK=1']) for name in ('A','H1','H2')}
        exec(compile(_BUNDLE['profile'],'<p5-roof-profile>','exec'),globals())
    times={};summary={};checks_before=counts['scored']
    for batch,length in CASES:
        label=f'{batch}x{length}f';samples=[];byte=[]
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-8:raise TimeoutError((label,si,counts))
            data=checker['generate_input'](batch=batch,ctx=80000,heads='tp8',layer='full',q_len=length,
                page=1,dist='coherent',spread='agentic' if batch>1 else 'fixed',seed=seed)
            cache.clear();graphs={}
            # Fresh graph pools, rotated ownership on every seed, not one fixed
            # first-capture position for all8seeds (the old d1 A:A bias).
            capture_order=order[si%len(order):]+order[:si%len(order)]
            for name in capture_order:
                g,y=graph(lambda ns=spaces[name]:ns['kernel_fn'](data));g.replay();torch.cuda.synchronize()
                check(data,y,label+':'+name,'scored');graphs[name]=(g,y)
            if si==0 and profile_modules:
                profiles[label]=profile(spaces['A'],data,profile_modules,lambda d,y,n:check(d,y,label+':'+n,'profile'))
            per={name:[] for name in order}
            for ri in range(8):
                rotated=order[(ri+si)%len(order):]+order[:(ri+si)%len(order)]
                if ri&1:rotated.reverse()
                for name in rotated:per[name].append(max(.001,replay(graphs[name][0])-fixed))
            samples.append({name:round(statistics.mean(v),3) for name,v in per.items()})
            byte.append(int(data['seq_lens'].long().sum())*256)
            graphs=data=None;cache.clear();gc.collect();torch.cuda.empty_cache()
        times[label]=samples;row={}
        for name in order[1:]:
            rr=[x[name]/x['A'] for x in samples];mean=statistics.mean(rr);se=statistics.stdev(rr)/len(rr)**.5
            row[name]=dict(r=mean,se=se,lo=mean-2.364624251*se,hi=mean+2.364624251*se)
        summary[label]=dict(control_us=statistics.mean(x['A'] for x in samples),bytes=statistics.mean(byte),ratios=row)
    result=dict(base_sha256=hashlib.sha256(_BUNDLE['base'].encode()).hexdigest(),counts=counts,failures=fail,
        seed_us=times,summary=summary,profiles=profiles,resources=resources,seeds=SEEDS,
        elapsed=time.monotonic()-begin,complete=counts['scored']==336 and not fail)
    blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode()
    assert len(blob)<8400,len(blob)
    return ['META exact S15 full attention; A:A/fresh rotated graph pools;H1/H2 one-head residency,F balanced tiles,G quad pages,P native verify1;8seeds8cold;TASK only',
            'RESULT_XZ85 '+blob]
