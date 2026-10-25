PF_MOD={}
PF_CASES=(16,32)
PF_SEEDS=tuple(172903+7919*i for i in range(8))
PF_ROUNDS=8
PF_RESULTS=[]
PF_CLOCKS=True

def pf_module(d,clk=False):
    key=d,clk
    if key not in PF_MOD:
        major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
        tag=hashlib.sha256(PF_CUDA[d].encode()).hexdigest()[:12]
        PF_MOD[key]=load_inline(name=f'vq8_probpoly{d}_{tag}_{arch}_{int(clk)}',cpp_sources=PF_CPP,cuda_sources=PF_CUDA[d],functions=['vq8_forward','vq8_resources'],
            extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}',f'-DVQ8_CLK={int(clk)}'])
    return PF_MOD[key]

def pf_forward(qkv,data,d,clocks=None):
    P,S=qkv;cfg=data['config'];B,L=cfg['batch'],cfg['q_len'];dev=P.device
    if P.dtype==torch.bfloat16:raw=P[0] if P.dim()==3 else P;nsplit=ss=0
    else:raw=P;nsplit=S;ss=P.stride(0)
    grid=torch.cuda.get_device_properties(dev).multi_processor_count;segments=B
    nc=(grid+16*L*segments)*64 if clocks is not None else 0
    count=(grid+segments)*128*68+256
    part=torch.empty((count+nc,),device=dev,dtype=torch.float32)
    if nc:part[count:].zero_()
    out=torch.empty((B*L,1024),device=dev,dtype=torch.bfloat16)
    st=6 if torch.cuda.get_device_capability()[0]==9 else 2
    pf_module(d,clocks is not None).vq8_forward(raw.data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),data['seq_lens'].data_ptr(),
        data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),0,part.data_ptr(),out.data_ptr(),
        B,L,16,2,raw.stride(-2),data['page_table'].shape[1],cfg['page_size'],0,nsplit,8,grid,st,
        cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,ss,1,torch.cuda.current_stream(dev).cuda_stream)
    if nc:clocks.append(part[count:].view(torch.int64).view(grid+16*L*segments,32))
    return out

def pf_namespace(host,d):
    g=host.__dict__;m=types.ModuleType('pf_layer_'+str(d));v=m.__dict__;v.update(g)
    for k,f in g.items():
        if isinstance(f,types.FunctionType) and f.__globals__ is g:
            h=types.FunctionType(f.__code__,v,f.__name__,f.__defaults__,f.__closure__);h.__kwdefaults__=f.__kwdefaults__;h.__dict__.update(f.__dict__);v[k]=h
    v['_vq8_attention']=lambda qkv,data,setting:pf_forward(qkv,data,d)
    return m

def lab(deadline,weights,state=None):
    global PF_RESULTS
    started=time.monotonic();lines=['META frozen final S11 acda6c62 full verify, FP32 polynomial probability exponent vs native ex2; rescale/max stays native FP32 ex2.',
        'META 8 identical seeds x8 rotating read-cold graph replays; A2 duplicate; original CUPTI critical-path increments. TASK attention gates after real QKV projection.']
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as pool:
        af=pool.submit(tl_activity);fs=[pool.submit(pf_module,d,c) for d in (2,3) for c in ((False,True) if PF_CLOCKS else (False,))]
        host,st=tl_host(weights);act=af.result()
        for f in fs:f.result()
    variants={'A':host,'A2':host,'P2':pf_namespace(host,2),'P3':pf_namespace(host,3)}
    PF_RESULTS=[];device=weights['o'].device
    for B in PF_CASES:
        label='v'+str(B)+'F'
        if time.monotonic()>deadline-8:lines.append('INCOMPLETE '+label);break
        def mk(seed):return host._ap_generate_input(batch=B,ctx=80000,layer='full',q_len=8,spread='agentic',dist='soft',page=1,seed=seed,device=device)
        data=mk(PF_SEEDS[0]);graphs={};out={};profiles={};gates={};cycles={}
        p=st['best7_plans'][(B,8,0)];qkv=host._best7_project(data['attn_input'],'qkv_gate',st,p)
        for d in (2,3):
            got=pf_forward(qkv,data,d);torch.cuda.synchronize();host._attention_store_agrees(qkv,data)
            gate=host._attention_agrees(got,host._attention_reference(qkv,data),host._attention_reference(qkv,data,True))
            assert gate[0],('attention gate',label,d,gate)
            gates[str(d)]=gate
            if PF_CLOCKS:
                rows=[];lab_cold_l2();pf_forward(qkv,data,d,rows);torch.cuda.synchronize()
                cycles[str(d)]=clock_summary(rows[0],torch.cuda.get_device_properties(device).multi_processor_count)
        del qkv,got
        for n,v in variants.items():
            for _ in range(2):v.kernel_fn(data,st)
            torch.cuda.synchronize()
            if n=='A2':graphs[n],out[n],_,_=tl_capture(v,st,data,act,False)
            else:graphs[n],out[n],profiles[n]=cs_profile(v,st,data,act)
        ratios={n:[] for n in variants if n!='A'};times={n:[] for n in variants};drift={n:[] for n in ratios}
        for seed in PF_SEEDS:
            if time.monotonic()>deadline-4:break
            if seed!=PF_SEEDS[0]:cs_copy(data,mk(seed))
            for g in graphs.values():g.replay()
            torch.cuda.synchronize()
            for n in ratios:drift[n].append(cs_delta(out[n],out['A']))
            per={n:[] for n in variants};names=list(variants)
            for i in range(PF_ROUNDS):
                for n in names[i%len(names):]+names[:i%len(names)]:per[n].extend(tl_times(graphs[n],1))
            av={n:statistics.mean(t) for n,t in per.items()}
            for n in variants:times[n].append(av[n])
            for n in ratios:ratios[n].append(av[n]/av['A'])
        row=dict(case=label,A_us=statistics.mean(times['A']),r={n:statistics.mean(r) for n,r in ratios.items()},
            se={n:statistics.stdev(r)/len(r)**.5 if len(r)>1 else 0 for n,r in ratios.items()},inc={n:p['inc']['attn'] for n,p in profiles.items()},
            span={n:p['stages']['attn']['span'] for n,p in profiles.items()},times=times,ratios=ratios,gates=gates,clocks=cycles,
            drift={n:[max(col) for col in zip(*d)] for n,d in drift.items()},seeds=len(times['A']))
        PF_RESULTS.append(row)
        line='PF '+json.dumps(row,separators=(',',':'));lines.append(line);print(line,flush=True)
        del graphs,out,profiles,data;gc.collect();torch.cuda.empty_cache()
    lines.append('END shapes=%d/2 elapsed=%.1fs'%(len(PF_RESULTS),time.monotonic()-started))
    return lines
