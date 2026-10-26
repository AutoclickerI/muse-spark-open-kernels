import ast,base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types
import torch
from torch.utils.cpp_extension import load_inline
SEEDS=(19100023,19204752,19309481)
CASES=((1,2048),(8,2048),(16,512),(16,1024),(16,2048),(16,4096),(16,8192),
       (32,512),(32,1024),(32,2048),(32,4096),(32,8192))
SPLITS=(0,1,2,4,8)
ROUNDS=6
RESULTS=[]

def diag_host():
    fn='<p5-window-exact-S15>';linecache.cache[fn]=(len(BASE),None,BASE.splitlines(True),fn)
    h=types.ModuleType('p5_window_s15');exec(compile(BASE,fn,'exec'),h.__dict__)
    return h

def diag_activity():
    return load_inline(name='muse_layer_timeline_'+hashlib.sha256(ACTIVITY.encode()).hexdigest()[:12],
        cpp_sources=ACTIVITY,functions=['profile_begin','profile_end','capture_nodes','exec_id'],
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_include_paths=['/usr/local/cuda/include'],extra_ldflags=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcupti','-lcudart'])

def diag_call(host,qkv,data,split):
    if globals().get('LOCAL_STANDIN',False):
        return host._layer_stage_attention(qkv,data,{'inv_freq':host._inv_freq(data['config']['rope_theta'],'cuda')}).view(data['config']['batch'],8,-1)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    q,gate=host.qk_prep(qkv,data,{'inv_freq':host._inv_freq(data['config']['rope_theta'],'cuda')},True)
    c=data['config']
    a=flash_attn_with_kvcache(q=q,k_cache=data['k_cache'],v_cache=data['v_cache'],
        page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],
        max_seqlen_q=8,softmax_scale=c['sm_scale']*c['qk_scale_factor'],causal=True,
        window_size=(c['window']-1,0),scheduler_metadata=None,num_splits=split)
    if isinstance(a,tuple):a=a[0]
    o=torch.empty((c['tokens'],host.Q_SIZE),device='cuda',dtype=torch.bfloat16)
    host._launch(host._norm_gate_kernel,(c['tokens'],),a,gate,o,host.RMS_EPS,HQ=host.HQ,D=host.HD,num_warps=4)
    return o.view(c['batch'],8,-1)

def diag_checker(spec,data):
    # Memoize only the independent, untimed public reference calculations.
    ns=dict(spec)
    for k,f in tuple(ns.items()):
        if isinstance(f,types.FunctionType) and f.__globals__ is spec:
            g=types.FunctionType(f.__code__,ns,f.__name__,f.__defaults__,f.__closure__);g.__kwdefaults__=f.__kwdefaults__;ns[k]=g
    ref=ns['ref_kernel'];cache={}
    def cached(data,query_dtype=ns['QUERY_DTYPE'],new_kv=None):
        key=(query_dtype,None if new_kv is None else tuple(x.contiguous().view(torch.uint8).cpu().numpy().tobytes() for x in new_kv))
        if key not in cache:cache[key]=ref(data,query_dtype,new_kv)
        return cache[key]
    ns['ref_kernel']=cached
    return lambda out:ns['check_implementation'](data,ns['collect'](data,out))

def lab(deadline,weights,state=None):
    global RESULTS
    spec={};exec(SPEC,spec);host=diag_host();act=diag_activity();legend=[];RESULTS=[]
    started=time.monotonic()
    for B,W in CASES:
        graphs={};out={};samples={str(s):[] for s in SPLITS};samples['A2']=[];gates={n:0 for n in samples};profiles={}
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-12:raise TimeoutError(('incomplete diagnostic',B,W,si))
            d=spec['generate_input'](B,80000,'tp8','sliding',seed,q_len=8,spread='fixed' if B==1 else 'agentic',dist='soft',page=1)
            d['config']['window']=W;d['config']['tokens']=B*8
            qkv=(d['qkv'].reshape(1,-1,host.QKV_SIZE),1)
            # BF16 input with S=1 uses the same qk_prep rounding as the layer.
            for n in samples:
                split=0 if n=='A2' else int(n)
                for _ in range(2):diag_call(host,qkv,d,split)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out[n]=diag_call(host,qkv,d,split)
                graphs[n]=graph
            check=diag_checker(spec,d)
            for n,g in graphs.items():
                g.replay();torch.cuda.synchronize();ok,msg=check(out[n]);assert ok,(B,W,seed,n,msg);gates[n]+=1
            del check
            per={n:[] for n in samples};names=list(samples)
            for j in range(ROUNDS):
                order=names[j%len(names):]+names[:j%len(names)]
                if j&1:order=order[::-1]
                for n in order:
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    lab_cold_l2();a.record();graphs[n].replay();b.record();b.synchronize();per[n].append(a.elapsed_time(b)*1000)
            for n in samples:samples[n].append(statistics.mean(per[n]))
            if si==0:
                for n,g in graphs.items():
                    lab_cold_l2();torch.cuda.synchronize();act.profile_begin();g.replay();torch.cuda.synchronize();rows=act.profile_end();nodes=[]
                    origin=min(v[0] for name,v in rows)
                    for name,v in rows:
                        if name not in legend:legend.append(name)
                        nodes.append([legend.index(name),v[0]-origin,v[1]-origin,*v[7:13]])
                    profiles[n]=nodes
            graphs.clear();out.clear();del d,qkv;gc.collect();torch.cuda.empty_cache()
        RESULTS.append(dict(B=B,window=W,seeds=list(SEEDS),times=samples,task=gates,profiles=profiles))
    archive=dict(s15_sha256=hashlib.sha256(BASE.encode()).hexdigest(),names=legend,results=RESULTS,elapsed_s=time.monotonic()-started,
        note='All event times include graph replay overhead. Subtract measured empty graph overhead for a stage cost model; node durations overlap under PDL.')
    blob=base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode())).decode()
    assert len(blob)<8600,('diagnostic archive cap',len(blob))
    return ['META P5 exact S15 FA3 num_splits0/1/2/4/8 +A:A; full-name CUPTI;12 batch/window cells;3fresh seeds6cold rounds; public attention TASK only.',
            'WINDOW_DIAG_XZ85 '+blob,'END cases='+str(len(RESULTS))+' elapsed='+str(round(time.monotonic()-started,2))]
