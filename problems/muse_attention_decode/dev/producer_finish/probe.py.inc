SEEDS=tuple(65000087+104729*i for i in range(8))
CASES=((1,1),(16,1),(32,1),(1,8),(16,8),(32,8))
ROUNDS=6
LOCAL=False
FINISH_MOD=None
FREQ={}
def pf_module():
    global FINISH_MOD
    if FINISH_MOD is None:
        arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
        cpp='#include <cstdint>\n'+re.search(r'void pf_forward\([^)]*\)',FINISH_CUDA,re.S).group(0)+';'
        FINISH_MOD=load_inline(name='p5finish_'+hashlib.sha256(FINISH_CUDA.encode()).hexdigest()[:12]+'_'+arch,cpp_sources=cpp,cuda_sources=FINISH_CUDA,functions=['pf_forward'],extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return FINISH_MOD

def pf_host():
    fn='<P5-finish-S19>';linecache.cache[fn]=(len(BASE),None,BASE.splitlines(True),fn);m=types.ModuleType('pf_s19');exec(compile(BASE,fn,'exec'),m.__dict__);return m

def pf_call(data,pair,empty=0):
    P,S=pair;raw=P[0] if P.dtype==torch.bfloat16 and P.dim()==3 else P
    c=data['config'];T=c['tokens'];dev=P.device
    q=torch.empty((T,16,64),device=dev,dtype=torch.float8_e4m3fn);clocks=torch.empty((T,3),device=dev,dtype=torch.int64)
    key=(c['rope_theta'],str(dev))
    if key not in FREQ:FREQ[key]=(1./(c['rope_theta'] or 1.)**(torch.arange(0,64,2,device=dev).float()/64)).contiguous()
    freq=FREQ[key]
    pf_module().pf_forward(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),freq.data_ptr(),q.data_ptr(),clocks.data_ptr(),T,raw.shape[-1],S,int(P.dtype==torch.bfloat16),int(bool(c['window'])),P.stride(0),raw.stride(-2),empty,torch.cuda.current_stream().cuda_stream)
    return q,clocks

def pf_ready_attention(d,q,g):
    # Independent untimed use of the probe's actual Q/cache values, followed
    # by the unchanged public TASK checker. Never a timed candidate.
    c=d['config'];B,L=c['batch'],c['q_len'];q=q.float().view(B,L,2,8,64)*c['qk_scale_factor'];out=torch.empty((B,L,16,64),device=q.device)
    kf=d['k_cache'].view(-1,2,64);vf=d['v_cache'].view(-1,2,64)
    for b in range(B):
        seq=int(d['seq_lens'][b]);lo=max(0,seq-c['window']+1) if c['window'] else 0
        past=torch.arange(lo,seq,device=q.device);page=c['page_size'];rows=d['page_table'][b].long()[past//page]*page+past%page
        new=d['cache_loc'].view(B,L)[b];rows=torch.cat((rows,new));k=kf[rows].float()*d['k_scale'];v=vf[rows].float()*d['v_scale']
        for j in range(L):
            at=max(0,seq+j-c['window']+1)-lo if c['window'] else 0;end=seq+j+1-lo
            a=torch.einsum('hgd,thd->hgt',q[b,j],k[at:end])*c['sm_scale'];out[b,j]=torch.einsum('hgt,thd->hgd',a.softmax(-1),v[at:end]).reshape(16,64)
    out=out*torch.rsqrt(out.square().mean(-1,keepdim=True)+1e-5);return (out*torch.sigmoid(g.view(B,L,16,64))).bfloat16().view(B,L,-1)

def pf_measure(graphs,names):
    per={n:[] for n in names}
    for ri in range(ROUNDS):
        order=names[ri%len(names):]+names[:ri%len(names)]
        if ri&1:order=order[::-1]
        for n in order:per[n].extend(tl_times(graphs[n],1))
    return {n:round(statistics.mean(v)*1000) for n,v in per.items()}

def lab(deadline,weights,state=None):
    started=time.monotonic();host=pf_host();pf_module()
    weights=PUBLIC_SPEC['generate_weights'](64000021,device='cuda',router_seed=64000109) if weights is None else weights
    state=host.prepare(weights);act=tl_activity();results=[]
    for B,L in CASES:
      for window in (2048,0):
        times={n:[] for n in ('A','A2','P','PE','PF')};critical=[];profiles=[];clocks=[];task=0;finish_task=0
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-15:raise TimeoutError(('incomplete sizing',B,L,window,si))
            d=host._ap_generate_input(batch=B,ctx=80000,layer='sliding' if window else 'full',q_len=L,spread='agentic' if B>1 else 'fixed',dist='soft' if window else 'coherent',page=1,seed=seed,device='cuda');graphs={};outs={}
            plan=state['best7_plans'][(B,L,window)];role='qkv3' if host._gate_split_on(d,state,plan) else 'qkv_gate'
            capture_order=('A','A2') if si%2==0 else ('A2','A')
            for n in capture_order:
                for _ in range(2):host.kernel_fn(d,state)
                torch.cuda.synchronize()
                graphs[n],outs[n],_,_=tl_capture(host,state,d,act,False)
            checker=eh_checker(d,weights)
            for n in ('A','A2'):
                graphs[n].replay();torch.cuda.synchronize();ok,msg=checker(outs[n]);assert ok,(B,L,window,seed,n,msg);task+=1
            del checker
            diagnostic,_,p=cs_profile(host,state,d,act)
            critical.append({k:v for k,v in p['inc'].items() if k in ('attn','qkv_gate','qkv3')});del diagnostic
            holder={}
            for n in ('P','PE','PF'):
                def call(n=n):
                    pair=host._best7_project(d['attn_input'],role,state,plan);holder['pair']=pair
                    if n!='P':holder[n]=pf_call(d,pair,int(n=='PE'))
                for _ in range(2):call()
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):call()
                graphs[n]=graph
            sample=pf_measure(graphs,list(times))
            for n in times:times[n].append(sample[n])
            graphs['PF'].replay();torch.cuda.synchronize();P,S=holder['pair'];flat=(P[:S].sum(0) if P.dtype==torch.float32 else P.reshape(-1,P.shape[-1])).bfloat16()
            if flat.shape[-1]==1280:flat=torch.cat((flat,torch.zeros((B*L,1024),device='cuda',dtype=torch.bfloat16)),-1)
            cfg=dict(d['config'],num_q_heads=16,num_kv_heads=2,head_dim=64);checkdata=dict(d,config=cfg,qkv=flat.view(B,L,-1))
            q,clk=holder['PF'];g=flat[:,1280:].float();got=pf_ready_attention(checkdata,q,g);ok,msg=ATTN_SPEC['check_implementation'](checkdata,ATTN_SPEC['collect'](checkdata,got));assert ok,(B,L,window,seed,'finish',msg);finish_task+=1
            vals=clk.cpu().tolist();clocks.append([max(x[2] for x in vals)-min(x[0] for x in vals),round(statistics.median(x[2]-x[1] for x in vals)),max(x[2]-x[1] for x in vals),max(x[1]-x[0] for x in vals)])
            if si==0:
                lab_cold_l2();torch.cuda.synchronize();act.profile_begin();graphs['PF'].replay();torch.cuda.synchronize();nodes=act.profile_end();start=min(v[0] for _,v in nodes)
                producer_nodes=[dict(name=n[:200],start_ns=v[0]-start,end_ns=v[1]-start,resource=list(v[7:13])) for n,v in nodes]
                abi=dict(role=role,partial_splits=S,dtype=str(P.dtype),stride=list(P.stride()),width=P.shape[-1])
            del graphs,outs,holder,d;gc.collect();torch.cuda.empty_cache()
        results.append(dict(B=B,L=L,window=window,seeds=list(SEEDS),times_ns=times,critical=critical,abi=abi,producer_nodes=producer_nodes,finish_clocks_ns=clocks,TASK=task,finish_TASK=finish_task))
    return {'packet':dict(identities=dict(base_sha256=hashlib.sha256(BASE.encode()).hexdigest(),finish_sha256=hashlib.sha256(FINISH_CUDA.encode()).hexdigest()),seeds=list(SEEDS),paired=results,checks=dict(layer_TASK=sum(r['TASK'] for r in results),finish_TASK=sum(r['finish_TASK'] for r in results)),complete=True,elapsed_s=time.monotonic()-started,scope='Sizing only: A/A2 exact S19 whole layer; P producer; PE producer+same-resource empty finish; PF producer+standalone full finish. No producer fusion implemented or hidden ALU asserted. PF/PE are component diagnostic ratios, not candidate score gains.'),'logs':['P5 producer-finish sizing complete; every baseline and probe-output uses public TASK; all raw seed times retained.']}
