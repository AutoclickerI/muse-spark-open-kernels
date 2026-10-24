def nt1_module(kind):
    if kind!='lop3':raise ValueError('S4 small-M admits only the measured one-LOP3 decoder')
    if kind not in NT1_MODULES:
        tag=hashlib.sha256(S4_NT1_CUDA.encode()).hexdigest()[:12]
        target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
        mod=load_inline(name='muse_nt1_limit_'+kind+tag,cpp_sources=NT1_CPP,cuda_sources=S4_NT1_CUDA,
            functions=['max_smem_attr','smem_needed','resident_blocks','run'],
            no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
        mod.smem_optin=mod.max_smem_attr()
        NT1_MODULES[kind]=mod
    return NT1_MODULES[kind]


@functools.lru_cache(None)
def issue_module():
    target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
    mod=load_inline(name='s4_d32g_'+hashlib.sha256(S4_D32_CUDA.encode()).hexdigest()[:12],
        cpp_sources=S4_D32_CPP,cuda_sources=S4_D32_CUDA,
        functions=['idist_init','idist_info','idist_route','idist_run','idist_finish'],
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
    ISSUE_OPTS['idist']=mod.idist_init()
    return mod


def _s4_prepare_banks(state,weights):
    bank=state['moe_bank']
    dev=weights['w13'].device
    state['s4_nt1_bank']=dict(gs13=bank['gs13'],gs2=bank['gs2'],pnw=bank['pnw'],sms=bank['grid'],
        counters={'lop3':torch.zeros(64,device=dev,dtype=torch.int32)},
        banks={'lop3':dict(w13p=bank['w13p'],w2p=bank['w2p'])})
    state['s4_d32_bank']=issue_word_bank(issue_prepare(dict(weights,post_norm_w=weights['post_expert_norm'])))


def _s4_gate_moe(name,tokens,length,state,weights):
    if name=='nt1_lop3' and tokens not in (1,8,16,32):raise ValueError('S4 small-M geometry')
    if name=='d32g_m128' and tokens!=128:raise ValueError('S4 D32g geometry')
    fixtures=[];wants=[]
    for seed in _ATTENTION_PUBLIC_SEEDS:
        fixture=_best7_moe_fixture(weights,tokens,length,seed)
        want=_best5_fp32(fixture,weights)
        if want is None:raise RuntimeError('prepare budget before S4 expert oracle')
        fixtures.append(fixture);wants.append(want)
    graph=_best5_graph(lambda data:_best6_call(data,state,name),fixtures[0])
    worst=0.;worst_row=0.
    for order in (range(3),range(2,-1,-1)):
        for index in order:
            _best6_refresh(graph,fixtures[index]);graph[1].replay()
            error,row=_best5_error(graph[2],wants[index])
            worst=max(worst,error);worst_row=max(worst_row,row)
    if worst>.01 or worst_row>.05:raise RuntimeError(f'S4 {name} FP32 gate {worst:.6f}/{worst_row:.6f}')
    return dict(source='three public streams, forward/reverse fresh route graph replays',
                fp32=worst,worst_row=worst_row,variant=name)
