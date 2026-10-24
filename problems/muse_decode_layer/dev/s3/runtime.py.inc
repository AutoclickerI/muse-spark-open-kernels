@functools.lru_cache(None)
def rc_module():
    target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
    mod=load_inline(name='rc_m256_'+hashlib.sha256(RC_CUDA.encode()).hexdigest()[:12],
        cpp_sources=RC_CPP,cuda_sources=RC_CUDA,
        functions=['rc_init','rc_info','rc_route','rc_run','rc_finish','n1_init','n1_smem','n1_resident','n1_run'],
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
    mod.rc_init()
    mod.n1_init()
    return mod


def _s3_gate_moe(tokens,length,state,weights):
    if tokens!=256:raise ValueError('C1g2112 is admitted only at M256')
    if not state['s3_bank']['encoded']:raise ValueError('C1g2112 requires lossless recoded bank')
    fixtures=[];wants=[]
    for seed in _ATTENTION_PUBLIC_SEEDS:
        fixture=_best7_moe_fixture(weights,tokens,length,seed)
        want=_best5_fp32(fixture,weights)
        if want is None:raise RuntimeError('prepare budget before C1g2112 oracle')
        fixtures.append(fixture);wants.append(want)
    graph=_best5_graph(lambda data:rc_call(data,state['s3_bank'],1,2112),fixtures[0])
    worst=0.;worst_row=0.
    for order in (range(3),range(2,-1,-1)):
        for index in order:
            _best6_refresh(graph,fixtures[index]);graph[1].replay()
            error,row=_best5_error(graph[2],wants[index])
            worst=max(worst,error);worst_row=max(worst_row,row)
    if worst>.01 or worst_row>.05:raise RuntimeError(f'C1g2112 FP32 gate {worst:.6f}/{worst_row:.6f}')
    return dict(source='three public streams, forward/reverse fresh route graph replays',
                fp32=worst,worst_row=worst_row,which=1,grid=2112,encoded=True,
                bank='lossless NVFP4 exponent/mantissa and folded sign/zero')
