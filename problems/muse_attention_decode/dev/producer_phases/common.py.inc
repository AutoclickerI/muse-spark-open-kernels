
# The gate above is frozen from the task; this caches its three FP32 passes
# per input only inside the untimed lab. Timed kernels never read these values.
def gate_prepare(data):
    c=data['config'];slots=new_slots(data);hk=c['num_kv_heads']
    actual=tuple(data[name].view(-1,hk,64)[slots].clone() for name in ('k_cache','v_cache'))
    _,want_k,want_v=ref_kernel(data)
    want=ref_kernel(data,new_kv=actual)[0].to(torch.bfloat16)
    fp8=ref_kernel(data,torch.float8_e4m3fn,new_kv=actual)[0].to(torch.bfloat16)
    tol=query_precision_bound(relative_l2(fp8,want),FP8_QUERY_FACTOR,FP8_QUERY_FLOOR,REL_L2_TOL,FP8_QUERY_SLACK)
    return dict(out=want,tol=tol,kv=(want_k,want_v),slots=slots)


def gate_cached(data,out,expected):
    c=data['config'];hk=c['num_kv_heads']
    for i,name in enumerate(('k_cache','v_cache')):
        actual=data[name].view(-1,hk,64)[expected['slots']]
        scale=data['k_scale' if i==0 else 'v_scale']
        ok,msg=compare(_dequant(actual,scale),_dequant(expected['kv'][i],scale),CACHE_TOL)
        if not ok:return False,'cache '+msg,float('inf')
    out=out.view_as(expected['out'])
    ok,msg=compare(out,expected['out'],expected['tol'])
    return ok,msg,relative_l2(out,expected['out'])


def timed(fn,data,flush,rounds=5):
    fn(data);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out=fn(data)
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    values=[]
    for _ in range(rounds):
        flush.zero_();start.record();graph.replay();end.record();end.synchronize()
        values.append(start.elapsed_time(end)*1000.)
    return sorted(values)[len(values)//2],out,graph
