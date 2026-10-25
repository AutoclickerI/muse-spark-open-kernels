_CS_CLOCK_MOD={}
def cs_clock_module(warp):
    if warp not in _CS_CLOCK_MOD:
        cuda=CS_CLOCK_CUDA[warp]
        import re
        cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
        major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
        tag=hashlib.sha256(cuda.encode()).hexdigest()[:12]
        _CS_CLOCK_MOD[warp]=load_inline(name=f'cluster_sliding_clock_{tag}_{arch}',cpp_sources=cpp,cuda_sources=cuda,functions=['sliding_forward','sliding_resources'],
            extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return _CS_CLOCK_MOD[warp]

def cs_phase(qkv,data,warp,host,st,nk=128,cs=None):
    cfg=dict(data['config'],num_q_heads=16,num_kv_heads=2,head_dim=64);qt=4 if cfg['batch']*2<=4 else 1
    mod=cs_clock_module(warp)
    if cs is None:
        resources=[(c,mod.sliding_resources(cfg['q_len'],qt,nk,c)) for c in range(1,17)]
        resources=[r for r in resources if r[1][-1]>0]
        viable=[r for r in resources if r[1][-1]>=cfg['batch']*2*qt]
        cs=max(viable or resources[:1],key=lambda r:r[0])[0]
    clocks=torch.empty((cfg['batch']*2*qt*cs,8),device=qkv[0].device,dtype=torch.int64)
    class Proxy:
        def sliding_forward(self,*args):return mod.sliding_forward(*args,clocks.data_ptr())
    original=globals()['cs_module'];globals()['cs_module']=lambda:Proxy()
    try:
        p=st['best7_plans'].get((cfg['batch'],cfg['q_len'],cfg['window']),st['best7_plans']['other'])
        role='qkv3' if host._gate_split_on(data,st,p) else 'qkv_gate'
        def run():
            actual=host._best7_project(data['attn_input'],role,st,p)
            return cs_attention(dict(data,config=cfg),actual,nk=nk,cs=cs)
        graph=torch.cuda.CUDAGraph();run();torch.cuda.synchronize()
        with torch.cuda.graph(graph):out=run()
        for _ in range(3):lab_cold_l2();graph.replay();torch.cuda.synchronize()
    finally:globals()['cs_module']=original
    c=clocks.cpu();t0=c[:,0].min();med=lambda x:round(float(x.double().median())/1000,3)
    return dict(warp=warp,nk=nk,cs=cs,ctas=len(c),max_us=[round(float((c[:,i]-t0).max())/1000,3) for i in range(8)],
        median_us=[med(c[:,i]-c[:,0]) for i in range(8)],phase_median_us=[med(c[:,i]-c[:,i-1]) for i in range(1,8)])
