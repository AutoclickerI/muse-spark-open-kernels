from pathlib import Path
import ast
root=Path(__file__).resolve().parents[1]
base=(root/'dev/xqa_legacy_candidate.py').read_text()
# Also compare scalar FP8->half V gather and its raw-FP8 PV variant in the same job.
scalar=ast.parse((root/'dev/xqa_p8_candidate.py').read_text())
scuda=next(ast.literal_eval(n.value) for n in scalar.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='X_CUDA' for t in n.targets))
base+='\nX_SCALAR_CUDA = r"""\n'+scuda+'\n"""\n'
base+=r'''
_X_FAMILIES = {}
X_FAMILY = 'matrix'
_XPLAN = {}
_PREPARED = False

def _xext():
    family=X_FAMILY
    if family not in _X_FAMILIES:
        major,minor=torch.cuda.get_device_capability();arch=f'{major}{minor}'
        src=X_CUDA if family=='matrix' else X_SCALAR_CUDA
        _X_FAMILIES[family]=load_inline(name=f'muse_xqa_diag_{family}_sm{arch}',cpp_sources=X_CPP,cuda_sources=src,
            functions=['xforward'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',
                f'-gencode=arch=compute_{arch},code=sm_{arch}'],verbose=False)
    return _X_FAMILIES[family]

def _set_xvariant(variant):
    global X_FAMILY,X_P8,X_STAGES,X_WAVES
    X_FAMILY,X_P8,X_STAGES,X_WAVES=variant

def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg=data['config']
    if cfg['q_len']==1:
        return _DECODE_FN(data)
    variant=_XPLAN.get((cfg['batch'],bool(cfg['window'])))
    if variant=='fa3':
        return _fa3_verify(data)
    if variant is not None:
        _set_xvariant(variant)
    return _xqa_verify(data)

def _synthetic(batch,window):
    device=torch.device('cuda'); length=80000;maxpages=length+8
    # Seeded synthetic inputs only: no timed input, output or cache is retained.
    gen=torch.Generator(device=device).manual_seed(197+batch+window)
    seq=torch.full((batch,),length,dtype=torch.int32,device=device)
    pt=torch.randperm(batch*maxpages,generator=gen,device=device,dtype=torch.int64).to(torch.int32).reshape(batch,maxpages)
    positions=(seq.long()[:,None]+torch.arange(8,device=device)).reshape(-1)
    loc=torch.gather(pt.long(),1,positions.view(batch,8)).reshape(-1)
    # FP8 raw cache, with all rows present and page-1 gathers. The magnitudes do
    # not change the instruction count of either attention implementation.
    shape=(batch*maxpages,1,2,64)
    kc=torch.randn(shape,generator=gen,device=device,dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    vc=torch.randn(shape,generator=gen,device=device,dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    qkv=torch.randn((batch,8,(16+2*2+16)*64),generator=gen,device=device,dtype=torch.bfloat16)
    return dict(qkv=qkv,positions=positions,cache_loc=loc,cache_seqlens=seq+8,
        cu_seqlens_q=torch.arange(0,batch*8+1,8,device=device,dtype=torch.int32),scheduler_metadata=None,
        k_cache=kc,v_cache=vc,k_scale=torch.ones((),device=device),v_scale=torch.ones((),device=device),
        page_table=pt,seq_lens=seq,
        config=dict(batch=batch,q_len=8,num_q_heads=16,num_kv_heads=2,head_dim=64,page_size=1,
            window=window,rope_theta=10000.0 if window else 0.0,qk_scale_factor=5.473,sm_scale=.125,kv_scale=1.0))

def _diagnose(include_fa3):
    import statistics
    # Own graph timing on synthetic data is the permitted prepare-time autotuner
    # (R-20261023-7). Never inspect or modify the judge's timing or graph state.
    flush=torch.empty(256<<20,dtype=torch.uint8,device='cuda')
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    variants=[(f,p,s,w) for f in ('matrix','scalar') for p in (False,True) for s in (3,5) for w in (1,2,4)]
    if include_fa3:variants.append('fa3')
    for batch in (1,16,32):
        for window in (0,2048):
            data=_synthetic(batch,window);timings=[]
            for variant in variants:
                if variant=='fa3':fn=_fa3_verify
                else:_set_xvariant(variant);fn=_xqa_verify
                for _ in range(2):fn(data)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=fn(data)
                times=[]
                for _ in range(5):
                    flush.zero_();start.record();graph.replay();end.record();end.synchronize()
                    times.append(start.elapsed_time(end)*1000)
                timings.append((statistics.median(times),variant))
                del graph,out
            timings.sort(key=lambda r:r[0])
            _XPLAN[(batch,bool(window))]=timings[0][1]
            print('XQA_DIAG',batch,window,'us',[(round(us,2),v) for us,v in timings],flush=True)
            print('XQA_PLAN',batch,window,timings[0][1],flush=True)
            del data
    torch.cuda.synchronize()

def prepare():
    global _PREPARED
    if not _PREPARED:
        _diagnose(torch.cuda.get_device_capability()[0]==9)
        _PREPARED=True

if torch.cuda.is_available():
    prepare()
'''
(root/'cand_xqa_diag.py').write_text(base)
