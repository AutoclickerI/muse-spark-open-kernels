import statistics as vs_stats
import time as vs_time
VS_TRACE = False


class VSGraph:
    def __init__(self, fn):
        self.fn = fn
        # Keep a distinct native capture stream alive for each live graph. Reusing
        # a pooled PyTorch stream can retire another graph's cuBLAS workspace.
        self.runtime = torch.cuda.cudart()
        handle = vs_ct.c_void_p()
        status = self.runtime.cudaStreamCreate(vs_ct.addressof(handle))
        if int(status):
            raise RuntimeError('capture stream creation failed')
        self.handle = handle.value
        self.stream = torch.cuda.ExternalStream(self.handle)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            fn()
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.out = fn()
        self.stream.synchronize()

    def close(self):
        if self.handle is not None:
            self.graph.reset()
            self.runtime.cudaStreamDestroy(self.handle)
            self.handle = None
            self.fn = None


def vs_error(got,want,fp32):
    got = got.float()
    if not bool(torch.isfinite(got).all()):
        raise ValueError('nonfinite projection')
    d = (got-want).norm(dim=1)
    denom = want.norm(dim=1).clamp_min(want.norm(dim=1).mean()*.1).clamp_min(1e-12)
    rel = float((got-want).norm()/want.norm().clamp_min(1e-12))
    row = float((d/denom).max())
    lim = 1e-4 if fp32 else .01
    if rel > lim or row > lim*5:
        raise ValueError(f'projection accuracy rel={rel:.4g} row={row:.4g}')
    return rel,row


def vs_control_library():
    arch = '9.0a' if torch.cuda.get_device_capability()[0] == 9 else '12.0'
    old = vs_os.environ.get('TORCH_CUDA_ARCH_LIST')
    vs_os.environ['TORCH_CUDA_ARCH_LIST'] = arch
    try:
        path = vs_load_inline(name='vs_best6_lt_'+vs_hash.sha256(PG_RAW.encode()).hexdigest()[:12]+arch.replace('.','_'),
            cpp_sources='',cuda_sources=PG_RAW,functions=None,is_python_module=False,no_implicit_headers=True,
            extra_cuda_cflags=['-O3','-std=c++20'],extra_ldflags=['-lcublasLt'],verbose=False)
    finally:
        if old is None:
            vs_os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
        else:
            vs_os.environ['TORCH_CUDA_ARCH_LIST'] = old
    return _PgNative(path)


def vs_control(w,x,name,lt,work):
    m,k = x.shape
    n = w.shape[0]
    fp32 = name == 'pool_router'
    dtype = torch.float32 if fp32 else torch.bfloat16
    recipe = {(128,'qkv_gate'):(1,23),(128,'pool_router'):(0,0),(256,'qkv_gate'):(0,0)}.get((m,name))
    if recipe is not None:
        nn,ordinal = recipe
        codes,_ = lt.options(m,n,k,nn,fp32,work.numel())
        if ordinal >= len(codes):
            raise ValueError(f'best6 Lt ordinal absent {m} {name} {recipe}')
        bank = w.T.contiguous() if nn else w
        out = torch.empty((m,n),device=x.device,dtype=dtype)
        def call():
            lt.run(x,bank,out,work,codes,ordinal,nn,fp32)
            return out.unsqueeze(0),1
        return call,'best6_Lt'+str(recipe)
    if m == 8 and name in ('pool_router','unpooler_slice'):
        return lambda:gemv(x,w,dtype),'best6_split_partials'
    def cublas():
        out = torch.mm(x,w.T,out_dtype=torch.float32) if fp32 else F.linear(x,w)
        return out.unsqueeze(0),1
    return cublas,'best6_cublas'


def vs_fixture(x,index):
    x.normal_()
    if index == 1:
        x[::3].mul_(8)
        x[:,::127].mul_(16)
    if index == 2:
        x.zero_()
        rows = torch.arange(x.shape[0],device=x.device)
        cols = (rows*1031+17)%x.shape[1]
        x[rows,cols] = torch.exp2((rows%7-3).float()).to(x.dtype)


def vs_times(graphs,x,flush,repeats):
    samples = [[] for _ in graphs]
    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for repeat in range(repeats):
        vs_fixture(x,repeat%3)
        for i in [(repeat+j)%len(graphs) for j in range(len(graphs))]:
            flush.sum()
            a.record(); graphs[i].graph.replay(); b.record()
            try:
                b.synchronize()
            except Exception as exc:
                raise RuntimeError(f'timing graph={i} repeat={repeat}: {exc}') from exc
            samples[i].append(a.elapsed_time(b)*1000)
    return samples


def lab(deadline,weights,host_state=None):
    start = vs_time.monotonic()
    lines = [f'WEIGHT_WIDE best6_sha256={VS_BEST6_SHA256} native={torch.cuda.get_device_capability()==(9,0)}']
    lib = vs_library()
    lt = vs_control_library()
    device = weights['qkv_gate'].device
    router = weights['router'].float()
    hi = router.to(torch.bfloat16)
    lo = (router-hi.float()).to(torch.bfloat16)
    banks = {name:weights[name].contiguous() for name in ('qkv_gate','o','unpooler')}
    banks['pool_router'] = torch.cat((weights['pooler'],hi,lo)).contiguous()
    banks['unpooler_slice'] = banks['unpooler'][:1024]
    work = torch.empty(4<<30,device=device,dtype=torch.uint8)
    flush = torch.empty(256<<20,device=device,dtype=torch.uint8)
    graphs = []
    checked = rejected = 0
    # Main verify loads first: complete both before the M8 diagnostic.
    for m in (128,256,8):
        total_old = total_new = 0.
        for name in ('qkv_gate','o','pool_router','unpooler_slice' if m<=8 else 'unpooler'):
            if VS_TRACE:
                print('VS_BEGIN',m,name,flush=True)
            if vs_time.monotonic() > deadline-10:
                return lines + [f'INCOMPLETE before M={m} role={name} budget_s={deadline-start:.1f}']
            w = banks[name]; n,k = w.shape; fp32 = name=='pool_router'
            x = torch.randn((m,k),device=device,dtype=torch.bfloat16)
            wf = w.float()
            fixtures = []
            for j in range(3):
                vs_fixture(x,j)
                fixtures.append((x.clone(),x.float()@wf.T))
            control,label = vs_control(w,x,name,lt,work)
            base = VSGraph(control)
            if VS_TRACE:
                print('VS_BASE_GRAPH',m,name,label,flush=True)
            old_times = vs_times([base],x,flush,5)[0]
            coarse = []
            errors = []
            for code,splits in vs_choices(m,n,k):
                if VS_TRACE:
                    print('VS_TRIAL',m,name,code,splits,flush=True)
                if vs_time.monotonic() > deadline-12:
                    lines.append(f'INCOMPLETE sweep M={m} role={name}')
                    break
                op = None; graph = None
                try:
                    op = VerifyStreamingProjection(w,m,code,splits,fp32,lib=lib)
                    graph = VSGraph(lambda:op(x))
                    for fx,want in fixtures:
                        x.copy_(fx); graph.graph.replay(); torch.cuda.synchronize()
                        vs_error(graph.out,want,fp32)
                    if bool(op.counters.any()):
                        raise ValueError('split counters did not reset')
                    us = vs_stats.median(vs_times([graph],x,flush,3)[0])
                    coarse.append((us,code,splits))
                    checked += 1
                except Exception as exc:
                    if 'CUDA error' in str(exc):
                        raise RuntimeError(f'M={m} role={name} code={code} split={splits}: {exc}') from exc
                    rejected += 1
                    errors.append(f'{code}/{splits}:{type(exc).__name__}:{str(exc)[:100]}')
                finally:
                    if graph is not None:
                        graph.close()
            coarse.sort()
            if not coarse:
                base.close()
                lines.append(f'REJECT_ALL M={m} role={name} details={errors[:3]}')
                continue
            finalists = coarse[:2]
            if VS_TRACE:
                print('VS_FINALISTS',m,name,finalists,flush=True)
            ops = [VerifyStreamingProjection(w,m,c,s,fp32,lib=lib) for _,c,s in finalists]
            graphs = [base]+[VSGraph(lambda op=op:op(x)) for op in ops]
            samples = vs_times(graphs,x,flush,20)
            meds = [vs_stats.median(t) for t in samples]
            winner = min(range(1,len(meds)),key=lambda i:vs_stats.median([a/b for a,b in zip(samples[i],samples[0])]))
            op = ops[winner-1]
            rel = row = router_rel = 0.
            for fx,want in fixtures:
                x.copy_(fx)
                for graph in graphs:
                    graph.graph.replay()
                torch.cuda.synchronize()
                parts,count = base.out
                vs_error(parts.sum(0) if count>1 else parts[0],want,fp32)
                for graph in graphs[1:]:
                    rr,rw = vs_error(graph.out,want,fp32)
                    rel,row = max(rel,rr),max(row,rw)
                if fp32:
                    actual = op.out[:,4096:4352]+op.out[:,4352:]
                    true_router = fx.float()@router.T
                    router_rel = max(router_rel,float((actual-true_router).norm()/true_router.norm().clamp_min(1e-12)))
                    if router_rel>1e-4:
                        raise ValueError('FP32 router hi+lo exceeds 1e-4')
            ratio = vs_stats.median([a/b for a,b in zip(samples[winner],samples[0])])
            bm,bn,bk,st,pipe = VS_CONFIGS[op.code]
            weight_bytes = w.nbytes
            lines.append(f'M={m} role={name} control={label} old_us={meds[0]:.4f} new_us={meds[winner]:.4f} '
                f'ratio={ratio:.5f} code={op.code} split={op.splits} tile={bm}x{bn}x{bk} stages={st} '
                f'pipe={int(pipe)} CTAs={op.tiles*op.splits} TBps={weight_bytes/(meds[winner]*1e6):.4f} '
                f'rel={rel:.3g} row={row:.3g} router={router_rel:.3g} tried={len(coarse)} reject={len(errors)} '
                f'coarse_top={[(round(t,2),c,s) for t,c,s in finalists]}')
            if errors:
                lines.append(f'REJECTIONS M={m} role={name} {errors[:4]}')
            total_old += meds[0]; total_new += meds[winner]
            for graph in graphs:
                graph.close()
            graphs = []
        lines.append(f'TOTAL M={m} old_us={total_old:.4f} new_us={total_new:.4f} bytes={sum(banks[n].nbytes for n in ("qkv_gate","o","pool_router","unpooler_slice" if m<=8 else "unpooler"))}')
    lines.append(f'WEIGHT_WIDE_SUMMARY checked={checked} rejected={rejected} fresh=3 cold_rotated=20 elapsed_s={vs_time.monotonic()-start:.2f}')
    return lines
