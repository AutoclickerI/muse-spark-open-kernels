import torch
import triton
import triton.language as tl

glue_HIDDEN = 8192
glue_D_EXPERT = 4096
glue_N_EXPERTS = 256
glue_TOP_K = 8
glue_LOCAL_EXPERTS = 32
glue_RMS_EPS = 1e-05
glue_POST_EPS = 1e-08
glue_ROUTER_MULT = 0.1767766952966369
glue_PDL = True

@triton.jit
def glue_pdl_start(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_launch_dependents()

@triton.jit
def glue_pdl_wait(USE: tl.constexpr):
    if USE:
        tl.extra.cuda.gdc_wait()

def glue_launch(kernel, grid, *args, use_pdl=None, **kw):
    enabled = glue_PDL if use_pdl is None else bool(use_pdl)
    if enabled:
        kw['launch_pdl'] = True
    return kernel[grid](*args, USE_PDL=enabled, **kw)

@triton.jit
def glue_group_barrier(CNT, n):
    """Spin until n programs have arrived at counter CNT (zeroed per call; the n programs must be
    co-resident, which holds for the small grids that use it)."""
    tl.atomic_add(CNT, 1, sem='release')
    while tl.atomic_add(CNT, 0, sem='acquire') < n:
        pass

@triton.jit
def glue_boundary_kernel(BR, S: tl.constexpr, bs_stride, br_stride, STREAM, STREAM_OUT, OUT, POSTW, ALPHA, BETA, PREW, SUMS, CNT, post_eps, pre_eps, NB: tl.constexpr, H: tl.constexpr, C: tl.constexpr, HAS_POSTW: tl.constexpr, USE_PDL: tl.constexpr):
    """Token t, chunk c of C. With C > 1 the chunks of a token meet at a barrier to share the four
    sums the two norms need: sum b^2, then sum (a s + b r_b)^2 = A + 2 r_b B + r_b^2 Cc."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    glue_pdl_start(USE_PDL)
    CH: tl.constexpr = H // C
    offs = c * CH + tl.arange(0, CH)
    alpha = tl.load(ALPHA + offs)
    beta = tl.load(BETA + offs)
    prew = tl.load(PREW + offs)
    if HAS_POSTW:
        postw = tl.load(POSTW + offs)
    glue_pdl_wait(USE_PDL)
    st = tl.load(STREAM + t * H + offs)
    br = tl.zeros((CH,), tl.float32)
    for s in tl.static_range(S):
        br += tl.load(BR + s * bs_stride + t * br_stride + offs, mask=offs < NB, other=0.0).to(tl.float32)
    if HAS_POSTW:
        br = br * postw
    if C == 1:
        br = br * tl.rsqrt(tl.sum(br * br, 0) / H + post_eps)
        st = alpha * st + beta * br
        rs = tl.rsqrt(tl.sum(st * st, 0) / H + pre_eps)
    else:
        a_s = alpha * st
        b_b = beta * br
        tl.atomic_add(SUMS + t * 4 + 0, tl.sum(br * br, 0))
        tl.atomic_add(SUMS + t * 4 + 1, tl.sum(a_s * a_s, 0))
        tl.atomic_add(SUMS + t * 4 + 2, tl.sum(a_s * b_b, 0))
        tl.atomic_add(SUMS + t * 4 + 3, tl.sum(b_b * b_b, 0))
        glue_group_barrier(CNT + t, C)
        sbb = tl.atomic_add(SUMS + t * 4 + 0, 0.0, sem='acquire')
        sss = tl.atomic_add(SUMS + t * 4 + 1, 0.0, sem='acquire')
        ssb = tl.atomic_add(SUMS + t * 4 + 2, 0.0, sem='acquire')
        sbb2 = tl.atomic_add(SUMS + t * 4 + 3, 0.0, sem='acquire')
        rb = tl.rsqrt(sbb / H + post_eps)
        st = a_s + b_b * rb
        rs = tl.rsqrt((sss + 2.0 * rb * ssb + rb * rb * sbb2) / H + pre_eps)
    tl.store(STREAM_OUT + t * H + offs, st)
    tl.store(OUT + t * H + offs, (st * rs * prew).to(tl.bfloat16))

def glue_chunks(tokens):
    return 16 if tokens <= 8 else 1

def glue_control_boundary(branch, stream, alpha, beta, prew, sync, postw=None, use_pdl=None):
    P, S = branch
    tokens = stream.shape[0]
    C = glue_chunks(tokens)
    s_out = torch.empty_like(stream)
    out = torch.empty((tokens, glue_HIDDEN), device=stream.device, dtype=torch.bfloat16)
    sums, cnt = sync
    glue_launch(glue_boundary_kernel, (tokens, C), P, S, P.stride(0), P.stride(1), stream, s_out, out, postw if postw is not None else alpha, alpha, beta, prew, sums, cnt, glue_POST_EPS, glue_RMS_EPS, NB=P.shape[2], H=glue_HIDDEN, C=C, HAS_POSTW=postw is not None, num_warps=8 if C == 1 else 4, use_pdl=use_pdl)
    return (out, s_out)

@triton.jit
def glue_route_norm_kernel(PR, S: tl.constexpr, ps_stride, pr_stride, BIAS, PREW, LSTART, X, IDS, WTS, LIDS, SUMS, CNT, mult, eps, D: tl.constexpr, NE: tl.constexpr, K: tl.constexpr, NLOCAL: tl.constexpr, C: tl.constexpr, USE_PDL: tl.constexpr):
    """Token t. Program 0: sigmoid scores from the hi+lo logits, biased top-8, renormalized
    weights, local ids (-1 elsewhere). Programs 1..C: rmsnorm(p) * (1 + w) over D / C each."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    glue_pdl_start(USE_PDL)
    if c == 0:
        e = tl.arange(0, NE)
        bias = tl.load(BIAS + e)
        lstart = tl.load(LSTART)
        glue_pdl_wait(USE_PDL)
        logit = tl.zeros((NE,), tl.float32)
        for s in tl.static_range(S):
            row = PR + s * ps_stride + t * pr_stride
            logit += tl.load(row + D + e).to(tl.float32) + tl.load(row + D + NE + e).to(tl.float32)
        score = tl.sigmoid(logit * mult)
        biased = score + bias
        ids = tl.zeros((K,), tl.int32)
        sk = tl.zeros((K,), tl.float32)
        kr = tl.arange(0, K)
        for k in tl.static_range(K):
            idx = tl.argmax(biased, 0)
            sel = e == idx
            ids = tl.where(kr == k, idx, ids)
            sk = tl.where(kr == k, tl.sum(tl.where(sel, score, 0.0), 0), sk)
            biased = tl.where(sel, -float('inf'), biased)
        tl.store(IDS + t * K + kr, ids)
        tl.store(WTS + t * K + kr, sk / (tl.sum(sk, 0) + 1e-15))
        lid = (ids - lstart + NE) % NE
        tl.store(LIDS + t * K + kr, tl.where(lid < NLOCAL, lid, -1))
    else:
        CH: tl.constexpr = D // C
        offs = (c - 1) * CH + tl.arange(0, CH)
        prew = tl.load(PREW + offs)
        glue_pdl_wait(USE_PDL)
        p = tl.zeros((CH,), tl.float32)
        for s in tl.static_range(S):
            p += tl.load(PR + s * ps_stride + t * pr_stride + offs).to(tl.float32)
        if C == 1:
            ss = tl.sum(p * p, 0)
        else:
            tl.atomic_add(SUMS + t, tl.sum(p * p, 0))
            glue_group_barrier(CNT + t, C)
            ss = tl.atomic_add(SUMS + t, 0.0, sem='acquire')
        tl.store(X + t * D + offs, (p * tl.rsqrt(ss / D + eps) * prew).to(tl.bfloat16))

def glue_control_route_and_norm(pr, state, data, sync):
    P, S = pr
    tokens = P.shape[1]
    C = glue_chunks(tokens)
    dev = P.device
    x = torch.empty((tokens, glue_D_EXPERT), device=dev, dtype=torch.bfloat16)
    ids = torch.empty((tokens, glue_TOP_K), device=dev, dtype=torch.int32)
    wts = torch.empty((tokens, glue_TOP_K), device=dev, dtype=torch.float32)
    lids = torch.empty((tokens, glue_TOP_K), device=dev, dtype=torch.int32)
    sums, cnt = sync
    glue_launch(glue_route_norm_kernel, (tokens, 1 + C), P, S, P.stride(0), P.stride(1), state['router_bias'], state['pre_expert_w'], data['local_start'], x, ids, wts, lids, sums, cnt, glue_ROUTER_MULT, glue_RMS_EPS, D=glue_D_EXPERT, NE=glue_N_EXPERTS, K=glue_TOP_K, NLOCAL=glue_LOCAL_EXPERTS, C=C, num_warps=4 if C > 1 else 8)
    return (x, ids, wts, lids)

@triton.jit
def glue_boundary_fast(BR, S: tl.constexpr, BS: tl.constexpr, RS: tl.constexpr, STREAM, SOUT, OUT, POSTW, ALPHA, BETA, PREW, NB: tl.constexpr, HAS_POSTW: tl.constexpr, REPLICAS: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    part = tl.program_id(1)
    glue_pdl_start(USE_PDL)
    j = tl.arange(0, 8192)
    alpha = tl.load(ALPHA + j)
    beta = tl.load(BETA + j)
    prew = tl.load(PREW + j)
    if HAS_POSTW:
        postw = tl.load(POSTW + j)
    glue_pdl_wait(USE_PDL)
    st = tl.load(STREAM + t * 8192 + j)
    br = tl.full((8192,), 0.0, tl.float32)
    for s in tl.static_range(S):
        br += tl.load(BR + s * BS + t * RS + j, j < NB, 0.0).to(tl.float32)
    if HAS_POSTW:
        br *= postw
    br *= tl.rsqrt(tl.sum(br * br, 0) / 8192 + 1e-08)
    st = alpha * st + beta * br
    inv = tl.rsqrt(tl.sum(st * st, 0) / 8192 + 1e-05)
    mask = j // (8192 // REPLICAS) == part
    tl.store(SOUT + t * 8192 + j, st, mask)
    tl.store(OUT + t * 8192 + j, (st * inv * prew).to(tl.bfloat16), mask)

@triton.jit
def glue_route_fast(PR, S: tl.constexpr, PS: tl.constexpr, RS: tl.constexpr, BIAS, PREW, LSTART, X, IDS, WTS, LIDS, REPLICAS: tl.constexpr, USE_PDL: tl.constexpr):
    t = tl.program_id(0)
    part = tl.program_id(1)
    glue_pdl_start(USE_PDL)
    d = tl.arange(0, 4096)
    e = tl.arange(0, 256)
    prew = tl.load(PREW + d)
    bias = tl.load(BIAS + e)
    start = tl.load(LSTART)
    glue_pdl_wait(USE_PDL)
    p = tl.full((4096,), 0.0, tl.float32)
    logit = tl.full((256,), 0.0, tl.float32)
    for s in tl.static_range(S):
        row = PR + s * PS + t * RS
        p += tl.load(row + d).to(tl.float32)
        logit += tl.load(row + 4096 + e).to(tl.float32) + tl.load(row + 4352 + e).to(tl.float32)
    norm = p * tl.rsqrt(tl.sum(p * p, 0) / 4096 + 1e-05) * prew
    tl.store(X + t * 4096 + d, norm.to(tl.bfloat16), d // (4096 // REPLICAS) == part)
    if part == 0:
        scores = tl.sigmoid(logit * 0.1767766952966369)
        biased = scores + bias
        bits = biased.to(tl.int32, bitcast=True)
        ordered = tl.where(bits < 0, ~bits, bits ^ -2147483648).to(tl.uint32)
        keys = ordered.to(tl.uint64) << 32 | (255 - e).to(tl.uint64)
        winners = tl.topk(keys, 8)
        ids = 255 - (winners & 4294967295).to(tl.int32)
        scores = tl.gather(scores, ids, 0)
        k = tl.arange(0, 8)
        tl.store(IDS + t * 8 + k, ids)
        tl.store(WTS + t * 8 + k, scores / (tl.sum(scores, 0) + 1e-15))
        local = (ids - start + 256) % 256
        tl.store(LIDS + t * 8 + k, tl.where(local < 32, local, -1))

def glue_boundary(branch, stream, alpha, beta, prew, sync=None, postw=None, use_pdl=None):
    p, s = branch
    t = stream.shape[0]
    out = torch.empty((t, 8192), device=stream.device, dtype=torch.bfloat16)
    sout = torch.empty_like(stream)
    warps, replicas = glue_GLUE_BOUNDARY.get((t, postw is not None), (8, 1))
    glue_launch(glue_boundary_fast, (t, replicas), p, s, p.stride(0), p.stride(1), stream, sout, out, postw if postw is not None else alpha, alpha, beta, prew, NB=p.shape[2], HAS_POSTW=postw is not None, REPLICAS=replicas, num_warps=warps, use_pdl=True)
    return (out, sout)

def glue_route_and_norm(pr, state, data, sync=None):
    p, s = pr
    t = p.shape[1]
    dev = p.device
    x = torch.empty((t, 4096), device=dev, dtype=torch.bfloat16)
    ids = torch.empty((t, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((t, 8), device=dev, dtype=torch.float32)
    lids = torch.empty_like(ids)
    warps, replicas = glue_GLUE_ROUTE.get(t, (4, 1))
    glue_launch(glue_route_fast, (t, replicas), p, s, p.stride(0), p.stride(1), state['router_bias'], state['pre_expert_w'], data['local_start'], x, ids, wts, lids, REPLICAS=replicas, num_warps=warps, use_pdl=True)
    return (x, ids, wts, lids)
glue_GLUE_BOUNDARY = {}
glue_GLUE_ROUTE = {}

def lab(deadline,weights):
    import time,statistics,json
    dev=next(iter(weights.values())).device
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    def capture(fn):
        fn();fn();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        g.replay();torch.cuda.synchronize()
        return g,out
    def clock(g):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        flush.zero_();a.record();g.replay();b.record();b.synchronize()
        return a.elapsed_time(b)*1000
    def pair(a,b,n=10):
        ratios=[];aa=[];bb=[]
        for i in range(n):
            if i%2:x=clock(a);y=clock(b)
            else:y=clock(b);x=clock(a)
            aa.append(x);bb.append(y);ratios.append(y/x)
        return [round(statistics.median(aa),3),round(statistics.median(bb),3),round(statistics.median(ratios),4)]
    rows=[]
    for t in (1,8,16,32,128,256):
        torch.manual_seed(773+t)
        sync=torch.zeros(15*t,device=dev,dtype=torch.int32)
        si=sync[:3*t].view(3,t);sf=sync[3*t:].view(torch.float32).view(3,t,4)
        for phase in (1,2):
            if time.monotonic()>deadline-8:raise RuntimeError('glue lab deadline before all shapes')
            nb=1024 if t<=8 and phase==2 else 8192
            splits=(2 if phase==1 else 8) if t<=32 else 1
            dtype=torch.float32 if splits>1 else torch.bfloat16
            p=torch.randn((splits,t,nb),device=dev,dtype=dtype)
            stream=torch.randn((t,8192),device=dev)
            a=torch.rand(8192,device=dev);b=torch.rand_like(a);prew=torch.rand_like(a)+.5
            pw=torch.rand_like(a)+.5 if phase==2 else None
            def control():
                sync.zero_()
                return glue_control_boundary((p,splits),stream,a,b,prew,(sf[0],si[0]),postw=pw,use_pdl=True)
            cg,ref=capture(control)
            trials=[]
            for nw in (4,8,16):
                for replicas in ((1,4,16) if t<=8 else (1,)):
                    glue_GLUE_BOUNDARY[t,phase==2]=(nw,replicas)
                    g,out=capture(lambda:glue_boundary((p,splits),stream,a,b,prew,postw=pw))
                    # Changed values after capture, then independent FP32 oracle.
                    errs=[]
                    for seed in range(3):
                        torch.manual_seed(414+t+phase+seed*7919)
                        p.normal_();stream.normal_();g.replay();torch.cuda.synchronize()
                        br=torch.nn.functional.pad(p.float().sum(0),(0,8192-nb))
                        if pw is not None:br=br*pw
                        st=a*stream+b*br*torch.rsqrt(br.square().mean(-1,keepdim=True)+1.e-8)
                        want=st*torch.rsqrt(st.square().mean(-1,keepdim=True)+1.e-5)*prew
                        err=float((out[1]-st).norm()/st.norm());errs.append(err)
                        torch.testing.assert_close(out[1],st,atol=3.e-5,rtol=3.e-5)
                        torch.testing.assert_close(out[0].float(),want.bfloat16().float(),atol=.004,rtol=.008)
                    times=pair(cg,g);trials.append([nw,replicas,*times,round(max(errs),8)])
            rows.append('GLUE_BOUNDARY '+json.dumps(dict(t=t,phase=phase,s=splits,trials=trials),separators=(',',':')))
        splits=4 if t<=32 else 1
        p=torch.randn((splits,t,4608),device=dev,dtype=torch.float32)
        state=dict(router_bias=torch.randn(256,device=dev)*.1,pre_expert_w=torch.rand(4096,device=dev)+.5)
        data=dict(local_start=torch.tensor(243,device=dev,dtype=torch.int32))
        def control_route():
            sync.zero_();return glue_control_route_and_norm((p,splits),state,data,(sf[1],si[1]))
        cg,ref=capture(control_route);trials=[]
        for nw in (4,8,16):
            glue_GLUE_ROUTE[t]=(nw,1)
            g,out=capture(lambda:glue_route_and_norm((p,splits),state,data))
            for seed in range(3):
                torch.manual_seed(15013+t+seed*7919);p.normal_();state['router_bias'].normal_(0,.1)
                g.replay();cg.replay();torch.cuda.synchronize()
                scores=(p[:,:,4096:4352]+p[:,:,4352:4608]).sum(0).mul(.1767766952966369).sigmoid()
                ids=(scores+state['router_bias']).topk(8,dim=-1).indices
                torch.testing.assert_close(out[1].long(),ids,atol=0,rtol=0)
                selected=scores.gather(1,ids);wt=selected/(selected.sum(-1,keepdim=True)+1.e-15)
                torch.testing.assert_close(out[2],wt,atol=1.e-6,rtol=1.e-5)
                br=p[:,:,:4096].sum(0);want=br*torch.rsqrt(br.square().mean(-1,keepdim=True)+1.e-5)*state['pre_expert_w']
                torch.testing.assert_close(out[0].float(),want.bfloat16().float(),atol=.004,rtol=.008)
                torch.testing.assert_close(out[3],ref[3],atol=0,rtol=0)
            times=pair(cg,g);trials.append([nw,*times])
        rows.append('GLUE_ROUTE '+json.dumps(dict(t=t,s=splits,trials=trials),separators=(',',':')))
    return rows
