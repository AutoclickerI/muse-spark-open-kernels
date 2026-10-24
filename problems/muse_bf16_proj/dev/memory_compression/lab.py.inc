import gc
import linecache
import statistics
import time
import triton
import triton.language as tl

MC_ROWS=[]
MC_SUPPORT={}
MC_SEED=725105
MC_BYTES=256<<20


class MCSpace:
    def __init__(self,ns):object.__setattr__(self,'ns',ns)
    def __getattr__(self,name):
        try:return self.ns[name]
        except KeyError:raise AttributeError(name) from None
    def __setattr__(self,name,value):self.ns[name]=value


def mc_module():
    filename='<memory_compression_best7_scale>'
    linecache.cache[filename]=(len(MC_SOURCE),None,MC_SOURCE.splitlines(True),filename)
    ns={'__name__':__name__,'__builtins__':__builtins__}
    exec(compile(MC_SOURCE,filename,'exec'),ns)
    return MCSpace(ns)


def mc_equal(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def mc_graph(fn):
    for _ in range(2):fn()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    graph.replay();torch.cuda.synchronize()
    return graph,out


def mc_pair(a,b,flush,rounds=20,refresh=None,check=None):
    ea=torch.cuda.Event(enable_timing=True);eb=torch.cuda.Event(enable_timing=True)
    values=[[],[]];bad=0
    for rep in range(rounds):
        if refresh is not None:refresh(rep)
        for side in ((0,1) if rep%2==0 else (1,0)):
            flush.add_(1);ea.record();(a if side==0 else b).replay();eb.record();eb.synchronize()
            values[side].append(ea.elapsed_time(eb)*1000.)
        if check is not None and not check():bad+=1
    ratios=[b/a for a,b in zip(*values)]
    return dict(normal_us=statistics.mean(values[0]),compressed_us=statistics.mean(values[1]),
        time_ratio=statistics.mean(ratios),bandwidth_ratio=statistics.mean(1/x for x in ratios),
        bit_mismatches=bad,rounds=rounds,seed_ratios=[statistics.mean(ratios[s::3]) for s in range(3)] if refresh else [],samples=values)


def mc_recoded_bank(m,weights,mod):
    device=weights['w13'].device
    metadata=dict(gs13=weights['w13_gs'][:32].contiguous(),gs2=weights['w2_gs'][:32].contiguous(),
        pnw=weights['post_norm_w'],cnt=torch.zeros(64,device=device,dtype=torch.int32),
        grid=torch.cuda.get_device_properties(device).multi_processor_count)
    return m._prepare_recoded_bank(weights,metadata,mod)


def mc_control_fill(dst,kind,weights,bank):
    n=dst.numel();device=dst.device
    gen=torch.Generator(device=device).manual_seed(MC_SEED)
    if kind=='zero':dst.zero_();return
    if kind=='pattern':
        dst.view(torch.int32).fill_(0x3f800000);return
    if kind=='random':
        src=torch.randint(0,256,(n,),device=device,dtype=torch.uint8,generator=gen)
        mc_copy_tensor(src,dst);return
    if kind=='bf16':
        src=torch.empty(n//2,device=device,dtype=torch.bfloat16).normal_(0.,.02,generator=gen)
        mc_copy_tensor(src,dst);return
    if kind=='raw_nvfp4':
        sources=[weights['w13'][:32],weights['w13_sf'][:32],weights['w2'][:32],weights['w2_sf'][:32]]
        shares=(16,2,8,1)
    elif kind=='recoded_nvfp4':sources=[bank['w13p'],bank['w2p']];shares=(2,1)
    else:raise ValueError(kind)
    at=0
    for i,(src,share) in enumerate(zip(sources,shares)):
        take=((n*share//sum(shares))//16)*16 if i+1<len(sources) else n-at
        raw=src.contiguous().view(torch.uint8).reshape(-1)
        if take>raw.numel():raise ValueError('host expert bank too small for control')
        mc_copy_tensor(raw[:take],dst[at:at+take]);at+=take
    assert at==n


def mc_controls(normal,compressed,weights,bank,flush,deadline,lines):
    device=normal.device;sms=torch.cuda.get_device_properties(device).multi_processor_count
    configs=[(sms,8),(sms*4,8),(sms*8,16)]
    sinks=[torch.empty(sms*8*512,device=device,dtype=torch.int32) for _ in range(3)]
    for kind in ('zero','pattern','random','raw_nvfp4','bf16','recoded_nvfp4'):
        if time.monotonic()>deadline-3:raise TimeoutError('control coverage incomplete')
        mc_control_fill(normal,kind,weights,bank)
        # A real SM store writes every byte to the compressible allocation.
        mc_copy_tensor(normal,compressed);torch.cuda.synchronize()
        if not mc_equal(normal,compressed):raise RuntimeError('allocation byte mismatch '+kind)
        trials=[]
        for ctas,depth in configs:
            a,ao=mc_graph(lambda:mc_read_tensor(normal,sinks[0],ctas,depth))
            aa,aao=mc_graph(lambda:mc_read_tensor(normal,sinks[1],ctas,depth))
            b,bo=mc_graph(lambda:mc_read_tensor(compressed,sinks[2],ctas,depth))
            noise=mc_pair(a,aa,flush,12,check=lambda:mc_equal(ao,aao))
            pair=mc_pair(a,b,flush,24,check=lambda:mc_equal(ao,bo))
            if noise['bit_mismatches'] or pair['bit_mismatches']:raise RuntimeError('stream sink mismatch '+kind)
            trials.append(dict(ctas=ctas,depth=depth,aa=noise,**pair))
            a.reset();aa.reset();b.reset()
        best=min(trials,key=lambda r:r['normal_us'])
        row=dict(kind='bandwidth',data=kind,bytes=normal.nbytes,trials=trials,selected=best);MC_ROWS.append(row)
        lines.append(f'BW {kind} N/C={best["normal_us"]:.2f}/{best["compressed_us"]:.2f}us '+
            f'GB/s={normal.nbytes/best["normal_us"]/1000:.0f}/{normal.nbytes/best["compressed_us"]/1000:.0f} '+
            f'gain={best["bandwidth_ratio"]:.4f} aa={best["aa"]["time_ratio"]:.4f} '+
            f'ct/d={best["ctas"]}/{best["depth"]} bit=1 all='+','.join(f'{r["bandwidth_ratio"]:.3f}' for r in trials))


def mc_mapped_bank(bank,compression):
    keys=('w13p','w2p','gs13','gs2','pnw');offsets={};size=0
    for key in keys:
        size=(size+255)//256*256;offsets[key]=size;size+=bank[key].nbytes
    owner=MappedAllocation((size+15)//16*16,compression)
    out=dict(bank,cnt=torch.zeros_like(bank['cnt']))
    try:
        for key in keys:
            out[key]=owner.tensor(tuple(bank[key].shape),bank[key].dtype,offsets[key])
            mc_copy_tensor(bank[key],out[key])
        torch.cuda.synchronize()
        for key in keys:
            if not mc_equal(out[key],bank[key]):raise RuntimeError('packed bank copy mismatch '+key)
        return out,owner
    except Exception:
        torch.cuda.synchronize();owner.close();raise


def mc_expert_input(tokens,seed,device):
    gen=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn((tokens,4096),device=device,generator=gen).bfloat16()
    z=torch.randn(256,device=device,generator=gen)
    u=torch.rand(256,device=device,generator=gen).clamp_min(1.e-12)
    scores=torch.rand((tokens,256),device=device,generator=gen)+.03*z/(-u.log()).sqrt()
    chosen=scores.topk(8,-1);ids=chosen.indices
    hot=torch.bincount((ids//32).reshape(-1),minlength=8).argmax()
    ids=((ids-32*hot)%256).to(torch.int32)
    return dict(hidden_states=x,topk_ids=ids,topk_weights=(chosen.values/chosen.values.sum(-1,keepdim=True)).float(),
        config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0))


@triton.jit
def mc_ordered_finish(Y,IDS,WTS,PNW,OUT,NP:tl.constexpr,PAIRS:tl.constexpr):
    t=tl.program_id(0);c=tl.arange(0,4096);f=1.+tl.load(PNW+c).to(tl.float32)
    p=tl.arange(0,PAIRS);allids=tl.load(IDS+p,p<NP,256)
    acc=tl.zeros((4096,),tl.float32)
    for slot in range(8):
        flat=t*8+slot;e=tl.load(IDS+flat)
        pos=tl.sum(((p<NP)&(allids>=0)&(allids<e)).to(tl.int32),0)+\
            tl.sum(((p<flat)&(allids==e)).to(tl.int32),0)
        valid=(e>=0)&(e<32)
        value=tl.load(Y+pos*4096+c,valid,0.)
        scaled=value*f
        norm=tl.sum(scaled*scaled,0)/4096.+1.e-8
        weight=tl.load(WTS+flat)
        acc+=weight*tl.rsqrt(norm)*scaled
    tl.store(OUT+t*4096+c,acc.to(tl.bfloat16))


def mc_ordered_call(m,data,bank):
    # The exact incumbent module still produces Y and its ordinary output.
    # Only this untimed verification control uses an ordered norm/combine.
    x=data['hidden_states'];ids=data['topk_ids'];wts=data['topk_weights'];tokens=x.shape[0];mod=bank['mod']
    stages=2
    while stages>1 and mod.smem_needed(1,stages,tokens)>mod.smem_optin:stages-=1
    inter=torch.empty((tokens*8,4096),device=x.device,dtype=torch.bfloat16)
    y=torch.empty((tokens*8,4096),device=x.device,dtype=torch.float32)
    ss=torch.empty(tokens*8,device=x.device,dtype=torch.float32)
    out=torch.empty((tokens,4096),device=x.device,dtype=torch.bfloat16);ordered=torch.empty_like(out)
    mod.run(x,ids,wts,bank['pnw'],bank['w13p'],bank['w2p'],bank['gs13'],bank['gs2'],inter,y,ss,
        bank['cnt'],out,0,1,stages,bank['grid']*mod.resident_blocks(1,stages,tokens))
    mc_ordered_finish[(tokens,)](y,ids,wts,bank['pnw'],ordered,NP=tokens*8,
        PAIRS=triton.next_power_of_2(tokens*8),num_warps=8)
    return out,ordered,y


def mc_verify_ordered(m,data,normal,compressed,banks):
    a,ao=mc_graph(lambda:mc_ordered_call(m,data,normal))
    b,bo=mc_graph(lambda:mc_ordered_call(m,data,compressed));worst=0.
    try:
        for rep in range(12):
            for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(banks[rep%3][key])
            a.replay();b.replay();torch.cuda.synchronize()
            # Y rows without a local pair are intentionally unwritten. Compare
            # only the exact routed prefix, computed from the supplied IDs.
            local=int(((data['topk_ids']>=0)&(data['topk_ids']<32)).sum())
            if not mc_equal(ao[2][:local],bo[2][:local]):raise RuntimeError('MoE W2 values differ across allocations')
            if not mc_equal(ao[1],bo[1]):raise RuntimeError('ordered MoE outputs differ across allocations')
            for native,ordered,_ in (ao,bo):
                error=float((native.float()-ordered.float()).norm()/native.float().norm().clamp_min(1.e-20))
                worst=max(worst,error)
                if not bool(torch.isfinite(ordered).all()) or error>.001:raise RuntimeError('ordered norm changed MoE math '+str(error))
        return dict(bit_identical=True,y_bit_identical=True,checks=12,max_native_relative=worst)
    finally:a.reset();b.reset()


def mc_moe(m,bank,compression,flush,deadline,lines):
    normal,an=mc_mapped_bank(bank,0);compressed,ac=mc_mapped_bank(bank,compression)
    try:
        if an.granted!=0 or ac.granted!=compression:raise RuntimeError('MoE bank grant differs from requested arms')
        lines.append(f'BANK recoded weights={ac.logical_size} mapped={ac.size} N/Cgrant={an.granted}/{ac.granted}; all w13p/w2p/global scales/postnorm byte-identical; counters normal')
        for tokens in (1,8):
            if time.monotonic()>deadline-3:raise TimeoutError('MoE coverage incomplete')
            banks=[mc_expert_input(tokens,MC_SEED+seed,flush.device) for seed in (11,122,233)]
            data={k:v.clone() if isinstance(v,torch.Tensor) else dict(v) for k,v in banks[0].items()}
            def refresh(rep):
                for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(banks[rep%3][key])
            a,ao=mc_graph(lambda:m.v9_kernel_fn(data,normal))
            aa,aao=mc_graph(lambda:m.v9_kernel_fn(data,normal))
            b,bo=mc_graph(lambda:m.v9_kernel_fn(data,compressed))
            noise=mc_pair(a,aa,flush,24,refresh,lambda:mc_equal(ao,aao))
            pair=mc_pair(a,b,flush,36,refresh,lambda:mc_equal(ao,bo))
            exact=noise['bit_mismatches']==0 and pair['bit_mismatches']==0
            ordered=mc_verify_ordered(m,data,normal,compressed,banks)
            MC_ROWS.append(dict(kind='moe',tokens=tokens,aa=noise,bit_identical=exact,ordered=ordered,**pair))
            lines.append(f'MOE NT1recoded M={tokens} N/C={pair["normal_us"]:.2f}/{pair["compressed_us"]:.2f}us r={pair["time_ratio"]:.4f} '+
                f'aa={noise["time_ratio"]:.4f} bit={int(exact)} aa_bad={noise["bit_mismatches"]}/24 ab_bad={pair["bit_mismatches"]}/36 '+
                's='+','.join(f'{r:.3f}' for r in pair['seed_ratios'])+
                f' Ybit=1 ordered_bit=1 native_rel={ordered["max_native_relative"]:.3g}')
            a.reset();aa.reset();b.reset()
        if not all(mc_equal(normal[k],compressed[k]) for k in ('w13p','w2p','gs13','gs2','pnw')):
            raise RuntimeError('weight bytes changed during read-only MoE replay')
    finally:
        torch.cuda.synchronize();an.close();ac.close()


def mc_dense(m,compression,flush,deadline,lines):
    device=flush.device;gen=torch.Generator(device=device).manual_seed(MC_SEED)
    for role,n,k in (('qkv_gate',2304,8192),('o',8192,1024)):
        if time.monotonic()>deadline-3:raise TimeoutError('dense coverage incomplete')
        w=torch.empty((n,k),device=device,dtype=torch.bfloat16).normal_(0.,.02,generator=gen)
        an=MappedAllocation(w.nbytes,0);ac=MappedAllocation(w.nbytes,compression)
        try:
            if an.granted!=0 or ac.granted!=compression:raise RuntimeError('dense bank grant differs from requested arms')
            wn=an.tensor((n,k),torch.bfloat16);wc=ac.tensor((n,k),torch.bfloat16)
            mc_copy_tensor(w,wn);mc_copy_tensor(w,wc);torch.cuda.synchronize()
            if not mc_equal(wn,wc) or not mc_equal(w,wn):raise RuntimeError('dense bank copy mismatch')
            inputs=[torch.randn((1,k),device=device,dtype=torch.bfloat16,generator=gen) for _ in range(3)];x=inputs[0].clone()
            plan=dict(projection={});state=dict(proj_plan={(1,role):'cublas'})
            normal=dict(state,**{role:wn});compressed=dict(state,**{role:wc})
            a,ao=mc_graph(lambda:m._best7_project(x,role,normal,plan)[0])
            aa,aao=mc_graph(lambda:m._best7_project(x,role,normal,plan)[0])
            b,bo=mc_graph(lambda:m._best7_project(x,role,compressed,plan)[0])
            refresh=lambda rep:x.copy_(inputs[rep%3])
            noise=mc_pair(a,aa,flush,24,refresh,lambda:mc_equal(ao,aao))
            pair=mc_pair(a,b,flush,36,refresh,lambda:mc_equal(ao,bo))
            exact=noise['bit_mismatches']==0 and pair['bit_mismatches']==0
            MC_ROWS.append(dict(kind='dense',role=role,tokens=1,aa=noise,bit_identical=exact,**pair))
            lines.append(f'DENSE {role} M=1 choice=cublas N/C={pair["normal_us"]:.2f}/{pair["compressed_us"]:.2f}us r={pair["time_ratio"]:.4f} '+
                f'grant={an.granted}/{ac.granted} aa={noise["time_ratio"]:.4f} bit={int(exact)} aa_bad={noise["bit_mismatches"]}/24 ab_bad={pair["bit_mismatches"]}/36 '+
                's='+','.join(f'{r:.3f}' for r in pair['seed_ratios']))
            if not exact:raise RuntimeError('dense outputs not bit-identical')
            a.reset();aa.reset();b.reset()
        finally:
            torch.cuda.synchronize();an.close();ac.close()


def lab(deadline,weights,host_state=None):
    global MC_ROWS,MC_SUPPORT
    MC_ROWS=[];started=time.monotonic();lines=[
        'META generic driver compression vs normal VMM; bytes identical; 256MiB cold flush outside events; raw graph us; BW logical GB/s not physical storage ratio; best7_scale='+_mchash.sha256(MC_SOURCE.encode()).hexdigest()[:16],
        'META seed725105; controls256MiB zero/FP32one pattern/random/raw real NVFP4 code+scale/BF16 N(0,.02)/recoded bank; NT1recoded M1/8 and dense Q/O M1; 24AA+36AB/BA same3 inputs/routes; incumbent norm atomics can differ even AA: report raw bit failures; separate untimed W2/ordered-norm bit-exact control; no host mutation']
    owners=[]
    try:
        MC_SUPPORT=mc_probe()
        lines.append('SUPPORT '+str(MC_SUPPORT))
        if MC_SUPPORT['create_status']!=0 or MC_SUPPORT['granted']!=1:
            lines.append('RESULT compression not granted; ratio/kernel comparisons unavailable; no inference about real-weight compressibility')
            return lines
        from concurrent.futures import ThreadPoolExecutor
        m=mc_module()
        with ThreadPoolExecutor(max_workers=2) as pool:
            native=pool.submit(mc_library);recoded=pool.submit(m._v9_recoded_module)
            native.result();mod=recoded.result()
        bank=mc_recoded_bank(m,weights,mod)
        normal=MappedAllocation(MC_BYTES,0);owners.append(normal)
        compressed=MappedAllocation(MC_BYTES,1);owners.append(compressed)
        lines.append(f'CONTROL_ALLOC N/Cgrant={normal.granted}/{compressed.granted} gran={normal.granularity}/{compressed.granularity}')
        if compressed.granted!=1:raise RuntimeError('large allocation compression not granted')
        a=normal.tensor((MC_BYTES,));b=compressed.tensor((MC_BYTES,))
        flush=torch.zeros(256<<20,device=weights['w13'].device,dtype=torch.uint8)
        mc_controls(a,b,weights,bank,flush,deadline,lines)
        torch.cuda.synchronize();normal.close();compressed.close();owners=[];a=b=None
        mc_moe(m,bank,1,flush,deadline,lines)
        mc_dense(m,1,flush,deadline,lines)
        zero=next(r['selected']['bandwidth_ratio'] for r in MC_ROWS if r.get('data')=='zero')
        bits=all(r.get('bit_identical',True) for r in MC_ROWS)
        lines.append(f'RESULT zero_BW_gain={zero:.4f} kernel_bit_exact={int(bits)}; zero null is inconclusive on activation, not evidence real weights are incompressible')
        lines.append(f'END controls=6 kernels=4 elapsed={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
        return lines
    except Exception as exc:
        import traceback
        lines.append('MC_ERROR '+type(exc).__name__+':'+str(exc)[:240])
        lines.extend(traceback.format_exc().splitlines()[-4:])
        lines.append(f'MC_INCOMPLETE rows={len(MC_ROWS)} elapsed={time.monotonic()-started:.1f}')
        return lines
    finally:
        torch.cuda.synchronize()
        for owner in owners:owner.close()
        gc.collect()
