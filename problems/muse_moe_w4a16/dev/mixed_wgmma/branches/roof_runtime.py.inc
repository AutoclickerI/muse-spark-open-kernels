ND_MODS=[]


def nd_build():
    if ND_MODS:return
    arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
    for i,cu in enumerate(ND_CUDA):
        mod=load_inline(name='nd_roof_'+hashlib.sha256(cu.encode()).hexdigest()[:12],
                        cpp_sources=ND_CPP,cuda_sources=cu,
                        functions=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run','nd_info'],
                        no_implicit_headers=True,with_pytorch_error_handling=False,
                        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
        mod.br_init();opt=mod.pmz_init();ND_MODS.append(mod)
        if i==0:NF_OPTS['pmz']=opt
    mod=load_inline(name='nd_stream_'+hashlib.sha256(ND_STREAM_CUDA.encode()).hexdigest()[:12],
                    cpp_sources='#include <pybind11/pybind11.h>\n#include <cstdint>\nvoid nd_read(uint64_t,uint64_t,uint64_t,uint64_t,int,int,uint64_t);',
                    cuda_sources=ND_STREAM_CUDA,functions=['nd_read'],no_implicit_headers=True,
                    with_pytorch_error_handling=False,
                    extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    ND_MODS.append(mod)


def nf_module():return ND_MODS[0]


def nd_prepare(w):
    bank=hh_prepare(w)
    bank['base'].update(br_ilv_bank(w))
    if ND_WIDE:bank['base'].update(br_wide(bank['base']))
    return bank


def nd_capture(fn):
    fn();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    return g


def nd_time(g,events):
    lab_cold_l2();events[0].record();g.replay();events[1].record();events[1].synchronize()
    return events[0].elapsed_time(events[1])*1000.


def nd_pair(a,b,events):
    aa=[];bb=[]
    for i in range(6):
        for g,dst in (((a,aa),(b,bb)) if i%2==0 else ((b,bb),(a,aa))):dst.append(nd_time(g,events))
    return statistics.median(aa),statistics.median(bb)


def nd_segments(plan,bank,branch,first,grid):
    h13,h2,nv,ids,tw,pos,tasks=plan['owners'];p=h13 if first else h2
    counts=h13[4].cpu().tolist();off=h13[5].cpu().tolist();slots=h13[8].cpu().tolist()
    seg=[];unique_weight=0;scheduled_weight=0;xbytes=0
    def add(ptr,n,repeat=1):
        nonlocal xbytes
        if repeat:seg.append((ptr,n,repeat))
    if branch=='heavy':
        for e,n in enumerate(counts):
            if n<=16:continue
            for key in ('w13','w13_sf') if first else ('w2','w2_sf'):
                t=bank['base'][key][e];nb=t.numel()*t.element_size();unique_weight+=nb
                for _ in range((n+31)//32):add(t.data_ptr(),nb);scheduled_weight+=nb
            # The original gathered/copied activation row repeats for every output tile.
            repeats=128 if first else 64
            for r in range(n):
                row=slots[e][r]//8 if first else off[e]+r
                add(p[3].data_ptr()+row*8192,8192,repeats);xbytes+=8192*repeats
    else:
        nblk=[(n+7)//8 if n<=16 else 0 for n in counts]
        ntile=512 if first else 256
        for e,nb in enumerate(nblk):
            if not nb:continue
            t=bank['native']['w13p' if first else 'w2p'][e];size=t.numel()*t.element_size();unique_weight+=size
            for _ in range(nb):add(t.data_ptr(),size);scheduled_weight+=size
        total=sum(nblk)*ntile;cc=min(ntile,max(1,(total+grid//2)//grid))
        ends=[0]
        for nb in nblk:ends.append(ends[-1]+nb*ntile)
        # Reproduce the actual persistent CTA partitions and activation cache refreshes.
        for cta in range(grid):
            last=None;e=0
            for u in range(total*cta//grid,total*(cta+1)//grid):
                while ends[e+1]<=u:e+=1
                rel=u-ends[e];nbk=nblk[e];span=cc*nbk;full=ntile//cc
                if rel<full*span:
                    mc=rel//span;rem=rel-mc*span;nb=rem//cc
                else:
                    tail=ntile-full*cc;nb=(rel-full*span)//tail
                if (e,nb)==last:continue
                last=(e,nb)
                for r in range(nb*8,min(nb*8+8,counts[e])):
                    row=slots[e][r]//8 if first else off[e]+r
                    add(p[3].data_ptr()+row*8192,8192);xbytes+=8192
    ptr=[];size=[];mask=[]
    for address,n,repeat in seg:
        assert n%16==0
        if repeat>1:
            assert n==8192
            for b in range(0,n*repeat,65536):
                ptr.append(address);size.append(min(65536,n*repeat-b)//16);mask.append(511)
        else:
            for b in range(0,n,65536):
                ptr.append(address+b);size.append(min(65536,n-b)//16);mask.append(-1)
    dev=p[3].device
    meta=[torch.tensor(ptr,device=dev,dtype=torch.int64),torch.tensor(size,device=dev,dtype=torch.int32),
          torch.tensor(mask,device=dev,dtype=torch.int32)]
    return meta,dict(unique_weight=unique_weight,scheduled_weight=scheduled_weight,activation=xbytes,
                     stream_bytes=scheduled_weight+xbytes,tasks=len(ptr))


def nd_plan(data,bank):
    streams=(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event())
    p=nf_plan(data,bank['base'],bank['native'],streams)
    p['steps'][0]();torch.cuda.synchronize()
    return p,streams


def nd_cell(plan,bank,branch,first):
    h13,h2,nv,ids,tw,pos,tasks=plan['owners'];m=nv[0].shape[0]
    stages=2
    while stages>1 and ND_MODS[0].pmz_smem(stages,m)>NF_OPTS['pmz']:stages-=1
    sms=torch.cuda.get_device_properties(nv[0].device).multi_processor_count
    grid=(1056 if m<=128 else 2112) if branch=='heavy' else sms*ND_MODS[0].pmz_resident(stages,m,first)
    vp=[t.data_ptr() for t in ((h13 if first else h2) if branch=='heavy' else nv)]
    # W2 always reads this same synthetic ACT, independently of W13's diagnostic outputs.
    if not first:nv[9].normal_(generator=torch.Generator(device=nv[0].device).manual_seed(172903))
    graphs=[]
    for mod in ND_MODS[:3]:
        if branch=='heavy':
            fn=lambda mod=mod:mod.br_run(vp,26 if first else 21,grid,m,torch.cuda.current_stream().cuda_stream)
        else:
            fn=lambda mod=mod:mod.pmz_run(vp,first,16,m,stages,grid,torch.cuda.current_stream().cuda_stream)
        graphs.append(nd_capture(fn))
    meta,counts=nd_segments(plan,bank,branch,first,grid)
    sink=torch.empty(sms*8*256,device=nv[0].device,dtype=torch.int32)
    stream_graphs=[]
    for mult in (4,8):
        fn=lambda mult=mult:ND_MODS[3].nd_read(*(t.data_ptr() for t in meta),sink.data_ptr(),counts['tasks'],sms*mult,torch.cuda.current_stream().cuda_stream)
        stream_graphs.append(nd_capture(fn))
    info=[]
    for mod in ND_MODS[:3]:
        info.append(list(mod.br_info(26 if first else 21)) if branch=='heavy' else list(mod.nd_info(first,m,stages)))
    return dict(graphs=graphs,stream_graphs=stream_graphs,counts=counts,info=info,grid=grid,owners=(plan,meta,sink))


def nd_error(out,want):
    d=out.float()-want.float();n=want.float().norm(dim=-1)
    return float(d.norm()/want.float().norm().clamp_min(1e-12)),float((d.norm(dim=-1)/n.clamp_min(n.mean()*.1+1e-12)).max())


def lab(deadline,weights,state=None):
    started=time.monotonic();nd_build();rows=[];checks=[];aa=[];resources={}
    events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
    lines=['META Four arms: exact S12 final nf; trivial dequant (all packed data consumed, original MMA/copies/epilogue); original load schedule without MMA/dequant/epilogue; max-MLP stream, same weight multiplicity and activation cache-refresh reads.',
           'LIMIT Stream includes packed codes/scales and BF16 activation reads; excludes small route/global-scale/post-norm metadata and output writes. Unique byte rates are logical proxies, not measured DRAM counters. Probes are intentionally incorrect, not eligible candidates. W2 ACT is fixed synthetic BF16 in all arms.']
    peak=0.
    for seed in range(12):
        if deadline-time.monotonic()<max(12.,peak*1.3+2.):break
        tick=time.monotonic();w=dc_weights(5353+7919*seed,next(v.device for v in weights.values() if torch.is_tensor(v)))
        bank=nd_prepare(w)
        for m in (128,256):
            ds=dc_fixtures(w,m)
            want=dc_oracles(ds[:1],w)[0]
            plan,streams=nd_plan(ds[0],bank)
            for f in plan['steps']:f()
            torch.cuda.synchronize();error=nd_error(plan['out'],want);checks.append(error)
            assert torch.isfinite(plan['out']).all() and error[0]<=.015 and error[1]<=.075,error
            del plan,streams
            for fi,data in enumerate(ds):
                for branch in ('heavy','NT1'):
                    for first in (1,0):
                        plan,streams=nd_plan(data,bank);cell=nd_cell(plan,bank,branch,first)
                        name=(m,branch,'W13' if first else 'W2');resources[name]=(cell['info'],cell['grid'])
                        graphs=cell['graphs'];a0,a1=nd_pair(graphs[0],graphs[0],events);aa.append(a1-a0)
                        us=[a0];rat=[1.]
                        for g in graphs[1:]:
                            a,b=nd_pair(graphs[0],g,events);us.append(b);rat.append(b/a)
                        stream_times=[]
                        for g in cell['stream_graphs']:
                            a,b=nd_pair(graphs[0],g,events);stream_times.append((b,b/a))
                        b,r=min(stream_times);us.append(b);rat.append(r)
                        rows.append(dict(key=name,seed=seed,fixture=fi,us=us,ratio=rat,counts=cell['counts']))
                        del cell,graphs,plan,streams
        peak=max(peak,time.monotonic()-tick)
        del bank,w;torch.cuda.empty_cache()
    for key in sorted(resources):
        rr=[r for r in rows if r['key']==key]
        mean=lambda field,i:statistics.mean(r[field][i] for r in rr)
        us=[mean('us',i) for i in range(4)];rat=[mean('ratio',i) for i in range(4)]
        ct={k:statistics.mean(r['counts'][k] for r in rr) for k in ('unique_weight','scheduled_weight','activation','stream_bytes')}
        lines.append('CELL M%d %s %s n=%d us=%s ratios=%s uniqueW_MB=%.2f scheduledW_MB=%.2f X_MB=%.2f stream_TBs=%.3f'%
                     (*key,len(rr),'/'.join(f'{x:.2f}' for x in us),'/'.join(f'{x:.4f}' for x in rat),
                      ct['unique_weight']/1e6,ct['scheduled_weight']/1e6,ct['activation']/1e6,ct['stream_bytes']/us[3]/1e6))
        lines.append('RES M%d %s %s grid=%d normal/trivial/loads=%s'%(*key,resources[key][1],resources[key][0]))
    if checks:lines.append('GATE unchanged S12 max L2/row=%.6f/%.6f'%(max(e[0] for e in checks),max(e[1] for e in checks)))
    if aa:lines.append('AA signed/abs/max=%.3f/%.3f/%.3f'%(statistics.mean(aa),statistics.mean(abs(x) for x in aa),max(abs(x) for x in aa)))
    lines.append(f'END seeds={len(checks)//2} cells={len(rows)} used={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    return lines
