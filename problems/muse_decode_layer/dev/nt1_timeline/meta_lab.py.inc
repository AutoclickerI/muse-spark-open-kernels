_META_PRODUCER=[]
def meta_producer():
    if not _META_PRODUCER:
        target='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
        tag=hashlib.sha256(META_GLUE_CUDA.encode()).hexdigest()[:12]
        _META_PRODUCER.append(load_inline(name='nt1_meta_producer_'+tag+'_'+target,cpp_sources=META_GLUE_CPP,cuda_sources=META_GLUE_CUDA,
            functions=['route_cl'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','-gencode=arch=compute_'+target+',code=sm_'+target],verbose=False))
    return _META_PRODUCER[0]

def meta_routed(pr,rs,state,meta,arrive,kind):
    if kind=='S11':x,ids,tw,lids=route_call(pr,rs)
    else:
        p,s=pr;m=p.shape[1];dev=p.device;x=torch.empty((m,4096),device=dev,dtype=torch.bfloat16)
        ids=torch.empty((m,8),device=dev,dtype=torch.int32);lids=torch.empty_like(ids);tw=torch.empty((m,8),device=dev)
        on=kind!='CL_OFF'
        meta_producer().route_cl(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),rs['bias'].data_ptr(),rs['pre'].data_ptr(),rs['start'].data_ptr(),
            x.data_ptr(),ids.data_ptr(),tw.data_ptr(),lids.data_ptr(),meta.data_ptr() if on else 0,arrive.data_ptr() if on else 0,m,8,256,torch.cuda.current_stream(dev).cuda_stream)
    d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0},route_meta=meta)
    out=metam_experts(d,state,101) if kind=='CL_META' else smallm_experts(d,state,69)
    return out,x,ids,tw,lids

def lab(deadline,weights):
    from concurrent.futures import ThreadPoolExecutor
    begin=time.monotonic();deadline-=5;device=next(v.device for v in weights.values() if isinstance(v,torch.Tensor))
    lines=['META P3a cluster-route MoERouteMeta v1 consumer: S11=exact acda6c62 route+smallm69; CL_OFF=cluster C8 NT256, no metadata; CL_BASE=cluster writes metadata+S11 consumer69; CL_META=same producer+consumer101 (Z/ZOUT/PDL/META). Metadata is 736 int32, generated inside timed route.',
        'META read flush, router+MoE CUDA graphs; AB/BA x6, 16 seeds of public weights/router; TASK routing and output gates for each arm; direct consumer=CL_META/CL_BASE and producer=CL_BASE/CL_OFF pairs isolate producer/consumer from changed route geometry. r=new/S11.']
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(fn) for fn in (smallm_module,metam_module,meta_producer)]
        for j in jobs:j.result()
    lines.append(f'BUILD seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
    ratios={};deltas={};times={};aa={m:[] for m in (8,16,32)};fails={};worst={};done=0;peak=0.
    oldtf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:
        for seed in range(16):
            if deadline-time.monotonic()<max(8,peak*1.15+2):lines.append(f'BUDGET before_seed={seed}');break
            started=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=device);w['post_norm_w']=w['post_expert_norm'];base=nt1_banks(w)
            state={k:base[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':base['bank']}
            smallm_bind(state,{8:69,16:69,32:69});metam_bind(state,{8:101,16:101,32:101})
            for m in (8,16,32):
                pr,rs=routed_fixture(w,m,9000+m+97*seed);truths={};graphs={}
                meta=torch.empty(736,device=device,dtype=torch.int32);arrive=torch.zeros(1,device=device,dtype=torch.int32)
                for kind in ('S11','CL_OFF','CL_BASE','CL_META'):
                    if kind in fails:continue
                    try:
                        graphs[kind]=TimelineGraph(lambda kind=kind:meta_routed(pr,rs,state,meta,arrive,kind))
                        worst[kind]=max(worst.get(kind,(0.,0.)),reuse_gate(graphs[kind].replay(),w,truths,pr,rs))
                    except Exception as exc:fails[kind]=f'M{m} seed{seed} {type(exc).__name__} {str(exc)[:140]}';graphs.pop(kind,None)
                if 'S11' not in graphs:continue
                control=TimelineGraph(lambda:meta_routed(pr,rs,state,meta,arrive,'S11'))
                a,b,_=timeline_timing(graphs['S11'],control);aa[m].append(b-a);times.setdefault((m,'S11'),[]).append(a)
                kinds=['CL_OFF','CL_BASE','CL_META']
                if seed%2:kinds.reverse()
                for kind in kinds:
                    if kind not in graphs:continue
                    a,b,r=timeline_timing(graphs['S11'],graphs[kind]);ratios.setdefault((m,kind),[]).append(r);times.setdefault((m,kind),[]).append(b);deltas.setdefault((m,kind),[]).append(b-a)
                    try:worst[kind]=max(worst.get(kind,(0.,0.)),reuse_gate(graphs[kind].replay(),w,truths,pr,rs))
                    except Exception as exc:fails[kind]=f'M{m} seed{seed} after {type(exc).__name__} {str(exc)[:140]}'
                for label,a,b in (('consumer','CL_BASE','CL_META'),('producer','CL_OFF','CL_BASE')):
                    if a in graphs and b in graphs:
                        x,y,r=timeline_timing(graphs[a],graphs[b]);ratios.setdefault((m,label),[]).append(r);deltas.setdefault((m,label),[]).append(y-x)
                graphs={};truths={}
            done+=1;peak=max(peak,time.monotonic()-started)
    finally:torch.backends.cuda.matmul.allow_tf32=oldtf
    for kind,why in fails.items():lines.append(f'FAIL {kind} {why}')
    lines.append('GATE '+','.join(f'{k}:{v[0]:.5f}/{v[1]:.5f}' for k,v in sorted(worst.items())))
    for m in (8,16,32):
        if not aa[m]:continue
        parts=[f'S11_us={statistics.mean(times[m,"S11"]):.2f}']
        for kind in ('CL_OFF','CL_BASE','CL_META','consumer','producer'):
            r=ratios.get((m,kind),[])
            if not r:continue
            se=statistics.stdev(r)/len(r)**.5 if len(r)>1 else 0.
            parts.append(f'{kind} r={statistics.mean(r):.5f} se={se:.5f} d_us={statistics.mean(deltas[m,kind]):+.2f} wins={sum(x<1 for x in r)}/{len(r)}'+(f' us={statistics.mean(times[m,kind]):.2f}' if (m,kind) in times else ''))
        parts.append(f'AA={statistics.mean(aa[m]):+.2f}/{statistics.mean(map(abs,aa[m])):.2f}/{max(map(abs,aa[m])):.2f}')
        lines.append(f'AGG M{m} '+' | '.join(parts))
    lines.append(f'END seeds={done} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}; whole-layer pairing required')
    return lines
