P5_Q1_SEEDS=tuple(173101+7919*i for i in range(8))
P5_Q1_ROUNDS=8

def p5_q1_probe():
    import gc,json,statistics,time
    started=time.monotonic();dev=torch.device('cuda',torch.cuda.current_device());sms=torch.cuda.get_device_properties(dev).multi_processor_count
    flush=torch.empty(1<<28,device=dev,dtype=torch.uint8);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    def timed(g):
        flush.view(torch.int64).sum();a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    print('P5_Q1_META decode1 full only: single-head native N128 persistent gather, eight live query rows and compact8-row partial slots; whole-layer promotion waits on its paired layer lab.',flush=True)
    print('P5_Q1_RES',{st:h1_vq8_ext().vq8_resources(st) for st in (3,4)},flush=True)
    ratios={n:[] for n in ('A2','GC','H3','H4')};times={n:[] for n in ('A','A2','GC','H3','H4')};errors=[]
    for seed in P5_Q1_SEEDS:
        data=_lab_input(1,0,80000,seed,dev,1)
        fns={'A':lambda d=data:_BASE_FN(d),'A2':lambda d=data:_BASE_FN(d),'GC':lambda d=data:gc_attention(d),
            'H3':lambda d=data:h1_vq8_attention(d,stages=3,min_tiles=2,grid=2*sms),
            'H4':lambda d=data:h1_vq8_attention(d,stages=4,min_tiles=2,grid=2*sms)}
        graphs={};out={}
        for n,fn in fns.items():
            fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out[n]=fn()
            graphs[n]=g
        for g in graphs.values():g.replay()
        torch.cuda.synchronize();err={}
        for n in times:assert bool(torch.isfinite(out[n]).all()),('nonfinite',n,seed)
        for n in ('H3','H4'):
            err[n]=float((out[n].float()-out['GC'].float()).norm()/out['GC'].float().norm());assert err[n]<.03,('q1 native FP8 gate',n,seed,err[n])
        W=data['qkv'].shape[-1];partbuf=(torch.randn((3,1,W+64),device=dev)*.03)[:,:,:W]
        partbuf[-1]+=data['qkv'].view(1,W).float()-partbuf.sum(0)
        got=h1_vq8_attention(data,qkv=partbuf,proj_splits=3,split_stride=partbuf.stride(0),stages=4,min_tiles=2,grid=2*sms)
        want=gc_attention(data,(partbuf,3));torch.cuda.synchronize();err['partial']=float((got.float()-want.float()).norm()/want.float().norm());assert bool(torch.isfinite(got).all()) and err['partial']<.03,('partial gate',seed,err['partial'])
        per={n:[] for n in graphs};names=list(graphs)
        for j in range(P5_Q1_ROUNDS):
            for n in names[j%len(names):]+names[:j%len(names)]:per[n].append(timed(graphs[n]))
        avg={n:statistics.mean(v) for n,v in per.items()}
        for n in ratios:ratios[n].append(avg[n]/avg['A'])
        for n in times:times[n].append(avg[n])
        errors.append(err);print('P5_Q1 '+json.dumps(dict(seed=seed,us=avg,err=err)),flush=True)
        if seed==P5_Q1_SEEDS[0]:
            clk=[];h1_vq8_attention(data,qkv=partbuf,proj_splits=3,split_stride=partbuf.stride(0),stages=4,min_tiles=2,grid=2*sms,clocks=clk);torch.cuda.synchronize()
            main=clk[0][:2*sms].cpu();live=main[main[:,10]!=-1];tiles=live[:,10].double().clamp_min(1);t0=main[:,0].min()
            print('P5_Q1_CLOCK '+json.dumps(dict(ctas=len(live),tile_min=int(tiles.min()),tile_max=int(tiles.max()),end_max_us=float((live[:,3]-t0).max())/1000,prepared_max_us=float((live[:,15]-t0).max())/1000,cycles=float((live[:,5]/tiles).median()),per_tile={n:float((live[:,i]/tiles).median()) for n,i in [('qkw',19),('smx',20),('pvw',21),('issue',22),('publish',23),('producer_issue',24)]})),flush=True)
        del graphs,out,got,want,fns,data,partbuf;gc.collect();torch.cuda.empty_cache()
    row=dict(r={n:statistics.mean(v) for n,v in ratios.items()},se={n:statistics.stdev(v)/len(v)**.5 if len(v)>1 else 0 for n,v in ratios.items()},us={n:statistics.mean(v) for n,v in times.items()},times=times,ratios=ratios,errors=errors,seeds=len(P5_Q1_SEEDS))
    print('P5_Q1_SUM '+json.dumps(row),flush=True);print('P5_Q1_END %.2fs'%(time.monotonic()-started),flush=True);return row

def kernel_fn(data:Inputs)->torch.Tensor:
    cfg=data['config']
    if cfg['batch']==1 and cfg['q_len']==1 and not cfg['window'] and cfg['num_kv_heads']==2:
        return h1_vq8_attention(data,stages=4,min_tiles=2)
    if cfg['q_len']==1:return gc_attention(data)
    if not cfg['window'] and cfg['batch'] in (16,32) and cfg['num_kv_heads']==2:return p8_vq8_attention(data,stages=4,min_tiles=8)
    return _BASE_FN(data)

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0]==9:
    p5_q1_probe()
