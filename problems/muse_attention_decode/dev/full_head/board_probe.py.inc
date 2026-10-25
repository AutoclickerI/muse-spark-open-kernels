P5_HEAD_SEEDS=(173101,181177,192103)
P5_HEAD_ROUNDS=8

def p5_head_probe():
    import gc,json,statistics,time
    started=time.monotonic();dev=torch.device('cuda',torch.cuda.current_device());sms=torch.cuda.get_device_properties(dev).multi_processor_count
    flush=torch.empty(1<<28,device=dev,dtype=torch.uint8);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    def timed(g):
        flush.view(torch.int64).sum();a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    print('P5_HEAD_META one-head N128 CTA192 vs measured two-head N128 CTA384; native stage-only evidence, no whole-layer promotion.',flush=True)
    print('P5_HEAD_RES', {st:h1_vq8_ext().vq8_resources(st) for st in (3,4)},flush=True)
    rows=[]
    for B in (16,32):
        ratios={n:[] for n in ('A2','H1','H2','H3')}
        for seed in P5_HEAD_SEEDS:
            data=_lab_input(B,0,80000,seed,dev,8)
            fns={'A':lambda d=data:p8_vq8_attention(d,stages=4,min_tiles=8),
                'A2':lambda d=data:p8_vq8_attention(d,stages=4,min_tiles=8),
                'H1':lambda d=data:h1_vq8_attention(d,stages=4,min_tiles=8,grid=sms),
                'H2':lambda d=data:h1_vq8_attention(d,stages=4,min_tiles=8,grid=2*sms),
                'H3':lambda d=data:h1_vq8_attention(d,stages=3,min_tiles=8,grid=2*sms)}
            graphs={};out={}
            for n,fn in fns.items():
                fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):out[n]=fn()
                graphs[n]=g
            for g in graphs.values():g.replay()
            torch.cuda.synchronize();errs={}
            for n in ratios:
                assert bool(torch.isfinite(out[n]).all()),('head nonfinite',B,n,seed)
                err=float((out[n].float()-out['A'].float()).norm()/out['A'].float().norm())
                errs[n]=err;assert err<.03,('head disagrees with native FP8 control',B,n,seed,err)
            per={n:[] for n in graphs};names=list(graphs)
            for j in range(P5_HEAD_ROUNDS):
                for n in names[j%len(names):]+names[:j%len(names)]:per[n].append(timed(graphs[n]))
            avg={n:statistics.mean(v) for n,v in per.items()}
            for n in ratios:ratios[n].append(avg[n]/avg['A'])
            print('P5_HEAD '+json.dumps(dict(B=B,seed=seed,us=avg,err=errs)),flush=True)
            # Real strided FP32 projection partials: native accumulation/rounding occurs inside prologue.
            W=data['qkv'].shape[-1];T=B*8;partbuf=(torch.randn((3,T,W+64),device=dev)*.03)[:,:,:W]
            partbuf[-1]+=data['qkv'].view(T,W).float()-partbuf.sum(0)
            outp=h1_vq8_attention(data,qkv=partbuf,proj_splits=3,split_stride=partbuf.stride(0),stages=4,min_tiles=8,grid=2*sms)
            want=p8_vq8_attention(data,qkv=partbuf,proj_splits=3,split_stride=partbuf.stride(0),stages=4,min_tiles=8)
            torch.cuda.synchronize();err=float((outp.float()-want.float()).norm()/want.float().norm());assert bool(torch.isfinite(outp).all()) and err<.03,('partial gate',B,seed,err)
            if seed==P5_HEAD_SEEDS[0]:
                clocks=[];h1_vq8_attention(data,qkv=partbuf,proj_splits=3,split_stride=partbuf.stride(0),stages=4,min_tiles=8,grid=2*sms,clocks=clocks);torch.cuda.synchronize()
                main=clocks[0][:2*sms].cpu();live=main[main[:,10]!=-1];tiles=live[:,10].double().clamp_min(1)
                print('P5_HEAD_CLOCK '+json.dumps(dict(B=B,ctas=len(live),tile_min=int(tiles.min()),tile_max=int(tiles.max()),cycles=float((live[:,5]/tiles).median()),per_tile={n:float((live[:,i]/tiles).median()) for n,i in [('qkw',19),('smx',20),('pvw',21),('issue',22),('publish',23),('producer_issue',24)]})),flush=True)
            del graphs,out,outp,want,fns,data,partbuf;gc.collect();torch.cuda.empty_cache()
        row=dict(B=B,r={n:statistics.mean(v) for n,v in ratios.items()},se={n:statistics.stdev(v)/len(v)**.5 if len(v)>1 else 0 for n,v in ratios.items()});rows.append(row)
        print('P5_HEAD_SUM '+json.dumps(row),flush=True)
    print('P5_HEAD_END %.2fs'%(time.monotonic()-started),flush=True)
    return rows

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0]==9:
    p5_head_probe()
