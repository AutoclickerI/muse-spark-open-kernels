P5_DSM_SEEDS=tuple(173101+7919*i for i in range(8))
P5_DSM_ROUNDS=8
P5_DSM_CLUSTERS=(1,2,4,8,16)
P5_DSM_TARGET={1:16,16:4,32:2}

def p5_dsm_probe():
    import gc,json,statistics,time,base64,lzma
    started=time.monotonic();dev=torch.device('cuda',torch.cuda.current_device())
    flush=torch.empty(1<<28,device=dev,dtype=torch.uint8);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    def timed(g):
        flush.view(torch.int64).sum();a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    print('LAB_BEGIN p5_verify_dsm_one_wave',flush=True)
    print('META fixed cluster1/2/4/8/16, QT4 atB1/QT1 atB16/32; native one-wave target16/4/2; distributed output rows across cluster CTAs, no last-CTA serialization. Stage evidence only.',flush=True)
    rows=[]
    for B in (1,16,32):
        qt=4 if B==1 else 1;res={c:cc_module().sliding_resources(8,qt,128,c) for c in P5_DSM_CLUSTERS}
        valid=[c for c,v in res.items() if v[-1]>0];assert P5_DSM_TARGET[B] in valid,res
        names=['A','A2']+['C'+str(c) for c in valid];times={n:[] for n in names};gates=[]
        for seed in P5_DSM_SEEDS:
            data=_lab_input(B,2048,80000,seed,dev,8)
            fns={'A':lambda d=data:_vq8(d,stages=4,min_tiles=2 if B==32 else 1),'A2':lambda d=data:_vq8(d,stages=4,min_tiles=2 if B==32 else 1)}
            for c in valid:fns['C'+str(c)]=lambda d=data,c=c:cc_attention(d,nk=128,qt=qt,cs=c)
            graphs={};out={}
            for n,fn in fns.items():
                fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):out[n]=fn()
                graphs[n]=g
            for g in graphs.values():g.replay()
            torch.cuda.synchronize();err={}
            for n in names:
                assert bool(torch.isfinite(out[n]).all()),('dsm nonfinite',B,n,seed)
                err[n]=float((out[n].float()-out['A'].float()).norm()/out['A'].float().norm());assert err[n]<.04,('dsm FP8 gate',B,n,seed,err[n])
            W=data['qkv'].shape[-1];T=B*8;partbuf=(torch.randn((3,T,W+64),device=dev)*.03)[:,:,:W]
            partbuf[-1]+=data['qkv'].view(T,W).float()-partbuf.sum(0)
            got=cc_attention(data,(partbuf,3),nk=128,qt=qt,cs=P5_DSM_TARGET[B]);torch.cuda.synchronize()
            wanted=out['C'+str(P5_DSM_TARGET[B])];err['partial']=float((got.float()-wanted.float()).norm()/wanted.float().norm());assert bool(torch.isfinite(got).all()) and err['partial']<.005,('partial gate',B,seed,err['partial']);gates.append(err)
            per={n:[] for n in names}
            for j in range(P5_DSM_ROUNDS):
                for n in names[j%len(names):]+names[:j%len(names)]:per[n].append(timed(graphs[n]))
            for n in names:times[n].append(statistics.mean(per[n]))
            del graphs,out,got,wanted,fns,data,partbuf;gc.collect();torch.cuda.empty_cache()
        rr={n:[x/y for x,y in zip(v,times['A'])] for n,v in times.items() if n!='A'}
        row=dict(B=B,res=res,us={n:statistics.mean(v) for n,v in times.items()},r={n:statistics.mean(v) for n,v in rr.items()},se={n:statistics.stdev(v)/len(v)**.5 if len(v)>1 else 0 for n,v in rr.items()},times=times,ratios=rr,gates=gates,seeds=len(P5_DSM_SEEDS));rows.append(row)
        print('DS '+json.dumps({k:v for k,v in row.items() if k not in ('times','ratios','gates')}),flush=True)
    print('DS_XZ85 '+base64.b85encode(lzma.compress(json.dumps(rows,separators=(',',':')).encode())).decode(),flush=True)
    print('LAB_END elapsed=%.2fs'%(time.monotonic()-started),flush=True);return rows

def kernel_fn(data:Inputs)->torch.Tensor:
    cfg=data['config'];B=cfg['batch'];L=cfg['q_len']
    if L==1 and B==1 and cfg['window']:return lc_attention(data)
    if L==8 and cfg['window'] and B in (1,16,32):return cc_attention(data,nk=128,cs={1:16,16:4,32:2}[B])
    if L==8 and not cfg['window'] and B in (16,32) and cfg['num_kv_heads']==2:return p8_vq8_attention(data,stages=4,min_tiles=8)
    return _BASE_FN(data)

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0]==9:
    p5_dsm_probe()
