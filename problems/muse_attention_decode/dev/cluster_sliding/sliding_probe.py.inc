P5_SLIDE_SEEDS=tuple(173101+7919*i for i in range(8))
P5_SLIDE_ROUNDS=8

def p5_sliding_probe():
    import gc,json,statistics,time,base64,lzma
    started=time.monotonic();dev=torch.device('cuda',torch.cuda.current_device())
    flush=torch.empty(1<<28,device=dev,dtype=torch.uint8);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    def timed(g):
        flush.view(torch.int64).sum();a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    print('LAB_BEGIN p5_sliding_padded_mask_fast_rope_and_verify',flush=True)
    print('META one-launch sliding decode/verify; original last-CTA, masked padded query rows, fast RoPE intrinsic and verify last-CTA. Native stage-only evidence; whole-layer promotion waits quota.',flush=True)
    rows=[]
    for L in (1,8):
        for B in (1,16,32):
            ratios={n:[] for n in ('A2','V','LAST','MASK','FAST')};times={n:[] for n in ('A','A2','V','LAST','MASK','FAST')};gates=[]
            for seed in P5_SLIDE_SEEDS:
                data=_lab_input(B,2048,80000,seed,dev,L)
                def vq(d):return _vq8(d,stages=4,min_tiles=2 if B==32 else 1)
                fns={'A':lambda d=data:_BASE_FN(d),'A2':lambda d=data:_BASE_FN(d),
                    'V':lambda d=data:vq(d) if L==8 else _BASE_FN(d),
                    'LAST':lambda d=data:lc_attention(d) if L==1 else vq(d),
                    'MASK':lambda d=data:mc_attention(d) if L==1 else vq(d),
                    'FAST':lambda d=data:fc_attention(d)}
                graphs={};out={}
                for n,fn in fns.items():
                    fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):out[n]=fn()
                    graphs[n]=g
                for g in graphs.values():g.replay()
                torch.cuda.synchronize();err={}
                for n in times:assert bool(torch.isfinite(out[n]).all()),('nonfinite sliding',B,L,n,seed)
                for n in ('MASK','FAST'):
                    err[n]=float((out[n].float()-out['LAST'].float()).norm()/out['LAST'].float().norm());assert err[n]<.04,('sliding FP8 gate',B,L,n,seed,err[n])
                W=data['qkv'].shape[-1];T=B*L;partbuf=(torch.randn((3,T,W+64),device=dev)*.03)[:,:,:W]
                partbuf[-1]+=data['qkv'].view(T,W).float()-partbuf.sum(0)
                got=fc_attention(data,(partbuf,3));torch.cuda.synchronize();err['partial']=float((got.float()-out['FAST'].float()).norm()/out['FAST'].float().norm());assert bool(torch.isfinite(got).all()) and err['partial']<.005,('sliding partial gate',B,L,seed,err['partial']);gates.append(err)
                per={n:[] for n in graphs};names=list(graphs)
                for j in range(P5_SLIDE_ROUNDS):
                    for n in names[j%len(names):]+names[:j%len(names)]:per[n].append(timed(graphs[n]))
                avg={n:statistics.mean(v) for n,v in per.items()}
                for n in ratios:ratios[n].append(avg[n]/avg['A'])
                for n in times:times[n].append(avg[n])
                del graphs,out,got,fns,data,partbuf;gc.collect();torch.cuda.empty_cache()
            def ratio(n,d):return statistics.mean(x/y for x,y in zip(times[n],times[d]))
            row=dict(B=B,L=L,r={n:statistics.mean(v) for n,v in ratios.items()},se={n:statistics.stdev(v)/len(v)**.5 if len(v)>1 else 0 for n,v in ratios.items()},us={n:statistics.mean(v) for n,v in times.items()},times=times,ratios=ratios,gates=gates,seeds=len(P5_SLIDE_SEEDS),fast_vs_last=ratio('FAST','LAST'),mask_vs_last=ratio('MASK','LAST'),fast_vs_v=ratio('FAST','V'))
            rows.append(row);print('SL '+json.dumps({k:v for k,v in row.items() if k not in ('times','ratios','gates')}),flush=True)
    archive=base64.b85encode(lzma.compress(json.dumps(rows,separators=(',',':')).encode())).decode();print('SL_XZ85 '+archive,flush=True)
    print('LAB_END elapsed=%.2fs'%(time.monotonic()-started),flush=True);return rows

def kernel_fn(data:Inputs)->torch.Tensor:
    cfg=data['config']
    if cfg['window']:return fc_attention(data)
    if cfg['q_len']==8 and cfg['batch'] in (16,32) and cfg['num_kv_heads']==2:return p8_vq8_attention(data,stages=4,min_tiles=8)
    return _BASE_FN(data)

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0]==9:
    p5_sliding_probe()
