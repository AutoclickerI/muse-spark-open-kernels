def vp_pair(old,new,x,banks,flush,repeats=20):
    times=[[],[]];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for r in range(repeats):
        x.copy_(banks[r%len(banks)])
        for side in ((0,1) if r%2==0 else (1,0)):
            flush.sum()
            a.record();(old if side==0 else new).graph.replay();b.record();b.synchronize()
            times[side].append(float(a.elapsed_time(b))*1000)
    ratios=[((times[1][i]/times[0][i])*(times[1][i+1]/times[0][i+1]))**.5 for i in range(0,repeats,2)]
    return statistics.median(times[0]),statistics.median(times[1]),statistics.median(ratios)


def vp_check(out,expected):
    rel=float((out.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12))
    if not bool(torch.isfinite(out).all()) or rel>.01:raise ValueError(f'FP32 self-reference rel={rel:.5g}')
    return rel


def lab(deadline,weights,state=None):
    rows=[];native=_pg_native();dev=weights['qkv_gate'].device
    router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
    ws={k:weights[k] for k in ('qkv_gate','o','unpooler')}
    ws['pool_router']=torch.cat((weights['pooler'],hi,lo),0).contiguous()
    workspace=torch.empty(4<<30,device=dev,dtype=torch.uint8)
    flush=torch.zeros((128<<20)//4,device=dev,dtype=torch.float32)
    gen=torch.Generator(device=dev).manual_seed(20261025)
    keys=[(m,name) for m in (256,128) for name in ('pool_router','qkv_gate','unpooler','o')]
    rows.append(f'VP_META sms={torch.cuda.get_device_properties(dev).multi_processor_count} workspace_gb={workspace.numel()/1e9:.2f} cold128MiB graph fresh3; raw_us includes replay overhead')
    for pos,(m,name) in enumerate(keys):
        now=time.monotonic();stop=now+max(0,deadline-now-5)/(len(keys)-pos)
        if now>=deadline-8:rows.append(f'VP_BUDGET {m} {name}');continue
        w=ws[name];n,k=w.shape;fp32=name=='pool_router';dtype=torch.float32 if fp32 else torch.bfloat16
        banks=[torch.randn(m,k,device=dev,dtype=torch.bfloat16,generator=gen) for _ in range(3)]
        x=banks[0].clone();reference=[torch.mm(v.float(),w.float().T) for v in banks]
        old=_PgGraph(lambda:torch.mm(x,w.T,out_dtype=torch.float32) if fp32 else torch.mm(x,w.T))
        candidates=[];data={};attempts=bad=0;reasons={}
        coarse_stop=time.monotonic()+.7*max(0,stop-time.monotonic())
        for nn in (0,1):
            layout_stop=time.monotonic()+max(0,coarse_stop-time.monotonic())/(2-nn)
            weight=w.T.contiguous() if nn else w
            try:codes,heuristics=native.options(m,n,k,nn,fp32,workspace.numel())
            except Exception as exc:rows.append(f'VP_REJECT layout {m} {name} nn={nn} {type(exc).__name__}:{str(exc)[:80]}');continue
            data[nn]=(weight,codes)
            rows.append(f'VP_ALGOS {m} {name} nn={nn} h={heuristics} expanded={len(codes)}')
            for index in range(len(codes)):
                if time.monotonic()>layout_stop:break
                attempts+=1;g=None
                try:
                    out=torch.empty((m,n),dtype=dtype,device=dev)
                    def run(weight=weight,codes=codes,index=index,nn=nn,out=out):
                        native.run(x,weight,out,workspace,codes,index,nn,fp32);return out
                    g=_PgGraph(run)
                    error=0.
                    for b in range(3):
                        x.copy_(banks[b]);g.graph.replay();error=max(error,vp_check(g.out,reference[b]))
                    t0,t1,ratio=vp_pair(old,g,x,banks,flush,4)
                    candidates.append((ratio,nn,index,error))
                except Exception as exc:
                    bad+=1;reason=f'{type(exc).__name__}:{str(exc).splitlines()[0][:100]}'
                    if reason not in reasons and len(reasons)<3:rows.append(f'VP_REJECT {m} {name} {nn}.{index} {reason}')
                    reasons[reason]=reasons.get(reason,0)+1
                finally:
                    if g is not None:g.close()
        finalists=sorted(candidates)[:3]
        for nn in (0,1):
            family=[r for r in candidates if r[1]==nn]
            if family and min(family) not in finalists:finalists.append(min(family))
        best=(1.,'torch',None);details=[]
        for _,nn,index,error in finalists:
            if time.monotonic()>min(stop,deadline-4):break
            g=None
            try:
                weight,codes=data[nn];out=torch.empty((m,n),dtype=dtype,device=dev)
                def run():native.run(x,weight,out,workspace,codes,index,nn,fp32);return out
                g=_PgGraph(run);t0,t1,ratio=vp_pair(old,g,x,banks,flush)
                for b in range(3):x.copy_(banks[b]);g.graph.replay();vp_check(g.out,reference[b])
                details.append(f'{nn}.{index}:{t0:.2f}/{t1:.2f}/{ratio:.4f}')
                if ratio<best[0]:best=(ratio,(nn,index),(t0,t1))
            except Exception as exc:rows.append(f'VP_REJECT final {m} {name} {nn}.{index} {type(exc).__name__}:{str(exc)[:80]}')
            finally:
                if g is not None:g.close()
        rows.append(f'VP_LT {m} {name} best={best} tried={attempts} rejected={bad} finals='+','.join(details))
        old.close()
        del banks,x,reference,old,data,candidates
    return rows
