"""H200 fixed-producer test: 12 public seeds and R0; no per-seed winner selection."""


class VecGraph:
    def __init__(self,fn,data):
        self.data={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
        fn(self.data);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=fn(self.data)
        self.graph.replay();torch.cuda.synchronize()

    def refresh(self,data):
        for key in ('hidden_states','topk_ids','topk_weights'):self.data[key].copy_(data[key])


def vector_measure(w,bank,old,datasets,flush):
    names=('V9NT1','V9NT2','Vec16','Vec32','Vec64')
    graphs=[]
    for nt in (1,2):
        st=dict(old,nt=nt,nstage_choice=2)
        graphs.append(VecGraph(lambda d,s=st:rv_kernel_fn(d,s),datasets[0]))
    for nt in (16,32,64):graphs.append(VecGraph(lambda d,n=nt:vec_call(d,bank,n,264),datasets[0]))
    worst=[0.]*len(names);rows=[0.]*len(names);times=[[] for _ in names]
    for data in datasets:
        want=reuse_fp32(data,w);den=want.float().norm().clamp_min(1e-12)
        rowden=want.float().norm(dim=-1);rowden=rowden.clamp_min(rowden.mean()*.1+1e-12)
        for j,g in enumerate(graphs):
            g.refresh(data);g.graph.replay();torch.cuda.synchronize();got=g.out.float()
            if not bool(torch.isfinite(got).all()):worst[j]=float('inf');rows[j]=float('inf');continue
            worst[j]=max(worst[j],float((got-want.float()).norm()/den))
            rows[j]=max(rows[j],float(((got-want.float()).norm(dim=-1)/rowden).max()))
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(8):
        for g in graphs:g.refresh(datasets[i%3])
        order=list(range(len(names)));order=order[i%len(order):]+order[:i%len(order)]
        if i%2:order.reverse()
        for j in order:
            flush.zero_();a.record();graphs[j].graph.replay();b.record();b.synchronize()
            times[j].append(a.elapsed_time(b)*1000.)
    return [(statistics.median(t),e,r) for t,e,r in zip(times,worst,rows)]


def vector_phase(data,bank,flush):
    results=[]
    for nt in (16,32,64):
        fs=vec_work(data,bank,nt,264)
        for f in fs:f()
        row=[]
        for f in fs:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):f()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);ts=[]
            for _ in range(6):
                flush.zero_();a.record();graph.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000.)
            row.append(statistics.median(ts))
        results.append(f'PHASE NT={nt} route,w13,act,w2,norm,combine='+','.join(f'{t:.1f}' for t in row))
    return results


def lab(deadline,weights,state=None):
    dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META packed register fragments, CTA-local format LUT from raw E4M3, two-stage cp.async Q/SF/X, BK64, no spills in sm90a. ORDER V9NT1,V9NT2,Vec16,Vec32,Vec64. 12 public generator seeds; 3 fresh real-router fixtures/seed/M,8 rotated cold128MiB graph timings. FP32 L2<=.01,row<=.05. R0 exact IDs uses fresh public weights, not original decode-host weights. Phases diagnostic, do not sum.']
    raw=zlib.decompress(base64.b64decode(CAPTURED_R0));ms=(8,16,32,128,256)
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8);records=[]
    for i in range(12):
        if time.monotonic()>deadline-15:lines.append(f'INCOMPLETE before_seed={i}');break
        seed=5353+7919*i;w=rg_generate_weights(seed,device=dev);w['post_norm_w']=w['post_expert_norm']
        old=rv_prepare(w);bank=vec_prepare(w)
        for m in (128,256):
            ds=[fx_make_fixture(w,m,9000+m+7919*j) for j in range(3)]
            if i==0:
                off=8*sum(ms[:ms.index(m)])
                ids=torch.tensor(list(raw[off:off+m*8]),device=dev,dtype=torch.int32).view(m,8)
                ids=torch.where(ids<32,ids,-1)
                r0=[dict(d,topk_ids=ids.clone()) for d in ds]
                vals=vector_measure(w,bank,old,r0,flush)
                lines.append(f'R0 M={m} us='+','.join(f'{v[0]:.1f}' for v in vals)+' l2='+','.join(f'{v[1]:.5f}' for v in vals)+' row='+','.join(f'{v[2]:.4f}' for v in vals))
                lines.extend(f'M={m} '+p for p in vector_phase(r0[0],bank,flush))
            vals=vector_measure(w,bank,old,ds,flush)
            counts=torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids']>=0],minlength=32)
            lines.append(f'S seed={seed} M={m} P={int(counts.sum())} E={int((counts>0).sum())} hot={int(counts.max())} us='+','.join(f'{v[0]:.1f}' for v in vals)+' l2='+','.join(f'{v[1]:.5f}' for v in vals)+' row='+','.join(f'{v[2]:.4f}' for v in vals))
            records.append((m,vals))
        # Graphs and scratch live only during each measurement; no closure name is deleted.
    names=('V9NT1','V9NT2','Vec16','Vec32','Vec64')
    for m in (128,256):
        for j,name in enumerate(names):
            vals=[v[j] for mm,v in records if mm==m]
            if not vals:continue
            good=all(v[1]<=.01 and v[2]<=.05 for v in vals)
            times=[v[0] for v in vals]
            lines.append(f'AGG {name} M={m} seeds={len(vals)} mean={statistics.mean(times):.1f} median={statistics.median(times):.1f} min={min(times):.1f} max={max(times):.1f} '+('PASS' if good else 'REJECT'))
    return lines
