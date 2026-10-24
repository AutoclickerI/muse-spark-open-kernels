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
