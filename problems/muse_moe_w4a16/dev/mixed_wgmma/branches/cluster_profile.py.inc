def oc_profiles(weights,bank):
    """One fixed fixture: whole-heavy/whole-light phase service, plus gather.

    These are isolated phases, not critical-path increments or promotion data.
    W2 always gets the same synthetic ACT independently of W13's outputs.
    """
    result=[];ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
    def capture(fn):
        fn();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        return g
    for m in (128,256):
        data=dc_fixtures(weights,m)[0];graphs=[];plans=[];stream_owners=[];gather=[]
        for arm in range(4):
            plan=op_plan(data,bank,arm);plan['steps'][0]();torch.cuda.synchronize();plans.append(plan)
            mod=OP_MODS[arm];h13,h2,nv=plan['owners'][:3]
            count=h13[4].cpu().tolist();sms=torch.cuda.get_device_properties(data['hidden_states'].device).multi_processor_count
            if arm==0:
                classes=[sum(max(1,(n+31)//32) for n in count if lo<n<=hi) for lo,hi in ((16,32),(32,64),(64,128),(128,192),(192,256))]
                result.append('OC_COUNTS M%d n=%s heavy_groups=%s'%(m,count,classes));gather.append(None)
                l13,l2=plan['owners'][7:9]
            else:
                gx=plan['owners'][7];l13,l2=plan['owners'][8:10]
                v=[t.data_ptr() for t in h13];v[3]=data['hidden_states'].data_ptr()
                gather.append(capture(lambda v=v,gx=gx,mod=mod:mod.oc_gather(v,gx.data_ptr(),m,torch.cuda.current_stream().cuda_stream)))
            p13,p2,pn,pl13,pl2=([t.data_ptr() for t in z] for z in (h13,h2,nv,l13,l2))
            sides=[torch.cuda.Stream() for _ in range(4)];fork=torch.cuda.Event();joins=[torch.cuda.Event() for _ in range(4)]
            light=torch.cuda.Stream();ljoin=torch.cuda.Event();stream_owners.append((sides,fork,joins,light,ljoin))
            phases=[]
            for first in (1,0):
                if not first:h13[10].normal_(generator=torch.Generator(device=h13[10].device).manual_seed(172903))
                def heavy(mod=mod,arm=arm,p=p13 if first else p2,first=first,sides=sides,fork=fork,joins=joins):
                    parent=torch.cuda.current_stream()
                    if arm==0:mod.br_run(p,26 if first else 21,1056 if m<=128 else 2112,m,parent.cuda_stream)
                    else:
                        fork.record(parent)
                        for side,join,n in zip(sides,joins,(64,128,192,256)):
                            side.wait_event(fork)
                            with torch.cuda.stream(side):mod.oc_run(p,n,first,1 if arm==3 or torch.cuda.get_device_capability()!=(9,0) else 4,sms*4,m,side.cuda_stream);join.record()
                        for join in joins:parent.wait_event(join)
                phases.append(capture(heavy))
            for first in (1,0):
                if not first:h13[10].normal_(generator=torch.Generator(device=h13[10].device).manual_seed(172903))
                grid=sms*mod.pmz_resident(2,m,first)
                def lite(mod=mod,p=pl13 if first else pl2,first=first,pn=pn,grid=grid,light=light,ljoin=ljoin,fork=fork):
                    parent=torch.cuda.current_stream();fork.record(parent);light.wait_event(fork)
                    with torch.cuda.stream(light):mod.br_run(p,70 if first else 71,sms*4,m,light.cuda_stream);ljoin.record()
                    mod.pmz_run(pn,first,8,m,2,grid,parent.cuda_stream);parent.wait_event(ljoin)
                phases.append(capture(lite))
            # Reset W2 input after the last W13 capture's eager replay.
            h13[10].normal_(generator=torch.Generator(device=h13[10].device).manual_seed(172903))
            graphs.append(phases)
        # Replay W13 timing before W2; every arm starts W2 with the same ACT.
        times=[[],[],[],[]];ratios=[[],[],[],[]]
        for phase in range(4):
            if phase in (1,3):
                for p in plans:p['owners'][0][10].normal_(generator=torch.Generator(device=p['out'].device).manual_seed(172903))
            a,_=op_pair(graphs[0][phase],graphs[0][phase],ev);times[0].append(a);ratios[0].append(1.)
            for arm in range(1,4):
                a,b=op_pair(graphs[0][phase],graphs[arm][phase],ev);times[arm].append(b);ratios[arm].append(b/a)
        for arm,name in enumerate(OP_NAMES):
            gu=0. if gather[arm] is None else op_pair(gather[arm],gather[arm],ev)[0]
            result.append('OC_PHASE M%d %s H13/H2/L13/L2_us=%s ratios=%s gather_us=%.2f'%(m,name,'/'.join('%.2f'%x for x in times[arm]),'/'.join('%.3f'%x for x in ratios[arm]),gu))
        del graphs,plans,stream_owners,gather
    return result
