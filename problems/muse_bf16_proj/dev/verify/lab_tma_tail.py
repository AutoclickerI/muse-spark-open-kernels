import triton
import triton.language as tl


@triton.jit
def vp_boundary_kernel(A,S,H,NEW,ALPHA,BETA,PRE,M:tl.constexpr,USE_PDL:tl.constexpr):
    if USE_PDL:tl.extra.cuda.gdc_launch_dependents()
    r=tl.program_id(0);c=tl.arange(0,8192)
    alpha=tl.load(ALPHA+c);beta=tl.load(BETA+c);pre=tl.load(PRE+c)
    if USE_PDL:tl.extra.cuda.gdc_wait()
    branch=tl.load(A+r.to(tl.int64)*8192+c).to(tl.float32)
    stream=tl.load(S+r.to(tl.int64)*8192+c)
    branch=branch*tl.rsqrt(tl.sum(branch*branch,0)/8192.+1e-8)
    new=alpha*stream+beta*branch
    tl.store(NEW+r.to(tl.int64)*8192+c,new)
    h=new*tl.rsqrt(tl.sum(new*new,0)/8192.+1e-5)*pre
    tl.store(H+r.to(tl.int64)*8192+c,h.to(tl.bfloat16))


def vp_chain(x,stream,wo,wp,alpha,beta,pre,co,cp,pdl,native):
    a=torch.mm(x,wo.T) if co is None else vp_tma(x,wo,co,torch.bfloat16,pdl,native)
    h=torch.empty((x.shape[0],8192),device=x.device,dtype=torch.bfloat16)
    new=torch.empty_like(stream)
    vp_boundary_kernel[(x.shape[0],)](a,stream,h,new,alpha,beta,pre,M=x.shape[0],USE_PDL=pdl,
        launch_pdl=pdl,num_warps=8)
    y=torch.mm(h,wp.T,out_dtype=torch.float32) if cp is None else vp_tma(h,wp,cp,torch.float32,pdl,native)
    return y


def vp_unique(m,n,k,sms):
    seen=set();configs=[]
    for c in vp_tma_choices(m,n,k):
        bt,ns,ncw,st,mc,ks,waves,pipe=c;br=64*ns*ncw
        jobs=(n//br//mc)*((m+bt-1)//bt)
        groups=jobs if not waves else min(jobs,max(1,sms*waves//(mc*ks)))
        canonical=(bt,ns,ncw,st,mc,ks,groups,pipe)
        if canonical not in seen:seen.add(canonical);configs.append(c)
    return configs


def vp_tma_lab(deadline,weights,local=False):
    rows=[];native=vp_tma_module();dev=weights['qkv_gate'].device
    router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
    ws={name:weights[name] for name in ('qkv_gate','o','unpooler')}
    ws['pool_router']=torch.cat((weights['pooler'],hi,lo),0).contiguous()
    flush=torch.zeros((128<<20)//4,device=dev,dtype=torch.float32)
    gen=torch.Generator(device=dev).manual_seed(772410)
    sms=torch.cuda.get_device_properties(dev).multi_processor_count
    keys=[(m,name) for m in (256,128) for name in ('pool_router','qkv_gate','unpooler','o')]
    rows.append(f'VP_TMA_META sms={sms} cold128MiB graph fresh3; raw_us; cfg=(BT,NS,WG,ST,MC,KS,waves,async)')
    winners={};measurements=[]
    for pos,(m,name) in enumerate(keys):
        now=time.monotonic();stop=now+max(0,deadline-now-10)/(len(keys)-pos)
        if now>=deadline-10:rows.append(f'VP_TMA_BUDGET {m} {name}');continue
        coarse_stop=now+.7*max(0,stop-now)
        w=ws[name];n,k=w.shape;dtype=torch.float32 if name=='pool_router' else torch.bfloat16
        banks=[torch.randn(m,k,device=dev,dtype=torch.bfloat16,generator=gen) for _ in range(3)]
        x=banks[0].clone();refs=[torch.mm(b.float(),w.float().T) for b in banks]
        old=_PgGraph(lambda:torch.mm(x,w.T,out_dtype=torch.float32) if dtype==torch.float32 else torch.mm(x,w.T))
        configs=vp_unique(m,n,k,sms);cost=[];table=[];bad={}
        for index,c in enumerate(configs):
            if time.monotonic()>coarse_stop:break
            graph=None
            try:
                graph=_PgGraph(lambda c=c:vp_tma(x,w,c,dtype,False,native))
                for b in range(3):x.copy_(banks[b]);graph.graph.replay();vp_check(graph.out,refs[b])
                t0,t1,ratio=vp_pair(old,graph,x,banks,flush,4)
                cost.append((ratio,index,t1));table.append(round(t1,1))
            except Exception as exc:
                reason=type(exc).__name__+':'+str(exc).splitlines()[0][:80]
                if reason not in bad and len(bad)<2:rows.append(f'VP_TMA_REJECT {m} {name} i={index} {reason}')
                bad[reason]=bad.get(reason,0)+1;table.append('x')
            finally:
                if graph is not None:graph.close()
        finalists=sorted(cost)[:4]
        # Keep a multicast and a non-multicast finalist where available.
        for mc in (1,2):
            family=[r for r in cost if configs[r[1]][4]==mc]
            if family and min(family) not in finalists:finalists.append(min(family))
        best=(float('inf'),None,None);details=[]
        for _,index,_ in finalists:
            if time.monotonic()>min(stop,deadline-8):break
            graph=None;c=configs[index]
            try:
                graph=_PgGraph(lambda c=c:vp_tma(x,w,c,dtype,False,native))
                t0,t1,ratio=vp_pair(old,graph,x,banks,flush)
                for b in range(3):x.copy_(banks[b]);graph.graph.replay();vp_check(graph.out,refs[b])
                details.append((index,round(t0,2),round(t1,2),round(ratio,4)))
                if ratio<best[0]:best=(ratio,c,(t0,t1))
            except Exception as exc:rows.append(f'VP_TMA_REJECT final {m} {name} {index} {type(exc).__name__}:{str(exc)[:70]}')
            finally:
                if graph is not None:graph.close()
        winners[(m,name)]=best[1]
        rows.append(f'VP_TMA_BEST {m} {name} pick={best[1]} raw_us={best[2]} ratio={best[0]:.4f} final={details} rejected={sum(bad.values())}')
        # Retain each tile/ring family's best coarse result. Printing every trial
        # would crowd the final keys out of the lab's9K output allowance.
        family_best={}
        for ratio,index,us in cost:
            family=configs[index][:5]
            if family not in family_best or ratio<family_best[family][0]:family_best[family]=(ratio,index,us)
        measurements.append((m,name,[(i,round(us,1)) for _,i,us in family_best.values()]))
        old.close();del banks,x,refs,old,configs,cost
    # Measure PDL only across a real projection/boundary/projection dependency chain.
    for m in (256,128):
        if time.monotonic()>deadline-8:rows.append(f'VP_CHAIN_BUDGET {m}');continue
        co=winners.get((m,'o'));cp=winners.get((m,'pool_router'))
        if co is None or cp is None:continue
        x=torch.randn(m,1024,device=dev,dtype=torch.bfloat16,generator=gen)
        banks=[torch.randn_like(x) for _ in range(3)]
        stream=torch.randn(m,8192,device=dev,generator=gen)
        alpha=torch.full((8192,),.8,device=dev);beta=torch.full_like(alpha,.5);pre=torch.ones_like(alpha)
        old=_PgGraph(lambda:vp_chain(x,stream,ws['o'],ws['pool_router'],alpha,beta,pre,None,None,False,native))
        for pdl in (False,True):
            graph=None
            try:
                graph=_PgGraph(lambda:vp_chain(x,stream,ws['o'],ws['pool_router'],alpha,beta,pre,co,cp,pdl,native))
                vp_check(graph.out,old.out)
                t0,t1,ratio=vp_pair(old,graph,x,banks,flush)
                vp_check(graph.out,old.out)
                rows.append(f'VP_CHAIN {m} pdl={pdl} raw_us={t0:.2f}/{t1:.2f} ratio={ratio:.4f} co={co} cp={cp}')
            except Exception as exc:rows.append(f'VP_CHAIN_REJECT {m} {pdl} {type(exc).__name__}:{str(exc)[:100]}')
            finally:
                if graph is not None:graph.close()
        old.close()
    for m,name,table in measurements:rows.append(f'VP_TMA_FAMILY {m} {name} index_us='+str(table).replace(' ',''))
    rows.append(f'VP_TMA_END left_s={deadline-time.monotonic():.2f}')
    return rows


def lab(deadline,weights,state=None):
    # The native portable reader and all Hopper ABI families are validated separately
    # on sm120; this host-screen branch keeps the lab within prepare's300-second cap.
    if torch.cuda.get_device_capability()[0]>=12:
        native=vp_tma_module();x=torch.randn(33,1024,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(512,1024,device='cuda',dtype=torch.bfloat16)
        for c in ((64,1,1,4,1,1,0,1),(64,1,1,4,2,2,1,1)):
            with _PgGraph(lambda:vp_tma(x,w,c,torch.float32,False,native)) as g:
                vp_check(g.out,x.float()@w.float().T)
        return ['VP_LOCAL portable TMA/cluster graph math passes; full Hopper families forced by dev/verify/validate.py']
    return vp_tma_lab(deadline,weights)
