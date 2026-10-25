"""Globaltimer evidence for the gate split's missing overlap; not a promotion lab."""


def gt_graph(fn):
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn();fn()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=fn()
    graph._trace_stream=stream
    graph.replay();torch.cuda.synchronize()
    return graph,out


class GtChain(GsChain):
    def __init__(self,v,attn,x,gs):
        super().__init__(v,attn,x,gs)
        self.stamps=torch.empty((gs[1]*16,8),device=x.device,dtype=torch.int64)
        self.y=torch.randn((x.shape[0],1024),device=x.device).bfloat16()

    def gate(self,y):
        t=self.x.shape[0];bn,s,ns=self.gs
        a=torch.empty_like(y)
        _gt_kernel[(1024//bn,s)](self.x,self.v['gate'],self.P,self.cnt,y,a,self.stamps,
            t,self.x.stride(0),8192,s,N=1024,BN=bn,BK=128,TB=max(16,triton.next_power_of_2(t)),
            NS=ns,num_warps=4,launch_pdl=True)
        return a

    def __call__(self):
        p,s=ro_bal(self.x,self.v['qkv3'],64,10)
        y=self.attn(ro_sumcast(p,s),gs=True)
        a=self.gate(y)
        o,so=ro_s10(a,self.v['o'],(64,512,128))
        h,_=ro_boundary(self.v,o,so,8192,False)
        _ro_sink[(len(self.x),)](h,self.sink,N=8192,NP=8192,launch_pdl=True,num_warps=8)
        return a,h


def gt_span(stamps):
    v=stamps.cpu()
    starts=v[:,0];valid=starts>0
    if not bool(valid.any()):return (0,0)
    return int(starts[valid].min()),int(v[valid,1].max())


def gt_one(chain,probe,graph,isolated):
    lab_cold_l2();graph.replay();torch.cuda.synchronize()
    attention=[gt_span(p) for p in probe]
    p=chain.stamps.cpu()
    base=min(int(p[:,0].min()),min((a for a,b in attention if a),default=int(p[:,0].min())))
    last=p[:,3]==chain.gs[1]-1
    result=dict(attn_start=min((a for a,b in attention if a),default=base),
                attn_end=max((b for a,b in attention if b),default=base),
                gate_start=int(p[:,0].min()),gate_dot_end=int(p[:,1].max()),
                gate_wait_start=int(p[last,4].min()),gate_wait_end=int(p[last,5].max()),
                gate_end=int(p[last,6].max()))
    return {k:(v-base)/1000 for k,v in result.items()}


def lab(deadline,weights):
    started=time.monotonic();dev=torch.device('cuda')
    weights=_gs_weights(weights,dev);gen=torch.Generator(device=dev).manual_seed(13)
    lines=['META gate overlap globaltimer, 8 cold replays; ns->us; attention stamps thread0 entry/return per CTA '
           '(return is a proxy for CTA completion); gate stamps entry/dot end/last-split wait start/end/store end; '
           'ISO=gate alone; CHAIN=qkv3->ungated attention->gate->O->b1; diagnostics only']
    for b,w in GS_SHAPES:
        if time.monotonic()>deadline-12:
            lines.append(f'SKIP b={b} w={w}');continue
        try:
            v=_ro_vectors(weights,dev,gen,b)
            v['qkv3'],v['gate']=v['qkv'][:1280],v['qkv'][1280:]
            attn=RoAttention(b,w,dev)
            ext,ns,cs,_,_,_=attn.plan(True)
            probe=torch.zeros((2,65536,2),device=dev,dtype=torch.int64)
            ext.gs_set_trace(probe.data_ptr())
            x=(torch.randn((b,8192),device=dev,generator=gen)*.5).bfloat16()
            ch=GtChain(v,attn,x,(64,4,4))
            isolated_ch=GtChain(v,attn,x,(64,4,4))
            # GsChain owns ordinary (not graph-pool) P/counter allocations.
            # Keep each owner alive while a captured graph uses its pointers.
            control_ch=GsChain(v,attn,x,None)
            untraced_ch=GsChain(v,attn,x,(64,4,4))
            chain,traced_out=gt_graph(ch)
            iso,isolated_out=gt_graph(lambda:isolated_ch.gate(isolated_ch.y))
            ctrl,_=gt_graph(control_ch)
            untraced,untraced_out=gt_graph(untraced_ch)
            paired=_ro_paired([ctrl,untraced,chain],[None,None,None],12)
            diffs=[(float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)),
                    float((a.float()-b.float()).abs().max())) for a,b in zip(traced_out,untraced_out)]
            assert all(torch.equal(a,b) for a,b in zip(traced_out,untraced_out)), f'trace changed output {diffs}'
            chain_rows=[gt_one(ch,probe,chain,False) for _ in range(8)]
            iso_rows=[gt_one(isolated_ch,probe,iso,True) for _ in range(8)]
            avg=lambda rows,k:statistics.median(r[k] for r in rows)
            fields=' '.join(f'{k}={avg(chain_rows,k):.3f}' for k in chain_rows[0])
            iso_dot=statistics.median(r['gate_dot_end']-r['gate_start'] for r in iso_rows)
            iso_end=statistics.median(r['gate_end']-r['gate_start'] for r in iso_rows)
            lines.append(f'TRACE b={b} w={w} {attn.kind} ns/cs={ns}/{cs} '+fields+
                         f' ISOdot={iso_dot:.3f} ISOend={iso_end:.3f} '
                         f'ctrl={paired[0][0]:.3f} gs_r={paired[1][1]:.5f} trace_r={paired[2][1]:.5f}')
            del chain,iso,ctrl,untraced,ch,isolated_ch,control_ch,untraced_ch,isolated_out,attn,v,probe
        except Exception as exc:
            lines.append(f'ERROR b={b} w={w} {type(exc).__name__}:{str(exc)[:160]}')
        torch.cuda.empty_cache()
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s left={deadline-time.monotonic():.1f}s')
    return lines
