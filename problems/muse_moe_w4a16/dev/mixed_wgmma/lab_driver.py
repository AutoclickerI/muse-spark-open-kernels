"""Exact captured core IDs, public decode weights/router for fresh x and prefill rows."""
import base64
import statistics
import time
import zlib


class MWGraph:
    def __init__(self,fn,data):
        self.data={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
        fn(self.data);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=fn(self.data)
    def refresh(self,data):
        for k in ('hidden_states','topk_ids','topk_weights'):self.data[k].copy_(data[k])
    def replay(self):
        self.graph.replay()
        return self.out


def mw_pair(old,new,ds,flush):
    aa=[];bb=[];rr=[]
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for j in range(6):
        old.refresh(ds[j%len(ds)]);new.refresh(ds[j%len(ds)])
        pair=[]
        for g in ((old,new) if j%2==0 else (new,old)):
            flush.zero_();a.record();g.replay();b.record();b.synchronize()
            pair.append(a.elapsed_time(b)*1000.)
        x,y=pair if j%2==0 else pair[::-1]
        aa.append(x);bb.append(y);rr.append(y/x)
    return statistics.median(aa),statistics.median(bb),statistics.median(rr)


def mw_gate(graph,ds,wants):
    worst=0.;row=0.
    for d,want in zip(ds,wants):
        graph.refresh(d);got=graph.replay();torch.cuda.synchronize()
        diff=got.float()-want.float()
        err=float(diff.norm()/want.float().norm().clamp_min(1.e-12))
        den=want.float().norm(dim=-1)
        erow=float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1).clamp_min(1.e-12)).max())
        if not bool(torch.isfinite(got).all()):err=float('inf')
        worst=max(worst,err);row=max(row,erow)
    return worst,row


def mw_phase_times(phases,flush):
    times=[]
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for f in phases:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):f()
        vals=[]
        for _ in range(4):
            flush.zero_();a.record();graph.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000.)
        times.append(statistics.median(vals))
    return ','.join(f'{t:.1f}' for t in times)


def lab(deadline,weights,state=None):
    dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META TileLang0.1.12 RS-WGMMA NVFP4/E4M3 converted in timed GEMMs; FP32 GS epilogue; paired raw graph us n6 cold128MiB; R0=26d7e05 fixture0 exact IDs, fresh x/public decode weights5353; G=public router5353, q_len1 for prefill; FP32 gate L2<=.01 row<=.05']
    w=rg_generate_weights(5353,device=dev,router_seed=5353)
    w['post_norm_w']=w['post_expert_norm']
    bank=mw_prepare(w);native=ctrl_prepare(w);marlin=marlin_prepare(w)
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    raw=zlib.decompress(base64.b64decode(CAPTURED_R0))
    sms=torch.cuda.get_device_properties(dev).multi_processor_count
    core=((64,16,128,2,sms*2,True),(64,32,128,2,sms*2,True),
          (64,16,128,0,sms,True),(64,16,128,2,sms*2,False))
    wide=(core[1],core[0])
    for m in (128,256,512,753,3012,16384):
        if time.monotonic()>deadline-15:
            lines.append(f'BUDGET M={m} before fixtures');break
        ds=[fx_make_fixture(w,m,9000+m+7919*j) for j in range(3)]
        route='R0' if m in (128,256) else 'G'
        if route=='R0':
            start=8*sum((8,16,32) if m==128 else (8,16,32,128))
            captured=torch.tensor(list(raw[start:start+8*m]),device=dev,dtype=torch.int32).view(m,8)
            for d in ds:d['topk_ids']=torch.where(captured<32,captured,255).clone()
        for d in ds:
            d['topk_ids']=torch.where(d['topk_ids']<0,255,d['topk_ids'])
            d['config']=dict(d['config'],bs=m,n_experts=256,top_k=8,sharding='ep8')
        ids=ds[0]['topk_ids'];counts=torch.bincount(ids[ids<32],minlength=32)
        summary=(int(counts.sum()),int((counts>0).sum()),int(counts.max()))
        if route=='R0':assert summary=={128:(209,18,55),256:(414,21,119)}[m]
        lines.append(f'ROUTE {route} M={m} pairs={summary[0]} E={summary[1]} hot={summary[2]}')
        wants=[reuse_fp32(d,w) for d in ds]
        # V9's fused shared-memory routing cannot hold a 3K/16K batch. Marlin
        # is the relevant serving-size control, rather than an invalid V9 launch.
        old_label='V9' if m<=753 else 'Marlin'
        old=MWGraph((lambda d:ctrl_kernel_fn(d,native)) if m<=753 else
                    (lambda d:marlin_kernel_fn(d,marlin)),ds[0])
        err,row=mw_gate(old,ds,wants)
        lines.append(f'GATE {old_label} M={m} fp32={err:.5f} row={row:.5f}')
        if err>.01 or row>.05:raise ValueError(f'{old_label} agreement {m} {err} {row}')
        mg=MWGraph(lambda d:marlin_kernel_fn(d,marlin),ds[0])
        err,row=mw_gate(mg,ds,wants)
        a,b,ratio=mw_pair(old,mg,ds,flush)
        lines.append(f'T Marlin M={m} us={b:.1f} {old_label}={a:.1f} ratio={ratio:.4f} fp32={err:.5f} row={row:.5f} '+('PASS' if err<=.01 and row<=.05 else 'REJECT'))
        del mg
        winner=None
        for choice in (core if m<=256 else wide):
            if time.monotonic()>deadline-12:
                lines.append(f'BUDGET M={m} before choice={choice}');break
            try:
                graph=MWGraph(lambda d,c=choice:mw_call(d,bank,c),ds[0])
                err,row=mw_gate(graph,ds,wants)
                a,b,ratio=mw_pair(old,graph,ds,flush)
                valid=err<=.01 and row<=.05
                label=f'{"RS" if choice[-1] else "SS"}n{choice[1]}k{choice[2]}s{choice[3]}g{choice[4]}'
                lines.append(f'T {label} M={m} us={b:.1f} {old_label}={a:.1f} ratio={ratio:.4f} fp32={err:.5f} row={row:.5f} '+('PASS' if valid else 'REJECT'))
                if valid and (winner is None or b<winner[0]):winner=(b,choice,label)
                del graph
            except Exception as e:
                lines.append(f'ERROR M={m} choice={choice} {type(e).__name__}: {str(e).splitlines()[0][:220]}')
        if winner and time.monotonic()<deadline-8:
            phases=mw_work(ds[0],bank,winner[1])
            for f in phases:f()
            lines.append(f'PHASE {winner[2]} M={m} route,w13,act,w2,norm,combine_us='+mw_phase_times(phases,flush))
            del phases
        del old,wants,ds
    lines.append(f'END left_s={deadline-time.monotonic():.1f}')
    return lines
