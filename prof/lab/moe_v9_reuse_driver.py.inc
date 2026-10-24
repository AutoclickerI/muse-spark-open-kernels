import base64
import hashlib
import statistics
import time
import zlib
import torch

REUSE_CHOICES=((2,3,1),(4,3,1),(1,0,2),(2,0,2),(4,0,2),(8,0,2),(16,0,2),
               (4,1,2),(8,1,2),(16,1,2),(32,2,2),
               (8,0,4),(16,0,4),(16,1,4),(32,2,4))


class ReuseGraph:
    def __init__(self,fn,data):
        self.data={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
        fn(self.data);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=fn(self.data)
    def refresh(self,data):
        for k in ('hidden_states','topk_ids','topk_weights'):self.data[k].copy_(data[k])
    def replay(self):self.graph.replay();return self.out


def reuse_pair(old,new,datasets,flush,n=6):
    aa=[];bb=[];rr=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(n):
        old.refresh(datasets[i%len(datasets)]);new.refresh(datasets[i%len(datasets)])
        v=[]
        for g in ((old,new) if i%2==0 else (new,old)):
            flush.zero_();a.record();g.replay();b.record();b.synchronize();v.append(a.elapsed_time(b)*1000)
        x,y=v if i%2==0 else v[::-1];aa.append(x);bb.append(y);rr.append(y/x)
    return statistics.median(aa),statistics.median(bb),statistics.median(rr)


def reuse_label(c):
    if c[1]==3:return 'v9nt'+str(c[0])
    return ('r' if c[1]==0 else 's' if c[1]==1 else 'f')+str(c[0]*8)+'g'+str(c[2])


def reuse_pct(v):
    a=v.detach().cpu().tolist()
    return f'{statistics.median(a):.0f}/{max(a):.0f}'


def reuse_diagnostics(lines,data,bank,choice,flush):
    phases,stamps=rr_reuse_work(data,bank,choice,True)
    for f in phases:f()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in phases:f()
    flush.zero_();graph.replay();torch.cuda.synchronize()
    for p,s in enumerate(stamps):
        rows=s[s[:,5]>0]
        lines.append(f'D {reuse_label(choice)} M={data["hidden_states"].shape[0]} p={p+1} cy busy={reuse_pct(rows[:,2])} dq={reuse_pct(rows[:,3])} mmaA={reuse_pct(rows[:,4])} tasks={int(rows[:,5].sum())} rb={int(rows[:,6].sum())} wMB={int(rows[:,8].sum())/1e6:.1f} aMB={int(rows[:,9].sum())/1e6:.1f}')


def lab(deadline,weights):
    device=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META cold graph paired medians n=6 flush128MiB; per-phase clocks are diagnostic; R0=captured 26d7e05 fixture0 exact IDs, refreshed x/tw; G=public decode generator seed5353 fields={router_seed:5353}; original R0 router seed/fields were not printed, so G is separate']
    # This is the public generator, including all preceding RNG draws, not a lookalike router.
    w=rg_generate_weights(5353,device=device,router_seed=5353)
    w['post_norm_w']=w['post_expert_norm']
    for name in ('router','router_bias','pre_ffn_norm','pre_expert_norm'):
        lines.append('GEN '+name+' sha256='+hashlib.sha256(w[name].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16])
    bank=rr_reuse_prepare(w)
    base=dict(bank,mod=base__module(),cnt=torch.zeros(64,device=device,dtype=torch.int32))
    phase_mod=phase__module('trace')
    choices=[c for c in REUSE_CHOICES if c[1]!=2 or torch.cuda.get_device_capability(device)==(9,0)]
    raw=zlib.decompress(base64.b64decode(CAPTURED_R0));ms=(8,16,32,128,256)
    flush=torch.empty(128<<20,device=device,dtype=torch.uint8)
    cap=torch.cuda.get_device_capability(device)
    if cap[0]>=12:choices=((2,3,1),(4,3,1),(4,0,2),(16,1,2))
    for m in (128,256):
        fresh=[fx_make_fixture(w,m,9000+m+7919*j) for j in range(3)]
        for d in fresh:
            d['config']=dict(d['config'],bs=m,top_k=8,n_experts=256,sharding='ep8')
        start=8*sum(ms[:ms.index(m)])
        captured=torch.tensor(list(raw[start:start+8*m]),device=device,dtype=torch.int32).view(m,8)
        captured=torch.where(captured<32,captured,-1)
        for routing in ('R0','G'):
            if time.monotonic()>deadline-12:
                lines.append(f'BUDGET M={m} route={routing}');continue
            ds=[dict(d,topk_ids=captured.clone()) if routing=='R0' else d for d in fresh]
            counts=torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids']>=0],minlength=32)
            n8=int(((counts+7)//8).sum());active=int((counts>0).sum())
            lines.append(f'ROUTE M={m} {routing} P={int(counts.sum())} E={active} hot={int(counts.max())} nb8={n8} counts='+','.join(map(str,counts.cpu().tolist())))
            if routing=='R0':
                expect={128:(209,18,55),256:(414,21,119)}[m]
                assert (int(counts.sum()),active,int(counts.max()))==expect
            wants=[rr_reuse_fp32(d,w) for d in ds]
            old=ReuseGraph(lambda d:base_kernel_fn(d,base),ds[0])
            # The unmodified baseline schedule also has to pass the same independent FP32 gate.
            for d,want in zip(ds,wants):
                old.refresh(d);got=old.replay();torch.cuda.synchronize();err=rr_reuse_relative(got,want)
                if err>.01:raise RuntimeError(f'V9 FP32 mismatch {m} {routing} {err}')
            scores=[]
            for choice in choices:
                if time.monotonic()>deadline-9:
                    lines.append('BUDGET '+reuse_label(choice));break
                fn=lambda d,c=choice:(base_kernel_fn(d,dict(base,nt=c[0])) if c[1]==3 else rr_reuse_call(d,bank,c))
                graph=ReuseGraph(fn,ds[0]);worst=0.
                for d,want in zip(ds,wants):
                    graph.refresh(d);got=graph.replay();torch.cuda.synchronize();worst=max(worst,rr_reuse_relative(got,want) if bool(torch.isfinite(got).all()) else float("inf"))
                a,b,ratio=reuse_pair(old,graph,ds,flush)
                lines.append(f'T {reuse_label(choice)} M={m} {routing} us={b:.1f} V9={a:.1f} ratio={ratio:.4f} fp32={worst:.5f} '+('PASS' if worst<.01 else 'REJECT'))
                if worst<.01:scores.append((b,choice))
            if scores and routing=='R0' and cap==(9,0):
                best,win=min(scores);lines.append(f'WIN M={m} R0 {reuse_label(win)} {best:.1f}us')
                reuse_scores=[(us,c) for us,c in scores if c[1]!=3]
                best_reuse=min(reuse_scores)[1]
                reuse_diagnostics(lines,ds[0],bank,best_reuse,flush)
                full=[c for _,c in reuse_scores if c[1]==2]
                if full and best_reuse[1]!=2:reuse_diagnostics(lines,ds[0],bank,full[0],flush)
                # V9 per-CTA clocks after a graph replay and an L2 flush.
                grid=base['grid']*phase_mod.resident_blocks(1,2,m)
                trace=torch.empty((grid,14),device=device,dtype=torch.int64)
                def traced():return phase__kernel_v9(ds[0],base,module=phase_mod,trace=trace)
                traced();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):traced()
                flush.zero_();g.replay();torch.cuda.synchronize()
                lines.append('D V9 M='+str(m)+' cycles '+','.join(name+'='+reuse_pct(trace[:,i]) for i,name in enumerate(('route','w13','wait','bar','w2','norm','combine','w13tiles','w2tiles','atomics')))+f' reloads={int(trace[:,12].sum())} nb8={n8} uniqueWeightMB={active*768*36864/1e6:.1f} tileWeightMB={n8*768*36864/1e6:.1f}')
                lines.append('D V9 M='+str(m)+' perCTA busy='+reuse_pct(trace[:,1]+trace[:,4]-trace[:,2])+' total='+reuse_pct(trace[:,11]-trace[:,10])+' starts='+','.join(map(str,trace[:4,10].cpu().tolist()))+' ends='+','.join(map(str,trace[:4,11].cpu().tolist())))
                # Cold phase medians for the winner, distinct from CTA clock diagnostics.
                fs,_=rr_reuse_work(ds[0],bank,best_reuse)
                for f in fs:f()
                times=[]
                for f in fs:
                    pg=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(pg):f()
                    t=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    for _ in range(6):flush.zero_();a.record();pg.replay();b.record();b.synchronize();t.append(a.elapsed_time(b)*1000)
                    times.append(statistics.median(t))
                lines.append(f'PHASE {reuse_label(best_reuse)} M={m} R0 route,w13,w2,norm,combine_us='+','.join(f'{t:.1f}' for t in times))
    return lines
