import time
import statistics


@functools.lru_cache(None)
def repair_module():
    arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
    mod=load_inline(name='m128_repair_'+hashlib.sha256(REPAIR_CUDA.encode()).hexdigest()[:12],
        cpp_sources=REPAIR_CPP,cuda_sources=REPAIR_CUDA,functions=REPAIR_FUNCTIONS,no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False)
    mod.rot64_init();return mod


class RepairABI:
    def __init__(self,name):self.mod=repair_module();self.name=name
    def ft_route(self,ids,c,off,order,task,slots,pos,state,out,nt,start,clear):
        ptrs=[v.data_ptr() for v in (ids,c,off,order,task,slots,pos,state,out)]
        return self.mod.rot64_route(ptrs,ids.shape[0],nt,start,clear,torch.cuda.current_stream().cuda_stream)
    def ft_run(self,v,first,which,grid,start):
        return self.mod.rot64_run([t.data_ptr() for t in v],first,which,grid,start,v[11].shape[0],v[9].shape[0],torch.cuda.current_stream().cuda_stream)


REPAIR_CHOICES=(('rot3a','rot64',3,264),('rot3b','rot64',3,528),('rot3c','rot64',3,1056),
    ('rot5a','rot64',5,264),('rot5b','rot64',5,528),('rot5c','rot64',5,1056),('ctrl4','rot64',4,528))


def repair_call(data,bank,which,grid):
    for f in repair_plan(data,bank,which,grid)['steps']:out=f()
    return out


def repair_rows(plan):
    c=plan['v13'][4].cpu().tolist();off=plan['v13'][5].cpu().tolist()
    return torch.cat([torch.arange(off[e],off[e]+c[e],device=plan['v13'][3].device) for e in range(32)])


def repair_stage_probe(data,bank,which,grid):
    p=repair_plan(data,bank,which,grid);fs=p['steps'];fs[0]();rows=repair_rows(p)
    # Poison scratch before the producers run: an omitted write stays visible.
    for index in (9,15,16,17):p['v13'][index].fill_(float('nan'))
    fs[1]();torch.cuda.synchronize()
    a=p['v13'][9][rows];nf_a=int((~torch.isfinite(a)).sum())
    out=fs[2]();torch.cuda.synchronize();nf_y=int((~torch.isfinite(p['v2'][15][rows])).sum())
    nf_i=int((~torch.isfinite(p['v2'][17][rows])).sum());nf_o=int((~torch.isfinite(out)).sum())
    return f'{nf_a}/{nf_y}/{nf_i}/{nf_o}'


def repair_measure(w,bank,ds,flush,probe=False):
    wants=dc_oracles(ds,w);graphs=[];errs=[];banks=[];stages=[]
    for name,abi,which,grid in REPAIR_CHOICES:
        b=dict(bank,mod=RepairABI(abi));banks.append(b)
        if probe:stages.append(name+'='+repair_stage_probe(ds[0],b,which,grid))
        g=MWGraph(lambda d,b=b,c=which,z=grid:repair_call(d,b,c,z),ds[0]);graphs.append(g)
        worst=row=0.
        for d,want in zip(ds,wants):
            g.refresh(d);got=g.replay();torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()):worst=row=float('inf');continue
            delta=got.float()-want.float();den=want.float().norm(dim=-1)
            worst=max(worst,float(delta.norm()/want.float().norm().clamp_min(1e-12)))
            row=max(row,float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max()))
        errs.append((worst,row))
    ts=[[] for g in graphs];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for repeat in range(4):
        order=list(range(len(graphs)))
        if repeat%2:order.reverse()
        for j in order:
            if errs[j][0]>.01 or errs[j][1]>.05:continue
            graphs[j].refresh(ds[repeat%len(ds)]);flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();ts[j].append(a.elapsed_time(b)*1000.)
    return [(statistics.median(v) if v else float('inf'),*e) for v,e in zip(ts,errs)],stages


def lab(deadline,weights,state=None):
    start=time.monotonic();dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META K64 register-A lifetime repair: cfg3/cfg5 alternate two banks across stages; cfg4 K128 control. Common route/reset, packed bank, B128 descriptors and W2 epilogue unchanged. Pointer-only ABI reuses shared native_abi.py; no implicit Torch headers. Old unsafe source is frozen in 66133279; local SASS model finds 63 lifetime violations per K64 path, fixed zero.',
           'META exact R0 + HOT128 (single expert) + 12 public-router weight seeds, three fresh captured inputs. FP32 finite,L2<=.01,row<=.05; four cold128MiB replays. Stage fields=count of nonfinite ACT/Y/INV/OUT after poisoning producer scratch. ORDER '+','.join(c[0] for c in REPAIR_CHOICES)]
    if time.monotonic()>deadline-40:return lines+['INCOMPLETE before_compile (need 40s allowance for pointer-only build)']
    mod=repair_module();lines.append(f'COMPILE_DONE elapsed={time.monotonic()-start:.1f} left={deadline-time.monotonic():.1f}')
    for prefix,which in (('rot64',3),('rot64',4),('rot64',5)):
        lines.append(f'CFG {prefix}/{which} W13={getattr(mod,prefix+"_info")(which,1)} W2={getattr(mod,prefix+"_info")(which,0)}')
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8);records=[]
    for i in range(12):
        if time.monotonic()>deadline-10:lines.append(f'INCOMPLETE before_seed={i}');break
        w=dc_weights(5353+7919*i,dev);bank=ft_prepare(w)
        if i==0:
            ds=dc_fixtures(w,128,True);v,st=repair_measure(w,bank,ds,flush,True)
            lines.append('R0 us='+','.join(f'{x[0]:.0f}' for x in v)+' gate='+','.join(f'{x[1]:.4f}' for x in v));lines.append('R0_STAGE '+','.join(st))
            if time.monotonic()>deadline-10:lines.append('INCOMPLETE before_hot');break
            ds=dc_fixtures(w,128)
            for d in ds:
                ids=torch.full_like(d['topk_ids'],-1);ids[:,0]=0;d['topk_ids']=ids
            v,st=repair_measure(w,bank,ds,flush,True)
            lines.append('HOT128 gate='+','.join(f'{x[1]:.4f}' for x in v));lines.append('HOT_STAGE '+','.join(st))
        if time.monotonic()>deadline-8:lines.append(f'INCOMPLETE before_seed_measure={i}');break
        ds=dc_fixtures(w,128);v,_=repair_measure(w,bank,ds,flush)
        c=torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids']>=0],minlength=32)
        lines.append(f'S{i} E/h={int((c>0).sum())}/{int(c.max())} us='+','.join(f'{x[0]:.0f}' for x in v)+' gate='+','.join(f'{x[1]:.4f}' for x in v));records.append(v)
    for j,(name,abi,which,grid) in enumerate(REPAIR_CHOICES):
        if records:
            vs=[v[j] for v in records];ok=all(e<=.01 and r<=.05 for t,e,r in vs)
            lines.append(f'AGG {name} n={len(vs)} us={statistics.mean(v[0] for v in vs):.1f} L2={max(v[1] for v in vs):.5f} row={max(v[2] for v in vs):.5f} '+('PASS' if ok else 'REJECT'))
    lines.append(f'END elapsed={time.monotonic()-start:.1f} left={deadline-time.monotonic():.1f}')
    return lines
