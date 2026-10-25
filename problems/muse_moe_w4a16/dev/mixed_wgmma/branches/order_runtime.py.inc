OP_MODS=[]
OP_ARM=0
OP_NAMES=('S12','PERMW','ORDER','PERMW_ORDER')


def op_build():
    if OP_MODS:return
    arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
    for c in OP_CUDA:
        mod=load_inline(name='nf_m256_'+hashlib.sha256(c.encode()).hexdigest()[:12],
                        cpp_sources=OP_CPP,cuda_sources=c,
                        functions=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run'],
                        no_implicit_headers=True,with_pytorch_error_handling=False,
                        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
        mod.br_init();NF_OPTS['pmz']=mod.pmz_init();OP_MODS.append(mod)


def nf_module():return OP_MODS[OP_ARM]


def op_prepare(w):
    w=dict(w,post_norm_w=w['post_expert_norm'])
    base=m128_hh_issue_prepare(nf_ilv(w));native=m128_hh_issue_nt_prepare(w)
    perm=nf_perm(nf_wide(dict(base)))
    return dict(bases=(base,perm,base,perm),native=native,
                streams=[(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event()) for _ in OP_NAMES])


def op_plan(data,bank,arm):
    global OP_ARM
    OP_ARM=arm
    return nf_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm])


def op_call(data,bank,arm):
    for f in op_plan(data,bank,arm)['steps']:out=f()
    return out


def op_time(g,ev):
    lab_cold_l2();ev[0].record();g.replay();ev[1].record();ev[1].synchronize()
    return ev[0].elapsed_time(ev[1])*1000.


def op_pair(a,b,ev):
    aa=[];bb=[]
    for i in range(8):
        for g,dst in (((a,aa),(b,bb)) if i%2==0 else ((b,bb),(a,aa))):dst.append(op_time(g,ev))
    return statistics.median(aa),statistics.median(bb)


def op_error(out,want):
    d=out.float()-want.float();n=want.float().norm(dim=-1)
    return float(d.norm()/want.float().norm().clamp_min(1e-12)),float((d.norm(dim=-1)/n.clamp_min(n.mean()*.1+1e-12)).max())


def lab(deadline,weights,state=None):
    started=time.monotonic();op_build();rows=[];aa=[];peak=0.
    ev=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
    for seed in range(16):
        if deadline-time.monotonic()<max(10.,peak*1.3+1.):break
        tick=time.monotonic();w=dc_weights(5353+7919*seed,next(t.device for t in weights.values() if torch.is_tensor(t)))
        bank=op_prepare(w)
        for m in (128,256):
            ds=dc_fixtures(w,m);wants=dc_oracles(ds,w)
            graphs=[MWGraph(lambda d,a=a:op_call(d,bank,a),ds[0]) for a in range(4)]
            for fi,(d,want) in enumerate(zip(ds,wants)):
                for g in graphs:g.refresh(d)
                outs=[g.replay().clone() for g in graphs];torch.cuda.synchronize()
                errors=[op_error(o,want) for o in outs];bits=[torch.equal(o,outs[0]) for o in outs]
                a,b=op_pair(graphs[0],graphs[0],ev);aa.append(b-a);us=[a];rat=[1.]
                for g in graphs[1:]:
                    a,b=op_pair(graphs[0],g,ev);us.append(b);rat.append(b/a)
                for i,g in enumerate(graphs):
                    e=op_error(g.replay(),want);errors[i]=tuple(max(a,b) for a,b in zip(errors[i],e))
                rows.append(dict(m=m,seed=seed,fixture=fi,us=us,ratio=rat,errors=errors,bits=bits))
            del graphs
        peak=max(peak,time.monotonic()-tick);del bank,w;torch.cuda.empty_cache()
    lines=['META exact S12 final 240248b2 vs PERMW, ORDER, PERMW_ORDER; same N32 tile math, geometry and global weight reads. ORDER changes only (expert,token-group,channel) to (expert,channel,token-group), shortening inter-CTA weight reuse distance.','GATE probes removed: all four arms compute the real output, FP32 TASK bounds .015/.075; cold read, AB/BA x8; same seeds/routes.']
    for m in (128,256):
        rr=[r for r in rows if r['m']==m]
        if not rr:continue
        for a,name in enumerate(OP_NAMES):
            pair=[r['ratio'][a] for r in rr];e=max(r['errors'][a][0] for r in rr);rw=max(r['errors'][a][1] for r in rr)
            lines.append(f'M{m} {name} n={len(rr)} us={statistics.mean(r["us"][a] for r in rr):.2f} pair={statistics.mean(pair):.4f} lo/hi={min(pair):.3f}/{max(pair):.3f} L2/row={e:.6f}/{rw:.6f} TASK={"PASS" if e<=.015 and rw<=.075 else "FAIL"} bit={sum(r["bits"][a] for r in rr)}')
    if aa:lines.append('AA signed/abs/max=%.3f/%.3f/%.3f'%(statistics.mean(aa),statistics.mean(abs(x) for x in aa),max(abs(x) for x in aa)))
    lines.append(f'END seeds={len(rows)//6} used={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    return lines
