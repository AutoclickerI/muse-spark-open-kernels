import gc,json,linecache,math,statistics,time
import torch

VD_LOCAL_INSTALL=None
VD_ROWS=[]
VD_DISABLED=[]


def vd_weights(device):
    g=torch.Generator(device=device).manual_seed(941732)
    out={name:(torch.randn((n,k),device=device,generator=g)/k**.5).bfloat16() for name,(n,k) in
        dict(qkv_gate=(2304,8192),o=(8192,1024),unpooler=(8192,4096)).items()}
    pool=(torch.randn((4096,8192),device=device,generator=g)/8192**.5).bfloat16()
    router=torch.randn((256,8192),device=device,generator=g)/8192**.5
    hi=router.bfloat16();lo=(router-hi.float()).bfloat16();out['pool_router']=torch.cat((pool,hi,lo))
    return out,router,.055*torch.randn(256,device=device,generator=g)


class VDGraph:
    def __init__(self,fn):
        self.stream=torch.cuda.Stream();self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            fn();fn()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.out=fn()
        torch.cuda.current_stream().wait_stream(self.stream)
        self.graph.replay()
    def replay(self):self.graph.replay()


def vd_pair(a,b,flush,rounds=6):
    samples=[[],[]];ratio=[];start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    for i in range(rounds):
        pair=[0.,0.]
        for which in ((0,1) if i%2==0 else (1,0)):
            flush.zero_();start.record();(a if which==0 else b).replay();end.record();end.synchronize()
            pair[which]=1000*start.elapsed_time(end);samples[which].append(pair[which])
        ratio.append(pair[1]/pair[0])
    return dict(a=statistics.median(samples[0]),b=statistics.median(samples[1]),ratio=statistics.median(ratio),samples=samples)


def vd_gate(got,ref,role,router_ref=None,bias=None,control=None):
    af=got.float();rf=ref.float();d=(af-rf).norm(dim=-1)
    norms=rf.norm(dim=-1);err=float(d.norm()/norms.norm().clamp_min(1e-12));row=float((d/norms.clamp_min(.1*norms.mean()).clamp_min(1e-12)).max())
    lim=1e-4 if role=='pool_router' else .01
    if not torch.isfinite(af).all() or err>lim or row>5*lim:raise ValueError(f'projection gate L2={err:.6f} row={row:.6f}')
    result=dict(l2=err,row=row,flips=0)
    if role=='pool_router':
        logits=af[:,4096:4352]+af[:,4352:];le=float((logits-router_ref).norm()/router_ref.norm().clamp_min(1e-12))
        if le>1e-4:raise ValueError(f'router hi+lo gate {le}')
        result['router_l2']=le
        want=router_ref if control is None else control.float()[:,4096:4352]+control.float()[:,4352:]
        a=(logits*.1767766952966369).sigmoid()+bias;b=(want*.1767766952966369).sigmoid()+bias
        ai=a.topk(8,-1).indices;bi=b.topk(8,-1).indices;mask=(ai!=bi).any(-1);result['flips']=int(mask.sum())
        if result['flips']:
            top=b.topk(9,-1).values;gap=top[:,7]-top[:,8];delta=(a-b).abs().amax(-1)
            # Distinguish ordering swaps inside top8 from actual set changes.
            setchange=~(ai[:,:,None]==bi[:,None,:]).any(-1).all(-1)
            near=(~setchange)|((gap<=2*delta+1e-6)&(b.gather(1,ai).amin(-1)>=top[:,7]-2*delta-1e-6))
            result.update(near_tie=bool(near[mask].all()),gap8_9=gap[mask].tolist(),score_delta=delta[mask].tolist(),actual=ai[mask].tolist(),control=bi[mask].tolist())
    return result


def vd_select_lt(w,m,x,fp32,work,flush,deadline,lines):
    candidates=[];attempted=0;rejected=0
    for nn in (0,1):
        op=VDLt(w,m,fp32,nn,workspace=work)
        for index in range(op.count):
            if time.monotonic()>deadline-15:break
            attempted+=1
            try:
                graph=VDGraph(lambda index=index:op(x,index))
                samples=vd_pair(graph,graph,flush,rounds=1)
                candidates.append((samples['a'],nn,index,op,graph))
            except Exception as exc:
                rejected+=1
                if rejected<=2:lines.append(f'LT_REJECT M={m} N={w.shape[0]} nn={nn} index={index} {type(exc).__name__}:{str(exc)[:80]}')
        # Counts are retained in the selected row metadata; output stays bounded.
    if not candidates:raise RuntimeError('no executable Lt candidates')
    candidates.sort(key=lambda r:r[0]);final=[]
    for _,nn,index,op,graph in candidates[:8]:
        score=vd_pair(graph,graph,flush,rounds=3)['a'];final.append((score,nn,index,op))
    final.sort(key=lambda r:r[0]);_,nn,index,op=final[0];op.index=index
    return op,dict(op.selection(),tried=attempted,rejected=rejected,index=index)


def vd_control(w,m,role,x,work):
    # Exact control family/ordinal measured in19f3a13b; independently print
    # best7's measured-Lt registry choices as an extra reference below.
    if (m,role) in ((128,'pool_router'),(256,'qkv_gate')):
        op=VDLt(w,m,role=='pool_router',0,False,workspace=work)
        return op,'best7_lt_TN0'
    class Control:
        def __init__(self):self.out=torch.empty((m,w.shape[0]),device=w.device,dtype=torch.float32 if role=='pool_router' else torch.bfloat16)
        def __call__(self,x):
            torch.mm(x,w.T,out=self.out,out_dtype=torch.float32) if role=='pool_router' else torch.mm(x,w.T,out=self.out)
            return self.out
    return Control(),'best7_cublas'


def lab(deadline,weights,host_state=None):
    global VD_ROWS,VD_DISABLED
    started=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False
    if VD_LOCAL_INSTALL is not None:VD_LOCAL_INSTALL()
    device=weights['w13'].device;banks,router,bias=vd_weights(device)
    # Compilation and weight-only repacks are untimed. Candidate timing is cold.
    vd_library('lt');vd_library('incumbent');vd_library('cutlass')
    work=torch.empty(1<<30,device=device,dtype=torch.uint8);registry_work=torch.empty(4<<30,device=device,dtype=torch.uint8);flush=torch.empty(256<<20,device=device,dtype=torch.uint8)
    lines=['META verify dense: DeepGEMM BF16, enumerated Lt TN/NN, CUTLASS StreamK + cluster persistent; weights BF16, router hi+lo FP32',
           'META M64/128/256 all4 roles; seed941732; same changed inputs; cold256MiB ABBA6 + AA; flips disable their cell, continue']
    lines.append('LEGEND D<M> role(q/o/pr/u) cell(lt/dg/c0/c1/c2/reg) draws A_us/B_us B:A AA_us L2; S=selected Lt; T=role sums')
    lines.append(f'SETUP seconds={time.monotonic()-started:.1f} remain={deadline-time.monotonic():.1f}')
    VD_ROWS=[];VD_DISABLED=[];totals={}
    # Scored verify loads first;64 is the nearby-shape diagnostic.
    for m in (128,256,64):
      for role in ('qkv_gate','o','pool_router','unpooler'):
        if time.monotonic()>deadline-12:
            lines.append(f'INCOMPLETE M={m} role={role}');return lines
        role_started=time.monotonic();w=banks[role];fp32=role=='pool_router'
        x=torch.randn((m,w.shape[1]),device=device,dtype=torch.bfloat16,generator=torch.Generator(device=device).manual_seed(771+m))
        control,label=vd_control(w,m,role,x,work);base=VDGraph(lambda:control(x));aa=vd_pair(base,base,flush)
        cells=[]
        try:
            op,info=vd_select_lt(w,m,x,fp32,work,flush,deadline,lines);cells.append(('lt_search',op,info))
        except Exception as exc:lines.append(f'CELL_BUILD_REJECT M={m} {role} lt {type(exc).__name__}:{str(exc)[:100]}')
        for cell,fn in [('deep_gemm',lambda:VDDeepGemm(w,m,fp32))]+[(f'cutlass{i}',lambda i=i:VDCutlass(w,m,fp32,i)) for i in range(3)]:
            try:op=fn();cells.append((cell,op,op.selection()))
            except Exception as exc:lines.append(f'CELL_BUILD_REJECT M={m} {role} {cell} {type(exc).__name__}:{str(exc)[:100]}')
        # Current best7 registry is broader than19f3a13b's pinned family. Include
        # it as an additional comparison, never quietly call it the same control.
        registry={(128,'qkv_gate'):(1,23),(128,'o'):(1,0),(128,'unpooler'):(1,0),(256,'pool_router'):(0,0),(256,'unpooler'):(0,0)}
        if (m,role) in registry:
            nn,index=registry[m,role]
            try:
                op=VDLt(w,m,fp32,nn,False,workspace=registry_work)
                if index>=op.count:raise ValueError('registry ordinal unavailable')
                op.index=index;cells.append(('best7_registry',op,dict(family='registry',nn=nn,index=index)))
            except Exception as exc:lines.append(f'REGISTRY_REJECT M={m} {role} {type(exc).__name__}:{str(exc)[:100]}')
        for cell,op,info in cells:
            if time.monotonic()>deadline-3:lines.append(f'INCOMPLETE cell M={m} {role} {cell}');return lines
            try:
                graph=VDGraph(lambda op=op:op(x));rr=[]
                for seed in (172903,180822,188741):
                    gen=torch.Generator(device=device).manual_seed(seed);x.normal_(generator=gen)
                    if seed==180822:x[:,:max(1,w.shape[1]//100)]*=16
                    if seed==188741:x.zero_();x[:,17]=16
                    base.replay();graph.replay();ref=x.float()@w.float().T;router_ref=x.float()@router.T if fp32 else None
                    gate=vd_gate(graph.out,ref,role,router_ref,bias,base.out)
                    if gate['flips']:
                        disabled=dict(m=m,role=role,cell=cell,seed=seed,reason='route_flip',**gate);VD_DISABLED.append(disabled)
                        lines.append('DISABLED '+json.dumps(disabled,separators=(',',':')));break
                    pair=vd_pair(base,graph,flush);gate=vd_gate(graph.out,ref,role,router_ref,bias,base.out)
                    if gate['flips']:
                        disabled=dict(m=m,role=role,cell=cell,seed=seed,reason='replay_route_flip',**gate);VD_DISABLED.append(disabled)
                        lines.append('DISABLED '+json.dumps(disabled,separators=(',',':')));break
                    row=dict(m=m,role=role,cell=cell,seed=seed,control=label,aa_us=aa['b']-aa['a'],selection=info,**gate,**pair)
                    VD_ROWS.append(row);rr.append(row)
                if rr:
                    a=statistics.mean(r['a'] for r in rr);b=statistics.mean(r['b'] for r in rr)
                    rolecode={'qkv_gate':'q','o':'o','pool_router':'pr','unpooler':'u'}[role]
                    cellcode={'lt_search':'lt','deep_gemm':'dg','cutlass0':'c0','cutlass1':'c1','cutlass2':'c2','best7_registry':'reg'}[cell]
                    lines.append(f'D{m} {rolecode} {cellcode} {len(rr)} {a:.3f}/{b:.3f} {statistics.mean(r["ratio"] for r in rr):.5f} {aa["b"]-aa["a"]:.3f} {max(r["l2"] for r in rr):.3g}')
                    if cell=='lt_search':lines.append(f'S{m} {rolecode} '+json.dumps(info,separators=(',',':')))
                    if len(rr)==3:totals.setdefault((m,cell),[0.,0.,0]);totals[m,cell][0]+=a;totals[m,cell][1]+=b;totals[m,cell][2]+=1
            except Exception as exc:
                disabled=dict(m=m,role=role,cell=cell,reason=type(exc).__name__+':'+str(exc)[:180]);VD_DISABLED.append(disabled)
                lines.append('DISABLED '+json.dumps(disabled,separators=(',',':')))
        # Per-role wall time is retained in the local raw record, not receipt output.
        cells=base=control=None;gc.collect()
      for (tokens,cell),(a,b,count) in totals.items():
        if tokens==m:lines.append(f'T{m} {cell} {count}/4 {a:.3f}/{b:.3f}')
    lines.append(f'END rows={len(VD_ROWS)} disabled={len(VD_DISABLED)} seconds={time.monotonic()-started:.1f} remain={deadline-time.monotonic():.1f}')
    return lines
