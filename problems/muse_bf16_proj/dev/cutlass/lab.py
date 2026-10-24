# Every optional failure is diagnostic; the scored host is unchanged.
VGC_LT_RECIPES = {(128,'qkv_gate'):(1,23),(128,'pool_router'):(0,0),
                  (128,'unpooler'):(1,0),(128,'o'):(1,0),
                  (256,'qkv_gate'):(0,0),(256,'pool_router'):(0,0),(256,'unpooler'):(0,0)}

def lab(deadline,weights,state=None):
    begin=time.monotonic();deadline=min(deadline,begin+225)
    rows=[]
    try:module=vgc_module()
    except Exception as exc:return [f'VGC_REJECT module {type(exc).__name__}:{str(exc)[-300:]}']
    compile_s=time.monotonic()-begin
    try:lt=_pg_native()
    except Exception as exc:
        lt=None;rows.append(f'VGC_REJECT lt {type(exc).__name__}:{str(exc)[:120]}')
    device=weights['qkv_gate'].device
    sms=torch.cuda.get_device_properties(device).multi_processor_count
    router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
    ws={name:weights[name] for name in ('qkv_gate','o','unpooler')}
    ws['pool_router']=torch.cat((weights['pooler'],hi,lo),0).contiguous()
    workspace=torch.empty(4<<30,device=device,dtype=torch.uint8) if lt is not None else None
    flush=torch.zeros((1<<30)//4,device=device,dtype=torch.float32)
    generator=torch.Generator(device=device).manual_seed(8374021)
    rows.append(f'VGC_META sm={sms} compile_s={compile_s:.2f} cold1GiB paired20 fresh3; FP32 pool+router singleGEMM; counter reset timed')
    keys=[(m,name) for m in (256,128) for name in ('qkv_gate','pool_router','unpooler','o')]
    for pos,(m,name) in enumerate(keys):
        now=time.monotonic();stop=now+max(0,deadline-now-3)/(len(keys)-pos)
        if now>=deadline-4:rows.append(f'VGC_BUDGET {m} {name}');continue
        old=None;torch_graph=None;lt_graph=None
        try:
            w=ws[name];n,k=w.shape;dtype=torch.float32 if name=='pool_router' else torch.bfloat16
            banks=[torch.randn(m,k,device=device,dtype=torch.bfloat16,generator=generator) for _ in range(2)]
            probe=torch.zeros_like(banks[0]);probe[:,127]=.5;banks.append(probe)
            x=banks[0].clone();refs=[b.float()@w.float().T for b in banks]
            torch_graph=_PgGraph(lambda:torch.mm(x,w.T,out_dtype=torch.float32) if dtype==torch.float32 else torch.mm(x,w.T))
            for b,ref in zip(banks,refs):x.copy_(b);torch_graph.graph.replay();vp_check(torch_graph.out,ref)
            old=torch_graph;control='cublas';lib_row='none'
            if lt is not None and (m,name) in VGC_LT_RECIPES:
                try:
                    nn,ordinal=VGC_LT_RECIPES[m,name]
                    weight=w.T.contiguous() if nn else w
                    codes,_=lt.options(m,n,k,nn,dtype==torch.float32,workspace.numel())
                    if ordinal>=len(codes):raise ValueError('H200 Lt ordinal unavailable')
                    out=torch.empty((m,n),device=device,dtype=dtype)
                    def run_lt():
                        lt.run(x,weight,out,workspace,codes,ordinal,nn,dtype==torch.float32)
                        return out
                    lt_graph=_PgGraph(run_lt)
                    for b,ref in zip(banks,refs):
                        x.copy_(b);lt_graph.graph.replay();vp_check(lt_graph.out,ref)
                    t0,t1,r=vp_pair(torch_graph,lt_graph,x,banks,flush,20)
                    lib_row=f'{t0:.2f}/{t1:.2f}/{r:.4f}'
                    if r<1:old=lt_graph;control=f'lt{nn}.{ordinal}'
                except Exception as exc:
                    rows.append(f'VGC_REJECT lt {m} {name} {type(exc).__name__}:{str(exc)[:90]}')
            gemv='rejected'
            try:
                with _PgGraph(lambda:vgc_gemv(x,w,dtype)) as graph:
                    for b,ref in zip(banks,refs):x.copy_(b);graph.graph.replay();vp_check(graph.out,ref)
                    t0,t1,ratio=vp_pair(old,graph,x,banks,flush,20)
                    gemv=f'{t0:.2f}/{t1:.2f}/{ratio:.4f}'
            except Exception as exc:
                rows.append(f'VGC_REJECT gemv {m} {name} {type(exc).__name__}:{str(exc)[:90]}')
            candidates=[];bad=0;reasons=set();families={};attempts=0
            coarse_stop=now+.70*max(0,stop-now)
            for recipe in vgc_choices(m,n,k,sms):
                if time.monotonic()>coarse_stop:break
                plan=None;graph=None;attempts+=1
                try:
                    plan=module.plan(m,n,k,dtype,recipe,device)
                    out=torch.empty((m,n),device=device,dtype=dtype)
                    graph=_PgGraph(lambda:plan.run(x,w,out))
                    error=0.
                    for b,ref in zip(banks,refs):
                        x.copy_(b);old.graph.replay();graph.graph.replay()
                        error=max(error,vp_check(graph.out,ref),vp_check(graph.out,old.out))
                    t0,t1,ratio=vp_pair(old,graph,x,banks,flush,4)
                    candidates.append((ratio,recipe,error,plan.info,plan.workspace.numel()))
                    # Record best tile/layout and each scheduler family, including losses.
                    family=(recipe[0],recipe[1])
                    if family not in families or ratio<families[family][0]:families[family]=(ratio,t1)
                except Exception as exc:
                    bad+=1;reason=f'{type(exc).__name__}:{str(exc).splitlines()[0][:100]}'
                    if reason not in reasons and len(reasons)<2:rows.append(f'VGC_REJECT {m} {name} c={recipe} {reason}')
                    reasons.add(reason)
                finally:
                    if graph is not None:graph.close()
                    if plan is not None:plan.close()
            finals=sorted(candidates)[:3]
            for mode in (1,2,3):
                family=[r for r in candidates if r[1][1]==mode]
                if family and min(family) not in finals:finals.append(min(family))
            best=(1.,None,None);final_rows=[]
            for _,recipe,error,info,size in finals:
                if time.monotonic()>min(stop,deadline-2):break
                plan=None
                try:
                    plan=module.plan(m,n,k,dtype,recipe,device)
                    out=torch.empty((m,n),device=device,dtype=dtype)
                    with _PgGraph(lambda:plan.run(x,w,out)) as graph:
                        t0,t1,ratio=vp_pair(old,graph,x,banks,flush,20)
                        for b,ref in zip(banks,refs):
                            x.copy_(b);old.graph.replay();graph.graph.replay()
                            vp_check(graph.out,ref);vp_check(graph.out,old.out)
                        final_rows.append((recipe,round(t0,3),round(t1,3),round(ratio,5)))
                        if ratio<best[0]:best=(ratio,recipe,(t0,t1))
                except Exception as exc:
                    bad+=1;rows.append(f'VGC_REJECT final {m} {name} {type(exc).__name__}:{str(exc)[:90]}')
                finally:
                    if plan is not None:plan.close()
            rows.append(f'VGC_CONTROLS {m} {name} selected={control} torch/Lt/r={lib_row} control/GEMV/r={gemv}')
            # Compact losses too: one best result per tile and per decomposition.
            tile_best={a:min(v for (t,_),v in families.items() if t==a) for a in range(8)
                       if any(t==a for t,_ in families)}
            mode_best={b:min(v for (_,d),v in families.items() if d==b) for b in range(4)
                       if any(d==b for _,d in families)}
            fam=' '.join(f't{a}:{v[1]:.1f}/{v[0]:.3f}' for a,v in tile_best.items())
            fam+=' '+' '.join(f'd{b}:{v[1]:.1f}/{v[0]:.3f}' for b,v in mode_best.items())
            rows.append(f'VGC_FAMILIES {m} {name} {fam}')
            rows.append(f'VGC_BEST {m} {name} pick={best[1]} us={best[2]} ratio={best[0]:.5f} tried={attempts} rejected={bad} finals={final_rows}')
        except Exception as exc:
            rows.append(f'VGC_REJECT shape {m} {name} {type(exc).__name__}:{str(exc)[:140]}')
        finally:
            if lt_graph is not None:lt_graph.close()
            if torch_graph is not None:torch_graph.close()
    rows.append(f'VGC_END elapsed={time.monotonic()-begin:.2f} left={deadline-time.monotonic():.2f}')
    return rows
