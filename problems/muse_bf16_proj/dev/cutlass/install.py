# Only H200-positive recipes belong here. The lab reader supplies them after receipt.
VGC_RECIPES = {}


def _vgc_materialize(value):
    partial,splits=value
    return partial.sum(0) if splits>1 else partial[0]


def vgc_prepare(state,incumbent,deadline,recipes=None):
    """Prepare-time winner selection at decode's exact ABI and shapes.

    recipes[(128|256,role)] = [recipe7,...] from VGC_BEST in the lab receipt.
    Passing an empty mapping does no compilation and retains the incumbent.
    Log returned lines LAST in the layer's prepare. The optional tuner never raises.
    """
    bank={'incumbent':incumbent,'plans':{}}
    state['_vgc_proj']=bank
    begin=time.monotonic();deadline=min(deadline,begin+240)
    rows=[];module=None
    if recipes is None:recipes=VGC_RECIPES
    keys=[(m,name) for m in (256,128) for name in ('qkv_gate','pool_router','unpooler','o')
          if recipes.get((m,name)) and name in state]
    if not keys:return ['VGC_INSTALL no H200-positive shortlist; incumbent retained']
    try:
        if deadline-time.monotonic()<100:
            return ['VGC_INSTALL_BUDGET compile; incumbent retained']
        module=vgc_module()
        dev=state['qkv_gate'].device
        flush=torch.zeros((1<<30)//4,device=dev,dtype=torch.float32)
        generator=torch.Generator(device=dev).manual_seed(377382)
        for pos,(m,name) in enumerate(keys):
            now=time.monotonic();stop=now+max(0,deadline-now-2)/(len(keys)-pos)
            if now>=deadline-2:rows.append(f'VGC_INSTALL_BUDGET {m} {name}');continue
            old=None;winner=None;best=1.
            try:
                w=state[name];n,k=w.shape;dtype=torch.float32 if name=='pool_router' else torch.bfloat16
                if m not in (128,256) or tuple(w.shape)!=dict(qkv_gate=(2304,8192),
                        pool_router=(4608,8192),unpooler=(8192,4096),o=(8192,1024))[name]:
                    raise ValueError('unexpected decode projection shape')
                banks=[torch.randn(m,k,device=dev,dtype=torch.bfloat16,generator=generator) for _ in range(2)]
                probe=torch.zeros_like(banks[0]);probe[:,127]=.5;banks.append(probe)
                x=banks[0].clone();refs=[b.float()@w.float().T for b in banks]
                old=_PgGraph(lambda:_vgc_materialize(incumbent(x,w,name,state,out_dtype=dtype)))
                for b,ref in zip(banks,refs):x.copy_(b);old.graph.replay();vp_check(old.out,ref)
                for recipe in recipes[m,name]:
                    if time.monotonic()>min(stop,deadline-2):break
                    plan=None;accepted=False
                    try:
                        plan=module.plan(m,n,k,dtype,recipe,dev)
                        out=torch.empty((m,n),device=dev,dtype=dtype)
                        with _PgGraph(lambda:plan.run(x,w,out)) as graph:
                            for b,ref in zip(banks,refs):
                                x.copy_(b);old.graph.replay();graph.graph.replay()
                                vp_check(graph.out,ref);vp_check(graph.out,old.out)
                            t0,t1,ratio=vp_pair(old,graph,x,banks,flush,20)
                            # Validate again after timing to catch persistent reduction-state leaks.
                            for b,ref in zip(banks,refs):
                                x.copy_(b);graph.graph.replay();vp_check(graph.out,ref)
                        rows.append(f'VGC_INSTALL_TRIAL {m} {name} c={recipe} us={t0:.3f}/{t1:.3f} r={ratio:.5f}')
                        if ratio<best:
                            if winner is not None:winner.close()
                            winner=plan;best=ratio;accepted=True
                    except Exception as exc:
                        rows.append(f'VGC_INSTALL_REJECT {m} {name} {type(exc).__name__}:{str(exc)[:110]}')
                    finally:
                        if plan is not None and not accepted:plan.close()
                if winner is not None:
                    bank['plans'][m,name]=(winner,w.data_ptr(),tuple(w.shape),tuple(w.stride()),dtype)
                rows.append(f'VGC_INSTALL_PICK {m} {name} c={winner.recipe if winner else None} r={best:.5f}')
            except Exception as exc:
                if winner is not None:winner.close()
                rows.append(f'VGC_INSTALL_REJECT shape {m} {name} {type(exc).__name__}:{str(exc)[:110]}')
            finally:
                if old is not None:old.close()
    except Exception as exc:
        for plan,*_ in bank['plans'].values():plan.close()
        bank['plans'].clear()
        rows.append(f'VGC_INSTALL_REJECT setup {type(exc).__name__}:{str(exc)[:140]}')
    rows.append(f'VGC_INSTALL_END elapsed_s={time.monotonic()-begin:.3f} selected={len(bank["plans"])}')
    return rows


def vgc_project(x,w,name,state,out_dtype=torch.bfloat16):
    """Drop-in project(x,w,name,state,out_dtype)->(partials,splits)."""
    bank=state['_vgc_proj'];selected=bank['plans'].get((x.shape[0],name))
    if (selected is None or x.dtype!=torch.bfloat16 or not x.is_contiguous() or
            selected[1:]!=(w.data_ptr(),tuple(w.shape),tuple(w.stride()),out_dtype)):
        return bank['incumbent'](x,w,name,state,out_dtype=out_dtype)
    plan=selected[0]
    out=torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=out_dtype)
    plan.run(x,w,out)
    return out.unsqueeze(0),1


def vgc_bind(incumbent):
    """Bind before original prepare; its own tuner continues to see its incumbent."""
    def project(x,w,name,state,out_dtype=torch.bfloat16):
        if '_vgc_proj' not in state:
            return incumbent(x,w,name,state,out_dtype=out_dtype)
        return vgc_project(x,w,name,state,out_dtype=out_dtype)
    return project
