# H200-positive recipes only. Empty until the matching lab receipt lands.
PFC_RECIPES = {}
PFC_SHAPES = {'pooler': (4096,8192), 'unpooler': (8192,4096),
              'qkv': (2304,8192), 'o': (8192,1024), 'router_cat': (512,8192)}


def pfc_choices(m,n,k,sms=132):
    # Cover every tile/multicast family before refinements. Large M favors DP;
    # narrow N also gets explicit split-K16/32, with all reductions timed.
    for mode,split in ((1,1),(3,1),(2,2),(2,4),(2,8),(0,1)):
        for tile in range(len(VGC_TILES)):
            yield (tile,mode,split,sms,0,1,0)
    for recipe in vgc_choices(m,n,k,sms):
        if recipe[4:]!=(0,1,0) or recipe[3]!=sms or recipe[2]==16:
            yield recipe
    if n<=2304:
        for tile in range(len(VGC_TILES)):
            yield (tile,2,32,sms,0,1,0)


def pfc_check(out,reference,router=False):
    a,b=out.float(),reference.float()
    diff=a-b
    rel=float(diff.norm()/b.norm().clamp_min(1e-12))
    norms=b.norm(dim=1)
    row=float((diff.norm(dim=1)/norms.clamp_min(norms.mean()*.1).clamp_min(1e-12)).max())
    if not bool(torch.isfinite(a).all()) or rel>(5e-5 if router else .008) or row>(2e-4 if router else .016):
        raise ValueError(f'FP32 gate rel={rel:.6g} worst_row={row:.6g}')
    return rel,row


def pfc_pair(old,new,x,banks,flush,repeats=20,deadline=float('inf')):
    samples=[[],[]]
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for r in range(repeats):
        if time.monotonic()>=deadline:raise TimeoutError('CUTLASS paired timing budget')
        x.copy_(banks[(r//2)%len(banks)])
        for side in ((0,1) if r%2==0 else (1,0)):
            flush.sum()
            a.record();(old if side==0 else new).graph.replay();b.record();b.synchronize()
            samples[side].append(float(a.elapsed_time(b))*1000)
    ratios=[(samples[1][j]*samples[1][j+1]/samples[0][j]/samples[0][j+1])**.5
            for j in range(0,repeats,2)]
    return statistics.median(samples[0]),statistics.median(samples[1]),statistics.median(ratios)


def pfc_banks(m,k,device):
    generator=torch.Generator(device=device).manual_seed(479183+m+k)
    banks=[torch.randn(m,k,device=device,dtype=torch.bfloat16,generator=generator) for _ in range(2)]
    probe=torch.zeros_like(banks[0]);probe[:,127%k]=.5;banks.append(probe)
    return banks


def pfc_reference(x,w):
    # Independent full-FP32 oracle, even if a caller enabled TF32 globally.
    previous=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:return x.float()@w.float().T
    finally:torch.backends.cuda.matmul.allow_tf32=previous


def pfc_prepare(st,deadline,recipes=None):
    """Re-gate and retime only receipt-approved recipes; reject, never raise.

    Runs after the inherited stage tuners and before the dense plan/group tuner.
    Thus the latter sees the selected CUTLASS product as another incumbent, and
    includes routing/hi+lo consumption when it compares mixed/separate groups.
    """
    st['_pfc_plans']={};st['_pfc_notes']=[]
    rows=st['_pfc_notes']
    recipes=PFC_RECIPES if recipes is None else recipes
    keys=[(m,key) for m in (16384,753,3012) for key in PFC_SHAPES if recipes.get((m,key))]
    if not keys:
        rows.append('PFC_INSTALL no H200-positive recipes; incumbent retained')
        return
    try:
        if deadline-time.monotonic()<100:
            rows.append('PFC_INSTALL_BUDGET compile; incumbent retained');return
        module=vgc_module()
        device=st['qkv'].device
        flush=torch.zeros((1<<30)//4,device=device,dtype=torch.float32)
        for pos,(m,key) in enumerate(keys):
            now=time.monotonic();stop=now+max(0,deadline-now-2)/(len(keys)-pos)
            if now>=deadline-2:rows.append(f'PFC_INSTALL_BUDGET {m} {key}');continue
            winner=None;old=None;best=1.;old_choice=st['dense_choices'].get((key,m),('cublas',))
            try:
                w=st[key];n,k=w.shape;dtype=torch.float32 if key=='router_cat' else torch.bfloat16
                if tuple(w.shape)!=PFC_SHAPES[key]:raise ValueError('unexpected prefill projection shape')
                banks=pfc_banks(m,k,device);x=banks[0].clone()
                refs=[pfc_reference(fresh,w) for fresh in banks]
                old=_PgGraph(lambda:_score_dense_run(x,st,key,old_choice))
                for fresh,ref in zip(banks,refs):
                    x.copy_(fresh);old.graph.replay();pfc_check(old.out,ref,key=='router_cat')
                for recipe in recipes[m,key]:
                    if time.monotonic()>=min(stop,deadline-2):break
                    plan=None;keep=False
                    try:
                        plan=module.plan(m,n,k,dtype,recipe,device)
                        def run():
                            out=torch.empty((m,n),device=device,dtype=dtype)
                            return plan.run(x,w,out)
                        with _PgGraph(run) as graph:
                            for fresh,ref in zip(banks,refs):
                                x.copy_(fresh);old.graph.replay();graph.graph.replay()
                                pfc_check(graph.out,ref,key=='router_cat')
                                pfc_check(graph.out,old.out,key=='router_cat')
                            t0,t1,ratio=pfc_pair(old,graph,x,banks,flush,20,min(stop,deadline-2))
                            for fresh,ref in zip(banks,refs):
                                x.copy_(fresh);graph.graph.replay();pfc_check(graph.out,ref,key=='router_cat')
                        rows.append(f'PFC_INSTALL_TRIAL {m} {key} c={recipe} us={t0:.3f}/{t1:.3f} r={ratio:.5f}')
                        if ratio<best:
                            if winner is not None:winner.close()
                            winner=plan;best=ratio;keep=True
                    except Exception as exc:
                        rows.append(f'PFC_INSTALL_REJECT {m} {key} {type(exc).__name__}:{str(exc)[:130]}')
                    finally:
                        if plan is not None and not keep:plan.close()
                if winner is not None:
                    recipe=tuple(winner.recipe)
                    st['_pfc_plans'][key,m,recipe]=winner
                    st['dense_choices'][key,m]=('cutlass',recipe)
                rows.append(f'PFC_INSTALL_PICK {m} {key} c={winner.recipe if winner else None} r={best:.5f}')
            except Exception as exc:
                if winner is not None:winner.close()
                rows.append(f'PFC_INSTALL_REJECT shape {m} {key} {type(exc).__name__}:{str(exc)[:130]}')
            finally:
                if old is not None:old.close()
    except Exception as exc:
        rows.append(f'PFC_INSTALL_REJECT setup {type(exc).__name__}:{str(exc)[:160]}')


def pfc_run(x,st,key,recipe):
    plan=st['_pfc_plans'][key,x.shape[0],tuple(recipe)]
    out=torch.empty((x.shape[0],st[key].shape[0]),device=x.device,dtype=plan.dtype)
    return plan.run(x,st[key],out)
