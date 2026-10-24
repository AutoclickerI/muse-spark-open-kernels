def _vp_materialize(value):
    partial, splits = value
    return partial.sum(0) if splits > 1 else partial[0]


def _vp_recipe(x, w, recipe, dtype, bank):
    if recipe[0] == 'grouped':
        return vg_tma(x,w,recipe[1],dtype,False,bank['grouped'])
    if recipe[0] == 'tma':
        return vp_tma(x, w, recipe[1], dtype, False, bank['tma'])
    if recipe[0] == 'lt':
        _, nn, ordinal = recipe
        out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=dtype)
        weight, codes = bank['layouts'][(x.shape[0], bank['name'], nn)]
        bank['lt'].run(x, weight, out, bank['workspace'], codes, ordinal, nn,
                       dtype == torch.float32)
        return out
    raise ValueError(f'unknown verify projection family {recipe[0]}')


def _verify_proj_prepare_trials(state, incumbent, deadline, recipes):
    """Prepare an H200-derived shortlist inside the caller's prepare budget.

    `recipes[(M,name)]` is a list of ('lt',NN,ordinal), ('tma',choice8)
    or ('grouped',choice9).
    Use the lab receipt to supply that shortlist, rather than re-running the
    full search inside a layer prepare. Returns compact lines for printing last.
    The incumbent remains selected unless a fully checked paired trial wins.
    """
    begin = time.monotonic()
    deadline = min(deadline, begin+240.)
    bank = {'incumbent':incumbent, 'plan':{}, 'layouts':{}}
    state['_verify_proj'] = bank
    rows = []
    native_failed = set()
    keys = [(m,name) for m in (256,128) for name in
            ('pool_router','qkv_gate','unpooler','o') if recipes.get((m,name))]
    if not keys:
        return ['VP_INSTALL no measured shortlist; incumbent retained']
    wanted={r[0] for key in keys for r in recipes[key]}
    # Pay cold module construction before dividing the remaining trial budget
    # across shapes; otherwise compilation consumes the first shape's slot.
    for family,factory in (('lt',_pg_native),('grouped',vg_tma_module),('tma',vp_tma_module)):
        if family not in wanted:continue
        if deadline-time.monotonic()<(10 if family=='lt' else 60):
            rows.append(f'VP_INSTALL_BUDGET compile {family}')
            native_failed.add(family)
            continue
        try:bank[family]=factory()
        except Exception as exc:
            native_failed.add(family)
            rows.append(f'VP_INSTALL_REJECT module {family} {type(exc).__name__}:{str(exc)[:100]}')
    dev = state['qkv_gate'].device
    flush = torch.zeros((128<<20)//4,device=dev,dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(772413)
    for pos,(m,name) in enumerate(keys):
        now = time.monotonic()
        if now >= deadline-2:
            rows.append(f'VP_INSTALL_BUDGET {m} {name}')
            continue
        stop = now+max(0,deadline-now-2)/(len(keys)-pos)
        dtype = torch.float32 if name == 'pool_router' else torch.bfloat16
        w = state[name]; n,k = w.shape
        banks = [torch.randn(m,k,device=dev,dtype=torch.bfloat16,generator=gen) for _ in range(3)]
        x = banks[0].clone()
        refs = [torch.mm(b.float(),w.float().T) for b in banks]
        old = _PgGraph(lambda:_vp_materialize(incumbent(x,w,name,state,out_dtype=dtype)))
        best = 1.; winner = None
        for recipe in recipes[(m,name)]:
            if time.monotonic() > min(stop,deadline-2):break
            graph = None
            try:
                family = recipe[0]
                if family in native_failed:continue
                if family not in ('lt','tma','grouped'):raise ValueError('unknown projection family')
                if family == 'lt':
                    _,nn,ordinal = recipe; key = (m,name,nn)
                    if 'workspace' not in bank:
                        bank['workspace'] = torch.empty(4<<30,device=dev,dtype=torch.uint8)
                    if key not in bank['layouts']:
                        codes,_ = bank['lt'].options(m,n,k,nn,dtype==torch.float32,bank['workspace'].numel())
                        bank['layouts'][key] = (w.T.contiguous() if nn else w,codes)
                    if not 0<=ordinal<len(bank['layouts'][key][1]):raise ValueError('Lt ordinal unavailable')
                bank['name'] = name
                graph = _PgGraph(lambda:_vp_recipe(x,w,recipe,dtype,bank))
                for b in range(3):
                    x.copy_(banks[b]);old.graph.replay();graph.graph.replay()
                    vp_check(graph.out,refs[b]);vp_check(graph.out,old.out)
                before,after,ratio = vp_pair(old,graph,x,banks,flush)
                if ratio<best:best=ratio;winner=recipe
                rows.append(f'VP_INSTALL_TRIAL {m} {name} {recipe} us={before:.2f}/{after:.2f} r={ratio:.4f}')
            except Exception as exc:
                rows.append(f'VP_INSTALL_REJECT {m} {name} {recipe} {type(exc).__name__}:{str(exc)[:100]}')
            finally:
                if graph is not None:graph.close()
        old.close()
        if winner is not None:
            bank['plan'][(m,name)] = (winner,w.data_ptr(),tuple(w.shape),tuple(w.stride()),dtype)
        rows.append(f'VP_INSTALL_PICK {m} {name} {winner or "incumbent"} r={best:.4f}')
    # Retain only winning layouts and their algorithm banks.
    keep = {(m,name,r[0][1]) for (m,name),r in bank['plan'].items() if r[0][0]=='lt'}
    bank['layouts'] = {key:value for key,value in bank['layouts'].items() if key in keep}
    if not keep:bank.pop('workspace',None)
    rows.append(f'VP_INSTALL_END elapsed_s={time.monotonic()-begin:.2f} left_s={deadline-time.monotonic():.2f}')
    return rows


def verify_proj_prepare(state, incumbent, deadline, recipes=None):
    """Isolate optional setup failures; the proven caller remains executable."""
    state['_verify_proj'] = {'incumbent':incumbent,'plan':{},'layouts':{}}
    if recipes is None:recipes=VERIFY_PROJ_RECIPES
    try:
        return _verify_proj_prepare_trials(state,incumbent,deadline,recipes)
    except Exception as exc:
        # A setup/control/budget error cannot invalidate the layer's incumbent.
        state['_verify_proj']['plan'].clear()
        return [f'VP_INSTALL_REJECT setup {type(exc).__name__}:{str(exc)[:160]}']


def verify_project(x, w, name, state, out_dtype=torch.bfloat16):
    """Drop-in for decode's project(); preserves its (partials,splits) tuple."""
    bank = state['_verify_proj']
    picked = bank['plan'].get((x.shape[0],name))
    if picked is None or x.dtype!=torch.bfloat16 or not x.is_contiguous() or (
            picked[1:] != (w.data_ptr(),tuple(w.shape),tuple(w.stride()),out_dtype)):
        return bank['incumbent'](x,w,name,state,out_dtype=out_dtype)
    recipe = picked[0]
    if recipe[0]=='lt':
        _,nn,ordinal = recipe
        weight,codes = bank['layouts'][(x.shape[0],name,nn)]
        out = torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=out_dtype)
        bank['lt'].run(x,weight,out,bank['workspace'],codes,ordinal,nn,out_dtype==torch.float32)
    elif recipe[0]=='grouped':
        out = vg_tma(x,w,recipe[1],out_dtype,False,bank['grouped'])
    else:
        out = vp_tma(x,w,recipe[1],out_dtype,False,bank['tma'])
    return out.unsqueeze(0),1


def verify_proj_bind(incumbent):
    """Install before prepare; its own existing trials still see the incumbent."""
    def project(x,w,name,state,out_dtype=torch.bfloat16):
        if '_verify_proj' not in state:
            return incumbent(x,w,name,state,out_dtype=out_dtype)
        return verify_project(x,w,name,state,out_dtype=out_dtype)
    return project
