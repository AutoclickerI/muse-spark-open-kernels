# This file is embedded with frozen S11 and activity.cpp by build.py.
TL_CASES = tuple((b, q, w) for q in (1, 8) for b in (1, 16, 32) for w in (2048, 0))
TL_REPS = 5
TL_RESULTS = []


def tl_activity():
    tag = hashlib.sha256(TL_ACTIVITY.encode()).hexdigest()[:12]
    return load_inline(name='muse_layer_timeline_' + tag, cpp_sources=TL_ACTIVITY,
        functions=['profile_begin', 'profile_end', 'capture_nodes', 'exec_id'],
        no_implicit_headers=True, with_pytorch_error_handling=False,
        extra_include_paths=['/usr/local/cuda/include'],
        extra_ldflags=['-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64', '-lcupti', '-lcudart'],
        verbose=False)


def tl_host(weights):
    if 'TL_EXISTING_HOST' in globals():
        return TL_EXISTING_HOST
    name = '<timeline_frozen_s11_acda6c62>'
    linecache.cache[name] = (len(TL_S11), None, TL_S11.splitlines(True), name)
    module = types.ModuleType('timeline_s11')
    exec(compile(TL_S11, name, 'exec'), module.__dict__)
    if torch.cuda.get_device_capability() != (9, 0):
        module._ap_fa3_scheduler_metadata = lambda *a, **kw: None
    if 'TL_LOCAL_INSTALL' in globals():
        TL_LOCAL_INSTALL(module)
    state = tl_prepare_selected(module, weights)
    return module, state


def tl_nbytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(tl_nbytes(x) for x in value)
    return 0


def tl_capture(host, state, data, act, mapped):
    ns = host.__dict__
    known, stages, outputs = set(), {}, {}
    replacements = {}
    def snapshot():
        return set(act.capture_nodes(torch.cuda.current_stream().cuda_stream))
    def wrap(name, label):
        original = ns[name]
        replacements[name] = original
        def call(*args, **kw):
            if not torch.cuda.is_current_stream_capturing():
                return original(*args, **kw)
            before = snapshot()
            value = original(*args, **kw)
            after = snapshot()
            stage = label(args, kw)
            ids = after - before
            assert not (known & ids), ('nested stage capture', stage)
            known.update(ids)
            for node in ids:
                stages[node] = stage
            outputs[stage] = (args, value)
            return value
        ns[name] = call
    if mapped:
        wrap('_best7_project', lambda a, k: a[1])
        wrap('_best7_attention', lambda a, k: 'attn')
        wrap('_gate_split_apply', lambda a, k: 'gate')
        wrap('_best7_boundary', lambda a, k: a[4])
        wrap('_best7_route', lambda a, k: 'route')
        wrap('_best7_expert', lambda a, k: 'moe')
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = host.kernel_fn(data, state)
    finally:
        ns.update(replacements)
    # CUDA 13 CUPTI encodes graph identity in the upper 32 bits and node
    # identity in the lower 32. Instantiation changes the graph id. This is
    # the same rekeying torch.cuda._graph_annotations._rekey_annotations uses.
    if stages:
        eid=act.exec_id(graph.raw_cuda_graph_exec())
        assert len({node >> 32 for node in stages}) == 1
        stages={(eid << 32) | (node & 0xffffffff):stage for node,stage in stages.items()}
    return graph, out, stages, outputs


def tl_times(graph, reps):
    samples = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        lab_cold_l2()
        a.record(); graph.replay(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b) * 1000)
    return samples


def tl_union(intervals):
    intervals = sorted(intervals)
    merged = []
    for a, b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def tl_footprints(data, state, outputs, host):
    """Compulsory unique logical input reads, not a hardware HBM traffic counter.

    Weights/activations counted once per consuming stage; reuse within the stage
    may add L2 reads. Intermediate reduction traffic is separately attributed.
    """
    cfg = data['config']; m = cfg['tokens']; b = cfg['batch']; q = cfg['q_len']
    result = {}
    for stage, (args, value) in outputs.items():
        if stage in ('qkv_gate', 'qkv3', 'o', 'pool_router', 'unpooler', 'unpooler_slice'):
            result[stage] = dict(read=tl_nbytes(state[stage]) + tl_nbytes(args[0]),
                                 weights=tl_nbytes(state[stage]), input=tl_nbytes(args[0]),
                                 output=tl_nbytes(value[0]), splits=value[1])
        elif stage == 'gate':
            result[stage] = dict(read=tl_nbytes(state['gate_rows']) + tl_nbytes(args[0]) + tl_nbytes(args[1]),
                                 weights=tl_nbytes(state['gate_rows']), output=tl_nbytes(value))
        elif stage == 'attn':
            seq = data['seq_lens'].cpu().tolist()
            # Union of the causal keys read by the q queries in each request.
            keys = sum(min(s + q, cfg['window'] + q - 1) if cfg['window'] else s + q for s in seq)
            kv = keys * cfg['num_kv_heads'] * cfg['head_dim'] * 2
            qkv = tl_nbytes(args[0][0])
            result[stage] = dict(read=kv + qkv + keys * 4 + tl_nbytes(data['seq_lens']) + 8,
                                 kv=kv, keys=keys, qkv=qkv, output=tl_nbytes(value))
        elif stage in ('b1', 'b2'):
            post = stage == 'b2'
            # branch, stream, pre norm weights, alpha, beta, optional post norm
            result[stage] = dict(read=tl_nbytes(args[0][0]) + tl_nbytes(args[1]) + 8192 * 4 * (4 if post else 3),
                                 output=tl_nbytes(value))
        elif stage == 'route':
            result[stage] = dict(read=tl_nbytes(args[0][0]) + tl_nbytes(state['router_bias']) +
                                 tl_nbytes(state['pre_expert_w']) + 4, output=tl_nbytes(value))
        elif stage == 'moe':
            lids = args[3].cpu().flatten().tolist()
            counts = [lids.count(e) for e in range(32)]
            bank = state['moe_bank']
            w13 = tl_nbytes(bank['w13p']) // 32
            w2 = tl_nbytes(bank['w2p']) // 32
            count = sum(c > 0 for c in counts)
            heavy = sum(c > 16 for c in counts)
            light = count - heavy
            tokens = sum(counts)
            result[stage] = dict(read=count * (w13 + w2) + tl_nbytes(args[1]) +
                                 tl_nbytes(args[2]) + tl_nbytes(args[3]) + 8192,
                                 w13=w13, w2=w2, experts=count, heavy=heavy, light=light,
                                 routed=tokens, tokens=m, counts=counts, output=tl_nbytes(value),
                                 repeated_weight_upper=sum(math.ceil(c / 8) for c in counts) * (w13+w2))
    return result


def tl_node_bytes(name, stage, info, ordinal, count):
    """Return logical read lower bound and attribution class per original node.

    Unknown multi-node paths stay explicit (class 'unresolved'), rather than
    distributing bytes in proportion to measured time (circular bandwidth).
    """
    if not info:
        return 0, 'init'
    if count == 1:
        return info['read'], 'stage_unique'
    if stage == 'attn':
        if any(s in name for s in ('attn_main', 'vq8_main', 'flash_fwd', 'Flash', 'flash::', '_attn_split_kernel')):
            return info['kv'] + info['keys'] * 4 + info['qkv'], 'kv_unique'
        if any(s in name for s in ('sumcast', 'qk_prep', 'qk_norm', 'prepare', 'qkv')):
            return info['qkv'], 'qkv_input'
        if 'combine' in name or 'split' in name:
            # Distinct split partials require exact runtime layout, unavailable
            # for FA3; expose unknown rather than invent HBM traffic.
            return info['output'] * 2, 'partial_read_lower_bound'
        if 'gate' in name or 'norm' in name:
            return info['output'], 'attn_output'
        return 0, 'unresolved'
    if stage == 'moe':
        # CUPTI names are demangled in activity.cpp.
        if 'moe_var' in name or 'moe_fused' in name:
            return info['read'], 'expert_unique'
        if 'phase<' in name and name.split('phase<')[1].split(',')[1].strip() == 'true':
            return info['light'] * info['w13'], 'light_w13_unique'
        if 'phase<' in name and name.split('phase<')[1].split(',')[1].strip() == 'false':
            return info['light'] * info['w2'], 'light_w2_unique'
        if 'mwbr::gemm<' in name:
            if name.split('mwbr::gemm<')[1].split(',')[1].strip() == 'true':
                return info['heavy'] * info['w13'], 'heavy_w13_unique'
            if name.split('mwbr::gemm<')[1].split(',')[1].strip() == 'false':
                return info['heavy'] * info['w2'], 'heavy_w2_unique'
        if 'route' in name:
            return info['tokens'] * 8 * 4, 'route_ids'
        if 'finish' in name:
            return info['routed'] * 4096 * 4, 'expert_output'
        return 0, 'unresolved'
    if 'reduce' in name.lower() or 'sum' in name.lower() or 'cast' in name.lower():
        return info.get('output', 0), 'partial_input_lower_bound'
    if 'gemv' in name or 'gemm' in name.lower() or 'cublas' in name:
        return info['read'], 'dense_unique'
    return 0, 'unresolved'


def tl_summarize(records, mapping, footprints):
    rows = [(name, r) for name, r in records if r[4] in mapping]
    if not rows:
        raise RuntimeError('CUPTI returned no mapped graph nodes')
    by_node = {}
    for name, r in rows:
        by_node.setdefault(r[4], []).append((name, r))
    assert all(len(v) == TL_REPS for v in by_node.values()), {k: len(v) for k, v in by_node.items()}
    assert set(by_node) == set(mapping), ('missing graph activity', set(mapping)-set(by_node))
    for v in by_node.values():
        v.sort(key=lambda nr: nr[1][0])
    traces = [[by_node[node][i] for node in by_node] for i in range(TL_REPS)]
    spans = [max(r[1] for _, r in trace) - min(r[0] for _, r in trace) for trace in traces]
    chosen = sorted(range(TL_REPS), key=lambda i: spans[i])[TL_REPS // 2]
    trace = sorted(traces[chosen], key=lambda nr: (nr[1][0], nr[1][1]))
    start = min(r[0] for _, r in trace)
    names = []
    stages = []
    nodes = []
    counters = {}
    count_stage = {s: sum(mapping[r[4]] == s for _, r in trace) for s in set(mapping.values())}
    for i, (name, r) in enumerate(trace):
        stage = mapping[r[4]]
        ordinal = counters.get(stage, 0); counters[stage] = ordinal + 1
        rd, kind = tl_node_bytes(name, stage, footprints.get(stage), ordinal, count_stage[stage])
        a, b = (r[0]-start)/1000, (r[1]-start)/1000
        gap = (trace[i+1][1][0]-r[1])/1000 if i+1 < len(trace) else None
        # Original graph id is preserved, not an eager profiler approximation.
        nodes.append(dict(name=name, stage=stage, node=r[4], stream=r[13], start=a, end=b,
                          next_gap=gap, read=rd, byte_kind=kind,
                          bw=rd/(b-a)/1e6 if b>a else 0,
                          duration_mad=statistics.median(abs((v[1][1]-v[1][0])/1000-(b-a)) for v in by_node[r[4]])))
    wall = spans[chosen]/1000
    union = tl_union((n['start'], n['end']) for n in nodes)
    # Average-rate proxy: piecewise sum of each active node's logical B/duration.
    cuts = sorted({t for n in nodes for t in (n['start'], n['end'])})
    low = sum(b-a for a,b in zip(cuts,cuts[1:]) if sum(n['bw'] for n in nodes
              if n['start'] <= a and n['end'] >= b) < 2.29)
    stage_rows = {}
    for s in count_stage:
        ns = [n for n in nodes if n['stage'] == s]
        lo = min(n['start'] for n in ns); hi = max(n['end'] for n in ns)
        rd = footprints.get(s, {}).get('read', 0)
        stage_rows[s] = dict(start=lo, end=hi, span=hi-lo, read=rd,
                            bw=rd/(hi-lo)/1e6 if hi>lo else 0,
                            residual=max(0, hi-lo-rd/4.58e6))
    deps = {}
    for before, after, label in [('qkv3' if 'qkv3' in stage_rows else 'qkv_gate', 'attn', 'qkv_attn'),
                                  ('pool_router', 'route', 'router_route')]:
        if before in stage_rows and after in stage_rows:
            deps[label] = stage_rows[after]['start']-stage_rows[before]['end']
    rd = sum(f['read'] for f in footprints.values())
    return dict(wall=wall, busy_union=union, kernel_sum=sum(n['end']-n['start'] for n in nodes),
                bubble=wall-union, overlap=sum(n['end']-n['start'] for n in nodes)-union,
                low50_proxy=low, bytes=rd, stream_floor=rd/4.58e6,
                dependencies=deps, stages=stage_rows, nodes=nodes,
                span_samples=[round(x/1000,3) for x in spans], footprints=footprints)


def tl_pack(results):
    # Compact columnar archive keeps every node in the judge's 9KB lab log.
    names, kinds, stages = [], [], []
    def intern(bank, value):
        if value not in bank: bank.append(value)
        return bank.index(value)
    packed = []
    for result in results:
        row = {k:v for k,v in result.items() if k not in ('nodes', 'footprints', 'stages', 'paired_samples')}
        row['nodes'] = [[intern(names,n['name'].replace('(anonymous namespace)','anon').split('(')[0]), intern(stages,n['stage']), n['node'], n['stream'],
                         round(n['start'],3),round(n['end'],3), n['read'],intern(kinds,n['byte_kind']),
                         round(n['duration_mad'],3)] for n in result['nodes']]
        row['paired_ratios']=[round(a/c,5) for a,c in result['paired_samples']]
        row['footprints'] = {s:{k:v for k,v in f.items() if k in
            ('read','experts','heavy','light','counts','kv','keys')} for s,f in result['footprints'].items()}
        packed.append(row)
    archive = dict(version=1, names=names, stages=stages, byte_kinds=kinds,
                   columns=['name','stage','node','stream','start_us','end_us','logical_read_B','byte_kind','duration_mad_us'],
                   results=packed)
    def rounded(x):
        if isinstance(x,float):return round(x,4)
        if isinstance(x,dict):return {k:rounded(v) for k,v in x.items()}
        if isinstance(x,list):return [rounded(v) for v in x]
        return x
    body = json.dumps(rounded(archive),separators=(',',':'))
    blob = base64.b85encode(lzma.compress(body.encode(),preset=6)).decode()
    return blob


def lab(deadline, weights, state=None):
    global TL_RESULTS
    started=time.monotonic()
    lines=['META S11 acda6c62 original graph nodes; CUPTI concurrent activity; cold 1GiB READ before EVERY replay; 5 repetitions.',
           'META B=compulsory logical reads, not measured DRAM traffic. low50=stage-average proxy; unavoidable latency and actual HBM duty are not identified by activity timestamps.']
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        activity_future=pool.submit(tl_activity)
        host, st=tl_host(weights)
        act=activity_future.result()
    device=weights['o'].device
    TL_RESULTS=[]
    empty=torch.cuda.CUDAGraph()
    with torch.cuda.graph(empty):pass
    empty_us=statistics.median(tl_times(empty,TL_REPS))
    for b,q,w in TL_CASES:
        label=('d' if q==1 else 'v')+str(b)+('S' if w else 'F')
        if time.monotonic()>deadline-8:
            lines.append('INCOMPLETE deadline before '+label)
            break
        try:
            # Match public scored distributions: B1 fixed, larger batches agentic.
            data=host._ap_generate_input(batch=b,ctx=80000,layer='sliding' if w else 'full',q_len=q,
                    spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=172903,device=device)
            for _ in range(2):host.kernel_fn(data,st)
            torch.cuda.synchronize()
            # CUPTI is enabled during capture so every original graph node has an id.
            act.profile_begin()
            try:
                graph,out,mapping,outputs=tl_capture(host,st,data,act,True)
                lab_cold_l2();graph.replay();torch.cuda.synchronize()
            finally:
                act.profile_end()
            # Establish S11's own replay variation before comparing captures.
            lab_cold_l2();graph.replay();torch.cuda.synchronize()
            self_out=[x.clone() for x in out]
            lab_cold_l2();graph.replay();torch.cuda.synchronize()
            def delta(a,c):
                if a.dtype in (torch.int32,torch.int64):return float((a!=c).any())
                return float((a.float()-c.float()).norm()/a.float().norm().clamp_min(1e-12))
            self_drift=[delta(a,c) for a,c in zip(out,self_out)]
            plain=tl_times(graph,TL_REPS)
            act.profile_begin()
            try:
                profiled=tl_times(graph,TL_REPS)
            finally:
                records=act.profile_end()
            # Re-capture without host graph queries. Same GPU DAG and same data;
            # pair controls to detect observer effects from graph construction.
            control,ctrl_out,_,_=tl_capture(host,st,data,act,False)
            control_times=[];mapped_times=[]
            for i in range(TL_REPS):
                order=[('A',graph),('B',control)]
                if i%2:order.reverse()
                for arm,g in order:
                    (mapped_times if arm=='A' else control_times).extend(tl_times(g,1))
            # Original outputs remain valid after all replays on identical inputs.
            cross_drift=[delta(a,c) for a,c in zip(out,ctrl_out)]
            # Atomic accumulation permits different reduction orders. Quantify the
            # difference; the official reference gate runs in screen.py/local_eval.
            assert torch.equal(out[2],ctrl_out[2]), ('capture changed route membership',label)
            assert all(bool(torch.isfinite(x).all()) for x in (*out[:2],*ctrl_out[:2]))
            print('TL_OUTPUT '+label+' self='+repr(self_drift)+' cross='+repr(cross_drift),flush=True)
            footprints=tl_footprints(data,st,outputs,host)
            # Any allocation initialization kernel outside a wrapped stage is explicit.
            graph_ids={r[4] for _,r in records if r[4]}
            for node in graph_ids-set(mapping):mapping[node]='init'
            result=tl_summarize(records,mapping,footprints)
            result.update(case=label,event_plain=statistics.median(plain),event_profiled=statistics.median(profiled),
                          empty_event=empty_us,event_corrected=statistics.median(plain)-empty_us,
                          observer_ratio=statistics.median(a/c for a,c in zip(mapped_times,control_times)),
                          profile_ratio=statistics.median(profiled)/statistics.median(plain),
                          self_drift=self_drift,cross_drift=cross_drift,
                          paired_samples=[[round(a,3),round(c,3)] for a,c in zip(mapped_times,control_times)])
            TL_RESULTS.append(result)
            # Preserve each completed result if a later shape fails.
            print('TL_PROGRESS '+label+' wall=%.3f node_sum=%.3f bubble=%.3f capture_A/B=%.4f profile/off=%.4f'%
                  (result['wall'],result['kernel_sum'],result['bubble'],result['observer_ratio'],result['profile_ratio']),flush=True)
            del control,graph,out,ctrl_out,outputs,data
            gc.collect()
        except Exception as exc:
            lines.append('FAIL '+label+' '+type(exc).__name__+': '+str(exc)[:180])
            print(lines[-1],flush=True)
    blob=tl_pack(TL_RESULTS)
    lines.append('TL_XZ85 '+blob)
    lines.append('END shapes=%d/12 elapsed=%.1fs archive_chars=%d'%(len(TL_RESULTS),time.monotonic()-started,len(blob)))
    assert sum(len(line)+1 for line in lines)<9000, ('archive exceeds receipt budget',len(blob))
    return lines
