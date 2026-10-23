# Four complete attention paths, selected only using prepare-time synthetic inputs.
# No graph, synthetic activation, cache value or output is retained in state.
_ATTENTION_TUNE_CHOICES = (
    ('fa3', {}),
    ('fp8_direct128', dict(bq=16, bn=128, group=8, warps=8, stages=2, contig=False)),
    ('fp8_gather128', dict(bq=16, bn=128, group=8, warps=8, stages=2, contig=True)),
    ('fp8_gather256', dict(bq=16, bn=256, group=8, warps=8, stages=2, contig=True)),
)


def _attention_bucket(data):
    cfg = data['config']
    prefix = max(0, data['page_table'].shape[1] * cfg['page_size'] - cfg['q_len'])
    return 0 if prefix < 16384 else 32768 if prefix < 49152 else 65536


def _attention_path(data, q8, st, name):
    if name == 'fa3':
        return _attention_fa3(data, q8, st)
    config = dict(_ATTENTION_TUNE_CHOICES)[name]
    return _attention_fp8(data, q8, st, **config)


def _tune_attention(st):
    import json
    dev = st['qkv'].device
    selected = {0: 'fa3'}
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    for prefix in (32768, 65536):
        L = 16384
        S = prefix + L
        gen = torch.Generator(device=dev).manual_seed(119431 + prefix)
        q = torch.randn((L, HQ, HEAD_DIM), generator=gen, device=dev, dtype=torch.float32)
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True))
        q8 = q.to(torch.float8_e4m3fn).reshape(L, Q_SIZE)
        k = torch.randn((S, 1, HKV, HEAD_DIM), generator=gen, device=dev, dtype=torch.float32)
        u = torch.randn((HKV, HEAD_DIM), generator=gen, device=dev, dtype=torch.float32)
        u *= torch.rsqrt(u.square().mean(-1, keepdim=True))
        k = (0.56 ** 0.5) * u + (0.44 ** 0.5) * k
        v = torch.randn(k.shape, generator=gen, device=dev, dtype=torch.float32)
        data = dict(
            config=dict(tokens=L, batch=1, q_len=L, window=0, page_size=1,
                        sm_scale=1.0 / 8.0, qk_scale_factor=5.4730064863875, kv_scale=1.0),
            k_cache=k.to(torch.float8_e4m3fn), v_cache=v.to(torch.float8_e4m3fn),
            page_table=torch.randperm(S, generator=gen, device=dev, dtype=torch.int64).to(torch.int32).view(1, S),
            seq_lens=torch.full((1,), prefix, device=dev, dtype=torch.int32),
            cache_seqlens=torch.full((1,), S, device=dev, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, L], device=dev, dtype=torch.int32),
            scheduler_metadata=None,
            k_scale=torch.ones(1, device=dev, dtype=torch.float32),
            v_scale=torch.ones(1, device=dev, dtype=torch.float32),
        )
        del q, k, u, v
        times = {}
        errors = {}
        reference = None
        for name, _ in _ATTENTION_TUNE_CHOICES:
            fn = lambda: _attention_path(data, q8, st, name)
            for _ in range(2):
                output = fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = fn()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(3):
                flush.fill_(1)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000.0)
            times[name] = sorted(samples)[1]
            if reference is None:
                reference = output.float().clone()
                errors[name] = 0.0
            else:
                delta = output.float() - reference
                relative = float(delta.norm() / reference.norm().clamp_min(1e-12))
                if not math.isfinite(relative) or relative > 0.04:
                    raise RuntimeError(f'Attention paths disagree at synthetic prefix={prefix}, {name}: {relative}')
                errors[name] = relative
                del delta
            del graph, output
        selected[prefix] = min(times, key=times.get)
        print('PREFILL_ATTENTION_TUNE', json.dumps(dict(prefix=prefix, query_length=L,
              us=times, relative_l2=errors, selected=selected[prefix]), sort_keys=True), flush=True)
        del data, q8, reference
    st['attention_dispatch'] = selected


_prepare_experts = prepare


def prepare(weights):
    st = _prepare_experts(weights)
    st['attention_dispatch'] = {0: 'fa3', 32768: 'fp8_gather128', 65536: 'fp8_gather128'}
    if IS_HOPPER:
        _tune_attention(st)
    return st


def _attention(data, q8, st):
    cfg = data['config']
    if cfg['window'] or cfg['q_len'] < 2048:
        return _attention_fa3(data, q8, st)
    return _attention_path(data, q8, st, st['attention_dispatch'][_attention_bucket(data)])
