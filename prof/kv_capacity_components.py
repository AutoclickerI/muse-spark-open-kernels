"""GPU0 full-width allocation census, separate from the tiny serving census.

Banks and router buffers use production shapes and code. Graphs reuse one
layer's immutable parameters across 62 forwards and retain five hidden taps.
Collectives are identity; on sm120 the attention stand-in comes from the
previous overlap probe. Graph numbers are a liveness proxy, not an H200 total.
"""

import argparse
import gc
import json
from pathlib import Path

import torch

from prefill_overlap import ROOT, load, operators


def storages(value):
    result = {}
    def visit(x):
        if torch.is_tensor(x):
            if x.is_cuda:
                s = x.untyped_storage()
                result[s.data_ptr()] = dict(bytes=s.nbytes(), shape=list(x.shape),
                                            dtype=str(x.dtype))
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
    visit(value)
    return dict(bytes=sum(v['bytes'] for v in result.values()),
                tensors=list(result.values()))


def memory():
    torch.cuda.synchronize()
    pools = {}
    for segment in torch.cuda.memory_snapshot():
        key = str(segment.get('segment_pool_id', (0, 0)))
        pools[key] = pools.get(key, 0) + segment['total_size']
    return dict(allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                cuda_free=torch.cuda.mem_get_info()[0], pools=pools)


def router(old_tree):
    from sglang.kernels.ops.muse_spark_v12.projection_hopper import RouterProjection
    from sglang.srt.managers.muse_workspace import _router_workspaces
    old = load('kv_old_router', old_tree / 'python/sglang/kernels/ops/muse_spark_v12/projection_hopper.py')
    torch.manual_seed(1234)
    weight = torch.randn(256, 8192, device='cuda', dtype=torch.float32) / 8192**.5
    sizes = [b * 8 for b in list(range(1, 9)) + list(range(10, 33, 2))]
    inputs = {m: torch.randn(m, 8192, device='cuda', dtype=torch.bfloat16) for m in sizes}
    before = memory()
    ops = [old.RouterProjection(weight) for _ in range(62)]
    expected = {}
    for op in ops:
        for m, x in inputs.items():
            expected[m] = op(x)
    original = memory()
    scratch = storages([op.scratch for op in ops])
    pairs = storages([op.pair for op in ops])
    del op, ops
    gc.collect()
    torch.cuda.empty_cache()
    ops = [RouterProjection(weight) for _ in range(62)]
    for op in ops:
        for m, x in inputs.items():
            assert torch.equal(op(x), expected[m]), m
    shared = memory()
    shared_scratch = storages(_router_workspaces)
    # Different capture streams exercise independent workspaces and counters.
    graphs, outputs, streams = [], [], [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for op in ops:
                op(inputs[256])
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for op in ops:
                out = op(inputs[256])
        graphs.append(graph)
        outputs.append(out)
    inputs[256].mul_(.5)
    torch.cuda.synchronize()
    for _ in range(3):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
        torch.cuda.synchronize()
        want = ops[0](inputs[256])
        assert all(torch.equal(out, want) for out in outputs)
    assert all(torch.count_nonzero(counts).item() == 0 for _, counts in _router_workspaces.values())
    return dict(before=before, original=original, shared=shared,
                original_scratch=scratch, original_pairs=pairs,
                shared_scratch=shared_scratch, capture_streams=2,
                graph_changed_input_exact=True, zero_completion_counters=True,
                batch_sizes=sizes)


def bank_and_graphs(backend, graphs):
    ref, base = operators()
    weights = ref.generate_weights(20261024)
    state = base.prepare(weights)
    result = dict(raw=storages({k: weights[k] for k in
                              ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs')}),
                  stock=storages(state['bank']))
    if backend == 'patch':
        import sglang.srt.layers.muse_spark_moe as port
        expert = port.prepare({**weights, 'w13_gs_gate': weights['w13_gs'],
                               'w13_gs_up': weights['w13_gs'],
                               'post_norm_w': weights['post_expert_norm']})
        result['patch_bank'] = storages({k: expert[k] for k in
                                         ('w13p', 'w2p', 'gs13_gate', 'gs13_up', 'gs2')})

        def experts(x, scores, ids, st):
            data = dict(hidden_states=x, topk_ids=ids, topk_weights=scores,
                        config=dict(local_count=32, d_hidden=4096,
                                    d_expert=4096, local_start=0))
            dense, _ = (port._grouped_call(data, expert) if x.shape[0] >= 753
                        else port._v9_call(data, expert, return_dense=True))
            return base.fused_post_expert_norm_weighted_sum(
                dense, scores, norm_weight=st['post_expert_w'],
                eps=base.POST_NORM_EPS, weight_first=True,
                slot_expert_ids=st['bank']['expert_map'][ids])
        base._experts = experts
    if not graphs:
        return result
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream()
    retained = []
    phases = []
    for name, batch, q_len, ctx in [('prefill16K', 1, 16384, 16384),
                                    ('verify32', 32, 8, 80000)]:
        case = dict(batch=batch, ctx=ctx, q_len=q_len, layer='full', page=1,
                    dist='soft', spread='fixed', seed=20261025)
        case.update(ref.case_fields(case, weights))
        data = ref.generate_input(**case)

        def forward():
            current = dict(data)
            taps = []
            for i in range(62):
                # Same tensor liveness; per-layer KV and weights are omitted
                # from this dynamic-only proxy and counted by source instead.
                x, residual, ids = base.kernel_fn(current, state)
                current['attn_input'], current['stream'] = x, residual
                if i in (6, 18, 30, 42, 54):
                    taps.append(residual.to(torch.bfloat16))
            return x, residual, ids, taps

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
        stream.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        before = memory()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=pool):
            output = forward()
        graph.replay()
        torch.cuda.synchronize()
        retained.append((graph, output, data))
        after = memory()
        row = dict(phase=name, before=before, after=after,
                   output=storages(output), pool=str(pool), layers=62,
                   target_taps=5, bank_backend=backend)
        phases.append(row)
        print(json.dumps(row), flush=True)
    result['phases'] = phases
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('router', 'banks', 'graphs'), required=True)
    ap.add_argument('--backend', choices=('stock', 'patch'), default='patch')
    ap.add_argument('--old-tree', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    result = (router(args.old_tree) if args.mode == 'router'
              else bank_and_graphs(args.backend, args.mode == 'graphs'))
    result.update(mode=args.mode, backend=args.backend,
                  device=torch.cuda.get_device_name(), sm=torch.cuda.get_device_capability())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(mode=args.mode, backend=args.backend, out=str(args.out))), flush=True)


if __name__ == '__main__':
    main()
