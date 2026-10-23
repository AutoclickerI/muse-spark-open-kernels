"""Bounded V9 serving dispatch vs stock Marlin; real NVFP4, eager/fresh graphs."""

import json

import torch
from sglang.srt.layers.muse_spark_serving import run_v9_dense
from sglang.srt.layers.muse_spark_moe import prepare
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from test_muse_moe_v4_layer import _weights, _inputs, _marlin_state, _stock


def main():
    device = torch.device("cuda")
    weights = _weights(device)
    state = prepare(weights, local_start=0)
    marlin, workspace = _marlin_state(weights, device)
    norm = (1 + weights["post_norm_w"].float()).bfloat16()

    def call(x, scores, ids):
        dense, mapping = run_v9_dense(state, x, scores, ids)
        return fused_post_expert_norm_weighted_sum(
            dense,
            scores,
            norm_weight=norm,
            eps=1e-8,
            weight_first=True,
            slot_expert_ids=mapping,
        ), mapping

    rows = []
    for m in (1, 8, 128, 256, 272, 753, 3012, 16384):
        x, ids, scores = _inputs(m, device, 91000 + m)
        reference = _stock(marlin, workspace, weights, x, ids, scores)
        eager, mapping = call(x, scores, ids)
        expected_map = torch.arange(m * 8, device=device, dtype=torch.int32).view(m, 8)
        expected_map = torch.where(ids < 32, expected_map, -1)
        assert torch.equal(mapping, expected_map), (m, "chunk-global or remote map")
        rel = float(
            (eager.float() - reference.float()).norm() / reference.float().norm()
        )
        maxdiff = float((eager.float() - reference.float()).abs().max())
        assert rel <= 0.015, (m, rel, maxdiff)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_map = call(x, scores, ids)
        for _ in range(2):
            graph.replay()
            assert torch.equal(captured, eager), (m, "graph mismatch")
            assert torch.equal(captured_map, expected_map)
            x.mul_(0.5)
            eager, _ = call(x, scores, ids)
        times = {}
        for label, fn in (
            ("stock", lambda: _stock(marlin, workspace, weights, x, ids, scores)),
            ("v9", lambda: call(x, scores, ids)),
        ):
            timed = torch.cuda.CUDAGraph()
            with torch.cuda.graph(timed):
                timed_output = fn()
            samples = []
            for _ in range(5):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                timed.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            times[label] = sorted(samples)[2]
            timed.reset()
            del timed_output
        graph.reset()
        rows.append(
            dict(
                m=m,
                relative_l2=rel,
                maxdiff=maxdiff,
                graph_maxdiff=0,
                sm120_contended_us=times,
            )
        )
        print(json.dumps(rows[-1]), flush=True)
        del reference, eager, captured, captured_map
    for offset in (0, 64):
        m = 256
        x, ids, scores = _inputs(m, device, 771 + offset)
        ids.copy_(torch.arange(8, device=device, dtype=ids.dtype).expand(m, -1))
        reference = _stock(marlin, workspace, weights, x, ids, scores)
        state["local_start"] = offset
        ids.add_(offset)
        eager, _ = call(x, scores, ids)
        rel = float(
            (eager.float() - reference.float()).norm() / reference.float().norm()
        )
        assert rel <= 0.015, ("skew", offset, rel)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, _ = call(x, scores, ids)
        graph.replay()
        assert torch.equal(captured, eager)
        graph.reset()
        rows.append(
            dict(m=m, skew=True, local_start=offset, relative_l2=rel, graph_maxdiff=0)
        )
        print(json.dumps(rows[-1]), flush=True)
    print(
        json.dumps(dict(device=torch.cuda.get_device_name(), results=rows)), flush=True
    )


if __name__ == "__main__":
    main()
