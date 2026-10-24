"""Serving prefill producer screen over the resident canonical V9 bank.

Run with ``tools/img python -u serving/tests/test_muse_prefill_expert.py``.
The stock side is Marlin; the producer uses raw17/V9 at 753 and per-call
canonical unpack/dequant plus DG-W8 at 3012 and 16384.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.muse_prefill_expert import PrefillExpertPlan
from sglang.srt.layers.muse_spark_moe import prepare
from test_muse_moe_v4_layer import _inputs, _marlin_state, _stock, _weights


TEST_TOKEN_COUNTS = (753, 3012, 16384)
REL_TOL = 1.5e-2


def _finalize(dense, scores, local_map, norm_weight):
    return fused_post_expert_norm_weighted_sum(
        dense,
        scores,
        norm_weight=norm_weight,
        eps=1.0e-8,
        weight_first=True,
        slot_expert_ids=local_map,
    )


def _graph_call(plan, x, scores, ids):
    # Warm all JITs and allocate graph-owned buffers before capture.
    plan(x, scores, ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dense, mapping = plan(x, scores, ids)
        out = (dense, mapping)
    graph.replay()
    torch.cuda.synchronize()
    return out[0].detach().clone(), out[1].detach().clone()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dev = torch.device("cuda")
    weights = _weights(dev)
    state = prepare(weights, local_start=0)
    plan = PrefillExpertPlan(state)
    marlin_layer, workspace = _marlin_state(weights, dev)
    norm = (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16)
    for m in TEST_TOKEN_COUNTS:
        x, ids, scores = _inputs(m, dev, 7000 + m)
        stock = _stock(marlin_layer, workspace, weights, x, ids, scores)
        dense, local_map = plan(x, scores, ids)
        got = _finalize(dense, scores, local_map, norm)
        graph_dense, graph_map = _graph_call(plan, x, scores, ids)
        graph_got = _finalize(graph_dense, scores, graph_map, norm)
        rel = float((got.float() - stock.float()).norm() / stock.float().norm().clamp_min(1e-8))
        graph_rel = float((graph_got.float() - got.float()).norm() / got.float().norm().clamp_min(1e-8))
        assert rel <= REL_TOL, (m, rel)
        assert graph_rel <= 1e-6, (m, graph_rel)
        # Atomic route scatter may choose a different valid order during graph
        # replay; the dense slots and final output must still agree.
        print({"m": m, "relative_l2": rel, "graph_relative_l2": graph_rel}, flush=True)


if __name__ == "__main__":
    main()
