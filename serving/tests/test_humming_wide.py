"""V11 producer, distinct global scales, remote masking, changed-input graph."""

import json

import torch
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.muse_spark_humming import WideExperts, unpack_expert
from sglang.srt.layers.muse_spark_moe import prepare
from test_muse_moe_v4_layer import _inputs, _marlin_state, _stock, _weights


def main():
    dev = torch.device("cuda")
    weights = _weights(dev)
    state = prepare(weights, local_start=0)
    # Check signed, subnormal and zero block scales without changing precision.
    from sglang.srt.layers.muse_spark_moe import _repack

    scales = torch.arange(256, device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    scales = torch.where(torch.isfinite(scales.float()), scales.float(), 0).to(
        scales.dtype
    )
    sf = scales.repeat(16, 1)
    raw = torch.randint(256, (16, 2048), dtype=torch.uint8, device=dev)
    tile = _repack(raw, sf, torch.arange(16, device=dev).view(1, 16))
    w, s = unpack_expert(tile, gated=False)
    code = torch.stack((raw & 15, raw >> 4), -1).view(16, 256, 16)
    code = torch.where(sf.float()[..., None] < 0, code ^ 8, code)
    code = torch.where(sf.float()[..., None] == 0, 0, code).view(16, 2048, 2)
    expected_w = code[..., 0] | (code[..., 1] << 4)
    assert torch.equal(w, expected_w)
    assert torch.equal(s.float(), sf.float().abs())
    # Exact effective weights from a full serving layer.
    w, s = unpack_expert(state["w13p"][0], gated=True)
    assert torch.equal(w, weights["w13"][0])
    assert torch.equal(s.float(), weights["w13_sf"][0].float())
    w, s = unpack_expert(state["w2p"][0], gated=False)
    assert torch.equal(w, weights["w2"][0])
    print("canonical roundtrip exact", flush=True)
    wide = WideExperts(state)
    # Force the complete admission/graph-timing ABI on sm120 with compiled
    # portable kernels. Only device query and constructor are substituted.
    import types
    from unittest.mock import patch

    import sglang.srt.layers.muse_spark_humming as adapter

    admission = types.SimpleNamespace(_humming_candidate=True, _muse_v4_state=state)
    tx, ti, ts = _inputs(256, dev, 117)
    from sglang.kernels.jit.utils import override_jit_cuda_arch

    with (
        override_jit_cuda_arch(12, 0),
        patch.object(adapter, "WideExperts", return_value=wide),
        patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
    ):
        adapter.maybe_run_wide(admission, tx, ts, ti)
    print("wide admission selected", admission._humming_wide is not None, flush=True)
    stock, workspace = _marlin_state(weights, dev)
    norm = (weights["post_norm_w"].float() + 1).bfloat16()

    def run(x, ids, scores):
        y, mapping = wide(x, ids)
        return fused_post_expert_norm_weighted_sum(
            y,
            scores,
            norm_weight=norm,
            eps=1e-8,
            weight_first=True,
            slot_expert_ids=mapping,
        )

    for m in (256, 272):
        x, ids, scores = _inputs(m, dev, 123 + m)
        expected = _stock(stock, workspace, weights, x, ids, scores)
        got = run(x, ids, scores)
        error = float((got.float() - expected.float()).norm() / expected.float().norm())
        assert error < 0.015, error
        for _ in range(3):
            run(x, ids, scores)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = run(x, ids, scores)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, got)
        x.mul_(0.75)
        g.replay()
        torch.cuda.synchronize()
        changed = run(x, ids, scores)
        assert torch.equal(out, changed)
        print(
            json.dumps(
                {
                    "m": m,
                    "relative_l2": error,
                    "maxdiff": float((got.float() - expected.float()).abs().max()),
                    "graph_maxdiff": 0,
                }
            ),
            flush=True,
        )
        g.reset()
        # Compare graph latency, with conversion outside all timed calls.
        from sglang.srt.layers.muse_spark_moe import run_layer

        timings = {}
        for name, fn in [
            ("grouped", lambda: run_layer(state, x, scores, ids, return_dense=True)),
            ("wide", lambda: wide(x, ids)),
        ]:
            for _ in range(3):
                fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(4):
                    fn()
            values = []
            for _ in range(7):
                a, b = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                a.record()
                graph.replay()
                b.record()
                b.synchronize()
                values.append(a.elapsed_time(b) * 1000 / 4)
            timings[name] = sorted(values)[len(values) // 2]
            graph.reset()
        print(json.dumps({"m": m, "sm120_producer_us": timings}), flush=True)


if __name__ == "__main__":
    main()
