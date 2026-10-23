"""Bounded V9 serving dispatch vs stock Marlin; real NVFP4, eager/fresh graphs."""

import json
from types import MethodType, SimpleNamespace

import torch
from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm import (
    FusedPostExpertNormWeightedSumTRTLLM,
)
from sglang.srt.layers.muse_spark_moe import prepare
from test_muse_moe_v4_layer import _weights, _inputs, _marlin_state, _stock


def main():
    device = torch.device("cuda")
    weights = _weights(device)
    state = prepare(weights, local_start=0)
    marlin, workspace = _marlin_state(weights, device)
    norm = (1 + weights["post_norm_w"].float()).bfloat16()

    # Invoke the real production branch with ONLY the canonical bank, as the
    # full-size loader does. Selecting Marlin here must fail on missing tensors.
    # The separate reference owns the original serialized Marlin tensors.
    owner = SimpleNamespace(_muse_v4_state=state)
    from sglang.srt.layers.muse_prefill_expert import PrefillExpertPlan
    assert PrefillExpertPlan.from_layer(owner).state is state, "second expert bank"
    owner.num_local_experts = 32
    owner.global_num_experts = 256
    owner.hidden_size = 4096
    owner._expert_map = torch.full((256,), -1, device=device, dtype=torch.int32)
    owner._expert_map[:32] = torch.arange(32, device=device)
    post_norm = object()
    owner._post_expert_norm = post_norm
    owner._post_expert_norm_weight = norm
    owner._fused_finalize_op = FusedPostExpertNormWeightedSumTRTLLM(
        topk=8, hidden_dim=4096, eps=1e-8, weight_first=True
    )
    for name in ("_use_muse_v4", "_marlin_expert_outputs", "_finalize_dense_expert_outputs"):
        setattr(owner, name, MethodType(getattr(MuseSpark1xFp4Experts, name), owner))

    def call(x, scores, ids):
        output = MuseSpark1xFp4Experts._forward_marlin(
            owner, x, scores, ids, post_norm
        )
        return output

    rows = []
    for m in (1, 8, 128, 256, 257, 512, 513, 752, 753, 3012, 16384):
        x, ids, scores = _inputs(m, device, 91000 + m)
        reference = _stock(marlin, workspace, weights, x, ids, scores)
        eager = call(x, scores, ids)
        from sglang.srt.layers.muse_spark_serving import run_v9_dense
        _, mapping = run_v9_dense(state, x, scores, ids)
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
            captured = call(x, scores, ids)
            _, captured_map = run_v9_dense(state, x, scores, ids)
        for _ in range(2):
            graph.replay()
            assert torch.equal(captured, eager), (m, "graph mismatch")
            assert torch.equal(captured_map, expected_map)
            x.mul_(0.5)
            eager = call(x, scores, ids)
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
                custom_producer=owner._use_muse_v4(m),
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
        eager = call(x, scores, ids)
        rel = float(
            (eager.float() - reference.float()).norm() / reference.float().norm()
        )
        assert rel <= 0.015, ("skew", offset, rel)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call(x, scores, ids)
        graph.replay()
        assert torch.equal(captured, eager)
        graph.reset()
        rows.append(
            dict(m=m, skew=True, local_start=offset, relative_l2=rel, graph_maxdiff=0)
        )
        print(json.dumps(rows[-1]), flush=True)
    # A whole EP rank may have no local routes. Its dense producer can leave
    # remote rows unwritten; finalize must keep their garbage out of the sum.
    for m in (256, 753):
        state["local_start"] = 0
        x, ids, scores = _inputs(m, device, 99000 + m)
        ids.fill_(200)
        eager = call(x, scores, ids)
        assert torch.count_nonzero(eager).item() == 0, (m, "remote EP rows")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call(x, scores, ids)
        for local in (False, True, False):
            ids.copy_(torch.arange(8, device=device).expand(m, -1) + (0 if local else 200))
            x.mul_(0.5)
            graph.replay()
            eager = call(x, scores, ids)
            assert torch.equal(captured, eager), (m, "changed EP ownership")
            reference = _stock(marlin, workspace, weights, x, ids, scores)
            delta = (eager.float() - reference.float()).norm()
            rel = float(delta / reference.float().norm().clamp_min(1e-12))
            assert rel <= 0.015, (m, "all-local/remote", rel)
        graph.reset()
        rows.append(dict(m=m, all_remote_exact=True, dynamic_ep_graph_exact=True))
        print(json.dumps(rows[-1]), flush=True)
    print(
        json.dumps(dict(device=torch.cuda.get_device_name(), results=rows)), flush=True
    )


if __name__ == "__main__":
    main()
