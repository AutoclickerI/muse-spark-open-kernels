"""Released-shape loader must own one legal Marlin bank, including prefill gaps.

Old canonical-only preparation deleted Marlin's parameters, so selecting the
new large-M winner crashed. The incoming raw-E4M3 repair does not solve
that mixed-producer layout ownership problem. This test
loads the real 32-expert NVFP4 geometry through the production quant method.
"""
import json
from types import MethodType

import torch
from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.layers.quantization.muse_spark_nvfp4_moe import MuseSparkNvFp4FusedMoEMethod
from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.runtime_context import get_context
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm import FusedPostExpertNormWeightedSumTRTLLM
from test_muse_moe_v4_layer import _weights, _inputs, _marlin_state, _stock
from types import SimpleNamespace


def main():
    weights = _weights(torch.device('cuda'))
    method = MuseSparkNvFp4FusedMoEMethod.__new__(MuseSparkNvFp4FusedMoEMethod)
    method.quant_config = SimpleNamespace(group_size=16, get_name=lambda: 'modelopt_fp4')
    method._moe_runner_backend = MoeRunnerBackend.MARLIN
    owner = torch.nn.Module()
    owner.quant_config = method.quant_config
    owner.params_dtype = torch.bfloat16
    owner.intermediate_size_per_partition = 4096
    owner.moe_runner_config = SimpleNamespace(is_gated=True)
    owner.num_local_experts, owner.global_num_experts = 32, 256
    owner.hidden_size, owner.local_expert_offset = 4096, 0
    with torch.device('cuda'):
        method.create_weights(owner, 32, 4096, 4096, torch.bfloat16)
    for name, key in (
        ('gate_up_proj', 'w13'), ('down_proj', 'w2'),
        ('gate_up_proj_weight_scale', 'w13_sf'), ('down_proj_weight_scale', 'w2_sf'),
        ('down_proj_weight_scale_2', 'w2_gs'),
    ):
        getattr(owner, name).data.copy_(weights[key])
    owner.gate_up_proj_weight_scale_2.data.copy_(torch.stack(
        (weights['w13_gs_gate'], weights['w13_gs_up']), 1))
    owner.gate_up_proj_input_scale.data.fill_(1)
    owner.down_proj_input_scale.data.fill_(1)
    post_norm = torch.nn.Module()
    post_norm.weight = torch.nn.Parameter(weights['post_norm_w'], requires_grad=False)
    owner.post_expert_norm = post_norm
    with get_context().override_server_args(moe_runner_backend='marlin'):
        method.process_weights_after_loading(owner)
    assert owner._muse_v4_state is None, 'unadmitted encoded-scale V9 bank'
    assert owner.w13_weight.dtype == owner.w2_weight.dtype == torch.int32
    assert not hasattr(owner, 'gate_up_proj'), 'second full bank retained'
    assert not hasattr(owner, 'down_proj'), 'second full bank retained'
    owner._expert_map = torch.full((256,), -1, device='cuda', dtype=torch.int32)
    owner._expert_map[:32] = torch.arange(32, device='cuda')
    owner._post_expert_norm = post_norm
    owner._post_expert_norm_weight = (1 + weights['post_norm_w'].float()).bfloat16()
    owner._post_expert_norm_eps = 1e-8
    owner._post_expert_norm_weight_first = True
    owner._fused_finalize_op = FusedPostExpertNormWeightedSumTRTLLM(
        topk=8, hidden_dim=4096, eps=1e-8, weight_first=True)
    for name in ('_use_muse_v4', '_marlin_expert_outputs', '_finalize_dense_expert_outputs'):
        setattr(owner, name, MethodType(getattr(MuseSpark1xFp4Experts, name), owner))
    reference, workspace = _marlin_state(weights, torch.device('cuda'))

    def call(x, scores, ids):
        return MuseSpark1xFp4Experts._forward_marlin(owner, x, scores, ids, post_norm)

    rows = []
    for m in (1, 8, 128, 256, 257, 511, 512, 513, 752, 753, 1024, 2048):
        x, ids, scores = _inputs(m, torch.device('cuda'), 62000 + m)
        ref, got = _stock(reference, workspace, weights, x, ids, scores), call(x, scores, ids)
        delta = (ref.float() - got.float()).abs()
        rel = float(delta.norm() / ref.float().norm())
        assert rel <= .015, (m, rel)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call(x, scores, ids)
        graph_drift = 0.0
        stock_self_drift = 0.0
        baseline_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(baseline_graph):
            baseline_captured = _stock(reference, workspace, weights, x, ids, scores)
        for _ in range(6):
            graph.replay()
            baseline_graph.replay()
            err = (captured.float() - got.float()).abs().max().item()
            graph_drift = max(graph_drift, err)
            # Marlin's cooperative partial reduction can itself vary between
            # capture and eager at large M; measure that control explicitly.
            ref = _stock(reference, workspace, weights, x, ids, scores)
            baseline_err = (baseline_captured.float() - ref.float()).abs().max().item()
            stock_self_drift = max(stock_self_drift, baseline_err)
            torch.testing.assert_close(captured, ref, rtol=.015, atol=.015)
            x.mul_(.75)
            got = call(x, scores, ids)
        graph.reset()
        baseline_graph.reset()
        assert graph_drift <= stock_self_drift, (m, 'graph drift exceeds stock',
                                                graph_drift, stock_self_drift)
        rows.append(dict(m=m, producer='Marlin', rel_l2=rel,
                         maxdiff=delta.max().item(), graph_maxdiff=graph_drift,
                         stock_self_graph_maxdiff=stock_self_drift))
        print(json.dumps(rows[-1]), flush=True)
    # No route on this EP rank may contribute uninitialized remote data.
    ids.fill_(200)
    assert torch.count_nonzero(call(x, scores, ids)) == 0
    print('ONE PACKED BANK; EAGER/GRAPH PASS; REMOTE ZERO', flush=True)


if __name__ == '__main__':
    main()
