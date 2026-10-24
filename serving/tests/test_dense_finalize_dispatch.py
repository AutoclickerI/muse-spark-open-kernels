"""Keep stock's fused large-M math; CuTe owns small M and explicit fused AR."""
from types import SimpleNamespace as NS

import torch
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum as stock,
)
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm import (
    FusedPostExpertNormWeightedSumTRTLLM,
)
from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts


def main():
    torch.manual_seed(29161)
    count = 0
    for h in (512, 4096):
        for weight_first in (True, False):
            norm = object()
            w = torch.randn(h, device='cuda', dtype=torch.bfloat16).mul_(.1).add_(1)
            owner = NS(hidden_size=h, _post_expert_norm=norm,
                       _post_expert_norm_weight=w, _post_expert_norm_eps=1e-8,
                       _post_expert_norm_weight_first=weight_first)
            op = FusedPostExpertNormWeightedSumTRTLLM(
                topk=8, hidden_dim=h, eps=1e-8, weight_first=weight_first)
            owner._fused_finalize_op = op
            for m in (8, 32, 128, 256, 512, 753, 2048):
                x = torch.randn(m, 8, h, device='cuda', dtype=torch.bfloat16)
                s = torch.rand(m, 8, device='cuda'); s /= s.sum(-1, keepdim=True)
                ids = torch.zeros(m, 8, device='cuda', dtype=torch.int32)
                ids[:, 2::3] = -1
                x[:, 2::3] = float('nan')

                def call():
                    return MuseSpark1xFp4Experts._finalize_dense_expert_outputs(
                        owner, x, s, ids, norm, out_dtype=torch.bfloat16)

                def reference():
                    return stock(x, s, norm_weight=w, eps=1e-8,
                                 weight_first=weight_first, slot_expert_ids=ids)

                got, expected = call(), reference()
                assert torch.isfinite(got).all()
                if m >= 32:
                    assert torch.equal(got, expected), (h, m, weight_first)
                else:
                    torch.testing.assert_close(got, expected, rtol=.015, atol=.015)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    captured = call()
                for _ in range(3):
                    x.mul_(.75)
                    g.replay()
                    eager = call()
                    assert torch.equal(captured, eager), (h, m, 'graph')
                    if m >= 32:
                        assert torch.equal(captured, reference()), (h, m, 'stock')
                g.reset()
                count += 1
            # Even at M>=32, a supplied AR workspace transfers reduction
            # ownership. Returning stock's local sum here would skip the AR.
            workspace = object()
            seen = []

            def fused(**kwargs):
                assert kwargs['ar_workspace'] is workspace
                seen.append(kwargs['output'].shape)
                kwargs['output'].zero_()

            owner._fused_finalize_op = fused
            result = MuseSpark1xFp4Experts._finalize_dense_expert_outputs(
                owner, x[:128], s[:128], ids[:128], norm,
                out_dtype=torch.bfloat16, ar_workspace=workspace)
            assert seen == [torch.Size([128, h])] and not result.count_nonzero()
    print(f'{count} shapes/forms; M>=32 stock exact; 84 fresh graph replays exact; AR ownership retained')


if __name__ == '__main__':
    main()
