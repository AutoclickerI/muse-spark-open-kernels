"""Runtime-M finalize must match static-M math without unseen-shape JITs.

The static control is the original pinned source copied to the evidence folder.
New uncaptured prefill lengths previously compiled in the request's TTFT path.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch
from sglang.kernels.ops.muse_spark_v12 import fused_post_expert_norm_weighted_sum_trtllm as opmod
from sglang.kernels.ops.muse_spark_v12 import fi_trtllm_moe_finalize_with_rmsnorm_cutedsl as dynamic

CONTROL = Path('/data2/jihoon/muse_kernel/harness/results/upstream-r01-evidence/finalize_static_control.py')
spec = importlib.util.spec_from_file_location('finalize_static_control', CONTROL)
static = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = static
spec.loader.exec_module(static)


def main():
    torch.manual_seed(2987)
    records = []
    for h in (1024, 4096):
        op = opmod.FusedPostExpertNormWeightedSumTRTLLM(
            topk=8, hidden_dim=h, eps=1e-8, weight_first=True)
        for m in (8, 32, 128, 256, 512, 753, 1537, 2509, 7939, 16384):
            x = torch.randn(m * 8, h, device='cuda', dtype=torch.bfloat16)
            weights = torch.rand(m, 8, device='cuda', dtype=torch.float32)
            weights /= weights.sum(-1, keepdim=True)
            mapping = torch.arange(m * 8, device='cuda', dtype=torch.int32)
            mapping[3::5] = -1
            norm = torch.randn(h, device='cuda', dtype=torch.bfloat16) * .1 + 1
            old, new = [torch.empty(m, h, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
            with patch.object(opmod, 'fi_trtllm_moe_finalize_with_rmsnorm_cutedsl', static):
                start = time.perf_counter()
                op.forward_cute_dsl(x, weights, mapping, norm, old)
                static_ms = (time.perf_counter() - start) * 1000
            before = len(dynamic._compile_cache)
            start = time.perf_counter()
            op.forward_cute_dsl(x, weights, mapping, norm, new)
            dynamic_ms = (time.perf_counter() - start) * 1000
            assert torch.equal(old, new), (h, m, (old.float() - new.float()).abs().max().item())
            if m in (2509, 7939, 16384):
                assert len(dynamic._compile_cache) == before, (h, m, 'new M compiled')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.forward_cute_dsl(x, weights, mapping, norm, new)
            for _ in range(3):
                x.mul_(.75)
                graph.replay()
                with patch.object(opmod, 'fi_trtllm_moe_finalize_with_rmsnorm_cutedsl', static):
                    op.forward_cute_dsl(x, weights, mapping, norm, old)
                assert torch.equal(old, new), (h, m, 'fresh graph')
            graph.reset()
            records.append(dict(h=h, m=m, maxdiff=0, graph_maxdiff=0,
                                cache_entries=len(dynamic._compile_cache),
                                static_first_call_ms=static_ms, dynamic_first_call_ms=dynamic_ms))
            print(json.dumps(records[-1]), flush=True)
    print('DYNAMIC M VS STATIC: EXACT; NO UNSEEN LARGE-M COMPILATION', flush=True)


if __name__ == '__main__':
    main()
