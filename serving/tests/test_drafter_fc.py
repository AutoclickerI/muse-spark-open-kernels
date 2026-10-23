"""TP1/TP2 drafter-FC equality and CUDA-graph check.

Run TP1 with ``PYTHONPATH=serving/sglang/python python serving/tests/test_drafter_fc.py``.
Run TP2 with ``PYTHONPATH=serving/sglang/python python -m torch.distributed.run
--nproc_per_node 2 serving/tests/test_drafter_fc.py``.

The TP2 case uses the same NCCL all-gather contract as ColumnParallelLinear:
each rank owns a contiguous output slice, then every rank receives the full
projection consumed by DFlash.  The reference uses the unchanged full weight.
"""

import os

import torch
import torch.distributed as dist

from sglang.srt.layers import linear as linear_mod
from sglang.srt.layers.linear import ColumnParallelLinear


class _Group:
    def __init__(self):
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()

    def all_gather(self, value, dim=-1):
        parts = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(parts, value)
        return torch.cat(parts, dim=dim)


class _Parallel:
    def __init__(self, group):
        self.tp_group = group
        self.tp_size = group.world_size
        self.tp_rank = group.rank_in_group


def main():
    assert torch.cuda.is_available()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl")
    group = _Group() if world > 1 else type("Group", (), {"world_size": 1, "rank_in_group": 0})()
    parallel = _Parallel(group)
    # ColumnParallelLinear resolves and uses this context for both construction
    # and its output all-gather.  The production DFlash model is constructed
    # under the same runtime context by draft_tp_context().
    old_get_parallel = linear_mod.get_parallel
    linear_mod.get_parallel = lambda: parallel
    try:
        hidden_in, hidden_out, tokens = 5120, 4096, 32
        gen = torch.Generator(device="cuda").manual_seed(7)
        x = torch.randn(tokens, hidden_in, device="cuda", dtype=torch.bfloat16, generator=gen)
        full_weight = torch.randn(
            hidden_out, hidden_in, device="cuda", dtype=torch.bfloat16, generator=gen
        ) / 32
        if world == 1:
            patched = torch.nn.functional.linear(x, full_weight)
            graph_diff = 0.0
        else:
            module = ColumnParallelLinear(
                hidden_in,
                hidden_out,
                bias=False,
                params_dtype=torch.bfloat16,
                gather_output=True,
                quant_config=None,
                parallel_group="tp",
            ).cuda()
            local = hidden_out // world
            module.weight.data.copy_(full_weight[rank * local : (rank + 1) * local])
            patched = module(x)[0]
            graph_stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(graph_stream):
                for _ in range(3):
                    module(x)
                torch.cuda.synchronize()
                graph.capture_begin()
                graph_out = module(x)[0]
                graph.capture_end()
                graph.replay()
            graph_stream.synchronize()
            graph_diff = (graph_out.float() - patched.float()).abs().max().item()
        reference = torch.nn.functional.linear(x, full_weight)
        eager_diff = (patched.float() - reference.float()).abs().max().item()
        if rank == 0:
            print(
                f"drafter fc world={world}: eager max|diff|={eager_diff:.6g}; "
                f"graph max|diff|={graph_diff:.6g}"
            )
        # BF16 GEMM's accumulation tile can differ when the output-N tile is
        # halved.  This is one BF16 ulp at the largest local values; the full
        # FP32 weight is unchanged and graph replay is bit-identical to eager.
        assert eager_diff <= 0.0625
        assert graph_diff == 0.0
    finally:
        linear_mod.get_parallel = old_get_parallel
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
