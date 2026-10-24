"""Existing column-sharded DFlash FC + NCCL stand-in for the multicast ABI.

Two physical NCCL ranks cover graph replay and DP/CP/group ownership guards.
The reference is the previously shipped sharded FC, so its GEMM is identical.
"""
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.models import dflash
from sglang.srt.runtime_context import get_context, get_parallel


class Group:
    def __init__(self, rank, world):
        self.rank_in_group, self.world_size = rank, world
        self.ca_comm = object()

    def all_gather(self, local, dim=-1):
        parts = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(parts, local)
        return torch.cat(parts, dim=dim)


def main():
    rank, world = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(.3, rank)
    dist.init_process_group('nccl')
    group = Group(rank, world)
    with (
        get_context().override_server_args(tp_size=world),
        get_parallel().override(tp_group=group, tp_size=world, tp_rank=rank,
                               attn_dp_size=1, attn_cp_size=1),
    ):
        torch.manual_seed(123 + rank)
        with torch.device('cuda'):
            fc = ColumnParallelLinear(5120, 4096, bias=False,
                params_dtype=torch.bfloat16, gather_output=True, parallel_group='tp')
        fc.weight.data.normal_(std=.01)
        saved = fc.weight.detach().clone()
        owner = SimpleNamespace(fc=fc)

        def gather(comm, local, out):
            assert comm is group.ca_comm
            out.copy_(group.all_gather(local))
            return out

        def supported(comm, *, num_tokens, n, device, dtype):
            assert comm is group.ca_comm
            return n == 4096 and 0 < num_tokens <= 256 and dtype == torch.bfloat16

        rows = []
        with patch.object(dflash, 'all_gather_supported', supported), patch.object(
            dflash, 'projection_all_gather', gather
        ):
            for m in (8, 32, 64, 128, 256):
                x = torch.randn(m, 5120, device='cuda', dtype=torch.bfloat16)
                ref = fc(x)[0]
                got = dflash.DFlashDraftModel._project_fc(owner, x)
                assert torch.equal(got, ref), (m, 'eager changed column FC')
                for _ in range(3):
                    dflash.DFlashDraftModel._project_fc(owner, x)
                dist.barrier()
                torch.cuda.synchronize()
                stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = dflash.DFlashDraftModel._project_fc(owner, x)
                for _ in range(3):
                    x.mul_(.75)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(captured, fc(x)[0]), (m, 'graph changed column FC')
                graph.reset()
                rows.append(dict(m=m, world=world, eager_maxdiff=0, graph_maxdiff=0))
            # Different context/sequence partitioning must retain the stock FC.
            for dp, cp in ((2, 1), (1, 2)):
                with get_parallel().override(attn_dp_size=dp, attn_cp_size=cp,
                                            attn_tp_size=world // (dp * cp)):
                    result = dflash.DFlashDraftModel._project_fc(owner, x)
                    assert isinstance(result, tuple) and torch.equal(result[0], fc(x)[0])
        assert torch.equal(fc.weight, saved), 'weights changed'
        if rank == 0:
            print(json.dumps(rows), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
