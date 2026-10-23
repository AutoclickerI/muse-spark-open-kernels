"""Two NCCL ranks execute all eight real TP8 projection slices, eager and graph.

Each physical rank owns four consecutive virtual TP8 ranks. This checks slice
ordering and collective ownership without claiming an eight-GPU NVLink test.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from sglang.srt.layers.muse_proj import MuseProjection
from sglang.srt.layers.muse_spark_serving import ProjectionMethod
from sglang.srt.layers import muse_spark_serving as adapter


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    torch.manual_seed(102411)
    rows = []
    for kind, shape in (("pooler", (4096, 8192)), ("unpooler", (8192, 4096))):
        w = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / shape[1] ** 0.5
        slices = w.chunk(8)[rank * 4 : (rank + 1) * 4]
        ops = [MuseProjection(s, kind, tokens=(), budget_s=0) for s in slices]
        for m in (8, 32, 128):
            x = torch.randn((m, shape[1]), device="cuda", dtype=torch.bfloat16)

            def call():
                local = torch.cat([op(x) for op in ops], -1)
                gathered = [torch.empty_like(local) for _ in range(2)]
                dist.all_gather(gathered, local)
                return torch.cat(gathered, -1)

            expected = F.linear(x, w)
            eager = call()
            rel = float(
                (eager.float() - expected.float()).norm() / expected.float().norm()
            )
            assert rel < 0.01, (rank, kind, m, rel)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = call()
            for _ in range(3):
                graph.replay()
                assert torch.equal(out, eager)
                x.mul_(0.5)
                eager = call()
            graph.reset()
            rows.append(dict(kind=kind, m=m, relative_l2=rel, graph_maxdiff=0))
    w = torch.randn((8192, 1024), device="cuda", dtype=torch.bfloat16) / 32
    w.mul_(rank + 1)
    op = MuseProjection(w, "o", tokens=(), budget_s=0)

    class Original:
        def apply(self, layer, x, bias=None):
            return F.linear(x, layer.weight, bias)

    method = ProjectionMethod(Original(), op)
    for m in (8, 32, 128, 256, 753):
        x = torch.randn((m, 1024), device="cuda", dtype=torch.bfloat16)
        reference = F.linear(x, w)
        dist.all_reduce(reference)
        with patch.object(adapter, "projection_supported", return_value=m <= 256):

            def call():
                y = method.apply(SimpleNamespace(weight=w), x)
                # The unchanged RowParallelLinear caller owns this reduction.
                dist.all_reduce(y)
                return y

            eager = call()
            assert torch.equal(eager, reference)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = call()
            graph.replay()
            assert torch.equal(out, eager)
            graph.reset()
        rows.append(dict(kind="o", m=m, maxdiff=0, graph_maxdiff=0))
    if rank == 0:
        print(json.dumps(dict(physical_tp=2, virtual_tp=8, results=rows)), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
