"""DFlash2 coalesced candidate gather: rank order, int64 bits, NCCL and graph.

Run with PYTHONPATH=serving/sglang/python and torchrun --nproc_per_node=2
inside serving/harness/img. TP1 is also supported. This compares the real
candidate_topk entry point with stock's separate typed gathers, including
near-ties and optional partition logsumexp. The id bit-pattern case catches
numeric packing that silently loses int64 ids or canonicalizes NaNs.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.models import dflash


def main():
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl")

    def gather(value, dim=-1):
        if world == 1:
            return value
        # Same rank-major all_gather/cat contract as GroupCoordinator.
        received = torch.empty(
            (world * value.shape[0], *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        dist.all_gather_into_tensor(received, value.contiguous())
        return torch.cat(received.chunk(world, dim=0), dim=dim)

    old_parallel, old_gather = (
        dflash.get_parallel,
        dflash.tensor_model_parallel_all_gather,
    )
    old_project, old_topk, old_lse = (
        dflash._project_candidate_logits,
        dflash._radix_topk,
        dflash.lilicorr_topk_lse,
    )
    dflash.get_parallel = lambda: SimpleNamespace(tp_size=world)
    dflash.tensor_model_parallel_all_gather = gather
    try:
        torch.manual_seed(23 + rank)
        head = SimpleNamespace(
            weight=torch.empty(1, 1, device="cuda", dtype=torch.bfloat16),
            org_vocab_size=202048,
            shard_indices=SimpleNamespace(
                num_org_elements=202048 // world,
                org_vocab_start_index=rank * (202048 // world),
            ),
        )
        logits = None
        dflash._project_candidate_logits = lambda *a, **kw: logits
        # Both paths consume the same local top-k operation; ties therefore
        # exercise exactly the rank/candidate ordering at the communication seam.
        dflash._radix_topk = lambda x, k: torch.topk(x, k, dim=-1)
        dflash.lilicorr_topk_lse = lambda x, k: (
            *torch.topk(x, k, dim=-1),
            torch.logsumexp(x.float(), -1),
        )

        def stock(hidden, partition=False):
            vals, ids = dflash._radix_topk(logits, 16)
            lse = torch.logsumexp(logits.float(), -1) if partition else None
            if world == 1:
                return ids.long(), vals, lse
            ids = ids.long() + head.shard_indices.org_vocab_start_index
            gv, gi = gather(vals.float()), gather(ids)
            top, sel = torch.topk(gv, 16, dim=-1)
            if partition:
                lse = torch.logsumexp(gather(lse[:, None]), -1)
            return torch.gather(gi, -1, sel), top, lse

        def patch(hidden, partition=False):
            return dflash.candidate_topk(hidden, head, 16, with_partition=partition)

        rows = []
        for c in (1, 4, 16, 32):
            n = 7 * c
            hidden = torch.empty((n, 1), device="cuda", dtype=torch.bfloat16)
            logits = torch.randn(n, 1024, device="cuda", dtype=torch.bfloat16)
            logits[0].fill_(0)  # exact rank and candidate ties
            for partition in (False, True):
                ref = stock(hidden, partition)
                got = patch(hidden, partition)
                assert all(
                    a is None and b is None or torch.equal(a, b)
                    for a, b in zip(ref, got)
                )
                for _ in range(3):
                    patch(hidden, partition)
                torch.cuda.synchronize()
                stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(graph):
                        gout = patch(hidden, partition)
                stream.synchronize()
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                assert all(
                    a is None and b is None or torch.equal(a, b)
                    for a, b in zip(ref, gout)
                )
                graph.reset()

            # Event times for graph replay include the local top-k and full
            # communication seam, but exclude the common LM-head matmul.
            def capture(fn):
                for _ in range(3):
                    fn(hidden)
                torch.cuda.synchronize()
                stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(graph):
                        for _ in range(16):
                            fn(hidden)
                stream.synchronize()
                return graph

            def elapsed(graph):
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                if world > 1:
                    dist.barrier()
                begin.record()
                for _ in range(100):
                    graph.replay()
                end.record()
                end.synchronize()
                return begin.elapsed_time(end) * 10 / 16

            gs, gp = capture(stock), capture(patch)
            pairs = [(elapsed(gs), elapsed(gp)) for _ in range(9)]
            before = sorted(row[0] for row in pairs)[4]
            after = sorted(row[1] for row in pairs)[4]
            gs.reset()
            gp.reset()
            rows.append(
                {
                    "c": c,
                    "stock_us": before,
                    "patch_us": after,
                    "eager_maxdiff": 0,
                    "graph_maxdiff": 0,
                }
            )

        # Exercise the unmocked projection and FlashInfer local top-k with a
        # full judge-size vocabulary shard; this catches dtype/layout seams
        # hidden by the focused candidate-order tests above.
        (
            dflash._project_candidate_logits,
            dflash._radix_topk,
            dflash.lilicorr_topk_lse,
        ) = old_project, old_topk, old_lse
        head.weight = (
            torch.randn(202048 // world, 1024, device="cuda", dtype=torch.bfloat16) / 32
        )
        hidden = torch.randn(28, 1024, device="cuda", dtype=torch.bfloat16)
        logits = old_project(
            hidden,
            head,
            num_org=head.shard_indices.num_org_elements,
            use_quant_head=False,
        )
        ref, got = stock(hidden), patch(hidden)
        assert all(torch.equal(a, b) for a, b in zip(ref[:2], got[:2]))
        for _ in range(3):
            patch(hidden)
        torch.cuda.synchronize()
        stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph):
                full_out = patch(hidden)
        stream.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(ref[:2], full_out[:2]))
        hidden.normal_()
        logits = old_project(
            hidden,
            head,
            num_org=head.shard_indices.num_org_elements,
            use_quant_head=False,
        )
        changed_ref = stock(hidden)
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(changed_ref[:2], full_out[:2]))
        graph.reset()

        dflash._project_candidate_logits = lambda *a, **kw: logits
        # NCCL must preserve every id bit, including words that look like fp32
        # NaNs. Test beyond float32's exact integer range as well.
        if world > 1:
            fake_ids = (
                torch.tensor(
                    [[16777217, 2143289345, 4294967295, 2**40 + 3] * 4],
                    device="cuda",
                    dtype=torch.int64,
                )
                + rank * 2**42
            )
            vals = torch.arange(16, device="cuda", dtype=torch.float32)[None]
            dflash._radix_topk = lambda x, k: (vals, fake_ids)
            head.shard_indices.org_vocab_start_index = 0
            logits = torch.empty(1, 16, device="cuda", dtype=torch.float32)
            hidden = torch.empty(1, 1, device="cuda", dtype=torch.bfloat16)
            ref, got = stock(hidden), patch(hidden)
            assert all(torch.equal(a, b) for a, b in zip(ref[:2], got[:2]))
        if rank == 0:
            print(
                json.dumps(
                    {
                        "world": world,
                        "result": "exact eager and graph",
                        "timings": rows,
                    },
                    indent=2,
                )
            )
    finally:
        dflash.get_parallel, dflash.tensor_model_parallel_all_gather = (
            old_parallel,
            old_gather,
        )
        (
            dflash._project_candidate_logits,
            dflash._radix_topk,
            dflash.lilicorr_topk_lse,
        ) = old_project, old_topk, old_lse
        # Release captured NCCL graph references before communicator teardown.
        if "graph" in locals():
            del graph
        import gc

        gc.collect()
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
