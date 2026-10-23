"""Real push-plane attention AR/norm equivalence and graph interleaving.

Run in harness/img with torchrun --nproc_per_node=2 (or plain python for TP1).
The local PCIe GPUs exercise the unicast protocol; multicast is compile-only.
"""

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from sglang.kernels.ops.communication.all_reduce import (
    AllReduceAlgo,
    Communicator,
    PushPlane,
    custom_all_reduce,
)
from sglang.kernels.ops.muse_spark_v12 import all_reduce_norm_residual as ar
from sglang.kernels.ops.muse_spark_v12 import norm_residual as nr
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
    _allocate_symmetric_memory,
)


def main():
    rank, world = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method="env://" if world > 1 else "tcp://127.0.0.1:29508",
        rank=rank,
        world_size=world,
    )
    dev = torch.device("cuda", rank)
    # Force the actual topology predicate without allocating a new TP plane.
    fake = object.__new__(CustomAllReduceV2)
    fake.disabled, fake.world_size, fake.device, fake.rank = False, 8, dev, 0
    fake.config = SimpleNamespace(num_push_blocks=132)
    fake._push_workspaces = [torch.empty(128 * 8192 * 2, device=dev, dtype=torch.uint8)]
    fake.uses_one_shot_push = lambda size: 0 < size <= 128 * 1024
    group = SimpleNamespace(
        ca_comm=fake,
        _can_use_flashinfer_allreduce=lambda x: False,
        _resolve_outplace_all_reduce_method=lambda x: "ca",
    )
    capability = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda *a: (9, 0)
    try:
        assert ar.all_reduce_norm_residual_supported(group, 8, 8192)
        assert not ar.all_reduce_norm_residual_supported(group, 32, 8192)
        assert not ar.all_reduce_norm_residual_supported(group, 0, 8192)
        group._resolve_outplace_all_reduce_method = lambda x: "pynccl"
        assert not ar.all_reduce_norm_residual_supported(group, 8, 8192)
    finally:
        torch.cuda.get_device_capability = capability
    nccl = dist.new_group(backend="nccl") if world > 1 else None
    # Match Hopper's existing cluster limit while executing the real kernel.
    nr._max_cluster_size = lambda: 8
    if world == 1:
        # No collective at TP1; check the default loader remained unchanged.
        for hidden in (1024, 8192):
            local = torch.randn(8, hidden, device=dev, dtype=torch.bfloat16)
            stream = torch.randn(8, hidden, device=dev)

            def compute():
                return nr.fused_norm_residual(local, stream, 1e-8, 1e-5)

            expected = compute()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = compute()
            for _ in range(16):
                graph.replay()
            torch.cuda.synchronize()
            assert all(torch.equal(a, b) for a, b in zip(expected[:2], got[:2]))
            graph.reset()
        print("TP1 default-loader eager/graph exact; AR dispatch disabled")
        dist.destroy_process_group()
        return
    slots, counters = 128 * 8192 * 2, 132
    slab, handle = _allocate_symmetric_memory(2 * world * slots, dev, dist.group.WORLD)
    slab.zero_()
    peers = [
        handle.get_buffer(i, [2 * world, slots], torch.uint8) for i in range(world)
    ]
    counter = torch.zeros(counters, dtype=torch.uint32, device=dev)
    obj = Communicator(
        push=PushPlane(
            rank, world, workspaces=peers, counter=counter.view(torch.uint8).view(-1, 4)
        )
    )
    comm = SimpleNamespace(obj=obj, world_size=world, has_multicast=False)
    torch.cuda.synchronize()
    dist.barrier()
    results = []
    for hidden in (1024, 8192):
        for tokens in (8, 32, 128):
            torch.manual_seed(rank + tokens + hidden)
            local = torch.randn(tokens, hidden, device=dev, dtype=torch.bfloat16)
            local[:, ::31] = 0
            local[:, 1::31] = -0.0
            stream = torch.randn(tokens, hidden, device=dev, dtype=torch.float32)
            weight = (
                torch.randn(hidden, device=dev, dtype=torch.bfloat16).mul_(0.1).add_(1)
            )
            alpha, beta = [torch.randn(hidden, device=dev).sigmoid() for _ in range(2)]
            for router in (False, True):

                def standalone():
                    reduced = custom_all_reduce(obj, local, AllReduceAlgo.ONE_SHOT_PUSH)
                    return nr.fused_norm_residual(
                        reduced,
                        stream,
                        1e-8,
                        1e-5,
                        pre_weight=weight,
                        alpha=alpha,
                        beta=beta,
                        want_router=router,
                    )

                def fused():
                    return ar.all_reduce_norm_residual(
                        comm,
                        local,
                        stream,
                        weight,
                        alpha,
                        beta,
                        1e-8,
                        1e-5,
                        want_router=router,
                    )

                expected = standalone()
                nccl_sum = local.clone()
                dist.all_reduce(nccl_sum, group=nccl)
                custom_sum = custom_all_reduce(obj, local, AllReduceAlgo.ONE_SHOT_PUSH)
                assert torch.equal(nccl_sum, custom_sum)
                actual = fused()
                diffs = [
                    float((a - b).abs().max())
                    for a, b in zip(expected, actual)
                    if a is not None
                ]
                assert max(diffs) == 0, (hidden, tokens, router, diffs)
                # Norm's cluster arithmetic already differs slightly from eager
                # PyTorch: record this separately from collective-fusion equality.
                reduced = custom_all_reduce(
                    obj, local, AllReduceAlgo.ONE_SHOT_PUSH
                ).float()
                post = (
                    (
                        reduced
                        * torch.rsqrt(reduced.square().mean(-1, keepdim=True) + 1e-8)
                    )
                    .bfloat16()
                    .float()
                )
                residual = alpha * stream + beta * post
                native = (
                    residual
                    * torch.rsqrt(residual.square().mean(-1, keepdim=True) + 1e-5)
                    * weight.float()
                ).bfloat16()
                native_diff = float((actual[0].float() - native.float()).abs().max())
                for _ in range(3):
                    standalone()
                    fused()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    # Alternate generic AR and fused AR to catch stale phase slots.
                    standalone()
                    out = fused()
                for _ in range(32):
                    graph.replay()
                torch.cuda.synchronize()
                graph_diff = max(
                    float((a - b).abs().max())
                    for a, b in zip(expected, out)
                    if a is not None
                )
                assert graph_diff == 0, graph_diff
                # New input on the same addresses must not read an old epoch.
                local.mul_(0.5).add_(rank / 8)
                expected = standalone()
                graph.replay()
                torch.cuda.synchronize()
                assert all(
                    a is None and b is None or torch.equal(a, b)
                    for a, b in zip(expected, out)
                )
                graph.reset()

                def capture(fn):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(8):
                            fn()
                    return g

                def time(g):
                    dist.barrier()
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    for _ in range(50):
                        g.replay()
                    end.record()
                    end.synchronize()
                    return begin.elapsed_time(end) * 1000 / 400

                gs, gf = capture(standalone), capture(fused)
                samples = [(time(gs), time(gf)) for _ in range(5)]
                gs.reset()
                gf.reset()
                results.append(
                    dict(
                        hidden=hidden,
                        tokens=tokens,
                        router=router,
                        eager_maxdiff=max(diffs),
                        graph_maxdiff=graph_diff,
                        torch_maxdiff=native_diff,
                        separate_us=sorted(a for a, b in samples)[2],
                        fused_us=sorted(b for a, b in samples)[2],
                    )
                )
    if rank == 0:
        print(
            json.dumps(
                dict(tp=world, sm=torch.cuda.get_device_capability(), results=results),
                indent=2,
            )
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
