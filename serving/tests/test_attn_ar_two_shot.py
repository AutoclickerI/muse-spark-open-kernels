"""Two logical push-plane ranks on GPU0, with real CUDA kernels and graphs.

Using separate streams and shared CUDA buffers tests every Lamport slot/phase
without using the other project's GPU. TP2 harness covers real process groups.
"""

import json
import statistics
from types import SimpleNamespace

import torch
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
)


def predicate_test():
    fake = object.__new__(CustomAllReduceV2)
    fake.disabled, fake.world_size, fake.device, fake.rank = (
        False,
        8,
        torch.device("cuda"),
        0,
    )
    fake.override_algo = None
    fake.max_push_size = 1 << 20
    fake.config = SimpleNamespace(num_push_blocks=132)
    fake._push_workspaces = [
        torch.empty(16 * fake.max_push_size, device="cuda", dtype=torch.uint8)
    ]
    fake.uses_one_shot_push = lambda n: 0 < n <= 128 * 1024
    fake.uses_two_shot_push = lambda n: 128 * 1024 < n <= 8 * 1024 * 1024
    group = SimpleNamespace(
        ca_comm=fake,
        _can_use_flashinfer_allreduce=lambda x: False,
        _resolve_outplace_all_reduce_method=lambda x: "ca",
    )
    original = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda *args: (9, 0)
    try:
        for tokens in (8, 32, 128, 256):
            assert ar.all_reduce_norm_residual_supported(group, tokens, 8192), tokens
        for tokens in (0, 257):
            assert not ar.all_reduce_norm_residual_supported(group, tokens, 8192)
        fake.max_push_size = 512 * 1024
        assert not ar.all_reduce_norm_residual_supported(group, 256, 8192)
        fake.max_push_size = 1 << 20
        group._resolve_outplace_all_reduce_method = lambda x: "pynccl"
        assert not ar.all_reduce_norm_residual_supported(group, 256, 8192)
    finally:
        torch.cuda.get_device_capability = original


def main():
    torch.cuda.set_device(0)
    predicate_test()
    nr._max_cluster_size = lambda: 8
    world, slots, counters = 2, 256 * 8192 * 2, 32
    # A stock 1024-thread AR grid must leave room for both logical ranks on
    # one GPU; real TP ranks each have their own device. The fused M256 norm
    # still launches256 CTAs, independently of this32-counter plane.
    workspaces = [
        torch.zeros(2 * world, slots, dtype=torch.uint8, device="cuda")
        for _ in range(world)
    ]
    comms = []
    # FFI borrows these pointers; retain every logical rank's owner tensor.
    counter_tensors = []
    for rank in range(world):
        counter = torch.zeros(counters, dtype=torch.uint32, device="cuda")
        counter_tensors.append(counter)
        obj = Communicator(
            push=PushPlane(
                rank,
                world,
                workspaces=workspaces,
                counter=counter.view(torch.uint8).view(-1, 4),
            )
        )
        comms.append(
            SimpleNamespace(
                obj=obj,
                world_size=world,
                has_multicast=False,
                uses_one_shot_push=lambda _: False,
                uses_two_shot_push=lambda _: True,
                override_algo=None,
            )
        )
    streams = [torch.cuda.Stream() for _ in range(world)]
    torch.cuda.synchronize()

    def parallel(fn, graph=None):
        if graph is None:
            result = []
            for rank in range(world):
                with torch.cuda.stream(streams[rank]):
                    result.append(fn(rank))
            torch.cuda.synchronize()
            return result
        with torch.cuda.graph(graph, stream=streams[0]):
            streams[1].wait_stream(streams[0])
            a = fn(0)
            with torch.cuda.stream(streams[1]):
                b = fn(1)
            streams[0].wait_stream(streams[1])
        return [a, b]

    results = []
    for hidden in (1024, 8192):
        for tokens in (8, 32, 128, 129, 256):
            torch.manual_seed(tokens + hidden)
            local = [
                torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
                for _ in range(world)
            ]
            for x in local:
                x[:, ::31] = 0.0
                x[:, 1::31] = -0.0
            stream = torch.randn(tokens, hidden, device="cuda")
            weight = (1 + 0.1 * torch.randn(hidden, device="cuda")).bfloat16()
            alpha, beta = [
                torch.randn(hidden, device="cuda").sigmoid() for _ in range(2)
            ]
            for router in (False, True):

                def stock(rank):
                    reduced = custom_all_reduce(
                        comms[rank].obj, local[rank], AllReduceAlgo.TWO_SHOT_PUSH
                    )
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

                def fused(rank):
                    return ar.all_reduce_norm_residual(
                        comms[rank],
                        local[rank],
                        stream,
                        weight,
                        alpha,
                        beta,
                        1e-8,
                        1e-5,
                        want_router=router,
                    )

                # Load the norm and both fused CUDA entry points before one
                # logical rank can wait for the other on this same host thread.
                nr.fused_norm_residual(
                    local[0],
                    stream,
                    1e-8,
                    1e-5,
                    pre_weight=weight,
                    alpha=alpha,
                    beta=beta,
                    want_router=router,
                )
                cluster, threads, _ = nr.norm_residual_launch_config(hidden, tokens)
                ar._jit_module(hidden, world, threads, cluster, False, router, True)
                # Prime allocation bins before launching cross-stream polling.
                scratch = [torch.empty_like(x) for x in local] + [
                    torch.empty_like(stream) for _ in range(8)
                ]
                del scratch
                torch.cuda.synchronize()
                expected, actual = parallel(stock), parallel(fused)

                def check(a, b):
                    for ra, rb in zip(a, b):
                        for ta, tb in zip(ra, rb):
                            if ta is not None:
                                assert torch.equal(
                                    ta.view(torch.uint8), tb.view(torch.uint8)
                                ), (hidden, tokens, router, (ta - tb).abs().max())

                check(expected, actual)
                for _ in range(2):
                    parallel(stock)
                    parallel(fused)
                graph = torch.cuda.CUDAGraph()

                def alternating(rank):
                    stock(rank)
                    return fused(rank)

                got = parallel(alternating, graph)
                for _ in range(20):
                    graph.replay()
                torch.cuda.synchronize()
                check(expected, got)
                # Changed inputs, same captured addresses and phase buffers.
                for x in local:
                    x.mul_(0.875).add_(0.125)
                torch.cuda.synchronize()
                expected = parallel(stock)
                graph.replay()
                torch.cuda.synchronize()
                check(expected, got)
                graph.reset()
                if hidden == 8192 and tokens in (32, 128, 256) and not router:
                    captures = {}
                    for name, fn in (("stock", stock), ("fused", fused)):
                        g = torch.cuda.CUDAGraph()
                        values = parallel(fn, g)
                        captures[name] = (g, values)
                    times = {name: [] for name in captures}
                    for rep in range(20):
                        order = (
                            ("stock", "fused") if rep % 2 == 0 else ("fused", "stock")
                        )
                        for name in order:
                            start, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            start.record()
                            captures[name][0].replay()
                            end.record()
                            end.synchronize()
                            times[name].append(start.elapsed_time(end) * 1000)
                    print(
                        "LOCAL_GPU0_LOGICAL_TP2",
                        tokens,
                        {name: statistics.median(t) for name, t in times.items()},
                        flush=True,
                    )
                    for g, values in captures.values():
                        g.reset()
                results.append(
                    dict(
                        hidden=hidden,
                        tokens=tokens,
                        router=router,
                        eager_maxdiff=0,
                        graph_maxdiff=0,
                        changed_graph_maxdiff=0,
                    )
                )
                print(json.dumps(results[-1]), flush=True)
    # Regression check for the pre-existing one-shot wrapper ABI (phase=None).
    for comm in comms:
        comm.uses_one_shot_push = lambda _: True
    local = [
        torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(world)
    ]
    stream = torch.randn(8, 1024, device="cuda")
    weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    alpha, beta = [torch.ones(1024, device="cuda") for _ in range(2)]
    for router in (False, True):
        nr.fused_norm_residual(
            local[0],
            stream,
            1e-8,
            1e-5,
            pre_weight=weight,
            alpha=alpha,
            beta=beta,
            want_router=router,
        )
        cluster, threads, _ = nr.norm_residual_launch_config(1024, 8)
        ar._jit_module(1024, world, threads, cluster, False, router, False)
        torch.cuda.synchronize()

        def one_stock(rank):
            summed = custom_all_reduce(
                comms[rank].obj, local[rank], AllReduceAlgo.ONE_SHOT_PUSH
            )
            return nr.fused_norm_residual(
                summed,
                stream,
                1e-8,
                1e-5,
                pre_weight=weight,
                alpha=alpha,
                beta=beta,
                want_router=router,
            )

        def one_fused(rank):
            return ar.all_reduce_norm_residual(
                comms[rank],
                local[rank],
                stream,
                weight,
                alpha,
                beta,
                1e-8,
                1e-5,
                want_router=router,
            )

        expected = parallel(one_stock)
        actual = parallel(one_fused)
        check(expected, actual)
        graph = torch.cuda.CUDAGraph()
        actual = parallel(one_fused, graph)
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        check(expected, actual)
        graph.reset()
    print("PASS one-shot phase=None ABI, eager/graph bitwise", flush=True)
    print(
        f"PASS {len(results)} real two-shot variants; 20 replays each; bitwise stock equality",
        flush=True,
    )


if __name__ == "__main__":
    main()
