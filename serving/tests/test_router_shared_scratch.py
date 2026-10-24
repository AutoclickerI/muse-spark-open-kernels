"""Real paired-router kernels share scratch across layers without stale counters.

Private per-layer scratch is the numerical reference. Each capture stream owns
one fixed M256 buffer; changed-input replay must match all private references.
"""
import json
import torch
from sglang.kernels.ops.muse_spark_v12.projection_hopper import RouterProjection, _module
from sglang.srt.managers.muse_workspace import router_workspace


def main():
    torch.manual_seed(271)
    routers = [RouterProjection(torch.randn(256, 8192, device='cuda') * .01)
               for _ in range(3)]
    private = [(torch.empty(16 * 17 * 64 * 64, device='cuda'),
                torch.zeros(16, device='cuda', dtype=torch.uint32)) for _ in routers]
    rows = []
    for m in (64, 128, 256):
        x = torch.randn(m, 8192, device='cuda', dtype=torch.bfloat16)
        def reference(router, work):
            out = torch.empty(m, 256, device='cuda')
            cfg = (64, 64, 4, 1, 8, 160 if m <= 128 else 256)
            _module().pair(x, router.pair, out, work, cfg)
            return out
        expected = [reference(r, w) for r, w in zip(routers, private)]
        eager = [r(x) for r in routers]
        assert all(torch.equal(a, b) for a, b in zip(expected, eager))
        # Dedicated capture streams must never alias another stream's counters.
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        workspaces, graphs, outputs = [], [], []
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                workspaces.append(router_workspace(x.device))
                for r in routers:
                    r(x)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs.append([r(x) for r in routers])
            graphs.append(graph)
        assert workspaces[0][0].data_ptr() != workspaces[1][0].data_ptr()
        for _ in range(4):
            x.mul_(.75)
            ready = torch.cuda.Event()
            ready.record()
            for stream, graph in zip(streams, graphs):
                stream.wait_event(ready)
                with torch.cuda.stream(stream):
                    graph.replay()
            for stream in streams:
                stream.synchronize()
            expected = [reference(r, w) for r, w in zip(routers, private)]
            for batch in outputs:
                assert all(torch.equal(a, b) for a, b in zip(expected, batch))
        for work in workspaces:
            assert torch.count_nonzero(work[1]) == 0, 'completion counters not reset'
        for graph in graphs:
            graph.reset()
        rows.append(dict(m=m, eager_maxdiff=0, graph_maxdiff=0,
                         distinct_stream_buffers=True, fresh_replays=8))
        print(json.dumps(rows[-1]), flush=True)


if __name__ == '__main__':
    main()
