"""GPU0 regression: fresh graph inputs, prefix-only FP8 writes, async length relay."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import gc
import weakref

import torch

from sglang.srt.managers import dflash_kv_graph
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_prefix_valid_impl_fp8


@contextmanager
def local_capture():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield SimpleNamespace(stream=stream)


def test_kv_graph():
    torch.manual_seed(20261024)
    fc = torch.nn.Linear(80, 16, bias=False, dtype=torch.bfloat16, device="cuda")
    cache_k = torch.full((1024, 1, 16), 3, dtype=torch.float8_e4m3fn, device="cuda")
    cache_v = torch.full_like(cache_k, 5)
    workspace = SimpleNamespace(
        _proj_workspace=torch.empty((256, 16), dtype=torch.bfloat16, device="cuda"),
        _k_workspace=torch.empty((256, 1, 16), dtype=torch.bfloat16, device="cuda"),
        _v_workspace=torch.empty((256, 1, 16), dtype=torch.bfloat16, device="cuda"),
    )

    def append(*, target_hidden, cache_loc, cache_loc_2d, positions, commit_lens):
        n = target_hidden.shape[0]
        hidden = workspace._proj_workspace[:n]
        torch.mm(target_hidden, fc.weight.T, out=hidden)
        key, value = workspace._k_workspace[:n], workspace._v_workspace[:n]
        key.copy_((hidden.float() + positions[:, None].float() / 1000).view(n, 1, 16))
        value.copy_(hidden.neg().view(n, 1, 16))
        _set_kv_buffer_prefix_valid_impl_fp8(
            key.view(-1, 1, 16), value.view(-1, 1, 16), cache_k, cache_v,
            1.0, 1.0, cache_loc_2d, commit_lens, row_dim=16,
            k_scale_is_tensor=False, v_scale_is_tensor=False,
        )

    worker = SimpleNamespace(block_size=8, draft_model=SimpleNamespace(fc=fc),
                             model_runner=SimpleNamespace(device="cuda"),
                             _fused_kv_helper=workspace,
                             _append_target_hidden_to_draft_kv_by_loc=append)
    with patch.object(dflash_kv_graph, "graph_capture", local_capture), torch.inference_mode():
        graph = dflash_kv_graph.DFlashKVGraph(worker, 32)
        assert torch.equal(cache_k.view(torch.uint8), torch.full_like(cache_k, 3).view(torch.uint8))
        assert torch.equal(cache_v.view(torch.uint8), torch.full_like(cache_v, 5).view(torch.uint8))
        # A longer eager prefill replaces the helper's external workspaces.
        # Capture must retain the old storage rather than dangling its pointers.
        old = [weakref.ref(getattr(workspace, name)) for name in vars(workspace)]
        workspace._proj_workspace = torch.empty((4096, 16), dtype=torch.bfloat16, device="cuda")
        workspace._k_workspace = torch.empty((4096, 1, 16), dtype=torch.bfloat16, device="cuda")
        workspace._v_workspace = torch.empty_like(workspace._k_workspace)
        gc.collect()
        assert all(ref() is not None for ref in old)
        for repeat in range(3):
            for bs in range(1, 33):
                n = bs * 8
                args = dict(target_hidden=torch.randn((n, 80), dtype=torch.bfloat16, device="cuda"),
                            cache_loc=torch.randperm(1023, device="cuda")[:n] + 1,
                            positions=torch.randint(0, 131072, (n,), device="cuda"),
                            commit_lens=torch.randint(0, 9, (bs,), dtype=torch.int32, device="cuda"))
                args["cache_loc_2d"] = args["cache_loc"].view(bs, 8)
                cache_k.fill_(3); cache_v.fill_(5)
                append(**args)
                expected_k, expected_v = cache_k.clone(), cache_v.clone()
                cache_k.fill_(3); cache_v.fill_(5)
                graph(**args)
                assert torch.equal(cache_k.view(torch.uint8), expected_k.view(torch.uint8)), (repeat, bs, "k")
                assert torch.equal(cache_v.view(torch.uint8), expected_v.view(torch.uint8)), (repeat, bs, "v")
    print("KV graph: 96 changed-input replays, all 32 batch sizes, exact FP8 bytes and untouched guards")


def test_length_relay():
    algo = SimpleNamespace(is_some=lambda: True)
    pool = SimpleNamespace(req_to_token=torch.empty((40, 1), dtype=torch.int32, device="cuda"))
    future = FutureMap("cuda", algo, pool, needs_cpu_seq_lens=True)
    for i in range(100):
        bs = 1 + i % 32
        indices_cpu = torch.randperm(39)[:bs] + 1
        indices = indices_cpu.to("cuda")
        lengths = torch.randint(1, 131072, (bs,), dtype=torch.int64, device="cuda")
        future.publish(indices, lengths)
        batch = SimpleNamespace(spec_info=SimpleNamespace(future_indices=indices),
                                req_pool_indices_cpu=indices_cpu)
        future.resolve_seq_lens_cpu(batch)
        assert torch.equal(batch.seq_lens_cpu, lengths.cpu())
        assert torch.equal(batch.seq_lens, lengths)
        assert batch.seq_lens_sum == int(lengths.cpu().sum())
    print("Length relay: 100 publish/resolve cycles, changed and reordered pool rows exact")
    future = FutureMap("cuda", algo, pool, needs_cpu_seq_lens=False)
    future.publish(indices, lengths)
    future.resolve_seq_lens_cpu(batch)
    assert batch.seq_lens_cpu is None and batch.seq_lens_sum is None
    assert torch.equal(batch.seq_lens, lengths)
    print("FA3-style GPU-only relay: exact device lengths, no CPU mirror")


if __name__ == "__main__":
    test_kv_graph()
    test_length_relay()
