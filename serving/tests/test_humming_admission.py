"""Memory admission charges VMM backing before a partial expert layout."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers import muse_spark_humming as adapter
from sglang.srt.model_executor import forward_context as context


def main():
    x = torch.empty(256, 4096, device="meta", dtype=torch.bfloat16)
    layer = SimpleNamespace(_humming_candidate=True, _muse_v4_state={})
    # Raw free memory looks ample, but the unbacked KV reservation needs it.
    pool = SimpleNamespace(
        post_capture_active=True,
        post_capture_backed_bytes=1 << 30,
        size=2_000_000,
        get_kv_size_bytes=lambda: (6 << 30, 6 << 30),
    )
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(
            torch.cuda, "mem_get_info", return_value=(12 << 30, 141_000_000_000)
        ),
        patch.object(context, "has_forward_context", return_value=True),
        patch.object(context, "get_token_to_kv_pool", return_value=pool),
    ):
        assert adapter.maybe_run_wide(layer, x, None, None) is None
    assert layer._humming_wide is None
    assert adapter.maybe_run_wide(layer, x[:128], None, None) is None
    print("VMM pending target+draft KV rejects copy; M128 retains V9: passed")


if __name__ == "__main__":
    main()
