"""FA3 QK-norm/KV-store fusion check on the local Blackwell GPUs.

Run from the repository root with::

    PYTHONPATH=serving/sglang/python python serving/tests/test_fused_qk_fa3.py

The reference is the stock sequence (BF16 Q/K norm and RoPE, followed by the
stock FP8 casts and cache writes).  The candidate is the Triton path used by
Muse's FA3 integration.  The graph replay checks that the same addresses and
cache locations are captured without host synchronization.
"""

import torch

from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm


def _attention(q, k, v, num_q_heads, num_kv_heads, head_dim):
    q = q.float().view(-1, num_q_heads, head_dim)
    k = k.float().view(-1, num_kv_heads, head_dim)
    v = v.float().view(-1, num_kv_heads, head_dim)
    k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    logits = torch.einsum("thd,shd->ths", q, k) * (head_dim**-0.5)
    return torch.einsum("ths,shd->thd", logits.softmax(dim=-1), v).reshape(
        q.shape[0], -1
    )


def main():
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    tokens, q_heads, kv_heads, head_dim = 32, 16, 2, 64
    op = FusedQKNorm(
        head_dim=head_dim, num_q_heads=q_heads, num_kv_heads=kv_heads, eps=1e-6
    ).to(device)
    gen = torch.Generator(device=device).manual_seed(123)
    qkv = torch.randn(
        tokens, (q_heads + 2 * kv_heads) * head_dim,
        device=device, dtype=torch.bfloat16, generator=gen
    )
    positions = torch.arange(tokens, device=device, dtype=torch.int64)
    phase = torch.randn(tokens, head_dim // 2, device=device, generator=gen)
    cos_sin = torch.cat((phase.cos(), phase.sin()), dim=-1).float().contiguous()
    locations = (torch.arange(tokens, device=device, dtype=torch.int64) * 3) + 17

    # Stock FA3: norm/RoPE remains BF16 until the attention backend casts Q/K/V.
    q_stock, k_stock = op.forward_native(qkv, positions, cos_sin)
    v_stock = qkv[:, (q_heads + kv_heads) * head_dim :]
    q_stock_fp8 = q_stock.to(torch.float8_e4m3fn)
    k_stock_fp8 = k_stock.float().clamp(-448, 448).to(torch.float8_e4m3fn)
    v_stock_fp8 = v_stock.float().clamp(-448, 448).to(torch.float8_e4m3fn)

    k_cache = torch.zeros(
        4096, kv_heads * head_dim, device=device, dtype=torch.float8_e4m3fn
    )
    v_cache = torch.zeros_like(k_cache)
    q_patch, k_patch = op._forward_triton(
        qkv, positions, cos_sin, (k_cache, v_cache, locations)
    )
    assert torch.equal(q_patch.view(torch.uint8), q_stock_fp8.view(torch.uint8))
    assert torch.equal(k_patch, k_stock)
    assert torch.equal(k_cache[locations], k_stock_fp8)
    assert torch.equal(v_cache[locations], v_stock_fp8)
    stock_out = _attention(
        q_stock_fp8,
        k_stock_fp8,
        v_stock_fp8,
        q_heads,
        kv_heads,
        head_dim,
    )
    patch_out = _attention(
        q_patch,
        k_cache[locations],
        v_cache[locations],
        q_heads,
        kv_heads,
        head_dim,
    )
    eager_diff = (patch_out - stock_out).abs().max().item()

    # Capture and replay the same FA3-side fusion.  Use a non-default stream,
    # matching SGLang's graph runner.
    graph_k = torch.zeros_like(k_cache)
    graph_v = torch.zeros_like(v_cache)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(3):
            op._forward_triton(qkv, positions, cos_sin, (graph_k, graph_v, locations))
        torch.cuda.synchronize()
        graph.capture_begin()
        graph_q, graph_k_norm = op._forward_triton(
            qkv, positions, cos_sin, (graph_k, graph_v, locations)
        )
        graph.capture_end()
        graph.replay()
    stream.synchronize()
    graph_diff = int(bool(max(
        (graph_q.view(torch.uint8) != q_patch.view(torch.uint8)).any().item(),
        (graph_k_norm != k_patch).any().item(),
        (graph_k[locations].view(torch.uint8) != k_cache[locations].view(torch.uint8))
        .any()
        .item(),
        (graph_v[locations].view(torch.uint8) != v_cache[locations].view(torch.uint8))
        .any()
        .item(),
    )))
    graph_out = _attention(
        graph_q,
        graph_k[locations],
        graph_v[locations],
        q_heads,
        kv_heads,
        head_dim,
    )
    graph_output_diff = (graph_out - stock_out).abs().max().item()
    print(
        f"FA3 QK/KV eager max|attention diff|={eager_diff:.6g}; "
        f"graph cache/query byte mismatch={graph_diff}; "
        f"graph max|attention diff|={graph_output_diff:.6g}"
    )
    assert eager_diff == 0.0
    assert graph_diff == 0
    assert graph_output_diff == 0.0


if __name__ == "__main__":
    main()
