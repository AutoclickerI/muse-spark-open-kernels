"""Graph and FA3 parity check for the Muse TARGET_VERIFY adapter.

On SM90 this compares the adapter with ``flash_attn_with_kvcache`` under a
CUDA graph. The local SM120 image has no FA3 cubin, so it still runs the same
graph against an independent FP32 page-table reference and reports FA3 as
unavailable instead of silently calling a different kernel.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.muse_verify_attention import run_muse_verify_attention


def _reference(q, k, v, page_table, cache_seqlens, scale, window_left):
    bsz, q_len, heads, dim = page_table.shape[0], q.shape[0] // page_table.shape[0], q.shape[1], q.shape[2]
    hkv = k.shape[2]
    group = heads // hkv
    qf, kf, vf = q.float(), k.float().view(-1, hkv, dim), v.float().view(-1, hkv, dim)
    out = torch.empty((q.shape[0], heads, dim), dtype=torch.bfloat16, device=q.device)
    for b in range(bsz):
        base = int(cache_seqlens[b]) - q_len
        for t in range(q_len):
            p = base + t
            lo = 0 if window_left < 0 else max(0, p - window_left)
            ids = torch.arange(lo, p + 1, device=q.device)
            rows = page_table[b, ids.long()]
            parts = []
            for hk in range(hkv):
                logits = torch.einsum(
                    "gd,sd->gs", qf[b * q_len + t, hk * group : (hk + 1) * group], kf[rows, hk]
                ) * scale
                parts.append(torch.einsum("gs,sd->gd", logits.softmax(-1), vf[rows, hk]))
            out[b * q_len + t] = torch.cat(parts).to(torch.bfloat16)
    return out


def main():
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    bsz, q_len, heads, hkv, dim, ctx = 2, 8, 16, 2, 64, 128
    gen = torch.Generator(device=device).manual_seed(771)
    q = torch.randn((bsz * q_len, heads, dim), device=device, dtype=torch.bfloat16, generator=gen).to(torch.float8_e4m3fn)
    k = torch.randn((bsz * ctx, 1, hkv, dim), device=device, dtype=torch.bfloat16, generator=gen).to(torch.float8_e4m3fn).contiguous()
    v = torch.randn((bsz * ctx, 1, hkv, dim), device=device, dtype=torch.bfloat16, generator=gen).to(torch.float8_e4m3fn).contiguous()
    page_table = torch.arange(bsz * ctx, device=device, dtype=torch.int32).view(bsz, ctx)
    cache_seqlens = torch.tensor((ctx, ctx - 5), device=device, dtype=torch.int32)
    for window_left in (-1, 63):
        got = run_muse_verify_attention(
            q, k, v, page_table, cache_seqlens,
            softmax_scale=0.125, window_left=window_left,
        )
        ref = _reference(q, k, v, page_table, cache_seqlens, 0.125, window_left)
        err = float((got.float() - ref.float()).norm() / ref.float().norm())
        assert err < 3e-4, (window_left, err)
        out = torch.empty_like(got)
        for _ in range(2):
            run_muse_verify_attention(q, k, v, page_table, cache_seqlens,
                                      softmax_scale=0.125, window_left=window_left, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_muse_verify_attention(q, k, v, page_table, cache_seqlens,
                                      softmax_scale=0.125, window_left=window_left, out=out)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.uint8), got.view(torch.uint8))
        print(f"MUSE_VERIFY graph window={window_left}: rel_l2={err:.3e} PASS")

    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        cu_q = torch.arange(0, bsz * q_len + 1, q_len, device=device, dtype=torch.int32)
        fa3 = flash_attn_with_kvcache(
            q=q, k_cache=k, v_cache=v, page_table=page_table,
            cache_seqlens=cache_seqlens, cu_seqlens_q=cu_q,
            max_seqlen_q=q_len, softmax_scale=0.125, causal=True,
            window_size=(-1, -1), num_splits=0, ver=3,
        )
        if isinstance(fa3, tuple):
            fa3 = fa3[0]
        for _ in range(2):
            flash_attn_with_kvcache(
                q=q, k_cache=k, v_cache=v, page_table=page_table,
                cache_seqlens=cache_seqlens, cu_seqlens_q=cu_q,
                max_seqlen_q=q_len, softmax_scale=0.125, causal=True,
                window_size=(-1, -1), num_splits=0, ver=3,
            )
        fa_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fa_graph):
            fa_graph_out = flash_attn_with_kvcache(
                q=q, k_cache=k, v_cache=v, page_table=page_table,
                cache_seqlens=cache_seqlens, cu_seqlens_q=cu_q,
                max_seqlen_q=q_len, softmax_scale=0.125, causal=True,
                window_size=(-1, -1), num_splits=0, ver=3,
            )
            if isinstance(fa_graph_out, tuple):
                fa_graph_out = fa_graph_out[0]
        fa_graph.replay()
        torch.cuda.synchronize()
        got = run_muse_verify_attention(q, k, v, page_table, cache_seqlens,
                                        softmax_scale=0.125, window_left=-1)
        fa_err = float((got.float() - fa_graph_out.float()).norm() / fa_graph_out.float().norm())
        print(f"MUSE_VERIFY vs FA3 graph-ready call: rel_l2={fa_err:.3e} PASS")
    except Exception as exc:
        print(f"MUSE_VERIFY FA3 unavailable on this GPU: {type(exc).__name__}: {str(exc).splitlines()[0]}")


if __name__ == "__main__":
    main()
