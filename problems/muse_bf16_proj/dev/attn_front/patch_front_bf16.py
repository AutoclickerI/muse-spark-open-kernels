"""qkv -> attention: BF16 qkv from one PDL Triton sum+cast kernel instead of torch's two kernels.

    python3 problems/muse_bf16_proj/dev/attn_front/patch_front_bf16.py <cand.py> <out.py>

`_attn_port_raw` (attn_main rows: decode 1 sliding/full, decode 16 full when 'auto' picks attn_main) turned the
qkv GEMV's FP32 split partials into BF16 with `partial.sum(0).to(bfloat16)`: two torch kernels, no PDL. Here
`_attn_sumcast_kernel` does it in one kernel, PDL-launched behind the GEMV, summing the splits in torch's order
with one rounding (bit-exact). attn_main itself keeps its normal launch. The layer attention ext (q_len 1 rows
that 'auto' resolves to 'layer', e.g. decode 16 sliding) gets the same BF16 input instead of looping over the
splits per call.

H200 87cd3a84 (prof/lab/attn_front.py, chain qkv GEMV -> attention -> PDL consumer, lab_cold_l2, paired vs S6;
all arms bit-exact): this recipe ('fuse') -4.21 / -4.46 / -5.25 us at b1 sliding / b1 full / b16 full, and
'laysum' -1.52 us at b16 sliding (layer ext). PDL-launching attn_main after it lost most of the gain at b1
(-0.11 / -2.27 us); the GEMV with an in-kernel split-K fixup + PDL attn_main was -5.86 us at b16 full only
(+0.34 / -2.69 at b1), so neither is applied.
"""
import hashlib
import sys
from pathlib import Path

SUMCAST = '''

@triton.jit
def _attn_sumcast_kernel(P, Y, PS, RS, N: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr, USE_PDL: tl.constexpr):
    """BF16 qkv [1, T, N]: the S FP32 split partials summed in split order (torch's order), one rounding."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    _pdl_start(USE_PDL)
    _pdl_wait(USE_PDL)
    c = b * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for s in tl.static_range(S):
        acc += tl.load(P + s * PS + t * RS + c, mask=c < N, other=0.0)
    tl.store(Y + t * N + c, acc.to(tl.bfloat16), mask=c < N)


def _attn_sumcast(partial, splits):
    """(BF16 [1, T, N], 1) from FP32 split partials [S, T, N] (unit stride along N)."""
    _, T, N = partial.shape
    y = torch.empty((1, T, N), device=partial.device, dtype=torch.bfloat16)
    _launch(_attn_sumcast_kernel, (T, triton.cdiv(N, 512)), partial, y, partial.stride(0), partial.stride(1), N=N,
            S=splits, BLOCK=512, num_warps=4, use_pdl=True)
    return y, 1
'''

OLD_RAW = '''    partial, splits = qkv
    raw = partial.sum(0) if splits > 1 else partial[0]
'''
NEW_RAW = '''    partial, splits = qkv
    if partial.dtype == torch.float32 and partial.stride(-1) == 1:
        partial, splits = _attn_sumcast(partial, splits)
    raw = partial.sum(0) if splits > 1 else partial[0]
'''
OLD_LAYER = 'def _layer_attention_custom(qkv, data):\n    qkv, splits = qkv\n'
NEW_LAYER = OLD_LAYER + ('    if qkv.dtype == torch.float32 and splits > 1 and qkv.stride(-1) == 1:\n'
                         '        qkv, splits = _attn_sumcast(qkv, splits)\n')


def _sub(src, old, new):
    if src.count(old) != 1:
        raise SystemExit(f'expected one {old[:60]!r}, found {src.count(old)}')
    return src.replace(old, new)


def patch(src: str) -> str:
    anchor = src.index('\n\n# (N, K) -> (BN, KS, BK)')  # after the candidate's _gemv_kernel
    src = src[:anchor] + SUMCAST + src[anchor:]
    src = _sub(src, OLD_RAW, NEW_RAW)
    return _sub(src, OLD_LAYER, NEW_LAYER)


if __name__ == '__main__':
    out = patch(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2], hashlib.sha256(out.encode()).hexdigest())
