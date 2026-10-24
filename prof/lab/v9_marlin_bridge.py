"""Lossless, timed V9-tile to Marlin NVFP4 layout conversion.

Only packed E2M1 words and Marlin's byte-scale representation are written.
The destination is a stream-owned, one-layer workspace, not a weight cache.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _weights_tile(P, O, N: tl.constexpr, GATED: tl.constexpr):
    group, kb, e = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kt = kb * 32 + tl.arange(0, 32)[:, None]
    src = tl.arange(0, 64)[None, :]
    tile = group * (2 if GATED else 1) + src // 32
    base = (e * (N // 16) + tile) * 9216
    base += kt // 32 * 1024 + kt // 4 % 8 * 128 + src % 32 // 4 * 16
    base += kt % 4 * 4 + src % 4
    words = tl.load(P + base, mask=(src < (64 if GATED else 32)), other=0)
    C: tl.constexpr = 64 if GATED else 32
    ix = tl.arange(0, C)[None, :]
    lane, half = ix % 32, ix // 32
    rr = lane // 4
    sl, j = lane % 4 // 2, lane % 2
    out = tl.full((32, C), 0, tl.uint32)
    for s in tl.static_range(8):
        v = (s % 4) * 2 + s // 4
        ri = half if GATED else tl.full((1, C), v // 4, tl.int32)
        ti = v // 4 if GATED else 0
        off = v % 4
        index = tl.broadcast_to(ti * 32 + rr * 4 + (off // 2) * 2, (32, C))
        a = tl.gather(words, index, 1)
        b = tl.gather(words, index + 1, 1)
        slot = sl * 4 + j * 2 + ri
        rot = tl.where(slot == 0, 0, tl.where(slot == 1, 4,
              tl.where(slot == 2, 10, tl.where(slot == 3, 2,
              tl.where(slot == 4, 6, tl.where(slot == 5, 12,
              tl.where(slot == 6, 24, 30)))))))
        code = tl.full((32, C), 0, tl.uint32)
        for bit in tl.static_range(4):
            dst = 6 if bit == 0 else 7 if bit == 1 else 8 if bit == 2 else 15
            pos = (dst + (off % 2) * 16 + rot) % 32
            word = tl.where(slot < 3, a, b)
            word = tl.where(slot >= 6, tl.where(((0x60306030 >> pos) & 1) != 0, a, b), word)
            code |= ((word >> pos) & 1) << bit
        out |= code << (4 * s)
    r = group * 16 + half * 4096
    target = (kt * (N // 64) + r // 64) * 128 + lane * 4 + r % 64 // 16
    tl.store(O + e * (4096 * N // 8) + target, out)


@tr.jit
def _scales(P, O, N: tl.constexpr, GATED: tl.constexpr, B: tl.constexpr):
    e = tl.program_id(1)
    i = tl.program_id(0) * B + tl.arange(0, B)
    # Stock's 8x8 permutation followed by [0,2,1,3] within each four.
    t = i // 4 * 4 + (i % 4 // 2) + (i % 2) * 2
    canonical = t // 64 * 64 + (t % 64 // 8) + (t % 8) * 8
    q, r = canonical // N, canonical % N
    if GATED:
        tile, ri = r % 4096 // 8, r // 4096
    else:
        tile, ri = r // 16, r // 8 % 2
    off = (e * (N // 16) + tile) * 36864 + 32768
    off += q // 32 * 512 + (r % 8 * 4 + q % 4) * 16 + (q // 4 % 8) * 2 + ri
    raw = tl.load(P + off)
    # Match stock's FP16 conversion, exponent shift and high-byte extraction.
    value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    bits = (value * 128).to(tl.float16).to(tl.uint16, bitcast=True)
    encoded = ((bits.to(tl.uint32) << 1) >> 8).to(tl.uint8)
    tl.store(O + e * (256 * N) + i, encoded)


def v9_to_marlin(bank, weight_out, scale_out, *, gated):
    """Populate caller-owned Marlin buffers on the current stream, every call."""
    n = 8192 if gated else 4096
    experts = bank.shape[0]
    if (bank.dtype != torch.uint8 or tuple(bank.shape[1:]) != (n // 16, 36864)
            or weight_out.dtype != torch.int32 or tuple(weight_out.shape) != (experts, 256, n * 2)
            or scale_out.dtype != torch.float8_e4m3fn or tuple(scale_out.shape) != (experts, 256, n)):
        raise ValueError("V9/Marlin bridge expects released packed 4096-wide expert layouts")
    _weights_tile[(256, 8, experts)](bank.view(torch.int32), weight_out, n, gated, num_warps=8)
    _scales[(256 * n // 256, experts)](bank, scale_out.view(torch.uint8), n, gated, 256)
