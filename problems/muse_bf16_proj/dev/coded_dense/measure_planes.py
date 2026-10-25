"""Measure sparse exponent-plane size; CPU statistical model, not kernel timing."""
import json
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent


def measure(K, seed, even_origin, rotate_common):
    rng = np.random.default_rng(seed)
    # Same dense distribution as reference.generate_weights; RN-even BF16 conversion.
    x = rng.standard_normal((512, K), dtype=np.float32) / np.float32(K ** .5)
    raw = x.view(np.uint32)
    bits = ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16).astype(np.uint16)
    exp = ((bits >> 7) & 255).astype(np.int16)
    hist = np.stack([np.bincount(e.ravel(), minlength=256) for e in exp.reshape(-1, 16, K)])
    hist[:, (0, 255)] = 0
    cs = np.pad(hist.cumsum(1), ((0, 0), (1, 0)))
    bases = np.arange(2, 240, 2) if even_origin else np.arange(1, 240)
    full = cs[:, bases + 16] - cs[:, bases]
    offsets = np.arange(0, 9, 2 if even_origin else 1) if rotate_common else np.array([8])
    common_starts = bases[:, None] + offsets
    common_counts = cs[:, common_starts + 8] - cs[:, common_starts]
    common = common_counts.max(2)
    # Preserve maximum 16-exponent coverage; choose the best upper-eight coverage among ties.
    score = full * (16 * K + 1) + common
    selected = score.argmax(1)
    base = bases[selected]
    common_base = common_starts[selected, common_counts[np.arange(len(hist)), selected].argmax(1)]
    d = exp - np.repeat(base, 16)[:, None]
    esc = (d < 0) | (d >= 16)
    common_base = np.repeat(common_base, 16)[:, None]
    rare = ((exp < common_base) | (exp >= common_base + 8)) & ~esc
    # Each 16x64 tile is independently addressable; 32 subblocks of 32 weights.
    blocks = rare.reshape(-1, 16, K // 64, 64).transpose(0, 2, 1, 3).reshape(-1, 32, 32)
    occupied = blocks.any(2)
    n = exp.size
    escapes = int(esc.sum())
    # One uint32 presence mask and one uint32 sparse-plane pointer per 1024 weights.
    # Existing escape offsets/window metadata are common and excluded from both sizes.
    sparse_bits = n * 11 + len(blocks) * 64 + int(occupied.sum()) * 32 + escapes * 64
    old_bits = n * 12 + escapes * 64
    p = hist.sum(0) / hist.sum()
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return dict(K=K, seed=seed, even_origin=even_origin, rotate_common=rotate_common, weights=n, exponent_entropy=entropy,
                escape_fraction=escapes / n, extension_fraction=float(rare.mean()),
                occupied32_fraction=float(occupied.mean()),
                sparse_bits_per_weight=sparse_bits / n,
                current_bits_per_weight=old_bits / n,
                byte_ratio=sparse_bits / old_bits,
                native_decoder_built=False)


if __name__ == '__main__':
    rows = [measure(k, s, even, rot) for rot in (False, True) for even in (True, False) for k in (1024, 4096, 8192) for s in (61791, 14357, 92003)]
    out = dict(source='reference dense randn/sqrt(K), CPU NumPy draws rounded to BF16; statistical model',
               format='11 fixed bits + uint32 presence and prefix per 1024 + uint32 per occupied 32-weight extension block; original BF16 escapes',
               rows=rows)
    (H / 'plane_sizes.json').write_text(json.dumps(out, indent=2) + '\n')
    for r in rows:
        print(f"rot={r['rotate_common']} even={r['even_origin']} K={r['K']} seed={r['seed']} entropy={r['exponent_entropy']:.4f} rare={r['extension_fraction']:.4%} mask32={r['occupied32_fraction']:.4%} bits={r['sparse_bits_per_weight']:.4f} ratio={r['byte_ratio']:.4f}")
