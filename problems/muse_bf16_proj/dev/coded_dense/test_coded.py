"""Bit-exact proof and GEMV correctness for the coded dense weights (coded.py).

    CUDA_VISIBLE_DEVICES=0 python3 problems/muse_bf16_proj/dev/coded_dense/test_coded.py [--seeds 3]

1. decode(encode(W)) == W bit for bit (the kernel's decode_step, escapes patched from the list) on every dense
   weight of reference.generate_weights (qkv_gate, o, pooler + router hi/lo, unpooler, its slice) for several
   seeds, and on adversarial tensors: every 16-bit pattern (NaN, inf, denormals, both zeros), all zeros, per-row
   scales over 2^-120..2^120, the window edges (base 1 and 239), constant tensors.
2. The GEMV's sum of split partials vs an FP32 product with the original W, at M = 1, 8, 16, 32 and several
   (S, NW, RT, PIPE) configs, including inf/NaN propagation.
"""
import argparse
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / 'muse_decode_layer'))
from coded import cd_decode, cd_encode, cd_gemv, cd_smem  # noqa: E402


def bits(w):
    return w.contiguous().view(torch.int16)


def check_exact(name, w):
    c = cd_encode(w)
    got = cd_decode(c)
    same = torch.equal(bits(got), bits(w))
    n = w.numel()
    coded_bytes = c['coded'].numel() + c['scales'].numel() * 4 + c['eoff'].numel() * 4 + c['nesc'] * 8
    print(f'EXACT {name:28s} {tuple(w.shape)} {"OK " if same else "FAIL"} esc={c["nesc"]} ({c["nesc"] / n * 1e6:.1f} ppm) '
          f'bytes={coded_bytes / (2 * n):.4f}x', flush=True)
    if not same:
        diff = (bits(got) != bits(w)).nonzero()[:5]
        print('  first diffs', diff.tolist(), bits(got)[tuple(diff.T)].tolist(), bits(w)[tuple(diff.T)].tolist())
    return same, c


def adversarial(dev):
    g = torch.Generator(device=dev).manual_seed(7)
    out = {}
    allbits = torch.arange(65536, device=dev, dtype=torch.int32)
    allbits = torch.where(allbits >= 32768, allbits - 65536, allbits).to(torch.int16)
    out['every 16-bit pattern'] = allbits[torch.randperm(65536, device=dev, generator=g)].view(256, 256).view(torch.bfloat16)
    out['zeros +/-'] = torch.where(torch.rand((64, 512), device=dev, generator=g) < 0.5, 0.0, -0.0).to(torch.bfloat16)
    rows = torch.randn((128, 1024), device=dev, generator=g)
    out['row scales 2^-120..2^120'] = (rows * torch.pow(2.0, torch.randint(-120, 121, (128, 1), device=dev, generator=g).float())).to(torch.bfloat16)
    out['denormal + tiny'] = (torch.randn((64, 512), device=dev, generator=g) * 2.0 ** -128).to(torch.bfloat16)
    out['huge (base 239 edge)'] = (torch.randn((64, 512), device=dev, generator=g) * 2.0 ** 125).to(torch.bfloat16)
    out['constant 1.0'] = torch.ones((32, 256), device=dev, dtype=torch.bfloat16)
    out['powers of two'] = torch.pow(2.0, torch.randint(-30, 30, (64, 512), device=dev, generator=g).float()).to(torch.bfloat16)
    wide = torch.randint(0, 65536, (256, 2048), device=dev, generator=g, dtype=torch.int32)
    out['uniform random bits'] = torch.where(wide >= 32768, wide - 65536, wide).to(torch.int16).view(torch.bfloat16)
    mix = (torch.randn((256, 2048), device=dev, generator=g) / 45).to(torch.bfloat16)
    sel = torch.rand(mix.shape, device=dev, generator=g) < 0.01
    mix[sel] = out['uniform random bits'][sel]
    out['randn + 1% random bits'] = mix
    esc = (torch.randn((256, 2048), device=dev, generator=g) / 45).to(torch.bfloat16)
    sel = torch.rand(esc.shape, device=dev, generator=g) < 0.02
    wild = torch.randn(esc.shape, device=dev, generator=g) * torch.pow(2.0, torch.randint(-140, 40, esc.shape, device=dev, generator=g).float())
    esc[sel] = wild.to(torch.bfloat16)[sel]
    esc[:3, :5] = 0.0
    out['randn + 2% finite escapes'] = esc
    nan = (torch.randn((128, 1024), device=dev, generator=g) / 32).to(torch.bfloat16)
    nan[5, 100] = float('inf'); nan[17, 900] = float('-inf'); nan[40, 3] = float('nan')
    out['randn + inf/nan'] = nan
    return out


def ref_weights(seed, dev):
    import reference as R
    w = R.generate_weights(seed, device=dev)
    router = w['router'].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return {'qkv_gate': w['qkv_gate'], 'o': w['o'],
            'pool_router': torch.cat([w['pooler'], hi, lo], 0).contiguous(),
            'unpooler': w['unpooler'], 'unpooler_slice': w['unpooler'][:1024].contiguous()}


def check_gemv(name, w, c, dev, configs, ms=(1, 8, 16, 32)):
    ok = True
    for M in ms:
        x = (torch.randn((M, w.shape[1]), device=dev) * 0.5).to(torch.bfloat16)
        ref = x.double() @ w.double().T
        for S, NW, RT, PIPE in configs:
            if (w.shape[0] // 16) % (NW * RT) or S > w.shape[1] // 64 or cd_smem(M, w.shape[1], S) > SMEM_MAX:
                continue
            P, _ = cd_gemv(x, c, S, NW=NW, RT=RT, PIPE=PIPE, pdl=False)
            y = P.sum(0).double()
            fin = torch.isfinite(ref)
            same_nonfinite = torch.equal(torch.isnan(y[~fin]), torch.isnan(ref[~fin])) and torch.equal(y[~fin & ~torch.isnan(ref)], ref[~fin & ~torch.isnan(ref)])
            err = ((y[fin] - ref[fin]).norm() / ref[fin].norm().clamp_min(1e-30)).item() if fin.any() else 0.0
            good = err < 3e-5 and same_nonfinite and torch.isfinite(y[fin]).all().item()
            ok &= good
            if not good or (S, NW, RT, PIPE) == configs[0]:
                print(f'GEMV {name:20s} M={M:2d} S={S:2d} NW={NW} RT={RT} PIPE={PIPE} err={err:.2e} '
                      f'nonfinite={"ok" if same_nonfinite else "MISMATCH"} {"OK" if good else "FAIL"}', flush=True)
    return ok


SMEM_MAX = torch.cuda.get_device_properties(0).shared_memory_per_block_optin


def check_integer_gemv(dev):
    """Integer-valued w (escapes included: zeros, +-128 x 2^-7 = +-1.0 sits outside a 7-exponent window? no: values
    k * 2^-7 span exponents 120..126) and x: every partial sum is an integer multiple of 2^-7 below 2^24 * 2^-7, so
    any summation order is exact and the kernel must match the FP64 product exactly."""
    g = torch.Generator(device=dev).manual_seed(11)
    ok = True
    for N, K in ((2304, 8192), (1024, 4096), (8192, 1024)):
        w = (torch.randint(-127, 128, (N, K), device=dev, generator=g).float() / 128).to(torch.bfloat16)
        w[torch.rand((N, K), device=dev, generator=g) < 1e-3] = 3.0   # exponent 128: outside any window with 120
        c = cd_encode(w)
        for M in (1, 8, 16, 32):
            x = torch.randint(-8, 9, (M, K), device=dev, generator=g).to(torch.bfloat16)
            ref = (x.double() @ w.double().T)
            for S, NW, RT, PIPE in ((1, 4, 1, 2), (7, 4, 2, 3), (16, 8, 1, 4)):
                if (N // 16) % (NW * RT) or S > K // 64 or cd_smem(M, K, S) > SMEM_MAX:
                    continue
                P, _ = cd_gemv(x, c, S, NW=NW, RT=RT, PIPE=PIPE, pdl=False)
                same = torch.equal(P.double().sum(0), ref) if S == 1 else torch.equal(P.double().sum(0), ref)
                ok &= same
                if not same:
                    print(f'INT GEMV {N}x{K} M={M} S={S} NW={NW} RT={RT} PIPE={PIPE} MISMATCH '
                          f'max abs {(P.double().sum(0) - ref).abs().max().item()}', flush=True)
        print(f'INT GEMV {N}x{K} esc={c["nesc"]} {"OK" if ok else "FAIL"}', flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    dev = 'cuda'
    ok = True
    configs = [(7, 4, 1, 3), (1, 4, 1, 2), (6, 4, 2, 2), (11, 8, 1, 4), (3, 2, 2, 3), (16, 4, 1, 4)]
    ok &= check_integer_gemv(dev)
    for seed in range(1234, 1234 + args.seeds):
        for name, w in ref_weights(seed, dev).items():
            good, c = check_exact(f'{name} seed {seed}', w)
            ok &= good
            if seed == 1234:
                ok &= check_gemv(name, w, c, dev, configs)
            del c
        torch.cuda.empty_cache()
    for name, w in adversarial(dev).items():
        good, c = check_exact(name, w)
        ok &= good
        if name in ('row scales 2^-120..2^120', 'randn + 2% finite escapes', 'zeros +/-', 'powers of two',
                    'randn + inf/nan'):
            wf = w
            if name == 'row scales 2^-120..2^120':  # keep products inside FP32 range for the reference
                wf = w.float().clamp(-2.0 ** 60, 2.0 ** 60).to(torch.bfloat16)
                c = cd_encode(wf)
            ok &= check_gemv(name, wf, c, dev, [(1, 4, 1, 2), (2, 2, 2, 3), (4, 4, 1, 4)])
    print('ALL OK' if ok else 'FAILURES', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
