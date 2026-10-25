"""S11: the frozen S10 file (6688fad9) + patch modules, applied in order (each module's patch(src) -> src).

Default patches:
- balanced split-K GEMV (glue lane, `problems/muse_bf16_proj/dev/gemv_bal/patch_gemv_bal.py`): qkv 64/7 and
  pool_router 64/6 at t = 1/8/16/32, unpooler 64/2 at t = 16/32. Whole layer on S11, paired (e605532e): S10's
  projections cost +2.9 / +2.4 / +3.1 / +1.9% at decode 1 / 16 / 32 / verify 1; every role contributes.
- lean K loop in N_full's heavy N32 kernel (MoE lane, `dev/s7n/nf_patch.py`, cell nf_cell 97dcd77b; kinds 20/21):
  858ff2fa M128 0.9613, M256 0.9426 vs N_swap, bit-equal.
- vq8 at verify 1 sliding, verify 16 full, verify 32 full (attention lane,
  `problems/muse_attention_decode/dev/vq8/patch_vq8_verify.py` with its checked vq8.cu 69132eea, dev/s11/patch_vq8.py; 44aa67ba: 15.30 vs 16.83, 153.9 vs 192.2,
  278.5 vs 381.4 us after a projection stand-in).
S10 is the base (not S8 + cells) so cell edits made since S10 cannot drift in.

    python3 problems/muse_decode_layer/dev/s11/build.py [--patch MODULE ...] [--out PATH]
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
D = R / 'problems/muse_decode_layer'
spec = importlib.util.spec_from_file_location('s9_build', D / 'dev/s9/build.py')
s9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s9)

BASE = D / 'cand_r3_best7_s10.py'
BASE_SHA = '6688fad999783d53a4660d85d40ee82862b87a209e03d289e44c95e41a3ffbc1'
PATCHES = [R / 'problems/muse_bf16_proj/dev/gemv_bal/patch_gemv_bal.py', D / 'dev/s7n/nf_patch.py',
           H / 'patch_vq8.py']
LABS = dict(gemv_bal='26fd09a7 (standalone) + 5cefdbc4 (S8 chains) + e605532e (S11 whole layer, paired)',
            lean_k='858ff2fa (M128 0.9613, M256 0.9426, bit-equal)', vq8='44aa67ba (projection chain, cold L2)')


def compose(patches, tag='S11'):
    s = BASE.read_text()
    assert hashlib.sha256(s.encode()).hexdigest() == BASE_SHA
    meta = dict(base=str(BASE.relative_to(R)), base_sha256=BASE_SHA, patches=[], labs=dict(LABS))
    for path in patches:
        path = path.resolve()
        s = s9.load(path).patch(s)
        meta['patches'].append(dict(path=str(path.relative_to(R)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    out_sha256=hashlib.sha256(s.encode()).hexdigest()))
    head = '"""Best7 ('
    s = s9.sub(s, head, head + f'{tag} = S10 + ' + ''.join(p['path'].rsplit('/', 1)[-1].removesuffix('.py') + ' + '
                                                          for p in meta['patches']))
    return s, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch', type=Path, nargs='*', default=PATCHES)
    ap.add_argument('--out', type=Path, default=D / 'cand_r3_best7_s11.py')
    a = ap.parse_args()
    s, meta = compose(a.patch)
    s9.finish(s, a.out.resolve(), meta, H / 'manifest.json')


if __name__ == '__main__':
    main()
