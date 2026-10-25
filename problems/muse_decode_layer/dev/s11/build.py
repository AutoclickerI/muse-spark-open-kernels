"""S11: the frozen S10 file (6688fad9) + patch modules, applied in order (each module's patch(src) -> src).

Default patches:
- balanced split-K GEMV (glue lane, `problems/muse_bf16_proj/dev/gemv_bal/patch_gemv_bal.py`): qkv 64/7 and
  pool_router 64/6 at t = 1/8/16/32, unpooler 64/2 at t = 16/32. Whole layer on S11, paired (e605532e): S10's
  projections cost +2.9 / +2.4 / +3.1 / +1.9% at decode 1 / 16 / 32 / verify 1; every role contributes.
- N_full's heavy N32 kernel with the lean K loop and W13 gate/up rows interleaved in prepare so SwiGLU finishes in
  registers (MoE lane, `dev/s7n/nf_patch_e200b0ee.py`, cell nf_cell e86f6fb3; kinds 26/21; nf_ilv permutes the u8
  W13 rows of s5_hh_base, same codes): lean K 858ff2fa M128 0.9613, M256 0.9426 vs N_swap; interleave 701d2867
  M128 0.9923, M256 0.9838 on top; bit-equal.
- gate split at decode 1 S/F, decode 16 S, decode 32 F (glue lane, `problems/muse_bf16_proj/dev/gate_split/
  patch_gate_split.py`): the gate GEMV beside attention, S 4; chains 9b7ffda0 / 997bd383 -2.0 / -1.8, -0.6 / -0.4,
  -0.8 / -2.2, -1.9 us per layer.
- boundary glue at t > 128 loads its vectors after the wait (glue lane, `dev/glue_big/patch_glue_late.py`; 3345699f
  -1.1 us per boundary at t = 256).
- vq8 page-ring kernel at verify 16 full / 32 full (attention lane, `patch_vq8_ring.py`, kernel out/vq8_c262e4c5.cu,
  via dev/s11/patch_vq8_rows.py; 5928ced8: 141.6 vs 191.9, 250.6 vs 378.9 us; in the layer e42953c4: vq8 off
  +4.0 / +7.1% of verify 16 / 32, verify-1 rows inside A:A so not routed).
- pinned tuner choices (`dev/s11/patch_pin.py`): every projection row at t = 1-256 and attention at decode 16 / 32 set
  to the whole-layer sweep's winner (4bea7724) instead of per-run timing of one stage alone.
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
PATCHES = [R / 'problems/muse_bf16_proj/dev/gemv_bal/patch_gemv_bal.py', R / 'problems/muse_bf16_proj/dev/gate_split/patch_gate_split.py',
           R / 'problems/muse_bf16_proj/dev/glue_big/patch_glue_late.py', D / 'dev/s7n/nf_patch_e200b0ee.py',
           H / 'patch_vq8_rows.py', H / 'patch_pin.py']
LABS = dict(gemv_bal='26fd09a7 (standalone) + 5cefdbc4 (S8 chains) + e605532e (S11 whole layer, paired)',
            nfull='858ff2fa lean K (M128 0.9613, M256 0.9426) + 701d2867 interleave (0.9923, 0.9838), bit-equal',
            vq8='5928ced8 page ring (projection chain) + e42953c4 (whole layer)', gate_split='9b7ffda0 + 997bd383',
            glue_late='3345699f', pins='4bea7724 (whole-layer sweep)')


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
