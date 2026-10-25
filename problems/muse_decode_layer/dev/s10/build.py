"""S10: S9's composition (patch modules, N_full, small-M flags; dev/s9/build.py) on the current lane cells, plus
prepare's native pool starting the two modules it compiled serially after the pool.

Prepare (S9 cold, forced Hopper glue, ccache off, dev/s9/forced_audit.json): the pool ended at 68 s (prefill_score),
then the main thread compiled the cuBLASLt wrapper (49 s, in _tune_projections) and the layer attention ext (31 s,
in the attention tuner) one after the other. Both now start in the pool at prepare start; their later calls find the
module built. Nothing else changes: the same functions build the same modules.
N_full now carries the MoE lane's 880c3881 stream swap (heavy N32 on the side stream, launched first: 0.9799 at
M=128, 0.9882 at M=256 vs N_full) through dev/s7n/build.py.

    python3 problems/muse_decode_layer/dev/s10/build.py [--base PATH] [--patch MODULE ...] [--flags 1:95,8:69,16:69,32:69]
"""
from pathlib import Path
import argparse
import importlib.util

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
D = R / 'problems/muse_decode_layer'
spec = importlib.util.spec_from_file_location('s9_build', D / 'dev/s9/build.py')
s9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s9)

POOL = (
    ("    with ThreadPoolExecutor(max_workers=11) as native_pool:\n"
     "        grouped_future=native_pool.submit(_pf_grouped_module)\n",
     "    with ThreadPoolExecutor(max_workers=14) as native_pool:\n"
     "        grouped_future=native_pool.submit(_pf_grouped_module)\n"
     "        native_pool.submit(_layer_attn_ext)  # used by the attention tuner and the 'layer' rows\n"
     "        if _fa3_on_device(device):native_pool.submit(_lt_module)  # used by _tune_projections\n"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=D / 'cand_r3_best7_s8.py')
    ap.add_argument('--patch', type=Path, nargs='*', default=s9.PATCHES, help='modules with patch(src) -> src, applied in order')
    ap.add_argument('--flags', default='1:95,8:69,16:69,32:69')
    a = ap.parse_args()
    s, meta = s9.compose(a.base.resolve(), a.patch, s9.parse_flags(a.flags), tag='S10')
    for old, new in POOL:
        s = s9.sub(s, old, new)
    s = s9.sub(s, 'M128/M256 N_full (H200 c01b1916, 14f942d5); ',
               'M128/M256 N_full (H200 c01b1916, 14f942d5, swap 880c3881) + LT/layer-ext compiles in the prepare pool; ')
    meta.update(prepare='LT and the layer attention ext start in the native pool (max_workers 11 -> 14)')
    s9.finish(s, D / 'cand_r3_best7_s10.py', meta, H / 'manifest.json')


if __name__ == '__main__':
    main()
