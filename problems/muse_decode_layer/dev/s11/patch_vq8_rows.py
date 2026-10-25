"""The attention lane's vq8 page-ring patch (problems/muse_attention_decode/dev/vq8/patch_vq8_ring.py) on the rows that
win in the whole layer: verify 16 full and verify 32 full. In S11's leave-one-out lab (e42953c4, paired, 16 seeds)
taking vq8 off costs +50.4 / +122.9 us there (+4.0 / +7.1% of verify 16 / 32) and -0.11 / -0.02 us at verify 1
sliding / full (inside A:A), so the verify-1 rows stay on S10's attention.
"""
from pathlib import Path
import importlib.util

ROWS = ('16f', '32f')


def patch(src, rows=ROWS):
    path = Path(__file__).resolve().parents[4] / 'problems/muse_attention_decode/dev/vq8/patch_vq8_ring.py'
    spec = importlib.util.spec_from_file_location('vq8_ring', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.patch(src, rows)
