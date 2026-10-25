"""The attention lane's vq8 verify-row patch (problems/muse_attention_decode/dev/vq8/patch_vq8_verify.py) with the
vq8.cu it checked (SHA-256 69132eea, the VQ8_CUDA literal of its S10 hand-off copy dev/vq8/out/s10_vq8.py, 9f487ac6),
since the lane's working vq8.cu moves on. Rows: verify 1 sliding, verify 16 full, verify 32 full.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util

V = Path(__file__).resolve().parents[4] / 'problems/muse_attention_decode/dev/vq8'
ROWS = ('1s', '16f', '32f')


def patch(src, rows=ROWS):
    spec = importlib.util.spec_from_file_location('vq8_patch', V / 'patch_vq8_verify.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    copy = (V / 'out/s10_vq8.py').read_text()
    cu = next(ast.literal_eval(n.value) for n in ast.parse(copy).body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'VQ8_CUDA')
    assert hashlib.sha256(cu.encode()).hexdigest() == mod.VQ8_SHA
    mod.VQ8_CU = cu
    return mod.patch(src, rows)
