"""Add an isolated E comparison: board recoded scales versus best7 raw scales."""
from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
BEST7 = ROOT / 'problems/muse_decode_layer/cand_r3_best7_reset.py'
BEST6 = ROOT / 'problems/muse_decode_layer/cand_r3_best6.py'
BOARD = ROOT / 'problems/muse_decode_layer/cand_r3_v9.py'


def parts(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | {
        n.targets[0].id: ast.dump(n.value, include_attributes=False)
        for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}


raw = BEST6.read_text()
best7 = BEST7.read_text()
a, b = parts(raw), parts(best7)
names = ('V9_CUDA_SRC', 'V9_CPP_SRC', '_repack', 'v9_prepare', 'v9_kernel_fn')
assert all(a[n] == b[n] for n in names), 'raw NT1 differs from best7'
helpers = (HERE / 'driver.py.inc').read_text().split('\ndef lab(deadline,weights,host_state):', 1)[0]
body = ('"""Paired E scale-bank diagnosis; the board host remains unchanged."""\n'
        + 'DP_BEST6_SOURCE=' + repr(raw) + '\n' + helpers + '\n'
        + (HERE / 'scale_bank_driver.py.inc').read_text())
sys.path.insert(0, str(ROOT / 'tools'))
from okc_run import rejected_words
assert not rejected_words(body), rejected_words(body)
out = ROOT / 'prof/lab/decode_scale_bank.py'
out.write_text(body)
manifest = dict(payload=str(out.relative_to(ROOT)), sha256=hashlib.sha256(body.encode()).hexdigest(),
                raw_source_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                best7_sha256=hashlib.sha256(best7.encode()).hexdigest(),
                host_source_sha256=hashlib.sha256(BOARD.read_bytes()).hexdigest(),
                best7_identical_definitions=names, seeds=[172903, 180822, 188741],
                cases=[2, 3, 4, 5, 6, 7, 8, 9], rounds=6)
(HERE / 'scale_bank_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(out, manifest['sha256'])
