"""Build a small-M experiment and a reviewable patch onto frozen S11 acda6c62.

Usage: python patch_s11.py --flags 1:95,8:69,16:69,32:65605 -o /tmp/s11_wpub.py
This creates the candidate, its standalone .cell.py, and a .patch against S11. No promotion is implied.
"""
import argparse
import ast
import difflib
import hashlib
from pathlib import Path

from build_cell import source

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'problems/muse_decode_layer/cand_r3_best7_s11_final.py'


def patch(base, flags):
    assert hashlib.sha256(base.encode()).hexdigest().startswith('acda6c62'), 'expected frozen S11 acda6c62'
    assert set(flags) == {1, 8, 16, 32}, 'provide the complete small-M flag map'
    cell = source(flags.values())
    begin = base.index('SMALLM_VAR_CUDA = ')
    end = base.index('def _s4_prepare_banks(', begin)
    out = base[:begin] + cell[cell.index('SMALLM_VAR_CUDA = '):].rstrip() + '\n\n\n' + base[end:]
    out = out.replace('SMALLM_FLAGS = {1: 95, 8: 69, 16: 69, 32: 69}', 'SMALLM_FLAGS = ' + repr(flags), 1)
    ast.parse(out)
    return out, cell


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--flags', required=True)
    ap.add_argument('-o', type=Path, required=True)
    args = ap.parse_args()
    flags = dict(tuple(map(int, row.split(':'))) for row in args.flags.split(','))
    base = BASE.read_text()
    candidate, cell = patch(base, flags)
    args.o.parent.mkdir(parents=True, exist_ok=True)
    paths = {args.o: candidate, args.o.with_suffix('.cell.py'): cell}
    rel = str(BASE.relative_to(ROOT))
    paths[args.o.with_suffix('.patch')] = ''.join(difflib.unified_diff(
        base.splitlines(True), candidate.splitlines(True), fromfile='a/' + rel, tofile='b/' + rel))
    for path, text in paths.items():
        path.write_text(text)
        print(path, hashlib.sha256(text.encode()).hexdigest()[:8])
    print('flags', flags)


if __name__ == '__main__':
    main()
