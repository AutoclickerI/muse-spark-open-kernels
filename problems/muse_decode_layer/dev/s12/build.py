"""S12 asserted composition on frozen S11 final. Each piece is independently removable.

python3 problems/muse_decode_layer/dev/s12/build.py
python3 problems/muse_decode_layer/dev/s12/build.py --without coded --out <path>
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
D = R / 'problems/muse_decode_layer'
BASE = D / 'cand_r3_best7_s11_final.py'
BASE_SHA = 'acda6c621b1f3b885f2111ea1b4482ae1d733e2f2493db2b7f0b852ccf66dfb7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pieces():
    return json.loads((H / 'pieces.json').read_text())


def compose(without=()):
    assert sha(BASE) == BASE_SHA
    src = BASE.read_text()
    meta = dict(base=str(BASE.relative_to(R)), base_sha256=BASE_SHA, patches=[])
    names = {p['name'] for p in pieces()}
    assert set(without) <= names, (without, names)
    for piece in pieces():
        if piece['name'] in without:
            continue
        path = R / piece['path']
        assert sha(path) == piece['sha256'], (path, sha(path), piece['sha256'])
        spec = importlib.util.spec_from_file_location('s12_' + piece['name'], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        src = mod.patch(src)
        meta['patches'].append(dict(piece, out_sha256=hashlib.sha256(src.encode()).hexdigest()))
    compile(src, '<S12>', 'exec')
    return src, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--without', nargs='*', default=[])
    ap.add_argument('--out', type=Path, default=D / 'cand_r3_best7_s12.py')
    ap.add_argument('--manifest', type=Path, default=H / 'manifest.json')
    args = ap.parse_args()
    src, meta = compose(args.without)
    import sys
    sys.path.insert(0, str(R / 'tools'))
    from okc_run import rejected_words
    assert not rejected_words(src), rejected_words(src)
    args.out.write_text(src)
    meta = dict(source=str(args.out.resolve().relative_to(R)), sha256=sha(args.out), **meta)
    args.manifest.write_text(json.dumps(meta, indent=2) + '\n')
    print(meta['source'], meta['sha256'])


if __name__ == '__main__':
    main()
