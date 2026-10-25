"""Build the whole-layer paired A/B lab on S10 (dev/s10/ablate_lab.py.inc; mechanism and variants in its docstring).

    python3 problems/muse_decode_layer/dev/s10/ablate_build.py
writes prof/lab/decode_ablate.py (payload; S8's one-LOP3 NT1 source embedded) and
problems/muse_decode_layer/lab/lab_decode_ablate.py (S10 + tools/h200_lab.py's wrapper; send it with
tools/okc_run.py send: a decode check run).
"""
from pathlib import Path
import ast
import hashlib
import sys

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
D = R / 'problems/muse_decode_layer'
sys.path.insert(0, str(R / 'tools'))
import h200_lab  # noqa: E402
from okc_run import rejected_words  # noqa: E402

S8 = D / 'cand_r3_best7_s8.py'
HOST = D / 'cand_r3_best7_s10.py'
HOST_SHA = '6688fad999783d53a4660d85d40ee82862b87a209e03d289e44c95e41a3ffbc1'


def value(text, key):
    return next(ast.literal_eval(n.value) for n in ast.parse(text).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == key)


def main():
    s8 = S8.read_text()
    body = (H / 'ablate_lab.py.inc').read_text()
    body = body.replace('__NT1_CUDA__', repr(value(s8, 'S4_NT1_CUDA')), 1).replace('__NT1_CPP__', repr(value(s8, 'NT1_CPP')), 1)
    payload = R / 'prof/lab/decode_ablate.py'
    payload.write_text(body)
    host = HOST.read_text()
    assert hashlib.sha256(host.encode()).hexdigest() == HOST_SHA
    embedded = [h200_lab.embed(payload)]
    src = host.rstrip() + '\n\n' + h200_lab.WRAPPER.format(
        names=', '.join(f'{n} {s}' for n, s, _ in embedded), base='S10 ' + HOST_SHA[:8], payloads=embedded,
        budget=300 - h200_lab.MARGIN_S, limit=h200_lab.LIMIT)
    assert not rejected_words(src), rejected_words(src)
    compile(src, 'lab', 'exec')
    out = D / 'lab/lab_decode_ablate.py'
    out.write_text(src)
    print(payload.relative_to(R), embedded[0][1], out.relative_to(R), hashlib.sha256(src.encode()).hexdigest())


if __name__ == '__main__':
    main()
