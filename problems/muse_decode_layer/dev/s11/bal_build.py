"""Build the balanced-GEMV whole-layer paired A/B lab on S11 (dev/s11/bal_lab.py.inc on dev/s10/ablate_lab.py.inc).

    python3 problems/muse_decode_layer/dev/s11/bal_build.py
writes prof/lab/decode_bal.py (payload) and problems/muse_decode_layer/lab/lab_decode_bal.py (S11 + tools/h200_lab.py's
wrapper; send it with tools/okc_run.py send: a decode check run).
"""
from pathlib import Path
import hashlib
import json
import sys

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
D = R / 'problems/muse_decode_layer'
sys.path.insert(0, str(R / 'tools'))
import h200_lab  # noqa: E402
from okc_run import rejected_words  # noqa: E402

HOST = D / 'cand_r3_best7_s11.py'


def main():
    body = (D / 'dev/s10/ablate_lab.py.inc').read_text().replace('__NT1_CUDA__', "''", 1).replace('__NT1_CPP__', "''", 1)
    body += '\n\n' + (H / 'bal_lab.py.inc').read_text()
    payload = R / 'prof/lab/decode_bal.py'
    payload.write_text(body)
    host = HOST.read_text()
    sha = hashlib.sha256(host.encode()).hexdigest()
    assert sha == json.loads((H / 'manifest.json').read_text())['sha256']
    embedded = [h200_lab.embed(payload)]
    src = host.rstrip() + '\n\n' + h200_lab.WRAPPER.format(
        names=', '.join(f'{n} {s}' for n, s, _ in embedded), base='S11 ' + sha[:8], payloads=embedded,
        budget=300 - h200_lab.MARGIN_S, limit=h200_lab.LIMIT)
    assert not rejected_words(src), rejected_words(src)
    compile(src, 'lab', 'exec')
    out = D / 'lab/lab_decode_bal.py'
    out.write_text(src)
    print(payload.relative_to(R), embedded[0][1], out.relative_to(R), hashlib.sha256(src.encode()).hexdigest())


if __name__ == '__main__':
    main()
