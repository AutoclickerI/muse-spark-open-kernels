"""Build the S11-vs-S10 leave-one-out whole-layer lab (dev/s11/s11ab_lab.py.inc on dev/s10/ablate_lab.py.inc and
dev/s11/bal_lab.py.inc; S10's N_full cell block embedded verbatim from cand_r3_best7_s10.py).

    python3 problems/muse_decode_layer/dev/s11/s11ab_build.py
writes prof/lab/decode_s11ab.py (payload) and problems/muse_decode_layer/lab/lab_decode_s11ab.py (S11 + tools/h200_lab.py's
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
S10 = D / 'cand_r3_best7_s10.py'
S10_SHA = '6688fad999783d53a4660d85d40ee82862b87a209e03d289e44c95e41a3ffbc1'
START = '# MoE lane H200 c01b1916e2c74af8'
END = "def nf_call(data,state):\n    for f in nf_plan(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'])['steps']:out=f()\n    return out\n"


def main():
    s10 = S10.read_text()
    assert hashlib.sha256(s10.encode()).hexdigest() == S10_SHA
    i = s10.index(START);j = s10.index(END) + len(END)
    block = s10[i:j]
    body = (D / 'dev/s10/ablate_lab.py.inc').read_text().replace('__NT1_CUDA__', "''", 1).replace('__NT1_CPP__', "''", 1)
    body += '\n\n' + (H / 'bal_lab.py.inc').read_text()
    body += '\n\n' + (H / 's11ab_lab.py.inc').read_text().replace('__S10_NF_BLOCK__', repr(block), 1)
    payload = R / 'prof/lab/decode_s11ab.py'
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
    out = D / 'lab/lab_decode_s11ab.py'
    out.write_text(src)
    print(payload.relative_to(R), embedded[0][1], out.relative_to(R), hashlib.sha256(src.encode()).hexdigest())


if __name__ == '__main__':
    main()
