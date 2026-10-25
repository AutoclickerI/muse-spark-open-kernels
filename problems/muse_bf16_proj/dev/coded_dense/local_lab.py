"""Run an h200_lab payload locally on generate_weights (proj-host weight names) with a read flush.

    CUDA_VISIBLE_DEVICES=0 python3 problems/muse_bf16_proj/dev/coded_dense/local_lab.py prof/lab/coded_gemv.py [seconds]
"""
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'problems/muse_decode_layer'))
import reference as R  # noqa: E402

_flush = []


def lab_cold_l2():
    if not _flush:
        _flush.append(torch.empty(1 << 30, dtype=torch.uint8, device='cuda'))
    _flush[0].view(torch.int64).sum()


def main():
    path = Path(sys.argv[1])
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    src = path.read_text()
    ns = {'lab_cold_l2': lab_cold_l2, '__name__': 'lab_payload'}
    import linecache
    fname = f'<lab {path.name}>'
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    exec(compile(src, fname, 'exec'), ns)
    w = R.generate_weights(1234, device='cuda')
    weights = {k: w[k] for k in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')}
    for line in ns['lab'](time.monotonic() + budget, weights):
        print(line, flush=True)


if __name__ == '__main__':
    main()
