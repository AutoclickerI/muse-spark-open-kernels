"""Screen a host-state lab payload locally: load a decode-layer candidate as the host, run its prepare() on
generate_weights, then run the payload in a copy of the host's globals (as tools/h200_lab.py does).

    CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/coded_dense/local_layer.py \
        problems/muse_decode_layer/cand_r3_best7_s10.py prof/lab/coded_layer.py [seconds]
"""
import importlib.util
import linecache
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'problems/muse_decode_layer'))
import reference as R  # noqa: E402


def main():
    host, payload = Path(sys.argv[1]), Path(sys.argv[2])
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
    spec = importlib.util.spec_from_file_location('host', host)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['host'] = mod
    spec.loader.exec_module(mod)
    if torch.cuda.get_device_capability() != (9, 0) and hasattr(mod, '_ap_fa3_scheduler_metadata'):
        mod._ap_fa3_scheduler_metadata = lambda *a, **k: None  # FA3's scheduler is Hopper-only (as local_eval)
    weights = R.generate_weights(1234, device='cuda')
    t0 = time.monotonic()
    state = mod.prepare(weights)
    print(f'HOST prepare {time.monotonic() - t0:.1f}s', flush=True)
    flush = torch.empty(1 << 30, dtype=torch.uint8, device='cuda')
    ns = dict(vars(mod))
    ns['lab_cold_l2'] = lambda: flush.view(torch.int64).sum()
    src = payload.read_text()
    fname = f'<lab {payload.name}>'
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    exec(compile(src, fname, 'exec'), ns)
    for line in ns['lab'](time.monotonic() + budget, weights, state):
        print(line, flush=True)


if __name__ == '__main__':
    main()
