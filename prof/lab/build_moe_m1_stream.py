"""Embed the current canonical bank, M1 GEMV, and real-router comparison in one payload."""
import hashlib
from pathlib import Path

from build_moe_small_debug import build as small_build

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / 'problems/muse_decode_layer/dev/small_m_gemv'
OUTPUT = ROOT / 'problems/muse_prefill_layer/lab/lab_moe_m1_stream.py'


def build():
    base = small_build().read_text()
    base = base[:base.index('\ndef lab(deadline, weights):')]
    cu = (DEV / 'gemv.cu').read_text()
    cpp = '''#include <pybind11/pybind11.h>
#include <stdint.h>
void run(int64_t x,int64_t ids,int64_t tw,int64_t w13,int64_t w2,int64_t gs13,int64_t gs2,
         int64_t pnw,int64_t inter,int64_t parts,int64_t y,int64_t sums,int64_t invnorm,
         int64_t counts,int64_t out,int64_t local_start,int64_t splits,bool pdl,int64_t grid,int64_t stream_handle);
'''
    kernel = '\nM1_STREAM_CUDA_SRC = ' + repr(cu) + '\nM1_STREAM_CPP_SRC = ' + repr(cpp) + '\n'
    kernel += (DEV / 'adapter.py.inc').read_text() + '\n'
    (DEV / 'dropin.py').write_text('import torch\nfrom torch.utils.cpp_extension import load_inline\n' + kernel)
    body = base + kernel
    body += (DEV / 'lab.py.inc').read_text()
    OUTPUT.write_text(body)
    print(OUTPUT.relative_to(ROOT), hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    return OUTPUT


if __name__ == '__main__':
    build()
