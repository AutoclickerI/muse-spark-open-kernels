"""Build the standalone grouped RS-WGMMA lab with independent FP32 and paired controls."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prof/lab'))
from build_moe_real_routing import namespace


def build(planes=False):
    source=(ROOT/'prof/lab/moe_v9_reuse_v2.py').read_text()
    fixtures=source[source.index('import math as fx_math'):source.index('\nREUSE_CHOICES=')]
    oracle=(HERE.parent/'v9_reuse/reuse.py').read_text()
    oracle=oracle[oracle.index('def _reuse_fp32_impl'):]
    driver=(HERE/'lab_driver.py.inc').read_text()
    if planes:driver=driver.replace('MW_LAB_PLANES=False','MW_LAB_PLANES=True')
    parts=[(HERE/'engine.py').read_text(),
           namespace((ROOT/'problems/muse_moe_w4a16/cand_v9.py').read_text(),'ctrl_'),
           namespace((ROOT/'problems/muse_moe_w4a16/baseline.py').read_text(),'marlin_'),
           fixtures,oracle,driver]
    out=ROOT/('prof/lab/moe_mixed_wgmma_planes.py' if planes else 'prof/lab/moe_mixed_wgmma.py')
    out.write_text('\n\n'.join(parts))
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--planes',action='store_true')
    build(ap.parse_args().planes)
