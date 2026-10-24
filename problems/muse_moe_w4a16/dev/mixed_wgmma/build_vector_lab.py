"""Assemble the packed-fragment producer and current raw-E4M3 V9 seed controls."""
import hashlib
from pathlib import Path
from build_prefill_lab import subset

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def main():
    base=(ROOT/'prof/lab/moe_mixed_wgmma_planes.py').read_text()
    base=base[:base.index('\ndef lab(deadline,weights,state=None):')]
    control=subset((ROOT/'prof/lab/moe_v9_seed.py').read_text(),('rv_prepare','rv_kernel_fn'))
    vector=(HERE/'vector.py').read_text()
    vector=vector.replace('from engine import mw_repack_planes,mw_route,mw_prefix,mw_swiglu,mw_norm,mw_combine','')
    vector=vector.replace("VECTOR_CUDA=Path(__file__).with_name('vector.cu').read_text()",
                          'VECTOR_CUDA='+repr((HERE/'vector.cu').read_text()))
    out=ROOT/'prof/lab/moe_vector_wgmma_seed.py'
    out.write_text(base+'\n'+control+'\n'+vector+'\n'+(HERE/'vector_driver.py.inc').read_text())
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)


if __name__=='__main__':main()
