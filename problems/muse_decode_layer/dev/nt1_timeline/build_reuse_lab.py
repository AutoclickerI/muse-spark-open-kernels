"""Freeze synthetic fixtures, exact S11 comparator, decode-once cell, and the 16-seed paired driver."""
from pathlib import Path
import ast,hashlib,re
from build_reuse import cell_source
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]

def build():
    src=(ROOT/'prof/lab/moe_nt1_timeline.py').read_text()
    # Retain fixture and error helpers; its old lab is overwritten below, and no old module is built.
    s11=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text()
    assert hashlib.sha256(s11.encode()).hexdigest().startswith('acda6c62')
    a=s11.index('SMALLM_VAR_CUDA = ');b=s11.index('def _s4_prepare_banks(',a)
    src+='\n'+s11[a:b]+'\n'+cell_source()+'\n'+(DEV/'reuse_lab.py.inc').read_text()
    out=ROOT/'prof/lab/moe_nt1_reuse.py';out.write_text(src);ast.parse(src)
    print(out.relative_to(ROOT),hashlib.sha256(src.encode()).hexdigest())
if __name__=='__main__':build()
