"""Freeze the structural WGMMA rebuild, exact S11 comparator, and 16-seed graph/consumer-node measurement."""
import ast,hashlib
from pathlib import Path
from build_wg import cell_source
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
def build():
 src=(ROOT/'prof/lab/moe_nt1_reuse.py').read_text()+'\n'+cell_source()+'\n'+(DEV/'wg_lab.py.inc').read_text()
 out=ROOT/'prof/lab/moe_nt1_wg.py';out.write_text(src);ast.parse(src)
 print(out.relative_to(ROOT),hashlib.sha256(src.encode()).hexdigest())
if __name__=='__main__':build()
