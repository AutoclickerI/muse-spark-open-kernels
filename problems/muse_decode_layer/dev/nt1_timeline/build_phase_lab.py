from pathlib import Path
import ast,hashlib
from build_reuse_phase import source
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
def build():
 s13=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s13_final.py').read_text()
 assert hashlib.sha256(s13.encode()).hexdigest().startswith('958f1258')
 src=(ROOT/'prof/lab/moe_nt1_reuse.py').read_text()+'\n'+source()[0]+'\n'+(DEV/'reuse_phase_lab.py.inc').read_text()
 p=ROOT/'prof/lab/moe_nt1_reuse_phase.py';p.write_text(src);ast.parse(src);print(p,hashlib.sha256(src.encode()).hexdigest())
if __name__=='__main__':build()
