from pathlib import Path
import ast,hashlib
from build_epilogue import source
from build_reuse_epilogue import source as once_source
from build_epilogue_pair import source as pair_source
from build_reuse_pair import source as once_pair_source
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
def build():
 src=(ROOT/'prof/lab/moe_nt1_reuse.py').read_text()
 for mode in range(4):src+='\n'+source(mode)[0]
 src+='\n'+pair_source()[0]+'\n'+once_source()[0]+'\n'+once_pair_source()[0]
 ps=(DEV/'reuse_phase_lab.py.inc').read_text();a=ps.index('def phase_summary(');b=ps.index('\ndef lab(',a);src+='\n'+ps[a:b]
 src+='\n'+(DEV/'epilogue_lab.py.inc').read_text()
 p=ROOT/'prof/lab/moe_nt1_epilogue.py';p.write_text(src);ast.parse(src);print(p,hashlib.sha256(src.encode()).hexdigest())
if __name__=='__main__':build()
