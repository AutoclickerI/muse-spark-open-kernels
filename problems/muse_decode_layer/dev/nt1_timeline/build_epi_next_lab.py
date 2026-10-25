"""Batch the four independent second-round cuts with exact S14 and queued PAIR controls."""
import ast,hashlib
from pathlib import Path
from build_epilogue_pair import source as pair_source
from build_epi_next import source,KINDS
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
def build():
 s=(ROOT/'prof/lab/moe_nt1_reuse.py').read_text()+'\n'+pair_source()[0]
 for k in KINDS:s+='\n'+source(k)[0]
 old=(DEV/'epilogue_lab.py.inc').read_text();a=old.index('def norm_atomic_model');b=old.index('\ndef epilogue_trace',a)
 s+='\n'+old[a:b]+'\n'+(DEV/'epi_next_lab.py.inc').read_text()
 ast.parse(s);p=ROOT/'prof/lab/moe_nt1_epi_next.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
