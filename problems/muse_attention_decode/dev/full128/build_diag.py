"""Isolate native tile math from stream-K/pipeline failures; synthetic component correctness only."""
import ast,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
s=(ROOT/'prof/lab/full128_timeline.py').read_text().split('def lab(deadline,weights,state=None):')[0]
s+='\nD128_TILE='+repr((HERE/'tile.cu.inc').read_text())+'\n'+(HERE/'diag.py.inc').read_text()
compile(s,'full128_diagnostic.py','exec');p=ROOT/'prof/lab/full128_diagnostic.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest(),len(s))
