"""Freeze one batched full-attention payload, including independent native tensor-sum probes."""
import hashlib
import argparse
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
prefix=(P/'phases_lab.py').read_text().split('\nSOURCES=',1)[0]
names=['ref','mix25','mix50','gap','mixgap','tc25']
ap=argparse.ArgumentParser();ap.add_argument('--ping',action='store_true');a=ap.parse_args()
if a.ping:names+=['ping','ping25']
src={'A':(P/'vq_runtime.py').read_text(),**{n:(H/(n+'.py')).read_text() for n in names}}
s=prefix+'\n'+(H/'checks.py').read_text()+'\nSOURCES='+repr(src)+'\n'+(H/'lab.py.inc').read_text()
compile(s,'softmax_mix_lab.py','exec');(H/'softmax_mix_lab.py').write_text(s)
print('Softmax batched payload',len(s.encode()),hashlib.sha256(s.encode()).hexdigest())
