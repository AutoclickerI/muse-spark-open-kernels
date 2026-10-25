"""Prebuild one unsent paired lab for the four screened, diagnosis-dependent rebuilds."""
import ast
import hashlib
from pathlib import Path
from build_pipe_ready import KINDS, source

D = Path(__file__).resolve().parent
R = D.parents[3]


def build():
    text = (R / 'prof/lab/moe_nt1_reuse.py').read_text()
    for kind in KINDS:
        text += '\n' + source(kind)[0]
    text += '\n' + (D / 'pipe_rebuild_lab.py.inc').read_text()
    ast.parse(text)
    path = R / 'prof/lab/moe_nt1_pipe_rebuild.py'
    path.write_text(text)
    print(path, hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    build()
