"""Standalone embedded-source H200 diagnostic with full causal leaf errors."""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
tree=ast.parse((HERE/'lab_producer_hybrid.py').read_text())
source=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
    and getattr(n.targets[0],'id',None)=='PG_SOURCE'))
body='PG_SOURCE = '+repr(source)+'\n'+(HERE/'diag_body.py.inc').read_text()
path=HERE/'lab_diag.py';path.write_text(body)
print(path,hashlib.sha256(body.encode()).hexdigest())
