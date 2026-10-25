"""Change only the statistics epilogue identified by 9845398be06b4c37."""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'stats_kernel.py').read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ps_tile')
lines=source.splitlines(True)
start=min(n.lineno for n in node.decorator_list)-1
replacement=(HERE/'parallel_tile.py.inc').read_text()
new=''.join(lines[:start])+replacement+'\n\n'+''.join(lines[node.end_lineno:])
assert ast.dump(next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='_ps_gemv'))==ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ps_gemv'))
path=HERE/'stats_parallel.py';path.write_text(new)
print(path,hashlib.sha256(new.encode()).hexdigest())
