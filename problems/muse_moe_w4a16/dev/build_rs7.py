"""Carry checked V7 into the small-M reader without changing the canonical bank."""
from pathlib import Path
p=Path(__file__).resolve().parent.parent
native=(p/'cand_v7.py').read_text().replace('def prepare(weights):','def _prepare_native(weights):').replace('def kernel_fn(data, state):','def _native_call(data, state):')
for old,new in (('cand_rs.py','cand_rs7.py'),('cand_rs128.py','cand_rs128_7.py')):
 source=(p/old).read_text()
 start=source.index('\nGROUPED_SRC = ')
 (p/new).write_text(native+source[start:])
