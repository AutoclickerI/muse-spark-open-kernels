"""Force only the final import-time diagnostic guard; require every full-80K probe gate to pass."""
import contextlib
import argparse
import hashlib
import io
import json
import linecache
from pathlib import Path

H = Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--board',default='phases_board.py');ap.add_argument('--rows',type=int,default=4);a=ap.parse_args()
p = H / a.board
s = p.read_text()
old = '# Full-attention diagnostic boot: attention has no harness prepare hook.\nif torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:\n    prepare()\n'
assert s.count(old) == 1
forced = s.replace(old, old.replace('torch.cuda.get_device_capability()[0] == 9', 'True'))
f = p.with_name(p.stem+'_force.py')
f.write_text(forced)
linecache.cache[str(f)] = (len(forced), None, forced.splitlines(True), str(f))
ns = {}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(forced, str(f), 'exec'), ns)
log = buf.getvalue()
print(log, flush=True)
assert ns['_lab_ran'] and sum(r.startswith('R B') for r in log.splitlines()) == a.rows, log
assert not any(s in log for s in ('LAB_ERROR', 'ERROR B', 'SKIP B')), log
(H / 'board_boot.json' if a.board=='phases_board.py' else p.with_name(p.stem+'_boot.json')).write_text(json.dumps(dict(passed=True,
    sha256=hashlib.sha256(p.read_bytes()).hexdigest(), cases=a.rows, ctx=80000,
    forced_change='last capability guard only; payload, seeds and full context unchanged'), indent=2) + '\n')
