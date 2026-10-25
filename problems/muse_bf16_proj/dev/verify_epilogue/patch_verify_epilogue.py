"""Patch S11 final with explicitly selected verify O/unpooler epilogue rows.

No rows are promoted by default. --force is a local admission candidate only.
"""
import argparse
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = {}
FORCED_ROWS = {(t, name): ((64, 128, 64, 4, 4, 1), 1024, True)
               for t in (128, 256) for name in ('o', 'unpooler')}


def run_block(src):
    old1 = "        a=_best7_project(attn,'o',state,plan)\n        h,stream1=_best7_boundary(a,data['stream'],state,plan,'b1',sync[0])"
    new1 = "        h,stream1=ve_project_boundary(attn,data['stream'],state,plan,'b1',sync[0])"
    old2 = "    f=_best7_project(moe,name,state,plan)\n    hidden,stream2=_best7_boundary(f,stream1,state,plan,'b2',sync[2])"
    new2 = "    if t in (128,256):\n        hidden,stream2=ve_project_boundary(moe,stream1,state,plan,'b2',sync[2])\n    else:\n" + '\n'.join('    ' + line for line in old2.splitlines())
    assert src.count(old1) == src.count(old2) == 1
    return src.replace(old1, new1).replace(old2, new2)


def patch(src, rows=None):
    rows = ROWS if rows is None else rows
    src = run_block(src)
    anchor = '    return _best7_run(data,state,plan,state[\'best7_prefetch_ops\'].get(key))'
    assert src.count(anchor) == 1
    src = src.replace(anchor, "    state=dict(state,ve_rows=VE_ROWS)\n" + anchor)
    return src + '\n\n' + (HERE / 'kernels.py').read_text() + '\nVE_ROWS = ' + repr(rows) + '\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    out = patch(a.source.read_text(), FORCED_ROWS if a.force else ROWS)
    a.out.write_text(out)
    print(a.out, hashlib.sha256(out.encode()).hexdigest())
