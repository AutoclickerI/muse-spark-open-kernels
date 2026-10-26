"""Exact S15 shape pieces and a composition selected from this receipt only."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[3]
BASE = '602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
TARGET = 'problems/muse_decode_layer/cand_r3_best7_s15.py'


def apply(base, rows, runtime=None):
    assert hashlib.sha256(base.encode()).hexdigest() == BASE
    assert rows and set(rows) <= {'d1S', 'd1F', 'v1S', 'v1F'}
    assert set(rows.values()) <= ({'V', 'N', 'C'} if runtime is None else {'N','T'})
    mapping = {(1, 1 if s[0] == 'd' else 8, 2048 if s[-1] == 'S' else 0): a
               for s, a in rows.items()}
    suffix = '''
_rt_shape_run = _best7_run
def _best7_run(data,state,plan,prefetch=None):
    global RT_ARM
    c=data['config'];RT_ARM=RT_POINT_ROWS.get((c['batch'],c['q_len'],c['window']),'A')
    return _rt_shape_run(data,state,plan,prefetch)
'''
    code = base + '\n' + ((H / 'runtime_generated.py.inc').read_text() if runtime is None else runtime)
    code += '\nRT_POINT_ROWS=' + repr(mapping) + '\n' + suffix
    ast.parse(code)
    return code


def write_piece(base, rows, tag, out, runtime=None):
    code = apply(base, rows, runtime)
    patch = ''.join(difflib.unified_diff(base.splitlines(True), code.splitlines(True),
                                       fromfile='a/' + TARGET, tofile='b/' + TARGET))
    p = out / (tag + '.py'); p.write_text(code)
    q = out / (tag + '.patch'); q.write_text(patch)
    return dict(rows=rows, candidate=str(p.relative_to(R)),
                candidate_sha256=hashlib.sha256(code.encode()).hexdigest(),
                patch=str(q.relative_to(R)), patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
                applies_onto=BASE)


def main():
    x = json.loads((H / 'h200_result.json').read_text())
    base = (R / TARGET).read_text()
    out = H / 'out/point_handoffs'; out.mkdir(parents=True, exist_ok=True)
    points = []
    for p in x['nominations']:
        points.append(dict(p, **write_piece(base, {p['shape']:p['arm']},
                                            p['shape'] + '_' + p['arm'], out)))
    chosen = {s:min((p for p in points if p['shape'] == s), key=lambda p:p['mean'])['arm']
              for s in sorted({p['shape'] for p in points})}
    composition = write_piece(base, chosen, 'selected', out) if chosen else None
    result = dict(receipt=x['receipt'], nominations=points, selected=composition,
                  decision='Every correct sub-1 point is handed to P4; composition requires held-out six-load R LCB>1')
    (H / 'point_nominations.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(points=len(points), selected=composition), indent=2))


if __name__ == '__main__':
    main()
