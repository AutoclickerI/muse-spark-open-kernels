"""Reproduce the pooled freeze gate; retain each individual receipt's failed gates.

Pool all three matched decode32 comparisons, without selecting the two wins. Require
identical relevant source paths, pooled AND median gain beyond the worst A:A radius.
Use the final receipt for other retained gains and controls, except light controls,
which are pooled consistently across all three runs (including its failed V1 control).
"""
import ast
import base64
import hashlib
import json
import lzma
import math
import statistics
import zlib
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[3]
IDS = ('0441f9fc583c42be', '740c63afadc14e01', '2bbb626c5ac2428a')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def values(source):
    nodes = ast.parse(source).body
    functions = {n.name: n for n in nodes if isinstance(n, ast.FunctionDef)}
    assignments = {t.id: n.value for n in nodes if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name)}
    return functions, assignments


def unpack(source):
    call = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'b85decode')
    raw = base64.b85decode(ast.literal_eval(call.args[0]))
    return (lzma.decompress(raw) if raw.startswith(b'\xfd7zXZ') else zlib.decompress(raw)).decode()


def source_identity(job, report):
    paths = list((R / 'runs/muse_decode_layer').glob('*_' + job + '.py'))
    assert len(paths) == 1, paths
    _, assignments = values(paths[0].read_text())
    payloads = ast.literal_eval(assignments['_LAB_PAYLOADS'])
    assert len(payloads) == 1
    # okc's wrapper contains the bootstrap as the third tuple field.
    bootstrap = payloads[0][2]
    body = unpack(bootstrap)
    bundle = json.loads(unpack(body))
    source = bundle['host']
    assert digest(source) == report['source_sha256']
    spec = bundle['variants']['bal2']
    lines = source.splitlines(keepends=True)
    for start, end, replacement in reversed(spec['edits']):
        lines[start:end] = [replacement]
    off = ''.join(lines)
    assert digest(off) == spec['sha256']
    f, v = values(source)
    off_f, _ = values(off)
    dispatch = f['_gb_gemv'].body[0]
    assert isinstance(dispatch, ast.If)
    assert 32 in ast.literal_eval(dispatch.test.values[0].comparators[0])
    settings = ast.literal_eval(v['BEST7_SETTINGS'])
    signature = {
        'gi_kernel': ast.dump(f['_gi_kernel']),
        'gi_run': ast.dump(f['gi_run']),
        'M32_call': ast.dump(dispatch.body[0]),
        'baseline_gb_kernel': ast.dump(f['_gb_kernel']),
        'baseline_gb_call': ast.dump(off_f['_gb_gemv']),
        'M32_settings': repr([settings[(32, 1, w)] for w in (2048, 0)]),
        'GB_ROWS': ast.dump(v['GB_ROWS']),
        'SMALLM_FLAGS': ast.dump(v['SMALLM_FLAGS']),
        'SMALLM_VAR_CUDA': ast.dump(v['SMALLM_VAR_CUDA']),
        'prepare_weights': ast.dump(f['_prepare_weights']),
    }
    for name in ('CD_ROWS', '_DCP_ROWS'):
        if name in v:
            assert all(key[0] != 32 for key in ast.literal_eval(v[name])), name
    return dict(job=job, host_sha256=digest(source), off_source_sha256=digest(off),
                identity_sha256={key: digest(value) for key, value in signature.items()})


def geomean(ratios):
    return math.exp(sum(map(math.log, ratios)) / len(ratios))


def main():
    reports = [json.loads((H / (job[:8] + '_report.json')).read_text()) for job in IDS]
    assert all(r['complete'] for r in reports)
    identities = [source_identity(job, r) for job, r in zip(IDS, reports)]
    assert all(e['identity_sha256'] == identities[0]['identity_sha256'] for e in identities)
    ratios = [r['loads']['decode 32']['bal2'] for r in reports]
    radius = max(r['decisions']['bal2']['decode 32']['AA_radius'] for r in reports)
    pooled, median = geomean(ratios), statistics.median(ratios)
    latest = reports[-1]
    light_controls = {}
    for load in latest['controls']['light']:
        rs = [r['controls']['light'][load]['ratio'] for r in reports]
        aa = max(r['controls']['light'][load]['AA_radius'] for r in reports)
        light_controls[load] = dict(ratios=rs, pooled_ratio=geomean(rs), worst_AA_radius=aa,
                                    within_AA=abs(geomean(rs)-1) <= aa)
    result = dict(
        method='Equal-run geometric pooling of all three matched T32 whole-layer ratios against '
               'the worst observed absolute A:A envelope; same gate as S13. Single-run failures remain recorded.',
        source_sha256=latest['source_sha256'], jobs=list(IDS), ratios=ratios,
        pooled_ratio=pooled, median_ratio=median, worst_AA_radius=radius, margin=pooled-1-radius,
        bal2_wins=pooled > 1+radius and median > 1+radius,
        kernel_path_identity=identities, light_controls=light_controls,
        light_wins=all(d['wins'] for d in latest['decisions']['light'].values()),
        latest_bal2_controls_pass=all(d['within_AA'] for d in latest['controls']['bal2'].values()),
        latest_base_controls_pass=all(d['within_AA'] for d in latest['controls']['S13'].values()),
        base_decode32_pooled_ratio=geomean([r['loads']['decode 32']['S13'] for r in reports]),
        latest_verify_base_wins=all(latest['decisions']['S13'][load]['wins']
                                   for load in ('verify 16', 'verify 32')))
    result['base_decode32_wins'] = result['base_decode32_pooled_ratio'] > 1+radius
    result['freeze_gate'] = all(result[key] for key in (
        'bal2_wins', 'light_wins', 'latest_bal2_controls_pass', 'latest_base_controls_pass',
        'base_decode32_wins', 'latest_verify_base_wins')) and all(
            d['within_AA'] for d in light_controls.values())
    (H / 'pooled_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('kernel_path_identity', 'light_controls')}, indent=2))
    assert result['freeze_gate']


if __name__ == '__main__':
    main()
