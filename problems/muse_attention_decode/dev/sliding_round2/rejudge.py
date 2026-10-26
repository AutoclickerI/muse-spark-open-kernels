"""Rebuild the P4 point-winner handoff directly from immutable P5 receipts."""
import ast, base64, contextlib, hashlib, io, json, lzma
from pathlib import Path
import report

H = Path(__file__).resolve().parent
R = H.parents[3]
JOBS = {
    'ae6cd27877cc4ee2': 'S11, legacy replay correctness unmeasured',
    'b05713ecf6574ad9': 'attention stage only; full-layer confirmation required',
    '402819c28279466d': 'exact S14, decode1 sliding',
    '6c26b14fdd28458b': 'exact S14, fixed-pool DSM cohort',
    '6a0b3f0987e94ba5': 'exact S14, fresh-pool DSM repeat',
    '27a914f72bfb47c8': 'exact S14, sliding round2',
    '857f78fa889f4646': 'exact S14, resident sliding verify',
    'bb7239afc1b3487a': 'S11, repaired replay TASK/cache gates',
    '58f0b30ba8b8472f': 'exact S14, fresh-pool global confirmation',
}
MARKERS = ('R2_XZ85', 'VR_XZ85', 'EH_XZ85', 'DS_XZ85', 'CS_XZ85', 'CG_XZ85')

def embedded_sources(source):
    """Read wrappers and compressed payloads without executing judge code."""
    yield source
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == '_LAB_PAYLOADS' for t in node.targets):
            for _, _, payload in ast.literal_eval(node.value):
                yield from embedded_sources(payload)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'b85decode' and node.args
                and isinstance(node.args[0], ast.Constant)):
            payload = lzma.decompress(base64.b85decode(node.args[0].value)).decode()
            yield from embedded_sources(payload)

def seed_ids(source, count):
    found = []
    for text in embedded_sources(source):
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and any(getattr(t, 'id', '').endswith('SEEDS') for t in node.targets):
                # Only the receipt's literal arithmetic tuple expression is
                # evaluated; no imports, kernels, or host preparation run.
                expr = ast.get_source_segment(text, node.value)
                ids = eval(expr, {'__builtins__': {}, 'tuple': tuple, 'range': range})
                if len(ids) == count and ids not in found:
                    found.append(ids)
    assert len(found) == 1, (count, found)
    return list(found[0])

def recipe(row, marker, name):
    if name in row.get('arms', {}):
        return row['arms'][name]
    if marker == 'DS_XZ85':
        return ['reset DSM stage', int(name[1:])]
    if marker == 'EH_XZ85':
        return ['exact reset history ' + name, int(name[-1]), 16]
    if marker == 'CG_XZ85':
        return ['exact repaired global ' + name, row['case']]
    return [name]

def implementation(job, marker):
    if marker == 'DS_XZ85':
        return 'Exact cc_CUDA/e43cb513 in receipt; stage ABI, not round2 cooperative prep'
    if marker == 'EH_XZ85':
        return ('Exact reset OLD4/early/late shader in receipt; early SHA0e07854b, OLD4 SHA3815beab. '
                'Use history_build.py/build_lab.py, not round2 early kernelc7a41b46')
    if marker == 'CG_XZ85':
        return ('Exact S11 global lab in receipt: global cs_CUDA SHA35c257f8; OLD/PP are '
                'its last/partial controls. Keep original settings and older-base scope')
    if job == '58f0b30ba8b8472f':
        return ('Frozen payload KERNELS[global] SHA35c257f8, KERNELS[last] SHA3815beab; '
                'PP calls exact S14 _layer_partial_port. followups.py global_confirm reconstructs these bytes')
    return ('layer_patch.py supports this [family,QT,splits] on exact S14: last9118a17c, '
            'earlyc7a41b46, dsm0f250847, dsm_olde43cb513. P4 ports onto current base')

def main():
    result = dict(
        status='All measured per-shape point estimates below 1 handed to P4; no per-load significance gate',
        ratio='mean across seeds of paired T_arm/T_A; SE is across-seed standard error',
        promotion='P4 composes, then held-out per-seed six-load R=exp(mean(log(T_old/T_new))), '
                  'T=46*t_sliding+16*t_full; promote only when lower confidence bound >1',
        correctness='TASK.md checker only; historical extra checks are recorded, not new admission rules',
        caveat='These are nominations, not whole-score promotions. S11 and stage-only estimates retain their scope. '
               'Known failed experimental controls remain unmeasured.',
        runs=[], nominations=[], incremental_nominations=[], pending=[])
    for job, scope in JOBS.items():
        matches = list((R/'runs').glob('*/*'+job+'.json'))
        if not matches:
            result['pending'].append(job)
            continue
        path, = matches
        receipt = json.loads(path.read_text())
        source = path.with_suffix('.py').read_text()
        rows = []
        for line in receipt['result'].get('log_tail', '').splitlines():
            if line.split(' ', 1)[0] in MARKERS:
                marker, blob = line.split(' ', 1)
                payload = json.loads(lzma.decompress(base64.b85decode(blob)))
                rows.extend((marker, row) for row in (payload if isinstance(payload, list) else [payload]))
        assert rows, (job, 'no complete P5 archive')
        entry = dict(job=job, receipt=str(path.relative_to(R)), scope=scope,
                     source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                     seed_ids=seed_ids(source, rows[0][1]['seeds']), results=[])
        for marker, row in rows:
            with contextlib.redirect_stdout(io.StringIO()):
                summary = report.report(row, marker, job == '6c26b14fdd28458b')
            entry['results'].append(summary)
            for name, arm in summary['arms'].items():
                nomination = dict(job=job, case=summary['case'], arm=name, recipe=recipe(row, marker, name),
                                  ratio=arm['ratio'], se=arm['se'], seeds=summary['seeds'],
                                  seed_ids_ref='point_handoff.json#/runs/'+str(len(result['runs']))+'/seed_ids', scope=scope,
                                  A2_A=arm['aa'], A2_A_se=arm['aa_se'], A2_A_range=arm['aa_range'])
                nomination['implementation'] = implementation(job, marker)
                if arm['nominate_to_p4']:
                    result['nominations'].append(nomination)
                if arm.get('measured') and arm.get('vs_old', 1) < 1:
                    result['incremental_nominations'].append(dict(nomination, denominator='OLD4',
                        ratio=arm['vs_old'], se=arm['vs_old_se']))
        result['runs'].append(entry)
    (H/'point_handoff.json').write_text(json.dumps(result, indent=2)+'\n')
    print('P4 point nominations:', len(result['nominations']),
          'incremental:', len(result['incremental_nominations']), 'pending:', result['pending'])
    for run in result['runs']:
        wins = [x for x in result['nominations'] if x['job'] == run['job']]
        print(run['job'], ' '.join(f"{x['case']}:{x['arm']}={x['ratio']:.6f}+/-{x['se']:.6f}" for x in wins) or 'no measured point winner')

if __name__ == '__main__':
    main()
