"""Judge production whole-load transfer, retaining P4's source/protocol gates."""
import argparse
import json
import math
import sys
from pathlib import Path
import types
import prod as _prod
_split = _prod._split
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'builds'))
import candidates as _night
build = types.SimpleNamespace(H=_prod.H, R=_prod.R, P=_prod.R / 'problems/muse_decode_layer', sha=_split.sha, load=_night.load)

H, R, P = build.H, build.R, build.P
sys.path.insert(0, str(R / 'tools'))
from lab_packet import paired_evidence
from promotion import evaluate, interval
codec = build.load(P / 'dev/s21/evidence.py', 'p3b_build_codec')


def analyze(path, tag, local=False):
    manifest = json.loads((H / f'prod_manifest_{tag}.json').read_text())
    if local:
        envelope = paired_evidence(path.read_text(), 'P3B_SPLIT')
        receipt = dict(id='LOCAL_ONLY', result=dict(gpu='sm120', passed=False))
    else:
        receipt = json.loads(path.read_text())
        assert receipt['result']['gpu'] == 'H200'
        assert build.sha(path.with_suffix('.py').read_bytes()) == manifest['wrapper_sha256']
        envelope = paired_evidence(receipt['result']['log_tail'], 'P3B_SPLIT')
    assert envelope is not None, 'No complete archive; do not infer a result'
    raw = codec.s21_decode_samples(codec.s21_unpack_evidence(envelope))
    full = evaluate(dict(id=receipt['id'], result=dict(gpu=receipt['result']['gpu'], passed=False)), raw, manifest)
    results = {}
    samples = {label: value['samples'] for label, value in raw['loads'].items()}
    aa = [math.exp(sum(math.log(rows[i]['A2']) for rows in samples.values()) / 6)
          for i in range(len(raw['seed_ids']))]
    for name, spec in manifest['arms'].items():
        if name == 'INC':
            continue
        admitted = full['pieces'][name]['measured'] and not local
        result = dict(mechanism=spec['mechanism'], measured=admitted,
                      TASK=full['arm_status'][name]['task_checks'],
                      nearby=full['arm_status'][name]['nearby_task_checks'],
                      interpretation='S22/candidate whole-load R; above one favors the candidate',
                      fresh_confirmation_required=True, G0_winner=False)
        if admitted:
            by_load = {label: [1 / row[name] for row in rows] for label, rows in samples.items()}
            ratios = [math.exp(sum(math.log(values[i]) for values in by_load.values()) / 6)
                      for i in range(len(raw['seed_ids']))]
            result.update(interval(ratios, aa), loads={label: interval(values) for label, values in by_load.items()})
            result['candidate_point_passes_lab_gate'] = (
                result['CI95'][0] > 1 and all(
                    not full['arm_status'][arm]['failures']
                    and full['arm_status'][arm]['nearby_task_checks']
                    == full['arm_status'][arm]['nearby_expected_checks']
                    for arm in ('A', 'A2', name)))
        results[name] = result
    output = dict(job=receipt['id'], native=not local, base_sha256=manifest['base_sha256'],
                  protocol=raw['measurement_protocol'], complete=raw['complete'], failures=raw['failures'],
                  TASK=raw['task_checks'], nearby=raw['nearby_task_checks'],
                  AA=interval(aa), arms=results,
                  decision='Rank whole-load R. Selected pieces/composition need new disjoint confirmation and official TASK before scoring; no stage-time transfer substituted.')
    target = H / ('out/prod_local_report' if local else 'prod_result') / tag
    target.mkdir(parents=True, exist_ok=True)
    (target / 'raw.json').write_text(json.dumps(raw, indent=2) + '\n')
    (target / 'promotion_protocol.json').write_text(json.dumps(full, indent=2) + '\n')
    (target / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'arms'}, indent=2))
    print(json.dumps(results, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('tag')
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    analyze(args.receipt, args.tag, args.local)
