"""Whole-load S22 production cohort for the split's candidates (P4 ec754e5 64x16 symmetric protocol)."""
import ast
import base64
import difflib
import json
import lzma
import sys
from pathlib import Path
import types
import prod as _prod
_split = _prod._split
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'builds'))
import candidates as _night
build = types.SimpleNamespace(H=_prod.H, R=_prod.R, P=_prod.R / 'problems/muse_decode_layer', sha=_split.sha, once=_split.once,
                              load=_night.load, BASE_SHA=_split.BASE_SHA)

H, R, P, sha = build.H, build.R, build.P, build.sha
sys.path.insert(0, str(R / 'tools'))
import h200_lab

PROTOCOL = dict(name='palindromic-cohort-v1', rounds=16, seed_count=64,
                aggregation='arithmetic-mean-all-rounds', selective_retiming=False)
SEEDS = [751527241 + 104729 * i for i in range(64)]


def main(enabled=None):
    candidates = json.loads((H / 'candidates.json').read_text())
    source = (H / 'out/base.py').read_text()
    assert sha(source) == candidates['base_sha256'] == build.BASE_SHA
    specs = json.loads((H / 'out/arms.json').read_text())
    if enabled is None:
        enabled = list(specs)
    assert enabled and set(enabled) <= specs.keys()
    # INC is an independent unchanged S22 graph. A:A is still A2/A; the extra
    # unchanged graph also exercises the original promotion archive protocol.
    arms = dict(INC=dict(sha256=sha(source), edits=[], rows=[], kind='S22 control'))
    arms.update({name: specs[name] for name in enabled})
    reference = '\n'.join(line for line in (P / 'reference.py').read_text().splitlines()
                          if not line.startswith(('from checks import', 'from nvfp4 import')))
    checker = '\n'.join(line for text in ((P / 'checks.py').read_text(),
                                         (P / 'nvfp4.py').read_text(), reference)
                        for line in text.splitlines() if not line.startswith('from __future__ import'))
    selected = set()
    for directory in ('s18', 's19', 's20', 's21', 's22'):
        file = P / 'dev' / directory / ('manifest.json' if directory == 's22' else 'seed_manifest.json')
        data = json.loads(file.read_text())
        selected.update(data.get('selection_seeds', []))
        selected.update(data.get('confirmation_seeds', []))
    selected.update(json.loads((H.parent / 'manifest.json').read_text())['seeds'])
    selected.update(json.loads((H / 'manifest.json').read_text())['seeds'])
    selected.update(json.loads((H.parent / 'builds/manifest.json').read_text())['confirmation_seeds'])
    assert not selected.intersection(SEEDS)
    hook = next(ast.literal_eval(n.value) for n in ast.parse((P / 'dev/s19/build_lab.py').read_text()).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'HOOK')
    cases = build.load(P / 'dev/nearby/cases.py', 'p3b_night_nearby').cases()
    # Every candidate gets all nearby TASK checks too. Most nearby cases are
    # unchanged; M128/256 cases actually execute the new expert branches.
    nearby_arms = ['A', *arms]
    bundle = dict(host=source, arms=arms)
    encoded_bundle = base64.b85encode(lzma.compress(json.dumps(bundle, separators=(',', ':')).encode(), preset=9)).decode()
    body = 'import torch,types,time,statistics,math\n' + (P / 'dev/s19/graph_helpers.py.inc').read_text()
    body += '\n' + (P / 'dev/s21/timing.py.inc').read_text()
    body += '\nAB_SEEDS=' + repr(tuple(SEEDS)) + '\nS17_WEIGHT_SEED=731527237\nS17_ROUTER_SEED=731527239\n'
    body += '\nimport base64,lzma,json\nS16_BUNDLE=json.loads(lzma.decompress(base64.b85decode(' + repr(encoded_bundle) + ')))\n'
    body += 'S16_CHECKER=' + repr(checker) + '\nS17_NEARBY_CASES=' + repr(cases)
    body += '\nS17_NEARBY_ARMS=' + repr(tuple(nearby_arms)) + '\nS19_HOST_HOOK=' + repr(hook) + '\n'
    core = (P / 'dev/s21/lab.py.inc').read_text()
    core = build.once(core, "    result['original_pair_repeat']=s21_repeat_original_case(checker,base,deadline)\n", '')
    core = core.replace("'S21_XZ85 '+blob", "'P3B_SPLIT_XZ85 '+blob")
    core = core.replace("assert len(blob)<11700,len(blob)", "assert len(blob)<11700,('whole-graph evidence exceeds transport',len(blob))")
    core = core.replace('META S21 on exact S19;64 disjoint held-out seeds;all-round symmetric cohorts;public TASK all arms;restricted leave-outs take A2 off their rows',
                        'META P3b split candidates vs exact S22;64 fresh seeds x16 symmetric whole-kernel graphs;every arm TASK;restricted unchanged rows use A2')
    # The original P4 sampler bounds diagnostics to the worst four complete
    # seed/shape traces. Every round still contributes to every score sample.
    body += (P / 'dev/s21/evidence.py').read_text() + '\n' + core
    encoded_body = base64.b85encode(lzma.compress(body.encode(), preset=9)).decode()
    payload = 'LAB_RESOURCES={"compile_s":320,"run_s":520,"output_bytes":11700,"context_poison":"none"}\nimport base64,lzma,linecache,hashlib\n'
    payload += '_body=lzma.decompress(base64.b85decode(' + repr(encoded_body) + ')).decode()\n'
    payload += 'assert hashlib.sha256(_body.encode()).hexdigest()==' + repr(sha(body)) + '\n'
    payload += '_f="<p3b-production-' + sha(body)[:12] + '>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=globals()["lab"]\ndef lab(deadline,weights,state):return _run(deadline,weights,state)\n'
    tag = '_'.join(enabled)
    path = H / ('out/p3b_split_prod.py' if tag == 'PF' else f'out/p3b_split_prod_{tag}.py')
    path.write_text(payload)
    wrapper, _ = h200_lab.build([(path.stem, sha(payload)[:16], payload)], 'muse_moe_w4a16',
                               phase='import', import_budget=850.0, lab_only=True)
    (H / f'out/prod_wrapper_{tag}.py').write_text(wrapper)
    assert len(wrapper.encode()) < 1000000
    expected = 64 * (12 * 3 + sum(len(a['rows']) for n, a in arms.items() if n != 'INC'))
    manifest = dict(sha256=sha(source), base_sha256=sha(source), baseline_arm='INC',
        source_role='A/A2/INC are exact S22; optional arms are production candidate comparisons',
        host='muse_moe_w4a16', import_budget=850, payload=str(path.relative_to(R)), payload_sha256=sha(payload),
        wrapper_sha256=sha(wrapper), checker_sha256=sha(checker),
        measurement_protocol=PROTOCOL, confirmation_seeds=SEEDS, selection_seeds=sorted(selected),
        weight_seed=731527237, router_seed=731527239,
        arms={n: {k: v for k, v in a.items() if k != 'edits'} for n, a in arms.items()},
        expected_TASK_checks=expected, nearby_arms=nearby_arms,
        nearby_expected_checks=len(cases) * 2 * len(nearby_arms),
        interpretation='Optional-arm time/S22; below one favors the arm. Selection needs fresh disjoint confirmation; no official scored TASK is inferred from the lab-only host.',
        protocol_source_commit='ec754e50641f0bffa77e4d17422a39e7815ae480')
    (H / f'prod_manifest_{tag}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('production batch', list(arms), 'TASK', expected, 'nearby', manifest['nearby_expected_checks'],
          'payload', sha(payload)[:16], 'wrapper', sha(wrapper)[:16], 'bytes', len(wrapper.encode()))
    return manifest


if __name__ == '__main__':
    main(sys.argv[1:] or None)
