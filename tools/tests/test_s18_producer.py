"""Execute the real producer loop with tiny CPU stand-ins, including an optional prepare failure."""
import base64
import hashlib
import json
import lzma
from pathlib import Path
import statistics
import sys
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
from promotion import evaluate


class S18ProducerRegression(unittest.TestCase):
    def test_optional_prepare_failure_emits_primary_pairs(self):
        h = ROOT/'problems/muse_decode_layer/dev/s18'
        template = (h/'lab.py.inc').read_text().replace('    times={};shapes={}',
                     (h/'nearby_lab.py.inc').read_text()+'\n    times={};shapes={}')
        # The shipping builder adds these fields before its compact sample serializer.
        template = template.replace('task_checks=task_checks,agreement_diagnostics=agreements,',
                     'task_checks=task_checks,nearby_task_checks=nearby_task_checks,nearby_expected_checks=nearby_expected_checks,agreement_diagnostics=agreements,')
        checker = '''
import types
def generate_weights(*a,**kw):return {'qkv_gate':types.SimpleNamespace(device='cuda')}
def generate_input(**kw):return {}
def ref_front(*a,**kw):return None
def ref_back(*a,**kw):return None
def collect(d,o):return o
def check_implementation(*a):return True,''
'''
        seeds = [100, 200]
        labels = {(1,1): 'decode 1', (16,1): 'decode 16', (32,1): 'decode 32',
                  (1,8): 'verify 1', (16,8): 'verify 16', (32,8): 'verify 32'}
        base = dict(_S16_LAB_BASE_STATE={}, _candidate_prepare=lambda w: {})
        exec('def kernel_fn(d,s):return object()', base)
        inc = dict(base)
        for name in ('_vq8_module','nf_module','cd_module','smallm_module','_glue_cl_ext','partial_port_ext'):
            inc[name] = lambda *a: None
        inc['_VQ8_ROWS'] = {}; inc['_PP_ROWS'] = ()
        sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
        arms = {n: dict(sha256=sha(n), source=n) for n in ('INC','bad')}
        cuda = types.SimpleNamespace(get_device_capability=lambda d: (9,0), synchronize=lambda: None, empty_cache=lambda: None)
        ns = dict(torch=types.SimpleNamespace(cuda=cuda, bfloat16='bf16'), time=time, statistics=statistics,
                  kernel_fn=base['kernel_fn'], S16_CHECKER=checker, S16_BUNDLE=dict(host='host', arms=arms),
                  S17_WEIGHT_SEED=1, S17_ROUTER_SEED=2, S17_NEARBY_CASES=[dict(label='nearby')],
                  S17_NEARBY_ARMS=('A','INC'), AB_SEEDS=seeds, AB_LOADS=labels,
                  AB_CASES=[(b,q,w) for b,q in labels for w in (2048,0)],
                  _ap_generate_input=lambda **kw: {}, ab_label=lambda c: ('d' if c[1]==1 else 'v')+str(c[0])+('S' if c[2] else 'F'),
                  ab_copy=lambda *a: None, ab_gate=lambda *a: '',
                  ab_time=lambda graphs, reps: {n: [1.02 if n=='INC' else 1.] for n in graphs})
        class Graph:
            def replay(self):
                pass
        ns['ab_graph'] = lambda call: (Graph(), call())
        exec(compile(template, '<real-s18-producer>', 'exec'), ns)
        ns['s16_source'] = lambda spec: spec['source']
        def namespace(name, source, base):
            if name == 'bad':
                raise RuntimeError('optional compile failed')
            return inc, 0
        ns['s16_namespace'] = namespace
        lines = ns['lab'](time.monotonic()+30, None, None)
        raw = json.loads(lzma.decompress(base64.b85decode(lines[1].split(' ',1)[1])))
        manifest = dict(sha256=sha('host'), base_sha256=sha('INC'), arms=arms,
                        checker_sha256=sha(checker), confirmation_seeds=seeds, selection_seeds=[],
                        expected_TASK_checks=96, nearby_expected_checks=4)
        out = evaluate(dict(id='producer', result=dict(passed=True)), raw, manifest)
        self.assertFalse(raw['complete'])
        self.assertFalse(raw['arm_status']['bad']['complete'])
        self.assertAlmostEqual(out['score']['R'], 1.02)
        self.assertTrue(out['freeze_gate'])
        self.assertFalse(out['pieces']['bad']['measured'])


if __name__ == '__main__':
    unittest.main()
