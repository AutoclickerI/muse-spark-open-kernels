"""Replay real evidence; no CUDA, credentials, network or judge quota."""
import base64
import copy
import importlib.util
import json
import lzma
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
S18 = ROOT / 'problems/muse_decode_layer/dev/s18'
sys.path.insert(0, str(ROOT / 'tools'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PromotionRegression(unittest.TestCase):
    def setUp(self):
        self.rep = module('s18_report_test', S18 / 'report.py')
        self.remote = module('s18_remote_test', S18 / 'remote_lab.py')
        self.path = next((ROOT / 'runs/muse_prefill_layer').glob('*5d0282bb*.json'))
        self.receipt = json.loads(self.path.read_text())
        self.manifest = json.loads((S18 / 'manifest.json').read_text())
        blob = next(l.split(' ', 1)[1] for l in self.receipt['result']['log_tail'].splitlines()
                    if l.startswith('S18_XZ85 '))
        self.raw = self.rep.decode(json.loads(lzma.decompress(base64.b85decode(blob))))

    def test_archived_host_paths_agree(self):
        # Exercise remote.report, including its official TASK receipt and wrapper hash checks.
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)/'project/problems/muse_decode_layer/dev/s18'
            home.mkdir(parents=True)
            for name in ('report.py', 'manifest.json', 'remote_manifest.json'):
                (home / name).write_bytes((S18 / name).read_bytes())
            with patch.object(self.remote, 'H', home), patch('builtins.print'):
                self.remote.report(self.path)
            remote = json.loads(next((home / 'batches').glob('*/report.json')).read_text())
        decode = self.rep.evaluate({'id': 'same-pairs-on-decode', 'result': {'passed': True, 'gpu': 'H200'}},
                                   copy.deepcopy(self.raw), self.manifest)
        self.assertTrue(decode['freeze_gate'])
        self.assertEqual(decode['score'], remote['score'])
        self.assertEqual(decode['freeze_gate'], remote['freeze_gate'])
        self.assertAlmostEqual(decode['score']['R'], 1.0043206, places=6)

    def test_archive_requires_native_device_evidence(self):
        out = self.rep.evaluate({'id': 'not-native', 'result': {'passed': True}},
                                copy.deepcopy(self.raw), self.manifest)
        self.assertIsNone(out['score']['R'])

    def test_later_integration_uses_shared_evaluator(self):
        later = module('s19_report_test', S18.parent/'s19/report.py')
        from promotion import evaluate
        self.assertIs(later.evaluate, evaluate)

    def test_missing_official_TASK_still_measures_R(self):
        out = self.rep.evaluate({'id': 'no-official', 'result': {'passed': False}}, self.scoped(), self.manifest)
        self.assertIsNotNone(out['score']['R'])
        self.assertFalse(out['freeze_gate'])

    def scoped(self):
        raw = copy.deepcopy(self.raw)
        identities = {'A': self.manifest['sha256'], 'A2': self.manifest['sha256'],
                      **{k: v['sha256'] for k, v in self.manifest['arms'].items()}}
        raw['arm_status'] = {name: dict(sha256=sha, native=True, complete=True,
                                      seed_ids=raw['seed_ids'], task_checks=192, expected_task_checks=192,
                                      nearby_task_checks=130 if name in ('A', 'INC') else 0,
                                      nearby_expected_checks=130 if name in ('A', 'INC') else 0,
                                      failures=[]) for name, sha in identities.items()}
        return raw

    def test_optional_arm_failure_preserves_primary_R(self):
        raw = self.scoped()
        raw['complete'] = False
        raw['failures'] = [{'scope': 'arm', 'arms': ['no_BLOCK'], 'reason': 'capture failed'}]
        raw['arm_status']['no_BLOCK'].update(complete=False, failures=['capture failed'])
        for load in raw['loads'].values():
            for row in load['samples']:
                row.pop('no_BLOCK')
        out = self.rep.evaluate({'id': 'optional', 'result': {'passed': True}}, raw, self.manifest)
        self.assertIsNotNone(out['score']['R'])
        self.assertTrue(out['freeze_gate'])
        self.assertFalse(out['pieces']['no_BLOCK']['measured'])

    def test_device_fault_and_bad_control_void_primary(self):
        for failure in ({'scope': 'device', 'arms': ['A', 'INC', 'A2'], 'reason': 'illegal address'},
                        {'scope': 'arm', 'arms': ['A2'], 'reason': 'bad control'}):
            with self.subTest(failure=failure):
                raw = self.scoped()
                raw['complete'] = False
                raw['failures'] = [failure]
                out = self.rep.evaluate({'id': 'fault', 'result': {'passed': True}}, raw, self.manifest)
                self.assertIsNone(out['score']['R'])
                self.assertFalse(out['freeze_gate'])

    def test_incomplete_identity_non_native_and_missing_nearby(self):
        for key, value in [('sha256', 'wrong'), ('native', False), ('nearby_task_checks', 129),
                           ('seed_ids', self.raw['seed_ids'][:-1])]:
            with self.subTest(key=key):
                raw = self.scoped()
                raw['arm_status']['A'][key] = value
                out = self.rep.evaluate({'id': 'invalid', 'result': {'passed': True}}, raw, self.manifest)
                self.assertFalse(out['freeze_gate'])
                if key != 'nearby_task_checks':
                    self.assertIsNone(out['score']['R'])


if __name__ == '__main__':
    unittest.main()
