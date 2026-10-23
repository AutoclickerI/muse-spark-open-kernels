"""GPU0 admission for the standalone prefill dense portfolio, including graphs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import torch
from sglang.srt.layers import muse_prefill_gemm as p

COUNTS = Counter()
RESULTS = []


def compare(got, expected):
    if isinstance(got, tuple):
        for a, b in zip(got, expected): compare(a, b)
        return
    error = float((got.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-15))
    assert error <= (1e-4 if got.dtype == torch.float32 else .01), error
    assert bool(torch.isfinite(got).all())


def weights():
    return {name: torch.randn(shape, device='cuda', dtype=torch.float32 if name == 'router' else torch.bfloat16) * .01
            for name, shape in dict(qkv_gate=(2304, 8192), o=(8192, 1024), pooler=(4096, 8192),
                                    unpooler=(8192, 4096), router=(256, 8192)).items()}


class TestPrefill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20261024)
        cls.op = p.PrefillGemmPlan(weights(), tokens=(16384, 3012, 753), budget_s=0)

    def test_all_persistent_schedules(self):
        op = self.op
        for m in (753, 3012, 16384):
            for role in (*op.weights, 'pool_router'):
                k = op.weights['pooler' if role == 'pool_router' else role].shape[1]
                x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
                expected = op._stock(x, role)
                for index in range(len(p.PG_TILES)):
                    for schedule in (0, 1):
                        config = ('persistent', index, schedule)
                        # The module's tuner rejects unsupported resource limits;
                        # arithmetic errors in an executable kernel fail this test.
                        try:
                            graph = p._PgGraph(lambda: op.run(x, role, config))
                        except torch.OutOfMemoryError:
                            raise
                        except Exception as exc:
                            if 'out of resource' in str(exc).lower() or 'shared memory' in str(exc).lower():
                                COUNTS['resource_rejections'] += 1
                                continue
                            raise
                        with graph:
                            compare(graph.out, op._stock(x, role))
                            for rep in range(3):
                                x.normal_()
                                if rep == 2: x[:, ::257] *= 16
                                graph.graph.replay()
                                compare(graph.out, op._stock(x, role))
                                COUNTS['persistent_fresh_graphs'] += 1
                        COUNTS[f'family_persistent_schedule{schedule}'] += 1
                        assert int(op.counts.count_nonzero()) == 0
                        COUNTS['counter_reset_checks'] += 1
                print('PG_TEST', m, role, dict(COUNTS), flush=True)

    def test_all_lt_heuristics(self):
        op = self.op
        for m in (753, 3012, 16384):
            for role in (*op.weights, 'pool_router'):
                k = op.weights['pooler' if role == 'pool_router' else role].shape[1]
                x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
                for nn in (0, 1):
                    codes = op._options(m, role, nn)
                    self.assertGreater(len(codes), 0)
                    for index in range(len(codes)):
                        config = ('lt', nn, index)
                        try:
                            graph = p._PgGraph(lambda: op.run(x, role, config))
                        except Exception as exc:
                            # Heuristics still need launch validation. A rejected
                            # algorithm cannot enter the selected production plan.
                            op._reject(m, role, config, exc)
                            COUNTS['lt_launch_rejections'] += 1
                            continue
                        with graph:
                            for rep in range(3):
                                x.normal_()
                                if rep == 2: x[:, ::257] *= 16
                                graph.graph.replay()
                                compare(graph.out, op._stock(x, role))
                                COUNTS['lt_fresh_graphs'] += 1
                        COUNTS[f'family_lt_nn{nn}'] += 1
                print('PG_LT_TEST', m, role, dict(COUNTS), flush=True)

    def test_budget_and_rejections(self):
        op = p.PrefillGemmPlan({'qkv_gate': self.op.weights['qkv_gate']}, tokens=(753,), budget_s=0)
        self.assertEqual(op.plans[(753, 'qkv_gate')], ('stock',))
        original = op.run
        def broken(x, role, config):
            if config[0] == 'attr': raise AttributeError('injected missing entrypoint')
            if config[0] == 'tuple': raise TypeError('injected tuple ABI')
            if config[0] == 'wrong': return torch.zeros_like(original(x, role, ('stock',)))
            return original(x, role, config)
        with patch.object(op, 'run', side_effect=broken), patch.object(op, 'variants',
                side_effect=lambda *args: iter((('stock',), ('attr',), ('tuple',), ('wrong',)))):
            op.prepare((753,), deadline=time.monotonic() + 20)
        self.assertEqual(len(op.rejections), 3)
        self.assertEqual(op.plans[(753, 'qkv_gate')], ('stock',))
        self.assertEqual(p._PgGraph.active, 0)
        COUNTS['injected_rejections'] += 3
        # A bad library query must not remove persistent paths or the incumbent.
        with patch.object(op, '_options', side_effect=AttributeError('layout unavailable')):
            variants = list(op.variants(753, 'qkv_gate'))
        self.assertTrue(any(v[0] == 'persistent' for v in variants))
        self.assertEqual(variants[0], ('stock',))
        COUNTS['injected_rejections'] += 2

    def test_prepare_and_capture(self):
        # Full production shapes, finite shared budget, cold paired finalists.
        started = time.monotonic()
        op = p.PrefillGemmPlan(self.op.weights, tokens=(16384, 3012, 753), budget_s=90)
        RESULTS.append(dict(prepare_s=time.monotonic() - started,
                            plans={str(k): v for k, v in op.plans.items()},
                            timings={str(k): v for k, v in op.timings.items()}, rejections=len(op.rejections)))
        for m in (753, 3012, 16384):
            for role in (*op.weights, 'pool_router'):
                k = op.weights['pooler' if role == 'pool_router' else role].shape[1]
                x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
                with p._PgGraph(lambda: op(x, role)) as graph:
                    for _ in range(3):
                        x.normal_(); graph.graph.replay(); compare(graph.out, op._stock(x, role))
                        COUNTS['selected_fresh_graphs'] += 1
        self.assertEqual(p._PgGraph.active, 0)

    def test_partial_boundaries(self):
        op = self.op
        for m in (1, 129, 769, 3013):
            x = torch.randn((m, 8192), device='cuda', dtype=torch.bfloat16)
            for role in ('router', 'pool_router'):
                with p._PgGraph(lambda: op.run(x, role, ('persistent', 0, 1))) as graph:
                    for _ in range(20):
                        x.normal_(); graph.graph.replay(); compare(graph.out, op._stock(x, role))
                        COUNTS['streamk_boundary_replays'] += 1
                self.assertEqual(int(op.counts.count_nonzero()), 0)

    def test_norm_epilogue(self):
        w = {name: self.op.weights[name] for name in ('pooler', 'router')}
        effective = torch.randn((4096,), device='cuda') * .01 + 1
        op = p.PrefillGemmPlan(w, tokens=(16384, 3012, 753), budget_s=0, pool_norm_weight=effective)
        for m in (753, 3012, 16384):
            x = torch.randn((m, 8192), device='cuda', dtype=torch.bfloat16)
            configs = [('persistent', 0, 0), ('persistent', 0, 1)]
            for nn in (0, 1):
                self.assertGreater(len(op._options(m, 'pool_router_norm', nn)), 0)
                configs.append(('lt', nn, 0))
            for config in configs:
                with p._PgGraph(lambda: op.run(x, 'pool_router_norm', config)) as graph:
                    for _ in range(3):
                        x.normal_(); graph.graph.replay()
                        compare(graph.out, op._stock(x, 'pool_router_norm'))
                        COUNTS['norm_epilogue_graphs'] += 1
            pooled = op._stock(x, 'pooler').float()
            reference = (pooled * torch.rsqrt(pooled.square().mean(1, keepdim=True) + 1e-5) * effective).bfloat16()
            compare(op._stock(x, 'pool_router_norm')[0], reference)
            COUNTS['norm_fp32_checks'] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestPrefill))
    report = dict(ok=result.wasSuccessful(), tests=result.testsRun, cc=torch.cuda.get_device_capability(),
                  module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(), counts=dict(COUNTS), results=RESULTS)
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
