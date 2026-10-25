"""Offline regression checks against immutable H200 profiles; no CUDA or judge calls."""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import build
import scenarios


class LeverChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}
        for job, (tag, _) in build.LAB_SOURCES.items():
            path = next((build.P.parent/'lab/profiles').glob('*.'+job+'.json'))
            data = build.profile_file(path)
            cls.models[tag] = build.make_model(build.source_tag(data), path, data)

    def test_scores_from_independent_load_sums(self):
        for tag, model in self.models.items():
            path = build.R/model['profile_source']
            data = build.profile_file(path)
            ratios = []
            for load in model['loads_us']:
                cases = [b for b in data['benchmarks'] if b['case']['load'] == load]
                baseline = sum(b['case']['weight']*b['baseline_ns'] for b in cases)
                elapsed = sum(b['case']['weight']*b['median_ns'] for b in cases)
                self.assertAlmostEqual(elapsed/1000, model['loads_us'][load])
                ratios.append(baseline/elapsed)
            self.assertAlmostEqual(math.prod(ratios)**(1/6), data['summary']['speedup'], places=12)
            self.assertEqual(model['modeled_speedup'], model['baseline_source_speedup'])
            self.assertEqual(scenarios.pending(tag), [])
            for row in model['rows']:
                self.assertEqual((row['delta_scale_us'], row['delta_m256_us']), (0., 0.))

    def test_default_is_incumbent(self):
        tag, _, _ = build.choose_source(None)
        self.assertEqual(tag, 'S11')

    def test_source_rejection(self):
        data = build.profile_file(build.R/self.models['S11']['profile_source'])
        with patch.dict(build.LAB_SOURCES, {data['job_id']: ('S11', '0'*64)}):
            with self.assertRaisesRegex(AssertionError, 'lab wrapper SHA'):
                build.source_tag(data)
        # Even a valid wrapper hash cannot admit an unrecognized timed prefix.
        candidates = tuple((tag, stem, '0'*64 if tag=='S11' else sha)
                           for tag, stem, sha in build.CANDIDATES)
        with patch.object(build, 'CANDIDATES', candidates):
            with self.assertRaises(AssertionError):
                build.source_tag(data)
        with patch.object(build, 'LAB_SOURCES', {}):
            with self.assertRaisesRegex(ValueError, 'Unsupported source SHA'):
                build.source_tag(data)

    def test_projection_dispatch_and_gate_allocation(self):
        for model in self.models.values():
            for row in model['rows']:
                self.assertAlmostEqual(sum(row['stages_us'][s] for s in build.ROLES),
                                       sum(g['us'] for g in row['projection_groups']))
                for stage in build.ROLES:
                    lo, hi = row['role_bounds_us'][stage]
                    self.assertLessEqual(lo, row['stages_us'][stage]+1e-9)
                    self.assertGreaterEqual(hi+1e-9, row['stages_us'][stage])
            first = model['rows'][0]
            balanced = next(g for g in first['projection_groups'] if '_gb_kernel' in g['names'])
            gate = next(g for g in first['projection_groups'] if '_gate_split_kernel' in g['names'])
            self.assertEqual(gate['roles'], ('qkv_gate',))
            expected = gate['us']+balanced['us']*(5/9*first['floors_us']['qkv_gate'])/sum(
                first['floors_us'][s]*balanced['floor_fractions'].get(s, 1.) for s in balanced['roles'])
            self.assertAlmostEqual(first['stages_us']['qkv_gate'], expected)

    def test_attention_and_moe_families(self):
        for tag, model in self.models.items():
            for row in model['rows']:
                if row['tokens'] >= 128 and row['case']['layer']=='full':
                    vq8 = sum(k['us'] for k in row['kernels'] if 'vq8_' in k['name'])
                    self.assertGreater(vq8, 150)
                    self.assertEqual(row['stages_us']['attention'], vq8)
                if row['tokens'] >= 128:
                    moe = sum(k['us'] for k in row['kernels'] if any(s in k['name'] for s in build.MOE_KERNELS))
                    self.assertAlmostEqual(row['stages_us']['moe'], moe-row['moe_branch_overlap_us'])
                    finish = next(k['name'] for k in row['kernels'] if 'mwbr::finish<' in k['name'])
                    self.assertIn('finish<4>' if tag=='S12' else 'finish<1>', finish)
                self.assertAlmostEqual(row['stages_us']['other'], row['unreported_us'])

    def test_ideal_floors_and_stream_are_distinct(self):
        for model in self.models.values():
            for row in model['rows']:
                for role, nbytes in row['work']['projection_bytes'].items():
                    # All fixture projections below t=256 are byte-bound.
                    if row['tokens'] < 256:
                        self.assertAlmostEqual(row['floors_us'][role], nbytes/4.8e12*1e6)
                    self.assertAlmostEqual(row['stream_roof_us'][role], max(
                        row['floors_us'][role], 2.36+nbytes/4.58e12*1e6))
        s11, s12 = (self.models[tag]['rows'][0] for tag in ('S11', 'S12'))
        self.assertEqual(s11['floors_us']['qkv_gate'], s12['floors_us']['qkv_gate'])
        self.assertLess(s12['floors_us']['pool_router'], s11['floors_us']['pool_router'])
        self.assertEqual(s12['work']['projection_bytes']['pool_router'],
                         4608*8192*1.5 + 288*4 + (288*128+1)*4)
        coded = [g for g in s12['projection_groups'] if any('cdg::coded_gemv<' in n for n in g['names'])]
        self.assertEqual([g['roles'] for g in coded], [('pool_router',)])

    def test_targets_and_absolute_excess(self):
        for model in self.models.values():
            for lane in model['ranking']:
                a, b, c = (lane['gains_pct'][str(f)] for f in build.TARGETS)
                self.assertLessEqual(a, b)
                self.assertLessEqual(b, c)
                self.assertEqual(lane['gain_pct'], b)
            for entry in model['excess_1_3_by_stage']:
                independent = sum(row['case']['weight']*max(0., row['stages_us'][entry['stage']]-
                                  1.3*row['floors_us'][entry['stage']]) for row in model['rows'])
                self.assertAlmostEqual(entry['weighted_us'], independent)
            # Counterfactual calculation also agrees with the separate scenarios tool.
            for lane in model['ranking']:
                select = {'attention': lambda r,k: k=='attention',
                          'proj GEMV': lambda r,k: k in build.ROLES,
                          'glue': lambda r,k: k in ('boundaries','route'),
                          'small-M MoE': lambda r,k: k=='moe' and r['tokens']<=32,
                          'MoE M128': lambda r,k: k=='moe' and r['tokens']==128,
                          'free M256': lambda r,k: k=='moe' and r['tokens']==256}[lane['owner']]
                for target in build.TARGETS:
                    capped = scenarios.cap(model['rows'], select, target)
                    gain = 100*(scenarios.score(model,capped)/model['modeled_speedup']-1)
                    self.assertAlmostEqual(gain, lane['gains_pct'][str(target)])

    def test_prior_source_keeps_accounting(self):
        path = next(build.P.glob('cand_r3_best7_s10.*.json'))
        data = build.profile_file(path)
        model = build.make_model(build.source_tag(data), path, data)
        self.assertAlmostEqual(model['modeled_speedup'], data['summary']['speedup'], places=12)


if __name__ == '__main__':
    unittest.main()
