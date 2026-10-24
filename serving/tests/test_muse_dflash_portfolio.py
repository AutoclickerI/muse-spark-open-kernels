"""TP8 DFlash portfolio agreement and capture checks (GPU0, sm120 labelled)."""
import argparse
import json
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.srt.layers.muse_dflash import DFlashDrafterPortfolio


ROLE_SHAPES = DFlashDrafterPortfolio.ROLE_SHAPES


def _capture(fn):
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = {}
        for role, (n, k, _mult, stock_nn) in ROLE_SHAPES.items():
            shape = (k, n) if stock_nn else (n, k)
            cls.weights[role] = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / (k ** 0.5)
        cls.portfolio = DFlashDrafterPortfolio(
            cls.weights, tokens=(8, 32, 128, 256), budget_s=0
        )

    def test_dense_roles_eager_and_graph(self):
        for role, (n, k, _mult, stock_nn) in ROLE_SHAPES.items():
            m = 8
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            weight = self.weights[role]
            expected = torch.mm(x, weight) if stock_nn else F.linear(x, weight)
            graph, got = _capture(lambda role=role, x=x: self.portfolio(role, x))
            self.assertTrue(torch.equal(got, expected), role)
            x.copy_(torch.randn_like(x))
            graph.replay()
            expected = torch.mm(x, weight) if stock_nn else F.linear(x, weight)
            self.assertTrue(torch.equal(got, expected), role)
            graph.reset()

    def test_alias_and_concat_capture(self):
        self.assertEqual(self.portfolio._role("fc"), "context_fc")
        m = 8
        left = torch.randn((m, 20480), device="cuda", dtype=torch.bfloat16)
        right = torch.randn_like(left)
        weight = self.weights["context_fc"]
        expected = F.linear(torch.cat((left, right), dim=-1), weight)
        graph, got = _capture(lambda: self.portfolio.fc_concat(left, right))
        self.assertTrue(torch.equal(got, expected))
        left.copy_(torch.randn_like(left))
        right.copy_(torch.randn_like(right))
        graph.replay()
        expected = F.linear(torch.cat((left, right), dim=-1), weight)
        self.assertTrue(torch.equal(got, expected))
        graph.reset()

    def test_unknown_and_shape_guards(self):
        with self.assertRaises(KeyError):
            self.portfolio("missing", torch.empty((8, 1), device="cuda", dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            DFlashDrafterPortfolio({"qkv": torch.empty((4, 4), device="cuda", dtype=torch.bfloat16)}, budget_s=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report = {"ok": result.wasSuccessful(), "tests": result.testsRun,
              "device": torch.cuda.get_device_name(), "cc": torch.cuda.get_device_capability()}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
