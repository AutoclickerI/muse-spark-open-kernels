"""Exact small-tree enumeration checks for the acceptance-ceiling estimator."""

import itertools
import unittest

from analyze_tree_ceiling import chain_expectation, coverage, optimal_tree


class TreeCeilingTest(unittest.TestCase):
    def test_best_first_matches_exhaustive_prefix_trees(self):
        probabilities = [
            [[0.65, 0.35], [0.65, 0.35]],
            [[0.7, 0.3], [0.4, 0.6]],
            [[0.8, 0.2], [0.55, 0.45]],
        ]
        paths = [
            p
            for depth in range(1, 4)
            for p in itertools.product(range(2), repeat=depth)
        ]

        def mass(path):
            result, prev = 1.0, 0
            for depth, index in enumerate(path):
                result *= probabilities[depth][prev][index]
                prev = index
            return result

        for budget in range(1, 5):
            best = 0.0
            for nodes in itertools.combinations(paths, budget):
                selected = set(nodes)
                if all(
                    path[:depth] in selected
                    for path in nodes
                    for depth in range(1, len(path))
                ):
                    best = max(best, 1 + sum(map(mass, nodes)))
            nodes, expected = optimal_tree(probabilities, budget)
            self.assertAlmostEqual(expected, best)
            self.assertEqual(len(nodes), budget)

    def test_certain_chain_has_no_tree_gain(self):
        probabilities = [[[0.999, 0.001], [0.001, 0.999]] for _ in range(7)]
        nodes, expected = optimal_tree(probabilities)
        path = (0,) * 7
        self.assertEqual(set(nodes), {path[:d] for d in range(1, 8)})
        self.assertAlmostEqual(expected, chain_expectation(probabilities, path))

    def test_truth_coverage_requires_the_whole_ancestry(self):
        ids = [[10, 11], [20, 21], [30, 31]]
        nodes = [(0,), (1,), (1, 0), (1, 0, 1)]
        self.assertEqual(coverage(nodes, ids, [11, 20, 31, 99]), 4)
        self.assertEqual(coverage(nodes, ids, [10, 20, 31, 99]), 2)
        self.assertEqual(coverage(nodes, ids, [12, 20, 31, 99]), 1)


if __name__ == "__main__":
    unittest.main()
