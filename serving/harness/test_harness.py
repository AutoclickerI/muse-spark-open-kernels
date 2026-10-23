"""CPU checks for failure modes that could make an invalid serving run look green."""
import copy
import unittest
from unittest.mock import patch

from compare import agreement, compare
from run import build_workload, summarize, graph_errors, run_level


class HarnessChecks(unittest.TestCase):
    def test_empty_outputs_never_pass(self):
        self.assertFalse(agreement([], [])['exact'])
        self.assertEqual(agreement([], [])['prefix'], 0)
        self.assertFalse(summarize([], [1])['1']['all_220'])

    def fixture(self):
        meta = dict(label='test', tp=1, workload=dict(sha256='same', requests=1),
                    checkpoint_sha256=dict(target='a', draft='b'), sampling_params={})
        row = dict(key='a', level=1, conv=0, turn=0, output_ids=list(range(220)), n_out=220)
        return meta, {'a': row}

    def test_missing_requests_fail(self):
        m, r = self.fixture()
        self.assertTrue(compare(m, {}, m, {})['problems'])
        self.assertTrue(compare(m, r, m, {})['problems'])

    def test_checkpoint_identity(self):
        m, r = self.fixture()
        other = copy.deepcopy(m)
        other['checkpoint_sha256']['target'] = 'changed'
        self.assertIn('different checkpoint contents', compare(m, r, other, r)['problems'])

    def test_load_audit_failure(self):
        m, r = self.fixture()
        m['load_check'] = {'model': {'unwritten_or_partial': {'missing': 1.0}}}
        self.assertTrue(compare(m, r, m, r)['problems'])

    def test_agreement(self):
        a = list(range(220)); b = a.copy(); b[7] = -1
        s = agreement(a, b)
        self.assertFalse(s['exact'])
        self.assertEqual(s['first_diff'], 7)
        self.assertEqual(s['first_divergence_position'], 7)
        self.assertEqual(s['positional'], 219/220)
        self.assertEqual(s['token_agreement_rate'], 219/220)

    def test_output_count_cannot_hide_missing_ids(self):
        m, r = self.fixture()
        r['a']['output_ids'] = []
        self.assertTrue(compare(m, r, m, r)['problems'])

    def test_multiturn_and_context_coverage(self):
        w = build_workload(0, [1, 4], {1: 1, 4: 4}, 2, [2600, 5200, 18000, 3900])
        self.assertEqual(len(w), 10)
        self.assertTrue(any(len(x['input_ids']) > 16384 for x in w))
        for first, second in zip(w[::2], w[1::2]):
            self.assertEqual(first['input_ids'][:-3], second['input_ids'][:len(first['input_ids'])-3])

    def test_automatic_graph_fallback_fails(self):
        log = dict(capture_completed={'target prefill': 1, 'target verify': 1, 'draft verify': 0},
                   prefill_graph_replayed=True, decode_graph_replayed=True)
        self.assertTrue(graph_errors(log, 1, []))
        self.assertFalse(graph_errors(log, 1, ['--disable-cuda-graph']))

    def test_request_evidence_written_before_level_returns(self):
        items = build_workload(0, [1], {1: 1}, 2, [20])
        persisted = []
        response = dict(output_ids=list(range(220)), completion_tokens=220,
                        finish_reason={'type': 'length'}, ttft=0.01, e2e=0.1)
        def generate(port, ids, timeout):
            if len(ids) != len(items[0]['input_ids']):
                self.assertEqual(len(persisted), 1)
            return dict(response, prompt_tokens=len(ids))
        with patch('run.stream_generate', side_effect=generate), patch('builtins.print'):
            result = run_level(1, 1, items, 1, persisted.append)
        self.assertEqual(len(persisted), 2)
        self.assertEqual(result, persisted)
        self.assertFalse(any('error' in row for row in result))

    def test_teacher_forced_diff_summary(self):
        # The metric helper is intentionally pure and must ignore the leading
        # null logprob for the first input token.
        from compare_teacher_forced import diffs
        self.assertEqual(diffs([None, -1.0, -2.0], [None, -1.25, -1.5]),
                         {'count': 2, 'max_abs_diff': 0.5, 'mean_abs_diff': 0.375})


if __name__ == '__main__':
    unittest.main()
