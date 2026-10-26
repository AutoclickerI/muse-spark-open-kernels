import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import okc_run as run


class QueueDecisionRegression(unittest.TestCase):
    def test_rank_by_decision_not_host_and_explicit_priority_still_works(self):
        self.assertEqual(run.queue_metadata('P4', 'promote held-out winner', priority=-99)['priority'], 1)
        self.assertEqual(run.queue_metadata('P10', 'rank exact coding')['priority'],run.ranked_lanes()['P10'])
        self.assertEqual(run.queue_metadata(None, 'legacy', priority=-1)['priority'], -1)
        # All unspecified lanes share the same default, independent of transport host.
        ranks=run.ranked_lanes()
        self.assertEqual(run.queue_metadata(None,'legacy')['lane'],'P1' if 'P1' in ranks else 'unspecified')
        self.assertEqual(run.queue_metadata(None,'legacy')['priority'],ranks.get('P1',max(ranks.values())+1))
        with self.assertRaises(ValueError):
            run.queue_metadata('P404', 'unknown')
        self.assertEqual(run.queue_metadata('P4', 'one\nline')['decision'], 'one line')

    def test_completed_P1_does_not_break_legacy_sends(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=Path(tmp)/'PLAN.md'
            plan.write_text('## Ranked work\n| 1 | P4 |\n| 3 | P10 |\n')
            with patch.object(run,'PLAN',plan):
                old=run.queue_metadata(None,'legacy')
                self.assertEqual(old['lane'],'unspecified')
                self.assertEqual(old['priority'],4)
                self.assertEqual(run.queue_metadata(None,'manual',priority=-1)['priority'],-1)
                with self.assertRaises(ValueError):run.queue_metadata('P1','closed lane')

    def test_waiters_refresh_PLAN_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp)/'PLAN.md'
            plan.write_text('## Ranked work\n| Rank | Id |\n|---|---|\n| 1 | P10 |\n| 2 | P4 |\n')
            entries = [(Path('p4'), dict(priority=1, lane='P4', t=1)),
                       (Path('p10'), dict(priority=3, lane='P10', t=2))]
            with patch.object(run, 'PLAN', plan), patch.object(run, 'entries', return_value=entries):
                self.assertEqual(run.waiters()[0][2], Path('p10'))

    def test_decision_metadata_reaches_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'candidate.py'
            source.write_text('# okc: problem=muse_prefill_layer\n')
            (root/'runs').mkdir()
            with patch.object(run, 'REPO', root), patch.object(run, 'PENDING', root/'runs/pending'), \
                 patch.object(run, 'LEDGER', root/'runs/ledger.jsonl'), patch.object(run, 'git_rev', return_value='test'), \
                 patch.object(run, 'okc_json', return_value=({'status': 'done', 'result': {'passed': True}}, None)):
                metadata = run.queue_metadata('P4', 'confirm composition')
                run.start_receipt('muse_prefill_layer', 'check', 'job', source.read_bytes(), source, '', 'stamp', **metadata)
                run.finish('job')
                row = json.loads((root/'runs/ledger.jsonl').read_text())
                self.assertEqual({k: row[k] for k in metadata}, metadata)
                self.assertEqual(row['priority'], 1)


if __name__ == '__main__':
    unittest.main()
