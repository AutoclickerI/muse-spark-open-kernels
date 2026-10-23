"""CPU contract checks for preparation against an in-flight batch."""

from contextlib import nullcontext
from collections import deque
from pathlib import Path
import importlib.util
from types import SimpleNamespace as NS
import unittest

path = Path(__file__).resolve().parents[1] / "sglang/python/sglang/srt/managers/dflash_overlap.py"
spec = importlib.util.spec_from_file_location("dflash_overlap", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PlanTests(unittest.TestCase):
    def setup_pipeline(self):
        calls = []
        req = NS(committed=10, aborted=False)
        draft = NS(future_indices=object(), _prepare_length_buf=object())
        batch = NS(spec_algorithm=NS(is_dflash=lambda: True),
                   forward_mode=NS(is_decode=lambda: True), has_grammar=False,
                   spec_info=draft, reqs=[req])
        scheduler = NS(device="cuda", running_batch=batch, last_batch=object(),
                       require_mlp_sync=False, waiting_queue=[], chunked_req=None,
                       enable_hisparse=False, model_worker=NS(),
                       device_module=NS(stream=lambda _: nullcontext()),
                       schedule_stream=object(), result_queue=deque([(object(), object())]))
        scheduler._apply_war_barrier = lambda: calls.append("fence")

        def process(*_):
            calls.append("output")
            req.committed = 18

        scheduler.process_batch_result = process

        def build(*, running_batch, last_batch):
            calls.append("build")
            self.assertIs(running_batch, last_batch)
            self.assertIsNot(running_batch, batch)
            self.assertTrue(running_batch.forward_mode.is_decode())
            self.assertIsNot(running_batch.spec_info, draft)
            self.assertIs(running_batch.spec_info.future_indices, draft.future_indices)
            self.assertIsNone(running_batch.spec_info._prepare_length_buf)
            self.assertEqual(running_batch.reqs[0].committed, 18)
            running_batch.filter_batch = lambda: setattr(running_batch, "reqs", [r for r in running_batch.reqs if not r.aborted])
            running_batch.is_empty = lambda: not running_batch.reqs
            return NS(batch_to_run=running_batch, running_batch=running_batch)

        scheduler.get_next_batch_to_run = build
        return module.DFlashOverlap(scheduler), scheduler, batch, req, calls

    def test_snapshot_lifetime_and_output_order(self):
        pipeline, scheduler, batch, req, calls = self.setup_pipeline()
        callback = pipeline.callback(batch)
        old_draft = batch.spec_info
        # These are the worker's real classes of mutation after the snapshot.
        batch.forward_mode = NS(is_decode=lambda: False)
        batch.spec_info = NS()
        callback()
        self.assertEqual(calls, ["fence", "output", "build"])
        self.assertTrue(pipeline.processed_previous)
        self.assertFalse(scheduler.result_queue)
        self.assertIsNotNone(old_draft._prepare_length_buf)
        plan = pipeline.take_prepared()
        self.assertIsNotNone(plan.batch_to_run)
        self.assertIsNone(pipeline.take_prepared())

    def test_abort_after_preparation(self):
        pipeline, scheduler, batch, req, calls = self.setup_pipeline()
        pipeline.callback(batch)()
        req.aborted = True
        plan = pipeline.take_prepared()
        self.assertIsNone(plan.batch_to_run)

    def test_boundaries_keep_normal_scheduling(self):
        for change in (lambda s, b: setattr(b, "has_grammar", True),
                       lambda s, b: setattr(s, "waiting_queue", [object()]),
                       lambda s, b: setattr(s, "chunked_req", object()),
                       lambda s, b: setattr(s, "require_mlp_sync", True),
                       lambda s, b: setattr(s, "device", "cpu")):
            pipeline, scheduler, batch, req, calls = self.setup_pipeline()
            change(scheduler, batch)
            self.assertIsNone(pipeline.callback(batch))
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
