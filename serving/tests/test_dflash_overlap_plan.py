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


class Batch(NS):
    def filter_batch(self):
        self.reqs = [r for r in self.reqs if not r.aborted and not r.finished()]

    def is_empty(self):
        return not self.reqs


class PlanTests(unittest.TestCase):
    def setup_pipeline(self):
        calls = []
        req = NS(committed=10, aborted=False, complete=False)
        req.finished = lambda: req.complete
        draft = NS(future_indices=object(), _prepare_length_buf=object())
        batch = Batch(spec_algorithm=NS(is_dflash=lambda: True),
                   forward_mode=NS(is_decode=lambda: True), has_grammar=False,
                   spec_info=draft, reqs=[req], batch_is_full=False)
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
            self.assertEqual(req.committed, 18)
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

    def test_next_turn_prefill_after_completion_has_priority(self):
        pipeline, scheduler, batch, req, calls = self.setup_pipeline()
        # Other requests remain runnable after this conversation completes.
        survivor = NS(committed=10, aborted=False, finished=lambda: False)
        batch.reqs.append(survivor)
        batch.batch_is_full = True
        process = scheduler.process_batch_result

        def complete(*args):
            process(*args)
            req.complete = True

        scheduler.process_batch_result = complete
        callback = pipeline.callback(batch)
        callback()
        # The completed result makes the client send its next turn. Ingress
        # happens in the next scheduler iteration, after the callback returned.
        prefill = object()
        scheduler.waiting_queue.append(prefill)

        def choose(*, running_batch, last_batch):
            # Match prefill admission: a finished row still occupying the
            # full running batch prevents a new request from being admitted.
            if scheduler.waiting_queue and len(running_batch.reqs) < 2:
                self.assertFalse(running_batch.batch_is_full)
                return NS(batch_to_run=prefill, running_batch=running_batch)
            return NS(batch_to_run=running_batch, running_batch=running_batch)

        scheduler.get_next_batch_to_run = choose
        plan = pipeline.take_prepared()
        if plan is None:
            plan = scheduler.get_next_batch_to_run(running_batch=batch, last_batch=batch)
        self.assertIs(plan.batch_to_run, prefill, "prepared decode displaced the next-turn prefill")
        self.assertEqual(calls, ["fence", "output"])
        self.assertTrue(pipeline.processed_previous)

    def test_normal_result_completion_releases_admission_slot(self):
        """Ordinary completions must release slots before prefill admission."""
        # The31/13 case is the captured c32 idle pass. The partial case must
        # keep a surviving decode row while admitting the next conversation.
        for done, live, waiting, cap in ((31, 0, 13, 32), (1, 1, 1, 2)):
            with self.subTest(done=done, live=live):
                pipeline, scheduler, batch, req, calls = self.setup_pipeline()
                batch.reqs = [NS(aborted=False, finished=lambda: True) for _ in range(done)]
                batch.reqs += [NS(aborted=False, finished=lambda: False) for _ in range(live)]
                batch.batch_is_full = True
                scheduler.waiting_queue = [object() for _ in range(waiting)]
                # This was the ordinary end-of-loop pop, not the callback.
                scheduler.result_queue.popleft()
                self.assertFalse(pipeline.processed_previous)
                self.assertIsNone(pipeline.take_prepared())
                self.assertGreaterEqual(cap - len(batch.reqs), waiting,
                                        "stale rows block queued prefills")
                self.assertEqual(len(batch.reqs), live)
                self.assertFalse(batch.batch_is_full)

    def test_first_token_flush_does_not_pop_result_twice(self):
        pipeline, scheduler, batch, req, calls = self.setup_pipeline()
        # The loop drained the prefill result before starting this decode.
        # last_batch still names prefill until this iteration finishes.
        scheduler.process_batch_result(*scheduler.result_queue.popleft())
        pipeline.callback(batch)()
        self.assertEqual(calls, ["output", "fence", "build"])
        self.assertFalse(pipeline.processed_previous)
        self.assertIsNotNone(pipeline.take_prepared())

    def test_first_token_precedes_next_forward(self):
        """A finished prefill must not wait for the following expensive forward."""
        import ast
        source = (path.parent / "scheduler.py").read_text()
        method = next(n for n in ast.walk(ast.parse(source))
                      if isinstance(n, ast.FunctionDef) and n.name == "is_disable_overlap_for_batch")
        method.decorator_list = []
        namespace = dict(ScheduleBatch=object, Optional=__import__('typing').Optional,
                         envs=NS(SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=NS(get=lambda: False)))
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<scheduler-policy>', 'exec'), namespace)
        pipeline, scheduler, batch, req, calls = self.setup_pipeline()
        scheduler.require_mlp_sync = False
        scheduler.dflash_overlap = pipeline
        batch.forward_mode = NS(is_decode=lambda: True, is_extend=lambda: False)
        batch.spec_algorithm.is_none = lambda: False
        batch.grammar_needs_sync = lambda: False
        last = NS(forward_mode=NS(is_extend=lambda: True),
                  spec_algorithm=NS(is_dflash=lambda: True), contains_last_prefill_chunk=True)
        # Model the actual loop order: launch-next used to precede emit-first.
        flush = namespace['is_disable_overlap_for_batch'](scheduler, batch, last)
        actions = []
        if flush:
            actions.append('emit-first')
        actions.append('launch-next')
        if not flush:
            actions.append('emit-first')
        self.assertEqual(actions[0], 'emit-first')
        # No user-visible token on a middle chunk; decode keeps pipelining.
        last.contains_last_prefill_chunk = False
        self.assertFalse(namespace['is_disable_overlap_for_batch'](scheduler, batch, last))
        last.forward_mode = NS(is_extend=lambda: False)
        self.assertFalse(namespace['is_disable_overlap_for_batch'](scheduler, batch, last))
        last.forward_mode = NS(is_extend=lambda: True)
        last.contains_last_prefill_chunk = True
        scheduler.require_mlp_sync = True
        self.assertFalse(pipeline.needs_first_token_flush(last))
        scheduler.require_mlp_sync = False
        scheduler.result_queue.clear()
        self.assertFalse(namespace['is_disable_overlap_for_batch'](scheduler, batch, last))

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
