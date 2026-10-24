"""Diagnostic-only TTFT timeline; CUDA events are read after stock result sync."""

import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
chain = (
    "integration_hook" if os.environ.get("MUSE_TTFT_PORTABLE") else "../harness/pyhook"
)
path = ROOT / "serving/tests" / chain / "sitecustomize.py"
runpy.run_path(str(path))
TARGETS = {
    "sglang.srt.managers.scheduler",
    "sglang.srt.speculative.dflash_worker_v2",
    "sglang.srt.managers.tp_worker",
    "sglang.srt.models.dflash",
}
active = None


def log(data):
    if data["kind"] == "choose" and not (
        data["waiting"] or data["finished"] or data["chosen"]
    ):
        return
    data["pid"] = os.getpid()
    with open(os.environ["MUSE_TTFT_TRACE"], "a") as out:
        out.write(json.dumps(data) + "\n")


def gpu_span(name, fn):
    def wrapped(self, *args, **kwargs):
        if active is None:
            return fn(self, *args, **kwargs)
        import torch

        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        t = time.perf_counter()
        a.record()
        result = fn(self, *args, **kwargs)
        b.record()
        active.append((name, t, time.perf_counter(), a, b))
        return result

    return wrapped


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__.endswith("tp_worker"):
            cls = module.TpModelWorker
            cls.forward_batch_generation = gpu_span(
                "target_forward", cls.forward_batch_generation
            )
            return
        if module.__name__.endswith("models.dflash"):
            cls = module.DFlashDraftModel
            cls.project_target_hidden = gpu_span("draft_fc", cls.project_target_hidden)
            return
        if module.__name__.endswith("dflash_worker_v2"):
            cls = module.DFlashWorkerV2
            for name in (
                "_append_target_hidden_to_draft_kv_by_loc",
                "_rebuild_compact_draft_cache",
            ):
                setattr(cls, name, gpu_span(name, getattr(cls, name)))
            return
        cls = module.Scheduler
        original_run, original_process = cls.run_batch, cls.process_batch_result
        pending = {}
        reference = None

        def run(self, batch, *args, **kwargs):
            global active
            nonlocal reference
            from sglang.srt.runtime_context import get_parallel

            if not batch.forward_mode.is_extend() or get_parallel().tp_rank != 0:
                return original_run(self, batch, *args, **kwargs)
            import torch

            before = time.perf_counter()
            reqs = [
                {
                    "rid": r.rid,
                    "prompt": len(r.origin_input_ids),
                    "recv": r.time_stats.scheduler_recv_time,
                    "queue": r.time_stats.wait_queue_entry_time,
                    "admitted": r.time_stats.forward_entry_time,
                    "inflight": r.inflight_middle_chunks,
                }
                for r in batch.reqs
            ]
            spans = []
            active = spans
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record(self.forward_stream)
            if reference is None:
                # Startup warmup only, before workload traffic. Later events
                # use this GPU clock anchor without adding synchronization.
                a.synchronize()
                reference = (a, time.perf_counter())
            try:
                result = original_run(self, batch, *args, **kwargs)
            finally:
                active = None
            b.record(self.forward_stream)
            pending[id(result)] = (
                before,
                time.perf_counter(),
                reqs,
                batch.extend_num_tokens,
                a,
                b,
                spans,
            )
            return result

        def process(self, batch, result, *args, **kwargs):
            item = pending.pop(id(result), None)
            enter = time.perf_counter()
            output = original_process(self, batch, result, *args, **kwargs)
            if item is not None:
                before, launched, reqs, tokens, a, b, spans = item
                # Original result processing already waits on copy_done.
                assert b.query()
                log(
                    {
                        "kind": "prefill",
                        "before": before,
                        "launched": launched,
                        "process_enter": enter,
                        "processed": time.perf_counter(),
                        "tokens": tokens,
                        "gpu_ms": a.elapsed_time(b),
                        "reqs": reqs,
                        "gpu_start": reference[1] + reference[0].elapsed_time(a) / 1000,
                        "gpu_end": reference[1] + reference[0].elapsed_time(b) / 1000,
                        "stages": [
                            {
                                "name": n,
                                "before": t0,
                                "after": t1,
                                "gpu_ms": x.elapsed_time(y),
                            }
                            for n, t0, t1, x, y in spans
                        ],
                        "finished": {
                            r.rid: r.time_stats.prefill_finished_time
                            for r in batch.reqs
                        },
                    }
                )
            return output

        original_choose = cls.get_next_batch_to_run

        def choose(self, running_batch, last_batch):
            from sglang.srt.runtime_context import get_parallel

            if get_parallel().tp_rank != 0:
                return original_choose(self, running_batch, last_batch)
            before = time.perf_counter()
            waiting = [r.rid for r in self.waiting_queue]
            running_count = len(running_batch.reqs)
            finished_count = sum(r.finished() for r in running_batch.reqs)
            full = running_batch.batch_is_full
            plan = original_choose(self, running_batch, last_batch)
            batch = plan.batch_to_run
            log(
                {
                    "kind": "choose",
                    "before": before,
                    "after": time.perf_counter(),
                    "waiting": waiting,
                    "running": running_count,
                    "finished": finished_count,
                    "full": full,
                    "chosen": None if batch is None else str(batch.forward_mode),
                    "chosen_reqs": [] if batch is None else [r.rid for r in batch.reqs],
                }
            )
            return plan

        cls.get_next_batch_to_run = choose
        cls.run_batch, cls.process_batch_result = run, process


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name in TARGETS:
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            spec.loader = Loader(spec.loader)
            return spec


sys.meta_path.insert(0, Finder())
