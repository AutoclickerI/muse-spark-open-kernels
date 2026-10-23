"""Opt-in host/device spans; no synchronization or changes to model outputs."""
import atexit
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys
import time

runpy.run_path(str(Path(__file__).resolve().parents[1] / "serving/harness/pyhook/sitecustomize.py"))
_path = os.getenv("MUSE_SERVING_TRACE")
_pending = []
_step = 0
_active = False
_profile_counts = {}
_completed = set()
_stopped = False
_anchor = None
_ab_mode = "patch"
_ab_before = None


def emit(row):
    row.update(pid=os.getpid())
    with open(_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def drain():
    for row, start, end in list(_pending):
        if end.query():
            if row.get("kind") == "graph_device":
                emit(dict(row, device_start_ms=_anchor.elapsed_time(start),
                          device_end_ms=_anchor.elapsed_time(end)))
            else:
                emit(dict(row, kind="device", device_interval_ms=start.elapsed_time(end)))
            _pending.remove((row, start, end))


def wrap(cls, name, label, device=False):
    original = getattr(cls, name)

    @functools.wraps(original)
    def call(self, *args, **kwargs):
        global _step, _ab_mode, _ab_before
        if not _active:
            return original(self, *args, **kwargs)
        import torch
        batch = args[0] if args else kwargs.get("batch")
        if device and os.getenv("MUSE_HOST_STEP_AB"):
            block = int(os.environ["MUSE_HOST_STEP_AB"])
            _ab_mode = "baseline" if (_step // block) % 2 == 0 else "patch"
            worker = self.model_worker
            if not hasattr(worker, "_ab_graph"):
                worker._ab_graph = worker._kv_materialize_graph
            worker._kv_materialize_graph = worker._ab_graph if _ab_mode == "patch" else None
        elif device and os.getenv("MUSE_HOST_AB") and batch.forward_mode.is_extend():
            mode = Path(os.environ["MUSE_HOST_AB"]).read_text().strip()
            if mode != _ab_mode:
                import importlib.util
                from types import MethodType
                if _ab_before is None:
                    spec = importlib.util.spec_from_file_location("host_ab_before", os.environ["MUSE_HOST_AB_BEFORE"])
                    _ab_before = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = _ab_before
                    spec.loader.exec_module(_ab_before)
                worker = self.model_worker
                if not hasattr(worker, "_ab_graph"):
                    worker._ab_graph = worker._kv_materialize_graph
                    self.future_map._ab_publish = self.future_map.publish
                    self.future_map._ab_resolve = self.future_map.resolve_seq_lens_cpu
                worker._kv_materialize_graph = worker._ab_graph if mode == "patch" else None
                self.future_map.publish = (self.future_map._ab_publish if mode == "patch" else
                                          MethodType(_ab_before.FutureMap.publish, self.future_map))
                self.future_map.resolve_seq_lens_cpu = (self.future_map._ab_resolve if mode == "patch" else
                                                       MethodType(_ab_before.FutureMap.resolve_seq_lens_cpu, self.future_map))
                _ab_mode = mode
                emit(dict(kind="ab_mode", mode=mode))
        if device:
            _step += 1
            drain()
        row = dict(name=label, step=_step, start_ns=time.perf_counter_ns(), ab_mode=_ab_mode)
        if hasattr(batch, "reqs"):
            row.update(bs=len(batch.reqs), mode=str(batch.forward_mode))
        start = end = None
        if device:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            # Include schedule and forward streams in the outer interval.
            start.record(self.schedule_stream)
        torch.cuda.nvtx.range_push(label)
        profile = None
        if label == "forward_batch_generation" and row.get("mode") == "2" and os.getenv("MUSE_PROFILE_COPIES"):
            bs = row.get("bs")
            _profile_counts[bs] = _profile_counts.get(bs, 0) + 1
            if bs in (1, 4) and _profile_counts[bs] == 8:
                import cProfile
                profile = cProfile.Profile()
                profile.enable()
        try:
            return original(self, *args, **kwargs)
        finally:
            if profile:
                profile.disable()
                profile.dump_stats(_path + f".c{bs}.pstats")
            torch.cuda.nvtx.range_pop()
            row["end_ns"] = time.perf_counter_ns()
            row["host_ms"] = (row["end_ns"] - row["start_ns"]) / 1e6
            emit(dict(row, kind="host"))
            if device:
                end.record(self.forward_stream)
                _pending.append((row, start, end))
            if label == "process_batch_result" and batch is not None:
                global _stopped
                _completed.update(
                    req.rid for req in batch.reqs
                    if req.finished() and req.sampling_params.max_new_tokens == 220
                )
                if (not _stopped and os.getenv("MUSE_SERVING_NSYS") and
                        len(_completed) >= int(os.getenv("MUSE_TRACE_REQUESTS", "30"))):
                    _stopped = True
                    torch.cuda.cudart().cudaProfilerStop()

    setattr(cls, name, call)


def patch(module):
    if module.__name__.endswith(".scheduler"):
        cls = module.Scheduler
        original = cls.run_event_loop

        @functools.wraps(original)
        def loop(self):
            global _active, _anchor
            _active = True
            import torch
            emit(dict(kind="config", enable_overlap=self.enable_overlap,
                      needs_cpu_seq_lens=self.future_map.needs_cpu_seq_lens,
                      scheduler_file=module.__file__,
                      worker_file=sys.modules[type(self.model_worker).__module__].__file__,
                      kv_graph=bool(getattr(self.model_worker, "_kv_materialize_graph", None))))
            if os.getenv("MUSE_SERVING_NSYS"):
                torch.cuda.cudart().cudaProfilerStart()
            if os.getenv("MUSE_TRACE_GRAPH_EVENTS"):
                candidates = []
                for _ in range(5):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    event.synchronize()
                    before_ns = time.perf_counter_ns()
                    event.record()
                    event.synchronize()
                    after_ns = time.perf_counter_ns()
                    candidates.append((after_ns-before_ns, event, before_ns, after_ns))
                _, _anchor, before_ns, after_ns = min(candidates, key=lambda row: row[0])
                emit(dict(kind="anchor", before_ns=before_ns, after_ns=after_ns))
                replay = torch.cuda.CUDAGraph.replay

                def graph_replay(graph):
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    result = replay(graph)
                    end.record()
                    _pending.append((dict(kind="graph_device", step=_step), start, end))
                    return result

                torch.cuda.CUDAGraph.replay = graph_replay
            return original(self)

        cls.run_event_loop = loop
        for name in ("get_next_batch_to_run", "update_running_batch", "process_batch_result"):
            wrap(cls, name, name)
        wrap(cls, "run_batch", "run_batch", device=True)
    elif module.__name__.endswith(".dflash_worker_v2"):
        cls = module.DFlashWorkerV2
        for name in ("forward_batch_generation", "_accept_block", "_append_target_hidden_to_draft_kv_by_loc"):
            wrap(cls, name, name)
    elif module.__name__.endswith(".overlap_utils"):
        wrap(module.FutureMap, "resolve_seq_lens_cpu", "resolve_seq_lens_cpu")
    elif module.__name__.endswith(".dflash_kv_graph"):
        wrap(module.DFlashKVGraph, "__call__", "draft_kv_graph")
    elif module.__name__.endswith(".dflash_overlap") and (os.getenv("MUSE_HOST_AB") or os.getenv("MUSE_HOST_STEP_AB")):
        original = module.DFlashOverlap.callback

        def callback(self, batch):
            return None if _ab_mode == "baseline" else original(self, batch)

        module.DFlashOverlap.callback = callback


class Probe(importlib.abc.MetaPathFinder):
    names = {"sglang.srt.managers.scheduler", "sglang.srt.speculative.dflash_worker_v2",
             "sglang.srt.managers.overlap_utils", "sglang.srt.managers.dflash_kv_graph",
             "sglang.srt.managers.dflash_overlap"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.names:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            original = spec.loader.exec_module

            def execute(module):
                original(module)
                patch(module)

            spec.loader.exec_module = execute
        return spec


if _path:
    sys.meta_path.insert(0, Probe())
    atexit.register(drain)
