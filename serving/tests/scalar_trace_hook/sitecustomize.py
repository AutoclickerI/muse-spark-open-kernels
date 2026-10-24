"""Diagnostic only: locate CUDA scalar-to-host copies without changing values."""
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / 'serving/harness/pyhook/sitecustomize.py'))
_state = threading.local()


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.original.exec_module(module)
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        class TraceScalars(TorchDispatchMode):
            def __init__(self, scope):
                super().__init__()
                self.scope, self.count, self.samples = scope, 0, []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if (func._schema.name == 'aten::_local_scalar_dense'
                        and args and args[0].is_cuda):
                    self.count += 1
                    if self.count & (self.count - 1) == 0:
                        self.samples.append(dict(index=self.count,
                            device=str(args[0].device), dtype=str(args[0].dtype),
                            stack=traceback.format_stack(limit=24)))
                return func(*args, **(kwargs or {}))

        def wrap(fn):
            def call(self, *args, **kwargs):
                if getattr(_state, 'active', False):
                    return fn(self, *args, **kwargs)
                mode = TraceScalars(fn.__name__)
                _state.active = True
                try:
                    with mode:
                        return fn(self, *args, **kwargs)
                finally:
                    _state.active = False
                    if mode.count:
                        with open(os.environ['MUSE_SCALAR_TRACE'], 'a') as out:
                            out.write(json.dumps(dict(pid=os.getpid(),
                                scope=mode.scope, count=mode.count, samples=mode.samples))+'\n')
            return call

        for name in ['ingest_requests', 'get_next_batch_to_run', 'run_batch', 'process_batch_result']:
            setattr(module.Scheduler, name, wrap(getattr(module.Scheduler, name)))


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != 'sglang.srt.managers.scheduler':
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        spec.loader = Loader(spec.loader)
        return spec


sys.meta_path.insert(0, Finder())
