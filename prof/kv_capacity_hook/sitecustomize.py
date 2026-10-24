"""Read-only per-process CUDA allocation census for the local serving harness."""
import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
exec(compile((ROOT / 'serving/harness/pyhook/sitecustomize.py').read_text(),
             'harness_hook', 'exec'))
OUT = Path(os.environ['KV_CAPACITY_DIR'])
OUT.mkdir(parents=True, exist_ok=True)
COUNT = 0


def scalar(value):
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'cannot serialize {type(value).__name__}')


def snapshot(stage, owner, census=False):
    import torch
    global COUNT
    if not torch.cuda.is_initialized():
        return
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    data = dict(stage=stage, pid=os.getpid(), source=getattr(sys.modules.get(type(owner).__module__), '__file__', ''),
                role='draft' if getattr(owner, 'is_draft_worker', False) else 'target',
                allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                cuda_free=torch.cuda.mem_get_info()[0],
                pre_model_load_memory=getattr(owner, 'pre_model_load_memory', None),
                max_total_num_tokens=getattr(owner, 'max_total_num_tokens', None),
                max_running_requests=getattr(owner, 'max_running_requests', None))
    pool = getattr(owner, 'token_to_kv_pool', None)
    if pool is not None:
        data['pool'] = dict(type=type(pool).__name__)
        for name in ('size', 'size_full', 'size_swa', 'post_capture_active', 'post_capture_backed_bytes'):
            value = getattr(pool, name, None)
            if isinstance(value, (int, float, bool, str)):
                data['pool'][name] = value
        try:
            data['pool']['kv_size_bytes'] = pool.get_kv_size_bytes()
        except (AttributeError, NotImplementedError):
            pass
    if census:
        seen, storages = set(), {}

        def walk(value, path):
            if torch.is_tensor(value):
                if value.device.type != 'cuda':
                    return
                storage = value.untyped_storage()
                key = str((value.device.index, storage.data_ptr(), storage.nbytes()))
                if key not in storages:
                    storages[key] = dict(bytes=storage.nbytes(), paths=[], shape=list(value.shape),
                                         dtype=str(value.dtype))
                if len(storages[key]['paths']) < 12:
                    storages[key]['paths'].append(path)
                return
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(v, path + '.' + str(k))
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    walk(v, path + '.' + str(i))
            elif isinstance(value, torch.nn.Module) or type(value).__module__.startswith('sglang.'):
                for k, v in vars(value).items() if hasattr(value, '__dict__') else ():
                    if not k.startswith('__'):
                        walk(v, path + '.' + k)

        walk(owner, 'root')
        data['storages'] = list(storages.values())
        data['storage_bytes'] = sum(row['bytes'] for row in storages.values())
        segments = torch.cuda.memory_snapshot()
        grouped = {}
        for segment in segments:
            key = str(segment.get('segment_pool_id', (0, 0)))
            row = grouped.setdefault(key, dict(reserved=0, allocated=0, requested=0, segments=0))
            row['reserved'] += segment['total_size']
            row['segments'] += 1
            for block in segment['blocks']:
                if block['state'].startswith('active'):
                    row['allocated'] += block['size']
                    row['requested'] += block.get('requested_size', 0)
        data['allocator_pools'] = grouped
        (OUT / f'{os.getpid()}-{COUNT:03d}-snapshot.json').write_text(json.dumps(segments, default=scalar))
    (OUT / f'{os.getpid()}-{COUNT:03d}-{stage}.json').write_text(json.dumps(data, indent=2, default=scalar) + '\n')
    COUNT += 1


def wrap(cls, name, census=False):
    original = getattr(cls, name)

    @functools.wraps(original)
    def call(self, *args, **kwargs):
        snapshot(name + '-before', self)
        result = original(self, *args, **kwargs)
        snapshot(name + '-after', self, census)
        return result

    setattr(cls, name, call)


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == 'sglang.srt.model_executor.model_runner':
            for name in ('load_model', 'alloc_memory_pool', 'init_cuda_graphs', 'post_capture_resize_kv_pool'):
                wrap(module.ModelRunner, name, census=True)
        elif module.__name__ == 'sglang.srt.managers.scheduler':
            wrap(module.Scheduler, '__init__', census=True)
            original = module.Scheduler.process_batch_result

            @functools.wraps(original)
            def process(self, batch, result):
                value = original(self, batch, result)
                if any(r.finished() and r.sampling_params.max_new_tokens == 220 for r in batch.reqs):
                    snapshot('request-completed', self, census=True)
                return value

            module.Scheduler.process_batch_result = process


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in ('sglang.srt.model_executor.model_runner', 'sglang.srt.managers.scheduler'):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            spec.loader = Loader(spec.loader)
            return spec


sys.meta_path.insert(0, Finder())
