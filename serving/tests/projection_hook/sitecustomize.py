"""Force Hopper projection Python branches at tiny fixture shapes."""

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
exec(
    compile(
        (ROOT / "serving/harness/pyhook/sitecustomize.py").read_text(),
        "harness_hook",
        "exec",
    )
)
PROJECTION = "sglang.kernels.ops.muse_spark_v12.projection_hopper"
MODEL = "sglang.srt.models.muse_spark"
WIDE = "sglang.srt.layers.muse_spark_humming"


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == MODEL:
            module._split_bf16_router_supported = lambda: True
            from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts

            original = MuseSpark1xFp4Experts._use_muse_v4
            MuseSpark1xFp4Experts._use_muse_v4 = lambda self, n: (
                (getattr(self, "_humming_candidate", False) and 0 < n <= 256)
                or original(self, n)
            )
            print("HOPPER_PROJECTION router branch forced", flush=True)
            return
        if module.__name__ == WIDE:

            def portable(layer, x, scores, ids):
                if not getattr(layer, "_humming_candidate", False) or x.shape[0] > 256:
                    return None
                y, local_ids = layer._marlin_expert_outputs(x, scores, ids)
                import torch

                flat = torch.arange(
                    ids.numel(), device=ids.device, dtype=torch.int32
                ).view_as(ids)
                return y, torch.where(local_ids >= 0, flat, torch.full_like(flat, -1))

            module.maybe_run_wide = portable
            print("HOPPER_PROJECTION dense wide-producer stand-in enabled", flush=True)
            return
        import torch

        actual_router = module.RouterProjection

        class Router:
            def __init__(self, w):
                self.rows = w.shape[0]
                padded = torch.nn.functional.pad(
                    w, (0, 8192 - w.shape[1], 0, (-self.rows) % 64)
                )
                self.op = actual_router(padded)
                self.high = self.op.high[: self.rows]
                self.low = self.op.low[: self.rows]

            def __call__(self, x):
                return self.op(torch.nn.functional.pad(x, (0, 8192 - x.shape[1])))[
                    :, : self.rows
                ].contiguous()

        module.router_supported = lambda x, w: (
            x.is_cuda and x.dtype == torch.bfloat16 and 0 < x.shape[0] <= 256
        )
        module.qkv_supported = lambda x, w: (
            x.is_cuda and x.dtype == torch.bfloat16 and 0 < x.shape[0] <= 16
        )
        module.RouterProjection = Router
        print("HOPPER_PROJECTION portable projection CUDA kernels enabled", flush=True)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in (PROJECTION, MODEL, WIDE):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            spec.loader = Loader(spec.loader)
            return spec


sys.meta_path.insert(0, Finder())
