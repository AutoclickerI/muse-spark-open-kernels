"""Exact-shape portable stand-ins for Hopper serving projection Python glue.

Never pads weights or activations. Real full-size kernels have separate tests.
"""

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
ADAPTER = "sglang.srt.layers.muse_spark_serving"
MODEL = "sglang.srt.models.muse_spark"


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == ADAPTER:
            import torch
            import torch.nn.functional as F

            class PortableProjection:
                def __init__(self, weight, kind, **kwargs):
                    self.weight = weight.detach()

                def __call__(self, x):
                    return F.linear(x, self.weight)

            module.MuseProjection = PortableProjection
            module.projection_supported = lambda x, w, kind: (
                x.is_cuda
                and x.dtype == w.dtype == torch.bfloat16
                and x.ndim == w.ndim == 2
                and x.shape[1] == w.shape[1]
                and 0 < x.shape[0] <= 256
            )

            def portable_v9(state, x, scores, ids):
                layer = state["portable_layer"]
                dense, local_ids = layer._marlin_expert_outputs(x, scores, ids)
                flat = torch.arange(
                    ids.numel(), device=ids.device, dtype=torch.int32
                ).view_as(ids)
                return dense, torch.where(local_ids >= 0, flat, -1)

            module.run_v9_dense = portable_v9
            print(
                "SERVING_PROJECTION exact-shape o/pooler/unpooler ABI stand-ins",
                flush=True,
            )
        else:
            # Tiny models have a BF16 o projection, so force all three seams.
            # No production predicate is modified on disk.
            module._split_bf16_router_supported = lambda: False
            from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts

            original = MuseSpark1xFp4Experts._use_muse_v4

            def use_v9(layer, n):
                if getattr(layer, "_humming_candidate", False) and n > 0:
                    layer._muse_v4_state = {"portable_layer": layer}
                    return True
                return original(layer, n)

            MuseSpark1xFp4Experts._use_muse_v4 = use_v9


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in (ADAPTER, MODEL):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            spec.loader = Loader(spec.loader)
            return spec


sys.meta_path.insert(0, Finder())
