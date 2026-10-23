"""Same-ABI collective stand-in, confined to local serving tests."""

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
exec(
    compile(
        (ROOT / "serving/harness/pyhook/sitecustomize.py").read_text(),
        "harness_sitecustomize",
        "exec",
    )
)
NAME = "sglang.kernels.ops.muse_spark_v12.all_reduce_norm_residual"


class PatchLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.original.exec_module(module)
        from sglang.srt.distributed import get_tp_group
        from sglang.kernels.ops.muse_spark_v12.norm_residual import fused_norm_residual

        def supported(group, tokens, hidden):
            return group.world_size > 1 and 0 < tokens <= 256 and hidden % 1024 == 0

        def portable(
            comm,
            branch,
            stream,
            pre_weight,
            alpha,
            beta,
            post_eps,
            pre_eps,
            *,
            want_router=False,
        ):
            reduced = get_tp_group().all_reduce(branch)
            return fused_norm_residual(
                reduced,
                stream,
                post_eps,
                pre_eps,
                pre_weight=pre_weight,
                alpha=alpha,
                beta=beta,
                want_router=want_router,
            )

        module.all_reduce_norm_residual_supported = supported
        module.all_reduce_norm_residual = portable
        print("ATTN_AR_NORM portable NCCL ownership stand-in enabled", flush=True)


class PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == NAME:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            spec.loader = PatchLoader(spec.loader)
            return spec


sys.meta_path.insert(0, PatchFinder())
