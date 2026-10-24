"""Exact-shape portable ABIs for the receipt-gated serving integration.

No padding or new weights. Production dispatch thresholds remain in force;
full-size expert arithmetic is checked independently by test_serving_v9_dispatch.
"""
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
exec(compile((ROOT / 'serving/harness/pyhook/sitecustomize.py').read_text(),
             'harness_hook', 'exec'))
AR = 'sglang.kernels.ops.muse_spark_v12.all_reduce_norm_residual'
ADAPTER = 'sglang.srt.layers.muse_spark_serving'
DRAFT = 'sglang.srt.models.dflash'
QUANT = 'sglang.srt.layers.quantization.muse_spark_nvfp4_moe'


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.original.exec_module(module)
        import torch
        if module.__name__ == AR:
            from sglang.srt.distributed import get_tp_group
            from sglang.kernels.ops.muse_spark_v12.norm_residual import fused_norm_residual

            def portable(comm, branch, stream, pre_weight, alpha, beta,
                         post_eps, pre_eps, *, want_router=False):
                reduced = get_tp_group().all_reduce(branch)
                return fused_norm_residual(reduced, stream, post_eps, pre_eps,
                    pre_weight=pre_weight, alpha=alpha, beta=beta, want_router=want_router)

            module.all_reduce_norm_residual_supported = lambda g, m, h: (
                g.world_size > 1 and 0 < m <= 256 and h % 1024 == 0)
            module.all_reduce_norm_residual = portable
        elif module.__name__ == ADAPTER:
            import torch.nn.functional as F

            class Projection:
                def __init__(self, weight, kind, **kw):
                    self.weight = weight.detach()

                def __call__(self, x):
                    return F.linear(x, self.weight)

            def dense(state, x, scores, ids):
                layer = state['portable_layer']
                rows, local = layer._marlin_expert_outputs(x, scores, ids)
                mapping = torch.arange(ids.numel(), device=ids.device,
                                       dtype=torch.int32).view_as(ids)
                return rows, torch.where(local >= 0, mapping, -1)

            module.MuseProjection = Projection
            module.projection_supported = lambda x, w, kind: (
                x.is_cuda and x.dtype == w.dtype == torch.bfloat16
                and x.ndim == w.ndim == 2 and x.shape[1] == w.shape[1]
                and 0 < x.shape[0] <= 256)
            module.run_v9_dense = dense
        elif module.__name__ == DRAFT:
            from sglang.srt.runtime_context import get_parallel

            def supported(comm, *, num_tokens, n, device, dtype):
                return (get_parallel().tp_size > 1 and 0 < num_tokens <= 64
                        and dtype == torch.bfloat16)

            def gather(comm, local, out):
                out.copy_(get_parallel().tp_group.all_gather(local, dim=-1))
                return out

            module.all_gather_supported = supported
            module.projection_all_gather = gather
        elif module.__name__ == QUANT:
            cls = module.MuseSparkNvFp4FusedMoEMethod
            original = cls._process_weights_for_marlin

            def prepare(self, layer):
                result = original(self, layer)
                # Force the same verify dispatch using the tiny model's exact
                # Marlin arithmetic. Real 32x4096 bank/repack arithmetic has a
                # separate layer test; it cannot use the tiny H512 checkpoint.
                layer._muse_v4_state = {'portable_layer': layer}
                return result

            cls._process_weights_for_marlin = prepare
        print('SERVING_INTEGRATION portable ABI loaded:', module.__name__, flush=True)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in (AR, ADAPTER, DRAFT, QUANT):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            spec.loader = Loader(spec.loader)
            return spec


sys.meta_path.insert(0, Finder())
