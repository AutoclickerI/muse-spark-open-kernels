"""Compile Hopper projection and exact-scale Humming kernels without executing."""

import dataclasses

from sglang.kernels.jit.utils import override_jit_cuda_arch
from sglang.kernels.ops.muse_spark_v12.projection_hopper import _module
from sglang.srt.layers.muse_spark_humming import _imports


def main():
    with override_jit_cuda_arch(9, 0, suffix="a"):
        _module()
    print("projection sm90a compile passed", flush=True)
    _imports()
    # Generate actual SM90 WGMMA code without loading its cubin on SM120.
    from unittest.mock import patch

    from humming import dtypes
    from humming.config import (
        GemmType,
        LayerConfig,
        MmaType,
        WeightScale2Type,
        WeightScaleType,
    )
    from humming.jit.compiler import NVRTCCompiler
    from humming.kernel.humming import HummingKernel
    from humming.tune.sm90 import Sm90Heuristics

    def init90(self):
        self.sm_version = 90
        self.sm_version_str = "90a"

    def compile_only(self):
        self.kernel_filename = NVRTCCompiler.compile(
            self.code,
            sm_version="90a",
            kernel_expr=self.kernel_expr,
            disable_fast_math=self.disable_fast_math,
            postprocess_cubin=self.postprocess_cubin,
        )

    for count, routes in ((64, 512), (32, 256)):
        layer = LayerConfig(
            shape_n=4096,
            shape_k=4096,
            num_experts=count,
            b_dtype=dtypes.float4e2m1,
            a_dtype=dtypes.bfloat16,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.float8e4m3,
            weight_scale_group_size=16,
            weight_scale_type=WeightScaleType.GROUP,
            weight_scale_2_type=WeightScale2Type.TENSOR,
            mma_type=MmaType.WGMMA,
        )
        cfg = Sm90Heuristics.get_config(layer, routes, gemm_type=GemmType.INDEXED)
        cfg.update(
            use_tma=True,
            use_tma_a=False,
            use_tma_as=False,
            use_tma_c=False,
            use_mbarrier=True,
            use_warp_spec=True,
            use_stream_k=False,
        )
        cfg.pop("use_f16_accum", None)
        values = {
            f.name: getattr(layer, f.name) for f in dataclasses.fields(layer) if f.init
        }
        with (
            patch.object(HummingKernel, "init_sm_version", init90),
            patch.object(HummingKernel, "prepare", compile_only),
        ):
            kernel = HummingKernel(
                **values, gemm_type=GemmType.INDEXED, use_f16_accum=False, **cfg
            )
        print(
            "Humming sm90a compile passed",
            count,
            cfg,
            kernel.kernel_filename,
            flush=True,
        )


if __name__ == "__main__":
    main()
