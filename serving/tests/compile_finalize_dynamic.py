"""Cross-compile runtime-M standalone and unchanged static AR for Hopper."""
from unittest.mock import patch

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from sglang.kernels.jit.utils import override_jit_cuda_arch
from sglang.kernels.ops.muse_spark_v12 import fi_trtllm_moe_finalize_with_rmsnorm_cutedsl as kernel

compile_native = cute.compile


def compile90(*args, **kwargs):
    kwargs['options'] = kwargs.get('options', '') + ' --gpu-arch sm_90a'
    return compile_native(*args, **kwargs)


with override_jit_cuda_arch(9, 0, suffix='a'), patch.object(cute, 'compile', compile90):
    for h, m, cluster in ((1024, 7939, 1), (4096, 8, 8), (4096, 128, 4), (4096, 7939, 1)):
        kernel._get_compiled_kernel(cutlass.BFloat16, m, h, 8, h, cuda.CUstream(0),
                                    cluster_size=cluster, expert_weights_dtype=cutlass.Float32)
        print('sm90a dynamic finalize compiled', h, m, cluster, flush=True)
    for m in (8, 128, 256):
        kernel._get_compiled_kernel(cutlass.BFloat16, m, 4096, 8, 4096, cuda.CUstream(0),
            cluster_size=1, expert_weights_dtype=cutlass.Float32,
            ar_world_size=8, ar_controller_count=32, ar_multicast=True)
        print('sm90a static AR compiled', m, flush=True)
