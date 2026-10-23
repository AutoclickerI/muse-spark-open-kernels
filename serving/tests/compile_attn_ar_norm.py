"""Compile the TP8 Hopper fusion variants without launching Hopper cubins."""
from sglang.kernels.jit.utils import override_jit_cuda_arch
from sglang.kernels.ops.muse_spark_v12.all_reduce_norm_residual import _jit_module


def main():
    with override_jit_cuda_arch(9, 0, suffix="a"):
        for cluster, threads in ((8, 128), (4, 256), (1, 256)):
            for multicast in (False, True):
                for router in (False, True):
                    for two_shot in (False, True):
                        _jit_module(8192, 8, threads, cluster, multicast, router, two_shot)
    print("24 sm_90a attention-AR/norm variants compiled; no Hopper execution")


if __name__ == "__main__":
    main()
