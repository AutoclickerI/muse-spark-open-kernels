"""Compile Humming's SM90 variants without loading/running them on the local SM120 GPU."""
import json
import os
import sys

os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType, WeightScale2Type
from humming.kernel.humming import HummingKernel
from humming.tune.sm90 import Sm90Heuristics


class Offline(HummingKernel):
    def init_sm_version(self):
        self.sm_version = 90
        self.sm_version_str = "90a"

    def load_cubin(self):
        self.cubin_loaded = True
        self.kernel_id = -1


use_mma = len(sys.argv) > 1 and sys.argv[1] == "mma"
bucket = len(sys.argv) > 1 and sys.argv[1] in ("bucket", "bucketk")
bucketk = len(sys.argv) > 1 and sys.argv[1] == "bucketk"
seen = set()
for n in (8192, 4096):
    layer = LayerConfig(shape_n=n, shape_k=4096, num_experts=32,
                        a_dtype=dtypes.bfloat16, b_dtype=dtypes.float4e2m1, c_dtype=dtypes.bfloat16,
                        bs_dtype=dtypes.float8e4m3, weight_scale_group_size=16,
                        weight_scale_2_type=WeightScale2Type.TENSOR, mma_type=MmaType.MMA if use_mma else MmaType.WGMMA)
    params = json.loads(layer.to_str())
    for m in (1, 16, 32, 128, 256, 512):
        cfg = Sm90Heuristics.get_config(layer, m, gemm_type=GemmType.INDEXED)
        if use_mma:
            bm=(cfg["block_shape"][0]+15)//16*16
            cfg["block_shape"]=(bm,*cfg["block_shape"][1:])
            cfg["warp_shape"]=(bm,*cfg["warp_shape"][1:])
        else:
            if bucket:
                bm = 8 if m < 32 else (32 if m < 256 else 64)
                bn, bk = cfg["block_shape"][1:]
                wn, wk = cfg["warp_shape"][1:]
                if bm >= 32:
                    bn, bk = 128, min(bk, 128)
                    if bucketk:
                        bk, wk = 64, 32
                        cfg["num_stages"] = 8
                cfg["block_shape"] = (bm, bn, bk)
                cfg["warp_shape"] = (bm, wn, wk)
            cfg.update(use_tma=True, use_tma_a=False, use_tma_as=False, use_tma_c=False,
                       use_mbarrier=True, use_warp_spec=True)
        key = (n, json.dumps(cfg, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        kernel = Offline(**(params | {"gemm_type": GemmType.INDEXED} | cfg))
        print(n, m, cfg, "compiled", kernel.kernel_filename, flush=True)
