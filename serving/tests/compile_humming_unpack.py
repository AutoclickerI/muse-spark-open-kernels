"""Compile both canonical-to-NVFP4 adapters for Hopper without running them."""

import triton
from sglang.srt.layers.muse_spark_humming import _unpack
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

for gated, rows in ((True, 8192), (False, 4096)):
    source = ASTSource(
        _unpack,
        signature={"C": "*i32", "W": "*u8", "S": "*fp8e4nv", "BITS": "*i32"},
        constexprs={"ROWS": rows, "GATED": gated, "BLOCK": 256},
    )
    compiled = triton.compile(source, target=GPUTarget("cuda", 90, 32))
    print("Hopper canonical unpack compiled", gated, compiled.metadata.name, flush=True)
