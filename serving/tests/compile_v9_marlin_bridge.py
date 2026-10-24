"""Compile both lossless bridge shapes for sm90a without execution."""

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'prof/lab'))
from v9_marlin_bridge import _weights_tile, _scales

for gated,n in ((False,4096),(True,8192)):
    for kernel,signature,const,warps in (
        (_weights_tile,{'P':'*i32','O':'*i32'},{'N':n,'GATED':gated},8),
        (_scales,{'P':'*u8','O':'*u8'},{'N':n,'GATED':gated,'B':256},4),
    ):
        compiled=triton.compile(ASTSource(kernel,signature=signature,constexprs=const),
                               target=GPUTarget('cuda',90,32),options={'num_warps':warps})
        assert '.target sm_90a' in compiled.asm['ptx']
        print('sm90a',kernel.__name__,gated,'shared',compiled.metadata.shared,flush=True)
