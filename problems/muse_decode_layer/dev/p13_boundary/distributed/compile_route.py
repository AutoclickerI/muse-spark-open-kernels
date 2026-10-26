"""Compile the changed scalar-route Triton glue for Hopper without executing it."""
import importlib.util
import json
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
H=Path(__file__).resolve().parent
path=H/'out/boundary_A_packed.py';spec=importlib.util.spec_from_file_location('p13b_native',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sig=dict(PR='*fp32',BIAS='*fp32',PREW='*fp32',LSTART='*i32',X='*bf16',IDS='*i32',WTS='*fp32',LIDS='*i32',STATS='*fp32')
results=[]
for parts in (4,8):
 for splits in (1,3,4):
    constants=dict(S=splits,PS=128*4608,RS=4608,NPARTS=parts,REPLICAS=1,USE_PDL=True)
    kernel=triton.compile(ASTSource(m._p13b_route_fast,signature=sig,constexprs=constants),target=GPUTarget('cuda',90,32),options=dict(num_warps=4,num_ctas=1))
    assert kernel.asm['cubin'] and 'griddepcontrol' in kernel.asm['ptx']
    results.append(dict(parts=parts,splits=splits,shared=kernel.metadata.shared,cubin_bytes=len(kernel.asm['cubin']),compiled=True,executed=False))
(H/'boundary_triton_compile.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
