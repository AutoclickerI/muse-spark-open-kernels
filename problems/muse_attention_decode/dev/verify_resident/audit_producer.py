"""Compile S14's unchanged verify16 vdsk producer for sm90; no GPU timing or execution."""
import ast,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')]
import torch,triton
from triton.backends.compiler import GPUTarget
import local_eval
m=local_eval.load_module('producer_s14',R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py')
X=torch.empty((128,8192),device='cuda',dtype=torch.bfloat16);W=torch.empty((2304,8192),device='cuda',dtype=torch.bfloat16);P=torch.empty((3,128,2304),device='cuda')
xd=m._VDSK_TD.from_tensor(X,[128,64]);wd=m._VDSK_TD.from_tensor(W,[64,64]);old=triton.runtime.driver.active.get_current_target
triton.runtime.driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
try:k=m._vdsk_gemm.warmup(X,W,P,xd,wd,128,2304,8192,3,128,64,64,8,True,grid=(36,3),num_warps=4,num_stages=8)
finally:triton.runtime.driver.active.get_current_target=old
meta=k.metadata
record=dict(source='S14 f1c77ce6 _vdsk_gemm',target='sm90',grid=[36,3],shared=meta.shared,num_warps=meta.num_warps,
            ptx_sha256=hashlib.sha256(k.asm['ptx'].encode()).hexdigest(),has_programmatic_release='griddepcontrol.launch_dependents' in k.asm['ptx'],no_execution=True)
(H/'out/producer_admission.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
