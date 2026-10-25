"""Prove source-identical JIT sharing also shares the actual CUDA function handle."""
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
ns={'__name__':'pa_pairing_check'}
exec(compile((HERE/'shared_followup_generated.py.inc').read_text(),str(HERE/'shared_followup_generated.py.inc'),'exec'),ns)
a=ns['pa_namespace'](ns['PA_BASE'],'pairing_a')
d=ns['pa_namespace'](ns['PA_D1'],'pairing_d')
m=ns['pa_namespace'](ns['PA_META'],'pairing_m')
x=torch.zeros((1,1024),device='cuda',dtype=torch.bfloat16)
g=torch.zeros_like(x,dtype=torch.float32);y=torch.empty_like(x)
def launch(module):return module['_norm_gate_kernel'][(1,)](x,g,y,1e-5,HQ=16,D=64,USE_PDL=True,launch_pdl=True,num_warps=4)
ca,cd=launch(a),launch(d)
before=dict(source_equal=a['_norm_gate_kernel'].src==d['_norm_gate_kernel'].src,
    cache_key_equal=a['_norm_gate_kernel'].cache_key==d['_norm_gate_kernel'].cache_key,
    compiled_object_equal=ca is cd,device_function_equal=ca.function==cd.function)
assert before['source_equal'] and not before['device_function_equal']
reused_d=ns['pa_share_jit'](a,d);reused_m=ns['pa_share_jit'](a,m)
assert d['_norm_gate_kernel'] is a['_norm_gate_kernel']
assert m['_gb_kernel'] is a['_gb_kernel']
assert m['_boundary_late'] is not a['_boundary_late']
cd=launch(d);torch.cuda.synchronize()
after=dict(compiled_object_equal=ca is cd,device_function_equal=ca.function==cd.function)
assert all(after.values())
result=dict(before=before,after=after,shared_d=reused_d,shared_m=reused_m,
    changed_boundary_remains_private=True)
(HERE/'pairing_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS actual device function handle equality after sharing; changed kernels remain private')
