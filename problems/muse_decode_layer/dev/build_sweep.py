from pathlib import Path
from vendor import vendor
p=Path(__file__).resolve().parents[1]
s=(p/'cand_chain6.py').read_text()
s+='\n\n# Diagnostic MoE v4 control: identical lossless packed weight layout.\n'+vendor(p.parent/'muse_moe_w4a16/cand_v4.py','diag4_')
s+='\n\n_diag_kernel_chain=kernel_fn\n'+(p/'dev/epilogue_runtime.py').read_text()+'\n\nkernel_fn=_diag_kernel_chain\n'
s+='\n\n'+(p/'dev/sweep_runtime.py').read_text()
(p/'cand_layer_sweep.py').write_text(s)
