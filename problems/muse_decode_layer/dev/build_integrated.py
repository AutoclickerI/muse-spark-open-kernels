from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from vendor import vendor
root=Path(__file__).resolve().parents[1]
s=(root/'cand_v3.py').read_text()
s=s.replace('"bank": _marlin_bank(weights),','"moe_bank": mv4_prepare({**weights, "post_norm_w": weights["post_expert_norm"]}),')
s=s.replace('moe = experts_marlin(x, wts, lids, state)', 'moe = experts_v4(x, wts, lids, state)')
s=s.replace('while c < 16 and 2 * c * tokens <= 128:', 'while c < 16 and tokens < 32 and 2 * c * tokens <= 128:')
s+='\n\n# Losslessly vendored MoE v4 (H200 check 84c97d9483b34573).\n'+vendor(root.parent/'muse_moe_w4a16/cand_v4.py','mv4_')
s+='''

# LIDS already encodes the current input's device-resident local_start. Passing
# these local IDs with origin zero avoids a host read and also handles wrapping.
def experts_v4(x, wts, lids, state):
    return mv4_kernel_fn({"hidden_states": x, "topk_ids": lids,
                          "topk_weights": wts,
                          "config": {"local_count": 32, "d_hidden": 4096,
                                     "d_expert": 4096, "local_start": 0}}, state["moe_bank"])
'''
(root/'cand_moe4.py').write_text(s)
