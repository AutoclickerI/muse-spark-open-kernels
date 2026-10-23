from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from vendor import vendor
root=Path(__file__).resolve().parents[1]
s=(root/'cand_glue.py').read_text()
s+='\n\n# Humming2 wins verify-32 on H200 receipt 15912e7130f643ac.\n'+vendor(root.parent/'muse_moe_w4a16/cand_humming2.py','hm_')
s+='''

_prepare_before_hybrid = prepare
_experts_before_hybrid = experts_v4

def prepare(weights):
    state = _prepare_before_hybrid(weights)
    state["humming"] = hm_prepare({"w13": weights["w13"], "w13_sf": weights["w13_sf"],
          "w13_gs": weights["w13_gs"], "w2": weights["w2"], "w2_sf": weights["w2_sf"],
          "w2_gs": weights["w2_gs"], "post_norm_w": weights["post_expert_norm"]})
    return state


def experts_v4(x, wts, lids, state):
    if x.shape[0] >= 256:
        return hm_kernel_fn({"hidden_states": x, "topk_ids": lids, "topk_weights": wts,
            "config": {"local_start": 0, "local_count": 32, "n_experts": 256,
                       "d_hidden": 4096, "d_expert": 4096}}, state["humming"])
    return _experts_before_hybrid(x,wts,lids,state)
'''
(root/'cand_hybrid.py').write_text(s)
