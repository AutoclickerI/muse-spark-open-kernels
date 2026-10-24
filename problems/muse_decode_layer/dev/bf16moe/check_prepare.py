"""Exercise production layout reclamation after a BMM-only selection."""
import importlib.util
from pathlib import Path
import torch
DIR=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('bf16_candidate',DIR/'cand_r3_bf16moe.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from check_engine import ref,moe_tune,oracle
w=ref.generate_weights(5353)
# The numerical checks and real kernel calls remain intact; inject just the
# selector result to reach the BMM-only reclamation branch on a portable GPU.
c._bf_is_hopper=lambda:True
c._bf_base_prepare=lambda weights:{'moe_bank':c.v9_prepare({**weights,'post_norm_w':weights['post_expert_norm']})}

def selected(weights,native,bank,deadline):
    choices={128:('bmm_tn_f32',32,.25,64,0,1),256:('bmm_nn_f32',32,.5,128,0,1)}
    for m,p in choices.items():
        hist=[]
        for j in range(3):
            d=moe_tune.make_fixture(weights,m,5353+m+7919*j);ids=d['topk_ids']
            hist.append(torch.bincount(ids[ids>=0],minlength=32).cpu().tolist())
        assert c._bf_bmm_plan(bank,m,p[1],hist,compact=True)
    return choices,{}

c._bf_tune=selected
st=c.prepare(w)
bank=st['bf16_moe_bank']
assert 'bmm_cache' not in bank
assert bank['w13'].shape[0]<32
assert len(bank['bmm'])==2
for m in (128,256):
    for seed in (71921,71922,71923):
        d=moe_tune.make_fixture(w,m,seed)
        want=oracle(d,w)
        got=c.experts_v9(d['hidden_states'],d['topk_weights'],d['topk_ids'],st)
        rel,row=c._bf_errors(got,want)
        assert rel<.01 and row<.05,(m,seed,rel,row)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=c.experts_v9(d['hidden_states'],d['topk_weights'],d['topk_ids'],st)
    for seed in (81611,81612,81613):
        fresh=moe_tune.make_fixture(w,m,seed)
        for key in ('hidden_states','topk_weights','topk_ids'):d[key].copy_(fresh[key])
        g.replay();torch.cuda.synchronize()
        rel,row=c._bf_errors(out,oracle(fresh,w))
        assert rel<.01 and row<.05,(m,seed,rel,row)
    g.reset()
print('PREPARE_RECLAIM PASS: two BMM shapes, six eager + six fresh graph cases, bytes',st['bf16_moe_bytes'])
