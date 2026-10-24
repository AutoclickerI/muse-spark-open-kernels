"""Direct reader parity: both kernels consume one stock packed weight bank."""

import json
import torch

from test_muse_moe_v4_layer import _weights,_inputs,_marlin_state,_stock
from sglang.srt.layers.muse_spark_expert_bank import make_marlin_reader_state
from sglang.srt.layers.muse_spark_serving import run_v9_dense
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum


def main():
    device=torch.device('cuda');w=_weights(device)
    # Include every finite E4M3 magnitude to test stock scale-byte inversion.
    for name in ('w13_sf','w2_sf'):
        raw=w[name].view(torch.uint8)
        raw.copy_((torch.arange(raw.numel(),device=device)%127).to(torch.uint8).view_as(raw))
    layer,locks=_marlin_state(w,device)
    before={name:getattr(layer,name).clone() for name in (
        'w13_weight','w2_weight','w13_weight_scale','w2_weight_scale')}
    state=make_marlin_reader_state(layer,w['post_norm_w'])
    from sglang.srt.layers.muse_spark_moe import prepare
    native=prepare(w,local_start=0)
    native.update(gs13_gate=state['gs13_gate'],gs13_up=state['gs13_up'],gs2=state['gs2'])
    for name,expected in before.items():
        assert torch.equal(getattr(layer,name).view(torch.uint8),expected.view(torch.uint8)),name
    for prefix in ('w13','w2'):
        assert getattr(layer,prefix+'_weight').untyped_storage().data_ptr()==getattr(layer,prefix+'_weight_scale').untyped_storage().data_ptr()
    norm=(1+w['post_norm_w'].float()).bfloat16()

    def call(x,scores,ids):
        dense,mapping=run_v9_dense(state,x,scores,ids)
        return fused_post_expert_norm_weighted_sum(dense,torch.where(mapping>=0,scores,0),
                                                  norm_weight=norm,eps=1e-8,weight_first=True)

    for m in (1,8,32,128,256):
        x,ids,scores=_inputs(m,device,83000+m)
        expected=_stock(layer,locks,w,x,ids,scores)
        eager=call(x,scores,ids)
        direct_y,direct_map=run_v9_dense(state,x,scores,ids)
        old_y,old_map=run_v9_dense(native,x,scores,ids)
        assert torch.equal(direct_y,old_y) and torch.equal(direct_map,old_map),(m,'native V9 byte equality')
        rel=float((eager.float()-expected.float()).norm()/expected.float().norm())
        assert rel<=.015,(m,rel)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): captured=call(x,scores,ids)
        maxself=0.0
        for _ in range(3):
            g.replay()
            current=call(x,scores,ids)
            maxself=max(maxself,float((captured.float()-current.float()).abs().max()))
            assert torch.equal(captured,current),(m,'graph self drift')
            x.mul_(.75)
        g.reset()
        print(json.dumps(dict(m=m,relative_l2=rel,maxdiff=float((eager.float()-expected.float()).abs().max()),graph_self_maxdiff=maxself)),flush=True)
    ids.fill_(200)
    assert torch.count_nonzero(call(x,scores,ids))==0
    print('SINGLE STOCK BANK/DIRECT READER/REMOTE ZERO PASS',flush=True)


if __name__=='__main__':main()
