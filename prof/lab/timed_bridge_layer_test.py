"""Rejected timed-bridge diagnostic; run with frozen shared-bank-r03 source.

The per-prefill bandwidth lower bound exceeds the c32 break-even, so this is
retained evidence, not the production direct-reader admission test.
"""

import json
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm import FusedPostExpertNormWeightedSumTRTLLM
from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.muse_spark_expert_bank import materialize_marlin
from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.layers.quantization.muse_spark_nvfp4_moe import MuseSparkNvFp4FusedMoEMethod
from sglang.srt.runtime_context import get_context
from test_muse_moe_v4_layer import _weights, _inputs, _marlin_state, _stock


def main():
    device = torch.device('cuda')
    w = _weights(device)
    method = MuseSparkNvFp4FusedMoEMethod.__new__(MuseSparkNvFp4FusedMoEMethod)
    method.quant_config = SimpleNamespace(group_size=16, get_name=lambda: 'modelopt_fp4')
    method._moe_runner_backend = MoeRunnerBackend.MARLIN
    owner = torch.nn.Module()
    owner.quant_config, owner.params_dtype = method.quant_config, torch.bfloat16
    owner.intermediate_size_per_partition = owner.hidden_size = 4096
    owner.num_local_experts, owner.global_num_experts = 32, 256
    owner.local_expert_offset = 0
    owner.moe_runner_config = SimpleNamespace(is_gated=True)
    with torch.device('cuda'):
        method.create_weights(owner, 32, 4096, 4096, torch.bfloat16)
    for name, key in [('gate_up_proj','w13'),('down_proj','w2'),
                      ('gate_up_proj_weight_scale','w13_sf'),('down_proj_weight_scale','w2_sf'),
                      ('down_proj_weight_scale_2','w2_gs')]:
        getattr(owner,name).data.copy_(w[key])
    owner.gate_up_proj_weight_scale_2.data.copy_(torch.stack((w['w13_gs_gate'],w['w13_gs_up']),1))
    owner.gate_up_proj_input_scale.data.fill_(1)
    owner.down_proj_input_scale.data.fill_(1)
    norm = torch.nn.Module()
    norm.weight = torch.nn.Parameter(w['post_norm_w'], requires_grad=False)
    owner.post_expert_norm = norm
    with get_context().override_server_args(moe_runner_backend='marlin'), patch(
        'sglang.srt.layers.muse_spark_expert_bank.shared_bank_supported', return_value=True
    ):
        method.process_weights_after_loading(owner)
    state = owner._muse_v4_state
    assert state['bank_layout']=='v9'
    assert state['w13p'].data_ptr()==owner.w13_weight.data_ptr()
    assert state['w2p'].data_ptr()==owner.w2_weight.data_ptr()
    assert owner.w13_weight.dtype==owner.w2_weight.dtype==torch.uint8
    assert owner.w13_weight.nbytes+owner.w2_weight.nbytes==905969664
    assert not hasattr(owner,'gate_up_proj') and not hasattr(owner,'down_proj')
    assert not hasattr(owner,'w13_weight_scale') and not hasattr(owner,'w2_weight_scale')
    ref, locks = _marlin_state(w,device)
    workspace = materialize_marlin(state)
    for name in ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale'):
        assert torch.equal(getattr(workspace,name).view(torch.uint8),getattr(ref,name).view(torch.uint8)),name
    for name in ('w13_weight_scale_2','w2_weight_scale_2'):
        assert torch.equal(getattr(owner,name),getattr(ref,name)),name
    print('ONE RESIDENT BANK; ALL MARLIN BYTES/GLOBAL SCALES EXACT',flush=True)
    owner._expert_map=torch.full((256,),-1,device=device,dtype=torch.int32)
    owner._expert_map[:32]=torch.arange(32,device=device)
    owner._post_expert_norm=norm
    owner._post_expert_norm_weight=(1+w['post_norm_w'].float()).bfloat16()
    owner._post_expert_norm_eps=1e-8
    owner._post_expert_norm_weight_first=True
    owner._fused_finalize_op=FusedPostExpertNormWeightedSumTRTLLM(topk=8,hidden_dim=4096,eps=1e-8,weight_first=True)
    for name in ('_use_muse_v4','_marlin_expert_outputs','_finalize_dense_expert_outputs'):
        setattr(owner,name,MethodType(getattr(MuseSpark1xFp4Experts,name),owner))

    def call(x,scores,ids):
        return MuseSpark1xFp4Experts._forward_marlin(owner,x,scores,ids,norm)

    for m in (1,8,32,128,256,257,511,512,513,752,753,1024,2048):
        x,ids,scores=_inputs(m,device,81000+m)
        expected=_stock(ref,locks,w,x,ids,scores)
        eager=call(x,scores,ids)
        delta=(eager.float()-expected.float()).abs()
        rel=float(delta.norm()/expected.float().norm())
        assert rel<=.015,(m,rel,delta.max().item())
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): captured=call(x,scores,ids)
        selfdiff=0.0
        for _ in range(3):
            graph.replay()
            current=call(x,scores,ids)
            selfdiff=max(selfdiff,float((captured.float()-current.float()).abs().max()))
            torch.testing.assert_close(captured,_stock(ref,locks,w,x,ids,scores),rtol=.015,atol=.015)
            x.mul_(.75)
        graph.reset()
        print(json.dumps(dict(m=m,producer='V9' if m<=256 else 'Marlin+timed_bridge',
                              maxdiff=delta.max().item(),relative_l2=rel,graph_self_maxdiff=selfdiff)),flush=True)
    ids.fill_(200)
    assert torch.count_nonzero(call(x,scores,ids))==0
    # Different layer graphs reuse the same lease but must rewrite its bytes.
    other=dict(state,w13p=state['w13p'].clone())
    other['w13p'][:,:,:32768].bitwise_xor_(0xA5)
    first=workspace.w13_weight.clone()
    assert materialize_marlin(other) is workspace
    assert not torch.equal(first,workspace.w13_weight)
    assert materialize_marlin(state) is workspace
    assert torch.equal(first,workspace.w13_weight)
    # Captured graphs for different layers share a lease, not its contents.
    a,b=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
    with torch.cuda.graph(a): saved_a=materialize_marlin(state).w13_weight.clone()
    with torch.cuda.graph(b): saved_b=materialize_marlin(other).w13_weight.clone()
    for graph in (b,a,b,a): graph.replay()
    assert torch.equal(saved_a,first) and not torch.equal(saved_b,first)
    a.reset();b.reset()
    # A second EP shard uses global IDs 64..95 with the same local bank.
    state['local_start']=64
    owner._expert_map.fill_(-1)
    owner._expert_map[64:96]=torch.arange(32,device=device)
    for m in (8,256,512):
        x,ids,scores=_inputs(m,device,92000+m)
        expected=_stock(ref,locks,w,x,ids,scores)
        shifted=(ids+64)%256
        eager=call(x,scores,shifted)
        torch.testing.assert_close(eager,expected,rtol=.015,atol=.015)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): captured=call(x,scores,shifted)
        graph.replay()
        torch.testing.assert_close(captured,expected,rtol=.015,atol=.015)
        graph.reset()
        print(json.dumps(dict(m=m,local_start=64,shifted_EP_graph=True)),flush=True)
    print('TIMED REWRITE/SHARED LEASE/REMOTE ZERO PASS',flush=True)


if __name__=='__main__':
    main()
