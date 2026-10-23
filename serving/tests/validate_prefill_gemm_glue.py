"""Strict portable stand-in for the SM90 loader/Lt branches and layout release."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
from sglang.srt.layers import muse_prefill_gemm as p


class Portable:
    def options(self, m, n, k, nn, fp32, workspace):
        assert nn in (0,1) and isinstance(fp32,bool) and workspace >= 1 << 30
        return torch.zeros((1,32),dtype=torch.uint8),1

    def run(self, x, w, out, workspace, codes, index, nn, fp32):
        assert index == 0 and len(codes) == 1
        assert out.dtype == (torch.float32 if fp32 else torch.bfloat16)
        value = torch.mm(x,w if nn else w.T,**({'out_dtype':torch.float32} if fp32 else {}))
        out.copy_(value)


if __name__ == '__main__':
    torch.manual_seed(207)
    seen=[]
    def compile_standin(**kwargs):
        assert os.environ['TORCH_CUDA_ARCH_LIST']=='9.0a'
        assert kwargs['is_python_module'] is False
        seen.append(kwargs['name'])
        return 'portable-abi'
    original=p._pg_lib
    p._pg_lib=None
    with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)), \
         patch('torch.utils.cpp_extension.load_inline',side_effect=compile_standin), \
         patch.object(p,'_PgNative',side_effect=lambda path:Portable()):
        assert isinstance(p._pg_native(),Portable)
    weights={name:torch.randn(shape,device='cuda',dtype=torch.float32 if name=='router' else torch.bfloat16)*.01
             for name,shape in dict(qkv_gate=(2304,8192),o=(8192,1024),pooler=(4096,8192),
                                    unpooler=(8192,4096),router=(256,8192)).items()}
    gain=(torch.randn((4096,),device='cuda')*.01+1).bfloat16()
    op=p.PrefillGemmPlan(weights,tokens=(753,),budget_s=0,pool_norm_weight=gain)
    branches=0
    for role in (*weights,'pool_router','pool_router_norm'):
        k=weights['pooler' if role.startswith('pool_router') else role].shape[1]
        x=torch.randn((753,k),device='cuda',dtype=torch.bfloat16)
        for nn in (0,1):
            op._options(753,role,nn)
            with p._PgGraph(lambda:op.run(x,role,('lt',nn,0))) as graph:
                for _ in range(3):
                    x.normal_();graph.graph.replay()
                    assert p._pg_equal(graph.out,op._stock(x,role))
            branches+=1
    # Both mixed roles must use the same NN weight copy and option table.
    assert (753,'pool_router_norm',1) not in op.codes
    assert 'pool_router_norm' not in op.transposes
    op.plans[(753,'pool_router_norm')]=('lt',1,0)
    op.release_trials()
    assert op.bank is None and op.lt_bank is not None and set(op.transposes)=={'pool_router'}
    x=torch.randn((753,8192),device='cuda',dtype=torch.bfloat16)
    with p._PgGraph(lambda:op(x,'pool_router_norm')) as graph:
        for _ in range(3):
            x.normal_();graph.graph.replay();assert p._pg_equal(graph.out,op._stock(x,'pool_router_norm'))
    pooled=op._stock(x,'pooler')
    reference=torch.nn.functional.rms_norm(pooled,(4096,),gain,1e-5).bfloat16()
    assert p._pg_equal(graph.out[0],reference)
    assert p._PgGraph.active==0 and seen
    p._pg_lib=original
    report=dict(ok=True,forced_hopper_lt_branches=branches,fresh_replays=branches*3+3,
                bf16_effective_norm_matches_torch=True,shared_mixed_layout=True,
                released_layout_graph_safe=True,loader_target='9.0a',
                module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest())
    print(json.dumps(report))
