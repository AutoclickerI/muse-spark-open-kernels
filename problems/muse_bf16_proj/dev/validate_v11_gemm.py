"""Actual QKV-family enumeration, V11 group rejection and Hopper profiler glue."""
import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);args=ap.parse_args()
spec=importlib.util.spec_from_file_location('v11_gemm_validation',args.candidate)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class Portable:
    def options(self,m,n,k,nn,fp32,workspace):
        assert m>0 and n>0 and k==8192 and not fp32 and nn in (0,1)
        return torch.zeros((2,32),dtype=torch.uint8),2
    def run(self,x,w,out,workspace,codes,index,nn,fp32):
        assert 0<=index<len(codes) and not fp32
        assert w.shape==((x.shape[1],out.shape[1]) if nn else (out.shape[1],x.shape[1]))
        assert out.shape[0]==x.shape[0] and out.dtype==torch.bfloat16
        out.zero_()


class Matmul:
    def __getitem__(self,grid):
        def launch(X,W,LO,Y,R,PART,COUNT,M,N,K,PN,MIXED,DUAL,BM,BN,BK,SMS,STREAM_K,*,num_warps,num_stages):
            assert grid==(SMS,) and 0<SMS<=132 and X.shape==(M,K) and W.shape==(N,K)
            assert not MIXED and not DUAL and Y.shape==(M,N) and COUNT.dtype==torch.int32
            assert (BM,BN,BK,num_warps,num_stages) in c.PG_TILES
            Y.zero_()
        return launch


class Pack:
    def __getitem__(self,grid):
        def launch(A,Y,M,N,S0,S1,B,*,num_warps):
            assert A.shape==Y.shape==(M,N) and A.stride()==(S0,S1)
            assert B==32 and num_warps==4 and grid==(c.triton.cdiv(M,32),c.triton.cdiv(N,32))
            Y.zero_()
        return launch


# Construct real metadata once, then execute every emitted configuration with
# exact signature stand-ins and metadata tensors. Never invent a native method.
real_weight=torch.randn((2304,8192),device='cuda',dtype=torch.bfloat16)*.01
op=c.PrefillGemmPlan({'qkv_gate':real_weight},tokens=(16384,),budget_s=0)
op.weights={'qkv_gate':torch.empty((2304,8192),device='meta',dtype=torch.bfloat16)}
op.device=torch.device('meta');op.native=Portable();op.incumbent=None
op.props=SimpleNamespace(multi_processor_count=132)
op.workspace=torch.empty(1<<30,device='meta',dtype=torch.uint8)
op.partial=torch.empty(2*132*128*256,device='meta',dtype=torch.float32)
op.counts=torch.zeros(32768,device='meta',dtype=torch.int32)
families=Counter();variants=0
with patch.object(c,'_pg_matmul',Matmul()),patch.object(c,'_pg_pack_matrix',Pack()):
    for m in (753,3012,16384):
        x=torch.empty((m,8192),device='meta',dtype=torch.bfloat16)
        for config in op.variants(m,'qkv_gate'):
            y=op.run(x,'qkv_gate',config)
            assert y.shape==(m,2304) and y.dtype==torch.bfloat16
            families[config[0]]+=1;variants+=1
assert set(families)=={'stock','lt','persistent','qkv_chunk','qkv_pad','qkv_swap','qkv_swap_persistent'}

# The wrapper's failure path must retain the original prepared V11 object.
sentinel=object();stub={'sentinel':sentinel}
with patch.object(c,'_pg_v11_prepare',return_value=stub), \
     patch.object(c,'prefill_gemm_prepare',side_effect=AttributeError('injected block ABI failure')), \
     patch.object(c,'_pg_profile_v11'):
    output=io.StringIO()
    with contextlib.redirect_stdout(output):retained=c.prepare({})
assert retained is stub and retained['sentinel'] is sentinel and retained['pg'] is None
assert 'retained=v11 AttributeError' in output.getvalue()

# A rejected alternative must not remove the real concatenated V11 router and
# its original 512-wide consumer. Actual CUDA graphs run the incumbent.
state=dict(pooler=torch.randn((4096,8192),device='cuda',dtype=torch.bfloat16)*.01,
           router_cat=torch.randn((512,8192),device='cuda',dtype=torch.bfloat16)*.01,
           router_bias=torch.randn((256,),device='cuda'),pg_groups={})
class Broken:
    def __call__(self,x,role):
        if role=='router':raise AttributeError('injected router ABI')
        raise TypeError('injected mixed tuple ABI')
state['pg']=Broken();lines=[]
c._pg_tune_v11_groups(state,time.monotonic()+30,lines)
assert len(state['pg_groups'])==3 and all(v==('v11',) for v in state['pg_groups'].values())
assert sum('AttributeError' in line for line in lines)==6 and sum('TypeError' in line for line in lines)==6
assert c._PgGraph.active==0

# Force the Hopper-only Python profiler branch using portable BF16 GEMMs.
small={name:torch.randn((n,128),device='cuda',dtype=torch.bfloat16) for name,n in
       [('qkv',2304),('o',8192),('pooler',4096),('unpooler',8192),('router_cat',512)]}
profile=[]
with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)):
    c._pg_profile_v11(small,time.monotonic()+25,profile)
assert profile and all(line.startswith(('PGC_KERNEL','PGC_REJECT')) for line in profile)
print(json.dumps(dict(ok=True,source_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                     forced_qkv_configurations=variants,families=dict(families),
                     wrapper_failure_retains_v11=True,rejected_group_failures=12,
                     hopper_profiler_lines=profile,active_tuning_streams=c._PgGraph.active)),flush=True)
