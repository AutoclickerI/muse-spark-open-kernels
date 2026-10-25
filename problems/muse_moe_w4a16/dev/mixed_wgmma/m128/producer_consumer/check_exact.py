"""Exhaust finite E4M3 scales and all E2M1 codes through the actual producer DQ."""
import importlib.util
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    spec=importlib.util.spec_from_file_location('pc_exact',ROOT/'prof/lab/moe_m128_producer_consumer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    codes=torch.arange(4096,device='cuda',dtype=torch.uint8).reshape(1,1,-1).expand(1,64,4096)&15
    q=(codes[...,::2]|(codes[...,1::2]<<4)).contiguous()
    raw=torch.ones((1,64,256),device='cuda',dtype=torch.uint8)*56
    domain=torch.arange(256,device='cuda',dtype=torch.uint8);domain[127]=0;domain[255]=0
    raw[0,:,:4]=domain.reshape(64,4);sf=raw.view(torch.float8_e4m3fn)
    code,neg,zero=m.m128_hh_encode_scales(sf)
    folded=q.reshape(1,64,256,8)^neg.to(torch.uint8)[...,None]*136
    folded=torch.where(zero[...,None],torch.zeros_like(folded),folded).reshape_as(q)
    packed=m.m128_hh_mw_repack_planes(folded)
    scales=code.reshape(1,4,2,8,128,2).permute(0,1,4,3,5,2).contiguous()
    out=torch.empty((64,64),device='cuda',dtype=torch.bfloat16)
    mod=m.pc_module();p=[v.data_ptr() for v in (packed,scales,out)]
    def call():mod.pc_probe(p,torch.cuda.current_stream().cuda_stream)
    call();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    graph.replay();torch.cuda.synchronize()
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device='cuda')
    c=codes[0,:,:64];want=(lut[(c&7).long()]*torch.where(c&8!=0,-1.,1.)*sf[0,:,:4].float().repeat_interleave(16,-1)/256.).bfloat16()
    # The admitted sign/zero fold canonicalizes signed zero. Every nonzero bit
    # pattern must agree; canonicalize zeros on both sides before bit compare.
    got=torch.where(out==0,torch.zeros_like(out),out);ref=torch.where(want==0,torch.zeros_like(want),want)
    assert torch.equal(got.view(torch.int16),ref.view(torch.int16)),float((got.float()-ref.float()).abs().max())
    report=dict(scale_patterns=254,FP4_codes=16,exact_finite_products=4064,checked_values=4096,actual_dq_and_mapping=True,graph_capture=True,zero_sign_canonicalized=True)
    (HERE/'exact.json').write_text(json.dumps(report,indent=2)+'\n');print('PC_EXACT_PASS',report,flush=True)


if __name__=='__main__':main()
