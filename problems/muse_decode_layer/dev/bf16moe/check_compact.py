"""Validate the new engine independently of the full-layer candidate, GPU0 only."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch
import deep_gemm

ROOT=Path(__file__).resolve().parents[4]
DIR=ROOT/'problems/muse_decode_layer'
sys.path.insert(0,str(DIR))
import reference as ref
import moe_tune

spec=importlib.util.spec_from_file_location('bf_check_base',DIR/'cand_r3_v9.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
helper=Path(__file__).parent/'hybrid.py'
exec(compile(helper.read_text(),str(helper),'exec'),base.__dict__)


def oracle(data,w):
    # Independently expand each touched expert using the public NVFP4 math.
    x=data['hidden_states'].float();ids=data['topk_ids'];tw=data['topk_weights']
    result=torch.zeros_like(x,dtype=torch.float32)
    for e in ids[(ids>=0)&(ids<32)].unique().tolist():
        t,s=(ids==e).nonzero(as_tuple=True)
        w13=ref.dequantize(w['w13'][e],w['w13_sf'][e],w['w13_gs'][e])
        h=x[t]@w13.T;g,u=h.chunk(2,-1)
        inter=g*torch.sigmoid(g)*u
        w2=ref.dequantize(w['w2'][e],w['w2_sf'][e],w['w2_gs'][e])
        y=inter@w2.T
        y=y*(1+w['post_expert_norm'].float())
        y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-8)
        result.index_add_(0,t,y*tw[t,s,None])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--json',default='runs/screen/bf16-compact.json')
    args=parser.parse_args()
    w=ref.generate_weights(5353)
    native=base.v9_prepare({**w,'post_norm_w':w['post_expert_norm']})
    bank=base._bf_make_bank(w);bank['dg']=deep_gemm;bank['expected']=96
    results=[]
    # Exhaustive code x finite positive E4M3 scale products must round exactly.
    codes=torch.arange(256,device='cuda',dtype=torch.uint8).view(1,1,256).expand(1,256,256).contiguous()
    scales=torch.arange(256,device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn)
    scales=torch.where(torch.isfinite(scales.float()),scales.float(),0.).to(torch.float8_e4m3fn)
    scales=scales.view(1,256,1).expand(1,256,32).contiguous()
    out=torch.empty((1,256,512),device='cuda',dtype=torch.bfloat16)
    base._bf_unpack[(1,32)](codes,scales.view(torch.uint8),out,N=256,K=512,B=4096)
    want=ref.dequantize(codes[0],scales[0],torch.tensor(1.,device='cuda'))
    assert torch.equal(out[0].float(),want), 'block-product expansion lost information'
    print('BLOCK_PRODUCTS exhaustive finite FP8 scales x all nibbles exact',flush=True)
    engines=('bmm_tn_f32','bmm_nn_f32','bmm_tn_bf16')
    for m in (32,128,256):
        histograms=[]
        for seed in (6711,14630,22549):
            fixture=moe_tune.make_fixture(w,m,seed)
            ids=fixture['topk_ids']
            histograms.append(torch.bincount(ids[ids>=0],minlength=32).cpu().tolist())
        assert base._bf_bmm_plan(bank,m,16,histograms,compact=True)
        for seed in (6711,14630,22549):
            data=moe_tune.make_fixture(w,m,seed)
            expected=oracle(data,w)
            for engine in engines:
                for share in (0.,.5):
                    policy=(engine,16 if engine.startswith('bmm_') else 32,share)
                    got=base._bf_hybrid(data,native,bank,policy).float()
                    torch.cuda.synchronize()
                    rel=float((got-expected).norm()/expected.norm().clamp_min(1e-15))
                    rows=(got-expected).norm(dim=-1)/expected.norm(dim=-1).clamp_min(1e-15)
                    row=float(rows.max())
                    assert torch.isfinite(got).all() and rel<.01 and row<.02,(m,seed,policy,rel,row)
                    result=dict(m=m,seed=seed,engine=engine,share=share,rel_l2=rel,worst_row=row)
                    results.append(result);print(result,flush=True)
                    if seed==6711:
                        for _ in range(2):base._bf_hybrid(data,native,bank,policy)
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):out=base._bf_hybrid(data,native,bank,policy)
                        for fresh_seed in (60001,60002):
                            fresh=moe_tune.make_fixture(w,m,fresh_seed)
                            for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(fresh[key])
                            graph.replay();torch.cuda.synchronize()
                            expected_graph=oracle(fresh,w)
                            rel_graph=float((out.float()-expected_graph).norm()/expected_graph.norm().clamp_min(1e-15))
                            assert torch.isfinite(out).all() and rel_graph<.01,('graph',m,policy,rel_graph)
                        graph.reset()
                        # Restore the eager test fixture after fresh replay.
                        fresh=moe_tune.make_fixture(w,m,seed)
                        for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(fresh[key])
        # Shape-preserving corner cases, including changing the hot expert set.
        for mode in ('remote','onehot','eight_hot','uniform'):
            data=moe_tune.make_fixture(w,m,9371)
            ids=data['topk_ids']
            if mode=='remote':ids.fill_(-1)
            elif mode=='onehot':
                ids.fill_(-1);ids[:,0]=17
            elif mode=='eight_hot':ids.copy_(torch.arange(8,device='cuda').expand(m,8).int())
            else:ids.copy_((torch.arange(m*8,device='cuda')%32).view(m,8).int())
            expected=oracle(data,w)
            for engine in engines:
                got=base._bf_hybrid(data,native,bank,(engine,16 if engine.startswith('bmm_') else 32,.25)).float()
                torch.cuda.synchronize()
                rel=float((got-expected).norm()/expected.norm().clamp_min(1e-15))
                assert torch.isfinite(got).all() and rel<.01,(m,mode,engine,rel)
                results.append(dict(m=m,mode=mode,engine=engine,rel_l2=rel))
    Path(args.json).write_text(json.dumps(results,indent=2)+'\n')
    print('COMPACT PASS',len(results),'eager cases; 36 fresh graph replays; overflow routing extremes included',flush=True)

if __name__=='__main__':main()
