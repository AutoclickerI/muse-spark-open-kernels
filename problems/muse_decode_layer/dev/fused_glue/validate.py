"""Actual native glue, fresh graph replays, FP32 references and router gates."""
from pathlib import Path
import importlib.util
import json
import sys
import time
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval


def synthetic(m,device='cuda'):
    g=torch.Generator(device=device).manual_seed(941732)
    def dense(n,k,dtype=torch.bfloat16):
        return (torch.randn((n,k),device=device,generator=g)/k**.5).to(dtype)
    def vec(n,std=.1):return (std*torch.randn(n,device=device,generator=g)).bfloat16()
    w=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),
        unpooler=dense(8192,4096),router=dense(256,8192,torch.float32),
        router_bias=torch.randn(256,device=device,generator=g)*.055,
        attn_residual_gate=vec(8192,.3),ffn_residual_gate=vec(8192,.3),
        pre_ffn_norm=vec(8192),pre_expert_norm=vec(4096),post_expert_norm=vec(4096),post_ffn_norm=vec(8192),next_pre_attn_norm=vec(8192))
    old=m.v9_prepare
    m.v9_prepare=lambda weights:{}
    try:return w,m._prepare_weights(w)
    finally:m.v9_prepare=old


def error(a,b):
    d=(a.float()-b.float()).norm(dim=-1);n=b.float().norm(dim=-1)
    return float(d.norm()/n.norm().clamp_min(1e-12)),float((d/n.clamp_min(.1*n.mean()).clamp_min(1e-12)).max())


def main():
    start=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False
    m=local_eval.load_module('glue_best7',ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py')
    fg=local_eval.load_module('glue_dropin',HERE/'dropin.py')
    weights,st=synthetic(m);g=torch.Generator(device='cuda').manual_seed(6621)
    worst=0.;rows=[]
    for t in (1,8,16,32):
      for recipe in (0,1):
        pool=fg.FusedGlue(st,t,True,recipe);tail=fg.FusedGlue(st,t,False,recipe)
        for splits,dtype in ((1,torch.bfloat16),(2,torch.float32),(16,torch.float32)):
          p=torch.randn((splits,t,8192),device='cuda',dtype=dtype,generator=g)/splits**.5
          stream=torch.randn((t,8192),device='cuda',generator=g)
          local=torch.tensor([96],device='cuda',dtype=torch.int32)
          moe=torch.randn((t,4096),device='cuda',dtype=torch.bfloat16,generator=g)
          pool((p,splits),stream,local);tail(moe,stream)
          graph=torch.cuda.CUDAGraph()
          with torch.cuda.graph(graph):pr,s1,route=pool((p,splits),stream,local);h2,s2=tail(moe,s1)
          for draw in range(3):
            p.normal_(generator=g);stream.normal_(generator=g);moe.normal_(generator=g)
            if draw==1:p.mul_(.1);stream.mul_(2)
            local.fill_(32*(draw+1));graph.replay()
            h,ref_s=m.boundary((p,splits),stream,st['alpha1'],st['beta1'],st['pre_ffn_w'])
            ref_pr=h.float()@st['pool_router'].float().T
            # Check the rounding-sensitive prologue's product separately from the final norm.
            pe,prow=error(pr[0][0],ref_pr)
            assert pe<.005 and prow<.025,(t,recipe,splits,'pool',pe,prow)
            reference=m.route_and_norm((ref_pr.unsqueeze(0),1),st,dict(local_start=local))
            scores=(ref_pr[:,4096:4352]+ref_pr[:,4352:]).mul(m.ROUTER_MULT).sigmoid()+st['router_bias']
            got=route[1].long();ids=reference[1].long();top=scores.topk(9,-1).values
            # Route differences are accepted only under the layer's explicit score gates.
            assert bool((scores.gather(1,got)>=top[:,7:8]-.05).all()),'route score floor'
            must=top[:,:8]>top[:,8:9]+.03
            assert bool((~must|(ids[:,:,None]==got[:,None,:]).any(-1)).all()),'must-keep route'
            xe,xrow=error(route[0],reference[0]);assert xe<.01 and xrow<.05
            f=moe.float()@st[tail.name].float().T
            ref_h2,ref_s2=m.boundary((f.unsqueeze(0),1),ref_s,st['alpha2'],st['beta2'],st['next_pre_attn_w'],postw=st['post_ffn_w'])
            for label,a,b in (('s1',s1,ref_s),('h2',h2,ref_h2),('s2',s2,ref_s2)):
              e,r=error(a,b);assert e<.005 and r<.025,(t,recipe,splits,label,e,r);worst=max(worst,e)
            assert bool((pool.counts==0).all()) and bool((tail.counts==0).all()),'counter reset'
          rows.append(dict(tokens=t,recipe=recipe,splits=splits,dtype=str(dtype),draws=3,pool_l2=pe))
    result=dict(checks=len(rows)*3,rows=rows,max_l2=worst,elapsed_s=time.monotonic()-start,native_glue=True)
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
