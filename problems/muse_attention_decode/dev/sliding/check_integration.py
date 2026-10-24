"""Check the layer adapter, one-launch property, and full verify's Hopper Python glue."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'problems/muse_attention_decode')]
from tools.local_eval import load_module, capture, copy_into

ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *args,**kwargs: None
candidate_path=ROOT/'problems/muse_attention_decode/cand_sliding_singlelaunch.py'
adapter_path=ROOT/'problems/muse_decode_layer/dev/sliding_attention.py'
mod=load_module('window_candidate',candidate_path)
adapter=load_module('layer_window',adapter_path)
rows=[]
for B in (1,16,32):
    for L in (1,8):
        data=ref.generate_input(B,80000,'tp8','sliding',811,q_len=L,page=1,spread='agentic',dist='soft')
        def layer(data):
            cfg={k:v for k,v in data['config'].items() if k not in ('num_q_heads','num_kv_heads','head_dim')}
            meta=dict(data,config=cfg)
            return adapter.attention_sliding(data['qkv'].view(B*L,-1),meta).view(B,L,1024)
        graph,out=capture(layer,data)
        fresh=ref.generate_input(B,80000,'tp8','sliding',812,q_len=L,page=1,spread='agentic',dist='soft')
        copy_into(data,fresh)
        graph.replay();torch.cuda.synchronize()
        ok,msg=ref.check_implementation(data,ref.collect(data,out))
        assert ok,(B,L,msg)
        record=dict(B=B,L=L,gate=True,msg=msg)
        if B==1:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                layer(data)
                torch.cuda.synchronize()
            kernels=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
            assert len(kernels)==1,kernels
            assert 'window_mma' in kernels[0],kernels
            record['cuda_kernels']=kernels
        for count in (1,4):
            partial=torch.randn((count,B*L,2304),device='cuda',dtype=torch.float32)
            data['qkv']=partial.sum(0).to(torch.bfloat16).view(B,L,2304)
            def from_parts(data):
                return adapter.attention_sliding((partial,count),data).view(B,L,1024)
            pg,po=capture(from_parts,data)
            partial.add_(0.03125)
            data['qkv']=partial.sum(0).to(torch.bfloat16).view(B,L,2304)
            pg.replay();torch.cuda.synchronize()
            passed,why=ref.check_implementation(data,ref.collect(data,po))
            assert passed,(B,L,count,why)
            record.setdefault('projection_partials',[]).append(dict(splits=count,gate=passed,msg=why))
            if B==1 and count==4:
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    from_parts(data)
                    torch.cuda.synchronize()
                pk=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                assert len(pk)==1,pk
                record['partial_input_cuda_kernels']=pk
            del pg,po,partial
        rows.append(record)
        print(record,flush=True)
        del graph,out,data,fresh
        torch.cuda.empty_cache()


def portable_fa3(**kw):
    """GPU-tensor stand-in for FA3; exercises actual norm/store/epilogue glue."""
    query=kw['q'];B=kw['page_table'].shape[0];L=kw['max_seqlen_q']
    HQ,D=query.shape[-2:]; HK=kw['k_cache'].shape[-2];PS=kw['k_cache'].shape[1]
    n=kw['page_table'].shape[1]*PS
    t=torch.arange(n,device=query.device)
    ids=kw['page_table'][:,t//PS].long()*PS+t%PS
    ids=ids.clamp_min(0)
    k=kw['k_cache'].view(-1,HK,D)[ids].float()
    v=kw['v_cache'].view(-1,HK,D)[ids].float()
    if 'k_descale' in kw:
        k=k*kw['k_descale'][:,None,:,None]
        v=v*kw['v_descale'][:,None,:,None]
    q=query.reshape(B,L,HK,HQ//HK,D).float()
    logits=torch.einsum('blhgd,bnhd->blhgn',q,k)*kw['softmax_scale']
    limit=kw['cache_seqlens'][:,None]-L+torch.arange(L,device=q.device)[None,:]
    mask=t[None,None,:]<=limit[:,:,None]
    logits=logits.masked_fill(~mask[:,:,None,None,:],-float('inf'))
    return torch.einsum('blhgn,bnhd->blhgd',logits.softmax(-1),v).reshape(B*L,HQ,D).to(torch.bfloat16)

mod.flash_attn_with_kvcache=portable_fa3
for scale in (1.,1./64.):
    data=ref.generate_input(2,2048,'tp8','full',913,q_len=8,page=1,kv_scale=scale)
    graph,out=capture(mod._fa3_verify,data)
    graph.replay();torch.cuda.synchronize()
    ok,msg=ref.check_implementation(data,ref.collect(data,out))
    assert ok,msg
    rows.append(dict(forced_hopper_fa3_glue=True,kv_scale=scale,gate=True,msg=msg))
    print(rows[-1],flush=True)

result=dict(candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),checks=rows)
(ROOT/'problems/muse_attention_decode/dev/sliding/integration_local.json').write_text(json.dumps(result,indent=2)+'\n')
