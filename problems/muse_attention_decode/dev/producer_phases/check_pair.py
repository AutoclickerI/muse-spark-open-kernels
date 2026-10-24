"""Force both layer FA3 glue routes through a portable attention operation."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_attention_decode')]
from local_eval import load_module,capture
m=load_module('verify1_pair_lab',ROOT/'prof/lab/verify1_sliding_fa3_pair.py')
ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *a,**k:None


def portable_fa3(**kw):
    q=kw['q'].float().view(1,8,16,64);seq=kw['cache_seqlens']-8
    keys=seq[:,None]+torch.arange(-2047,8,device=q.device)[None,:]
    rows=torch.gather(kw['page_table'].long(),1,keys.long())
    k=kw['k_cache'].view(-1,2,64)[rows].float();v=kw['v_cache'].view(-1,2,64)[rows].float()
    score=torch.einsum('blhgd,bshd->bhlgs',q.view(1,8,2,8,64),k)*kw['softmax_scale']
    qp=seq[:,None]+torch.arange(8,device=q.device)[None,:]
    mask=(keys[:,None,:]<=qp[:,:,None])&(keys[:,None,:]>=qp[:,:,None]-2047)
    score=score.masked_fill(~mask[:,None,:,None,:],-torch.inf)
    out=torch.einsum('bhlgs,bshd->blhgd',score.softmax(-1),v)
    return out.reshape(8,16,64).to(torch.bfloat16)

# The image's Hopper FA3 binary cannot execute on sm120. Its exact Python
# glue, query dtype, cache store and epilogue run with the portable operation.
m._attention_fa3_public.__globals__['__builtins__']=__builtins__
# The public function imports locally; replace just that function's import
# in a private test copy, leaving the payload and installed packages intact.
import inspect
source=inspect.getsource(m._attention_fa3_public).replace('    from sgl_kernel.flash_attn import flash_attn_with_kvcache\n','')
m.flash_attn_with_kvcache=portable_fa3
exec(compile(source,'<forced_fa3_public>','exec'),m.__dict__)
m.PDL=False
rows=[]
for count in (0,1,4):
    data=ref.generate_input(1,80000,'tp8','sliding',941,q_len=8,page=1,dist='soft');data['config']['tokens']=8
    raw=data['qkv'].view(8,2304)
    if count:
        raw=torch.randn((count,8,2304),device='cuda',dtype=torch.float32)
        data['qkv']=raw.sum(0).to(torch.bfloat16).view(1,8,2304);qkv=(raw,count)
    else:qkv=(raw.unsqueeze(0),1)
    state={'inv_freq':m._inv_freq(data['config']['rope_theta'],raw.device)}
    for name in ('single','current','best5'):
        def fn(d):
            return m.attention_verify1_sliding(qkv,d).view(1,8,1024) if name=='single' else m.layer_fa3(qkv,d,state,name).view(1,8,1024)
        graph,out=capture(fn,data);raw.add_(.015625)
        data['qkv']=(raw.sum(0) if count else raw).to(torch.bfloat16).view(1,8,2304)
        graph.replay();torch.cuda.synchronize()
        ok,msg=ref.check_implementation(data,ref.collect(data,out));assert ok,msg
        expected=m.gate_prepare(data);cached,cmsg,error=m.gate_cached(data,out,expected);assert cached,cmsg
        rows.append(dict(partials=count,path=name,gate=ok,cached_gate=cached,error=error,graph=True,msg=msg));print(rows[-1],flush=True)
result=dict(payload_sha256=hashlib.sha256((ROOT/'prof/lab/verify1_sliding_fa3_pair.py').read_bytes()).hexdigest(),fa3='portable operation, exact layer glue; no FA3 timing claim',checks=rows)
(ROOT/'problems/muse_attention_decode/dev/producer_phases/pair_validation_local.json').write_text(json.dumps(result,indent=2)+'\n')
