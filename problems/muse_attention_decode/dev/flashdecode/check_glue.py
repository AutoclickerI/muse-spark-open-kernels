"""Exercise the candidate's unavailable FA3 sliding glue with a portable tensor operation."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'problems/muse_attention_decode')]
from tools.local_eval import load_module,capture

ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *args,**kwargs:None
mod=load_module('verify8_candidate',ROOT/'problems/muse_attention_decode/cand_verify8_flashdecode.py')
# This portable operation exercises the real query prep, RoPE, store, scale and epilogue.
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
rows=[]
for scale in (1.,1./64.):
    data=ref.generate_input(2,128,'tp8','sliding',1618,q_len=8,page=1,kv_scale=scale)
    graph,out=capture(mod._fa3_verify,data)
    graph.replay();torch.cuda.synchronize()
    ok,msg=ref.check_implementation(data,ref.collect(data,out))
    row=dict(kv_scale=scale,ok=ok,graph=True,msg=msg)
    rows.append(row);print(row,flush=True)
    assert ok,row
(ROOT/'problems/muse_attention_decode/dev/flashdecode/glue_local.json').write_text(json.dumps(rows,indent=2)+'\n')
