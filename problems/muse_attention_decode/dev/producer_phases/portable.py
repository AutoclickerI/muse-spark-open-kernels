"""A portable attention operation for checking Hopper-only Python glue, without timing claims."""
import torch

def portable_fa3(**kw):
    B=kw['page_table'].shape[0];L=kw['max_seqlen_q'];q=kw['q'].float().view(B,L,16,64)
    seq=kw['cache_seqlens']-L;left=kw['window_size'][0]
    if left>=0:
        keys=seq[:,None]+torch.arange(-left,L,device=q.device)[None,:]
    else:keys=torch.arange(kw['page_table'].shape[1],device=q.device)[None,:].expand(B,-1)
    valid=(keys>=0)&(keys<kw['cache_seqlens'][:,None])
    safe=keys.clamp(0,kw['page_table'].shape[1]-1)
    rows=torch.gather(kw['page_table'].long(),1,safe.long()).clamp_min(0)
    k=kw['k_cache'].view(-1,2,64)[rows].float();v=kw['v_cache'].view(-1,2,64)[rows].float()
    if 'k_descale' in kw:k*=kw['k_descale'][:,None,:,None];v*=kw['v_descale'][:,None,:,None]
    score=torch.einsum('blhgd,bshd->bhlgs',q.view(B,L,2,8,64),k)*kw['softmax_scale']
    qp=seq[:,None]+torch.arange(L,device=q.device)[None,:]
    mask=valid[:,None,:]&(keys[:,None,:]<=qp[:,:,None])
    if left>=0:mask &= keys[:,None,:]>=qp[:,:,None]-left
    score=score.masked_fill(~mask[:,None,:,None,:],-torch.inf)
    out=torch.einsum('bhlgs,bshd->blhgd',score.softmax(-1),v)
    return out.reshape(B*L,16,64).to(torch.bfloat16)
