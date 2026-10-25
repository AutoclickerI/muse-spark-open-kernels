"""Single-launch sliding attention over every window key, with in-kernel Q/K prep and cluster combine."""
import hashlib
import re
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import build
CUDA=build.source()
CPP='#include <cstdint>\n#include <vector>\n'+ '\n'.join(
    re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',CUDA,re.S).group(0)+';'
    for fn in ('sliding_forward','sliding_resources'))
_MOD={}
_PLANS={}
_FREQ={}

def module():
    major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
    if arch not in _MOD:
        tag=hashlib.sha256(CUDA.encode()).hexdigest()[:12]
        _MOD[arch]=load_inline(name=f'cluster_sliding_{tag}_{arch}',cpp_sources=CPP,cuda_sources=CUDA,
            functions=['sliding_forward','sliding_resources'],extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'])
    return _MOD[arch]

def plan(cfg,pages,device,qt,nk):
    B,L,HK=cfg['batch'],cfg['q_len'],cfg['num_kv_heads']
    key=(str(device),B,L,HK,pages,cfg['page_size'],cfg['window'],qt,nk)
    if key not in _PLANS:
        groups=B*HK*qt
        limit=min(16,max(1,(min(pages*cfg['page_size'],cfg['window'])+nk-1)//nk))
        rows=[]
        for cs in range(1,limit+1):
            resource=module().sliding_resources(L,qt,nk,cs)
            if resource[-1]>0:rows.append((cs,resource))
        if not rows:raise RuntimeError('no resident cluster for sliding attention')
        viable=[x for x in rows if x[1][-1]>=groups]
        _PLANS[key]=max(viable or rows[:1],key=lambda x:x[0])
    return _PLANS[key]

def attention(data,qkv=None,qt=None,nk=128,cs=None,gate=True):
    cfg=data['config'];B,L,HQ,HK=cfg['batch'],cfg['q_len'],cfg['num_q_heads'],cfg['num_kv_heads']
    assert cfg['window']>0 and L in (1,8) and HQ==8*HK and cfg['head_dim']==64
    P,S=(data['qkv'],0) if qkv is None else qkv
    split=S if P.dtype==torch.float32 else 0
    raw=P[0] if P.dtype==torch.bfloat16 and P.dim()==3 and qkv is not None else P
    qt=(4 if B*HK<=4 else 1) if qt is None else qt
    assert qt in (1,4) and nk in (64,128)
    if cs is None:cs=plan(cfg,data['page_table'].shape[1],P.device,qt,nk)[0]
    key=(cfg['rope_theta'],str(P.device))
    if key not in _FREQ:_FREQ[key]=(1.0/(cfg['rope_theta']**(torch.arange(0,64,2,device=P.device).float()/64))).contiguous()
    out=torch.empty((B,L,HQ*64),device=P.device,dtype=torch.bfloat16)
    module().sliding_forward(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),
        data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),
        _FREQ[key].data_ptr(),out.data_ptr(),B,L,HQ,HK,raw.stride(-2),data['page_table'].shape[1],cfg['page_size'],cfg['window'],
        split,P.stride(0) if split else 0,qt,nk,cs,raw.shape[-1] if gate else (HQ+2*HK)*64,
        cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,torch.cuda.current_stream(P.device).cuda_stream)
    return out
