"""Runtime shared by component admission and the self-contained H200 payload."""
import hashlib,re
import torch
from torch.utils.cpp_extension import load_inline
MODULES={};FREQ={};COUNTERS={}
LOCAL=False

def module(name):
    major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
    key=name,arch
    if key not in MODULES:
        src=SOURCES[name];tag=hashlib.sha256(src.encode()).hexdigest()[:12]
        cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',src,re.S).group(0)+';' for fn in ('sliding_resources','sliding_forward'))
        MODULES[key]=load_inline(name=f'p5oracle_{tag}_{arch}',cpp_sources=cpp,cuda_sources=src,functions=['sliding_forward','sliding_resources'],extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'])
    return MODULES[key]

def coordinates(data):
    c=data['config'];seq=data['seq_lens'].long();lo=(seq-c['window']+1).clamp_min(0)
    keys=torch.minimum(lo[:,None]+torch.arange(2048,device=seq.device)[None,:],(seq-1).clamp_min(0)[:,None])
    pages=data['page_table'].gather(1,keys//c['page_size']).long()
    rows=pages*c['page_size']+keys%c['page_size']
    return ((rows[:,None,:]*c['num_kv_heads']+torch.arange(c['num_kv_heads'],device=seq.device)[None,:,None])*64).contiguous()

def attention(data,coords,name,mode=0,clocks=None,qkv=None):
    c=data['config'];B,L,HQ,HK=c['batch'],c['q_len'],c['num_q_heads'],c['num_kv_heads'];family='d' if L==1 else 'v'
    cs=(8 if B==16 else 4) if family=='d' else 2
    if LOCAL and family=='d':cs=1
    P,S=(data['qkv'],0) if qkv is None else qkv;split=S if P.dtype==torch.float32 else 0
    raw=P[0] if P.dtype==torch.bfloat16 and P.dim()==3 and qkv is not None else P
    k=(c['rope_theta'],str(P.device))
    if k not in FREQ:FREQ[k]=(1./c['rope_theta']**(torch.arange(0,64,2,device=P.device).float()/64)).contiguous()
    groups=B*HK;k2=str(P.device),groups
    if k2 not in COUNTERS:COUNTERS[k2]=torch.zeros((groups,32),device=P.device,dtype=torch.int32)
    out=torch.empty((B,L,HQ*64),device=P.device,dtype=torch.bfloat16)
    args=(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),FREQ[k].data_ptr(),out.data_ptr(),B,L,HQ,HK,raw.stride(-2),data['page_table'].shape[1],c['page_size'],c['window'],split,P.stride(0) if split else 0,1,128,cs,raw.shape[-1],c['sm_scale']*c['qk_scale_factor']*1.4426950408889634,torch.cuda.current_stream(P.device).cuda_stream)
    ptr=0 if clocks is None else clocks.data_ptr()
    if family=='v':
        part=torch.empty((groups*16*64*68,),device=P.device)
        args+=part.data_ptr(),COUNTERS[k2].data_ptr(),cs,0,ptr
    if not name.endswith('A'):
        args+=coords.data_ptr(),mode
        if family=='d':args+=(ptr,)
    module(name).sliding_forward(*args)
    return out
