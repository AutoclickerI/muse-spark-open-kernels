"""Native resident-history dispatch; local stand-ins are injected only by the screen tools."""
import hashlib,re
import torch
from torch.utils.cpp_extension import load_inline
_MODULES={};_FREQ={};_COUNTERS={}
TOPOLOGY={'C2P':2,'C4P':4,'C8P':8,'C16P':16,'C4S':4,'C8S':8}

def module(name):
    major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
    key=name,arch
    if key not in _MODULES:
        src=KERNELS[name];tag=hashlib.sha256(src.encode()).hexdigest()[:12]
        cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',src,re.S).group(0)+';' for fn in ('sliding_resources','sliding_forward'))
        _MODULES[key]=load_inline(name=f'p5res_{tag}_{arch}',cpp_sources=cpp,cuda_sources=src,functions=['sliding_forward','sliding_resources'],extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'])
    return _MODULES[key]

def attention(data,qkv=None,name='C4P',early=False):
    cfg=data['config'];B,L,HQ,HK=cfg['batch'],cfg['q_len'],cfg['num_q_heads'],cfg['num_kv_heads']
    assert L==8 and cfg['window']<=2048 and cfg['window']>0 and HQ==8*HK and cfg['head_dim']==64
    cs=TOPOLOGY[name];P,S=(data['qkv'],0) if qkv is None else qkv;split=S if P.dtype==torch.float32 else 0
    raw=P[0] if P.dtype==torch.bfloat16 and P.dim()==3 and qkv is not None else P
    key=cfg['rope_theta'],str(P.device)
    if key not in _FREQ:_FREQ[key]=(1./cfg['rope_theta']**(torch.arange(0,64,2,device=P.device).float()/64)).contiguous()
    freq=_FREQ[key];groups=B*HK
    key=str(P.device),groups
    if key not in _COUNTERS:_COUNTERS[key]=torch.zeros((groups,32),device=P.device,dtype=torch.int32)
    # Native DSM ignores global scratch. The explicit local stand-in uses the
    # same ABI and scratch for its nonspinning last-CTA merge.
    part=torch.empty((groups*16*64*68,),device=P.device);out=torch.empty((B,8,HQ*64),device=P.device,dtype=torch.bfloat16)
    module(name).sliding_forward(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),freq.data_ptr(),out.data_ptr(),B,L,HQ,HK,raw.stride(-2),data['page_table'].shape[1],cfg['page_size'],cfg['window'],split,P.stride(0) if split else 0,1,128,cs,raw.shape[-1],cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,torch.cuda.current_stream(P.device).cuda_stream,part.data_ptr(),_COUNTERS[key].data_ptr(),cs,int(early))
    return out
