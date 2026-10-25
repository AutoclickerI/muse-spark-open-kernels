"""Launch-only runtime shared verbatim by component screens and whole-layer lab arms."""
import hashlib,re
import torch
from torch.utils.cpp_extension import load_inline
_MODULES={};_COUNTERS={};_FREQ={}

def module(kind):
    major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
    key=kind,arch
    if key not in _MODULES:
        cuda=KERNELS[kind];tag=hashlib.sha256(cuda.encode()).hexdigest()[:12]
        cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+f+r'\([^)]*\)',cuda,re.S).group(0)+';' for f in ('sliding_forward','sliding_resources'))
        _MODULES[key]=load_inline(name=f'p5r2_{tag}_{arch}',cpp_sources=cpp,cuda_sources=cuda,functions=['sliding_forward','sliding_resources'],extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'])
    return _MODULES[key]

def attention(data,qkv=None,kind='last',qt=1,ns=8):
    cfg=data['config'];B,L,HQ,HK=cfg['batch'],cfg['q_len'],cfg['num_q_heads'],cfg['num_kv_heads']
    assert L in (1,8) and HQ==8*HK and cfg['head_dim']==64 and cfg['window']>0
    assert qt in (1,2,4,8) and 1<=ns<=32
    if kind=='early':assert ns==16 and cfg['window']<=2048
    if kind in ('dsm','dsm_old'):assert ns<=16
    if kind=='dsm_old':assert qt in (1,4)
    P,S=(data['qkv'],0) if qkv is None else qkv;split=S if P.dtype==torch.float32 else 0
    raw=P[0] if P.dtype==torch.bfloat16 and P.dim()==3 and qkv is not None else P
    groups=B*HK*qt;key=str(P.device),groups
    if key not in _COUNTERS:_COUNTERS[key]=torch.zeros((groups,32),device=P.device,dtype=torch.int32)
    counter=_COUNTERS[key];part=torch.empty((groups*ns*L*(8//qt)*68,),device=P.device)
    key=cfg['rope_theta'],str(P.device)
    if key not in _FREQ:_FREQ[key]=(1./cfg['rope_theta']**(torch.arange(0,64,2,device=P.device).float()/64)).contiguous()
    out=torch.empty((B,L,HQ*64),device=P.device,dtype=torch.bfloat16)
    args=(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),_FREQ[key].data_ptr(),out.data_ptr(),B,L,HQ,HK,raw.stride(-2),data['page_table'].shape[1],cfg['page_size'],cfg['window'],split,P.stride(0) if split else 0,qt,128,ns if kind in ('dsm','dsm_old') else 1,raw.shape[-1],cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,torch.cuda.current_stream(P.device).cuda_stream)
    if kind not in ('dsm','dsm_old'):args+=part.data_ptr(),counter.data_ptr(),ns
    module(kind).sliding_forward(*args)
    return out

# Public shape dispatch only. Every arm computes all queries and window keys.
ARMS={
 (16,1):{'L4':('last',1,4),'L8':('last',1,8),'L16':('last',1,16),'LQ2':('last',2,4),'LQ4':('last',4,2),
         'E16':('early',1,16),'D4':('dsm',1,4),'D8':('dsm',1,8),'DQ2':('dsm',2,4)},
 (32,1):{'L2':('last',1,2),'L4':('last',1,4),'L8':('last',1,8),'L16':('last',1,16),'LQ2':('last',2,2),
         'E16':('early',1,16),'D2':('dsm',1,2),'D4':('dsm',1,4)},
 (1,8):{'LQ1':('last',1,16),'LQ2':('last',2,16),'LQ4':('last',4,16),'L8Q8':('last',8,8),'L16Q8':('last',8,16),
        'E4':('early',4,16),'E8':('early',8,16),'D8':('dsm',4,8),'D16':('dsm',4,16),'DQ8':('dsm',8,8),'RESET_D16':('dsm_old',4,16)}
}
SELECTED={(16,1):('last',1,8),(32,1):('last',1,4),(1,8):('last',8,8)}

def apply_arm(data,qkv,arm):
    kind,qt,ns=arm
    return attention(data,qkv,kind=kind,qt=qt,ns=ns)
