import hashlib
import time
import statistics

NT1_CPP='''#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <vector>
int64_t max_smem_attr();int64_t smem_needed(int64_t,int64_t,int64_t);int64_t resident_blocks(int64_t,int64_t,int64_t);
void run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);
'''
NT1_MODULES={}
NT1_HOPPER=torch.cuda.get_device_capability()==(9,0)


def nt1_module(kind):
    if kind not in NT1_MODULES:
        source=NT1_SOURCES[kind];tag=hashlib.sha256(source.encode()).hexdigest()[:12]
        mod=load_inline(name='muse_timeline_'+kind+tag,cpp_sources=NT1_CPP,cuda_sources=source,
            functions=['max_smem_attr','smem_needed','resident_blocks','run'],no_implicit_headers=True,
            with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',
            *(['-gencode=arch=compute_90a,code=sm_90a'] if NT1_HOPPER else [])],verbose=False)
        mod.smem_optin=mod.max_smem_attr();NT1_MODULES[kind]=mod
    return NT1_MODULES[kind]


CONTROL_MODULE=[]
ACTIVITY_MODULE=[]


def control_module():
    if not CONTROL_MODULE:
        cpp='''#include <pybind11/pybind11.h>
#include <stdint.h>
void mark(int64_t,int64_t,int64_t);
void resolution(int64_t,int64_t);
void empty(int64_t,int64_t,int64_t,int64_t,int64_t);
void streaming(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
int64_t control_optin();
'''
        tag=hashlib.sha256((cpp+CONTROL_CUDA).encode()).hexdigest()[:12]
        mod=load_inline(name='muse_timeline_control_'+tag,cpp_sources=cpp,cuda_sources=CONTROL_CUDA,
            functions=['mark','resolution','empty','streaming','control_optin'],
            no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',
                *(['-gencode=arch=compute_90a,code=sm_90a'] if NT1_HOPPER else [])],verbose=False)
        mod.control_optin();CONTROL_MODULE.append(mod)
    return CONTROL_MODULE[0]


def activity_module():
    """CUPTI kernel records (graph-node start/end); optional: the stamps stand without it."""
    if not ACTIVITY_MODULE:
        tag=hashlib.sha256(ACTIVITY_CPP.encode()).hexdigest()[:12]
        ACTIVITY_MODULE.append(load_inline(name='muse_timeline_activity_'+tag,cpp_sources=ACTIVITY_CPP,
            functions=['profile_begin','profile_end'],no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_include_paths=['/usr/local/cuda/include'],
            extra_ldflags=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcupti'],verbose=False))
    return ACTIVITY_MODULE[0]


def nt1_banks(w):
    dev=w['w13'].device;r=torch.arange(512,device=dev)[:,None]*8+torch.arange(8,device=dev)
    bank={}
    for name,rows in (('w13',torch.cat((r,r+4096),1)),('w2',torch.arange(4096,device=dev).view(256,16))):
        bank[name+'p']=torch.stack([_repack(w[name][e],w[name+'_sf'][e],rows) for e in range(32)])
    return dict(bank=bank,counters={k:torch.zeros(64,device=dev,dtype=torch.int32) for k in NT1_SOURCES},
        pnw=w['post_norm_w'].contiguous(),gs13=w['w13_gs'].float().contiguous(),gs2=w['w2_gs'].float().contiguous(),
        sms=torch.cuda.get_device_properties(dev).multi_processor_count)


def nt1_call(d,state,kind='INC',grid=264,stages=2,trace_out=False):
    x=d['hidden_states'].contiguous();m=len(x);dev=x.device;mod=nt1_module(kind)
    if mod.smem_needed(1,stages,m)>mod.smem_optin:raise ValueError('stage depth exceeds shared capacity')
    grid=min(grid,state['sms']*mod.resident_blocks(1,stages,m))
    if grid<1:raise ValueError('no resident CTA')
    inter=torch.empty((m*8,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((m*8,4096),device=dev);ss=torch.empty(m*8,device=dev)
    out=torch.empty_like(x);ids=d['topk_ids'].int().contiguous();tw=d['topk_weights'].float().contiguous()
    tensors=[x,ids,tw,state['pnw'],state['bank']['w13p'],state['bank']['w2p'],state['gs13'],state['gs2'],inter,y,ss,state['counters'][kind],out]
    stamps=torch.empty((grid,64),device=dev,dtype=torch.int64) if kind.startswith('TRACE') else None
    if stamps is not None:tensors.append(stamps)
    mod.run([v.data_ptr() for v in tensors],m,d['config']['local_start'],stages,grid,torch.cuda.current_stream(dev).cuda_stream)
    return (out,stamps) if trace_out else out


def nt1_error(got,want):
    a,b=got.float(),want.float();den=b.norm(dim=-1)
    return float((a-b).norm()/b.norm().clamp_min(1e-15)),float(((a-b).norm(dim=-1)/den.clamp_min(den.mean()*.1).clamp_min(1e-15)).max())
