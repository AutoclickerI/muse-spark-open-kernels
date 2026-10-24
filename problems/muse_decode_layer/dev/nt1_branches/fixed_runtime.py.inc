import hashlib
import statistics
import time

NT1_CPP = '''#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdint.h>
int64_t max_smem_attr();int64_t smem_needed(int64_t,int64_t,int64_t);
int64_t resident_blocks(int64_t,int64_t,int64_t);
void run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);
'''
NT1_MODULES = {}
NT1_HOPPER = torch.cuda.get_device_capability() == (9, 0)


def nt1_module(kind):
    if kind not in NT1_MODULES:
        source = NT1_SOURCES[kind]
        tag = hashlib.sha256(source.encode()).hexdigest()[:12]
        mod = load_inline(name='muse_nt1_limit_' + kind + tag, cpp_sources=NT1_CPP, cuda_sources=source,
            functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'],
            no_implicit_headers=True, with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo',
                *(['-gencode=arch=compute_90a,code=sm_90a'] if NT1_HOPPER else [])], verbose=False)
        mod.smem_optin = mod.max_smem_attr()
        NT1_MODULES[kind] = mod
    return NT1_MODULES[kind]


def nt1_banks(weights):
    device=weights['w13'].device
    rows=torch.arange(512,device=device)[:,None]*8+torch.arange(8,device=device)
    maps={'w13':torch.cat((rows,rows+4096),1),'w2':torch.arange(4096,device=device).view(256,16)}
    recode={};half={};micro={}
    for name,rowmap in maps.items():
        base=torch.stack([_repack(weights[name][e],weights[name+'_sf'][e],rowmap) for e in range(32)])
        key=name+'p';recode[key]=base;tiles=rowmap.shape[0]
        w=base[...,:32768].reshape(32,tiles,2,16384).permute(0,2,1,3)
        sf=base[...,32768:].reshape(32,tiles,2,2048).permute(0,2,1,3)
        half[key]=torch.cat((w,sf),-1).contiguous().reshape(32,tiles*2,18432)
        w=base[...,:32768].reshape(32,tiles,8,8,32,16).permute(0,1,3,2,4,5).reshape(32,tiles,8,4096)
        sf=base[...,32768:].reshape(32,tiles,8,32,8,2).permute(0,1,4,2,3,5).reshape(32,tiles,8,512)
        micro[key]=torch.cat((w,sf),-1).contiguous()
    banks={kind:(half if kind=='K2' else micro if kind in ('D','ABCD','D128','ABCD128') else recode) for kind in NT1_SOURCES}
    return dict(banks=banks,counters={k:torch.zeros(128,device=device,dtype=torch.int32) for k in banks},
        tickets=torch.zeros(256*768,device=device,dtype=torch.int32),
        gs13=weights['w13_gs'].float().contiguous(),gs2=weights['w2_gs'].float().contiguous(),
        pnw=weights['post_norm_w'].contiguous(),sms=torch.cuda.get_device_properties(device).multi_processor_count)


def nt1_call(data, state, kind='INC', stages=2, grid=264, scratch=False):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].int().contiguous(); tw = data['topk_weights'].float().contiguous()
    m = x.shape[0]; p = m * 8; dev = x.device
    if not 1<=m<=32:raise ValueError('small-M specialization expects 1..32 tokens')
    mod = nt1_module(kind)
    if mod.smem_needed(1, stages, m) > mod.smem_optin:
        raise ValueError(f'{kind} stages{stages} shared memory exceeds device capacity')
    resident = mod.resident_blocks(1, stages, m)
    if resident < 1:
        raise ValueError('no resident CTA for NT1')
    grid = min(grid, state['sms'] * resident)
    bank = state['banks'][kind]
    inter = torch.empty((p, 4096), device=dev, dtype=torch.bfloat16)
    y = torch.empty((p, 4096), device=dev)
    ss = torch.empty((p,), device=dev)
    cnt = state['counters'][kind]
    out = torch.empty_like(x)
    tensors = [x, ids, tw, state['pnw'], bank['w13p'], bank['w2p'], state['gs13'], state['gs2'], inter, y, ss]
    tensors += [cnt, out]
    if kind=='K2':
        parts=torch.empty((p,2,8192),device=dev)
        tensors += [parts,state['tickets']]
    meta=None
    if kind in ('A','B','C','ABC','D','ABCD','D128','ABCD128'):
        meta=torch.empty(2944,device=dev,dtype=torch.uint8)
        tensors += [meta]
    mod.run([v.data_ptr() for v in tensors], m, data['config']['local_start'], stages, grid,
            torch.cuda.current_stream(dev).cuda_stream)
    return (out,inter,y,ss,meta) if scratch else out


def nt1_fp32(data, weights):
    result = torch.zeros_like(data['hidden_states'], dtype=torch.float32)
    ids = data['topk_ids'] - data['config']['local_start']
    grid = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device=ids.device)
    def expand(name, e):
        p = weights[name][e]
        code = torch.stack((p & 15, p >> 4), -1).flatten(-2)
        value = grid[(code & 7).long()] * torch.where(code & 8 != 0, -1., 1.)
        return (value.view(-1, 256, 16) * weights[name + '_sf'][e].float()[..., None] *
                weights[name + '_gs'][e]).reshape(-1, 4096)
    for e in range(32):
        token, slot = (ids == e).nonzero(as_tuple=True)
        if not token.numel():
            continue
        h = data['hidden_states'][token].float() @ expand('w13', e).T
        a = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        y = a @ expand('w2', e).T
        z = y * (1. + weights['post_norm_w'].float())
        result.index_add_(0, token, z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-8) * data['topk_weights'][token, slot, None])
    return result


def nt1_error(got, want):
    g, w = got.float(), want.float()
    denom = w.norm(dim=-1)
    return (float((g-w).norm()/w.norm().clamp_min(1e-15)),
            float(((g-w).norm(dim=-1)/denom.clamp_min(.1*denom.mean()).clamp_min(1e-15)).max()))

