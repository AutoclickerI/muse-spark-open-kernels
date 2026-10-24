import triton
from recoded import rc_repack,RC_SIGNS,RC_ZEROS
"""Packed WGMMA pipeline with a distributed vector W2 finalizer."""
import functools
import hashlib
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
from engine import mw_repack_planes
FX_CUDA = Path(__file__).with_name('base.cu').read_text() + Path(__file__).with_name('issue.cu').read_text() + Path(__file__).parents[1].joinpath('recode/recode.cu').read_text()
FX_CPP = '#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\nint64_t fx_init();std::vector<int64_t> fx_info(int64_t,int64_t,int64_t);void fx_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void fx_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void fx_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);'
FX_CONFIGS = ((16, 128, 4, 1), (32, 128, 4, 1))

@functools.lru_cache(None)
def fx_module():
    target = '-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability() == (9, 0) else '-gencode=arch=compute_120,code=sm_120'
    m = load_inline(name='w13_issue_' + hashlib.sha256(FX_CUDA.encode()).hexdigest()[:12], cpp_sources=FX_CPP + WI_CPP, cuda_sources=FX_CUDA, functions=['fx_init', 'fx_info', 'fx_route', 'fx_run', 'fx_finish', 'wi_init', 'wi_info', 'wi_run', 'rc_init', 'rc_info', 'rc_run', 'wi_mixed', 'wi_mixed_info', 'wi_quad'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', target], verbose=False)
    m.fx_init()
    m.wi_init()
    m.rc_init()
    return m

def fx_prepare(w):
    bank = {}
    for key in ('w13', 'w2'):
        sf = w[key + '_sf'].view(torch.uint8)
        e, n, k16 = sf.shape
        bank[key] = mw_repack_planes(w[key])
        bank[key + '_sf'] = sf.reshape(e, n // 16, 2, 8, k16 // 2, 2).permute(0, 1, 4, 3, 5, 2).contiguous()
    bank['w13_gs'] = torch.cat((w.get('w13_gs_gate', w['w13_gs']), w.get('w13_gs_up', w['w13_gs']))).float().contiguous()
    bank['w2_gs'] = w['w2_gs'].float().contiguous()
    bank['pnw'] = w.get('post_norm_w', w.get('post_expert_norm')).contiguous()
    return fx_from_packed(bank)

def fx_from_packed(bank):
    ng = bank['w13'].shape[1]
    rr = torch.arange(ng // 2, device=bank['w13'].device).reshape(-1, 2)
    order = torch.cat((rr, rr + ng // 2), 1).flatten()
    return dict(bank, w13=bank['w13'].index_select(1, order).contiguous(), w13_sf=bank['w13_sf'].index_select(1, order).contiguous(), mod=fx_module())

def fx_plan(data, bank, which=0, grid=528, mode=1):
    nt, kt, st, sk = FX_CONFIGS[which]
    x = data['hidden_states']
    m = x.shape[0]
    dev = x.device
    cap = m * 8 + 32 * nt
    gc = cap // nt
    c = torch.empty(32, device=dev, dtype=torch.int32)
    off = torch.empty(33, device=dev, dtype=torch.int32)
    order = torch.empty_like(c)
    task = torch.empty_like(off)
    slots = torch.empty((32, m), device=dev, dtype=torch.int32)
    ids = data['topk_ids'].to(torch.int32).contiguous()
    tw = data['topk_weights'].float().contiguous()
    pos = torch.empty_like(ids)
    clear = gc * 193 + m
    state = torch.empty(clear + m, device=dev, dtype=torch.int32)
    act = torch.empty((cap, 4096), device=dev, dtype=torch.bfloat16)
    p13 = torch.empty((sk, cap, 8192) if sk > 1 else (1,), device=dev, dtype=torch.float32)
    p2 = torch.empty((cap // nt * 64 * 8 * 2,), device=dev, dtype=torch.float32)
    y = torch.empty((cap, 4096), device=dev, dtype=torch.float32)
    ss = torch.empty((cap, 64), device=dev, dtype=torch.float32)
    inv = torch.empty(cap, device=dev, dtype=torch.float32)
    out = torch.empty_like(x)
    common = [c, off, order, task, slots, act]
    tail = [ids, tw, pos, bank['pnw'], y, ss, inv, state, out]
    v13 = [bank['w13'], bank['w13_sf'], bank['w13_gs'], x] + common + [p13] + tail
    v2 = [bank['w2'], bank['w2_sf'], bank['w2_gs'], act] + common + [p2] + tail
    start = data['config'].get('local_start', 0)
    vp13 = [v.data_ptr() for v in v13]
    vp2 = [v.data_ptr() for v in v2]
    rp = [v.data_ptr() for v in (ids, c, off, order, task, slots, pos, state, out)]
    owners = (v13, v2, ids, tw)

    def stream():
        return torch.cuda.current_stream().cuda_stream

    def route():
        _ = owners
        bank['mod'].fx_route(rp, m, nt, start, clear, stream())

    def first():
        _ = owners
        bank['mod'].fx_run(vp13, 1, which, grid, start, 1, 0, m, cap, stream())

    def second():
        _ = owners
        bank['mod'].fx_run(vp2, 0, which, grid, start, mode, 0, m, cap, stream())
        return out

    def finish():
        _ = owners
        bank['mod'].fx_finish(vp2, m, start, stream())
        return out
    steps = (route, first, second, finish) if mode == 1 else (route, first, second)
    return dict(steps=steps, v2=v2, mode=mode, which=which, grid=grid, start=start, profile=p2, pointers=vp2, M=m, cap=cap, v13=v13)

def fx_work(data, bank, which=0, grid=528, mode=1):
    return fx_plan(data, bank, which, grid, mode)['steps']

def fx_call(data, bank, which=0, grid=528, mode=1):
    for f in fx_work(data, bank, which, grid, mode):
        out = f()
    return out

WI_CPP='int64_t wi_init();std::vector<int64_t> wi_info(int64_t);void wi_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);int64_t rc_init();std::vector<int64_t> rc_info(int64_t,int64_t,int64_t);void rc_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void wi_mixed(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> wi_mixed_info(int64_t);void wi_quad(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'

def wi_prepare(weights):
    bank=fx_prepare(weights)
    for name in ('w13','w2'):
        q=bank[name];sf=bank[name+'_sf'];qr=torch.empty_like(q);sr=torch.empty_like(sf)
        rc_repack[(triton.cdiv(q.numel()//2,256),)](q,sf,qr,sr,q.numel()//2,RC_SIGNS,RC_ZEROS)
        bank[name+'_rec']=qr;bank[name+'_rec_sf']=sr
    bank['mix_stream']=torch.cuda.Stream(device=bank['w13'].device);bank['mix_start']=torch.cuda.Event();bank['mix_done']=torch.cuda.Event()
    return bank

def wi_plan(data,bank,policy=-1,grid=2112):
    plan=fx_plan(data,bank,1,grid,1)
    if policy==-1:return plan
    v=plan['v13'].copy();v2=plan['v2'].copy()
    if policy in (2,3,4,5,6,8,9):v[:2]=[bank['w13_rec'],bank['w13_rec_sf']]
    if policy in (3,4,5,6,8,9):v2[:2]=[bank['w2_rec'],bank['w2_rec_sf']]
    ptrs=[x.data_ptr() for x in v];p2=[x.data_ptr() for x in v2];m=plan['M'];cap=plan['cap'];start=plan['start']
    def first():
        _=v
        if policy in (7,8):
            bank['mod'].wi_quad(ptrs,0 if policy==7 else 1,grid,m,cap,start,torch.cuda.current_stream().cuda_stream)
        elif policy in (5,6,9):
            parent=torch.cuda.current_stream()
            if policy==5:
                bank['mod'].wi_mixed(ptrs,0,grid,m,cap,start,parent.cuda_stream);bank['mod'].wi_mixed(ptrs,1,grid,m,cap,start,parent.cuda_stream)
            else:
                bank['mix_start'].record(parent);bank['mix_stream'].wait_event(bank['mix_start'])
                with torch.cuda.stream(bank['mix_stream']):
                    if policy==9:bank['mod'].wi_quad(ptrs,2,grid,m,cap,start,bank['mix_stream'].cuda_stream)
                    else:bank['mod'].wi_mixed(ptrs,0,grid,m,cap,start,bank['mix_stream'].cuda_stream)
                    bank['mix_done'].record()
                if policy==9:bank['mod'].wi_quad(ptrs,3,grid,m,cap,start,parent.cuda_stream)
                else:bank['mod'].wi_mixed(ptrs,1,grid,m,cap,start,parent.cuda_stream)
                parent.wait_event(bank['mix_done'])
        elif policy==3:bank['mod'].rc_run(ptrs,1,1,grid,start,1,0,m,cap,torch.cuda.current_stream().cuda_stream)
        else:bank['mod'].wi_run(ptrs,2 if policy==4 else policy,grid,m,cap,start,torch.cuda.current_stream().cuda_stream)
    def second():
        _=v2
        bank['mod'].rc_run(p2,0,1,grid,start,1,0,m,cap,torch.cuda.current_stream().cuda_stream)
    steps=plan['steps'];plan['steps']=(steps[0],first,second if policy in (3,4,5,6,8,9) else steps[2],steps[3]);return plan

def wi_call(data,bank,policy=-1,grid=2112):
    for f in wi_plan(data,bank,policy,grid)['steps']:out=f()
    return out
