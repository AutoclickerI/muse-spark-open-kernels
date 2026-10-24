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
    device = weights['w13'].device
    rows = torch.arange(512, device=device)[:, None] * 8 + torch.arange(8, device=device)
    rows13 = torch.cat((rows, rows + 4096), 1)
    rows2 = torch.arange(4096, device=device).view(256, 16)
    banks = {}
    for name, repack in (('recode', _repack), ('raw', _raw_repack)):
        bank = {}
        for weight, rowmap in (('w13', rows13), ('w2', rows2)):
            value = torch.empty((32, rowmap.shape[0], 36864), device=device, dtype=torch.uint8)
            for expert in range(32):
                value[expert] = repack(weights[weight][expert], weights[weight + '_sf'][expert], rowmap)
            bank[weight + 'p'] = value
        banks[name] = bank
    banks['lop26'] = banks['recode']
    half = {}
    for key in ('w13p','w2p'):
        base = banks['recode'][key]; tiles = base.shape[1]
        w = base[...,:32768].reshape(32,tiles,2,16384).permute(0,2,1,3)
        sf = base[...,32768:].reshape(32,tiles,2,2048).permute(0,2,1,3)
        half[key] = torch.cat((w,sf),-1).contiguous().reshape(32,tiles*2,18432)
    banks['split2'] = half
    common = dict(gs13=weights['w13_gs'].float().contiguous(), gs2=weights['w2_gs'].float().contiguous(),
                  pnw=weights['post_norm_w'].contiguous(), sms=torch.cuda.get_device_properties(device).multi_processor_count,
                  counters={kind:torch.zeros(64,device=device,dtype=torch.int32) for kind in banks})
    return dict(common,banks=banks,tickets=torch.zeros(256*768,device=device,dtype=torch.int32))


def nt1_call(data, state, kind='recode', stages=2, grid=264):
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
    if kind=='split2':
        parts=torch.empty((p,2,8192),device=dev)
        tensors += [parts,state['tickets']]
    mod.run([v.data_ptr() for v in tensors], m, data['config']['local_start'], stages, grid,
            torch.cuda.current_stream(dev).cuda_stream)
    return out


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


def lab(deadline, weights):
    from concurrent.futures import ThreadPoolExecutor
    device = weights['w13'].device
    lines = ['META NT1 branch decision lab; lossless NVFP4 recode; lop26 <=124 registers; K2 four-warps split with fused reduction; cold128MiB graphs',
             'META 12 public weight/router seeds; paired identical real routes; independent FP32 <=1pct/row5pct; no claim before H200']
    choices = [('R2','recode',2,264),('RAW','raw',2,264),
               ('L26','lop26',2,264),
               ('K2s2','split2',2,528),('K2s3','split2',3,528),('K2s4','split2',4,528),
               ('R3','recode',3,264)]
    if NT1_BRANCH=='decode':choices=[c for c in choices if c[0] in ('R2','RAW','L26')]
    elif NT1_BRANCH=='occupancy':choices=[c for c in choices if c[0]!='L26']
    elif NT1_BRANCH!='both':raise ValueError(NT1_BRANCH)
    kinds=tuple(dict.fromkeys(c[1] for c in choices))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for mod in pool.map(nt1_module,kinds):pass
    rows = {(m, c[0]): [] for m in (1, 8, 16, 32) for c in choices}
    worst = {c[0]: 0. for c in choices}; flush = torch.empty(128<<20, device=device, dtype=torch.uint8)
    prev = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for i in range(12):
            if time.monotonic() > deadline-8:
                lines.append(f'BUDGET before_seed={i}'); break
            w = rg_generate_weights(5353+7919*i, device=device); w['post_norm_w'] = w['post_expert_norm']
            bank = nt1_banks(w)
            for m in (1, 8, 16, 32):
                if time.monotonic() > deadline-5:
                    break
                d = fx_make_fixture(w, m, 9000+m); want = nt1_fp32(d, w)
                graphs = {}; dispatch = []
                for label, kind, stages, requested in choices:
                    mod = nt1_module(kind); grid = min(requested, bank['sms']*mod.resident_blocks(1, stages, m))
                    dispatch.append(f'{label}:{stages}/{grid}')
                    graph = LabGraph(lambda data, k=kind,s=stages,g=requested: nt1_call(data,bank,k,s,g), d)
                    got = graph.replay(); torch.cuda.synchronize()
                    if not label.startswith('D'):
                        rel, row = nt1_error(got, want); worst[label] = max(worst[label], rel)
                        if not bool(torch.isfinite(got).all()) or rel>.01 or row>.05:
                            return lines+[f'REJECT {label} M{m} seed{i} fp32={rel:.6f}/{row:.6f}']
                    graphs[label] = graph
                if i == 0:
                    lines.append('DISPATCH M'+str(m)+' '+','.join(dispatch))
                samples = {c[0]: [] for c in choices}; ratios = {c[0]: [] for c in choices}
                a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                labels = list(graphs)
                for repeat in range(6):
                    order = labels[repeat%len(labels):]+labels[:repeat%len(labels)]
                    if repeat%2: order = order[::-1]
                    timing = {}
                    for label in order:
                        flush.zero_(); a.record(); graphs[label].replay(); b.record(); b.synchronize()
                        timing[label] = a.elapsed_time(b)*1000
                    for label in labels:
                        samples[label].append(timing[label]); ratios[label].append(timing[label]/timing['R2'])
                base = statistics.median(samples['R2'])
                vals = [base*statistics.median(ratios[c[0]]) for c in choices]
                for choice,value in zip(choices, vals): rows[m,choice[0]].append(value)
                if i<2:
                    local=d['topk_ids'][d['topk_ids']>=0];counts=torch.bincount(local.long(),minlength=32)
                    lines.append(f'S seed{i} M{m} P/E/nb8={local.numel()}/{int((counts>0).sum())}/{int(((counts+7)//8).sum())} us='+','.join(f'{v:.1f}' for v in vals))
                # Captured closures own the bank; release graph objects before next seed.
                graphs = {}; graph = None
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    for m in (1, 8, 16, 32):
        count = len(rows[m,'R2'])
        if count:
            lines.append(f'AGG M{m} seeds={count} order='+','.join(c[0] for c in choices)+' mean_us='+
                ','.join(f'{statistics.mean(rows[m,c[0]]):.2f}' if rows[m,c[0]] else '-' for c in choices))
    lines.append('GATE '+','.join(f'{k}:{v:.6f}' for k,v in worst.items() if not k.startswith('D')))
    return lines+[f'END left_s={deadline-time.monotonic():.1f}; all recipes FP32 gated; full 12-seed means required for promotion']
