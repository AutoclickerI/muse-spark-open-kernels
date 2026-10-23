# Original board-best entry points remain the scored incumbent.
import math
import statistics
import time

_tg_incumbent_prepare = prepare
_tg_incumbent_kernel = kernel_fn
_TG_MOD = None
TG_CHOICES = tuple((w,s,f,n) for w,s in ((32,3),(64,3),(64,5),(32,5))
                   for f,n in ((1,8),(1,16),(.5,8)))


def _tg_cuda_flags(capability=None):
    capability = torch.cuda.get_device_capability() if capability is None else capability
    flags = ['-O3','-std=c++20','-lineinfo']
    if capability==(9,0):
        flags += ['-gencode=arch=compute_90a,code=sm_90a']
    return flags


def _tg_module():
    global _TG_MOD
    if _TG_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        declarations = '''#include <torch/extension.h>
int64_t tg_capacity(int64_t,int64_t,int64_t);
void tg_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 int64_t,int64_t,int64_t,int64_t,int64_t);
'''
        tag = hashlib.sha256(TG_CUDA.encode()).hexdigest()[:12]
        _TG_MOD = load_inline(name='muse_tma_grouped_'+tag,cpp_sources=declarations,
            cuda_sources=TG_CUDA,functions=['tg_capacity','tg_run'],
            extra_cuda_cflags=_tg_cuda_flags(),verbose=False)
    return _TG_MOD


def _tg_prepare_bank(weights):
    bank = {'mod':_tg_module()}
    dev = weights['w13'].device
    for name,gated in (('w13',True),('w2',False)):
        b,s = repack_fast(weights[name][:32],weights[name+'_sf'][:32],gated)
        # Raw17 has four m64 blocks per tile. This design has one consumer WG
        # processing two m64 blocks: gate/up or adjacent down-projection features.
        e,f,k = b.shape[:3]
        bank[name] = b.view(e,f,k,2,2,128,16).permute(0,1,3,2,4,5,6).contiguous()
        bank[name+'_sf'] = s.view(e,f,k,2,2,32,4,2).permute(0,1,3,2,4,5,6,7).contiguous()
    bank.update(gs13=weights['w13_gs'][:32].float().contiguous(),
        gs2=weights['w2_gs'][:32].float().contiguous(),pnw=weights['post_norm_w'].contiguous(),
        counter=torch.zeros(2,device=dev,dtype=torch.int32),
        sms=torch.cuda.get_device_properties(dev).multi_processor_count,grids={})
    return bank


def _tg_configure(bank,m,choice):
    width,stages,factor,min_n = choice
    resident = int(bank['mod'].tg_capacity(m,width,stages))
    if resident<1:
        raise RuntimeError('no resident grouped CTA')
    # At most one block per SM; capacity is checked on the actual specialization.
    grid = max(1,int(bank['sms']*factor))
    if grid > bank['sms']*resident:
        raise RuntimeError('persistent grid exceeds resident capacity')
    bank['grids'][(m,choice)] = grid


def _tg_run(data,bank,choice):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    tw = data['topk_weights'].float().contiguous()
    m = x.shape[0]
    if (m,choice) not in bank['grids']:
        _tg_configure(bank,m,choice)
    grid = bank['grids'][(m,choice)]
    width,stages,factor,min_n = choice
    dev = x.device
    inter = torch.empty((m*8,4096),dtype=torch.bfloat16,device=dev)
    y = torch.empty((m*8,4096),dtype=torch.float32,device=dev)
    out = torch.empty_like(x)
    bank['mod'].tg_run(x,ids,tw,bank['pnw'],bank['w13'],bank['w13_sf'],bank['w2'],bank['w2_sf'],
        bank['gs13'],bank['gs2'],inter,y,bank['counter'],out,
        int(data['config'].get('local_start',0)),width,stages,grid,min_n)
    return out


def _tg_router_weights(weights):
    if all(k in weights for k in ('router','router_bias','pooler','pre_ffn_norm','pre_expert_norm')):
        return weights
    # Standalone MoE supplies no router. Generate a full router weight fixture with
    # the decode model's mean alignment, bias, low-rank stream and norm statistics.
    dev = weights['w13'].device
    gen = torch.Generator(device=dev).manual_seed(5353)
    mean,_ = _basis(dev)
    norm = (.1*torch.randn(8192,generator=gen,device=dev)).bfloat16()
    router = .045*torch.randn(256,8192,generator=gen,device=dev)
    router += .07*torch.randn(256,1,generator=gen,device=dev)*mean/math.sqrt(8192)
    return dict(router=router,router_bias=.055*torch.randn(256,generator=gen,device=dev),
        pre_ffn_norm=norm,pooler=(torch.randn(4096,8192,generator=gen,device=dev)/math.sqrt(8192)).bfloat16(),
        pre_expert_norm=(.1*torch.randn(4096,generator=gen,device=dev)).bfloat16())


def _tg_dequant(q,s,gs):
    # Independent FP32 self-reference. No invocation of the problem reference.
    codes = torch.stack((q&15,q>>4),-1).flatten(-2).long()
    lut = torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=q.device)
    v = lut[codes&7]*torch.where((codes&8)!=0,-1.,1.)
    return v*s.float().repeat_interleave(16,-1)*gs.float()


def _tg_reference(data,weights):
    x = data['hidden_states'].float()
    out = torch.zeros_like(x)
    start = int(data['config'].get('local_start',0))
    ids = data['topk_ids']
    gain = 1+weights['post_norm_w'].float()
    for e in range(32):
        tok,slot = torch.where(ids==e+start)
        if not tok.numel():
            continue
        h = x[tok]@_tg_dequant(weights['w13'][e],weights['w13_sf'][e],weights['w13_gs'][e]).T
        a = torch.nn.functional.silu(h[:,:4096])*h[:,4096:]
        y = a@_tg_dequant(weights['w2'][e],weights['w2_sf'][e],weights['w2_gs'][e]).T
        y = y*gain
        y = y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-8)
        out.index_add_(0,tok,y*data['topk_weights'][tok,slot,None])
    return out.to(torch.bfloat16)


def _tg_check(out,ref):
    if out.shape!=ref.shape or out.dtype!=torch.bfloat16:
        raise ValueError('wrong shape/dtype')
    if not bool(torch.isfinite(out).all()):
        raise ValueError('nonfinite')
    delta = (out.float()-ref.float()).norm(dim=-1)
    scale = ref.float().norm(dim=-1)
    rel = float(delta.norm()/scale.norm().clamp_min(1e-12))
    row = float((delta/scale.clamp_min(.1*scale.mean()).clamp_min(1e-12)).max())
    # 0.8 x the official 1.5% global / 7.5% per-row bound.
    if rel>.012 or row>.060:
        raise ValueError(f'FP32 gate rel={rel:.5g} row={row:.5g}')
    return rel,row


class _TgGraph:
    def __init__(self,fn,data):
        self.data = {k:(v.clone() if torch.is_tensor(v) else v) for k,v in data.items()
                     if k in ('hidden_states','topk_ids','topk_weights','config')}
        fn(self.data)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn(self.data)
    def refresh(self,data):
        for k in ('hidden_states','topk_ids','topk_weights'):
            self.data[k].copy_(data[k])
    def replay(self):
        self.graph.replay()
        return self.out


def _tg_pair(old,new,datasets,flush,repeats=20,deadline=float('inf')):
    old_times,new_times,ratios = [],[],[]
    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for i in range(repeats):
        if i and time.monotonic()>deadline:
            break
        d = datasets[i%len(datasets)]
        old.refresh(d);new.refresh(d)
        pair = []
        for graph in ((old,new) if i%2==0 else (new,old)):
            flush.sum()
            a.record();graph.replay();b.record();b.synchronize()
            pair.append(float(a.elapsed_time(b))*1000)
        t0,t1 = pair if i%2==0 else pair[::-1]
        old_times.append(t0);new_times.append(t1)
        ratios.append(t1/t0)
    if not ratios:raise RuntimeError('no samples')
    return statistics.median(old_times),statistics.median(new_times),statistics.median(ratios),len(ratios)


def _tg_reject(summary,where,exc):
    summary.append(f'TG_REJECT {where} {type(exc).__name__} {str(exc).splitlines()[0][:130]}')


def _tg_tune(weights,st,started):
    deadline = started+230
    summary = st['tg_summary']
    rw = _tg_router_weights(weights)
    bank = st['tg_bank']
    flush = torch.zeros((128<<20)//4,dtype=torch.float32,device=weights['w13'].device)
    for shape_index,m in enumerate((128,256)):
        if time.monotonic()>deadline:
            summary.append(f'TG_BUDGET M={m} incumbent');continue
        shape_deadline = min(deadline,time.monotonic()+(deadline-time.monotonic())/(2-shape_index))
        try:
            datasets = [make_fixture(rw,m,9000+7919*j+m,mode='real') for j in range(3)]
            refs = [_tg_reference(d,weights) for d in datasets]
            info = route_summary(datasets[0])
            summary.append(f'TG_ROUTE M={m} {info}')
            # The incumbent is preserved even if a self-reference diagnostic fails.
            old = _TgGraph(lambda d:_tg_incumbent_kernel(d,st),datasets[0])
            for d,ref in zip(datasets,refs):
                old.refresh(d);_tg_check(old.replay(),ref)
        except Exception as exc:
            _tg_reject(summary,f'incumbent M={m}',exc);continue
        candidates = []
        for choice in TG_CHOICES:
            if time.monotonic()>shape_deadline:
                summary.append(f'TG_BUDGET M={m}');break
            try:
                _tg_configure(bank,m,choice)
                new = _TgGraph(lambda d,c=choice:_tg_run(d,bank,c),datasets[0])
                error = 0.
                for d,ref in zip(datasets,refs):
                    new.refresh(d);error=max(error,_tg_check(new.replay(),ref)[0])
                t0,t1,ratio,count = _tg_pair(old,new,datasets,flush,repeats=4,deadline=shape_deadline)
                summary.append(f'TG_TRIAL M={m} c={choice} us={t0:.2f}/{t1:.2f} ratio={ratio:.4f} rel={error:.5f}')
                candidates.append((ratio,choice))
                del new
            except Exception as exc:
                _tg_reject(summary,f'M={m} c={choice}',exc)
        best = 1.
        for _,choice in sorted(candidates)[:3]:
            if time.monotonic()>shape_deadline:break
            try:
                new = _TgGraph(lambda d,c=choice:_tg_run(d,bank,c),datasets[0])
                t0,t1,ratio,count = _tg_pair(old,new,datasets,flush,deadline=shape_deadline)
                for d,ref in zip(datasets,refs):
                    new.refresh(d);_tg_check(new.replay(),ref)
                summary.append(f'TG_FINAL M={m} c={choice} us={t0:.2f}/{t1:.2f} ratio={ratio:.4f} n={count}')
                if count==20 and ratio<best:
                    st['tg_dispatch'][m]=choice;best=ratio
                del new
            except Exception as exc:
                _tg_reject(summary,f'final M={m} c={choice}',exc)
        summary.append(f'TG_PICK M={m} {st["tg_dispatch"].get(m,"v9_phase")} ratio={best:.4f}')
        del old,datasets,refs


def prepare(weights):
    started = time.monotonic()
    # The incumbent's original preparation/dispatch is independent of optional trials.
    st = _tg_incumbent_prepare(weights)
    st.update(tg_dispatch={},tg_summary=[])
    try:
        st['tg_bank'] = _tg_prepare_bank(weights)
        _tg_tune(weights,st,started)
    except Exception as exc:
        _tg_reject(st['tg_summary'],'optional preparation',exc)
        st['tg_dispatch'].clear()
    print('TG_BEGIN',flush=True)
    for line in st['tg_summary']:print(line,flush=True)
    print(f'TG_END prepare_s={time.monotonic()-started:.3f} picks={st["tg_dispatch"]}',flush=True)
    return st


def kernel_fn(data,state):
    m = data['hidden_states'].shape[0]
    choice = state['tg_dispatch'].get(m)
    if choice is None:
        return _tg_incumbent_kernel(data,state)
    return _tg_run(data,state['tg_bank'],choice)
