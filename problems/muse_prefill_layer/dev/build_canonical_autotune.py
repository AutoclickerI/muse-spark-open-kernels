"""One V6 bank, both direct readers, synthetic prepare-time selection for Hopper."""
from pathlib import Path
import ast,re
p=Path(__file__).resolve().parents[1]
base=(p/'cand_canonical_prefill.py').read_text()
group=(p/'cand_canonical_grouped.py').read_text()
tree=ast.parse(group)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GROUPED_SRC' for t in n.targets))
src=ast.literal_eval(node.value)
a=group.index('_mod = None',group.index('GROUPED_SRC ='))
b=group.index('_DQ_SLOTS =',a)
host=re.sub(r'\b_mod\b','_grouped_mod',group[a:b])
tuning='''

_experts_pipeline = experts_tiled
_prepare_pipeline = prepare


def _expert_bucket(tokens):
    return 753 if tokens < 2048 else 3012 if tokens < 8192 else 16384


def _tune_experts(st):
    # No timed data is available here. All distributions, activations and routes
    # are synthetic. Only shape-based choices survive; no graph/output is retained.
    import json, math
    dev = st['w13p'].device
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    selected = {}
    measured = {}
    for T in (753, 3012, 16384):
        gen = torch.Generator(device=dev).manual_seed(7311 + T)
        x = torch.randn((T, D_EXPERT), generator=gen, device=dev, dtype=torch.bfloat16)
        probs = torch.zeros((T,8),device=dev,dtype=torch.float32)
        probs[:,0] = 1.
        sample = {}
        for distribution in ('flat','hot','busy'):
            e = torch.full((T,8), -1, device=dev, dtype=torch.int32)
            pos = torch.zeros_like(e)
            r = torch.arange(T,device=dev,dtype=torch.int32)
            counts = torch.zeros(32,device=dev,dtype=torch.int32)
            if distribution == 'flat':
                e[:,0] = r % 32
                pos[:,0] = r // 32
                counts[:] = T//32 + (torch.arange(32,device=dev) < T%32).int()
            elif distribution == 'busy':
                routes = r[:,None]*4 + torch.arange(4,device=dev,dtype=torch.int32)[None,:]
                e[:,:4] = routes % 32
                pos[:,:4] = routes // 32
                counts[:] = (T*4)//32 + (torch.arange(32,device=dev) < (T*4)%32).int()
                probs.zero_();probs[:,:4] = .25
            else:
                e[:,0] = 0
                pos[:,0] = r
                counts[0] = T
            for name,fn in (('pipeline',_experts_pipeline),('grouped',experts_grouped)):
                for _ in range(2):
                    out=fn(x,probs,e,pos,counts,st)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out=fn(x,probs,e,pos,counts,st)
                start=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                times=[]
                for _ in range(3):
                    flush.fill_(1)
                    start.record();graph.replay();end.record();end.synchronize()
                    times.append(start.elapsed_time(end)*1000.)
                sample[name+':'+distribution]=sorted(times)[1]
                if name == 'pipeline':
                    reference_out=out.clone()
                else:
                    err=float((out.float()-reference_out.float()).norm()/reference_out.float().norm().clamp_min(1e-12))
                    if not math.isfinite(err) or err > .01:
                        raise RuntimeError(f'Canonical readers disagree at synthetic T={T}, {distribution}: {err}')
                    sample['relative_l2:'+distribution]=err
                    del reference_out
                del graph,out
        costs={name:math.prod(sample[name+':'+d] for d in ('flat','hot','busy'))**(1./3.) for name in ('pipeline','grouped')}
        selected[T]=min(costs,key=costs.get)
        measured[T]=dict(us=sample,selected=selected[T])
        print('CANONICAL_EXPERT_TUNE',json.dumps(dict(tokens=T,**measured[T]),sort_keys=True),flush=True)
    st['expert_dispatch']=selected
    return measured


def prepare(weights):
    st=_prepare_pipeline(weights)
    st['grouped_mod']=_grouped_module()
    st['expert_dispatch']={753:'grouped',3012:'grouped',16384:'grouped'}
    if IS_HOPPER:
        _tune_experts(st)
    return st


def experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st):
    name=st['expert_dispatch'][_expert_bucket(pooled.shape[0])]
    fn={'pipeline':_experts_pipeline,'grouped':experts_grouped}[name]
    return fn(pooled,topk_w,slot_e,slot_pos,counts,st)
'''
(p/'cand_canonical_autotune.py').write_text(base+'\nGROUPED_SRC = '+repr(src)+'\n'+host+tuning)
print(p/'cand_canonical_autotune.py')
