"""Portable sm_120 diagnostic: compare uniform and router-derived expert fixtures.
This is evidence only; it does not submit or alter a candidate."""
import argparse
import importlib.util
import json
import time

import torch

ROOT = __file__

def load(path):
    spec = importlib.util.spec_from_file_location('routing_candidate', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def graph_us(mod, st, data, name, flush):
    def fn():
        return mod._score_expert(*data, st, name)
    fn(); fn(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    vals=[]
    for _ in range(5):
        flush.sum()
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        vals.append(a.elapsed_time(b)*1000.0)
    del graph, out
    return sorted(vals)[len(vals)//2]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('candidate'); ap.add_argument('--json', required=True)
    args=ap.parse_args(); mod=load(args.candidate)
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(args.candidate).parent))
    import reference as ref
    dev='cuda'; weights=ref.generate_weights(ref.WEIGHTS_SEED,device=dev)
    st=mod.prepare(weights)
    st['_generator']=torch.Generator(device=dev).manual_seed(478191)
    st['_deadline']=time.monotonic()+600
    flush=torch.empty(1<<28,device=dev,dtype=torch.uint8)
    result={'device':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'candidate':args.candidate,'shapes':[]}
    # These two portable implementations are representative choices in the queued portfolios.
    choices=['fused_bf16_64','raw_v11'] if 'fused_bf16_64' in getattr(mod,'_EXPERT_CHOICES',()) else ['raw11','raw_v11']
    for T in (753,3012,16384):
        syn=mod._expert_synthetic(T,st,2)
        real=mod._expert_realistic(T,st)
        row={'tokens':T}
        for label,data in (('synthetic',syn),('realistic',real)):
            counts=data[-1]
            times={}
            for name in choices:
                try: times[name]=graph_us(mod,st,data,name,flush)
                except Exception as exc: times[name]=f'{type(exc).__name__}: {str(exc)[:160]}'
            numeric={k:v for k,v in times.items() if isinstance(v,(int,float))}
            row[label]={'routes':int(counts.sum()),'distinct':int((counts>0).sum()),'counts':counts.tolist(),'us':times,
                        'selected':min(numeric,key=numeric.get) if numeric else None}
        print('ROUTING_LOCAL',json.dumps(row),flush=True); result['shapes'].append(row)
        del syn,real
        torch.cuda.empty_cache()
    with open(args.json,'w') as f: json.dump(result,f,indent=2)

if __name__=='__main__': main()
