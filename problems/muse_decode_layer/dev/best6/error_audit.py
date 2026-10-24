"""Inject optional RuntimeErrors at admission, cold replay and final replay gates."""
from pathlib import Path
import importlib.util,sys,time,json,hashlib
import torch
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer')]
import local_eval

def main():
    path=ROOT/'problems/muse_decode_layer/cand_r3_best6.py'
    m=local_eval.load_module('best6_error_audit',path)
    ref=local_eval.load_module('reference',path.with_name('reference.py'))
    weights=ref.generate_weights(ref.WEIGHTS_SEED);state=m.prepare(weights)
    previous=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    fixtures={};wants={}
    for n in (128,256):
        for seed in (9000+n,9000+n+7919,9000+n+15838):
            data,route=m._best6_fixture(weights,n,seed)
            fixtures[n,seed]=(data,route);wants[id(data)]=m._best5_fp32(data,weights)
    original_fixture=m._best6_fixture;original_oracle=m._best5_fp32;original_call=m._best6_call;original_graph=m._best5_graph
    m._best6_fixture=lambda weights,n,seed:fixtures[n,seed]
    m._best5_fp32=lambda data,weights:wants[id(data)]
    m._BEST5_DEADLINE=time.monotonic()+200
    flush=torch.empty(64<<20,device='cuda',dtype=torch.uint8);rows=[]
    try:
        for target in ('raw17','humming_masked','dg_w8'):
            for phase,fail_at in (('admission',0),('cold_replay',4),('post_timing_gate',16)):
                message='shape_m is not within any range: injected '+target+' '+phase
                def call(data,st,name):
                    if name==target:
                        if fail_at==0:raise RuntimeError(message)
                        # A real V9 computation stands in for the optional backend
                        # only to exercise failure handling; this is not a speed test.
                        return original_call(data,st,'v9nt1')
                    return original_call(data,st,name)
                class Replay:
                    def __init__(self,graph):self.graph=graph;self.calls=0
                    def replay(self):
                        self.calls+=1
                        if self.calls==fail_at:raise RuntimeError(message)
                        return self.graph.replay()
                def graph(fn,data):
                    static,captured,out=original_graph(fn,data)
                    if fn.__defaults__ and fn.__defaults__[0]==target:captured=Replay(captured)
                    return static,captured,out
                m._best6_call=call;m._best5_graph=graph
                st=dict(state,moe6_adapters=dict(state['moe6_adapters'],available=[target]))
                m._best6_tune(st,weights,flush,torch.device('cuda'))
                for n,row in st['moe6_rows'].items():
                    assert row['reason']=='measured',row
                    assert row['selected'].startswith('v9nt'),row
                    assert message in row['rejected'][target],row
                    rows.append(dict(target=target,phase=phase,m=n,selected=row['selected'],rejected=row['rejected'][target]))
        # All three builder failures leave an empty optional set and a valid V9 path.
        m._BEST6_GROUPED_ERROR='injected raw17 compiler RuntimeError'
        m._best6_hopper=lambda device:True
        def unavailable(*a,**kw):raise RuntimeError('shape_m is not within any range: injected build')
        m._pf_repack_v6=unavailable;m._best6_humming=unavailable
        rejected=m._best6_build_adapters(weights)
        assert rejected['available']==[] and set(rejected['rejected'])=={'raw17','dg_w8','humming_masked'},rejected
        # Expired portfolio setup returns immediately to the measured V9 defaults.
        m._BEST5_DEADLINE=time.monotonic()-1;expired=dict(state)
        m._best6_tune(expired,weights,flush,torch.device('cuda'))
        assert all(n.startswith('v9nt') for n in expired['moe6_plan'].values())
        result=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=rows,build_rejections=rejected['rejected'],expired_plan=expired['moe6_plan'])
        (HERE/'error_audit_local.json').write_text(json.dumps(result,indent=2)+'\n')
        print('BEST6_ERROR_AUDIT',len(rows),'passing injected RuntimeErrors; all builder errors isolated; expired default valid',flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous
        m._best6_fixture=original_fixture;m._best5_fp32=original_oracle;m._best6_call=original_call;m._best5_graph=original_graph
if __name__=='__main__':main()
