"""Paired cold-cache causal comparison: doubled-K traversal versus per-tile hi+lo.
Local sm_120 only. All operands and outputs are checked; no harness state is inspected.
"""
import importlib.util
import json
from pathlib import Path
import sys
import torch
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P));import reference as ref

def load(name):
    spec=importlib.util.spec_from_file_location(name,P/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('cand_portfolio');new=load('cand_reduce_packed')
weights=ref.generate_weights(ref.WEIGHTS_SEED)
states=[]
for c in (old,new):
    state=c._prepare_small(weights)
    state['router_pair']=torch.cat([state['router_hi'],state['router_lo']],1).contiguous()
    states.append(state)
flush=torch.zeros(128*1024*1024,device='cuda',dtype=torch.uint8)
report=[]
for layer in ('router','qkv_gate'):
    for m in (128,256):
        data=ref.generate_input(m=m,layer=layer,seed=5671,dist='outliers');x=data['x'];n=data['config']['n']
        scratch=[]
        for _ in (old,new):
            scratch.append((torch.empty(m*n*17,device='cuda',dtype=torch.float32),torch.zeros((n//64)*((m+63)//64),device='cuda',dtype=torch.uint32)))
        for mode,split in ((0,8),(0,16),(1,8),(1,16),(2,132)):
            config=(64,64,4,mode,split)
            calls=[lambda:old._small(data,states[0]),
                   lambda:old._launch_hop(x,states[0],layer,n,config,scratch[0])]
            for threads in (128,160,256):
                calls.append(lambda th=threads:new._launch_hop(x,states[1],layer,n,config+(th,),scratch[1]))
            graphs=[];outputs=[]
            for call in calls:
                graph,out=old._graph(call);graphs.append(graph);outputs.append(out)
            for seed in (5671,14421,90931):
                fresh=ref.generate_input(m=m,layer=layer,seed=seed,dist='outliers');x.copy_(fresh['x'])
                for graph,out in zip(graphs,outputs):
                    graph.replay();torch.cuda.synchronize();ok,msg=ref.check_implementation(data,out,weights);assert ok,msg
            times=old._cold_times(graphs,flush,31)
            row={'layer':layer,'m':m,'config':config,'sglang_us':times[0],'concat_k_us':times[1],
                 'fused_by_threads_us':dict(zip((128,160,256),times[2:])),
                 'fused_256_vs_concat':times[1]/times[-1]}
            report.append(row);print(json.dumps(row),flush=True)
path=P/'dev/fusion_comparison_local.json';path.write_text(json.dumps(report,indent=2)+'\n')
