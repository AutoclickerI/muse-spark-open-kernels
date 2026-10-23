"""Verify that the fused projection graph consists of exactly one GPU kernel."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
import reference as ref
path=Path(sys.argv[1]) if len(sys.argv)>1 else P/'cand_reduce.py'
spec=importlib.util.spec_from_file_location('candidate',path);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
weights=ref.generate_weights(ref.WEIGHTS_SEED);state=c.prepare(weights)
report=[]
for layer in ('router','qkv_gate'):
    for m in (128,256):
        data=ref.generate_input(m=m,layer=layer,seed=6123)
        entries={}
        for side,fn in [('baseline',lambda:c._small(data,state)),('fused',lambda:c.kernel_fn(data,state))]:
            graph,out=c._graph(fn)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                graph.replay();torch.cuda.synchronize()
            events=[event for event in prof.events() if event.device_type==torch.autograd.DeviceType.CUDA]
            kernels=[event for event in events if not event.name.startswith(('[memory]','Memcpy','Memset'))]
            entries[side]={'count':len(kernels),'kernels':[{'name':e.name,'us':e.device_time_total} for e in kernels]}
            if side=='fused':
                assert len(kernels)==1,(layer,m,entries[side])
                ok,msg=ref.check_implementation(data,out,weights);assert ok,msg
        report.append({'layer':layer,'m':m,'choice':state['choices'][(layer,m,data['config']['n'],data['config']['k'])],**entries})
        print(json.dumps(report[-1]),flush=True)
out_path=P/'dev'/(path.stem+'_kernel_counts_local.json');out_path.write_text(json.dumps(report,indent=2)+'\n')
print('Single fused kernel verified for all four router/qkv verify shapes.',flush=True)
