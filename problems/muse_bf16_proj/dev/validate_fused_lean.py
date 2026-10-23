"""Check the lean host ABI, pruned kernels, CUDA graphs and forced Hopper dispatch."""
import importlib.util,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P));import reference as ref
candidate=Path(sys.argv[1]) if len(sys.argv)>1 else P/'cand_fused_lean.py'
spec=importlib.util.spec_from_file_location('c',candidate);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
weights=ref.generate_weights(ref.WEIGHTS_SEED);state=c.prepare(weights);cases=0
for (layer,tuned_m),configs in c.FUSED_CONFIGS.items():
    for cfg in configs:
        bt,bk,st,*_=cfg
        if c._shared_bytes(layer,bt,bk,st)>torch.cuda.get_device_properties(0).shared_memory_per_block_optin:continue
        for m in (9,33,128,130,256):
            data=ref.generate_input(m=m,layer=layer,seed=9877,dist='outliers');x=data['x'];n=data['config']['n']
            tiles=(n//64)*((m+bt-1)//bt)
            scratch=(torch.empty(tiles*17*bt*64,device='cuda',dtype=torch.float32),torch.zeros(tiles,device='cuda',dtype=torch.uint32))
            graph,out=c._graph(lambda:c._launch_hop(x,state,layer,n,cfg,scratch))
            for seed in (9877,7891,38791):
                fresh=ref.generate_input(m=m,layer=layer,seed=seed,dist='outliers');x.copy_(fresh['x']);graph.replay();torch.cuda.synchronize()
                ok,msg=ref.check_implementation(data,out,weights);assert ok,(layer,m,cfg,msg)
                assert not bool(scratch[1].view(torch.int32).any()),'counter reset failed'
                cases+=1
# Both Lt weight orientations, all returned algorithm codes, including qkv for v2 incumbents.
for layer in ('qkv_gate','pooler','unpooler','o'):
    for m in (128,256):
        data=ref.generate_input(m=m,layer=layer,seed=3131,dist='outliers');x=data['x'];n=data['config']['n'];native=state['mod']
        for nn in (False,True):
            w=weights[layer].T.contiguous() if nn else weights[layer]
            out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
            codes=(native.lt_nn_options if nn else native.lt_options)(w,x,out)
            for i in range(codes.shape[0]):
                run=native.lt_nn_run if nn else native.lt_run
                graph,out=c._graph(lambda:(run(w,x,out,state['lt_workspace'],codes,i),out)[1])
                ok,msg=ref.check_implementation(data,out,weights);assert ok,(layer,m,nn,i,msg);cases+=1
print(json.dumps({'lean_native_kernel_and_lt_cases':cases,'ok':True}),flush=True)
# Force Hopper's Python branch through a portable stand-in with this exact raw ABI.
real=c._module()
class Portable:
    def __getattr__(self,name):return getattr(real,name)
    def hopper(self,w,x,out,work,counts,*cfg):
        k=x.shape[1];value=torch.mm(x,w[:out.shape[1],:k].T,out_dtype=torch.float32)
        if w.shape[1]==2*k:value.add_(torch.mm(x,w[:out.shape[1],k:].T,out_dtype=torch.float32))
        out.copy_(value)
c._MOD=Portable();c._hopper=lambda:True
state=c.prepare(weights);glue=0
for layer in weights:
    n,k=weights[layer].shape
    for m in (128,256):
        data=ref.generate_input(m=m,layer=layer,seed=7117,dist='outliers');key=(layer,m,n,k)
        configs=list(c.FUSED_CONFIGS.get((layer,m),()))
        if layer not in ('router','qkv_gate'):
            configs += [('lt',i) for i in range(state['lt_algorithms'][key].shape[0])]
            configs += [('lt_nn',i) for i in range(state['lt_nn_algorithms'][key].shape[0])]
        for cfg in configs:
            state['choices'][key]=cfg;state['scratch'][key]=(torch.empty(1,device='cuda'),torch.zeros(1,device='cuda',dtype=torch.uint32))
            out=c.kernel_fn(data,state);ok,msg=ref.check_implementation(data,out,weights);assert ok,(layer,m,cfg,msg);glue+=1
print(json.dumps({'lean_forced_hopper_glue_cases':glue,'ok':True}),flush=True)
