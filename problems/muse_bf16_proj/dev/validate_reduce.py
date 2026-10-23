"""Exercise the Hopper portfolio, including fresh-input graph replay and resettable tickets.
On sm_120 the TMA pipeline uses mma.sync as the WGMMA stand-in. Oversized shared-memory
specializations use a portable ABI stand-in only for forcing prepare/dispatch glue.
"""
import importlib.util
import json
import sys
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[3]
p=root/'problems/muse_bf16_proj'
sys.path.insert(0,str(p))
import reference as ref
spec=importlib.util.spec_from_file_location('portfolio',p/'cand_reduce.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


def validate_kernels():
    weights=ref.generate_weights(ref.WEIGHTS_SEED)
    state=c.prepare(weights)
    failures=[]; checked=0; cases=[]
    limit=torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    for layer in ('router','qkv_gate','pooler','unpooler','o'):
        for m in (9,33,128,130,256):
            if m<128 and layer not in ('router','unpooler'): continue
            data=ref.generate_input(m=m,layer=layer,seed=12000+m,dist='outliers')
            x=data['x']; n=data['config']['n']; k=x.shape[1]
            tiles=(n//64)*((m+63)//64)
            scratch=(torch.empty(tiles*17*256*64,device='cuda',dtype=torch.float32),
                     torch.zeros(tiles,device='cuda',dtype=torch.uint32))
            for bt,bk,st in c.HOP_CONFIGS:
                smem=c._shared_bytes(layer,bt,bk,st)
                if smem>limit: continue
                for mode,split in ((0,1),(0,4),(0,16),(1,2),(1,8),(2,132),(2,264)):
                    if mode!=2 and split>k//bk: continue
                    cfg=(bt,bk,st,mode,split)
                    out=c._launch_hop(x,state,layer,n,cfg,scratch)
                    torch.cuda.synchronize()
                    expected=x.double()@weights[layer][:n].double().T
                    error=float((out.double()-expected).norm()/expected.norm())
                    ok,msg=ref.check_implementation(data,out,weights)
                    ok=ok and not bool(scratch[1].view(torch.int32).any())
                    if not ok:
                        failures.append((layer,m,cfg,error,int(scratch[1].view(torch.int32).max())))
                        print('FAIL',failures[-1],flush=True)
                    checked+=1
            print('kernels',layer,m,'checked',checked,'bad',len(failures),flush=True)
            cases.append((layer,m))
    # Stress the actual graph replay path with changed activations and a finite, prior output.
    for layer in ('router','qkv_gate','unpooler','o'):
        m=130; data=ref.generate_input(m=m,layer=layer,seed=7721,dist='outliers')
        x=data['x']; n=data['config']['n']; tiles=(n//64)*((m+63)//64)
        scratch=(torch.empty(tiles*17*256*64,device='cuda',dtype=torch.float32),torch.zeros(tiles,device='cuda',dtype=torch.uint32))
        for mode,split in ((0,16),(1,8),(2,132),(2,264)):
            cfg=(128,64,3,mode,split)
            c._launch_hop(x,state,layer,n,cfg,scratch)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): out=c._launch_hop(x,state,layer,n,cfg,scratch)
            for seed in range(3):
                fresh=ref.generate_input(m=m,layer=layer,seed=88000+seed,dist='outliers')
                x.copy_(fresh['x']); g.replay(); torch.cuda.synchronize()
                ok,msg=ref.check_implementation(data,out,weights)
                assert ok,(layer,cfg,seed,msg)
                if bool(scratch[1].view(torch.int32).any()): raise AssertionError('nonzero counters after replay')
                checked+=1
    assert not failures,failures
    print(json.dumps({'kernel_cases':checked,'failures':failures}),flush=True)


def validate_glue():
    weights=ref.generate_weights(ref.WEIGHTS_SEED)
    # Avoid compiling another extension; force sm_90 preparation and every candidate via its ABI.
    real=c._module()
    class Portable:
        def skinny(self,*args): return real.skinny(*args)
        def lt_options(self,*args): return real.lt_options(*args)
        def lt_run(self,*args): return real.lt_run(*args)
        def hopper(self,w,x,out,work,counts,*config):
            k=x.shape[1]
            value=torch.mm(x,w[:out.shape[1],:k].T,out_dtype=torch.float32)
            if w.shape[1]==2*k: value.add_(torch.mm(x,w[:out.shape[1],k:].T,out_dtype=torch.float32))
            out.copy_(value)
    c._MOD=Portable()
    c._hopper=lambda:True
    state=c.prepare(weights)
    # Explicitly select every high-shared-memory path and scheduler through public kernel_fn.
    selected=[]
    for layer in weights:
        n,k=weights[layer].shape
        for m in (128,256):
            data=ref.generate_input(m=m,layer=layer,seed=1917,dist='outliers')
            key=(layer,m,n,k)
            scratch=state['scratch'].get(key) or (torch.empty(1,device='cuda'),torch.zeros(1,device='cuda',dtype=torch.uint32))
            configs=list(c._variants(m,n,k,132))
            if layer not in ('router','qkv_gate'): configs += [('lt',i) for i in range(state['lt_algorithms'][key].shape[0])]
            for config in configs:
                state['choices'][key]=config; state['scratch'][key]=scratch
                out=c.kernel_fn(data,state)
                ok,msg=ref.check_implementation(data,out,weights)
                assert ok,(layer,m,config,msg)
                selected.append((layer,m,config))
    print(json.dumps({'forced_hopper_glue_cases':len(selected),'ok':True}),flush=True)

if __name__=='__main__':
    (validate_glue if sys.argv[-1]=='glue' else validate_kernels)()
