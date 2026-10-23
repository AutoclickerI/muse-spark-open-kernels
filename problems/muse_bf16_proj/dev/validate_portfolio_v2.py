"""Validate v2 assembly and every dispatch ABI without inventing an H200 receipt.

With a candidate argument, check its actual receipt-derived configuration. Without
one, use explicitly local control configurations to exercise the assembly pipeline.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch

P=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(P),str(P/'dev')]
import build_portfolio_v2 as builder
import reference as ref

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('candidate',nargs='?',type=Path)
ap.add_argument('--build-only',action='store_true')
ap.add_argument('--arch',choices=('12.0','9.0a'),default='12.0')
args=ap.parse_args()
if args.candidate:
    spec=importlib.util.spec_from_file_location('v2',args.candidate)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    provenance=str(args.candidate)
else:
    controls={(layer,m):None for layer in builder.LAYERS for m in (128,256)}
    controls[('router',128)]=(64,64,4,0,8)
    controls[('qkv_gate',256)]=(64,64,4,1,8)
    controls[('pooler',128)]=('lt',0)
    controls[('unpooler',256)]=('lt',0)
    source,counts=builder.assemble(controls,{'job':'LOCAL_ASSEMBLY_CONTROL'})
    c=types.ModuleType('v2');exec(compile(source,'<local-v2-assembly>','exec'),c.__dict__)
    provenance='LOCAL_ASSEMBLY_CONTROL'
    print(json.dumps({'provenance':provenance,'specializations':counts}),flush=True)

if args.build_only:
    c._hopper=lambda:args.arch=='9.0a'
    c._module()
    print(json.dumps({'provenance':provenance,'arch':args.arch,'build':'ok'}),flush=True)
    sys.exit(0)

weights=ref.generate_weights(ref.WEIGHTS_SEED)
state=c.prepare(weights)
native=c._module()
cases=0
for (layer,m),row in c.FUSED_CONFIGS.items():
    data=ref.generate_input(m=m,layer=layer,seed=9191,dist='outliers')
    x=data['x'];n=data['config']['n'];key=(layer,m,n,x.shape[1])
    configs=[('pair',)+tuple(cfg) for cfg in row
             if c._v2_shared_bytes(layer,cfg)<=torch.cuda.get_device_properties(0).shared_memory_per_block_optin]
    # Native assembly/ABI checks use each new path; sm90-specific incumbents below
    # are forced through a stand-in before any launch on this sm120 device.
    for config in configs:
        scratch=state['scratch'].get(key) or (
            torch.empty(m*n*17,device='cuda',dtype=torch.float32),
            torch.zeros((n//64)*((m+63)//64),device='cuda',dtype=torch.uint32))
        graph,out=c._graph(lambda:c._v2_run(data,state,config,scratch))
        for seed in (9191,71979,32789):
            fresh=ref.generate_input(m=m,layer=layer,seed=seed,dist='outliers')
            x.copy_(fresh['x']);graph.replay();torch.cuda.synchronize()
            ok,msg=ref.check_implementation(data,out,weights)
            assert ok,(layer,m,config,msg)
            cases+=1
print(json.dumps({'v2_native_fused_cases':cases,'provenance':provenance,'ok':True}),flush=True)

class Portable:
    def __getattr__(self,name):return getattr(native,name)

    def lt_options(self,w,x,out):
        codes=native.lt_options(w,x,out)
        layer=next(layer for layer,value in weights.items() if layer!='router' and value.shape==w.shape)
        old=c.H200_WINNERS[(layer,x.shape[0])]
        needed=old[1]+1 if old and old[0]=='lt' else 0
        # Heuristic ordinals differ across devices. Duplicate valid sm120 codes
        # only inside this glue stand-in, while retaining the exact H200 ordinal.
        if needed>codes.shape[0]:
            assert codes.shape[0]>0,'no portable Lt algorithm'
            codes=torch.cat((codes,codes[:1].repeat(needed-codes.shape[0],1)),0)
        return codes

    def product(self,w,x,out):
        k=x.shape[1]
        value=torch.mm(x,w[:out.shape[1],:k].T,out_dtype=torch.float32)
        if w.shape[1]==2*k:
            value.add_(torch.mm(x,w[:out.shape[1],k:].T,out_dtype=torch.float32))
        out.copy_(value)

    def hopper(self,w,x,out,work,counts,*config):self.product(w,x,out)
    def old_hopper(self,w,x,out,work,counts,*config):self.product(w,x,out)

c._MOD=Portable();c._hopper=lambda:True
state=c.prepare(weights)
cases=0
for layer in weights:
    n,k=weights[layer].shape
    for m in (128,256):
        data=ref.generate_input(m=m,layer=layer,seed=2331,dist='outliers')
        key=(layer,m,n,k)
        configs=[c.H200_WINNERS[(layer,m)],None]
        configs += [('pair',)+tuple(cfg) for cfg in c.FUSED_CONFIGS.get((layer,m),())]
        if layer!='router':
            configs += [('lt_nn',i) for i in range(state['lt_nn_algorithms'][key].shape[0])]
        scratch=(torch.empty(m*n*17,device='cuda',dtype=torch.float32),
                 torch.zeros((n//64)*((m+63)//64),device='cuda',dtype=torch.uint32))
        for config in configs:
            state['choices'][key]=config;state['scratch'][key]=scratch
            out=c.kernel_fn(data,state)
            ok,msg=ref.check_implementation(data,out,weights)
            assert ok,(layer,m,config,msg)
            cases+=1
print(json.dumps({'v2_forced_hopper_glue_cases':cases,'provenance':provenance,'ok':True}),flush=True)
