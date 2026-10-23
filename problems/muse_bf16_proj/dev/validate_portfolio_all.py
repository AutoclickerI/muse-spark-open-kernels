"""Force all Hopper dispatch branches of the broad portfolio through portable math.

Only this validator substitutes kernels. The submitted candidate is unmodified.
The source can also be cross-built for sm90a without executing Hopper instructions.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
import reference as ref
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--candidate',type=Path,default=P/'cand_portfolio_all.py')
ap.add_argument('--build-only',action='store_true')
ap.add_argument('--metadata-only',action='store_true',help='strict ABI coverage of every real Hopper variant, without GPU execution')
ap.add_argument('--arch',choices=('9.0a','12.0'),default='9.0a')
args=ap.parse_args()
spec=importlib.util.spec_from_file_location('all_proj',args.candidate)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
if args.build_only:
    c._hopper=lambda:args.arch=='9.0a';c._module()
    print(json.dumps({'arch':args.arch,'all_module_build':'ok'}),flush=True)
    sys.exit(0)

from validate_portfolio_hopper import exercise, load
coverage=exercise(load(args.candidate),args.candidate.resolve())
print(json.dumps({'strict_hopper_coverage':coverage}),flush=True)
if args.metadata_only:sys.exit(0)

weights=ref.generate_weights(ref.WEIGHTS_SEED)
native=c._module()

class Portable:
    def __getattr__(self,name):return getattr(native,name)

    def _product(self,w,x,out):
        k=x.shape[1]
        product=torch.mm(x,w[:out.shape[1],:k].T,out_dtype=torch.float32)
        if w.shape[1]==2*k:
            product.add_(torch.mm(x,w[:out.shape[1],k:].T,out_dtype=torch.float32))
        out.copy_(product)

    def hopper(self,w,x,out,work,counts,bt,bk,st,mode,split,threads):self._product(w,x,out)

    def lt_options(self,w,x,out):
        codes=native.lt_options(w,x,out)
        layer=next(l for l,v in weights.items() if l!='router' and v.shape==w.shape)
        old=c.H200_WINNERS.get((layer,x.shape[0]))
        count=old[1]+1 if old and old[0]=='lt' else 0
        if count>codes.shape[0]:
            assert codes.shape[0]>0
            codes=torch.cat((codes,codes[:1].repeat(count-codes.shape[0],1)),0)
        return codes

c._MOD=Portable();c._hopper=lambda:True
# This exercises prepare's H200-only incumbent setup and the exact public dispatch.
state=c.prepare(weights)
props=SimpleNamespace(shared_memory_per_block_optin=227*1024,multi_processor_count=132)
counts={};checked=0
for layer in weights:
    full_n,k=weights[layer].shape
    for m in (1,8,16,32,128,256):
        data=ref.generate_input(m=m,layer=layer,seed=80131,dist='outliers')
        n=data['config']['n'];key=(layer,m,n,k)
        variants=list(dict.fromkeys(c._all_variants(state,layer,m,n,k,props)))
        for config in variants:
            # Force every large-shared-memory Hopper path through the portable ABI.
            # Libraries, GEMVs, TC slabs and small-M kernels stay native sm120.
            state['choices'][key]=config
            out=c.kernel_fn(data,state)
            ok,msg=ref.check_implementation(data,out,weights)
            assert ok,(layer,m,config,msg)
            counts[config[0]]=counts.get(config[0],0)+1;checked+=1
        print(json.dumps({'layer':layer,'m':m,'forced_configs':len(variants)}),flush=True)
print(json.dumps({'forced_hopper_cases':checked,'families':counts,'ok':True}),flush=True)
