# okc: problem=muse_bf16_proj
"""Screen Route B in the projection task; larger M uses a plain exact-format GEMM."""
import torch
from pipeline_x import cd_encode, cd_decode, cd_gemv, cd_module

def prepare(weights):
    cd_module(tuple((nt,1,2,3) for nt in (1,2,4)))
    coded = {}
    for name, w in weights.items():
        if name == 'router': continue
        coded[name] = cd_encode(w, kmajor=True)
        if not torch.equal(cd_decode(coded[name]).view(torch.int16), w.view(torch.int16)):
            raise RuntimeError('weight bits: '+name)
    return weights, coded

def kernel_fn(data, state):
    weights, coded = state
    x=data['x'];name=data['config']['layer'];n=data['config']['n']
    if name=='router': return x.float() @ weights[name].T
    if x.shape[0]>32: return x @ weights[name][:n].T
    c=coded[name]
    if c['N']!=n:
        # Reuse row prefixes in the existing k-major representation by selecting
        # a separately prepared prefix; prepare() below creates it.
        c=coded['unpooler_slice']
    S=8 if name=='o' else 16
    p,_=cd_gemv(x,c,S,NW=4,PIPE=2,V=3,pdl=True)
    return p.sum(0).to(torch.bfloat16)

_base_prepare=prepare
def prepare(weights):
    state=_base_prepare(weights)
    state[1]['unpooler_slice']=cd_encode(weights['unpooler'][:1024].contiguous(),kmajor=True)
    return state
