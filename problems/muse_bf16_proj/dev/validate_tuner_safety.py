"""Fault injection for generated all/max/native tuning code, without judge sends."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import time

import torch

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'dev'))
from build_portfolio_all import build,safe_tuner

for file,extended in (('cand_portfolio_all.py',False),('cand_portfolio_max.py',True),('cand_portfolio_native.py',True)):
    source=safe_tuner((P/file).read_text(),extended=extended,native=file=='cand_portfolio_native.py')
    compile(source,file,'exec')
    assert 'deadline=started+240' in source and 'ALL_REJECT_FINAL' in source

with tempfile.TemporaryDirectory(prefix='projection_tuner_test_') as tmp:
    path=Path(tmp)/'portfolio.py';build(path)
    spec=importlib.util.spec_from_file_location('safe_projection_generated',path)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class Module:
    algo_bytes=16
    def all_options(self,w,x,out,nn,kind,workspace_bytes):
        # A failed layout query must leave the remaining trials eligible.
        if nn:raise ValueError('injected layout failure')
        return torch.zeros((1,16),dtype=torch.uint8),1


device='cuda'
weights={name:torch.randn((1024 if name=='unpooler' else 256 if name=='router' else 64,128),device=device,
                         dtype=torch.float32 if name=='router' else torch.bfloat16)*.01
         for name in ('qkv_gate','o','pooler','unpooler','router')}
def base(weights):
    router=weights['router'];hi=router.to(torch.bfloat16);lo=(router-hi.float()).to(torch.bfloat16)
    return {**weights,'mod':Module(),'router_hi':hi,'router_lo':lo,'router_hi_t':hi.T.contiguous(),'router_lo_t':lo.T.contiguous()}
c._prepare_small=base;c._hopper=lambda:False
def run(data,state,config):
    x=data['x'];name=data['config']['layer'];n=data['config']['n']
    y=(x.float()@weights[name][:n].float().T)
    if name!='router':y=y.to(torch.bfloat16)
    if config[0]=='broken':raise ValueError('injected launch failure')
    if config[0]=='wrong':return torch.zeros_like(y)
    if config[0]=='late_wrong' and float(x[0,0])==0:return torch.zeros_like(y)
    return y
c._all_run=run
c._all_variants=lambda *args:iter((('incumbent',),('broken',),('wrong',),('late_wrong',),('good',)))
class Graph:
    def __init__(self,fn):self.fn=fn;self.out=fn()
    def replay(self):self.out.copy_(self.fn())
def graph(fn):
    g=Graph(fn);return g,g.out
c._graph=graph
c._all_measure=lambda *args:[10,5]
log=io.StringIO()
with contextlib.redirect_stdout(log):state=c.prepare(weights)
text=log.getvalue();print(text)
assert text.count('ALL_REJECT_FINAL')==30
assert 'injected launch failure' in text and 'ALL incorrect' in text
assert text.count('injected layout failure')==36
assert len(state['choices'])==30
assert all(v[0] in ('incumbent','good') for v in state['choices'].values())

# Consume the 240 s budget inside base preparation. No trial or per-shape
# synthetic activation should be allocated once the deadline has passed.
real_time=time.monotonic;shift=[0]
def slow_base(weights):
    result=base(weights);shift[0]=241;return result
c._prepare_small=slow_base
time.monotonic=lambda:real_time()+shift[0]
try:
    with contextlib.redirect_stdout(log):expired=c.prepare(weights)
finally:time.monotonic=real_time
assert not expired['choices']
print('PASS generated all/max/native syntax; 30 coarse launch/numeric/finalist failures rejected; expired deadline skips all 30 shapes')
