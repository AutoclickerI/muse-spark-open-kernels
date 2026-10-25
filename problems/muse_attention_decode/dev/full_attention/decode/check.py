"""Every arm: independent math probe, TASK cache/output gates, fresh capture, clocks and partial ABI."""
import argparse
import concurrent.futures
import hashlib
import json
import linecache
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('names',nargs='*');a=ap.parse_args()
manifest=json.loads((H/'manifest.json').read_text())['variants'];names=a.names or [r['name'] for r in manifest]
src=(H/'inputs.py').read_text();ns={};exec(compile(src,'<decode-inputs>','exec'),ns)
mods={}
for name in names:
 s=(H/(name+'.py')).read_text();f=str(H/(name+'.py'));linecache.cache[f]=(len(s),None,s.splitlines(True),f);m={};exec(compile(s,f,'exec'),m);mods[name]=m
with concurrent.futures.ThreadPoolExecutor(4) as pool:
 fs=[pool.submit(m['ext'],c) for m in mods.values() for c in (False,True)]
 for f in fs:f.result()
def probe(m):
 if 'CUDA' not in m:return dict(native_WG_standin=True)
 gen=torch.Generator(device='cuda').manual_seed(991)
 p=torch.rand((16,32),device='cuda',generator=gen);sums=torch.empty((16,8),device='cuda')
 x=torch.linspace(-125,0,65537,device='cuda');y=torch.empty_like(x)
 m['ext']().decode_probes(p.data_ptr(),sums.data_ptr(),x.data_ptr(),y.data_ptr(),x.numel(),torch.cuda.current_stream().cuda_stream)
 want=p.half().double().sum(-1)[:,None].expand(16,8)
 err=float((sums.double()-want).abs().max());exp=float(((y-torch.exp2(x)).abs()/torch.exp2(x)).max())
 assert err<2e-5 and exp<.0002,(err,exp)
 return dict(half_P_row_sum_max_abs=err,exp_max_relative=exp)
proof=[]
for name,m in mods.items():
 row=dict(name=name,driver_sha256=hashlib.sha256((H/(name+'.py')).read_bytes()).hexdigest(),probe=probe(m),fixtures=[]);print('PROBE PASS',name,row['probe'],flush=True)
 for B,ctx,page,scale in [(2,257,64,1/64),(16,4096,1,1),(16,80000,1,1),(32,80000,1,1)]:
  for seed in (61791,92003):
   d=ns['generate_input'](batch=B,ctx=ctx,heads='tp8',layer='full',q_len=1,page=page,kv_scale=scale,dist='coherent',spread='agentic',seed=seed)
   out=m['attention'](d);want=ns['gate_prepare'](d);ok,msg,err=ns['gate_cached'](d,out,want);assert ok,(name,B,ctx,seed,msg)
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):z=m['attention'](d)
   for _ in range(3):g.replay()
   torch.cuda.synchronize();ok,msg,_=ns['gate_cached'](d,z,want);assert ok,(name,msg)
   clocks=[];clock=m['attention'](d,clocks=clocks);torch.cuda.synchronize();assert torch.equal(clock,out),(name,'plain-clock mismatch')
   raw=d['qkv'].view(B,-1);buf=torch.randn((3,B+1,raw.shape[-1]),device='cuda')*.03;part=buf[:,:B];part[-1]+=raw.float()-part.sum(0)
   pg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(pg):po=m['attention'](d,qkv=(part,3))
   pg.replay();torch.cuda.synchronize();ok,msg,pe=ns['gate_cached'](d,po,want);assert ok,(name,'partial',msg)
   r=dict(batch=B,ctx=ctx,page=page,scale=scale,seed=seed,error=err,partial_error=pe,bound=want['tol'],clock_bitwise=True)
   row['fixtures'].append(r);print('ATTN PASS',name,r,flush=True)
   del d,out,want,z,g,clock,clocks,raw,buf,part,pg,po;torch.cuda.empty_cache()
 proof.append(row)
(H/('validation.json' if not a.names else 'validation_'+'_'.join(names)+'.json')).write_text(json.dumps(dict(passed=True,variants=proof),indent=2)+'\n')
