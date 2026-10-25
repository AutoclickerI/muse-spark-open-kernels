"""Probe math then full attention/cache/capture and strided FP32 projection ABI."""
import argparse
import concurrent.futures
import hashlib
import json
import linecache
from pathlib import Path
import torch
from checks import check_probe
H=Path(__file__).resolve().parent;P=H.parent
ap=argparse.ArgumentParser();ap.add_argument('names',nargs='*',default=['ref','mix25','mix50','gap','mixgap','tc25']);a=ap.parse_args()
mods={}
for name in a.names:
 s=(H/(name+'.py')).read_text();f=str(H/(name+'.py'));linecache.cache[f]=(len(s),None,s.splitlines(True),f);m={};exec(compile(s,f,'exec'),m);m['SOFT_TC']=name=='tc25';m['SOURCE_SHA256']=hashlib.sha256(m['VQ8_CUDA'].encode()).hexdigest();mods[name]=m
with concurrent.futures.ThreadPoolExecutor(len(mods)) as pool:
 fs=[pool.submit(m['vq8_ext']) for m in mods.values()]
 for f in fs:f.result()
prefix=(P/'phases_lab.py').read_text().split('\nSOURCES=',1)[0];ns={};exec(compile(prefix,'<softmax-inputs>','exec'),ns)
proof=[]
for name,m in mods.items():
 r=check_probe(m,torch);print('PROBE PASS',name,r,flush=True)
 row=dict(name=name,cuda_sha256=m['SOURCE_SHA256'],probe=r,attention=[])
 for B,ctx,page,scale in [(2,257,64,1/64),(16,4096,1,1),(16,80000,1,1),(32,80000,1,1)]:
  data=ns['generate_input'](batch=B,ctx=ctx,heads='tp8',layer='full',q_len=8,page=page,kv_scale=scale,dist='coherent',spread='agentic',seed=61791)
  y=m['vq8_attention'](data,stages=6,min_tiles=1);want=ns['gate_prepare'](data);ok,msg,err=ns['gate_cached'](data,y,want);assert ok,(name,msg)
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):z=m['vq8_attention'](data,stages=6,min_tiles=1)
  g.replay();torch.cuda.synchronize();ok,msg,_=ns['gate_cached'](data,z,want);assert ok,(name,msg)
  raw=data['qkv'].view(B*8,-1);buf=torch.randn((3,B*8+1,raw.shape[-1]),device='cuda')*.03;part=buf[:,:B*8];part[-1]+=raw.float()-part.sum(0)
  got=m['attention_vq8']((part,3),data,setting=(6,1));torch.cuda.synchronize();ok,msg,pe=ns['gate_cached'](data,got.view_as(y),want);assert ok,(name,'strided partial',msg)
  pg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(pg):got=m['attention_vq8']((part,3),data,setting=(6,1))
  pg.replay();torch.cuda.synchronize();ok,msg,_=ns['gate_cached'](data,got.view_as(y),want);assert ok,(name,msg)
  clocks=[];q=m['vq8_attention'](data,stages=6,min_tiles=1,clocks=clocks);torch.cuda.synchronize();assert torch.equal(q,y),(name,'clock math differs')
  row['attention'].append(dict(batch=B,ctx=ctx,page=page,scale=scale,error=err,partial_error=pe,bound=want['tol'],clock_bitwise=True))
  print('ATTN PASS',name,row['attention'][-1],flush=True)
  del data,y,z,g,want,buf,part,got,pg,clocks,q;torch.cuda.empty_cache()
 proof.append(row)
name='validation.json' if a.names==['ref','mix25','mix50','gap','mixgap','tc25'] else 'validation_'+'_'.join(a.names)+'.json'
(H/name).write_text(json.dumps(dict(passed=True,variants=proof),indent=2)+'\n')
