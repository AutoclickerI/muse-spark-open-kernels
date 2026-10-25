"""Check register-V PV layout independently, then full q8 attention/cache/capture gates."""
import hashlib
import importlib.util
import json
import linecache
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
def load(n):
    p=H/(n+'.py');spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old,new=load('vq_runtime'),load('hybrid')
ext=new.vq8_ext()
gen=torch.Generator(device='cuda').manual_seed(61791)
pv=[]
for mode in ('random','all finite E4M3'):
    if mode=='random':
        p=(torch.rand((64,64),device='cuda',generator=gen)*256).to(torch.float8_e4m3fn)
        v=torch.randn((64,64),device='cuda',generator=gen).to(torch.float8_e4m3fn)
    else:
        bits=torch.arange(256,device='cuda',dtype=torch.int32).to(torch.uint8)
        bits=bits[(bits!=127)&(bits!=255)].repeat(17)[:4096].reshape(64,64).contiguous()
        p=bits.view(torch.float8_e4m3fn);v=bits.flip(0).contiguous().view(torch.float8_e4m3fn)
    out=torch.empty((64,64),device='cuda',dtype=torch.float32)
    ext.hybrid_pv_probe(p.data_ptr(),v.data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream)
    ref=p.double()@v.double();err=float((out.double()-ref).norm()/ref.norm())
    assert err<1e-5,(mode,err)
    pv.append(dict(case=mode,error=err));print('PV PASS',pv[-1],flush=True)
s=(H/'phases_lab.py').read_text().split('\nSOURCES=',1)[0];ns={};linecache.cache['<full-hybrid-inputs>']=(len(s),None,s.splitlines(True),'<full-hybrid-inputs>');exec(compile(s,'<full-hybrid-inputs>','exec'),ns)
rows=[]
for B,ctx,page,scale in ((2,257,64,1/64),(16,4096,1,1),(16,80000,1,1),(32,80000,1,1)):
    data=ns['generate_input'](batch=B,ctx=ctx,heads='tp8',layer='full',q_len=8,page=page,kv_scale=scale,dist='coherent',spread='agentic',seed=61791)
    a=old.vq8_attention(data,stages=6,min_tiles=1)
    hk=data['config']['num_kv_heads'];slots=data['cache_loc'].reshape(-1)
    ca=tuple(data[k].view(-1,hk,64)[slots].clone() for k in ('k_cache','v_cache'))
    b=new.vq8_attention(data,stages=6,min_tiles=1)
    torch.cuda.synchronize();assert all(torch.equal(v,data[k].view(-1,hk,64)[slots]) for k,v in zip(('k_cache','v_cache'),ca))
    want=ns['gate_prepare'](data);ok,msg,err=ns['gate_cached'](data,b,want);assert ok,msg
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):z=new.vq8_attention(data,stages=6,min_tiles=1)
    graph.replay();torch.cuda.synchronize();ok,msg,_=ns['gate_cached'](data,z,want);assert ok,msg
    raw=data['qkv'].view(B*8,-1)
    partial=(raw.float()*.25).unsqueeze(0).repeat(4,1,1)
    part_out=new.attention_vq8((partial,4),data,setting=(6,1))
    torch.cuda.synchronize();assert torch.equal(part_out.view_as(b),b),'FP32 projection-partials ABI differs'
    pg=torch.cuda.CUDAGraph()
    with torch.cuda.graph(pg):part_out=new.attention_vq8((partial,4),data,setting=(6,1))
    pg.replay();torch.cuda.synchronize();assert torch.equal(part_out.view_as(b),b)
    rows.append(dict(batch=B,ctx=ctx,page=page,scale=scale,error=err,old_error=ns['relative_l2'](a,want['out']),old_relative=ns['relative_l2'](b,a),bound=want['tol'],partial_splits=4,partial_bitwise=True))
    print('ATTN PASS',rows[-1],flush=True)
    del data,a,b,z,graph,want,partial,part_out,pg
    torch.cuda.empty_cache()
(H/'hybrid_validation.json').write_text(json.dumps(dict(passed=True,source_sha256=hashlib.sha256((H/'hybrid.py').read_bytes()).hexdigest(),pv=pv,rows=rows),indent=2)+'\n')
