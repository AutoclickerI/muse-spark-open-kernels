"""Bit-exact proof of shared bodies and GPU escape repartitioning at every split."""
import importlib.util,json,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('da_codec',H/'adapter.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
g=torch.Generator(device='cuda').manual_seed(910031)
weights={name:(torch.randn((n,k),device='cuda',generator=g)/(k**.5)).to(torch.float32 if name=='router' else torch.bfloat16) for name,n,k in [('qkv_gate',2304,8192),('o',8192,1024),('pooler',4096,8192),('unpooler',8192,4096),('router',256,8192)]}
bank=m.da_prepare(weights,True,True);ns=bank['ns'];records=[];total=0
for name,w in bank['roles'].items():
    if name=='unpooler_slice':continue
    groups=(w[:4096],w[4096:4352],w[4352:]) if name=='pool_router' else (w,)
    for s in (1,2,4,8):
        parts=[];metadata=[];tables=[]
        for group in groups:
            p=ns['dd_decodediet_tt_mw_pack_weight'](group.contiguous(),64,1,s,True);assert p is not None,(name,s)
            parts.append(p['w']);metadata.append(p['esc']);tables.extend([(p['table']&0xffffffff,p['table']>>32)]*p['esc'].shape[0]);total+=group.numel()
        expected=torch.cat(parts);e=torch.cat(metadata);actual=bank['wide'][128,name,next(i for i,v in enumerate(m.DA_WIDE) if v[1]==s and not v[3])]
        assert torch.equal(expected,actual['wp']),(name,s,'body')
        assert torch.equal(e.sort(1).values,actual['esc'][:,:256].sort(1).values),(name,s,'escape multiset')
        assert torch.equal(torch.tensor(tables,device='cuda',dtype=torch.int64),actual['esc'][:,256:].long()),(name,s,'origins')
        row=dict(role=name,splits=s,weights=w.numel(),escapes=int((e!=-1).sum()),max_escapes=int((e!=-1).sum(1).max()));records.append(row);print(row,flush=True)
(H/'codec_validation.json').write_text(json.dumps(dict(passed=True,weight_patterns_compared=total,records=records,prepare_times=bank['times']),indent=2)+'\n')
