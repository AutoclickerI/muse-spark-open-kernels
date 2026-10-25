"""Reviewable cooperative-prep cluster candidate; promotion requires paired H200 evidence."""
import ast,hashlib,re,sys
from pathlib import Path
import build_compact
HERE=Path(__file__).resolve().parent
BASE_SHA='acda6c621b1f3b885f2111ea1b4482ae1d733e2f2493db2b7f0b852ccf66dfb7'

def patch(src,rows=((1,1,True),),nk=128,cs=None):
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA,'apply onto frozen S11 acda6c62'
    cuda=build_compact.source();cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
    d=(HERE/'driver.py').read_text();lines=d.splitlines(True)
    cell='\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(d).body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('_MOD','_PLANS','_FREQ') for t in n.targets)))
    for name in ('CUDA','CPP','_MOD','_PLANS','_FREQ','module','plan','attention'):cell=re.sub(r'\b'+name+r'\b','cs_'+name,cell)
    cell='import hashlib\ncs_CUDA='+repr(cuda)+'\ncs_CPP='+repr(cpp)+'\n'+cell
    old="def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n"
    assert src.count(old)==1
    src=src.replace(old,old+"    if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in state.get('cluster_rows',()):\n        return _cluster_attention(qkv,data)\n")
    old='        native_pool.submit(_vq8_module)  # vq8 rows (q_len 8)\n';assert src.count(old)==1
    src=src.replace(old,old+'        native_pool.submit(cs_module)\n')
    old="    state['vq8_rows']=_vq8_prepare(device)\n";assert src.count(old)==1
    src=src.replace(old,old+"    state['cluster_rows']=_cluster_prepare(device)\n")
    return src+'\n\n'+cell+'\n_CLUSTER_ROWS='+repr(rows)+'\n_CLUSTER_NK='+repr(nk)+'\n_CLUSTER_CS='+repr(cs)+'\n'+r'''

def _cluster_attention(qkv,data):
    cfg=dict(data['config'],num_q_heads=HQ,num_kv_heads=HKV,head_dim=HD)
    return cs_attention(dict(data,config=cfg),qkv,nk=_CLUSTER_NK,cs=_CLUSTER_CS).view(-1,Q_SIZE)

def _cluster_prepare(device):
    for batch,length,sliding in _CLUSTER_ROWS:
        assert sliding and length==1
        for ctx in (257,2305):
            qkv,data=_attn_port_synthetic(batch,length,2048,device,ctx)
            got=_cluster_attention(qkv,data);_attention_store_agrees(qkv,data)
            ok,error,bound=_attention_agrees(got,_attention_reference(qkv,data),_attention_reference(qkv,data,True))
            if not ok:raise RuntimeError(f'cluster {(batch,length,sliding)} ctx {ctx}: {error:.5f}>{bound:.5f}')
    print('CLUSTER_ROWS',_CLUSTER_ROWS,'NK',_CLUSTER_NK,'CS',_CLUSTER_CS,flush=True)
    return _CLUSTER_ROWS
'''

if __name__=='__main__':
    s=patch(Path(sys.argv[1]).read_text());p=Path(sys.argv[2]);p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
