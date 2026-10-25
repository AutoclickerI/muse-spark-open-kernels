"""Add public-shape fixed-cluster verify sliding to the qualified B1 sliding and full-verify wins."""
import ast,hashlib,re,sys,importlib.util
from pathlib import Path
import patch_last,build
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]

def patch(src,clusters=None):
    clusters={1:16,16:4,32:2} if clusters is None else clusters
    out=patch_last.patch(src)
    spec=importlib.util.spec_from_file_location('_p5_dsm_full_patch',HERE.parent/'full128/patch_full128.py');pf=importlib.util.module_from_spec(spec);spec.loader.exec_module(pf);out=pf.replace_cell(out)
    cuda=build.source();d=(HERE/'driver.py').read_text();cell='\n'.join(ast.get_source_segment(d,n) for n in ast.parse(d).body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('_MOD','_PLANS','_FREQ') for t in n.targets)))
    for name in ('CUDA','CPP','_MOD','_PLANS','_FREQ','module','plan','attention'):cell=re.sub(r'\b'+name+r'\b','cc_'+name,cell)
    cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
    old="def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n";assert out.count(old)==1
    out=out.replace(old,old+"    if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in state.get('dsm_rows',()):\n        return _dsm_attention(qkv,data)\n")
    old='        native_pool.submit(_vq8_module)  # vq8 rows (q_len 8)\n';assert old in out;out=out.replace(old,old+'        native_pool.submit(cc_module)\n')
    old="    state['vq8_rows']=_vq8_prepare(device)\n";assert old in out;out=out.replace(old,old+"    state['dsm_rows']=_dsm_prepare(device)\n")
    return out+'\n\ncc_CUDA='+repr(cuda)+'\ncc_CPP='+repr(cpp)+'\n'+cell+'\n_DSM_CS='+repr(clusters)+'\n'+r'''

def _dsm_attention(qkv,data):
    cfg=dict(data['config'],num_q_heads=HQ,num_kv_heads=HKV,head_dim=HD)
    return cc_attention(dict(data,config=cfg),qkv,nk=128,cs=_DSM_CS[cfg['batch']]).view(-1,Q_SIZE)

def _dsm_prepare(device):
    rows=tuple((b,8,True) for b in _DSM_CS)
    for batch,length,sliding in rows:
        for ctx in (257,2305):
            qkv,data=_attn_port_synthetic(batch,length,2048,device,ctx)
            got=_dsm_attention(qkv,data);_attention_store_agrees(qkv,data)
            ok,error,bound=_attention_agrees(got,_attention_reference(qkv,data),_attention_reference(qkv,data,True))
            if not ok:raise RuntimeError(f'DSM {(batch,length)} ctx{ctx}: {error:.5f}>{bound:.5f}')
    print('DSM_ROWS',rows,'CS',_DSM_CS,flush=True)
    return rows
'''
if __name__=='__main__':
    s=patch(Path(sys.argv[1]).read_text());p=Path(sys.argv[2]);p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
