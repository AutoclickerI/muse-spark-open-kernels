"""Experimental compact early-history last-CTA on exact S14 final. H200 pair required."""
import ast,hashlib,re,sys
from pathlib import Path
import history_build
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BASE_SHA='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'

def cell(early=True,clock=False):
    d=(HERE.parent/'cluster_sliding/last_driver.py').read_text()
    # No input-dependent planning, no per-call metadata/reset kernels.
    d=d.replace("ns=cs or plan(cfg,data['page_table'].shape[1],P.device,qt,nk)[0];assert 1<=ns<=32", "ns=cs or 16;assert nk==128 and 16<=ns<=32 and cfg['window']<=ns*nk")
    if clock:
        d=d.replace('cs=None,gate=True):','cs=None,gate=True,clockout=None):')
        d=d.replace('part.data_ptr(),counter.data_ptr(),ns)', 'part.data_ptr(),counter.data_ptr(),ns,clockout.data_ptr())')
    code='\n'.join(ast.get_source_segment(d,n) for n in ast.parse(d).body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('_MOD','_PLANS','_FREQ','_COUNTERS') for t in n.targets)))
    cuda=history_build.source(early,clock)
    cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
    return 'import hashlib\nfrom torch.utils.cpp_extension import load_inline\nCUDA='+repr(cuda)+'\nCPP='+repr(cpp)+'\n'+code

def prefixed_cell(prefix='eh_'):
    code=cell();lines=code.splitlines(True);literals={}
    for node in reversed(ast.parse(code).body):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('CUDA','CPP'):
            literals[node.targets[0].id]=ast.literal_eval(node.value)
            lines[node.lineno-1:node.end_lineno]=[]
    code=''.join(lines)
    for name in ('CUDA','CPP','_MOD','_PLANS','_FREQ','_COUNTERS','module','plan','attention'):
        code=re.sub(r'\b'+name+r'\b',prefix+name,code)
    return prefix+'CUDA='+repr(literals['CUDA'])+'\n'+prefix+'CPP='+repr(literals['CPP'])+'\n'+code

def patch(src):
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA,'apply onto exact S14 final f1c77ce6'
    code=prefixed_cell()
    old="def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n"
    assert src.count(old)==1
    src=src.replace(old,old+"    if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in state.get('early_history_rows',()):\n        return _early_history_attention(qkv,data)\n")
    old='        native_pool.submit(_vq8_module)  # vq8 rows (q_len 8)\n';assert src.count(old)==1
    src=src.replace(old,old+'        native_pool.submit(eh_module)\n')
    old="    state['vq8_rows']=_vq8_prepare(device)\n";assert src.count(old)==1
    src=src.replace(old,old+"    state['early_history_rows']=_early_history_prepare(device)\n")
    return src+'\n'+code+'''

def _early_history_attention(qkv,data):
    cfg=dict(data['config'],num_q_heads=HQ,num_kv_heads=HKV,head_dim=HD)
    return eh_attention(dict(data,config=cfg),qkv,nk=128,cs=16).view(-1,Q_SIZE)

def _early_history_prepare(device):
    for ctx in (10,257,2047,2305):
        qkv,data=_attn_port_synthetic(1,1,2048,device,ctx)
        got=_early_history_attention(qkv,data);_attention_store_agrees(qkv,data)
        ok,error,bound=_attention_agrees(got,_attention_reference(qkv,data),_attention_reference(qkv,data,True))
        if not ok:raise RuntimeError(f'early history ctx {ctx}: {error:.5f}>{bound:.5f}')
    return ((1,1,True),)
'''
if __name__=='__main__':
    s=patch((ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text());p=HERE/'out/s14_early_history.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
