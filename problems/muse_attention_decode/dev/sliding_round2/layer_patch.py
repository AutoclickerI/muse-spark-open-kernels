"""Apply only selected P5 sliding arms onto exact S14; no unselected kernel sources shipped."""
import argparse,ast,hashlib,json,re
from pathlib import Path
import kernels
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BASE_SHA='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
DEFAULT={'16,1':['last',1,8],'32,1':['last',1,4],'1,8':['last',8,8]}

def patch(src,selection=None):
    selection=DEFAULT if selection is None else selection
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA
    selected={tuple(map(int,k.split(','))):tuple(v) for k,v in selection.items()}
    assert all(k in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for k in selected)
    assert all(v[0] in ('last','early','dsm','dsm_old') and v[1] in (1,2,4,8) for v in selected.values())
    sources={k:kernels.build.source() if k=='dsm_old' else kernels.source(k) for k in {v[0] for v in selected.values()}}
    runtime=(HERE/'runtime.py').read_text();nodes=ast.parse(runtime).body;lines=runtime.splitlines(True)
    for n in reversed(nodes):
        if isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('ARMS','SELECTED') for t in n.targets):lines[n.lineno-1:n.end_lineno]=[]
    cell=''.join(lines)
    for name in ('KERNELS','_MODULES','_COUNTERS','_FREQ','module','attention','apply_arm'):cell=re.sub(r'\b'+name+r'\b','r2_'+name,cell)
    cell='r2_KERNELS='+repr(sources)+'\n'+cell
    anchor="def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n"
    assert src.count(anchor)==1
    src=src.replace(anchor,anchor+"    if cfg['window'] and (cfg['batch'],cfg['q_len']) in state.get('r2_rows',{}):\n        cfg=dict(cfg,num_q_heads=HQ,num_kv_heads=HKV,head_dim=HD)\n        return r2_apply_arm(dict(data,config=cfg),qkv,state['r2_rows'][(cfg['batch'],cfg['q_len'])]).view(-1,Q_SIZE)\n")
    anchor="    state['vq8_rows']=_vq8_prepare(device)\n";assert src.count(anchor)==1
    src=src.replace(anchor,anchor+"    state['r2_rows']=_r2_prepare(device)\n")
    src+='\n'+cell+'\n_R2_SELECTION='+repr(selected)+'\n'+'''
def _r2_prepare(device):
    for kind in r2_KERNELS:r2_module(kind)
    for (B,L),arm in _R2_SELECTION.items():
        for ctx in (257,2305):
            qkv,data=_attn_port_synthetic(B,L,2048,device,ctx)
            cfg=dict(data['config'],num_q_heads=HQ,num_kv_heads=HKV,head_dim=HD)
            got=r2_apply_arm(dict(data,config=cfg),qkv,arm).view(-1,Q_SIZE)
            _attention_store_agrees(qkv,data)
            ok,err,bound=_attention_agrees(got,_attention_reference(qkv,data),_attention_reference(qkv,data,True))
            if not ok:raise RuntimeError(f'round2 {(B,L)} {arm} ctx{ctx}: {err}>{bound}')
    return _R2_SELECTION
'''
    compile(src,'p5_round2_layer','exec');assert len(src.encode())<1_000_000
    return src

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selection',type=json.loads,default=DEFAULT);ap.add_argument('--out',type=Path,default=HERE/'out/s14_round2.py');a=ap.parse_args()
    s=patch((ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text(),a.selection);a.out.write_text(s)
    print(a.out,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':main()
