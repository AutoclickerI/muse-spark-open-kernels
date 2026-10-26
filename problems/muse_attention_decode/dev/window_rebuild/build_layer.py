"""One-file exact-S15 integration candidate with selected streaming attention only."""
import ast,hashlib,json,re
from pathlib import Path
import kernel
H=Path(__file__).resolve().parent;R=H.parents[3]

def runtime(sources):
    s=(H/'runtime.py').read_text()
    for name in ('KERNELS','_MODULES','_FREQ','_COUNTERS','TOPOLOGY','module','attention'):
        s=re.sub(r'\b'+name+r'\b','wr_'+name,s)
    return 'wr_KERNELS='+repr(sources)+'\n'+s

def patch(base,selection=None):
    assert hashlib.sha256(base.encode()).hexdigest()=='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
    rows=selection or {(32,8):('STREAM2',False)}
    sources={n:kernel.source(2 if n.endswith('2') else 4,stream=n.startswith('STREAM')) for n,early in set(rows.values())}
    anchor="def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n"
    assert base.count(anchor)==1
    base=base.replace(anchor,anchor+"    if 0<cfg['window']<=2048 and (cfg['batch'],cfg['q_len']) in state.get('wr_rows',{}):\n        cfg=dict(cfg,num_q_heads=HQ,num_kv_heads=HKV,head_dim=HD)\n        name,early=state['wr_rows'][(cfg['batch'],cfg['q_len'])]\n        return wr_attention(dict(data,config=cfg),qkv,name,early).view(-1,Q_SIZE)\n")
    anchor="    state['vq8_rows']=_vq8_prepare(device)\n";assert base.count(anchor)==1
    base=base.replace(anchor,anchor+"    state['wr_rows']=_window_prepare(device)\n")
    base+='\n'+runtime(sources)+'\n_WINDOW_SELECTION='+repr(rows)+'''
def _window_prepare(device):
    for name in wr_KERNELS:wr_module(name)
    return _WINDOW_SELECTION
'''
    compile(base,'p5_window_layer','exec');assert len(base.encode())<1_000_000
    return base

def main():
    (H/'out').mkdir(exist_ok=True)
    base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
    s=patch(base);p=H/'out/s15_window_raw32.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
    return s

if __name__=='__main__':main()
