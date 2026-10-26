"""Exact-S15 integration of the SFU probability block at selected sliding shapes."""
import ast,difflib,hashlib,json,re
from pathlib import Path
import structural
H=Path(__file__).resolve().parent;R=H.parents[3]
ALL={(16,1):'dEA',(32,1):'dEA',(16,8):'vEA',(32,8):'vEA'}

def runtime(sources):
    s=(H/'runtime.py').read_text();lines=s.splitlines(True)
    for n in reversed(ast.parse(s).body):
        if isinstance(n,ast.FunctionDef) and n.name=='coordinates':lines[n.lineno-1:n.end_lineno]=[]
    s=''.join(lines)
    for name in ('SOURCES','MODULES','FREQ','COUNTERS','LOCAL','module','attention'):s=re.sub(r'\b'+name+r'\b','so_'+name,s)
    return 'so_SOURCES='+repr(sources)+'\n'+s

def patch(base,selection=None):
    assert hashlib.sha256(base.encode()).hexdigest()=='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
    rows=ALL if selection is None else selection
    sources={n:structural.source(n[0]) for n in set(rows.values())}
    anchor="def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n";assert base.count(anchor)==1
    base=base.replace(anchor,anchor+"    if 0<cfg['window']<=2048 and (cfg['batch'],cfg['q_len']) in state.get('so_rows',{}):\n        cfg=dict(cfg,num_q_heads=HQ,num_kv_heads=HKV,head_dim=HD)\n        return so_attention(dict(data,config=cfg),None,state['so_rows'][(cfg['batch'],cfg['q_len'])],qkv=qkv).view(-1,Q_SIZE)\n")
    anchor="    state['vq8_rows']=_vq8_prepare(device)\n";assert base.count(anchor)==1
    base=base.replace(anchor,anchor+"    state['so_rows']=_so_prepare(device)\n")
    s=base+'\n'+runtime(sources)+'\n_SO_SELECTION='+repr(rows)+'''
def _so_prepare(device):
    for name in so_SOURCES:so_module(name)
    return _SO_SELECTION
'''
    compile(s,'p5_sliding_exp','exec');assert len(s.encode())<1_000_000;return s

def main():
    (H/'out').mkdir(exist_ok=True);base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text();s=patch(base)
    p=H/'out/s15_sliding_exp.py';p.write_text(s);meta=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),selection={str(k):v for k,v in ALL.items()},status='screen artifact; no promotion before native pair')
    (H/'candidate_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(p,meta['candidate_sha256']);return s

if __name__=='__main__':main()
