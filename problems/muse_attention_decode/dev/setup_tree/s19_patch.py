"""Shape-only S19 patch; native kernels are identical to the measured arms."""
import ast,difflib,hashlib,importlib.util,json,re,sys
from pathlib import Path
import kernels_new as k
from pack_setup import base
H=Path(__file__).resolve().parent
def runtime(sources):
    s=(H.parent/'sliding_oracle/runtime.py').read_text();lines=s.splitlines(True)
    for n in reversed(ast.parse(s).body):
        if isinstance(n,ast.FunctionDef) and n.name=='coordinates':lines[n.lineno-1:n.end_lineno]=[]
    s=''.join(lines)
    for name in ('SOURCES','MODULES','FREQ','COUNTERS','LOCAL','module','attention'):s=re.sub(r'\b'+name+r'\b','st_'+name,s)
    return 'st_SOURCES='+repr(sources)+'\n'+s
def patch(selection):
    source=base();sources={n:k.source(n[0],n[1]) for n in sorted(set(selection.values()))}
    anchor="def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n";assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"    if 0<cfg['window']<=2048 and (cfg['batch'],cfg['q_len']) in state.get('st_rows',{}):\n        cfg=dict(cfg,num_q_heads=HQ,num_kv_heads=HKV,head_dim=HD)\n        return st_attention(dict(data,config=cfg),None,state['st_rows'][(cfg['batch'],cfg['q_len'])],qkv=qkv).view(-1,Q_SIZE)\n")
    anchor="    state['vq8_rows']=_vq8_prepare(device)\n";assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"    state['st_rows']=_st_prepare(device)\n")
    source+='\n'+runtime(sources)+'\n_ST_SELECTION='+repr(selection)+'''
def _st_prepare(device):
    for name in st_SOURCES:st_module(name)
    return _ST_SELECTION
'''
    compile(source,'p5_s19_setup','exec');return source
def build(arm):
    assert arm in ('Q','T','H','selected')
    rows={(32,1):'dHA'} if arm=='selected' else {(16,1):'d'+arm+'A',(32,1):'d'+arm+'A',(16,8):'v'+arm+'A',(32,8):'v'+arm+'A'} if arm in ('Q','H') else {(16,8):'vTA',(32,8):'vTA'}
    s=patch(rows);(H/'out').mkdir(exist_ok=True)
    sp=importlib.util.spec_from_file_location('p5_setup_ship',H.parents[2]/'muse_decode_layer/dev/s16/pack.py');ship=importlib.util.module_from_spec(sp);sp.loader.exec_module(ship)
    shipped=ship.pack(s);p=H/'out'/f's19_{arm}.py';p.write_text(shipped)
    meta=dict(base_sha256=hashlib.sha256(base().encode()).hexdigest(),candidate_sha256=hashlib.sha256(shipped.encode()).hexdigest(),expanded_sha256=hashlib.sha256(s.encode()).hexdigest(),selection={str(k):v for k,v in rows.items()},file=str(p.relative_to(H)))
    (H/f'{arm}_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(p,meta['candidate_sha256']);return p
if __name__=='__main__':build(sys.argv[1])
