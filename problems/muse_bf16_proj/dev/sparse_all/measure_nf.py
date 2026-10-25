"""Exact native NF packed sizes; packers only, no Hopper launch or compilation."""
import importlib.util,sys,json
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'problems/muse_decode_layer'))
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
m=load('dc_nf_measure',H/'candidate.py');ref=load('dense_nf_ref',R/'problems/muse_decode_layer/reference.py')
w=ref.generate_weights(5353);roles=m.dc_role_weights(w);ns=m._dc_base_nf_vendor();rows=[]
for t,name in sorted(m.NF_PORT_SELECTED):
    if (t,name)==(32,'o'):continue
    a=roles[name]
    if (t,name)==(32,'unpooler'):
        p=ns['dd_decodediet_tt_mw_pack_weight'](a,64,4,1,True);raw=p['w'];meta=p['esc'].shape[0]*66*4;fmt=1
    else:
        family,cfg,*_=m.NF_SMALL[t,name];p,tab,stream=m._nf_group_pack(ns,a,name,family);p=stream(p,1,*cfg)
        raw=p['wp'];meta=p['esc'].shape[0]*34*4
        fmt=1 if family in ('cp','d9','q8','q16','q32','p32','mw') else 0
    bank,st=m.dc_pack(raw,fmt,n=a.shape[0],k=a.shape[1]);st.update(t=t,matrix=name,metadata_bytes=meta,
        native_total_bytes=bank.numel()+meta,incumbent_bytes=raw.numel()+meta,saved=raw.numel()-bank.numel())
    st['native_bits']=(bank.numel()+meta)*8/a.numel();rows.append(st);print(t,name,st['native_bits'],st['saved'],flush=True)
(H/'native_sizes.json').write_text(json.dumps(dict(seed=5353,rows=rows),indent=2)+'\n')
