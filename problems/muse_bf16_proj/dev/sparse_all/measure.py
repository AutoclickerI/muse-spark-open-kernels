"""Measured bank sizes on actual local reference weights; no reference in payload."""
import importlib.util,json,gc,hashlib,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
sys.path.insert(0,str(R/'problems/muse_decode_layer'))
m=load('sparse_measure',H/'candidate.py');ref=load('dense_measure_ref',R/'problems/muse_decode_layer/reference.py')
rows=[]
for seed in (5353,61791,14357):
    weights=ref.generate_weights(seed);roles=m.dc_role_weights(weights)
    for name,w in roles.items():
        c=m.cd_encode(w);bank,stats=m.dc_encode_cd(c)
        # Retained CD metadata is streamed by the M1 native reader, in addition
        # to the point index used by scalar readers and shared-WGMMA producers.
        common=sum(c[key].numel()*c[key].element_size() for key in ('scales','eoff','esc'))
        stats.update(cd_metadata_bytes=common,cd_total_bytes=stats['total_bytes']+common,
            saved_vs_bf16=stats['raw16_bytes']-stats['total_bytes'],saved_vs_12_body=stats['raw12_bytes']-stats['body_bytes'])
        rows.append(dict(seed=seed,matrix=name,shape=list(w.shape),**stats));print(seed,name,stats['bits_per_weight'],stats['saved_vs_bf16'],flush=True)
        del c,bank
    del roles,weights;gc.collect();torch.cuda.empty_cache()
result=dict(candidate_sha256=hashlib.sha256((H/'candidate.py').read_bytes()).hexdigest(),
    provenance='reference.generate_weights, full published generator including expert RNG prefix; no modified values',rows=rows)
(H/'sizes.json').write_text(json.dumps(result,indent=2)+'\n')
