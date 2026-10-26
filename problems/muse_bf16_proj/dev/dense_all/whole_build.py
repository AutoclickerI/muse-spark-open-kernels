"""Batch coding, early weight issue and QKV/gate launch merging on frozen S19."""
import argparse,base64,hashlib,json,lzma
from pathlib import Path
import patch
H=Path(__file__).resolve().parent;R=H.parents[3]
p=argparse.ArgumentParser();p.add_argument('--selection',type=Path);p.add_argument('--confirm',action='store_true');a=p.parse_args()
selection={(t,r):arm for t,r,arm in json.loads(a.selection.read_text())} if a.selection else {}
base=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text()
source=patch.patch(base,selection) if selection else base
source+='\n_gate_split_on=globals()["_gate_split_on"]\n'+(H/'merge.py.inc').read_text().replace('if (cfg[\'tokens\'],bool(cfg[\'window\'])) in DA_MERGE_POINTS:',"if DA_DO_MERGE and (cfg['tokens'],bool(cfg['window'])) in DA_MERGE_POINTS:")+'\nDA_DO_MERGE=False\n'
if selection:
    source=source.replace("if key in DA_SELECTED and 'da' in state:","if DA_DO_CODE and key in DA_SELECTED and 'da' in state:")
    source+='\nDA_DO_CODE=False\n'
if a.confirm:
    assert selection=={(1,'gate_apply'):3}
    source=source.replace("if DA_DO_CODE and key in DA_SELECTED and 'da' in state:return da_project(x,'gate'", "if DA_DO_CODE and state.get('da_window',False) and key in DA_SELECTED and 'da' in state:return da_project(x,'gate'")
    source+='\n_da_window_run=globals()["_best7_run"]\ndef _best7_run(data,state,plan,prefetch=None):\n    state["da_window"]=bool(data["config"]["window"])\n    return _da_window_run(data,state,plan,prefetch)\n'
(H/'whole_source.py').write_text(source)
# Keep all published checks and data generation unchanged. No reference runs in a candidate.
nv=(R/'problems/muse_decode_layer/nvfp4.py').read_text();checks=(R/'problems/muse_decode_layer/checks.py').read_text();ref=(R/'problems/muse_decode_layer/reference.py').read_text().replace('from checks import compare, query_precision_bound, relative_l2','').replace('from nvfp4 import BLOCK, dequantize, random_weights','')
helpers=(R/'problems/muse_decode_layer/dev/s19/graph_helpers.py.inc').read_text()
payload='LAB_RESOURCES={"compile_s":260,"run_s":350,"output_bytes":8744,"context_poison":"device"}\nimport base64,lzma,torch,time,types,statistics,json,hashlib,linecache\nWHOLE_SOURCE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(source.encode())).decode())+')).decode()\nWHOLE_TASK='+repr([nv,checks,ref])+'\nWHOLE_SELECTION='+repr(selection)+'\nWHOLE_CONFIRM='+repr(a.confirm)+'\n'+helpers+'\n'+(H/'whole_lab.py.inc').read_text()
compile(payload,'<P10 whole batch>','exec');(H/'whole_payload.py').write_text(payload)
manifest=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),selection=[[*k,v] for k,v in selection.items()],seeds=[12000319+104729*i for i in range(8)],arms=['A','AA','CODE','EARLY','MERGE','BOTH'] if selection else ['A','AA','MERGE'])
manifest['confirmation']=a.confirm
if a.confirm:manifest.update(seeds=[14000319+104729*i for i in range(16)],arms=['A','AA','CODE'])
(H/'whole_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['payload_sha256'],len(payload))
