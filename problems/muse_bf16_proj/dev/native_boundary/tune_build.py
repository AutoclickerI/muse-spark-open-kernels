"""Whole-layer native schedule arms selected after the one-boundary oracle."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
import native_patch
H=Path(__file__).resolve().parent;R=H.parents[3];O=H/'generated';oracle=json.loads((H/'oracle_results.json').read_text());assert oracle['raw']['complete']
base=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text()
def functions(p,names):
 s=p.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)
body=native_patch.patch(base,{})
helpers=functions(R/'problems/muse_decode_layer/dev/s19/graph_helpers.py.inc',('ab_copy','ab_graph'))
helper=functions(H/'sizing_lab.py.inc',('nb_load','nb_time'))
checks=[(R/'problems/muse_decode_layer/nvfp4.py').read_text(),(R/'problems/muse_decode_layer/checks.py').read_text(),(R/'problems/muse_decode_layer/reference.py').read_text().replace('from checks import compare, query_precision_bound, relative_l2','').replace('from nvfp4 import BLOCK, dequantize, random_weights','')]
ref=(R/'problems/muse_bf16_proj/reference.py').read_text();names={'FP32_LAYERS','REL_L2_TOL','ROUTER_TOL','PROBE_TOL','ref_kernel','probe_rows','check_implementation'};checker=(R/'problems/muse_bf16_proj/checks.py').read_text()+'\n'+'\n'.join(ast.get_source_segment(ref,n) for n in ast.parse(ref).body if getattr(n,'name',None) in names or isinstance(n,ast.Assign) and any(getattr(t,'id',None) in names for t in n.targets))
payload='LAB_RESOURCES={"compile_s":220,"run_s":260,"output_bytes":13744,"context_poison":"none"}\nimport torch,time,types,linecache,hashlib,json,statistics,math,lzma,base64,struct,gc\nNB_POINTS_ARMS='+repr(native_patch.proposals())+'\nNB_BASE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(body.encode())).decode())+')).decode()\nNB_TASK='+repr(checks)+'\nNB_CHECK='+repr(checker)+'\n'+helpers+'\n'+helper+'\n'+(H/'codec.py.inc').read_text()+'\n'+(H/'tune_lab.py.inc').read_text()
compile(payload,'<P10 native issue tuner>','exec');(O/'tune_payload.py').write_text(payload);(H/'tune_manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),oracle=oracle['job'],weight_seed=23000311,router_seed=23000313,seeds=[23000319+104729*i for i in range(8)],arms={k:[[*key,*v] for key,v in pts.items()] for k,pts in native_patch.proposals().items()},rounds=16),indent=2)+'\n');print(len(payload))
