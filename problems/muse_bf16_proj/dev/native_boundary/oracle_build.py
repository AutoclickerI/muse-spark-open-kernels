"""One-boundary deletion/split oracle; no added device nodes in timed graphs."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];O=H/'generated';O.mkdir(exist_ok=True)
local='--local-selection' in sys.argv
if local:sized=dict(selected=sys.argv[sys.argv.index('--local-selection')+1],job='LOCAL-GLUE-ONLY')
else:sized=json.loads((H/'sizing_results.json').read_text())
selected=sized['selected'];stem='oracle_'+selected if local else 'oracle'
def functions(path,names):
 s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)
base=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text();helpers=functions(R/'problems/muse_decode_layer/dev/s19/graph_helpers.py.inc',('ab_copy','ab_graph'))
helper=functions(H/'sizing_lab.py.inc',('nb_load','nb_clone','nb_time'))
checks=[(R/'problems/muse_decode_layer/nvfp4.py').read_text(),(R/'problems/muse_decode_layer/checks.py').read_text(),(R/'problems/muse_decode_layer/reference.py').read_text().replace('from checks import compare, query_precision_bound, relative_l2','').replace('from nvfp4 import BLOCK, dequantize, random_weights','')]
body='LAB_RESOURCES={"compile_s":180,"run_s":220,"output_bytes":11744,"context_poison":"none"}\nimport torch,time,types,linecache,hashlib,json,statistics,math,lzma,base64,struct,gc\nNB_SELECTED='+repr(selected)+'\nNB_BASE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(base.encode())).decode())+')).decode()\nNB_TASK='+repr(checks)+'\n'+helpers+'\n'+helper+'\n'+(H/'codec.py.inc').read_text()+'\n'+(H/'oracle_lab.py.inc').read_text()
compile(body,'<P10 native one-boundary oracle>','exec');(O/(stem+'_payload.py')).write_text(body);((O if local else H)/(stem+'_manifest.json')).write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),payload_sha256=hashlib.sha256(body.encode()).hexdigest(),selected=selected,sizing_receipt=sized['job'],weight_seed=22000311,router_seed=22000313,input_seeds=[22000319+104729*i for i in range(8)],rounds=16,protocol='palindromic-cohort-v1;all rounds arithmetic mean;all round times retained'),indent=2)+'\n');print(selected,len(body))
