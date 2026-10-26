"""S19 native graph sizing: CUPTI adds no device work; one outside replay boundary."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];O=H/'generated';O.mkdir(exist_ok=True)
base=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text();assert hashlib.sha256(base.encode()).hexdigest().startswith('8b415553')
def functions(path,names):
 s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)
f=R/'problems/muse_decode_layer/dev/s15_timeline/timeline_generated.py.inc';activity=next(ast.literal_eval(n.value) for n in ast.parse(f.read_text()).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='T15_ACTIVITY' for t in n.targets))
helpers=functions(R/'problems/muse_decode_layer/dev/s19/graph_helpers.py.inc',('ab_copy','ab_graph'))
profile=functions(R/'problems/muse_decode_layer/dev/s15_timeline/lab_body.inc',('t15_activity','t15_capture'))
checker=[(R/'problems/muse_decode_layer/nvfp4.py').read_text(),(R/'problems/muse_decode_layer/checks.py').read_text(),(R/'problems/muse_decode_layer/reference.py').read_text().replace('from checks import compare, query_precision_bound, relative_l2','').replace('from nvfp4 import BLOCK, dequantize, random_weights','')]
body='LAB_RESOURCES={"compile_s":190,"run_s":260,"output_bytes":11744,"context_poison":"none"}\nimport torch,time,types,linecache,hashlib,json,statistics,math,lzma,base64,struct,gc\nfrom torch.utils.cpp_extension import load_inline\nNB_BASE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(base.encode())).decode())+')).decode()\nNB_TASK='+repr(checker)+'\nT15_ACTIVITY='+repr(activity)+'\n'+helpers+'\n'+profile+'\n'+(R/'problems/muse_decode_layer/dev/s15_timeline/analysis.py').read_text()+'\n'+(H/'codec.py.inc').read_text()+'\n'+(H/'sizing_lab.py.inc').read_text()
compile(body,'<P10 native boundary sizing>','exec');(O/'sizing_payload.py').write_text(body);(H/'sizing_manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),payload_sha256=hashlib.sha256(body.encode()).hexdigest(),weight_seed=21000311,router_seed=21000313,input_seeds=[21000319+104729*i for i in range(8)],rounds=16,protocol='palindromic-cohort-v1;all rounds arithmetic mean;CUPTI only in separate control'),indent=2)+'\n')
print(len(body))
