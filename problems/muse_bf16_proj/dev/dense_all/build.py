"""Build the complete dense coding/fixed-cost recipe on the current layer."""
import ast,base64,hashlib,json,lzma,zlib
from component_source import projection_source
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
base=(R/'problems/muse_decode_layer/cand_r3_best7_s18_final.py').read_text()
from recipe import helper_body
body=helper_body()
s=base+body;compile(s,'<P10 all dense>','exec');(H/'candidate.py').write_text(s)
(H/'adapter.py').write_text(projection_source()+body)
# Checker source is copied, not imported, and is used only by the lab.
ref=(R/'problems/muse_bf16_proj/reference.py').read_text();names={'FP32_LAYERS','REL_L2_TOL','ROUTER_TOL','PROBE_TOL','ref_kernel','probe_rows','check_implementation'};nodes=[n for n in ast.parse(ref).body if getattr(n,'name',None) in names or isinstance(n,ast.Assign) and any(getattr(t,'id',None) in names for t in n.targets)];checker=(R/'problems/muse_bf16_proj/checks.py').read_text()+'\n'+'\n'.join(ast.get_source_segment(ref,n) for n in nodes)
layer=(R/'problems/muse_decode_layer/reference.py').read_text();layer=layer.replace('from checks import compare, query_precision_bound, relative_l2','').replace('from nvfp4 import BLOCK, dequantize, random_weights','')
nvfp4=(R/'problems/muse_decode_layer/nvfp4.py').read_text();checks=(R/'problems/muse_decode_layer/checks.py').read_text();task_sources=[nvfp4,checks,layer]
payload='LAB_RESOURCES = {"compile_s": 400, "run_s": 250, "output_bytes": 8744, "context_poison": "device"}\nimport base64,lzma,torch\nDA_SOURCE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(s.encode(),preset=9)).decode())+')).decode()\nDA_CHECK_SOURCE='+repr(checker)+'\nDA_TASK_SOURCES='+repr(task_sources)+'\n'+(H/'protocol.py.inc').read_text()+'\n'+(H/'lab.py.inc').read_text();compile(payload,'<P10 all dense lab>','exec');(H/'payload.py').write_text(payload)
(H/'manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),bytes=len(payload.encode())),indent=2)+'\n');print(len(s),len(payload))
