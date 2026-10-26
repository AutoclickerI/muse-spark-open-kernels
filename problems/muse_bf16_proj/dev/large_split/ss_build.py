"""Package the exact shared-weight token-warpgroup candidate matrix."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
import ss_source
H=Path(__file__).resolve().parent;R=H.parents[3]
base=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text();assert hashlib.sha256(base.encode()).hexdigest().startswith('8b415553')
body='\nSS_CONFIGS='+repr(ss_source.CONFIGS)+'\nSS_CPP='+repr(ss_source.CPP)+'\nSS_CUDA='+repr(ss_source.source())+'\n'+(H/'ss_runtime.py.inc').read_text()
ref=(R/'problems/muse_bf16_proj/reference.py').read_text();names={'FP32_LAYERS','REL_L2_TOL','ROUTER_TOL','PROBE_TOL','ref_kernel','probe_rows','check_implementation'};nodes=[n for n in ast.parse(ref).body if getattr(n,'name',None) in names or isinstance(n,ast.Assign) and any(getattr(t,'id',None) in names for t in n.targets)];checker=(R/'problems/muse_bf16_proj/checks.py').read_text()+'\n'+'\n'.join(ast.get_source_segment(ref,n) for n in nodes)
lref=(R/'problems/muse_decode_layer/reference.py').read_text().replace('from checks import compare, query_precision_bound, relative_l2','').replace('from nvfp4 import BLOCK, dequantize, random_weights','');tasks=[(R/'problems/muse_decode_layer/nvfp4.py').read_text(),(R/'problems/muse_decode_layer/checks.py').read_text(),lref]
s=base+body;f=H/'generated';f.mkdir(exist_ok=True);(f/'ss_source.py').write_text(s)
payload='LAB_RESOURCES={"compile_s":230,"run_s":300,"output_bytes":8744,"context_poison":"device"}\nimport torch,triton,time,hashlib,statistics,json,base64,lzma,linecache,types\nSS_BASE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(s.encode())).decode())+')).decode()\nSS_CHECK='+repr(checker)+'\nSS_TASK='+repr(tasks)+'\n'+(H/'ss_lab.py.inc').read_text();compile(payload,'<P10 SS lab>','exec');(f/'ss_payload.py').write_text(payload)
(H/'ss_manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),source_sha256=hashlib.sha256(s.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),configs=ss_source.CONFIGS,seeds=[16000319+104729*i for i in range(8)]),indent=2)+'\n');print(len(payload))
