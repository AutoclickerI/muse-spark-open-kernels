"""Package every large-shape mechanism arm and seed in one self-contained lab."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(H))
import source
base=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text();assert hashlib.sha256(base.encode()).hexdigest().startswith('8b415553')
native=ast.get_source_segment(source.T.read_text(),next(n for n in ast.parse(source.T.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='da_wide_source'))
code=native+'\n'+ast.get_source_segment(Path(source.__file__).read_text(),next(n for n in ast.parse(Path(source.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='source'));code=code.replace('def source(ns,mode):','def ls_source(ns,mode):').replace('mode in MODES','mode in LS_MODES').replace('da_wide_source(ns,INSTANCES','da_wide_source(ns,LS_INSTANCES')
body='import torch,triton,triton.language as tl,time,hashlib\nLS_MODES='+repr(source.MODES)+'\nLS_CONFIG='+repr(source.CONFIG)+'\nLS_INSTANCES='+repr(source.INSTANCES)+'\nLS_CPP='+repr(source.CPP)+'\n'+code+'\n'+(H/'runtime.py.inc').read_text()
# Reuse the unchanged official TASK sources, lab-only.
ref=(R/'problems/muse_bf16_proj/reference.py').read_text();names={'FP32_LAYERS','REL_L2_TOL','ROUTER_TOL','PROBE_TOL','ref_kernel','probe_rows','check_implementation'};nodes=[n for n in ast.parse(ref).body if getattr(n,'name',None) in names or isinstance(n,ast.Assign) and any(getattr(t,'id',None) in names for t in n.targets)];checker=(R/'problems/muse_bf16_proj/checks.py').read_text()+'\n'+'\n'.join(ast.get_source_segment(ref,n) for n in nodes)
lref=(R/'problems/muse_decode_layer/reference.py').read_text().replace('from checks import compare, query_precision_bound, relative_l2','').replace('from nvfp4 import BLOCK, dequantize, random_weights','');tasks=[(R/'problems/muse_decode_layer/nvfp4.py').read_text(),(R/'problems/muse_decode_layer/checks.py').read_text(),lref]
f=H/'generated';f.mkdir(exist_ok=True);(f/'source.py').write_text(base+'\n'+body)
payload='LAB_RESOURCES={"compile_s":260,"run_s":300,"output_bytes":8744,"context_poison":"device"}\nimport torch,triton,triton.language as tl,time,hashlib,statistics,json,base64,lzma,linecache,types\nLS_BASE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress((base+'\n'+body).encode())).decode())+')).decode()\nLS_CHECK='+repr(checker)+'\nLS_TASK='+repr(tasks)+'\n'+(H/'lab.py.inc').read_text()
compile(payload,'<P10 split oracle>','exec');(f/'payload.py').write_text(payload);manifest=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),source_sha256=hashlib.sha256((base+'\n'+body).encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),shapes=[[*k,*v] for k,v in source.CONFIG.items()],modes=source.MODES,seeds=[15000319+104729*i for i in range(8)])
(H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['payload_sha256'],len(payload))
