"""Reviewed public projection source behind the frozen S19 partial-output ABI."""
import ast,base64,hashlib,json,lzma,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];O=H/'out';O.mkdir(exist_ok=True)
base=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text();assert hashlib.sha256(base.encode()).hexdigest().startswith('8b415553')
# GPU-only histograms of existing reviewed NF packers; no timed code changes.
s=(R/'problems/muse_bf16_proj/dev/public_nf/port_body.py').read_text();encoded=next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='NF_VENDOR' for t in n.targets));packed=next(n.value for n in ast.walk(ast.parse(encoded)) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('eNr'));vendor=zlib.decompress(base64.b64decode(packed)).decode();gpu=[]
for n in ast.parse(vendor).body:
 if not isinstance(n,ast.FunctionDef) or not n.name.endswith('_c12_bytes'):continue
 f=ast.get_source_segment(vendor,n)
 if 'hist = torch.bincount(eh.flatten(), minlength=128).cpu()' in f:
  f=f.replace('hist = torch.bincount(eh.flatten(), minlength=128).cpu()','hist = torch.bincount(eh.flatten(), minlength=128)').replace('window = torch.stack([hist[b:b + 8].sum() for b in range(121)])','window = hist.unfold(0,8,1).sum(1)');gpu.append(f)
body='\nimport time,types,hashlib\nPP_GPU_PACK_SOURCE='+repr('\n\n'.join(gpu))+'\n'+(H/'operator_kernels.py.inc').read_text()+'\n'+(H/'coded.py.inc').read_text()+'\n'+(H/'runtime.py.inc').read_text()
source=base+body;compile(source,'<P10 reviewed public projection ABI>','exec');(O/'source.py').write_text(source);(O/'helper.py.inc').write_text(body);(H/'port_body.py.inc').write_text(body)
(H/'source_manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),helper_sha256=hashlib.sha256(body.encode()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),NF_GPU_histograms=len(gpu),operator_pin=json.loads((H/'review.json').read_text())['pin'],operator_adapted_sha256=hashlib.sha256((H/'operator_kernels.py.inc').read_bytes()).hexdigest()),indent=2)+'\n');print(len(source))
