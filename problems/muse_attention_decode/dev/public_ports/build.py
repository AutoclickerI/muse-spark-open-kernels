"""Pack reviewed ports and exact S19 controls; no donor driver is imported."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'out';o.mkdir(exist_ok=True)
cu=(H/'lee_kernels.cu.inc').read_text()+'\n'+(H/'lee_host.cu.inc').read_text()
lee='CUDA='+repr(cu)+'\n'+(H/'lee_runtime.py.inc').read_text();(o/'lee.py').write_text(lee);(o/'lee.cu').write_text(cu)
op=(H/'operatorstack.py.inc').read_text();(o/'operator.py').write_text(op)
meta=dict(operator_sha256=hashlib.sha256(op.encode()).hexdigest(),lee_sha256=hashlib.sha256(lee.encode()).hexdigest(),lee_cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),notices_sha256=hashlib.sha256((H/'NOTICES.txt').read_bytes()).hexdigest())
(o/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
