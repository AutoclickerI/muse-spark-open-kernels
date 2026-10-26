"""Batch contiguous weight-bank and split/depth arms, keeping same-geometry controls."""
import base64,hashlib,json,lzma
from pathlib import Path
import ss_build,packed_source,ss_source
H=Path(__file__).resolve().parent
body='\nPS_CONFIGS='+repr(packed_source.CONFIGS)+'\nPS_CPP='+repr(ss_source.CPP)+'\nPS_CUDA='+repr(packed_source.source())+'\n'+(H/'packed_runtime.py.inc').read_text();unpacked=ss_build.body.replace('k%(64*s*KG))return -1;','(k%64 || (KG>1 && k%(64*s*KG))))return -1;');assert unpacked!=ss_build.body
source=ss_build.base+unpacked+body
previous=json.loads((H/'ss_points.json').read_text());stock={(t,r):(ss_source.CONFIGS.index(tuple(c)),s) for t,r,c,s in previous}
payload='LAB_RESOURCES={"compile_s":220,"run_s":300,"output_bytes":8744,"context_poison":"device"}\nimport torch,triton,time,hashlib,json,statistics,linecache,types,base64,lzma\nPS_BASE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(source.encode())).decode())+')).decode()\nSS_CHECK='+repr(ss_build.checker)+'\nSS_TASK='+repr(ss_build.tasks)+'\nPS_STOCK='+repr(stock)+'\n'+(H/'codec.py.inc').read_text()+'\n'+(H/'packed_lab.py.inc').read_text();compile(payload,'<P10 packed weights>','exec');(H/'generated/packed_payload.py').write_text(payload);(H/'generated/packed_source.py').write_text(source)
(H/'packed_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),configs=packed_source.CONFIGS,seeds=[18000319+104729*i for i in range(8)],weight_seed=18000311,router_seed=18000313),indent=2)+'\n');print(len(payload))
