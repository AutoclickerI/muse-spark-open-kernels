"""Build the wide decoder/MMA pipeline and matched raw-byte oracle batch."""
import base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent
from recipe import helper_body
from build import checker,task_sources,base
body=helper_body()+'\n'+(H/'wide_pipeline.py.inc').read_text();source=base+body
payload='LAB_RESOURCES={"compile_s":300,"run_s":300,"output_bytes":8744,"context_poison":"device"}\nimport torch,time,statistics,json,base64,lzma,hashlib,linecache\nPIPE_SOURCE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(source.encode())).decode())+')).decode()\nPIPE_CHECK='+repr(checker)+'\nPIPE_TASK='+repr(task_sources)+'\n'+(H/'pipeline_lab.py.inc').read_text()
compile(payload,'<P10 pipeline lab>','exec');(H/'pipeline_payload.py').write_text(payload);(H/'pipeline_source.py').write_text(source)
(H/'pipeline_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest()),indent=2)+'\n');print(len(payload))
