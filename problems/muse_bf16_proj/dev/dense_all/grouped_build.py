"""Build the wide decoder/MMA pipeline and matched raw-byte oracle batch."""
import base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent
from recipe import helper_body
from build import checker,task_sources,base
body=helper_body()+'\n'+(H/'wide_pipeline.py.inc').read_text()+'\n'+(H/'grouped.py.inc').read_text();source=base+body
payload='LAB_RESOURCES={"compile_s":300,"run_s":300,"output_bytes":8744,"context_poison":"device"}\nimport torch,time,statistics,json,base64,lzma,hashlib,linecache\nPIPE_SOURCE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(source.encode())).decode())+')).decode()\nPIPE_CHECK='+repr(checker)+'\nPIPE_TASK='+repr(task_sources)+'\n'+(H/'pipeline_lab.py.inc').read_text().replace("(bn,64,1,2,2,coded,2) for bn in (64,128,256)","(bn,64,1,2,4,coded,2) for bn in (64,128)").replace("src=m['da_pipeline_source'](ns,instances,piped)","src=m['da_grouped_source'](ns,instances) if piped else m['da_wide_source'](ns,instances,False).replace('p10_daw_normal','p10_daw_grouped_control')").replace("(p,c,bn,s) for p in (False,True) for c in (True,False) for bn in (t,t//2)","(p,c,bn,s) for p in (False,True) for c in (True,False) for bn in (128,64)").replace("instances.index((bn,64,1,2,2,coded,2))","instances.index((bn,64,1,2,4,coded,2))").replace("'META P10 register-A decoder/MMA overlap factorial,raw exact oracle,8seeds8cold,actual neighbours'","'META P10 grouped K128 decoder/MMA,adjacent-token grid,raw exact oracle,8seeds8cold actual neighbours'")
compile(payload,'<P10 pipeline lab>','exec');(H/'grouped_payload.py').write_text(payload);(H/'grouped_source.py').write_text(source)
(H/'grouped_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest()),indent=2)+'\n');print(len(payload))
