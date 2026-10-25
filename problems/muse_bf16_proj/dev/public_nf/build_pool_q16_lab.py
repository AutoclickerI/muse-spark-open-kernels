"""Package geometry and corrected-packer paired experiments, preserving attribution."""
import ast
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'nf_port_lab.py').read_text();tree=ast.parse(s)
helpers=s.split('\nPORT_SOURCE =',1)[0]
n=next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(x,'id','')=='S13_SOURCE' for x in n.targets))
source=ast.get_source_segment(s,n)
body=helpers+'\nimport hashlib,base64,zlib\nPORT_SOURCE = '+repr((H/'port_body.py').read_text())+'\n'+source+'\n'+(H/'pool_q16_lab.py.inc').read_text()
compile(body,'<pool-q16 lab>','exec');(H/'pool_q16_lab.py').write_text(body)
