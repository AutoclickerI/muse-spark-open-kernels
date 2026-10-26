import ast,base64,hashlib,importlib.util,json,zlib
from pathlib import Path
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('actbuilder',H/'build.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.main()
s=(H.parent/'balance/expanded.py').read_text();core=s[:s.index('BALANCE_CELL=')]
body=core+'\nS18_SOURCE='+repr(b.source())+'\nACT_CELL='+repr((H/'activation_cell.py').read_text())+'\n'+(H/'runtime.py.inc').read_text()+'\n'+(H/'lab.py.inc').read_text()
compile(body,'<p3b-act>','exec');data=base64.b85encode(zlib.compress(body.encode(),9)).decode();p='import base64,zlib\nexec(compile(zlib.decompress(base64.b85decode('+repr(data)+')),"<p3b-act>","exec"))\ndef lab(deadline,weights,state=None):return act_lab(deadline,weights,state)\n';(H/'act_payload.py').write_text(p);(H/'expanded.py').write_text(body);(H/'payload.json').write_text(json.dumps(dict(sha256=hashlib.sha256(p.encode()).hexdigest(),bytes=len(p)),indent=2)+'\n')
