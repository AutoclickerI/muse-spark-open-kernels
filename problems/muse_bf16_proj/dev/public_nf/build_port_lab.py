import ast,base64,zlib
from pathlib import Path
H=Path(__file__).resolve().parent
helpers=(H/'register_x_lab.py').read_text().split('\nSOURCES =',1)[0]
helpers=helpers.replace("    if all(k in weights","    weights = {} if weights is None else weights\n    if all(k in weights")
port=(H/'port_body.py').read_text();base=(H.parents[2]/'muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
s=helpers+'\nimport hashlib\nimport base64,zlib\nPORT_SOURCE = '+repr(port)+'\nS13_SOURCE = zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(base.encode(),9)).decode())+')).decode()\n'+(H/'port_lab.py.inc').read_text()
(H/'nf_port_lab.py').write_text(s);print('lab bytes',len(s))
