"""One independent P10 H200 batch: eight arms, four M, eight seeds."""
from pathlib import Path
import base64,zlib,hashlib
H=Path(__file__).resolve().parent;R=H.parents[3]
s='import torch,base64,zlib\nSP_SOURCE='+repr((H/'adapter.py').read_text())+'\nBASE_SOURCE=zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress((R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_bytes(),9)).decode())+')).decode()\n'+(H/'lab.py.inc').read_text()
compile(s,'<P10 sparse pool batch>','exec');(H/'p10_sparse_pool_lab.py').write_text(s);print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
