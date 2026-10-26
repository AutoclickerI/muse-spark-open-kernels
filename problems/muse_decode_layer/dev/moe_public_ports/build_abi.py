import ast,base64,hashlib,json,lzma
from pathlib import Path
from abi import operator_abi
H=Path(__file__).resolve().parent
p=(H/'out/public_moe.py').read_text();n=next(n for n in ast.parse(p).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='BUNDLE' for t in n.targets));blob=next(n.value for n in ast.walk(n) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000);b=json.loads(lzma.decompress(base64.b85decode(blob)))
abiop=b['operator']+'\n'+(H/'abi_helpers.py.inc').read_text()+'\n'+operator_abi(b['operator']);compile(abiop,'<abi-operator>','exec');(H/'operator_abi.py.inc').write_text(abiop.replace('\n# \n','\n#\n'))
b['abi_operator']=abiop;b['abi_finalize']=(H/'abi_finalize.cu.inc').read_text()
body='import base64,gc,hashlib,json,linecache,lzma,math,statistics,struct,time,types\nimport torch\nBUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(json.dumps(b).encode(),preset=9)).decode())+')))\n'+(H/'runtime.py.inc').read_text()+'\n'+(H/'abi_lab.py.inc').read_text()
compile(body,'<public-abi>','exec');(H/'out/public_moe_abi.py').write_text(body)
(H/'abi_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(body.encode()).hexdigest(),identities={k:hashlib.sha256(v.encode()).hexdigest() for k,v in b.items()},seeds=[275163221+104729*i for i in range(8)],weight_seed=275163217,rounds=4),indent=2)+'\n')
print('ABI payload',len(body))
