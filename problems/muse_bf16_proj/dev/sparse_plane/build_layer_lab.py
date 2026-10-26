from pathlib import Path
import ast,base64,lzma,json,hashlib
H=Path(__file__).resolve().parent;R=H.parents[3];S=R/'problems/muse_decode_layer/dev/s15'
helpers=(S/'ab_helpers.py.inc').read_text();wanted={'AB_CASES','AB_LOADS','ab_label','ab_namespace','ab_copy','ab_graph','ab_time'};parts=[]
for n in ast.parse(helpers).body:
    name=n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
    if name in wanted:parts.append(ast.get_source_segment(helpers,n))
problem=R/'problems/muse_decode_layer';reference='\n'.join(x for x in (problem/'reference.py').read_text().splitlines() if not x.startswith(('from checks import','from nvfp4 import')))
checker='\n'.join(x for text in ((problem/'checks.py').read_text(),(problem/'nvfp4.py').read_text(),reference) for x in text.splitlines() if not x.startswith('from __future__ import'))
body='import torch,types,time,statistics,math\n'+'\n\n'.join(parts)+'\nPOOL_CHECKER='+repr(checker)+'\nPOOL_PATCH='+repr((H/'layer_patch.py.inc').read_text())+'\n'+(H/'layer_lab.py.inc').read_text()
encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode();tag=hashlib.sha256(body.encode()).hexdigest()
s='import base64,lzma,linecache,hashlib\n_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\nassert hashlib.sha256(_body.encode()).hexdigest()=='+repr(tag)+'\n_file="<p10-pool-layer-'+tag[:12]+'>"\nlinecache.cache[_file]=(len(_body),None,_body.splitlines(True),_file)\nexec(compile(_body,_file,"exec"),globals())\n_pool_lab=lab\ndef lab(deadline,weights,state=None):\n    return _pool_lab(deadline,weights,state)\n'
compile(s,'<pool layer payload>','exec');(H/'pool_layer_payload.py').write_text(s)
(H/'layer_lab_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),body_sha256=tag,bytes=len(s.encode()),base_sha256='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb',shapes=12,seeds=16,checks=768),indent=2)+'\n');print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
