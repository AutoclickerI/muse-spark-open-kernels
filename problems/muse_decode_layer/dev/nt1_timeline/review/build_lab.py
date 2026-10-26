"""Pack independent review arms with exact S15 stage control and a lossless seed-level receipt."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[4];sys.path.insert(0,str(D));import build
text=(R/'prof/lab/moe_nt1_reuse.py').read_text()
_,cell,_=build.original();cell=cell.replace('smallm_','rv0_').replace('_SMALLM_MODULE','_RV0_MODULE').replace('SMALLM_FLAGS_BUILT','RV0_FLAGS_BUILT').replace('SMALLM_VAR_','RV0_')
text+='\n'+cell
m=json.loads((D/'manifest.json').read_text())
for arm,spec in m.items():
 c=(R/spec['cell']).read_text();assert hashlib.sha256(c.encode()).hexdigest()==spec['cell_sha256'];text+='\n'+c
text+='\n'+(D/'receipt.py').read_text()+'\n'+(D/'lab.py.inc').read_text();ast.parse(text)
expanded=text;digest=hashlib.sha256(expanded.encode()).hexdigest();blob=base64.b85encode(lzma.compress(expanded.encode(),preset=9)).decode()
text='import base64,lzma,linecache,hashlib\n_p3a_body=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n'
text+='assert hashlib.sha256(_p3a_body.encode()).hexdigest()=='+repr(digest)+'\n_p3a_file="<p3a-review-"+'+repr(digest[:12])+'+">"\n'
text+='linecache.cache[_p3a_file]=(len(_p3a_body),None,_p3a_body.splitlines(True),_p3a_file)\nexec(compile(_p3a_body,_p3a_file,"exec"),globals())\n_p3a_lab=lab\ndef lab(deadline,weights):\n    return _p3a_lab(deadline,weights)\n'
p=R/'prof/lab/moe_p3a_review.py';p.write_text(text);print(p,hashlib.sha256(text.encode()).hexdigest(),len(text),'expanded',digest)
(D/'payload.json').write_text(json.dumps(dict(payload=str(p.relative_to(R)),sha256=hashlib.sha256(text.encode()).hexdigest(),base_sha256=build.BASE_SHA,expanded_sha256=digest,expanded_bytes=len(expanded),packed_bytes=len(text),arms=m,requested_seeds=16,M=[1,8,16,32]),indent=2)+'\n')
