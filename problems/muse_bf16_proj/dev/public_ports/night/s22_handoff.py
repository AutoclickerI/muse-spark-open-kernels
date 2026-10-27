"""Rebase the 72f273a8-qualified P10 composition onto P4's S22 4b8c803d.

S22 expanded comes from the committed shipment blob (sha256 d1ccdc35), so P4's working edits cannot
change the base. Composition: the parent port points that won on held-out seeds (v1 unpooler_slice
dropped: 0.9966 there) and the P16 pool/router T16 carrier in place of the parent's CW8_S2 pool.
Writes out/s22_p10.py (expanded) and out/s22_p10_packed.py (S22's own packed shipment form) and
prints both identities.
"""
import ast,base64,hashlib,importlib.util,json,lzma,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
BLOB='43cad1581817ae161ae70588ac4819ef7ddb8002';SHIPPED='d1ccdc358ef5fb0f038a1d0e741465f512f498e90d39a16e56f3f0b0e47615f3'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
def s22():
 packed=subprocess.check_output(['git','cat-file','blob',BLOB],cwd=R,text=True);assert sha(packed)==SHIPPED
 blob=next(n.args[0].value for n in ast.walk(ast.parse(packed)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode')
 s=lzma.decompress(base64.b85decode(blob)).decode();assert sha(s).startswith('4b8c803d');return s,packed,blob
if __name__=='__main__':
 selection=json.loads((H/'s22_selection.json').read_text())
 sp=importlib.util.spec_from_file_location('ng_s22_patch',H/'patch.py');patch=importlib.util.module_from_spec(sp);sp.loader.exec_module(patch)
 base,packed,blob=s22();out=patch.apply(base,selection,selection['groups'])
 shipped=packed.replace(blob,base64.b85encode(lzma.compress(out.encode(),preset=9)).decode(),1).replace(sha(base),sha(out))
 (H/'out').mkdir(exist_ok=True);(H/'out/s22_p10.py').write_text(out);(H/'out/s22_p10_packed.py').write_text(shipped)
 print(json.dumps(dict(base_sha256=sha(base),sha256=sha(out),shipped_sha256=sha(shipped),groups=selection['groups'],points=selection['points'],pool=selection['pool'])))
