"""Pack exact S15 FA3 dispatch, official attention checker and full-name CUPTI."""
import ast,base64,hashlib,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]

def definitions(path):
    text=path.read_text()
    return '\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body
        if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef)))

def build():
    (H/'out').mkdir(exist_ok=True)
    base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
    assert hashlib.sha256(base.encode()).hexdigest()=='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
    spec='import math\nimport torch\n'+definitions(R/'problems/muse_attention_decode/checks.py')+'\n'+definitions(R/'problems/muse_attention_decode/reference.py')
    cpp=(R/'problems/muse_decode_layer/dev/timeline/activity.cpp').read_text()
    body='BASE='+repr(base)+'\nSPEC='+repr(spec)+'\nACTIVITY='+repr(cpp)+'\n'+(H/'diagnostic.py.inc').read_text()
    compile(body,'window_diagnostic','exec')
    (H/'out/diagnostic_body.py').write_text(body)
    blob=base64.b85encode(lzma.compress(body.encode())).decode()
    packet='import base64,lzma,linecache\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n_FN="<p5-window-diagnostic>"\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n_DIAG=lab\ndef lab(deadline,weights,state=None):return _DIAG(deadline,weights,state)\n'
    p=H/'out/diagnostic.py';p.write_text(packet);print(p,hashlib.sha256(packet.encode()).hexdigest())
    return body

if __name__=='__main__':build()
