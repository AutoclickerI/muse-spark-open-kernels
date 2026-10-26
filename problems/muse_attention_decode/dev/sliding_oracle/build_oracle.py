"""Pack all four loads and same-binary oracle arms into one attention-host payload."""
import ast,base64,hashlib,lzma
from pathlib import Path
import source
H=Path(__file__).resolve().parent;R=H.parents[3]

def build():
    (H/'out').mkdir(exist_ok=True)
    base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
    assert hashlib.sha256(base.encode()).hexdigest()=='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
    spec='import math,functools\nimport torch\n'
    for name in ('checks.py','reference.py'):
        s=(R/'problems/muse_attention_decode'/name).read_text();spec+='\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef)))+'\n'
    text=(H.parent/'window_rebuild/diagnostic.py.inc').read_text()
    helpers='\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in ('diag_host','diag_activity','diag_call','diag_checker'))
    sources={f+k:(source.original(f) if k=='A' else source.source(f,clocks=k=='CLOCK')) for f in ('d','v') for k in ('A','M','CLOCK')}
    body='import base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
    body+='BASE='+repr(base)+'\nSPEC='+repr(spec)+'\nACTIVITY='+repr((R/'problems/muse_decode_layer/dev/timeline/activity.cpp').read_text())+'\nSOURCES='+repr(sources)+'\n'
    body+=(H/'runtime.py').read_text()+'\n'+helpers+'\n'+(H/'lab.py.inc').read_text()
    compile(body,'sliding_oracle','exec');(H/'out/body.py').write_text(body)
    blob=base64.b85encode(lzma.compress(body.encode())).decode()
    packet='import base64,lzma,linecache\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n_FN="<p5-sliding-oracle>"\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n_LAB=lab\ndef lab(deadline,weights,state=None):return _LAB(deadline,weights,state)\n'
    p=H/'out/sliding_oracle.py';p.write_text(packet);print(p,hashlib.sha256(packet.encode()).hexdigest());return body

if __name__=='__main__':build()
