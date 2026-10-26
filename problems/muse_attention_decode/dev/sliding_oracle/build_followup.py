"""Whole-layer confirmation and real projection/PDL probes after the stage oracle."""
import ast,base64,hashlib,lzma
from pathlib import Path
import source,structural
H=Path(__file__).resolve().parent;R=H.parents[3]

def definitions(path):
    s=path.read_text();return '\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef)))

def build():
    (H/'out').mkdir(exist_ok=True);base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
    assert hashlib.sha256(base.encode()).hexdigest()=='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
    text=(R/'prof/lab/last_sliding_timeline.py').read_text();header=['import base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types','import torch','from torch.utils.cpp_extension import load_inline']
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('TL_ACTIVITY','TL_REPS') for t in n.targets):header.append(ast.get_source_segment(text,n))
        elif isinstance(n,ast.FunctionDef) and n.name in ('tl_activity','tl_times','tl_capture','tl_summarize','tl_union','tl_node_bytes','tl_footprints','tl_nbytes','cs_copy','cs_delta','cs_profile'):header.append(ast.get_source_segment(text,n))
    spec='import math,functools\nimport torch\n'+'\n'.join(definitions(R/'problems/muse_decode_layer'/n) for n in ('nvfp4.py','checks.py','reference.py'))
    attn='import math,functools\nimport torch\n'+'\n'.join(definitions(R/'problems/muse_attention_decode'/n) for n in ('checks.py','reference.py'))
    text=(H.parent/'early_history/lab_body.inc').read_text();checker=next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='eh_checker');assert '.8*tol' not in checker
    text=(H.parent/'window_rebuild/diagnostic.py.inc').read_text();checker+='\n'+next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='diag_checker')
    sources={}
    for f in ('d','v'):
        sources[f+'OA']=source.original(f);sources[f+'EA']=structural.source(f)
        sources[f+'CLOCK']=structural.context_clock(f);sources[f+'ECLOCK']=structural.context_clock(f,True)
    body='\n'.join(header)+'\nPUBLIC_SPEC={"torch":torch}\nexec('+repr(spec)+',PUBLIC_SPEC)\nATTN_SPEC={"torch":torch}\nexec('+repr(attn)+',ATTN_SPEC)\n'+checker+'\nBASE='+repr(base)+'\nSOURCES='+repr(sources)+'\n'+(H/'runtime.py').read_text()+'\n'+(H/'followup.py.inc').read_text()
    compile(body,'p5_sliding_followup','exec');(H/'out/followup_body.py').write_text(body)
    blob=base64.b85encode(lzma.compress(body.encode())).decode();packet='import base64,lzma,linecache\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n_FN="<p5-sliding-exp-pair>"\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n_PAIR=lab\ndef lab(deadline,weights,state=None):return _PAIR(deadline,weights,state)\n'
    p=H/'out/sliding_exp_pair.py';p.write_text(packet);print(p,hashlib.sha256(packet.encode()).hexdigest());return body

if __name__=='__main__':build()
