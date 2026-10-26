"""Exact-S15 whole-layer pair, packed structural arms, clocks and FA3 scheduler introspection."""
import ast,base64,hashlib,lzma,sys
from pathlib import Path
import kernel,build_layer
H=Path(__file__).resolve().parent;R=H.parents[3]
CONFIGS={'WG2C2':(2,False),'WG2C4':(4,False),'STREAM2':(2,True),'STREAM4':(4,True)}

def build(pipeline=False):
    (H/'out').mkdir(exist_ok=True)
    base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text();assert hashlib.sha256(base.encode()).hexdigest()=='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
    text=(R/'prof/lab/last_sliding_timeline.py').read_text();header=['import gc,hashlib,json,linecache,math,statistics,time,types,base64,lzma','import torch','from torch.utils.cpp_extension import load_inline']
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('TL_ACTIVITY','TL_REPS') for t in n.targets):header.append(ast.get_source_segment(text,n))
        elif isinstance(n,ast.FunctionDef) and (n.name in ('tl_activity','tl_times','tl_capture','tl_summarize','tl_union','tl_node_bytes','tl_footprints','tl_nbytes','cs_copy','cs_delta','cs_profile')):header.append(ast.get_source_segment(text,n))
    spec='import math,functools\nimport torch\n'
    for name in ('nvfp4.py','checks.py','reference.py'):
        text=(R/'problems/muse_decode_layer'/name).read_text();spec+='\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef)))+'\n'
    text=(H.parent/'early_history/lab_body.inc').read_text();checker=next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='eh_checker')
    assert '.8*tol' not in checker
    configs={'STREAM2':(2,True),'PIPE2':(2,True)} if pipeline else CONFIGS
    sources={n:kernel.source(c,stream=s,pipeline=n=='PIPE2') for n,(c,s) in configs.items()}
    sources.update({n+'_CLOCK':kernel.source(c,stream=s,clocks=True,pipeline=n=='PIPE2') for n,(c,s) in configs.items() if pipeline or n in ('WG2C2','STREAM2')})
    body='\n'.join(header)+'\nPUBLIC_SPEC={"torch":torch}\nexec('+repr(spec)+',PUBLIC_SPEC)\n'+checker+'\nBASE='+repr(base)+'\n'+build_layer.runtime(sources)+'\n'+(H/'pair.py.inc').read_text()
    if pipeline:
        body+="\nwr_TOPOLOGY={'STREAM2':2,'PIPE2':2}\n"
        body=body.replace('22100027+104729*i for i in range(24)','25100021+104729*i for i in range(18)')
        body=body.replace("names=('A','A2','F8','WG2C2','WG2C4','STREAM2','STREAM4','RAW2')", "names=('A','A2','RAW2','PIPE2')")
        body=body.replace("name!='RAW2'", "name not in ('RAW2','PIPE2')")
        body=body.replace("('WG2C2','STREAM2')", "('STREAM2','PIPE2')")
        body=body.replace("name+'_CLOCK',True,clocks", "name+'_CLOCK',False,clocks")
        body=body.replace("name.startswith('STREAM')", "name.startswith(('STREAM','PIPE'))")
        body=body.replace('structural window pair vs exact S15: A/A2, FA3split8(v32only),2WG residentC2/C4,streaming deferred-new-KVC2/C4,RAW2 same stream without early edge.', 'pipeline follow-up vs exact S15: A/A2, measured RAW2 and PIPE2; ordinary edges in both; next QK before its V transpose, overlap PV/softmax, same FP8 work.')
        body=body.replace('24 disjoint seeds10cold', '18 disjoint seeds10cold')
    compile(body,'p5_window_pair','exec');(H/'out/pair_body.py').write_text(body)
    blob=base64.b85encode(lzma.compress(body.encode())).decode()
    packet='import base64,lzma,linecache\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n_FN="<p5-window-pair>"\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n_PAIR=lab\ndef lab(deadline,weights,state=None):return _PAIR(deadline,weights,state)\n'
    p=H/('out/pipeline_pair.py' if pipeline else 'out/pair.py');p.write_text(packet);print(p,hashlib.sha256(packet.encode()).hexdigest());return body

if __name__=='__main__':build('--pipeline' in sys.argv)
