"""Self-preparing whole-layer sizing on any spare import-time host."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(H.parent/'setup_tree'));import pack_setup
sys.path.insert(0,str(H.parent/'sliding_oracle'));import build_followup
def build():
    text=(R/'prof/lab/last_sliding_timeline.py').read_text();header=['import gc,hashlib,json,linecache,math,re,statistics,time,types','import torch','from torch.utils.cpp_extension import load_inline']
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('TL_ACTIVITY','TL_REPS') for t in n.targets):header.append(ast.get_source_segment(text,n))
        elif isinstance(n,ast.FunctionDef) and n.name in ('tl_activity','tl_times','tl_capture','tl_summarize','tl_union','tl_node_bytes','tl_footprints','tl_nbytes','cs_copy','cs_delta','cs_profile'):header.append(ast.get_source_segment(text,n))
    spec='import math,functools\nimport torch\n'+'\n'.join(build_followup.definitions(R/'problems/muse_decode_layer'/n) for n in ('nvfp4.py','checks.py','reference.py'))
    attn='import math,functools\nimport torch\n'+'\n'.join(build_followup.definitions(R/'problems/muse_attention_decode'/n) for n in ('checks.py','reference.py'))
    text=(H.parent/'early_history/lab_body.inc').read_text();checker=next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='eh_checker')
    body='\n'.join(header)+'\nPUBLIC_SPEC={"torch":torch}\nexec('+repr(spec)+',PUBLIC_SPEC)\nATTN_SPEC={"torch":torch}\nexec('+repr(attn)+',ATTN_SPEC)\n'+checker+'\nBASE='+repr(pack_setup.base())+'\nFINISH_CUDA='+repr((H/'finish.cu').read_text())+'\n'+(H/'probe.py.inc').read_text()
    compile(body,'P5finishsizing','exec');(H/'out').mkdir(exist_ok=True);(H/'out/body.py').write_text(body)
    blob=base64.b85encode(lzma.compress(body.encode())).decode();packet='import base64,lzma,linecache\nLAB_RESOURCES={"compile_s":220,"run_s":480,"output_bytes":8744,"context_poison":"none"}\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n_FN="<P5finishsizing>"\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n_PF=globals()["lab"]\ndef lab(deadline,weights,state=None):return _PF(deadline,weights,state)\n'
    p=H/'out/sizing.py';p.write_text(packet);print(p,hashlib.sha256(packet.encode()).hexdigest());return packet
if __name__=='__main__':build()
