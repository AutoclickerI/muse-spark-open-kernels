"""One attention-host packet: S19 layer arms and finer same-binary setup oracles."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
import kernels_new as k
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(H.parent/'sliding_oracle'));import build_followup as old
def base():
    s=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text()
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_source' for t in n.targets))
    s=lzma.decompress(base64.b85decode(node.value.func.value.args[0].args[0].value)).decode()
    assert hashlib.sha256(s.encode()).hexdigest()=='62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a';return s

def build(hoist=False):
    (H/'out').mkdir(exist_ok=True)
    text=(R/'prof/lab/last_sliding_timeline.py').read_text();header=['import base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types','import torch','from torch.utils.cpp_extension import load_inline']
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('TL_ACTIVITY','TL_REPS') for t in n.targets):header.append(ast.get_source_segment(text,n))
        elif isinstance(n,ast.FunctionDef) and n.name in ('tl_activity','tl_times','tl_capture','tl_summarize','tl_union','tl_node_bytes','tl_footprints','tl_nbytes','cs_copy','cs_delta','cs_profile'):header.append(ast.get_source_segment(text,n))
    spec='import math,functools\nimport torch\n'+'\n'.join(old.definitions(R/'problems/muse_decode_layer'/n) for n in ('nvfp4.py','checks.py','reference.py'))
    attn='import math,functools\nimport torch\n'+'\n'.join(old.definitions(R/'problems/muse_attention_decode'/n) for n in ('checks.py','reference.py'))
    text=(H.parent/'early_history/lab_body.inc').read_text();checker=next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='eh_checker')
    text=(H.parent/'window_rebuild/diagnostic.py.inc').read_text();checker+='\n'+next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='diag_checker')
    sources={f+a+'A':k.source(f,a) for f in ('d','v') for a in (('A','Q') if f=='d' else ('A','Q','T'))};sources.update({f+'M':k.oracle(f) for f in ('d','v')})
    template=(H/'lab.py.inc').read_text();modes=k.MODES
    if hoist:
        sources={f+a+'A':k.source(f,a) for f in ('d','v') for a in ('A','H')};modes={}
        template=template.replace('42000037+104729*i for i in range(12)','52000073+104729*i for i in range(16)').replace('ROUNDS=8','ROUNDS=10')
        template=template.replace("names=['A','A2','RAW','Q']+(['T'] if L==8 else [])","names=['A','A2','RAW','H']")
        template=template.replace("(('A','Q') if L==1 else ('A','Q','T'))","('A','H')")
        template=template.replace('            sample=setup_oracle(B,L,seed+7000000)\n            for n in ORACLE_MODES:oracle[n].append(sample[n])\n','')
        template=template.replace('oracle_TASK=2*len(SEEDS)','oracle_TASK=0')
        template=template.replace('Q=one query-prep owner plus cluster Q copy; T=parallel within-CTA pair merge then DSM cross-CTA merge.','H=cooperatively preloaded page/physical-row map in registers before PDL wait, tile-owner warp shuffle replaces per-tile global page loads; all math and DSM merge unchanged.')
    body='\n'.join(header)+'\nPUBLIC_SPEC={"torch":torch}\nexec('+repr(spec)+',PUBLIC_SPEC)\nATTN_SPEC={"torch":torch}\nexec('+repr(attn)+',ATTN_SPEC)\n'+checker+'\nBASE='+repr(base())+'\nSOURCES='+repr(sources)+'\nORACLE_MODES='+repr(modes)+'\n'+(H.parent/'sliding_oracle/runtime.py').read_text()+'\n'+template
    compile(body,'p5_setup_tree','exec');(H/('out/hoist_body.py' if hoist else 'out/body.py')).write_text(body)
    resources=dict(compile_s=200 if hoist else 250,run_s=400 if hoist else 500,output_bytes=8744,context_poison='none')
    blob=base64.b85encode(lzma.compress(body.encode())).decode();packet='import base64,lzma,linecache\nLAB_RESOURCES='+json.dumps(resources,separators=(',',':'))+'\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n_FN="<p5-setup-tree>"\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n_SETUP_LAB=globals()["lab"]\ndef lab(deadline,weights,state=None):return _SETUP_LAB(deadline,weights,state)\n'
    p=H/('out/hoist.py' if hoist else 'out/setup_tree.py');p.write_text(packet);print(p,hashlib.sha256(packet.encode()).hexdigest());return body
if __name__=='__main__':build('--hoist' in sys.argv)
