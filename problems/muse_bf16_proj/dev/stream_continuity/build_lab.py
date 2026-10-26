"""One full-six-load continuity batch, with unchanged copied public TASK criteria."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
from build import build
H=Path(__file__).resolve().parent;R=H.parents[3]

def functions(path,names):
    s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)

def checker():
    sources=[(R/'problems/muse_decode_layer'/p).read_text() for p in ('nvfp4.py','checks.py','reference.py')]
    funcs={n.name:(s,n) for s in sources for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    roots={'generate_weights','collect','check_implementation','ref_front','ref_back'}
    while True:
        more={n.id for f in roots for n in ast.walk(funcs[f][1]) if isinstance(n,ast.Name) and n.id in funcs}
        if more<=roots:break
        roots|=more
    result='import math,functools\nimport torch\n'
    for s in sources:
        for n in ast.parse(s).body:
            if isinstance(n,(ast.Assign,ast.AnnAssign)):result+=ast.get_source_segment(s,n)+'\n'
    for name,(s,n) in funcs.items():
        if name in roots:result+=ast.get_source_segment(s,n)+'\n'
    return result

def main():
    sources=build();timeline=R/'problems/muse_decode_layer/dev/s15_timeline'
    generated=(timeline/'timeline_generated.py.inc').read_text()
    activity=next(ast.literal_eval(n.value) for n in ast.parse(generated).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='T15_ACTIVITY')
    bundle=base64.b85encode(lzma.compress(json.dumps(sources,separators=(',',':')).encode(),preset=9)).decode()
    body='import ast,base64,gc,hashlib,json,linecache,lzma,math,statistics,struct,time,types\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
    body+='SC_SOURCES=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\nSC_CHECKER='+repr(checker())+'\nT15_ACTIVITY='+repr(activity)+'\nT15_REPS=3\n'
    body+=(timeline/'analysis.py').read_text()+'\n_sc_old_stage_name=stage_name\ndef stage_name(role,name):\n    return "moe_heavy" if "chunk::kernel" in name else "moe_small" if "moe_sc_late" in name else _sc_old_stage_name(role,name)\n'
    body+=functions(R/'problems/muse_decode_layer/dev/s10/ablate_lab.py.inc',('ab_copy','ab_tensors'))+'\n'
    body+=functions(timeline/'lab_body.inc',('t15_activity','t15_capture','t15_profile','t15_signature','t15_busiest'))+'\n'
    body+=functions(H.parent/'residual_tail/lab_body.inc',('rt_task_fixture',))+'\n'
    body+=(H/'setup.py.inc').read_text()+'\n'+(H/'lab_body.inc').read_text();compile(body,'sc-lab','exec')
    tag=hashlib.sha256(body.encode()).hexdigest();encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    boot='import base64,hashlib,linecache,lzma\n_SC_BODY=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\nassert hashlib.sha256(_SC_BODY.encode()).hexdigest()=='+repr(tag)+'\n_SC_FILE="<p7-continuity-'+tag[:12]+'>"\nlinecache.cache[_SC_FILE]=(len(_SC_BODY),None,_SC_BODY.splitlines(True),_SC_FILE)\nexec(compile(_SC_BODY,_SC_FILE,"exec"),globals())\n_SC_RUN=lab\ndef lab(deadline,weights,state=None):\n    return _SC_RUN(deadline,weights,state)\n'
    (H/'lab_generated.py.inc').write_text(body);(H/'lab_continuity.py').write_text(boot)
    (H/'lab_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(boot.encode()).hexdigest(),body_sha256=tag,arm_sha256={a:hashlib.sha256(s.encode()).hexdigest() for a,s in sources.items()},seeds=[21450123+130363*i for i in range(16)],TASK_checks=1728,preselected='X',host='muse_prefill_layer',measured_problem='muse_decode_layer',host_reason='decode daily20checks exhausted; same whole-decode lab as a prefill-host check',phase='import',budget_s=500),indent=2)+'\n')
    print('payload',hashlib.sha256(boot.encode()).hexdigest()[:8])
if __name__=='__main__':main()
