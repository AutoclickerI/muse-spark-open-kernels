"""Import-time attention host: synthesize full weights, prepare S14, pair all PDL fixes."""
import ast
import base64
import hashlib
import lzma
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def functions(path,names):
    s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body
        if isinstance(n,ast.FunctionDef) and n.name in names)


def main():
    base=(REPO/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
    candidate=(HERE/'audit_trial.py').read_text()
    synth='HIDDEN=8192;Q_SIZE=1024;QKV_SIZE=2304;D_EXPERT=4096;LOCAL_EXPERTS=32;N_EXPERTS=256\n'
    synth+='ROUTER_STD=.045;ROUTER_ALIGN=.07;ROUTER_BIAS_STD=.055;STREAM_SEED=5354;STREAM_RANK=4\n'
    synth+='BLOCK=16;E4M3_MAX=448.;SCALE_LOG2_MEAN=7.;SCALE_LOG2_STD=.4\n'
    synth+=functions(REPO/'problems/muse_decode_layer/nvfp4.py',('random_weights',))+'\n'
    synth+=functions(REPO/'problems/muse_decode_layer/reference.py',('_stream_basis','_router','generate_weights'))+'\n'
    helpers=functions(REPO/'problems/muse_decode_layer/dev/s10/ablate_lab.py.inc',
        ('ab_copy','ab_graph','ab_tensors','ab_gate','ab_time','ab_summary'))
    body='import ast,base64,functools,gc,hashlib,json,linecache,lzma,math,statistics,time,types\nimport torch\n'
    body+='PA_BASE='+repr(base)+'\nPA_CANDIDATE='+repr(candidate)+'\n'+synth+helpers+'\n'
    body+=(HERE/'lab_body.inc').read_text();compile(body,'pdl_audit_lab','exec')
    tag=hashlib.sha256(body.encode()).hexdigest()
    (HERE/'lab_body_generated.py.inc').write_text(body)
    zipped=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    boot='import base64,lzma,linecache,hashlib\n_PA_BODY=lzma.decompress(base64.b85decode('+repr(zipped)+')).decode()\n'
    boot+='assert hashlib.sha256(_PA_BODY.encode()).hexdigest()=='+repr(tag)+'\n_PA_FILE="<pdl-audit-'+tag[:12]+'>"\n'
    boot+='linecache.cache[_PA_FILE]=(len(_PA_BODY),None,_PA_BODY.splitlines(True),_PA_FILE)\nexec(compile(_PA_BODY,_PA_FILE,"exec"),globals())\n_PA_LAB=lab\ndef lab(deadline,weights,state=None):\n    return _PA_LAB(deadline,weights,state)\n'
    p=HERE/'lab_pdl_audit.py';p.write_text(boot)
    print('candidate',hashlib.sha256(candidate.encode()).hexdigest())
    print('payload',hashlib.sha256(boot.encode()).hexdigest(),'body',tag,'bytes',len(boot))


if __name__=='__main__':main()
