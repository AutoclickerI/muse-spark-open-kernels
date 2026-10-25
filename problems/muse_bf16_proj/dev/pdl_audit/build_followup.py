"""Pair the suspected first-tile cost with metadata-only, hints, and the prior full-d1 recipe."""
import ast
import base64
import hashlib
import lzma
from pathlib import Path
from patch import patch

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def metadata_only(base):
    fixed=patch(base);original={n.name:ast.get_source_segment(base,n) for n in ast.parse(base).body if isinstance(n,ast.FunctionDef)}
    lines=fixed.splitlines(True);changes=[]
    for n in ast.parse(fixed).body:
        if isinstance(n,ast.FunctionDef) and n.name in ('_gemv_kernel','_gb_kernel','_gi_kernel'):
            changes.append((n.lineno-1,n.end_lineno,original[n.name]+'\n'))
    for lo,hi,code in sorted(changes,reverse=True):lines[lo:hi]=[code]
    return ''.join(lines)


def main():
    base=(REPO/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
    meta=metadata_only(base);(HERE/'meta_trial.py').write_text(meta)
    prefix=(HERE/'lab_body_generated.py.inc').read_text().split('\ndef lab(',1)[0]
    prefix+='\nPA_HINT='+repr((HERE/'hint_trial.py').read_text())+'\nPA_META='+repr(meta)+'\nPA_D1='+repr((HERE/'d1_s14_trial.py').read_text())+'\n'
    body=prefix+(HERE/'followup_body.inc').read_text();compile(body,'pdl_followup','exec')
    (HERE/'followup_generated.py.inc').write_text(body);tag=hashlib.sha256(body.encode()).hexdigest()
    compressed=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    boot='import base64,lzma,linecache,hashlib\n_PA_BODY=lzma.decompress(base64.b85decode('+repr(compressed)+')).decode()\n'
    boot+='assert hashlib.sha256(_PA_BODY.encode()).hexdigest()=='+repr(tag)+'\n_PA_FILE="<pdl-followup-'+tag[:12]+'>"\n'
    boot+='linecache.cache[_PA_FILE]=(len(_PA_BODY),None,_PA_BODY.splitlines(True),_PA_FILE)\nexec(compile(_PA_BODY,_PA_FILE,"exec"),globals())\n_PA_LAB=lab\ndef lab(deadline,weights,state=None):\n    return _PA_LAB(deadline,weights,state)\n'
    p=HERE/'lab_pdl_followup.py';p.write_text(boot)
    print('meta',hashlib.sha256(meta.encode()).hexdigest(),'payload',hashlib.sha256(boot.encode()).hexdigest(),'body',tag,'bytes',len(boot))


if __name__=='__main__':main()
