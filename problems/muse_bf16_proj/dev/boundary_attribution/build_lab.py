"""Batch exact 6cb confirmation and S14 boundary attribution in one import-time lab."""
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path
from patch import patch

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def main():
    confirmation=(HERE.parent/'pdl_audit/lab_pdl_shared_followup.py').read_text()
    assert hashlib.sha256(confirmation.encode()).hexdigest()=='6cb428bc12b1180f6072a064ff9444dc7616ec2bdd3bc171ef3f13e26e3c99f8'
    base=(REPO/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
    variants={n:patch(base,m) for n,m in [('R','release'),('P','hint'),('RP','release_hint'),('C','geometry'),('S','serial')]}
    for n,s in variants.items():(HERE/({'R':'release','P':'hint','RP':'release_hint','C':'geometry','S':'serial'}[n]+'_trial.py')).write_text(s)
    activity=(REPO/'problems/muse_decode_layer/dev/timeline/activity.cpp').read_text()+'\n'+(HERE/'dag.cpp.inc').read_text()
    body='import ast,base64,gc,hashlib,json,linecache,lzma,math,statistics,struct,time,types\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
    body+='BD_CONFIRM_SOURCE='+repr(confirmation)+'\nBD_SOURCES='+repr(variants)+'\nBD_ACTIVITY='+repr(activity)+'\n'
    body+="exec(compile(BD_CONFIRM_SOURCE,'<exact-6cb-bootstrap>','exec'),globals())\nBD_CONFIRM_RUN=lab\nBD_NAMESPACE_ORIGINAL=pa_namespace\n"
    body+=(HERE/'lab_body.inc').read_text();compile(body,'boundary_lab','exec')
    (HERE/'boundary_generated.py.inc').write_text(body);tag=hashlib.sha256(body.encode()).hexdigest()
    encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    boot='import base64,lzma,linecache,hashlib\n_BD_BODY=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
    boot+='assert hashlib.sha256(_BD_BODY.encode()).hexdigest()=='+repr(tag)+'\n_BD_FILE="<boundary-'+tag[:12]+'>"\n'
    boot+='linecache.cache[_BD_FILE]=(len(_BD_BODY),None,_BD_BODY.splitlines(True),_BD_FILE)\nexec(compile(_BD_BODY,_BD_FILE,"exec"),globals())\n_BD_RUN=lab\ndef lab(deadline,weights,state=None):\n    return _BD_RUN(deadline,weights,state)\n'
    (HERE/'lab_boundary_batch.py').write_text(boot)
    (HERE/'build_manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),confirmation_sha256=hashlib.sha256(confirmation.encode()).hexdigest(),payload_sha256=hashlib.sha256(boot.encode()).hexdigest(),body_sha256=tag,variants={n:hashlib.sha256(s.encode()).hexdigest() for n,s in variants.items()}),indent=2)+'\n')
    print('body',tag,'payload',hashlib.sha256(boot.encode()).hexdigest(),'bytes',len(boot))
    for n,s in variants.items():print(n,hashlib.sha256(s.encode()).hexdigest())


if __name__=='__main__':main()
