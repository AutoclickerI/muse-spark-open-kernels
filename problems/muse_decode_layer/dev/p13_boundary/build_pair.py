"""One native paired lab with prespecified combined/small/large fusion arms."""
import ast
import base64
import json
import lzma
import sys
from pathlib import Path
H=Path(__file__).resolve().parent
R=H.parents[3]
sys.path.insert(0,str(H))
import p13_build
import build_fusion
sha=p13_build.sha


def main():
    build_fusion.main();source=(H/'out/candidate.py').read_text();base=p13_build.base();assert source.startswith(base)
    suffix=source[len(base):]
    suffixes=dict(A=suffix,SMALL=suffix.replace('P13_LARGE_ROWS=(128,256)','P13_LARGE_ROWS=()'),LARGE=suffix.replace('P13_ROWS=(1,8,16)','P13_ROWS=()'))
    identities={'INC':sha(base),**{n:sha(base+s) for n,s in suffixes.items()},'A2':sha(source)}
    # Reuse the exact oracle public TASK/checker and timing helpers, not its
    # altered-work ready buffers or deletion-arm functions.
    old=(H/'out/body.py').read_text();tree=ast.parse(old);header=old[:old.index('# Altered-work diagnostic.')]
    names={'p13_fixture','p13_host','p13_namespace','p13_copy','p13_graph','p13_time'}
    helpers='\n'.join(ast.get_source_segment(old,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
    sys.path.insert(0,str(R/'tools'))
    from local_eval import tests_of
    nearby=tests_of((R/'problems/muse_decode_layer/TASK.md').read_text())
    nearby+=[dict(batch=b,q_len=q,ctx=ctx,layer=layer,dist='soft',page=1) for b,q,ctx,layer in ((2,1,173,'sliding'),(7,1,231,'full'),(2,8,307,'sliding'),(16,8,509,'full'),(32,8,523,'sliding'))]
    body=header+'\n'+helpers+'\nSUFFIXES='+repr(suffixes)+'\nIDENTITIES='+repr(identities)+'\nNEARBY_CASES='+repr(nearby)+'\n'+(H/'fusion_lab.py.inc').read_text()
    compile(body,'<P13-pair>','exec');(H/'out/pair_body.py').write_text(body)
    blob=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    payload='LAB_RESOURCES={"compile_s":230,"run_s":520,"output_bytes":13744,"context_poison":"none"}\nimport base64,lzma,linecache\n'
    payload+='_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n_FN="<P13-fusion-pair>"\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n_PAIR=globals()["lab"]\ndef lab(deadline,weights,state=None):return _PAIR(deadline,weights,state)\n'
    (H/'out/p13_pair.py').write_text(payload)
    manifest=dict(sha256=identities['A'],base_sha256=identities['INC'],baseline_arm='INC',checker_sha256=json.loads((H/'manifest.json').read_text())['checker_sha256'],
                  arms={n:dict(sha256=identities[n],rows=[]) for n in ('INC','SMALL','LARGE')},
                  identities=identities,confirmation_seeds=[170001021+104729*i for i in range(8)],selection_seeds=json.loads((H/'manifest.json').read_text())['seeds'],
                  nearby_arms=['A','INC','SMALL','LARGE'],nearby_expected_checks=len(nearby)*4,
                  expected_TASK_checks=8*12*5,measurement_protocol=dict(name='palindromic-cohort-v1',rounds=10,seed_count=8,aggregation='arithmetic-mean-all-rounds',selective_retiming=False),
                  payload_sha256=sha(payload),body_sha256=sha(body),nearby_cases=nearby,oracle=json.loads((H/'result.json').read_text())['job'])
    (H/'pair_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('nearby_cases','arms')},indent=2))


if __name__=='__main__':main()
