"""One five-arm production-graph cohort per P13 mechanism step."""
import argparse
import ast
import base64
import json
import lzma
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path[:0]=[str(H),str(R/'tools')]
import mechanisms as build
from local_eval import tests_of

def definitions(path):
    s=path.read_text()
    return '\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=('boundary','route'));a=ap.parse_args()
    sys.argv=['build.py']+(['--route'] if a.kind=='route' else []);build.main()
    meta=json.loads((H/(a.kind+'_manifest.json')).read_text());base,_=build.base()
    spec='import math,functools\nimport torch\n'+'\n'.join(definitions(R/'problems/muse_decode_layer'/name) for name in ('nvfp4.py','checks.py','reference.py'))
    old=(H.parent/'oracle.py.inc').read_text();tree=ast.parse(old)
    names={'p13_fixture','p13_host','p13_namespace','p13_copy','p13_graph','p13_time'}
    helpers='\n'.join(ast.get_source_segment(old,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
    text=(R/'problems/muse_attention_decode/dev/early_history/lab_body.inc').read_text()
    checker=next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='eh_checker')
    text=(R/'problems/muse_decode_layer/dev/s21/timing.py.inc').read_text()
    timing='\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in ('s21_telemetry','s21_orders'))
    suffixes={n:(H/'out'/(a.kind+'_'+n+'.inc')).read_text() for n in ('A','B')}
    identities=dict(INC=build.sha(base),A=meta['arms']['A']['sha256'],A2=meta['arms']['A']['sha256'],B=meta['arms']['B']['sha256'],B2=meta['arms']['B']['sha256'])
    nearby=tests_of((R/'problems/muse_decode_layer/TASK.md').read_text())
    nearby += [dict(batch=b,q_len=q,ctx=c,layer=l,dist='soft',page=1) for b,q,c,l in ((2,1,173,'sliding'),(7,1,231,'full'),(2,8,307,'sliding'),(16,8,509,'full'),(32,8,523,'sliding'))]
    prefix=180003000 if a.kind=='boundary' else 190003000
    seeds=[prefix+21+104729*i for i in range(8)]
    protocol=dict(name='palindromic-cohort-v1',rounds=10,seed_count=8,aggregation='arithmetic-mean-all-rounds',selective_retiming=False)
    body='import base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types\nimport torch\n'
    body+='BASE='+repr(base)+'\nPUBLIC_SPEC={"torch":torch}\nexec('+repr(spec)+',PUBLIC_SPEC)\n'+checker+'\nS21_NVML=None\nS21_TELEMETRY_ERROR=None\n'+timing+'\n'
    body+=(H.parent/'codec.py').read_text()+'\n'+helpers+'\n'
    constants=dict(KIND=a.kind,SUFFIXES=suffixes,IDENTITIES=identities,NEARBY_CASES=nearby,SEEDS=tuple(seeds),WEIGHT_SEED=prefix+3,ROUTER_SEED=prefix+9,PROTOCOL=protocol,PREPARE='p13b_prepare' if a.kind=='boundary' else 'p13r_prepare',WARM='_p13b_warm' if a.kind=='boundary' else '_p13r_warm')
    body+='\n'.join(k+'='+repr(v) for k,v in constants.items())+'\n'+(H/'measure.py.inc').read_text()
    compile(body,'<P13-'+a.kind+'-pair>','exec');(H/'out'/(a.kind+'_body.py')).write_text(body)
    blob=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    payload='LAB_RESOURCES={"compile_s":230,"run_s":520,"output_bytes":13744,"context_poison":"none"}\nimport base64,lzma,linecache\n'
    payload+='_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n_FN="<P13-'+a.kind+'-pair>"\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n_PAIR=globals()["lab"]\ndef lab(deadline,weights,state=None):return _PAIR(deadline,weights,state)\n'
    (H/'out'/('p13_'+a.kind+'.py')).write_text(payload)
    manifest=dict(kind=a.kind,base_sha256=build.sha(base),base_series=meta['base_series'],base_shipped_sha256=meta['base_shipped_sha256'],identities=identities,arms=meta['arms'],
                  confirmation_seeds=seeds,selection_seeds=[160001021+104729*i for i in range(8)]+[170001021+104729*i for i in range(8)],
                  weight_seed=prefix+3,router_seed=prefix+9,measurement_protocol=protocol,checker_sha256=build.sha(spec),
                  nearby_cases=nearby,expected_checks=480,nearby_checks=54,payload_sha256=build.sha(payload),body_sha256=build.sha(body),
                  fixture='Both soft, exact public reference busiest shard each seed; all local_start, rounds, telemetry retained; no off-timed candidate work.')
    (H/(a.kind+'_lab_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print('PAYLOAD',a.kind,manifest['payload_sha256'])

if __name__=='__main__':main()
