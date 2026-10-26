"""Self-contained S19 finish-deletion lab. Oracle only, never shipped as a kernel."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
import kernels
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(H.parent/'sliding_oracle'));import build_followup as old

def main():
    (H/'out').mkdir(exist_ok=True)
    src=(R/'prof/lab/last_sliding_timeline.py').read_text();header=['import gc,hashlib,json,linecache,math,re,statistics,time,types','import torch','from torch.utils.cpp_extension import load_inline']
    for n in ast.parse(src).body:
        if isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('TL_ACTIVITY','TL_REPS') for t in n.targets):header.append(ast.get_source_segment(src,n))
        elif isinstance(n,ast.FunctionDef) and n.name=='tl_activity':header.append(ast.get_source_segment(src,n))
    spec='import math,functools\nimport torch\n'+'\n'.join(old.definitions(R/'problems/muse_decode_layer'/n) for n in ('nvfp4.py','checks.py','reference.py'))
    src=(H.parent/'early_history/lab_body.inc').read_text();checker=next(ast.get_source_segment(src,n) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='eh_checker')
    timing=(R/'problems/muse_decode_layer/dev/s21/timing.py.inc').read_text();helpers='S21_NVML=None\nS21_TELEMETRY_ERROR=None\n'+'\n'.join(ast.get_source_segment(timing,n) for n in ast.parse(timing).body if isinstance(n,ast.FunctionDef) and n.name in ('s21_telemetry','s21_orders'))
    sources=kernels.sources();hashes={k:{n:hashlib.sha256(v.encode()).hexdigest() for n,v in s.items()} if isinstance(s,dict) else hashlib.sha256(s.encode()).hexdigest() for k,s in sources.items()}
    body='\n'.join(header)+'\nPUBLIC_SPEC={"torch":torch}\nexec('+repr(spec)+',PUBLIC_SPEC)\n'+checker+'\nBASE='+repr(kernels.base())+'\nCF_SOURCES='+repr(sources)+'\nCF_HASHES='+repr(hashes)+'\n'+helpers+'\n'+(H/'sample_codec.py').read_text()+'\n'+(H/'runtime.py.inc').read_text()
    compile(body,'P5_finish_deletion','exec');(H/'out/body.py').write_text(body)
    blob=base64.b85encode(lzma.compress(body.encode())).decode()
    payload='LAB_RESOURCES={"compile_s":380,"run_s":430,"output_bytes":11744,"context_poison":"none"}\nimport base64,lzma,linecache\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n_FN="<P5_finish_deletion>"\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n_CF=globals()["lab"]\ndef lab(deadline,weights,state=None):return _CF(deadline,weights,state)\n'
    p=H/'out/deletion.py';p.write_text(payload)
    manifest=dict(base_sha256=hashlib.sha256(kernels.base().encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),body_sha256=hashlib.sha256(body.encode()).hexdigest(),sources=hashes,scope='oracle only; not a correct candidate',weight_seed=74000033,router_seed=74000039,seeds=[85000121+104729*i for i in range(8)],measurement_protocol=dict(name='palindromic-cohort-v1',rounds=6,seed_count=8,aggregation='arithmetic-mean-all-rounds',selective_retiming=False),per_path='DEL_S/DEL_F synthesized from the measured DEL path and the unchanged BASE path, not separately timed duplicate calls.')
    (H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(p,manifest['payload_sha256'])
if __name__=='__main__':main()
