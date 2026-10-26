"""Freeze current P4 composition and batch port nominees plus coded pool mechanisms."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];O=H/'out'
basepath=R/'problems/muse_decode_layer/cand_r3_best7_s21_final.py';base=basepath.read_text();results=json.loads((H/'results.json').read_text())
points={}
for i,p in enumerate(results['nominees']):
    role='unpooler_slice' if p['role']=='unpooler' and p['t']<=8 else p['role']
    points['PORT'+str(i)]=[[p['t'],role,p['provider']]]
points['PORTS']=[p for pts in points.values() for p in pts]
pool=['CW8_WAIT','CW8_EARLY','CW8_S2','CW8_S2_WAIT','CW4_EARLY','CW4_S2','TW256_S2','TW256_S2_E','TW128_S3','TW128_S3_E']
source=base+'\n'+(O/'helper.py.inc').read_text()+'\n'+(H/'pool_tma.py.inc').read_text()+'\n'+(H/'pool.py.inc').read_text();compile(source,'<P10 S21 pool variants>','exec');(O/'whole_source.py').write_text(source)
prior=(O/'public_projection.py').read_text();tree=ast.parse(prior);constants={}
for n in tree.body:
    if isinstance(n,ast.Assign):
        for t in n.targets:
            if getattr(t,'id','') in ('PP_TASK','PP_CHECK'):constants[t.id]=ast.literal_eval(n.value)
functions='\n\n'.join(ast.get_source_segment(prior,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('pp_module','pp_copy','pp_capture','pp_time'))
functions+='\n'+next(ast.get_source_segment(prior,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pp_time').replace('def pp_time(','def pw_time(').replace('range(2)','range(4)')
seeds=[32000319+104729*i for i in range(8)]
payload='LAB_RESOURCES={"compile_s":420,"run_s":390,"output_bytes":13744,"context_poison":"none"}\nimport torch,triton,triton.language as tl,time,types,linecache,hashlib,json,statistics,math,lzma,base64,struct,gc\nPP_SOURCE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(source.encode())).decode())+')).decode()\n'+''.join(k+'='+repr(v)+'\n' for k,v in constants.items())+'PW_POINTS='+repr(points)+'\nPW_POOL='+repr(pool)+'\nPW_SEEDS='+repr(seeds)+'\n'+functions+'\n'+(H.parent/'native_boundary/codec.py.inc').read_text()+'\n'+(H/'whole_lab.py.inc').read_text();compile(payload,'<P10 whole S21 port batch>','exec');(O/'public_whole.py').write_text(payload)
(H/'whole_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),base_path=str(basepath.relative_to(R)),base_sha256=hashlib.sha256(base.encode()).hexdigest(),selection_job=results['job'],points=points,pool=pool,seeds=seeds,rounds=8,weight_seed=32000311,router_seed=32000313,protocol='palindromic-cohort-v1;arithmetic mean all rounds;all per-round times retained'),indent=2)+'\n');print(len(payload))
