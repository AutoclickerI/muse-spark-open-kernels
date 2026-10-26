"""P4 ec754e5 protocol, exact S22 control, outcome-specific attention candidates."""
import argparse,ast,base64,hashlib,importlib.util,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4];O=H/'out';P4=R/'problems/muse_decode_layer/dev';sys.path.insert(0,str(R/'tools'))
import h200_lab
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
b=json.loads((O/'bundle.json').read_text());meta=json.loads((H/'manifest.json').read_text());lines=b['base'].splitlines(True)
ap=argparse.ArgumentParser();ap.add_argument('--keep',type=Path);args=ap.parse_args()
kept=json.loads(args.keep.read_text()) if args.keep else list(meta['arms'])
assert kept and len(kept)==len(set(kept)) and set(kept)<=meta['arms'].keys()
primary='INC' if 'INC' in kept else next(n for n in ('STAGE','LANE1','SOFTMAX','CHAIN','OCHAIN') if n in kept)
names={('INC' if n==primary else n):n for n in kept}
arms={name:dict(sha256=meta['arms'][logical]['sha256'],edits=[(len(lines),len(lines),b['tails'][logical])],rows=[] if name=='INC' else meta['arms'][logical]['rows']) for name,logical in names.items()}
needed=set()
for n in kept:
 for node in ast.parse(b['tails'][n]).body:
  if isinstance(node,ast.Assign) and any(getattr(t,'id','')=='_NIGHT_SOURCES' for t in node.targets):needed.update(ast.literal_eval(node.value))
modules={n:b['modules'][n] for n in sorted(needed)}
checker='\n'.join(x for text in ((P4.parent/'checks.py').read_text(),(P4.parent/'nvfp4.py').read_text(),
 '\n'.join(l for l in (P4.parent/'reference.py').read_text().splitlines() if not l.startswith(('from checks import','from nvfp4 import'))))
 for x in text.splitlines() if not x.startswith('from __future__ import'))
hook=next(ast.literal_eval(n.value) for n in ast.parse((P4/'s19/build_lab.py').read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='HOOK')
prebuild='''
_NIGHT_LAB_MODULES={}
def _night_lab_build():
    import types,linecache,hashlib
    from concurrent.futures import ThreadPoolExecutor
    def one(name):
        s=_NIGHT_LAB_SOURCES[name];f='<p5b-night-prebuild-'+name+'>';linecache.cache[f]=(len(s),None,s.splitlines(True),f)
        m=types.ModuleType('p5b_night_'+name);m.__file__=f;exec(compile(s,f,'exec'),m.__dict__)
        if install:=globals().get('_NIGHT_LOCAL_MODULE'):install(name,m)
        if hasattr(m,'setup'):m.setup('cuda')
        _NIGHT_LAB_MODULES[name]=m
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,_NIGHT_LAB_SOURCES))
_S19_BUILDS.append(_night_lab_build)
'''
hook='\n_NIGHT_LAB_SOURCES='+repr(modules)+'\n'+prebuild+hook
sp=importlib.util.spec_from_file_location('p5b_night_nearby',P4/'nearby/cases.py');case=importlib.util.module_from_spec(sp);sp.loader.exec_module(case)
seeds=tuple(1081730011+104729*i for i in range(64))
# Reject accidental reuse of any archived selection/confirmation seed set.
used=set()
def collect(v,key=''):
 if isinstance(v,dict):
  for k,x in v.items():collect(x,k)
 elif isinstance(v,list):
  if key in ('seeds','seed_ids','confirmation_seeds','selection_seeds'):used.update(x for x in v if isinstance(x,int))
  else:
   for x in v:
    if isinstance(x,(dict,list)):collect(x)
for p in [P4/d/'seed_manifest.json' for d in ('s18','s19','s20','s21')]+[P4/'s22/manifest.json',H.parent/'lab_manifest.json']:
 collect(json.loads(p.read_text()))
assert not used.intersection(seeds)
body='import torch,types,time,statistics,math\n'+(P4/'s19/graph_helpers.py.inc').read_text()+'\n'+(P4/'s21/timing.py.inc').read_text()
blob=base64.b85encode(lzma.compress(json.dumps(dict(host=b['base'],arms=arms),separators=(',',':')).encode(),preset=9)).decode()
body+='\nimport base64,lzma,json\nS16_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(blob)+')))\n'
body+='S16_CHECKER='+repr(checker)+'\nAB_SEEDS='+repr(seeds)+'\nS17_WEIGHT_SEED=1081730007\nS17_ROUTER_SEED=1081730009\nS19_HOST_HOOK='+repr(hook)+'\nS17_NEARBY_CASES='+repr(case.cases())+"\nS17_NEARBY_ARMS=('A','INC')\n"
core=(P4/'s21/lab.py.inc').read_text();core=core.replace("            v,shared[name]=s16_namespace(name,source,base)","            v,shared[name]=s16_namespace(name,source,base)\n            v['_NIGHT_MODULES']=base['_NIGHT_LAB_MODULES']")
assert "v['_NIGHT_MODULES']" in core
core=core.replace("    result['original_pair_repeat']=s21_repeat_original_case(checker,base,deadline)\n",'')
core=core.replace("    result['measurement_protocol']=dict(S21_PROTOCOL)","    result['stage_resources']={n:{str((mode,x)):m._MOD.stage_resources(mode,x) for mode in (3,4,5) for x in (68,324)} for n,m in base['_NIGHT_LAB_MODULES'].items() if n=='lee_stage'}\n    result['measurement_protocol']=dict(S21_PROTOCOL)")
core=core.replace("'S21_XZ85 '+blob","'P5N_XZ85 '+blob").replace('META S21 on exact S19;64 disjoint held-out seeds;all-round symmetric cohorts;public TASK all arms;restricted leave-outs take A2 off their rows','META P5b overnight candidates vs exact S22;ec754e5 64freshseeds16symmetric rounds;public TASK every active arm;production graphs')
body+='\n'+(P4/'s21/evidence.py').read_text()+'\n'+core
enc=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
payload='LAB_RESOURCES={"compile_s":350,"run_s":490,"output_bytes":11700,"context_poison":"none"}\nimport base64,lzma,linecache,hashlib\n_body=lzma.decompress(base64.b85decode('+repr(enc)+')).decode()\nassert hashlib.sha256(_body.encode()).hexdigest()=='+repr(sha(body))+'\n_f="<p5b-night-'+sha(body)[:12]+'>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=globals()["lab"]\ndef lab(deadline,weights,state):return _run(deadline,weights,state)\n'
(O/'p5b_night.py').write_text(payload)
wrapper,_=h200_lab.build([('p5b_night',sha(payload)[:16],payload)],'muse_attention_decode',phase='import',import_budget=840.,lab_only=True);assert len(wrapper.encode())<1_000_000
(O/'wrapper.py').write_text(wrapper)
labels=['d1S','d1F','d16S','d16F','d32S','d32F','v1S','v1F','v16S','v16F','v32S','v32F']
expected=64*sum(2+sum(not arm['rows'] or label in arm['rows'] for arm in arms.values()) for label in labels)
meta.update(arms={n:{**meta['arms'][logical],**{k:v for k,v in arms[n].items() if k!='edits'},'logical':logical} for n,logical in names.items()},
 sha256=meta['base_sha256'],checker_sha256=sha(checker),payload_sha256=sha(payload),wrapper_sha256=sha(wrapper),confirmation_seeds=seeds,selection_seeds=sorted(used),weight_seed=1081730007,router_seed=1081730009,
 baseline_arm='A',primary_arm='INC',nearby_arms=['A','INC'],expected_TASK_checks=expected,nearby_expected_checks=260,measurement_protocol=dict(name='palindromic-cohort-v1',rounds=16,seed_count=64,aggregation='arithmetic-mean-all-rounds',selective_retiming=False))
(H/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(sha(payload)[:16],len(wrapper),expected)
