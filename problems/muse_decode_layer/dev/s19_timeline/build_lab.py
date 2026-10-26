"""Re-use S15's CUPTI/DAG observer on pinned S19; one self-preparing import lab."""
import ast, base64, hashlib, json, lzma, sys
from pathlib import Path
H = Path(__file__).resolve().parent
R = H.parents[3]
OLD = H.parent / 's15_timeline'
sys.path.insert(0, str(R / 'tools'))
from h200_lab import build

def sha(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()

def functions(path, names):
    s = path.read_text()
    return '\n\n'.join(ast.get_source_segment(s, n) for n in ast.parse(s).body
                        if isinstance(n, ast.FunctionDef) and n.name in names)

def main():
    frozen = json.loads((H.parent/'s19/frozen.json').read_text())
    packed = (R/frozen['file']).read_text()
    tree = ast.parse(packed)
    assignment = next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_source' for t in n.targets))
    blob = next(n.value for n in ast.walk(assignment.value) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000)
    source = lzma.decompress(base64.b85decode(blob)).decode()
    assert sha(packed) == frozen['shipped_sha256']
    assert sha(source) == frozen['expanded_sha256']
    # Re-use the existing builder's activity/DAG extension statements directly.
    old_builder=(OLD/'build_lab.py').read_text()
    main_node=next(n for n in ast.parse(old_builder).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    first=next(i for i,n in enumerate(main_node.body) if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='activity')
    last=next(i for i,n in enumerate(main_node.body) if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='helpers')
    observer_ns={'HERE':OLD,'ROOT':R}
    exec(compile(ast.Module(body=main_node.body[first:last],type_ignores=[]),'<s15-observer-builder>','exec'),observer_ns)
    activity=observer_ns['activity']
    checker = '\n'.join(line for p in ('checks.py','nvfp4.py','reference.py')
                        for line in (R/'problems/muse_decode_layer'/p).read_text().splitlines()
                        if not line.startswith(('from __future__ import','from checks import','from nvfp4 import')))
    analysis = (OLD/'analysis.py').read_text().replace("if role!='moe':return role", "if role!='moe':return role\n    if 'chunk::kernel<' in name:return 'moe_heavy'")
    helpers = functions(H.parent/'s19/graph_helpers.py.inc', ('ab_copy','ab_tensors','ab_gate','ab_time'))
    reuse = functions(OLD/'lab_body.inc', ('t15_activity','t15_namespace','t15_capture','t15_signature','t15_busiest','t15_pack'))
    # Keep full representative kernel names and a detailed phase role in metadata, and per-round controls.
    reuse = reuse.replace("n['role']=stage_name(n['role'],name);n['name']=names.index(name)", "n['phase']=phase_name(n['role'],name);n['role']=stage_name(n['role'],name);n['name']=names.index(name)")
    reuse = reuse.replace("names=[n.replace('void ','').replace('(anonymous namespace)::','').split('(')[0][:16]+'@'+str(i) for i,n in enumerate(names)]", "names=[n.replace('void ','').replace('(anonymous namespace)::','').split('(')[0][:16]+'@'+str(i) for i,n in enumerate(names)]")
    reuse = reuse.replace("seed['joins_count']=len(account['joins'])", "seed['joins_count']=0").replace("for join in account['joins']:", "for join in ():")
    reuse = reuse.replace('    result.update(version=1,', "    result['round_values_offset']=len(binary);binary.extend(result.pop('round_values'))\n    result.update(version=1,")
    reuse = reuse.replace("stages=sorted({stage for cell in result['cases'] for seed in cell['seeds'] for stage in seed['account']['critical_us']})", 'stages=[]')
    reuse = reuse.replace("seed['critical_offset']=len(binary)", "seed['wall_ns']=wall;seed['critical_offset']=len(binary)")
    reuse = reuse.replace('result.update(version=1,', 'result.update(version=2,')
    reuse = reuse.replace("for key in ('profile_wall_samples_us',):", "for key in ():")
    reuse = reuse.replace("seed.pop('work');seed.pop('max_accounting_error')", "seed.pop('profile_wall_samples_us');seed.pop('work');seed.pop('max_accounting_error')")
    # All-round numeric data is retained in a second compact varint stream rather than rounded JSON.
    body = 'import base64,gc,hashlib,json,linecache,lzma,math,re,statistics,struct,time,types\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
    body += 'T15_ACTIVITY='+repr(activity)+'\nS19_SOURCE='+repr(source)+'\nS19_CHECKER='+repr(checker)+'\n'
    body += analysis+'\n'+helpers+'\n'+reuse+'\n'+(OLD/'footprints.py.inc').read_text()+'\n'+(H/'runtime.py.inc').read_text()
    compile(body, '<s19-timeline>', 'exec')
    (H/'generated').mkdir(exist_ok=True)
    (H/'generated/body.py').write_text(body)
    encoded = base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    payload = 'LAB_RESOURCES={"compile_s":200,"run_s":430,"output_bytes":11744,"context_poison":"none"}\nimport base64,lzma,linecache,hashlib\n'
    payload += '_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\nassert hashlib.sha256(_body.encode()).hexdigest()=='+repr(sha(body))+'\n'
    payload += '_f="<s19-timeline-'+sha(body)[:12]+'>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=globals()["lab"]\ndef lab(deadline,weights,state=None):return _run(deadline,weights,state)\n'
    p=H/'generated/s19_timeline.py';p.write_text(payload)
    wrapper,_=build([(p.stem,sha(payload)[:16],payload)],'muse_prefill_layer',phase='import',import_budget=780,lab_only=True)
    (H/'generated/wrapper.py').write_text(wrapper)
    meta=dict(source_sha256=sha(source),shipped_sha256=sha(packed),observer_sha256=sha(activity),checker_sha256=sha(checker),payload_sha256=sha(payload),wrapper_sha256=sha(wrapper),seeds=[927410331+104729*i for i in range(3)],weight_seed=927410327,router_seed=927410329,rounds=3,shapes=12)
    (H/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))
if __name__=='__main__': main()
