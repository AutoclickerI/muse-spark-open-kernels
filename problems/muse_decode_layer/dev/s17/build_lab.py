"""Build S17's exact candidate, leave-outs, ILP alternative and disjoint paired lab."""
import ast,base64,difflib,hashlib,json,lzma,sys
from pathlib import Path
import compose
H=Path(__file__).resolve().parent;R=H.parents[3];S16=H.parent/'s16'
sys.path[:0]=[str(S16),str(R/'tools')]
from pack import pack
import h200_lab
sha=compose.digest


def main():
    late='--late' in sys.argv
    if late:
        import late_compose
        create=late_compose.build;choices=late_compose.DEFAULT;stem='s17_late'
    else:create=compose.compose;choices=compose.DEFAULT;stem='s17'
    source,meta=create();(H/'generated'/(stem+'.py')).write_text(source)
    packed=pack(source);shipped=H.parents[1]/('cand_r3_best7_s17_'+meta['sha256'][:12]+'.py');shipped.write_text(packed)
    variants={}
    for name in choices:
        text,_=create([n for n in choices if n!=name]);variants['no_'+name]=text
    if late:
        prepare_constraints=late_compose.prepare_constraints
        variants['S17_first']=prepare_constraints.apply(compose.compose()[0])
    else:variants['ILP_alternative']=compose.compose(ilp=True)[0]
    variants['INC']=prepare_constraints.apply(compose.frozen()) if late else compose.frozen()
    if late:
        current=capture_current(json.loads((H/'current_incumbent.json').read_text())['packed_source'])
        assert sha(current)=='ee5fe24973f440e5c0daf34c9a65ca4848fc03dd661bd08b6d92b40280cab745'
        variants['CURRENT']=current
    lines=source.splitlines(True);arms={}
    for name,text in variants.items():
        other=text.splitlines(True)
        edits=[(a,b,''.join(other[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,lines,other,autojunk=False).get_opcodes() if op!='equal']
        restored=list(lines)
        for a,b,replacement in reversed(edits):restored[a:b]=[replacement]
        assert ''.join(restored)==text
        arms[name]=dict(sha256=sha(text),edits=edits,rows=[])
    # Reuse only generic graph/cold-time helpers from the already-verified lab.
    import capture
    helper=capture.body((S16.parent/'s15/s15ab_payload.py').read_text())
    wanted={'ab_label','ab_namespace','ab_copy','ab_graph','ab_tensors','ab_gate','ab_time','AB_CASES','AB_LOADS'}
    nodes=[]
    for n in ast.parse(helper).body:
        name=n.name if isinstance(n,ast.FunctionDef) else getattr(n.targets[0],'id','') if isinstance(n,ast.Assign) else ''
        if name in wanted:nodes.append(ast.get_source_segment(helper,n))
    problem=R/'problems/muse_decode_layer'
    reference='\n'.join(line for line in (problem/'reference.py').read_text().splitlines() if not line.startswith(('from checks import','from nvfp4 import')))
    checker='\n'.join(line for text in ((problem/'checks.py').read_text(),(problem/'nvfp4.py').read_text(),reference) for line in text.splitlines() if not line.startswith('from __future__ import'))
    seed_manifest=json.loads((H/'seed_manifest.json').read_text())['late' if late else 'first']
    selection_seeds=seed_manifest['selection_seeds']
    origin=304910273 if late else 204910273
    seeds=[origin+104729*i for i in range(16)];assert not set(seeds)&set(selection_seeds)
    assert seeds==seed_manifest['confirmation_seeds']
    bundle=base64.b85encode(lzma.compress(json.dumps(dict(host=source,arms=arms),separators=(',',':')).encode(),preset=9)).decode()
    body='import torch,types,time,statistics,math\n'+'\n\n'.join(nodes)+'\nAB_SEEDS='+repr(tuple(seeds))+'\n'
    body+=f'S17_WEIGHT_SEED={origin-2}\nS17_ROUTER_SEED={origin-4}\n'
    body+='import base64,lzma,json\nS16_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\nS16_CHECKER='+repr(checker)+'\n'
    template=(H/'lab.py.inc').read_text()
    # Compact the same retained 1e-8-precision seed samples as fixed-point int32.
    # This changes only serialization; no arm, seed, checker or replay is omitted.
    a=template.index('    def compact(value):')
    template=template[:a]+'''    import struct
    names=list(next(iter(loads.values()))['samples'][0]);labels=list(loads)
    values=[round(row[name]*100000000)-100000000 for label in labels for row in loads[label]['samples'] for name in names]
    result['sample_arms']=names;result['sample_loads']=labels;result['sample_scale']=100000000
    result['sample_i32_b85']=base64.b85encode(struct.pack('<'+'i'*len(values),*values)).decode()
    for cell in result['loads'].values():cell.pop('samples')
    def compact(value):
        if isinstance(value,float):return round(value,8)
        if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
        if isinstance(value,list):return [compact(v) for v in value]
        return value
    blob=base64.b85encode(lzma.compress(json.dumps(compact(result),separators=(',',':')).encode())).decode()
    assert len(blob)<8500,len(blob)
    return ['META S17 frozen S16 comparison;16 held-out input seeds+fresh weights/router;all arms TASK;46/16 weighted six-load R;no per-load veto',
            'S17_XZ85 '+blob,'END complete='+str(result['complete'])]
'''
    import importlib.util
    spec=importlib.util.spec_from_file_location('s17_nearby_cases',H.parent/'nearby/cases.py')
    cases=importlib.util.module_from_spec(spec);spec.loader.exec_module(cases)
    nearby_arms=('A','INC','CURRENT') if late else ('A','INC')
    body+='S17_NEARBY_CASES='+repr(cases.cases())+'\nS17_NEARBY_ARMS='+repr(nearby_arms)+'\n'
    nearby=(H/'nearby_lab.py.inc').read_text()
    template=template.replace('    times={};shapes={}',nearby+'\n    times={};shapes={}')
    template=template.replace('task_checks=task_checks,agreement_diagnostics=agreements,','task_checks=task_checks,nearby_task_checks=nearby_task_checks,nearby_expected_checks=nearby_expected_checks,agreement_diagnostics=agreements,')
    template=template.replace("result['complete']=not failures", "result['complete']=nearby_task_checks==nearby_expected_checks and not failures")
    body+=template
    encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    payload='import base64,lzma,linecache,hashlib\n_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
    payload+='assert hashlib.sha256(_body.encode()).hexdigest()=='+repr(sha(body))+'\n_f="<s17-lab-'+sha(body)[:12]+'>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=lab\ndef lab(deadline,weights,state):return _run(deadline,weights,state)\n'
    pp=H/'generated/payload.py';pp.write_text(payload)
    host=packed+'''
_s16_original_origin=globals()['_s16_origin_prepare']
def _s16_clone_state(value):
    if isinstance(value,dict):return {k:_s16_clone_state(v) for k,v in value.items()}
    if isinstance(value,list):return [_s16_clone_state(v) for v in value]
    if isinstance(value,tuple):return tuple(_s16_clone_state(v) for v in value)
    return value
def _s16_origin_prepare(weights):
    global _S16_LAB_BASE_STATE
    state=_s16_original_origin(weights);_S16_LAB_BASE_STATE=_s16_clone_state(state);return state
_candidate_prepare=globals()['prepare']
def prepare(weights):return _candidate_prepare(weights)
'''
    hp=H/'generated/host.py';hp.write_text(host)
    wrapped,_=h200_lab.build([('s17_pair',sha(payload)[:16],payload)],'muse_decode_layer',phase='import',import_budget=620,base_file=hp)
    assert len(wrapped.encode())<1_000_000
    wp=H.parents[1]/'lab'/('lab_s17_'+meta['sha256'][:12]+'.py');wp.write_text(wrapped)
    meta.update(candidate=str((H/'generated'/(stem+'.py')).relative_to(R)),shipped_source=str(shipped.relative_to(R)),shipped_sha256=sha(packed),baseline_arm='CURRENT' if late else 'INC',nearby_expected_checks=len(cases.cases())*2*len(nearby_arms),
        wrapper=str(wp.relative_to(R)),wrapper_sha256=sha(wrapped),payload_sha256=sha(payload),checker_sha256=sha(checker),
        selection_seeds=selection_seeds,confirmation_seeds=seeds,confirmation_weight_seed=origin-2,confirmation_router_seed=origin-4,arms=arms)
    (H/'generated'/('lab_late_manifest.json' if late else 'lab_manifest.json')).write_text(json.dumps(meta,indent=2)+'\n');print(meta['sha256'],meta['wrapper_sha256'],'bytes',len(wrapped))
    if late:(H/'late_selection.json').write_text(json.dumps({k:v for k,v in meta.items() if k not in ('arms','selection_seeds','confirmation_seeds')},indent=2)+'\n')


def capture_current(source):
    n=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
    return lzma.decompress(base64.b85decode(ast.literal_eval(n.args[0]))).decode()
if __name__=='__main__':main()
