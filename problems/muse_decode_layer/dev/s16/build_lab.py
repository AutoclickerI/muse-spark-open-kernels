"""Build the selected second-round candidate, off arms, packed host and lab."""
import argparse,ast,base64,difflib,hashlib,json,lzma,sys
from pathlib import Path
import patches

H=Path(__file__).resolve().parent;R=H.parents[3]


def sha(s):return hashlib.sha256(s.encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,default=H/'generated/selection.json');a=ap.parse_args()
    selected=json.loads(a.selection.read_text())['patches'];source,meta=patches.compose(selected)
    candidate=H/'generated/s16.py';candidate.write_text(source)
    hostlines=source.splitlines(True);arms={}
    for spec in selected:
        text,_=patches.compose([s for s in selected if s['name']!=spec['name']])
        arms[spec['name']]=dict(source=text,rows=spec['rows'])
    arms['S15']=dict(source=patches.BASE.read_text(),rows=sorted({tuple(r) for s in selected for r in s['rows']}))
    for name,spec in arms.items():
        text=spec.pop('source');lines=text.splitlines(True)
        edits=[(a,b,''.join(lines[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,hostlines,lines,autojunk=False).get_opcodes() if op!='equal']
        restored=list(hostlines)
        for lo,hi,replacement in reversed(edits):restored[lo:hi]=[replacement]
        assert ''.join(restored)==text
        spec.update(sha256=sha(text),edits=edits)
    # Reuse the immutable helpers actually queued in S15; cleanup cannot erase them.
    s15=(H.parent/'s15/s15ab_payload.py').read_text()
    n=next(n for n in ast.walk(ast.parse(s15)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
    helper=lzma.decompress(base64.b85decode(ast.literal_eval(n.args[0]))).decode();nodes=[]
    wanted={'ab_label','ab_namespace','ab_copy','ab_graph','ab_tensors','ab_gate','ab_time','AB_CASES','AB_LOADS'}
    for n in ast.parse(helper).body:
        name=n.name if isinstance(n,ast.FunctionDef) else getattr(n.targets[0],'id','') if isinstance(n,ast.Assign) else ''
        if name in wanted:nodes.append(ast.get_source_segment(helper,n))
    bundle=base64.b85encode(lzma.compress(json.dumps(dict(host=source,arms=arms),separators=(',',':')).encode(),preset=9)).decode()
    problem=R/'problems/muse_decode_layer'
    reference='\n'.join(line for line in (problem/'reference.py').read_text().splitlines()
                        if not line.startswith(('from checks import','from nvfp4 import')))
    checker='\n'.join(line for text in ((problem/'checks.py').read_text(),(problem/'nvfp4.py').read_text(),reference)
                      for line in text.splitlines() if not line.startswith('from __future__ import'))
    selection_seeds=sorted(set([172903+7919*i for i in range(16)]+[173101+7919*i for i in range(8)]+[3102917+104729*i for i in range(16)]+[3121907+7919*i for i in range(16)]+[197731]+[6107923+7919*i for i in range(8)]+[5353+7919*i for i in range(16)]+[9000+m+97*i for m in (1,8,16,32) for i in range(16)]))
    confirmation_seeds=[104910273+104729*i for i in range(16)]
    assert not set(selection_seeds)&set(confirmation_seeds)
    body='import torch,types,time,statistics,math\n'+'\n\n'.join(nodes)+'\nAB_SEEDS='+repr(tuple(confirmation_seeds))+'\n'
    body+='import base64,lzma,json\nS16_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\n'
    body+='S16_CHECKER='+repr(checker)+'\n'
    body+=(H/'lab.py.inc').read_text()
    packed=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    payload='import base64,lzma,linecache,hashlib\n_body=lzma.decompress(base64.b85decode('+repr(packed)+')).decode()\n'
    payload+='assert hashlib.sha256(_body.encode()).hexdigest()=='+repr(sha(body))+'\n_f="<s16-lab-'+sha(body)[:12]+'>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=lab\ndef lab(deadline,weights,state):\n    return _run(deadline,weights,state)\n'
    pp=H/'generated/s16_payload.py';pp.write_text(payload)
    license=(H.parent/'s15/proj/LICENSE.NathenFernandes').read_text()
    notice='\n'.join('# '+line for line in license.splitlines())
    src=base64.b85encode(lzma.compress(source.encode(),preset=9)).decode()
    outer='# okc: problem=muse_decode_layer\n'+notice+'\nimport torch,base64,lzma,linecache,hashlib\n_source=lzma.decompress(base64.b85decode('+repr(src)+')).decode()\n'
    outer+='assert hashlib.sha256(_source.encode()).hexdigest()=='+repr(sha(source))+'\n_f="<s16-expanded-'+sha(source)[:12]+'>"\nlinecache.cache[_f]=(len(_source),None,_source.splitlines(True),_f)\nexec(compile(_source,_f,"exec"),globals())\n'
    outer+='''
_s16_original_origin=globals()['_s16_origin_prepare']
def _s16_clone_state(value):
    if isinstance(value,dict):return {k:_s16_clone_state(v) for k,v in value.items()}
    if isinstance(value,list):return [_s16_clone_state(v) for v in value]
    if isinstance(value,tuple):return tuple(_s16_clone_state(v) for v in value)
    return value
def _s16_origin_prepare(weights):
    global _S16_LAB_BASE_STATE
    state=_s16_original_origin(weights)
    _S16_LAB_BASE_STATE=_s16_clone_state(state)
    return state
_candidate_prepare=globals()['prepare']
def prepare(weights):return _candidate_prepare(weights)
'''
    outer+='\n_s16_entry_kernel=globals()["kernel_fn"]\ndef kernel_fn(data,state):\n    return _s16_entry_kernel(data,state)\n'
    host=H/'generated/s16_host.py';host.write_text(outer)
    sys.path.insert(0,str(R/'tools'));import h200_lab
    wrapped,_=h200_lab.build([(pp.stem,sha(payload)[:16],payload)],'muse_decode_layer',base_file=host,phase='import',import_budget=620)
    assert len(wrapped.encode())<1_000_000,len(wrapped.encode())
    target=R/'problems/muse_decode_layer/lab'/('lab_s16_'+meta['sha256'][:12]+'.py');target.write_text(wrapped)
    meta.update(candidate=str(candidate.relative_to(R)),wrapper=str(target.relative_to(R)),wrapper_sha256=sha(wrapped),
                payload=str(pp.relative_to(R)),payload_sha256=sha(payload),host_file=str(host.relative_to(R)),arms=arms,checker_sha256=sha(checker),selection_seeds=selection_seeds,confirmation_seeds=confirmation_seeds,confirmation_weight_seed=104910271,confirmation_router_seed=104910269)
    (H/'generated/lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('S16 candidate',meta['sha256'],'wrapper',meta['wrapper_sha256'],'bytes',len(wrapped))


if __name__=='__main__':main()
