"""One scored S18 host also supplies the entire held-out paired comparison."""
import ast,base64,difflib,hashlib,importlib.util,json,lzma,sys
from pathlib import Path
import compose
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1]
sys.path[:0]=[str(H.parent/'s16'),str(R/'tools')]
from pack import pack
import h200_lab
sha=compose.sha

def main():
 source,meta=compose.build();(H/'generated/s18.py').write_text(source)
 packed=pack(source);candidate=P/('cand_r3_best7_s18_'+meta['sha256'][:12]+'.py');candidate.write_text(packed)
 variants={}
 for name in compose.UNCERTAIN:variants['no_'+name]=compose.build([n for n in compose.DEFAULT if n!=name])[0]
 variants['INC']=compose.baseline();lines=source.splitlines(True);arms={}
 for name,text in variants.items():
  other=text.splitlines(True);edits=[(a,b,''.join(other[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,lines,other,autojunk=False).get_opcodes() if op!='equal'];restored=list(lines)
  for a,b,t in reversed(edits):restored[a:b]=[t]
  assert ''.join(restored)==text;arms[name]=dict(sha256=sha(text),edits=edits,rows=[])
 problem=R/'problems/muse_decode_layer'
 reference='\n'.join(line for line in (problem/'reference.py').read_text().splitlines() if not line.startswith(('from checks import','from nvfp4 import')))
 checker='\n'.join(line for text in ((problem/'checks.py').read_text(),(problem/'nvfp4.py').read_text(),reference) for line in text.splitlines() if not line.startswith('from __future__ import'))
 seeds=json.load(open(H/'seed_manifest.json'));assert not set(seeds['selection_seeds'])&set(seeds['confirmation_seeds'])
 bundle=base64.b85encode(lzma.compress(json.dumps(dict(host=source,arms=arms),separators=(',',':')).encode(),preset=9)).decode()
 body='import torch,types,time,statistics,math\n'+(H/'graph_helpers.py.inc').read_text()+'\nAB_SEEDS='+repr(tuple(seeds['confirmation_seeds']))+'\n'
 body+=f"S17_WEIGHT_SEED={seeds['weight_seed']}\nS17_ROUTER_SEED={seeds['router_seed']}\n"
 body+='import base64,lzma,json\nS16_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\nS16_CHECKER='+repr(checker)+'\n'
 sp=importlib.util.spec_from_file_location('s18_nearby_cases',H.parent/'nearby/cases.py');cases=importlib.util.module_from_spec(sp);sp.loader.exec_module(cases)
 body+='S17_NEARBY_CASES='+repr(cases.cases())+"\nS17_NEARBY_ARMS=('A','INC')\n"
 template=(H/'lab.py.inc').read_text();template=template.replace('    times={};shapes={}',(H/'nearby_lab.py.inc').read_text()+'\n    times={};shapes={}')
 template=template.replace('task_checks=task_checks,agreement_diagnostics=agreements,','task_checks=task_checks,nearby_task_checks=nearby_task_checks,nearby_expected_checks=nearby_expected_checks,agreement_diagnostics=agreements,')
 template=template.replace("result['complete']=not failures","result['complete']=nearby_task_checks==nearby_expected_checks and not failures")
 a=template.index('    def compact(value):');template=template[:a]+'''    import struct
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
    return ['META S18 on frozen S17;16 disjoint held-out seeds,fresh weights/router;public TASK all arms;only uncertain piece leaveouts;scored host unchanged timed code',
            'S18_XZ85 '+blob,'END complete='+str(result['complete'])]
'''
 body+=template;encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
 payload='import base64,lzma,linecache,hashlib\n_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
 payload+='assert hashlib.sha256(_body.encode()).hexdigest()=='+repr(sha(body))+'\n_f="<s18-lab-'+sha(body)[:12]+'>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=lab\ndef lab(deadline,weights,state):return _run(deadline,weights,state)\n'
 (H/'generated/payload.py').write_text(payload)
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
 wrapper,_=h200_lab.build([('s18_pair',sha(payload)[:16],payload)],'muse_decode_layer',phase='import',import_budget=620,base_file=hp)
 wp=P/'lab'/('lab_s18_'+meta['sha256'][:12]+'.py');wp.write_text(wrapper);assert len(wrapper.encode())<1_000_000
 meta.update(candidate=str((H/'generated/s18.py').relative_to(R)),shipped_source=str(candidate.relative_to(R)),shipped_sha256=sha(packed),wrapper=str(wp.relative_to(R)),wrapper_sha256=sha(wrapper),payload_sha256=sha(payload),checker_sha256=sha(checker),baseline_arm='INC',nearby_expected_checks=260,arms=arms,selection_seeds=seeds['selection_seeds'],confirmation_seeds=seeds['confirmation_seeds'],expected_TASK_checks=12*16*(len(arms)+2))
 (H/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['sha256'],meta['wrapper_sha256'],len(wrapper),meta['expected_TASK_checks'])
if __name__=='__main__':main()
