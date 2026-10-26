"""One import-time S20/S19 pair, seven independent leave-outs, fresh seeds."""
import ast,base64,difflib,hashlib,importlib.util,json,lzma,sys
from pathlib import Path
import compose
H=compose.H;R=compose.R;P=compose.P;sha=compose.sha
sys.path[:0]=[str(R/'tools'),str(H.parent/'s16')]
from pack import pack
import h200_lab
HOST='muse_prefill_layer';BUDGET=820.
ARMS={'INC':(None,None),**{'no_'+p:([x for x in compose.DEFAULT if x!=p],compose.ROWS[p]) for p in compose.DEFAULT}}

def seed_manifest():
    old=json.loads((H.parent/'s19/seed_manifest.json').read_text())
    selected=set(old['selection_seeds']+old['confirmation_seeds'])
    review=[]
    def collect(v,key=''):
        if isinstance(v,dict):
            for k,x in v.items():collect(x,k)
        elif isinstance(v,list):
            if key in ('seeds','seed_ids','confirmation_seeds','selection_seeds'):
                selected.update(x for x in v if isinstance(x,int))
            else:
                for x in v:
                    if isinstance(x,(dict,list)):collect(x)
    for p in sorted((R/'problems').rglob('*.json')):
        if p.name not in ('P4_HANDOFF.json','point_nominations.json'):continue
        data=json.loads(p.read_text());collect(data)
        review.append(dict(path=str(p.relative_to(R)),sha256=sha(p.read_bytes()),keys=list(data)))
    for i in range(16):selected.add(14000317+104729*i);selected.add(975353+7919*i)
    seeds=[1729315067+104729*i for i in range(16)];assert not selected.intersection(seeds)
    out=dict(selection_seeds=sorted(selected),confirmation_seeds=seeds,weight_seed=1729315001,router_seed=1729315003)
    (H/'seed_manifest.json').write_text(json.dumps(out,indent=1)+'\n')
    (H/'handoff_review.json').write_text(json.dumps(dict(files=review,included=list(compose.DEFAULT),
        decisions={'C3':'Latest phases C3 repeats original fixed nominee; CI crosses 1, explicit no_C3',
                   'GATE':'P10 exact12 T1 sliding G0 winner', 'M64':'Latest phases v1F stage ratio .90979 CI [.89851,.92106]',
                   'PRIVATE1':'Exact current-S19 source edit at M1, explicit leave-out; alias bias retained as caveat',
                   'AFFINE32':'Exact current-S19 M32 nominee, preserve M16',
                   'H2':'P7 new S18 G0 winner, native heavy W2 pre-wait recipe ported to S19',
                   'B1':'P7 new S18 G0 winner; preserve S19 late b2 revert, stage b1 immutable vectors'},
        exclusions='Already selected/rejected in S19 and older points omitted; current weak EXP/B4/E and other CIs crossing 1 omitted. C4/T16 are alternatives to C3, not additive. U64 loses the M64 point. No free-work oracles.'),indent=2)+'\n')
    return out

def main():
    (H/'generated').mkdir(exist_ok=True)
    source,meta=compose.build();(H/'generated/s20.py').write_text(source)
    shipped='\n'.join(line.rstrip() for line in pack(source).splitlines())+'\n'
    candidate=P/'cand_r3_best7_s20_final.py';candidate.write_text(shipped)
    lines=source.splitlines(True);arms={}
    for name,(enabled,rows) in ARMS.items():
        text=compose.base() if enabled is None else compose.build(enabled)[0]
        other=text.splitlines(True)
        edits=[(a,b,''.join(other[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,lines,other,autojunk=False).get_opcodes() if op!='equal']
        restored=list(lines)
        for a,b,t in reversed(edits):restored[a:b]=[t]
        assert ''.join(restored)==text
        arms[name]=dict(sha256=sha(text),edits=edits,rows=rows or [])
    problem=R/'problems/muse_decode_layer'
    ref='\n'.join(x for x in (problem/'reference.py').read_text().splitlines() if not x.startswith(('from checks import','from nvfp4 import')))
    checker='\n'.join(x for t in ((problem/'checks.py').read_text(),(problem/'nvfp4.py').read_text(),ref) for x in t.splitlines() if not x.startswith('from __future__ import'))
    sd=seed_manifest()
    hook=next(ast.literal_eval(n.value) for n in ast.parse((H.parent/'s19/build_lab.py').read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='HOOK')
    bundle=base64.b85encode(lzma.compress(json.dumps(dict(host=source,arms=arms),separators=(',',':')).encode(),preset=9)).decode()
    body='import torch,types,time,statistics,math\n'+(H.parent/'s19/graph_helpers.py.inc').read_text()+'\nAB_SEEDS='+repr(tuple(sd['confirmation_seeds']))+'\n'
    body+=f"S17_WEIGHT_SEED={sd['weight_seed']}\nS17_ROUTER_SEED={sd['router_seed']}\n"
    body+='import base64,lzma,json\nS16_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\nS16_CHECKER='+repr(checker)+'\n'
    cases=compose.load(H.parent/'nearby/cases.py','s20_cases').cases()
    body+='S17_NEARBY_CASES='+repr(cases)+"\nS17_NEARBY_ARMS=('A','INC')\nS19_HOST_HOOK="+repr(hook)+'\n'
    body+=(H/'lab.py.inc').read_text()
    encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    payload='LAB_RESOURCES={"compile_s":360,"run_s":460,"output_bytes":8744,"context_poison":"none"}\nimport base64,lzma,linecache,hashlib\n_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
    payload+='assert hashlib.sha256(_body.encode()).hexdigest()=='+repr(sha(body))+'\n_f="<s20-lab-'+sha(body)[:12]+'>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=lab\ndef lab(deadline,weights,state):return _run(deadline,weights,state)\n'
    pp=H/'generated/s20_pair.py';pp.write_text(payload)
    wrapper,_=h200_lab.build([(pp.stem,sha(payload)[:16],payload)],HOST,phase='import',import_budget=BUDGET,lab_only=True)
    assert len(wrapper.encode())<1_000_000,len(wrapper.encode())
    wp=H/'generated/wrapper.py';wp.write_text(wrapper)
    labels=['d1S','d1F','v1S','v1F','d16S','d16F','d32S','d32F','v16S','v16F','v32S','v32F']
    expected=16*sum(2+sum(not a['rows'] or label in a['rows'] for a in arms.values()) for label in labels)
    meta.update(candidate=str((H/'generated/s20.py').relative_to(R)),shipped_source=str(candidate.relative_to(R)),shipped_sha256=sha(shipped),
        host=HOST,import_budget=BUDGET,payload=str(pp.relative_to(R)),payload_sha256=sha(payload),wrapper=str(wp.relative_to(R)),wrapper_sha256=sha(wrapper),
        sent_wrapper=f'problems/{HOST}/lab/lab_s20_pair.py',checker_sha256=sha(checker),baseline_arm='INC',nearby_expected_checks=260,
        nearby_arms=['A','INC'],arms={n:dict(sha256=a['sha256'],rows=a['rows']) for n,a in arms.items()},**sd,expected_TASK_checks=expected,official_TASK_run=None)
    (H/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(meta['sha256'],meta['shipped_sha256'][:12],meta['wrapper_sha256'][:12],len(wrapper),expected)
if __name__=='__main__':main()
