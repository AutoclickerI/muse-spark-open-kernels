"""Prepare the one S21 batch lab: 64 fixed seeds, all-round symmetric timing, leave-outs."""
import ast,base64,difflib,json,lzma,sys
from pathlib import Path
import compose
H=compose.H;R=compose.R;P=compose.P;sha=compose.sha
sys.path[:0]=[str(R/'tools'),str(H.parent/'s16')]
from pack import pack
import h200_lab
HOST='muse_bf16_proj';BUDGET=820.
PROTOCOL=dict(name='palindromic-cohort-v1',rounds=16,seed_count=64,aggregation='arithmetic-mean-all-rounds',selective_retiming=False)
ARMS={'INC':(None,None),**{'no_'+p:([x for x in compose.DEFAULT if x!=p],compose.ROWS[p]) for p in compose.DEFAULT}}

def seed_manifest():
    old=json.loads((H.parent/'s20/seed_manifest.json').read_text());selected=set(old['selection_seeds']+old['confirmation_seeds']);review=[]
    def collect(v,key=''):
        if isinstance(v,dict):
            for k,x in v.items():collect(x,k)
        elif isinstance(v,list):
            if key in ('seeds','seed_ids','confirmation_seeds','selection_seeds'):selected.update(x for x in v if isinstance(x,int))
            else:
                for x in v:
                    if isinstance(x,(dict,list)):collect(x)
    for p in sorted((R/'problems').rglob('*.json')):
        if p.name not in ('P4_HANDOFF.json','point_nominations.json'):continue
        collect(json.loads(p.read_text()));review.append(dict(path=str(p.relative_to(R)),sha256=sha(p.read_bytes())))
    seeds=[193400003+104729*i for i in range(PROTOCOL['seed_count'])];assert not selected.intersection(seeds)
    out=dict(selection_seeds=sorted(selected),confirmation_seeds=seeds,weight_seed=193400001,router_seed=193400005)
    (H/'seed_manifest.json').write_text(json.dumps(out,indent=1)+'\n');(H/'handoff_review.json').write_text(json.dumps(dict(files=review,included=list(compose.DEFAULT),
        waiting=['P3a small-M MoE ladder'],
        exclusions='B1, PRIVATE1, AFFINE32 and H2 dropped after S20 leave-outs; no standalone confirmation of S20-noB1'),indent=2)+'\n')
    return out

def main():
    (H/'generated').mkdir(exist_ok=True)
    source,meta=compose.build();(H/'generated/s21.py').write_text(source)
    shipped='\n'.join(line.rstrip() for line in pack(source).splitlines())+'\n';candidate=P/'cand_r3_best7_s21_final.py';candidate.write_text(shipped)
    lines=source.splitlines(True);arms={}
    for name,(enabled,rows) in ARMS.items():
        text=compose.base() if enabled is None else compose.build(enabled)[0];other=text.splitlines(True)
        edits=[(a,b,''.join(other[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,lines,other,autojunk=False).get_opcodes() if op!='equal'];restored=list(lines)
        for a,b,t in reversed(edits):restored[a:b]=[t]
        assert ''.join(restored)==text;arms[name]=dict(sha256=sha(text),edits=edits,rows=rows or [])
    problem=R/'problems/muse_decode_layer';ref='\n'.join(x for x in (problem/'reference.py').read_text().splitlines() if not x.startswith(('from checks import','from nvfp4 import')))
    checker='\n'.join(x for t in ((problem/'checks.py').read_text(),(problem/'nvfp4.py').read_text(),ref) for x in t.splitlines() if not x.startswith('from __future__ import'))
    sd=seed_manifest();hook=next(ast.literal_eval(n.value) for n in ast.parse((H.parent/'s19/build_lab.py').read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='HOOK')
    # Revisit the selected old seed only after confirmation, with its original
    # S20/S19 code and weights. It is never part of the 64 held-out seed pairs.
    repeat_arms={}
    for name,enabled in [('S20',compose.s20.DEFAULT),('INC',None),('no_H2',[p for p in compose.s20.DEFAULT if p!='H2']),('no_B1',[p for p in compose.s20.DEFAULT if p!='B1'])]:
        text=compose.base() if enabled is None else compose.s20.build(enabled)[0];other=text.splitlines(True)
        edits=[(a,b,''.join(other[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,lines,other,autojunk=False).get_opcodes() if op!='equal']
        repeat_arms[name]=dict(sha256=sha(text),edits=edits)
    assert repeat_arms['S20']['sha256']=='9e6f525f76d3baed9bfbef7ceb4051629519fc5bb41ed29fb8782834d0ef3a9b'
    bundle=base64.b85encode(lzma.compress(json.dumps(dict(host=source,arms=arms,repeat_arms=repeat_arms),separators=(',',':')).encode(),preset=9)).decode()
    body='import torch,types,time,statistics,math\n'+(H.parent/'s19/graph_helpers.py.inc').read_text()+'\n'+(H/'timing.py.inc').read_text()+'\nAB_SEEDS='+repr(tuple(sd['confirmation_seeds']))+'\n'
    body+=f"S17_WEIGHT_SEED={sd['weight_seed']}\nS17_ROUTER_SEED={sd['router_seed']}\n"
    body+='import base64,lzma,json\nS16_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\nS16_CHECKER='+repr(checker)+'\n'
    cases=compose.load(H.parent/'nearby/cases.py','s21_nearby_cases').cases()
    body+='S17_NEARBY_CASES='+repr(cases)+"\nS17_NEARBY_ARMS=('A','INC')\nS19_HOST_HOOK="+repr(hook)+'\n'
    body+="S21_REPEAT_ARMS=S16_BUNDLE['repeat_arms']\n"
    body+=(H/'evidence.py').read_text()+'\n'+(H/'repeat_case.py.inc').read_text()+'\n'+(H/'lab.py.inc').read_text();encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    payload='LAB_RESOURCES={"compile_s":300,"run_s":520,"output_bytes":11700,"context_poison":"none"}\nimport base64,lzma,linecache,hashlib\n_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
    payload+='assert hashlib.sha256(_body.encode()).hexdigest()=='+repr(sha(body))+'\n_f="<s21-lab-'+sha(body)[:12]+'>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=lab\ndef lab(deadline,weights,state):return _run(deadline,weights,state)\n'
    pp=H/'generated/s21_pair.py';pp.write_text(payload)
    wrapper,_=h200_lab.build([(pp.stem,sha(payload)[:16],payload)],HOST,phase='import',import_budget=BUDGET,lab_only=True)
    assert len(wrapper.encode())<1_000_000;wp=H/'generated/wrapper.py';wp.write_text(wrapper)
    labels=['d1S','d1F','v1S','v1F','d16S','d16F','d32S','d32F','v16S','v16F','v32S','v32F']
    expected=len(sd['confirmation_seeds'])*sum(2+sum(not a['rows'] or label in a['rows'] for a in arms.values()) for label in labels)
    meta.update(candidate=str((H/'generated/s21.py').relative_to(R)),shipped_source=str(candidate.relative_to(R)),shipped_sha256=sha(shipped),host=HOST,import_budget=BUDGET,
        payload=str(pp.relative_to(R)),payload_sha256=sha(payload),wrapper=str(wp.relative_to(R)),wrapper_sha256=sha(wrapper),sent_wrapper=f'problems/{HOST}/lab/lab_s21_pair.py',
        checker_sha256=sha(checker),baseline_arm='INC',nearby_expected_checks=260,nearby_arms=['A','INC'],arms={n:dict(sha256=a['sha256'],rows=a['rows']) for n,a in arms.items()},
        **sd,expected_TASK_checks=expected,measurement_protocol=PROTOCOL,official_TASK_run=None,
        diagnostic_repeat=dict(seed=1730886002,weight_seed=1729315001,router_seed=1729315003,shape='v16S',placement='after all confirmation pairs',sources={n:a['sha256'] for n,a in repeat_arms.items()}))
    (H/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['sha256'],meta['shipped_sha256'][:12],meta['wrapper_sha256'][:12],len(wrapper),expected)
if __name__=='__main__':main()
