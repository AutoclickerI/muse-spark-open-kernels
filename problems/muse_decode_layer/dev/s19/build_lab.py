"""S19 candidate, its arms, and the import-time paired lab against frozen S18 on a lab-only prefill host."""
import base64,difflib,hashlib,importlib.util,json,lzma,sys
from pathlib import Path
import compose
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1]
sys.path[:0]=[str(H.parent/'s16'),str(R/'tools')]
from pack import pack
import h200_lab
sha=compose.sha
HOST='muse_prefill_layer';BUDGET=780.0;HOST_FILE=H/'prefill_host.py'
D=compose.DEFAULT
# name: (compose kwargs, rows the arm changes; None = every shape). 7c0fd3cc chose S19's rows; these arms re-test
# the three choices it made on fresh seeds (P10 at T=16, pdl_M's _boundary_late, the neutral FAST).
ARMS={
 'INC':(None,None),
 'P10_d16':(dict(p10={(8,2304):4,(16,2304):4,(16,1280):4}),['d16S','d16F']),
 'late_pdl':(dict(enabled=[n for n in D if n!='LATE']),['v32S','v32F']),
 'no_FAST':(dict(enabled=[n for n in D if n!='FAST']),['d1S','d1F']),
}
HOOK='''
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
_S19_BUILD_S={}
def _s19_timed(fn):
    def run(*args):
        import time as _t;t0=_t.monotonic()
        try:return fn(*args)
        finally:_S19_BUILD_S[getattr(fn,'__name__',repr(fn))]=round(_t.monotonic()-t0,2)
    return run
_s16_native_builds[:]=[_s19_timed(fn) for fn in _s16_native_builds]
_S19_BUILDS[:]=[_s19_timed(fn) for fn in _S19_BUILDS]
_S19_PHASE_S={}
def _s19_phase(name):
    fn=globals()[name]
    def run(*args,**kwargs):
        import time as _t;t0=_t.monotonic()
        try:return fn(*args,**kwargs)
        finally:_S19_PHASE_S[name]=[round(t0-_S19_T0[0],1),round(_t.monotonic()-t0,1)]
    globals()[name]=run
_S19_T0=[0.]
for _n in ('_prepare_weights','_best5_tune_moe','_best6_tune','_tune_projections','_tune_measured_lt','_tune_attention_portfolio',
           '_prepare_recoded_bank','_s4_prepare_banks','_best7_prepare_settings','nf_prepare','_s16_prepare_delta'):_s19_phase(_n)
_s19_hook_prepare=_candidate_prepare
def _candidate_prepare(weights):
    import time as _t;_S19_T0[0]=_t.monotonic();return _s19_hook_prepare(weights)
'''


def seeds():
    p=H/'seed_manifest.json'
    if p.exists():return json.loads(p.read_text())
    s18=json.loads((H.parent/'s18/seed_manifest.json').read_text())
    # S18's confirmation seeds chose S19's pieces and 7c0fd3cc's (504910273 + 104729*i) chose their rows: both select.
    m=dict(selection_seeds=s18['selection_seeds']+s18['confirmation_seeds']+[504910273+104729*i for i in range(16)],
           confirmation_seeds=[610351273+104729*i for i in range(16)],weight_seed=610351271,router_seed=610351269)
    assert not set(m['selection_seeds'])&set(m['confirmation_seeds']);p.write_text(json.dumps(m,indent=1)+'\n');return m


def main():
    source,meta=compose.build();(H/'generated/s19.py').write_text(source);meta['pins']=compose.pins()
    packed=pack(source);candidate=P/('cand_r3_best7_s19_'+meta['sha256'][:12]+'.py');candidate.write_text(packed)
    lines=source.splitlines(True);arms={}
    for name,(kw,rows) in ARMS.items():
        text=(H.parent/'s18/generated/s18.py').read_text() if kw is None else compose.build(**kw)[0]
        if kw is None:assert sha(text)==compose.S18_SHA
        other=text.splitlines(True);edits=[(a,b,''.join(other[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,lines,other,autojunk=False).get_opcodes() if op!='equal'];restored=list(lines)
        for a,b,t in reversed(edits):restored[a:b]=[t]
        assert ''.join(restored)==text;arms[name]=dict(sha256=sha(text),edits=edits,rows=rows or [])
    problem=R/'problems/muse_decode_layer'
    reference='\n'.join(line for line in (problem/'reference.py').read_text().splitlines() if not line.startswith(('from checks import','from nvfp4 import')))
    checker='\n'.join(line for text in ((problem/'checks.py').read_text(),(problem/'nvfp4.py').read_text(),reference) for line in text.splitlines() if not line.startswith('from __future__ import'))
    sd=seeds()
    bundle=base64.b85encode(lzma.compress(json.dumps(dict(host=source,arms=arms),separators=(',',':')).encode(),preset=9)).decode()
    body='import torch,types,time,statistics,math\n'+(H/'graph_helpers.py.inc').read_text()+'\nAB_SEEDS='+repr(tuple(sd['confirmation_seeds']))+'\n'
    body+=f"S17_WEIGHT_SEED={sd['weight_seed']}\nS17_ROUTER_SEED={sd['router_seed']}\n"
    body+='import base64,lzma,json\nS16_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\nS16_CHECKER='+repr(checker)+'\n'
    sp=importlib.util.spec_from_file_location('s19_nearby_cases',H.parent/'nearby/cases.py');cases=importlib.util.module_from_spec(sp);sp.loader.exec_module(cases)
    body+='S17_NEARBY_CASES='+repr(cases.cases())+"\nS17_NEARBY_ARMS=('A','INC')\nS19_HOST_HOOK="+repr(HOOK)+'\n'
    body+=(H/'lab.py.inc').read_text();encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    payload='import base64,lzma,linecache,hashlib\n_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
    payload+='assert hashlib.sha256(_body.encode()).hexdigest()=='+repr(sha(body))+'\n_f="<s19-lab-'+sha(body)[:12]+'>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=lab\ndef lab(deadline,weights,state):return _run(deadline,weights,state)\n'
    pp=H/'generated/s19_pair.py';pp.write_text(payload)
    wrapper,_=h200_lab.build([(pp.stem,sha(payload)[:16],payload)],HOST,phase='import',import_budget=BUDGET,base_file=HOST_FILE)
    assert len(wrapper.encode())<1_000_000,len(wrapper.encode());wp=H/'generated/wrapper.py';wp.write_text(wrapper)
    expected=16*sum(2+sum(1 for a in arms.values() if not a['rows'] or compose_label in a['rows']) for compose_label in
                    [('d' if q==1 else 'v')+str(b)+('S' if w else 'F') for b,q,w in ((1,1,2048),(1,1,0),(1,8,2048),(1,8,0),(16,1,2048),(16,1,0),(32,1,2048),(32,1,0),(16,8,2048),(16,8,0),(32,8,2048),(32,8,0))])
    meta.update(candidate=str((H/'generated/s19.py').relative_to(R)),shipped_source=str(candidate.relative_to(R)),shipped_sha256=sha(packed),
                host=HOST,host_file=str(HOST_FILE.relative_to(R)),import_budget=BUDGET,payload=str(pp.relative_to(R)),payload_sha256=sha(payload),
                wrapper=str(wp.relative_to(R)),wrapper_sha256=sha(wrapper),sent_wrapper=f'problems/{HOST}/lab/lab_{pp.stem}.py',checker_sha256=sha(checker),
                baseline_arm='INC',nearby_expected_checks=260,arms={k:dict(sha256=v['sha256'],rows=v['rows']) for k,v in arms.items()},
                selection_seeds=sd['selection_seeds'],confirmation_seeds=sd['confirmation_seeds'],expected_TASK_checks=expected,official_TASK_run=None)
    (H/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['sha256'],meta['shipped_sha256'][:12],meta['wrapper_sha256'][:12],len(wrapper),expected)
if __name__=='__main__':main()
