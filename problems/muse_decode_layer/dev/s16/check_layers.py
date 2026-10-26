"""Synthetic receipt tests for S15n/e/L selection, never H200 evidence."""
import copy,json
from pathlib import Path
import decide,layer_decisions,patches
H=Path(__file__).resolve().parent


def main():
    specs=patches.registry();keys=json.loads((H/'layer_recovery.json').read_text());raws={}
    for name,host,groups in [('release','E',('attention','moe','projections')),('library','L',('o','pool128','unpool128','v1full','v32slide'))]:
        loads={}
        for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
            load=f'{b}x{q}';r={'A2':[1.,.0001,.999,1.001],host:[.9,.0001,.899,.901]}
            for group in groups:
                active=any((bb,qq)==(b,q) for bb,qq,w in specs['layer_'+name+'_'+group]['rows'])
                value=1. if active else .9;r['no_'+group]=[value,.0001,value-.001,value+.001]
            if name=='release':r['no_glue']=[.9,.0001,.899,.901]
            loads[load]=r
        raws[name]=dict(complete=True,seeds=8,shapes={str(i):dict(n=8) for i in range(12)},loads=loads,errors={},failures=[])
    cases=decide.importlib.util.spec_from_file_location('layer_test_cases',H.parent/'nearby/cases.py');m=decide.importlib.util.module_from_spec(cases);cases.loader.exec_module(m)
    rows=[[c['label'],1. if c['label'].startswith('scored-') else 2.,1. if c['label'].startswith('scored-') else 1.2,3.,1. if c['label'].startswith('scored-') else .6,1. if c['label'].startswith('scored-') else .6,True] for c in m.cases()]
    raws['nearby']=dict(complete=True,shape_count=65,baseline_error=None,errors={},rows=rows)
    names={entry['sha256']:name for name,entry in keys.items()};original=decide.read_receipt
    def read(entry,files):return dict(log='RESULT '+json.dumps(raws[names[entry['sha256']]]),job='synthetic-only')
    decide.read_receipt=read
    try:
        selected,evidence,_=layer_decisions.nominate([]);assert len(selected)==8,[s['name'] for s in selected]
        # One losing load cannot veto a positive six-load policy.
        raws['release']['loads']['32x8']['E']=[1.05,.0001,1.04,1.06]
        selected,proof,_=layer_decisions.nominate([])
        assert any(s['name']=='layer_release_moe' for s in selected)
        # Untouched-load noise cannot nominate a shape-dispatchable loser.
        raws['library']['loads']['1x1']['no_unpool128']=[1.,.0001,.999,1.001]
        raws['library']['loads']['16x8']['no_unpool128']=[.899,.0001,.898,.900]
        selected,_,_=layer_decisions.nominate([])
        assert not any(s['name']=='layer_library_unpool128' for s in selected)
        raws['library']['loads']['32x8'].pop('no_v32slide')
        selected,evidence,_=layer_decisions.nominate([]);assert not any(s['name'].startswith('layer_library_') for s in selected)
        raws['nearby']['rows'][-1][-1]=False
        selected,evidence,_=layer_decisions.nominate([]);assert not any(s['name']=='layer_nearby' for s in selected)
    finally:decide.read_receipt=original
    result=dict(passed=True,synthetic_only=True,checks=['eight positive score nominations; scored-neutral nearby diagnostic','incomplete layer batch rolled back','nearby-only evidence cannot promote', 'untouched noise cannot select an affected-shape loser'])
    (H/'layer_decision_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
