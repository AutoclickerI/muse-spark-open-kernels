"""Regression tests for the score rule and the original over-gating failures."""
import copy,json
from pathlib import Path
from report import evaluate
from score_gate import promotion,interval,scores
H=Path(__file__).resolve().parent


def main():
    seeds=list(range(200,216));ratios=[1.06,1.05,1.03,.99,1.01,1.02]
    raw=dict(host_sha256='synthetic',checker_sha256='public-TASK',seeds=16,seed_ids=seeds,complete=True,failures=[],task_checks=12*16*4,
        loads={str(i):dict(samples=[dict(A=1.,A2=1+(.002 if j%2 else -.002),S15=r*(1+(.0005 if j%2 else -.0005)),piece=.999) for j in range(16)]) for i,r in enumerate(ratios)})
    manifest=dict(sha256='synthetic',checker_sha256='public-TASK',confirmation_seeds=seeds,selection_seeds=list(range(16)),arms={'S15':{},'piece':{}})
    receipt={'id':'synthetic','result':{'passed':True}}
    result=evaluate(receipt,raw,manifest)
    assert result['freeze_gate'] and not result['pieces']['piece']['promote']
    assert result['score']['loads']['3']['R']<1
    noisy=copy.deepcopy(raw)
    for cell in noisy['loads'].values():
        for j,row in enumerate(cell['samples']):row['S15']=1+(.2 if j%2 else -.2)
    assert not evaluate(receipt,noisy,manifest)['freeze_gate']
    for key,value in [('failures',['official TASK failure']),('task_checks',1),('complete',False)]:
        bad=copy.deepcopy(raw);bad[key]=value
        assert not evaluate(receipt,bad,manifest)['freeze_gate']
    try:promotion(raw,'S15','A',official_pass=True,selection_seeds=[200])
    except AssertionError:pass
    else:raise AssertionError('selection seed leaked into confirmation')
    assert not evaluate({'id':'s','result':{'passed':False}},raw,manifest)['freeze_gate']
    x=[2.,.6,1,1,1,1];y=[1.05]*6
    assert sum(x)>sum(y) and interval(x)['R']<interval(y)['R']
    shape_times={('d' if q==1 else 'v')+str(b)+w:[dict(A=2 if w=='S' else 4,B=1) for _ in range(16)] for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for w in ('S','F')}
    rr,_=scores(shape_times,'A','B');assert all(abs(v-(46*2+16*4)/62)<1e-12 for v in rr)
    report=dict(passed=True,synthetic_only=True,checks=['six-load win with losing load and losing piece','noise CI crossing one rejected','official TASK required on every arm/seed','selection/confirmation disjoint','geomean policy ranking differs from arithmetic mean','46/16 load weights'],R=result['score']['R'],CI95=result['score']['CI95'])
    (H/'score_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
