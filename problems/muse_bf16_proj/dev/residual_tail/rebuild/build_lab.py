"""Rank original N and its one-change tail specialization on disjoint seeds."""
import base64
import hashlib
import json
import lzma
from pathlib import Path

H=Path(__file__).resolve().parent;P=H.parent

def once(s,a,b):
    assert s.count(a)==1,(a[:90],s.count(a))
    return s.replace(a,b,1)

def main():
    body=(P/'lab_generated.py.inc').read_text()
    import ast
    old=next(ast.get_source_segment(body,n) for n in ast.parse(body).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='RT_RUNTIME')
    body=once(body,old,'RT_RUNTIME='+repr((H/'runtime_generated.py.inc').read_text()))
    body=once(body,'RT_SEEDS=tuple(9101669+104729*i for i in range(32))','RT_SEEDS=tuple(13101669+130363*i for i in range(32))')
    body=once(body,"RT_ARMS=('A','A2','V','N','C')","RT_ARMS=('A','A2','N','T')")
    body=once(body,"ns['rt_module']();act=t15_activity()","ns['rt_module']();ns['rt_fast_module']();act=t15_activity()")
    body=once(body,'        sig={}\n','        sig={};criticals={}\n')
    mark="            if arm not in ('A','A2'):assert len(bounds)==2 and all('boundary_tail' in n['name'] for n in bounds),('new branch not reached',arm)"
    extra='''
            streams={}
            for n in p['trace']:
                if n['stream'] not in streams:streams[n['stream']]=len(streams)
            criticals[arm]=dict(wall_us=round(p['account']['wall_us'],3),critical_us={k:round(v,3) for k,v in p['account']['critical_us'].items()},
                max_accounting_error=p['max_accounting_error'],
                trace=[[n['role'],round(n['start_ns']/1000,3),round(n['end_ns']/1000,3),streams[n['stream']]] for n in p['trace']])
            if arm=='T':assert 'rtf_cl::boundary_tail' in next(n['name'] for n in bounds if n['role']=='b2'),'fast tail not launched'
'''
    body=once(body,mark,mark+extra)
    body=once(body,"        assert sig['A']==sig['A2'],'A:A schedule fingerprint differs'","""        assert sig['A']==sig['A2'],'A:A schedule fingerprint differs'
        assert profiles['N'][0]['name']==profiles['T'][0]['name'] and profiles['N'][0]['registers']==profiles['T'][0]['registers'],'N b1 changed'""")
    body=once(body,'aa_schedule_identical=True,profiles=profiles,samples=samples','aa_schedule_identical=True,profiles=profiles,criticals=criticals,samples=samples')
    body=body.replace('V=tail,N=normstats,C=both','N=measured stats,T=dedicated tail;N b1 identical')
    compile(body,'p7-normtail-rebuild','exec');tag=hashlib.sha256(body.encode()).hexdigest()
    encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    boot='import base64,hashlib,linecache,lzma\n_RF_BODY=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\nassert hashlib.sha256(_RF_BODY.encode()).hexdigest()=='+repr(tag)+'\n_RF_FILE="<p7-normtail-'+tag[:12]+'>"\nlinecache.cache[_RF_FILE]=(len(_RF_BODY),None,_RF_BODY.splitlines(True),_RF_FILE)\nexec(compile(_RF_BODY,_RF_FILE,"exec"),globals())\n_RF_RUN=lab\ndef lab(deadline,weights,state=None):\n    return _RF_RUN(deadline,weights,state)\n'
    (H/'lab_generated.py.inc').write_text(body);(H/'lab_normtail.py').write_text(boot)
    m=dict(payload_sha256=hashlib.sha256(boot.encode()).hexdigest(),body_sha256=tag,
           selection_job='e841e2e8b1d04f0c',shapes=['d1S','d1F','v1S','v1F'],
           heldout_seeds=[13101669+130363*i for i in range(32)],arms=['A','A2','N','T'],
           expected_TASK_checks=512,rounds=8,mechanism='Only T b2 tail path changes; N b1 handle unchanged; compact complete CUPTI stage intervals/critical increments')
    (H/'lab_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('payload',m['payload_sha256'][:8])

if __name__=='__main__':main()
