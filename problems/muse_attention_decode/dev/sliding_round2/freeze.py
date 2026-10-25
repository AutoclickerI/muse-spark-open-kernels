"""Record the exact screened source and queue evidence without retaining rebuildable output trees."""
import ast,hashlib,json,os
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    packet=H/'out/sliding_round2_board.py'
    files=sorted(p.name for p in H.iterdir() if p.suffix in ('.py','.inc'))
    record=dict(files={f:sha(H/f) for f in files},board_sha256=sha(packet),layer_sha256=sha(H/'out/s14_round2.py'),
                payload_sha256=sha(H/'out/sliding_round2_lab.py'),base_sha256=sha(R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py'),
                native=json.loads((H/'out/native_admission.json').read_text()),
                component=json.loads((H/'out/variant_admission.json').read_text()),
                index=json.loads((H/'out/index_admission.json').read_text()),H200_job_ids=[],promotion='none; native pair pending')
    for name in ('normal_board','forced_board'):
        x=json.loads((H/'out'/(name+'.json')).read_text());assert x['sha256']==record['board_sha256'];assert all(v['ok'] for v in x['tests'])
        record[name]=dict(tests=len(x['tests']),shapes=len(x['shapes']),static=x['static'],ok=True)
    for name in ('sliding_round2_lab','followup_dsm','followup_history'):
        p=H/'out'/(name+'_admission.json');x=json.loads(p.read_text())
        assert x['payload_sha256']==sha(H/'out'/(name+'.py')),name
        record[name]=dict(payload_sha256=sha(H/'out'/(name+'.py')),cases=[r['case'] for r in x['results']],
                         arms={r['case']:list(r['arms']) for r in x['results']},
                         task_passed={r['case']:r['task_passed'] for r in x['results']},local_only=True,
                         one_attention_node=all(len(p['nodes'])==1 for r in x['results'] for n,p in r['profiles'].items() if n not in ('A','A2')))
        assert record[name]['one_attention_node']
        assert all(all(v==2 for v in r['task_passed'].values()) for r in x['results'])
    dependencies=['prof/lab/last_sliding_timeline.py','problems/muse_attention_decode/dev/early_history/lab_body.inc','problems/muse_attention_decode/dev/early_history/history_build.py',*['problems/muse_decode_layer/'+f for f in ('nvfp4.py','checks.py','reference.py')]]
    record['dependencies']={p:sha(R/p) for p in dependencies}
    record['packing']=json.loads((H/'out/packing_admission.json').read_text())
    record['layer_screens']={}
    for name,folder,file in [('normal','layer_normal','normal_local.json'),('forced','layer_forced','forced_local.json')]:
        x=json.loads((H/'out'/folder/file).read_text());assert x['sha256']==record['layer_sha256'];assert all(t['ok'] for t in x['tests'])
        record['layer_screens'][name]=dict(tests=len(x['tests']),shapes=len(x['shapes']),static=len(x['static']),ok=True)
    queues=[]
    targets=(record['board_sha256'][:8],'0ccc2548','f1c270e6','0cec6706')
    for p in (R/'runs/waiting').glob('*.json'):
        try:x=json.loads(p.read_text())
        except (ValueError,OSError):continue
        if x.get('sha256','')[:8] in targets:os.kill(x['pid'],0);queues.append(x)
    record['queues']=queues
    (H/'admission.json').write_text(json.dumps(record,indent=2)+'\n');print(record['board_sha256'],len(queues),'live queues')
if __name__=='__main__':main()
