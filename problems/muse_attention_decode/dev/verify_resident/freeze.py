"""Preserve final source, local admission and actual queue identity; generated build trees stay ignored."""
import hashlib,json,os,re
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    native=json.loads((H/'out/native_admission.json').read_text())
    payload=json.loads((H/'out/payload_admission.json').read_text())
    assert payload['payload_sha256']==sha(H/'out/resident_verify_lab.py')
    assert len(payload['results'])==2 and all(all(n==2 for n in r['task'].values()) for r in payload['results'])
    assert all(len(p['nodes'])==1 for r in payload['results'] for n,p in r['profiles'].items() if n not in ('A','A2'))
    comp=json.loads((H/'out/component_admission.json').read_text());assert len(comp['rows'])==72 and all(v['ok'] for v in comp['rows'])
    record=dict(source_files={p.name:sha(p) for p in H.iterdir() if p.suffix in ('.py','.inc')},
        component_sha256=sha(H/'out/component.py'),board_sha256=sha(H/'out/board.py'),local_wrapper_sha256=sha(H/'out/board_local.py'),
        payload_sha256=sha(H/'out/resident_verify_lab.py'),floor=json.loads((H/'floor.json').read_text()),native=native,
        edge=json.loads((H/'out/edge_admission.json').read_text()),producer=json.loads((H/'out/producer_admission.json').read_text()),
        index=json.loads((H/'out/index_admission.json').read_text()),component_cells=comp,
        local_payload=dict(standin=payload['standin'],shapes=2,arms_per_shape=10,seeds=2,rounds=2,reference_margin=.8,
                           task={r['case']:r['task'] for r in payload['results']},one_attention_launch=True,
                           native_multi_CTA_placement_verified=False),H200_job_ids=[],promotion='none; one native batch queued')
    for name in ('normal_board','forced_board'):
        x=json.loads((H/'out'/(name+'.json')).read_text());assert x['sha256']==record['local_wrapper_sha256'];assert all(t['ok'] for t in x['tests'])
        record[name]=dict(tests=len(x['tests']),shapes=len(x['shapes']),static=x['static'],ok=True)
    queues=[]
    for p in (R/'runs/waiting').glob('*.json'):
        try:x=json.loads(p.read_text())
        except (OSError,ValueError):continue
        if x.get('sha256')==record['board_sha256'] or x.get('sha256','')[:8] in ('0ccc2548','f1c270e6','0cec6706','295a2792'):
            os.kill(x['pid'],0);queues.append(x)
    record['queues']=queues
    (H/'admission.json').write_text(json.dumps(record,indent=2)+'\n');print(record['board_sha256'],len(queues),'live P5 queues')
if __name__=='__main__':main()
