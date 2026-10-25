"""Final pre-reset handoff audit; no judge calls and no queue mutations."""
import collections,hashlib,json
from pathlib import Path
import patches
H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    arms=patches.registry();ad=json.loads((H/'admission.json').read_text());native=json.loads((H/'native_compile.json').read_text())
    preparations={}
    for name in arms:
        digest=patches.compose([name])[1]['sha256']
        assert native[name]['sha256']==digest and native[name]['passed'],(name,'native')
        preparations[name]={}
        for mode in ('normal','forced'):
            row=ad[name][mode]
            assert row['sha256']==digest and row['passed'] and row['counts']['tests']==13 and row['counts']['shapes']==12,(name,mode)
            folder=H/Path(row['local_report']).parent
            precision_path=folder/'precision_inventory.json';precision=json.loads(precision_path.read_text())
            calls=json.loads((folder/'calls.json').read_text())['calls']
            assert precision and any(c['captured'] for c in calls.values()),(name,mode,'prepare/graph evidence absent')
            preparations[name][mode]=dict(sha256=digest,inventory_sha256=hashlib.sha256(precision_path.read_bytes()).hexdigest(),
                tensor_paths=len(precision),classes=dict(collections.Counter(t['holds'] for t in precision.values())),calls=calls)
    (H/'prepare_audit.json').write_text(json.dumps(preparations,indent=2)+'\n')
    rehearsal=json.loads((H/'rehearsal.json').read_text())
    assert rehearsal['passed'] and not rehearsal['remote_send'] and not rehearsal['performance_claim']
    assert all(row['paired_shapes']==12 and row['seeds']==16 and row['packed_tests']==13 and row['packed_shapes']==12 and row['task_checks']==1536 for row in rehearsal['screens'].values())
    decisions=json.loads((H/'decision_validation.json').read_text());assert decisions['passed'] and decisions['synthetic_only']
    initial=json.loads((H/'reset_inventory.json').read_text());waiting={p.stem:json.loads(p.read_text()) for p in (R/'runs/waiting').glob('*.json')}
    for row in initial:
        current=waiting[str(row['pid'])]
        for key in ('file','sha256','priority','until','not_before'):assert current.get(key)==row.get(key),(row['pid'],key)
        snapshot=R/'runs/sent'/row['sha256'][:16]/Path(row['file']).name
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==row['sha256']
    result=dict(passed=True,base_sha256=patches.BASE_SHA,arms=len(arms),normal_forced_task_checks=13,
                graph_shapes=12,native_arms=len(native),rehearsal_elapsed_s=rehearsal['elapsed_s'],
                reset_senders_unchanged=[r['pid'] for r in initial],new_h200_jobs=[],
                depends_on_reset=True,response_command='python3 problems/muse_decode_layer/dev/s16/respond.py')
    (H/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
