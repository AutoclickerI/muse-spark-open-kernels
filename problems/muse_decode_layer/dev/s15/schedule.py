"""Admit the combined payload, then queue exactly one check at the UTC reset."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[3]
RESET='2026-10-26T00:00:00+00:00'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    candidate=R/'problems/muse_decode_layer/cand_r3_best7_s15.py'
    manifest=json.loads((H/'manifest.json').read_text())
    assert sha(candidate)==manifest['sha256']
    for mode in ('normal','forced'):
        report=json.loads((H/f'screen/{mode}/{mode}_local.json').read_text())
        assert report['sha256']==manifest['sha256']
        for key,count in (('tests',13),('shapes',12),('static',25)):
            assert len(report[key])==count and all(x['ok'] for x in report[key]),(mode,key)
        lab=json.loads((H/('lab_'+mode+'_validation.json')).read_text())
        assert lab['host_sha256']==manifest['sha256'] and lab['complete'] and lab['seeds']==16
    pieces=json.loads((H/'piece_admission.json').read_text())
    assert set(pieces)=={'proj','last','regmax'}
    for piece,row in pieces.items():
        assert set(row)=={'normal','forced'}
        for mode,report in row.items():
            assert all(c['count']==c['pass_count'] for c in report['checks'].values())
    layouts=json.loads((H/'hopper_layout_inventory.json').read_text())
    assert layouts['passed'] and layouts['candidate_sha256']==manifest['sha256'] and len(layouts['rows'])==9
    assert 'Permission is hereby granted, free of charge' in candidate.read_text()
    payload=H/'s15ab_payload.py';host=H/'s15ab_host.py'
    ab=json.loads((H/'ab_manifest.json').read_text())
    assert sha(payload)==ab['payload_sha256'] and sha(host)==ab['host_file_sha256']
    wrapper=R/'problems/muse_decode_layer/lab/lab_s15ab_payload.py'
    wrapper_sha=sha(wrapper)
    assert all(json.loads((H/('lab_'+mode+'_local.json')).read_text())['sha256']==wrapper_sha for mode in ('normal','forced'))
    command=[sys.executable,str(R/'tools/h200_lab.py'),'send',str(payload.relative_to(R)),
             '--host','muse_decode_layer','--base-file',str(host.relative_to(R)),
             '--check','--not-before',RESET,'--priority','-1',
             '--note','P4 combined S15 60254883 on exact S14 f1c77ce6: nine attributed NF rows + last-CTA B1 sliding + N128 regmax full verify stage4; every piece/role off, S14 back,A:A,16 fresh seeds,all12 shapes,32 small/8 other cold rounds; normal+forced combined and each piece pass; MIT retained. P10b selected-row admission coordinated via README.']
    subprocess.run(command,cwd=R,check=True)
    assert sha(wrapper)==wrapper_sha,'send rebuilt a different wrapper than the admitted one'
    queue=dict(candidate=str(candidate.relative_to(R)),candidate_sha256=manifest['sha256'],
               wrapper=str(wrapper.relative_to(R)),wrapper_sha256=wrapper_sha,payload_sha256=sha(payload),
               not_before=RESET,mode='check',priority=-1,job=None,
               status='scheduled; no H200 performance result or final promotion',
               command=command)
    (H/'queue.json').write_text(json.dumps(queue,indent=2)+'\n')
    print(json.dumps(queue,indent=2))


if __name__=='__main__':main()
