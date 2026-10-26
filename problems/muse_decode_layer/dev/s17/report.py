"""S17 held-out promotion against frozen S16, with lossless sample decoding."""
import argparse,base64,hashlib,json,lzma,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(H.parent/'s16'))
from score_gate import promotion


def decode(raw):
    names=raw['sample_arms'];labels=raw['sample_loads'];scale=raw['sample_scale']
    binary=base64.b85decode(raw.pop('sample_i32_b85'));count=len(names)*sum(raw['loads'][label]['n'] for label in labels)
    assert len(binary)==4*count
    values=struct.unpack('<'+'i'*count,binary);offset=0
    for label in labels:
        rows=[]
        for _ in range(raw['loads'][label]['n']):
            rows.append({name:1+values[offset+j]/scale for j,name in enumerate(names)});offset+=len(names)
        raw['loads'][label]['samples']=rows
    assert offset==count
    return raw


def evaluate(receipt,raw,manifest):
    assert raw['host_sha256']==manifest['sha256'] and raw['checker_sha256']==manifest['checker_sha256']
    assert raw['seed_ids']==manifest['confirmation_seeds']
    if not raw['complete']:
        return dict(job=receipt['id'],source_sha256=manifest['sha256'],freeze_gate=False,
            score=dict(R=None,CI95=None,TASK_pass=False,promote=False),measured=False,
            reason='Incomplete confirmation; no promotion or timing-loss claim',failures=raw['failures'],raw=raw)
    expected=manifest.get('nearby_expected_checks',260)
    assert raw['nearby_expected_checks']==expected
    score=promotion(raw,manifest.get('baseline_arm','INC'),'A',official_pass=receipt['result'].get('passed'),selection_seeds=manifest['selection_seeds'])
    assert not score['TASK_pass'] or raw['nearby_task_checks']==expected
    pieces={arm:promotion(raw,arm,'A',official_pass=receipt['result'].get('passed'),selection_seeds=manifest['selection_seeds']) for arm in manifest['arms'] if arm!='INC'}
    return dict(job=receipt['id'],source_sha256=manifest['sha256'],freeze_gate=score['promote'],score=score,pieces=pieces,raw=raw,
        piece_interpretation='Leave-out/A diagnostic R; >1 favors retaining the piece. No per-piece veto. Any composition selected using these confirmations needs new disjoint confirmation seeds.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args()
    receipt=json.loads(a.receipt.read_text());wrapper=hashlib.sha256(a.receipt.with_suffix('.py').read_bytes()).hexdigest()
    batch=H/'batches'/wrapper[:16];m=json.loads((batch/'lab_manifest.json').read_text());assert m['wrapper_sha256']==wrapper
    lines=[l.split(' ',1)[1] for l in receipt['result']['log_tail'].splitlines() if l.startswith('S17_XZ85 ')]
    if not lines:
        errors=[l for l in receipt['result']['log_tail'].splitlines() if l.startswith(('LAB_ERROR','LAB_TRUNCATED'))]
        result=dict(job=receipt['id'],source_sha256=m['sha256'],freeze_gate=False,R=None,CI95=None,measured=False,
            host_passed=receipt['result'].get('passed'),reason='No paired seed archive; not a timing loss',errors=errors)
        (batch/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return
    assert len(lines)==1
    result=evaluate(receipt,decode(json.loads(lzma.decompress(base64.b85decode(lines[0])))),m)
    (H/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (batch/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['freeze_gate']:
        p=R/m['shipped_source'];assert hashlib.sha256(p.read_bytes()).hexdigest()==m['shipped_sha256']
        (R/'problems/muse_decode_layer/cand_r3_best7_s17_final.py').write_bytes(p.read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw','pieces')},indent=2))
if __name__=='__main__':main()
