"""Promote the composed S16 by its held-out six-load R, with TASK checks only."""
import argparse,base64,hashlib,json,lzma
from pathlib import Path
from score_gate import promotion
H=Path(__file__).resolve().parent;R=H.parents[3]


def evaluate(receipt,raw,manifest):
    assert raw['host_sha256']==manifest['sha256']
    assert raw['checker_sha256']==manifest['checker_sha256']
    assert raw['seed_ids']==manifest['confirmation_seeds']
    baseline=manifest.get('baseline_arm','S15')
    score=promotion(raw,baseline,'A',official_pass=receipt['result'].get('passed'),selection_seeds=manifest['selection_seeds'])
    # Off arms and A:A diagnose interactions and noise. They cannot veto the
    # score of the composition, including a piece or load that loses alone.
    pieces={arm:promotion(raw,arm,'A',official_pass=receipt['result'].get('passed'),selection_seeds=manifest['selection_seeds'])
            for arm in manifest['arms'] if arm!=baseline}
    return dict(job=receipt['id'],source_sha256=manifest['sha256'],freeze_gate=score['promote'],score=score,pieces=pieces,raw=raw)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args()
    receipt=json.loads(a.receipt.read_text());wrapper_sha=hashlib.sha256(a.receipt.with_suffix('.py').read_bytes()).hexdigest()
    batch=H/'batches'/wrapper_sha[:16];manifest=json.loads((batch/'lab_manifest.json').read_text())
    assert wrapper_sha==manifest['wrapper_sha256']
    blobs=[line.split(' ',1)[1] for line in receipt['result']['log_tail'].splitlines() if line.startswith('S16_XZ85 ')]
    assert len(blobs)==1,'missing/truncated confirmation archive'
    raw=json.loads(lzma.decompress(base64.b85decode(blobs[0])))
    result=evaluate(receipt,raw,manifest)
    (H/'s16_report.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['freeze_gate']:
        response=json.loads((batch/'response.json').read_text())
        assert response['expanded_sha256']==manifest['sha256']
        packed=(R/response['shipped_source']).read_bytes()
        assert hashlib.sha256(packed).hexdigest()==response['shipped_sha256']
        final=R/'problems/muse_decode_layer/cand_r3_best7_s16_final.py';final.write_bytes(packed)
        print(final,result['score'])
    else:print(manifest.get('baseline_arm','S15'),'remains incumbent; integration confirmation:',result['score'])
if __name__=='__main__':main()
