"""S15 receipt audit using the score gate; exact 60254883 is the owner-frozen base."""
import argparse,base64,lzma
import hashlib
import json
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(H.parent/'s16'))
from score_gate import legacy_summary, promotion


def main():
    ap=argparse.ArgumentParser();ap.add_argument('job');a=ap.parse_args()
    files=list((R/'runs/muse_decode_layer').glob('*_'+a.job+'*.json'));assert len(files)==1,files
    receipt=json.loads(files[0].read_text());queued=json.loads((H/'queue.json').read_text())
    assert hashlib.sha256(files[0].with_suffix('.py').read_bytes()).hexdigest()==queued['wrapper_sha256']
    log=receipt['result'].get('log_tail','')
    rows=[line[7:] for line in log.splitlines() if line.startswith('RESULT ')]
    rows += [lzma.decompress(base64.b85decode(line.split(' ',1)[1])).decode() for line in log.splitlines() if line.startswith('RESULT_XZ85 ')]
    assert len(rows)==1,'missing/truncated RESULT'
    raw=json.loads(rows[0]);manifest=json.loads((H/'ab_manifest.json').read_text())
    assert raw['host_sha256']==manifest['sha256']
    if 'seed_ids' in raw:
        scores={name:promotion(raw,name,'A',official_pass=receipt['result'].get('passed'),selection_seeds=manifest['selection_seeds'])
                for name in raw['loads']['decode 1']['r'] if name!='A2'}
    else:scores={name:legacy_summary(raw['loads'],name) for name in raw['loads']['decode 1']['r'] if name!='A2'}
    report=dict(job=receipt['id'],source_sha256=raw['host_sha256'],official_pass=receipt['result'].get('passed',False),
                incumbent='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb',
                incumbent_basis='Jihoon freezes exact S15; S14 retired; no new admission gate on the incumbent',
                scores=scores,incumbent_frozen=True,measured_promotion_gate=scores['S14']['promote'],raw=raw)
    (H/(receipt['id'][:8]+'_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='raw'},indent=2))
if __name__=='__main__':main()
