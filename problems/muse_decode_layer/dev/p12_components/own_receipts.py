"""Source-verified lane receipts survive shared-worktree ledger restoration."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent;R=H.parents[3]
def receipts():
    lines=(R/'runs/ledger.jsonl').read_text().rsplit('\n',1)[0].splitlines();rows=[]
    for line in lines:
        try:rows.append(json.loads(line))
        except ValueError:continue
    known={x['job'] for x in rows}
    for c in json.loads((H/'receipt_catalog.json').read_text()):
        f=R/c['receipt'];src=R/c['source']
        if c['job'] in known or not f.exists():continue
        assert hashlib.sha256(src.read_bytes()).hexdigest()==c['sha256'],c['job']
        d=json.loads(f.read_text());r=d.get('result') or {};stem=f.stem.split('_')
        rows.append(dict(job=c['job'],sha256=c['sha256'],source=c['source'],utc=stem[0],mode=stem[1],
            status=d.get('status'),speedup=r.get('speedup'),note='source-verified own receipt catalog'))
    return rows
