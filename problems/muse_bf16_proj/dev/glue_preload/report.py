"""Archive the complete paired receipt, even when human-readable lines elide."""
import base64
import json
import lzma
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
job=sys.argv[1]
receipt,=REPO.glob(f'runs/*/*_{job}.json')
raw=json.loads(receipt.read_text());lines=raw['result']['log_tail'].splitlines()
encoded=next(l.split(' ',1)[1] for l in lines if l.startswith('GP_PRELOAD_XZ85 '))
cells=json.loads(lzma.decompress(base64.b85decode(encoded)))
by_case={r['case']:r for r in cells};loads={}
for load in ('d1','d16','d32','v1','v16','v32'):
    a,b=by_case[load+'S'],by_case[load+'F'];out={}
    for arm in sorted(set(a['paired'])&set(b['paired'])):
        den=46*a['paired'][arm][3]+16*b['paired'][arm][3]
        out[arm]=(46*a['paired'][arm][3]*a['paired'][arm][0]+16*b['paired'][arm][3]*b['paired'][arm][0])/den
    loads[load]=out
valid=len(cells)==12 and all(r['seeds']==12 and not r['gates'] for r in cells)
out=dict(job=job,host='muse_decode_layer',base_sha256='958f1258',valid=valid,
    loads=loads,cells=cells,receipt=str(receipt.relative_to(REPO)))
path=HERE/f'h200_{job}.json';path.write_text(json.dumps(out,indent=2)+'\n')
print('valid',valid,'shapes',len(cells))
for k,v in loads.items():print(k,' '.join(f'{n}={r:.6f}' for n,r in v.items()))
for r in cells:
    if r['gates'] or r['seeds']!=12:print('INVALID',r['case'],r['seeds'],r['gates'])
print(path)
