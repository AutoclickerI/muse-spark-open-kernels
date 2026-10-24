"""Extract combined boundary/route activity time from the best7 profile receipt."""
from pathlib import Path
import hashlib
import json
import re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
p=ROOT/'problems/muse_decode_layer/profiles/cand_r3_best7_reset.c0e74bedf7b74be5.txt'
rows=[]
for section in p.read_text().split('profile ')[1:]:
    header=section.splitlines()[0]
    if 'verify ' not in header:continue
    row=dict(case=header,boundary_us=0.,route_us=0.)
    for line in section.splitlines()[1:]:
        if line.startswith(('_boundary_fast','_route_fast')):
            found=re.search(r'([\d.]+) µs.*?(\d+)\s*$',line)
            if found:row['boundary_us' if '_boundary' in line else 'route_us']+=float(found[1])
    row['combined_activity_us']=row['boundary_us']+row['route_us']
    row['above_6us']=row['combined_activity_us']>6
    rows.append(row)
result=dict(source=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    caveat='Profile activity includes PDL waits; the whole-layer paired lab decides critical-path gain.',rows=rows)
(HERE/'profile_glue.json').write_text(json.dumps(result,indent=2)+'\n')
for row in rows:print(row['case'].split(':',1)[0],row['boundary_us'],row['route_us'],row['combined_activity_us'])
