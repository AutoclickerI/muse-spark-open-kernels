"""S11 completion increments from the existing timeline; no new S14 timing claim."""
import json,re,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];s=subprocess.check_output(['git','show','ec55726:problems/muse_decode_layer/dev/timeline/README.md'],cwd=ROOT,text=True);out={}
for load in ('d16','d32','v16','v32'):
 cells={}
 for case,stage,span,inc in re.findall(r'\| ('+load+r'[SF]) \| (b[12]) \| ([\d.]+) \| ([\d.]+)',s):cells.setdefault(case,{})[stage]=dict(residence_us=float(span),increment_us=float(inc))
 assert len(cells)==2 and all(len(v)==2 for v in cells.values())
 out[load]=dict(cases=cells,combined_increment_per_layer_us=sum((46 if case.endswith('S') else 16)*sum(r['increment_us'] for r in v.values()) for case,v in cells.items())/62)
(HERE/'s11_increments.json').write_text(json.dumps(out,indent=2)+'\n')
print({k:v['combined_increment_per_layer_us'] for k,v in out.items()})
