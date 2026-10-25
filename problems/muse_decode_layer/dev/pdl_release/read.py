"""Read paired whole-layer ratios and their A:A controls from an exact receipt."""
import argparse
import json
import re
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args()
d=json.loads(a.receipt.read_text())
if 'result' in d and isinstance(d['result'],dict) and 'loads' in d['result']:r=d['result']
else:
 found=[json.loads(s) for s in re.findall(r'^RESULT (\{.*\})$',d['result'].get('log_tail',''),re.M)]
 found=[x for x in found if x.get('seeds')==8 and 'native_sources_unchanged' in x]
 assert len(found)==1,'expected one release result';r=found[0]
print('complete',r['complete'],'errors',r['errors'],'failures',r['failures'])
print('| load | A:A | E/S15 | E SE | no-attn | no-MoE | no-proj | no-glue |')
print('|---|---:|---:|---:|---:|---:|---:|---:|')
for load,v in r['loads'].items():
 print('| '+load+' | '+' | '.join(f'{x:.6f}' for x in [v['A2'][0],v['E'][0],v['E'][1],v['no_attention'][0],v['no_moe'][0],v['no_projections'][0],v['no_glue'][0]])+' |')
print('Smaller is faster; compare E against both A:A and the E-identical no-glue control. Local sm120 timings are not H200 evidence.')
