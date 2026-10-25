"""Read paired whole-layer ratios and their A:A controls from an exact receipt."""
import argparse
import base64
import json
import lzma
import re
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args()
d=json.loads(a.receipt.read_text())
if 'result' in d and isinstance(d['result'],dict) and 'loads' in d['result']:r=d['result']
else:
 found=[json.loads(s) for s in re.findall(r'^RESULT (\{.*\})$',d['result'].get('log_tail',''),re.M)]
 found += [json.loads(lzma.decompress(base64.b85decode(s))) for s in re.findall(r'^RESULT_XZ85 (\S+)$',d['result'].get('log_tail',''),re.M)]
 found=[x for x in found if x.get('seeds')==8 and 'native_sources_unchanged' in x]
 assert len(found)==1,'expected one library result';r=found[0]
print('complete',r['complete'],'errors',r['errors'],'failures',r['failures'])
print('| load | A:A | L/S15 | L SE | no-O | no-pool128 | no-unpool128 | no-v1full | no-v32slide |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for load,v in r['loads'].items():
 print('| '+load+' | '+' | '.join(f'{x:.6f}' for x in [v['A2'][0],v['L'][0],v['L'][1],v['no_o'][0],v['no_pool128'][0],v['no_unpool128'][0],v['no_v1full'][0],v['no_v32slide'][0]])+' |')
print('Smaller is faster; a replacement wins if L beats its leave-one-out beyond A:A spread. Local sm120 timings are not H200 evidence.')
