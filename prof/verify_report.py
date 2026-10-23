"""Independent checks against raw receipts, not precomputed report display rows."""
import json,sys,math,subprocess,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));import profile_report as p
D=json.load(open(ROOT/'prof/report.json'));checks=0
for v in D:
 for side in ('ours','sglang'):
  r=v[side];receipt=json.loads((ROOT/r['receipt']).read_text())
  for load,z in r['loads'].items():
   rows=[x for x in receipt['result']['records'] if x.get('event')=='benchmark' and x['case']['load']==load]
   assert math.isclose(z['judge_us'],sum(x['case']['weight']*x['median_ns']/1000 for x in rows),abs_tol=1e-7)
   assert abs(sum(x['us'] for x in z['rows'].values())-z['profile_us'])<=sum(x['case']['weight'] for x in rows)*.7
   assert math.isclose(sum(x['us'] for x in z['components'].values()),sum(x['us'] for x in z['rows'].values()),abs_tol=1e-6)
   checks+=3
  gm=math.exp(sum(math.log(x['paired_baseline_us']/x['judge_us']) for x in r['loads'].values())/len(r['loads']))
  assert abs(gm-r['speedup'])<.001,(gm,r['speedup']);checks+=1
for prob in p.PROBLEMS:
 data=json.loads(subprocess.check_output([sys.executable,'-I','-S','-B',str(ROOT/'tools/profile_report.py'),prob,'--serving','--json'],cwd='/tmp'))
 assert all(not x.startswith('decode') for x in data[0]['ours']['loads']);checks+=1
for x in (ROOT/'prof/sass').glob('decode_*.ptx'):
 assert '.target sm_90a' in x.read_text()
 assert '0 bytes spill stores, 0 bytes spill loads' in x.with_suffix('.ptxas.log').read_text();checks+=1
assert '96 bytes spill stores, 136 bytes spill loads' in (ROOT/'prof/sass/moe.compile.log').read_text()
assert json.load(open(ROOT/'prof/upstream_sync/diff.json'))==[];checks+=2
for f in ('tools/profile_report.py','prof/analyze.py','prof/compile_static.py','prof/write_report.py'):
 ast.parse((ROOT/f).read_text());checks+=1
s=f'{checks} independent accounting, geomean, serving-filter, compiler-target/spill, syntax and upstream checks passed.\n'
(ROOT/'prof/validation.txt').write_text(s);print(s)
