"""Keep ptxas resource evidence without generated cubins or verbose logs."""
import json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
out=[]
for variant in ('native_original','native'):
 manifest=json.loads((H/(variant+'_compile.json' if variant=='native_original' else 'native_compile.json')).read_text())
 allowed={x['name'] for x in manifest}
 if variant=='native_original' and (H/'router_compile.json').exists():allowed.update(x['name'] for x in json.loads((H/'router_compile.json').read_text()))
 for p in sorted((H/variant).glob('*.log')):
  if p.stem not in allowed:continue
  s=p.read_text()
  for match in re.finditer(r"Compiling entry function '([^']+)'.*?\n(.*?)(?=ptxas info    : Compiling entry|\Z)",s,re.S):
   sym,body=match.groups();used=re.search(r'Used (\d+) registers',body);spill=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',body)
   if not used:continue
   pretty=subprocess.run(['c++filt',sym],capture_output=True,text=True,check=True).stdout.strip()
   out.append(dict(variant=variant,unit=p.stem,symbol=pretty,registers=int(used[1]),stack_bytes=int(spill[1]) if spill else None,spill_stores=int(spill[2]) if spill else None,spill_loads=int(spill[3]) if spill else None))
(H/'resources.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out:
 if r['variant']=='native':print(r['unit'],r['registers'],r['spill_stores'],r['spill_loads'],r['symbol'].split('(')[0][:190])
