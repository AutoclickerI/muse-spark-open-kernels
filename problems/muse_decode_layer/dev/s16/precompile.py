"""CPU-only native compilation for all arm sources, alongside serial GPU screens."""
import json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
import local_eval,patches,static_cache
static_cache.install(local_eval)
rows={}
for name in patches.registry():
    source,meta=patches.compose([name]);path=H/'generated/native_sources'/f'{name}.py'
    path.parent.mkdir(exist_ok=True,parents=True);path.write_text(source)
    results=local_eval.hopper_compile(path)
    rows[name]=dict(sha256=meta['sha256'],checks=results,passed=all(r['ok'] for r in results))
    (H/'native_compile.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('PASS' if rows[name]['passed'] else 'FAIL',name,len(results),flush=True)
    assert rows[name]['passed'],rows[name]
